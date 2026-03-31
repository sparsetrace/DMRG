"""
DMRG.py
=======

DMRG-style two-site sweeping for Hugging Face Vision Transformers (ViT).

What this does
--------------
This module progressively rewrites a pretrained ViT by replacing encoder
blocks with a new block type and then locally re-optimizing the model with
two-site sweeps, inspired by Density Matrix Renormalization Group (DMRG).

The workflow is:

1. Start from a pretrained ViT image-classification model.
2. Open a two-block window, beginning at the top of the encoder:
      (L-2, L-1), (L-3, L-2), ..., (0, 1)
3. When a layer is first visited, replace its original transformer block
   with a new student block.
4. Freeze the rest of the model and train only the active two-block window
   (plus final layernorm / classifier, and optional boundary norms).
5. Sweep back upward:
      (0, 1), (1, 2), ..., (L-2, L-1)
6. Repeat for additional sweeps if requested.

Why this is useful
------------------
This is intended for block-wise model rewriting / compression. For example,
you can start from a pretrained ViT and gradually replace its standard blocks
with cheaper or more structured student blocks while preserving performance
through task loss and optional teacher distillation.

Current scope
-------------
- ViT only (Hugging Face ViTForImageClassification-style models)
- Two-site down-up sweeps
- Lazy block replacement during the first down pass
- Optional teacher distillation
- Local checkpointing after each window
- Optional upload to the Hugging Face Hub

Important assumption
--------------------
Replacement blocks must preserve the *external* hidden-state interface of the
ViT residual stream. Internal bottlenecks are fine, but the block must still
accept and return hidden states compatible with the surrounding encoder.

Mental model
------------
- model: the network being rewritten in-place
- teacher: an optional frozen reference model for distillation
- new_block: the student block installed layer by layer
- first down pass: "studentizes" the encoder
- later passes: refine the studentized model
"""

from __future__ import annotations

import copy
import inspect
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Batch = Union[Dict[str, Any], Tuple[Any, ...], List[Any]]
LoaderLike = Iterable[Batch]
LoaderIter = Iterator[Batch]


class ViTBlockAdapter(nn.Module):
    """
    Wrap a custom block so it behaves like a Hugging Face ViT encoder layer.

    Expected block behavior:
      - block(hidden_states) -> Tensor
      - or block(hidden_states=hidden_states, ...)
      - or block(...) -> tuple/list where the first item is the updated hidden states

    The adapter always returns a tuple because HF ViT encoder layers do.
    """

    def __init__(self, block: nn.Module, layer_index: int):
        super().__init__()
        self.block = block
        self._dmrg_replaced = True
        self._dmrg_layer_index = int(layer_index)

    def forward(
        self,
        hidden_states: Tensor,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ):
        out = None

        call_attempts = [
            lambda: self.block(hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions),
            lambda: self.block(hidden_states, head_mask=head_mask, output_attentions=output_attentions),
            lambda: self.block(hidden_states=hidden_states),
            lambda: self.block(hidden_states),
        ]

        last_err = None
        for fn in call_attempts:
            try:
                out = fn()
                break
            except TypeError as exc:
                last_err = exc

        if out is None:
            raise TypeError(
                "Could not call the custom block from the ViT adapter. "
                "Expected a block that accepts hidden states either positionally or by keyword."
            ) from last_err

        if isinstance(out, torch.Tensor):
            return (out, None) if output_attentions else (out,)

        if isinstance(out, (tuple, list)):
            return tuple(out)

        if isinstance(out, dict):
            hidden = out.get("hidden_states")
            attn = out.get("attentions")
            if hidden is None:
                raise ValueError("Custom block returned a dict without a 'hidden_states' entry.")
            return (hidden, attn) if output_attentions else (hidden,)

        raise TypeError(
            f"Unsupported block output type from adapter: {type(out)!r}. "
            "Return a Tensor, tuple/list, or dict with 'hidden_states'."
        )


class DMRG:
    """
    DMRG-style two-site sweeps for Hugging Face ViT image-classification models.

    Public API:
      - initialize with a ViT model (or HF model name) and a train loader
      - optionally provide a frozen teacher for distillation
      - call .run(...) with a replacement block class / factory / template module

    Design choices for v1:
      - ViT only
      - always sweep down then up
      - distillation happens automatically when a teacher exists
      - local checkpoints are written after each window by default
      - if repo_id is given, checkpoints are also uploaded to the Hub
      - if hub_path is given, checkpoints are uploaded inside that repo subfolder

    Notes on custom blocks:
      - the external hidden size must remain compatible with the ViT residual stream
      - the block may compress internally (e.g. bottleneck / low-rank / smaller inner dim)
      - one fresh block instance is created per layer when that layer is first opened
    """

    def __init__(
        self,
        model: Union[str, nn.Module],
        train_loader: Optional[LoaderLike] = None,
        *,
        teacher: Union[None, str, nn.Module] = None,
        device: str = "auto",
        hf_token: Optional[str] = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.hf_token = hf_token
        self.train_loader = train_loader
        self.history: List[Dict[str, Any]] = []
        self._created_hub_repos: set[str] = set()

        self.model = self._load_model(model)
        self.teacher = self._load_teacher(teacher, model)

        self._validate_vit_model(self.model)
        if self.teacher is not None:
            self._validate_vit_model(self.teacher)

        self.model.to(self.device)
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        new_block: Union[nn.Module, type, Callable[..., nn.Module]],
        *,
        sweeps: int = 1,
        steps_per_window: int = 500,
        repo_id: Optional[str] = None,
        hub_path: Optional[str] = None,
        repo_subfolder: Optional[str] = None,
        output_dir: Union[str, Path] = "./dmrg_checkpoints",
        block_kwargs: Optional[Dict[str, Any]] = None,
        lr_down: float = 1e-4,
        lr_up: float = 5e-5,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        alpha_hard: float = 0.2,
        temperature: float = 2.0,
        train_boundary_ln: bool = True,
        use_bf16: bool = True,
        grad_clip_norm: Optional[float] = None,
        save_every_window: bool = True,
        save_final: bool = True,
        private_repo: bool = True,
        verbose: int = 1,
        max_windows: Optional[int] = None,
    ) -> nn.Module:
        """
        Run down-up DMRG sweeps.

        Parameters
        ----------
        new_block:
            One of:
              - a block class (e.g. MyStudentBlock)
              - a callable returning a block instance
              - a template nn.Module (deep-copied per replaced layer)

            The block should preserve the external hidden-state size of the ViT layer.
            Internal bottlenecks are fine.

        sweeps:
            Number of full down-up sweeps.

        steps_per_window:
            Gradient updates for each local window.

        repo_id:
            If provided, each saved checkpoint is uploaded to this Hugging Face Hub repo.

        hub_path:
            Optional folder path inside the Hub repo. For example, hub_path="exp1/checkpoints"
            will upload a checkpoint named "sweep_00_down_10_11" to:
                exp1/checkpoints/sweep_00_down_10_11

        repo_subfolder:
            Backward-compatible alias for hub_path.

        output_dir:
            Local directory where checkpoints and metadata are saved.

        max_windows:
            Optional debugging budget. For example, max_windows=1 runs only the
            first window (L-2, L-1), which is useful for smoke tests.
        """
        if hub_path is not None and repo_subfolder is not None and hub_path != repo_subfolder:
            raise ValueError("Provide only one of hub_path or repo_subfolder, or make them identical.")
        if hub_path is None:
            hub_path = repo_subfolder

        loader = self._require_loader()
        loader_iter = self._infinite_loader(loader)
        output_dir = Path(output_dir)
        block_kwargs = block_kwargs or {}
        block_factory = self._make_block_factory(new_block, block_kwargs=block_kwargs)

        layers = self._get_layers(self.model)
        num_layers = len(layers)
        if num_layers < 2:
            raise ValueError("Need at least 2 ViT encoder layers for a two-site DMRG sweep.")

        self.history = []
        windows_seen = 0

        for sweep_idx in range(sweeps):
            if verbose:
                print(f"\n=== SWEEP {sweep_idx + 1}/{sweeps} (DOWN) ===")

            for i in range(num_layers - 2, -1, -1):
                window = (i, i + 1)
                replaced_now = self._ensure_replaced_many(window, block_factory)
                self._open_two_site_window(window, train_boundary_ln=train_boundary_ln)

                if verbose:
                    msg = f"-- window {window} (down)"
                    if replaced_now:
                        msg += f" | replaced {replaced_now}"
                    print(msg)

                metrics = self._train_window_steps(
                    loader_iter,
                    k_updates=steps_per_window,
                    lr=lr_down,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    use_bf16=use_bf16,
                    alpha_hard=alpha_hard,
                    temperature=temperature,
                    grad_clip_norm=grad_clip_norm,
                    verbose=verbose,
                )

                record = self._record_window(
                    sweep_idx=sweep_idx,
                    direction="down",
                    window=window,
                    replaced_now=replaced_now,
                    metrics=metrics,
                )

                if save_every_window:
                    self._save_checkpoint(
                        output_dir=output_dir,
                        record=record,
                        repo_id=repo_id,
                        hub_path=hub_path,
                        private_repo=private_repo,
                    )

                windows_seen += 1
                if max_windows is not None and windows_seen >= max_windows:
                    return self.model

            if verbose:
                print(f"\n=== SWEEP {sweep_idx + 1}/{sweeps} (UP) ===")

            for i in range(0, num_layers - 1):
                window = (i, i + 1)
                replaced_now = self._ensure_replaced_many(window, block_factory)
                self._open_two_site_window(window, train_boundary_ln=train_boundary_ln)

                if verbose:
                    msg = f"-- window {window} (up)"
                    if replaced_now:
                        msg += f" | replaced {replaced_now}"
                    print(msg)

                metrics = self._train_window_steps(
                    loader_iter,
                    k_updates=steps_per_window,
                    lr=lr_up,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    use_bf16=use_bf16,
                    alpha_hard=alpha_hard,
                    temperature=temperature,
                    grad_clip_norm=grad_clip_norm,
                    verbose=verbose,
                )

                record = self._record_window(
                    sweep_idx=sweep_idx,
                    direction="up",
                    window=window,
                    replaced_now=replaced_now,
                    metrics=metrics,
                )

                if save_every_window:
                    self._save_checkpoint(
                        output_dir=output_dir,
                        record=record,
                        repo_id=repo_id,
                        hub_path=hub_path,
                        private_repo=private_repo,
                    )

                windows_seen += 1
                if max_windows is not None and windows_seen >= max_windows:
                    return self.model

        if save_final:
            final_record = {
                "kind": "final",
                "num_layers": len(self._get_layers(self.model)),
                "studentized_layers": self.count_replaced_layers(),
                "windows_completed": len(self.history),
            }
            self._save_checkpoint(
                output_dir=output_dir,
                record=final_record,
                repo_id=repo_id,
                hub_path=hub_path,
                private_repo=private_repo,
                tag="final",
            )

        return self.model

    def run_first_window(self, new_block: Union[nn.Module, type, Callable[..., nn.Module]], **kwargs) -> nn.Module:
        """Convenience helper for a smoke test of the very first window: (L-2, L-1)."""
        kwargs.setdefault("max_windows", 1)
        return self.run(new_block, **kwargs)

    def count_replaced_layers(self) -> int:
        return sum(1 for layer in self._get_layers(self.model) if self._is_replaced_layer(layer))

    # ------------------------------------------------------------------
    # Model loading / validation
    # ------------------------------------------------------------------

    def _load_model(self, model: Union[str, nn.Module]) -> nn.Module:
        if isinstance(model, str):
            try:
                from transformers import AutoModelForImageClassification
            except ImportError as exc:
                raise ImportError(
                    "Loading a model by name requires transformers to be installed."
                ) from exc
            loaded = AutoModelForImageClassification.from_pretrained(model, token=self.hf_token)
            return loaded

        if isinstance(model, nn.Module):
            return model

        raise TypeError("model must be a torch.nn.Module or a Hugging Face model name string.")

    def _load_teacher(
        self,
        teacher: Union[None, str, nn.Module],
        original_model_spec: Union[str, nn.Module],
    ) -> Optional[nn.Module]:
        if teacher is None:
            return None

        if teacher == "self":
            if isinstance(original_model_spec, str):
                return self._load_model(original_model_spec)
            return copy.deepcopy(self.model)

        return self._load_model(teacher)

    @staticmethod
    def _validate_vit_model(model: nn.Module) -> None:
        if not hasattr(model, "vit"):
            raise ValueError("Expected a Hugging Face ViT-style model with a `.vit` attribute.")
        if not hasattr(model.vit, "encoder") or not hasattr(model.vit.encoder, "layer"):
            raise ValueError("Expected model.vit.encoder.layer to exist.")
        if not hasattr(model, "classifier"):
            raise ValueError("Expected a ViTForImageClassification-like model with a `.classifier` head.")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _require_loader(self) -> LoaderLike:
        if self.train_loader is None:
            raise ValueError("No train loader was provided. Pass train_loader to __init__.")
        return self.train_loader

    @staticmethod
    def _get_layers(model: nn.Module) -> List[nn.Module]:
        return list(model.vit.encoder.layer)

    @staticmethod
    def _count_trainable_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def _is_replaced_layer(layer: nn.Module) -> bool:
        return bool(getattr(layer, "_dmrg_replaced", False))

    @staticmethod
    def _is_norm_param(name: str) -> bool:
        n = name.lower()
        return (
            "layernorm" in n
            or ".ln" in n
            or n.endswith("ln.weight")
            or n.endswith("ln.bias")
            or ".norm" in n
        )

    @staticmethod
    def _infinite_loader(loader: LoaderLike) -> LoaderIter:
        while True:
            for batch in loader:
                yield batch

    @staticmethod
    def _distill_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
        t = float(temperature)
        return F.kl_div(
            F.log_softmax(student_logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        ) * (t * t)

    @staticmethod
    def _unpack_batch(batch: Batch) -> Tuple[Tensor, Optional[Tensor]]:
          if isinstance(batch, Mapping):
              if "pixel_values" not in batch:
                  raise KeyError("Expected batch mapping to contain 'pixel_values'.")
              labels = batch.get("labels", batch.get("label"))
              return batch["pixel_values"], labels
      
          if isinstance(batch, (tuple, list)):
              if len(batch) == 0:
                  raise ValueError("Received an empty batch.")
              if len(batch) == 1:
                  return batch[0], None
              return batch[0], batch[1]
      
          raise TypeError(
              f"Unsupported batch type: {type(batch)!r}. "
              "Expected a mapping with pixel_values/labels or a tuple/list like (pixel_values, labels)."
          )

    def _record_window(
        self,
        *,
        sweep_idx: int,
        direction: str,
        window: Tuple[int, int],
        replaced_now: Sequence[int],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = {
            "kind": "window",
            "sweep_idx": int(sweep_idx),
            "direction": str(direction),
            "window": [int(window[0]), int(window[1])],
            "replaced_now": [int(x) for x in replaced_now],
            "studentized_layers": self.count_replaced_layers(),
            "num_layers": len(self._get_layers(self.model)),
            "metrics": metrics,
        }
        self.history.append(record)
        return record

    @staticmethod
    def _normalize_hub_path(hub_path: Optional[str], tag: str) -> str:
        if hub_path is None or hub_path.strip() == "":
            return tag
        prefix = hub_path.strip("/")
        return f"{prefix}/{tag}"

    # ------------------------------------------------------------------
    # Block creation / surgery
    # ------------------------------------------------------------------

    def _make_block_factory(
        self,
        new_block: Union[nn.Module, type, Callable[..., nn.Module]],
        *,
        block_kwargs: Dict[str, Any],
    ) -> Callable[[nn.Module, int], nn.Module]:
        template_module = new_block if isinstance(new_block, nn.Module) else None

        if template_module is not None:

            def factory(model: nn.Module, layer_index: int) -> nn.Module:
                block = copy.deepcopy(template_module)
                self._maybe_configure_block(block, model=model, layer_index=layer_index, block_kwargs=block_kwargs)
                return block

            return factory

        if inspect.isclass(new_block) and issubclass(new_block, nn.Module):
            signature_target = new_block.__init__

            def factory(model: nn.Module, layer_index: int) -> nn.Module:
                kwargs = self._build_block_kwargs(signature_target, model, layer_index, block_kwargs)
                return new_block(**kwargs)

            return factory

        if callable(new_block):
            signature_target = new_block

            def factory(model: nn.Module, layer_index: int) -> nn.Module:
                kwargs = self._build_block_kwargs(signature_target, model, layer_index, block_kwargs)
                block = new_block(**kwargs)
                if not isinstance(block, nn.Module):
                    raise TypeError("Custom block factory must return a torch.nn.Module.")
                return block

            return factory

        raise TypeError(
            "new_block must be an nn.Module template, an nn.Module class, or a callable returning an nn.Module."
        )

    def _build_block_kwargs(
        self,
        signature_target: Callable[..., Any],
        model: nn.Module,
        layer_index: int,
        block_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        sig = inspect.signature(signature_target)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        cfg = getattr(model, "config", None)
        candidate_kwargs: Dict[str, Any] = {
            "model": model,
            "layer_index": layer_index,
            "config": cfg,
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "num_labels": getattr(cfg, "num_labels", None),
            "image_size": getattr(cfg, "image_size", None),
            "patch_size": getattr(cfg, "patch_size", None),
        }

        candidate_kwargs.update(block_kwargs)
        candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None}

        if accepts_var_kwargs:
            return candidate_kwargs

        allowed = set(sig.parameters.keys())
        return {k: v for k, v in candidate_kwargs.items() if k in allowed}

    @staticmethod
    def _maybe_configure_block(
        block: nn.Module,
        *,
        model: nn.Module,
        layer_index: int,
        block_kwargs: Dict[str, Any],
    ) -> None:
        if hasattr(block, "configure_for_dmrg") and callable(block.configure_for_dmrg):
            block.configure_for_dmrg(model=model, layer_index=layer_index, **block_kwargs)

    def _ensure_replaced_many(
        self,
        indices: Sequence[int],
        block_factory: Callable[[nn.Module, int], nn.Module],
    ) -> List[int]:
        replaced_now: List[int] = []
        for idx in indices:
            if self._ensure_replaced(idx, block_factory):
                replaced_now.append(int(idx))
        return replaced_now

    def _ensure_replaced(
        self,
        idx: int,
        block_factory: Callable[[nn.Module, int], nn.Module],
    ) -> bool:
        layers = self._get_layers(self.model)
        if self._is_replaced_layer(layers[idx]):
            return False

        block = block_factory(self.model, idx)
        adapted = ViTBlockAdapter(block, layer_index=idx)
        self.model.vit.encoder.layer[idx] = adapted
        return True

    def _open_two_site_window(
        self,
        window: Tuple[int, int],
        *,
        train_boundary_ln: bool,
    ) -> None:
        i, j = window
        if j != i + 1:
            raise ValueError("Window must be adjacent: (i, i+1).")

        layers = self._get_layers(self.model)
        num_layers = len(layers)

        for p in self.model.parameters():
            p.requires_grad = False

        for k in (i, j):
            for p in layers[k].parameters():
                p.requires_grad = True

        if train_boundary_ln:
            neighbors = set()
            if i - 1 >= 0:
                neighbors.add(i - 1)
            if j + 1 < num_layers:
                neighbors.add(j + 1)
            for nb in neighbors:
                for name, p in layers[nb].named_parameters():
                    if self._is_norm_param(name):
                        p.requires_grad = True

        if hasattr(self.model.vit, "layernorm"):
            for p in self.model.vit.layernorm.parameters():
                p.requires_grad = True

        if hasattr(self.model, "classifier"):
            for p in self.model.classifier.parameters():
                p.requires_grad = True

        self.model.eval()
        layers[i].train()
        layers[j].train()
        if hasattr(self.model.vit, "layernorm"):
            self.model.vit.layernorm.train()
        if hasattr(self.model, "classifier"):
            self.model.classifier.train()

    # ------------------------------------------------------------------
    # Training / checkpointing
    # ------------------------------------------------------------------

    def _train_window_steps(
        self,
        loader_iter: LoaderIter,
        *,
        k_updates: int,
        lr: float,
        weight_decay: float,
        warmup_ratio: float,
        use_bf16: bool,
        alpha_hard: float,
        temperature: float,
        grad_clip_norm: Optional[float],
        verbose: int,
    ) -> Dict[str, Any]:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters are open in the current window.")

        try:
            optimizer = torch.optim.AdamW(
                trainable,
                lr=lr,
                weight_decay=weight_decay,
                fused=(self.device.type == "cuda"),
            )
        except TypeError:
            optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        warmup_steps = int(warmup_ratio * k_updates)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, k_updates - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        amp_enabled = bool(self.device.type == "cuda" and use_bf16)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
            if amp_enabled
            else nullcontext()
        )

        teacher_active = self.teacher is not None
        last_loss = None
        loss_ema = None
        last_hard = None
        last_kl = None

        for step in range(1, k_updates + 1):
            batch = next(loader_iter)
            pixel_values, labels = self._unpack_batch(batch)
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            if labels is not None:
                labels = labels.to(self.device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = None
                if teacher_active:
                    teacher_out = self.teacher(pixel_values=pixel_values)
                    teacher_logits = teacher_out.logits

            with autocast_ctx:
                if labels is not None:
                    out = self.model(pixel_values=pixel_values, labels=labels)
                    student_logits = out.logits
                    hard_loss = out.loss
                else:
                    out = self.model(pixel_values=pixel_values)
                    student_logits = out.logits
                    hard_loss = None

                if teacher_logits is not None:
                    kl = self._distill_loss(student_logits, teacher_logits, temperature=temperature)
                    if hard_loss is not None:
                        loss = alpha_hard * hard_loss + (1.0 - alpha_hard) * kl
                    else:
                        loss = kl
                else:
                    if hard_loss is None:
                        raise ValueError(
                            "No labels were provided in the batch and no teacher exists for pure distillation."
                        )
                    kl = None
                    loss = hard_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip_norm))

            optimizer.step()
            scheduler.step()

            last_loss = float(loss.detach().item())
            loss_ema = last_loss if loss_ema is None else (0.95 * loss_ema + 0.05 * last_loss)
            last_hard = None if hard_loss is None else float(hard_loss.detach().item())
            last_kl = None if kl is None else float(kl.detach().item())

            if verbose >= 2 and (step == 1 or step == k_updates or step % max(1, k_updates // 5) == 0):
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"   step {step:5d}/{k_updates} | loss={last_loss:.4f} | "
                    f"loss_ema={loss_ema:.4f} | lr={lr_now:.2e}"
                )

        return {
            "k_updates": int(k_updates),
            "lr": float(lr),
            "last_loss": last_loss,
            "loss_ema": loss_ema,
            "hard_loss": last_hard,
            "kl_loss": last_kl,
            "trainable_params": int(self._count_trainable_params(self.model)),
            "teacher_active": bool(teacher_active),
        }

    def _save_checkpoint(
        self,
        *,
        output_dir: Path,
        record: Dict[str, Any],
        repo_id: Optional[str],
        hub_path: Optional[str],
        private_repo: bool,
        tag: Optional[str] = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if tag is None:
            if record.get("kind") == "window":
                sweep_idx = record["sweep_idx"]
                direction = record["direction"]
                i, j = record["window"]
                tag = f"sweep_{sweep_idx:02d}_{direction}_{i:02d}_{j:02d}"
            else:
                tag = "checkpoint"

        ckpt_dir = output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            save_kwargs = {}
            if self.hf_token is not None:
                save_kwargs["token"] = self.hf_token
            try:
                self.model.save_pretrained(ckpt_dir, **save_kwargs)
            except TypeError:
                self.model.save_pretrained(ckpt_dir)
        else:
            torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")

        with open(ckpt_dir / "dmrg_meta.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if repo_id is not None:
            try:
                from huggingface_hub import create_repo, upload_folder
            except ImportError as exc:
                raise ImportError(
                    "Uploading checkpoints to the Hugging Face Hub requires `huggingface_hub` to be installed."
                ) from exc

            if repo_id not in self._created_hub_repos:
                create_repo(
                    repo_id=repo_id,
                    token=self.hf_token,
                    private=private_repo,
                    exist_ok=True,
                )
                self._created_hub_repos.add(repo_id)

            path_in_repo = self._normalize_hub_path(hub_path, tag)

            upload_folder(
                repo_id=repo_id,
                folder_path=str(ckpt_dir),
                path_in_repo=path_in_repo,
                token=self.hf_token,
                commit_message=f"DMRG {tag}",
            )
