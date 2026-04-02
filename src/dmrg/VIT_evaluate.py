from __future__ import annotations

"""
Evaluate either:
1) a standard Hugging Face image-classification model, or
2) a DMRG checkpoint folder/repo produced by the newer DMRG.py style.

What this adds relative to the earlier evaluator:
- auto-detects `dmrg_meta.json`
- reconstructs replaced ViT encoder layers before loading checkpoint weights
- still works for normal HF repos with plain `AutoModelForImageClassification`
- can load from a local path or a Hub repo/subfolder

Important limitation
--------------------
The current DMRG checkpoint metadata stores one replacement block class/module
for all replaced layers. This loader therefore assumes all replaced layers use
that same block class. If you later move to heterogeneous per-layer block types,
you should also save a per-layer architecture manifest and update this loader.
"""

import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {"params_total": total}


def _install_module_aliases(module_aliases: Optional[Mapping[str, str]] = None) -> None:
    """
    Make local modules available under alternate import names.

    Example:
        {"mhdm": "dmrg.mhdm"}

    means:
        import dmrg.mhdm as real_module
        sys.modules["mhdm"] = real_module
    """
    if not module_aliases:
        return

    for wanted_name, real_name in module_aliases.items():
        if wanted_name in sys.modules:
            continue
        module = importlib.import_module(real_name)
        sys.modules[wanted_name] = module


class ViTBlockAdapter(nn.Module):
    """
    Wrap a custom block so it behaves like the current HF ViT encoder layer.

    Expected HF ViTLayer behavior here:
      input: hidden_states
      output: hidden_states tensor
    """

    def __init__(self, block: nn.Module, layer_index: int):
        super().__init__()
        self.block = block
        self._dmrg_replaced = True
        self._dmrg_layer_index = int(layer_index)

    @staticmethod
    def _extract_tensor(x: Any, name: str = "value") -> torch.Tensor:
        while isinstance(x, (tuple, list)):
            if len(x) == 0:
                raise ValueError(f"{name} was an empty tuple/list.")
            x = x[0]

        if isinstance(x, dict):
            if "hidden_states" in x:
                x = x["hidden_states"]
            elif "last_hidden_state" in x:
                x = x["last_hidden_state"]
            else:
                raise KeyError(
                    f"{name} was a dict but had neither 'hidden_states' nor 'last_hidden_state'."
                )
            while isinstance(x, (tuple, list)):
                if len(x) == 0:
                    raise ValueError(f"{name} dict entry was an empty tuple/list.")
                x = x[0]

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must resolve to a Tensor, got {type(x)!r}")

        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self._extract_tensor(hidden_states, name="hidden_states")

        out = None
        last_err = None
        call_attempts = [
            lambda: self.block(
                hidden_states=hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            ),
            lambda: self.block(
                hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            ),
            lambda: self.block(hidden_states=hidden_states, **kwargs),
            lambda: self.block(hidden_states, **kwargs),
        ]

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

        return self._extract_tensor(out, name="block output")


# -----------------------------------------------------------------------------
# Path / import helpers
# -----------------------------------------------------------------------------


def _resolve_repo_or_local_dir(model_id_or_path: str, subfolder: Optional[str], token: Optional[str]) -> Path:
    """
    Return a local directory containing the requested model/checkpoint.

    - If model_id_or_path already exists locally, use it.
    - Otherwise snapshot-download the Hub repo, then optionally descend into subfolder.
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        root = candidate.resolve()
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "Loading a Hub DMRG checkpoint requires `huggingface_hub` to be installed."
            ) from exc
        root = Path(snapshot_download(repo_id=model_id_or_path, token=token)).resolve()

    ckpt_dir = root / subfolder if subfolder else root
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Requested checkpoint directory does not exist: {ckpt_dir}")
    return ckpt_dir


def _import_module_from_checkpoint(
    ckpt_dir: Path,
    *,
    module_name: Optional[str] = None,
    source_file: Optional[str] = None,
):
    """
    Import a module from the checkpoint directory.

    Preference order:
    1) explicit source file inside ckpt_dir
    2) import by module name after adding ckpt_dir to sys.path
    """
    if str(ckpt_dir) not in sys.path:
        sys.path.insert(0, str(ckpt_dir))

    if source_file:
        path = ckpt_dir / source_file
        if path.exists():
            mod_name = Path(source_file).stem
            return importlib.import_module(mod_name)

    if not module_name:
        raise ValueError("Need either source_file or module_name to import the custom block module.")

    try:
        return importlib.import_module(module_name)
    except Exception:
        if "." in module_name:
            return importlib.import_module(module_name.split(".")[-1])
        raise


# -----------------------------------------------------------------------------
# DMRG reconstruction helpers
# -----------------------------------------------------------------------------


def _build_block_kwargs(
    signature_target: Any,
    model: nn.Module,
    layer_index: int,
    user_block_kwargs: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    sig = inspect.signature(signature_target)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    cfg = getattr(model, "config", None)
    candidate_kwargs: dict[str, Any] = {
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

    if user_block_kwargs:
        candidate_kwargs.update(dict(user_block_kwargs))

    candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if v is not None}

    if accepts_var_kwargs:
        return candidate_kwargs

    allowed = set(sig.parameters.keys())
    return {k: v for k, v in candidate_kwargs.items() if k in allowed}


def _instantiate_replacement_block(
    block_ctor: Any,
    model: nn.Module,
    layer_index: int,
    user_block_kwargs: Optional[Mapping[str, Any]],
) -> nn.Module:
    signature_target = block_ctor.__init__ if inspect.isclass(block_ctor) else block_ctor
    kwargs = _build_block_kwargs(signature_target, model, layer_index, user_block_kwargs)
    block = block_ctor(**kwargs)
    if not isinstance(block, nn.Module):
        raise TypeError("Custom block constructor must return a torch.nn.Module.")
    return block


def _replace_dmrg_layers(
    model: nn.Module,
    meta: Mapping[str, Any],
    ckpt_dir: Path,
    *,
    dmrg_block_kwargs: Optional[Mapping[str, Any]] = None,
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> None:
    replaced = [int(x) for x in meta.get("replaced_layer_indices", [])]
    if not replaced:
        return

    block_class_name = meta.get("block_class_name")
    block_module_name = meta.get("block_module_name")
    block_source_file = meta.get("block_source_file")

    if not block_class_name:
        raise ValueError(
            "dmrg_meta.json does not contain block_class_name; cannot reconstruct replaced layers."
        )

    mod = _import_module_from_checkpoint(
        ckpt_dir,
        module_name=block_module_name,
        source_file=block_source_file,
    )
    block_ctor = getattr(mod, block_class_name)

    for idx in replaced:
        per_layer_kwargs = {}
        if dmrg_block_kwargs:
            per_layer_kwargs.update(dict(dmrg_block_kwargs))
        if dmrg_block_kwargs_by_layer and idx in dmrg_block_kwargs_by_layer:
            per_layer_kwargs.update(dict(dmrg_block_kwargs_by_layer[idx]))

        block = _instantiate_replacement_block(block_ctor, model, idx, per_layer_kwargs)

        ref_layer = model.vit.encoder.layer[idx]
        ref_param = next(ref_layer.parameters(), None)
        if ref_param is not None:
            block = block.to(device=ref_param.device, dtype=ref_param.dtype)
        adapted = ViTBlockAdapter(block, layer_index=idx)
        if ref_param is not None:
            adapted = adapted.to(device=ref_param.device, dtype=ref_param.dtype)

        model.vit.encoder.layer[idx] = adapted


def _load_checkpoint_weights(model: nn.Module, ckpt_dir: Path) -> None:
    safetensors_file = ckpt_dir / "model.safetensors"
    safetensors_index = ckpt_dir / "model.safetensors.index.json"
    pytorch_bin = ckpt_dir / "pytorch_model.bin"
    pytorch_index = ckpt_dir / "pytorch_model.bin.index.json"

    if safetensors_file.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading `model.safetensors` requires the `safetensors` package."
            ) from exc
        state = load_file(str(safetensors_file))
        missing, unexpected = model.load_state_dict(state, strict=False)
    elif pytorch_bin.exists():
        state = torch.load(str(pytorch_bin), map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
    elif safetensors_index.exists() or pytorch_index.exists():
        try:
            from transformers.modeling_utils import load_sharded_checkpoint
        except ImportError as exc:
            raise ImportError(
                "Loading sharded checkpoints requires transformers.modeling_utils.load_sharded_checkpoint."
            ) from exc
        missing, unexpected = load_sharded_checkpoint(model, str(ckpt_dir), strict=False)
    else:
        raise FileNotFoundError(
            f"No supported weight file found in {ckpt_dir} "
            "(expected model.safetensors, pytorch_model.bin, or sharded index files)."
        )

    if unexpected:
        print(f"[load warning] unexpected keys when loading checkpoint: {len(unexpected)}")
    if missing:
        print(f"[load warning] missing keys when loading checkpoint: {len(missing)}")


def _load_dmrg_model_and_processor(
    model_id_or_path: str,
    *,
    subfolder: Optional[str],
    trust_remote_code: bool,
    token: Optional[str],
    module_aliases: Optional[Mapping[str, str]],
    dmrg_block_kwargs: Optional[Mapping[str, Any]],
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]],
    device: torch.device,
):
    ckpt_dir = _resolve_repo_or_local_dir(model_id_or_path, subfolder=subfolder, token=token)

    with open(ckpt_dir / "dmrg_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    if str(ckpt_dir) not in sys.path:
        sys.path.insert(0, str(ckpt_dir))
    _install_module_aliases(module_aliases)

    base_model_repo = meta.get("base_model_repo")
    if not base_model_repo:
        raise ValueError(
            "dmrg_meta.json does not contain `base_model_repo`. "
            "Current evaluator expects DMRG checkpoints saved from a named base model."
        )

    model = AutoModelForImageClassification.from_pretrained(
        base_model_repo,
        subfolder=meta.get("base_model_subfolder"),
        trust_remote_code=trust_remote_code,
        token=token,
    )

    _replace_dmrg_layers(
        model,
        meta,
        ckpt_dir,
        dmrg_block_kwargs=dmrg_block_kwargs,
        dmrg_block_kwargs_by_layer=dmrg_block_kwargs_by_layer,
    )
    _load_checkpoint_weights(model, ckpt_dir)
    model.to(device)
    model.eval()

    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(
            str(ckpt_dir),
            trust_remote_code=trust_remote_code,
            token=token,
        )
    except Exception:
        processor = AutoImageProcessor.from_pretrained(
            base_model_repo,
            subfolder=meta.get("processor_subfolder"),
            trust_remote_code=trust_remote_code,
            token=token,
        )

    return model, processor, meta


# -----------------------------------------------------------------------------
# Unified model loader
# -----------------------------------------------------------------------------


def load_model_and_processor(
    model_id_or_path: str,
    *,
    subfolder: Optional[str] = None,
    trust_remote_code: bool = False,
    module_aliases: Optional[Mapping[str, str]] = None,
    dmrg_block_kwargs: Optional[Mapping[str, Any]] = None,
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]] = None,
    device: Optional[str] = None,
):
    """
    Load either:
      - a standard HF image-classification model, or
      - a DMRG checkpoint folder/repo that contains dmrg_meta.json.

    Returns
    -------
    model, processor, meta_or_none
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    tok = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ["HF_HUB_TOKEN"] = tok

    ckpt_dir: Optional[Path] = None
    try:
        ckpt_dir = _resolve_repo_or_local_dir(model_id_or_path, subfolder=subfolder, token=tok)
    except Exception:
        ckpt_dir = None

    is_dmrg = ckpt_dir is not None and (ckpt_dir / "dmrg_meta.json").exists()

    if is_dmrg:
        return _load_dmrg_model_and_processor(
            model_id_or_path,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            token=tok,
            module_aliases=module_aliases,
            dmrg_block_kwargs=dmrg_block_kwargs,
            dmrg_block_kwargs_by_layer=dmrg_block_kwargs_by_layer,
            device=device_t,
        )

    _install_module_aliases(module_aliases)

    processor = AutoImageProcessor.from_pretrained(
        model_id_or_path,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        token=tok,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id_or_path,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        token=tok,
    ).to(device_t)
    model.eval()
    return model, processor, None


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.inference_mode()
def VIT_metrics(
    model_id: str,
    subfolder: Optional[str] = None,
    dataset_id: str = "ILSVRC/imagenet-1k",
    split: str = "validation",
    max_images: Optional[int] = 50_000,
    batch_size: int = 128,
    trust_remote_code: bool = False,
    module_aliases: Optional[Mapping[str, str]] = None,
    dmrg_block_kwargs: Optional[Mapping[str, Any]] = None,
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]] = None,
    shuffle_stream: bool = False,
    shuffle_buffer: int = 10_000,
    seed: int = 0,
    use_amp: bool = True,
    device: Optional[str] = None,
) -> dict:
    """
    Evaluate either a standard HF image-classification repo or a DMRG checkpoint.

    Examples
    --------
    # Standard HF model
    metrics = VIT_metrics("google/vit-base-patch16-224")

    # DMRG checkpoint saved in a Hub subfolder
    metrics = VIT_metrics(
        "your-name/your-repo",
        subfolder="dmrg/final",
        trust_remote_code=False,
        dmrg_block_kwargs={"some_arg": 123},  # only if your custom block needs it
    )
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model, processor, meta = load_model_and_processor(
        model_id,
        subfolder=subfolder,
        trust_remote_code=trust_remote_code,
        module_aliases=module_aliases,
        dmrg_block_kwargs=dmrg_block_kwargs,
        dmrg_block_kwargs_by_layer=dmrg_block_kwargs_by_layer,
        device=device,
    )

    params = count_params(model)

    tok = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ["HF_HUB_TOKEN"] = tok

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True, token=True)
    except TypeError:
        ds = load_dataset(dataset_id, split=split, streaming=True, use_auth_token=True)

    if shuffle_stream:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    if max_images is not None:
        ds = ds.take(max_images)

    n = 0
    top1 = 0
    top5 = 0
    total_loss = 0.0

    images: list[Any] = []
    labels: list[int] = []

    amp_enabled = bool(use_amp and device_t.type == "cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)

    def flush_batch() -> None:
        nonlocal n, top1, top5, total_loss, images, labels
        if not images:
            return

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device_t)
        y = torch.tensor(labels, dtype=torch.long, device=device_t)

        with autocast:
            logits = model(pixel_values=pixel_values).logits

        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        top1 += (logits.argmax(dim=1) == y).sum().item()
        top5 += logits.topk(5, dim=1).indices.eq(y.view(-1, 1)).any(dim=1).sum().item()
        n += y.size(0)

        images, labels = [], []

    for ex in ds:
        img = ex["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        images.append(img)

        lab = ex.get("label", ex.get("labels"))
        labels.append(int(lab))

        if len(images) >= batch_size:
            flush_batch()

    flush_batch()

    if n == 0:
        raise RuntimeError("Evaluated 0 samples (stream empty or wrong split?).")

    out = {
        "acc1": top1 / n,
        "acc5": top5 / n,
        "xent": total_loss / n,
        "n": n,
        **params,
    }

    if meta is not None:
        out["dmrg_checkpoint"] = True
        out["studentized_layers"] = meta.get("studentized_layers")
        out["replaced_layer_indices"] = meta.get("replaced_layer_indices")
    else:
        out["dmrg_checkpoint"] = False

    return out
