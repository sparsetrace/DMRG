from __future__ import annotations

"""
Evaluate either:
1) a standard Hugging Face image-classification model, or
2) a DMRG checkpoint folder/repo produced by the newer DMRG.py style.

What this supports
------------------
- normal HF loading with AutoImageProcessor + AutoModelForImageClassification
- DMRG checkpoint detection via dmrg_meta.json
- reconstruction of replaced ViT encoder layers before loading checkpoint weights
- fallback model construction from local config.json when dmrg_meta.json lacks base_model_repo
- fallback processor loading when checkpoint lacks preprocessor_config.json
- optional local module aliases for custom imports like `mhdm`

Important limitation
--------------------
This loader assumes one replacement block class for all replaced layers unless you
provide per-layer constructor kwargs. That matches the current newer DMRG saver,
which records one block class/module/source globally rather than a full per-layer
architecture manifest.
"""

import importlib
import importlib.util
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
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def count_params(model: nn.Module) -> dict[str, int]:
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


def _hf_pretrained_kwargs(
    *,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build kwargs for HF `.from_pretrained(...)` calls, omitting None values.

    Some transformers versions error on subfolder=None, so do not pass it unless set.
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if token is not None:
        kwargs["token"] = token
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return kwargs


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


def _resolve_repo_or_local_dir(
    model_id_or_path: str,
    subfolder: Optional[str],
    token: Optional[str],
) -> Path:
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


def _find_dmrg_checkpoint_dir(
    model_id_or_path: str,
    subfolder: Optional[str],
    token: Optional[str],
) -> Optional[Path]:
    """
    Best-effort detection for a DMRG checkpoint.

    For local paths, check immediately.
    For Hub repos, try to fetch dmrg_meta.json only; if present, snapshot the repo.
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        ckpt_dir = candidate.resolve() / subfolder if subfolder else candidate.resolve()
        if (ckpt_dir / "dmrg_meta.json").exists():
            return ckpt_dir
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    try:
        download_kwargs = {
            "repo_id": model_id_or_path,
            "filename": "dmrg_meta.json",
            "token": token,
        }
        if subfolder is not None:
            download_kwargs["subfolder"] = subfolder
        hf_hub_download(**download_kwargs)
    except Exception:
        return None

    return _resolve_repo_or_local_dir(model_id_or_path, subfolder=subfolder, token=token)


def _load_module_from_file(path: Path):
    """
    Import a Python module directly from a file path with a unique temporary module name.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot import missing module file: {path}")

    unique_name = f"_dmrg_ckpt_{path.stem}_{abs(hash(str(path)))}"
    if unique_name in sys.modules:
        return sys.modules[unique_name]

    spec = importlib.util.spec_from_file_location(unique_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


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
            return _load_module_from_file(path)

    if not module_name:
        raise ValueError("Need either source_file or module_name to import the custom block module.")

    try:
        return importlib.import_module(module_name)
    except Exception:
        if "." in module_name:
            try:
                return importlib.import_module(module_name.split(".")[-1])
            except Exception:
                pass
        raise


def _module_defined_nn_module_classes(module) -> list[type[nn.Module]]:
    """
    Find nn.Module subclasses defined directly in a module.
    """
    out: list[type[nn.Module]] = []
    for name, obj in vars(module).items():
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, nn.Module):
            continue
        if obj.__module__ != module.__name__:
            continue
        if name in {"ViTBlockAdapter", "DMRG"}:
            continue
        out.append(obj)
    return out


def _choose_unique_block_class(candidates: list[type[nn.Module]], context: str) -> type[nn.Module]:
    """
    Choose a likely block class from candidates, or raise if ambiguous.
    """
    if not candidates:
        raise ValueError(f"No nn.Module subclasses found in {context}.")

    if len(candidates) == 1:
        return candidates[0]

    preferred = [
        cls for cls in candidates
        if ("Block" in cls.__name__ or "Layer" in cls.__name__)
        and not cls.__name__.endswith("Adapter")
    ]
    if len(preferred) == 1:
        return preferred[0]

    names = ", ".join(cls.__name__ for cls in candidates)
    raise ValueError(
        f"Could not uniquely infer the custom block class from {context}. "
        f"Candidates: {names}. Pass dmrg_block_class_name and/or dmrg_block_source_file explicitly."
    )


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


def _infer_block_ctor(
    ckpt_dir: Path,
    meta: Mapping[str, Any],
    *,
    dmrg_block_class_name: Optional[str] = None,
    dmrg_block_module_name: Optional[str] = None,
    dmrg_block_source_file: Optional[str] = None,
):
    """
    Infer the custom replacement block constructor from:
    1) explicit overrides
    2) dmrg_meta.json fields
    3) scanning .py files in the checkpoint directory
    """
    class_name = dmrg_block_class_name or meta.get("block_class_name")
    module_name = dmrg_block_module_name or meta.get("block_module_name")
    source_file = dmrg_block_source_file or meta.get("block_source_file")

    if source_file or module_name:
        module = _import_module_from_checkpoint(
            ckpt_dir,
            module_name=module_name,
            source_file=source_file,
        )
        if class_name:
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Module {module.__name__!r} does not define class {class_name!r}."
                )
            return getattr(module, class_name)
        return _choose_unique_block_class(
            _module_defined_nn_module_classes(module),
            context=f"module {module.__name__}",
        )

    py_files = sorted(
        p for p in ckpt_dir.glob("*.py")
        if p.name not in {"__init__.py", "VIT_evaluate.py", "DMRG.py", "dmrg.py"}
    )

    all_candidates: list[tuple[Path, type[nn.Module]]] = []
    for path in py_files:
        try:
            module = _load_module_from_file(path)
            classes = _module_defined_nn_module_classes(module)
            for cls in classes:
                all_candidates.append((path, cls))
        except Exception:
            continue

    if not all_candidates:
        raise ValueError(
            "Could not infer the custom block constructor. "
            "No usable nn.Module subclasses were found in checkpoint .py files. "
            "Pass dmrg_block_class_name and/or dmrg_block_source_file explicitly."
        )

    preferred = [
        (path, cls)
        for path, cls in all_candidates
        if ("Block" in cls.__name__ or "Layer" in cls.__name__)
    ]
    pool = preferred if preferred else all_candidates

    if len(pool) == 1:
        return pool[0][1]

    names = ", ".join(f"{cls.__name__} ({path.name})" for path, cls in pool)
    raise ValueError(
        "Could not uniquely infer the custom block constructor from checkpoint files. "
        f"Candidates: {names}. Pass dmrg_block_class_name and/or dmrg_block_source_file explicitly."
    )


def _replace_dmrg_layers(
    model: nn.Module,
    meta: Mapping[str, Any],
    ckpt_dir: Path,
    *,
    dmrg_block_kwargs: Optional[Mapping[str, Any]] = None,
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]] = None,
    dmrg_replaced_layer_indices: Optional[list[int]] = None,
    dmrg_block_class_name: Optional[str] = None,
    dmrg_block_module_name: Optional[str] = None,
    dmrg_block_source_file: Optional[str] = None,
) -> None:
    replaced = dmrg_replaced_layer_indices
    if replaced is None:
        replaced = meta.get("replaced_layer_indices", [])

    replaced = [int(x) for x in replaced]
    if not replaced:
        raise ValueError(
            "Could not determine which layers were replaced. "
            "dmrg_meta.json lacks `replaced_layer_indices`, and no override was provided."
        )

    block_ctor = _infer_block_ctor(
        ckpt_dir,
        meta,
        dmrg_block_class_name=dmrg_block_class_name,
        dmrg_block_module_name=dmrg_block_module_name,
        dmrg_block_source_file=dmrg_block_source_file,
    )

    for idx in replaced:
        per_layer_kwargs: dict[str, Any] = {}
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


def _build_base_model_for_dmrg(
    ckpt_dir: Path,
    meta: Mapping[str, Any],
    *,
    trust_remote_code: bool,
    token: Optional[str],
) -> nn.Module:
    """
    Build the base model for a DMRG checkpoint.

    Preference:
    1) if dmrg_meta.json contains base_model_repo, use that config source
    2) otherwise use ckpt_dir/config.json directly
    """
    base_model_repo = meta.get("base_model_repo")
    base_model_subfolder = meta.get("base_model_subfolder")

    if base_model_repo:
        cfg_source = base_model_repo
        cfg_subfolder = base_model_subfolder
    else:
        cfg_source = str(ckpt_dir)
        cfg_subfolder = None

    cfg = AutoConfig.from_pretrained(
        cfg_source,
        **_hf_pretrained_kwargs(
            trust_remote_code=trust_remote_code,
            token=token,
            subfolder=cfg_subfolder,
        ),
    )

    try:
        model = AutoModelForImageClassification.from_config(
            cfg,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForImageClassification.from_config(cfg)

    return model


def _load_processor_with_fallbacks(
    model_id_or_path: str,
    ckpt_dir: Path,
    meta: Mapping[str, Any],
    *,
    trust_remote_code: bool,
    token: Optional[str],
    fallback_processor_id: Optional[str] = None,
    fallback_processor_subfolder: Optional[str] = None,
    config_name_or_path: Optional[str] = None,
):
    """
    Try several places for the image processor.

    Order:
    1) checkpoint folder itself
    2) repo/path root (without subfolder)
    3) base_model_repo + processor_subfolder/base_model_subfolder from meta
    4) config._name_or_path if it looks usable
    5) explicit fallback processor args
    """
    candidates: list[tuple[str, Optional[str]]] = []
    seen: set[tuple[str, Optional[str]]] = set()

    def add_candidate(source: Optional[str], sf: Optional[str]) -> None:
        if not source:
            return
        key = (source, sf)
        if key not in seen:
            seen.add(key)
            candidates.append(key)

    add_candidate(str(ckpt_dir), None)
    add_candidate(model_id_or_path, None)

    base_model_repo = meta.get("base_model_repo")
    processor_subfolder = meta.get("processor_subfolder")
    base_model_subfolder = meta.get("base_model_subfolder")

    add_candidate(base_model_repo, processor_subfolder)
    add_candidate(base_model_repo, base_model_subfolder)
    add_candidate(base_model_repo, None)

    if config_name_or_path and config_name_or_path not in {str(ckpt_dir), model_id_or_path}:
        add_candidate(config_name_or_path, None)

    add_candidate(fallback_processor_id, fallback_processor_subfolder)
    add_candidate(fallback_processor_id, None)

    last_err = None
    for source, sf in candidates:
        try:
            return AutoImageProcessor.from_pretrained(
                source,
                **_hf_pretrained_kwargs(
                    trust_remote_code=trust_remote_code,
                    token=token,
                    subfolder=sf,
                ),
            )
        except Exception as exc:
            last_err = exc

    raise RuntimeError(
        "Could not load an image processor. "
        "The checkpoint folder does not contain preprocessor_config.json, "
        "and no fallback processor source worked."
    ) from last_err


def _load_dmrg_model_and_processor(
    model_id_or_path: str,
    *,
    subfolder: Optional[str],
    trust_remote_code: bool,
    token: Optional[str],
    module_aliases: Optional[Mapping[str, str]],
    dmrg_block_kwargs: Optional[Mapping[str, Any]],
    dmrg_block_kwargs_by_layer: Optional[Mapping[int, Mapping[str, Any]]],
    dmrg_replaced_layer_indices: Optional[list[int]],
    dmrg_block_class_name: Optional[str],
    dmrg_block_module_name: Optional[str],
    dmrg_block_source_file: Optional[str],
    fallback_processor_id: Optional[str],
    fallback_processor_subfolder: Optional[str],
    device: torch.device,
):
    ckpt_dir = _resolve_repo_or_local_dir(model_id_or_path, subfolder=subfolder, token=token)

    with open(ckpt_dir / "dmrg_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    if str(ckpt_dir) not in sys.path:
        sys.path.insert(0, str(ckpt_dir))

    _install_module_aliases(module_aliases)

    model = _build_base_model_for_dmrg(
        ckpt_dir,
        meta,
        trust_remote_code=trust_remote_code,
        token=token,
    )

    _replace_dmrg_layers(
        model,
        meta,
        ckpt_dir,
        dmrg_block_kwargs=dmrg_block_kwargs,
        dmrg_block_kwargs_by_layer=dmrg_block_kwargs_by_layer,
        dmrg_replaced_layer_indices=dmrg_replaced_layer_indices,
        dmrg_block_class_name=dmrg_block_class_name,
        dmrg_block_module_name=dmrg_block_module_name,
        dmrg_block_source_file=dmrg_block_source_file,
    )

    _load_checkpoint_weights(model, ckpt_dir)
    model.to(device)
    model.eval()

    processor = _load_processor_with_fallbacks(
        model_id_or_path,
        ckpt_dir,
        meta,
        trust_remote_code=trust_remote_code,
        token=token,
        fallback_processor_id=fallback_processor_id,
        fallback_processor_subfolder=fallback_processor_subfolder,
        config_name_or_path=getattr(model.config, "_name_or_path", None),
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
    dmrg_replaced_layer_indices: Optional[list[int]] = None,
    dmrg_block_class_name: Optional[str] = None,
    dmrg_block_module_name: Optional[str] = None,
    dmrg_block_source_file: Optional[str] = None,
    fallback_processor_id: Optional[str] = None,
    fallback_processor_subfolder: Optional[str] = None,
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

    ckpt_dir = _find_dmrg_checkpoint_dir(model_id_or_path, subfolder=subfolder, token=tok)
    is_dmrg = ckpt_dir is not None

    if is_dmrg:
        return _load_dmrg_model_and_processor(
            model_id_or_path,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            token=tok,
            module_aliases=module_aliases,
            dmrg_block_kwargs=dmrg_block_kwargs,
            dmrg_block_kwargs_by_layer=dmrg_block_kwargs_by_layer,
            dmrg_replaced_layer_indices=dmrg_replaced_layer_indices,
            dmrg_block_class_name=dmrg_block_class_name,
            dmrg_block_module_name=dmrg_block_module_name,
            dmrg_block_source_file=dmrg_block_source_file,
            fallback_processor_id=fallback_processor_id,
            fallback_processor_subfolder=fallback_processor_subfolder,
            device=device_t,
        )

    _install_module_aliases(module_aliases)

    processor = AutoImageProcessor.from_pretrained(
        model_id_or_path,
        **_hf_pretrained_kwargs(
            trust_remote_code=trust_remote_code,
            token=tok,
            subfolder=subfolder,
        ),
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id_or_path,
        **_hf_pretrained_kwargs(
            trust_remote_code=trust_remote_code,
            token=tok,
            subfolder=subfolder,
        ),
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
    dmrg_replaced_layer_indices: Optional[list[int]] = None,
    dmrg_block_class_name: Optional[str] = None,
    dmrg_block_module_name: Optional[str] = None,
    dmrg_block_source_file: Optional[str] = None,
    fallback_processor_id: Optional[str] = None,
    fallback_processor_subfolder: Optional[str] = None,
    shuffle_stream: bool = False,
    shuffle_buffer: int = 10_000,
    seed: int = 0,
    use_amp: bool = True,
    device: Optional[str] = None,
) -> dict[str, Any]:
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
        fallback_processor_id="google/vit-base-patch16-224",
    )

    # Sparse old-style DMRG checkpoint with one custom block file
    metrics = VIT_metrics(
        "jcandane/ViU",
        subfolder="dmrg_test_first_window/sweep_00_down_10_11",
        dmrg_block_source_file="DiffusionBlock.py",
        fallback_processor_id="jcandane/ViU",
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
        dmrg_replaced_layer_indices=dmrg_replaced_layer_indices,
        dmrg_block_class_name=dmrg_block_class_name,
        dmrg_block_module_name=dmrg_block_module_name,
        dmrg_block_source_file=dmrg_block_source_file,
        fallback_processor_id=fallback_processor_id,
        fallback_processor_subfolder=fallback_processor_subfolder,
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

    out: dict[str, Any] = {
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
        out["dmrg_kind"] = meta.get("kind")
    else:
        out["dmrg_checkpoint"] = False

    return out
