"""
VIT_evaluate.py.py
===========

Evaluate a Hugging Face image-classification model on an image dataset stream.

What this script does
---------------------
This script loads a pretrained image-classification model and its matching
image processor from the Hugging Face Hub, streams a dataset split
(default: ImageNet-1k validation), runs batched inference, and reports
basic evaluation metrics.

Main features
-------------
- Loads a model from `model_id`
- Optionally loads from a `subfolder` inside the Hub repo
- Streams the dataset instead of downloading the full split up front
- Supports evaluating only the first `max_images` samples
- Computes:
    - top-1 accuracy (`acc1`)
    - top-5 accuracy (`acc5`)
    - average cross-entropy loss (`xent`)
    - total parameter count (`params_total`)
- Uses GPU automatically if available
- Optionally uses AMP for faster inference on CUDA

Typical use
-----------
This is useful for comparing a baseline ViT checkpoint against a rewritten /
distilled / DMRG-trained checkpoint stored either as:
- two different model repos, or
- two different subfolders inside the same repo

Example
-------
metrics_vit = VIT_metrics(
    "jcandane/ViU",
    subfolder="vit",
    max_images=8192,
    shuffle_stream=True,
    seed=0,
)

metrics_student = VIT_metrics(
    "jcandane/ViU",
    subfolder="viu",
    max_images=8192,
    shuffle_stream=True,
    seed=0,
)

Important note
--------------
For fair comparison between two checkpoints, both evaluations should use the
same dataset split, the same preprocessing, and ideally the same sampled
subset of examples.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return {"params_total": total}


@torch.inference_mode()
def VIT_metrics(
    model_id: str,
    subfolder: Optional[str] = None,
    dataset_id: str = "ILSVRC/imagenet-1k",
    split: str = "validation",
    max_images: Optional[int] = 50_000,
    batch_size: int = 128,
    trust_remote_code: bool = False,
    # reproducible subset (useful when max_images << full set)
    shuffle_stream: bool = False,
    shuffle_buffer: int = 10_000,
    seed: int = 0,
    # perf
    use_amp: bool = True,
    device: Optional[str] = None,
    ) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # HF token plumbing (works with HF_TOKEN or HF_HUB_TOKEN)
    tok = os.environ.get("HF_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if tok:
        os.environ["HF_HUB_TOKEN"] = tok

    processor = AutoImageProcessor.from_pretrained(
        model_id, subfolder=subfolder, trust_remote_code=trust_remote_code
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id, subfolder=subfolder, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()

    params = count_params(model)

    # --- dataset stream ---
    # token=True is newer; fallback to use_auth_token for older datasets versions
    try:
        ds = load_dataset(dataset_id, split=split, streaming=True, token=True)
    except TypeError:
        ds = load_dataset(dataset_id, split=split, streaming=True, use_auth_token=True)

    if shuffle_stream:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    if max_images is not None:
        ds = ds.take(max_images)  # ensures we evaluate exactly max_images (unless stream ends)

    # --- metrics ---
    n = 0
    top1 = 0
    top5 = 0
    total_loss = 0.0

    images, labels = [], []

    # AMP context for GPU
    amp_enabled = bool(use_amp and device.type == "cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled)

    def flush_batch():
        nonlocal n, top1, top5, total_loss, images, labels
        if not images:
            return

        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        y = torch.tensor(labels, dtype=torch.long, device=device)

        with autocast:
            logits = model(pixel_values=pixel_values).logits

        # sum loss over batch so average is correct at the end
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()

        top1 += (logits.argmax(dim=1) == y).sum().item()
        top5 += logits.topk(5, dim=1).indices.eq(y.view(-1, 1)).any(dim=1).sum().item()
        n += y.size(0)

        images, labels = [], []

    for ex in ds:
        img = ex["image"]
        if hasattr(img, "convert"):  # PIL
            img = img.convert("RGB")
        images.append(img)

        # label field name is usually "label" here; keep it robust anyway
        lab = ex.get("label", ex.get("labels"))
        labels.append(int(lab))

        if len(images) >= batch_size:
            flush_batch()

    # last partial batch
    flush_batch()

    if n == 0:
        raise RuntimeError("Evaluated 0 samples (stream empty or wrong split?).")

    return {
        "acc1": top1 / n,
        "acc5": top5 / n,
        "xent": total_loss / n,
        "n": n,
        **params,
    }
