"""ComfyUI input preparation for the in-process RevealLayer node."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

PROMPT = "Decompose the image into foreground and background."
FLUX_SHIFT_FACTOR = 0.1159
FLUX_SCALING_FACTOR = 0.3611


def validate_flux_model(model_patcher: Any) -> dict[str, Any]:
    base = getattr(model_patcher, "model", None)
    config = getattr(getattr(base, "model_config", None), "unet_config", None)
    if config is None:
        raise TypeError("MODEL input is not a ComfyUI FLUX ModelPatcher.")
    cfg = dict(config)
    expected = {
        "image_model": "flux",
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 3072,
        "context_in_dim": 4096,
        "vec_in_dim": 768,
        "num_heads": 24,
        "depth": 19,
        "depth_single_blocks": 38,
        "patch_size": 2,
    }
    mismatches = [f"{key}={cfg.get(key)!r} (needs {value!r})" for key, value in expected.items() if cfg.get(key) != value]
    if mismatches:
        raise ValueError("RevealLayer requires the standard FLUX.1-dev MODEL: " + "; ".join(mismatches))
    return cfg


def encode_comfy_clip(clip: Any) -> dict[str, torch.Tensor]:
    def encode(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        # RevealLayer's released pipeline encodes both prompts at length 512.
        # ComfyUI's FLUX default is 256; override it only for this call.
        tokens = clip.tokenize(text, tokenizer_options={"t5xxl_min_length": 512})
        try:
            scheduled = clip.encode_from_tokens_scheduled(tokens, show_pbar=False)
        except TypeError:
            scheduled = clip.encode_from_tokens_scheduled(tokens)
        if not scheduled:
            raise RuntimeError(f"ComfyUI CLIP produced no conditioning for prompt {text!r}.")
        if len(scheduled) != 1:
            raise RuntimeError("RevealLayer requires unscheduled FLUX CLIP conditioning; disable CLIP keyframe scheduling.")
        cond, pooled_dict = scheduled[0]
        pooled = pooled_dict.get("pooled_output") if isinstance(pooled_dict, dict) else None
        if not isinstance(cond, torch.Tensor) or not isinstance(pooled, torch.Tensor):
            raise ValueError("Connect a FLUX DualCLIP model that returns T5 embeddings and pooled CLIP output.")
        if cond.ndim != 3 or cond.shape[0] != 1 or cond.shape[-1] != 4096:
            raise ValueError(f"Expected one FLUX T5 embedding batch [1,T,4096], got {tuple(cond.shape)}.")
        if pooled.ndim != 2 or pooled.shape != (1, 768):
            raise ValueError(f"Expected one pooled FLUX CLIP embedding [1,768], got {tuple(pooled.shape)}.")
        return cond.detach().to(device="cpu").contiguous(), pooled.detach().to(device="cpu").contiguous()

    positive_cond, positive_pooled = encode(PROMPT)
    negative_cond, negative_pooled = encode("")
    return {
        "positive_cond": positive_cond,
        "positive_pooled": positive_pooled,
        "negative_cond": negative_cond,
        "negative_pooled": negative_pooled,
    }


def _resize_geometry(image: Image.Image, raw_boxes: list[list[float]], resolution: int = 1024):
    orig_w, orig_h = image.size
    scale = resolution / max(orig_w, orig_h)
    content_w = int(round(orig_w * scale))
    content_h = int(round(orig_h * scale))
    target_w = max(16, (content_w // 16) * 16)
    target_h = max(16, (content_h // 16) * 16)
    offset_x = (content_w - target_w) // 2
    offset_y = (content_h - target_h) // 2
    scale_x, scale_y = content_w / orig_w, content_h / orig_h
    accepted = []
    for x1, y1, x2, y2 in raw_boxes:
        sx1, sy1 = x1 * scale_x, y1 * scale_y
        sx2, sy2 = x2 * scale_x, y2 * scale_y
        scaled_w, scaled_h = sx2 - sx1, sy2 - sy1
        cx1 = max(0.0, min(sx1 - offset_x, float(target_w)))
        cy1 = max(0.0, min(sy1 - offset_y, float(target_h)))
        cx2 = max(0.0, min(sx2 - offset_x, float(target_w)))
        cy2 = max(0.0, min(sy2 - offset_y, float(target_h)))
        visible_w, visible_h = cx2 - cx1, cy2 - cy1
        if visible_w > 1 and visible_h > 1 and scaled_w > 0 and scaled_h > 0 and visible_w / scaled_w >= 0.5 and visible_h / scaled_h >= 0.5:
            accepted.append([
                max(0, min(math.floor(cx1 / 16) * 16, target_w)),
                max(0, min(math.floor(cy1 / 16) * 16, target_h)),
                max(0, min(math.ceil(cx2 / 16) * 16, target_w)),
                max(0, min(math.ceil(cy2 / 16) * 16, target_h)),
            ])
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS)
    cropped = resized.crop((offset_x, offset_y, offset_x + target_w, offset_y + target_h))
    boxes = [[0, 0, target_w, target_h], [0, 0, target_w, target_h]] + accepted
    return cropped, boxes[:12]


def encode_comfy_vae_inputs(
    vae: Any,
    image: Image.Image,
    boxes: list[list[float]],
    max_layers: int,
) -> dict[str, Any]:
    latent_channels = getattr(vae, "latent_channels", None)
    if latent_channels is None or int(latent_channels) != 16:
        raise ValueError("Connect a FLUX VAE with 16 latent channels.")
    compression = vae.spacial_compression_encode()
    if int(compression) != 8:
        raise ValueError(f"RevealLayer requires 8x VAE compression; the connected VAE uses {compression}x.")

    cropped, validation_boxes = _resize_geometry(image, boxes)
    validation_boxes = validation_boxes[:max_layers]
    width, height = cropped.size
    layer_count = len(validation_boxes)
    background_and_objects = layer_count - 1

    full_pixels = torch.from_numpy(np.asarray(cropped.convert("RGB"), dtype=np.float32).copy() / 255.0).unsqueeze(0).contiguous()
    full_latent = vae.encode(full_pixels)
    if full_latent.ndim != 4 or full_latent.shape[1] != 16:
        raise RuntimeError(f"Connected VAE returned unexpected latent shape {tuple(full_latent.shape)}.")

    image_m11 = full_pixels.permute(0, 3, 1, 2) * 2.0 - 1.0
    all_images = image_m11.expand(1, background_and_objects, 3, height, width).reshape(background_and_objects, 3, height, width)
    mask_pixel = torch.ones((1, background_and_objects, 1, height, width), dtype=torch.float32)
    overlap_counter = torch.zeros((1, height, width), dtype=torch.float32)
    for x1, y1, x2, y2 in validation_boxes[2:]:
        overlap_counter[:, y1:y2, x1:x2] += 1
    is_fg_union = overlap_counter > 0
    is_overlap_global = overlap_counter > 1
    mask_pixel[:, 0] = is_fg_union
    for layer, (x1, y1, x2, y2) in enumerate(validation_boxes[2:], start=1):
        mask_pixel[:, layer, :, y1:y2, x1:x2] = is_overlap_global[:, y1:y2, x1:x2].unsqueeze(1)
    mask_flat = mask_pixel.reshape(background_and_objects, 1, height, width)
    masked_m11 = all_images * (1.0 - mask_flat)
    masked_pixels = ((masked_m11 + 1.0) / 2.0).permute(0, 2, 3, 1).contiguous()
    masked_latents = vae.encode(masked_pixels)

    grey_pixels = torch.full((layer_count, height, width, 3), 0.5, dtype=torch.float32)
    grey_latents = vae.encode(grey_pixels)
    if masked_latents.shape[0] != background_and_objects or grey_latents.shape[0] != layer_count:
        raise RuntimeError("Connected VAE returned an unexpected masked/gray batch size.")

    return {
        "validation_boxes": validation_boxes,
        "width": width,
        "height": height,
        "full_latent": full_latent.detach().to(device="cpu").contiguous(),
        "masked_latents": masked_latents.detach().to(device="cpu").contiguous(),
        "grey_latents": grey_latents.detach().to(device="cpu").contiguous(),
    }
