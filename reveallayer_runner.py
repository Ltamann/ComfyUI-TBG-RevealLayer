"""Run RevealLayer through ComfyUI's FLUX model and model manager."""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .native_decoder import AutoencoderKLTransformerTraining
from .native_refiner import HybridRefiner
from .native_flux import forward_reveallayer
from .native_bridge import FLUX_SCALING_FACTOR, FLUX_SHIFT_FACTOR, validate_flux_model


class _RevealLayerAuxiliary(nn.Module):
    def __init__(self, layer_pe: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.layer_pe = nn.Parameter(layer_pe.to(dtype=dtype))
        self.refiner = None
        self.to(dtype=dtype)

    def get_dtype(self):
        return self.layer_pe.dtype


def _target_path(value: str | Path, label: str, directory=False) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Missing {label} asset; retry TBG RevealLayer's first-run setup.")
    path = Path(raw).expanduser().resolve()
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {path}")
    return path


def _load_layer_pe(checkpoint_dir: Path, hidden_size: int) -> torch.Tensor:
    path = checkpoint_dir / "layer_pe.pt"
    if not path.is_file():
        raise FileNotFoundError(f"RevealLayer layer positional weights are missing from {checkpoint_dir}.")
    layer_pe = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(layer_pe, torch.Tensor) or layer_pe.ndim != 5 or layer_pe.shape[0] != 1 or layer_pe.shape[2:4] != (1, 1) or layer_pe.shape[-1] != hidden_size:
        raise ValueError(f"Unexpected RevealLayer layer positional tensor shape in {path}: {getattr(layer_pe, 'shape', None)}")
    return layer_pe


def _load_reveallayer_lora(model: Any, checkpoint_dir: Path):
    import comfy.sd

    from safetensors.torch import load_file

    lora_path = checkpoint_dir / "pytorch_lora_weights.safetensors"
    if not lora_path.is_file():
        raise FileNotFoundError(f"RevealLayer LoRA weights are missing: {lora_path}")
    lora = load_file(str(lora_path), device="cpu")
    lora = {key.replace("transformer.module.", "transformer."): value for key, value in lora.items()}
    previous_patch_count = sum(len(patches) for patches in model.patches.values())
    patched_model, _ = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0.0)
    new_patch_count = sum(len(patches) for patches in patched_model.patches.values()) if patched_model else 0
    if patched_model is None or new_patch_count <= previous_patch_count:
        raise RuntimeError("ComfyUI did not map RevealLayer's LoRA to the connected FLUX transformer.")
    return patched_model


def load_reveallayer_models(
    model: Any,
    checkpoint_path: str | Path,
    transparent_decoder_path: str | Path,
    max_layers: int,
):
    import comfy.model_patcher

    config = validate_flux_model(model)
    if not config.get("guidance_embed", False):
        raise ValueError("RevealLayer requires the guidance-embedded FLUX.1-dev transformer.")
    checkpoint_dir = _target_path(checkpoint_path, "RevealLayer checkpoint", directory=True)
    decoder_file = _target_path(transparent_decoder_path, "transparent decoder checkpoint")
    base_dtype = model.model_dtype() or model.model.get_dtype()
    if base_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"RevealLayer requires FP16, BF16 or FP32 FLUX weights; got {base_dtype}.")

    layer_pe = _load_layer_pe(checkpoint_dir, int(config["hidden_size"]))
    if layer_pe.shape[1] < max_layers:
        raise ValueError(f"The checkpoint has {layer_pe.shape[1]} layer embeddings; max_layers={max_layers} was requested.")
    auxiliary = _RevealLayerAuxiliary(layer_pe, base_dtype)
    refiner_file = checkpoint_dir / "Refiner.pt"
    auxiliary.refiner = HybridRefiner(feature_dim=int(config["hidden_size"])).to(dtype=base_dtype)
    refiner_state = torch.load(refiner_file, map_location="cpu", weights_only=True)
    if not isinstance(refiner_state, dict):
        raise TypeError(f"Expected a Refiner state dictionary in {refiner_file}.")
    auxiliary.refiner.load_state_dict(refiner_state, strict=True)

    auxiliary_patcher = comfy.model_patcher.ModelPatcher(
        auxiliary, load_device=model.load_device, offload_device=model.offload_device
    )
    patched_model = _load_reveallayer_lora(model, checkpoint_dir)

    decoder = AutoencoderKLTransformerTraining().to(dtype=base_dtype)
    decoder_state = torch.load(decoder_file, map_location="cpu", weights_only=True)
    if not isinstance(decoder_state, dict):
        raise TypeError(f"Expected a transparent decoder state dictionary in {decoder_file}.")
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    decoder_patcher = comfy.model_patcher.ModelPatcher(
        decoder, load_device=model.load_device, offload_device=model.offload_device
    )
    return patched_model, auxiliary_patcher, decoder_patcher


def _make_latent_image_ids(boxes, height, width, device):
    patch_height = height // 16
    patch_width = width // 16
    ids = []
    for layer, box in enumerate(boxes):
        if box is None:
            continue
        grid = torch.zeros((patch_height, patch_width, 3), device=device, dtype=torch.float32)
        grid[..., 0] = layer
        grid[..., 1] = torch.arange(patch_height, device=device, dtype=torch.float32)[:, None]
        grid[..., 2] = torch.arange(patch_width, device=device, dtype=torch.float32)[None, :]
        x1, y1, x2, y2 = box
        ids.append(grid[y1 // 16:y2 // 16, x1 // 16:x2 // 16].reshape(-1, 3))
    return torch.cat(ids, dim=0)


def _make_adapter_data(masked_latents, boxes, height, width, device, dtype):
    layer_count = len(boxes)
    per_layer = layer_count - 1
    patch_h, patch_w = height // 16, width // 16
    latent_h, latent_w = height // 8, width // 8
    # ComfyUI VAE.encode returns the unscaled latent. RevealLayer conditions on
    # the same normalized FLUX latent space used by its denoising transformer.
    masked_latents = (masked_latents.to(device=device, dtype=dtype) - FLUX_SHIFT_FACTOR) * FLUX_SCALING_FACTOR
    masked_latents = masked_latents.reshape(per_layer, 16, latent_h // 2, 2, latent_w // 2, 2)
    masked_latents = masked_latents.permute(0, 2, 4, 1, 3, 5).reshape(per_layer, patch_h, patch_w, 64)

    mask_pixel = torch.ones((1, per_layer, 1, height, width), device=device, dtype=dtype)
    overlap_counter = torch.zeros((1, height, width), device=device)
    for box in boxes[2:]:
        x1, y1, x2, y2 = box
        overlap_counter[:, y1:y2, x1:x2] += 1
    overlap_region = overlap_counter > 1
    mask_pixel[:, 0] = overlap_counter > 0
    for index, (x1, y1, x2, y2) in enumerate(boxes[2:], start=1):
        mask_pixel[:, index, :, y1:y2, x1:x2] = overlap_region[:, y1:y2, x1:x2].unsqueeze(1)

    mask_flat = mask_pixel.reshape(per_layer, 1, height, width)
    mask_latents = mask_flat.reshape(per_layer, 1, latent_h, 8, latent_w, 8)
    mask_latents = mask_latents.permute(0, 2, 4, 1, 3, 5).reshape(per_layer, latent_h, latent_w, 64)
    mask_latents = mask_latents.permute(0, 3, 1, 2)
    mask_latents = F.max_pool2d(mask_latents, kernel_size=2, stride=2)
    mask_latents = mask_latents.permute(0, 2, 3, 1).to(dtype=dtype)
    return torch.cat((masked_latents, mask_latents), dim=-1).reshape(1, per_layer, patch_h, patch_w, 128)


def _flow_schedule(steps: int, image_sequence_length: int, device):
    sigmas = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
    slope = (1.15 - 0.5) / (4096 - 256)
    mu = image_sequence_length * slope + (0.5 - slope * 256)
    exp_mu = math.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
    sigmas = torch.from_numpy(sigmas).to(device=device, dtype=torch.float32)
    timesteps = sigmas * 1000.0
    return timesteps, torch.cat((sigmas, torch.zeros(1, device=device, dtype=torch.float32)))


def _run_transformer(model, auxiliary, hidden_states, adapter_data, split_sizes, boxes,
                     positive, pooled, timestep, img_ids, txt_ids, guidance, transformer_options):
    import comfy.model_management

    comfy.model_management.load_models_gpu([model, auxiliary])
    diffusion_model = model.get_model_object("diffusion_model")
    return forward_reveallayer(
        model=diffusion_model,
        hidden_states=hidden_states,
        adapter_data=adapter_data,
        split_sizes=split_sizes,
        list_layer_box=boxes,
        encoder_hidden_states=positive,
        pooled_projections=pooled,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        layer_pe=auxiliary.model.layer_pe,
        refiner=auxiliary.model.refiner,
        guidance=guidance,
        transformer_options=transformer_options,
    )


def run_reveallayer(
    model: Any,
    auxiliary: Any,
    decoder: Any,
    vae_data: dict[str, Any],
    clip_data: dict[str, torch.Tensor],
    steps: int,
    guidance_scale: float,
    seed: int,
    progress=None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> torch.Tensor:
    import comfy.model_management

    device = model.load_device
    dtype = model.model_dtype() or model.model.get_dtype()
    boxes = vae_data["validation_boxes"]
    height, width = vae_data["height"], vae_data["width"]
    layer_count = len(boxes)
    logging.info(
        "RevealLayer: %d sampling steps, %d layers, working size %dx%d, guidance %.2f, seed %d",
        steps, layer_count, width, height, guidance_scale, seed,
    )
    latent_height, latent_width = height // 8, width // 8
    full_latent = (vae_data["full_latent"].to(device=device, dtype=dtype) - FLUX_SHIFT_FACTOR) * FLUX_SCALING_FACTOR
    full_latent = full_latent.reshape(1, 1, 16, latent_height, latent_width)
    latents = torch.randn(
        (1, layer_count, 16, latent_height, latent_width),
        generator=torch.Generator(device=device).manual_seed(int(seed)),
        device=device,
        dtype=dtype,
    )
    adapter_data = _make_adapter_data(
        vae_data["masked_latents"], boxes, height, width, device, dtype
    )
    split_sizes = [(height // 16) * (width // 16)]
    split_sizes.extend(((y2 - y1) // 16) * ((x2 - x1) // 16) for x1, y1, x2, y2 in boxes[2:])
    img_ids = _make_latent_image_ids(boxes, height, width, device)
    positive = clip_data["positive_cond"].to(device=device, dtype=dtype)
    positive_pooled = clip_data["positive_pooled"].to(device=device, dtype=dtype)
    negative = clip_data["negative_cond"].to(device=device, dtype=dtype)
    negative_pooled = clip_data["negative_pooled"].to(device=device, dtype=dtype)
    text_ids = torch.zeros((positive.shape[1], 3), device=device, dtype=torch.float32)
    guidance = torch.full((1,), float(guidance_scale), device=device, dtype=torch.float32)
    timesteps, sigmas = _flow_schedule(steps, img_ids.shape[0], device)
    transformer_options = dict(model.model_options.get("transformer_options", {}))
    transformer_options["prefetch_dynamic_vbars"] = model.is_dynamic()

    for index, timestep in enumerate(timesteps):
        comfy.model_management.throw_exception_if_processing_interrupted()
        latents[:, :1] = full_latent
        time = (timestep / 1000.0).expand(latents.shape[0])
        positive_prediction = _run_transformer(
            model, auxiliary, latents, adapter_data, split_sizes, boxes,
            positive, positive_pooled, time, img_ids, text_ids, guidance, transformer_options,
        )
        negative_prediction = _run_transformer(
            model, auxiliary, latents, adapter_data, split_sizes, boxes,
            negative, negative_pooled, time, img_ids, text_ids, guidance, transformer_options,
        )
        prediction = negative_prediction + 3.5 * (positive_prediction - negative_prediction)
        delta = float((sigmas[index + 1] - sigmas[index]).item())
        latents = latents + prediction * delta
        if progress is not None:
            progress.update_absolute(progress_offset + index + 1, total=progress_total)

    grey = (vae_data["grey_latents"].to(device=device, dtype=dtype) - FLUX_SHIFT_FACTOR) * FLUX_SCALING_FACTOR
    grey = grey.reshape(1, layer_count, 16, latent_height, latent_width)
    for layer, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = (coordinate // 8 for coordinate in box)
        grey[:, layer, :, y1:y2, x1:x2] = latents[:, layer, :, y1:y2, x1:x2]

    # The transparent decoder uses ordinary PyTorch modules rather than ComfyUI
    # manual-cast ops. A low-VRAM partial load can leave those weights on CPU
    # while its input remains on the FLUX device, so ask ComfyUI to fully load
    # this small decoder before running it.
    comfy.model_management.load_models_gpu([decoder], force_full_load=True)
    # The transformer uses normalized FLUX latents; both the standard VAE
    # decoder and RevealLayer's transparent decoder expect raw VAE latents.
    # Match CustomFluxPipelineCfg's inverse transform before either decoding.
    decoder_weight = decoder.model.decoder.fc_in.weight
    decoder_input = (grey[0] / FLUX_SCALING_FACTOR + FLUX_SHIFT_FACTOR).to(
        device=decoder_weight.device, dtype=decoder_weight.dtype
    )
    foreground, alpha = decoder.model(decoder_input, [boxes])
    foreground = (foreground + 1.0).clamp(0.0, 2.0) / 2.0
    alpha = (alpha + 1.0).clamp(0.0, 2.0) / 2.0
    rgba = torch.cat((foreground, alpha), dim=1).permute(0, 2, 3, 1)
    return rgba.detach().to(device="cpu", dtype=torch.float32).contiguous()


def unload_patcher(patcher: Any) -> None:
    if patcher is not None:
        import comfy.model_management

        comfy.model_management.unload_model_and_clones(
            patcher, unload_additional_models=False, all_devices=True
        )
