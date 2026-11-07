"""ComfyUI node for in-process RevealLayer inference."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

import comfy.utils

from .native_bridge import encode_comfy_clip, encode_comfy_vae_inputs
from .dependencies import ensure_dependencies
from .model_downloads import ensure_reveallayer_models


def _parse_comfy_bboxes(data: Any, frame_index: int = 0) -> list[list[float]]:
    """Convert a ComfyUI BOUNDING_BOX frame to RevealLayer XYXY pixel boxes."""
    if data is None:
        return []
    if isinstance(data, dict):
        frame = [data]
    elif isinstance(data, (list, tuple)):
        frame = list(data)
        if frame and isinstance(frame[0], (list, tuple)):
            if len(frame) == 1:
                frame = list(frame[0])
            elif frame_index < len(frame):
                frame = list(frame[frame_index])
            else:
                raise ValueError("BOUNDING_BOX batch must have one frame or one frame per image.")
    else:
        raise TypeError(f"Expected a ComfyUI BOUNDING_BOX value, got {type(data).__name__}.")

    boxes: list[list[float]] = []
    for index, item in enumerate(frame, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Bounding box #{index} must contain x, y, width, and height fields.")
        try:
            x, y, width, height = (float(item[key]) for key in ("x", "y", "width", "height"))
        except KeyError as exc:
            raise ValueError(f"Bounding box #{index} is missing {exc.args[0]!r}.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Bounding box #{index} contains a non-numeric coordinate.") from exc
        if not np.isfinite((x, y, width, height)).all():
            raise ValueError(f"Bounding box #{index} contains a non-finite coordinate.")
        if width <= 0 or height <= 0:
            raise ValueError(f"Bounding box #{index} must have positive width and height.")
        boxes.append([x, y, x + width, y + height])
    return boxes


def _limit_boxes(boxes: list[list[float]], max_layers: int) -> list[list[float]]:
    limit = max(0, int(max_layers) - 2)
    if len(boxes) > limit:
        logging.warning(
            "RevealLayer received %d object boxes with max_layers=%d; using the first %d boxes.",
            len(boxes), max_layers, limit,
        )
    return boxes[:limit]


def _as_pil_rgb(frame: np.ndarray) -> Image.Image:
    frame = np.nan_to_num(frame[..., :3], nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _rgba_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGBA"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array)


class TBGRevealLayer:
    """Decompose an image using ComfyUI-managed FLUX, CLIP, and VAE models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "max_layers": ("INT", {"default": 12, "min": 2, "max": 12, "step": 1}),
                "resize_output_to_input": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "bounding_boxes": ("BOUNDING_BOX", {"tooltip": "Optional pixel boxes from Create Bounding Boxes or a grounding node. Two layer slots are reserved; excess boxes are skipped."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "foreground_rgba",
        "background_rgba",
        "object_layers_rgba",
        "object_masks",
        "all_layers_rgba",
        "composite_rgba",
    )
    FUNCTION = "decompose"
    CATEGORY = "TBG/RevealLayer"
    DESCRIPTION = "Occlusion-aware image decomposition. Downloads missing RevealLayer checkpoints on first use."

    def decompose(
        self,
        model: Any,
        clip: Any,
        vae: Any,
        image: torch.Tensor,
        steps: int,
        guidance_scale: float,
        seed: int,
        max_layers: int,
        resize_output_to_input: bool,
        bounding_boxes: Any = None,
        unique_id: str | None = None,
    ):
        batch = image.detach().to(device="cpu", dtype=torch.float32)
        if batch.ndim != 4 or batch.shape[-1] not in (3, 4) or batch.shape[0] < 1:
            raise ValueError(f"Expected a non-empty ComfyUI IMAGE batch [B,H,W,3|4], got {tuple(batch.shape)}.")
        frames = [_as_pil_rgb(frame) for frame in batch.numpy()]
        if isinstance(bounding_boxes, (list, tuple)) and bounding_boxes and isinstance(bounding_boxes[0], (list, tuple)):
            if len(bounding_boxes) not in (1, len(frames)):
                raise ValueError("BOUNDING_BOX batch must have one frame or one frame per image.")
        frame_boxes = [
            _limit_boxes(_parse_comfy_bboxes(bounding_boxes, index), max_layers)
            for index in range(len(frames))
        ]
        ensure_dependencies()
        checkpoint_dir, decoder_file = ensure_reveallayer_models(unique_id)
        from .reveallayer_runner import load_reveallayer_models, run_reveallayer, unload_patcher

        progress_total = int(steps) * len(frames)
        progress_bar = (
            comfy.utils.ProgressBar(progress_total, node_id=unique_id)
            if comfy.utils.PROGRESS_BAR_ENABLED else None
        )

        patched_model = auxiliary = decoder = None
        try:
            patched_model, auxiliary, decoder = load_reveallayer_models(
                model, checkpoint_dir, decoder_file, int(max_layers)
            )
            clip_data = encode_comfy_clip(clip)

            foregrounds: list[torch.Tensor] = []
            backgrounds: list[torch.Tensor] = []
            object_layers: list[torch.Tensor] = []
            object_masks: list[torch.Tensor] = []
            all_layers: list[torch.Tensor] = []
            composites: list[torch.Tensor] = []

            for index, frame in enumerate(frames):
                data = encode_comfy_vae_inputs(
                    vae=vae,
                    image=frame,
                    boxes=frame_boxes[index],
                    max_layers=int(max_layers),
                )
                rgba = run_reveallayer(
                    patched_model,
                    auxiliary,
                    decoder,
                    data,
                    clip_data,
                    int(steps),
                    float(guidance_scale),
                    int(seed) + index,
                    progress=progress_bar,
                    progress_offset=index * int(steps),
                    progress_total=progress_total,
                )
                layers = [Image.fromarray(np.clip(np.nan_to_num(layer.numpy()) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGBA") for layer in rgba]
                if resize_output_to_input:
                    size = frames[index].size
                    layers = [layer.resize(size, Image.Resampling.LANCZOS) for layer in layers]

                foregrounds.append(_rgba_tensor(layers[0]))
                backgrounds.append(_rgba_tensor(layers[1]))
                all_layers.extend(_rgba_tensor(layer) for layer in layers)
                object_layers.extend(_rgba_tensor(layer) for layer in layers[2:])
                object_masks.extend(
                    torch.from_numpy(np.asarray(layer.getchannel("A"), dtype=np.float32).copy() / 255.0)
                    for layer in layers[2:]
                )
                composite = layers[0]
                for layer in layers[1:]:
                    composite = Image.alpha_composite(composite, layer)
                composites.append(_rgba_tensor(composite))

            if not object_layers:
                output_height, output_width = foregrounds[0].shape[:2]
                object_layers = [torch.zeros((output_height, output_width, 4), dtype=torch.float32)]
                object_masks = [torch.zeros((output_height, output_width), dtype=torch.float32)]

            return (
                torch.stack(foregrounds),
                torch.stack(backgrounds),
                torch.stack(object_layers),
                torch.stack(object_masks),
                torch.stack(all_layers),
                torch.stack(composites),
            )
        finally:
            for patcher in (decoder, auxiliary, patched_model):
                if patcher is not None:
                    unload_patcher(patcher)


NODE_CLASS_MAPPINGS = {"TBGRevealLayer": TBGRevealLayer}
NODE_DISPLAY_NAME_MAPPINGS = {"TBGRevealLayer": "TBG RevealLayer"}
