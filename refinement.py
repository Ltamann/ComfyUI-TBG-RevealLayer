"""Align per-layer editing references and collect native sampler list outputs."""

import json

import torch
import torch.nn.functional as F

from .node import _parse_comfy_bboxes

DETAIL_INSTRUCTION = "Improve fine detail, texture, and edge clarity while preserving identity, pose, geometry, colors, lighting, and composition."
UPSAMPLER_SYSTEM = (
    "Write one detailed English image-editing instruction for FLUX 3 Image Klein. "
    "Three images are attached in order: ref_image_0 is the layer to refine, "
    "ref_image_1 is the original scene, and ref_image_2 is the SAM3 extraction. "
    "Follow the user's layer role exactly. Describe visible identity and appearance accurately; "
    "do not invent features or import unrelated subjects. Retain the tags <obj_1>, <ref_image_0>, "
    "<ref_image_1>, and <ref_image_2> where relevant. Preserve placement, framing, neutral canvas, "
    "and every requested exclusion. For background layers, preserve the completed background "
    "and never restore the foreground objects that were removed. Return only the expanded edit "
    "instruction, without reasoning, headings, coordinates, or JSON. A fixed layout table is appended separately."
)


def resize_rgb(images, height, width):
    return F.interpolate(images.movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False).movedim(1, -1).clamp(0, 1)


def flatten_rgba(images):
    return images[..., :3] * images[..., 3:4] + 0.5 * (1 - images[..., 3:4])


def composite_layers(layers):
    premultiplied = torch.zeros_like(layers[0, ..., :3])
    alpha = torch.zeros_like(layers[0, ..., 3:4])
    for layer in layers:
        a = layer[..., 3:4]
        premultiplied = layer[..., :3] * a + premultiplied * (1 - a)
        alpha = a + alpha * (1 - a)
    rgb = torch.where(alpha > 0, premultiplied / alpha.clamp_min(1e-8), 0)
    return torch.cat((rgb, alpha), dim=-1).unsqueeze(0)


class TBGRevealLayerReferences:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original_image": ("IMAGE",),
            "reveallayer_layers": ("IMAGE",),
            "sam_masks": ("MASK",),
            "bounding_boxes": ("BOUNDING_BOX",),
            "resolution": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
            "instruction": ("STRING", {"default": DETAIL_INSTRUCTION, "multiline": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "TBG_REFINEMENT_PLAN", "IMAGE", "IMAGE")
    RETURN_NAMES = ("layer_reference", "original_reference", "sam_reference", "edit_instruction", "layout_table", "upsampler_system", "plan", "comparison_batch", "sam_layers_rgba")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False, False, False)
    FUNCTION = "prepare"
    CATEGORY = "TBG/RevealLayer"
    DESCRIPTION = "One single-image reference set per foreground, background, and object layer. Native list mapping runs one edit per set."

    def prepare(self, original_image, reveallayer_layers, sam_masks, bounding_boxes, resolution, instruction):
        if original_image.ndim != 4 or original_image.shape[0] != 1:
            raise ValueError("Layer refinement expects one source image at a time.")
        original = original_image.detach().to(device="cpu", dtype=torch.float32)[..., :3].clamp(0, 1)
        _, height, width, _ = original.shape
        boxes = _parse_comfy_bboxes(bounding_boxes)
        layers = reveallayer_layers.detach().to(device="cpu", dtype=torch.float32)
        if layers.ndim != 4 or layers.shape[-1] != 4 or tuple(layers.shape[1:3]) != (height, width):
            raise ValueError("Connect full RGBA all_layers_rgba and enable resize_output_to_input in TBG RevealLayer.")
        if len(layers) != len(boxes) + 2:
            raise ValueError("Layer/box counts differ. Use max_layers >= box count + 2 (up to 12), and remove invalid or tiny boxes.")
        if sam_masks.ndim != 3 or len(sam_masks) != len(boxes):
            raise ValueError("SAM3 must segment the edited boxes with individual_masks=true and no text conditioning, producing one mask per box.")
        masks = sam_masks.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        if tuple(masks.shape[1:]) != (height, width):
            masks = F.interpolate(masks.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
        union = masks.amax(dim=0) if len(masks) else torch.zeros(height, width)
        sam_alpha = torch.cat((union.unsqueeze(0), (1 - union).unsqueeze(0), masks), dim=0)
        sam_rgba = torch.cat((original.expand(len(layers), -1, -1, -1), sam_alpha.unsqueeze(-1)), dim=-1)
        scale = resolution / max(height, width)
        work_h = max(16, round(height * scale / 16) * 16)
        work_w = max(16, round(width * scale / 16) * 16)
        rgb_layers = resize_rgb(flatten_rgba(layers), work_h, work_w)
        rgb_sam = resize_rgb(flatten_rgba(sam_rgba), work_h, work_w)
        rgb_original = resize_rgb(original, work_h, work_w)
        prompts, layouts, jobs = [], [], []
        for index in range(len(layers)):
            role = "foreground" if index == 0 else "background" if index == 1 else f"object_{index - 1}"
            box = [0, 0, width, height] if index < 2 else boxes[index - 2]
            x1, y1, x2, y2 = box
            normalized = [max(0, min(1000, round(v))) for v in (y1 / height * 1000, x1 / width * 1000, y2 / height * 1000, x2 / width * 1000)]
            if role == "background":
                purpose = (
                    "Refine the completed background <obj_1> in <ref_image_0>. "
                    "Use <ref_image_1> only for visible scene materials and lighting, and <ref_image_2> "
                    "only for visible background evidence. Keep the selected foreground subjects removed. "
                    "Retain the inferred background behind them; the gray holes in the SAM reference are missing evidence, not scene content."
                )
            elif index == 0:
                purpose = (
                    "Refine the global foreground layer <obj_1> in <ref_image_0>, using the original scene "
                    "<ref_image_1> for faithful appearance and the visible foreground extraction <ref_image_2> for detail."
                )
            else:
                purpose = (
                    "Refine only the selected subject <obj_1> in <ref_image_0>. Restore its exact identity and "
                    "appearance from the same bounded subject in <ref_image_1>, using its SAM extraction "
                    "<ref_image_2> as additional visible-detail evidence. Preserve completed occluded parts from the first reference. "
                    "Do not copy other subjects or the original scene background into this layer."
                )
            prompts.append(purpose + " " + instruction.strip() + " Keep the subject at its existing location and size. Preserve the neutral canvas outside this layer; change nothing else.")
            rows = [{"id": "obj_1", "from": "ref_image_1" if index >= 2 else "ref_image_0", "src_bbox": normalized, "tgt_bbox": normalized, "desc": role.replace("_", " ") + " layer; preserve identity and layout"}]
            layouts.append(json.dumps(rows, ensure_ascii=False))
            jobs.append({"index": index, "role": role, "box_xyxy": box})
        comparisons = torch.cat([torch.cat((rgb_original, rgb_sam[i:i + 1], rgb_layers[i:i + 1])) for i in range(len(layers))])
        plan = {"jobs": jobs, "alpha": layers[..., 3].clone(), "height": height, "width": width, "work_height": work_h, "work_width": work_w}
        count = len(layers)
        return ([rgb_layers[i:i + 1] for i in range(count)], [rgb_original.clone() for _ in range(count)],
                [rgb_sam[i:i + 1] for i in range(count)], prompts, layouts, UPSAMPLER_SYSTEM, plan, comparisons, sam_rgba)


class TBGRevealLayerCollect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"refined_images": ("IMAGE",), "plan": ("TBG_REFINEMENT_PLAN",)}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("refined_layers_rgba", "refined_objects_rgba", "refined_composite_rgba")
    FUNCTION = "collect"
    CATEGORY = "TBG/RevealLayer"
    DESCRIPTION = "Collect one RGB edit per layer and restore the original RevealLayer alpha and layer order."

    def collect(self, refined_images, plan):
        if len(plan) != 1:
            raise ValueError("Expected one refinement plan, not multiple source-image jobs.")
        plan = plan[0]
        if len(refined_images) != len(plan["jobs"]):
            raise ValueError("Refinement output count does not match the layer plan; keep each sampler job at batch size one.")
        rgb = []
        for image in refined_images:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] < 3:
                raise ValueError("Each refinement job must return exactly one RGB image.")
            rgb.append(resize_rgb(image.detach().to(device="cpu", dtype=torch.float32)[..., :3], plan["height"], plan["width"]))
        rgb = torch.cat(rgb)
        alpha = plan["alpha"].unsqueeze(-1)
        layers = torch.cat((torch.where(alpha > 0, rgb, 0), alpha), dim=-1)
        objects = layers[2:] if len(layers) > 2 else torch.zeros_like(layers[:1])
        return layers, objects, composite_layers(layers)


NODE_CLASS_MAPPINGS = {"TBGRevealLayerReferences": TBGRevealLayerReferences, "TBGRevealLayerCollect": TBGRevealLayerCollect}
NODE_DISPLAY_NAME_MAPPINGS = {"TBGRevealLayerReferences": "TBG RevealLayer · Prepare Layer References", "TBGRevealLayerCollect": "TBG RevealLayer · Collect Refined Layers"}
