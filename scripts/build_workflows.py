"""Regenerate the two release workflows and their embedded Markdown guide."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(grounded, refinement=False):
    nodes, links = [], []

    def node(number, kind, title, pos, size, widgets, outputs=(), inputs=()):
        value = dict(
            id=number, type=kind, title=title, pos=list(pos), size=list(size),
            flags={}, order=len(nodes), mode=0,
            inputs=[dict(name=name, type=typ, link=None, **extra) for name, typ, extra in inputs],
            outputs=[dict(name=name, type=typ, links=[], slot_index=i) for i, (name, typ) in enumerate(outputs)],
            properties={"Node name for S&R": kind}, widgets_values=widgets,
        )
        nodes.append(value)
        return value

    def connect(source, slot, target, name, kind):
        target_slot = next(i for i, value in enumerate(target["inputs"]) if value["name"] == name)
        number = len(links) + 1
        links.append([number, source["id"], slot, target["id"], target_slot, kind])
        source["outputs"][slot]["links"].append(number)
        target["inputs"][target_slot]["link"] = number

    image = node(1, "LoadImage", "1 · Select input image", (40, 40), (340, 310), ["example.png", "image"], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
    dims = node(2, "GetImageSize", "Image dimensions", (40, 390), (280, 90), [], [("width", "INT"), ("height", "INT"), ("batch_size", "INT")], [("image", "IMAGE", {})])
    connect(image, 0, dims, "image", "IMAGE")
    model = node(3, "UNETLoader", "FLUX.1-dev · base model", (430, 40), (340, 90), ["FLUX\\flux1_dev.safetensors", "default"], [("MODEL", "MODEL")])
    clip = node(4, "DualCLIPLoader", "FLUX · CLIP-L + T5XXL", (430, 170), (340, 150), ["clip_l.safetensors", "t5xxl_fp16.safetensors", "flux", "cpu"], [("CLIP", "CLIP")])
    vae = node(5, "VAELoader", "FLUX · VAE", (430, 360), (340, 65), ["ae.safetensors"], [("VAE", "VAE")])
    boxes = node(6, "CreateBoundingBoxes", "2 · Review / draw object boxes", (850, 40), (460, 640), [1024, 1024, [], []], [("preview", "IMAGE"), ("bboxes", "BOUNDING_BOX"), ("elements", "ARRAY")], [
        ("width", "INT", {"widget": {"name": "width"}}),
        ("height", "INT", {"widget": {"name": "height"}}),
        ("background", "IMAGE", {"shape": 7}),
        ("bboxes", "BOUNDING_BOX,ARRAY,STRING", {"shape": 7}),
    ])
    boxes["widgets_values_named"] = dict(width=1024, height=1024, editor_state=[], last_incoming=[])
    connect(dims, 0, boxes, "width", "INT")
    connect(dims, 1, boxes, "height", "INT")
    connect(image, 0, boxes, "background", "IMAGE")
    reveal = node(7, "TBGRevealLayer", "TBG RevealLayer · occlusion-aware image decomposition", (1390, 40), (460, 300), [30, 1.0, 43, 12, True], [
        ("foreground_rgba", "IMAGE"), ("background_rgba", "IMAGE"),
        ("object_layers_rgba", "IMAGE"), ("object_masks", "MASK"),
        ("all_layers_rgba", "IMAGE"), ("composite_rgba", "IMAGE"),
    ], [("model", "MODEL", {}), ("clip", "CLIP", {}), ("vae", "VAE", {}), ("image", "IMAGE", {}), ("bounding_boxes", "BOUNDING_BOX", {"shape": 7})])
    for source, name, kind in [(model, "model", "MODEL"), (clip, "clip", "CLIP"), (vae, "vae", "VAE"), (image, "image", "IMAGE")]:
        connect(source, 0, reveal, name, kind)
    connect(boxes, 1, reveal, "bounding_boxes", "BOUNDING_BOX")
    prefix = "TBG_RevealLayer_grounded" if grounded else "TBG_RevealLayer_manual"
    layers = node(8, "SaveImage", "RGBA layers", (1930, 40), (360, 300), [prefix + "/layer"], [], [("images", "IMAGE", {})])
    connect(reveal, 4, layers, "images", "IMAGE")
    composite = node(9, "SaveImage", "Composite", (1930, 380), (360, 300), [prefix + "/composite"], [], [("images", "IMAGE", {})])
    connect(reveal, 5, composite, "images", "IMAGE")
    masks = node(10, "MaskToImage", "Object alpha masks", (1390, 400), (220, 60), [], [("IMAGE", "IMAGE")], [("mask", "MASK", {})])
    connect(reveal, 3, masks, "mask", "MASK")
    preview = node(11, "PreviewImage", "Object mask preview", (1640, 430), (240, 250), [], [], [("images", "IMAGE", {})])
    connect(masks, 0, preview, "images", "IMAGE")
    box_preview = node(12, "PreviewImage", "Box preview", (850, 730), (460, 250), [], [], [("images", "IMAGE", {})])
    connect(boxes, 0, box_preview, "images", "IMAGE")
    guide = (ROOT / "workflows" / "README.md").read_text(encoding="utf-8")
    heading = "# This is the OPTIONAL GROUNDED workflow\n\nUse your existing SAM3.1 checkpoint. Edit the grounding text (`person:2` by default).\n\n" if grounded else "# This is the MANUAL BOX workflow\n\nDraw boxes in Create Bounding Boxes. No grounding model is loaded.\n\n"
    node(13, "MarkdownNote", "Instructions · model download links", (40, 1100), (1000, 950), [heading + guide])
    if grounded:
        sam = node(14, "CheckpointLoaderSimple", "Optional · existing SAM3.1 checkpoint", (40, 560), (340, 100), ["sam3.1_multiplex_fp16.safetensors"], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
        text = node(15, "CLIPTextEncode", "Grounding text · category:count", (430, 560), (340, 150), ["person:2"], [("CONDITIONING", "CONDITIONING")], [("clip", "CLIP", {})])
        connect(sam, 1, text, "clip", "CLIP")
        detect = node(16, "SAM3_Detect", "Optional · text to pixel boxes", (430, 760), (340, 150), [0.35, 0, False], [("masks", "MASK"), ("bboxes", "BOUNDING_BOX")], [("model", "MODEL", {}), ("image", "IMAGE", {}), ("conditioning", "CONDITIONING", {"shape": 7})])
        connect(sam, 0, detect, "model", "MODEL")
        connect(image, 0, detect, "image", "IMAGE")
        connect(text, 0, detect, "conditioning", "CONDITIONING")
        connect(detect, 1, boxes, "bboxes", "BOUNDING_BOX")
    if refinement:
        if not grounded:
            raise ValueError("The refinement workflow requires the SAM3 branch.")
        sam_extract = node(17, "SAM3_Detect", "SAM3 · segment the final edited boxes", (2450, 40), (340, 140), [0.35, 2, True], [("masks", "MASK"), ("bboxes", "BOUNDING_BOX")], [("model", "MODEL", {}), ("image", "IMAGE", {}), ("bboxes", "BOUNDING_BOX", {"shape": 7})])
        connect(sam, 0, sam_extract, "model", "MODEL")
        connect(image, 0, sam_extract, "image", "IMAGE")
        connect(boxes, 1, sam_extract, "bboxes", "BOUNDING_BOX")
        refs = node(18, "TBGRevealLayerReferences", "Prepare one aligned edit per layer", (2450, 240), (440, 350), [1024, "Improve fine detail, texture, and edge clarity while preserving identity, pose, geometry, colors, lighting, and composition."], [("layer_reference", "IMAGE"), ("original_reference", "IMAGE"), ("sam_reference", "IMAGE"), ("edit_instruction", "STRING"), ("layout_table", "STRING"), ("upsampler_system", "STRING"), ("plan", "TBG_REFINEMENT_PLAN"), ("comparison_batch", "IMAGE"), ("sam_layers_rgba", "IMAGE")], [("original_image", "IMAGE", {}), ("reveallayer_layers", "IMAGE", {}), ("sam_masks", "MASK", {}), ("bounding_boxes", "BOUNDING_BOX", {})])
        for output in refs["outputs"][:5]:
            output["shape"] = 6
        connect(image, 0, refs, "original_image", "IMAGE")
        connect(reveal, 4, refs, "reveallayer_layers", "IMAGE")
        connect(sam_extract, 0, refs, "sam_masks", "MASK")
        connect(boxes, 1, refs, "bounding_boxes", "BOUNDING_BOX")
        sam_save = node(19, "SaveImage", "Compare · SAM3 RGBA layers", (2450, 650), (340, 250), ["TBG_RevealLayer_refined/sam_layer"], [], [("images", "IMAGE", {})])
        connect(refs, 8, sam_save, "images", "IMAGE")
        compare = node(20, "PreviewImage", "Per layer: original → SAM3 → RevealLayer", (2850, 650), (430, 300), [], [], [("images", "IMAGE", {})])
        connect(refs, 7, compare, "images", "IMAGE")
        edit_model = node(21, "UNETLoader", "FLUX 3 Image Klein · distilled", (3000, 40), (380, 90), ["FLUX\\flux3-image-klein-bf16.safetensors", "default"], [("MODEL", "MODEL")])
        edit_clip = node(22, "CLIPLoader", "FLUX 3 · Qwen3-VL 4B text encoder", (3000, 180), (380, 110), ["Flux3\\qwen3vl_4b_int8_convrot.safetensors", "flux3", "default"], [("CLIP", "CLIP")])
        edit_vae = node(23, "VAELoader", "FLUX 3 · FLUX2 VAE", (3000, 340), (380, 65), ["flux2-vae.safetensors"], [("VAE", "VAE")])
        pup = node(24, "CLIPLoader", "FLUX 3 · existing prompt upsampler", (3450, 40), (400, 110), ["Flux3\\qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors", "flux3", "default"], [("CLIP", "CLIP")])
        expand = node(25, "TextGenerateV2", "Expand each role-specific edit using all 3 references", (3450, 210), (430, 260), ["", 1536, "off", False, True, "off"], [("generated_text", "STRING"), ("thinking", "STRING")], [("clip", "CLIP", {}), ("prompt", "STRING", {"widget": {"name": "prompt"}}), ("images.image1", "IMAGE", {"shape": 7}), ("images.image2", "IMAGE", {"shape": 7}), ("images.image3", "IMAGE", {"shape": 7}), ("system_prompt", "STRING", {"shape": 7})])
        connect(pup, 0, expand, "clip", "CLIP")
        connect(refs, 3, expand, "prompt", "STRING")
        connect(refs, 0, expand, "images.image1", "IMAGE")
        connect(refs, 1, expand, "images.image2", "IMAGE")
        connect(refs, 2, expand, "images.image3", "IMAGE")
        connect(refs, 5, expand, "system_prompt", "STRING")
        prompt = node(26, "StringConcatenate", "Append the fixed edit-layout table after expansion", (3930, 40), (390, 150), ["", "", " "], [("STRING", "STRING")], [("string_a", "STRING", {"widget": {"name": "string_a"}}), ("string_b", "STRING", {"widget": {"name": "string_b"}})])
        connect(expand, 0, prompt, "string_a", "STRING")
        connect(refs, 4, prompt, "string_b", "STRING")
        encode = node(27, "CLIPTextEncode", "Encode each expanded layout edit", (3930, 240), (390, 170), [""], [("CONDITIONING", "CONDITIONING")], [("clip", "CLIP", {}), ("text", "STRING", {"widget": {"name": "text"}})])
        connect(edit_clip, 0, encode, "clip", "CLIP")
        connect(prompt, 0, encode, "text", "STRING")
        conditioning = node(28, "ReferenceMultiLatent", "Reference slots 0=Reveal, 1=original, 2=SAM", (3930, 470), (390, 180), ["keep resolution"], [("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("empty_latent", "LATENT")], [("positive", "CONDITIONING", {}), ("vae", "VAE", {"shape": 7}), ("images.image0", "IMAGE", {"shape": 7}), ("images.image1", "IMAGE", {"shape": 7}), ("images.image2", "IMAGE", {"shape": 7})])
        connect(encode, 0, conditioning, "positive", "CONDITIONING")
        connect(edit_vae, 0, conditioning, "vae", "VAE")
        for slot in range(3):
            connect(refs, slot, conditioning, f"images.image{slot}", "IMAGE")
        sampler = node(29, "KSampler", "Native · one FLUX3 edit per layer", (4410, 40), (340, 300), [43, "fixed", 16, 1.0, "euler", "simple", 1.0], [("LATENT", "LATENT")], [("model", "MODEL", {}), ("positive", "CONDITIONING", {}), ("negative", "CONDITIONING", {}), ("latent_image", "LATENT", {})])
        connect(edit_model, 0, sampler, "model", "MODEL")
        connect(conditioning, 0, sampler, "positive", "CONDITIONING")
        connect(conditioning, 1, sampler, "negative", "CONDITIONING")
        connect(conditioning, 2, sampler, "latent_image", "LATENT")
        decode = node(30, "VAEDecode", "Decode each RGB edit", (4410, 420), (260, 75), [], [("IMAGE", "IMAGE")], [("samples", "LATENT", {}), ("vae", "VAE", {})])
        connect(sampler, 0, decode, "samples", "LATENT")
        connect(edit_vae, 0, decode, "vae", "VAE")
        collect = node(31, "TBGRevealLayerCollect", "Restore alpha and layer order", (4830, 40), (340, 150), [], [("refined_layers_rgba", "IMAGE"), ("refined_objects_rgba", "IMAGE"), ("refined_composite_rgba", "IMAGE")], [("refined_images", "IMAGE", {}), ("plan", "TBG_REFINEMENT_PLAN", {})])
        connect(decode, 0, collect, "refined_images", "IMAGE")
        connect(refs, 6, collect, "plan", "TBG_REFINEMENT_PLAN")
        for number, slot, title, name, pos in [(32, 0, "Refined RGBA layers", "layer", (5250, 40)), (33, 2, "Refined composite", "composite", (5250, 420))]:
            save = node(number, "SaveImage", title, pos, (370, 300), ["TBG_RevealLayer_refined/" + name], [], [("images", "IMAGE", {})])
            connect(collect, slot, save, "images", "IMAGE")
        nodes[12]["title"] = "Third workflow · references, bbox layout, native FLUX3 refinement"
        nodes[12]["widgets_values"][0] = "# SAM3 + RevealLayer + FLUX3 per-layer refinement\n\nOne source image per run. The second SAM3 pass segments the final boxes; leave its text conditioning disconnected. Each job uses three separate reference slots, not a three-image target batch. The native sampler iterates one job per foreground, background, and object layer. The collector restores RevealLayer alpha; SAM3 background is visible evidence with holes, not an inpainted background. Select your existing local FLUX3, Qwen3-VL, FLUX2 VAE, and prompt upsampler files. No refinement models are downloaded by this pack.\n\n" + guide
    done = set()
    for order in range(len(nodes)):
        available = next(n for n in nodes if n["id"] not in done and all(link[1] in done for link in links if link[3] == n["id"]))
        available["order"] = order
        done.add(available["id"])
    return dict(last_node_id=max(n["id"] for n in nodes), last_link_id=len(links), nodes=nodes, links=links, groups=[], config={}, extra={"ds": {"scale": 0.65, "offset": [70, 50]}}, version=0.4)


if __name__ == "__main__":
    for grounded in (False, True):
        name = "TBG_RevealLayer_Grounded.json" if grounded else "TBG_RevealLayer_Manual.json"
        (ROOT / "workflows" / name).write_text(json.dumps(build(grounded), indent=2) + "\n", encoding="utf-8", newline="\n")
        print(name)
    name = "TBG_RevealLayer_SAM3_Flux3_Refine.json"
    (ROOT / "workflows" / name).write_text(json.dumps(build(True, refinement=True), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(name)
