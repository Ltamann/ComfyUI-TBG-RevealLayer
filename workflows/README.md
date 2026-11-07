# TBG RevealLayer

*occlusion-aware image decomposition.*

## Choose a workflow

- **TBG_RevealLayer_Manual.json**: draw pixel boxes with native Create Bounding Boxes. No grounding model is loaded.
- **TBG_RevealLayer_Grounded.json**: an optional native SAM3 prepass turns text into pixel boxes and sends them through Create Bounding Boxes for review and edits. Reuses an existing local SAM3 checkpoint; this pack does not download grounding weights.

Both workflows contain this guide as a **MarkdownNote** node.

## Quick start

1. Select your image and installed FLUX.1-dev, CLIP-L + T5XXL (type flux), and FLUX VAE in the native loaders.
2. Manual workflow: draw one box around each object. Get Image Size supplies the editor dimensions, so boxes use original-image pixels.
3. Grounded workflow: select an existing SAM3.1 checkpoint. Set its separate CLIP Text Encode to `person:2` for up to two people. Native SAM3 supports comma-separated categories, for example `person:2, handbag:1`. Its CLIP input must come from the SAM3 checkpoint loader, not the FLUX text encoder.
4. Queue once to generate boxes. Review or edit them and queue again. Native canvas edits take precedence until upstream detections change. If nothing is detected, draw boxes manually.
5. Start with **30 steps**, FLUX guidance **1.0**. RevealLayer's LoRA is applied internally at strength 1.0; do not load it twice.

`max_layers` includes two reserved slots. Three layers allow one object box; four allow two; twelve allow up to ten. Excess boxes are skipped in input order. The returned RGBA layers and composite are saved under `output/TBG_RevealLayer_manual/` or `output/TBG_RevealLayer_grounded/`.

## RevealLayer checkpoints — automatic first use

Missing files (~697 MB in total) are downloaded from [qihoo360/RevealLayer](https://huggingface.co/qihoo360/RevealLayer/tree/main) into this pack's `models/reveallayer/`. Existing verified files are reused. The release pins the upstream revision and checks SHA-256 hashes. Interrupted downloads resume when queued again.

For offline installation, place these files in `models/reveallayer/` inside the node folder:

- [pytorch_lora_weights.safetensors](https://huggingface.co/qihoo360/RevealLayer/resolve/main/pytorch_lora_weights.safetensors)
- [layer_pe.pt](https://huggingface.co/qihoo360/RevealLayer/resolve/main/layer_pe.pt)
- [Refiner.pt](https://huggingface.co/qihoo360/RevealLayer/resolve/main/Refiner.pt)
- [transparent_decoder_ckpt.pth](https://huggingface.co/qihoo360/RevealLayer/resolve/main/xvae/transparent_decoder_ckpt.pth) goes in `models/reveallayer/xvae/`.

## Native model downloads — select existing local files

These separate ComfyUI models are **not automatically downloaded** by this pack. Reuse files in your configured shared model folders, or obtain them under their publishers' terms:

| Model | Download | Native folder |
|---|---|---|
| FLUX.1-dev | [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors) | `models/diffusion_models/` |
| CLIP-L and T5XXL | [FLUX text encoders](https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main) | `models/text_encoders/` or your shared CLIP folder |
| FLUX VAE | [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors) | `models/vae/` |
| Optional SAM3.1 | [sam3.1_multiplex_fp16.safetensors](https://huggingface.co/Comfy-Org/sam3.1/blob/main/checkpoints/sam3.1_multiplex_fp16.safetensors) | `models/checkpoints/` |

The manual workflow has no SAM3 loader or detector and does not require SAM3. The grounded workflow requires a ComfyUI version exposing the native `SAM3_Detect` node. [ComfyUI update instructions](https://docs.comfy.org/installation/update_comfyui).

## First-use Python requirements

Missing extension requirements are installed automatically when TBG RevealLayer executes. If installation was necessary, restart ComfyUI once and queue again. ComfyUI Manager can install `requirements.txt` in advance. PyTorch, CUDA, and native model-loader dependencies are supplied by ComfyUI.

## Credits

[RevealLayer research and source](https://github.com/360CVGroup/RevealLayer): 360 AI Research and Wenzhou University; checkpoints published by qihoo360. Optional native grounding: [SAM3](https://github.com/facebookresearch/sam3) by Meta, packaged for ComfyUI by Comfy-Org. Thank you to the teams for making these tools available. Each model retains its own license and terms.
