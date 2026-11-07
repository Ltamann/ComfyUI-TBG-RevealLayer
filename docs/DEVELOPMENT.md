# TBG RevealLayer development notes

## 0.2.0 — 2026-11-06

User-confirmed package direction: **TBG RevealLayer — occlusion-aware image decomposition.** One minimal decomposition node; optional native grounding prepass; automatic RevealLayer assets inside the custom-node folder; no automatic grounding-model downloads.

### Interface and setup

- Replaced `RevealLayerDecompose` with `TBGRevealLayer`. Old coordinate strings, format selector, and model path widgets were removed deliberately. No legacy alias is retained.
- `bounding_boxes` is the only optional socket. It accepts pixel dictionaries from Create Bounding Boxes or native detection nodes. The hidden `unique_id` is used only for progress.
- One box frame broadcasts to the image batch; a matching frame batch is processed per image. Invalid batch counts and coordinates are rejected before setup or inference. Extra object boxes are limited to `max_layers - 2`.
- First execution checks `requirements.txt`; missing requirements are installed with ComfyUI's Python. Installation requests one clean restart. Importing the package performs no installation or model download. PyTorch/CUDA are not listed as extension install targets.
- `model_downloads.py` allows only four pinned public `qihoo360/RevealLayer` artifacts. It verifies size/SHA-256, uses a local file lock, resumes interrupted transfers, and replaces the final file only after verification. Download progress and cancellation use ComfyUI's native mechanisms.
- Existing LoRA, embeddings, Refiner, and transparent decoder files were moved from the obsolete upstream clone to `models/reveallayer/`. Each file was hashed before and after the move. All four match the pinned manifest and a test with network access blocked verified their reuse without requests.

### Optional grounding workflow

The installed ComfyUI already exposes native `SAM3_Detect`, and its checkpoint list includes the existing shared `sam3.1_multiplex_fp16.safetensors`. The source detection config recognizes SAM3.1 and the checkpoint contains its text encoder. Reusing those native nodes avoids another custom detector, third-party node dependency, or new model download. Existing raw Grounding DINO files were found in the shared models folder, but they are not needed or modified by this release.

`TBG_RevealLayer_Grounded.json` routes Checkpoint Loader → CLIP Text Encode → SAM3 Detect → Create Bounding Boxes → TBG RevealLayer. Native SAM3 text syntax supports `category:count`; the default `person:2` requests up to two people. FLUX and SAM3 use separate CLIP paths. Mask refinement is disabled for the box prepass.

`TBG_RevealLayer_Manual.json` contains no grounding loader/detector. Both graphs connect Get Image Size to the canvas dimensions, and both include a MarkdownNote containing the full workflow guide and model download links. The MarkdownNote frontend type and native node schemas were checked against the installed frontend/server.

Actual native prepass validation through the running ComfyUI API succeeded: the supplied image and existing SAM3.1 checkpoint with `person:2` produced two boxes. The native Create Bounding Boxes output reported matching input dimensions and two accepted incoming pixel boxes. The execution took approximately 4.6 seconds. No new grounding weights were downloaded. For direct API prompts, literal custom-widget arrays must use ComfyUI's `{"__value__": []}` wrapper; normal UI workflow serialization handles this.

### Retained quality corrections and research

The [RevealLayer pipeline](https://github.com/360CVGroup/RevealLayer/blob/main/models/custom_pipeline.py) reverses FLUX normalization before calling the transparent decoder. The earlier integration omitted this conversion, causing a washed-out grid independently of sampling steps. The corrected handoff is `raw = normalized / 0.3611 + 0.1159`, applied to all layers on the decoder's device/dtype. The regression failed before the fix and passes afterward. Decoder full loading through ComfyUI's manager prevents ordinary PyTorch weights remaining on CPU during CUDA execution.

With the actual VAE encoder and transparent decoder checkpoints, a CPU/FP32 experiment bypassing sampling reproduced the old defect and improved input reconstruction:

| Working dimensions | Old decoder input PSNR | Corrected decoder input PSNR |
|---|---:|---:|
| 256 × 112 | 16.42 dB | 26.87 dB |
| 1024 × 432 | 18.82 dB | 37.09 dB |

This measures reconstruction against the same resized input, not object extraction quality. The local LoRA, Refiner, and decoder hashes matched official Hugging Face LFS metadata. Diagnostic images containing user source content are excluded from source control; `tests/audit_decoder.py` can reproduce the experiment with caller-supplied paths.

The upstream pipeline pads T5 text to 512 tokens; ComfyUI's default FLUX minimum is 256. The node requests 512 for each encode call without changing the shared CLIP object. Tests check that option and both conditioning lengths; they do not run T5 inference.

Section 3.2 of the [paper](https://arxiv.org/html/2605.11818v1#S3.SS2) states a fixed training instruction. The original inference script uses the same instruction. The current node retains it. Arbitrary instructions/per-layer text control are not established capabilities; optional grounding interprets text separately and provides boxes. Box metadata text is ignored.

### Release checks and limits

Tests cover schema removal, optional/native pixel boxes, batch routing, truncation, decoder latent range, rotary-position equivalence, T5-length requests, automatic dependency setup, hash validation, atomic replacement, resume/retry and cancellation, graph links/types, embedded Markdown instructions, optional grounding isolation, and reproducible workflow generation.

Final checks: **23 tests pass**. The manual workflow has 13 nodes and 14 links; the optional grounded workflow has 16 nodes and 19 links. Every native socket name/type was compared with the running ComfyUI server's schemas. Both workflows contain the exact Markdown guide with model links, and neither contains absolute machine paths. A real HTTPS download of the pinned 75,312-byte layer-embedding file passed its SHA-256 check. The cleanup script parsed successfully; deletion was not executed. Git whitespace checks pass after text line-ending normalization. The source Git manifest excludes checkpoints, old clones/environments, logs, diagnostics, caches, and the unused logo.

The full post-change FLUX GPU decomposition workflow has not been rerun in the already-running service, which still holds the previous Python node modules. Restart and load an updated workflow before evaluating generated object boundaries and background completion. A prior standalone `comfy.sd` import encountered an existing Transformers/huggingface-hub version conflict in the embedded environment; this task does not change ComfyUI's native dependencies to repair it. The native grounding test above ran successfully in the current service.

All implementation and cleanup are confined to this custom-node folder. The shared model folder and ComfyUI core source remain unchanged. Model downloads are local ignored runtime data; no weights, dependency environments, user images, logs, or generated caches belong in the source release.

Automatic execution review rejected recursive removal of obsolete directories and temporary results, including a separate explicit deletion of the verified temporary directory, with the reason "blocked by policy". Physical deletion is therefore not claimed. Obsolete source documents/workflows were removed through reviewed file patches. Remaining old clone, worker environment, unused ViT checkpoint, diagnostics, logs, caches, and unused logo are excluded from Git. `scripts/cleanup_legacy.ps1` provides an explicit allowlist cleanup after verifying the four preserved model files and checking every target remains inside this folder. Run it locally to complete physical disk cleanup; use `-WhatIf` to inspect the targets first.
