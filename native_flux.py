from functools import partial

import torch

import comfy.patcher_extension
from comfy.ldm.flux.layers import timestep_embedding


def _crop_layer_tokens(values, boxes, factor):
    tokens = []
    for layer, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = (coordinate // factor for coordinate in box)
        tokens.append(values[:, layer, y1:y2, x1:x2].reshape(values.shape[0], -1, values.shape[-1]))
    return torch.cat(tokens, dim=1)


def _fill_layer_tokens(tokens, values, boxes, factor):
    offset = 0
    for layer, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = (coordinate // factor for coordinate in box)
        count = (y2 - y1) * (x2 - x1)
        values[:, layer, y1:y2, x1:x2] = tokens[:, offset:offset + count].reshape(
            tokens.shape[0], y2 - y1, x2 - x1, tokens.shape[-1]
        )
        offset += count
    return values


def _build_attention_mask(boxes, latent_height, latent_width, text_length, device, dtype):
    positions = []
    layers = []
    rois = []
    for layer, box in enumerate(boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        left = max(0, x1 // 16)
        top = max(0, y1 // 16)
        right = min(latent_width, x2 // 16)
        bottom = min(latent_height, y2 // 16)
        for y in range(top, bottom):
            for x in range(left, right):
                positions.append((x, y))
                layers.append(layer)
                rois.append((left, top, right, bottom))

    image_length = len(positions)
    total_length = text_length + image_length
    mask = torch.full(
        (total_length, total_length), torch.finfo(dtype).min, device=device, dtype=dtype
    )
    text_slice = slice(0, text_length)
    image_slice = slice(text_length, total_length)
    mask[text_slice, :] = 0
    mask[image_slice, text_slice] = 0
    if image_length == 0:
        return mask

    positions = torch.tensor(positions, device=device)
    layers = torch.tensor(layers, device=device)
    rois = torch.tensor(rois, device=device)
    image_mask = mask[image_slice, image_slice]

    same_layer = layers[:, None] == layers[None, :]
    image_mask[same_layer] = 0

    background_sees_other_layers = (layers[:, None] == 1) & (layers[None, :] != 1)
    image_mask[background_sees_other_layers] = 0

    foreground_sees_full_layer = layers[:, None] >= 2
    full_layer_key = layers[None, :] == 0
    x_inside_roi = (positions[None, :, 0] >= rois[:, None, 0]) & (positions[None, :, 0] < rois[:, None, 2])
    y_inside_roi = (positions[None, :, 1] >= rois[:, None, 1]) & (positions[None, :, 1] < rois[:, None, 3])
    image_mask[foreground_sees_full_layer & full_layer_key & x_inside_roi & y_inside_roi] = 0
    return mask


def forward_reveallayer(model, *args, transformer_options=None, **kwargs):
    if transformer_options is None:
        transformer_options = {}
    return comfy.patcher_extension.WrapperExecutor.new_class_executor(
        partial(_forward_reveallayer, model),
        model,
        comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options
        ),
    ).execute(*args, transformer_options=transformer_options, **kwargs)


def _forward_reveallayer(
    model,
    hidden_states,
    adapter_data,
    split_sizes,
    list_layer_box,
    encoder_hidden_states,
    pooled_projections,
    timestep,
    img_ids,
    txt_ids,
    layer_pe,
    refiner,
    guidance=None,
    transformer_options=None,
):
    transformer_options = (transformer_options or {}).copy()
    transformer_options["patches"] = transformer_options.get("patches", {}).copy()
    transformer_options["patches_replace"] = transformer_options.get("patches_replace", {}).copy()

    batch, layer_count, channels, height, width = hidden_states.shape
    patch_size = model.patch_size
    if channels != 16 or patch_size != 2 or height % 2 or width % 2:
        raise ValueError("RevealLayer requires 16-channel FLUX latents with dimensions divisible by its 2x2 patches.")

    hidden_states = hidden_states.reshape(batch, layer_count, channels, height // 2, 2, width // 2, 2)
    hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6)
    hidden_states = hidden_states.reshape(batch, layer_count, height // 2, width // 2, channels * 4)
    img_grid = model.img_in(hidden_states)
    layer_pe = layer_pe.to(device=img_grid.device, dtype=img_grid.dtype)
    img_grid = img_grid + layer_pe[:, :layer_count]
    full_hidden_states = torch.zeros_like(img_grid)

    img = _crop_layer_tokens(img_grid, list_layer_box, 16)
    if img_ids.ndim == 2:
        img_ids = img_ids.unsqueeze(0).expand(batch, -1, -1)
    if txt_ids.ndim == 2:
        txt_ids = txt_ids.unsqueeze(0).expand(batch, -1, -1)
    img_ids = img_ids.to(device=img.device, dtype=torch.float32)
    txt_ids = txt_ids.to(device=img.device, dtype=torch.float32)
    timestep = timestep.to(device=img.device, dtype=torch.float32)
    vec = model.time_in(timestep_embedding(timestep, 256).to(img.dtype))
    if model.params.guidance_embed and guidance is not None:
        guidance = guidance.to(device=img.device, dtype=torch.float32)
        vec = vec + model.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
    if model.vector_in is not None:
        vec = vec + model.vector_in(pooled_projections[:, :model.params.vec_in_dim])
    if model.txt_norm is not None:
        encoder_hidden_states = model.txt_norm(encoder_hidden_states)
    txt = model.txt_in(encoder_hidden_states)

    if "post_input" in transformer_options["patches"]:
        for patch in transformer_options["patches"]["post_input"]:
            output = patch({
                "img": img,
                "txt": txt,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
                "transformer_options": transformer_options,
            })
            img = output["img"]
            txt = output["txt"]
            img_ids = output["img_ids"]
            txt_ids = output["txt_ids"]

    pe = model.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

    vec_orig = vec
    if model.params.global_modulation:
        vec = (model.double_stream_modulation_img(vec_orig), model.double_stream_modulation_txt(vec_orig))

    block_replacements = transformer_options["patches_replace"].get("dit", {})
    attention_mask = _build_attention_mask(
        list_layer_box, height // 2, width // 2, txt.shape[1], img.device, img.dtype
    )
    transformer_options["total_blocks"] = len(model.double_blocks)
    transformer_options["block_type"] = "double"
    for index, block in enumerate(model.double_blocks):
        transformer_options["block_index"] = index
        replacement = block_replacements.get(("double_block", index))
        if replacement is None:
            img, txt = block(
                img=img, txt=txt, vec=vec, pe=pe,
                attn_mask=attention_mask,
                transformer_options=transformer_options,
            )
        else:
            def block_wrap(args):
                out_img, out_txt = block(
                    img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"],
                    attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"),
                )
                return {"img": out_img, "txt": out_txt}

            output = replacement(
                {"img": img, "txt": txt, "vec": vec, "pe": pe,
                 "attn_mask": attention_mask,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            img, txt = output["img"], output["txt"]

    text_length = txt.shape[1]
    full_layer_length = split_sizes[0]
    if refiner is not None:
        clean_tokens = _crop_layer_tokens(adapter_data, list_layer_box[1:], 16)
        refiner_pe = model.pe_embedder(img_ids[:, full_layer_length:, :])
        delta = refiner.forward_feature_attn(
            hidden_states=img[:, full_layer_length:, :],
            clean_tokens=clean_tokens,
            split_sizes=split_sizes,
            temb=vec_orig,
            pe=refiner_pe,
        )
        img = torch.cat((img[:, :full_layer_length], img[:, full_layer_length:] + delta), dim=1)
    combined = torch.cat((txt, img), dim=1)

    if model.params.global_modulation:
        single_vec, _ = model.single_stream_modulation(vec_orig)
    else:
        single_vec = vec_orig
    transformer_options["total_blocks"] = len(model.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [text_length, combined.shape[1]]
    for index, block in enumerate(model.single_blocks):
        transformer_options["block_index"] = index
        replacement = block_replacements.get(("single_block", index))
        if replacement is None:
            combined = block(
                combined, vec=single_vec, pe=pe,
                attn_mask=attention_mask,
                transformer_options=transformer_options,
            )
        else:
            def block_wrap(args):
                out_img = block(
                    args["img"], vec=args["vec"], pe=args["pe"],
                    attn_mask=args.get("attn_mask"), transformer_options=args.get("transformer_options"),
                )
                return {"img": out_img}

            output = replacement(
                {"img": combined, "vec": single_vec, "pe": pe,
                 "attn_mask": attention_mask,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )
            combined = output["img"]

    image_tokens = combined[:, text_length:]
    image_tokens = _fill_layer_tokens(image_tokens, full_hidden_states, list_layer_box, 16)
    image_tokens = image_tokens.reshape(batch, -1, model.hidden_size)
    image_tokens = model.final_layer(image_tokens, vec_orig)
    image_tokens = image_tokens.reshape(batch, layer_count, height // 2, width // 2, channels, 2, 2)
    image_tokens = image_tokens.permute(0, 1, 4, 2, 5, 3, 6)
    return image_tokens.reshape(batch, layer_count, channels, height, width)
