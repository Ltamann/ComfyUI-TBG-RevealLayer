import einops
from collections import OrderedDict
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torchvision

from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1

class MLPBlock(torchvision.ops.misc.MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, freqs_cis):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        B, L, C = input.shape
        x = self.ln_1(input)
        if freqs_cis is not None:
            query = x.view(B, L, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
            query = apply_rope1(query, freqs_cis)
            query = query.transpose(1, 2).reshape(B, L, self.hidden_dim)
        x, _ = self.self_attention(query, query, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, freqs_cis):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input # + self.pos_embedding
        x = self.dropout(input)
        for l in self.layers:
            x = l(x, freqs_cis)
        x = self.ln(x)
        return x


class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 768
        layers = 12
        heads = 12

        self.encoder = Encoder(
            seq_length=-1,
            num_layers=layers,
            num_heads=heads,
            hidden_dim=ch,
            mlp_dim=ch*4,
            dropout=0.0,
            attention_dropout=0.0,
        )
        self.fc_in = nn.Linear(16, ch)
        self.fc_out = nn.Linear(ch, 256)

    def forward(self, x, freqs_cis):
        out = self.fc_in(x)
        out = self.encoder(out, freqs_cis)
        out = self.fc_out(out)
        return out


def unpatchify(x, patch_size=8):
    return einops.rearrange(x, "(c p1 p2) h w -> c (h p1) (w p2)", p1=patch_size, p2=patch_size)


def crop_each_layer(hidden_states, use_layers, list_layer_box, H, W, pos_embedding):
    token_list = []
    pe_list = []
    for layer_idx in range(hidden_states.shape[1]):
        if list_layer_box[layer_idx] is None:
            continue
        else:
            x1, y1, x2, y2 = list_layer_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            layer_token = hidden_states[:, layer_idx, y1:y2, x1:x2]
            c, h, w = layer_token.shape
            layer_token = layer_token.reshape(c, -1)
            token_list.append(layer_token)
            ids = prepare_latent_image_ids(H * 2, W * 2, hidden_states.device, hidden_states.dtype)
            ids[:, 0] = use_layers[layer_idx]
            pe = pos_embedding(ids.unsqueeze(0)).reshape(1, 1, H, W, 32, 2, 2)
            pe_list.append(pe[:, :, y1:y2, x1:x2].reshape(1, 1, -1, 32, 2, 2))
    token_list = torch.cat(token_list, dim=1).permute(1, 0)
    return token_list, torch.cat(pe_list, dim=2)


def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


class AutoencoderKLTransformerTraining(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = ViTEncoder()
        self.pos_embedding = EmbedND(dim=64, theta=10000, axes_dim=[8, 28, 28])

    def encode(self, z_2d, box, use_layers):
        B, C, T, H, W = z_2d.shape

        z, freqs_cis = [], []
        for b in range(B):
            _z = z_2d[b]
            _z, cis = crop_each_layer(_z, use_layers[b], box[b], H, W, self.pos_embedding)
            z.append(_z)
            freqs_cis.append(cis)

        return z, freqs_cis

    def decode(self, z, freqs_cis, box, H, W):
        B = len(z)
        pad = torch.zeros(4, H, W, device=z[0].device, dtype=z[0].dtype)
        pad[3, :, :] = -1
        x = []
        for b in range(B):
            _x = []
            _z = self.decoder(z[b].unsqueeze(0), freqs_cis[b]).squeeze(0)
            current_index = 0
            for layer_idx in range(len(box[b])):
                if box[b][layer_idx] == None:
                    _x.append(pad.clone())
                else:
                    x1, y1, x2, y2 = box[b][layer_idx]
                    x1_tok, y1_tok, x2_tok, y2_tok = x1 // 8, y1 // 8, x2 // 8, y2 // 8
                    token_length = (x2_tok - x1_tok) * (y2_tok - y1_tok)
                    tokens = _z[current_index:current_index + token_length]
                    pixels = einops.rearrange(tokens, "(h w) c -> c h w", h=y2_tok - y1_tok, w=x2_tok - x1_tok)
                    unpatched = unpatchify(pixels)
                    pixels = pad.clone()
                    pixels[:, y1:y2, x1:x2] = unpatched
                    _x.append(pixels)
                    current_index += token_length
            _x = torch.stack(_x, dim=1)
            x.append(_x)
        x = torch.stack(x, dim=0)
        return x

    def forward(self, z_2d, box, use_layers=None):
        z_2d = z_2d.transpose(0, 1).unsqueeze(0)
        use_layers = use_layers or [list(range(z_2d.shape[2]))]
        z, freqs_cis = self.encode(z_2d, box, use_layers)
        H, W = z_2d.shape[-2:]
        x_hat = self.decode(z, freqs_cis, box, H * 8, W * 8)
        assert x_hat.shape[0] == 1, x_hat.shape
        x_hat = einops.rearrange(x_hat[0], "c t h w -> t c h w")
        x_hat_rgb, x_hat_alpha = x_hat[:, :3], x_hat[:, 3:]
        return x_hat_rgb, x_hat_alpha
