"""RevealLayer's checkpoint-compatible feature refiner using ComfyUI attention."""

import torch
import torch.nn as nn

import comfy.ops
from comfy.ldm.flux.math import attention
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class FluxRefinerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        operations = comfy.ops.manual_cast
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.comfy_attention = ComfyAttention()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ada_regressor = nn.Sequential(nn.SiLU(), operations.Linear(hidden_size, 6 * hidden_size))
        self.to_qkv = operations.Linear(hidden_size, hidden_size * 3, bias=True)
        self.to_out = operations.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            operations.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            operations.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, x, t_emb, pe):
        batch, sequence, _ = x.shape
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_regressor(t_emb).unsqueeze(1).chunk(6, dim=-1)
        qkv = self.to_qkv(_modulate(self.norm1(x), shift_attn, scale_attn))
        q, k, v = qkv.reshape(batch, sequence, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attended = attention(
            AttentionTensorContainer(q),
            AttentionTensorContainer(k),
            AttentionTensorContainer(v),
            pe=pe,
            preferred_attention=self.comfy_attention,
        )
        x = x + gate_attn * self.to_out(attended.reshape(batch, sequence, self.hidden_size))
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class HybridRefiner(nn.Module):
    def __init__(self, feature_dim=3072, attn_hidden=1024, num_layers=1, max_layers=32):
        super().__init__()
        operations = comfy.ops.manual_cast
        self.attn_hidden = attn_hidden
        self.input_proj = operations.Linear(feature_dim + 128, attn_hidden)
        self.time_proj = nn.Sequential(nn.SiLU(), operations.Linear(feature_dim, attn_hidden))
        self.layer_id_emb = operations.Embedding(max_layers, attn_hidden)
        self.blocks = nn.ModuleList([FluxRefinerBlock(attn_hidden, num_heads=8) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(attn_hidden, elementwise_affine=False, eps=1e-6)
        self.final_linear = operations.Linear(attn_hidden, feature_dim)

    def forward_feature_attn(self, hidden_states, clean_tokens, split_sizes, temb, pe):
        compute_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(compute_dtype)
        clean_tokens = clean_tokens.to(compute_dtype)
        temb = temb.to(compute_dtype)
        time_base = self.time_proj(temb).to(compute_dtype)
        x_splits = torch.split(hidden_states, split_sizes, dim=1)
        clean_splits = torch.split(clean_tokens, split_sizes, dim=1)
        pe_splits = torch.split(pe, split_sizes, dim=2)
        refined = []

        for index, (x, clean, layer_pe) in enumerate(zip(x_splits, clean_splits, pe_splits)):
            layer_id = torch.tensor([index + 1], device=hidden_states.device)
            layer_condition = time_base + self.layer_id_emb(layer_id).to(compute_dtype)
            current = self.input_proj(torch.cat((x, clean.to(compute_dtype)), dim=-1))
            for block in self.blocks:
                current = block(current, t_emb=layer_condition, pe=layer_pe)
            refined.append(current)

        output = self.final_norm(torch.cat(refined, dim=1))
        return self.final_linear(output)
