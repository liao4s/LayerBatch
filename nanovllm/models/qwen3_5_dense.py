"""
Qwen3.5 Dense model implementation for nanovllm.

Architecture:
- Hybrid attention: 3 linear attention (Gated DeltaNet) + 1 full attention per 4 layers
- Dense MLP: gate_proj + up_proj (merged) + down_proj with SiLU activation
- Full attention has output gating: q_proj outputs 2x dim (query + gate)
- Partial rotary embedding (25% of head_dim)
- Uses (1+w) style RMSNorm (weight initialized to 0, applies as (1+w)*norm(x))

This module reuses the shared attention components from qwen3_5.py (the MoE variant)
and only defines the dense MLP and model assembly.
"""
from __future__ import annotations

import torch
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

# Reuse shared components from qwen3_5 (MoE variant)
from nanovllm.models.qwen3_5 import (
    Qwen3_5RMSNorm,
    Qwen3_5GatedDeltaNet,
    Qwen3_5FullAttention,
)


# ============================================================
# Dense MLP
# ============================================================

class Qwen3_5DenseMLP(nn.Module):
    """Standard dense MLP with gate_proj + up_proj (merged) + down_proj."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


# ============================================================
# Decoder Layer
# ============================================================

class Qwen3_5DenseDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.hidden_size = text_config.hidden_size
        self.layer_type = text_config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=text_config.hidden_size,
                num_k_heads=text_config.linear_num_key_heads,
                num_v_heads=text_config.linear_num_value_heads,
                head_k_dim=text_config.linear_key_head_dim,
                head_v_dim=text_config.linear_value_head_dim,
                conv_kernel_size=text_config.linear_conv_kernel_dim,
                rms_norm_eps=text_config.rms_norm_eps,
            )
        elif self.layer_type == "full_attention":
            rope_params = text_config.rope_parameters
            self.self_attn = Qwen3_5FullAttention(
                hidden_size=text_config.hidden_size,
                num_heads=text_config.num_attention_heads,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                max_position=text_config.max_position_embeddings,
                rms_norm_eps=text_config.rms_norm_eps,
                rope_theta=rope_params["rope_theta"],
                partial_rotary_factor=rope_params.get("partial_rotary_factor", 0.25),
            )

        # Dense MLP instead of MoE
        self.mlp = Qwen3_5DenseMLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
        )

        # NOTE: Qwen3.5 uses (1+w) style RMSNorm with weights initialized to 0
        self.input_layernorm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Same residual connection pattern as Qwen3.5 MoE variant
        if residual is not None:
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Token mixing
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ============================================================
# Text Model (Language Model)
# ============================================================

class Qwen3_5DenseTextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.embed_tokens = VocabParallelEmbedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DenseDecoderLayer(config, i)
            for i in range(text_config.num_hidden_layers)
        ])
        self.norm = Qwen3_5RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # After last layer, need to add final residual
        hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================
# Top-level: Qwen3.5 Dense For Causal LM
# ============================================================

class Qwen3_5DenseForCausalLM(nn.Module):

    # Pack gate_proj + up_proj into gate_up_proj for TP efficiency
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Weight name prefix to strip (VLM structure)
    weight_prefix = "model."

    # Prefixes to skip during weight loading
    skip_prefixes = ("mtp.", "model.visual.", "model.merger.")

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.language_model = Qwen3_5DenseTextModel(config)
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.language_model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
