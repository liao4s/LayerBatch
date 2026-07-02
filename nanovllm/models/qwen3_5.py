"""
Qwen3.5 MoE model implementation for nanovllm.

Architecture:
- Hybrid attention: 3 linear attention (Gated DeltaNet) + 1 full attention per 4 layers
- MoE MLP: TopK router + fused experts (3D tensors) + shared expert + shared expert gate
- Full attention has output gating: q_proj outputs 2x dim (query + gate)
- Partial rotary embedding (25% of head_dim)
- Uses (1+w) style RMSNorm (weight initialized to 0, applies as (1+w)*norm(x))
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention

from nanovllm.layers.linear import (
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    ColumnParallelLinear,
    ReplicatedLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# ============================================================
# Qwen3.5-style RMSNorm: uses (1 + weight) * norm(x)
# Weight is initialized to zeros, so at init the norm is identity-like.
# This is DIFFERENT from standard Qwen3 RMSNorm which uses weight * norm(x).
# ============================================================

class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling, as used in Qwen3.5."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # NOTE: initialized to zeros! The checkpoint stores zero-initialized weights.
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        # (1 + weight) scaling: when weight=0, this is identity after normalization
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


# ============================================================
# RMSNorm with gating (used by GatedDeltaNet output)
# Also uses (1 + weight) style but additionally applies silu(gate)
# ============================================================

class RMSNormGated(nn.Module):
    """RMSNorm followed by SiLU gating: norm(x) * (1+w) * silu(gate)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ============================================================
# Utility: L2 norm (used in Gated DeltaNet)
# ============================================================

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


# ============================================================
# Torch fallback: chunk-based gated delta rule
# ============================================================

def torch_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_scores = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_scores @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ============================================================
# Torch fallback: recurrent gated delta rule (for single-step decode)
# ============================================================

def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ============================================================
# Gated DeltaNet (Linear Attention)
# ============================================================


class Qwen3_5GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet linear attention layer with tensor parallelism.

    TP Strategy (matching vllm):
    - Heads are sharded across TP ranks (each rank processes num_v_heads/tp_size heads)
    - in_proj_qkv, in_proj_z, in_proj_a/b: ColumnParallelLinear (shard output dim)
    - conv1d: per-rank channels with custom weight loader
    - out_proj: RowParallelLinear (shard input dim, all_reduce output)
    - Prefill: FlashInfer chunk_gated_delta_rule (float32 inter-chunk state)
    - Decode: fused_recurrent_gated_delta_rule (float32 state, L2norm in kernel)
    """

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        tp_rank = dist.get_rank()

        self.hidden_size = hidden_size
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_kernel_size = conv_kernel_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank

        # Total (unsharded) dimensions
        self.total_num_k_heads = num_k_heads
        self.total_num_v_heads = num_v_heads
        self.total_key_dim = num_k_heads * head_k_dim
        self.total_value_dim = num_v_heads * head_v_dim
        self.total_conv_dim = self.total_key_dim * 2 + self.total_value_dim

        # Per-rank head counts
        self.num_k_heads = num_k_heads // tp_size
        self.num_v_heads = num_v_heads // tp_size
        self.key_dim = self.num_k_heads * head_k_dim
        self.value_dim = self.num_v_heads * head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Projections (ColumnParallelLinear with custom weight loaders)
        self.in_proj_qkv = ColumnParallelLinear(hidden_size, self.total_conv_dim, bias=False)
        self.in_proj_qkv.weight.weight_loader = self._qkv_weight_loader

        self.in_proj_z = ColumnParallelLinear(hidden_size, self.total_value_dim, bias=False)
        self.in_proj_a = ColumnParallelLinear(hidden_size, num_v_heads, bias=False)
        self.in_proj_b = ColumnParallelLinear(hidden_size, num_v_heads, bias=False)

        # Conv1d (depthwise, per-rank channels)
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=conv_kernel_size,
            groups=self.conv_dim,
            padding=conv_kernel_size - 1,
        )
        self.conv1d.weight.weight_loader = self._conv1d_weight_loader

        # Per-rank parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.dt_bias.weight_loader = self._per_v_head_weight_loader
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log.weight_loader = self._per_v_head_weight_loader

        # Output norm and projection
        self.norm = RMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.out_proj = RowParallelLinear(self.total_value_dim, hidden_size, bias=False)

        # State buffers (set by model_runner.allocate_linear_attn_states)
        self.recurrent_state_buf: torch.Tensor | None = None
        self.conv_state_buf: torch.Tensor | None = None

    def _qkv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader: checkpoint [Q_all|K_all|V_all] → per-rank [Q_shard|K_shard|V_shard]."""
        q_w = loaded_weight[:self.total_key_dim]
        k_w = loaded_weight[self.total_key_dim:self.total_key_dim * 2]
        v_w = loaded_weight[self.total_key_dim * 2:]
        q_shard = q_w[self.tp_rank * self.key_dim:(self.tp_rank + 1) * self.key_dim]
        k_shard = k_w[self.tp_rank * self.key_dim:(self.tp_rank + 1) * self.key_dim]
        v_shard = v_w[self.tp_rank * self.value_dim:(self.tp_rank + 1) * self.value_dim]
        param.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))

    def _conv1d_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Custom loader: checkpoint [Q_conv|K_conv|V_conv] → per-rank shards."""
        q_c = loaded_weight[:self.total_key_dim]
        k_c = loaded_weight[self.total_key_dim:self.total_key_dim * 2]
        v_c = loaded_weight[self.total_key_dim * 2:]
        q_shard = q_c[self.tp_rank * self.key_dim:(self.tp_rank + 1) * self.key_dim]
        k_shard = k_c[self.tp_rank * self.key_dim:(self.tp_rank + 1) * self.key_dim]
        v_shard = v_c[self.tp_rank * self.value_dim:(self.tp_rank + 1) * self.value_dim]
        param.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))

    def _per_v_head_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """Shard [total_v_heads] → [v_heads_per_rank]."""
        start = self.tp_rank * self.num_v_heads
        param.data.copy_(loaded_weight[start:start + self.num_v_heads])

    def _forward_prefill(
        self,
        hidden_states: torch.Tensor,
        slot_idx: int | None = None,
    ) -> torch.Tensor:
        """Prefill using FlashInfer (float32 state) with Triton fallback."""
        seq_len = hidden_states.shape[0]
        batch_size = 1
        hidden_states_3d = hidden_states.unsqueeze(0)

        # Projections (per-rank outputs)
        mixed_qkv = self.in_proj_qkv(hidden_states_3d)   # [1, seq_len, conv_dim]
        mixed_qkv = mixed_qkv.transpose(1, 2)            # [1, conv_dim, seq_len]
        z = self.in_proj_z(hidden_states_3d)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states_3d)
        a = self.in_proj_a(hidden_states_3d)

        # Save conv state
        if slot_idx is not None and self.conv_state_buf is not None:
            self.conv_state_buf[slot_idx].copy_(
                mixed_qkv[0, :, -(self.conv_kernel_size - 1):]
            )

        # Causal conv1d + SiLU
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)

        # Split Q, K, V (per-rank)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # GQA expand if needed
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        save_state = (slot_idx is not None and self.recurrent_state_buf is not None)

        h0 = torch.zeros(1, self.num_v_heads, self.head_v_dim, self.head_k_dim,
                         dtype=torch.float32, device=hidden_states.device)

        # Use FlashInfer GDN prefill kernel when available (float32 inter-chunk state,
        # no O(T*H*V*K) intermediate tensors — essential for long-context without OOM).
        # Falls back to Triton fla_ops for short sequences or when flashinfer is absent.
        try:
            from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk
            from nanovllm.layers.fla_ops.l2norm import l2norm_fwd

            q_normed = l2norm_fwd(query)
            k_normed = l2norm_fwd(key)

            T = seq_len
            q_3d = q_normed.squeeze(0).contiguous()        # [T, H_v, K]
            k_3d = k_normed.squeeze(0).contiguous()        # [T, H_v, K]
            v_3d = value.squeeze(0).contiguous()            # [T, H_v, V]
            g_2d = g.squeeze(0).contiguous().float()        # [T, H_v]
            beta_2d = beta.squeeze(0).contiguous().float()  # [T, H_v]

            cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=hidden_states.device)

            result = fi_chunk(
                q=q_3d, k=k_3d, v=v_3d,
                g=torch.exp(g_2d), beta=beta_2d,
                initial_state=h0,
                output_final_state=save_state,
                cu_seqlens=cu_seqlens,
            )
            if save_state:
                core_attn_out, last_recurrent_state = result
            else:
                core_attn_out = result
                last_recurrent_state = None
            core_attn_out = core_attn_out.unsqueeze(0)  # [1, T, H_v, V]

        except ImportError:
            # Fallback to Triton fla_ops (may OOM on very long sequences)
            from nanovllm.layers.fla_ops.chunk import chunk_gated_delta_rule as triton_chunk
            core_attn_out, last_recurrent_state = triton_chunk(
                q=query, k=key, v=value, g=g, beta=beta,
                initial_state=h0,
                output_final_state=save_state,
                use_qk_l2norm_in_kernel=True,
            )

        # Save state to buffer
        if save_state and last_recurrent_state is not None:
            # last_recurrent_state: [1, H_v, V, K] → buf[slot]: [H_v, V, K] (same layout)
            self.recurrent_state_buf[slot_idx].copy_(
                last_recurrent_state.squeeze(0).to(self.recurrent_state_buf.dtype)
            )


        # Norm + output projection
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output.squeeze(0)

    def _forward_decode_batched(
        self,
        hidden_states: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Decode using fused_recurrent_gated_delta_rule (float32 state, L2norm in kernel)."""
        B = hidden_states.shape[0]

        # Projections
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        # Conv1d with state buffer
        mixed_qkv_col = mixed_qkv.unsqueeze(-1)
        conv_state = self.conv_state_buf[slot_indices]
        conv_input = torch.cat([conv_state, mixed_qkv_col], dim=-1)
        self.conv_state_buf[slot_indices] = conv_input[:, :, 1:]
        mixed_qkv_act = F.silu(
            F.conv1d(conv_input, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
        ).squeeze(-1)

        # Split Q, K, V
        query, key, value = mixed_qkv_act.split(
            [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.view(B, self.num_k_heads, self.head_k_dim)
        key = key.view(B, self.num_k_heads, self.head_k_dim)
        value = value.view(B, self.num_v_heads, self.head_v_dim)

        # GQA expand
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=1)
            key = key.repeat_interleave(ratio, dim=1)

        # Compute gating
        beta_val = b.sigmoid()
        g_log = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Use fused_recurrent (does L2 norm + scale + state update in float32)
        from nanovllm.layers.fla_ops.fused_recurrent import fused_recurrent_gated_delta_rule
        q_4d = query.unsqueeze(1)       # [B, 1, H_v, K]
        k_4d = key.unsqueeze(1)         # [B, 1, H_v, K]
        v_4d = value.unsqueeze(1)       # [B, 1, H_v, V]
        g_3d = g_log.unsqueeze(1)       # [B, 1, H_v]
        beta_3d = beta_val.unsqueeze(1) # [B, 1, H_v]

        # Read state: buf[slot, H_v, V, K] → [B, H_v, V, K] (same layout, no transpose)
        state = self.recurrent_state_buf[slot_indices].contiguous()

        o_4d, final_state = fused_recurrent_gated_delta_rule(
            q=q_4d, k=k_4d, v=v_4d, g=g_3d, beta=beta_3d,
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        # Write state back: [B, H_v, V, K] → buffer (same layout)
        self.recurrent_state_buf[slot_indices] = final_state.to(self.recurrent_state_buf.dtype)

        # Norm + output projection
        output = o_4d.squeeze(1)  # [B, H_v, V]
        output = output.to(hidden_states.dtype)
        output = output.reshape(B * self.num_v_heads, self.head_v_dim)
        z_flat = z.reshape(B * self.num_v_heads, self.head_v_dim)
        output = self.norm(output, z_flat)
        output = output.view(B, self.value_dim)
        return self.out_proj(output)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from nanovllm.utils.context import get_context
        context = get_context()

        if context.is_prefill and context.cu_seqlens_q is not None:
            cu_seqlens = context.cu_seqlens_q
            num_seqs = len(cu_seqlens) - 1
            outputs = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_hidden = hidden_states[start:end]
                if (context.linear_attn_slot_indices is not None
                        and self.recurrent_state_buf is not None):
                    slot_idx = context.linear_attn_slot_indices[i].item()
                else:
                    slot_idx = None
                seq_out = self._forward_prefill(seq_hidden, slot_idx=slot_idx)
                outputs.append(seq_out)
            return torch.cat(outputs, dim=0)
        else:
            if (context.linear_attn_slot_indices is not None
                    and self.recurrent_state_buf is not None):
                return self._forward_decode_batched(
                    hidden_states, context.linear_attn_slot_indices
                )
            else:
                return torch.zeros_like(hidden_states)

class Qwen3_5FullAttention(nn.Module):
    """
    Full attention layer with output gating.

    Reference: q_proj outputs [num_heads * head_dim * 2] which is reshaped to
    [bs, seq, num_heads, head_dim*2], then chunk(2, dim=-1) -> (query, gate).
    gate is then reshaped to [bs, seq, num_heads*head_dim].
    query goes through q_norm, then RoPE.
    After attention, output *= sigmoid(gate).

    Weights:
    - self_attn.q_proj.weight  [num_heads * head_dim * 2, hidden_size]
    - self_attn.k_proj.weight  [num_kv_heads * head_dim, hidden_size]
    - self_attn.v_proj.weight  [num_kv_heads * head_dim, hidden_size]
    - self_attn.o_proj.weight  [hidden_size, num_heads * head_dim]
    - self_attn.q_norm.weight  [head_dim]  (zeros-initialized, (1+w) style)
    - self_attn.k_norm.weight  [head_dim]  (zeros-initialized, (1+w) style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int,
        rms_norm_eps: float,
        rope_theta: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        
        # Determine whether to shard or replicate KV heads
        if num_kv_heads < tp_size:
            # KV heads are less than TP size: replicate them instead of sharding
            self.num_kv_heads = num_kv_heads
            kv_output_size = num_kv_heads * head_dim
            use_replicated_kv = True
        else:
            # Normal sharding case
            self.num_kv_heads = num_kv_heads // tp_size
            kv_output_size = num_kv_heads * head_dim
            use_replicated_kv = False
        
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # q_proj outputs query + gate (2x query size)
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            num_heads * head_dim * 2,  # query + gate
            bias=False,
        )
        
        # K,V projections: use ReplicatedLinear if num_kv_heads < tp_size
        if use_replicated_kv:
            self.k_proj = ReplicatedLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
            self.v_proj = ReplicatedLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
        else:
            self.k_proj = ColumnParallelLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size,
                kv_output_size,
                bias=False,
            )
        
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )

        # Rotary embedding (partial)
        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = rotary_dim
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=rope_theta,
        )

        # Q/K norms — Qwen3.5 uses (1+w) style RMSNorm
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)

        # Attention kernel
        # When KV is replicated and expanded to match Q heads, use num_heads for kv_heads
        attn_kv_heads = self.num_heads if use_replicated_kv else self.num_kv_heads
        self.attn = Attention(
            self.num_heads,
            head_dim,
            self.scaling,
            attn_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Q projection: outputs [N, num_heads * head_dim * 2 / tp_size]
        q_out = self.q_proj(hidden_states)

        # Reshape to [N, num_heads, head_dim * 2], then split query and gate
        q_out = q_out.view(-1, self.num_heads, self.head_dim * 2)
        query, gate = q_out.chunk(2, dim=-1)  # each [N, num_heads, head_dim]
        # gate is flattened: [N, num_heads * head_dim]
        gate = gate.reshape(-1, self.num_heads * self.head_dim)

        # K, V projections
        k_full = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v_full = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        # For GQA with TP: select the correct KV head(s) for this rank's Q heads
        # Each rank's Q heads map to specific KV heads based on GQA ratio
        if self.total_num_kv_heads < dist.get_world_size():
            tp_rank = dist.get_rank()
            gqa_ratio = self.total_num_heads // self.total_num_kv_heads
            kv_head_idx = (tp_rank * self.num_heads) // gqa_ratio
            # All Q heads on this rank map to the same KV head
            k = k_full[:, kv_head_idx:kv_head_idx+1, :].expand(-1, self.num_heads, -1).contiguous()
            v = v_full[:, kv_head_idx:kv_head_idx+1, :].expand(-1, self.num_heads, -1).contiguous()
        else:
            k = k_full
            v = v_full

        # Apply Q/K norms (Qwen3.5 uses (1+w) style)
        query = self.q_norm(query)
        k = self.k_norm(k)

        # Apply partial rotary embedding
        query, k = self.rotary_emb(positions, query, k)

        # Attention
        o = self.attn(query, k, v)

        # Apply output gate: output *= sigmoid(gate)
        o_flat = o.flatten(1, -1)  # [N, q_size]
        o_flat = o_flat * torch.sigmoid(gate)

        # Output projection
        output = self.o_proj(o_flat)
        return output


# ============================================================
# MoE: Top-K Router
# ============================================================

class Qwen3_5TopKRouter(nn.Module):
    """
    Top-K router with softmax normalization.

    Weight: mlp.gate.weight [num_experts, hidden_size]
    """

    def __init__(self, num_experts: int, hidden_size: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [N, hidden_size]
        router_logits = F.linear(hidden_states, self.weight)  # [N, num_experts]
        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        top_k_values, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_values = top_k_values / top_k_values.sum(dim=-1, keepdim=True)
        top_k_values = top_k_values.to(hidden_states.dtype)
        return top_k_values, top_k_indices


# ============================================================
# MoE: Fused Experts (3D Parameter Tensors)
# ============================================================

class Qwen3_5Experts(nn.Module):
    """
    Fused expert weights stored as 3D tensors.

    Weights:
    - experts.gate_up_proj  [num_experts, 2 * moe_intermediate_size, hidden_size]
    - experts.down_proj     [num_experts, hidden_size, moe_intermediate_size]
    """

    def __init__(self, num_experts: int, hidden_size: int, moe_intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from nanovllm.utils.context import get_context
        context = get_context()
        if context.is_prefill:
            return self._forward_sparse(hidden_states, top_k_indices, top_k_weights)
        else:
            return self._forward_dense(hidden_states, top_k_indices, top_k_weights)

    def _forward_sparse(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse dispatch — used for prefill (not in CUDA Graph).
        Uses nonzero/where to only compute active expert-token pairs."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, N]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def _forward_dense(
        self,
        hidden_states: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dense gather-based dispatch — CUDA Graph compatible for decode.
        Batches all top_k experts in a single gather+bmm for efficiency."""
        N = hidden_states.shape[0]
        top_k = top_k_indices.shape[1]

        # Gather all expert weights at once: [N*top_k, ...]
        flat_idx = top_k_indices.reshape(-1)                    # [N*top_k]
        gate_up_w = self.gate_up_proj[flat_idx]                 # [N*top_k, 2*inter, hidden]
        down_w = self.down_proj[flat_idx]                       # [N*top_k, hidden, inter]

        # Expand hidden states: [N, hidden] -> [N*top_k, hidden]
        h_expand = hidden_states.repeat_interleave(top_k, dim=0)  # [N*top_k, hidden]

        # Batched gate_up projection
        x = torch.bmm(gate_up_w, h_expand.unsqueeze(-1)).squeeze(-1)  # [N*top_k, 2*inter]
        gate, up = x.chunk(2, dim=-1)
        x = F.silu(gate) * up                                  # [N*top_k, inter]

        # Batched down projection
        x = torch.bmm(down_w, x.unsqueeze(-1)).squeeze(-1)     # [N*top_k, hidden]

        # Reshape and weighted sum
        x = x.view(N, top_k, -1)                               # [N, top_k, hidden]
        final = (x * top_k_weights.unsqueeze(-1)).sum(dim=1)    # [N, hidden]

        return final


# ============================================================
# MoE: Sparse MoE Block (Router + Experts + Shared Expert)
# ============================================================

class Qwen3_5SparseMoeBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
    ):
        super().__init__()
        self.gate = Qwen3_5TopKRouter(num_experts, hidden_size, num_experts_per_tok)
        self.experts = Qwen3_5Experts(num_experts, hidden_size, moe_intermediate_size)
        self.shared_expert = Qwen3_5SharedExpertMLP(
            hidden_size, shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_expert_output = self.shared_expert(hidden_states)
        top_k_weights, top_k_indices = self.gate(hidden_states)
        expert_output = self.experts(hidden_states, top_k_indices, top_k_weights)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        output = expert_output + shared_expert_output
        return output


class Qwen3_5SharedExpertMLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Decoder Layer
# ============================================================

class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.hidden_size = text_config.hidden_size
        self.layer_type = text_config.layer_types[layer_idx]
        self.layer_idx = layer_idx

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

        self.mlp = Qwen3_5SparseMoeBlock(
            hidden_size=text_config.hidden_size,
            num_experts=text_config.num_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            moe_intermediate_size=text_config.moe_intermediate_size,
            shared_expert_intermediate_size=text_config.shared_expert_intermediate_size,
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
        # Qwen3.5 uses standard residual connection pattern (NOT fused norm+residual):
        #   residual = hidden_states
        #   hidden_states = norm(hidden_states)
        #   hidden_states = attn(hidden_states)
        #   hidden_states = residual + hidden_states
        #
        # We use the residual parameter to carry state between layers.
        # On first layer: residual is None, so we use hidden_states as residual.
        # On subsequent layers: hidden_states is the output from previous mlp,
        #   residual carries the pre-mlp state. We add them first.
        if residual is not None:
            hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        torch.cuda.nvtx.range_push(f"{self.layer_type}_{self.layer_idx}")
        # Token mixing
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        torch.cuda.nvtx.range_pop()

        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        torch.cuda.nvtx.range_push(f"{self.layer_idx}_mlp")
        hidden_states = self.mlp(hidden_states)
        torch.cuda.nvtx.range_pop()

        # Return hidden_states (mlp output, not yet added to residual)
        # and residual (pre-mlp state). They'll be added at start of next layer.
        return hidden_states, residual


# ============================================================
# Text Model (Language Model)
# ============================================================

class Qwen3_5TextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.embed_tokens = VocabParallelEmbedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, i)
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
# Top-level: Qwen3.5 For Causal LM
# ============================================================

class Qwen3_5ForCausalLM(nn.Module):

    packed_modules_mapping = {}

    # Weight name prefix to strip (VLM structure)
    weight_prefix = "model."

    # Prefixes to skip during weight loading
    skip_prefixes = ("mtp.", "model.visual.", "model.merger.")

    def __init__(self, config):
        super().__init__()
        text_config = config.text_config if hasattr(config, 'text_config') else config
        self.language_model = Qwen3_5TextModel(config)
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
