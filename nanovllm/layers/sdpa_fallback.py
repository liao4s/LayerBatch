"""Pure-Torch SDPA fallback for prefill and decode.

Motivation
----------
On Ampere hosts where neither ``flash_attn`` nor ``flashinfer`` is usable,
we still need SOMETHING to compute attention.  Torch's built-in
``scaled_dot_product_attention`` (SDPA) is universal — it ships with
PyTorch and works on any CUDA-capable device, including sm_80.  It is
slower than FA/FlashInfer but correct and dependency-free.

Two entry points:

* ``sdpa_prefill_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, scale, ...)``
  — variable-length prefill, causal.  Handles both the no-prefix-cache
  path (K/V passed in ragged) and the prefix-cache path (K/V read from
  paged cache via ``block_tables``).

* ``sdpa_decode_paged(q, k_cache, v_cache, block_tables, context_lens, scale)``
  — one-token-per-seq decode from paged cache.  Wraps the tensor gather +
  a batched SDPA call.

Layout
------
Nano-vllm passes q/k/v as ``[N_total, num_heads, head_dim]`` (packed
varlen) with cu_seqlens delimiters.  We reshape to per-seq
``[H, S, D]`` for SDPA (which expects ``[..., L, E]`` with heads leading
via broadcasting), run the kernel, then reshape back.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _sdpa_one_seq(q_seq: torch.Tensor,
                  k_seq: torch.Tensor,
                  v_seq: torch.Tensor,
                  scale: float,
                  causal: bool = True) -> torch.Tensor:
    """Run SDPA on one sequence.

    q_seq: [Sq, Hq, D]
    k_seq, v_seq: [Sk, Hkv, D]  (Hkv can be < Hq under GQA — we broadcast)
    Returns: [Sq, Hq, D]
    """
    Hq, D = q_seq.shape[-2], q_seq.shape[-1]
    Hkv = k_seq.shape[-2]
    # SDPA wants [B, H, S, D]; pack seq as batch=1.
    q = q_seq.transpose(0, 1).unsqueeze(0)  # [1, Hq, Sq, D]
    k = k_seq.transpose(0, 1).unsqueeze(0)  # [1, Hkv, Sk, D]
    v = v_seq.transpose(0, 1).unsqueeze(0)  # [1, Hkv, Sk, D]
    # GQA: expand kv heads to Hq via repeat_interleave.
    if Hkv != Hq:
        assert Hq % Hkv == 0, f"num_qo_heads {Hq} must be a multiple of num_kv_heads {Hkv}"
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
    return o.squeeze(0).transpose(0, 1)  # [Sq, Hq, D]


def sdpa_prefill_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        cu_seqlens_q: torch.Tensor,
                        cu_seqlens_k: torch.Tensor,
                        scale: float,
                        block_tables: torch.Tensor | None = None,
                        k_cache: torch.Tensor | None = None,
                        v_cache: torch.Tensor | None = None,
                        block_size: int = 256) -> torch.Tensor:
    """Variable-length causal prefill.

    q: [Nq, Hq, D]
    k, v: [Nk, Hkv, D] — used only if block_tables is None
    cu_seqlens_{q,k}: [B+1] int32 on GPU
    block_tables: [B, max_num_blocks] int32 on GPU (populated when there is
        a prefix-cache hit); when set, K/V are gathered from k_cache/v_cache.
    k_cache, v_cache: paged buffers [num_blocks, block_size, Hkv, D].
    block_size: KV cache page size.

    Returns: [Nq, Hq, D]
    """
    device = q.device
    # We need cu_seqlens on CPU to iterate — this is a tiny tensor (B+1 ints).
    cu_q_cpu = cu_seqlens_q.detach().to("cpu", non_blocking=False).tolist()
    cu_k_cpu = cu_seqlens_k.detach().to("cpu", non_blocking=False).tolist()
    B = len(cu_q_cpu) - 1
    out = torch.empty_like(q)
    if block_tables is not None:
        # Prefix-cache path: gather K/V from paged cache per seq.
        # Move block_tables to CPU once for indexing.
        bt_cpu = block_tables.detach().to("cpu").tolist()
    for i in range(B):
        q_start, q_end = cu_q_cpu[i], cu_q_cpu[i + 1]
        k_start, k_end = cu_k_cpu[i], cu_k_cpu[i + 1]
        Sq = q_end - q_start
        Sk = k_end - k_start
        if Sq == 0:
            continue
        q_seq = q[q_start:q_end]  # [Sq, Hq, D]
        if block_tables is not None:
            # gather K/V from paged cache using block_tables[i].
            num_blocks_needed = (Sk + block_size - 1) // block_size
            block_ids = bt_cpu[i][:num_blocks_needed]
            # k_cache[b] is [block_size, Hkv, D]; cat into [Sk, Hkv, D].
            k_pieces = []
            v_pieces = []
            remaining = Sk
            for b in block_ids:
                take = min(block_size, remaining)
                k_pieces.append(k_cache[b, :take])
                v_pieces.append(v_cache[b, :take])
                remaining -= take
            k_seq = torch.cat(k_pieces, dim=0)
            v_seq = torch.cat(v_pieces, dim=0)
        else:
            k_seq = k[k_start:k_end]
            v_seq = v[k_start:k_end]
        out[q_start:q_end] = _sdpa_one_seq(q_seq, k_seq, v_seq, scale, causal=True)
    return out


def sdpa_decode_paged(q: torch.Tensor,
                      k_cache: torch.Tensor,
                      v_cache: torch.Tensor,
                      block_tables: torch.Tensor,
                      context_lens: torch.Tensor,
                      scale: float,
                      block_size: int = 256) -> torch.Tensor:
    """Single-token-per-seq decode from paged cache.

    q: [B, Hq, D]  (one query token per seq)
    k_cache, v_cache: [num_blocks, block_size, Hkv, D]
    block_tables: [B, max_num_blocks] int32
    context_lens: [B] int32 — total tokens (cached + this one) per seq

    Returns: [B, 1, Hq, D]  (matches flash_attn_with_kvcache output shape).
    """
    B, Hq, D = q.shape
    ctx_cpu = context_lens.detach().to("cpu").tolist()
    bt_cpu = block_tables.detach().to("cpu").tolist()
    outs = []
    for i in range(B):
        Sk = ctx_cpu[i]
        num_blocks_needed = (Sk + block_size - 1) // block_size
        block_ids = bt_cpu[i][:num_blocks_needed]
        k_pieces, v_pieces = [], []
        remaining = Sk
        for b in block_ids:
            take = min(block_size, remaining)
            k_pieces.append(k_cache[b, :take])
            v_pieces.append(v_cache[b, :take])
            remaining -= take
        k_seq = torch.cat(k_pieces, dim=0)  # [Sk, Hkv, D]
        v_seq = torch.cat(v_pieces, dim=0)
        # Add a Sq=1 dimension.
        q_seq = q[i].unsqueeze(0)  # [1, Hq, D]
        o = _sdpa_one_seq(q_seq, k_seq, v_seq, scale, causal=False)  # [1, Hq, D]
        outs.append(o)
    o_all = torch.stack(outs, dim=0)  # [B, 1, Hq, D]
    return o_all
