"""POD-Attention CTA-level fused kernels for LayerBatch.

The single-launch kernel partitions its CTA grid into two ranges:
  - First N_ATTN CTAs run a paged flash-attention-decode tile (Group A's FA).
  - Remaining N_GEMM CTAs run a bf16 GEMM tile (Group B's LA in_proj).

Both ranges of CTAs land on the SAME SMs (no Green-Context partitioning), so
the SM warp scheduler interleaves their warps:
  - Attention warps are mostly stalling on HBM (reading paged KV).
  - GEMM warps are mostly issuing tensor-core MMAs (reading weight tiles
    from L2 once and reusing register files).
This co-residency keeps the TC and LSU pipelines BOTH busy on each SM, which
is the core POD-Attention principle.

Algorithm origin: stage-1 of vLLM's `triton_decode_attention.py` (split-KV
online softmax) — adapted to nanovllm's paged KV layout (block_table format),
extended with the second CTA branch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Stage-1 grouped paged decode (attention path, vLLM-style adapted to our KV
# layout) WITH a piggy-backed GEMM branch on the tail CTAs.
# ---------------------------------------------------------------------------
@triton.jit
def _pod_fused_kernel(
    # ---------- attention path inputs ----------
    Q_ptr,                  # [B, H_q, D]                      (bf16)
    K_cache_ptr,            # [num_blocks, block_size, H_kv, D] (bf16)
    V_cache_ptr,            # [num_blocks, block_size, H_kv, Dv](bf16)
    sm_scale,               # float
    BlockTbl_ptr,           # [B, max_blocks_per_seq]           (int32)
    CtxLen_ptr,             # [B]                               (int32)
    Mid_O_ptr,              # [B, H_q, NUM_KV_SPLITS, D + 1]    (fp32)
                            # ... last lane in last dim stores log(l) + m
    stride_q_b, stride_q_h,
    stride_kbs, stride_kh,
    stride_vbs, stride_vh,
    stride_blk_b,
    stride_mid_b, stride_mid_h, stride_mid_s,

    # ---------- GEMM path inputs ----------
    A_ptr,                  # [M, K]   bf16 (input × weight^T or weight)
    B_ptr,                  # [K, N]   bf16
    C_ptr,                  # [M, N]   bf16
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,

    # ---------- shape / config ----------
    B: tl.constexpr,                    # batch size for attention
    KV_GROUP_NUM: tl.constexpr,         # H_q / H_kv
    Q_HEAD_NUM: tl.constexpr,           # H_q
    BLOCK_DMODEL: tl.constexpr,         # head_dim (Q,K)
    BLOCK_DV: tl.constexpr,             # head_dim (V) — usually = BLOCK_DMODEL
    BLOCK_N: tl.constexpr,              # KV-tile rows per iter
    BLOCK_H: tl.constexpr,              # Q-heads per CTA
    NUM_KV_SPLITS: tl.constexpr,        # split-KV factor
    PAGE_SIZE: tl.constexpr,            # block_size

    M_GEMM: tl.constexpr, N_GEMM: tl.constexpr, K_GEMM: tl.constexpr,
    BLOCK_M_G: tl.constexpr, BLOCK_N_G: tl.constexpr, BLOCK_K_G: tl.constexpr,

    N_ATTN_CTAS: tl.constexpr,          # B * head_groups * NUM_KV_SPLITS
    HEAD_GROUPS: tl.constexpr,          # cdiv(H_q, min(BLOCK_H, KV_GROUP_NUM))
):
    """One launch, two work paths.

    pid in [0, N_ATTN_CTAS)             → attention CTA
    pid in [N_ATTN_CTAS, N_ATTN_CTAS+M_tiles*N_tiles) → GEMM CTA
    """
    pid = tl.program_id(0)

    if pid < N_ATTN_CTAS:
        # ===================================================================
        # ATTENTION PATH (paged flash-attention decode, split-KV)
        # ===================================================================
        # Decode pid into (batch, head_group, kv_split)
        cur_batch = pid // (HEAD_GROUPS * NUM_KV_SPLITS)
        rem = pid - cur_batch * (HEAD_GROUPS * NUM_KV_SPLITS)
        cur_head_id = rem // NUM_KV_SPLITS
        split_kv_id = rem - cur_head_id * NUM_KV_SPLITS

        if KV_GROUP_NUM > BLOCK_H:
            VALID_BLOCK_H: tl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: tl.constexpr = KV_GROUP_NUM

        cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_NUM, BLOCK_H)
        cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
        mask_h = mask_h & (cur_head < Q_HEAD_NUM)

        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_dv = tl.arange(0, BLOCK_DV)

        cur_batch_seq_len = tl.load(CtxLen_ptr + cur_batch)

        offs_q = cur_batch * stride_q_b + cur_head[:, None] * stride_q_h + offs_d[None, :]
        q = tl.load(Q_ptr + offs_q, mask=mask_h[:, None], other=0.0)

        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

        if split_kv_end > split_kv_start:
            for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                # Map token index → (block_id, slot_in_block) via block_table
                kv_block_id = tl.load(
                    BlockTbl_ptr + stride_blk_b * cur_batch + offs_n // PAGE_SIZE,
                    mask=offs_n < split_kv_end, other=0,
                )
                kv_loc = kv_block_id * PAGE_SIZE + offs_n % PAGE_SIZE

                offs_buf_k = (kv_loc[None, :] * stride_kbs +
                              cur_kv_head * stride_kh + offs_d[:, None])
                k = tl.load(K_cache_ptr + offs_buf_k,
                            mask=offs_n[None, :] < split_kv_end, other=0.0)

                qk = tl.dot(q, k.to(q.dtype))
                qk *= sm_scale
                qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                              qk, float("-inf"))

                offs_buf_v = (kv_loc[:, None] * stride_vbs +
                              cur_kv_head * stride_vh + offs_dv[None, :])
                v = tl.load(V_cache_ptr + offs_buf_v,
                            mask=offs_n[:, None] < split_kv_end, other=0.0)

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc = acc * re_scale[:, None]
                acc = acc + tl.dot(p.to(v.dtype), v)
                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

            offs_mid_o = (cur_batch * stride_mid_b +
                          cur_head[:, None] * stride_mid_h +
                          split_kv_id * stride_mid_s + offs_dv[None, :])
            tl.store(Mid_O_ptr + offs_mid_o,
                     acc / e_sum[:, None],
                     mask=mask_h[:, None])
            offs_mid_o_logic = (cur_batch * stride_mid_b +
                                cur_head * stride_mid_h +
                                split_kv_id * stride_mid_s + BLOCK_DV)
            tl.store(Mid_O_ptr + offs_mid_o_logic,
                     e_max + tl.log(e_sum),
                     mask=mask_h)
    else:
        # ===================================================================
        # GEMM PATH — bf16 [M, K] × [K, N] → [M, N], output written in bf16
        # ===================================================================
        gpid = pid - N_ATTN_CTAS
        n_n_tiles_g = (N_GEMM + BLOCK_N_G - 1) // BLOCK_N_G
        tile_m_g = gpid // n_n_tiles_g
        tile_n_g = gpid - tile_m_g * n_n_tiles_g

        offs_m_g = tile_m_g * BLOCK_M_G + tl.arange(0, BLOCK_M_G)
        offs_n_g = tile_n_g * BLOCK_N_G + tl.arange(0, BLOCK_N_G)
        offs_k_g = tl.arange(0, BLOCK_K_G)

        a_ptrs = A_ptr + offs_m_g[:, None] * stride_a_m + offs_k_g[None, :] * stride_a_k
        b_ptrs = B_ptr + offs_k_g[:, None] * stride_b_k + offs_n_g[None, :] * stride_b_n

        acc_g = tl.zeros([BLOCK_M_G, BLOCK_N_G], dtype=tl.float32)
        m_mask_g = offs_m_g < M_GEMM
        n_mask_g = offs_n_g < N_GEMM
        for k_iter in range(0, K_GEMM, BLOCK_K_G):
            k_mask_g = (k_iter + offs_k_g) < K_GEMM
            a_tile = tl.load(a_ptrs, mask=m_mask_g[:, None] & k_mask_g[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=k_mask_g[:, None] & n_mask_g[None, :], other=0.0)
            acc_g = acc_g + tl.dot(a_tile, b_tile)
            a_ptrs = a_ptrs + BLOCK_K_G * stride_a_k
            b_ptrs = b_ptrs + BLOCK_K_G * stride_b_k

        c_ptrs = C_ptr + offs_m_g[:, None] * stride_c_m + offs_n_g[None, :] * stride_c_n
        tl.store(c_ptrs, acc_g.to(C_ptr.dtype.element_ty),
                 mask=m_mask_g[:, None] & n_mask_g[None, :])


# ---------------------------------------------------------------------------
# Stage-2 reduce kernel (NO GEMM piggy-back here — kept simple).
# ---------------------------------------------------------------------------
@triton.jit
def _pod_reduce_kernel(
    Mid_O_ptr,           # [B, H, NUM_KV_SPLITS, D+1]
    O_ptr,               # [B, H, D]
    CtxLen_ptr,
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_o_b, stride_o_h,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_seq_len = tl.load(CtxLen_ptr + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    base = cur_batch * stride_mid_b + cur_head * stride_mid_h
    for split in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_seq_len, NUM_KV_SPLITS)
        s_start = kv_len_per_split * split
        s_end = tl.minimum(s_start + kv_len_per_split, cur_seq_len)
        if s_end > s_start:
            tv = tl.load(Mid_O_ptr + base + split * stride_mid_s + offs_d)
            tlogic = tl.load(Mid_O_ptr + base + split * stride_mid_s + BLOCK_DV)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc = acc * old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc = acc + exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(O_ptr + cur_batch * stride_o_b + cur_head * stride_o_h + offs_d,
             acc / e_sum)


# ---------------------------------------------------------------------------
# Python entry points.
# ---------------------------------------------------------------------------
def pod_fused_attn_gemm(
    q: torch.Tensor,                # [B, H_q, D]   bf16
    k_cache: torch.Tensor,          # [num_blocks, page_size, H_kv, D] bf16
    v_cache: torch.Tensor,          # [num_blocks, page_size, H_kv, D] bf16
    block_table: torch.Tensor,      # [B, max_blocks] int32
    context_lens: torch.Tensor,     # [B] int32
    sm_scale: float,
    A: torch.Tensor,                # [M, K] bf16  (LA in_proj input)
    B: torch.Tensor,                # [K, N] bf16  (LA in_proj weight)
    C_out: torch.Tensor,            # [M, N] bf16  (output)
    num_kv_splits: int = 16,
    block_n: int = 64,
    block_h: int = 16,
    mid_o: torch.Tensor | None = None,   # [B, H_q, NUM_KV_SPLITS, D+1] fp32, optional pre-alloc
    out: torch.Tensor | None = None,     # [B, H_q, D] bf16, optional pre-alloc
):
    """Fused decode flash-attention + GEMM.  Returns attention output [B, H_q, D].

    Layout assumptions:
      - K/V cache block shape: [num_blocks, page_size, H_kv, D] (last-axis contiguous).
      - Q is bf16, [B, H_q, D] (last-axis contiguous).
    """
    Bsz, H_q, D = q.shape
    H_kv = k_cache.shape[2]
    KV_GROUP_NUM = H_q // H_kv
    page_size = k_cache.shape[1]
    BLOCK_DMODEL = D
    BLOCK_DV = D

    # Mid_O shape: [B, H_q, NUM_KV_SPLITS, D + 1]; pre-allocated by caller for
    # CUDA-Graph capture compatibility.  Fallback to dynamic alloc only when not given.
    if mid_o is None:
        mid_o = torch.empty((Bsz, H_q, num_kv_splits, D + 1),
                            dtype=torch.float32, device=q.device)
    if out is None:
        out = torch.empty((Bsz, H_q, D), dtype=q.dtype, device=q.device)

    HEAD_GROUPS = max(1, triton.cdiv(H_q, min(block_h, KV_GROUP_NUM)))
    N_ATTN_CTAS = Bsz * HEAD_GROUPS * num_kv_splits

    # GEMM grid: cdiv(M, BLOCK_M_G) × cdiv(N, BLOCK_N_G)
    M, K = A.shape
    K_, N = B.shape
    assert K == K_, f"GEMM K mismatch: {K} vs {K_}"
    assert C_out.shape == (M, N)
    BLOCK_M_G = 16   # decode: M is small (1..8)
    BLOCK_N_G = 64
    BLOCK_K_G = 32
    n_m_tiles = triton.cdiv(M, BLOCK_M_G)
    n_n_tiles = triton.cdiv(N, BLOCK_N_G)
    N_GEMM_CTAS = n_m_tiles * n_n_tiles
    grid = (N_ATTN_CTAS + N_GEMM_CTAS,)

    _pod_fused_kernel[grid](
        q, k_cache, v_cache, sm_scale, block_table, context_lens, mid_o,
        q.stride(0), q.stride(1),
        k_cache.stride(-3), k_cache.stride(-2),
        v_cache.stride(-3), v_cache.stride(-2),
        block_table.stride(0),
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        A, B, C_out,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C_out.stride(0), C_out.stride(1),
        B=Bsz,
        KV_GROUP_NUM=KV_GROUP_NUM,
        Q_HEAD_NUM=H_q,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        M_GEMM=M, N_GEMM=N, K_GEMM=K,
        BLOCK_M_G=BLOCK_M_G, BLOCK_N_G=BLOCK_N_G, BLOCK_K_G=BLOCK_K_G,
        N_ATTN_CTAS=N_ATTN_CTAS,
        HEAD_GROUPS=HEAD_GROUPS,
        num_warps=4, num_stages=3,
    )

    # Stage-2 reduce (attn-only)
    _pod_reduce_kernel[(Bsz, H_q)](
        mid_o, out, context_lens,
        mid_o.stride(0), mid_o.stride(1), mid_o.stride(2),
        out.stride(0), out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        num_warps=4, num_stages=2,
    )
    return out


def pod_attn_only(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int = 16,
    block_n: int = 64,
    block_h: int = 16,
    mid_o: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    """Same as pod_fused_attn_gemm but with NO_GEMM path = empty work.

    Used as a baseline to measure the GEMM piggy-back's marginal cost.
    Builds a degenerate 1×1 GEMM that exits the inner loop in 0 iterations.
    """
    dummy = torch.zeros((1, 0), dtype=q.dtype, device=q.device)
    dummy_w = torch.zeros((0, 1), dtype=q.dtype, device=q.device)
    dummy_c = torch.zeros((1, 1), dtype=q.dtype, device=q.device)
    return pod_fused_attn_gemm(
        q, k_cache, v_cache, block_table, context_lens, sm_scale,
        dummy, dummy_w, dummy_c, num_kv_splits, block_n, block_h,
        mid_o=mid_o, out=out,
    )
