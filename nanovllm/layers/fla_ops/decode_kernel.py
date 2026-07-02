"""
Triton kernel for batched single-step GatedDeltaNet decode.
Replaces the naive PyTorch implementation in Qwen3_5GatedDeltaNet._forward_decode_batched.

The kernel processes:
  state *= g (decay)
  kv_mem = state @ k (recall)
  delta = (v - kv_mem) * beta
  state += k^T @ delta (update)
  output = state @ q (query)

State layout: [max_slots, H, K, V] (float32 buffer)
Input shapes: q,k [B, H, K], v [B, H, V], g,beta [B, H]
Output: [B, H, V]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gdn_decode_kernel(
    # Pointers
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    state_ptr,
    slot_indices_ptr,
    # Dimensions
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    # Block dims
    BV: tl.constexpr,
    # Strides
    stride_state_slot,  # stride for slot dimension
    stride_state_h,     # stride for head dimension
    stride_state_k,     # stride for K dimension
):
    """
    Grid: (cdiv(V, BV), B * H)
    Each program handles BV elements of the V dimension for one (batch, head) pair.
    """
    i_v = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # Get slot index for this batch element
    slot_idx = tl.load(slot_indices_ptr + i_b).to(tl.int64)

    # Load g and beta scalars
    g_val = tl.load(g_ptr + i_b * H + i_h).to(tl.float32)
    beta_val = tl.load(beta_ptr + i_b * H + i_h).to(tl.float32)
    g_exp = tl.exp(g_val)

    # V-dimension offsets for this block
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V

    # Load v vector slice: v[i_b, i_h, o_v]
    v_val = tl.load(v_ptr + i_b * H * V + i_h * V + o_v, mask=mask_v, other=0.0).to(tl.float32)

    # State base pointer for this (slot, head): state[slot, h, :, :]
    state_base = state_ptr + slot_idx * stride_state_slot + i_h * stride_state_h

    # Iterate over K dimension to compute:
    #   kv_mem[v] = sum_k(state[k,v] * k[k])
    #   state[k,v] = state[k,v] * g + k[k] * delta[v]
    #   output[v] = sum_k(state[k,v] * q[k])

    # Load q and k vectors for this (batch, head)
    # q, k shape: [B, H, K]
    q_base = q_ptr + i_b * H * K + i_h * K
    k_base = k_ptr + i_b * H * K + i_h * K

    # Accumulate output and kv_mem
    b_o = tl.zeros([BV], dtype=tl.float32)
    b_kv_mem = tl.zeros([BV], dtype=tl.float32)

    # First pass: decay state and compute kv_mem
    for i_k in range(K):
        # Load state[slot, h, k, v_block]
        state_offset = state_base + i_k * stride_state_k + o_v
        s = tl.load(state_offset, mask=mask_v, other=0.0).to(tl.float32)
        # Decay
        s = s * g_exp
        # Load k[b, h, k]
        k_val = tl.load(k_base + i_k).to(tl.float32)
        # kv_mem[v] += s[k,v] * k[k]
        b_kv_mem += s * k_val
        # Store decayed state temporarily (we need to add delta*k later)
        tl.store(state_offset, s.to(state_offset.dtype.element_ty), mask=mask_v)

    # Compute delta = (v - kv_mem) * beta
    b_delta = (v_val - b_kv_mem) * beta_val

    # Second pass: update state and compute output
    for i_k in range(K):
        state_offset = state_base + i_k * stride_state_k + o_v
        s = tl.load(state_offset, mask=mask_v, other=0.0).to(tl.float32)
        k_val = tl.load(k_base + i_k).to(tl.float32)
        q_val = tl.load(q_base + i_k).to(tl.float32)
        # Update: state += k * delta
        s = s + k_val * b_delta
        # Output: o += state * q
        b_o += s * q_val
        # Store final state
        tl.store(state_offset, s.to(state_offset.dtype.element_ty), mask=mask_v)

    # Store output: o[b, h, v_block]
    tl.store(o_ptr + i_b * H * V + i_h * V + o_v, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)


def gdn_decode_batched(
    q: torch.Tensor,       # [B, H, K] float (already scaled, L2-normed)
    k: torch.Tensor,       # [B, H, K] float
    v: torch.Tensor,       # [B, H, V] float
    g: torch.Tensor,       # [B, H] float (log-space decay, NOT exp'd yet)
    beta: torch.Tensor,    # [B, H] float (sigmoid'd)
    state_buf: torch.Tensor,  # [max_slots, H, K, V] float32
    slot_indices: torch.Tensor,  # [B] int64
) -> torch.Tensor:
    """
    Batched single-step decode using Triton kernel.
    Returns output [B, H, V] and updates state_buf in-place.
    """
    B, H, K = q.shape
    V = v.shape[-1]

    output = torch.empty(B, H, V, dtype=q.dtype, device=q.device)

    BV = min(triton.next_power_of_2(V), 128)
    grid = (triton.cdiv(V, BV), B * H)

    gdn_decode_kernel[grid](
        q, k, v, g, beta, output,
        state_buf,
        slot_indices,
        B=B, H=H, K=K, V=V, BV=BV,
        stride_state_slot=state_buf.stride(0),
        stride_state_h=state_buf.stride(1),
        stride_state_k=state_buf.stride(2),
        num_warps=4,
        num_stages=1,
    )
    return output
