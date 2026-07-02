import torch
from torch import nn
import triton
import triton.language as tl

# flash_attn is optional.  On A100 the pip-installed wheel might be FA3
# (Hopper-only) which imports OK but fails at kernel launch, or might not
# be present at all.  We guard the import so the engine can still boot on
# Ampere and fall back to pure-Torch SDPA.
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - environment without flash-attn
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    _HAS_FLASH_ATTN = False

from nanovllm.utils.context import get_context
from nanovllm.layers import flashinfer_attn as _fi
from nanovllm.layers import sdpa_fallback as _sdpa

# ---------------------------------------------------------------------------
# POD-Attention decode hook.
# When `_USE_POD_DECODE` is True (set by ModelRunner under the right CLI flag),
# the decode forward path uses the Triton paged-flash-attention kernel from
# `nanovllm.layers.pod_kernels`, which can also piggy-back a GEMM (POD-style
# CTA-level fusion).  The piggy-back GEMM is supplied via the per-Context
# `pod_gemm` field; if absent, the kernel runs attention only.
# ---------------------------------------------------------------------------
_USE_POD_DECODE = False
# Tile config resolved by ModelRunner._resolve_dynamic_defaults().  Defaults are
# safe but model-agnostic; ModelRunner pushes the per-model values via
# `set_pod_runtime_cfg(...)` before warmup so CUDA-Graph captures use them.
_POD_RUNTIME_CFG = dict(num_kv_splits=16, block_n=64, block_h=16, num_warps=4, num_stages=3)

def set_use_pod_decode(flag: bool):
    global _USE_POD_DECODE
    _USE_POD_DECODE = bool(flag)
def get_use_pod_decode() -> bool:
    return _USE_POD_DECODE

def set_pod_runtime_cfg(cfg: dict):
    """Override the POD kernel tile config used by all `Attention.forward()` decode calls."""
    global _POD_RUNTIME_CFG
    _POD_RUNTIME_CFG = dict(_POD_RUNTIME_CFG)
    _POD_RUNTIME_CFG.update({k: v for k, v in cfg.items() if v is not None})



@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            slot = getattr(context, "fi_slot", "default")
            # Path priority for prefill:
            #   1. FlashInfer paged wrapper (if planned for this step and
            #      the runner is on Hopper — Ampere never plans it).
            #   2. FlashInfer ragged wrapper (same conditions).
            #   3. Legacy flash_attn_varlen_func — used when flash-attn is
            #      importable AND flashinfer isn't (or wasn't planned).
            #      This is the fast path on Hopper without --use-flashinfer,
            #      and on Ampere when flash-attn 2 is installed.
            #   4. Pure-torch SDPA fallback — used on Ampere hosts that have
            #      NEITHER a working flash-attn NOR flashinfer.  Slow but
            #      universal, so the engine at least runs.
            if _fi.get_use_flashinfer_prefill() and _fi.is_planned(slot):
                o = _fi.run_prefill(q, k_cache, v_cache, slot=slot)
            elif _fi.get_use_flashinfer_prefill() and _fi.is_ragged_planned():
                o = _fi.run_ragged(q, k, v)
            elif _HAS_FLASH_ATTN:
                if context.block_tables is not None:    # prefix cache
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:
                # SDPA fallback.  Uses paged K/V read when block_tables is
                # populated; otherwise reads from local k, v tensors.
                block_size = k_cache.shape[1] if k_cache.numel() else 256
                o = _sdpa.sdpa_prefill_varlen(
                    q, k, v,
                    cu_seqlens_q=context.cu_seqlens_q,
                    cu_seqlens_k=context.cu_seqlens_k,
                    scale=self.scale,
                    block_tables=context.block_tables,
                    k_cache=k_cache if context.block_tables is not None else None,
                    v_cache=v_cache if context.block_tables is not None else None,
                    block_size=block_size,
                )
        else:    # decode
            if _USE_POD_DECODE:
                # POD path: paged Triton flash-attn decode, with optional GEMM piggy-back.
                # `context.pod_gemm` is a tuple (A, B, C_out) when the runner has scheduled
                # a GEMM to be fused with this flash_attn launch; else the kernel runs
                # attention only (degenerate GEMM grid).
                from nanovllm.layers.pod_kernels import pod_fused_attn_gemm, pod_attn_only

                # Lazy-allocate pre-sized scratch buffers.  Allocated ONCE during
                # warmup (eager) so that subsequent CUDA-Graph capture of the
                # decode loop sees stable static buffers (no in-graph allocations).
                B_, H_, D_ = q.shape
                NUM_KV_SPLITS = int(_POD_RUNTIME_CFG.get("num_kv_splits", 16))
                BLOCK_N = int(_POD_RUNTIME_CFG.get("block_n", 64))
                BLOCK_H = int(_POD_RUNTIME_CFG.get("block_h", 16))
                if (not hasattr(self, "_pod_mid_o")
                        or self._pod_mid_o is None
                        or self._pod_mid_o.shape[0] < B_
                        or self._pod_mid_o.shape[2] != NUM_KV_SPLITS):
                    self._pod_mid_o = torch.empty(B_, H_, NUM_KV_SPLITS, D_ + 1,
                                                  dtype=torch.float32, device=q.device)
                    self._pod_out = torch.empty(B_, H_, D_,
                                                dtype=q.dtype, device=q.device)
                mid_o = self._pod_mid_o[:B_]
                out_buf = self._pod_out[:B_]

                gemm = getattr(context, "pod_gemm", None)
                if gemm is not None:
                    A, W, C_out = gemm
                    o = pod_fused_attn_gemm(q, k_cache, v_cache,
                                             context.block_tables, context.context_lens,
                                             self.scale, A, W, C_out,
                                             num_kv_splits=NUM_KV_SPLITS,
                                             block_n=BLOCK_N, block_h=BLOCK_H,
                                             mid_o=mid_o, out=out_buf)
                    context.pod_gemm = None
                else:
                    o = pod_attn_only(q, k_cache, v_cache,
                                       context.block_tables, context.context_lens,
                                       self.scale,
                                       num_kv_splits=NUM_KV_SPLITS,
                                       block_n=BLOCK_N, block_h=BLOCK_H,
                                       mid_o=mid_o, out=out_buf)
                # Match flash_attn_with_kvcache return shape: [B, 1, H, D]
                o = o.unsqueeze(1)
            else:
                if _HAS_FLASH_ATTN:
                    o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                                cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                                softmax_scale=self.scale, causal=True)
                else:
                    # No flash-attn: try the POD Triton kernel first (works
                    # on Ampere + Hopper).  If that path is unavailable for
                    # any reason, fall back to pure-torch SDPA decode.
                    try:
                        from nanovllm.layers.pod_kernels import pod_attn_only
                        B_, H_, D_ = q.shape
                        NUM_KV_SPLITS = int(_POD_RUNTIME_CFG.get("num_kv_splits", 16))
                        BLOCK_N = int(_POD_RUNTIME_CFG.get("block_n", 64))
                        BLOCK_H = int(_POD_RUNTIME_CFG.get("block_h", 16))
                        if (not hasattr(self, "_pod_mid_o")
                                or self._pod_mid_o is None
                                or self._pod_mid_o.shape[0] < B_
                                or self._pod_mid_o.shape[2] != NUM_KV_SPLITS):
                            self._pod_mid_o = torch.empty(B_, H_, NUM_KV_SPLITS, D_ + 1,
                                                          dtype=torch.float32, device=q.device)
                            self._pod_out = torch.empty(B_, H_, D_, dtype=q.dtype, device=q.device)
                        mid_o = self._pod_mid_o[:B_]
                        out_buf = self._pod_out[:B_]
                        o = pod_attn_only(q, k_cache, v_cache,
                                          context.block_tables, context.context_lens,
                                          self.scale,
                                          num_kv_splits=NUM_KV_SPLITS,
                                          block_n=BLOCK_N, block_h=BLOCK_H,
                                          mid_o=mid_o, out=out_buf)
                        o = o.unsqueeze(1)
                    except Exception:
                        # Universal SDPA fallback (slow but always works).
                        block_size = k_cache.shape[1]
                        o = _sdpa.sdpa_decode_paged(q, k_cache, v_cache,
                                                    context.block_tables,
                                                    context.context_lens,
                                                    self.scale,
                                                    block_size=block_size)
        return o
