from dataclasses import dataclass, field
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    seq_ids: list[int] = field(default_factory=list)
    linear_attn_slot_indices: torch.Tensor | None = None  # [B] maps batch pos → buffer slot
    # Python-side metadata populated by prepare_decode for the LB scheduler.
    # `max_ctx_len`  = max context_lens this step (drives partition selection)
    # `total_tokens` = sum(context_lens) + bs (drives min/max-total-tokens gates)
    # Both are plain Python ints — no GPU sync needed to read them.
    max_ctx_len: int = 0
    total_tokens: int = 0
    # POD-Attention piggy-back GEMM: (A_input, B_weight, C_out) — set by LayerBatch
    # to schedule a GEMM tile concurrently with the next FA decode flash_attn launch.
    # Cleared automatically after one consumption.  Listed AFTER positional fields
    # to keep `set_context()` positional call sites working unchanged.
    pod_gemm: tuple | None = None
    fi_slot: str = "default"
    # Chunked prefill continuation flag.  When True, LA layers load their
    # recurrent + conv state from the per-slot buffer (populated by the
    # prior chunk) instead of starting from zero.  Set by ModelRunner
    # `_run_chunked_prefill` for chunks after the first.
    la_chunk_continuation: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, seq_ids=None,
                linear_attn_slot_indices=None, max_ctx_len=0, total_tokens=0,
                fi_slot="default", la_chunk_continuation=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables, seq_ids or [],
                       linear_attn_slot_indices, max_ctx_len, total_tokens,
                       pod_gemm=None, fi_slot=fi_slot,
                       la_chunk_continuation=la_chunk_continuation)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
