"""
Ported Triton kernels from flash-linear-attention (fla) for GatedDeltaNet.
- fused_recurrent_gated_delta_rule: Triton kernel for decode (single-step recurrent)
- chunk_gated_delta_rule: Triton kernel for prefill (chunk-based)
- l2norm_fwd: Triton L2 normalization
"""
from .fused_recurrent import fused_recurrent_gated_delta_rule
from .l2norm import l2norm_fwd

__all__ = [
    "fused_recurrent_gated_delta_rule",
    "l2norm_fwd",
]
