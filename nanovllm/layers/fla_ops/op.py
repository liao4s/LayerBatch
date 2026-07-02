"""
Triton op utilities ported from flash-linear-attention.
"""
import os
import triton
import triton.language as tl

from .utils import is_gather_supported

# Use standard tl.exp/log (fast_expf not available in all triton versions)
exp = tl.exp
log = tl.log
log2 = tl.log2

if not is_gather_supported:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather

if hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
elif hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
else:
    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None
