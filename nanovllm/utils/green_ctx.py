"""CUDA Green Context wrapper for SM-disjoint dual-stream execution.

Splits the device's SMs into 2 partitions ("FA" and "LA"), creates a CUstream
inside each Green Context, and exposes them as torch.cuda.ExternalStream so
PyTorch's `with torch.cuda.stream(...)` can target them.

Why Green Context (and not just stream priorities or MPS):
- Hard SM partitioning means kernels in the FA partition cannot oversubscribe
  the SMs assigned to the LA partition (and vice versa). Two streams placed in
  the same context would otherwise round-robin all SMs and could starve one
  another or thrash L2.
- This lets us co-execute two kernels of fundamentally different bottleneck
  profile (memory-bound paged-attention decode vs compute-bound recurrent
  Triton kernel) without the GPU's grid scheduler reshuffling resources mid-flight.
"""
from __future__ import annotations

import atexit
import torch

try:
    from cuda.bindings import driver as _drv  # noqa: F401
    _HAS_CUDA_PYTHON = True
except Exception:  # pragma: no cover
    _drv = None
    _HAS_CUDA_PYTHON = False


_INITIALIZED = False
_FA_GCTX = None
_LA_GCTX = None
_FA_STREAM = None
_LA_STREAM = None
_FA_SM = 0
_LA_SM = 0
_DEVICE = -1


def _C(ret, name=""):
    err = ret[0]
    if err.value != 0:
        msg = _drv.cuGetErrorString(err)[1]
        raise RuntimeError(f"CUDA driver error in {name}: {err}, {msg}")
    rest = ret[1:]
    return rest if len(rest) != 1 else rest[0]


def is_supported() -> bool:
    """Report whether Green Context APIs are present in this build of cuda-python."""
    if not _HAS_CUDA_PYTHON:
        return False
    needed = ("cuGreenCtxCreate", "cuDevSmResourceSplitByCount",
              "cuDevResourceGenerateDesc", "cuGreenCtxStreamCreate",
              "cuDeviceGetDevResource")
    return all(hasattr(_drv, n) for n in needed)


def init_green_contexts(fa_sm: int, la_sm_min: int = 0,
                         device_index: int = 0,
                         non_blocking: bool = True):
    """Create FA/LA Green Contexts and wrap their streams as torch streams.

    Args:
      fa_sm:        target SM count for the FA partition (driver may round).
      la_sm_min:    informational only (the LA path uses the remainder).
      device_index: physical device index (after CUDA_VISIBLE_DEVICES).
      non_blocking: stream is CU_STREAM_NON_BLOCKING (does not synchronize
                    against the legacy default stream).

    Returns:
      (fa_stream, la_stream) — torch.cuda.ExternalStream
    """
    global _INITIALIZED, _FA_GCTX, _LA_GCTX, _FA_STREAM, _LA_STREAM
    global _FA_SM, _LA_SM, _DEVICE

    if not is_supported():
        raise RuntimeError(
            "Green Context unavailable: cuda-python or required API missing. "
            "Need cuda-python >= 12.4 against driver R535+."
        )

    if _INITIALIZED:
        if _DEVICE != device_index:
            raise RuntimeError("green_ctx already init'd on a different device")
        return _FA_STREAM, _LA_STREAM

    _C(_drv.cuInit(0), "cuInit")
    torch.cuda.init()
    torch.cuda.set_device(device_index)
    _ = torch.zeros(1, device='cuda')  # ensure primary context exists

    dev = _C(_drv.cuDeviceGet(device_index), "cuDeviceGet")
    total_res = _C(_drv.cuDeviceGetDevResource(
        dev, _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    ), "cuDeviceGetDevResource")
    total_sms = total_res.sm.smCount
    fa_sm = max(8, min(fa_sm, total_sms - 8))

    out = _drv.cuDevSmResourceSplitByCount(1, total_res, 0, fa_sm)
    if out[0].value != 0:
        msg = _drv.cuGetErrorString(out[0])[1]
        raise RuntimeError(f"cuDevSmResourceSplitByCount: {msg}")
    groups, _nb, remainder = out[1], out[2], out[3]
    fa_res, la_res = groups[0], remainder

    _FA_SM = fa_res.sm.smCount
    _LA_SM = la_res.sm.smCount

    fa_desc = _C(_drv.cuDevResourceGenerateDesc([fa_res], 1), "fa_desc")
    la_desc = _C(_drv.cuDevResourceGenerateDesc([la_res], 1), "la_desc")
    flag = _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    _FA_GCTX = _C(_drv.cuGreenCtxCreate(fa_desc, dev, flag.value), "cuGreenCtxCreate(fa)")
    _LA_GCTX = _C(_drv.cuGreenCtxCreate(la_desc, dev, flag.value), "cuGreenCtxCreate(la)")

    sflag = _drv.CUstream_flags.CU_STREAM_NON_BLOCKING.value if non_blocking else 0
    fa_h = _C(_drv.cuGreenCtxStreamCreate(_FA_GCTX, sflag, 0), "cuGreenCtxStreamCreate(fa)")
    la_h = _C(_drv.cuGreenCtxStreamCreate(_LA_GCTX, sflag, 0), "cuGreenCtxStreamCreate(la)")

    _FA_STREAM = torch.cuda.ExternalStream(int(fa_h), device=device_index)
    _LA_STREAM = torch.cuda.ExternalStream(int(la_h), device=device_index)

    _DEVICE = device_index
    _INITIALIZED = True
    atexit.register(_cleanup)
    return _FA_STREAM, _LA_STREAM


def get_streams():
    """Return (fa_stream, la_stream) or (None, None) before init."""
    return _FA_STREAM, _LA_STREAM


def get_sm_counts():
    """Return (fa_sm_count, la_sm_count). Both 0 before init."""
    return _FA_SM, _LA_SM


def _cleanup():
    global _FA_GCTX, _LA_GCTX
    try:
        if _FA_GCTX is not None and _drv is not None:
            _drv.cuGreenCtxDestroy(_FA_GCTX); _FA_GCTX = None
    except Exception:
        pass
    try:
        if _LA_GCTX is not None and _drv is not None:
            _drv.cuGreenCtxDestroy(_LA_GCTX); _LA_GCTX = None
    except Exception:
        pass
