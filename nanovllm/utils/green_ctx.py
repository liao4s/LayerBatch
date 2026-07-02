"""CUDA Green Context wrapper for SM-disjoint dual-stream execution.

Splits the device's SMs into N partitions ("FA" / "LA" pairs), creates a CUstream
inside each Green Context, and exposes them as torch.cuda.ExternalStream so
PyTorch's `with torch.cuda.stream(...)` can target them.

Multi-partition support: `init_green_contexts_multi(...)` lets you register
several (fa_sm, la_sm) splits in the same process so that the model_runner can
pick a different SM allocation per decode step based on workload (max ctx_len).
The legacy `init_green_contexts(fa_sm, ...)` is preserved as a single-partition
shim that delegates to the multi API.

Why Green Context (and not just stream priorities or MPS):
- Hard SM partitioning means kernels in the FA partition cannot oversubscribe
  the SMs assigned to the LA partition (and vice versa).
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


# All Green-Context partitions registered in this process. Each entry:
#   {"fa_sm": int, "la_sm": int,
#    "fa_gctx": CUgreenCtx, "la_gctx": CUgreenCtx,
#    "fa_stream": torch.cuda.ExternalStream,
#    "la_stream": torch.cuda.ExternalStream}
_PARTITIONS: list[dict] = []
_DEVICE = -1
_CLEANUP_REGISTERED = False


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


def _ensure_primary_context(device_index: int):
    _C(_drv.cuInit(0), "cuInit")
    torch.cuda.init()
    torch.cuda.set_device(device_index)
    _ = torch.zeros(1, device='cuda')  # primary context creation


def _create_one_partition(fa_sm: int, device_index: int, non_blocking: bool) -> dict:
    """Build one (fa_gctx, la_gctx, fa_stream, la_stream) bundle for the given fa_sm.
       The LA partition is `total_sms - actual_fa_sm`. Driver may round fa_sm to the
       hardware's allowed multiple — actual counts are returned in the dict."""
    dev = _C(_drv.cuDeviceGet(device_index), "cuDeviceGet")
    total_res = _C(_drv.cuDeviceGetDevResource(
        dev, _drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    ), "cuDeviceGetDevResource")
    total_sms = total_res.sm.smCount
    fa_sm_clamped = max(8, min(fa_sm, total_sms - 8))

    out = _drv.cuDevSmResourceSplitByCount(1, total_res, 0, fa_sm_clamped)
    if out[0].value != 0:
        msg = _drv.cuGetErrorString(out[0])[1]
        raise RuntimeError(f"cuDevSmResourceSplitByCount(fa_sm={fa_sm_clamped}): {msg}")
    groups, _nb, remainder = out[1], out[2], out[3]
    fa_res, la_res = groups[0], remainder

    fa_actual = fa_res.sm.smCount
    la_actual = la_res.sm.smCount

    fa_desc = _C(_drv.cuDevResourceGenerateDesc([fa_res], 1), "fa_desc")
    la_desc = _C(_drv.cuDevResourceGenerateDesc([la_res], 1), "la_desc")
    flag = _drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
    fa_gctx = _C(_drv.cuGreenCtxCreate(fa_desc, dev, flag.value), "cuGreenCtxCreate(fa)")
    la_gctx = _C(_drv.cuGreenCtxCreate(la_desc, dev, flag.value), "cuGreenCtxCreate(la)")

    sflag = _drv.CUstream_flags.CU_STREAM_NON_BLOCKING.value if non_blocking else 0
    fa_h = _C(_drv.cuGreenCtxStreamCreate(fa_gctx, sflag, 0), "cuGreenCtxStreamCreate(fa)")
    la_h = _C(_drv.cuGreenCtxStreamCreate(la_gctx, sflag, 0), "cuGreenCtxStreamCreate(la)")

    fa_stream = torch.cuda.ExternalStream(int(fa_h), device=device_index)
    la_stream = torch.cuda.ExternalStream(int(la_h), device=device_index)

    return {
        "fa_sm": fa_actual, "la_sm": la_actual,
        "fa_gctx": fa_gctx, "la_gctx": la_gctx,
        "fa_stream": fa_stream, "la_stream": la_stream,
    }


def init_green_contexts_multi(partitions: list,
                              device_index: int = 0,
                              non_blocking: bool = True) -> list:
    """Create N independent Green-Context partitions for dynamic SM allocation.

    Args:
      partitions:   list of (fa_sm, la_sm_min) tuples — la_sm_min is informational
                    only; the LA partition gets `total_sms - actual_fa_sm` SMs.
                    Each entry produces an independent (fa_stream, la_stream) pair.
      device_index: physical CUDA device index (after CUDA_VISIBLE_DEVICES).
      non_blocking: streams are CU_STREAM_NON_BLOCKING.

    Returns:
      list of dicts (one per input partition), each with keys
        {fa_sm, la_sm, fa_gctx, la_gctx, fa_stream, la_stream}.

    Idempotent across calls on the same device: subsequent calls APPEND new
    partitions to the registry rather than tearing down existing ones.
    """
    global _DEVICE, _CLEANUP_REGISTERED, _PARTITIONS

    if not is_supported():
        raise RuntimeError(
            "Green Context unavailable: cuda-python or required API missing. "
            "Need cuda-python >= 12.4 against driver R535+."
        )

    if _DEVICE >= 0 and _DEVICE != device_index:
        raise RuntimeError(f"green_ctx already init'd on device {_DEVICE}, requested {device_index}")
    if _DEVICE < 0:
        _ensure_primary_context(device_index)
        _DEVICE = device_index

    new_parts = []
    for fa_sm, _la_sm_min in partitions:
        bundle = _create_one_partition(fa_sm, device_index, non_blocking)
        _PARTITIONS.append(bundle)
        new_parts.append(bundle)

    if not _CLEANUP_REGISTERED:
        atexit.register(_cleanup)
        _CLEANUP_REGISTERED = True

    return new_parts


def init_green_contexts(fa_sm: int, la_sm_min: int = 0,
                        device_index: int = 0,
                        non_blocking: bool = True):
    """Single-partition API kept for backwards compatibility.

    Returns (fa_stream, la_stream).  If green_ctx is already initialized, returns
    the FIRST registered partition's streams (matches old idempotent behaviour).
    """
    global _PARTITIONS
    if _PARTITIONS:
        if _DEVICE != device_index:
            raise RuntimeError("green_ctx already init'd on a different device")
        p = _PARTITIONS[0]
        return p["fa_stream"], p["la_stream"]
    parts = init_green_contexts_multi([(fa_sm, la_sm_min)],
                                      device_index=device_index,
                                      non_blocking=non_blocking)
    p = parts[0]
    return p["fa_stream"], p["la_stream"]


def get_streams():
    """Return the FIRST partition's (fa_stream, la_stream) for legacy callers.
       For multi-partition, use get_partitions()."""
    if not _PARTITIONS:
        return None, None
    p = _PARTITIONS[0]
    return p["fa_stream"], p["la_stream"]


def get_partitions() -> list:
    """Return the list of registered partition bundles."""
    return list(_PARTITIONS)


def get_sm_counts():
    """Legacy: (fa_sm, la_sm) of the FIRST partition. Both 0 before init."""
    if not _PARTITIONS:
        return 0, 0
    p = _PARTITIONS[0]
    return p["fa_sm"], p["la_sm"]


def _cleanup():
    global _PARTITIONS
    if _drv is None:
        return
    for p in _PARTITIONS:
        for k in ("fa_gctx", "la_gctx"):
            try:
                if p.get(k) is not None:
                    _drv.cuGreenCtxDestroy(p[k])
                    p[k] = None
            except Exception:
                pass
    _PARTITIONS = []
