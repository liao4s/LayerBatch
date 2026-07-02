"""FlashInfer-based prefill backend — Ampere-aware.

Design goals (this rev)
-----------------------
1. On Ampere (A100, A10, sm_80/86): NEVER touch flashinfer.  Auto-detect at
   startup and refuse to import the package, refuse to JIT, refuse to plan.
   Return None from every helper so callers fall back to the alternative
   attention path (flash_attn 2 if installed, else pure-torch SDPA).

2. On Hopper (H100, H20, sm_90+): flashinfer works fine.  Keep the same
   opt-in `--use-flashinfer-prefill` behaviour we shipped in the prior
   patch, with lazy import + arch-list gating.

3. Zero flashinfer JIT during init/warmup on Ampere.  The check runs BEFORE
   any wrapper class touch, so the JIT compile chain (which is what fails
   on old CUDA toolkits) never gets invoked.

Why the whole "no flashinfer on Ampere" stance
----------------------------------------------
Some flashinfer 0.6.x wheels ship a hardcoded JIT manifest that includes
Hopper-only kernels (`gdn_prefill_sm90`) whose PTX intrinsics
(``cuda::ptx::tensormap_replace_global_dim`` etc.) require CCCL >= 2.7 /
CUDA >= 12.5.  On an Ampere host with an older toolkit, the JIT dies at
first plan() call — even though we never actually use those kernels.  The
only reliable code-side fix is to not touch flashinfer at all on such
hosts.  On Hopper the toolkit is usually new enough for the Hopper kernels
to build cleanly, so we keep the flag available there.

If a user really wants flashinfer on Ampere (e.g. matched cubin install
that skips JIT), they can force it by exporting
``NANOVLLM_FORCE_FLASHINFER_ON_AMPERE=1`` before starting the server.
"""
from __future__ import annotations
import os
import warnings
import torch


# ---------------------------------------------------------------------------
# Ampere auto-detect (runs at first attribute access; safe on non-CUDA hosts)
# ---------------------------------------------------------------------------
def _current_cc():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        return None


def is_ampere() -> bool:
    """True if the current device is Ampere (major==8) — A100 (8.0), A10 (8.6)."""
    cc = _current_cc()
    return cc is not None and cc[0] == 8


def _force_flashinfer_on_ampere() -> bool:
    """Escape hatch for users who explicitly want to try flashinfer on A100
    (e.g. matched flashinfer-cubin install that skips JIT)."""
    return os.environ.get("NANOVLLM_FORCE_FLASHINFER_ON_AMPERE", "0") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Module-level state.  flashinfer is NEVER imported on Ampere without the
# escape-hatch env var.  On Hopper it is imported lazily on first use.
# ---------------------------------------------------------------------------
_USE_FLASHINFER_PREFILL: bool = False
_WORKSPACE: torch.Tensor | None = None
_WORKSPACE_BYTES: int = 128 * 1024 * 1024

_PREFILL_WRAPPER = None
_RAGGED_WRAPPER = None
_LB_PREFILL_WRAPPERS: dict = {}

_PLANNED_FOR_STEP: bool = False
_RAGGED_PLANNED: bool = False

_flashinfer_mod = None
_import_error: str | None = None
_preflight_ok: bool = False


# ---------------------------------------------------------------------------
# Arch-list gating for Hopper JIT.  Sets TORCH_CUDA_ARCH_LIST to running
# device's cap so JIT skips arches the device cannot run.
# ---------------------------------------------------------------------------
def _set_arch_list_to_running_device():
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return
    cc = _current_cc()
    if cc is None:
        return
    major, minor = cc
    cap = f"{major}.{minor}a" if major == 9 else f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = cap


def _lazy_import_flashinfer():
    """Import flashinfer, gated by Ampere check.  Returns None on Ampere
    (unless the escape hatch env var is set) or on import failure.

    We do NOT cache the Ampere-skip decision as a permanent import error so
    that re-checks after the escape-hatch env var flips see the new state.
    Only genuine import failures get cached.
    """
    global _flashinfer_mod, _import_error
    if _flashinfer_mod is not None:
        return _flashinfer_mod
    # Ampere gate re-evaluated every call so the escape hatch env var can flip.
    if is_ampere() and not _force_flashinfer_on_ampere():
        # Don't stash into _import_error — it's a runtime policy, not a load
        # failure.  This lets a later call with the env var set proceed.
        return None
    if _import_error is not None:
        return None
    _set_arch_list_to_running_device()
    try:
        import flashinfer as _fi
        _flashinfer_mod = _fi
        return _fi
    except Exception as e:
        _import_error = (
            f"flashinfer import failed: {type(e).__name__}: {e}. "
            "Recommended install: `pip install flashinfer-python==0.6.11.post2 "
            "flashinfer-cubin==0.6.11.post2`.  On Ampere you can also install "
            "`flash-attn>=2.5.0,<3` and skip --use-flashinfer-prefill entirely.")
        return None


def is_supported() -> bool:
    """True iff flashinfer is usable on this host (Ampere always False)."""
    return _lazy_import_flashinfer() is not None


def set_use_flashinfer_prefill(flag: bool):
    """Enable/disable the FlashInfer prefill backend globally.

    On Ampere this is a no-op with a warning — flashinfer is intentionally
    skipped to avoid JIT-compilation hazards.  On Hopper it triggers a
    preflight check that fails fast if the install is broken.
    """
    global _USE_FLASHINFER_PREFILL
    if not flag:
        _USE_FLASHINFER_PREFILL = False
        return
    if is_ampere() and not _force_flashinfer_on_ampere():
        warnings.warn(
            "--use-flashinfer-prefill requested but Ampere GPU (sm_80/86) "
            "detected. Ignoring and using flash_attn 2 / pure-torch SDPA "
            "fallback. To override set NANOVLLM_FORCE_FLASHINFER_ON_AMPERE=1.")
        _USE_FLASHINFER_PREFILL = False
        return
    ok, msg = preflight_check()
    if not ok:
        raise RuntimeError(
            "Cannot enable --use-flashinfer-prefill: " + msg + "\n"
            "Workarounds:\n"
            "  1. Install matching flashinfer wheels:\n"
            "       pip install flashinfer-python==0.6.11.post2 \\\n"
            "                   flashinfer-cubin==0.6.11.post2\n"
            "  2. Install flash-attn 2 and drop the flag:\n"
            "       pip install 'flash-attn>=2.5.0,<3'"
        )
    _USE_FLASHINFER_PREFILL = True


def get_use_flashinfer_prefill() -> bool:
    return _USE_FLASHINFER_PREFILL


def preflight_check() -> tuple[bool, str]:
    """Validate flashinfer usability.  Auto-fails on Ampere (returns False,
    reason).  Cheap when the module is already imported."""
    global _preflight_ok
    if _preflight_ok:
        return True, ""
    if is_ampere() and not _force_flashinfer_on_ampere():
        return False, (
            "Ampere GPU detected (sm_80/86) — flashinfer path disabled to "
            "avoid Hopper-kernel JIT hazards.  Use flash_attn 2 or the "
            "pure-torch SDPA fallback.")
    fi = _lazy_import_flashinfer()
    if fi is None:
        return False, _import_error or "flashinfer import failed"
    if not torch.cuda.is_available():
        return False, "CUDA is not available; FlashInfer requires a CUDA device."
    try:
        _ = fi.BatchPrefillWithPagedKVCacheWrapper
        _ = fi.BatchPrefillWithRaggedKVCacheWrapper
    except Exception as e:
        return False, (
            f"flashinfer wrapper class lookup failed: {type(e).__name__}: {e}. "
            "Your flashinfer install is incomplete; reinstall with the pin above.")
    _preflight_ok = True
    return True, ""


def _ensure_workspace(device: torch.device) -> torch.Tensor:
    global _WORKSPACE
    if _WORKSPACE is None or _WORKSPACE.device != device:
        _WORKSPACE = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _WORKSPACE


def _get_or_make_paged_wrapper(slot: str = "default"):
    global _PREFILL_WRAPPER
    fi = _lazy_import_flashinfer()
    if fi is None:
        raise RuntimeError("flashinfer not usable: " + (_import_error or ""))
    if slot == "default":
        if _PREFILL_WRAPPER is None:
            ws = _ensure_workspace(torch.device("cuda"))
            _PREFILL_WRAPPER = fi.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
        return _PREFILL_WRAPPER
    if slot not in _LB_PREFILL_WRAPPERS:
        ws = _ensure_workspace(torch.device("cuda"))
        _LB_PREFILL_WRAPPERS[slot] = fi.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="NHD")
    return _LB_PREFILL_WRAPPERS[slot]


def _get_or_make_ragged_wrapper():
    global _RAGGED_WRAPPER
    fi = _lazy_import_flashinfer()
    if fi is None:
        raise RuntimeError("flashinfer not usable: " + (_import_error or ""))
    if _RAGGED_WRAPPER is None:
        ws = _ensure_workspace(torch.device("cuda"))
        _RAGGED_WRAPPER = fi.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD")
    return _RAGGED_WRAPPER


# ---------------------------------------------------------------------------
# Metadata builder — pure-CPU, no flashinfer dependency.
# ---------------------------------------------------------------------------
def build_paged_metadata(seqlens_k, block_tables_list, page_size: int, device: torch.device):
    import numpy as np
    B = len(seqlens_k)
    assert B == len(block_tables_list), "seqlens / block_tables length mismatch"
    pages_per_req = [(L + page_size - 1) // page_size for L in seqlens_k]
    paged_kv_indptr = np.zeros(B + 1, dtype=np.int32)
    np.cumsum(pages_per_req, out=paged_kv_indptr[1:])
    total_pages = int(paged_kv_indptr[-1])

    flat_indices = np.empty(total_pages, dtype=np.int32)
    cursor = 0
    for L, bt in zip(pages_per_req, block_tables_list):
        flat_indices[cursor:cursor + L] = np.asarray(bt[:L], dtype=np.int32)
        cursor += L

    last_page_len = np.empty(B, dtype=np.int32)
    for i, L in enumerate(seqlens_k):
        rem = L % page_size
        last_page_len[i] = rem if rem != 0 else page_size

    return dict(
        paged_kv_indptr_cpu=torch.from_numpy(paged_kv_indptr).contiguous(),
        paged_kv_indices_gpu=torch.from_numpy(flat_indices).to(device, non_blocking=True),
        paged_kv_last_page_len_cpu=torch.from_numpy(last_page_len).contiguous(),
    )


# ---------------------------------------------------------------------------
# Plan / run — paged path.  Returns False fast on Ampere (never JIT).
# ---------------------------------------------------------------------------
def plan_prefill(qo_indptr_cpu, metadata, num_qo_heads, num_kv_heads, head_dim,
                 page_size, sm_scale, q_dtype, kv_dtype, causal=True, slot="default") -> bool:
    global _PLANNED_FOR_STEP
    fi = _lazy_import_flashinfer()
    if fi is None:
        return False
    wrapper = _get_or_make_paged_wrapper(slot)
    wrapper.plan(
        qo_indptr=qo_indptr_cpu,
        paged_kv_indptr=metadata["paged_kv_indptr_cpu"],
        paged_kv_indices=metadata["paged_kv_indices_gpu"],
        paged_kv_last_page_len=metadata["paged_kv_last_page_len_cpu"],
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        page_size=page_size, causal=causal, sm_scale=sm_scale,
        q_data_type=q_dtype, kv_data_type=kv_dtype,
    )
    if slot == "default":
        _PLANNED_FOR_STEP = True
    return True


def reset_planned_flag():
    global _PLANNED_FOR_STEP
    _PLANNED_FOR_STEP = False


def run_prefill(q, k_cache, v_cache, slot: str = "default"):
    return _get_or_make_paged_wrapper(slot).run(q, (k_cache, v_cache))


def is_planned(slot: str = "default") -> bool:
    if slot == "default":
        return _PLANNED_FOR_STEP
    return slot in _LB_PREFILL_WRAPPERS


# ---------------------------------------------------------------------------
# Plan / run — ragged path.
# ---------------------------------------------------------------------------
def plan_ragged(qo_indptr_cpu, kv_indptr_cpu, num_qo_heads, num_kv_heads,
                head_dim, sm_scale, q_dtype, kv_dtype, causal=True) -> bool:
    global _RAGGED_PLANNED
    fi = _lazy_import_flashinfer()
    if fi is None:
        return False
    wrapper = _get_or_make_ragged_wrapper()
    wrapper.plan(
        qo_indptr=qo_indptr_cpu, kv_indptr=kv_indptr_cpu,
        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim, head_dim_vo=head_dim,
        causal=causal, sm_scale=sm_scale,
        q_data_type=q_dtype, kv_data_type=kv_dtype,
    )
    _RAGGED_PLANNED = True
    return True


def run_ragged(q, k, v):
    return _get_or_make_ragged_wrapper().run(q, k, v)


def is_ragged_planned() -> bool:
    return _RAGGED_PLANNED


def reset_ragged_planned():
    global _RAGGED_PLANNED
    _RAGGED_PLANNED = False
