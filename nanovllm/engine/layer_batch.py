"""Layer-Batch pipelined decode for hybrid LA/FA models (Qwen3.5).

Strict pipeline design (LA-stream / FA-stream)
---------------------------------------------
At any wall-clock instant the GPU has at most two units in flight:
  * one **LA-block** (3 consecutive LA layers) on the LA stream
  * one **FA layer** (1 layer)                on the FA stream
Both streams sit on **disjoint Green-Context SM partitions** (default 60 %
SMs for LA, 40 % SMs for FA on H20 → ~47/31).  The two units belong to two
different *groups* of requests (nano-batches) so they share no data — they
truly run in parallel on disjoint SMs.

Scheduling
----------
With Qwen3.5 (24 layers, every 4th is FA) the model decomposes into 12
*units* that strictly alternate LA, FA, LA, FA, …  A decode step with B
requests is split into Group-A = `[:B1]` and Group-B = `[B1:]` (default
B1=B/2).  Both groups walk through the same 12-unit sequence, **but Group-B
lags Group-A by exactly one unit**, so at every step t∈{1..n-1} one of the
two groups is doing LA and the other is doing FA — the two stream
partitions are always doing different work types.

CUDA Graph capture — anchor-on-LA design
----------------------------------------
`torch.cuda.graph` captures kernels by sniffing `cudaStreamWaitEvent`
calls between a "capturing" stream and other streams.  A long-standing
PyTorch issue is that this propagation is **unreliable when the capture
origin is a regular stream and the joining stream is a Green-Context
`torch.cuda.ExternalStream`** — some of the cross-stream wait edges are
silently dropped, and at replay one stream's kernels race ahead of the
other's, producing garbled output on a fraction of decode steps.

The fix in this module is structural: we make `la_stream` itself the
capture origin (passed in from `model_runner.capture_layer_batch_graphs`
via `torch.cuda.graph(..., stream=la_stream)`), and we keep **all work
inside the captured region on `la_stream` or `fa_stream`** — no work ever
crosses into the regular default stream while capture is active.  The
only cross-stream sync that needs to land in the graph is between the
two ExternalStreams (LA ↔ FA), which capture handles correctly.

Eager mode is unchanged: when no capture is active, `anchor_stream` is
just the current default stream, and the green streams join through
`wait_stream` exactly as before.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from nanovllm.utils.context import Context, get_context
import nanovllm.utils.context as _ctxmod


def build_units(layer_types: List[str]) -> List[Tuple[str, List[int]]]:
    """Group consecutive same-type layers into units. Returns list of (kind, [layer_idxs])."""
    units: List[Tuple[str, List[int]]] = []
    cur_kind: Optional[str] = None
    cur: List[int] = []
    for i, t in enumerate(layer_types):
        kind = "LA" if t == "linear_attention" else "FA"
        if kind != cur_kind:
            if cur:
                units.append((cur_kind, cur))  # type: ignore[arg-type]
            cur_kind = kind
            cur = [i]
        else:
            cur.append(i)
    if cur:
        units.append((cur_kind, cur))  # type: ignore[arg-type]
    return units


def _slice_decode_context(B1: int) -> Tuple[Context, Context]:
    """Slice the current decode context into Group-A ([:B1]) and Group-B ([B1:])."""
    src = get_context()
    bt = src.block_tables
    cl = src.context_lens
    sm = src.slot_mapping
    li = src.linear_attn_slot_indices
    ctxA = Context(
        is_prefill=False,
        slot_mapping=sm[:B1] if sm is not None else None,
        context_lens=cl[:B1] if cl is not None else None,
        block_tables=bt[:B1] if bt is not None else None,
        linear_attn_slot_indices=(li[:B1] if li is not None else None),
    )
    ctxB = Context(
        is_prefill=False,
        slot_mapping=sm[B1:] if sm is not None else None,
        context_lens=cl[B1:] if cl is not None else None,
        block_tables=bt[B1:] if bt is not None else None,
        linear_attn_slot_indices=(li[B1:] if li is not None else None),
    )
    return ctxA, ctxB


def _set_ctx(ctx: Context):
    """Replace thread-local _CONTEXT with the given Context object (no copy)."""
    _ctxmod._CONTEXT = ctx


def run_layer_batch_decode(
    language_model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    layer_types: List[str],
    B1: int,
    fa_stream: torch.cuda.Stream,
    la_stream: torch.cuda.Stream,
) -> torch.Tensor:
    """One decode step using the strict LA-stream / FA-stream pipeline.

    Behaviour:
      - Eager:           anchor_stream = current default stream; fa/la join via wait_stream;
                         drain back to default at the end.
      - Inside `torch.cuda.graph(..., stream=la_stream)` capture:
                         anchor_stream = la_stream (the capture origin); fa_stream joins
                         via wait_stream; final concat + RMSNorm runs on la_stream so the
                         captured graph never touches a regular non-ExternalStream.

    Returns:
      hidden_states post-norm, shape [B, H].
    """
    units = build_units(layer_types)
    n_units = len(units)
    if n_units < 2:
        raise RuntimeError(f"layer-batch needs >=2 units, got {n_units}")

    ctxA, ctxB = _slice_decode_context(B1)
    saved_ctx = get_context()

    # `anchor_stream` is whatever is current at entry. In capture mode the caller
    # has set this to `la_stream` via `torch.cuda.graph(..., stream=la_stream)`,
    # which means `embed_tokens`, the final concat and the final RMSNorm all run
    # on `la_stream` and the captured graph never crosses into a non-Green-Ctx
    # stream.
    anchor_stream = torch.cuda.current_stream()

    # ---- phase 1: embedding on anchor stream ----
    h_full = language_model.embed_tokens(input_ids)

    # Bring fa_stream and la_stream into sync (and capture state) with anchor,
    # and record h_full / positions on both green streams so the caching
    # allocator does not free their memory while the green streams are still
    # consuming it.  See note in phase 2 about why record_stream is required
    # for ExternalStream.
    if fa_stream is not anchor_stream:
        fa_stream.wait_stream(anchor_stream)
        h_full.record_stream(fa_stream)
        positions.record_stream(fa_stream)
    if la_stream is not anchor_stream:
        la_stream.wait_stream(anchor_stream)
        h_full.record_stream(la_stream)
        positions.record_stream(la_stream)

    hA = h_full[:B1]
    hB = h_full[B1:]
    posA = positions[:B1]
    posB = positions[B1:]
    rA: Optional[torch.Tensor] = None
    rB: Optional[torch.Tensor] = None

    layers = language_model.layers

    # Track the LAST stream each group used. When the next unit lives on a
    # different stream we issue `stream.wait_stream(prev_stream)` BEFORE the
    # `with torch.cuda.stream(stream)` block — keeping the cross-stream sync
    # outside the stream-context manager avoids confusing the capture engine
    # about which stream is "current" while it logs the wait edge.
    last_stream_A: torch.cuda.Stream = anchor_stream
    last_stream_B: torch.cuda.Stream = anchor_stream

    # ---- phase 2: pipelined unit-by-unit schedule ----
    for t in range(n_units + 1):

        # ---- Group-A: unit t ----
        if t < n_units:
            kind, layer_ids = units[t]
            stream = la_stream if kind == "LA" else fa_stream
            if stream is not last_stream_A:
                stream.wait_stream(last_stream_A)
                # Tell the caching allocator that hA / rA — created on the
                # previous unit's stream — are now consumed on `stream`.
                # Without this, the allocator can free the underlying memory
                # the moment last_stream_A's last kernel finishes and recycle
                # it for a fresh allocation on fa_stream BEFORE the new
                # kernel here actually reads it (race that the captured
                # graph then preserves into a nondeterministic replay).
                hA.record_stream(stream)
                if rA is not None:
                    rA.record_stream(stream)
            _set_ctx(ctxA)
            with torch.cuda.stream(stream):
                for li in layer_ids:
                    hA, rA = layers[li](posA, hA, rA)
            last_stream_A = stream

        # ---- Group-B: unit (t-1) ----
        if 1 <= t <= n_units:
            u = t - 1
            kind, layer_ids = units[u]
            stream = la_stream if kind == "LA" else fa_stream
            if stream is not last_stream_B:
                stream.wait_stream(last_stream_B)
                hB.record_stream(stream)
                if rB is not None:
                    rB.record_stream(stream)
            _set_ctx(ctxB)
            with torch.cuda.stream(stream):
                for li in layer_ids:
                    hB, rB = layers[li](posB, hB, rB)
            last_stream_B = stream

    # ---- phase 3: drain both groups back to anchor → final residual + RMSNorm ----
    if last_stream_A is not anchor_stream:
        anchor_stream.wait_stream(last_stream_A)
    if last_stream_B is not anchor_stream and last_stream_B is not last_stream_A:
        anchor_stream.wait_stream(last_stream_B)
    _set_ctx(saved_ctx)

    hA = hA + rA
    hB = hB + rB
    hidden = torch.cat([hA, hB], dim=0)
    hidden = language_model.norm(hidden)
    return hidden
