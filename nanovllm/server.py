"""
OpenAI-compatible API server for nanovllm.

Usage:
    python -m nanovllm.server --model /path/to/model [--host 0.0.0.0] [--port 8000]

Endpoints:
    POST /v1/chat/completions   — Chat completions (streaming & non-streaming)
    POST /v1/completions        — Text completions (streaming & non-streaming)
    GET  /v1/models             — List available models
    GET  /health                — Health check
    POST /shutdown              — Gracefully shutdown the server and all processes
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import threading
import uuid
import gc
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

def _resolve_ignore_eos(request) -> bool:
    """Resolve ignore_eos from OpenAI-compatible body. Supports top-level
    `ignore_eos` and `nvext.ignore_eos` (Triton NIM convention). Returns False
    when neither is specified."""
    val = getattr(request, "ignore_eos", None)
    if val is None:
        nvext = getattr(request, "nvext", None) or {}
        val = nvext.get("ignore_eos") if isinstance(nvext, dict) else None
    return bool(val) if val is not None else False

from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


# ============================================================
# Pydantic models — OpenAI-compatible request/response
# ============================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[List[str], str]] = None
    chat_template_kwargs: Optional[dict] = None
    ignore_eos: Optional[bool] = None
    nvext: Optional[dict] = None

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str]] = ""
    temperature: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[List[str], str]] = None
    ignore_eos: Optional[bool] = None
    nvext: Optional[dict] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

class ChatCompletionStreamDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = None

class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionStreamChoice]


# ============================================================
# Request tracking
# ============================================================

@dataclass
class PendingRequest:
    request_id: str
    seq_id: int
    prompt_tokens: int
    loop: asyncio.AbstractEventLoop
    # For non-streaming: resolved when generation completes
    future: asyncio.Future
    # For streaming: tokens pushed here incrementally
    token_queue: Optional[asyncio.Queue] = None
    stream: bool = False
    # Final completion_tokens count (set by worker just before sentinel)
    completion_tokens_final: int = 0


# ============================================================
# AsyncEngineWrapper — bridges sync engine with async server
# ============================================================

class AsyncEngineWrapper:
    """Wraps LLMEngine and runs the step loop in a background thread."""

    def __init__(self, model_path: str, served_model_name: str | None = None, **engine_kwargs):
        self.engine = LLMEngine(model_path, **engine_kwargs)
        self.tokenizer = self.engine.tokenizer
        self.model_name = served_model_name or model_path.rstrip("/").split("/")[-1]

        # seq_id -> PendingRequest
        self._pending: dict[int, PendingRequest] = {}
        self._lock = threading.Lock()

        # Track token counts for streaming incremental decode.
        # _seq_prev_text holds the text we have ALREADY emitted to the client;
        # used to compute incremental diffs when re-decoding the cumulative
        # token sequence (which is the only way to handle multi-byte UTF-8
        # tokens correctly — see _stable_decoded_text below).
        self._seq_prev_tokens: dict[int, int] = {}
        self._seq_prev_text: dict[int, str] = {}

        # Background engine loop
        self._running = True
        self._has_work = threading.Event()
        self._thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        """Thoroughly clean up all engine resources, child processes, and GPU memory."""
        logger.info("Shutting down engine...")
        self._running = False
        self._has_work.set()  # Wake up the engine loop so it can exit

        # Wait for engine loop thread to finish
        if self._thread.is_alive():
            self._thread.join(timeout=15)
            if self._thread.is_alive():
                logger.warning("Engine loop thread did not exit cleanly")

        # Resolve any remaining pending futures with cancellation
        with self._lock:
            for pending in self._pending.values():
                if not pending.future.done():
                    pending.loop.call_soon_threadsafe(
                        pending.future.cancel
                    )
                if pending.stream and pending.token_queue is not None:
                    try:
                        pending.loop.call_soon_threadsafe(
                            pending.token_queue.put_nowait, None
                        )
                    except Exception:
                        pass
            self._pending.clear()
            self._seq_prev_tokens.clear()

        # Shut down the LLM engine (which cleans up model_runner, dist, shm, child processes)
        try:
            self.engine.exit()
        except Exception as e:
            logger.warning("Error during engine exit: %s", e)

        # Release references to allow GC
        del self.engine
        del self.tokenizer

        # Force GPU memory release
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        # Force garbage collection
        gc.collect()
        logger.info("Engine shutdown complete.")

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        stream: bool = False,
    ) -> PendingRequest:
        """Add a request to the engine. Returns a PendingRequest with future/queue.

        Raises ValueError if the prompt exceeds max_model_len.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # Tokenize if needed
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt)
        else:
            prompt_ids = prompt

        # Pre-check prompt length before touching the engine.
        # LLMEngine.add_request() also validates, but checking here gives
        # the server layer a chance to return HTTP 400 instead of crashing.
        max_model_len = self.engine.config.max_model_len
        if len(prompt_ids) > max_model_len:
            raise ValueError(
                f"Prompt too long: {len(prompt_ids)} tokens exceeds "
                f"max_model_len={max_model_len}. Please reduce the prompt length."
            )

        # Add to engine (creates Sequence internally)
        self.engine.add_request(prompt_ids, sampling_params)

        # The seq_id of the just-added sequence is the last one in waiting queue
        seq = self.engine.scheduler.waiting[-1]
        seq_id = seq.seq_id

        token_queue = asyncio.Queue() if stream else None

        pending = PendingRequest(
            request_id=request_id,
            seq_id=seq_id,
            prompt_tokens=len(prompt_ids),
            loop=loop,
            future=future,
            token_queue=token_queue,
            stream=stream,
        )

        with self._lock:
            self._pending[seq_id] = pending
            self._seq_prev_tokens[seq_id] = 0

        # Wake up the engine loop
        self._has_work.set()
        return pending

    def _engine_loop(self):
        """Background thread: continuously steps the engine."""
        # Set Triton allocator in this thread (ContextVar is per-thread in Triton 3.6+)
        try:
            import triton, torch
            from triton.runtime._allocation import Allocator
            class _TorchAllocator(Allocator):
                def __call__(self, size, align, stream):
                    return torch.empty(size, dtype=torch.uint8, device="cuda:0").data_ptr()
            triton.set_allocator(_TorchAllocator())
        except Exception:
            pass
        while self._running:
            # Wait until there's work
            self._has_work.wait(timeout=0.05)

            if self.engine.is_finished():
                self._has_work.clear()
                continue

            try:
                outputs, _ = self.engine.step()
            except Exception as e:
                # Log the error for debugging
                import traceback
                logger.error("Error in engine step(): %s", e)
                traceback.print_exc()
                # Resolve all pending futures with the error
                with self._lock:
                    for pending in self._pending.values():
                        if not pending.future.done():
                            pending.loop.call_soon_threadsafe(
                                pending.future.set_exception, e
                            )
                        # Also send sentinel to streaming queues to unblock clients
                        if pending.stream and pending.token_queue is not None:
                            pending.loop.call_soon_threadsafe(
                                pending.token_queue.put_nowait, None
                            )
                    self._pending.clear()
                    self._seq_prev_tokens.clear()
                continue

            # Check for streaming updates: push incremental tokens
            with self._lock:
                # For streaming requests, check all running sequences for new tokens
                for seq in list(self.engine.scheduler.running):
                    sid = seq.seq_id
                    pending = self._pending.get(sid)
                    if pending is None or not pending.stream:
                        continue
                    cur_count = seq.num_completion_tokens
                    prev_count = self._seq_prev_tokens.get(sid, 0)
                    if cur_count > prev_count:
                        # CRITICAL: do NOT decode just the new tokens. A multi-
                        # byte UTF-8 character (CJK, emoji) can span two or more
                        # tokens whose bytes split the codepoint; decoding such
                        # a partial token alone yields U+FFFD ('\uFFFD'). The
                        # only correct streaming protocol is cumulative decode +
                        # incremental diff, deferring any trailing replacement
                        # chars (incomplete prefix) until later tokens complete
                        # the codepoint.
                        all_ids = seq.completion_token_ids[:cur_count]
                        full_text = self.tokenizer.decode(
                            all_ids, skip_special_tokens=True)
                        # Strip trailing U+FFFD (incomplete UTF-8 char) — those
                        # bytes will resolve once more tokens arrive.
                        stable = full_text.rstrip("\uFFFD")
                        prev_emitted = self._seq_prev_text.get(sid, "")
                        if len(stable) > len(prev_emitted):
                            new_chunk = stable[len(prev_emitted):]
                            self._seq_prev_text[sid] = stable
                            pending.loop.call_soon_threadsafe(
                                pending.token_queue.put_nowait, new_chunk)
                        # Update token count even if no text was emitted.
                        self._seq_prev_tokens[sid] = cur_count

            # Process completed sequences
            for seq_id, token_ids in outputs:
                with self._lock:
                    pending = self._pending.pop(seq_id, None)
                    self._seq_prev_tokens.pop(seq_id, None)
                if pending is None:
                    continue

                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                result = {
                    "text": text,
                    "token_ids": token_ids,
                    "prompt_tokens": pending.prompt_tokens,
                    "completion_tokens": len(token_ids),
                }

                if pending.stream:
                    # Push final text remainder (anything between the last
                    # streamed `stable` text and the full decoded text). At
                    # finish time there are no more tokens to come, so we MUST
                    # emit any deferred trailing bytes — including any final
                    # U+FFFD that would otherwise be silently dropped.
                    prev_emitted = self._seq_prev_text.get(seq_id, "")
                    if len(text) > len(prev_emitted):
                        tail = text[len(prev_emitted):]
                        pending.loop.call_soon_threadsafe(
                            pending.token_queue.put_nowait, tail)
                    self._seq_prev_text.pop(seq_id, None)
                    # Push remaining tokens + sentinel; record final usage
                    pending.completion_tokens_final = len(token_ids)
                    pending.loop.call_soon_threadsafe(
                        pending.token_queue.put_nowait, None  # sentinel
                    )

                # Resolve the future
                if not pending.future.done():
                    pending.loop.call_soon_threadsafe(
                        pending.future.set_result, result
                    )

            # Check if there's still work
            if self.engine.is_finished():
                self._has_work.clear()


# ============================================================
# FastAPI application
# ============================================================

def create_app(engine: AsyncEngineWrapper, shutdown_event: asyncio.Event | None = None) -> FastAPI:
    app = FastAPI(title="nanovllm API Server", version="0.1.0")

    # ----------------------------------------------------------
    # Health check
    # ----------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ----------------------------------------------------------
    # Shutdown endpoint — gracefully stop the server
    # ----------------------------------------------------------
    @app.post("/shutdown")
    async def shutdown():
        """Gracefully shutdown the server, clean up all processes and GPU resources."""
        if shutdown_event is not None:
            shutdown_event.set()
        return {"status": "shutting_down", "message": "Server is shutting down..."}

    # ----------------------------------------------------------
    # List models
    # ----------------------------------------------------------
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nanovllm",
                }
            ],
        }

    # ----------------------------------------------------------
    # Chat completions
    # ----------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        # Apply chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        tpl_kwargs = request.chat_template_kwargs or {}
        try:
            prompt = engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **tpl_kwargs,
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = "".join(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in messages
            ) + "<|im_start|>assistant\n"

        sampling_params = SamplingParams(
            temperature=max(request.temperature, 0.01),
            max_tokens=request.max_tokens,
            ignore_eos=_resolve_ignore_eos(request),
        )

        try:
            pending = engine.add_request(prompt, sampling_params, stream=request.stream)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if request.stream:
            return StreamingResponse(
                _stream_chat_response(pending, engine.model_name),
                media_type="text/event-stream",
            )
        else:
            result = await pending.future
            response = ChatCompletionResponse(
                id=pending.request_id,
                created=int(time.time()),
                model=engine.model_name,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=result["text"]),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            )
            return response

    async def _stream_chat_response(
        pending: PendingRequest, model_name: str
    ) -> AsyncGenerator[str, None]:
        request_id = pending.request_id
        created = int(time.time())

        # First chunk: role
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(role="assistant", content=""),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        # Content chunks
        while True:
            token_text = await pending.token_queue.get()
            if token_text is None:  # sentinel: generation done
                break
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=ChatCompletionStreamDelta(content=token_text),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk: finish_reason
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=ChatCompletionStreamDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        # Usage chunk (always emit; see completion path).
        usage_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": pending.prompt_tokens,
                "completion_tokens": pending.completion_tokens_final,
                "total_tokens": pending.prompt_tokens + pending.completion_tokens_final,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ----------------------------------------------------------
    # Text completions
    # ----------------------------------------------------------
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        prompt = request.prompt
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        sampling_params = SamplingParams(
            temperature=max(request.temperature, 0.01),
            max_tokens=request.max_tokens,
            ignore_eos=_resolve_ignore_eos(request),
        )

        try:
            pending = engine.add_request(prompt, sampling_params, stream=request.stream)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if request.stream:
            return StreamingResponse(
                _stream_completion_response(pending, engine.model_name),
                media_type="text/event-stream",
            )
        else:
            result = await pending.future
            response = CompletionResponse(
                id=pending.request_id,
                created=int(time.time()),
                model=engine.model_name,
                choices=[
                    CompletionChoice(
                        index=0,
                        text=result["text"],
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                ),
            )
            return response

    async def _stream_completion_response(
        pending: PendingRequest, model_name: str
    ) -> AsyncGenerator[str, None]:
        request_id = pending.request_id
        created = int(time.time())

        while True:
            token_text = await pending.token_queue.get()
            if token_text is None:
                break
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created,
                model=model_name,
                choices=[
                    CompletionStreamChoice(index=0, text=token_text)
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk with finish_reason
        chunk = CompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                CompletionStreamChoice(index=0, text="", finish_reason="stop")
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
        # Usage chunk (OpenAI stream_options.include_usage convention).
        # Always emit so clients counting via usage get accurate counts even
        # when the client did not opt in to stream_options — bench_water needs this.
        usage_chunk = {
            "id": request_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": pending.prompt_tokens,
                "completion_tokens": pending.completion_tokens_final,
                "total_tokens": pending.prompt_tokens + pending.completion_tokens_final,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return app


# ============================================================
# Graceful cleanup utilities
# ============================================================

def _kill_child_processes():
    """Kill all child processes spawned by this server (TP workers etc.)."""
    import multiprocessing
    current = os.getpid()
    try:
        import psutil
        parent = psutil.Process(current)
        children = parent.children(recursive=True)
        for child in children:
            logger.info("Terminating child process PID=%d", child.pid)
            child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=5)
        for p in alive:
            logger.warning("Force killing child process PID=%d", p.pid)
            p.kill()
    except ImportError:
        # psutil not available, use os-level signal
        try:
            # Send SIGTERM to the entire process group
            os.killpg(os.getpgid(current), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass


def _cleanup_shared_memory():
    """Clean up any leaked shared memory segments."""
    try:
        from multiprocessing.shared_memory import SharedMemory
        for name in ["nanovllm"]:
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                logger.info("Cleaned up shared memory: %s", name)
            except FileNotFoundError:
                pass
    except Exception:
        pass


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="nanovllm OpenAI-compatible API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model directory")

    # --- Server options ---
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind")

    # --- Config parameters (matching nanovllm.config.Config dataclass) ---
    parser.add_argument("--max-num-batched-tokens", type=int, default=200000,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="Maximum number of sequences per iteration")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use (0.0 ~ 1.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs, use eager mode only")
    parser.add_argument("--kvcache-block-size", type=int, default=256,
                        help="KV cache block size (must be multiple of 256)")
    parser.add_argument("--enable-prefix-caching", action="store_true", default=True,
                        help="Enable prefix caching (default: enabled)")
    parser.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false",
                        help="Disable prefix caching")
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="Custom model name for API responses. If not set, uses the model directory name")

    # --- Layer-Batch (SM-disjoint dual-stream decode) ---
    parser.add_argument("--enable-layer-batch", action="store_true",
                        help="Enable Layer-Batch parallel decode (Green Context dual-stream)")
    parser.add_argument("--layer-batch-fa-sm", type=int, default=-1,
                        help="SMs in the FA partition. -1=auto (40%% of detected total SM).")
    parser.add_argument("--layer-batch-la-sm", type=int, default=-1,
                        help="Informational only; LA = total_sm - fa_sm. -1=auto.")
    parser.add_argument("--layer-batch-split", type=float, default=0.5,
                        help="nano-batch1 fraction of the batch (0,1)")
    parser.add_argument("--layer-batch-min-bs", type=int, default=2,
                        help="Disable LB below this batch size")
    parser.add_argument("--layer-batch-max-bs", type=int, default=-1,
                        help="Disable LB above this batch size. -1=auto (= max-num-seqs).")
    parser.add_argument("--layer-batch-use-graph", type=int, default=1, choices=[0, 1],
                        help="0 = eager LB; 1 = CUDA-Graph LB (default, verified safe after "
                             "the 0a74ec1 fix).")
    parser.add_argument("--layer-batch-partitions", type=str, default="",
                        help="Dynamic SM allocation buckets, CSV of fa_sm:la_sm:max_ctx triples, "
                             "e.g. '24:54:4096,39:39:32768,56:22:1000000'. The runtime picks the "
                             "FIRST bucket whose max_ctx >= the current batch's max(context_lens). "
                             "Empty (default) = use legacy single-partition mode driven by "
                             "--layer-batch-fa-sm / --layer-batch-la-sm.")
    parser.add_argument("--layer-batch-min-total-tokens", type=int, default=256,
                        help="Disable LB when sum(context_lens)+bs < this. Default 256.")
    parser.add_argument("--layer-batch-max-total-tokens", type=int, default=-1,
                        help="Disable LB when sum(context_lens)+bs > this. -1=auto (= max_model_len * max_num_seqs).")
    parser.add_argument("--layer-batch-no-greenctx", action="store_true",
                        help="POD-Attention experimental: skip Green-Context partitioning and use "
                             "two regular cuda streams that share all SMs. The grid scheduler will "
                             "co-locate Group-A and Group-B kernels on the same SMs, mixing TC- and "
                             "MEM-bound warps to raise SM utilization.")
    parser.add_argument("--layer-batch-simple", action="store_true",
                        help="Streamlined LayerBatch: exactly 2 cuda streams, no Green-Context, no "
                             "multi-partition skeleton. Two nano-batches share all 78 SMs and the "
                             "GPU grid scheduler co-locates their CTAs opportunistically. "
                             "Overrides --layer-batch-no-greenctx and --layer-batch-partitions.")
    parser.add_argument("--pod-attention-decode", action="store_true",
                        help="POD-Attention CTA-level fusion: replace decode flash_attn with a "
                             "Triton paged kernel that can piggy-back a GEMM at zero cost. Best at "
                             "long ctx (>=200K total tokens). Requires LB to dispatch the GEMM.")
    # POD kernel tile-config knobs (-1 sentinel = auto, resolved in ModelRunner)
    parser.add_argument("--pod-num-kv-splits", type=int, default=-1,
                        help="POD: K/V split factor (auto: pow2 scaling with max_model_len).")
    parser.add_argument("--pod-block-n",       type=int, default=-1,
                        help="POD: BLOCK_N attention K/V tile rows (auto: 64).")
    parser.add_argument("--pod-block-h",       type=int, default=-1,
                        help="POD: BLOCK_H Q-heads per CTA (auto: max(16, num_q/num_kv)).")
    parser.add_argument("--pod-num-warps",     type=int, default=4,
                        help="POD kernel num_warps (default 4).")
    parser.add_argument("--pod-num-stages",    type=int, default=3,
                        help="POD kernel num_stages (default 3).")
    parser.add_argument("--pod-gemm-block-m",  type=int, default=16, help="POD GEMM M tile.")
    parser.add_argument("--pod-gemm-block-n",  type=int, default=64, help="POD GEMM N tile.")
    parser.add_argument("--pod-gemm-block-k",  type=int, default=32, help="POD GEMM K tile.")

    # --- Prefill-LayerBatch (cache-hit-aware dual-stream prefill) ---
    parser.add_argument("--enable-prefill-layer-batch", action="store_true",
                        help="Enable Prefill LayerBatch: classify prefill seqs by prefix-cache hit, "
                             "run high/low-hit groups in parallel on two streams under Green Context "
                             "with asymmetric SM allocation. Designed for long-ctx + mixed-hit workloads.")
    parser.add_argument("--prefill-lb-hit-threshold", type=float, default=0.05,
                        help="Hit-ratio cut between low-hit and high-hit (default 0.05).")
    parser.add_argument("--prefill-lb-min-len", type=int, default=100_000,
                        help="Minimum prompt length (tokens) of any seq required to fire (default 100000).")
    parser.add_argument("--prefill-lb-low-hit-sm",  type=int, default=-1,
                        help="SMs for the LOW-hit (compute-bound) stream. -1=auto (70%% of total).")
    parser.add_argument("--prefill-lb-high-hit-sm", type=int, default=-1,
                        help="SMs for the HIGH-hit (bandwidth-bound) stream. -1=auto (30%% of total).")

    parser.add_argument("--use-flashinfer-prefill", action="store_true",
                        help="Use FlashInfer paged-prefill wrappers instead of flash_attn_varlen_func. "
                             "AUTO-DISABLED on Ampere (A100, sm_80/86) to avoid Hopper-kernel JIT "
                             "compile hazards; on Ampere the engine falls back to flash-attn 2 (if "
                             "installed) or the pure-torch SDPA fallback.  Set env var "
                             "NANOVLLM_FORCE_FLASHINFER_ON_AMPERE=1 to override.")

    # --- Chunked prefill ---
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="Split prefill into fixed-size chunks to cap peak activation "
                             "memory at O(chunk_size) instead of O(max_seq_len).  Only chunks "
                             "seqs whose new-token count exceeds --prefill-chunk-size.")
    parser.add_argument("--prefill-chunk-size", type=int, default=2048,
                        help="Chunk size in tokens (rounded up to a multiple of "
                             "--kvcache-block-size).  Default 2048 fits well within an A100 "
                             "activation budget.")

    args = parser.parse_args()

    logger.info("Starting nanovllm API server...")
    logger.info("  Model:              %s", args.model)
    logger.info("  Host:               %s:%d", args.host, args.port)
    logger.info("  TP size:            %d", args.tensor_parallel_size)
    logger.info("  Max model len:      %d", args.max_model_len)
    logger.info("  Max num seqs:       %d", args.max_num_seqs)
    logger.info("  Max batched tokens: %d", args.max_num_batched_tokens)
    logger.info("  GPU mem util:       %.2f", args.gpu_memory_utilization)
    logger.info("  KV block size:      %d", args.kvcache_block_size)
    logger.info("  Enforce eager:      %s", args.enforce_eager)
    logger.info("  Prefix caching:     %s", args.enable_prefix_caching)
    logger.info("  Served model name:  %s", args.served_model_name or "(auto)")

    # Build engine kwargs from all Config-compatible parameters
    engine_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
        "kvcache_block_size": args.kvcache_block_size,
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.enable_layer_batch:
        # Parse --layer-batch-partitions CSV "fa:la:thr,fa:la:thr,..."
        partitions = []
        if args.layer_batch_partitions.strip():
            for tri in args.layer_batch_partitions.split(","):
                parts = tri.strip().split(":")
                if len(parts) != 3:
                    raise SystemExit(f"--layer-batch-partitions: bad triple {tri!r}, "
                                      f"expected fa_sm:la_sm:max_ctx_threshold")
                fa, la, thr = int(parts[0]), int(parts[1]), int(parts[2])
                partitions.append((fa, la, thr))
            partitions.sort(key=lambda t: t[2])  # ascending by max_ctx_thr

        engine_kwargs.update({
            "enable_layer_batch":     True,
            "layer_batch_fa_sm":      args.layer_batch_fa_sm,
            "layer_batch_la_sm":      args.layer_batch_la_sm,
            "layer_batch_split":      args.layer_batch_split,
            "layer_batch_min_bs":     args.layer_batch_min_bs,
            "layer_batch_max_bs":     args.layer_batch_max_bs,
            "layer_batch_use_graph":  bool(args.layer_batch_use_graph),
            "layer_batch_min_total_tokens": args.layer_batch_min_total_tokens,
            "layer_batch_max_total_tokens": args.layer_batch_max_total_tokens,
            "layer_batch_no_greenctx":      bool(args.layer_batch_no_greenctx),
            "layer_batch_simple":           bool(args.layer_batch_simple),
        })
        if partitions:
            engine_kwargs["layer_batch_partitions"] = partitions
        if args.layer_batch_simple:
            logger.info("  Layer-Batch:        ENABLED SIMPLE (2 streams, no GC, no partition)")
        elif args.layer_batch_no_greenctx:
            logger.info("  Layer-Batch:        ENABLED no-greenctx (2 streams sharing all SMs)")
        elif partitions:
            logger.info("  Layer-Batch:        ENABLED dynamic, %d partitions: %s",
                        len(partitions), partitions)
        else:
            logger.info("  Layer-Batch:        ENABLED static (FA SM=%d, LA SM=%d)",
                        args.layer_batch_fa_sm, args.layer_batch_la_sm)
        logger.info("                      bs in [%d..%d], total_tokens in [%d..%d], graph=%d",
                    args.layer_batch_min_bs, args.layer_batch_max_bs,
                    args.layer_batch_min_total_tokens, args.layer_batch_max_total_tokens,
                    args.layer_batch_use_graph)

    if args.pod_attention_decode:
        engine_kwargs["pod_attention_decode"] = True
        engine_kwargs["pod_num_kv_splits"]   = int(args.pod_num_kv_splits)
        engine_kwargs["pod_block_n"]         = int(args.pod_block_n)
        engine_kwargs["pod_block_h"]         = int(args.pod_block_h)
        engine_kwargs["pod_num_warps"]       = int(args.pod_num_warps)
        engine_kwargs["pod_num_stages"]      = int(args.pod_num_stages)
        engine_kwargs["pod_gemm_block_m"]    = int(args.pod_gemm_block_m)
        engine_kwargs["pod_gemm_block_n"]    = int(args.pod_gemm_block_n)
        engine_kwargs["pod_gemm_block_k"]    = int(args.pod_gemm_block_k)
        logger.info("  POD-decode:         ENABLED (Triton paged flash_attn + GEMM piggy-back) "
                    "kv_splits=%s block_n=%s block_h=%s warps=%s stages=%s",
                    args.pod_num_kv_splits, args.pod_block_n, args.pod_block_h,
                    args.pod_num_warps, args.pod_num_stages)

    if args.enable_prefill_layer_batch:
        engine_kwargs["enable_prefill_layer_batch"] = True
        engine_kwargs["prefill_lb_hit_threshold"]  = float(args.prefill_lb_hit_threshold)
        engine_kwargs["prefill_lb_min_len"]        = int(args.prefill_lb_min_len)
        engine_kwargs["prefill_lb_low_hit_sm"]     = int(args.prefill_lb_low_hit_sm)
        engine_kwargs["prefill_lb_high_hit_sm"]    = int(args.prefill_lb_high_hit_sm)
    if args.use_flashinfer_prefill:
        engine_kwargs["use_flashinfer_prefill"] = True
        logger.info("  FlashInfer paged-prefill: requested (Ampere hosts auto-disable this)")
    if args.enable_chunked_prefill:
        engine_kwargs["enable_chunked_prefill"] = True
        engine_kwargs["prefill_chunk_size"] = int(args.prefill_chunk_size)
        logger.info("  Chunked prefill: ENABLED (chunk_size=%d)", args.prefill_chunk_size)
        logger.info("  Prefill-LB:         ENABLED (hit_thr=%.2f min_len=%d low_sm=%s high_sm=%s)",
                    args.prefill_lb_hit_threshold, args.prefill_lb_min_len,
                    args.prefill_lb_low_hit_sm, args.prefill_lb_high_hit_sm)

    engine = AsyncEngineWrapper(args.model, served_model_name=args.served_model_name, **engine_kwargs)

    # Shutdown event for coordinated cleanup
    shutdown_event = asyncio.Event()
    app = create_app(engine, shutdown_event)

    logger.info("Server ready at http://%s:%d", args.host, args.port)
    logger.info("  POST /v1/chat/completions")
    logger.info("  POST /v1/completions")
    logger.info("  GET  /v1/models")
    logger.info("  GET  /health")
    logger.info("  POST /shutdown        (graceful shutdown)")

    # --- Custom uvicorn server with graceful shutdown ---
    server_config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info",
    )
    server = uvicorn.Server(server_config)

    # Set up signal handlers for thorough cleanup
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        shutdown_event.set()
        server.should_exit = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Run server in a thread so we can monitor the shutdown event
    server_thread = threading.Thread(target=server.run, daemon=False)
    server_thread.start()

    try:
        # Wait for shutdown signal (from signal handler or /shutdown endpoint)
        while not shutdown_event.is_set() and server_thread.is_alive():
            server_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, shutting down...")
        shutdown_event.set()
        server.should_exit = True

    # --- Thorough cleanup ---
    logger.info("Cleaning up resources...")

    # 1. Stop accepting new requests and shut down uvicorn
    server.should_exit = True
    server_thread.join(timeout=10)

    # 2. Shut down the engine (model_runner, child processes, dist, shm)
    engine.shutdown()

    # 3. Clean up any leaked shared memory
    _cleanup_shared_memory()

    # 4. Kill any remaining child processes
    _kill_child_processes()

    # 5. Final GPU cleanup
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    except Exception:
        pass

    # 6. Force garbage collection
    gc.collect()

    logger.info("All resources cleaned up. Exiting.")
    sys.exit(0)


if __name__ == "__main__":
    main()
