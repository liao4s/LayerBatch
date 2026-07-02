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

class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, List[str]] = ""
    temperature: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    top_p: float = 1.0
    n: int = 1
    stop: Optional[Union[List[str], str]] = None

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

        # Track token counts for streaming incremental decode
        self._seq_prev_tokens: dict[int, int] = {}

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
                        new_ids = seq.completion_token_ids[prev_count:cur_count]
                        self._seq_prev_tokens[sid] = cur_count
                        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                        if text:
                            pending.loop.call_soon_threadsafe(
                                pending.token_queue.put_nowait, text
                            )

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
                    # Push remaining tokens + sentinel
                    prev_count = 0  # already pushed incrementally
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

        # Final chunk
        chunk = CompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                CompletionStreamChoice(index=0, text="", finish_reason="stop")
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"
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
