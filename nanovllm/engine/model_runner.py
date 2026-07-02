import os
import pickle
import torch
import torch.distributed as dist
from collections import deque
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.logger import init_logger
from nanovllm.utils.green_ctx import init_green_contexts, init_green_contexts_multi, get_streams, get_sm_counts, is_supported as _gctx_supported
from nanovllm.engine.layer_batch import run_layer_batch_decode

logger = init_logger(__name__)


def get_model_class(hf_config):
    """Select model class based on HuggingFace config."""
    model_type = getattr(hf_config, 'model_type', '')
    if model_type == 'qwen3_5_moe':
        from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM
    elif model_type == 'qwen3_5':
        from nanovllm.models.qwen3_5_dense import Qwen3_5DenseForCausalLM
        return Qwen3_5DenseForCausalLM
    else:
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self._has_linear_attn = getattr(hf_config, 'model_type', '') in ('qwen3_5_moe', 'qwen3_5')
        self._fa_stream = None
        self._la_stream = None
        self._layer_types: list[str] = []
        # Pre-init prefill-LB partition handle (real setup happens after warmup
        # since green-ctx creation needs CUDA + model loaded). Setting it to
        # None up-front so warmup_model -> run() can safely check the attribute.
        self._lb_prefill_partition: dict | None = None

        import datetime
        _port = int(os.environ.get("NANOVLLM_DIST_PORT", "2333")); dist.init_process_group("nccl", f"tcp://localhost:{_port}", world_size=self.world_size, rank=rank,
                                timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(rank)
        # Resolve config sentinels (-1) into concrete values now that the CUDA device is set.
        self._resolve_dynamic_defaults()
        # Set up Triton allocator (required for kernels needing scratch memory in Triton 3.6+)
        try:
            import triton
            from triton.runtime._allocation import Allocator
            class _TorchAllocator(Allocator):
                def __call__(self, size, align, stream):
                    return torch.empty(size, dtype=torch.uint8, device=torch.cuda.current_device()).data_ptr()
            triton.set_allocator(_TorchAllocator())
        except Exception:
            pass
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = get_model_class(hf_config)(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # POD-decode toggle (applies even without LB).  MUST be set BEFORE
        # warmup + graph capture so the pod kernel is captured into the decode
        # CUDA-Graphs (otherwise the captured graphs invoke flash_attn_with_kvcache
        # and the toggle has no runtime effect at decode time).
        if getattr(self.config, "pod_attention_decode", False):
            from nanovllm.layers.attention import set_use_pod_decode, set_pod_runtime_cfg
            set_use_pod_decode(True)
            set_pod_runtime_cfg(getattr(self, "_pod_runtime_cfg", {}) or {})
            logger.info("POD-Attention decode kernel ENABLED (CTA-fused flash_attn + GEMM); cfg=%s",
                        getattr(self, "_pod_runtime_cfg", {}))

        # FlashInfer paged-prefill toggle.  On Ampere this is auto-disabled
        # (see flashinfer_attn.set_use_flashinfer_prefill).  On Hopper it
        # runs a preflight check that fails fast if the install is broken —
        # BEFORE we burn time on warmup.
        if getattr(self.config, "use_flashinfer_prefill", False):
            from nanovllm.layers import flashinfer_attn as _fi
            _fi.set_use_flashinfer_prefill(True)
            if _fi.get_use_flashinfer_prefill():
                logger.info("FlashInfer paged-prefill backend ENABLED.")
            else:
                logger.warning("FlashInfer paged-prefill was requested but "
                               "AUTO-DISABLED (likely Ampere host).  Prefill "
                               "will use flash_attn 2 or torch SDPA fallback.")

        self.warmup_model()
        self._compute_linear_attn_budget()
        self.allocate_kv_cache()
        if self._has_linear_attn:
            self.allocate_linear_attn_states()
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Initialize Green Context partitions for layer-batch parallelism (rank 0 only).
        # `_lb_partitions` is a list of dicts:
        #   [{fa_sm, la_sm, max_ctx_thr, fa_stream, la_stream}, ...]
        # sorted ascending by max_ctx_thr. The runtime picks the first one whose
        # max_ctx_thr >= max(context_lens). Empty list = LB disabled.
        # `_fa_stream` / `_la_stream` are kept as aliases to the FIRST partition's
        # streams for backwards compatibility (some old paths reference them).
        self._lb_partitions: list = []
        self._fa_stream = None
        self._la_stream = None
        self._lb_graphs: dict = {}   # key (bs, B1, partition_idx) -> CUDAGraph
        # ---- Streamlined Layer-Batch (recommended path) ----
        # Bypasses init_green_contexts entirely; creates exactly 2 streams.
        # Set self._lb_simple=True so the run/replay path uses _run_layer_batch_simple.
        self._lb_simple = False
        self._lb_simple_fa_stream = None
        self._lb_simple_la_stream = None
        self._lb_simple_graphs: dict = {}   # {bucket_bs -> (B1, graph)}

        if self.config.enable_layer_batch and self.config.layer_batch_simple and self.rank == 0:
            try:
                fa_s = torch.cuda.Stream(device=torch.cuda.current_device())
                la_s = torch.cuda.Stream(device=torch.cuda.current_device())
                self._lb_simple_fa_stream = fa_s
                self._lb_simple_la_stream = la_s
                self._lb_simple = True
                # Cache layer types for the LB pipeline.
                hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
                self._layer_types = list(hf_text.layer_types)
                logger.info("Layer-Batch SIMPLE mode: created exactly 2 streams "
                            "(fa=stream@%#x, la=stream@%#x; no Green-Context, no SM partition).",
                            int(fa_s.cuda_stream), int(la_s.cuda_stream))
                # The two streams will be used through Phase B as plain torch streams;
                # the run_layer_batch_decode pipeline (lag-by-1 nano-batch schedule)
                # is reused unchanged.
                if not self.enforce_eager and self.config.layer_batch_use_graph:
                    self.capture_layer_batch_simple_graphs()
            except Exception as e:
                logger.warning("Layer-Batch SIMPLE init failed (%s) - falling back.", e)
                import traceback; traceback.print_exc()
                self._lb_simple = False
                self._lb_simple_fa_stream = None
                self._lb_simple_la_stream = None

        if self.config.enable_layer_batch and not self._lb_simple and self.rank == 0:
            try:
                no_gc = bool(getattr(self.config, "layer_batch_no_greenctx", False))
                if no_gc:
                    # POD-Attention shared-SM mode: skip Green-Context
                    # partitioning, use two regular `torch.cuda.Stream`s that
                    # both share all 78 SMs.  The grid scheduler can co-locate
                    # Group-A and Group-B kernels on the same SMs, mixing
                    # TC-bound and MEM-bound warps for higher SM utilization.
                    fa_s = torch.cuda.Stream(device=torch.cuda.current_device())
                    la_s = torch.cuda.Stream(device=torch.cuda.current_device())
                    self._lb_partitions = [{
                        "fa_sm": 0, "la_sm": 0, "max_ctx_thr": 10**9,
                        "fa_stream": fa_s, "la_stream": la_s,
                    }]
                    self._fa_stream = fa_s
                    self._la_stream = la_s
                    logger.info("Layer-Batch: NO-GREENCTX mode — 2 regular streams sharing all SMs")
                else:
                    if not _gctx_supported():
                        raise RuntimeError("cuda-python green-ctx APIs unavailable")
                    # Build the partition list from config: prefer
                    # `layer_batch_partitions`; fall back to legacy single (fa, la).
                    triples = list(getattr(self.config, "layer_batch_partitions", []) or [])
                    if not triples:
                        triples = [(self.config.layer_batch_fa_sm,
                                    self.config.layer_batch_la_sm,
                                    10**9)]
                    triples.sort(key=lambda t: t[2])  # ascending by max_ctx_thr

                    bundles = init_green_contexts_multi(
                        [(fa, la) for fa, la, _ in triples],
                        device_index=torch.cuda.current_device(),
                        non_blocking=True,
                    )
                    self._lb_partitions = [
                        {"fa_sm":   b["fa_sm"],
                         "la_sm":   b["la_sm"],
                         "max_ctx_thr": thr,
                         "fa_stream": b["fa_stream"],
                         "la_stream": b["la_stream"]}
                        for (fa, la, thr), b in zip(triples, bundles)
                    ]
                    self._fa_stream = self._lb_partitions[0]["fa_stream"]
                    self._la_stream = self._lb_partitions[0]["la_stream"]

                    logger.info("Layer-Batch: %d partition(s) ready", len(self._lb_partitions))
                    for i, p in enumerate(self._lb_partitions):
                        logger.info("  partition %d: FA=%d SMs, LA=%d SMs, max_ctx_thr=%d",
                                    i, p["fa_sm"], p["la_sm"], p["max_ctx_thr"])

                # Cache the ordered list of layer types for run-time scheduling.
                hf_text = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
                self._layer_types = list(hf_text.layer_types)

                if not self.enforce_eager and self.config.layer_batch_use_graph:
                    self.capture_layer_batch_graphs()
            except Exception as e:
                logger.warning("Layer-Batch init failed (%s) — falling back to standard decode.", e)
                import traceback; traceback.print_exc()
                self._lb_partitions = []
                self._fa_stream = None
                self._la_stream = None
                self.config.enable_layer_batch = False

        # ---- Prefill-LayerBatch (cache-hit-aware dual-stream prefill) ----
        # Independent Green Context partition with asymmetric SM split:
        #   low-hit-stream gets ~70% SMs (compute-bound — QKV+MLP on many new tokens)
        #   high-hit-stream gets ~30% SMs (HBM-bound — paged KV cache reads dominate)
        # The partition is independent from decode-LB partitions so they can coexist.
        # (`self._lb_prefill_partition` is pre-initialized to None at the top of
        # __init__ so warmup_model() can safely check it before this block runs.)
        if self.config.enable_prefill_layer_batch and self.rank == 0:
            try:
                if not _gctx_supported():
                    raise RuntimeError("cuda-python green-ctx APIs unavailable")
                bundles = init_green_contexts_multi(
                    [(int(self.config.prefill_lb_low_hit_sm), int(self.config.prefill_lb_high_hit_sm))],
                    device_index=torch.cuda.current_device(),
                    non_blocking=True,
                )
                b = bundles[0]
                # Naming: "fa_stream" in the bundle is the FIRST partition (the bigger one);
                # we map it to low_hit (compute-bound). The "la_stream" maps to high_hit.
                self._lb_prefill_partition = {
                    "low_hit_sm":   b["fa_sm"],
                    "high_hit_sm":  b["la_sm"],
                    "low_stream":   b["fa_stream"],
                    "high_stream":  b["la_stream"],
                }
                logger.info("Prefill-LayerBatch: partition ready (low-hit=%d SMs, high-hit=%d SMs)",
                            b["fa_sm"], b["la_sm"])
            except Exception as e:
                logger.warning("Prefill-LayerBatch init failed (%s) — feature disabled.", e)
                import traceback; traceback.print_exc()
                self._lb_prefill_partition = None
                self.config.enable_prefill_layer_batch = False

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    _stale = SharedMemory(name="nanovllm", create=False)
                    _stale.close()
                    _stale.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # Cap warmup length for linear attention models to avoid OOM in Triton autotuner.
        # The chunk kernel allocates O(T * H * V * K) intermediate state which is too large for T=200K.
        if self._has_linear_attn:
            max_model_len = min(max_model_len, 8192)
            max_num_batched_tokens = min(max_num_batched_tokens, max_model_len)
        # When chunked prefill is enabled, warmup measures ONE chunk's peak
        # memory — that matches the steady-state runtime workload (each
        # `_run_chunked_prefill` step processes one chunk_size worth of
        # tokens per seq).  Warmup with the full max_model_len would either
        # OOM or would need to walk through chunked prefill on empty
        # block_tables, which the paged KV cache path can't do (cache not
        # yet allocated).
        if getattr(self.config, "enable_chunked_prefill", False):
            eff_chunk = self._effective_chunk_size()
            max_model_len = min(max_model_len, eff_chunk)
            max_num_batched_tokens = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        num_seqs = max(num_seqs, 1)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def _compute_linear_attn_budget(self):
        """Pre-compute how much memory the linear attention state buffers will need.
        Called BEFORE allocate_kv_cache so the KV cache budget can be reduced accordingly.
        Also reserves memory for CUDA Graph capture overhead.
        """
        self._linear_attn_budget_bytes = 0
        self._cuda_graph_reserve_bytes = 0
        if not self._has_linear_attn:
            return
        from nanovllm.models.qwen3_5 import Qwen3_5GatedDeltaNet
        hf_config = self.config.hf_config
        linear_layers = [m for m in self.model.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
        if not linear_layers:
            return
        num_layers = len(linear_layers)
        layer0 = linear_layers[0]
        dtype = hf_config.torch_dtype
        elem_size = dtype.itemsize
        # Per-slot bytes across all layers
        recurrent_per_slot = num_layers * layer0.num_v_heads * layer0.head_v_dim * layer0.head_k_dim * 4  # float32
        conv_per_slot = num_layers * layer0.conv_dim * (layer0.conv_kernel_size - 1) * elem_size
        bytes_per_slot = recurrent_per_slot + conv_per_slot
        # Use a conservative default: 32 slots ≈ 1GB for Qwen3.5-35B-A3B
        # This supports 32 concurrent sequences, sufficient for most serving scenarios
        max_slots = min(32, self.config.max_num_seqs)
        self._linear_attn_budget_bytes = bytes_per_slot * max_slots
        self._linear_attn_max_slots = max_slots
        self._linear_attn_bytes_per_slot = bytes_per_slot
        # Reserve memory for CUDA Graph capture (activations for decode with max_bs tokens)
        # MoE models need extra memory for the dense expert dispatch (gather-based)
        if not self.enforce_eager:
            num_total_layers = hf_config.num_hidden_layers
            text_config = hf_config.text_config if hasattr(hf_config, 'text_config') else hf_config
            moe_inter = getattr(text_config, 'moe_intermediate_size', 0)
            num_experts_per_tok = getattr(text_config, 'num_experts_per_tok', 0)
            max_decode_bs = min(32, self.config.max_num_seqs)  # typical max decode batch
            if moe_inter > 0:
                # Dense MoE dispatch: per top_k slot, gather gate_up [N,2*inter,hidden] + down [N,hidden,inter]
                # Peak ≈ one iteration's temporaries: gate_up_w + down_w + intermediates
                elem_size = hf_config.torch_dtype.itemsize
                hidden = text_config.hidden_size
                per_token_peak = (2 * moe_inter * hidden + hidden * moe_inter) * elem_size  # gate_up + down
                self._cuda_graph_reserve_bytes = max_decode_bs * per_token_peak + num_total_layers * 2 * 1024 * 1024
            else:
                self._cuda_graph_reserve_bytes = num_total_layers * 2 * 1024 * 1024  # ~2MB per layer
        budget_mb = self._linear_attn_budget_bytes / 1024 / 1024
        graph_mb = self._cuda_graph_reserve_bytes / 1024 / 1024
        logger.info("Reserved memory: linear_attn=%.0fMB (%d slots), "
                    "cuda_graph=%.0fMB", budget_mb, max_slots, graph_mb)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # When num_kv_heads < tp_size, KV heads are replicated (not sharded)
        if hf_config.num_key_value_heads >= self.world_size:
            num_kv_heads = hf_config.num_key_value_heads // self.world_size
        else:
            num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # Count the number of layers that actually need KV cache (have Attention modules)
        num_attn_layers = sum(1 for module in self.model.modules() if hasattr(module, "k_cache") and hasattr(module, "v_cache"))
        if num_attn_layers == 0:
            num_attn_layers = hf_config.num_hidden_layers  # fallback
        block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # Subtract reserved memory for linear attn buffers and CUDA Graph before computing KV cache blocks
        reserved = getattr(self, '_linear_attn_budget_bytes', 0) + getattr(self, '_cuda_graph_reserve_bytes', 0)
        kv_budget = int(total * config.gpu_memory_utilization - used - peak + current) - reserved
        config.num_kvcache_blocks = kv_budget // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_attn_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_linear_attn_states(self):
        """Pre-allocate recurrent/conv state buffers for GatedDeltaNet layers (CUDA Graph safe).

        Budget was pre-computed by _compute_linear_attn_budget() and deducted from KV cache.
        """
        from nanovllm.models.qwen3_5 import Qwen3_5GatedDeltaNet
        hf_config = self.config.hf_config

        linear_layers = [m for m in self.model.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
        if not linear_layers:
            return

        num_layers = len(linear_layers)
        layer0 = linear_layers[0]
        num_v_heads = layer0.num_v_heads
        head_k_dim = layer0.head_k_dim
        head_v_dim = layer0.head_v_dim
        conv_dim = layer0.conv_dim
        conv_kernel_size = layer0.conv_kernel_size
        dtype = hf_config.torch_dtype
        elem_size = dtype.itemsize

        max_slots = getattr(self, '_linear_attn_max_slots', min(128, self.config.max_num_seqs))

        # Use float32 for recurrent state buffer to prevent numerical degradation
        # during decode (model config specifies mamba_ssm_dtype=float32)
        self.linear_attn_recurrent_buf = torch.zeros(
            num_layers, max_slots, num_v_heads, head_v_dim, head_k_dim,
            dtype=torch.float32, device="cuda",
        )
        self.linear_attn_conv_buf = torch.zeros(
            num_layers, max_slots, conv_dim, conv_kernel_size - 1,
            dtype=dtype, device="cuda",
        )

        for i, module in enumerate(linear_layers):
            module.recurrent_state_buf = self.linear_attn_recurrent_buf[i]
            module.conv_state_buf = self.linear_attn_conv_buf[i]

        self._linear_attn_slot_map: dict[int, int] = {}
        self._linear_attn_free_slots: deque[int] = deque(range(max_slots))

        recurrent_mb = self.linear_attn_recurrent_buf.numel() * elem_size / 1024 / 1024
        conv_mb = self.linear_attn_conv_buf.numel() * elem_size / 1024 / 1024
        logger.info("Allocated linear attention state buffers: "
                    "recurrent=%.1fMB, conv=%.1fMB "
                    "(%d layers x %d slots, dtype=%s)",
                    recurrent_mb, conv_mb, num_layers, max_slots, dtype)

    def allocate_linear_attn_slot(self, seq_id: int) -> int:
        """Allocate a buffer slot for a new sequence. Returns slot index."""
        if not self._has_linear_attn:
            return -1
        slot_idx = self._linear_attn_free_slots.popleft()
        self._linear_attn_slot_map[seq_id] = slot_idx
        # Zero out the slot for the new sequence
        self.linear_attn_recurrent_buf[:, slot_idx].zero_()
        self.linear_attn_conv_buf[:, slot_idx].zero_()
        return slot_idx

    def free_linear_attn_slot(self, seq_id: int):
        """Free a buffer slot when sequence finishes."""
        if not self._has_linear_attn:
            return
        slot_idx = self._linear_attn_slot_map.pop(seq_id, None)
        if slot_idx is not None:
            self._linear_attn_free_slots.append(slot_idx)

    def _get_linear_attn_slot_indices(self, seqs: list[Sequence]) -> torch.Tensor | None:
        """Get slot indices tensor for a batch of sequences."""
        if not self._has_linear_attn or not hasattr(self, '_linear_attn_slot_map'):
            return None
        indices = [self._linear_attn_slot_map.get(seq.seq_id, 0) for seq in seqs]
        return torch.tensor(indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def _build_prefill_inputs(self, seqs: list[Sequence]):
        """Build prefill inputs + ctx kwargs for a (sub)set of sequences.

        FlashInfer metadata is only emitted when the flag is on AND
        flashinfer is actually usable (i.e. `_fi.get_use_flashinfer_prefill()`
        returns True — this is False on Ampere due to the auto-disable).
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        seqlens_k_list: list[int] = []
        block_lists: list[list[int]] = []
        for seq in seqs:
            seqlen = len(seq)
            # Slice by the seq's *current* num_tokens (which chunked prefill
            # temporarily overrides to the chunk boundary).  Using seq[a:] would
            # walk to len(token_ids) and blow past the chunk.
            input_ids.extend(seq.token_ids[seq.num_cached_tokens:seqlen])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            seqlens_k_list.append(seqlen_k)
            block_lists.append(list(seq.block_table))
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        # Read the effective FlashInfer flag: the config value AND whether
        # flashinfer is actually usable on this host (auto-off on Ampere).
        from nanovllm.layers import flashinfer_attn as _fi
        use_fi = _fi.get_use_flashinfer_prefill()
        all_have_blocks = all(seq.block_table for seq in seqs)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache hit detected
            block_tables = self.prepare_block_tables(seqs)
        elif use_fi and all_have_blocks:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q_cpu = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
        cu_seqlens_k_cpu = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")
        cu_seqlens_q = cu_seqlens_q_cpu.pin_memory().cuda(non_blocking=True)
        cu_seqlens_k = cu_seqlens_k_cpu.pin_memory().cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        linear_attn_slots = self._get_linear_attn_slot_indices(seqs)
        ctx_kwargs = dict(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping, context_lens=None, block_tables=block_tables,
            seq_ids=[seq.seq_id for seq in seqs],
            linear_attn_slot_indices=linear_attn_slots,
        )
        if use_fi and all_have_blocks:
            ctx_kwargs["_fi_qo_indptr_cpu"] = cu_seqlens_q_cpu
            ctx_kwargs["_fi_seqlens_k"] = seqlens_k_list
            ctx_kwargs["_fi_block_lists"] = block_lists
        return input_ids, positions, ctx_kwargs

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, ctx_kwargs = self._build_prefill_inputs(seqs)
        fi_qo = ctx_kwargs.pop("_fi_qo_indptr_cpu", None)
        fi_seqlens = ctx_kwargs.pop("_fi_seqlens_k", None)
        fi_block_lists = ctx_kwargs.pop("_fi_block_lists", None)
        ctx_kwargs["fi_slot"] = "default"
        set_context(**ctx_kwargs)
        from nanovllm.layers import flashinfer_attn as _fi
        _fi.reset_planned_flag()
        _fi.reset_ragged_planned()
        # Only plan flashinfer when it is actually enabled — on Ampere this
        # is False by design and no plan/JIT is triggered.
        if _fi.get_use_flashinfer_prefill():
            if fi_qo is not None and fi_seqlens is not None and fi_block_lists is not None:
                self._plan_flashinfer("default", fi_qo, fi_seqlens, fi_block_lists)
            else:
                qo_indptr = [0]
                for seq in seqs:
                    qo_indptr.append(qo_indptr[-1] + len(seq) - seq.num_cached_tokens)
                qo_indptr_cpu = torch.tensor(qo_indptr, dtype=torch.int32, device="cpu")
                self._plan_flashinfer_ragged(qo_indptr_cpu)
        return input_ids, positions

    def _flashinfer_dims(self):
        hf = self.config.hf_config
        if hf.num_key_value_heads >= self.world_size:
            num_kv_heads = hf.num_key_value_heads // self.world_size
        else:
            num_kv_heads = hf.num_key_value_heads
        num_qo_heads = hf.num_attention_heads // self.world_size
        head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        return num_qo_heads, num_kv_heads, head_dim, hf.torch_dtype

    def _plan_flashinfer(self, slot, qo_indptr_cpu, seqlens_k, block_lists):
        from nanovllm.layers import flashinfer_attn as _fi
        num_qo_heads, num_kv_heads, head_dim, kv_dtype = self._flashinfer_dims()
        scale = head_dim ** -0.5
        meta = _fi.build_paged_metadata(seqlens_k, block_lists,
                                        page_size=self.block_size,
                                        device=torch.device("cuda"))
        _fi.plan_prefill(qo_indptr_cpu=qo_indptr_cpu, metadata=meta,
                         num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                         head_dim=head_dim, page_size=self.block_size,
                         sm_scale=scale, q_dtype=kv_dtype, kv_dtype=kv_dtype,
                         causal=True, slot=slot)

    def _plan_flashinfer_ragged(self, qo_indptr_cpu: torch.Tensor):
        from nanovllm.layers import flashinfer_attn as _fi
        num_qo_heads, num_kv_heads, head_dim, kv_dtype = self._flashinfer_dims()
        scale = head_dim ** -0.5
        _fi.plan_ragged(qo_indptr_cpu=qo_indptr_cpu, kv_indptr_cpu=qo_indptr_cpu,
                        num_qo_heads=num_qo_heads, num_kv_heads=num_kv_heads,
                        head_dim=head_dim, sm_scale=scale,
                        q_dtype=kv_dtype, kv_dtype=kv_dtype, causal=True)

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        # Python-side aggregates for the LB scheduler — computed BEFORE we move
        # context_lens to GPU so we don't need any cuda → cpu sync to read them.
        max_ctx_len = max(context_lens) if context_lens else 0
        total_tokens = sum(context_lens) + len(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        linear_attn_slots = self._get_linear_attn_slot_indices(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    seq_ids=[seq.seq_id for seq in seqs],
                    linear_attn_slot_indices=linear_attn_slots,
                    max_ctx_len=max_ctx_len, total_tokens=total_tokens)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _resolve_dynamic_defaults(self):
        """Resolve config fields that were left as -1 sentinels.

        Done in ModelRunner __init__ AFTER torch.cuda.set_device, BEFORE warmup /
        graph capture, so every downstream code path sees concrete numbers.
        Centralizes hardware + model-aware defaults so users do not have to set
        H20-specific magic numbers.
        """
        import torch as _t
        cfg = self.config
        hf_text = cfg.hf_config.text_config if hasattr(cfg.hf_config, "text_config") else cfg.hf_config

        try:
            total_sm = int(_t.cuda.get_device_properties(_t.cuda.current_device()).multi_processor_count)
        except Exception:
            total_sm = 78  # H20 fallback
        self._total_sm = total_sm

        # ---- LayerBatch SM defaults (fa/la) ----
        if int(getattr(cfg, "layer_batch_fa_sm", -1)) <= 0:
            cfg.layer_batch_fa_sm = max(8, min(total_sm - 8, int(round(total_sm * 0.4))))
        if int(getattr(cfg, "layer_batch_la_sm", -1)) <= 0:
            cfg.layer_batch_la_sm = max(8, total_sm - cfg.layer_batch_fa_sm)

        # ---- LayerBatch upper bs ----
        if int(getattr(cfg, "layer_batch_max_bs", -1)) <= 0:
            cfg.layer_batch_max_bs = int(cfg.max_num_seqs)

        # ---- LayerBatch max total tokens ----
        if int(getattr(cfg, "layer_batch_max_total_tokens", -1)) <= 0:
            cfg.layer_batch_max_total_tokens = int(cfg.max_model_len) * int(cfg.max_num_seqs)

        # ---- Dynamic SM partitions auto-policy ----
        # Only auto-fill when user explicitly enabled LB and did NOT pick simple/no-greenctx.
        if (cfg.enable_layer_batch
                and not getattr(cfg, "layer_batch_simple", False)
                and not getattr(cfg, "layer_batch_no_greenctx", False)
                and not list(getattr(cfg, "layer_batch_partitions", []) or [])):
            mlen = int(cfg.max_model_len)
            short_thr = max(1024, mlen // 8)
            mid_thr   = max(short_thr * 2, mlen // 2)
            long_thr  = max(mid_thr * 2, mlen * 8)
            la_heavy_fa = max(8, min(total_sm - 8, int(round(total_sm * 0.30))))
            balanced_fa = max(8, min(total_sm - 8, int(round(total_sm * 0.50))))
            fa_heavy_fa = max(8, min(total_sm - 8, int(round(total_sm * 0.70))))
            cfg.layer_batch_partitions = [
                (la_heavy_fa,           total_sm - la_heavy_fa,  short_thr),
                (balanced_fa,           total_sm - balanced_fa,  mid_thr),
                (fa_heavy_fa,           total_sm - fa_heavy_fa,  long_thr),
            ]
            logger.info("Auto-filled layer_batch_partitions for total_sm=%d, max_model_len=%d: %s",
                        total_sm, mlen, cfg.layer_batch_partitions)

        # ---- POD tile config defaults ----
        # head_dim, num_q_heads, num_kv_heads from model config
        head_dim = int(getattr(hf_text, "head_dim",
                               getattr(hf_text, "hidden_size", 0) // max(1, getattr(hf_text, "num_attention_heads", 1))))
        n_q  = int(getattr(hf_text, "num_attention_heads", 1))
        n_kv = int(getattr(hf_text, "num_key_value_heads", n_q))
        kv_group = max(1, n_q // max(1, n_kv))

        if int(getattr(cfg, "pod_num_kv_splits", -1)) <= 0:
            # cap to 32, floor 4; scale with ctx length
            base = max(4, min(32, int(cfg.max_model_len) // 8192))
            # round up to nearest power of 2 for kernel alignment
            p2 = 1
            while p2 < base:
                p2 <<= 1
            cfg.pod_num_kv_splits = max(4, p2)
        if int(getattr(cfg, "pod_block_n", -1)) <= 0:
            cfg.pod_block_n = 64
        if int(getattr(cfg, "pod_block_h", -1)) <= 0:
            cfg.pod_block_h = max(16, kv_group)

        self._pod_runtime_cfg = dict(
            num_kv_splits=cfg.pod_num_kv_splits,
            block_n=cfg.pod_block_n,
            block_h=cfg.pod_block_h,
            num_warps=cfg.pod_num_warps,
            num_stages=cfg.pod_num_stages,
            gemm_block_m=cfg.pod_gemm_block_m,
            gemm_block_n=cfg.pod_gemm_block_n,
            gemm_block_k=cfg.pod_gemm_block_k,
        )

        # ---- Prefill-LayerBatch SM defaults ----
        if int(getattr(cfg, "prefill_lb_low_hit_sm", -1)) <= 0:
            cfg.prefill_lb_low_hit_sm = max(8, min(total_sm - 8, int(round(total_sm * 0.7))))
        if int(getattr(cfg, "prefill_lb_high_hit_sm", -1)) <= 0:
            cfg.prefill_lb_high_hit_sm = max(8, total_sm - cfg.prefill_lb_low_hit_sm)

        if self.rank == 0:
            logger.info("Resolved dynamic defaults: total_sm=%d, fa_sm=%d/la_sm=%d, "
                        "lb_max_bs=%d, lb_max_total_tok=%d, pod_kv_splits=%d, "
                        "pod_block_n=%d, pod_block_h=%d (head_dim=%d, kv_group=%d)",
                        total_sm, cfg.layer_batch_fa_sm, cfg.layer_batch_la_sm,
                        cfg.layer_batch_max_bs, cfg.layer_batch_max_total_tokens,
                        cfg.pod_num_kv_splits, cfg.pod_block_n, cfg.pod_block_h,
                        head_dim, kv_group)

    def _layer_batch_eligible(self, is_prefill: bool, bs: int, total_tokens: int = -1) -> bool:
        if is_prefill:
            return False
        if not getattr(self.config, "enable_layer_batch", False):
            return False
        if not self._lb_partitions and not self._lb_simple:
            return False
        if bs < int(self.config.layer_batch_min_bs):
            return False
        max_bs = int(getattr(self.config, 'layer_batch_max_bs', 0))
        if max_bs > 0 and bs > max_bs:
            return False
        # New: total-token gates (sum(context_lens) + bs).
        # `total_tokens < 0` means caller didn't compute it — skip the gate (back-compat).
        if total_tokens >= 0:
            min_t = int(getattr(self.config, "layer_batch_min_total_tokens", 0))
            max_t = int(getattr(self.config, "layer_batch_max_total_tokens", 10**12))
            if total_tokens < min_t or total_tokens > max_t:
                return False
        # Need at least 2 nano-batches; both must be non-empty.
        split = float(self.config.layer_batch_split)
        B1 = max(1, min(bs - 1, int(round(bs * split))))
        return 1 <= B1 < bs

    def _pick_partition_idx(self, max_ctx_len: int) -> int:
        """Return index of the FIRST partition whose max_ctx_thr >= max_ctx_len.
           Falls back to the last partition if none qualifies."""
        for i, p in enumerate(self._lb_partitions):
            if max_ctx_len <= p["max_ctx_thr"]:
                return i
        return len(self._lb_partitions) - 1

    def _run_layer_batch_decode(self, input_ids: torch.Tensor,
                                 positions: torch.Tensor) -> torch.Tensor:
        bs = input_ids.size(0)
        split = float(self.config.layer_batch_split)
        ctx = get_context()
        max_ctx_len = int(getattr(ctx, "max_ctx_len", 0) or 0)
        part_idx = self._pick_partition_idx(max_ctx_len)
        partition = self._lb_partitions[part_idx]
        fa_stream = partition["fa_stream"]
        la_stream = partition["la_stream"]

        # Use captured graph if available for this partition
        if self._lb_graphs:
            # Pick smallest captured bs >= bs whose (bs, B1, part_idx) is captured.
            def _bucket_ok(b):
                B1_b = max(1, min(b - 1, int(round(b * split))))
                return (b, B1_b, part_idx) in self._lb_graphs
            bucket_bs = next((b for b in self.graph_bs if b >= bs and _bucket_ok(b)), None)
            if bucket_bs is not None:
                B1 = max(1, min(bucket_bs - 1, int(round(bucket_bs * split))))
                gv = self.graph_vars
                # Stage inputs into static buffers (all on the default stream)
                gv["input_ids"][:bs] = input_ids
                gv["positions"][:bs] = positions
                gv["slot_mapping"].fill_(-1)
                gv["slot_mapping"][:bs] = ctx.slot_mapping
                gv["context_lens"].zero_()
                gv["context_lens"][:bs] = ctx.context_lens
                gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
                if "linear_attn_slot_indices" in gv and ctx.linear_attn_slot_indices is not None:
                    gv["linear_attn_slot_indices"].zero_()
                    gv["linear_attn_slot_indices"][:bs] = ctx.linear_attn_slot_indices
                # bs<bucket_bs: pad with a safe slot (one not in use).
                if bs < bucket_bs:
                    if "linear_attn_slot_indices" in gv and ctx.linear_attn_slot_indices is not None:
                        used = set(ctx.linear_attn_slot_indices[:bs].tolist())
                        free_slot = next((s_ for s_ in range(self.config.max_num_seqs) if s_ not in used), 0)
                        gv["linear_attn_slot_indices"][bs:bucket_bs] = free_slot
                    gv["context_lens"][bs:bucket_bs] = 1
                    if bs > 0:
                        gv["block_tables"][bs:bucket_bs, :ctx.block_tables.size(1)] = ctx.block_tables[-1:]
                graph = self._lb_graphs[(bucket_bs, B1, part_idx)]
                # Annotate timeline so nsys shows which partition is running.
                torch.cuda.nvtx.range_push(f"LB_replay[part{part_idx} bs={bucket_bs}]")
                # The captured graph runs on `la_stream` of THIS partition; sync the
                # default stream's input writes to it before replay, then drain back.
                la_stream.wait_stream(torch.cuda.current_stream())
                graph.replay()
                torch.cuda.current_stream().wait_stream(la_stream)
                torch.cuda.nvtx.range_pop()
                return self.model.compute_logits(gv["outputs"][:bs])

        # Eager fallback (no captured graph for this (bs, partition) bucket)
        B1 = max(1, min(bs - 1, int(round(bs * split))))
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            lm = getattr(self.model, "model", None)
        torch.cuda.nvtx.range_push(f"LB_eager[part{part_idx}]")
        hidden = run_layer_batch_decode(
            language_model=lm,
            input_ids=input_ids,
            positions=positions,
            layer_types=self._layer_types,
            B1=B1,
            fa_stream=fa_stream,
            la_stream=la_stream,
        )
        torch.cuda.nvtx.range_pop()
        return self.model.compute_logits(hidden)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))

        bs = input_ids.size(0)

        # Layer-Batch parallel decode (Green Context dual-stream).
        ctx_obj = get_context()
        total_tokens = int(getattr(ctx_obj, "total_tokens", 0) or 0)
        if self._lb_simple and self._layer_batch_eligible(is_prefill, bs, total_tokens=total_tokens):
            return self._run_layer_batch_simple(input_ids, positions)
        if self._layer_batch_eligible(is_prefill, bs, total_tokens=total_tokens):
            return self._run_layer_batch_decode(input_ids, positions)

        # Standard CUDA-graph decode.
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        if "linear_attn_slot_indices" in graph_vars and context.linear_attn_slot_indices is not None:
            graph_vars["linear_attn_slot_indices"].zero_()
            graph_vars["linear_attn_slot_indices"][:bs] = context.linear_attn_slot_indices
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def _run_layer_batch_simple(self, input_ids: torch.Tensor,
                                  positions: torch.Tensor) -> torch.Tensor:
        """Streamlined LB decode: 2 streams, no Green Context, no partition skeleton.

        - Same nano-batch lag-by-1 schedule (`run_layer_batch_decode`).
        - Both streams share all 78 SMs; the GPU grid scheduler co-locates
          their CTAs only when there are spare SMs (opportunistic parallelism).
        - Captured CUDA graph keyed by `bucket_bs` only (no partition_idx in key).
        """
        bs = input_ids.size(0)
        split = float(self.config.layer_batch_split)
        ctx = get_context()
        fa_stream = self._lb_simple_fa_stream
        la_stream = self._lb_simple_la_stream

        if self._lb_simple_graphs:
            bucket_bs = next((b for b in self.graph_bs if b >= bs and b in self._lb_simple_graphs), None)
            if bucket_bs is not None:
                B1, graph = self._lb_simple_graphs[bucket_bs]
                gv = self.graph_vars
                # Stage inputs into static buffers (default stream).
                gv["input_ids"][:bs] = input_ids
                gv["positions"][:bs] = positions
                gv["slot_mapping"].fill_(-1)
                gv["slot_mapping"][:bs] = ctx.slot_mapping
                gv["context_lens"].zero_()
                gv["context_lens"][:bs] = ctx.context_lens
                gv["block_tables"][:bs, :ctx.block_tables.size(1)] = ctx.block_tables
                if "linear_attn_slot_indices" in gv and ctx.linear_attn_slot_indices is not None:
                    gv["linear_attn_slot_indices"].zero_()
                    gv["linear_attn_slot_indices"][:bs] = ctx.linear_attn_slot_indices
                if bs < bucket_bs:
                    if "linear_attn_slot_indices" in gv and ctx.linear_attn_slot_indices is not None:
                        used = set(ctx.linear_attn_slot_indices[:bs].tolist())
                        free_slot = next((s_ for s_ in range(self.config.max_num_seqs) if s_ not in used), 0)
                        gv["linear_attn_slot_indices"][bs:bucket_bs] = free_slot
                    gv["context_lens"][bs:bucket_bs] = 1
                    if bs > 0:
                        gv["block_tables"][bs:bucket_bs, :ctx.block_tables.size(1)] = ctx.block_tables[-1:]
                torch.cuda.nvtx.range_push(f"LBsimple_replay[bs={bucket_bs}]")
                la_stream.wait_stream(torch.cuda.current_stream())
                graph.replay()
                torch.cuda.current_stream().wait_stream(la_stream)
                torch.cuda.nvtx.range_pop()
                return self.model.compute_logits(gv["outputs"][:bs])

        # Eager fallback
        B1 = max(1, min(bs - 1, int(round(bs * split))))
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            lm = getattr(self.model, "model", None)
        torch.cuda.nvtx.range_push("LBsimple_eager")
        hidden = run_layer_batch_decode(
            language_model=lm, input_ids=input_ids, positions=positions,
            layer_types=self._layer_types, B1=B1,
            fa_stream=fa_stream, la_stream=la_stream,
        )
        torch.cuda.nvtx.range_pop()
        return self.model.compute_logits(hidden)

    @torch.inference_mode()
    def capture_layer_batch_simple_graphs(self):
        from nanovllm.utils.context import Context
        import nanovllm.utils.context as _ctxmod
        """Capture LB graphs keyed by bucket_bs only (single-partition).

        Reuses the standard CUDA-graph buffers (`graph_vars`) so the simple-LB
        path shares input staging with the standard decode path.  graph_pool
        is NOT shared with the standard captures (cross-stream caching-allocator
        interaction with ExternalStream is unreliable; see precision-fix notes).
        """
        if not self._lb_simple:
            return
        if not hasattr(self, "graph_vars"):
            logger.warning("Simple-LB capture skipped: standard decode graph_vars not allocated.")
            return

        config = self.config
        gv = self.graph_vars
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Use existing static buffers from capture_cudagraph
        in_full = gv["input_ids"]
        pos_full = gv["positions"]
        slot_full = gv["slot_mapping"]
        cl_full = gv["context_lens"]
        bt_full = gv["block_tables"]
        out_full = gv["outputs"]
        las_full = gv.get("linear_attn_slot_indices")

        split = float(config.layer_batch_split)
        min_bs = int(config.layer_batch_min_bs)
        graph_bs = [b for b in self.graph_bs if b >= min_bs and b <= max_bs]
        graph_bs = [b for b in graph_bs if b >= 2]

        lm = getattr(self.model, "language_model", None)
        if lm is None:
            lm = getattr(self.model, "model", None)

        n_captured = 0
        # Each capture gets its own private pool — matches capture_layer_batch_graphs.
        for bs in reversed(graph_bs):
            B1 = max(1, min(bs - 1, int(round(bs * split))))
            if not (1 <= B1 < bs):
                continue
            full_ctx = Context(
                is_prefill=False,
                slot_mapping=slot_full[:bs], context_lens=cl_full[:bs],
                block_tables=bt_full[:bs],
                linear_attn_slot_indices=(las_full[:bs] if las_full is not None else None),
            )

            # Eager warmup so allocations happen outside graph capture.
            _ctxmod._CONTEXT = full_ctx
            try:
                _ = run_layer_batch_decode(
                    lm, in_full[:bs], pos_full[:bs],
                    self._layer_types, B1,
                    fa_stream=self._lb_simple_fa_stream,
                    la_stream=self._lb_simple_la_stream,
                )
            except Exception as e:
                logger.warning("Simple-LB warmup failed at bs=%d (%s); skipping this bucket.", bs, e)
                _ctxmod._CONTEXT = Context()
                continue

            graph = torch.cuda.CUDAGraph()
            try:
                _ctxmod._CONTEXT = full_ctx
                with torch.cuda.graph(graph, stream=self._lb_simple_la_stream):
                    hidden = run_layer_batch_decode(
                        lm, in_full[:bs], pos_full[:bs],
                        self._layer_types, B1,
                        fa_stream=self._lb_simple_fa_stream,
                        la_stream=self._lb_simple_la_stream,
                    )
                    out_full[:bs] = hidden
                self._lb_simple_graphs[bs] = (B1, graph)
                n_captured += 1
            except Exception as e:
                logger.warning("Simple-LB capture failed at bs=%d (%s); skipping this bucket.", bs, e)
            finally:
                _ctxmod._CONTEXT = Context()
                torch.cuda.synchronize()

        logger.info(
            "Layer-Batch SIMPLE: captured %d CUDA graphs (split=%.2f, bs buckets=%s)",
            n_captured, split, sorted(self._lb_simple_graphs.keys()),
        )

    @torch.inference_mode()
    def capture_layer_batch_graphs(self):
        """Capture CUDA graphs for the Layer-Batch decode path.

        Reuses the static buffers from `capture_cudagraph` (input_ids, positions,
        slot_mapping, context_lens, block_tables, linear_attn_slot_indices, outputs).
        For each (bs, B1) bucket we capture one graph that runs the layer-batched
        forward pass using two Green-Context streams.
        """
        from nanovllm.engine.layer_batch import run_layer_batch_decode
        from nanovllm.utils.context import Context
        import nanovllm.utils.context as _ctxmod

        gv = self.graph_vars
        sm_full = gv["slot_mapping"]
        cl_full = gv["context_lens"]
        bt_full = gv["block_tables"]
        li_full = gv.get("linear_attn_slot_indices")
        out_full = gv["outputs"]
        in_full  = gv["input_ids"]
        pos_full = gv["positions"]
        max_blocks = bt_full.size(1)

        split = float(self.config.layer_batch_split)
        min_bs = max(2, int(self.config.layer_batch_min_bs))
        graph_bs = [b for b in self.graph_bs if b >= min_bs]

        lm = getattr(self.model, "language_model", None) or getattr(self.model, "model", None)
        layer_types = self._layer_types

        n_captured = 0
        # Outer loop: for each (fa_sm, la_sm) partition we capture an independent
        # set of (bs, B1) graphs.  Each captured graph references THAT partition's
        # streams, so replaying it actually exercises that SM split.  The cost is
        # `len(graph_bs) * len(partitions)` graphs (default 4 × 3 = 12 graphs,
        # ~50 MB each).
        for part_idx, partition in enumerate(self._lb_partitions):
            fa_s = partition["fa_stream"]
            la_s = partition["la_stream"]
            for bs in reversed(graph_bs):
                B1 = max(1, min(bs - 1, int(round(bs * split))))
                B2 = bs - B1
                if B1 < 1 or B2 < 1:
                    continue

                full_ctx = Context(
                    is_prefill=False,
                    slot_mapping=sm_full[:bs],
                    context_lens=cl_full[:bs],
                    block_tables=bt_full[:bs, :max_blocks],
                    linear_attn_slot_indices=(li_full[:bs] if li_full is not None else None),
                )

                # Per-bucket try/except: a single (bs, partition) bucket that
                # fails to capture (e.g. tiny bs where some kernel asserts
                # "batch size must be positive") MUST NOT take down the whole LB
                # init.  Skip the bucket and continue — runtime will fall back
                # to the standard CUDA-graph for those bs values.
                try:
                    # Warmup once (needed before capture, and per-partition so Triton
                    # autotune cache and cuBLAS handles all settle on this stream pair).
                    _ctxmod._CONTEXT = full_ctx
                    _ = run_layer_batch_decode(lm, in_full[:bs], pos_full[:bs],
                                                layer_types, B1, fa_s, la_s)
                    torch.cuda.synchronize()

                    graph = torch.cuda.CUDAGraph()
                    _ctxmod._CONTEXT = full_ctx
                    with torch.cuda.graph(graph, stream=la_s):
                        hidden = run_layer_batch_decode(
                            lm, in_full[:bs], pos_full[:bs],
                            layer_types, B1, fa_s, la_s,
                        )
                        out_full[:bs] = hidden
                    torch.cuda.synchronize()
                    self._lb_graphs[(bs, B1, part_idx)] = graph
                    n_captured += 1
                except Exception as e:
                    logger.warning("LB capture skipped: bs=%d part_idx=%d (B1=%d, B2=%d): %s",
                                   bs, part_idx, B1, B2, e)
                finally:
                    _ctxmod._CONTEXT = Context()

        logger.info("Layer-Batch: captured %d CUDA graphs (split=%.2f, partitions=%d, bs buckets=%s)",
                    n_captured, split, len(self._lb_partitions),
                    sorted({k[0] for k in self._lb_graphs}))

    def _classify_prefill_seqs(self, seqs: list[Sequence]):
        """Classify prefill seqs by prefix-cache hit ratio.

        Returns (low_idx, high_idx, eligible).  Eligible iff prefill-LB is on,
        both groups non-empty, and at least one seq has length >= min_len.
        """
        cfg = self.config
        if not (cfg.enable_prefill_layer_batch and self._lb_prefill_partition):
            return None, None, False
        threshold = float(cfg.prefill_lb_hit_threshold)
        min_len   = int(cfg.prefill_lb_min_len)
        low_idx, high_idx = [], []
        has_long = False
        for i, seq in enumerate(seqs):
            n = len(seq)
            hit = (seq.num_cached_tokens / n) if n > 0 else 0.0
            if hit < threshold:
                low_idx.append(i)
            else:
                high_idx.append(i)
            if n >= min_len:
                has_long = True
        eligible = (len(low_idx) >= 1 and len(high_idx) >= 1 and has_long)
        return low_idx, high_idx, eligible

    @torch.inference_mode()
    def _run_prefill_split(self, seqs: list[Sequence],
                            low_idx: list[int], high_idx: list[int]) -> torch.Tensor:
        """Run prefill with cache-hit-aware dual-stream split.

        Both groups run their FULL model.forward() on independent streams under
        a Green Context partition with asymmetric SM allocation:
          - low_idx (low cache hit, compute-bound)  -> low_stream  (~70% SMs)
          - high_idx (high cache hit, HBM-bound)    -> high_stream (~30% SMs)

        Streams hold ExternalStream objects from green_ctx; PyTorch's caching
        allocator does not track their lifetimes automatically, so we
        record_stream on every input tensor that crosses into them.

        Returns logits [len(seqs), vocab] in the ORIGINAL seqs order.
        """
        from nanovllm.utils.context import set_context as _set_ctx
        from nanovllm.layers import flashinfer_attn as _fi

        low_seqs  = [seqs[i] for i in low_idx]
        high_seqs = [seqs[i] for i in high_idx]
        in_low,  pos_low,  ctx_low  = self._build_prefill_inputs(low_seqs)
        in_high, pos_high, ctx_high = self._build_prefill_inputs(high_seqs)

        # FlashInfer per-subset plan (only when actually usable — Ampere
        # skips this path because get_use_flashinfer_prefill() is False).
        fi_low_qo  = ctx_low.pop("_fi_qo_indptr_cpu", None)
        fi_low_sk  = ctx_low.pop("_fi_seqlens_k", None)
        fi_low_bl  = ctx_low.pop("_fi_block_lists", None)
        fi_high_qo = ctx_high.pop("_fi_qo_indptr_cpu", None)
        fi_high_sk = ctx_high.pop("_fi_seqlens_k", None)
        fi_high_bl = ctx_high.pop("_fi_block_lists", None)
        if _fi.get_use_flashinfer_prefill():
            if fi_low_qo is not None and low_seqs:
                self._plan_flashinfer("low", fi_low_qo, fi_low_sk, fi_low_bl)
                ctx_low["fi_slot"] = "low"
            if fi_high_qo is not None and high_seqs:
                self._plan_flashinfer("high", fi_high_qo, fi_high_sk, fi_high_bl)
                ctx_high["fi_slot"] = "high"

        part = self._lb_prefill_partition
        s_low  = part["low_stream"]
        s_high = part["high_stream"]

        # Cross-stream visibility for input tensors (allocator bookkeeping).
        for t in (in_low, pos_low, ctx_low["cu_seqlens_q"], ctx_low["cu_seqlens_k"],
                  ctx_low["slot_mapping"]):
            t.record_stream(s_low)
        if ctx_low["block_tables"] is not None:
            ctx_low["block_tables"].record_stream(s_low)
        if ctx_low["linear_attn_slot_indices"] is not None:
            ctx_low["linear_attn_slot_indices"].record_stream(s_low)
        for t in (in_high, pos_high, ctx_high["cu_seqlens_q"], ctx_high["cu_seqlens_k"],
                  ctx_high["slot_mapping"]):
            t.record_stream(s_high)
        if ctx_high["block_tables"] is not None:
            ctx_high["block_tables"].record_stream(s_high)
        if ctx_high["linear_attn_slot_indices"] is not None:
            ctx_high["linear_attn_slot_indices"].record_stream(s_high)

        cur = torch.cuda.current_stream()
        s_low.wait_stream(cur)
        s_high.wait_stream(cur)

        torch.cuda.nvtx.range_push(f"prefill-split[low={len(low_seqs)} high={len(high_seqs)}]")
        # Group A: low-hit (compute-bound) on the bigger SM partition.
        _set_ctx(**ctx_low)
        with torch.cuda.stream(s_low):
            logits_low = self.model.compute_logits(self.model(in_low, pos_low))

        # Group B: high-hit (HBM-bound) on the smaller SM partition.
        _set_ctx(**ctx_high)
        with torch.cuda.stream(s_high):
            logits_high = self.model.compute_logits(self.model(in_high, pos_high))

        cur.wait_stream(s_low)
        cur.wait_stream(s_high)
        torch.cuda.nvtx.range_pop()

        # Reorder logits back to the original seqs order.
        out = torch.empty(len(seqs), logits_low.shape[-1],
                          dtype=logits_low.dtype, device=logits_low.device)
        if low_idx:
            idx_t = torch.tensor(low_idx, dtype=torch.long, device=out.device)
            out.index_copy_(0, idx_t, logits_low)
        if high_idx:
            idx_t = torch.tensor(high_idx, dtype=torch.long, device=out.device)
            out.index_copy_(0, idx_t, logits_high)
        return out

    def _effective_chunk_size(self) -> int:
        """Chunk size aligned up to a multiple of `kvcache_block_size`.

        Alignment matters: a chunk that ends mid-block would leave the last
        block partially populated, and the NEXT chunk's slot_mapping loop
        `range(num_cached_blocks, num_blocks)` starts on the next FULL block,
        skipping the partial-block tokens.  Aligning to the block size makes
        every intermediate boundary land on a block edge so the mapping
        stays contiguous and correct.
        """
        raw = int(self.config.prefill_chunk_size)
        blk = int(self.block_size)
        return ((raw + blk - 1) // blk) * blk

    @torch.inference_mode()
    def _run_chunked_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Chunked prefill: split each seq's un-cached prefill work into
        block-aligned chunks and process them sequentially through the model.

        Peak activation memory drops from O(max_new_tokens) to O(chunk_size).

        Correctness invariants (see DESIGN_CHUNKED_PREFILL.md for the full
        argument):

          1. Attention (paged FA / FlashInfer / SDPA) reads all prior K/V
             from the paged cache using `block_tables + cu_seqlens_k`.  Each
             chunk writes its own slot-mapped K/V before running attention,
             so the cache is fresh at read time.
          2. LA (GatedDeltaNet) recurrent + conv states are chained across
             chunks via the persistent per-slot buffers.  Chunk 0 starts
             from zeros (identical to the non-chunked path).  Chunks 1+ set
             `context.la_chunk_continuation = True` and the LA layer loads
             `recurrent_state_buf[slot]` as h0 and prepends
             `conv_state_buf[slot]` to the causal conv1d input.
          3. RoPE positions are absolute (`range(chunk_start, chunk_end)`),
             so the sinusoidal frequencies match a non-chunked run exactly.
          4. Only the LAST chunk of each seq contributes a sampled token —
             the LM head is only evaluated on positions that are the seq's
             final position.

        Returns: [B, vocab] logits (one row per input seq) suitable for
        `self.sampler`.
        """
        from nanovllm.utils.context import set_context, reset_context
        chunk_size = self._effective_chunk_size()
        B = len(seqs)

        # Snapshot per-seq state so we can restore even if a chunk raises.
        orig = [(s.num_cached_tokens, s.num_tokens) for s in seqs]

        # Per-seq chunk boundary list of (start, end) tuples.  Chunks are
        # aligned to the block size so slot_mapping stays consistent (see
        # _effective_chunk_size docstring).
        per_seq_chunks: list[list[tuple[int, int]]] = []
        for s in seqs:
            cached = s.num_cached_tokens
            end = s.num_tokens
            if end - cached <= chunk_size:
                # Single-chunk seq — degenerate case, still allowed.
                per_seq_chunks.append([(cached, end)])
                continue
            chunks = []
            cursor = cached
            while cursor < end:
                # Round the chunk end to a block boundary unless we're at
                # the actual seq end (which may be partial).
                proposed_end = cursor + chunk_size
                if proposed_end < end:
                    proposed_end = (proposed_end // self.block_size) * self.block_size
                    # Guard against a degenerate case where the alignment
                    # would leave zero progress.
                    if proposed_end <= cursor:
                        proposed_end = cursor + self.block_size
                    proposed_end = min(proposed_end, end)
                else:
                    proposed_end = end
                chunks.append((cursor, proposed_end))
                cursor = proposed_end
            per_seq_chunks.append(chunks)

        max_chunks = max(len(cs) for cs in per_seq_chunks)
        if self.rank == 0:
            logger.info("[ChunkedPrefill] bs=%d chunk_size=%d max_chunks=%d "
                        "per_seq_chunk_counts=%s new_tokens=%s",
                        B, chunk_size, max_chunks,
                        [len(cs) for cs in per_seq_chunks],
                        [seqs[i].num_tokens - orig[i][0] for i in range(B)])

        # Placeholders for the finishing-chunk logits, in ORIGINAL seq order.
        stashed: list[torch.Tensor | None] = [None] * B
        vocab_size = None
        sample_dtype = None
        sample_device = None

        try:
            for chunk_idx in range(max_chunks):
                # Which seqs still have work at this chunk index?
                active_indices = [i for i in range(B) if chunk_idx < len(per_seq_chunks[i])]
                active_seqs = [seqs[i] for i in active_indices]

                # Adjust each active seq's num_cached_tokens / num_tokens to
                # this chunk's window.  The block manager is not touched;
                # blocks were already allocated by the scheduler for the
                # full seq before run() was called.
                for i in active_indices:
                    cs, ce = per_seq_chunks[i][chunk_idx]
                    seqs[i].num_cached_tokens = cs
                    seqs[i].num_tokens = ce

                input_ids, positions, ctx_kwargs = self._build_prefill_inputs(active_seqs)
                fi_qo = ctx_kwargs.pop("_fi_qo_indptr_cpu", None)
                fi_seqlens = ctx_kwargs.pop("_fi_seqlens_k", None)
                fi_block_lists = ctx_kwargs.pop("_fi_block_lists", None)
                ctx_kwargs["fi_slot"] = "default"
                # LA continuation only kicks in from the second chunk onward.
                ctx_kwargs["la_chunk_continuation"] = (chunk_idx > 0)
                set_context(**ctx_kwargs)

                from nanovllm.layers import flashinfer_attn as _fi
                _fi.reset_planned_flag()
                _fi.reset_ragged_planned()
                if _fi.get_use_flashinfer_prefill():
                    if fi_qo is not None and fi_seqlens is not None and fi_block_lists is not None:
                        self._plan_flashinfer("default", fi_qo, fi_seqlens, fi_block_lists)
                    else:
                        qo_indptr = [0]
                        for s in active_seqs:
                            qo_indptr.append(qo_indptr[-1] + s.num_tokens - s.num_cached_tokens)
                        qo_indptr_cpu = torch.tensor(qo_indptr, dtype=torch.int32, device="cpu")
                        self._plan_flashinfer_ragged(qo_indptr_cpu)

                # Forward + logits.  compute_logits internally slices to the
                # last position of each seq via cu_seqlens_q, giving one row
                # per active seq.  Non-finishing rows are discarded.
                hidden = self.model(input_ids, positions)
                step_logits = self.model.compute_logits(hidden)  # [num_active, V] on rank 0

                if self.rank == 0 and step_logits is not None:
                    if vocab_size is None:
                        vocab_size = step_logits.shape[-1]
                        sample_dtype = step_logits.dtype
                        sample_device = step_logits.device
                    for local_pos, seq_idx in enumerate(active_indices):
                        if chunk_idx == len(per_seq_chunks[seq_idx]) - 1:
                            # Finishing chunk for this seq.  Copy out so we
                            # can free `step_logits` at the end of the loop.
                            stashed[seq_idx] = step_logits[local_pos:local_pos + 1].clone()

                # Free intermediates before the next chunk allocates fresh ones.
                del hidden, step_logits, input_ids, positions
                reset_context()
        finally:
            # Restore per-seq state so downstream (block manager, scheduler)
            # sees the original values.
            for i, s in enumerate(seqs):
                s.num_cached_tokens = orig[i][0]
                s.num_tokens = orig[i][1]

        if self.rank != 0:
            return None
        # Assemble final logits in original seq order.
        assert all(l is not None for l in stashed), \
            "chunked prefill: some seq did not produce a finishing-chunk logit"
        final = torch.cat(stashed, dim=0)
        return final

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if self.rank == 0 and is_prefill:
            logger.info("[Prefill-LB] entering run: bs=%d enable_pref_lb=%s part=%s",
                        len(seqs), self.config.enable_prefill_layer_batch,
                        "set" if self._lb_prefill_partition else "None")
        # Prefill-LayerBatch: cache-hit-aware dual-stream prefill.
        # Chunked prefill: decides ONE step-time whether to fire based on the
        # per-seq new-token counts.  Fires when any seq exceeds chunk_size,
        # otherwise falls through to the normal path (chunked has extra
        # per-chunk Python overhead we don't want to pay for short prompts).
        if is_prefill and self.config.enable_chunked_prefill:
            chunk_size = self._effective_chunk_size()
            new_tokens_max = max((s.num_tokens - s.num_cached_tokens) for s in seqs)
            if new_tokens_max > chunk_size:
                if self.rank == 0:
                    logger.info("[ChunkedPrefill] FIRE: bs=%d max_new=%d chunk_size=%d",
                                len(seqs), new_tokens_max, chunk_size)
                temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
                logits = self._run_chunked_prefill(seqs)
                token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
                reset_context()
                return token_ids
        if is_prefill and self.config.enable_prefill_layer_batch and self._lb_prefill_partition is not None:
            low_idx, high_idx, eligible = self._classify_prefill_seqs(seqs)
            if eligible:
                if self.rank == 0:
                    logger.info("[Prefill-LB] FIRE: low_idx=%s high_idx=%s, lens=%s",
                                low_idx, high_idx, [len(s) for s in seqs])
                temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
                logits = self._run_prefill_split(seqs, low_idx, high_idx)
                token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
                reset_context()
                return token_ids
            else:
                if self.rank == 0:
                    logger.info("[Prefill-LB] skip: bs=%d low=%d high=%d eligible=%s lens=%s",
                                len(seqs),
                                len(low_idx) if low_idx else 0,
                                len(high_idx) if high_idx else 0,
                                eligible,
                                [len(seq) for seq in seqs])

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # Pre-allocate linear attention slot indices for graph capture
        linear_attn_slot_indices = None
        if self._has_linear_attn:
            linear_attn_slot_indices = torch.zeros(max_bs, dtype=torch.int64)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs],
                        block_tables=block_tables[:bs],
                        linear_attn_slot_indices=linear_attn_slot_indices[:bs] if linear_attn_slot_indices is not None else None)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if linear_attn_slot_indices is not None:
            self.graph_vars["linear_attn_slot_indices"] = linear_attn_slot_indices
