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

        import datetime
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank,
                                timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(rank)
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

        self.warmup_model()
        self._compute_linear_attn_budget()
        self.allocate_kv_cache()
        if self._has_linear_attn:
            self.allocate_linear_attn_states()
        if not self.enforce_eager:
            self.capture_cudagraph()
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
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
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

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        linear_attn_slots = self._get_linear_attn_slot_indices(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables,
                    seq_ids=[seq.seq_id for seq in seqs],
                    linear_attn_slot_indices=linear_attn_slots)
        return input_ids, positions

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
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        linear_attn_slots = self._get_linear_attn_slot_indices(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                    seq_ids=[seq.seq_id for seq in seqs],
                    linear_attn_slot_indices=linear_attn_slots)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
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
            # Update linear attention slot indices for CUDA Graph
            if "linear_attn_slot_indices" in graph_vars and context.linear_attn_slot_indices is not None:
                graph_vars["linear_attn_slot_indices"].zero_()
                graph_vars["linear_attn_slot_indices"][:bs] = context.linear_attn_slot_indices
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
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
