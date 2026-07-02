import os
import json
from dataclasses import dataclass, field
from transformers import AutoConfig


class Qwen3_5DenseTextConfig:
    """Config for Qwen3.5 Dense text model (hybrid attention + dense MLP)."""

    def __init__(self, config_dict):
        text_config = config_dict.get('text_config', config_dict)
        self.hidden_size = text_config['hidden_size']
        self.num_hidden_layers = text_config['num_hidden_layers']
        self.num_attention_heads = text_config['num_attention_heads']
        self.num_key_value_heads = text_config['num_key_value_heads']
        self.head_dim = text_config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.hidden_act = text_config['hidden_act']
        self.intermediate_size = text_config['intermediate_size']
        self.max_position_embeddings = text_config['max_position_embeddings']
        self.rms_norm_eps = text_config['rms_norm_eps']
        self.vocab_size = text_config['vocab_size']
        self.layer_types = text_config['layer_types']

        # Linear attention params
        self.linear_num_key_heads = text_config['linear_num_key_heads']
        self.linear_num_value_heads = text_config['linear_num_value_heads']
        self.linear_key_head_dim = text_config['linear_key_head_dim']
        self.linear_value_head_dim = text_config['linear_value_head_dim']
        self.linear_conv_kernel_dim = text_config['linear_conv_kernel_dim']

        # RoPE params
        self.rope_parameters = text_config.get('rope_parameters', {})

        # dtype
        dtype_str = text_config.get('dtype', 'bfloat16')
        import torch
        self.torch_dtype = getattr(torch, dtype_str, torch.bfloat16)


class Qwen3_5DenseConfig:
    """Top-level config for Qwen3.5 Dense model."""

    def __init__(self, config_path):
        config_file = None
        if os.path.isdir(config_path):
            for name in ['config.json']:
                candidate = os.path.join(config_path, name)
                if os.path.isfile(candidate):
                    config_file = candidate
                    break
        elif os.path.isfile(config_path):
            config_file = config_path

        if config_file is None:
            raise FileNotFoundError(f"No config file found in {config_path}")

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        self.model_type = config_dict.get('model_type', 'qwen3_5')
        self.tie_word_embeddings = config_dict.get('tie_word_embeddings', False)
        self.text_config = Qwen3_5DenseTextConfig(config_dict)

        # Expose text_config attributes at top level for compatibility
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.max_position_embeddings = self.text_config.max_position_embeddings
        self.vocab_size = self.text_config.vocab_size
        self.torch_dtype = self.text_config.torch_dtype


class Qwen3_5MoeTextConfig:
    """Config for Qwen3.5 MoE text model, loaded from local config file."""

    def __init__(self, config_dict):
        text_config = config_dict.get('text_config', config_dict)
        self.hidden_size = text_config['hidden_size']
        self.num_hidden_layers = text_config['num_hidden_layers']
        self.num_attention_heads = text_config['num_attention_heads']
        self.num_key_value_heads = text_config['num_key_value_heads']
        self.head_dim = text_config.get('head_dim', self.hidden_size // self.num_attention_heads)
        self.hidden_act = text_config['hidden_act']
        self.max_position_embeddings = text_config['max_position_embeddings']
        self.rms_norm_eps = text_config['rms_norm_eps']
        self.vocab_size = text_config['vocab_size']
        self.layer_types = text_config['layer_types']

        # Linear attention params
        self.linear_num_key_heads = text_config['linear_num_key_heads']
        self.linear_num_value_heads = text_config['linear_num_value_heads']
        self.linear_key_head_dim = text_config['linear_key_head_dim']
        self.linear_value_head_dim = text_config['linear_value_head_dim']
        self.linear_conv_kernel_dim = text_config['linear_conv_kernel_dim']

        # MoE params
        self.num_experts = text_config['num_experts']
        self.num_experts_per_tok = text_config['num_experts_per_tok']
        self.moe_intermediate_size = text_config['moe_intermediate_size']
        self.shared_expert_intermediate_size = text_config['shared_expert_intermediate_size']

        # RoPE params
        self.rope_parameters = text_config.get('rope_parameters', {})

        # dtype
        dtype_str = text_config.get('dtype', 'bfloat16')
        dtype_map = {
            'bfloat16': 'torch.bfloat16',
            'float16': 'torch.float16',
            'float32': 'torch.float32',
        }
        import torch
        self.torch_dtype = getattr(torch, dtype_str, torch.bfloat16)


class Qwen3_5MoeConfig:
    """Top-level config for Qwen3.5 MoE model."""

    def __init__(self, config_path):
        # Try to load config.json first, then fall back to known config file names
        config_file = None
        if os.path.isdir(config_path):
            for name in ['config.json', 'qwen3.5-35B-A3B-config']:
                candidate = os.path.join(config_path, name)
                if os.path.isfile(candidate):
                    config_file = candidate
                    break
        elif os.path.isfile(config_path):
            config_file = config_path

        if config_file is None:
            raise FileNotFoundError(f"No config file found in {config_path}")

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        self.model_type = config_dict.get('model_type', 'qwen3_5_moe')
        self.tie_word_embeddings = config_dict.get('tie_word_embeddings', False)
        self.text_config = Qwen3_5MoeTextConfig(config_dict)

        # Expose text_config attributes at top level for compatibility
        self.hidden_size = self.text_config.hidden_size
        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.num_attention_heads = self.text_config.num_attention_heads
        self.num_key_value_heads = self.text_config.num_key_value_heads
        self.head_dim = self.text_config.head_dim
        self.max_position_embeddings = self.text_config.max_position_embeddings
        self.vocab_size = self.text_config.vocab_size
        self.torch_dtype = self.text_config.torch_dtype


def load_hf_config(model_path):
    """Load config, with fallback for unsupported model types."""
    # Check if config has a model_type we need to handle specially
    config_file = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_file):
        # Try known alternative config files
        for name in ['qwen3.5-35B-A3B-config']:
            candidate = os.path.join(model_path, name)
            if os.path.isfile(candidate):
                config_file = candidate
                break

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        model_type = config_dict.get('model_type', '')
        if model_type == 'qwen3_5_moe':
            return Qwen3_5MoeConfig(model_path)
        if model_type == 'qwen3_5':
            return Qwen3_5DenseConfig(model_path)

    # Default: use transformers AutoConfig
    return AutoConfig.from_pretrained(model_path)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    hf_config: object = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # ----- Layer-Batch parallelism (Green-Context dual-stream decode) -----
    enable_layer_batch: bool = False
    # SM count for the FA partition. The remaining SMs go to the LA partition.
    # Driver may round to alignment; values are clamped to [8, total_sms-8].
    # Pipeline LB design: LA-block stream runs 3 LA layers; FA-layer stream runs 1 FA layer.
    # LA is the compute-heavier path → give it more SMs.
    # Defaults: -1 means auto = round(total_sm * 0.4) for fa, total_sm - fa for la.
    # The auto-resolution happens at ModelRunner init (after CUDA device set).
    layer_batch_fa_sm: int = -1
    layer_batch_la_sm: int = -1       # informational; LA uses remainder = total - fa_sm
    # Fraction in (0, 1): nano_batch1 size = ceil(bs * split). 0.5 = even split.
    layer_batch_split: float = 0.5
    # Disable layer-batch when batch size is below this threshold.
    layer_batch_min_bs: int = 2
    # Disable layer-batch when batch size is at or above this threshold.
    # At very large bs the GPU is already saturated and SM-disjoint partitioning
    # actually slows decode down (each half loses BW vs the un-split full kernel).
    # Empirically (Qwen3.5-2B / H20-78SM): LB wins at bs≈4-10, neutral at bs<=4,
    # and hurts at bs>=16; pick a cutoff in [12, 14].
    layer_batch_max_bs: int = -1     # -1 = auto: max_num_seqs (no upper limit)
    # Enable CUDA Graph capture for the LB path. True is now safe.
    #
    # NOTE: previous versions defaulted to False because the LB+graph
    # combination produced ~5-10% garbled output at high concurrency.
    # Three independent bugs caused that:
    #   1. The captured graph runs on `la_stream` (a Green-Context
    #      `ExternalStream`); the runtime input copies happen on the
    #      default stream — there was no cross-stream wait, so the
    #      replay raced against the input writes.  Fix: explicit
    #      `la_stream.wait_stream(default)` before `graph.replay()`.
    #   2. PyTorch's caching allocator did not track ExternalStream
    #      tensor lifetimes, so intermediate tensors crossing
    #      la_stream ↔ fa_stream could be freed/recycled mid-use,
    #      yielding nondeterministic replay (verified: same-input
    #      replay diverged by >20.0 max-abs across trials).  Fix:
    #      explicit `tensor.record_stream(stream)` at every
    #      cross-ExternalStream boundary in `run_layer_batch_decode`.
    #   3. The bs<bucket_bs padding code reused the first real
    #      request's LA slot for padding entries, which collide
    #      across Group-A / Group-B writes.  Fix: pick a slot not in
    #      use by any real request this step.
    # All three are fixed.  Verified 0/64 garble at c=16, 0/96 at c=32.
    layer_batch_use_graph: bool = True

    # ---- Dynamic SM allocation (multiple Green-Context partitions) ----
    # List of (fa_sm, la_sm, max_ctx_threshold).  At each decode step the
    # runner picks the FIRST partition whose `max_ctx_threshold` >= the
    # batch's max(context_lens); if none match, the last partition is used.
    #
    # Empty list ([]) → falls back to single-partition mode using
    # `layer_batch_fa_sm` / `layer_batch_la_sm` above (legacy behaviour).
    #
    # Default 3-bucket policy targets H20-78SM + Qwen3.5-2B:
    #   short ctx (≤4K)    : LA gets large partition (LA's 18 GEMMs dominate)
    #   medium  (≤32K)     : balanced
    #   long ctx (rest)    : FA gets large partition (flash_attn dominates)
    # Default empty: auto-fill at init based on total_sm + max_model_len if user
    # enables layer-batch AND does not pick simple / no-greenctx variants.
    # Auto policy (3-bucket, total_sm-aware):
    #   short ctx ≤ 4K        : LA gets ~70%   (LA's many GEMMs dominate)
    #   medium  ≤ floor(max_model_len/2) : balanced 50/50
    #   long ctx (rest)        : FA gets ~70%  (flash_attn dominates)
    layer_batch_partitions: list = field(default_factory=list)

    # ---- Total-token enable/disable thresholds ----
    # `total_tokens` = sum(context_lens) + bs (tokens being read+written this step).
    # LB only fires when `total_tokens` is in [min, max] AND bs is in [min_bs, max_bs].
    # min: below this the per-step work is too small for split-SM to pay off.
    # max: above this HBM is saturated and partitioning hurts (PD-Mux observation).
    layer_batch_min_total_tokens: int = 256
    layer_batch_max_total_tokens: int = -1   # -1 = auto = max_model_len * max_num_seqs

    # ---- POD-Attention experimental modes ----
    # `no_greenctx`: skip CUDA Green-Context partitioning and instead create
    # two regular `torch.cuda.Stream()` instances that BOTH share all 78 SMs.
    # The grid scheduler will co-locate kernels from both streams onto the
    # same SMs, so a TC-bound GEMM warp and a memory-bound recurrent warp
    # can cohabitate one SM — exactly the POD-Attention idea, but at the
    # stream-level granularity (instead of CTA-level fusion).
    layer_batch_no_greenctx: bool = False

    # Streamlined LayerBatch: exactly 2 cuda streams, no Green-Context, no
    # multi-partition skeleton.  Two nano-batches share all SMs; the GPU grid
    # scheduler co-locates LA-on-stream-A's and FA-on-stream-B's CTAs only when
    # there are spare SMs (purely opportunistic — no hardware partitioning).
    # Mutually exclusive with layer_batch_no_greenctx and layer_batch_partitions
    # (when set, those two are ignored).
    layer_batch_simple: bool = False

    # POD-Attention CTA-fused decode kernel (paged flash-attn + optional GEMM
    # piggy-back).  When True, FA layers' decode flash_attn calls go through
    # the Triton kernel in `nanovllm.layers.pod_kernels` instead of the
    # external flash-attn library.
    pod_attention_decode: bool = False

    # ---- Prefill-LayerBatch (cache-hit-aware dual-stream prefill) ----
    # Motivation: at very long contexts (>100K) with high prefix-cache hit
    # ratio, prefill becomes HBM-bandwidth bound (loading large KV cache).
    # Conversely, low-hit prefill is compute-bound (QKV proj + MLP on many
    # new tokens). Splitting these into two groups and running them on two
    # CUDA streams (under Green Context with asymmetric SM allocation) may
    # let the compute-bound group's GEMMs and the bandwidth-bound group's
    # KV reads overlap on the same SM (analogous to POD-Attention's idea
    # but at the prefill batch level).
    enable_prefill_layer_batch: bool = False
    # Hit-ratio threshold to classify a sequence as "high-hit" vs "low-hit".
    # Default 0.05: any sequence with >5% of its tokens already cached is
    # treated as high-hit; <5% goes to low-hit. The split fires only when
    # both groups are non-empty AND at least one sequence has length >= min_len.
    prefill_lb_hit_threshold: float = 0.05
    # Minimum sequence length (in tokens) of at least one seq in the batch
    # for the split to fire. Below this the prefill is small enough that
    # parallel-stream overhead exceeds the gain.
    prefill_lb_min_len: int = 100_000
    # SM allocation for the Green Context partition used by prefill split.
    # -1 sentinels: low_hit_sm = round(total_sm * 0.7) (compute-bound path
    # gets more SMs); high_hit_sm = total_sm - low_hit_sm.
    prefill_lb_low_hit_sm: int = -1
    prefill_lb_high_hit_sm: int = -1

    # ---- POD-Attention Triton kernel tile configuration ----
    # All -1 sentinel values are auto-resolved by ModelRunner._resolve_pod_config()
    # using head_dim, num_kv_heads, page_size, and max_model_len.
    pod_num_kv_splits: int = -1   # default auto: clamp(max_model_len // 8192, 4, 32) rounded up to power of 2
    pod_block_n:       int = -1   # default auto: 64 (matches Hopper L1 line, decode-tuned)
    pod_block_h:       int = -1   # default auto: max(16, num_q_heads_per_kv) — must be >= H_q/H_kv
    pod_num_warps:     int = 4    # constant: 4 warps balances reg pressure + occupancy
    pod_num_stages:    int = 3    # constant: 3 stages for software-pipelined K/V loads
    pod_gemm_block_m:  int = 16   # GEMM M-tile (decode: M is small, 1..8 typical)
    pod_gemm_block_n:  int = 64   # GEMM N-tile
    pod_gemm_block_k:  int = 32   # GEMM K-tile

    # ---- FlashInfer paged-prefill backend (Ampere-aware) ----
    # Enables FlashInfer's BatchPrefillWithPagedKVCacheWrapper for prefill
    # attention.  Auto-detected off on Ampere (A100, sm_80/86) — flashinfer
    # JIT hazards make it unreliable there.  Users can force it on Ampere
    # via the environment variable NANOVLLM_FORCE_FLASHINFER_ON_AMPERE=1
    # (only recommended if they installed flashinfer-cubin so JIT is off).
    use_flashinfer_prefill: bool = False

    # ---- Chunked prefill ----
    # Splits each seq's prefill work into fixed-size token chunks, running
    # them sequentially through the model to cap peak activation memory at
    # O(chunk_size) instead of O(max_seq_len).  Chunks are aligned to
    # `kvcache_block_size` for cache consistency; the effective chunk size
    # is `ceil(prefill_chunk_size / kvcache_block_size) * kvcache_block_size`.
    # Enable when your prompts approach the max_model_len and your GPU is
    # small (e.g. A100-40GB) — for shorter prompts the extra Python-loop
    # overhead outweighs the memory win.
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int = 2048

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = load_hf_config(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
