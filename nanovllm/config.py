import os
import json
from dataclasses import dataclass
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
    # LA is the compute-heavier path → give it more SMs.  Defaults target H20 78 SMs:
    #   FA partition  31 SMs ≈ 40 %      LA partition (remainder)  47 SMs ≈ 60 %
    layer_batch_fa_sm: int = 31
    layer_batch_la_sm: int = 47       # informational; LA uses remainder = total - fa_sm
    # Fraction in (0, 1): nano_batch1 size = ceil(bs * split). 0.5 = even split.
    layer_batch_split: float = 0.5
    # Disable layer-batch when batch size is below this threshold.
    layer_batch_min_bs: int = 2
    # Disable layer-batch when batch size is at or above this threshold.
    # At very large bs the GPU is already saturated and SM-disjoint partitioning
    # actually slows decode down (each half loses BW vs the un-split full kernel).
    # Empirically (Qwen3.5-2B / H20-78SM): LB wins at bs≈4-10, neutral at bs<=4,
    # and hurts at bs>=16; pick a cutoff in [12, 14].
    layer_batch_max_bs: int = 12
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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = load_hf_config(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
