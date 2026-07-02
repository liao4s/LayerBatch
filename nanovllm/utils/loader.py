import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.utils.logger import init_logger

logger = init_logger(__name__)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    weight_prefix = getattr(model, "weight_prefix", "")
    skip_prefixes = getattr(model, "skip_prefixes", ("mtp.",))
    loaded_count = 0
    skipped_count = 0
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip weights we don't need (e.g., mtp, visual)
                should_skip = False
                for sp in skip_prefixes:
                    if weight_name.startswith(sp):
                        should_skip = True
                        break
                if should_skip:
                    skipped_count += 1
                    continue

                # Strip weight prefix (e.g., "model." for VLM models)
                param_name = weight_name
                if weight_prefix and param_name.startswith(weight_prefix):
                    param_name = param_name[len(weight_prefix):]

                # Check packed modules mapping
                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = param_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        loaded_count += 1
                        break
                else:
                    try:
                        param = model.get_parameter(param_name)
                    except (AttributeError, KeyError):
                        # Parameter not found in model, skip
                        logger.warning("Skipped weight '%s' (no matching param '%s')", weight_name, param_name)
                        skipped_count += 1
                        continue
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
                    loaded_count += 1
    logger.info("Loaded %d weights, skipped %d", loaded_count, skipped_count)
