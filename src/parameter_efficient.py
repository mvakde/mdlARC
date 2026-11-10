"""Selective block tuning and mergeable LoRA for the native transformer."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F


LORA_TARGETS = ("attention.qkv_proj", "attention.out_proj", "ff.fc_in", "ff.fc_out")


class LoRALinear(nn.Linear):
    def __init__(self, original, rank, alpha):
        # Reuse the frozen weights without constructing another dense matrix.
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(original.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(original.weight.new_zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value):
        return F.linear(value, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(value, self.lora_A), self.lora_B)


def configure(model, method, *, last=4, rank=16, alpha=16):
    if method not in {"last_blocks", "lora"}:
        raise ValueError("Adaptation method must be last_blocks or lora")
    if method == "last_blocks" and (type(last) is not int or not 1 <= last <= len(model.blocks)):
        raise ValueError("last must select at least one existing block")
    if method == "lora" and (type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("LoRA rank and alpha must be positive")
    active = {id(p) for p in model.parameters() if p.requires_grad}
    model.requires_grad_(False)
    selected = []
    if method == "last_blocks":
        for index in range(len(model.blocks) - last, len(model.blocks)):
            # Preserve parameters disabled by the architecture itself.
            block = model.blocks[index]
            for parameter in block.parameters():
                parameter.requires_grad_(id(parameter) in active)
            selected.append(index)
    else:
        for index, block in enumerate(model.blocks):
            for target in LORA_TARGETS:
                parent_name, name = target.rsplit(".", 1)
                parent = block.get_submodule(parent_name)
                original = getattr(parent, name)
                if type(original) is not nn.Linear:
                    raise ValueError(f"Expected an unmodified Linear at blocks.{index}.{target}")
                setattr(parent, name, LoRALinear(original, rank, alpha))
                selected.append(f"blocks.{index}.{target}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"method": method, "selected": selected, "rank": rank if method == "lora" else None,
            "alpha": alpha if method == "lora" else None,
            "trainable_parameters": trainable, "stored_parameters": total,
            "trainable_fraction": trainable / total}


def configure_trainable_blocks(model, first=None, last=None):
    """Freeze every parameter outside the first/last transformer blocks (kaggle/multi_gpu.py's rule).

    Embeddings, final norm, heads and loop scalars freeze too. Frozen middle blocks stay in autograd, so
    earlier blocks still receive gradients through them.
    """
    count = len(model.blocks)
    if any(type(value) is not int or value < 0 or value > count for value in (first, last)):
        raise ValueError("First/last trainable block counts must be integers within model depth.")
    selected = sorted(set(range(first)) | set(range(count - last, count)))
    if not selected:
        raise ValueError("At least one transformer block must be trainable.")
    active = {id(p) for p in model.parameters() if p.requires_grad}
    model.requires_grad_(False)
    for index in selected:
        for parameter in model.blocks[index].parameters():
            parameter.requires_grad_(id(parameter) in active)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"method": "trainable_blocks", "trainable_blocks": selected, "trainable_parameters": trainable,
            "stored_parameters": total, "trainable_fraction": trainable / total}


def configure_finetune(model, args):
    """Apply a training stage's finetune_method / finetune_trainable_blocks; None when it trains everything."""
    method = getattr(args, "finetune_method", None) or "full"
    blocks = getattr(args, "finetune_trainable_blocks", None)
    if method == "lora" and blocks is None:
        return configure(model, "lora", rank=args.lora_rank, alpha=args.lora_alpha)
    if method != "full" or (blocks is not None and len(blocks) != 2):
        raise ValueError("finetune_method must be full or lora; finetune_trainable_blocks is [first, last] without LoRA")
    return None if blocks is None else configure_trainable_blocks(model, *blocks)


def has_lora(model):
    return any(isinstance(layer, LoRALinear) for layer in model.modules())


def merged_model_state(model):
    """state_dict() with every adapter merged into its frozen weight: ordinary checkpoint keys."""
    state = model.state_dict()
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            key = name + "."
            state[key + "weight"] = (layer.weight.detach() +
                layer.scaling * (layer.lora_B.detach() @ layer.lora_A.detach()))
            del state[key + "lora_A"]
            del state[key + "lora_B"]
    return state


def adapter_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
            if name.endswith((".lora_A", ".lora_B"))}


@contextmanager
def merged_state_dict(model):
    """Export standard model keys while preserving live, unmerged adapters."""
    @torch.no_grad()
    def merge(module, state, prefix, metadata):
        for name, layer in module.named_modules():
            if isinstance(layer, LoRALinear):
                key = prefix + name + "."
                state[key + "weight"] = (layer.weight.detach() +
                    layer.scaling * (layer.lora_B.detach() @ layer.lora_A.detach())).cpu()
                del state[key + "lora_A"]
                del state[key + "lora_B"]
    handle = model.register_state_dict_post_hook(merge)
    try:
        yield
    finally:
        handle.remove()
