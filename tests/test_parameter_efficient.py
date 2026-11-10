import copy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import parameter_efficient as peft


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv_proj = nn.Linear(4, 12, bias=False)
        self.attention.out_proj = nn.Linear(4, 4, bias=False)
        self.ff = nn.Module()
        self.ff.fc_in = nn.Linear(4, 8)
        self.ff.fc_out = nn.Linear(8, 4)

    def forward(self, x):
        x = x + self.attention.out_proj(self.attention.qkv_proj(x).chunk(3, -1)[0])
        return x + self.ff.fc_out(torch.tanh(self.ff.fc_in(x)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 4)
        self.blocks = nn.ModuleList([Block() for _ in range(6)])
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def test_lora_preserves_initial_function_trains_every_layer_and_merges():
    torch.manual_seed(42)
    base = Model()
    model = copy.deepcopy(base)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    x = torch.randn(3, 4)
    report = peft.configure(model, "lora", rank=2, alpha=2)
    assert len(report["selected"]) == 24
    torch.testing.assert_close(model(x), base(x), rtol=0, atol=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()
    for name, p in model.named_parameters():
        if name in before:
            assert torch.equal(p, before[name])
    for block in model.blocks:
        assert block.attention.qkv_proj.lora_B.abs().sum() > 0
        assert block.ff.fc_out.lora_B.abs().sum() > 0
    unmerged = peft.adapter_state(model)
    with peft.merged_state_dict(model):
        merged = model.state_dict()
    assert not any("lora_" in key for key in merged)
    restored = Model()
    restored.load_state_dict(merged, strict=True)
    torch.testing.assert_close(restored(x), model(x), rtol=1e-5, atol=1e-5)
    assert all(torch.equal(value, peft.adapter_state(model)[name]) for name, value in unmerged.items())
    assert any("lora_A" in key for key in model.state_dict())


def test_last_blocks_stop_early_graph_and_preserve_disabled_parameters():
    model = Model()
    model.blocks[-1].ff.fc_out.bias.requires_grad_(False)
    peft.configure(model, "last_blocks", last=2)
    outputs = []
    for block in model.blocks:
        block.register_forward_hook(lambda module, args, out: outputs.append(out.requires_grad))
    model(torch.randn(3, 4)).sum().backward()
    assert outputs == [False, False, False, False, True, True]
    assert model.blocks[0].ff.fc_in.weight.grad is None
    assert model.blocks[-1].ff.fc_in.weight.grad is not None
    assert model.blocks[-1].ff.fc_out.bias.grad is None


@pytest.mark.parametrize("method,kwargs", [("unknown", {}), ("last_blocks", {"last": 0}),
    ("last_blocks", {"last": 7}), ("lora", {"rank": 0}), ("lora", {"alpha": float("nan")})])
def test_invalid_adaptation_rejected(method, kwargs):
    with pytest.raises(ValueError):
        peft.configure(Model(), method, **kwargs)
