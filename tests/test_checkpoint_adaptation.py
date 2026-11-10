import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import checkpoint_adaptation as adaptation
import launch_run


def config():
    return {"method": "lora", "inputs": {name: {"path": name + ".json", "size": 1,
            "sha256": "a" * 64} for name in ("runtime", "challenges", "solutions")}}


def test_adaptation_routes_through_standard_plan(monkeypatch):
    spec = {"run_name": "adaptation", "checkpoint": "input.pt", "enable_finetune": False,
            "skip_data_build": True, "skip_evals": True, "checkpoint_adaptation": config()}
    plan = launch_run.build_plan(spec, Path("spec.json"))
    assert launch_run.preview_plan(plan)["checkpoint_adaptation"] == config()
    assert launch_run.preview_plan(plan)["pretrain"] is None
    seen = []
    monkeypatch.setattr(adaptation, "run_adaptation", lambda p: seen.append(p))
    launch_run.execute_plan(plan)
    assert seen == [plan]


@pytest.mark.parametrize("mutation", [{"rank": 0}, {"epochs": True}, {"method": "full"},
    {"lora_lr": float("inf")}, {"checkpoint_epochs": [91]}, {"enable_compile": "yes"}])
def test_invalid_config_rejected(mutation):
    with pytest.raises(ValueError):
        adaptation.validate_config({**config(), **mutation})


def test_decode_only_reuses_trained_checkpoint_without_training(monkeypatch, tmp_path):
    payloads = {"runtime": {}, "challenges": {"a": {"train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[2]]}]}}, "solutions": {"a": [[[2]]]},
        "training_stats": {"seconds": 123, "trainable_parameters": 2}}
    inputs = {}
    for name, value in payloads.items():
        raw = json.dumps(value).encode()
        path = tmp_path / (name + ".json")
        path.write_bytes(raw)
        inputs[name] = {"path": path.name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    directory = tmp_path / "runs/decode"
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"already trained")
    def unpack(path, destination):
        target = destination / "kaggle/submission_utils.py"
        target.parent.mkdir(parents=True)
        target.write_bytes((ROOT / "kaggle/submission_utils.py").read_bytes())
        return {"SAVED_DEFAULTS": {}, "RUNTIME_SHA256": "a" * 64,
                "CHECKPOINT_IDENTITY": {"epoch": 330, "size": 999, "sha256": "a" * 64}}
    monkeypatch.setattr(adaptation, "unpack_runtime", unpack)
    monkeypatch.setattr(launch_run, "REPO_ROOT", tmp_path)
    phases = []
    def execute(command, **kwargs):
        phase = command[command.index("--phase") + 1]
        phases.append(phase)
        request = json.loads(Path(command[-1]).read_text())
        assert request["save_path"] == str(checkpoint)
        assert request["checkpoint_identity"]["epoch"] == 90
        if phase == "probe":
            adaptation.write_json(directory / "decoding_probe.json",
                {"selected_batch_size": 16, "selected_speculative": False})
        else:
            assert phase == "decode"
            assert request["config"]["eval_batch_size"] == 16
            assert request["config"]["inference_speculative"] is False
            adaptation.write_json(directory / "predictions.json", {"a": [{"attempt_1": [[2]], "attempt_2": [[0]]}]})
            adaptation.write_json(directory / "decoding_stats.json", {"seconds": 2})
    monkeypatch.setattr(adaptation.subprocess, "run", execute)
    adaptation.run_adaptation({"spec_path": tmp_path / "spec.json", "run_dir": directory,
        "pretrain": {"checkpoint_path": checkpoint}, "drive_checkpoint": {"size": 15, "sha256": "b" * 64},
        "checkpoint_adaptation": {"method": "lora", "inputs": inputs, "decode_only": True,
            "input_epoch": 90, "decode_probe": True, "eval_batch_size": 32, "inference_speculative": True}})
    assert phases == ["probe", "decode"]
    info = json.loads((directory / "validation_run_info.json").read_text())
    assert info["training"]["seconds"] == 123 and info["scores"]["top2"]["percent"] == 100


def test_real_preserved_runtime_lora_training_exports_standard_checkpoint(tmp_path):
    # A separate interpreter prevents the current checkout's imports from
    # substituting for the immutable submission implementation.
    script = r'''
import gc, json, os, sys
from dataclasses import asdict
from pathlib import Path
root, directory = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "src"))
import checkpoint_adaptation as ad
runtime = directory / "runtime"
pinned = ad.unpack_runtime(root / "kaggle/submit_arc2_600m.ipynb", runtime)
sys.path[:0] = [str(runtime / "src"), str(runtime / "kaggle")]
sys.modules["flash_attn"] = None
sys.modules["flash_attn_interface"] = None
import torch
from tinytransformer import TinyTransformer, TinyTransformerConfig
from common import sequence_format_for_attention
import build
build.resolve_device = lambda value: torch.device("cpu")
torch.set_num_threads(1)
model = TinyTransformer(TinyTransformerConfig(d_model=16, n_heads=4, d_ff=32,
    n_layers=4, num_examples=2, num_task_embedding_latents=4, task_embeddings=False,
    attention_pattern="pair_group", split_target_demo_attention=True, n_predict=2,
    dropout=0, embedding_dropout=0, attention_dropout=0))
source = {"model_state": model.state_dict(), "config": asdict(model.config),
    "sequence_format": sequence_format_for_attention("pair_group"),
    "augmentation_sampling_mode": "context_palette", "task_ids": ["a", "b"],
    "embedding_task_ids": ["a", "b"], "epoch": 330}
checkpoint_path = directory / "input.pt"
torch.save(source, checkpoint_path)
data = {name: {"train": [{"input": [[1, 0]], "output": [[0, 1]]},
    {"input": [[2, 0]], "output": [[0, 2]]}], "test": [{"input": [[3, 0]]}]} for name in ["a", "b"]}
data_path = directory / "data.json"
data_path.write_text(json.dumps(data))
save = directory / "adaptation/tiny.pt"
request = {"config": {"method": "lora", "epochs": 2, "batch_size": 1,
    "accumulation": 2, "rank": 2, "alpha": 2, "train_augments": 2,
    "enable_compile": False, "attention_backend": "native", "checkpoint_epochs": [1]},
    "defaults": pinned["SAVED_DEFAULTS"], "checkpoint_identity": {"epoch": 330,
    "augmentation_sampling_mode": "context_palette", "config": {"d_model": 16}},
    "source_checkpoint": None, "checkpoint_path": str(checkpoint_path), "runtime": str(runtime),
    "run_dir": str(directory), "data_path": str(data_path), "save_path": str(save),
    "runtime_sha256": pinned["RUNTIME_SHA256"]}
del model
gc.collect()
ad.train_worker(request)
trained = torch.load(save, map_location="cpu", weights_only=False)
assert trained["epoch"] == 2
assert "optimizer_state" not in trained
assert not any("lora_" in key for key in trained["model_state"])
assert torch.equal(trained["model_state"]["token_embedding.weight"], source["model_state"]["token_embedding.weight"])
assert not torch.equal(trained["model_state"]["blocks.0.attention.qkv_proj.weight"], source["model_state"]["blocks.0.attention.qkv_proj.weight"])
config = dict(trained["config"])
config.pop("architecture_version")
restored = TinyTransformer(TinyTransformerConfig(**config))
restored.load_state_dict(trained["model_state"], strict=True)
adapters = torch.load(save.with_suffix(".adapters.pt"), map_location="cpu", weights_only=False)
assert adapters["epoch"] == 2 and adapters["optimizer"]["state"]
assert len(adapters["adapter_state"]) == 32
assert list(save.parent.glob("tiny.epoch*.adapters.pt"))
'''
    result = subprocess.run([sys.executable, "-c", script, str(ROOT), str(tmp_path)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    stats = json.loads((tmp_path / "training_stats.json").read_text())
    assert stats["optimizer"] == "adamw" and len(stats["epoch_seconds"]) == 2
    assert stats["optimizer_updates"] == 4
    progress = json.loads((tmp_path / "training_progress.json").read_text())
    assert progress["optimizer_updates"] == 4 and progress["microbatch_steps"] == 8
