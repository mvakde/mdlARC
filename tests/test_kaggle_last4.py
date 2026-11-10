"""Last-four recipe, answer isolation and real one-epoch DDP checkpoint/decode."""

import ast
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
# Also run this file in an isolated process against the exact shipped model runtime.
RUNTIME = Path(os.environ.get("MDLARC_TEST_RUNTIME", ROOT))
sys.path[:0] = [str(RUNTIME / "src"), str(RUNTIME / "kaggle"), str(ROOT / "kaggle")]
import build
import make_notebook
import multi_gpu as mg
import submission_utils as ku
from checkpoint_profiles import PROFILES
from common import sequence_format_for_attention
from tinytransformer import TinyTransformer, TinyTransformerConfig


def notebook(profile):
    return json.loads((ROOT / "kaggle" / PROFILES[profile]["filename"]).read_text())


@pytest.mark.parametrize("profile", ["600m-last4", "600m-val240-last4"])
def test_last4_notebook_pins_runtime_and_one_epoch_controls(profile, tmp_path):
    import nbformat

    path = ROOT / "kaggle" / PROFILES[profile]["filename"]
    nbformat.validate(nbformat.read(path, as_version=4))
    nb = notebook(profile)
    for item in nb["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
            assert item["execution_count"] is None and not item["outputs"]
    original = ROOT / "kaggle/submit_arc2_600m.ipynb"
    assert hashlib.sha256(original.read_bytes()).hexdigest() == PROFILES[profile]["runtime_source_sha256"]
    sources = make_notebook.frozen_runtime(path)
    pinned = make_notebook.frozen_runtime(original)
    assert {k: v for k, v in sources.items() if k.startswith("src/")} == {
        k: v for k, v in pinned.items() if k.startswith("src/")}
    settings = {}
    for node in ast.parse("".join(nb["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                settings[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert {key: settings[key] for key in ["SMOKE_TEST", "EPOCHS", "NUM_GPUS", "TRAIN_FIRST_LAYERS",
        "TRAIN_LAST_LAYERS", "TRAIN_BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "TRAIN_AUGMENTS",
        "EVAL_AUGMENTS", "NORMUON_LR", "ADAMW_LR", "EVAL_BATCH_SIZE", "PREFILL_TOKEN_BUDGET",
        "INFERENCE_SPECULATIVE"]} == dict(SMOKE_TEST=True, EPOCHS=330, NUM_GPUS=4,
        TRAIN_FIRST_LAYERS=0, TRAIN_LAST_LAYERS=4, TRAIN_BATCH_SIZE=2,
        GRADIENT_ACCUMULATION_STEPS=2, TRAIN_AUGMENTS=80, EVAL_AUGMENTS=80,
        NORMUON_LR=0.0003, ADAMW_LR=0.00028, EVAL_BATCH_SIZE=1,
        PREFILL_TOKEN_BUDGET=4096, INFERENCE_SPECULATIVE=True)
    data = "".join(nb["cells"][7]["source"])
    assert "training_challenges = ku.prepare_challenges(TEST_CHALLENGES, data_path)" in data
    assert 'smoke_task_count=NUM_GPUS, smoke_strategy="largest"' in data
    # Execute the actual request cell in both modes, observing everything sent to workers.
    for smoke in [True, False]:
        folder = tmp_path / str(smoke)
        folder.mkdir()
        def launch(path):
            request = json.loads(path.read_text())
            ft = request["finetune"]
            assert ft["epochs"] == (1 if smoke else 330)
            assert ft["batch_size"] == ft["accumulation"] == 2
            assert ft["max_augments"] == 80
            assert request["eval_augments"] == (2 if smoke else 80)
            assert request["data_path"] != request["decode_data_path"]
            assert request["checkpoint_path"] == "original-pretrained.pt"
            assert request["trainable_blocks"] == {"first": 0, "last": 4}
            for rank in range(4):
                mg.write_json(Path(request["run_dir"]) / f"train-{rank}.json",
                    dict(optimizer_updates=7, epochs=ft["epochs"], epoch_seconds=[1] * ft["epochs"],
                         trainable_blocks=[20, 21, 22, 23], effective_batch_size=16))
        scope = dict(settings, WORKING=folder, SMOKE_TEST=smoke, json=json,
            data_path="all-training.json", decode_data_path="decode-subset.json",
            CHECKPOINT_PATH="original-pretrained.pt", perf_counter=lambda: 0,
            mg=SimpleNamespace(write_json=mg.write_json, launch_training=launch,
                               make_inference_checkpoint=lambda *a: None))
        exec("".join(nb["cells"][9]["source"]), scope)
    final = "".join(nb["cells"][11]["source"])
    assert 'if SMOKE_TEST and repairs:' in final
    if "validation" in PROFILES[profile]:
        assert "find_validation_challenges" in data and "find_test_challenges" not in data
        assert final.index("mg.launch_inference") < final.index("mg.merge_shards") < final.index("solutions_path.read_text")
        assert "score_validation_groups" in final and "READY:" not in final
        assert 'else "submission.json"' not in final
    else:
        assert "find_test_challenges" in data and "find_validation_challenges" not in data
        assert "solutions_path" not in final and "validation_groups" not in final
    rebuilt = make_notebook.make_notebook(profile, tmp_path / "rebuilt.ipynb")
    assert rebuilt.read_bytes() == path.read_bytes()


def test_last4_builder_rejects_model_refresh(tmp_path):
    with pytest.raises(ValueError, match="pinned model runtime"):
        make_notebook.make_notebook("600m-last4", tmp_path / "bad.ipynb", refresh_runtime=True)


def test_five_epoch_timing_notebook_runs_full_workload_and_reports_measured_decode(tmp_path, monkeypatch):
    import nbformat

    profile = "600m-val240-last4-timing5"
    path = ROOT / "kaggle" / PROFILES[profile]["filename"]
    nbformat.validate(nbformat.read(path, as_version=4))
    nb = notebook(profile)
    for item in nb["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
    source = make_notebook.frozen_runtime(path)
    pinned = make_notebook.frozen_runtime(ROOT / "kaggle/submit_arc2_600m.ipynb")
    assert {k: v for k, v in source.items() if k.startswith("src/")} == {
        k: v for k, v in pinned.items() if k.startswith("src/")}
    settings = {}
    for node in ast.parse("".join(nb["cells"][1]["source"])).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                settings[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert settings["SMOKE_TEST"] is False and settings["EPOCHS"] == 5
    assert settings["TRAIN_BATCH_SIZE"] == settings["GRADIENT_ACCUMULATION_STEPS"] == 2
    assert settings["NUM_GPUS"] == 4 and settings["EVAL_BATCH_SIZE"] == 1
    assert settings["TRAIN_AUGMENTS"] == settings["EVAL_AUGMENTS"] == 80
    assert settings["INFERENCE_SPECULATIVE"] is True and settings["PREFILL_TOKEN_BUDGET"] == 4096
    tasks = {f"task{i:03d}": {"train": [{"input": [[1]], "output": [[2]]}] * (4 if i < 6 else 3),
        "test": [{"input": [[3]]}] * (2 if i < 96 else 1)} for i in range(240)}
    data_path = tmp_path / "all.json"
    mg.write_json(data_path, tasks)
    decode_path = tmp_path / "decode.json"
    decoded = ku.prepare_challenges(data_path, decode_path, smoke_test=settings["SMOKE_TEST"],
        smoke_task_count=4, smoke_strategy="largest")
    assert len(decoded) == 240 and sum(len(t["test"]) for t in decoded.values()) == 336
    solutions_path = tmp_path / settings["VALIDATION_IDENTITY"]["solutions"]["filename"]
    mg.write_json(solutions_path, {key: [[[2]]] * len(task["test"]) for key, task in tasks.items()})
    events = []

    def train_request(path):
        request = json.loads(path.read_text())
        assert request["finetune"] == dict(epochs=5, batch_size=2, accumulation=2,
            max_augments=80, num_workers=0, normuon_lr=0.0003, adamw_lr=0.00028)
        assert request["eval_augments"] == 80 and request["inference_speculative"] is True
        assert request["checkpoint_path"] == "original-pretrained.pt"
        assert len(json.loads(Path(request["decode_data_path"]).read_text())) == 240
        for rank in range(4):
            mg.write_json(Path(request["run_dir"]) / f"train-{rank}.json", dict(
                optimizer_updates=10, epochs=5, epoch_seconds=[10] * 5,
                trainable_blocks=[20, 21, 22, 23], effective_batch_size=16))

    def decode_request(path):
        events.append("decode")

    def merge(request, challenges, output):
        events.append("merge")
        predictions = {key: [{"attempt_1": [[2]], "attempt_2": [[2]]}] * len(task["test"])
                       for key, task in challenges.items()}
        mg.write_json(output, predictions)
        return [], [{"rank": i} for i in range(4)]

    def verify_answers(path, identity):
        assert events == ["decode", "merge"]
        assert Path(path) == solutions_path
        events.append("score")

    worker = SimpleNamespace(write_json=mg.write_json, launch_training=train_request,
        make_inference_checkpoint=lambda *args: None, launch_inference=decode_request, merge_shards=merge)
    scope = dict(settings, WORKING=tmp_path, json=json, data_path=data_path, decode_data_path=decode_path,
        CHECKPOINT_PATH="original-pretrained.pt", perf_counter=lambda: 0, mg=worker)
    exec("".join(nb["cells"][9]["source"]), scope)
    scope.update(training_challenges=tasks, challenges=decoded, torch=torch,
        checkpoint_hash="checkpoint-sha", RUNTIME_SHA256="runtime-sha", TEST_CHALLENGES=data_path,
        validation_groups={"original40": list(tasks)[:40], "synthetic200": list(tasks)[40:]},
        ku=SimpleNamespace(verify_dataset_file=verify_answers, score_validation_groups=ku.score_validation_groups))
    ticks = iter([100, 340])
    scope["perf_counter"] = lambda: next(ticks)
    exec("".join(nb["cells"][11]["source"]), scope)
    assert events == ["decode", "merge", "score"]
    report = json.loads((tmp_path / "timing240_benchmark.json").read_text())
    assert report["training_epochs"] == 5 and report["decode_tasks"] == 240
    assert report["augmented_query_cases"] == 26880 and report["decode_wall_seconds"] == 240
    assert report["aggregate_augmented_cases_per_second"] == 112
    assert not (tmp_path / "submission.json").exists()
    rebuilt = make_notebook.make_notebook(profile, tmp_path / "rebuilt-timing.ipynb")
    assert rebuilt.read_bytes() == path.read_bytes()


def test_last4_freeze_preserves_disabled_parameters_and_optimizer_excludes_all_others():
    import train

    model = TinyTransformer(TinyTransformerConfig(d_model=24, n_heads=4, d_ff=48,
        n_layers=24, num_examples=4, num_task_embedding_latents=5,
        task_embeddings=False, n_predict=4))
    model.blocks[22].ff.fc_in.weight.requires_grad_(False)
    originally_active = {id(p) for p in model.parameters() if p.requires_grad}
    result = mg.configure_trainable_blocks(model, first=0, last=4)
    assert result["trainable_blocks"] == [20, 21, 22, 23]
    expected = {id(p) for i in range(20, 24) for p in model.blocks[i].parameters()} & originally_active
    assert {id(p) for p in model.parameters() if p.requires_grad} == expected
    muon, adam = train._build_param_groups(model, 0.1, 0.01, 0.01, 0.01)
    assert {id(p) for group in [*muon, *adam] for p in group["params"]} == expected
    assert not model.blocks[22].ff.fc_in.weight.requires_grad


def test_group_scores_keep_original_and_synthetic_separate():
    tasks = {key: {"train": [], "test": [{"input": [[1]]}]} for key in ["original", "synthetic"]}
    guesses = {key: [{"attempt_1": [[1]], "attempt_2": [[2]]}] for key in tasks}
    answers = {"original": [[[2]]], "synthetic": [[[3]]]}
    groups = {"original40": ["original"], "synthetic200": ["synthetic"]}
    scores = ku.score_validation_groups(guesses, tasks, answers, groups)
    assert scores["original40"]["top2"]["percent"] == 100
    assert scores["synthetic200"]["top2"]["percent"] == 0
    assert scores["combined_timing_workload"]["top2"]["percent"] == 50
    partial = ku.score_validation_groups({"original": guesses["original"]},
        {"original": tasks["original"]}, answers, groups)
    assert partial["synthetic200"]["tasks"] == 0
    with pytest.raises(ValueError, match="disjoint"):
        ku.score_validation_groups(guesses, tasks, answers, dict(groups, extra=["original"]))


def _last4_cpu_rank(rank, folder):
    import train
    folder = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        build.resolve_device = lambda value: torch.device("cpu")
        train._build_optimizer = lambda args, model, *rest: torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001)
        mg.train_worker(json.loads((folder / "request.json").read_text()), device_override="cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_real_one_epoch_last4_ddp_updates_reload_speculative_decode(tmp_path, monkeypatch):
    import evaluate

    torch.set_num_threads(1)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    monkeypatch.setattr(evaluate, "DEFAULT_MAX_NEW_TOKENS", 8)
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)
    speculative_calls = []
    native_speculative = evaluate._batched_speculative_decode

    def checked_speculative(model, *args, **kwargs):
        assert model.config.n_predict == 4
        speculative_calls.append(True)
        return native_speculative(model, *args, **kwargs)

    monkeypatch.setattr(evaluate, "_batched_speculative_decode", checked_speculative)
    torch.manual_seed(42)
    model = TinyTransformer(TinyTransformerConfig(d_model=24, n_heads=4, d_ff=48,
        n_layers=24, num_examples=4, num_task_embedding_latents=5, task_embeddings=False,
        attention_pattern="pair_group", split_target_demo_attention=True,
        attention_dropout=0.0, dropout=0.0, embedding_dropout=0.0, n_predict=4))
    checkpoint = dict(config=asdict(model.config), model_state=model.state_dict(),
        sequence_format=sequence_format_for_attention("pair_group"),
        augmentation_sampling_mode="context_palette", epoch=330,
        task_ids=[f"old/{i}" for i in range(4)], embedding_task_ids=[f"old/{i}" for i in range(4)])
    source = tmp_path / "original.pt"
    torch.save(checkpoint, source)
    tasks = {f"task{i}": {"train": [
        {"input": [[i]], "output": [[i + 1]]}, {"input": [[i + 1]], "output": [[i + 2]]},
        {"input": [[i + 2]], "output": [[i + 3]]}], "test": [{"input": [[i]]}]} for i in range(4)}
    data = tmp_path / "all.json"
    mg.write_json(data, tasks)
    decode = tmp_path / "decode.json"
    mg.write_json(decode, {key: tasks[key] for key in ["task0", "task1"]})
    folder = tmp_path / "run"
    request = dict(working=str(tmp_path), run_dir=str(folder), num_gpus=2,
        checkpoint_path=str(source), checkpoint_identity=dict(epoch=330,
            augmentation_sampling_mode="context_palette", config={"n_layers": 24, "n_predict": 4}),
        data_path=str(data), decode_data_path=str(decode), save_path=str(folder / "adapted.pt"),
        inference_checkpoint=str(folder / "inference.pt"), attention_backend="native", enable_compile=False,
        smoke_test=True, max_finetune_hours=8.0, eval_augments=2, eval_batch_size=1,
        prefill_token_budget=4096, inference_speculative=True, trainable_blocks={"first": 0, "last": 4},
        defaults=dict(make_notebook.DEFAULTS, augmentation_sampling_mode="context_palette",
            training_demo_schedule="all", inference_demo_policy="leave_one_out"),
        finetune=dict(epochs=1, batch_size=2, accumulation=2, max_augments=2,
            num_workers=0, normuon_lr=0.0003, adamw_lr=0.00028))
    mg.write_json(tmp_path / "request.json", request)
    mp.spawn(_last4_cpu_rank, args=(str(tmp_path),), nprocs=2, join=True)
    saved = torch.load(request["save_path"], map_location="cpu", weights_only=False)
    assert saved["epoch"] == 1 and saved["distributed_world_size"] == 2
    changed = set()
    for name, before in checkpoint["model_state"].items():
        if name == "task_factor_embedding.weight":
            continue  # Runtime task rows are initialized fresh, then frozen.
        if not torch.equal(before, saved["model_state"][name]):
            assert name.startswith(tuple(f"blocks.{i}." for i in range(20, 24))), name
            changed.add(int(name.split(".")[1]))
    assert changed == {20, 21, 22, 23}
    mg.make_inference_checkpoint(request["save_path"], request["inference_checkpoint"], 1)
    for rank in range(2):
        report = json.loads((folder / f"train-{rank}.json").read_text())
        assert report["optimizer_updates"] > 0 and len(report["epoch_seconds"]) == 1
        assert report["effective_batch_size"] == 8  # Two CPU ranks, same 2/2 per rank.
        mg.decode_worker(request, rank, 2, device_override="cpu")
        report = json.loads((folder / f"decode-{rank}.json").read_text())
        assert report["inference_speculative"] is True and report["batch_size"] == 1
        assert report["prefill_token_budget"] == 4096
    subset = json.loads(decode.read_text())
    assert len(speculative_calls) >= 2
    mg.merge_shards(request, subset, tmp_path / "smoke.json")
    ku.validate_submission(json.loads((tmp_path / "smoke.json").read_text()), subset)
