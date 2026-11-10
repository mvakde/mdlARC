"""Checkpoint transplant, target isolation, and submission-format integration."""

from dataclasses import asdict
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "kaggle"))

import build
import make_notebook
from checkpoint_profiles import PROFILES
import submission_utils as ku
import train
from common import SEQUENCE_FORMAT
from tinytransformer import TinyTransformer, TinyTransformerConfig


@pytest.fixture
def source_checkpoint():
    torch.manual_seed(42)
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2,
        num_examples=7, num_task_embedding_latents=5,
        dropout=0.0, attention_dropout=0.0, embedding_dropout=0.0,
        n_predict=4,
    ))
    return {
        "sequence_format": SEQUENCE_FORMAT,
        "augmentation_sampling_mode": "single_grid_prefix",
        "model_state": model.state_dict(), "config": asdict(model.config),
        "task_ids": [f"public/{i}" for i in range(7)],
        "embedding_task_ids": [f"public/{i}" for i in range(7)],
        "optimizer_state": {"old": "must not resume"}, "epoch": 25,
        "global_step": 999, "scheduler_state": {"last_epoch": 25},
        "task_variations": {"pretraining": True},
        "rearc_data_path": "/missing/old/pretraining/file.json",
    }


@pytest.fixture
def challenge_path(tmp_path):
    tasks = {
        "hidden_a": {
            "train": [{"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}],
            "test": [{"input": [[5]], "output": [[6]]}, {"input": [[7, 8]]}],
        },
        "hidden_b": {
            "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
            "test": [{"input": [[3, 4]]}],
        },
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(tasks))
    return path


def config(tmp_path, data_path):
    cfg = ku.finetune_config(
        make_notebook.DEFAULTS, data_path=data_path, save_path=tmp_path / "finetuned.pt",
        epochs=1, batch_size=1, accumulation=1, max_augments=2, num_workers=0,
        normuon_lr=0.0003, adamw_lr=0.003,
    )
    cfg.device = "cpu"
    return cfg


def test_runtime_tasks_reset_factors_and_preserve_every_shared_weight(source_checkpoint, challenge_path, tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda value: torch.device("cpu"))
    adapted = ku.adapt_checkpoint(source_checkpoint)
    cfg = config(tmp_path, challenge_path)
    model, dataset, loader, device, _ = build.build_model_and_data(cfg, checkpoint=adapted)
    assert dataset.task_ids == ["hidden_a", "hidden_b"]
    assert dataset.embedding_task_ids == ["hidden_a", "hidden_b"]
    assert model.task_factor_embedding.weight.shape == (2, 5)
    assert model.config.num_task_embedding_latents == 5  # More latents than tasks.
    assert model.config.n_predict == 4
    assert all(p.requires_grad for p in model.parameters())
    for name, value in model.state_dict().items():
        if name != "task_factor_embedding.weight":
            assert torch.equal(value, source_checkpoint["model_state"][name]), name
    assert source_checkpoint["model_state"]["task_factor_embedding.weight"].shape == (7, 5)
    assert not set(adapted).intersection({"optimizer_state", "scheduler_state", "epoch", "task_variations", "rearc_data_path"})
    assert dataset.rearc_source_path is None
    assert all(e.split == "train" for e in dataset.examples)
    rng = torch.get_rng_state().clone()
    ku.check_backward(model, loader, device)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in model.parameters())


def test_test_outputs_removed_and_smoke_subset_explicit(challenge_path, tmp_path):
    original = challenge_path.read_bytes()
    clean = ku.prepare_challenges(challenge_path, tmp_path / "runtime.json")
    assert clean["hidden_a"]["train"][0]["output"] == [[2]]
    assert clean["hidden_a"]["test"][0] == {"input": [[5]]}
    assert challenge_path.read_bytes() == original
    tasks = json.loads(challenge_path.read_text())
    tasks["hidden_c"] = tasks["hidden_b"]
    challenge_path.write_text(json.dumps(tasks))
    smoke = ku.prepare_challenges(challenge_path, tmp_path / "smoke.json", smoke_test=True)
    assert set(smoke) == {"hidden_a", "hidden_b"}


def test_checkpoint_format_and_row_corruption_rejected(source_checkpoint):
    bad = dict(source_checkpoint, sequence_format="old_whole_task_format")
    with pytest.raises(ValueError, match="sequence format"):
        ku.adapt_checkpoint(bad)
    bad = dict(source_checkpoint, embedding_task_ids=["one"])
    with pytest.raises(ValueError, match="rows"):
        ku.adapt_checkpoint(bad)


def test_profile_verifies_bytes_and_metadata_before_checkpoint_load(source_checkpoint, tmp_path):
    path = tmp_path / "pretrain__tiny.pt"
    path.write_bytes(b"trusted checkpoint bytes")
    identity = dict(PROFILES["public-v3-e25"], size=path.stat().st_size,
                    sha256=ku.sha256_file(path), config={"d_model": 24})
    assert ku.verify_checkpoint_file(path, identity) == identity["sha256"]
    ku.verify_checkpoint_metadata(source_checkpoint, identity)
    with pytest.raises(ValueError, match="epoch 330"):
        ku.verify_checkpoint_metadata(source_checkpoint, dict(identity, epoch=330))
    with pytest.raises(ValueError, match="augmentation"):
        ku.verify_checkpoint_metadata(source_checkpoint, dict(identity, augmentation_sampling_mode="context_palette"))
    with pytest.raises(ValueError, match="d_model"):
        ku.verify_checkpoint_metadata(source_checkpoint, dict(identity, config={"d_model": 1536}))
    path.write_bytes(b"x" * identity["size"])
    with pytest.raises(ValueError, match="SHA-256"):
        ku.verify_checkpoint_file(path, identity)


def test_group_checkpoint_transplant_preserves_weights_and_excludes_query_targets(challenge_path, tmp_path, monkeypatch):
    from common import sequence_format_for_attention

    monkeypatch.setattr(build, "resolve_device", lambda value: torch.device("cpu"))
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=2,
        num_examples=7, num_task_embedding_latents=5, task_embeddings=False,
        attention_pattern="pair_group", split_target_demo_attention=True,
        attention_dropout=0.0, n_predict=4,
    ))
    checkpoint = {
        "sequence_format": sequence_format_for_attention("pair_group"),
        "augmentation_sampling_mode": "context_palette", "epoch": 330,
        "model_state": model.state_dict(), "config": asdict(model.config),
        "task_ids": [f"old/{i}" for i in range(7)],
        "embedding_task_ids": [f"old/{i}" for i in range(7)],
    }
    identity = dict(PROFILES["600m-e330"], config={"attention_pattern": "pair_group", "task_embeddings": False})
    ku.verify_checkpoint_metadata(checkpoint, identity)
    defaults = dict(make_notebook.DEFAULTS, augmentation_sampling_mode="context_palette",
                    training_demo_schedule="all", inference_demo_policy="leave_one_out")
    data = tmp_path / "runtime.json"
    tasks = ku.prepare_challenges(challenge_path, data)
    cfg = ku.finetune_config(defaults, data_path=data, save_path=tmp_path / "ft.pt",
                             epochs=90, batch_size=1, accumulation=1, max_augments=80,
                             num_workers=0, normuon_lr=0.0016, adamw_lr=0.00028)
    transplanted, dataset, loader, device, _ = build.build_model_and_data(cfg, checkpoint=ku.adapt_checkpoint(checkpoint))
    assert transplanted.config.attention_pattern == "pair_group"
    assert transplanted.config.split_target_demo_attention
    assert not transplanted.config.task_embeddings
    assert dataset.augmentation_sampling_mode == "context_palette"
    assert all(example.split == "train" for example in dataset.examples)
    assert len(dataset.examples) == sum(len(task["train"]) for task in tasks.values())
    for name, value in transplanted.state_dict().items():
        if name != "task_factor_embedding.weight":
            assert torch.equal(value, checkpoint["model_state"][name]), name
    ku.check_backward(transplanted, loader, device)


def test_discovery_only_accepts_competition_test_and_unambiguous_checkpoint(tmp_path):
    mount = tmp_path / "competitions" / ku.COMPETITION
    mount.mkdir(parents=True)
    test = mount / "arc-agi_test_challenges.json"
    test.write_text("{}")
    evaluation = mount / "arc-agi_evaluation_challenges.json"
    evaluation.write_text("{}")
    assert ku.find_test_challenges(tmp_path) == test
    with pytest.raises(ValueError, match="test challenges"):
        ku.find_test_challenges(tmp_path, evaluation)
    unrelated = tmp_path / "other" / test.name
    unrelated.parent.mkdir()
    unrelated.write_text("{}")
    assert ku.find_test_challenges(tmp_path) == test
    checkpoint = tmp_path / "pretrain__tiny.pt"
    checkpoint.write_bytes(b"test")
    assert ku.find_checkpoint(tmp_path) == checkpoint
    (tmp_path / "tiny.pt").write_bytes(b"another")
    with pytest.raises(ValueError, match="2 candidates"):
        ku.find_checkpoint(tmp_path)
    assert ku.sha256_file(checkpoint) == hashlib.sha256(b"test").hexdigest()


@pytest.mark.parametrize("grid", [[], [[]], [[True]], [[1.0]], [[10]], [[-1]], [[1], [1, 2]], [[0] * 31]])
def test_invalid_grids_rejected(grid):
    assert not ku.valid_grid(grid)


def test_submission_coverage_order_and_malformed_repair(challenge_path, tmp_path):
    tasks = json.loads(challenge_path.read_text())
    submission = {
        "hidden_a": [
            {"attempt_1": [], "attempt_2": [[2]]},
            {"attempt_1": [[]], "attempt_2": [[10]]},
        ],
        "hidden_b": [{"attempt_1": [[3]], "attempt_2": [[4]]}],
    }
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(submission))
    final = tmp_path / "submission.json"
    repairs = ku.finalize_submission(raw, final, tasks)
    assert len(repairs) == 3
    saved = json.loads(final.read_text())
    assert saved["hidden_a"][0] == {"attempt_1": [[2]], "attempt_2": [[2]]}
    assert saved["hidden_a"][1] == {"attempt_1": [[7, 8]], "attempt_2": [[7, 8]]}
    ku.validate_submission(saved, tasks)
    del submission["hidden_b"]
    raw.write_text(json.dumps(submission))
    with pytest.raises(ValueError, match="exactly the runtime tasks"):
        ku.finalize_submission(raw, final, tasks)
    assert json.loads(final.read_text()) == saved  # Existing complete output not replaced on failure.
    saved["hidden_a"].pop()
    with pytest.raises(ValueError, match="number of test predictions"):
        ku.validate_submission(saved, tasks)


@pytest.mark.parametrize("group_context", [False, True])
def test_cpu_training_save_decode_submission_integration(source_checkpoint, challenge_path, tmp_path, monkeypatch, group_context):
    # CPU checks the real data/model/checkpoint/decode pipeline. The production
    # NorMuon optimizer and CUDA compiler are exercised by the Kaggle smoke run.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    import evaluate
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)
    monkeypatch.setattr(evaluate, "DEFAULT_MAX_NEW_TOKENS", 16)
    monkeypatch.setattr(train, "_build_optimizer", lambda args, model, *rest:
                        torch.optim.AdamW(model.parameters(), lr=0.001))
    data = tmp_path / "runtime.json"
    tasks = ku.prepare_challenges(challenge_path, data)
    cfg = config(tmp_path, data)
    if group_context:
        from common import sequence_format_for_attention
        group_config = dict(source_checkpoint["config"], attention_pattern="pair_group",
                            split_target_demo_attention=True, task_embeddings=False)
        group_config.pop("architecture_version")
        group_model = TinyTransformer(TinyTransformerConfig(**group_config))
        source_checkpoint = dict(source_checkpoint, config=asdict(group_model.config),
                                 model_state=group_model.state_dict(),
                                 sequence_format=sequence_format_for_attention("pair_group"),
                                 augmentation_sampling_mode="context_palette")
        cfg.augmentation_sampling_mode = "context_palette"
        cfg.training_demo_schedule = "all"
        cfg.inference_demo_policy = "leave_one_out"
    model, dataset, loader, device, _ = build.build_model_and_data(cfg, checkpoint=ku.adapt_checkpoint(source_checkpoint))
    before = model.blocks[0].attention.qkv_proj.weight.detach().clone()
    train.train_model(cfg, model, loader, dataset, device, data)
    assert not torch.equal(model.blocks[0].attention.qkv_proj.weight, before)
    saved = build.load_checkpoint(cfg.save_path)
    assert saved["epoch"] == 1
    assert saved["config"]["num_task_embedding_latents"] == 5
    assert saved["embedding_task_ids"] == ["hidden_a", "hidden_b"]
    cfg.checkpoint_mode = None
    cfg.reset_training_state = False
    _, _, raw = evaluate.run_evaluation(
        cfg, run_name="decode", max_augments=2, data_path=data,
        checkpoint_path=cfg.save_path, batch_size=2, splits=["test"],
        timing_path=tmp_path / "timing.txt",
    )
    final = tmp_path / "submission.json"
    ku.finalize_submission(raw, final, tasks)
    ku.validate_submission(json.loads(final.read_text()), tasks)


def _assert_runtime_dependencies_packaged(sources):
    local_modules = {path.stem: path.relative_to(ROOT).as_posix()
                     for path in (ROOT / "src").glob("*.py")}
    for name, source in sources.items():
        for node in ast.walk(ast.parse(source)):
            imports = []
            if isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module.split(".")[0]]
            elif isinstance(node, ast.Import):
                imports = [alias.name.split(".")[0] for alias in node.names]
            for module in imports:
                if module in local_modules:
                    assert local_modules[module] in sources, (name, module)


def test_notebook_builder_packages_runtime_dependencies():
    sources = {name: (ROOT / name).read_text() for name in make_notebook.RUNTIME_FILES}
    _assert_runtime_dependencies_packaged(sources)


@pytest.mark.parametrize("filename", ["submit_arc2.ipynb", "submit_arc2_600m.ipynb", "validate_arc2_600m_val40.ipynb"])
def test_notebook_cells_and_embedded_runtime_are_valid(filename):
    import nbformat

    path = ROOT / "kaggle" / filename
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    payload, expected_hash = None, None
    for item in notebook.cells:
        if item.cell_type != "code":
            continue
        tree = ast.parse(item.source)
        assert item.execution_count is None and not item.outputs
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                if statement.targets[0].id == "RUNTIME_B64":
                    payload = ast.literal_eval(statement.value)
                if statement.targets[0].id == "RUNTIME_SHA256":
                    expected_hash = ast.literal_eval(statement.value)
    raw = base64.b64decode(payload)
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        # The frozen notebook may predate new optional runtime modules. Its
        # own import closure must still be complete, and every file allowlisted.
        names = set(archive.namelist())
        assert names <= set(make_notebook.RUNTIME_FILES)
        assert {"src/build.py", "src/tinytransformer.py", "kaggle/submission_utils.py"} <= names
        _assert_runtime_dependencies_packaged({name: archive.read(name).decode("utf-8") for name in names})
        for name in archive.namelist():
            ast.parse(archive.read(name).decode("utf-8"))
        # Runtime is an intentionally frozen snapshot; unrelated later source
        # edits need not change a previously prepared submission notebook.
        assert "config.num_task_embedding_latents = min(" not in archive.read("src/build.py").decode()


def test_600m_notebook_settings_and_wheel_restore(tmp_path):
    notebook = json.loads((ROOT / "kaggle/submit_arc2_600m.ipynb").read_text())
    settings = ast.parse("".join(notebook["cells"][1]["source"]))
    values = {}
    for node in settings.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    assert values["CHECKPOINT_IDENTITY"]["epoch"] == 330
    assert values["CHECKPOINT_IDENTITY"]["sha256"] == PROFILES["600m-e330"]["sha256"]
    assert values["EPOCHS"] == 90
    assert values["NUM_GPUS"] == 4
    assert values["TRAIN_BATCH_SIZE"] == 1
    assert values["GRADIENT_ACCUMULATION_STEPS"] == 10
    assert values["NUM_GPUS"] * values["TRAIN_BATCH_SIZE"] * values["GRADIENT_ACCUMULATION_STEPS"] == 40
    assert values["SMOKE_TEST"] is True
    assert values["TRAIN_LOG_MODE"] == "step"
    assert values["PROGRESS_INTERVAL_SECONDS"] == 30
    assert values["EVAL_BATCH_SIZE"] == 1
    assert values["TRAIN_FIRST_LAYERS"] == values["TRAIN_LAST_LAYERS"] == 2
    assert values["NORMUON_LR"] == 0.0016 and values["ADAMW_LR"] == 0.00028
    assert values["TEST_CHALLENGES"] is None
    environment = ast.parse("".join(notebook["cells"][3]["source"]))
    function = next(node for node in environment.body if isinstance(node, ast.FunctionDef)
                    and node.name == "restore_wheel_filename")
    import shutil
    namespace = dict(Path=Path, json=json, hashlib=hashlib, shutil=shutil, WORKING=tmp_path / "working")
    namespace["WORKING"].mkdir()
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<wheel helper>", "exec"), namespace)
    source = tmp_path / "sanitized.whl"
    source.write_bytes(b"wheel fixture")
    name = "flash_attn-2.8.3.post1+cu128-cp313-cp313-linux_x86_64.whl"
    manifest = dict(wheel=name, wheel_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (tmp_path / "wheel-manifest.json").write_text(json.dumps(manifest))
    restored = namespace["restore_wheel_filename"](source)
    assert restored.name == name and restored.read_bytes() == source.read_bytes()
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        namespace["restore_wheel_filename"](source)


def test_freeze_update_preserves_packaged_model_source(tmp_path):
    source = ROOT / "kaggle/submit_arc2_600m.ipynb"
    before = make_notebook.frozen_runtime(source)
    output = tmp_path / "frozen-update.ipynb"
    make_notebook.make_notebook("600m-e330", output, runtime_notebook=source)
    after = make_notebook.frozen_runtime(output)
    assert {name: data for name, data in before.items() if name.startswith("src/")} == {
        name: data for name, data in after.items() if name.startswith("src/")}
    assert after["kaggle/multi_gpu.py"] == (ROOT / "kaggle/multi_gpu.py").read_bytes()
