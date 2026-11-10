from pathlib import Path

import pytest

import launch_run
from src.training_benchmark import next_candidate, validate_config


def spec(**updates):
    return {"enable_finetune": False, "skip_evals": True, "evals": {},
            "training_benchmark": {}, **updates}


@pytest.mark.parametrize("updates", [
    {"checkpoint": "saved.pt"}, {"eval_only": True}, {"skip_evals": False},
    {"enable_finetune": True}, {"finetunes": {"x": {}}},
    {"defaults": {"gradient_accumulation_steps": 2}},
])
def test_benchmark_cannot_accidentally_resume_or_run_other_stages(updates):
    with pytest.raises(ValueError, match="training_benchmark"):
        launch_run.build_plan(spec(**updates), Path("test.json"))


@pytest.mark.parametrize("config", [
    {"memory_limit_gb": float("nan")}, {"memory_limit_gb": -1},
    {"allocator_margin_gb": 70}, {"batch_multiple": 0},
    {"start_batch_size": 31}, {"max_batch_size": 16},
    {"modes": ["sampled", "sampled"]}, {"modes": ["unknown"]},
    {"measure_steps": True}, {"unknown": 1},
])
def test_invalid_probe_configuration_fails_before_remote_allocation(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_full_epoch_benchmark_requires_fixed_batch_and_preserves_recipe():
    with pytest.raises(ValueError, match='fixed batch'):
        validate_config({'modes': ['epoch']})
    config = validate_config({'modes': ['epoch'], 'search': False,
                             'start_batch_size': 64, 'max_batch_size': 64})
    assert config['warmup_epochs'] == 1 and config['measure_epochs'] == 2
    plan = launch_run.build_plan(spec(training_benchmark=config,
        defaults={'dihedral_embeddings': False}), Path('test.json'))
    assert plan['pretrain']['dihedral_embeddings'] is False


def test_search_closes_an_aligned_bracket_without_repeating_a_candidate():
    low, high = 0, 512
    visited = set()
    while (candidate := next_candidate(low, high, 8)) is not None:
        assert candidate not in visited
        visited.add(candidate)
        assert low < candidate < high and candidate % 8 == 0
        if candidate <= 232:
            low = candidate
        else:
            high = candidate
    assert (low, high) == (232, 240)


def test_benchmark_routes_through_launcher_and_skips_training_and_decode(monkeypatch):
    plan = launch_run.build_plan(spec(), Path("test.json"))
    plan["skip_data_build"] = True
    calls = []
    monkeypatch.setattr(launch_run, "validate_run_split", lambda _: None)
    monkeypatch.setattr("src.training_benchmark.run_benchmark", lambda p: calls.append(p))
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *_: pytest.fail("ordinary training started"))
    monkeypatch.setattr(launch_run, "run_eval_stage_isolated", lambda *_: pytest.fail("decode started"))
    launch_run.execute_plan(plan)
    assert calls == [plan]
    assert launch_run.preview_plan(plan)["training_benchmark"]["memory_limit_gb"] == 70


def test_maximum_grid_batch_has_valid_packed_sequences_and_injected_dihedral_ids():
    import torch
    from types import SimpleNamespace
    from src.training_benchmark import make_batch
    from common import MAX_SEQ_LEN, IO_SEPARATOR_TOKEN_ID, VOCAB_SIZE

    batch = make_batch(SimpleNamespace(num_examples=17), "grid_30", 8, 42)
    assert batch["max_seqlen"] == MAX_SEQ_LEN
    assert batch["input_ids"].numel() == 8 * MAX_SEQ_LEN
    assert batch["input_ids"].max() < VOCAB_SIZE
    assert torch.equal(batch["dihedral_ids"], torch.arange(8, dtype=torch.int32))
    sequences = batch["input_ids"].reshape(8, MAX_SEQ_LEN)
    assert (sequences[torch.arange(8), batch["sep_indices"].long()] == IO_SEPARATOR_TOKEN_ID).all()


def test_capacity_rejection_is_saved_with_an_indexed_device_and_decimal_limit(tmp_path, monkeypatch):
    import json
    import torch
    import build
    from src.training_benchmark import run_trial

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec(training_benchmark={"allocator_margin_gb": 0.25})))
    output = tmp_path / "trial.json"
    total, external = 80 * 10**9, 1 * 10**9
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (total - external, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "test A100")
    budgets = []

    def set_fraction(fraction, device):
        if device.index is None:
            raise ValueError("Device must have an explicit index.")
        budgets.append(fraction * total)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_fraction)

    def reject(*args, **kwargs):
        raise torch.OutOfMemoryError("bounded capacity rejection")

    monkeypatch.setattr(build, "build_model_and_data", reject)
    run_trial({"spec_path": str(path), "output": str(output), "mode": "grid_30", "batch_size": 32})
    record = json.loads(output.read_text())
    assert record["status"] == "memory_limit"
    assert record["memory_limit_bytes"] == 70 * 10**9
    assert budgets == [68.75 * 10**9]


def test_longest_batch_repeats_only_actual_maximum_length_pairs(tmp_path):
    import json
    from common import ARCExampleDataset
    from src.training_benchmark import make_batch

    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"task": {"train": [
        {"input": [[1]], "output": [[2]]},
        {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
    ], "test": []}}))
    dataset = ARCExampleDataset(path, splits=("train",))
    batch = make_batch(dataset, "longest", 8, 42)
    longest = max(dataset.build_training_sequence(example, 0, None)[0].numel()
                  for example in dataset.examples)
    assert batch["max_seqlen"] == longest
    assert batch["input_ids"].numel() == longest * 8


def test_group_longest_probe_uses_complete_context_not_largest_target(tmp_path):
    import json
    from common import ARCExampleDataset
    from src.training_benchmark import make_batch

    def pair(side):
        return {"input": [[1] * side for _ in range(side)],
                "output": [[2] * side for _ in range(side)]}
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"large_target": {"train": [pair(30), pair(1)]},
                                "long_context": {"train": [pair(20)] * 5}}))
    dataset = ARCExampleDataset(path, splits=("train",), attention_pattern="pair_group")
    batch = make_batch(dataset, "longest", 2, 42)
    assert batch["task_ids"] == ["long_context"] * 2
    assert batch["max_seqlen"] == 4201
    assert batch["input_ids"].numel() == 2 * 4201


def test_group_synthetic_probe_respects_six_demo_cap(tmp_path):
    import json
    import torch
    from common import ARCExampleDataset, IO_SEPARATOR_TOKEN_ID
    from src.training_benchmark import make_batch

    pair = {"input": [[1]], "output": [[2]]}
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"many_demos": {"train": [pair] * 20}}))
    dataset = ARCExampleDataset(path, splits=("train",), attention_pattern="pair_group")
    batch = make_batch(dataset, "grid_30", 2, 42)
    assert batch["max_seqlen"] == 1 + 7 * 1860
    assert batch["positions_3d"][:, 2].max() == 14  # Six demos plus the target.
    assert batch["sep_indices"].dtype == torch.int32
    tokens = batch["input_ids"].reshape(2, batch["max_seqlen"])
    assert (tokens[torch.arange(2), batch["sep_indices"].long()] == IO_SEPARATOR_TOKEN_ID).all()


def test_group_variation_profile_selects_capped_context_beyond_dense_capacity(tmp_path, monkeypatch):
    import json
    import variations
    from variations import VariationTrainingDataset
    from src import training_benchmark as benchmark

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    def pair(side):
        return {"input": [[1] * side for _ in range(side)],
                "output": [[2] * side for _ in range(side)]}
    (tmp_path / "large_target.json").write_text(json.dumps({"train": [pair(30)], "test": [pair(1)]}))
    (tmp_path / "long_context.json").write_text(json.dumps({"train": [pair(20)] * 4, "test": [pair(20)] * 4}))
    (tmp_path / "parents.json").write_text(json.dumps({"parent": {"train": []}}))
    index = {"snapshot": "task-variations-v1", "archive_sha256": "test-digest",
             "parents": {"parent": [{"seed": 0, "path": "large_target.json", "pair_count": 2},
                                    {"seed": 1, "path": "long_context.json", "pair_count": 8}]},
             "eval_support_task_ids": []}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    profile = benchmark.variation_profile(index_path, "pair_group")
    assert profile["sequence_length_percentiles"]["max"] == 5881
    assert all(item["variant_index"] == 1 for item in profile["longest_refs"])
    dataset = VariationTrainingDataset(json_path=tmp_path / "parents.json", index_path=index_path,
                                      seed=42, max_augments=1, attention_pattern="pair_group")
    dataset.benchmark_variation_profile = profile
    assert benchmark.make_batch(dataset, "variation_longest", 2, 42)["max_seqlen"] == 5881
    assert benchmark.make_batch(dataset, "grid_30", 1, 42)["max_seqlen"] == 1 + 7 * 1860
    dataset.benchmark_variation_profile = dict(profile, attention_pattern="dense")
    with pytest.raises(ValueError, match="attention_pattern"):
        benchmark.make_batch(dataset, "variation_longest", 1, 42)


def test_fixed_probe_records_capacity_rejection_without_searching_or_failing(tmp_path, monkeypatch):
    import json
    from src.training_benchmark import run_benchmark

    plan = launch_run.build_plan(spec(training_benchmark={
        "search": False, "start_batch_size": 92, "max_batch_size": 92,
        "batch_multiple": 4, "modes": ["longest", "grid_30"],
    }), Path("test.json"))
    plan["run_dir"] = tmp_path
    calls = []

    def trial(command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        calls.append(request)
        Path(request["output"]).write_text(json.dumps({
            "mode": request["mode"], "batch_size": request["batch_size"],
            "status": "memory_limit",
        }))

    monkeypatch.setattr("src.training_benchmark.subprocess.run", trial)
    run_benchmark(plan)
    results = json.loads((tmp_path / "training_benchmark/results.json").read_text())
    assert len(calls) == 2
    assert all(c["batch_size"] == 92 for c in calls)
    assert all(s["first_rejected_batch"] == 92 for s in results["summary"].values())
    assert all(s["largest_confirmed_batch"] is None for s in results["summary"].values())


def test_variation_worst_case_scans_all_variants_and_uses_the_lazy_variation_path(tmp_path, monkeypatch):
    import json
    import variations
    from variations import VariationTrainingDataset
    from src import training_benchmark as benchmark

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    small = {"input": [[1]], "output": [[2]]}
    large = {"input": [[1] * 30 for _ in range(30)], "output": [[2] * 30 for _ in range(30)]}
    (tmp_path / "small.json").write_text(json.dumps({"train": [small], "test": [small]}))
    (tmp_path / "large.json").write_text(json.dumps({"train": [small], "test": [large]}))
    (tmp_path / "parents.json").write_text(json.dumps({"parent": {"train": []}}))
    index = {"snapshot": "task-variations-v1", "archive_sha256": "test-digest",
        "parents": {"parent": [{"seed": 0, "path": "small.json", "pair_count": 2},
                               {"seed": 1, "path": "large.json", "pair_count": 2}]},
        "eval_support_task_ids": []}
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))
    profile = benchmark.variation_profile(index_path)
    assert profile["solved_pairs"] == 4 and profile["variants"] == 2
    assert profile["sequence_length_percentiles"]["max"] == 1861
    assert profile["longest_refs"][0]["variant_index"] == 1
    assert profile["longest_refs"][0]["pair_index"] == 1
    assert profile["longest_refs"][0]["input_shape"] == [30, 30]
    dataset = VariationTrainingDataset(json_path=tmp_path / "parents.json", index_path=index_path,
        seed=42, max_augments=80)
    assert not dataset.examples
    dataset.benchmark_variation_profile = profile
    batch = benchmark.make_batch(dataset, "variation_longest", 8, 42)
    assert batch["max_seqlen"] == 1865  # Target plus the one available demo.
    assert batch["input_ids"].numel() == 8 * 1865


def test_unstable_boundary_backs_off_and_keeps_other_modes(tmp_path, monkeypatch):
    import json
    from src.training_benchmark import run_benchmark

    plan = launch_run.build_plan(spec(training_benchmark={
        "start_batch_size": 16, "max_batch_size": 32, "batch_multiple": 8,
        "modes": ["grid_20", "grid_30"],
    }), Path("test.json"))
    plan["run_dir"] = tmp_path

    def trial(command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        rejected = request["batch_size"] == 32 and "confirm" in request_path.name
        Path(request["output"]).write_text(json.dumps({
            "mode": request["mode"], "batch_size": request["batch_size"],
            "status": "memory_limit" if rejected else "fits", "examples_per_second": 100,
        }))

    monkeypatch.setattr("src.training_benchmark.subprocess.run", trial)
    run_benchmark(plan)
    results = json.loads((tmp_path / "training_benchmark/results.json").read_text())
    assert all(s["largest_confirmed_batch"] == 24 for s in results["summary"].values())
    assert all(s["first_rejected_batch"] == 32 for s in results["summary"].values())
