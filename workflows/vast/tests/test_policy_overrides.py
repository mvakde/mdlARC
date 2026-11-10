"""Per-launch policy changes remain explicit, validated, and isolated."""

import json

import pytest

from workflows.vast import policy, run


def test_overrides_decode_json_types():
    assert policy.parse_overrides([
        "min_download_mbps=0", 'gpu_names=["H100_PCIE"]',
        "max_gpu_memory_gb=null", 'success_action="keep"',
    ]) == {"min_download_mbps": 0, "gpu_names": ["H100_PCIE"],
           "max_gpu_memory_gb": None, "success_action": "keep"}


@pytest.mark.parametrize("values,message", [
    (["inet_down=0"], "known policy field"),
    (["min_download_mbps"], "FIELD=JSON_VALUE"),
    (["min_download_mbps=0", "min_download_mbps=100"], "Duplicate"),
    (["success_action=keep"], "JSON value"),
    (["gpu_names=["], "JSON value"),
])
def test_invalid_override_syntax_is_rejected(values, message):
    with pytest.raises(ValueError, match=message):
        policy.parse_overrides(values)


def args_for_policy(tmp_path, *overrides):
    path = tmp_path / "policy.json"
    saved = {"image": "nvcr.io/nvidia/pytorch:26.05-py3", "gpu_names": ["H100_PCIE"],
             "min_cuda_version": 13.2, "min_download_mbps": 2000,
             "max_hourly_usd": 3, "max_total_usd": 40, "max_runtime_hours": 15}
    path.write_text(json.dumps(saved))
    command = ["launch", "--config", "unused.json", "--policy", str(path), "--gpu-template", "h100"]
    for override in overrides:
        command += ["--policy-override", override]
    return run.build_parser().parse_args(command), path, saved


def test_only_selected_fields_change_and_saved_policy_is_unchanged(tmp_path):
    args, path, saved = args_for_policy(tmp_path, "min_download_mbps=0")
    expected = policy.apply_template(policy.validate(saved), "h100")
    selected = run.launch_policy(args)
    assert selected == {**expected, "min_download_mbps": 0}
    assert json.loads(path.read_text()) == saved


def test_override_precedence_is_after_gpu_template_and_image_flags(tmp_path):
    args, _, _ = args_for_policy(tmp_path, "min_gpu_memory_gb=78", 'image_selection="cuda"')
    args.image_selection = "fixed"
    selected = run.launch_policy(args)
    assert selected["min_gpu_memory_gb"] == 78
    assert selected["image_selection"] == "cuda"


@pytest.mark.parametrize("override", [
    "min_download_mbps=-1", "min_download_mbps=true", 'min_download_mbps="0"',
    "min_download_mbps=NaN", "max_total_usd=0", "backup_grace_seconds=60000",
    'failure_action="destroy"', "min_cuda_version=12.9",
])
def test_overrides_keep_normal_policy_validation(tmp_path, override):
    args, _, _ = args_for_policy(tmp_path, override)
    with pytest.raises(ValueError):
        run.launch_policy(args)
