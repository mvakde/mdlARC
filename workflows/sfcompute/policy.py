"""Autoresearch node limits, separate from the scientific experiment spec."""

import json
import math
from pathlib import Path

from workflows.vast.policy import number

TEMPLATES = {"h100": "h100-1", "h100-1": "h100-1", "h100-8": "h100-8"}
DEFAULTS = {
    "schema_version": 1, "chip": "h100-1", "num_gpus": 1,
    "image": "pytorch-2.13-cuda13.3", "python": "python3",
    "remote_root": "/home/dev/mdlarc-sfcompute", "scratch_gib": 0,
    "workspace": "default", "api_url": "https://givemeanode.com",
    "min_cpu_cores_effective": 8, "min_cpu_ram_gb": 32,
    "max_hourly_usd": 6, "max_total_usd": 50, "max_runtime_hours": 8,
    "max_upload_gb": 50, "max_download_gb": 20,
    "startup_timeout_seconds": 900, "artifact_interval_seconds": 120,
    "backup_grace_seconds": 1800, "success_action": "stop", "failure_action": "keep",
    "checkpoint_upload_mode": "automatic", "checkpoint_upload_every_epochs": None,
    "cache_runtime_validation": True, "validate_model_on_start": False,
}


def validate(value):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown SF Compute policy field(s)")
    policy = {**DEFAULTS, **value}
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported SF Compute policy version")
    if policy["chip"] not in {"h100-1", "h100-8"}:
        raise ValueError("SF Compute supports h100-1 and h100-8 node shapes")
    expected = 1 if policy["chip"] == "h100-1" else 8
    if type(policy["num_gpus"]) is not int or policy["num_gpus"] != expected:
        raise ValueError("num_gpus must match the SF Compute chip shape")
    for key in ("startup_timeout_seconds", "artifact_interval_seconds", "backup_grace_seconds"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(policy["scratch_gib"]) is not int or not 0 <= policy["scratch_gib"] <= 550:
        raise ValueError("scratch_gib must be an integer between 0 and 550")
    for key in ("max_hourly_usd", "max_total_usd", "max_runtime_hours", "max_upload_gb", "max_download_gb"):
        number(policy, key, positive=True)
    for key in ("min_cpu_cores_effective", "min_cpu_ram_gb"):
        number(policy, key, positive=False)
    for key in ("cache_runtime_validation", "validate_model_on_start"):
        if type(policy[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("success_action", "failure_action"):
        if policy[key] not in {"stop", "keep"}:
            raise ValueError("SF Compute cleanup supports stop or keep; disks are never deleted")
    if policy["checkpoint_upload_mode"] not in {"automatic", "manual"}:
        raise ValueError("checkpoint_upload_mode must be automatic or manual")
    interval = policy["checkpoint_upload_every_epochs"]
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("checkpoint_upload_every_epochs must be null or a positive integer")
    if policy["backup_grace_seconds"] >= policy["max_runtime_hours"] * 3600:
        raise ValueError("Backup grace must be shorter than the runtime allowance")
    if policy["max_hourly_usd"] * policy["max_runtime_hours"] > policy["max_total_usd"]:
        raise ValueError("Total allowance must cover the hourly ceiling for the whole runtime")
    for key in ("image", "python", "remote_root", "workspace"):
        if not isinstance(policy[key], str) or not policy[key] or any(c.isspace() for c in policy[key]):
            raise ValueError(f"{key} must be a nonempty value without whitespace")
    from workflows.vast.common import safe_name
    safe_name(policy["workspace"])
    root = Path(policy["remote_root"])
    if not root.is_relative_to("/home/dev") or root == Path("/home/dev") or ".." in root.parts:
        raise ValueError("remote_root must be a dedicated directory under persistent /home/dev")
    if policy["api_url"] not in {"https://givemeanode.com", "https://autoresearch.sfcompute.com"}:
        raise ValueError("Use an official SF Compute API URL")
    return policy


def load(path):
    return validate(json.loads(Path(path).read_text()))


def parse_overrides(values):
    result = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or key not in DEFAULTS or key in result:
            raise ValueError("Policy overrides require unique known FIELD=JSON_VALUE arguments")
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Policy override {key} requires JSON") from None
    return result


def quote(node, policy):
    micros = (node.get("rate") or {}).get("node_per_min_usd_micros")
    if type(micros) is not int or micros < 0:
        raise ValueError("SF Compute did not report a valid whole-node price")
    hourly = micros * 60 / 1e6
    total = hourly * policy["max_runtime_hours"]
    if hourly > policy["max_hourly_usd"] or total > policy["max_total_usd"]:
        raise ValueError("SF Compute node exceeds the saved compute allowance")
    return {"hourly_usd": hourly, "estimated_max_usd": total}


def verify_node(node, policy, name, identity=None, *, ready=False):
    aliases = {"h100": "h100-1", "8xh100": "h100-8"}
    if (node.get("name") != name or (identity is not None and node.get("id") != identity)
            or aliases.get(node.get("chip"), node.get("chip")) != policy["chip"]
            or node.get("image") != policy["image"]):
        raise ValueError("SF Compute node differs from the saved identity, shape or image")
    estimate = quote(node, policy)
    if ready:
        if not str(node.get("status", "")).startswith("running"):
            raise ValueError("SF Compute node is not running; observation never wakes it")
        resources = node.get("resources") or {}
        for key, minimum in (("cpu_limit", policy["min_cpu_cores_effective"]),
                             ("memory_limit_bytes", policy["min_cpu_ram_gb"] * 2**30)):
            value = resources.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError(f"Allocated SF Compute node fails {key} requirement")
    return estimate
