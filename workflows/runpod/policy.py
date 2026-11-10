"""Runpod policy and estimates, separate from the scientific experiment spec."""

import json
import math
from pathlib import Path
import re

from workflows.runpod.api import STARTUP
from workflows.vast.policy import GPU_TEMPLATES, NGC_CUDA_IMAGES, number

TEMPLATES = {
    "a100-40gb": ["NVIDIA A100-PCIE-40GB"],
    "a100-80gb": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "h100": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"],
    "h200": ["NVIDIA H200", "NVIDIA H200 NVL"],
    "b200": ["NVIDIA B200"],
}
DEFAULTS = {
    "schema_version": 1, "image": "nvcr.io/nvidia/pytorch:25.11-py3", "python": "python3",
    "gpu_type_ids": TEMPLATES["a100-80gb"], "num_gpus": 1, "cloud_type": "SECURE",
    "allowed_cuda_versions": ["13.0"], "data_center_ids": [], "countries": [],
    "min_cpu_cores_effective": 16, "min_cpu_ram_gb": 32,
    "min_upload_mbps": 50, "min_download_mbps": 2000,
    "container_disk_gb": 30, "volume_gb": 100,
    "max_hourly_usd": 2, "max_total_usd": 25, "storage_reserve_usd": 5,
    "max_runtime_hours": 8, "max_upload_gb": 50, "max_download_gb": 20,
    "startup_timeout_seconds": 900, "artifact_interval_seconds": 120, "backup_grace_seconds": 1800,
    "success_action": "destroy", "failure_action": "stop", "remote_root": "/workspace/mdlarc-runpod",
    "cache_runtime_validation": True, "validate_model_on_start": False,
}


def validate(value):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown Runpod policy field(s)")
    policy = {**DEFAULTS, **value}
    if type(policy["schema_version"]) is not int or policy["schema_version"] != 1:
        raise ValueError("Unsupported Runpod policy version")
    for key in ("cache_runtime_validation", "validate_model_on_start"):
        if type(policy[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    for key in ("gpu_type_ids", "allowed_cuda_versions", "data_center_ids", "countries"):
        if not isinstance(policy[key], list) or not all(isinstance(x, str) and x.strip() == x and x for x in policy[key]):
            raise ValueError(f"{key} must be a list of names")
    if not policy["gpu_type_ids"] or not policy["allowed_cuda_versions"]:
        raise ValueError("Set allowed GPU types and host CUDA versions before launching")
    if not all(re.fullmatch(r"\d{1,2}\.\d{1,2}", v) for v in policy["allowed_cuda_versions"]):
        raise ValueError("allowed_cuda_versions must use major.minor version strings")
    minimums = {image: str(version) for version, image in NGC_CUDA_IMAGES}
    minimums["nvcr.io/nvidia/pytorch:25.11-py3"] = "13.0"
    minimum = minimums.get(policy["image"])
    version_tuple = lambda value: tuple(map(int, value.split(".")))
    if minimum and any(version_tuple(v) < version_tuple(minimum) for v in policy["allowed_cuda_versions"]):
        raise ValueError(f"Selected image requires CUDA >= {minimum}; choose matching allowed_cuda_versions")
    for key in ("num_gpus", "container_disk_gb", "volume_gb", "startup_timeout_seconds",
                "artifact_interval_seconds", "backup_grace_seconds"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("max_hourly_usd", "max_total_usd", "max_runtime_hours", "max_upload_gb", "max_download_gb"):
        number(policy, key, positive=True)
    for key in ("min_cpu_cores_effective", "min_cpu_ram_gb", "min_upload_mbps", "min_download_mbps", "storage_reserve_usd"):
        number(policy, key, positive=False)
    if policy["cloud_type"] not in {"SECURE", "COMMUNITY"}:
        raise ValueError("cloud_type must be SECURE or COMMUNITY")
    if policy["backup_grace_seconds"] >= policy["max_runtime_hours"] * 3600:
        raise ValueError("Backup grace must be shorter than the runtime allowance")
    for key in ("image", "python", "remote_root"):
        if not isinstance(policy[key], str) or not policy[key] or any(c.isspace() for c in policy[key]):
            raise ValueError(f"{key} must be a nonempty value without whitespace")
    remote = Path(policy["remote_root"])
    if not remote.is_relative_to("/workspace") or ".." in remote.parts or str(remote) == "/workspace":
        raise ValueError("remote_root must be a dedicated directory on the persistent /workspace volume")
    if policy["success_action"] not in {"stop", "destroy", "keep"} or policy["failure_action"] not in {"stop", "keep"}:
        raise ValueError("Success supports stop/destroy/keep; failure supports stop/keep")
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
            raise ValueError(f"Policy override {key} requires a JSON value") from None
    return result


def validate_api_cuda(policy, supported):
    unsupported = [v for v in policy["allowed_cuda_versions"] if v not in supported]
    if unsupported:
        raise ValueError(f"Runpod's live API does not accept CUDA version(s) {', '.join(unsupported)}. "
                         f"Supported versions: {', '.join(supported)}. Select a compatible image and "
                         "allowed_cuda_versions explicitly; no pod was requested")


def quote(hourly, policy, *, count=1):
    # Runpod can return rates as decimal strings. Account for storage separately;
    # the allowance is an estimate, not a provider-enforced account billing cap.
    if isinstance(hourly, bool) or not isinstance(hourly, (str, int, float)):
        raise ValueError("Runpod did not report a valid compute price")
    try:
        hourly = float(hourly) * count
    except ValueError:
        raise ValueError("Runpod did not report a valid compute price") from None
    if not math.isfinite(hourly) or hourly < 0 or hourly > policy["max_hourly_usd"]:
        raise ValueError("Pod exceeds the hourly price allowance")
    total = hourly * policy["max_runtime_hours"] + policy["storage_reserve_usd"]
    if total > policy["max_total_usd"]:
        raise ValueError("Pod exceeds the estimated total allowance including storage reserve")
    return {"hourly_usd": hourly, "storage_reserved_usd": policy["storage_reserve_usd"], "estimated_max_usd": total}


def choose_gpu(rows, policy):
    candidates = []
    for row in rows:
        if row.get("id") not in policy["gpu_type_ids"]:
            continue
        price = row.get("lowestPrice") or {}
        if price.get("stockStatus") not in {"High", "Medium", "Low"}:
            continue
        try:
            estimate = quote(price.get("uninterruptablePrice"), policy, count=policy["num_gpus"])
        except ValueError:
            continue
        candidates.append((estimate, row["id"]))
    if not candidates:
        raise ValueError("No Runpod GPU has reported stock and pricing within the saved policy")
    return min(candidates, key=lambda item: item[0]["estimated_max_usd"])


def create_body(policy, label, gpu, public_key):
    return {"name": label, "imageName": policy["image"], "computeType": "GPU",
            "cloudType": policy["cloud_type"], "gpuTypeIds": [gpu], "gpuCount": policy["num_gpus"],
            "gpuTypePriority": "custom", "interruptible": False, "supportPublicIp": True,
            "allowedCudaVersions": policy["allowed_cuda_versions"],
            "dataCenterIds": policy["data_center_ids"], "countryCodes": policy["countries"],
            "minVCPUPerGPU": math.ceil(policy["min_cpu_cores_effective"] / policy["num_gpus"]),
            "minRAMPerGPU": math.ceil(policy["min_cpu_ram_gb"] / policy["num_gpus"]),
            "minUploadMbps": policy["min_upload_mbps"], "minDownloadMbps": policy["min_download_mbps"],
            "containerDiskInGb": policy["container_disk_gb"], "volumeInGb": policy["volume_gb"],
            "volumeMountPath": "/workspace", "ports": ["22/tcp"],
            "env": {"SSH_PUBLIC_KEY": public_key},
            "dockerStartCmd": ["bash", "-lc", STARTUP]}


def verify_pod(pod, policy, label, *, adopted=False):
    from workflows.runpod.api import normalize
    instance = normalize(pod)
    if (pod.get("name") != label or instance["image_uuid"] != policy["image"]
            or instance["gpu_name"] not in policy["gpu_type_ids"]
            or instance["num_gpus"] != policy["num_gpus"]
            or (not adopted and pod.get("interruptible") is not False)
            or pod.get("desiredStatus") != "RUNNING"):
        raise ValueError("Allocated pod differs from the saved Runpod identity or GPU/image policy")
    for key, minimum in (("volumeInGb", policy["volume_gb"]), ("containerDiskInGb", policy["container_disk_gb"]),
                         ("vcpuCount", policy["min_cpu_cores_effective"]), ("memoryInGb", policy["min_cpu_ram_gb"])):
        value = pod.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError(f"Allocated pod fails {key} requirement")
    if pod.get("volumeMountPath") != "/workspace":
        raise ValueError("Pod must retain its run data on /workspace")
    machine = pod.get("machine") or {}
    if machine.get("secureCloud") is not (policy["cloud_type"] == "SECURE"):
        raise ValueError("Allocated pod differs from the saved cloud type")
    if policy["data_center_ids"] and machine.get("dataCenterId") not in policy["data_center_ids"]:
        raise ValueError("Allocated pod is outside the allowed data centers")
    for key, minimum in (("maxUploadSpeedMbps", policy["min_upload_mbps"]),
                         ("maxDownloadSpeedMbps", policy["min_download_mbps"])):
        value = machine.get(key)
        if minimum and (type(value) not in (int, float) or not math.isfinite(value) or value < minimum):
            raise ValueError(f"Allocated pod fails {key} requirement")
    return quote(pod.get("costPerHr"), policy)
