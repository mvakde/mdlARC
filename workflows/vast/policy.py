"""Validated infrastructure policy kept separate from scientific experiment specs."""

import json
import math
from pathlib import Path

from workflows.vast.common import keep_b300_running

NGC_CUDA_IMAGES = (
    (13.2, "nvcr.io/nvidia/pytorch:26.05-py3"),
    (13.3, "nvcr.io/nvidia/pytorch:26.07-py3"),
    (13.4, "nvcr.io/nvidia/pytorch:26.09-py3"),
)
GPU_TEMPLATES = {
    "a100-40gb": "A100 40 GB",
    "a100-80gb": "A100 80 GB",
    "h100": "H100",
    "h200": "H200",
    "b200": "B200",
}
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "specs/vast"

DEFAULTS = {
    "schema_version": 1,
    "image": "SET_YOUR_CONTAINER_IMAGE",
    "image_selection": "cuda",
    "python": "python3",
    "gpu_names": [],
    "min_cuda_version": None,
    "num_gpus": 1,
    "min_gpu_memory_gb": 0,
    "max_gpu_memory_gb": None,
    "min_cpu_cores_effective": 0,
    "min_cpu_ram_gb": 0,
    "disk_gb": 100,
    "min_reliability": 0.99,
    "min_upload_mbps": 10,
    "min_download_mbps": 10,
    "max_hourly_usd": 2,
    "max_total_usd": 20,
    "max_runtime_hours": 8,
    "max_upload_gb": 50,
    "upload_reserve_gb": None,
    "max_download_gb": 20,
    "checkpoint_upload_mode": "automatic",
    "checkpoint_upload_every_epochs": None,
    "rental_margin_minutes": 30,
    "startup_timeout_seconds": 900,
    "artifact_interval_seconds": 120,
    "backup_grace_seconds": 1800,
    "success_action": "destroy",
    "failure_action": "stop",
    "remote_root": "/workspace/mdlarc-vast",
    "countries": [],
}


def validate(value, *, launching=False):
    if not isinstance(value, dict) or set(value) - DEFAULTS.keys():
        raise ValueError("Unknown Vast policy field(s)")
    policy = {**DEFAULTS, **value}
    if policy["schema_version"] != 1:
        raise ValueError("Unsupported Vast policy version")
    for key in ("gpu_names", "countries"):
        if not isinstance(policy[key], list) or not all(isinstance(x, str) and x for x in policy[key]):
            raise ValueError(f"{key} must be a list of names")
    for key in ("num_gpus", "startup_timeout_seconds", "artifact_interval_seconds", "backup_grace_seconds"):
        if type(policy[key]) is not int or policy[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("min_gpu_memory_gb", "min_cpu_cores_effective", "min_cpu_ram_gb", "min_reliability", "min_upload_mbps",
                "min_download_mbps", "rental_margin_minutes"):
        number(policy, key, positive=False)
    for key in ("disk_gb", "max_hourly_usd", "max_total_usd", "max_runtime_hours", "max_download_gb"):
        number(policy, key, positive=True)
    for key in ("max_upload_gb", "upload_reserve_gb"):
        if policy[key] is not None:
            number(policy, key, positive=True)
    interval = policy["checkpoint_upload_every_epochs"]
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("checkpoint_upload_every_epochs must be null or a positive integer")
    if policy["min_cuda_version"] is not None:
        number(policy, "min_cuda_version", positive=True)
    if policy["max_gpu_memory_gb"] is not None:
        number(policy, "max_gpu_memory_gb", positive=True)
        if policy["max_gpu_memory_gb"] < policy["min_gpu_memory_gb"]:
            raise ValueError("max_gpu_memory_gb must be >= min_gpu_memory_gb")
    if policy["image_selection"] not in ("cuda", "fixed"):
        raise ValueError("image_selection must be cuda or fixed")
    if policy["checkpoint_upload_mode"] not in ("automatic", "manual"):
        raise ValueError("checkpoint_upload_mode must be automatic or manual")
    for minimum, image in NGC_CUDA_IMAGES:
        if policy["image"] == image and (policy["min_cuda_version"] is not None
                                        and policy["min_cuda_version"] < minimum):
            raise ValueError(f"Set min_cuda_version >= {minimum:g} for {image}")
    if policy["min_reliability"] > 1:
        raise ValueError("min_reliability must be between 0 and 1")
    if policy["backup_grace_seconds"] >= policy["max_runtime_hours"] * 3600:
        raise ValueError("Backup grace must be shorter than the total runtime allowance")
    for key in ("image", "python", "remote_root"):
        if not isinstance(policy[key], str) or not policy[key] or any(c.isspace() for c in policy[key]):
            raise ValueError(f"{key} must be a nonempty value without whitespace")
    if not Path(policy["remote_root"]).is_absolute() or ".." in Path(policy["remote_root"]).parts or policy["remote_root"] == "/":
        raise ValueError("remote_root must be an absolute dedicated directory")
    if launching and (policy["image"] == DEFAULTS["image"] or not policy["gpu_names"]):
        raise ValueError("Set your container image and allowed GPU names before launching")
    if launching and policy["min_cuda_version"] is None:
        raise ValueError("Set min_cuda_version for your selected container before renting")
    if policy["success_action"] not in {"stop", "destroy", "keep"} or policy["failure_action"] not in {"stop", "keep"}:
        raise ValueError("Success supports stop/destroy/keep; failure supports stop/keep")
    return policy


def number(policy, key, *, positive):
    value = policy[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"Invalid numeric policy field: {key}")


def load(path, *, launching=False):
    return validate(json.loads(Path(path).read_text()), launching=launching)


def parse_overrides(values):
    """Decode explicit per-launch FIELD=JSON overrides without editing a policy."""
    overrides = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or key not in DEFAULTS:
            raise ValueError("Policy overrides must use a known policy field as FIELD=JSON_VALUE")
        if key in overrides:
            raise ValueError(f"Duplicate policy override: {key}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            raise ValueError(f"Policy override {key} requires a JSON value (quote strings)") from None
    return overrides


def apply_template(policy, name):
    """GPU templates inherit the user's saved cost and lifecycle constraints."""
    if name not in GPU_TEMPLATES:
        raise ValueError("Unknown Vast GPU template")
    values = json.loads((TEMPLATE_DIR / f"{name}.json").read_text())
    allowed = {"image", "image_selection", "gpu_names", "num_gpus", "min_cuda_version",
               "min_gpu_memory_gb", "max_gpu_memory_gb"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise ValueError("GPU templates may only override GPU and container settings")
    return validate({**policy, **values})


def image_rules(policy):
    """Only the explicitly supported NGC tags participate in CUDA selection."""
    base_minimum = next((minimum for minimum, image in NGC_CUDA_IMAGES if image == policy["image"]), None)
    if policy["image_selection"] == "fixed" or base_minimum is None:
        return []
    return [{"min_cuda_version": minimum, "image": image} for minimum, image in NGC_CUDA_IMAGES
            if minimum >= base_minimum]


def resolve_image(policy, offer):
    """Resolve once before rental; the saved policy is reused during recovery."""
    quote_offer(offer, policy)
    rules = image_rules(policy)
    if not rules:
        return policy.copy()
    matches = [rule for rule in rules if rule["min_cuda_version"] <= offer["cuda_max_good"]]
    if not matches:
        raise ValueError("Offer does not support the configured NGC container")
    selected = matches[-1]
    return validate({**policy, "image": selected["image"],
                     "min_cuda_version": max(policy["min_cuda_version"], selected["min_cuda_version"])},
                    launching=True)


def offer_query(policy):
    query = {
        "type": "on-demand", "limit": 100, "order": [["dph_total", "asc"]],
        "allocated_storage": policy["disk_gb"],
        "verified": {"eq": True}, "rentable": {"eq": True}, "rented": {"eq": False},
        "num_gpus": {"eq": policy["num_gpus"]},
        "gpu_ram": {"gte": policy["min_gpu_memory_gb"] * 1000},
        "cpu_ram": {"gte": policy["min_cpu_ram_gb"] * 1000},
        "disk_space": {"gte": policy["disk_gb"]},
        "reliability": {"gte": policy["min_reliability"]},
        "inet_up": {"gte": policy["min_upload_mbps"]},
        "inet_down": {"gte": policy["min_download_mbps"]},
        "duration": {"gte": policy["max_runtime_hours"] * 3600 + policy["rental_margin_minutes"] * 60},
        "dph_total": {"lte": policy["max_hourly_usd"]},
    }
    if policy["max_gpu_memory_gb"] is not None:
        query["gpu_ram"]["lte"] = policy["max_gpu_memory_gb"] * 1000
    if policy["gpu_names"]:
        # REST offers use names with spaces; CLI-style underscores are aliases.
        query["gpu_name"] = {"in": [x.replace("_", " ") for x in policy["gpu_names"]]}
    if policy["countries"]:
        query["geolocation"] = {"in": policy["countries"]}
    if policy["min_cuda_version"] is not None:
        query["cuda_max_good"] = {"gte": policy["min_cuda_version"]}
    if policy["min_cpu_cores_effective"] > 0:
        query["cpu_cores_effective"] = {"gte": policy["min_cpu_cores_effective"]}
    return query


def quote_offer(offer, policy):
    """Recheck the raw response, including bandwidth reservation, before renting."""
    minimum_cuda = policy["min_cuda_version"]
    if minimum_cuda is not None:
        advertised_cuda = offer.get("cuda_max_good")
        if (type(advertised_cuda) not in (int, float) or not math.isfinite(advertised_cuda)
                or advertised_cuda < minimum_cuda):
            raise ValueError(f"Offer must report CUDA support >= {minimum_cuda:g}")
    checks = {
        "gpu_ram": policy["min_gpu_memory_gb"] * 1000,
        "cpu_ram": policy["min_cpu_ram_gb"] * 1000,
        "disk_space": policy["disk_gb"], "reliability": policy["min_reliability"],
        "inet_up": policy["min_upload_mbps"], "inet_down": policy["min_download_mbps"],
        "duration": policy["max_runtime_hours"] * 3600 + policy["rental_margin_minutes"] * 60,
    }
    if policy["min_cpu_cores_effective"] > 0:
        checks["cpu_cores_effective"] = policy["min_cpu_cores_effective"]
    for key, minimum in checks.items():
        if type(offer.get(key)) not in (int, float) or not math.isfinite(offer[key]) or offer[key] < minimum:
            raise ValueError(f"Offer fails {key} requirement")
    if policy["max_gpu_memory_gb"] is not None and offer["gpu_ram"] > policy["max_gpu_memory_gb"] * 1000:
        raise ValueError("Offer exceeds max_gpu_memory_gb requirement")
    if offer.get("num_gpus") != policy["num_gpus"] or offer.get("rentable") is not True or offer.get("rented") not in (False, None):
        raise ValueError("Offer is unavailable or has the wrong GPU count")
    verified = offer.get("verified", offer.get("verification") == "verified")
    if verified is not True:
        raise ValueError("Offer is not verified")
    name = str(offer.get("gpu_name", "")).replace(" ", "_")
    if policy["gpu_names"] and name not in [x.replace(" ", "_") for x in policy["gpu_names"]]:
        raise ValueError("Offer has an unapproved GPU")
    if policy["countries"]:
        country = str(offer.get("country", offer.get("geolocation", ""))).split(",")[-1].strip()
        if country not in policy["countries"]:
            raise ValueError("Offer has an unapproved location")
    return quote_cost(offer, policy)


def quote_cost(offer, policy):
    """Apply the same price and transfer allowance to new and existing rentals."""
    hourly = offer.get("dph_total")
    if isinstance(hourly, bool) or not isinstance(hourly, (int, float)) or not math.isfinite(hourly) or hourly < 0 or hourly > policy["max_hourly_usd"]:
        raise ValueError("Offer exceeds hourly price")
    rates = [offer.get("inet_up_cost"), offer.get("inet_down_cost")]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in rates):
        raise ValueError("Offer lacks bandwidth prices")
    upload_reserve = policy.get("upload_reserve_gb")
    if policy["max_upload_gb"] is None:
        if upload_reserve is None:
            raise ValueError("Unlimited uploads require upload_reserve_gb for the spending estimate")
    else:
        upload_reserve = max(policy["max_upload_gb"], upload_reserve or 0)
    reserved = upload_reserve * rates[0] + policy["max_download_gb"] * rates[1]
    total = hourly * policy["max_runtime_hours"] + reserved
    if total > policy["max_total_usd"]:
        raise ValueError("Offer exceeds total estimated allowance including bandwidth")
    return {"hourly_usd": hourly, "bandwidth_reserved_usd": reserved, "estimated_max_usd": total}


def existing_instance(instance, policy, *, verified_previous_runs=False):
    """Use a user-selected rental's actual image and disk without creating one."""
    if verified_previous_runs and not keep_b300_running({"instance_id": instance.get("id")}, instance):
        raise ValueError("Verified previous-run reuse is limited to kept B300 instances")
    if instance.get("actual_status") not in {"running", "loading"} or instance.get("intended_status") != "running":
        raise ValueError("Selected instance must already be running or loading")
    if not verified_previous_runs and str(instance.get("label") or "").startswith(("mdlarc:", "mdlarc-vast|")):
        raise ValueError("Selected instance already belongs to a managed run; recover that run instead")
    if instance.get("num_gpus") != policy["num_gpus"]:
        raise ValueError("Selected instance has the wrong GPU count")
    gpu = str(instance.get("gpu_name", "")).replace(" ", "_")
    if gpu not in [name.replace(" ", "_") for name in policy["gpu_names"]]:
        raise ValueError("Selected instance has an unapproved GPU")
    checks = {"gpu_ram": policy["min_gpu_memory_gb"] * 1000,
              "cpu_ram": policy["min_cpu_ram_gb"] * 1000,
              "cpu_cores_effective": policy["min_cpu_cores_effective"],
              "inet_up": policy["min_upload_mbps"], "inet_down": policy["min_download_mbps"]}
    for key, minimum in checks.items():
        if minimum and (type(instance.get(key)) not in (int, float)
                        or not math.isfinite(instance[key]) or instance[key] < minimum):
            raise ValueError(f"Selected instance fails {key} requirement")
    if policy["max_gpu_memory_gb"] is not None and instance["gpu_ram"] > policy["max_gpu_memory_gb"] * 1000:
        raise ValueError("Selected instance exceeds max_gpu_memory_gb requirement")
    image = instance.get("image_uuid")
    image_minimum = next((minimum for minimum, name in NGC_CUDA_IMAGES if name == image), None)
    if image_minimum is None and image != policy["image"]:
        raise ValueError("Selected instance's container is not an approved runtime")
    disk = instance.get("disk_space")
    if type(disk) not in (int, float) or not math.isfinite(disk) or disk <= 0:
        raise ValueError("Selected instance has invalid allocated disk metadata")
    # A manually rented container cannot be replaced through adoption. Record
    # its actual allocation, then run the normal remote runtime preflight.
    selected = validate({**policy, "image": image, "image_selection": "fixed", "disk_gb": disk,
                         "min_cuda_version": max(policy["min_cuda_version"], image_minimum or 0)}, launching=True)
    cuda = instance.get("cuda_max_good")
    if type(cuda) not in (int, float) or not math.isfinite(cuda):
        raise ValueError("Selected instance's driver does not support its container")
    if cuda < selected["min_cuda_version"]:
        # An explicitly adopted NGC container can use CUDA 13 minor-version
        # compatibility on R580+ drivers. Keep the user's declared host floor
        # and require the worker's native compile/BF16/FA4 checks before data or
        # training; toolkit versions alone do not establish runtime readiness.
        driver_release = str(instance.get("driver_version", "")).split(".")[0]
        minor_compatible = (image_minimum is not None and int(cuda) == int(image_minimum) == 13
                            and cuda >= policy["min_cuda_version"]
                            and driver_release.isdigit() and int(driver_release) >= 580)
        if not minor_compatible:
            raise ValueError("Selected instance's driver does not support its container")
    return selected, quote_cost(instance, selected)


def choose_offer(offers, policy):
    matches = []
    for offer in offers:
        try:
            matches.append((quote_offer(offer, policy), offer))
        except ValueError:
            continue
    if not matches:
        raise RuntimeError("No available offer meets the saved policy, including the declared CUDA minimum; nothing was rented")
    return min(matches, key=lambda item: (item[0]["estimated_max_usd"], item[1]["id"]))
