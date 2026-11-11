import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any


SCORE_RESULTS = True
VISUALIZE = False
DEFAULT_REARC_SAMPLE_COUNT = 1000

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
DATASET_SCRIPT_DIR = REPO_ROOT / "dataset_building_scripts"
DATASET_BUILD_VENV_DIR = REPO_ROOT / ".dataset-build-venv"
DATASET_BUILD_ENV_PATH = REPO_ROOT / ".dataset-build.env"
sys.path.insert(0, str(SRC_DIR))
from research_options import RESEARCH_DEFAULTS, validate_research_options
from run_layout import runs_root


DEFAULT_DATASET = {
    "prefinesub": False,
    "style": "finetune",
    "dataset": "arc2",
    "add_conceptarc": True,
    "keep_train_test_separate": False,
    "add_filtered": True,
    # ReARC is opt-in: set e.g. "add_rearc": 1000 (pairs sampled per ARC-1 training task).
    "add_rearc": None,
    "add_nvarc_training": False,
    "add_nvarc_full": False,
}

BASE_ARGS = {
    "epochs": 90,
    "batch_size": 32,
    "num_workers": None,
    "pin_memory": None,
    "persistent_workers": None,
    "prefetch_factor": None,
    "gradient_accumulation_steps": 1,
    "enable_aug": True,
    "max_augments": 80,
    # None inherits the checkpoint policy; fresh runs keep the saved fallback.
    "augmentation_sampling_mode": None,
    "nvarc_puzzles_per_epoch": None,
    "checkpoint_epochs": [],
    # Keep at most this many local periodic/final checkpoints (None keeps all).
    "checkpoint_keep_last": None,
    # Every run keeps tiny.epochNNN.pt every 50 epochs (backed up like any checkpoint) and
    # overwrites a local-only tiny.rolling.pt every 10 epochs for resuming. 0 disables either.
    "kept_checkpoint_every": 50,
    "rolling_checkpoint_every": 10,
    "optimizer": "normuon",
    "normuon_lr": 1.66e-3,
    "normuon_qkv_lr_multiplier": 1.0,
    "normuon_attention_lr_multiplier": 1.0,
    "normuon_fc_in_lr_multiplier": 1.0,
    "normuon_fc_out_lr_multiplier": 1.0,
    "normuon_residual_output_lr_multiplier": 1.0,
    "normuon_example_value_latent_lr_multiplier": 1.0,
    "normuon_other_lr_multiplier": 1.0,
    "normuon_lr_floor": None,
    "normuon_qkv_lr_floor": None,
    "normuon_attention_lr_floor": None,
    "normuon_fc_in_lr_floor": None,
    "normuon_fc_out_lr_floor": None,
    "normuon_example_value_latent_lr_floor": None,
    "normuon_other_lr_floor": None,
    "normuon_momentum": 0.95,
    "normuon_momentum_min": 0.85,
    "normuon_warmup_steps": 0,
    "normuon_cooldown_steps": 0,
    "normuon_beta2": 0.95,
    "normuon_bucket_size": 0,
    "adamw_lr": 3e-4,
    "adamw_decay_lr_multiplier": 1.0,
    "adamw_attention_lr_multiplier": 1.0,
    "adamw_token_embedding_lr_multiplier": 1.0,
    "adamw_task_factor_lr_multiplier": 1.0,
    "adamw_example_value_latent_lr_multiplier": 1.0,
    "adamw_dihedral_embedding_lr_multiplier": 1.0,
    "adamw_no_decay_lr_multiplier": 1.0,
    "adamw_loop_lr_multiplier": 1.0,
    "adamw_lr_floor": None,
    "adamw_decay_lr_floor": None,
    "adamw_attention_lr_floor": None,
    "adamw_token_embedding_lr_floor": None,
    "adamw_task_factor_lr_floor": None,
    "adamw_example_value_latent_lr_floor": None,
    "adamw_dihedral_embedding_lr_floor": None,
    "adamw_no_decay_lr_floor": None,
    "weight_decay": 0.1,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01,
    "task_embedding_weight_decay": 0.01,
    "dihedral_embedding_weight_decay": 0.01,
    # Cautious weight decay: decay only coordinates whose update shares the weight's sign.
    "cautious_weight_decay": False,
    "grad_clip": 1.0,
    "dropout": 0.1,
    "embedding_dropout": 0.1,
    "attention_dropout": 0.1,
    "seed": 42,
    "device": "cuda",
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 8,
    "input_shortcuts": None,
    "rope_spatial_base": None,
    "attention_pattern": None,
    "training_demo_schedule": "all",
    "inference_demo_policy": "all",
    # Pair-group demo-context caps (1-6). None keeps the uncapped schedules (all demos or
    # half or more of them; uncapped leave-one-out); an integer selects the balanced capped
    # schedules (demo_schedule.py). Family-sampled data always trains capped (6 when None),
    # and decoding a checkpoint trained capped defaults to 6.
    "max_training_demos": None,
    "inference_max_demos": None,
    "pair_identity_vertices": 0,
    "pair_identity_strength": 0.25,
    # Block-upscale augmentation (src/common.py): training draws one of up to scale_max_specs
    # specs per regular training sequence; inference_scale_views mixes demo-bounded scale views
    # into a full decode. Both opt-in (scale costs ~1.57x tokens/epoch and ~1.38x peak memory).
    "scale_augmentation": False,
    "scale_max_specs": 12,
    "scale_anisotropic": True,
    "inference_scale_views": False,
    # Full-decode selection (opt-in): demo-derived symbolic guards drop candidates, then each
    # output's top rescore_top_k voted candidates are re-ranked by summed log-likelihood over the
    # decode's views (prefix-cached: one prompt prefill per view). Plain-vote top-2 stays reported.
    "inference_symbolic_guards": False,
    "inference_rescore": False,
    "rescore_top_k": 20,
    "rescore_prefix_cache": True,
    "depth_diagnostics_every": 0,
    # torch.compile extras: compiled autograd for the backward dispatch,
    # hierarchical (per-block) compilation of the repeated transformer block,
    # the torch.compile mode (max-autotune* for Triton GEMMs with fused
    # epilogues, reduce-overhead for CUDA graphs) and inductor's coordinate
    # descent tuning of kernel launch configs.
    "compiled_autograd": False,
    "compile_regions": False,
    "compile_mode": "default",
    "coordinate_descent_tuning": False,
    # R1's effective latent-bank cap; an explicit size remains configurable.
    "num_task_embedding_latents": None,
    # None inherits checkpoints; fresh models use task conditioning and no
    # transform embedding because demonstrations share the target transform.
    "task_embeddings": None,
    "dihedral_embeddings": None,
    # None inherits checkpoint settings; fresh training resolves to 4 / 0.3.
    "n_predict": None,
    "mtp_weight": None,
    # Omitted research options inherit saved checkpoint configuration.
    **dict.fromkeys(RESEARCH_DEFAULTS),
    "mtp_decay": [0.8, 0.9, 1.0],
    # Saturation-safe hard-target cross-entropy (src/stable_losses.py) for every training
    # loss instead of F.cross_entropy; the loop-composition-v3 (Kaggle d1024) recipe used it.
    "stable_cross_entropy": False,
    "inference_temperature": None,
    "inference_top_k": None,
    # Self-speculative greedy decoding (MTP or NextLat drafts, main-head verification); a
    # sampled decode (temperature > 0) turns the default off.
    "inference_speculative": True,
    # torch.compile scope for training: "blocks" compiles each transformer block once (looped
    # blocks reuse it), "model" compiles the whole forward, "none" skips compilation.
    "compile_scope": "blocks",
    "warmup_pct": 0.0,
    "resume_warmup_pct": 0.0,
    "wsd_decay_start_pct": 0.8,
    "lr_floor": 0.0,
    "train_log_mode": "never",
    "log_location": "none",
}

TOP_LEVEL_KEYS = {
    "distributed",
    "finetune_only",
    "finetune_resume",
    "evaluation_budget_benchmark",
    "speculative_benchmark",
    "backend_benchmark",
    "training_benchmark",
    "run_name",
    "FA",
    "checkpoint",
    "validate_cuda",
    "eval_only",
    "skip_data_build",
    "skip_evals",
    "score_results",
    "visualize",
    "dataset",
    "enable_finetune",
    "defaults",
    "pretrain",
    "finetune_defaults",
    "finetunes",
    "eval_defaults",
    "evals",
}
DATASET_KEYS = set(DEFAULT_DATASET) | {"pretrain_data", "evaluation"}
TRAINING_CONFIG_KEYS = set(BASE_ARGS) | {
    "FA",
    "rearc_pairs_per_epoch",
    "rearc_max_demos",
    "nvarc_pairs_per_epoch",
    "reset_training_state",
    "inherit_checkpoint_rearc_data_path",
}
EVAL_CONFIG_KEYS = TRAINING_CONFIG_KEYS | {
    "mode",
    "checkpoint",
    "from",
    "allow_missing_tasks",
    "extend_checkpoint_task_ids",
    "group",
    "checkpoint_task_id_prefix",
    "reference_submission",
    "candidates_path",
    "rescore_top_k",
    # Decode a looped checkpoint at another iteration count (K-independent loop parameters only).
    "eval_loop_iterations",
    # Summed padded prompt tokens per decode batch (None: MDLARC_PREFILL_TOKEN_BUDGET
    # or 524,288).
    "prefill_token_budget",
}

build = None
evaluate = None
train = None
utils = None


def load_runtime_modules() -> None:
    global build, evaluate, train, utils
    if build is not None:
        return
    import build as build_module
    import evaluate as evaluate_module
    import train as train_module
    import utils as utils_module

    build = build_module
    evaluate = evaluate_module
    train = train_module
    utils = utils_module
    print("Modules imported successfully.")


def merge_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(part)
    return merged


def expect_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise TypeError(f"{name} must be a JSON object.")


def reject_unknown_keys(config: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(key for key in config if key not in allowed)
    if unknown:
        raise ValueError(
            f"{name} contains unsupported option(s): " + ", ".join(unknown)
        )


def resolve_input_checkpoint(value: Any, run_dir: Path) -> Path | None:
    """Resolve the spec's local checkpoint path ("runs/<run>/..." is relative to the runs root)."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("checkpoint must be a local path string.")
    path = Path(value)
    if not path.is_absolute() and path.parts[:1] == ("runs",):
        path = run_dir.parent.joinpath(*path.parts[1:])
    return path


def training_overrides(
    value: Any,
    name: str,
    *,
    allowed_keys: set[str] = TRAINING_CONFIG_KEYS,
) -> dict[str, Any]:
    config = dict(expect_dict(value, name))
    reject_unknown_keys(config, allowed_keys, name)
    validate_research_options(config, n_layers=config.get("n_layers"))
    from src.checkpoint_io import validate_checkpoint_keep_last
    validate_checkpoint_keep_last(config.get("checkpoint_keep_last"))
    if "inference_speculative" in config and not (
            isinstance(config["inference_speculative"], bool) or config["inference_speculative"] == "auto"):
        raise ValueError(f"{name}.inference_speculative must be a boolean or auto.")
    for key in ("max_training_demos", "inference_max_demos"):
        cap = config.get(key)
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 6):
            raise ValueError(f"{name}.{key} must be null or an integer from 1 to 6.")
    if not isinstance(config.get("stable_cross_entropy", False), bool):
        raise ValueError(f"{name}.stable_cross_entropy must be a boolean.")
    if config.get("dihedral_embeddings") is not None and not isinstance(config["dihedral_embeddings"], bool):
        raise ValueError(f"{name}.dihedral_embeddings must be a boolean.")
    if config.get("task_embeddings") is not None and not isinstance(config["task_embeddings"], bool):
        raise ValueError(f"{name}.task_embeddings must be a boolean.")
    if config.get("attention_pattern") not in (None, "dense", "pair_group"):
        raise ValueError(f"{name}.attention_pattern must be 'dense' or 'pair_group'.")
    for key, allowed in (("training_demo_schedule", ("all", "half_or_more", "two")),
                         ("inference_demo_policy", ("all", "leave_one_out", "two"))):
        if key in config and config[key] not in allowed:
            raise ValueError(f"{name}.{key} must be one of {allowed}")
    vertices = config.get("pair_identity_vertices", 0)
    if isinstance(vertices, bool) or not isinstance(vertices, int) or vertices < 0 or vertices == 1:
        raise ValueError(f"{name}.pair_identity_vertices must be 0 or integer >= 2")
    strength = config.get("pair_identity_strength", 0.25)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength <= 0:
        raise ValueError(f"{name}.pair_identity_strength must be positive and finite")
    if config.get("augmentation_sampling_mode") not in (None, "full_task_prefix", "single_grid_prefix", "context_palette"):
        raise ValueError(
            f"{name}.augmentation_sampling_mode must be 'full_task_prefix', 'single_grid_prefix' or 'context_palette'."
        )
    n_predict = config.get("n_predict")
    if config.get("input_shortcuts") not in (None, "learned", "fixed"):
        raise ValueError(f"{name}.input_shortcuts must be learned or fixed.")
    spatial_base = config.get("rope_spatial_base")
    if spatial_base is not None and (
        isinstance(spatial_base, bool) or not isinstance(spatial_base, (int, float))
        or not math.isfinite(spatial_base) or spatial_base <= 1
    ):
        raise ValueError(f"{name}.rope_spatial_base must be finite and > 1.")
    for key in ("scale_augmentation", "scale_anisotropic", "inference_scale_views", "inference_symbolic_guards",
                "inference_rescore", "rescore_prefix_cache"):
        if key in config and not isinstance(config[key], bool):
            raise ValueError(f"{name}.{key} must be a boolean.")
    scale_specs = config.get("scale_max_specs", 12)
    if isinstance(scale_specs, bool) or not isinstance(scale_specs, int) or scale_specs < 1:
        raise ValueError(f"{name}.scale_max_specs must be an integer >= 1.")
    if config.get("compile_scope") not in (None, "model", "blocks", "none"):
        raise ValueError(f"{name}.compile_scope must be 'model', 'blocks' or 'none'.")
    for key in ("rearc_pairs_per_epoch", "rearc_max_demos"):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name}.{key} must be an integer >= 1.")
    diagnostics = config.get("depth_diagnostics_every", 0)
    if isinstance(diagnostics, bool) or not isinstance(diagnostics, int) or diagnostics < 0:
        raise ValueError(f"{name}.depth_diagnostics_every must be an integer >= 0.")
    for key in ("kept_checkpoint_every", "rolling_checkpoint_every"):
        every = config.get(key, 0)
        if isinstance(every, bool) or not isinstance(every, int) or every < 0:
            raise ValueError(f"{name}.{key} must be an integer >= 0.")
    if not isinstance(config.get("cautious_weight_decay", False), bool):
        raise ValueError(f"{name}.cautious_weight_decay must be a boolean.")
    for key in ("compiled_autograd", "compile_regions", "coordinate_descent_tuning"):
        if not isinstance(config.get(key, False), bool):
            raise ValueError(f"{name}.{key} must be a boolean.")
    compile_mode = config.get("compile_mode", "default")
    if compile_mode not in ("default", "max-autotune-no-cudagraphs", "max-autotune", "reduce-overhead"):
        raise ValueError(f"{name}.compile_mode must be default, max-autotune-no-cudagraphs, max-autotune or reduce-overhead.")
    residual_lr = config.get("normuon_residual_output_lr_multiplier", 1.0)
    if (isinstance(residual_lr, bool) or not isinstance(residual_lr, (int, float))
            or not math.isfinite(residual_lr) or residual_lr < 0):
        raise ValueError(f"{name}.normuon_residual_output_lr_multiplier must be finite and >= 0.")
    if n_predict is not None and (
        isinstance(n_predict, bool) or not isinstance(n_predict, int) or n_predict < 1
    ):
        raise ValueError(f"{name}.n_predict must be an integer >= 1.")
    mtp_weight = config.get("mtp_weight")
    if mtp_weight is not None and (
        isinstance(mtp_weight, bool)
        or not isinstance(mtp_weight, (int, float))
        or not math.isfinite(mtp_weight)
        or mtp_weight < 0
    ):
        raise ValueError(f"{name}.mtp_weight must be finite and >= 0.")
    if "mtp_decay" in config:
        decay = config["mtp_decay"]
        if (
            not isinstance(decay, list)
            or len(decay) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in decay
            )
            or not 0 <= decay[0] < decay[1] <= 1
            or not 0 <= decay[2] <= 1
        ):
            raise ValueError(
                f"{name}.mtp_decay must be [start, end, final_multiplier] with finite "
                "values satisfying 0 <= start < end <= 1 and 0 <= final_multiplier <= 1."
            )
    return config


def normalize_fa_value(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be 2, 3 or 4, not a boolean.")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be 2, 3 or 4.") from exc
    if version not in (2, 3, 4):
        raise ValueError(f"{name} must be 2, 3 or 4.")
    return version


def normalize_fa_config(config: dict[str, Any], name: str) -> None:
    if "FA" in config:
        config["FA"] = normalize_fa_value(config["FA"], f"{name}.FA")
        if config["FA"] == 4 and config.get("attention_dropout", 0.0) != 0:
            raise ValueError(f"{name}: FA=4 requires attention_dropout=0.0.")


def greedy_inference(config: dict[str, Any]) -> bool:
    temperature, top_k = config.get("inference_temperature"), config.get("inference_top_k")
    return (temperature is not None and temperature <= 0) or (temperature is None and top_k is None) or top_k == 1


def default_speculative_off_when_sampling(config: dict[str, Any], explicit: bool) -> None:
    """Speculative decoding is on by default for greedy decodes only; an explicit request is validated.

    "auto" resolves here: speculative exactly when the decode is greedy.
    """
    if config.get("inference_speculative") == "auto":
        config["inference_speculative"] = greedy_inference(config)
    if not explicit and config.get("inference_speculative") and not greedy_inference(config):
        config["inference_speculative"] = False


def validate_eval_mode(config: dict[str, Any], name: str) -> None:
    prefix = config.get("checkpoint_task_id_prefix")
    if prefix is not None and (not isinstance(prefix, str) or not prefix or not prefix.endswith("/")):
        raise ValueError(f"{name}.checkpoint_task_id_prefix must be a nonempty namespace ending in '/'.")
    if config.get("mode") not in ("full-decode", "accuracy-only", "rescore"):
        raise ValueError(f'{name}.mode must be "full-decode", "accuracy-only" or "rescore".')
    if config.get("mode") == "rescore" and name != "eval_defaults" and not isinstance(config.get("candidates_path"), str):
        raise ValueError(f"{name}.candidates_path must name a decode's candidates.json.gz for rescore.")
    top_k = config.get("rescore_top_k")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError(f"{name}.rescore_top_k must be a positive integer.")
    budget = config.get("prefill_token_budget")
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError(f"{name}.prefill_token_budget must be a positive integer or null.")
    loops = config.get("eval_loop_iterations")
    if loops is not None and (isinstance(loops, bool) or not isinstance(loops, int) or loops < 2):
        raise ValueError(f"{name}.eval_loop_iterations must be null or an integer >= 2.")
    if config.get("inference_speculative") and config.get("mode") == "full-decode":
        if not greedy_inference(config):
            raise ValueError(f"{name}.inference_speculative requires greedy inference (temperature=0).")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return path


def oracle_summary_fields(submission_path: Path) -> dict[str, Any]:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from utils import oracle_summary_fields as fields
    return fields(submission_path)


def write_score_summary(
    submission_path: Path,
    score_result: dict[str, Any],
    *,
    checkpoint_ref: str,
    run_name: str,
    group: str = "all",
) -> Path:
    return write_json(
        Path(submission_path).with_name("score.json"),
        {
            "score": score_result["score"],
            "max_score": score_result["max_score"],
            "percentage": score_result["percentage"],
            "fully_solved_tasks": score_result["fully_solved_tasks"],
            "checkpoint_ref": checkpoint_ref,
            "run_name": run_name,
            "group": group,
            "timestamp": iso_now(),
            **oracle_summary_fields(submission_path),
        },
    )


def dataset_builder_python() -> str:
    bin_dir = DATASET_BUILD_VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
    candidate = bin_dir / ("python.exe" if os.name == "nt" else "python")
    return str(candidate) if candidate.exists() else sys.executable


def dataset_builder_env() -> dict[str, str]:
    env = dict(os.environ)
    bin_dir = DATASET_BUILD_VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
    if bin_dir.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = (
            f"{bin_dir}{os.pathsep}{current_path}" if current_path else str(bin_dir)
        )
    update_env_from_file(env, DATASET_BUILD_ENV_PATH)
    return env


def update_env_from_file(env: Any, path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if parts and parts[0] == "export":
            parts = parts[1:]
        if len(parts) != 1 or "=" not in parts[0]:
            continue
        key, value = parts[0].split("=", 1)
        if key:
            env[key] = value


# Search paths in which an empty entry (a leading, trailing or doubled ":") or "." means the current
# directory. Some container images (for example NGC PyTorch 26.09) end LIBRARY_PATH with ":"; gcc,
# which Triton calls for torch.compile, then reads this repository's specs/ folder as a gcc spec file
# and fails with "cannot read spec file './specs'".
SEARCH_PATH_VARIABLES = (
    "PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "COMPILER_PATH", "CPATH", "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH", "PKG_CONFIG_PATH", "PYTHONPATH",
)


def drop_current_directory_search_entries(env: Any) -> None:
    """Remove empty and "." entries from the search-path variables in ``env``."""
    for name in SEARCH_PATH_VARIABLES:
        value = env.get(name)
        if value is None:
            continue
        kept = [entry for entry in value.split(os.pathsep) if entry not in ("", ".")]
        if kept:
            env[name] = os.pathsep.join(kept)
        else:
            del env[name]


def run_dataset_script(script_path: Path, *args: str) -> None:
    command = [dataset_builder_python(), str(script_path), *args]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, env=dataset_builder_env())


def ensure_grouped_assets(dataset_cfg: dict[str, Any]) -> None:
    if "pretrain_data" in dataset_cfg:
        from dataset_building_scripts.build_variation_dataset import EVALUATION_FILES, ORIGINAL_FILES

        source = dataset_cfg["evaluation"]["source"]
        required = list(EVALUATION_FILES[source]) if source in EVALUATION_FILES else [ORIGINAL_FILES[source]]
        data = dataset_cfg["pretrain_data"]
        originals = data["sources"] if data["kind"] == "original_tasks" else data["original_sources"]
        required.extend(ORIGINAL_FILES[name] for name in originals)
        if all((REPO_ROOT / "assets_tmp" / path).exists() for path in required):
            return
        options = ("--conceptarc-only",) if set(required) == {ORIGINAL_FILES["conceptarc"]} else ()
        run_dataset_script(DATASET_SCRIPT_DIR / "download_and_group.py", *options)
        return
    if dataset_cfg["dataset"] == "conceptarc":
        conceptarc_path = REPO_ROOT / "assets_tmp" / "ConceptARC" / "concept_all.json"
        if not conceptarc_path.exists():
            run_dataset_script(
                DATASET_SCRIPT_DIR / "download_and_group.py",
                "--conceptarc-only",
            )
        return

    required_paths = [
        REPO_ROOT / "assets_tmp" / "ARC-1" / "arc1_train_both.json",
        REPO_ROOT / "assets_tmp" / "ARC-2" / "arc2_train_both.json",
        REPO_ROOT / "assets_tmp" / "ConceptARC" / "concept_all.json",
    ]
    if dataset_cfg.get("add_rearc") is not None:
        required_paths.append(REPO_ROOT / "assets_tmp" / "RE-ARC" / "tasks")
    if all(path.exists() for path in required_paths):
        return
    run_dataset_script(DATASET_SCRIPT_DIR / "download_and_group.py")


def build_dataset(dataset_cfg: dict[str, Any]) -> None:
    ensure_grouped_assets(dataset_cfg)
    if dataset_cfg.get("prefinesub", False):
        run_dataset_script(DATASET_SCRIPT_DIR / "build_datasets.py", "--prefinesub")
        return

    args = [
        "--style",
        str(dataset_cfg["style"]),
        "--dataset",
        str(dataset_cfg["dataset"]),
    ]
    if dataset_cfg["add_conceptarc"]:
        args.append("--add-conceptarc")
    if dataset_cfg["keep_train_test_separate"]:
        args.append("--keep-train-test-separate")
    if dataset_cfg["add_filtered"]:
        args.append("--add-filtered")
    if dataset_cfg["add_rearc"] is not None:
        args.extend(["--add-rearc", str(dataset_cfg["add_rearc"])])
    if dataset_cfg["add_nvarc_training"]:
        args.append("--add-nvarc-training")
    if dataset_cfg["add_nvarc_full"]:
        args.append("--add-nvarc-full")
    run_dataset_script(DATASET_SCRIPT_DIR / "build_datasets.py", *args)


def stage_dir(run_dir: Path, stage: str, name: str | None = None) -> Path:
    if stage == "pretrain":
        return run_dir / "pretrain"
    if stage == "finetune" and name is not None:
        return run_dir / "finetunes" / name
    if stage == "eval" and name is not None:
        return run_dir / "evals" / name
    raise ValueError(f"Unsupported stage: {stage!r}")


def stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "epochs": stage["epochs"],
        "FA": stage.get("FA"),
        "max_augments": stage["max_augments"],
        "augmentation_sampling_mode": stage.get("augmentation_sampling_mode") or (
            "checkpoint" if stage.get("checkpoint_path") else "full_task_prefix"
        ),
        "rearc_pairs_per_epoch": stage.get("rearc_pairs_per_epoch"),
        "rearc_max_demos": stage.get("rearc_max_demos"),
        "compile_scope": stage.get("compile_scope", "blocks"),
        "nvarc_pairs_per_epoch": stage.get("nvarc_pairs_per_epoch"),
        "nvarc_puzzles_per_epoch": stage.get("nvarc_puzzles_per_epoch"),
        "checkpoint_epochs": stage["checkpoint_epochs"],
        "checkpoint_keep_last": stage.get("checkpoint_keep_last"),
        "kept_checkpoint_every": stage.get("kept_checkpoint_every", 0),
        "rolling_checkpoint_every": stage.get("rolling_checkpoint_every", 0),
        "cautious_weight_decay": stage.get("cautious_weight_decay", False),
        "resume_warmup_pct": stage["resume_warmup_pct"],
        "mtp_decay": stage["mtp_decay"],
        "train_log_file": stage["train_log_file"],
        "save_path": stage["save_path"],
        "batch_size_per_gpu": stage["batch_size"],
        "gradient_accumulation_steps": stage["gradient_accumulation_steps"],
        "training_demo_schedule": stage.get("training_demo_schedule", "all"),
        "max_training_demos": stage.get("max_training_demos"),
        "stable_cross_entropy": stage.get("stable_cross_entropy", False),
    }
    if stage.get("checkpoint_path"):
        summary["checkpoint_path"] = stage["checkpoint_path"]
    if stage.get("scale_augmentation"):
        summary["scale_augmentation"] = {"max_specs": stage.get("scale_max_specs", 12),
                                         "anisotropic": stage.get("scale_anisotropic", True)}
    if stage.get("inference_scale_views"):
        summary["inference_scale_views"] = {"max_specs": stage.get("scale_max_specs", 12),
                                            "anisotropic": stage.get("scale_anisotropic", True)}
    if summary["FA"] is None:
        summary.pop("FA")
    for key, fresh_default in (("n_predict", 4), ("mtp_weight", 0.3), ("rope_spatial_base", 1000.0),
                               ("task_embeddings", True), ("dihedral_embeddings", False),
                               ("attention_pattern", "dense")):
        value = stage.get(key)
        summary[key] = value if value is not None else (
            "checkpoint" if stage.get("checkpoint_path") else fresh_default
        )
    for key, fresh_default in RESEARCH_DEFAULTS.items():
        value = stage.get(key)
        summary[key] = value if value is not None else (
            "checkpoint" if stage.get("checkpoint_path") else fresh_default
        )
    return summary


def checkpoint_path_for_epoch(save_path: Path, epoch: int, total_epochs: int) -> Path:
    width = max(2, len(str(total_epochs)))
    return save_path.with_name(f"{save_path.stem}.epoch{epoch:0{width}d}{save_path.suffix}")


def checkpoint_is_saved(stage: dict[str, Any], epoch: int) -> bool:
    total_epochs = int(stage["epochs"])
    if epoch == total_epochs:
        return True
    kept = stage.get("kept_checkpoint_every", 0)
    if kept and 0 < epoch < total_epochs and epoch % kept == 0:
        return True
    schedule = stage.get("checkpoint_epochs", [])
    if schedule is None:
        return False
    if isinstance(schedule, bool):
        raise TypeError("checkpoint_epochs must be an int or list of ints, not bool.")
    if isinstance(schedule, int):
        return epoch > 0 and epoch <= total_epochs and epoch % schedule == 0
    return epoch in set(int(item) for item in schedule)


def resolve_checkpoint_ref(ref: str, plan: dict[str, Any]) -> Path:
    parts = ref.split(".")
    if parts == ["checkpoint"]:
        if not plan.get("eval_only", False):
            raise ValueError(
                "The 'checkpoint' eval reference is only available when eval_only is true."
            )
        checkpoint_path = plan["pretrain"].get("checkpoint_path")
        if checkpoint_path is None:
            raise ValueError("eval_only requires a top-level checkpoint path.")
        return Path(checkpoint_path)
    if parts == ["pretrain", "final"]:
        if plan.get("eval_only", False):
            checkpoint_path = plan["pretrain"].get("checkpoint_path")
            if checkpoint_path is None:
                raise ValueError("eval_only requires a top-level checkpoint path.")
            return Path(checkpoint_path)
        return plan["pretrain"]["save_path"]
    if len(parts) == 3 and parts[:2] == ["pretrain", "epoch"]:
        epoch = int(parts[2])
        stage = plan["pretrain"]
    elif len(parts) == 3 and parts[0] == "finetune" and parts[2] == "final":
        name = parts[1]
        if name not in plan["finetunes"]:
            raise ValueError(f"Unknown finetune target: {name}")
        return plan["finetunes"][name]["save_path"]
    elif len(parts) == 4 and parts[0] == "finetune" and parts[2] == "epoch":
        name = parts[1]
        if name not in plan["finetunes"]:
            raise ValueError(f"Unknown finetune target: {name}")
        epoch = int(parts[3])
        stage = plan["finetunes"][name]
    else:
        raise ValueError(f"Unsupported checkpoint ref: {ref}")

    if not checkpoint_is_saved(stage, epoch):
        raise ValueError(
            f"{ref} was requested but epoch {epoch} is not saved by {stage['name']}."
        )
    keep_last = stage.get("checkpoint_keep_last")
    if keep_last is not None:
        schedule = stage.get("checkpoint_epochs") or []
        scheduled = (range(schedule, stage["epochs"] + 1, schedule)
                     if isinstance(schedule, int) and schedule > 0 else schedule)
        retained = sorted({int(item) for item in scheduled if int(item) <= stage["epochs"]})
        # The final file takes one retention slot; keep_last=1 saves it only once.
        retained = retained[-(keep_last - 1):] if keep_last > 1 else []
        if epoch not in retained:
            raise ValueError(f"{ref} will be removed by checkpoint_keep_last={keep_last}; "
                             "use a final checkpoint or increase retention.")
    return checkpoint_path_for_epoch(stage["save_path"], epoch, stage["epochs"])


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Experiment spec must be a JSON object.")
    return raw


def build_dataset_config(raw_dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    reject_unknown_keys(raw_dataset_cfg, DATASET_KEYS, "dataset")
    if (raw_dataset_cfg.get("pretrain_data") or {}).get("kind") == "family_sampled":
        from dataset_building_scripts.family_data import normalize_config
        return merge_dicts(DEFAULT_DATASET, normalize_config(raw_dataset_cfg, REPO_ROOT))
    if "pretrain_data" in raw_dataset_cfg:
        from dataset_building_scripts.task_splits import load_parent_split

        evaluation_datasets = {
            "arc1_eval": "arc1", "arc2_eval": "arc2", "conceptarc": "conceptarc",
            "arc2_train": "arc2", "arc1_filtered": "arc1",
        }
        default_source = {"arc1": "arc1_eval", "arc2": "arc2_eval", "conceptarc": "conceptarc"}.get(
            raw_dataset_cfg.get("dataset", "arc2")
        )
        evaluation = dict(expect_dict(raw_dataset_cfg.get("evaluation"), "dataset.evaluation"))
        reject_unknown_keys(evaluation, {"source", "split_manifest"}, "dataset.evaluation")
        evaluation.setdefault("source", default_source)
        eval_source = evaluation["source"]
        if not isinstance(eval_source, str) or eval_source not in evaluation_datasets:
            raise ValueError("dataset.evaluation.source is unsupported")
        dataset_name = evaluation_datasets[eval_source]
        if raw_dataset_cfg.get("dataset", dataset_name) != dataset_name:
            raise ValueError("dataset.dataset and dataset.evaluation.source disagree")
        split_path = evaluation.get("split_manifest")
        if eval_source in {"conceptarc", "arc2_train", "arc1_filtered"}:
            manifest, _ = load_parent_split(REPO_ROOT, split_path)
            if manifest["source"] != eval_source:
                raise ValueError("Parent split source must match dataset.evaluation.source")
        elif "split_manifest" in evaluation:
            raise ValueError("Use a parent-task source, rather than an official eval source, with split_manifest")
        if raw_dataset_cfg.get("prefinesub", False):
            raise ValueError("dataset.pretrain_data cannot be combined with prefinesub.")
        disallowed = (
            "add_conceptarc", "add_filtered", "add_rearc",
            "add_nvarc_training", "add_nvarc_full", "keep_train_test_separate",
        )
        if any(raw_dataset_cfg.get(key) not in (None, False) for key in disallowed):
            raise ValueError("Legacy dataset add-ons cannot be combined with pretrain_data; use original_sources.")
        data = expect_dict(raw_dataset_cfg["pretrain_data"], "dataset.pretrain_data")
        kind = data.get("kind")
        if kind not in {"task_variations", "original_tasks"}:
            raise ValueError("dataset.pretrain_data.kind must be 'task_variations' or 'original_tasks'.")
        allowed_data_keys = {
            "kind", "snapshot", "sources", "original_sources", "sampling",
            "eval_support_repeats_per_epoch", "public_pretraining", "compositions",
        } if kind == "task_variations" else {"kind", "sources", "eval_support_repeats_per_epoch"}
        reject_unknown_keys(data, allowed_data_keys, "dataset.pretrain_data")
        if kind == "task_variations" and data.get("snapshot") != "task-variations-v1":
            raise ValueError(
                "dataset.pretrain_data requires kind='task_variations' "
                "and snapshot='task-variations-v1'."
            )
        public_pretraining = data.get("public_pretraining", False)
        if not isinstance(public_pretraining, bool):
            raise ValueError("public_pretraining must be a boolean")
        if public_pretraining and (raw_dataset_cfg.get("style") != "pretrain" or split_path):
            raise ValueError("public_pretraining requires pretrain style without a held-out parent split")
        allowed_sources = {"arc2_train", "conceptarc", "arc1_filtered"}
        if public_pretraining:
            allowed_sources.add("arc2_eval")
        sources = data.get("sources")
        originals = data.get("original_sources", [])
        for name, value in (("sources", sources), ("original_sources", originals)):
            if not isinstance(value, list) or (name == "sources" and not value):
                qualifier = "nonempty " if name == "sources" else ""
                raise ValueError(f"dataset.pretrain_data.{name} must be a {qualifier}list.")
            if (
                any(not isinstance(item, str) or item not in allowed_sources for item in value)
                or len(value) != len(set(value))
            ):
                raise ValueError(f"dataset.pretrain_data.{name} contains duplicate or unsupported sources.")
        sampling = expect_dict(data.get("sampling"), "dataset.pretrain_data.sampling")
        reject_unknown_keys(sampling, {
            "variations_per_parent_per_epoch", "seed_schedule", "target_pairs", "embedding_scope",
            "deduplicate_variation_inputs",
        }, "dataset.pretrain_data.sampling")
        sampling = merge_dicts({
            "variations_per_parent_per_epoch": 1,
            "seed_schedule": "shuffled_cycle",
            "target_pairs": "all_solved",
            "embedding_scope": "parent",
        }, sampling)
        count = sampling["variations_per_parent_per_epoch"]
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 42:
            raise ValueError("variations_per_parent_per_epoch must be an integer from 1 to 42.")
        for key, expected in (
            ("seed_schedule", "shuffled_cycle"),
            ("target_pairs", "all_solved"),
        ):
            if sampling[key] != expected:
                raise ValueError(f"dataset.pretrain_data.sampling.{key} must be {expected!r}.")
        if sampling["embedding_scope"] not in ("parent", "variation"):
            raise ValueError("dataset.pretrain_data.sampling.embedding_scope must be 'parent' or 'variation'.")
        if not isinstance(sampling.get("deduplicate_variation_inputs", True), bool):
            raise ValueError("dataset.pretrain_data.sampling.deduplicate_variation_inputs must be a boolean.")
        style = raw_dataset_cfg.get("style", "finetune")
        if style not in {"pretrain", "finetune"}:
            raise ValueError("dataset.style must be 'pretrain' or 'finetune'.")
        repeats = data.get("eval_support_repeats_per_epoch", 1 if style == "pretrain" and not split_path and not public_pretraining else 0)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0:
            raise ValueError("eval_support_repeats_per_epoch must be a nonnegative integer.")
        if split_path and repeats != 0:
            raise ValueError("A parent split must keep evaluation support out of pretraining.")
        if style == "pretrain" and not split_path and not public_pretraining and repeats < 1:
            raise ValueError("pretrain style requires at least one eval support repeat.")
        if public_pretraining and repeats != 0:
            raise ValueError("public_pretraining uses solved original sources rather than evaluation support repeats")
        if style == "finetune" and repeats != 0:
            raise ValueError("finetune style must not include eval support in pretraining.")
        if kind == "original_tasks" and repeats > 1:
            raise ValueError("original_tasks supports at most one eval support repeat per epoch.")
        pretrain_data = {
            "kind": kind,
            "sources": list(sources),
            "eval_support_repeats_per_epoch": repeats,
        }
        if kind == "task_variations":
            pretrain_data.update(
                snapshot="task-variations-v1", original_sources=list(originals), sampling=sampling,
            )
            if public_pretraining:
                pretrain_data["public_pretraining"] = True
            if "compositions" in data:
                compositions = dict(expect_dict(data["compositions"], "dataset.pretrain_data.compositions"))
                reject_unknown_keys(compositions, {"archive_path", "size", "sha256"}, "compositions")
                if not public_pretraining:
                    raise ValueError("compositions currently require public_pretraining")
                path = Path(compositions.get("archive_path", ""))
                if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in str(path)
                        or path.suffix != ".zip" or path.parts[0] != "assets_tmp"):
                    raise ValueError("compositions.archive_path must be an assets_tmp relative ZIP path")
                if type(compositions.get("size")) is not int or compositions["size"] <= 0:
                    raise ValueError("compositions.size must be a positive integer")
                if not isinstance(compositions.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", compositions["sha256"]):
                    raise ValueError("compositions.sha256 must be a SHA-256 digest")
                pretrain_data["compositions"] = compositions
        return merge_dicts(DEFAULT_DATASET, {
            "style": style,
            "dataset": dataset_name,
            "evaluation": evaluation,
            "add_conceptarc": False,
            "add_filtered": False,
            "add_rearc": None,
            "add_nvarc_training": False,
            "add_nvarc_full": False,
            "pretrain_data": pretrain_data,
        })
    if "evaluation" in raw_dataset_cfg:
        raise ValueError("dataset.evaluation requires dataset.pretrain_data")
    if "prefinesub" in raw_dataset_cfg and not isinstance(raw_dataset_cfg["prefinesub"], bool):
        raise ValueError("dataset.prefinesub must be a boolean.")
    if raw_dataset_cfg.get("prefinesub", False):
        disallowed = sorted(key for key in raw_dataset_cfg if key != "prefinesub")
        if disallowed:
            raise ValueError(
                "dataset.prefinesub cannot be combined with "
                + ", ".join(f"dataset.{key}" for key in disallowed)
                + "."
            )
        return merge_dicts(
            DEFAULT_DATASET,
            {
                "prefinesub": True,
                "style": "pretrain",
                "dataset": "arc2",
                "add_conceptarc": True,
                "add_filtered": True,
                "add_rearc": DEFAULT_REARC_SAMPLE_COUNT,
                "add_nvarc_training": True,
                "add_nvarc_full": True,
            },
        )
    if raw_dataset_cfg.get("dataset") == "conceptarc":
        dataset_cfg = merge_dicts(
            DEFAULT_DATASET,
            {
                "style": "pretrain",
                "dataset": "conceptarc",
                "add_conceptarc": False,
                "keep_train_test_separate": False,
                "add_filtered": False,
                "add_rearc": None,
                "add_nvarc_training": False,
                "add_nvarc_full": False,
            },
            raw_dataset_cfg,
        )
    else:
        dataset_cfg = merge_dicts(DEFAULT_DATASET, raw_dataset_cfg)
    if dataset_cfg["style"] not in {"pretrain", "finetune"}:
        raise ValueError("dataset.style must be 'pretrain' or 'finetune'.")
    if dataset_cfg["dataset"] not in {"arc1", "arc2", "conceptarc"}:
        raise ValueError("dataset.dataset must be 'arc1', 'arc2', or 'conceptarc'.")
    for key in (
        "prefinesub",
        "add_conceptarc",
        "keep_train_test_separate",
        "add_filtered",
        "add_nvarc_training",
        "add_nvarc_full",
    ):
        if not isinstance(dataset_cfg[key], bool):
            raise ValueError(f"dataset.{key} must be a boolean.")
    if dataset_cfg["dataset"] == "conceptarc":
        if dataset_cfg["style"] != "pretrain":
            raise ValueError("dataset.dataset='conceptarc' requires dataset.style='pretrain'.")
        if not dataset_cfg["keep_train_test_separate"]:
            raise ValueError(
                "dataset.dataset='conceptarc' requires "
                "dataset.keep_train_test_separate=true."
            )
        enabled_addons = [
            key
            for key in (
                "add_conceptarc",
                "add_filtered",
                "add_nvarc_training",
                "add_nvarc_full",
            )
            if dataset_cfg[key]
        ]
        if dataset_cfg["add_rearc"] is not None:
            enabled_addons.append("add_rearc")
        if enabled_addons:
            raise ValueError(
                "dataset.dataset='conceptarc' is a ConceptARC-only mode and cannot "
                "enable " + ", ".join(f"dataset.{key}" for key in enabled_addons) + "."
            )
    elif dataset_cfg["keep_train_test_separate"]:
        raise ValueError(
            "dataset.keep_train_test_separate=true is only valid when "
            "dataset.dataset='conceptarc'."
        )
    return dataset_cfg


def build_plan(spec: dict[str, Any], spec_path: Path) -> dict[str, Any]:
    reject_unknown_keys(spec, TOP_LEVEL_KEYS, "spec")
    if not isinstance(spec.get("validate_cuda", False), bool):
        raise ValueError("validate_cuda must be boolean")
    distributed = expect_dict(spec.get("distributed"), "distributed")
    reject_unknown_keys(distributed, {"nproc_per_node"}, "distributed")
    nproc = distributed.get("nproc_per_node", 1)
    if type(nproc) is not int or nproc < 1:
        raise ValueError("distributed.nproc_per_node must be a positive integer")
    if nproc > 1 and "training_benchmark" in spec:
        raise ValueError("training_benchmark measures a single process; disable DDP for this mode")
    evaluation_budget_benchmark = None
    if "evaluation_budget_benchmark" in spec:
        from src.evaluation_budget_benchmark import validate_config as validate_budget_config

        evaluation_budget_benchmark = validate_budget_config(spec["evaluation_budget_benchmark"])
        if (not spec.get("eval_only") or not spec.get("checkpoint")
                or spec.get("enable_finetune", True)
                or any(key in spec for key in ("training_benchmark", "backend_benchmark", "speculative_benchmark"))):
            raise ValueError("evaluation_budget_benchmark requires eval_only, a checkpoint, and enable_finetune=false; benchmark modes cannot be combined.")
    speculative_benchmark = None
    if "speculative_benchmark" in spec:
        from src.speculative_benchmark import validate_config as validate_speculative_config

        speculative_benchmark = validate_speculative_config(spec["speculative_benchmark"])
        if (not spec.get("eval_only") or not spec.get("checkpoint")
                or spec.get("enable_finetune", True) or "training_benchmark" in spec or "backend_benchmark" in spec):
            raise ValueError("speculative_benchmark requires eval_only, a checkpoint, and enable_finetune=false; benchmark modes cannot be combined.")
    backend_benchmark = None
    if "backend_benchmark" in spec:
        from src.backend_benchmark import validate_config as validate_backend_config

        backend_benchmark = validate_backend_config(spec["backend_benchmark"])
        if (not spec.get("eval_only") or not spec.get("checkpoint")
                or spec.get("enable_finetune", True) or "training_benchmark" in spec):
            raise ValueError("backend_benchmark requires eval_only, a checkpoint, and enable_finetune=false.")
    benchmark = None
    if "training_benchmark" in spec:
        from src.training_benchmark import validate_config

        benchmark = validate_config(spec["training_benchmark"])
        if spec.get("checkpoint") or spec.get("eval_only") or not spec.get("skip_evals") or spec.get("enable_finetune", True):
            raise ValueError("training_benchmark requires fresh training, skip_evals=true and enable_finetune=false.")
        if spec.get("finetunes") or spec.get("evals"):
            raise ValueError("training_benchmark cannot include finetunes or evaluations.")
    run_name = str(spec.get("run_name") or spec_path.stem)
    run_dir = runs_root(REPO_ROOT) / run_name
    eval_only = bool(spec.get("eval_only", False))
    finetune_only = spec.get("finetune_only", False)
    if type(finetune_only) is not bool:
        raise ValueError("finetune_only must be boolean")
    if finetune_only and eval_only:
        raise ValueError("finetune_only cannot be combined with eval_only")
    finetune_resume = spec.get("finetune_resume", False)
    if type(finetune_resume) is not bool or (finetune_resume and not finetune_only):
        raise ValueError("finetune_resume must be boolean and requires finetune_only=true")
    dataset_cfg = build_dataset_config(expect_dict(spec.get("dataset"), "dataset"))
    skip_evals = bool(spec.get("skip_evals", False))
    if dataset_cfg.get("pretrain_data", {}).get("public_pretraining") and not skip_evals:
        raise ValueError("Public-answer pretraining requires skip_evals=true; public eval is part of training")
    if dataset_cfg.get("pretrain_data", {}).get("mode") == "full" and (not skip_evals or spec.get("enable_finetune", True)):
        raise ValueError("Full family data requires skip_evals=true and enable_finetune=false; public answers belong to training")
    standalone_checkpoint = resolve_input_checkpoint(spec.get("checkpoint"), run_dir)
    if eval_only and standalone_checkpoint is None:
        raise ValueError("eval_only requires a top-level checkpoint path.")
    if eval_only and skip_evals:
        raise ValueError("eval_only cannot be combined with skip_evals.")
    enable_finetune = bool(
        spec.get("enable_finetune", dataset_cfg["style"] == "finetune")
    )
    if enable_finetune and dataset_cfg["style"] != "finetune":
        raise ValueError("enable_finetune requires dataset.style='finetune'.")
    if finetune_only and (not enable_finetune or standalone_checkpoint is None):
        raise ValueError("finetune_only requires a checkpoint and enable_finetune=true")
    if skip_evals and spec.get("evals") not in (None, {}):
        raise ValueError("skip_evals cannot be combined with evals.")

    task_data_mode = "pretrain_data" in dataset_cfg
    variation_mode = task_data_mode and dataset_cfg["pretrain_data"]["kind"] in {"task_variations", "family_sampled"}
    assets_root = run_dir / "data" if task_data_mode else REPO_ROOT / "assets"
    assets = {
        "train_data_path": assets_root / "challenges.json",
        "rearc_data_path": (
            assets_root / "rearc_challenges.json"
            if dataset_cfg["add_rearc"] is not None
            else None
        ),
        "nvarc_training_path": (
            assets_root / "nvarc_training"
            if dataset_cfg["add_nvarc_training"]
            else None
        ),
        "nvarc_full_path": (
            assets_root / "nvarc_full"
            if dataset_cfg["add_nvarc_full"]
            else None
        ),
        "private_data_path": assets_root / "priv_challenges.json",
        "evaluation_data_path": assets_root / (
            "priv_challenges.json" if task_data_mode or dataset_cfg["style"] == "finetune" else "challenges.json"
        ),
        "solutions_path": assets_root / "solutions.json",
    }
    eval_groups = {}
    if task_data_mode and dataset_cfg["evaluation"].get("split_manifest"):
        from dataset_building_scripts.task_splits import load_parent_split

        manifest, _ = load_parent_split(REPO_ROOT, dataset_cfg["evaluation"]["split_manifest"])
        eval_groups = {
            group: {
                "data_path": assets_root / f"eval_{group}_challenges.json",
                "solutions_path": assets_root / f"eval_{group}_solutions.json",
            }
            for group in manifest["eval_groups"]
        }
    assets["evaluation_groups"] = eval_groups

    top_level_attention = {}
    if "FA" in spec:
        top_level_attention["FA"] = normalize_fa_value(spec["FA"], "FA")
    shared = merge_dicts(
        BASE_ARGS,
        top_level_attention,
        training_overrides(spec.get("defaults"), "defaults"),
    )
    normalize_fa_config(shared, "defaults")
    speculative_explicit = "inference_speculative" in (spec.get("defaults") or {})
    default_speculative_off_when_sampling(shared, speculative_explicit)

    pretrain_dir = stage_dir(run_dir, "pretrain")
    pretrain = merge_dicts(
        shared,
        training_overrides(spec.get("pretrain"), "pretrain"),
        {
            "name": f"{run_name}/pretrain",
            "data_path": assets["train_data_path"],
            **({"variation_index_path": assets_root / "variation_index.json"} if variation_mode else {}),
            **({"dataset_manifest_path": assets_root / "dataset_manifest.json"} if task_data_mode and not variation_mode else {}),
            "rearc_data_path": assets["rearc_data_path"],
            "nvarc_training_path": assets["nvarc_training_path"],
            "nvarc_full_path": assets["nvarc_full_path"],
            "checkpoint_path": standalone_checkpoint,
            "checkpoint_mode": "resume",
            # A resumed checkpoint trained without scale keeps its recipe unless the spec sets the flag.
            "scale_augmentation_explicit": any("scale_augmentation" in (spec.get(key) or {})
                                               for key in ("defaults", "pretrain")),
            "train_log_file": pretrain_dir / "training_log.txt",
            "save_path": pretrain_dir / "tiny.pt",
        },
    )
    if pretrain.get("reset_training_state", False):
        pretrain["checkpoint_mode"] = "finetune"
    normalize_fa_config(pretrain, "pretrain")
    validate_research_options(pretrain, n_layers=pretrain["n_layers"])
    if standalone_checkpoint is None and (pretrain.get("loop_iterations") or 1) > 1:
        if pretrain.get("loop_start_layer") is None or pretrain.get("loop_end_layer") is None:
            raise ValueError("Fresh looping requires explicit loop_start_layer and loop_end_layer.")

    finetune_defaults = merge_dicts(
        shared,
        training_overrides(spec.get("finetune_defaults"), "finetune_defaults"),
    )
    normalize_fa_config(finetune_defaults, "finetune_defaults")
    raw_finetunes = expect_dict(spec.get("finetunes"), "finetunes")
    if raw_finetunes and not enable_finetune:
        raise ValueError("finetunes were provided but enable_finetune is false.")
    if enable_finetune and not raw_finetunes:
        raw_finetunes = {"default": {}}

    finetunes: dict[str, dict[str, Any]] = {}
    for name, overrides in raw_finetunes.items():
        stage_overrides = training_overrides(overrides, f"finetunes.{name}")
        current_dir = stage_dir(run_dir, "finetune", name)
        finetunes[name] = merge_dicts(
            finetune_defaults,
            stage_overrides,
            {
                "name": f"{run_name}/finetunes/{name}",
                "data_path": assets["private_data_path"],
                "rearc_data_path": None,
                "nvarc_training_path": None,
                "nvarc_full_path": None,
                "checkpoint_path": standalone_checkpoint if finetune_only else pretrain["save_path"],
                "checkpoint_mode": "resume" if finetune_resume else "finetune",
                # Fine-tuning retains historical embedding rows while its
                # examples come only from the selected adaptation subset.
                **({"allow_missing_tasks": True} if finetune_resume else {}),
                "train_log_file": current_dir / "training_log.txt",
                "save_path": current_dir / "tiny.pt",
            },
        )
        normalize_fa_config(finetunes[name], f"finetunes.{name}")
        validate_research_options(finetunes[name], n_layers=finetunes[name]["n_layers"])

    eval_defaults = merge_dicts(
        {
            "mode": "full-decode",
            "batch_size": 120,
            "max_augments": shared["max_augments"],
            "enable_aug": shared["enable_aug"],
            "augmentation_sampling_mode": shared["augmentation_sampling_mode"],
            "seed": shared["seed"],
            "inference_temperature": shared["inference_temperature"],
            "inference_top_k": shared["inference_top_k"],
            "inference_speculative": shared["inference_speculative"],
            "allow_missing_tasks": True,
            "extend_checkpoint_task_ids": True,
        },
        training_overrides(
            spec.get("eval_defaults"),
            "eval_defaults",
            allowed_keys=EVAL_CONFIG_KEYS,
        ),
    )
    normalize_fa_config(eval_defaults, "eval_defaults")
    speculative_explicit = speculative_explicit or "inference_speculative" in (spec.get("eval_defaults") or {})
    default_speculative_off_when_sampling(eval_defaults, speculative_explicit)
    validate_eval_mode(eval_defaults, "eval_defaults")
    raw_evals = {} if skip_evals else expect_dict(spec.get("evals"), "evals")
    if not raw_evals and not skip_evals:
        if eval_only:
            raw_evals = {"final": {"checkpoint": "checkpoint"}}
        elif dataset_cfg["prefinesub"]:
            raw_evals = {}
        else:
            raw_evals = (
                {name: {"checkpoint": f"finetune.{name}.final"} for name in finetunes}
                or {"final": {"checkpoint": "pretrain.final"}}
            )
        if eval_groups:
            raw_evals = {
                f"{name}_{group}": {**cfg, "group": group}
                for name, cfg in raw_evals.items() for group in eval_groups
            }

    evals: dict[str, dict[str, Any]] = {}
    for name, overrides in raw_evals.items():
        eval_overrides = training_overrides(
            overrides,
            f"evals.{name}",
            allowed_keys=EVAL_CONFIG_KEYS,
        )
        checkpoint_ref = eval_overrides.get("checkpoint") or eval_overrides.get("from")
        if not checkpoint_ref:
            raise ValueError(f"evals.{name} must define checkpoint.")
        evals[name] = merge_dicts(
            eval_defaults,
            eval_overrides,
            {
                "checkpoint": checkpoint_ref,
                "run_name": f"{run_name}/evals/{name}",
            },
        )
        group = evals[name].get("group", "all")
        if not isinstance(group, str) or group not in {"all", *eval_groups}:
            raise ValueError(f"evals.{name}.group is not a configured evaluation group: {group!r}")
        evals[name].update(
            group=group,
            **(eval_groups[group] if group != "all" else {
                "data_path": assets["evaluation_data_path"],
                "solutions_path": assets["solutions_path"],
            }),
        )
        normalize_fa_config(evals[name], f"evals.{name}")
        default_speculative_off_when_sampling(
            evals[name], speculative_explicit or "inference_speculative" in (overrides or {}))
        validate_eval_mode(evals[name], f"evals.{name}")
        evals[name].pop("from", None)

    plan = {
        "distributed": {"nproc_per_node": nproc},
        "run_name": run_name,
        "run_dir": run_dir,
        "spec_path": spec_path,
        "eval_only": eval_only,
        "finetune_only": finetune_only,
        "finetune_resume": finetune_resume,
        "validate_cuda": spec.get("validate_cuda", False),
        "skip_data_build": bool(spec.get("skip_data_build", False)),
        "skip_evals": skip_evals,
        "score_results": bool(
            spec.get("score_results", False if dataset_cfg["prefinesub"] else SCORE_RESULTS)
        ),
        "visualize": bool(spec.get("visualize", VISUALIZE)),
        "dataset": dataset_cfg,
        "assets": assets,
        "shared": shared,
        "pretrain": pretrain,
        "finetunes": finetunes,
        "evals": evals,
    }
    if nproc > 1 and not eval_only:
        for stage in (pretrain, *finetunes.values()):
            if stage["device"] != "cuda":
                raise ValueError("DDP requires device='cuda'; each worker selects its local GPU")
    for eval_cfg in evals.values():
        resolve_checkpoint_ref(str(eval_cfg["checkpoint"]), plan)
    if benchmark is not None:
        if pretrain["gradient_accumulation_steps"] != 1:
            raise ValueError("training_benchmark measures physical batches; set gradient_accumulation_steps=1.")
        if pretrain.get("dihedral_encoding") not in (None, "residual"):
            raise ValueError("training_benchmark requires injected dihedral embeddings.")
        plan["training_benchmark"] = benchmark
    if backend_benchmark is not None:
        plan["backend_benchmark"] = backend_benchmark
    if speculative_benchmark is not None:
        plan["speculative_benchmark"] = speculative_benchmark
    if evaluation_budget_benchmark is not None:
        plan["evaluation_budget_benchmark"] = evaluation_budget_benchmark
    return plan


def preview_plan(plan: dict[str, Any]) -> dict[str, Any]:
    preview = {
        "distributed": plan["distributed"],
        "run_name": plan["run_name"],
        "eval_only": plan["eval_only"],
        "skip_data_build": plan["skip_data_build"],
        "skip_evals": plan["skip_evals"],
        "score_results": plan["score_results"],
        "visualize": plan["visualize"],
        "timing_path": plan["run_dir"] / "timing.txt",
        "dataset": plan["dataset"],
        "pretrain": (
            None if plan["eval_only"] or plan.get("finetune_only") else stage_summary(plan["pretrain"])
        ),
        "finetunes": (
            {}
            if plan["eval_only"]
            else {
                name: stage_summary(stage)
                for name, stage in plan["finetunes"].items()
            }
        ),
        "evals": {
            name: {
                "checkpoint": stage["checkpoint"],
                "mode": stage["mode"],
                "group": stage["group"],
                "data_path": stage["data_path"],
                "solutions_path": stage["solutions_path"],
                **({"FA": stage["FA"]} if "FA" in stage else {}),
                "max_augments": stage["max_augments"],
                "batch_size": stage["batch_size"],
                **(
                    {"accuracy_path": stage_dir(plan["run_dir"], "eval", name) / "accuracy.json"}
                    if stage["mode"] == "accuracy-only"
                    else {"rescore_path": stage_dir(plan["run_dir"], "eval", name) / "rescore.json"}
                    if stage["mode"] == "rescore"
                    else {
                        "submission_path": stage_dir(plan["run_dir"], "eval", name) / "submission.json",
                        "score_path": stage_dir(plan["run_dir"], "eval", name) / "score.json",
                    }
                ),
            }
            for name, stage in plan["evals"].items()
        },
    }
    if plan["eval_only"]:
        preview["source_checkpoint"] = plan["pretrain"]["checkpoint_path"]
    if plan.get("training_benchmark"):
        preview["training_benchmark"] = plan["training_benchmark"]
    if plan.get("backend_benchmark"):
        preview["backend_benchmark"] = plan["backend_benchmark"]
    if plan.get("speculative_benchmark"):
        preview["speculative_benchmark"] = plan["speculative_benchmark"]
    if plan.get("evaluation_budget_benchmark"):
        preview["evaluation_budget_benchmark"] = plan["evaluation_budget_benchmark"]
    return preview


def run_training_stage_distributed(plan: dict[str, Any], stage: dict[str, Any], label: str) -> None:
    from distributed_training import training_process

    with training_process(plan.get("distributed", {}).get("nproc_per_node", 1)):
        run_training_stage(stage, label)


def run_training_stage(stage: dict[str, Any], label: str) -> None:
    load_runtime_modules()
    Path(stage["save_path"]).parent.mkdir(parents=True, exist_ok=True)
    cfg = argparse.Namespace(**stage)
    print(f"Building {label}...")
    model, dataset, dataloader, device, data_path = build.build_model_and_data(cfg)
    print(f"Starting {label}...")
    t_start = perf_counter()
    train.train_model(
        cfg,
        model=model,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        data_path=data_path,
    )
    print(f"{label} finished in {perf_counter() - t_start:.2f}s")


def run_training_stage_isolated(plan: dict[str, Any], stage_name: str) -> None:
    """Run one training stage in a process whose CUDA context can fully exit."""
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--config",
        str(plan["spec_path"]),
        "--train-stage",
        stage_name,
    ]
    nproc = plan.get("distributed", {}).get("nproc_per_node", 1)
    if nproc > 1:
        # Single-node DDP: torchrun starts one worker per GPU (src/distributed_training.py).
        command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                   "--nnodes=1", f"--nproc-per-node={nproc}", "--max-restarts=0", *command[2:]]
    print(
        f"Starting training stage {stage_name!r} in an isolated process...",
        flush=True,
    )
    # Packed sequences change length every micro-batch, which fragments the caching
    # allocator (a d384 x6 ReARC run reserved 125-139 GiB for 56 GiB live). Expandable
    # segments let the allocator grow mappings instead of hoarding whole blocks; an
    # explicit PYTORCH_CUDA_ALLOC_CONF in the environment wins.
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def run_eval_stage(plan: dict[str, Any], name: str, eval_cfg: dict[str, Any]) -> None:
    load_runtime_modules()
    checkpoint_path = resolve_checkpoint_ref(str(eval_cfg["checkpoint"]), plan)
    data_path = eval_cfg.get("data_path", plan["assets"]["evaluation_data_path"])
    solutions_path = eval_cfg.get("solutions_path", plan["assets"]["solutions_path"])
    cfg = argparse.Namespace(
        **merge_dicts(
            plan["shared"],
            eval_cfg,
            {
                "data_path": data_path,
                "checkpoint_path": checkpoint_path,
                "solutions_path": solutions_path,
            },
        )
    )
    result = evaluate.run_evaluation(
        cfg,
        run_name=str(eval_cfg["run_name"]),
        max_augments=int(eval_cfg["max_augments"]),
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        batch_size=int(eval_cfg["batch_size"]),
        splits=["test"],
        task_ids=None,
        timing_path=plan["run_dir"] / "timing.txt",
    )
    if eval_cfg["mode"] == "accuracy-only":
        print(f"Evaluation complete for {name}. Accuracy saved to {result[2]}.")
        return
    if eval_cfg["mode"] == "rescore":
        print(f"Evaluation complete for {name}. Rescore report saved to {result[2]}.")
        return
    submission_path = result[2]
    print(f"Evaluation complete for {name}. submission.json generated.")
    if plan["score_results"]:
        score_result = utils.score_arc_submission(solutions_path, submission_path)
        write_score_summary(
            submission_path,
            score_result,
            checkpoint_ref=str(eval_cfg["checkpoint"]),
            run_name=str(plan["run_name"]),
            group=eval_cfg.get("group", "all"),
        )
        if eval_cfg.get("reference_submission"):
            from evaluation_comparison import write_comparison
            write_comparison(REPO_ROOT / eval_cfg["reference_submission"], submission_path, solutions_path, data_path)
    if plan["visualize"]:
        if plan["score_results"]:
            utils.visualize_submissions(
                submission_path,
                solutions_path,
                mode="!",
            )
        else:
            utils.visualize_submissions(submission_path, mode="submission")


def run_eval_stage_isolated(plan: dict[str, Any], name: str) -> None:
    """Run evaluation with compiler/CUDA state independent from training."""
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--config",
        str(plan["spec_path"]),
        "--eval-stage",
        name,
    ]
    print(
        f"Starting evaluation {name!r} in a fresh process "
        "to isolate compiler state...",
        flush=True,
    )
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def prepare_data(plan: dict[str, Any]) -> None:
    if plan["skip_data_build"]:
        raise ValueError("The selected spec skips data building")
    if "pretrain_data" in plan["dataset"]:
        if plan["dataset"]["pretrain_data"]["kind"] == "family_sampled":
            from dataset_building_scripts.family_data import build_family_dataset
            result = build_family_dataset(plan["dataset"], plan["run_dir"] / "data", root=REPO_ROOT)
            print(f"Built family data: {result}")
            return
        from dataset_building_scripts.build_variation_dataset import build_original_dataset, build_variation_dataset
        ensure_grouped_assets(plan["dataset"])
        builder = build_variation_dataset if plan["dataset"]["pretrain_data"]["kind"] == "task_variations" else build_original_dataset
        result = builder(plan["dataset"], plan["run_dir"] / "data")
        print(f"Built task data: {result}")
    else:
        build_dataset(plan["dataset"])


def validate_run_split(plan: dict[str, Any]) -> None:
    """Keep cached data tied to the split requested by the current spec."""
    family_data = plan.get("dataset", {}).get("pretrain_data", {})
    if family_data.get("kind") == "family_sampled":
        from dataset_building_scripts.family_data import sha, encoded
        index_path = plan["pretrain"]["variation_index_path"]
        index = load_spec(index_path)
        catalog = load_spec(index_path.with_name("family_catalog.json"))
        if (index.get("family_catalog") != family_data.get("catalog")
                or index.get("catalog_sha256") != sha(encoded(catalog))
                or any(catalog[key] != family_data[key] for key in ("mode", "sources", "sampling", "discovery"))):
            raise ValueError("Cached family data differs from the pinned spec; use a new run for a refreshed catalog")
    evaluation = plan.get("dataset", {}).get("evaluation", {})
    split_path = evaluation.get("split_manifest")
    if split_path is None:
        return
    from dataset_building_scripts.task_splits import load_parent_split

    _, identity = load_parent_split(REPO_ROOT, split_path)
    index_path = plan["pretrain"].get("variation_index_path") or plan["pretrain"]["dataset_manifest_path"]
    if load_spec(index_path).get("parent_split") != identity:
        raise ValueError("Run data parent split does not match the current spec; rebuild data for this split")
    copied_manifest = index_path.with_name("split_manifest.json")
    if hashlib.sha256(copied_manifest.read_bytes()).hexdigest() != identity["sha256"]:
        raise ValueError("Run data split manifest does not match its recorded identity")


def execute_plan(plan: dict[str, Any]) -> None:
    print(f"Using experiment spec: {plan['spec_path']}")
    print(f"Resolved run name: {plan['run_name']}")
    if not plan["skip_data_build"]:
        prepare_data(plan)
    validate_run_split(plan)
    if plan.get("validate_cuda", False):
        plan["run_dir"].mkdir(parents=True, exist_ok=True)  # an eval-only run has no folder yet
        subprocess.run([sys.executable, "-u", str(SRC_DIR / "ablation_validation.py"),
                        str(plan["run_dir"] / "cuda_validation.json"),
                        str(plan["pretrain"].get("FA", 2))], check=True, cwd=REPO_ROOT)

    if plan.get("evaluation_budget_benchmark"):
        from src.evaluation_budget_benchmark import run_benchmark

        run_benchmark(plan)
        return

    if plan.get("speculative_benchmark"):
        from src.speculative_benchmark import run_benchmark

        run_benchmark(plan)
        return

    if plan.get("backend_benchmark"):
        from src.backend_benchmark import run_benchmark

        run_benchmark(plan)
        return

    if plan.get("training_benchmark"):
        from src.training_benchmark import run_benchmark

        run_benchmark(plan)
        return

    if plan["eval_only"]:
        print("Skipping training stages (eval_only=true).")
    else:
        if plan.get("finetune_only"):
            print("Skipping pretraining stage (finetune_only=true).")
        else:
            run_training_stage_isolated(plan, "pretrain")
        for name in plan["finetunes"]:
            run_training_stage_isolated(plan, f"finetune:{name}")

    for name in plan["evals"]:
        run_eval_stage_isolated(plan, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local mdlARC experiment spec.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment spec.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the resolved plan and exit without running anything.",
    )
    parser.add_argument(
        "--eval-stage",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--train-stage",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prepare-data", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    update_env_from_file(os.environ, DATASET_BUILD_ENV_PATH)
    drop_current_directory_search_entries(os.environ)
    spec_path = Path(args.config).expanduser().resolve()
    plan = build_plan(load_spec(spec_path), spec_path)
    if args.prepare_data:
        prepare_data(plan)
        return
    if args.train_stage is not None:
        if plan["eval_only"]:
            raise ValueError("An eval-only spec has no training stages.")
        validate_run_split(plan)
        if args.train_stage == "pretrain":
            if plan.get("finetune_only"):
                raise ValueError("A fine-tune-only spec cannot execute pretraining")
            run_training_stage_distributed(plan, plan["pretrain"], "pretrain")
            return
        if args.train_stage.startswith("finetune:"):
            name = args.train_stage.split(":", 1)[1]
            stage = plan["finetunes"].get(name)
            if stage is not None:
                run_training_stage_distributed(plan, stage, f"finetune:{name}")
                return
        raise ValueError(f"Unknown training stage: {args.train_stage!r}")
    if args.eval_stage is not None:
        eval_cfg = plan["evals"].get(args.eval_stage)
        if eval_cfg is None:
            raise ValueError(f"Unknown evaluation stage: {args.eval_stage!r}")
        validate_run_split(plan)
        run_eval_stage(plan, args.eval_stage, eval_cfg)
        return
    if args.preview:
        print(json.dumps(jsonable(preview_plan(plan)), indent=2))
        return
    execute_plan(plan)


if __name__ == "__main__":
    main()
