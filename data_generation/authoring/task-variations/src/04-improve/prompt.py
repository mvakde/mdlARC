"""Build a concise instruction plus an exact per-puzzle workspace payload."""

from __future__ import annotations

import hashlib
import json
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from task_variations.generate.prompt import format_puzzle
from task_variations.generate.json_storage import bound_bytes, evidence_bytes

from .config import RunnerConfig
from .data import RegenerationSource, load_regeneration_source
from .host_checks import validate_policy
from .workspace_render import source_svg


PLACEHOLDERS = {
    "{TASK_JSON}": "task",
    "{SEVEN_SECTION_SUMMARY}": "summary",
    "{CURRENT_GENERATOR_CODE}": "code",
    "{HUMAN_REVIEW_INSTRUCTIONS}": "review",
}

WORKSPACE_AGENTS = """# Reviewed ARC generator workspace

- Work only in this isolated Git repository; do not inspect parent or sibling directories.
- All JSON and JSONL you create, including generated code, must be compact with no indentation. Use `json.dumps(value, separators=(",", ":"))` in Python and `JSON.stringify(value)` in JavaScript.
- Bound disk use: estimate bytes, file counts and peak temporary space across tasks, seeds, pairs, steps, workers, retries and retained runs. Keep scratch diagnostics and logs bounded; avoid duplicate reconstructible payloads. Preserve required evidence and hashes. Use the existing lossless trace codec for saved execution traces, keeping initial and exceptional before states. Generated code follows these rules too.
- Before storage optimization edits, report measured projected net savings, runtime and peak disk cost, including journals, indexes and backups.
- Read `task.json`, `source_task.txt`, `source_task.svg`, `summary.xml`, `review.md`, and `requirements.md` before editing.
- Edit `generator.py` and use `scratch/` for experiments, review coverage notes, and renderings. Do not alter other files or create symlinks.
- Read `acceptance.json` for protected, measurable review criteria. Optional `reference_checks.py` supplies independent task features and a reference solver; it is not editable.
- Inspect `reviewed_samples.json` for original generated examples and their seeds. Use `python render.py reviewed_samples.json --output scratch/reviewed.png` to view them.
- Use `python render.py --seed 123 --output scratch/candidate.png` to inspect your own samples; source rendering is `python render.py task.json --output scratch/source.png`.
- Use normal file-editing tools and preserve correct parts of the reviewed generator when useful.
- Run `python verify.py` after editing and continue until it passes.
- `environment.json` records the host Python executable and NumPy version. Use that exact executable for verification and rendering if your shell's `python` is different; the host environment already contains NumPy.
- Inspect the per-puzzle and cross-seed distributions in `scratch/verification.json`. Map every review request to its measurements, tests, or a concrete visual check in `scratch/review_coverage.md`. Report remaining uncertainty explicitly.
- A pass without a configured reference solver establishes structural validity, not semantic correctness. Independently infer test outputs from training pairs and the test input only; never consult test labels, seeds, or hidden generation state for this check.
- Do not commit changes. The host records the diff from the staged baseline.
- In the final response, summarize the implementation and checks; do not paste the source code.
"""

WORKSPACE_GITIGNORE = """__pycache__/
*.py[cod]
scratch/
"""

FILE_REFERENCES = {
    "{TASK_JSON}": "Read the original solved ARC task in `task.json`, its readable grids in `source_task.txt`, and its rendering in `source_task.svg`.",
    "{SEVEN_SECTION_SUMMARY}": "Read the complete seven-section interpretation in `summary.xml`.",
    "{CURRENT_GENERATOR_CODE}": "The reviewed implementation is `generator.py`; edit that file in place.",
    "{HUMAN_REVIEW_INSTRUCTIONS}": "Read the complete human review in `review.md`.",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptBuild:
    prompt_text: str
    prompt_sha256: str
    prompt_version: str
    template_sha256: str
    workspace_files: Dict[str, str]
    workspace_file_sha256: Dict[str, str]
    workspace_manifest_sha256: str
    source: RegenerationSource

    @property
    def target(self):
        return self.source.target

    @property
    def summary(self):
        return self.source.summary


def _module_source(name: str) -> str:
    path = Path(__file__).with_name(name)
    return path.read_text(encoding="utf-8")


def _acceptance_files(config, source):
    policy = {"schema_version": 1, "require_reference_solver": False, "criteria": [],
              "note": "No protected task-specific criteria configured. Review coverage and semantic correctness still require inspection."}
    files = {}
    if config.acceptance_root is not None:
        prefix = config.acceptance_root / source.target.source_split / source.target.puzzle_id
        policy_path = prefix.with_suffix(".json")
        if policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        checks_path = prefix.with_suffix(".py")
        if checks_path.exists():
            files["reference_checks.py"] = checks_path.read_text(encoding="utf-8")
    validate_policy(policy)
    if policy.get("require_reference_solver") and "reference_checks.py" not in files:
        raise ValueError("Acceptance policy requires a reference_checks.py module")
    files["acceptance.json"] = json.dumps(policy, separators=(",", ":"), sort_keys=True) + "\n"
    return files


def _reviewed_samples(config, source):
    seeds = source.generator.sampling_context["completed_seeds"]
    # Include explicit seed references from prose when available, then spread
    # a small deterministic sample across the completed collection.
    import re
    mentioned = [int(v) for v in re.findall(r"\bseed\s*[-:=#]?\s*(\d+)\b", source.review.notes, re.I)]
    spread = [seeds[0], seeds[len(seeds)//2], seeds[-1]] if seeds else []
    selected = list(dict.fromkeys([s for s in mentioned if s in seeds] + spread))[:6]
    samples = []
    for seed in selected:
        path = source.generator.manifest_path.parent / ("seed-%010d.json" % seed)
        binding = (source.review.source_binding or {}).get("sample_sha256", {})
        expected = binding.get(path.name)
        raw = bound_bytes(path, expected) if expected else evidence_bytes(path)
        task = json.loads(raw)
        # Validate evidence using the existing ARC schema checker.
        from task_variations.generate.visualization import _validate_arc_json
        _validate_arc_json(task)
        samples.append({"seed": seed, "sha256": hashlib.sha256(raw).hexdigest(), "task": task})
    sampling = dict(source.generator.sampling_context["sampling"])
    # Cache reuse describes how samples were loaded, not their evidence or settings.
    sampling.pop("self_test_mode", None)
    return json.dumps({"generator_sha256": source.generator.code_sha256,
                       "sampling": sampling, "samples": samples,
                       "sampling_failures": source.generator.sampling_context.get("failures", [])},
                      separators=(",", ":"), sort_keys=True) + "\n"



def build_prompt(
    config: RunnerConfig,
    puzzle_id: str,
    source_split: str = "auto",
) -> PromptBuild:
    source = load_regeneration_source(config, puzzle_id, source_split)
    try:
        template_raw = config.prompt_template_path.read_bytes()
        template = template_raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Could not read prompt template %s: %s" % (config.prompt_template_path, exc)) from exc
    for placeholder in PLACEHOLDERS:
        if template.count(placeholder) != 1:
            raise ValueError("Prompt template must contain %s exactly once" % placeholder)

    requirements = template
    for placeholder, reference in FILE_REFERENCES.items():
        requirements = requirements.replace(placeholder, reference)
    if not requirements.endswith("\n"):
        requirements += "\n"
    if source.review.author == "automated":
        requirements = requirements.replace("A human then inspected", "An automated reviewer then inspected")
        requirements = requirements.replace("human review", "automated review").replace("human instruction", "review instruction")
        requirements = requirements.replace(
            "The automated review is the authoritative description of what was missing or incorrect in the generated samples.",
            "The automated review proposes corrections and extensions grounded in recorded evidence. Recheck its claims against the original task; it has not been approved by a human.")
        requirements = "Review origin: automated (not human-approved).\n\n" + requirements

    content_files = {
        ".gitignore": WORKSPACE_GITIGNORE,
        "AGENTS.md": WORKSPACE_AGENTS,
        "generator.py": source.generator.code_text.rstrip() + "\n",
        "requirements.md": requirements,
        "review.md": source.review.notes.rstrip() + "\n",
        "source_task.txt": format_puzzle(source.target).rstrip() + "\n",
        "task.json": json.dumps(source.target.data, separators=(",", ":"), sort_keys=True) + "\n",
        "source_task.svg": source_svg(source.target.data),
        "reviewed_samples.json": _reviewed_samples(config, source),
        "environment.json": json.dumps({"python_executable": sys.executable,
                                         "numpy_version": importlib.metadata.version("numpy")},
                                        separators=(",", ":"), sort_keys=True) + "\n",
        "summary.xml": source.summary.text.strip() + "\n",
        "verify.py": _module_source("workspace_verify.py"),
        "host_checks.py": _module_source("host_checks.py"),
        "render.py": _module_source("workspace_render.py"),
        **_acceptance_files(config, source),
    }
    content_hashes = {name: _sha256(text) for name, text in sorted(content_files.items())}
    manifest = {
        "schema_version": 1,
        "prompt_version": config.prompt_version,
        "puzzle_id": source.target.puzzle_id,
        "source_split": source.target.source_split,
        "editable_files": ["generator.py"],
        "scratch_directory": "scratch/",
        "content_file_sha256": content_hashes,
        "prompt_template_sha256": hashlib.sha256(template_raw).hexdigest(),
    }
    manifest_text = json.dumps(manifest, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n"
    workspace_files = dict(content_files)
    workspace_files["workspace_manifest.json"] = manifest_text
    workspace_hashes = {name: _sha256(text) for name, text in sorted(workspace_files.items())}
    manifest_sha256 = workspace_hashes["workspace_manifest.json"]

    prompt_text = """Improve the reviewed ARC puzzle generator in this isolated Git repository.

Read `AGENTS.md`, then inspect the staged task JSON, source rendering, summary, review, reviewed samples, requirements, and acceptance criteria. Edit only `generator.py` as the deliverable; use `scratch/` for analysis, renderings, and review coverage notes. Do not create commits or inspect anything outside this repository. Use normal file-editing tools, run `python verify.py`, and continue until it passes. Inspect actual sample renderings and the measured distributions in `scratch/verification.json`. The host will re-run independent checks with fresh seeds. In your final response, summarize review coverage, measurements, and checks without pasting the program; distinguish reference-solver verification from any remaining semantic uncertainty.

Workspace manifest SHA-256: %s
""" % manifest_sha256

    return PromptBuild(
        prompt_text=prompt_text,
        prompt_sha256=_sha256(prompt_text),
        prompt_version=config.prompt_version,
        template_sha256=hashlib.sha256(template_raw).hexdigest(),
        workspace_files=workspace_files,
        workspace_file_sha256=workspace_hashes,
        workspace_manifest_sha256=manifest_sha256,
        source=source,
    )
