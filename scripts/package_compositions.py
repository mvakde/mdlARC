#!/usr/bin/env python3
"""Package saved, checksum-verified arc-synth-data samples without generating puzzles."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_building_scripts.build_variation_dataset import _solved_pairs


GROUPS = {"arc2/evaluation": "arc2_eval", "arc2/training": "arc2_train",
          "conceptarc/training": "conceptarc", "arc1-filtered/training": "arc1_filtered"}
SEED = re.compile(r"seed-(\d+)\.json\Z")


def selected_samples(synth_root, include_author_completed, pipeline="v3"):
    results = synth_root / "results"
    roots = sorted(results.glob("task-composition-v3*")) if pipeline in ("v3", "all") else []
    if pipeline in ("v2", "all"):
        roots += sorted(results.glob("task-compsition-v2*"))
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        status_paths = sorted(root.glob("*/status.json"))
        if not status_paths:
            continue
        v3 = root.name.startswith("task-composition-v3")
        plan = json.loads((root / "plan.json").read_text())
        dataset = plan.get("dataset", "arc2/evaluation")
        if dataset not in GROUPS:
            raise ValueError(f"Unsupported composition lineage: {dataset}")
        for status_path in status_paths:
            job = status_path.parent
            status = json.loads(status_path.read_text())
            state = status.get("state")
            if state == "completed":
                prefix = status["samples"].rstrip("/") + "/"
                stage = "review-improve" if v3 else "improve"
            elif v3 and state == "author_completed" and include_author_completed:
                prefix, stage = "llm-inputs/author/checked-samples/", "author"
            else:
                continue
            if v3 and status.get("dataset") != dataset:
                raise ValueError(f"Composition status lineage changed: {job}")
            checkpoint = json.loads((job / "checkpoints.json").read_text())[stage]
            samples = {name: digest for name, digest in checkpoint.items()
                       if name.startswith(prefix) and SEED.fullmatch(name[len(prefix):])}
            if not samples:
                raise ValueError(f"No checkpointed composition samples: {job}")
            actual = {p.relative_to(job).as_posix() for p in (job / prefix).glob("seed-*.json")}
            if actual != set(samples):
                raise ValueError(f"Composition samples differ from saved checkpoint: {job}")
            for name, expected in sorted(samples.items()):
                path = job / name
                path.resolve().relative_to(job.resolve())
                if path.is_symlink() or job.is_symlink():
                    raise ValueError(f"Linked composition sample: {path}")
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != expected:
                    raise ValueError(f"Composition sample checksum changed: {path}")
                _solved_pairs(json.loads(data), str(path))
                seed = int(SEED.fullmatch(Path(name).name)[1])
                relative = f"task-compositions/{root.name}/{job.name}/seed{seed}.json"
                yield {"path": relative, "composition_id": f"composition/{root.name}/{GROUPS[dataset]}/{job.name}",
                       "source": GROUPS[dataset], "seed": seed, "sha256": digest,
                       "stage": stage, "parents": status["parents"]}, data


def package(synth_root, output, *, include_author_completed=False, pipeline="v3"):
    output = Path(output).resolve()
    relative_output = output.relative_to(ROOT)
    if output.exists():
        raise ValueError("Archive already exists; select a new output name to preserve its identity")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix=".compositions-", dir=output.parent) as temp:
        pending = Path(temp) / "compositions.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for entry, data in selected_samples(Path(synth_root).resolve(), include_author_completed, pipeline):
                entries.append(entry)
                archive.writestr(entry["path"], data)
            if not entries:
                raise ValueError("No eligible saved composition samples were found")
            archive.writestr("composition_manifest.json", json.dumps({"schema_version": 1, "samples": entries}, separators=(",", ":")))
        pending.rename(output)
    digest = hashlib.sha256()
    with output.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"archive_path": relative_output.as_posix(), "size": output.stat().st_size,
            "sha256": digest.hexdigest(), "tasks": len({item["composition_id"] for item in entries}),
            "samples": len(entries), "stages": sorted({item["stage"] for item in entries}),
            "sources": sorted({item["source"] for item in entries})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("v3", "v2", "all"), default="v3")
    parser.add_argument("--include-author-completed", action="store_true",
                        help="Include locally checked v3 author samples, recording their unreviewed stage")
    args = parser.parse_args()
    print(json.dumps(package(args.synth_root, args.output, include_author_completed=args.include_author_completed,
                             pipeline=args.pipeline), indent=2))


if __name__ == "__main__":
    main()
