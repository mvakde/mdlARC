#!/usr/bin/env python3
"""Prepare ARC1 + filtered original grid chains, without models or remote work."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_building_scripts.intermediate_grids import prepare_originals, reserve_diagnostic_pairs, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsl-root", type=Path, default=ROOT.parent / "dsl-v2")
    parser.add_argument("--catalog", type=Path, default=ROOT.parent / "arc-synth-data/data/programs/catalog.json")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-spec", type=Path, required=True)
    parser.add_argument("--base-spec", type=Path,
                        help="Preserve an existing recipe's hyperparameters in the new spec")
    parser.add_argument("--diagnostic-holdouts", type=int, default=0,
                        help="Reserve this many labeled demos and enable supervised target chains; requires --base-spec")
    args = parser.parse_args()
    for path, parent in ((args.package, ROOT / "assets_tmp"), (args.audit_dir, ROOT / "results")):
        if not path.resolve().is_relative_to(parent):
            parser.error(f"Use an ignored destination under {parent}")
    base = args.base_spec or (args.output_spec if args.output_spec.exists() else ROOT / "specs/pretrain_eval.json")
    spec = json.loads(base.read_text())
    if args.diagnostic_holdouts:
        if not args.base_spec or args.diagnostic_holdouts < 0:
            parser.error("Positive --diagnostic-holdouts requires --base-spec with an existing prepared dataset")
        identity = reserve_diagnostic_pairs(spec["dataset"]["prepared"], root=ROOT, package=args.package,
                                           count=args.diagnostic_holdouts, seed=int(spec["defaults"].get("seed", 42)))
    else:
        identity = prepare_originals(root=ROOT, dsl_root=args.dsl_root, catalog_path=args.catalog,
                                    package=args.package, audit_dir=args.audit_dir)
    if not args.base_spec:
        spec["pretrain"].update(epochs=1000)
    spec.update(run_name=f"arc1-filtered-eval-demos-random-intermediates-e{spec['pretrain']['epochs']}-" + identity["sha256"][:8],
                enable_finetune=False)
    spec["dataset"] = {"dataset": "arc1", "style": "pretrain", "add_filtered": True,
                       "add_conceptarc": False, "add_rearc": None, "add_nvarc_training": False,
                       "add_nvarc_full": False, "prefinesub": False, "keep_train_test_separate": False,
                       "prepared": identity}
    spec["defaults"].update(attention_pattern="pair_group", training_demo_schedule="two",
                             max_training_demos=2, pair_identity_vertices=0)
    spec["finetunes"] = {}
    spec["eval_defaults"].update(inference_demo_policy="two")
    spec["evals"] = {"final": {"checkpoint": "pretrain.final"}}
    if args.diagnostic_holdouts:
        spec["defaults"]["predict_intermediate_grids"] = True
        spec["run_name"] = f"arc1-filtered-supervised-intermediates-e{spec['pretrain']['epochs']}-" + identity["sha256"][:8]
        spec["evals"]["intermediate_diagnostic"] = {"checkpoint": "pretrain.final", "mode": "intermediate-diagnostic",
            "enable_aug": False, "max_augments": 1, "diagnostic_cases": args.diagnostic_holdouts, "batch_size": 32}
    args.output_spec.parent.mkdir(parents=True, exist_ok=True)
    args.output_spec.write_bytes(encoded(spec))
    print(json.dumps({"prepared_spec": str(args.output_spec)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
