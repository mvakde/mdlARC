"""The Kaggle d1024 specs load and resolve to the loop-composition-v3 recipe."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import launch_run

SPECS = sorted((ROOT / "specs/paper/kaggle_d1024").glob("*.json"))


@pytest.mark.parametrize("path", SPECS, ids=[path.stem for path in SPECS])
def test_kaggle_d1024_specs_resolve_to_their_recipe(path):
    plan = launch_run.build_plan(launch_run.load_spec(path), path)
    stage = plan["pretrain"]
    assert plan["distributed"] == {"nproc_per_node": 8}
    assert plan["dataset"]["pretrain_data"]["kind"] == "family_sampled"
    assert plan["dataset"]["pretrain_data"]["catalog"]["archive_path"] == (
        "assets_tmp/arc2-family-full-counts-v1-20261024-r1.zip")
    assert stage["FA"] == 4 and stage["attention_dropout"] == 0.0
    assert (stage["d_model"], stage["d_ff"], stage["n_layers"], stage["n_heads"]) == (1024, 2688, 8, 16)
    assert (stage["loop_iterations"], stage["loop_start_layer"], stage["loop_end_layer"]) == (6, 2, 6)
    assert stage["loop_exit_weight"] == 0.3 and stage["n_predict"] == 4
    assert stage["attention_pattern"] == "pair_group" and stage["training_demo_schedule"] == "half_or_more"
    assert stage["max_training_demos"] == 4
    # Keys whose launcher default differs from the code these runs used.
    assert stage["stable_cross_entropy"] is True and stage["compile_scope"] == "model"
    assert not stage["scale_augmentation"] and plan["skip_evals"]
    assert stage["checkpoint_keep_last"] == 3
