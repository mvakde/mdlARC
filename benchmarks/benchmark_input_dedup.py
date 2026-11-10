"""Compare unique-input pool construction/selection with the March implementation.

CPU microbenchmark; excludes model execution and the historical dataset's
up-front eight-geometry tensor/position cache. No network or training runs.
"""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common

MARCH_COMMIT = "42e0a274dda3d75de776e12bbf949c671cdcca52"


def median_seconds(fn, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result = fn()
        samples.append(perf_counter() - start)
    return statistics.median(samples), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets_tmp/ARC-1/arc1_train_both.json")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--selections", type=int, default=20000)
    args = parser.parse_args()
    tasks = dict(list(sorted(json.loads(args.data.read_text()).items()))[:args.tasks])
    with tempfile.TemporaryDirectory(prefix="mdlarc-march-dedup-") as folder:
        source_path = Path(folder) / "march_common.py"
        source_path.write_bytes(subprocess.check_output(
            ["git", "show", f"{MARCH_COMMIT}:src/common.py"], cwd=ROOT,
        ))
        spec = importlib.util.spec_from_file_location("mdlarc_march_common", source_path)
        march = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = march
        spec.loader.exec_module(march)

        old_examples, inputs_by_task, colors, protected = [], {}, {}, {}
        for task_id, task in tasks.items():
            inputs_by_task[task_id] = {}
            colors[task_id] = common.extract_task_colors(task)
            protected[task_id] = common.extract_task_output_only_colors(task)
            for split in ("train", "test"):
                for index, pair in enumerate(task.get(split, [])):
                    inputs_by_task[task_id][(task_id, split, index)] = pair["input"]
                    tokens = [torch.tensor(march.encode_example(
                        march.apply_dihedral_transform(pair["input"], geometry),
                        include_output=False, append_end=False,
                    )) for geometry in range(8)]
                    old_examples.append(SimpleNamespace(
                        task_id=task_id, split=split, pair_index=index,
                        tokens=tokens[0], tokens_by_dihedral=tokens,
                    ))
        # Match the old destination domain, including its fixed black. The
        # second new timing also measures the actual black-can-move policy.
        old_destinations = {task_id: [c for c in range(1, 10) if c not in protected[task_id]]
                            for task_id in tasks}

        def old_build():
            return march.build_augmentor(
                old_examples, old_destinations, max_augments=80,
                enable_color=True, enable_dihedral=True, seed=42,
                color_apply_to_test_split=True, dihedral_apply_to_test_split=True,
            )

        def new_build(match_old):
            pools = {}
            for task_id, inputs in inputs_by_task.items():
                reserved = sorted(set(protected[task_id]) | ({0} if match_old else set()))
                pools.update(common.build_unique_input_augments(
                    inputs, colors[task_id], protected_colors=reserved,
                    max_augments=80, seed=42,
                ))
            return common.Augmentor(pools, max_augments=80)

        old_seconds, old = median_seconds(old_build, args.repeats)
        matched_seconds, matched = median_seconds(lambda: new_build(True), args.repeats)
        new_seconds, new = median_seconds(lambda: new_build(False), args.repeats)
        for task_id, inputs in inputs_by_task.items():
            seen = {common._input_grid_bytes(grid) for grid in inputs.values()}
            for key, grid in inputs.items():
                pool = new.augments_by_key[key]
                geometry_keys = common._input_geometry_bytes(grid)
                assert pool.dihedral_indices[0] == 0
                assert pool.color_maps[pool.color_map_indices[0]].tolist() == list(range(14))
                assert 1 <= len(pool.dihedral_indices) <= 80
                for index, geometry in zip(pool.color_map_indices[1:], pool.dihedral_indices[1:]):
                    mapping = pool.color_maps[index].tolist()
                    assert sorted(mapping[:10]) == list(range(10))
                    assert all(mapping[color] == color for color in protected[task_id])
                    translated = geometry_keys[geometry].translate(bytes.maketrans(bytes(range(14)), bytes(mapping)))
                    assert translated not in seen
                    seen.add(translated)
        old_example = old_examples[0]
        new_example = SimpleNamespace(task_id=old_example.task_id, split=old_example.split,
                                      pair_index=old_example.pair_index)

        def select(augmentor, example):
            for epoch in range(args.selections):
                augmentor.set_epoch(epoch)
                augmentor.select_for_example(example)

        old_select, _ = median_seconds(lambda: select(old, old_example), args.repeats)
        new_select, _ = median_seconds(lambda: select(new, new_example), args.repeats)
        def entries(augmentor):
            return sum(len(pool.dihedral_indices) for pool in augmentor.augments_by_key.values())
        print(json.dumps(dict(
            march_commit=MARCH_COMMIT, dataset=str(args.data), tasks=len(tasks), pairs=len(old_examples),
            repeats=args.repeats, selections=args.selections,
            construction_seconds=dict(march=old_seconds, matched_domain=matched_seconds, single_grid=new_seconds),
            construction_speedup=dict(matched_domain=old_seconds / matched_seconds, single_grid=old_seconds / new_seconds),
            entries=dict(march=entries(old), matched_domain=entries(matched), single_grid=entries(new)),
            selection_seconds=dict(march=old_select, single_grid=new_select),
            selection_speedup=old_select / new_select,
            scope="CPU pool construction and epoch selection; excludes model/GPU throughput and historical geometry cache setup",
            validated="Every new pool: original retained, task-wide distinct additions, protected colors and injective mappings",
        ), indent=2))


if __name__ == "__main__":
    main()
