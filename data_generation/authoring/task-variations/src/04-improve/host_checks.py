"""Host-owned checks, copied unchanged into each generator workspace.

These checks inspect returned grids without delegating acceptance to the
generator's validator. Optional task-specific checks live in a protected
reference_checks.py, never in generator.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, deque

import numpy as np


AUTOMATIC_SEED_COUNT = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def grid_key(grid):
    array = np.asarray(grid)
    return (tuple(array.shape), tuple(int(v) for v in array.flat))


def pair_key(pair):
    return (grid_key(pair["input"]), grid_key(pair["output"]))


def puzzle_key(puzzle):
    return tuple(tuple(pair_key(p) for p in puzzle[split]) for split in ("train", "test"))


def check_grid(grid, label):
    require(isinstance(grid, np.ndarray), f"{label}: expected a NumPy array")
    require(grid.ndim == 2, f"{label}: expected a two-dimensional grid")
    require(all(1 <= n <= 30 for n in grid.shape), f"{label}: grid dimensions must be 1..30")
    require(np.issubdtype(grid.dtype, np.integer), f"{label}: expected an integer dtype")
    require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label}: colors must be 0..9")


def check_puzzle(puzzle, source_pairs, requested=(None, None)):
    require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle must contain exactly train and test")
    require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
            "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
            "illegal generated pair counts")
    for actual, expected in zip((train_count, test_count), requested):
        require(expected is None or actual == expected, "generator ignored a requested pair count")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    f"{label}: expected exactly input and output")
            for role in ("input", "output"):
                check_grid(pair[role], f"{label}.{role}")
            key = pair_key(pair)
            require(key[0] not in seen_inputs, f"{label}: duplicate input")
            require(key not in seen_pairs, f"{label}: duplicate pair")
            # Equality includes both shape and contents. Source dimensions remain legal.
            require(key not in source_pairs, f"{label}: exact source-pair copy")
            seen_inputs.add(key[0])
            seen_pairs.add(key)


def component_sizes(mask):
    """Sizes of four-connected components in a boolean grid."""
    remaining = set(zip(*np.nonzero(mask)))
    sizes = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        size = 1
        while queue:
            r, c = queue.popleft()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
                    size += 1
        sizes.append(size)
    return sorted(sizes)


def structural_signature(grid):
    """Normalize recoloring and empty outer margins; preserve geometry."""
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    rows, cols = np.nonzero(grid != background)
    cropped = grid if not len(rows) else grid[rows.min():rows.max()+1, cols.min():cols.max()+1]
    palette = {background: 0}
    values = []
    for value in cropped.flat:
        value = int(value)
        if value not in palette:
            palette[value] = len(palette)
        values.append(palette[value])
    return hashlib.sha256(json.dumps([list(cropped.shape), values]).encode()).hexdigest()


def pair_features(pair):
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    sizes = component_sizes(inp != background)
    return {
        "input_height": int(inp.shape[0]),
        "input_width": int(inp.shape[1]),
        "input_shape": list(inp.shape),
        "output_shape": list(out.shape),
        "input_palette": [int(c) for c in colors],
        "input_color_count": len(colors),
        "dominant_color": background,
        "foreground_component_count": len(sizes),
        "foreground_component_sizes": sizes,
        "structure": structural_signature(inp),
    }


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def distribution(values):
    counts = Counter(_encoded(value) for value in values)
    return [{"value": json.loads(key), "count": count} for key, count in sorted(counts.items())]


def validate_policy(policy):
    require(isinstance(policy, dict) and policy.get("schema_version") == 1,
            "acceptance.json requires schema_version 1")
    require(type(policy.get("require_reference_solver", False)) is bool,
            "require_reference_solver must be boolean")
    require(type(policy.get("check_training_predictions", False)) is bool,
            "check_training_predictions must be boolean")
    criteria = policy.get("criteria", [])
    require(isinstance(criteria, list), "acceptance criteria must be a list")
    ids = set()
    for criterion in criteria:
        require(isinstance(criterion, dict), "each criterion must be an object")
        identifier = criterion.get("id")
        require(isinstance(identifier, str) and identifier and identifier not in ids,
                "criterion IDs must be unique non-empty strings")
        ids.add(identifier)
        require(isinstance(criterion.get("review_request"), str) and criterion["review_request"].strip(),
                f"{identifier}: link the criterion to a review request")
        features = criterion.get("features")
        require(isinstance(features, list) and features and all(isinstance(f, str) and f for f in features),
                f"{identifier}: features must be a non-empty list")
        require(criterion.get("scope") in ("per_pair", "within_puzzle", "across_seeds"),
                f"{identifier}: invalid scope")
        require(criterion.get("split", "all") in ("all", "train", "test"), f"{identifier}: invalid split")
        require(any(k in criterion for k in ("min_distinct", "max_distinct", "min", "max", "must_include")),
                f"{identifier}: no measurable acceptance condition")
        for key in ("min_distinct", "max_distinct"):
            if key in criterion:
                require(type(criterion[key]) is int and criterion[key] > 0,
                        f"{identifier}: {key} must be a positive integer")
        for bound in ("min", "max"):
            if bound in criterion:
                require(type(criterion[bound]) in (int, float) and len(features) == 1,
                        f"{identifier}: numeric bounds require one scalar feature")
        if "must_include" in criterion:
            require(isinstance(criterion["must_include"], list), f"{identifier}: must_include must be a list")


def evaluate_criteria(records, policy):
    results = []
    for criterion in policy.get("criteria", []):
        selected = [r for r in records if criterion.get("split", "all") in ("all", r["split"])]
        groups = {}
        for index, record in enumerate(selected):
            key = record["seed"] if criterion["scope"] == "within_puzzle" else (
                index if criterion["scope"] == "per_pair" else "all")
            names = criterion["features"]
            require(all(name in record["features"] for name in names),
                    f"{criterion['id']}: feature missing; provide a protected reference_checks.features hook")
            values = [record["features"][name] for name in names]
            groups.setdefault(key, []).append(values[0] if len(values) == 1 else values)
        require(bool(groups), f"{criterion['id']}: no samples to measure")
        failures = []
        for key, values in groups.items():
            if len({_encoded(v) for v in values}) < criterion.get("min_distinct", 1):
                failures.append(f"group {key}: insufficient distinct values")
            if "max_distinct" in criterion and len({_encoded(v) for v in values}) > criterion["max_distinct"]:
                failures.append(f"group {key}: too many distinct values")
            for bound, compare in (("min", lambda a, b: a >= b), ("max", lambda a, b: a <= b)):
                if bound in criterion and not all(type(v) in (int, float) and compare(v, criterion[bound]) for v in values):
                    failures.append(f"group {key}: {bound} bound failed")
            for expected in criterion.get("must_include", []):
                if _encoded(expected) not in {_encoded(v) for v in values}:
                    failures.append(f"group {key}: missing {expected}")
        results.append({"id": criterion["id"], "review_request": criterion["review_request"],
                        "passed": not failures, "failures": failures,
                        "distribution": distribution([v for values in groups.values() for v in values])})
    return results


def check_predictions(solver, puzzle, label, check_training=False):
    """Never hand a test output, seed, or generator namespace to the solver."""
    train = [{role: np.array(pair[role], dtype=int, copy=True) for role in ("input", "output")}
             for pair in puzzle["train"]]
    for index, pair in enumerate(puzzle["test"]):
        expected = np.array(pair["output"], dtype=int, copy=True)
        predicted = solver(copy.deepcopy(train), np.array(pair["input"], dtype=int, copy=True))
        check_grid(predicted, f"{label}.prediction[{index}]")
        require(np.array_equal(predicted, expected), f"{label}: reference prediction mismatch for test[{index}]")
    if check_training:
        # Opt in only for families where other demonstrations or the input alone
        # suffice; remove the queried training pair's label as well.
        for index, pair in enumerate(train):
            examples = [p for i, p in enumerate(train) if i != index]
            predicted = solver(copy.deepcopy(examples), pair["input"].copy())
            check_grid(predicted, f"{label}.train_prediction[{index}]")
            require(np.array_equal(predicted, pair["output"]),
                    f"{label}: reference prediction mismatch for train[{index}]")


def run_checks(namespace, source_task, policy, seed_base, reference=None):
    validate_policy(policy)
    reference = reference or {}
    solver, feature_hook = reference.get("solve"), reference.get("features")
    require(not (policy.get("require_reference_solver", False) or policy.get("check_training_predictions", False)) or callable(solver),
            "this task requires a protected reference_checks.solve(train_pairs, test_input)")
    source_pairs = {pair_key(pair) for split in ("train", "test") for pair in source_task[split]}
    report = {"schema_version": 1, "seed_base": seed_base, "valid": False, "errors": [],
              "reference_solver": "configured" if callable(solver) else "not_configured",
              "review_criteria": "configured" if policy.get("criteria") else "not_configured",
              "checked_calls": 0, "automatic_seeds": [], "records": [], "criteria": []}
    generate = namespace["generate_puzzle"]

    def sample(seed, requested=(None, None), *, record=False):
        label = f"seed={seed}, counts={requested}"
        try:
            puzzle = generate(seed, num_train=requested[0], num_test=requested[1])
            check_puzzle(puzzle, source_pairs, requested)
            if callable(solver):
                check_predictions(solver, puzzle, label, policy.get("check_training_predictions", False))
            report["checked_calls"] += 1
            if record:
                for split in ("train", "test"):
                    for index, pair in enumerate(puzzle[split]):
                        features = pair_features(pair)
                        if callable(feature_hook):
                            extra = feature_hook(pair["input"].copy(), pair["output"].copy())
                            require(isinstance(extra, dict) and not (set(extra) & set(features)),
                                    "reference features must be a dict with distinct names")
                            features.update(extra)
                        _encoded(features)  # reject non-JSON metrics instead of silently losing them
                        report["records"].append({"seed": seed, "split": split, "pair": index, "features": features})
            return puzzle_key(puzzle)
        except Exception as exc:
            raise ValueError(f"{label}: {exc}") from exc

    try:
        if callable(solver):
            check_predictions(solver, source_task, "source task", policy.get("check_training_predictions", False))
        train_counts, test_counts, puzzles = set(), set(), set()
        first_key = None
        for index in range(AUTOMATIC_SEED_COUNT):
            seed = (seed_base + index * 104729) % (2**63)
            report["automatic_seeds"].append(seed)
            key = sample(seed, record=True)
            if index == 0:
                first_key = key
            puzzles.add(key)
            train_counts.add(len(key[0]))
            test_counts.add(len(key[1]))
            if index < 3:
                require(sample(seed) == key, f"seed={seed}: nondeterministic automatic generation")
        require(len(train_counts) >= 4 and len(test_counts) >= 3,
                "automatic counts need at least four train counts and three test counts")
        require(len(puzzles) >= 40, "fewer than 40 distinct puzzles across 42 fresh seeds")
        # Exercise every legal fully specified combination, plus all partial counts.
        cases = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        cases += [(a, None) for a in range(2, 11)] + [(None, b) for b in range(1, 5)]
        for index, requested in enumerate(cases):
            seed = (seed_base + 10_000_019 + index) % (2**63)
            key = sample(seed, requested)
            if requested in ((2, 1), (10, 4), (None, 4), (2, None)):
                require(sample(seed, requested) == key, f"seed={seed}: nondeterministic requested counts")
        invalid = [(1, None), (11, None), (None, 0), (None, 5)]
        invalid += [(a, b) for a in range(2, 5) for b in range(a, 5)]
        for requested in invalid:
            try:
                generate(seed_base, num_train=requested[0], num_test=requested[1])
            except ValueError:
                continue
            raise ValueError(f"illegal counts {requested} must raise ValueError")
        require(sample(report["automatic_seeds"][0]) == first_key,
                "nondeterministic generation after intervening seeds and count requests")
        report["criteria"] = evaluate_criteria(report["records"], policy)
        for criterion in report["criteria"]:
            if not criterion["passed"]:
                report["errors"].append(f"{criterion['id']}: " + "; ".join(criterion["failures"][:4]))
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    # Include both global and per-puzzle distributions, even on partial failure.
    names = sorted({name for row in report["records"] for name in row["features"]})
    report["distributions"] = {name: distribution([r["features"][name] for r in report["records"] if name in r["features"]])
                               for name in names}
    report["within_puzzle_distinct"] = {
        str(seed): {name: len({_encoded(r["features"][name]) for r in report["records"]
                               if r["seed"] == seed and name in r["features"]}) for name in names}
        for seed in report["automatic_seeds"]}
    report["valid"] = not report["errors"]
    return report
