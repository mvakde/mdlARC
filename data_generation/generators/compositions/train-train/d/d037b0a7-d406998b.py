"""Downward rays, exposed at alternating ranks in each row's active set.

Only standard-library modules and NumPy are used. No parent generator is loaded.
"""

import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train")
    b = _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= v <= 30 for v in a.shape):
        raise ValueError("grid must be a 2D rectangle with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or not np.all((a >= 0) & (a <= 9)):
        raise ValueError("grid cells must be integer colors 0..9")
    return a


def _read_input(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    most = int(np.argmax(counts))
    if int(counts[most]) * 2 <= grid.size:
        raise ValueError("input requires a strict majority background")
    background = int(colors[most])
    seeds = []
    for column in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, column] != background)
        if len(rows) > 1:
            raise ValueError("at most one source is allowed in each column")
        if len(rows):
            row = int(rows[0])
            seeds.append((row, column, int(grid[row, column])))
    if not seeds:
        raise ValueError("at least one source is required")
    return grid, background, seeds


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the given grid; preserve sources even when their rank is even."""
    grid, _, seeds = _read_input(input_grid)
    output = grid.copy()
    # Seeds are in increasing column order. Membership is recomputed at every
    # row, including sources beginning on this row; invisible rays still count.
    for row in range(grid.shape[0]):
        active = [source for source in seeds if source[0] <= row]
        for _, column, color in active[::-2]:
            output[row, column] = color
    return output


def _expected_and_features(grid):
    """Cellwise parity formulation, separate from solve's row/rank sweep."""
    grid, background, seeds = _read_input(grid)
    height, width = grid.shape
    expected = grid.copy()
    full = grid.copy()
    static = grid.copy()
    left = grid.copy()
    absolute = grid.copy()
    gap_reset = grid.copy()
    selected = suppressed = 0
    resurrection = False
    seed_suppressed = False
    starts = [s[0] for s in seeds]
    for index, (start, column, color) in enumerate(seeds):
        full[start:, column] = color
        if (len(seeds) - 1 - index) % 2 == 0:
            static[start:, column] = color
        if (width - 1 - column) % 2 == 0:
            absolute[start:, column] = color
        right_at_start = sum(s[0] <= start for s in seeds[index + 1:])
        seed_suppressed |= right_at_start % 2 == 1
        states = []
        for row in range(start + 1, height):
            right = sum(s[0] <= row for s in seeds[index + 1:])
            visible = right % 2 == 0
            states.append(visible)
            if visible:
                expected[row, column] = color
                selected += 1
            else:
                suppressed += 1
            if sum(s[0] <= row for s in seeds[:index]) % 2 == 0:
                left[row, column] = color
            # Counterfactual: reset alternation at every physical empty column.
            run_right = 0
            for other in range(column + 1, width):
                if not any(s[1] == other and s[0] <= row for s in seeds):
                    break
                run_right += 1
            if run_right % 2 == 0:
                gap_reset[row, column] = color
        compressed = [state for i, state in enumerate(states)
                      if i == 0 or state != states[i - 1]]
        resurrection |= any(compressed[i:i + 3] == [True, False, True]
                            for i in range(len(compressed) - 2))
    return expected, {
        "shape": (height, width), "background": background, "seeds": seeds,
        "count": len(seeds), "starts": tuple(starts),
        "level": len(set(starts)) == 1,
        "resurrection": resurrection, "seed_suppressed": seed_suppressed,
        "selected": selected, "suppressed": suppressed,
        "bottom": height - 1 in starts, "top": 0 in starts,
        "full_diff": not np.array_equal(full, expected),
        "static_diff": not np.array_equal(static, expected),
        "left_diff": not np.array_equal(left, expected),
        "absolute_diff": not np.array_equal(absolute, expected),
        "gap_diff": not np.array_equal(gap_reset, expected),
    }


def _signature(grid):
    a = np.asarray(grid)
    return (a.shape, tuple(int(v) for v in a.flat))


def _layout(grid):
    a, _, seeds = _read_input(grid)
    return (a.shape, tuple((r, c) for r, c, _ in seeds))


def _columns(rng, width, count, mode):
    if mode == "dense":
        start = int(rng.integers(width - count + 1))
        return np.arange(start, start + count)
    if mode == "spaced" and width >= 2 * count - 1:
        # Sample in a contracted interval, then insert mandatory empty columns.
        base = np.sort(rng.choice(width - count + 1, count, replace=False))
        return base + np.arange(count)
    return np.sort(rng.choice(width, count, replace=False))


def _rows(rng, height, count, mode):
    if mode == "level":
        return np.full(count, int(rng.integers(height - 2)), dtype=int)
    if mode == "waves":
        rows = rng.integers(0, height, size=count)
        early, bottom = rng.choice(count, 2, replace=False)
        rows[int(early)] = 0
        rows[int(bottom)] = height - 1
        return rows
    if mode in ("rising", "falling"):
        rows = np.sort(rng.integers(0, height, size=count))
        return rows if mode == "rising" else rows[::-1]
    if mode == "two_level":
        levels = np.sort(rng.choice(height, 2, replace=False))
        rows = rng.choice(levels, size=count)
        rows[0], rows[-1] = levels
        rng.shuffle(rows)
        return rows
    if mode in ("valley", "ridge"):
        pivot = int(rng.integers(1, count))
        first = np.sort(rng.integers(0, height, size=pivot))[::-1]
        second = np.sort(rng.integers(0, height, size=count - pivot))
        rows = np.concatenate((first, second))
        return rows if mode == "valley" else height - 1 - rows
    return rng.integers(0, height, size=count)


def _make_pair(rng, height, width, count, background, mode, spacing,
               seen, *, force_zero=False, mandatory=None):
    for _ in range(1200):
        columns = _columns(rng, width, count, spacing)
        rows = _rows(rng, height, count, mode)
        available = [c for c in range(10) if c != background]
        palette = rng.choice(available, int(rng.integers(1, 5)), replace=False)
        colors = rng.choice(palette, count)
        if force_zero:
            colors[int(rng.integers(count))] = 0
        grid = np.full((height, width), background, dtype=np.int8)
        grid[rows, columns] = colors
        key = _layout(grid)
        if key in seen:
            continue
        output, features = _expected_and_features(grid)
        if not features["selected"] or not features["suppressed"]:
            continue
        if mandatory == "waves" and not features["resurrection"]:
            continue
        if mandatory == "level" and not (features["absolute_diff"] and features["gap_diff"]):
            continue
        # In the nonzero mandatory training pair, 0 is a visible *trail* color.
        if force_zero and not np.any((grid != output) & (output == 0)):
            continue
        seen.add(key)
        return {"input": grid, "output": output}
    raise RuntimeError("could not sample a distinct, informative ray arrangement")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    train_count, test_count = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    seen = set()
    train = []
    # Two contrasting demonstrations suffice even at the minimum count.
    wave_count = int(rng.choice([3, 5, 7]))
    level_count = int(rng.choice([2, 4, 6]))
    backgrounds = [0, int(rng.integers(1, 10))]
    rng.shuffle(backgrounds)
    for mandatory, count, background in (
        ("waves", wave_count, backgrounds[0]),
        ("level", level_count, backgrounds[1]),
    ):
        height = int(rng.integers(8, 15)) if mandatory == "waves" else int(rng.integers(5, 13))
        # Leave room for gaps, including odd-width gaps that defeat physical parity.
        width = int(rng.integers(max(count + 3, 2 * count), min(25, 3 * count + 8)))
        train.append(_make_pair(rng, height, width, count, background, mandatory,
                                "random", seen, force_zero=background != 0,
                                mandatory=mandatory))
    modes = ("irregular", "rising", "falling", "two_level", "valley", "ridge", "level")
    for _ in range(train_count - 2):
        count = int(rng.integers(2, 8))
        width = int(rng.integers(max(6, count), 26))
        train.append(_make_pair(
            rng, int(rng.integers(5, 17)), width, count,
            int(rng.integers(10)), str(rng.choice(modes)),
            str(rng.choice(["dense", "spaced", "random"])), seen))
    rng.shuffle(train)
    max_count = max(len(_read_input(p["input"])[2]) for p in train)
    max_width = max(p["input"].shape[1] for p in train)
    max_height = max(p["input"].shape[0] for p in train)
    test = []
    for index in range(test_count):
        regime = (seed + index) % 3
        if regime == 0:
            # More simultaneous participants, without a new operation.
            count = int(rng.integers(max_count + 1, 10))
            width = int(rng.integers(count + 1, 31))
            mode = str(rng.choice(["irregular", "rising", "valley", "two_level"]))
            spacing = str(rng.choice(["dense", "spaced", "random"]))
        elif regime == 1:
            # Longer horizontal distances, not a longer parity sequence by default.
            count = int(rng.integers(2, min(max_count, 6) + 1))
            width = int(rng.integers(max_width + 1, 31))
            mode = str(rng.choice(["two_level", "falling", "irregular", "level"]))
            spacing = "spaced"
        else:
            # Familiar dimensions with simultaneous and nonmonotone activation.
            count = int(rng.integers(3, min(max_count + 1, 8) + 1))
            width = int(rng.integers(max(count, 6), max(max_width, count) + 1))
            mode = str(rng.choice(["valley", "ridge", "irregular", "two_level"]))
            spacing = str(rng.choice(["dense", "random"]))
        test.append(_make_pair(
            rng, int(rng.integers(5, max_height + 1)), width, count,
            int(rng.integers(10)), mode, spacing, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen_inputs, seen_layouts = set(), set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape must be preserved")
            expected, f = _expected_and_features(inp)
            _require(np.array_equal(out, expected), "incorrect alternating active-ray trails")
            _require(np.array_equal(solve(inp), out), "solver and cellwise rule disagree")
            key, layout = _signature(inp), _layout(inp)
            _require(key not in seen_inputs and layout not in seen_layouts,
                     "duplicate input or color-independent layout")
            seen_inputs.add(key)
            seen_layouts.add(layout)
            background = f["background"]
            original = inp != background
            _require(np.array_equal(out[original], inp[original]), "source was erased")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "unencoded output color")
            _require(f["selected"] > 0 and f["suppressed"] > 0,
                     "sample must have both exposed and suppressed trail cells")
            _require(f["full_diff"], "sample does not expose B's contribution")
            _require(np.any(inp != out), "sample does not expose A's contribution")
            if split == "train":
                training.append((inp, out, f))
    features = [f for _, _, f in training]
    _require(any(f["resurrection"] and f["bottom"] and f["top"] for f in features),
             "training needs a ray that disappears and reappears, plus boundary starts")
    _require(any(f["level"] and f["count"] % 2 == 0 for f in features),
             "training needs simultaneous even-count activation")
    _require({f["count"] % 2 for f in features} == {0, 1},
             "training must show both active-sequence phases")
    for name in ("static_diff", "left_diff", "absolute_diff", "gap_diff"):
        _require(any(f[name] for f in features), f"training permits shortcut {name}")
    _require(any(f["seed_suppressed"] for f in features),
             "training must preserve a source at a suppressed rank")
    backgrounds = {f["background"] for f in features}
    _require(0 in backgrounds and any(c != 0 for c in backgrounds),
             "training needs zero and nonzero backgrounds")
    _require(any(f["background"] != 0 and np.any((a != b) & (b == 0))
                 for a, b, f in training), "training needs a zero-colored trail")


def _puzzle_key(puzzle):
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"]))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """Bounded diversity, discrimination, interface, and metamorphic checks."""
    first = generate_puzzle(271828, 7, 3)
    _require(_puzzle_key(first) == _puzzle_key(generate_puzzle(271828, 7, 3)),
             "seed is not deterministic")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(8000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b,
                     "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    shapes, layouts, palettes, backgrounds, counts = set(), set(), set(), set(), set()
    profiles, keys, test_parities = set(), set(), set()
    resurrection_count = repeated_colors = aligned_count = bottom_count = 0
    # A color permutation and insertion of an empty column must commute with solve.
    permutation = np.array([7, 3, 9, 0, 6, 1, 8, 4, 2, 5], dtype=np.int8)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        keys.add(_puzzle_key(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                _, f = _expected_and_features(inp)
                seeds = f["seeds"]
                shapes.add(inp.shape)
                layouts.add(_layout(inp))
                palettes.add((f["background"], tuple(sorted(set(s[2] for s in seeds)))))
                backgrounds.add(f["background"])
                counts.add(len(seeds))
                profiles.add(f["starts"])
                resurrection_count += f["resurrection"]
                aligned_count += f["level"]
                bottom_count += f["bottom"]
                repeated_colors += len(set(s[2] for s in seeds)) < len(seeds)
                if split == "test":
                    test_parities.add(len(seeds) % 2)
                _require(np.array_equal(solve(permutation[inp]), permutation[out]),
                         "literal-color equivariance failed")
                if inp.shape[1] < 30:
                    where = (seed + len(seeds)) % (inp.shape[1] + 1)
                    expanded = np.insert(inp, where, f["background"], axis=1)
                    expanded_out = np.insert(out, where, f["background"], axis=1)
                    _require(np.array_equal(solve(expanded), expanded_out),
                             "empty columns changed sequence rank")
    _require(len(keys) == 42, "cross-seed puzzles repeat")
    _require(len(shapes) >= 60 and len(layouts) >= 180 and len(profiles) >= 150,
             "too little structural variation across 42 seeds")
    _require(counts == set(range(2, 10)) and test_parities == {0, 1},
             "source counts or test phases collapsed")
    _require(len(palettes) >= 100 and backgrounds == set(range(10)),
             "palette/background roles collapsed")
    _require(resurrection_count >= 42 and aligned_count >= 42 and bottom_count >= 42,
             "core structural modes absent")
    _require(repeated_colors >= 80, "repeated source colors underrepresented")
    # Explicit boundary and same-row activation examples, including an invisible
    # source that still flips a ray to its left on its own starting row.
    example = np.zeros((7, 7), dtype=np.int8)
    example[0, 0], example[2, 3], example[6, 6] = 2, 4, 8
    result = solve(example)
    _require(tuple(result[:, 0]) == (2, 2, 0, 0, 0, 0, 2),
             "a late right-hand source failed to restore the left ray")
    _require(result[6, 3] == 0 and result[6, 6] == 8,
             "bottom source failed to change membership at the boundary")


if __name__ == "__main__":
    self_test()
    # Local inspection artifacts stay in scratch; JSON is deliberately compact.
    path = Path("scratch")
    path.mkdir(exist_ok=True)
    p = generate_puzzle(0)
    serial = {s: [{k: a.tolist() for k, a in pair.items()} for pair in p[s]]
              for s in ("train", "test")}
    (path / "seed-0.json").write_text(json.dumps(serial, separators=(",", ":")))
    lines = []
    for split in ("train", "test"):
        for i, pair in enumerate(p[split]):
            lines.append(f"{split} {i + 1}")
            for role in ("input", "output"):
                lines.append(role + ":")
                lines.extend("".join(str(int(c)) for c in row) for row in pair[role])
    (path / "seed-0.txt").write_text("\n".join(lines) + "\n")
    print("self_test passed; seed-0 inspection files written in scratch/")
