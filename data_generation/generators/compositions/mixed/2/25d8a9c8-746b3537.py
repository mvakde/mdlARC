"""Row-uniformity mask, then adjacent full-vector compression on both axes.

Standalone: only NumPy and the standard library. No parent files or saved pairs
are consulted. Diagnostics are returned in memory; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


POSITIVE = 5
BACKGROUND = 0


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside the permitted range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or any(n < 1 or n > 30 for n in grid.shape)
            or grid.dtype.kind not in "iu" or np.any((grid < 0) | (grid > 9))):
        raise ValueError("grid must be an integer ARC grid of size 1..30")
    return grid.astype(np.int8, copy=False)


def _collapse_vectors(board):
    """Keep the first member of each adjacent equal full-vector run.

    Both index sets are measured on the entire incoming board. Equality is not
    global uniqueness: separated recurrences have separate representatives.
    """
    rows = np.r_[True, np.any(board[1:] != board[:-1], axis=1)]
    columns = np.r_[True, np.any(board[:, 1:] != board[:, :-1], axis=0)]
    return board[np.ix_(rows, columns)].copy()


def _mask(grid):
    labels = np.where(np.all(grid == grid[:, :1], axis=1), POSITIVE, BACKGROUND)
    return np.repeat(labels[:, None], grid.shape[1], axis=1).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the input and fixed family conventions."""
    return _collapse_vectors(_mask(_grid(input_grid)))


def _reference(grid):
    """Direct specification check, using Python row sets and an ordered scan."""
    labels = [POSITIVE if len(set(map(int, row))) == 1 else BACKGROUND for row in grid]
    return np.asarray([v for i, v in enumerate(labels) if i == 0 or v != labels[i - 1]],
                      dtype=np.int8).reshape(-1, 1)


def _key(grid):
    return grid.shape, grid.tobytes()


def _answer_key(output):
    return int(output[0, 0]), output.shape[0]


def _thicknesses(rng, runs, force_repeat=False):
    """Four thickness regimes; allocation never exceeds 30 source rows."""
    mode = int(rng.integers(4))
    result = np.ones(runs, dtype=int)
    if mode == 1:
        result *= int(rng.integers(1, min(5, 30 // runs) + 1))
    elif mode == 2:
        result[int(rng.integers(runs))] += int(rng.integers(1, min(9, 30 - runs) + 1)) if runs < 30 else 0
    elif mode == 3:
        for _ in range(int(rng.integers(0, 31 - runs))):
            result[int(rng.integers(runs))] += 1
    if force_repeat and result.max() == 1 and runs < 30:
        result[int(rng.integers(runs))] += 1
    return result


def _mixed(rng, palette, width, style=None):
    a, b = map(int, rng.choice(palette, size=2, replace=False))
    if style is None:
        style = str(rng.choice(("first", "middle", "last", "alternating", "blocks", "random")))
    if style in ("first", "middle", "last"):
        row = np.full(width, a, dtype=np.int8)
        position = 0 if style == "first" else width - 1
        if style == "middle":
            position = int(rng.integers(1, width - 1)) if width >= 3 else 0
        row[position] = b
    elif style == "alternating":
        row = np.asarray([a if c % 2 == 0 else b for c in range(width)], dtype=np.int8)
    elif style == "blocks":
        cut = int(rng.integers(1, width))
        row = np.asarray([a] * cut + [b] * (width - cut), dtype=np.int8)
    else:
        row = np.asarray(rng.choice(palette, size=width), dtype=np.int8)
        row[0], row[-1] = a, b
    return row


def _make_grid(rng, start, runs, thicknesses, width, palette, kind):
    classes = [bool(start) != bool(i % 2) for i in range(runs)]
    negative_indices = [i for i, positive in enumerate(classes) if not positive]
    forced = {}
    if kind == "equality":
        styles = list(rng.permutation(["first", "middle", "last"]))
        for j, i in enumerate(negative_indices):
            # Keep EVERY diagnostic negative run internally identical. Otherwise
            # a partial predicate can add transitions in an unconstrained run
            # that accidentally compensate for transitions lost elsewhere.
            forced[i] = _mixed(rng, palette, width, str(styles[j % 3]))
    # A recurring column partition in the interaction demonstration. The two
    # outer full column vectors are equal but separated by a different vector.
    if kind == "interaction":
        column_lengths = rng.integers(1, 4, size=3)
        width = int(column_lengths.sum())
    rows, run_ranges = [], []
    for i, (positive, count) in enumerate(zip(classes, thicknesses)):
        begin = len(rows)
        for _ in range(int(count)):
            if positive:
                row = np.full(width, int(rng.choice(palette)), dtype=np.int8)
            elif i in forced:
                row = forced[i].copy()
            elif kind == "interaction":
                a, b = map(int, rng.choice(palette, size=2, replace=False))
                row = np.repeat(np.asarray([a, b, a], dtype=np.int8), column_lengths)
            else:
                row = _mixed(rng, palette, width)
                if rows and not np.all(rows[-1] == rows[-1][0]) and rng.random() < 0.22:
                    row = rows[-1].copy()
            rows.append(row)
        run_ranges.append((begin, len(rows)))
    # Guarantee adjacent different raw rows that A must merge BEFORE B acts.
    if kind in ("equality", "interaction"):
        for positive in ((True, False) if kind == "interaction" else (True,)):
            index = next(i for i, flag in enumerate(classes) if flag == positive)
            begin, end = run_ranges[index]
            _require(end - begin >= 2, "missing contrast allocation")
            if positive:
                first = int(rows[begin][0])
                second = int(rng.choice([int(c) for c in palette if int(c) != first]))
                rows[begin + 1] = np.full(width, second, dtype=np.int8)
            else:
                # Recoloring a mixed row preserves its equality class but makes
                # the two complete row vectors different.
                rows[begin + 1] = (rows[begin] + 1).astype(np.int8) % 10
    if kind == "interaction":
        # Exact separated row recurrence, for both classes whenever possible.
        for positive in (True, False):
            indices = [i for i, flag in enumerate(classes) if flag == positive]
            if len(indices) >= 2:
                rows[run_ranges[indices[-1]][0]] = rows[run_ranges[indices[0]][0]].copy()
    return np.asarray(rows, dtype=np.int8)


def _candidate(rng, kind, width_limit, answer=None):
    if answer is None:
        runs = int(rng.choice((7, 9))) if kind == "equality" else int(rng.integers(3, 7))
        start = bool(rng.integers(2))
        if kind == "ordinary":
            runs = 1 if rng.random() < 0.18 else int(rng.integers(2, 11))
    else:
        start, runs = answer
        start = bool(start)
    thicknesses = _thicknesses(rng, runs, force_repeat=kind != "ordinary")
    if kind in ("equality", "interaction"):
        for positive in ((True, False) if kind == "interaction" else (True,)):
            index = next(i for i in range(runs) if (start != bool(i % 2)) == positive)
            thicknesses[index] = max(2, int(thicknesses[index]))
        while int(thicknesses.sum()) > 30:
            eligible = [i for i in range(runs) if thicknesses[i] > 2]
            thicknesses[int(rng.choice(eligible))] -= 1
    width = int(rng.integers(3 if kind != "ordinary" else 2, width_limit + 1))
    if kind == "ordinary" and runs == 1 and start and rng.random() < 0.2:
        width = 1
    palette = np.asarray(rng.choice(10, size=int(rng.integers(2, 11)), replace=False), dtype=np.int8)
    grid = _make_grid(rng, start, runs, thicknesses, width, palette, kind)
    if kind == "equality":
        # Make literal input-5 detection fail at the first output label. A
        # bijective palette swap preserves every equality and run allocation.
        first_color = int(grid[0, 0])
        target = (6 if first_color == 5 else first_color) if start else 5
        permutation = np.arange(10, dtype=np.int8)
        permutation[first_color], permutation[target] = target, first_color
        grid = permutation[grid]
    return {"input": grid, "output": solve(grid)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    width_limit = int(rng.choice((6, 12, 30)))
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, kind, answer=None):
        for _ in range(64):
            pair = _candidate(rng, kind, width_limit, answer)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return pair
        raise RuntimeError("bounded unique-input construction exhausted")

    first = add("train", "equality")
    # Opposite starting label rules out a fixed alternating output template.
    add("train", "interaction", (int(first["output"][0, 0] == 0), int(rng.integers(3, 7))))
    for _ in range(num_train - 2):
        add("train", "ordinary")
    rng.shuffle(puzzle["train"])
    used_answers = {_answer_key(pair["output"]) for pair in puzzle["train"]}
    for _ in range(num_test):
        # A new class-run arrangement, sometimes shorter and sometimes longer.
        choices = [(start, runs) for start in (0, 1) for runs in range(1, 15)
                   if ((POSITIVE if start else BACKGROUND), runs) not in used_answers]
        answer = choices[int(rng.integers(len(choices)))]
        pair = add("test", "ordinary", answer)
        used_answers.add(_answer_key(pair["output"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _shortcut_labels(grid, kind):
    if kind == "endpoints":
        return grid[:, 0] == grid[:, -1]
    if kind == "first_two":
        return grid[:, 0] == grid[:, min(1, grid.shape[1] - 1)]
    if kind == "last_two":
        return grid[:, -1] == grid[:, max(0, grid.shape[1] - 2)]
    if kind == "majority":
        return np.asarray([max(np.unique(row, return_counts=True)[1]) > len(row) / 2 for row in grid])
    if kind == "literal_five":
        return grid[:, 0] == POSITIVE
    raise ValueError("unknown shortcut")


def _class_answer(labels):
    values = np.where(labels, POSITIVE, BACKGROUND).astype(np.int8)
    keep = np.r_[True, values[1:] != values[:-1]]
    return values[keep, None]


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, train_answers, test_answers = set(), set(), set()
    shortcuts = {s: False for s in ("endpoints", "first_two", "last_two", "majority", "literal_five")}
    mismatches = set()
    different_positive = different_negative = recurrence = False
    ablation_a = ablation_b_rows = ablation_b_columns = precompress = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected = _reference(inp)
            _require(np.array_equal(out, expected), "incorrect semantic output")
            _require(np.array_equal(solve(inp), expected), "solver disagreement")
            _require(out.shape[1] == 1 and set(map(int, out.flat)) <= {0, 5}, "invalid output convention")
            _require(np.all(out[1:, 0] != out[:-1, 0]), "uncompressed output")
            if split == "test":
                signature = _answer_key(out)
                _require(signature not in train_answers | test_answers, "test answer is not held out")
                test_answers.add(signature)
                continue
            train_answers.add(_answer_key(out))
            labels = np.all(inp == inp[:, :1], axis=1)
            for kind in shortcuts:
                shortcuts[kind] |= not np.array_equal(_class_answer(_shortcut_labels(inp, kind)), out)
            for row, positive in zip(inp, labels):
                colors, counts = np.unique(row, return_counts=True)
                if not positive and len(row) >= 3 and len(colors) == 2 and sorted(map(int, counts)) == [1, len(row) - 1]:
                    position = int(np.flatnonzero(row == colors[int(np.argmin(counts))])[0])
                    mismatches.add("first" if position == 0 else "last" if position == len(row) - 1 else "middle")
            for i in range(1, len(inp)):
                different = not np.array_equal(inp[i], inp[i - 1])
                different_positive |= bool(labels[i] and labels[i - 1] and different)
                different_negative |= bool(not labels[i] and not labels[i - 1] and different)
            recurrence |= len(out) >= 3
            mask = _mask(inp)
            ablation_a |= not np.array_equal(_collapse_vectors(inp), out)
            ablation_b_rows |= not np.array_equal(mask[:, :1], out)
            row_only = mask[np.r_[True, np.any(mask[1:] != mask[:-1], axis=1)]]
            ablation_b_columns |= not np.array_equal(row_only, out)
            precompress |= not np.array_equal(_mask(_collapse_vectors(inp))[:, :1], out)
    _require(all(shortcuts.values()), "training permits a tested simpler predicate")
    _require(mismatches == {"first", "middle", "last"}, "missing whole-row equality contrasts")
    _require(different_positive and different_negative, "missing abstraction-before-collapse contrasts")
    _require(recurrence, "missing separated class recurrence")
    _require(ablation_a and ablation_b_rows and ablation_b_columns and precompress,
             "a parent operation or causal order has no observable effect")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, file-free semantic, count, metamorphic and 42-seed checks."""
    shapes, palettes, masks, answers, thicknesses, fingerprints = (set() for _ in range(6))
    extremes, overlap, pairs = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed not reproducible")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                pairs += 1
                shapes.add(inp.shape)
                palette = tuple(map(int, np.unique(inp)))
                palettes.add(palette)
                overlap += 0 in palette or 5 in palette
                labels = np.all(inp == inp[:, :1], axis=1)
                masks.add(tuple(map(bool, labels)))
                starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
                thicknesses.add(tuple(map(int, np.diff(np.r_[starts, len(labels)]))))
                answers.add(_answer_key(out))
                if len(out) == 1:
                    extremes.add(int(out[0, 0]))
                # Bijective recoloring and full-column duplication preserve A.
                _require(np.array_equal(solve((inp + 3) % 10), out), "literal input colors leaked")
                if inp.shape[1] <= 15:
                    _require(np.array_equal(solve(np.repeat(inp, 2, axis=1)), out), "column dilation changed answer")
                if inp.shape[0] <= 15:
                    _require(np.array_equal(solve(np.repeat(inp, 2, axis=0)), out), "run dilation changed answer")
    _require(len(fingerprints) == 42, "seeds lack distinct puzzles")
    _require(len(shapes) >= 40 and len(palettes) >= 40, "shape or palette variation too small")
    _require(len(masks) >= 100 and len(thicknesses) >= 70 and len(answers) >= 20,
             "class arrangements or compression geometry too restricted")
    _require(extremes == {0, 5} and overlap >= 50, "missing extremes or color-role overlap")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(1000 + a * 10 + b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "counts changed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exhaust all two-color grids of shapes at most 3x3 (682 grids).
    for height in range(1, 4):
        for width in range(1, 4):
            for number in range(2 ** (height * width)):
                grid = np.asarray([(number >> bit) & 1 for bit in range(height * width)],
                                  dtype=np.int8).reshape(height, width)
                _require(np.array_equal(solve(grid), _reference(grid)), "small-grid rule failure")
    for height, width in ((1, 30), (30, 1), (30, 30)):
        grid = np.full((height, width), 0, dtype=np.int8)
        _require(np.array_equal(solve(grid), [[5]]), "uniform boundary failure")
    # Maximal alternation and a controlled predicate change in an isolated row.
    grid = np.asarray([[2, 2, 2] if i % 2 else [2, 3, 2] for i in range(30)], dtype=np.int8)
    _require(solve(grid).shape == (30, 1), "separated recurrence lost")
    changed = grid.copy()
    changed[2] = 2
    _require(not np.array_equal(solve(grid), solve(changed)), "parent A intervention has no effect")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "palettes": len(palettes), "row_masks": len(masks),
            "run_thickness_vectors": len(thicknesses), "binary_answers": len(answers),
            "exhaustive_small_grids": 682, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
