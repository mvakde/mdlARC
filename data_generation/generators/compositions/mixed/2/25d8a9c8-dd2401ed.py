"""Whole-row masks supply the counts for one measured delimiter advance.

No filesystem access, parent imports, stored examples, or puzzle-level solver state.
Diagnostics in self_test are streamed and bounded; no execution traces are saved.
"""
from __future__ import annotations

import hashlib
import numpy as np


BACKGROUND, REFERENCE, CANDIDATE, DELIMITER = 0, 5, 2, 8
DATA_COLORS = tuple(c for c in range(10) if c != DELIMITER)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside its legal range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return grid


def _facts(value):
    grid = _grid(value)
    h, w = grid.shape
    lines = np.flatnonzero(np.all(grid == DELIMITER, axis=1))
    if w < 3 or len(lines) != 1:
        raise ValueError("expected width >= 3 and one full row of 8")
    s = int(lines[0])
    destination = 2 * s + 1
    if not 1 <= s <= 14 or destination >= h:
        raise ValueError("the measured next window and destination must fit")
    if not np.all(grid[destination] == BACKGROUND):
        raise ValueError("the destination must be a clear structural row")
    if np.count_nonzero(grid == DELIMITER) != w:
        raise ValueError("8 is reserved for the delimiter")
    mask = np.all(grid == grid[:, :1], axis=1)
    mask[[s, destination]] = False
    return {"grid": grid, "s": s, "destination": destination, "mask": mask,
            "reference": int(np.count_nonzero(mask[:s])),
            "candidate": int(np.count_nonzero(mask[s + 1:destination])),
            "later": int(np.count_nonzero(mask[destination + 1:]))}


def _render(facts, mask=None, promotion="threshold"):
    """Render the derived marks; alternate modes are validator ablations only."""
    grid, s, d = facts["grid"], facts["s"], facts["destination"]
    mask = facts["mask"].copy() if mask is None else np.array(mask, dtype=bool, copy=True)
    mask[[s, d]] = False
    reference = int(np.count_nonzero(mask[:s]))
    candidate = int(np.count_nonzero(mask[s + 1:d]))
    if promotion == "threshold":
        accepted = candidate >= reference
    elif promotion == "always":
        accepted = True
    elif promotion == "never":
        accepted = False
    elif promotion == "whole_far_side":
        accepted = int(np.count_nonzero(mask[s + 1:])) >= reference
    else:
        raise ValueError("unknown promotion mode")
    row_colors = np.zeros(grid.shape[0], dtype=np.int8)
    row_colors[mask] = CANDIDATE
    row_colors[:s] = np.where(mask[:s], REFERENCE, BACKGROUND)
    row_colors[s + 1:d] = np.where(
        mask[s + 1:d], REFERENCE if accepted else CANDIDATE, BACKGROUND)
    row_colors[d] = DELIMITER
    return np.repeat(row_colors[:, None], grid.shape[1], axis=1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the visible delimiter, whole-row equality, and fixed roles."""
    return _render(_facts(input_grid))


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def _positions(rng, length, count, style=None):
    mask = np.zeros(length, dtype=bool)
    if count == 0:
        return mask
    if style is None:
        style = str(rng.choice(("contiguous", "spread", "edges", "random")))
    if style == "contiguous":
        start = int(rng.integers(length - count + 1))
        positions = np.arange(start, start + count)
    elif style == "spread":
        positions = np.rint(np.linspace(0, length - 1, count)).astype(int)
        if bool(rng.integers(2)):
            positions = length - 1 - positions
    elif style == "edges":
        order = [x for pair in zip(range(length), range(length - 1, -1, -1))
                 for x in pair]
        positions = list(dict.fromkeys(order))[:count]
    else:
        positions = rng.choice(length, size=count, replace=False)
    mask[positions] = True
    _require(int(mask.sum()) == count, "position construction lost a row")
    return mask


def _mixed_row(rng, palette, width, style=None):
    a, b = (int(c) for c in rng.choice(palette, size=2, replace=False))
    if style is None:
        style = str(rng.choice(("first", "middle", "last", "alternating",
                                "runs", "balanced", "random")))
    if style in ("first", "middle", "last"):
        row = np.full(width, a, dtype=np.int8)
        position = {"first": 0, "last": width - 1}.get(style)
        if position is None:
            position = int(rng.integers(1, width - 1))
        row[position] = b
    elif style == "alternating":
        row = np.array([a, b] * ((width + 1) // 2), dtype=np.int8)[:width]
    elif style == "runs":
        cut = int(rng.integers(1, width))
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
    elif style == "balanced":
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, size=width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = b if row[0] == a else a
    return row


def _width(rng):
    band = int(rng.choice(3, p=(0.50, 0.32, 0.18)))
    low, high = ((3, 9), (10, 18), (19, 30))[band]
    return int(rng.integers(low, high + 1))


def _spec(rng, span, reference, candidate, *, core=False, positive_tail=False):
    capacity = 30 - (2 * span + 2)
    minimum = 1 if positive_tail else 0
    _require(capacity >= minimum, "requested tail does not fit")
    if capacity == minimum:
        tail = capacity
    elif rng.random() < 0.35:
        tail = int(rng.integers(minimum, min(capacity, max(2, minimum)) + 1))
    else:
        tail = int(rng.integers(minimum, capacity + 1))
    return {"s": span, "reference": reference, "candidate": candidate,
            "width": _width(rng), "tail": tail, "core": core,
            "positive_tail": positive_tail}


def _counts_for(rng, span, outcome):
    if outcome == "equal":
        reference = int(rng.integers(1, span + 1))
        return reference, reference
    if outcome == "below":
        reference = int(rng.integers(1, span + 1))
        return reference, int(rng.integers(reference))
    if outcome == "above":
        reference = int(rng.integers(1, span))
        return reference, int(rng.integers(reference + 1, span + 1))
    if outcome == "empty_reference":
        return 0, int(rng.integers(span + 1))
    raise ValueError("unknown count outcome")


def _make_input(rng, spec, pool):
    s, width, tail = spec["s"], spec["width"], spec["tail"]
    d = 2 * s + 1
    h = d + 1 + tail
    mask = np.zeros(h, dtype=bool)
    mask[:s] = _positions(rng, s, spec["reference"])
    mask[s + 1:d] = _positions(rng, s, spec["candidate"])
    if tail:
        minimum = 1 if spec["positive_tail"] else 0
        if (spec["candidate"] < spec["reference"] and rng.random() < 0.5
                and spec["reference"] - spec["candidate"] <= tail):
            minimum = max(minimum, spec["reference"] - spec["candidate"])
        tail_count = int(rng.integers(minimum, tail + 1))
        mask[d + 1:] = _positions(rng, tail, tail_count)
    palette_size = int(rng.integers(2, len(pool) + 1))
    palette = rng.choice(pool, size=palette_size, replace=False)
    positive_colors = rng.permutation(palette)
    negatives = [i for i in range(h) if i not in (s, d) and not mask[i]]
    forced = {}
    if spec["core"]:
        _require(len(negatives) >= 3, "core needs three equality contrasts")
        slots = rng.permutation(negatives)[:3]
        forced = dict(zip((int(i) for i in slots), ("first", "middle", "last")))
    grid = np.empty((h, width), dtype=np.int8)
    positive_index = 0
    for i in range(h):
        if i == s:
            grid[i] = DELIMITER
        elif i == d:
            grid[i] = BACKGROUND
        elif mask[i]:
            grid[i] = positive_colors[positive_index % palette_size]
            positive_index += 1
        else:
            grid[i] = _mixed_row(rng, palette, width, forced.get(i))
    return grid


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    pool = rng.choice(DATA_COLORS, size=int(rng.integers(3, 10)), replace=False)
    if rng.random() < 0.4 and BACKGROUND not in pool:
        pool[-1] = BACKGROUND
    s1, s2 = (int(x) for x in rng.choice(np.arange(4, 11), size=2, replace=False))
    q = int(rng.integers(1, 3))
    train = [_spec(rng, s1, q, q, core=True, positive_tail=True),
             _spec(rng, s2, q + 1, q, core=True, positive_tail=True)]
    if num_train >= 3:
        r, c = _counts_for(rng, s1, "above")
        train.append(_spec(rng, s1, r, c))
    if num_train >= 4:
        s = int(rng.integers(1, 14))
        train.append(_spec(rng, s, int(rng.integers(1, s + 1)), 0,
                           positive_tail=True))
    if num_train >= 5:
        s = int(rng.integers(1, 15))
        train.append(_spec(rng, s, 0, int(rng.integers(1, s + 1))))
    while len(train) < num_train:
        outcome = str(rng.choice(("equal", "below", "above", "empty_reference")))
        s = int(rng.integers(2 if outcome == "above" else 1, 15))
        r, c = _counts_for(rng, s, outcome)
        train.append(_spec(rng, s, r, c))
    allowed = ["equal", "below"]
    if num_train >= 3:
        allowed.append("above")
    if num_train >= 5:
        allowed.append("empty_reference")
    outcomes = [str(rng.choice(allowed)) for _ in range(num_test)]
    if num_test >= 2:
        outcomes[:2] = ["below", str(rng.choice([x for x in allowed if x != "below"]))]
        rng.shuffle(outcomes)
    train_spans = {spec["s"] for spec in train}
    test = []
    for i, outcome in enumerate(outcomes):
        choices = [s for s in range(1, 15) if outcome != "above" or s >= 2]
        if i == 0 or rng.random() < 0.5:
            choices = [s for s in choices if s not in train_spans]
        else:
            seen = [s for s in choices if s in train_spans]
            choices = seen or choices
        s = int(rng.choice(choices))
        r, c = _counts_for(rng, s, outcome)
        test.append(_spec(rng, s, r, c))
    # One core contains equality contrasts at least as wide as every test.
    train[0]["width"] = max(train[0]["width"], max(spec["width"] for spec in test))
    rng.shuffle(train)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, specs in (("train", train), ("test", test)):
        for spec in specs:
            for _ in range(16):
                grid = _make_input(rng, spec, pool)
                key = _key(grid)
                if key not in used:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
            used.add(key)
            puzzle[split].append({"input": grid, "output": solve(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_styles(grid, mask, s, destination):
    styles = set()
    for i, row in enumerate(grid):
        if mask[i] or i in (s, destination):
            continue
        values, counts = np.unique(row, return_counts=True)
        if len(values) == 2 and sorted(counts.tolist()) == [1, len(row) - 1]:
            minority = values[int(np.argmin(counts))]
            position = int(np.flatnonzero(row == minority)[0])
            styles.add("first" if position == 0 else
                       "last" if position == len(row) - 1 else "middle")
    return styles


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must have exactly input/output")
    facts = _facts(pair["input"])
    grid, s, d, mask = (facts[x] for x in ("grid", "s", "destination", "mask"))
    output = _grid(pair["output"])
    _require(output.shape == grid.shape, "shape must be preserved")
    accepted = facts["candidate"] >= facts["reference"]
    # Inspect row-by-row without invoking the public solver or render helper.
    for i, row in enumerate(output):
        if i == d:
            color = DELIMITER
        elif i == s or not mask[i]:
            color = BACKGROUND
        elif i < s or (s < i < d and accepted):
            color = REFERENCE
        else:
            color = CANDIDATE
        _require(np.all(row == color), f"wrong status at row {i}")
    _require(np.count_nonzero(output == DELIMITER) == grid.shape[1],
             "exactly one delimiter must move")
    _require(not np.array_equal(grid, output), "pair must change")
    facts["styles"] = _exception_styles(grid, mask, s, d)
    facts["positive_colors"] = set(int(grid[i, 0]) for i in np.flatnonzero(mask))
    facts["accepted"] = accepted
    return facts


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            facts = _inspect_pair(pair)
            key = _key(facts["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(facts)
    train, test = infos["train"], infos["test"]
    equal = [f for f in train if 1 <= f["reference"] == f["candidate"] <= 2]
    below = [f for f in train if f["reference"] == f["candidate"] + 1
             and f["candidate"] > 0]
    _require(any(a["candidate"] == b["candidate"] for a in equal for b in below),
             "missing matched low-equality/reference-threshold contrasts")
    _require(any(f["later"] > 0 for f in equal), "accepted core needs a protected tail")
    _require(any(f["candidate"] + f["later"] >= f["reference"] for f in below),
             "rejected core must defeat whole-far-side counting")
    for candidates in (equal, below):
        _require(any(f["styles"] == {"first", "middle", "last"}
                     and len(f["positive_colors"]) >= 2 for f in candidates),
                 "core needs complete-row and color-independent contrasts")
    _require(len({f["s"] for f in train}) >= 2, "training must vary ruler span")
    _require(test[0]["s"] not in {f["s"] for f in train}, "first test span must be held out")
    widest_test = max(f["grid"].shape[1] for f in test)
    _require(any(f["grid"].shape[1] >= widest_test
                 and f["styles"] == {"first", "middle", "last"} for f in train),
             "wide tests need equally wide full-row contrasts")
    if nt >= 3:
        _require(any(f["candidate"] > f["reference"] > 0 for f in train),
                 "missing strict-excess demonstration")
    if nt >= 4:
        _require(any(f["reference"] > 0 and f["candidate"] == 0 and f["later"] > 0
                     for f in train), "missing zero-candidate advance")
    if any(f["candidate"] > f["reference"] for f in test):
        _require(any(f["candidate"] > f["reference"] for f in train),
                 "strict excess was not demonstrated")
    if any(f["reference"] == 0 for f in test):
        _require(any(f["reference"] == 0 for f in train), "empty reference not demonstrated")
    if ne >= 2:
        _require({f["accepted"] for f in test} == {False, True},
                 "multiple tests need both decisions")
    # Concrete ablations preserve geometry and all conventions except the named mechanism.
    for mode in ("always", "never", "whole_far_side"):
        _require(any(not np.array_equal(_render(f, promotion=mode), pair["output"])
                     for f, pair in zip(train, puzzle["train"])),
                 f"training does not defeat promotion shortcut: {mode}")
    for predicate in (lambda g: g[:, 0] == g[:, -1],
                      lambda g: g[:, 0] == g[:, 1],
                      lambda g: g[:, -2] == g[:, -1],
                      lambda g: np.ones(g.shape[0], dtype=bool)):
        _require(any(not np.array_equal(_render(f, predicate(f["grid"])), pair["output"])
                     for f, pair in zip(train, puzzle["train"])),
                 "training does not defeat a partial/omitted row predicate")


def self_test():
    """Stream 42 seeds, all legal counts, ablations and metamorphic/boundary checks."""
    shapes, palettes, spans, patterns, structures, fingerprints = (set() for _ in range(6))
    outcomes = set()
    zero_rows = rich = no_tail = long_tail = 0
    accepted_tests = rejected_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)),
                 "seed replay differs")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _facts(pair["input"])
                g, s, d, mask = (f[k] for k in ("grid", "s", "destination", "mask"))
                shapes.add(g.shape)
                palettes.add(tuple(int(x) for x in np.unique(g)))
                spans.add(s)
                patterns.add((s, f["reference"], f["candidate"], f["later"]))
                structures.add((g.shape, s, tuple(bool(x) for x in mask)))
                decision = "below" if f["candidate"] < f["reference"] else (
                    "equal" if f["candidate"] == f["reference"] else "above")
                outcomes.add((split, decision))
                zero_rows += sum(bool(mask[i]) and int(g[i, 0]) == 0 for i in range(len(g)))
                rich += len(np.unique(g)) >= 7
                no_tail += d == len(g) - 1
                long_tail += len(g) - d - 1 > s
                if split == "test":
                    accepted_tests += f["candidate"] >= f["reference"]
                    rejected_tests += f["candidate"] < f["reference"]
                original = g.copy()
                _require(np.array_equal(solve(g), pair["output"]), "solver mismatch")
                _require(np.array_equal(g, original), "solver mutated its input")
                # Reassign every raw data color, keeping only the two structural rows.
                mapping = np.arange(10, dtype=np.int8)
                mapping[list(DATA_COLORS)] = np.roll(np.array(DATA_COLORS, dtype=np.int8), 1)
                recolored = mapping[g]
                recolored[s] = DELIMITER
                recolored[d] = BACKGROUND
                _require(np.array_equal(solve(recolored), pair["output"]),
                         "raw color identities changed the answer")
                # Permuting objects inside windows must permute their labels without changing acceptance.
                order = np.arange(len(g))
                order[:s] = order[:s][::-1]
                order[s + 1:d] = order[s + 1:d][::-1]
                _require(np.array_equal(solve(g[order]), np.asarray(pair["output"])[order]),
                         "row order changed the count decision")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(20261023 + 100 * nt + ne, nt, ne)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == ne,
                     "explicit counts changed")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    boundary_cases = (
        ([[1, 2, 1], [8, 8, 8], [3, 3, 3], [0, 0, 0], [0, 0, 0]],
         [0, 0, 5, 8, 2]),
        ([[0, 0, 0], [8, 8, 8], [2, 3, 2], [0, 0, 0]], [5, 0, 0, 8]),
        ([[1, 2, 1], [8, 8, 8], [3, 4, 3], [0, 0, 0]], [0, 0, 0, 8]),
    )
    for grid, colors in boundary_cases:
        _require(np.array_equal(solve(np.array(grid)), np.repeat(np.array(colors)[:, None], 3, axis=1)),
                 "zero-reference/candidate or structural-row boundary failed")
    boundary = np.full((30, 30), 3, dtype=np.int8)
    boundary[14] = DELIMITER
    boundary[29] = BACKGROUND
    result = solve(boundary)
    _require(np.all(result[:14] == REFERENCE) and np.all(result[15:29] == REFERENCE)
             and np.all(result[14] == BACKGROUND) and np.all(result[29] == DELIMITER),
             "maximal span and boundary destination failed")
    _require(len(fingerprints) == 42, "42 seeds did not produce distinct puzzles")
    _require(len(shapes) >= 70 and len(patterns) >= 100 and len(structures) >= 170,
             "insufficient geometric/count/mask diversity")
    _require(len(spans) >= 12 and len(palettes) >= 35, "insufficient span or palette diversity")
    _require({("test", x) for x in ("below", "equal", "above")} <= outcomes,
             "tests need varied comparison outcomes")
    _require(zero_rows >= 10 and rich >= 10 and no_tail >= 5 and long_tail >= 10,
             "missing color/empty-tail/multiple-window coverage")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "palettes": len(palettes), "spans": sorted(spans),
            "count_patterns": len(patterns), "row_mask_structures": len(structures),
            "accepted_tests": int(accepted_tests), "rejected_tests": int(rejected_tests),
            "uniform_zero_data_rows": zero_rows, "rich_palettes": rich,
            "empty_tails": no_tail, "tails_longer_than_window": long_tail,
            "legal_count_combinations": 30}
