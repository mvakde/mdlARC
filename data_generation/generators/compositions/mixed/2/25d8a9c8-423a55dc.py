"""Whole-row equality feeds a lowest-positive-row anchored left shear.

Only NumPy and the standard library are used. Generation and self_test keep
their diagnostics in memory; this module does not read or write any files.
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
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or grid.dtype.kind not in "iu" or np.any((grid < 0) | (grid > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid


def _row_mask(grid):
    return np.all(grid == grid[:, :1], axis=1)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention from the fixed rule and this input alone."""
    grid = _grid(input_grid)
    positive = np.flatnonzero(_row_mask(grid))
    result = np.zeros(grid.shape, dtype=np.int8)
    if positive.size:
        anchor = int(positive[-1])
        width = grid.shape[1]
        for row in positive:
            length = max(0, width - (anchor - int(row)))
            result[row, :length] = POSITIVE
    return result


def _palette(rng, previous, calibration):
    if calibration:
        others = rng.permutation([c for c in range(10) if c not in (0, 5)])
        return np.array([0, 5, *others[:int(rng.integers(1, 9))]], dtype=np.int8)
    if previous and rng.random() < 0.3:
        return previous[int(rng.integers(len(previous)))].copy()
    return rng.choice(10, size=int(rng.integers(2, 11)), replace=False).astype(np.int8)


def _mixed_row(rng, palette, width, style=None, base=None):
    a = int(rng.choice(palette)) if base is None else int(base)
    b = int(rng.choice([c for c in palette if c != a]))
    style = style or str(rng.choice(("exception", "alternating", "runs", "balanced", "random")))
    if style in ("first", "middle", "last", "exception"):
        row = np.full(width, a, dtype=np.int8)
        if style == "first":
            position = 0
        elif style == "last":
            position = width - 1
        elif style == "middle":
            position = int(rng.integers(1, width - 1))
        else:
            position = int(rng.integers(width))
        row[position] = b
    elif style == "alternating":
        row = np.where(np.arange(width) % 2, b, a).astype(np.int8)
    elif style == "runs":
        cut = int(rng.integers(1, width))
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
    elif style == "balanced":
        row = np.array([a] * (width // 2) + [b] * (width - width // 2), dtype=np.int8)
        rng.shuffle(row)
    else:
        row = rng.choice(palette, size=width).astype(np.int8)
        if np.all(row == row[0]):
            row[-1] = b if int(row[0]) == a else a
    return row


def _make_input(rng, mask, width, palette, calibration=False):
    grid = np.empty((len(mask), width), dtype=np.int8)
    previous_mixed = []
    common = int(rng.choice(palette))
    for r, positive in enumerate(mask):
        if positive:
            grid[r] = common if rng.random() < 0.4 else int(rng.choice(palette))
        elif previous_mixed and rng.random() < 0.14:
            grid[r] = previous_mixed[int(rng.integers(len(previous_mixed)))]
        else:
            grid[r] = _mixed_row(rng, palette, width)
            previous_mixed.append(grid[r].copy())
    if calibration:
        rows = np.flatnonzero(mask)
        colors = np.array([0, 5, int(palette[2])], dtype=np.int8)
        rng.shuffle(colors)
        for r, color in zip(rows, colors):
            grid[r] = color
        anchor = int(rows[-1])
        styles = ["first", "middle", "last"]
        rng.shuffle(styles)
        # These negatives are below the true anchor. Accepting even one moves
        # the anchor and leaves a false positive, so clipping cannot hide it.
        for r, style, base in zip(range(anchor + 1, anchor + 4), styles, colors):
            grid[r] = _mixed_row(rng, palette, width, style, int(base))
    _require(np.array_equal(_row_mask(grid), mask), "row constructor changed the mask")
    return grid


def _core_plans(rng):
    width = 30 if rng.random() < 0.12 else int(rng.integers(5, 16))
    gap = int(rng.integers(3, min(6, width - 2) + 1))
    top, tail = int(rng.integers(0, 3)), int(rng.integers(3, 6))
    anchor = top + gap
    calibration = np.zeros(anchor + tail + 1, dtype=bool)
    calibration[[top, anchor - 1, anchor]] = True
    clip_width = int(rng.choice([w for w in range(3, 11) if w != width]))
    overshoot = int(rng.integers(1, 4))
    clip_top, clip_tail = int(rng.integers(0, 3)), int(rng.integers(1, 4))
    clip_anchor = clip_top + clip_width + overshoot
    height = max(8, clip_anchor + clip_tail + 1)
    if height == len(calibration):
        height += 1
    clipping = np.zeros(height, dtype=bool)
    distances = [0, 1, clip_width - 1, clip_width, clip_width + overshoot]
    clipping[[clip_anchor - d for d in distances]] = True
    return [(calibration, width, True), (clipping, clip_width, False)]


def _random_plan(rng, training):
    height = int(rng.integers(4, 24 if training else 31))
    width = 30 if rng.random() < 0.12 else int(rng.integers(3, 24))
    if not training and rng.random() < 0.22:
        category = str(rng.choice(("all", "none", "single")))
        mask = np.full(height, category == "all", dtype=bool)
        if category == "single":
            mask[int(rng.integers(height))] = True
    else:
        mask = rng.random(height) < float(rng.choice((0.25, 0.45, 0.65, 0.8)))
        if training:
            # Extra demonstrations retain both mechanisms; extreme cases are
            # reserved as one possible compositional test shift.
            if np.count_nonzero(mask) < 3:
                mask[rng.choice(height, size=3, replace=False)] = True
            if np.all(mask):
                mask[int(rng.integers(height))] = False
    return mask, width, False


def _span(mask):
    rows = np.flatnonzero(mask)
    return int(rows[-1] - rows[0]) if rows.size else 0


def _shift_plan(rng, train, regime):
    if regime == "span":
        span = max(_span(_row_mask(p["input"])) for p in train)
        span += int(rng.integers(1, 4))
        top = int(rng.integers(0, min(2, 29 - span) + 1))
        tail = int(rng.integers(0, min(3, 29 - span - top) + 1))
        mask = rng.random(top + span + tail + 1) < float(rng.choice((0.3, 0.6)))
        mask[:top] = False
        mask[top + span + 1:] = False
        mask[[top, top + span - 1, top + span]] = True
        mask[top + 1] = False
        width = int(rng.integers(3, 31))
    elif regime == "narrow":
        height = int(rng.integers(6, 31))
        width = 2
        anchor = int(rng.integers(3, height))
        mask = rng.random(height) < float(rng.choice((0.3, 0.6)))
        mask[anchor + 1:] = False
        mask[[anchor - 3, anchor - 1, anchor]] = True
        mask[anchor - 2] = False
    else:
        # A new two-positive arrangement combines the calibrated predicate and
        # shear, without allowing the primary test to bypass either stage.
        height, width = int(rng.integers(5, 31)), int(rng.integers(3, 31))
        anchor = int(rng.integers(2, height - 1))
        span = int(rng.integers(1, min(anchor, width - 1) + 1))
        mask = np.zeros(height, dtype=bool)
        mask[[anchor - span, anchor]] = True
    return mask, width, False


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("span", "narrow", "mask")))
    cores = _core_plans(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs, test_answers, palettes = set(), set(), []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(64):
                if split == "train":
                    plan = cores[index] if index < 2 else _random_plan(rng, True)
                elif index == 0:
                    plan = _shift_plan(rng, puzzle["train"], regime)
                else:
                    plan = _random_plan(rng, False)
                mask, width, calibration = plan
                palette = _palette(rng, palettes, calibration)
                grid = _make_input(rng, mask, width, palette, calibration)
                output = solve(grid)
                signature, answer = _key(grid), _key(output)
                if (signature in seen_inputs or np.array_equal(grid, output)
                        or (split == "test" and answer in test_answers)):
                    continue
                puzzle[split].append({"input": grid, "output": output})
                palettes.append(palette)
                seen_inputs.add(signature)
                if split == "test":
                    test_answers.add(answer)
                break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(grid, classifier="equal", shear="bottom"):
    """Finite diagnostic competitors, never used to construct target outputs."""
    equal = _row_mask(grid)
    masks = {
        "equal": equal,
        "all": np.ones(len(grid), dtype=bool),
        "none": np.zeros(len(grid), dtype=bool),
        "endpoints": grid[:, 0] == grid[:, -1],
        "first_two": grid[:, 0] == grid[:, 1],
        "last_two": grid[:, -1] == grid[:, -2],
        "nonzero": equal & (grid[:, 0] != 0),
        "five": equal & (grid[:, 0] == 5),
        "majority": np.array([max(np.bincount(row, minlength=10)) >= len(row) - 1
                               for row in grid]),
    }
    rows = np.flatnonzero(masks[classifier])
    result = np.zeros(grid.shape, dtype=np.int8)
    if not rows.size:
        return result
    anchor, width = int(rows[-1]), grid.shape[1]
    for index, r in enumerate(rows):
        distance = anchor - int(r)
        if shear == "none":
            distance = 0
        elif shear == "canvas":
            distance = len(grid) - 1 - int(r)
        elif shear == "rank":
            distance = len(rows) - 1 - index
        elif shear == "double":
            distance *= 2
        elif shear == "rigid":
            distance = 1
        elif shear == "block":
            bottom = int(r)
            while bottom + 1 < len(grid) and masks[classifier][bottom + 1]:
                bottom += 1
            distance = bottom - int(r)
        length = max(0, width - distance)
        if shear == "right":
            result[r, min(distance, width):] = POSITIVE
        elif shear == "wrap":
            result[r, :] = POSITIVE
        else:
            result[r, :length] = POSITIVE
    return result


def _exception_style(row):
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(counts.tolist()) != [1, len(row) - 1]:
        return None
    exception = values[int(np.argmin(counts))]
    position = int(np.flatnonzero(row == exception)[0])
    return "first" if position == 0 else "last" if position == len(row) - 1 else "middle"


def validate_generated_puzzle(puzzle):
    """Check geometry, causal witnesses, demonstrations, and a visible shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, test_answers, train_facts, test_facts = set(), set(), [], []
    contrasts, uniform_colors, heights, widths = set(), set(), set(), set()
    has_gap = has_tail = has_threshold = has_visible_upper = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "canvas dimensions changed")
            signature = _key(grid)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            mask = _row_mask(grid)
            rows = np.flatnonzero(mask)
            anchor = int(rows[-1]) if rows.size else -1
            width = grid.shape[1]
            # Per-cell defining equation, checked independently of solve's slice code.
            for r, row in enumerate(grid):
                for c in range(width):
                    expected = POSITIVE if bool(mask[r]) and c + anchor - r < width else BACKGROUND
                    _require(int(output[r, c]) == expected, "predicate/shear/clipping equation violated")
            facts = (width, _span(mask), int(rows.size))
            if split == "test":
                answer = _key(output)
                _require(answer not in test_answers, "duplicate test answer")
                test_answers.add(answer)
                test_facts.append(facts)
                continue
            train_facts.append(facts)
            _require(3 <= rows.size < len(grid), "training must expose both row classes and shear")
            heights.add(len(grid))
            widths.add(width)
            distances = {anchor - int(r) for r in rows}
            has_visible_upper |= any(1 <= d < width for d in distances)
            has_gap |= bool(np.any(np.diff(rows) > 1))
            has_tail |= anchor < len(grid) - 1
            has_threshold |= {width - 1, width}.issubset(distances) and any(d > width for d in distances)
            for r, row in enumerate(grid):
                if mask[r]:
                    if anchor - r < width:
                        uniform_colors.add(int(row[0]))
                elif r > anchor:
                    style = _exception_style(row)
                    if style is not None:
                        contrasts.add(style)
    _require(len(heights) >= 2 and len(widths) >= 2, "training lacks dimension variation")
    _require(has_gap and has_tail and has_threshold and has_visible_upper, "missing shear witnesses")
    _require({0, 5}.issubset(uniform_colors) and len(uniform_colors) >= 3, "color abstraction not exposed")
    _require(contrasts == {"first", "middle", "last"}, "missing equality counterexamples below anchor")
    primary = _grid(puzzle["test"][0]["input"])
    _require(np.any(_row_mask(primary)) and np.any(~_row_mask(primary)), "primary test needs both row classes")
    _require(not np.array_equal(_alternative(primary, "equal", "none"), puzzle["test"][0]["output"]),
             "primary test bypasses the shear")
    _require(any(w < min(f[0] for f in train_facts)
                 or span > max(f[1] for f in train_facts)
                 or count < min(f[2] for f in train_facts)
                 or count == len(p["input"])
                 for (w, span, count), p in zip(test_facts, puzzle["test"])), "missing train/test shift")
    classifiers = ("all", "none", "endpoints", "first_two", "last_two", "nonzero", "five", "majority")
    shears = ("none", "canvas", "rank", "double", "rigid", "block", "right", "wrap")
    for classifier, shear in [(c, "bottom") for c in classifiers] + [("equal", s) for s in shears]:
        _require(any(not np.array_equal(_alternative(_grid(p["input"]), classifier, shear), p["output"])
                     for p in puzzle["train"]), f"training admits shortcut {classifier}/{shear}")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming checks over 42 seeds; return small descriptive metrics."""
    fingerprints, shapes, masks, palettes, structures = set(), set(), set(), set(), set()
    shifts, erased, one_cell, zero_rows, marker_rows = set(), 0, 0, 0, 0
    extreme_tests = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterministic generation")
        fingerprints.add(_fingerprint(puzzle))
        train = puzzle["train"]
        max_span = max(_span(_row_mask(p["input"])) for p in train)
        min_width = min(p["input"].shape[1] for p in train)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                shapes.add(grid.shape)
                mask = _row_mask(grid)
                masks.add(tuple(bool(x) for x in mask))
                palettes.add(tuple(int(x) for x in np.unique(grid)))
                labels = {}
                normalized = bytes(labels.setdefault(int(x), len(labels)) for x in grid.flat)
                structures.add((grid.shape, hashlib.sha256(normalized).hexdigest()))
                sizes = np.count_nonzero(output == POSITIVE, axis=1)
                erased += int(np.count_nonzero(mask & (sizes == 0)))
                one_cell += int(np.count_nonzero(sizes == 1))
                zero_rows += int(np.count_nonzero(mask & (grid[:, 0] == 0)))
                marker_rows += int(np.count_nonzero(mask & (grid[:, 0] == 5)))
                # Any bijective input recoloring preserves both stages.
                relabeled = np.random.default_rng(seed + 1000).permutation(10)
                _require(np.array_equal(solve(relabeled[grid]), output), "color identities affected output")
                # Reversing each row preserves equality.
                _require(np.array_equal(solve(grid[:, ::-1]), output), "within-row arrangement affected mask")
                if split == "test":
                    if _span(mask) > max_span:
                        shifts.add("span")
                    if grid.shape[1] < min_width:
                        shifts.add("narrow")
                    if np.all(mask) or mask.sum() < min(np.count_nonzero(_row_mask(p["input"])) for p in train):
                        shifts.add("mask")
        # Appending a mixed row does not move the positive anchor.
        grid = train[0]["input"]
        if len(grid) < 30:
            mixed = np.zeros((1, grid.shape[1]), dtype=np.int8)
            mixed[0, -1] = 1
            extended = np.vstack((grid, mixed))
            _require(np.array_equal(solve(extended)[:-1], solve(grid)), "canvas bottom used as anchor")
        # Additional tests may exercise degenerate masks; the primary test is
        # always mixed and contains a visibly sheared positive above its anchor.
        multiple = generate_puzzle(seed, 5, 4)
        for pair in multiple["test"][1:]:
            mask = _row_mask(pair["input"])
            if np.all(mask) or mask.sum() <= 1:
                extreme_tests.add("all" if np.all(mask) else "none" if not np.any(mask) else "single")
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 50 and len(masks) >= 90 and len(structures) >= 150, "structural diversity")
    _require(len(palettes) >= 30 and zero_rows > 30 and marker_rows > 30, "palette diversity")
    _require(erased > 50 and one_cell >= 42, "clipping boundaries not covered")
    _require(shifts == {"span", "narrow", "mask"} and extreme_tests == {"all", "none", "single"}, "shift diversity")
    _require(max(max(s) for s in shapes) == 30, "ARC boundary not exercised")
    legal_counts = 0
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            puzzle = generate_puzzle(1234, n, m)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (n, m), "explicit counts changed")
            legal_counts += 1
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-computed edge cases independent of the constructor's shape ranges.
    for grid, expected in (
        ([[8]], [[5]]),
        ([[2], [0], [5]], [[0], [0], [5]]),
        ([[1, 2], [2, 1]], [[0, 0], [0, 0]]),
        ([[4, 4, 4], [1, 2, 1], [0, 0, 0], [5, 6, 5]],
         [[5, 0, 0], [0, 0, 0], [5, 5, 5], [0, 0, 0]]),
    ):
        _require(np.array_equal(solve(np.array(grid)), expected), "hand-computed edge case")
    # A new monochromatic lower row changes the anchor, and hence other rows.
    before = np.array([[7, 7, 7, 7, 7], [1, 2, 1, 1, 1], [3, 3, 3, 3, 3],
                       [4, 4, 9, 4, 4], [6, 6, 6, 6, 8]])
    after = before.copy()
    after[-1] = 6
    _require(np.array_equal(solve(before)[0], [5, 5, 5, 0, 0]), "old-anchor counterfactual")
    _require(np.array_equal(solve(after)[0], [5, 0, 0, 0, 0]), "A did not control B's anchor")
    return {"seeds": 42, "pairs": 210, "supplementary_multi_test_pairs": 378,
            "distinct_puzzles": len(fingerprints),
            "shapes": len(shapes), "row_masks": len(masks), "palettes": len(palettes),
            "color_normalized_inputs": len(structures), "fully_erased_positive_rows": erased,
            "one_cell_rows": one_cell, "monochromatic_zero_rows": zero_rows,
            "monochromatic_five_rows": marker_rows, "shift_types": sorted(shifts),
            "extreme_test_types": sorted(extreme_tests), "legal_count_combinations": legal_counts,
            "independent_solver": False, "blind_solve": False}
