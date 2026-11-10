"""Whole-row equality masks placed by a bottom-anchored unit shear.

No files, example lookup, or random state are used by solve. Generation and
self_test retain no execution traces and write no artifacts.
"""

import hashlib

import numpy as np


MARKER = 5
BACKGROUND = 0


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _counts(num_train, num_test):
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _grid(value):
    try:
        grid = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("grid must be a rectangular integer array") from error
    if (grid.ndim != 2 or grid.dtype.kind not in "iu"
            or not all(1 <= n <= 30 for n in grid.shape)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("grid must have integer colors 0..9 and dimensions 1..30")
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Classify whole rows, then translate row r right by h-1-r cells."""
    grid = _grid(input_grid)
    height, width = grid.shape
    if height + width - 1 > 30:
        raise ValueError("sheared canvas would exceed 30 columns")
    result = np.full((height, height + width - 1), BACKGROUND, dtype=np.int8)
    positive = np.all(grid == grid[:, :1], axis=1)
    for row in np.flatnonzero(positive):
        start = height - 1 - int(row)
        result[row, start:start + width] = MARKER
    return result


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _mask(grid):
    return tuple(len(set(map(int, row))) == 1 for row in grid)


def _answer_key(shape, mask):
    return (*shape, tuple(bool(v) for v in mask))


def _shape(rng):
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return int(rng.integers(2, 9)), int(rng.integers(2, 11))
    if kind == 1:
        choices = ((30, 1), (29, 2), (2, 29), (1, 30),
                   (15, 16), (16, 15), (3, 28), (28, 3))
        return choices[int(rng.integers(len(choices)))]
    if kind == 2:
        height = int(rng.integers(1, 7))
        return height, int(rng.integers(1, 32 - height))
    height = int(rng.integers(1, 31))
    return height, int(rng.integers(1, 32 - height))


def _random_mask(rng, height, width):
    if width == 1:
        return np.ones(height, dtype=bool)
    kind = int(rng.integers(0, 10))
    if kind == 0:
        return np.zeros(height, dtype=bool)
    if kind == 1:
        return np.ones(height, dtype=bool)
    if kind == 2:
        return np.arange(height) % 2 == int(rng.integers(2))
    if kind == 3:
        mask = np.zeros(height, dtype=bool)
        start = int(rng.integers(height))
        stop = int(rng.integers(start + 1, height + 1))
        mask[start:stop] = True
        return mask
    return rng.random(height) < float(rng.choice((0.25, 0.4, 0.6, 0.75)))


def _negative_row(rng, palette, width, style=None):
    a, b = map(int, rng.choice(palette, 2, replace=False))
    style = style or str(rng.choice(("exception", "alternating", "runs", "random", "balanced")))
    if style in ("first", "middle", "last", "exception"):
        row = np.full(width, a, dtype=np.int8)
        if style == "first":
            position = 0
        elif style == "last" or width == 2:
            position = width - 1
        else:
            position = int(rng.integers(1, width - 1))
        row[position] = b
    elif style == "alternating":
        row = np.where(np.arange(width) % 2, a, b).astype(np.int8)
    elif style in ("runs", "balanced"):
        cut = int(rng.integers(1, width)) if style == "runs" else width // 2
        row = np.array([a] * cut + [b] * (width - cut), dtype=np.int8)
        if style == "balanced":
            rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, width), dtype=np.int8)
        if np.all(row == row[0]):
            row[-1] = a if int(row[0]) != a else b
    return row


def _make_grid(rng, shape, mask, palette, forced=None, positive_colors=None):
    forced = forced or {}
    positive_colors = positive_colors or {}
    grid = np.empty(shape, dtype=np.int8)
    previous = {True: [], False: []}
    for r, positive in enumerate(mask):
        positive = bool(positive)
        if positive:
            grid[r] = positive_colors.get(r, int(rng.choice(palette)))
        elif r in forced:
            grid[r] = _negative_row(rng, palette, shape[1], forced[r])
        elif previous[False] and rng.random() < 0.18:
            grid[r] = previous[False][int(rng.integers(len(previous[False])))]
        else:
            grid[r] = _negative_row(rng, palette, shape[1])
        previous[positive].append(grid[r].copy())
    return grid


def _palette(rng, pool):
    count = int(rng.integers(2, len(pool) + 1))
    return np.asarray(rng.choice(pool, count, replace=False), dtype=np.int8)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = np.asarray(rng.choice(10, int(rng.integers(2, 9)), replace=False), dtype=np.int8)
    shift = str(rng.choice(("geometry", "arrangement", "palette")))
    height = int(rng.integers(5, 10))
    width = int(rng.integers(3, 13))
    first_shape = (height, width)
    first_mask = np.ones(height, dtype=bool)
    negatives = rng.choice(np.arange(1, height - 1), 3, replace=False)
    first_mask[negatives] = False
    styles = rng.permutation(("first", "middle", "last"))
    first_forced = dict(zip(map(int, negatives), map(str, styles)))
    second_height = int(rng.choice([h for h in range(5, 10) if h != height]))
    second_shape = (second_height, int(rng.integers(3, 13)))
    second_mask = rng.random(second_height) < 0.55
    second_mask[[0, -1]] = False
    second_mask[1] = second_mask[-2] = True
    puzzle = {"train": [], "test": []}
    seen = set()

    def append(split, shape, mask, palette, forced=None, positive_colors=None):
        for _ in range(64):
            grid = _make_grid(rng, shape, mask, palette, forced, positive_colors)
            key = _key(grid)
            if key not in seen:
                seen.add(key)
                puzzle[split].append({"input": grid, "output": solve(grid)})
                return
        raise RuntimeError("could not construct a distinct input within 64 attempts")

    # Two distinct uniform colors disprove literal-color selection. Three
    # isolated exceptions disprove endpoint-only and prefix/suffix shortcuts.
    append("train", first_shape, first_mask, pool, first_forced,
           {0: int(pool[0]), height - 1: int(pool[1])})
    append("train", second_shape, second_mask, _palette(rng, pool))
    for _ in range(2, num_train):
        for _attempt in range(64):
            shape = first_shape if rng.random() < 0.25 else _shape(rng)
            try:
                append("train", shape, _random_mask(rng, *shape), _palette(rng, pool))
            except RuntimeError:
                # A tiny shape can exhaust its finite set of inputs. Resample
                # the configuration rather than silently duplicating a pair.
                continue
            break
        else:
            raise RuntimeError("could not construct an additional training example")

    train_shapes = {pair["input"].shape for pair in puzzle["train"]}
    train_answers = {_answer_key(pair["input"].shape, _mask(pair["input"]))
                     for pair in puzzle["train"]}
    test_answers = set()
    unseen = sorted(set(range(10)) - set(map(int, pool)))
    for i in range(num_test):
        for _attempt in range(128):
            if i == 0 and shift != "geometry":
                shape = (first_shape, second_shape)[int(rng.integers(2))]
            elif i == 0 or rng.random() < 0.7:
                shape = _shape(rng)
            else:
                shape = (first_shape, second_shape)[int(rng.integers(2))]
            if i == 0 and shift == "geometry" and shape in train_shapes:
                continue
            mask = _random_mask(rng, *shape)
            answer_key = _answer_key(shape, mask)
            if answer_key in test_answers:
                continue
            if i == 0 and shift == "arrangement" and answer_key in train_answers:
                continue
            if i == 0 and shift == "palette" and not np.any(mask):
                continue
            palette = _palette(rng, pool)
            positive_colors = None
            if (i == 0 and shift == "palette") or (i > 0 and rng.random() < 0.3):
                novel = int(rng.choice(unseen))
                palette = np.array([novel, *palette], dtype=np.int8)
                if np.any(mask):
                    positive_colors = {int(np.flatnonzero(mask)[0]): novel}
            try:
                append("test", shape, mask, palette, positive_colors=positive_colors)
            except RuntimeError:
                continue
            test_answers.add(answer_key)
            break
        else:
            raise RuntimeError("could not construct a test shift within 128 attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _exception_style(row):
    if len(row) < 3:
        return None
    values, counts = np.unique(row, return_counts=True)
    if len(values) != 2 or sorted(map(int, counts)) != [1, len(row) - 1]:
        return None
    position = int(np.flatnonzero(row == values[int(np.argmin(counts))])[0])
    return "first" if position == 0 else "last" if position == len(row) - 1 else "middle"


def validate_generated_puzzle(puzzle):
    """Check cells directly, plus the generator's demonstration guarantees."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, contrasts, positive_colors = set(), set(), set()
    heights, train_shapes, train_answers, test_answers, train_colors = set(), set(), set(), set(), set()
    ablation_a = ablation_b = wrong_direction = False
    span_demo = blank_edges_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            h, w = grid.shape
            _require(h + w - 1 <= 30 and out.shape == (h, h + w - 1), "incorrect canvas dimensions")
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            mask = _mask(grid)
            for r, positive in enumerate(mask):
                # Scalar cell-by-cell oracle, separate from solve's vectorized predicate.
                for c in range(h + w - 1):
                    expected = MARKER if positive and h - 1 - r <= c < h - 1 - r + w else BACKGROUND
                    _require(int(out[r, c]) == expected, "row predicate, displacement, or padding is incorrect")
            _require(int(np.count_nonzero(out == MARKER)) == sum(mask) * w, "incorrect marker area")
            answer = _answer_key(grid.shape, mask)
            if split == "train":
                heights.add(h)
                train_shapes.add(grid.shape)
                train_answers.add(answer)
                train_colors.update(map(int, grid.flat))
                span_demo |= h > 1 and mask[0] and mask[-1]
                blank_edges_demo |= not mask[0] and not mask[-1] and any(mask)
                for r, positive in enumerate(mask):
                    if positive:
                        positive_colors.add(int(grid[r, 0]))
                        ablation_b |= h - 1 - r > 0
                        wrong_direction |= h - 1 - r != r
                    else:
                        style = _exception_style(grid[r])
                        if style:
                            contrasts.add(style)
                    # Omitting A would copy the unclassified payload into B's canvas.
                    label = MARKER if positive else BACKGROUND
                    ablation_a |= any(int(v) != label for v in grid[r])
            else:
                _require(answer not in test_answers, "tests repeat an output")
                test_answers.add(answer)
    _require(len(heights) >= 2, "training needs two heights to establish dimension dependence")
    _require(contrasts == {"first", "middle", "last"}, "missing whole-row equality contrasts")
    _require(len(positive_colors) >= 2, "uniform rows must use more than one input color")
    _require(span_demo and blank_edges_demo, "missing full-extent or blank-edge demonstration")
    _require(ablation_a and ablation_b and wrong_direction, "a parent contribution is not observable")
    first = np.asarray(puzzle["test"][0]["input"])
    _require(first.shape not in train_shapes
             or _answer_key(first.shape, _mask(first)) not in train_answers
             or bool(set(map(int, first.flat)) - train_colors), "missing train/test shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name])
                digest.update(bytes(grid.shape))
                digest.update(grid.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, in-memory checks over 42 seeds, all counts, and every legal shape."""
    shapes, palettes, palette_sizes, masks, fingerprints, structures = set(), set(), set(), set(), set(), set()
    all_positive = all_negative = single_rows = single_columns = boundary = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed replay differs")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                shapes.add(grid.shape)
                palette = tuple(map(int, np.unique(grid)))
                palettes.add(palette)
                palette_sizes.add(len(palette))
                mask = _mask(grid)
                masks.add(mask)
                all_positive += all(mask)
                all_negative += not any(mask)
                single_rows += grid.shape[0] == 1
                single_columns += grid.shape[1] == 1
                boundary += sum(grid.shape) - 1 == 30
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in grid.flat)
                structures.add((grid.shape, hashlib.sha256(normalized).hexdigest()))
                _require(np.array_equal(solve(grid.copy()), pair["output"]), "solve disagreement")
        anchor = puzzle["train"][0]["input"]
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(permutation[anchor]), solve(anchor)), "literal input color affected equality")
        changed = anchor.copy()
        changed[0, 0] = (int(changed[0, 0]) + 1) % 10
        expected = solve(anchor)
        expected[0] = 0
        _require(np.array_equal(solve(changed), expected), "one-cell equality change did not remove exactly one bar")
        negative = int(np.flatnonzero(~np.array(_mask(anchor)))[0])
        values, counts = np.unique(anchor[negative], return_counts=True)
        changed = anchor.copy()
        changed[negative] = values[int(np.argmax(counts))]
        expected = solve(anchor)
        start = anchor.shape[0] - 1 - negative
        expected[negative, start:start + anchor.shape[1]] = MARKER
        _require(np.array_equal(solve(changed), expected), "repairing a mixed row did not add exactly one shifted bar")

    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * a + b, a, b)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "count request changed")
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None), ("4", 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    shape_checks = 0
    rng = np.random.default_rng(4261)
    for h in range(1, 31):
        for w in range(1, 32 - h):
            grid = rng.integers(0, 10, size=(h, w), dtype=np.int8)
            grid[0] = 0
            out = solve(grid)
            expected = np.array([
                [0] * (h - 1 - r)
                + [5 if len(set(map(int, grid[r]))) == 1 else 0] * w
                + [0] * r for r in range(h)], dtype=np.int8)
            _require(np.array_equal(out, expected), "boundary-shape scalar check failed")
            shape_checks += 1
    _require(len(fingerprints) == 42 and len(shapes) >= 45, "insufficient seed or geometric diversity")
    _require(len(palettes) >= 30 and len(palette_sizes) >= 6, "insufficient palette diversity")
    _require(len(masks) >= 50 and len(structures) >= 140, "insufficient structural diversity")
    _require(all_positive >= 3 and all_negative >= 3, "missing extreme row masks")
    _require(single_rows and single_columns and boundary >= 5, "missing boundary geometry")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "input_palettes": len(palettes), "palette_sizes": sorted(palette_sizes),
            "row_masks": len(masks), "color_normalized_input_structures": len(structures),
            "all_positive_pairs": all_positive, "all_negative_pairs": all_negative,
            "single_row_pairs": single_rows, "single_column_pairs": single_columns,
            "width_30_outputs": boundary, "legal_count_combinations": legal_counts,
            "legal_shape_checks": shape_checks}
