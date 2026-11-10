"""Generate aligned two-panel puzzles whose output marks joint background."""

import numpy as np


_SOURCE_ROWS = (
    (("0330", "0303", "0030", "3333", "1111", "1100", "1100", "0110"), ("0000", "0020", "0002", "0000")),
    (("3333", "0330", "0033", "3000", "0001", "0001", "0100", "1001"), ("0000", "2000", "2000", "0220")),
    (("3333", "3000", "3033", "3303", "1110", "0111", "1011", "0111"), ("0000", "0000", "0200", "0000")),
    (("0000", "0330", "0000", "3003", "0001", "1011", "1111", "0101"), ("2220", "0000", "0000", "0020")),
    (("0303", "3330", "0003", "3330", "0011", "0011", "0100", "1100"), ("2000", "0000", "2020", "0002")),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _signature(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _row_signature(rows):
    return ((len(rows), len(rows[0])), tuple(int(v) for row in rows for v in row))


_SOURCE_INPUTS = frozenset(_row_signature(i) for i, _ in _SOURCE_ROWS)
_SOURCE_PAIRS = frozenset((_row_signature(i), _row_signature(o)) for i, o in _SOURCE_ROWS)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _random_shape(rng):
    if rng.random() < 0.24:
        return 4, 4
    return int(rng.integers(3, 16)), int(rng.integers(3, 19))


def _shape(rng, base, variable, force_change=False):
    if not variable or (not force_change and rng.random() < 0.25):
        return base
    shape = _random_shape(rng)
    if force_change and shape == base:
        shape = (shape[0], 3 if shape[1] != 3 else 4)
    return shape


def _pattern(rng, height, width):
    """Sample one panel independently, with varied geometry and density."""
    style = int(rng.integers(0, 6))
    mask = np.zeros((height, width), dtype=bool)
    if style == 0:
        return rng.random((height, width)) < rng.uniform(0.08, 0.9)
    if style == 1:
        for _ in range(int(rng.integers(1, 4))):
            r0, c0 = int(rng.integers(height)), int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            mask[r0:r1, c0:c1] = True
    elif style == 2:
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                mask[int(rng.integers(height)), :] = True
            else:
                mask[:, int(rng.integers(width))] = True
    elif style == 3:
        rr, cc = np.indices((height, width))
        stride = int(rng.integers(2, 6))
        signed = cc if rng.random() < 0.5 else -cc
        mask = (rr + signed + int(rng.integers(stride))) % stride == 0
        if rng.random() < 0.45:
            mask |= rng.random((height, width)) < 0.07
    elif style == 4:
        r, c = int(rng.integers(height)), int(rng.integers(width))
        steps = int(rng.integers(max(2, height + width), max(3, 3 * height * width // 2)))
        for _ in range(steps):
            mask[r, c] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r = min(height - 1, max(0, r + dr))
            c = min(width - 1, max(0, c + dc))
    else:
        r0, c0 = int(rng.integers(height - 1)), int(rng.integers(width - 1))
        r1 = int(rng.integers(r0 + 2, height + 1))
        c1 = int(rng.integers(c0 + 2, width + 1))
        mask[r0:r1, c0] = mask[r0:r1, c1 - 1] = True
        mask[r0, c0:c1] = mask[r1 - 1, c0:c1] = True
    return mask


def _state_masks(rng, height, width, kind):
    area = height * width
    if kind in ("anchor", "sparse", "broad"):
        if kind == "anchor":
            empty_count = (int(rng.integers(1, 3)) if rng.random() < 0.4
                           else int(rng.integers(1, max(2, area // 2))))
        elif kind == "sparse":
            empty_count = int(rng.integers(1, min(area - 2, max(3, area // 9)) + 1))
        else:
            empty_count = int(rng.integers(max(2, int(0.56 * area)),
                                           max(3, int(0.88 * area)) + 1))
        states = np.empty(area, dtype=np.int8)
        states[:empty_count] = 0
        if kind == "anchor":
            states[empty_count:empty_count + 3] = (1, 2, 3)
            states[empty_count + 3:] = rng.choice(
                4, size=area - empty_count - 3, p=(0.18, 0.28, 0.28, 0.26))
        else:
            states[empty_count:] = rng.integers(1, 4, size=area - empty_count)
        rng.shuffle(states)
        states = states.reshape(height, width)
        return (states == 1) | (states == 3), (states == 2) | (states == 3)

    upper = _pattern(rng, height, width)
    lower = _pattern(rng, height, width)
    if kind == "identical":
        lower = upper.copy()
    elif kind == "contained":
        lower = upper & lower
    elif kind == "disjoint":
        lower &= ~upper
    elif kind == "empty_panel":
        if rng.random() < 0.5:
            upper[:] = False
        else:
            lower[:] = False
    elif kind == "full_union":
        if not np.any(upper) or np.all(upper):
            upper = rng.random((height, width)) < 0.5
            upper.flat[0], upper.flat[1] = True, False
        lower = ~upper | (upper & (rng.random((height, width)) < rng.uniform(0.1, 0.8)))
    return upper, lower


def _kind(rng, split):
    kinds = ("independent", "sparse", "broad", "identical", "contained",
             "disjoint", "empty_panel", "full_union")
    weights = ((0.48, 0.16, 0.08, 0.07, 0.07, 0.06, 0.05, 0.03)
               if split == "train" else
               (0.36, 0.14, 0.10, 0.11, 0.09, 0.08, 0.06, 0.06))
    return kinds[int(rng.choice(len(kinds), p=weights))]


def _colors(rng, background, highlight, anchor_colors=None):
    available = [c for c in range(10) if c not in (background, highlight)]
    if anchor_colors is None:
        chosen = rng.choice(available, size=2, replace=False)
        return int(chosen[0]), int(chosen[1])
    choice = rng.random()
    if choice < 0.23:
        return anchor_colors[1], anchor_colors[0]
    if choice < 0.50:
        color = int(rng.choice(available))
        return color, color
    chosen = rng.choice(available, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _make_pair(rng, shape, background, highlight, colors, kind):
    height, width = shape
    upper_mask, lower_mask = _state_masks(rng, height, width, kind)
    upper = np.full(shape, background, dtype=np.int8)
    lower = np.full(shape, background, dtype=np.int8)
    upper[upper_mask], lower[lower_mask] = colors[0], colors[1]
    output = np.full(shape, background, dtype=np.int8)
    output[~upper_mask & ~lower_mask] = highlight
    return {"input": np.vstack((upper, lower)), "output": output}


def _fallback_pair(shape, background, highlight, colors, seen, require_anchor=False):
    """Deterministically find an unused layout if random draws collide."""
    height, width = shape
    area = height * width
    for code in range(1024):
        if require_anchor:
            states = np.array([0, 1, 2, 3] +
                              [((code >> (bit - 4)) & 1) for bit in range(4, area)],
                              dtype=np.int8)
            upper_mask = ((states == 1) | (states == 3)).reshape(shape)
            lower_mask = ((states == 2) | (states == 3)).reshape(shape)
        else:
            upper_mask = np.array([(code >> bit) & 1 for bit in range(area)],
                                  dtype=bool).reshape(shape)
            lower_mask = np.array([(code >> (area + bit)) & 1 for bit in range(area)],
                                  dtype=bool).reshape(shape)
        upper = np.full(shape, background, dtype=np.int8)
        lower = np.full(shape, background, dtype=np.int8)
        upper[upper_mask], lower[lower_mask] = colors[0], colors[1]
        output = np.full(shape, background, dtype=np.int8)
        output[~upper_mask & ~lower_mask] = highlight
        pair = {"input": np.vstack((upper, lower)), "output": output}
        signature = _signature(pair["input"])
        if signature not in seen and signature not in _SOURCE_INPUTS:
            return pair
    raise AssertionError("deterministic fallback found no unused input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    roles = rng.choice(10, size=2, replace=False)
    background, highlight = int(roles[0]), int(roles[1])
    base = _random_shape(rng)
    variable = bool(rng.random() < 0.68)
    anchor_colors = _colors(rng, background, highlight)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for attempt in range(24):
                anchor = split == "train" and index == 0
                shape = _shape(rng, base, variable,
                               force_change=(variable and split == "train" and index == 1))
                colors = anchor_colors if anchor else _colors(rng, background, highlight, anchor_colors)
                kind = "anchor" if anchor else _kind(rng, split)
                pair = _make_pair(rng, shape, background, highlight, colors, kind)
                signature = _signature(pair["input"])
                if signature not in seen and signature not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _fallback_pair(shape, background, highlight, colors, seen, anchor)
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= side <= 30 for side in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer dtype")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} has an illegal color")


def _analyze(pair, background, highlight):
    inp, out = pair["input"], pair["output"]
    height, width = out.shape
    _require(inp.shape == (2 * height, width), "input is not two aligned panels")
    upper, lower = inp[:height], inp[height:]
    _require(highlight not in inp, "highlight appears in an input")
    _require(len(set(np.unique(upper)) - {background}) <= 1, "upper has multiple foreground colors")
    _require(len(set(np.unique(lower)) - {background}) <= 1, "lower has multiple foreground colors")
    expected = np.full((height, width), background, dtype=np.int8)
    expected[(upper == background) & (lower == background)] = highlight
    _require(np.array_equal(out, expected), "output is not the joint-background mask")
    states = (upper != background).astype(np.int8) + 2 * (lower != background).astype(np.int8)
    return {"shape": (height, width),
            "states": set(int(x) for x in np.unique(states)),
            "empty": int(np.count_nonzero(states == 0)),
            "colors": (next((int(x) for x in np.unique(upper) if x != background), None),
                       next((int(x) for x in np.unique(lower) if x != background), None))}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} keys")
            _validate_grid(pair["input"], f"{label}.input")
            _validate_grid(pair["output"], f"{label}.output")
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(input_key not in _SOURCE_INPUTS and pair_key not in _SOURCE_PAIRS, "source copy")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

    first = puzzle["train"][0]
    output_colors = set(int(x) for x in np.unique(first["output"]))
    input_colors = set(int(x) for x in np.unique(first["input"]))
    _require(len(output_colors) == 2 and len(output_colors & input_colors) == 1,
             "training anchor does not identify background and highlight")
    background = next(iter(output_colors & input_colors))
    highlight = next(iter(output_colors - {background}))
    states_seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze(pair, background, highlight)
            if split == "train":
                states_seen.update(analysis["states"])
    _require(states_seen == {0, 1, 2, 3}, "training does not demonstrate all aligned states")


def _predict_from_training(train, test_input):
    """Use returned examples and the test input, without generator parameters."""
    first = train[0]
    input_colors = set(int(x) for x in np.unique(first["input"]))
    output_colors = set(int(x) for x in np.unique(first["output"]))
    background = next(iter(input_colors & output_colors))
    highlight = next(iter(output_colors - {background}))
    height = test_input.shape[0] // 2
    upper, lower = test_input[:height], test_input[height:]
    return np.where((upper == background) & (lower == background),
                    highlight, background)


def self_test() -> None:
    first = generate_puzzle(7301, 7, 3)
    second = generate_puzzle(7301, 7, 3)
    _require(all(np.array_equal(a[role], b[role]) for split in ("train", "test")
                 for a, b in zip(first[split], second[split]) for role in ("input", "output")),
             "generation is not deterministic")
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                    (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + index, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested, "requested count was ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_train": 2.5}):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")

    train_counts, test_counts, shapes, palettes, structures, input_keys = set(), set(), set(), set(), set(), set()
    varied_size = varied_colors = matching = contained = disjoint = empty_panel = blank = 0
    sparse_train = normal_test = broad_train = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            predicted = _predict_from_training(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]], pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]),
                     "a test output cannot be recovered from the training examples")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        anchor = puzzle["train"][0]
        bg = next(iter(set(np.unique(anchor["output"])) & set(np.unique(anchor["input"]))))
        hi = next(iter(set(np.unique(anchor["output"])) - {bg}))
        palettes.add((bg, hi))
        analyses = [_analyze(p, bg, hi) for s in ("train", "test") for p in puzzle[s]]
        varied_size += len({a["shape"] for a in analyses}) > 1
        varied_colors += len({a["colors"] for a in analyses}) > 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair, bg, hi)
                h, w = a["shape"]
                upper, lower = pair["input"][:h] != bg, pair["input"][h:] != bg
                area = h * w
                shapes.add(a["shape"])
                input_keys.add(_signature(pair["input"]))
                structures.add((a["shape"], tuple(upper.flat), tuple(lower.flat)))
                matching += np.array_equal(upper, lower)
                contained += bool(np.all(lower <= upper) or np.all(upper <= lower))
                disjoint += not bool(np.any(upper & lower))
                empty_panel += not bool(np.any(upper)) or not bool(np.any(lower))
                blank += a["empty"] == 0
                if split == "train":
                    sparse_train += a["empty"] <= 2
                    broad_train += a["empty"] / area >= 0.55
                else:
                    normal_test += 0.12 <= a["empty"] / area <= 0.50
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    _require(len(shapes) >= 35 and (4, 4) in shapes, "panel shapes lack diversity")
    _require(len(palettes) >= 20 and len(structures) >= 200 and len(input_keys) >= 200,
             "palettes or mask layouts lack diversity")
    _require(varied_size >= 15 and varied_colors >= 30, "within-puzzle variation is too rare")
    _require(all(value > 0 for value in (matching, contained, disjoint, empty_panel, blank,
                                         sparse_train, broad_train, normal_test)),
             "a reviewed mask or density case was not exercised")
