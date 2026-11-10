"""Paired strokes project as outward staircases; exact collisions recolor."""

import numpy as np


def _key(grid):
    arr = np.asarray(grid, dtype=np.int8)
    return arr.shape, arr.tobytes()


def _source_inputs():
    definitions = (
        (22, 18, ((3, 1, 3, 12, 14), (8, 7, 2, 3, 6))),
        (23, 17, ((8, 6, 4, 3, 5), (3, 17, 3, 10, 13))),
        (24, 19, ((3, 6, 4, 4, 8), (8, 19, 1, 11, 13))),
        (24, 20, ((3, 3, 3, 11, 16), (8, 10, 5, 5, 8))),
    )
    keys = set()
    for h, w, objects in definitions:
        grid = np.zeros((h, w), dtype=np.int8)
        for color, start, length, low, high in objects:
            grid[start:start + length, low] = color
            grid[start:start + length, high] = color
        keys.add(_key(grid))
    return keys


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside legal range")
    return value


def _counts(seed, train, test):
    train = _count(train, "num_train", 2, 10)
    test = _count(test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _mask(shape, spec):
    """spec: orientation, segment length, axis start, low track, separation."""
    orientation, length, start, low, separation = spec
    axis_size = shape[0] if orientation == "v" else shape[1]
    normal_size = shape[1] if orientation == "v" else shape[0]
    axis = np.arange(axis_size)
    steps = np.where(
        axis < start, (start - axis - 1) // length + 1,
        np.where(axis >= start + length,
                 (axis - start - length) // length + 1, 0))
    mask = np.zeros(shape, dtype=bool)
    for normal in (low - steps, low + separation + steps):
        good = (normal >= 0) & (normal < normal_size)
        if orientation == "v":
            mask[axis[good], normal[good]] = True
        else:
            mask[normal[good], axis[good]] = True
    return mask


def _seed_mask(shape, spec):
    orientation, length, start, low, separation = spec
    mask = np.zeros(shape, dtype=bool)
    if orientation == "v":
        mask[start:start + length, low] = True
        mask[start:start + length, low + separation] = True
    else:
        mask[low, start:start + length] = True
        mask[low + separation, start:start + length] = True
    return mask


def _visible_cadence(shape, spec):
    orientation, length, start, _, _ = spec
    axis_size = shape[0] if orientation == "v" else shape[1]
    return max(start, axis_size - start - length) >= length + 1


def _parse_seed(grid, color):
    rows, cols = np.where(grid == color)
    assert len(rows), "missing seed"
    found = []
    for orientation in ("v", "h"):
        axis = rows if orientation == "v" else cols
        normal = cols if orientation == "v" else rows
        tracks = np.unique(normal)
        if len(tracks) != 2 or tracks[1] - tracks[0] < 2:
            continue
        a, b = (np.sort(axis[normal == track]) for track in tracks)
        if not len(a) or not np.array_equal(a, b):
            continue
        if not np.array_equal(a, np.arange(a[0], a[0] + len(a))):
            continue
        spec = (orientation, len(a), int(a[0]), int(tracks[0]),
                int(tracks[1] - tracks[0]))
        if np.array_equal(_seed_mask(grid.shape, spec), grid == color):
            found.append(spec)
    assert len(found) == 1, "malformed or ambiguous seed"
    return found[0]


def _max_run(mask):
    best = 0
    for grid in (mask, mask.T):
        for line in grid:
            current = 0
            for cell in line:
                current = current + 1 if cell else 0
                best = max(best, current)
    return best


def _kind(overlap, seeds):
    if not np.any(overlap):
        return "none"
    if np.any(overlap & seeds):
        return "seed_hit"
    if _max_run(overlap) >= 4:
        return "long"
    if _max_run(overlap) >= 2:
        return "short"
    return "isolated"


def _colors(rng, intersection, fixed):
    if fixed is not None:
        return fixed
    available = [color for color in range(10) if color != intersection]
    return tuple(int(x) for x in rng.choice(available, size=3, replace=False))


def _spec(rng, shape, orientation, length=None, center=None):
    h, w = shape
    axis_size = h if orientation == "v" else w
    normal_size = w if orientation == "v" else h
    if length is None:
        length = int(rng.integers(1, min(6, axis_size // 3) + 1))
    separation = int(rng.integers(2, min(8, normal_size - 2)))
    if center is None or rng.random() < 0.35:
        start = int(rng.integers(0, axis_size - length + 1))
        low = int(rng.integers(0, normal_size - separation))
    else:
        axis_center = center[0] if orientation == "v" else center[1]
        normal_center = center[1] if orientation == "v" else center[0]
        start = int(np.clip(round(axis_center + rng.integers(-7, 8) - length / 2),
                            0, axis_size - length))
        low = int(np.clip(round(normal_center + rng.integers(-7, 8) - separation / 2),
                          0, normal_size - separation - 1))
    return orientation, length, start, low, separation


def _orientations(rng, allowed, forced):
    if forced is not None:
        return forced
    if allowed in ("v", "h"):
        return allowed, allowed
    choices = (("v", "v"), ("h", "h"), ("v", "h"), ("h", "v"))
    return choices[int(rng.integers(0, 4))]


def _sample_pair(rng, intersection, fixed, allowed, target, forced, test):
    for _ in range(1800):
        shape = (int(rng.integers(13, 31)), int(rng.integers(13, 31)))
        orientation = _orientations(rng, allowed, forced)
        lengths = [None, None]
        if test:
            lengths = [int(rng.integers(4, 7)) if rng.random() < 0.48 else None
                       for _ in range(2)]
        if target == "long" and rng.random() < 0.45:
            lengths = [int(rng.integers(3, 7))] * 2
        first = _spec(rng, shape, orientation[0], lengths[0])
        seed1 = _seed_mask(shape, first)
        center = tuple(np.mean(np.argwhere(seed1), axis=0))
        second = _spec(rng, shape, orientation[1], lengths[1], center)
        seed2 = _seed_mask(shape, second)
        if not _visible_cadence(shape, first) or not _visible_cadence(shape, second):
            continue
        if np.any(seed1 & seed2):
            continue
        mask1, mask2 = _mask(shape, first), _mask(shape, second)
        if int(mask1.sum()) <= 2 * first[1] + 1 or int(mask2.sum()) <= 2 * second[1] + 1:
            continue
        overlap = mask1 & mask2
        kind = _kind(overlap, seed1 | seed2)
        if target == "crossing" and kind == "none":
            continue
        if target not in ("any", "crossing") and kind != target:
            continue
        background, color1, color2 = _colors(rng, intersection, fixed)
        inp = np.full(shape, background, dtype=np.int8)
        inp[seed1], inp[seed2] = color1, color2
        try:
            if _parse_seed(inp, color1) != first or _parse_seed(inp, color2) != second:
                continue
        except AssertionError:
            continue
        out = np.full(shape, background, dtype=np.int8)
        out[mask1 & ~mask2] = color1
        out[mask2 & ~mask1] = color2
        out[overlap] = intersection
        if np.array_equal(inp, out):
            continue
        return {"input": inp, "output": out}
    if target not in ("any", "none", "crossing"):
        return _sample_pair(rng, intersection, fixed, allowed, "crossing", forced, test)
    raise RuntimeError(f"could not construct {target} pair")


def _facts(pair, intersection):
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    background = int(values[int(np.argmax(counts))])
    assert int(max(counts)) > int(sorted(counts)[-2])
    colors = [int(value) for value in values if value != background]
    assert intersection not in values
    specs = [_parse_seed(inp, color) for color in colors]
    assert all(_visible_cadence(inp.shape, spec) for spec in specs)
    masks = [_mask(inp.shape, spec) for spec in specs]
    seeds = [_seed_mask(inp.shape, spec) for spec in specs]
    assert not np.any(seeds[0] & seeds[1])
    assert all(int(mask.sum()) > int(seed.sum()) + 1 for mask, seed in zip(masks, seeds))
    expected = np.full(inp.shape, background, dtype=np.int8)
    expected[masks[0] & ~masks[1]] = colors[0]
    expected[masks[1] & ~masks[0]] = colors[1]
    overlap = masks[0] & masks[1]
    expected[overlap] = intersection
    assert np.array_equal(out, expected)
    return {"orientations": {spec[0] for spec in specs},
            "lengths": tuple(spec[1] for spec in specs),
            "kind": _kind(overlap, seeds[0] | seeds[1]),
            "overlap": int(overlap.sum()),
            "colors": (background, *colors),
            "specs": specs}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, input_colors = set(), set(), set()
    source = _source_inputs()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= d <= 30 for d in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        assert not np.array_equal(pair["input"], pair["output"])
        key = _key(pair["input"])
        pair_key = (key, _key(pair["output"]))
        assert key not in seen_inputs and pair_key not in seen_pairs and key not in source
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        input_colors.update(int(x) for x in np.unique(pair["input"]))
    extra = {int(x) for pair in train for x in np.unique(pair["output"])
             if int(x) not in input_colors}
    assert len(extra) == 1
    intersection = next(iter(extra))
    train_facts = [_facts(pair, intersection) for pair in train]
    test_facts = [_facts(pair, intersection) for pair in test]
    assert any(f["overlap"] for f in train_facts)
    assert any(f["kind"] == "none" for f in train_facts)
    train_orientations = set().union(*(f["orientations"] for f in train_facts))
    test_orientations = set().union(*(f["orientations"] for f in test_facts))
    assert test_orientations <= train_orientations
    for pair, facts in zip(train + test, train_facts + test_facts):
        allowed = set(int(x) for x in np.unique(pair["input"])) | {intersection}
        assert set(int(x) for x in np.unique(pair["output"])) <= allowed
        assert all(1 <= spec[1] <= 6 and 2 <= spec[4] <= 7 for spec in facts["specs"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    intersection = int(rng.integers(0, 10))
    fixed = _colors(rng, intersection, None) if rng.random() < 0.32 else None
    allowed = ("v", "h", "both")[int(rng.choice(3, p=[0.24, 0.22, 0.54]))]
    child_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.int64)
    train_targets = ["none", str(rng.choice(("isolated", "short", "long", "seed_hit")))]
    train_targets += [str(rng.choice(("any", "none", "isolated", "short", "long", "seed_hit")))
                      for _ in range(num_train - 2)]
    rng.shuffle(train_targets)
    train_forced = [None] * num_train
    if allowed == "both":
        places = rng.choice(num_train, size=2, replace=False)
        train_forced[int(places[0])] = ("v", "v")
        train_forced[int(places[1])] = ("h", "h")
    test_targets = [str(rng.choice(("any", "none", "isolated", "short", "long", "seed_hit"),
                                   p=[0.19, 0.15, 0.15, 0.16, 0.19, 0.16]))
                    for _ in range(num_test)]
    train, test, used = [], [], set()
    source = _source_inputs()
    plans = [("train", t, train_forced[i]) for i, t in enumerate(train_targets)]
    plans += [("test", t, None) for t in test_targets]
    for index, (split, target, forced) in enumerate(plans):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        for _ in range(12):
            pair = _sample_pair(pair_rng, intersection, fixed, allowed,
                                target, forced, split == "test")
            key = _key(pair["input"])
            if key not in used and key not in source:
                used.add(key)
                (train if split == "train" else test).append(pair)
                break
        else:
            raise RuntimeError("could not construct unique pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[s], second[s]) for role in ("input", "output"))
               for s in ("train", "test"))


def _check_by_bands(pair, intersection):
    """A second, cellwise projection used only as a self-test oracle."""
    inp = pair["input"]
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    colors = [int(value) for value in values if value != background]
    masks = []
    for color in colors:
        orientation, length, start, low, separation = _parse_seed(inp, color)
        mask = np.zeros(inp.shape, dtype=bool)
        axis_size = inp.shape[0] if orientation == "v" else inp.shape[1]
        normal_size = inp.shape[1] if orientation == "v" else inp.shape[0]
        for axis in range(axis_size):
            if axis < start:
                distance = start - axis
            elif axis >= start + length:
                distance = axis - (start + length) + 1
            else:
                distance = 0
            displacement = (distance + length - 1) // length
            for normal in (low - displacement, low + separation + displacement):
                if 0 <= normal < normal_size:
                    row, col = (axis, normal) if orientation == "v" else (normal, axis)
                    mask[row, col] = True
        masks.append(mask)
    predicted = np.full(inp.shape, background, dtype=np.int8)
    predicted[masks[0] & ~masks[1]] = colors[0]
    predicted[masks[1] & ~masks[0]] = colors[1]
    predicted[masks[0] & masks[1]] = intersection
    assert np.array_equal(predicted, pair["output"])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        learned = {int(v) for pair in puzzle["train"]
                   for v in set(np.unique(pair["output"])) - set(np.unique(pair["input"]))}
        assert len(learned) == 1
        intersection = next(iter(learned))
        for pair in puzzle["train"] + puzzle["test"]:
            _check_by_bands(pair, intersection)
    assert _same(puzzles[17], generate_puzzle(17))
    assert not _same(puzzles[0], puzzles[1])
    cases = ((2, 1), (10, 1), (10, 4), (None, 1),
             (None, 4), (2, None), (10, None))
    for index, (a, b) in enumerate(cases):
        puzzle = generate_puzzle(1000 + index, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            assert len(puzzle["train"]) == a
        if b is not None:
            assert len(puzzle["test"]) == b
    illegal = ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
               {"num_test": 5}, {"num_train": 2, "num_test": 2},
               {"num_train": 3, "num_test": 4})
    for kwargs in illegal:
        try:
            generate_puzzle(3, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_key(pair["input"]) for s in ("train", "test")
                      for pair in p[s]) for p in puzzles}) >= 40
    kinds, orientations, lengths, shapes, layouts = set(), set(), set(), set(), set()
    train_long = test_none = seed_hit = varied_palettes = 0
    for puzzle in puzzles:
        input_colors = {int(v) for pair in puzzle["train"] + puzzle["test"]
                        for v in np.unique(pair["input"])}
        extra = {int(v) for pair in puzzle["train"] for v in np.unique(pair["output"])} - input_colors
        assert len(extra) == 1
        intersection = next(iter(extra))
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                facts = _facts(pair, intersection)
                kinds.add(facts["kind"])
                orientations.add(tuple(sorted(facts["orientations"])))
                lengths.update(facts["lengths"])
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, tuple(facts["specs"])))
                palettes.add(facts["colors"])
                train_long += split == "train" and facts["kind"] == "long"
                test_none += split == "test" and facts["kind"] == "none"
                seed_hit += facts["kind"] == "seed_hit"
        varied_palettes += len(palettes) > 1
    assert kinds == {"none", "isolated", "short", "long", "seed_hit"}
    assert {("v",), ("h",), ("h", "v")} <= orientations
    assert lengths == set(range(1, 7))
    assert len(shapes) >= 80 and len(layouts) >= 250
    assert train_long >= 2 and test_none >= 2 and seed_hit >= 5
    assert varied_palettes >= 15
