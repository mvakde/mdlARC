"""Exact reflection of a patterned half into its congruent blank half."""
import numpy as np

_SOURCE_INPUTS = (
    ("0000","0000","0000","0000","0000","0002","0002","0802","0382","3382"),
    ("0000","0000","0000","0000","0000","0009","0039","0039","2039","2439"),
    ("0000","0000","0000","0000","0000","0100","7100","7130","7133","7144"),
)

def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name)
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name)
    return value

def _counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        if seed % 4 == 0:
            return 2, 1
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test

def _key(grid):
    return (grid.shape, grid.tobytes())

def _source_copy(grid):
    return grid.shape == (10, 4) and tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS

def _candidates(grid):
    height, width = grid.shape
    found = []
    if height % 2 == 0:
        half = height // 2
        for side, blank, source in (("top", grid[:half], grid[half:]), ("bottom", grid[half:], grid[:half])):
            bg = int(blank[0, 0])
            if np.all(blank == bg) and np.any(source == bg) and np.any(source != bg):
                found.append(("horizontal", side, bg, source))
    if width % 2 == 0:
        half = width // 2
        for side, blank, source in (("left", grid[:, :half], grid[:, half:]), ("right", grid[:, half:], grid[:, :half])):
            bg = int(blank[0, 0])
            if np.all(blank == bg) and np.any(source == bg) and np.any(source != bg):
                found.append(("vertical", side, bg, source))
    return found

def _solve(grid):
    found = _candidates(grid)
    if len(found) != 1:
        raise AssertionError("blank destination is not unique")
    axis, side, _, source = found[0]
    out = grid.copy()
    if axis == "horizontal":
        half = grid.shape[0] // 2
        if side == "top":
            out[:half] = source[::-1]
        else:
            out[half:] = source[::-1]
    else:
        half = grid.shape[1] // 2
        if side == "left":
            out[:, :half] = source[:, ::-1]
        else:
            out[:, half:] = source[:, ::-1]
    return out

def _dimensions(rng, size_class, column, fixed):
    if fixed is not None:
        depth, cross = fixed
    elif size_class == "near":
        depth, cross = int(rng.integers(4, 7)), int(rng.integers(3, 7))
    elif size_class == "moderate":
        depth, cross = int(rng.integers(5, 11)), int(rng.integers(6, 14))
    else:
        depth, cross = int(rng.integers(9, 16)), int(rng.integers(10, 23))
    if column and cross % 2 == 0:
        cross = cross + 1 if cross < 23 else cross - 1
    return depth, cross

def _mask(rng, depth, cross, style, density):
    mask = np.zeros((depth, cross), dtype=bool)
    if style == "grow":
        initial = int(rng.integers(1, min(3, cross - 1) + 1))
        final = cross if rng.random() < .76 else int(rng.integers(initial + 1, cross + 1))
        increments = np.zeros(depth - 1, dtype=int)
        for _ in range(final - initial):
            increments[int(rng.integers(depth - 1))] += 1
        widths = np.r_[initial, initial + np.cumsum(increments)]
        anchor = int(rng.integers(3))
        drift = int(rng.integers(-1, 2))
        for row, count in enumerate(widths):
            left = 0 if anchor == 0 else cross - int(count) if anchor == 1 else (cross - int(count)) // 2
            left = int(np.clip(left + drift * row // max(depth - 1, 1), 0, cross - count))
            mask[row, left:left + count] = True
        return mask
    if style == "column":
        col = int(rng.integers(cross))
        first = int(rng.integers(0, max(1, depth - 2)))
        last = int(rng.integers(first + 2, depth + 1))
        mask[first:last, col] = True
        return mask
    if style == "bands":
        for row in rng.choice(depth, size=int(rng.integers(2, min(5, depth) + 1)), replace=False):
            span = int(rng.integers(max(2, cross // 3), cross + 1))
            left = int(rng.integers(cross - span + 1))
            mask[int(row), left:left + span] = True
            if rng.random() < .4 and row + 1 < depth:
                mask[int(row) + 1, left:left + span] = True
    elif style == "patch":
        for _ in range(int(rng.integers(1, 5))):
            high = int(rng.integers(1, max(2, depth // 2 + 1)))
            wide = int(rng.integers(1, max(2, cross // 2 + 1)))
            row = int(rng.integers(depth - high + 1))
            col = int(rng.integers(cross - wide + 1))
            mask[row:row + high, col:col + wide] = True
    else:
        area = depth * cross
        if style == "sparse":
            low, high = 2, min(6, area)
        elif style == "dense" or density == "dense":
            low, high = max(2, int(.73 * area)), area
        elif density == "sparse":
            low, high = 2, min(area, max(4, depth + cross))
        else:
            low, high = max(2, area // 5), max(3, 3 * area // 5)
        count = int(rng.integers(low, high))
        mask.flat[rng.choice(area, size=count, replace=False)] = True
    if density == "dense" and style in ("bands", "patch"):
        mask |= rng.random(mask.shape) < .2
    if np.all(mask):
        mask[int(rng.integers(depth)), int(rng.integers(cross))] = False
    return mask

def _palette(rng, regime, mode, background, colored, pool, seen, force_multi):
    if regime == "train":
        available = [c for c in pool if c != background]
    elif mode == "same":
        available = [c for c in seen if c != background]
    elif mode == "new":
        available = [c for c in range(10) if c not in seen and c != background]
    else:
        available = [c for c in range(10) if c != background]
    if not available:
        available = [c for c in range(10) if c != background]
    maximum = min(5, colored, len(available))
    minimum = min(maximum, 2 if force_multi or mode == "partial" else 1)
    count = int(rng.integers(minimum, maximum + 1))
    if not force_multi and mode != "partial" and rng.random() < .18:
        count = 1
    if regime == "test" and mode == "partial":
        old = [c for c in seen if c != background]
        new = [c for c in range(10) if c not in seen and c != background]
        if old and new and count >= 2:
            chosen = [int(rng.choice(old)), int(rng.choice(new))]
            other = [c for c in available if c not in chosen]
            if count > 2:
                chosen.extend(int(v) for v in rng.choice(other, size=count - 2, replace=False))
            return tuple(chosen)
    return tuple(int(v) for v in rng.choice(available, size=count, replace=False))

def _paint(rng, mask, palette, background, style):
    depth, cross = mask.shape
    pattern = np.full(mask.shape, background, dtype=np.int8)
    if style in ("grow", "column", "patch"):
        for col in range(cross):
            pattern[mask[:, col], col] = palette[min(len(palette) - 1, col * len(palette) // cross)]
    elif style == "bands":
        for row in range(depth):
            pattern[row, mask[row]] = palette[min(len(palette) - 1, row * len(palette) // depth)]
    else:
        positions = np.argwhere(mask)
        rng.shuffle(positions)
        for index, (row, col) in enumerate(positions):
            pattern[row, col] = palette[index % len(palette)]
    missing = [color for color in palette if color not in set(int(v) for v in pattern[mask])]
    positions = np.argwhere(mask)
    for index, color in enumerate(missing):
        row, col = positions[index * len(positions) // len(missing)]
        pattern[row, col] = color
    return pattern

def _input(pattern, axis, side, bg):
    if axis == "horizontal":
        blank = np.full_like(pattern, bg)
        return np.vstack((blank, pattern)) if side == "top" else np.vstack((pattern[::-1], blank))
    source = pattern.T
    blank = np.full_like(source, bg)
    return np.hstack((blank, source)) if side == "left" else np.hstack((source[:, ::-1], blank))

def _make_pair(rng, axis, side, size_class, style, density, bg, regime, mode, pool, seen, fixed):
    depth, cross = _dimensions(rng, size_class, style == "column", fixed)
    for _ in range(120):
        mask = _mask(rng, depth, cross, style, density)
        if np.count_nonzero(mask) < 2 or np.all(mask):
            continue
        palette = _palette(rng, regime, mode, bg, int(mask.sum()), pool, seen, regime == "train" and style == "grow")
        pattern = _paint(rng, mask, palette, bg, style)
        grid = _input(pattern, axis, side, bg)
        if len(_candidates(grid)) == 1 and not _source_copy(grid):
            return {"input": grid, "output": _solve(grid)}
    raise RuntimeError("could not construct an unambiguous pair")

def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    total = train_count + test_count
    simple = seed % 8 == 0
    mixed_axis = not simple and rng.random() < .43
    mixed_size = not simple and rng.random() < .72
    mixed_side = not simple and rng.random() < .58
    base_axis = "horizontal" if simple or rng.random() < .72 else "vertical"
    base_side = ("top" if base_axis == "horizontal" else "left") if simple else (
        ("top", "bottom")[int(rng.integers(2))] if base_axis == "horizontal" else ("left", "right")[int(rng.integers(2))])
    base_size = (5, 4) if simple else _dimensions(rng, ("near", "moderate", "large")[int(rng.integers(3))], False, None)
    pool = tuple(int(v) for v in rng.choice(10, size=int(rng.integers(3, 6)), replace=False))
    pair_seeds = rng.integers(0, 2**32, size=total, dtype=np.uint64)
    children = [np.random.default_rng(np.random.SeedSequence([int(pair_seeds[i]), seed, i])) for i in range(total)]
    puzzle = {"train": [], "test": []}
    used = set()
    seen = set()
    for index in range(total):
        regime = "train" if index < train_count else "test"
        local = index if regime == "train" else index - train_count
        prng = children[index]
        if mixed_axis and index < 2:
            axis = ("horizontal", "vertical")[index]
        else:
            axis = ("horizontal", "vertical")[int(prng.integers(2))] if mixed_axis else base_axis
        if mixed_side:
            side = ("top", "bottom")[int(prng.integers(2))] if axis == "horizontal" else ("left", "right")[int(prng.integers(2))]
        elif mixed_axis:
            side = "top" if axis == "horizontal" else "left"
        else:
            side = base_side
        size_class = ("near", "moderate", "large")[int(prng.integers(3))] if mixed_size else "near"
        fixed = None if mixed_size else base_size
        style = "grow" if regime == "train" and local == 0 else (
            "grow", "grow", "bands", "patch", "scatter", "sparse", "column", "dense")[int(prng.integers(8))]
        density = ("sparse", "middle", "middle", "middle", "dense")[int(prng.integers(5))]
        bg = int(prng.integers(10))
        mode = ("same", "partial", "new")[(seed + local) % 3] if regime == "test" else "train"
        for _ in range(80):
            pair = _make_pair(prng, axis, side, size_class, style, density, bg, regime, mode, pool, seen, fixed)
            key = _key(pair["input"])
            if key not in used:
                break
        else:
            raise RuntimeError("could not construct distinct pairs")
        used.add(key)
        puzzle[regime].append(pair)
        if regime == "train":
            _, _, background, source = _candidates(pair["input"])[0]
            seen.update(int(v) for v in np.unique(source) if int(v) != background)
    validate_generated_puzzle(puzzle)
    return puzzle

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    train_axes, test_axes = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_copy(inp)
            key = _key(inp)
            assert key not in keys
            keys.add(key)
            found = _candidates(inp)
            assert len(found) == 1
            axis, _, bg, source = found[0]
            assert np.count_nonzero(source != bg) >= 2 and np.any(source == bg)
            assert np.array_equal(_solve(inp), out)
            assert set(int(v) for v in np.unique(inp)) == set(int(v) for v in np.unique(out))
            assert np.array_equal(out, out[::-1] if axis == "horizontal" else out[:, ::-1])
            (train_axes if split == "train" else test_axes).add(axis)
    assert test_axes <= train_axes

def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])) for split in ("train", "test"))

def self_test() -> None:
    samples = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in samples:
        validate_generated_puzzle(puzzle)
    assert _signature(generate_puzzle(17)) == _signature(generate_puzzle(17))
    assert len({_signature(p) for p in samples}) == 42
    assert len({len(p["train"]) for p in samples}) >= 4
    assert len({len(p["test"]) for p in samples}) >= 3
    pairs = [p for puzzle in samples for split in ("train", "test") for p in puzzle[split]]
    infos = [_candidates(p["input"])[0] for p in pairs]
    assert {x[0] for x in infos} == {"horizontal", "vertical"}
    assert {x[1] for x in infos} == {"top", "bottom", "left", "right"}
    assert {x[2] for x in infos} == set(range(10))
    assert len({p["input"].shape for p in pairs}) >= 25
    assert sum(len({p["input"].shape for split in ("train", "test") for p in puzzle[split]}) > 1 for puzzle in samples) >= 15
    assert sum(len({_candidates(p["input"])[0][1] for split in ("train", "test") for p in puzzle[split]}) > 1 for puzzle in samples) >= 12
    assert sum(len(set(int(v) for v in np.unique(source))) == 2 for _, _, _, source in infos) >= 8
    measurements = {name: 0 for name in (
        "original_scale", "growing", "color_run", "sparse", "dense", "column",
        "no_seam", "no_outer", "mixed_axis", "mixed_background", "new_test_shape",
        "same_palette", "partial_palette", "new_palette")}
    for puzzle in samples:
        all_pairs = puzzle["train"] + puzzle["test"]
        all_info = [_candidates(pair["input"])[0] for pair in all_pairs]
        if len({info[0] for info in all_info}) > 1:
            measurements["mixed_axis"] += 1
        if len({info[2] for info in all_info}) > 1:
            measurements["mixed_background"] += 1
        train_colors = {int(v) for axis, side, bg, source in all_info[:len(puzzle["train"])]
                        for v in np.unique(source) if int(v) != bg}
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        for index, (pair, info) in enumerate(zip(all_pairs, all_info)):
            axis, side, bg, source = info
            if pair["input"].shape == (10, 4):
                measurements["original_scale"] += 1
            logical = source if side in ("top", "left") else (
                source[::-1] if axis == "horizontal" else source[:, ::-1])
            if axis == "vertical":
                logical = logical.T
            mask = logical != bg
            colored = int(mask.sum())
            occupancy = [int(v) for v in mask.sum(axis=1)]
            if occupancy[0] and occupancy[-1] == mask.shape[1] and all(
                a <= b for a, b in zip(occupancy, occupancy[1:])):
                measurements["growing"] += 1
            if any(np.count_nonzero((logical[:-1, col] == logical[1:, col])
                                    & (logical[:-1, col] != bg)) >= 2
                   for col in range(logical.shape[1])):
                measurements["color_run"] += 1
            if colored <= 5:
                measurements["sparse"] += 1
            if colored / mask.size >= .7:
                measurements["dense"] += 1
            if np.count_nonzero(mask.any(axis=0)) == 1:
                measurements["column"] += 1
            if not mask[0].any():
                measurements["no_seam"] += 1
            if not mask[-1].any():
                measurements["no_outer"] += 1
            if index >= len(puzzle["train"]):
                if pair["input"].shape not in train_shapes:
                    measurements["new_test_shape"] += 1
                colors = {int(v) for v in np.unique(source) if int(v) != bg}
                old, new = colors & train_colors, colors - train_colors
                category = "partial_palette" if old and new else "same_palette" if old else "new_palette"
                measurements[category] += 1
    assert measurements["original_scale"] >= 5
    assert measurements["growing"] >= 25 and measurements["color_run"] >= 30
    assert all(measurements[name] >= 5 for name in (
        "sparse", "dense", "column", "no_seam", "no_outer", "mixed_axis",
        "new_test_shape", "same_palette", "partial_palette", "new_palette"))
    assert measurements["mixed_background"] >= 30
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(501 + train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(601, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 2.5},
                      {"num_test": True}):
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
