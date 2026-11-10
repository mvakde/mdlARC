"""Synthetic 3-by-3 ARC grids enlarged cellwise to 9-by-9 grids."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ((3, 0, 2), (0, 2, 2), (0, 0, 3)),
    ((3, 3, 0), (7, 4, 0), (0, 0, 4)),
    ((0, 1, 0), (0, 0, 6), (6, 1, 0)),
)
_LAYOUTS = ("random", "corner", "center", "row", "column", "diagonal",
            "checkerboard", "patch", "disconnected", "many")


def _normalise_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _normalise_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _normalise_count(num_train, "num_train", 2, 10)
    num_test = _normalise_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    else:
        num_test = 1 + seed % min(4, num_train - 1)
    return num_train, num_test


def _signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _source_copy(grid):
    return any(np.array_equal(grid, np.asarray(rows, dtype=np.int8))
               for rows in _SOURCE_INPUT_ROWS)


def _scale(grid):
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)


def _mask(rng, layout, count=None):
    """Return the positions for one color; the rest can hold several colors."""
    if layout == "random":
        count = int(rng.integers(2, 8)) if count is None else count
        positions = rng.choice(9, count, replace=False)
    elif layout in ("corner", "center"):
        singleton = int(rng.choice((0, 2, 6, 8))) if layout == "corner" else 4
        positions = [i for i in range(9) if i != singleton]
    elif layout == "row":
        row = int(rng.integers(0, 3))
        positions = [3 * row + i for i in range(3)]
    elif layout == "column":
        column = int(rng.integers(0, 3))
        positions = [3 * i + column for i in range(3)]
    elif layout == "diagonal":
        positions = (0, 4, 8) if int(rng.integers(0, 2)) else (2, 4, 6)
    elif layout == "checkerboard":
        positions = (0, 2, 4, 6, 8) if int(rng.integers(0, 2)) else (1, 3, 5, 7)
    elif layout == "disconnected":
        positions = (0, 2, 6, 8)
    elif layout == "patch":
        target = int(rng.integers(3, 7))
        chosen = {int(rng.integers(0, 9))}
        while len(chosen) < target:
            frontier = set()
            for pos in chosen:
                row, col = divmod(pos, 3)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < 3 and 0 <= cc < 3 and 3 * rr + cc not in chosen:
                        frontier.add(3 * rr + cc)
            chosen.add(int(rng.choice(sorted(frontier))))
        positions = sorted(chosen)
    elif layout == "many":
        positions = [int(rng.integers(0, 9))]
    else:
        raise ValueError("unknown layout")
    mask = np.zeros(9, dtype=bool)
    mask[list(positions)] = True
    return mask


def _colored_grid(rng, mask, primary, secondary, desired_colors, allowed):
    """Color every position; all chosen colors are visible in the input."""
    primary = int(primary)
    count = int(np.count_nonzero(mask))
    if count == 9:
        return np.full((3, 3), primary, dtype=np.int8)
    allowed = [int(x) for x in allowed if int(x) != primary]
    if secondary is not None:
        secondary = int(secondary)
        allowed = [secondary] + [x for x in allowed if x != secondary]
    desired_colors = max(2, min(int(desired_colors), 10 - count, len(allowed) + 1))
    other_colors = allowed[:desired_colors - 1]
    remaining = 9 - count
    values = other_colors + [other_colors[i % len(other_colors)]
                             for i in range(remaining - len(other_colors))]
    rng.shuffle(values)
    flat = np.empty(9, dtype=np.int8)
    flat[mask] = primary
    flat[~mask] = values
    return flat.reshape(3, 3)


def _make_pair(rng, layout, primary, secondary, desired_colors, allowed,
               seen, mask=None, count=None):
    for attempt in range(80):
        if mask is not None and attempt < 20:
            actual_mask = mask
        elif attempt < 40:
            actual_mask = _mask(rng, layout, count)
        else:
            actual_mask = _mask(rng, "random")
        grid = _colored_grid(rng, actual_mask, primary, secondary, desired_colors,
                             rng.permutation(allowed))
        key = _signature(grid)
        if key in seen or _source_copy(grid):
            continue
        seen.add(key)
        return {"input": grid, "output": _scale(grid)}, actual_mask.copy()
    # Every two-color palette has far more possible inputs than the maximum
    # 14 pairs in one puzzle. This deterministic last resort cannot deadlock.
    for bits in range(1, 511):
        actual_mask = np.asarray([(bits >> i) & 1 for i in range(9)], dtype=bool)
        flat = np.full(9, int(secondary), dtype=np.int8)
        flat[actual_mask] = int(primary)
        grid = flat.reshape(3, 3)
        key = _signature(grid)
        if key in seen or _source_copy(grid):
            continue
        seen.add(key)
        return {"input": grid, "output": _scale(grid)}, actual_mask
    raise RuntimeError("no unique input remains")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalise_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    # The only learned convention is the fixed 3x scale. These two colors are
    # design anchors for visible role changes, not hidden semantic roles.
    anchors = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    # Some puzzles leave one color unseen in training for a novel-color test.
    # The rest permit all ten colors, so this is not a fixed train/test split.
    reserve = int(rng.choice([x for x in range(10) if x not in anchors])) if rng.random() < 0.6 else None
    train_allowed = [x for x in range(10) if x != reserve]
    child_rngs = [np.random.default_rng(int(x)) for x in
                  rng.integers(0, np.iinfo(np.int64).max,
                               size=num_train + num_test, dtype=np.int64)]
    result = {"train": [], "test": []}
    seen = set()
    masks = []
    first_count = int(rng.integers(5, 9))
    second_count = int(rng.choice([x for x in range(5, 9) if x != first_count]))
    counts = (first_count, second_count)
    for i in range(2):
        primary, secondary = (anchors[0], anchors[1]) if i == 0 else (anchors[1], anchors[0])
        pair, mask = _make_pair(child_rngs[i], "random", primary, secondary,
                                int(rng.integers(2, min(6, 11 - counts[i]))),
                                train_allowed, seen, count=counts[i])
        result["train"].append(pair)
        masks.append(mask)

    # Other training pairs independently explore motif, frequency, and palette.
    for i in range(2, num_train):
        layout = str(rng.choice(_LAYOUTS))
        primary = int(rng.choice(train_allowed))
        secondary = int(rng.choice([x for x in train_allowed if x != primary]))
        desired = int(rng.integers(6, 10)) if layout == "many" else int(rng.integers(2, 10))
        preset = masks[0] if i == 2 and rng.random() < 0.35 else None
        pair, mask = _make_pair(child_rngs[i], layout, primary, secondary,
                                desired, train_allowed, seen, mask=preset)
        result["train"].append(pair)
        masks.append(mask)

    train_colors = set(int(x) for pair in result["train"] for x in np.unique(pair["input"]))
    unseen = set(range(10)) - train_colors
    rare_train = sorted(train_colors, key=lambda c: sum(int(np.count_nonzero(
        pair["input"] == c)) for pair in result["train"]))
    for j in range(num_test):
        pair_rng = child_rngs[num_train + j]
        layout = str(rng.choice(_LAYOUTS + ("uniform",)))
        mode = str(rng.choice(("familiar", "partial", "novel", "role_change",
                               "palette_reuse")))
        if layout == "uniform":
            color = int(rng.choice(tuple(sorted(train_colors | unseen))))
            grid = np.full((3, 3), color, dtype=np.int8)
            if _signature(grid) in seen:
                color = next(x for x in range(10) if _signature(np.full((3, 3), x,
                              dtype=np.int8)) not in seen)
                grid.fill(color)
            seen.add(_signature(grid))
            pair = {"input": grid, "output": _scale(grid)}
        else:
            if mode == "palette_reuse":
                reference = result["train"][int(rng.integers(0, 2))]["input"]
                allowed = sorted(int(x) for x in np.unique(reference))
                primary = int(rng.choice(allowed))
                secondary = int(rng.choice([x for x in allowed if x != primary]))
                desired = len(allowed)
                layout = "random"
            elif mode == "novel" and unseen:
                primary = int(rng.choice(tuple(sorted(unseen))))
                allowed = sorted(unseen | train_colors)
                secondary = int(rng.choice(tuple(sorted(train_colors))))
            elif mode == "partial" and unseen:
                primary = int(rng.choice(tuple(sorted(train_colors))))
                secondary = int(rng.choice(tuple(sorted(unseen))))
                allowed = sorted(train_colors | unseen)
            else:
                primary = int(rng.choice(rare_train[:max(1, len(rare_train) // 2)])) if mode == "role_change" else int(rng.choice(tuple(sorted(train_colors))))
                allowed = sorted(train_colors)
                secondary = int(rng.choice([x for x in allowed if x != primary]))
            if mode != "palette_reuse":
                desired = int(rng.integers(6, 10)) if layout == "many" else int(rng.integers(2, 10))
            preset = None if mode == "palette_reuse" else (
                masks[int(rng.integers(0, len(masks)))] if rng.random() < 0.32 else None)
            count = int(rng.integers(2, 11 - desired)) if mode == "palette_reuse" else None
            pair, _ = _make_pair(pair_rng, layout, primary, secondary,
                                 desired, allowed, seen, mask=preset, count=count)
        result["test"].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert bool(np.all((0 <= grid) & (grid <= 9)))
        inp, out = pair["input"], pair["output"]
        assert inp.shape == (3, 3) and out.shape == (9, 9)
        assert np.array_equal(out, _scale(inp))
        assert not _source_copy(inp)
        signature = _signature(inp)
        assert signature not in seen
        seen.add(signature)
    # Nonuniform demonstrations show positional and color-preserving copying.
    for pair in puzzle["train"][:2]:
        assert len(np.unique(pair["input"])) >= 2
    first, second = (puzzle["train"][i]["input"] for i in (0, 1))
    first_color = int(np.bincount(first.flat, minlength=10).argmax())
    second_color = int(np.bincount(second.flat, minlength=10).argmax())
    assert first_color != second_color
    assert first_color in second and second_color in first
    assert int(np.bincount(first.flat, minlength=10).max()) != int(
        np.bincount(second.flat, minlength=10).max())


def _puzzles_equal(left, right):
    return all(np.array_equal(a[key], b[key]) for split in ("train", "test")
               for a, b in zip(left[split], right[split]) for key in ("input", "output")) \
        and all(len(left[split]) == len(right[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(puzzles[7], generate_puzzle(7))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(314159 + train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
        validate_generated_puzzle(p)
        assert _puzzles_equal(p, generate_puzzle(314159 + train + test, train, test))
    for kwargs in ({"num_test": 1}, {"num_test": 4},
                   {"num_train": 2}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(9001, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    inputs = [pair["input"] for p in puzzles for split in ("train", "test")
              for pair in p[split]]
    signatures = {_signature(grid) for grid in inputs}
    color_counts = {len(np.unique(grid)) for grid in inputs}
    max_frequencies = {int(np.bincount(grid.flat, minlength=10).max()) for grid in inputs}
    normalized = {tuple(int(x) for x in np.unique(grid, return_inverse=True)[1].flat)
                  for grid in inputs}
    assert len(signatures) >= 200 and len(normalized) >= 100
    assert color_counts == set(range(1, 10))
    assert max_frequencies == set(range(1, 10))
    assert any(len(set(int(x) for x in np.unique(p["train"][0]["input"])) &
                   set(int(x) for x in np.unique(p["test"][0]["input"]))) >= 2
               for p in puzzles)
    assert any(len({tuple(int(x) for x in (grid.flat == int(np.bincount(grid.flat, minlength=10).argmax())).flat)
                    for grid in (p["train"][0]["input"], p["test"][0]["input"])}) == 1
               for p in puzzles)

    patterns = set()
    partial_tests = all_novel_tests = uniform_tests = exact_palette_reuses = 0
    recolored_layout_puzzles = varied_train_palettes = varied_train_frequencies = 0
    for puzzle in puzzles:
        train, test = puzzle["train"], puzzle["test"]
        grids = [pair["input"] for pair in train + test]
        train_colors = set(int(x) for pair in train for x in np.unique(pair["input"]))
        if len({len(np.unique(pair["input"])) for pair in train}) > 1:
            varied_train_palettes += 1
        if len({int(np.bincount(pair["input"].flat, minlength=10).max())
                for pair in train}) > 1:
            varied_train_frequencies += 1
        masks = [{int(c): frozenset(int(i) for i in np.flatnonzero(grid == c))
                  for c in np.unique(grid)} for grid in grids]
        if any(3 <= len(a) <= 8 and a == b and ca != cb
               for i in range(len(train)) for j in range(i + 1, len(grids))
               for ca, a in masks[i].items() for cb, b in masks[j].items()):
            recolored_layout_puzzles += 1
        for pair in test:
            grid = pair["input"]
            colors = set(int(x) for x in np.unique(grid))
            if colors - train_colors and colors & train_colors:
                partial_tests += 1
            elif colors - train_colors:
                all_novel_tests += 1
            if len(colors) == 1:
                uniform_tests += 1
            if any(colors == set(int(x) for x in np.unique(t["input"]))
                   and not np.array_equal(grid, t["input"]) for t in train):
                exact_palette_reuses += 1
        for grid_masks in masks:
            for color, cells in grid_masks.items():
                if len(cells) == 1 and next(iter(cells)) in (0, 2, 6, 8):
                    patterns.add("corner")
                if cells == {4}:
                    patterns.add("center")
                if cells in ({0, 1, 2}, {3, 4, 5}, {6, 7, 8},
                             {0, 3, 6}, {1, 4, 7}, {2, 5, 8}):
                    patterns.add("line")
                    if len(grid_masks) >= 6:
                        patterns.add("many_colors_and_line")
                if cells in ({0, 4, 8}, {2, 4, 6}):
                    patterns.add("diagonal")
                if cells in ({0, 2, 4, 6, 8}, {1, 3, 5, 7}):
                    patterns.add("checkerboard")
                if cells == {0, 2, 6, 8}:
                    patterns.add("disconnected")
                if color == 0 and len(cells) == 1:
                    patterns.add("rare_zero")
    assert varied_train_palettes >= 30 and varied_train_frequencies == 42
    assert recolored_layout_puzzles >= 15
    assert partial_tests >= 10 and all_novel_tests >= 1
    assert uniform_tests >= 3 and exact_palette_reuses >= 5
    assert patterns == {"corner", "center", "line", "diagonal",
                        "checkerboard", "disconnected", "rare_zero",
                        "many_colors_and_line"}
