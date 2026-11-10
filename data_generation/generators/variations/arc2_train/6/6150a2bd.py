"""Solved ARC puzzles whose entire canvas is turned through 180 degrees."""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[5, 5, 2], [1, 0, 0], [0, 0, 0]], dtype=np.int8),
    np.array([[3, 3, 8], [3, 7, 0], [5, 0, 0]], dtype=np.int8),
    np.array([[6, 3, 5], [6, 8, 0], [4, 0, 0]], dtype=np.int8),
)
_MOTIFS = ("stair", "scatter", "cluster", "lines", "hollow", "irregular", "near_symmetric")


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _dimension(rng: np.random.Generator, parity: int | None = None) -> int:
    band = int(rng.choice(3, p=[0.36, 0.39, 0.25]))
    low, high = ((3, 7), (7, 15), (15, 31))[band]
    result = int(rng.integers(low, high))
    if parity is not None and result % 2 != parity:
        result = result + 1 if result < 30 else result - 1
    return result


def _shape(rng: np.random.Generator, ordinal: int, first_shape: tuple[int, int] | None) -> tuple[int, int]:
    if ordinal == 0:
        return _dimension(rng, 1), _dimension(rng, 1)
    if ordinal == 1:
        result = (_dimension(rng, 0), _dimension(rng))
        if result == first_shape:
            result = (result[0], 3 if result[1] != 3 else 5)
        return result
    return _dimension(rng), _dimension(rng)


def _ferrers(mask: np.ndarray) -> bool:
    lengths = []
    for row in mask:
        length = int(np.count_nonzero(row))
        if not bool(np.all(row[:length])) or bool(np.any(row[length:])):
            return False
        lengths.append(length)
    return all(a >= b for a, b in zip(lengths, lengths[1:]))


def _component_count(mask: np.ndarray) -> int:
    height, width = mask.shape
    seen = set()
    count = 0
    for row, col in np.argwhere(mask):
        start = (int(row), int(col))
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nr, nc = neighbor
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return count


def _enclosed_hole(mask: np.ndarray) -> bool:
    height, width = mask.shape
    background = ~mask
    stack = [(r, c) for r in range(height) for c in (0, width - 1) if background[r, c]]
    stack.extend((r, c) for c in range(width) for r in (0, height - 1) if background[r, c])
    reached = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in reached:
            continue
        reached.add((r, c))
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nr, nc = neighbor
            if 0 <= nr < height and 0 <= nc < width and background[nr, nc] and neighbor not in reached:
                stack.append(neighbor)
    return int(background.sum()) > len(reached)


def _mask(rng: np.random.Generator, height: int, width: int, motif: str, landmark: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if motif == "stair":
        used = int(rng.integers(2, height + 1))
        slope = float(rng.uniform(0.7, 2.1))
        previous = width
        for row in range(used):
            length = width if row == 0 else max(1, min(previous, int(np.ceil(width * ((used - row) / used) ** slope))))
            mask[row, :length] = True
            previous = length
    elif motif == "scatter":
        mask = rng.random((height, width)) < rng.uniform(0.06, 0.40)
    elif motif == "cluster":
        rh = int(rng.integers(2, height + 1))
        cw = int(rng.integers(2, width + 1))
        top = int(rng.integers(0, height - rh + 1))
        left = int(rng.integers(0, width - cw + 1))
        mask[top:top + rh, left:left + cw] = rng.random((rh, cw)) < rng.uniform(0.55, 0.95)
    elif motif == "lines":
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        start = int(rng.integers(0, width - 1))
        end = int(rng.integers(start + 2, width + 1))
        mask[row, start:end] = True
        start = int(rng.integers(0, height - 1))
        end = int(rng.integers(start + 2, height + 1))
        mask[start:end, col] = True
        if rng.random() < 0.5:
            mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif motif == "hollow":
        rh = int(rng.integers(3, height + 1))
        cw = int(rng.integers(3, width + 1))
        top = int(rng.integers(0, height - rh + 1))
        left = int(rng.integers(0, width - cw + 1))
        mask[top, left:left + cw] = True
        mask[top + rh - 1, left:left + cw] = True
        mask[top:top + rh, left] = True
        mask[top:top + rh, left + cw - 1] = True
        if rng.random() < 0.35:
            mask[top, left + int(rng.integers(1, cw - 1))] = False
    elif motif == "irregular":
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        for _ in range(int(rng.integers(4, min(height * width, 30) + 1))):
            mask[row, col] = True
            direction = int(rng.integers(4))
            row = min(height - 1, max(0, row + (direction == 0) - (direction == 1)))
            col = min(width - 1, max(0, col + (direction == 2) - (direction == 3)))
        mask |= rng.random((height, width)) < 0.025
    elif motif == "near_symmetric":
        for _ in range(int(rng.integers(2, max(3, height * width // 8 + 1)))):
            row, col = int(rng.integers(height)), int(rng.integers(width))
            mask[row, col] = True
            mask[height - 1 - row, width - 1 - col] = True
        row, col = int(rng.integers(height)), int(rng.integers(width))
        mask[row, col] = not mask[row, col]
    else:
        raise ValueError("unknown motif")

    if motif != "stair" and rng.random() < 0.35:
        mask[int(rng.integers(height)), width - 1] = True
    if motif != "stair" and rng.random() < 0.35:
        mask[height - 1, int(rng.integers(width))] = True
    if motif not in ("stair", "hollow") and height >= 4 and width >= 4 and rng.random() < 0.35:
        mask[int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))] = False

    protected = np.zeros_like(mask)
    if landmark:
        mask[height // 2, width // 2] = True
        mask[0, width - 1] = True
        mask[0, 0] = False
        protected[height // 2, width // 2] = True
        protected[0, width - 1] = True

    cap = (height * width - 1) // 2
    candidates = np.argwhere(mask & ~protected)
    for index in rng.permutation(len(candidates))[:max(0, int(mask.sum()) - cap)]:
        row, col = candidates[int(index)]
        mask[row, col] = False
    if motif == "stair" and not landmark:
        # Restore a contiguous, nonincreasing staircase after the density cap.
        lengths = [int(np.count_nonzero(row)) for row in mask]
        lengths[0] = width
        for row in range(1, height):
            lengths[row] = min(lengths[row], lengths[row - 1])
        while sum(lengths) > cap:
            for row in range(height - 1, 0, -1):
                if lengths[row] and sum(lengths) > cap:
                    lengths[row] -= 1
        mask[:] = False
        for row, length in enumerate(lengths):
            mask[row, :length] = True

    if int(mask.sum()) < 3:
        for index in rng.permutation(height * width):
            r, c = divmod(int(index), width)
            if landmark and (r, c) == (0, 0):
                continue
            mask[r, c] = True
            if int(mask.sum()) >= 3:
                break
    if landmark:
        mask[0, 0] = False
        mask[0, width - 1] = True
        mask[height // 2, width // 2] = True
    return mask


def _background(grid: np.ndarray) -> int:
    counts = np.bincount(grid.reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(v) for v in grid.flat)


def _palette(rng: np.random.Generator, background: int, size: int, preferred: tuple[int, ...] = ()) -> tuple[int, ...]:
    colors = [int(v) for v in preferred if int(v) != background]
    colors = list(dict.fromkeys(colors))[:size]
    available = [v for v in rng.permutation(10) if int(v) != background and int(v) not in colors]
    colors.extend(int(v) for v in available[:size - len(colors)])
    return tuple(colors)


def _colored_grid(rng: np.random.Generator, shape: tuple[int, int], motif: str,
                  background: int, palette: tuple[int, ...], landmark: bool = False) -> np.ndarray:
    mask = _mask(rng, *shape, motif, landmark)
    if len(palette) > int(mask.sum()):
        palette = palette[:int(mask.sum())]
    positions = np.argwhere(mask)
    grid = np.full(shape, background, dtype=np.int8)
    order = rng.permutation(len(positions))
    for index, color in zip(order[:len(palette)], palette):
        r, c = positions[int(index)]
        grid[r, c] = color
    remaining = order[len(palette):]
    coloring = int(rng.integers(4))
    offset = int(rng.integers(len(palette)))
    for index in remaining:
        r, c = positions[int(index)]
        if coloring == 0:
            choice = int(rng.integers(len(palette)))
        elif coloring == 1:
            choice = (int(r) + offset) % len(palette)
        elif coloring == 2:
            choice = (int(c) + offset) % len(palette)
        else:
            choice = (int(r) + int(c) + offset) % len(palette)
        grid[r, c] = palette[choice]
    return grid


def _make_pair(rng: np.random.Generator, shape: tuple[int, int], motif: str,
               background: int, palette: tuple[int, ...], landmark: bool = False) -> dict:
    for _ in range(24):
        grid = _colored_grid(rng, shape, motif, background, palette, landmark)
        if np.array_equal(grid, grid[::-1, ::-1]):
            continue
        if landmark:
            rotated = grid[::-1, ::-1]
            alternatives = (grid, np.rot90(grid), np.rot90(grid, 3),
                            grid[::-1, :], grid[:, ::-1], grid.T, np.rot90(grid.T, 2))
            if any(np.array_equal(rotated, candidate) for candidate in alternatives):
                continue
        if _background(grid) != background:
            continue
        counts = np.bincount(grid.flat, minlength=10)
        if np.count_nonzero(counts == counts[background]) != 1:
            continue
        return {"input": grid, "output": grid[::-1, ::-1].copy()}
    # Rare tiny-grid fallback retains the same canvas transformation.
    grid = np.full(shape, background, dtype=np.int8)
    grid[0, -1] = palette[0]
    grid[shape[0] // 2, shape[1] // 2] = palette[-1]
    grid[1, 0] = palette[0]
    return {"input": grid, "output": grid[::-1, ::-1].copy()}


def _infer_dihedral_rule(training: list[dict]) -> str:
    operations = {
        "identity": lambda x: x,
        "quarter": lambda x: np.rot90(x),
        "half": lambda x: np.rot90(x, 2),
        "three_quarters": lambda x: np.rot90(x, 3),
        "row_reflection": lambda x: x[::-1, :],
        "column_reflection": lambda x: x[:, ::-1],
        "diagonal_reflection": lambda x: x.T,
        "other_diagonal_reflection": lambda x: np.rot90(x.T, 2),
    }
    fitting = [name for name, operation in operations.items()
               if all(np.array_equal(operation(pair["input"]), pair["output"])
                      for pair in training)]
    assert fitting == ["half"], "training must identify a unique canvas symmetry"
    return fitting[0]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    base_background = int(rng.integers(10))
    vary_background = bool(rng.random() < 0.55)
    layout_shift = bool(rng.integers(2))
    palette_shift = int(rng.integers(3))  # same, different, or expanded
    near_source_test = bool(rng.random() < 0.18)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_motifs = []
    first_shape = None
    first_palette = ()
    first_background = base_background

    for split, count in (("train", train_count), ("test", test_count)):
        for ordinal in range(count):
            absolute = len(seen)
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            if split == "train":
                if ordinal == 0:
                    motif = str(pair_rng.choice(_MOTIFS[1:]))
                elif ordinal == 1 and pair_rng.random() < 0.4:
                    motif = "stair"
                else:
                    motif = str(pair_rng.choice(_MOTIFS))
            elif layout_shift:
                choices = [m for m in _MOTIFS if m not in training_motifs]
                motif = str(pair_rng.choice(choices if choices else _MOTIFS))
            else:
                motif = str(pair_rng.choice(training_motifs))

            shape = _shape(pair_rng, absolute, first_shape)
            if split == "test" and ordinal == 0 and near_source_test:
                shape = (3, 3)
                motif = "stair"
            if absolute == 0:
                first_shape = shape
            background = base_background
            if vary_background and absolute > 0 and not (split == "test" and palette_shift == 0):
                background = (base_background + 1 + int(pair_rng.integers(9))) % 10

            if split == "train":
                size = int(pair_rng.integers(2, 6)) if ordinal == 0 else (1 if pair_rng.random() < 0.28 else int(pair_rng.integers(1, 8)))
                size = min(size, 9)
                preferred = (0,) if background != 0 and ordinal > 0 and pair_rng.random() < 0.55 else ()
                palette = _palette(pair_rng, background, size, preferred)
                if ordinal == 0:
                    first_palette = palette
            else:
                if ordinal == 0 and near_source_test:
                    background = 0
                    palette = _palette(pair_rng, background, int(pair_rng.integers(2, 5)))
                elif palette_shift == 0:
                    background = first_background
                    palette = first_palette
                elif palette_shift == 1:
                    other = [v for v in range(10) if v != background and v not in first_palette]
                    size = min(len(other), int(pair_rng.integers(1, 6)))
                    palette = _palette(pair_rng, background, size, tuple(other))
                else:
                    extra = int(pair_rng.integers(1, 4))
                    palette = _palette(pair_rng, background, min(9, len(first_palette) + extra), first_palette)

            for _ in range(12):
                pair = _make_pair(pair_rng, shape, motif, background, palette, split == "train" and ordinal == 0)
                key = _grid_key(pair["input"])
                if key not in seen and not any(np.array_equal(pair["input"], src) for src in _SOURCE_INPUTS):
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            puzzle[split].append(pair)
            seen.add(key)
            if split == "train":
                training_motifs.append(motif)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = pair["input"], pair["output"]
        for grid in (inp, out):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert all(1 <= side <= 30 for side in grid.shape)
            assert bool(np.all((grid >= 0) & (grid <= 9)))
        assert inp.shape == out.shape
        assert np.array_equal(out, inp[::-1, ::-1])
        assert not np.array_equal(inp, out)
        assert np.array_equal(np.bincount(inp.flat, minlength=10), np.bincount(out.flat, minlength=10))
        key = _grid_key(inp)
        assert key not in seen
        seen.add(key)
        assert not any(np.array_equal(inp, src) for src in _SOURCE_INPUTS)
    inferred = _infer_dihedral_rule(puzzle["train"])
    for pair in puzzle["test"]:
        prediction = np.rot90(pair["input"], 2) if inferred == "half" else None
        assert np.array_equal(prediction, pair["output"])


def _equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(17)
    assert _equal(automatic, generate_puzzle(17))
    assert not _equal(automatic, generate_puzzle(18))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(9137, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        assert _equal(puzzle, generate_puzzle(9137, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.5, 1), (2, True)):
        try:
            generate_puzzle(301, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    train_counts, test_counts, shapes, backgrounds = set(), set(), set(), set()
    staircases = single_color = zero_payload = holes = right_edge = bottom_edge = 0
    enclosed = separated = near_symmetric = small = large = wide_or_tall = 0
    near_source = combined = 0
    palette_layout_combinations = set()
    mixed_shapes = mixed_backgrounds = palette_growth = palette_reuse = 0
    masks = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        first_grid = puzzle["train"][0]["input"]
        second_grid = puzzle["train"][1]["input"]
        assert first_grid.shape[0] % 2 == first_grid.shape[1] % 2 == 1
        assert first_grid[first_grid.shape[0] // 2, first_grid.shape[1] // 2] != _background(first_grid)
        assert not _ferrers(first_grid != _background(first_grid))
        assert second_grid.shape[0] % 2 == 0 and first_grid.shape != second_grid.shape
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_shapes = set()
        pair_backgrounds = set()
        train_colors = set()
        first_grid = puzzle["train"][0]["input"]
        first_background = _background(first_grid)
        first_foreground = set(int(v) for v in np.unique(first_grid[first_grid != first_background]))
        train_categories = set()
        for pair in puzzle["train"]:
            training_grid = pair["input"]
            training_mask = training_grid != _background(training_grid)
            training_coords = np.argwhere(training_mask)
            low, high = training_coords.min(axis=0), training_coords.max(axis=0)
            train_categories.add((_ferrers(training_mask), _component_count(training_mask) == 1,
                                  bool(np.any(~training_mask[low[0]:high[0] + 1, low[1]:high[1] + 1]))))
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    assert out[h - 1 - r, w - 1 - c] == inp[r, c]
            bg = _background(inp)
            color_counts = np.bincount(inp.flat, minlength=10)
            assert np.count_nonzero(color_counts == color_counts.max()) == 1
            assert 3 <= int(np.count_nonzero(inp != bg)) <= (inp.size - 1) // 2
            mask = inp != bg
            shapes.add(inp.shape)
            backgrounds.add(bg)
            pair_shapes.add(inp.shape)
            pair_backgrounds.add(bg)
            masks.add((inp.shape, tuple(bool(v) for v in mask.flat)))
            staircases += _ferrers(mask)
            component_count = _component_count(mask)
            separated += component_count > 1
            single_color += len(set(int(v) for v in inp[mask])) == 1
            zero_payload += bg != 0 and bool(np.any(inp[mask] == 0))
            right_edge += bool(np.any(mask[:, -1]))
            bottom_edge += bool(np.any(mask[-1]))
            combined += component_count > 1 and bool(np.any(mask[:, -1])) and bool(np.any(mask[-1]))
            enclosed += _enclosed_hole(mask)
            near_symmetric += int(np.count_nonzero(mask ^ mask[::-1, ::-1])) == 2
            small += h <= 6 and w <= 6
            large += max(h, w) >= 20
            wide_or_tall += max(h, w) >= 2 * min(h, w)
            coords = np.argwhere(mask)
            top, left = coords.min(axis=0)
            bottom, right = coords.max(axis=0)
            hole = bool(np.any(~mask[top:bottom + 1, left:right + 1]))
            holes += hole
            if any(pair is test_pair for test_pair in puzzle["test"]):
                foreground = set(int(v) for v in np.unique(inp[mask]))
                if foreground == first_foreground:
                    palette_relation = "same"
                elif not foreground & first_foreground:
                    palette_relation = "different"
                elif foreground - first_foreground:
                    palette_relation = "grown"
                else:
                    palette_relation = "subset"
                category = (_ferrers(mask), component_count == 1, hole)
                palette_layout_combinations.add((category in train_categories, palette_relation))
                near_source += inp.shape == (3, 3) and bg == 0 and _ferrers(mask)
        for pair in puzzle["train"]:
            train_colors.update(int(v) for v in np.unique(pair["input"]))
        for pair in puzzle["test"]:
            test_colors = set(int(v) for v in np.unique(pair["input"]))
            palette_growth += bool(test_colors - train_colors)
            palette_reuse += bool(test_colors & train_colors)
        mixed_shapes += len(pair_shapes) > 1
        mixed_backgrounds += len(pair_backgrounds) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(masks) >= 150
    assert len(backgrounds) >= 8
    assert mixed_shapes == 42 and mixed_backgrounds >= 10
    assert staircases >= 5 and single_color >= 8 and zero_payload >= 8
    assert holes >= 30 and right_edge >= 20 and bottom_edge >= 20
    assert palette_growth >= 8 and palette_reuse >= 8
    assert enclosed >= 8 and separated >= 50 and near_symmetric >= 4
    assert small >= 15 and large >= 25 and wide_or_tall >= 30
    assert near_source >= 2 and combined >= 20
    assert (True, "same") in palette_layout_combinations
    assert (True, "different") in palette_layout_combinations
    assert (False, "same") in palette_layout_combinations
    assert (False, "grown") in palette_layout_combinations
