import numpy as np


_SOURCE_PAIRS = (
    (
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 3, 1, 2, 1, 1],
                [1, 1, 3, 1, 2, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [3, 0, 2],
                [3, 0, 2],
            ],
            dtype=np.int8,
        ),
    ),
    (
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1, 1, 1],
                [1, 2, 2, 3, 1, 1, 1],
                [1, 1, 1, 2, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [2, 2, 0],
                [2, 2, 3],
                [0, 0, 2],
            ],
            dtype=np.int8,
        ),
    ),
    (
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 5, 5, 1, 1, 1],
                [1, 5, 5, 1, 1, 1],
                [1, 6, 6, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [5, 5],
                [5, 5],
                [6, 6],
            ],
            dtype=np.int8,
        ),
    ),
    (
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 2, 1, 1],
                [1, 1, 2, 3, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ),
        np.array(
            [
                [0, 2],
                [2, 3],
            ],
            dtype=np.int8,
        ),
    ),
)


def _checked_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _checked_optional_count(
        num_test, "num_test", 1, 4
    )
    marker = abs(int(seed))

    if train_count is None and test_count is None:
        test_count = 1 + marker % 4
        base_train = 2 + (marker // 4) % 9
        train_count = max(base_train, test_count + 1)

    elif train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + marker % number_of_choices

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists")
        test_count = 1 + marker % maximum_test

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return any(
        np.array_equal(input_grid, source_input)
        and np.array_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _is_tight(mask: np.ndarray) -> bool:
    return (
        bool(np.any(mask[0, :]))
        and bool(np.any(mask[-1, :]))
        and bool(np.any(mask[:, 0]))
        and bool(np.any(mask[:, -1]))
    )


def _is_connected(mask: np.ndarray) -> bool:
    positions = np.argwhere(mask)
    if len(positions) == 0:
        return False

    start = (int(positions[0, 0]), int(positions[0, 1]))
    visited = {start}
    stack = [start]
    height, width = mask.shape

    while stack:
        row, column = stack.pop()
        for delta_row, delta_column in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            new_row = row + delta_row
            new_column = column + delta_column
            if (
                0 <= new_row < height
                and 0 <= new_column < width
                and bool(mask[new_row, new_column])
                and (new_row, new_column) not in visited
            ):
                visited.add((new_row, new_column))
                stack.append((new_row, new_column))

    return len(visited) == int(np.count_nonzero(mask))


def _components(mask):
    unseen = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    groups = []
    h, w = mask.shape
    while unseen:
        start = unseen.pop()
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                nr, nc = neighbor
                if 0 <= nr < h and 0 <= nc < w and neighbor in unseen:
                    unseen.remove(neighbor)
                    group.append(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _has_cavity(mask):
    blank = ~mask
    h, w = mask.shape
    reached = {(r, c) for r in range(h) for c in range(w)
               if blank[r, c] and (r in (0, h - 1) or c in (0, w - 1))}
    stack = list(reached)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and blank[nr, nc] and (nr, nc) not in reached:
                reached.add((nr, nc))
                stack.append((nr, nc))
    return int(blank.sum()) > len(reached)


def _shape(rng, test):
    q = rng.random()
    if q < (0.10 if test else 0.06):
        return (2, 2)
    if q < (0.24 if test else 0.19):
        length = int(rng.integers(3, 10))
        return (1, length) if rng.random() < 0.5 else (length, 1)
    if q < 0.48:
        return int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if q < 0.81:
        return int(rng.integers(2, 7)), int(rng.integers(2, 7))
    return int(rng.integers(4, 10)), int(rng.integers(4, 10))


def _mask(rng, shape, style):
    h, w = shape
    if style == "small_l" and shape == (2, 2):
        result = np.ones(shape, dtype=bool)
        result[int(rng.integers(0, 2)), int(rng.integers(0, 2))] = False
        return result
    if style == "solid":
        return np.ones(shape, dtype=bool)
    if style == "bars":
        axes = [axis for axis, length in enumerate(shape) if length >= 3]
        if axes:
            axis = axes[int(rng.integers(0, len(axes)))]
            length = shape[axis]
            stripe = np.zeros(length, dtype=bool)
            stripe[0] = stripe[-1] = True
            if length >= 5 and rng.random() < 0.5:
                candidates = np.arange(2, length - 2)
                extra = int(rng.integers(1, min(3, len(candidates)) + 1))
                stripe[rng.choice(candidates, size=extra, replace=False)] = True
            array = stripe[:, None] if axis == 0 else stripe[None, :]
            return np.broadcast_to(array, shape).copy()
    if style == "hollow" and h >= 3 and w >= 3:
        result = np.ones(shape, dtype=bool)
        r0 = int(rng.integers(1, h - 1))
        c0 = int(rng.integers(1, w - 1))
        r1 = int(rng.integers(r0 + 1, h))
        c1 = int(rng.integers(c0 + 1, w))
        result[r0:r1, c0:c1] = False
        return result
    if style == "bent" and h >= 2 and w >= 2:
        result = np.zeros(shape, dtype=bool)
        row = 0 if rng.random() < 0.5 else h - 1
        col = 0 if rng.random() < 0.5 else w - 1
        result[row, :] = True
        result[:, col] = True
        if h >= 4 and w >= 4 and rng.random() < 0.55:
            next_row = int(rng.integers(1, h - 1))
            result[next_row, min(col + 1, w - 1):] = True
        return result
    if style == "blocks" and h >= 2 and w >= 2:
        result = np.ones(shape, dtype=bool)
        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(1, max(2, h // 2 + 1)))
            cw = int(rng.integers(1, max(2, w // 2 + 1)))
            r0 = 0 if rng.random() < 0.5 else h - rh
            c0 = 0 if rng.random() < 0.5 else w - cw
            result[r0:r0 + rh, c0:c0 + cw] = False
        if result.sum() >= 2 and _is_tight(result) and not result.all():
            return result
    for _ in range(80):
        result = rng.random(shape) < rng.uniform(0.30, 0.78)
        if result.sum() >= 2 and _is_tight(result) and not result.all():
            return result
    result = np.zeros(shape, dtype=bool)
    result[0, 0] = result[-1, -1] = True
    if shape == (2, 2):
        result[0, 1] = True
    return result


def _style(rng, shape, test):
    h, w = shape
    if test and shape == (2, 2) and rng.random() < 0.5:
        return "small_l"
    if h >= 3 and w >= 3:
        return str(rng.choice(
            ("solid", "bars", "bent", "blocks", "irregular", "hollow"),
            p=(0.12, 0.18, 0.15, 0.15, 0.22, 0.18),
        ))
    if max(shape) < 3:
        return str(rng.choice(("solid", "bent", "irregular"), p=(0.3, 0.4, 0.3)))
    return str(rng.choice(
        ("solid", "bars", "bent", "blocks", "irregular"),
        p=(0.16, 0.21, 0.18, 0.18, 0.27),
    ))


def _palette(rng, background, mask, earlier, train_colors, test):
    available = [c for c in range(1, 10) if c != background]
    if earlier and rng.random() < (0.33 if test else 0.27):
        possible = [p for p in earlier if background not in p and len(p) <= int(mask.sum())]
        if possible:
            return possible[int(rng.integers(0, len(possible)))]
    size = min(
        int(rng.choice((1, 2, 3, 4), p=(0.17, 0.47, 0.26, 0.10))),
        int(mask.sum()),
    )
    if test and rng.random() < 0.43:
        novel = [c for c in available if c not in train_colors]
        if novel:
            first = int(rng.choice(novel))
            rest = [c for c in available if c != first]
            return (first,) + tuple(int(v) for v in rng.choice(rest, size=size - 1, replace=False))
    return tuple(int(v) for v in rng.choice(available, size=size, replace=False))


def _colorize(rng, mask, palette, background):
    result = np.full(mask.shape, background, dtype=np.int8)
    positions = np.argwhere(mask)
    count = len(palette)
    if count == 1:
        result[mask] = palette[0]
        return result
    rows, cols = positions[:, 0], positions[:, 1]
    modes, weights = ["rows", "columns", "patches", "mixed"], [0.25, 0.25, 0.30, 0.20]
    groups = _components(mask)
    if len(groups) >= count:
        modes.append("components")
        weights = [0.20, 0.20, 0.24, 0.14, 0.22]
    mode = str(rng.choice(modes, p=weights))
    if mode == "rows" and len(np.unique(rows)) < count:
        mode = "patches"
    if mode == "columns" and len(np.unique(cols)) < count:
        mode = "patches"
    if mode == "rows":
        labels = {int(v): min(count - 1, i * count // len(np.unique(rows)))
                  for i, v in enumerate(np.unique(rows))}
        values = [palette[labels[int(r)]] for r in rows]
    elif mode == "columns":
        labels = {int(v): min(count - 1, i * count // len(np.unique(cols)))
                  for i, v in enumerate(np.unique(cols))}
        values = [palette[labels[int(c)]] for c in cols]
    elif mode == "components":
        for i, group in enumerate(groups):
            for r, c in group:
                result[r, c] = palette[i % count]
        return result
    elif mode == "patches":
        anchors = positions[rng.choice(len(positions), size=count, replace=False)]
        distance = np.abs(positions[:, None, :] - anchors[None, :, :]).sum(axis=2)
        values = [palette[int(i)] for i in np.argmin(distance, axis=1)]
    else:
        values = [palette[int(i)] for i in rng.integers(0, count, size=len(positions))]
        for i, position in enumerate(rng.choice(len(positions), size=count, replace=False)):
            values[int(position)] = palette[i]
    for (r, c), value in zip(positions, values):
        result[r, c] = value
    # Keep every selected color visible if a short band happened to miss one.
    missing = [color for color in palette if color not in result[mask]]
    if missing:
        for position, color in zip(rng.choice(len(positions), size=len(missing), replace=False), missing):
            r, c = positions[int(position)]
            result[r, c] = color
    return result


def _margins(rng):
    margins = [int(v) for v in rng.integers(1, 8, size=4)]
    if rng.random() < 0.21:
        side = int(rng.integers(0, 4))
        margins[side] = 0
        for offset in (1, 2, 3):
            other = (side + offset) % 4
            margins[other] = max(margins[other], int(rng.integers(2, 7)))
    elif rng.random() < 0.31:
        margins[int(rng.integers(0, 4))] = 1
        margins[int(rng.integers(0, 4))] = int(rng.integers(5, 9))
    return margins


def _pair(rng, mask, palette, background):
    crop = _colorize(rng, mask, palette, background)
    top, bottom, left, right = _margins(rng)
    h, w = mask.shape
    inp = np.full((top + h + bottom, left + w + right), background, dtype=np.int8)
    inp[top:top + h, left:left + w] = crop
    out = np.where(crop == background, 0, crop).astype(np.int8)
    return {"input": inp, "output": out}


def _background(inp):
    edge = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
    return int(np.argmax(np.bincount(edge.astype(int), minlength=10)))


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    base_background = int(rng.integers(1, 10))
    train_plan = ["bars", "solid"] + [None] * (train_count - 2)
    rng.shuffle(train_plan)
    puzzle = {"train": [], "test": []}
    seen = set()
    earlier, train_colors = [], set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(150):
                local = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                test = section == "test"
                shape = _shape(local, test)
                if not test and train_plan[index] == "bars" and max(shape) < 3:
                    continue
                style = _style(local, shape, test) if test or train_plan[index] is None else train_plan[index]
                mask = _mask(local, shape, style)
                if not test and train_plan[index] == "bars" and _is_connected(mask):
                    continue
                background = base_background if local.random() < 0.52 else int(
                    local.choice([c for c in range(1, 10) if c != base_background])
                )
                palette = _palette(local, background, mask, earlier, train_colors, test)
                pair = _pair(local, mask, palette, background)
                inp, out = pair["input"], pair["output"]
                key = _grid_key(inp)
                if key in seen or _matches_source_pair(inp, out) or _background(inp) != background:
                    continue
                puzzle[section].append(pair)
                seen.add(key)
                if not test:
                    colors = tuple(int(c) for c in np.unique(out) if c != 0)
                    earlier.append(colors)
                    train_colors.update(colors)
                break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    train_gap = train_solid = train_disconnected = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            ikey = _grid_key(inp)
            pkey = _pair_key(inp, out)
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert not np.array_equal(inp, out) and not _matches_source_pair(inp, out)
            background = _background(inp)
            assert background != 0
            edge = np.concatenate((inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]))
            assert np.count_nonzero(edge == background) > len(edge) // 2
            positions = np.argwhere(inp != background)
            assert len(positions) >= 2
            r0, c0 = positions.min(axis=0)
            r1, c1 = positions.max(axis=0)
            crop = inp[r0:r1 + 1, c0:c1 + 1]
            mask = crop != background
            assert _is_tight(mask)
            assert np.array_equal(out, np.where(mask, crop, 0))
            assert not np.any(inp == 0)
            outside = np.ones(inp.shape, dtype=bool)
            outside[r0:r1 + 1, c0:c1 + 1] = False
            assert np.all(inp[outside] == background)
            margins = (int(r0), int(inp.shape[0] - r1 - 1), int(c0), int(inp.shape[1] - c1 - 1))
            assert sum(value == 0 for value in margins) <= 1
            if section == "train":
                train_gap |= bool(np.any(~mask))
                train_solid |= bool(np.all(mask))
                train_disconnected |= not _is_connected(mask)
    assert train_gap and train_solid and train_disconnected


def _puzzle_fingerprint(puzzle):
    return tuple(tuple(_pair_key(p["input"], p["output"]) for p in puzzle[section])
                 for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    train_shapes, test_shapes, masks, backgrounds, margins = set(), set(), set(), set(), set()
    metrics = {key: 0 for key in (
        "test_solid", "test_disconnected", "test_gap", "test_l", "test_novel",
        "one_color", "many_colors", "cavity", "contact", "reused", "variable_background",
        "variable_train_shape", "variable_train_palette", "test_at_least_max_train",
        "asymmetric", "thin", "large", "shape_2x4", "shape_4x2", "shape_4x4",
        "large_disconnected_multicolor_multigap", "same_color_line",
    )}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = {int(v) for p in puzzle["train"] for v in np.unique(p["output"]) if v != 0}
        train_palettes = {frozenset(int(v) for v in np.unique(p["output"]) if v != 0)
                          for p in puzzle["train"]}
        metrics["variable_background"] += len({_background(p["input"]) for p in puzzle["train"]}) > 1
        metrics["variable_train_shape"] += len({p["output"].shape for p in puzzle["train"]}) > 1
        metrics["variable_train_palette"] += len(train_palettes) > 1
        metrics["test_at_least_max_train"] += any(
            p["output"].size >= max(q["output"].size for q in puzzle["train"])
            for p in puzzle["test"]
        )
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg = _background(inp)
                backgrounds.add(bg)
                pos = np.argwhere(inp != bg)
                r0, c0 = pos.min(axis=0)
                r1, c1 = pos.max(axis=0)
                observed = inp[r0:r1 + 1, c0:c1 + 1].copy()
                observed[observed == bg] = 0
                assert np.array_equal(observed, out)
                mask = out != 0
                colors = frozenset(int(v) for v in np.unique(out) if v != 0)
                metrics["one_color"] += len(colors) == 1
                metrics["many_colors"] += len(colors) >= 3
                metrics["cavity"] += _has_cavity(mask)
                side_margins = (int(r0), int(inp.shape[0] - r1 - 1),
                                int(c0), int(inp.shape[1] - c1 - 1))
                metrics["contact"] += 0 in side_margins
                metrics["asymmetric"] += max(side_margins) - min(side_margins) >= 4
                metrics["thin"] += min(out.shape) == 1
                metrics["large"] += out.size >= 25
                metrics["shape_2x4"] += out.shape == (2, 4)
                metrics["shape_4x2"] += out.shape == (4, 2)
                metrics["shape_4x4"] += out.shape == (4, 4)
                metrics["same_color_line"] += any(
                    len({int(v) for v in line if v != 0}) == 1 and np.count_nonzero(line) >= 2
                    for line in list(out) + list(out.T)
                )
                margins.add(side_margins)
                masks.add((out.shape, tuple(int(v) for v in mask.flat)))
                if section == "train":
                    train_shapes.add(out.shape)
                else:
                    test_shapes.add(out.shape)
                    metrics["test_solid"] += bool(mask.all())
                    metrics["test_disconnected"] += not _is_connected(mask)
                    metrics["test_gap"] += bool(np.any(~mask))
                    metrics["test_l"] += out.shape == (2, 2) and mask.sum() == 3
                    metrics["test_novel"] += bool(colors - train_colors)
                    metrics["reused"] += colors in train_palettes
                    metrics["large_disconnected_multicolor_multigap"] += (
                        out.size >= 16 and not _is_connected(mask)
                        and len(colors) >= 3 and np.count_nonzero(~mask) >= 2
                    )
    assert len(train_shapes) >= 25 and len(test_shapes) >= 20
    assert len(masks) >= 100 and len(backgrounds) >= 7 and len(margins) >= 100
    assert min(metrics.values()) >= 2, metrics
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + offset, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for offset, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _puzzle_fingerprint(generate_puzzle(314159, 7, 3)) == _puzzle_fingerprint(
        generate_puzzle(314159, 7, 3))
    assert len({_puzzle_fingerprint(generate_puzzle(4000 + seed, 6, 2))
                for seed in range(12)}) == 12
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0}, {"num_test": True}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {kwargs}")
