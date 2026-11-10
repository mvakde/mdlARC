import numpy as np


_SOURCE_ROW_PAIRS = (
    (
        ("600", "250", "210"),
        ("012", "052", "006"),
    ),
    (
        (
            "000000000",
            "000000000",
            "000000000",
            "003250000",
            "003300000",
            "014000000",
            "000000000",
        ),
        (
            "000000000",
            "000000410",
            "000003300",
            "000052300",
            "000000000",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "033500000",
            "110000000",
            "200000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
        ),
        (
            "000000200",
            "000001100",
            "000533000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "800050000",
            "033200000",
            "032200000",
            "100000000",
        ),
        (
            "000000000",
            "000000000",
            "000000001",
            "000002230",
            "000002330",
            "000050008",
            "000000000",
            "000000000",
            "000000000",
        ),
    ),
)


def _array_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_array_from_rows(input_rows), _array_from_rows(output_rows))
    for input_rows, output_rows in _SOURCE_ROW_PAIRS
)


def _checked_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train", 2, 10)
    test_count = _checked_count(num_test, "num_test", 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        seed_index = int(seed)
        test_count = 1 + seed_index % 4
        train_span = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed_index // 4) % train_span
        )
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        lower = max(2, test_count + 1)
        train_count = int(rng.integers(lower, 11))
        return train_count, test_count

    assert test_count is None
    upper = min(4, train_count - 1)
    if upper < 1:
        raise ValueError("no legal num_test exists for num_train")
    test_count = int(rng.integers(1, upper + 1))
    return train_count, test_count


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _component_sizes(coordinates) -> list[int]:
    remaining = set(coordinates)
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _has_filled_2_by_2(coordinates) -> bool:
    cells = set(coordinates)
    return any({(r, c), (r + 1, c), (r, c + 1),
                (r + 1, c + 1)} <= cells for r, c in cells)


def _has_enclosed_hole(coordinates) -> bool:
    cells = set(coordinates)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    remaining = {(r, c) for r in range(min_r, max_r + 1)
                 for c in range(min_c, max_c + 1) if (r, c) not in cells}
    while remaining:
        stack = [remaining.pop()]
        touches_boundary = False
        while stack:
            r, c = stack.pop()
            touches_boundary |= r in (min_r, max_r) or c in (min_c, max_c)
            for neighbor in _neighbors((r, c)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        if not touches_boundary:
            return True
    return False


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.astype(np.int64).ravel(), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    return int(np.argmax(counts))


def _rotate_grid(grid: np.ndarray, background: int,
                 pivot: tuple[int, int]) -> np.ndarray:
    result = np.full(grid.shape, background, dtype=grid.dtype)
    pr, pc = pivot
    for row, col in np.argwhere(grid != background):
        target = (2 * pr - int(row), 2 * pc - int(col))
        assert 0 <= target[0] < grid.shape[0]
        assert 0 <= target[1] < grid.shape[1]
        result[target] = grid[row, col]
    return result


def _window(rng: np.random.Generator, regime: str,
            informative: bool) -> tuple[int, int, int, int]:
    if informative:
        row_span = int(rng.integers(2, 6))
        col_span = int(rng.integers(2, 6))
        above = int(rng.integers(1, row_span))
        left = int(rng.integers(1, col_span))
    else:
        limit = 7 if regime == "source_train" else 9
        row_span = int(rng.integers(2, limit + 1))
        col_span = int(rng.integers(2, limit + 1))
        above = int(rng.integers(0, row_span + 1))
        left = int(rng.integers(0, col_span + 1))
    return -above, row_span - above, -left, col_span - left


def _neighbors(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _sample_offsets(rng: np.random.Generator, bounds: tuple[int, ...],
                    mode: str) -> tuple[tuple[int, int], ...]:
    min_r, max_r, min_c, max_c = bounds
    available = {(r, c) for r in range(min_r, max_r + 1)
                 for c in range(min_c, max_c + 1) if (r, c) != (0, 0)}
    ordered = sorted(available)
    if mode in ("connected", "path"):
        chosen = {ordered[int(rng.integers(len(ordered)))]}
        target = int(rng.integers(4, min(len(ordered), 24) + 1))
        last = next(iter(chosen))
        while len(chosen) < target:
            frontier = sorted({n for p in chosen for n in _neighbors(p)
                               if n in available and n not in chosen})
            local = [n for n in _neighbors(last)
                     if n in available and n not in chosen]
            choices = local if mode == "path" and local and rng.random() < .82 else frontier
            next_cell = choices[int(rng.integers(len(choices)))]
            chosen.add(next_cell)
            last = next_cell
    elif mode == "sparse":
        target = int(rng.integers(3, min(len(ordered), 8,
                                         max(3, len(ordered) // 5)) + 1))
        indices = rng.choice(len(ordered), size=target, replace=False)
        chosen = {ordered[int(i)] for i in indices}
    elif mode == "clusters":
        cluster_count = int(rng.integers(2, 5))
        seeds = [ordered[int(rng.integers(len(ordered)))]]
        for _ in range(cluster_count - 1):
            scores = [min(abs(p[0] - q[0]) + abs(p[1] - q[1])
                          for q in seeds) for p in ordered]
            far = [p for p, score in zip(ordered, scores)
                   if score == max(scores) and p not in seeds]
            if far:
                seeds.append(far[int(rng.integers(len(far)))])
        chosen = set(seeds)
        target = int(rng.integers(max(5, len(chosen)),
                                  min(len(ordered), 25) + 1))
        while len(chosen) < target:
            center = seeds[int(rng.integers(len(seeds)))]
            frontier = sorted({n for p in chosen for n in _neighbors(p)
                               if n in available and n not in chosen})
            if not frontier:
                break
            distances = [abs(p[0] - center[0]) + abs(p[1] - center[1])
                         for p in frontier]
            near = [p for p, d in zip(frontier, distances)
                    if d <= min(distances) + 1]
            chosen.add(near[int(rng.integers(len(near)))])
    elif mode == "dense":
        fraction = float(rng.uniform(.50, .90))
        target = max(4, min(len(ordered), round(len(ordered) * fraction)))
        indices = rng.choice(len(ordered), size=target, replace=False)
        chosen = {ordered[int(i)] for i in indices}
    elif mode == "hollow":
        chosen = {p for p in available
                  if p[0] in (min_r, max_r) or p[1] in (min_c, max_c)}
        removable = sorted(chosen)
        rng.shuffle(removable)
        for p in removable[:int(rng.integers(0, max(1, len(removable) // 5) + 1))]:
            if len(chosen) > 5:
                chosen.remove(p)
        interior = sorted(available - chosen)
        if interior:
            extra = int(rng.integers(0, min(4, len(interior)) + 1))
            if extra:
                chosen.update(interior[int(i)] for i in rng.choice(
                    len(interior), size=extra, replace=False))
    else:
        raise ValueError("unknown shape mode")
    return tuple(sorted(chosen))


def _informative(offsets: tuple[tuple[int, int], ...]) -> bool:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    cells = set(offsets)
    return (len(cells) >= 5 and min(rows) < 0 < max(rows)
            and min(cols) < 0 < max(cols)
            and cells != {(-r, -c) for r, c in cells}
            and cells != {(-r, c) for r, c in cells}
            and cells != {(r, -c) for r, c in cells}
            and any(r != 0 and c != 0 for r, c in cells))


def _canvas(rng: np.random.Generator,
            offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    row_radius = max(abs(r) for r, _ in offsets)
    col_radius = max(abs(c) for _, c in offsets)
    base_h, base_w = 2 * row_radius + 1, 2 * col_radius + 1
    style = int(rng.integers(5))
    ranges = ((0, 3, 0, 3), (3, 13, 3, 13), (7, 15, 0, 4),
              (0, 4, 7, 15), (1, 9, 1, 9))
    lo_h, hi_h, lo_w, hi_w = ranges[style]
    extra_h = int(rng.integers(lo_h, min(hi_h, 31 - base_h)))
    extra_w = int(rng.integers(lo_w, min(hi_w, 31 - base_w)))
    height, width = base_h + extra_h, base_w + extra_w
    while height * width <= 2 * (len(offsets) + 1):
        if height <= width and height < 30 or width == 30:
            height += 1
            extra_h += 1
        else:
            width += 1
            extra_w += 1
    pivot_row = row_radius + int(rng.integers(extra_h + 1))
    pivot_col = col_radius + int(rng.integers(extra_w + 1))
    return height, width, pivot_row, pivot_col


def _make_geometry(rng: np.random.Generator, regime: str,
                   informative: bool, used_layouts: set) -> dict:
    modes = ("connected", "path", "sparse", "clusters", "dense", "hollow")
    for attempt in range(120):
        mode = ("connected", "path")[int(rng.integers(2))] if informative else modes[int(rng.integers(len(modes)))]
        bounds = _window(rng, regime, informative)
        offsets = _sample_offsets(rng, bounds, mode)
        if len(offsets) < 3 or offsets in used_layouts:
            continue
        rows = [r for r, _ in offsets]
        cols = [c for _, c in offsets]
        if min(rows) == max(rows) or min(cols) == max(cols):
            continue
        if set(offsets) == {(-r, -c) for r, c in offsets}:
            continue
        if informative and not _informative(offsets):
            continue
        height, width, pr, pc = _canvas(rng, offsets)
        return {"offsets": offsets, "height": height,
                "width": width, "pivot": (pr, pc)}
    # At most fourteen layouts are needed; this deterministic set has 81.
    for row_radius in range(1, 10):
        for col_radius in range(1, 10):
            fallback = ((-row_radius, 0), (-row_radius, col_radius),
                        (0, -col_radius), (0, col_radius),
                        (row_radius, -col_radius), (row_radius, 0),
                        (row_radius, col_radius))
            if fallback in used_layouts:
                continue
            height, width, pr, pc = _canvas(rng, fallback)
            return {"offsets": fallback, "height": height,
                    "width": width, "pivot": (pr, pc)}
    raise AssertionError("unreachable layout exhaustion")


def _assigned_colors(rng: np.random.Generator,
                     offsets: tuple[tuple[int, int], ...],
                     palette: tuple[int, ...]) -> dict:
    mode = int(rng.integers(4))
    cells = list(offsets)
    if len(palette) == 1:
        return {cell: palette[0] for cell in cells}
    if mode == 0:  # scattered colors
        values = list(palette)
        values += [palette[int(rng.integers(len(palette)))]
                   for _ in range(len(cells) - len(palette))]
        rng.shuffle(values)
        return dict(zip(cells, values))
    if mode == 1:  # a main color with isolated accents
        assigned = {cell: palette[0] for cell in cells}
        for color, index in zip(palette[1:], rng.choice(
                len(cells), size=len(palette) - 1, replace=False)):
            assigned[cells[int(index)]] = color
        return assigned
    if mode == 2:  # compact same-color patches
        indices = rng.choice(len(cells), size=len(palette), replace=False)
        centers = [cells[int(i)] for i in indices]
        assigned = {}
        for cell in cells:
            distance = [abs(cell[0] - p[0]) + abs(cell[1] - p[1])
                        for p in centers]
            assigned[cell] = palette[int(np.argmin(distance))]
        return assigned
    axis = int(rng.integers(2))  # ordered row or column runs
    ranked = sorted(cells, key=lambda p: (p[axis], p[1 - axis]))
    assigned = {}
    for index, cell in enumerate(ranked):
        color_index = min(len(palette) - 1,
                          index * len(palette) // len(ranked))
        assigned[cell] = palette[color_index]
    return assigned


def _build_pair(rng: np.random.Generator, geometry: dict,
                background: int, pivot_color: int,
                palette: tuple[int, ...]) -> dict:
    grid = np.full((geometry["height"], geometry["width"]),
                   background, dtype=np.int8)
    pr, pc = geometry["pivot"]
    grid[pr, pc] = pivot_color
    for (dr, dc), color in _assigned_colors(
            rng, geometry["offsets"], palette).items():
        grid[pr + dr, pc + dc] = color
    output = _rotate_grid(grid, background, (pr, pc))
    assert not np.array_equal(grid, output)
    return {"input": grid, "output": output}


def _sample_palette(rng: np.random.Generator, pool: tuple[int, ...],
                    maximum: int, required: int | None = None,
                    minimum: int = 1) -> tuple[int, ...]:
    choices = [c for c in pool if c != required]
    limit = min(maximum, len(choices) + (required is not None))
    count = int(rng.integers(minimum, limit + 1))
    others = count - (required is not None)
    result = ([required] if required is not None else [])
    if others:
        result += [choices[int(i)] for i in rng.choice(
            len(choices), size=others, replace=False)]
    return tuple(result)


def _candidate_marker_colors(pair: dict, background: int) -> set[int]:
    inp, out = pair["input"], pair["output"]
    return {int(color) for color in np.unique(inp)
            if color != background and np.count_nonzero(inp == color) == 1
            and np.count_nonzero(out == color) == 1
            and np.array_equal(np.argwhere(inp == color),
                               np.argwhere(out == color))}


def _analyze_pair(pair: dict, background: int,
                  pivot_color: int) -> dict:
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    assert np.count_nonzero(inp == pivot_color) == 1
    assert np.count_nonzero(out == pivot_color) == 1
    pivot = tuple(int(x) for x in np.argwhere(inp == pivot_color)[0])
    assert tuple(int(x) for x in np.argwhere(out == pivot_color)[0]) == pivot
    assert np.array_equal(_rotate_grid(inp, background, pivot), out)
    assert not np.array_equal(inp, out)
    payload = {(int(r) - pivot[0], int(c) - pivot[1])
               for r, c in np.argwhere((inp != background) &
                                       (inp != pivot_color))}
    assert len(payload) >= 3
    assert all((-r + pivot[0] >= 0 and -r + pivot[0] < inp.shape[0]
                and -c + pivot[1] >= 0 and -c + pivot[1] < inp.shape[1])
               for r, c in payload)
    palette = {int(inp[pivot[0] + r, pivot[1] + c]) for r, c in payload}
    assert background not in palette and pivot_color not in palette
    assert len(palette) <= 5
    rows = [r for r, _ in payload]
    cols = [c for _, c in payload]
    assert min(rows) < max(rows) and min(cols) < max(cols)
    footprint = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return {"offsets": tuple(sorted(payload)), "palette": frozenset(palette),
            "components": _component_sizes(payload),
            "density": len(payload) / footprint,
            "footprint": footprint,
            "overlap": bool(payload & {(-r, -c) for r, c in payload}),
            "filled_square": _has_filled_2_by_2(payload),
            "pivot": pivot}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    backgrounds = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            bg = _dominant_color(inp)
            assert bg == _dominant_color(out)
            assert np.count_nonzero(inp == bg) > inp.size // 2
            backgrounds.append(bg)
            key, pair_key = _grid_key(inp), _pair_key(inp, out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(inp, out)
    candidates = None
    for pair in puzzle["train"]:
        bg = _dominant_color(pair["input"])
        local = _candidate_marker_colors(pair, bg)
        candidates = local if candidates is None else candidates & local
    assert candidates is not None and len(candidates) == 1
    pivot_color = next(iter(candidates))
    analyses = {"train": [], "test": []}
    layouts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            bg = _dominant_color(pair["input"])
            assert bg != pivot_color
            info = _analyze_pair(pair, bg, pivot_color)
            assert info["offsets"] not in layouts
            layouts.add(info["offsets"])
            analyses[split].append(info)
    assert backgrounds[0] != backgrounds[1]
    assert analyses["train"][0]["palette"] != analyses["train"][1]["palette"]
    assert any(_informative(info["offsets"]) for info in analyses["train"])
    train_colors = set().union(*(info["palette"] for info in analyses["train"]))
    assert any(info["palette"] - train_colors for info in analyses["test"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    pivot_color = int(rng.integers(10))
    all_payload = tuple(c for c in range(10) if c != pivot_color)
    geometries = {"train": [], "test": []}
    used_layouts = set()
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            geometry = _make_geometry(pair_rng, regime,
                                      split == "train" and index == 0,
                                      used_layouts)
            geometries[split].append((geometry, pair_rng))
            used_layouts.add(geometry["offsets"])
    first_limit = min(4, len(geometries["train"][0][0]["offsets"]))
    first_palette = _sample_palette(rng, all_payload, first_limit, minimum=2)
    remainder = tuple(c for c in all_payload if c not in first_palette)
    second_limit = min(3, len(geometries["train"][1][0]["offsets"]))
    second_palette = _sample_palette(rng, remainder, second_limit)
    train_pool = tuple(sorted(set(first_palette) | set(second_palette)))
    unseen = tuple(c for c in all_payload if c not in train_pool)
    assert unseen
    palettes = {"train": [first_palette, second_palette], "test": []}
    for geometry, _ in geometries["train"][2:]:
        palettes["train"].append(_sample_palette(
            rng, train_pool, min(5, len(geometry["offsets"]))))
    for index, (geometry, _) in enumerate(geometries["test"]):
        required = unseen[int(rng.integers(len(unseen)))] if index == 0 else None
        palettes["test"].append(_sample_palette(
            rng, all_payload, min(5, len(geometry["offsets"])), required))
    puzzle = {"train": [], "test": []}
    first_background = None
    for split in ("train", "test"):
        for index, ((geometry, pair_rng), palette) in enumerate(
                zip(geometries[split], palettes[split])):
            bg_choices = [c for c in range(10)
                          if c != pivot_color and c not in palette
                          and not (split == "train" and index == 1
                                   and c == first_background)]
            background = bg_choices[int(rng.integers(len(bg_choices)))]
            if split == "train" and index == 0:
                first_background = background
            pair = _build_pair(pair_rng, geometry, background,
                               pivot_color, palette)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _predict_from_training(train: list, test_input: np.ndarray) -> np.ndarray:
    candidates = None
    for pair in train:
        bg = _dominant_color(pair["input"])
        local = _candidate_marker_colors(pair, bg)
        candidates = local if candidates is None else candidates & local
    assert candidates is not None and len(candidates) == 1
    marker = next(iter(candidates))
    assert np.count_nonzero(test_input == marker) == 1
    pivot = tuple(int(x) for x in np.argwhere(test_input == marker)[0])
    return _rotate_grid(test_input, _dominant_color(test_input), pivot)


def self_test() -> None:
    fixed = [generate_puzzle(seed) for seed in range(4200, 4242)]
    for seed, puzzle in zip(range(4200, 4242), fixed):
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            prediction = _predict_from_training(
                [{k: v.copy() for k, v in p.items()}
                 for p in puzzle["train"]], pair["input"].copy())
            assert np.array_equal(prediction, pair["output"]), seed
    assert len({len(p["train"]) for p in fixed}) >= 4
    assert len({len(p["test"]) for p in fixed}) >= 3
    for nt, ns in ((2, 1), (10, 1), (10, 4), (6, 2)):
        puzzle = generate_puzzle(123 + nt + ns, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        validate_generated_puzzle(puzzle)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(987, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3},
                   {"num_train": True}, {"num_test": 2.5}):
        try:
            generate_puzzle(5, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    assert _puzzles_equal(generate_puzzle(777, 7, 3),
                          generate_puzzle(777, 7, 3))
    assert not _puzzles_equal(generate_puzzle(777, 7, 3),
                              generate_puzzle(778, 7, 3))
    shapes, layouts, backgrounds, pivot_colors = set(), set(), set(), set()
    palette_sizes, component_counts, filled = set(), set(), set()
    row_signs, col_signs = set(), set()
    metrics = {name: 0 for name in (
        "one_color", "sparse", "dense", "holes", "same_color_run",
        "isolated_accent", "same_color_opposite", "different_color_opposite",
        "pivot_inside", "one_color_overlap", "disconnected_multidirection",
        "tight", "roomy", "tall", "wide", "test_connected",
        "test_disconnected", "connected_train_disconnected_test",
        "disconnected_train_connected_test")}
    all_pairs = 0
    for puzzle in fixed:
        marker = next(iter(set.intersection(*[
            _candidate_marker_colors(p, _dominant_color(p["input"]))
            for p in puzzle["train"]])))
        pivot_colors.add(marker)
        assert _dominant_color(puzzle["train"][0]["input"]) != (
            _dominant_color(puzzle["train"][1]["input"]))
        puzzle_backgrounds, puzzle_palettes = set(), set()
        train_components, test_components = [], []
        train_colors, test_colors = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                all_pairs += 1
                inp = pair["input"]
                bg = _dominant_color(inp)
                info = _analyze_pair(pair, bg, marker)
                backgrounds.add(bg)
                shapes.add(inp.shape)
                layouts.add(info["offsets"])
                palette_sizes.add(len(info["palette"]))
                component_counts.add(len(info["components"]))
                filled.add(info["filled_square"])
                puzzle_backgrounds.add(bg)
                puzzle_palettes.add(info["palette"])
                if split == "train":
                    train_components.append(len(info["components"]))
                    train_colors.update(info["palette"])
                else:
                    test_components.append(len(info["components"]))
                    test_colors.update(info["palette"])
                offsets = set(info["offsets"])
                pr, pc = info["pivot"]
                colors = {offset: int(inp[pr + offset[0], pc + offset[1]])
                          for offset in offsets}
                rows = [r for r, _ in offsets]
                cols = [c for _, c in offsets]
                inside = (min(rows) < 0 < max(rows)
                          and min(cols) < 0 < max(cols))
                extra_h = inp.shape[0] - 2 * max(abs(r) for r in rows) - 1
                extra_w = inp.shape[1] - 2 * max(abs(c) for c in cols) - 1
                metrics["one_color"] += len(info["palette"]) == 1
                metrics["sparse"] += info["density"] < .25
                metrics["dense"] += info["density"] >= .7
                metrics["holes"] += _has_enclosed_hole(offsets)
                metrics["same_color_run"] += (
                    len(info["palette"]) >= 2 and any(
                        colors[p] == colors[q] for p in offsets
                        for q in _neighbors(p) if q in offsets))
                metrics["isolated_accent"] += (
                    len(info["palette"]) >= 2 and any(
                        list(colors.values()).count(color) == 1
                        for color in info["palette"]))
                metrics["same_color_opposite"] += any(
                    colors[p] == colors[(-p[0], -p[1])]
                    for p in offsets if (-p[0], -p[1]) in offsets)
                metrics["different_color_opposite"] += any(
                    colors[p] != colors[(-p[0], -p[1])]
                    for p in offsets if (-p[0], -p[1]) in offsets)
                metrics["pivot_inside"] += inside
                metrics["one_color_overlap"] += (
                    len(info["palette"]) == 1 and info["overlap"])
                metrics["disconnected_multidirection"] += (
                    len(info["components"]) >= 2 and inside)
                metrics["tight"] += extra_h <= 2 and extra_w <= 2
                metrics["roomy"] += extra_h >= 4 and extra_w >= 4
                metrics["tall"] += inp.shape[0] >= 1.5 * inp.shape[1]
                metrics["wide"] += inp.shape[1] >= 1.5 * inp.shape[0]
                if split == "test":
                    metrics["test_connected"] += len(info["components"]) == 1
                    metrics["test_disconnected"] += len(info["components"]) >= 2
                row_signs.update(0 if r == 0 else (1 if r > 0 else -1)
                                 for r, _ in info["offsets"])
                col_signs.update(0 if c == 0 else (1 if c > 0 else -1)
                                 for _, c in info["offsets"])
        assert len(puzzle_backgrounds) >= 2
        assert len(puzzle_palettes) >= 2
        assert test_colors - train_colors
        metrics["connected_train_disconnected_test"] += (
            1 in train_components and any(c >= 2 for c in test_components))
        metrics["disconnected_train_connected_test"] += (
            any(c >= 2 for c in train_components) and 1 in test_components)
    assert len(shapes) >= 35 and len(layouts) >= all_pairs // 2
    assert len(backgrounds) >= 8 and len(pivot_colors) >= 8
    assert 1 in palette_sizes and max(palette_sizes) >= 4
    assert 1 in component_counts and max(component_counts) >= 3
    assert filled == {False, True}
    assert row_signs == {-1, 0, 1} and col_signs == {-1, 0, 1}
    for name, minimum in (
        ("one_color", 12), ("sparse", 8), ("dense", 20),
        ("holes", 10), ("same_color_run", 20),
        ("isolated_accent", 10), ("same_color_opposite", 10),
        ("different_color_opposite", 10), ("pivot_inside", 15),
        ("one_color_overlap", 5), ("disconnected_multidirection", 5),
        ("tight", 10), ("roomy", 10), ("tall", 10), ("wide", 10),
        ("test_connected", 8), ("test_disconnected", 8),
        ("connected_train_disconnected_test", 4),
        ("disconnected_train_connected_test", 4)):
        assert metrics[name] >= minimum, (name, metrics[name])
