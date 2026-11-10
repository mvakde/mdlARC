import numpy as np


def _grid_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid_from_rows(
            (
                "0500000",
                "0550000",
                "0050055",
                "0550050",
                "0500550",
                "0500500",
                "0500500",
            )
        ),
        np.zeros((3, 1), dtype=np.int8),
    ),
    (
        _grid_from_rows(
            (
                "050",
                "055",
                "005",
            )
        ),
        np.zeros((2, 1), dtype=np.int8),
    ),
    (
        _grid_from_rows(
            (
                "050005005000",
                "050005005000",
                "055055055000",
                "005050050000",
                "005050550000",
                "555050500000",
                "000050500555",
                "000550500500",
                "055500500500",
            )
        ),
        np.zeros((5, 1), dtype=np.int8),
    ),
    (
        _grid_from_rows(
            (
                "000000055000",
                "550000005000",
                "055000555000",
                "005000500000",
                "005000555500",
                "055000000500",
                "050055500500",
                "055550500500",
                "000000500500",
                "550055500500",
                "050050005500",
                "050050005000",
            )
        ),
        np.zeros((4, 1), dtype=np.int8),
    ),
    (
        _grid_from_rows(
            (
                "0050005000",
                "0050005500",
                "0055000500",
                "0005000550",
                "0555000050",
                "0500005550",
                "0500005000",
                "0550055000",
                "0050050000",
            )
        ),
        np.zeros((3, 1), dtype=np.int8),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_test is None:
        largest_test_count = min(4, num_train - 1)
        if largest_test_count < 1:
            raise ValueError(
                "no legal test count exists for num_train"
            )
        num_test = int(
            rng.integers(1, largest_test_count + 1)
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _orthogonal_components(
    grid: np.ndarray,
    color: int,
) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for start_row in range(height):
        for start_col in range(width):
            if (
                visited[start_row, start_col]
                or int(grid[start_row, start_col]) != color
            ):
                continue

            visited[start_row, start_col] = True
            stack = [(start_row, start_col)]
            component: list[tuple[int, int]] = []

            while stack:
                row, col = stack.pop()
                component.append((row, col))

                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and int(grid[next_row, next_col]) == color
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            components.append(component)

    return components


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def _sample_palettes(rng: np.random.Generator) -> list[tuple[int, int]]:
    count = 2 if rng.random() < 0.38 else 1
    colors = [int(x) for x in rng.choice(10, size=2 * count, replace=False)]
    return [(colors[2 * i], colors[2 * i + 1]) for i in range(count)]


def _sample_size(rng: np.random.Generator, minimum_h: int,
                 minimum_w: int, regime: str) -> tuple[int, int]:
    # Size and aspect are sampled independently of the wall count once the
    # geometric minimum has been imposed.
    classes = ((3, 6), (7, 12), (13, 20), (21, 28))
    weights = ([.19, .47, .27, .07] if regime == "source_train"
               else [.14, .36, .37, .13])
    category = int(rng.choice(4, p=weights))
    low, high = classes[category]
    if high < max(minimum_h, minimum_w):
        high = min(30, max(minimum_h, minimum_w) + int(rng.integers(0, 5)))
    height = int(rng.integers(max(low, minimum_h), high + 1))
    width = int(rng.integers(max(low, minimum_w), high + 1))
    aspect = int(rng.integers(0, 5))
    if aspect == 0:
        width = min(30, max(minimum_w, width + int(rng.integers(2, 8))))
    elif aspect == 1:
        height = min(30, max(minimum_h, height + int(rng.integers(2, 8))))
    return height, width


def _allocate_lanes(rng: np.random.Generator, width: int, walls: int,
                    left_corner: bool, right_corner: bool) -> tuple[list[tuple[int, int]], int, int]:
    if walls == 0:
        return [], width, width
    baseline = [3 if left_corner else 1] + [1] * (walls - 1) + [3 if right_corner else 1]
    parts = [baseline[0]]
    for i in range(walls):
        parts.append(1)  # wall lane
        parts.append(baseline[i + 1])
    extra = width - sum(parts)
    if extra < 0:
        raise ValueError("too many walls for width")
    if extra:
        weights = rng.dirichlet(np.full(len(parts), .72))
        additions = rng.multinomial(extra, weights)
        parts = [part + int(add) for part, add in zip(parts, additions)]
    lanes = []
    cursor = parts[0]
    for i in range(walls):
        lane_width = parts[2 * i + 1]
        lanes.append((cursor, cursor + lane_width - 1))
        cursor += lane_width + parts[2 * i + 2]
    return lanes, parts[0], parts[-1]


def _draw_spanning_wall(rng: np.random.Generator, mask: np.ndarray,
                        left: int, right: int) -> None:
    height = mask.shape[0]
    current = int(rng.integers(left, right + 1))
    mask[0, current] = True
    for row in range(1, height):
        if right > left and rng.random() < .68:
            if rng.random() < .26:
                nxt = int(rng.integers(left, right + 1))
            else:
                nxt = min(right, max(left, current + int(rng.choice([-2, -1, 1, 2]))))
        else:
            nxt = current
        mask[row, min(current, nxt):max(current, nxt) + 1] = True
        current = nxt
        if right > left and rng.random() < .14:
            side = int(rng.choice([-1, 1]))
            protrusion = current + side
            if left <= protrusion <= right:
                mask[row, protrusion] = True


def _draw_corner(rng: np.random.Generator, mask: np.ndarray,
                 corner: str, horizontal_space: int) -> None:
    height, width = mask.shape
    # c=r=1 cuts off precisely one corner cell; larger values create tapered
    # and highly unequal regions, including wide adjacent-edge excursions.
    cmax = min(horizontal_space - 2, width - 2, 8)
    rmax = min(height - 2, 8)
    if cmax < 1 or rmax < 1:
        raise ValueError("no room for corner cut")
    if rng.random() < .30:
        c, depth = int(rng.integers(1, min(2, cmax) + 1)), int(rng.integers(1, min(2, rmax) + 1))
    else:
        c, depth = int(rng.integers(1, cmax + 1)), int(rng.integers(1, rmax + 1))
    # Distribute horizontal steps among the rows; the last row reaches the
    # side boundary. Each successive row overlaps the previous path.
    drops = rng.multinomial(c, rng.dirichlet(np.full(depth, 1.25)))
    local = np.zeros_like(mask)
    x = c
    local[0, x] = True
    for row in range(1, depth + 1):
        nxt = x - int(drops[row - 1])
        local[row, nxt:x + 1] = True
        x = nxt
    if rng.random() < .22:
        local[depth:, 0] = True
    if corner[0] == "b":
        local = local[::-1]
    if corner[1] == "r":
        local = local[:, ::-1]
    mask |= local


def _draw_network(rng: np.random.Generator, height: int,
                  width: int) -> np.ndarray:
    # A boundary-to-boundary wall with an attached rectangular loop. The
    # connected wall makes three background regions: two sides and a pocket.
    mask = np.zeros((height, width), dtype=bool)
    col = int(rng.integers(4, width - 4))
    mask[:, col] = True
    side = int(rng.choice([-1, 1]))
    ring_width = int(rng.integers(3, min(5, col - 1 if side < 0 else width - col - 2) + 1))
    ring_height = int(rng.integers(3, min(6, height - 2) + 1))
    top = int(rng.integers(1, height - ring_height))
    far_col = col + side * (ring_width - 1)
    low, high = sorted((col, far_col))
    bottom = top + ring_height - 1
    mask[top, low:high + 1] = True
    mask[bottom, low:high + 1] = True
    mask[top:bottom + 1, far_col] = True
    return mask


def _make_mask(rng: np.random.Generator, region_target: int,
               regime: str, mode: str) -> np.ndarray:
    if mode == "network":
        height, width = _sample_size(rng, 8, 10, regime)
        return _draw_network(rng, height, width)
    corners = 0 if mode == "full" else 1 if mode == "corner" else 2
    walls = region_target - 1 - corners
    if walls < 0:
        raise ValueError("invalid region target and mode")
    orientation = int(rng.integers(0, 2))
    if corners == 1:
        side_choices = [int(rng.integers(0, 2))]
    elif corners == 2:
        side_choices = [0, 1]
    else:
        side_choices = []
    left_corner = 0 in side_choices
    right_corner = 1 in side_choices
    if walls:
        min_width = 2 * walls + 1 + 2 * corners
    else:
        min_width = 3 if corners == 1 else 5
    # Work in vertical coordinates, then transpose. This preserves an
    # independent horizontal/vertical choice even on rectangular grids.
    height, width = _sample_size(rng, 3, min_width, regime)
    mask = np.zeros((height, width), dtype=bool)
    lanes, left_space, right_space = _allocate_lanes(
        rng, width, walls, left_corner, right_corner)
    for left, right in lanes:
        _draw_spanning_wall(rng, mask, left, right)
    for side in side_choices:
        corner = ("t" if rng.random() < .5 else "b") + ("l" if side == 0 else "r")
        if walls:
            space = left_space if side == 0 else right_space
        elif corners == 2:
            space = (width - 1) // 2 + 1
        else:
            space = width
        _draw_corner(rng, mask, corner, space)
    return mask.T.copy() if orientation else mask


def _mask_valid(mask: np.ndarray, target: int, network: bool) -> bool:
    h, w = mask.shape
    if not (3 <= h <= 30 and 3 <= w <= 30):
        return False
    if int(mask.sum()) >= h * w / 2:
        return False
    background = np.where(mask, 1, 0).astype(np.int8)
    regions = _orthogonal_components(background, 0)
    walls = _orthogonal_components(background, 1)
    if len(regions) != target or not (1 <= len(walls) <= 4):
        return False
    if network != (len(regions) > len(walls) + 1):
        return False
    for wall in walls:
        touches = [any(r == 0 for r, _ in wall), any(r == h - 1 for r, _ in wall),
                   any(c == 0 for _, c in wall), any(c == w - 1 for _, c in wall)]
        if sum(touches) < 2:
            return False
    return True


def _choose_mode(rng: np.random.Generator, count: int) -> str:
    choices = ["full"]
    weights = [.50]
    if count >= 2:
        choices.append("corner")
        weights.append(.38)
    if count >= 3:
        choices.append("double_corner")
        weights.append(.12)
    total = sum(weights)
    return str(rng.choice(choices, p=np.asarray(weights) / total))


def _generate_pair(rng: np.random.Generator, regime: str,
                   region_target: int, palette: tuple[int, int],
                   mode: str | None = None) -> dict:
    background, divider = palette
    for attempt in range(180):
        chosen = mode if mode is not None else _choose_mode(rng, region_target)
        mask = _make_mask(rng, region_target, regime, chosen)
        if not _mask_valid(mask, region_target, chosen == "network"):
            continue
        grid = np.where(mask, divider, background).astype(np.int8)
        output = np.full((len(_orthogonal_components(grid, background)), 1),
                         background, dtype=np.int8)
        if not _is_source_pair(grid, output):
            return {"input": grid, "output": output}
    raise ValueError(f"could not make a {region_target}-region {mode} pair")


def _pair_features(pair: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    background = int(output[0, 0])
    divider = next(int(x) for x in np.unique(grid) if int(x) != background)
    regions = _orthogonal_components(grid, background)
    walls = _orthogonal_components(grid, divider)
    h, w = grid.shape
    spans = []
    thicknesses = []
    boundary_types = []
    for component in walls:
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        row_span = max(rows) - min(rows) + 1
        col_span = max(cols) - min(cols) + 1
        if min(rows) == 0 and max(rows) == h - 1:
            spans.append(col_span)
            boundary_types.append("vertical")
            thicknesses.append(max(sum(r == y for r in rows) for y in set(rows)))
        elif min(cols) == 0 and max(cols) == w - 1:
            spans.append(row_span)
            boundary_types.append("horizontal")
            thicknesses.append(max(sum(c == x for c in cols) for x in set(cols)))
        else:
            spans.append(min(row_span, col_span))
            boundary_types.append("adjacent")
            thicknesses.append(max(max(sum(r == y for r in rows) for y in set(rows)),
                                   max(sum(c == x for c in cols) for x in set(cols))))
    adjacent = any(sum((any(r == 0 for r, _ in wall),
                        any(r == h - 1 for r, _ in wall),
                        any(c == 0 for _, c in wall),
                        any(c == w - 1 for _, c in wall))) >= 2
                   and not (any(r == 0 for r, _ in wall) and any(r == h - 1 for r, _ in wall))
                   and not (any(c == 0 for _, c in wall) and any(c == w - 1 for _, c in wall))
                   for wall in walls)
    three_sides = any(sum((any(r == 0 for r, _ in wall),
                           any(r == h - 1 for r, _ in wall),
                           any(c == 0 for _, c in wall),
                           any(c == w - 1 for _, c in wall))) >= 3
                      for wall in walls)
    return {"shape": (h, w), "regions": len(regions), "walls": len(walls),
            "palette": (background, divider), "min_area": min(map(len, regions)),
            "max_area": max(map(len, regions)), "max_span": max(spans),
            "max_thickness": max(thicknesses), "boundary_types": boundary_types,
            "adjacent": adjacent, "three_sides": three_sides,
            "network": len(regions) > len(walls) + 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "illegal pair counts")
    seen = set()
    seen_pairs = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                _require(isinstance(array, np.ndarray) and array.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= n <= 30 for n in array.shape)
                         and np.issubdtype(array.dtype, np.integer)
                         and np.all((0 <= array) & (array <= 9)), "illegal grid")
            _require(_grid_key(grid) not in seen, "duplicate input")
            _require(_pair_key(grid, output) not in seen_pairs, "duplicate pair")
            seen.add(_grid_key(grid))
            seen_pairs.add(_pair_key(grid, output))
            _require(not _is_source_pair(grid, output), "copied source pair")
            _require(not np.array_equal(grid, output), "unchanged pair")
            _require(output.shape[1] == 1 and np.unique(output).size == 1,
                     "output must be a uniform column")
            background = int(output[0, 0])
            colors = {int(x) for x in np.unique(grid)}
            _require(len(colors) == 2 and background in colors,
                     "input requires background and divider colors")
            divider = next(iter(colors - {background}))
            _require(int(np.count_nonzero(grid == background)) >
                     int(np.count_nonzero(grid == divider)), "background is not dominant")
            regions = _orthogonal_components(grid, background)
            walls = _orthogonal_components(grid, divider)
            _require(2 <= len(regions) <= 5 and 1 <= len(walls) <= 4,
                     "unsupported region or divider count")
            _require(output.shape == (len(regions), 1)
                     and np.all(output == background), "wrong region-count output")
            h, w = grid.shape
            for wall in walls:
                contacts = sum((any(r == 0 for r, _ in wall),
                                any(r == h - 1 for r, _ in wall),
                                any(c == 0 for _, c in wall),
                                any(c == w - 1 for _, c in wall)))
                _require(contacts >= 2, "divider does not reach two boundaries")
            _require(len(regions) >= len(walls) + 1 and
                     len(regions) <= len(walls) + 2,
                     "ambiguous barrier topology")
            records[split].append(_pair_features(pair))
    train = records["train"]
    tests = records["test"]
    _require(len({r["regions"] for r in train}) >= 2,
             "training needs multiple output lengths")
    _require({r["palette"] for r in tests} <= {r["palette"] for r in train},
             "test color roles were not demonstrated")
    if any(r["network"] for r in tests):
        _require(any(r["network"] for r in train),
                 "connected-wall pocket was not demonstrated in training")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    nt, ns = _resolve_counts(rng, num_train, num_test)
    palettes = _sample_palettes(rng)
    use_network = rng.random() < .19
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=nt + ns, dtype=np.int64)
    all_inputs = set()

    def make(split: str, index: int, target: int, mode: str | None,
             palette: tuple[int, int], child_seed: np.integer) -> dict:
        pair_rng = np.random.default_rng(int(child_seed))
        for _ in range(30):
            pair = _generate_pair(pair_rng, split, target, palette, mode)
            key = _grid_key(pair["input"])
            if key not in all_inputs:
                all_inputs.add(key)
                return pair
        raise ValueError("could not make unique pair")

    train_targets = [2, int(rng.integers(3, 6))]
    if use_network:
        train_targets[1] = 3
    train_targets += [int(rng.integers(2, 6)) for _ in range(nt - 2)]
    train = []
    for i, target in enumerate(train_targets):
        mode = "network" if use_network and i == 1 else None
        palette = palettes[i] if i < len(palettes) else palettes[int(rng.integers(len(palettes)))]
        train.append(make("source_train", i, target, mode, palette, child_seeds[i]))
    test_targets = [int(rng.integers(2, 6)) for _ in range(ns)]
    if use_network:
        test_targets[0] = 3
    if ns > 1 and test_targets[1] == test_targets[0]:
        test_targets[1] = 2 + (test_targets[0] - 1) % 4
    test = []
    for i, target in enumerate(test_targets):
        mode = "network" if use_network and i == 0 else None
        palette = palettes[int(rng.integers(len(palettes)))]
        test.append(make("target_test", i, target, mode, palette, child_seeds[nt + i]))
    train = [train[int(i)] for i in rng.permutation(nt)]
    test = [test[int(i)] for i in rng.permutation(ns)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _union_find_region_count(grid: np.ndarray, background: int) -> int:
    h, w = grid.shape
    parent = list(range(h * w))
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != background:
                continue
            i = r * w + c
            for nr, nc in ((r - 1, c), (r, c - 1)):
                if nr >= 0 and nc >= 0 and int(grid[nr, nc]) == background:
                    parent[root(i)] = root(nr * w + nc)
    return len({root(r * w + c) for r in range(h) for c in range(w)
                if int(grid[r, c]) == background})


def self_test() -> None:
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(8000 + (train_count or 0) * 10 + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    first = generate_puzzle(314159, 6, 2)
    second = generate_puzzle(314159, 6, 2)
    _require(all(np.array_equal(a[key], b[key]) for split in ("train", "test")
                 for a, b in zip(first[split], second[split])
                 for key in ("input", "output")), "not deterministic")
    _require(len({_pair_key(p["train"][0]["input"], p["train"][0]["output"])
                  for p in (generate_puzzle(300 + i) for i in range(8))}) >= 7,
             "different seeds did not vary")
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    regions, test_regions, adjacent, tiny, wide, networks = set(), set(), 0, 0, 0, 0
    three_sided = 0
    small = near_square = wide_grid = tall_grid = thick = unequal = 0
    two_palette_puzzles = mixed_boundary_puzzles = 0
    boundary_types = set()
    mixed_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(10000 + seed)
        validate_generated_puzzle(puzzle)
        puzzle_palettes = set()
        puzzle_boundaries = set()
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        if len({p["output"].shape[0] for p in puzzle["test"]}) > 1:
            mixed_tests += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                bg = int(output[0, 0])
                _require(output.shape == (_union_find_region_count(grid, bg), 1),
                         "independent count disagrees")
                f = _pair_features(pair)
                puzzle_palettes.add(f["palette"])
                puzzle_boundaries.update(f["boundary_types"])
                boundary_types.update(f["boundary_types"])
                shapes.add(f["shape"])
                palettes.add(f["palette"])
                layouts.add((grid.shape, (grid != bg).tobytes()))
                regions.add(f["regions"])
                if split == "test":
                    test_regions.add(f["regions"])
                adjacent += f["adjacent"]
                three_sided += f["three_sides"]
                tiny += f["min_area"] <= 2
                wide += f["max_span"] >= 5
                networks += f["network"]
                small += max(f["shape"]) <= 6
                near_square += abs(f["shape"][0] - f["shape"][1]) <= 2
                wide_grid += f["shape"][1] >= f["shape"][0] + 4
                tall_grid += f["shape"][0] >= f["shape"][1] + 4
                thick += f["max_thickness"] >= 3
                unequal += f["max_area"] >= 5 * f["min_area"]
        two_palette_puzzles += len(puzzle_palettes) == 2
        mixed_boundary_puzzles += len(puzzle_boundaries) >= 2
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "pair counts collapsed")
    _require(regions == {2, 3, 4, 5} and test_regions == {2, 3, 4, 5},
             "region counts collapsed")
    _require(len(shapes) >= 40 and len(layouts) >= 150 and len(palettes) >= 15,
             "structural or palette diversity collapsed")
    _require(adjacent >= 10 and three_sided >= 2 and tiny >= 3
             and wide >= 10 and networks >= 2,
             "reviewed geometric features absent")
    _require(small >= 5 and near_square >= 20 and wide_grid >= 10 and tall_grid >= 10,
             "size and aspect diversity collapsed")
    _require(thick >= 10 and unequal >= 10,
             "thickness or region-area diversity collapsed")
    _require(boundary_types == {"vertical", "horizontal", "adjacent"}
             and mixed_boundary_puzzles >= 10 and two_palette_puzzles >= 3,
             "boundary or within-puzzle palette diversity collapsed")
    _require(mixed_tests >= 3, "multiple tests do not vary in region count")
