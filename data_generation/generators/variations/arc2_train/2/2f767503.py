import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _validate_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

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
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _neighbors4(
    cell: tuple[int, int],
    height: int,
    width: int,
):
    row, col = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components(
    cells: set[tuple[int, int]],
    height: int,
    width: int,
) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    result = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)

        component = {start}
        stack = [start]

        while stack:
            cell = stack.pop()
            for neighbor in _neighbors4(cell, height, width):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        result.append(component)

    return result


def _bar_midpoint(
    grid: np.ndarray,
    color: int,
) -> tuple[tuple[int, int], str] | None:
    coords = [
        tuple(map(int, coord))
        for coord in np.argwhere(grid == color)
    ]

    if len(coords) != 3:
        return None

    rows = sorted(row for row, _ in coords)
    cols = sorted(col for _, col in coords)

    if (
        len(set(rows)) == 1
        and cols[1] == cols[0] + 1
        and cols[2] == cols[1] + 1
    ):
        return (rows[0], cols[1]), "horizontal"

    if (
        len(set(cols)) == 1
        and rows[1] == rows[0] + 1
        and rows[2] == rows[1] + 1
    ):
        return (rows[1], cols[0]), "vertical"

    return None


def _parse_input(grid: np.ndarray) -> dict:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)

    assert len(values) == 4

    maximum_count = int(counts.max())
    assert int(np.count_nonzero(counts == maximum_count)) == 1

    background = int(values[int(np.argmax(counts))])
    non_background = [
        int(value)
        for value in values
        if int(value) != background
    ]

    bar_options = []
    for color in non_background:
        found = _bar_midpoint(grid, color)
        if found is not None:
            midpoint, axis = found
            bar_options.append((color, midpoint, axis))

    assert len(bar_options) == 1
    bar, midpoint, bar_axis = bar_options[0]

    remaining_colors = [
        color for color in non_background if color != bar
    ]
    marker_options = [
        color
        for color in remaining_colors
        if int(np.count_nonzero(grid == color)) == 1
    ]

    assert len(marker_options) == 1
    marker = marker_options[0]

    target_options = [
        color
        for color in remaining_colors
        if color != marker
    ]
    assert len(target_options) == 1

    target = target_options[0]
    assert int(np.count_nonzero(grid == target)) >= 4

    marker_coord = tuple(
        map(int, np.argwhere(grid == marker)[0])
    )

    dr = midpoint[0] - marker_coord[0]
    dc = midpoint[1] - marker_coord[1]

    assert abs(dr) + abs(dc) == 1

    if bar_axis == "horizontal":
        assert dc == 0
    else:
        assert dr == 0

    direction = dr, dc
    direction_name = {
        (-1, 0): "U",
        (1, 0): "D",
        (0, -1): "L",
        (0, 1): "R",
    }[direction]

    ray = []
    row = midpoint[0] + dr
    col = midpoint[1] + dc

    while 0 <= row < height and 0 <= col < width:
        ray.append((row, col))
        row += dr
        col += dc

    assert ray

    target_cells = {
        tuple(map(int, coord))
        for coord in np.argwhere(grid == target)
    }
    target_components = _components(
        target_cells, height, width
    )

    ray_set = set(ray)
    hit_components = [
        component
        for component in target_components
        if component & ray_set
    ]
    missed_components = [
        component
        for component in target_components
        if not component & ray_set
    ]

    assert hit_components
    assert missed_components

    return {
        "roles": (background, target, bar, marker),
        "midpoint": midpoint,
        "direction": direction,
        "direction_name": direction_name,
        "ray": ray_set,
        "hit": hit_components,
        "miss": missed_components,
    }


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_input(grid)
    output = grid.copy()
    background = parsed["roles"][0]

    for component in parsed["hit"]:
        for row, col in component:
            output[row, col] = background

    return output


def _grow_component(
    rng: np.random.Generator,
    core: set[tuple[int, int]],
    desired_size: int,
    height: int,
    width: int,
    blocked: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    style: str = "organic",
) -> set[tuple[int, int]] | None:
    component = set(core)
    first = min(core)
    line_axis = int(rng.integers(0, 2))
    if len(core) > 1:
        line_axis = 0 if len({p[0] for p in core}) == 1 else 1

    if component & blocked or component & forbidden:
        return None

    for cell in component:
        if any(
            neighbor in blocked
            for neighbor in _neighbors4(
                cell, height, width
            )
        ):
            return None

    while len(component) < desired_size:
        candidates = set()

        for cell in component:
            candidates.update(
                _neighbors4(cell, height, width)
            )

        candidates.difference_update(component)
        candidates.difference_update(blocked)
        candidates.difference_update(forbidden)

        candidates = [
            cell
            for cell in candidates
            if not any(
                neighbor in blocked
                for neighbor in _neighbors4(
                    cell, height, width
                )
            )
        ]

        if not candidates:
            return None

        candidates.sort()
        if style == "line":
            weights = [9 if q[line_axis] == first[line_axis] else 1 for q in candidates]
        elif style == "patch":
            weights = [1 + 5 * sum(n in component for n in _neighbors4(q, height, width)) for q in candidates]
        elif style == "chain":
            weights = [7 if sum(n in component for n in _neighbors4(q, height, width)) == 1 else 1 for q in candidates]
        else:
            weights = [1] * len(candidates)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        selected = candidates[int(rng.choice(len(candidates), p=weights))]
        component.add(selected)

    return component


# The first part of generator.py supplies the grid parser, four-connected
# components, the exact solver, and bounded component growth.
_SOURCE_INPUTS = (
    ("4477744744", "4777774747", "7447774777", "7777777777", "7777747777", "4747777777", "7774747777", "7555774777", "7797777777", "7777777774"),
    ("777777777777", "477477477777", "777777474777", "774774777744", "747777774777", "747777774474", "744777577447", "477747597774", "477444577744", "774747774777", "774777777777", "777777777477", "747777477777", "777777777777"),
    ("7477777777777", "4477577477777", "4479577474477", "7777577474774", "7777777477777", "7777774447777", "7477777777777", "7777777777747", "7777477777477", "7474777777777"),
    ("77747777777", "47777774777", "47777777777", "77757774774", "47957447744", "47754744777", "77774777777", "77774747777", "77777777474"),
)


def _predict_variant(grid, variant):
    info = _parse_input(grid)
    bg, target, _, _ = info["roles"]
    cells = {tuple(map(int, p)) for p in np.argwhere(grid == target)}
    if variant == "eight":
        remaining = set(cells)
        comps = []
        while remaining:
            start = remaining.pop()
            group, stack = {start}, [start]
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        q = (r + dr, c + dc)
                        if q in remaining:
                            remaining.remove(q)
                            group.add(q)
                            stack.append(q)
            comps.append(group)
    else:
        comps = _components(cells, *grid.shape)
    ray = set(info["ray"])
    if variant == "wide":
        dr, dc = info["direction"]
        for r, c in info["ray"]:
            for side in (-1, 1):
                q = (r + side * dc, c - side * dr)
                if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1]:
                    ray.add(q)
    hits = [comp for comp in comps if comp & ray]
    if variant == "first":
        mr, mc = info["midpoint"]
        dr, dc = info["direction"]
        hits = [min(hits, key=lambda comp: min((r - mr) * dr + (c - mc) * dc
                                              for r, c in comp & info["ray"]))]
    out = grid.copy()
    for comp in hits:
        for r, c in (comp & set(info["ray"])) if variant == "pixels" else comp:
            out[r, c] = bg
    return out


def _witness(grid, variant):
    return not np.array_equal(_solve_grid(grid), _predict_variant(grid, variant))


def _dimensions(rng):
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return int(rng.integers(7, 14)), int(rng.integers(7, 14))
    if mode == 1:
        return int(rng.integers(11, 23)), int(rng.integers(11, 23))
    if mode == 2:
        return int(rng.integers(18, 31)), int(rng.integers(18, 31))
    if mode == 3:
        return int(rng.integers(18, 31)), int(rng.integers(7, 13))
    return int(rng.integers(7, 13)), int(rng.integers(18, 31))


def _attempt_new_pair(rng, roles, direction, mandatory, test):
    h, w = _dimensions(rng)
    dr, dc = direction
    axis_len, cross_len = (h, w) if dr else (w, h)
    positions = []
    for m in range(axis_len):
        forward = axis_len - 1 - m if dr + dc > 0 else m
        backward = m if dr + dc > 0 else axis_len - 1 - m
        if forward >= (7 if mandatory else 3) and backward >= (2 if mandatory else 1):
            positions.append(m)
    if not positions or cross_len < 4:
        return None
    m = positions[int(rng.integers(len(positions)))]
    cross = int(rng.integers(1, cross_len - 1))
    mid = (m, cross) if dr else (cross, m)
    perpendicular = (dc, -dr)
    def at(t, p=0):
        return (mid[0] + t * dr + p * perpendicular[0], mid[1] + t * dc + p * perpendicular[1])
    def inside(q):
        return 0 <= q[0] < h and 0 <= q[1] < w
    length = axis_len - 1 - m if dr + dc > 0 else m
    back_length = m if dr + dc > 0 else axis_len - 1 - m
    ray = {at(t) for t in range(1, length + 1)}
    barcells = {at(0, p) for p in (-1, 0, 1)}
    marker = at(-1)
    forbidden = barcells | {marker}
    missed = []
    if mandatory:
        anchors = [at(2), at(length - 1)]
        diag = at(3, 1)
        if not inside(diag):
            diag = at(3, -1)
        if not inside(diag):
            return None
        missed.append({diag})
    else:
        maximum = min(4, (length + 1) // 3)
        requested = int(rng.integers(1, maximum + 1))
        stations = list(range(1, length + 1))
        rng.shuffle(stations)
        chosen = []
        for t in stations:
            if all(abs(t - old) >= 3 for old in chosen):
                chosen.append(t)
                if len(chosen) == requested:
                    break
        anchors = [at(t) for t in chosen]
    if (mandatory or rng.random() < .48) and back_length >= 2:
        dist = int(rng.integers(2, back_length + 1))
        back = at(-dist)
        other = set().union(*missed) if missed else set()
        if back not in other and not any(q in other for q in _neighbors4(back, h, w)):
            missed.append({back})
    hits = []
    reserved = set().union(*missed) if missed else set()
    for index, anchor in enumerate(anchors):
        core = {anchor}
        if mandatory and index == 1:
            options = [at(length - 1, p) for p in (-1, 1)]
            options = [q for q in options if inside(q) and q not in reserved and q not in forbidden]
            if not options:
                return None
            core.add(options[int(rng.integers(len(options)))])
        others = reserved | set(anchors[index + 1:])
        if hits:
            others |= set().union(*hits)
        desired = 1 if mandatory and index == 0 else int(rng.choice((1, 2, 3, 4, 5, 7, 9, 12, 16), p=(.12, .15, .17, .13, .12, .11, .09, .07, .04)))
        desired = max(desired, len(core))
        style = ("line", "patch", "chain", "organic")[int(rng.integers(0, 4))]
        grown = _grow_component(rng, core, desired, h, w, others, forbidden, style)
        if grown is None:
            grown = _grow_component(rng, core, len(core), h, w, others, forbidden, style)
        if grown is None:
            return None
        hits.append(grown)
    groups = hits + missed
    occupied = set().union(*groups)
    density = int(rng.integers(0, 3))
    number = (int(rng.integers(1, 4)), int(rng.integers(4, 9)), int(rng.integers(9, 18)))[density]
    if test:
        number += int(rng.integers(0, 4))
    for miss_index in range(number):
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        candidates = [q for q in cells if q not in forbidden and q not in ray and q not in occupied and not any(n in occupied for n in _neighbors4(q, h, w))]
        if not candidates:
            break
        if miss_index == 0 and rng.random() < .55:
            edges = [q for q in candidates if q[0] in (0, h - 1) or q[1] in (0, w - 1)]
            if edges:
                candidates = edges
        core = {candidates[0]}
        size = int(rng.choice((1, 2, 3, 4, 6, 8, 11, 15), p=(.25, .17, .14, .12, .12, .09, .07, .04)))
        if miss_index == 0 and density == 2:
            size = max(size, 6)
        style = ("line", "patch", "chain", "organic")[int(rng.integers(0, 4))]
        grown = _grow_component(rng, core, size, h, w, occupied, forbidden | ray, style)
        if grown is None:
            grown = core
        if len(occupied) + len(grown) >= h * w // 2:
            break
        groups.append(grown)
        occupied |= grown
    if len(occupied) < 4 or len(groups) <= len(hits):
        return None
    bg, target, bar, mark = roles
    grid = np.full((h, w), bg, dtype=np.int8)
    for r, c in occupied:
        grid[r, c] = target
    for r, c in barcells:
        grid[r, c] = bar
    grid[marker] = mark
    try:
        info = _parse_input(grid)
    except AssertionError:
        return None
    if len(info["hit"]) != len(hits):
        return None
    if mandatory and not all(_witness(grid, v) for v in ("eight", "wide", "pixels", "first")):
        return None
    return {"input": grid, "output": _solve_grid(grid)}


def _make_pair(rng, roles, direction, mandatory, test):
    for _ in range(200):
        pair = _attempt_new_pair(rng, roles, direction, mandatory, test)
        if pair is not None:
            return pair
    raise RuntimeError("bounded construction failed")


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_copy(grid):
    rows = tuple("".join(str(int(v)) for v in row) for row in grid)
    return rows in _SOURCE_INPUTS


def _features(grid):
    info = _parse_input(grid)
    hits, misses = info["hit"], info["miss"]
    hitcells, misscells = set().union(*hits), set().union(*misses)
    dr, dc = info["direction"]
    q = (info["midpoint"][0] - 2 * dr, info["midpoint"][1] - 2 * dc)
    back = set()
    while 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1]:
        back.add(q)
        q = (q[0] - dr, q[1] - dc)
    diagonal = any(q in misscells for r, c in hitcells for q in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)))
    edge = any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in hitcells | misscells)
    return {"roles": info["roles"], "direction": info["direction_name"], "hits": len(hits), "misses": len(misses),
            "hit_sizes": tuple(sorted(map(len, hits))), "miss_sizes": tuple(sorted(map(len, misses))),
            "diagonal": diagonal, "backward": bool(back & misscells), "edge": edge,
            **{v: _witness(grid, v) for v in ("eight", "wide", "pixels", "first")}}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30 and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _grid_key(inp) not in seen and not _source_copy(inp)
            seen.add(_grid_key(inp))
            assert np.array_equal(out, _solve_grid(inp)) and not np.array_equal(inp, out)
            feature = _features(inp)
            if split == "train":
                train_features.append(feature)
    for variant in ("eight", "wide", "pixels", "first"):
        assert any(f[variant] for f in train_features), variant
    assert any(f["backward"] for f in train_features)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed")
    rng = np.random.default_rng(int(seed))
    nt, ns = _resolve_counts(rng, num_train, num_test)
    base_roles = tuple(int(v) for v in rng.choice(10, size=4, replace=False))
    variable_palette = bool(rng.integers(0, 2))
    orientation_mode = int(rng.integers(0, 3))
    base_direction = _DIRS[int(rng.integers(0, 4))]
    train_dirs = [base_direction] + [base_direction if orientation_mode == 0 else _DIRS[int(rng.integers(0, 4))] for _ in range(nt - 1)]
    test_dirs = []
    for _ in range(ns):
        if orientation_mode == 0:
            test_dirs.append(base_direction)
        elif orientation_mode == 1:
            test_dirs.append(train_dirs[int(rng.integers(0, nt))])
        else:
            unseen = [d for d in _DIRS if d not in train_dirs]
            test_dirs.append(unseen[int(rng.integers(len(unseen)))] if unseen else _DIRS[int(rng.integers(0, 4))])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, directions in (("train", train_dirs), ("test", test_dirs)):
        for index, direction in enumerate(directions):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            roles = base_roles if not variable_palette or (split == "train" and index == 0) else tuple(int(v) for v in pair_rng.choice(10, size=4, replace=False))
            for _ in range(12):
                pair = _make_pair(pair_rng, roles, direction, split == "train" and index == 0, split == "test")
                if _grid_key(pair["input"]) not in seen and not _source_copy(pair["input"]):
                    break
            else:
                raise RuntimeError("duplicate construction")
            seen.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(p["input"]), _grid_key(p["output"])) for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_key(generate_puzzle(71, 5, 2)) == _puzzle_key(generate_puzzle(71, 5, 2))
    assert _puzzle_key(generate_puzzle(71, 5, 2)) != _puzzle_key(generate_puzzle(72, 5, 2))
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + index, nt, ns)
        assert (nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns)
    for nt, ns in ((1, 1), (11, 1), (2, 0), (6, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(101, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    features = [_features(pair["input"]) for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    shapes = {pair["input"].shape for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]}
    assert len(shapes) >= 50 and len({f["direction"] for f in features}) == 4
    assert {1, 2}.issubset({f["hits"] for f in features}) and max(f["hits"] for f in features) >= 3
    assert min(min(f["hit_sizes"]) for f in features) == 1 and max(max(f["miss_sizes"]) for f in features) >= 6
    assert any(f["diagonal"] for f in features) and any(not f["backward"] for f in features)
    assert any(f["edge"] for f in features)
    assert any(len({f["roles"] for split in ("train", "test") for f in [_features(pair["input"]) for pair in puzzle[split]]}) > 1 for puzzle in puzzles)
    rows = [(index, split, pair["input"], _features(pair["input"]), _parse_input(pair["input"]))
            for index, puzzle in enumerate(puzzles) for split in ("train", "test") for pair in puzzle[split]]
    assert all(len({grid.shape for i, _, grid, _, _ in rows if i == index}) > 1 for index in range(42))
    assert all(len({f["hits"] for i, _, _, f, _ in rows if i == index}) > 1 for index in range(42))
    assert all(len({f["misses"] for i, _, _, f, _ in rows if i == index}) > 1 for index in range(42))
    assert {f["hits"] for _, _, _, f, _ in rows} == {1, 2, 3, 4}
    assert min(f["misses"] for _, _, _, f, _ in rows) <= 2 and max(f["misses"] for _, _, _, f, _ in rows) >= 12
    assert all(any(f[v] for i, split, _, f, _ in rows if i == index and split == "train")
               for index in range(42) for v in ("eight", "wide", "pixels", "first"))
    assert any(f["diagonal"] for _, split, _, f, _ in rows if split == "train")
    assert any(f["diagonal"] for _, split, _, f, _ in rows if split == "test")
    assert any(f["hits"] >= 2 for _, split, _, f, _ in rows if split == "test")
    assert any(f["hits"] >= 3 for _, split, _, f, _ in rows if split == "train")
    assert any(f["hits"] >= 3 for _, split, _, f, _ in rows if split == "test")
    assert any(2 in f["hit_sizes"] for _, _, _, f, _ in rows)
    assert any(any(len(comp) == 2 and comp <= info["ray"] for comp in info["hit"]) for _, _, _, _, info in rows)
    assert any(max(f["miss_sizes"]) >= 6 and min(f["hit_sizes"]) <= 2 for _, _, _, f, _ in rows)
    def normalized(comp):
        mr = min(r for r, c in comp)
        mc = min(c for r, c in comp)
        return frozenset((r - mr, c - mc) for r, c in comp)
    assert any({normalized(c) for c in info["hit"]} & {normalized(c) for c in info["miss"]}
               for _, _, _, _, info in rows)
    assert any(f["misses"] >= 9 and f["hits"] >= 2 and f["diagonal"] for _, _, _, f, _ in rows)
    assert any(len(comp) >= 5 and (len({r for r, c in comp}) == 1 or len({c for r, c in comp}) == 1)
               for _, _, _, _, info in rows for comp in info["hit"] + info["miss"])
    assert any(any(sum(q in comp for q in _neighbors4(p, *grid.shape)) >= 3 for p in comp)
               for _, _, grid, _, info in rows for comp in info["hit"] + info["miss"])
    assert any(any({(r,c),(r+1,c),(r,c+1),(r+1,c+1)} <= comp for r,c in comp)
               for _, _, _, _, info in rows for comp in info["hit"] + info["miss"])
    assert any(len({f["roles"] for i, _, _, f, _ in rows if i == index}) == 1 for index in range(42))
    assert any(len({f["direction"] for i, split, _, f, _ in rows if i == index and split == "train"}) == 1 for index in range(42))
    assert any(len({f["direction"] for i, split, _, f, _ in rows if i == index and split == "train"}) > 1 for index in range(42))
    assert any(any(f["direction"] not in {g["direction"] for j, split, _, g, _ in rows if j == index and split == "train"}
                   for i, split, _, f, _ in rows if i == index and split == "test") for index in range(42))
    assert any(not f["backward"] for _, _, _, f, _ in rows)
    assert any(f["backward"] and not any((info["midpoint"][0]-2*info["direction"][0],
                                           info["midpoint"][1]-2*info["direction"][1]) in comp for comp in info["miss"])
               for _, _, _, f, info in rows)
    assert min(len(info["ray"]) for _, _, _, _, info in rows) <= 4
    assert any(30 in grid.shape for _, _, grid, _, _ in rows)
    assert any(grid.shape in {(10, 10), (14, 12), (10, 13), (9, 11)} for _, _, grid, _, _ in rows)
    assert any(max(grid.shape) <= 12 for _, _, grid, _, _ in rows)
    assert any(grid.shape[0] >= 2 * grid.shape[1] for _, _, grid, _, _ in rows)
    assert any(grid.shape[1] >= 2 * grid.shape[0] for _, _, grid, _, _ in rows)
