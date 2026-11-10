import numpy as np


_SIDE_TOP = 0
_SIDE_BOTTOM = 1
_SIDE_LEFT = 2
_SIDE_RIGHT = 3
_ALL_SIDES = (_SIDE_TOP, _SIDE_BOTTOM, _SIDE_LEFT, _SIDE_RIGHT)


def _require(condition: bool, message: str = "validation failed") -> None:
    if not condition:
        raise AssertionError(message)


def _array_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _array_from_rows((
            "88888888888",
            "86668888888",
            "86668855588",
            "96668855598",
            "88888855588",
            "88888888888",
            "88898888888",
            "88866688888",
            "88866688888",
            "88866688888",
            "88889888888",
        )),
        _array_from_rows((
            "88888888888",
            "86668888888",
            "86668888888",
            "86968888998",
            "88888888888",
            "88888888888",
            "88888888888",
            "88866688888",
            "88899688888",
            "88866688888",
            "88888888888",
        )),
    ),
    (
        _array_from_rows((
            "888888888",
            "888888888",
            "888888888",
            "888888888",
            "886668888",
            "886668888",
            "886669888",
            "888888888",
            "888888888",
        )),
        _array_from_rows((
            "888888888",
            "888888888",
            "888888888",
            "888888888",
            "886668888",
            "886668888",
            "886968888",
            "888888888",
            "888888888",
        )),
    ),
    (
        _array_from_rows((
            "888888888888888",
            "888888888888888",
            "885559888888888",
            "885558888899888",
            "885559888866688",
            "888888888866688",
            "888888888866688",
            "888866688888888",
            "888966688888888",
            "888866688988888",
            "888898885558888",
            "888888885558888",
            "888855585558888",
            "888855588898888",
            "888955588888888",
        )),
        _array_from_rows((
            "888888888888888",
            "888888888888888",
            "888899888888888",
            "888888888888888",
            "888899888866688",
            "888888888899688",
            "888888888866688",
            "888866688888888",
            "888899688888888",
            "888866688988888",
            "888888888988888",
            "888888888888888",
            "888888888898888",
            "888888888898888",
            "888998888888888",
        )),
    ),
    (
        _array_from_rows((
            "8888855588888",
            "8888855588888",
            "6668955598888",
            "6668889888888",
            "6668888888888",
            "9998666989898",
            "8888666886669",
            "8889666896668",
            "8888898886669",
            "8555888988888",
            "9555866688555",
            "9555866689555",
            "8889866688555",
        )),
        _array_from_rows((
            "8888888888888",
            "8888888888888",
            "6668999998888",
            "9998889888888",
            "6668888888888",
            "8888696888888",
            "8888696886968",
            "8888696889998",
            "8888888886968",
            "8888888888888",
            "9988866688888",
            "9989866989988",
            "8889866688888",
        )),
    ),
)


def _validate_count(
    name: str,
    value: int | None,
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
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count("num_train", num_train, 2, 10)
    test_count = _validate_count("num_test", num_test, 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif train_count is None:
        lower = max(2, int(test_count) + 1)
        train_count = int(rng.integers(lower, 11))
    else:
        upper = min(4, int(train_count) - 1)
        if upper < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = int(rng.integers(1, upper + 1))

    if not (
        2 <= int(train_count) <= 10
        and 1 <= int(test_count) <= 4
        and int(train_count) > int(test_count)
    ):
        raise ValueError("could not select a legal pair-count combination")

    return int(train_count), int(test_count)


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    style = int(rng.integers(0, 5))
    return {
        "retaining": colors[0],
        "erasing": colors[1],
        "backgrounds": colors[2:4] if style in (1, 3, 4) else colors[2:3],
        "markers": colors[4:7] if style == 4 else (colors[4:6] if style in (2, 3) else colors[4:5]),
        "style": style,
    }


def _marker_geometry(row: int, col: int, side: int, offset: int) -> tuple[int, int, int, int]:
    if side == _SIDE_TOP:
        return row - 1, col + offset, 1, 0
    if side == _SIDE_BOTTOM:
        return row + 3, col + offset, -1, 0
    if side == _SIDE_LEFT:
        return row + offset, col - 1, 0, 1
    return row + offset, col + 3, 0, -1


def _separate(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    # No body overlap or four-connected contact, even when materials match.
    return not (abs(ar - br) <= 3 and abs(ac - bc) <= 2) and not (
        abs(ar - br) <= 2 and abs(ac - bc) <= 3
    )


def _available_markers(
    height: int, width: int, positions: list[tuple[int, int]],
) -> list[list[tuple[int, int, int, int, int, int]]]:
    result = []
    for index, (row, col) in enumerate(positions):
        options = []
        for side in _ALL_SIDES:
            for offset in range(3):
                mr, mc, dr, dc = _marker_geometry(row, col, side, offset)
                if not (0 <= mr < height and 0 <= mc < width):
                    continue
                if any(
                    other != index and _touches_square(mr, mc, other_row, other_col)
                    for other, (other_row, other_col) in enumerate(positions)
                ):
                    continue
                options.append((side, offset, mr, mc, dr, dc))
        result.append(options)
    return result


def _touches_square(mr: int, mc: int, row: int, col: int) -> bool:
    return (
        (mr in (row - 1, row + 3) and col <= mc <= col + 2)
        or (mc in (col - 1, col + 3) and row <= mr <= row + 2)
    )


def _layout(
    rng: np.random.Generator, count: int, crowded: bool,
) -> tuple[int, int, list[tuple[int, int]], list[list[tuple[int, int, int, int, int, int]]]]:
    for _ in range(80):
        base = max(8, int(np.ceil(np.sqrt(count))) * (4 if crowded else 5))
        height = int(np.clip(base + int(rng.integers(-1, 7)), 8, 28))
        width = int(np.clip(base + int(rng.integers(-1, 7)), 8, 28))
        if int(rng.integers(0, 3)) == 0:
            height = int(np.clip(height + int(rng.integers(1, 7)), 8, 30))
        if int(rng.integers(0, 3)) == 0:
            width = int(np.clip(width + int(rng.integers(1, 7)), 8, 30))
        candidates = [(r, c) for r in range(height - 2) for c in range(width - 2)]
        # One randomized priority list creates irregular origins, with an independent
        # preference for canvas edges in some scenes.
        edge_bias = int(rng.integers(0, 4)) == 0
        priorities = {}
        for p in candidates:
            edge = p[0] in (0, height - 3) or p[1] in (0, width - 3)
            corner = p[0] in (0, height - 3) and p[1] in (0, width - 3)
            priorities[p] = float(rng.random()) + (0.55 if edge_bias and edge else 0) + (0.2 if edge_bias and corner else 0)
        candidates.sort(key=lambda p: priorities[p], reverse=True)
        positions = []
        for p in candidates:
            if all(_separate(p, q) for q in positions):
                positions.append(p)
                if len(positions) == count:
                    break
        if len(positions) != count:
            continue
        options = _available_markers(height, width, positions)
        if all(options):
            return height, width, positions, options
    raise ValueError("unable to place squares with visible adjacent markers")


def _choose_markers(
    rng: np.random.Generator,
    options: list[tuple[int, int, int, int, int, int]],
    anchor: bool,
    regime: str,
    force_sparse: bool,
    allow_complete: bool,
) -> list[tuple[int, int, int, int, int, int]]:
    if force_sparse:
        desired = 1
    elif anchor:
        desired = int(rng.integers(3, 7))
    else:
        desired = int(rng.integers(1, 8 if regime == "source_train" else 9))
    desired = min(desired, len(options), 12 if allow_complete else 8)
    selected = []
    complete_chance = 0.23 if regime == "source_train" else 0.32
    if allow_complete and not force_sparse and float(rng.random()) < complete_chance:
        sides = [s for s in _ALL_SIDES if sum(x[0] == s for x in options) == 3]
        if sides:
            side = sides[int(rng.integers(len(sides)))]
            selected = [x for x in options if x[0] == side]
            desired = max(desired, 3)
    if not selected and not anchor and float(rng.random()) < 0.22:
        side = options[int(rng.integers(len(options)))][0]
        side_options = [x for x in options if x[0] == side]
        desired = min(desired, len(side_options))
        options = side_options
    for i in rng.permutation(len(options)):
        option = options[int(i)]
        if not allow_complete and (
            {x[1] for x in selected if x[0] == option[0]} | {option[1]}
        ) == {0, 1, 2}:
            continue
        if option not in selected:
            selected.append(option)
        if len(selected) == desired:
            break
    return selected


def _pair_colors(parameters: dict, split: str, index: int, rng: np.random.Generator) -> tuple[int, list[int]]:
    backgrounds = parameters["backgrounds"]
    markers = parameters["markers"]
    if split == "train":
        background = backgrounds[index % len(backgrounds)]
        if parameters["style"] == 4:
            marker_colors = markers
        else:
            marker_colors = [markers[index % len(markers)]]
    else:
        background = backgrounds[int(rng.integers(len(backgrounds)))]
        if parameters["style"] == 4:
            marker_colors = markers
        else:
            marker_colors = [markers[int(rng.integers(len(markers)))]]
    return background, marker_colors


def _generate_pair(
    rng: np.random.Generator, regime: str, parameters: dict,
    split: str, index: int, anchor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train":
        count = int(rng.integers(2 if anchor else 1, 6))
    else:
        count = int(rng.integers(6, 9)) if float(rng.random()) < 0.38 else int(rng.integers(2, 6))
    crowded = (float(rng.random()) < (0.63 if regime == "target_test" else 0.4))
    background, marker_colors = _pair_colors(parameters, split, index, rng)
    for _ in range(30):
        height, width, positions, options = _layout(rng, count, crowded)
        if anchor:
            kinds = ["retaining", "erasing"] + ["retaining" if int(rng.integers(2)) == 0 else "erasing" for _ in range(count - 2)]
            rng.shuffle(kinds)
        else:
            kinds = ["retaining" if int(rng.integers(2)) == 0 else "erasing" for _ in range(count)]
        grid = np.full((height, width), background, dtype=np.int8)
        for (row, col), kind in zip(positions, kinds):
            grid[row:row + 3, col:col + 3] = parameters[kind]
        selected = []
        sparse_index = int(rng.integers(count)) if regime == "target_test" and float(rng.random()) < 0.55 else -1
        allow_complete = float(rng.random()) < (0.44 if regime == "target_test" else 0.35)
        for j, opts in enumerate(options):
            selected.append(_choose_markers(rng, opts, anchor, regime, j == sparse_index, allow_complete))
        # Give every destination one color. Opposing projections can overlap,
        # while no different colors compete for a single output cell.
        color_by_target = {}
        all_markers = []
        for kind, markers in zip(kinds, selected):
            for side, offset, mr, mc, dr, dc in markers:
                distance = 2 if kind == "retaining" else 1
                target = (mr + distance * dr, mc + distance * dc)
                if target not in color_by_target:
                    color_by_target[target] = marker_colors[int(rng.integers(len(marker_colors)))]
                all_markers.append((mr, mc, target, color_by_target[target]))
        if len(marker_colors) > 1 and len(color_by_target) > 1:
            targets = list(color_by_target)
            rng.shuffle(targets)
            for k, target in enumerate(targets[:len(marker_colors)]):
                color_by_target[target] = marker_colors[k]
            all_markers = [(mr, mc, target, color_by_target[target]) for mr, mc, target, _ in all_markers]
        for mr, mc, _, color in all_markers:
            grid[mr, mc] = color
        # Keep the local background recognizable, including crowded cases.
        values, counts = np.unique(grid, return_counts=True)
        if int(values[int(np.argmax(counts))]) != background or list(counts).count(max(counts)) > 1:
            continue
        output = _transform(grid, background, parameters["retaining"], parameters["erasing"])[0]
        if not np.array_equal(grid, output):
            return grid, output
    raise ValueError("unable to build a nondegenerate pair")


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    points = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    result = []
    while points:
        start = points.pop()
        component = [start]
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in points:
                    points.remove(p)
                    component.append(p)
                    stack.append(p)
        result.append(component)
    return result


def _parse_input(grid: np.ndarray, background: int, retaining: int, erasing: int) -> list[dict]:
    _require(len({background, retaining, erasing}) == 3)
    squares = []
    for kind, color in (("retaining", retaining), ("erasing", erasing)):
        for component in _components_of_color(grid, color):
            row = min(p[0] for p in component)
            col = min(p[1] for p in component)
            _require(len(component) == 9 and np.all(grid[row:row + 3, col:col + 3] == color), "material is not a solid 3x3 square")
            squares.append({"kind": kind, "row": row, "column": col, "markers": []})
    _require(bool(squares), "no material squares")
    material_cells = sum(([(r, c) for r in range(s["row"], s["row"] + 3) for c in range(s["column"], s["column"] + 3)] for s in squares), [])
    _require(len(set(material_cells)) == len(material_cells), "material squares overlap")
    for raw_r, raw_c in np.argwhere((grid != background) & (grid != retaining) & (grid != erasing)):
        mr, mc = int(raw_r), int(raw_c)
        associations = []
        for s in squares:
            row, col = s["row"], s["column"]
            for side in _ALL_SIDES:
                for offset in range(3):
                    r, c, dr, dc = _marker_geometry(row, col, side, offset)
                    if (mr, mc) == (r, c):
                        associations.append((s, side, offset, dr, dc))
        _require(len(associations) == 1, "marker lacks a unique square and inward direction")
        square, side, offset, dr, dc = associations[0]
        square["markers"].append({"row": mr, "column": mc, "side": side, "offset": offset, "dr": dr, "dc": dc, "color": int(grid[mr, mc])})
    _require(all(s["markers"] for s in squares), "square has no marker")
    _require(any(int(x) == background for x in grid.flat), "background absent")
    return squares


def _transform(grid: np.ndarray, background: int, retaining: int, erasing: int) -> tuple[np.ndarray, list[dict]]:
    squares = _parse_input(grid, background, retaining, erasing)
    output = grid.copy()
    targets = {}
    for square in squares:
        if square["kind"] == "erasing":
            r, c = square["row"], square["column"]
            output[r:r + 3, c:c + 3] = background
    for square in squares:
        for marker in square["markers"]:
            r, c, dr, dc, color = (marker[k] for k in ("row", "column", "dr", "dc", "color"))
            if square["kind"] == "retaining":
                output[r, c] = background
                target = (r + 2 * dr, c + 2 * dc)
            else:
                target = (r + dr, c + dc)
            _require(target not in targets or targets[target] == color, "different marker colors collide")
            targets[target] = color
    for (r, c), color in targets.items():
        output[r, c] = color
    return output, squares


def _check_local_rule(inp: np.ndarray, out: np.ndarray, background: int, retaining: int, erasing: int) -> None:
    """Check changed and unchanged cells without calling the output builder."""
    squares = _parse_input(inp, background, retaining, erasing)
    erased = set()
    exterior_clears = set()
    projected = {}
    for square in squares:
        if square["kind"] == "erasing":
            erased.update((r, c) for r in range(square["row"], square["row"] + 3) for c in range(square["column"], square["column"] + 3))
        for marker in square["markers"]:
            r, c = marker["row"], marker["column"]
            if square["kind"] == "retaining":
                exterior_clears.add((r, c))
                distance = 2
            else:
                distance = 1
            point = (r + distance * marker["dr"], c + distance * marker["dc"])
            _require(point not in projected or projected[point] == marker["color"], "conflicting marker projections")
            projected[point] = marker["color"]
    _require(inp.shape == out.shape, "input/output dimensions differ")
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            point = (r, c)
            if point in projected:
                expected = projected[point]
            elif point in erased or point in exterior_clears:
                expected = background
            else:
                expected = int(inp[r, c])
            _require(int(out[r, c]) == expected, "local material rule failed")


def _dominant(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    _require(list(counts).count(max(counts)) == 1, "background count is tied")
    return int(colors[int(np.argmax(counts))])


def _infer_materials(train: list[dict]) -> tuple[int, int]:
    backgrounds = [_dominant(p["input"]) for p in train]
    first = train[0]["input"]
    candidates = []
    for color in np.unique(first):
        value = int(color)
        if value != backgrounds[0] and any(
            len(component) == 9 for component in _components_of_color(first, value)
        ):
            candidates.append(value)
    solutions = []
    for retaining in candidates:
        for erasing in candidates:
            if retaining == erasing:
                continue
            try:
                infos = []
                for pair, background in zip(train, backgrounds):
                    expected, squares = _transform(pair["input"], background, retaining, erasing)
                    _require(np.array_equal(expected, pair["output"]))
                    infos.append(squares)
                _require({s["kind"] for info in infos for s in info} == {"retaining", "erasing"})
            except AssertionError:
                continue
            solutions.append((retaining, erasing))
    _require(len(solutions) == 1, "training does not identify both material responses uniquely")
    return solutions[0]


def _has_complete_edge(square: dict) -> bool:
    return any({m["offset"] for m in square["markers"] if m["side"] == side} == {0, 1, 2} for side in _ALL_SIDES)


def _check_grid(grid: np.ndarray) -> None:
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
    _require(all(1 <= x <= 30 for x in grid.shape), "grid outside ARC dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "grid dtype is not integer")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9, "grid has invalid colors")


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _is_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    return any(np.array_equal(inp, a) and np.array_equal(out, b) for a, b in _SOURCE_PAIRS)


def _analyze_pair(pair: dict, backgrounds: set[int], retaining: int, erasing: int, regime: str) -> tuple[list[dict], int]:
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input/output shapes differ")
    background = _dominant(inp)
    _require(background in backgrounds, "test background unseen in training")
    expected, squares = _transform(inp, background, retaining, erasing)
    _require(np.array_equal(expected, out), "output violates inward material rule")
    _require(not np.array_equal(inp, out), "degenerate unchanged pair")
    count = len(squares)
    _require(1 <= count <= 5 if regime == "source_train" else 2 <= count <= 8, "square count outside regime")
    return squares, background


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts invalid")
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys invalid")
        _check_grid(pair["input"])
        _check_grid(pair["output"])
        key = _grid_key(pair["input"])
        _require(key not in seen, "duplicate input")
        seen.add(key)
        _require(not _is_source_pair(pair["input"], pair["output"]), "source pair copied")
    retaining, erasing = _infer_materials(puzzle["train"])
    train_backgrounds = {_dominant(p["input"]) for p in puzzle["train"]}
    train_markers = set()
    kinds, sides, offsets = set(), set(), set()
    for pair in puzzle["train"]:
        squares, background = _analyze_pair(pair, train_backgrounds, retaining, erasing, "source_train")
        for square in squares:
            kinds.add(square["kind"])
            for marker in square["markers"]:
                sides.add(marker["side"])
                offsets.add(marker["offset"])
                train_markers.add(marker["color"])
        _require(background not in (retaining, erasing), "background is a material color")
    _require(kinds == {"retaining", "erasing"}, "both responses absent from training")
    _require(sides == set(_ALL_SIDES) and offsets == {0, 1, 2}, "training lacks directions or edge offsets")
    _require(not train_markers & (train_backgrounds | {retaining, erasing}), "semantic color roles overlap")
    for pair in puzzle["test"]:
        squares, background = _analyze_pair(pair, train_backgrounds, retaining, erasing, "target_test")
        _require(all(m["color"] in train_markers for s in squares for m in s["markers"]), "test marker color unseen in training")
        _require(background not in (retaining, erasing), "test background is material color")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    for _ in range(40):
        puzzle = {"train": [], "test": []}
        try:
            for split, n in (("train", train_count), ("test", test_count)):
                regime = "source_train" if split == "train" else "target_test"
                for index in range(n):
                    pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                    inp, out = _generate_pair(pair_rng, regime, parameters, split, index, split == "train" and index < 2)
                    puzzle[split].append({"input": inp, "output": out})
            validate_generated_puzzle(puzzle)
            return puzzle
        except (AssertionError, ValueError):
            continue
    raise ValueError("unable to construct a coherent puzzle")


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for seed, nt, ns in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        _require((nt is None or len(p["train"]) == nt) and (ns is None or len(p["test"]) == ns))
        _require(_puzzles_equal(p, generate_puzzle(seed, nt, ns)), "nondeterministic puzzle")
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(100, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    train_counts, test_counts, shapes, layouts, material_counts, marker_counts = set(), set(), set(), set(), set(), set()
    review_counts = {
        "edge_flush": 0, "corner_flush": 0, "staggered": 0,
        "sparse_train": 0, "sparse_test": 0, "one_side": 0,
        "complete_train": 0, "complete_test": 0, "no_complete_test": 0,
        "overlap": 0, "several_sides": 0, "mixed_markers": 0,
        "one_material_test": 0, "low_count_test": 0,
        "crowded_test": 0, "background_varying_puzzle": 0,
        "marker_varying_puzzle": 0, "different_shapes_puzzle": 0,
    }
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        ret, era = _infer_materials(p["train"])
        puzzle_backgrounds, puzzle_markers, puzzle_shapes = set(), set(), set()
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                bg = _dominant(inp)
                puzzle_backgrounds.add(bg)
                puzzle_shapes.add(inp.shape)
                _check_local_rule(inp.copy(), pair["output"].copy(), bg, ret, era)
                squares = _parse_input(inp, bg, ret, era)
                shapes.add(inp.shape)
                material_counts.add(len(squares))
                all_markers = [m for s in squares for m in s["markers"]]
                puzzle_markers.update(m["color"] for m in all_markers)
                marker_counts.add(len(all_markers))
                layouts.add((inp.shape, tuple(sorted((s["row"], s["column"], s["kind"], tuple(sorted((m["side"], m["offset"]) for m in s["markers"]))) for s in squares))))
                review_counts["edge_flush"] += any(s["row"] in (0, inp.shape[0] - 3) or s["column"] in (0, inp.shape[1] - 3) for s in squares)
                review_counts["corner_flush"] += any(s["row"] in (0, inp.shape[0] - 3) and s["column"] in (0, inp.shape[1] - 3) for s in squares)
                review_counts["staggered"] += len({s["row"] for s in squares}) >= 3 and len({s["column"] for s in squares}) >= 3
                review_counts["one_side"] += any(len({m["side"] for m in s["markers"]}) == 1 for s in squares)
                review_counts["several_sides"] += any(len({m["side"] for m in s["markers"]}) >= 3 for s in squares)
                review_counts["mixed_markers"] += len({m["color"] for m in all_markers}) > 1
                targets = [(m["row"] + (2 if s["kind"] == "retaining" else 1) * m["dr"], m["column"] + (2 if s["kind"] == "retaining" else 1) * m["dc"]) for s in squares for m in s["markers"]]
                review_counts["overlap"] += len(targets) > len(set(targets))
                if split == "train":
                    review_counts["sparse_train"] += any(len(s["markers"]) == 1 for s in squares)
                    review_counts["complete_train"] += any(_has_complete_edge(s) for s in squares)
                else:
                    has_complete = any(_has_complete_edge(s) for s in squares)
                    review_counts["sparse_test"] += any(len(s["markers"]) == 1 for s in squares)
                    review_counts["complete_test"] += has_complete
                    review_counts["no_complete_test"] += not has_complete
                    review_counts["one_material_test"] += len({s["kind"] for s in squares}) == 1
                    review_counts["low_count_test"] += len(squares) <= 5
                    review_counts["crowded_test"] += len(squares) >= 6
        review_counts["background_varying_puzzle"] += len(puzzle_backgrounds) > 1
        review_counts["marker_varying_puzzle"] += len(puzzle_markers) > 1
        review_counts["different_shapes_puzzle"] += len(puzzle_shapes) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3)
    _require(len(shapes) >= 15 and len(layouts) >= 100 and len(material_counts) >= 7 and len(marker_counts) >= 12)
    minimums = {
        "edge_flush": 10, "corner_flush": 3, "staggered": 10,
        "sparse_train": 5, "sparse_test": 5, "one_side": 10,
        "complete_train": 5, "complete_test": 5, "no_complete_test": 5,
        "overlap": 10, "several_sides": 10, "mixed_markers": 5,
        "one_material_test": 3, "low_count_test": 10,
        "crowded_test": 5, "background_varying_puzzle": 3,
        "marker_varying_puzzle": 3, "different_shapes_puzzle": 30,
    }
    _require(all(review_counts[key] >= threshold for key, threshold in minimums.items()), "review diversity axis below minimum coverage")
