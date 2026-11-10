import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ROTATIONS = {
    "down": 0,
    "right": 1,
    "up": 2,
    "left": 3,
}

# Exact source pairs, used only to ensure that generated pairs are new.
_SOURCE_PAIR_ROWS = (
    (
        (
            "777777",
            "777777",
            "777775",
            "777777",
            "777777",
            "777770",
            "777700",
            "770000",
            "700000",
            "777000",
        ),
        (
            "777777",
            "777777",
            "777777",
            "777777",
            "777777",
            "777770",
            "777500",
            "770000",
            "700000",
            "777000",
        ),
    ),
    (
        (
            "77777777",
            "77777777",
            "77777777",
            "77777707",
            "77777007",
            "75770007",
        ),
        (
            "77777777",
            "77777777",
            "77777757",
            "77777707",
            "77777007",
            "77770007",
        ),
    ),
    (
        (
            "77777777",
            "00777777",
            "00007777",
            "00000077",
            "00007777",
            "00777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "57777777",
        ),
        (
            "77777777",
            "00777777",
            "00007777",
            "00000077",
            "00007777",
            "00777777",
            "57777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
        ),
    ),
    (
        (
            "7777777777777777",
            "7777777777777777",
            "7770777777777777",
            "7770077777777777",
            "7770007777777777",
            "7770007777777777",
            "7770000777777777",
            "7770000077777777",
            "7700000007777777",
            "7700000000775777",
        ),
        (
            "7777777777777777",
            "7777777777777777",
            "7770777777777777",
            "7770077777777777",
            "7770007777777777",
            "7770005777777777",
            "7770000777777777",
            "7770000077777777",
            "7700000007777777",
            "7700000000777777",
        ),
    ),
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _pair_signature(pair: dict) -> tuple:
    return (_grid_signature(pair["input"]), _grid_signature(pair["output"]))


def _is_source_duplicate(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(input_grid), rows(output_grid)) in _SOURCE_PAIR_ROWS


def _inside(shape: tuple[int, int], point: tuple[int, int]) -> bool:
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _connected(mask: np.ndarray) -> bool:
    cells = {tuple(int(x) for x in p) for p in np.argwhere(mask)}
    if len(cells) < 4:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in _DIRECTIONS:
            p = (r + dr, c + dc)
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return len(seen) == len(cells)


def _input_roles(grid: np.ndarray) -> tuple[int, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    background = int(values[int(np.argmax(counts))])
    singletons = [int(v) for v, n in zip(values, counts) if n == 1]
    assert len(singletons) == 1
    marker = singletons[0]
    obstacle = next(int(v) for v in values if v not in (background, marker))
    assert int(np.count_nonzero(grid == background)) > int(np.count_nonzero(grid == obstacle))
    return background, obstacle, marker


def _borders(mask: np.ndarray, point: tuple[int, int]) -> bool:
    return any(_inside(mask.shape, (point[0] + dr, point[1] + dc))
               and bool(mask[point[0] + dr, point[1] + dc])
               for dr, dc in _DIRECTIONS)


def _trace_input(grid: np.ndarray) -> dict:
    """An input-only interpretation, independent of construction parameters."""
    background, obstacle, marker = _input_roles(grid)
    mask = grid == obstacle
    assert _connected(mask)
    old = tuple(int(x) for x in np.argwhere(grid == marker)[0])
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            p = (old[0] + dr, old[1] + dc)
            if _inside(grid.shape, p):
                assert not mask[p]

    hits = []
    for dr, dc in _DIRECTIONS:
        p = old
        distance = 0
        while True:
            p = (p[0] + dr, p[1] + dc)
            distance += 1
            if not _inside(grid.shape, p):
                break
            if mask[p]:
                hits.append(((dr, dc), distance, p))
                break
            assert int(grid[p]) == background
    assert len(hits) == 1
    approach, distance, hit = hits[0]
    assert distance >= 3
    contact = (hit[0] - approach[0], hit[1] - approach[1])
    assert int(grid[contact]) == background

    diagonals = ((approach[0] + approach[1], approach[1] - approach[0]),
                 (approach[0] - approach[1], approach[1] + approach[0]))
    valid = []
    for dr, dc in diagonals:
        p = (contact[0] + dr, contact[1] + dc)
        if _inside(grid.shape, p) and int(grid[p]) == background and _borders(mask, p):
            valid.append((dr, dc))
    assert len(valid) <= 1
    diagonal = valid[0] if valid else None
    target = contact
    length = 0
    if diagonal is not None:
        while True:
            p = (target[0] + diagonal[0], target[1] + diagonal[1])
            if not _inside(grid.shape, p):
                termination = "edge"
                break
            if mask[p]:
                termination = "block"
                break
            if not _borders(mask, p):
                termination = "end"
                break
            assert int(grid[p]) == background
            target = p
            length += 1
    else:
        # Both first diagonals fail. The actual reason can be a blocked
        # cell or an edge; a zero-length route needs no side convention.
        failures = []
        for dr, dc in diagonals:
            p = (contact[0] + dr, contact[1] + dc)
            failures.append("edge" if not _inside(grid.shape, p)
                            else "block" if mask[p] else "end")
        termination = "block" if "block" in failures else "edge" if "edge" in failures else "end"
    return dict(roles=(background, obstacle, marker), old=old, contact=contact,
                target=target, approach=approach, diagonal=diagonal,
                route_length=length, termination=termination,
                distance=distance, obstacle_count=int(mask.sum()))


def _solve_input(grid: np.ndarray) -> np.ndarray:
    trace = _trace_input(grid)
    answer = grid.copy()
    answer[trace["old"]] = trace["roles"][0]
    answer[trace["target"]] = trace["roles"][2]
    return answer


def _valid_count(value: int, name: str, low: int, high: int) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, seed: int,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    if num_train is not None:
        num_train = _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _valid_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _make_canonical_input(rng: np.random.Generator, colors: tuple[int, int, int],
                          length: int, side: int, mode: str, shape: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a down-approach route; obstacle rows are connected intervals.

    The visible face advances one column per row. Its far flank and the
    rows after the route change separately, producing shoulders and tails.
    """
    assert side in (-1, 1) and 0 <= length <= 8
    assert mode in ("block", "end", "edge")
    assert shape in ("wide", "tall", "square", "free")
    assert length or mode != "end"
    edge_kind = "side" if length == 0 else ("side" if rng.integers(2) else "bottom")
    gap = int(rng.integers(2, 8))
    top = int(rng.integers(0, 4))
    contact_row = top + gap
    thickness = int(rng.integers(2, 8))
    rows: list[tuple[int, int, int]] = []
    far = -thickness

    def append_row(j: int, boundary: int) -> None:
        nonlocal far
        far += int(rng.integers(-2, 3))
        far = min(-1, max(-10, far))
        rows.append((j, far, boundary))

    for j in range(1, length + 1):
        append_row(j, j - 1)
    if mode == "block":
        append_row(length + 1, length + 1)
    elif mode == "end":
        append_row(length + 1, length - 1)
    elif edge_kind == "side" and length == 0:
        append_row(1, 0)

    if mode != "edge" or edge_kind == "side":
        tail = int(rng.integers(0, 5))
        previous = rows[-1][2]
        for extra in range(tail):
            j = rows[-1][0] + 1
            boundary = previous + int(rng.integers(-2, 3))
            boundary = max(-1, min(length if mode in ("end", "edge") else length + 3, boundary))
            # The first post-end row must not touch the next diagonal cell.
            if mode == "end" and j == length + 2:
                boundary = min(boundary, length)
            append_row(j, boundary)
            previous = boundary

    min_col = min(-1, *(lo for _, lo, _ in rows))
    max_col = max(length, *(hi for _, _, hi in rows))
    left_pad = int(rng.integers(0, 5))
    right_pad = int(rng.integers(0, 5))
    bottom_pad = int(rng.integers(0, 5))
    if mode == "end":
        right_pad = max(1, right_pad)
    if mode == "edge" and edge_kind == "side":
        right_pad = 0
    if mode == "edge" and edge_kind == "bottom":
        bottom_pad = 0
    extra = int(rng.integers(2, 9))
    natural_width = max_col - min_col + 1 + left_pad + right_pad
    natural_height = contact_row + rows[-1][0] + 1 + bottom_pad
    if shape == "wide":
        right_pad += max(0, natural_height + extra - natural_width)
    elif shape == "tall":
        bottom_pad += max(0, natural_width + extra - natural_height)
    elif shape == "square":
        difference = natural_height - natural_width
        if difference > 0:
            right_pad += difference
        else:
            bottom_pad -= difference
    if mode == "edge" and edge_kind == "side":
        # Place extra width on the opposite side to retain the edge stop.
        left_pad += right_pad
        right_pad = 0
    if mode == "edge" and edge_kind == "bottom":
        top += bottom_pad
        contact_row += bottom_pad
        bottom_pad = 0

    width = max_col - min_col + 1 + left_pad + right_pad
    height = contact_row + rows[-1][0] + 1 + bottom_pad
    assert 6 <= width <= 30 and 6 <= height <= 30
    column_offset = left_pad - min_col
    grid = np.full((height, width), colors[0], dtype=np.int8)
    for j, lo, hi in rows:
        grid[contact_row + j, column_offset + lo:column_offset + hi + 1] = colors[1]
    old = (top, column_offset)
    target = (contact_row + length, column_offset + length)
    grid[old] = colors[2]
    if side == -1:
        grid = np.fliplr(grid).copy()
        target = (target[0], width - 1 - target[1])
    return grid, target


def _generate_pair(rng: np.random.Generator, colors: tuple[int, int, int],
                   orientation: str, length: int, side: int, mode: str,
                   shape: str) -> dict:
    grid, target = _make_canonical_input(rng, colors, length, side, mode, shape)
    quarter_turns = _ROTATIONS[orientation]
    for _ in range(quarter_turns):
        target = (grid.shape[1] - 1 - target[1], target[0])
        grid = np.rot90(grid).copy()
    old = tuple(int(x) for x in np.argwhere(grid == colors[2])[0])
    output = grid.copy()
    output[old] = colors[0]
    output[target] = colors[2]
    trace = _trace_input(grid)
    assert trace["route_length"] == length and trace["termination"] == mode
    assert np.array_equal(output, _solve_input(grid))
    return {"input": grid, "output": output}


def _child_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    palette_mode = "varied" if int(rng.integers(2)) else "stable"
    base_colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    def colors_for_pair() -> tuple[int, int, int]:
        if palette_mode == "stable":
            return base_colors
        return tuple(int(x) for x in rng.choice(10, size=3, replace=False))

    directions = list(_ROTATIONS)
    rng.shuffle(directions)
    first_side = -1 if int(rng.integers(2)) else 1
    train_specs = [
        (directions[0], int(rng.integers(2, 6)), first_side, "block"),
        (directions[1], int(rng.integers(1, 6)), -first_side, "end"),
    ]
    if num_train >= 3:
        if int(rng.integers(2)):
            train_specs.append((directions[2], 0, first_side, "block"))
        else:
            train_specs.append((directions[2], int(rng.integers(1, 6)), first_side, "edge"))
    while len(train_specs) < num_train:
        mode = ("block", "end", "edge")[int(rng.integers(3))]
        length = int(rng.integers(0 if mode == "block" else 1, 7))
        train_specs.append((directions[int(rng.integers(4))], length,
                            -1 if int(rng.integers(2)) else 1, mode))
    rng.shuffle(train_specs)
    trained_modes = {spec[3] for spec in train_specs}
    has_zero = any(spec[1] == 0 for spec in train_specs)
    trained_directions = {spec[0] for spec in train_specs}
    held_out = [d for d in directions if d not in trained_directions]
    test_specs = []
    for _ in range(num_test):
        direction_pool = held_out if held_out and int(rng.integers(2)) else list(trained_directions)
        orientation = direction_pool[int(rng.integers(len(direction_pool)))]
        mode = tuple(sorted(trained_modes))[int(rng.integers(len(trained_modes)))]
        if has_zero and mode == "block" and int(rng.integers(3)) == 0:
            length = 0
        else:
            length = int(rng.integers(1, 9))
        test_specs.append((orientation, length,
                           -1 if int(rng.integers(2)) else 1, mode))

    puzzle = {"train": [], "test": []}
    used = set()
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for orientation, length, side, mode in specs:
            for _ in range(250):
                child = _child_rng(rng)
                shape = ("wide", "tall", "square", "free")[int(child.integers(4))]
                try:
                    pair = _generate_pair(child, colors_for_pair(), orientation,
                                          length, side, mode, shape)
                except AssertionError:
                    continue
                signature = _grid_signature(pair["input"])
                if signature not in used and not _is_source_duplicate(pair["input"], pair["output"]):
                    used.add(signature)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct, valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_signature(inp)
            both = _pair_signature(pair)
            assert key not in seen_inputs and both not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(both)
            assert not _is_source_duplicate(inp, out)
            trace = _trace_input(inp)
            assert np.array_equal(out, _solve_input(inp))
            assert int(np.count_nonzero(out == trace["roles"][2])) == 1
            assert np.array_equal(inp == trace["roles"][1], out == trace["roles"][1])
            analyses[section].append(trace)
    train = analyses["train"]
    test = analyses["test"]
    assert len({a["approach"] for a in train}) >= 2
    assert any(a["route_length"] >= 2 for a in train)
    bends = {a["approach"][0] * a["diagonal"][1] -
             a["approach"][1] * a["diagonal"][0]
             for a in train if a["diagonal"] is not None}
    assert bends == {-1, 1}
    assert all(a["route_length"] <= 8 for a in train + test)
    assert {a["termination"] for a in test} <= {a["termination"] for a in train}
    if any(a["route_length"] == 0 for a in test):
        assert any(a["route_length"] == 0 for a in train)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
                   for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(123, 5, 2), generate_puzzle(123, 5, 2))
    for seed, a, b in ((91, 2, 1), (92, 10, 4), (93, None, 4),
                       (94, 2, None), (95, 10, 1), (96, None, 1),
                       (97, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(500, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_pair_signature(pair) for s in ("train", "test")
                      for pair in p[s]) for p in corpus}) == 42
    assert len({len(p["train"]) for p in corpus}) >= 4
    assert len({len(p["test"]) for p in corpus}) >= 3
    for source_input, source_output in _SOURCE_PAIR_ROWS:
        inp = np.array([[int(x) for x in row] for row in source_input], dtype=np.int8)
        out = np.array([[int(x) for x in row] for row in source_output], dtype=np.int8)
        assert np.array_equal(_solve_input(inp), out)
    records = [(s, pair, _trace_input(pair["input"])) for p in corpus
               for s in ("train", "test") for pair in p[s]]
    assert len({pair["input"].shape for _, pair, _ in records}) >= 30
    assert len({a["approach"] for _, _, a in records}) == 4
    assert len({a["route_length"] for _, _, a in records}) >= 7
    assert {a["termination"] for _, _, a in records} == {"block", "end", "edge"}
    assert any(s == "test" and a["route_length"] <= 2 for s, _, a in records)
    assert any(s == "test" and a["route_length"] >= 6 for s, _, a in records)
    assert any(s == "test" and a["route_length"] == 0 for s, _, a in records)
    assert any(s == "test" and a["termination"] == "block" and
               a["route_length"] >= 2 for s, _, a in records)
    roles_per_puzzle = [{_trace_input(pair["input"])["roles"]
                         for s in ("train", "test") for pair in p[s]}
                        for p in corpus]
    assert sum(len(roles) == 1 for roles in roles_per_puzzle) >= 8
    assert sum(len(roles) > 1 for roles in roles_per_puzzle) >= 8
    for section in ("train", "test"):
        subset = [(pair, a) for s, pair, a in records if s == section]
        aspects = {"wide" if pair["input"].shape[1] > pair["input"].shape[0] + 2
                   else "tall" if pair["input"].shape[0] > pair["input"].shape[1] + 2
                   else "square" for pair, _ in subset}
        assert aspects == {"wide", "tall", "square"}
        assert {a["approach"] for _, a in subset} == set(_DIRECTIONS)
        assert len({a["target"] for _, a in subset}) >= 15
    test_interior = {"block": 0, "end": 0}
    nonconvex = 0
    rectangles = 0
    for section, pair, a in records:
        mask = pair["input"] == a["roles"][1]
        positions = np.argwhere(mask)
        top, left = positions.min(axis=0)
        bottom, right = positions.max(axis=0)
        if section == "test" and a["termination"] in test_interior and top <= a["target"][0] <= bottom and left <= a["target"][1] <= right:
            test_interior[a["termination"]] += 1
        nonconvex += any((rows := np.flatnonzero(mask[:, c])).size > 1 and
                         int(rows[-1] - rows[0] + 1) != rows.size
                         for c in range(mask.shape[1]))
        rectangles += bool(np.all(mask[top:bottom + 1, left:right + 1]))
    assert test_interior["block"] >= 10 and test_interior["end"] >= 10
    assert nonconvex >= 30 and rectangles >= 5
    shorter = longer = reused = held_out = 0
    for puzzle in corpus:
        train = [_trace_input(pair["input"]) for pair in puzzle["train"]]
        test = [_trace_input(pair["input"]) for pair in puzzle["test"]]
        maximum = max(a["route_length"] for a in train)
        approaches = {a["approach"] for a in train}
        shorter += any(a["route_length"] < maximum for a in test)
        longer += any(a["route_length"] > maximum for a in test)
        reused += any(a["approach"] in approaches for a in test)
        held_out += any(a["approach"] not in approaches for a in test)
    assert min(shorter, longer, reused, held_out) >= 10
    for _, pair, a in records:
        assert np.array_equal(_solve_input(pair["input"]), pair["output"])
        assert not _is_source_duplicate(pair["input"], pair["output"])
