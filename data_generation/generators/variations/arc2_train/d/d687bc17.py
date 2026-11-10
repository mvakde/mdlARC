import numpy as np


_ROLE_TOP = 0
_ROLE_BOTTOM = 1
_ROLE_LEFT = 2
_ROLE_RIGHT = 3
_ROLES = (_ROLE_TOP, _ROLE_BOTTOM, _ROLE_LEFT, _ROLE_RIGHT)


def _source_pair(input_text: str, output_text: str) -> tuple:
    return (tuple(input_text.split()), tuple(output_text.split()))


_SOURCE_PAIRS = {
    _source_pair(
        """
011111111110
200000000004
200000000704
200000020004
200030000004
200000000004
200000004004
200000000004
200800000004
200001007004
200000000004
077777777770
""",
        """
011111111110
200001000004
200000000004
220000000004
200000000004
200000000004
200000000044
200000000004
200000000004
200000000004
200000007704
077777777770
""",
    ),
    _source_pair(
        """
044444444444440
200000000000003
200000000030003
200200000000003
200000070000003
200300000000003
200008000000003
200000000402003
200000000000003
088888888888880
""",
        """
044444444444440
200000000400003
200000000000033
220000000000003
200000000000003
200000000000033
200000000000003
220000000000003
200008000000003
088888888888880
""",
    ),
    _source_pair(
        """
06666666660
40000000008
40200000008
40000000608
40008000008
40000000008
40000000008
40040000008
40000000008
40000080008
40800000008
40000000008
40000000008
03333333330
""",
        """
06666666660
40000000608
40000000008
40000000008
40000000088
40000000008
40000000008
44000000008
40000000008
40000000088
40000000088
40000000008
40000000008
03333333330
""",
    ),
    _source_pair(
        """
04444444444444440
10000000000000002
10000008000010002
10020000000000302
10000000000000002
10000010007008002
10000000000000002
10002000000000002
10000000000000002
10000600400000402
10080000000100002
10000000000000002
10000000000000002
08888888888888880
""",
        """
04444444444444440
10000000400000402
11000000000000002
10000000000000022
10000000000000002
11000000000000002
10000000000000002
10000000000000022
10000000000000002
10000000000000002
11000000000000002
10000000000000002
10080008000008002
08888888888888880
""",
    ),
}


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        # This arithmetic schedule guarantees count diversity across
        # consecutive seeds while remaining deterministic.
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed // 4) % span
    elif num_train is None:
        minimum_train = max(2, int(num_test) + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train for the requested num_test")
        span = 11 - minimum_train
        num_train = minimum_train + int(rng.integers(0, span))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test for the requested num_train")
        num_test = 1 + int(rng.integers(0, maximum_test))

    assert num_train is not None
    assert num_test is not None

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _make_frame(
    height: int,
    width: int,
    background: int,
    sides: tuple[int, int, int, int],
) -> np.ndarray:
    top, bottom, left, right = sides
    grid = np.full((height, width), background, dtype=np.int8)
    grid[0, 1:-1] = top
    grid[-1, 1:-1] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right
    return grid


def _frame_info(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    height, width = grid.shape
    assert 7 <= height <= 30 and 7 <= width <= 30
    background = int(grid[0, 0])
    assert all(int(grid[row, col]) == background for row, col in
               ((0, -1), (-1, 0), (-1, -1)))
    lines = (grid[0, 1:-1], grid[-1, 1:-1],
             grid[1:-1, 0], grid[1:-1, -1])
    sides = tuple(int(line[0]) for line in lines)
    assert len(set(sides)) == 4 and background not in sides
    assert all(np.all(line == color) for line, color in zip(lines, sides))
    return background, sides


def _destination(role: int, row: int, col: int,
                 height: int, width: int) -> tuple[int, int]:
    if role == _ROLE_TOP:
        return 1, col
    if role == _ROLE_BOTTOM:
        return height - 2, col
    if role == _ROLE_LEFT:
        return row, 1
    if role == _ROLE_RIGHT:
        return row, width - 2
    raise AssertionError("invalid role")


def _solve_grid(input_grid: np.ndarray) -> np.ndarray:
    height, width = input_grid.shape
    background, sides = _frame_info(input_grid)
    roles = {color: role for role, color in enumerate(sides)}
    output = _make_frame(height, width, background, sides)
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            color = int(input_grid[row, col])
            role = roles.get(color)
            if role is None:
                continue
            target = _destination(role, row, col, height, width)
            assert int(output[target]) in (background, color)
            output[target] = color
    return output


def _sample_dimensions(rng: np.random.Generator, exemplar: bool) -> tuple[int, int]:
    # The independent span and aspect choices produce small, wide, tall,
    # square, and near-30 canvases in both splits.
    band = int(rng.choice(3, p=(0.30, 0.46, 0.24)))
    low, high = ((8, 11) if exemplar else (7, 11), (11, 19), (19, 31))[band]
    aspect = int(rng.integers(0, 3))
    if aspect == 0:
        size = int(rng.integers(low, high))
        return size, size
    short = int(rng.integers(low, high))
    long = min(30, short + int(rng.integers(2, 13)))
    return (short, long) if aspect == 1 else (long, short)


def _sample_background(rng: np.random.Generator, base: int,
                       exclude: tuple[int, ...] = (),
                       change_from: int | None = None) -> int:
    allowed = [c for c in range(10)
               if c not in exclude and c != change_from]
    if base in allowed and rng.random() < 0.55:
        return base
    return int(rng.choice(allowed))


def _random_sides(rng: np.random.Generator, pool: tuple[int, ...],
                  background: int,
                  required: tuple[int, int] | None = None,
                  forbidden: tuple[int, ...] = ()) -> tuple[int, int, int, int]:
    available = [c for c in pool if c != background and c not in forbidden]
    if required is None:
        return tuple(int(c) for c in rng.choice(available, size=4, replace=False))
    role, color = required
    assert color in available
    chosen = [int(c) for c in rng.choice(
        [c for c in available if c != color], size=3, replace=False)]
    chosen.insert(role, color)
    return tuple(chosen)


def _sample_role_counts(rng: np.random.Generator, area: int,
                        exemplar: bool) -> tuple[int, int, int, int]:
    if exemplar:
        counts = np.ones(4, dtype=np.int64)
        merge_role = int(rng.integers(0, 4))
        boundary_role = (merge_role + 1 + int(rng.integers(0, 3))) % 4
        counts[merge_role] += 1
        counts[boundary_role] += 1
        for _ in range(int(rng.integers(0, 3))):
            counts[int(rng.integers(0, 4))] += 1
        return tuple(int(x) for x in counts)
    maximum = min(18, max(4, area // 4))
    total = int(rng.integers(2, maximum + 1))
    weights = rng.dirichlet(np.full(4, float(rng.uniform(0.45, 2.2))))
    return tuple(int(x) for x in rng.multinomial(total, weights))


def _sample_decoys(rng: np.random.Generator, background: int,
                   sides: tuple[int, int, int, int],
                   mandatory: int | None = None) -> list[int]:
    available = [c for c in range(10) if c != background and c not in sides]
    total = int(rng.integers(0, 7))
    if mandatory is not None:
        assert mandatory in available
        total = max(1, total)
    if total == 0:
        return []
    palette_size = int(rng.integers(1, min(4, len(available), total) + 1))
    if mandatory is None:
        palette = [int(x) for x in rng.choice(available, size=palette_size,
                                            replace=False)]
        result = palette.copy()
    else:
        palette = [mandatory]
        if palette_size > 1:
            palette.extend(int(x) for x in rng.choice(
                [c for c in available if c != mandatory],
                size=palette_size - 1, replace=False))
        result = [mandatory] + palette[1:]
    while len(result) < total:
        result.append(int(rng.choice(palette)))
    rng.shuffle(result)
    return result


def _build_input(rng: np.random.Generator, height: int, width: int,
                 background: int, sides: tuple[int, int, int, int],
                 counts: tuple[int, int, int, int],
                 decoys: list[int], exemplar: bool) -> np.ndarray:
    grid = _make_frame(height, width, background, sides)
    remaining = list(counts)
    targets: dict[tuple[int, int], int] = {}
    occupied: set[tuple[int, int]] = set()
    if exemplar:
        merge_role = next(role for role, count in enumerate(counts) if count >= 2)
        boundary_role = next(role for role, count in enumerate(counts)
                             if role != merge_role and count >= 2)
        span = width - 2 if merge_role in (0, 1) else height - 2
        parallel = int(rng.choice((1, span))) if rng.random() < 0.55 else int(rng.integers(1, span + 1))
        if merge_role == _ROLE_TOP:
            cells = ((1, parallel), (2, parallel))
        elif merge_role == _ROLE_BOTTOM:
            cells = ((height - 2, parallel), (height - 3, parallel))
        elif merge_role == _ROLE_LEFT:
            cells = ((parallel, 1), (parallel, 2))
        else:
            cells = ((parallel, width - 2), (parallel, width - 3))
        for row, col in cells:
            grid[row, col] = sides[merge_role]
            occupied.add((row, col))
        targets[_destination(merge_role, *cells[0], height, width)] = sides[merge_role]
        remaining[merge_role] -= 2
        boundary_cells = []
        span = width - 2 if boundary_role in (0, 1) else height - 2
        for parallel in range(2, span):
            if boundary_role == _ROLE_TOP:
                cell = (1, parallel)
            elif boundary_role == _ROLE_BOTTOM:
                cell = (height - 2, parallel)
            elif boundary_role == _ROLE_LEFT:
                cell = (parallel, 1)
            else:
                cell = (parallel, width - 2)
            if cell not in occupied and cell not in targets:
                boundary_cells.append(cell)
        row, col = boundary_cells[int(rng.integers(0, len(boundary_cells)))]
        grid[row, col] = sides[boundary_role]
        occupied.add((row, col))
        targets[(row, col)] = sides[boundary_role]
        standalone_target = (row, col)
        remaining[boundary_role] -= 1

    role_order = [role for role, count in enumerate(remaining) for _ in range(count)]
    rng.shuffle(role_order)
    for role in role_order:
        color = sides[role]
        candidates = []
        weights = []
        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if (row, col) in occupied:
                    continue
                target = _destination(role, row, col, height, width)
                if target in targets and targets[target] != color:
                    continue
                if exemplar and target == standalone_target:
                    continue
                if exemplar and target == (row, col):
                    continue
                near_corner = row in (1, height - 2) and col in (1, width - 2)
                boundary = target == (row, col)
                adjacent = any(abs(row - r) + abs(col - c) == 1 for r, c in occupied)
                candidates.append((row, col, target))
                weights.append(1.0 + 2.0 * near_corner + 1.5 * boundary + 1.5 * adjacent)
        if not candidates:
            raise ValueError("no compatible marker location")
        probability = np.asarray(weights) / sum(weights)
        row, col, target = candidates[int(rng.choice(len(candidates), p=probability))]
        grid[row, col] = color
        occupied.add((row, col))
        targets[target] = color

    for color in decoys:
        available = [(row, col) for row in range(1, height - 1)
                     for col in range(1, width - 1)
                     if (row, col) not in occupied
                     and (not exemplar or (row, col) not in targets)]
        if not available:
            raise ValueError("no decoy location")
        row, col = available[int(rng.integers(0, len(available)))]
        grid[row, col] = color
        occupied.add((row, col))
    return grid


def _analyze_input(input_grid: np.ndarray) -> dict:
    height, width = input_grid.shape
    background, sides = _frame_info(input_grid)
    roles = {color: role for role, color in enumerate(sides)}
    counts = [0, 0, 0, 0]
    decoys: dict[int, int] = {}
    occupied: dict[tuple[int, int], int] = {}
    targets: dict[tuple[int, int], int] = {}
    target_counts: dict[tuple[int, int], int] = {}
    boundary_targets: list[tuple[int, int]] = []
    moved_roles: set[int] = set()
    merges = boundary = near_corner = 0
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            color = int(input_grid[row, col])
            if color == background:
                continue
            occupied[(row, col)] = color
            role = roles.get(color)
            if role is None:
                decoys[color] = decoys.get(color, 0) + 1
                continue
            counts[role] += 1
            target = _destination(role, row, col, height, width)
            if target in targets:
                assert targets[target] == color
                merges += 1
            targets[target] = color
            target_counts[target] = target_counts.get(target, 0) + 1
            if target == (row, col):
                boundary += 1
                boundary_targets.append(target)
            else:
                moved_roles.add(role)
            near_corner += row in (1, height - 2) and col in (1, width - 2)
    same_adjacent = different_adjacent = isolated = 0
    same_side_adjacent = different_side_adjacent = isolated_side = 0
    for (row, col), color in occupied.items():
        neighbors = [occupied[p] for p in ((row - 1, col), (row + 1, col),
                                            (row, col - 1), (row, col + 1))
                     if p in occupied]
        isolated += not neighbors
        same_adjacent += any(c == color for c in neighbors)
        different_adjacent += any(c != color for c in neighbors)
        if color in roles:
            side_neighbors = [c for c in neighbors if c in roles]
            isolated_side += not neighbors
            same_side_adjacent += any(c == color for c in side_neighbors)
            different_side_adjacent += any(c != color for c in side_neighbors)
    return {"height": height, "width": width, "background": background,
            "sides": sides, "role_counts": tuple(counts),
            "valid_count": sum(counts), "decoys": decoys,
            "merges": merges, "boundary": boundary,
            "standalone_boundary": sum(target_counts[t] == 1 for t in boundary_targets),
            "moved_roles": frozenset(moved_roles),
            "near_corner": near_corner, "same_adjacent": same_adjacent,
            "different_adjacent": different_adjacent, "isolated": isolated,
            "same_side_adjacent": same_side_adjacent,
            "different_side_adjacent": different_side_adjacent,
            "isolated_side": isolated_side,
            "marker_count": len(occupied)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    # Palette order and a preferred background are puzzle-level scaffolding.
    # The four directional roles and the actual background are read anew from
    # each frame; no hidden color assignment is needed to solve a test.
    palette = tuple(int(c) for c in rng.permutation(10))
    pool_size = int(rng.integers(7, 9))
    train_pool = palette[:pool_size]
    base_background = 0 if rng.random() < 0.45 else int(rng.integers(1, 10))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    pair_rngs = [np.random.default_rng(int(s)) for s in pair_seeds]

    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    first_sides = None
    first_background = None
    first_role = None
    prior_decoy = None
    seen_train_sides: set[int] = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = pair_rngs[index if split == "train" else num_train + index]
            exemplar = split == "train" and index == 0
            for attempt in range(50):
                height, width = _sample_dimensions(pair_rng, exemplar)
                if split == "train" and index == 1:
                    assert first_sides is not None and first_role is not None
                    background = _sample_background(
                        pair_rng, base_background,
                        exclude=(first_sides[first_role], prior_decoy),
                        change_from=first_background)
                    new_role = (first_role + 1 + int(pair_rng.integers(0, 3))) % 4
                    sides = _random_sides(pair_rng, train_pool, background,
                                          required=(new_role, first_sides[first_role]),
                                          forbidden=(prior_decoy,))
                    mandatory_decoy = prior_decoy
                else:
                    background = _sample_background(pair_rng, base_background)
                    if split == "train":
                        sides = _random_sides(pair_rng, train_pool, background)
                    else:
                        novel = [c for c in range(10)
                                 if c not in seen_train_sides and c != background]
                        if novel and pair_rng.random() < 0.55:
                            role = int(pair_rng.integers(0, 4))
                            color = int(pair_rng.choice(novel))
                            sides = _random_sides(pair_rng, tuple(range(10)),
                                                  background, required=(role, color))
                        else:
                            sides = _random_sides(pair_rng, tuple(range(10)), background)
                    mandatory_decoy = None
                if exemplar:
                    mandatory_decoy = int(pair_rng.choice(
                        [c for c in range(10) if c != background and c not in sides]))
                counts = _sample_role_counts(pair_rng, (height - 2) * (width - 2), exemplar)
                decoys = _sample_decoys(pair_rng, background, sides, mandatory_decoy)
                try:
                    input_grid = _build_input(pair_rng, height, width, background,
                                              sides, counts, decoys, exemplar)
                    output_grid = _solve_grid(input_grid)
                except (ValueError, AssertionError):
                    continue
                key = _grid_key(input_grid)
                if (key in used_inputs or np.array_equal(input_grid, output_grid)
                    or (_rows(input_grid), _rows(output_grid)) in _SOURCE_PAIRS):
                    continue
                break
            else:
                raise ValueError("could not construct a unique pair")
            used_inputs.add(key)
            puzzle[split].append({"input": input_grid, "output": output_grid})
            if split == "train":
                seen_train_sides.update(sides)
                if exemplar:
                    first_sides = sides
                    first_background = background
                    first_role = int(pair_rng.integers(0, 4))
                    prior_decoy = first_sides[(first_role + 1 + int(pair_rng.integers(0, 3))) % 4]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs: set[tuple] = set()
    pairs: set[tuple] = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert (_rows(inp), _rows(out)) not in _SOURCE_PAIRS
            analysis = _analyze_input(inp)
            assert analysis["valid_count"] >= 1
            assert analysis["marker_count"] <= max(22, (inp.shape[0]-2)*(inp.shape[1]-2)//3)
            assert np.array_equal(out, _solve_grid(inp))
            assert not np.array_equal(inp, out)
            assert _frame_info(out) == (analysis["background"], analysis["sides"])
            assert all(color not in analysis["sides"] and color != analysis["background"]
                       for color in analysis["decoys"])
            analyses[split].append(analysis)
    first, second = analyses["train"][:2]
    assert all(n >= 1 for n in first["role_counts"])
    assert first["decoys"] and first["merges"] and first["boundary"]
    assert first["standalone_boundary"] and first["moved_roles"] == set(_ROLES)
    assert first["same_side_adjacent"]
    assert first["background"] != second["background"]
    assert any(color in first["sides"] and role != first["sides"].index(color)
               for role, color in enumerate(second["sides"]))
    assert set(first["sides"]).intersection(second["decoys"])
    train_roles = {role for analysis in analyses["train"]
                   for role, count in enumerate(analysis["role_counts"]) if count}
    assert train_roles == set(_ROLES)
    assert any(a["decoys"] for a in analyses["train"])
    # The frame itself gives the destination for every test marker. The
    # demonstrations additionally establish every direction, decoy removal,
    # boundary stability, merging, and that literal colors change roles.


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[split], second[split])
                   for k in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    for input_rows, output_rows in _SOURCE_PAIRS:
        source_input = np.asarray([[int(c) for c in row] for row in input_rows],
                                  dtype=np.int8)
        source_output = np.asarray([[int(c) for c in row] for row in output_rows],
                                   dtype=np.int8)
        assert np.array_equal(_solve_grid(source_input), source_output)
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzles_equal(generate_puzzle(77, 6, 3), generate_puzzle(77, 6, 3))
    assert not _puzzles_equal(generate_puzzle(77, 6, 3), generate_puzzle(78, 6, 3))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    for request in ({"num_train": 1}, {"num_train": 11},
                    {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(100, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    shapes = set()
    palettes = set()
    layouts = set()
    features = {"zero_side": 0, "no_decoy": 0, "repeat_decoy": 0,
                "nonzero_background": 0, "zero_side_color": 0,
                "zero_decoy_color": 0, "boundary": 0, "corner": 0,
                "same_adjacent": 0, "different_adjacent": 0,
                "merge": 0, "isolated": 0, "novel_test_side": 0,
                "test_smaller": 0, "test_larger": 0,
                "test_sparser": 0, "test_denser": 0}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test = [_analyze_input(p["input"]) for p in puzzle["test"]]
        combined = train + test
        assert len({(a["height"], a["width"]) for a in combined}) >= 2
        assert len({a["background"] for a in combined}) >= 2
        assert len({a["sides"] for a in combined}) >= 2
        assert len({a["valid_count"] for a in combined}) >= 2
        assert len({sum(a["decoys"].values()) for a in combined}) >= 2
        assert len({a["role_counts"] for a in combined}) >= 2
        train_sides = {c for a in train for c in a["sides"]}
        features["novel_test_side"] += any(set(a["sides"]) - train_sides for a in test)
        features["test_smaller"] += any(a["height"]*a["width"] <
                                        min(t["height"]*t["width"] for t in train) for a in test)
        features["test_larger"] += any(a["height"]*a["width"] >
                                       max(t["height"]*t["width"] for t in train) for a in test)
        features["test_sparser"] += any(a["valid_count"] <
                                        min(t["valid_count"] for t in train) for a in test)
        features["test_denser"] += any(a["valid_count"] >
                                       max(t["valid_count"] for t in train) for a in test)
        for pair, a in zip(puzzle["train"] + puzzle["test"], train + test):
            inp, out = pair["input"], pair["output"]
            # Reconstruct with the input frame and input marker coordinates.
            predicted = np.full(inp.shape, a["background"], dtype=np.int8)
            predicted[0, :] = inp[0, :]
            predicted[-1, :] = inp[-1, :]
            predicted[:, 0] = inp[:, 0]
            predicted[:, -1] = inp[:, -1]
            for role, color in enumerate(a["sides"]):
                points = np.argwhere(inp[1:-1, 1:-1] == color)
                for local_row, local_col in points:
                    row, col = int(local_row)+1, int(local_col)+1
                    predicted[_destination(role, row, col, *inp.shape)] = color
            assert np.array_equal(predicted, out)
            shapes.add(inp.shape)
            palettes.add((a["background"], a["sides"]))
            layouts.add((inp.shape, tuple(map(tuple, np.argwhere(inp[1:-1,1:-1] != a["background"])))))
            features["zero_side"] += 0 in a["role_counts"]
            features["no_decoy"] += not a["decoys"]
            features["repeat_decoy"] += any(n > 1 for n in a["decoys"].values())
            features["nonzero_background"] += a["background"] != 0
            features["zero_side_color"] += 0 in a["sides"]
            features["zero_decoy_color"] += 0 in a["decoys"]
            for name, key in (("boundary", "boundary"), ("corner", "near_corner"),
                              ("same_adjacent", "same_side_adjacent"),
                              ("different_adjacent", "different_side_adjacent"),
                              ("merge", "merges"), ("isolated", "isolated_side")):
                features[name] += a[key] > 0
    assert len(shapes) >= 60 and len(palettes) >= 100 and len(layouts) >= 200
    assert all(value > 0 for value in features.values()), features
