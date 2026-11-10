import numpy as np


_ORIENTATIONS = ("H", "V", "D", "A")

# D means a "\" diagonal with constant row - column.
# A means a "/" diagonal with constant row + column.

_SOURCE_SIGNATURES = frozenset({
    (
        "7772777/7772777/7772777/5555555/7772777/7772777/7772777",
        "25",
    ),
    (
        "9777778777/7977778777/7797778777/7779778777/"
        "0000908000/7777798777/7777779777/7777778977/"
        "7777778797/7777778779",
        "089",
    ),
    (
        "717737775777/771737775777/777137775777/777717775777/"
        "777731775777/222222125222/777737715777/777737775777/"
        "777737775177/777737775717/777737775771/777737775777",
        "3215",
    ),
    (
        "7777777777579747/7777777777579747/7777777777579747/"
        "7777777777579747/7777777777579747/7777777777579747/"
        "7777777777579747/7777777777579747/7777777777579747/"
        "7777777777579747/7777777777579747/7777777777579747/"
        "3333333333533333/7777777777579747/1111111111519111/"
        "7777777777579747",
        "41935",
    ),
    (
        "7777774175777777/7777774717777777/7777774571777777/"
        "7777774777177777/7777754777717777/0777574777771777/"
        "7075774777777173/9999999999999939/7570774777777371/"
        "5777074777773777/7777704777737777/7777770777377777/"
        "7777774073777777/7777774737777777/7777774370777777/"
        "7777773777077777",
        "540913",
    ),
})


def _line_cells(
    line: tuple[str, int],
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    orientation, value = line

    if orientation == "H":
        return [(value, col) for col in range(width)]

    if orientation == "V":
        return [(row, value) for row in range(height)]

    if orientation == "D":
        return [
            (row, row - value)
            for row in range(height)
            if 0 <= row - value < width
        ]

    if orientation == "A":
        return [
            (row, value - row)
            for row in range(height)
            if 0 <= value - row < width
        ]

    raise AssertionError("unknown line orientation")


def _available_lines(
    height: int,
    width: int,
    orientation: str,
    min_length: int,
) -> list[tuple[str, int]]:
    if orientation == "H":
        values = range(height)
    elif orientation == "V":
        values = range(width)
    elif orientation == "D":
        values = range(-(width - 1), height)
    elif orientation == "A":
        values = range(height + width - 1)
    else:
        raise AssertionError("unknown line orientation")

    return [
        (orientation, int(value))
        for value in values
        if len(
            _line_cells(
                (orientation, int(value)),
                height,
                width,
            )
        ) >= min_length
    ]


def _cell_members(
    lines: list[tuple[str, int]],
    height: int,
    width: int,
) -> dict[tuple[int, int], list[int]]:
    members: dict[tuple[int, int], list[int]] = {}

    for index, line in enumerate(lines):
        for cell in _line_cells(line, height, width):
            members.setdefault(cell, []).append(index)

    return members


def _unique_topological_order(
    number_of_nodes: int,
    edges: set[tuple[int, int]],
) -> list[int] | None:
    adjacency = [set() for _ in range(number_of_nodes)]
    indegree = [0] * number_of_nodes

    for lower, upper in edges:
        if lower != upper and upper not in adjacency[lower]:
            adjacency[lower].add(upper)
            indegree[upper] += 1

    remaining = set(range(number_of_nodes))
    result: list[int] = []

    while remaining:
        choices = [
            node
            for node in remaining
            if indegree[node] == 0
        ]

        if len(choices) != 1:
            return None

        node = choices[0]
        remaining.remove(node)
        result.append(node)

        for successor in adjacency[node]:
            indegree[successor] -= 1

    return result


def _order_is_observable(
    lines: list[tuple[str, int]],
    order: list[int],
    height: int,
    width: int,
) -> bool:
    owner = np.full((height, width), -1, dtype=np.int16)

    for index in order:
        for row, col in _line_cells(lines[index], height, width):
            owner[row, col] = index

    # Each latent line must retain enough visible cells to determine its
    # orientation without hidden generator information.
    if any(
        np.count_nonzero(owner == index) < 2
        for index in range(len(lines))
    ):
        return False

    edges: set[tuple[int, int]] = set()

    for cell, indices in _cell_members(
        lines, height, width
    ).items():
        if len(indices) < 2:
            continue

        visible = int(owner[cell])
        if visible not in indices:
            return False

        edges.update(
            (index, visible)
            for index in indices
            if index != visible
        )

    return (
        _unique_topological_order(len(lines), edges)
        == list(order)
    )


def _sample_dimensions(
    rng: np.random.Generator, number: int
) -> tuple[int, int]:
    shape = int(rng.integers(5))
    short = 3 if number <= 4 else 7
    if shape == 0:
        return (int(rng.integers(short, 11)),
                int(rng.integers(short, 11)))
    if shape == 1:
        return (int(rng.integers(8, 22)),
                int(rng.integers(8, 22)))
    if shape == 2:
        return (int(rng.integers(20, 31)),
                int(rng.integers(20, 31)))
    long_side = int(rng.integers(16, 31))
    short_side = int(rng.integers(max(4, short), 15))
    if shape == 3:
        return short_side, long_side
    return long_side, short_side


def _orientation_group(
    rng: np.random.Generator,
    number: int,
    available: tuple[str, ...],
    required: tuple[str, ...],
) -> tuple[str, ...]:
    minimum = max(2, len(required))
    size = int(rng.integers(minimum, min(number, len(available)) + 1))
    group = list(required)
    options = [orientation for orientation in available
               if orientation not in group]
    for value in rng.permutation(np.array(options))[:size - len(group)]:
        group.append(str(value))
    return tuple(group)


def _intersects(
    first: tuple[str, int], second: tuple[str, int],
    height: int, width: int,
) -> bool:
    return bool(set(_line_cells(first, height, width))
                & set(_line_cells(second, height, width)))


def _fourway_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
    number: int,
    pools: dict[str, list[tuple[str, int]]],
) -> list[tuple[str, int]] | None:
    for _ in range(120):
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        pivot = (row, col)
        anchors = {"H": ("H", row), "V": ("V", col),
                   "D": ("D", row - col), "A": ("A", row + col)}
        if any(line not in pools[orientation]
               for orientation, line in anchors.items()):
            continue
        shuffled = [str(value) for value in rng.permutation(
            np.array(_ORIENTATIONS))]
        anchor_lines = [anchors[orientation] for orientation in shuffled]
        lines = [anchor_lines[0]]
        used = set(anchor_lines)
        for following in anchor_lines[1:]:
            bridges = [line for pool in pools.values() for line in pool
                       if line not in used
                       and pivot not in _line_cells(line, height, width)
                       and _intersects(lines[-1], line, height, width)
                       and _intersects(line, following, height, width)]
            if not bridges:
                break
            bridge = bridges[int(rng.integers(len(bridges)))]
            lines.extend((bridge, following))
            used.add(bridge)
        if len(lines) != 7:
            continue
        while len(lines) < number:
            candidates = [line for pool in pools.values() for line in pool
                          if line not in used
                          and _intersects(lines[-1], line, height, width)]
            if not candidates:
                break
            line = candidates[int(rng.integers(len(candidates)))]
            lines.append(line)
            used.add(line)
        if len(lines) == number and _order_is_observable(
            lines, list(range(number)), height, width
        ):
            return lines
    return None


def _build_line_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
    number: int,
    group: tuple[str, ...],
) -> list[tuple[str, int]] | None:
    pools = {orientation: _available_lines(height, width, orientation, 3)
             for orientation in group}
    if any(not pool for pool in pools.values()):
        return None

    if (number >= 7 and len(group) == 4 and
            rng.random() < 0.4):
        fourway = _fourway_layout(rng, height, width, number, pools)
        if fourway is not None:
            return fourway

    for _ in range(100):
        sequence = [str(rng.choice(np.array(group)))]
        for _ in range(number - 1):
            choices = [orientation for orientation in group
                       if orientation != sequence[-1]]
            sequence.append(str(rng.choice(np.array(choices))))
        if set(sequence) != set(group):
            continue

        lines = [pools[sequence[0]][int(rng.integers(len(pools[sequence[0]])))]]
        used = set(lines)
        for orientation in sequence[1:]:
            candidates = [line for line in pools[orientation]
                          if line not in used and
                          _intersects(lines[-1], line, height, width)]
            if not candidates:
                break
            line = candidates[int(rng.integers(len(candidates)))]
            lines.append(line)
            used.add(line)
        if len(lines) == number and _order_is_observable(
            lines, list(range(number)), height, width
        ):
            return lines
    return None


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    background: int,
    number: int,
    available: tuple[str, ...],
    required: tuple[str, ...] = (),
) -> dict:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    for _ in range(80):
        height, width = _sample_dimensions(rng, number)
        group = _orientation_group(rng, number, available, required)
        lines = _build_line_layout(rng, height, width, number, group)
        if lines is None:
            continue
        palette = np.array([color for color in range(10)
                            if color != background], dtype=np.int8)
        colors = [int(color) for color in rng.choice(
            palette, size=number, replace=False)]
        grid = np.full((height, width), background, dtype=np.int8)
        for line, color in zip(lines, colors):
            for row, col in _line_cells(line, height, width):
                grid[row, col] = color
        counts = np.bincount(grid.ravel(), minlength=10)
        if counts[background] <= max(counts[color] for color in colors):
            continue
        return {"input": grid, "output": np.array([colors], dtype=np.int8)}
    raise RuntimeError("could not construct an unambiguous line layout")


def _validated_optional_count(
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
        raise ValueError(
            f"{name} must be an integer or None"
        )

    result = int(value)

    if not low <= result <= high:
        raise ValueError(
            f"{name} is outside its legal range"
        )

    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validated_optional_count(
        num_train,
        "num_train",
        2,
        10,
    )
    test_count = _validated_optional_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(
            rng.integers(test_count + 1, 11)
        )

    elif train_count is None:
        train_count = int(
            rng.integers(
                max(2, test_count + 1),
                11,
            )
        )

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(
            rng.integers(1, maximum_test + 1)
        )

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _array_signature(
    array: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, ...]]:
    return (
        array.shape,
        tuple(int(value) for value in array.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _array_signature(pair["input"]),
        _array_signature(pair["output"]),
    )


def _grid_text(array: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in array
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)

    train_count, test_count = _resolve_counts(
        rng,
        num_train,
        num_test,
    )

    # Only the back-to-front rule is a shared convention. Background color
    # is a visible geometric role and may change from pair to pair.
    shift = int(rng.integers(4))
    stable_background = bool(rng.integers(2))
    background_palette = [int(value) for value in rng.permutation(10)]
    demonstration = int(rng.integers(train_count))
    rich_demonstration = (demonstration + 1) % train_count

    pair_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=train_count + test_count,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs: set[tuple] = set()

    for index in range(train_count + test_count):
        pair_rng = np.random.default_rng(pair_seeds[index])
        training = index < train_count
        if training:
            regime = "source_train"
            if index == demonstration:
                number = 2
            elif shift == 0:
                number = int(pair_rng.integers(2, 7))
            elif shift == 3:
                number = int(pair_rng.integers(4, 10))
            else:
                number = int(pair_rng.integers(2, 10))
            if shift == 1 and index == rich_demonstration:
                number = max(3, number)
            available = ("H", "V", "D") if shift == 2 else _ORIENTATIONS
            required = ("D", "A") if shift == 1 and index == rich_demonstration else ()
            split = "train"
        else:
            regime = "target_test"
            if shift == 0:
                number = int(pair_rng.integers(6, 10))
            elif shift == 3:
                number = int(pair_rng.integers(2, 7))
            elif shift == 2:
                number = int(pair_rng.integers(2, 9))
            else:
                number = int(pair_rng.integers(3, 10))
            available = _ORIENTATIONS
            required = ("A",) if shift == 2 else (("D", "A") if shift == 1 else ())
            split = "test"

        background = background_palette[0 if stable_background else index % 10]
        for _ in range(20):
            pair = _generate_pair(pair_rng, regime, background,
                                  number, available, required)
            signature = _array_signature(pair["input"])
            if signature not in used_inputs:
                break
        else:
            raise RuntimeError("could not generate a distinct pair")
        used_inputs.add(signature)
        puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30
    assert 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _infer_line(
    input_grid: np.ndarray,
    color: int,
) -> tuple[str, int]:
    coordinates = np.argwhere(input_grid == color)
    assert len(coordinates) >= 2

    rows = coordinates[:, 0]
    cols = coordinates[:, 1]
    candidates: list[tuple[str, int]] = []

    if np.all(rows == rows[0]):
        candidates.append(("H", int(rows[0])))

    if np.all(cols == cols[0]):
        candidates.append(("V", int(cols[0])))

    differences = rows - cols
    if np.all(differences == differences[0]):
        candidates.append(("D", int(differences[0])))

    sums = rows + cols
    if np.all(sums == sums[0]):
        candidates.append(("A", int(sums[0])))

    # A visible color must determine exactly one permitted straight line.
    assert len(candidates) == 1
    return candidates[0]


def _solve_input(input_grid: np.ndarray) -> dict:
    counts = np.bincount(input_grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[background]) == 1
    colors = [int(color) for color in np.unique(input_grid)
              if int(color) != background]
    assert 2 <= len(colors) <= 9
    lines = [_infer_line(input_grid, color) for color in colors]
    assert len(set(lines)) == len(lines)
    height, width = input_grid.shape
    assert all(len(_line_cells(line, height, width)) >= 3
               for line in lines)
    members = _cell_members(lines, height, width)
    edges: set[tuple[int, int]] = set()
    for cell, indices in members.items():
        if len(indices) < 2:
            continue
        visible = int(input_grid[cell])
        assert visible in colors
        upper = colors.index(visible)
        assert upper in indices
        edges.update((index, upper) for index in indices if index != upper)
    order = _unique_topological_order(len(lines), edges)
    assert order is not None
    reconstruction = np.full(input_grid.shape, background, dtype=np.int16)
    for index in order:
        for row, col in _line_cells(lines[index], height, width):
            reconstruction[row, col] = colors[index]
    assert np.array_equal(reconstruction, input_grid)
    return {
        "background": background,
        "number": len(lines),
        "lines": lines,
        "order": [colors[index] for index in order],
        "orientation_counts": {orientation: sum(line[0] == orientation
                                                for line in lines)
                               for orientation in _ORIENTATIONS},
        "max_overlap": max(len(indices) for indices in members.values()),
    }


def _analyze_pair(pair: dict) -> dict:
    result = _solve_input(pair["input"])
    output = pair["output"]
    assert output.shape == (1, result["number"])
    assert [int(color) for color in output[0]] == result["order"]
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    all_pairs = puzzle["train"] + puzzle["test"]
    input_signatures: set[tuple] = set()
    pair_signatures: set[tuple] = set()
    analyses: list[dict] = []

    for pair in all_pairs:
        assert isinstance(pair, dict)
        assert set(pair) == {"input", "output"}

        _check_grid(pair["input"])
        _check_grid(pair["output"])

        input_signature = _array_signature(pair["input"])
        pair_signature = _pair_signature(pair)

        assert input_signature not in input_signatures
        assert pair_signature not in pair_signatures

        input_signatures.add(input_signature)
        pair_signatures.add(pair_signature)

        source_signature = (
            _grid_text(pair["input"]),
            _grid_text(pair["output"]),
        )
        assert source_signature not in _SOURCE_SIGNATURES

        analyses.append(_analyze_pair(pair))

    train_count = len(puzzle["train"])
    train_analyses = analyses[:train_count]
    # A direct two-line comparison demonstrates which end of the output
    # is earlier. Geometry and color identity are visible in every input.
    assert any(analysis["number"] == 2 for analysis in train_analyses)


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split], second[split]
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        _pair_signature(pair)
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    automatic = generate_puzzle(314159)
    explicit = generate_puzzle(271828, num_train=6, num_test=2)
    validate_generated_puzzle(automatic)
    validate_generated_puzzle(explicit)
    replay = generate_puzzle(271828, num_train=6, num_test=2)
    assert _puzzles_equal(explicit, replay)
    alternatives = [generate_puzzle(seed, 5, 2) for seed in range(900, 906)]
    assert len({_puzzle_signature(puzzle) for puzzle in alternatives}) >= 5
    legal_cases = [
        (11, 2, 1),
        (12, 10, 1),
        (13, 10, 4),
        (14, None, 1),
        (15, None, 4),
        (16, 2, None),
        (17, 10, None),
    ]
    for seed, train_count, test_count in legal_cases:
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    illegal_cases = [
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ]
    for arguments in illegal_cases:
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts did not raise ValueError")

    corpus = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in corpus}
    test_counts = {len(puzzle["test"]) for puzzle in corpus}
    backgrounds: set[int] = set()
    shapes: set[tuple[int, int]] = set()
    orientations: set[tuple[str, ...]] = set()
    counts_by_split = {"train": set(), "test": set()}
    overlaps: set[int] = set()
    parallels: set[int] = set()
    grid_signatures: set[tuple] = set()
    puzzle_signatures: set[tuple] = set()
    changing_background = stable_background = 0
    test_without_triples = heldout_slope = both_slopes_both = 0
    varied_counts = reversed_complexity = 0
    demonstration_positions: set[int] = set()
    total_grids = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        puzzle_signatures.add(_puzzle_signature(puzzle))
        train_analyses = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_analyses = [_analyze_pair(pair) for pair in puzzle["test"]]
        all_analyses = train_analyses + test_analyses
        distinct_backgrounds = {a["background"] for a in all_analyses}
        changing_background += len(distinct_backgrounds) > 1
        stable_background += len(distinct_backgrounds) == 1
        demonstration_positions.add(next(i for i, a in enumerate(train_analyses)
                                         if a["number"] == 2))
        varied_counts += len({a["number"] for a in all_analyses}) > 1
        reversed_complexity += max(a["number"] for a in train_analyses) > max(
            a["number"] for a in test_analyses)
        heldout_slope += all(a["orientation_counts"]["A"] == 0
                             for a in train_analyses) and any(
            a["orientation_counts"]["A"] > 0 for a in test_analyses)
        both_slopes_both += (any(all(a["orientation_counts"][o] > 0
                                     for o in ("D", "A")) for a in train_analyses)
                             and any(all(a["orientation_counts"][o] > 0
                                         for o in ("D", "A")) for a in test_analyses))
        test_without_triples += any(a["max_overlap"] == 2 for a in test_analyses)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair)
                _check_grid(pair["input"])
                _check_grid(pair["output"])
                assert _solve_input(pair["input"])["order"] == list(
                    int(value) for value in pair["output"][0])
                shapes.add(pair["input"].shape)
                counts_by_split[split].add(analysis["number"])
                orientations.add(tuple(o for o in _ORIENTATIONS
                                       if analysis["orientation_counts"][o]))
                overlaps.add(analysis["max_overlap"])
                parallels.add(max(analysis["orientation_counts"].values()))
                backgrounds.add(analysis["background"])
                grid_signatures.add(_array_signature(pair["input"]))
                total_grids += 1
                source_signature = (
                    _grid_text(pair["input"]),
                    _grid_text(pair["output"]),
                )
                assert source_signature not in _SOURCE_SIGNATURES
    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(backgrounds) >= 8
    assert len(shapes) >= 70
    assert counts_by_split["train"] == set(range(2, 10))
    assert counts_by_split["test"] == set(range(2, 10))
    assert len(orientations) >= 9
    assert {2, 3, 4} <= overlaps
    assert max(parallels) >= 4
    assert any(min(shape) <= 5 for shape in shapes)
    assert any(max(shape) >= 25 for shape in shapes)
    assert any(30 in shape for shape in shapes)
    assert any(max(shape) >= 2 * min(shape) for shape in shapes)
    assert changing_background >= 8 and stable_background >= 8
    assert test_without_triples >= 8
    assert heldout_slope >= 2 and both_slopes_both >= 2
    assert varied_counts >= 35 and reversed_complexity >= 5
    assert len(demonstration_positions) >= 3
    assert len(grid_signatures) >= int(0.95 * total_grids)
