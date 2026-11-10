from __future__ import annotations

import numpy as np


# Exact source inputs are retained only so generated puzzles can explicitly
# reject accidental copies.
_SOURCE_INPUT_ROWS = {
    (
        "5555555555",
        "5000005555",
        "5500005005",
        "5000000005",
        "5500000005",
        "5500000055",
        "5000000005",
        "5005500005",
        "5555505505",
        "5555555555",
    ),
    (
        "5555555555",
        "5550505005",
        "5500500005",
        "5000000005",
        "5500000005",
        "5550000055",
        "5000000005",
        "5000000555",
        "5505005555",
        "5555555555",
    ),
    (
        "5555555555",
        "5550005555",
        "5550000555",
        "5500000005",
        "5000000055",
        "5000000005",
        "5550550005",
        "5500550055",
        "5550555055",
        "5555555555",
    ),
    (
        "5555555555",
        "5000550055",
        "5500050005",
        "5550000055",
        "5500000005",
        "5000000055",
        "5500050005",
        "5005050005",
        "5505550505",
        "5555555555",
    ),
    (
        "5555555555",
        "5550000055",
        "5500000055",
        "5550000005",
        "5000000555",
        "5000005555",
        "5000000005",
        "5005500505",
        "5555555505",
        "5555555555",
    ),
}


def _checked_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_optional_count("num_train", num_train, 2, 10)
    num_test = _checked_optional_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity across consecutive
        # seeds while remaining entirely deterministic.
        num_test = 1 + seed % 4
        first_legal_train = num_test + 1
        span = 10 - first_legal_train + 1
        num_train = first_legal_train + (seed // 4) % span
        return num_train, num_test

    if num_train is None:
        choices = np.arange(num_test + 1, 11, dtype=np.int64)
        num_train = int(choices[int(rng.integers(0, len(choices)))])
    else:
        largest_test = min(4, num_train - 1)
        choices = np.arange(1, largest_test + 1, dtype=np.int64)
        num_test = int(choices[int(rng.integers(0, len(choices)))])

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict[str, int]:
    palette = rng.permutation(10)[:3]
    return {
        "wall": int(palette[0]),
        "open": int(palette[1]),
        "highlight": int(palette[2]),
    }


def _count_open_runs(mask: np.ndarray) -> int:
    return sum(
        bool(value) and (index == 0 or not bool(mask[index - 1]))
        for index, value in enumerate(mask)
    )


def _candidate_specs(
    grid: np.ndarray,
    wall: int,
    open_color: int,
) -> list[tuple[int, int]]:
    """Return (start_column, horizontal_step) candidate descriptions."""
    specs: list[tuple[int, int]] = []

    for column in range(1, grid.shape[1] - 1):
        if int(grid[1, column]) != open_color:
            continue

        # A wall on the left launches a down-right ray.
        if int(grid[1, column - 1]) == wall:
            specs.append((column, 1))

        # A wall on the right launches a down-left ray.
        if int(grid[1, column + 1]) == wall:
            specs.append((column, -1))

    return specs


def _analyze_rays(
    grid: np.ndarray,
    wall: int,
    open_color: int,
) -> dict:
    height, width = grid.shape
    candidates = []

    for start, direction in _candidate_specs(grid, wall, open_color):
        cells: list[tuple[int, int]] = []
        row = 1
        column = start

        while (
            0 <= row < height
            and 0 <= column < width
            and int(grid[row, column]) == open_color
        ):
            cells.append((row, column))
            row += 1
            column += direction

        # The enclosing structural frame ensures that every ray collides
        # with structure before leaving the array.
        assert 0 <= row < height
        assert 0 <= column < width
        assert int(grid[row, column]) == wall

        candidates.append(
            {
                "start": start,
                "direction": direction,
                "cells": tuple(cells),
            }
        )

    assert candidates

    longest = max(len(candidate["cells"]) for candidate in candidates)
    winners = [
        candidate
        for candidate in candidates
        if len(candidate["cells"]) == longest
    ]

    return {
        "candidates": candidates,
        "winners": winners,
        "longest": longest,
    }


def _solve_visible_input(
    grid: np.ndarray,
    wall: int,
    open_color: int,
    highlight: int,
) -> tuple[np.ndarray, dict]:
    analysis = _analyze_rays(grid, wall, open_color)
    assert len(analysis["winners"]) == 1

    output = grid.copy()
    for row, column in analysis["winners"][0]["cells"]:
        output[row, column] = highlight

    return output, analysis


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _is_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (10, 10):
        return False

    rows = tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )
    return rows in _SOURCE_INPUT_ROWS


def _components(grid: np.ndarray, color: int) -> int:
    remaining = {tuple(map(int, point)) for point in np.argwhere(grid == color)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            row, column = stack.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return components


def _detached_wall_size(grid: np.ndarray, wall: int) -> int:
    attached = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        row, column = stack.pop()
        for nr, nc in ((row - 1, column), (row + 1, column),
                       (row, column - 1), (row, column + 1)):
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                    and grid[nr, nc] == wall and (nr, nc) not in attached):
                attached.add((nr, nc))
                stack.append((nr, nc))
    return int(np.count_nonzero(grid == wall)) - len(attached)


def _stop_kind(grid: np.ndarray, candidate: dict) -> str:
    row, column = candidate["cells"][-1]
    row += 1
    column += candidate["direction"]
    if row == grid.shape[0] - 1:
        return "bottom"
    if column in (0, grid.shape[1] - 1):
        return "side"
    return "interior"


def _top_mask(
    rng: np.random.Generator, width: int, start: int,
    direction: int, run_count: int,
) -> np.ndarray | None:
    # Build the selected opening first, then independent separated openings.
    for _ in range(30):
        maximum_width = min(width - 2 - start + 1 if direction == 1 else start,
                            width - 3 if run_count == 1 else max(2, width // 3))
        if maximum_width < 2:
            return None
        minimum_width = min(maximum_width, 3 if run_count == 1 else 2)
        run_width = int(rng.integers(minimum_width, maximum_width + 1))
        left = start if direction == 1 else start - run_width + 1
        right = left + run_width - 1
        mask = np.zeros(width, dtype=bool)
        mask[left:right + 1] = True
        forbidden = mask.copy()
        forbidden[max(1, left - 1)] = True
        forbidden[min(width - 2, right + 1)] = True
        possible = [c for c in range(1, width - 2)
                    if not forbidden[c] and not forbidden[c + 1]]
        rng.shuffle(possible)
        for column in possible:
            if _count_open_runs(mask[1:-1]) >= run_count:
                break
            if mask[column] or mask[column + 1]:
                continue
            if mask[column - 1] or mask[column + 2]:
                continue
            length = int(rng.integers(2, min(4, width - 1 - column) + 1))
            if column + length >= width - 1 or np.any(mask[column - 1:column + length + 1]):
                length = 2
            if column + length < width - 1 and not np.any(mask[column - 1:column + length + 1]):
                mask[column:column + length] = True
        if _count_open_runs(mask[1:-1]) == run_count:
            return mask
    return None


def _construct_pair(
    rng: np.random.Generator, height: int, width: int,
    direction: int, stop: str, run_count: int,
    wall: int, open_color: int, highlight: int, detached: bool,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    assert direction in (-1, 1)
    assert stop in ("bottom", "side", "interior")
    # A start must have an inward neighbor in its opening.
    starts = range(1, width - 3) if direction == 1 else range(3, width - 1)
    possibilities = []
    for start in starts:
        frame_length = min(height - 2, width - 1 - start if direction == 1 else start)
        side_length = width - 1 - start if direction == 1 else start
        if stop == "bottom" and frame_length == height - 2 and side_length >= height - 2:
            possibilities.append((start, frame_length))
        elif stop == "side" and side_length < height - 2 and side_length >= max(3, min(height, width) // 3):
            possibilities.append((start, side_length))
        elif stop == "interior" and frame_length >= max(5, min(height, width) // 3 + 2):
            possibilities.append((start, frame_length))
    if not possibilities:
        return None
    start, frame_length = possibilities[int(rng.integers(len(possibilities)))]
    if stop == "interior":
        length = int(rng.integers(max(3, min(height, width) // 3), frame_length))
    else:
        length = frame_length
    mask = _top_mask(rng, width, start, direction, run_count)
    if mask is None:
        return None
    grid = np.full((height, width), open_color, dtype=np.int8)
    grid[0, :] = wall
    grid[-1, :] = wall
    grid[:, 0] = wall
    grid[:, -1] = wall
    grid[1, 1:-1] = np.where(mask[1:-1], open_color, wall)
    reserved = {(1 + step, start + direction * step) for step in range(length)}

    def paint(row: int, column: int) -> None:
        if (row, column) not in reserved:
            grid[row, column] = wall

    # Smooth but changing side profiles make steps and pockets attached to
    # the existing frame. They also give genuine first-wall collisions.
    left_depth = int(rng.integers(0, 3))
    right_depth = int(rng.integers(0, 3))
    maximum_depth = max(2, min(5, width // 3))
    for row in range(2, height - 1):
        left_depth = max(0, min(maximum_depth, left_depth + int(rng.integers(-1, 2))))
        right_depth = max(0, min(maximum_depth, right_depth + int(rng.integers(-1, 2))))
        if rng.random() < 0.16:
            left_depth = int(rng.integers(0, maximum_depth + 1))
        if rng.random() < 0.16:
            right_depth = int(rng.integers(0, maximum_depth + 1))
        for column in range(1, left_depth + 1):
            paint(row, column)
        for column in range(width - 1 - right_depth, width - 1):
            paint(row, column)

    # Top teeth and ledges grow from wall cells that are visibly anchored
    # to the frame; bottom teeth add a second source of pockets.
    for column in range(1, width - 1):
        if not mask[column] and rng.random() < 0.68:
            depth = int(rng.integers(1, min(5, height - 2)))
            for row in range(2, depth + 2):
                paint(row, column)
        if rng.random() < 0.18:
            depth = int(rng.integers(1, min(4, height - 2)))
            for row in range(height - 2, height - depth - 2, -1):
                paint(row, column)
    for _ in range(int(rng.integers(1, max(3, height // 4)))):
        row = int(rng.integers(3, height - 1))
        extent = int(rng.integers(2, max(3, width // 2 + 1)))
        if rng.random() < 0.5:
            for column in range(1, min(extent + 1, width - 1)):
                paint(row, column)
        else:
            for column in range(max(1, width - 1 - extent), width - 1):
                paint(row, column)

    if stop == "interior":
        row = 1 + length
        column = start + direction * length
        if direction == 1:
            grid[row, column:width - 1] = wall
        else:
            grid[row, 1:column + 1] = wall

    if detached:
        places = []
        for row in range(3, height - 2):
            for column in range(3, width - 2):
                if (row, column) in reserved or grid[row, column] != open_color:
                    continue
                if all(grid[neighbor] == open_color for neighbor in
                       ((row - 1, column), (row + 1, column),
                        (row, column - 1), (row, column + 1))):
                    places.append((row, column))
        if not places:
            return None
        grid[places[int(rng.integers(len(places)))]] = wall

    # Validity is decided from the final visible grid, not construction
    # targets. The open region and wall remain four-connected.
    if _components(grid, wall) != (2 if detached else 1) or _components(grid, open_color) != 1:
        return None
    analysis = _analyze_rays(grid, wall, open_color)
    if len(analysis["winners"]) != 1:
        return None
    winner = analysis["winners"][0]
    if (winner["start"], winner["direction"]) != (start, direction):
        return None
    if len(winner["cells"]) != length or _stop_kind(grid, winner) != stop:
        return None
    lengths = sorted(len(candidate["cells"]) for candidate in analysis["candidates"])
    if len(lengths) < 2 or lengths[-2] < 2:
        return None
    output = grid.copy()
    for cell in winner["cells"]:
        output[cell] = highlight
    return grid, output, {
        "direction": direction, "length": length, "stop": stop,
        "run_count": run_count, "start": start, "candidate_count": len(lengths),
        "gap": lengths[-1] - lengths[-2], "shape": grid.shape,
    }


def _pair_palette(rng: np.random.Generator, highlight: int,
                  base_wall: int, base_open: int, varied: bool) -> tuple[int, int]:
    if not varied or rng.random() < 0.25:
        return base_wall, base_open
    choices = [color for color in range(10) if color != highlight]
    selected = rng.choice(choices, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _fresh_pair(
    rng: np.random.Generator, direction: int, stop: str, runs: int,
    highlight: int, base_wall: int, base_open: int, varied_palette: bool,
    seen: set[tuple], source_scale: bool,
) -> tuple[dict, dict]:
    if source_scale and runs == 3:
        runs = 2
    detached = bool(rng.random() < 0.13) and not source_scale
    for attempt in range(900):
        if source_scale:
            height, width = 10, 10
        else:
            height = int(rng.integers(9, 21))
            width = int(rng.integers(13 if runs == 3 else 9, 21))
            if stop == "bottom" and width < height:
                height, width = width, height
            if stop == "side" and height < width:
                height, width = width, height
            if (height, width) == (10, 10):
                continue
        wall, open_color = _pair_palette(rng, highlight, base_wall, base_open, varied_palette)
        proposal = _construct_pair(rng, height, width, direction, stop, runs,
                                   wall, open_color, highlight, detached)
        if proposal is None:
            continue
        input_grid, output_grid, info = proposal
        key = _grid_key(input_grid)
        if key in seen or _is_source_input(input_grid):
            continue
        seen.add(key)
        return {"input": input_grid, "output": output_grid}, info
    raise RuntimeError(f"could not construct pair: {direction}, {stop}, {runs}")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, int(seed), num_train, num_test)
    palette = rng.permutation(10)
    base_wall, base_open, highlight = map(int, palette[:3])
    varied_palette = bool(rng.random() < 0.45)
    seen: set[tuple] = set()
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=num_train + num_test, dtype=np.uint64)
    # Both slopes are demonstrated even at the minimum training count.
    directions = [-1, 1]
    directions += [int(rng.choice((-1, 1))) for _ in range(num_train - 2)]
    rng.shuffle(directions)
    stops = ["bottom", "side", "interior"]
    train, test = [], []
    for index, direction in enumerate(directions):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        stop = stops[index % 3] if num_train >= 3 else ("interior", "bottom")[index]
        runs = 1 if index == 0 else int(pair_rng.choice((1, 2, 3), p=(0.45, 0.4, 0.15)))
        source_scale = index == 0 or (index > 1 and pair_rng.random() < 0.27)
        pair, _ = _fresh_pair(pair_rng, direction, stop, runs, highlight,
                              base_wall, base_open, varied_palette, seen, source_scale)
        train.append(pair)
    for index in range(num_test):
        pair_rng = np.random.default_rng(int(pair_seeds[num_train + index]))
        direction = int(pair_rng.choice((-1, 1)))
        stop = str(pair_rng.choice(stops))
        runs = int(pair_rng.choice((1, 2, 3), p=(0.36, 0.44, 0.20)))
        source_scale = bool(pair_rng.random() < 0.31)
        pair, _ = _fresh_pair(pair_rng, direction, stop, runs, highlight,
                              base_wall, base_open, varied_palette, seen, source_scale)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_raw_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _inspect_pair(pair: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape and grid.dtype == output.dtype
    height, width = grid.shape
    assert height >= 9 and width >= 9
    colors = set(map(int, np.unique(grid)))
    assert len(colors) == 2
    wall = int(grid[0, 0])
    assert wall in colors
    open_color = next(iter(colors - {wall}))
    assert np.all(grid[0] == wall) and np.all(grid[-1] == wall)
    assert np.all(grid[:, 0] == wall) and np.all(grid[:, -1] == wall)
    assert _detached_wall_size(grid, wall) in (0, 1)
    assert _components(grid, open_color) == 1
    changed = grid != output
    assert np.any(changed) and np.all(grid[changed] == open_color)
    new_colors = set(map(int, np.unique(output[changed])))
    assert len(new_colors) == 1
    highlight = next(iter(new_colors))
    assert highlight not in colors
    assert np.array_equal(grid[~changed], output[~changed])
    expected, analysis = _solve_visible_input(grid, wall, open_color, highlight)
    assert np.array_equal(expected, output)
    lengths = sorted(len(candidate["cells"]) for candidate in analysis["candidates"])
    assert lengths[-2] >= 2
    winner = analysis["winners"][0]
    return {
        "highlight": highlight, "wall": wall, "open": open_color,
        "direction": winner["direction"], "start": winner["start"],
        "length": len(winner["cells"]), "stop": _stop_kind(grid, winner),
        "runs": len(analysis["candidates"]) // 2,
        "shape": grid.shape, "gap": lengths[-1] - lengths[-2],
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _validate_raw_grid(pair["input"])
            _validate_raw_grid(pair["output"])
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _is_source_input(pair["input"])
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[split].append(_inspect_pair(pair))
    highlights = {info["highlight"] for split in infos for info in infos[split]}
    assert len(highlights) == 1
    learned = infos["train"][0]["highlight"]
    assert {info["direction"] for info in infos["train"]} == {-1, 1}
    assert any(info["runs"] == 1 for info in infos["train"])
    assert any(info["stop"] == "interior" for info in infos["train"])
    assert any(info["stop"] in ("bottom", "side") for info in infos["train"])
    # Prediction uses only the training-learned highlight and visible input.
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid = pair["input"]
            wall = int(grid[0, 0])
            open_color = next(int(c) for c in np.unique(grid) if int(c) != wall)
            prediction, _ = _solve_visible_input(grid, wall, open_color, learned)
            assert np.array_equal(prediction, pair["output"])


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[name], pb[name]) for pa, pb in zip(a[split], b[split])
        for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(314160, 7, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4),
                                                       (None, 1), (None, 4),
                                                       (2, None), (10, None))):
        puzzle = generate_puzzle(7000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    counts, shapes, starts, lengths, stops, slopes, runs, gaps = [set() for _ in range(8)]
    layouts = set()
    detached_count = source_seven = source_eight = close_gaps = 0
    varied_palette_puzzles = mixed_test_puzzles = reused_start_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        train_infos = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_infos = [_inspect_pair(pair) for pair in puzzle["test"]]
        varied_palette_puzzles += len({(i["wall"], i["open"]) for i in train_infos}) > 1
        mixed_test_puzzles += len({i["direction"] for i in test_infos}) > 1
        reused_start_puzzles += any(i["start"] in {j["start"] for j in train_infos}
                                    for i in test_infos)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair)
                shapes.add(info["shape"])
                starts.add(info["start"])
                lengths.add(info["length"])
                stops.add(info["stop"])
                slopes.add(info["direction"])
                runs.add(info["runs"])
                gaps.add(info["gap"])
                layouts.add(_grid_key(pair["input"]))
                detached_count += _detached_wall_size(pair["input"], info["wall"]) == 1
                close_gaps += info["gap"] == 1
                source_seven += info["shape"] == (10, 10) and info["length"] == 7
                source_eight += info["shape"] == (10, 10) and info["length"] == 8
                if min(info["shape"]) >= 16:
                    assert info["length"] > 4
    assert len(counts) >= 8 and len(shapes) >= 25 and len(layouts) >= 150
    assert len(starts) >= 8 and len(lengths) >= 6
    assert stops == {"bottom", "side", "interior"}
    assert slopes == {-1, 1} and {1, 2}.issubset(runs) and len(gaps) >= 2
    assert 5 <= detached_count <= 80 and 10 <= close_gaps
    assert source_seven >= 5 and source_eight >= 20
    assert varied_palette_puzzles >= 8
    assert mixed_test_puzzles >= 8 and reused_start_puzzles >= 8
