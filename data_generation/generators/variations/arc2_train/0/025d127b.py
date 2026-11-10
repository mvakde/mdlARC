import numpy as np


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(v) for v in grid.shape),
        tuple(int(v) for v in grid.reshape(-1)),
    )


def _rows_signature(rows: tuple[str, ...]) -> tuple:
    width = len(rows[0])
    assert all(len(row) == width for row in rows)
    return (
        (len(rows), width),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_ROW_PAIRS = (
    (
        (
            "000000000",
            "088888000",
            "080000800",
            "008000080",
            "000800008",
            "000088888",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "008888800",
            "008000080",
            "000800008",
            "000080008",
            "000088888",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "066600000",
            "060060000",
            "006006000",
            "000600600",
            "000066600",
            "000000000",
            "002220000",
            "002002000",
            "000222000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "006660000",
            "006006000",
            "000600600",
            "000060600",
            "000066600",
            "000000000",
            "000222000",
            "000202000",
            "000222000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0444444000",
            "0400000400",
            "0040000040",
            "0004000004",
            "0000444444",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0044444400",
            "0040000040",
            "0004000004",
            "0000400004",
            "0000444444",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
)

_SOURCE_PAIR_SIGNATURES = frozenset(
    (_rows_signature(inp), _rows_signature(out))
    for inp, out in _SOURCE_ROW_PAIRS
)


def _checked_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if name == "num_train":
        if not 2 <= value <= 10:
            raise ValueError("num_train must be between 2 and 10")
    else:
        if not 1 <= value <= 4:
            raise ValueError("num_test must be between 1 and 4")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(num_train, "num_train")
    test_count = _checked_count(num_test, "num_test")

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        lowest_train = test_count + 1
        span = 11 - lowest_train
        train_count = lowest_train + (seed // 4) % span
    elif train_count is None:
        assert test_count is not None
        lowest_train = max(2, test_count + 1)
        train_count = lowest_train + seed % (11 - lowest_train)
    else:
        largest_test = min(4, train_count - 1)
        test_count = 1 + seed % largest_test

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count
    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {
        "direction_mode": ("left", "right", "mixed")[
            int(rng.choice(3, p=[0.24, 0.24, 0.52]))
        ],
        "size_mode": ("extrapolate", "overlap", "recombine")[
            int(rng.choice(3, p=[0.30, 0.35, 0.35]))
        ],
    }


def _build_object(
    rng: np.random.Generator,
    height: int,
    cap_length: int,
    direction: int,
    color: int,
    residual: bool = False,
) -> dict:
    assert height >= 3 and cap_length >= 3 and direction in (-1, 1)
    minimum_offset = 2 if residual else 1
    offset = int(rng.integers(minimum_offset, height - 1))
    steps = np.zeros(height - 1, dtype=np.int8)
    steps[-1] = 1
    if offset > 1:
        chosen = rng.choice(
            np.arange(1, height - 2), size=offset - 1, replace=False
        )
        steps[chosen] = 1
    starts = np.concatenate((np.array([0]), np.cumsum(steps)))
    cells = [(0, x) for x in range(cap_length)]
    for y in range(1, height - 1):
        x = int(starts[y])
        cells.extend(((y, x), (y, x + cap_length)))
    cells.extend((height - 1, x) for x in range(offset, offset + cap_length))
    width = cap_length + offset
    if direction == -1:
        cells = [(y, width - 1 - x) for y, x in cells]
    return {
        "height": height,
        "width": width,
        "cap_length": cap_length,
        "offset": offset,
        "direction": direction,
        "color": color,
        "cells": tuple(sorted(cells)),
    }


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _positions(rng: np.random.Generator, objects: list[dict]) -> list[tuple[int, int]] | None:
    count = len(objects)
    layout = ("row", "column", "scatter")[int(rng.choice(3, p=[0.23, 0.23, 0.54]))]
    if count == 1:
        return [(0, 0)]
    for preferred in (layout, "scatter", "column", "row"):
        if preferred == "row":
            gaps = [int(rng.integers(2, 6)) for _ in range(count - 1)]
            if sum(obj["width"] for obj in objects) + sum(gaps) > 30:
                continue
            base_y = int(rng.integers(0, 31 - max(obj["height"] for obj in objects)))
            x = 0
            result = []
            for index, obj in enumerate(objects):
                y = min(30 - obj["height"], max(0, base_y + int(rng.integers(-3, 4))))
                result.append((y, x))
                x += obj["width"] + (gaps[index] if index < count - 1 else 0)
            return result
        if preferred == "column":
            gaps = [int(rng.integers(2, 6)) for _ in range(count - 1)]
            if sum(obj["height"] for obj in objects) + sum(gaps) > 30:
                continue
            base_x = int(rng.integers(0, 31 - max(obj["width"] for obj in objects)))
            y = 0
            result = []
            for index, obj in enumerate(objects):
                x = min(30 - obj["width"], max(0, base_x + int(rng.integers(-4, 5))))
                result.append((y, x))
                y += obj["height"] + (gaps[index] if index < count - 1 else 0)
            return result
        for _ in range(12):
            result = []
            boxes = []
            for obj in objects:
                placed = False
                for _ in range(80):
                    y = int(rng.integers(0, 31 - obj["height"]))
                    x = int(rng.integers(0, 31 - obj["width"]))
                    box = (y, x, y + obj["height"] - 1, x + obj["width"] - 1)
                    if all(_separated(box, other) for other in boxes):
                        result.append((y, x))
                        boxes.append(box)
                        placed = True
                        break
                if not placed:
                    break
            if len(result) == count:
                return result
    return None


def _margins(rng: np.random.Generator, extent: int) -> tuple[int, int]:
    spare = 30 - extent
    if spare == 0:
        return 0, 0
    if rng.random() < 0.13:
        left = int(rng.integers(0, spare + 1))
        return left, spare - left
    choices = (0, 0, 0, 1, 2, 3, 4, 6)
    first = min(spare, int(rng.choice(choices)))
    second = min(spare - first, int(rng.choice(choices)))
    return first, second


def _place_and_render(
    rng: np.random.Generator, objects: list[dict], background: int
) -> tuple[np.ndarray, np.ndarray] | None:
    positions = _positions(rng, objects)
    if positions is None:
        return None
    min_y = min(y for y, _ in positions)
    min_x = min(x for _, x in positions)
    max_y = max(y + obj["height"] for obj, (y, _) in zip(objects, positions))
    max_x = max(x + obj["width"] for obj, (_, x) in zip(objects, positions))
    top, bottom = _margins(rng, max_y - min_y)
    left, right = _margins(rng, max_x - min_x)
    height = top + max_y - min_y + bottom
    width = left + max_x - min_x + right
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full_like(input_grid, background)
    for obj, (origin_y, origin_x) in zip(objects, positions):
        origin_y += top - min_y
        origin_x += left - min_x
        for local_y, local_x in obj["cells"]:
            y, x = origin_y + local_y, origin_x + local_x
            assert input_grid[y, x] == background
            input_grid[y, x] = obj["color"]
            fixed = local_y == obj["height"] - 1 or local_x == (
                obj["width"] - 1 if obj["direction"] == 1 else 0
            )
            target_x = x if fixed else x + obj["direction"]
            assert output_grid[y, target_x] == background
            output_grid[y, target_x] = obj["color"]
    if int(np.count_nonzero(input_grid == background)) * 2 <= input_grid.size:
        return None
    return input_grid, output_grid


def _choose_colors(
    rng: np.random.Generator,
    count: int,
    background: int,
    familiar: set[int],
    is_test: bool,
) -> list[int]:
    available = [c for c in range(10) if c != background]
    colors = []
    for index in range(count):
        if index and rng.random() < 0.39:
            colors.append(colors[int(rng.integers(0, len(colors)))])
            continue
        choices = [c for c in available if c not in colors] or available
        if is_test and familiar and rng.random() < 0.52:
            known = [c for c in choices if c in familiar]
            if known:
                choices = known
        if background != 0 and 0 in choices and rng.random() < 0.24:
            colors.append(0)
        else:
            colors.append(int(rng.choice(choices)))
    return colors


def _sample_pair(
    rng: np.random.Generator,
    parameters: dict,
    is_test: bool,
    object_count: int,
    background: int,
    familiar_caps: tuple[int, ...],
    familiar_colors: set[int],
    force_residual: bool = False,
    force_mixed: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    size_mode = parameters["size_mode"]
    colors = _choose_colors(rng, object_count, background, familiar_colors, is_test)
    directions = []
    for index in range(object_count):
        if force_mixed and index < 2:
            direction = 1 if index == 0 else -1
        elif parameters["direction_mode"] == "left":
            direction = -1
        elif parameters["direction_mode"] == "right":
            direction = 1
        else:
            direction = 1 if rng.random() < 0.5 else -1
        directions.append(direction)
    objects = []
    for index in range(object_count):
        cap_limit = (11, 9, 7)[object_count - 1]
        height_limit = (12, 11, 8)[object_count - 1]
        if size_mode == "extrapolate":
            cap_low, cap_high = (7, cap_limit) if is_test else (3, min(6, cap_limit))
        else:
            cap_low, cap_high = 3, cap_limit
        if is_test and size_mode == "recombine" and familiar_caps and rng.random() < 0.72:
            usable = [c for c in familiar_caps if cap_low <= c <= cap_high]
            cap = int(rng.choice(usable)) if usable else int(rng.integers(cap_low, cap_high + 1))
        else:
            cap = int(rng.integers(cap_low, cap_high + 1))
        residual = force_residual and index == 0
        height_low = 4 if residual else 3
        height = int(rng.integers(height_low, height_limit + 1))
        objects.append(_build_object(rng, height, cap, directions[index], colors[index], residual))
    return _place_and_render(rng, objects, background)


def _analyze_input(grid: np.ndarray) -> list[dict]:
    _validate_grid(grid)
    counts = np.bincount(grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert int(counts[background]) * 2 > grid.size
    foreground = grid != background
    assert np.any(foreground)
    height, width = grid.shape
    visited = np.zeros_like(foreground, dtype=bool)
    analyses = []
    for start_y, start_x in np.argwhere(foreground):
        start_y, start_x = int(start_y), int(start_x)
        if visited[start_y, start_x]:
            continue
        color = int(grid[start_y, start_x])
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        coords_list = []
        while queue:
            y, x = queue.pop()
            coords_list.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        coords = np.asarray(coords_list, dtype=np.int64)
        assert np.all(grid[coords[:, 0], coords[:, 1]] == color)
        ymin, ymax = int(coords[:, 0].min()), int(coords[:, 0].max())
        object_height = ymax - ymin + 1
        assert 3 <= object_height <= 12
        rows = [np.sort(coords[coords[:, 0] == y, 1]) for y in range(ymin, ymax + 1)]
        assert all(len(xs) for xs in rows)
        top, bottom = rows[0], rows[-1]
        cap_length = len(top)
        assert 3 <= cap_length <= 11
        assert len(bottom) == cap_length
        assert np.all(np.diff(top) == 1) and np.all(np.diff(bottom) == 1)
        displacement = int(bottom[0] - top[0])
        assert displacement != 0
        direction = 1 if displacement > 0 else -1
        offset = abs(displacement)
        assert 1 <= offset <= object_height - 2
        starts = []
        for row_index, xs in enumerate(rows):
            oriented = np.sort(direction * xs)
            starts.append(int(oriented[0]))
            if row_index not in (0, object_height - 1):
                assert len(xs) == 2
                assert int(oriented[1] - oriented[0]) == cap_length
        steps = np.diff(np.asarray(starts))
        assert np.all((steps == 0) | (steps == 1))
        assert int(steps[0]) == 0 and int(steps[-1]) == 1
        assert int(starts[-1] - starts[0]) == offset
        far_x = int(coords[:, 1].max() if direction == 1 else coords[:, 1].min())
        assert np.any((coords[:, 1] == far_x) & (coords[:, 0] < ymax))
        analyses.append({
            "color": color, "coords": coords, "direction": direction,
            "height": object_height, "cap_length": cap_length, "offset": offset,
            "steps": tuple(int(step) for step in steps),
            "ymin": ymin, "ymax": ymax, "far_x": far_x,
            "bbox": (ymin, int(coords[:, 1].min()), ymax, int(coords[:, 1].max())),
        })
    assert 1 <= len(analyses) <= 3
    for first in range(len(analyses)):
        for second in range(first + 1, len(analyses)):
            assert _separated(analyses[first]["bbox"], analyses[second]["bbox"])
    return analyses


def _derive_output_from_input(input_grid: np.ndarray, analyses: list[dict] | None = None) -> np.ndarray:
    if analyses is None:
        analyses = _analyze_input(input_grid)
    counts = np.bincount(input_grid.reshape(-1).astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    expected = np.full_like(input_grid, background)
    targets = set()
    for info in analyses:
        for y_raw, x_raw in info["coords"]:
            y, x = int(y_raw), int(x_raw)
            fixed = y == info["ymax"] or x == info["far_x"]
            target_x = x if fixed else x + info["direction"]
            assert 0 <= target_x < input_grid.shape[1]
            assert (y, target_x) not in targets
            targets.add((y, target_x))
            expected[y, target_x] = info["color"]
    return expected


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_offsets, test_offsets = [], []
    train_directions, test_directions = set(), set()
    train_counts, test_counts = [], []
    train_backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            signature = (_grid_signature(inp), _grid_signature(out))
            assert signature[0] not in seen_inputs and signature not in seen_pairs
            assert signature not in _SOURCE_PAIR_SIGNATURES
            seen_inputs.add(signature[0])
            seen_pairs.add(signature)
            analyses = _analyze_input(inp)
            assert np.array_equal(out, _derive_output_from_input(inp, analyses))
            assert np.array_equal(np.unique(inp), np.unique(out))
            background = int(np.argmax(np.bincount(inp.reshape(-1).astype(np.int64), minlength=10)))
            for color in np.unique(inp):
                if int(color) != background:
                    assert int(np.sum(inp == color)) == int(np.sum(out == color))
            if split == "train":
                train_backgrounds.add(background)
                train_counts.append(len(analyses))
                train_offsets.extend(info["offset"] for info in analyses)
                train_directions.update(info["direction"] for info in analyses)
            else:
                test_counts.append(len(analyses))
                test_offsets.extend(info["offset"] for info in analyses)
                test_directions.update(info["direction"] for info in analyses)
    assert len(train_backgrounds) >= 2
    assert max(train_offsets) >= 2
    if max(test_counts) >= 2:
        assert max(train_counts) >= 2
    assert test_directions.issubset(train_directions)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    parameters = _sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_pairs = set()
    evidence_index = int(rng.integers(0, train_count))
    multi_index = int(rng.integers(0, train_count))
    first_background = int(rng.integers(0, 10))
    other_background = int(rng.choice([c for c in range(10) if c != first_background]))
    train_backgrounds = rng.permutation([first_background, other_background]).tolist()
    familiar_caps = []
    familiar_colors = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            accepted = False
            if split == "train" and index < 2:
                background = int(train_backgrounds[index])
            else:
                background = int(rng.integers(0, 10))
            for attempt in range(80):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                if split == "train" and index == multi_index:
                    object_count = 2 if pair_rng.random() < 0.7 else 3
                else:
                    object_count = int(pair_rng.choice((1, 2, 3), p=(0.48, 0.36, 0.16)))
                if split == "train" and index == evidence_index:
                    force_residual = True
                else:
                    force_residual = False
                force_mixed = (split == "train" and index == multi_index and parameters["direction_mode"] == "mixed")
                candidate = _sample_pair(
                    pair_rng, parameters, split == "test", object_count,
                    background, tuple(familiar_caps), familiar_colors,
                    force_residual, force_mixed,
                )
                if candidate is None:
                    continue
                inp, out = candidate
                signature = (_grid_signature(inp), _grid_signature(out))
                if signature[0] in used_inputs or signature in used_pairs or signature in _SOURCE_PAIR_SIGNATURES:
                    continue
                analyses = _analyze_input(inp)
                if not np.array_equal(out, _derive_output_from_input(inp, analyses)):
                    continue
                puzzle[split].append({"input": inp, "output": out})
                used_inputs.add(signature[0])
                used_pairs.add(signature)
                if split == "train":
                    familiar_caps.extend(info["cap_length"] for info in analyses)
                    familiar_colors.update(info["color"] for info in analyses)
                accepted = True
                break
            if not accepted:
                raise ValueError("could not construct enough distinct valid pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_signature(p["input"]), _grid_signature(p["output"])) for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) >= 40
    backgrounds, object_colors, caps, heights, offsets, directions = set(), set(), set(), set(), set(), set()
    test_counts, train_counts, shapes = set(), set(), set()
    step_profiles = set()
    same_color_pairs = mixed_direction_pairs = edge_pairs = full_size_pairs = 0
    uniform_puzzles = mixed_puzzles = 0
    overlap_puzzles = extrapolation_puzzles = smaller_test_puzzles = 0
    multi_test_pairs = varied_test_sizes = varied_test_offsets = 0
    diagonal_test_layouts = same_color_mixed_test_pairs = 0
    zero_object_pairs = small_test_objects = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_caps = [i["cap_length"] for pair in puzzle["train"] for i in _analyze_input(pair["input"])]
        test_caps = [i["cap_length"] for pair in puzzle["test"] for i in _analyze_input(pair["input"])]
        puzzle_directions = {
            i["direction"]
            for split in ("train", "test")
            for pair in puzzle[split]
            for i in _analyze_input(pair["input"])
        }
        uniform_puzzles += len(puzzle_directions) == 1
        mixed_puzzles += len(puzzle_directions) == 2
        overlap_puzzles += bool(set(train_caps) & set(test_caps))
        extrapolation_puzzles += min(test_caps) > max(train_caps)
        smaller_test_puzzles += min(test_caps) < max(train_caps)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze_input(inp)
                assert np.array_equal(out, _derive_output_from_input(inp, info))
                background = int(np.argmax(np.bincount(inp.reshape(-1).astype(np.int64), minlength=10)))
                backgrounds.add(background)
                object_colors.update(i["color"] for i in info)
                caps.update(i["cap_length"] for i in info)
                heights.update(i["height"] for i in info)
                offsets.update(i["offset"] for i in info)
                directions.update(i["direction"] for i in info)
                step_profiles.update((i["height"], i["offset"], i["steps"]) for i in info)
                shapes.add(inp.shape)
                (train_counts if split == "train" else test_counts).add(len(info))
                same_color_pairs += len({i["color"] for i in info}) < len(info)
                mixed_direction_pairs += len({i["direction"] for i in info}) > 1
                zero_object_pairs += any(i["color"] == 0 for i in info)
                edge_pairs += any(np.any(inp[edge] != background) for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]))
                full_size_pairs += 30 in inp.shape
                if split == "test":
                    small_test_objects += sum(i["height"] == 3 for i in info)
                    if len(info) >= 2:
                        multi_test_pairs += 1
                        varied_test_sizes += len({(i["height"], i["cap_length"]) for i in info}) > 1
                        varied_test_offsets += len({i["offset"] for i in info}) > 1
                        same_color_mixed_test_pairs += (
                            len({i["color"] for i in info}) < len(info)
                            and len({i["direction"] for i in info}) > 1
                        )
                        boxes = [i["bbox"] for i in info]
                        diagonal_test_layouts += any(
                            (a[2] < b[0] or b[2] < a[0])
                            and (a[3] < b[1] or b[3] < a[1])
                            for index, a in enumerate(boxes)
                            for b in boxes[index + 1:]
                        )
    assert len(backgrounds) >= 8 and 0 in object_colors
    assert max(caps) >= 9 and max(heights) >= 9 and {1, 2}.issubset(offsets)
    assert directions == {-1, 1} and test_counts == {1, 2, 3}
    assert 1 in train_counts and 2 in train_counts
    assert same_color_pairs >= 5 and mixed_direction_pairs >= 5
    assert edge_pairs >= 10 and full_size_pairs >= 3
    assert overlap_puzzles >= 8 and extrapolation_puzzles >= 5
    assert smaller_test_puzzles >= 8 and small_test_objects >= 3
    assert uniform_puzzles >= 8 and mixed_puzzles >= 8
    assert multi_test_pairs >= 10 and varied_test_sizes >= 8
    assert varied_test_offsets >= 8 and diagonal_test_layouts >= 3
    assert same_color_mixed_test_pairs >= 2 and zero_object_pairs >= 8
    assert len(step_profiles) >= 40
    assert len(shapes) >= 30
    for seed, train_count, test_count in ((100, 3, 2), (101, 7, 3), (102, 10, 4), (200, 2, 1), (201, 10, 1), (202, 10, 4)):
        p = generate_puzzle(seed, train_count, test_count)
        assert len(p["train"]) == train_count and len(p["test"]) == test_count
        validate_generated_puzzle(p)
    for seed, train_count, test_count in ((300, None, 1), (301, None, 4), (302, 2, None), (303, 10, None)):
        p = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(p)
        if train_count is not None:
            assert len(p["train"]) == train_count
        if test_count is not None:
            assert len(p["test"]) == test_count
    assert _puzzle_signature(generate_puzzle(900, 6, 3)) == _puzzle_signature(generate_puzzle(900, 6, 3))
    assert _puzzle_signature(generate_puzzle(900, 6, 3)) != _puzzle_signature(generate_puzzle(901, 6, 3))
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 3}, {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(400, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for rows_in, rows_out in _SOURCE_ROW_PAIRS:
        inp = np.array([[int(c) for c in row] for row in rows_in], dtype=np.int8)
        out = np.array([[int(c) for c in row] for row in rows_out], dtype=np.int8)
        assert np.array_equal(out, _derive_output_from_input(inp))
