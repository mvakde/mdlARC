"""Generate puzzles whose four erased lattice crossings bound an inclusive crop."""

import hashlib

import numpy as np


_PITCHES = (4, 5, 6, 7)
_SOURCE_INPUT_HASHES = {
    (17, 21): "8626cbb52969a4604a6b3b93d7ea55530576bdff2c663878b43fe67666035278",
    (18, 22): "d2301a276485968769cfc89c6e91995cd6411f7a8e5869552391aa958d57769f",
    (19, 20): "aebdeaecd961f10f1365cd8c04be697a1a2440ba2b3c6ec09c7ec12db6cdd8b5",
}


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _optional_count("num_train", num_train, 2, 10)
    num_test = _optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _source_copy(grid: np.ndarray) -> bool:
    expected = _SOURCE_INPUT_HASHES.get(tuple(grid.shape))
    return expected is not None and hashlib.sha256(
        grid.astype(np.int8, copy=False).tobytes()
    ).hexdigest() == expected


def _orientation(height: int, width: int) -> str:
    if height == width:
        return "square"
    return "landscape" if height < width else "portrait"


def _geometry_options(orientation: str, unequal: bool) -> list[tuple[int, int, int, int]]:
    options = []
    for row_pitch in _PITCHES:
        for col_pitch in _PITCHES:
            if (row_pitch != col_pitch) != unequal:
                continue
            for row_intervals in range(1, 5):
                crop_height = row_pitch * row_intervals + 1
                if crop_height >= 30:
                    continue
                for col_intervals in range(1, 5):
                    crop_width = col_pitch * col_intervals + 1
                    if crop_width >= 30:
                        continue
                    if _orientation(crop_height, crop_width) == orientation:
                        options.append((row_pitch, col_pitch, row_intervals, col_intervals))
    return options


_GEOMETRIES = {
    (orientation, unequal): _geometry_options(orientation, unequal)
    for orientation in ("square", "landscape", "portrait")
    for unequal in (False, True)
}


def _sample_geometry(
    rng: np.random.Generator, orientation: str, unequal: bool
) -> tuple[int, int, int, int]:
    options = _GEOMETRIES[orientation, unequal]
    weights = np.asarray([
        (2.0 if row_pitch in (4, 5, 6) else 0.65)
        * (2.0 if col_pitch in (4, 5, 6) else 0.65)
        / (max(row_intervals, col_intervals) ** 0.7)
        for row_pitch, col_pitch, row_intervals, col_intervals in options
    ], dtype=float)
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))]


def _axis_layout(
    rng: np.random.Generator, pitch: int, intervals: int
) -> tuple[int, np.ndarray, tuple[int, int]]:
    eligible_counts = []
    for line_count in range(max(3, intervals + 1), 1 + 29 // pitch + 1):
        base_length = (line_count - 1) * pitch + 1
        if base_length > 30:
            continue
        if base_length == 30 and intervals == line_count - 1:
            continue
        eligible_counts.append(line_count)
    if not eligible_counts:
        raise ValueError("impossible axis geometry")

    weights = np.asarray([1 / (n - 2) ** 1.35 for n in eligible_counts], dtype=float)
    weights /= weights.sum()
    line_count = eligible_counts[int(rng.choice(len(eligible_counts), p=weights))]
    base_length = (line_count - 1) * pitch + 1
    maximum_margin = min(2 * (pitch - 1), 30 - base_length)
    minimum_margin = int(intervals == line_count - 1)
    total_margin = int(rng.integers(minimum_margin, maximum_margin + 1))
    low = max(0, total_margin - (pitch - 1))
    high = min(pitch - 1, total_margin)
    leading_margin = int(rng.integers(low, high + 1))
    trailing_margin = total_margin - leading_margin
    lines = leading_margin + np.arange(line_count, dtype=np.int64) * pitch
    first = int(rng.integers(0, line_count - intervals))
    markers = (int(lines[first]), int(lines[first + intervals]))
    length = int(lines[-1] + 1 + trailing_margin)
    return length, lines, markers


def _sample_roles(
    rng: np.random.Generator, prior_roles: list[tuple[int, int, int]]
) -> tuple[int, int, int]:
    if not prior_roles:
        return tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    previous = prior_roles[-1]
    mode = int(rng.choice(4, p=(0.18, 0.23, 0.28, 0.31)))
    if mode == 0:
        return previous
    if mode == 1:
        permutation = rng.permutation(3)
        while all(int(permutation[i]) == i for i in range(3)):
            permutation = rng.permutation(3)
        return tuple(previous[int(i)] for i in permutation)
    if mode == 2:
        retained = int(rng.choice(previous))
        others = rng.choice([x for x in range(10) if x not in previous], size=2, replace=False)
        colors = [retained, int(others[0]), int(others[1])]
        rng.shuffle(colors)
        return tuple(colors)
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _payload_density(rng: np.random.Generator) -> float:
    band = int(rng.choice(3, p=(0.25, 0.35, 0.40)))
    bounds = ((0.035, 0.085), (0.085, 0.16), (0.16, 0.25))[band]
    return float(rng.uniform(*bounds))


def _place_payload(
    grid: np.ndarray,
    eligible: np.ndarray,
    count: int,
    color: int,
    rng: np.random.Generator,
    clustered: bool,
) -> None:
    available = eligible.copy()
    placed: list[tuple[int, int]] = []
    for _ in range(count):
        neighbors = []
        if clustered and placed and rng.random() < 0.72:
            row, col = placed[int(rng.integers(len(placed)))]
            for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and available[rr, cc]:
                    neighbors.append((rr, cc))
        if neighbors:
            row, col = neighbors[int(rng.integers(len(neighbors)))]
        else:
            flat = np.flatnonzero(available)
            row, col = np.unravel_index(int(rng.choice(flat)), grid.shape)
        grid[row, col] = color
        available[row, col] = False
        placed.append((int(row), int(col)))


def _make_pair(
    rng: np.random.Generator,
    geometry: tuple[int, int, int, int],
    roles: tuple[int, int, int],
    payload_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    row_pitch, col_pitch, row_intervals, col_intervals = geometry
    background, lattice, payload = roles
    for _ in range(256):
        height, line_rows, (top, bottom) = _axis_layout(rng, row_pitch, row_intervals)
        width, line_cols, (left, right) = _axis_layout(rng, col_pitch, col_intervals)
        grid = np.full((height, width), background, dtype=np.int8)
        lattice_mask = np.zeros(grid.shape, dtype=bool)
        lattice_mask[line_rows, :] = True
        lattice_mask[:, line_cols] = True
        grid[lattice_mask] = lattice
        grid[np.ix_((top, bottom), (left, right))] = background

        inside = np.zeros(grid.shape, dtype=bool)
        inside[top:bottom + 1, left:right + 1] = True
        inside_candidates = ~lattice_mask & inside
        outside_candidates = ~lattice_mask & ~inside
        inside_available = int(inside_candidates.sum())
        outside_available = int(outside_candidates.sum())
        if inside_available == 0 or outside_available == 0:
            continue

        density = _payload_density(rng)
        inside_count = int(np.rint(inside_available * density))
        outside_count = int(np.rint(outside_available * density))
        if payload_mode == "inside_only":
            inside_count = max(1, inside_count)
            outside_count = 0
        elif payload_mode == "outside_only":
            inside_count = 0
            outside_count = max(1, outside_count)
        else:
            inside_count = max(1, inside_count)
            outside_count = max(1, outside_count)
        inside_count = min(inside_count, inside_available)
        outside_count = min(outside_count, outside_available)
        clustered = bool(rng.random() < 0.45)
        _place_payload(grid, inside_candidates, inside_count, payload, rng, clustered)
        _place_payload(grid, outside_candidates, outside_count, payload, rng, clustered)

        if int(np.count_nonzero(grid == background)) <= int(np.count_nonzero(grid == lattice)):
            continue
        if _source_copy(grid):
            continue
        return grid, grid[top:bottom + 1, left:right + 1].copy()
    raise RuntimeError("failed to construct a pair")


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), grid.dtype.str, grid.tobytes()


def _analyze_input(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = grid.shape
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 3
    candidates = []
    for lattice in colors:
        row_counts = np.count_nonzero(grid == lattice, axis=1)
        col_counts = np.count_nonzero(grid == lattice, axis=0)
        line_rows = np.flatnonzero((row_counts == width) | (row_counts == width - 2))
        line_cols = np.flatnonzero((col_counts == height) | (col_counts == height - 2))
        if len(line_rows) < 3 or len(line_cols) < 3:
            continue
        row_diffs, col_diffs = np.diff(line_rows), np.diff(line_cols)
        if not (np.all(row_diffs == row_diffs[0]) and np.all(col_diffs == col_diffs[0])):
            continue
        row_pitch, col_pitch = int(row_diffs[0]), int(col_diffs[0])
        if row_pitch not in _PITCHES or col_pitch not in _PITCHES:
            continue
        deficient_rows = line_rows[row_counts[line_rows] == width - 2]
        deficient_cols = line_cols[col_counts[line_cols] == height - 2]
        if len(deficient_rows) != 2 or len(deficient_cols) != 2:
            continue
        corners = grid[np.ix_(deficient_rows, deficient_cols)]
        background = int(corners[0, 0])
        if background == lattice or not np.all(corners == background):
            continue
        other = [color for color in colors if color not in (background, lattice)]
        if len(other) != 1:
            continue
        payload = other[0]
        lattice_mask = np.zeros(grid.shape, dtype=bool)
        lattice_mask[line_rows, :] = True
        lattice_mask[:, line_cols] = True
        gap_mask = np.zeros(grid.shape, dtype=bool)
        gap_mask[np.ix_(deficient_rows, deficient_cols)] = True
        if not np.all(grid[lattice_mask & ~gap_mask] == lattice):
            continue
        if np.any(grid[~lattice_mask] == lattice):
            continue
        top, bottom = (int(x) for x in deficient_rows)
        left, right = (int(x) for x in deficient_cols)
        row_intervals = (bottom - top) // row_pitch
        col_intervals = (right - left) // col_pitch
        if row_intervals < 1 or col_intervals < 1:
            continue
        output = grid[top:bottom + 1, left:right + 1].copy()
        if output.shape[0] >= height or output.shape[1] >= width:
            continue
        if int(np.count_nonzero(grid == background)) <= int(np.count_nonzero(grid == lattice)):
            continue
        inside_payload = int(np.count_nonzero(output == payload))
        outside_payload = int(np.count_nonzero(grid == payload)) - inside_payload
        candidates.append((output, {
            "role": (background, lattice, payload),
            "row_pitch": row_pitch, "col_pitch": col_pitch,
            "row_intervals": row_intervals, "col_intervals": col_intervals,
            "line_rows": tuple(int(x) for x in line_rows),
            "line_cols": tuple(int(x) for x in line_cols),
            "bounds": (top, bottom, left, right),
            "input_shape": grid.shape, "output_shape": output.shape,
            "orientation": _orientation(*output.shape),
            "inside_payload": inside_payload,
            "outside_payload": outside_payload,
            "payload_density": (inside_payload + outside_payload) / int((~lattice_mask).sum()),
        }))
    assert len(candidates) == 1, "input must identify one lattice and rectangle"
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        analyses = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                input_grid, output_grid = pair["input"], pair["output"]
                for grid in (input_grid, output_grid):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert int(grid.min()) >= 0 and int(grid.max()) <= 9
                assert not _source_copy(input_grid)
                input_key = _grid_key(input_grid)
                pair_key = input_key, _grid_key(output_grid)
                assert input_key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(input_key)
                seen_pairs.add(pair_key)
                predicted, analysis = _analyze_input(input_grid)
                assert np.array_equal(predicted, output_grid)
                assert input_grid.shape != output_grid.shape
                analyses[section].append(analysis)

        # Each pair supplies its own palette and pitch. The learned convention is
        # the same four-gap inclusive crop throughout the complete puzzle.
        train = analyses["train"]
        assert any(a["inside_payload"] and a["outside_payload"] for a in train)
        assert len({a["orientation"] for a in train}) >= 2
        assert len({(a["row_pitch"], a["col_pitch"], a["row_intervals"], a["col_intervals"])
                    for a in train}) >= 2
        if any(a["row_pitch"] != a["col_pitch"] for a in analyses["test"]):
            assert any(a["row_pitch"] != a["col_pitch"] for a in train)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed generated puzzle") from exc


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)

    # The only shared convention is visible in every pair: four background
    # gaps at lattice crossings mark the exact inclusive crop. No color map,
    # pitch, or orientation is hidden at puzzle level.
    source_like = bool(seed % 7 == 0)
    extended = bool(rng.random() < 0.30) and not source_like
    train_orientations = []
    if source_like:
        train_orientations.extend(("square", "landscape"))
    else:
        first = ("square", "landscape", "portrait")[int(rng.integers(3))]
        second_choices = [x for x in ("square", "landscape", "portrait") if x != first]
        train_orientations.extend((first, second_choices[int(rng.integers(2))]))
    while len(train_orientations) < num_train:
        train_orientations.append(("square", "landscape", "portrait")[int(rng.integers(3))])
    test_orientations = []
    if source_like:
        test_orientations.append("portrait")
    while len(test_orientations) < num_test:
        test_orientations.append(("square", "landscape", "portrait")[int(rng.integers(3))])

    unequal_train_index = int(rng.integers(num_train)) if extended else -1
    puzzle = {"train": [], "test": []}
    prior_roles: list[tuple[int, int, int]] = []
    seen_inputs = set()
    for section, orientations in (("train", train_orientations), ("test", test_orientations)):
        for index, orientation in enumerate(orientations):
            unequal = False
            if extended:
                unequal = (index == unequal_train_index) if section == "train" else bool(rng.random() < 0.65)
                if section == "train" and index != unequal_train_index:
                    unequal = bool(rng.random() < 0.18)
            if source_like and index < 2 and section == "train":
                geometry = (4, 4, 2, 2) if index == 0 else (5, 5, 1, 2)
            elif source_like and index == 0 and section == "test":
                geometry = (5, 5, 2, 1)
            else:
                geometry = _sample_geometry(rng, orientation, unequal)
            roles = _sample_roles(rng, prior_roles)
            prior_roles.append(roles)
            payload_mode = "both" if section == "train" and index == 0 else (
                "both", "inside_only", "outside_only"
            )[int(rng.choice(3, p=(0.68, 0.16, 0.16)))]
            for _ in range(64):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
                pair_rng = np.random.default_rng(pair_seed)
                input_grid, output_grid = _make_pair(pair_rng, geometry, roles, payload_mode)
                key = _grid_key(input_grid)
                if key not in seen_inputs:
                    seen_inputs.add(key)
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("failed to generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[section]) == len(second[section]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[section], second[section])
        ) for section in ("train", "test")
    )


def self_test() -> None:
    first = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(first, generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(first, generate_puzzle(314160, 7, 3))
    for seed, train_count, test_count in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                                          (200, None, 1), (201, None, 4),
                                          (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": 1.5}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")

    train_counts, test_counts = set(), set()
    signatures, shapes, geometries, roles, orientations = set(), set(), set(), set(), set()
    three_line_axes = outside_only = inside_only = dense = high_outside = unequal = 0
    three_line_test = outside_only_test = first_last = near_edge = deep_crop = 0
    adjacent_payload = same_role = swapped_role = new_color = 0
    familiar_test = unseen_test = 0
    source_shapes = 0
    within_pitch = within_role = within_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_geometries, local_pitches, local_roles = set(), set(), set()
        local_analyses = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid, output = pair["input"], pair["output"]
                assert not _source_copy(grid)
                predicted, a = _analyze_input(grid)
                assert np.array_equal(predicted, output)
                signatures.add(_grid_key(grid))
                shapes.add(a["input_shape"])
                orientations.add((section, a["orientation"]))
                roles.add(a["role"])
                geometry = (a["row_pitch"], a["col_pitch"], a["row_intervals"], a["col_intervals"])
                geometries.add(geometry)
                local_geometries.add(geometry)
                local_pitches.add((a["row_pitch"], a["col_pitch"]))
                local_roles.add(a["role"])
                local_analyses[section].append(a)
                three_line_axes += int(len(a["line_rows"]) == 3) + int(len(a["line_cols"]) == 3)
                three_line_test += int(section == "test" and
                                       (len(a["line_rows"]) == 3 or len(a["line_cols"]) == 3))
                outside_only += int(a["inside_payload"] == 0)
                outside_only_test += int(section == "test" and a["inside_payload"] == 0)
                inside_only += int(a["outside_payload"] == 0)
                dense += int(a["payload_density"] >= 0.125)
                high_outside += int(a["outside_payload"] > 36)
                unequal += int(a["row_pitch"] != a["col_pitch"])
                source_shapes += int(a["input_shape"] in _SOURCE_INPUT_HASHES)
                top, bottom, left, right = a["bounds"]
                first_last += int((top == a["line_rows"][0] and bottom == a["line_rows"][-1]) or
                                  (left == a["line_cols"][0] and right == a["line_cols"][-1]))
                edge_distance = min(top, left, grid.shape[0] - 1 - bottom, grid.shape[1] - 1 - right)
                near_edge += int(edge_distance <= 1)
                deep_crop += int(edge_distance >= 4)
                payload = a["role"][2]
                adjacent_payload += int(bool(
                    np.any((grid[:-1, :] == payload) & (grid[1:, :] == payload)) or
                    np.any((grid[:, :-1] == payload) & (grid[:, 1:] == payload))
                ))
        training_orientations = {a["orientation"] for a in local_analyses["train"]}
        familiar_test += int(any(a["orientation"] in training_orientations
                                 for a in local_analyses["test"]))
        unseen_test += int(any(a["orientation"] not in training_orientations
                               for a in local_analyses["test"]))
        ordered = local_analyses["train"] + local_analyses["test"]
        for previous, current in zip(ordered, ordered[1:]):
            same_role += int(previous["role"] == current["role"])
            swapped_role += int(sorted(previous["role"]) == sorted(current["role"]) and
                                previous["role"] != current["role"])
            new_color += int(bool(set(current["role"]) - set(previous["role"])))
        within_pitch += int(len(local_pitches) > 1)
        within_role += int(len(local_roles) > 1)
        within_geometry += int(len(local_geometries) > 2)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 200 and len(shapes) >= 40 and len(geometries) >= 30
    assert len(roles) >= 30 and len(orientations) == 6
    assert three_line_axes >= 20 and outside_only >= 5 and inside_only >= 5
    assert dense >= 30 and high_outside >= 1 and unequal >= 10
    assert source_shapes >= 1 and within_pitch >= 15 and within_role >= 30
    assert within_geometry >= 15
    assert three_line_test >= 5 and outside_only_test >= 2 and first_last >= 10
    assert near_edge >= 20 and deep_crop >= 1 and adjacent_payload >= 20
    assert same_role >= 3 and swapped_role >= 3 and new_color >= 10
    assert familiar_test >= 10 and unseen_test >= 1
