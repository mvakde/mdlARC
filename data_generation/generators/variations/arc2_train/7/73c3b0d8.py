import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(v) for v in grid.shape),
        tuple(int(v) for v in grid.reshape(-1)),
    )


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _grid_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_RAW_PAIRS = (
    (
        (
            "04000000",
            "00000400",
            "00000000",
            "00000000",
            "00400000",
            "00000000",
            "00000000",
            "00004000",
            "00000000",
            "22222222",
            "00040000",
            "00000000",
        ),
        (
            "00000000",
            "04000000",
            "00000400",
            "00000000",
            "40000000",
            "04400004",
            "00400040",
            "00040400",
            "00004000",
            "22222222",
            "00000000",
            "00040000",
        ),
    ),
    (
        (
            "000000",
            "000000",
            "040000",
            "000040",
            "000000",
            "222222",
            "000000",
            "000000",
            "000000",
            "000000",
        ),
        (
            "400000",
            "040000",
            "004000",
            "040404",
            "000040",
            "222222",
            "000000",
            "000000",
            "000000",
            "000000",
        ),
    ),
    (
        (
            "000",
            "040",
            "000",
            "000",
            "222",
            "000",
        ),
        (
            "000",
            "000",
            "040",
            "000",
            "222",
            "000",
        ),
    ),
    (
        (
            "00000",
            "00000",
            "00000",
            "00000",
            "00400",
            "00000",
            "22222",
        ),
        (
            "00000",
            "00000",
            "00000",
            "40004",
            "04040",
            "00400",
            "22222",
        ),
    ),
    (
        (
            "00004000",
            "40000000",
            "00000000",
            "22222222",
            "00000000",
            "00000000",
            "04000000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
        ),
        (
            "00400000",
            "04004000",
            "40000000",
            "22222222",
            "00000000",
            "00000000",
            "00000000",
            "04000000",
            "00000000",
            "00000000",
            "00000000",
            "00000000",
        ),
    ),
)


_SOURCE_PAIR_KEYS = frozenset(
    (
        _grid_key(_grid_from_rows(input_rows)),
        _grid_key(_grid_from_rows(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_RAW_PAIRS
)


def _normalize_count(
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

    normalized = int(value)
    if not minimum <= normalized <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return normalized


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % span

    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        if minimum_train > 10:
            raise ValueError("No legal num_train exists for num_test")
        span = 11 - minimum_train
        train_count = minimum_train + seed % span

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("No legal num_test exists for num_train")
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    roles = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    return {"roles": roles, "local_palette": bool(rng.integers(0, 2))}


def _can_add_particle(
    position: tuple[int, int],
    positions: list[tuple[int, int]],
    height: int,
    barrier_row: int,
) -> bool:
    row, column = position

    if not 0 <= row < height - 1:
        return False
    if row == barrier_row or row == barrier_row - 1:
        return False

    for other_row, other_column in positions:
        if (row, column) == (other_row, other_column):
            return False

        # Input particles must remain isolated under 4-connectivity.
        if abs(row - other_row) + abs(column - other_column) == 1:
            return False

        # Every particle's one-row destination must be background.
        if (row + 1, column) == (other_row, other_column):
            return False
        if (other_row + 1, other_column) == (row, column):
            return False

    return True


def _add_from_candidates(
    rng: np.random.Generator,
    positions: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    count: int,
    height: int,
    barrier_row: int,
) -> bool:
    if count == 0:
        return True

    added = 0
    for candidate_index in rng.permutation(len(candidates)):
        candidate = candidates[int(candidate_index)]
        if _can_add_particle(
            candidate,
            positions,
            height,
            barrier_row,
        ):
            positions.append(candidate)
            added += 1
            if added == count:
                return True

    return False


def _analyze_input(input_grid: np.ndarray) -> dict:
    _require(
        isinstance(input_grid, np.ndarray),
        "Input must be a NumPy array",
    )
    _require(input_grid.ndim == 2, "Input must be two-dimensional")

    height, width = input_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)

    _require(
        len(values) == 3,
        "Each input must contain exactly three semantic colors",
    )

    maximum_count = int(counts.max())
    _require(
        int(np.count_nonzero(counts == maximum_count)) == 1,
        "Background color must be uniquely identifiable",
    )
    background = int(values[int(np.argmax(counts))])

    barrier_candidates = []
    for row in range(height):
        if np.all(input_grid[row] == input_grid[row, 0]):
            row_color = int(input_grid[row, 0])
            if row_color != background:
                barrier_candidates.append((row, row_color))

    _require(
        len(barrier_candidates) == 1,
        "Input must have exactly one non-background full-width row",
    )
    barrier_row, barrier = barrier_candidates[0]

    remaining_colors = {
        int(value) for value in values
    } - {background, barrier}
    _require(
        len(remaining_colors) == 1,
        "Particle color must be uniquely identifiable",
    )
    particle = remaining_colors.pop()

    expected_barrier_mask = np.zeros(input_grid.shape, dtype=bool)
    expected_barrier_mask[barrier_row, :] = True
    _require(
        np.array_equal(input_grid == barrier, expected_barrier_mask),
        "Barrier color may appear only in the complete barrier row",
    )

    particle_locations = np.argwhere(input_grid == particle)
    particles = tuple(
        (int(row), int(column))
        for row, column in particle_locations
    )
    _require(len(particles) >= 1, "At least one particle is required")

    for index, (row, column) in enumerate(particles):
        _require(
            row < height - 1,
            "Particles may not begin in the bottom row",
        )
        _require(
            row != barrier_row - 1,
            "Particles may not begin directly above the barrier",
        )
        _require(
            int(input_grid[row + 1, column]) == background,
            "Every particle must have a background destination below it",
        )

        for other_row, other_column in particles[index + 1 :]:
            _require(
                abs(row - other_row) + abs(column - other_column) != 1,
                "Input particles must be isolated",
            )

    triggers = tuple(
        (row, column)
        for row, column in particles
        if row == barrier_row - 2
    )

    return {
        "shape": (height, width),
        "background": background,
        "barrier": barrier,
        "particle": particle,
        "barrier_row": barrier_row,
        "particles": particles,
        "triggers": triggers,
    }


def _render_output(
    input_grid: np.ndarray,
    analysis: dict,
) -> np.ndarray:
    height, width = analysis["shape"]
    background = analysis["background"]
    barrier = analysis["barrier"]
    particle = analysis["particle"]
    barrier_row = analysis["barrier_row"]

    output = np.full(
        (height, width),
        background,
        dtype=input_grid.dtype,
    )
    output[barrier_row, :] = barrier

    for row, column in analysis["particles"]:
        landing_row = row + 1
        output[landing_row, column] = particle

        if landing_row == barrier_row - 1:
            distance = 1
            splash_row = landing_row - 1

            while splash_row >= 0:
                left_column = column - distance
                right_column = column + distance

                if left_column >= 0:
                    output[splash_row, left_column] = particle
                if right_column < width:
                    output[splash_row, right_column] = particle

                distance += 1
                splash_row -= 1

    return output


def _make_pair(height, width, barrier_row, positions, roles):
    background, barrier, particle = roles
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[barrier_row, :] = barrier
    for row, column in positions:
        input_grid[row, column] = particle
    return input_grid, _render_output(input_grid, _analyze_input(input_grid))


def _draw_dimension(rng, maximum, is_height):
    if is_height:
        bands = ((6, 8), (9, 13), (14, 22), (23, 28))
    else:
        bands = ((3, 5), (6, 10), (11, 19), (20, 28))
    weights = (0.20, 0.36, 0.35, 0.09)
    available = [i for i, (low, _) in enumerate(bands) if low <= maximum]
    probabilities = np.array([weights[i] for i in available], dtype=float)
    probabilities /= probabilities.sum()
    band = bands[int(rng.choice(available, p=probabilities))]
    return int(rng.integers(band[0], min(band[1], maximum) + 1))


def _barrier_category(height, barrier_row):
    fraction = barrier_row / (height - 1)
    return "high" if fraction < 0.34 else "low" if fraction > 0.66 else "middle"


def _splash_geometry(height, width, barrier_row, columns):
    if not columns:
        return "none"
    if any(column in (0, width - 1) for column in columns):
        return "edge"
    if len(columns) == 1:
        column = columns[0]
        if min(column, width - 1 - column) >= barrier_row - 1:
            return "full_v"
    return "interior_clipped"


def _ray_overlap(barrier_row, columns):
    return any(
        (right - left) % 2 == 0 and (right - left) // 2 <= barrier_row - 1
        for i, left in enumerate(columns)
        for right in columns[i + 1 :]
    )


def _choose_trigger_columns(rng, width, barrier_row, count, geometry, overlap):
    if count == 0:
        return []
    if count == 1:
        if geometry == "edge":
            options = [0, width - 1]
        elif geometry == "near":
            options = [1, width - 2] if width >= 4 else [1]
        elif geometry == "full_v":
            options = [
                c for c in range(1, width - 1)
                if min(c, width - 1 - c) >= barrier_row - 1
            ]
        else:
            options = list(range(1, width - 1))
        if not options:
            return None
        return [int(rng.choice(options))]

    candidates = [
        (left, right)
        for left in range(width)
        for right in range(left + 2, width)
        if _ray_overlap(barrier_row, (left, right)) == overlap
        and (geometry != "interior" or (left > 0 and right < width - 1))
    ]
    if not candidates:
        return None
    left, right = candidates[int(rng.integers(len(candidates)))]
    columns = [left, right]
    if count == 3:
        options = [
            c for c in range(width)
            if all(abs(c - other) >= 2 for other in columns)
            and (overlap or not _ray_overlap(barrier_row, tuple(sorted(columns + [c]))))
        ]
        if not options:
            return None
        columns.append(int(rng.choice(options)))
    return sorted(columns)


def generate_pair(rng, regime, puzzle_parameters, spec, limits=(28, 28)):
    if regime not in {"source_train", "target_test"}:
        raise ValueError("Unknown pair regime")
    for _ in range(400):
        source_scale = [
            shape for shape in ((6, 3), (7, 5))
            if shape[0] <= limits[0] and shape[1] <= limits[1]
        ]
        if source_scale and rng.random() < 0.12:
            height, width = source_scale[int(rng.integers(len(source_scale)))]
        else:
            height = _draw_dimension(rng, limits[0], True)
            width = _draw_dimension(rng, limits[1], False)
        candidates = [
            b for b in range(2, height)
            if (not spec["above"] or b >= 3)
            and (not spec["below"] or b <= height - 3)
            and (not spec["triggers"] or b >= 2)
            and (spec["barrier"] == "any" or _barrier_category(height, b) == spec["barrier"])
        ]
        if not candidates:
            continue
        barrier_row = int(rng.choice(candidates))
        columns = _choose_trigger_columns(
            rng, width, barrier_row, spec["triggers"],
            spec["geometry"], spec["overlap"],
        )
        if columns is None:
            continue
        positions = [(barrier_row - 2, c) for c in columns]
        above = [
            (r, c) for r in range(barrier_row - 2) for c in range(width)
        ]
        below = [
            (r, c) for r in range(barrier_row + 1, height - 1)
            for c in range(width)
        ]
        if not _add_from_candidates(
            rng, positions, above, spec["above"], height, barrier_row
        ):
            continue
        if not _add_from_candidates(
            rng, positions, below, spec["below"], height, barrier_row
        ):
            continue
        if len(positions) < 1:
            continue
        roles = puzzle_parameters["roles"]
        if puzzle_parameters["local_palette"] and spec["palette"] is not None:
            roles = spec["palette"]
        pair = _make_pair(height, width, barrier_row, positions, roles)
        return pair
    if not spec.get("fallback"):
        relaxed = dict(spec)
        relaxed.update(
            triggers=min(spec["triggers"], 1),
            above=min(spec["above"], 1),
            below=min(spec["below"], 1),
            geometry="near",
            overlap=False,
            barrier="any",
            fallback=True,
        )
        return generate_pair(rng, regime, puzzle_parameters, relaxed, limits)
    raise RuntimeError("Unable to construct a valid pair")


def _random_spec(rng, profile, side="both", geometry=None, multi=False, overlap=False):
    if profile == "plain":
        above = int(rng.integers(1, 4)) if side in ("above", "both") else 0
        below = int(rng.integers(1, 4)) if side in ("below", "both") else 0
        if rng.random() < 0.30 and side != "both":
            above = 1 if above else 0
            below = 1 if below else 0
        triggers = 0
    else:
        triggers = int(rng.integers(2, 4)) if multi else 1
        above = int(rng.integers(1, 3)) if side in ("above", "both") else 0
        below = int(rng.integers(1, 3)) if side in ("below", "both") else 0
    return {
        "triggers": triggers,
        "above": above,
        "below": below,
        "geometry": geometry or ("interior" if multi else "near"),
        "overlap": overlap,
        "barrier": str(rng.choice(("high", "middle", "low"))),
        "palette": None,
    }


def _palette_for_pair(rng, parameters, used):
    if not parameters["local_palette"]:
        return parameters["roles"]
    for _ in range(30):
        palette = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
        if palette not in used:
            return palette
    raise RuntimeError("Unable to draw a new palette")


def _pair_features(pair):
    info = _analyze_input(pair["input"])
    b = info["barrier_row"]
    h, w = info["shape"]
    columns = tuple(c for _, c in info["triggers"])
    moved = {(r + 1, c) for r, c in info["particles"]}
    rays = set()
    for column in columns:
        for distance in range(1, b):
            row = b - 1 - distance
            for ray_column in (column - distance, column + distance):
                if 0 <= ray_column < w:
                    rays.add((row, ray_column))
    return {
        "analysis": info,
        "category": _barrier_category(h, b),
        "geometry": _splash_geometry(h, w, b, columns),
        "multiple": len(columns) >= 2,
        "overlap": _ray_overlap(b, columns),
        "ray_move_overlap": bool(rays & moved),
        "above": any(r < b - 2 for r, _ in info["particles"]),
        "below": any(r > b for r, _ in info["particles"]),
        "edge": any(c in (0, w - 1) for c in columns),
        "side_clip": any(
            c < b - 1 or w - 1 - c < b - 1 for c in columns
        ),
    }


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_pairs, used_palettes = set(), set(), set()

    plain_side = str(rng.choice(("above", "below")))
    splash_side = "below" if plain_side == "above" else "above"
    splash_multi = bool(rng.random() < 0.38)
    splash_geometry = "interior" if splash_multi else str(
        rng.choice(("near", "full_v", "interior"), p=(0.50, 0.27, 0.23))
    )
    splash_overlap = splash_multi and bool(rng.random() < 0.6)
    specs = [
        _random_spec(rng, "plain", plain_side),
        _random_spec(rng, "splash", splash_side, splash_geometry,
                     splash_multi, splash_overlap),
    ]
    if splash_geometry == "near":
        specs[1]["barrier"] = str(rng.choice(("high", "middle", "low")))
    for _ in range(train_count - 2):
        profile = str(rng.choice(("plain", "splash", "mixed"), p=(0.30, 0.35, 0.35)))
        side = str(rng.choice(("above", "below", "both", "none")))
        if profile == "plain":
            side = "both" if side == "none" else side
            specs.append(_random_spec(rng, "plain", side))
        else:
            multi = bool(rng.random() < 0.4)
            geometry = "interior" if multi else str(
                rng.choice(("near", "full_v", "edge", "interior"))
            )
            specs.append(_random_spec(rng, "splash", side, geometry,
                                      multi, multi and bool(rng.random() < 0.5)))
    rng.shuffle(specs)

    def append_pair(split, regime, spec, limits=(28, 28), train_features=None):
        for _ in range(120):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            spec = dict(spec)
            spec["palette"] = _palette_for_pair(pair_rng, parameters, used_palettes)
            input_grid, output_grid = generate_pair(
                pair_rng, regime, parameters, spec, limits
            )
            pair = {"input": input_grid, "output": output_grid}
            if train_features is not None:
                feature = _pair_features(pair)
                if feature["ray_move_overlap"] and not any(
                    f["ray_move_overlap"] for f in train_features
                ):
                    continue
            key = _pair_key(pair)
            input_key = _grid_key(input_grid)
            if key in seen_pairs or input_key in seen_inputs or key in _SOURCE_PAIR_KEYS:
                continue
            puzzle[split].append(pair)
            seen_inputs.add(input_key)
            seen_pairs.add(key)
            used_palettes.add(tuple(
                _analyze_input(input_grid)[name]
                for name in ("background", "barrier", "particle")
            ))
            return
        raise RuntimeError("Unable to construct a unique pair")

    for spec in specs:
        append_pair("train", "source_train", spec)

    train_features = [_pair_features(p) for p in puzzle["train"]]
    maximum_train_shape = (
        max(f["analysis"]["shape"][0] for f in train_features),
        max(f["analysis"]["shape"][1] for f in train_features),
    )
    has_multi = any(f["multiple"] for f in train_features)
    has_overlap = any(f["overlap"] for f in train_features)
    has_side_clip = any(f["side_clip"] for f in train_features)
    for _ in range(test_count):
        profile = str(rng.choice(("plain", "splash", "mixed"), p=(0.29, 0.33, 0.38)))
        side = str(rng.choice(("above", "below", "both", "none")))
        if profile == "plain":
            spec = _random_spec(rng, "plain", side if side != "none" else str(rng.choice(("above", "below"))))
        else:
            multi = has_multi and bool(rng.random() < 0.36)
            geometries = ["near", "full_v", "interior"]
            if has_side_clip:
                geometries.append("edge")
            geometry = "interior" if multi else str(rng.choice(geometries))
            spec = _random_spec(rng, "splash", side, geometry,
                                multi, multi and has_overlap and bool(rng.random() < 0.45))
        append_pair("test", "target_test", spec, maximum_train_shape,
                    train_features)
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_generated_puzzle_impl(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10, "Illegal train count")
    _require(1 <= len(puzzle["test"]) <= 4, "Illegal test count")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "Train count must exceed test count")
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Each pair must contain exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grids must be two-dimensional NumPy arrays")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, "Illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "Illegal ARC color")
            _require(inp.shape == out.shape, "Shape changed")
            _require(not np.array_equal(inp, out), "Degenerate pair")
            input_key, pair_key = _grid_key(inp), _pair_key(pair)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "Duplicate input or pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "Original source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_input(inp)
            _require(np.array_equal(out, _render_output(inp, info)),
                     "Incorrect falling and splash output")
            features[split].append(_pair_features(pair))

    train = features["train"]
    test = features["test"]
    _require(any(f["geometry"] != "none" for f in train),
             "Training must show a splash")
    _require(any(f["geometry"] == "none" for f in train),
             "Training must show ordinary-only descent")
    _require(any(f["above"] for f in train), "Training must show ordinary descent above")
    _require(any(f["below"] for f in train), "Training must show ordinary descent below")
    palettes = [tuple(f["analysis"][name] for name in
                      ("background", "barrier", "particle")) for f in train + test]
    if len(set(palettes)) > 1:
        _require(len(set(palettes[:len(train)])) > 1,
                 "Local palette variation must be shown in training")
    max_h = max(f["analysis"]["shape"][0] for f in train)
    max_w = max(f["analysis"]["shape"][1] for f in train)
    for f in test:
        h, w = f["analysis"]["shape"]
        _require(h <= max_h and w <= max_w, "Test exceeds demonstrated grid scale")
        if f["multiple"]:
            _require(any(t["multiple"] for t in train),
                     "Multiple splashes not demonstrated")
        if f["overlap"]:
            _require(any(t["overlap"] for t in train),
                     "Overlapping splashes not demonstrated")
        if f["ray_move_overlap"]:
            _require(any(t["ray_move_overlap"] for t in train),
                     "Ray and moved-particle overlap not demonstrated")
        if f["edge"]:
            _require(any(t["side_clip"] for t in train),
                     "Edge clipping not demonstrated")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"Malformed generated puzzle: {exc}") from exc


def _puzzles_equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a["input"], b["input"]) and
            np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        ) for split in ("train", "test")
    )


def _independent_transform(input_grid):
    info = _analyze_input(input_grid)
    background, particle = info["background"], info["particle"]
    barrier_row = info["barrier_row"]
    height, width = input_grid.shape
    output = input_grid.copy()
    for row, column in info["particles"]:
        output[row, column] = background
    for row, column in info["particles"]:
        output[row + 1, column] = particle
        if row + 1 == barrier_row - 1:
            for distance in range(1, row + 2):
                upward = row + 1 - distance
                if not 0 <= upward < height:
                    break
                for sideways in (-distance, distance):
                    ray_column = column + sideways
                    if 0 <= ray_column < width:
                        output[upward, ray_column] = particle
    return output


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    explicit = ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2))
    for seed, nt, ne in explicit:
        p = generate_puzzle(seed, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
        puzzles.append(p)
    partial = ((201, None, 1), (202, None, 4),
               (203, 2, None), (204, 10, None))
    for seed, nt, ne in partial:
        p = generate_puzzle(seed, nt, ne)
        assert nt is None or len(p["train"]) == nt
        assert ne is None or len(p["test"]) == ne
        puzzles.append(p)
    assert _puzzles_equal(generate_puzzle(777, 8, 3), generate_puzzle(777, 8, 3))
    assert not _puzzles_equal(generate_puzzle(778, 8, 3), generate_puzzle(779, 8, 3))
    assert len({len(p["train"]) for p in puzzles[:42]}) >= 4
    assert len({len(p["test"]) for p in puzzles[:42]}) >= 3
    illegal = ((1, None), (11, None), (None, 0), (None, 5),
               (2, 2), (3, 4), (4, 4))
    for nt, ne in illegal:
        try:
            generate_puzzle(999, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")

    shapes, palettes, layouts, barrier_categories = set(), set(), set(), set()
    train_splash_positions, test_geometries, train_multi = set(), set(), 0
    train_slot_triggers = (set(), set())
    test_trigger_counts = set()
    source_like_shapes = set()
    large_pairs = 0
    side_short_train = 0
    ray_move_train = 0
    ray_move_test = 0
    mixed_train = 0
    within_shape = 0
    within_barrier = 0
    within_count = 0
    local_palette_puzzles = 0
    plain_lone_tests = 0
    overlap_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_features = [_pair_features(p) for p in puzzle["train"]]
        test_features = [_pair_features(p) for p in puzzle["test"]]
        for slot in (0, 1):
            train_slot_triggers[slot].add(len(
                train_features[slot]["analysis"]["triggers"]
            ))
        all_features = train_features + test_features
        within_shape += len({f["analysis"]["shape"] for f in all_features}) > 1
        within_barrier += len({f["category"] for f in all_features}) > 1
        within_count += len({len(f["analysis"]["particles"]) for f in all_features}) > 1
        if len({tuple(f["analysis"][name] for name in
                ("background", "barrier", "particle"))
                for f in train_features + test_features}) > 1:
            local_palette_puzzles += 1
        for split, fs in (("train", train_features), ("test", test_features)):
            for pair, f in zip(puzzle[split], fs):
                assert np.array_equal(pair["output"], _independent_transform(pair["input"]))
                info = f["analysis"]
                shapes.add(info["shape"])
                if info["shape"] in ((6, 3), (7, 5)):
                    source_like_shapes.add(info["shape"])
                large_pairs += info["shape"][0] >= 15 or info["shape"][1] >= 15
                palettes.add(tuple(info[name] for name in
                                   ("background", "barrier", "particle")))
                layouts.add((info["shape"], info["barrier_row"],
                             info["particles"]))
                barrier_categories.add((split, f["category"]))
                if split == "train" and f["geometry"] != "none":
                    train_splash_positions.add(tuple(info["triggers"]))
                if split == "test":
                    test_geometries.add(f["geometry"])
                    test_trigger_counts.add(len(info["triggers"]))
                    plain_lone_tests += f["geometry"] == "none" and len(info["particles"]) == 1
                    overlap_tests += f["overlap"]
                    ray_move_test += f["ray_move_overlap"]
                if split == "train":
                    train_multi += f["multiple"]
                    ray_move_train += f["ray_move_overlap"]
                    mixed_train += bool(info["triggers"]) and f["above"] and f["below"]
                    if any(
                        max(column, info["shape"][1] - 1 - column)
                        < info["barrier_row"] - 1
                        for _, column in info["triggers"]
                    ):
                        side_short_train += 1
    assert len(shapes) >= 35 and len(layouts) >= 120
    assert len(palettes) >= 25
    assert len(barrier_categories) == 6
    assert source_like_shapes == {(6, 3), (7, 5)}
    assert large_pairs >= 40
    assert within_shape >= 40 and within_barrier >= 35 and within_count >= 40
    assert local_palette_puzzles >= 8
    assert len(puzzles) - local_palette_puzzles >= 8
    assert len(train_splash_positions) >= 15
    assert train_slot_triggers[0] == train_slot_triggers[1] == {0, 1, 2, 3}
    assert test_trigger_counts == {0, 1, 2, 3}
    assert {"none", "edge", "full_v", "interior_clipped"} <= test_geometries
    assert train_multi >= 8 and overlap_tests >= 1 and plain_lone_tests >= 1
    assert side_short_train >= 5 and ray_move_train >= 1 and ray_move_test >= 1
    assert mixed_train >= 5
