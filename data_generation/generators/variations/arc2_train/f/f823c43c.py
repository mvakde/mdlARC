import hashlib

import numpy as np


_SOURCE_PAIR_HASHES = {
    "ed7b012ad47b6c16ffca6de639ef57a4cd2805fcc2236621c729da7783c8a688",
    "10b4714566d5e387a09f66b3d2cb483895536f9ff3e1ccd7c444db3b8096cac3",
    "66161a75b1c960b8b4c6ff2b2b161bd14d6d67e75d489cd3a4d38d1d1cef61a8",
}

_MAX_TILE_AREA = 36


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_count(value, name: str, lower: int, upper: int):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        span = 10 - test_count
        train_count = test_count + 1 + (seed // 4) % span
    elif train_count is None:
        span = 10 - test_count
        train_count = test_count + 1 + seed % span
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _puzzle_signature(puzzle: dict) -> tuple:
    result = []
    for split in ("train", "test"):
        split_signature = []
        for pair in puzzle[split]:
            split_signature.append(
                (
                    _grid_signature(pair["input"]),
                    _grid_signature(pair["output"]),
                )
            )
        result.append(tuple(split_signature))
    return tuple(result)


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != {"train", "test"} or set(second) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _axis_period(grid: np.ndarray, axis: int) -> int:
    length = grid.shape[axis]

    for period in range(1, length // 2 + 1):
        if axis == 0:
            periodic = np.array_equal(grid[period:, :], grid[:-period, :])
        else:
            periodic = np.array_equal(grid[:, period:], grid[:, :-period])

        if periodic:
            return period

    return length


def _fundamental_period(grid: np.ndarray) -> tuple[int, int]:
    return _axis_period(grid, 0), _axis_period(grid, 1)


def _recover_texture(
    input_grid: np.ndarray,
    corruption: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = input_grid.shape
    known_coordinates = np.argwhere(input_grid != corruption)
    known_values = input_grid[
        known_coordinates[:, 0],
        known_coordinates[:, 1],
    ].astype(np.int64)

    legitimate_colors = np.unique(known_values)
    _require(
        legitimate_colors.size == 2,
        "Exactly two legitimate colors must remain observable",
    )

    first_color = int(legitimate_colors[0])
    second_color = int(legitimate_colors[1])

    best_area = _MAX_TILE_AREA + 1
    best_count = 0
    best_grid = None
    best_period = None

    max_row_period = height // 2
    max_column_period = width // 2

    for row_period in range(1, max_row_period + 1):
        max_pc_for_area = _MAX_TILE_AREA // row_period
        for column_period in range(
            1,
            min(max_column_period, max_pc_for_area) + 1,
        ):
            area = row_period * column_period
            if area > best_area:
                continue

            phases = (
                (known_coordinates[:, 0] % row_period) * column_period
                + known_coordinates[:, 1] % column_period
            )

            total_counts = np.bincount(phases, minlength=area)
            if np.any(total_counts == 0):
                continue

            first_counts = np.bincount(
                phases[known_values == first_color],
                minlength=area,
            )

            mixed_phase = (
                (first_counts != 0)
                & (first_counts != total_counts)
            )
            if np.any(mixed_phase):
                continue

            tile_flat = np.where(
                first_counts > 0,
                first_color,
                second_color,
            )

            if (
                not np.any(tile_flat == first_color)
                or not np.any(tile_flat == second_color)
            ):
                continue

            tile = tile_flat.reshape(row_period, column_period)
            reconstructed = tile[
                np.arange(height)[:, None] % row_period,
                np.arange(width)[None, :] % column_period,
            ]

            if area < best_area:
                best_area = area
                best_count = 1
                best_grid = reconstructed
                best_period = (row_period, column_period)
            elif area == best_area:
                best_count += 1

    _require(best_grid is not None, "No supported periodic reconstruction exists")
    _require(
        best_count == 1,
        "The smallest supported periodic reconstruction is ambiguous",
    )

    return best_grid.astype(np.int8), best_period


def _sample_tile(rng: np.random.Generator) -> np.ndarray:
    """Return a two-color primitive tile; its colors are assigned per pair."""
    styles = (
        "isolated", "block", "horizontal", "vertical", "staggered",
        "diagonal", "ell", "separated", "checker", "balanced", "irregular",
    )
    for _ in range(80):
        style = styles[int(rng.integers(len(styles)))]
        if style == "checker":
            tile = np.array([[False, True], [True, False]], dtype=bool)
        elif style == "horizontal":
            row_period = int(rng.integers(2, 7))
            tile = np.zeros((row_period, 1), dtype=bool)
            tile[:int(rng.integers(1, row_period))] = True
        elif style == "vertical":
            column_period = int(rng.integers(2, 7))
            tile = np.zeros((1, column_period), dtype=bool)
            tile[:, :int(rng.integers(1, column_period))] = True
        else:
            if style in ("block", "staggered", "diagonal", "ell"):
                row_period = int(rng.integers(3, 7))
                column_period = int(rng.integers(3, 7))
            else:
                row_period = int(rng.integers(2, 7))
                column_period = int(rng.integers(2, 7))
            tile = np.zeros((row_period, column_period), dtype=bool)
            if style == "isolated":
                tile[0, 0] = True
            elif style == "block":
                block_height = int(rng.integers(2, row_period))
                block_width = int(rng.integers(2, column_period))
                tile[:block_height, :block_width] = True
            elif style == "staggered":
                row_step = int(rng.integers(1, row_period))
                column_step = int(rng.integers(1, column_period))
                for k in range(int(rng.integers(2, min(row_period, column_period) + 1))):
                    tile[(k * row_step) % row_period, (k * column_step) % column_period] = True
            elif style == "diagonal":
                for row in range(row_period):
                    tile[row, row % column_period] = True
            elif style == "ell":
                tile[0, :int(rng.integers(2, column_period + 1))] = True
                tile[:int(rng.integers(2, row_period + 1)), 0] = True
            elif style == "separated":
                positions = rng.choice(row_period * column_period,
                                       size=int(rng.integers(2, min(6, row_period * column_period) + 1)),
                                       replace=False)
                tile.flat[positions] = True
            elif style == "balanced":
                tile = rng.random((row_period, column_period)) < 0.5
            else:
                density = float(rng.uniform(0.2, 0.65))
                tile = rng.random((row_period, column_period)) < density
        if not (np.any(tile) and np.any(~tile)):
            continue
        if tile.shape[0] * tile.shape[1] > _MAX_TILE_AREA:
            continue
        # A repeated tile must really have the declared fundamental period.
        doubled = np.tile(tile, (3, 3))
        if _fundamental_period(doubled) != tile.shape:
            continue
        tile = np.roll(tile, int(rng.integers(tile.shape[0])), axis=0)
        tile = np.roll(tile, int(rng.integers(tile.shape[1])), axis=1)
        return tile
    raise RuntimeError("Unable to sample a primitive tile")


def _sample_size(rng: np.random.Generator, period: int, partial: bool) -> int:
    tail = int(rng.integers(1, period)) if partial and period > 1 else 0
    minimum_repeats = max(3, 8 if period == 1 else 3)
    maximum_repeats = (30 - tail) // period
    repeats = int(rng.integers(minimum_repeats, maximum_repeats + 1))
    return repeats * period + tail


def _sample_dimensions(rng: np.random.Generator, period: tuple[int, int]) -> tuple[int, int]:
    row_period, column_period = period
    boundary = float(rng.random())
    if boundary < 0.36:
        row_partial = column_partial = False
    elif boundary < 0.78:
        available = [axis for axis, value in enumerate(period) if value > 1]
        chosen = available[int(rng.integers(len(available)))]
        row_partial = chosen == 0
        column_partial = chosen == 1
    else:
        row_partial = row_period > 1 and rng.random() < 0.6
        column_partial = column_period > 1 and rng.random() < 0.6
    return (_sample_size(rng, row_period, row_partial),
            _sample_size(rng, column_period, column_partial))


def _sample_mask(rng: np.random.Generator, output: np.ndarray,
                 period: tuple[int, int], split: str) -> np.ndarray:
    height, width = output.shape
    mask = np.zeros((height, width), dtype=bool)
    style = int(rng.integers(10))
    if style in (0, 9):
        rate = float(rng.uniform(0.09, 0.26 if split == "train" else 0.33))
        mask |= rng.random(output.shape) < rate
    if style in (1, 9):
        for _ in range(int(rng.integers(1, 5))):
            ph = int(rng.integers(2, min(6, height) + 1))
            pw = int(rng.integers(2, min(6, width) + 1))
            row = int(rng.integers(height - ph + 1))
            col = int(rng.integers(width - pw + 1))
            mask[row:row + ph, col:col + pw] = True
    if style == 2:
        rows = rng.choice(height, size=int(rng.integers(1, min(3, height) + 1)), replace=False)
        mask[rows, :] = True
    if style == 3:
        cols = rng.choice(width, size=int(rng.integers(1, min(3, width) + 1)), replace=False)
        mask[:, cols] = True
    if style == 4:
        rs = int(rng.integers(2, 7))
        cs = int(rng.integers(2, 7))
        mask[int(rng.integers(rs))::rs, int(rng.integers(cs))::cs] = True
    if style == 5:
        spacing = int(rng.integers(3, 9))
        direction = -1 if rng.random() < 0.5 else 1
        rr = np.arange(height)[:, None]
        cc = np.arange(width)[None, :]
        mask |= (rr + direction * cc + int(rng.integers(spacing))) % spacing == 0
    if style == 6:
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            for _ in range(int(rng.integers(5, 19))):
                mask[row, col] = True
                step = int(rng.integers(4))
                row = max(0, min(height - 1, row + (step == 0) - (step == 1)))
                col = max(0, min(width - 1, col + (step == 2) - (step == 3)))
                if rng.random() < 0.22:
                    mask[max(0, row - 1):min(height, row + 2), col] = True
    if style == 7:
        for _ in range(int(rng.integers(4, 12))):
            length = int(rng.integers(2, 6))
            if rng.random() < 0.5:
                row = int(rng.integers(height))
                col = int(rng.integers(max(1, width - length + 1)))
                mask[row, col:col + length] = True
            else:
                row = int(rng.integers(max(1, height - length + 1)))
                col = int(rng.integers(width))
                mask[row:row + length, col] = True
    if style == 8:
        corner = int(rng.integers(4))
        row = 0 if corner < 2 else height - 1
        col = 0 if corner % 2 == 0 else width - 1
        ph = int(rng.integers(2, min(7, height) + 1))
        pw = int(rng.integers(2, min(7, width) + 1))
        rows = range(0, ph) if row == 0 else range(height - ph, height)
        cols = range(0, pw) if col == 0 else range(width - pw, width)
        for r in rows:
            for c in cols:
                if rng.random() < 0.72:
                    mask[r, c] = True
        mask[row, col] = True
    if style in (1, 6, 7, 8, 9) and rng.random() < 0.48:
        mask |= rng.random(output.shape) < float(rng.uniform(0.015, 0.085))

    colors = [int(x) for x in np.unique(output)]
    one_color = rng.random() < 0.23
    selected_color = colors[int(rng.integers(2))] if one_color else None
    if one_color:
        mask &= output == selected_color

    maximum = int(0.45 * height * width)
    locations = np.argwhere(mask)
    if len(locations) > maximum:
        chosen = locations[rng.choice(len(locations), size=maximum, replace=False)]
        mask[:] = False
        mask[chosen[:, 0], chosen[:, 1]] = True

    # Preserve at least two trustworthy observations for every tile phase.
    row_period, column_period = period
    for row_phase in range(row_period):
        for column_phase in range(column_period):
            locations = np.argwhere(mask[row_phase::row_period,
                                         column_phase::column_period])
            total = len(range(row_phase, height, row_period)) * len(range(column_phase, width, column_period))
            missing = max(0, 2 - (total - len(locations)))
            if missing:
                chosen = locations[rng.choice(len(locations), size=missing, replace=False)]
                mask[row_phase + chosen[:, 0] * row_period,
                     column_phase + chosen[:, 1] * column_period] = False

    # Ensure the chosen damage role is observable; ordinary pairs restore both colors.
    desired = [selected_color] if one_color else colors
    for color in desired:
        if np.any(mask & (output == color)):
            continue
        known_by_phase = np.zeros(period, dtype=int)
        for r, c in np.argwhere(~mask):
            known_by_phase[r % row_period, c % column_period] += 1
        eligible = [(int(r), int(c)) for r, c in np.argwhere((~mask) & (output == color))
                    if known_by_phase[r % row_period, c % column_period] > 2]
        if not eligible:
            raise AssertionError("No safe corruption cell of a needed color")
        mask[eligible[int(rng.integers(len(eligible)))]] = True
    return mask


def _pair_hash(input_grid: np.ndarray, output_grid: np.ndarray) -> str:
    raw = bytes((input_grid.shape[0], input_grid.shape[1]))
    raw += input_grid.astype(np.uint8).tobytes()
    raw += output_grid.astype(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def _generate_pair(rng: np.random.Generator, corruption: int,
                   split: str, tile: np.ndarray | None,
                   palette: tuple[int, int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    for _ in range(120):
        chosen_tile = tile.copy() if tile is not None else _sample_tile(rng)
        period = chosen_tile.shape
        height, width = _sample_dimensions(rng, period)
        if palette is None:
            available = [color for color in range(10) if color != corruption]
            picked = rng.choice(available, size=2, replace=False)
            chosen_palette = (int(picked[0]), int(picked[1]))
        else:
            chosen_palette = palette
        row_shift = int(rng.integers(period[0]))
        column_shift = int(rng.integers(period[1]))
        accent = chosen_tile[(np.arange(height)[:, None] + row_shift) % period[0],
                             (np.arange(width)[None, :] + column_shift) % period[1]]
        output = np.where(accent, chosen_palette[1], chosen_palette[0]).astype(np.int8)
        mask = _sample_mask(rng, output, period, split)
        input_grid = output.copy()
        input_grid[mask] = corruption
        try:
            recovered, recovered_period = _recover_texture(input_grid, corruption)
        except AssertionError:
            continue
        if recovered_period != period or not np.array_equal(recovered, output):
            continue
        if _pair_hash(input_grid, output) in _SOURCE_PAIR_HASHES:
            continue
        return input_grid, output, chosen_tile, chosen_palette
    raise RuntimeError("Unable to generate an unambiguous pair")


def _validate_grid(grid: np.ndarray, name: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{name} must be a NumPy array")
    _require(grid.ndim == 2, f"{name} must be two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{name} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must use integer colors")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{name} has illegal colors")


def _inspect_pair(pair: dict) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "Malformed pair")
    source = pair["input"]
    target = pair["output"]
    _validate_grid(source, "input")
    _validate_grid(target, "output")
    _require(source.shape == target.shape, "Dimensions changed")
    input_colors = {int(x) for x in np.unique(source)}
    output_colors = {int(x) for x in np.unique(target)}
    _require(len(input_colors) == 3 and len(output_colors) == 2,
             "A pair needs two texture colors and one mark color")
    changed = source != target
    _require(bool(np.any(changed)), "Pair has no restoration")
    marks = np.unique(source[changed])
    _require(len(marks) == 1, "Changes use more than one mark color")
    corruption = int(marks[0])
    _require(corruption not in output_colors and input_colors == output_colors | {corruption},
             "Inconsistent mark role")
    _require(np.array_equal(source == corruption, changed),
             "Exactly the marked cells must change")
    period = _fundamental_period(target)
    _require(period[0] <= 6 and period[1] <= 6 and period[0] * period[1] <= _MAX_TILE_AREA,
             "Unsupported texture period")
    tile = target[:period[0], :period[1]]
    _require(len(np.unique(tile)) == 2, "Fundamental tile is not two-colored")
    for r in range(period[0]):
        for c in range(period[1]):
            observations = source[r::period[0], c::period[1]]
            _require(int(np.count_nonzero(observations != corruption)) >= 2,
                     "Tile phase lacks repeated observations")
    recovered, recovered_period = _recover_texture(source, corruption)
    _require(recovered_period == period and np.array_equal(recovered, target),
             "Input does not uniquely determine the output texture")
    _require(_pair_hash(source, target) not in _SOURCE_PAIR_HASHES,
             "Pair copies a source example")
    return {
        "corruption": corruption,
        "period": period,
        "palette": tuple(sorted(output_colors)),
        "exact": target.shape[0] % period[0] == 0 and target.shape[1] % period[1] == 0,
        "restored_colors": {int(x) for x in np.unique(target[changed])},
        "tile_counts": tuple(int(x) for x in np.unique(tile, return_counts=True)[1]),
        "mask": changed,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "Illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            input_signature = _grid_signature(pair["input"])
            pair_signature = (input_signature, _grid_signature(pair["output"]))
            _require(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                     "Duplicate generated pair")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            metadata[split].append(info)
    corruption = metadata["train"][0]["corruption"]
    _require(all(info["corruption"] == corruption
                 for split in ("train", "test") for info in metadata[split]),
             "Mark color must stay fixed throughout a puzzle")
    _require(any(len(info["restored_colors"]) == 2 for info in metadata["train"]),
             "Training must show restoration to both texture colors")
    # The marker is inferred from training changes, then applied to test inputs.
    for pair in puzzle["test"]:
        predicted, _ = _recover_texture(pair["input"], corruption)
        _require(np.array_equal(predicted, pair["output"]),
                 "Test is not inferable from the learned mark and its input")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    corruption = int(rng.integers(10))  # The one convention learned across pairs.
    quiet = rng.random() < 0.29
    shared_tile = _sample_tile(rng) if quiet else None
    shared_palette = None
    if quiet:
        available = [color for color in range(10) if color != corruption]
        picked = rng.choice(available, size=2, replace=False)
        shared_palette = (int(picked[0]), int(picked[1]))
    puzzle = {"train": [], "test": []}
    previous_tiles = []
    previous_palettes = []
    seen_inputs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            accepted = False
            for _ in range(140):
                child_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
                pair_rng = np.random.default_rng(child_seed)
                if quiet:
                    tile, palette = shared_tile, shared_palette
                else:
                    tile = (previous_tiles[int(pair_rng.integers(len(previous_tiles)))].copy()
                            if previous_tiles and pair_rng.random() < 0.18 else None)
                    palette = (previous_palettes[int(pair_rng.integers(len(previous_palettes)))]
                               if previous_palettes and pair_rng.random() < 0.27 else None)
                source, target, chosen_tile, chosen_palette = _generate_pair(
                    pair_rng, corruption, split, tile, palette)
                signature = _grid_signature(source)
                if signature in seen_inputs:
                    continue
                if split == "train" and index == train_count - 1:
                    if not any(len(np.unique(p["output"][p["input"] != p["output"]])) == 2
                               for p in puzzle["train"]):
                        if len(np.unique(target[source != target])) != 2:
                            continue
                puzzle[split].append({"input": source, "output": target})
                previous_tiles.append(chosen_tile)
                previous_palettes.append(chosen_palette)
                seen_inputs.add(signature)
                accepted = True
                break
            if not accepted:
                raise RuntimeError("Unable to generate enough distinct valid pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _mask_features(mask: np.ndarray) -> dict:
    height, width = mask.shape
    full_row = bool(np.any(np.all(mask, axis=1)))
    full_column = bool(np.any(np.all(mask, axis=0)))
    corner = bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1])
    isolated = False
    short_run = False
    for r, c in np.argwhere(mask):
        neighbors = (int(r > 0 and mask[r - 1, c]) + int(r + 1 < height and mask[r + 1, c])
                     + int(c > 0 and mask[r, c - 1]) + int(c + 1 < width and mask[r, c + 1]))
        if neighbors == 0:
            isolated = True
        if (c + 1 < width and mask[r, c + 1]) or (r + 1 < height and mask[r + 1, c]):
            short_run = True
    broken_runs = False
    for lines in (mask, mask.T):
        for line in lines:
            lengths = []
            run = 0
            for value in line:
                if value:
                    run += 1
                elif run:
                    lengths.append(run)
                    run = 0
            if run:
                lengths.append(run)
            if sum(2 <= length <= 5 for length in lengths) >= 2:
                broken_runs = True
    jagged = False
    seen = set()
    for row, column in np.argwhere(mask):
        start = (int(row), int(column))
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        for r, c in component:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < height and 0 <= nc < width
                        and mask[nr, nc] and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    component.append((nr, nc))
        if len(component) >= 5:
            rows = [r for r, _ in component]
            columns = [c for _, c in component]
            if len(component) < (max(rows) - min(rows) + 1) * (max(columns) - min(columns) + 1):
                jagged = True
    return {"full_row": full_row, "full_column": full_column,
            "corner": corner, "isolated": isolated, "cluster": short_run,
            "mixed": isolated and short_run, "broken_runs": broken_runs,
            "jagged": jagged}


def _canonical_tile(output: np.ndarray, period: tuple[int, int]) -> tuple:
    tile = output[:period[0], :period[1]]
    return min(
        tuple(int(x) for x in np.roll(np.roll(tile, r, axis=0), c, axis=1).flat)
        for r in range(period[0]) for c in range(period[1])
    )


def self_test() -> None:
    a = generate_puzzle(73)
    _require(_puzzles_equal(a, generate_puzzle(73)), "Automatic generation is not deterministic")
    b = generate_puzzle(911, num_train=7, num_test=3)
    _require(_puzzles_equal(b, generate_puzzle(911, 7, 3)), "Explicit generation is not deterministic")
    _require(not _puzzles_equal(b, generate_puzzle(912, 7, 3)), "Different seeds repeated")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(2000 + index, train, test)
        _require((len(p["train"]), len(p["test"])) == (train, test), "Boundary counts changed")
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(3000 + index, train, test)
        _require((train is None or len(p["train"]) == train)
                 and (test is None or len(p["test"]) == test), "Partial counts changed")
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
    )):
        try:
            generate_puzzle(4000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    train_counts, test_counts = set(), set()
    puzzle_signatures, shape_signatures, period_signatures = set(), set(), set()
    palettes, colors, motif_counts = set(), set(), set()
    within = {name: 0 for name in (
        "multiple_motifs", "multiple_periods", "multiple_shapes",
        "same_tile_new_dimensions", "one_and_both_damage")}
    measurements = {name: 0 for name in (
        "train_block_or_rich", "test_isolated", "balanced", "stripe", "wide_rectangular",
        "exact_test", "partial_test", "first_train_exact", "first_train_partial",
        "shared_period", "shared_palette", "one_color_damage", "both_color_damage",
        "full_row", "full_column", "corner", "isolated", "cluster", "mixed",
        "broken_runs", "jagged")}
    total_pairs = 0
    for seed in range(100, 142):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzle_signatures.add(_puzzle_signature(p))
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        infos = {split: [_inspect_pair(pair) for pair in p[split]] for split in ("train", "test")}
        first = infos["train"][0]
        measurements["first_train_exact" if first["exact"] else "first_train_partial"] += 1
        all_infos = infos["train"] + infos["test"]
        all_pairs = p["train"] + p["test"]
        motif_keys = [(info["period"], _canonical_tile(pair["output"], info["period"]))
                      for pair, info in zip(all_pairs, all_infos)]
        colored_tiles = [(info["palette"], motif_key)
                         for info, motif_key in zip(all_infos, motif_keys)]
        shapes = [pair["input"].shape for pair in all_pairs]
        within["multiple_motifs"] += int(len(set(motif_keys)) > 1)
        within["multiple_periods"] += int(len({info["period"] for info in all_infos}) > 1)
        within["multiple_shapes"] += int(len(set(shapes)) > 1)
        within["same_tile_new_dimensions"] += int(any(
            colored_tiles[i] == colored_tiles[j] and shapes[i] != shapes[j]
            for i in range(len(all_pairs)) for j in range(i + 1, len(all_pairs))))
        within["one_and_both_damage"] += int(
            any(len(info["restored_colors"]) == 1 for info in all_infos)
            and any(len(info["restored_colors"]) == 2 for info in all_infos))
        if len({x["period"] for x in all_infos}) < len(all_infos):
            measurements["shared_period"] += 1
        if len({x["palette"] for x in all_infos}) < len(all_infos):
            measurements["shared_palette"] += 1
        mark = infos["train"][0]["corruption"]
        for split in ("train", "test"):
            for pair, info in zip(p[split], infos[split]):
                total_pairs += 1
                predicted, _ = _recover_texture(pair["input"], mark)
                _require(np.array_equal(predicted, pair["output"]), "Training-derived mark failed")
                shape_signatures.add(pair["input"].shape)
                period_signatures.add(info["period"])
                palettes.add(info["palette"])
                colors.add(mark)
                tile_counts = info["tile_counts"]
                motif_counts.add((info["period"], tile_counts))
                minor_count = min(tile_counts)
                if split == "train" and minor_count >= 3:
                    measurements["train_block_or_rich"] += 1
                if split == "test" and minor_count == 1:
                    measurements["test_isolated"] += 1
                if tile_counts[0] == tile_counts[1]:
                    measurements["balanced"] += 1
                if 1 in info["period"]:
                    measurements["stripe"] += 1
                if sorted(info["period"]) == [2, 6]:
                    measurements["wide_rectangular"] += 1
                if split == "test":
                    measurements["exact_test" if info["exact"] else "partial_test"] += 1
                measurements["one_color_damage" if len(info["restored_colors"]) == 1
                             else "both_color_damage"] += 1
                for name, present in _mask_features(info["mask"]).items():
                    measurements[name] += int(present)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Counts lack diversity")
    _require(len(puzzle_signatures) == 42 and len(shape_signatures) >= 35
             and len(period_signatures) >= 15 and len(motif_counts) >= 30
             and len(palettes) >= 20 and len(colors) >= 7, "Structure lacks diversity")
    floors = {
        "train_block_or_rich": 20, "test_isolated": 5, "balanced": 5,
        "stripe": 5, "wide_rectangular": 3, "exact_test": 10, "partial_test": 10,
        "first_train_exact": 5, "first_train_partial": 5, "shared_period": 5,
        "shared_palette": 5, "one_color_damage": 10, "both_color_damage": 30,
        "full_row": 2, "full_column": 2, "corner": 5, "isolated": 10,
        "cluster": 10, "mixed": 5, "broken_runs": 3, "jagged": 5,
    }
    for name, floor in floors.items():
        _require(measurements[name] >= floor, f"Insufficient review coverage: {name}")
    within_floors = {"multiple_motifs": 15, "multiple_periods": 15,
                    "multiple_shapes": 42, "same_tile_new_dimensions": 8,
                    "one_and_both_damage": 20}
    for name, floor in within_floors.items():
        _require(within[name] >= floor, f"Insufficient within-puzzle coverage: {name}")
