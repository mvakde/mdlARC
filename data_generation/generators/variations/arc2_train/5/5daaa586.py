import hashlib
import numpy as np


_SIDES = ("top", "bottom", "left", "right")
_SOURCE_INPUT_DIGESTS = {
    ((12, 12), "baa34aa40af5bccdab7411824c6f538f5e379b48c4d97c0ab9ecac14958c79e4"),
    ((22, 22), "60b1bf2f9445726c154720ee1e648c124f00a9dcf960fefec503539ad2a0bc0c"),
    ((14, 15), "3593ec308426e3a0f37e815734247d73bc458fdecdc6d9d36078069342247d6d"),
    ((16, 15), "1e029e2d64d7e26945ca2e497b283e192efd61c78924ea08ffa0f78edf347bba"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _coerce_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _coerce_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _coerce_count(num_train, "num_train", 2, 10)
    test = _coerce_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _is_source_input(grid: np.ndarray) -> bool:
    signature = (grid.shape, hashlib.sha256(np.asarray(grid, dtype=np.int8).tobytes()).hexdigest())
    return signature in _SOURCE_INPUT_DIGESTS


def _grid_key(grid: np.ndarray) -> tuple:
    array = np.ascontiguousarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _sample_crop_dimension(rng: np.random.Generator) -> int:
    probabilities = (0.28, 0.44, 0.24, 0.04)
    band = int(rng.choice(4, p=probabilities))
    bounds = ((6, 9), (10, 15), (16, 23), (24, 28))[band]
    return int(rng.integers(bounds[0], bounds[1] + 1))


def _sample_margins(rng: np.random.Generator, crop_size: int) -> tuple[int, int]:
    available = 30 - crop_size
    before = int(rng.integers(1, min(available - 1, 8) + 1))
    after = int(rng.integers(1, min(available - before, 8) + 1))
    return before, after


def _candidate_input(rng: np.random.Generator, side: str,
                     force_repeat: bool, short_stroke: bool) -> np.ndarray:
    crop_height = _sample_crop_dimension(rng)
    crop_width = _sample_crop_dimension(rng)
    top, bottom_margin = _sample_margins(rng, crop_height)
    left, right_margin = _sample_margins(rng, crop_width)
    bottom = top + crop_height - 1
    right = left + crop_width - 1
    height = bottom + bottom_margin + 1
    width = right + right_margin + 1

    background, top_color, bottom_color, left_color, right_color = (
        int(x) for x in rng.choice(10, size=5, replace=False)
    )
    colors = {"top": top_color, "bottom": bottom_color, "left": left_color, "right": right_color}
    active_color = colors[side]
    grid = np.full((height, width), background, dtype=np.int8)
    grid[top, :] = top_color
    grid[bottom, :] = bottom_color
    grid[:, left] = left_color
    grid[:, right] = right_color

    # Every crossing independently retains one of its two incident guides.
    for row, horizontal_color in ((top, top_color), (bottom, bottom_color)):
        for col, vertical_color in ((left, left_color), (right, right_color)):
            grid[row, col] = horizontal_color if rng.integers(2) else vertical_color

    horizontal = side in ("top", "bottom")
    tracks = np.arange(left + 1, right) if horizontal else np.arange(top + 1, bottom)
    depth_limit = (bottom - top - 1) if horizontal else (right - left - 1)
    max_tracks = min(len(tracks) - 1, max(3, int(np.ceil(len(tracks) * 0.82))))
    track_count = 2 if short_stroke else int(rng.integers(2, max_tracks + 1))
    selected = [int(x) for x in rng.choice(tracks, size=track_count, replace=False)]

    arrangement = 0 if short_stroke else int(rng.integers(4))
    repeated_count = (0, 1, max(2, track_count // 2), track_count)[arrangement]
    if force_repeat:
        repeated_count = max(1, repeated_count)
    repeated_count = min(track_count, repeated_count)
    repeated_tracks = set(int(x) for x in rng.choice(selected, size=repeated_count, replace=False))
    anchor = int(rng.choice(tuple(sorted(repeated_tracks)))) if force_repeat else None
    short_far_track = int(rng.choice(selected)) if short_stroke else None
    for track in selected:
        marker_count = 1
        if track in repeated_tracks:
            marker_count = int(rng.integers(2, min(4, depth_limit) + 1))
        if short_stroke:
            depths = (2,) if track == short_far_track else (int(rng.integers(1, 3)),)
        elif track == anchor:
            far = int(rng.integers(3, depth_limit + 1))
            near = int(rng.integers(1, far - 1))
            depths = (near, far)
        else:
            depths = rng.choice(np.arange(1, depth_limit + 1), size=marker_count, replace=False)
        for depth_value in depths:
            depth = int(depth_value)
            if side == "top":
                row, col = top + depth, track
            elif side == "bottom":
                row, col = bottom - depth, track
            elif side == "left":
                row, col = track, left + depth
            else:
                row, col = track, right - depth
            grid[row, col] = active_color

    exterior = []
    for row in range(height):
        if row in (top, bottom):
            continue
        for col in range(width):
            if col in (left, right):
                continue
            if not (top < row < bottom and left < col < right):
                exterior.append((row, col))
    clutter = int(rng.integers(4))
    if clutter == 0:
        exterior_count = 0
    elif clutter == 1:
        exterior_count = int(rng.integers(1, min(4, len(exterior)) + 1))
    elif clutter == 2:
        exterior_count = int(rng.integers(5, min(12, len(exterior)) + 1)) if len(exterior) >= 5 else len(exterior)
    else:
        exterior_count = int(rng.integers(13, min(25, len(exterior)) + 1)) if len(exterior) >= 13 else len(exterior)
    if exterior_count:
        for index in rng.choice(len(exterior), size=exterior_count, replace=False):
            row, col = exterior[int(index)]
            grid[row, col] = active_color
    return grid


def _analyze_input(grid: np.ndarray) -> dict:
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(counts[background] > max(int(x) for i, x in enumerate(counts) if i != background), "background is not uniquely modal")
    rows = tuple(int(x) for x in np.flatnonzero(np.all(grid != background, axis=1)))
    cols = tuple(int(x) for x in np.flatnonzero(np.all(grid != background, axis=0)))
    _require(len(rows) == len(cols) == 2, "two horizontal and vertical guides are required")
    top, bottom = rows
    left, right = cols
    height, width = grid.shape
    _require(0 < top < bottom < height - 1 and 0 < left < right < width - 1, "guides need outer margins")
    _require(bottom - top >= 5 and right - left >= 5, "window needs visible interior")
    non_vertical_cols = [i for i in range(width) if i not in cols]
    non_horizontal_rows = [i for i in range(height) if i not in rows]
    horizontal_colors = []
    vertical_colors = []
    for row in rows:
        values = np.unique(grid[row, non_vertical_cols])
        _require(len(values) == 1, "horizontal guide is not uniform")
        horizontal_colors.append(int(values[0]))
    for col in cols:
        values = np.unique(grid[non_horizontal_rows, col])
        _require(len(values) == 1, "vertical guide is not uniform")
        vertical_colors.append(int(values[0]))
    palette = (background, *horizontal_colors, *vertical_colors)
    _require(len(set(palette)) == 5, "background and guides need five distinct colors")
    crossings = []
    for row_index, row in enumerate(rows):
        for col_index, col in enumerate(cols):
            value = int(grid[row, col])
            _require(value in (horizontal_colors[row_index], vertical_colors[col_index]), "invalid crossing color")
            crossings.append("H" if value == horizontal_colors[row_index] else "V")
    off_guide = np.ones(grid.shape, dtype=bool)
    off_guide[list(rows), :] = False
    off_guide[:, list(cols)] = False
    marker_mask = off_guide & (grid != background)
    marker_colors = np.unique(grid[marker_mask])
    _require(len(marker_colors) == 1, "one off-guide marker identity is required")
    active_color = int(marker_colors[0])
    _require(active_color in palette[1:], "marker must match a guide")
    side = dict(zip(palette[1:], _SIDES))[active_color]
    marker_coords = np.argwhere(marker_mask)
    inside = marker_coords[(marker_coords[:, 0] > top) & (marker_coords[:, 0] < bottom) & (marker_coords[:, 1] > left) & (marker_coords[:, 1] < right)]
    exterior_count = len(marker_coords) - len(inside)
    _require(len(inside) >= 2, "at least two interior markers are required")
    tracks = {}
    track_depths = {}
    for row, col in inside:
        track = int(col) if side in ("top", "bottom") else int(row)
        tracks[track] = tracks.get(track, 0) + 1
        depth = (int(row) - top if side == "top" else
                 bottom - int(row) if side == "bottom" else
                 int(col) - left if side == "left" else right - int(col))
        track_depths.setdefault(track, []).append(depth)
    available_tracks = (right - left - 1) if side in ("top", "bottom") else (bottom - top - 1)
    _require(2 <= len(tracks) < available_tracks, "markers need at least two tracks and one empty track")
    _require(set(int(x) for x in np.unique(grid)) == set(palette), "unexpected input color")
    return {
        "background": background, "palette": palette, "rows": rows, "cols": cols,
        "crossings": tuple(crossings), "side": side, "active_color": active_color,
        "inside": inside.copy(), "exterior_count": exterior_count, "tracks": tracks,
        "track_depths": track_depths,
        "crop_shape": (bottom - top + 1, right - left + 1),
    }


def _expected_output(grid: np.ndarray, info: dict) -> np.ndarray:
    top, bottom = info["rows"]
    left, right = info["cols"]
    side = info["side"]
    color = info["active_color"]
    inside = info["inside"]
    output = grid[top:bottom + 1, left:right + 1].copy()
    if side in ("top", "bottom"):
        for col in info["tracks"]:
            marker_rows = inside[inside[:, 1] == col, 0]
            if side == "top":
                output[:int(marker_rows.max()) - top + 1, col - left] = color
            else:
                output[int(marker_rows.min()) - top:, col - left] = color
    else:
        for row in info["tracks"]:
            marker_cols = inside[inside[:, 0] == row, 1]
            if side == "left":
                output[row - top, :int(marker_cols.max()) - left + 1] = color
            else:
                output[row - top, int(marker_cols.min()) - left:] = color
    return output


def _added_count(grid: np.ndarray, output: np.ndarray, info: dict) -> int:
    top, bottom = info["rows"]
    left, right = info["cols"]
    return int(np.count_nonzero(output != grid[top:bottom + 1, left:right + 1]))


def _challenging(info: dict) -> bool:
    return (info["exterior_count"] >= 5 or len(set(info["crossings"])) == 2
            or max(info["tracks"].values()) >= 3 or max(info["crop_shape"]) >= 16)


def _generate_pair(rng: np.random.Generator, regime: str, side: str,
                   force_repeat: bool, allow_short: bool) -> tuple[np.ndarray, np.ndarray]:
    short_stroke = allow_short and (regime == "source_train" or rng.integers(8) == 0)
    for _ in range(512):
        grid = _candidate_input(rng, side, force_repeat, short_stroke)
        if _is_source_input(grid):
            continue
        try:
            info = _analyze_input(grid)
        except AssertionError:
            continue
        if info["side"] != side or (force_repeat and max(info["tracks"].values()) < 2):
            continue
        output = _expected_output(grid, info)
        added = _added_count(grid, output, info)
        if added < (1 if allow_short else 3):
            continue
        if regime == "target_test" and not _challenging(info):
            continue
        return grid, output
    raise RuntimeError("failed to construct a valid pair")


def _choose_sides(rng: np.random.Generator, seed: int, train_count: int, test_count: int) -> tuple[list[str], list[str]]:
    mode = seed % 5
    if mode == 0:
        train = ["top", "right"] if train_count == 2 else ["top", "bottom", "right"]
        choices = ("top", "bottom", "right")
        test = ["left"] * test_count
    else:
        held_out = str(rng.choice(_SIDES)) if mode == 1 else None
        choices = tuple(side for side in _SIDES if side != held_out)
        horizontal = str(rng.choice(tuple(side for side in choices if side in ("left", "right"))))
        vertical = str(rng.choice(tuple(side for side in choices if side in ("top", "bottom"))))
        train = [horizontal, vertical]
        if mode == 1:
            test = [held_out] * test_count
        elif mode == 2:
            test = []
        elif mode == 3:
            test = list(str(x) for x in rng.choice(_SIDES, size=test_count))
            if test_count >= 2 and len(set(test)) == 1:
                test[-1] = str(rng.choice(tuple(side for side in _SIDES if side != test[0])))
        else:
            test = list(str(x) for x in rng.choice(_SIDES, size=test_count))
    while len(train) < train_count:
        train.append(str(rng.choice(choices)))
    train = [train[int(i)] for i in rng.permutation(len(train))]
    if mode == 2:
        seen_sides = tuple(side for side in _SIDES if side in train)
        test = list(str(x) for x in rng.choice(seen_sides, size=test_count))
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed_value = _coerce_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    train_sides, test_sides = _choose_sides(rng, seed_value, train_count, test_count)
    pair_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (split, side) in enumerate([("train", side) for side in train_sides] + [("test", side) for side in test_sides]):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        regime = "source_train" if split == "train" else "target_test"
        force_repeat = split == "train" and index == 0
        allow_short = split == "test" or (split == "train" and train_count >= 4 and index == train_count - 1 and seed_value % 7 == 0)
        for _ in range(32):
            input_grid, output_grid = _generate_pair(pair_rng, regime, side, force_repeat, allow_short)
            key = _grid_key(input_grid)
            if key not in seen:
                seen.add(key)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                break
        else:
            raise RuntimeError("failed to generate distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, name: str) -> None:
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{name} must be a 2D NumPy array")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must have an integer dtype")
    _require(all(1 <= size <= 30 for size in grid.shape), f"{name} size is outside ARC bounds")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9, f"{name} has illegal colors")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle requires train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            name = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{name} has incorrect fields")
            grid, output = pair["input"], pair["output"]
            _validate_grid(grid, name + ".input")
            _validate_grid(output, name + ".output")
            input_key = _grid_key(grid)
            pair_key = (input_key, _grid_key(output))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair or input")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _is_source_input(grid), "source input copied")
            info = _analyze_input(grid)
            expected = _expected_output(grid, info)
            _require(np.array_equal(output, expected), f"{name} violates the crop-and-fill rule")
            _require(output.shape == info["crop_shape"], f"{name} crop size is wrong")
            _require(_added_count(grid, output, info) >= 1, f"{name} has no completion")
            _require(set(int(x) for x in np.unique(output)) == set(info["palette"]), f"{name} lost a guide color")
            if split == "test":
                _require(_challenging(info), f"{name} needs a supported test challenge")
            analyses[split].append(info)
    train_sides = {info["side"] for info in analyses["train"]}
    _require(bool(train_sides & {"top", "bottom"}) and bool(train_sides & {"left", "right"}), "training must show both orientations")
    _require(any(any(len(depths) >= 2 and max(depths) - min(depths) >= 2
                     for depths in info["track_depths"].values())
                 for info in analyses["train"]), "training must demonstrate separated repeated markers")
    _require(all((info["side"] in train_sides or
                  (info["side"] in ("top", "bottom") and bool(train_sides & {"top", "bottom"})) or
                  (info["side"] in ("left", "right") and bool(train_sides & {"left", "right"})))
                 for info in analyses["test"]), "test orientation lacks training evidence")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split]) for key in ("input", "output"))
               for split in ("train", "test"))


def _independent_prediction(grid: np.ndarray) -> np.ndarray:
    # Reconstruct from the observable guides and markers without generation metadata.
    info = _analyze_input(grid)
    top, bottom = info["rows"]
    left, right = info["cols"]
    answer = grid[top:bottom + 1, left:right + 1].copy()
    color = info["active_color"]
    for row, col in info["inside"]:
        if info["side"] == "top":
            answer[:row - top + 1, col - left] = color
        elif info["side"] == "bottom":
            answer[row - top:, col - left] = color
        elif info["side"] == "left":
            answer[row - top, :col - left + 1] = color
        else:
            answer[row - top, col - left:] = color
    return answer


def self_test() -> None:
    train_counts, test_counts = set(), set()
    input_shapes, output_shapes, layouts, palettes = set(), set(), set(), set()
    sides, crossing_patterns, exterior_counts, repeat_maxima, track_counts = set(), set(), set(), set(), set()
    backgrounds, source_shapes = set(), {(12, 12), (22, 22), (14, 15), (16, 15)}
    mixed_puzzles = seen_side_tests = unseen_side_tests = multi_side_tests = 0
    role_swaps = background_changes = 0
    mixed_crossings = all_single = multi_repeated = three_plus = 0
    no_exterior = high_exterior = short_strokes = combinations = 0
    small_windows = large_windows = source_shape_pairs = 0
    margins, aspects, marker_depths = set(), set(), set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_analyze_input(pair["input"]) for pair in puzzle[split]] for split in ("train", "test")}
        all_infos = infos["train"] + infos["test"]
        if len({info["palette"] for info in all_infos}) > 1:
            mixed_puzzles += 1
        if len({info["background"] for info in all_infos}) > 1:
            background_changes += 1
        if any(any(a != b and x in a[1:] and x in b[1:] and a.index(x) != b.index(x)
                   for x in range(10)) for a in (info["palette"] for info in all_infos)
               for b in (info["palette"] for info in all_infos)):
            role_swaps += 1
        train_side_set = {info["side"] for info in infos["train"]}
        test_side_set = {info["side"] for info in infos["test"]}
        seen_side_tests += bool(test_side_set & train_side_set)
        unseen_side_tests += bool(test_side_set - train_side_set)
        multi_side_tests += len(test_side_set) > 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                grid, output = pair["input"], pair["output"]
                _require(np.array_equal(_independent_prediction(grid), output), "independent reconstruction failed")
                input_shapes.add(grid.shape)
                output_shapes.add(output.shape)
                layouts.add((grid.shape, info["rows"], info["cols"], info["side"]))
                palettes.add(info["palette"])
                backgrounds.add(info["background"])
                sides.add(info["side"])
                crossing_patterns.add(info["crossings"])
                exterior_counts.add(info["exterior_count"])
                repeat_maxima.add(max(info["tracks"].values()))
                track_counts.add(len(info["tracks"]))
                marker_depths.update(depth for depths in info["track_depths"].values() for depth in depths)
                mixed = len(set(info["crossings"])) == 2
                repeat_max = max(info["tracks"].values())
                exterior = info["exterior_count"]
                mixed_crossings += mixed
                all_single += repeat_max == 1
                multi_repeated += sum(n >= 2 for n in info["tracks"].values()) >= 2
                three_plus += repeat_max >= 3
                no_exterior += exterior == 0
                high_exterior += exterior >= 10
                short_strokes += _added_count(grid, output, info) <= 2
                combinations += mixed and repeat_max >= 3 and exterior >= 10
                small_windows += max(output.shape) <= 9
                large_windows += max(output.shape) >= 16
                source_shape_pairs += grid.shape in source_shapes
                top, bottom = info["rows"]
                left, right = info["cols"]
                margins.add((top, grid.shape[0] - bottom - 1, left, grid.shape[1] - right - 1))
                aspects.add("tall" if output.shape[0] > 1.4 * output.shape[1]
                            else "wide" if output.shape[1] > 1.4 * output.shape[0]
                            else "balanced")
                total += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity is too low")
    _require(len(input_shapes) >= 80 and len(output_shapes) >= 45, "size diversity is too low")
    _require(len(layouts) >= int(total * 0.8), "layout diversity is too low")
    _require(len(palettes) >= int(total * 0.8) and len(backgrounds) >= 5, "palette diversity is too low")
    _require(mixed_puzzles >= 35 and background_changes >= 30 and role_swaps >= 15,
             "within-puzzle color reassignment is too low")
    _require(sides == set(_SIDES) and len(crossing_patterns) >= 8, "side or crossing diversity is too low")
    _require(0 in exterior_counts and max(exterior_counts) >= 17, "exterior clutter range is too narrow")
    _require(1 in repeat_maxima and max(repeat_maxima) >= 3 and max(track_counts) >= 7, "marker arrangements are too narrow")
    _require(seen_side_tests >= 5 and unseen_side_tests >= 5 and multi_side_tests >= 3, "test-side regimes are too narrow")
    _require(bool(input_shapes & source_shapes), "source canvas shapes were unnecessarily excluded")
    _require(mixed_crossings >= 50 and all_single >= 20 and multi_repeated >= 20
             and three_plus >= 20, "crossing or marker arrangements are too narrow")
    _require(no_exterior >= 10 and high_exterior >= 20 and short_strokes >= 3,
             "clutter or stroke-length diversity is too low")
    _require(min(marker_depths) == 1 and max(marker_depths) >= 18,
             "marker endpoints do not span short and long depths")
    _require(combinations >= 10 and small_windows >= 5 and large_windows >= 20,
             "compatible geometry and marker combinations are too rare")
    _require(source_shape_pairs >= 1 and len(margins) >= int(total * 0.6)
             and aspects == {"tall", "wide", "balanced"}, "geometry diversity is too low")
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(12000 + ntrain * 10 + ntest, ntrain, ntest)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest), "explicit counts ignored")
    for index, args in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(13000 + index, *args)
        if args[0] is not None:
            _require(len(puzzle["train"]) == args[0], "partial train count ignored")
        if args[1] is not None:
            _require(len(puzzle["test"]) == args[1], "partial test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(14000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a = generate_puzzle(15000, 6, 2)
    b = generate_puzzle(15000, 6, 2)
    c = generate_puzzle(15001, 6, 2)
    _require(_puzzles_equal(a, b) and not _puzzles_equal(a, c), "determinism or cross-seed diversity failed")
