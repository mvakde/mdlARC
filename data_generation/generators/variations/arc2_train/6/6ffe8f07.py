"""Compositional rectangular-source ray puzzles."""

import hashlib
from collections import Counter

import numpy as np


_SIDES = ("left", "right", "top", "bottom")
_SOURCE_INPUT_DIGESTS = {
    "4a3a07aef52d387049f3031b4486a02d7e691ef0b91b8221ff66c21afb7a5909",
    "82511bec1a442112a9cb0e78117a7743dba08dcd15fae231afc88451f9ada83e",
    "6543f0e5672e9e98b6cc0d7787fdb58b26a607e6213fc85f59c3f9bc7fb1b2c5",
    "8c1c223553a896a58d3dfc409aa5c43eba275475dbe6dddcc42c64e724eb6fa0",
    "0f887b8a0aa7412eba02188209f14b3fac1d3bfbe5de6974438ed4e9328cf32a",
    "938dd4171af2bc8767c877d25b0e02ba29d9ddc07118f896f7b3df12c6f3d529",
}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _signature(grid):
    return (grid.shape, grid.tobytes())


def _input_digest(grid):
    return hashlib.sha256(
        np.asarray(grid, dtype=np.uint8).tobytes() + bytes(grid.shape)
    ).hexdigest()


def _bounds(grid, color):
    coords = np.argwhere(grid == color)
    if len(coords) == 0:
        return None
    top, left = np.min(coords, axis=0)
    bottom, right = np.max(coords, axis=0)
    top, bottom, left, right = map(int, (top, bottom, left, right))
    if not np.all(grid[top:bottom + 1, left:right + 1] == color):
        return None
    if len(coords) != (bottom - top + 1) * (right - left + 1):
        return None
    return top, bottom, left, right


def _rays(grid, bounds, blocker, beam):
    top, bottom, left, right = bounds
    height, width = grid.shape
    result = grid.copy()
    for row in range(top, bottom + 1):
        for start, stop, step in ((left - 1, -1, -1), (right + 1, width, 1)):
            for col in range(start, stop, step):
                if grid[row, col] == blocker:
                    break
                result[row, col] = beam
    for col in range(left, right + 1):
        for start, stop, step in ((top - 1, -1, -1), (bottom + 1, height, 1)):
            for row in range(start, stop, step):
                if grid[row, col] == blocker:
                    break
                result[row, col] = beam
    return result


def _source_size(rng, teaching=False):
    minimum = 3 if teaching else 2
    if rng.random() < 0.27:
        if rng.random() < 0.5:
            return int(rng.integers(minimum, 5)), int(rng.integers(6, 12))
        return int(rng.integers(6, 11)), int(rng.integers(minimum, 5))
    return int(rng.integers(minimum, 9)), int(rng.integers(minimum, 10))


def _canvas_and_source(rng, source_size, teaching=False, central=False, regime="source_train"):
    sh, sw = source_size
    margin = 5 if teaching else (4 if central else 0)
    min_height = max(9, sh + 2 * margin + (0 if margin else 2))
    min_width = max(9, sw + 2 * margin + (0 if margin else 2))
    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    if teaching or central:
        top = int(rng.integers(margin, height - sh - margin + 1))
        left = int(rng.integers(margin, width - sw - margin + 1))
    else:
        edge_chance = 0.40 if regime == "target_test" else 0.28
        if rng.random() < edge_chance:
            top = int(rng.choice([0, 1, height - sh, max(0, height - sh - 1)]))
        else:
            top = int(rng.integers(0, height - sh + 1))
        if rng.random() < edge_chance:
            left = int(rng.choice([0, 1, width - sw, max(0, width - sw - 1)]))
        else:
            left = int(rng.integers(0, width - sw + 1))
    return height, width, (top, top + sh - 1, left, left + sw - 1)


def _side_space(shape, bounds, side):
    top, bottom, left, right = bounds
    height, width = shape
    if side == "left":
        return left
    if side == "right":
        return width - right - 1
    if side == "top":
        return top
    return height - bottom - 1


def _side_slice(bounds, side, along_start, along_length, distance, thickness):
    top, bottom, left, right = bounds
    if side == "left":
        near = left - distance
        return slice(along_start, along_start + along_length), slice(near - thickness + 1, near + 1)
    if side == "right":
        near = right + distance
        return slice(along_start, along_start + along_length), slice(near, near + thickness)
    if side == "top":
        near = top - distance
        return slice(near - thickness + 1, near + 1), slice(along_start, along_start + along_length)
    near = bottom + distance
    return slice(near, near + thickness), slice(along_start, along_start + along_length)


def _paint(grid, region, color, background):
    rows, cols = region
    if rows.start < 0 or cols.start < 0 or rows.stop > grid.shape[0] or cols.stop > grid.shape[1]:
        return False
    if rows.stop <= rows.start or cols.stop <= cols.start:
        return False
    if not np.all(grid[rows, cols] == background):
        return False
    # Keep same-color rectangles as visually separate components.
    surround = grid[max(0, rows.start - 1):min(grid.shape[0], rows.stop + 1),
                    max(0, cols.start - 1):min(grid.shape[1], cols.stop + 1)]
    if np.any(surround == color):
        return False
    grid[rows, cols] = color
    return True


def _teaching_objects(grid, bounds, roles, rng):
    background, _, blocker, penetrable, _ = roles
    top, bottom, left, right = bounds
    source_height = bottom - top + 1
    distance = int(rng.integers(2, min(5, left)))
    near = left - distance
    thickness = int(rng.integers(1, min(4, near + 1)))
    lane_start = int(rng.integers(top, bottom + 1))
    lane_length = int(rng.integers(1, min(source_height - 1, bottom - lane_start + 1) + 1))
    _paint(grid, _side_slice(bounds, "left", lane_start, lane_length, distance, thickness),
           blocker, background)
    available = grid.shape[1] - right - 1
    pen_distance = int(rng.integers(2, min(5, available + 1)))
    pen_thickness = int(rng.integers(1, min(5, available - pen_distance + 2)))
    _paint(grid, _side_slice(bounds, "right", top - 1, source_height + 2,
                            pen_distance, pen_thickness), penetrable, background)


def _random_side_object(grid, bounds, color, background, rng, *, targeted=True):
    side = _SIDES[int(rng.integers(0, 4))]
    space = _side_space(grid.shape, bounds, side)
    if space == 0:
        return False
    top, bottom, left, right = bounds
    if side in ("left", "right"):
        band_start, band_end, axis_size = top, bottom, grid.shape[0]
    else:
        band_start, band_end, axis_size = left, right, grid.shape[1]
    distance = int(rng.integers(1, space + 1))
    max_thickness = min(7, space - distance + 1)
    thickness = int(rng.integers(1, max_thickness + 1))
    long = rng.random() < 0.35
    max_length = min(12 if long else 6, axis_size)
    length = int(rng.integers(2 if max_length >= 2 else 1, max_length + 1))
    if targeted:
        low = max(0, band_start - length + 1)
        high = min(axis_size - length, band_end)
    else:
        low, high = 0, axis_size - length
    if high < low:
        return False
    along = int(rng.integers(low, high + 1))
    return _paint(grid, _side_slice(bounds, side, along, length, distance, thickness),
                  color, background)


def _behind_objects(grid, bounds, roles, rng):
    background, _, blocker, penetrable, _ = roles
    sides = [side for side in _SIDES if _side_space(grid.shape, bounds, side) >= 4]
    if not sides:
        return False
    side = sides[int(rng.integers(0, len(sides)))]
    top, bottom, left, right = bounds
    start, length = (top, bottom - top + 1) if side in ("left", "right") else (left, right - left + 1)
    block_distance = int(rng.integers(1, 3))
    block_thickness = int(rng.integers(1, 3))
    pen_distance = block_distance + block_thickness + 1
    if pen_distance > _side_space(grid.shape, bounds, side):
        return False
    assert _paint(grid, _side_slice(bounds, side, start, length, block_distance,
                                    block_thickness), blocker, background)
    pen_thickness = min(int(rng.integers(1, 5)),
                        _side_space(grid.shape, bounds, side) - pen_distance + 1)
    return _paint(grid, _side_slice(bounds, side, start, length, pen_distance,
                                    pen_thickness), penetrable, background)


def _make_pair(rng, roles, profile, regime, source_size=None):
    teaching = profile == "teach"
    central = profile == "behind"
    if source_size is None:
        source_size = _source_size(rng, teaching)
    if teaching and min(source_size) < 3:
        source_size = _source_size(rng, True)
    shape = _canvas_and_source(rng, source_size, teaching, central, regime)
    height, width, bounds = shape
    background, source, blocker, penetrable, beam = roles
    grid = np.full((height, width), background, dtype=np.int8)
    top, bottom, left, right = bounds
    grid[top:bottom + 1, left:right + 1] = source
    if teaching:
        _teaching_objects(grid, bounds, roles, rng)
        extras = int(rng.integers(0, 4))
    elif profile == "behind":
        _behind_objects(grid, bounds, roles, rng)
        extras = int(rng.integers(0, 4))
    else:
        extras = {"empty": 0, "stop": 1, "pen": 1,
                  "mixed": int(rng.integers(2, 6)),
                  "crowded": int(rng.integers(6, 11))}[profile]
    for index in range(extras):
        if profile == "stop":
            color = blocker
        elif profile == "pen":
            color = penetrable
        else:
            color = blocker if rng.random() < 0.48 else penetrable
        for _ in range(24):
            if _random_side_object(grid, bounds, color, background, rng,
                                   targeted=rng.random() < 0.78):
                break
    output = _rays(grid, bounds, blocker, beam)
    return grid, output


def _stop_data(grid, bounds, blocker):
    top, bottom, left, right = bounds
    height, width = grid.shape
    stops = {side: [] for side in _SIDES}
    for row in range(top, bottom + 1):
        for side, cells in (("left", range(left - 1, -1, -1)),
                            ("right", range(right + 1, width))):
            distance = next((abs(col - (left if side == "left" else right))
                             for col in cells if grid[row, col] == blocker), None)
            stops[side].append(distance)
    for col in range(left, right + 1):
        for side, cells in (("top", range(top - 1, -1, -1)),
                            ("bottom", range(bottom + 1, height))):
            distance = next((abs(row - (top if side == "top" else bottom))
                             for row in cells if grid[row, col] == blocker), None)
            stops[side].append(distance)
    return stops


def _components(mask):
    visited = np.zeros_like(mask, dtype=bool)
    result = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        visited[row, col] = True
        pending = [(row, col)]
        cells = []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not visited[nr, nc]):
                    visited[nr, nc] = True
                    pending.append((nr, nc))
        result.append(cells)
    return result


def _teaching_evidence(grid, output, roles):
    background, source, blocker, penetrable, beam = roles
    bounds = _bounds(grid, source)
    if bounds is None:
        return False
    top, bottom, left, right = bounds
    stops = _stop_data(grid, bounds, blocker)
    selective = any(None in lanes and any(value is not None for value in lanes)
                    for lanes in stops.values())
    pen_changed = np.argwhere((grid == penetrable) & (output == beam))
    distant_pen = any(
        (top <= row <= bottom and (col < left - 1 or col > right + 1)) or
        (left <= col <= right and (row < top - 1 or row > bottom + 1))
        for row, col in pen_changed
    )
    return (selective and distant_pen and len(pen_changed) > 0
            and np.any((grid == penetrable) & (output == penetrable))
            and np.any((grid == background) & (output == beam)))


def _roles_from_training(train):
    inputs = set().union(*(set(map(int, np.unique(p["input"]))) for p in train))
    outputs = set().union(*(set(map(int, np.unique(p["output"]))) for p in train))
    added = outputs - inputs
    if len(inputs) != 4 or len(added) != 1:
        return []
    beam = next(iter(added))
    candidates = []
    for background in inputs:
        values, counts = np.unique(train[0]["input"], return_counts=True)
        if int(values[int(np.argmax(counts))]) != background or int(np.sum(counts == np.max(counts))) != 1:
            continue
        for source in inputs - {background}:
            for blocker in inputs - {background, source}:
                penetrable = next(iter(inputs - {background, source, blocker}))
                roles = (background, source, blocker, penetrable, beam)
                valid = True
                for pair in train:
                    grid, output = pair["input"], pair["output"]
                    bounds = _bounds(grid, source)
                    if bounds is None or min(bounds[1] - bounds[0] + 1,
                                             bounds[3] - bounds[2] + 1) < 2:
                        valid = False
                        break
                    if not np.array_equal(_rays(grid, bounds, blocker, beam), output):
                        valid = False
                        break
                    if beam in grid or not np.any(grid == background):
                        valid = False
                        break
                if valid:
                    candidates.append(roles)
    return candidates


def _predict_from_demonstrations(train, test_input):
    """Infer the five roles from examples, then check each projected cell."""
    candidates = _roles_from_training(train)
    assert len(candidates) == 1
    _, source, blocker, _, beam = candidates[0]
    bounds = _bounds(test_input, source)
    assert bounds is not None
    top, bottom, left, right = bounds
    result = test_input.copy()
    for row in range(test_input.shape[0]):
        for col in range(test_input.shape[1]):
            if top <= row <= bottom and col < left:
                path = test_input[row, col:left]
            elif top <= row <= bottom and col > right:
                path = test_input[row, right + 1:col + 1]
            elif left <= col <= right and row < top:
                path = test_input[row:top, col]
            elif left <= col <= right and row > bottom:
                path = test_input[bottom + 1:row + 1, col]
            else:
                continue
            if not np.any(path == blocker):
                result[row, col] = beam
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(v) for v in rng.choice(10, size=5, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_sizes = []
    profiles = ("empty", "stop", "pen", "behind", "mixed", "crowded")
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(80):
                if split == "train" and index == 0:
                    profile = "teach"
                else:
                    weights = (0.12, 0.15, 0.15, 0.14, 0.26, 0.18) if split == "train" else (
                        0.14, 0.13, 0.11, 0.21, 0.18, 0.23)
                    profile = str(rng.choice(profiles, p=weights))
                size = None
                if split == "test" and train_sizes and rng.random() < 0.34:
                    size = train_sizes[int(rng.integers(0, len(train_sizes)))]
                grid, output = _make_pair(rng, roles, profile,
                                          "source_train" if split == "train" else "target_test", size)
                signature = _signature(grid)
                if signature in seen or _input_digest(grid) in _SOURCE_INPUT_DIGESTS:
                    continue
                if np.array_equal(grid, output):
                    continue
                if split == "train" and index == 0 and not _teaching_evidence(grid, output, roles):
                    continue
                if split == "train" and index == 1 and train_sizes and (
                    _bounds(grid, roles[1])[1] - _bounds(grid, roles[1])[0] + 1,
                    _bounds(grid, roles[1])[3] - _bounds(grid, roles[1])[2] + 1
                ) == train_sizes[0]:
                    continue
                seen.add(signature)
                puzzle[split].append({"input": grid, "output": output})
                if split == "train":
                    b = _bounds(grid, roles[1])
                    train_sizes.append((b[1] - b[0] + 1, b[3] - b[2] + 1))
                break
            else:
                raise RuntimeError("bounded pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert 0 <= int(np.min(grid)) <= int(np.max(grid)) <= 9
        grid, output = pair["input"], pair["output"]
        assert grid.shape == output.shape and not np.array_equal(grid, output)
        key = _signature(grid)
        pair_key = (key, _signature(output))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        assert _input_digest(grid) not in _SOURCE_INPUT_DIGESTS
    candidates = _roles_from_training(puzzle["train"])
    assert len(candidates) == 1, "training must uniquely establish color roles"
    roles = candidates[0]
    background, source, blocker, penetrable, beam = roles
    assert _teaching_evidence(puzzle["train"][0]["input"],
                              puzzle["train"][0]["output"], roles)
    sizes = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid, output = pair["input"], pair["output"]
            bounds = _bounds(grid, source)
            assert bounds is not None
            top, bottom, left, right = bounds
            assert bottom - top + 1 >= 2 and right - left + 1 >= 2
            assert np.array_equal(output, _rays(grid, bounds, blocker, beam))
            assert np.array_equal(output, _predict_from_demonstrations(
                puzzle["train"], grid.copy()))
            assert set(map(int, np.unique(grid))).issubset({background, source, blocker, penetrable})
            assert beam not in grid
            assert np.any(grid == background)
            assert np.all(output[grid == source] == source)
            assert np.all(output[grid == blocker] == blocker)
            assert np.all(output[grid != output] == beam)
            assert np.all(np.isin(grid[grid != output], [background, penetrable]))
            for color in (blocker, penetrable):
                for cells in _components(grid == color):
                    rows = [r for r, _ in cells]
                    cols = [c for _, c in cells]
                    assert len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            if split == "train":
                sizes.append((bottom - top + 1, right - left + 1))
    assert len(set(sizes)) >= 2


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[split], b[split])) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    role_sets, canvas_sizes, source_sizes = set(), set(), set()
    blocked_side_counts, stop_distances = set(), set()
    profiles = set()
    distant_pen = 0
    adjacent_stop = 0
    no_pen = 0
    edge_source = 0
    within_canvas = 0
    within_source = 0
    within_blocking = 0
    within_objects = 0
    review = Counter()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _roles_from_training(puzzle["train"])[0]
        role_sets.add(roles)
        local_canvas = set()
        local_source = set()
        local_blocking = set()
        local_objects = set()
        for pair in puzzle["train"] + puzzle["test"]:
            grid, output = pair["input"], pair["output"]
            background, source, blocker, penetrable, beam = roles
            bounds = _bounds(grid, source)
            top, bottom, left, right = bounds
            canvas_sizes.add(grid.shape)
            local_canvas.add(grid.shape)
            source_sizes.add((bottom - top + 1, right - left + 1))
            local_source.add((bottom - top + 1, right - left + 1))
            stops = _stop_data(grid, bounds, blocker)
            blocked = sum(any(x is not None for x in v) for v in stops.values())
            blocked_side_counts.add(blocked)
            local_blocking.add(blocked)
            stop_distances.update(x for v in stops.values() for x in v if x is not None)
            review[f"blocked_{blocked}"] += 1
            review["full_side"] += any(v and all(x is not None for x in v) for v in stops.values())
            review["selective_side"] += any(None in v and any(x is not None for x in v) for v in stops.values())
            review["unequal_stops"] += any(len({x for x in v if x is not None}) > 1 for v in stops.values())
            review["far_stop"] += any(x is not None and x >= 5 for v in stops.values() for x in v)
            adjacent_stop += any(x == 1 for v in stops.values() for x in v)
            edge_source += min(top, left, grid.shape[0] - bottom - 1,
                               grid.shape[1] - right - 1) <= 1
            no_pen += not np.any(grid == penetrable)
            bc = _components(grid == blocker)
            pc = _components(grid == penetrable)
            local_objects.add((len(bc), len(pc)))
            review["empty"] += not bc and not pc
            review["single_stopper"] += len(bc) == 1 and not pc
            review["crowded"] += len(bc) + len(pc) >= 5
            review["thick_blocker"] += any(
                min(max(r for r, _ in cells) - min(r for r, _ in cells) + 1,
                    max(c for _, c in cells) - min(c for _, c in cells) + 1) >= 2
                for cells in bc)
            review["large_blocker"] += any(len(cells) >= 16 for cells in bc)
            review["pen_all"] += any(all(output[r, c] == beam for r, c in cells) for cells in pc)
            review["pen_none"] += any(all(output[r, c] == penetrable for r, c in cells) for cells in pc)
            review["pen_partial"] += any(
                0 < sum(output[r, c] == beam for r, c in cells) < len(cells)
                for cells in pc)
            review["small_canvas"] += min(grid.shape) < 18
            review["large_canvas"] += max(grid.shape) >= 26
            review["long_narrow"] += min(bottom - top + 1, right - left + 1) <= 3 and max(
                bottom - top + 1, right - left + 1) >= 6
            review["source_canvas_shape"] += grid.shape in ((19, 19), (19, 18))
            distant_pen += any(
                grid[r, c] == penetrable and output[r, c] == beam and
                ((top <= r <= bottom and (c < left - 1 or c > right + 1)) or
                 (left <= c <= right and (r < top - 1 or r > bottom + 1)))
                for r, c in np.argwhere(grid == penetrable))
            profiles.add((int(np.any(grid == blocker)), int(np.any(grid == penetrable)),
                          sum(any(x is not None for x in v) for v in stops.values())))
            # This predictor sees demonstrations and the input, never this output.
            assert np.array_equal(_predict_from_demonstrations(
                puzzle["train"], grid.copy()), output)
            total += 1
        within_canvas += len(local_canvas) >= 2
        within_source += len(local_source) >= 2
        within_blocking += len(local_blocking) >= 2
        within_objects += len(local_objects) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(role_sets) >= 35 and len(canvas_sizes) >= 70
    assert len(source_sizes) >= 30 and len(blocked_side_counts) >= 4
    assert len(stop_distances) >= 5 and len(profiles) >= 8
    assert distant_pen >= 20 and adjacent_stop >= 10
    assert no_pen >= 10 and edge_source >= 20 and within_canvas >= 35
    assert within_source == within_blocking == within_objects == 42
    assert all(review[f"blocked_{n}"] > 0 for n in range(5))
    assert review["full_side"] >= 20 and review["selective_side"] >= 50
    assert review["unequal_stops"] >= 10 and review["far_stop"] >= 30
    assert review["empty"] >= 10 and review["single_stopper"] >= 10
    assert review["crowded"] >= 20
    assert review["thick_blocker"] >= 50 and review["large_blocker"] >= 20
    assert min(review["pen_all"], review["pen_none"], review["pen_partial"]) >= 20
    assert review["small_canvas"] >= 30 and review["large_canvas"] >= 30
    assert review["long_narrow"] >= 30 and review["source_canvas_shape"] >= 1
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    assert _equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _equal_puzzles(generate_puzzle(12345, 7, 3), generate_puzzle(12346, 7, 3))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(3001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
