"""Generate flared-emitter lane puzzles."""

import hashlib

import numpy as np


_SOURCE_INPUT_SHA256 = {
    "bb94dfcc385a50e232583cd7c26f1837699ed79b60429ff0e1ec02fa77a0a93a",
    "be80eedd283f4b908c06aee299e11b0a863d82c0f7e233313752e7ec5d719cd0",
    "533a8c42281e05cff20bd92f478bfd8a2f33d1256bd0da2b8e7240887fc8cad9",
    "e236051ff067a331674e8cc742fe18f5d2b919f1eb36d0195ccddf78747b1d9b",
}
_DIRECTIONS = ("down", "up", "right", "left")
_BASE_FLARE = (
    (0, 0),
    (1, -1), (1, 0), (1, 1),
    (2, -2), (2, -1), (2, 1), (2, 2),
    (3, -3), (3, 3),
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count("num_train", num_train, 2, 10)
    test = _optional_count("num_test", num_test, 1, 4)
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


def _flare(variant: int) -> tuple[tuple[int, int], ...]:
    if variant == 0:
        return _BASE_FLARE
    if variant == 1:
        return _BASE_FLARE + ((4, -4), (4, 4))
    raise ValueError("unknown flare variant")


def _point(apex: tuple[int, int], direction: str, depth: int, lane: int) -> tuple[int, int]:
    r, c = apex
    if direction == "down":
        return r + depth, c + lane
    if direction == "up":
        return r - depth, c + lane
    if direction == "right":
        return r + lane, c + depth
    if direction == "left":
        return r + lane, c - depth
    raise ValueError("unknown direction")


def _local(apex: tuple[int, int], direction: str, row: int, col: int) -> tuple[int, int]:
    r, c = apex
    if direction == "down":
        return row - r, col - c
    if direction == "up":
        return r - row, col - c
    if direction == "right":
        return col - c, row - r
    if direction == "left":
        return c - col, row - r
    raise ValueError("unknown direction")


def _emitter_cells(apex: tuple[int, int], direction: str, variant: int) -> set[tuple[int, int]]:
    return {_point(apex, direction, d, t) for d, t in _flare(variant)}


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    largest = int(np.argmax(counts))
    assert int(np.sum(counts == counts[largest])) == 1
    background = int(values[largest])
    assert counts[largest] > max(int(count) for i, count in enumerate(counts) if i != largest)
    candidates = []
    for value, count in zip(values, counts):
        color = int(value)
        if color == background or int(count) not in (10, 12):
            continue
        cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        variant = 0 if len(cells) == 10 else 1
        for apex in cells:
            for direction in _DIRECTIONS:
                if cells == _emitter_cells(apex, direction, variant):
                    candidates.append((color, apex, direction, variant))
    assert len(candidates) == 1
    emitter, apex, direction, variant = candidates[0]
    signal = next(int(v) for v in values if int(v) not in (background, emitter))
    return {
        "background": background, "emitter": emitter, "signal": signal,
        "apex": apex, "direction": direction, "variant": variant,
    }


def _axis_lengths(grid: np.ndarray, direction: str) -> tuple[int, int]:
    if direction in ("down", "up"):
        return int(grid.shape[0]), int(grid.shape[1])
    return int(grid.shape[1]), int(grid.shape[0])


def _forward_length(grid: np.ndarray, info: dict) -> int:
    axis, _ = _axis_lengths(grid, info["direction"])
    r, c = info["apex"]
    apex_axis = {"down": r, "up": grid.shape[0] - 1 - r,
                 "right": c, "left": grid.shape[1] - 1 - c}[info["direction"]]
    return axis - apex_axis - (4 + info["variant"])


def _active_offsets(grid: np.ndarray, info: dict) -> tuple[int, ...]:
    depth = 4 + info["variant"]
    half = 3 + info["variant"]
    active = set()
    for row, col in np.argwhere(grid == info["signal"]):
        d, t = _local(info["apex"], info["direction"], int(row), int(col))
        if d >= depth and abs(t) <= half:
            active.add(t)
    return tuple(sorted(active))


def _expected_output_from_input(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    active = _active_offsets(grid, info)
    assert active
    result = grid.copy()
    axis, _ = _axis_lengths(grid, info["direction"])
    forward = _forward_length(grid, info)
    max_depth = 4 + info["variant"] + forward - 1
    last = {t: max(d for d, lane in _flare(info["variant"]) if lane == t) for t in active}
    for t in active:
        for d in range(last[t] + 1, max_depth + 1):
            row, col = _point(info["apex"], info["direction"], d, t)
            assert 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
            result[row, col] = info["signal"]
    return result


def _sample_dimension(rng: np.random.Generator, regime: str) -> int:
    if rng.random() < (0.30 if regime == "source_train" else 0.18):
        return 20
    return int(rng.integers(13, 31))


def _sample_geometry(rng: np.random.Generator, regime: str, direction: str,
                     variant: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    height = _sample_dimension(rng, regime)
    width = _sample_dimension(rng, regime)
    depth = 4 + variant
    half = 3 + variant
    axis = height if direction in ("down", "up") else width
    cross = width if direction in ("down", "up") else height
    assert cross >= 2 * half + 3 and axis >= depth + 6
    max_apex = axis - depth - 6
    positions = np.arange(max_apex + 1)
    if regime == "target_test":
        weights = 1.0 + 0.7 * (max_apex - positions) / max(1, max_apex)
        weights /= weights.sum()
        apex_axis = int(rng.choice(positions, p=weights))
    else:
        apex_axis = int(rng.choice(positions))
    transverse = int(rng.integers(half, cross - half))
    if direction == "down":
        apex = (apex_axis, transverse)
    elif direction == "up":
        apex = (height - 1 - apex_axis, transverse)
    elif direction == "right":
        apex = (transverse, apex_axis)
    else:
        apex = (transverse, width - 1 - apex_axis)
    forward = axis - apex_axis - depth
    return (height, width), apex, forward


def _sample_palette(rng: np.random.Generator, previous: list[tuple[int, int, int]],
                    force_new_background: bool = False) -> tuple[int, int, int]:
    if previous and rng.random() < 0.48:
        prior = previous[int(rng.integers(0, len(previous)))]
        emitter, signal = prior[1:]
        if force_new_background or rng.random() < 0.52:
            choices = [v for v in range(10) if v not in (emitter, signal, prior[0])]
            background = int(rng.choice(choices))
            return background, emitter, signal
        return prior
    background = int(rng.integers(0, 10))
    if force_new_background:
        used = {p[0] for p in previous}
        choices = [v for v in range(10) if v not in used]
        if choices:
            background = int(rng.choice(choices))
    colors = [v for v in range(10) if v != background]
    emitter, signal = rng.choice(colors, size=2, replace=False)
    return background, int(emitter), int(signal)


def _sample_combos(rng: np.random.Generator, train_count: int) -> list[tuple[str, int]]:
    base = str(rng.choice(("down", "down", "down", "up", "right", "left")))
    directions = [base]
    if train_count >= 3 and rng.random() < 0.52:
        directions.append(str(rng.choice([d for d in _DIRECTIONS if d != base])))
    if train_count >= 6 and len(directions) == 2 and rng.random() < 0.25:
        directions.append(str(rng.choice([d for d in _DIRECTIONS if d not in directions])))
    variants = [0]
    if train_count >= 3 and rng.random() < 0.38:
        variants.append(1)
    elif train_count == 2 and len(directions) == 1 and rng.random() < 0.20:
        variants.append(1)
    combos = [(direction, variant) for direction in directions for variant in variants]
    if len(combos) > train_count:
        combos = combos[:train_count]
    return combos


def _sample_offsets(rng: np.random.Generator, half: int, count: int,
                    preferred: set[int] | None = None) -> tuple[int, ...]:
    pool = list(range(-half, half + 1))
    chosen = []
    for _ in range(count):
        if preferred:
            weights = np.array([2.5 if t in preferred else 1.0 for t in pool], dtype=float)
            weights /= weights.sum()
            t = int(rng.choice(pool, p=weights))
        else:
            t = int(rng.choice(pool))
        chosen.append(t)
        pool.remove(t)
    return tuple(sorted(chosen))


def _sample_train_programs(rng: np.random.Generator, combos: list[tuple[str, int]],
                           count: int) -> list[tuple[str, int, tuple[int, ...]]]:
    assigned = combos.copy()
    while len(assigned) < count:
        assigned.append(combos[int(rng.integers(0, len(combos)))])
    rng.shuffle(assigned)
    if count == 2:
        lane_counts = [1, 1]
    else:
        lane_counts = [int(rng.choice((1, 2, 3, 4, 5), p=(.28, .27, .23, .16, .06)))
                       for _ in range(count)]
        if 1 not in lane_counts:
            lane_counts[int(rng.integers(0, count))] = 1
        if max(lane_counts) == 1:
            lane_counts[int(rng.integers(0, count))] = int(rng.integers(2, 5))
    rng.shuffle(lane_counts)
    exposed: set[int] = set()
    programs = []
    for direction, variant in assigned:
        count_here = min(lane_counts[len(programs)], 7 + 2 * variant)
        offsets = _sample_offsets(rng, 3 + variant, count_here,
                                  set(range(-3 - variant, 4 + variant)) - exposed)
        exposed.update(offsets)
        programs.append((direction, variant, offsets))
    return programs


def _sample_test_programs(rng: np.random.Generator,
                          train: list[tuple[str, int, tuple[int, ...]]],
                          count: int) -> list[tuple[str, int, tuple[int, ...]]]:
    combos = sorted({(direction, variant) for direction, variant, _ in train})
    patterns = {p for _, _, p in train}
    result = []
    for _ in range(count):
        direction, variant = combos[int(rng.integers(0, len(combos)))]
        exposed = sorted({t for _, v, offsets in train if v == variant for t in offsets})
        weights = (.26, .29, .24, .15, .06)
        desired = int(rng.choice((1, 2, 3, 4, 5), p=weights))
        lane_count = min(desired, len(exposed))
        candidates = []
        for _ in range(16):
            offsets = tuple(sorted(int(v) for v in rng.choice(exposed, size=lane_count, replace=False)))
            candidates.append(offsets)
            if offsets not in patterns and offsets not in [p[2] for p in result]:
                break
        result.append((direction, variant, offsets))
    return result


def _choose_signal_depth(rng: np.random.Generator, depth: int, forward: int) -> int:
    choice = int(rng.integers(0, 3))
    if choice == 0:
        return depth + int(rng.integers(0, min(3, forward)))
    if choice == 1:
        return depth + forward - 1 - int(rng.integers(0, min(3, forward)))
    return depth + int(rng.integers(0, forward))


def generate_pair(rng: np.random.Generator, regime: str, direction: str, variant: int,
                  offsets: tuple[int, ...], palette: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    shape, apex, forward = _sample_geometry(rng, regime, direction, variant)
    background, emitter, signal = palette
    grid = np.full(shape, background, dtype=np.int8)
    emitter_cells = _emitter_cells(apex, direction, variant)
    for row, col in emitter_cells:
        grid[row, col] = emitter
    depth = 4 + variant
    half = 3 + variant
    signals = set()
    for lane in offsets:
        first = _choose_signal_depth(rng, depth, forward)
        signals.add(_point(apex, direction, first, lane))
        if rng.random() < .38:
            extra = int(rng.integers(1, 3))
            possible = [d for d in range(depth, depth + forward) if d != first]
            for d in rng.choice(possible, size=min(extra, len(possible)), replace=False):
                signals.add(_point(apex, direction, int(d), lane))
    if rng.random() < .18:
        extra_count = 0
    elif rng.random() < .16:
        extra_count = int(rng.integers(13, 23))
    else:
        extra_count = int(rng.integers(1, 13))
    buckets = {"behind": [], "side": [], "lateral": []}
    for row in range(shape[0]):
        for col in range(shape[1]):
            if (row, col) in emitter_cells or (row, col) in signals:
                continue
            d, t = _local(apex, direction, row, col)
            if d < 0:
                buckets["behind"].append((row, col))
            elif d < depth and abs(t) > half:
                buckets["side"].append((row, col))
            elif d >= depth and abs(t) > half:
                buckets["lateral"].append((row, col))
    for _ in range(extra_count):
        available = [name for name, cells in buckets.items() if cells]
        if not available:
            break
        weights = np.array([{"behind": .30, "side": .30, "lateral": .40}[v]
                            for v in available], dtype=float)
        weights /= weights.sum()
        category = str(rng.choice(available, p=weights))
        cells = buckets[category]
        signals.add(cells.pop(int(rng.integers(0, len(cells)))))
    for row, col in signals:
        grid[row, col] = signal
    return grid, _expected_output_from_input(grid)


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    array = np.ascontiguousarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # The shared convention is the same visible flared-emitter rule for all pairs.
    # The repertoire below only controls which visible orientations and shapes
    # are demonstrated before they can appear in a test.
    combos = _sample_combos(rng, train_count)
    train_programs = _sample_train_programs(rng, combos, train_count)
    test_programs = _sample_test_programs(rng, train_programs, test_count)
    puzzle = {"train": [], "test": []}
    palettes: list[tuple[int, int, int]] = []
    seen = set()
    for section, regime, programs in (("train", "source_train", train_programs),
                                       ("test", "target_test", test_programs)):
        for direction, variant, offsets in programs:
            force_background = (section == "train" and len(palettes) == 1 and
                                train_count >= 3 and rng.random() < .7)
            palette = _sample_palette(rng, palettes, force_background)
            palettes.append(palette)
            for _ in range(8):
                input_grid, output_grid = generate_pair(rng, regime, direction,
                                                         variant, offsets, palette)
                signature = _grid_signature(input_grid)
                if signature not in seen:
                    break
            else:
                raise ValueError("could not construct a unique input")
            seen.add(signature)
            puzzle[section].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_pair_rule(input_grid: np.ndarray, output_grid: np.ndarray, info: dict) -> None:
    """Inspect each cell directly, independently of the ray-writing routine."""
    offsets = _active_offsets(input_grid, info)
    assert offsets
    last = {t: max(d for d, lane in _flare(info["variant"]) if lane == t) for t in offsets}
    changed = 0
    for row in range(input_grid.shape[0]):
        for col in range(input_grid.shape[1]):
            d, t = _local(info["apex"], info["direction"], row, col)
            expected = int(input_grid[row, col])
            if expected == info["background"] and t in last and d > last[t]:
                expected = info["signal"]
                changed += 1
            assert int(output_grid[row, col]) == expected
    assert changed > 0


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    records = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            input_key = _grid_signature(inp)
            pair_key = (input_key, _grid_signature(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            if inp.shape == (20, 20):
                assert hashlib.sha256(np.ascontiguousarray(inp, dtype=np.int8).tobytes()).hexdigest() not in _SOURCE_INPUT_SHA256
            info = _analyze_input(inp)
            assert 3 == len(np.unique(out))
            assert _forward_length(inp, info) >= 6
            offsets = _active_offsets(inp, info)
            assert 1 <= len(offsets) <= 5
            _assert_pair_rule(inp, out, info)
            records[section].append((info, offsets))
    train_combos = {(i["direction"], i["variant"]) for i, _ in records["train"]}
    train_lanes = {variant: {t for i, offsets in records["train"] if i["variant"] == variant
                             for t in offsets} for variant in (0, 1)}
    for info, offsets in records["test"]:
        assert (info["direction"], info["variant"]) in train_combos
        assert set(offsets).issubset(train_lanes[info["variant"]])


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((section, _grid_signature(pair["input"]), _grid_signature(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    directions, variants, shapes, offsets, palettes, backgrounds = set(), set(), set(), set(), set(), set()
    marker_counts, distractor_kinds, forward_lengths = set(), set(), set()
    seen_zero_distractor = seen_dense = seen_repeated_palette = seen_mixed_background = False
    seen_mixed_orientation = seen_mixed_shape = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        infos = []
        triples = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                info = _analyze_input(grid)
                active = _active_offsets(grid, info)
                infos.append(info)
                triples.append((info["background"], info["emitter"], info["signal"]))
                directions.add(info["direction"])
                variants.add(info["variant"])
                shapes.add(grid.shape)
                offsets.add(active)
                palettes.add(triples[-1])
                backgrounds.add(info["background"])
                forward_lengths.add(_forward_length(grid, info))
                marker_counts.add(int(np.count_nonzero(grid == info["signal"])))
                kinds = set()
                inert = 0
                for row, col in np.argwhere(grid == info["signal"]):
                    d, t = _local(info["apex"], info["direction"], int(row), int(col))
                    if d < 0:
                        kinds.add("behind"); inert += 1
                    elif d < 4 + info["variant"] and abs(t) > 3 + info["variant"]:
                        kinds.add("side"); inert += 1
                    elif d >= 4 + info["variant"] and abs(t) > 3 + info["variant"]:
                        kinds.add("lateral"); inert += 1
                distractor_kinds.update(kinds)
                seen_zero_distractor |= inert == 0
                seen_dense |= int(np.count_nonzero(grid == info["signal"])) >= 20
        seen_repeated_palette |= len(set(triples)) < len(triples)
        seen_mixed_background |= len({v[0] for v in triples}) > 1
        seen_mixed_orientation |= len({i["direction"] for i in infos}) > 1
        seen_mixed_shape |= len({i["variant"] for i in infos}) > 1
    assert directions == set(_DIRECTIONS) and variants == {0, 1}
    assert len(shapes) >= 45 and (20, 20) in shapes
    assert len(offsets) >= 35 and max(len(v) for v in offsets) >= 4
    assert len(palettes) >= 80 and len(backgrounds) == 10
    assert max(marker_counts) >= 20 and min(marker_counts) == 1
    assert max(forward_lengths) >= 17 and min(forward_lengths) <= 8
    assert distractor_kinds == {"behind", "side", "lateral"}
    assert all((seen_zero_distractor, seen_dense, seen_repeated_palette,
                seen_mixed_background, seen_mixed_orientation, seen_mixed_shape))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert requested is None or actual == requested
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(4000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts did not raise ValueError")
    assert _puzzle_signature(generate_puzzle(123456, 7, 3)) == _puzzle_signature(generate_puzzle(123456, 7, 3))
    assert _puzzle_signature(generate_puzzle(98765)) == _puzzle_signature(generate_puzzle(98765))
