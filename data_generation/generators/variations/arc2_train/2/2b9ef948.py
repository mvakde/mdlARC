"""Generate the seed, endpoint-vector, and diagonal-ray ARC family."""

import hashlib

import numpy as np


_DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)
_SOURCE_PAIR_HASHES = {
    ((15, 20), "c1eac77bad85f94b97b55f24ac51e3a1e198a4b223bcf115e5abea71eab63e3a"),
    ((30, 30), "98108d130a094789be26292a567950b03ce4d79b108ede208231fb8a73f043f5"),
    ((30, 30), "6487787af782b27aff7d80bd0629250d37afa74d1239ad0b702564b9fcac0ed1"),
    ((12, 13), "5be77eb6f28a930ab2e4efb1495927720c13b6704e45c198be3e3a954bf881bb"),
    ((25, 20), "9c3ff97f4a5f0b7104ec4785b5def30a9b2ee20702d2a761eefb7879a3a26271"),
}


def _integer(rng, low, high):
    if low > high:
        raise ValueError("empty sampling interval")
    return int(rng.integers(low, high + 1))


def _checked_count(name, value, low, high):
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


def _counts(seed, rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        code = abs(int(seed))
        test = code % 4 + 1
        minimum = max(2, test + 1)
        train = minimum + (code // 4) % (11 - minimum)
    elif train is None:
        train = _integer(rng, max(2, test + 1), 10)
    elif test is None:
        test = _integer(rng, 1, min(4, train - 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _seed_cells(top, left):
    return {(top + r, left + c) for r in range(3) for c in range(3)}


def _perimeter(top, left):
    return {
        (top + r, left + c)
        for r in range(3)
        for c in range(3)
        if r in (0, 2) or c in (0, 2)
    }


def _neighbors(cell):
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _path_valid(path):
    if len(path) < 4 or len(set(path)) != len(path):
        return False
    cells = set(path)
    degrees = {
        p: sum(q in cells for q in _neighbors(p)) for p in cells
    }
    return (
        degrees[path[0]] == degrees[path[-1]] == 1
        and all(degrees[p] == 2 for p in path[1:-1])
    )


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        r, c = path[-1]
        tr, tc = target
        if (r == tr) == (c == tc):
            return []
        while (r, c) != target:
            r += (tr > r) - (tr < r)
            c += (tc > c) - (tc < c)
            path.append((r, c))
    return path


def _bends(path):
    steps = [
        (b[0] - a[0], b[1] - a[1])
        for a, b in zip(path, path[1:])
    ]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _paths(start, end, height, width, rng, forbidden):
    """Offer straight, monotone stepped, and outside-box detour routes."""
    raw = []

    def add(path):
        if (
            path
            and path[0] == start
            and path[-1] == end
            and all(0 <= r < height and 0 <= c < width for r, c in path)
            and all(p not in forbidden for p in path)
            and _path_valid(path)
        ):
            raw.append(path)

    sr, sc = start
    er, ec = end
    dr, dc = er - sr, ec - sc
    if dr == 0 or dc == 0:
        add(_polyline([start, end]))
    else:
        add(_polyline([start, (sr, ec), end]))
        add(_polyline([start, (er, sc), end]))
        if abs(dc) >= 2:
            for k in range(1, abs(dc)):
                middle = sc + ((dc > 0) - (dc < 0)) * k
                add(_polyline([start, (sr, middle), (er, middle), end]))
        if abs(dr) >= 2:
            for k in range(1, abs(dr)):
                middle = sr + ((dr > 0) - (dr < 0)) * k
                add(_polyline([start, (middle, sc), (middle, ec), end]))
        for _ in range(10):
            steps = [(1 if dr > 0 else -1, 0)] * abs(dr)
            steps += [(0, 1 if dc > 0 else -1)] * abs(dc)
            rng.shuffle(steps)
            path = [start]
            r, c = start
            for step_r, step_c in steps:
                r += step_r
                c += step_c
                path.append((r, c))
            add(path)

    outside_cols = [
        c for c in range(width)
        if c < min(sc, ec) or c > max(sc, ec)
    ]
    outside_rows = [
        r for r in range(height)
        if r < min(sr, er) or r > max(sr, er)
    ]
    rng.shuffle(outside_cols)
    rng.shuffle(outside_rows)
    for column in outside_cols[:8]:
        add(_polyline([start, (sr, column), (er, column), end]))
        if dr != 0:
            middle = sr + ((dr > 0) - (dr < 0)) * _integer(rng, 1, abs(dr))
            if middle != er:
                add(_polyline([
                    start, (sr, column), (middle, column),
                    (middle, ec), end,
                ]))
    for row in outside_rows[:8]:
        add(_polyline([start, (row, sc), (row, ec), end]))
        if dc != 0:
            middle = sc + ((dc > 0) - (dc < 0)) * _integer(rng, 1, abs(dc))
            if middle != ec:
                add(_polyline([
                    start, (row, sc), (row, middle),
                    (er, middle), end,
                ]))

    seen = set()
    result = []
    for path in raw:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _output(height, width, structural, signal, top, left):
    grid = np.full((height, width), signal, dtype=np.int8)
    for r, c in _perimeter(top, left):
        grid[r, c] = structural
    for corner, step in (
        ((top, left), (-1, -1)),
        ((top, left + 2), (-1, 1)),
        ((top + 2, left), (1, -1)),
        ((top + 2, left + 2), (1, 1)),
    ):
        r, c = corner
        while 0 <= r < height and 0 <= c < width:
            grid[r, c] = structural
            r += step[0]
            c += step[1]
    return grid


def _dimensions(rng):
    mode = _integer(rng, 0, 5)
    if mode == 0:
        return _integer(rng, 10, 15), _integer(rng, 10, 15)
    if mode == 1:
        return _integer(rng, 14, 24), _integer(rng, 14, 24)
    if mode == 2:
        return _integer(rng, 24, 30), _integer(rng, 24, 30)
    if mode == 3:
        side = _integer(rng, 10, 30)
        return side, side
    if mode == 4:
        return _integer(rng, 10, 20), _integer(rng, 20, 30)
    return _integer(rng, 20, 30), _integer(rng, 10, 20)


def _direction_plan(rng, train_count, test_count):
    """Pair directions are visible locally; the plan only shapes the split."""
    mode = _integer(rng, 0, 3)
    anchor = _integer(rng, 0, 7)
    if mode == 0:  # A simple, consistent-direction puzzle.
        return [anchor] * (train_count + test_count)
    if mode == 1:  # A test direction withheld from training.
        other = (anchor + _integer(rng, 1, 3)) % 8
        train = [anchor, other]
        train += [_integer(rng, 0, 1) * (other - anchor) + anchor
                  for _ in range(train_count - 2)]
        held = (anchor + 4) % 8
        if held in train:
            held = (anchor + 5) % 8
        return train + [held] * test_count
    if mode == 2:  # Directions mixed in both splits.
        values = [_integer(rng, 0, 7) for _ in range(train_count + test_count)]
        if values[0] == values[1]:
            values[1] = (values[0] + _integer(rng, 1, 7)) % 8
        return values
    # Axial and diagonal cases can coexist in training and testing.
    axial = (1, 3, 4, 6)
    diagonal = (0, 2, 5, 7)
    train = [axial[_integer(rng, 0, 3)],
             diagonal[_integer(rng, 0, 3)]]
    train += [_integer(rng, 0, 7) for _ in range(train_count - 2)]
    return train + [_integer(rng, 0, 7) for _ in range(test_count)]


def _magnitude(rng, maximum):
    if maximum < 1:
        raise ValueError("no feasible vector magnitude")
    if _integer(rng, 0, 2) == 0:
        return _integer(rng, 1, min(4, maximum))
    return _integer(rng, 1, maximum)


def _coordinate(rng, low, high, preference):
    if preference == "low":
        return _integer(rng, low, min(high, low + 2))
    if preference == "high":
        return _integer(rng, max(low, high - 2), high)
    if preference == "middle":
        mid = (low + high) // 2
        return _integer(rng, max(low, mid - 2), min(high, mid + 2))
    return _integer(rng, low, high)


def _target_position(rng, height, width, dr, dc):
    low_r, high_r = max(0, dr), min(height - 3, height - 3 + dr)
    low_c, high_c = max(0, dc), min(width - 3, width - 3 + dc)
    if low_r > high_r or low_c > high_c:
        raise ValueError("seed translation exceeds canvas")
    placement = _integer(rng, 0, 7)
    preferences = (
        ("any", "any"), ("middle", "middle"),
        ("low", "any"), ("high", "any"),
        ("any", "low"), ("any", "high"),
        ("low", "low"), ("high", "high"),
    )
    pref_r, pref_c = preferences[placement]
    return (
        _coordinate(rng, low_r, high_r, pref_r),
        _coordinate(rng, low_c, high_c, pref_c),
    )


def _palette(rng, shared_colors, prior=None):
    if shared_colors is None:
        colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    else:
        colors = [int(x) for x in rng.permutation(shared_colors)]
    # Roles are background, structural, guide, signal. In a two-example
    # training set, make the visible role assignment change explicitly.
    if prior is not None and tuple(colors[:3]) == tuple(prior[:3]):
        colors = colors[1:] + colors[:1]
    return tuple(colors)


def _generate_pair(rng, direction, route_preference, palette, shape=None):
    background, structural, guide, signal = palette
    for attempt in range(300):
        height, width = shape if shape is not None else _dimensions(rng)
        sign_r, sign_c = _DIRECTIONS[direction]
        dr = sign_r * _magnitude(rng, height - 6) if sign_r else 0
        dc = sign_c * _magnitude(rng, width - 6) if sign_c else 0
        if abs(dr) + abs(dc) < 3:
            continue
        try:
            target_top, target_left = _target_position(
                rng, height, width, dr, dc
            )
        except ValueError:
            continue
        seed_top, seed_left = target_top - dr, target_left - dc
        forbidden = {
            (r, c)
            for r in range(seed_top - 1, seed_top + 4)
            for c in range(seed_left - 1, seed_left + 4)
        }
        start_r_low = max(0, -dr)
        start_r_high = min(height - 1, height - 1 - dr)
        start_c_low = max(0, -dc)
        start_c_high = min(width - 1, width - 1 - dc)
        starts = [
            (r, c)
            for r in range(start_r_low, start_r_high + 1)
            for c in range(start_c_low, start_c_high + 1)
            if (r, c) not in forbidden
            and (r + dr, c + dc) not in forbidden
        ]
        if not starts:
            continue
        for index in rng.permutation(len(starts))[:min(30, len(starts))]:
            start = starts[int(index)]
            end = (start[0] + dr, start[1] + dc)
            options = _paths(start, end, height, width, rng, forbidden)
            shortest = abs(dr) + abs(dc) + 1
            if route_preference == "short":
                options = [p for p in options if len(p) == shortest]
            elif route_preference == "detour":
                options = [p for p in options if len(p) > shortest]
            if not options:
                continue
            # Choose bend class before a route so the many monotone paths do
            # not drown out straight and single-elbow examples.
            bend_classes = sorted({_bends(p) for p in options})
            bends = bend_classes[_integer(rng, 0, len(bend_classes) - 1)]
            options = [p for p in options if _bends(p) == bends]
            path = options[_integer(rng, 0, len(options) - 1)]
            input_grid = np.full(
                (height, width), background, dtype=np.int8
            )
            for r, c in _perimeter(seed_top, seed_left):
                input_grid[r, c] = structural
            input_grid[seed_top + 1, seed_left + 1] = signal
            for r, c in path[1:-1]:
                input_grid[r, c] = guide
            input_grid[start] = signal
            input_grid[end] = structural
            output_grid = _output(
                height, width, structural, signal, target_top, target_left
            )
            return input_grid, output_grid
    raise ValueError("unable to place seed and guide")


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_hash(input_grid, output_grid):
    digest = hashlib.sha256()
    digest.update(input_grid.astype(np.int8, copy=False).tobytes())
    digest.update(output_grid.astype(np.int8, copy=False).tobytes())
    return input_grid.shape, digest.hexdigest()


def _analyze(input_grid, output_grid):
    """Recover roles and vector from grids, independently of sampled state."""
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape
    colors = {int(x) for x in np.unique(input_grid)}
    assert len(colors) == 4
    candidates = []
    for top in range(height - 2):
        for left in range(width - 2):
            structural = int(input_grid[top, left])
            signal = int(input_grid[top + 1, left + 1])
            if structural == signal:
                continue
            if any(input_grid[r, c] != structural
                   for r, c in _perimeter(top, left)):
                continue
            if int(np.count_nonzero(input_grid == structural)) != 9:
                continue
            if int(np.count_nonzero(input_grid == signal)) != 2:
                continue
            frame = _perimeter(top, left)
            seed = _seed_cells(top, left)
            seed_center = (top + 1, left + 1)
            structural_points = {
                tuple(map(int, p))
                for p in np.argwhere(input_grid == structural)
            }
            signal_points = {
                tuple(map(int, p))
                for p in np.argwhere(input_grid == signal)
            }
            structural_end = next(iter(structural_points - frame))
            signal_end = next(iter(signal_points - {seed_center}))
            for guide in colors - {structural, signal}:
                background = next(iter(colors - {
                    structural, signal, guide
                }))
                guide_points = {
                    tuple(map(int, p))
                    for p in np.argwhere(input_grid == guide)
                }
                if len(guide_points) < 2:
                    continue
                path_points = guide_points | {
                    structural_end, signal_end
                }
                if any(
                    max(abs(r - sr), abs(c - sc)) <= 1
                    for r, c in path_points for sr, sc in seed
                ):
                    continue
                degrees = {
                    p: sum(q in path_points for q in _neighbors(p))
                    for p in path_points
                }
                if (
                    degrees[structural_end] != 1
                    or degrees[signal_end] != 1
                    or any(degrees[p] != 2 for p in guide_points)
                ):
                    continue
                ordered = [signal_end]
                previous = None
                while ordered[-1] != structural_end:
                    next_points = [
                        p for p in _neighbors(ordered[-1])
                        if p in path_points and p != previous
                    ]
                    if len(next_points) != 1:
                        break
                    previous = ordered[-1]
                    ordered.append(next_points[0])
                    if len(ordered) > len(path_points):
                        break
                if (
                    ordered[-1] != structural_end
                    or len(ordered) != len(path_points)
                ):
                    continue
                dr = structural_end[0] - signal_end[0]
                dc = structural_end[1] - signal_end[1]
                translated = (top + dr, left + dc)
                if not (
                    0 <= translated[0] <= height - 3
                    and 0 <= translated[1] <= width - 3
                ):
                    continue
                expected = _output(
                    height, width, structural, signal, *translated
                )
                if not np.array_equal(expected, output_grid):
                    continue
                assert {int(x) for x in np.unique(output_grid)} == {
                    structural, signal
                }
                assert int(np.count_nonzero(input_grid == background)) > 0
                candidates.append({
                    "shape": (height, width),
                    "palette": (background, structural, guide, signal),
                    "seed": (top, left),
                    "translated": translated,
                    "vector": (dr, dc),
                    "guide_start": signal_end,
                    "guide_end": structural_end,
                    "path_length": len(ordered),
                    "route_excess": len(ordered) - abs(dr) - abs(dc) - 1,
                    "bends": _bends(ordered),
                })
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {
                "input", "output"
            }
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(pair["input"], pair["output"])
            assert _source_hash(
                pair["input"], pair["output"]
            ) not in _SOURCE_PAIR_HASHES
            metadata[split].append(
                _analyze(pair["input"], pair["output"])
            )
    # There is no hidden color map: every pair exposes all four local roles,
    # and training examples demonstrate the same ordered endpoint rule.
    for split in ("train", "test"):
        for item in metadata[split]:
            assert item["vector"] != (0, 0)
            assert item["route_excess"] >= 0
            assert item["path_length"] >= 4
    assert len(metadata["train"]) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(
        seed, rng, num_train, num_test
    )
    directions = _direction_plan(rng, train_count, test_count)
    shared_colors = (
        tuple(int(x) for x in rng.choice(10, size=4, replace=False))
        if _integer(rng, 0, 3) == 0 else None
    )
    puzzle = {"train": [], "test": []}
    used = set()
    prior_palette = None
    train_shapes = []
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        local_index = index if split == "train" else index - train_count
        palette = _palette(
            rng, shared_colors,
            prior_palette if index == 1 else None
        )
        if index == 0:
            prior_palette = palette
        if split == "train" and local_index < 2:
            route_preference = ("short", "detour")[
                (local_index + (seed % 2)) % 2
            ]
        else:
            route_preference = ("short", "detour", "any")[
                _integer(rng, 0, 2)
            ]
        # Test canvas size is a pair-level choice. Reusing a training size,
        # choosing a large or compact size, and sampling freely all occur.
        shape = None
        if split == "test":
            size_choice = _integer(rng, 0, 7)
            if size_choice < 2:
                shape = train_shapes[
                    _integer(rng, 0, len(train_shapes) - 1)
                ]
            elif size_choice < 4:
                shape = (
                    _integer(rng, 24, 30),
                    _integer(rng, 24, 30),
                )
            elif size_choice == 4:
                shape = (
                    _integer(rng, 10, 15),
                    _integer(rng, 10, 15),
                )
        for _ in range(50):
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            pair = _generate_pair(
                pair_rng, directions[index], route_preference, palette,
                shape=shape,
            )
            input_key = _key(pair[0])
            if input_key not in used:
                used.add(input_key)
                puzzle[split].append({
                    "input": pair[0], "output": pair[1]
                })
                if split == "train":
                    train_shapes.append(pair[0].shape)
                break
        else:
            raise ValueError("unable to create a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a[name], b[name])
            for a, b in zip(first[split], second[split])
            for name in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes = set()
    palettes = set()
    vectors = set()
    bends = set()
    excess = set()
    aspects = set()
    signs = set()
    background = set()
    structural = set()
    guides = set()
    source_shapes = set()
    input_keys = set()
    same_palette_puzzles = 0
    role_varying_puzzles = 0
    mixed_aspect_puzzles = 0
    relative_areas = set()
    equal_shape_puzzles = 0
    consistent_direction_puzzles = 0
    heldout_direction_puzzles = 0
    familiar_direction_puzzles = 0
    small_components = 0
    long_components = 0
    overlapping_footprints = 0
    short_two_bend = 0
    three_plus_bend = 0
    straight_guides = 0
    one_bend_guides = 0
    central_targets = 0
    corner_targets = 0
    near_edges = set()
    compact_recolored_complex = 0
    zero_roles = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_meta = [
            _analyze(p["input"], p["output"])
            for p in puzzle["train"]
        ]
        test_meta = [
            _analyze(p["input"], p["output"])
            for p in puzzle["test"]
        ]
        all_meta = train_meta + test_meta
        local_palettes = {m["palette"] for m in all_meta}
        local_aspects = {
            (m["shape"][0] > m["shape"][1])
            - (m["shape"][0] < m["shape"][1])
            for m in all_meta
        }
        same_palette_puzzles += len({
            frozenset(m["palette"]) for m in all_meta
        }) == 1
        role_varying_puzzles += all(
            len({m["palette"][i] for m in all_meta}) > 1
            for i in range(3)
        )
        mixed_aspect_puzzles += len(local_aspects) > 1
        assert len({m["shape"] for m in all_meta}) > 1
        assert {
            train_meta[0]["route_excess"] == 0,
            train_meta[1]["route_excess"] == 0,
        } == {True, False}
        train_directions = {
            tuple((v > 0) - (v < 0) for v in m["vector"])
            for m in train_meta
        }
        test_directions = {
            tuple((v > 0) - (v < 0) for v in m["vector"])
            for m in test_meta
        }
        consistent_direction_puzzles += len(
            train_directions | test_directions
        ) == 1
        heldout_direction_puzzles += bool(
            test_directions - train_directions
        )
        familiar_direction_puzzles += bool(
            test_directions & train_directions
        )
        equal_shape_puzzles += bool(
            {m["shape"] for m in train_meta}
            & {m["shape"] for m in test_meta}
        )
        train_areas = [np.prod(m["shape"]) for m in train_meta]
        test_areas = [np.prod(m["shape"]) for m in test_meta]
        relative_areas.add(
            "larger" if max(test_areas) > max(train_areas)
            else "smaller" if max(test_areas) < min(train_areas)
            else "mixed"
        )
        for m, p in zip(all_meta, puzzle["train"] + puzzle["test"]):
            shapes.add(m["shape"])
            palettes.add(m["palette"])
            vectors.add(m["vector"])
            bends.add(m["bends"])
            excess.add(m["route_excess"])
            aspects.add(
                (m["shape"][0] > m["shape"][1])
                - (m["shape"][0] < m["shape"][1])
            )
            signs.add((
                (m["vector"][0] > 0) - (m["vector"][0] < 0),
                (m["vector"][1] > 0) - (m["vector"][1] < 0),
            ))
            background.add(m["palette"][0])
            structural.add(m["palette"][1])
            guides.add(m["palette"][2])
            if m["shape"] in {
                (15, 20), (30, 30), (12, 13), (25, 20)
            }:
                source_shapes.add(m["shape"])
            input_keys.add(_key(p["input"]))
            dr, dc = m["vector"]
            small_components += abs(dr) == 1 or abs(dc) == 1
            long_components += max(abs(dr), abs(dc)) > 8
            overlapping_footprints += (
                abs(dr) <= 2 and abs(dc) <= 2
            )
            short_two_bend += (
                m["bends"] == 2 and m["route_excess"] == 0
            )
            three_plus_bend += m["bends"] >= 3
            straight_guides += m["bends"] == 0
            one_bend_guides += m["bends"] == 1
            height, width = m["shape"]
            top, left = m["translated"]
            distances = (
                top, height - 3 - top, left, width - 3 - left
            )
            central_targets += min(distances) >= 4
            corner_targets += (
                min(distances[:2]) <= 2
                and min(distances[2:]) <= 2
            )
            near_edges.update(
                index for index, distance in enumerate(distances)
                if distance <= 2
            )
            compact_recolored_complex += (
                max(height, width) <= 15
                and m["palette"][0] != 0
                and m["bends"] >= 3
            )
            zero_roles.update(
                index for index, color in enumerate(m["palette"])
                if color == 0
            )
        assert len(local_palettes) > 1
    assert len(shapes) >= 35
    assert len(palettes) >= 50
    assert len(vectors) >= 30
    assert aspects == {-1, 0, 1}
    assert signs == set(_DIRECTIONS)
    assert len(background) >= 8
    assert len(structural) >= 8
    assert len(guides) >= 8
    assert source_shapes
    assert 0 in background and 0 in structural and 0 in guides
    assert {0, 1, 2, 3} <= bends
    assert 0 in excess and any(x > 0 for x in excess)
    assert same_palette_puzzles >= 2
    assert role_varying_puzzles >= 15
    assert mixed_aspect_puzzles >= 20
    assert relative_areas == {"larger", "smaller", "mixed"}
    assert equal_shape_puzzles >= 3
    assert consistent_direction_puzzles >= 3
    assert heldout_direction_puzzles >= 8
    assert familiar_direction_puzzles >= 8
    assert small_components >= 15
    assert long_components >= 20
    assert overlapping_footprints >= 3
    assert short_two_bend >= 8
    assert three_plus_bend >= 30
    assert straight_guides >= 20
    assert one_bend_guides >= 8
    assert central_targets >= 15
    assert corner_targets >= 15
    assert near_edges == {0, 1, 2, 3}
    assert compact_recolored_complex >= 2
    assert zero_roles == {0, 1, 2, 3}
    assert len(input_keys) >= sum(
        len(p["train"]) + len(p["test"]) for p in puzzles
    )
    # Two different shortest routes between identical colored endpoints
    # must encode one displacement and therefore one output.
    demonstration_paths = _paths(
        (5, 5), (10, 10), 16, 16, np.random.default_rng(771),
        {(r, c) for r in range(-1, 4) for c in range(-1, 4)},
    )
    elbow = next(p for p in demonstration_paths
                 if _bends(p) == 1 and len(p) == 11)
    stepped = next(p for p in demonstration_paths
                   if _bends(p) >= 3 and len(p) == 11)
    demonstration_inputs = []
    demonstration_output = _output(16, 16, 0, 2, 5, 5)
    for path in (elbow, stepped):
        grid = np.full((16, 16), 8, dtype=np.int8)
        for r, c in _perimeter(0, 0):
            grid[r, c] = 0
        grid[1, 1] = 2
        for r, c in path[1:-1]:
            grid[r, c] = 9
        grid[path[0]] = 2
        grid[path[-1]] = 0
        assert _analyze(grid, demonstration_output)["vector"] == (5, 5)
        demonstration_inputs.append(grid)
    assert not np.array_equal(
        demonstration_inputs[0], demonstration_inputs[1]
    )
    a = generate_puzzle(912345, 6, 3)
    b = generate_puzzle(912345, 6, 3)
    assert _equal(a, b)
    assert sum(
        not _equal(generate_puzzle(20000 + i, 4, 2),
                   generate_puzzle(30000 + i, 4, 2))
        for i in range(8)
    ) >= 7
    for seed, train, test in (
        (40001, 2, 1), (40002, 10, 1), (40003, 10, 4),
        (50001, None, 1), (50002, None, 4),
        (50003, 2, None), (50004, 10, None),
    ):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(60000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count was accepted")
