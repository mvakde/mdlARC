"""Generators for the facing-edge reflection ARC family."""

from itertools import combinations

import numpy as np


_DIRECTIONS = ("right", "left", "down", "up")

# Exact input/output coordinate signatures from the four supplied source pairs.
_ORIGINAL = (
    (((5, 0), (6, 1), (7, 0), (7, 1)), ((6, 4), (6, 5), (7, 4), (7, 5)),
     ((5, 2), (6, 3), (7, 2), (7, 3)), ((5, 5), (6, 4), (7, 4), (7, 5))),
    (((1, 4), (2, 4), (3, 4), (3, 5)), ((6, 4), (6, 5), (7, 4), (7, 5)),
     ((3, 4), (4, 4), (5, 4), (5, 5)), ((6, 4), (6, 5), (7, 4), (8, 4))),
    (((10, 4), (10, 5), (11, 4), (12, 3)), ((6, 4), (6, 5), (7, 4), (7, 5)),
     ((8, 4), (8, 5), (9, 4), (10, 3)), ((5, 3), (6, 4), (7, 4), (7, 5))),
    (((5, 10), (6, 9), (6, 10), (7, 9)), ((6, 4), (6, 5), (7, 4), (7, 5)),
     ((5, 7), (6, 6), (6, 7), (7, 6)), ((5, 4), (6, 4), (6, 5), (7, 5))),
)


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _normalize(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected(cells, diagonal):
    cells = set(cells)
    seen = set()
    pending = [next(iter(cells))]
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while pending:
        cell = pending.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        pending.extend((r + dr, c + dc) for dr, dc in offsets
                       if (r + dr, c + dc) in cells and (r + dr, c + dc) not in seen)
    return len(seen) == len(cells)


def _square(cells):
    r0, r1, c0, c1 = _bbox(cells)
    side = r1 - r0 + 1
    return (side >= 2 and c1 - c0 + 1 == side and len(cells) == side * side
            and set(cells) == {(r, c) for r in range(r0, r1 + 1)
                               for c in range(c0, c1 + 1)})


def _axis_asymmetric(cells, axis):
    r0, r1, c0, c1 = _bbox(cells)
    if axis == "horizontal":
        reflected = {(r, c0 + c1 - c) for r, c in cells}
    else:
        reflected = {(r0 + r1 - r, c) for r, c in cells}
    return set(cells) != reflected


def _build_shapes():
    # The 4 by 4 enumeration gives many irregular and diagonally joined
    # patterns, including the geometry of both unusual original examples.
    buckets = {n: set() for n in range(3, 9)}
    for n in buckets:
        for indexes in combinations(range(16), n):
            cells = tuple((i // 4, i % 4) for i in indexes)
            shape = _normalize(cells)
            r0, r1, c0, c1 = _bbox(shape)
            if r1 == r0 or c1 == c0 or _square(shape):
                continue
            if _connected(shape, True):
                buckets[n].add(shape)
    return {n: tuple(sorted(shapes)) for n, shapes in buckets.items()}


_SHAPES = _build_shapes()
_SHAPES_ASYM = {
    axis: {n: tuple(shape for shape in _SHAPES[n]
                    if _axis_asymmetric(shape, axis)) for n in _SHAPES}
    for axis in ("horizontal", "vertical")
}
_SHAPES_BY_CONNECTION = {
    axis: {n: {four: tuple(shape for shape in bucket[n]
                           if _connected(shape, False) == four)
               for four in (True, False)} for n in _SHAPES}
    for axis, bucket in ((None, _SHAPES),
                         ("horizontal", _SHAPES_ASYM["horizontal"]),
                         ("vertical", _SHAPES_ASYM["vertical"]))
}


def _axis(direction):
    return "horizontal" if direction in ("right", "left") else "vertical"


def _direction_and_gap(source, target):
    sr0, sr1, sc0, sc1 = _bbox(source)
    tr0, tr1, tc0, tc1 = _bbox(target)
    row_overlap = max(sr0, tr0) <= min(sr1, tr1)
    col_overlap = max(sc0, tc0) <= min(sc1, tc1)
    possible = []
    if row_overlap and sc1 < tc0:
        possible.append(("right", tc0 - sc1 - 1))
    if row_overlap and tc1 < sc0:
        possible.append(("left", sc0 - tc1 - 1))
    if col_overlap and sr1 < tr0:
        possible.append(("down", tr0 - sr1 - 1))
    if col_overlap and tr1 < sr0:
        possible.append(("up", sr0 - tr1 - 1))
    assert len(possible) == 1
    return possible[0]


def _move_and_reflect(source, target, direction):
    sr0, sr1, sc0, sc1 = _bbox(source)
    tr0, tr1, tc0, tc1 = _bbox(target)
    if direction == "right":
        moved = {(r, c + tc0 - sc1 - 1) for r, c in source}
        reflected = {(r, 2 * tc0 - 1 - c) for r, c in moved}
    elif direction == "left":
        moved = {(r, c + tc1 + 1 - sc0) for r, c in source}
        reflected = {(r, 2 * tc1 + 1 - c) for r, c in moved}
    elif direction == "down":
        moved = {(r + tr0 - sr1 - 1, c) for r, c in source}
        reflected = {(2 * tr0 - 1 - r, c) for r, c in moved}
    elif direction == "up":
        moved = {(r + tr1 + 1 - sr0, c) for r, c in source}
        reflected = {(2 * tr1 + 1 - r, c) for r, c in moved}
    else:
        raise ValueError("unknown direction")
    return moved, reflected


def _face_contact(moved, target, direction):
    tr0, tr1, tc0, tc1 = _bbox(target)
    if direction == "right":
        return any((r, tc0 - 1) in moved for r in range(tr0, tr1 + 1))
    if direction == "left":
        return any((r, tc1 + 1) in moved for r in range(tr0, tr1 + 1))
    if direction == "down":
        return any((tr0 - 1, c) in moved for c in range(tc0, tc1 + 1))
    return any((tr1 + 1, c) in moved for c in range(tc0, tc1 + 1))


def _choose_directions(rng, num_train, num_test):
    mode = str(rng.choice(("simple", "familiar", "heldout", "mixed"),
                          p=(0.16, 0.27, 0.29, 0.28)))
    horizontal = ("right", "left")
    vertical = ("down", "up")
    if mode == "simple":
        direction = str(rng.choice(_DIRECTIONS))
        return [direction] * num_train, [direction] * num_test

    if mode == "heldout":
        held = str(rng.choice(_DIRECTIONS))
        other_h = [d for d in horizontal if d != held]
        other_v = [d for d in vertical if d != held]
        train = [str(rng.choice(other_h)), str(rng.choice(other_v))]
        choices = tuple(d for d in _DIRECTIONS if d != held)
        train.extend(str(rng.choice(choices)) for _ in range(num_train - 2))
        test = [held] * num_test
        if num_test > 1 and rng.random() < 0.35:
            test[-1] = str(rng.choice(train))
    else:
        train = [str(rng.choice(horizontal)), str(rng.choice(vertical))]
        train.extend(str(rng.choice(_DIRECTIONS)) for _ in range(num_train - 2))
        choices = tuple(d for d in _DIRECTIONS if d in train) if mode == "familiar" else _DIRECTIONS
        test = [str(rng.choice(choices)) for _ in range(num_test)]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _choose_colors(rng, mode, base, index):
    if mode == "fixed":
        return base
    if mode == "swap":
        return base if index % 2 == 0 else (base[0], base[2], base[1])
    return tuple(int(v) for v in rng.choice(10, size=3, replace=False))


def _sample_shape(rng, size, axis_asym=None):
    four_connected = bool(rng.random() < 0.55)
    bucket = _SHAPES_BY_CONNECTION[axis_asym][size][four_connected]
    if not bucket:
        bucket = _SHAPES_BY_CONNECTION[axis_asym][size][not four_connected]
    return bucket[int(rng.integers(len(bucket)))]


def _sample_gap(rng):
    band = int(rng.choice(4, p=(0.12, 0.36, 0.34, 0.18)))
    low, high = ((1, 2), (2, 4), (4, 8), (8, 13))[band]
    return int(rng.integers(low, high))


def _sample_canvas_axis(rng, span):
    available = 30 - span
    assert available >= 0
    band = int(rng.choice(4, p=(0.25, 0.35, 0.29, 0.11)))
    if band == 0:
        extra = int(rng.integers(0, min(available, 2) + 1))
    elif band == 1:
        extra = int(rng.integers(min(available, 2), min(available, 8) + 1))
    elif band == 2:
        extra = int(rng.integers(min(available, 6), min(available, 18) + 1))
    else:
        extra = available if rng.random() < 0.5 else int(rng.integers(0, available + 1))
    before = int(rng.integers(0, extra + 1))
    return before, extra - before


def _construct_pair(rng, direction, shape, marker_size, colors):
    _, sr1, _, sc1 = _bbox(shape)
    extent = sr1 + 1 if _axis(direction) == "horizontal" else sc1 + 1
    if direction == "right":
        leading = {r for r, c in shape if c == sc1}
    elif direction == "left":
        leading = {r for r, c in shape if c == 0}
    elif direction == "down":
        leading = {c for r, c in shape if r == sr1}
    else:
        leading = {c for r, c in shape if r == 0}
    alignments = [a for a in range(-marker_size + 1, extent)
                  if any(a <= x < a + marker_size for x in leading)]
    alignment = alignments[int(rng.integers(len(alignments)))]
    gap = _sample_gap(rng)
    if direction == "right":
        tr, tc = alignment, sc1 + 1 + gap
    elif direction == "left":
        tr, tc = alignment, -marker_size - gap
    elif direction == "down":
        tr, tc = sr1 + 1 + gap, alignment
    else:
        tr, tc = -marker_size - gap, alignment
    target = {(tr + r, tc + c) for r in range(marker_size)
              for c in range(marker_size)}
    source = set(shape)
    moved, reflected = _move_and_reflect(source, target, direction)
    assert _face_contact(moved, target, direction)
    all_cells = source | target | moved | reflected
    r0, r1, c0, c1 = _bbox(all_cells)
    span_h, span_w = r1 - r0 + 1, c1 - c0 + 1
    before_r, after_r = _sample_canvas_axis(rng, span_h)
    before_c, after_c = _sample_canvas_axis(rng, span_w)
    height = span_h + before_r + after_r
    width = span_w + before_c + after_c
    # Keep the uniform background identifiable by its cell count, even on
    # tight canvases with a three by three marker.
    minimum_area = len(source) + len(target) + max(len(source), len(target)) + 1
    while height * width < minimum_area:
        if height <= width and height < 30:
            height += 1
        elif width < 30:
            width += 1
        else:
            height += 1
    dr, dc = before_r - r0, before_c - c0
    source = {(r + dr, c + dc) for r, c in source}
    target = {(r + dr, c + dc) for r, c in target}
    moved = {(r + dr, c + dc) for r, c in moved}
    reflected = {(r + dr, c + dc) for r, c in reflected}
    background, source_color, target_color = colors
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    for r, c in source:
        inp[r, c] = source_color
    for r, c in target:
        inp[r, c] = target_color
    for r, c in moved:
        out[r, c] = source_color
    for r, c in reflected:
        out[r, c] = target_color
    return {"input": inp, "output": out}


def _validate_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _counts(rng, seed, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        lower = max(2, num_test + 1)
        num_train = lower + (seed // 4) % (11 - lower)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _pair_key(pair):
    inp, out = pair["input"], pair["output"]
    return inp.shape, inp.tobytes(), out.tobytes()


def _original_pair(pair):
    inp, out = pair["input"], pair["output"]
    if inp.shape != (13, 13) or out.shape != (13, 13):
        return False
    for src, target, moved, reflected in _ORIGINAL:
        expected_in = np.full((13, 13), 7, dtype=np.int8)
        expected_out = np.full((13, 13), 7, dtype=np.int8)
        for r, c in src:
            expected_in[r, c] = 2
        for r, c in target:
            expected_in[r, c] = 5
        for r, c in moved:
            expected_out[r, c] = 2
        for r, c in reflected:
            expected_out[r, c] = 5
        if np.array_equal(inp, expected_in) and np.array_equal(out, expected_out):
            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, seed, num_train, num_test)
    train_dirs, test_dirs = _choose_directions(rng, num_train, num_test)
    directions = train_dirs + test_dirs
    total = len(directions)

    # Palette behavior is a puzzle-level style, while the geometric roles
    # remain visible in each input and can change literal colors by pair.
    palette_mode = str(rng.choice(("fixed", "swap", "free"), p=(0.25, 0.35, 0.40)))
    base_colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    colors = [_choose_colors(rng, palette_mode, base_colors, i) for i in range(total)]
    marker_sizes = [2 if rng.random() < 0.73 else 3 for _ in range(total)]
    if 3 in marker_sizes[num_train:] and 3 not in marker_sizes[:num_train]:
        marker_sizes[int(rng.integers(num_train))] = 3
    if 2 in marker_sizes[num_train:] and 2 not in marker_sizes[:num_train]:
        marker_sizes[int(rng.integers(num_train))] = 2

    needed_axes = {_axis(d) for d in test_dirs}
    witness_indices = {}
    for index, direction in enumerate(train_dirs):
        axis = _axis(direction)
        if axis in needed_axes and axis not in witness_indices:
            witness_indices[axis] = index
    assert set(witness_indices) == needed_axes

    pairs = []
    used_inputs = set()
    for index, direction in enumerate(directions):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        axis_asym = _axis(direction) if index in witness_indices.values() else None
        size = int(pair_rng.choice((3, 4, 5, 6, 7, 8),
                                   p=(0.10, 0.38, 0.18, 0.16, 0.11, 0.07)))
        repeatable = [p for p in pairs if len(p["shape"]) == size and
                      (axis_asym is None or _axis_asymmetric(p["shape"], axis_asym))]
        if repeatable and pair_rng.random() < 0.38:
            shape = repeatable[int(pair_rng.integers(len(repeatable)))]["shape"]
        else:
            shape = _sample_shape(pair_rng, size, axis_asym)
        for attempt in range(30):
            pair = _construct_pair(pair_rng, direction, shape, marker_sizes[index],
                                   colors[index])
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in used_inputs and not _original_pair(pair):
                break
            shape = _sample_shape(pair_rng, size, axis_asym)
        else:
            raise AssertionError("could not construct a distinct pair")
        used_inputs.add(key)
        pairs.append({"input": pair["input"], "output": pair["output"],
                      "shape": shape})
    puzzle = {
        "train": [{"input": p["input"], "output": p["output"]}
                  for p in pairs[:num_train]],
        "test": [{"input": p["input"], "output": p["output"]}
                 for p in pairs[num_train:]],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    background = int(values[int(np.argmax(counts))])
    objects = [(int(v), {(int(r), int(c)) for r, c in np.argwhere(inp == v)})
               for v in values if int(v) != background]
    markers = [(color, cells) for color, cells in objects if _square(cells)]
    assert len(markers) == 1
    target_color, target = markers[0]
    source_color, source = next((color, cells) for color, cells in objects
                                if color != target_color)
    assert 3 <= len(source) <= 8
    assert _connected(source, True) and not _square(source)
    marker_size = int(len(target) ** 0.5)
    assert marker_size in (2, 3)
    direction, gap = _direction_and_gap(source, target)
    assert 1 <= gap <= 12

    # Independently derive the output from the input. For each source cell,
    # first shift its leading bounding edge to the square's facing edge,
    # then invert its signed distance from that edge into the square side.
    sr0, sr1, sc0, sc1 = _bbox(source)
    tr0, tr1, tc0, tc1 = _bbox(target)
    if direction == "right":
        shift = tc0 - 1 - sc1
        moved = {(r, c + shift) for r, c in source}
        reflected = {(r, tc0 + (sc1 - c)) for r, c in source}
    elif direction == "left":
        shift = tc1 + 1 - sc0
        moved = {(r, c + shift) for r, c in source}
        reflected = {(r, tc1 - (c - sc0)) for r, c in source}
    elif direction == "down":
        shift = tr0 - 1 - sr1
        moved = {(r + shift, c) for r, c in source}
        reflected = {(tr0 + (sr1 - r), c) for r, c in source}
    else:
        shift = tr1 + 1 - sr0
        moved = {(r + shift, c) for r, c in source}
        reflected = {(tr1 - (r - sr0), c) for r, c in source}
    assert _face_contact(moved, target, direction)
    assert reflected & target
    assert not moved & reflected
    h, w = inp.shape
    assert all(0 <= r < h and 0 <= c < w for r, c in moved | reflected)
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    for r, c in moved:
        expected[r, c] = source_color
    for r, c in reflected:
        expected[r, c] = target_color
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    assert len(moved) == len(reflected) == len(source)
    return {"direction": direction, "axis": _axis(direction), "gap": gap,
            "shape": _normalize(source), "source_size": len(source),
            "marker_size": marker_size, "colors": (background, source_color,
                                                      target_color),
            "asym": _axis_asymmetric(source, _axis(direction)),
            "four_connected": _connected(source, False)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            assert pair["input"].dtype == pair["output"].dtype
            input_key = (pair["input"].shape, pair["input"].tobytes())
            pair_key = _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _original_pair(pair)
            analyses[split].append(_analyze_pair(pair))
    test_axes = {a["axis"] for a in analyses["test"]}
    for axis in test_axes:
        assert any(a["axis"] == axis and a["asym"] for a in analyses["train"])
    test_sizes = {a["marker_size"] for a in analyses["test"]}
    assert test_sizes <= {a["marker_size"] for a in analyses["train"]}


def _puzzles_equal(first, second):
    return all(np.array_equal(a[side], b[side]) for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for side in ("input", "output")) and all(
                   len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzles_equal(generate_puzzle(288), generate_puzzle(288))
    assert _puzzles_equal(generate_puzzle(918273, 7, 3),
                          generate_puzzle(918273, 7, 3))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(4000 + 10 * nt + nq, nt, nq)
        assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(5000 + (nt or 0) + (nq or 0), nt, nq)
        if nt is not None:
            assert len(p["train"]) == nt
        if nq is not None:
            assert len(p["test"]) == nq
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(6000, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    all_analyses = []
    shapes, dimensions, layouts, roles, pair_signatures = set(), set(), set(), set(), set()
    repeated_shape_puzzles = variable_color_puzzles = role_swap_puzzles = 0
    diagonal_pairs = four_connected_pairs = three_by_three_fours = 0
    asymmetric_train = zero_margin = varied_sizes = varied_markers = 0
    simple_puzzles = heldout_puzzles = familiar_puzzles = mixed_tests = 0
    train_directions_seen, test_directions_seen = set(), set()
    direction_sets = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        analyses = {split: [_analyze_pair(pair) for pair in puzzle[split]]
                    for split in ("train", "test")}
        directions = tuple(a["direction"] for split in ("train", "test")
                           for a in analyses[split])
        direction_sets.add((tuple(sorted(set(directions[:len(puzzle["train"])]))),
                            tuple(sorted(set(directions[len(puzzle["train"]):])))))
        train_dirs = set(directions[:len(puzzle["train"])])
        test_dirs = set(directions[len(puzzle["train"]):])
        train_directions_seen.update(train_dirs)
        test_directions_seen.update(test_dirs)
        simple_puzzles += len(train_dirs | test_dirs) == 1
        heldout_puzzles += bool(test_dirs - train_dirs)
        familiar_puzzles += bool(test_dirs & train_dirs)
        mixed_tests += len(test_dirs) >= 2
        local_shapes = [a["shape"] for split in ("train", "test")
                        for a in analyses[split]]
        local_colors = [a["colors"] for split in ("train", "test")
                        for a in analyses[split]]
        repeated_shape_puzzles += len(local_shapes) > len(set(local_shapes))
        variable_color_puzzles += len(set(local_colors)) > 1
        role_swap_puzzles += any(a[0] == b[0] and a[1] == b[2]
                                 and a[2] == b[1] for a in local_colors
                                 for b in local_colors if a != b)
        varied_sizes += len({a["source_size"] for split in ("train", "test")
                             for a in analyses[split]}) > 1
        varied_markers += len({a["marker_size"] for split in ("train", "test")
                               for a in analyses[split]}) > 1
        asymmetric_train += any(a["asym"] for a in analyses["train"])
        for split in ("train", "test"):
            for pair, analysis in zip(puzzle[split], analyses[split]):
                all_analyses.append(analysis)
                shapes.add(analysis["shape"])
                dimensions.add(pair["input"].shape)
                roles.add(analysis["colors"])
                pair_signatures.add(_pair_key(pair))
                diagonal_pairs += not analysis["four_connected"]
                four_connected_pairs += analysis["four_connected"]
                r0, r1, c0, c1 = _bbox(analysis["shape"])
                three_by_three_fours += (analysis["source_size"] == 4
                                         and r1 - r0 == 2 and c1 - c0 == 2)
                fg = np.argwhere(pair["input"] != analysis["colors"][0])
                zero_margin += (np.any(fg[:, 0] == 0) or np.any(fg[:, 1] == 0)
                                or np.any(fg[:, 0] == pair["input"].shape[0] - 1)
                                or np.any(fg[:, 1] == pair["input"].shape[1] - 1))
                layouts.add((pair["input"].shape, analysis["direction"],
                             analysis["gap"], analysis["shape"]))
    assert len(pair_signatures) == len(all_analyses)
    assert len(shapes) >= 90 and len(dimensions) >= 50 and len(layouts) >= 150
    assert len(roles) >= 35 and len(direction_sets) >= 8
    assert {a["direction"] for a in all_analyses} == set(_DIRECTIONS)
    assert train_directions_seen == test_directions_seen == set(_DIRECTIONS)
    assert {a["source_size"] for a in all_analyses} == set(range(3, 9))
    assert {a["marker_size"] for a in all_analyses} == {2, 3}
    assert max(a["gap"] for a in all_analyses) >= 10
    assert diagonal_pairs >= 8 and four_connected_pairs >= 8
    assert three_by_three_fours >= 3 and zero_margin >= 15
    assert repeated_shape_puzzles >= 2 and variable_color_puzzles >= 15
    assert role_swap_puzzles >= 2 and varied_sizes >= 20 and varied_markers >= 5
    assert simple_puzzles >= 2 and heldout_puzzles >= 4
    assert familiar_puzzles >= 4 and mixed_tests >= 2
    assert asymmetric_train == 42
