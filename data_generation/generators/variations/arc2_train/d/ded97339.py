"""Sparse original markers connect along their common rows and columns."""

import numpy as np


# Exact source inputs, expressed as marker coordinates for the source palette.
_SOURCE_MARKERS = (
    ((1, 7), (3, 2), (6, 5), (6, 9), (8, 2)),
    ((3, 1), (3, 6), (7, 4), (9, 6)),
    ((1, 1), (1, 5), (4, 1), (7, 5), (7, 9), (9, 2)),
    ((1, 1), (2, 3), (3, 7), (5, 1), (8, 4), (8, 7)),
)

# Incidence patterns, not pixel templates: each row and column index is mapped
# independently to sampled coordinates. Extra markers and isolates are composed
# with them below. In particular, cross_blank has no seed at its intersection.
_PATTERNS = {
    "horizontal": ((0, 0), (0, 1)),
    "vertical": ((0, 0), (1, 0)),
    "three_row": ((0, 0), (0, 1), (0, 2)),
    "three_col": ((0, 0), (1, 0), (2, 0)),
    "parallel_h": ((0, 0), (0, 1), (1, 2), (1, 3)),
    "parallel_v": ((0, 0), (1, 0), (2, 1), (3, 1)),
    "independent": ((0, 0), (0, 1), (1, 2), (2, 2)),
    "l": ((0, 0), (0, 1), (1, 0)),
    "chain": ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    "branch": ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    "rectangle": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "cross_seed": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "cross_blank": ((1, 0), (1, 2), (0, 1), (2, 1)),
    "source_mix": ((0, 0), (0, 1), (1, 0), (2, 2), (3, 2)),
    "staircase": ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)),
    "lattice": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
}
_ANCHORS = ("l", "chain", "branch", "rectangle", "cross_seed", "source_mix", "staircase")
_INDEPENDENT_EVIDENCE = ("independent", "cross_blank", "parallel_h", "parallel_v")
_TEST_KINDS = tuple(_PATTERNS) + ("lattice", "lattice", "lattice", "cross_blank", "source_mix", "rectangle")


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _counts(seed, rng, num_train, num_test):
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        nt = ns + 1 + (seed // 4) % (10 - ns)
    elif nt is None:
        nt = int(rng.integers(ns + 1, 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2
    return int(colors[index]), int(colors[1 - index])


def _solve(grid, background, marker):
    result = grid.copy()
    seeds = grid == marker
    for row in range(grid.shape[0]):
        cols = np.flatnonzero(seeds[row])
        if len(cols) >= 2:
            result[row, cols[0]:cols[-1] + 1] = marker
    for col in range(grid.shape[1]):
        rows = np.flatnonzero(seeds[:, col])
        if len(rows) >= 2:
            result[rows[0]:rows[-1] + 1, col] = marker
    return result


def _sample_size(rng, minimum):
    band = int(rng.choice(3, p=(0.24, 0.48, 0.28)))
    low, high = ((6, 9), (10, 18), (19, 30))[band]
    low = max(low, minimum)
    if low > high:
        low, high = minimum, 30
    return int(rng.integers(low, high + 1))


def _place_pattern(rng, height, width, kind):
    pattern = _PATTERNS[kind]
    if kind == "lattice":
        middle = int(rng.integers(2, height - 2))
        rows = (int(rng.integers(2)), middle, height - 1 - int(rng.integers(2)))
        cols = (int(rng.integers(2)), width - 1 - int(rng.integers(2)))
        return {(rows[r], cols[c]) for r, c in pattern}
    row_count = 1 + max(r for r, _ in pattern)
    col_count = 1 + max(c for _, c in pattern)
    rows = sorted(int(x) for x in rng.choice(height, row_count, replace=False))
    cols = sorted(int(x) for x in rng.choice(width, col_count, replace=False))
    if rng.integers(2):
        rows.reverse()
    if rng.integers(2):
        cols.reverse()
    return {(rows[r], cols[c]) for r, c in pattern}


def _add_extensions(rng, coords, height, width, kind):
    if kind in ("horizontal", "vertical", "three_row", "three_col"):
        return coords
    for _ in range(int(rng.choice(4, p=(0.42, 0.32, 0.18, 0.08)))):
        used_rows = {r for r, _ in coords}
        used_cols = {c for _, c in coords}
        if rng.integers(2):
            row = int(rng.choice(sorted(used_rows)))
            available = [c for c in range(width) if c not in used_cols]
            if available:
                coords.add((row, int(rng.choice(available))))
        else:
            col = int(rng.choice(sorted(used_cols)))
            available = [r for r in range(height) if r not in used_rows]
            if available:
                coords.add((int(rng.choice(available)), col))
    return coords


def _cue_isolate(rng, coords, height, width):
    used_rows = {r for r, _ in coords}
    used_cols = {c for _, c in coords}
    candidates = []
    for row in used_rows:
        columns = sorted(c for r, c in coords if r == row)
        if len(columns) >= 2:
            for col in range(columns[0] + 1, columns[-1]):
                if col not in used_cols:
                    candidates.extend((r, col) for r in range(height)
                                      if r not in used_rows and abs(r - row) >= 2)
    for col in used_cols:
        rows = sorted(r for r, c in coords if c == col)
        if len(rows) >= 2:
            for row in range(rows[0] + 1, rows[-1]):
                if row not in used_rows:
                    candidates.extend((row, c) for c in range(width)
                                      if c not in used_cols and abs(c - col) >= 2)
    if not candidates:
        return False
    coords.add(candidates[int(rng.integers(len(candidates)))])
    return True


def _add_isolates(rng, coords, height, width, count):
    for _ in range(count):
        used_rows = {r for r, _ in coords}
        used_cols = {c for _, c in coords}
        rows = [r for r in range(height) if r not in used_rows]
        cols = [c for c in range(width) if c not in used_cols]
        if not rows or not cols:
            return False
        # Frequently choose an exact one-row near miss; its column remains free.
        near = [r for r in rows if any(abs(r - original) == 1 for original in used_rows)]
        row_pool = near if near and rng.integers(3) == 0 else rows
        coords.add((int(rng.choice(row_pool)), int(rng.choice(cols))))
    return True


def _source_copy(grid, background, marker):
    if grid.shape != (10, 10) or (background, marker) != (0, 8):
        return False
    coords = tuple(map(tuple, np.argwhere(grid == marker).tolist()))
    return coords in _SOURCE_MARKERS


def _make_pair(rng, kind, palette, isolate_count, cue=False):
    pattern = _PATTERNS[kind]
    min_height = max(6, 2 + max(r for r, _ in pattern) + isolate_count)
    min_width = max(6, 2 + max(c for _, c in pattern) + isolate_count)
    for _ in range(350):
        if kind == "lattice":
            height = int(rng.integers(6, 10))
            width = int(rng.integers(6, 10))
        else:
            height = _sample_size(rng, min_height)
            width = _sample_size(rng, min_width)
        coords = _add_extensions(rng, _place_pattern(rng, height, width, kind), height, width, kind)
        if cue:
            if not _cue_isolate(rng, coords, height, width):
                continue
            remaining = isolate_count - 1
        else:
            remaining = isolate_count
        if not _add_isolates(rng, coords, height, width, remaining):
            continue
        background, marker = palette
        grid = np.full((height, width), background, dtype=np.int8)
        for row, col in coords:
            grid[row, col] = marker
        if 2 * len(coords) >= grid.size:
            continue
        output = _solve(grid, background, marker)
        if np.array_equal(grid, output) or _source_copy(grid, background, marker):
            continue
        return {"input": grid, "output": output}
    # Deterministic last resort for legal requests. Widely spaced rank
    # coordinates leave room for the requested isolates and a cue.
    coords = {(2 + 4 * r, 2 + 4 * c) for r, c in pattern}
    if cue:
        used_rows = {r for r, _ in coords}
        used_cols = {c for _, c in coords}
        horizontal = next(((r, min(c for rr, c in coords if rr == r) + 2)
                           for r in used_rows if sum(rr == r for rr, _ in coords) >= 2), None)
        if horizontal is not None:
            coords.add((20, horizontal[1]))
        else:
            col = next(c for c in used_cols if sum(cc == c for _, cc in coords) >= 2)
            coords.add((min(r for r, cc in coords if cc == col) + 2, 20))
    for index in range(isolate_count - int(cue)):
        coords.add((21 + index, 21 + index))
    background, marker = palette
    grid = np.full((30, 30), background, dtype=np.int8)
    for row, col in coords:
        grid[row, col] = marker
    return {"input": grid, "output": _solve(grid, background, marker)}


def _palette(rng, base, flexible, index):
    if not flexible:
        return base
    if index == 0:
        return base
    if index == 1:
        return base[::-1]
    if rng.integers(3) == 0:
        return base if rng.integers(2) else base[::-1]
    colors = rng.choice(10, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # The stable convention is semantic: the sparse color denotes original
    # markers, and their own color fills aligned intervals. Literal colors may
    # change between pairs because both roles are visible in each input.
    colors = rng.choice(10, size=2, replace=False)
    base = (int(colors[0]), int(colors[1]))
    flexible = bool(rng.integers(2))
    anchor = _ANCHORS[int(rng.integers(len(_ANCHORS)))]
    second = _INDEPENDENT_EVIDENCE[int(rng.integers(len(_INDEPENDENT_EVIDENCE)))]
    first_two = [anchor, second]
    if rng.integers(2):
        first_two.reverse()
    train_kinds = first_two + [str(rng.choice(tuple(_PATTERNS))) for _ in range(train_count - 2)]
    test_kinds = [str(rng.choice(_TEST_KINDS)) for _ in range(test_count)]

    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (section, kind) in enumerate(
        [("train", k) for k in train_kinds] + [("test", k) for k in test_kinds]
    ):
        palette = _palette(rng, base, flexible, index)
        cue = section == "train" and index < 2
        isolates = (1 + int(rng.integers(2))) if cue else int(rng.choice(4, p=(0.27, 0.34, 0.25, 0.14)))
        for _ in range(40):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _make_pair(pair_rng, kind, palette, isolates, cue)
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[section].append(pair)
                break
        else:
            raise RuntimeError("could not generate a unique input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _features(grid, marker):
    seeds = grid == marker
    row_counts = np.count_nonzero(seeds, axis=1)
    col_counts = np.count_nonzero(seeds, axis=0)
    coords = np.argwhere(seeds)
    isolated = sum(row_counts[r] == 1 and col_counts[c] == 1 for r, c in coords)
    shared = any(row_counts[r] >= 2 and col_counts[c] >= 2 for r, c in coords)
    return {
        "horizontal": bool(np.any(row_counts >= 2)),
        "vertical": bool(np.any(col_counts >= 2)),
        "shared": bool(shared),
        "isolated": int(isolated),
        "three_row": bool(np.any(row_counts >= 3)),
        "three_col": bool(np.any(col_counts >= 3)),
    }


def _nonrecursive_cue(input_grid, output_grid, marker):
    seeds = input_grid == marker
    added = (output_grid == marker) & ~seeds
    row_counts = np.count_nonzero(seeds, axis=1)
    col_counts = np.count_nonzero(seeds, axis=0)
    for row, col in np.argwhere(seeds):
        if row_counts[row] != 1 or col_counts[col] != 1:
            continue
        for other in np.flatnonzero(added[:, col]):
            if abs(int(other) - row) >= 2 and np.any(~(output_grid[min(row, other) + 1:max(row, other), col] == marker)):
                return True
        for other in np.flatnonzero(added[row]):
            if abs(int(other) - col) >= 2 and np.any(~(output_grid[row, min(col, other) + 1:max(col, other)] == marker)):
                return True
    return False


def _review_indicators(grid, output, marker):
    coords = [tuple(map(int, x)) for x in np.argwhere(grid == marker)]
    rows = {r: sorted(c for rr, c in coords if rr == r) for r, _ in coords}
    cols = {c: sorted(r for r, cc in coords if cc == c) for _, c in coords}
    horizontal = {r: (cs[0], cs[-1]) for r, cs in rows.items() if len(cs) >= 2}
    vertical = {c: (rs[0], rs[-1]) for c, rs in cols.items() if len(rs) >= 2}
    isolates = [(r, c) for r, c in coords if len(rows[r]) == len(cols[c]) == 1]
    adjacent = any(b - a == 1 for groups in (rows, cols) for group in groups.values()
                   for a, b in zip(group, group[1:]))
    long_gap = any(b - a >= 2 for groups in (rows, cols) for group in groups.values()
                   for a, b in zip(group, group[1:]))
    crossing = any(r in horizontal and c in vertical
                   and horizontal[r][0] < c < horizontal[r][1]
                   and vertical[c][0] < r < vertical[c][1]
                   for r, c in np.argwhere((output == marker) & (grid != marker)))
    return {
        "near": any(any(abs(r - rr) == 1 for rr in horizontal) or
                    any(abs(c - cc) == 1 for cc in vertical) for r, c in isolates),
        "inside": any(left < c < right for _, c in isolates for left, right in horizontal.values()),
        "outside": any(c < left or c > right for _, c in isolates for left, right in horizontal.values()),
        "above": any(r < rr for r, _ in isolates for rr in horizontal),
        "below": any(r > rr for r, _ in isolates for rr in horizontal),
        "adjacent_long": adjacent and long_gap,
        "crossing": crossing,
        "dense": np.count_nonzero(output == marker) * 2 > output.size,
        "zero_isolates": len(isolates) == 0,
        "many_isolates": len(isolates) >= 2,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    training = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, result = pair["input"], pair["output"]
            for arr in (grid, result):
                assert isinstance(arr, np.ndarray) and arr.ndim == 2
                assert all(1 <= size <= 30 for size in arr.shape)
                assert np.issubdtype(arr.dtype, np.integer)
                assert 0 <= int(arr.min()) <= int(arr.max()) <= 9
            assert grid.shape == result.shape
            key = _grid_key(grid)
            pair_key = key, _grid_key(result)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            background, marker = _roles(grid)
            assert set(np.unique(result).tolist()) == {background, marker}
            assert np.array_equal(result, _solve(grid, background, marker))
            assert not np.array_equal(grid, result)
            assert not _source_copy(grid, background, marker)
            if section == "train":
                training.append((grid, result, marker, _features(grid, marker)))
    assert any(f["horizontal"] for _, _, _, f in training)
    assert any(f["vertical"] for _, _, _, f in training)
    assert any(f["shared"] for _, _, _, f in training)
    assert any(f["isolated"] >= 1 for _, _, _, f in training)
    assert any(_nonrecursive_cue(g, o, m) for g, o, m, _ in training)


def _same_puzzle(a, b):
    return all(np.array_equal(x[role], y[role])
               for section in ("train", "test")
               for x, y in zip(a[section], b[section])
               for role in ("input", "output")) and all(len(a[s]) == len(b[s]) for s in ("train", "test"))


def self_test() -> None:
    assert not _same_puzzle(generate_puzzle(607), generate_puzzle(608))
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(607, train, test)
        validate_generated_puzzle(p)
        assert _same_puzzle(p, generate_puzzle(607, train, test))
        if train is not None:
            assert len(p["train"]) == train
        if test is not None:
            assert len(p["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(12, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    train_counts, test_counts, shapes, palettes, signatures = set(), set(), set(), set(), set()
    markers, isolates, additions = set(), set(), set()
    both, shared, triples, crossings, single_orientation = 0, 0, 0, 0, 0
    variable_palette = 0
    review_counts = {section: {name: 0 for name in
                     ("near", "inside", "outside", "above", "below", "adjacent_long",
                      "crossing", "dense", "zero_isolates", "many_isolates")}
                     for section in ("train", "test")}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        local_palettes = set()
        for section in ("train", "test"):
            for pair in p[section]:
                g, o = pair["input"], pair["output"]
                bg, marker = _roles(g)
                # A separate row/column implementation checks the visible rule.
                predicted = g.copy()
                coords = np.argwhere(g == marker)
                for a in coords:
                    for b in coords:
                        if a[0] == b[0]:
                            predicted[a[0], min(a[1], b[1]):max(a[1], b[1]) + 1] = marker
                        if a[1] == b[1]:
                            predicted[min(a[0], b[0]):max(a[0], b[0]) + 1, a[1]] = marker
                assert np.array_equal(predicted, o)
                f = _features(g, marker)
                for name, present in _review_indicators(g, o, marker).items():
                    review_counts[section][name] += bool(present)
                both += f["horizontal"] and f["vertical"]
                shared += f["shared"]
                triples += f["three_row"] or f["three_col"]
                single_orientation += f["horizontal"] != f["vertical"]
                # An added cell belongs to both an original horizontal and
                # vertical span, but is absent from the input.
                seed_rows = np.count_nonzero(g == marker, axis=1)
                seed_cols = np.count_nonzero(g == marker, axis=0)
                crossings += any(seed_rows[r] >= 2 and seed_cols[c] >= 2
                                 for r, c in np.argwhere((o == marker) & (g == bg)))
                shapes.add(g.shape)
                palettes.add((bg, marker))
                local_palettes.add((bg, marker))
                markers.add(len(coords))
                isolates.add(f["isolated"])
                additions.add(int(np.count_nonzero(o != g)))
                signatures.add((f["horizontal"], f["vertical"], f["shared"],
                                f["isolated"], f["three_row"], f["three_col"], len(coords)))
        variable_palette += len(local_palettes) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 55 and len(palettes) >= 15 and len(signatures) >= 20
    assert len(markers) >= 6 and {0, 1, 2}.issubset(isolates)
    assert len(additions) >= 15 and both > 30 and shared > 20
    assert triples > 10 and crossings > 0 and single_orientation > 5
    assert variable_palette >= 8
    for section in ("train", "test"):
        for name in ("near", "inside", "outside", "above", "below",
                     "adjacent_long", "crossing", "zero_isolates", "many_isolates"):
            assert review_counts[section][name] > 0, (section, name)
    assert review_counts["train"]["dense"] > 0
    assert review_counts["test"]["dense"] > 0
