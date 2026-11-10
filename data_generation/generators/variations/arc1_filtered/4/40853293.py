"""Generate endpoint-joining ARC puzzles with vertical priority at crossings."""

from collections import Counter

import numpy as np


# Original inputs are represented only to prevent exact copies.
_SOURCE = {
    ((30, 20), ((2, 2, 6, 13, 6), (3, 6, 3, 6, 11),
                (5, 20, 2, 20, 7), (6, 18, 4, 27, 4),
                (8, 12, 14, 17, 14))),
    ((20, 10), ((3, 4, 2, 4, 7), (4, 2, 3, 10, 3),
                (7, 8, 2, 8, 5), (8, 14, 1, 14, 6),
                (9, 12, 5, 18, 5))),
    ((20, 20), ((2, 3, 1, 3, 16), (3, 1, 3, 18, 3),
                (5, 2, 9, 9, 9), (7, 7, 7, 7, 13),
                (8, 14, 8, 14, 14))),
}


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _decode(grid):
    bg = _background(grid)
    segments = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        points = np.argwhere(grid == color)
        assert points.shape == (2, 2)
        (r0, c0), (r1, c1) = [tuple(map(int, p)) for p in points]
        assert (r0 == r1) != (c0 == c1)
        assert abs(r1 - r0) + abs(c1 - c0) >= 2
        segments.append((color, (r0, c0, r1, c1)))
    assert len(segments) <= 9
    return bg, segments


def _contains(segment, point):
    r0, c0, r1, c1 = segment
    r, c = point
    return ((r0 == r1 == r and c0 <= c <= c1)
            or (c0 == c1 == c and r0 <= r <= r1))


def _crosses(h, v):
    row, c0, _, c1 = h
    r0, col, r1, _ = v
    return c0 < col < c1 and r0 < row < r1


def _segments_clear(segments):
    for i, segment in enumerate(segments):
        ends = ((segment[0], segment[1]), (segment[2], segment[3]))
        for j, other in enumerate(segments):
            if i != j and any(_contains(other, end) for end in ends):
                return False
    return True


def _crossing_counts(segments):
    vertical = [s for s in segments if s[1] == s[3]]
    horizontal = [s for s in segments if s[0] == s[2]]
    return [sum(_crosses(h, v) for v in vertical) for h in horizontal]


def _solve(grid):
    bg, decoded = _decode(grid)
    result = np.full(grid.shape, bg, dtype=np.int8)
    for vertical in (False, True):
        for color, (r0, c0, r1, c1) in decoded:
            if (c0 == c1) == vertical:
                result[r0:r1 + 1, c0:c1 + 1] = color
    return result


def _shape(rng):
    kind = int(rng.integers(0, 3))
    if kind == 0:
        side = int(rng.integers(10, 31))
        return side, side
    short = int(rng.integers(10, 28))
    long = int(rng.integers(short + 2, 31))
    return (long, short) if kind == 1 else (short, long)


def _span(rng, extent):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        length = int(rng.integers(2, 4))
    elif mode == 1:
        length = int(rng.integers(4, max(5, extent // 2 + 1)))
    elif mode == 2:
        length = int(rng.integers(max(2, extent // 2), extent))
    else:
        length = int(rng.integers(2, extent))
    length = min(length, extent - 1)
    start = int(rng.integers(0, extent - length))
    if rng.random() < 0.22:
        start = 0 if rng.random() < 0.5 else extent - length - 1
    return start, start + length


def _random_segment(rng, height, width, vertical):
    if vertical:
        r0, r1 = _span(rng, height)
        col = int(rng.integers(0, width))
        return r0, col, r1, col
    c0, c1 = _span(rng, width)
    row = int(rng.integers(0, height))
    return row, c0, row, c1


def _crossing_core(rng, height, width, required):
    row = int(rng.integers(2, height - 2))
    cols = sorted(int(x) for x in rng.choice(
        np.arange(2, width - 2), size=required, replace=False
    ))
    left = int(rng.integers(0, cols[0]))
    right = int(rng.integers(cols[-1] + 1, width))
    segments = [(row, left, row, right)]
    for col in cols:
        top = int(rng.integers(0, row))
        bottom = int(rng.integers(row + 1, height))
        segments.append((top, col, bottom, col))
    return segments


def _geometry(rng, height, width, nv, nh, target):
    """Build a chosen crossing class, then place all remaining pairs freely."""
    for _ in range(55):
        segments = _crossing_core(rng, height, width, target) if target else []
        if not _segments_clear(segments):
            continue
        remaining = [True] * (nv - target) + [False] * (nh - bool(target))
        rng.shuffle(remaining)
        for vertical in remaining:
            for _ in range(220):
                candidate = _random_segment(rng, height, width, vertical)
                trial = segments + [candidate]
                if not _segments_clear(trial):
                    continue
                crossings = _crossing_counts(trial)
                if target < 2 and crossings and max(crossings) > target:
                    continue
                segments = trial
                break
            else:
                break
        else:
            if max(_crossing_counts(segments), default=0) == target:
                return segments
    raise RuntimeError("Could not place endpoint pairs")


def _census(rng, first_train):
    if first_train:
        return [(1, 1), (1, 2), (2, 1), (2, 2),
                (2, 3), (3, 2), (3, 3), (4, 2)][int(rng.integers(0, 8))]
    total = int(rng.integers(2, 8))
    choices = [(nv, total - nv) for nv in range(total + 1)
               if nv <= 4 and total - nv <= 4]
    return choices[int(rng.integers(0, len(choices)))]


def _make_pair(rng, first_train=False, different_census=None):
    for _ in range(75):
        height, width = _shape(rng)
        nv, nh = _census(rng, first_train)
        if different_census == (nv, nh):
            continue
        if not nv or not nh:
            target = 0
        elif first_train:
            target = int(rng.integers(1, min(nv, 3) + 1))
        else:
            options = tuple(range(min(nv, 4) + 1))
            target = int(rng.choice(options))
        try:
            segments = _geometry(rng, height, width, nv, nh, target)
        except RuntimeError:
            continue
        bg = int(rng.integers(0, 10)) if rng.random() < 0.46 else 0
        colors = rng.choice([c for c in range(10) if c != bg],
                            size=nv + nh, replace=False)
        inp = np.full((height, width), bg, dtype=np.int8)
        for color, (r0, c0, r1, c1) in zip(colors, segments):
            inp[r0, c0] = inp[r1, c1] = int(color)
        out = _solve(inp)
        if not np.array_equal(inp, out):
            return {"input": inp, "output": out}
    raise RuntimeError("Could not construct a nontrivial pair")


def _checked_count(value, minimum, maximum, name):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return int(value)


def _census_from_pair(pair):
    _, decoded = _decode(pair["input"])
    nv = sum(s[1] == s[3] for _, s in decoded)
    return nv, len(decoded) - nv


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _checked_count(num_train, 2, 10, "num_train")
    num_test = _checked_count(num_test, 1, 4, "num_test")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 1) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # The only learned convention is vertical priority. Color and background
    # have local roles, readable directly from each input.
    train, test = [], []
    for index in range(num_train + num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        pair = _make_pair(pair_rng, first_train=(index == 0),
                          different_census=(None if index != 1 else
                                            _census_from_pair(train[0])))
        (train if index < num_train else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    bg, decoded = _decode(grid)
    return (grid.shape, tuple(sorted((c, *s) for c, s in decoded))) if bg == 0 else None


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_crossings = 0
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.astype(np.int8).tobytes())
            pair_key = (key, out.astype(np.int8).tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _signature(inp) not in _SOURCE
            bg, decoded = _decode(inp)
            assert 2 <= len(decoded) <= 7
            assert np.count_nonzero(inp == bg) > inp.size // 2
            assert _segments_clear([s for _, s in decoded])
            crossings = _crossing_counts([s for _, s in decoded])
            if regime == "train":
                train_crossings += sum(crossings)
            assert np.array_equal(_solve(inp), out)
    assert train_crossings >= 1
    assert _census_from_pair(train[0]) != _census_from_pair(train[1])


def _same_puzzle(first, second):
    return all(
        len(first[regime]) == len(second[regime]) and
        all(np.array_equal(a[k], b[k])
            for a, b in zip(first[regime], second[regime])
            for k in ("input", "output"))
        for regime in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _same_puzzle(generate_puzzle(112233, 7, 3),
                        generate_puzzle(112233, 7, 3))
    assert _same_puzzle(puzzles[5], generate_puzzle(5))
    assert not _same_puzzle(puzzles[5], puzzles[6])

    shapes, censuses, backgrounds, crossing_classes = set(), set(), set(), set()
    short, long, edge, double_train, zero_test, single_test = 0, 0, 0, 0, 0, 0
    coverage = Counter()
    shape_classes = {"train": set(), "test": set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_backgrounds, puzzle_shapes = set(), set()
        orientations = {}
        train_colors = set()
        first_two_palettes = []
        for regime in ("train", "test"):
            for index, pair in enumerate(puzzle[regime]):
                inp = pair["input"]
                h, w = inp.shape
                shapes.add((h, w))
                bg, decoded = _decode(inp)
                colors = {color for color, _ in decoded}
                backgrounds.add(bg)
                census = _census_from_pair(pair)
                censuses.add(census)
                crossings = _crossing_counts([s for _, s in decoded])
                crossing = max(crossings, default=0)
                crossing_classes.add(crossing)
                double_train += regime == "train" and crossing >= 2
                zero_test += regime == "test" and crossing == 0
                single_test += regime == "test" and crossing == 1
                coverage["triple_cross"] += crossing >= 3
                coverage["3v2h"] += census == (3, 2)
                coverage["2v3h"] += census == (2, 3)
                coverage["single_axis"] += 0 in census
                coverage["zero_marker"] += bg != 0 and 0 in colors
                coverage["mixed_crossings"] += (0 in crossings and 1 in crossings
                                                and any(x >= 2 for x in crossings))
                coverage["nonzero_double"] += bg != 0 and crossing >= 2
                shape_classes[regime].add("square" if h == w else
                                          "tall" if h > w else "wide")
                puzzle_backgrounds.add(bg)
                puzzle_shapes.add((h, w))
                if regime == "train":
                    train_colors.update(colors)
                    if index < 2:
                        first_two_palettes.append(colors)
                else:
                    coverage["new_test_color"] += bool(colors - train_colors)
                for color, (r0, c0, r1, c1) in decoded:
                    orientations.setdefault(color, set()).add(
                        "V" if c0 == c1 else "H"
                    )
                    length = abs(r1 - r0) + abs(c1 - c0)
                    short += length in (2, 3)
                    long += length >= 10
                    edge += (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1)
        coverage["background_varies"] += len(puzzle_backgrounds) > 1
        coverage["shape_varies"] += len(puzzle_shapes) > 1
        coverage["orientation_reversal"] += any(
            len(values) > 1 for values in orientations.values()
        )
        coverage["palette_overlap"] += bool(
            first_two_palettes[0] & first_two_palettes[1]
        )
        coverage["palette_disjoint"] += not (
            first_two_palettes[0] & first_two_palettes[1]
        )
    assert len(shapes) >= 80 and len(censuses) >= 12
    assert len(backgrounds) >= 8 and {0, 1, 2} <= crossing_classes
    assert min(short, long, edge, double_train, zero_test, single_test) >= 8
    assert shape_classes == {"train": {"tall", "wide", "square"},
                             "test": {"tall", "wide", "square"}}
    for name in ("3v2h", "2v3h", "single_axis", "zero_marker",
                 "mixed_crossings", "nonzero_double", "triple_cross",
                 "new_test_color", "palette_overlap", "palette_disjoint"):
        assert coverage[name] >= 3, (name, coverage[name])
    assert coverage["background_varies"] >= 20
    assert coverage["shape_varies"] == 42
    assert coverage["orientation_reversal"] >= 30

    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (None, 1),
                          (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(8765, ntrain, ntest))
    for ntrain, ntest in ((1, None), (11, None), (None, 0),
                          (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
