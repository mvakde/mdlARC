"""Compose sparse vertical streams into ordered, bottom-aligned pyramids."""

import hashlib
from collections import Counter
import numpy as np


_SOURCE_DIGESTS = {
    "cbb5847bf030f5756f352c165941baba52b996eb7a009f67060395e89f6a6614",
    "1b631a9f740e2be6b597af080bbe99a20656198f10de504383b358791ffde9b5",
    "3240d07016d919c9a7dcb1ff578e24e7bb487100dd8071c41c27c355c66f9d89",
    "8ff3e33dd5ef0ff5315798f1777494873e78ebff7976c024df28220155e2c5b3",
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _pair_digest(pair):
    a = np.asarray(pair["input"], dtype=np.int16)
    b = np.asarray(pair["output"], dtype=np.int16)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes() + b.tobytes()).hexdigest()


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int16)
    return a.shape, a.tobytes()


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(counts[index] > grid.size // 2, "background is not unambiguous")
    return int(colors[index])


def _paint(input_grid, background):
    height, width = input_grid.shape
    output = np.full_like(input_grid, background)
    filled = np.zeros(input_grid.shape, dtype=bool)
    info = []
    for col in np.flatnonzero(np.any(input_grid != background, axis=0)):
        column = int(col)
        rows = np.flatnonzero(input_grid[:, column] != background)
        sequence = input_grid[rows, column].copy()
        n = len(sequence)
        k = int(np.sqrt(n))
        _require(k >= 1 and k * k == n, "stream count is not a square")
        _require(k <= height and column >= k - 1 and column + k <= width,
                 "pyramid is clipped")
        offset = 0
        for level in range(k):
            row = height - k + level
            left, right = column - level, column + level + 1
            _require(not np.any(filled[row, left:right]), "pyramids overlap")
            count = 2 * level + 1
            output[row, left:right] = sequence[offset:offset + count]
            filled[row, left:right] = True
            offset += count
        info.append({"column": column, "rows": rows, "sequence": sequence,
                     "k": k, "colors": frozenset(int(c) for c in sequence)})
    _require(bool(info), "no streams")
    return output, info


def _dimension(rng, minimum, maximum=30):
    if minimum > maximum:
        raise ValueError("configuration does not fit an ARC grid")
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return maximum
    if mode < 5:
        return int(rng.integers(minimum, min(maximum, minimum + 6) + 1))
    return int(rng.integers(minimum, maximum + 1))


def _sample_sizes(rng, count, regime):
    weights = np.array([0.07, 0.34, 0.36, 0.15, 0.08] if regime == "source_train"
                       else [0.06, 0.25, 0.30, 0.23, 0.16])
    for _ in range(40):
        sizes = [int(rng.choice(np.arange(1, 6), p=weights)) for _ in range(count)]
        if sum(2 * k - 1 for k in sizes) <= 30:
            return sizes
    return [2] * count


def _layout(rng, sizes, width, center_first=False):
    bases = [2 * k - 1 for k in sizes]
    slack = width - sum(bases)
    _require(slack >= 0, "bases do not fit")
    count = len(sizes)
    if center_first and count == 1:
        gaps = [slack // 2, slack - slack // 2]
    else:
        mode = int(rng.integers(0, 4))
        active = list(range(count + 1))
        if mode == 0 and count > 1:
            active = list(range(1, count))  # both sides touch
        elif mode == 1:
            active.remove(0 if rng.integers(0, 2) else count)
        elif mode == 2 and slack >= 2:
            gaps = [0] * (count + 1)
            gaps[0] = gaps[-1] = 1
            active = list(range(count + 1))
            slack -= 2
        if mode != 2 or width - sum(bases) < 2:
            gaps = [0] * (count + 1)
        if slack:
            weights = rng.dirichlet(np.ones(len(active)))
            assigned = rng.multinomial(slack, weights)
            for index, value in zip(active, assigned):
                gaps[index] += int(value)
    columns = []
    left = gaps[0]
    for index, k in enumerate(sizes):
        columns.append(left + k - 1)
        left += bases[index] + gaps[index + 1]
    return columns


def _sample_rows(rng, height, n):
    mode = int(rng.integers(0, 4))
    bottom = bool(rng.integers(0, 3) == 0)
    if n == height:
        return np.arange(height)
    if mode == 0:  # contiguous run
        start = height - n if bottom else int(rng.integers(0, height - n + 1))
        return np.arange(start, start + n)
    if mode == 1:  # regular spacing
        step = int(rng.integers(1, (height - 1) // (n - 1) + 1)) if n > 1 else 1
        span = (n - 1) * step
        start = height - span - 1 if bottom else int(rng.integers(0, height - span))
        return start + np.arange(n) * step
    if mode == 2:  # clustered runs separated by a few larger gaps
        extra = height - n
        gaps = np.zeros(n + 1, dtype=int)
        positions = rng.choice(np.arange(n + 1), size=min(n + 1, 4), replace=False)
        pieces = rng.multinomial(extra, rng.dirichlet(np.ones(len(positions))))
        gaps[positions] = pieces
        if bottom:
            trailing = int(gaps[-1])
            gaps[-1] = 0
            gaps[0] += trailing
        return gaps[0] + np.arange(n) + np.cumsum(gaps[1:n + 1]) - gaps[1:n + 1]
    if bottom:
        return np.sort(np.r_[rng.choice(np.arange(height - 1), n - 1, replace=False), height - 1])
    return np.sort(rng.choice(np.arange(height), n, replace=False))


def _palettes(rng, sizes, background):
    available = [c for c in range(10) if c != background]
    rng.shuffle(available)
    count = len(sizes)
    mode = int(rng.integers(0, 4))
    if mode == 0:  # identical available colors
        shared = available[:int(rng.integers(1, 4))]
        return [shared[:] for _ in sizes]
    if mode == 1:  # disjoint colors
        palettes, offset = [], 0
        for index in range(count):
            remaining = len(available) - offset - (count - index - 1)
            take = int(rng.integers(1, min(3, remaining) + 1))
            palettes.append(available[offset:offset + take])
            offset += take
        return palettes
    if mode == 2 and count > 1:  # common color and private colors
        return [[available[0], available[index + 1]] for index in range(count)]
    return [list(rng.choice(available, size=int(rng.integers(1, 5)), replace=False))
            for _ in sizes]


def _sequence(rng, k, palette, mode=None):
    n = k * k
    if mode is None:
        mode = int(rng.integers(0, 6))
    if len(palette) == 1 or mode == 0 or n == 1:
        return np.full(n, int(rng.choice(palette)), dtype=np.int8)
    colors = list(palette)
    rng.shuffle(colors)
    if mode == 1:  # uniformly colored pyramid levels
        levels = [colors[i % len(colors)] for i in range(k)]
        return np.array([c for level, c in enumerate(levels) for _ in range(2 * level + 1)],
                        dtype=np.int8)
    if mode == 2:  # unequal bands, possibly returning to an earlier color
        cuts = sorted(int(c) for c in rng.choice(np.arange(1, n),
                      size=min(len(colors), n - 1), replace=False))
        values = [colors[i % len(colors)] for i in range(len(cuts) + 1)]
        if len(values) > 2 and rng.integers(0, 2):
            values[-1] = values[0]
        return np.array([values[np.searchsorted(cuts, i, side="right")]
                         for i in range(n)], dtype=np.int8)
    if mode == 3:  # alternating
        return np.array([colors[i % len(colors)] for i in range(n)], dtype=np.int8)
    if mode == 4:  # explicit A-B-A return
        a, b = colors[:2]
        return np.array([a if i < n // 3 or i >= 2 * n // 3 else b
                         for i in range(n)], dtype=np.int8)
    return np.array(rng.choice(colors, size=n), dtype=np.int8)


def _diagnostic(rng, background):
    colors = list(rng.choice([c for c in range(10) if c != background], 4,
                             replace=False))
    a, b, c, d = colors
    return np.array([a, b, b, c, c, a, d, d, d], dtype=np.int8)


def _make_pair(rng, regime, index, train_count):
    if regime == "source_train" and index == 0:
        sizes = [2]
    elif regime == "source_train" and index == 1:
        sizes = [3] if train_count == 2 and rng.integers(0, 2) else [3, 2]
    elif regime == "source_train" and index == 2:
        sizes = [2, 3]
    else:
        count = int(rng.choice([1, 2, 3, 4], p=[0.32, 0.39, 0.22, 0.07]
                               if regime == "source_train" else [0.23, 0.40, 0.27, 0.10]))
        sizes = _sample_sizes(rng, count, regime)
    if len(sizes) > 1 and rng.integers(0, 2):
        rng.shuffle(sizes)
    height = _dimension(rng, max(4, max(k * k for k in sizes)))
    width = _dimension(rng, sum(2 * k - 1 for k in sizes))
    columns = _layout(rng, sizes, width, regime == "source_train" and index == 0)
    background = int(rng.choice(np.arange(10), p=[0.37] + [0.07] * 9))
    palettes = _palettes(rng, sizes, background)
    input_grid = np.full((height, width), background, dtype=np.int8)
    for stream_index, (column, k, palette) in enumerate(zip(columns, sizes, palettes)):
        if regime == "source_train" and index == 0:
            sequence = np.full(4, int(rng.choice(palette)), dtype=np.int8)
        elif regime == "source_train" and index == 1 and k == 3:
            sequence = _diagnostic(rng, background)
        else:
            sequence = _sequence(rng, k, palette)
        rows = _sample_rows(rng, height, k * k)
        input_grid[rows, column] = sequence
    output_grid, _ = _paint(input_grid, background)
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count in (("source_train", train_count), ("target_test", test_count)):
        destination = puzzle["train" if regime == "source_train" else "test"]
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(80):
                pair = _make_pair(pair_rng, regime, index, train_count)
                key = _grid_key(pair["input"])
                if (key not in seen and _pair_digest(pair) not in _SOURCE_DIGESTS
                        and not np.array_equal(pair["input"], pair["output"])):
                    seen.add(key)
                    destination.append(pair)
                    break
            else:
                raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             f"{label} must be a 2D NumPy array")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} has noninteger dtype")
    _require(all(1 <= d <= 30 for d in grid.shape), f"{label} has invalid dimensions")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has invalid colors")


def _mixed_depths(sequence, k):
    return sum(len(set(int(c) for c in sequence[level * level:(level + 1) ** 2])) > 1
               for level in range(1, k))


def _has_color_return(sequence):
    values = [int(c) for c in sequence]
    for color in set(values):
        first, last = values.index(color), len(values) - 1 - values[::-1].index(color)
        if any(value != color for value in values[first:last]):
            return True
    return False


def _distinguishes_lateral_scans(sequence, k):
    """Check the visible rows against simple alternative particle scans."""
    differs = [False, False, False, False]
    for level in range(1, k):
        row = list(sequence[level * level:(level + 1) ** 2])
        width = len(row)
        orders = [list(reversed(range(width))),
                  list(reversed(range(width))) if level % 2 else list(range(width)),
                  [j for i in range((width + 1) // 2)
                   for j in (i, width - 1 - i) if i != width - 1 - i] + [width // 2],
                  [j for i in range((width + 1) // 2)
                   for j in (width - 1 - i, i) if i != width - 1 - i] + [width // 2]]
        for index, order in enumerate(orders):
            candidate = [None] * width
            for color, position in zip(row, order):
                candidate[position] = int(color)
            differs[index] |= candidate != row
    return differs


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "invalid pair counts")
    seen = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain only input and output")
            for role in ("input", "output"):
                _validate_grid(pair[role], role)
            _require(pair["input"].shape == pair["output"].shape,
                     "dimensions changed")
            key = _grid_key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_pair_digest(pair) not in _SOURCE_DIGESTS, "source pair copied")
            background = _background(pair["input"])
            expected, streams = _paint(pair["input"], background)
            _require(np.array_equal(expected, pair["output"]), "incorrect transformation")
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "unchanged pair")
            _require(int(np.count_nonzero(pair["input"] != background)) ==
                     int(np.count_nonzero(pair["output"] != background)),
                     "particle count changed")
            _require(_background(pair["output"]) == background,
                     "background role changed")
            if split == "train":
                training.append(streams)
    _require(any(any(s["k"] == 2 for s in streams) for streams in training),
             "training lacks a two-level example")
    _require(any(any(s["k"] == 3 and _mixed_depths(s["sequence"], 3) >= 2
                     for s in streams) for streams in training),
             "training does not teach lateral order at two depths")
    contradicted = [False] * 4
    for streams in training:
        for stream in streams:
            for index, different in enumerate(_distinguishes_lateral_scans(
                    stream["sequence"], stream["k"])):
                contradicted[index] |= different
    _require(all(contradicted), "training is compatible with a competing lateral scan")
    _require(any(any(len(s["colors"]) == 1 for s in streams) for streams in training),
             "training lacks a monochrome example")
    _require(any(any(_has_color_return(s["sequence"]) for s in streams)
                 for streams in training), "training does not distinguish color grouping")


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[role], y[role]) for x, y in zip(a[split], b[split])
        for role in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, backgrounds = set(), set(), set(), set(), set()
    sizes, stream_counts, spacing, palette_relations = set(), set(), set(), set()
    features = Counter()
    boundary_contacts, base_gaps = set(), set()
    bottom_hits = 0
    return_sequences = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local_backgrounds, local_sizes, local_counts, large_sides = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = _background(grid)
                expected, streams = _paint(grid, bg)
                _require(np.array_equal(expected, pair["output"]), "reconstruction failed")
                backgrounds.add(bg)
                local_backgrounds.add(bg)
                local_counts.add(len(streams))
                shapes.add(grid.shape)
                stream_counts.add((split, len(streams)))
                if split == "train" and grid.shape[1] % 2 == 0:
                    features["even_train"] += 1
                if split == "test" and grid.shape[1] > 13:
                    features["wide_test"] += 1
                if 30 in grid.shape:
                    features["dimension_30"] += 1
                if grid.size <= 100:
                    features["compact"] += 1
                if bg != 0 and np.any(grid == 0):
                    features["zero_particle"] += 1
                layouts.add((grid.shape, tuple((s["column"], s["k"], tuple(s["rows"]))
                                               for s in streams)))
                ks = [s["k"] for s in streams]
                local_sizes.update(ks)
                if 1 in ks and max(ks) >= 2:
                    features["one_alongside"] += 1
                left = streams[0]["column"] - ks[0] + 1 == 0
                right = streams[-1]["column"] + ks[-1] - 1 == grid.shape[1] - 1
                boundary_contacts.add(int(left) + int(right))
                if len(streams) >= 2:
                    features["equal_sizes" if len(set(ks)) == 1 else "unequal_sizes"] += 1
                    if len(set(ks)) > 1:
                        large_sides.add(ks[0] > ks[-1])
                    for first, second in zip(streams, streams[1:]):
                        base_gaps.add(second["column"] - second["k"] -
                                      first["column"] - first["k"] + 1)
                    if sum(len(s["colors"]) > 1 for s in streams) >= 2:
                        features["several_multicolor"] += 1
                for s in streams:
                    sizes.add(s["k"])
                    rows = s["rows"]
                    spacing.add("contiguous" if np.all(np.diff(rows) == 1) else
                                "regular" if len(set(np.diff(rows))) == 1 else "irregular")
                    bottom_hits += int(rows[-1] == grid.shape[0] - 1)
                    return_sequences += int(_has_color_return(s["sequence"]))
                    features["lower_footprint"] += int(any(
                        pair["output"][row, s["column"]] != bg for row in rows))
                    if len(s["colors"]) > 1:
                        sequence = s["sequence"]
                        cuts = [i for i in range(1, len(sequence))
                                if sequence[i] != sequence[i - 1]]
                        features["whole_level_bands"] += int(bool(cuts) and all(
                            int(np.sqrt(i)) ** 2 == i for i in cuts))
                if len(streams) >= 2:
                    colors = [s["colors"] for s in streams]
                    overlap = colors[0] & colors[1]
                    palette_relations.add("identical" if colors[0] == colors[1] else
                                          "partial" if overlap else "disjoint")
        features["within_background"] += int(len(local_backgrounds) >= 2)
        features["within_sizes"] += int(len(local_sizes) >= 3)
        features["within_stream_counts"] += int(len(local_counts) >= 2)
        features["within_large_sides"] += int(len(large_sides) >= 2)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(len(shapes) >= 30 and len(layouts) >= 150, "layouts lack diversity")
    _require(sizes == {1, 2, 3, 4, 5}, "pyramid sizes lack diversity")
    _require({("train", 1), ("train", 2), ("test", 1),
              ("test", 2)} <= stream_counts, "stream counts lack diversity")
    _require(len(backgrounds) >= 7 and 0 in backgrounds, "backgrounds lack diversity")
    _require({"contiguous", "regular", "irregular"} <= spacing,
             "spacing lacks diversity")
    _require(bottom_hits >= 5 and return_sequences >= 5,
             "lower placement or color returns lack diversity")
    _require({"identical", "partial", "disjoint"} <= palette_relations,
             "palette relations lack diversity")
    _require(boundary_contacts == {0, 1, 2} and len(base_gaps) >= 5,
             "boundary positions or base gaps lack diversity")
    for feature in ("even_train", "wide_test", "dimension_30", "compact",
                    "zero_particle", "one_alongside", "equal_sizes",
                    "unequal_sizes", "several_multicolor", "lower_footprint",
                    "whole_level_bands"):
        _require(features[feature] >= 3, f"{feature} lacks diversity")
    for feature in ("within_background", "within_sizes", "within_stream_counts",
                    "within_large_sides"):
        _require(features[feature] >= 5, f"{feature} lacks within-puzzle diversity")
    for index, (nt, nv) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                      (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, nt, nv)
        _require(nt is None or len(puzzle["train"]) == nt, "train count ignored")
        _require(nv is None or len(puzzle["test"]) == nv, "test count ignored")
    a, b = generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3)
    _require(_puzzles_equal(a, b), "nondeterministic")
    _require(not _puzzles_equal(a, generate_puzzle(314160, 7, 3)),
             "different seeds repeated")
    for nt, nv in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(23, nt, nv)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
