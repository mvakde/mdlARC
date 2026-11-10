from __future__ import annotations

import numpy as np


# Exact source inputs are excluded by content, without excluding their sizes or colors.
_SOURCE_ROWS = (
    ("0000090000", "0000059000", "0000055900", "0000055590", "0000055559", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("7000000000", "0700000000", "0070000000", "0007550000", "0000750000", "0000070000", "0000007000", "0000000700", "0000000070", "0000000007"),
    ("0000000000", "2000000000", "0200000000", "0025500000", "0002500000", "0005200000", "0005520000", "0005552000", "0000000200", "0000000020"),
    ("0100000000", "0015500000", "0001500000", "0000100000", "0000510000", "0000551000", "0000555100", "0000555510", "0000555551", "0000000000"),
)


def _key(grid):
    return (grid.shape, np.asarray(grid, dtype=np.uint8).tobytes())


_SOURCE_KEYS = {
    _key(np.array([[int(c) for c in row] for row in rows], dtype=np.int8))
    for rows in _SOURCE_ROWS
}


def _count(value, low, high, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _counts(rng, num_train, num_test):
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
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


def _line(h, w, index):
    return tuple((r, r + index) for r in range(h) if 0 <= r + index < w)


def _triangle(h, w, d0, side, size, start):
    cells = []
    for i in range(size):
        row = start + i
        col = row + d0
        if not (0 <= row < h and 0 <= col < w):
            return None
        if side == 1:
            segment = range(col + 1, col + size - i + 1)
        elif side == -1:
            segment = range(col - i - 1, col)
        else:
            return None
        for c in segment:
            if not 0 <= c < w:
                return None
            cells.append((row, c))
    return tuple(cells)


def _starts(h, w, d0, side, size):
    return [r for r in range(h - size + 1) if _triangle(h, w, d0, side, size, r) is not None]


def _turn(grid, quarter_turns, mirror):
    result = np.rot90(grid, quarter_turns)
    return np.fliplr(result) if mirror else result


def _unturn(grid, quarter_turns, mirror):
    result = np.fliplr(grid) if mirror else grid
    return np.rot90(result, -quarter_turns)


def _output(shape, background, signal, d0, specs):
    h, w = shape
    out = np.full(shape, background, dtype=np.int8)
    indices = {d0} | {d0 + side * (size + 2) for side, size, _start, _color in specs}
    for index in indices:
        for r, c in _line(h, w, index):
            out[r, c] = signal
    return out


def _components(grid, colors):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, colors))}
    parts = []
    while remaining:
        first = min(remaining)
        color = int(grid[first])
        component = {first}
        frontier = [first]
        remaining.remove(first)
        while frontier:
            r, c = frontier.pop()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining and int(grid[nxt]) == color:
                    remaining.remove(nxt)
                    component.add(nxt)
                    frontier.append(nxt)
        parts.append((color, component))
    return parts


def _parse_canonical(grid, background, signal):
    h, w = grid.shape
    seed = {(int(r), int(c)) for r, c in np.argwhere(grid == signal)}
    if len(seed) < 5 or len({c - r for r, c in seed}) != 1:
        return None
    d0 = next(iter(seed))[1] - next(iter(seed))[0]
    if seed != set(_line(h, w, d0)):
        return None
    marker_colors = [int(c) for c in np.unique(grid) if c not in (background, signal)]
    if not 1 <= len(marker_colors) <= 2:
        return None
    components = _components(grid, marker_colors)
    if not 1 <= len(components) <= 2:
        return None
    specs = []
    for color, cells in components:
        size = next((n for n in range(1, 10) if n * (n + 1) // 2 == len(cells)), None)
        if size is None:
            return None
        bands = {c - r - d0 for r, c in cells}
        if all(x > 0 for x in bands):
            side = 1
        elif all(x < 0 for x in bands):
            side = -1
        else:
            return None
        start = min(r for r, _c in cells)
        if set(_triangle(h, w, d0, side, size, start) or ()) != cells:
            return None
        if len(_line(h, w, d0 + side * (size + 2))) < 2:
            return None
        specs.append((side, size, start, color))
    if len(_line(h, w, d0)) < max(size + 1 for _side, size, _start, _color in specs):
        return None
    if len(specs) == 2 and specs[0][0] == specs[1][0]:
        a, b = sorted(specs, key=lambda spec: spec[2])
        if a[1] == b[1] or a[2] + a[1] + 1 > b[2]:
            return None
    return d0, tuple(sorted(specs))


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    colors, counts = np.unique(input_grid, return_counts=True)
    assert 3 <= len(colors) <= 4
    largest = int(np.argmax(counts))
    assert counts[largest] > max(count for i, count in enumerate(counts) if i != largest)
    background = int(colors[largest])
    candidates = []
    for signal in colors:
        signal = int(signal)
        if signal == background:
            continue
        for turn in range(4):
            for mirror in (False, True):
                canonical = _turn(input_grid, turn, mirror)
                parsed = _parse_canonical(canonical, background, signal)
                if parsed is None:
                    continue
                d0, specs = parsed
                predicted = _unturn(_output(canonical.shape, background, signal, d0, specs), turn, mirror)
                candidates.append((predicted, signal, turn, mirror, d0, specs, canonical.shape))
    assert candidates
    assert all(np.array_equal(item[0], candidates[0][0]) for item in candidates)
    predicted, signal, turn, mirror, d0, specs, canonical_shape = candidates[0]
    assert np.array_equal(output_grid, predicted)
    assert not np.array_equal(input_grid, output_grid)
    assert set(int(c) for c in np.unique(output_grid)) == {background, signal}
    physical_seed = {(int(r), int(c)) for r, c in np.argwhere(input_grid == signal)}
    slope = 1 if len({c - r for r, c in physical_seed}) == 1 else -1
    assert len({c - slope * r for r, c in physical_seed}) == 1
    return {
        "shape": input_grid.shape,
        "background": background,
        "signal": signal,
        "marker_colors": tuple(sorted(set(color for _side, _size, _start, color in specs))),
        "sizes": tuple(sorted(size for _side, size, _start, _color in specs)),
        "arrangement": "single" if len(specs) == 1 else ("opposite" if specs[0][0] != specs[1][0] else "same"),
        "slope": slope,
        "seed_length": len(physical_seed),
        "line_lengths": tuple(len(_line(canonical_shape[0], canonical_shape[1], d0 + side * (size + 2))) for side, size, _start, _color in specs),
        "layout": (input_grid.shape, slope, d0, tuple((side, size, start) for side, size, start, _color in specs)),
    }


def _shape(rng, regime):
    p = float(rng.random())
    if p < (0.23 if regime == "source_train" else 0.14):
        return int(rng.integers(7, 12)), int(rng.integers(7, 12))
    if p < 0.60:
        return int(rng.integers(11, 21)), int(rng.integers(11, 21))
    if p < 0.79:
        return int(rng.integers(20, 31)), int(rng.integers(20, 31))
    short = int(rng.integers(7, 15))
    long = int(rng.integers(20, 31))
    return (short, long) if int(rng.integers(2)) == 0 else (long, short)


def _feasible(h, w, specs):
    choices = []
    for d0 in range(1 - h, w):
        if len(_line(h, w, d0)) < max(5, max(size + 1 for _side, size in specs)):
            continue
        starts = []
        for side, size in specs:
            if len(_line(h, w, d0 + side * (size + 2))) < 2:
                break
            available = _starts(h, w, d0, side, size)
            if not available:
                break
            starts.append(available)
        if len(starts) != len(specs):
            continue
        if len(specs) == 2 and specs[0][0] == specs[1][0]:
            combinations = [(a, b) for a in starts[0] for b in starts[1]
                            if a + specs[0][1] + 1 <= b or b + specs[1][1] + 1 <= a]
            if not combinations:
                continue
        else:
            combinations = None
        choices.append((d0, starts, combinations))
    return choices


def _construct(rng, specs, colors, orientation, regime):
    background, signal, marker_colors = colors
    for attempt in range(11):
        h, w = _shape(rng, regime) if attempt < 10 else (30, 30)
        candidates = _feasible(h, w, specs)
        if candidates:
            break
    else:
        raise AssertionError("no feasible canvas")
    d0, starts, combinations = candidates[int(rng.integers(len(candidates)))]
    if combinations is None:
        chosen = [int(options[int(rng.integers(len(options)))]) for options in starts]
    else:
        chosen = combinations[int(rng.integers(len(combinations)))]
    input_grid = np.full((h, w), background, dtype=np.int8)
    for row, col in _line(h, w, d0):
        input_grid[row, col] = signal
    detailed = []
    for (side, size), start, color in zip(specs, chosen, marker_colors):
        cells = _triangle(h, w, d0, side, size, start)
        assert cells is not None
        for row, col in cells:
            assert input_grid[row, col] == background
            input_grid[row, col] = color
        detailed.append((side, size, start, color))
    output_grid = _output((h, w), background, signal, d0, detailed)
    turn, mirror = orientation
    return np.array(_turn(input_grid, turn, mirror), copy=True), np.array(_turn(output_grid, turn, mirror), copy=True)


def _palette(rng, prior=None, repeat_signal=None, multi=False):
    pool = list(range(10))
    old_background, old_signal, old_marker = prior if prior is not None else (None, None, None)
    backgrounds = [c for c in pool if c != old_background]
    background = int(rng.choice(backgrounds))
    signals = [c for c in pool if c != background and c != old_signal]
    if repeat_signal is not None and repeat_signal != background and rng.random() < 0.55:
        signal = int(repeat_signal)
    else:
        signal = int(rng.choice(signals))
    markers = [c for c in pool if c not in (background, signal, old_marker)]
    marker = int(rng.choice(markers))
    if multi:
        second = int(rng.choice([c for c in pool if c not in (background, signal, marker)]))
        return background, signal, (marker, second)
    return background, signal, (marker,)


def _orientations(rng, count, mixed):
    options = {slope: [(turn, mirror) for turn in range(4) for mirror in (False, True)
                       if (1 if (turn % 2) == int(mirror) else -1) == slope]
               for slope in (-1, 1)}
    if mixed:
        slopes = [-1, 1] + [int(rng.choice([-1, 1])) for _ in range(count - 2)]
        rng.shuffle(slopes)
    else:
        slopes = [int(rng.choice([-1, 1]))] * count
    return [options[slope][int(rng.integers(4))] for slope in slopes], slopes


def _specs(rng, num_train, num_test):
    core = [int(x) for x in rng.choice(np.array([2, 3, 4, 5, 6]), size=3, replace=False)]
    if rng.random() < 0.20:
        core[2] = 1
    first_side = int(rng.choice([-1, 1]))
    train = [
        [(first_side, core[0]), (-first_side, core[1])],
        [(int(rng.choice([-1, 1])), core[2])],
    ]
    same_taught = num_train >= 4 and rng.random() < 0.38
    if same_taught:
        sizes = [int(x) for x in rng.choice(np.array([2, 3, 4, 5, 6]), size=2, replace=False)]
        side = int(rng.choice([-1, 1]))
        train.append([(side, sizes[0]), (side, sizes[1])])
    while len(train) < num_train:
        count = 1 if rng.random() < 0.52 else 2
        possible = [n for n in range(2, 8)] + ([1] if any(size == 1 for group in train for _side, size in group) else [])
        sizes = [int(x) for x in rng.choice(np.array(possible), size=count, replace=False)]
        side = int(rng.choice([-1, 1]))
        train.append([(side, sizes[0])] if count == 1 else [(side, sizes[0]), (-side, sizes[1])])
    rng.shuffle(train)
    taught_sizes = {size for group in train for _side, size in group}
    test = []
    for _ in range(num_test):
        arrangement = "single" if rng.random() < 0.43 else "opposite"
        if same_taught and rng.random() < 0.15:
            arrangement = "same"
        count = 1 if arrangement == "single" else 2
        sizes = []
        for _marker in range(count):
            if rng.random() < 0.51:
                choices = sorted(taught_sizes - set(sizes))
            else:
                choices = [n for n in range(2, 9) if n not in sizes]
                if 1 in taught_sizes:
                    choices.append(1)
            sizes.append(int(rng.choice(choices)))
        side = int(rng.choice([-1, 1]))
        test.append([(side, sizes[0])] if count == 1 else
                    [(side, sizes[0]), (side if arrangement == "same" else -side, sizes[1])])
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    train_specs, test_specs = _specs(rng, num_train, num_test)
    mixed_orientations = rng.random() < 0.63
    train_orientations, train_slopes = _orientations(rng, num_train, mixed_orientations)
    orientation_options = [(turn, mirror) for turn in range(4) for mirror in (False, True)
                           if (1 if (turn % 2) == int(mirror) else -1) in set(train_slopes)]
    test_orientations = [orientation_options[int(rng.integers(len(orientation_options)))] for _ in range(num_test)]
    teach_multicolor = rng.random() < 0.58
    opposite_index = next(i for i, group in enumerate(train_specs)
                          if len(group) == 2 and group[0][0] != group[1][0])
    palettes = []
    prior = None
    for i, group in enumerate(train_specs):
        multi = len(group) == 2 and (i == opposite_index and teach_multicolor or teach_multicolor and rng.random() < 0.35)
        palette = _palette(rng, prior if i == 1 else None, multi=multi)
        palettes.append(palette)
        if i == 0:
            prior = (palette[0], palette[1], palette[2][0])
    train_signals = [palette[1] for palette in palettes]
    for group in test_specs:
        multi = len(group) == 2 and teach_multicolor and rng.random() < 0.53
        palette = _palette(rng, repeat_signal=int(rng.choice(train_signals)), multi=multi)
        palettes.append(palette)
    all_specs = train_specs + test_specs
    all_orientations = train_orientations + test_orientations
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=len(all_specs), dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for i, (group, palette, orientation, pair_seed) in enumerate(zip(all_specs, palettes, all_orientations, pair_seeds)):
        pair_rng = np.random.default_rng(pair_seed)
        colors = (palette[0], palette[1], tuple(palette[2][j % len(palette[2])] for j in range(len(group))))
        regime = "source_train" if i < num_train else "target_test"
        for _retry in range(20):
            inp, out = _construct(pair_rng, group, colors, orientation, regime)
            if _key(inp) not in seen and _key(inp) not in _SOURCE_KEYS:
                break
        else:
            raise AssertionError("could not make a distinct pair")
        seen.add(_key(inp))
        puzzle["train" if i < num_train else "test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            input_key, pair_key = _key(inp), (_key(inp), _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(inp, out))
    train, test = analyses["train"], analyses["test"]
    assert any(a["arrangement"] == "single" for a in train)
    assert any(a["arrangement"] == "opposite" for a in train)
    train_sizes = {size for a in train for size in a["sizes"]}
    assert len(train_sizes) >= 3
    train_slopes = {a["slope"] for a in train}
    for a in test:
        assert a["slope"] in train_slopes
        if a["arrangement"] == "same":
            assert any(t["arrangement"] == "same" for t in train)
        if len(a["marker_colors"]) == 2:
            assert any(len(t["marker_colors"]) == 2 for t in train)
        if 1 in a["sizes"]:
            assert 1 in train_sizes
        if max(a["sizes"]) > max(train_sizes):
            assert len(train_sizes) >= 3


def _equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(pa[name], pb[name])
        for pa, pb in zip(a[split], b[split]) for name in ("input", "output"))
        for split in ("train", "test"))


def _direct_rule_check(inp, out, analysis):
    """Check the line rule in physical coordinates, without canonical rendering."""
    background, signal, slope = analysis["background"], analysis["signal"], analysis["slope"]
    seed = {(int(r), int(c)) for r, c in np.argwhere(inp == signal)}
    seed_index = next(iter(seed))[1] - slope * next(iter(seed))[0]
    colors = [int(c) for c in np.unique(inp) if c not in (background, signal)]
    components = _components(inp, colors)
    expected_indices = {seed_index}
    for _color, cells in components:
        size = next(n for n in range(1, 10) if n * (n + 1) // 2 == len(cells))
        distances = {c - slope * r - seed_index for r, c in cells}
        assert all(distance > 0 for distance in distances) or all(distance < 0 for distance in distances)
        side = 1 if min(distances) > 0 else -1
        expected_indices.add(seed_index + side * (size + 2))
    expected_signal = {(r, slope * r + index)
                       for index in expected_indices for r in range(inp.shape[0])
                       if 0 <= slope * r + index < inp.shape[1]}
    actual_signal = {(int(r), int(c)) for r, c in np.argwhere(out == signal)}
    assert actual_signal == expected_signal
    assert np.all((out == background) | (out == signal))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert _equal(generate_puzzle(9137, 7, 3), generate_puzzle(9137, 7, 3))
    assert _equal(generate_puzzle(2718), generate_puzzle(2718))
    assert len({tuple(_key(p["input"]) for split in ("train", "test") for p in puzzle[split]) for puzzle in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_analyses = []
    varied_slopes = 0
    varied_backgrounds = 0
    varied_markers = 0
    recurring_signals = 0
    multi_train = multi_test = same_test = 0
    single_test = opposite_test = 0
    novel_test = familiar_test = 0
    small = large = elongated = 0
    combined = 0
    for puzzle in puzzles:
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        for split, analyses in (("train", train), ("test", test)):
            for pair, analysis in zip(puzzle[split], analyses):
                _direct_rule_check(pair["input"], pair["output"], analysis)
        all_analyses.extend(train + test)
        varied_slopes += len({a["slope"] for a in train + test}) > 1
        varied_backgrounds += len({a["background"] for a in train + test}) > 1
        varied_markers += len({a["marker_colors"] for a in train + test}) > 1
        recurring_signals += bool({a["signal"] for a in train} & {a["signal"] for a in test})
        multi_train += any(len(a["marker_colors"]) == 2 for a in train)
        multi_test += any(len(a["marker_colors"]) == 2 for a in test)
        same_test += any(a["arrangement"] == "same" for a in test)
        single_test += any(a["arrangement"] == "single" for a in test)
        opposite_test += any(a["arrangement"] == "opposite" for a in test)
        train_sizes = {size for a in train for size in a["sizes"]}
        novel_test += any(any(size not in train_sizes for size in a["sizes"]) for a in test)
        familiar_test += any(all(size in train_sizes for size in a["sizes"]) for a in test)
        for a in train + test:
            h, w = a["shape"]
            small += min(h, w) <= 10
            large += max(h, w) >= 24
            elongated += max(h, w) / min(h, w) >= 2
            combined += (a["slope"] == -1 and len(a["marker_colors"]) == 2
                         and a["arrangement"] == "opposite")
    shapes = {a["shape"] for a in all_analyses}
    layouts = {a["layout"] for a in all_analyses}
    sizes = {s for a in all_analyses for s in a["sizes"]}
    assert len(shapes) >= 65 and len(layouts) >= 180
    assert {1, 2, 3, 4, 5, 6, 7, 8} <= sizes
    assert {a["slope"] for a in all_analyses} == {-1, 1}
    assert {a["arrangement"] for a in all_analyses} == {"single", "opposite", "same"}
    assert min(min(a["shape"]) for a in all_analyses) <= 9
    assert max(max(a["shape"]) for a in all_analyses) >= 28
    assert any(max(a["shape"]) / min(a["shape"]) >= 2 for a in all_analyses)
    assert varied_slopes >= 12 and varied_backgrounds == 42 and varied_markers == 42
    assert recurring_signals >= 8
    assert sum(len(a["marker_colors"]) == 2 for a in all_analyses) >= 10
    assert multi_train >= 8 and multi_test >= 3 and same_test >= 1
    assert single_test >= 10 and opposite_test >= 10
    assert novel_test >= 5 and familiar_test >= 10
    assert small >= 20 and large >= 20 and elongated >= 10 and combined >= 5
    for offset, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + offset, ntrain, ntest)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
    for offset, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + offset, ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
