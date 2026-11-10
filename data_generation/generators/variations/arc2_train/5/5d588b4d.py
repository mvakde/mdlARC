"""Generate unary-run ARC puzzles with pair-local colors and dimensions."""

import numpy as np


_SOURCE_SPECS = frozenset({
    (8, 11, 6, 0, 3), (7, 23, 4, 0, 1),
    (8, 5, 3, 0, 7), (7, 23, 3, 0, 4),
    (13, 16, 8, 0, 9), (7, 10, 7, 0, 6),
})


def _length(n):
    return n * n + 2 * n - 2


def _sequence(n, background, foreground):
    cells = []
    lengths = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
    for i, count in enumerate(lengths):
        cells.extend([foreground] * count)
        if i < len(lengths) - 1:
            cells.append(background)
    return cells


def _output(n, width, background, foreground):
    cells = _sequence(n, background, foreground)
    rows = (len(cells) + width - 1) // width
    flat = np.full(rows * width, background, dtype=np.int8)
    flat[:len(cells)] = cells
    return flat.reshape(rows, width)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _width_features(n, width):
    """Classify actual row cuts and final padding of the serialized sequence."""
    cells = _sequence(n, 0, 1)
    total = len(cells)
    pad = (-total) % width
    cuts = range(width, total, width)
    inside = any(cells[pos - 1] == cells[pos] == 1 for pos in cuts)
    separator = any(cells[pos - 1] == 0 or cells[pos] == 0 for pos in cuts)
    peak_start = sum(range(1, n)) + (n - 1)
    peak = any(peak_start < pos < peak_start + n for pos in cuts)
    return pad, inside, separator, peak


def _width(rng, n, goal, cut_goal, excluded=(), allow_full=False):
    candidates = [w for w in range(n if allow_full else n + 1, 31)
                  if (_length(n) + w - 1) // w <= 30 and w not in excluded]
    if not candidates:
        candidates = [w for w in range(n if allow_full else n + 1, 31)
                      if (_length(n) + w - 1) // w <= 30]
    if goal == "single":
        preferred = [w for w in candidates if w >= _length(n)]
    elif goal == "exact":
        preferred = [w for w in candidates if _width_features(n, w)[0] == 0]
    elif goal == "one":
        preferred = [w for w in candidates if _width_features(n, w)[0] == 1]
    elif goal == "long":
        preferred = [w for w in candidates if _width_features(n, w)[0] >= 5]
    elif goal == "wrapped":
        preferred = [w for w in candidates if w < _length(n)]
    else:
        preferred = candidates
    if preferred:
        candidates = preferred
    cut_index = {"inside": 1, "separator": 2, "peak": 3}.get(cut_goal)
    if cut_index is not None:
        preferred = [w for w in candidates if _width_features(n, w)[cut_index]]
        if preferred:
            candidates = preferred
    return int(rng.choice(np.asarray(candidates)))


def _height(rng):
    band = int(rng.integers(0, 5))
    if band == 0:
        return int(rng.integers(2, 5))
    if band == 1:
        return int(rng.integers(24, 31))
    return int(rng.integers(2, 31))


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _build_pair(n, width, height, background, foreground):
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[0, :n] = foreground
    return {"input": input_grid, "output": _output(n, width, background, foreground)}


def _choose_colors(rng, base, mixed, previous_foreground, reuse):
    background = int(rng.integers(0, 10)) if mixed else base
    if reuse and previous_foreground is not None and previous_foreground != background:
        foreground = previous_foreground
    else:
        foreground = int(rng.choice(np.asarray([c for c in range(10) if c != background])))
    return background, foreground


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)

    # The rule and semantic roles are the only shared conventions. This flag
    # chooses whether literal background colors stay stable in this puzzle.
    source_like = seed % 6 == 0
    mixed = False if source_like else bool(rng.random() < 0.72)
    base = int(rng.integers(0, 10))
    color_order = [int(x) for x in rng.permutation(
        np.asarray([c for c in range(10) if c != base]))]
    train_palette, test_palette = color_order[:5], color_order[5:]
    regime = seed % 3  # extrapolation, interpolation, or familiar length
    if source_like:
        anchors = [3, 6]
    elif regime == 1:
        anchors = [4, 6]
    else:
        anchors = [int(rng.choice(np.asarray((3, 4)))), int(rng.integers(6, 10))]
    train_ns = anchors[:]
    pool = [3, 4, 6] if source_like else [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    if regime == 1:
        pool.remove(5)
    while len(train_ns) < train_count:
        train_ns.append(int(rng.choice(np.asarray(pool))))

    # Shuffle only after selecting the two substantial demonstrations.
    order = rng.permutation(train_count)
    train_ns = [train_ns[int(i)] for i in order]
    anchor_small = int(np.where(order == 0)[0][0])
    anchor_large = int(np.where(order == 1)[0][0])
    puzzle = {"train": [], "test": []}
    used = set()
    specs = []
    previous_fg = None
    for i, n in enumerate(train_ns):
        if i == anchor_small:
            goal = "single"
        elif i == anchor_large:
            goal = "wrapped"
        else:
            goal = str(rng.choice(np.asarray(("exact", "one", "long", "wrapped", "free"))))
        cut = str(rng.choice(np.asarray(("inside", "separator", "peak", "free"))))
        full = i == anchor_large and bool(rng.random() < 0.22)
        other_anchor_widths = tuple(specs[j][1] for j in (anchor_small, anchor_large)
                                    if j < len(specs))
        width = n if full else _width(rng, n, goal, cut,
                                      excluded=other_anchor_widths if i in (anchor_small, anchor_large) else ())
        height = _height(rng)
        if source_like:
            background, foreground = base, train_palette[i % len(train_palette)]
        else:
            background, foreground = _choose_colors(rng, base, mixed, previous_fg, rng.random() < 0.24)
        previous_fg = foreground
        # A subset of larger training sets repeats one full specification
        # with a different blank-tail height.
        if train_count >= 3 and i == train_count - 1 and i not in (anchor_small, anchor_large) and rng.random() < 0.28:
            n, width, _, background, foreground = specs[anchor_small]
            height = 2 + ((specs[anchor_small][2] - 1 + int(rng.integers(1, 29))) % 29)
        for _ in range(100):
            if (height, width, n, background, foreground) not in _SOURCE_SPECS:
                pair = _build_pair(n, width, height, background, foreground)
                if _key(pair["input"]) not in used:
                    break
            height = 2 + height % 29
        else:
            raise ValueError("could not create unique training input")
        used.add(_key(pair["input"]))
        specs.append((n, width, height, background, foreground))
        puzzle["train"].append(pair)

    train_lengths = [s[0] for s in specs]
    if source_like:
        first_test_n = int(rng.choice(np.asarray((7, 8))))
    elif regime == 0:
        first_test_n = int(rng.integers(max(train_lengths) + 1, 17)) if max(train_lengths) < 16 else 16
    elif regime == 1:
        first_test_n = 5
    else:
        first_test_n = int(rng.choice(np.asarray(train_lengths)))
    test_ns = [first_test_n]
    for _ in range(test_count - 1):
        category = int(rng.integers(0, 3))
        if category == 0:
            test_ns.append(int(rng.choice(np.asarray(train_lengths))))
        elif category == 1:
            test_ns.append(int(rng.integers(5, 17)))
        else:
            test_ns.append(5 if 5 not in train_lengths else int(rng.integers(2, 17)))

    train_full = any(s[0] == s[1] for s in specs)
    for i, n in enumerate(test_ns):
        matched_widths = [s[1] for s in specs if s[0] == n]
        if i == 0 and regime == 2:
            goal = "free"
        else:
            goal = str(rng.choice(np.asarray(("exact", "one", "long", "wrapped", "free"))))
        cut = str(rng.choice(np.asarray(("inside", "separator", "peak", "free"))))
        familiar_width = bool(i > 0 and rng.random() < 0.28)
        if familiar_width:
            options = [s[1] for s in specs if s[1] >= n and (_length(n) + s[1] - 1) // s[1] <= 30]
            width = int(rng.choice(np.asarray(options))) if options else _width(rng, n, goal, cut, matched_widths, train_full)
        else:
            width = _width(rng, n, goal, cut, matched_widths, train_full)
        if width == n and not train_full:
            width = _width(rng, n, goal, cut, matched_widths)
        height = _height(rng)
        prior = specs[int(rng.integers(0, len(specs)))][4]
        if source_like:
            background, foreground = base, test_palette[i % len(test_palette)]
        else:
            background, foreground = _choose_colors(rng, base, mixed, prior, rng.random() < 0.48)
        for _ in range(100):
            if (height, width, n, background, foreground) not in _SOURCE_SPECS:
                pair = _build_pair(n, width, height, background, foreground)
                if _key(pair["input"]) not in used:
                    break
            height = 2 + height % 29
        else:
            raise ValueError("could not create unique test input")
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    background = int(inp[1, 0])
    foreground = int(inp[0, 0])
    assert background != foreground and height >= 2
    n = 0
    while n < width and int(inp[0, n]) == foreground:
        n += 1
    assert 2 <= n <= width
    assert np.all(inp[0, n:] == background)
    assert np.all(inp[1:] == background)
    assert set(int(c) for c in np.unique(inp)) == {background, foreground}
    assert out.shape == ((_length(n) + width - 1) // width, width)
    assert not np.array_equal(inp, out)
    assert (height, width, n, background, foreground) not in _SOURCE_SPECS
    # Inspect the returned cells directly, including the last-row padding.
    position = 0
    for run in list(range(1, n + 1)) + list(range(n - 1, 0, -1)):
        assert np.all(out.flat[position:position + run] == foreground)
        position += run
        if position < _length(n):
            assert int(out.flat[position]) == background
            position += 1
    assert position == _length(n)
    assert np.all(out.flat[position:] == background)
    return n, width, height, background, foreground


def validate_generated_puzzle(puzzle: dict) -> None:
    """Assert schema, rule, source exclusion, and training coverage."""
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        keys = set()
        specs = {}
        for split in ("train", "test"):
            specs[split] = []
            for pair in puzzle[split]:
                spec = _inspect_pair(pair)
                key = _key(pair["input"])
                assert key not in keys
                keys.add(key)
                specs[split].append(spec)
        train = specs["train"]
        assert len({s[0] for s in train}) >= 2
        assert min(s[0] for s in train) <= 4
        assert max(s[0] for s in train) >= 6
        assert len({s[1] for s in train}) >= 2
        assert any(_length(s[0]) <= s[1] for s in train)
        assert any(_length(s[0]) > s[1] for s in train)
        if any(s[0] == s[1] for s in specs["test"]):
            assert any(s[0] == s[1] for s in train)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    """Check interfaces and measured variation over deterministic seed sets."""
    assert _puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123))
    assert _puzzle_key(generate_puzzle(456, 8, 3)) == _puzzle_key(generate_puzzle(456, 8, 3))
    assert _puzzle_key(generate_puzzle(500, 6, 2)) != _puzzle_key(generate_puzzle(501, 6, 2))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for offset, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                     (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + offset, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for offset, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                     (2, 2), (3, 3), (2, 4), (4, 4))):
        try:
            generate_puzzle(2000 + offset, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    lengths, widths, heights, backgrounds, foregrounds = set(), set(), set(), set(), set()
    exact = one = long = full = same_length_test = height_clone = mixed = reuse = 0
    train_five = holdout_five = short_train = extrapolated = familiar_width = 0
    test_single = stable = same_width_different_n = combined = source_like_count = 0
    cuts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        all_specs = train + test
        mixed += len({s[3] for s in all_specs}) > 1
        stable += len({s[3] for s in all_specs}) == 1
        reuse += bool({s[4] for s in train} & {s[4] for s in test})
        train_five += any(s[0] == 5 for s in train)
        holdout_five += (any(s[0] == 5 for s in test) and
                         any(s[0] < 5 for s in train) and any(s[0] > 5 for s in train)
                         and all(s[0] != 5 for s in train))
        short_train += any(s[0] == 2 for s in train)
        extrapolated += any(s[0] > max(t[0] for t in train) for s in test)
        source_like_count += (len({s[3] for s in all_specs}) == 1 and
                              not ({s[4] for s in train} & {s[4] for s in test}) and
                              any(s[0] > max(t[0] for t in train) for s in test))
        familiar_width += any(s[0] not in {t[0] for t in train} and
                              s[1] in {t[1] for t in train} for s in test)
        test_single += any(_length(s[0]) <= s[1] for s in test)
        same_width_different_n += any(a[1] == b[1] and a[0] != b[0]
                                      for i, a in enumerate(all_specs)
                                      for b in all_specs[i + 1:])
        combined += (len({s[3] for s in all_specs}) > 1 and
                     bool({s[4] for s in train} & {s[4] for s in test}) and
                     any((-_length(s[0])) % s[1] >= 5 for s in test))
        same_length_test += any(s[0] in {t[0] for t in train} and
                                s[1] not in {t[1] for t in train if t[0] == s[0]}
                                for s in test)
        height_clone += any(a[:2] == b[:2] and a[3:] == b[3:] and a[2] != b[2]
                            for i, a in enumerate(all_specs) for b in all_specs[i + 1:])
        for n, width, height, bg, fg in all_specs:
            lengths.add(n); widths.add(width); heights.add(height)
            backgrounds.add(bg); foregrounds.add(fg)
            pad, inside, separator, peak = _width_features(n, width)
            exact += pad == 0
            one += pad == 1
            long += pad >= 5
            full += n == width
            if inside: cuts.add("inside")
            if separator: cuts.add("separator")
            if peak: cuts.add("peak")
    assert {2, 3, 4, 5, 6, 7, 8}.issubset(lengths)
    assert max(lengths) >= 14 and len(widths) >= 20 and len(heights) >= 23
    assert len(backgrounds) == len(foregrounds) == 10
    assert exact >= 5 and one >= 5 and long >= 30
    assert full >= 2 and cuts == {"inside", "separator", "peak"}
    assert same_length_test >= 5 and height_clone >= 2
    assert mixed >= 12 and reuse >= 8
    assert train_five >= 5 and holdout_five >= 5 and short_train >= 5
    assert extrapolated >= 8 and familiar_width >= 5 and test_single >= 1
    assert stable >= 5 and same_width_different_n >= 5 and combined >= 5
    assert source_like_count >= 3
