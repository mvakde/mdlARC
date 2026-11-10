"""Generate complete-frame counting puzzles with a row-major 3 by 3 tally."""

import numpy as np


_SOURCE_PAIR_ROWS = {
    (("000000000", "006111111", "000160601", "000106001", "000100061",
      "060160001", "000111111", "000600000", "000000000"),
     ("666", "660", "000")),
    (("0000000", "0400040", "0040000", "1111114", "1000410",
      "1404010", "1111110"), ("444", "000", "000")),
    (("000003000", "301111100", "001003100", "001000130", "001003100",
      "001300100", "031000100", "001030130", "001111100"),
     ("333", "300", "000")),
    (("200000200", "111111100", "102000100", "100020120", "100002100",
      "120000102", "111111100", "000200000", "000000020"),
     ("222", "200", "000")),
}


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    n = _optional_count(num_train, "num_train", 2, 10)
    m = _optional_count(num_test, "num_test", 1, 4)
    if n is not None and m is not None:
        if n <= m:
            raise ValueError("num_train must exceed num_test")
        return n, m
    if n is None and m is None:
        m = 1 + seed % 4
        low = max(2, m + 1)
        n = low + (seed // 4) % (11 - low)
    elif n is None:
        n = int(rng.integers(max(2, m + 1), 11))
    else:
        m = int(rng.integers(1, min(4, n - 1) + 1))
    return n, m


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _source_copy(pair):
    return (_rows(pair["input"]), _rows(pair["output"])) in _SOURCE_PAIR_ROWS


def _frame_bounds(mask):
    coordinates = np.argwhere(mask)
    if len(coordinates) < 8:
        return None
    top, left = coordinates.min(axis=0)
    bottom, right = coordinates.max(axis=0)
    top, bottom, left, right = map(int, (top, bottom, left, right))
    if bottom - top < 2 or right - left < 2:
        return None
    expected = np.zeros(mask.shape, dtype=bool)
    expected[top, left:right + 1] = True
    expected[bottom, left:right + 1] = True
    expected[top:bottom + 1, left] = True
    expected[top:bottom + 1, right] = True
    if not np.array_equal(mask, expected):
        return None
    return top, bottom, left, right


def _contains_frame(mask):
    """Detect even a smaller accidental payload rectangle."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    for ti, top in enumerate(rows):
        for bottom in rows[ti + 1:]:
            if bottom - top < 2:
                continue
            for li, left in enumerate(columns):
                for right in columns[li + 1:]:
                    if right - left < 2:
                        continue
                    if (mask[top, left:right + 1].all()
                            and mask[bottom, left:right + 1].all()
                            and mask[top:bottom + 1, left].all()
                            and mask[top:bottom + 1, right].all()):
                        return True
    return False


def _analyze(pair):
    """Recover all roles and the count from the visible grids alone."""
    grid, output = pair["input"], pair["output"]
    assert output.shape == (3, 3)
    colors = [int(v) for v in np.unique(grid)]
    assert len(colors) == 3
    frames = [(color, _frame_bounds(grid == color)) for color in colors]
    frames = [(color, bounds) for color, bounds in frames if bounds is not None]
    assert len(frames) == 1
    frame, (top, bottom, left, right) = frames[0]
    others = [color for color in colors if color != frame]
    frequencies = [(int(np.count_nonzero(grid == color)), color) for color in others]
    frequencies.sort(reverse=True)
    assert frequencies[0][0] > frequencies[1][0]
    background, payload = frequencies[0][1], frequencies[1][1]
    assert not _contains_frame(grid == payload)
    inside = int(np.count_nonzero(grid[top + 1:bottom, left + 1:right] == payload))
    outside = int(np.count_nonzero(grid == payload)) - inside
    assert 0 <= inside <= 9
    assert inside + outside >= 1
    assert inside != 0 or outside >= 1
    expected = np.full((3, 3), background, dtype=np.int8)
    expected.flat[:inside] = payload
    assert np.array_equal(output, expected)
    return {
        "background": background, "frame": frame, "payload": payload,
        "count": inside, "outside": outside,
        "height": grid.shape[0], "width": grid.shape[1],
        "top": top, "bottom": bottom, "left": left, "right": right,
        "frame_height": bottom - top + 1, "frame_width": right - left + 1,
        "area": (bottom - top - 1) * (right - left - 1),
    }


def _orientation(info):
    h, w = info["frame_height"], info["frame_width"]
    return "square" if h == w else ("horizontal" if w > h else "vertical")


def _count_and_exterior_schedule(n, rng):
    # The first demonstration crosses the first output row. Larger training
    # sets sometimes demonstrate the third row as well.
    first = int(rng.choice((7, 8) if rng.random() < 0.48 else (4, 5, 6)))
    second_options = [v for v in range(1, 9) if v != first]
    second = int(rng.choice(second_options))
    counts = [first, second]
    outside = [int(rng.integers(2, 8)), int(rng.integers(0, 11))]
    if n >= 3:
        counts.append(first if n >= 4 else int(rng.integers(0, 10)))
        candidates = [v for v in range(11) if v != outside[0]]
        outside.append(int(rng.choice(candidates)))
    if n >= 4:
        total = counts[0] + outside[0]
        candidates = [v for v in range(10) if v != first and 0 <= total - v <= 10]
        counts.append(int(rng.choice(candidates)))
        outside.append(total - counts[-1])
    while len(counts) < n:
        counts.append(int(rng.integers(0, 10)))
        outside.append(int(rng.integers(0, 11)))
    for i, count in enumerate(counts):
        if count == 0 and outside[i] == 0:
            outside[i] = int(rng.integers(1, 5))
    order = rng.permutation(n)
    return [(counts[int(i)], outside[int(i)]) for i in order]


def _palette_schedule(n, m, rng):
    colors = [int(v) for v in rng.permutation(10)]
    stable = rng.random() < 0.46
    train = []
    if stable:
        bg, frame = colors[:2]
        for i in range(n):
            payload = colors[2 + i % min(n, 5)]
            train.append((bg, frame, payload))
        payloads = [p for _, _, p in train]
        tests = []
        for _ in range(m):
            if rng.random() < 0.58:
                payload = int(rng.choice(payloads))
            else:
                new = [c for c in colors[2:] if c not in payloads]
                payload = int(rng.choice(new if new else payloads))
            tests.append((bg, frame, payload))
    else:
        train.append(tuple(colors[:3]))
        for i in range(1, n):
            if i == 1 and rng.random() < 0.42:
                bg, frame = train[0][1], train[0][0]
                payload = int(rng.choice([c for c in colors if c not in (bg, frame)]))
            else:
                selected = [int(v) for v in rng.choice(10, 3, replace=False)]
                bg, frame, payload = selected
                if i == 1 and (bg, frame) == train[0][:2]:
                    bg, frame = frame, bg
            train.append((bg, frame, payload))
        payloads = [p for _, _, p in train]
        tests = []
        for _ in range(m):
            selected = [int(v) for v in rng.choice(10, 2, replace=False)]
            bg, frame = selected
            available = [c for c in colors if c not in (bg, frame)]
            reuse = [c for c in payloads if c in available]
            novel = [c for c in available if c not in payloads]
            payload = int(rng.choice((reuse if reuse else available)
                                     if rng.random() < 0.58 else
                                     (novel if novel else available)))
            tests.append((bg, frame, payload))
    return train, tests


def _geometry(rng, count, regime, orientation=None):
    limit = 12 if regime == "source_train" else 14
    candidates = [(h, w) for h in range(3, limit + 1)
                  for w in range(3, limit + 1)
                  if (h - 2) * (w - 2) >= count
                  and (count != 9 or (h - 2) * (w - 2) > 9)]
    if orientation is None:
        orientation = ("square", "horizontal", "vertical")[int(rng.integers(0, 3))]
    candidates = [(h, w) for h, w in candidates
                  if ("square" if h == w else "horizontal" if w > h else "vertical")
                  == orientation]
    shape_mode = int(rng.choice(3, p=(0.24, 0.21, 0.55)))
    if shape_mode == 0:
        narrow = [(h, w) for h, w in candidates if min(h - 2, w - 2) == 1]
        if narrow:
            candidates = narrow
    elif shape_mode == 1:
        extreme = [(h, w) for h, w in candidates
                   if max(h, w) / min(h, w) >= 2.0]
        if extreme:
            candidates = extreme
    # Weight modest frames, but retain the whole legal range.
    weights = np.array([1 / (1 + 0.10 * (h - 3) * (w - 3))
                        for h, w in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _sample_positions(rng, positions, number, bounds, shape):
    if number == 0:
        return []
    top, bottom, left, right = bounds
    height, width = shape
    def distance(position):
        row, column = position
        return max(max(top - row, 0, row - bottom),
                   max(left - column, 0, column - right))
    groups = [
        [p for p in positions if distance(p) == 1],
        [p for p in positions if p[0] in (0, height - 1)
         and p[1] in (0, width - 1)],
        [p for p in positions if distance(p) >= 4],
    ]
    selected = []
    if rng.random() < 0.68:
        group = groups[int(rng.integers(0, len(groups)))]
        if group:
            selected.append(group[int(rng.integers(0, len(group)))])
    remaining = [p for p in positions if p not in selected]
    if len(selected) < number:
        indices = rng.choice(len(remaining), number - len(selected), replace=False)
        selected.extend(remaining[int(i)] for i in np.atleast_1d(indices))
    return selected


def _make_pair(rng, colors, count, outside, regime, orientation=None):
    background, frame, payload = colors
    for _ in range(120):
        fh, fw = _geometry(rng, count, regime, orientation)
        margin_limit = 7 if regime == "source_train" else 8
        margins = [int(rng.integers(0, margin_limit)) for _ in range(4)]
        top, bottom_margin, left, right_margin = margins
        height, width = top + fh + bottom_margin, left + fw + right_margin
        if height > 30 or width > 30:
            continue
        bottom, right = top + fh - 1, left + fw - 1
        grid = np.full((height, width), background, dtype=np.int8)
        grid[top, left:right + 1] = frame
        grid[bottom, left:right + 1] = frame
        grid[top:bottom + 1, left] = frame
        grid[top:bottom + 1, right] = frame
        inside_positions = [(r, c) for r in range(top + 1, bottom)
                            for c in range(left + 1, right)]
        outside_positions = [(int(r), int(c)) for r, c in np.argwhere(grid == background)
                             if not (top < r < bottom and left < c < right)]
        if len(outside_positions) < outside:
            continue
        if len(inside_positions) + len(outside_positions) - count - outside <= count + outside + 2:
            continue
        for _ in range(12):
            indices = rng.choice(len(inside_positions), count, replace=False)
            inside = [inside_positions[int(i)] for i in np.atleast_1d(indices)]
            exterior = _sample_positions(rng, outside_positions, outside,
                                         (top, bottom, left, right), grid.shape)
            candidate = grid.copy()
            for r, c in inside + exterior:
                candidate[r, c] = payload
            if _contains_frame(candidate == payload):
                continue
            output = np.full((3, 3), background, dtype=np.int8)
            output.flat[:count] = payload
            pair = {"input": candidate, "output": output}
            try:
                _analyze(pair)
            except AssertionError:
                continue
            return pair
    raise RuntimeError("unable to construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    n, m = _pair_counts(seed, rng, num_train, num_test)
    train_specs = _count_and_exterior_schedule(n, rng)
    test_specs = [(int(rng.integers(0, 10)), int(rng.integers(0, 11)))
                  for _ in range(m)]
    test_specs = [(count, outside if count or outside else 1)
                  for count, outside in test_specs]
    train_palettes, test_palettes = _palette_schedule(n, m, rng)
    repeated_orientation = ("square", "horizontal", "vertical")[
        int(rng.integers(0, 3))] if rng.random() < 0.26 else None
    used = set()
    puzzle = {"train": [], "test": []}
    for split, specs, palettes, regime in (
        ("train", train_specs, train_palettes, "source_train"),
        ("test", test_specs, test_palettes, "target_test"),
    ):
        for (count, outside), colors in zip(specs, palettes):
            for _ in range(80):
                pair = _make_pair(rng, colors, count, outside, regime,
                                  repeated_orientation)
                key = _grid_key(pair["input"])
                if key not in used and not _source_copy(pair):
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to obtain a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _source_copy(pair)
            analyses[split].append(_analyze(pair))
    train, test = analyses["train"], analyses["test"]
    assert any(4 <= x["count"] <= 8 for x in train)
    assert len({x["count"] for x in train}) >= 2
    assert any(x["outside"] > 0 for x in train)
    train_roles = {(x["background"], x["frame"]) for x in train}
    if len(train_roles) == 1:
        assert {(x["background"], x["frame"]) for x in test} == train_roles
    train_features = {(x["height"], x["width"], x["top"], x["left"],
                       x["frame_height"], x["frame_width"], x["count"],
                       x["outside"], x["background"], x["frame"], x["payload"])
                      for x in train}
    for x in test:
        feature = (x["height"], x["width"], x["top"], x["left"],
                   x["frame_height"], x["frame_width"], x["count"],
                   x["outside"], x["background"], x["frame"], x["payload"])
        assert feature not in train_features


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(12345, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(generate_puzzle(12345, 7, 3))
    assert _puzzle_key(a) != _puzzle_key(generate_puzzle(12346, 7, 3))
    requests = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                (2, None), (10, None), (6, 2))
    for seed, (n, m) in enumerate(requests, 20000):
        puzzle = generate_puzzle(seed, n, m)
        validate_generated_puzzle(puzzle)
        if n is not None:
            assert len(puzzle["train"]) == n
        if m is not None:
            assert len(puzzle["test"]) == m
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(40000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    train_sizes, test_sizes, counts, test_counts = set(), set(), set(), set()
    orientations, test_orientations, shapes, frame_shapes = set(), set(), set(), set()
    margins, outside_values, palettes, regimes, input_keys = set(), set(), set(), set(), set()
    same_total = same_inside = third_row = repeated = mixed = 0
    exterior_relations = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_sizes.add(len(puzzle["train"]))
        test_sizes.add(len(puzzle["test"]))
        train = [_analyze(p) for p in puzzle["train"]]
        test = [_analyze(p) for p in puzzle["test"]]
        third_row += any(x["count"] >= 7 for x in train)
        repeated += len({_orientation(x) for x in train + test}) == 1
        mixed += len({_orientation(x) for x in train + test}) > 1
        regimes.add(len({(x["background"], x["frame"]) for x in train}) > 1)
        same_total += any(a["count"] != b["count"] and
                          a["count"] + a["outside"] == b["count"] + b["outside"]
                          for i, a in enumerate(train) for b in train[i + 1:])
        same_inside += any(a["count"] == b["count"] and a["outside"] != b["outside"]
                           for i, a in enumerate(train) for b in train[i + 1:])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _analyze(pair)
                input_keys.add(_grid_key(pair["input"]))
                counts.add(x["count"])
                shapes.add(pair["input"].shape)
                frame_shapes.add((x["frame_height"], x["frame_width"]))
                margins.add((x["top"], x["height"] - x["bottom"] - 1,
                             x["left"], x["width"] - x["right"] - 1))
                outside_values.add(x["outside"])
                palettes.add((x["background"], x["frame"], x["payload"]))
                orientations.add(_orientation(x))
                if split == "test":
                    test_counts.add(x["count"])
                    test_orientations.add(_orientation(x))
                    exterior_relations.add(np.sign(x["outside"] - x["count"]))
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert counts == set(range(10)) and test_counts == set(range(10))
    assert orientations == test_orientations == {"horizontal", "vertical", "square"}
    assert len(shapes) >= 35 and len(frame_shapes) >= 35 and len(margins) >= 60
    assert len(outside_values) >= 10 and len(palettes) >= 35
    assert regimes == {False, True} and third_row >= 5
    assert repeated >= 3 and mixed >= 10
    assert same_total >= 3 and same_inside >= 3
    assert exterior_relations == {-1, 0, 1}
    assert len(input_keys) >= 200
