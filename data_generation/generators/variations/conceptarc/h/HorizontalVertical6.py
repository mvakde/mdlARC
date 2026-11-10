"""Orientation-of-combined-marks ARC puzzle generator."""

import hashlib

import numpy as np


_SOURCE = frozenset({
    "bd48ad3bd913d520dd88d384be89da12249d3429e3a8c9dbca9fb5a1ccba5abb",
    "44797b11d162abf8883c32b2509db6e5079ee12275e15eca14365aaae1f7d642",
    "59a1856473b0e1d1dfa1e3cbeb4160e8677f31180a2faa8fafa17a5233fb1e5d",
    "2857d4a133f800de6c98c9759923d1b1129235202eff186ed0905a7602f6cd19",
    "a342b17f4bdd37b8ee198b7ec10c4182daa47e5f156dc248cce2e124d1904a44",
    "6d2de8c4808ae619c241eddee2566b7dfd8053bbb002d3267999bf4f434b0cf1",
})


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _palette(rng, marker):
    candidates = [c for c in range(10) if c != marker]
    if marker != 0 and rng.random() < 0.43:
        background = 0
    else:
        background = int(rng.choice(candidates))
    candidates.remove(background)
    foreground = 1 if background == 0 and 1 in candidates and rng.random() < 0.3 \
        else int(rng.choice(candidates))
    return background, foreground


def _size(rng, fixed_14):
    if fixed_14:
        return 14
    roll = rng.random()
    if roll < 0.17:
        return int(rng.integers(8, 12))
    if roll < 0.35:
        return 14
    if roll < 0.82:
        return int(rng.integers(12, 23))
    return int(rng.integers(23, 31))


def _box_shape(rng, size, orientation, style):
    low = 5 if style == "islands" else 3
    high = size - 2
    if rng.random() < 0.3:
        high = min(high, 7)
    short = int(rng.integers(low, high + 1))
    roll = rng.random()
    if roll < 0.23:
        long = short + 1
    elif roll < 0.48:
        long = int(rng.integers(max(short + 1, size - 2), size + 1))
    else:
        long = int(rng.integers(short + 1, size + 1))
    return (long, short) if orientation == "tall" else (short, long)


def _anchors(rng, height, width):
    def on_edge(length):
        if rng.random() < 0.78:
            return int(rng.integers(1, length - 1))
        return 0 if rng.integers(2) else length - 1
    return [(0, on_edge(width)), (height - 1, on_edge(width)),
            (on_edge(height), 0), (on_edge(height), width - 1)]


def _path(mask, rng, start, end):
    row, col = start
    mask[row, col] = True
    while (row, col) != end:
        steps = []
        if row < end[0]:
            steps.append((row + 1, col))
        if row > end[0]:
            steps.append((row - 1, col))
        if col < end[1]:
            steps.append((row, col + 1))
        if col > end[1]:
            steps.append((row, col - 1))
        row, col = steps[int(rng.integers(len(steps)))]
        mask[row, col] = True


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    result = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        stack, cells = [(row, col)], []
        visited[row, col] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] \
                            and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
        result.append(cells)
    return result


def _mask(rng, height, width, orientation, style):
    for _ in range(32):
        mask = np.zeros((height, width), dtype=bool)
        anchors = _anchors(rng, height, width)
        if style in ("ribbon", "branched"):
            if style == "ribbon":
                order = rng.permutation(4)
                for i in range(3):
                    _path(mask, rng, anchors[int(order[i])], anchors[int(order[i + 1])])
            else:
                hub = (int(rng.integers(1, height - 1)),
                       int(rng.integers(1, width - 1)))
                for anchor in anchors:
                    _path(mask, rng, hub, anchor)
            for _ in range(int(rng.integers(0, max(2, (height + width) // 2)))):
                points = np.argwhere(mask)
                r, c = points[int(rng.integers(len(points)))]
                dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    mask[rr, cc] = True
        elif style == "islands":
            mask = rng.random((height, width)) < rng.uniform(0.025, 0.11)
            # A local stroke can disagree with the aspect of all marks together.
            if orientation == "tall":
                r = int(rng.integers(1, height - 1))
                c = int(rng.integers(1, width - 1))
                span = int(rng.integers(2, min(width, 6) + 1))
                mask[r, c:min(width, c + span)] = True
            else:
                c = int(rng.integers(1, width - 1))
                r = int(rng.integers(1, height - 1))
                span = int(rng.integers(2, min(height, 6) + 1))
                mask[r:min(height, r + span), c] = True
            for _ in range(int(rng.integers(0, 4))):
                r, c = int(rng.integers(height)), int(rng.integers(width))
                mask[r, c] = True
                if rng.random() < 0.6 and r + 1 < height:
                    mask[r + 1, c] = True
        elif style == "scatter":
            mask = rng.random((height, width)) < rng.uniform(0.08, 0.30)
        elif style == "strokes":
            for _ in range(int(rng.integers(3, 9))):
                r, c = int(rng.integers(height)), int(rng.integers(width))
                span = int(rng.integers(2, min(8, max(height, width)) + 1))
                if rng.integers(2):
                    mask[r, c:min(width, c + span)] = True
                else:
                    mask[r:min(height, r + span), c] = True
        elif style == "dense":
            mask = rng.random((height, width)) < rng.uniform(0.31, 0.56)
        else:
            raise ValueError("unknown mark style")
        for r, c in anchors:
            mask[r, c] = True
        if style != "islands" or len(_components(mask)) >= 2:
            return mask
    raise RuntimeError("could not construct separated islands")


def _make_pair(rng, orientation, style, palette, marker, fixed_14):
    size = _size(rng, fixed_14)
    height, width = _box_shape(rng, size, orientation, style)
    mask = _mask(rng, height, width, orientation, style)
    top = int(rng.integers(0 if orientation == "tall" else 1, size - height + 1))
    left = int(rng.integers(0, size - width + (0 if orientation == "tall" else 1)))
    background, foreground = palette
    inp = np.full((size, size), background, dtype=np.uint8)
    inp[top:top + height, left:left + width][mask] = foreground
    if np.count_nonzero(inp == foreground) >= size * size // 2:
        raise ValueError("foreground too dense")
    out = inp.copy()
    if orientation == "tall":
        out[:, -1] = marker
    else:
        out[0, :] = marker
    return {"input": inp, "output": out}


def _specs(rng, num_train, num_test, basic):
    if basic:
        train = [("tall", "ribbon"), ("wide", "branched")]
        train += [("tall" if rng.integers(2) else "wide",
                   "ribbon" if rng.integers(2) else "branched")
                  for _ in range(num_train - 2)]
    else:
        train = [("tall", "islands"), ("wide", "islands")]
        if num_train == 2:
            i = int(rng.integers(2))
            train[i] = (train[i][0], "ribbon" if rng.integers(2) else "branched")
        else:
            train.append(("tall" if rng.integers(2) else "wide",
                          "ribbon" if rng.integers(2) else "branched"))
            styles = ("ribbon", "branched", "islands", "scatter", "strokes", "dense")
            train += [("tall" if rng.integers(2) else "wide",
                       styles[int(rng.integers(len(styles)))])
                      for _ in range(num_train - 3)]
    rng.shuffle(train)
    styles = ("ribbon", "branched") if basic else \
        ("islands", "islands", "scatter", "strokes", "ribbon", "branched", "dense")
    test = [("tall" if rng.integers(2) else "wide",
             styles[int(rng.integers(len(styles)))])
            for _ in range(num_test)]
    return train, test


def _source_hash(grid):
    return hashlib.sha256(bytes(grid.shape) +
                          np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    roll = rng.random()
    palette_mode = "source" if roll < 0.10 else "shared" if roll < 0.55 else "vary"
    marker = 3 if palette_mode == "source" else int(rng.integers(10))
    shared_palette = (0, 1) if palette_mode == "source" else _palette(rng, marker)
    fixed_14 = rng.random() < 0.12
    basic = rng.random() < 0.18
    train_specs, test_specs = _specs(rng, num_train, num_test, basic)
    puzzle = {"train": [], "test": []}
    seen, first_palette = set(), None
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, (orientation, style) in enumerate(specs):
            for _ in range(100):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                palette = shared_palette if palette_mode != "vary" else _palette(pair_rng, marker)
                if split == "train" and index == 1 and palette_mode == "vary" \
                        and palette == first_palette:
                    continue
                try:
                    pair = _make_pair(pair_rng, orientation, style, palette,
                                      marker, fixed_14)
                except (ValueError, RuntimeError):
                    continue
                inp = pair["input"]
                key = (inp.shape, inp.tobytes())
                if key in seen or _source_hash(inp) in _SOURCE:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_palette = palette
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    return puzzle


def _facts(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape and inp.shape[0] == inp.shape[1]
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2 and counts.max() > counts.min()
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    positions = np.argwhere(inp == foreground)
    assert len(positions) >= 4
    top, left = positions.min(axis=0)
    bottom, right = positions.max(axis=0)
    height, width = int(bottom - top + 1), int(right - left + 1)
    assert min(height, width) >= 3 and height != width
    changed = out != inp
    assert int(changed.sum()) == inp.shape[0]
    marker_values = set(np.unique(out[changed]).tolist())
    assert len(marker_values) == 1
    marker = int(marker_values.pop())
    assert marker not in (background, foreground)
    expected = inp.copy()
    if height > width:
        assert np.all(inp[:, -1] == background)
        expected[:, -1] = marker
        orientation = "tall"
    else:
        assert np.all(inp[0, :] == background)
        expected[0, :] = marker
        orientation = "wide"
    assert np.array_equal(out, expected)
    assert _source_hash(inp) not in _SOURCE
    corners = sum(bool(inp[r, c] == foreground)
                  for r in (top, bottom) for c in (left, right))
    unselected_touch = bool(top == 0) if orientation == "tall" \
        else bool(right == inp.shape[1] - 1)
    return (orientation, background, foreground, marker, height, width,
            len(_components(inp == foreground)), corners, unselected_touch)


def _predict_from_train(train, test_input):
    learned = set()
    for pair in train:
        changed = pair["input"] != pair["output"]
        learned.update(np.unique(pair["output"][changed]).tolist())
    assert len(learned) == 1
    marker = int(learned.pop())
    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 2
    foreground = int(colors[int(np.argmin(counts))])
    positions = np.argwhere(test_input == foreground)
    height, width = np.ptp(positions, axis=0) + 1
    result = test_input.copy()
    if height > width:
        result[:, -1] = marker
    else:
        result[0, :] = marker
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= value <= 30 for value in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            facts[split].append(_facts(pair))
    assert {f[0] for f in facts["train"]} == {"tall", "wide"}
    assert len({f[3] for split in facts for f in facts[split]}) == 1
    if any(f[6] > 1 for f in facts["test"]):
        assert any(f[6] > 1 for f in facts["train"])
    if len({(f[1], f[2]) for f in facts["test"]}) > 1:
        assert len({(f[1], f[2]) for f in facts["train"]}) > 1
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_train(puzzle["train"],
                                                 pair["input"]), pair["output"])


def _fingerprint(puzzle):
    return tuple(tuple((p["input"].shape, p["input"].tobytes(),
                        p["output"].tobytes()) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        puzzles.append(puzzle)
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    facts = [_facts(pair) for puzzle in puzzles
             for split in ("train", "test") for pair in puzzle[split]]
    shapes = {pair["input"].shape for puzzle in puzzles
              for split in ("train", "test") for pair in puzzle[split]}
    assert min(s[0] for s in shapes) <= 10 and max(s[0] for s in shapes) >= 27
    assert (14, 14) in shapes and len(shapes) >= 18
    assert len({f[7] for f in facts}) >= 3 and 0 in {f[7] for f in facts}
    assert any(abs(f[4] - f[5]) == 1 for f in facts)
    assert any(f[6] >= 4 for f in facts)
    assert any(f[8] for f in facts)
    assert any((f[1], f[2], f[3]) == (0, 1, 3) for f in facts)
    assert sum(len({(_facts(p)[1], _facts(p)[2]) for p in puzzle["train"]}) > 1
               for puzzle in puzzles) >= 8
    assert {0, 1, 2, 3, 4} <= {f[7] for f in facts}
    assert sum(max(f[4], f[5]) >= pair["input"].shape[0] - 1
               for puzzle in puzzles for split in ("train", "test")
               for pair in puzzle[split] for f in (_facts(pair),)) >= 30
    assert all(any(_facts(pair)[0] == orientation for pair in puzzle["train"])
               for puzzle in puzzles for orientation in ("tall", "wide"))
    assert all(any(_facts(pair)[6] > 1 for pair in puzzle["train"])
               for puzzle in puzzles
               if any(_facts(pair)[6] > 1 for pair in puzzle["test"]))
    assert all(any(_facts(pair)[6] > 1 and _facts(pair)[0] == orientation
                   for puzzle in puzzles for pair in puzzle["train"])
               for orientation in ("tall", "wide"))
    assert all(any(_facts(pair)[6] >= 4 for puzzle in puzzles
                   for pair in puzzle[split]) for split in ("train", "test"))
    assert all(any(_facts(pair)[8] for puzzle in puzzles
                   for pair in puzzle[split]) for split in ("train", "test"))
    assert any(all(pair["input"].shape == (14, 14)
                   for split in ("train", "test") for pair in puzzle[split])
               for puzzle in puzzles)
    assert sum(len({pair["input"].shape for split in ("train", "test")
                    for pair in puzzle[split]}) > 1 for puzzle in puzzles) >= 25
    assert any(_facts(pair)[6] > 1 and max(_facts(pair)[4:6]) >= pair["input"].shape[0] - 1
               for puzzle in puzzles for split in ("train", "test")
               for pair in puzzle[split])
    opposite_local = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                fact = _facts(pair)
                if fact[6] < 2:
                    continue
                inp = pair["input"]
                largest = max(_components(inp == fact[2]), key=len)
                rows, cols = zip(*largest)
                local_height = max(rows) - min(rows) + 1
                local_width = max(cols) - min(cols) + 1
                if (fact[0] == "tall" and local_width > local_height) or (
                    fact[0] == "wide" and local_height > local_width
                ):
                    opposite_local += 1
    assert opposite_local >= 20
    sequences = {tuple(_facts(pair)[0] for pair in puzzle["test"])
                 for puzzle in puzzles}
    assert any(set(s) == {"wide"} for s in sequences)
    assert any(set(s) == {"tall"} for s in sequences)
    assert any(len(set(s)) == 2 for s in sequences)
    assert {"tall", "wide"} == {_facts(p["train"][0])[0] for p in puzzles}
    for seed, a, b in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                       (103, None, 1), (104, None, 4), (105, 2, None),
                       (106, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")
