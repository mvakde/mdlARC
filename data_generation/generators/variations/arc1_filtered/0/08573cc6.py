import numpy as np


_SOURCE = {
    (11, 4, 3, 0, 1, 5, 6),
    (11, 6, 6, 0, 1, 3, 2),
    (12, 6, 5, 0, 1, 5, 3),
    (13, 5, 6, 0, 1, 2, 8),
}
_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def _walk(n, row, col):
    cells, seen = [], {(row, col)}
    for arm in range(40):
        dr, dc = _DIRECTIONS[arm % 4]
        for step in range(arm + 2):
            row, col = row + dr, col + dc
            if not (0 <= row < n and 0 <= col < n):
                edge = "left" if col < 0 else "right" if col >= n else "top" if row < 0 else "bottom"
                return cells, edge, arm, step
            if (row, col) in seen:
                return None
            seen.add((row, col))
            cells.append((row, col, arm))
    return None


def _all_layouts():
    layouts = []
    for n in range(10, 19):
        for row in range(3, n - 2):
            for col in range(3, n - 2):
                walked = _walk(n, row, col)
                if walked is None:
                    continue
                cells, edge, arm, clip = walked
                occupied = {(r, c) for r, c, _ in cells}
                if arm >= 4 and 0 < clip < arm + 2 and not occupied.intersection({(0, 0), (0, 1)}):
                    layouts.append((n, row, col, edge, arm, clip))
    return tuple(layouts)


_LAYOUTS = _all_layouts()


def _layout(rng, used, selected, predicate):
    choices = [x for x in _LAYOUTS if x[:3] not in used and predicate(x)]
    assert choices
    sizes = sorted({x[0] for x in choices})
    n = sizes[int(rng.integers(len(sizes)))]
    choices = [x for x in choices if x[0] == n]
    edges = {x[3] for x in selected}
    traces = {(x[4], x[5]) for x in selected}
    weights = np.array([1 + 3 * (x[3] not in edges) + 6 * ((x[4], x[5]) not in traces) for x in choices], dtype=float)
    weights /= weights.sum()
    picked = choices[int(rng.choice(len(choices), p=weights))]
    used.add(picked[:3])
    selected.append(picked)
    return picked


def _pair(layout, colors):
    n, row, col = layout[:3]
    background, marker, horizontal, vertical = colors
    inp = np.full((n, n), background, dtype=np.int8)
    out = np.full((n, n), background, dtype=np.int8)
    inp[0, 0], inp[0, 1], inp[row, col] = horizontal, vertical, marker
    out[row, col] = marker
    for r, c, arm in _walk(n, row, col)[0]:
        out[r, c] = horizontal if arm % 2 == 0 else vertical
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be from {low} through {high}")
    return int(value)


def _cue(rng, palette):
    values = rng.choice(palette, size=2, replace=False)
    return int(values[0]), int(values[1])


def _colors(rng, cue, backgrounds, markers, require_new):
    available = [x for x in range(10) if x not in cue]
    bg_options = [x for x in available if x not in backgrounds] if require_new else available
    background = int(rng.choice(bg_options or available))
    marker_options = [x for x in available if x != background]
    new_markers = [x for x in marker_options if x not in markers]
    marker = int(rng.choice(new_markers if require_new and new_markers else marker_options))
    backgrounds.add(background)
    markers.add(marker)
    return background, marker, *cue


def _avoid_source(layout, colors):
    if (*layout[:3], *colors) not in _SOURCE:
        return colors
    bg, marker, horizontal, vertical = colors
    replacement = next(x for x in range(10) if x not in (bg, marker, horizontal, vertical))
    return replacement, marker, horizontal, vertical


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    # Direction, turn order, arm lengths, and ordered cue roles stay fixed.
    # The cue pool permits both familiar and new colors without encoding a rule.
    mode = int(rng.integers(4))
    palette = tuple(int(x) for x in rng.choice(10, size=int(rng.integers(4, 7)), replace=False))
    used, train_layouts, test_layouts = set(), [], []
    backgrounds, markers, train_cues = set(), set(), []
    puzzle = {"train": [], "test": []}

    for i in range(num_train):
        if i == 0:
            allowed = lambda x: x[0] in (11, 12) and x[3] in ("left", "bottom") and x[4] <= 6
        elif i == 1:
            first = train_layouts[0]
            if mode == 0:
                allowed = lambda x: x[0] in (11, 12) and x[0] != first[0] and x[3] in ("left", "bottom") and x[4] >= 5
            elif mode == 1:
                allowed = lambda x: 12 <= x[0] <= 15 and x[0] != first[0] and x[3] in ("top", "right") and x[4] >= 6
            elif mode == 2:
                allowed = lambda x: 12 <= x[0] <= 16 and x[0] != first[0] and x[3] != first[3] and x[4] >= 5
            else:
                allowed = lambda x: 13 <= x[0] <= 16 and x[4] >= 8
        else:
            allowed = lambda x: 10 <= x[0] <= 16
        if i == 1:
            old_allowed = allowed
            allowed = lambda x: old_allowed(x) and (x[4], x[5]) != (first[4], first[5])
        layout = _layout(rng, used, train_layouts, allowed)
        if i == 0:
            cue = _cue(rng, palette)
        elif i == 1:
            if int(rng.integers(3)) == 0:
                cue = train_cues[0][::-1]
            else:
                options = [(a, b) for a in palette for b in palette if a != b and (a, b) != train_cues[0]]
                cue = options[int(rng.integers(len(options)))]
        else:
            choice = int(rng.integers(3))
            if choice == 0:
                cue = train_cues[int(rng.integers(len(train_cues)))]
            elif choice == 1:
                cue = train_cues[int(rng.integers(len(train_cues)))][::-1]
            else:
                cue = _cue(rng, palette)
        train_cues.append(cue)
        colors = _avoid_source(layout, _colors(rng, cue, backgrounds, markers, i < 2))
        backgrounds.add(colors[0])
        puzzle["train"].append(_pair(layout, colors))

    seen_cues = {c for cue in train_cues for c in cue}
    unseen_cues = [c for c in range(10) if c not in seen_cues]
    novelty_order = tuple(int(x) for x in rng.permutation(3))
    for i in range(num_test):
        if i == 0 and mode == 0:
            allowed = lambda x: 13 <= x[0] <= 16 and x[3] == "top" and x[4] >= 7
        elif i == 0 and mode == 1:
            familiar = {x[3] for x in train_layouts}
            allowed = lambda x: 11 <= x[0] <= 16 and x[3] in familiar
        elif i == 0 and mode == 2:
            held = {"top", "right", "bottom", "left"} - {x[3] for x in train_layouts}
            allowed = lambda x: 11 <= x[0] <= 17 and x[3] in held
            if not any(x[:3] not in used and allowed(x) for x in _LAYOUTS):
                allowed = lambda x: 11 <= x[0] <= 17 and x[3] != train_layouts[0][3]
        elif i == 0:
            deepest = max(x[4] for x in train_layouts)
            allowed = lambda x: 14 <= x[0] <= 18 and x[4] > deepest
            if not any(x[:3] not in used and allowed(x) for x in _LAYOUTS):
                allowed = lambda x: 14 <= x[0] <= 18 and x[4] >= 9
        else:
            allowed = lambda x: 11 <= x[0] <= 18
        layout = _layout(rng, used, test_layouts, allowed)
        novelty = novelty_order[i % 3]
        if novelty == 0:
            familiar = train_cues[int(rng.integers(len(train_cues)))]
            cue = familiar if int(rng.integers(2)) else familiar[::-1]
        elif novelty == 1:
            known, new = int(rng.choice(sorted(seen_cues))), int(rng.choice(unseen_cues))
            cue = (known, new) if int(rng.integers(2)) else (new, known)
        else:
            cue = _cue(rng, unseen_cues)
        colors = _avoid_source(layout, _colors(rng, cue, backgrounds, markers, False))
        puzzle["test"].append(_pair(layout, colors))

    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    n = inp.shape[0]
    assert inp.shape == out.shape == (n, n) and 10 <= n <= 18
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert np.count_nonzero(counts == max(counts)) == 1
    horizontal, vertical = int(inp[0, 0]), int(inp[0, 1])
    assert len({background, horizontal, vertical}) == 3
    remaining = [(r, c) for r, c in np.argwhere(inp != background) if (r, c) not in ((0, 0), (0, 1))]
    assert len(remaining) == 1
    row, col = (int(x) for x in remaining[0])
    marker = int(inp[row, col])
    assert len({background, horizontal, vertical, marker}) == 4
    expected_input = np.full((n, n), background, dtype=np.int8)
    expected_input[0, 0], expected_input[0, 1], expected_input[row, col] = horizontal, vertical, marker
    assert np.array_equal(inp, expected_input)
    assert 3 <= row <= n - 3 and 3 <= col <= n - 3

    # Independent segment construction checks the cellwise generator.
    expected = np.full((n, n), background, dtype=np.int8)
    expected[row, col] = marker
    r, c, result = row, col, None
    for arm in range(40):
        dr, dc = _DIRECTIONS[arm % 4]
        for step in range(arm + 2):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n):
                edge = "left" if nc < 0 else "right" if nc >= n else "top" if nr < 0 else "bottom"
                result = (edge, arm, step)
                break
            assert expected[nr, nc] == background
            expected[nr, nc] = horizontal if arm % 2 == 0 else vertical
            r, c = nr, nc
        if result is not None:
            break
    assert result is not None
    edge, arm, clip = result
    assert arm >= 4 and 0 < clip < arm + 2
    assert np.array_equal(out, expected)
    assert out[0, 0] == out[0, 1] == background
    assert set(int(x) for x in np.unique(out)) == {background, marker, horizontal, vertical}
    assert not np.array_equal(inp, out)
    assert (n, row, col, background, marker, horizontal, vertical) not in _SOURCE
    return n, row, col, edge, arm, clip, background, marker, horizontal, vertical


def validate_generated_puzzle(puzzle: dict) -> None:
    assert type(puzzle) is dict and set(puzzle) == {"train", "test"}
    assert type(puzzle["train"]) is list and type(puzzle["test"]) is list
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    info = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert type(pair) is dict and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = (pair["input"].shape, pair["input"].astype(np.int8).tobytes())
            pair_key = (input_key, pair["output"].shape, pair["output"].astype(np.int8).tobytes())
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            details = _inspect_pair(pair)
            info[section].append(details)
            if section == "train":
                assert 10 <= details[0] <= 16
            else:
                assert 11 <= details[0] <= 18
    train = info["train"]
    assert any(x[0] in (11, 12) and x[3] in ("left", "bottom") and x[4] <= 6 for x in train)
    assert len({(x[4], x[5]) for x in train}) >= 2
    assert len({x[0] for x in train}) >= 2
    assert len({x[6] for x in train}) >= 2
    assert len({x[7] for x in train}) >= 2
    assert len({(x[8], x[9]) for x in train}) >= 2
    seen_cues = {c for x in train for c in x[8:10]}
    novelty = [sum(c not in seen_cues for c in x[8:10]) for x in info["test"]]
    if ns >= 3:
        assert {0, 1, 2} <= set(novelty)


def _signature(puzzle):
    return tuple(
        tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes()) for p in puzzle[section])
        for section in ("train", "test")
    )


def self_test() -> None:
    signatures, train_counts, test_counts = set(), set(), set()
    train_traces, test_traces, edges, backgrounds, markers, sizes = set(), set(), set(), set(), set(), set()
    novelty, overlaps = set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_inspect_pair(p) for p in puzzle["train"]]
        test = [_inspect_pair(p) for p in puzzle["test"]]
        overlaps += bool({x[0] for x in train} & {x[0] for x in test})
        seen_cues = {c for x in train for c in x[8:10]}
        train_traces.update((x[4], x[5]) for x in train)
        test_traces.update((x[4], x[5]) for x in test)
        novelty.update(sum(c not in seen_cues for c in x[8:10]) for x in test)
        for x in train + test:
            edges.add(x[3])
            backgrounds.add(x[6])
            markers.add(x[7])
            sizes.add(x[0])
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(train_traces) >= 12 and len(test_traces) >= 12
    assert edges == {"left", "right", "top", "bottom"}
    assert sizes >= set(range(10, 19))
    assert len(backgrounds) == len(markers) == 10
    assert novelty == {0, 1, 2} and overlaps >= 10

    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        first, second = generate_puzzle(902, nt, ns), generate_puzzle(902, nt, ns)
        assert _signature(first) == _signature(second)
        validate_generated_puzzle(first)
        if nt is not None:
            assert len(first["train"]) == nt
        if ns is not None:
            assert len(first["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(902, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
