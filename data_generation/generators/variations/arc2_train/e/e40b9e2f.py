"""Colored quarter-turn completion around a visible square core."""

import numpy as np


_SOURCE_PAIRS = (
    (
        ("0000000000", "0000000000", "0000000000", "0300000000",
         "0066000000", "0066000000", "0006000000", "0000000000",
         "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0360300000",
         "0066600000", "0666000000", "0306300000", "0000000000",
         "0000000000", "0000000000"),
    ),
    (
        ("0000000000", "0000000000", "0000000000", "0000700000",
         "0004740000", "0007470000", "0004740000", "0000004000",
         "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0040704000",
         "0004740000", "0077477000", "0004740000", "0040704000",
         "0000000000", "0000000000"),
    ),
    (
        ("0000000000", "0000000000", "0000009000", "0008880000",
         "0008880000", "0008880000", "0000000000", "0000000000",
         "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0090009000", "0008880000",
         "0008880000", "0008880000", "0090009000", "0000000000",
         "0000000000", "0000000000"),
    ),
    (
        ("0000000000", "0000000000", "0000200000", "0333230000",
         "0002320000", "0333230000", "0000000000", "0000000000",
         "0000000000", "0000000000"),
        ("0000000000", "0003030000", "0003230000", "0333233300",
         "0022322000", "0333233300", "0003230000", "0003030000",
         "0000000000", "0000000000"),
    ),
)


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _text_key(inp, out):
    def rows(grid):
        return tuple("".join(str(int(x)) for x in row) for row in grid)
    return rows(inp), rows(out)


def _orbit(x, y):
    cells = []
    for _ in range(4):
        cells.append((x, y))
        x, y = -y, x
    return tuple(sorted(set(cells)))


def _pool(span):
    center2 = span - 1
    core_radius2 = 1 if span % 2 == 0 else 2
    result = set()
    for row in range(span):
        for col in range(span):
            rel = (2 * row - center2, 2 * col - center2)
            if max(map(abs, rel)) > core_radius2:
                result.add(_orbit(*rel))
    return sorted(result)


def _index(center2, rel2):
    return ((center2 + rel2[0]) // 2, (center2 + rel2[1]) // 2)


def _adjacent(mask, row, col):
    h, w = mask.shape
    for rr in range(max(0, row - 1), min(h, row + 2)):
        for cc in range(max(0, col - 1), min(w, col + 2)):
            if mask[rr, cc]:
                return True
    return False


def _connected(mask):
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        row, col = stack.pop()
        for rr in range(max(0, row - 1), min(h, row + 2)):
            for cc in range(max(0, col - 1), min(w, col + 2)):
                cell = (rr, cc)
                if mask[cell] and cell not in seen:
                    seen.add(cell)
                    stack.append(cell)
    return len(seen) == len(cells)


def _background(grid):
    border = np.concatenate(
        (grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1])
    )
    assert len(border) and np.all(border == border[0])
    return int(border[0])


def _closure(inp, center2, background):
    result = inp.copy()
    h, w = inp.shape
    for row, col in np.argwhere(inp != background):
        color = int(inp[row, col])
        x = 2 * int(row) - center2[0]
        y = 2 * int(col) - center2[1]
        for _ in range(4):
            rr2, cc2 = center2[0] + x, center2[1] + y
            if rr2 % 2 or cc2 % 2:
                return None
            rr, cc = rr2 // 2, cc2 // 2
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if result[rr, cc] not in (background, color):
                return None
            result[rr, cc] = color
            x, y = -y, x
    return result


def _candidate_centers(inp, background):
    """Use only the input: full symmetric 2x2 or 3x3 cores."""
    h, w = inp.shape
    found = {}
    for side in (2, 3):
        for top in range(1, h - side):
            for left in range(1, w - side):
                core = inp[top:top + side, left:left + side]
                if np.any(core == background):
                    continue
                if not np.array_equal(core, np.rot90(core)):
                    continue
                center2 = (2 * top + side - 1, 2 * left + side - 1)
                completed = _closure(inp, center2, background)
                if completed is None or np.array_equal(completed, inp):
                    continue
                occupied = np.argwhere(completed != background)
                low = occupied.min(axis=0)
                high = occupied.max(axis=0)
                if int(high[0] - low[0]) != int(high[1] - low[1]):
                    continue
                if tuple(int(a + b) for a, b in zip(low, high)) != center2:
                    continue
                found[center2] = completed
    return found


def _pair_colors(rng, previous_background, ncolors):
    backgrounds = [x for x in range(10) if x != previous_background]
    background = int(backgrounds[int(rng.integers(len(backgrounds)))])
    choices = [x for x in range(10) if x != background]
    if background != 0 and rng.random() < 0.42:
        choices.remove(0)
        extra = rng.choice(choices, size=ncolors - 1, replace=False)
        palette = [0] + [int(x) for x in extra]
    else:
        palette = [int(x) for x in rng.choice(
            choices, size=ncolors, replace=False
        )]
    rng.shuffle(palette)
    return background, tuple(palette)


def _core(rng, span, palette):
    center2 = span - 1
    local = np.full((span, span), -1, dtype=np.int8)
    if span % 2 == 0:
        for row in (span // 2 - 1, span // 2):
            for col in (span // 2 - 1, span // 2):
                local[row, col] = palette[0]
        kind = "solid"
    else:
        kind = ("solid", "alternating", "ring", "cross", "three")[
            int(rng.integers(0, 5 if len(palette) == 3 else 4))
        ]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if kind == "solid":
                    color = palette[0]
                elif kind == "alternating":
                    color = palette[0] if (dr + dc) % 2 == 0 else palette[1]
                elif kind == "ring":
                    color = palette[0] if dr == dc == 0 else palette[1]
                elif kind == "cross":
                    color = palette[0] if dr == 0 or dc == 0 else palette[1]
                else:
                    color = palette[0] if dr == dc == 0 else (
                        palette[1] if dr == 0 or dc == 0 else palette[2]
                    )
                local[span // 2 + dr, span // 2 + dc] = color
    assert center2 >= 3
    return local, kind


def _make_local(rng, span, palette, desired):
    visible, kind = _core(rng, span, palette)
    output = visible.copy()
    style = ("arms", "corners", "steps", "mixed")[
        int(rng.integers(4))
    ]
    center2 = span - 1
    orbits = _pool(span)
    selected = set()
    counts = []
    colors = list(palette)
    rng.shuffle(colors)
    maximum = min(len(orbits), max(desired + 4, len(palette) + 2))
    target_radius2 = span - 1
    reached = 2 if span % 2 else 1

    while len(selected) < maximum and (
        len(selected) < desired or reached < target_radius2
    ):
        mask = visible != -1
        frontier = []
        weights = []
        for orbit in orbits:
            if orbit in selected:
                continue
            cells = [_index(center2, rel) for rel in orbit]
            accessible = [cell for cell in cells if _adjacent(mask, *cell)]
            if not accessible:
                continue
            radius2 = max(max(map(abs, rel)) for rel in orbit)
            # Favor coherent growth, with a route to the requested outer ring.
            outward = 1 + 2.0 * (radius2 > reached)
            if len(selected) >= desired - 2 and reached < target_radius2:
                outward *= 3
            x, y = orbit[0]
            diagonal = min(abs(x), abs(y)) / max(abs(x), abs(y))
            if style == "arms":
                shape_weight = 1.7 - diagonal
            elif style == "corners":
                shape_weight = 0.7 + diagonal
            elif style == "steps":
                shape_weight = (
                    1.7 - diagonal if len(selected) % 2 == 0
                    else 0.7 + diagonal
                )
            else:
                shape_weight = 1.0
            frontier.append((orbit, cells, radius2))
            weights.append(
                outward * shape_weight * (0.8 + 0.2 * len(accessible))
            )
        if not frontier:
            return None
        probabilities = np.asarray(weights) / sum(weights)
        orbit, cells, radius2 = frontier[int(rng.choice(
            len(frontier), p=probabilities
        ))]
        color = colors[len(selected)] if len(selected) < len(colors) else (
            palette[int(rng.integers(len(palette)))]
        )
        for row, col in cells:
            output[row, col] = color
        wanted = int(rng.choice(4, p=(0.29, 0.29, 0.24, 0.18))) + 1
        if len(selected) < len(palette) or (
            len(selected) == maximum - 1 and not any(x < 4 for x in counts)
        ):
            wanted = min(wanted, 3)
        chosen = 0
        while chosen < wanted:
            mask = visible != -1
            options = [cell for cell in cells if visible[cell] == -1
                       and _adjacent(mask, *cell)]
            if not options:
                break
            cell = options[int(rng.integers(len(options)))]
            visible[cell] = color
            chosen += 1
        if chosen == 0:
            return None
        counts.append(chosen)
        selected.add(orbit)
        reached = max(reached, radius2)
    if reached != target_radius2 or len(selected) < len(palette):
        return None
    if not any(count < 4 for count in counts):
        return None
    if any(color not in visible for color in palette):
        return None
    return visible, output, kind


def _canvas(rng, local_input, local_output, background):
    span = local_input.shape[0]
    height = int(rng.integers(span + 2, min(30, span + 14) + 1))
    width = int(rng.integers(span + 2, min(30, span + 14) + 1))
    top = int(rng.integers(1, height - span))
    left = int(rng.integers(1, width - span))
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    region = np.s_[top:top + span, left:left + span]
    inp[region] = np.where(local_input == -1, background, local_input)
    out[region] = np.where(local_output == -1, background, local_output)
    return inp, out


def _analyze_pair(inp, out, require_unique=False):
    assert inp.shape == out.shape
    assert not np.array_equal(inp, out)
    background = _background(inp)
    assert _background(out) == background
    input_mask = inp != background
    output_mask = out != background
    assert _connected(input_mask) and _connected(output_mask)
    assert np.array_equal(inp[input_mask], out[input_mask])
    assert set(int(x) for x in np.unique(inp)) == set(
        int(x) for x in np.unique(out)
    )
    colors = set(int(x) for x in np.unique(inp)) - {background}
    assert len(colors) in (2, 3)
    occupied = np.argwhere(output_mask)
    low = occupied.min(axis=0)
    high = occupied.max(axis=0)
    center2 = tuple(int(a + b) for a, b in zip(low, high))
    candidates = _candidate_centers(inp, background)
    assert center2 in candidates
    assert np.array_equal(candidates[center2], out)
    if require_unique:
        assert len(candidates) == 1
    span = int(high[0] - low[0] + 1)
    assert span in (4, 5, 6, 7, 8, 9)
    assert int(high[1] - low[1] + 1) == span
    center_type = "half" if center2[0] % 2 else "cell"
    assert (span % 2 == 0) == (center_type == "half")
    core_radius2 = 1 if center_type == "half" else 2
    counts = {}
    orbit_colors = {}
    for row, col in np.argwhere(output_mask):
        rel = (2 * int(row) - center2[0], 2 * int(col) - center2[1])
        if max(map(abs, rel)) <= core_radius2:
            continue
        orbit = _orbit(*rel)
        color = int(out[row, col])
        assert orbit_colors.setdefault(orbit, color) == color
    for row, col in np.argwhere(input_mask):
        rel = (2 * int(row) - center2[0], 2 * int(col) - center2[1])
        if max(map(abs, rel)) > core_radius2:
            orbit = _orbit(*rel)
            counts[orbit] = counts.get(orbit, 0) + 1
    assert counts.keys() == orbit_colors.keys()
    assert all(1 <= count <= 4 for count in counts.values())
    assert any(count < 4 for count in counts.values())
    added_colors = set(int(x) for x in np.unique(out[~input_mask & output_mask]))
    return {
        "background": background,
        "colors": colors,
        "center": center2,
        "type": center_type,
        "span": span,
        "orbit_count": len(counts),
        "visible_counts": tuple(counts.values()),
        "added_colors": added_colors,
        "layout": (
            center_type,
            tuple(sorted(
                (2 * int(r) - center2[0], 2 * int(c) - center2[1])
                for r, c in np.argwhere(input_mask)
            )),
        ),
    }


def _validate_count(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    train = None if num_train is None else _validate_count(
        "num_train", num_train, 2, 10
    )
    test = None if num_test is None else _validate_count(
        "num_test", num_test, 1, 4
    )
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_span(rng, center_type, split):
    if center_type == "half":
        spans = (4, 6, 8)
        probabilities = (0.56, 0.34, 0.10) if split == "train" else (
            0.35, 0.39, 0.26
        )
    else:
        spans = (5, 7, 9)
        probabilities = (0.56, 0.34, 0.10) if split == "train" else (
            0.35, 0.39, 0.26
        )
    return int(rng.choice(spans, p=probabilities))


def _sample_desired(rng, span, ncolors, split):
    if span in (4, 5):
        low, high = ncolors, 4
    elif span in (6, 7):
        low, high = max(3, ncolors), 7
    else:
        low, high = max(5, ncolors), 11
    if split == "test" and span >= 7 and rng.random() < 0.55:
        low = max(low, (low + high) // 2)
    return int(rng.integers(low, min(high, len(_pool(span))) + 1))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    ncolors = 3 if rng.random() < 0.20 else 2
    train_types = ["half", "cell"] + [
        ("half", "cell")[int(rng.integers(2))]
        for _ in range(train_count - 2)
    ]
    rng.shuffle(train_types)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_background = None
    for split, types in (
        ("train", train_types),
        ("test", [
            ("half", "cell")[int(rng.integers(2))]
            for _ in range(test_count)
        ]),
    ):
        for center_type in types:
            for _ in range(300):
                background, palette = _pair_colors(
                    rng, previous_background, ncolors
                )
                span = _sample_span(rng, center_type, split)
                desired = _sample_desired(rng, span, ncolors, split)
                local = _make_local(rng, span, palette, desired)
                if local is None:
                    continue
                inp, out = _canvas(rng, local[0], local[1], background)
                if _grid_key(inp) in seen:
                    continue
                if _text_key(inp, out) in _SOURCE_PAIRS:
                    continue
                try:
                    facts = _analyze_pair(
                        inp, out, require_unique=(split == "test")
                    )
                except AssertionError:
                    continue
                if facts["type"] != center_type or facts["span"] != span:
                    continue
                if split == "train" and not facts["added_colors"] == set(palette):
                    continue
                puzzle[split].append({"input": inp, "output": out})
                seen.add(_grid_key(inp))
                previous_background = background
                break
            else:
                raise ValueError("could not construct a unique connected pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert _text_key(inp, out) not in _SOURCE_PAIRS
            analyses[split].append(_analyze_pair(
                inp, out, require_unique=(split == "test")
            ))
    assert {x["type"] for x in analyses["train"]} == {"half", "cell"}
    tiers = {len(x["colors"]) for split in analyses.values() for x in split}
    assert len(tiers) == 1
    for facts in analyses["train"]:
        assert facts["added_colors"] == facts["colors"]
    # Pair palettes change; training establishes color-independent copying.
    assert all(x["added_colors"] for x in analyses["test"])


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a["input"], b["input"])
            and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    automatic = generate_puzzle(12345)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(67890, 7, 3)
    assert (len(explicit["train"]), len(explicit["test"])) == (7, 3)
    assert _equal(generate_puzzle(314159, 6, 2),
                  generate_puzzle(314159, 6, 2))
    assert not _equal(generate_puzzle(314159, 6, 2),
                      generate_puzzle(314160, 6, 2))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(8000 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, (train, test) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(9000 + index, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(77, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    train_counts, test_counts = set(), set()
    backgrounds, spans, layouts = set(), set(), set()
    test_types, orbit_members = set(), set()
    foreground_zero = 0
    complete_orbit_pairs = 0
    smaller_tests = 0
    three_color_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_facts = [_analyze_pair(x["input"], x["output"])
                       for x in puzzle["train"]]
        test_facts = [_analyze_pair(
                        x["input"], x["output"], require_unique=True)
                      for x in puzzle["test"]]
        assert len({x["background"] for x in train_facts}) >= 2
        three_color_puzzles += len(train_facts[0]["colors"]) == 3
        smaller_tests += any(
            x["span"] <= max(t["span"] for t in train_facts)
            and x["orbit_count"] <= max(t["orbit_count"] for t in train_facts)
            for x in test_facts
        )
        for split, facts_list in (("train", train_facts),
                                  ("test", test_facts)):
            for facts in facts_list:
                backgrounds.add(facts["background"])
                spans.add(facts["span"])
                layouts.add(facts["layout"])
                orbit_members.update(facts["visible_counts"])
                foreground_zero += 0 in facts["colors"]
                complete_orbit_pairs += 4 in facts["visible_counts"]
                if split == "test":
                    test_types.add(facts["type"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 8 and len(spans) == 6
    assert len(layouts) >= 100
    assert test_types == {"half", "cell"}
    assert orbit_members == {1, 2, 3, 4}
    assert foreground_zero >= 10
    assert complete_orbit_pairs >= 10
    assert smaller_tests >= 8
    assert three_color_puzzles >= 2
