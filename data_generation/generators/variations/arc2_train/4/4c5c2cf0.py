"""Generate X-marker reflection puzzles."""

import numpy as np


SIGNS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
X_OFFSETS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))

# Exact input signatures from the four original pairs.
SOURCE_INPUTS = (
    ((14, 14), ((3, 8, 8), (4, 7, 8), (4, 8, 8), (4, 9, 8),
                (5, 6, 8), (5, 7, 8), (5, 8, 8), (6, 4, 3),
                (6, 6, 3), (7, 5, 3), (8, 4, 3), (8, 6, 3))),
    ((14, 14), ((3, 3, 2), (3, 6, 2), (4, 3, 2), (4, 4, 2),
                (4, 6, 2), (5, 5, 2), (5, 6, 2), (6, 4, 2),
                (6, 5, 2), (7, 3, 2), (7, 6, 4), (7, 8, 4),
                (8, 7, 4), (9, 6, 4), (9, 8, 4))),
    ((12, 12), ((6, 4, 8), (6, 6, 8), (7, 5, 8), (8, 4, 8),
                (8, 6, 8), (9, 2, 1), (9, 3, 1), (10, 1, 1),
                (10, 3, 1), (11, 2, 1))),
    ((15, 15), ((6, 5, 7), (6, 7, 7), (7, 6, 7), (8, 5, 7),
                (8, 7, 7), (8, 9, 4), (9, 7, 4), (9, 8, 4),
                (9, 10, 4), (10, 8, 4), (11, 8, 4), (11, 9, 4),
                (12, 7, 4), (13, 7, 4))),
)


def _checked_int(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} outside {minimum}..{maximum}")
    return value


def _counts(seed, num_train, num_test):
    train = _checked_int("num_train", num_train, 2, 10)
    test = _checked_int("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (tuple(grid.shape), grid.tobytes())


def _source_key(inp):
    if int(inp[0, 0]) != 0:
        return None
    return (tuple(inp.shape), tuple((int(r), int(c), int(inp[r, c]))
                                     for r, c in np.argwhere(inp != 0)))


def _x_candidates(inp):
    found = []
    for color in np.unique(inp):
        cells = np.argwhere(inp == color)
        if len(cells) != 5:
            continue
        locations = {(int(r), int(c)) for r, c in cells}
        for r, c in locations:
            if locations == {(r + dr, c + dc) for dr, dc in X_OFFSETS}:
                found.append((int(color), r, c))
                break
    return found


def _dimensions(rng, regime):
    kind = int(rng.integers(0, 5))
    if kind == 0:
        height, width = int(rng.integers(6, 12)), int(rng.integers(6, 15))
    elif kind == 1:
        height, width = int(rng.integers(11, 19)), int(rng.integers(11, 20))
    elif kind == 2:
        height, width = int(rng.integers(17, 31)), int(rng.integers(16, 31))
    elif kind == 3:
        height = width = int(rng.integers(8, 31))
    else:
        height, width = int(rng.integers(6, 31)), int(rng.integers(6, 31))
    if regime == "target_test" and rng.random() < 0.25:
        height, width = int(rng.integers(19, 31)), int(rng.integers(19, 31))
    return height, width


def _informative(cells):
    if len(cells) < 3 or len({r for r, _ in cells}) < 2 or len({c for _, c in cells}) < 2:
        return False
    if cells == {(c, r) for r, c in cells}:
        return False
    low_r, high_r = min(r for r, _ in cells), max(r for r, _ in cells)
    low_c, high_c = min(c for _, c in cells), max(c for _, c in cells)
    return (cells != {(low_r + high_r - r, c) for r, c in cells}
            and cells != {(r, low_c + high_c - c) for r, c in cells})


def _motif(rng, row_limit, col_limit, need_informative, multicolor):
    required = 3 if need_informative else 2 if multicolor else 1
    for _ in range(100):
        reach = int(rng.integers(0, 4))
        if reach == 0:
            span_r = int(rng.integers(1, min(3, row_limit) + 1))
            span_c = int(rng.integers(1, min(3, col_limit) + 1))
        elif reach in (1, 2):
            span_r = int(rng.integers(min(3, row_limit), min(6, row_limit) + 1))
            span_c = int(rng.integers(min(3, col_limit), min(6, col_limit) + 1))
        else:
            span_r = int(rng.integers(min(4, row_limit), min(10, row_limit) + 1))
            span_c = int(rng.integers(min(4, col_limit), min(10, col_limit) + 1))
        low_r = (1 if rng.random() < 0.5 else
                 int(rng.integers(1, row_limit - span_r + 2)))
        low_c = (1 if rng.random() < 0.5 else
                 int(rng.integers(1, col_limit - span_c + 2)))
        high_r = low_r + span_r - 1
        high_c = low_c + span_c - 1
        universe = [(r, c) for r in range(low_r, high_r + 1)
                    for c in range(low_c, high_c + 1) if (r, c) != (1, 1)]
        if len(universe) < required:
            continue
        mode = int(rng.integers(0, 6))
        if mode < 3:
            probability = (float(rng.uniform(0.12, 0.42)) if mode == 0 else
                           float(rng.uniform(0.35, 0.68)) if mode == 1 else
                           float(rng.uniform(0.66, 0.94)))
            cells = {cell for cell in universe if rng.random() < probability}
        elif mode < 5:
            target = int(rng.integers(min(4, len(universe)), min(len(universe), 45) + 1))
            cells = {universe[int(rng.integers(len(universe)))]}
            available = set(universe)
            while len(cells) < target:
                frontier = sorted((r, c) for r, c in available - cells
                                  if any(max(abs(r - a), abs(c - b)) == 1 and
                                         (mode == 4 or abs(r - a) + abs(c - b) == 1)
                                         for a, b in cells))
                if not frontier:
                    break
                cells.add(frontier[int(rng.integers(len(frontier)))])
        else:
            row_line = bool(rng.integers(0, 2))
            line = [cell for cell in universe if
                    (cell[0] == (low_r + high_r) // 2 if row_line else
                     cell[1] == (low_c + high_c) // 2)]
            cells = {cell for cell in line if rng.random() < 0.8}
        if len(cells) >= required and (not need_informative or _informative(cells)):
            return cells
    # Bounded fallback for a small canvas.
    cells = {(r, c) for r in range(1, row_limit + 1)
             for c in range(1, col_limit + 1) if (r, c) != (1, 1)}
    if need_informative:
        cells = {(1, 2), (2, 2), (3, 3)}
    return cells


def _color_plans(rng, total):
    palette_mode = int(rng.integers(0, 4))
    background_mode = int(rng.integers(0, 3))
    base = list(map(int, rng.permutation(10)))
    plans = []
    for index in range(total):
        background = (base[0] if palette_mode == 0 or background_mode == 0 or
                      (background_mode == 1 and index % 3) else
                      int(rng.integers(0, 10)))
        choices = [c for c in range(10) if c != background]
        if palette_mode == 0:
            marker, payload = base[1:3]
        elif palette_mode == 1:
            marker, payload = (base[1], base[2]) if index % 2 == 0 else (base[2], base[1])
        elif palette_mode == 2:
            marker, payload = tuple(map(int, rng.choice(choices, 2, replace=False)))
        else:
            local = [c for c in base[1:6] if c != background]
            marker, payload = tuple(map(int, rng.choice(local, 2, replace=False)))
        if background in (marker, payload):
            marker, payload = tuple(map(int, rng.choice(choices, 2, replace=False)))
        plans.append((background, marker, payload))
    return plans


def _quadrants(rng, train_count, test_count):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        held = SIGNS[int(rng.integers(0, 4))]
        allowed = [q for q in SIGNS if q != held]
        train = [allowed[i % 3] for i in range(train_count)]
        rng.shuffle(train)
        test = [held] * test_count
    elif mode == 1:
        train = [SIGNS[int(rng.integers(0, 4))] for _ in range(train_count)]
        if train_count >= 3 and len(set(train)) == 1:
            train[1] = SIGNS[(SIGNS.index(train[0]) + 1) % 4]
        test = [SIGNS[int(rng.integers(0, 4))] for _ in range(test_count)]
    elif mode == 2:
        train = [SIGNS[int(rng.integers(0, 4))] for _ in range(train_count)]
        test = [train[int(rng.integers(train_count))] for _ in range(test_count)]
    else:
        common = SIGNS[int(rng.integers(0, 4))]
        train = [common] * train_count
        test = [common if rng.random() < 0.7 else
                SIGNS[int(rng.integers(0, 4))] for _ in range(test_count)]
    return train + test


def _pair(rng, regime, quadrant, colors, need_informative, multicolor):
    height, width = _dimensions(rng, regime)
    if need_informative:
        height, width = max(height, 8), max(width, 8)
    motif = _motif(rng, (height - 1) // 2, (width - 1) // 2,
                   need_informative, multicolor)
    reach_r = max(r for r, _ in motif)
    reach_c = max(c for _, c in motif)
    center_r = int(rng.integers(reach_r, height - reach_r))
    center_c = int(rng.integers(reach_c, width - reach_c))
    background, marker, payload = colors
    inp = np.full((height, width), background, dtype=np.int8)
    for dr, dc in X_OFFSETS:
        inp[center_r + dr, center_c + dc] = marker
    if multicolor:
        extras = [c for c in range(10) if c not in colors]
        extra_count = 1 if len(motif) == 2 else int(rng.integers(1, 3))
        palette = [payload] + list(map(int, rng.choice(extras, extra_count, replace=False)))
    else:
        palette = [payload]
    assigned = [palette[int(rng.integers(len(palette)))] for _ in motif]
    if multicolor:
        for index, color in enumerate(palette):
            assigned[index] = color
        rng.shuffle(assigned)
    out = inp.copy()
    source_r, source_c = quadrant
    for (dr, dc), color in zip(sorted(motif), assigned):
        inp[center_r + source_r * dr, center_c + source_c * dc] = color
        for target_r, target_c in SIGNS:
            out[center_r + target_r * dr, center_c + target_c * dc] = color
    return inp, out


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _checked_int("seed", seed, 0, 2**63 - 1)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    total = train_count + test_count
    colors = _color_plans(rng, total)
    quadrants = _quadrants(rng, train_count, test_count)
    multi_family = rng.random() < 0.22
    required_multi_train = int(rng.integers(train_count)) if multi_family else -1
    pair_seeds = rng.integers(0, 2**63 - 1, size=total)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(total):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        regime = "source_train" if index < train_count else "target_test"
        multicolor = (index == required_multi_train or
                      (multi_family and index != 0 and pair_rng.random() < 0.35))
        for _ in range(100):
            inp, out = _pair(pair_rng, regime, quadrants[index], colors[index],
                             index == 0, multicolor)
            key = _grid_key(inp)
            if (key not in seen and _source_key(inp) not in SOURCE_INPUTS
                    and len(_x_candidates(inp)) == 1):
                break
        else:
            raise RuntimeError("could not produce a unique scene")
        seen.add(key)
        puzzle["train" if index < train_count else "test"].append(
            {"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    candidates = _x_candidates(inp)
    assert len(candidates) == 1
    marker, center_r, center_c = candidates[0]
    assert marker != background
    payload = {(int(r), int(c)): int(inp[r, c]) for r, c in np.argwhere(
        (inp != background) & (inp != marker))}
    assert payload
    signs, motif = set(), set()
    expected = inp.copy()
    for (r, c), color in payload.items():
        assert r != center_r and c != center_c
        signs.add((-1 if r < center_r else 1, -1 if c < center_c else 1))
        dr, dc = abs(r - center_r), abs(c - center_c)
        motif.add((dr, dc))
        assert (dr, dc) != (1, 1)
        for sr, sc in SIGNS:
            rr, cc = center_r + sr * dr, center_c + sc * dc
            assert 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
            assert int(inp[rr, cc]) in (background, color)
            expected[rr, cc] = color
    assert len(signs) == 1 and len(motif) == len(payload)
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    assert set(map(int, np.unique(out))) == set(map(int, colors))
    assert int(np.count_nonzero(out != background)) == 5 + 4 * len(payload)
    assert _source_key(inp) not in SOURCE_INPUTS
    return {"background": background, "marker": marker,
            "payload_colors": {c for c in payload.values()},
            "quadrant": next(iter(signs)), "shape": inp.shape,
            "center": (center_r, center_c), "motif": motif,
            "count": len(payload), "multi": len(set(payload.values())) > 1}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    train_info, test_info = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key, pair_key = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(inp, out)
            (train_info if split == "train" else test_info).append(info)
    assert any(_informative(item["motif"]) for item in train_info)
    if any(item["multi"] for item in test_info):
        assert any(item["multi"] for item in train_info)


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _predict_from_examples(train_pairs, test_input):
    # Check that the demonstrations teach the same operation, then predict
    # using only the new input. No test label or generation state is read.
    def complete(inp):
        colors, counts = np.unique(inp, return_counts=True)
        background = int(colors[int(np.argmax(counts))])
        candidates = _x_candidates(inp)
        assert len(candidates) == 1
        marker, axis_r, axis_c = candidates[0]
        result = inp.copy()
        for r, c in np.argwhere((inp != background) & (inp != marker)):
            r, c = int(r), int(c)
            color = int(inp[r, c])
            for rr in (r, 2 * axis_r - r):
                for cc in (c, 2 * axis_c - c):
                    result[rr, cc] = color
        return result

    for pair in train_pairs:
        assert np.array_equal(complete(pair["input"]), pair["output"])
    return complete(test_input)


def _component_count(cells, diagonal):
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            r, c = frontier.pop()
            adjacent = {point for point in remaining
                        if max(abs(point[0] - r), abs(point[1] - c)) == 1
                        and (diagonal or abs(point[0] - r) + abs(point[1] - c) == 1)}
            remaining.difference_update(adjacent)
            frontier.extend(adjacent)
    return count


def self_test() -> None:
    train_counts, test_counts, shapes, backgrounds = set(), set(), set(), set()
    quadrants, motif_counts, layouts, palettes = set(), set(), set(), set()
    puzzle_keys = set()
    small = large = rectangle = corner_free = distant = multi = 0
    review_hits = {name: 0 for name in (
        "five", "below_five", "above_eighteen", "symmetric", "thin",
        "connected_four", "connected_eight_only", "disconnected_eight",
        "background_zero", "odd_train", "side_thirty", "held_out",
        "familiar", "mixed_test", "low_quadrant_variation", "variable_background",
        "stable_palette", "role_exchange", "novel_test_color",
        "test_smaller", "test_larger", "multi_train", "multi_test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        copied_train = [{role: pair[role].copy() for role in ("input", "output")}
                        for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_examples(copied_train, pair["input"].copy()),
                                  pair["output"])
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        by_split = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair["input"], pair["output"])
                by_split[split].append(info)
                shapes.add(info["shape"])
                backgrounds.add(info["background"])
                quadrants.add(info["quadrant"])
                motif_counts.add(info["count"])
                palettes.add((info["background"], info["marker"],
                              tuple(sorted(info["payload_colors"]))))
                layouts.add((info["shape"], info["center"], info["quadrant"],
                             tuple(sorted(info["motif"]))))
                h, w = info["shape"]
                small += min(h, w) <= 10
                large += max(h, w) >= 25
                rectangle += h != w
                mr = max(r for r, _ in info["motif"])
                mc = max(c for _, c in info["motif"])
                corner_free += (mr, mc) not in info["motif"]
                distant += (min(r for r, _ in info["motif"]) >= 2 and
                            min(c for _, c in info["motif"]) >= 2)
                multi += info["multi"]
                review_hits["five"] += info["count"] == 5
                review_hits["below_five"] += info["count"] < 5
                review_hits["above_eighteen"] += info["count"] > 18
                review_hits["symmetric"] += info["motif"] == {(c, r) for r, c in info["motif"]}
                review_hits["thin"] += len({r for r, c in info["motif"]}) == 1 or len({c for r, c in info["motif"]}) == 1
                four = _component_count(info["motif"], False)
                eight = _component_count(info["motif"], True)
                review_hits["connected_four"] += four == 1
                review_hits["connected_eight_only"] += four > 1 and eight == 1
                review_hits["disconnected_eight"] += eight > 1
                review_hits["background_zero"] += info["background"] == 0
                review_hits["odd_train"] += split == "train" and (h % 2 == 1 or w % 2 == 1)
                review_hits["side_thirty"] += h == 30 or w == 30
        train_info, test_info = by_split["train"], by_split["test"]
        train_quadrants = {item["quadrant"] for item in train_info}
        test_quadrants = {item["quadrant"] for item in test_info}
        all_info = train_info + test_info
        review_hits["held_out"] += bool(test_quadrants - train_quadrants)
        review_hits["familiar"] += bool(test_quadrants & train_quadrants)
        review_hits["mixed_test"] += len(test_quadrants) > 1
        review_hits["low_quadrant_variation"] += len(train_quadrants | test_quadrants) == 1
        review_hits["variable_background"] += len({item["background"] for item in all_info}) > 1
        review_hits["stable_palette"] += len({(item["background"], item["marker"],
                                               tuple(sorted(item["payload_colors"])))
                                              for item in all_info}) == 1
        review_hits["role_exchange"] += bool({item["marker"] for item in all_info} &
                                             set().union(*(item["payload_colors"] for item in all_info)))
        known = set().union(*(item["payload_colors"] | {item["marker"]} for item in train_info))
        review_hits["novel_test_color"] += any((item["payload_colors"] | {item["marker"]}) - known
                                                for item in test_info)
        review_hits["test_smaller"] += min(min(item["shape"]) for item in test_info) < max(max(item["shape"]) for item in train_info)
        review_hits["test_larger"] += max(max(item["shape"]) for item in test_info) > max(max(item["shape"]) for item in train_info)
        review_hits["multi_train"] += any(item["multi"] for item in train_info)
        review_hits["multi_test"] += any(item["multi"] for item in test_info)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(layouts) >= 150
    assert len(shapes) >= 35 and len(backgrounds) >= 8
    assert len(quadrants) == 4 and len(motif_counts) >= 12
    assert len(palettes) >= 60
    assert min(small, large, rectangle, corner_free, distant, multi) > 0
    assert all(value > 0 for value in review_hits.values())
    for tr, te in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(100 + tr + te, tr, te)
        assert (len(p["train"]), len(p["test"])) == (tr, te)
        validate_generated_puzzle(p)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(200 + (tr or 0) + (te or 0), tr, te)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    assert _puzzle_key(generate_puzzle(777, 7, 3)) == _puzzle_key(generate_puzzle(777, 7, 3))
    assert len({_puzzle_key(generate_puzzle(s, 5, 2)) for s in range(300, 312)}) == 12
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (5, False)):
        try:
            generate_puzzle(999, *args)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {args}")
