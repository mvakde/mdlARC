"""Marker-guided nested rectangles for the reviewed ARC puzzle family."""

import numpy as np


# Exact source inputs, described by visible geometry and literal colors.
_SOURCE_INPUT_FINGERPRINTS = frozenset({
    ((15, 13), 0, 5, 3, (5, 7, 4, 6), ((1, 5), (5, 1), (8, 10), (12, 5))),
    ((12, 13), 0, 5, 1, (4, 7, 5, 7), ((1, 4), (6, 1), (6, 11), (11, 4))),
    ((15, 14), 0, 5, 4, (5, 8, 4, 7), ((2, 6), (6, 12), (8, 1), (13, 7))),
    ((15, 15), 0, 5, 8, (5, 7, 5, 8), ((1, 7), (6, 2), (9, 12), (13, 5))),
})


def _validated_optional_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _validated_optional_count(num_train, "num_train", 2, 10)
    test = _validated_optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choice_color(rng, excluded):
    choices = [c for c in range(10) if c not in excluded]
    return int(choices[int(rng.integers(len(choices)))])


def _colors(rng, mode, anchor, first_structure, train_structures, split, index, reuse):
    """Semantic roles stay fixed; this puzzle's palette mode controls literal changes."""
    if anchor is None:
        perm = rng.permutation(10)
        return int(perm[0]), int(perm[1]), int(perm[2])
    anchor_bg, anchor_marker = anchor
    if mode in (0, 2):
        bg = anchor_bg
    else:
        excluded = {anchor_bg} if split == "train" and index == 1 else set()
        if mode == 1:
            excluded.add(anchor_marker)
        bg = _choice_color(rng, excluded)
    if mode in (0, 1):
        marker = anchor_marker
    else:
        excluded = {bg}
        if split == "train" and index == 1:
            excluded.add(anchor_marker)
        marker = _choice_color(rng, excluded)
    forbidden = {bg, marker}
    if split == "train":
        if index == 1:
            forbidden.add(first_structure)
        structure = _choice_color(rng, forbidden)
    else:
        candidates = (set(train_structures) if reuse
                      else set(range(10)) - set(train_structures))
        pool = sorted(candidates - forbidden)
        if not pool:
            pool = [c for c in range(10) if c not in forbidden]
        structure = int(pool[int(rng.integers(len(pool)))])
    return bg, marker, structure


def _sample_seed_size(rng, category, larger=False):
    if category == "square":
        side = int(rng.integers(3, 9 if larger else 8))
        return side, side
    short = int(rng.integers(3, 8 if larger else 7))
    long = short + int(rng.integers(1, 6 if larger else 5))
    return (short, long) if category == "wide" else (long, short)


def _sample_outer_size(rng, seed_size, larger=False):
    cap = 10 if larger else 8
    return tuple(side + 4 + int(rng.integers(0, min(cap, 24 - side) + 1))
                 for side in seed_size)


def _paint_perimeter(grid, top, bottom, left, right, color):
    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def _draw_pair(rng, colors, seed_size, outer_size, alignment):
    background, marker, structure = colors
    seed_h, seed_w = seed_size
    outer_h, outer_w = outer_size
    assert outer_h >= seed_h + 4 and outer_w >= seed_w + 4
    assert outer_h <= 28 and outer_w <= 28
    height = outer_h + 2 + int(rng.integers(0, min(6, 28 - outer_h) + 1))
    width = outer_w + 2 + int(rng.integers(0, min(6, 28 - outer_w) + 1))
    outer_top = 1 + int(rng.integers(0, height - outer_h - 1))
    outer_left = 1 + int(rng.integers(0, width - outer_w - 1))
    outer_bottom = outer_top + outer_h - 1
    outer_right = outer_left + outer_w - 1
    # Split the free clearance independently along rows and columns.
    seed_top = outer_top + 2 + int(rng.integers(0, outer_h - seed_h - 3))
    seed_left = outer_left + 2 + int(rng.integers(0, outer_w - seed_w - 3))
    seed_bottom = seed_top + seed_h - 1
    seed_right = seed_left + seed_w - 1

    cols = list(range(outer_left + 1, outer_right))
    rows = list(range(outer_top + 1, outer_bottom))
    top_col = int(rng.choice(cols))
    left_row = int(rng.choice(rows))
    if alignment == "both":
        bottom_col, right_row = top_col, left_row
    elif alignment == "one":
        if int(rng.integers(2)) == 0:
            bottom_col = top_col
            right_row = int(rng.choice([r for r in rows if r != left_row]))
        else:
            bottom_col = int(rng.choice([c for c in cols if c != top_col]))
            right_row = left_row
    elif alignment == "none":
        bottom_col = int(rng.choice([c for c in cols if c != top_col]))
        right_row = int(rng.choice([r for r in rows if r != left_row]))
    else:
        raise ValueError("unknown marker alignment")

    inp = np.full((height, width), background, dtype=np.int8)
    _paint_perimeter(inp, seed_top, seed_bottom, seed_left, seed_right, structure)
    for row, col in ((outer_top - 1, top_col), (outer_bottom + 1, bottom_col),
                     (left_row, outer_left - 1), (right_row, outer_right + 1)):
        inp[row, col] = marker
    out = inp.copy()
    _paint_perimeter(out, outer_top, outer_bottom, outer_left, outer_right, structure)
    return {"input": inp, "output": out}


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= extent <= 30 for extent in grid.shape)
    assert grid.dtype.kind in ("i", "u")
    assert 0 <= int(np.min(grid)) <= int(np.max(grid)) <= 9


def _array_key(grid):
    return grid.shape, grid.tobytes()


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    _assert_grid(inp)
    _assert_grid(out)
    assert inp.shape == out.shape and inp.dtype == out.dtype
    assert not np.array_equal(inp, out)
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 3
    assert np.count_nonzero(counts == np.max(counts)) == 1
    background = int(values[int(np.argmax(counts))])
    others = [int(v) for v in values if int(v) != background]
    marker_candidates = [v for v in others if np.count_nonzero(inp == v) == 4]
    assert len(marker_candidates) == 1
    marker = marker_candidates[0]
    structure = next(v for v in others if v != marker)

    seed_cells = np.argwhere(inp == structure)
    st, sb = int(seed_cells[:, 0].min()), int(seed_cells[:, 0].max())
    sl, sr = int(seed_cells[:, 1].min()), int(seed_cells[:, 1].max())
    seed_size = sb - st + 1, sr - sl + 1
    assert min(seed_size) >= 3
    expected_seed = np.zeros(inp.shape, dtype=bool)
    _paint_perimeter(expected_seed, st, sb, sl, sr, True)
    assert np.array_equal(inp == structure, expected_seed)

    markers = np.argwhere(inp == marker)
    assert markers.shape == (4, 2)
    for row, col in markers:
        neighborhood = inp[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        assert np.count_nonzero(neighborhood != background) == 1
    top_i = np.flatnonzero(markers[:, 0] == markers[:, 0].min())
    bottom_i = np.flatnonzero(markers[:, 0] == markers[:, 0].max())
    left_i = np.flatnonzero(markers[:, 1] == markers[:, 1].min())
    right_i = np.flatnonzero(markers[:, 1] == markers[:, 1].max())
    assert all(len(x) == 1 for x in (top_i, bottom_i, left_i, right_i))
    ti, bi, li, ri = (int(x[0]) for x in (top_i, bottom_i, left_i, right_i))
    assert len({ti, bi, li, ri}) == 4
    ot, ob = int(markers[ti, 0] + 1), int(markers[bi, 0] - 1)
    ol, oright = int(markers[li, 1] + 1), int(markers[ri, 1] - 1)
    height, width = inp.shape
    assert 0 < ot < ob < height - 1 and 0 < ol < oright < width - 1
    assert ol < markers[ti, 1] < oright and ol < markers[bi, 1] < oright
    assert ot < markers[li, 0] < ob and ot < markers[ri, 0] < ob
    gaps = st - ot - 1, ob - sb - 1, sl - ol - 1, oright - sr - 1
    assert min(gaps) >= 1
    outer_mask = np.zeros(inp.shape, dtype=bool)
    _paint_perimeter(outer_mask, ot, ob, ol, oright, True)
    assert np.all(inp[outer_mask] == background)
    expected = inp.copy()
    _paint_perimeter(expected, ot, ob, ol, oright, structure)
    assert np.array_equal(out, expected)
    assert set(int(v) for v in np.unique(out)) == {background, marker, structure}
    vertical_aligned = markers[ti, 1] == markers[bi, 1]
    horizontal_aligned = markers[li, 0] == markers[ri, 0]
    alignment = ("both" if vertical_aligned and horizontal_aligned else
                 "one" if vertical_aligned or horizontal_aligned else "none")
    return {"shape": inp.shape, "background": background, "marker": marker,
            "structure": structure, "seed_box": (st, sb, sl, sr),
            "seed_size": seed_size, "outer_box": (ot, ob, ol, oright),
            "outer_size": (ob - ot + 1, oright - ol + 1), "gaps": gaps,
            "alignment": alignment,
            "markers": tuple(sorted((int(r), int(c)) for r, c in markers))}


def _fingerprint(meta):
    return (meta["shape"], meta["background"], meta["marker"],
            meta["structure"], meta["seed_box"], meta["markers"])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen_inputs, seen_pairs = set(), set()
    train, test = [], []
    for split, collected in (("train", train), ("test", test)):
        for pair in puzzle[split]:
            meta = _analyze_pair(pair)
            collected.append(meta)
            inp_key, out_key = _array_key(pair["input"]), _array_key(pair["output"])
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            assert _fingerprint(meta) not in _SOURCE_INPUT_FINGERPRINTS
    assert len({m["seed_size"] for m in train}) >= 2
    assert len({m["alignment"] for m in train}) >= 2
    assert len({m["structure"] for m in train}) >= 2
    train_backgrounds = {m["background"] for m in train}
    train_markers = {m["marker"] for m in train}
    for meta in test:
        if meta["background"] not in train_backgrounds:
            assert len(train_backgrounds) >= 2
        if meta["marker"] not in train_markers:
            assert len(train_markers) >= 2
        assert min(meta["gaps"]) >= 1


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette_mode = int(rng.integers(4))
    categories = ["square", "wide", "tall"]
    alignments = ["both", "one", "none"]
    rng.shuffle(categories)
    rng.shuffle(alignments)
    train, test = [], []
    train_meta = []
    seen_inputs = set()
    anchor = None
    first_structure = None
    train_structures = []

    def add_pair(destination, colors, seed_size, outer_size, alignment):
        for _ in range(24):
            pair = _draw_pair(rng, colors, seed_size, outer_size, alignment)
            meta = _analyze_pair(pair)
            key = _array_key(pair["input"])
            if key not in seen_inputs and _fingerprint(meta) not in _SOURCE_INPUT_FINGERPRINTS:
                seen_inputs.add(key)
                destination.append(pair)
                return meta
        raise AssertionError("unable to construct a unique pair")

    for index in range(train_count):
        category = categories[index % 3] if index < 3 else categories[int(rng.integers(3))]
        alignment = alignments[index % 3] if index < 3 else alignments[int(rng.integers(3))]
        seed_size = _sample_seed_size(rng, category)
        outer_size = _sample_outer_size(rng, seed_size)
        colors = _colors(rng, palette_mode, anchor, first_structure,
                         train_structures, "train", index, False)
        if anchor is None:
            anchor = colors[:2]
            first_structure = colors[2]
        meta = add_pair(train, colors, seed_size, outer_size, alignment)
        train_meta.append(meta)
        train_structures.append(colors[2])

    first_reuse = bool(int(rng.integers(2)))
    for index in range(test_count):
        strategy = 3 if index == 0 and seed % 3 == 0 else int(rng.integers(3))
        if strategy == 0:
            familiar = train_meta[int(rng.integers(train_count))]
            seed_size, outer_size = familiar["seed_size"], familiar["outer_size"]
        elif strategy == 1:
            familiar = train_meta[int(rng.integers(train_count))]
            seed_size = familiar["seed_size"]
            outer_size = _sample_outer_size(rng, seed_size, larger=True)
        elif strategy == 2:
            category = categories[int(rng.integers(3))]
            seed_size = _sample_seed_size(rng, category, larger=True)
            outer_size = _sample_outer_size(rng, seed_size, larger=True)
        if strategy == 3:
            # Combine scale, unequal aspect ratio, offset markers, and a new
            # structural color in a bounded subset of tests.
            category = "wide" if int(rng.integers(2)) == 0 else "tall"
            seed_size = _sample_seed_size(rng, category, larger=True)
            outer_size = (
                max(seed_size[0] + 4, max(m["outer_size"][0] for m in train_meta) + 1),
                max(seed_size[1] + 4, max(m["outer_size"][1] for m in train_meta) + 1),
            )
        alignment = "none" if strategy == 3 else alignments[int(rng.integers(3))]
        reuse = False if strategy == 3 else (
            first_reuse if index % 2 == 0 else not first_reuse
        )
        colors = _colors(rng, palette_mode, anchor, first_structure,
                         train_structures, "test", index, reuse)
        if strategy == 3:
            bg, marker, structure = colors
            if palette_mode in (1, 3):
                unused = set(range(10)) - {m["background"] for m in train_meta}
                unused -= {marker, structure}
                if unused:
                    bg = _choice_color(rng, set(range(10)) - unused)
            if palette_mode in (2, 3):
                unused = set(range(10)) - {m["marker"] for m in train_meta}
                unused -= {bg, structure}
                if unused:
                    marker = _choice_color(rng, set(range(10)) - unused)
            colors = bg, marker, structure
        add_pair(test, colors, seed_size, outer_size, alignment)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, _array_key(pair["input"]), _array_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(731, 7, 3)) == _puzzle_key(generate_puzzle(731, 7, 3))

    checked = list(puzzles)
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        checked.append(puzzle)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1100 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        checked.append(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(2000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {args}")

    canvas_sizes, seed_sizes, outer_sizes = set(), set(), set()
    test_canvas_sizes, test_seed_sizes, test_outer_sizes = set(), set(), set()
    gaps, styles, test_styles, palettes = set(), set(), set(), set()
    train_role_changes = test_color_reuse = test_color_novel = 0
    train_bg_changes = train_marker_changes = varied_test_colors = 0
    familiar_tests = new_size_tests = larger_tests = changed_role_tests = combined_tests = 0
    diverse_train_canvas = diverse_train_outer = 0
    for puzzle in checked:
        validate_generated_puzzle(puzzle)
        training = [_analyze_pair(p) for p in puzzle["train"]]
        testing = [_analyze_pair(p) for p in puzzle["test"]]
        train_colors = {m["structure"] for m in training}
        train_role_changes += len({(m["background"], m["marker"]) for m in training}) > 1
        train_bg_changes += len({m["background"] for m in training}) > 1
        train_marker_changes += len({m["marker"] for m in training}) > 1
        diverse_train_canvas += len({m["shape"] for m in training}) > 1
        diverse_train_outer += len({m["outer_size"] for m in training}) > 1
        varied_test_colors += len(testing) > 1 and len({m["structure"] for m in testing}) > 1
        for meta in testing:
            test_color_reuse += meta["structure"] in train_colors
            test_color_novel += meta["structure"] not in train_colors
            test_canvas_sizes.add(meta["shape"])
            test_seed_sizes.add(meta["seed_size"])
            test_outer_sizes.add(meta["outer_size"])
            test_styles.add(meta["alignment"])
            familiar = (meta["seed_size"], meta["outer_size"]) in {
                (m["seed_size"], m["outer_size"]) for m in training
            }
            larger = (meta["outer_size"][0] * meta["outer_size"][1] >
                      max(m["outer_size"][0] * m["outer_size"][1] for m in training))
            changed_role = (
                meta["background"] not in {m["background"] for m in training}
                or meta["marker"] not in {m["marker"] for m in training}
            )
            familiar_tests += familiar
            new_size_tests += not familiar
            larger_tests += larger
            changed_role_tests += changed_role
            combined_tests += (
                larger and changed_role and meta["structure"] not in train_colors
                and meta["alignment"] == "none"
                and meta["seed_size"][0] != meta["seed_size"][1]
            )
        for meta in training + testing:
            canvas_sizes.add(meta["shape"])
            seed_sizes.add(meta["seed_size"])
            outer_sizes.add(meta["outer_size"])
            gaps.add(meta["gaps"])
            styles.add(meta["alignment"])
            palettes.add((meta["background"], meta["marker"], meta["structure"]))
    assert len(canvas_sizes) >= 45 and len(seed_sizes) >= 30
    assert len(outer_sizes) >= 55 and len(gaps) >= 75
    assert styles == {"both", "one", "none"} and len(palettes) >= 90
    assert train_role_changes >= 15
    assert test_color_reuse >= 15 and test_color_novel >= 15
    assert len(test_canvas_sizes) >= 35 and len(test_seed_sizes) >= 20
    assert len(test_outer_sizes) >= 35 and test_styles == styles
    assert diverse_train_canvas >= 40 and diverse_train_outer >= 40
    assert train_bg_changes >= 10 and train_marker_changes >= 10
    assert varied_test_colors >= 10
    assert familiar_tests >= 10 and new_size_tests >= 20 and larger_tests >= 10
    assert changed_role_tests >= 15 and combined_tests >= 5
