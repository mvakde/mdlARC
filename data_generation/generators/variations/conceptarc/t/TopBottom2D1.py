import numpy as np


_SOURCE_INPUTS = {
    ("0000000000", "0000400000", "0004440000", "0000000000", "0003330000", "0003330000", "0003330000", "0000000000", "0000000000", "0000000000"),
    ("00330000", "00330000", "00000000", "04040000", "04440000", "00000000", "00000000", "00000000"),
    ("000000000", "000333330", "000000000", "000000000", "004444000", "000000000", "000000000", "000000000", "000000000"),
    ("00000000", "00040000", "00404000", "00040000", "00000000", "00333300", "00300300", "00033000"),
    ("444000", "444000", "000000", "003000", "333300", "000000"),
    ("0000000000", "0000000000", "4444444444", "3333333333", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("00000", "03330", "03030", "03330", "04040", "40404"),
}


def _signature(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _text_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _choose_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("Illegal pair count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("Training count must exceed test count")
    return int(num_train), int(num_test)


def _mask(rng, height, width, kind):
    mask = np.zeros((height, width), dtype=bool)
    if kind == "solid":
        mask[:] = True
    elif kind == "frame":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == "hole":
        mask[:] = True
        row = int(rng.integers(1, height - 1))
        col = int(rng.integers(1, width - 1))
        mask[row, col] = False
        if height >= 5 and width >= 5 and rng.random() < .5:
            mask[row, min(col + 1, width - 2)] = False
    elif kind == "checker":
        rr, cc = np.indices(mask.shape)
        mask[:] = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif kind == "pieces":
        mask[0, 0] = mask[0, -1] = True
        mask[-1, 0] = mask[-1, -1] = True
        if height >= 5 and width >= 6 and rng.random() < .6:
            mask[0, 1] = True
            mask[-1, -2] = True
    elif kind == "bend":
        steps = [0] * (height - 1) + [1] * (width - 1)
        rng.shuffle(steps)
        row = col = 0
        mask[row, col] = True
        for step in steps:
            row += step == 0
            col += step == 1
            mask[row, col] = True
        if height > 2 and width > 2 and rng.random() < .6:
            mask[int(rng.integers(1, height - 1)), 0] = True
    elif kind == "branch":
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        mask[:, col] = True
        mask[row, :] = True
        if height > 2 and width > 2:
            branch_row = int(rng.integers(height))
            mask[branch_row, :max(1, col)] = True
    elif kind == "asym":
        mask[:] = True
        if height > 1 and width > 1:
            mask[0, :max(1, width // 3)] = False
            mask[-1, -max(1, width // 3):] = False
            mask[0, -1] = mask[-1, 0] = True
            if height >= 3 and width >= 4:
                mask[1, -1] = False
    elif kind == "speckle":
        mask[:] = rng.random(mask.shape) < float(rng.uniform(.25, .48))
    else:
        raise ValueError("Unknown motif")
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        if not edge.any():
            edge[int(rng.integers(edge.size))] = True
    return mask


def _random_mask(rng, width_limit=9):
    kind = str(rng.choice(("solid", "frame", "hole", "checker", "pieces",
                           "bend", "branch", "asym", "speckle"),
                          p=(.11, .13, .10, .10, .16, .12, .11, .11, .06)))
    if kind in ("frame", "hole"):
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, width_limit + 1))
    elif kind == "pieces":
        h, w = int(rng.integers(3, 7)), int(rng.integers(4, width_limit + 1))
    else:
        h, w = int(rng.integers(1, 7)), int(rng.integers(2, width_limit + 1))
    return _mask(rng, h, w, kind)


def _pair_masks(rng, split, anchor, full_width):
    if split == "train" and anchor == 0:
        upper = _mask(rng, int(rng.integers(4, 7)),
                      int(rng.integers(5, 9)), "pieces")
        lower = _mask(rng, 3, int(rng.integers(3, 5)), "solid")
    elif split == "train" and anchor == 1:
        upper = _mask(rng, int(rng.integers(2, 4)),
                      int(rng.integers(6, 9)),
                      "solid" if rng.random() < .55 else "asym")
        lower = _mask(rng, int(rng.integers(5, 7)),
                      int(rng.integers(6, min(upper.shape[1], 8) + 1)),
                      "pieces")
    else:
        upper = _random_mask(rng)
        lower = _random_mask(rng)
    if full_width and upper.shape[1] < lower.shape[1]:
        upper = _mask(rng, max(upper.shape[0], 2),
                      lower.shape[1] + int(rng.integers(0, 4)),
                      str(rng.choice(("solid", "checker", "branch", "asym"))))
    return upper, lower


def _positions(rng, upper, lower, full_width, contact):
    uw, lw = upper.shape[1], lower.shape[1]
    if full_width:
        width = uw
        return width, 0, int(rng.integers(width - lw + 1))
    width = max(uw, lw) + int(rng.integers(0, 7))
    if contact is False:
        width = max(width, uw + lw + int(rng.integers(0, 4)))
    width = min(width, 30)
    candidates = []
    for ux in range(width - uw + 1):
        upper_columns = set(ux + np.flatnonzero(upper[-1]))
        for lx in range(width - lw + 1):
            touching = bool(upper_columns & set(lx + np.flatnonzero(lower[0])))
            if contact is None or touching == contact:
                candidates.append((ux, lx))
    if not candidates:
        candidates = [(ux, lx) for ux in range(width - uw + 1)
                      for lx in range(width - lw + 1)]
    if rng.random() < .38:
        edge = [(ux, lx) for ux, lx in candidates
                if ux in (0, width - uw) or lx in (0, width - lw)]
        if edge:
            candidates = edge
    ux, lx = candidates[int(rng.integers(len(candidates)))]
    return width, ux, lx


def _make_pair(rng, split, anchor, colors):
    background, upper_color, lower_color = colors
    full_width = rng.random() < (.35 if split == "train" else .22)
    if split == "train" and anchor == 0:
        full_width = False
        blank = int(rng.integers(1, 4))
    elif split == "train" and anchor == 1:
        blank = 0
    else:
        blank = 0 if rng.random() < (.48 if split == "train" else .62) else int(rng.integers(1, 4))
    upper, lower = _pair_masks(rng, split, anchor, full_width)
    contact = None if blank else (True if full_width else rng.random() < .54)
    width, upper_x, lower_x = _positions(rng, upper, lower, full_width, contact)
    top, bottom = int(rng.integers(0, 5)), int(rng.integers(0, 5))
    height = top + upper.shape[0] + blank + lower.shape[0] + bottom
    foreground = int(upper.sum() + lower.sum())
    required_height = (2 * foreground) // width + 1
    extra = max(0, 5 - height, required_height - height)
    if bottom == 0:
        top += extra
    else:
        bottom += extra
    height += extra
    assert height <= 30
    grid = np.full((height, width), background, dtype=np.int8)
    upper_y = top
    lower_y = top + upper.shape[0] + blank
    upper_region = grid[upper_y:upper_y + upper.shape[0], upper_x:upper_x + upper.shape[1]]
    lower_region = grid[lower_y:lower_y + lower.shape[0], lower_x:lower_x + lower.shape[1]]
    upper_region[upper] = upper_color
    lower_region[lower] = lower_color
    return {"input": grid,
            "output": grid[upper_y:upper_y + upper.shape[0],
                           upper_x:upper_x + upper.shape[1]].copy()}


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert counts[index] > sum(counts) - counts[index]
    return int(colors[index])


def _facts(grid):
    background = _background(grid)
    colors = [int(c) for c in np.unique(grid) if c != background]
    assert len(colors) == 2
    bounds = {}
    for color in colors:
        rows, cols = np.where(grid == color)
        bounds[color] = (int(rows.min()), int(rows.max()), int(cols.min()),
                         int(cols.max()), len(rows))
    colors.sort(key=lambda c: bounds[c][0])
    upper, lower = colors
    ur0, ur1, uc0, uc1, un = bounds[upper]
    lr0, lr1, lc0, lc1, ln = bounds[lower]
    assert ur0 < lr0 and ur1 < lr0
    blank = lr0 - ur1 - 1
    contact = blank == 0 and bool(set(np.flatnonzero(grid[ur1] == upper)) &
                                  set(np.flatnonzero(grid[lr0] == lower)))
    return (background, upper, lower, blank, contact,
            uc0 == 0 and uc1 == grid.shape[1] - 1,
            un, ln, (ur1 - ur0 + 1) * (uc1 - uc0 + 1),
            (lr1 - lr0 + 1) * (lc1 - lc0 + 1), lr1 == grid.shape[0] - 1)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    variable_colors = bool(rng.integers(2))
    if variable_colors:
        a = 0
        b, c = sorted(int(x) for x in rng.choice(np.arange(1, 10), 2,
                                                 replace=False))
    else:
        b, a, c = (int(x) for x in rng.choice(np.arange(10), 3, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if variable_colors:
                if split == "train" and index == 0:
                    colors = (b, a, c)
                elif split == "train" and index == 1:
                    colors = (a, c, b)
                else:
                    colors = tuple(int(x) for x in rng.choice(np.arange(10), 3, replace=False))
            else:
                upper = a if split == "train" and index == 0 else \
                    c if split == "train" and index == 1 else \
                    (a, c)[int(rng.integers(2))]
                colors = (b, upper, c if upper == a else a)
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(40):
                pair = _make_pair(pair_rng, split,
                                  index if split == "train" else -1, colors)
                signature = _signature(pair["input"])
                if signature not in seen and _text_rows(pair["input"]) not in _SOURCE_INPUTS:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a unique pair")
        if split == "train":
            rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_backgrounds, train_selected, train_palettes = set(), set(), set()
    train_blanks, train_count_signs, train_area_signs = set(), set(), set()
    train_combinations, train_hue_order = set(), set()
    train_hole = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            key = _signature(grid)
            pair_key = (key, _signature(output))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _text_rows(grid) not in _SOURCE_INPUTS
            (background, upper, lower, blank, contact, full_width,
             upper_count, lower_count, upper_area, lower_area,
             lower_bottom) = _facts(grid)
            assert upper != lower != background != upper
            rows, cols = np.where(grid == upper)
            expected = grid[rows.min():rows.max() + 1,
                            cols.min():cols.max() + 1]
            assert np.array_equal(output, expected)
            assert grid.shape != output.shape
            assert blank in (0, 1, 2, 3)
            if split == "train":
                train_backgrounds.add(background)
                train_selected.add(upper)
                train_palettes.add(frozenset((upper, lower)))
                train_blanks.add(blank == 0)
                train_count_signs.add(int(np.sign(upper_count - lower_count)))
                train_area_signs.add(int(np.sign(upper_area - lower_area)))
                train_combinations.add((int(np.sign(upper_count - lower_count)),
                                        int(np.sign(upper_area - lower_area))))
                train_hue_order.add(upper < lower)
                train_hole |= bool(np.any(output == background))
    assert train_blanks == {False, True}
    assert -1 in train_count_signs and 1 in train_count_signs
    assert -1 in train_area_signs and 1 in train_area_signs
    assert {(-1, 1), (1, -1)} <= train_combinations
    assert train_hue_order == {False, True}
    assert train_hole
    assert len(train_selected) >= 2
    if len(train_backgrounds) == 1:
        assert len(train_palettes) == 1
        bg = next(iter(train_backgrounds))
        palette = next(iter(train_palettes))
        assert train_selected == palette
        for pair in puzzle["test"]:
            facts = _facts(pair["input"])
            assert facts[0] == bg and frozenset(facts[1:3]) == palette
    else:
        assert 0 in train_backgrounds
        assert any(0 in _facts(p["input"])[1:3] for p in puzzle["train"])


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(p[k], q[k])
        for p, q in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def self_test() -> None:
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    puzzle_signatures, upper_shapes = set(), set()
    split_features = {s: {name: set() for name in
                          ("blank", "contact", "full_width", "bottom",
                           "hole", "disconnected", "bg", "count_sign", "area_sign")}
                      for s in ("train", "test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(tuple(_signature(p["input"]) for s in
                                    ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                bg, upper, lower, blank, contact, full, un, ln, ua, la, bottom = _facts(grid)
                rows, cols = np.where(grid == upper)
                assert np.array_equal(output, grid[rows.min():rows.max() + 1,
                                                   cols.min():cols.max() + 1])
                assert _text_rows(grid) not in _SOURCE_INPUTS
                palettes.add((bg, tuple(sorted((upper, lower)))))
                shapes.add((grid.shape, output.shape))
                layouts.add((grid.shape, output.shape, blank, un, ln, full))
                mask = output == upper
                upper_shapes.add((mask.shape, tuple(mask.flat)))
                features = split_features[split]
                for name, value in (("blank", blank), ("contact", contact),
                                    ("full_width", full), ("bottom", bottom),
                                    ("hole", bool(np.any(output == bg))),
                                    ("disconnected", _components(mask) > 1),
                                    ("bg", bg), ("count_sign", int(np.sign(un - ln))),
                                    ("area_sign", int(np.sign(ua - la)))):
                    features[name].add(value)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(palettes) >= 20 and len(shapes) >= 60
    assert len(layouts) >= 150 and len(upper_shapes) >= 65
    for split in ("train", "test"):
        features = split_features[split]
        assert len(features["blank"]) >= 3
        assert features["contact"] == {False, True}
        assert features["full_width"] == {False, True}
        assert features["bottom"] == {False, True}
        assert features["hole"] == {False, True}
        assert features["disconnected"] == {False, True}
        assert len(features["bg"]) >= 5
        assert {-1, 1} <= features["count_sign"]
        assert {-1, 1} <= features["area_sign"]
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4),
                                             (None, 1), (None, 4),
                                             (2, None), (10, None)):
        puzzle = generate_puzzle(718283, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(718283, requested_train,
                                                      requested_test))
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
    for requested_train, requested_test in ((1, None), (11, None),
                                             (None, 0), (None, 5),
                                             (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(19, requested_train, requested_test)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts were accepted")
