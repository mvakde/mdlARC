"""ARC family: shift every non-background cell exactly one row down."""

import numpy as np


_SOURCE = (
    (((0, 2, 2), (0, 0, 2), (0, 0, 0)), ((0, 0, 0), (0, 2, 2), (0, 0, 2))),
    (((0, 0, 0), (1, 1, 1), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (1, 1, 1))),
    (((0, 1, 0), (1, 1, 0), (0, 0, 0)), ((0, 0, 0), (0, 1, 0), (1, 1, 0))),
    (((1, 1, 1), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 1, 1), (0, 0, 0))),
    (((0, 0, 0), (0, 1, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 1, 0))),
    (((2, 0, 0), (2, 0, 0), (0, 0, 0)), ((0, 0, 0), (2, 0, 0), (2, 0, 0))),
)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_KEYS = frozenset((_key(np.asarray(a)), _key(np.asarray(b))) for a, b in _SOURCE)


def _counts(seed, rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _dimensions(rng, small=False, large=False):
    if small:
        return int(rng.integers(3, 7)), int(rng.integers(3, 7))
    if large:
        return int(rng.integers(15, 31)), int(rng.integers(15, 31))
    band = int(rng.choice(4, p=[0.14, 0.39, 0.32, 0.15]))
    bounds = ((3, 4), (4, 9), (8, 17), (16, 31))[band]
    height, width = int(rng.integers(*bounds)), int(rng.integers(*bounds))
    if band and rng.random() < 0.4:
        other = ((3, 7), (3, 10), (5, 15), (6, 21))[band]
        if rng.random() < 0.5:
            height = int(rng.integers(*other))
        else:
            width = int(rng.integers(*other))
    return height, width


def _grow(rng, height, width):
    box_h = int(rng.integers(2, min(height - 1, 8) + 1))
    box_w = int(rng.integers(2, min(width, 8) + 1))
    maximum = min(box_h * box_w, max(5, (box_h * box_w * 2) // 3), 28)
    target = int(rng.integers(4, maximum + 1))
    cells = {(int(rng.integers(box_h)), int(rng.integers(box_w)))}
    while len(cells) < target:
        frontier = {(rr, cc) for r, c in cells
                    for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                    if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells}
        if not frontier:
            break
        cells.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    return cells


def _shape(rng, height, width, family):
    usable_h = height - 1
    if family == "single":
        return {(0, 0)}
    if family == "hdomino":
        return {(0, 0), (0, 1)}
    if family == "vdomino":
        return {(0, 0), (1, 0)}
    if family in ("hbar", "vbar"):
        vertical = family == "vbar"
        limit = usable_h if vertical else width
        length = int(rng.integers(3, min(limit, 22) + 1))
        return {(r, 0) for r in range(length)} if vertical else {(0, c) for c in range(length)}
    if family == "L":
        cells = {(0, 0), (1, 0), (1, 1)}
        for _ in range(int(rng.integers(4))):
            cells = {(c, 1-r) for r, c in cells}
        return cells
    if family == "block":
        hh = int(rng.integers(2, min(usable_h, 6) + 1))
        ww = int(rng.integers(2, min(width, 7) + 1))
        return {(r, c) for r in range(hh) for c in range(ww)}
    if family == "poly":
        return _grow(rng, height, width)
    if family == "concave":
        hh = int(rng.integers(2, min(usable_h, 6) + 1))
        ww = int(rng.integers(3, min(width, 8) + 1))
        cells = {(hh-1, c) for c in range(ww)}
        cells.update((r, c) for r in range(hh) for c in (0, ww-1))
        return {(hh-1-r, c) for r, c in cells} if rng.random() < 0.5 else cells
    if family == "hole":
        hh = int(rng.integers(3, min(usable_h, 7) + 1))
        ww = int(rng.integers(3, min(width, 8) + 1))
        return {(r, c) for r in range(hh) for c in range(ww)
                if r in (0, hh-1) or c in (0, ww-1)}
    raise AssertionError("unknown shape family")


def _family(rng, height, width, preferred):
    choices = ["single", "hdomino", "vdomino", "L", "poly", "hbar", "block"]
    if height >= 4:
        choices.append("vbar")
        choices.append("hole")
    if width >= 3:
        choices.append("concave")
    if preferred in choices:
        return preferred
    return choices[int(rng.integers(len(choices)))]


def _separated_pieces(rng, height, width):
    box_h, box_w = min(height - 1, 8), min(width, 8)
    desired = int(rng.integers(2, 4))
    cells = set()
    for _ in range(desired):
        for _ in range(48):
            length = int(rng.integers(1, 4))
            vertical = bool(rng.integers(2))
            if vertical and length > box_h or not vertical and length > box_w:
                continue
            r = int(rng.integers(box_h - (length if vertical else 1) + 1))
            c = int(rng.integers(box_w - (1 if vertical else length) + 1))
            piece = {(r+k, c) for k in range(length)} if vertical else \
                    {(r, c+k) for k in range(length)}
            if all(abs(pr-cr) + abs(pc-cc) > 1
                   for pr, pc in piece for cr, cc in cells):
                cells.update(piece)
                break
        else:
            return {(0, 0), (0, 3), (2, 1)}
    return cells


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _place(rng, cells, height, width, stop_early=False):
    min_r, min_c = min(r for r, _ in cells), min(c for _, c in cells)
    cells = {(r-min_r, c-min_c) for r, c in cells}
    extent_h = max(r for r, _ in cells) + 1
    extent_w = max(c for _, c in cells) + 1
    max_top = height - 1 - extent_h - int(stop_early)
    assert max_top >= 0 and extent_w <= width
    top = int(rng.integers(max_top + 1))
    left = int(rng.integers(width - extent_w + 1))
    return {(r+top, c+left) for r, c in cells}


def _pair(rng, mode, palette, seen_colors, split, index, dimensions,
          preferred, background, stop_early=False, novel=False,
          fresh_train_color=False):
    height, width = dimensions
    family = _family(rng, height, width, preferred)
    cells = _shape(rng, height, width, family)
    pieces = mode == "pieces" and ((split == "train" and index == 1) or split == "test")
    multi = mode == "multicolor" and ((split == "train" and index == 1) or split == "test")
    if pieces:
        height, width = max(height, 5), max(width, 5)
        cells = _separated_pieces(rng, height, width)
    if multi and len(cells) < 3:
        cells = _shape(rng, height, width, "L")
    if 2 * len(cells) >= height * width:
        cells = _shape(rng, height, width, "L")
    cells = _place(rng, cells, height, width, stop_early)
    available = [color for color in palette if color != background]
    if fresh_train_color:
        available = [color for color in available if color not in seen_colors]
        if not available:
            available = [color for color in range(10)
                         if color != background and color not in seen_colors]
    if novel:
        available = [color for color in range(10)
                     if color != background and color not in seen_colors]
    if not available:
        available = [color for color in range(10) if color != background]
    primary = int(available[int(rng.integers(len(available)))])
    grid = np.full((height, width), background, dtype=np.int8)
    ordered = sorted(cells)
    secondary_options = [c for c in range(10) if c not in (background, primary)]
    secondary = int(secondary_options[int(rng.integers(len(secondary_options)))])
    for position, (r, c) in enumerate(ordered):
        grid[r, c] = secondary if multi and position % 3 == 1 else primary
    output = np.full_like(grid, background)
    for r, c in ordered:
        output[r+1, c] = grid[r, c]
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    # Direction is the puzzle-wide convention. Object complexity is introduced
    # in training before it can appear in test pairs.
    mode = ("simple", "pieces", "multicolor")[seed % 3]
    palette_size = (1, 2, 4, 6)[(seed // 3) % 4]
    palette = tuple(int(x) for x in rng.choice(10, size=palette_size, replace=False))
    source_like = seed % 11 == 0
    novel_test = seed % 5 == 2
    seen_colors = set()
    used = set()
    puzzle = {"train": [], "test": []}
    total = train_count + test_count
    pair_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=total, dtype=np.uint32)
    ordinal = 0
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[ordinal]))
            ordinal += 1
            for _ in range(32):
                anchor = split == "train" and index == 0
                small = anchor or (source_like and index < 2)
                large = not small and index == 2 and train_count >= 5
                dimensions = _dimensions(pair_rng, small=small, large=large)
                preferred = ("hbar" if anchor else
                             "L" if split == "train" and index == 1 and source_like else
                             "single" if split == "test" and source_like and index == 0 else None)
                background = (0 if source_like and seed % 22 == 0
                              else int(pair_rng.integers(10)))
                pair = _pair(pair_rng, mode, palette, seen_colors, split, index,
                             dimensions, preferred, background, stop_early=anchor,
                             novel=novel_test and split == "test" and index == 0,
                             fresh_train_color=novel_test and split == "train"
                             and index == 1)
                key = _key(pair["input"])
                full_key = (key, _key(pair["output"]))
                if key not in used and full_key not in _SOURCE_KEYS:
                    used.add(key)
                    if split == "train":
                        bg = _background(pair["input"])
                        seen_colors.update(int(x) for x in np.unique(pair["input"])
                                           if x != bg)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("failed to make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    train_stop = train_multi_piece = train_multicolor = False
    test_multi_piece = test_multicolor = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and a.shape[0] >= 3 and a.shape[1] >= 3
            background = _background(a)
            mask = a != background
            assert np.any(mask) and np.sum(mask) < a.size / 2
            assert not np.any(mask[-1])
            expected = np.full_like(a, background)
            expected[1:, :] = a[:-1, :]
            assert np.array_equal(b, expected) and not np.array_equal(a, b)
            key = _key(a)
            pair_key = (key, _key(b))
            assert key not in inputs and pair_key not in pairs
            assert pair_key not in _SOURCE_KEYS
            inputs.add(key)
            pairs.add(pair_key)
            pieces = len(_components(mask)) > 1
            colors = len(set(int(x) for x in a[mask])) > 1
            if split == "train":
                train_stop |= not np.any(b[-1] != background)
                train_multi_piece |= pieces
                train_multicolor |= colors
            else:
                test_multi_piece |= pieces
                test_multicolor |= colors
    assert train_stop
    assert not test_multi_piece or train_multi_piece
    assert not test_multicolor or train_multicolor


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_pairs = [p for puzzle in puzzles for split in ("train", "test")
                 for p in puzzle[split]]
    shapes = {p["input"].shape for p in all_pairs}
    assert len(shapes) > 35 and any(h >= 20 or w >= 20 for h, w in shapes)
    assert len({_key(p["input"]) for p in all_pairs}) > 150
    assert len({_background(p["input"]) for p in all_pairs}) == 10
    assert any(0 in p["input"][p["input"] != _background(p["input"])]
               for p in all_pairs)
    areas = {int(np.sum(p["input"] != _background(p["input"]))) for p in all_pairs}
    assert {1, 2, 3, 4, 5}.issubset(areas) and max(areas) >= 12
    counts = {name: 0 for name in ("hdomino", "vdomino", "block", "hbar",
                                   "vbar", "hole", "pieces", "multicolor",
                                   "edge", "interior", "bottom", "remaining",
                                   "novel_color", "test_smaller", "test_larger",
                                   "varied_background", "varied_dimensions")}
    for puzzle in puzzles:
        train_areas, test_areas, train_colors = [], [], set()
        dimensions, backgrounds = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                bg = _background(a)
                mask = a != bg
                rows, columns = np.nonzero(mask)
                area = len(rows)
                dimensions.add(a.shape)
                backgrounds.add(bg)
                colors = {int(x) for x in a[mask]}
                counts["hdomino"] += area == 2 and len(set(rows)) == 1
                counts["vdomino"] += area == 2 and len(set(columns)) == 1
                counts["hbar"] += area >= 4 and len(set(rows)) == 1
                counts["vbar"] += area >= 4 and len(set(columns)) == 1
                crop = mask[rows.min():rows.max()+1, columns.min():columns.max()+1]
                counts["block"] += area >= 4 and area == crop.size
                counts["hole"] += any(
                    all(0 < r < crop.shape[0]-1 and 0 < c < crop.shape[1]-1
                        for r, c in component)
                    for component in _components(~crop))
                counts["pieces"] += len(_components(mask)) > 1
                counts["multicolor"] += len(colors) > 1
                counts["edge"] += rows.min() == 0 or columns.min() == 0 or columns.max() == a.shape[1]-1
                counts["interior"] += columns.min() > 0 and columns.max() < a.shape[1]-1
                counts["bottom"] += np.any(b[-1] != bg)
                counts["remaining"] += not np.any(b[-1] != bg)
                if split == "train":
                    train_areas.append(area)
                    train_colors.update(colors)
                else:
                    test_areas.append(area)
                    counts["novel_color"] += bool(colors - train_colors)
        counts["test_smaller"] += min(test_areas) < min(train_areas)
        counts["test_larger"] += max(test_areas) > max(train_areas)
        counts["varied_background"] += len(backgrounds) > 1
        counts["varied_dimensions"] += len(dimensions) > 1
    assert all(counts[name] > 0 for name in counts)
    assert counts["varied_dimensions"] >= 35
    assert counts["varied_background"] >= 30
    assert counts["bottom"] >= 20 and counts["remaining"] >= 50
    assert counts["hole"] >= 5 and counts["pieces"] >= 10
    assert counts["multicolor"] >= 10 and counts["novel_color"] >= 10
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        # Training establishes the displacement; predict test outputs from
        # training grids and test inputs without hidden generator parameters.
        for train_pair in puzzle["train"]:
            a, b = train_pair["input"], train_pair["output"]
            bg = _background(a)
            observed = sorted(zip(*np.nonzero(b != bg)))
            source = sorted((r+1, c) for r, c in zip(*np.nonzero(a != bg)))
            assert observed == source
        for test_pair in puzzle["test"]:
            a = test_pair["input"]
            bg = _background(a)
            prediction = np.full(a.shape, bg, dtype=a.dtype)
            for r, c in zip(*np.nonzero(a != bg)):
                prediction[r+1, c] = a[r, c]
            assert np.array_equal(prediction, test_pair["output"])
    assert _signature(generate_puzzle(17)) == _signature(generate_puzzle(17))
    for seed, train, test in ((51, 2, 1), (52, 10, 1), (53, 10, 4),
                              (54, None, 1), (55, None, 4),
                              (56, 2, None), (57, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(60, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
