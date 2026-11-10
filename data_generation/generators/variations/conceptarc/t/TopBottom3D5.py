"""Layered-shape ARC puzzles: select the visibly frontmost shape and crop it."""

import numpy as np


_SOURCE_INPUTS = tuple(tuple(text.split()) for text in (
    """001111110000
001111110000
001122217770
033322217770
033322217770
033333117770
033333110000
033333000000
033300000000""",
    """000000000
000000330
099990330
499994444
499994444
499994444
099990000
097777000
007777000
007700000
000000000
000000000
000000000""",
    """0000000000
0000044400
0000333400
0002223400
0055523000
0005520000
0000500000
0000000000
0000000000""",
    """0000000000000
0000088880000
0044448880000
0004448886666
0004446666666
0044443666666
0000333660066
0000333000000
0000000000000""",
    """000000000000000
002233330000000
002233330000000
002200330000000
002444444000000
002444444000000
002200330000000
002200330000000
000000000000000
000000000000000""",
    """0030000000
0063666660
0066366660
0066636660
0000003000
0000000300""",
))


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _bounds(mask):
    positions = np.argwhere(mask)
    assert len(positions)
    top, left = positions.min(axis=0)
    bottom, right = positions.max(axis=0) + 1
    return int(top), int(bottom), int(left), int(right)


def _extract(grid, color):
    top, bottom, left, right = _bounds(grid == color)
    crop = grid[top:bottom, left:right]
    return np.where(crop == color, color, 0).astype(np.int8)


def _interruption_scores(grid):
    """Count visible A-B-A continuations, with B covering A."""
    colors = [int(c) for c in np.unique(grid) if c]
    scores = {color: 0 for color in colors}
    for lines in (grid, grid.T):
        for line in lines:
            runs = []
            for value in line:
                value = int(value)
                if not runs or runs[-1] != value:
                    runs.append(value)
            for a, b, c in zip(runs, runs[1:], runs[2:]):
                if a and b and a == c:
                    scores[b] += 1
                    scores[a] -= 1
    return scores


def _visible_foreground(grid):
    scores = _interruption_scores(grid)
    if not scores:
        return None
    order = sorted(scores, key=lambda c: scores[c], reverse=True)
    if scores[order[0]] < 2 or (len(order) > 1 and scores[order[0]] - scores[order[1]] < 2):
        return None
    return order[0]


def _tight(mask):
    t, b, l, r = _bounds(mask)
    return mask[t:b, l:r]


def _make_mask(rng, family):
    if family == "square":
        side = int(rng.integers(2, 7))
        mask = np.ones((side, side), dtype=bool)
    elif family == "rectangle":
        h = int(rng.integers(2, 7))
        w = int(rng.integers(2, 8))
        mask = np.ones((h, w), dtype=bool)
    elif family == "wide":
        h = int(rng.integers(2, 4))
        w = int(rng.integers(5, 10))
        mask = np.ones((h, w), dtype=bool)
    elif family == "wedge":
        h = int(rng.integers(3, 8))
        w = int(rng.integers(3, 8))
        mask = np.zeros((h, w), dtype=bool)
        right_aligned = bool(rng.integers(2))
        for row in range(h):
            length = max(1, w - int(round(row * (w - 1) / (h - 1))))
            if right_aligned:
                mask[row, w - length:] = True
            else:
                mask[row, :length] = True
    elif family == "path":
        h = int(rng.integers(4, 10))
        w = int(rng.integers(3, min(10, h + 2)))
        mask = np.zeros((h, w), dtype=bool)
        if rng.integers(3):
            cols = np.rint(np.linspace(0, w - 1, h)).astype(int)
        else:
            cols = np.zeros(h, dtype=int)
            col = int(rng.integers(w))
            for row in range(h):
                cols[row] = col
                col = max(0, min(w - 1, col + int(rng.choice([-1, 0, 1]))))
            if len(set(cols)) < 3:
                cols = np.rint(np.linspace(0, w - 1, h)).astype(int)
        mask[np.arange(h), cols] = True
        if rng.integers(3) == 0:
            row = int(rng.integers(1, h - 1))
            mask[row, max(0, cols[row] - 1):min(w, cols[row] + 2)] = True
    elif family in ("notch", "hole", "irregular"):
        h = int(rng.integers(3, 8))
        w = int(rng.integers(3, 8))
        mask = np.ones((h, w), dtype=bool)
        if family == "hole":
            mask[int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))] = False
        elif family == "notch" and rng.integers(2):
            depth = int(rng.integers(1, min(3, w)))
            rows = int(rng.integers(1, min(3, h)))
            mask[h - rows:, w - depth:] = False
        else:
            for _ in range(int(rng.integers(1, 4))):
                row = int(rng.integers(h))
                depth = int(rng.integers(1, min(3, w)))
                if rng.integers(2):
                    mask[row, :depth] = False
                else:
                    mask[row, w - depth:] = False
            if family == "irregular" and h >= 4:
                col = int(rng.integers(w))
                mask[0, col] = False
    else:
        raise ValueError("Unknown target family")
    mask = _tight(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(4) == 0:
        mask = np.rot90(mask)
    return mask.copy()


def _underlay_mask(rng, h, w, style):
    mask = np.ones((h, w), dtype=bool)
    if style == "step":
        step_from_left = bool(rng.integers(2))
        for row in range(h):
            inset = min(2, (row // max(2, h // 3)) % 3)
            if step_from_left:
                mask[row, :inset] = False
            else:
                mask[row, w - inset:] = False
    elif style == "notch":
        nh = int(rng.integers(1, min(4, h)))
        nw = int(rng.integers(1, min(4, w)))
        mask[:nh, :nw] = False
    elif style == "gap" and h > 4 and w > 4:
        row = int(rng.integers(1, h - 1))
        col = int(rng.integers(1, w - 1))
        mask[row, col] = False
    return mask


def _paint(grid, top, left, mask, color):
    h, w = mask.shape
    region = grid[top:top + h, left:left + w]
    region[mask] = color


def _scene(rng, family, underlays, detached, mode):
    target_mask = _make_mask(rng, family)
    h, w = target_mask.shape
    ty, tx = 20, 20
    colors = [int(c) for c in rng.choice(np.arange(1, 10), 1 + underlays + detached, replace=False)]
    target = colors[0]
    grid = np.zeros((48, 48), dtype=np.int8)
    styles = ("rect", "step", "notch", "gap")
    for index in range(underlays):
        if mode == "edge" and index == 0:
            side = int(rng.integers(4))
            top = ty - int(rng.integers(2, 5))
            bottom = ty + h + int(rng.integers(2, 5))
            left = tx - int(rng.integers(2, 5))
            right = tx + w + int(rng.integers(2, 5))
            if side == 0:
                left = tx + int(rng.integers(1, max(2, w)))
            elif side == 1:
                right = tx + w - int(rng.integers(1, max(2, w)))
            elif side == 2:
                top = ty + int(rng.integers(1, max(2, h)))
            else:
                bottom = ty + h - int(rng.integers(1, max(2, h)))
        elif mode == "bridge" and index < 2:
            midpoint = w // 2
            if index == 0:
                left = tx - int(rng.integers(2, 5))
                right = tx + midpoint + int(rng.integers(1, 3))
            else:
                left = tx + midpoint - int(rng.integers(1, 3))
                right = tx + w + int(rng.integers(2, 5))
            top = ty - int(rng.integers(2, 5))
            bottom = ty + h + int(rng.integers(2, 5))
        elif index == 0 or mode == "cover":
            top = ty - int(rng.integers(2, 5))
            bottom = ty + h + int(rng.integers(2, 5))
            left = tx - int(rng.integers(2, 5))
            right = tx + w + int(rng.integers(2, 5))
            if index:
                top += int(rng.integers(-3, 4))
                bottom += int(rng.integers(-3, 4))
                left += int(rng.integers(-3, 4))
                right += int(rng.integers(-3, 4))
        else:
            top = ty - int(rng.integers(1, 5)) + int(rng.integers(-2, 3))
            bottom = ty + h + int(rng.integers(1, 5)) + int(rng.integers(-2, 3))
            left = tx - int(rng.integers(1, 5)) + int(rng.integers(-2, 3))
            right = tx + w + int(rng.integers(1, 5)) + int(rng.integers(-2, 3))
        if bottom - top < 3 or right - left < 3:
            return None
        style = str(rng.choice(styles, p=[0.35, 0.25, 0.25, 0.15]))
        mask = _underlay_mask(rng, bottom - top, right - left, style)
        _paint(grid, top, left, mask, colors[index + 1])
    pre_target = grid[ty:ty + h, tx:tx + w].copy()
    if np.count_nonzero(pre_target[target_mask]) < min(2, np.count_nonzero(target_mask)):
        return None
    _paint(grid, ty, tx, target_mask, target)
    main_top, main_bottom, main_left, main_right = _bounds(grid != 0)
    available_sides = ["top", "bottom", "left", "right"]
    rng.shuffle(available_sides)
    for index in range(detached):
        dh = int(rng.integers(2, 5))
        dw = int(rng.integers(2, 5))
        side = available_sides[index]
        if side == "top":
            top = main_top - dh - int(rng.integers(2, 5))
            left = int(rng.integers(main_left - 2, main_right + 2))
        elif side == "bottom":
            top = main_bottom + int(rng.integers(2, 5))
            left = int(rng.integers(main_left - 2, main_right + 2))
        elif side == "left":
            top = int(rng.integers(main_top - 2, main_bottom + 2))
            left = main_left - dw - int(rng.integers(2, 5))
        else:
            top = int(rng.integers(main_top - 2, main_bottom + 2))
            left = main_right + int(rng.integers(2, 5))
        if top < 0 or left < 0 or top + dh > 48 or left + dw > 48:
            return None
        mask = _underlay_mask(rng, dh, dw, str(rng.choice(("rect", "notch"))))
        _paint(grid, top, left, mask, colors[1 + underlays + index])
    if rng.integers(2):
        grid = np.fliplr(grid)
    if rng.integers(2):
        grid = np.flipud(grid)
    if rng.integers(2):
        grid = np.rot90(grid)
    top, bottom, left, right = _bounds(grid != 0)
    margins = [int(v) for v in rng.integers(1, 4, size=4)]
    ih = bottom - top + margins[0] + margins[1]
    iw = right - left + margins[2] + margins[3]
    if ih > 30 or iw > 30:
        return None
    grid = grid[top:bottom, left:right]
    grid = np.pad(grid, ((margins[0], margins[1]), (margins[2], margins[3])))
    if len(set(int(c) for c in np.unique(grid)) - {0}) != 1 + underlays + detached:
        return None
    if _visible_foreground(grid) != target:
        return None
    return grid.astype(np.int8), _extract(grid, target)


def _family(rng, regime, index):
    if regime == "train" and index == 0:
        pool = ("square", "rectangle", "notch", "wedge")
    elif regime == "train" and index == 1:
        pool = ("wedge", "path", "wide", "notch", "irregular")
    elif regime == "test":
        pool = ("wide", "path", "irregular", "wedge", "hole", "rectangle", "notch", "square")
    else:
        pool = ("square", "rectangle", "wide", "wedge", "path", "notch", "hole", "irregular")
    return str(rng.choice(pool))


def _pair(rng, family, preferred_complexity):
    for _ in range(160):
        if preferred_complexity == "simple":
            n_under = int(rng.choice((1, 1, 1, 2, 2, 3)))
        elif preferred_complexity == "layered":
            n_under = int(rng.choice((1, 2, 2, 3, 3)))
        else:
            n_under = int(rng.integers(1, 4))
        detached = int(rng.choice((0, 0, 1, 1, 2)))
        mode = str(rng.choice(("cover", "stagger", "bridge", "edge")))
        if n_under == 1 and mode == "bridge":
            mode = "cover"
        candidate = _scene(rng, family, n_under, detached, mode)
        if candidate is not None:
            return candidate
    # Deterministic safe construction if an unusual mask exhausts the sampled scenes.
    below, target = (int(c) for c in rng.choice(np.arange(1, 10), 2, replace=False))
    h, w = (int(v) for v in rng.integers(8, 12, size=2))
    grid = np.zeros((h + 2, w + 2), dtype=np.int8)
    grid[1:h + 1, 1:w + 1] = below
    grid[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2] = target
    return grid, _extract(grid, target)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("Illegal pair count")
    options = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (num_train is None or a == num_train)
               and (num_test is None or b == num_test)]
    if not options:
        raise ValueError("Train count must exceed test count")
    rng = np.random.default_rng(seed)
    train_count, test_count = options[int(rng.integers(len(options)))]
    # Zero is the source background; depth and crop are the only shared conventions.
    complexity = str(rng.choice(("simple", "mixed", "layered")))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_target = None
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            family = _family(rng, regime, index)
            for _ in range(200):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                input_grid, output_grid = _pair(pair_rng, family, complexity)
                key = _rows(input_grid)
                target = int(np.max(output_grid))
                if key in seen or key in _SOURCE_INPUTS:
                    continue
                if regime == "train" and index == 1 and target == first_target:
                    continue
                break
            else:
                raise RuntimeError("Could not obtain a distinct pair")
            if regime == "train" and index == 0:
                first_target = target
            seen.add(key)
            puzzle[regime].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_targets = []
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            key = _rows(inp)
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            assert np.any(inp == 0) and np.any(out != 0)
            out_colors = {int(c) for c in np.unique(out)} - {0}
            assert len(out_colors) == 1
            target = next(iter(out_colors))
            assert target in inp and np.array_equal(out, _extract(inp, target))
            assert _visible_foreground(inp) == target
            assert not np.array_equal(inp, out)
            assert 2 <= len(set(int(c) for c in np.unique(inp)) - {0}) <= 6
            if regime == "train":
                train_targets.append(target)
    assert len(set(train_targets)) >= 2


def _signature(puzzle):
    return tuple(tuple((_rows(pair["input"]), _rows(pair["output"])) for pair in puzzle[regime])
                 for regime in ("train", "test"))


def _nonzero_components(grid):
    remaining = set(map(tuple, np.argwhere(grid != 0)))
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            y, x = frontier.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        frontier.append(neighbor)
    return count


def self_test() -> None:
    signatures, counts, sizes, masks, ncolors, score_gaps = set(), set(), set(), set(), set(), set()
    families = set()
    components = set()
    adjacent_repeats = partial_overlaps = multiple_underlays = 0
    train_wide_or_sparse = test_compact = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        target_sequence = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                target = int(np.max(out))
                target_sequence.append(target)
                sizes.add((inp.shape, out.shape))
                masks.add(tuple(map(tuple, out != 0)))
                ncolors.add(len(np.unique(inp)) - 1)
                components.add(_nonzero_components(inp))
                ordered = sorted(_interruption_scores(inp).values(), reverse=True)
                score_gaps.add(ordered[0] - ordered[1])
                h, w = out.shape
                sparse = np.count_nonzero(out) <= max(h, w) + 2 and not np.all(out != 0)
                if regime == "train" and (w >= h + 3 or sparse):
                    train_wide_or_sparse += 1
                if regime == "test" and h <= 5 and w <= 5:
                    test_compact += 1
                tt, tb, tl, tr = _bounds(inp == target)
                intersecting = 0
                for color in (int(c) for c in np.unique(inp) if c and c != target):
                    ct, cb, cl, cr = _bounds(inp == color)
                    if max(ct, tt) < min(cb, tb) and max(cl, tl) < min(cr, tr):
                        intersecting += 1
                        if not (ct <= tt and cb >= tb and cl <= tl and cr >= tr):
                            partial_overlaps += 1
                if intersecting >= 2:
                    multiple_underlays += 1
                if np.all(out != 0):
                    families.add("wide" if w >= h + 3 else "solid")
                elif sparse:
                    families.add("sparse")
                else:
                    families.add("irregular")
        adjacent_repeats += sum(a == b for a, b in zip(target_sequence, target_sequence[1:]))
    assert len(signatures) == 42
    assert len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3
    assert len(sizes) >= 30 and len(masks) >= 25
    assert {2, 3, 4, 5, 6} <= ncolors and {"wide", "solid", "sparse", "irregular"} <= families
    assert {1, 2, 3} <= components
    assert adjacent_repeats > 0 and partial_overlaps > 0 and multiple_underlays > 0
    assert train_wide_or_sparse > 0 and test_compact > 0
    assert min(score_gaps) >= 2
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(541, a, b)
        validate_generated_puzzle(puzzle)
        assert (a is None or len(puzzle["train"]) == a)
        assert (b is None or len(puzzle["test"]) == b)
        assert _signature(puzzle) == _signature(generate_puzzle(541, a, b))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(541, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
