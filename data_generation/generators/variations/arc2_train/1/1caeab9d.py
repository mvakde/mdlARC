"""Generate puzzles that align congruent colored silhouettes to one reference."""

import numpy as np

_SOURCE_ROWS = (
    (("0000000444", "0000000444", "0222000000", "0222000000", "0000000000", "0000111000", "0000111000", "0000000000", "0000000000", "0000000000"),
     ("0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0222111444", "0222111444", "0000000000", "0000000000", "0000000000")),
    (("0220000000", "0220000110", "0000440110", "0000440000", "0000000000"),
     ("0000000000", "0220440110", "0220440110", "0000000000", "0000000000")),
    (("0000000000", "0002000000", "0102000000", "0100004000", "0000004000"),
     ("0000000000", "0000000000", "0102004000", "0102004000", "0000000000")),
    (("0000000022", "0000000022", "0110000200", "0110000000", "1000000000", "0000440000", "0000440000", "0004000000", "0000000000", "0000000000"),
     ("0000000000", "0000000000", "0110440022", "0110440022", "1004000200", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000")),
)
_SOURCE_PAIRS = tuple(
    (np.array([[int(c) for c in row] for row in ins], dtype=np.int8),
     np.array([[int(c) for c in row] for row in outs], dtype=np.int8))
    for ins, outs in _SOURCE_ROWS
)
_FAMILIES = ("rectangle", "bar", "L", "T", "stair", "ring", "composite")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng, family, max_width):
    """Return normalized cells; several families have independently sampled dimensions."""
    if family == "rectangle":
        h, w = int(rng.integers(2, 9)), int(rng.integers(2, max_width + 1))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif family == "bar":
        if int(rng.integers(2)) and max_width >= 3:
            w = int(rng.integers(3, max_width + 1))
            cells = {(0, c) for c in range(w)}
        else:
            h = int(rng.integers(3, 11))
            cells = {(r, 0) for r in range(h)}
    elif family == "L":
        h, w = int(rng.integers(3, 9)), int(rng.integers(2, max_width + 1))
        stem = 0 if int(rng.integers(2)) else w - 1
        foot = 0 if int(rng.integers(2)) else h - 1
        cells = {(r, stem) for r in range(h)} | {(foot, c) for c in range(w)}
    elif family == "T":
        h, w = int(rng.integers(3, 9)), int(rng.integers(3, max_width + 1))
        cross = 0 if int(rng.integers(2)) else h - 1
        stem = int(rng.integers(1, w - 1))
        cells = {(cross, c) for c in range(w)} | {(r, stem) for r in range(h)}
    elif family == "stair":
        h, w = int(rng.integers(3, 9)), int(rng.integers(3, max_width + 1))
        cells = set()
        for r in range(h):
            c = round(r * (w - 1) / (h - 1))
            cells.add((r, c))
            if c + 1 < w and int(rng.integers(3)):
                cells.add((r, c + 1))
        if int(rng.integers(2)):
            cells = {(r, w - 1 - c) for r, c in cells}
    elif family == "ring":
        h, w = int(rng.integers(3, 9)), int(rng.integers(3, max_width + 1))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    else:
        h, w = int(rng.integers(4, 10)), int(rng.integers(4, max_width + 1))
        cells = {(r, c) for r in range(2) for c in range(2)}
        if int(rng.integers(2)):
            cells |= {(h - 1, w - 1)}
        else:
            cells |= {(h - 2, c) for c in range(w - 2, w)}
            cells |= {(h - 1, c) for c in range(w - 2, w)}
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _colors(rng, reference, stable, stable_palette, count):
    if stable:
        return stable_palette[0], tuple(stable_palette[1:count])
    if int(rng.integers(100)) < 45:
        background = 0
    else:
        background = int(rng.choice([c for c in range(1, 10) if c != reference]))
    options = [c for c in range(10) if c not in (background, reference)]
    movers = tuple(int(c) for c in rng.choice(options, size=count - 1, replace=False))
    return background, movers


def _biased_position(rng, start, stop):
    if start == stop:
        return start
    if int(rng.integers(4)) == 0:
        return start if int(rng.integers(2)) else stop
    return int(rng.integers(start, stop + 1))


def _one_pair(rng, axis, reference, stable, stable_palette, regime, classic,
              informative, simple=False, repeated_shape=None, repeated_count=None):
    count = (3 if simple else repeated_count if repeated_count is not None else
             int(rng.choice((2, 3, 4, 5), p=(.13, .55, .25, .07))))
    if informative:
        count = max(3, count)
    max_width = min(8, 30 // count)
    families = (("rectangle", "bar") if simple or
                (classic and regime == "source_train") else _FAMILIES)
    weights = None if len(families) == 2 else ((.11, .10, .18, .15, .15, .15, .16)
                                                if regime == "target_test" else
                                                (.25, .16, .14, .12, .11, .10, .12))
    if repeated_shape is None:
        family = str(rng.choice(families, p=weights))
        if family in ("T", "stair", "ring") and max_width < 3:
            family = "L"
        if family == "composite" and max_width < 4:
            family = "L"
        shape = _shape(rng, family, min(4, max_width) if simple else max_width)
    else:
        shape = repeated_shape
    sh, sw = max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1
    base_width = count * sw
    spare = 30 - base_width
    packing = int(rng.choice(3, p=(.33, .40, .27)))
    if packing == 0:
        extra = int(rng.integers(0, min(2, spare) + 1))
    elif packing == 1:
        extra = int(rng.integers(min(3, spare), min(10, spare) + 1))
    else:
        extra = int(rng.integers(min(11, spare), spare + 1))
    if spare and int(rng.integers(12)) == 0:
        extra = spare
    allocation = rng.multinomial(extra, rng.dirichlet(np.full(count + 1, .75)))
    starts = []
    cursor = int(allocation[0])
    for i in range(count):
        starts.append(cursor)
        cursor += sw + int(allocation[i + 1])
    width = base_width + extra
    min_height = max(sh + 2, 2 * count * len(shape) // width + 2)
    if min_height > 30:
        raise ValueError("shape cannot fit with a dominant background")
    scale = int(rng.choice(3, p=(.30, .45, .25)))
    if scale == 0:
        height = int(rng.integers(min_height, min(30, min_height + 3) + 1))
    elif scale == 1:
        height = int(rng.integers(min_height, min(30, min_height + 12) + 1))
    else:
        height = int(rng.integers(max(min_height, 22), 31))
    max_top = height - sh
    background, movers = _colors(rng, reference, stable, stable_palette, count)
    object_colors = list(movers)
    object_colors.insert(int(rng.integers(count)), reference)
    mode = str(rng.choice(("above", "below", "straddle", "free")))
    if mode == "above":
        target = _biased_position(rng, 1, max_top)
    elif mode == "below":
        target = _biased_position(rng, 0, max_top - 1)
    elif mode == "straddle":
        target = _biased_position(rng, 1, max_top - 1)
    else:
        target = _biased_position(rng, 0, max_top)
    tops = {reference: target}
    for i, color in enumerate(movers):
        if mode == "above" or (mode == "straddle" and i % 2 == 0):
            tops[color] = _biased_position(rng, 0, target - 1)
        elif mode == "below" or mode == "straddle":
            tops[color] = _biased_position(rng, target + 1, max_top)
        else:
            choices = [t for t in range(max_top + 1) if t != target]
            tops[color] = int(rng.choice(choices))
    if not informative:
        aligned = int(rng.choice((0, 1, count - 1), p=(.65, .25, .10)))
        if aligned:
            for color in rng.choice(movers, size=aligned, replace=False):
                tops[int(color)] = target
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    for color, left in zip(object_colors, starts):
        for r, c in shape:
            input_grid[tops[color] + r, left + c] = color
            output_grid[target + r, left + c] = color
    if axis == 1:
        input_grid, output_grid = input_grid.T.copy(), output_grid.T.copy()
    return input_grid, output_grid


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_copy(inp, out):
    return any(inp.shape == a.shape and np.array_equal(inp, a) and
               np.array_equal(out, b) for a, b in _SOURCE_PAIRS)


def _layout_signature(inp, out, axis, reference):
    bg = _background(inp)
    ins, outs = _objects(inp, bg), _objects(out, bg)
    ordered = sorted(ins, key=lambda c: min(v[1 - axis] for v in ins[c]))
    rank = ordered.index(reference)
    signs = frozenset(int(np.sign(min(v[axis] for v in outs[c]) -
                                   min(v[axis] for v in ins[c])))
                      for c in ins if c != reference)
    return rank, signs


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    simple_puzzle = bool(rng.integers(5) == 0)
    axis = 0 if simple_puzzle else int(rng.integers(2))
    reference = int(rng.integers(1, 10))
    stable = simple_puzzle or bool(rng.integers(4) == 0)
    background = 0 if simple_puzzle or int(rng.integers(2)) else int(rng.choice(
        [c for c in range(1, 10) if c != reference]))
    stable_palette = (background,) + tuple(int(c) for c in rng.permutation(
        [c for c in range(10) if c not in (background, reference)]))
    classic = bool(rng.integers(5) == 0)
    reuse_training_shape = bool(rng.integers(5) < 2)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = first_count = first_layout = None
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for i in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            repeat = split == "train" and i == 1 and reuse_training_shape
            for _ in range(100):
                inp, out = _one_pair(pair_rng, axis, reference, stable,
                                     stable_palette, regime, classic,
                                     split == "train" and i < 2,
                                     simple_puzzle,
                                     first_shape if repeat else None,
                                     first_count if repeat else None)
                key = _grid_key(inp)
                layout = _layout_signature(inp, out, axis, reference)
                if (key not in seen and not _source_copy(inp, out) and
                    (not repeat or (layout[0] != first_layout[0] and
                                    layout[1] != first_layout[1]))):
                    break
            else:
                raise RuntimeError("unable to construct a unique pair")
            seen.add(key)
            puzzle[split].append({"input": inp, "output": out})
            if split == "train" and i == 0:
                bg = _background(inp)
                objects = _objects(inp, bg)
                first_count = len(objects)
                first_shape = _normalized(next(iter(objects.values())))
                if axis == 1:
                    first_shape = tuple(sorted((c, r) for r, c in first_shape))
                first_layout = layout
    validate_generated_puzzle(puzzle)
    return puzzle


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _require(counts[index] > grid.size // 2, "background must occupy a majority")
    return int(values[index])


def _objects(grid, background):
    return {int(c): tuple(map(tuple, np.argwhere(grid == c).tolist()))
            for c in np.unique(grid) if int(c) != background}


def _origin(cells):
    return min(r for r, _ in cells), min(c for _, c in cells)


def _normalized(cells):
    top, left = _origin(cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _solve(input_grid, axis, reference):
    bg = _background(input_grid)
    objects = _objects(input_grid, bg)
    _require(reference in objects, "reference is absent")
    target = min(cell[axis] for cell in objects[reference])
    result = np.full(input_grid.shape, bg, dtype=np.int8)
    for color, cells in objects.items():
        shift = target - min(cell[axis] for cell in cells)
        for r, c in cells:
            rr, cc = (r + shift, c) if axis == 0 else (r, c + shift)
            _require(0 <= rr < result.shape[0] and 0 <= cc < result.shape[1],
                     "translated object leaves grid")
            _require(result[rr, cc] == bg, "translated objects overlap")
            result[rr, cc] = color
    return result


def _candidates(train):
    first = train[0]["input"]
    colors = _objects(first, _background(first))
    candidates = set()
    for axis in (0, 1):
        for color in colors:
            valid = True
            for pair in train:
                inp = pair["input"]
                if color not in _objects(inp, _background(inp)):
                    valid = False
                    break
                try:
                    predicted = _solve(inp, axis, color)
                except AssertionError:
                    valid = False
                    break
                if not np.array_equal(predicted, pair["output"]):
                    valid = False
                    break
            if valid:
                candidates.add((axis, color))
    return candidates


def _check_pair(pair, axis, reference):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "output size changed")
    bg = _background(inp)
    _require(_background(out) == bg, "background changed")
    ins, outs = _objects(inp, bg), _objects(out, bg)
    _require(2 <= len(ins) <= 5 and set(ins) == set(outs), "invalid object colors")
    _require(reference in ins, "reference missing")
    forms = {_normalized(cells) for cells in ins.values()}
    _require(len(forms) == 1, "copies have different silhouettes")
    span_axis = 1 - axis
    spans = []
    for color, cells in ins.items():
        _require(_normalized(outs[color]) == next(iter(forms)), "silhouette changed")
        _require(min(v[span_axis] for v in cells) ==
                 min(v[span_axis] for v in outs[color]), "perpendicular position changed")
        spans.append((min(v[span_axis] for v in cells),
                      max(v[span_axis] for v in cells)))
    spans.sort()
    _require(all(spans[i][1] < spans[i + 1][0]
                 for i in range(len(spans) - 1)), "perpendicular spans overlap")
    _require(np.array_equal(_solve(inp, axis, reference), out),
             "output does not align exact cells to reference")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(type(puzzle) is dict and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(type(puzzle["train"]) is list and type(puzzle["test"]) is list,
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and
             len(train) > len(test), "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        _require(type(pair) is dict and set(pair) == {"input", "output"},
                 "pair must have input and output")
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                     np.issubdtype(grid.dtype, np.integer) and
                     all(1 <= side <= 30 for side in grid.shape) and
                     np.all((0 <= grid) & (grid <= 9)), "invalid grid")
        ikey = _grid_key(pair["input"])
        pkey = ikey, _grid_key(pair["output"])
        _require(ikey not in seen_inputs and pkey not in seen_pairs,
                 "duplicate pair")
        _require(not _source_copy(pair["input"], pair["output"]),
                 "source pair copied")
        seen_inputs.add(ikey)
        seen_pairs.add(pkey)
    candidates = _candidates(train)
    _require(len(candidates) == 1, "training does not uniquely reveal axis/reference")
    axis, reference = next(iter(candidates))
    for i, pair in enumerate(train):
        _check_pair(pair, axis, reference)
        if i < 2:
            bg = _background(pair["input"])
            ins = _objects(pair["input"], bg)
            outs = _objects(pair["output"], bg)
            _require(len(ins) >= 3 and
                     {c for c in ins if ins[c] == outs[c]} == {reference},
                     "first training demonstrations must identify reference")
    for pair in test:
        _check_pair(pair, axis, reference)


def _same_puzzle(a, b):
    return (all(len(a[s]) == len(b[s]) for s in ("train", "test")) and
            all(np.array_equal(x[key], y[key]) for split in ("train", "test")
                for x, y in zip(a[split], b[split])
                for key in ("input", "output")))


def _components(cells):
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _has_hole(shape):
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    empty = {(r, c) for r in range(h) for c in range(w)} - set(shape)
    visited = set()
    for cell in empty:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        reaches_edge = False
        while stack:
            r, c = stack.pop()
            reaches_edge |= r in (0, h - 1) or c in (0, w - 1)
            for neighbor in ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1)):
                if neighbor in empty and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if not reaches_edge:
            return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(_same_puzzle(puzzles[0], generate_puzzle(0)), "not deterministic")
    _require(not _same_puzzle(puzzles[0], puzzles[1]), "seeds did not vary")
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3, "counts lack range")
    axes, references, sizes, counts = set(), set(), set(), set()
    backgrounds, shapes = set(), set()
    ranks, motion_modes = set(), set()
    aligned = changed = irregular = disconnected = hollow = 0
    compact = large = zero_object = adjacent = uneven = large_move = 0
    irregular_train = simple_test = shape_reuse = mixed_convention = 0
    simple_puzzles = classic_shifts = same_order = reversed_order = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        axis, reference = next(iter(_candidates(puzzle["train"])))
        axes.add(axis)
        references.add(reference)
        local_backgrounds, local_shapes = set(), {}
        local_counts, local_orders = set(), []
        train_simple, test_simple = [], []
        for split, pair in ((s, q) for s in ("train", "test") for q in puzzle[s]):
            inp, out = pair["input"], pair["output"]
            bg = _background(inp)
            objects, output_objects = _objects(inp, bg), _objects(out, bg)
            backgrounds.add(bg)
            local_backgrounds.add(bg)
            sizes.add(inp.shape)
            counts.add(len(objects))
            local_counts.add(len(objects))
            shape = _normalized(next(iter(objects.values())))
            shapes.add(shape)
            h = max(r for r, _ in shape) + 1
            w = max(c for _, c in shape) + 1
            simple = len(shape) == h * w
            (train_simple if split == "train" else test_simple).append(simple)
            irregular += not simple
            disconnected += _components(shape) > 1
            hollow += _has_hole(shape)
            aligned += np.array_equal(inp, out)
            changed += not np.array_equal(inp, out)
            zero_object += bg != 0 and 0 in objects
            ordered = sorted(objects, key=lambda c: min(p[1-axis] for p in objects[c]))
            rank = ordered.index(reference)
            ranks.add(rank)
            if len(objects) == 3:
                mover_order = tuple(c for c in ordered if c != reference)
                local_orders.append((split, frozenset(mover_order), mover_order))
            starts = [min(p[1-axis] for p in objects[c]) for c in ordered]
            span = max(p[1-axis] for p in next(iter(objects.values()))) - min(
                p[1-axis] for p in next(iter(objects.values()))) + 1
            gaps = [starts[i+1] - starts[i] - span for i in range(len(starts)-1)]
            compact += inp.shape[1-axis] - len(objects)*span <= 1
            large += 30 in inp.shape
            adjacent += 0 in gaps
            uneven += len(set(gaps)) > 1
            shifts = [min(p[axis] for p in output_objects[c]) -
                      min(p[axis] for p in objects[c])
                      for c in objects if c != reference]
            large_move += any(abs(s) >= 5 for s in shifts)
            mode = frozenset(int(np.sign(s)) for s in shifts if s)
            motion_modes.add(mode)
            local_shapes.setdefault(shape, set()).add((rank, mode))
            _require(np.array_equal(_solve(inp, axis, reference), out),
                     "independent reconstruction failed")
        irregular_train += not all(train_simple)
        simple_test += any(test_simple)
        simple_puzzles += (axis == 0 and local_backgrounds == {0} and
                           local_counts == {3} and all(train_simple + test_simple))
        classic_shifts += all(train_simple) and not all(test_simple)
        comparisons = [(a[2], b[2]) for a in local_orders if a[0] == "train"
                       for b in local_orders if b[0] == "test" and a[1] == b[1]]
        same_order += any(a == b for a, b in comparisons)
        reversed_order += any(a != b for a, b in comparisons)
        shape_reuse += any(len(v) > 1 for v in local_shapes.values())
        mixed_convention += len(local_backgrounds) > 1
    _require(axes == {0, 1} and references == set(range(1, 10)) and
             len(sizes) >= 40 and counts == {2, 3, 4, 5}
             and len(backgrounds) >= 5 and len(shapes) >= 35 and
             ranks >= {0, 1, 2} and len(motion_modes) >= 3 and
             aligned > 0 and changed > 0 and irregular > 0 and
             disconnected > 0 and hollow > 0 and compact > 0 and
             large > 0 and zero_object > 0 and adjacent > 0 and
             uneven > 0 and large_move > 0 and irregular_train > 0 and
             simple_test > 0 and shape_reuse > 0 and mixed_convention > 0 and
             simple_puzzles > 0 and classic_shifts > 0 and
             same_order > 0 and reversed_order > 0,
             "structural diversity fell below self-test bounds")
    for seed, train, test in ((501, 2, 1), (502, 10, 1), (503, 10, 4),
                              (504, None, 1), (505, None, 4),
                              (506, 2, None), (507, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            _require(len(p["train"]) == train, "train count ignored")
        if test is not None:
            _require(len(p["test"]) == test, "test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(700, train, test)
        except ValueError:
            continue
        raise AssertionError("invalid count accepted")
