import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "4444444417771",
            "4117771414444",
            "4111111414114",
            "4111111414114",
            "4111111411441",
            "4111111411111",
            "4444444411111",
        ),
        (
            "44444444",
            "41144444",
            "41141144",
            "41141144",
            "41114414",
            "41111114",
            "44444444",
        ),
    ),
    (
        (
            "4111111111714",
            "4444441111444",
            "1114141111111",
            "1444441111111",
            "1411111111111",
            "1411111111111",
            "1711111111111",
        ),
        (
            "411111",
            "444444",
            "111414",
            "144444",
            "141111",
            "141411",
            "144411",
        ),
    ),
    (
        (
            "444",
            "414",
            "444",
            "777",
            "111",
            "777",
            "444",
            "414",
            "444",
        ),
        (
            "444",
            "414",
            "444",
            "444",
            "414",
            "444",
        ),
    ),
    (
        (
            "444411111",
            "411411111",
            "444444111",
            "111114111",
            "111117144",
            "117111141",
            "114444441",
        ),
        (
            "444411111111",
            "411411111111",
            "444444111144",
            "111114111141",
            "111114444441",
        ),
    ),
    (
        (
            "4444444444",
            "4141414741",
            "4141414141",
            "4141414141",
            "1111111111",
            "1111111111",
            "1117111111",
            "1414111111",
            "1414111111",
            "1414111111",
            "1444111111",
        ),
        (
            "4444444444",
            "4141444441",
            "4141444441",
            "4141444441",
            "1111144411",
        ),
    ),
    (
        (
            "44411",
            "41141",
            "41114",
            "44444",
            "11177",
            "11111",
            "77111",
            "44444",
            "41114",
            "14114",
            "11444",
        ),
        (
            "44411111",
            "41141111",
            "41114111",
            "44444111",
            "11144444",
            "11141114",
            "11114114",
            "11111444",
        ),
    ),
)


def _grid_rows(grid):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(input_grid, output_grid):
    return (_grid_rows(input_grid), _grid_rows(output_grid)) in _SOURCE_PAIR_ROWS


def _grid_key(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _puzzle_fingerprint(puzzle):
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _puzzles_equal(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _validate_optional_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(seed, rng, num_train, num_test):
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        test_count = 1 + (seed % 4)
        number_of_legal_train_counts = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((seed // 4) % number_of_legal_train_counts)
        )
        return train_count, test_count

    if train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _shift(cells, dr, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _crop(cells, background, foreground):
    assert cells
    top, bottom, left, right = _bbox(cells)
    grid = np.full((bottom - top + 1, right - left + 1), background, dtype=np.int8)
    for r, c in cells:
        grid[r - top, c - left] = foreground
    return grid


def _parse_input(grid):
    """Enumerate roles from pixels only, including both axes for singletons.

    Missing cells outside the canvas are background. Eight-connectivity assigns
    every foreground cell to exactly one bar; endpoint contacts are incidental.
    The two backing sides must be opposite and their facing sides clear.
    """
    colors = [int(c) for c in np.unique(grid)]
    if len(colors) != 3:
        return []
    by_color = {c: {tuple(map(int, p)) for p in np.argwhere(grid == c)} for c in colors}
    candidates = []
    for marker in colors:
        bars = _components(by_color[marker], False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]):
            continue
        length = len(bars[0])
        for axis in (0, 1):
            # axis zero is horizontal: (normal, tangent) = (row, col).
            convert = lambda cells: {(p[axis], p[1 - axis]) for p in cells}
            oriented_bars = [convert(b) for b in bars]
            if any(len({r for r, c in b}) != 1 or
                   max(c for r, c in b) - min(c for r, c in b) + 1 != length
                   for b in oriented_bars):
                continue
            for background in colors:
                if background == marker:
                    continue
                foreground = next(c for c in colors if c not in (marker, background))
                components = _components(by_color[foreground] | by_color[marker])
                if len(components) != 2:
                    continue
                records = []
                for component in components:
                    own_bars = [b for b in bars if b <= component]
                    if len(own_bars) != 1:
                        break
                    bar = convert(own_bars[0])
                    r0, _, c0, _ = _bbox(bar)
                    fg = convert(component & by_color[foreground])
                    fg = _shift(fg, -r0, -c0)
                    negative = any((-1, c) in fg for c in range(length))
                    positive = any((1, c) in fg for c in range(length))
                    if negative == positive:
                        break
                    records.append((negative, fg, component))
                if len(records) != 2 or records[0][0] == records[1][0]:
                    continue
                records.sort(key=lambda item: not item[0])
                candidates.append({"colors": (background, foreground, marker),
                                   "axis": axis, "bar_length": length,
                                   "anchor": records[0][1], "mover": records[1][1],
                                   "components": [records[0][2], records[1][2]]})
    return candidates


def _predict_model(parsed, model=(-1, "transparent", "union")):
    seam, guides, blend = model
    a, b = parsed["anchor"], _shift(parsed["mover"], seam)
    if blend == "union":
        cells = a | b
    elif blend == "xor":
        cells = a ^ b
    elif blend == "anchor":
        cells = a
    elif blend == "mover":
        cells = b
    else:
        front, back = (a, b) if blend == "anchor_block" else (b, a)
        t, bottom, left, right = _bbox(front)
        cells = front | {(r, c) for r, c in back
                         if not (t <= r <= bottom and left <= c <= right)}
    markers = {(r, c) for r in (0, seam) for c in range(parsed["bar_length"])}
    if guides == "erase":
        cells = cells - markers
    elif guides == "paint":
        cells = cells | markers
    if not cells:
        return None
    bg, fg, _ = parsed["colors"]
    result = _crop(cells, bg, fg)
    return result.T.copy() if parsed["axis"] else result


_MODELS = tuple((seam, guides, blend) for seam in (-1, 0, 1)
                for guides in ("transparent", "erase", "paint")
                for blend in ("union", "xor", "anchor", "mover", "anchor_block", "mover_block"))


def _learn_models(train_pairs):
    models = set(_MODELS)
    for pair in train_pairs:
        candidates = _parse_input(pair["input"])
        assert len(candidates) == 1, "input must identify its palette and normal axis"
        models = {m for m in models if np.array_equal(
            _predict_model(candidates[0], m), pair["output"])}
    assert (-1, "transparent", "union") in models
    return models


def solve(train_pairs, test_input):
    """Prediction uses training labels and test pixels, never test labels or RNG."""
    models = _learn_models(train_pairs)
    parsed = _parse_input(test_input)
    assert len(parsed) == 1, "ambiguous test role assignment"
    predictions = [_predict_model(parsed[0], m) for m in sorted(models)]
    assert predictions and predictions[0] is not None
    assert all(np.array_equal(predictions[0], p) for p in predictions)
    return predictions[0]


def _holes(cells):
    t, b, l, r = _bbox(cells)
    blank = {(y, x) for y in range(t - 1, b + 2) for x in range(l - 1, r + 2)} - cells
    return sum(not any(y in (t - 1, b + 1) or x in (l - 1, r + 1) for y, x in part)
               for part in _components(blank, False))


def _analyze_pair(input_grid, output_grid):
    choices = _parse_input(input_grid)
    assert len(choices) == 1, "ambiguous input geometry/color roles"
    p = choices[0]
    expected = _predict_model(p)
    assert np.array_equal(expected, output_grid), "incorrect translation, union, or crop"
    a, b, length = p["anchor"], _shift(p["mover"], -1), p["bar_length"]
    merged = a | b
    markers = {(r, c) for r in (-1, 0) for c in range(length)}
    t, bottom, left, right = _bbox(merged)
    visible = {q for q in markers if t <= q[0] <= bottom and left <= q[1] <= right}
    bg, fg, _ = p["colors"]
    assert int(np.sum(output_grid == fg)) == len(a) + len(b) - len(a & b)
    boxes = [_bbox(x) for x in p["components"]]
    at, ab, al, ar = boxes[0]
    bt, bb, bl, br = boxes[1]
    xs = ar < bl or br < al
    ys = ab < bt or bb < at
    stage = "diagonal" if xs and ys else "horizontal" if xs else "vertical" if ys else "interleaved"
    side_contacts = []
    for part in (p["anchor"], p["mover"]):
        side_contacts.append(int((0, -1) in part) + 2 * int((0, length) in part))
    components = p["components"]
    minr, maxr, minc, maxc = _bbox(set.union(*components))
    return {"colors": p["colors"], "bar_length": length, "axis": p["axis"],
            "staging": stage, "overlap_count": len(a & b),
            "bbox_overlap": not (_bbox(a)[1] < _bbox(b)[0] or _bbox(b)[1] < _bbox(a)[0]
                                 or _bbox(a)[3] < _bbox(b)[2] or _bbox(b)[3] < _bbox(a)[2]),
            "uncovered_guides": len(markers - merged),
            "visible_uncovered": len(visible - merged), "covered_guides": len(markers & merged),
            "side_contacts": tuple(side_contacts),
            "holes": tuple(_holes(part) for part in (a, b)),
            "four_components": len(_components(set.union(*components), False)),
            "fragment_sizes": (len(a), len(b)),
            "full_width_bar": any(_bbox(part)[3] - _bbox(part)[2] + 1 == length for part in (a, b)),
            "inside_bar": any(_bbox(part)[0] < 0 < _bbox(part)[1] and
                              _bbox(part)[2] < 0 and _bbox(part)[3] >= length
                              for part in (p["anchor"], p["mover"])),
            "thick": any(all((y + dy, x + dx) in part for dy in range(3) for dx in range(3))
                         for part in (a, b) for y, x in part),
            "margins": (minr, input_grid.shape[0] - maxr - 1,
                        minc, input_grid.shape[1] - maxc - 1),
            "marker_at_edge": any(r in (0, input_grid.shape[0] - 1) or c in (0, input_grid.shape[1] - 1)
                                  for r, c in np.argwhere(input_grid == p["colors"][2])),
            "solid_output": bool(np.all(output_grid == fg)),
            "alternative_distinct": not np.array_equal(expected, _predict_model(p, (1, "transparent", "union")))}


def _motif(rng, height, width, style, thickness=1):
    """Compositional strokes; the last row is the potential registration side."""
    h, w, t = height, width, min(thickness, max(1, (min(height, width) - 1) // 2))
    cells = set()
    if style in ("ring", "u", "c"):
        for y in range(h):
            for x in range(w):
                if y >= h - t or (style != "u" and y < t) or x < t or (style != "c" and x >= w - t):
                    cells.add((y, x))
    elif style == "step":
        # A diagonal chamfer remains an eight-connected outline, without repair.
        chamfer = int(rng.integers(1, max(2, min(h, w) - 1)))
        flip = bool(rng.integers(2))
        for y in range(h):
            lo = max(0, chamfer - y)
            for x in range(w):
                if x >= lo and (x < lo + t or x >= w - t or y < t or y >= h - t):
                    cells.add((y, w - 1 - x if flip else x))
    elif style == "arm":
        turn = int(rng.integers(0, w))
        top_end = int(rng.integers(0, w))
        for y in range(h):
            for x in range(w):
                if (y >= h - t or turn <= x < min(w, turn + t) or
                    (y < t and min(turn, top_end) <= x <= max(turn, top_end))):
                    cells.add((y, x))
    elif style == "branch":
        spine = int(rng.integers(w))
        for y in range(h):
            cells.add((y, spine))
            if y == h - 1 or rng.random() < .55:
                lo = int(rng.integers(spine + 1))
                hi = int(rng.integers(spine, w))
                cells.update((y, x) for x in range(lo, hi + 1))
    elif style == "comb":
        spine = int(rng.integers(w))
        for y in range(h):
            lo = int(rng.integers(spine + 1))
            hi = int(rng.integers(spine, w))
            cells.update((y, x) for x in range(lo, hi + 1))
        cells.update((y, x) for y in range(max(0, h - 2), h) for x in range(w))
    else:
        cells = {(y, x) for y in range(h) for x in range(w)}
        # Filled masses may carry an asymmetric notch, or remain truly solid.
        if style == "notched" and h > 2 and w > 2:
            nh, nw = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            cells -= {(y, x) for y in range(nh) for x in range(nw)}
    return cells


_STYLES = ("ring", "step", "u", "c", "arm", "branch", "filled", "notched", "comb")


def _ordinary_fragment(rng, length, scale, sign, style=None, full=False):
    width = length if full else int(rng.integers(max(length, 2), max(length + 2, scale + 4)))
    height = int(rng.integers(2, scale + 2))
    offset = int(rng.integers(width - length + 1))
    style = style or str(rng.choice(_STYLES))
    thickness = int(rng.choice([1, 1, 1, 2, 3]))
    mask = _motif(rng, height, width, style, thickness)
    # Branches have a deliberately variable amount of backing.
    mask.update((height - 1, offset + c) for c in range(length))
    return {(sign * (height - y), x - offset) for y, x in mask}


def _geometry(rng, length, scale, kind, collision, uncovered, simple=False):
    markers = {(0, c) for c in range(length)}
    if kind == "nested":
        # The anchor's bar lies inside a closed cavity; the other piece may fit
        # wholly inside it, or deliberately cross a wall in a taught union case.
        w = int(rng.integers(max(length, 2), max(length + 2, scale + 2)))
        h = int(rng.integers(2, scale + 1))
        offset = int(rng.integers(w - length + 1))
        padl, padr = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        t = int(rng.choice([1, 1, 2]))
        l, r = -offset - padl - t + 1, w - offset + padr + t - 2
        bottom = h + int(rng.integers(0, 3))
        a = {(y, x) for y in range(-t, bottom + t) for x in range(l, r + 1)
             if y < 0 or y >= bottom or x < l + t or x > r - t}
        mover_h = bottom + t + int(rng.integers(1, 3)) if collision else h
        b = {(mover_h - y, x - offset) for y, x in _motif(rng, mover_h, w, str(rng.choice(_STYLES)))}
        b.update((1, c) for c in range(length))
    elif kind in ("interlock", "weave_witness"):
        l = -int(rng.integers(1, 5))
        r = length + int(rng.integers(1, 5))
        h1, h2 = int(rng.integers(2, scale + 1)), int(rng.integers(2, scale + 1))
        boundary = {x: -1 if 0 <= x < length else int(rng.integers(-h1 + 1, h2 - 1))
                    for x in range(l, r)}
        if kind == "weave_witness":
            # Foreground from each piece lies under the other's bbox. Thus
            # neither order of opaque rectangle pasting can imitate the union.
            boundary[l], boundary[r - 1] = 0, -2
        a = {(y, x) for x in range(l, r) for y in range(-h1, boundary[x] + 1)}
        b = {(y + 1, x) for x in range(l, r) for y in range(boundary[x] + 1, h2)}
        if collision:
            side = l if rng.integers(2) else r - 1
            b.update((y + 1, side) for y in range(-h1, boundary[side] + 1))
        # Small cavities away from the registration seam diversify filled pieces.
        if h1 >= 4 and r - l > 4 and rng.random() < .6:
            a.discard((-h1 + 1, l + 1))
        if h2 >= 4 and r - l > 4 and rng.random() < .6:
            b.discard((h2 - 1, r - 2))
    else:
        style_a = str(rng.choice(("ring", "u", "arm", "filled") if simple else _STYLES))
        style_b = str(rng.choice(("ring", "u", "filled") if simple else _STYLES))
        full = rng.random() < .22
        a = _ordinary_fragment(rng, length, scale, -1, style_a, full)
        b = _ordinary_fragment(rng, length, scale, 1, style_b, full and rng.random() < .6)
        # Incidental extensions are sampled independently on either endpoint.
        # Shared bridges retain the reviewed comb subclass, without requiring it.
        contacts = [int(rng.integers(4)) for _ in range(2)] if not simple else [0, 0]
        if collision:
            shared = int(rng.integers(2))
            contacts = [v | (1 << shared) for v in contacts]
        for i, cells in enumerate((a, b)):
            for bit, x in enumerate((-1, length)):
                if contacts[i] & (1 << bit):
                    # A short horizontal stroke connects the side arm to backing.
                    back = -1 if i == 0 else 1
                    cells.add((back, x))
                    cells.add((back, 0 if x == -1 else length - 1))
                    depth = int(rng.integers(1, scale + 1)) if collision else 0
                    cells.update(((-back) * y, x) for y in range(depth + 1))
        if not collision and a & _shift(b, -1):
            # Keep no-collision ordinary pieces as pure abutments if incidental
            # endpoint pixels would otherwise collide with one another.
            a = {q for q in a if q[0] < 0}
            b = {q for q in b if q[0] > 0}
    if uncovered:
        # Delete a variable subset of backing cells; connectivity and visibility
        # are checked from the rendered pixels, not assumed from construction.
        for cells, row in ((a, -1), (b, 1)):
            n = int(rng.integers(0, length))
            if n:
                for x in rng.choice(length, n, replace=False):
                    cells.discard((row, int(x)))
        if all((-1, c) in a and (1, c) in b for c in range(length)):
            a.discard((-1, int(rng.integers(length))))
    if a & markers or b & markers:
        return None
    if len(_components(a | markers)) != 1 or len(_components(b | markers)) != 1:
        return None
    return a, b


def _stage(rng, a, b, length, mode, palette, padding):
    marker = {(0, c) for c in range(length)}
    groups = []
    for fg in (a, b):
        t, bot, l, r = _bbox(fg | marker)
        groups.append((_shift(fg, -t, -l), _shift(marker, -t, -l), bot - t + 1, r - l + 1))
    if rng.integers(2):
        groups.reverse()
    af, am, ah, aw = groups[0]
    bf, bm, bh, bw = groups[1]
    aset, bset = af | am, bf | bm
    avoid = {(r + dy, c + dx) for r, c in aset for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    offset = None
    if mode == "interleaved":
        # Search only bbox-overlapping offsets, allowing genuine interleaving.
        options = [(dy, dx) for dy in range(-bh + 1, ah) for dx in range(-bw + 1, aw)]
        for i in rng.permutation(len(options)):
            dy, dx = options[int(i)]
            if _shift(bset, dy, dx).isdisjoint(avoid):
                offset = (dy, dx)
                break
    if offset is None:
        gap = int(rng.integers(1, 9))
        if mode == "horizontal":
            offset = (int(rng.integers(-bh + 1, ah)), aw + gap)
        elif mode == "diagonal":
            offset = (ah + gap, aw + int(rng.integers(1, 7)))
        else:
            offset = (ah + gap, int(rng.integers(-bw + 1, aw)))
    dy, dx = offset
    foreground = af | _shift(bf, dy, dx)
    guides = am | _shift(bm, dy, dx)
    if not _shift(bset, dy, dx).isdisjoint(avoid):
        return None
    t, bot, l, r = _bbox(foreground | guides)
    height, width = bot - t + 1, r - l + 1
    if max(height, width) > 30:
        return None
    if padding == "tight":
        pads = [0, 0, 0, 0]
    else:
        pads = [int(rng.integers(0, 6)) for _ in range(4)]
    pads[0] = min(pads[0], 30 - height)
    pads[1] = min(pads[1], 30 - height - pads[0])
    pads[2] = min(pads[2], 30 - width)
    pads[3] = min(pads[3], 30 - width - pads[2])
    if padding == "large":
        if rng.integers(2):
            pads[1] = 30 - height - pads[0]
        else:
            pads[3] = 30 - width - pads[2]
    bg, fg, mc = palette
    grid = np.full((height + pads[0] + pads[1], width + pads[2] + pads[3]), bg, dtype=np.int8)
    for cells, color in ((foreground, fg), (guides, mc)):
        for y, x in cells:
            grid[y - t + pads[0], x - l + pads[2]] = color
    return grid


def _make_pair(rng, profile, used):
    for attempt in range(160):
        # A bounded fallback retains the requested semantics while using an easy
        # ring/arm construction and disjoint staging, never a stored source pair.
        kind = profile["kind"] if attempt < 100 else "ordinary"
        geometry = _geometry(rng, profile["length"], profile["scale"], kind,
                             profile["collision"], profile["uncovered"], profile["simple"])
        if geometry is None:
            continue
        a, b = geometry
        grid = _stage(rng, a, b, profile["length"], profile["staging"] if attempt < 100 else "horizontal",
                      profile["palette"], profile["padding"])
        if grid is None:
            continue
        output = _crop(a | _shift(b, -1), *profile["palette"][:2])
        k = profile["rotation"]
        pair = {"input": np.rot90(grid, k).copy(), "output": np.rot90(output, k).copy()}
        if max(pair["output"].shape) > 30 or _is_source_pair(pair["input"], pair["output"]):
            continue
        if _grid_key(pair["input"]) in used:
            continue
        try:
            info = _analyze_pair(pair["input"], pair["output"])
        except AssertionError:
            continue
        if bool(info["overlap_count"]) != profile["collision"]:
            continue
        if bool(info["visible_uncovered"]) != profile["uncovered"]:
            continue
        if not info["alternative_distinct"] or not info["covered_guides"]:
            continue
        return pair
    raise AssertionError("bounded geometric construction exhausted")


def _sample_parameters(rng):
    difficulty = str(rng.choice(["simple", "moderate", "moderate", "rich"]))
    return {"simple": difficulty == "simple",
            "changing_palette": difficulty == "rich" or (difficulty == "moderate" and rng.random() < .5),
            "mixed_axis": difficulty == "rich" or (difficulty == "moderate" and rng.random() < .4),
            "palette": tuple(map(int, rng.choice(10, 3, replace=False))),
            "rotation": int(rng.integers(4)),
            "extensions": difficulty != "simple" and rng.random() < .85,
            "regime": str(rng.choice(["shared", "shared", "larger", "smaller", "interpolation", "placement"])),
            "staging": str(rng.choice(["horizontal", "vertical", "diagonal", "interleaved"]))}


def _profiles(rng, params, count, regime):
    training = regime == "source_train"
    profiles = []
    for index in range(count):
        scale = int(rng.integers(3, 8))
        if params["regime"] == "larger":
            scale = int(rng.integers(3, 5) if training else rng.integers(6, 10))
        elif params["regime"] == "smaller":
            scale = int(rng.integers(6, 9) if training else rng.integers(2, 4))
        length = int(rng.integers(1, 7))
        if params["simple"]:
            length = int(rng.integers(1, 5))
        if params["regime"] == "interpolation":
            length = (1 if index % 2 == 0 else 3) if training else 2
        collision = params["extensions"] and rng.random() < .5
        uncovered = params["extensions"] and length > 1 and rng.random() < .45
        if training and index == 0:
            collision, uncovered = False, False
        if training and index == 1 and params["extensions"]:
            collision, uncovered = True, True
            length = max(2, length)
        palette = tuple(map(int, rng.choice(10, 3, replace=False))) if params["changing_palette"] else params["palette"]
        rotation = params["rotation"]
        if params["mixed_axis"]:
            # Both axes appear in the first two demonstrations. Flipping a whole
            # scene does not change the attachment-based mating order.
            rotation = (rotation + index % 2 + 2 * int(rng.integers(2))) % 4
        staging = str(rng.choice(["horizontal", "vertical", "diagonal", "interleaved"]))
        if params["regime"] == "placement" and training:
            staging = params["staging"]
        kind = "ordinary" if params["simple"] else str(rng.choice(["ordinary", "ordinary", "nested", "interlock"]))
        profiles.append({"length": length, "scale": scale, "kind": kind,
                         "collision": bool(collision), "uncovered": bool(uncovered),
                         "simple": params["simple"], "palette": palette,
                         "rotation": rotation, "staging": staging,
                         "padding": str(rng.choice(["tight", "random", "random", "large"]))})
    return profiles


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    ntrain, ntest = _resolve_counts(int(seed), rng, num_train, num_test)
    params = _sample_parameters(rng)
    puzzle, used = {"train": [], "test": []}, set()
    train_profiles = None
    for split, regime, count in (("train", "source_train", ntrain), ("test", "target_test", ntest)):
        profiles = _profiles(rng, params, count, regime)
        if split == "train":
            train_profiles = profiles
        for profile in profiles:
            pair = _make_pair(rng, profile, used)
            used.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
    # Repair only a missing demonstration, without restricting test geometry.
    # Inference uses training labels; test labels are used only for comparison.
    for attempt in range(32):
        models = _learn_models(puzzle["train"])
        if all(all(np.array_equal(_predict_model(_parse_input(pair["input"])[0], m), pair["output"])
                   for m in models) for pair in puzzle["test"]):
            break
        index = attempt % ntrain
        profile = dict(train_profiles[index])
        profile["kind"] = "weave_witness"
        pair = _make_pair(rng, profile, used)
        used.add(_grid_key(pair["input"]))
        puzzle["train"][index] = pair
    else:
        raise AssertionError("could not expose all assembly conventions")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys, infos = [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _is_source_pair(pair["input"], pair["output"])
            keys.append(_grid_key(pair["input"]))
            infos[split].append(_analyze_pair(pair["input"], pair["output"]))
    assert len(set(keys)) == len(keys)
    assert all(i["alternative_distinct"] for i in infos["train"])
    train_axes = {i["axis"] for i in infos["train"]}
    assert {i["axis"] for i in infos["test"]} <= train_axes
    if any(i["overlap_count"] for i in infos["test"]):
        assert any(i["overlap_count"] for i in infos["train"])
    if any(i["visible_uncovered"] for i in infos["test"]):
        assert any(i["visible_uncovered"] and i["covered_guides"] for i in infos["train"])
    # This finite hypothesis check distinguishes seam order, transparent guides,
    # accidental marker painting/erasure, collision cancellation, and dropping a
    # fragment whenever those alternatives would affect any requested test.
    models = _learn_models(puzzle["train"])
    for pair in puzzle["test"]:
        parsed = _parse_input(pair["input"])[0]
        predictions = [_predict_model(parsed, m) for m in models]
        assert all(np.array_equal(p, pair["output"]) for p in predictions)


def self_test() -> None:
    # All six source regressions are retained solely for checking/exclusion.
    source = [{"input": np.array([[int(c) for c in row] for row in inp], dtype=np.int8),
               "output": np.array([[int(c) for c in row] for row in out], dtype=np.int8)}
              for inp, out in _SOURCE_PAIR_ROWS]
    for pair in source:
        info = _analyze_pair(pair["input"], pair["output"])
        assert info["overlap_count"] == info["uncovered_guides"] == 0
    for pair in source[4:]:
        assert np.array_equal(solve(source[:4], pair["input"].copy()), pair["output"])
    train_counts, test_counts, fingerprints = set(), set(), set()
    inputs, outputs, structures, palettes = set(), set(), set(), set()
    all_info, within = [], []
    regime_evidence = []
    mutant_checks = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        info = []
        for split in puzzle:
            for pair in puzzle[split]:
                p = _parse_input(pair["input"])[0]
                i = _analyze_pair(pair["input"], pair["output"])
                info.append(i)
                inputs.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                palettes.add(i["colors"])
                structures.add((_grid_key(_crop(p["anchor"], 0, 1)), _grid_key(_crop(p["mover"], 0, 1))))
                # Negative semantic checks: the analyzer must reject a wrong
                # seam, marker overpainting/erasure, or collision cancellation.
                for mutant in ((1, "transparent", "union"), (-1, "erase", "union"),
                               (-1, "paint", "union"), (-1, "transparent", "xor"),
                               (-1, "transparent", "anchor_block"), (-1, "transparent", "mover_block")):
                    if mutant in mutant_checks:
                        continue
                    wrong = _predict_model(p, mutant)
                    if wrong is not None and not np.array_equal(wrong, pair["output"]):
                        try:
                            _analyze_pair(pair["input"], wrong)
                        except AssertionError:
                            mutant_checks.add(mutant)
                        else:
                            raise AssertionError("incorrect semantic alternative accepted")
        all_info.extend(info)
        within.append(info)
        train_info = info[:len(puzzle["train"])]
        test_info = info[len(puzzle["train"]):]
        train_areas = [x["output"].size for x in puzzle["train"]]
        test_areas = [x["output"].size for x in puzzle["test"]]
        tl = {x["bar_length"] for x in train_info}
        ql = {x["bar_length"] for x in test_info}
        regime_evidence.append((min(test_areas) > max(train_areas),
                                max(test_areas) < min(train_areas), bool(tl & ql),
                                {1, 3} <= tl and 2 not in tl and 2 in ql,
                                len({x["staging"] for x in test_info}) > 1))
        for pair in puzzle["test"]:
            predicted = solve([{k: v.copy() for k, v in x.items()} for x in puzzle["train"]], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(fingerprints) == 42
    assert len(inputs) >= 50 and len(outputs) >= 40 and len(structures) >= 180 and len(palettes) >= 40
    assert {i["axis"] for i in all_info} == {0, 1}
    assert {i["staging"] for i in all_info} == {"horizontal", "vertical", "diagonal", "interleaved"}
    assert {i["bar_length"] for i in all_info} >= set(range(1, 7))
    assert sum(i["overlap_count"] == 0 for i in all_info) > 80
    assert sum(i["overlap_count"] > 0 for i in all_info) > 20
    assert sum(i["overlap_count"] > 5 for i in all_info) > 5
    assert len({i["uncovered_guides"] for i in all_info}) >= 5
    assert sum(any(i["holes"]) for i in all_info) > 50
    assert sum(i["four_components"] > 2 for i in all_info) >= 5
    assert sum(i["inside_bar"] for i in all_info) >= 10
    assert sum(i["full_width_bar"] for i in all_info) >= 15
    assert sum(i["solid_output"] for i in all_info) >= 2
    assert sum(i["marker_at_edge"] for i in all_info) >= 5
    assert sum(i["thick"] for i in all_info) > 20
    assert len({i["side_contacts"] for i in all_info}) >= 8
    assert sum(i["overlap_count"] == 0 and i["bbox_overlap"] for i in all_info) > 10
    assert sum(len({i["colors"] for i in infos}) > 1 for infos in within) >= 8
    assert sum(len({i["axis"] for i in infos}) > 1 for infos in within) >= 8
    assert sum(len({i["colors"] for i in infos}) == len({i["axis"] for i in infos}) == 1 for infos in within) >= 5
    assert all(len({i["fragment_sizes"] for i in infos}) > 1 for infos in within)
    assert len(mutant_checks) == 6
    assert all(any(row[column] for row in regime_evidence) for column in range(5))
    assert sum(0 in i["margins"] for i in all_info) > 50
    assert any(min(shape) <= 3 for shape in inputs) and any(30 in shape for shape in inputs)
    assert max(max(i["fragment_sizes"]) / min(i["fragment_sizes"]) for i in all_info) >= 5
    assert any(len({i["axis"] for i in infos}) > 1 and
               len({i["colors"] for i in infos}) > 1 and
               any(any(i["holes"]) and 0 in i["margins"] for i in infos)
               for infos in within)
    for args in ((None, None), (7, 3), (2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(713, *args)
        assert _puzzles_equal(puzzle, generate_puzzle(713, *args))
        validate_generated_puzzle(puzzle)
        for split, count in zip(("train", "test"), args):
            assert count is None or len(puzzle[split]) == count
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(1, *args)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
