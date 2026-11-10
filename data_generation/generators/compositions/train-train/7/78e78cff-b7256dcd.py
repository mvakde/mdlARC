"""Mirror-compatible outlets. Standard library + NumPy; no parent imports."""
from __future__ import annotations

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _normal(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _reflect(cell, axis):
    r, c = cell
    kind, k = axis
    if kind == "H":
        return k - r, c
    if kind == "V":
        return r, k - c
    if kind == "D":
        return c + k, r - k
    if kind == "A":
        return k - c, k - r
    raise ValueError("unknown axis")


def _axes(cells):
    cells = frozenset(cells)
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    candidates = [("H", r0 + r1), ("V", c0 + c1)]
    if r1 - r0 == c1 - c0:
        candidates += [("D", r0 - c0), ("A", r0 + c1)]
    return tuple(a for a in candidates if frozenset(_reflect(p, a) for p in cells) == cells)


def _half_turn(cells):
    rsum = min(r for r, c in cells) + max(r for r, c in cells)
    csum = min(c for r, c in cells) + max(c for r, c in cells)
    return frozenset((rsum - r, csum - c) for r, c in cells) == cells


def _connected(cells):
    unseen = set(cells)
    stack = [unseen.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            p = r + dr, c + dc
            if p in unseen:
                unseen.remove(p)
                stack.append(p)
    return not unseen


def _gap_cells(box, gap):
    top, bottom, left, right = box
    side, start, width = gap
    if side in ("top", "bottom"):
        line = top if side == "top" else bottom
        return frozenset((line, c) for c in range(start, start + width))
    line = left if side == "left" else right
    return frozenset((r, line) for r in range(start, start + width))


def _active_by_axis(box, gaps, axes):
    sets = [_gap_cells(box, gap) for gap in gaps]
    existing = set(sets)
    return tuple(frozenset(i for i, cells in enumerate(sets)
                           if frozenset(_reflect(p, a) for p in cells) in existing)
                 for a in axes)


def _active(box, gaps, axes):
    return frozenset().union(*_active_by_axis(box, gaps, axes))


def _parse(grid):
    a = _grid(grid)
    bg = int(a[0, 0])
    _require(bool(np.all(a[0] == bg) and np.all(a[-1] == bg)
                  and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)), "unclear outer border")
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 4, "four role colors required")
    singleton = [int(c) for c, n in zip(colors, counts) if c != bg and n == 1]
    _require(len(singleton) == 1, "one singleton paint marker required")
    paint = singleton[0]
    choices = []
    for color in colors:
        color = int(color)
        if color in (bg, paint):
            continue
        points = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        top, bottom = min(r for r, c in points), max(r for r, c in points)
        left, right = min(c for r, c in points), max(c for r, c in points)
        if bottom - top < 2 or right - left < 2:
            continue
        if not all(r in (top, bottom) or c in (left, right) for r, c in points):
            continue
        if not all(a[r, c] == color for r in (top, bottom) for c in (left, right)):
            continue
        other = np.argwhere((a != bg) & (a != color))
        if not all(top < r < bottom and left < c < right for r, c in other):
            continue
        choices.append((color, (top, bottom, left, right)))
    _require(len(choices) == 1, "frame must be uniquely enclosing")
    frame, box = choices[0]
    top, bottom, left, right = box
    neutral = next(int(c) for c in colors if c not in (bg, frame, paint))
    motif = frozenset(tuple(map(int, p)) for p in np.argwhere(a == neutral))
    _require(len(motif) >= 4 and _connected(motif), "one intact connected motif required")
    marker = tuple(map(int, np.argwhere(a == paint)[0]))
    gaps = []
    for side in SIDES:
        horizontal = side in ("top", "bottom")
        low, high = (left + 1, right) if horizontal else (top + 1, bottom)
        line = top if side == "top" else bottom if side == "bottom" else left if side == "left" else right
        values = a[line, low:high] if horizontal else a[low:high, line]
        _require(bool(np.all((values == bg) | (values == frame))), "invalid side cell")
        _require(values[0] == frame and values[-1] == frame, "corner guards required")
        start = None
        for offset, value in enumerate(list(values) + [frame]):
            if value == bg and start is None:
                start = offset
            elif value == frame and start is not None:
                gaps.append((side, low + start, offset - start))
                start = None
    _require(len(gaps) >= 2, "at least two openings required")
    # The enclosing test above also forbids every unrelated exterior cell.
    return {"bg": bg, "frame": frame, "neutral": neutral, "paint": paint,
            "box": box, "motif": motif, "marker": marker, "gaps": tuple(gaps),
            "axes": _axes(motif)}


def _answer(grid, box, marker, gaps, axes):
    out = grid.copy()
    top, bottom, left, right = box
    paint = int(grid[marker])
    out[marker] = int(grid[0, 0])
    if not axes:
        return out
    out[top + 1:bottom, left + 1:right] = paint
    for i in _active(box, gaps, axes):
        side, start, width = gaps[i]
        if side == "top":
            out[:top + 1, start:start + width] = paint
        elif side == "bottom":
            out[bottom:, start:start + width] = paint
        elif side == "left":
            out[start:start + width, :left + 1] = paint
        else:
            out[start:start + width, right:] = paint
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every decision from the input's pixels alone; never mutate input."""
    a = _grid(input_grid)
    m = _parse(a)
    return _answer(a, m["box"], m["marker"], m["gaps"], m["axes"])


def _orbit(cell, transforms):
    result, stack = {cell}, [cell]
    while stack:
        p = stack.pop()
        for axis in transforms:
            q = _reflect(p, axis)
            if q not in result:
                result.add(q)
                stack.append(q)
    return result


def _motif(rng, kind):
    if kind == "rotational":
        width = int(rng.integers(2, 7))
        return frozenset([(0, c) for c in range(1, width + 1)]
                         + [(1, c) for c in range(width)])
    desired = {"cart": {"H"}, "diag": {"D"},
               "double": {"H", "V"}, "all": {"H", "V", "D", "A"},
               "asymmetric": set()}[kind]
    for _ in range(300):
        h = int(rng.integers(3, 9))
        w = h if kind in ("diag", "all") else int(rng.integers(3, 9))
        transforms = []
        if "H" in desired:
            transforms.append(("H", h - 1))
        if "V" in desired:
            transforms.append(("V", w - 1))
        if "D" in desired:
            transforms.append(("D", 0))
        start = ((h - 1) // 2, (w - 1) // 2)
        cells = _orbit(start, transforms)
        # Orbit of the central cell/block is connected; every added orbit
        # attaches to its reflection of the previous connected set.
        for _ in range(int(rng.integers(3, 15))):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIRS
                               if 0 <= r + dr < h and 0 <= c + dc < w
                               and (r + dr, c + dc) not in cells})
            if not frontier:
                break
            cells.update(_orbit(frontier[int(rng.integers(len(frontier)))], transforms))
        cells = _normal(cells)
        if len(cells) >= 4 and {a[0] for a in _axes(cells)} == desired and _connected(cells):
            return cells
    raise RuntimeError("bounded motif sampling exhausted")


def _candidates(box):
    top, bottom, left, right = box
    result = []
    for side in SIDES:
        low, high = (left + 2, right - 1) if side in ("top", "bottom") else (top + 2, bottom - 1)
        for width in range(1, min(4, high - low) + 1):
            for start in range(low, high - width + 1):
                result.append((side, start, width))
    return result


def _compatible(gap, selected):
    side, start, width = gap
    for other_side, other_start, other_width in selected:
        if gap == (other_side, other_start, other_width):
            continue
        if side == other_side and not (start + width < other_start or other_start + other_width < start):
            return False
    return True


def _sample_gaps(rng, box, axes, kind):
    candidates = _candidates(box)
    lookup = {_gap_cells(box, gap): gap for gap in candidates}
    groups = []
    for axis in axes:
        options = []
        for gap in candidates:
            image = frozenset(_reflect(p, axis) for p in _gap_cells(box, gap))
            mate = lookup.get(image)
            if mate is not None and _compatible(gap, [mate]):
                if kind == "cart" and (gap == mate or gap[0] != mate[0] or gap[2] < 2):
                    continue
                if kind == "diag" and (gap == mate or gap[2] < 2):
                    continue
                options.append(tuple(dict.fromkeys((gap, mate))))
        groups.append(options)
    for _ in range(300):
        selected = []
        if axes:
            if kind in ("double", "all"):
                order = rng.permutation(len(axes))
                for j in order:
                    choices = groups[int(j)]
                    if not choices:
                        continue
                    for trial in rng.permutation(len(choices))[:30]:
                        group = choices[int(trial)]
                        if all(_compatible(g, selected) for g in group):
                            selected.extend(g for g in group if g not in selected)
                            break
            else:
                choices = groups[0]
                if not choices:
                    return None
                selected = list(choices[int(rng.integers(len(choices)))])
                if kind == "cart":
                    fixed = [g for g in candidates if g not in selected
                             and _compatible(g, selected)
                             and frozenset(_reflect(p, axes[0]) for p in _gap_cells(box, g)) == _gap_cells(box, g)]
                    if not fixed:
                        continue
                    selected.append(fixed[int(rng.integers(len(fixed)))])
                if kind == "diag":
                    # Teach exact equality with a pair whose reflected spans
                    # overlap but are shifted by one cell. Both stay empty.
                    false_pairs = []
                    candidate_set = set(candidates)
                    for gap in candidates:
                        if gap[2] < 2 or gap in selected or not _compatible(gap, selected):
                            continue
                        image = frozenset(_reflect(p, axes[0]) for p in _gap_cells(box, gap))
                        mate = lookup.get(image)
                        if mate is None:
                            continue
                        side, start, width = mate
                        for delta in (-1, 1):
                            fake = side, start + delta, width
                            if fake in candidate_set and fake not in selected and _compatible(fake, selected + [gap]):
                                false_pairs.append((gap, fake))
                    if not false_pairs:
                        continue
                    selected.extend(false_pairs[int(rng.integers(len(false_pairs)))])
        target = int(rng.integers(max(3, len(selected) + 1), max(4, len(selected) + 2, 9)))
        for j in rng.permutation(len(candidates)):
            gap = candidates[int(j)]
            if gap not in selected and _compatible(gap, selected):
                selected.append(gap)
                if len(selected) >= target:
                    break
        if len(selected) < 3:
            continue
        selected = tuple(sorted(selected, key=lambda g: (SIDES.index(g[0]), g[1])))
        active = _active(box, selected, axes)
        if not axes:
            return selected
        if not active or len(active) == len(selected) or not any(selected[i][2] >= 2 for i in active):
            continue
        if kind in ("double", "all"):
            by_axis = _active_by_axis(box, selected, axes)
            if any(s == active for s in by_axis):
                continue
        if kind == "diag" and not _has_partial_overlap(box, selected, axes[0]):
            continue
        return selected
    return None


def _make_pair(rng, kind, palette):
    bg, frame, neutral, paint = palette
    for _ in range(180):
        motif = _motif(rng, kind)
        mh = max(r for r, c in motif) + 1
        mw = max(c for r, c in motif) + 1
        if kind == "diag":
            before, after = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            pads = (before, after, before, after)
        elif kind == "all":
            p = int(rng.integers(2, 5))
            pads = (p, p, p, p)
        elif kind == "double":
            pr, pc = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            pads = (pr, pr, pc, pc)
        else:
            pads = tuple(int(x) for x in rng.integers(2, 6, size=4))
            if kind == "cart" and pads[0] == pads[1]:
                continue
        pt, pb, pl, pr = pads
        ch, cw = mh + pt + pb, mw + pl + pr
        mt, mb, ml, mr = (int(x) for x in rng.integers(1, 5, size=4))
        height, width = ch + 2 + mt + mb, cw + 2 + ml + mr
        if max(height, width) > 30:
            continue
        box = mt, mt + ch + 1, ml, ml + cw + 1
        top, bottom, left, right = box
        absolute = frozenset((top + 1 + pt + r, left + 1 + pl + c) for r, c in motif)
        axes = _axes(absolute)
        gaps = _sample_gaps(rng, box, axes, kind)
        if gaps is None:
            continue
        grid = np.full((height, width), bg, dtype=np.int8)
        grid[top, left:right + 1] = frame
        grid[bottom, left:right + 1] = frame
        grid[top:bottom + 1, left] = frame
        grid[top:bottom + 1, right] = frame
        for gap in gaps:
            for cell in _gap_cells(box, gap):
                grid[cell] = bg
        for cell in absolute:
            grid[cell] = neutral
        empty = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)
                 if grid[r, c] == bg]
        marker = empty[int(rng.integers(len(empty)))]
        grid[marker] = paint
        output = _answer(grid, box, marker, gaps, axes)
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        grid, output = np.rot90(grid, turns), np.rot90(output, turns)
        if flip:
            grid, output = np.fliplr(grid), np.fliplr(output)
        return {"input": grid.copy(), "output": output.copy()}
    raise RuntimeError("bounded chamber sampling exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name + " outside legal integer range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle with exact requested counts."""
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    anchor = tuple(int(x) for x in rng.choice(10, 4, replace=False))
    variable_palette = bool(rng.integers(2))
    first = ["cart", "diag"]
    rng.shuffle(first)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            if split == "train":
                kind = first[index] if index < 2 else "rotational" if index == 2 else str(rng.choice(("cart", "diag", "rotational", "asymmetric")))
            else:
                kind = str(rng.choice(("double", "all")))
            for _ in range(30):
                palette = tuple(int(x) for x in rng.choice(10, 4, replace=False)) if variable_palette else anchor
                if split == "train" and index == 1 and variable_palette:
                    while palette == first_palette:
                        palette = tuple(int(x) for x in rng.choice(10, 4, replace=False))
                pair = _make_pair(rng, kind, palette)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_palette = palette
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _paint_mask(shape, box, gaps, axes):
    """Cellwise membership mask used by validation, including exact port spans."""
    top, bottom, left, right = box
    active = _active(box, gaps, axes)
    mask = np.zeros(shape, dtype=bool)
    if not axes:
        return mask
    for r in range(shape[0]):
        for c in range(shape[1]):
            selected = top < r < bottom and left < c < right
            for i in active:
                side, start, width = gaps[i]
                selected |= ((side == "top" and r <= top and start <= c < start + width)
                             or (side == "bottom" and r >= bottom and start <= c < start + width)
                             or (side == "left" and c <= left and start <= r < start + width)
                             or (side == "right" and c >= right and start <= r < start + width))
            mask[r, c] = selected
    return mask


def _has_partial_overlap(box, gaps, axis):
    sets = [_gap_cells(box, g) for g in gaps]
    return any(image != other and bool(image & other)
               for cells in sets
               for image in (frozenset(_reflect(p, axis) for p in cells),)
               for other in sets)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, details = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and _key(inp) not in seen, "shape change or duplicate")
            seen.add(_key(inp))
            m = _parse(inp)
            mask = _paint_mask(inp.shape, m["box"], m["gaps"], m["axes"])
            expected = inp.copy()
            expected[m["marker"]] = m["bg"]
            expected[mask] = m["paint"]
            _require(np.array_equal(expected, out), "incorrect paint or preservation mask")
            _require(np.array_equal(out[inp == m["frame"]], inp[inp == m["frame"]]), "damaged frame")
            _require(not np.array_equal(inp, out), "identity example")
            if not m["axes"]:
                _require(np.all(out[inp == m["neutral"]] == m["neutral"]), "trimmed negative motif")
            else:
                active = _active(m["box"], m["gaps"], m["axes"])
                # A-only: fill and extrude every gap. It must differ.
                # Build this counterfactual directly, irrespective of reflection.
                top, bottom, left, right = m["box"]
                a_only = inp.copy()
                a_only[top + 1:bottom, left + 1:right] = m["paint"]
                for i, gap in enumerate(m["gaps"]):
                    side, start, width = gap
                    if side == "top":
                        a_only[:top + 1, start:start + width] = m["paint"]
                    elif side == "bottom":
                        a_only[bottom:, start:start + width] = m["paint"]
                    elif side == "left":
                        a_only[start:start + width, :left + 1] = m["paint"]
                    else:
                        a_only[start:start + width, right:] = m["paint"]
                _require(not np.array_equal(a_only, out), "B has no observable contribution")
                b_only = inp.copy()
                b_only[m["marker"]] = m["bg"]
                b_only[inp == m["neutral"]] = m["paint"]
                _require(not np.array_equal(b_only, out), "A has no observable contribution")
                _require(bool(active) and any(m["gaps"][i][2] >= 2 for i in active), "outflow/width witness missing")
            details[split].append(m)
    first_two = details["train"][:2]
    _require(sorted(len(m["axes"]) for m in first_two) == [1, 1], "two primitive axis demonstrations required")
    _require({m["axes"][0][0] in ("H", "V") for m in first_two} == {True, False}, "Cartesian and diagonal evidence required")
    cart = next(m for m in first_two if m["axes"][0][0] in ("H", "V"))
    kind, k = cart["axes"][0]
    top, bottom, left, right = cart["box"]
    _require(k != (top + bottom if kind == "H" else left + right), "displaced axis evidence missing")
    gapsets = [_gap_cells(cart["box"], g) for g in cart["gaps"]]
    _require(any(i != j and cart["gaps"][i][0] == cart["gaps"][j][0]
                 and frozenset(_reflect(p, cart["axes"][0]) for p in a) == b
                 for i, a in enumerate(gapsets) for j, b in enumerate(gapsets)), "same-side pair evidence missing")
    _require(any(frozenset(_reflect(p, cart["axes"][0]) for p in a) == a for a in gapsets), "self-match evidence missing")
    diag = next(m for m in first_two if m["axes"][0][0] in ("D", "A"))
    _require(_has_partial_overlap(diag["box"], diag["gaps"], diag["axes"][0]), "partial-overlap contrast missing")
    if nt >= 3:
        m = details["train"][2]
        _require(not m["axes"] and _half_turn(m["motif"]), "rotational-only negative required")
    for m in details["test"]:
        by_axis = _active_by_axis(m["box"], m["gaps"], m["axes"])
        union = frozenset().union(*by_axis)
        _require(len(m["axes"]) >= 2 and all(s != union for s in by_axis), "test must require union of axes")
    palettes = [{tuple(m[k] for k in ("bg", "frame", "neutral", "paint")) for m in details[s]} for s in ("train", "test")]
    if len(palettes[0]) == 1:
        _require(palettes[1] <= palettes[0], "palette reassignment lacks training evidence")


def _regressions():
    # Exact whole-set equality, partial overlap, unequal width, fixed ports,
    # and same-side partners are tested without random sampling.
    box = (2, 12, 2, 12)
    h = (("H", 14),)
    gaps = (("top", 5, 3), ("bottom", 6, 3))
    _require(not _active(box, gaps, h), "partial overlap must fail")
    gaps = (("top", 5, 3), ("bottom", 5, 4))
    _require(not _active(box, gaps, h), "containment must fail")
    gaps = (("top", 5, 3), ("bottom", 5, 3), ("left", 6, 3))
    _require(_active(box, gaps, h) == {0, 1, 2}, "exact/self matches must pass")
    gaps = (("left", 4, 2), ("left", 9, 2), ("right", 5, 1))
    _require(_active(box, gaps, h) == {0, 1}, "same-side reflection must pass")
    zig = frozenset({(0, 1), (0, 2), (1, 0), (1, 1)})
    _require(_half_turn(zig) and not _axes(zig), "rotation is not reflection")
    _require(_axes({(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)}) == (("A", 2),), "diagonal reflection regression")
    # Input-only solving of rule-valid cases outside the sampled positive
    # regime: no active ports, and a one-cell-high cavity.
    grid = np.zeros((15, 15), dtype=np.int8)
    grid[2, 2:13] = grid[12, 2:13] = 1
    grid[2:13, 2] = grid[2:13, 12] = 1
    grid[2, 5:8] = grid[12, 6:9] = 0
    for cell in ((5, 5), (5, 6), (6, 5), (7, 5), (7, 6)):
        grid[cell] = 2
    grid[9, 9] = 3
    m = _parse(grid)
    _require(bool(m["axes"]) and not _active(m["box"], m["gaps"], m["axes"]), "empty outflow fixture")
    expected = grid.copy()
    expected[3:12, 3:12] = 3
    _require(np.array_equal(solve(grid), expected), "positive cavity without outflow")
    tiny = np.zeros((5, 11), dtype=np.int8)
    tiny[1, 1:10] = tiny[3, 1:10] = 1
    tiny[2, 1] = tiny[2, 9] = 1
    tiny[1, 4:6] = tiny[3, 4:6] = 0
    tiny[2, 3:7] = 2
    tiny[2, 8] = 3
    result = solve(tiny)
    _require(np.all(result[2, 2:9] == 3) and np.all(result[:, 4:6] == 3), "thin valid cavity")


def self_test():
    """Bounded 42-seed variation audit plus interface and metamorphic checks."""
    _regressions()
    shapes, motifs, profiles, palettes, sizes, signatures = set(), set(), set(), set(), set(), set()
    self_matches = same_width_decoys = between_axes = union_cases = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        sig = tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])
        _require(sig == tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in again[s]), "nondeterminism")
        signatures.add(sig)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                saved = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(saved, inp), "solver mismatch/mutation")
                _require(np.array_equal(solve(np.rot90(inp)), np.rot90(out)), "rotation equivariance failed")
                _require(np.array_equal(solve(np.fliplr(inp)), np.fliplr(out)), "reflection equivariance failed")
                perm = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(perm[inp]), perm[out]), "color equivariance failed")
                m = _parse(inp)
                shapes.add(inp.shape)
                motifs.add(tuple(sorted(_normal(m["motif"]))))
                profiles.add(tuple((g[0], g[2]) for g in m["gaps"]))
                palettes.add(tuple(m[k] for k in ("bg", "frame", "neutral", "paint")))
                sizes.add(len(m["motif"]))
                active = _active(m["box"], m["gaps"], m["axes"])
                self_matches += any(frozenset(_reflect(p, a) for p in _gap_cells(m["box"], g)) == _gap_cells(m["box"], g)
                                    for a in m["axes"] for g in m["gaps"])
                same_width_decoys += any(g[2] == m["gaps"][j][2] for i, g in enumerate(m["gaps"]) if i not in active for j in active)
                between_axes += any(a[0] in ("H", "V") and a[1] % 2 for a in m["axes"])
                union_cases += split == "test"
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(motifs) >= 60, "insufficient geometry diversity")
    _require(len(profiles) >= 80 and len(palettes) >= 30 and len(sizes) >= 12, "insufficient family diversity")
    _require(self_matches >= 5 and same_width_decoys >= 20 and between_axes >= 10 and union_cases == 42, "missing conceptual regimes")
    # All legal explicit count combinations, not just boundary calls.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(2026 + nt * 10 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "count mismatch")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "oriented_motifs": len(motifs), "gap_profiles": len(profiles),
            "palettes": len(palettes), "motif_sizes": sorted(sizes),
            "pairs_with_self_matches": self_matches, "equal_width_decoy_pairs": same_width_decoys,
            "pairs_with_between_cell_axes": between_axes, "multi_axis_tests": union_cases}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
