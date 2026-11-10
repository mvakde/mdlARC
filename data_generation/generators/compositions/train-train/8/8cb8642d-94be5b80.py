"""Palette stitches: a stretched X measured in the visible glyph's units.

Only standard library and NumPy are used.  The parser/solver has no access to
generation parameters.  All coordinates in public grids are ordinary pixels.
"""

import itertools

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        todo = [unseen.pop()]
        cells = []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _x_mask(h, w):
    """Four 45-degree arms joined by a middle stroke, in glyph units."""
    mask = np.zeros((h, w), dtype=bool)
    if h > w:
        return _x_mask(w, h).T.copy()
    for c in range(w):
        depth = min(c, w-1-c, (h-1)//2)
        mask[depth, c] = True
        mask[h-1-depth, c] = True
    return mask


def _x_cell(r, c, h, w):
    """Row-wise formulation, separate from the column-wise drawing routine."""
    if h > w:
        return _x_cell(c, r, w, h)
    d = min(r, h-1-r)
    return (c == d or c == w-1-d
            or (r in ((h-1)//2, h//2) and d <= c < w-d))


def _stamp(view, r, c, mask, color):
    h, w = mask.shape
    tile = view[r*h:(r+1)*h, c*w:(c+1)*w]
    tile[mask] = color


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg),
             "a uniform exterior border must identify background")
    boxes = []
    for cells in _components(a != bg):
        yy, xx = zip(*cells)
        y, x, h, w = min(yy), min(xx), max(yy)-min(yy)+1, max(xx)-min(xx)+1
        _require(len(cells) == h*w, "input objects must fill rectangular boxes")
        boxes.append((y, x, h, w))
    palettes, panel_boxes = [], []
    for y, x, h, w in boxes:
        b = a[y:y+h, x:x+w]
        if (h == 3 and 3 <= w <= 7 and len(np.unique(b[0])) == w
                and np.all(b == b[0])):
            palettes.append((y, x, h, w, 1, tuple(map(int, b[0]))))
        elif (w == 3 and 3 <= h <= 7 and len(np.unique(b[:, 0])) == h
                and np.all(b == b[:, :1])):
            palettes.append((y, x, h, w, 0, tuple(map(int, b[:, 0]))))
        else:
            panel_boxes.append((y, x, h, w))
    _require(len(palettes) == 1, "exactly one isolated swatch palette is required")
    py, px, ph, pw, axis, colors = palettes[0]
    n = len(colors)
    _require(1 <= len(panel_boxes) <= 3, "one to three panels are required")
    panels = []
    for y, x, h, w in sorted(panel_boxes):
        b = a[y:y+h, x:x+w]
        frame = int(b[0, 0])
        _require(frame not in colors and frame != bg,
                 "frame must differ from swatches and background")
        _require(np.all(b[0] == frame) and np.all(b[-1] == frame)
                 and np.all(b[:, 0] == frame) and np.all(b[:, -1] == frame),
                 "panel has no intact one-cell frame")
        interior = b[1:-1, 1:-1]
        _require(set(map(int, np.unique(interior))) <= {frame, *colors},
                 "unrecognized interior color")
        active = [i for i, color in enumerate(colors) if np.any(interior == color)]
        _require(1 <= len(active) < n and active == list(range(active[0], active[-1]+1)),
                 "visible colors must form a proper consecutive palette run")
        anchors, glyph = [], None
        for i in active:
            pos = np.argwhere(interior == colors[i])
            lo, hi = pos.min(axis=0), pos.max(axis=0)+1
            r0, c0, r1, c1 = map(int, (*lo, *hi))
            other = interior[r0:r1, c0:c1] == colors[i]
            if glyph is None:
                glyph = other
            _require(glyph.shape == other.shape and np.array_equal(glyph, other),
                     "visible colors must each supply exactly one identical glyph")
            anchors.append((i, r0, c0))
        mh, mw = glyph.shape
        _require(2 <= mh <= 4 and 2 <= mw <= 4 and not glyph.all()
                 and np.count_nonzero(glyph) >= 3,
                 "glyph must have a tight, non-solid 2..4 by 2..4 bounding box")
        _require(len(_components(glyph)) <= 2, "glyph is excessively fragmented")
        ih, iw = interior.shape
        _require(ih % mh == 0 and iw % mw == 0, "glyph units must tile the interior")
        rows, cols = ih//mh, iw//mw
        _require(3 <= rows <= n and 3 <= cols <= n
                 and (rows, cols)[axis] == n,
                 "palette must span a longest axis in glyph units")
        observation = np.full_like(interior, frame)
        sites = []
        for i, r0, c0 in anchors:
            _require(r0 % mh == 0 and c0 % mw == 0, "anchor is off the glyph lattice")
            r, c = r0//mh, c0//mw
            _require((r, c)[axis] == i and _x_cell(r, c, rows, cols),
                     "anchor phase or stretched-X location is incorrect")
            _stamp(observation, r, c, glyph, colors[i])
            sites.append((i, r, c))
        _require(np.array_equal(observation, interior),
                 "panel contains cells outside its declared complete anchors")
        panels.append({"box": (y, x, h, w), "frame": frame,
                       "glyph": glyph, "rows": rows, "cols": cols,
                       "anchors": sites})
    return {"background": bg, "palette_box": (py, px, ph, pw),
            "axis": axis, "colors": colors, "panels": panels}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover each local glyph, measure its panel, then complete the weave."""
    a = _grid(input_grid)
    info = _parse(a)
    result = a.copy()
    py, px, ph, pw = info["palette_box"]
    result[py:py+ph, px:px+pw] = info["background"]
    for p in info["panels"]:
        y, x, h, w = p["box"]
        interior = result[y+1:y+h-1, x+1:x+w-1]
        interior[:] = info["background"]
        for r, c in np.argwhere(_x_mask(p["rows"], p["cols"])):
            i = (r, c)[info["axis"]]
            _stamp(interior, int(r), int(c), p["glyph"], info["colors"][i])
    return result


def _glyph(rng, h, w):
    """Small coherent Ls, outlines, spines and stepped bodies; no solid tiles."""
    mask = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(0, 6))
    if style == 0:
        mask[:, 0] = True
        mask[-1, :] = True
    elif style == 1:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    elif style == 2:
        mask[:, int(rng.integers(w))] = True
        mask[int(rng.integers(h)), :] = True
    elif style == 3:
        for r in range(h):
            stop = 1 + r*(w-1)//max(1, h-1)
            mask[r, :stop] = True
    elif style == 4:
        mask[:, 0] = True
        mask[0, :] = True
        mask[-1, -1] = True  # occasionally a second, restrained component
    else:
        # A narrow staircase can have neither a complete row nor a complete
        # column. Template extraction must not rely on a solid bridge.
        for r in range(h):
            start = r*(w-1)//(h-1)
            mask[r, start:min(w, start+2)] = True
    if mask.all():
        mask[0, 0] = False
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    return mask.copy()


def _pack(rng, objects, bg):
    """Enumerate small shelf packings, then vary offsets and outer margins."""
    gap = int(rng.integers(1, 4))
    layouts = []
    k = len(objects)
    for order in itertools.permutations(range(k)):
        for cuts in range(1 << (k-1)):
            rows, current = [], []
            for j, index in enumerate(order):
                current.append(index)
                if j == k-1 or cuts & (1 << j):
                    rows.append(current)
                    current = []
            heights = [max(objects[i].shape[0] for i in row) for row in rows]
            widths = [sum(objects[i].shape[1] for i in row)+gap*(len(row)-1)
                      for row in rows]
            ch, cw = sum(heights)+gap*(len(rows)-1), max(widths)
            if ch <= 28 and cw <= 28:
                layouts.append((rows, heights, widths, ch, cw))
    if not layouts:
        return None
    rows, heights, widths, ch, cw = layouts[int(rng.integers(len(layouts)))]
    h = ch + int(rng.integers(2, min(6, 30-ch)+1))
    w = cw + int(rng.integers(2, min(6, 30-cw)+1))
    grid = np.full((h, w), bg, dtype=np.int8)
    positions = [None]*k
    top = int(rng.integers(1, h-ch))
    for row, rh, rw in zip(rows, heights, widths):
        left = int(rng.integers(1, w-rw))
        for i in row:
            obj = objects[i]
            y = top + int(rng.integers(0, rh-obj.shape[0]+1))
            grid[y:y+obj.shape[0], left:left+obj.shape[1]] = obj
            positions[i] = (y, left)
            left += obj.shape[1]+gap
        top += rh+gap
    return grid, positions


def _make_pair(rng, bg, transpose, split, index):
    first = split == "train" and index == 0
    second = split == "train" and index == 1
    if first:
        count, n = 2, int(rng.integers(4, 6))
    elif second:
        count, n = int(rng.integers(1, 3)), 5
    elif split == "test" and index == 0:
        # Hold out either longer color schedules or a third simultaneous panel.
        count = int(rng.integers(1, 4))
        n = int(rng.integers(6, 8)) if count <= 2 else int(rng.integers(4, 6))
    else:
        count = int(rng.choice([1, 1, 2] if split == "train" else [1, 1, 2, 3]))
        if split == "test" and count == 1:
            n = int(rng.integers(6, 8))
        else:
            n = int(rng.integers(3, 6))
    pool = rng.permutation([c for c in range(10) if c != bg])
    colors = [int(c) for c in pool[:n]]
    if first and (colors == sorted(colors) or colors == sorted(colors, reverse=True)):
        colors = colors[1:] + colors[:1]
    frames = [int(c) for c in pool[n:]]
    for attempt in range(120):
        objects, specs = [], []
        for j in range(count):
            if first and j == 0:
                rows = n  # square in glyph units
            elif first and j == 1:
                rows = n-1
            elif second:
                rows = 3  # an extended middle stroke is taught before testing
            elif count == 2 and n >= 6:
                rows = int(rng.integers(3, 5))
            else:
                rows = int(rng.integers(3, n+1))
            if first and j == 0:
                mh, mw = 2, int(rng.integers(2, 4))
            elif first and j == 1:
                # Deliberately tall in pixels but wide in glyph units.
                mh, mw = int(rng.integers(3, 5)), 2
            else:
                cap = 3 if count > 1 else 4
                mh = int(rng.integers(2, min(cap, 26//rows)+1))
                mw = int(rng.integers(2, min(cap, 26//n)+1))
            glyph = _glyph(rng, mh, mw)
            frame = int(rng.choice(frames))
            b = np.full((rows*mh+2, n*mw+2), frame, dtype=np.int8)
            visible_count = 2 if first else (1 if second or split == "test"
                                           else int(rng.integers(1, min(3, n-1)+1)))
            start = 1 if first else int(rng.integers(0, n-visible_count+1))
            sites = _x_mask(rows, n)
            for i in range(start, start+visible_count):
                choices = np.flatnonzero(sites[:, i])
                r = int(rng.choice(choices))
                _stamp(b[1:-1, 1:-1], r, i, glyph, colors[i])
            objects.append(b)
            specs.append((rows, n, glyph, frame))
        objects.append(np.tile(np.array(colors, dtype=np.int8), (3, 1)))
        packed = _pack(rng, objects, bg)
        if packed is None:
            continue
        inp, positions = packed
        # Ground truth is rendered directly from constructed geometry, not solve.
        out = np.full_like(inp, bg)
        for (rows, cols, glyph, frame), (y, x), obj in zip(specs, positions, objects):
            h, w = obj.shape
            out[y:y+h, x:x+w] = frame
            interior = out[y+1:y+h-1, x+1:x+w-1]
            interior[:] = bg
            for r, c in np.argwhere(_x_mask(rows, cols)):
                _stamp(interior, int(r), int(c), glyph, colors[c])
        if transpose:
            inp, out = inp.T.copy(), out.T.copy()
        return {"input": inp, "output": out}
    raise RuntimeError("could not pack compatible panel geometry")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if n <= t:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    bg_base = int(rng.integers(10))
    variable_bg = bool(rng.integers(2))
    first_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, size in (("train", n), ("test", t)):
        for i in range(size):
            transpose = first_axis ^ i if split == "train" and i < 2 else int(rng.integers(2))
            bg = int(rng.integers(10)) if variable_bg else bg_base
            for _ in range(20):
                pair = _make_pair(rng, bg, transpose, split, i)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference_output(inp, info):
    """Validate every pixel using local division and a row-wise X predicate."""
    expected = np.full_like(inp, info["background"])
    for p in info["panels"]:
        y, x, h, w = p["box"]
        expected[y:y+h, x:x+w] = p["frame"]
        mh, mw = p["glyph"].shape
        for dy in range(h-2):
            for dx in range(w-2):
                r, c = dy//mh, dx//mw
                marked = _x_cell(r, c, p["rows"], p["cols"])
                ink = marked and p["glyph"][dy % mh, dx % mw]
                expected[y+1+dy, x+1+dx] = (info["colors"][(r, c)[info["axis"]]]
                                           if ink else info["background"])
    return expected


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, "illegal counts")
    seen, training, infos = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "unchanged pair or size")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            _require(np.array_equal(out, _reference_output(inp, info)), "incorrect stitch completion")
            py, px, ph, pw = info["palette_box"]
            _require(np.all(out[py:py+ph, px:px+pw] == info["background"]), "palette not erased")
            for p in info["panels"]:
                y, x, h, w = p["box"]
                for i, r, c in p["anchors"]:
                    mh, mw = p["glyph"].shape
                    before = inp[y+1+r*mh:y+1+(r+1)*mh, x+1+c*mw:x+1+(c+1)*mw]
                    after = out[y+1+r*mh:y+1+(r+1)*mh, x+1+c*mw:x+1+(c+1)*mw]
                    _require(np.array_equal(before[p["glyph"]], after[p["glyph"]]), "anchor moved")
                # Ablating A: repeating the glyph throughout the whole lattice differs.
                _require(not np.all(_x_mask(p["rows"], p["cols"])), "A has no effect")
                # Ablating B: a solid or monochrome X cannot equal this output.
                _require(not p["glyph"].all() and len(info["colors"]) >= 3, "B has no effect")
            infos.append(info)
            if split == "train":
                training.append(info)
    _require({i["axis"] for i in training} == {0, 1}, "demonstrate both palette directions")
    types, unit_reversal = set(), False
    for info in training:
        for p in info["panels"]:
            r, c = p["rows"], p["cols"]
            types.add("square" if r == c else "tall" if r > c else "wide")
            mh, mw = p["glyph"].shape
            unit_reversal |= (r-c)*(r*mh-c*mw) < 0
    _require(types == {"square", "tall", "wide"}, "missing X geometry demonstration")
    _require(unit_reversal, "demonstrate why glyph units, rather than pixels, control the X")
    _require(any(len(p["anchors"]) >= 2 and p["anchors"][0][0] > 0
                 and p["anchors"][-1][0] < len(i["colors"])-1
                 for i in training for p in i["panels"]), "missing bidirectional completion witness")
    _require(any(len(p["anchors"]) == 1 for i in training for p in i["panels"]),
             "demonstrate singleton template extraction")


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _check_ablations(out, info):
    p = info["panels"][0]
    y, x, h, w = p["box"]
    # Retain B's exact shapes and colors but remove A's site selection.
    without_a = out.copy()
    interior = without_a[y+1:y+h-1, x+1:x+w-1]
    interior[:] = info["background"]
    for r in range(p["rows"]):
        for c in range(p["cols"]):
            _stamp(interior, r, c, p["glyph"], info["colors"][(r, c)[info["axis"]]])
    _require(not np.array_equal(without_a, out), "A ablation did not change the answer")
    # Retain A's coarse X but remove B's exact silhouette copying.
    without_glyph = out.copy()
    interior = without_glyph[y+1:y+h-1, x+1:x+w-1]
    interior[:] = info["background"]
    for r, c in np.argwhere(_x_mask(p["rows"], p["cols"])):
        _stamp(interior, int(r), int(c), np.ones_like(p["glyph"]),
               info["colors"][(r, c)[info["axis"]]])
    _require(not np.array_equal(without_glyph, out), "glyph ablation did not change the answer")
    monochrome = out.copy()
    monochrome[np.isin(out, info["colors"])] = info["colors"][0]
    _require(not np.array_equal(monochrome, out), "palette ablation did not change the answer")


def self_test():
    """Bounded 42-seed diversity, invariants, count boundaries and metamorphisms."""
    shapes, glyphs, lengths, panel_counts, axes, backgrounds, signatures = (set() for _ in range(7))
    even_short = odd_short = wide_stroke = False
    full_rows = {"train": set(), "test": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _parse(inp)
                before_solve = inp.copy()
                _require(np.array_equal(solve(inp), out), "solver mismatch")
                _require(np.array_equal(before_solve, inp), "solver modified its input")
                _check_ablations(out, info)
                shapes.add(inp.shape)
                lengths.add(len(info["colors"]))
                panel_counts.add(len(info["panels"]))
                axes.add(info["axis"])
                backgrounds.add(info["background"])
                for p in info["panels"]:
                    glyphs.add(_key(p["glyph"]))
                    full_rows[split].add(bool(np.any(np.all(p["glyph"], axis=1))))
                    short = min(p["rows"], p["cols"])
                    even_short |= short % 2 == 0
                    odd_short |= short % 2 == 1
                    wide_stroke |= abs(p["rows"]-p["cols"]) >= 3
                # Swapping literal colors and transposing preserve the rule.
                perm = np.random.default_rng(seed+10000).permutation(10)
                _require(np.array_equal(solve(perm[inp]), perm[out]), "color equivariance failed")
                _require(np.array_equal(solve(inp.T), out.T), "transpose equivariance failed")
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(glyphs) >= 30,
             "insufficient structural diversity")
    _require(lengths == {3, 4, 5, 6, 7} and panel_counts == {1, 2, 3}
             and axes == {0, 1} and len(backgrounds) >= 8,
             "missing size, count, direction or color regimes")
    _require(even_short and odd_short and wide_stroke, "missing middle-stroke cases")
    _require(all(values == {False, True} for values in full_rows.values()),
             "demonstrate glyphs with and without a filled row in both splits")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(42+n*10+t, n, t)
            _require(len(p["train"]) == n and len(p["test"]) == t, "counts ignored")
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_signature(generate_puzzle(123, 5, 2)) == _signature(generate_puzzle(123, 5, 2)),
             "seed did not reproduce the whole puzzle")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds plus all legal explicit counts)")
