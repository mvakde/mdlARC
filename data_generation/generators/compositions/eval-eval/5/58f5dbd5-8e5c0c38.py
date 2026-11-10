"""Mirror-coupled die cutting; all semantic information is in the input.

Closed marker frames hold material. Four marker corners register punch maps.
Color equality pairs them. A punch vetoes both members of a material reflection
orbit, whose axis is determined from the original material, before any cutting.
"""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _reflect(mask, axis2):
    """axis2 is twice the vertical axis coordinate, local to the window."""
    columns = axis2 - np.arange(mask.shape[1])
    valid = (columns >= 0) & (columns < mask.shape[1])
    result = np.zeros_like(mask, dtype=bool)
    result[:, valid] = mask[:, columns[valid]]
    return result


def _axis(mask):
    _require(bool(mask.any()), "empty material")
    scores = np.zeros(2 * mask.shape[1] - 1, dtype=int)
    for row in mask.astype(np.int16):
        scores += np.convolve(row, row)
    winners = np.flatnonzero(scores == scores.max())
    _require(len(winners) == 1, "tied material axes")
    axis2 = int(winners[0])
    return axis2, mask & _reflect(mask, axis2)


def _cut(material, punch, axis2):
    return material & _reflect(material, axis2) & ~punch & ~_reflect(punch, axis2)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            y, x = stack.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(np.asarray(cells, dtype=int))
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > a.size // 2, "background must occupy a strict majority")
    candidates = []
    for marker in sorted(set(map(int, np.unique(a))) - {bg}):
        sources = []
        for cells in _components(a == marker):
            y, x = map(int, cells.min(axis=0))
            b, r = map(int, cells.max(axis=0) + 1)
            h, w = b - y, r - x
            if h < 5 or w < 6 or len(cells) != 2 * (h + w) - 4:
                continue
            window = a[y:b, x:r]
            if not (np.all(window[0] == marker) and np.all(window[-1] == marker)
                    and np.all(window[:, 0] == marker) and np.all(window[:, -1] == marker)):
                continue
            interior = window[1:-1, 1:-1]
            colors = set(map(int, np.unique(interior))) - {bg}
            if len(colors) != 1 or marker in colors:
                continue
            color = colors.pop()
            material = interior == color
            axis2, core = _axis(material)
            sources.append({"box": (y, x, b, r), "color": color,
                            "material": material, "axis2": axis2, "core": core})
        if sources:
            candidates.append((marker, sources))
    _require(len(candidates) == 1, "marker frames are not unique")
    marker, sources = candidates[0]
    sources.sort(key=lambda s: s["box"])
    _require(len({s["color"] for s in sources}) == len(sources), "duplicate material color")
    covered = np.zeros(a.shape, dtype=bool)
    for s in sources:
        y, x, b, r = s["box"]
        _require(not covered[y:b, x:r].any(), "overlapping material windows")
        covered[y:b, x:r] = True
    external = ~covered
    marker_cells = set(map(tuple, np.argwhere(external & (a == marker)).tolist()))
    ys = sorted({p[0] for p in marker_cells})
    xs = sorted({p[1] for p in marker_cells})
    stamps, used_corners = {}, set()
    colors = sorted(set(map(int, np.unique(a[external]))) - {bg, marker})
    for color in colors:
        cells = np.argwhere(external & (a == color))
        y0, x0 = map(int, cells.min(axis=0))
        y1, x1 = map(int, cells.max(axis=0))
        options = []
        for y in (v for v in ys if v < y0):
            for b in (v for v in ys if v > y1):
                for x in (v for v in xs if v < x0):
                    for r in (v for v in xs if v > x1):
                        corners = {(y, x), (y, r), (b, x), (b, r)}
                        if not corners <= marker_cells or b - y < 4 or r - x < 5:
                            continue
                        if covered[y:b+1, x:r+1].any():
                            continue
                        window = a[y:b+1, x:r+1]
                        rim = np.zeros(window.shape, dtype=bool)
                        rim[0] = rim[-1] = True
                        rim[:, 0] = rim[:, -1] = True
                        corner_mask = np.zeros(window.shape, dtype=bool)
                        corner_mask[0, 0] = corner_mask[0, -1] = True
                        corner_mask[-1, 0] = corner_mask[-1, -1] = True
                        if not np.all(window[rim & ~corner_mask] == bg):
                            continue
                        interior = window[1:-1, 1:-1]
                        if not np.all((interior == bg) | (interior == color)):
                            continue
                        options.append((y, x, b+1, r+1, corners, interior == color))
        _require(len(options) == 1, "unregistered or ambiguous punch pattern")
        y, x, b, r, corners, punch = options[0]
        _require(not (corners & used_corners), "shared registration corner")
        _require(not covered[y:b, x:r].any(), "overlapping registers")
        covered[y:b, x:r] = True
        used_corners |= corners
        stamps[color] = {"box": (y, x, b, r), "punch": punch.copy()}
    _require(used_corners == marker_cells, "unused registration marker")
    _require(np.all(a[~covered] == bg), "unregistered content")
    _require(set(s["color"] for s in sources) <= set(stamps), "missing matching punch")
    for s in sources:
        punch = stamps[s["color"]]["punch"]
        _require(punch.shape == s["material"].shape, "matched window sizes differ")
        s["punch"] = punch
    return {"background": bg, "marker": marker, "sources": sources,
            "stamps": stamps, "shape": a.shape}


def _answer(info, mode="rule"):
    out = np.full(info["shape"], info["background"], dtype=np.int8)
    for s in info["sources"]:
        y, x, b, r = s["box"]
        out[y:b, x] = out[y:b, r-1] = info["marker"]
        out[y, x:r] = out[b-1, x:r] = info["marker"]
        m, p, axis = s["material"], s["punch"], s["axis2"]
        if mode == "no_symmetry":
            keep = m & ~p
        elif mode == "no_material_support":
            keep = m & ~p & ~_reflect(p, axis)
        elif mode == "no_mirrored_punch":
            keep = s["core"] & ~p
        elif mode == "no_punch":
            keep = s["core"]
        elif mode == "supported_punch_only":
            keep = s["core"] & ~(p & _reflect(p, axis))
        elif mode == "reaxis":
            _, keep = _axis(m & ~p)
        elif mode == "horizontal":
            a2, _ = _axis(m.T)
            keep = _cut(m.T, p.T, a2).T
        elif mode == "frame_center":
            keep = _cut(m, p, m.shape[1] - 1)
        elif mode == "flip_punch_rows":
            keep = _cut(m, np.flipud(p), axis)
        elif mode == "flip_punch_cols":
            keep = _cut(m, np.fliplr(p), axis)
        else:
            keep = _cut(m, p, axis)
        out[y+1:b-1, x+1:r-1][keep] = s["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only solver. Registration rims do not vote for an axis."""
    return _answer(_decode(input_grid))


def _different(info, mode, answer):
    try:
        return not np.array_equal(_answer(info, mode), answer)
    except ValueError:
        return True


def _material(rng, h, w, parity, noisy):
    widths = [v for v in range(2, w) if (v - 1) % 2 == parity]
    for _ in range(64):
        cw = int(rng.choice(widths))
        ch = int(rng.integers(2, h+1))
        left, top = int(rng.integers(w-cw+1)), int(rng.integers(h-ch+1))
        local = np.zeros((ch, cw), dtype=bool)
        kind = int(rng.integers(5))
        if kind == 0:
            local[:] = True
            if ch > 2 and cw > 2:
                local[1:-1, 1:-1] = False
        elif kind == 1:
            # Independently varying branches, necks, and possible axis-only rows.
            for row in range(ch):
                radius = int(rng.integers(cw//2+1))
                for col in range(cw):
                    local[row, col] = abs(2*col-(cw-1)) <= 2*radius
        else:
            local = rng.random((ch, cw)) < float(rng.uniform(.2, .85))
            local |= np.fliplr(local)
            if kind == 3 and ch >= 4:
                local[int(rng.integers(1, ch-1))] = False
            if kind == 4:
                local[:] = True
                for row in range(ch):
                    if rng.random() < .65 and cw > 2:
                        local[row, 1:-1] = False
        row = int(rng.integers(ch))
        local[row, 0] = local[row, -1] = True
        if local.sum() < 4:
            continue
        core = np.zeros((h, w), dtype=bool)
        core[top:top+ch, left:left+cw] = local
        axis = 2*left + cw-1
        material = core.copy()
        if not noisy:
            return material, axis, core
        amount = int(rng.integers(1, max(2, min(8, int(core.sum()*.35))+1))) if noisy else 0
        choices = list(map(tuple, np.argwhere(~core).tolist()))
        for i in rng.permutation(len(choices)):
            y, x = choices[int(i)]
            rx = axis-x
            if 2*x == axis or (0 <= rx < w and material[y, rx]):
                continue
            proposal = material.copy()
            proposal[y, x] = True
            try:
                a2, supported = _axis(proposal)
            except ValueError:
                continue
            if a2 == axis and np.array_equal(supported, core):
                material = proposal
                amount -= 1
                if amount <= 0:
                    break
        if not noisy or np.count_nonzero(material & ~core):
            return material, axis, core
    # Guaranteed asymmetric addition with either axis parity, in every window.
    core = np.zeros((h, w), dtype=bool)
    cw = 3 if parity == 0 else 2
    core[0:2, 0:cw] = True
    material = core.copy()
    material[2, 0] = True
    axis, supported = _axis(material)
    _require(np.array_equal(supported, core), "material fallback failed")
    return material, axis, core


def _punch(rng, material, axis, core, density):
    off = np.argwhere(core & (2*np.arange(core.shape[1])[None, :] != axis))
    for _ in range(32):
        p = rng.random(material.shape) < density
        y, x = map(int, off[int(rng.integers(len(off)))])
        p[y, x], p[y, axis-x] = True, False
        keep = _cut(material, p, axis)
        if keep.any() and np.any(core & ~p & _reflect(p, axis)):
            return p
    p = np.zeros(material.shape, dtype=bool)
    y, x = map(int, off[0])
    p[y, x] = True
    return p


def _item(rng, h, w, parity, density, witness=False, noisy=True):
    for _ in range(96 if witness else 1):
        material, axis, core = _material(rng, h, w, parity, noisy)
        punch = _punch(rng, material, axis, core, density)
        if witness:
            if not np.any(material & ~core & ~punch & ~_reflect(punch, axis)):
                continue
            # Teach that axes are established BEFORE cuts. Require a unique,
            # different answer under the attractive reverse-order hypothesis.
            try:
                _, reverse = _axis(material & ~punch)
                if np.array_equal(reverse, _cut(material, punch, axis)):
                    continue
                ha, _ = _axis(material.T)
                if np.array_equal(_cut(material.T, punch.T, ha).T,
                                  _cut(material, punch, axis)):
                    continue
                if any(np.array_equal(_cut(material, altered, axis), _cut(material, punch, axis))
                       for altered in (np.flipud(punch), np.fliplr(punch))):
                    continue
            except ValueError:
                continue
        return material, punch
    # Explicit witness: original axis 2; after raw punching the best axis is 3.
    material = np.zeros((h, w), dtype=bool)
    material[0, 1:4] = True
    material[1, :5] = True
    material[2, [1, 3, 4]] = True
    punch = np.zeros_like(material)
    punch[0, 1] = punch[1, 0] = punch[2, 1] = True
    punch[0, 2] = True
    return material, punch


def _place(rng, widgets):
    """Free rectangle packing, with a jittered shelf fallback, never clipping."""
    heights = [a.shape[0] for a in widgets]
    widths = [a.shape[1] for a in widgets]
    n, mh, mw = len(widgets), max(heights), max(widths)
    options = [(c, (n+c-1)//c) for c in range(1, n+1)
               if c*mw+c-1 <= 30 and ((n+c-1)//c)*mh+(n+c-1)//c-1 <= 30]
    _require(bool(options), "packing capacity exceeded")
    nc, nr = options[int(rng.integers(len(options)))]
    base_h, base_w = nr*mh+nr-1, nc*mw+nc-1
    h = int(rng.integers(base_h, min(30, base_h+6)+1))
    w = int(rng.integers(base_w, min(30, base_w+6)+1))
    order = sorted(range(n), key=lambda i: (-heights[i]*widths[i], float(rng.random())))
    if rng.random() < .75:
        for _ in range(8):
            occupied = np.zeros((h, w), dtype=bool)
            positions = [None]*n
            for i in order:
                ih, iw = widgets[i].shape
                choices = [(y, x) for y in range(h-ih+1) for x in range(w-iw+1)
                           if not occupied[y:y+ih, x:x+iw].any()]
                if not choices:
                    break
                y, x = choices[int(rng.integers(len(choices)))]
                positions[i] = (y, x)
                occupied[max(0, y-1):min(h, y+ih+1), max(0, x-1):min(w, x+iw+1)] = True
            if all(p is not None for p in positions):
                return (h, w), positions
    oy, ox = int(rng.integers(h-base_h+1)), int(rng.integers(w-base_w+1))
    slots = [(oy+(i//nc)*(mh+1), ox+(i % nc)*(mw+1)) for i in range(n)]
    rng.shuffle(slots)
    positions = [(y+int(rng.integers(mh-ih+1)), x+int(rng.integers(mw-iw+1)))
                 for (y, x), ih, iw in zip(slots, heights, widths)]
    return (h, w), positions


def _make_pair(rng, count, limits, density, teaching=None, large=False):
    bg, marker = map(int, rng.choice(10, 2, replace=False))
    distractors = 1 if teaching == 0 else int(rng.integers(3))
    # All widgets can always fit in a rectangular shelf under these limits.
    if count >= 3 and distractors > 0:
        limits = (min(limits[0], 4), min(limits[1], 5))
    elif count == 2 and distractors > 0:
        limits = (min(limits[0], 6), limits[1])
    colors = list(map(int, rng.choice([c for c in range(10) if c not in (bg, marker)],
                                    count+distractors, replace=False)))
    widgets, source_records = [], []
    for i in range(count):
        h = int(rng.integers(6 if large and i == 0 else 3, limits[0]+1))
        if teaching == 0 and i == 0:
            w = int(rng.choice([v for v in range(6, limits[1]+1) if v % 2 == 0]))
        else:
            w = int(rng.integers(7 if large and i == 0 else 4, limits[1]+1))
        parity = teaching if i == 0 and teaching is not None else int(rng.integers(2))
        material, punch = _item(rng, h, w, parity, density,
                                witness=(teaching == 0 and i == 0),
                                noisy=(i == 0 or rng.random() < .7))
        source = np.full((h+2, w+2), bg, dtype=np.int8)
        source[0] = source[-1] = marker
        source[:, 0] = source[:, -1] = marker
        source[1:-1, 1:-1][material] = colors[i]
        stamp = np.full(source.shape, bg, dtype=np.int8)
        stamp[0, 0] = stamp[0, -1] = stamp[-1, 0] = stamp[-1, -1] = marker
        stamp[1:-1, 1:-1][punch] = colors[i]
        widgets.extend((source, stamp))
        source_records.append((material, punch))
    for i in range(distractors):
        if teaching == 0 and i == 0:
            material, punch = source_records[0]
            axis, _ = _axis(material)
            mask = punch.copy()
            y, x = map(int, np.argwhere(_cut(material, punch, axis))[0])
            mask[y, x] = True  # a demonstrably different color assignment
        else:
            h, w = int(rng.integers(3, limits[0]+1)), int(rng.integers(4, limits[1]+1))
            mask = rng.random((h, w)) < float(rng.uniform(.1, .7))
            mask[int(rng.integers(h)), int(rng.integers(w))] = True
        stamp = np.full((mask.shape[0]+2, mask.shape[1]+2), bg, dtype=np.int8)
        stamp[0, 0] = stamp[0, -1] = stamp[-1, 0] = stamp[-1, -1] = marker
        stamp[1:-1, 1:-1][mask] = colors[count+i]
        widgets.append(stamp)
    shape, positions = _place(rng, widgets)
    inp = np.full(shape, bg, dtype=np.int8)
    for widget, (y, x) in zip(widgets, positions):
        inp[y:y+widget.shape[0], x:x+widget.shape[1]] = widget
    # Empty space makes the background observable, even in dense small scenes.
    while np.count_nonzero(inp == bg) <= inp.size//2:
        axis = 0 if inp.shape[0] <= inp.shape[1] and inp.shape[0] < 30 else 1
        if inp.shape[axis] == 30:
            axis = 1-axis
        _require(inp.shape[axis] < 30, "cannot establish background majority")
        padding = [(0, 0), (0, 0)]
        padding[axis] = (0, 1)
        inp = np.pad(inp, padding, constant_values=bg)
    info = _decode(inp)
    out = _answer(info)
    if teaching == 0:
        _require(_different(info, "reaxis", out) and _different(info, "horizontal", out),
                 "missing ordering or direction evidence")
    return {"input": inp, "output": out}


def _count(name, value, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # count, scale, or punch density
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(nt+ne):
        test = index >= nt
        if shift == 0:
            count = int(rng.integers(3, 5) if test else rng.integers(1, 3))
        else:
            count = int(rng.integers(1, 3) if shift == 1 else rng.integers(1, 5))
        # Teaching witnesses need an interior at least 3x5.
        if index == 0:
            count = min(count, 2)
        limits = ((10, 11), (8, 10), (6, 7), (5, 6))[count-1]
        if shift == 1:
            limits = (limits[0], limits[1]) if test else (min(5, limits[0]), min(6, limits[1]))
        if shift == 2:
            density = float(rng.uniform(.32, .6) if test else rng.uniform(.06, .2))
        else:
            density = float(rng.uniform(.1, .43))
        teaching = index if index < 2 else None
        for _ in range(24):
            try:
                pair = _make_pair(rng, count, limits, density, teaching, large=(shift == 1 and test))
            except ValueError:
                continue
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle["test" if test else "train"].append(pair)
                break
        else:
            raise RuntimeError("bounded pair generation exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", ne, 1, 4)
    _require(nt > ne, "invalid split counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(_answer(info), out), "wrong reflected cuts or register erasure")
            _require(np.all((out == inp) | (out == info["background"])), "only deletion is allowed")
            for s in info["sources"]:
                # Coordinate reconstruction, distinct from convolution scoring.
                cells = set(map(tuple, np.argwhere(s["material"]).tolist()))
                scores = [sum((y, a-x) in cells for y, x in cells)
                          for a in range(2*s["material"].shape[1]-1)]
                _require(scores.count(max(scores)) == 1 and int(np.argmax(scores)) == s["axis2"],
                         "axis scoring disagreement")
                y, x, b, r = s["box"]
                keep = out[y+1:b-1, x+1:r-1] == s["color"]
                _require(np.array_equal(keep, _reflect(keep, s["axis2"])), "cuts broke the original symmetry")
                _require(not np.any(keep & (s["punch"] | _reflect(s["punch"], s["axis2"]))),
                         "punched orbit survived")
            if split == "train":
                training.append((info, out))
    for mode in ("no_symmetry", "no_material_support", "no_mirrored_punch", "no_punch",
                 "supported_punch_only", "reaxis", "horizontal", "frame_center",
                 "flip_punch_rows", "flip_punch_cols"):
        _require(any(_different(info, mode, out) for info, out in training),
                 "demonstrations do not distinguish " + mode)
    parities = {s["axis2"] % 2 for info, _ in training for s in info["sources"]}
    _require(parities == {0, 1}, "training must demonstrate both axis parities")
    matching_witness = False
    for info, _ in training:
        for s in info["sources"]:
            expected = _cut(s["material"], s["punch"], s["axis2"])
            for color, stamp in info["stamps"].items():
                if color != s["color"] and stamp["punch"].shape == s["material"].shape:
                    matching_witness |= not np.array_equal(expected, _cut(s["material"], stamp["punch"], s["axis2"]))
    _require(matching_witness, "color matching has no demonstrated consequence")


def self_test():
    shapes, windows, axes, colors, counts, fingerprints = set(), set(), set(), set(), set(), set()
    sparse = dense = islands = off_center = empty_margins = distractors = 0
    for seed in range(42):
        nt, ne = 2 + seed % 9, 1 + seed % min(4, 1 + seed % 9)
        p = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(p)
        fingerprint = tuple(_key(pair["input"]) for split in p for pair in p[split])
        fingerprints.add(fingerprint)
        if seed < 3:
            repeat = generate_puzzle(seed, nt, ne)
            assert all(np.array_equal(a[k], b[k]) for split in p for a, b in zip(p[split], repeat[split])
                       for k in ("input", "output"))
        for split in p:
            for pair in p[split]:
                inp = pair["input"].copy()
                assert np.array_equal(solve(inp), pair["output"])
                assert np.array_equal(inp, pair["input"]), "solve modified its input"
                info = _decode(inp)
                shapes.add(inp.shape)
                colors.add(info["background"])
                counts.add(len(info["sources"]))
                distractors += len(info["stamps"]) > len(info["sources"])
                for s in info["sources"]:
                    windows.add(s["material"].shape)
                    axes.add(s["axis2"] % 2)
                    off_center += s["axis2"] != s["material"].shape[1]-1
                    islands += len(_components(s["core"])) > 1
                    sparse += s["punch"].sum() <= s["punch"].size*.2
                    dense += s["punch"].sum() >= s["punch"].size*.4
                    empty_margins += not (s["punch"][0].any() and s["punch"][-1].any()
                                          and s["punch"][:, 0].any() and s["punch"][:, -1].any())
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(windows) >= 20
    assert axes == {0, 1} and colors == set(range(10)) and counts == {1, 2, 3, 4}
    assert min(sparse, dense, islands, off_center, empty_margins, distractors) >= 10
    # Rule extensions beyond the sampler: five materials, entirely punched
    # material, a 28-column registered window, and clean supported islands.
    scene = np.zeros((23, 20), dtype=np.int8)
    expected = scene.copy()
    for i in range(5):
        color = i+2
        material = np.zeros((3, 4), dtype=bool)
        material[:2, :3] = True
        material[2, 0] = True
        punch = np.ones((3, 4), dtype=bool)
        for slot, is_source in ((2*i, True), (2*i+1, False)):
            y, x = (slot//3)*6, (slot % 3)*7
            panel = scene[y:y+5, x:x+6]
            if is_source:
                panel[0] = panel[-1] = 1
                panel[:, 0] = panel[:, -1] = 1
                panel[1:-1, 1:-1][material] = color
                rim = expected[y:y+5, x:x+6]
                rim[0] = rim[-1] = 1
                rim[:, 0] = rim[:, -1] = 1
            else:
                panel[0, 0] = panel[0, -1] = panel[-1, 0] = panel[-1, -1] = 1
                panel[1:-1, 1:-1][punch] = color
    assert np.array_equal(solve(scene), expected)
    wide = np.zeros((11, 30), dtype=np.int8)
    wide[:5, 0] = wide[:5, -1] = 1
    wide[0] = wide[4] = 1
    wide[6, 0] = wide[6, -1] = wide[10, 0] = wide[10, -1] = 1
    wide[1, [1, 3]] = wide[3, [1, 3]] = 2
    wide[2, 2] = wide[8, 2] = 2  # supported axis pixel, and its registered punch
    expected = wide.copy()
    expected[6:] = 0
    expected[2, 2] = 0
    assert np.array_equal(solve(wide), expected)
    tie = np.zeros((3, 4), dtype=bool)
    tie[0, 0] = tie[1, 2] = True
    try:
        _axis(tie)
    except ValueError:
        pass
    else:
        raise AssertionError("tied material axis accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42000 + nt*10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(123, 2, 1)
    info = _decode(bad["test"][0]["input"])
    y, x, _, _ = info["sources"][0]["box"]
    bad["test"][0]["output"][y, x] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged material frame accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 variation seeds and all legal counts)")
