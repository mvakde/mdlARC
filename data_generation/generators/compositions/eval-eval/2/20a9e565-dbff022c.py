"""Contour color relay: arithmetic geometry, directed states, whole-object holes.

Generation paints explicit strokes and analytic interiors. The input-only solver
segments objects, fits cell predicates, and floods padded whole-object background.
Only the standard library and NumPy are used; no parent code or stored examples.
"""
from __future__ import annotations

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_KINDS = ("rectangle", "ladder", "bevel", "waist")
_SIDES = ("top", "bottom", "left", "right")


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((a >= 0) & (a <= 9)), "bad grid colors"
    return a.astype(np.int64, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = _STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    result = []
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        stack, cells = [q], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _markers(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _roles(a):
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "background tie"
    bg = int(vals[np.argmax(counts)])
    matches = []
    for color in vals:
        if color == bg:
            continue
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        bounds = r0, r1, c0, c1
        if min(r1-r0, c1-c0) >= 1 and r1-r0+c1-c0 >= 4:
            if set(map(tuple, cells)) == _markers(bounds):
                matches.append((int(color), bounds))
    assert len(matches) == 1, "malformed or ambiguous Ls"
    marker, bounds = matches[0]
    r0, r1, c0, c1 = bounds
    assert np.all(np.isin(a[r0:r1+1, c0:c1+1], (bg, marker))), "window not erased"
    clean = a.copy()
    clean[clean == marker] = bg
    keys = []
    for cells in _components(clean != bg, True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        h, w = r1-r0+1, c1-c0+1
        if len(cells) != h*w or len({int(clean[q]) for q in cells}) < 2:
            continue
        lanes = []
        if h == 2 and 4 <= w <= 6:
            if r0 == 0:
                lanes.append(("top", clean[r0, c0:c1+1], clean[r1, c0:c1+1]))
            if r1 == len(a)-1:
                lanes.append(("bottom", clean[r1, c0:c1+1], clean[r0, c0:c1+1]))
        if w == 2 and 4 <= h <= 6:
            if c0 == 0:
                lanes.append(("left", clean[r0:r1+1, c0], clean[r0:r1+1, c1]))
            if c1 == a.shape[1]-1:
                lanes.append(("right", clean[r0:r1+1, c1], clean[r0:r1+1, c0]))
        for side, source, fill in lanes:
            source, fill = tuple(map(int, source)), tuple(map(int, fill))
            if len(set(source)) == len(source):
                keys.append(dict(side=side, mapping=dict(zip(source, fill)), cells=cells))
    assert len(keys) == 1, "missing or ambiguous local key"
    key = keys[0]
    assert bg not in key["mapping"] and marker not in key["mapping"]
    assert not {bg, marker} & set(key["mapping"].values())
    for q in key["cells"]:
        clean[q] = bg
    return bg, marker, bounds, key, clean


def _orient(a, direction):
    """Transform the named input direction to rightward continuation."""
    if direction == "right":
        return a.copy()
    if direction == "left":
        return a[:, ::-1].copy()
    if direction == "down":
        return a.T.copy()
    return a.T[:, ::-1].copy()


def _unorient(a, direction):
    if direction == "right":
        return a.copy()
    if direction == "left":
        return a[:, ::-1].copy()
    if direction == "down":
        return a.T.copy()
    return a[:, ::-1].T.copy()


def _bounds_after(bounds, shape, direction):
    mask = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = bounds
    mask[r0:r1+1, c0:c1+1] = True
    q = np.argwhere(_orient(mask, direction))
    lo, hi = q.min(axis=0), q.max(axis=0)
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _template(kind, arg, h, w):
    """Input-side cell predicates. None means illegal silhouette geometry."""
    if min(h, w) < 3:
        return None
    rr, cc = np.indices((h, w))
    if kind == "rectangle":
        return (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
    if kind == "ladder":
        if (h-1) % arg:
            return None
        return (rr % arg == 0) | (cc == 0) | (cc == w-1)
    if kind == "bevel":
        if min(h, w) < 2*arg+1:
            return None
        lo = np.maximum(0, arg-np.minimum(rr, h-1-rr))
    else:
        if h < 7 or w < 2*arg+3:
            return None
        # A depth-two pinch approaches its waist one cell per row. A two-cell
        # jump would create a real four-neighbor leak through the shoulder.
        ramp = np.minimum(arg, np.minimum(rr-h//3+1, h-h//3-rr))
        lo = np.where((rr >= h//3) & (rr < h-h//3), ramp, 0)
    hi = w-1-lo
    return (cc == lo) | (cc == hi) | (((rr == 0) | (rr == h-1)) & (cc >= lo) & (cc <= hi))


def _descriptors():
    yield "rectangle", 0
    for s in (2, 3, 4):
        yield "ladder", s
    for s in (1, 2):
        yield "bevel", s
        yield "waist", s


def _cavities(wall, diagonal=False):
    """Find enclosed background in the full local object, with exterior padding."""
    padded = np.pad(wall, 1, constant_values=False)
    enclosed = np.zeros(wall.shape, dtype=bool)
    for cells in _components(~padded, diagonal):
        if any(r == 0 or c == 0 or r == padded.shape[0]-1 or c == padded.shape[1]-1
               for r, c in cells):
            continue
        for r, c in cells:
            enclosed[r-1, c-1] = True
    return enclosed


def _affine(values):
    d = values[1]-values[0]
    if any(v != values[0]+i*d for i, v in enumerate(values)):
        return None
    return int(values[0]), int(d)


def _geometry(model, i):
    y = model["y"][0] + i*model["y"][1]
    h = model["h"][0] + i*model["h"][1]
    w = model["w"][0] + i*model["w"][1]
    x = model["x"] + i*(model["w"][0]+model["gap"]) + i*(i-1)//2*model["w"][1]
    return y, x, h, w


def _render_model(model, bounds, bg, mapping, policy="relay"):
    r0, r1, c0, c1 = bounds
    out = np.full((r1-r0+1, c1-c0+1), bg, dtype=int)
    walls = np.full_like(out, bg)
    fill_count, target_ids, target_colors, hole_count = 0, [], [], 0
    color = model["colors"][0]
    for i in range(40):
        y, x, h, w = _geometry(model, i)
        if x > c1:
            break
        assert min(h, w) >= 3 and max(h, w) <= 90, "continuation has invalid geometry"
        wall = _template(model["kind"], model["arg"], h, w)
        assert wall is not None and color in mapping, "invalid intermediate contour"
        if i >= model["visible"] and x+w > c0 and y+h > r0 and y <= r1:
            holes = _cavities(wall, policy == "eight")
            hole_count += len(_components(holes))
            target_ids.append(i)
            target_colors.append(color)
            draw_color = (model["colors"][i % len(model["colors"])]
                          if policy == "observed_cycle" else color)
            fill = mapping[draw_color]
            if policy == "recursive":
                # Two lookups suffice to witness violation of the one-step rule.
                fill = mapping.get(fill, fill)
            local = np.full((h, w), bg, dtype=int)
            local[wall] = draw_color
            if policy != "unfilled":
                local[holes] = fill
            a, b, c, d = max(r0, y), min(r1+1, y+h), max(c0, x), min(c1+1, x+w)
            target = out[a-r0:b-r0, c-c0:d-c0]
            wtarget = walls[a-r0:b-r0, c-c0:d-c0]
            patch = local[a-y:b-y, c-x:d-x]
            wmask = wall[a-y:b-y, c-x:d-x]
            hmask = holes[a-y:b-y, c-x:d-x]
            assert np.all(target == bg), "overlapping predicted boxes"
            target[:] = patch
            wtarget[wmask] = draw_color
            fill_count += int(np.count_nonzero(hmask))
        color = mapping[color]
    assert target_ids, "window has no predicted object"
    return out, dict(walls=walls, filled_cells=fill_count, target_ids=target_ids,
                     target_colors=target_colors, holes=hole_count,
                     skipped=min(target_ids)-model["visible"])


def _analyze(input_grid):
    a = _array(input_grid)
    bg, marker, bounds, key, clean = _roles(a)
    mapping = key["mapping"]
    answers = []
    for direction in ("right", "left", "down", "up"):
        canonical = _orient(clean, direction)
        bb = _bounds_after(bounds, a.shape, direction)
        positions = np.argwhere(canonical != bg)
        if not len(positions) or positions[:, 1].max() >= bb[2]:
            continue
        objects = []
        for cells in _components(canonical != bg, True):
            rr, cc = zip(*cells)
            y, x = min(rr), min(cc)
            patch = canonical[y:max(rr)+1, x:max(cc)+1]
            colors = set(map(int, patch[patch != bg]))
            if len(colors) != 1:
                break
            objects.append((y, x, patch, next(iter(colors))))
        else:
            if len(objects) not in (3, 4):
                continue
            objects.sort(key=lambda o: o[1])
            ys = [o[0] for o in objects]
            xs = [o[1] for o in objects]
            hs = [o[2].shape[0] for o in objects]
            ws = [o[2].shape[1] for o in objects]
            colors = [o[3] for o in objects]
            fits = [_affine(v) for v in (ys, hs, ws)]
            gaps = [xs[i+1]-xs[i]-ws[i] for i in range(len(xs)-1)]
            if None in fits or len(set(gaps)) != 1 or not 1 <= gaps[0] <= 3:
                continue
            if any(c not in mapping for c in colors):
                continue
            if any(mapping[c] != d for c, d in zip(colors, colors[1:])):
                continue
            for kind, arg in _descriptors():
                templates = [_template(kind, arg, h, w) for h, w in zip(hs, ws)]
                if any(t is None for t in templates):
                    continue
                if not all(np.array_equal(t, o[2] != bg) for t, o in zip(templates, objects)):
                    continue
                model = dict(kind=kind, arg=arg, y=fits[0], h=fits[1], w=fits[2],
                             x=xs[0], gap=gaps[0], colors=colors, visible=len(objects))
                try:
                    out, info = _render_model(model, bb, bg, mapping)
                except AssertionError:
                    continue
                answers.append((_unorient(out, direction), model, info, direction, bb))
    assert answers, "no supported input-derived relay continuation"
    expected = answers[0][0]
    assert all(np.array_equal(expected, answer[0]) for answer in answers), "competing fits disagree"
    _, model, info, direction, bb = answers[0]
    info.update(background=bg, marker=marker, bounds=bounds, key=key, model=model,
                direction=direction, canonical_bounds=bb, candidates=len(answers))
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input alone; never consult generation state."""
    return _analyze(input_grid)[0]


def _paint(kind, arg, h, w, color, fill, bg):
    """Ground truth: row strokes and interior intervals, with no flood or fitter."""
    a = np.full((h, w), bg, dtype=int)
    for r in range(h):
        lo = 0
        if kind == "bevel":
            lo = max(0, arg-min(r, h-1-r))
        elif kind == "waist" and h//3 <= r < h-h//3:
            lo = min(arg, r-h//3+1, h-h//3-r)
        hi = w-1-lo
        if r in (0, h-1) or (kind == "ladder" and r % arg == 0):
            a[r, lo:hi+1] = color
        else:
            a[r, lo], a[r, hi] = color, color
            a[r, lo+1:hi] = fill
    return a


def _place_key(a, mapping, side, start):
    for k, (source, fill) in enumerate(mapping.items()):
        p = start+k
        if side == "top":
            a[0, p], a[1, p] = source, fill
        elif side == "bottom":
            a[-1, p], a[-2, p] = source, fill
        elif side == "left":
            a[p, 0], a[p, 1] = source, fill
        else:
            a[p, -1], a[p, -2] = source, fill


def _sample_pair(rng, split, index, direction, side, conservative=False):
    first, second = split == "train" and index == 0, split == "train" and index == 1
    kind = "bevel" if first else "ladder" if second else str(rng.choice(_KINDS))
    if conservative and not first and not second:
        kind = "rectangle"
    arg = int(rng.integers(1, 3)) if kind in ("bevel", "waist") else (
        int(rng.integers(2, 5)) if kind == "ladder" else 0)
    if conservative:
        arg = 1 if kind == "bevel" else 2 if kind == "ladder" else arg
    nvisible = 3 if first or second or conservative else int(rng.integers(3, 5))
    skip = 0 if split == "train" else int(rng.integers(1, 3))
    span = 1 if first or second or conservative else int(rng.integers(1, 3))
    if conservative:
        skip = int(split == "test")
    total = nvisible+skip+span
    wmin = 2*arg+1 if kind == "bevel" else 2*arg+3 if kind == "waist" else 3
    wstep = 0 if first or second or conservative else int(rng.choice([-1, 0, 0, 1]))
    w0 = wmin+int(rng.integers(0, 3))+(total-1)*max(0, -wstep)
    gap = 1 if conservative else int(rng.integers(1, 4))
    if kind == "ladder":
        hstep = arg if second else arg*int(rng.choice([-1, 0, 1]))
        h0 = 1+arg*int(rng.integers(2, 4))+(total-1)*max(0, -hstep)
    else:
        hmin = 7 if kind == "waist" else max(3, 2*arg+1) if kind == "bevel" else 3
        hstep = int(rng.integers(1, 3)) if first else int(rng.choice([-2, -1, 0, 1, 2]))
        h0 = hmin+int(rng.integers(0, 5))+(total-1)*max(0, -hstep)
    drift = 0 if conservative else int(rng.integers(-2, 3))
    if conservative:
        w0, wstep, h0, hstep = wmin, 0, 5, 1
        if kind == "ladder":
            h0, hstep = 5, 2
    roles = list(map(int, rng.permutation(10)))
    bg, marker = roles[:2]
    nkey = int(rng.integers(max(5 if first else 4, nvisible+1), 7))
    path = roles[2:2+nkey]
    end = int(rng.integers(0, nkey-1 if first else nkey))
    mapping = dict(zip(path, path[1:]+[path[end]]))
    order = list(map(int, rng.permutation(nkey)))
    mapping = {path[i]: mapping[path[i]] for i in order}
    colors, color = [], path[0]
    for _ in range(total):
        colors.append(color)
        color = mapping[color]
    geometry, x = [], 0
    for i in range(total):
        h, w, y = h0+i*hstep, w0+i*wstep, i*drift
        if min(h, w) < 3 or max(h, w) > 30 or _template(kind, arg, h, w) is None:
            return None
        geometry.append((y, x, h, w))
        x += w+gap
    targets = geometry[nvisible+skip:]
    r0, r1 = min(y for y, x, h, w in targets), max(y+h-1 for y, x, h, w in targets)
    c0, c1 = targets[0][1], targets[-1][1]+targets[-1][3]-1
    if second:
        r0 += 1  # Genuine chambers meet the crop edge after their cap is omitted.
    elif not first and not conservative and rng.random() < .65:
        room = r1-r0-2
        if room > 0:
            r0 += int(rng.integers(0, room//2+1))
            r1 -= int(rng.integers(0, room//2+1))
        if c1-c0 >= 3 and rng.random() < .4:
            if rng.random() < .5:
                c0 += 1
            else:
                c1 -= 1
    if min(r1-r0, c1-c0) < 1 or r1-r0+c1-c0 < 4:
        return None
    visible = geometry[:nvisible]
    ymin = min([r0]+[y for y, x, h, w in visible])
    ymax = max([r1]+[y+h-1 for y, x, h, w in visible])
    xmax = max([c1]+[x+w-1 for y, x, h, w in visible])
    height, width = ymax-ymin+1, xmax+1
    if max(height, width) > 30:
        return None
    canvas = np.full((height, width), bg, dtype=int)
    for i, (y, x, h, w) in enumerate(visible):
        canvas[y-ymin:y-ymin+h, x:x+w] = _paint(kind, arg, h, w, colors[i], bg, bg)
    answer = np.full((r1-r0+1, c1-c0+1), bg, dtype=int)
    for i in range(nvisible+skip, total):
        y, x, h, w = geometry[i]
        local = _paint(kind, arg, h, w, colors[i], mapping[colors[i]], bg)
        lo, hi, lc, hc = max(y, r0), min(y+h, r1+1), max(x, c0), min(x+w, c1+1)
        if lo < hi and lc < hc:
            answer[lo-r0:hi-r0, lc-c0:hc-c0] = local[lo-y:hi-y, lc-x:hc-x]
    bounds = (r0-ymin, r1-ymin, c0, c1)
    # Turn the scene but redraw UL/LR annotations in final input coordinates.
    mask = np.zeros(canvas.shape, dtype=bool)
    mask[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1] = True
    canvas, answer, mask = [_unorient(v, direction) for v in (canvas, answer, mask)]
    pts = np.argwhere(mask)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    bounds = int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])
    padding = [0, 0, 0, 0]  # top, bottom, left, right
    padding[_SIDES.index(side)] = 3
    for axis, slots in ((0, (0, 1)), (1, (2, 3))):
        remaining = 30-canvas.shape[axis]-sum(padding[j] for j in slots)
        if remaining < 0:
            return None
        for j in slots:
            if padding[j] == 0:
                extra = int(rng.integers(0, min(remaining, 2)+1))
                padding[j] = extra
                remaining -= extra
    canvas = np.pad(canvas, ((padding[0], padding[1]), (padding[2], padding[3])),
                    constant_values=bg)
    bounds = (bounds[0]+padding[0], bounds[1]+padding[0],
              bounds[2]+padding[2], bounds[3]+padding[2])
    for q in _markers(bounds):
        canvas[q] = marker
    axis_length = canvas.shape[1] if side in ("top", "bottom") else canvas.shape[0]
    if axis_length < nkey:
        return None
    start = int(rng.integers(0, axis_length-nkey+1))
    _place_key(canvas, mapping, side, start)
    try:
        predicted, info = _analyze(canvas)
    except AssertionError:
        return None
    if not np.array_equal(answer, predicted):
        return None
    if info["skipped"] != skip:
        return None  # A transverse crop must not silently remove the first target.
    walls = _unorient(info["walls"], info["direction"])
    if not np.any(walls != bg) or not np.any(answer != walls):
        return None
    if first:
        if not _lesson_features(info)["relay_witness"] or not _lesson_features(info)["diagonal_witness"]:
            return None
    if second and not _lesson_features(info)["crop_witness"]:
        return None
    return {"input": canvas, "output": answer}


def _grid_key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= value <= high:
        raise ValueError(name+" out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    directions = list(map(str, rng.permutation(["right", "left", "down", "up"])))
    sides = list(map(str, rng.permutation(_SIDES)))
    profile = int(rng.integers(0, 4))
    train_pool = directions[:2] if profile == 3 else [directions[0]]
    if profile == 1:
        test_pool = [d for d in directions if (d in ("right", "left")) !=
                     (directions[0] in ("right", "left"))]
    elif profile == 2:
        test_pool = [{"right": "left", "left": "right", "up": "down", "down": "up"}[directions[0]]]
    else:
        test_pool = train_pool if profile == 0 else directions
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        pool = train_pool if split == "train" else test_pool
        for index in range(count):
            direction = pool[index % len(pool)]
            side = sides[index % 4] if split == "train" else str(rng.choice(sides))
            for attempt in range(300):
                pair = _sample_pair(rng, split, index, direction, side, attempt >= 150)
                if pair is not None and _grid_key(pair["input"]) not in seen:
                    seen.add(_grid_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded relay construction exhausted")
    return puzzle


def _lesson_features(info):
    m, mapping, bb, bg = info["model"], info["key"]["mapping"], info["canonical_bounds"], info["background"]
    target, _ = _render_model(m, bb, bg, mapping)
    cycle, _ = _render_model(m, bb, bg, mapping, "observed_cycle")
    recursive, _ = _render_model(m, bb, bg, mapping, "recursive")
    eight, _ = _render_model(m, bb, bg, mapping, "eight")
    # This intentionally wrong policy treats the observation window as a world.
    crop_wall = info["walls"] != bg
    crop_holes = _cavities(crop_wall)
    proper_holes = (target != info["walls"])
    return dict(relay_witness=not np.array_equal(target, cycle),
                chain_witness=not np.array_equal(target, recursive),
                diagonal_witness=not np.array_equal(target, eight),
                crop_witness=bool(np.any(proper_holes & ~crop_holes)),
                growth=bool(m["h"][1] or m["w"][1]),
                hole_count=info["holes"], unseen_wall=bool(set(info["target_colors"])-set(m["colors"])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _array(pair["input"]), _array(pair["output"])
            signature = _grid_key(a)
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "incorrect output"
            assert info["marker"] not in b, "annotation survived"
            assert set(map(int, b.flat)) <= set(map(int, a.flat))
            assert b.shape == (info["bounds"][1]-info["bounds"][0]+1,
                               info["bounds"][3]-info["bounds"][2]+1)
            unfilled = _unorient(info["walls"], info["direction"])
            assert np.any(unfilled != info["background"]), "no continued boundary"
            assert np.any(b != unfilled), "omitting B would not change the answer"
            assert np.any(b != info["background"]), "omitting A would not change the answer"
            if split == "train":
                assert info["skipped"] == 0
            else:
                assert info["skipped"] in (1, 2)
            analyses[split].append(info)
    first, second = analyses["train"][:2]
    f0, f1 = _lesson_features(first), _lesson_features(second)
    assert f0["growth"] and f0["relay_witness"] and f0["unseen_wall"]
    assert f0["chain_witness"] and f0["diagonal_witness"], "missing lookup/topology lesson"
    assert f1["growth"] and f1["hole_count"] >= 2 and f1["crop_witness"]
    assert first["key"]["side"] != second["key"]["side"], "missing direction lesson"


def self_test():
    """Bounded semantic witnesses and input-derived variation over 42 seeds."""
    # Direct geometry/topology cross-check, not an independent external solver.
    for kind, arg in _descriptors():
        for h, w in ((3, 3), (5, 5), (7, 5), (9, 7), (13, 9)):
            wall = _template(kind, arg, h, w)
            if wall is None:
                continue
            strokes = _paint(kind, arg, h, w, 1, 0, 0)
            filled = _paint(kind, arg, h, w, 1, 2, 0)
            assert np.array_equal(strokes == 1, wall)
            assert np.array_equal(filled == 2, _cavities(wall))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, kinds, directions, key_sides = set(), set(), set(), set(), set()
    deltas, drift, skips, spans, lengths, backgrounds, structures = set(), set(), set(), set(), set(), set(), set()
    counters = dict(crop=0, diagonal=0, relay=0, chain=0, unseen=0, multi=0, width_growth=0, shrink=0)
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for p in puzzle[split]:
                a, b = p["input"], p["output"]
                copy = a.copy()
                answer, info = _analyze(a)
                assert np.array_equal(a, copy) and np.array_equal(answer, b)
                assert np.array_equal(solve(a.tolist()), b)
                m, f = info["model"], _lesson_features(info)
                shapes.add((a.shape, b.shape))
                kinds.add(m["kind"])
                directions.add(info["direction"])
                key_sides.add(info["key"]["side"])
                deltas.add((m["h"][1], m["w"][1]))
                drift.add(m["y"][1])
                skips.add(info["skipped"])
                spans.add(len(info["target_ids"]))
                lengths.add(len(info["key"]["mapping"]))
                backgrounds.add(info["background"])
                roles = {info["background"]: 0}
                normalized = tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, normalized))
                for name, prop in (("crop", "crop_witness"), ("diagonal", "diagonal_witness"),
                                   ("relay", "relay_witness"), ("chain", "chain_witness"),
                                   ("unseen", "unseen_wall")):
                    counters[name] += int(f[prop])
                counters["multi"] += int(f["hole_count"] > 1)
                counters["width_growth"] += int(m["w"][1] != 0)
                counters["shrink"] += int(m["h"][1] < 0 or m["w"][1] < 0)
    assert len(signatures) == 42 and kinds == set(_KINDS)
    assert len(shapes) >= 60 and len(structures) >= 150
    assert directions == {"right", "left", "up", "down"} and key_sides == set(_SIDES)
    assert len(deltas) >= 8 and len(drift) >= 4 and skips == {0, 1, 2}
    assert spans == {1, 2} and lengths == {4, 5, 6} and len(backgrounds) >= 8
    assert all(counters[k] >= 42 for k in ("crop", "diagonal", "relay", "chain", "unseen", "multi")), counters
    assert counters["width_growth"] >= 5 and counters["shrink"] >= 10, counters
    repeated = generate_puzzle(17)
    assert tuple(_grid_key(p["input"]) for s in puzzles[17] for p in puzzles[17][s]) == tuple(
        _grid_key(p["input"]) for s in repeated for p in repeated[s])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(501, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    broken["test"][0]["output"][0, 0] = (broken["test"][0]["output"][0, 0]+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), geometry_shapes=len(shapes),
                normalized_structures=len(structures), kinds=sorted(kinds), directions=sorted(directions),
                key_sides=sorted(key_sides), geometry_deltas=sorted(deltas), drift=sorted(drift),
                skips=sorted(skips), target_counts=sorted(spans), key_lengths=sorted(lengths),
                backgrounds=sorted(backgrounds), witnesses=counters,
                semantic_verification="generator-owned checks only")
