"""Growth-front printing: registered temporal XOR followed by textured cells.

Only NumPy and the standard library are used. Generation paints latent masks;
prediction fits the input and evaluates separate cell predicates. Nothing is
saved by this module, including self_test (42 puzzles, processed one at a time).
"""

import hashlib
import numpy as np


_KINDS = (("solid", 1), ("frame", 1), ("comb", 2), ("comb", 3),
          ("ladder", 2), ("ladder", 3), ("stair", 1), ("stair", 2))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _corners(box):
    r0, r1, c0, c1 = box
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _bbox(binary):
    p = np.argwhere(binary)
    assert len(p)
    r0, c0 = p.min(axis=0)
    r1, c1 = p.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _crop(a, box):
    r0, r1, c0, c1 = box
    return a[r0:r1 + 1, c0:c1 + 1].copy()


def _mask(kind, h, w, spacing):
    """Generation-side strokes. Every permitted mask has a tight h-by-w box."""
    if h < 1 or w < 1:
        return None
    a = np.zeros((h, w), dtype=bool)
    if kind == "solid":
        a[:] = True
    elif kind == "frame":
        if min(h, w) < 3:
            return None
        a[0] = a[-1] = True
        a[:, 0] = a[:, -1] = True
    elif kind in ("comb", "ladder"):
        if h < spacing + 2 or w < (3 if kind == "ladder" else 2):
            return None
        a[:, 0] = True
        if kind == "ladder":
            a[:, -1] = True
        for r in range(0, h, spacing):
            a[r] = True
        a[-1] = True
    elif kind == "stair":
        if w < 2 or h < (w - 1) * spacing + 1:
            return None
        for r in range(h):
            a[r, :min(w, r // spacing + 1)] = True
    else:
        return None
    return a


def _occupied(kind, spacing, h, w, r, c):
    """Solver-side predicates, including the finite silhouette boundary."""
    inside = (r >= 0) & (c >= 0) & (r < h) & (c < w)
    if kind == "solid":
        shape = np.ones_like(inside)
    elif kind == "frame":
        shape = (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    elif kind == "comb":
        shape = (c == 0) | (r % spacing == 0) | (r == h - 1)
    elif kind == "ladder":
        shape = (c == 0) | (c == w - 1) | (r % spacing == 0) | (r == h - 1)
    else:
        shape = c <= r // spacing
    return inside & shape


def _admissible(kind, spacing, h, w):
    if min(h, w) < 1:
        return False
    if kind == "frame":
        return min(h, w) >= 3
    if kind in ("comb", "ladder"):
        return h >= spacing + 2 and w >= (3 if kind == "ladder" else 2)
    if kind == "stair":
        return w >= 2 and h >= (w - 1) * spacing + 1
    return True


def _trans(a, turns, flip):
    a = np.rot90(a, turns)
    return np.fliplr(a) if flip else a


def _untrans(a, turns, flip):
    if flip:
        a = np.fliplr(a)
    return np.rot90(a, -turns).copy()


def _asymmetric(a):
    return all(not np.array_equal(a, _trans(a, k, f))
               for k in range(4) for f in (False, True) if k or f)


def _read_roles(a):
    bg = int(a[0, 0])
    assert all(np.all(side == bg) for side in (a[0], a[-1], a[:, 0], a[:, -1])), "uniform background rim required"
    markers = []
    for raw in np.unique(a):
        color = int(raw)
        pos = np.argwhere(a == color)
        if color == bg or len(pos) != 6:
            continue
        box = _bbox(a == color)
        r0, r1, c0, c1 = box
        if r1 > r0 and c1 > c0 and r1 - r0 + c1 - c0 >= 4:
            if set(map(tuple, pos)) == _corners(box):
                markers.append((color, box))
    assert len(markers) == 1, "one canonical pair of L markers required"
    marker, box = markers[0]
    assert set(map(int, np.unique(_crop(a, box)))) <= {bg, marker}
    candidates = []
    colors = list(map(int, np.unique(a)))
    for signal in colors:
        if signal in (bg, marker) or np.count_nonzero(a == signal) <= 16:
            continue
        for pin in colors:
            if pin in (bg, marker, signal) or not 3 <= np.count_nonzero(a == pin) <= 4:
                continue
            payload = (a != bg) & (a != marker) & (a != signal) & (a != pin)
            if len(np.unique(a[payload])) < 2:
                continue
            sb = _bbox(payload)
            stamp = _crop(a, sb)
            if not all(2 <= n <= 4 for n in stamp.shape):
                continue
            if np.any((stamp == signal) | (stamp == marker) | (stamp == pin)):
                continue
            if not all(np.any(side != bg) for side in
                       (stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1])):
                continue
            # The template is separated from every silhouette and the query.
            others = (a == signal) | (a == marker) | (a == pin)
            ob = _bbox(others)
            if not (sb[1] + 1 < ob[0] or ob[1] + 1 < sb[0] or
                    sb[3] + 1 < ob[2] or ob[3] + 1 < sb[2]):
                continue
            candidates.append((signal, pin, stamp, sb))
    assert len(candidates) == 1, "unique silhouette/template roles required"
    signal, pin, stamp, sb = candidates[0]
    return bg, marker, box, signal, pin, stamp, sb


def _linear(values):
    d = values[1] - values[0]
    return d if all(v == values[0] + i * d for i, v in enumerate(values)) else None


def _canonical_models(a, bg, box, signal, pin, stamp):
    r0, r1, c0, c1 = box
    silhouette = (a == signal) | (a == pin)
    cols = np.flatnonzero(np.any(silhouette, axis=0))
    if not len(cols) or cols[-1] >= c0:
        return
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    if not 3 <= len(groups) <= 4:
        return
    sh, sw = stamp.shape
    heights, widths, tops, starts, masks = [], [], [], [], []
    for cs in groups:
        rs = np.flatnonzero(np.any(silhouette[:, cs], axis=1))
        patch = a[rs[0]:rs[-1] + 1, cs[0]:cs[-1] + 1].copy()
        if patch[0, 0] != pin or np.count_nonzero(patch == pin) != 1:
            return
        patch[patch == pin] = signal
        hp, wp = patch.shape
        if hp % sh or wp % sw or not set(map(int, np.unique(patch))) <= {bg, signal}:
            return
        h, w = hp // sh, wp // sw
        blocks = (patch == signal).reshape(h, sh, w, sw)
        coarse = blocks[:, 0, :, 0]
        if not np.array_equal(blocks, np.broadcast_to(coarse[:, None, :, None], blocks.shape)):
            return
        masks.append(coarse)
        heights.append(h)
        widths.append(w)
        tops.append(int(rs[0]))
        starts.append(int(cs[0]))
    dh, dw, dy = _linear(heights), _linear(widths), _linear(tops)
    if dh is None or dw is None or dy is None or (dh == 0 and dw == 0):
        return
    gaps = [starts[i + 1] - starts[i] - widths[i] * sw for i in range(len(groups) - 1)]
    if len(set(gaps)) != 1 or gaps[0] < 1:
        return
    for kind, spacing in _KINDS:
        fits = True
        for m in masks:
            h, w = m.shape
            rr, cc = np.indices(m.shape)
            if not _admissible(kind, spacing, h, w) or not np.array_equal(m, _occupied(kind, spacing, h, w, rr, cc)):
                fits = False
                break
        if fits:
            yield dict(kind=kind, spacing=spacing, h=heights[0], w=widths[0],
                       dh=dh, dw=dw, y=tops[0], dy=dy, x=starts[0], gap=gaps[0],
                       n=len(groups), sh=sh, sw=sw)


def _geometry(m, i):
    h, w = m["h"] + i * m["dh"], m["w"] + i * m["dw"]
    y = m["y"] + i * m["dy"]
    x = m["x"] + i * (m["w"] * m["sw"] + m["gap"]) + i * (i - 1) // 2 * m["dw"] * m["sw"]
    return y, x, h, w


def _predict(m, box, stamp, bg, mode="xor"):
    r0, r1, c0, c1 = box
    rr, cc = np.indices((r1 - r0 + 1, c1 - c0 + 1))
    rr, cc = rr + r0, cc + c0
    out = np.full(rr.shape, bg, dtype=np.int8)
    occupied = np.zeros(rr.shape, dtype=bool)
    stages = []
    for i in range(m["n"], 32):
        y, x, h, w = _geometry(m, i)
        _, _, ph, pw = _geometry(m, i - 1)
        if x > c1:
            break
        if not all(_admissible(m["kind"], m["spacing"], hh, ww) for hh, ww in ((h, w), (ph, pw))):
            return None
        if m["gap"] < max(0, -m["dw"] * m["sw"]) + 1:
            return None
        r, c = (rr - y) // m["sh"], (cc - x) // m["sw"]
        now = _occupied(m["kind"], m["spacing"], h, w, r, c)
        before = _occupied(m["kind"], m["spacing"], ph, pw, r, c)
        change = now ^ before if mode == "xor" else now & ~before if mode == "added" else now
        if np.any(occupied & change):
            return None
        occupied |= change
        out[change] = stamp[(rr[change] - y) % m["sh"], (cc[change] - x) % m["sw"]]
        if x + max(w, pw) * m["sw"] > c0 and y <= r1 and y + max(h, ph) * m["sh"] > r0:
            stages.append(i)
    if not stages:
        return None
    return out, stages


def _analyze(input_grid):
    a = _grid(input_grid)
    bg, marker, box, signal, pin, stamp, sb = _read_roles(a)
    query = a == marker
    answers = []
    for turns in range(4):
        for flip in (False, True):
            canonical = _trans(a, turns, flip)
            cb = _bbox(_trans(query, turns, flip))
            st = _trans(stamp, turns, flip)
            for model in _canonical_models(canonical, bg, cb, signal, pin, st):
                predicted = _predict(model, cb, st, bg)
                if predicted is None:
                    continue
                out, stages = predicted
                answers.append((_untrans(out, turns, flip), model, turns, flip, cb, st, stages))
    assert answers, "no supported changing-silhouette sequence"
    result = answers[0][0]
    assert all(np.array_equal(result, p[0]) for p in answers), "ambiguous transition crop"
    _, model, turns, flip, cb, st, stages = answers[0]
    info = dict(bg=bg, marker=marker, signal=signal, pin=pin, box=box, stamp=stamp,
                stamp_box=sb, model=model, turns=turns, flip=flip,
                canonical_box=cb, canonical_stamp=st, stages=stages,
                candidates=len(answers))
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every geometric, color-role, and template parameter from this input."""
    return _analyze(input_grid)[0]


def _make_stamp(rng, shape, bg, payload, witness=False):
    h, w = shape
    r, c = np.indices(shape)
    for _ in range(40):
        style = int(rng.integers(4))
        if style == 0:
            labels = r + c
        elif style == 1:
            labels = r if rng.integers(2) else c
        elif style == 2:
            labels = np.minimum.reduce([r, c, h - 1 - r, w - 1 - c])
        else:
            labels = rng.integers(len(payload), size=shape)
        a = np.asarray(payload, dtype=np.int8)[labels % len(payload)].copy()
        if witness or rng.random() < .65:
            n = 1 if witness else int(rng.integers(1, max(2, h * w // 3)))
            a.flat[rng.choice(h * w, size=n, replace=False)] = bg
        if len(set(map(int, np.unique(a))) - {bg}) < 2:
            continue
        if not all(np.any(side != bg) for side in (a[0], a[-1], a[:, 0], a[:, -1])):
            continue
        if witness and not _asymmetric(a):
            continue
        return a
    a = np.full(shape, payload[0], dtype=np.int8)
    a[0, 0], a[0, 1], a[-1, -1] = payload[1], payload[2], bg
    return a


def _difference(a, b):
    h, w = max(len(a), len(b)), max(a.shape[1], b.shape[1])
    previous = np.zeros((h, w), dtype=bool)
    current = previous.copy()
    previous[:len(a), :a.shape[1]] = a
    current[:len(b), :b.shape[1]] = b
    return previous ^ current


def _paint_query(objects, indices, box, stamp, bg):
    """Labels: explicit changed-cell block writes, never calling solve."""
    r0, r1, c0, c1 = box
    sh, sw = stamp.shape
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=np.int8)
    for i in indices:
        y, x, mask = objects[i]
        for r, c in np.argwhere(_difference(objects[i - 1][2], mask)):
            top, left = y + int(r) * sh, x + int(c) * sw
            lo_r, hi_r = max(top, r0), min(top + sh, r1 + 1)
            lo_c, hi_c = max(left, c0), min(left + sw, c1 + 1)
            if lo_r < hi_r and lo_c < hi_c:
                out[lo_r - r0:hi_r - r0, lo_c - c0:hi_c - c0] = stamp[lo_r - top:hi_r - top, lo_c - left:hi_c - left]
    return out


def _construct(rng, config, stamp, bg, signal, marker, pin, transform, crop_mode, variant=None):
    kind, spacing, h0, w0, dh, dw, dy, gap, n, skip, span = config
    sh, sw = stamp.shape
    objects = []
    x = 0
    for i in range(n + skip + span):
        mask = _mask(kind, h0 + i * dh, w0 + i * dw, spacing)
        if mask is None or max(mask.shape) > 20:
            return None
        objects.append((i * dy, x, mask))
        x += mask.shape[1] * sw + gap
    if gap < max(0, -dw * sw) + 1:
        return None
    indices = list(range(n + skip, n + skip + span))
    r0 = min(objects[i][0] for i in indices)
    r1 = max(objects[i][0] + max(len(objects[i][2]), len(objects[i - 1][2])) * sh - 1 for i in indices)
    c0 = objects[indices[0]][1]
    c1 = max(objects[i][1] + max(objects[i][2].shape[1], objects[i - 1][2].shape[1]) * sw - 1 for i in indices)
    if crop_mode == "partial":
        # Deliberately change the phase inside a tile, rather than changing its origin.
        r0 += 1
    elif crop_mode == "slice" and span == 1:
        if r1 - r0 >= 4 and rng.random() < .6:
            r0 += int(rng.integers(0, r1 - r0 - 1))
        if c1 - c0 >= 4 and rng.random() < .5:
            c0 += int(rng.integers(0, c1 - c0 - 1))
    if r1 - r0 < 1 or c1 - c0 < 1 or r1 - r0 + c1 - c0 < 4:
        return None
    box = (r0, r1, c0, c1)
    answer = _paint_query(objects, indices, box, stamp, bg)
    if len(set(map(int, np.unique(answer))) - {bg}) < 2 or not np.any(answer == bg):
        return None
    full_tile = False
    for i in indices:
        y, x, mask = objects[i]
        for r, c in np.argwhere(_difference(objects[i - 1][2], mask)):
            top, left = y + int(r) * sh, x + int(c) * sw
            full_tile |= r0 <= top and top + sh - 1 <= r1 and c0 <= left and left + sw - 1 <= c1
    if not full_tile:
        return None
    if not any(answer[r - r0, c - c0] != bg for r, c in _corners(box)):
        return None
    visible = objects[:n]
    ymin = min([r0] + [y for y, x, a in visible])
    ymax = max([r1] + [y + len(a) * sh - 1 for y, x, a in visible])
    xmax = max([c1] + [x + a.shape[1] * sw - 1 for y, x, a in visible])
    header = sh + 3 + (0 if variant is None else variant // 20)
    height, width = header + ymax - ymin + 2, xmax + 3
    if variant is not None:
        width = max(width, 1 + variant % 20 + sw + 1)
    width = max(width, sw + 2)
    if max(height, width) > 30:
        return None
    canvas = np.full((height, width), bg, dtype=np.int8)
    dr, dc = header - ymin, 1
    for y, x, a in visible:
        expanded = np.repeat(np.repeat(a, sh, axis=0), sw, axis=1)
        canvas[y + dr:y + dr + len(expanded), x + dc:x + dc + expanded.shape[1]] = np.where(expanded, signal, bg)
        canvas[y + dr, x + dc] = pin
    sc = int(rng.integers(1, width - sw)) if variant is None else 1 + variant % 20
    canvas[1:1 + sh, sc:sc + sw] = stamp
    box = (r0 + dr, r1 + dr, c0 + dc, c1 + dc)
    query = np.zeros(canvas.shape, dtype=bool)
    query[box[0]:box[1] + 1, box[2]:box[3] + 1] = True
    turns, flip = transform
    canvas = _trans(canvas, turns, flip).copy()
    answer = _trans(answer, turns, flip).copy()
    box = _bbox(_trans(query, turns, flip))
    for r, c in _corners(box):
        canvas[r, c] = marker
    return {"input": canvas, "output": answer}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def _sample_config(rng, witness, test, regime, sh, sw, focus, solid_mode):
    skip = 1 if test and regime == "distance" else 0
    if witness == 0:
        return ("solid", 1, int(rng.integers(1, 7)), int(rng.integers(1, 3)),
                int(rng.integers(1, 3)), 0, int(rng.integers(-1, 2)), int(rng.integers(1, 3)), 3, 0, 1)
    if witness == 1:
        return ("frame", 1, int(rng.integers(3, 8)), 3, 1, 0,
                int(rng.choice([-1, 1])), 1, 3, 0, 1)
    if solid_mode == "width":
        # Width-changing stages compete with the 30-cell progression axis;
        # give this feasible primitive explicit representation.
        return ("solid", 1, int(rng.integers(2, 8)), 1, 0, 1,
                int(rng.integers(-1, 2)), 1, 3, 0, 1)
    if solid_mode == "multi":
        return ("solid", 1, int(rng.integers(1, 5)), 1, 1, 0,
                int(rng.integers(-1, 2)), int(rng.integers(1, 3)), 3, skip,
                1 if regime == "phase" else 2)
    kind, spacing = focus
    n = 4 if rng.random() < .2 else 3
    span = 2 if rng.random() < .3 and regime != "phase" else 1
    dh = int(rng.choice([-1, 0, 1, 1, 2]))
    dw = int(rng.choice([-1, 0, 0, 0, 1]))
    if dh == dw == 0:
        dh = 1
    h = int(rng.integers(2, 5))
    w = int(rng.integers(1, 4))
    if kind == "frame":
        h, w = max(h, 3), max(w, 3)
    elif kind in ("comb", "ladder"):
        h, w = max(h, spacing + 2), max(w, 3 if kind == "ladder" else 2)
    elif kind == "stair":
        w = max(w, 2)
        h = max(h, (w - 1) * spacing + 1)
    total = n + skip + span
    h += (total - 1) * max(0, -dh)
    w += (total - 1) * max(0, -dw)
    return (kind, spacing, h, w, dh, dw, int(rng.integers(-2, 3)),
            int(rng.integers(1, 4)) + max(0, -dw * sw), n, skip, span)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    stable = bool(rng.integers(2))
    regime = str(rng.choice(["axis", "distance", "phase", "scale"]))
    train_axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        test = split == "test"
        for index in range(count):
            witness = index if not test and index < 2 else None
            roles = palette if stable else list(map(int, rng.permutation(10)))
            bg, signal, marker, pin = roles[:4]
            payload = roles[4:4 + int(rng.integers(3 if witness is not None else 2, 5))]
            axis = 1 - train_axis if test and regime == "axis" else train_axis
            turns, flip = axis + 2 * int(rng.integers(2)), bool(rng.integers(2))
            crop_mode = "partial" if test and regime == "phase" else "whole" if witness is not None or regime == "phase" else "slice"
            available = [k for k in _KINDS if not (test and regime == "distance" and k[0] in ("frame", "ladder"))]
            focus = available[int(rng.integers(len(available)))]
            solid_mode = "mixed"
            if witness is None and focus[0] == "solid":
                modes = ["multi", "mixed"] if test and regime == "distance" else ["width", "multi", "mixed"]
                solid_mode = str(rng.choice(modes))
            chosen = None
            # 144 stochastic attempts, then a finite 40-member constructive pool.
            # Only one proposal is retained; this loop has no disk I/O or logs.
            for attempt in range(184):
                fallback = attempt >= 144
                if fallback:
                    variant = attempt - 144
                    shape = (3 if test and regime == "scale" else 2, 2)
                    stamp = np.asarray([[payload[1], payload[2] if len(payload) > 2 else payload[0]],
                                        [payload[0], bg]], dtype=np.int8)
                    if shape[0] == 3:
                        stamp = np.vstack([stamp[:1], stamp])
                    kind, w, h = ("frame", 3, 3) if witness == 1 else ("solid", 1, 2)
                    config = (kind, 1, h, w, 1, 0, -1 if witness == 1 else 0, 1,
                              3, int(test and regime == "distance"), 1)
                else:
                    variant = None
                    if regime == "scale":
                        shape = (int(rng.integers(3, 5)), 2) if test else (2, 2)
                    elif witness == 1:
                        shape = (int(rng.integers(2, 4)), 2)
                    elif witness is None and (focus[0] in ("frame", "ladder") or test and regime == "distance"):
                        shape = (int(rng.integers(2, 4)), 2)
                    else:
                        shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
                    stamp = _make_stamp(rng, shape, bg, payload, witness is not None)
                    config = _sample_config(rng, witness, test, regime, *shape, focus, solid_mode)
                candidate = _construct(rng, config, stamp, bg, signal, marker, pin,
                                       (turns, flip), crop_mode, variant)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                try:
                    prediction, info = _analyze(candidate["input"])
                except AssertionError:
                    continue
                if not np.array_equal(prediction, candidate["output"]):
                    continue
                # Do not let the transition rule be equivalent to printing the
                # entire future shape, even in a locally cropped example.
                whole = _predict(info["model"], info["canonical_box"], info["canonical_stamp"], bg, "whole")
                if whole is None or np.array_equal(prediction, _untrans(whole[0], info["turns"], info["flip"])):
                    continue
                chosen = candidate
                break
            if chosen is None:
                raise RuntimeError("bounded construction pool exhausted")
            seen.add(_key(chosen["input"]))
            puzzle[split].append(chosen)
    return puzzle


def _alternative(info, mode):
    p = _predict(info["model"], info["canonical_box"], info["canonical_stamp"], info["bg"], mode)
    assert p is not None
    return _untrans(p[0], info["turns"], info["flip"])


def _full_tile(info):
    m, box = info["model"], info["canonical_box"]
    for i in info["stages"]:
        y, x, h, w = _geometry(m, i)
        _, _, ph, pw = _geometry(m, i - 1)
        rr, cc = np.indices((max(h, ph), max(w, pw)))
        changed = (_occupied(m["kind"], m["spacing"], h, w, rr, cc) ^
                   _occupied(m["kind"], m["spacing"], ph, pw, rr, cc))
        for r, c in np.argwhere(changed):
            top, left = y + int(r) * m["sh"], x + int(c) * m["sw"]
            if box[0] <= top and top + m["sh"] - 1 <= box[1] and box[2] <= left and left + m["sw"] - 1 <= box[3]:
                return True
    return False


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            predicted, info = _analyze(a)
            assert np.array_equal(predicted, b), "incorrect transition printing"
            assert info["marker"] not in b and info["signal"] not in b and info["pin"] not in b
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(info["stamp"]))) | {info["bg"]}
            assert len(set(map(int, np.unique(b))) - {info["bg"]}) >= 2 and np.any(b == info["bg"])
            assert _full_tile(info), "at least one full textured cell must remain visible"
            assert not np.array_equal(b, _alternative(info, "whole")), "whole-shape shortcut not distinguished"
            # B ablation: flatten each printed cell to a single payload color.
            flat = np.full(info["canonical_stamp"].shape, next(int(v) for v in info["stamp"].flat if v != info["bg"]), dtype=np.int8)
            wrong = _predict(info["model"], info["canonical_box"], flat, info["bg"])[0]
            assert not np.array_equal(b, _untrans(wrong, info["turns"], info["flip"])), "texture is not consequential"
            information[split].append(info)
    train = information["train"]
    assert any(not np.array_equal(np.asarray(p["output"]), _alternative(i, "added"))
               for p, i in zip(puzzle["train"], train)), "show disappeared old border cells"
    assert any(np.any(i["stamp"] == i["bg"]) and _asymmetric(i["stamp"]) for i in train), "show holes and orientation"
    assert any(i["model"]["kind"] == "frame" for i in train), "hollow growth witness required"
    # A real shift in distance, axis, tile scale, or fine-grid crop phase.
    def phase(i):
        m, b = i["model"], i["canonical_box"]
        y, x, _, _ = _geometry(m, i["stages"][0])
        return (b[0] - y) % m["sh"], (b[2] - x) % m["sw"]
    def axis(i):
        return i["turns"] % 2
    tests = information["test"]
    shift = (min(i["stages"][0] - i["model"]["n"] for i in tests) > max(i["stages"][0] - i["model"]["n"] for i in train)
             or {axis(i) for i in tests}.isdisjoint({axis(i) for i in train})
             or min(i["stamp"].size for i in tests) > max(i["stamp"].size for i in train)
             or (all(phase(i) == (0, 0) for i in train) and all(phase(i) != (0, 0) for i in tests)))
    assert shift, "missing observable train/test shift"


def self_test():
    """Bounded semantic/diversity tests; no retained puzzle corpus or trace files."""
    # Hand-derived semantic fixtures, independent of the label painter:
    # a 5-high frame becomes 6-high (or 5 becomes 4). The old bottom-center
    # disappears and the other changed row consists of three complete tiles.
    tile = np.asarray([[3, 4], [5, 0]], dtype=np.int8)
    for shrinking_fixture, phase in ((False, False), (False, True), (True, False)):
        initial, delta = (7, -1) if shrinking_fixture else (3, 1)
        fixture = _construct(np.random.default_rng(13),
                             ("frame", 1, initial, 3, delta, 0, 1, 1, 3, 0, 1),
                             tile, 0, 1, 2, 6, (0, False),
                             "partial" if phase else "whole")
        assert fixture is not None
        height = 5 if shrinking_fixture else 6
        expected = np.zeros((height * 2, 6), dtype=np.int8)
        expected[-4:-2, 2:4] = tile
        for c in range(3):
            expected[-2:, 2 * c:2 * c + 2] = tile
        if phase:
            expected = expected[1:]
        original = fixture["input"].copy()
        assert np.array_equal(fixture["output"], expected)
        assert np.array_equal(solve(original), expected)
        assert np.array_equal(original, fixture["input"])
    # The finite fallback pool must actually have more distinct valid inputs
    # than any legal request can have preceding inputs (at most thirteen).
    for kind, w, h, skip, sh, mode in (("solid", 1, 2, 0, 2, "whole"),
                                      ("frame", 3, 3, 0, 2, "whole"),
                                      ("solid", 1, 2, 1, 2, "slice"),
                                      ("solid", 1, 2, 0, 2, "partial"),
                                      ("solid", 1, 2, 0, 3, "slice")):
        stamp = tile if sh == 2 else np.vstack([tile[:1], tile])
        pool = set()
        for variant in range(40):
            pair = _construct(np.random.default_rng(14),
                              (kind, 1, h, w, 1, 0, -1 if kind == "frame" else 0,
                               1, 3, skip, 1), stamp, 0, 1, 2, 6,
                              (0, False), mode, variant)
            assert pair is not None
            assert np.array_equal(solve(pair["input"]), pair["output"])
            pool.add(_key(pair["input"]))
        assert len(pool) == 40
    metrics = {k: set() for k in ("kinds", "steps", "stamps", "input_shapes", "output_shapes",
                                "directions", "backgrounds", "patterns", "puzzles")}
    shrinking = changing_width = partial = multistage = removed = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                _, info = _analyze(a)
                m = info["model"]
                for role in ("input", "output"):
                    digest.update(repr(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
                metrics["kinds"].add(m["kind"])
                metrics["steps"].add((m["dh"], m["dw"], m["dy"]))
                metrics["stamps"].add(info["stamp"].shape)
                metrics["input_shapes"].add(a.shape)
                metrics["output_shapes"].add(b.shape)
                metrics["directions"].add((info["turns"], info["flip"]))
                metrics["backgrounds"].add(info["bg"])
                masks = []
                for j in range(m["n"]):
                    _, _, h, w = _geometry(m, j)
                    rr, cc = np.indices((h, w))
                    masks.append(_occupied(m["kind"], m["spacing"], h, w, rr, cc))
                metrics["patterns"].add(hashlib.sha256(b"".join(repr(v.shape).encode() + v.tobytes() for v in masks)).digest())
                shrinking += m["dh"] < 0 or m["dw"] < 0
                changing_width += m["dw"] != 0
                partial += any((info["canonical_box"][j] - _geometry(m, info["stages"][0])[j // 2]) % (m["sh"] if j == 0 else m["sw"]) for j in (0, 2))
                multistage += len(info["stages"]) > 1
                removed += not np.array_equal(b, _alternative(info, "added"))
        metrics["puzzles"].add(digest.digest())
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ("input", "output"))
    assert len(metrics["puzzles"]) == 42
    assert metrics["kinds"] == {"solid", "frame", "comb", "ladder", "stair"}, metrics["kinds"]
    for name, minimum in (("steps", 12), ("stamps", 6), ("input_shapes", 45),
                          ("output_shapes", 30), ("directions", 8), ("backgrounds", 8), ("patterns", 25)):
        assert len(metrics[name]) >= minimum, (name, len(metrics[name]))
    assert shrinking >= 5 and changing_width >= 5 and partial >= 10 and multistage >= 5 and removed >= 42, (shrinking, changing_width, partial, multistage, removed)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1900 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    damaged = generate_puzzle(2901, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": 210, "distinct": {k: len(v) for k, v in metrics.items()},
            "shrinking": int(shrinking), "changing_width": int(changing_width),
            "partial": int(partial), "multiple_target_stages": int(multistage),
            "removed_cell_witnesses": int(removed)}
