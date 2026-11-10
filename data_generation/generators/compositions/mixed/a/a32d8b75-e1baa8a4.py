"""Compress a framed stencil, rotate it, and stamp exchanged unrotated tiles.

Only NumPy and the standard library are used. Generation writes no files.
The solver decodes all conventions from pixels and fixed geometric key meanings.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
_ARROWS = ((0, 1), (1, 2), (2, 1), (1, 0))
_ALTERNATIVES = (
    "no_rows", "no_columns", "no_compression", "global_unique",
    "no_rotation", "opposite_rotation", "rotate_tile", "no_exchange",
    "top_left", "fill_holes", "only_low", "only_high", "compress_tile",
    "fit_instead_of_clip",
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer 2D grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return [i for i in range(len(lines))
            if i == 0 or not np.array_equal(lines[i], lines[i - 1])]


def _compress(a):
    # Compute both sets of maximal-run representatives in the ORIGINAL stencil.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _global_unique(a):
    # Deliberately wrong alternative: deletes separated equal vectors as well.
    def keep(lines):
        seen, result = set(), []
        for i, line in enumerate(lines):
            k = line.tobytes()
            if k not in seen:
                seen.add(k)
                result.append(i)
        return result
    return a[np.ix_(keep(a), keep(a.T))]


def _repeated(a, axis):
    lines = a if axis == 0 else a.T
    return any(np.array_equal(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i + 2, len(lines)))


def _read_panel(panel, frame):
    """Read four disconnected rectangular frames; no template coordinates."""
    h, w = panel.shape
    if h < 7 or w < 7:
        raise ValueError("panel too small")
    ground = int(panel[0, 0])
    if ground == frame:
        raise ValueError("frame and ground coincide")
    if not (np.all(panel[0] == ground) and np.all(panel[-1] == ground)
            and np.all(panel[:, 0] == ground) and np.all(panel[:, -1] == ground)):
        raise ValueError("panel needs a visible outside margin")
    todo = set(map(tuple, np.argwhere(panel == frame)))
    boxes, allowed = [], np.zeros(panel.shape, bool)
    while todo:
        stack, component = [todo.pop()], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in todo:
                    todo.remove(point)
                    stack.append(point)
        points = np.asarray(component)
        r0, c0 = points.min(0)
        r1, c1 = points.max(0) + 1
        if min(r1 - r0, c1 - c0) < 3:
            raise ValueError("not a closed frame")
        box = panel[r0:r1, c0:c1]
        if not (np.all(box[0] == frame) and np.all(box[-1] == frame)
                and np.all(box[:, 0] == frame) and np.all(box[:, -1] == frame)
                and not np.any(box[1:-1, 1:-1] == frame)):
            raise ValueError("invalid frame outline")
        if np.any(allowed[r0:r1, c0:c1]):
            raise ValueError("overlapping frames")
        allowed[r0:r1, c0:c1] = True
        boxes.append(box[1:-1, 1:-1].copy())
        if len(boxes) > 4:
            raise ValueError("too many fields")
    if len(boxes) != 4 or np.any(panel[~allowed] != ground):
        raise ValueError("expected exactly four framed fields on blank ground")
    info, others = {}, []
    for box in boxes:
        points = set(map(tuple, np.argwhere(box != ground)))
        ink = np.unique(box[box != ground])
        kind = None
        if box.shape == (3, 3) and len(ink) == 1:
            for a, (r, c) in enumerate(_CORNERS):
                if points == {(2 * r, 2 * c)}:
                    kind, value = "anchor", a
            for k, point in enumerate(_ARROWS):
                if points == {(1, 1), point}:
                    kind, value = "turn", k
        if kind is not None:
            if kind in info:
                raise ValueError("ambiguous geometric key")
            info[kind] = value
        elif ground not in box and box.shape[0] == box.shape[1] and len(np.unique(box)) == 2:
            if "tile" in info or not 2 <= len(box) <= 5:
                raise ValueError("ambiguous binary square tile")
            info["tile"] = box
        else:
            others.append(box)
    if set(info) != {"anchor", "turn", "tile"} or len(others) != 1:
        raise ValueError("missing tile, stencil, or controls")
    stencil = others[0]
    ink = np.unique(stencil[stencil != ground])
    if len(ink) != 1 or not np.any(stencil == ground):
        raise ValueError("stencil must contain ground and one occupied color")
    info.update(stencil=stencil != ground, ground=ground, frame=frame)
    return info


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    # A full divider cuts off exactly one panel. Horizontal panels use the same
    # geometric fields in the actual grid orientation, with no implicit rotation.
    for axis in (0, 1):
        for cut in range(1, a.shape[axis] - 1):
            line = a[cut] if axis == 0 else a[:, cut]
            if not np.all(line == line[0]):
                continue
            frame = int(line[0])
            before = a[:cut] if axis == 0 else a[:, :cut]
            after = a[cut + 1:] if axis == 0 else a[:, cut + 1:]
            for panel, canvas, side in ((before, after, "top" if axis == 0 else "left"),
                                        (after, before, "bottom" if axis == 0 else "right")):
                if np.any(canvas == frame):
                    continue
                try:
                    info = _read_panel(panel, frame)
                except ValueError:
                    continue
                info["layout"] = side
                candidates.append((canvas.copy(), info))
    if len(candidates) != 1:
        raise ValueError("expected one uniquely decoded canvas and instruction panel")
    return candidates[0]


def _paint(canvas, info, behavior="normal", convention=(1, 0, 0)):
    """Forward stamping with block/canvas intersection, including partial tiles."""
    raw, tile = info["stencil"], info["tile"]
    if behavior == "no_rows":
        mask = raw[:, _starts(raw, 1)]
    elif behavior == "no_columns":
        mask = raw[_starts(raw, 0), :]
    elif behavior == "no_compression":
        mask = raw.copy()
    elif behavior == "global_unique":
        mask = _global_unique(raw)
    else:
        mask = _compress(raw)
    if behavior == "compress_tile":
        tile = _compress(tile)
    sign, offset, transform = convention
    turn = (sign * info["turn"] + offset) % 4
    if behavior == "no_rotation":
        turn = 0
    if behavior == "opposite_rotation":
        turn = -turn % 4
    mask = np.rot90(mask, -turn)
    if behavior == "rotate_tile":
        tile = np.rot90(tile, -turn)
    colors = np.unique(tile)
    swapped = tile if behavior == "no_exchange" else colors[0] + colors[1] - tile
    ar, ac = _CORNERS[info["anchor"]]
    if transform >= 4:
        ac = 1 - ac
    for _ in range(transform % 4):
        ar, ac = ac, 1 - ar
    if behavior == "top_left":
        ar, ac = 0, 0
    th, tw = tile.shape
    mh, mw = mask.shape[0] * th, mask.shape[1] * tw
    h, w = canvas.shape
    top, left = ar * (h - mh), ac * (w - mw)
    if behavior == "fit_instead_of_clip":
        top, left = max(0, top), max(0, left)
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    if behavior == "fill_holes":
        mask = np.ones(mask.shape, bool)
    for r, c in np.argwhere(mask):
        y, x = top + int(r) * th, left + int(c) * tw
        y0, x0, y1, x1 = max(0, y), max(0, x), min(h, y + th), min(w, x + tw)
        if y0 >= y1 or x0 >= x1:
            continue
        values = swapped[y0 - y:y1 - y, x0 - x:x1 - x]
        active = np.ones(values.shape, bool)
        if behavior in ("only_low", "only_high"):
            active = values == colors[0 if behavior == "only_low" else 1]
        block = out[y0:y1, x0:x1]
        block[active] = values[active]
        covered[y0:y1, x0:x1] |= active
    return out, covered


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the input; neither generation nor examples are consulted."""
    canvas, info = _decode(input_grid)
    return _paint(canvas, info)[0].astype(int)


def _inverse_expected(canvas, info):
    """Scalar inverse-coordinate check, separate from rotation/block painting."""
    raw, tile = info["stencil"], info["tile"]
    rows = [0] + [i for i in range(1, len(raw)) if np.any(raw[i] != raw[i - 1])]
    cols = [0] + [i for i in range(1, raw.shape[1]) if np.any(raw[:, i] != raw[:, i - 1])]
    s = raw[np.ix_(rows, cols)]
    sh, sw = s.shape
    k, n = info["turn"], len(tile)
    rh, rw = (sw, sh) if k % 2 else (sh, sw)
    ar, ac = _CORNERS[info["anchor"]]
    top, left = ar * (len(canvas) - n * rh), ac * (canvas.shape[1] - n * rw)
    colors = list(map(int, np.unique(tile)))
    out, covered = canvas.copy(), np.zeros(canvas.shape, bool)
    for r in range(len(canvas)):
        for c in range(canvas.shape[1]):
            y, x = r - top, c - left
            if not (0 <= y < n * rh and 0 <= x < n * rw):
                continue
            u, v = y // n, x // n
            if k == 0:
                sy, sx = u, v
            elif k == 1:
                sy, sx = sh - 1 - v, u
            elif k == 2:
                sy, sx = sh - 1 - u, sw - 1 - v
            else:
                sy, sx = v, sw - 1 - u
            if s[sy, sx]:
                value = int(tile[y % n, x % n])
                out[r, c] = colors[1] if value == colors[0] else colors[0]
                covered[r, c] = True
    return out, covered


def _mask(rng, h, w, witness):
    for _ in range(128):
        mode = int(rng.integers(6))
        rr, cc = np.indices((h, w))
        if mode == 0:
            a = (rr + cc) % 2 == 0
        elif mode == 1:
            a = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
        elif mode == 2:
            a = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
            a[int(rng.integers(h)), int(rng.integers(w))] = False
        else:
            a = rng.random((h, w)) < rng.uniform(.2, .8)
        if witness == 0:
            a[2] = a[0]
            a[:, 2] = a[:, 0]
        if not a.any() or a.all() or _compress(a).shape != a.shape:
            continue
        if witness is not None and (np.array_equal(a, np.rot90(a, 2))
                                    or (a.shape[0] == a.shape[1] and np.array_equal(a, np.rot90(a)))):
            continue
        if witness is not None and not all(edge.any() for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
            continue
        return a
    # An alternating grid with a changed corner always has observable boundaries.
    a = (np.indices((h, w)).sum(0) % 2 == 0)
    if h >= 3 and w >= 3:
        a[-1, -1] = not a[-1, -1]
    if witness == 0:
        a = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [1, 0, 1, 1], [1, 1, 1, 0]], bool)
    return a


def _runs(rng, length, limit, mode):
    values = np.ones(length, dtype=int)
    if mode == "none":
        return values
    if mode == "test_long":
        index = int(rng.integers(length))
        values[index] = int(rng.integers(4, min(7, limit - length + 1) + 1))
    elif mode == "witness":
        values[int(rng.integers(length))] = 2
    else:
        max_value = min(3, limit // length)
        if max_value >= 2 and rng.random() < .25:
            values[:] = int(rng.integers(2, max_value + 1))
        else:
            for i in rng.permutation(length):
                extra = min(2, limit - int(values.sum()))
                values[i] += int(rng.integers(extra + 1))
    spare = limit - int(values.sum())
    for i in rng.permutation(length):
        extra = min(spare, 3 - values[i])
        if extra > 0:
            delta = int(rng.integers(extra + 1))
            values[i] += delta
            spare -= delta
    if mode == "witness" and values.min() > 1:
        values[int(rng.integers(length))] = 1
    return values


def _tile(rng, n, colors, witness):
    for _ in range(64):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if not bits.any() or bits.all():
            continue
        if witness is not None and np.array_equal(bits, np.rot90(bits, 2)):
            continue
        if witness == 0 and _compress(bits).shape == bits.shape:
            continue
        return np.where(bits, colors[0], colors[1]).astype(np.int8)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(np.int8)


def _canvas(rng, h, w, colors):
    rr, cc = np.indices((h, w))
    mode = int(rng.integers(6))
    if mode == 0:
        ix = np.zeros((h, w), int)
    elif mode == 1:
        ix = rr // int(rng.integers(2, 6))
    elif mode == 2:
        ix = cc // int(rng.integers(2, 6))
    elif mode == 3:
        ix = rr // int(rng.integers(2, 6)) + cc // int(rng.integers(2, 6))
    elif mode == 4:
        ix = np.minimum.reduce((rr, h - 1 - rr, cc, w - 1 - cc))
    else:
        ix = np.zeros((h, w), int)
        for _ in range(4):
            r0, r1 = sorted(map(int, rng.integers(h + 1, size=2)))
            c0, c1 = sorted(map(int, rng.integers(w + 1, size=2)))
            ix[r0:r1, c0:c1] = int(rng.integers(len(colors)))
    return np.asarray(colors, dtype=np.int8)[ix % len(colors)]


def _framed(panel, r, c, inside, frame):
    h, w = inside.shape
    panel[r:r + h + 2, c:c + w + 2] = frame
    panel[r + 1:r + h + 1, c + 1:c + w + 1] = inside


def _make_pair(rng, split, index, turn, anchor, layout, seen):
    witness = index if split == "train" and index < 2 else None
    horizontal = layout in ("top", "bottom")
    for _ in range(160):
        n = int(rng.integers(2, 4 if witness is not None else 5))
        p = int(rng.integers(13, 16))
        if witness == 0:
            h, w = 4, 4
        elif witness == 1:
            h, w = int(rng.integers(3, 6)), int(rng.integers(3, 5))
        else:
            h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        latent = _mask(rng, h, w, witness)
        h, w = latent.shape
        # The panel's long dimension is <=30; all four boxes have a blank margin.
        max_raw_h, max_raw_w = 17 - n, p - 4
        if split == "test":
            long_axis = int(rng.integers(2))
            if (max_raw_h - h if long_axis == 0 else max_raw_w - w) < 3:
                continue
            row_mode = "test_long" if long_axis == 0 else str(rng.choice(["normal", "none"]))
            col_mode = "test_long" if long_axis == 1 else str(rng.choice(["normal", "none"]))
        elif witness is not None:
            row_mode = col_mode = "witness"
        else:
            row_mode, col_mode = rng.choice(["normal", "none"], 2, p=[.8, .2])
        rows, cols = _runs(rng, h, max_raw_h, row_mode), _runs(rng, w, max_raw_w, col_mode)
        raw = np.repeat(np.repeat(latent, rows, axis=0), cols, axis=1)
        # Stencils that impersonate either 3x3 key are excluded geometrically.
        if raw.shape == (3, 3) and raw.sum() <= 2:
            continue
        min_h = n + raw.shape[0] + 13
        if min_h > 30:
            continue
        full_h = int(rng.integers(min_h, 31))
        k_local = (3 - turn) % 4 if horizontal else turn
        a_local = -anchor % 4 if horizontal else anchor
        rotated = np.rot90(latent.T if horizontal else latent, -turn)
        # In the base packing orientation, the canvas has full_h rows.
        mh, mw = (rotated.shape[1] * n, rotated.shape[0] * n) if horizontal else tuple(np.array(rotated.shape) * n)
        if witness == 0:
            if mw + 1 > 29 - p or mh + 1 > full_h:
                continue
            cw = int(rng.integers(mw + 1, 30 - p))
        elif witness == 1:
            if mh + 1 > full_h:
                continue
            cw = mw - int(rng.integers(1, n))
            if not 5 <= cw <= 29 - p:
                continue
        else:
            cw = int(rng.integers(5, 30 - p))
        palette = rng.permutation(10)
        ground, frame, ta, tb = map(int, palette[:4])
        # Local role sharing is allowed; only the frame is structurally exclusive.
        ink = int(rng.choice(palette[2:int(rng.integers(5, 9))]))
        marker_a, marker_d = map(int, rng.choice(palette[2:7], 2))
        tile = _tile(rng, n, (ta, tb), witness)
        corner, arrow = np.full((3, 3), ground, np.int8), np.full((3, 3), ground, np.int8)
        ar, ac = _CORNERS[a_local]
        corner[2 * ar, 2 * ac] = marker_a
        arrow[1, 1] = marker_d
        arrow[_ARROWS[k_local]] = marker_d
        panel = np.full((full_h, p), ground, np.int8)
        spare = full_h - min_h
        gaps = rng.multinomial(spare, [.25] * 4)
        tr = 1 + int(gaps[0])
        sr = tr + n + 3 + int(gaps[1])
        kr = sr + raw.shape[0] + 3 + int(gaps[2])
        _framed(panel, tr, int(rng.integers(1, p - n - 2)), tile, frame)
        _framed(panel, sr, int(rng.integers(1, p - raw.shape[1] - 2)), np.where(raw, ink, ground), frame)
        keys = [corner, arrow]
        if rng.random() < .5:
            keys.reverse()
        kc = int(rng.integers(1, p - 11)) if p > 12 else 1
        _framed(panel, kr, kc, keys[0], frame)
        _framed(panel, kr, kc + 6, keys[1], frame)
        if witness is not None:
            canvas_colors = [ground, int(palette[4])]
        else:
            pool = [int(v) for v in palette if v != frame]
            canvas_colors = rng.choice(pool, int(rng.integers(1, 5)), replace=False)
        canvas = _canvas(rng, full_h, cw, canvas_colors)
        if horizontal:
            panel, canvas = panel.T.copy(), canvas.T.copy()
        info = dict(tile=tile.T.copy() if horizontal else tile,
                    stencil=raw.T.copy() if horizontal else raw,
                    turn=turn, anchor=anchor)
        axis = 0 if horizontal else 1
        divider_shape = (1, canvas.shape[1]) if horizontal else (canvas.shape[0], 1)
        divider = np.full(divider_shape, frame, np.int8)
        parts = (panel, divider, canvas) if layout in ("top", "left") else (canvas, divider, panel)
        inp = np.concatenate(parts, axis=axis)
        out, covered = _paint(canvas, info)
        if _key(inp) in seen or not covered.any() or np.array_equal(out, canvas):
            continue
        if witness is not None:
            needed = ["no_rows", "no_columns", "no_compression", "no_exchange",
                      "top_left", "fill_holes", "only_low", "only_high"]
            if turn:
                needed.extend(("no_rotation", "rotate_tile"))
            if turn % 2:
                needed.append("opposite_rotation")
            if witness == 0:
                needed.extend(("global_unique", "compress_tile"))
            if witness == 1:
                needed.append("fit_instead_of_clip")
            if any(np.array_equal(_paint(canvas, info, b)[0], out) for b in needed
                   if not (b == "top_left" and anchor == 0)):
                continue
        return {"input": inp.astype(int), "output": out.astype(int)}
    raise RuntimeError("bounded pair construction exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    q, e = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    b = int(rng.integers(1, 4))
    a = (b + int(rng.choice([1, 3]))) % 4
    # Test uses an unseen combination of already shown geometric controls.
    held = (q, b)
    layouts = ["left", "right", "top", "bottom"]
    rng.shuffle(layouts)
    # The second demonstration clips at an anchored far edge. Its full mosaic
    # origin is negative, making the tile phase visible rather than inferred.
    allowed = [i for i in range(1, 4)
               if b == 2 or (b == 1 and layouts[i] in ("left", "right"))
               or (b == 3 and layouts[i] in ("top", "bottom"))]
    j = int(rng.choice(allowed))
    layouts[1], layouts[j] = layouts[j], layouts[1]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i < 2:
                k, anchor = (q, a) if i == 0 else (e, b)
            elif split == "test" and i == 0:
                k, anchor = held
            else:
                k, anchor = int(rng.integers(4)), int(rng.integers(4))
                if split == "train" and (k, anchor) == held:
                    k = (k + 1) % 4
            layout = layouts[i % 4] if split == "train" else str(rng.choice(layouts))
            pair = _make_pair(rng, split, i, k, anchor, layout, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(decoded):
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                c = (sign, offset, transform)
                if all(np.array_equal(_paint(canvas, ins, convention=c)[0], out)
                       for canvas, ins, out in decoded):
                    survivors.append(c)
    _require(survivors == [(1, 0, 0)], "training leaves key geometry ambiguous")
    for behavior in _ALTERNATIVES:
        _require(any(not np.array_equal(_paint(c, ins, behavior)[0], out)
                     for c, ins, out in decoded), f"training permits {behavior}")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, decoded_train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            canvas, info = _decode(inp)
            expected, covered = _inverse_expected(canvas, info)
            _require(np.array_equal(expected, out), "incorrect output")
            _require(covered.any() and np.any(out != canvas), "no visible stamping")
            _require(np.array_equal(out[~covered], canvas[~covered]), "transparent cells changed")
            raw, small = info["stencil"], _compress(info["stencil"])
            row_starts, col_starts = _starts(raw, 0), _starts(raw, 1)
            rows = np.diff(row_starts + [raw.shape[0]])
            cols = np.diff(col_starts + [raw.shape[1]])
            rebuilt = np.repeat(np.repeat(small, rows, axis=0), cols, axis=1)
            _require(np.array_equal(rebuilt, raw), "stencil bands lost information")
            _require(_compress(small).shape == small.shape, "compression not idempotent")
            runs = tuple(map(tuple, (rows, cols)))
            if split == "train":
                _require(max(max(r) for r in runs) <= 3, "training band exceeds curriculum")
                decoded_train.append((canvas, info, out))
            else:
                _require(max(max(r) for r in runs) >= 4, "missing test stretch extrapolation")
    first = decoded_train[0][1]
    raw, small = first["stencil"], _compress(first["stencil"])
    _require(raw.shape[0] > small.shape[0] and raw.shape[1] > small.shape[1], "missing both-axis witness")
    _require(_repeated(small, 0) and _repeated(small, 1), "missing separated repeats")
    for axis in (0, 1):
        runs = tuple(np.diff(_starts(raw, axis) + [raw.shape[axis]]))
        _require(len(set(runs)) > 1, "first witness lacks nonuniform band widths")
    train_controls = {(ins["turn"], ins["anchor"]) for _, ins, _ in decoded_train}
    _, test_info = _decode(puzzle["test"][0]["input"])
    _require((test_info["turn"], test_info["anchor"]) not in train_controls, "missing held-out control combination")
    for canvas, ins, out in decoded_train[:2]:
        mask = np.rot90(_compress(ins["stencil"]), -ins["turn"])
        extent = np.asarray(mask.shape) * len(ins["tile"])
        if ins is first:
            _require(np.all(extent < canvas.shape), "first witness needs visible margins")
        else:
            _require(np.any(extent > canvas.shape), "second witness needs clipping")
    _training_evidence(decoded_train[:2])


def self_test():
    """42 seeds, all 30 legal count combinations, no files or bulk retention."""
    shapes, structures, layouts, sizes, controls, schedules = set(), set(), set(), set(), set(), set()
    clipped, holes, repeated, one_axis = 0, 0, 0, 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(inp.astype(np.uint8).tobytes())
                digest.update(out.astype(np.uint8).tobytes())
                canvas, info = _decode(inp)
                small, n = _compress(info["stencil"]), len(info["tile"])
                mask = np.rot90(small, -info["turn"])
                shapes.add((inp.shape, out.shape, small.shape))
                structures.add(_key(small))
                layouts.add(info["layout"])
                sizes.add(n)
                controls.add((info["turn"], info["anchor"]))
                raw = info["stencil"]
                runs = tuple(tuple(np.diff(_starts(raw, axis) + [raw.shape[axis]])) for axis in (0, 1))
                schedules.add(runs)
                clipped += bool(any(x * n > y for x, y in zip(mask.shape, canvas.shape)))
                holes += bool(not mask.all())
                repeated += bool(_repeated(small, 0) or _repeated(small, 1))
                one_axis += (max(runs[0]) == 1) != (max(runs[1]) == 1)
                _require(np.array_equal(solve(inp), out), "solve mismatch")
                if split == "train":
                    _require(max(max(r) for r in runs) <= 3, "training stretch cap violated")
                else:
                    _require(max(max(r) for r in runs) >= 4, "missing test stretch shift")
        fingerprints.add(digest.digest())
    _require(len(fingerprints) == 42 and len(shapes) >= 100 and len(structures) >= 65, "insufficient geometry variation")
    _require(layouts == {"left", "right", "top", "bottom"} and sizes == {2, 3, 4}, "missing layout/size variation")
    _require(len(controls) == 16 and len(schedules) >= 100, "missing controls/bands")
    _require(clipped >= 42 and holes == 210 and repeated >= 42 and one_axis > 0, "missing structural properties")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1042 + nt * 10 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(2026), generate_puzzle(2026)
    _require(all(np.array_equal(x[r], y[r]) for s in p for x, y in zip(p[s], q[s]) for r in x), "nondeterminism")
    damaged = {s: [{r: a.copy() for r, a in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": 210, "shape_combinations": len(shapes),
            "stencil_structures": len(structures), "band_schedules": len(schedules),
            "clipped_pairs": int(clipped), "one_axis_only_pairs": int(one_axis),
            "legal_count_combinations": 30}
