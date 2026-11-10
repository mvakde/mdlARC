"""Exchange/rotate/stamp, then contract full adjacent vectors on the canvas.

One visibly bounded instruction strip can border any side of the canvas.
Generation and validation retain no seed-dependent decoding information.
No files, execution traces, or example caches are written by this module.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
DIRECTIONS = ((0, 1), (1, 2), (2, 1), (1, 0))
HOST_SEEDS = (0, 896573341, 489965937, 779133176, 119475206, 112911742,
              457519451, 793730589, 326084828, 865918980, 284520828,
              573928719, 211032781, 826430576, 206223320, 684431279,
              940413189, 701512826, 489108456, 794394897, 964884238,
              388720683, 718931536, 443600347, 39813147, 279959823,
              839892468, 976726511, 162320278, 838751629, 410336052,
              587840838, 575115378, 61172602, 521625340, 547471183,
              808486788, 705561178, 30388492, 684221510, 362950707, 771043053)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _starts(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _compress(a):
    # Both partitions are measured on the same completed canvas.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _loop_compress(a):
    """A second implementation: columns first, then rows, copying literal cells."""
    cols = [0] + [c for c in range(1, a.shape[1])
                  if not np.array_equal(a[:, c], a[:, c - 1])]
    b = a[:, cols]
    rows = [0] + [r for r in range(1, len(b))
                  if not np.array_equal(b[r], b[r - 1])]
    return b[rows].copy()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        stack = [todo.pop()]
        points = []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(points)
    return result


def _decode_strip(panel, frame, axis):
    ground = int(panel[0, 0])
    if ground == frame:
        raise ValueError("missing panel margin")
    boxes = []
    allowed = np.zeros(panel.shape, bool)
    for points in _components(panel == frame):
        coords = np.asarray(points)
        lo, hi = coords.min(0), coords.max(0) + 1
        r, c = map(int, lo)
        h, w = map(int, hi - lo)
        if min(h, w) < 4:
            raise ValueError("invalid frame")
        outer = panel[r:r + h, c:c + w]
        if not (np.all(outer[0] == frame) and np.all(outer[-1] == frame)
                and np.all(outer[:, 0] == frame) and np.all(outer[:, -1] == frame)
                and not np.any(outer[1:-1, 1:-1] == frame)):
            raise ValueError("frame must enclose one complete rectangle")
        if r == 0 or c == 0 or r + h >= len(panel) or c + w >= panel.shape[1]:
            raise ValueError("frames require visible ground margins")
        boxes.append((r, c, h, w, outer[1:-1, 1:-1].copy()))
        allowed[r:r + h, c:c + w] = True
    boxes.sort(key=lambda b: b[axis])
    if len(boxes) != 4 or np.any(panel[~allowed] != ground):
        raise ValueError("expected exactly four separated framed fields")
    for a, b in zip(boxes, boxes[1:]):
        if a[axis] + a[axis + 2] >= b[axis]:
            raise ValueError("fields need a ground gap along the strip")
    tile, stencil, corner, direction = [b[4] for b in boxes]
    if tile.shape[0] != tile.shape[1] or not 3 <= len(tile) <= 5:
        raise ValueError("tile must be a complete 3..5 square")
    if len(np.unique(tile)) != 2 or ground in tile:
        raise ValueError("tile must have two non-ground colors")
    ink = stencil[stencil != ground]
    mask = stencil != ground
    if (len(np.unique(ink)) != 1 or min(stencil.shape) < 2
            or max(stencil.shape) > 5 or not np.all(np.any(mask, axis=0)[[0, -1]])
            or not np.all(np.any(mask, axis=1)[[0, -1]])):
        raise ValueError("stencil needs tight visible monochrome support")
    if corner.shape != (3, 3) or direction.shape != (3, 3):
        raise ValueError("controls must be bounded 3-by-3 fields")
    cp = set(map(tuple, np.argwhere(corner != ground)))
    dp = set(map(tuple, np.argwhere(direction != ground)))
    anchors = [i for i, (r, c) in enumerate(CORNERS) if cp == {(2 * r, 2 * c)}]
    turns = [i for i, p in enumerate(DIRECTIONS) if dp == {(1, 1), p}]
    if (len(anchors) != 1 or len(turns) != 1
            or len(np.unique(direction[direction != ground])) != 1):
        raise ValueError("invalid geometric controls")
    return dict(tile=tile, mask=mask, anchor=anchors[0], turn=turns[0],
                ground=ground, frame=frame,
                origins=[(b[0], b[1]) for b in boxes])


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        # axis=0: a horizontal separator, with a horizontally read strip.
        for cut in range(5, a.shape[axis] - 5):
            line = a[cut, :] if axis == 0 else a[:, cut]
            if not np.all(line == line[0]):
                continue
            frame = int(line[0])
            for before in (True, False):
                if axis == 0:
                    panel = a[:cut] if before else a[cut + 1:]
                    canvas = a[cut + 1:] if before else a[:cut]
                else:
                    panel = a[:, :cut] if before else a[:, cut + 1:]
                    canvas = a[:, cut + 1:] if before else a[:, :cut]
                if frame in canvas:
                    continue
                try:
                    info = _decode_strip(panel, frame, 1 - axis)
                except (ValueError, IndexError):
                    continue
                side = ("top" if before else "bottom") if axis == 0 else ("left" if before else "right")
                info["side"] = side
                candidates.append((canvas.copy(), info))
    if len(candidates) != 1:
        raise ValueError(f"expected one unambiguous canvas/strip split, found {len(candidates)}")
    return candidates[0]


def _anchor_transform(anchor, transform):
    r, c = CORNERS[anchor]
    if transform >= 4:
        c = 1 - c
    for _ in range(transform % 4):
        r, c = c, 1 - r
    return r, c


def _render(canvas, info, behavior="normal", convention=(1, 0, 0)):
    """Input-only inverse-coordinate painter; does not use forward construction."""
    tile, mask = info["tile"], info["mask"]
    n, (sh, sw) = len(tile), mask.shape
    sign, offset, transform = convention
    k = (sign * info["turn"] + offset) % 4
    if behavior == "no_rotation":
        k = 0
    rh, rw = (sw, sh) if k % 2 else (sh, sw)
    mh, mw = n * rh, n * rw
    if mh > len(canvas) or mw > canvas.shape[1]:
        raise ValueError("mosaic does not fit the canvas")
    ar, ac = _anchor_transform(info["anchor"], transform)
    yy, xx = np.indices(canvas.shape)
    y, x = yy - ar * (len(canvas) - mh), xx - ac * (canvas.shape[1] - mw)
    inside = (y >= 0) & (x >= 0) & (y < mh) & (x < mw)
    mr, mc = np.clip(y // n, 0, rh - 1), np.clip(x // n, 0, rw - 1)
    if k == 0:
        sy, sx = mr, mc
    elif k == 1:
        sy, sx = sh - 1 - mc, mr
    elif k == 2:
        sy, sx = sh - 1 - mr, sw - 1 - mc
    else:
        sy, sx = mc, sw - 1 - mr
    active = inside & (True if behavior == "fill_holes" else mask[sy, sx])
    ty, tx = y % n, x % n
    if behavior == "rotate_tile":
        for _ in range(k):
            ty, tx = n - 1 - tx, ty
    colors = np.unique(tile)
    original = tile[ty, tx]
    values = original if behavior == "no_exchange" else colors.sum() - original
    if behavior == "only_first":
        active &= values == colors[0]
    if behavior == "only_second":
        active &= values == colors[1]
    result = canvas.copy()
    result[active] = values[active]
    return result, active


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible fields, stamp at native resolution, then collapse vectors."""
    canvas, info = _decode(input_grid)
    return _compress(_render(canvas, info)[0])


def _forward_stamp(canvas, tile, mask, turn, anchor):
    result = canvas.copy()
    covered = np.zeros(canvas.shape, bool)
    rotated = np.rot90(mask, -turn)
    n = len(tile)
    h, w = np.array(rotated.shape) * n
    ar, ac = CORNERS[anchor]
    top, left = ar * (len(canvas) - h), ac * (canvas.shape[1] - w)
    swapped = np.unique(tile).sum() - tile
    for r, c in np.argwhere(rotated):
        box = np.s_[top + r * n:top + (r + 1) * n, left + c * n:left + (c + 1) * n]
        result[box] = swapped
        covered[box] = True
    return result, covered


def _composition(rng, n, count):
    return np.ones(count, int) + rng.multinomial(n - count, np.full(count, 1 / count))


def _tile(rng, n, colors):
    for _ in range(40):
        bh = int(rng.integers(2, min(3, n - 1) + 1))
        bw = int(rng.integers(2, min(3, n - 1) + 1))
        bits = rng.integers(0, 2, (bh, bw))
        if (len(np.unique(bits)) < 2 or len(_starts(bits, 0)) != bh
                or len(_starts(bits, 1)) != bw):
            continue
        expanded = np.repeat(np.repeat(bits, _composition(rng, n, bh), axis=0),
                             _composition(rng, n, bw), axis=1)
        if np.array_equal(expanded, np.rot90(expanded, 2)):
            continue
        return np.asarray(colors)[expanded]
    bits = np.array([[1, 0], [0, 0]])
    expanded = np.repeat(np.repeat(bits, [n - 1, 1], axis=0), [1, n - 1], axis=1)
    return np.asarray(colors)[expanded]


def _stencil(rng, n, witness, irregular):
    if witness:
        core = np.ones((2, 2), bool)
        core.flat[int(rng.integers(4))] = False
        rows, cols = [0, 1, 0], [0, 1, 0, 1]
        if rng.random() < .5:
            rows, cols = cols, rows
        return core[np.ix_(rows, cols)]
    maximum = 3 if n == 5 else 5
    h, w = map(int, rng.integers(2, maximum + 1, size=2))
    for _ in range(24):
        style = int(rng.integers(5))
        rr, cc = np.indices((h, w))
        if style == 0:
            mask = rng.random((h, w)) < rng.uniform(.25, .8)
        elif style == 1:
            mask = (rr % 2 == 0) & (cc % 2 == 0)
        elif style == 2:
            mask = (rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)
        elif style == 3:
            mask = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
        else:
            core = rng.random((2, 2)) < .7
            mask = core[np.ix_(np.arange(h) % 2, np.arange(w) % 2)]
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if min(mask.shape) >= 2 and not mask.all():
            if irregular and not np.any(~mask[1:-1, 1:-1]):
                continue
            return mask
    return np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], bool)


def _frame(a, color):
    result = np.full(np.array(a.shape) + 2, color, int)
    result[1:-1, 1:-1] = a
    return result


def _pack(rng, fields, ground, frame, side, canvas_shape):
    axis = 0 if side in ("left", "right") else 1
    length = canvas_shape[axis]
    minimum_short = max(a.shape[1 - axis] + 2 for a in fields) + 2
    # The geometry chooser already allocated this strip thickness.
    thickness = canvas_shape[2]
    assert thickness >= minimum_short
    shape = (length, thickness) if axis == 0 else (thickness, length)
    panel = np.full(shape, ground, int)
    frames = [_frame(a, frame) for a in fields]
    spare = length - sum(a.shape[axis] for a in frames) - 5
    gaps = 1 + rng.multinomial(spare, np.full(5, .2))
    pos = int(gaps[0])
    for j, box in enumerate(frames):
        off = int(rng.integers(1, thickness - box.shape[1 - axis]))
        r, c = (pos, off) if axis == 0 else (off, pos)
        panel[r:r + box.shape[0], c:c + box.shape[1]] = box
        pos += box.shape[axis] + int(gaps[j + 1])
    return panel


def _geometry(rng, n, mask, turn, side):
    vertical = side in ("left", "right")
    axis = 0 if vertical else 1
    rotated_shape = np.array(mask.shape if turn % 2 == 0 else mask.shape[::-1]) * n
    minimum_long = n + mask.shape[axis] + 19
    minimum_short = max(n, mask.shape[1 - axis], 3) + 4
    long_low = max(minimum_long, int(rotated_shape[axis]) + 1)
    thickness_high = min(minimum_short + 2, 28 - int(rotated_shape[1 - axis]))
    if long_low > 30 or thickness_high < minimum_short:
        raise ValueError("layout does not fit")
    length = int(rng.integers(long_low, 31))
    thickness = int(rng.integers(minimum_short, thickness_high + 1))
    short_low = max(8, int(rotated_shape[1 - axis]) + 1)
    short = int(rng.integers(short_low, 30 - thickness))
    h, w = (length, short) if vertical else (short, length)
    return h, w, thickness


def _phase_labels(tile, axis, length, origin):
    vectors = tile if axis == 0 else tile.T
    ids = np.r_[0, np.cumsum(np.any(vectors[1:] != vectors[:-1], axis=1))]
    phase = (np.arange(length) - origin) % len(tile)
    return ids[phase]


def _substrate(rng, shape, tile, mask, turn, anchor, colors, style):
    h, w = shape
    rotated = np.rot90(mask, -turn)
    mh, mw = np.array(rotated.shape) * len(tile)
    ar, ac = CORNERS[anchor]
    top, left = ar * (h - mh), ac * (w - mw)
    ry = _phase_labels(tile, 0, h, top)
    cx = _phase_labels(tile, 1, w, left)
    if style == 0:
        canvas = np.full(shape, int(colors[0]), int)
    elif style == 1:  # Periodic weave: witnesses retain literal crossing colors.
        base = rng.choice(colors, size=(int(ry.max()) + 1, int(cx.max()) + 1))
        base[0, 0] = colors[0]
        base[-1, -1] = colors[-1]
        canvas = base[np.ix_(ry, cx)]
    else:
        # Unique macro labels permit unequal broad bands and irregular patches,
        # while the repeated tile phases still guarantee exact duplicate vectors.
        ry = ry + ((np.arange(h) - top) // len(tile) + 20) * (int(ry.max()) + 1)
        cx = cx + ((np.arange(w) - left) // len(tile) + 20) * (int(cx.max()) + 1)
        _, ry = np.unique(ry, return_inverse=True)
        _, cx = np.unique(cx, return_inverse=True)
        nr, nc = int(ry.max()) + 1, int(cx.max()) + 1
        rr, cc = np.indices((nr, nc))
        if style == 2:
            base = np.asarray(colors)[(rr // int(rng.integers(1, 4))) % len(colors)]
        elif style == 3:
            base = np.asarray(colors)[(cc // int(rng.integers(1, 4))) % len(colors)]
        elif style == 4:
            base = np.asarray(colors)[(rr + cc // 2) % len(colors)]
        else:
            base = np.full((nr, nc), int(colors[0]), int)
            for _ in range(int(rng.integers(2, 7))):
                r, c = int(rng.integers(nr)), int(rng.integers(nc))
                rh, cw = int(rng.integers(1, nr + 1)), int(rng.integers(1, nc + 1))
                base[r:r + rh, c:c + cw] = int(rng.choice(colors))
        canvas = base[np.ix_(ry, cx)]
    # A small pre-existing mark inside a block is overwritten. It makes newly
    # created vector equalities visible; no mark is interpreted as an instruction.
    dr = int(np.flatnonzero(np.all(tile[1:] == tile[:-1], axis=1))[0])
    dc = int(np.flatnonzero(np.all(tile[:, 1:] == tile[:, :-1], axis=0))[0])
    mr, mc = map(int, np.argwhere(rotated)[int(rng.integers(rotated.sum()))])
    r, c = top + mr * len(tile) + dr, left + mc * len(tile) + dc
    final_color = int(np.unique(tile).sum() - tile[dr, dc])
    options = [v for v in range(10) if v not in (int(canvas[r, c]), final_color)]
    canvas[r, c] = int(rng.choice(options))
    return canvas


def _unique_vectors(a):
    rows, cols = [], []
    for r in range(len(a)):
        if not any(np.array_equal(a[r], a[p]) for p in rows):
            rows.append(r)
    for c in range(a.shape[1]):
        if not any(np.array_equal(a[:, c], a[:, p]) for p in cols):
            cols.append(c)
    return a[np.ix_(rows, cols)]


def _different(a, b):
    return a.shape != b.shape or not np.array_equal(a, b)


def _witness_failures(canvas, info, completed, output):
    failures = []
    for behavior in ("no_exchange", "rotate_tile", "fill_holes", "only_first", "only_second"):
        if behavior == "rotate_tile" and info["turn"] == 0:
            continue  # The informative quarter-turn is in the first witness.
        alt = _compress(_render(canvas, info, behavior)[0])
        if not _different(alt, output):
            failures.append(behavior)
    if info["turn"] != 0:
        try:
            alt = _compress(_render(canvas, info, "no_rotation")[0])
        except ValueError:
            pass  # The unrotated rectangle can have incompatible dimensions.
        else:
            if not _different(alt, output):
                failures.append("no_rotation")
    alternatives = {"no_A": _compress(canvas), "no_B": completed,
                    "rows_only": completed[_starts(completed, 0)],
                    "columns_only": completed[:, _starts(completed, 1)],
                    "global_unique": _unique_vectors(completed),
                    "input_partitions": completed[np.ix_(_starts(canvas, 0), _starts(canvas, 1))]}
    try:
        alternatives["compress_before_stamp"] = _render(_compress(canvas), info)[0]
    except ValueError:
        pass  # Native-size blocks no longer fit; early compression is invalid.
    for name, alt in alternatives.items():
        if not _different(alt, output):
            failures.append(name)
    # Equality partitions must actually change because of stamping.
    for axis in (0, 1):
        before, after = (canvas, completed) if axis == 0 else (canvas.T, completed.T)
        eq0 = np.all(before[1:] == before[:-1], axis=1)
        eq1 = np.all(after[1:] == after[:-1], axis=1)
        if not np.any(~eq0 & eq1):
            failures.append(f"no_created_equality_{axis}")
        if not np.any(eq0 & ~eq1):
            failures.append(f"no_broken_equality_{axis}")
    return failures


def _make_pair(rng, n, control, witness, irregular, side):
    turn, anchor = control
    for _ in range(120):
        palette = rng.permutation(10)
        ground, frame = map(int, palette[:2])
        tile = _tile(rng, n, palette[2:4])
        mask = _stencil(rng, n, witness, irregular)
        try:
            h, w, thickness = _geometry(rng, n, mask, turn, side)
        except ValueError:
            continue
        color_count = int(rng.integers(2, 5))
        substrate_colors = palette[4:4 + color_count]
        if not witness and rng.random() < .4:
            substrate_colors = np.r_[substrate_colors[:2], palette[2:4]]
        style = (0 if witness == 1 else 1) if witness else int(rng.integers(6))
        canvas = _substrate(rng, (h, w), tile, mask, turn, anchor, substrate_colors, style)
        # The divider color is structural and cannot occur in the canvas.
        canvas[canvas == frame] = ground
        completed, _ = _forward_stamp(canvas, tile, mask, turn, anchor)
        output = _loop_compress(completed)
        corner = np.full((3, 3), ground, int)
        cr, cc = CORNERS[anchor]
        corner[2 * cr, 2 * cc] = int(rng.choice(palette[2:]))
        direction = np.full((3, 3), ground, int)
        marker = int(rng.choice(palette[2:]))
        direction[1, 1] = marker
        direction[DIRECTIONS[turn]] = marker
        fields = [tile, np.where(mask, int(rng.choice(palette[2:])), ground), corner, direction]
        panel = _pack(rng, fields, ground, frame, side, (h, w, thickness))
        axis = 1 if side in ("left", "right") else 0
        divider = np.full((h, 1) if axis == 1 else (1, w), frame, int)
        parts = [panel, divider, canvas] if side in ("left", "top") else [canvas, divider, panel]
        inp = np.concatenate(parts, axis=axis)
        info = dict(tile=tile, mask=mask, turn=turn, anchor=anchor)
        if witness and _witness_failures(canvas, info, completed, output):
            continue
        if (output.shape[0] >= h or output.shape[1] >= w
                or not _different(_compress(canvas), output)):
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("bounded pair construction exhausted")


def _conventions(pairs):
    decoded = [(*_decode(p["input"]), np.asarray(p["output"])) for p in pairs]
    survivors = []
    for sign in (1, -1):
        for offset in range(4):
            for transform in range(8):
                convention = (sign, offset, transform)
                try:
                    if all(np.array_equal(_compress(_render(c, i, convention=convention)[0]), out)
                           for c, i, out in decoded):
                        survivors.append(convention)
                except ValueError:
                    continue
    return survivors


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    q, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    neighbor = (anchor + int(rng.choice([1, 3]))) % 4
    controls = [(q, anchor), (even, neighbor)]
    # Optional later demonstrations explicitly repeat a direction/anchor.
    additions = [(q, neighbor), (even, anchor)]
    controls += additions[:max(0, min(2, nt - 2))]
    while len(controls) < nt:
        controls.append((int(rng.integers(4)), int(rng.integers(4))))
    unseen = [(k, a) for k in range(4) for a in range(4) if (k, a) not in controls]
    rng.shuffle(unseen)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", nt), ("test", nq)):
        for j in range(count):
            witness = j + 1 if split == "train" and j < 2 else 0
            control = controls[j] if split == "train" else unseen[j]
            n = (3 + j if witness else int(rng.choice([3, 4] if mode == 0 and split == "train" else [3, 4, 5])))
            if split == "test" and mode == 0:
                n = 5
            side = str(rng.choice(["left", "right", "top", "bottom"]))
            for _ in range(12):
                pair = _make_pair(rng, n, control, witness, split == "test" and mode == 1, side)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retry exhausted")
    if _conventions(puzzle["train"][:2]) != [(1, 0, 0)]:
        raise RuntimeError("witnesses failed to disambiguate geometric conventions")
    return puzzle


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    source, output = _grid(pair["input"]), _grid(pair["output"])
    canvas, info = _decode(source)
    completed, footprint = _render(canvas, info)
    forward, forward_footprint = _forward_stamp(canvas, info["tile"], info["mask"], info["turn"], info["anchor"])
    assert np.array_equal(completed, forward)
    assert np.array_equal(footprint, forward_footprint)
    assert np.array_equal(completed[~footprint], canvas[~footprint])
    expected = _compress(completed)
    assert np.array_equal(output, expected) and np.array_equal(output, _loop_compress(completed))
    starts = [_starts(completed, axis) for axis in (0, 1)]
    lengths = [np.diff(np.r_[s, completed.shape[a]]) for a, s in enumerate(starts)]
    assert np.array_equal(np.repeat(np.repeat(output, lengths[0], axis=0), lengths[1], axis=1), completed)
    assert len(_starts(output, 0)) == len(output) and len(_starts(output, 1)) == output.shape[1]
    assert output.shape[0] < canvas.shape[0] and output.shape[1] < canvas.shape[1]
    assert _different(_compress(canvas), output), "A is answer-irrelevant"
    assert len(np.unique(info["tile"])) == 2
    assert np.any(np.all(info["tile"][1:] == info["tile"][:-1], axis=1))
    assert np.any(np.all(info["tile"][:, 1:] == info["tile"][:, :-1], axis=0))
    return canvas, info, completed, output


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {}
    for split in ("train", "test"):
        analyses[split] = []
        for p in puzzle[split]:
            analysis = _analyze(p)
            key = _key(np.asarray(p["input"]))
            assert key not in seen
            seen.add(key)
            analyses[split].append(analysis)
    for canvas, info, completed, output in analyses["train"][:2]:
        assert not _witness_failures(canvas, info, completed, output)
    assert _conventions(puzzle["train"][:2]) == [(1, 0, 0)]
    train_controls = {(i["turn"], i["anchor"]) for _, i, _, _ in analyses["train"]}
    assert all((i["turn"], i["anchor"]) not in train_controls for _, i, _, _ in analyses["test"])
    assert len({(i["turn"], i["anchor"]) for _, i, _, _ in analyses["test"]}) == nq


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                h.update(bytes(a.shape))
                h.update(a.astype(np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """42 specified seeds, streamed; no payloads or traces retained on disk."""
    structures, shapes, outputs, stencils, digests = set(), set(), set(), set(), set()
    sides, sizes, controls, colors = set(), set(), set(), set()
    counts = Counter()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digests.add(_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                canvas, info, completed, out = _analyze(pair)
                assert np.array_equal(solve(pair["input"]), out)
                a = np.asarray(pair["input"])
                roles = {}
                norm = bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)
                structures.add((a.shape, hashlib.sha256(norm).digest()))
                shapes.add(a.shape)
                outputs.add(out.shape)
                stencils.add(_key(info["mask"]))
                sides.add(info["side"])
                sizes.add(len(info["tile"]))
                controls.add((info["turn"], info["anchor"]))
                colors.update(map(int, np.unique(a)))
                counts["pairs"] += 1
                counts["two_axis_compression"] += out.shape[0] < len(canvas) and out.shape[1] < canvas.shape[1]
                counts["separated_row_repeat"] += len(_starts(completed, 0)) > len({row.tobytes() for row in completed})
                counts["separated_column_repeat"] += len(_starts(completed, 1)) > len({col.tobytes() for col in completed.T})
    assert len(digests) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(outputs) >= 25 and len(stencils) >= 40
    assert sides == {"left", "right", "top", "bottom"} and sizes == {3, 4, 5}
    assert len(controls) == 16 and len(colors) == 10
    assert counts["pairs"] == 210 and counts["two_axis_compression"] == 210
    assert min(counts["separated_row_repeat"], counts["separated_column_repeat"]) >= 84
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(0, nt, nq)
            validate_generated_puzzle(p)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # A small exact regression distinguishes adjacency from global deduplication.
    a = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3], [1, 1, 2, 1]])
    assert np.array_equal(_compress(a), np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]]))
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, distinct_puzzles=len(digests), input_shapes=len(shapes),
                output_shapes=len(outputs), normalized_inputs=len(structures),
                stencil_patterns=len(stencils), sides=sorted(sides), tile_sizes=sorted(sizes),
                control_combinations=len(controls), **counts)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
