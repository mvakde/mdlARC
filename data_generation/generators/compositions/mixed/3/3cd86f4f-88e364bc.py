"""Compass-controlled bottom-anchored row shear, then rigid boundary motion.

No files, examples, parent modules or puzzle state are consulted by solve().
All experiments are in memory; this module never writes diagnostic payloads.
"""
from __future__ import annotations

import hashlib
import numpy as np

DIRECTIONS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dy, dx in CARDINAL:
                p = (r + dy, c + dx)
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _glyph(sub):
    """Decode both color-role assignments geometrically, without a color legend."""
    h, w = sub.shape
    if min(h, w) < 4:
        return None
    carrier = int(sub[0, 0])
    border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
    others = set(map(int, np.unique(sub))) - {carrier}
    if len(others) != 2 or np.any(border != carrier):
        return None
    readings = []
    for tip in sorted(others):
        trail = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(sub == tip)))
        trails = set(map(tuple, np.argwhere(sub == trail)))
        if len(tips) < 2:
            continue
        for dy, dx in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                run = []
                p = (r - dy, c - dx)
                while p in trails:
                    run.append(p)
                    p = (p[0] - dy, p[1] - dx)
                if len(run) < 2 or used.intersection(run):
                    break
                used.update(run)
                lengths.append(len(run))
            if len(lengths) == len(tips) and used == trails:
                readings.append(dict(carrier=carrier, trail=trail, tip=tip,
                                     direction=(dy, dx), lengths=tuple(lengths)))
    return readings[0] if len(readings) == 1 else None


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "uniform outer margin")
    keys = []
    for cells in _components(a != bg):
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        key = _glyph(a[r0:r1 + 1, c0:c1 + 1])
        if key is not None:
            key.update(box=(r0, r1, c0, c1), cells=cells)
            keys.append(key)
    _require(len(keys) == 1, "exactly one uniquely decodable key required")
    key = keys[0]
    key_mask = np.zeros(a.shape, bool)
    for p in key["cells"]:
        key_mask[p] = True
    wall = (a == key["carrier"]) & ~key_mask
    body_mask = (a != bg) & ~key_mask & ~wall
    body = set(map(tuple, np.argwhere(body_mask)))
    _require(body and np.any(wall), "missing payload or walls")
    r0, r1, c0, c1 = _bbox(body)
    _require(len(body) == (r1 - r0 + 1) * (c1 - c0 + 1), "complete rectangular payload required")
    palette = set(map(int, a[body_mask]))
    _require(palette.isdisjoint({bg, key["carrier"], key["trail"], key["tip"]}), "role overlap")
    # Association is enclosure membership, not proximity to the key.
    free = ~wall & ~key_mask
    region = next((s for s in _components(free) if (r0, c0) in s), set())
    _require(body <= region and not any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                                        for r, c in region), "payload must be enclosed")
    return dict(grid=a, bg=bg, key=key, wall=wall, body=body,
                box=(r0, r1, c0, c1), palette=palette)


def _shift(points, dy, dx):
    return {(r + dy, c + dx) for r, c in points}


def _fits(points, wall):
    h, w = wall.shape
    return all(0 <= r < h and 0 <= c < w and not wall[r, c] for r, c in points)


def _step_ok(points, direction, wall, corner=True):
    dy, dx = direction
    if not _fits(_shift(points, dy, dx), wall):
        return False
    return not (corner and dy and dx) or (
        _fits(_shift(points, dy, 0), wall) and _fits(_shift(points, 0, dx), wall))


def _travel(points, direction, wall, corner=True):
    _require(_fits(points, wall), "illegal sheared placement")
    current, distance = points, 0
    # Every finite arena blocks a nonzero vector within thirty steps.
    while _step_ok(current, direction, wall, corner):
        current = _shift(current, *direction)
        distance += 1
        _require(distance <= 30, "unbounded motion")
    return distance


def _sheared(info, slope=None):
    dx = info["key"]["direction"][1] if slope is None else slope
    bottom = info["box"][1]
    return {(r, c + (bottom - r) * dx) for r, c in info["body"]}


def _paint(info, distance, slope=None):
    a, out = info["grid"], info["grid"].copy()
    dy, dx = info["key"]["direction"]
    shear = dx if slope is None else slope
    bottom = info["box"][1]
    for p in info["body"]:
        out[p] = info["bg"]
    for r, c in info["body"]:
        out[r + distance * dy, c + (bottom - r) * shear + distance * dx] = a[r, c]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    shape = _sheared(info)
    distance = _travel(shape, info["key"]["direction"], info["wall"])
    return _paint(info, distance)


def _make_key(rng, direction, carrier, trail, tip):
    dy, dx = direction
    count = int(rng.choice((2, 2, 3)))
    length = int(rng.choice((2, 2, 3)))
    points, offset = [], 0
    for _ in range(count):
        stagger = int(rng.integers(0, 2))
        base = (0, offset) if dy else (offset, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dy,
                           base[1] + (j + stagger) * dx, tip if j == length else trail))
        offset += int(rng.integers(2, 4))
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in points])
    pads = [int(rng.choice((1, 1, 2))) for _ in range(4)]
    key = np.full((r1 - r0 + 1 + pads[0] + pads[1],
                   c1 - c0 + 1 + pads[2] + pads[3]), carrier, np.int8)
    for r, c, color in points:
        key[r - r0 + pads[0], c - c0 + pads[2]] = color
    _require(_glyph(key) is not None, "constructed ambiguous key")
    return key


def _payload(rng, h, w, colors, anchor):
    k = int(rng.integers(1, 7)) if not anchor else int(rng.integers(3, 7))
    palette = rng.choice(colors, size=k, replace=False)
    rr, cc = np.indices((h, w))
    style = int(rng.integers(0, 7))
    if style == 0:
        ix = np.zeros((h, w), int)
    elif style == 1:
        ix = rng.integers(0, k, size=(h, w))
    elif style == 2:
        ix = np.broadcast_to(rng.integers(0, k, size=w), (h, w))
    elif style == 3:
        ix = np.broadcast_to(rng.integers(0, k, size=h)[:, None], (h, w))
    elif style == 4:
        ix = (rr + cc) % k
    elif style == 5:
        ix = (rr // int(rng.integers(1, 3)) + cc // int(rng.integers(1, 3))) % k
    else:
        ix = np.zeros((h, w), int)
        ix[int(rng.integers(h)):, int(rng.integers(w)):] = k - 1
    payload = palette[ix].astype(np.int8, copy=True)
    if anchor:
        # A visible asymmetric ordered row prevents reflection/color-remapping shortcuts.
        payload[0, :] = palette[2]
        payload[0, 0] = palette[0]
        payload[0, 1] = palette[1]
    return payload


def _outer_wall(rng, h, w):
    footprint = np.ones((h, w), bool)
    if rng.random() < .65:
        for corner in rng.choice(4, size=int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            rows = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cols = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rows, cols] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dy, dx in CARDINAL:
        interior &= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return footprint & ~interior, interior


def _safe_sweep(shape, original_sweep, direction, distance):
    safe = set(original_sweep)
    dy, dx = direction
    for t in range(distance + 1):
        current = _shift(shape, t * dy, t * dx)
        safe |= current
        if t < distance and dy and dx:
            safe |= _shift(current, dy, 0) | _shift(current, 0, dx)
    return safe


def _arena(rng, payload, direction, ah, aw, mode):
    """Reserve the deformation/motion corridor, then add an explicit stopping wall."""
    h, w = payload.shape
    dy, dx = direction
    wall, interior = _outer_wall(rng, ah, aw)
    positions = [(r, c) for r in range(1, ah - h) for c in range(1, aw - w)]
    for index in rng.permutation(len(positions)):
        r0, c0 = positions[int(index)]
        original = {(r0 + r, c0 + c) for r in range(h) for c in range(w)}
        shape = {(r0 + r, c0 + c + (h - 1 - r) * dx) for r in range(h) for c in range(w)}
        deformation = {(r0 + r, c) for r in range(h)
                       for c in range(c0 + min(0, (h - 1 - r) * dx),
                                      c0 + w + max(0, (h - 1 - r) * dx))}
        if not all(0 <= r < ah and 0 <= c < aw and interior[r, c] for r, c in deformation):
            continue
        limit = _travel(shape, direction, wall)
        if limit < (2 if mode != "natural" else 1):
            continue
        target = limit
        blocker = None
        if mode != "natural":
            for t in rng.permutation(np.arange(1, limit)):
                t = int(t)
                safe = _safe_sweep(shape, deformation, direction, t)
                current = _shift(shape, t * dy, t * dx)
                next_body = _shift(current, dy, dx)
                if mode == "corner":
                    choices = (_shift(current, dy, 0) | _shift(current, 0, dx)) - next_body - safe
                else:
                    choices = next_body - safe
                choices = sorted(p for p in choices if interior[p])
                if choices:
                    blocker = choices[int(rng.integers(len(choices)))]
                    target = t
                    break
            if blocker is None:
                continue
        safe = _safe_sweep(shape, deformation, direction, target)
        protect = set(safe)
        if mode == "corner":
            protect |= _shift(shape, (target + 1) * dy, (target + 1) * dx)
        new_wall = wall.copy()
        if blocker is not None:
            new_wall[blocker] = True
        # Short piers and posts vary the actual collision geometry, never the rule.
        for _ in range(int(rng.integers(0, 5))):
            br, bc = int(rng.integers(1, ah - 1)), int(rng.integers(1, aw - 1))
            vr, vc = CARDINAL[int(rng.integers(4))]
            for j in range(int(rng.integers(1, 5))):
                p = br + j * vr, bc + j * vc
                if 0 <= p[0] < ah and 0 <= p[1] < aw and interior[p] and p not in protect:
                    new_wall[p] = True
        _require(_travel(shape, direction, new_wall) == target, "stop construction")
        return new_wall, original, shape, (r0, c0), target
    return None


def _make_pair(rng, direction, h, w, anchor, stop_mode):
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, trail, tip = palette[:4]
    payload = _payload(rng, h, w, palette[4:], anchor)
    key = _make_key(rng, direction, carrier, trail, tip)
    kh, kw = key.shape
    dy, dx = direction
    span = w + (h - 1) * abs(dx)
    for _ in range(48):
        gap = int(rng.integers(1, 3))
        min_h, min_w = h + 6, span + 6
        layouts = []
        for side in ("top", "bottom", "left", "right"):
            max_h = 28 - kh - gap if side in ("top", "bottom") else 28
            max_w = 28 if side in ("top", "bottom") else 28 - kw - gap
            if max_h >= min_h and max_w >= min_w:
                layouts.append((side, max_h, max_w))
        _require(layouts, "payload/key do not fit a legal canvas")
        side, max_h, max_w = layouts[int(rng.integers(len(layouts)))]
        ah = int(rng.integers(min_h, min(max_h, min_h + 7) + 1))
        aw = int(rng.integers(min_w, min(max_w, min_w + 7) + 1))
        arena = _arena(rng, payload, direction, ah, aw, stop_mode)
        if arena is None:
            continue
        wall, original, shape, (pr, pc), distance = arena
        if side in ("top", "bottom"):
            ch, cw = ah + kh + gap + 2, max(aw, kw) + 2
        else:
            ch, cw = max(ah, kh) + 2, aw + kw + gap + 2
        ch += int(rng.integers(0, min(2, 30 - ch) + 1))
        cw += int(rng.integers(0, min(2, 30 - cw) + 1))
        if side in ("top", "bottom"):
            ar, kr = (kh + gap + 1, 1) if side == "top" else (1, ah + gap + 1)
            ac = int(rng.integers(1, cw - aw))
            kc = int(rng.integers(1, cw - kw))
        else:
            ac, kc = (kw + gap + 1, 1) if side == "left" else (1, aw + gap + 1)
            ar = int(rng.integers(1, ch - ah))
            kr = int(rng.integers(1, ch - kh))
        a = np.full((ch, cw), bg, np.int8)
        a[kr:kr + kh, kc:kc + kw] = key
        a[ar:ar + ah, ac:ac + aw][wall] = carrier
        a[ar + pr:ar + pr + h, ac + pc:ac + pc + w] = payload
        b = a.copy()
        b[ar + pr:ar + pr + h, ac + pc:ac + pc + w] = bg
        for r in range(h):
            rr = ar + pr + r + distance * dy
            cc = ac + pc + (h - 1 - r) * dx + distance * dx
            b[rr, cc:cc + w] = payload[r]
        # Reject the very unlikely accidental second glyph in a payload pattern.
        try:
            info = _read(a)
        except AssertionError:
            continue
        _require(info["key"]["direction"] == direction, "key render mismatch")
        return {"input": a, "output": b}
    raise RuntimeError("bounded arena construction exhausted")


def _integer(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    sign = int(rng.choice((-1, 1)))
    first_h = int(rng.integers(2, 5))
    second_h = int(rng.choice([v for v in range(2, 5) if v != first_h]))
    plans = [((0, sign), first_h, int(rng.integers(3, 6)), True, "direct"),
             ((int(rng.choice((-1, 1))), -sign), second_h, int(rng.integers(3, 6)), True, "corner")]
    vertical = nt >= 3 and rng.random() < .55
    choices = DIRECTIONS if vertical else tuple(d for d in DIRECTIONS if d[1])
    for i in range(2, nt):
        d = (int(rng.choice((-1, 1))), 0) if i == 2 and vertical else choices[int(rng.integers(len(choices)))]
        mode = "corner" if d[0] and d[1] and rng.random() < .35 else str(rng.choice(("direct", "natural")))
        plans.append((d, int(rng.integers(1, 5)), int(rng.integers(1, 6)), False, mode))
    shift = str(rng.choice(("taller", "wider")))
    for _ in range(nq):
        d = choices[int(rng.integers(len(choices)))]
        h, w = (int(rng.integers(5, 8)), int(rng.integers(1, 6))) if shift == "taller" else (
            int(rng.integers(2, 5)), int(rng.integers(7, 11)))
        mode = "corner" if d[0] and d[1] and rng.random() < .45 else str(rng.choice(("direct", "natural")))
        plans.append((d, h, w, False, mode))
    puzzle, seen = {"train": [], "test": []}, set()
    for index, plan in enumerate(plans):
        for _ in range(12):
            pair = _make_pair(rng, *plan)
            fingerprint = (pair["input"].shape, pair["input"].tobytes())
            if fingerprint not in seen:
                break
        _require(fingerprint not in seen, "unique input construction exhausted")
        seen.add(fingerprint)
        puzzle["train" if index < nt else "test"].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    info = _read(pair["input"])
    a, b = info["grid"], _grid(pair["output"])
    _require(a.shape == b.shape, "canvas changed")
    fixed = (a != info["bg"]) & ~np.isin(a, list(info["palette"]))
    _require(np.array_equal(a[fixed], b[fixed]), "walls/key changed")
    dest = set(map(tuple, np.argwhere(np.isin(b, list(info["palette"])))))
    r0, bottom, c0, c1 = info["box"]
    dy, dx = info["key"]["direction"]
    db = max(r for r, _ in dest)
    bottom_cols = [c for r, c in dest if r == db]
    delta_r, delta_c = db - bottom, min(bottom_cols) - c0
    t = delta_r // dy if dy else delta_c // dx
    _require(t >= 0 and (delta_r, delta_c) == (t * dy, t * dx), "bottom anchor/vector mismatch")
    # Check the row formula directly from returned grids, rather than call solve.
    expected_cells = set()
    for r in range(r0, bottom + 1):
        nr = r + t * dy
        nc = c0 + (bottom - r) * dx + t * dx
        _require(0 <= nr < b.shape[0] and 0 <= nc <= b.shape[1] - (c1 - c0 + 1), "row outside canvas")
        _require(np.array_equal(b[nr, nc:nc + c1 - c0 + 1], a[r, c0:c1 + 1]), "complete row/order changed")
        expected_cells |= {(nr, c) for c in range(nc, nc + c1 - c0 + 1)}
    _require(dest == expected_cells, "payload cell conservation")
    unchanged = ~(np.isin(a, list(info["palette"])) | np.isin(b, list(info["palette"])))
    _require(np.array_equal(a[unchanged], b[unchanged]), "unrelated cells changed")
    for color in info["palette"]:
        _require(np.count_nonzero(a == color) == np.count_nonzero(b == color), "color conservation")
    shape = _sheared(info)
    _require(_fits(shape, info["wall"]), "shear hits a wall")
    for step in range(t):
        _require(_step_ok(_shift(shape, step * dy, step * dx), (dy, dx), info["wall"]), "motion crossed a wall")
    final = _shift(shape, t * dy, t * dx)
    _require(not _step_ok(final, (dy, dx), info["wall"]), "stopped too early")
    _require(t > 0, "generated examples must show sliding")
    _require(np.array_equal(solve(a), b), "input-only solve mismatch")
    corner = bool(dy and dx and _fits(_shift(final, dy, dx), info["wall"]))
    no_shear_t = _travel(info["body"], (dy, dx), info["wall"])
    omit_a = not np.array_equal(_paint(info, no_shear_t, slope=0), b)
    omit_stop = not np.array_equal(_paint(info, 0), b)
    reverse_shape = _sheared(info, -dx)
    reverse_diff = not _fits(reverse_shape, info["wall"])
    if not reverse_diff:
        reverse_t = _travel(reverse_shape, (dy, dx), info["wall"])
        reverse_diff = not np.array_equal(_paint(info, reverse_t, slope=-dx), b)
    wr0, wr1, wc0, wc1 = _bbox(set(map(tuple, np.argwhere(info["wall"]))))
    sr0, sr1, sc0, sc1 = _bbox(shape)
    bbox_distances = []
    if dy > 0:
        bbox_distances.append(wr1 - 1 - sr1)
    elif dy < 0:
        bbox_distances.append(sr0 - wr0 - 1)
    if dx > 0:
        bbox_distances.append(wc1 - 1 - sc1)
    elif dx < 0:
        bbox_distances.append(sc0 - wc0 - 1)
    return dict(direction=(dy, dx), body_shape=(bottom - r0 + 1, c1 - c0 + 1),
                distance=t, corner=corner, omit_a=omit_a, omit_stop=omit_stop,
                reverse_diff=reverse_diff, internal_stop=t < min(bbox_distances),
                key_shape=(info["key"]["box"][1] - info["key"]["box"][0] + 1,
                           info["key"]["box"][3] - info["key"]["box"][2] + 1),
                palette_size=len(info["palette"]), bg=info["bg"])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            record = _audit_pair(pair)
            a = _grid(pair["input"])
            fingerprint = (a.shape, a.tobytes())
            _require(fingerprint not in seen, "duplicate input")
            seen.add(fingerprint)
            rows[split].append(record)
    train = rows["train"]
    _require({r["direction"][1] for r in train} >= {-1, 1}, "both shear signs need demonstrations")
    _require(len({r["body_shape"][0] for r in train}) >= 2, "row-index/height evidence")
    _require(any(r["corner"] for r in train), "missing corner-block demonstration")
    _require(any(r["internal_stop"] for r in train), "missing internal wall demonstration")
    _require(sum(r["omit_a"] and r["omit_stop"] and r["reverse_diff"] for r in train) >= 2,
             "both inherited concepts need counterexamples")
    if any(r["direction"][1] == 0 for r in rows["test"]):
        _require(any(r["direction"][1] == 0 for r in train), "vertical query needs a zero-slope demonstration")
    max_h = max(r["body_shape"][0] for r in train)
    max_w = max(r["body_shape"][1] for r in train)
    _require(all(r["body_shape"][0] > max_h for r in rows["test"])
             or all(r["body_shape"][1] > max_w for r in rows["test"]), "missing structural test shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, all count combinations and targeted corruption checks; no writes."""
    signatures, shapes, key_shapes, body_shapes, directions = set(), set(), set(), set(), set()
    backgrounds, sizes, distances = set(), set(), set()
    corners = internal = vertical = narrow = uniform = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_fingerprint(p))
        if seed in (0, 17, 41):
            _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), "determinism")
        for split in ("train", "test"):
            for pair in p[split]:
                r = _audit_pair(pair)
                shapes.add(pair["input"].shape)
                key_shapes.add(r["key_shape"])
                body_shapes.add(r["body_shape"])
                directions.add(r["direction"])
                backgrounds.add(r["bg"])
                sizes.add(r["palette_size"])
                distances.add(r["distance"])
                corners += r["corner"]
                internal += r["internal_stop"]
                vertical += r["direction"][1] == 0
                narrow += r["body_shape"][1] == 1
                uniform += r["palette_size"] == 1
    _require(len(signatures) == 42 and len(shapes) >= 35, "puzzle/geometry diversity")
    _require(len(key_shapes) >= 15 and len(body_shapes) >= 25, "key/body diversity")
    _require(directions == set(DIRECTIONS) and len(backgrounds) == 10, "direction/color roles")
    _require(len(sizes) >= 5 and len(distances) >= 5, "palette/distance diversity")
    _require(corners >= 42 and internal >= 42 and vertical and narrow and uniform, "mechanism coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + nt * 10 + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A unit witness with hand-computed coordinates checks anchor, sign, and row order.
    wall = np.zeros((10, 12), bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    points = {(3, 3), (3, 4), (4, 2), (4, 3), (5, 1), (5, 2)}
    _require(_travel(points, (0, 1), wall) == 6, "whole-body endpoint")
    wall[4, 6] = True
    _require(_travel(points, (0, 1), wall) == 2, "middle row blocks whole body")
    p = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
