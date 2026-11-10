"""Complementary compass loom: locally oriented, color-addressed stencils.

Only Python's standard library and NumPy are used. Outputs are constructed from
sampled records; solve() reconstructs everything from the visible input instead.
"""
from __future__ import annotations

import copy
import itertools
import numpy as np


_NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
_MATRICES = {
    "identity": ((1, 0), (0, 1)),
    "r90": ((0, -1), (1, 0)),  # np.rot90: counterclockwise
    "r180": ((-1, 0), (0, -1)),
    "r270": ((0, 1), (-1, 0)),
    "flip_lr": ((1, 0), (0, -1)),
    "flip_ud": ((-1, 0), (0, 1)),
    "transpose": ((0, 1), (1, 0)),
    "anti_transpose": ((0, -1), (-1, 0)),
}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(array):
    a = np.asarray(array, dtype=np.int8)
    return a.shape, a.tobytes()


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid extent"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(int, copy=True)


def _transform(a, name):
    if name == "identity":
        return a.copy()
    if name.startswith("r"):
        return np.rot90(a, {"r90": 1, "r180": 2, "r270": 3}[name]).copy()
    if name == "flip_lr":
        return a[:, ::-1].copy()
    if name == "flip_ud":
        return a[::-1].copy()
    if name == "transpose":
        return a.T.copy()
    if name == "anti_transpose":
        return a.T[::-1, ::-1].copy()
    raise AssertionError("unknown symmetry")


def _matrix(guides, row, col):
    assert row["key"] != col["key"]
    destination = {row["key"]: row["side"], col["key"]: col["side"]}
    m = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        m += np.outer(_NORMAL[destination[color]], _NORMAL[side])
    assert np.array_equal(m.T @ m, np.eye(2, dtype=int))
    return m


def _name(m):
    return next(n for n, v in _MATRICES.items() if np.array_equal(m, v))


def _side_after(side, name):
    vector = tuple(np.array(_MATRICES[name]) @ np.array(_NORMAL[side]))
    return next(s for s, v in _NORMAL.items() if v == vector)


def _guide_cells(r, c, h, w, side):
    if side == "T":
        return [(r - 1, x) for x in range(c, c + w)]
    if side == "B":
        return [(r + h, x) for x in range(c, c + w)]
    if side == "L":
        return [(y, c - 1) for y in range(r, r + h)]
    return [(y, c + w) for y in range(r, r + h)]


def _components(stage, color):
    remaining = set(map(tuple, np.argwhere(stage == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _parse(input_grid):
    """Decode frame syntax and guide endpoints without generation metadata."""
    g = _array(input_grid)
    H, W = g.shape
    bg = int(g[0, 0])
    layouts = []
    for d in range(5, W - 7):
        divider = int(g[0, d])
        if divider == bg or not np.all(g[:, d] == divider):
            continue
        stage = g[:, :d]
        if not (np.all(stage[0] == bg) and np.all(stage[-1] == bg)
                and np.all(stage[:, 0] == bg)):
            continue
        frame = int(g[0, d + 1])
        if len({bg, divider, frame}) != 3 or np.any((stage == divider) | (stage == frame)):
            continue
        if not all(np.all(g[rr:rr + 2, cc:cc + 2] == frame)
                   for rr in (0, H - 2) for cc in (d + 1, W - 2)):
            continue
        interior = g[2:-2, d + 3:-2]
        if not interior.size or not np.all(interior == bg):
            continue
        layouts.append((d, divider, frame))
    assert len(layouts) == 1, "unique divider and double frame required"
    d, divider, frame = layouts[0]
    fh, fw = H - 4, W - d - 5
    rows, cols = [], []

    def read_token(outer, inner, axis, pos, side):
        outer, inner = int(outer), int(inner)
        if outer == inner == frame:
            return None
        assert outer not in (bg, divider, frame) and inner not in (bg, divider, frame)
        assert outer != inner, "body and compass identities must differ"
        return dict(body=outer, key=inner, axis=axis, pos=pos, side=side)

    for r in range(fh):
        y = r + 2
        pair = [read_token(g[y, d + 1], g[y, d + 2], "row", r, "L"),
                read_token(g[y, -1], g[y, -2], "row", r, "R")]
        assert sum(t is not None for t in pair) <= 1, "two row tokens at one coordinate"
        rows.extend(t for t in pair if t is not None)
    for c in range(fw):
        x = d + 3 + c
        pair = [read_token(g[0, x], g[1, x], "col", c, "T"),
                read_token(g[-1, x], g[-2, x], "col", c, "B")]
        assert sum(t is not None for t in pair) <= 1, "two column tokens at one coordinate"
        cols.extend(t for t in pair if t is not None)
    markers = set(t["key"] for t in rows + cols)
    assert len(markers) == 2, "two visible compass identities required"
    assert not markers & set(t["body"] for t in rows + cols)
    stage = g[:, :d]
    bars = []
    for marker in sorted(markers):
        for cells in _components(stage, marker):
            rr, cc = zip(*cells)
            assert len(cells) >= 3 and len(cells) % 2 == 1, "odd guide length >=3 required"
            horizontal = len(set(rr)) == 1
            vertical = len(set(cc)) == 1
            assert horizontal != vertical, "guide must be straight"
            lo, hi = (min(cc), max(cc)) if horizontal else (min(rr), max(rr))
            assert hi - lo + 1 == len(cells)
            bars.append(dict(key=marker, cells=cells, horizontal=horizontal,
                             fixed=rr[0] if horizontal else cc[0], lo=lo, hi=hi))
    sources, used_bars = [], []
    for hi, horizontal in enumerate(bars):
        if not horizontal["horizontal"]:
            continue
        for vi, vertical in enumerate(bars):
            if vertical["horizontal"] or horizontal["key"] == vertical["key"]:
                continue
            y, x = horizontal["fixed"], vertical["fixed"]
            c, c1 = horizontal["lo"], horizontal["hi"]
            r, r1 = vertical["lo"], vertical["hi"]
            hs = "T" if y == r - 1 else "B" if y == r1 + 1 else None
            vs = "L" if x == c - 1 else "R" if x == c1 + 1 else None
            if hs is None or vs is None:
                continue
            h, w = r1 - r + 1, c1 - c + 1
            if r < 1 or c < 1 or r1 >= H - 1 or c1 >= d - 1:
                continue
            payload = stage[r:r + h, c:c + w]
            colors = set(map(int, np.unique(payload))) - {bg}
            if len(colors) != 1 or colors & (markers | {divider, frame}):
                continue
            color = next(iter(colors))
            if np.count_nonzero(stage == color) != np.count_nonzero(payload == color):
                continue
            guides = {hs: horizontal["key"], vs: vertical["key"]}
            valid = True
            for side in _NORMAL:
                cells = _guide_cells(r, c, h, w, side)
                expected = guides.get(side, bg)
                if any(int(stage[p]) != expected for p in cells):
                    valid = False
            if not valid:
                continue
            sources.append(dict(r=r, c=c, mask=payload == color, color=color, guides=guides))
            used_bars.extend((hi, vi))
    assert 1 <= len(sources) <= 3 and len({t["color"] for t in sources}) == len(sources)
    assert sorted(used_bars) == list(range(len(bars))), "ambiguous or unmatched source guides"
    sources.sort(key=lambda t: (t["r"], t["c"]))
    reconstructed = np.full(stage.shape, bg, dtype=int)
    for t in sources:
        r, c = t["r"], t["c"]
        h, w = t["mask"].shape
        assert h % 2 == w % 2 == 1
        for y, x in np.argwhere(t["mask"]):
            assert reconstructed[r + y, c + x] == bg
            reconstructed[r + y, c + x] = t["color"]
        for side, marker in t["guides"].items():
            for p in _guide_cells(r, c, h, w, side):
                assert reconstructed[p] == bg
                reconstructed[p] = marker
    assert np.array_equal(reconstructed, stage), "unexplained source ink"
    info = dict(bg=bg, divider=divider, frame=frame, d=d, shape=(fh, fw),
                sources=sources, rows=rows, cols=cols, markers=markers)
    _requests(info)  # Also check full, guide-defined footprint bounds.
    return info


def _requests(info, mode="correct"):
    """Return visible-source placements, grouped in source priority order."""
    requests = []
    for index, t in enumerate(info["sources"]):
        rows = [p for p in info["rows"] if p["body"] == t["color"]]
        cols = [p for p in info["cols"] if p["body"] == t["color"]]
        candidates = zip(rows, cols) if mode == "zip" else itertools.product(rows, cols)
        for row, col in candidates:
            if row["key"] == col["key"]:
                if mode != "ignore_complement":
                    continue
                col = dict(col, key=next(k for k in info["markers"] if k != row["key"]))
            matrix = _matrix(t["guides"], row, col)
            h, w = t["mask"].shape
            radius = np.abs(matrix) @ np.array([h // 2, w // 2])
            r, c = row["pos"], col["pos"]
            assert radius[0] <= r < info["shape"][0] - radius[0], "row footprint clips field"
            assert radius[1] <= c < info["shape"][1] - radius[1], "column footprint clips field"
            requests.append((index, r, c, matrix))
    return requests


def _predict(info, mode="correct"):
    """Coordinate-scatter implementation used by solve and diagnostic ablations."""
    out = np.full(info["shape"], info["bg"], dtype=int)
    occupied = set()
    reserved = {}
    req = _requests(info, mode)
    if mode == "reverse":
        req.sort(key=lambda q: -q[0])
    elif mode == "size":
        req.sort(key=lambda q: info["sources"][q[0]]["mask"].size)
    elif mode == "destination":
        req.sort(key=lambda q: (q[1], q[2], q[0]))
    for i, r, c, matrix in req:
        t = info["sources"][i]
        mask = t["mask"]
        offsets = np.argwhere(mask) - np.array(mask.shape) // 2
        if mode == "tight":
            points = np.argwhere(mask)
            offsets = points - (points.min(axis=0) + points.max(axis=0)) // 2
        if mode == "solid":
            offsets = np.indices(mask.shape).reshape(2, -1).T - np.array(mask.shape) // 2
        if mode != "no_orientation":
            offsets = offsets @ matrix.T
        for y, x in offsets + np.array([r, c]):
            p = int(y), int(x)
            if (0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1] and p not in occupied
                    and (mode != "reserve_box" or reserved.get(p, i) == i)):
                out[p] = t["color"]
                occupied.add(p)
        if mode == "reserve_box":
            radius = np.abs(matrix) @ (np.array(mask.shape) // 2)
            for y in range(r - radius[0], r + radius[0] + 1):
                for x in range(c - radius[1], c + radius[1] + 1):
                    reserved.setdefault((y, x), i)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; no examples, random state, files or cache."""
    return _predict(_parse(input_grid))


def _sample_mask(rng, h, w, *, teaching=False):
    for _ in range(100):
        mode = int(rng.integers(5))
        mask = np.zeros((h, w), dtype=bool)
        if mode == 0:
            mask = rng.random((h, w)) < rng.uniform(.22, .90)
        elif mode == 1:
            mask[[0, -1], :] = True
            mask[:, [0, -1]] = True
            mask &= rng.random((h, w)) < .85
        elif mode == 2:
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _step in range(int(rng.integers(h + w, 3 * h * w))):
                mask[r, c] = True
                dr, dc = _pick(rng, ((-1, 0), (1, 0), (0, -1), (0, 1)))
                r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
        elif mode == 3:
            for r in range(h):
                mask[r, round(r * (w - 1) / (h - 1))] = True
            mask |= rng.random((h, w)) < .13
        else:
            mask[:] = True
            mask &= rng.random((h, w)) < rng.uniform(.55, 1.01)
        if teaching or rng.random() < .40:
            eligible_sides = ([s for s in _NORMAL if (h if s in ("T", "B") else w) >= 5]
                              if teaching else list(_NORMAL))
            side = _pick(rng, eligible_sides)
            thickness = 2 if teaching else 1
            if side == "T":
                mask[:thickness] = False
            elif side == "B":
                mask[-thickness:] = False
            elif side == "L":
                mask[:, :thickness] = False
            else:
                mask[:, -thickness:] = False
        if mask.any() and (not teaching or len({_key(_transform(mask, n)) for n in _MATRICES}) == 8):
            return mask
    # A finite fallback with blank padding and no nontrivial D4 symmetry.
    mask = np.zeros((h, w), dtype=bool)
    mask[-2, -3:] = True
    mask[-1, -3] = True
    return mask


def _guides(rng, markers):
    sides = [_pick(rng, ("T", "B")), _pick(rng, ("L", "R"))]
    keys = list(markers)
    rng.shuffle(keys)
    return dict(zip(sides, keys))


def _position_sources(rng, sources, H, sw):
    needed = 2 + sum(t["mask"].shape[0] + 2 for t in sources) + len(sources) - 1
    gaps = rng.multinomial(H - needed, [1 / (len(sources) + 1)] * (len(sources) + 1))
    top = 1 + int(gaps[0])
    for i, t in enumerate(sources):
        h, w = t["mask"].shape
        t["r"] = top + 1
        t["c"] = 2 + int(rng.integers(sw - w - 3))
        top += h + 3 + int(gaps[i + 1])


def _assemble(records):
    bg, divider, frame = records["palette"][:3]
    H, sw, fw = records["H"], records["sw"], records["fw"]
    g = np.full((H, sw + 5 + fw), bg, dtype=int)
    g[:, sw] = divider
    g[:, sw + 1:] = frame
    g[2:-2, sw + 3:-2] = bg
    for t in records["sources"]:
        r, c = t["r"], t["c"]
        h, w = t["mask"].shape
        view = g[r:r + h, c:c + w]
        assert np.all(view == bg)
        view[t["mask"]] = t["color"]
        for side, key in t["guides"].items():
            for p in _guide_cells(r, c, h, w, side):
                assert g[p] == bg
                g[p] = key
    for p in records["rows"] + records["cols"]:
        v, body, key = p["pos"], p["body"], p["key"]
        if p["side"] == "L":
            g[v + 2, sw + 1:sw + 3] = [body, key]
        elif p["side"] == "R":
            g[v + 2, -2:] = [key, body]
        elif p["side"] == "T":
            g[:2, sw + 3 + v] = [body, key]
        else:
            g[-2:, sw + 3 + v] = [key, body]
    # Construction-based ground truth: transform arrays, reverse-paint masks.
    out = np.full((H - 4, fw), bg, dtype=int)
    for t in reversed(records["sources"]):
        rows = [p for p in records["rows"] if p["body"] == t["color"]]
        cols = [p for p in records["cols"] if p["body"] == t["color"]]
        for row, col in itertools.product(rows, cols):
            if row["key"] == col["key"]:
                continue
            mask = _transform(t["mask"], _name(_matrix(t["guides"], row, col)))
            h, w = mask.shape
            r, c = row["pos"] - h // 2, col["pos"] - w // 2
            assert 0 <= r <= out.shape[0] - h and 0 <= c <= out.shape[1] - w
            view = out[r:r + h, c:c + w]
            view[mask] = t["color"]
    return {"input": g, "output": out}


def _port(rng, axis, pos, body, key, side=None):
    return dict(axis=axis, pos=int(pos), body=int(body), key=int(key),
                side=side or _pick(rng, ("L", "R") if axis == "row" else ("T", "B")))


def _teaching_records(rng, palette):
    markers = palette[3:5]
    h, w = _pick(rng, ((3, 5), (5, 3), (5, 5)))
    sources = [dict(mask=_sample_mask(rng, h, w, teaching=True), color=palette[5],
                    guides=_guides(rng, markers)),
               dict(mask=np.ones((3, 3), dtype=bool), color=palette[6],
                    guides=_guides(rng, markers))]
    sources[1]["mask"][_pick(rng, ((0, 0), (0, 2), (2, 0), (2, 2)))] = False
    H, sw = int(rng.integers(18, 31)), int(rng.integers(w + 4, 10))
    fw = int(rng.integers(13, 26 - sw))
    _position_sources(rng, sources, H, sw)
    rec = dict(palette=palette, H=H, sw=sw, fw=fw, sources=sources, rows=[], cols=[])
    # Source's horizontal guide must become a row-edge normal in a quarter-turn.
    hs = next(s for s in sources[0]["guides"] if s in ("T", "B"))
    vs = next(s for s in sources[0]["guides"] if s in ("L", "R"))
    a, b = sources[0]["guides"][hs], sources[0]["guides"][vs]
    row_triples = [v for v in itertools.combinations(range(2, H - 6), 3)
                   if min(np.diff(v)) >= 4]
    col_triples = [v for v in itertools.combinations(range(2, fw - 2), 3)
                   if min(np.diff(v)) >= 4]
    rr, cc = _pick(rng, row_triples), _pick(rng, col_triples)
    rkeys, ckeys = [a, a, b], [b, b, a]
    rng.shuffle(rkeys)
    rng.shuffle(ckeys)
    rec["rows"] = [_port(rng, "row", r, palette[5], k) for r, k in zip(rr, rkeys)]
    rec["cols"] = [_port(rng, "col", c, palette[5], k) for c, k in zip(cc, ckeys)]
    row = next(p for p in rec["rows"] if p["key"] == a)
    col = next(p for p in rec["cols"] if p["key"] == b)
    turn = _pick(rng, ("r90", "r270"))
    row["side"], col["side"] = _side_after(hs, turn), _side_after(vs, turn)
    # Seek a real conflict with the later source centered above an early copy.
    first_row = rec["rows"][0]
    possibilities = []
    for target_col in rec["cols"]:
        if target_col["key"] == first_row["key"]:
            continue
        matrix = _matrix(sources[0]["guides"], first_row, target_col)
        points = (np.argwhere(sources[0]["mask"]) - np.array(sources[0]["mask"].shape) // 2) @ matrix.T
        cells = set(map(tuple, points + [first_row["pos"], target_col["pos"]]))
        for dc in (-1, 1):
            r, c = first_row["pos"] - 1, target_col["pos"] + dc
            if not (1 <= c < fw - 1) or c in cc:
                continue
            for key in markers:
                for rs in ("L", "R"):
                    for cs in ("T", "B"):
                        rp = _port(rng, "row", r, palette[6], key, rs)
                        cp = _port(rng, "col", c, palette[6], next(k for k in markers if k != key), cs)
                        m = _matrix(sources[1]["guides"], rp, cp)
                        other = set(map(tuple, (np.argwhere(sources[1]["mask"]) - 1) @ m.T + [r, c]))
                        if cells & other and other - cells:
                            possibilities.append((rp, cp))
    if not possibilities:
        return None
    rp, cp = _pick(rng, possibilities)
    rec["rows"].append(rp)
    rec["cols"].append(cp)
    return rec


def _random_records(rng, palette, *, test, extra_axis):
    n = int(_pick(rng, (1, 2, 2, 3)))
    markers = palette[3:5]
    sources = []
    for i in range(n):
        h, w = _pick(rng, ((3, 3), (3, 5), (5, 3), (5, 5)))
        sources.append(dict(mask=_sample_mask(rng, h, w), color=palette[5 + i],
                            guides=_guides(rng, markers)))
    rad = max(max(t["mask"].shape) // 2 for t in sources)
    counts = [(int(rng.integers(2, 4)), int(rng.integers(2, 4)))]
    if test:
        counts[0] = (4, counts[0][1]) if extra_axis == 0 else (counts[0][0], 4)
    states = ["active"]
    for _ in sources[1:]:
        state = _pick(rng, ("active", "active", "blocked", "row", "col", "none"))
        states.append(state)
        counts.append((int(rng.integers(1, 3)) if state in ("active", "blocked", "row") else 0,
                       int(rng.integers(1, 3)) if state in ("active", "blocked", "col") else 0))
    absent = bool(rng.random() < .40)
    absent_counts = (int(rng.integers(1, 3)), int(rng.integers(1, 3))) if absent else (0, 0)
    needed = 2 + sum(t["mask"].shape[0] + 2 for t in sources) + n - 1
    min_h = max(17, needed, 4 + 2 * rad + sum(a for a, _ in counts) + absent_counts[0])
    H = int(rng.integers(min_h, 31))
    sw = max(t["mask"].shape[1] for t in sources) + 4
    sw += int(rng.integers(10 - sw))
    min_w = max(11, 2 * rad + sum(b for _, b in counts) + absent_counts[1])
    fw = int(rng.integers(min_w, 26 - sw))
    _position_sources(rng, sources, H, sw)
    rec = dict(palette=palette, H=H, sw=sw, fw=fw, sources=sources, rows=[], cols=[])
    pools = [list(map(int, rng.permutation(np.arange(rad, H - 4 - rad)))),
             list(map(int, rng.permutation(np.arange(rad, fw - rad))))]
    for i, ((nr, nc), state) in enumerate(zip(counts, states)):
        blocked_key = int(_pick(rng, markers))
        for axis, count in enumerate((nr, nc)):
            if i == 0:
                keys = [markers[j % 2] for j in range(count)]
                rng.shuffle(keys)
            elif state == "blocked":
                keys = [blocked_key] * count
            elif state == "active":
                keys = [markers[axis]] * count
            else:
                keys = [int(_pick(rng, markers)) for _ in range(count)]
            name = "rows" if axis == 0 else "cols"
            for key in keys:
                rec[name].append(_port(rng, "row" if axis == 0 else "col", pools[axis].pop(),
                                       sources[i]["color"], key))
    for axis, count in enumerate(absent_counts):
        for _ in range(count):
            rec["rows" if axis == 0 else "cols"].append(
                _port(rng, "row" if axis == 0 else "col", pools[axis].pop(), palette[8],
                      int(_pick(rng, markers))))
    return rec


def _features(info, output=None):
    correct = _predict(info)
    requests = _requests(info)
    products, rejected, inactive = [], 0, 0
    for t in info["sources"]:
        rows = [p for p in info["rows"] if p["body"] == t["color"]]
        cols = [p for p in info["cols"] if p["body"] == t["color"]]
        eligible = sum(r["key"] != c["key"] for r, c in itertools.product(rows, cols))
        products.append((len(rows), len(cols), eligible))
        rejected += len(rows) * len(cols) - eligible
        inactive += eligible == 0
    result = dict(shape=info["shape"], sources=len(info["sources"]), products=tuple(products),
                  rejected=rejected, inactive=inactive, placements=len(requests),
                  symmetries=tuple(_name(q[3]) for q in requests),
                  mask_shapes=tuple(t["mask"].shape for t in info["sources"]),
                  guide_corners=tuple(tuple(sorted(t["guides"])) for t in info["sources"]),
                  port_sides=tuple(sorted({p["side"] for p in info["rows"] + info["cols"]})),
                  blank_edges=sum(not t["mask"][0].any() or not t["mask"][-1].any()
                                  or not t["mask"][:, 0].any() or not t["mask"][:, -1].any()
                                  for t in info["sources"]))
    for mode in ("zip", "no_orientation", "ignore_complement", "reverse", "destination", "size",
                 "tight", "solid", "reserve_box"):
        result[mode] = int(np.count_nonzero(correct != _predict(info, mode)))
    if output is not None:
        assert np.array_equal(correct, output), "incorrect field output"
    return result


def _check_count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, 2, 10, "num_train")
    ne = _check_count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    stable, extra_axis = bool(rng.random() < .25), int(rng.integers(2))
    # Bounded search guarantees actual visible witnesses, rather than assuming
    # that chosen parameters survive mask symmetry, merging, or occlusion.
    for _ in range(160):
        records = _teaching_records(rng, palette)
        if records is None:
            continue
        first = _assemble(records)
        flipped = copy.deepcopy(records)
        flipped["sources"][0]["guides"] = {
            side: next(k for k in palette[3:5] if k != key)
            for side, key in flipped["sources"][0]["guides"].items()}
        second = _assemble(flipped)
        info = _parse(first["input"])
        features = _features(info)
        if (all(features[m] > 0 for m in ("zip", "no_orientation", "ignore_complement", "reverse",
                                          "destination", "size", "tight", "solid", "reserve_box"))
                and not np.array_equal(first["output"], second["output"])):
            break
    else:
        raise AssertionError("could not construct teaching witnesses")
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    for split, count in (("train", nt - 2), ("test", ne)):
        for _index in range(count):
            for _attempt in range(80):
                colors = palette if stable else list(map(int, rng.permutation(10)))
                record = _random_records(rng, colors, test=split == "test", extra_axis=extra_axis)
                pair = _assemble(record)
                _parse(pair["input"])
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise AssertionError("distinct input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information, measurements = set(), {}, {}
    for split in ("train", "test"):
        information[split], measurements[split] = [], []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert out.shape == info["shape"], "return the complete field interior"
            assert set(map(int, np.unique(out))) <= {info["bg"]} | {t["color"] for t in info["sources"]}
            f = _features(info, out)
            assert f["placements"] > 0
            information[split].append(info)
            measurements[split].append(f)
    train, test = measurements["train"], measurements["test"]
    for mode in ("zip", "no_orientation", "ignore_complement", "reverse", "destination", "size",
                 "tight", "solid", "reserve_box"):
        assert any(f[mode] for f in train), f"missing training witness against {mode}"
    assert any(f["rejected"] and f["products"][0][2] >= 4 for f in train)
    assert all(max(max(a, b) for a, b, _ in f["products"]) <= 3 for f in train)
    assert all(4 in f["products"][0][:2] and min(f["products"][0][:2]) >= 2 for f in test)
    # Exact controlled contrast: only the first source's guide identities change.
    a, b = information["train"][:2]
    ga, gb = map(lambda p: _array(p["input"]), puzzle["train"][:2])
    t, u = a["sources"][0], b["sources"][0]
    assert a["rows"] == b["rows"] and a["cols"] == b["cols"] and a["shape"] == b["shape"]
    assert (t["r"], t["c"]) == (u["r"], u["c"]) and np.array_equal(t["mask"], u["mask"])
    assert set(t["guides"]) == set(u["guides"])
    expected_differences = set()
    for side, key in t["guides"].items():
        assert key != u["guides"][side]
        expected_differences.update(_guide_cells(t["r"], t["c"], *t["mask"].shape, side))
    assert set(map(tuple, np.argwhere(ga != gb))) == expected_differences
    assert not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"])
    assert any(n in ("r90", "r270") for f in train for n in f["symmetries"])
    assert any(n in ("flip_lr", "flip_ud", "transpose", "anti_transpose")
               for f in train for n in f["symmetries"])
    return None


def self_test():
    """Bounded 42-seed diversity, counts, counterfactual and mutation checks."""
    signatures, shapes, masks, palettes, products, corners = set(), set(), set(), set(), set(), set()
    symmetries = {"train": set(), "test": set()}
    source_counts, inactive, blank, merges, pairs = set(), 0, 0, 0, 0
    original = None
    for seed in range(42):
        p = generate_puzzle(seed)
        if seed == 0:
            original = p
        signatures.add(tuple(_key(x["input"]) for s in ("train", "test") for x in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _parse(pair["input"])
                f = _features(info, pair["output"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(pair["input"].shape)
                palettes.add((info["bg"], info["divider"], info["frame"], tuple(sorted(info["markers"]))))
                source_counts.add(f["sources"])
                products.update(f["products"])
                corners.update(f["guide_corners"])
                symmetries[split].update(f["symmetries"])
                inactive += f["inactive"] > 0
                blank += f["blank_edges"] > 0
                for t in info["sources"]:
                    masks.add(_key(t["mask"]))
                claims = []
                for i, r, c, m in _requests(info):
                    t = info["sources"][i]
                    cells = set(map(tuple, (np.argwhere(t["mask"]) - np.array(t["mask"].shape) // 2) @ m.T + [r, c]))
                    merges += any(i == j and bool(cells & old) for j, old in claims)
                    claims.append((i, cells))
                pairs += 1
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(masks) >= 60 and len(palettes) >= 35
    assert source_counts == {1, 2, 3} and len(products) >= 12 and len(corners) == 4
    assert all(v == set(_MATRICES) for v in symmetries.values())
    assert inactive >= 5 and blank >= 42 and merges >= 5
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(71491 + a * 17 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for mode in ("zip", "no_orientation", "ignore_complement", "reverse", "destination", "size",
                 "tight", "solid", "reserve_box", "cell", "crop", "token"):
        bad = copy.deepcopy(original)
        pair = bad["train"][0]
        if mode == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif mode == "crop":
            pair["output"] = pair["output"][1:]
        elif mode == "token":
            d = _parse(pair["input"])["d"]
            port = _parse(pair["input"])["rows"][0]
            y, x = port["pos"] + 2, d + 1 if port["side"] == "L" else pair["input"].shape[1] - 1
            pair["input"][y, x] = _parse(pair["input"])["frame"]
        else:
            pair["output"] = _predict(_parse(pair["input"]), mode)
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError, IndexError, StopIteration):
            pass
        else:
            raise AssertionError(f"validator accepted {mode} mutation")
    # Recoloring is semantic, including black in any role.
    for shift in range(10):
        pair = original["test"][0]
        assert np.array_equal(solve((pair["input"] + shift) % 10), (pair["output"] + shift) % 10)
    # Solve wider family members, beyond the sampler's lengths 3 and 5. The
    # transformed full rectangle touches two interior boundaries exactly.
    for shape in ((7, 3), (3, 7), (7, 7)):
        mask = np.ones(shape, dtype=bool)
        mask[0] = False
        mask[1, 1] = False
        for name, values in _MATRICES.items():
            matrix = np.array(values)
            radius = np.abs(matrix) @ (np.array(shape) // 2)
            rows, cols = [], []
            for side, marker in (("T", 3), ("L", 4)):
                target = _side_after(side, name)
                axis = "row" if target in ("L", "R") else "col"
                port = dict(axis=axis, pos=int(radius[0] if axis == "row" else 12 - radius[1]),
                            body=5, key=marker, side=target)
                (rows if axis == "row" else cols).append(port)
            records = dict(palette=list(range(10)), H=23, sw=11, fw=13,
                           sources=[dict(mask=mask, color=5, guides={"T": 3, "L": 4}, r=2, c=2)],
                           rows=rows, cols=cols)
            pair = _assemble(records)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    return dict(seeds=42, pairs=pairs, input_shapes=len(shapes), masks=len(masks), palettes=len(palettes),
                products=len(products), source_counts=sorted(source_counts), inactive_pairs=inactive,
                blank_edge_pairs=blank, same_body_intersections=merges,
                symmetries={s: sorted(v) for s, v in symmetries.items()},
                semantic_verification="generator-owned checks only")
