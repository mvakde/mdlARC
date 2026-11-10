"""Stencil-released fragments slide to the edge matching their former frame.

The enclosure visibly encodes both directions. No learned palette, seed, or
generation metadata is needed by solve. Coordinates below are canonicalized to
horizontal motion; a vertical input is decoded by transposing the whole scene.
"""
from __future__ import annotations

import copy
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.uint8).tobytes()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _span(m):
    return all((m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any()))


def _decode(value):
    """Locate the enclosure, plaques, and registers from visible pixels."""
    original = _grid(value)
    bg = int(original[0, 0])
    if not all(np.all(edge == bg) for edge in
               (original[0], original[-1], original[:, 0], original[:, -1])):
        raise ValueError("the outer canvas margin must be background")
    candidates = []
    for axis, g in ((1, original), (0, original.T)):
        for cap in np.unique(g):
            if cap == bg:
                continue
            cells = np.argwhere(g == cap)
            y, x = cells.min(axis=0)
            b, r = cells.max(axis=0)
            h, w = int(b-y-1), int(r-x-1)
            if h < 5 or w < 8 or len(cells) != 2*(w+2):
                continue
            if not (np.all(g[y, x:r+1] == cap) and np.all(g[b, x:r+1] == cap)):
                continue
            low, high = int(g[y+1, x]), int(g[y+1, r])
            if len({bg, int(cap), low, high}) != 4:
                continue
            if not (np.all(g[y+1:b, x] == low) and np.all(g[y+1:b, r] == high)):
                continue
            candidates.append((axis, g, int(cap), low, high,
                               int(y), int(x), int(b), int(r)))
    if len(candidates) != 1:
        raise ValueError("expected one visibly directed enclosure")
    axis, g, cap, low, high, y, x, b, r = candidates[0]
    stage = g[y+1:b, x+1:r]
    covered = stage == bg
    plaques = []
    for yy, xx in np.argwhere(stage != bg):
        yy, xx = int(yy), int(xx)
        if covered[yy, xx]:
            continue
        frame = int(stage[yy, xx])
        if frame not in (low, high) or yy+3 >= stage.shape[0] or xx+3 >= stage.shape[1]:
            raise ValueError("unrecognized plaque")
        color = int(stage[yy+1, xx+1])
        if color in (bg, cap, low, high):
            raise ValueError("payload must be distinct from local markers")
        bb, rr = yy+2, xx+2
        while bb < stage.shape[0] and stage[bb, xx+1] == color:
            bb += 1
        while rr < stage.shape[1] and stage[yy+1, rr] == color:
            rr += 1
        if bb >= stage.shape[0] or rr >= stage.shape[1]:
            raise ValueError("open plaque frame")
        patch = stage[yy:bb+1, xx:rr+1]
        if not (np.all(patch[0] == frame) and np.all(patch[-1] == frame)
                and np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
                and np.all(patch[1:-1, 1:-1] == color)
                and not covered[yy:bb+1, xx:rr+1].any()):
            raise ValueError("invalid solid framed plaque")
        ih, iw = patch.shape[0]-2, patch.shape[1]-2
        if not (2 <= ih <= 5 and 2 <= iw <= 5):
            raise ValueError("plaque interiors must be 2..5 on each side")
        covered[yy:bb+1, xx:rr+1] = True
        plaques.append(dict(r=yy, c=xx, h=ih, w=iw, color=color,
                            d=int(frame == high), frame=frame))
    if not plaques or len(plaques) > 6 or not covered.all():
        raise ValueError("stage must contain one to six plaques and background")
    outside = np.ones(g.shape, dtype=bool)
    outside[y:b+1, x:r+1] = False
    patterns, boxes = {}, {}
    for color in np.unique(g[outside]):
        color = int(color)
        if color == bg:
            continue
        if color in (cap, low, high):
            raise ValueError("enclosure markers cannot be stencil ink")
        cells = np.argwhere(outside & (g == color))
        top, left = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0)+1)
        register = g[top:bottom, left:right]
        m = register == color
        if (not outside[top:bottom, left:right].all()
                or not np.all((register == bg) | m)
                or not 2 <= m.shape[0] <= 5 or not 2 <= m.shape[1] <= 5
                or not _span(m) or not 0 < m.sum() < m.size):
            raise ValueError("ambiguous or empty/full stencil register")
        patterns[color], boxes[color] = m.copy(), (top, left, bottom, right)
    for p in plaques:
        if p["color"] not in patterns or patterns[p["color"]].shape != (p["h"], p["w"]):
            raise ValueError("each plaque needs one equal-sized same-color stencil")
    all_boxes = list(boxes.values()) + [(y, x, b+1, r+1)]
    for i, a in enumerate(all_boxes):
        for z in all_boxes[i+1:]:
            if not (a[2] < z[0] or z[2] < a[0] or a[3] < z[1] or z[3] < a[1]):
                raise ValueError("registers and enclosure need background separation")
    return dict(axis=axis, bg=bg, cap=cap, rails=(low, high),
                shape=stage.shape, plaques=plaques, patterns=patterns,
                boxes=boxes, enclosure=(y, x, b+1, r+1))


def _components(mask, eight=False):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        todo, cells = [(r, c)], []
        seen[r, c] = True
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
            if eight:
                neighbors += ((-1, -1), (-1, 1), (1, -1), (1, 1))
            for dy, dx in neighbors:
                yy, xx = y+dy, x+dx
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _fragments(info, mode="normal", transform=None):
    fragments = []
    for index, p in enumerate(info["plaques"]):
        m = info["patterns"][p["color"]]
        if transform is not None:
            m = transform(m)
            if m.shape != (p["h"], p["w"]):
                raise ValueError("transformed stencil has incompatible dimensions")
        remaining = np.ones(m.shape, dtype=bool) if mode == "no_carve" else ~m
        parts = ([np.argwhere(remaining)] if mode in ("welded", "no_carve")
                 else _components(remaining, eight=mode == "eight"))
        for part in parts:
            fragments.append(dict(cells=part + [p["r"]+1, p["c"]+1],
                                  d=p["d"], color=p["color"], plaque=index))
    return fragments


def _row_intervals(f):
    return {int(r): (int(f["cells"][f["cells"][:, 0] == r, 1].min()),
                     int(f["cells"][f["cells"][:, 0] == r, 1].max()))
            for r in np.unique(f["cells"][:, 0])}


def _orders(fragments):
    """Initial row order induces an acyclic destination-first partial order.

    Tied ready fragments have disjoint row supports, so their order has no
    effect. Mutual interleaving and cycles are outside this family.
    """
    rows = [_row_intervals(f) for f in fragments]
    orders = []
    for d in (0, 1):
        indices = [i for i, f in enumerate(fragments) if f["d"] == d]
        edges = {i: set() for i in indices}
        degree = {i: 0 for i in indices}
        for at, i in enumerate(indices):
            for j in indices[at+1:]:
                shared = rows[i].keys() & rows[j].keys()
                if not shared:
                    continue
                i_left = all(rows[i][r][1] < rows[j][r][0] for r in shared)
                j_left = all(rows[j][r][1] < rows[i][r][0] for r in shared)
                if not (i_left or j_left):
                    raise ValueError("same-direction fragments mutually interleave")
                a, b = (i, j) if i_left else (j, i)
                if d == 1:
                    a, b = b, a
                edges[a].add(b)
                degree[b] += 1
        order = []
        while len(order) < len(indices):
            ready = [i for i in indices if degree[i] == 0 and i not in order]
            if not ready:
                raise ValueError("cyclic initial fragment order")
            i = min(ready, key=lambda n: tuple(map(tuple, fragments[n]["cells"])))
            order.append(i)
            for j in edges[i]:
                degree[j] -= 1
        orders.append(order)
    return orders


def _settle(info, fragments, bbox=False):
    h, w = info["shape"]
    final = [dict(f, cells=f["cells"].copy()) for f in fragments]
    for d, order in enumerate(_orders(fragments)):
        occupied = np.zeros((h, w), dtype=bool)
        for i in order:
            cells = final[i]["cells"]
            if bbox:
                y, x = cells.min(axis=0)
                b, r = cells.max(axis=0)+1
                footprint = np.array([(yy, xx) for yy in range(y, b) for xx in range(x, r)])
            else:
                footprint = cells.copy()
            if occupied[footprint[:, 0], footprint[:, 1]].any():
                raise ValueError("initial fragment hits an earlier settled fragment")
            delta = -1 if d == 0 else 1
            while True:
                proposed = footprint + [0, delta]
                if (proposed[:, 1].min() < 0 or proposed[:, 1].max() >= w
                        or occupied[proposed[:, 0], proposed[:, 1]].any()):
                    break
                footprint = proposed
                cells = cells + [0, delta]
            final[i]["cells"] = cells
            occupied[footprint[:, 0], footprint[:, 1]] = True
    return final


def _render(info, fragments):
    out = np.full(info["shape"], info["bg"], dtype=np.int64)
    occupied = np.zeros(info["shape"], dtype=bool)
    for f in fragments:
        rr, cc = f["cells"].T
        if occupied[rr, cc].any():
            raise ValueError("opposite classes overlap at rest")
        occupied[rr, cc] = True
        out[rr, cc] = f["color"]
    return out if info["axis"] == 1 else out.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, carve, release four-connected pieces, and slide without jumping."""
    info = _decode(input_grid)
    return _render(info, _settle(info, _fragments(info)))


def _alternative(info, mode, transform=None):
    fs = _fragments(info, mode=mode, transform=transform)
    if mode == "edge_first":
        final = [dict(f, cells=f["cells"].copy()) for f in fs]
        for d, order in enumerate(_orders(fs)):
            occupied = set()
            for i in order:
                cells = fs[i]["cells"]
                shifts = range(-int(cells[:, 1].min()),
                               info["shape"][1]-int(cells[:, 1].max()))
                if d:
                    shifts = reversed(shifts)
                for shift in shifts:
                    moved = cells+[0, shift]
                    sites = set(map(tuple, moved))
                    if not sites & occupied:
                        final[i]["cells"] = moved
                        occupied.update(sites)
                        break
                else:
                    raise ValueError("edge-first alternative cannot fit")
        return _render(info, final)
    return _render(info, fs if mode == "no_slide" else
                   _settle(info, fs, bbox=mode == "bbox"))


def _differs(info, output, mode, transform=None):
    try:
        return not np.array_equal(_alternative(info, mode, transform), output)
    except ValueError:
        return True  # This proposed rule cannot even produce a valid output.


def _sample_mask(rng, h, w):
    """Full-span subtractive ink; residuals range from bars to many islands."""
    for _ in range(80):
        mode = int(rng.integers(8))
        m = np.zeros((h, w), dtype=bool)
        if mode == 0:
            m[:] = True
            m[int(rng.integers(h)), int(rng.integers(w))] = False
        elif mode == 1:
            m[0] = m[-1] = True
        elif mode == 2:
            m[0] = True
            m[:, int(rng.integers(w))] = True
        elif mode == 3:
            rr, cc = np.indices((h, w))
            m = (rr+cc) % 2 == int(rng.integers(2))
        elif mode == 4:
            m[:] = True
            # A small connected elbow, sometimes at a different offset.
            y, x = int(rng.integers(h-1)), int(rng.integers(w-1))
            m[y, x:x+2] = False
            m[y+1, x] = False
        elif mode == 5:
            m[0, 0] = m[-1, -1] = True
        else:
            m = rng.random((h, w)) < float(rng.uniform(.25, .85))
        if rng.random() < .5:
            m = m[::-1]
        if rng.random() < .5:
            m = m[:, ::-1]
        if _span(m) and 0 < m.sum() < m.size:
            return m.copy()
    m = np.ones((h, w), dtype=bool)
    m[h//2, w//2] = False
    return m


def _overlap(a, b, margin=0):
    return (max(a["r"], b["r"]) < min(a["r"]+a["h"]+2, b["r"]+b["h"]+2)+margin
            and max(a["c"], b["c"]) < min(a["c"]+a["w"]+2, b["c"]+b["w"]+2)+margin)


def _scene(rng, purpose, colors):
    """Build solid plaques and masks, rather than invoking parent generators."""
    masks, plaques = {}, []
    if purpose in ("release", "interlock"):
        ih = 5 if purpose == "release" else int(rng.integers(3, 6))
        iw = 5 if purpose == "release" else int(rng.integers(3, 6))
        h = int(rng.integers(ih+3, 15))
        w = int(rng.integers(max(16, 2*(iw+2)+2), 27))
        row = int(rng.integers(h-ih-1))
        free = w-2*(iw+2)
        gaps = rng.multinomial(free, [.3, .4, .3])
        first = int(gaps[0])
        second = first+iw+2+int(gaps[1])
        if purpose == "release":
            remaining = np.array([[0, 0, 0, 0, 1],
                                  [0, 0, 1, 0, 1],
                                  [0, 0, 1, 0, 0],
                                  [1, 0, 1, 0, 0],
                                  [1, 1, 0, 0, 0]], dtype=bool)
            # Three pieces form a stable overlap chain. Two pieces also
            # touch diagonally, distinguishing four- from eight-connectivity.
            if rng.random() < .5:
                remaining = remaining[::-1]
            if rng.random() < .5:
                remaining = remaining[:, ::-1]
            m = ~remaining
            masks[colors[0]] = m
            # Identical payload and stencil, opposite frames, crossing paths.
            plaques = [dict(r=row, c=second, h=ih, w=iw, d=0, color=colors[0]),
                       dict(r=row, c=first, h=ih, w=iw, d=1, color=colors[0])]
            masks[colors[1]] = _sample_mask(rng, ih, iw)
            if np.array_equal(masks[colors[1]], m):
                masks[colors[1]] = np.ones_like(m)
                masks[colors[1]][1, 1] = False
        else:
            d = int(rng.integers(2))
            a = np.ones((ih, iw), dtype=bool)
            z = a.copy()
            a[-2, :2] = False
            a[-1, 0] = False
            z[-2, 1] = False
            z[-1, :2] = False
            if d:
                a, z = a[:, ::-1].copy(), z[:, ::-1].copy()
            masks[colors[0]], masks[colors[1]] = a, z
            positions = (second, first) if d else (first, second)
            plaques = [dict(r=row, c=c, h=ih, w=iw, d=d, color=color)
                       for c, color in zip(positions, colors)]
    else:
        test = purpose == "test"
        h = int(rng.integers(10 if test else 7, 15 if test else 17))
        w = int(rng.integers(19 if test else 13, 27))
        count = int(rng.integers(3, 7)) if test else int(rng.integers(1, 6))
        for _ in range(count):
            for attempt in range(100):
                existing = list(masks)
                reused = bool(existing and rng.random() < .3)
                if reused:
                    color = _pick(rng, existing)
                    ih, iw = masks[color].shape
                    m = masks[color]
                else:
                    available = [c for c in colors if c not in masks]
                    if not available:
                        continue
                    color = _pick(rng, available)
                    ih, iw = (int(rng.integers(2, 6 if test else 5)) for _ in range(2))
                    m = _sample_mask(rng, ih, iw)
                if ih+2 > h or iw+2 > w:
                    continue
                row = int(rng.integers(h-ih-1))
                col = int(rng.integers(w-iw-1))
                if plaques and rng.random() < .6:
                    other = _pick(rng, plaques)
                    row = _pick(rng, (other["r"], other["r"]+other["h"]+1,
                                      other["r"]-ih-1, other["r"]+1))
                if rng.random() < .15:
                    col = _pick(rng, (0, w-iw-2))
                p = dict(r=row, c=col, h=ih, w=iw, d=int(rng.integers(2)), color=color)
                if not 0 <= row <= h-ih-2 or any(_overlap(p, q, 1) for q in plaques):
                    continue
                trial = dict(shape=(h, w), bg=-1, axis=1, plaques=plaques+[p],
                             patterns=dict(masks))
                trial["patterns"][color] = m
                try:
                    _render(trial, _settle(trial, _fragments(trial)))
                except ValueError:
                    continue
                plaques.append(p)
                masks[color] = m
                break
            else:
                break
        if len(plaques) < (3 if test else 1):
            raise ValueError("insufficient feasible plaques")
        spare = [c for c in colors if c not in masks]
        for color in rng.choice(spare, min(len(spare), int(rng.integers(3))), replace=False):
            ih, iw = (int(rng.integers(2, 6)) for _ in range(2))
            masks[int(color)] = _sample_mask(rng, ih, iw)
    return (h, w), plaques, masks


def _layout(rng, shape, masks):
    """Choose a fitting bank on any side, optionally scatter its registers."""
    sh, sw = shape[0]+2, shape[1]+2
    ids = [int(c) for c in rng.permutation(list(masks))]
    options = []
    for nc in range(1, len(ids)+1):
        nr = (len(ids)+nc-1)//nc
        heights = [max(masks[ids[j]].shape[0] for j in range(i*nc, min((i+1)*nc, len(ids))))
                   for i in range(nr)]
        widths = [max(masks[ids[j]].shape[1] for j in range(i, len(ids), nc)) for i in range(nc)]
        for gap in (1, 2):
            bh, bw = sum(heights)+(nr-1)*gap, sum(widths)+(nc-1)*gap
            for side in ("above", "below", "left", "right"):
                h, w = ((sh+bh+1, max(sw, bw)) if side in ("above", "below")
                        else (max(sh, bh), sw+bw+1))
                if h <= 28 and w <= 28:
                    options.append((side, h, w, nc, heights, widths, gap, bh, bw))
    if not options:
        raise ValueError("no external-register layout fits")
    side, h, w, nc, heights, widths, gap, bh, bw = _pick(rng, options)
    if side in ("above", "below"):
        sy, by = (bh+1, 0) if side == "above" else (0, sh+1)
        sx, bx = int(rng.integers(w-sw+1)), int(rng.integers(w-bw+1))
    else:
        sx, bx = (bw+1, 0) if side == "left" else (0, sw+1)
        sy, by = int(rng.integers(h-sh+1)), int(rng.integers(h-bh+1))
    extra_h, extra_w = int(rng.integers(29-h)), int(rng.integers(29-w))
    top, left = int(rng.integers(extra_h+1))+1, int(rng.integers(extra_w+1))+1
    h, w = h+extra_h+2, w+extra_w+2
    sy, sx, by, bx = sy+top, sx+left, by+top, bx+left
    slots = []
    for j, color in enumerate(ids):
        row, col = divmod(j, nc)
        ih, iw = masks[color].shape
        yy = by+sum(heights[:row])+row*gap+int(rng.integers(heights[row]-ih+1))
        xx = bx+sum(widths[:col])+col*gap+int(rng.integers(widths[col]-iw+1))
        slots.append((yy, xx))
    if rng.random() < .55:
        occupied = np.zeros((h, w), dtype=bool)
        occupied[max(0, sy-1):sy+sh+1, max(0, sx-1):sx+sw+1] = True
        scattered = []
        for color in ids:
            ih, iw = masks[color].shape
            choices = [(yy, xx) for yy in range(1, h-ih) for xx in range(1, w-iw)
                       if not occupied[yy:yy+ih, xx:xx+iw].any()]
            if not choices:
                break
            yy, xx = _pick(rng, choices)
            scattered.append((yy, xx))
            occupied[yy-1:yy+ih+1, xx-1:xx+iw+1] = True
        if len(scattered) == len(ids):
            slots = scattered
    return (h, w), (sy, sx), list(zip(ids, slots))


def _make_pair(rng, params, purpose):
    cap, low, high = params["markers"]
    bg = _pick(rng, [c for c in range(10) if c not in (cap, low, high)])
    colors = [int(c) for c in rng.permutation([c for c in range(10)
                                             if c not in (bg, cap, low, high)])]
    shape, plaques, masks = _scene(rng, purpose, colors)
    canvas, (sy, sx), slots = _layout(rng, shape, masks)
    inp = np.full(canvas, bg, dtype=int)
    h, w = shape
    inp[sy, sx:sx+w+2] = cap
    inp[sy+h+1, sx:sx+w+2] = cap
    inp[sy+1:sy+h+1, sx] = low
    inp[sy+1:sy+h+1, sx+w+1] = high
    for p in plaques:
        yy, xx = sy+1+p["r"], sx+1+p["c"]
        ih, iw = p["h"], p["w"]
        inp[yy:yy+ih+2, xx:xx+iw+2] = (low, high)[p["d"]]
        inp[yy+1:yy+ih+1, xx+1:xx+iw+1] = p["color"]
    for color, (yy, xx) in slots:
        m = masks[color]
        inp[yy:yy+m.shape[0], xx:xx+m.shape[1]][m] = color
    if params["axis"] == 0:
        inp = inp.T.copy()
    info = _decode(inp)
    out = _render(info, _settle(info, _fragments(info)))
    # Reject layouts in which a required teaching distinction disappeared.
    if purpose == "release" and not _differs(info, out, "welded"):
        raise ValueError("release witness disappeared")
    if purpose == "release" and not _differs(info, out, "edge_first"):
        raise ValueError("stable-order witness disappeared")
    if purpose == "interlock" and not _differs(info, out, "bbox"):
        raise ValueError("interlock witness disappeared")
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = dict(markers=tuple(int(c) for c in rng.choice(10, 3, replace=False)),
                  axis=int(rng.integers(2)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            purpose = ("release" if split == "train" and index == 0 else
                       "interlock" if split == "train" and index == 1 else
                       "free" if split == "train" else "test")
            for _ in range(160):
                try:
                    pair = _make_pair(rng, params, purpose)
                except ValueError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, visible geometry, motion invariants, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _count(len(puzzle["train"]), 2, 10, "num_train")
    _count(len(puzzle["test"]), 1, 4, "num_test")
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, training = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            conventions.add((info["axis"], info["cap"], info["rails"]))
            initial = _fragments(info)
            final = _settle(info, initial)
            assert np.array_equal(_render(info, final), output), "incorrect release or settling"
            canonical = output if info["axis"] == 1 else output.T
            assert canonical.shape == info["shape"]
            # Rigid translation and every row/color's mass are preserved.
            for a, z in zip(initial, final):
                offsets = z["cells"]-a["cells"]
                assert np.all(offsets[:, 0] == 0) and len(set(offsets[:, 1])) == 1
                assert (offsets[0, 1] <= 0 if a["d"] == 0 else offsets[0, 1] >= 0)
            for color in {f["color"] for f in initial}:
                expected_rows = np.zeros(info["shape"][0], dtype=int)
                for f in initial:
                    if f["color"] == color:
                        expected_rows += np.bincount(f["cells"][:, 0], minlength=len(expected_rows))
                assert np.array_equal(np.sum(canonical == color, axis=1), expected_rows)
            # Each piece is blocked on its next unit step by edge or own class.
            for i, f in enumerate(final):
                candidate = f["cells"]+[0, -1 if f["d"] == 0 else 1]
                if candidate[:, 1].min() < 0 or candidate[:, 1].max() >= info["shape"][1]:
                    continue
                others = {tuple(cell) for j, z in enumerate(final)
                          if j != i and z["d"] == f["d"] for cell in z["cells"]}
                assert any(tuple(cell) in others for cell in candidate), "piece is not settled"
            for d, order in enumerate(_orders(initial)):
                rank = {j: k for k, j in enumerate(order)}
                for i, a in enumerate(initial):
                    for j, z in enumerate(initial[i+1:], i+1):
                        if a["d"] != d or z["d"] != d:
                            continue
                        rows_a, rows_z = _row_intervals(final[i]), _row_intervals(final[j])
                        for row in rows_a.keys() & rows_z.keys():
                            lead, follow = ((rows_a[row], rows_z[row]) if rank[i] < rank[j]
                                            else (rows_z[row], rows_a[row]))
                            assert lead[1] < follow[0] if d == 0 else follow[1] < lead[0]
            if split == "train":
                training.append((info, output))
    assert len(conventions) == 1, "generated puzzle changes marker convention"
    for mode in ("no_carve", "no_slide", "welded", "bbox", "eight", "edge_first"):
        assert any(_differs(info, out, mode) for info, out in training), f"missing {mode} witness"
    transforms = (lambda m: m[::-1], lambda m: m[:, ::-1], lambda m: m[::-1, ::-1],
                  lambda m: m.T, lambda m: m.T[::-1], lambda m: m.T[:, ::-1],
                  lambda m: m.T[::-1, ::-1])
    for transform in transforms:
        assert any(_differs(info, out, "normal", transform) for info, out in training), "orientation ambiguous"
    assert any(a["color"] == b["color"] and a["d"] != b["d"]
               for info, _ in training for a in info["plaques"] for b in info["plaques"])
    # A contrasting ignored register makes equality matching consequential.
    assert any(set(info["patterns"])-{p["color"] for p in info["plaques"]} for info, _ in training)
    matching_witness = False
    for info, out in training:
        for p in info["plaques"]:
            for color, m in info["patterns"].items():
                own = info["patterns"][p["color"]]
                if color == p["color"] or m.shape != own.shape or np.array_equal(m, own):
                    continue
                changed = dict(info, patterns=dict(info["patterns"]))
                changed["patterns"][p["color"]] = m
                try:
                    matching_witness |= not np.array_equal(
                        _render(changed, _settle(changed, _fragments(changed))), out)
                except ValueError:
                    matching_witness = True
    assert matching_witness, "color matching never affects the label"


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit; no filesystem, network, or hidden solver state."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    axes, markers, shapes, stages, counts, dimensions, backgrounds = (set() for _ in range(7))
    fragment_counts, masks, palettes, sides, layouts = (set() for _ in range(5))
    mixed_dimensions = single_class = disconnected = interlocks = 0
    travels = set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                initial = _fragments(info)
                final = _settle(info, initial)
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
                axes.add(info["axis"]); markers.add((info["cap"], info["rails"]))
                shapes.add(pair["input"].shape); stages.add(info["shape"])
                counts.add(len(info["plaques"])); backgrounds.add(info["bg"])
                dims = {(p["h"], p["w"]) for p in info["plaques"]}
                dimensions.update(dims); mixed_dimensions += len(dims) > 1
                single_class += len({p["d"] for p in info["plaques"]}) == 1
                fragment_counts.add(len(initial)); palettes.add(tuple(sorted(info["patterns"])))
                for m in info["patterns"].values():
                    masks.add((m.shape, m.tobytes()))
                    disconnected += len(_components(~m)) > 1
                y, x, b, r = info["enclosure"]
                for top, left, bottom, right in info["boxes"].values():
                    if bottom < y: sides.add("above")
                    if top > b: sides.add("below")
                    if right < x: sides.add("left")
                    if left > r: sides.add("right")
                layouts.add(tuple((p["r"], p["c"], p["h"], p["w"], p["d"]) for p in info["plaques"]))
                travels.update(abs(int(a["cells"][0, 1]-z["cells"][0, 1]))
                               for a, z in zip(initial, final))
                interlocks += _differs(info, pair["output"], "bbox")
    assert axes == {0, 1} and len(markers) >= 25
    assert len(shapes) >= 60 and len(stages) >= 45
    assert {1, 2, 3, 4, 5} <= counts and len(fragment_counts) >= 8
    assert len(dimensions) >= 12 and len(masks) >= 100
    assert len(palettes) >= 50 and len(layouts) >= 100
    assert backgrounds == set(range(10)) and sides == {"above", "below", "left", "right"}
    assert min(mixed_dimensions, single_class, disconnected, interlocks) > 0
    assert min(travels) == 1 and max(travels) >= 15
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            a = generate_puzzle(271828, nt, ne)
            assert len(a["train"]) == nt and len(a["test"]) == ne
    for counts_bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                       (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    broken = copy.deepcopy(puzzles[0])
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted a corrupt label")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and every legal count combination)")
