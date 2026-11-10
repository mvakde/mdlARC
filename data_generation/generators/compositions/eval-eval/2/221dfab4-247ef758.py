"""Clocked echo printer: masks supply one axis, colored edge marks the other.

All semantic choices are visible. Generation records are used only to construct
examples; solve parses a single input and never accesses those records.
"""
from __future__ import annotations

import copy
import json
import numpy as np


SIDES = ("top", "right", "bottom", "left")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _orient(a, side):
    return np.rot90(a, -SIDES.index(side)).copy()


def _parse(value):
    a = _grid(value)
    h, w = a.shape
    if min(h, w) < 5:
        raise ValueError("the family needs a frame and an interior")
    corners = [int(a[r, c]) for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))]
    if len(set(corners)) != 1:
        raise ValueError("four frame-colored corners are required")
    frame = corners[0]
    candidates = []
    for color in np.unique(a):
        color = int(color)
        if color == frame:
            continue
        cells = np.argwhere(a == color)
        for side, fixed, edge, lateral in (
            ("top", cells[:, 0], 0, cells[:, 1]),
            ("right", cells[:, 1], w-1, cells[:, 0]),
            ("bottom", cells[:, 0], h-1, cells[:, 1]),
            ("left", cells[:, 1], 0, cells[:, 0]),
        ):
            if np.all(fixed == edge):
                low = int(lateral.min())
                if np.array_equal(np.sort(lateral), np.arange(low, low + len(cells))):
                    candidates.append((side, color))
    if len(candidates) != 1:
        raise ValueError("one uniquely oriented, unpaired edge bar is required")
    side, marker = candidates[0]
    g = np.rot90(a, SIDES.index(side)).copy()
    depth, span = g.shape
    where = np.flatnonzero(g[0] == marker)
    start, length = int(where[0]), len(where)
    if not (1 <= start and start + length <= span-1):
        raise ValueError("the bar must leave both corners intact")
    if not np.all(g[:, 0] == frame) or not np.all(g[:, -1] == frame):
        raise ValueError("edges perpendicular to the seed edge are plain frame")
    interior = g[1:-1, 1:-1]
    vals, counts = np.unique(interior, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    if counts.max() <= interior.size / 2 or bg in (frame, marker):
        raise ValueError("the interior background must have a strict majority")
    if np.any((interior == frame) | (interior == marker)):
        raise ValueError("frame and ruler colors are reserved for structure")
    cols = {}
    for c in range(1, span-1):
        top, bottom = int(g[0, c]), int(g[-1, c])
        if start <= c < start+length:
            if top != marker or bottom != frame:
                raise ValueError("the bar has no opposite partner")
        else:
            if top != bottom or top in (bg, marker):
                raise ValueError("coordinate marks must be paired")
            if top != frame:
                cols.setdefault(top, []).append(c)
    templates = []
    for color in vals:
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(g == color)
        cells = cells[(cells[:, 0] > 0) & (cells[:, 0] < depth-1)
                      & (cells[:, 1] > 0) & (cells[:, 1] < span-1)]
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        shape = tuple(map(int, hi-lo+1))
        if any(s % 2 == 0 for s in shape):
            raise ValueError("template bounding boxes must be odd on both axes")
        center = (lo+hi)//2
        offsets = cells-center
        rows = sorted({int(r) for r, _ in cells if r % 6 == 4})
        cs = cols.get(color, [])
        active = bool(rows and cs)
        if active:
            rh, rw = shape[0]//2, shape[1]//2
            if not all(1+rh <= r <= depth-2-rh for r in rows):
                raise ValueError("an echo would cross the depth border")
            if not all(1+rw <= c <= span-2-rw for c in cs):
                raise ValueError("an echo would cross the lateral border")
        mask = g[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        templates.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                              shape=shape, center=tuple(map(int, center)), mask=mask,
                              offsets=offsets, rows=rows, cols=cs, active=active,
                              first=tuple(map(int, cells[0]))))
    if not templates:
        raise ValueError("at least one colored template is required")
    templates.sort(key=lambda t: t["first"])
    stationary = {tuple(map(int, p)) for t in templates if not t["active"] for p in t["cells"]}
    for t in templates:
        if t["active"]:
            for r in t["rows"]:
                for c in t["cols"]:
                    if stationary.intersection(map(tuple, t["offsets"] + [r, c])):
                        raise ValueError("echo masks cannot cover a retained source template")
    return dict(g=g, side=side, depth=depth, span=span, bg=bg, frame=frame,
                marker=marker, start=start, length=length, templates=templates, cols=cols)


def _ruler(g, p):
    out = g.copy()
    for r in range(1, p["depth"]-1):
        color = p["bg"] if r % 2 else p["frame"] if r % 6 == 4 else p["marker"]
        out[r, p["start"]:p["start"]+p["length"]] = color
    return out


def _prediction(p, mode="correct", track=True):
    """Forward mask placement; alternative modes are diagnostic ablations only."""
    out = p["g"].copy()
    ts = p["templates"]
    if mode == "no_dispatch":
        return _ruler(out, p) if track else out
    plans = []
    for t in ts:
        rows, cols = t["rows"], t["cols"]
        if mode == "center":
            rows = [t["center"][0]] if rows else []
        elif mode == "all_ticks":
            rad = t["shape"][0]//2
            rows = [r for r in range(4, p["depth"]-1, 6) if 1+rad <= r <= p["depth"]-2-rad]
        elif mode == "bbox_hits":
            rows = [r for r in range(t["lo"][0], t["lo"][0]+t["shape"][0]) if r % 6 == 4]
        elif mode == "local_scan":
            rows = sorted({int(r) for r, c in t["cells"] if r % 6 == 4
                           and p["start"] <= c < p["start"]+p["length"]})
        if not rows or not cols:
            continue
        if mode == "erase_boxes":
            r, c = t["lo"]
            bh, bw = t["shape"]
            out[r:r+bh, c:c+bw] = p["bg"]
        else:
            out[t["cells"][:, 0], t["cells"][:, 1]] = p["bg"]
        offsets = t["offsets"]
        if mode == "solid":
            rh, rw = t["shape"][0]//2, t["shape"][1]//2
            offsets = np.array([(r, c) for r in range(-rh, rh+1) for c in range(-rw, rw+1)])
        centers = list(zip(rows, cols)) if mode == "zip_product" else [(r, c) for r in rows for c in cols]
        plans.append((t, centers, offsets))
    if mode == "last_wins":
        plans.reverse()
    elif mode == "destination":
        plans.sort(key=lambda item: min(item[1]))
    elif mode == "small_first":
        plans.sort(key=lambda item: item[0]["shape"][0]*item[0]["shape"][1])
    occupied = set()
    for t, centers, offsets in plans:
        for r, c in centers:
            for rr, cc in offsets + [r, c]:
                rr, cc = int(rr), int(cc)
                # Only diagnostic alternative rules can request non-fitting boxes.
                if 1 <= rr < p["depth"]-1 and 1 <= cc < p["span"]-1 and (rr, cc) not in occupied:
                    out[rr, cc] = t["color"]
                    occupied.add((rr, cc))
    return _ruler(out, p) if track else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    return _orient(_prediction(p), p["side"])


def _mask(rng, h, w, full_rows=False):
    """Lines, solids, outlines, hollow shapes, paths and multipart constellations."""
    m = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(7))
    if min(h, w) == 1 or kind == 0:
        m[:] = True
    elif kind == 1:
        m[[0, -1]] = True
        m[:, [0, -1]] = True
    elif kind == 2:
        for r in range(h):
            m[r, round(r*(w-1)/(h-1))] = True
        for c in range(w):
            m[round(c*(h-1)/(w-1)), c] = True
    elif kind == 3:
        m[h//2, :] = True
        m[:, w//2] = True
        m |= rng.random((h, w)) < .22
    else:
        m = rng.random((h, w)) < float(rng.uniform(.2, .85))
        if kind == 6 and h >= 5:
            m[int(rng.integers(1, h-1))] = False
    # Tightness gives a unique bbox center even for disconnected masks.
    m[0, int(rng.integers(w))] = True
    m[-1, int(rng.integers(w))] = True
    m[int(rng.integers(h)), 0] = True
    m[int(rng.integers(h)), -1] = True
    if full_rows:
        for r in range(h):
            if not m[r].any():
                m[r, int(rng.integers(w))] = True
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _record(mask, top, left, cols=()):
    return dict(mask=mask.copy(), top=int(top), left=int(left), cols=list(map(int, cols)))


def _cells(t):
    return np.argwhere(t["mask"]) + [t["top"], t["left"]]


def _rows(t):
    return sorted({int(r) for r, _ in _cells(t) if r % 6 == 4})


def _free_placement(rng, mask, depth, span, occupied, *, tops=None, need=None, fit=False):
    candidates = []
    h, w = mask.shape
    for top in range(1, depth-h):
        if tops is not None and top not in tops:
            continue
        rows = {top+int(r) for r in np.flatnonzero(mask.any(axis=1)) if (top+int(r)) % 6 == 4}
        if need == "none" and rows or need == "two" and len(rows) < 2 or need == "some" and not rows:
            continue
        if fit and any(not 1+h//2 <= r <= depth-2-h//2 for r in rows):
            continue
        for left in range(1, span-w):
            points = set(map(tuple, np.argwhere(mask)+[top, left]))
            if not points & occupied:
                candidates.append((top, left))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _choose_cols(rng, t, span, start, length, used, count):
    rad = t["mask"].shape[1]//2
    choices = [c for c in range(1+rad, span-1-rad)
               if c not in used and not start <= c < start+length]
    rng.shuffle(choices)
    return sorted(choices[:count]) if len(choices) >= count else None


def _assemble(rng, depth, span, start, length, records, side, absent=0):
    """Construction-label path: reverse mask painting from sampled records."""
    palette = list(map(int, rng.permutation(10)))
    bg, frame, marker = palette[:3]
    records = sorted(records, key=lambda t: tuple(map(int, _cells(t)[0])))
    g = np.full((depth, span), bg, dtype=np.int8)
    g[[0, -1], :] = frame
    g[:, [0, -1]] = frame
    g[0, start:start+length] = marker
    for t, color in zip(records, palette[3:]):
        t["color"] = color
        cells = _cells(t)
        g[cells[:, 0], cells[:, 1]] = color
        for c in t["cols"]:
            g[0, c] = g[-1, c] = color
    unused = palette[3+len(records):]
    available = [c for c in range(1, span-1) if g[0, c] == g[-1, c] == frame]
    rng.shuffle(available)
    for color in unused[:absent]:
        if not available:
            break
        c = available.pop()
        g[0, c] = g[-1, c] = color
    out = g.copy()
    active = [t for t in records if _rows(t) and t["cols"]]
    for t in active:
        cells = _cells(t)
        out[cells[:, 0], cells[:, 1]] = bg
    # Later masks first, earlier masks last: transparent holes are never painted.
    for t in reversed(active):
        h, w = t["mask"].shape
        for r in _rows(t):
            for c in t["cols"]:
                view = out[r-h//2:r+h//2+1, c-w//2:c+w//2+1]
                view[t["mask"]] = t["color"]
    for r in range(1, depth-1):
        out[r, start:start+length] = bg if r % 2 else frame if r % 6 == 4 else marker
    return {"input": _orient(g, side), "output": _orient(out, side)}


def _bar(rng, span):
    length = int(rng.integers(1, max(2, (span-4)//3+1)))
    start = int(rng.integers(1, span-length-1))
    return start, length


def _propose(rng, side, kind, test=False):
    depth = (int(rng.integers(25, 31)) if test else
             int(rng.integers(7 if kind == "random" else 20, 25)))
    span = int(rng.integers(18, 31)) if kind != "random" else int(rng.integers(12, 31))
    start, length = _bar(rng, span)
    occupied, used, records = set(), set(), []
    if kind == "product":
        h, w = int(rng.choice([7, 9, 11])), int(rng.choice([3, 5, 7]))
        mask = _mask(rng, h, w, full_rows=True)
        pos = _free_placement(rng, mask, depth, span, occupied, need="two", fit=True)
        if pos is None:
            return None
        t = _record(mask, *pos)
        cols = _choose_cols(rng, t, span, start, length, used, 2)
        adjacent = [c for c in (start-1, start+length)
                    if 1+w//2 <= c <= span-2-w//2]
        if cols is None or not adjacent:
            return None
        c = int(rng.choice(adjacent))
        cols = sorted({c, next((v for v in cols if v != c), cols[0])})
        if len(cols) != 2:
            return None
        t["cols"] = cols
        records.append(t)
        occupied.update(map(tuple, _cells(t)))
        used.update(cols)
        # Teach the other incomplete-address case: marks, but no scan hit.
        mask = _mask(rng, int(rng.choice([1, 3])), int(rng.choice([1, 3, 5])))
        pos = _free_placement(rng, mask, depth, span, occupied, need="none")
        if pos is None:
            return None
        t = _record(mask, *pos)
        cs = _choose_cols(rng, t, span, start, length, used, 1)
        if cs is None:
            return None
        t["cols"] = cs
        records.append(t)
    elif kind == "selectivity":
        h, w = int(rng.choice([9, 11])), int(rng.choice([5, 7]))
        depth = max(depth, 2+h//2+16)
        mask = _mask(rng, h, w)
        mask[2, :] = False  # scan row 10 is a genuine hole across the template
        mask[0, 0] = mask[-1, -1] = mask[8, -1] = True
        mid = h//2
        mask[mid, w//2-1] = True
        mask[mid, w//2-2] = False
        mask[mid, w//2] = False  # the bounding-box center need not be occupied
        # Source slack is independent of destination coordinates.
        source_choices = [l for l in range(1, span-w)
                          if not any(start <= c < start+length for c in range(l+1, l+4))]
        if not source_choices:
            return None
        left = int(rng.choice(source_choices))
        first = _record(mask, 8, left)
        records.append(first)
        occupied.update(map(tuple, _cells(first)))
        small = _mask(rng, 3, 3)
        small[1, 0] = small[1, 1] = True
        pos = _free_placement(rng, small, depth, span, occupied, tops=[15])
        if pos is None:
            return None
        second = _record(small, *pos)
        records.append(second)
        occupied.update(map(tuple, _cells(second)))
        choices = [c for c in range(1+w//2, span-2-w//2)
                   if all(not start <= v < start+length for v in (c-1, c, c+1))]
        if not choices:
            return None
        cb = int(rng.choice(choices))
        first["cols"], second["cols"] = [cb+1], [cb]
        # A retained line occupies the missing scan row *inside* the active box.
        nested = _record(np.ones((1, 3), dtype=bool), 10, left+1)
        records.append(nested)
    else:
        count = int(rng.integers(1, 6))
        for _ in range(count):
            h = int(rng.choice([1, 3, 5, 7, 9, 11, 13] if test else [1, 3, 5, 7, 9, 11]))
            w = int(rng.choice([1, 3, 5, 7, 9]))
            if h > depth-2 or w > span-4:
                continue
            mask = _mask(rng, h, w)
            has_marks = bool(rng.random() < .77)
            pos = _free_placement(rng, mask, depth, span, occupied, fit=has_marks)
            if pos is None:
                continue
            t = _record(mask, *pos)
            if has_marks:
                cs = _choose_cols(rng, t, span, start, length, used,
                                  int(rng.choice([1, 2, 3], p=[.5, .35, .15])))
                if cs is None:
                    continue
                t["cols"] = cs
                used.update(cs)
            records.append(t)
            occupied.update(map(tuple, _cells(t)))
        if not records:
            return None
    return _assemble(rng, depth, span, start, length, records, side,
                     absent=int(rng.integers(0, 3)))


ABLATIONS = ("no_dispatch", "center", "all_ticks", "bbox_hits", "local_scan",
             "solid", "last_wins", "destination", "small_first", "erase_boxes", "zip_product")


def _features(value):
    p = _parse(value)
    expected = _prediction(p)
    differences = {name: int(np.count_nonzero(expected != _prediction(p, name))) for name in ABLATIONS}
    before_track = _prediction(p, track=False)
    stripe = np.zeros_like(expected, dtype=bool)
    stripe[1:-1, p["start"]:p["start"]+p["length"]] = True
    objects = np.isin(before_track, [t["color"] for t in p["templates"]])
    depths = np.arange(p["depth"])[:, None]
    differences.update(
        track_odd=bool(np.any(stripe & objects & (depths % 2 == 1))),
        track_normal=bool(np.any(stripe & objects & ((depths % 6 == 0) | (depths % 6 == 2)))),
        track_high=bool(np.any(stripe & objects & (depths % 6 == 4))),
        product=any(len(t["rows"]) >= 2 and len(t["cols"]) >= 2 for t in p["templates"]),
        no_rows=any(t["cols"] and not t["rows"] for t in p["templates"]),
        no_cols=any(t["rows"] and not t["cols"] for t in p["templates"]),
        empty_center=any(t["active"] and not t["mask"][t["shape"][0]//2, t["shape"][1]//2] for t in p["templates"]),
        source_interleave=any(a["lo"][0] <= b["lo"][0] < a["lo"][0]+a["shape"][0]
                              and a["lo"][1] <= b["lo"][1] < a["lo"][1]+a["shape"][1]
                              for i, a in enumerate(p["templates"]) for b in p["templates"][i+1:]),
    )
    return differences


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fallback(rng, side, kind, test=False):
    """Finite construction after rejection sampling; same rule and evidence."""
    if kind == "product":
        records = [_record(np.ones((7, 5), bool), 4, 1, [7, 16]),
                   _record(np.ones((1, 3), bool), 1, 20, [22])]
        return _assemble(rng, 24, 26, 8, 2, records, side)
    if kind == "selectivity":
        mask = np.ones((9, 5), bool)
        mask[2] = False
        mask[4, [0, 2]] = False
        records = [_record(mask, 8, 2, [11]),
                   _record(np.ones((3, 3), bool), 15, 20, [10]),
                   _record(np.ones((1, 3), bool), 10, 3)]
        return _assemble(rng, 24, 26, 14, 2, records, side)
    mask = _mask(rng, 3, 3)
    return _assemble(rng, 30 if test else 20, 26, 14, 2,
                     [_record(mask, 9, 2, [18])], side)


def _make(rng, side, kind, test=False):
    # Bounded natural proposals; forced profiles vary masks, spans, placements,
    # palettes and ruler widths rather than stamping evidence pixels on a scene.
    for _ in range(200):
        pair = _propose(rng, side, kind, test)
        if pair is None:
            continue
        try:
            p = _parse(pair["input"])
        except ValueError:
            continue
        if not np.array_equal(_orient(_prediction(p), p["side"]), pair["output"]):
            raise AssertionError("construction and visible-input solution disagree")
        if kind == "product":
            f = _features(pair["input"])
            required = ("product", "no_rows", "local_scan", "center", "zip_product", "track_odd", "track_normal", "track_high")
            if not all(f[k] for k in required):
                continue
        elif kind == "selectivity":
            f = _features(pair["input"])
            required = ("bbox_hits", "no_cols", "solid", "last_wins", "destination", "small_first", "erase_boxes")
            if not all(f[k] for k in required):
                continue
        return pair
    pair = _fallback(rng, side, kind, test)
    p = _parse(pair["input"])
    assert np.array_equal(_orient(_prediction(p), p["side"]), pair["output"])
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    base = int(rng.integers(4))
    test_turn = int(rng.choice([1, 3]))
    puzzle = {"train": [], "test": []}
    for i in range(nt):
        turn = 0 if i < 2 else int(rng.choice([0, 2]))
        kind = "product" if i == 0 else "selectivity" if i == 1 else "random"
        puzzle["train"].append(_make(rng, SIDES[(base+turn) % 4], kind))
    for _ in range(ne):
        puzzle["test"].append(_make(rng, SIDES[(base+test_turn) % 4], "random", test=True))
    # Twenty distinct global bijections exceed the thirteen possible preceding
    # inputs. This finite duplicate repair preserves all geometry and equality.
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        originals = {role: value.copy() for role, value in pair.items()}
        for index in range(20):
            mapping = np.arange(10, dtype=np.int8)
            if index >= 10:
                p = _parse(originals["input"])
                mapping[p["bg"]], mapping[p["frame"]] = p["frame"], p["bg"]
            mapping = (mapping + index % 10) % 10
            if _key(mapping[originals["input"]]) not in seen:
                for role in pair:
                    pair[role] = mapping[originals[role]]
                break
        else:
            raise AssertionError("duplicate repair exhausted")
        seen.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            p = _parse(a)
            expected = solve(a)
            assert np.array_equal(expected, b), "wrong echo pixels, ruler phase, or preservation"
            out = np.rot90(b, SIDES.index(p["side"]))
            assert np.array_equal(out[0], p["g"][0]) and np.array_equal(out[-1], p["g"][-1])
            assert np.array_equal(out[:, 0], p["g"][:, 0]) and np.array_equal(out[:, -1], p["g"][:, -1])
            for r in range(1, p["depth"]-1):
                c = p["bg"] if r % 2 else p["frame"] if r % 6 == 4 else p["marker"]
                assert np.all(out[r, p["start"]:p["start"]+p["length"]] == c)
            features[split].append(_features(a))
    required = ABLATIONS + ("product", "no_rows", "no_cols", "track_odd", "track_normal", "track_high")
    for name in required:
        assert any(f[name] for f in features["train"]), f"missing demonstration: {name}"
    # Empty centers are an extra taught feature when a query actually uses one.
    if any(f["empty_center"] for f in features["test"]):
        assert any(f["empty_center"] for f in features["train"]), "empty centers not demonstrated"


def _puzzle_key(p):
    return tuple((_key(pair["input"]), _key(pair["output"])) for s in ("train", "test") for pair in p[s])


def self_test():
    signatures, shapes, masks, products, sides, lengths, roles, counts = set(), set(), set(), set(), set(), set(), set(), set()
    feature_union = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                sides.add((split, p["side"]))
                lengths.add(p["length"])
                roles.add((p["bg"], p["frame"], p["marker"]))
                counts.add(len(p["templates"]))
                f = _features(pair["input"])
                feature_union.update(k for k, v in f.items() if v)
                for t in p["templates"]:
                    masks.add((t["shape"], t["mask"].tobytes()))
                    if t["active"]:
                        products.add((len(t["rows"]), len(t["cols"])))
                # All rotations and palette permutations must conjugate solve.
                if split == "test":
                    for turn in range(4):
                        assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
                    mapping = np.random.default_rng(700+seed).permutation(10)
                    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        train_p = [_parse(pair["input"]) for pair in puzzle["train"]]
        test_p = [_parse(pair["input"]) for pair in puzzle["test"]]
        assert {p["side"] for p in train_p}.isdisjoint(p["side"] for p in test_p)
        assert min(p["depth"] for p in test_p) > max(p["depth"] for p in train_p)
    assert len(signatures) == 42 and len(shapes) >= 65 and len(masks) >= 140
    assert sides == {(split, side) for split in ("train", "test") for side in SIDES}
    assert {1, 2, 3, 4, 5} <= counts
    assert len(lengths) >= 6 and len(roles) >= 100
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= products
    assert set(ABLATIONS) | {"source_interleave", "empty_center"} <= feature_union
    rng = np.random.default_rng(1042)
    for side in SIDES:
        fallback_puzzle = {"train": [_fallback(rng, side, "product"),
                                      _fallback(rng, side, "selectivity")],
                           "test": [_fallback(rng, side, "random", test=True)]}
        validate_generated_puzzle(fallback_puzzle)
        # Short clocks, a nearly full-width bar, and final-tick truncation are
        # legal family members even though the sampling curriculum is longer.
        for depth, span, start, length, records in (
            (5, 9, 3, 1, [_record(np.ones((1, 3), bool), 2, 2)]),
            (9, 9, 1, 4, [_record(np.ones((3, 3), bool), 3, 1, [6])]),
            (30, 9, 7, 1, [_record(np.ones((1, 1), bool), 28, 1, [1])]),
            (30, 12, 5, 1, [_record(np.ones((13, 3), bool), 10, 1, [3, 8])]),
        ):
            pair = _assemble(rng, depth, span, start, length, records, side)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(6000+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(73, 2, 1)
    for mode in ("no_dispatch", "bbox_hits", "erase_boxes", "last_wins"):
        bad = copy.deepcopy(puzzle)
        for pair in bad["train"]:
            p = _parse(pair["input"])
            pair["output"] = _orient(_prediction(p, mode), p["side"])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted {mode}")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "template_masks": len(masks), "products": sorted(products),
            "ruler_lengths": sorted(lengths), "template_counts": sorted(counts)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
