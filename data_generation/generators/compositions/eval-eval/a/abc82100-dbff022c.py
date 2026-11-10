"""ARC family: registered stencil feasibility maps inside colored cavities.

Only standard library and NumPy are used. No parent code, examples, seed, or
training data are consulted by solve(). See task_spec.md for the full rule.
"""
from __future__ import annotations

import json

import numpy as np

_FOUR = ((-1, 0), (0, -1), (0, 1), (1, 0))
_EIGHT = _FOUR + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = {tuple(map(int, q)) for q in np.argwhere(mask)}
    result = []
    steps = _EIGHT if diagonal else _FOUR
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        cells, stack = {q}, [q]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(frozenset(cells))
    return result


def _regions(a, bg):
    h, w = a.shape
    regions = []
    for cells in _components(a == bg):
        boundary = set()
        edge = False
        for r, c in cells:
            edge |= r in (0, h - 1) or c in (0, w - 1)
            for dr, dc in _FOUR:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and a[rr, cc] != bg:
                    boundary.add((rr, cc))
        regions.append({"cells": cells, "boundary": frozenset(boundary), "edge": edge})
    return regions


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent input color")
    return int(counts.argmax())


def _parse_stencils(a, bg, neutral):
    """Enumerate exact two-label rays near eight-connected stencil components."""
    components = _components(a == neutral, diagonal=True)
    if not components:
        raise ValueError("no stencil")
    near = {}
    for i, cells in enumerate(components):
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    near.setdefault((r + dr, c + dc), set()).add(i)
    options = [[] for _ in components]
    h, w = a.shape
    for sr, sc in np.argwhere((a != bg) & (a != neutral)):
        sr, sc = int(sr), int(sc)
        source = int(a[sr, sc])
        for dr, dc in _FOUR:
            tr, tc = sr + dr, sc + dc
            qr, qc = sr + 2 * dr, sc + 2 * dc
            if not (0 <= qr < h and 0 <= qc < w):
                continue
            target = int(a[tr, tc])
            if target in (bg, neutral, source) or a[qr, qc] not in (bg, neutral):
                continue
            for i in sorted(near.get((qr, qc), ())):
                cells = components[i]
                options[i].append({
                    "source": source, "target": target,
                    "q": (qr, qc), "direction": (dr, dc), "mask": cells,
                    "labels": frozenset(((sr, sc), (tr, tc))),
                    "offsets": tuple(sorted((r - qr, c - qc) for r, c in cells)),
                })
    if any(len(x) != 1 for x in options):
        raise ValueError("stencil lacks a unique label ray")
    records = [x[0] for x in options]
    if len({x["source"] for x in records}) != len(records):
        raise ValueError("two stencils name the same source")
    labels = [p for rec in records for p in rec["labels"]]
    if len(labels) != len(set(labels)):
        raise ValueError("stencils share labels")
    return records


def _scene(a, bg, records):
    """Erase reference material only in a topology work copy, never in output."""
    clean = a.copy()
    reference = set()
    for rec in records:
        reference.update(rec["mask"] | rec["labels"])
    for q in reference:
        clean[q] = bg
    objects = []
    for cells in _components(clean != bg, diagonal=True):
        colors = {int(clean[q]) for q in cells}
        if len(colors) != 1:
            raise ValueError("scene contours touch different colors")
        objects.append({"color": next(iter(colors)), "cells": cells, "holes": []})
    regions = _regions(clean, bg)
    exterior = set().union(*(x["cells"] for x in regions if x["edge"]))
    if not reference <= exterior:
        raise ValueError("references must lie outside scene contours")
    for rec in records:
        if sum(len(o["cells"]) for o in objects if o["color"] == rec["source"]) < 3:
            raise ValueError("a stencil source must name a scene contour")
    cavities = []
    for region in regions:
        if region["edge"]:
            continue
        colors = {int(clean[q]) for q in region["boundary"]}
        if len(colors) != 1:
            raise ValueError("cavity has mixed boundary colors")
        color = next(iter(colors))
        item = {**region, "color": color}
        cavities.append(item)
        owners = [o for o in objects if region["boundary"] <= o["cells"]]
        if len(owners) != 1:
            raise ValueError("cavity must belong to one eight-connected contour")
        owners[0]["holes"].append(item)
    return clean, cavities, objects, frozenset(exterior)


def _decode(a):
    """The neutral role is inferred from this input's geometry, not training."""
    bg = _background(a)
    candidates = []
    for neutral in map(int, np.unique(a)):
        if neutral == bg:
            continue
        try:
            records = _parse_stencils(a, bg, neutral)
            scene = _scene(a, bg, records)
            candidates.append((bg, neutral, records, *scene))
        except ValueError:
            continue
    if len(candidates) != 1:
        raise ValueError("input does not identify exactly one neutral stencil role")
    return candidates[0]


def _fit(cells, offsets):
    """Origins in C satisfying a+M subset C, including a in C for hole origins."""
    return frozenset((r, c) for r, c in cells
                     if all((r + dr, c + dc) in cells for dr, dc in offsets))


def _box_offsets(offsets):
    # Include the registration, since it too must belong to the cavity.
    cells = set(offsets) | {(0, 0)}
    return tuple((r, c)
                 for r in range(min(r for r, c in cells), max(r for r, c in cells) + 1)
                 for c in range(min(c for r, c in cells), max(c for r, c in cells) + 1))


def _answer(a, records, cavities):
    out = a.copy()
    by_source = {x["source"]: x for x in records}
    for cavity in cavities:
        rec = by_source.get(cavity["color"])
        if rec is not None:
            for q in _fit(cavity["cells"], rec["offsets"]):
                out[q] = rec["target"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Preserve input ink; paint all valid registered stencil origins in cavities."""
    a = _array(input_grid)
    bg, neutral, records, clean, cavities, objects, exterior = _decode(a)
    return _answer(a, records, cavities)


def _turn(cells, k, reflect=False):
    result = []
    for r, c in cells:
        if reflect:
            r = -r
        for _ in range(k % 4):
            r, c = -c, r
        result.append((r, c))
    return tuple(sorted(result))


def _canonical(offsets):
    # The origin remains fixed; different registrations are different masks.
    return min(_turn(offsets, k, flip) for k in range(4) for flip in (False, True))


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _mask(rng, kind=None, small=False):
    kind = kind or _choose(rng, ("line", "hook", "branch", "chevron", "growth", "hole"))
    limit = 4 if small else 6
    if kind == "line":
        cells = {(0, c) for c in range(int(rng.integers(2, limit)))}
    elif kind in ("hook", "branch"):
        h, w = int(rng.integers(2, limit)), int(rng.integers(2, limit))
        row = h - 1 if kind == "hook" else int(rng.integers(h))
        cells = {(r, 0) for r in range(h)} | {(row, c) for c in range(w)}
    elif kind == "chevron":
        radius = int(rng.integers(1, 3 if small else 4))
        cells = {(r, abs(r)) for r in range(-radius, radius + 1)}
    elif kind == "hole":
        above = int(rng.integers(1, 3 if small else 4))
        below = int(rng.integers(1, 3 if small else 4))
        left = int(rng.integers(0, 2))
        cells = {(r, 1) for r in range(-above, below + 1)}
        cells |= {(r, c) for r in (-above, below) for c in range(-left, 2)}
    else:
        h, w = int(rng.integers(2, limit)), int(rng.integers(2, limit))
        desired = int(rng.integers(3, min(15, h * w) + 1))
        cells = {(0, 0)}
        for _ in range(150):
            if len(cells) >= desired:
                break
            r, c = _choose(rng, sorted(cells))
            dr, dc = _choose(rng, _FOUR)
            if 0 <= r + dr < h and 0 <= c + dc < w:
                cells.add((r + dr, c + dc))
        # Choose any occupied left-edge cell as an off-center origin.
        qr, qc = _choose(rng, [(r, c) for r, c in sorted(cells) if c == 0])
        cells = {(r - qr, c - qc) for r, c in cells}
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    return _turn(cells, k, flip), _turn(((0, 1),), k, flip)[0]


def _stencil_item(offsets, direction, source, target, bg, neutral):
    dr, dc = direction
    sl, tl = (-2 * dr, -2 * dc), (-dr, -dc)
    cells = set(offsets) | {sl, tl, (0, 0)}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h, w = max(r for r, c in cells) - r0 + 1, max(c for r, c in cells) - c0 + 1
    a = np.full((h, w), bg, dtype=np.int8)
    for r, c in offsets:
        a[r - r0, c - c0] = neutral
    a[sl[0] - r0, sl[1] - c0], a[tl[0] - r0, tl[1] - c0] = source, target
    return {"array": a, "kind": "stencil", "offsets": offsets, "direction": direction,
            "source": source, "target": target, "q": (-r0, -c0)}


def _dilate(mask, diagonal=False):
    result = mask.copy()
    for dr, dc in (_EIGHT if diagonal else _FOUR):
        r0, r1 = max(0, dr), min(mask.shape[0], mask.shape[0] + dr)
        c0, c1 = max(0, dc), min(mask.shape[1], mask.shape[1] + dc)
        result[r0:r1, c0:c1] |= mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return result


def _cavity(rng, kind, h=4, w=4, offsets=()):
    if kind == "embed":
        cells = set(offsets) | {(0, 0)}
        # Diagonal stencils are connected by shortest paths to the origin.
        # This creates a single cavity, while preserving each actual offset.
        for r, c in tuple(sorted(cells)):
            rr, cc = r, c
            while rr:
                rr -= 1 if rr > 0 else -1
                cells.add((rr, cc))
            while cc:
                cc -= 1 if cc > 0 else -1
                cells.add((rr, cc))
        return cells
    if kind == "diamond":
        radius = int(rng.integers(1, 4))
        return {(r, c) for r in range(-radius, radius + 1)
                for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    if kind == "profile":
        cells = set()
        left, right = 0, w - 1
        for r in range(h):
            if r:
                shared = int(rng.integers(left, right + 1))
                left, right = int(rng.integers(shared + 1)), int(rng.integers(shared, w))
            cells.update((r, c) for c in range(left, right + 1))
        return cells
    return {(r, c) for r in range(h) for c in range(w)}


def _contour_item(rng, source, bg, kind, offsets=(), small=False):
    maskcells = set(offsets) | {(0, 0)}
    mh = max(r for r, c in maskcells) - min(r for r, c in maskcells) + 1
    mw = max(c for r, c in maskcells) - min(c for r, c in maskcells) + 1
    if kind in ("large", "open"):
        h, w = mh + int(rng.integers(1, 3)), mw + int(rng.integers(1, 3))
        cells = _cavity(rng, "rectangle", h, w)
    elif kind == "tiny":
        cells = {(0, 0)}
    elif kind == "multi":
        h = int(rng.integers(2, 5))
        w1, w2 = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        cells = {(r, c) for r in range(h) for c in range(w1)}
        cells |= {(r, c) for r in range(h) for c in range(w1 + 1, w1 + 1 + w2)}
    else:
        h, w = int(rng.integers(2, 5 if small else 7)), int(rng.integers(2, 5 if small else 8))
        cells = _cavity(rng, kind, h, w, offsets)
    thickness = 1 if small or kind in ("embed", "tiny", "multi", "open") else _choose(rng, (1, 1, 2))
    r0, c0 = min(r for r, c in cells) - thickness, min(c for r, c in cells) - thickness
    h, w = max(r for r, c in cells) - r0 + thickness + 1, max(c for r, c in cells) - c0 + thickness + 1
    cavity = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        cavity[r - r0, c - c0] = True
    full = cavity.copy()
    diagonal = kind not in ("diamond", "embed") and bool(rng.integers(2))
    for _ in range(thickness):
        full = _dilate(full, diagonal)
    wall = full & ~cavity
    if kind == "open":
        # A real four-neighbor leak, not a missing diagonal corner.
        r, c = map(int, _choose(rng, list(np.argwhere(cavity))))
        side = int(rng.integers(4))
        if side == 0:
            wall[:r + 1, c] = False
        elif side == 1:
            wall[r:, c] = False
        elif side == 2:
            wall[r, :c + 1] = False
        else:
            wall[r, c:] = False
    a = np.full((h, w), bg, dtype=np.int8)
    a[wall] = source
    k = int(rng.integers(4))
    a = np.rot90(a, k).copy() if kind not in ("large", "embed", "open") else a
    if rng.random() < .5 and kind not in ("large", "embed", "open"):
        a = np.fliplr(a).copy()
    local = [region["cells"] for region in _regions(a, bg) if not region["edge"]]
    return {"array": a, "kind": "contour", "source": source, "cavities": local}


def _pack(rng, items, bg, small=False):
    """Pack heterogeneous boxes with random positions and occasional edge contacts."""
    area = sum((x["array"].shape[0] + 1) * (x["array"].shape[1] + 1) for x in items)
    min_h = max(x["array"].shape[0] for x in items)
    min_w = max(x["array"].shape[1] for x in items)
    order = sorted(range(len(items)), key=lambda i: -items[i]["array"].size)
    for attempt in range(22):
        low_h, low_w = max(8, min_h), max(8, min_w)
        if attempt >= 14:
            h = w = 30
        else:
            h, w = int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))
            if rng.random() < .15:
                h = w = max(h, w)
        if h * w < area:
            continue
        blocked = np.zeros((h, w), dtype=bool)
        positions = [None] * len(items)
        for i in order:
            ih, iw = items[i]["array"].shape
            available = [(r, c) for r in range(h - ih + 1) for c in range(w - iw + 1)
                         if not np.any(blocked[r:r + ih, c:c + iw])]
            if not available:
                break
            border = [(r, c) for r, c in available if r == 0 or c == 0 or r + ih == h or c + iw == w]
            r, c = _choose(rng, border if border and rng.random() < .3 else available)
            positions[i] = (r, c)
            blocked[max(0, r - 1):min(h, r + ih + 1), max(0, c - 1):min(w, c + iw + 1)] = True
        else:
            a = np.full((h, w), bg, dtype=np.int8)
            records, cavities = [], []
            for item, (r, c) in zip(items, positions):
                ih, iw = item["array"].shape
                a[r:r + ih, c:c + iw] = item["array"]
                if item["kind"] == "stencil":
                    qr, qc = item["q"]
                    q = (r + qr, c + qc)
                    dr, dc = item["direction"]
                    records.append({
                        "source": item["source"], "target": item["target"],
                        "offsets": item["offsets"], "direction": item["direction"], "q": q,
                        "mask": frozenset((q[0] + dr, q[1] + dc) for dr, dc in item["offsets"]),
                        "labels": frozenset(((q[0] - dr, q[1] - dc), (q[0] - 2 * dr, q[1] - 2 * dc))),
                    })
                else:
                    cavities.extend({"color": item["source"],
                                     "cells": frozenset((r + rr, c + cc) for rr, cc in cells)}
                                    for cells in item["cavities"])
            return a, records, cavities
    return None


def _make_pair(rng, bg, neutral, colors, lesson, unfamiliar=()):
    for attempt in range(65):
        small = attempt >= 40
        if lesson == 0:
            mapping = [(colors[0], colors[1])]
        elif lesson == 1:
            mapping = [(colors[0], colors[2]), (colors[2], colors[3])]
        else:
            n = _choose(rng, (1, 1, 2, 2, 3))
            sources = list(map(int, rng.choice(colors, n, replace=False)))
            policy = _choose(rng, ("random", "shared", "chain"))
            if policy == "shared":
                target = _choose(rng, [x for x in colors if x not in sources])
                mapping = [(s, target) for s in sources]
            elif policy == "chain" and n > 1:
                mapping = list(zip(sources, sources[1:] + [_choose(rng, [x for x in colors if x not in sources])]))
            else:
                mapping = [(s, _choose(rng, [x for x in colors if x != s])) for s in sources]
        specifications = []
        for i, (source, target) in enumerate(mapping):
            kind = "hook" if lesson == 0 else "hole" if lesson == 1 and i == 0 else None
            if unfamiliar and i == 0:
                kind = "growth"
            offsets, direction = _mask(rng, kind, small)
            specifications.append((offsets, direction, source, target))
        if unfamiliar and _canonical(specifications[0][0]) in unfamiliar:
            continue
        items = [_stencil_item(m, d, s, t, bg, neutral) for m, d, s, t in specifications]
        sources = {s for s, t in mapping}
        inactive = [x for x in colors if x not in sources]
        first = specifications[0]
        if lesson == 0:
            # Same source, four contrasts: roomy, exact sparse fit, too small,
            # open. A fifth unlisted contour teaches local applicability.
            recipes = [(first[2], "large", first[0]), (first[2], "embed", first[0]),
                       (first[2], "tiny", first[0]), (first[2], "open", first[0]),
                       (_choose(rng, inactive), "rectangle", ())]
        elif lesson == 1:
            recipes = [(first[2], "embed", first[0]), (first[2], "large", first[0]),
                       (specifications[1][2], "large", specifications[1][0]),
                       (first[2], "tiny", first[0])]
            if rng.random() < .5:
                recipes.append((_choose(rng, inactive), "diamond", ()))
        else:
            recipes = [(s, _choose(rng, ("large", "embed")), m) for m, d, s, t in specifications]
            desired = int(rng.integers(max(2, len(recipes)), 7))
            while len(recipes) < desired:
                rec = _choose(rng, specifications)
                kind = _choose(rng, ("profile", "diamond", "multi", "open", "tiny", "large", "embed", "inert"))
                if kind == "inert":
                    recipes.append((_choose(rng, inactive), _choose(rng, ("rectangle", "diamond", "profile")), ()))
                else:
                    recipes.append((rec[2], kind, rec[0]))
        items += [_contour_item(rng, s, bg, kind, m, small) for s, kind, m in recipes]
        packed = _pack(rng, items, bg, small)
        if packed is None:
            continue
        a, records, cavities = packed
        try:
            decoded = _decode(a)
            if decoded[0] != bg or decoded[1] != neutral:
                continue
            signatures = lambda recs: {(x["source"], x["target"], x["q"], x["offsets"]) for x in recs}
            if signatures(decoded[2]) != signatures(records):
                continue
            out = _answer(a, records, cavities)
            if not np.array_equal(_answer(a, decoded[2], decoded[4]), out) or np.array_equal(a, out):
                continue
            feature = _features(a, decoded)
            if lesson == 0 and not (feature["sparse_fit"] and feature["open_box_fit"] and feature["inert_closed"]
                                    and feature["no_fit"] and feature["offcenter"]):
                continue
            if lesson == 1 and not (feature["hole_fit"] and feature["chain"]):
                continue
            return {"input": a, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    backgrounds, neutral, colors = [palette[0], palette[2]], palette[1], palette[3:]
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            lesson = i if split == "train" and i < 2 else None
            bg = backgrounds[i] if lesson is not None else _choose(rng, backgrounds)
            for _ in range(15):
                pair = _make_pair(rng, bg, neutral, colors, lesson, familiar if split == "test" else ())
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        familiar.update(_canonical(x["offsets"]) for x in _decode(pair["input"])[2])
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, decoded=None):
    bg, neutral, records, clean, cavities, objects, exterior = decoded or _decode(a)
    by_source = {x["source"]: x for x in records}
    result = {"background": bg, "neutral": neutral, "shape": a.shape,
              "stencils": len(records), "contours": len(objects), "cavities": len(cavities),
              "mask_sizes": tuple(len(x["offsets"]) for x in records),
              "mapping": tuple(sorted((x["source"], x["target"]) for x in records)),
              "directions": tuple(x["direction"] for x in records),
              "geometries": tuple(_canonical(x["offsets"]) for x in records),
              "cavity_shapes": [], "valid_counts": [], "sparse_fit": False,
              "hole_fit": False, "no_fit": False, "partial_fit": False, "chain": False,
              "diagonal": False, "offcenter": False,
              "open_active": any(o["color"] in by_source and not o["holes"] for o in objects),
              "open_box_fit": False,
              "inert_closed": any(o["color"] not in by_source and o["holes"] for o in objects),
              "multi_hole": any(len(o["holes"]) > 1 for o in objects),
              "border": any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                            for o in objects for r, c in o["cells"]),
              "exterior_fit": False}
    for obj in objects:
        rec = by_source.get(obj["color"])
        if rec is None or obj["holes"]:
            continue
        r0, r1 = min(r for r, c in obj["cells"]), max(r for r, c in obj["cells"])
        c0, c1 = min(c for r, c in obj["cells"]), max(c for r, c in obj["cells"])
        room = frozenset((r, c) for r in range(r0 + 1, r1)
                         for c in range(c0 + 1, c1) if clean[r, c] == bg)
        result["open_box_fit"] |= bool(_fit(room, rec["offsets"]))
    for rec in records:
        result["exterior_fit"] |= bool(_fit(exterior, rec["offsets"]))
        offsets = rec["offsets"]
        result["offcenter"] |= (0, 0) in offsets and (
            min(r for r, c in offsets) + max(r for r, c in offsets) != 0
            or min(c for r, c in offsets) + max(c for r, c in offsets) != 0)
    for cavity in cavities:
        cells = cavity["cells"]
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        result["cavity_shapes"].append(tuple(sorted((r - r0, c - c0) for r, c in cells)))
        rec = by_source.get(cavity["color"])
        if rec is None:
            continue
        origins = _fit(cells, rec["offsets"])
        result["valid_counts"].append(len(origins))
        result["sparse_fit"] |= bool(origins - _fit(cells, _box_offsets(rec["offsets"])))
        result["no_fit"] |= not origins
        result["partial_fit"] |= 0 < len(origins) < len(cells)
        result["hole_fit"] |= bool(origins) and (0, 0) not in rec["offsets"]
        target = by_source.get(rec["target"])
        result["chain"] |= bool(origins) and target is not None and target["target"] != rec["target"]
        owners = [o for o in objects if cavity["boundary"] <= o["cells"]]
        if owners:
            wall = np.zeros(a.shape, dtype=bool)
            for q in owners[0]["cells"]:
                wall[q] = True
            result["diagonal"] |= len(_components(wall)) > 1
    result["cavity_shapes"] = tuple(result["cavity_shapes"])
    result["valid_counts"] = tuple(result["valid_counts"])
    return result


def validate_generated_puzzle(puzzle):
    """Check schema, input decoding, cavity ownership and teaching witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, features, geometries = set(), {"train": [], "test": []}, set()
    train_mapping = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _array(pair["input"]), _array(pair["output"])
            _require(a.shape == out.shape, "dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            decoded = _decode(a)
            bg, neutral, records, clean, cavities, objects, exterior = decoded
            _require(np.array_equal(_answer(a, records, cavities), out), "incorrect feasible-origin map")
            _require(np.array_equal(a[a != bg], out[a != bg]), "input ink changed")
            changed = set(map(tuple, np.argwhere(a != out)))
            _require(bool(changed), "example must show a positive placement")
            _require(not changed & exterior, "exterior was painted")
            for cavity in cavities:
                rec = next((r for r in records if r["source"] == cavity["color"]), None)
                if rec is None:
                    _require(all(out[q] == bg for q in cavity["cells"]), "unlisted cavity changed")
            feature = _features(a, decoded)
            features[split].append(feature)
            if split == "train":
                geometries.update(feature["geometries"])
                for source, target in feature["mapping"]:
                    train_mapping.setdefault(source, set()).add(target)
            else:
                _require(bool(set(feature["geometries"]) - geometries), "test lacks a new registered stencil")
    train = features["train"]
    for name in ("sparse_fit", "hole_fit", "no_fit", "partial_fit", "chain", "offcenter",
                 "open_active", "open_box_fit", "inert_closed", "exterior_fit"):
        _require(any(f[name] for f in train), "missing training witness: " + name)
    _require(any(len(v) > 1 for v in train_mapping.values()), "local mappings do not change in training")
    return None


def _puzzle_key(puzzle):
    return tuple((_key(_array(p["input"])), _key(_array(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded geometry regressions, API checks, and a measured 42-seed sweep."""
    # Hand-defined L stencil, roomy rectangle, diagonal diamond, open ring,
    # tiny mapped cavity and unlisted ring. The expected edits are explicit.
    a = np.zeros((18, 24), dtype=np.int8)
    a[2, 1:3] = (1, 2)
    a[2, 3:5] = 8
    a[3, 3] = 8
    a[6:12, 1:7] = 1
    a[7:11, 2:6] = 0
    a[6:11, 10:15] = 1
    a[7:10, 11:14] = 0
    a[6, 12] = 0  # open: no marks despite a roomy interior
    a[14:17, 1:4] = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    a[14:17, 6:9] = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
    a[6:11, 18:23] = [[0, 0, 1, 0, 0], [0, 1, 0, 1, 0],
                       [1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]
    expected = a.copy()
    expected[7:10, 2:5] = 2
    expected[8, 20] = 2
    _require(np.array_equal(solve(a), expected), "manual L/closure regression failed")
    for k in range(4):
        permutation = np.array([4, 6, 0, 1, 2, 3, 5, 7, 9, 8], dtype=np.int8)
        _require(np.array_equal(solve(permutation[np.rot90(a, k)]), permutation[np.rot90(expected, k)]),
                 "rotation/palette equivariance failed")
    # Empty registration: a five-cell C with an empty middle-left origin.
    b = np.zeros((13, 15), dtype=np.int8)
    b[2, 1:3] = (4, 5)
    b[1:4, 3:5] = [[8, 8], [0, 8], [8, 8]]
    b[6:12, 1:7] = 4
    b[7:11, 2:6] = 0
    expected_b = b.copy()
    expected_b[8:10, 2:5] = 5
    _require(np.array_equal(solve(b), expected_b), "manual hole-registration regression failed")
    # A changes the answer even when no wall changes.
    shorter = a.copy()
    shorter[3, 3] = 0
    _require(not np.array_equal(solve(shorter), expected), "stencil ablation did not change output")
    # B changes the answer even when the mask and its labels are fixed.
    opened = a.copy()
    opened[6, 3] = 0
    _require(np.all(solve(opened)[7:11, 2:6] == 0), "enclosure ablation failed")

    puzzles, features = [], []
    for seed in range(42):
        nt = 2 + seed % 9
        ne = 1 + seed % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                feature = _features(pair["input"])
                feature["split"] = split
                features.append(feature)
                local.append(feature)
        _require(len({f["mapping"] for f in local}) >= 2, "mapping diversity missing")
        _require(len({f["geometries"] for f in local}) >= 2, "geometry diversity missing")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity missing")
    report = {
        "seeds": 42, "pairs": len(features),
        "grid_shapes": len({f["shape"] for f in features}),
        "registered_geometries": len({g for f in features for g in f["geometries"]}),
        "cavity_shapes": len({g for f in features for g in f["cavity_shapes"]}),
        "backgrounds": sorted({f["background"] for f in features}),
        "neutral_colors": sorted({f["neutral"] for f in features}),
        "stencil_counts": sorted({f["stencils"] for f in features}),
        "contour_counts": sorted({f["contours"] for f in features}),
        "directions": sorted({d for f in features for d in f["directions"]}),
        "witness_pairs": {name: sum(bool(f[name]) for f in features)
                          for name in ("sparse_fit", "hole_fit", "no_fit", "partial_fit", "chain",
                                       "open_active", "open_box_fit", "inert_closed", "diagonal", "multi_hole", "border")},
    }
    _require(report["grid_shapes"] >= 30 and report["registered_geometries"] >= 35
             and report["cavity_shapes"] >= 35, "structural variety missing")
    _require(set(report["stencil_counts"]) == {1, 2, 3}, "stencil count variety missing")
    _require(len(report["backgrounds"]) >= 8 and len(report["neutral_colors"]) >= 8, "palette variety missing")
    _require(set(map(tuple, report["directions"])) == set(_FOUR), "direction variety missing")
    for name in ("diagonal", "multi_hole", "border"):
        _require(report["witness_pairs"][name] >= 3, "variation missing: " + name)
    _require(_puzzle_key(generate_puzzle(19, 5, 2)) == _puzzle_key(generate_puzzle(19, 5, 2)), "nondeterminism")
    for nt, ne in ((2, 1), (10, 4), (10, 1), (5, 4)):
        p = generate_puzzle(2026, nt, ne)
        _require(len(p["train"]) == nt and len(p["test"]) == ne, "wrong explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = puzzles[0]
    broken = {s: [{k: _array(v) for k, v in pair.items()} for pair in p[s]] for s in ("train", "test")}
    broken["test"][0]["output"] = broken["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
