"""Folded stencil deficits: cancel symmetric anchors, stamp the missing mates.

Only standard Python and NumPy are used. solve reads every role from one input.
"""
from __future__ import annotations

import numpy as np

_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


class _Invalid(ValueError):
    pass


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise _Invalid("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
                        found.add(q)
        result.append(found)
    return result


def _positions(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise _Invalid("Background must be the unique most frequent input color")
    return int(counts.argmax())


def _instructions(a, bg, neutral):
    """Every component of a proposed mask color needs one directed label ray."""
    masks = _components(_positions(a, neutral))
    if not masks or neutral == bg:
        raise _Invalid("No stencil masks")
    near = {}
    for i, mask in enumerate(masks):
        for q in _halo(mask):
            near.setdefault(q, []).append(i)
    options = [[] for _ in masks]
    for s in sorted(tuple(map(int, p)) for p in np.argwhere((a != bg) & (a != neutral))):
        source = int(a[s])
        if source == neutral:
            continue
        for dr, dc in _DIRECTIONS:
            t = (s[0] + dr, s[1] + dc)
            q = (s[0] + 2 * dr, s[1] + 2 * dc)
            if not _inside(q, a.shape):
                continue
            target = int(a[t])
            if target in (bg, neutral, source) or int(a[q]) not in (bg, neutral):
                continue
            for i in near.get(q, ()):
                options[i].append({"mask": masks[i], "q": q, "source": source,
                                   "target": target, "labels": {s, t},
                                   "direction": (dr, dc),
                                   "offsets": tuple(sorted((r-q[0], c-q[1]) for r, c in masks[i]))})
    if any(len(x) != 1 for x in options):
        raise _Invalid("A mask has missing or ambiguous labels")
    records = [x[0] for x in options]
    labels = [p for rec in records for p in rec["labels"]]
    if len(labels) != len(set(labels)) or len({r["source"] for r in records}) != len(records):
        raise _Invalid("Labels are shared or a source has multiple stencils")
    return records


def _guide(cells):
    if len(cells) == 1:
        return ("P", next(iter(cells)))
    rows, cols = {r for r, c in cells}, {c for r, c in cells}
    if len(rows) == 1 and len(cells) == max(cols) - min(cols) + 1:
        return ("H", (next(iter(rows)), min(cols)))
    if len(cols) == 1 and len(cells) == max(rows) - min(rows) + 1:
        return ("V", (min(rows), next(iter(cols))))
    raise _Invalid("The sole guide must be a straight segment or singleton")


def _image(p, guide):
    kind, (r, c) = guide
    return (2*r-p[0] if kind in ("H", "P") else p[0],
            2*c-p[1] if kind in ("V", "P") else p[1])


def _linear(p, guide):
    kind = guide[0]
    return (-p[0] if kind in ("H", "P") else p[0],
            -p[1] if kind in ("V", "P") else p[1])


def _decode(a):
    bg = _background(a)
    colors = set(map(int, np.unique(a)))
    candidates = []
    for neutral in sorted(colors - {bg}):
        try:
            records = _instructions(a, bg, neutral)
            named = {x for rec in records for x in (rec["source"], rec["target"])}
            leftovers = colors - {bg, neutral} - named
            if len(leftovers) != 1:
                continue
            marker = next(iter(leftovers))
            guide_cells = _positions(a, marker)
            guide = _guide(guide_cells)
            legend = set().union(*(rec["mask"] | rec["labels"] for rec in records))
            anchors = {rec["source"]: _positions(a, rec["source"]) - legend
                       for rec in records}
            if any(not x for x in anchors.values()):
                continue
            if any(not _inside(_image(p, guide), a.shape) or _image(p, guide) == p
                   for cells in anchors.values() for p in cells):
                continue
            decoded = {"grid": a, "background": bg, "neutral": neutral, "marker": marker,
                       "guide": guide, "guide_cells": guide_cells, "records": records,
                       "legend": legend, "anchors": anchors}
            _paint(decoded)
            candidates.append(decoded)
        except _Invalid:
            continue
    if len(candidates) != 1:
        raise _Invalid("Input does not identify exactly one stencil/guide reading")
    return candidates[0]


def _paint(d, *, cancel=True, typed=True, orient=True, move=True, stencil=True, strict=True):
    paint, visits, orphans, matched, clipped = {}, {}, [], [], set()
    a, guide = d["grid"], d["guide"]
    all_anchors = set().union(*d["anchors"].values())
    inert = set(np.ndindex(a.shape)) - d["legend"] - d["guide_cells"] - all_anchors
    for rec in d["records"]:
        for p in sorted(d["anchors"][rec["source"]]):
            q = _image(p, guide)
            if cancel and q in (d["anchors"][rec["source"]] if typed else all_anchors):
                matched.append((rec["source"], p))
                continue
            orphans.append((rec["source"], p))
            center = q if move else p
            offsets = rec["offsets"] if stencil else ((0, 0),)
            for offset in offsets:
                dr, dc = _linear(offset, guide) if orient else offset
                dest = (center[0] + dr, center[1] + dc)
                if not _inside(dest, a.shape):
                    if dest[0] < 0: clipped.add("top")
                    if dest[0] >= a.shape[0]: clipped.add("bottom")
                    if dest[1] < 0: clipped.add("left")
                    if dest[1] >= a.shape[1]: clipped.add("right")
                    continue
                if strict:
                    if dest in d["legend"] or dest in d["guide_cells"]:
                        raise _Invalid("A stamp covers the legend or guide")
                    if dest in inert and int(a[dest]) != d["background"]:
                        raise _Invalid("A stamp covers inert content")
                    if dest in paint and paint[dest] != rec["target"]:
                        raise _Invalid("Conflicting stamp colors")
                paint[dest] = rec["target"]
                visits[dest] = visits.get(dest, 0) + 1
    if strict and not orphans:
        raise _Invalid("At least one unmatched anchor must produce a stamp")
    return paint, {"orphans": orphans, "matched": matched, "clipped": clipped,
                   "overlap": sum(x > 1 for x in visits.values()),
                   "covered_anchors": len(set(paint) & all_anchors)}


def _render(d, **options):
    out = d["grid"].copy()
    for p in d["legend"] | set().union(*d["anchors"].values()):
        out[p] = d["background"]
    paint, _ = _paint(d, **options)
    for p, color in paint.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode this input alone; compare original anchors, then paint once."""
    return _render(_decode(_grid(input_grid)))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _turn(cells, direction):
    dr, dc = direction
    # Right-facing coordinates mapped to any of the four label directions.
    return {(dc*r + dr*c, -dr*r + dc*c) for r, c in cells}


def _shape(rng, hole, guide, teaching=False):
    for _ in range(40):
        if hole:
            depth, width = int(rng.integers(2, 4)), int(rng.integers(1, 4))
            cells = {(r, width) for r in range(-depth, depth+1)}
            cells |= {(r, c) for r in (-depth, depth) for c in range(-1, width+1)}
            # This spur touches the registration hole and joins the lower rim.
            cells |= {(r, 0) for r in range(1, depth)}
            if rng.random() < .5:
                cells.add((-depth+1, width-1))
        else:
            family = _choose(rng, ("growth", "growth", "twig", "elbow", "rim", "bar"))
            h, w = int(rng.integers(2, 5)), int(rng.integers(2, 6))
            if family == "bar" and not teaching:
                cells = {(0, c) for c in range(w)}
            elif family == "elbow":
                cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
            elif family == "rim":
                cells = {(r, c) for r in range(h) for c in range(w)
                         if r in (0, h-1) or c in (0, w-1)}
                cells.discard((h-1, w-1))
            elif family == "twig":
                r = 0
                cells = {(0, 0)}
                for c in range(1, w):
                    r += int(rng.choice([-1, 0, 1]))
                    cells.add((r, c))
                cells.add((1, 0))
            else:
                cells = {(0, 0)}
                desired = int(rng.integers(3, min(16, h*w)+1))
                for _ in range(100):
                    if len(cells) >= desired:
                        break
                    frontier = sorted({(r+dr, c+dc) for r, c in cells
                                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                       if -h+1 <= r+dr < h and 0 <= c+dc < w} - cells)
                    cells.add(_choose(rng, frontier))
        if rng.random() < .5:
            cells = {(-r, c) for r, c in cells}
        direction = _choose(rng, _DIRECTIONS)
        cells = _turn(cells, direction)
        offcenter = sum(r for r, c in cells) != 0 or sum(c for r, c in cells) != 0
        if not teaching or ({_linear(p, guide) for p in cells} != cells and offcenter):
            return tuple(sorted(cells)), direction
    raise RuntimeError("Cannot sample an asymmetric stencil")


def _mapping(rng, pool, count, index, train):
    a, b, c, e = pool[:4]
    if train and index == 0:
        pairs = [(a, b)]
        for source in pool[2:count+1]:
            pairs.append((source, b))
        return pairs
    if train and index == 1:
        return [(a, c), (c, e)] + ([(b, e)] if count == 3 else [])
    shuffled = [int(x) for x in rng.permutation(pool)]
    mode = _choose(rng, ("shared", "chain", "cycle", "independent"))
    if mode == "shared":
        return [(shuffled[i], shuffled[count]) for i in range(count)]
    if mode == "cycle" and count > 1:
        return [(shuffled[i], shuffled[(i+1) % count]) for i in range(count)]
    if mode == "chain":
        return [(shuffled[i], shuffled[i+1]) for i in range(count)]
    return [(shuffled[i], shuffled[count+i % (len(pool)-count)]) for i in range(count)]


def _make_pair(rng, params, kind, index, train, known):
    count = int(_choose(rng, (1, 1, 2, 2, 3)))
    if train and index == 1:
        count = max(2, count)
    mapping = _mapping(rng, params["pool"], count, index, train)
    teaching = train and index < 2
    for attempt in range(36):
        low = 16 if count == 1 else 20
        h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        if attempt >= 24:
            h = w = 30
        bg = int(_choose(rng, params["backgrounds"]))
        a = np.full((h, w), bg, dtype=np.int8)
        cr, cc = int(rng.integers(h//2-2, h//2+3)), int(rng.integers(w//2-2, w//2+3))
        guide = (kind, (cr, cc))
        length = int(rng.integers(2, 9))
        if kind == "P":
            marker_cells = {(cr, cc)}
        elif kind == "H":
            start = int(rng.integers(0, w-length+1))
            marker_cells = {(cr, c) for c in range(start, start+length)}
        else:
            start = int(rng.integers(0, h-length+1))
            marker_cells = {(r, cc) for r in range(start, start+length)}
        for p in marker_cells:
            a[p] = params["marker"]
        records, reserved, neutral_halo = [], _halo(marker_cells), set()
        try:
            for i, (source, target) in enumerate(mapping):
                hole = bool((train and index == 1 and i == 0) or
                            (not (train and index == 0 and i == 0) and rng.random() < .35))
                offsets, direction = _shape(rng, hole, guide, teaching or (train and index == 2))
                if not train and i == 0 and offsets in known:
                    raise _Invalid("Test must read a new registered stencil")
                dr, dc = direction
                relative = set(offsets) | {(0, 0), (-dr, -dc), (-2*dr, -2*dc)}
                r0, r1 = min(r for r, c in relative), max(r for r, c in relative)
                c0, c1 = min(c for r, c in relative), max(c for r, c in relative)
                places = [(r, c) for r in range(-r0, h-r1) for c in range(-c0, w-c1)]
                rng.shuffle(places)
                found = False
                for qr, qc in places:
                    mask = {(qr+r, qc+c) for r, c in offsets}
                    labels = {(qr-dr, qc-dc), (qr-2*dr, qc-2*dc)}
                    footprint = mask | labels | {(qr, qc)}
                    if footprint & reserved or mask & neutral_halo:
                        continue
                    b = a.copy()
                    for p in mask:
                        b[p] = params["neutral"]
                    b[qr-2*dr, qc-2*dc], b[qr-dr, qc-dc] = source, target
                    try:
                        parsed = _instructions(b, bg, params["neutral"])
                    except _Invalid:
                        continue
                    if len(parsed) != i+1:
                        continue
                    rec = next((x for x in parsed if x["source"] == source), None)
                    if rec is None or rec["q"] != (qr, qc) or rec["offsets"] != offsets:
                        continue
                    a = b
                    records.append(rec)
                    reserved |= footprint
                    neutral_halo |= _halo(mask)
                    found = True
                    break
                if not found:
                    raise _Invalid("Instruction packing failed")
            legend = set().union(*(r["mask"] | r["labels"] for r in records))
            anchors = {s: set() for s, t in mapping}
            signature = {(r["source"], r["target"], r["q"], r["offsets"]) for r in records}
            d = {"grid": a, "background": bg, "neutral": params["neutral"],
                 "marker": params["marker"], "guide": guide, "guide_cells": marker_cells,
                 "records": records, "legend": legend, "anchors": anchors}
            cells = [p for p in np.ndindex(a.shape) if p not in reserved
                     and _inside(_image(p, guide), a.shape) and _image(p, guide) != p]
            rng.shuffle(cells)

            def accept(additions, complete=None, protect_match=True):
                if len({p for s, p in additions}) != len(additions):
                    return False
                for source, p in additions:
                    if p not in cells or int(a[p]) != bg:
                        return False
                old = {s: set(x) for s, x in anchors.items()}
                for source, p in additions:
                    anchors[source].add(p)
                    a[p] = source
                try:
                    paint, features = _paint(d)
                    if complete is not None:
                        s, p, wanted = complete
                        rec = next(r for r in records if r["source"] == s)
                        q = _image(p, guide)
                        size = sum(_inside((q[0]+dr, q[1]+dc), a.shape)
                                   for dr, dc in map(lambda x: _linear(x, guide), rec["offsets"]))
                        if (wanted == "full" and size != len(rec["offsets"])) or (wanted == "clip" and not 0 < size < len(rec["offsets"])):
                            raise _Invalid("Requested stamp visibility is absent")
                    if protect_match and any(p in paint for s, p in features["matched"]):
                        raise _Invalid("A matched-pair witness is covered")
                    parsed = _instructions(a, bg, params["neutral"])
                    if {(r["source"], r["target"], r["q"], r["offsets"]) for r in parsed} != signature:
                        raise _Invalid("Anchors change instruction parsing")
                    return True
                except _Invalid:
                    for s in anchors:
                        anchors[s] = old[s]
                    for source, p in additions:
                        a[p] = bg
                    return False

            # Orphans first: each instruction has an unobscured complete stamp.
            for source, target in mapping:
                for p in cells:
                    if accept([(source, p)], complete=(source, p, "full")):
                        break
                else:
                    raise _Invalid("No full stamp fits")
            # One explicit clipping witness is taught, not reserved for tests.
            if train and index == 0:
                source = mapping[0][0]
                for p in cells:
                    if _image(p, guide) not in anchors[source] and accept([(source, p)], complete=(source, p, "clip")):
                        break
                else:
                    raise _Invalid("No clipped teaching stamp fits")
            # Completed same-colored orbit: it vanishes without stamping.
            source = mapping[0][0]
            for p in cells:
                q = _image(p, guide)
                if accept([(source, p), (source, q)]):
                    break
            else:
                raise _Invalid("No symmetric pair fits")
            # Different source colors at reflected sites do not cancel.
            if count > 1 and (teaching or rng.random() < .6):
                s, t = mapping[0][0], mapping[1][0]
                for p in cells:
                    if accept([(s, p), (t, _image(p, guide))]):
                        break
                else:
                    raise _Invalid("No different-colored orbit fits")
            # Variable runs, clusters and scatter, including compatible overlap.
            for source, target in mapping:
                desired = int(rng.integers(1, 6))
                for j in range(desired):
                    preferred = [(r+dr, c+dc) for r, c in sorted(anchors[source])
                                 for dr, dc in _DIRECTIONS]
                    rng.shuffle(preferred)
                    choices = preferred + cells if rng.random() < (.8 if not train else .45) else cells
                    for p in choices:
                        if _image(p, guide) not in anchors[source] and accept([(source, p)]):
                            break
            paint, _ = _paint(d)
            inactive = sorted({r["target"] for r in records} - set(anchors))
            for color in inactive:
                for p in cells[:int(rng.integers(5, 20))]:
                    if int(a[p]) != bg or p in paint or p in legend or p in marker_cells:
                        continue
                    a[p] = color
                    try:
                        if {(r["source"], r["target"], r["q"], r["offsets"]) for r in _instructions(a, bg, params["neutral"])} != signature:
                            raise _Invalid("Inert content changes parsing")
                    except _Invalid:
                        a[p] = bg
            if _background(a) != bg:
                raise _Invalid("Background ceases to be dominant")
            decoded = _decode(a)
            if decoded["marker"] != params["marker"] or decoded["neutral"] != params["neutral"]:
                raise _Invalid("Visible roles differ from construction")
            output = _render(d)
            if not np.array_equal(_render(decoded), output):
                raise _Invalid("Input interpretation differs from construction")
            if train and index < 3:
                alternatives = [{"cancel": False}, {"orient": False}, {"move": False}, {"stencil": False}]
                if count > 1:
                    alternatives.append({"typed": False})
                for options in alternatives:
                    if np.array_equal(output, _render(decoded, strict=False, **options)):
                        raise _Invalid("Teaching example hides a required mechanism")
            return {"input": a, "output": output}
        except _Invalid:
            continue
    raise RuntimeError("Bounded scene construction exhausted")


def _count(n, low, high, name):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= int(n) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(n)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.permutation(10)]
    params = {"marker": palette[0], "neutral": palette[1],
              "backgrounds": palette[2:4] if rng.random() < .45 else palette[2:3],
              "pool": palette[4:]}
    subsets = [("H",), ("V",), ("P",), ("H", "V"), ("H", "P"), ("V", "P")]
    if num_train >= 3:
        subsets += [("H", "V", "P")]
    kinds = list(_choose(rng, subsets))
    rng.shuffle(kinds)
    train_kinds = kinds + [str(_choose(rng, kinds)) for _ in range(num_train-len(kinds))]
    puzzle, seen, known = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = train_kinds[i] if split == "train" else str(_choose(rng, kinds))
            for _ in range(12):
                pair = _make_pair(rng, params, kind, i, split == "train", known)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("Cannot obtain distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                known.update(r["offsets"] for r in _decode(pair["input"])["records"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, roles, trained, known, mappings = set(), set(), set(), set(), {}
    oriented_kinds = set()
    witnesses = {"cancel": False, "typed": False, "orient": False, "move": False, "stencil": False}
    hole_taught = clipping_taught = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            d = _decode(a)
            roles.add((d["marker"], d["neutral"]))
            paint, features = _paint(d)
            assert features["orphans"] and features["matched"]
            assert np.array_equal(_render(d), out)
            assert not np.array_equal(a, out)
            assert all(out[p] == a[p] for p in d["guide_cells"])
            anchors = set().union(*d["anchors"].values())
            for p in d["legend"] | anchors:
                assert out[p] == paint.get(p, d["background"])
            assert all(out[p] == a[p] for p in np.ndindex(a.shape)
                       if p not in paint and p not in d["legend"] and p not in anchors)
            holes = any((0, 0) not in r["offsets"] for r in d["records"])
            if split == "train":
                trained.add(d["guide"][0])
                if not np.array_equal(out, _render(d, strict=False, orient=False)):
                    oriented_kinds.add(d["guide"][0])
                known.update(r["offsets"] for r in d["records"])
                for r in d["records"]:
                    mappings.setdefault(r["source"], set()).add(r["target"])
                hole_taught |= holes
                clipping_taught |= bool(features["clipped"])
                for name in witnesses:
                    witnesses[name] |= not np.array_equal(out, _render(d, strict=False, **{name: False}))
            else:
                assert d["guide"][0] in trained
                assert not holes or hole_taught
                assert not features["clipped"] or clipping_taught
                assert any(r["offsets"] not in known for r in d["records"])
    assert len(roles) == 1, "Shared guide and neutral roles changed within a puzzle"
    assert trained <= oriented_kinds, "An operation lacks an orientation witness"
    assert all(witnesses.values()), "Training fails to distinguish a shortcut"
    assert any(len(v) > 1 for v in mappings.values()), "Training must teach local color assignments"
    assert hole_taught and clipping_taught


def _puzzle_key(puzzle):
    return tuple((np.asarray(p[k]).shape, np.asarray(p[k]).tobytes())
                 for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))


def _semantic_fixtures():
    # These outputs are written cell by cell, without _image/_linear/_paint.
    a = np.zeros((18, 20), dtype=np.int8)
    a[8, 9:12] = 5
    a[1, 1:3] = (1, 2)
    a[1, 3:5], a[2, 3] = 8, 8
    for p in ((4, 14), (6, 7), (10, 7), (0, 19), (14, 10)):
        a[p] = 1
    a[17, 19] = 2
    out = np.zeros_like(a)
    out[8, 9:12], out[17, 19] = 5, 2
    for p in ((12, 14), (12, 15), (11, 14), (16, 19), (15, 19),
              (2, 10), (2, 11), (1, 10)):
        out[p] = 2
    fixtures = [(a, out)]

    a = np.zeros((18, 20), dtype=np.int8)
    a[8, 10] = 5
    a[3, 1:3] = (1, 2)
    for p in ((1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
              (4, 4), (5, 4), (5, 3), (5, 2), (4, 3)):
        a[p] = 8
    for p in ((7, 13), (6, 16), (10, 4), (16, 1)):
        a[p] = 1
    a[17, 0] = 2
    out = np.zeros_like(a)
    out[8, 10], out[17, 0] = 5, 2
    for p in ((11, 8), (11, 7), (11, 6), (10, 6), (9, 6),
              (8, 6), (7, 6), (7, 7), (7, 8), (8, 7),
              (2, 19), (2, 18), (1, 18), (0, 18)):
        out[p] = 2
    fixtures.append((a, out))

    a = np.zeros((16, 19), dtype=np.int8)
    a[4:7, 9] = 5
    a[1, 1:3], a[13, 1:3] = (1, 2), (2, 3)
    for p in ((1, 3), (2, 3), (2, 4), (13, 3), (13, 4)):
        a[p] = 8
    for p in ((8, 5), (10, 7), (10, 11), (4, 2)):
        a[p] = 1
    a[8, 13], a[11, 17], a[0, 18] = 2, 2, 3
    out = np.zeros_like(a)
    out[4:7, 9], out[0, 18] = 5, 3
    for p in ((8, 13), (9, 13), (9, 12), (4, 16), (5, 16), (5, 15)):
        out[p] = 2
    for p in ((8, 5), (8, 4), (11, 1), (11, 0)):
        out[p] = 3
    fixtures.append((a, out))
    return fixtures


def self_test():
    """Bounded semantic invariants and structural variation across 42 seeds."""
    for a, expected in _semantic_fixtures():
        assert np.array_equal(solve(a), expected)
        # Input-only parsing must follow geometry and roles under these changes.
        for k in range(4):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(expected, k))
        assert np.array_equal(solve(a.T), expected.T)
        permutation = np.random.default_rng(91).permutation(10)
        assert np.array_equal(solve(permutation[a]), permutation[expected])
    puzzles, shapes, masks, backgrounds, markers, neutral_colors = [], set(), set(), set(), set(), set()
    kinds, directions, counts, overlaps, covered, hole_count = set(), set(), set(), 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                d = _decode(a)
                assert np.array_equal(solve(a), pair["output"])
                shapes.add(a.shape)
                backgrounds.add(d["background"])
                markers.add(d["marker"])
                neutral_colors.add(d["neutral"])
                kinds.add(d["guide"][0])
                counts.add(len(d["records"]))
                _, features = _paint(d)
                overlaps += features["overlap"] > 0
                covered += features["covered_anchors"] > 0
                for r in d["records"]:
                    masks.add(r["offsets"])
                    directions.add(r["direction"])
                    hole_count += (0, 0) not in r["offsets"]
                # Affine consistency: reflecting an entire registered footprint
                # agrees with moving and orienting it around the reflected anchor.
                for r in d["records"]:
                    for anchor in d["anchors"][r["source"]]:
                        center = _image(anchor, d["guide"])
                        direct = {_image((anchor[0]+dr, anchor[1]+dc), d["guide"]) for dr, dc in r["offsets"]}
                        offset = {(center[0]+dr, center[1]+dc) for dr, dc in
                                  map(lambda x: _linear(x, d["guide"]), r["offsets"])}
                        assert direct == offset
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 60 and len(masks) >= 100
    assert len(backgrounds) >= 8 and len(markers) >= 8 and len(neutral_colors) >= 8
    assert kinds == {"H", "V", "P"} and directions == set(_DIRECTIONS) and counts == {1, 2, 3}
    assert overlaps >= 20 and covered >= 20 and hole_count >= 40
    assert _puzzle_key(generate_puzzle(17, 2, 1)) == _puzzle_key(generate_puzzle(17, 2, 1))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(700+nt*10+ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal explicit counts")
