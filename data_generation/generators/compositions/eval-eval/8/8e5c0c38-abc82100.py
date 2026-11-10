"""Balanced stencil relief: copies supply counted collective mirror evidence.

No parent code, files, examples, seed, or training state enter solve(). Generation
and validation are bounded and in memory; this module writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np

_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a


def _cells(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        component.add(p)
                        stack.append(p)
        result.append(component)
    return result


def _decode(input_grid):
    """Recover all roles and the unique cardinal registration ray from input."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4:
        raise ValueError("the family has exactly four input colors")
    bg = int(colors[int(counts.argmax())])
    if int(counts.max()) <= a.size // 2:
        raise ValueError("background must be a strict majority")
    points = {int(c): _cells(a, int(c)) for c in colors if c != bg}
    candidates = []
    for neutral, mask in points.items():
        if len(mask) < 3 or len(_components(mask)) != 1:
            continue
        for target, targets in points.items():
            if target == neutral or len(targets) != 1:
                continue
            source = next(c for c in points if c not in (neutral, target))
            src = points[source]
            if len(src) < 4 or len(_components(src)) < 2:
                continue
            tr, tc = next(iter(targets))
            for dr, dc in _DIRECTIONS:
                sl = tr - dr, tc - dc
                q = tr + dr, tc + dc
                if sl not in src or not (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]):
                    continue
                if int(a[q]) not in (bg, neutral):
                    continue
                if min(max(abs(r-q[0]), abs(c-q[1])) for r, c in mask) > 1:
                    continue
                candidates.append({"background": bg, "neutral": neutral, "source": source,
                                   "target": target, "q": q, "source_label": sl,
                                   "target_label": (tr, tc), "direction": (dr, dc),
                                   "mask": mask, "anchors": src - {sl},
                                   "offsets": {(r-q[0], c-q[1]) for r, c in mask}})
    if len(candidates) != 1:
        raise ValueError("instruction reading is missing or ambiguous")
    return candidates[0]


def _expand(anchors, offsets, shape):
    return {(r+dr, c+dc) for r, c in anchors for dr, dc in offsets
            if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]}


def _coverage(anchors, offsets, shape):
    visits = {}
    for r, c in anchors:
        for dr, dc in offsets:
            p = r+dr, c+dc
            if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]:
                visits[p] = visits.get(p, 0) + 1
    return visits


def _reflect(p, orientation, axis2):
    r, c = p
    return (r, axis2-c) if orientation == 0 else (axis2-r, c)


def _balanced(visits, shape, orientation=0):
    """Keep occupied sites with equally many copies at the reflected site."""
    scored = [(sum(visits.get(_reflect(p, orientation, a), 0) == n
                   for p, n in visits.items()), a)
              for a in range(2*shape[1-orientation]-1)]
    best = max(score for score, a in scored)
    winners = [a for score, a in scored if score == best]
    if len(winners) != 1:
        raise ValueError("balanced reflection support has a tied maximum")
    axis = winners[0]
    return axis, {p for p, n in visits.items()
                  if visits.get(_reflect(p, orientation, axis), 0) == n}


def _weighted_differs(visits, shape, orientation, kept):
    """Does every maximum-total-layers axis give another retained subset?"""
    scores = [sum(n for p, n in visits.items()
                  if visits.get(_reflect(p, orientation, axis), 0) == n)
              for axis in range(2*shape[1-orientation]-1)]
    best = max(scores)
    return all(kept != {p for p, n in visits.items()
                        if visits.get(_reflect(p, orientation, axis), 0) == n}
               for axis, score in enumerate(scores) if score == best)


def _details(a):
    rec = _decode(a)
    orientation = int(rec["direction"][0] != 0)
    visits = _coverage(rec["anchors"], rec["offsets"], a.shape)
    rec["coverage"] = visits
    union = set(visits)
    axis, kept = _balanced(visits, a.shape, orientation)
    return rec, orientation, union, axis, kept


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution using convolution scores and reflected array indexing."""
    a = _grid(input_grid)
    rec = _decode(a)
    ink = np.zeros(a.shape, dtype=np.int16)
    # Array accumulation is separate from the construction's coordinate counts.
    for r, c in rec["anchors"]:
        for dr, dc in rec["offsets"]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                ink[rr, cc] += 1
    horizontal = rec["direction"][0] != 0
    work = ink.T if horizontal else ink
    scores = np.zeros(2*work.shape[1]-1, dtype=np.int64)
    # Each positive coverage level is a separate equality channel. A site counts
    # once in the score, whether one copy or several copies cover it.
    for level in range(1, int(work.max())+1):
        for row in (work == level).astype(np.int64):
            scores += np.convolve(row, row)
    axes = np.flatnonzero(scores == scores.max())
    if len(axes) != 1:
        raise ValueError("reflection support has a tied maximum")
    reflected = int(axes[0]) - np.arange(work.shape[1])
    valid = (reflected >= 0) & (reflected < work.shape[1])
    keep = np.zeros_like(work, dtype=bool)
    keep[:, valid] = (work[:, valid] > 0) & (work[:, valid] == work[:, reflected[valid]])
    if horizontal:
        keep = keep.T
    output = np.full(a.shape, rec["background"], dtype=np.int8)
    output[keep] = rec["target"]
    return output


def _alternatives(rec, orientation, union, axis, shape):
    """Counterfactual sets, computed without accessing the intended output."""
    anchors, offsets = rec["anchors"], rec["offsets"]
    length = shape[1-orientation]
    scores = [sum(_reflect(p, orientation, a) in anchors for p in anchors)
              for a in range(2*length-1)]
    winners = [a for a, score in enumerate(scores) if score == max(scores)]
    clean_then_stamp = []
    raw_axis_filter = []
    for a in winners:
        clean = {p for p in anchors if _reflect(p, orientation, a) in anchors}
        clean_then_stamp.append(_expand(clean, offsets, shape))
        raw_axis_filter.append({p for p in union if _reflect(p, orientation, a) in union})
    separate = set()
    for anchor in anchors:
        footprint = _expand({anchor}, offsets, shape)
        separate |= {p for p in footprint if _reflect(p, orientation, axis) in footprint}
    return clean_then_stamp, raw_axis_filter, separate


def _features(pair):
    a = np.asarray(pair["input"])
    rec, orient, union, axis, kept = _details(a)
    alternatives, raw_axes, separate = _alternatives(rec, orient, union, axis, a.shape)
    visits = rec["coverage"]
    clipped = 0
    for anchor in rec["anchors"]:
        footprint = _expand({anchor}, rec["offsets"], a.shape)
        clipped += len(footprint) < len(rec["offsets"])
    binary_same_axis = {p for p in union if _reflect(p, orient, axis) in union}
    binary_scores = [sum(_reflect(p, orient, a2) in union for p in union)
                     for a2 in range(2*a.shape[1-orient]-1)]
    binary_alts = [{p for p in union if _reflect(p, orient, a2) in union}
                   for a2, score in enumerate(binary_scores) if score == max(binary_scores)]
    return {"shape": tuple(a.shape), "background": rec["background"],
            "neutral": rec["neutral"], "source": rec["source"], "target": rec["target"],
            "direction": rec["direction"], "axis2": axis, "parity": axis % 2,
            "anchors": len(rec["anchors"]), "mask_cells": len(rec["offsets"]),
            "mask_extent": (max(r for r,c in rec["offsets"])-min(r for r,c in rec["offsets"])+1,
                            max(c for r,c in rec["offsets"])-min(c for r,c in rec["offsets"])+1),
            "hole": (0, 0) not in rec["offsets"], "union": len(union),
            "kept": len(kept), "discarded": len(union-kept), "clipped_copies": clipped,
            "overlap_cells": sum(n > 1 for n in visits.values()),
            "max_coverage": max(visits.values()),
            "kept_overlap": sum(visits[p] > 1 for p in kept),
            "paired_overlap": sum(visits[p] > 1 and _reflect(p, orient, axis) != p for p in kept),
            "unequal_mirror_cells": len(binary_same_axis-kept),
            "binary_filter_differs": all(kept != alt for alt in binary_alts),
            "layer_weighted_score_differs": _weighted_differs(visits, a.shape, orient, kept),
            "covered_anchors": len(union & rec["anchors"]),
            "cooperative_cells": len(kept-separate),
            "clean_first_differs": all(kept != alt for alt in alternatives),
            "raw_axis_differs": all(kept != alt for alt in raw_axes),
            "without_stencil_differs": bool(kept-rec["anchors"]),
            "geometry": tuple(sorted(rec["offsets"]))}


def _mask(rng, hole):
    """Connected shapes with freely changing row profiles, branches and holes."""
    for _ in range(80):
        h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        qr = int(rng.integers(h))
        style = int(rng.integers(5))
        if style == 0:
            cells = {(r, c) for r in range(h) for c in range(int(rng.integers(2, w+1)))}
        elif style == 1:
            cells = {(r, 1) for r in range(h)}
            cells |= {(int(rng.integers(h)), c) for c in range(w)}
            cells |= {(r, c) for r in range(h) for c in range(w) if rng.random() < .22}
        elif style == 2:
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h-1) or c in (0, w-1)}
        elif style == 3:
            cells = {(qr, 0)}
            for _ in range(int(rng.integers(5, 20))):
                p = sorted(cells)[int(rng.integers(len(cells)))]
                dr, dc = _DIRECTIONS[int(rng.integers(4))]
                q = p[0]+dr, p[1]+dc
                if 0 <= q[0] < h and 0 <= q[1] < w:
                    cells.add(q)
        else:
            cells = {(r, abs(r-qr)) for r in range(h)}
            cells |= {(r, abs(r-qr)+1) for r in range(h) if rng.random() < .65}
        # Sometimes register inside a C-shaped outline, with a corridor for labels.
        qc = 2 if hole and style == 2 and w >= 4 and 0 < qr < h-1 else 0
        q = qr, qc
        cells -= {(qr, qc-1), (qr, qc-2)}
        if hole:
            cells.discard(q)
        else:
            cells.add(q)
        if (len(cells) < 3 or len(cells) > 24 or len(_components(cells)) != 1
                or min(max(abs(r-qr), abs(c-qc)) for r,c in cells) > 1):
            continue
        offsets = {(r-qr, c-qc) for r,c in cells}
        if hole != ((0, 0) not in offsets):
            continue
        return offsets
    # Guaranteed connected, off-center and label-clear for either registration.
    return {(r, 1) for r in (-1, 0, 1)} | ({(0, 0), (1, 2)} if not hole else {(1, 2)})


def _anchors(rng, shape, count, style):
    h, w = shape
    # Position and parity vary independently of the canvas center.
    axis = int(rng.integers(w//2, 3*w//2))
    choices = [(r, c) for r in range(h) for c in range(w)
               if c < axis-c < w]
    rng.shuffle(choices)
    anchors = set()
    desired_pairs = max(1, (count-int(rng.integers(1, 3)))//2)
    preferred_row = int(rng.integers(h))
    if style == 1:
        choices.sort(key=lambda p: abs(p[0]-preferred_row))
    elif style == 2:
        preferred_col = int(rng.integers(w))
        choices.sort(key=lambda p: abs(p[0]-preferred_row)+abs(p[1]-preferred_col))
    for r, c in choices:
        anchors |= {(r, c), (r, axis-c)}
        if len(anchors) >= 2*desired_pairs:
            break
    candidates = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(candidates)
    if style == 2:
        candidates.sort(key=lambda p: min(abs(p[0]-r)+abs(p[1]-c) for r,c in anchors))
    for p in candidates:
        anchors.add(p)
        if len(anchors) == count:
            return anchors
    raise AssertionError("anchor capacity exhausted")


def _instruction_location(rng, shape, offsets, anchors, union):
    relative = offsets | {(0, 0), (0, -1), (0, -2)}
    r0, r1 = min(r for r,c in relative), max(r for r,c in relative)
    c0, c1 = min(c for r,c in relative), max(c for r,c in relative)
    locations = [(r, c) for r in range(-r0, shape[0]-r1)
                 for c in range(-c0, shape[1]-c1)]
    rng.shuffle(locations)
    for qr, qc in locations:
        used = {(qr+r, qc+c) for r,c in relative}
        # No overlap with input anchors or prospective ink; keep label isolated.
        if used & (anchors | union):
            continue
        sl = qr, qc-2
        if any(max(abs(sl[0]-r), abs(sl[1]-c)) <= 1 for r,c in anchors):
            continue
        if any(abs(qr-r)+abs(qc-1-c) == 1 for r,c in anchors):
            continue
        return qr, qc
    return None


def _scoring_demonstration(rng):
    """Constructive fallback for the rarer site-count-versus-layer-count contrast.

    A two-row strip with a registration hole interacts with unequal source
    clusters and bottom clipping. Padding preserves that exact geometry.
    """
    h, w = int(rng.integers(17, 31)), int(rng.integers(15, 31))
    top, left = h-17, int(rng.integers(w-14))
    offsets = {(r, c) for r in (0, 1) for c in range(-3, 3)} - {(0, 0)}
    anchors = {(5, c) for c in range(5, 8)} | {(r, 6) for r in (6, 15, 16)} | {(16, 7)}
    anchors = {(r+top, c+left) for r, c in anchors}
    q = 12+top, 12+left
    bg, neutral, source, target = map(int, rng.choice(10, 4, replace=False))
    a = np.full((h, w), bg, dtype=np.int8)
    for r, c in offsets:
        a[q[0]+r, q[1]+c] = neutral
    a[q[0]-2, q[1]], a[q[0]-1, q[1]] = source, target
    for p in anchors:
        a[p] = source
    visits = _coverage(anchors, offsets, (h, w))
    axis, kept = _balanced(visits, (h, w), 1)
    _require(_weighted_differs(visits, (h, w), 1, kept), "fallback lost scoring contrast")
    out = np.full_like(a, bg)
    for p in kept:
        out[p] = target
    turns = int(rng.integers(4))
    if rng.integers(2):
        a, out = np.fliplr(a), np.fliplr(out)
    a, out = np.rot90(a, turns).copy(), np.rot90(out, turns).copy()
    _require(np.array_equal(solve(a), out), "fallback decoding disagreement")
    return {"input": a, "output": out}


def _make_pair(rng, count, hole, teaching, style):
    for _ in range(1600 if teaching == 2 else 400):
        shape = int(rng.integers(14, 31)), int(rng.integers(14, 31))
        offsets = _mask(rng, hole)
        anchors = _anchors(rng, shape, count, int(rng.choice((1, 2, 2))) if teaching == 2 else style)
        visits = _coverage(anchors, offsets, shape)
        union = set(visits)
        try:
            axis, kept = _balanced(visits, shape)
        except ValueError:
            continue
        if len(kept) < 8 or len(kept) < .4*len(union) or not union-kept:
            continue
        # Symmetry must have off-axis evidence, not just a surviving vertical line.
        if sum(2*c != axis for r,c in kept) < 4:
            continue
        rec = {"anchors": anchors, "offsets": offsets}
        clean, raw, separate = _alternatives(rec, 0, union, axis, shape)
        if teaching and (any(kept == alt for alt in clean) or not kept-separate
                         or not kept-anchors):
            continue
        binary_at_axis = {p for p in union if _reflect(p, 0, axis) in union}
        if teaching and (not binary_at_axis-kept or not any(visits[p] > 1 and 2*p[1] != axis for p in kept)):
            continue
        if teaching:
            scores = [sum(_reflect(p, 0, a) in union for p in union) for a in range(2*shape[1]-1)]
            if any(kept == {p for p in union if _reflect(p, 0, a) in union}
                   for a, score in enumerate(scores) if score == max(scores)):
                continue
        if teaching == 2 and not _weighted_differs(visits, shape, 0, kept):
            continue
        q = _instruction_location(rng, shape, offsets, anchors, union)
        if q is None:
            continue
        bg, neutral, source, target = map(int, rng.choice(10, 4, replace=False))
        a = np.full(shape, bg, dtype=np.int8)
        for r,c in offsets:
            a[q[0]+r, q[1]+c] = neutral
        a[q[0], q[1]-2], a[q[0], q[1]-1] = source, target
        for p in anchors:
            a[p] = source
        out = np.full(shape, bg, dtype=np.int8)
        for p in kept:
            out[p] = target
        # All four label directions are visible rotations/reflections of one rule.
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        if flip:
            a, out = np.fliplr(a), np.fliplr(out)
        a, out = np.rot90(a, turns).copy(), np.rot90(out, turns).copy()
        try:
            decoded = _decode(a)
        except ValueError:
            continue
        if len(decoded["anchors"]) != count:
            continue
        _require(np.array_equal(solve(a), out), "construction and input solver disagree")
        return {"input": a, "output": out}
    if teaching == 2:
        return _scoring_demonstration(rng)
    raise RuntimeError("bounded construction attempts exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name + " is outside the allowed integer range")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    style = int(rng.integers(3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, length in (("train", num_train), ("test", num_test)):
        for index in range(length):
            teaching = split == "train" and index < 2
            count = int(rng.integers(5 if teaching else 3, 8) if split == "train" else rng.integers(8, 13))
            if teaching and index == 1:
                count = 7
            hole = index == 1 if split == "train" and index < 2 else bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, count, hole, index+1 if teaching else 0,
                                  style if rng.random() < .7 else int(rng.integers(3)))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid example counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "dimensions changed")
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            rec, orient, union, axis, kept = _details(a)
            _require(_cells(out, rec["target"]) == kept, "incorrect retained target cells")
            _require(set(map(int, np.unique(out))) == {rec["background"], rec["target"]}, "unexpected output color")
            _require(len(kept) >= 8 and len(kept) >= .4*len(union), "insufficient support")
            _require(bool(union-kept), "symmetry has no effect")
            _require(sum(_reflect(p, orient, axis) != p for p in kept) >= 4, "no off-axis evidence")
            _require(np.array_equal(solve(a), out), "array solver disagreement")
            f = _features(pair)
            features[split].append(f)
    teaching = features["train"][:2]
    _require({f["hole"] for f in teaching} == {False, True}, "missing occupied/hole teaching")
    _require(all(f["clean_first_differs"] and f["without_stencil_differs"] and f["binary_filter_differs"]
                 and f["unequal_mirror_cells"] > 0 and f["paired_overlap"] > 0
                 and f["cooperative_cells"] > 0 for f in teaching), "concepts under-demonstrated")
    _require(any(f["layer_weighted_score_differs"] for f in teaching), "site-count scoring not demonstrated")
    _require(all(3 <= f["anchors"] <= 7 for f in features["train"]), "training count outside schedule")
    _require(all(8 <= f["anchors"] <= 12 for f in features["test"]), "test count shift missing")


def self_test():
    """42 seeds, semantic counterfactuals, equivariance, mutation and count checks.

    Keeps only aggregate feature sets and digests, not a corpus of grids.
    """
    try:
        _balanced({(0, 0): 1, (1, 1): 1}, (2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous support axis was accepted")
    fallback = _features(_scoring_demonstration(np.random.default_rng(9001)))
    _require(fallback["hole"] and fallback["anchors"] == 7 and fallback["paired_overlap"] > 0
             and fallback["unequal_mirror_cells"] > 0 and fallback["layer_weighted_score_differs"]
             and fallback["binary_filter_differs"] and fallback["clean_first_differs"], "fallback teaching invariants")
    signatures, shapes, geometry, directions, parities = set(), set(), set(), set(), set()
    backgrounds, roles, anchor_counts = set(), set(), {"train": set(), "test": set()}
    totals = {"pairs": 0, "overlap": 0, "clipping": 0, "raw_axis": 0, "cooperative": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                digest.update(bytes(a.shape) + a.tobytes() + out.tobytes())
                f = _features(pair)
                shapes.add(f["shape"])
                geometry.add(f["geometry"])
                directions.add(f["direction"])
                parities.add(f["parity"])
                backgrounds.add(f["background"])
                roles.add((f["background"], f["neutral"], f["source"], f["target"]))
                anchor_counts[split].add(f["anchors"])
                totals["pairs"] += 1
                totals["overlap"] += f["overlap_cells"] > 0
                totals["clipping"] += f["clipped_copies"] > 0
                totals["raw_axis"] += f["raw_axis_differs"]
                totals["cooperative"] += f["cooperative_cells"] > 0
                if seed < 3:
                    for turns in range(4):
                        _require(np.array_equal(solve(np.rot90(a, turns)), np.rot90(out, turns)), "rotation equivariance")
                    _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(out)), "reflection equivariance")
        signatures.add(digest.hexdigest())
        if seed < 3:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p,q in zip(puzzle[s],again[s])
                         for k in ("input", "output")), "nondeterministic puzzle")
    _require(len(signatures) == 42 and len(shapes) >= 60 and len(geometry) >= 80, "insufficient structural variation")
    _require(directions == set(_DIRECTIONS) and parities == {0, 1}, "missing orientation/parity")
    _require(len(backgrounds) == 10 and len(roles) >= 100, "insufficient palette role variation")
    _require(anchor_counts == {"train": set(range(3, 8)), "test": set(range(8, 13))}, "missing anchor counts")
    _require(totals["overlap"] >= 30 and totals["clipping"] >= 30
             and totals["raw_axis"] >= 30 and totals["cooperative"] >= 150, "missing interactions")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(8500 + 10*a+b, a, b)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(4,None)):
        try:
            generate_puzzle(9, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(901, 2, 1)
    out = bad["test"][0]["output"]
    rec = _decode(bad["test"][0]["input"])
    out[next(iter(_cells(out, rec["target"])))] = rec["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect symmetric subset accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "registered_geometries": len(geometry), "totals": totals}
