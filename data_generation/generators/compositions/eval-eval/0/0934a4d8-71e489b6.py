"""Mirror damage reports: consensus, asymmetric defects, then a query crop.

Only NumPy and the standard library are used. No generation metadata is needed
by solve. Nothing is written to disk by generation, validation, or self_test.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _orbits(shape):
    h, w = shape
    r, c = np.indices(shape)
    return np.minimum(r, h - 1 - r) * ((w + 1) // 2) + np.minimum(c, w - 1 - c)


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[y:y+h, x:x+w] for y in range(3) for x in range(3)])


def _decode(input_grid):
    """Recover roles, pristine orbits and loss orbits from input pixels only."""
    a = _grid(input_grid)
    if len(np.unique(a)) != 3:
        raise ValueError("There must be two material colors and one query color")
    rectangles = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        top, bottom = int(rr.min()), int(rr.max()) + 1
        left, right = int(cc.min()), int(cc.max()) + 1
        if len(rr) >= 6 and (bottom-top)*(right-left) == len(rr):
            rectangles.append((int(color), (top, bottom, left, right)))
    if len(rectangles) != 1:
        raise ValueError("Query color must be the unique solid rectangular color")
    ink, box = rectangles[0]
    corners = [int(a[r, c]) for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))
               if a[r, c] != ink]
    if not corners or len(set(corners)) != 1:
        raise ValueError("Uncovered corners must identify the field color")
    field = corners[0]
    solid = next(int(c) for c in np.unique(a) if c not in (field, ink))
    ids = _orbits(a.shape)
    n = int(ids.max()) + 1
    visible = a != ink
    counts = np.bincount(ids[visible], minlength=n)
    votes = np.bincount(ids[a == solid], minlength=n)
    if np.any(counts == 0) or np.any(2 * votes == counts):
        raise ValueError("Every orbit needs a surviving, untied vote")
    minority = np.minimum(votes, counts-votes)
    if np.any(minority > 1) or np.any((minority > 0) & (counts < 3)):
        raise ValueError("Each damaged orbit needs at least two intact witnesses")
    intended = 2 * votes > counts
    loss_ids = intended & (votes < counts)
    extra_ids = ~intended & (votes > 0)
    clean = np.where(intended[ids], solid, field).astype(np.int8)
    if any(clean[r, c] != field for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("Intended corners must be field")
    holes = loss_ids[ids]
    observed_holes = visible & (a == field) & intended[ids]
    extras = visible & (a == solid) & ~intended[ids]
    # The corner convention is deliberately protected from corruption.
    if any(observed_holes[r, c] or extras[r, c]
           for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError("Corner witnesses may not be corrupted")
    return {"input": a, "ink": ink, "field": field, "solid": solid,
            "box": box, "ids": ids, "clean": clean, "holes": holes,
            "observed_holes": observed_holes, "extras": extras,
            "loss_ids": loss_ids, "extra_ids": extra_ids, "counts": counts}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the query window of the symmetry-pooled loss report."""
    d = _decode(input_grid)
    out = d["clean"].copy()
    out[d["holes"]] = d["field"]
    out[_dilate(d["holes"]) & ~d["holes"]] = d["ink"]
    t, b, l, r = d["box"]
    return out[t:b, l:r].copy()


def _motif(rng, h, w, style):
    """Variable rectilinear structure, including genuine small cavities."""
    a = np.zeros((h, w), dtype=bool)
    if style == 0:
        # A partition into unequal rectangular patches.
        yy = sorted({0, h, *map(int, rng.integers(1, h, size=3))})
        xx = sorted({0, w, *map(int, rng.integers(1, w, size=3))})
        for y, v in zip(yy, yy[1:]):
            for x, u in zip(xx, xx[1:]):
                a[y:v, x:u] = rng.random() < .55
    elif style == 1:
        for _ in range(int(rng.integers(2, 7))):
            y, v = sorted(map(int, rng.choice(h+1, size=2, replace=False)))
            x, u = sorted(map(int, rng.choice(w+1, size=2, replace=False)))
            a[y:v, x:u] = True
    elif style == 2:
        a[:] = True
        for _ in range(int(rng.integers(1, 5))):
            y, v = sorted(map(int, rng.choice(np.arange(1, h), size=2, replace=False)))
            x, u = sorted(map(int, rng.choice(np.arange(1, w), size=2, replace=False)))
            a[y:v, x:u] = False
        if rng.random() < .5:
            a[h//2, w//2] = True
    else:
        x = int(rng.integers(1, w-2))
        a[:, x:x+int(rng.integers(1, 4))] = True
        for _ in range(int(rng.integers(1, 5))):
            y = int(rng.integers(h-1))
            lo, hi = sorted(map(int, rng.choice(w+1, size=2, replace=False)))
            a[y:y+int(rng.integers(1, 4)), lo:hi] = True
    # These features are not corruption: all their reflected copies agree.
    y, x = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
    a[y:y+3, x:x+3] = True
    a[y+1, x+1] = False
    if min(h, w) >= 9 and rng.random() < .7:
        y, x = h-4, w-4
        a[y:y+3, x:x+3] = False
        a[y+1, x+1] = True
    # Do not prescribe a field border or a majority-color role.
    a[0, 0] = False
    return a


def _window(rng, shape, orientation=None, cross=None, edge=None, avoid=None):
    h, w = shape
    if cross == 1:
        swapped = _window(rng, (w, h), None, 0,
                          None if edge is None else (3, 2, 1, 0)[edge],
                          None if avoid is None else (3, 2, 1, 0)[avoid])
        if swapped is None:
            return None
        t, b, l, r = swapped
        return l, r, t, b
    if cross == 0:
        rh = int(rng.integers(5, min(h-2, 13)+1))
        rw = int(rng.integers(2, min(w//2-1, 10)+1))
        t = int(np.clip(h//2-int(rng.integers(1, max(2, rh//2))), 0, h-rh))
        side = int(rng.integers(2))
        l = int(rng.integers(0, w//2-rw+1))
        if side:
            l = w-rw-l
    else:
        orient = orientation or str(rng.choice(["tall", "wide", "square", "thin"]))
        if orient == "tall":
            rh = int(rng.integers(4, min(h-1, 13)+1))
            rw = int(rng.integers(2, min(rh-1, w//2, 8)+1))
        elif orient == "wide":
            rw = int(rng.integers(4, min(w-1, 13)+1))
            rh = int(rng.integers(2, min(rw-1, h//2, 8)+1))
        elif orient == "square":
            rh = rw = int(rng.integers(3, min(h//2, w//2, 9)+1))
        elif rng.random() < .5:
            rh, rw = 1, int(rng.integers(6, min(w-1, 14)+1))
        else:
            rh, rw = int(rng.integers(6, min(h-1, 14)+1)), 1
        t, l = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
    if edge == 0:
        t = 0
    elif edge == 1:
        l = w-rw
    elif edge == 2:
        t = h-rh
    elif edge == 3:
        l = 0
    box = t, t+rh, l, l+rw
    if avoid is not None and _touches(shape, box)[avoid]:
        return None
    if cross is not None:
        start, end, n = (box[0], box[1], h) if cross == 0 else (box[2], box[3], w)
        if not 2*start < n-1 < 2*(end-1):
            return None
    return box


def _touches(shape, box):
    t, b, l, r = box
    return (t == 0, r == shape[1], b == shape[0], l == 0)


def _construct_output(clean, loss_orbits, ids, box, field, ink):
    """Label renderer, using construction facts rather than solve or _decode."""
    out = clean.copy()
    lost = np.isin(ids, list(loss_orbits))
    for y, x in np.argwhere(lost):
        out[max(0, y-1):y+2, max(0, x-1):x+2] = ink
    out[lost] = field
    t, b, l, r = box
    return out[t:b, l:r].copy()


def _make_pair(rng, settings, index, is_test, seen):
    regime = settings["regime"]
    anchor = index if not is_test and index < 2 else None
    for _ in range(640):
        qh, qw = int(rng.integers(7, 16)), int(rng.integers(7, 16))
        h, w = 2*qh-int(rng.integers(2)), 2*qw-int(rng.integers(2))
        style = settings["style"] if rng.random() < .55 else int(rng.integers(4))
        motif = _motif(rng, qh, qw, style)
        ids = _orbits((h, w))
        base = motif.ravel()[ids]
        colors = tuple(map(int, rng.choice(10, size=3, replace=False))) if settings["vary"] else settings["colors"]
        field, solid, ink = colors
        clean = np.where(base, solid, field).astype(np.int8)
        orientation = None
        cross = None
        edge = None
        avoid = None
        if regime == 0:
            if is_test:
                edge = settings["edge"]
            else:
                avoid = settings["edge"]
        if regime == 1:
            orientation = settings["orientation"]
            if is_test:
                orientation = "wide" if orientation == "tall" else "tall"
        if regime == 3:
            cross = settings["axis"] if anchor == 1 else None
            if is_test:
                cross = 1-settings["axis"]
        # The two demonstrations are broad windows; thin queries are optional.
        if anchor is not None and orientation is None and cross is None:
            orientation = str(rng.choice(["tall", "wide", "square"]))
        box = _window(rng, (h, w), orientation, cross, edge, avoid)
        if box is None:
            continue
        t, b, l, r = box
        if regime == 3 and not is_test:
            crossing = (2*t < h-1 < 2*(b-1), 2*l < w-1 < 2*(r-1))
            if crossing[1-settings["axis"]]:
                continue
        mask = np.zeros((h, w), dtype=bool)
        mask[t:b, l:r] = True
        if mask.sum() > .34*mask.size:
            continue
        sizes = np.bincount(ids.ravel(), minlength=qh*qw)
        covered = np.bincount(ids[mask], minlength=qh*qw)
        visible = sizes-covered
        if np.any(visible == 0):
            continue
        eligible = (sizes == 4) & (visible >= 3)
        eligible[0] = False
        target_ids = np.unique(ids[mask])
        losses = target_ids[eligible[target_ids] & motif.ravel()[target_ids]]
        gains = target_ids[eligible[target_ids] & ~motif.ravel()[target_ids]]
        if len(losses) < 3 or len(gains) < 2:
            continue
        count = 1 if anchor == 0 else 2 if anchor == 1 else int(rng.integers(1, min(6, len(losses))+1))
        if not is_test and regime == 2 and anchor is None:
            count = int(rng.integers(1, 3))
        if is_test and regime == 2:
            count = int(rng.integers(3, min(6, len(losses))+1))
        # Anchor 1 pools two remote losses whose *report* neighborhoods meet.
        chosen = []
        if anchor == 1:
            coords = [(int(i), int(y), int(x)) for y, x in np.argwhere(mask)
                      for i in [ids[y, x]] if i in losses]
            options = [(a[0], b_[0]) for j, a in enumerate(coords) for b_ in coords[j+1:]
                       if a[0] != b_[0] and 1 <= max(abs(a[1]-b_[1]), abs(a[2]-b_[2])) <= 2]
            if not options:
                continue
            chosen = list(options[int(rng.integers(len(options)))])
        else:
            chosen = list(map(int, rng.choice(losses, size=count, replace=False)))
        lost = np.isin(ids, chosen)
        # Teach erasure with an extra whose target location remains outside halos.
        remote_gains = [int(i) for i in gains if np.any(mask & (ids == i) & ~_dilate(lost))]
        if not remote_gains:
            continue
        extras = [int(rng.choice(remote_gains))]
        extra_budget = int(rng.integers(1, min(5, len(gains))+1))
        others = [int(i) for i in gains if i not in extras]
        if extra_budget > 1:
            extras += list(map(int, rng.choice(others, size=extra_budget-1, replace=False)))
        inp = clean.copy()
        inp[mask] = ink
        # Choose a different visible quadrant for the two pooled training losses.
        sources = []
        for j, orbit in enumerate(chosen):
            positions = np.argwhere((ids == orbit) & ~mask)
            if anchor == 1 and j == 1:
                previous = sources[0]
                different = [p for p in positions if (p[0] < h/2, p[1] < w/2)
                             != (previous[0] < h/2, previous[1] < w/2)]
                if different:
                    positions = np.asarray(different)
            pos = positions[int(rng.integers(len(positions)))]
            sources.append(pos)
            inp[tuple(pos)] = field
        for orbit in extras:
            positions = np.argwhere((ids == orbit) & ~mask)
            pos = positions[int(rng.integers(len(positions)))]
            inp[tuple(pos)] = solid
        if _key(inp) in seen:
            continue
        out = _construct_output(clean, chosen, ids, box, field, ink)
        try:
            d = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(d["clean"], clean) or not np.array_equal(d["holes"], lost):
            continue
        # Explicit semantic witnesses, rather than mere color/shape diversity.
        features = _features(d)
        if not features["extra_witness"] or not features["loss_witness"]:
            continue
        if anchor == 0 and not features["true_cavity_preserved"]:
            continue
        if anchor == 1 and not all(features["single_axis_fails"]):
            continue
        if not np.array_equal(solve(inp), out):
            raise AssertionError("Construction and input-only prediction disagree")
        return {"input": inp, "output": out}
    raise RuntimeError(f"Bounded construction exhausted: regime={regime}, pair={index}, test={is_test}")


def _count(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    nq = _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    settings = {"regime": int(rng.integers(4)), "style": int(rng.integers(4)),
                "vary": bool(rng.integers(2)), "colors": tuple(map(int, rng.choice(10, 3, replace=False))),
                "edge": int(rng.integers(4)), "axis": int(rng.integers(2)),
                "orientation": str(rng.choice(["tall", "wide"]))}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            pair = _make_pair(rng, settings, i, split == "test", seen)
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    t, b, l, r = d["box"]
    crop = np.s_[t:b, l:r]
    loss = d["holes"]
    halo = _dilate(loss) & ~loss
    all_changes = (d["loss_ids"] | d["extra_ids"])[d["ids"]]
    symmetric_errors = _dilate(all_changes) & ~all_changes
    predicted = d["clean"].copy()
    predicted[loss] = d["field"]
    predicted[halo] = d["ink"]
    wrong = d["clean"].copy()
    wrong[all_changes] = d["field"]
    wrong[symmetric_errors] = d["ink"]
    counts = np.zeros_like(loss, dtype=np.int16)
    for orbit in np.flatnonzero(d["loss_ids"]):
        counts += _dilate(d["ids"] == orbit)
    single_axis_fails = []
    for axis in (0, 1):
        partial = d["observed_holes"] | np.flip(d["observed_holes"], axis=axis)
        alternative = d["clean"].copy()
        alternative[partial] = d["field"]
        alternative[_dilate(partial) & ~partial] = d["ink"]
        single_axis_fails.append(not np.array_equal(predicted[crop], alternative[crop]))
    solid = d["clean"] == d["solid"]
    padded = np.pad(solid, 1)
    neighbors = sum(padded[y:y+solid.shape[0], x:x+solid.shape[1]].astype(np.int16)
                    for y in range(3) for x in range(3))
    cavity = (~solid) & (neighbors == 8) & ~halo
    return {"shape": d["input"].shape, "window": (b-t, r-l),
            "palette": (d["field"], d["solid"], d["ink"]),
            "losses": int(d["loss_ids"].sum()), "extras": int(d["extra_ids"].sum()),
            "loss_witness": bool(loss[crop].any() and halo[crop].any()),
            "extra_witness": not np.array_equal(predicted[crop], wrong[crop]),
            "overlap": bool(((counts >= 2) & halo)[crop].any()),
            "true_cavity_preserved": bool(cavity[crop].any()),
            "single_axis_fails": tuple(single_axis_fails),
            "crosses": (2*t < loss.shape[0]-1 < 2*(b-1), 2*l < loss.shape[1]-1 < 2*(r-1)),
            "edges": _touches(loss.shape, d["box"]),
            "asymmetric_crop": not np.array_equal(d["clean"][crop], d["clean"][crop][::-1, ::-1])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    training = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen, "Duplicate input"
            seen.add(_key(a))
            d = _decode(a)
            predicted = solve(a)
            assert np.array_equal(predicted, out), "Incorrect report crop"
            assert d["observed_holes"].sum() == d["loss_ids"].sum()
            assert d["extras"].sum() == d["extra_ids"].sum()
            f = _features(d)
            assert f["loss_witness"] and f["extra_witness"]
            t, b, l, r = d["box"]
            pristine = d["clean"][t:b, l:r]
            assert not np.array_equal(pristine, out), "Parent-B ablation must change answer"
            assert np.all(out[d["holes"][t:b, l:r]] == d["field"]), "Loss centers stay empty"
            if split == "train":
                training.append(f)
    assert any(f["losses"] == 1 for f in training), "Teach a single loss"
    assert any(f["losses"] == 2 and f["overlap"] for f in training), "Teach pooled overlapping losses"
    assert any(f["true_cavity_preserved"] for f in training), "Teach legitimate cavities"
    assert all(any(f["single_axis_fails"][axis] for f in training) for axis in (0, 1)), "Teach both reflections"


def self_test():
    """Bounded 42-seed diversity audit plus focused semantic counterexamples."""
    shapes, windows, palettes, clean_structures, signatures = set(), set(), set(), set(), set()
    counts, cross_axes, edges = set(), set(), set()
    pair_count = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                d = _decode(pair["input"])
                f = _features(d)
                shapes.add(f["shape"])
                windows.add(f["window"])
                palettes.add(f["palette"])
                counts.add(f["losses"])
                cross_axes.update(i for i, yes in enumerate(f["crosses"]) if yes)
                edges.update(i for i, yes in enumerate(f["edges"]) if yes)
                # Normalize all symmetries so rotations cannot inflate this count.
                binary = d["clean"] == d["solid"]
                canonical = min((v.shape, v.tobytes()) for k in range(4)
                                for v in (np.rot90(binary, k), np.rot90(binary.T, k)))
                clean_structures.add(canonical)
                digest.update(str(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                pair_count += 1
        signatures.add(digest.digest())
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p
                       for a, b in zip(p[s], q[s]) for k in a)
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(windows) >= 35 and len(palettes) >= 30
    assert len(clean_structures) >= 150 and len(counts) >= 5
    assert cross_axes == {0, 1} and edges == {0, 1, 2, 3}
    # A hand-specified input: two losses in different quadrants, one extra,
    # a true cavity, diagonal halo cells, and a 5x5 report crossing boundaries.
    motif = np.ones((6, 6), dtype=bool)
    motif[0, 0] = False
    motif[2, 2] = False
    motif[4, 4] = False
    ids = _orbits((12, 12))
    base = motif.ravel()[ids]
    a = np.where(base, 6, 2).astype(np.int8)
    a[1:6, 1:6] = 9
    a[8, 2] = 2  # projects to (3,2)
    a[2, 7] = 2  # projects to (2,4)
    a[7, 7] = 6  # extra projects to legitimate field (4,4)
    expected = np.array([[6, 6, 9, 9, 9], [9, 9, 9, 2, 9],
                         [9, 2, 9, 9, 9], [9, 9, 9, 2, 6],
                         [6, 6, 6, 6, 6]], dtype=np.int8)
    assert np.array_equal(solve(a), expected), "Hand-derived report regression"
    # Ignoring either reflection direction loses one of the two remote defects.
    for axis in (0, 1):
        local = np.flip(a, axis=axis)[1:6, 1:6]
        assert not np.array_equal(local, expected)
    for k in range(4):
        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(expected, k))
        assert np.array_equal(solve(np.rot90(a.T, k)), np.rot90(expected.T, k))
    perm = np.array([7, 0, 5, 8, 1, 3, 2, 4, 9, 6], dtype=np.int8)
    assert np.array_equal(solve(perm[a]), perm[expected])
    tied = a.copy()
    # An entirely visible four-cell orbit with two votes for either value.
    tied[1, 0] = tied[10, 0] = 2
    try:
        _decode(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied orbit accepted")
    wrong = generate_puzzle(0, 2, 1)
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "window_shapes": len(windows),
            "palettes": len(palettes), "clean_structures_modulo_D4": len(clean_structures),
            "loss_counts": sorted(counts)}
