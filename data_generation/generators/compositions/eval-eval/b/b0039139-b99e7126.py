"""Count-controlled, overlapping completion of a tile's own ink stencil.

No files, hidden puzzle state, or source examples are used. Colors have visible
roles; the sole convention shared across puzzles is the transformation itself.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, group = [start], []
        for r, c in todo:
            group.append((int(r), int(c)))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(group)
    return result


def _crop(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "empty stencil")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] + 1 == i:
            result[-1] = result[-1][0], i
        else:
            result.append((i, i))
    return result


def _sweep(mask, count, axis):
    """Construction by Boolean superposition of translated arrays."""
    shape = list(mask.shape)
    shape[axis] += count - 1
    footprint = np.zeros(shape, dtype=bool)
    for step in range(count):
        if axis == 1:
            footprint[:, step:step + mask.shape[1]] |= mask
        else:
            footprint[step:step + mask.shape[0], :] |= mask
    return footprint


def _origins(points, shape, rows, cols, supplied):
    """All fitting translations, with no score or tie-breaking heuristic."""
    h, w = shape
    return [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)
            if all((y - r, x - c) in points for y, x in supplied)]


def _decode(input_grid):
    """Input-only parsing; compute the sweep with coordinate sets."""
    a = _grid(input_grid)
    divider = int(a[0, 0])
    _require(all(int(a[p]) == divider for p in
                 ((0, 0), (0, -1), (-1, 0), (-1, -1))), "damaged frame corners")
    border = np.zeros(a.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    marks = np.argwhere(border & (a != divider))
    _require(len(marks) == 1, "one frame marker is required")
    marker = tuple(map(int, marks[0]))
    y, x = marker
    _require((x == a.shape[1] - 1 and 0 < y < a.shape[0] - 1) or
             (y == a.shape[0] - 1 and 0 < x < a.shape[1] - 1), "invalid marker side")
    axis = 1 if x == a.shape[1] - 1 else 0
    clean = a.copy()
    clean[marker] = divider
    rr = _runs(np.flatnonzero(np.all(clean == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(clean == divider, axis=0)))
    _require(len(rr) >= 3 and len(cc) >= 3, "missing lattice")
    _require(rr[0][0] == cc[0][0] == 0 and rr[-1][1] == a.shape[0] - 1 and
             cc[-1][1] == a.shape[1] - 1, "incomplete frame")
    thicknesses = {b - b0 + 1 for b0, b in rr + cc}
    _require(len(thicknesses) == 1, "irregular divider thickness")
    t = thicknesses.pop()
    hs = {rr[i + 1][0] - rr[i][1] - 1 for i in range(len(rr) - 1)}
    ws = {cc[i + 1][0] - cc[i][1] - 1 for i in range(len(cc) - 1)}
    _require(len(hs) == len(ws) == 1, "irregular tiles")
    h, w = hs.pop(), ws.pop()
    _require(h >= 2 and w >= 2 and t >= 1, "tiles too small")
    rows, cols = len(rr) - 1, len(cc) - 1
    records = {}
    for r in range(rows):
        for c in range(cols):
            tile = a[rr[r][1] + 1:rr[r + 1][0], cc[c][1] + 1:cc[c + 1][0]]
            _require(divider not in tile, "divider inside a tile")
            records.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    _require(len(records) == 2, "exactly two interior tile types are required")
    ordered = sorted(records.values(), key=lambda v: len(v[1]), reverse=True)
    ordinary, op = ordered[0]
    glyph, supplied = ordered[1]
    _require(len(op) > len(supplied) >= 2, "ordinary tile must have strict majority")
    oc, gc = set(map(int, np.unique(ordinary))), set(map(int, np.unique(glyph)))
    _require(len(oc) == len(gc) == 2 and len(oc & gc) == 1, "ambiguous ink roles")
    background = next(iter(oc & gc))
    counter = next(iter(oc - gc))
    ink = next(iter(gc - oc))
    _require(int(a[marker]) == ink, "frame marker must use glyph ink")
    cm = ordinary == counter
    count = len(_components(cm))
    _require(count == len(_components(cm, True)), "counter has diagonal contact")
    mask = glyph == ink
    tight = _crop(mask)
    # Set addition is separate from the array superposition used in generation.
    points = {(int(r) + (s if axis == 0 else 0), int(c) + (s if axis == 1 else 0))
              for r, c in np.argwhere(tight) for s in range(count)}
    shape = (tight.shape[0] + (count - 1 if axis == 0 else 0),
             tight.shape[1] + (count - 1 if axis == 1 else 0))
    origins = _origins(points, shape, rows, cols, supplied)
    _require(len(origins) == 1, "swept stencil must have exactly one registration")
    origin = origins[0]
    active = {(origin[0] + r, origin[1] + c) for r, c in points}
    _require(set(supplied) < active, "a strict partial stencil is required")
    return {"array": a, "divider": divider, "marker": marker, "axis": axis,
            "geometry": (h, w, t, rows, cols), "rr": rr, "cc": cc,
            "ordinary": ordinary, "glyph": glyph, "background": background,
            "counter": counter, "ink": ink, "count": count, "counter_area": int(cm.sum()),
            "mask": mask, "tight": tight, "points": points, "shape": shape,
            "origin": origin, "supplied": set(supplied), "active": active}


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    result = d["array"].copy()
    for r, c in d["active"] - d["supplied"]:
        result[d["rr"][r][1] + 1:d["rr"][r + 1][0],
               d["cc"][c][1] + 1:d["cc"][c + 1][0]] = d["glyph"]
    return result


def _counter_mask(rng, shape, count, area=None):
    """Separated seeds plus safe growth; bounded, including exact-area witnesses."""
    h, w = shape
    sites = [(r, c) for r in range(0, h, 2) for c in range(0, w, 2)]
    _require(count <= len(sites), "counter does not fit")
    for _ in range(80):
        labels = np.zeros(shape, dtype=np.int8)
        chosen = rng.choice(len(sites), count, replace=False)
        for label, k in enumerate(chosen, 1):
            labels[sites[int(k)]] = label
        target = area if area is not None else int(rng.integers(count, min(h * w - 1, count + 6) + 1))
        for _ in range(target - count):
            growths = []
            for r in range(h):
                for c in range(w):
                    if labels[r, c]:
                        continue
                    near = set(int(labels[y, x]) for y in range(max(0, r - 1), min(h, r + 2))
                               for x in range(max(0, c - 1), min(w, c + 2)) if labels[y, x])
                    edge = {int(labels[y, x]) for y, x in
                            ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                            if 0 <= y < h and 0 <= x < w and labels[y, x]}
                    if len(near) == 1 and edge == near:
                        growths.append((r, c, next(iter(near))))
            if not growths:
                break
            r, c, label = growths[int(rng.integers(len(growths)))]
            labels[r, c] = label
        mask = labels > 0
        if area is None or int(mask.sum()) == area:
            return mask
    raise AssertionError("exact-area counter construction exhausted")


def _design(rng, shape, axis, count, thickness, crop_required=False):
    h, w = shape
    max_r, max_c = (30 - thickness) // (h + thickness), (30 - thickness) // (w + thickness)
    for _ in range(120):
        mh = int(rng.integers(2, min(h, max_r - (count - 1 if axis == 0 else 0)) + 1))
        mw = int(rng.integers(2, min(w, max_c - (count - 1 if axis == 1 else 0)) + 1))
        mask = rng.random((mh, mw)) < rng.uniform(.3, .8)
        if mask.sum() < 3 or mask.all() or _crop(mask).shape != mask.shape:
            continue
        # Every generated design keeps an inactive position in the swept box.
        swept = _sweep(mask, count, axis)
        if swept.all():
            continue
        if crop_required and not np.any(mask.take(range(1, mask.shape[axis]), axis=axis) &
                                        mask.take(range(mask.shape[axis] - 1), axis=axis)):
            continue
        if crop_required and (mh, mw) == shape:
            continue
        # Check the visible swept footprint: sweeping can hide an asymmetry
        # that was present in the small mask before superposition.
        if crop_required and any(np.array_equal(swept, _sweep(
                np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4), count, axis))
                for k in range(1, 8)):
            continue
        full = np.zeros(shape, dtype=bool)
        top, left = int(rng.integers(h - mh + 1)), int(rng.integers(w - mw + 1))
        full[top:top + mh, left:left + mw] = mask
        return full
    # An asymmetric 2x3 mask, or its transpose, is a finite constructive fallback.
    m = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    if axis == 0:
        m = m.T
    if m.shape[0] > h or m.shape[1] > w or _sweep(m, count, axis).shape[0] > max_r or _sweep(m, count, axis).shape[1] > max_c:
        return None
    full = np.zeros(shape, dtype=bool)
    full[:m.shape[0], :m.shape[1]] = m
    return full


def _fragment(rng, footprint, rows, cols, origin, excluded=None):
    local = list(map(tuple, np.argwhere(footprint)))
    points = set(local)
    maximum = min(len(local) - 1, (rows * cols - 1) // 2)
    if maximum < 2:
        return None
    # Random sampling varies omissions; the exhaustive one-omission pass is bounded.
    proposals = []
    for _ in range(90):
        n = int(rng.integers(2, maximum + 1))
        proposals.append(rng.choice(len(local), n, replace=False).tolist())
    if maximum == len(local) - 1:
        proposals.extend([[j for j in range(len(local)) if j != i] for i in rng.permutation(len(local))])
    for chosen in proposals:
        supplied = {(origin[0] + int(local[i][0]), origin[1] + int(local[i][1])) for i in chosen}
        relative = frozenset(local[i] for i in chosen)
        if relative != excluded and _origins(points, footprint.shape, rows, cols, supplied) == [origin]:
            return supplied
    return None


def _paint(geometry, palette, ordinary, glyph, positions, marker):
    h, w, t, rows, cols = geometry
    a = np.full((rows * (h + t) + t, cols * (w + t) + t), palette[0], dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            a[t + r * (h + t):t + r * (h + t) + h,
              t + c * (w + t):t + c * (w + t) + w] = glyph if (r, c) in positions else ordinary
    a[marker] = palette[3]
    return a


def _make_pair(rng, count, axis, palette, witness=False, familiar=None):
    for attempt in range(160):
        if familiar is None:
            # Witnesses have room for several multi-cell counter objects.
            shapes = [(h, w) for h in range(2 if not witness else 3, 5) for w in range(2 if not witness else 3, 5)
                      if ((h + 1) // 2) * ((w + 1) // 2) >= count]
            h, w = shapes[int(rng.integers(len(shapes)))]
            # Keep at least two stencil cells on the swept axis.
            feasible = [t for t in (1, 2) if (30 - t) // ((h if axis == 0 else w) + t) >= count + 1]
            if not feasible:
                continue
            t = int(rng.choice(feasible))
            mask = _design(rng, (h, w), axis, count, t, witness)
            if mask is None:
                continue
        else:
            h, w, t, _, _ = familiar["geometry"]
            mask = familiar["mask"].copy()
        tight = _crop(mask)
        footprint = _sweep(tight, count, axis)
        max_r, max_c = (30 - t) // (h + t), (30 - t) // (w + t)
        if footprint.shape[0] > max_r or footprint.shape[1] > max_c:
            return None
        rows = int(rng.integers(max(2, footprint.shape[0]), max_r + 1))
        cols = int(rng.integers(max(2, footprint.shape[1]), max_c + 1))
        origin = (int(rng.integers(rows - footprint.shape[0] + 1)),
                  int(rng.integers(cols - footprint.shape[1] + 1)))
        excluded = None
        if familiar is not None:
            excluded = frozenset((r - familiar["origin"][0], c - familiar["origin"][1])
                                 for r, c in familiar["supplied"])
        supplied = _fragment(rng, footprint, rows, cols, origin, excluded)
        if supplied is None:
            continue
        cm = _counter_mask(rng, (h, w), count, area=4 if witness else None)
        if not np.any(cm & ~mask):
            continue  # Full tile copying must visibly erase some counter ink.
        ordinary = np.where(cm, palette[2], palette[1]).astype(np.int8)
        glyph = np.where(mask, palette[3], palette[1]).astype(np.int8)
        geometry = h, w, t, rows, cols
        height, width = rows * (h + t) + t, cols * (w + t) + t
        marker = ((int(rng.integers(1, height - 1)), width - 1) if axis == 1 else
                  (height - 1, int(rng.integers(1, width - 1))))
        active = {(origin[0] + int(r), origin[1] + int(c)) for r, c in np.argwhere(footprint)}
        return {"input": _paint(geometry, palette, ordinary, glyph, supplied, marker),
                "output": _paint(geometry, palette, ordinary, glyph, active, marker)}
    return None


def _count_arg(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = bool(rng.random() < .25)
    # Both first demonstrations actually sweep, with equal counter area but
    # different component counts on the SAME axis, so axis cannot replace count.
    counts = [2, 3]
    rng.shuffle(counts)
    first_axis = int(rng.integers(2))
    shift = int(rng.integers(3))  # new count, new design, or new fragment
    puzzle, seen, training = {"train": [], "test": []}, set(), []
    for split, amount in (("train", num_train), ("test", num_test)):
        for index in range(amount):
            witness = split == "train" and index < 2
            if witness:
                count, axis = counts[index], first_axis
            elif split == "train":
                count = int(rng.choice(counts)) if index == 2 else int(rng.integers(1, 4))
                axis = 1 - first_axis if index == 2 else int(rng.integers(2))
            else:
                count = 4 if shift == 0 else int(rng.choice([d["count"] for d in training]))
                axis = int(rng.choice([d["axis"] for d in training if d["count"] > 1]))
            pair = None
            for attempt in range(100):
                palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
                familiar = None
                if split == "test" and shift == 2:
                    familiar = training[int(rng.integers(len(training)))]
                    count, axis = familiar["count"], familiar["axis"]
                pair = _make_pair(rng, count, axis, palette, witness, familiar)
                if pair is None or _key(pair["input"]) in seen:
                    pair = None
                    continue
                d = _decode(pair["input"])
                if split == "train" and index == 1 and _key(d["tight"]) == _key(training[0]["tight"]):
                    pair = None
                    continue
                if split == "train" and index == 1:
                    old = training[0]
                    if (np.array_equal(_sweep(old["tight"], count, axis), _sweep(d["tight"], count, axis)) or
                            np.array_equal(_sweep(d["tight"], old["count"], old["axis"]),
                                           _sweep(old["tight"], old["count"], old["axis"]))):
                        pair = None
                        continue
                if split == "train" and index >= 2 and all(np.array_equal(
                        _sweep(d["tight"], old["count"], old["axis"]),
                        _sweep(old["tight"], old["count"], old["axis"])) for old in training):
                    pair = None
                    continue
                if split == "test" and shift == 1 and _key(d["tight"]) in {_key(f["tight"]) for f in training}:
                    pair = None
                    continue
                break
            _require(pair is not None, "bounded pair sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                training.append(d)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(inp.shape == out.shape, "canvas changed")
            _require(np.array_equal(solve(inp.copy()), out), "incorrect completion")
            _require(np.any(d["ordinary"][~d["mask"]] != d["background"]), "ink-only copying is indistinguishable")
            _require(len(d["points"]) < d["shape"][0] * d["shape"][1], "rectangle shortcut")
            # Verify that every changed tile is replaced in full and everything
            # outside missing interiors, including the frame marker, is preserved.
            changed_zone = np.zeros(inp.shape, dtype=bool)
            for r, c in d["active"] - d["supplied"]:
                ys, xs = slice(d["rr"][r][1] + 1, d["rr"][r + 1][0]), slice(d["cc"][c][1] + 1, d["cc"][c + 1][0])
                _require(np.array_equal(out[ys, xs], d["glyph"]), "not an exact tile copy")
                changed_zone[ys, xs] = True
            _require(np.array_equal(inp[~changed_zone], out[~changed_zone]), "unaffected cell changed")
            if split == "train":
                training.append(d)
    taught_axes = {d["axis"] for d in training if d["count"] > 1}
    _require(all(_decode(q["input"])["axis"] in taught_axes for q in puzzle["test"]), "test axis lacks sweep evidence")
    _require(any(a["counter_area"] == b["counter_area"] and a["count"] != b["count"] and a["axis"] == b["axis"]
                 for a in training for b in training), "area/count confound")
    _require(any(d["mask"].shape != d["tight"].shape for d in training), "missing stencil padding variation")
    _require(all(d["shape"][d["axis"]] > d["tight"].shape[d["axis"]]
                 for d in training if d["count"] > 1), "counter has no effect")
    _require(any(len(d["points"]) < d["count"] * int(d["tight"].sum()) for d in training), "missing overlap witness")
    _require(len({_key(d["tight"]) for d in training}) >= 2, "training needs different blueprints")
    for fixed in training:
        _require(any(not np.array_equal(_sweep(fixed["tight"], d["count"], d["axis"]),
                                        _sweep(d["tight"], d["count"], d["axis"])) for d in training),
                 "a single training blueprint explains all demonstrations")
    for k in range(1, 8):
        _require(any(not np.array_equal(_sweep(d["tight"], d["count"], d["axis"]), _sweep(
            np.rot90(d["tight"][:, ::-1] if k >= 4 else d["tight"], k % 4), d["count"], d["axis"]))
            for d in training), "orientation is underdetermined after sweeping")


def _alternative_footprints(d):
    """Concrete competing hypotheses for the bounded teaching-evidence audit."""
    m, n, a = d["tight"], d["count"], d["axis"]
    result = {"no_counter": m, "ink_area": _sweep(m, d["counter_area"], a),
              "fixed_two": _sweep(m, 2, a), "fixed_three": _sweep(m, 3, a),
              "n_minus_one": _sweep(m, max(1, n - 1), a), "n_plus_one": _sweep(m, n + 1, a),
              "opposite_axis": _sweep(m, n, 1 - a), "stretch": np.repeat(m, n, axis=a)}
    gap_shape = list(m.shape)
    gap_shape[a] = 1
    parts = []
    for k in range(n):
        if k:
            parts.append(np.zeros(gap_shape, dtype=bool))
        parts.append(m)
    result["separated_repeats"] = np.concatenate(parts, axis=a)
    long_shape = list(m.shape)
    long_shape[a] += 2 * (n - 1)
    stride = np.zeros(long_shape, dtype=bool)
    correct = _sweep(m, n, a)
    xor = np.zeros(correct.shape, dtype=bool)
    for k in range(n):
        if a == 1:
            stride[:, 2 * k:2 * k + m.shape[1]] |= m
            xor[:, k:k + m.shape[1]] ^= m
        else:
            stride[2 * k:2 * k + m.shape[0], :] |= m
            xor[k:k + m.shape[0], :] ^= m
    result.update(stride_two=stride, rectangle=np.ones(correct.shape, dtype=bool), exclusive_overlap=xor)
    for k in range(1, 8):
        result[f"dihedral_{k}"] = _sweep(np.rot90(m[:, ::-1] if k >= 4 else m, k % 4), n, a)
    return result


def self_test():
    """42 seeds, streamed metrics, all explicit counts, and semantic corruptions.

    Only a bounded few current puzzles and small metric sets are retained, never
    all 42 seed payloads. There are no saved traces, logs, caches, or datasets.
    """
    structures, shapes, geometries, counters, glyphs, origins, palettes = (set() for _ in range(7))
    signatures, axes, counts, thicknesses = set(), set(), set(), set()
    familiar_tests = new_tests = padded = disconnected = overlap = pairs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        sig = hashlib.sha256()
        train_designs = {_key(_decode(q["input"])["tight"]) for q in p["train"]}
        for split in ("train", "test"):
            for q in p[split]:
                d = _decode(q["input"])
                before = q["input"].copy()
                _require(np.array_equal(solve(q["input"]), q["output"]), "solve disagreement")
                _require(np.array_equal(q["input"], before), "solve mutated input")
                sig.update(q["input"].tobytes())
                pairs += 1
                shapes.add(q["input"].shape)
                geometries.add(d["geometry"])
                counters.add(_key(d["ordinary"] == d["counter"]))
                glyphs.add(_key(d["tight"]))
                structures.add((d["geometry"], _key(d["mask"]), tuple(sorted(d["supplied"])), d["count"], d["axis"]))
                origins.add(d["origin"])
                palettes.add((d["divider"], d["background"], d["counter"], d["ink"]))
                axes.add(d["axis"])
                counts.add(d["count"])
                thicknesses.add(d["geometry"][2])
                padded += d["mask"].shape != d["tight"].shape
                disconnected += len(_components(d["tight"])) > 1
                overlap += len(d["points"]) < d["count"] * int(d["tight"].sum())
                if split == "test":
                    familiar_tests += _key(d["tight"]) in train_designs
                    new_tests += _key(d["tight"]) not in train_designs
        signatures.add(sig.digest())
        for nt, teaching in ((4, p), (2, generate_puzzle(seed, 2, 1))):
            survivors = None
            for q in teaching["train"]:
                d = _decode(q["input"])
                correct = _sweep(d["tight"], d["count"], d["axis"])
                fits = {name for name, arr in _alternative_footprints(d).items()
                        if np.array_equal(arr, correct)}
                survivors = fits if survivors is None else survivors & fits
            _require(not survivors, f"simpler alternative survives seed {seed}, {nt} training pairs")
    _require(len(signatures) == 42 and len(structures) >= 150, "structural diversity collapsed")
    _require(len(glyphs) >= 30 and len(counters) >= 25, "design diversity collapsed")
    _require(len(geometries) >= 30 and len(shapes) >= 20 and len(origins) >= 6, "geometry diversity collapsed")
    _require(axes == {0, 1} and counts == {1, 2, 3, 4} and thicknesses == {1, 2}, "missing family modes")
    _require(familiar_tests >= 5 and new_tests >= 5 and padded >= 50 and disconnected >= 3, "missing topology/transfer variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(919, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "ignored explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    for seed in (0, 41):
        a, b = generate_puzzle(seed), generate_puzzle(seed)
        _require(all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s])
                     for k in ("input", "output")), "nondeterminism")
    p = generate_puzzle(0)
    q = p["train"][0]
    d = _decode(q["input"])
    # Destroy a changed non-ink pixel, the common ink-only copying mistake.
    r, c = min(d["active"] - d["supplied"])
    dy, dx = map(int, np.argwhere((d["ordinary"] != d["glyph"]) & ~d["mask"])[0])
    y, x = d["rr"][r][1] + 1 + dy, d["cc"][c][1] + 1 + dx
    original = q["output"][y, x]
    q["output"][y, x] = d["ordinary"][dy, dx]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted ink-only copying")
    q["output"][y, x] = original
    damaged = q["input"].copy()
    damaged[d["marker"]] = d["divider"]
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing axis marker accepted")
    # Even a perfect-overlap tie must be rejected, never resolved by scan order.
    _require(len(_origins({(0, 0), (0, 1), (1, 0), (1, 1)}, (2, 2), 5, 5, {(2, 2), (2, 3)})) > 1,
             "ambiguity fixture invalid")
    return {"seeds": 42, "pairs": pairs, "structures": len(structures), "canvas_shapes": len(shapes),
            "geometries": len(geometries), "counter_designs": len(counters), "glyph_designs": len(glyphs),
            "origins": len(origins), "palettes": len(palettes), "counts": sorted(counts),
            "padded_glyphs": int(padded), "disconnected_glyphs": int(disconnected),
            "overlap_pairs": int(overlap), "familiar_tests": int(familiar_tests), "new_tests": int(new_tests),
            "alternative_models": 19, "alternative_audit_puzzles": 84, "alternative_survivors": {}}
