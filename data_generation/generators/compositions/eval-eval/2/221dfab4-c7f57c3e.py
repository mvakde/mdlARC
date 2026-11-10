"""Resonant toggles: periodic cross-sections act on invariant motif scaffolds.

Only the standard library and NumPy are used. No file access or saved state is
needed for generation, solving, validation, or the bounded self-test.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


_WITNESSES = frozenset(("odd_0", "odd_1", "even_positive", "zero",
                       "pixel_parity", "ordinary_ticks", "box_gap",
                       "accent_only", "relocation", "scaled_swap",
                       "anchor_shortcut", "block_sampling"))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or any(n < 1 or n > 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer 1..30 by 1..30 ARC grid")
    return g.astype(np.int8, copy=True)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(g, side):
    return (g, g.T[::-1], g[::-1], g.T)[side]


def _restore(g, side):
    return (g, g[::-1].T, g[::-1], g.T)[side].copy()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        group, todo = [root], [root]
        while todo:
            y, x = todo.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    p = y + dy, x + dx
                    if p in remaining:
                        remaining.remove(p)
                        todo.append(p)
                        group.append(p)
        groups.append(group)
    return groups


def _bounds(points):
    return (min(y for y, x in points), max(y for y, x in points),
            min(x for y, x in points), max(x for y, x in points))


def _signature(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _primitive(crop):
    divisor = 0
    for a in (crop, crop.T):
        for row in a:
            boundaries = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            for n in np.diff(boundaries):
                divisor = math.gcd(divisor, int(n))
    small = crop[::divisor, ::divisor]
    _require(np.array_equal(crop, np.repeat(np.repeat(small, divisor, 0), divisor, 1)),
             "nonuniform pixel blocks")
    return divisor, small


def _separated(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h < yy or yy + hh < y or x + w < xx or xx + ww < x


def _bar(g, bg):
    candidates = []
    for color in np.unique(g):
        if color == bg:
            continue
        points = np.argwhere(g == color)
        yy, xx = points.T
        for side, fixed, edge, lateral in ((0, yy, 0, xx), (1, xx, g.shape[1] - 1, yy),
                                           (2, yy, g.shape[0] - 1, xx), (3, xx, 0, yy)):
            if (np.all(fixed == edge) and
                    np.array_equal(np.sort(lateral), np.arange(int(lateral.min()),
                                                               int(lateral.min()) + len(points)))):
                candidates.append((side, int(lateral.min()), len(points), int(color)))
    _require(len(candidates) == 1, "expected one unambiguous boundary bar")
    return candidates[0]


def _read_input(value):
    """Recover the entire scene from input pixels; never consult generation."""
    original = _grid(value)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] * 2 > original.size, "background must occupy a majority")
    side, start, length, marker = _bar(original, bg)
    g = _canonical(original, side).copy()
    end = start + length
    _require(np.all(g[1:, start:end] == bg), "ruler corridor must initially be empty")
    mask = (g != bg) & (g != marker)
    _require(not np.any(mask[0]), "motifs must not touch the seed edge")
    objects = []
    for group in _components(mask):
        y0, y1, x0, x1 = _bounds(group)
        crop = g[y0:y1 + 1, x0:x1 + 1]
        _require(np.count_nonzero(crop != bg) == len(group), "interleaved motifs")
        scale, small = _primitive(crop)
        raw = {(int(y), int(x)): int(small[y, x])
               for y, x in np.argwhere(small != bg)}
        objects.append(dict(box=(y0, x0), scale=scale, raw=_signature(raw), group=group))
    signatures = sorted({o["raw"] for o in objects})
    _require(len(signatures) == 2, "exactly two visible primitive motif states required")
    _require({o["raw"] for o in objects if o["scale"] == 1} == set(signatures),
             "both states must have a unit reference")
    maps = [{(y, x): c for y, x, c in sig} for sig in signatures]
    shared = set(maps[0].values()) & set(maps[1].values())
    _require(len(shared) == 1, "states must share exactly one carrier color")
    carrier = next(iter(shared))
    templates, offsets = [], []
    for mapping in maps:
        carrier_points = {p for p, c in mapping.items() if c == carrier}
        _require(len(carrier_points) >= 3, "scaffold is too small")
        y0, _, x0, _ = _bounds(carrier_points)
        offsets.append((y0, x0))
        templates.append({(y - y0, x - x0): c for (y, x), c in mapping.items()})
    detector = {p for p, c in templates[0].items() if c == carrier}
    _require(detector == {p for p, c in templates[1].items() if c == carrier},
             "carrier patterns do not register")
    union = set(templates[0]) | set(templates[1])
    ymin, ymax, xmin, xmax = _bounds(union)
    rectangles = []
    for obj in objects:
        state = signatures.index(obj["raw"])
        s = obj["scale"]
        oy, ox = offsets[state]
        ay, ax = obj["box"][0] + oy * s, obj["box"][1] + ox * s
        rect = (ay + ymin * s, ax + xmin * s, (ymax - ymin + 1) * s, (xmax - xmin + 1) * s)
        y, x, h, w = rect
        _require(1 <= y and y + h <= len(g) and 0 <= x and x + w <= g.shape[1],
                 "replacement union clips or touches seed edge")
        _require(x + w < start or x > end, "replacement union must be clear of corridor")
        _require(all(_separated(rect, other) for other in rectangles), "replacement unions collide")
        rectangles.append(rect)
        carrier_pixels = [(r, c) for r, c in obj["group"] if int(g[r, c]) == carrier]
        rows = sorted({r for r, c in carrier_pixels})
        hits = sum(r % 6 == 4 for r in rows)
        all_rows = {r for r, c in obj["group"]}
        obj.update(state=state, anchor=(ay, ax), rect=rect, hits=hits,
                   pixel_hits=sum(r % 6 == 4 for r, c in carrier_pixels),
                   box_hits=sum(r % 6 == 4 for r in range(min(rows), max(rows) + 1)),
                   full_hits=sum(r % 6 == 4 for r in all_rows),
                   even_rows=sum(r % 2 == 0 for r in rows), rows=rows,
                   block_hits=sum((ay + r * s) % 6 == 4 for r in {p[0] for p in detector}))
    return dict(grid=g, side=side, background=bg, marker=marker, carrier=carrier,
                start=start, length=length, templates=templates, detector=detector,
                union=union, objects=objects)


def _paint(g, mapping, anchor, scale):
    for (y, x), color in mapping.items():
        yy, xx = anchor[0] + y * scale, anchor[1] + x * scale
        g[yy:yy + scale, xx:xx + scale] = color


def _draw_ruler(out, start, length, bg, marker, carrier):
    out[:, start:start + length] = bg
    out[0::6, start:start + length] = marker
    out[2::6, start:start + length] = marker
    out[4::6, start:start + length] = carrier


def _prediction(info, toggle="resonant"):
    out = info["grid"].copy()
    for obj in info["objects"]:
        flip = bool(obj["hits"] % 2) if toggle == "resonant" else toggle == "all"
        if not flip:
            continue
        s, (ay, ax) = obj["scale"], obj["anchor"]
        for y, x in info["union"]:
            out[ay + y * s:ay + (y + 1) * s, ax + x * s:ax + (x + 1) * s] = info["background"]
        _paint(out, info["templates"][1 - obj["state"]], (ay, ax), s)
    if toggle == "recolor":
        for d in range(4, len(out), 6):
            foreground = (info["grid"][d] != info["background"]) & (info["grid"][d] != info["marker"])
            out[d, foreground] = info["carrier"]
    _draw_ruler(out, info["start"], info["length"], info["background"], info["marker"], info["carrier"])
    return _restore(out, info["side"])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a family member using this input alone; leave the input unchanged."""
    return _prediction(_read_input(input_grid))


def _witnesses(info):
    result = set()
    moving = set(info["templates"][0]) != set(info["templates"][1])
    for obj in info["objects"]:
        hits = obj["hits"]
        if hits % 2:
            result.add("odd_" + str(obj["state"]))
            if moving:
                result.add("relocation")
            if obj["scale"] > 1:
                result.add("scaled_swap")
        if hits >= 2 and hits % 2 == 0:
            result.add("even_positive")
        if hits == 0:
            result.add("zero")
        for name, count in (("pixel_parity", obj["pixel_hits"]),
                            ("ordinary_ticks", obj["even_rows"]),
                            ("box_gap", obj["box_hits"]), ("accent_only", obj["full_hits"]),
                            ("anchor_shortcut", int(obj["anchor"][0] % 6 == 4)),
                            ("block_sampling", obj["block_hits"])):
            if count % 2 != hits % 2:
                result.add(name)
    return result


def _pick(rng, items):
    items = sorted(items)
    return items[int(rng.integers(len(items)))]


def _design(rng, change, gap_along_depth):
    """Fresh irregular connected footprints, with an accent bridge across a gap."""
    h, w = int(rng.integers(4, 6)), int(rng.integers(2, 5))
    kind = int(rng.integers(4))
    spine = int(rng.integers(w))
    core = set()
    for y in range(h):
        if kind in (0, 1):
            spine = int(np.clip(spine + rng.integers(-1, 2), 0, w - 1))
            core.add((y, spine))
            if rng.random() < .7 or y in (0, h - 1):
                a, b = sorted((spine, int(rng.integers(w))))
                core.update((y, x) for x in range(a, b + 1))
        elif kind == 2:
            core.update(((y, 0), (y, w - 1)))
            if y in (0, h - 1):
                core.update((y, x) for x in range(w))
        else:
            core.add((y, w // 2))
            if y == h // 2 or rng.random() < .5:
                core.update((y, x) for x in range(w))
    # The full motif stays connected while its unchanged detector can have gaps.
    bridge_row = int(rng.integers(1, h - 1))
    changing = {p for p in core if p[0] == bridge_row}
    carrier = core - changing
    if change == "relocate":
        carrier, changing = core, set()

    def attachment(side):
        candidates = {p for p in core if p[side // 2] ==
                      (min(p[side // 2] for p in core) if side % 2 == 0
                       else max(p[side // 2] for p in core))}
        y, x = _pick(rng, candidates)
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[side]
        py, px = y + dy, x + dx
        result = {(py, px)}
        if rng.random() < .6:
            lateral = (0, 1) if dy else (1, 0)
            sign = int(rng.choice([-1, 1]))
            q = py + lateral[0] * sign, px + lateral[1] * sign
            if -1 <= q[0] <= h and -1 <= q[1] <= w and q not in core:
                result.add(q)
        return result - core

    side = int(rng.integers(4))
    a = attachment(side)
    b = a if change == "recolor" else attachment((side + int(rng.integers(1, 4))) % 4)
    maps = []
    for state, deco in enumerate((a, b)):
        mapping = {p: 1 for p in carrier}
        accents = sorted(changing | deco)
        for i, p in enumerate(accents):
            mapping[p] = 2 + state * 2 + (i % 2 if rng.random() < .55 else 0)
        maps.append(mapping)
    rotation = int(rng.choice([0, 2])) if gap_along_depth else int(rng.integers(4))
    reflect = bool(rng.integers(2))
    transformed = []
    for mapping in maps:
        result = {}
        for (y, x), color in mapping.items():
            for _ in range(rotation):
                y, x = x, -y
            result[y, -x if reflect else x] = color
        transformed.append(result)
    ymin, ymax, xmin, xmax = _bounds(set(transformed[0]) | set(transformed[1]))
    templates = [{(y - ymin, x - xmin): c for (y, x), c in m.items()} for m in transformed]
    return templates, (ymax - ymin + 1, xmax - xmin + 1)


def _row_features(templates, top, scale):
    carrier = {top + y * scale + k for (y, x), c in templates[0].items() if c == 1 for k in range(scale)}
    hits = sum(y % 6 == 4 for y in carrier)
    bbox = sum(y % 6 == 4 for y in range(min(carrier), max(carrier) + 1))
    return hits, bbox


def _positions(rng, shape, sizes, templates, corridor, profiles):
    """Place the largest unions first; row constraints teach specific contrasts."""
    depth, width = shape
    start, length = corridor
    end = start + length
    placed = [None] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda j: sizes[j], reverse=True):
        s = sizes[i]
        ys = []
        h = max(y for t in templates for y, x in t) + 1
        w = max(x for t in templates for y, x in t) + 1
        for y in range(1, depth - h * s + 1):
            hits, bbox = _row_features(templates, y, s)
            profile = profiles[i]
            if (profile == "odd" and hits % 2 != 1 or
                    profile == "cancel" and not (hits >= 2 and hits % 2 == 0) or
                    profile == "gap" and not (hits == 0 and bbox % 2 == 1) or
                    profile == "zero" and hits != 0):
                continue
            ys.append(y)
        if not ys:
            return None
        # Randomized row and column trials; no stored proposal history.
        for y in rng.permutation(ys):
            xs = []
            for x in range(width - w * s + 1):
                rect = (int(y), x, h * s, w * s)
                if (x + w * s < start or x > end) and all(
                        _separated(rect, other) for other in placed if other is not None):
                    xs.append(x)
            if xs:
                placed[i] = (int(y), int(rng.choice(xs)), h * s, w * s)
                break
        if placed[i] is None:
            return None
    return placed


def _candidate(rng, role, side):
    first = role == "cancellation"
    teaching = role in ("cancellation", "scale")
    change = "combined" if teaching else str(rng.choice(["recolor", "relocate", "combined"]))
    templates, logical_shape = _design(rng, change, teaching)
    h, w = logical_shape
    if max(h, w) > 7 or min(h, w) > 6:
        return None
    if role == "test":
        scales = [1, 1, 3] + [1] * int(rng.integers(0, 3))
        profiles = ["any", "any", "odd"] + ["any"] * (len(scales) - 3)
        depth, width = int(rng.integers(27, 31)), int(rng.integers(27, 31))
    elif teaching:
        scales = [1, 1, 2, 1]
        profiles = ["odd", "odd", "cancel" if first else "odd", "gap"]
        depth, width = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    else:
        scales = [1, 1] + [int(rng.choice([1, 1, 2])) for _ in range(int(rng.integers(1, 5)))]
        profiles = ["any"] * len(scales)
        depth, width = int(rng.integers(17, 31)), int(rng.integers(18, 31))
    states = [0, 1] + [int(rng.integers(2)) for _ in scales[2:]]
    length = int(rng.integers(1, 4))
    if rng.random() < (.88 if role == "test" else .55):
        # Preserve enough lateral capacity for enlarged motifs; vary the margin.
        start = (1 if length == 1 else 0) + int(rng.integers(0, 2))
        if rng.integers(2):
            start = width - length - start
    else:
        start = int(rng.integers(1, width - length))
    placements = _positions(rng, (depth, width), scales, templates, (start, length), profiles)
    if placements is None:
        return None
    palette = rng.permutation(10).tolist()
    bg, marker = palette[0], palette[6]
    colored = [{p: palette[c] for p, c in template.items()} for template in templates]
    inp, out = np.full((depth, width), bg, dtype=np.int8), np.full((depth, width), bg, dtype=np.int8)
    detector_rows = sorted({y for (y, x), c in templates[0].items() if c == 1})
    for rect, s, state in zip(placements, scales, states):
        y, x = rect[:2]
        physical_rows = {y + r * s + k for r in detector_rows for k in range(s)}
        hit_count = sum(d % 6 == 4 for d in physical_rows)
        _paint(inp, colored[state], (y, x), s)
        _paint(out, colored[state ^ (hit_count % 2)], (y, x), s)
    inp[0, start:start + length] = marker
    _draw_ruler(out, start, length, bg, marker, palette[1])
    pair = {"input": _restore(inp, side), "output": _restore(out, side)}
    try:
        info = _read_input(pair["input"])
    except ValueError:
        return None
    return pair, info


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base, regime = int(rng.integers(4)), int(rng.integers(4))
    puzzle, seen, coverage = {"train": [], "test": []}, set(), set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            role = ("cancellation" if index == 0 else "scale" if index == 1 else "normal") if split == "train" else "test"
            side = (base if regime in (0, 3) else (base + (2 * index if regime == 1 else index)) % 4) if split == "train" else (
                base if regime == 3 else (base + 1 + 2 * (index % 2)) % 4)
            required = _WITNESSES - coverage if split == "train" and index == num_train - 1 else set()
            if role == "cancellation":
                required = {"odd_0", "odd_1", "even_positive", "zero", "box_gap", "accent_only", "relocation"}
            if role == "scale":
                required = set(required) | {"scaled_swap", "pixel_parity", "block_sampling"}
            for attempt in range(400):
                candidate = _candidate(rng, role, side)
                if candidate is None:
                    continue
                pair, info = candidate
                evidence = _witnesses(info)
                if _key(pair["input"]) in seen or not set(required) <= evidence:
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    coverage.update(evidence)
                break
            else:
                raise RuntimeError(f"bounded synthesis exhausted for {split} example {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Full pixel checks plus geometric and teaching invariants, with no RNG."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _count("num_train", nt, 2, 10)
    _count("num_test", ns, 1, 4)
    _require(nt > ns, "training must outnumber tests")
    seen, coverage, ablations = set(), set(), set()
    train_scales, test_scales = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "output dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _read_input(a)
            expected = _prediction(info)
            _require(np.array_equal(expected, b), "incorrect resonant exchange pixels")
            canonical_b = _canonical(b, info["side"])
            outside = np.ones(a.shape if info["side"] in (0, 2) else a.T.shape, dtype=bool)
            outside[:, info["start"]:info["start"] + info["length"]] = False
            original_carrier = (info["grid"] == info["carrier"]) & outside
            _require(np.array_equal(original_carrier, (canonical_b == info["carrier"]) & outside),
                     "scaffold moved or changed")
            for obj in info["objects"]:
                y, x, h, w = obj["rect"]
                outside[y:y + h, x:x + w] = False
            _require(np.array_equal(canonical_b[outside], info["grid"][outside]), "unrelated pixels changed")
            scales = {o["scale"] for o in info["objects"]}
            if split == "train":
                coverage.update(_witnesses(info))
                train_scales.update(scales)
                for mode in ("all", "none", "recolor"):
                    if not np.array_equal(_prediction(info, mode), b):
                        ablations.add(mode)
            else:
                test_scales.update(scales)
    _require(_WITNESSES <= coverage, "training omits a distinguishing witness: " + str(sorted(_WITNESSES - coverage)))
    _require(ablations == {"all", "none", "recolor"}, "one parent can be omitted without changing training answers")
    _require(2 in train_scales, "training must demonstrate scale two")
    _require(3 in test_scales, "generated puzzle must test scale three")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = _grid(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, bounded RAM, no disk writes; return compact diversity evidence."""
    shapes, detectors, schedules, counts, palettes, sides, signatures = (set() for _ in range(7))
    hit_counts, structures, changes = set(), set(), set()
    pair_count = reexposures = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                info = _read_input(pair["input"])
                shapes.add(pair["input"].shape)
                detectors.add(tuple(sorted(info["detector"])))
                schedules.add(tuple(sorted(o["scale"] for o in info["objects"])))
                counts.add(len(info["objects"]))
                palettes.add(tuple(sorted(map(int, np.unique(pair["input"])))))
                sides.add(info["side"])
                hit_counts.update(o["hits"] for o in info["objects"])
                changes.add(set(info["templates"][0]) != set(info["templates"][1]))
                colors = {}
                normalized = bytes(colors.setdefault(int(c), len(colors)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, normalized))
                before = pair["input"].copy()
                _require(np.array_equal(solve(before), pair["output"]), "solver disagreement")
                _require(np.array_equal(before, pair["input"]), "solve modified input")
        # Equivariance is checked on a held-out geometry, without seed metadata.
        pair = puzzle["test"][0]
        for a, b in ((np.rot90(pair["input"]), np.rot90(pair["output"])),
                     (np.fliplr(pair["input"]), np.fliplr(pair["output"]))):
            _require(np.array_equal(solve(a), b), "orientation equivariance failed")
        permutation = np.random.default_rng(10000 + seed).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "palette equivariance failed")
        # Remove just the newly drawn ruler between two exposures. The unchanged
        # detector must make this operation an involution on all motif states.
        pair = puzzle["train"][0]
        info = _read_input(pair["input"])
        # A new accent at an outer edge can itself look like a bar after a swap.
        # Use admissible interior scenes for the input-only re-exposure check.
        if len(info["grid"]) < 30 and info["grid"].shape[1] <= 28:
            padded = np.pad(info["grid"], ((0, 1), (1, 1)), constant_values=info["background"])
            scene = _restore(padded, info["side"])
            padded_info = _read_input(scene)
            once = _canonical(solve(scene), info["side"])
            once[1:, padded_info["start"]:padded_info["start"] + info["length"]] = info["background"]
            twice = solve(_restore(once, info["side"]))
            _require(np.array_equal(twice, _prediction(padded_info, "none")), "two exposures did not cancel")
            reexposures += 1
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + nt * 10 + ns, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "wrong explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    info = _read_input(sample["train"][0]["input"])
    incomplete = np.full_like(info["grid"], info["background"])
    incomplete[0, info["start"]:info["start"] + info["length"]] = info["marker"]
    for obj in info["objects"]:
        _paint(incomplete, info["templates"][0], obj["anchor"], obj["scale"])
    corner = np.zeros((5, 5), dtype=np.int8)
    corner[0, 0] = 1
    for invalid in (_restore(incomplete, info["side"]), corner):
        try:
            solve(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("missing state or ambiguous bar accepted")
    # Solver-domain checks outside sampled scale schedules and bar widths.
    # Expectations use these explicit templates and hand-counted pulse counts.
    detector = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (2, 0): 1}
    templates = [detector.copy(), detector.copy()]
    templates[0][1, 1] = 2
    templates[1][1, 2] = 3
    for large_scale in (0, 4, 5):
        inp = np.zeros((20, 30), dtype=np.int8)
        expected = inp.copy()
        inp[0, :7] = 7
        for anchor, state in (((2, 8), 0), ((9, 8), 1)):
            _paint(inp, templates[state], anchor, 1)
            _paint(expected, templates[1 - state], anchor, 1)  # one special hit
        if large_scale:
            _paint(inp, templates[0], (3, 12), large_scale)
            # Scale four covers physical rows 3..14 (two hits); scale five
            # covers rows 3..17 (three hits). No logical-row shortcut works.
            _paint(expected, templates[0 if large_scale == 4 else 1], (3, 12), large_scale)
        _draw_ruler(expected, 0, 7, 0, 7, 1)
        _require(np.array_equal(solve(inp), expected), "general family-domain scale check failed")
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(detectors) >= 30,
             "insufficient structural variation")
    _require(len(schedules) >= 8 and len(counts) >= 3 and len(palettes) >= 20,
             "insufficient scale/count/palette variation")
    _require(sides == {0, 1, 2, 3} and hit_counts >= {0, 1, 2} and changes == {False, True},
             "missing semantic regimes")
    _require(reexposures >= 10, "insufficient input-only involution checks")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "scaffolds": len(detectors), "scale_schedules": len(schedules),
            "motif_counts": sorted(counts), "palettes": len(palettes), "hit_counts": sorted(hit_counts),
            "color_normalized_inputs": len(structures), "involution_checks": reexposures,
            "independent_solver": False}
