"""Synchronized receivers: diamond-cap endpoints jointly address embedded pluses.

Only standard-library modules and NumPy are used. No parent module is imported.
See task_spec.md for the rule and the admissible-input domain.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return g.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _cross(p):
    r, c = p
    return [(r, c)] + [(r + a, c + b) for a, b in CARDINALS]


def _diamond(center, radius):
    r, c = center
    return [(r + a, c + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius]


def _interior(center, radius):
    r, c = center
    return [(r + a, c + b) for a in range(1 - radius, radius)
            for b in range(1 - radius, radius) if abs(a) + abs(b) < radius]


def _cap(center, radius, direction):
    r, c = center
    a, b = direction
    return [(r + a * (k + 1), c + b * (radius - k + 1))
            for k in range(radius + 1)]


def _step(p, d, t):
    return p[0] + d[0] * t, p[1] + d[1] * t


def _background(g):
    parity = np.indices(g.shape).sum(axis=0) % 2
    colors = []
    for k in (0, 1):
        counts = np.bincount(g[parity == k], minlength=10)
        color = int(counts.argmax())
        if counts[color] * 2 <= int((parity == k).sum()):
            raise ValueError("checkerboard colors must have strict parity majorities")
        colors.append(color)
    if colors[0] == colors[1]:
        raise ValueError("the checkerboard uses two distinct colors")
    return tuple(colors), np.asarray(colors, dtype=np.int8)[parity]


def _centers(g, background):
    """All uniform five-cell crosses, simultaneously, in any nonfield color."""
    if min(g.shape) < 3:
        return {}
    m = g[1:-1, 1:-1]
    valid = ((m != background[0]) & (m != background[1])
             & (m == g[:-2, 1:-1]) & (m == g[2:, 1:-1])
             & (m == g[1:-1, :-2]) & (m == g[1:-1, 2:]))
    return {(int(r) + 1, int(c) + 1): int(m[r, c]) for r, c in np.argwhere(valid)}


def _emitters(g, background, base):
    found = []
    h, w = g.shape
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        sh = (h - 2 * radius, w - 2 * radius)
        offsets = _diamond((0, 0), radius)
        a, b = offsets[0]
        color = g[radius + a:radius + a + sh[0], radius + b:radius + b + sh[1]]
        mask = (color != background[0]) & (color != background[1])
        for a, b in offsets[1:]:
            mask &= g[radius + a:radius + a + sh[0], radius + b:radius + b + sh[1]] == color
        for r0, c0 in np.argwhere(mask):
            center = (int(r0) + radius, int(c0) + radius)
            if not all(g[p] == base[p] for p in _interior(center, radius)):
                continue
            structural = int(color[r0, c0])
            caps = {}
            for d in DIRECTIONS:
                visible = [p for p in _cap(center, radius, d) if _inside(p, g.shape)]
                if len(visible) < 2:
                    continue
                v = int(g[visible[0]])
                if v not in (*background, structural) and all(g[p] == v for p in visible):
                    caps[d] = v
            if caps:
                found.append(dict(center=center, radius=radius, structural=structural, caps=caps))
    found.sort(key=lambda e: e["center"])
    if not 1 <= len(found) <= 4:
        raise ValueError("expected one through four unambiguous capped diamond outlines")
    owned = set()
    for e in found:
        points = _diamond(e["center"], e["radius"])
        for d in e["caps"]:
            points += [p for p in _cap(e["center"], e["radius"], d) if _inside(p, g.shape)]
        if owned.intersection(points):
            raise ValueError("emitter footprints overlap")
        owned.update(points)
    return found, owned


def _analyze(g):
    bg, base = _background(g)
    emitters, owned = _emitters(g, bg, base)
    centers = _centers(g, bg)
    if any(owned.intersection(_cross(p)) for p in centers):
        raise ValueError("receiver motifs may not include emitter cells")
    return dict(background=bg, base=base, emitters=emitters, owned=owned, centers=centers)


def _matches(analysis, shape):
    result = []
    centers = analysis["centers"]
    for index, e in enumerate(analysis["emitters"]):
        for d, color in e["caps"].items():
            cap = _cap(e["center"], e["radius"], d)
            # A clipped endpoint has no outward continuation; a pair needs both.
            if not _inside(cap[0], shape) or not _inside(cap[-1], shape):
                continue
            for t in range(1, max(shape)):
                p, q = _step(cap[0], d, t), _step(cap[-1], d, t)
                if not _inside(p, shape) or not _inside(q, shape):
                    break
                if p in centers and q in centers:
                    result.append((index, d, t, p, q, color))
    return result


def _claims(matches, owned):
    claimed = {}
    for _, _, _, p, q, color in matches:
        for cell in _cross(p) + _cross(q):
            if cell in owned:
                raise ValueError("a selected receiver overlaps a source")
            if cell in claimed and claimed[cell] != color:
                raise ValueError("different payloads claim the same receiver cell")
            claimed[cell] = color
    return claimed


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input alone; all output colors are on caps."""
    g = _grid(input_grid)
    a = _analyze(g)
    out = g.copy()
    for p, color in _claims(_matches(a, g.shape), a["owned"]).items():
        out[p] = color
    return out


def _direct_prediction(g, analysis):
    """Validation uses direct cross scans and equal rotated-coordinate depths.

    This shares emitter recovery, but neither the production cross-mask nor the
    endpoint stepping. It is a generator-owned check, not an external oracle.
    """
    bg = analysis["background"]
    centers = set()
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            points = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            if g[r, c] not in bg and all(g[p] == g[r, c] for p in points):
                centers.add((r, c))
    assert centers == set(analysis["centers"])
    claims = {}
    for e in analysis["emitters"]:
        r, c = e["center"]
        radius = e["radius"]
        for (dr, dc), color in e["caps"].items():
            lanes = ({}, {})
            for p in centers:
                u, v = (p[0] - r) * dr, (p[1] - c) * dc
                if min(u, v) >= 2 and u - v in (-radius, radius):
                    lanes[int(u - v == radius)][min(u, v)] = p
            for depth in lanes[0].keys() & lanes[1].keys():
                for p in (lanes[0][depth], lanes[1][depth]):
                    rr, cc = p
                    for cell in ((rr, cc), (rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        assert cell not in claims or claims[cell] == color
                        claims[cell] = color
    out = g.copy()
    for p, color in claims.items():
        out[p] = color
    return out


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _paint(assignments, points, color, shape):
    if any(not _inside(p, shape) for p in points):
        return False
    if any(p in assignments and assignments[p] != color for p in points):
        return False
    assignments.update({p: color for p in points})
    return True


def _palette(rng, stable):
    if stable is not None:
        return stable
    return tuple(map(int, rng.permutation(10)))


def _source_assignments(e, shape):
    points = {p: e["structural"] for p in _diamond(e["center"], e["radius"])}
    for d, color in e["caps"].items():
        visible = [p for p in _cap(e["center"], e["radius"], d) if _inside(p, shape)]
        if len(visible) < 2:
            return None
        points.update({p: color for p in visible})
    return points


def _render(shape, palette, specs, receiver_assignments):
    parity = np.indices(shape).sum(axis=0) % 2
    g = np.asarray(palette[:2], dtype=np.int8)[parity]
    all_source = {}
    for e in specs:
        cells = _source_assignments(e, shape)
        if cells is None or any(p in all_source for p in cells):
            return None
        all_source.update(cells)
    if set(receiver_assignments) & set(all_source):
        return None
    for e in specs:
        if set(receiver_assignments) & set(_interior(e["center"], e["radius"])):
            return None
    for p, color in {**all_source, **receiver_assignments}.items():
        if not _inside(p, shape):
            return None
        g[p] = color
    try:
        a = _analyze(g)
        if a["emitters"] != sorted(specs, key=lambda e: e["center"]):
            return None
        matches = _matches(a, shape)
        if not matches:
            return None
        out = solve(g)
        if np.array_equal(g, out):
            return None
        return {"input": g, "output": out}
    except ValueError:
        return None


def _witness(rng, max_radius, directions, kind, stable):
    palette = _palette(rng, stable)
    low = 2 * max_radius + 21
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    center = tuple((n - 1) // 2 for n in shape)
    caps = {d: palette[3 + i % 4] for i, d in enumerate(directions)}
    e = dict(center=center, radius=max_radius, structural=palette[2], caps=caps)
    strokes = {}
    for i, d in enumerate(directions):
        cap = _cap(center, max_radius, d)
        # Shared ink permits the small-radius examples to contain genuinely
        # embedded/overlapping motifs, rather than excluding their geometry.
        inks = (palette[7], palette[7])
        depths = (1, 2, 5) if kind == 0 else (1,)
        for t in depths:
            for endpoint, ink in zip((cap[0], cap[-1]), inks):
                if not _paint(strokes, _cross(_step(endpoint, d, t)), ink, shape):
                    return None
        # A real but unpaired plus on a rail distinguishes independent rays.
        if not _paint(strokes, _cross(_step(cap[0], d, 8 if kind == 0 else 7)), inks[0], shape):
            return None
        if kind == 1:
            if not _paint(strokes, _cross(_step(cap[-1], d, 8)), inks[1], shape):
                return None
            # At equal depth, one complete cross and one T remain unselected.
            p, q = _step(cap[0], d, 4), _step(cap[-1], d, 4)
            missing = (q[0] - d[0], q[1])
            if not _paint(strokes, _cross(p), inks[0], shape):
                return None
            if not _paint(strokes, [z for z in _cross(q) if z != missing], inks[1], shape):
                return None
        # A complete cross on an interior cap lane does not receive anything.
        p = _step(cap[len(cap) // 2], d, 6 if kind == 0 else 5)
        if not _paint(strokes, _cross(p), palette[7], shape):
            return None
        # An attached arm extension must survive the selective five-cell mask.
        p = _step(cap[0], d, depths[-1])
        if not _paint(strokes, [(p[0] + 2 * d[0], p[1])], inks[0], shape):
            return None
    return _render(shape, palette, [e], strokes)


def _available_depth(center, radius, d, shape):
    distances = [center[k] if d[k] < 0 else shape[k] - 1 - center[k] for k in (0, 1)]
    return min(distances) - radius - 2  # includes the receiver's outward arm


def _candidate(rng, shape, radius, palette, far=False, allow_clip=True):
    center = tuple(int(rng.integers(radius, n - radius)) for n in shape)
    eligible = [d for d in DIRECTIONS if _available_depth(center, radius, d, shape) >= (7 if far else 1)]
    if not eligible:
        return None
    rng.shuffle(eligible)
    directions = eligible[:int(rng.integers(1, len(eligible) + 1))]
    structural = palette[2 + int(rng.integers(2))]
    payloads = [c for c in palette[3:7] if c != structural]
    caps = {d: int(rng.choice(payloads)) for d in directions}
    # Occasionally add an inactive/clipped cap: its endpoints have no paired receivers.
    if rng.random() < .3:
        for d in DIRECTIONS:
            visible = [p for p in _cap(center, radius, d) if _inside(p, shape)]
            if (d not in caps and len(visible) >= 2
                    and (allow_clip or len(visible) == radius + 1) and rng.random() < .35):
                caps[d] = int(rng.choice(payloads))
    e = dict(center=center, radius=radius, structural=structural, caps=caps)
    strokes = {}
    for d in directions:
        depth = _available_depth(center, radius, d, shape)
        count = int(rng.integers(1, min(4, depth) + 1))
        if far:
            count = max(3, count)
        ts = list(map(int, rng.choice(np.arange(1, depth + 1), count, replace=False)))
        if far and max(ts) < 6:
            ts[-1] = depth
        cap = _cap(center, radius, d)
        inks = (palette[7], palette[8] if rng.random() < .4 else palette[7])
        for t in ts:
            for endpoint, ink in zip((cap[0], cap[-1]), inks):
                if not _paint(strokes, _cross(_step(endpoint, d, t)), ink, shape):
                    return None
        unused = [t for t in range(1, depth + 1) if t not in ts]
        if unused and rng.random() < .8:
            p = _step(cap[0], d, int(rng.choice(unused)))
            if not _paint(strokes, _cross(p), inks[0], shape):
                return None
        if rng.random() < .6:
            p = _step(cap[len(cap) // 2], d, int(rng.choice(ts)))
            if not _paint(strokes, _cross(p), palette[9], shape):
                return None
        if rng.random() < .65:
            p = _step(cap[0], d, ts[0])
            _paint(strokes, [(p[0], p[1] + 2 * d[1])], inks[0], shape)
    source = _source_assignments(e, shape)
    if source is None or set(source) & set(strokes):
        return None
    return e, strokes


def _scene(rng, max_radius, count, stable, far=False, allow_clip=True):
    palette = _palette(rng, stable)
    low = (12, 21, 26, 30)[count - 1]
    if far:
        low = 25
    for restart in range(12):
        shape = tuple(map(int, rng.integers(low, 31, size=2))) if restart < 7 else (30, 30)
        specs, strokes, reserved = [], {}, set()
        for i in range(count):
            for _ in range(90):
                radius = int(rng.integers(2, max_radius + 1))
                candidate = _candidate(rng, shape, radius, palette, far=far, allow_clip=allow_clip)
                if candidate is None:
                    continue
                e, local = candidate
                cells = set(_source_assignments(e, shape)) | set(local) | set(_interior(e["center"], radius))
                # Different receivers may overlap within one emitter's scene,
                # but separate source groups have a one-cell isolation moat.
                moat = {q for p in cells for q in _cross(p)}
                if moat & reserved:
                    continue
                specs.append(e)
                strokes.update(local)
                reserved.update(cells)
                break
            else:
                break
        if len(specs) == count:
            pair = _render(shape, palette, specs, strokes)
            if pair is not None:
                return pair
    # Finite, rule-preserving fallback retains the emitter count. Free layouts
    # are attempted first; the fallback uses separated outward-facing objects.
    shape = (30, 30)
    centers = ((7, 7), (7, 22), (22, 22), (22, 7))
    specs, strokes = [], {}
    for center, d in zip(centers[:count], DIRECTIONS):
        e = dict(center=center, radius=2, structural=palette[2], caps={d: palette[3]})
        specs.append(e)
        cap = _cap(center, 2, d)
        for t in ((1, 2, 5, 6) if far else (1, 2)):
            for p in (cap[0], cap[-1]):
                if not _paint(strokes, _cross(_step(p, d, t)), palette[7], shape):
                    # Long-reach fallback points the single emitter inward.
                    if far:
                        return _far_fallback(palette)
                    raise AssertionError("fallback geometry")
    pair = _render(shape, palette, specs, strokes)
    assert pair is not None
    return pair


def _far_fallback(palette):
    e = dict(center=(5, 5), radius=2, structural=palette[2], caps={(1, 1): palette[3]})
    strokes = {}
    cap = _cap(e["center"], 2, (1, 1))
    for t in (1, 2, 6, 10):
        for p in (cap[0], cap[-1]):
            assert _paint(strokes, _cross(_step(p, (1, 1), t)), palette[7], (30, 30))
    pair = _render((30, 30), palette, [e], strokes)
    assert pair is not None
    return pair


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _integer(num_train, "num_train", 2, 10)
    ntest = _integer(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    max_radius = int(rng.integers(2, 5))
    stable = tuple(map(int, rng.permutation(10))) if rng.random() < .2 else None
    order = list(rng.permutation(4))
    cut = int(rng.integers(1, 4))
    teaching = ([DIRECTIONS[int(i)] for i in order[:cut]],
                [DIRECTIONS[int(i)] for i in order[cut:]])
    far_shift = rng.random() < .3
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            for attempt in range(60):
                if split == "train" and i < 2:
                    pair = _witness(rng, max_radius, teaching[i], i, stable)
                else:
                    n = (1 if far_shift else int(rng.integers(2, 5))) if split == "test" else int(rng.integers(1, 4))
                    allow_clip = split == "train" or any(_features(p)["clipped"] for p in puzzle["train"])
                    pair = _scene(rng, max_radius, n, stable, far=split == "test" and far_shift,
                                  allow_clip=allow_clip)
                if pair is not None and _key(pair["input"]) not in seen:
                    # Inspect teaching counterexamples after all stamps interact.
                    if split == "train" and i < 2:
                        f = _features(pair)
                        if not f["unpaired"] or not f["interior_plus"] or (i == 0 and not f["overlap"]):
                            continue
                        if i == 1 and not f["broken_pair"]:
                            continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique scene construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(ntrain)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    g = _grid(pair["input"])
    a = _analyze(g)
    matches = _matches(a, g.shape)
    selected = {p for m in matches for p in m[3:5]}
    claimed = _claims(matches, a["owned"])
    independent = set()
    interior_plus, broken_pair = False, False
    directions = set()
    for e in a["emitters"]:
        for d in e["caps"]:
            cap = _cap(e["center"], e["radius"], d)
            for t in range(1, max(g.shape)):
                ps = [_step(cap[k], d, t) for k in (0, -1)]
                independent.update(p for p in ps if p in a["centers"])
                if any(p in a["centers"] for p in ps) and all(_inside(p, g.shape) for p in ps):
                    for p in ps:
                        if p not in a["centers"]:
                            pts = _cross(p)
                            v = int(g[p])
                            if v not in a["background"] and all(_inside(q, g.shape) for q in pts):
                                broken_pair |= sum(g[q] == v for q in pts) == 4
                interior_plus |= any(_step(p, d, t) in a["centers"] and _step(p, d, t) not in selected
                                     for p in cap[1:-1])
    for m in matches:
        directions.add(m[1])
    selected_union = {q for p in selected for q in _cross(p)}
    receiver_cells = set(map(tuple, np.argwhere(g != a["base"]))) - a["owned"]
    attached = any(q in receiver_cells - selected_union for p in selected_union for q in _cross(p)[1:])
    return dict(shape=g.shape, background=a["background"], count=len(a["emitters"]),
                radii=tuple(e["radius"] for e in a["emitters"]),
                masks=tuple(sum(1 << DIRECTIONS.index(d) for d in e["caps"]) for e in a["emitters"]),
                matches=len(matches), depths=tuple(m[2] for m in matches), directions=directions,
                pluses=len(a["centers"]), selected=len(selected),
                overlap=len(selected_union) < 5 * len(selected),
                unpaired=bool(independent - selected), interior_plus=interior_plus,
                broken_pair=broken_pair, attached=attached,
                clipped=sum(any(not _inside(p, g.shape) for p in _cap(e["center"], e["radius"], d))
                            for e in a["emitters"] for d in e["caps"]),
                changed=int(np.count_nonzero(g != pair["output"])),
                geometry=(g.shape, tuple((e["center"], e["radius"], tuple(e["caps"])) for e in a["emitters"]),
                          tuple(sorted(a["centers"]))), claims=claimed)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training_features = set(), []
    max_test_radius = 2
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            a = _analyze(g)
            f = _features({"input": g, "output": out})
            assert np.array_equal(out, solve(g))
            assert np.array_equal(out, _direct_prediction(g, a))
            assert f["changed"] > 0 and f["matches"] > 0
            assert all(out[p] == g[p] for p in a["owned"])
            # Every changed cell is an addressed receiver cell; rays themselves
            # are not painted, and all unrelated strokes/field pixels survive.
            assert all(tuple(p) in f["claims"] for p in np.argwhere(g != out))
            assert all(out[p] == color for p, color in f["claims"].items())
            if split == "train":
                training_features.append(f)
            else:
                max_test_radius = max(max_test_radius, max(f["radii"]))
    assert set().union(*(f["directions"] for f in training_features)) == set(DIRECTIONS)
    assert any(f["overlap"] for f in training_features)
    assert any(f["unpaired"] for f in training_features)
    assert any(f["interior_plus"] for f in training_features)
    assert any(f["broken_pair"] for f in training_features)
    assert any(f["attached"] for f in training_features)
    assert any(len(set(f["depths"])) >= 2 for f in training_features)
    for d in DIRECTIONS:
        assert max(max(f["radii"]) for f in training_features if d in f["directions"]) >= max_test_radius
    if any(_features(p)["clipped"] for p in puzzle["test"]):
        assert any(f["clipped"] for f in training_features), "teach clipped caps before testing them"


def self_test():
    """A bounded 42-seed diversity audit, rule regressions, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, shapes, geometries, radii, masks, counts, palettes = set(), set(), set(), set(), set(), set(), set()
    totals = Counter()
    for puzzle in puzzles:
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                digest.update(repr(_key(pair["input"])).encode())
                shapes.add(f["shape"])
                geometries.add(f["geometry"])
                radii.update(f["radii"])
                masks.update(f["masks"])
                counts.add(f["count"])
                palettes.add(f["background"])
                totals[split + "_pairs"] += 1
                for name in ("overlap", "unpaired", "interior_plus", "broken_pair", "attached", "clipped"):
                    totals[name] += bool(f[name])
                totals["long_match"] += any(t >= 6 for t in f["depths"])
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(geometries) >= 150
    assert radii == {2, 3, 4} and counts == {1, 2, 3, 4}
    assert len(masks) >= 10 and len(palettes) >= 30
    assert all(totals[name] >= 20 for name in ("overlap", "unpaired", "interior_plus", "broken_pair", "attached"))
    assert totals["long_match"] >= 5
    assert any(min(s) <= 16 for s in shapes) and any(30 in s for s in shapes)
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    # Every local binary neighborhood is checked by the production detector.
    for bits in range(512):
        g = np.array([2 if bits >> i & 1 else 0 for i in range(9)], dtype=np.int8).reshape(3, 3)
        expected = all(g[p] == 2 for p in _cross((1, 1)))
        assert ((1, 1) in _centers(g, (0, 1))) == expected
    # Equality of diagonal distance matters: moving one endpoint's receiver
    # forward by one step deactivates its partner as well.
    palette = tuple(range(10))
    e = dict(center=(5, 5), radius=3, structural=2, caps={(1, 1): 3})
    cap = _cap(e["center"], 3, (1, 1))
    for offset in (0, 1):
        strokes = {}
        for p, t in ((cap[0], 3), (cap[-1], 3 + offset)):
            assert _paint(strokes, _cross(_step(p, (1, 1), t)), 7, (20, 20))
        parity = np.indices((20, 20)).sum(axis=0) % 2
        g = np.asarray(palette[:2], dtype=np.int8)[parity]
        for p, color in {**_source_assignments(e, g.shape), **strokes}.items():
            g[p] = color
        out = solve(g)
        assert np.count_nonzero(g != out) == (10 if offset == 0 else 0)
    # Count boundaries, strict typing, and deliberate wrong-label rejection.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(1000 + 10 * ntrain + ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, True), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    p = damaged["test"][0]
    point = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][point] = p["input"][point]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a deleted selected cell")
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                geometries=len(geometries), radii=sorted(radii), counts=sorted(counts),
                cap_masks=len(masks), backgrounds=len(palettes), observed=dict(totals))


def _fingerprint(puzzle):
    return tuple((_key(_grid(pair["input"])), _key(_grid(pair["output"])))
                 for s in ("train", "test") for pair in puzzle[s])


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
