"""Alternating crease fabric: input-only reconstruction and defect marking.

No I/O, parent imports, cached puzzles, or generation metadata is used by solve.
All checks and measurements here are author-owned, not independent approval.
"""
from __future__ import annotations

import hashlib
import numpy as np

BACKGROUND, CREASE, HIGHLIGHT = 0, 5, 8
MATERIALS = (1, 2, 3, 4, 6, 7, 9)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with sides 1..30")
    return a.astype(np.int8, copy=False)


def _strokes(grid):
    """Only the intact gray creases determine geometry and orientation."""
    points = np.argwhere(grid == CREASE)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in points:
            lines.setdefault(int(c - slope*r), []).append(int(r))
        strokes = []
        for q, rows in sorted(lines.items()):
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                strokes.append((q, 2*run[0] + slope*q,
                                2*run[-1] + slope*q))
        if len(strokes) >= 3 and all(hi-lo >= 2 for _, lo, hi in strokes):
            candidates.append((slope, strokes))
    if len(candidates) != 1:
        raise ValueError("Creases must have one recognizable diagonal slope")
    return candidates[0]


def _geometry(shape, slope, strokes, variant=""):
    rows, cols = np.indices(shape)
    q = cols - slope*rows
    t = rows + slope*cols
    left = np.zeros(shape, dtype=np.int16)
    low = np.full(shape, 100, dtype=np.int16)
    high = np.full(shape, -100, dtype=np.int16)
    active_count = np.zeros(shape, dtype=np.int16)
    for sq, lo, hi in strokes:
        active = (lo <= t) & (t <= hi)
        if variant == "pixel_centers":
            active &= (t-sq) % 2 == 0
        low = np.where(active, np.minimum(low, sq), low)
        high = np.where(active, np.maximum(high, sq), high)
        active_count += active
        left += active & (sq < q)
    mask = (active_count >= 2) & (low < q) & (q < high)
    if variant == "consecutive":
        adjacent = np.zeros(shape, dtype=bool)
        ordered = sorted(strokes)
        for a, b in zip(ordered, ordered[1:]):
            adjacent |= ((max(a[1], b[1]) <= t) & (t <= min(a[2], b[2]))
                         & (a[0] < q) & (q < b[0]))
        mask &= adjacent
    if variant == "static_parity":
        left = sum((sq < q).astype(np.int16) for sq, _, _ in strokes)
    # The first open band is class 0, the second class 1, and so on.
    parity = (left - 1) % 2
    return mask, parity


def _neighbors(shape, point):
    r, c = point
    return [(rr, cc) for rr in range(max(0, r-1), min(shape[0], r+2))
            for cc in range(max(0, c-1), min(shape[1], c+2))
            if (rr, cc) != point]


def _compose(clean, holes, fills, reverse=False):
    centers = set(holes)
    demands = {}
    for p in reversed(holes) if reverse else holes:
        other = fills[1] if clean[p] == fills[0] else fills[0]
        for q in _neighbors(clean.shape, p):
            if q in centers or clean[q] == CREASE:
                continue
            if q in demands and demands[q] != other:
                raise ValueError("Incompatible non-center ring assignments")
            demands[q] = other
    out = clean.copy()
    for p, color in demands.items():
        out[p] = color
    for p in holes:
        out[p] = HIGHLIGHT
    return out


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors = set(map(int, np.unique(grid)))
    fills = tuple(sorted(colors - {BACKGROUND, CREASE}))
    if HIGHLIGHT in colors or len(fills) != 2 or BACKGROUND not in colors:
        raise ValueError("Input must contain background, creases, and two materials")
    slope, strokes = _strokes(grid)
    mask, parity = _geometry(grid.shape, slope, strokes)
    mask &= grid != CREASE
    scores = []
    for phase in (0, 1):
        proposed = np.array(fills, dtype=np.int8)[parity ^ phase]
        scores.append(int(np.count_nonzero(mask & (grid == proposed))))
    if scores[0] == scores[1]:
        raise ValueError("Material phase is ambiguous")
    phase = int(scores[1] > scores[0])
    mapping = tuple(fills[i ^ phase] for i in (0, 1))
    clean = np.zeros(grid.shape, dtype=np.int8)
    for role in (0, 1):
        region = mask & (parity == role)
        if region.sum() < 8 or np.count_nonzero(region & (grid == mapping[role])) < .72*region.sum():
            raise ValueError("Insufficient surviving evidence for each material")
        clean[region] = mapping[role]
    clean[grid == CREASE] = CREASE
    holes = [tuple(map(int, p)) for p in np.argwhere(mask & (grid == BACKGROUND))]
    if not holes:
        raise ValueError("At least one actual fabric erasure is required")
    out = _compose(clean, holes, fills)
    return dict(grid=grid, clean=clean, holes=holes, fills=fills, mapping=mapping,
                slope=slope, strokes=strokes, mask=mask, parity=parity, output=out)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive geometry, material phase, defect type, and rings from input alone."""
    return _analyze(input_grid)["output"]


def _counterfactual(a, variant):
    mask, parity = _geometry(a["grid"].shape, a["slope"], a["strokes"], variant)
    mask &= a["grid"] != CREASE
    clean = np.zeros(a["grid"].shape, dtype=np.int8)
    clean[mask] = np.array(a["mapping"], dtype=np.int8)[parity[mask]]
    clean[a["grid"] == CREASE] = CREASE
    holes = [tuple(map(int, p)) for p in np.argwhere(mask & (a["grid"] == 0))]
    try:
        return _compose(clean, holes, a["fills"])
    except ValueError:
        # Conflicting color demands already distinguish this alternative.
        return None


def _teaching(a):
    holes, clean = a["holes"], a["clean"]
    painted = {p for h in holes for p in _neighbors(clean.shape, h)} | set(holes)
    wrong = [tuple(map(int, p)) for p in np.argwhere(a["mask"] &
             (a["grid"] != 0) & (a["grid"] != clean))]
    speckles = [tuple(map(int, p)) for p in np.argwhere((clean == 0) & (a["grid"] != 0))]
    flags = dict(
        both_material_holes={int(clean[p]) for p in holes} == set(a["fills"]),
        adjacent_centers=any(clean[p] == clean[q] and max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                             for i, p in enumerate(holes) for q in holes[:i]),
        unmarked_wrong=any(p not in painted for p in wrong),
        unmarked_speckle=any(p not in painted for p in speckles),
    )
    for variant in ("consecutive", "pixel_centers", "static_parity"):
        other = _counterfactual(a, variant)
        flags[variant] = other is None or not np.array_equal(other, a["output"])
    # A parity change matters at an erased site, not just at intact cloth.
    _, static = _geometry(clean.shape, a["slope"], a["strokes"], "static_parity")
    changed = [int(static[p] != a["parity"][p]) for p in holes]
    flags["hole_phase_change"] = 0 in changed and 1 in changed
    return flags


def _sample_specs(rng, count, style, witness):
    gaps = rng.integers(6, 10, size=count-1) if witness else rng.integers(3, 8, size=count-1)
    qs = [0] + list(map(int, np.cumsum(gaps)))
    intervals = []
    for i, q in enumerate(qs):
        if witness:
            lo, hi = -int(rng.integers(10, 16)), int(rng.integers(10, 16))
            if i == 1:
                lo, hi = -int(rng.integers(2, 6)), int(rng.integers(3, 8))
        elif style == "staggered":
            center = int(round(-7 + 14*i/(count-1)))
            length = int(rng.integers(6, 11) if count <= 5 else rng.integers(5, 10))
            lo, hi = center-length+1, center+length-1
        elif style == "aligned":
            lo, hi = 0, 2*int(rng.integers(4, 13))
        else:
            length = int(rng.integers(4, 14))
            before = int(rng.integers(1, length-1))
            lo = -2*before
            hi = lo + 2*(length-1)
        lo += (q-lo) % 2
        hi -= (hi-q) % 2
        intervals.append((q, int(lo), int(hi)))
    if style == "staggered" and not witness and rng.integers(2):
        # Longitudinal order is independent of transverse rank.
        permutation = rng.permutation(count)
        intervals = [(q, intervals[int(j)][1] + (q-intervals[int(j)][1]) % 2,
                       intervals[int(j)][2] - (intervals[int(j)][2]-q) % 2)
                      for q, j in zip(qs, permutation)]
    return intervals


def _render(rng, specs, slope, fills, padding):
    points = []
    for q, lo, hi in specs:
        for t in range(lo, hi+1, 2):
            points.append(((t-slope*q)//2, (q+slope*t)//2))
    points = np.array(points, dtype=int)
    points -= points.min(axis=0)
    span = points.max(axis=0) + 1
    if np.any(span > 30):
        return None
    extra = np.array([int(rng.integers(0, min(padding, 30-int(n))+1)) for n in span])
    shape = tuple(map(int, span + extra))
    offset = np.array([int(rng.integers(0, int(e)+1)) for e in extra])
    points += offset
    grid = np.zeros(shape, dtype=np.int8)
    grid[points[:, 0], points[:, 1]] = CREASE
    actual_slope, strokes = _strokes(grid)
    if actual_slope != slope:
        return None
    mask, parity = _geometry(shape, slope, strokes)
    mask &= grid != CREASE
    if min(np.count_nonzero(mask & (parity == role)) for role in (0, 1)) < 18:
        return None
    clean = grid.copy()
    clean[mask] = np.array(fills, dtype=np.int8)[parity[mask]]
    return clean


def _damage(rng, clean, fills, witness, crowded):
    candidates = [tuple(map(int, p)) for p in np.argwhere(np.isin(clean, fills))]
    rng.shuffle(candidates)
    holes = []
    def add(p):
        if p in holes or p not in candidate_set:
            return False
        if any(clean[p] != clean[q] and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in holes):
            return False
        if sum(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in holes) >= 3:
            return False
        # At most a quarter of either material is erased.
        if sum(clean[q] == clean[p] for q in holes) + 1 > .20*np.count_nonzero(clean == clean[p]):
            return False
        holes.append(p)
        return True
    candidate_set = set(candidates)
    if witness:
        for color in fills:
            for p in candidates:
                if clean[p] == color and add(p):
                    break
            else:
                return None
        for p in list(holes):
            for q in _neighbors(clean.shape, p):
                if clean[q] == clean[p] and add(q):
                    break
            if len(holes) >= 3:
                break
        if len(holes) < 3:
            return None
    target = int(rng.integers(6, 14) if witness else rng.integers(5, 19) if crowded else rng.integers(2, 8))
    for p in candidates:
        if len(holes) >= target:
            break
        if add(p) and crowded and rng.random() < .30:
            around = _neighbors(clean.shape, p)
            rng.shuffle(around)
            for q in around:
                if clean[q] == clean[p] and add(q):
                    break
    inp = clean.copy()
    for p in holes:
        inp[p] = 0
    occupied = set(holes)
    wrong_count = int(rng.integers(1, 5)) if witness or rng.random() < .7 else 0
    for p in candidates:
        if wrong_count == 0:
            break
        if p in occupied:
            continue
        if sum(inp[q] != clean[q] for q in candidates if clean[q] == clean[p]) >= .26*np.count_nonzero(clean == clean[p]):
            continue
        inp[p] = fills[1] if clean[p] == fills[0] else fills[0]
        occupied.add(p)
        wrong_count -= 1
    outside = [tuple(map(int, p)) for p in np.argwhere(clean == 0)]
    rng.shuffle(outside)
    for p in outside[:int(rng.integers(2, 8) if witness else rng.integers(0, 7))]:
        inp[p] = int(rng.choice(fills))
    try:
        out = _compose(clean, holes, fills)
        a = _analyze(inp)
    except ValueError:
        return None
    if not np.array_equal(a["clean"], clean) or not np.array_equal(a["output"], out):
        return None
    if witness and not all(_teaching(a).values()):
        return None
    return dict(input=inp, output=out)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_fills = tuple(map(int, rng.choice(MATERIALS, 2, replace=False)))
    changing_palette = bool(rng.integers(2))
    base_slope = int(rng.choice([-1, 1]))
    regime = int(rng.integers(3))  # new count, new endpoint pattern, or joint transfer
    witness_index = int(rng.integers(nt))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index == witness_index
            test = split == "test"
            fills = tuple(map(int, rng.choice(MATERIALS, 2, replace=False))) if changing_palette else base_fills
            # Both directions are taught with even the minimum two examples.
            slope = base_slope * (-1 if index % 2 else 1) if not test else int(rng.choice([-1, 1]))
            for attempt in range(240):
                if witness:
                    n = int(rng.integers(3, 5)) if attempt < 120 else 3
                    style = "common"
                elif test and regime in (0, 2):
                    n = int(rng.integers(6, 8))
                    style = "staggered" if regime == 2 else str(rng.choice(["common", "aligned"]))
                else:
                    n = int(rng.integers(4 if test else 3, 6))
                    if test:
                        style = "staggered"
                    elif regime == 1:
                        style = str(rng.choice(["common", "aligned"]))
                    else:
                        style = str(rng.choice(["common", "aligned", "staggered"], p=[.45, .35, .20]))
                specs = _sample_specs(rng, n, style, witness)
                if test and regime in (1, 2) and max(lo for _, lo, _ in specs) <= min(hi for _, _, hi in specs):
                    continue
                clean = _render(rng, specs, slope, fills, 7 if test else 5)
                if clean is None:
                    continue
                pair = _damage(rng, clean, fills, witness, bool(test or rng.integers(2)))
                if pair is None or _signature(pair["input"]) in seen:
                    continue
                seen.add(_signature(pair["input"]))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_flags = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen, "Duplicate inputs"
            seen.add(key)
            a = _analyze(inp)
            assert np.array_equal(out, a["output"]), "Incorrect input-derived output"
            assert np.array_equal(out[inp == CREASE], inp[inp == CREASE]), "Creases are immutable"
            assert set(map(tuple, np.argwhere(out == HIGHLIGHT))) == set(a["holes"])
            assert np.array_equal(out, _compose(a["clean"], a["holes"], a["fills"], reverse=True))
            assert not np.array_equal(out, a["clean"]), "Omitting B must change the answer"
            if split == "train":
                train_flags.append(_teaching(a))
    # Inspect visible arrays rather than a sampler mode or a seed tag.
    for name in train_flags[0]:
        assert any(flags[name] for flags in train_flags), f"Missing training witness: {name}"


def _literal_geometry_check(a):
    """Slow cell-wise cross-check; distinct from vectorized label construction."""
    clean = np.zeros(a["grid"].shape, dtype=np.int8)
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            if a["grid"][r, c] == CREASE:
                clean[r, c] = CREASE
                continue
            q, t = c-a["slope"]*r, r+a["slope"]*c
            active = sorted(sq for sq, lo, hi in a["strokes"] if lo <= t <= hi)
            if len(active) >= 2 and active[0] < q < active[-1]:
                index = (sum(sq < q for sq in active)-1) % 2
                clean[r, c] = a["mapping"][index]
    return clean


def self_test():
    """42 seeds, analytical fixture, bad-label probes, and legal counts; no I/O."""
    # Hand calculation: / creases have q=r+c and t=r-c.
    fixture = np.zeros((9, 9), dtype=np.int8)
    for q, lo, hi in ((4, -4, 4), (8, -2, 2), (12, -4, 4)):
        for t in range(lo, hi+1, 2):
            fixture[(q+t)//2, (q-t)//2] = CREASE
    mask, parity = _geometry(fixture.shape, -1, _strokes(fixture)[1])
    # At t=3 the short middle crease is inactive: one material spans it.
    assert mask[5, 2] and parity[5, 2] == 0
    assert mask[6, 3] and parity[6, 3] == 0
    # At t=1 the middle crease is active: it switches the distal material.
    assert mask[5, 4] and parity[5, 4] == 1
    # A continuous scanline between actual marker centers still brackets.
    assert mask[4, 3] and parity[4, 3] == 0
    clean = np.ones((7, 9), dtype=np.int8)
    clean[:, 6:] = 2
    holes = [(0, 0), (0, 1), (3, 7)]
    rendered = _compose(clean, holes, (1, 2))
    assert rendered[0, 0] == rendered[0, 1] == HIGHLIGHT
    assert rendered[1, 0] == 2 and rendered[2, 6] == 1
    clean[1, 1] = CREASE
    assert _compose(clean, holes, (1, 2))[1, 1] == CREASE
    try:
        _compose(clean, [(3, 5), (3, 6)], (1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting rings accepted")
    hashes, shapes, layouts, slopes, counts, palettes = set(), set(), set(), set(), set(), set()
    clipped = crowded = staggered = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                assert np.array_equal(a["clean"], _literal_geometry_check(a))
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                digest.update(repr(pair["input"].shape).encode())
                digest.update(pair["input"].tobytes())
                digest.update(pair["output"].tobytes())
                shapes.add(pair["input"].shape)
                slopes.add(a["slope"])
                counts.add(len(a["strokes"]))
                palettes.add(a["fills"])
                q0 = min(q for q, _, _ in a["strokes"])
                t0 = min(lo for _, lo, _ in a["strokes"])
                layouts.add(tuple((q-q0, lo-t0, hi-t0) for q, lo, hi in a["strokes"]))
                clipped += any(r in (0, a["grid"].shape[0]-1) or c in (0, a["grid"].shape[1]-1) for r, c in a["holes"])
                crowded += _teaching(a)["adjacent_centers"]
                staggered += max(lo for _, lo, _ in a["strokes"]) > min(hi for _, _, hi in a["strokes"])
        hashes.add(digest.hexdigest())
        repeat = generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], repeat[s]) for k in x)
    assert len(hashes) == 42 and len(shapes) >= 45 and len(layouts) >= 160, (len(hashes), len(shapes), len(layouts))
    assert slopes == {-1, 1} and {3, 4, 5, 6, 7} <= counts and len(palettes) >= 15, (slopes, counts, len(palettes))
    # Strict brackets of fully visible strokes lie inside their bounding box;
    # centers cannot lie on a canvas edge. Rings can reach it.
    assert clipped == 0 and crowded >= 42 and staggered >= 15, (clipped, crowded, staggered)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+nt*10+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect label accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(hashes),
            "shapes": len(shapes), "crease_layouts": len(layouts),
            "crease_counts": sorted(counts), "material_palettes": len(palettes),
            "clipped_pairs": clipped, "adjacent_center_pairs": crowded,
            "no_common_interval_pairs": staggered, "independent_semantic_check": False}
