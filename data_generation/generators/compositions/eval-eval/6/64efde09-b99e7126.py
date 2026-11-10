"""Matching-channel colonies: pooled optical codes become their own tile map.

Only NumPy and the standard library are used. No files, seeds, examples or
generation parameters are consulted by solve(). The generator writes nothing.
"""
from collections import Counter
import hashlib
from itertools import combinations

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _undo(a, t):
    b = np.rot90(a, -(t % 4))
    return (np.fliplr(b) if t >= 4 else b).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _normal(a):
    roles = {}
    return tuple(roles.setdefault(int(x), len(roles)) for x in a)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def _blank(h, w, background, double, single):
    tile = np.full((h + 2, w + 2), background, dtype=np.int8)
    tile[0, 0] = tile[0, -1] = double
    tile[-1, 0] = single
    return tile


def _glyph(rows, cols, background, double, single):
    tile = _blank(len(rows), len(cols), background, double, single)
    tile[1:-1, 0] = rows
    tile[0, 1:-1] = cols
    matches = np.asarray(rows)[:, None] == np.asarray(cols)[None, :]
    tile[1:-1, 1:-1] = np.where(matches, np.asarray(rows)[:, None], background)
    return tile


def _origins(mask, nr, nc, supplied):
    h, w = mask.shape
    return [(r, c) for r in range(nr - h + 1) for c in range(nc - w + 1)
            if all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c]
                   for y, x in supplied)]


def _symmetries(mask):
    return {t for t in range(8) if np.array_equal(_orient(mask, t), mask)}


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and i == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _lattice(a):
    divider = int(a[0, 0])
    ys = _runs(np.flatnonzero(np.all(a == divider, axis=1)))
    xs = _runs(np.flatnonzero(np.all(a == divider, axis=0)))
    assert len(ys) >= 3 and len(xs) >= 3
    assert ys[0][0] == xs[0][0] == 0
    assert ys[-1][1] == a.shape[0]-1 and xs[-1][1] == a.shape[1]-1
    thickness = {b-a+1 for a, b in ys + xs}
    assert len(thickness) == 1
    t = thickness.pop()
    assert t in (1, 2)
    heights = {ys[i+1][0]-ys[i][1]-1 for i in range(len(ys)-1)}
    widths = {xs[i+1][0]-xs[i][1]-1 for i in range(len(xs)-1)}
    assert len(heights) == len(widths) == 1
    ph, pw = heights.pop(), widths.pop()
    assert 4 <= ph <= 6 and 4 <= pw <= 6
    return ph, pw, t, len(ys)-1, len(xs)-1, divider


def _infer(input_grid):
    """Decode geometry, pool local edge observations, then register their mask."""
    a = _grid(input_grid)
    ph, pw, t, nr, nc, divider = _lattice(a)
    colors, counts = np.unique(a[a != divider], return_counts=True)
    assert len(colors) >= 5 and np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[np.argmax(counts)])
    records, role_pair, canonical_shape = [], None, None
    for r in range(nr):
        for c in range(nc):
            y, x = t+r*(ph+t), t+c*(pw+t)
            tile = a[y:y+ph, x:x+pw]
            assert divider not in tile
            corners = [int(tile[i, j]) for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
            freq = Counter(corners)
            assert sorted(freq.values()) == [1, 1, 2] and freq[background] == 1
            double = next(k for k, v in freq.items() if v == 2)
            single = next(k for k, v in freq.items() if v == 1 and k != background)
            if role_pair is None:
                role_pair = double, single
            assert role_pair == (double, single)
            candidates = []
            for transform in range(8):
                b = _undo(tile, transform)
                if (int(b[0, 0]), int(b[0, -1]), int(b[-1, 0]), int(b[-1, -1])) == (
                        double, double, single, background):
                    candidates.append((transform, b))
            assert len(candidates) == 1, "socket compass must identify exactly one transform"
            transform, b = candidates[0]
            h, w = b.shape[0]-2, b.shape[1]-2
            assert 2 <= h <= 4 and 2 <= w <= 4 and (h, w) != (2, 2)
            if canonical_shape is None:
                canonical_shape = h, w
            assert canonical_shape == (h, w)
            assert np.all(b[1:-1, 1:-1] == background), "input payloads must be blank"
            assert np.all(b[-1, 1:-1] == background) and np.all(b[1:-1, -1] == background)
            observed_rows = b[1:-1, 0].copy()
            observed_cols = b[0, 1:-1].copy()
            observed = list(map(int, observed_rows)) + list(map(int, observed_cols))
            assert all(v == background or v not in (divider, double, single) for v in observed)
            records.append((r, c, y, x, transform, b, observed_rows, observed_cols))
    h, w = canonical_shape
    assert nr >= h and nc >= w
    row_code, col_code = [None]*h, [None]*w
    supplied, allocations = [], []
    for r, c, _, _, _, _, rr, cc in records:
        known = 0
        for target, source in ((row_code, rr), (col_code, cc)):
            for i, value in enumerate(map(int, source)):
                if value == background:
                    continue
                assert target[i] is None or target[i] == value, "conflicting local channel clues"
                target[i] = value
                known += 1
        if known:
            supplied.append((r, c))
            allocations.append(known)
    assert all(v is not None for v in row_code + col_code), "every channel slot needs evidence"
    assert 2 <= len(set(row_code + col_code)) <= 4
    assert 2 <= len(supplied) < nr*nc-len(supplied), "ordinary sockets must have strict majority"
    mask = np.asarray(row_code)[:, None] == np.asarray(col_code)[None, :]
    assert 4 <= int(mask.sum()) < h*w
    origins = _origins(mask, nr, nc, supplied)
    assert len(origins) == 1, "colony translation must be unique; no overlap tie breaker"
    origin = origins[0]
    active = {(origin[0]+int(i), origin[1]+int(j)) for i, j in np.argwhere(mask)}
    assert set(supplied) < active
    glyph = _glyph(row_code, col_code, background, *role_pair)
    expected = a.copy()
    transforms = []
    single_copy_losses = []
    for r, c, y, x, transform, b, rr, cc in records:
        if (r, c) in active:
            replacement = _orient(glyph, transform)
            assert replacement.shape == (ph, pw)
            expected[y:y+ph, x:x+pw] = replacement
        if (r, c) in supplied:
            transforms.append(transform)
            partial = (rr[:, None] == cc[None, :]) & (rr[:, None] != background)
            single_copy_losses.append(bool(np.any(mask & ~partial)))
    return expected, {
        "shape": a.shape, "tile": (h, w), "physical_tile": (ph, pw),
        "thickness": t, "lattice": (nr, nc), "origin": origin,
        "mask": _key(mask), "mask_array": mask, "code": _normal(row_code+col_code),
        "channel_count": len(set(row_code+col_code)),
        "ink_colors": len({row_code[int(i)] for i, j in np.argwhere(mask)}),
        "visible": len(supplied), "missing": len(active)-len(supplied),
        "active": len(active), "supplied": tuple(supplied), "allocations": tuple(allocations),
        "pooled": max(allocations) < h+w, "single_copy_losses": tuple(single_copy_losses),
        "transforms": tuple(transforms), "background": background, "divider": divider,
        "roles": (divider, background, *role_pair), "palette": tuple(map(int, np.unique(a))),
        "glyph": glyph, "records": records, "active_positions": active,
    }


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the completed colony using only the supplied input pixels."""
    return _infer(input_grid)[0]


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5) if (h, w) != (2, 2))


def _geometry(rng, h, w):
    options = []
    for swap in (False, True):
        ph, pw = (w+2, h+2) if swap else (h+2, w+2)
        for t in (1, 2):
            maxr, maxc = (30-t)//(ph+t), (30-t)//(pw+t)
            if maxr >= h and maxc >= w:
                options.append((ph, pw, t, maxr, maxc, swap))
    ph, pw, t, maxr, maxc, swap = options[int(rng.integers(len(options)))]
    nr, nc = int(rng.integers(h, maxr+1)), int(rng.integers(w, maxc+1))
    return ph, pw, t, nr, nc, swap


def _fragment(rng, mask, nr, nc, origin, training):
    active = [tuple(map(int, p)) for p in np.argwhere(mask)]
    sizes = list(range(3, min(len(active)-1, (nr*nc-1)//2, mask.shape[0]+mask.shape[1])+1)) if training else [2]
    rng.shuffle(sizes)
    for size in sizes:
        # At most C(12,6)=924 candidates at one size for this equality family.
        choices = list(combinations(active, size))
        for index in rng.permutation(len(choices)):
            local = choices[int(index)]
            absolute = [(r+origin[0], c+origin[1]) for r, c in local]
            if _origins(mask, nr, nc, absolute) == [origin]:
                return absolute
    return None


def _assemble(rng, rows, cols, geometry, origin, supplied, palette, pool, first):
    ph, pw, t, nr, nc, swap = geometry
    divider, background, double, single = map(int, palette[:4])
    h, w = len(rows), len(cols)
    blank = _blank(h, w, background, double, single)
    glyph = _glyph(rows, cols, background, double, single)
    mask = np.asarray(rows)[:, None] == np.asarray(cols)[None, :]
    active = {(origin[0]+int(r), origin[1]+int(c)) for r, c in np.argwhere(mask)}
    m, slots = len(supplied), h+w
    clues = [set() for _ in supplied]
    order = list(map(int, rng.permutation(slots)))
    if pool:
        owners = list(map(int, rng.permutation(m)))
        for i, slot in enumerate(order):
            clues[owners[i % m]].add(slot)
    else:
        clues[0] = set(range(slots))
        for i in range(1, m):
            clues[i].add(int(rng.integers(slots)))
    # Optional agreeing redundant observations, without destroying required pooling.
    for slot in range(slots):
        if rng.random() < .25:
            owner = int(rng.integers(m))
            if not pool or len(clues[owner] | {slot}) < slots:
                clues[owner].add(slot)
    allowed = list(range(8)) if h == w else [1, 3, 5, 7] if swap else [0, 2, 4, 6]
    transforms = {(r, c): int(rng.choice(allowed)) for r in range(nr) for c in range(nc)}
    if first:
        # The first demonstration contains a reflection and a quarter turn.
        for p, transform in zip(supplied, (0, 4, 1)):
            transforms[p] = transform
    partial = {}
    for p, known in zip(supplied, clues):
        b = blank.copy()
        for slot in known:
            if slot < h:
                b[slot+1, 0] = rows[slot]
            else:
                b[0, slot-h+1] = cols[slot-h]
        partial[p] = b
    a = np.full((nr*(ph+t)+t, nc*(pw+t)+t), divider, dtype=np.int8)
    out = a.copy()
    for r in range(nr):
        for c in range(nc):
            p = r, c
            y, x = t+r*(ph+t), t+c*(pw+t)
            a[y:y+ph, x:x+pw] = _orient(partial.get(p, blank), transforms[p])
            out[y:y+ph, x:x+pw] = _orient(glyph if p in active else blank, transforms[p])
    return {"input": a, "output": out}


def _make_pair(rng, training, index, forbidden_mask=None, _random_attempts=128):
    first = training and index == 0
    second = training and index == 1
    for attempt in range(_random_attempts+1):
        fallback = attempt == _random_attempts
        h, w = (3, 3) if first else _SHAPES[int(rng.integers(len(_SHAPES)))]
        if fallback:
            h, w = (3, 3) if first else (4, 4)
        palette = list(map(int, rng.permutation(10)))
        k = 2 if first else int(rng.integers(2, 5))
        labels = rng.integers(k, size=h+w)
        if fallback:
            labels = np.array([0, 0, 1, 0, 1, 0] if first else [0, 0, 0, 1, 0, 0, 1, 1])
        rows = np.asarray(palette[4:8], dtype=np.int8)[labels[:h]]
        cols = np.asarray(palette[4:8], dtype=np.int8)[labels[h:]]
        mask = rows[:, None] == cols[None, :]
        if not (4 <= int(mask.sum()) < h*w and 2 <= len(set(map(int, np.r_[rows, cols]))) <= 4):
            continue
        if first and (len(set(map(int, rows))) < 2 or set(map(int, rows)) != set(map(int, cols))):
            continue
        if second and (_key(mask) == forbidden_mask or _symmetries(mask) != {0}):
            continue
        geometry = _geometry(rng, h, w)
        if fallback:
            geometry = (h+2, w+2, 1, h, w, False)
        ph, pw, t, nr, nc, swap = geometry
        origin = int(rng.integers(nr-h+1)), int(rng.integers(nc-w+1))
        supplied = _fragment(rng, mask, nr, nc, origin, training)
        if supplied is None:
            continue
        pool = first or second or not training or bool(rng.random() < .8)
        return _assemble(rng, rows, cols, geometry, origin, supplied, palette, pool, first)
    raise RuntimeError("constructive fallback failed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, used, first_mask = {"train": [], "test": []}, set(), None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(32):
                pair = _make_pair(rng, split == "train", i, first_mask)
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("input uniqueness retry bound exceeded")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train" and i == 0:
                first_mask = _infer(pair["input"])[1]["mask"]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            prediction, f = _infer(a)
            assert np.array_equal(prediction, b), "output must complete channels, matching pixels and macro copies"
            if split == "train":
                assert f["visible"] >= 3
                train.append(f)
            else:
                assert f["visible"] == 2
    assert len({f["mask"] for f in train}) >= 2
    assert any(f["pooled"] and all(f["single_copy_losses"]) and f["ink_colors"] >= 2 for f in train)
    assert any(any(t % 2 for t in f["transforms"]) and any(t >= 4 for t in f["transforms"])
               and f["tile"][0] == f["tile"][1] for f in train)
    symmetry_rules = set(range(8))
    for f in train:
        symmetry_rules &= _symmetries(f["mask_array"])
    assert symmetry_rules == {0}, "demonstrations must distinguish canonical macro orientation"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain small fingerprints/counters, never puzzle dumps."""
    puzzles, shapes, tiles, geometries, masks, codes, palettes, transforms = (set() for _ in range(8))
    totals, visible, missing, origins = Counter(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                predicted, f = _infer(pair["input"])
                assert np.array_equal(before, pair["input"])
                assert np.array_equal(predicted, pair["output"])
                shapes.add(f["shape"])
                tiles.add(f["tile"])
                geometries.add((f["physical_tile"], f["thickness"], f["lattice"]))
                masks.add(f["mask"])
                codes.add((f["tile"], f["code"]))
                palettes.add(f["roles"])
                transforms.update(f["transforms"])
                visible.add(f["visible"])
                missing.add(f["missing"])
                origins.add(f["origin"])
                totals["pairs"] += 1
                totals["pooled"] += f["pooled"]
                totals["full_code_on_one"] += not f["pooled"]
                totals["requires_macro_completion"] += f["missing"] > 0
                totals["two_or_more_ink_colors"] += f["ink_colors"] >= 2
                totals["redundant_clues"] += sum(f["allocations"]) > sum(f["tile"])
                totals["two_cell_dividers"] += f["thickness"] == 2
                totals["rectangular_stencil"] += f["tile"][0] != f["tile"][1]
                totals["interior_origin_axis"] += any(0 < f["origin"][i] < f["lattice"][i]-f["tile"][i] for i in (0, 1))
                totals["no_zero"] += 0 not in f["palette"]
    assert len(puzzles) == 42 and len(shapes) >= 20 and len(geometries) >= 35
    assert len(tiles) == 8 and len(masks) >= 50 and len(codes) >= 80
    assert len(palettes) >= 100 and transforms == set(range(8))
    assert totals["requires_macro_completion"] == 210
    assert all(totals[k] > 0 for k in ("full_code_on_one", "interior_origin_axis", "two_cell_dividers", "no_zero"))
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(7001, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    p = generate_puzzle(0, 2, 1)
    permutation = np.random.default_rng(95).permutation(10)
    recolored = {s: [{k: permutation[np.asarray(v)] for k, v in pair.items()} for pair in p[s]]
                 for s in ("train", "test")}
    validate_generated_puzzle(recolored)
    # Exercise construction when every random attempt has been exhausted.
    fallback_rng = np.random.default_rng(96)
    fallback = {"train": [_make_pair(fallback_rng, True, 0, _random_attempts=0),
                           _make_pair(fallback_rng, True, 1, _random_attempts=0)],
                "test": [_make_pair(fallback_rng, False, 0, _random_attempts=0)]}
    validate_generated_puzzle(fallback)
    # A missing colony tile must receive rim labels as well as its inner stencil.
    pair = p["test"][0]
    _, f = _infer(pair["input"])
    absent = f["active_positions"] - set(f["supplied"])
    r, c, y, x, transform, *_ = next(rec for rec in f["records"] if rec[:2] in absent)
    damaged = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in ("train", "test")}
    marker = np.zeros_like(f["glyph"], dtype=bool)
    marker[0, 1:-1] = True
    dy, dx = np.argwhere(_orient(marker, transform))[0]
    damaged["test"][0]["output"][y+int(dy), x+int(dx)] = f["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("ink-only copying error accepted")
    # Remove every observation of one local row: the input must become undecodable.
    broken = p["train"][0]["input"].copy()
    _, f = _infer(broken)
    ph, pw = f["physical_tile"]
    for _, _, y, x, transform, canonical, *_ in f["records"]:
        canonical = canonical.copy()
        canonical[1, 0] = f["background"]
        broken[y:y+ph, x:x+pw] = _orient(canonical, transform)
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("unobserved channel slot was guessed")
    # Add a contradictory observation of an already evidenced row slot.
    broken = p["train"][0]["input"].copy()
    _, f = _infer(broken)
    known_row = next(int(rec[5][1, 0]) for rec in f["records"]
                     if int(rec[5][1, 0]) != f["background"])
    other_channel = next(int(color) for color in np.r_[f["glyph"][1:-1, 0], f["glyph"][0, 1:-1]]
                         if int(color) != known_row)
    record = next(rec for rec in f["records"] if rec[:2] not in f["supplied"])
    _, _, y, x, transform, canonical, *_ = record
    canonical = canonical.copy()
    canonical[1, 0] = other_channel
    ph, pw = f["physical_tile"]
    broken[y:y+ph, x:x+pw] = _orient(canonical, transform)
    try:
        solve(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting observations were accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "grid_shapes": len(shapes),
            "geometries": len(geometries), "stencil_shapes": sorted(tiles),
            "masks": len(masks), "normalized_codes": len(codes), "role_palettes": len(palettes),
            "visible_counts": sorted(visible), "missing_counts": sorted(missing),
            "origins": sorted(origins), "forced_fallback_pairs": 3, "totals": dict(totals)}
