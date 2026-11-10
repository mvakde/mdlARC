"""Endpoint-seeded stencil growth. All inference uses only the input grid.

An ink coordinate (i,j) is a coarse displacement (sr*i,sc*j), where
(sr,sc) is the outward quadrant of the diamond cap. Newly copied tiles
repeat the displacement program; only original cap endpoints initiate it.
"""
from collections import Counter, deque
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _runs(values):
    groups = []
    for raw in values:
        i = int(raw)
        if groups and i == groups[-1][1] + 1:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _inside(p, shape):
    return all(0 <= p[i] < shape[i] for i in (0, 1))


def _outline(center, radius):
    r, c = center
    return {(r + i, c + j) for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1) if abs(i) + abs(j) == radius}


def _cap(center, radius, direction):
    r, c = center
    sr, sc = direction
    return [(r + sr * (i + 1), c + sc * (radius - i + 1))
            for i in range(radius + 1)]


def _parse(grid):
    """Decode the lattice, ordinary tile, diamond and each local cap program."""
    divider = int(grid[0, 0])
    row_bands = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    col_bands = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    assert len(row_bands) >= 3 and len(col_bands) >= 3
    assert row_bands[0][0] == col_bands[0][0] == 0
    assert row_bands[-1][1] == grid.shape[0] - 1
    assert col_bands[-1][1] == grid.shape[1] - 1
    thicknesses = {b - a + 1 for a, b in row_bands + col_bands}
    heights = {row_bands[i + 1][0] - row_bands[i][1] - 1
               for i in range(len(row_bands) - 1)}
    widths = {col_bands[i + 1][0] - col_bands[i][1] - 1
              for i in range(len(col_bands) - 1)}
    assert len(thicknesses) == len(heights) == len(widths) == 1
    t, h, w = thicknesses.pop(), heights.pop(), widths.pop()
    assert t in (1, 2) and 2 <= h <= 3 and 2 <= w <= 3
    nr, nc = len(row_bands) - 1, len(col_bands) - 1
    tiles = {}
    records = {}
    for r in range(nr):
        for c in range(nc):
            y, x = row_bands[r][1] + 1, col_bands[c][1] + 1
            tile = grid[y:y + h, x:x + w].copy()
            assert divider not in tile
            tiles[r, c] = tile
            records.setdefault(_key(tile), [tile, []])[1].append((r, c))
    ordinary_record = max(records.values(), key=lambda item: len(item[1]))
    ordinary, ordinary_positions = ordinary_record
    assert len(ordinary_positions) > nr * nc / 2, "ordinary tiles need a strict majority"
    ordinary_colors = set(map(int, np.unique(ordinary)))
    assert len(ordinary_colors) in (2, 3)
    solid_records = [rec for rec in records.values() if len(np.unique(rec[0])) == 1]
    assert len(solid_records) == 1
    solid, diamond_positions = solid_records[0]
    structural = int(solid[0, 0])
    assert structural not in ordinary_colors
    diamond_positions = set(diamond_positions)
    # A complete Manhattan outline determines its center and radius uniquely.
    r0, r1 = min(r for r, c in diamond_positions), max(r for r, c in diamond_positions)
    c0, c1 = min(c for r, c in diamond_positions), max(c for r, c in diamond_positions)
    assert r1 - r0 == c1 - c0 and (r1 - r0) % 2 == 0
    radius = (r1 - r0) // 2
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    assert radius in (2, 3) and diamond_positions == _outline(center, radius)
    ordinary_set = set(ordinary_positions)
    caps, occupied = [], set(diamond_positions)
    program_records = {}
    for direction in DIRECTIONS:
        full = _cap(center, radius, direction)
        visible = [p for p in full if _inside(p, (nr, nc))]
        exceptional = [p for p in visible if p not in ordinary_set]
        if not exceptional:
            continue
        assert len(visible) >= 2 and len(exceptional) == len(visible)
        tile = tiles[visible[0]]
        assert all(np.array_equal(tiles[p], tile) for p in visible)
        novel = set(map(int, np.unique(tile))) - ordinary_colors
        assert len(novel) == 1
        ink = novel.pop()
        assert ink not in (structural, divider)
        mask = tile == ink
        assert not mask[0, 0] and mask.any() and not mask.all()
        assert len(np.unique(tile[~mask])) == 1
        assert int(tile[~mask][0]) in ordinary_colors
        # The ink role is a program identity within this input.
        assert ink not in program_records or _key(tile) == program_records[ink]
        program_records[ink] = _key(tile)
        assert not occupied.intersection(visible)
        occupied.update(visible)
        offsets = tuple(tuple(map(int, p)) for p in np.argwhere(mask))
        endpoints = [p for p in (full[0], full[-1]) if _inside(p, (nr, nc))]
        caps.append(dict(direction=direction, tile=tile, ink=ink, mask=mask,
                         offsets=offsets, full=full, visible=visible, endpoints=endpoints))
    assert caps and occupied == set(tiles) - ordinary_set
    return dict(geometry=(h, w, t, nr, nc), divider=divider, ordinary=ordinary,
                structural=structural, center=center, radius=radius, caps=caps,
                occupied=occupied, tiles=tiles)


def _growth(info, *, all_cap_cells=False, recursive=True, diagonal=False,
            transpose=False):
    """Finite directed reachability; alternate modes are teaching diagnostics."""
    shape = info["geometry"][3:]
    blocked = info["occupied"]
    claims, depths = {}, {}
    for cap in info["caps"]:
        sr, sc = cap["direction"]
        offsets = ((1, 1),) if diagonal else cap["offsets"]
        if transpose:
            offsets = tuple((j, i) for i, j in offsets)
        tile_id = _key(cap["tile"])
        roots = cap["visible"] if all_cap_cells else cap["endpoints"]
        visited = set(roots)
        queue = deque((p, 0) for p in roots)
        while queue:
            (r, c), depth = queue.popleft()
            for i, j in offsets:
                p = (r + sr * i, c + sc * j)
                if not _inside(p, shape) or p in blocked:
                    continue
                assert p not in claims or _key(claims[p]) == tile_id, "incompatible growth"
                claims[p] = cap["tile"]
                depths[p] = min(depths.get(p, depth + 1), depth + 1)
                if recursive and p not in visited:
                    visited.add(p)
                    queue.append((p, depth + 1))
    return claims, depths


def _render_growth(grid, info, claims, ink_only=False):
    out = grid.copy()
    h, w, t, _, _ = info["geometry"]
    for (r, c), tile in claims.items():
        y, x = t + r * (h + t), t + c * (w + t)
        if ink_only:
            mask = next(cap["mask"] for cap in info["caps"]
                        if np.array_equal(cap["tile"], tile))
            view = out[y:y + h, x:x + w]
            view[mask] = tile[mask]
        else:
            out[y:y + h, x:x + w] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the saturated endpoint-seeded stencil growth from input alone."""
    grid = _check_grid(input_grid)
    info = _parse(grid)
    claims, _ = _growth(info)
    return _render_growth(grid, info, claims)


def _fixed_point_claims(info):
    """Second closure calculation, using simultaneous sets rather than a queue."""
    shape = info["geometry"][3:]
    claims = {}
    for cap in info["caps"]:
        sr, sc = cap["direction"]
        roots = set(cap["endpoints"])
        reached = roots.copy()
        for _ in range(shape[0] * shape[1] + 1):
            nxt = {(r + sr * i, c + sc * j) for r, c in reached
                   for i, j in cap["offsets"]}
            nxt = {p for p in nxt if _inside(p, shape) and p not in info["occupied"]}
            expanded = reached | nxt
            if expanded == reached:
                break
            reached = expanded
        else:
            raise AssertionError("closure did not saturate")
        for p in reached - roots:
            assert p not in claims or np.array_equal(claims[p], cap["tile"])
            claims[p] = cap["tile"]
    return claims


def _evidence(grid, info=None):
    info = _parse(grid) if info is None else info
    claims, depths = _growth(info)
    out = _render_growth(grid, info, claims)
    alternatives = {
        "all_cap": _growth(info, all_cap_cells=True)[0],
        "one_step": _growth(info, recursive=False)[0],
        "diagonal": _growth(info, diagonal=True)[0],
        "transpose": _growth(info, transpose=True)[0],
    }
    evidence = {name: not np.array_equal(out, _render_growth(grid, info, alt))
                for name, alt in alternatives.items()}
    evidence["ink_only"] = not np.array_equal(out, _render_growth(grid, info, claims, True))
    evidence["depth"] = max(depths.values(), default=0)
    # Count directions that have observable offspring from an original end.
    evidence["directions"] = {cap["direction"] for cap in info["caps"] if any(
        _inside((r + cap["direction"][0] * i, c + cap["direction"][1] * j),
                info["geometry"][3:])
        and (r + cap["direction"][0] * i, c + cap["direction"][1] * j)
        not in info["occupied"] for r, c in cap["endpoints"] for i, j in cap["offsets"])}
    return evidence


def _paint(geometry, divider, ordinary, structural, center, radius, programs):
    h, w, t, nr, nc = geometry
    tiles = np.tile(ordinary, (nr, nc, 1, 1))
    # np.tile's result is (nr,nc,h,w) for a 2D tile and four repeat factors.
    for p in _outline(center, radius):
        assert _inside(p, (nr, nc))
        tiles[p] = structural
    for direction, glyph in programs.items():
        for p in _cap(center, radius, direction):
            if _inside(p, (nr, nc)):
                tiles[p] = glyph
    grid = np.full((nr * (h + t) + t, nc * (w + t) + t), divider, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = t + r * (h + t), t + c * (w + t)
            grid[y:y + h, x:x + w] = tiles[r, c]
    return grid


def _random_mask(rng, shape):
    n = shape[0] * shape[1]
    mask = np.zeros(shape, dtype=bool)
    count = int(rng.integers(1, n))
    mask.flat[rng.choice(np.arange(1, n), count, replace=False)] = True
    return mask


def _ordinary(rng, shape, colors):
    n, k = shape[0] * shape[1], len(colors)
    labels = np.concatenate((np.arange(k), rng.integers(k, size=n - k)))
    rng.shuffle(labels)
    return np.asarray(colors, dtype=np.int8)[labels.reshape(shape)]


def _make_pair(rng, *, directions=None, mixed=False, witness=False, radius=2,
               forbidden_masks=(), stable_palette=None):
    for attempt in range(240):
        h, w = ((2, 2) if attempt >= 200 else
                ((2, 2), (2, 3), (3, 2), (3, 3))[int(rng.choice(4, p=[.4, .25, .25, .1]))])
        t = 1 if witness or radius == 3 else int(rng.choice([1, 2], p=[.85, .15]))
        maxr, maxc = (30 - t) // (h + t), (30 - t) // (w + t)
        if min(maxr, maxc) < 2 * radius + 1:
            continue
        nr = maxr if witness else int(rng.integers(2 * radius + 1, maxr + 1))
        nc = maxc if witness else int(rng.integers(2 * radius + 1, maxc + 1))
        center = (int(rng.integers(radius, nr - radius)),
                  int(rng.integers(radius, nc - radius)))
        chosen = directions
        if chosen is None:
            count = int(rng.integers(2 if mixed else 1, 5))
            chosen = tuple(DIRECTIONS[i] for i in sorted(map(int, rng.choice(4, count, replace=False))))
        if any(len([p for p in _cap(center, radius, d) if _inside(p, (nr, nc))]) < 2
               for d in chosen):
            continue
        if witness and any(not all(_inside(p, (nr, nc)) for p in _cap(center, radius, d))
                           for d in chosen):
            continue
        k = int(rng.integers(2, min(4, len(chosen)) + 1)) if mixed else 1
        masks = []
        for _ in range(k):
            for _ in range(40):
                mask = _random_mask(rng, (h, w))
                if _key(mask) not in set(forbidden_masks) | {_key(m) for m in masks}:
                    masks.append(mask)
                    break
            else:
                break
        if len(masks) != k:
            continue
        nordinary = 3 if h * w >= 6 and rng.random() < .3 else 2
        palette = (stable_palette if stable_palette is not None else
                   tuple(map(int, rng.permutation(10))))
        divider, structural = palette[:2]
        ordinary_colors = palette[2:2 + nordinary]
        inks = palette[2 + nordinary:2 + nordinary + k]
        ordinary = _ordinary(rng, (h, w), ordinary_colors)
        glyphs = []
        for mask, ink in zip(masks, inks):
            backgrounds = [bg for bg in ordinary_colors if np.any(ordinary[~mask] != bg)]
            bg = int(rng.choice(backgrounds))
            glyphs.append(np.where(mask, ink, bg).astype(np.int8))
        ids = list(range(k)) + list(map(int, rng.integers(k, size=len(chosen) - k)))
        rng.shuffle(ids)
        programs = {d: glyphs[i] for d, i in zip(chosen, ids)}
        grid = _paint((h, w, t, nr, nc), divider, ordinary, structural, center, radius, programs)
        try:
            info = _parse(grid)
            evidence = _evidence(grid, info)
            if not _growth(info)[0] or not evidence["all_cap"] or not evidence["diagonal"]:
                continue
            if witness and (evidence["directions"] != set(chosen)
                            or not all(evidence[name] for name in
                                       ("one_step", "ink_only", "transpose"))):
                continue
            # At test, every distinct cap program has at least one offspring.
            if mixed:
                claims = _growth(info)[0]
                if {_key(tile) for tile in claims.values()} != {_key(g) for g in glyphs}:
                    continue
            return {"input": grid, "output": solve(grid)}
        except AssertionError:
            continue
    # Simple constructive fallback: 9x9 lattice, horizontal/diagonal branching.
    # Each quadrant's endpoints have room for a second generation. It retains
    # the requested direction set and mixed-program requirement.
    chosen = directions or (DIRECTIONS[0], DIRECTIONS[2], DIRECTIONS[3])
    palette = stable_palette or tuple(map(int, rng.permutation(10)))
    ordinary = np.array([[palette[2], palette[3]], [palette[3], palette[2]]], dtype=np.int8)
    masks = (np.array([[0, 1], [0, 1]], dtype=bool),
             np.array([[0, 0], [1, 1]], dtype=bool))
    programs = {}
    for i, d in enumerate(chosen):
        j = i % 2 if mixed else int(_key(masks[0]) in forbidden_masks)
        programs[d] = np.where(masks[j], palette[4 + j], palette[3]).astype(np.int8)
    grid = _paint((2, 2, 1, 9, 9), palette[0], ordinary, palette[1], (4, 4), radius, programs)
    return {"input": grid, "output": solve(grid)}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = tuple(map(int, rng.permutation(10))) if rng.random() < .25 else None
    radius = 3 if rng.random() < .22 else 2
    # Two complementary, shuffled witnesses cover every outward quadrant.
    partitions = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    ids = partitions[int(rng.integers(len(partitions)))]
    groups = (tuple(DIRECTIONS[i] for i in ids),
              tuple(DIRECTIONS[i] for i in range(4) if i not in ids))
    slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    witnesses = {slots[i]: groups[i] for i in range(2)}
    puzzle, seen, train_masks = {"train": [], "test": []}, set(), set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            witness = split == "train" and index in witnesses
            pair_radius = radius if witness else int(rng.integers(2, radius + 1))
            for _ in range(32):
                pair = _make_pair(rng, directions=witnesses.get(index) if witness else None,
                                  mixed=split == "test", witness=witness, radius=pair_radius,
                                  forbidden_masks=train_masks if witness else (),
                                  stable_palette=stable_palette)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_masks.update(_key(cap["mask"]) for cap in _parse(pair["input"])["caps"])
                    break
            else:
                raise RuntimeError("bounded unique-input search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check actual pixels, closure, preservation, teaching and split shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, directions, masks, direction_radii = set(), set(), set(), set()
    evidence = Counter()
    max_train_radius = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            claims, _ = _growth(info)
            simultaneous = _fixed_point_claims(info)
            assert set(claims) == set(simultaneous)
            assert all(np.array_equal(claims[p], simultaneous[p]) for p in claims)
            assert np.array_equal(out, _render_growth(inp, info, simultaneous))
            assert np.array_equal(out, solve(inp))
            h, w, t, nr, nc = info["geometry"]
            changed_tiles = set()
            allowed_pixels = np.zeros(inp.shape, dtype=bool)
            for p, tile in info["tiles"].items():
                r, c = p
                y, x = t + r * (h + t), t + c * (w + t)
                actual = out[y:y + h, x:x + w]
                if p in claims:
                    assert np.array_equal(actual, claims[p])
                    assert np.any((actual != tile) & (actual != next(
                        cap["ink"] for cap in info["caps"] if np.array_equal(cap["tile"], actual))))
                    allowed_pixels[y:y + h, x:x + w] = True
                    changed_tiles.add(p)
                else:
                    assert np.array_equal(actual, tile)
            assert changed_tiles == set(claims)
            assert np.array_equal(out[~allowed_pixels], inp[~allowed_pixels])
            # Closure is saturated: every feasible outgoing target is present.
            for cap in info["caps"]:
                sr, sc = cap["direction"]
                roots = set(cap["endpoints"])
                # Identical programs on different sides retain local direction:
                # only inspect this side's directed reachability here.
                local = _fixed_point_claims({**info, "caps": [cap]})
                for r, c in roots | set(local):
                    for i, j in cap["offsets"]:
                        p = (r + sr * i, c + sc * j)
                        if _inside(p, (nr, nc)) and p not in info["occupied"]:
                            assert p in local
            ev = _evidence(inp, info)
            assert ev["all_cap"] and ev["diagonal"], "each pair must use both parent concepts"
            if split == "train":
                assert len({_key(cap["tile"]) for cap in info["caps"]}) == 1
                directions.update(ev["directions"])
                direction_radii.update((d, info["radius"]) for d in ev["directions"])
                masks.update(_key(cap["mask"]) for cap in info["caps"])
                for name in ("all_cap", "one_step", "diagonal", "transpose", "ink_only"):
                    evidence[name] += ev[name]
                max_train_radius = max(max_train_radius, info["radius"])
            else:
                assert len({_key(cap["tile"]) for cap in info["caps"]}) >= 2
                assert {_key(tile) for tile in claims.values()} == {
                    _key(cap["tile"]) for cap in info["caps"]}
                assert info["radius"] <= max_train_radius
                assert all(max(r for d, r in direction_radii if d == direction)
                           >= info["radius"] for direction in DIRECTIONS)
    assert directions == set(DIRECTIONS)
    assert len(masks) >= 2
    assert all(evidence[name] for name in ("all_cap", "one_step", "diagonal", "transpose", "ink_only"))


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        all(np.array_equal(x[k], y[k]) for k in ("input", "output"))
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed semantic and structural checks, including mutations."""
    # Hand-calculated fixtures test the intended rule without generating labels.
    ordinary = np.array([[2, 1], [1, 2]], dtype=np.int8)
    glyph = np.array([[1, 7], [1, 7]], dtype=np.int8)
    fixture = _paint((2, 2, 1, 9, 9), 0, ordinary, 3, (4, 4), 2,
                     {(1, 1): glyph})
    expected_positions = {(5, 8), (6, 8), (7, 6), (8, 6),
                          (7, 7), (8, 7), (7, 8), (8, 8)}
    info = _parse(fixture)
    assert set(_growth(info)[0]) == expected_positions
    assert np.array_equal(solve(fixture), _render_growth(
        fixture, info, {p: glyph for p in expected_positions}))
    assert (6, 7) not in _growth(info)[0]  # Would be born from the dormant middle.
    clipped_fixture = _paint((2, 2, 1, 9, 9), 0, ordinary, 3, (2, 4), 2,
                             {(-1, -1): glyph})
    clipped_info = _parse(clipped_fixture)
    assert clipped_info["caps"][0]["endpoints"] == [(1, 1)]
    assert set(_growth(clipped_info)[0]) == {(1, 0), (0, 0)}
    # (0,2) is a visible interior cell, not a replacement geometric endpoint.
    assert (0, 1) not in _growth(clipped_info)[0]
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _equal(puzzles[0], generate_puzzle(0))
    shapes, geometries, masks, layouts, motifs, cap_counts = set(), set(), set(), set(), set(), set()
    radii, depths, clipped, mixed_counts, colors = set(), set(), 0, set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                shapes.add(pair["input"].shape)
                geometries.add(info["geometry"])
                masks.update(_key(cap["mask"]) for cap in info["caps"])
                motifs.add(_normalized(info["ordinary"]))
                radii.add(info["radius"])
                cap_counts.add(len(info["caps"]))
                mixed_counts.add(len({_key(cap["tile"]) for cap in info["caps"]}))
                depths.add(_evidence(pair["input"], info)["depth"])
                clipped += sum(len(cap["visible"]) < len(cap["full"]) for cap in info["caps"])
                colors.update(map(int, np.unique(pair["input"])))
                layouts.add((info["geometry"], info["center"], info["radius"],
                             tuple((c["direction"], _key(c["mask"])) for c in info["caps"])))
    assert len(shapes) >= 10 and len(geometries) >= 15
    assert len(masks) >= 35 and len(layouts) >= 120 and len(motifs) >= 30
    assert radii == {2, 3} and cap_counts == {1, 2, 3, 4}
    assert {1, 2, 3} <= mixed_counts and max(depths) >= 4 and clipped >= 20
    assert colors == set(range(10))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(4100 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The solver must not mutate its argument.
    inp = puzzles[0]["test"][0]["input"].copy()
    before = inp.copy()
    solve(inp)
    assert np.array_equal(inp, before)
    for kind in ("divider", "missing_copy", "ink_only"):
        bad = {s: [{k: np.asarray(g).copy() for k, g in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
        pair = bad["test"][0]
        info = _parse(pair["input"])
        if kind == "divider":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "missing_copy":
            p = next(iter(_growth(info)[0]))
            h, w, t, _, _ = info["geometry"]
            y, x = t + p[0] * (h + t), t + p[1] * (w + t)
            pair["output"][y:y + h, x:x + w] = pair["input"][y:y + h, x:x + w]
        else:
            pair["output"] = _render_growth(pair["input"], info, _growth(info)[0], True)
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"validator accepted {kind}")
    return {"seeds": 42, "shapes": len(shapes), "geometries": len(geometries),
            "masks": len(masks), "layouts": len(layouts), "ordinary_motifs": len(motifs),
            "clipped_caps": clipped, "max_generation": max(depths)}
