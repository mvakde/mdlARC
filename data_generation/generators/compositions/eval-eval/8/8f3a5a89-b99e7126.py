"""A repeated floor-plan tile surveys a room; rebuild its exterior rim in tiles.

Only Python's standard library and NumPy are used. No file IO, saved examples,
parent imports, cached puzzles, or hidden solver parameters are required.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    todo = [start]
    h, w = allowed.shape
    while todo:
        r, c = todo.pop()
        for dr, dc in N4:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and allowed[y, x] and not reached[y, x]:
                reached[y, x] = True
                todo.append((y, x))
    return reached


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        start = tuple(map(int, np.argwhere(remaining)[0]))
        component = _flood(remaining, start)
        remaining[component] = False
        result.append(component)
    return result


def _neighbors(mask, offsets=N8):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr, dc in offsets:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _topology(walls, seed):
    region = _flood(~walls, seed)
    if not region.any():
        raise ValueError("marker must project onto floor")
    exterior = _flood(np.pad(~region, 1, constant_values=True), (0, 0))
    rim = region & _neighbors(exterior)[1:-1, 1:-1]
    orth = region & _neighbors(exterior, N4)[1:-1, 1:-1]
    islands = walls & ~exterior[1:-1, 1:-1]
    return {"region": region, "rim": rim, "islands": islands,
            "diagonal": rim & ~orth,
            "hole_witness": region & ~rim & _neighbors(islands)}


def _origins(floor, lattice, supplied):
    h, w = floor.shape
    nr, nc = lattice
    return [(r, c) for r in range(nr - h + 1) for c in range(nc - w + 1)
            if all(0 <= y - r < h and 0 <= x - c < w and floor[y - r, x - c]
                   for y, x in supplied)]


def _runs(indices):
    groups = []
    for n in map(int, indices):
        if groups and groups[-1][-1] + 1 == n:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _decode(value):
    grid = _grid(value)
    candidates = []
    for color in map(int, np.unique(grid)):
        rows = _runs(np.flatnonzero(np.all(grid == color, axis=1)))
        cols = _runs(np.flatnonzero(np.all(grid == color, axis=0)))
        if not rows or not cols:
            continue
        thicknesses = {len(g) for g in rows + cols}
        if len(thicknesses) != 1:
            continue
        t = thicknesses.pop()
        ys = [(a, b) for a, b in zip([0] + [g[-1] + 1 for g in rows],
                                    [g[0] for g in rows] + [grid.shape[0]])]
        xs = [(a, b) for a, b in zip([0] + [g[-1] + 1 for g in cols],
                                    [g[0] for g in cols] + [grid.shape[1]])]
        if (t in (1, 2) and len({b - a for a, b in ys}) == 1
                and len({b - a for a, b in xs}) == 1
                and 3 <= ys[0][1] <= 5 and 3 <= xs[0][1] <= 5):
            candidates.append((color, t, ys, xs))
    if len(candidates) != 1:
        raise ValueError("expected one regular internal divider lattice, without an outer frame")
    divider, t, ys, xs = candidates[0]
    records = {}
    for r, (a, b) in enumerate(ys):
        for c, (d, e) in enumerate(xs):
            tile = grid[a:b, d:e]
            if np.any(tile == divider):
                raise ValueError("divider color inside tile")
            records.setdefault(tile.tobytes(), [tile.copy(), []])[1].append((r, c))
    if len(records) != 3:
        raise ValueError("expected ordinary, floor-plan, and marker tile types")
    solids = [rec for rec in records.values() if len(np.unique(rec[0])) == 1]
    if len(solids) != 1 or len(solids[0][1]) != 1:
        raise ValueError("expected exactly one solid marker tile")
    marker_tile, marker_positions = solids[0]
    marker = int(marker_tile.flat[0])
    repeated = sorted((rec for rec in records.values() if rec is not solids[0]),
                      key=lambda rec: len(rec[1]), reverse=True)
    (ordinary, ordinary_positions), (glyph, supplied) = repeated
    if not len(ordinary_positions) > len(supplied) >= 2:
        raise ValueError("ordinary pattern must be strictly modal; at least two plan copies are needed")
    ordinary_colors = set(map(int, np.unique(ordinary)))
    novel = set(map(int, np.unique(glyph))) - ordinary_colors
    if len(ordinary_colors) not in (2, 3) or len(novel) != 1:
        raise ValueError("floor plan must have exactly one novel wall ink")
    wall_color = novel.pop()
    walls = glyph == wall_color
    if (not walls.any() or walls.all() or len(np.unique(glyph[~walls])) != 1
            or int(glyph[~walls][0]) not in ordinary_colors
            or marker in ordinary_colors | set(map(int, np.unique(glyph))) | {divider}):
        raise ValueError("ambiguous floor, wall, or marker roles")
    origins = _origins(~walls, (len(ys), len(xs)), supplied)
    if len(origins) != 1:
        raise ValueError("landmarks must uniquely register the floor plan without using the marker")
    origin = origins[0]
    mr, mc = marker_positions[0]
    local_seed = (mr - origin[0], mc - origin[1])
    h, w = walls.shape
    if not (0 <= local_seed[0] < h and 0 <= local_seed[1] < w and not walls[local_seed]):
        raise ValueError("marker lies outside the registered floor")
    if len(supplied) >= int((~walls).sum()):
        raise ValueError("survey must be a strict partial floor stencil")
    top = _topology(walls, local_seed)
    target = {(origin[0] + int(r), origin[1] + int(c)) for r, c in np.argwhere(top["rim"])}
    target.discard((mr, mc))
    return {"grid": grid, "divider": divider, "thickness": t, "ys": ys, "xs": xs,
            "ordinary": ordinary, "glyph": glyph, "walls": walls, "marker": marker,
            "marker_pos": (mr, mc), "seed": local_seed, "origin": origin,
            "lattice": (len(ys), len(xs)), "supplied": set(supplied), "target": target,
            **top}


def _put(grid, info, position, tile):
    r, c = position
    a, b = info["ys"][r]
    d, e = info["xs"][c]
    grid[a:b, d:e] = tile


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer lattice, plan, translation, selected room, and exterior rim from input."""
    info = _decode(input_grid)
    out = info["grid"].copy()
    for p in info["supplied"] - info["target"]:
        _put(out, info, p, info["ordinary"])
    for p in info["target"] - info["supplied"]:
        _put(out, info, p, info["glyph"])
    return out


def _paint(geometry, divider, ordinary, glyph, marker_color, marker_pos, positions):
    h, w, t, nr, nc = geometry
    grid = np.full((nr * (h + t) - t, nc * (w + t) - t), divider, np.int8)
    for r in range(nr):
        for c in range(nc):
            tile = marker_color if (r, c) == marker_pos else glyph if (r, c) in positions else ordinary
            grid[r * (h + t):r * (h + t) + h, c * (w + t):c * (w + t) + w] = tile
    return grid


def _transform(mask, rng):
    result = np.rot90(mask, int(rng.integers(4)))
    return (np.fliplr(result) if rng.integers(2) else result).copy()


def _wall_map(rng, kind):
    if kind == "island_witness":
        walls = np.zeros((5, 5), bool)
        walls[:, 3] = True
        walls[2, 1] = walls[0 if rng.integers(2) else 4, 0] = True
        return _transform(walls, rng)
    if kind == "selection_witness":
        # A bent separator with unequal, two-dimensional rooms.
        return _transform(np.array([[0, 1, 0, 0], [0, 1, 1, 0],
                                    [0, 0, 1, 0], [0, 0, 1, 0]], bool), rng)
    h, w = map(int, rng.integers(3, 6, size=2))
    walls = np.zeros((h, w), bool)
    mode = int(rng.integers(3))
    if mode == 0:
        # Boundary intrusions and islands, possibly with several free rooms.
        walls = rng.random((h, w)) < float(rng.uniform(.12, .37))
    elif mode == 1:
        col = int(rng.integers(1, w - 1))
        previous = col
        for r in range(h):
            if rng.random() < .45:
                col = int(rng.integers(1, w - 1))
            walls[r, min(col, previous):max(col, previous) + 1] = True
            previous = col
        # Independently placed extra cells may become lobes or islands.
        walls |= rng.random((h, w)) < .10
    else:
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            walls[r, c] = True
            if rng.random() < .5:
                walls[r, c:min(w, c + int(rng.integers(1, 3)))] = True
    return _transform(walls, rng)


def _fragment(rng, walls, geometry, origin, seed, top, witness):
    h, w, _, nr, nc = geometry
    eligible_rim = [tuple(map(int, p)) for p in np.argwhere(top["rim"]) if tuple(p) != seed]
    if len(eligible_rim) < 2:
        return None
    omitted = eligible_rim[int(rng.integers(len(eligible_rim)))]
    cells = [tuple(map(int, p)) for p in np.argwhere(~walls) if tuple(p) not in (seed, omitted)]
    mandatory = set()
    if witness:
        for mask in ((~walls) & ~top["region"], top["rim"]):
            choices = [i for i, p in enumerate(cells) if mask[p]]
            if not choices:
                return None
            mandatory.add(int(rng.choice(choices)))
    if witness == 1:
        choices = [i for i, p in enumerate(cells) if top["hole_witness"][p]]
        if not choices:
            return None
        mandatory.add(int(rng.choice(choices)))
    absolute = [(origin[0] + r, origin[1] + c) for r, c in cells]
    covers = []
    for r in range(nr - h + 1):
        for c in range(nc - w + 1):
            if (r, c) != origin:
                covers.append(sum(1 << i for i, (y, x) in enumerate(absolute)
                                  if 0 <= y - r < h and 0 <= x - c < w and not walls[y - r, x - c]))
    def unique(code):
        return all(code & cover != code for cover in covers)
    code = (1 << len(cells)) - 1
    if not unique(code):
        return None
    # Random deletion gives different inclusion-minimal registering fragments;
    # then random additions vary clue density without losing registration.
    for i in map(int, rng.permutation(len(cells))):
        trial = code & ~(1 << i)
        if i not in mandatory and trial.bit_count() >= 2 and unique(trial):
            code = trial
    maximum = min(len(cells), (nr * nc - 2) // 2, max(3, int((~walls).sum()) // 2))
    minimum = max(2, code.bit_count())
    if minimum > maximum:
        return None
    count = int(rng.integers(minimum, maximum + 1))
    missing = [i for i in range(len(cells)) if not (code >> i) & 1]
    if count > code.bit_count():
        for i in rng.choice(missing, count - code.bit_count(), replace=False):
            code |= 1 << int(i)
    return {absolute[i] for i in range(len(cells)) if (code >> i) & 1}


def _pair(rng, palette, kind="random", fallback=False):
    walls = _wall_map(rng, kind)
    if fallback:
        walls = np.zeros((5, 5), bool)
        walls[:, 1] = True
        walls[0, 4] = True
        walls = _transform(walls, rng)
    h, w = walls.shape
    regions = _components(~walls)
    usable = [m for m in regions if int(m.sum()) >= 4]
    if not walls.any() or not usable:
        return None
    witness = 1 if kind == "island_witness" else 2 if kind == "selection_witness" else 0
    if witness and (len(regions) < 2 or len({int(m.sum()) for m in regions}) < 2):
        return None
    region = (max(usable, key=lambda m: int(m.sum())) if witness == 1
              else min(usable, key=lambda m: int(m.sum())) if witness == 2 else usable[int(rng.integers(len(usable)))])
    top = _topology(walls, tuple(map(int, np.argwhere(region)[0])))
    choices = np.argwhere(region & top["rim"] & ~top["diagonal"]) if witness == 1 else np.argwhere(region)
    seed = tuple(map(int, choices[int(rng.integers(len(choices)))]))
    if witness == 1 and not (top["hole_witness"].any() and top["diagonal"].any()):
        return None
    t = 1 if max(h, w) == 5 or rng.random() < .75 else 2
    nr, nc = int(rng.integers(h, (30 + t) // (h + t) + 1)), int(rng.integers(w, (30 + t) // (w + t) + 1))
    geometry = (h, w, t, nr, nc)
    origin = (int(rng.integers(nr - h + 1)), int(rng.integers(nc - w + 1)))
    visible = _fragment(rng, walls, geometry, origin, seed, top, witness)
    if visible is None:
        return None
    divider, bg, minor, ink, marker, third = palette
    colors = [bg, minor] + ([third] if rng.random() < .25 else [])
    labels = rng.integers(len(colors), size=h * w)
    labels[:len(colors)] = rng.permutation(len(colors))
    rng.shuffle(labels)
    ordinary = np.array(colors, np.int8)[labels.reshape(h, w)]
    if not np.any(ordinary[~walls] != bg):
        ordinary[tuple(np.argwhere(~walls)[0])] = minor
    glyph = np.where(walls, ink, bg).astype(np.int8)
    marker_pos = (origin[0] + seed[0], origin[1] + seed[1])
    target = {(origin[0] + int(r), origin[1] + int(c)) for r, c in np.argwhere(top["rim"])} - {marker_pos}
    inp = _paint(geometry, divider, ordinary, glyph, marker, marker_pos, visible)
    out = _paint(geometry, divider, ordinary, glyph, marker, marker_pos, target)
    return {"input": inp, "output": out}


def _relocate_marker(pair, rng):
    """Counterfactual: same survey, a different room, a different exterior rim."""
    info = _decode(pair["input"])
    r0, c0 = info["origin"]
    options = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere((~info["walls"]) & ~info["region"])
               if (r0 + int(r), c0 + int(c)) not in info["supplied"]]
    if not options:
        return None
    new_pos = options[int(rng.integers(len(options)))]
    inp = info["grid"].copy()
    _put(inp, info, info["marker_pos"], info["ordinary"])
    _put(inp, info, new_pos, info["marker"])
    # Construct the target directly from the chosen local region, not solve().
    top = _topology(info["walls"], (new_pos[0] - r0, new_pos[1] - c0))
    target = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(top["rim"])} - {new_pos}
    geometry = (*info["walls"].shape, info["thickness"], *info["lattice"])
    out = _paint(geometry, info["divider"], info["ordinary"], info["glyph"], info["marker"], new_pos, target)
    return {"input": inp, "output": out}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples; defaults are exactly four training and one test."""
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    stable_palette = rng.random() < .3
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            kind = "island_witness" if split == "train" and i == 0 else "selection_witness" if split == "train" and i == 1 else "random"
            for attempt in range(80):
                roles = palette if stable_palette else tuple(map(int, rng.choice(10, 6, replace=False)))
                if split == "test" and i == 0:
                    pair = _relocate_marker(puzzle["train"][1], rng)
                else:
                    pair = _pair(rng, roles, kind, fallback=attempt >= 64 and kind != "island_witness")
                if pair is not None and _key(pair["input"]) not in seen:
                    # Reject degenerate construction draws, never alter labels.
                    info = _decode(pair["input"])
                    if info["target"] - info["supplied"] and info["supplied"] - info["target"]:
                        break
                pair = None
            if pair is None:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _features(pair):
    info = _decode(pair["input"])
    target, supplied = info["target"], info["supplied"]
    r0, c0 = info["origin"]
    exterior = supplied - target
    far = {p for p in supplied if not info["region"][p[0] - r0, p[1] - c0]}
    interior = {p for p in supplied if info["region"][p[0] - r0, p[1] - c0] and not info["rim"][p[0] - r0, p[1] - c0]}
    regions = _components(~info["walls"])
    shape = info["walls"].shape
    rr = [p[0] for p in supplied]
    cc = [p[1] for p in supplied]
    return {"shape": pair["input"].shape, "tile_shape": shape, "lattice": info["lattice"],
            "thickness": info["thickness"], "walls": _key(info["walls"]),
            "ordinary": _normalized(info["ordinary"]), "origin": info["origin"],
            "rooms": len(regions), "room_areas": tuple(sorted(int(m.sum()) for m in regions)),
            "selected_area": int(info["region"].sum()), "islands": len(_components(info["islands"])),
            "hole_witness": bool(info["hole_witness"].any()),
            "diagonal": int(info["diagonal"].sum()) - int(info["diagonal"][info["seed"]]),
            "added": len(target - supplied), "removed": len(exterior), "retained": len(target & supplied),
            "far_removed": len(far), "interior_removed": len(interior),
            "visible": len(supplied), "floor_count": int((~info["walls"]).sum()),
            "marker_on_rim": bool(info["rim"][info["seed"]]),
            "nonink_copy": bool(np.any(info["ordinary"][~info["walls"]] != info["glyph"][~info["walls"]])),
            "non_bbox_registration": max(rr) - min(rr) + 1 < shape[0] or max(cc) - min(cc) + 1 < shape[1]}


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def validate_generated_puzzle(puzzle):
    """Check the family rule, local roles, complete tile edits, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            before = inp.copy()
            predicted = solve(inp)
            assert np.array_equal(inp, before) and np.array_equal(out, predicted)
            # Re-render from tile sets to check whole-tile copying, restoration,
            # divider preservation, and the unchanged solid marker explicitly.
            geom = (*info["walls"].shape, info["thickness"], *info["lattice"])
            assert np.array_equal(out, _paint(geom, info["divider"], info["ordinary"], info["glyph"],
                                             info["marker"], info["marker_pos"], info["target"]))
            f = _features({"input": inp, "output": out})
            assert f["added"] > 0 and f["removed"] > 0 and f["nonink_copy"]
            if split == "train":
                train.append(f)
    assert any(f["hole_witness"] and f["interior_removed"] and f["far_removed"] and f["diagonal"] for f in train)
    assert any(f["rooms"] >= 2 and f["selected_area"] < max(f["room_areas"]) for f in train)
    assert any(f["rooms"] >= 2 and f["selected_area"] == max(f["room_areas"]) for f in train)
    assert len({f["walls"] for f in train}) >= 2
    assert len({f["ordinary"] for f in train}) >= 2
    # Verify the guaranteed counterfactual test in an order-independent way.
    assert any(_same_survey_other_room(a["input"], b["input"])
               for a in puzzle["train"] for b in puzzle["test"])


def _same_survey_other_room(left, right):
    a, b = _decode(left), _decode(right)
    return (a["origin"] == b["origin"] and a["marker"] == b["marker"]
            and a["divider"] == b["divider"] and a["thickness"] == b["thickness"]
            and a["lattice"] == b["lattice"] and a["supplied"] == b["supplied"]
            and _key(a["ordinary"]) == _key(b["ordinary"]) and _key(a["glyph"]) == _key(b["glyph"])
            and not np.array_equal(a["region"], b["region"]))


def _puzzle_digest(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded 42-seed structural/semantic diversity and metamorphic checks; no IO."""
    # Hand-specified topology: a divider, an island, and an edge intrusion.
    walls = np.zeros((5, 5), bool)
    walls[:, 3] = True
    walls[2, 1] = walls[0, 0] = True
    top = _topology(walls, (4, 0))
    expected = {(r, c) for r in range(5) for c in range(3)} - {(0, 0), (2, 1), (3, 1)}
    assert set(map(tuple, np.argwhere(top["rim"]))) == expected
    assert top["diagonal"][1, 1] and top["hole_witness"][3, 1]
    assert set(map(tuple, np.argwhere(top["islands"]))) == {(2, 1)}
    # Attaching the island to the exterior changes its neighboring rim.
    attached = walls.copy()
    attached[2, 2] = True
    assert _topology(attached, (4, 0))["rim"][3, 1]
    features, digests = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed))
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                features.append(_features(pair))
        # Floor-plan mosaics commute with whole-canvas rotations/reflections.
        pair = puzzle["test"][0]
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
        assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
    assert len(digests) == 42
    assert len({f["walls"] for f in features}) >= 50
    assert len({f["ordinary"] for f in features}) >= 80
    assert len({f["shape"] for f in features}) >= 15
    assert len({f["tile_shape"] for f in features}) >= 7
    assert {f["thickness"] for f in features} == {1, 2}
    assert len({f["rooms"] for f in features}) >= 3
    assert {f["marker_on_rim"] for f in features} == {True, False}
    assert any(f["non_bbox_registration"] for f in features)
    assert len({f["visible"] for f in features}) >= 5
    # Every legal count combination, with no retained puzzle collection.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            validate_generated_puzzle(p)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
