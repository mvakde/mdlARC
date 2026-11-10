"""Interlocking self-portraits: jointly assemble and register four stencils.

Only Python's standard library and NumPy are used. Inference never reads a
seed, another example, a file, or construction metadata.
"""
from math import isqrt

import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _normalize(points):
    p = np.asarray(points, dtype=int)
    p = p - p.min(axis=0)
    return tuple(sorted(map(tuple, p.tolist())))


def _rotate(shape, turns):
    p = np.asarray(shape, dtype=int)
    for _ in range(turns % 4):
        p = np.column_stack((p[:, 1], p[:, 0].max() - p[:, 0]))
    return _normalize(p)


def _connected(points):
    cells = set(points)
    if not cells:
        return False
    stack = [next(iter(cells))]
    visited = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS:
            q = (r + dr, c + dc)
            if q in cells and q not in visited:
                visited.add(q)
                stack.append(q)
    return visited == cells


def _corners(side):
    return ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))


def _mask(points, side):
    return sum(1 << (r * side + c) for r, c in points)


def _placements(shape, side, corner, rotate=True):
    candidates = {}
    for k in range(4 if rotate else 1):
        s = _rotate(shape, k)
        h = max(r for r, c in s) + 1
        w = max(c for r, c in s) + 1
        if h > side or w > side:
            continue
        y = 0 if corner < 2 else side - h
        x = 0 if corner % 2 == 0 else side - w
        cells = {(r + y, c + x) for r, c in s}
        hits = [q in cells for q in _corners(side)]
        if sum(hits) == 1 and hits[corner]:
            candidates.setdefault(_mask(cells, side), k)
    return tuple(candidates.items())


def _assemblies(shapes, side, rotate=True):
    """All corner packings, including geometrically ambiguous ones.

    Extreme-corner ownership fixes each rotated bounding box's translation.
    Bit masks deduplicate rotations of symmetric shapes without guessing.
    """
    if len(shapes) != 4 or sum(map(len, shapes)) != side * side:
        return []
    anchor = _placements(shapes[0], side, 0, rotate=False)
    if not anchor:
        return []
    table = {(i, c): _placements(shapes[i], side, c, rotate)
             for i in range(1, 4) for c in range(1, 4)}
    result = []
    full = (1 << (side * side)) - 1

    def visit(corner, remaining, occupied, masks, pieces, turns):
        if corner == 4:
            if occupied == full:
                result.append((tuple(masks), tuple(pieces), tuple(turns)))
            return
        for i in remaining:
            for bits, k in table[i, corner]:
                if not bits & occupied:
                    visit(corner + 1, tuple(j for j in remaining if j != i),
                          occupied | bits, masks + [bits], pieces + [i], turns + [k])

    visit(1, (1, 2, 3), anchor[0][0], [anchor[0][0]], [0], [0])
    return result


def _owners(assembly, side):
    """Return piece identity and corner identity at each assembled pixel."""
    piece = np.empty((side, side), dtype=np.int8)
    corner = np.empty_like(piece)
    masks, pieces, _ = assembly
    for c, bits in enumerate(masks):
        for j in range(side * side):
            if bits >> j & 1:
                piece.flat[j] = pieces[c]
                corner.flat[j] = c
    return piece, corner


def _joint(assemblies, side, rows, columns, positions):
    """Search packing AND common coarse translation, not successive solvers."""
    result = []
    for a in assemblies:
        owner, _ = _owners(a, side)
        for y in range(rows - side + 1):
            for x in range(columns - side + 1):
                if all(0 <= r-y < side and 0 <= c-x < side and owner[r-y, c-x] == i
                       for i, group in enumerate(positions) for r, c in group):
                    result.append((a, (y, x)))
    return result


def _runs(indices):
    groups = []
    for i in map(int, indices):
        if groups and groups[-1][1] + 1 == i:
            groups[-1] = (groups[-1][0], i)
        else:
            groups.append((i, i))
    return groups


def _parse(input_grid):
    grid = _grid(input_grid)
    divider = int(grid[0, 0])
    rg = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cg = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    _require(len(rg) >= 3 and len(cg) >= 3, "missing divider lattice")
    _require(rg[0][0] == cg[0][0] == 0 and rg[-1][1] == grid.shape[0]-1
             and cg[-1][1] == grid.shape[1]-1, "lattice must have an outer border")
    thicknesses = {b-a+1 for a, b in rg + cg}
    gaps = {rg[i+1][0]-rg[i][1]-1 for i in range(len(rg)-1)}
    gaps |= {cg[i+1][0]-cg[i][1]-1 for i in range(len(cg)-1)}
    _require(len(thicknesses) == len(gaps) == 1, "irregular lattice")
    side, thickness = gaps.pop(), thicknesses.pop()
    _require(side >= 2, "tiles too small")
    rows, columns = len(rg)-1, len(cg)-1
    _require(rows >= side and columns >= side, "stencil square cannot fit")
    records = {}
    for r in range(rows):
        for c in range(columns):
            y, x = rg[r][1]+1, cg[c][1]+1
            tile = grid[y:y+side, x:x+side].copy()
            _require(divider not in tile, "divider color cannot occur inside tiles")
            records.setdefault(tile.tobytes(), [tile, []])[1].append((r, c))
    _require(len(records) == 5, "one ordinary tile and four piece tile types required")
    ordered = sorted(records.values(), key=lambda v: -len(v[1]))
    ordinary, ordinary_positions = ordered[0]
    source = ordered[1:]
    _require(len(ordinary_positions) > sum(len(v[1]) for v in source), "ordinary tiles need a strict majority")
    _require(len(np.unique(ordinary)) == 2, "ordinary texture needs two colors")
    common = set(map(int, np.unique(ordinary)))
    for tile, _ in source:
        common &= set(map(int, np.unique(tile)))
    _require(len(common) == 1, "piece background must be identifiable by shared color")
    background = common.pop()
    anchor_index, key, key_location = None, None, None
    for i, (tile, _) in enumerate(source):
        patches = [(r, c) for r in range(side-1) for c in range(side-1)
                   if len(set(map(int, tile[r:r+2, c:c+2].flat))) == 4
                   and background not in tile[r:r+2, c:c+2]]
        if patches:
            _require(anchor_index is None and len(patches) == 1, "key must be unique")
            anchor_index = i
            key_location = patches[0]
            r, c = key_location
            key = tile[r:r+2, c:c+2].copy()
    _require(anchor_index is not None, "no keyed anchor")
    source = [source[anchor_index]] + [v for i, v in enumerate(source) if i != anchor_index]
    shapes, carriers = [], []
    for i, (tile, _) in enumerate(source):
        points = np.argwhere(tile != background)
        shape = _normalize(points)
        _require(_connected(shape), "disconnected piece footprint")
        colors = set(map(int, tile[tile != background]))
        if i == 0:
            outside_key = tile != background
            kr, kc = key_location
            outside_key[kr:kr+2, kc:kc+2] = False
            carrier = set(map(int, tile[outside_key]))
            _require(len(carrier) <= 1 and not carrier & set(map(int, key.flat)), "invalid anchor carrier")
            carriers.append(next(iter(carrier)) if carrier else None)
        else:
            _require(len(colors) == 1, "non-anchor pieces must be uniform")
            carriers.append(colors.pop())
        shapes.append(shape)
    _require(isqrt(sum(map(len, shapes))) == side and sum(map(len, shapes)) == side*side,
             "total footprint area must equal tile area")
    positions = [v[1] for v in source]
    _require(sum(map(len, positions)) < side*side, "at least one coarse copy must be missing")
    return {"grid": grid, "side": side, "thickness": thickness, "rows": rows,
            "columns": columns, "divider": divider, "background": background,
            "ordinary": ordinary, "source": source, "shapes": shapes, "positions": positions,
            "key": key, "key_location": key_location, "carriers": carriers}


def _render(info, assembly, origin):
    side, t = info["side"], info["thickness"]
    _, corner = _owners(assembly, side)
    glyphs = [np.where(corner == c, int(info["key"].flat[c]), info["background"])
              .astype(np.int8) for c in range(4)]
    result = info["grid"].copy()
    for r in range(side):
        for c in range(side):
            y = t + (r + origin[0]) * (side + t)
            x = t + (c + origin[1]) * (side + t)
            result[y:y+side, x:x+side] = glyphs[int(corner[r, c])]
    return result


def _infer(input_grid):
    info = _parse(input_grid)
    assemblies = _assemblies(info["shapes"], info["side"])
    matches = _joint(assemblies, info["side"], info["rows"], info["columns"], info["positions"])
    _require(len(matches) == 1, "packing and coarse clues must have exactly one joint solution")
    assembly, origin = matches[0]
    info.update(assemblies=assemblies, assembly=assembly, origin=origin)
    return _render(info, assembly, origin), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _infer(input_grid)[0]


def _partition(rng, side):
    """Grow connected, uneven corner regions, with a readable anchor key."""
    owner = np.full((side, side), -1, dtype=np.int8)
    owner[:2, :2] = 0
    for i, q in enumerate(_corners(side)):
        owner[q] = i
    weights = rng.uniform(0.4, 2.0, 4)
    while np.any(owner == -1):
        options, probability = [], []
        for r, c in np.argwhere(owner == -1):
            labels = {int(owner[r+dr, c+dc]) for dr, dc in NEIGHBORS
                      if 0 <= r+dr < side and 0 <= c+dc < side and owner[r+dr, c+dc] >= 0}
            for i in sorted(labels):
                options.append((int(r), int(c), i))
                probability.append(weights[i] / (1 + np.count_nonzero(owner == i)) ** .6)
        p = np.asarray(probability)
        r, c, i = options[int(rng.choice(len(options), p=p/p.sum()))]
        owner[r, c] = i
    return owner


def _choose_clues(rng, owner, assemblies, rows, columns, origin, cap):
    side = len(owner)
    positions = [[(r+origin[0], c+origin[1]) for r, c in map(tuple, np.argwhere(owner == i))]
                 for i in range(4)]
    # Full evidence is unique; greedily remove redundant evidence in random
    # order, always retaining at least one supplied copy of each piece.
    trial = [list(v) for v in positions]
    cells = [(i, q) for i, group in enumerate(trial) for q in group]
    rng.shuffle(cells)
    for i, q in cells:
        if len(trial[i]) == 1:
            continue
        trial[i].remove(q)
        if len(_joint(assemblies, side, rows, columns, trial)) != 1:
            trial[i].append(q)
    if sum(map(len, trial)) > cap:
        return None
    return [sorted(v) for v in trial], positions


def _paint_tiles(side, t, rows, columns, divider, ordinary, tiles, positions):
    result = np.full((rows*(side+t)+t, columns*(side+t)+t), divider, dtype=np.int8)
    lookup = {q: i for i, group in enumerate(positions) for q in group}
    for r in range(rows):
        for c in range(columns):
            y, x = t+r*(side+t), t+c*(side+t)
            result[y:y+side, x:x+side] = tiles[lookup[r, c]] if (r, c) in lookup else ordinary
    return result


def _make_pair(rng, side, first, test, used_layouts, first_clues=None):
    for _ in range(400):
        owner = _partition(rng, side)
        layout = _signature(owner)
        if layout in used_layouts:
            continue
        base_shapes = [_normalize(np.argwhere(owner == i)) for i in range(4)]
        if first and (len(base_shapes[0]) < 5 or len({_rotate(base_shapes[0], k) for k in range(4)}) != 4
                      or any(len(s) < 2 for s in base_shapes[1:])
                      or not any(len({_rotate(s, k) for k in range(4)}) == 4 for s in base_shapes[1:])):
            continue
        turns = [0, *map(int, rng.integers(0, 4, 3))]
        if first:
            turns = [0, 1, 2, 3]
        shapes = [_rotate(s, k) for s, k in zip(base_shapes, turns)]
        assemblies = _assemblies(shapes, side)
        if first and len({_signature(_owners(a, side)[1]) for a in assemblies}) < 2:
            continue
        if first:
            t, rows, columns = 1, 5, 5
        else:
            t = int(rng.choice([1, 1, 2]))
            maximum = (30-t)//(side+t)
            rows = int(rng.integers(side, maximum+1))
            columns = int(rng.integers(side, maximum+1))
        origin = (int(rng.integers(rows-side+1)), int(rng.integers(columns-side+1)))
        cap = min(side*side-1, (rows*columns-1)//2)
        if first:
            cap = min(cap, 10)
        if test:
            cap = min(cap, first_clues-2)
        chosen = _choose_clues(rng, owner, assemblies, rows, columns, origin, cap)
        if chosen is None:
            continue
        clues, complete = chosen
        if not test:
            target = int(rng.integers(7, min(10, cap)+1)) if first else min(cap, sum(map(len, clues))+int(rng.integers(1, 4)))
            unused = [(i, q) for i, group in enumerate(complete) for q in group if q not in clues[i]]
            rng.shuffle(unused)
            for i, q in unused[:max(0, target-sum(map(len, clues)))]:
                clues[i].append(q)
        palette = list(map(int, rng.permutation(10)))
        bg, divider = palette[:2]
        key = np.array(palette[2:6], dtype=np.int8).reshape(2, 2)
        anchor_carrier = palette[6]
        # Piece colors identify visible copies, but never output corners.
        carriers = [anchor_carrier, *map(int, rng.choice(palette[2:], 3, replace=True))]
        if len(set(carriers)) == 1:
            carriers[1] = palette[7]
        tiles = []
        for i, shape in enumerate(shapes):
            h, w = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            offset = (int(rng.integers(side-h+1)), int(rng.integers(side-w+1)))
            tile = np.full((side, side), bg, dtype=np.int8)
            for r, c in shape:
                tile[r+offset[0], c+offset[1]] = carriers[i]
            if i == 0:
                blocks = [(r, c) for r in range(side-1) for c in range(side-1)
                          if all((r+dr, c+dc) in shape for dr in range(2) for dc in range(2))]
                r, c = blocks[int(rng.integers(len(blocks)))]
                tile[r+offset[0]:r+offset[0]+2, c+offset[1]:c+offset[1]+2] = key
            tiles.append(tile)
        if len({v.tobytes() for v in tiles}) != 4:
            continue
        accent = int(rng.choice(palette[6:]))
        ordinary = np.full((side, side), bg, dtype=np.int8)
        nmarked = int(rng.integers(1, side*side//2+1))
        ordinary.flat[rng.choice(side*side, nmarked, replace=False)] = accent
        if ordinary.tobytes() in {v.tobytes() for v in tiles}:
            continue
        grid = _paint_tiles(side, t, rows, columns, divider, ordinary, tiles, clues)
        try:
            output, info = _infer(grid)
        except ValueError:
            continue
        # The first canonical packing shortcut must visibly fail in train[0].
        if first and (_signature(_owners(info["assemblies"][0], side)[1]) == _signature(owner)
                      or _joint(_assemblies(info["shapes"], side, rotate=False), side, rows, columns, info["positions"])):
            continue
        if not np.array_equal(_owners(info["assembly"], side)[1], owner):
            continue
        return {"input": grid, "output": output}, info, layout
    raise RuntimeError("bounded construction exhausted; no coupled example found")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4, "num_test must be 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, layouts = {"train": [], "test": []}, set(), set()
    first_clues, first_key = None, None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            first = split == "train" and i == 0
            side = 4 if split == "train" and i < 2 else (3 if split == "train" and i == 2 else int(rng.choice([3, 4])))
            # A reserved TL 2x2 block leaves only two unassigned cells on a
            # 3x3 square, each with three connected owner choices: nine cuts.
            if side == 3 and sum(layout[0] == (3, 3) for layout in layouts) == 9:
                side = 4
            for _ in range(20):
                pair, info, layout = _make_pair(rng, side, first, split == "test", layouts, first_clues)
                signature = _signature(pair["input"])
                key_repeat = split == "train" and i == 1 and np.array_equal(info["key"], first_key)
                if signature not in seen and not key_repeat:
                    break
            else:
                raise RuntimeError("unable to construct distinct inputs")
            if first:
                first_clues = sum(map(len, info["positions"]))
                first_key = info["key"].copy()
            seen.add(signature)
            layouts.add(layout)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "invalid pair counts")
    seen, information = set(), {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            expected, info = _infer(inp)
            _require(np.array_equal(expected, out), "incorrect self-portrait completion")
            side, t = info["side"], info["thickness"]
            _, corner = _owners(info["assembly"], side)
            y0, x0 = info["origin"]
            # Verify both directions of region correspondence from output cells.
            for r in range(side):
                for c in range(side):
                    y, x = t+(r+y0)*(side+t), t+(c+x0)*(side+t)
                    tile = out[y:y+side, x:x+side]
                    color = int(info["key"].flat[int(corner[r, c])])
                    _require(np.array_equal(tile == color, corner == corner[r, c]), "micro/macro footprint mismatch")
                    _require(np.all(tile[corner != corner[r, c]] == info["background"]), "whole-tile background must be copied")
            information[split].append(info)
    first = information["train"][0]
    s = first["side"]
    _require(7 <= sum(map(len, first["positions"])) <= 10, "opening clue count must be 7..10")
    _require(len({_rotate(first["shapes"][0], k) for k in range(4)}) == 4,
             "opening anchor must expose its orientation")
    _require(len({_signature(_owners(a, s)[1]) for a in first["assemblies"]}) >= 2,
             "first demonstration must expose geometry-only ambiguity")
    _require(not np.array_equal(_render(first, first["assemblies"][0], first["origin"]), puzzle["train"][0]["output"]),
             "first-packing shortcut needs a contrary demonstration")
    _require(not _joint(_assemblies(first["shapes"], s, rotate=False), s, first["rows"], first["columns"], first["positions"]),
             "training must expose rotation's geometric role")
    _require(len({_signature(f["key"]) for f in information["train"]}) >= 2, "training keys must vary")
    _require(len({_signature(_owners(f["assembly"], f["side"])[1]) for f in information["train"]}) >= 2,
             "training needs different region geometry")
    nfirst = sum(map(len, first["positions"]))
    _require(all(sum(map(len, f["positions"])) <= nfirst-2 for f in information["test"]),
             "tests must shift to sparser coarse evidence")


def _equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(left[s], right[s]) for k in ("input", "output")) for s in left)


def self_test():
    """Bounded 42-seed structural, semantic, ablation, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, inputs = [], set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                out, f = _infer(pair["input"])
                _require(np.array_equal(out, pair["output"]), "solve mismatch")
                _require(_signature(pair["input"]) not in inputs, "cross-seed duplicate")
                inputs.add(_signature(pair["input"]))
                records.append(f)
    _require(_equal(puzzles[0], generate_puzzle(0)), "nondeterminism")
    _require(not _equal(puzzles[0], puzzles[1]), "seeds did not vary")
    _require({f["side"] for f in records} == {3, 4}, "missing miniature scale")
    _require({f["thickness"] for f in records} == {1, 2}, "missing divider thickness")
    _require(len({_signature(f["ordinary"]) for f in records}) >= 100, "textures lack variety")
    _require(len({_signature(_owners(f["assembly"], f["side"])[1]) for f in records}) >= 60,
             "assembled cuts lack structural diversity")
    _require(len({f["grid"].shape for f in records}) >= 15, "canvas geometry lacks variety")
    _require(len({tuple(map(len, f["shapes"])) for f in records}) >= 30, "piece areas lack variety")
    _require({f["background"] for f in records} == set(range(10)), "fixed background colors")
    _require(any(len(set(f["carriers"])) < 4 for f in records), "shared carriers missing")
    _require(any(max(map(len, f["shapes"])) >= 4*min(map(len, f["shapes"])) for f in records),
             "unequal footprints missing")
    _require(any(all(0 < f["origin"][j] < (f["rows"], f["columns"])[j]-f["side"] for j in (0, 1))
                 for f in records), "interior registration missing")
    _require(any(any(len(group) > 1 for group in f["positions"]) for f in records), "repeated clues missing")
    _require({k for f in records for k in f["assembly"][2]} == {0, 1, 2, 3}, "rotation spectrum missing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*10+b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": len(records), "miniature_sizes": sorted({f["side"] for f in records}),
            "canvas_shapes": len({f["grid"].shape for f in records}),
            "corner_partitions": len({_signature(_owners(f["assembly"], f["side"])[1]) for f in records}),
            "area_profiles": len({tuple(map(len, f["shapes"])) for f in records}),
            "ordinary_textures": len({_signature(f["ordinary"]) for f in records}),
            "backgrounds": sorted({f["background"] for f in records}),
            "clue_counts": sorted({sum(map(len, f["positions"])) for f in records}),
            "geometry_only_ambiguous_pairs": sum(len({_signature(_owners(a, f["side"])[1])
                                                      for a in f["assemblies"]}) > 1 for f in records),
            "note": "Measured from pixels using the generator-owned parser and search; not independent review."}
