import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "044444034695",
        "040404036729",
        "044444000000",
        "000000000000",
        "033330000000",
        "030030000000",
        "030030077770",
        "030030070770",
        "003030070070",
        "000300077070",
        "000000077770",
        "000000000000",
    ),
    (
        "2600888800",
        "4500800800",
        "6800800800",
        "8100888800",
        "0000000000",
        "0555550444",
        "0500050404",
        "0055550444",
        "0000000404",
        "0000000444",
    ),
    (
        "026044444022220",
        "071040004020220",
        "000040004022020",
        "000044444022220",
    ),
    (
        "0111110033330",
        "0100010030030",
        "0110010030030",
        "0011110033330",
        "0000000000000",
        "0000000055500",
        "0020000500050",
        "0202000500005",
        "0020000055555",
        "0000000050000",
        "0000000000200",
        "5467800002020",
        "9831200000200",
    ),
)


def _rows_fingerprint(rows):
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_INPUT_FINGERPRINTS = frozenset(
    _rows_fingerprint(rows) for rows in _SOURCE_INPUT_ROWS
)


def _grid_fingerprint(grid):
    return (
        (int(grid.shape[0]), int(grid.shape[1])),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_fingerprint(pair):
    return (
        _grid_fingerprint(pair["input"]),
        _grid_fingerprint(pair["output"]),
    )


def _puzzle_fingerprint(puzzle):
    result = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            result.append(
                (
                    split,
                    _grid_fingerprint(pair["input"]),
                    _grid_fingerprint(pair["output"]),
                )
            )
    return tuple(result)


def _validate_count_argument(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _validate_count_argument(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_count_argument(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        seed_index = int(seed)
        num_test = 1 + seed_index % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed_index // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _components(mask, diagonal=False):
    """Coordinates of connected true pixels, in deterministic raster order."""
    remaining = set(zip(*np.nonzero(mask)))
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(cells))
    return result


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "ambiguous background"
    return int(colors[np.argmax(counts)])


def _regions(grid, background):
    h, w = grid.shape
    result = []
    for cells in _components(grid == background):
        boundary = set()
        edge = False
        for r, c in cells:
            edge |= r == 0 or c == 0 or r == h - 1 or c == w - 1
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != background:
                    boundary.add((rr, cc))
        result.append({"cells": cells, "boundary": boundary, "edge": edge})
    return result


def _key_candidates(grid, background):
    """Read a bounded strip, even when other ink shares its canvas edge."""
    h, w = grid.shape
    candidates = []
    for cells in _components(grid != background, diagonal=True):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        sh, sw = r1-r0+1, c1-c0+1
        if len(cells) != sh*sw or len({int(grid[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and sw >= 2:
            if r0 == 0:
                lanes.append(("top", grid[r0, c0:c1+1], grid[r1, c0:c1+1]))
            if r1 == h-1:
                lanes.append(("bottom", grid[r1, c0:c1+1], grid[r0, c0:c1+1]))
        if sw == 2 and sh >= 2:
            if c0 == 0:
                lanes.append(("left", grid[r0:r1+1, c0], grid[r0:r1+1, c1]))
            if c1 == w-1:
                lanes.append(("right", grid[r0:r1+1, c1], grid[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            sources, fills = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append({"side": side, "sources": sources, "fills": fills,
                                   "mapping": dict(zip(sources, fills)),
                                   "key_cells": frozenset(cells), "background": background,
                                   "corner": (r0 == 0 or r1 == h-1) and (c0 == 0 or c1 == w-1)})
    return candidates


def _apply_key(grid, key, regions=None):
    out = grid.copy()
    if regions is None:
        regions = _regions(grid, key["background"])
    for region in regions:
        if region["edge"]:
            continue
        assert not region["boundary"] & key["key_cells"], "key encloses background"
        colors = {int(grid[q]) for q in region["boundary"]}
        assert len(colors) == 1, "cavity has mixed ownership"
        color = next(iter(colors))
        if color in key["mapping"]:
            for q in region["cells"]:
                out[q] = key["mapping"][color]
    return out


def _find_key(grid):
    background = _background(grid)
    candidates = _key_candidates(grid, background)
    assert candidates, "no local edge key"
    assert len({k["key_cells"] for k in candidates}) == 1, "multiple strips"
    # A corner 2-by-2 key is legal only if all plausible directions agree.
    regions = _regions(grid, background)
    predictions = [_apply_key(grid, k, regions) for k in candidates]
    assert all(np.array_equal(predictions[0], p) for p in predictions[1:]), "ambiguous corner key"
    key = candidates[0].copy()
    key["sides"] = tuple(k["side"] for k in candidates)
    return key


def _solve_input(grid):
    return _apply_key(grid, _find_key(grid))


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    key = _find_key(inp)
    bg, mapping = key["background"], key["mapping"]
    assert np.array_equal(_apply_key(inp, key), out), "incorrect transformation"
    assert np.array_equal(inp[inp != bg], out[inp != bg]), "ink changed"
    assert np.any(inp != out), "each scene must demonstrate a fill"
    ids = np.full(inp.shape, -1, dtype=int)
    objects = []
    for color in np.unique(inp):
        if color == bg:
            continue
        mask = inp == color
        for q in key["key_cells"]:
            mask[q] = False
        for cells in _components(mask, diagonal=True):
            index = len(objects)
            for q in cells:
                ids[q] = index
            rr, cc = zip(*cells)
            objects.append({"color": int(color), "cells": cells, "holes": [],
                            "shape": (max(rr)-min(rr)+1, max(cc)-min(cc)+1),
                            "border": any(r in (0, inp.shape[0]-1) or c in (0, inp.shape[1]-1) for r, c in cells),
                            "diagonal": len(_components(ids == index)) > 1})
    assert objects
    irregular = 0
    for region in _regions(inp, bg):
        if region["edge"]:
            continue
        owners = {int(ids[q]) for q in region["boundary"]}
        assert len(owners) == 1 and -1 not in owners
        owner = next(iter(owners))
        objects[owner]["holes"].append(region["cells"])
        rr, cc = zip(*region["cells"])
        irregular += len(region["cells"]) != (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1)
    active_closed = [o for o in objects if o["color"] in mapping and o["holes"]]
    active_colors = [o["color"] for o in objects if o["color"] in mapping]
    chain = any(mapping[o["color"]] in mapping and
                mapping[mapping[o["color"]]] != mapping[o["color"]]
                for o in active_closed)
    self_map = any(mapping[o["color"]] == o["color"] for o in active_closed)
    fills = set(key["fills"])
    sources = set(mapping)
    return {"side": key["side"], "sides": key["sides"], "background": bg,
            "key_length": len(mapping), "mapping": mapping, "sources": key["sources"],
            "fills": key["fills"], "corner_key": key["corner"], "objects": objects,
            "object_count": len(objects), "object_shapes": tuple(o["shape"] for o in objects),
            "cavity_counts": tuple(len(o["holes"]) for o in objects),
            "irregular_holes": int(irregular), "diagonal": any(o["diagonal"] and o["holes"] for o in objects),
            "border_objects": sum(o["border"] for o in objects),
            "chain": chain, "self_map": self_map,
            "direction_witness": any(mapping[o["color"]] != o["color"] for o in active_closed),
            "open_active": sum(o["color"] in mapping and not o["holes"] for o in objects),
            "inert_count": sum(o["color"] not in mapping for o in objects),
            "fill_only_inert": sum(o["color"] in fills-sources for o in objects),
            "absent_inert": sum(o["color"] not in fills | sources for o in objects),
            "fill_only_closed": sum(o["color"] in fills-sources and bool(o["holes"]) for o in objects),
            "absent_closed": sum(o["color"] not in fills | sources and bool(o["holes"]) for o in objects),
            "repeated_active": len(active_colors) > len(set(active_colors)),
            "shared_fill": len(fills) < len(mapping),
            "active_entries": tuple(i for i, c in enumerate(key["sources"]) if c in active_colors),
            "changed_cells": int(np.count_nonzero(inp != out))}


def _dilate(mask, diagonal=False):
    result = mask.copy()
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    if diagonal:
        result[1:, 1:] |= mask[:-1, :-1]
        result[1:, :-1] |= mask[:-1, 1:]
        result[:-1, 1:] |= mask[1:, :-1]
        result[:-1, :-1] |= mask[1:, 1:]
    return result


def _crop(mask):
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _cavity(rng, h, w, rectangular=False):
    if rectangular or h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)
    # A connected row-convex polyomino: steps, lobes and one-cell necks,
    # without accidental islands of contour ink inside the cavity.
    a = np.zeros((h, w), dtype=bool)
    left, right = 0, w-1
    for r in range(h):
        if r:
            anchor = int(rng.integers(left, right+1))
            left = int(rng.integers(0, anchor+1))
            right = int(rng.integers(anchor, w))
        a[r, left:right+1] = True
    if np.all(a):
        a[-1, -1] = False
    return a


def _shape(rng, kind=None, small=False):
    if kind is None:
        kind = str(rng.choice(["rectangle", "profile", "profile", "diamond", "multi", "open"]))
    if kind == "diamond":
        radius = int(rng.integers(1, 3 if small else 5))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        mask = abs(rr-radius) + abs(cc-radius) == radius
    elif kind == "multi":
        h, w = (5, 5) if small else (int(rng.integers(5, 12)), int(rng.integers(5, 13)))
        mask = np.ones((h, w), dtype=bool)
        zones = [(1, h-1, 1, w-1)]
        wanted = int(rng.integers(2, 7))
        while len(zones) < wanted:
            possible = [i for i, (a,b,c,d) in enumerate(zones) if b-a >= 3 or d-c >= 3]
            if not possible:
                break
            i = int(rng.choice(possible))
            a,b,c,d = zones.pop(i)
            horizontal = b-a >= 3 and (d-c < 3 or rng.random() < .5)
            if horizontal:
                cut = int(rng.integers(a+1, b-1))
                zones.extend([(a,cut,c,d), (cut+1,b,c,d)])
            else:
                cut = int(rng.integers(c+1, d-1))
                zones.extend([(a,b,c,cut), (a,b,cut+1,d)])
        for a,b,c,d in zones:
            patch = _cavity(rng, b-a, d-c, rng.random() < .35)
            mask[a:b, c:d][patch] = False
        for r, c in ((0,0), (0,w-1), (h-1,0), (h-1,w-1)):
            if rng.random() < .6:
                mask[r,c] = False
    else:
        h = int(rng.integers(2 if kind == "profile" else 1, 3 if small else 8))
        w = int(rng.integers(2 if kind == "profile" else 1, 4 if small else 9))
        cavity = _cavity(rng, h, w, kind in ("rectangle", "open"))
        thickness = 1 if small else int(rng.choice([1,1,1,2]))
        cavity = np.pad(cavity, thickness)
        mask = cavity.copy()
        diagonal = kind == "rectangle" or rng.random() < .35
        for _ in range(thickness):
            mask = _dilate(mask, diagonal)
        mask &= ~cavity
        if kind == "open":
            r, c = np.argwhere(cavity)[int(rng.integers(np.count_nonzero(cavity)))]
            mask[:r+1, c] = False
    # Local wall thickening and attached appendages vary independently of orientation.
    if not small and kind not in ("open", "diamond") and rng.random() < .45:
        mask = np.pad(mask, int(rng.integers(1, 3)))
        for _ in range(int(rng.integers(1, 9))):
            external = np.zeros(mask.shape, dtype=bool)
            for region in _regions(mask.astype(int), 0):
                if region["edge"]:
                    for q in region["cells"]:
                        external[q] = True
            choices = np.argwhere(_dilate(mask) & external)
            if len(choices):
                q = choices[int(rng.integers(len(choices)))]
                mask[tuple(q)] = True
    mask = _crop(mask)
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    assert len(_components(mask, diagonal=True)) == 1
    return mask.copy()


def _make_mapping(rng, roles):
    need_chain, need_self = "chain" in roles, "self" in roles
    n = int(rng.integers(3 if need_chain and need_self else 2, 8))
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    sources = colors[:n]
    outside = colors[n:]
    style = int(rng.integers(4))
    if style == 0 and n <= 4:
        fills = list(map(int, rng.choice(outside, n, replace=True)))
    elif style == 1:
        fills = sources[1:] + [outside[0]]
    elif style == 2:
        destinations = list(map(int, rng.choice(colors, int(rng.integers(1, 4)), replace=False)))
        fills = list(map(int, rng.choice(destinations, n)))
    else:
        fills = list(map(int, rng.choice(colors, n)))
    # Reserve a nontrivial direction witness and any required lesson roles.
    fills[0] = sources[1] if need_chain else outside[0]
    if need_chain:
        fills[1] = outside[0]
    self_color = sources[2] if need_chain and need_self else sources[-1]
    if need_self:
        fills[sources.index(self_color)] = self_color
    if "fill_only" in roles:
        # Already ensured by the nontrivial/chain destination above.
        assert outside[0] in fills
    if "absent" in roles:
        absent = outside[-1]
        fills = [outside[0] if c == absent else c for c in fills]
    mapping = dict(zip(sources, fills))
    return mapping, {"chain": sources[0], "self": self_color,
                     "active": sources[0], "repeat": int(rng.choice(sources))}


def _put_key(grid, side, sources, fills, start):
    for i, (source, fill) in enumerate(zip(sources, fills)):
        p = start+i
        if side == "top":
            grid[0,p], grid[1,p] = source, fill
        elif side == "bottom":
            grid[-1,p], grid[-2,p] = source, fill
        elif side == "left":
            grid[p,0], grid[p,1] = source, fill
        else:
            grid[p,-1], grid[p,-2] = source, fill


def _pack(rng, masks, sources, fills, side, compact=False):
    """Random legal positions, then a guaranteed spacious packing fallback."""
    n = len(masks)
    maxh, maxw = max(a.shape[0] for a in masks), max(a.shape[1] for a in masks)
    area = sum((a.shape[0]+1)*(a.shape[1]+1) for a in masks) + 3*(len(sources)+1)
    for attempt in range(25):
        if compact and attempt < 6:
            if rng.random() < .5:
                h, w = max(4, maxh), int(rng.integers(max(12,maxw), 31))
            else:
                h, w = int(rng.integers(max(12,maxh), 31)), max(4, maxw)
        else:
            lowh, loww = max(6,maxh), max(6,maxw)
            if side in ("left", "right"):
                lowh = max(lowh,len(sources))
            else:
                loww = max(loww,len(sources))
            h, w = int(rng.integers(lowh,31)), int(rng.integers(loww,31))
            if rng.random() < .2:
                h = w = max(h,w)
            if attempt >= 18:
                h = w = 30
        if h*w < area or (side in ("left","right") and h < len(sources)) or (side in ("top","bottom") and w < len(sources)):
            continue
        grid = np.zeros((h,w), dtype=np.int8)
        axis = w if side in ("top","bottom") else h
        # Corners are sampled directly; invalid 2-by-2 directions are checked later.
        start = int(rng.choice([0, axis-len(sources)])) if rng.random() < .4 else int(rng.integers(axis-len(sources)+1))
        _put_key(grid, side, sources, fills, start)
        blocked = _dilate(grid != 0, True)
        placements = [None]*n
        for i in sorted(range(n), key=lambda j: -masks[j].size):
            mh,mw = masks[i].shape
            positions = [(r,c) for r in range(h-mh+1) for c in range(w-mw+1)
                         if not np.any(blocked[r:r+mh,c:c+mw])]
            if not positions:
                break
            border = [(r,c) for r,c in positions if r == 0 or c == 0 or r+mh == h or c+mw == w]
            choices = border if border and rng.random() < .5 else positions
            r,c = choices[int(rng.integers(len(choices)))]
            placements[i] = (r,c)
            blocked[max(0,r-1):min(h,r+mh+1),max(0,c-1):min(w,c+mw+1)] = True
        else:
            return grid, placements
    return None


def _generate_pair(rng, regime, side, bg, requirements=(), compact=False):
    assert regime in ("source_train", "target_test")
    for attempt in range(16):
        specs = list(requirements)
        minimum = max(1, len(specs))
        target = int(rng.integers(minimum, max(minimum+1, 10 if not compact else 4)))
        while len(specs) < target:
            role = str(rng.choice(["active", "active", "repeat", "inert"]))
            specs.append((role, None))
        if not any(role in ("active", "chain", "self") and kind != "open" for role,kind in specs):
            specs[0] = ("active", None)
        roles = [r for r,k in specs]
        mapping, named = _make_mapping(rng, roles)
        sources = list(mapping)
        fills = list(mapping.values())
        fill_only = sorted(set(fills)-set(sources))
        absent = sorted(set(range(1,10))-set(sources)-set(fills))
        inert = sorted(set(range(1,10))-set(sources))
        shapes, object_colors = [], []
        for index, (role, kind) in enumerate(specs):
            if role in ("chain", "self", "repeat"):
                color = named[role]
            elif role in ("fill_only", "absent", "inert"):
                choices = fill_only if role == "fill_only" else absent if role == "absent" else inert
                color = int(rng.choice(choices))
            else:
                color = named["active"] if index == 0 else int(rng.choice(sources))
            object_colors.append(color)
            # Small masks provide a deterministic bounded fallback for dense scenes.
            small = compact or attempt >= 10
            if kind is None and role in ("fill_only", "absent"):
                kind = str(rng.choice(["rectangle", "profile", "diamond", "multi"]))
            if kind is None and index == 0:
                kind = str(rng.choice(["rectangle", "profile", "diamond", "multi"]))
            shape = _shape(rng, kind, small=small)
            shapes.append(shape)
        order = rng.permutation(len(sources))
        sources, fills = [sources[i] for i in order], [fills[i] for i in order]
        packed = _pack(rng, shapes, sources, fills, side, compact)
        if packed is None:
            continue
        grid, placements = packed
        for shape, color, (r,c) in zip(shapes, object_colors, placements):
            patch = grid[r:r+shape.shape[0],c:c+shape.shape[1]]
            patch[shape] = color
        # Background is an input-visible dominant color, including in narrow scenes.
        if np.count_nonzero(grid == 0) <= max(np.count_nonzero(grid == c) for c in range(1,10)):
            continue
        permutation = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
        palette = np.array([bg]+permutation, dtype=np.int8)
        grid = palette[grid]
        try:
            out = _solve_input(grid)
            analysis = _analyze_pair(grid, out)
        except AssertionError:
            continue
        if not analysis["direction_witness"]:
            continue
        if any(role == "chain" for role,kind in requirements) and not analysis["chain"]:
            continue
        if any(role == "self" for role,kind in requirements) and not analysis["self_map"]:
            continue
        if any(kind == "open" for role,kind in requirements) and not analysis["open_active"]:
            continue
        return {"input": grid, "output": out}
    # Fixed geometry, variable colors and key placement, guarantees legal requests.
    # Lesson shapes remain intact; this branch avoids unbounded rejection.
    specs = list(requirements) or [("active", "rectangle")]
    mapping, named = _make_mapping(rng, [r for r,k in specs])
    grid = np.zeros((30,30), dtype=np.int8)
    _put_key(grid, "top", list(mapping), list(mapping.values()), 10)
    for i,(role,kind) in enumerate(specs):
        r,c = 4+(i//4)*7, 1+(i%4)*7
        shape = np.ones((5,5), dtype=bool)
        shape[1:4,1:4] = False
        if kind == "diamond":
            rr,cc = np.indices((5,5))
            shape = abs(rr-2)+abs(cc-2) == 2
        elif kind == "profile":
            shape[3,3] = True
            shape[0,0] = False
        elif kind == "multi":
            shape[2,:] = True
        elif kind == "open":
            shape[0,2] = False
        if role == "absent":
            color = min(set(range(1,10))-set(mapping)-set(mapping.values()))
        elif role == "fill_only":
            color = min(set(mapping.values())-set(mapping))
        elif role == "inert":
            color = min(set(range(1,10))-set(mapping))
        else:
            color = named.get(role, named["active"])
        grid[r:r+5,c:c+5][shape] = color
    # Rotate the entire scene to the requested side, retaining inward direction.
    turns = {"top":0,"left":1,"bottom":2,"right":3}[side]
    grid = np.rot90(grid,turns).copy()
    palette = np.array([bg]+list(map(int,rng.permutation([c for c in range(10) if c != bg]))))
    grid = palette[grid]
    return {"input":grid, "output":_solve_input(grid)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    # There are no hidden color conventions: the local key carries each mapping.
    # The sampled puzzle-level curriculum only controls familiar/novel sides.
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    regime = str(rng.choice(["homogeneous", "familiar", "mixed", "novel"], p=[.15,.25,.25,.35]))
    train_pool = sides[:1] if regime == "homogeneous" else sides[:int(rng.integers(2,4))]
    train_sides = [train_pool[i % len(train_pool)] for i in range(num_train)]
    if regime == "novel":
        test_sides = [sides[-1]]*num_test
    elif regime == "mixed":
        test_sides = [str(rng.choice(sides)) for _ in range(num_test)]
    else:
        test_sides = [str(rng.choice(train_pool[:num_train])) for _ in range(num_test)]
    backgrounds = list(map(int,rng.permutation(10)))
    puzzle = {"train":[], "test":[]}
    # Plan training from visible test features, never use a test label for prediction.
    for i,side in enumerate(test_sides):
        puzzle["test"].append(_generate_pair(rng,"target_test",side,backgrounds[(i+2)%10],
                                               compact=rng.random() < .13))
    test_features = [_analyze_pair(p["input"],p["output"]) for p in puzzle["test"]]
    lessons = [[("active","profile"),("fill_only",None)],
               [("active","diamond"),("absent",None)]]
    if any(f["chain"] for f in test_features):
        lessons[int(rng.integers(2))][0] = ("chain", str(rng.choice(["profile","diamond"])))
    if any(f["self_map"] for f in test_features):
        lessons[int(rng.integers(2))].append(("self","diamond"))
    if any(f["open_active"] for f in test_features):
        lessons[int(rng.integers(2))].append(("active","open"))
    if any(max(f["cavity_counts"]) > 1 for f in test_features):
        lessons[int(rng.integers(2))].append(("active","multi"))
    # Both diagonal closure and nonrectangular cavities are always illustrated;
    # the first two scenes can combine lessons even with the minimal counts.
    if not any(kind == "profile" for row in lessons for role,kind in row):
        lessons[0].append(("active","profile"))
    if not any(kind == "diamond" for row in lessons for role,kind in row):
        lessons[1].append(("active","diamond"))
    seen = {_grid_fingerprint(p["input"]) for p in puzzle["test"]}
    for i,side in enumerate(train_sides):
        required = lessons[i] if i < 2 else ()
        for retry in range(20):
            pair = _generate_pair(rng,"source_train",side,backgrounds[(i+retry)%10],required,
                                  compact=i >= 2 and rng.random() < .16)
            fingerprint = _grid_fingerprint(pair["input"])
            if fingerprint not in seen and fingerprint not in _SOURCE_INPUT_FINGERPRINTS:
                seen.add(fingerprint)
                puzzle["train"].append(pair)
                break
        else:
            raise AssertionError("unique training scene construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {"train","test"}
    assert all(isinstance(puzzle[s],list) for s in ("train","test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train":[],"test":[]}
    for split in ("train","test"):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {"input","output"}
            for grid in pair.values():
                assert isinstance(grid,np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype,np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            fingerprint = _grid_fingerprint(pair["input"])
            assert fingerprint not in seen and fingerprint not in _SOURCE_INPUT_FINGERPRINTS
            seen.add(fingerprint)
            analyses[split].append(_analyze_pair(pair["input"],pair["output"]))
    train, test = analyses["train"], analyses["test"]
    for feature in ("chain","self_map","open_active"):
        assert not any(f[feature] for f in test) or any(f[feature] for f in train), "missing " + feature + " lesson"
    assert any(f["direction_witness"] for f in train)
    assert any(f["diagonal"] for f in train), "four-connectivity not demonstrated"
    assert any(f["irregular_holes"] for f in train), "missing nonrectangular cavity lesson"
    assert any(f["fill_only_closed"] for f in train) and any(f["absent_closed"] for f in train)
    train_sides = {side for f in train for side in f["sides"]}
    # Equivalent interpretations of a corner key need only one familiar side.
    assert all(set(f["sides"]) & train_sides or len(train_sides) >= 2
               for f in test), "unexplained novel direction"


def _independent_fill(grid):
    """Supplementary check: for each color, flood the complement from outside.

    This differs from construction's background-component boundary algorithm.
    Key discovery is shared, so this is deliberately not called a reference solver.
    """
    key = _find_key(grid)
    bg, mapping = key["background"], key["mapping"]
    expected = grid.copy()
    for color, fill in mapping.items():
        wall = grid == color
        for q in key["key_cells"]:
            wall[q] = False
        passable = np.pad(~wall,1,constant_values=True)
        outside = np.zeros(passable.shape,dtype=bool)
        stack = [(0,0)]
        outside[0,0] = True
        while stack:
            r,c = stack.pop()
            for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < passable.shape[0] and 0 <= cc < passable.shape[1] and passable[rr,cc] and not outside[rr,cc]:
                    outside[rr,cc] = True
                    stack.append((rr,cc))
        enclosed = ~outside[1:-1,1:-1] & (grid == bg)
        expected[enclosed] = fill
    return expected


_SOURCE_OUTPUT_ROWS = (('044444034695', '046464036729', '044444000000', '000000000000', '033330000000', '033330000000', '033330077770', '033330070770', '003330070070', '000300077070', '000000077770', '000000000000'), ('2600888800', '4500811800', '6800811800', '8100888800', '0000000000', '0555550444', '0500050454', '0055550444', '0000000454', '0000000444'), ('026044444022220', '071040004027220', '000040004022720', '000044444022220'), ('0111110033330', '0177710036630', '0117710036630', '0011110033330', '0000000000000', '0000000055500', '0020000500050', '0282000500005', '0020000055555', '0000000050000', '0000000000200', '5467800002820', '9831200000200'))


def self_test() -> None:
    # Regressions are separate from generated-content source-copy exclusion.
    for input_rows, output_rows in zip(_SOURCE_INPUT_ROWS, _SOURCE_OUTPUT_ROWS):
        inp = np.array([list(map(int,r)) for r in input_rows], dtype=np.int8)
        out = np.array([list(map(int,r)) for r in output_rows], dtype=np.int8)
        for turn in range(4):
            for offset in (0,3,7):
                a, b = [(np.rot90(grid,turn)+offset) % 10 for grid in (inp,out)]
                assert np.array_equal(_solve_input(a),b)
                assert np.array_equal(_independent_fill(a),b)
                _analyze_pair(a,b)

    # Conflicting corner directions must be rejected, not resolved by scan order.
    corner = np.zeros((9,10),dtype=np.int8)
    corner[:2,:2] = [[1,2],[3,4]]
    corner[4:7,5:8] = [[1,1,1],[1,0,1],[1,1,1]]
    try:
        _solve_input(corner)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting corner key accepted")
    corner[:2,:2] = [[1,2],[2,4]]
    assert _solve_input(corner)[5,6] == 2

    # A matched open/closed contrast, one-step chain, self-map diamond,
    # unmapped cavity and two-hole border object, with hand-specified fills.
    contrast = np.zeros((12,16),dtype=np.int8)
    contrast[:2,:4] = [[1,2,3,4],[2,5,3,5]]
    ring = np.array([[1,1,1,1],[1,0,0,1],[1,1,1,1]],dtype=np.int8)
    contrast[4:7,:4] = ring
    contrast[4:7,6:10] = ring
    contrast[4,7] = 0
    rr,cc = np.indices((5,5))
    contrast[7:12,:5][abs(rr-2)+abs(cc-2) == 2] = 3
    contrast[8:11,8:12] = 6*ring
    contrast[3:8,13:16] = [[2,2,2],[2,0,2],[2,2,2],[2,0,2],[2,2,2]]
    expected = contrast.copy()
    expected[5,1:3] = 2
    for q in ((9,2),(8,2),(10,2),(9,1),(9,3)):
        expected[q] = 3
    expected[4,14] = expected[6,14] = 5
    assert np.array_equal(_solve_input(contrast),expected)
    assert np.array_equal(_independent_fill(contrast),expected)
    f = _analyze_pair(contrast,expected)
    assert f["chain"] and f["self_map"] and f["open_active"] and f["diagonal"]

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_fingerprint(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_fingerprint(generate_puzzle(17)) == _puzzle_fingerprint(puzzles[17])
    measured = {"train":[],"test":[]}
    shapes, structures, within = set(), set(), []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = []
        for split in ("train","test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(_independent_fill(inp.copy()),out)
                f = _analyze_pair(inp,out)
                measured[split].append(f)
                local.append(f)
                shapes.add(inp.shape)
                palette = {f["background"]:0}
                normalized = []
                for c in inp.flat:
                    if int(c) not in palette:
                        palette[int(c)] = len(palette)
                    normalized.append(palette[int(c)])
                structures.add((inp.shape,tuple(normalized)))
        within.append({name:len({str(f[name]) for f in local}) for name in
                       ("key_length","background","object_count","object_shapes","mapping")})
    both = measured["train"] + measured["test"]
    assert len(shapes) >= 100 and len(structures) >= 300
    assert min(min(s) for s in shapes) <= 4 and max(max(s) for s in shapes) == 30
    assert sum(h == w for h,w in shapes) >= 5
    for split, records in measured.items():
        assert {f["side"] for f in records} == {"top","bottom","left","right"}
        assert {f["background"] for f in records} == set(range(10))
        assert {f["key_length"] for f in records} == set(range(2,8))
        assert min(f["object_count"] for f in records) == 1
        assert max(f["object_count"] for f in records) >= 8
        for feature in ("chain","self_map","open_active","repeated_active","shared_fill",
                        "fill_only_inert","absent_inert","border_objects","corner_key",
                        "diagonal","irregular_holes"):
            assert sum(bool(f[feature]) for f in records) >= 5, (split,feature)
        assert set(i for f in records for i in f["active_entries"]) == set(range(7))
    assert max(max(f["cavity_counts"]) for f in both) >= 5
    assert max(max(s) for f in both for s in f["object_shapes"]) > 10
    assert any(f["side"] == "right" and f["background"] != 0 and f["irregular_holes"]
               and f["border_objects"] and f["repeated_active"] for f in both)
    for name in within[0]:
        assert sum(f[name] >= 2 for f in within) >= 38, name
    familiar = novel = homogeneous = 0
    for p in puzzles:
        train_sides = {_find_key(q["input"])["side"] for q in p["train"]}
        test_sides = {_find_key(q["input"])["side"] for q in p["test"]}
        familiar += bool(test_sides & train_sides)
        novel += bool(test_sides-train_sides)
        homogeneous += len(train_sides | test_sides) == 1
    assert familiar >= 5 and novel >= 5 and homogeneous >= 2

    for index,(train,test) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(1000+index,train,test)
        validate_generated_puzzle(p)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
        assert _puzzle_fingerprint(p) == _puzzle_fingerprint(generate_puzzle(1000+index,train,test))
    for train,test in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(3,1.5)):
        try:
            generate_puzzle(0,train,test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    broken = {s:[{k:v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in ("train","test")}
    key = _find_key(broken["test"][0]["input"])
    q = min(key["key_cells"])
    broken["test"][0]["output"][q] = key["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("key mutation accepted")
