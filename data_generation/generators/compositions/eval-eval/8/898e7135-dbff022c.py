"""Shape-addressed casting: a keyed palette of small holes colors large holes.

All recovery is from the input. The sampler does not call a parent generator.
Only Python's standard library and NumPy are used.
"""
from functools import lru_cache
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask, diagonal=False):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in N8 if diagonal else N4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _normal(cells):
    a, b = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-a), int(c-b)) for r, c in cells))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _rotate(cells, q):
    for _ in range(q % 4):
        cells = [(c, -r) for r, c in cells]
    return _normal(cells)


@lru_cache(maxsize=4096)
def _orientations(shape, reflect=True):
    bases = [shape, tuple((r, -c) for r, c in shape)] if reflect else [shape]
    return tuple(sorted({_rotate(s, q) for s in bases for q in range(4)}))


def _canonical(cells):
    return _orientations(_normal(cells))[0]


def _mask(shape):
    result = np.zeros(_size(shape), dtype=bool)
    for p in shape:
        result[p] = True
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _dilate(mask, diagonal=False):
    result = mask.copy()
    h, w = mask.shape
    for dr, dc in N8 if diagonal else N4:
        a, b = max(0, dr), min(h, h+dr)
        c, d = max(0, dc), min(w, w+dc)
        result[a:b, c:d] |= mask[a-dr:b-dr, c-dc:d-dc]
    return result


def _regions(grid, bg):
    h, w = grid.shape
    result = []
    for cells in _components(grid == bg):
        edge = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        boundary = {(r+dr, c+dc) for r, c in cells for dr, dc in N4
                    if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] != bg}
        result.append((cells, edge, boundary))
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger grid"
    assert np.all((a >= 0) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _key(grid, bg):
    candidates = []
    h, w = grid.shape
    for cells in _components(grid != bg, True):
        a, b, c, d = _bbox(cells)
        if len(cells) != (b-a)*(d-c) or len({int(grid[p]) for p in cells}) < 2:
            continue
        lanes = []
        # Length >=3 removes the ambiguous two-by-two corner convention.
        if b-a == 2 and d-c >= 3:
            if a == 0:
                lanes.append(("top", grid[a, c:d], grid[b-1, c:d]))
            if b == h:
                lanes.append(("bottom", grid[b-1, c:d], grid[a, c:d]))
        if d-c == 2 and b-a >= 3:
            if c == 0:
                lanes.append(("left", grid[a:b, c], grid[a:b, d-1]))
            if d == w:
                lanes.append(("right", grid[a:b, d-1], grid[a:b, c]))
        for side, outer, inner in lanes:
            sources, fills = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(sources)) == len(sources):
                candidates.append((side, dict(zip(sources, fills)), set(cells)))
    assert len(candidates) == 1, "expected one unambiguous local edge key"
    return candidates[0]


def _decode(cells, scale):
    a, b, c, d = _bbox(cells)
    if (b-a) % scale or (d-c) % scale:
        return None
    blocks = {}
    for r, col in cells:
        p = (r-a)//scale, (col-c)//scale
        blocks[p] = blocks.get(p, 0)+1
    if any(n != scale*scale for n in blocks.values()):
        return None
    logical = np.zeros(((b-a)//scale, (d-c)//scale), dtype=np.int8)
    for p in blocks:
        logical[p] = 1
    return logical, (a, b, c, d)


def _recover(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > grid.size//2, "background must occupy a strict majority"
    side, mapping, key_cells = _key(grid, bg)
    objects = []
    for color in range(10):
        if color == bg:
            continue
        mask = grid == color
        for p in key_cells:
            mask[p] = False
        objects.extend((color, cells) for cells in _components(mask, True))
    # The compact source holes form a dictionary indexed by D4 shape, rather
    # than by position, area, or the literal color of a large frame.
    palette, source_shapes, source_regions = {}, {}, []
    for cells, edge, boundary in _regions(grid, bg):
        if edge:
            continue
        colors = {int(grid[p]) for p in boundary}
        assert len(colors) == 1 and not boundary & key_cells, "mixed cavity ownership"
        color = next(iter(colors))
        if color not in mapping:
            continue
        shape = _normal(cells)
        key = _canonical(shape)
        fill = mapping[color]
        assert key not in palette or palette[key] == fill, "conflicting shape palette"
        palette[key] = fill
        source_shapes.setdefault(key, []).append(shape)
        source_regions.append((color, cells, boundary))
    assert palette, "no closed mapped source cavity"
    candidates = []
    for wall in sorted({color for color, cells in objects} - set(mapping)):
        wall_objects = [cells for color, cells in objects if color == wall]
        for scale in range(2, 11):
            decoded = [_decode(cells, scale) for cells in wall_objects]
            if any(v is None for v in decoded):
                continue
            target_holes, matched, reflection, rotation = [], 0, False, False
            valid = True
            for logical, bounds in decoded:
                holes = [(cells, boundary) for cells, edge, boundary in _regions(logical, 0) if not edge]
                if not holes:
                    valid = False
                    break
                for cells, boundary in holes:
                    shape = _normal(cells)
                    key = _canonical(shape)
                    fill = palette.get(key, bg)
                    active = key in palette
                    if active:
                        matched += 1
                        reflection |= all(shape not in _orientations(s, False) for s in source_shapes[key])
                        rotation |= all(shape != s for s in source_shapes[key]) and not all(
                            shape not in _orientations(s, False) for s in source_shapes[key])
                    target_holes.append({"cells": cells, "bounds": bounds, "shape": shape,
                                         "fill": fill, "active": active})
            if valid and matched:
                candidates.append((wall, scale, target_holes, reflection, rotation, len(wall_objects)))
    assert len(candidates) == 1, "ambiguous or absent large-frame color/scale"
    wall, scale, target_holes, reflection, rotation, target_count = candidates[0]
    assert all(not any(_decode(cells, k) is not None for k in range(2, 11))
               for color, cells in objects if color != wall and color not in mapping), \
        "unlisted compact contours must not imitate another large-frame role"
    out = grid.copy()
    changed_mask = np.zeros(grid.shape, dtype=bool)
    for hole in target_holes:
        a, b, c, d = hole["bounds"]
        for r, col in hole["cells"]:
            view = out[a+r*scale:a+(r+1)*scale, c+col*scale:c+(col+1)*scale]
            assert np.all(view == bg), "target cavity is not empty"
            if hole["active"]:
                view[:] = hole["fill"]
                changed_mask[a+r*scale:a+(r+1)*scale, c+col*scale:c+(col+1)*scale] = True
    # Every mapped object is a compact, unit-scale source; open sources cannot
    # contribute. This restriction also prevents accidentally keyed large frames.
    source_objects = [(color, cells) for color, cells in objects if color in mapping]
    assert all(not any(_decode(cells, k) is not None for k in range(2, 11))
               for color, cells in source_objects), "source contour must be unit scale"
    owners = []
    for color, cells, boundary in source_regions:
        owner = [obj for ink, obj in source_objects if ink == color and boundary <= set(obj)]
        assert len(owner) == 1, "source wall must be eight-connected"
        owners.append(owner[0])
    open_count = sum(cells not in owners for color, cells in source_objects)
    areas = [len(s) for s in palette]
    info = {"background": bg, "wall": wall, "scale": scale, "side": side,
            "mapping": mapping, "source_shapes": tuple(sorted(palette)),
            "shape_palette": palette, "source_regions": source_regions,
            "source_count": len(source_objects), "source_holes": len(source_regions),
            "target_count": target_count, "target_holes": len(target_holes),
            "matched": sum(h["active"] for h in target_holes),
            "unmatched": sum(not h["active"] for h in target_holes),
            "broadcast": sum(h["active"] for h in target_holes) > len(palette),
            "multicolor_frame": any(len({h["fill"] for h in target_holes
                                         if h["active"] and h["bounds"] == b}) > 1
                                    for b in {h["bounds"] for h in target_holes}),
            "reflection": reflection, "rotation": rotation, "open": open_count,
            "diagonal_source": any(len(_components(_mask(_normal(cells)))) > 1
                                   for color, cells in source_objects if cells in owners),
            "chain": any(mapping[color] in mapping and mapping[mapping[color]] != mapping[color]
                         for color, cells, boundary in source_regions),
            "self_map": any(mapping[color] == color for color, cells, boundary in source_regions),
            "equal_area": len(set(areas)) < len(areas),
            "equal_area_different_fill": any(len(a) == len(b) and a != b and palette[a] != palette[b]
                                              for a in palette for b in palette),
            "irregular": any(len(s) != np.prod(_size(s)) for s in palette),
            "changed": int(changed_mask.sum()), "change_mask": changed_mask,
            "input_shape": tuple(grid.shape)}
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill only large matched cavities; use no examples or generation state."""
    return _recover(input_grid)[0]


def _grow(rng, area, max_span=4):
    for _ in range(40):
        cells = {(0, 0)}
        while len(cells) < area:
            choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in N4} - cells)
            choices = [p for p in choices if max(_size(_normal(cells | {p}))) <= max_span]
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        s = _normal(cells)
        # A source cavity has no inner wall island. Its exterior is connected.
        m = np.pad(_mask(s), 1)
        if len(cells) == area and len(_components(~m)) == 1:
            return s
    return tuple((0, i) for i in range(min(area, max_span)))


def _source_mask(rng, shape, opened=False, diagonal=False):
    hole = np.pad(_mask(shape), 1)
    wall = _dilate(hole, diagonal) & ~hole
    if opened:
        # Open a shortest straight channel from a cavity cell to the exterior.
        r, c = map(int, np.argwhere(hole)[0])
        wall[:r+1, c] = False
    # Reject accidental extra cavities or a block-scaled compact source.
    regions = [cells for cells, edge, boundary in _regions(wall.astype(np.int8), 0) if not edge]
    assert len(regions) == (0 if opened else 1)
    if not opened:
        assert _normal(regions[0]) == shape
    assert len(_components(wall, True)) == 1
    return wall


def _target_mask(rng, shapes, scale):
    """Place several logical cavities in one frame, then expand its ink."""
    limit = 30//scale
    for _ in range(100):
        oriented = [s for s in shapes]
        h = int(rng.integers(max(_size(s)[0] for s in shapes)+2, min(limit, 9)+1))
        w = int(rng.integers(max(_size(s)[1] for s in shapes)+2, min(limit, 10)+1))
        occupied = set()
        placements = []
        for shape in oriented:
            sh, sw = _size(shape)
            positions = [(r, c) for r in range(1, h-sh) for c in range(1, w-sw)]
            rng.shuffle(positions)
            for a, b in positions:
                cells = [(a+r, b+c) for r, c in shape]
                if all(p not in occupied and not any((p[0]+dr, p[1]+dc) in occupied for dr, dc in N4)
                       for p in cells):
                    occupied.update(cells)
                    placements.append(cells)
                    break
            else:
                break
        if len(placements) != len(shapes):
            continue
        # Trim spare exterior rows/columns, retain at least a one-cell border.
        a, b, c, d = _bbox(occupied)
        logical = np.ones((b-a+2, d-c+2), dtype=bool)
        for r, col in occupied:
            logical[r-a+1, col-c+1] = False
        # Corner cuts and a thin orthogonal frame vary the exterior independently
        # of the matching holes. Do not change hole ownership or add holes.
        if len(shapes) == 1 and rng.random() < .45:
            cavity = ~logical
            logical = _dilate(cavity, bool(rng.integers(2))) & ~cavity
        elif rng.random() < .45:
            for p in ((0, 0), (0, logical.shape[1]-1), (logical.shape[0]-1, 0),
                      (logical.shape[0]-1, logical.shape[1]-1)):
                if rng.random() < .6:
                    logical[p] = False
        if len(_components(logical, True)) != 1:
            continue
        if int(logical.sum())*scale*scale > 365:
            continue
        return np.repeat(np.repeat(logical, scale, 0), scale, 1)
    return None


def _free(blocked, h, w):
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return []
    sums = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, objects, mapping, side):
    entries = list(mapping.items())
    rng.shuffle(entries)
    n = len(entries)
    for attempt in range(35):
        if attempt < 20:
            h, w = int(rng.integers(18, 31)), int(rng.integers(18, 31))
        else:
            h = w = 30
        grid = np.zeros((h, w), dtype=np.int8)
        blocked = np.zeros((h, w), dtype=bool)
        axis = w if side in ("top", "bottom") else h
        start = int(rng.integers(axis-n+1))
        for i, (src, fill) in enumerate(entries):
            p = start+i
            if side == "top":
                grid[0, p], grid[1, p] = src, fill
            elif side == "bottom":
                grid[-1, p], grid[-2, p] = src, fill
            elif side == "left":
                grid[p, 0], grid[p, 1] = src, fill
            else:
                grid[p, -1], grid[p, -2] = src, fill
        blocked[:] = _dilate(grid != 0, True)
        order = sorted(range(len(objects)), key=lambda j: -objects[j][0].size)
        placements = [None]*len(objects)
        for j in order:
            mask, color = objects[j]
            sh, sw = mask.shape
            places = _free(blocked, sh, sw)
            if len(places) == 0:
                break
            r, c = map(int, places[int(rng.integers(len(places)))])
            grid[r:r+sh, c:c+sw][mask] = color
            placements[j] = (r, c)
            blocked[max(0, r-1):min(h, r+sh+1), max(0, c-1):min(w, c+sw+1)] = True
        else:
            if np.count_nonzero(grid == 0) > grid.size//2:
                return grid, placements
    return None


L = ((0, 0), (1, 0), (2, 0), (2, 1))
T = ((0, 0), (0, 1), (0, 2), (1, 1))


def _make_pair(rng, scale, side, lesson=None, group=False, reflected=False):
    for attempt in range(150):
        n = int(rng.integers(3, 6))
        sources = list(range(1, n+1))
        wall = 9
        mapping = {c: int(rng.integers(1, 9)) for c in sources}
        if lesson == "matching":
            # The lesson is a relationship, not one fixed pair of tetrominoes.
            for _ in range(80):
                first = _grow(rng, int(rng.integers(4, 8)), 3)
                if len(_orientations(first)) > len(_orientations(first, False)):
                    break
            else:
                first = L
            for _ in range(80):
                second = _grow(rng, len(first), 3)
                if _canonical(second) != _canonical(first):
                    break
            else:
                first, second = L, T
            shapes = [first, second]
            mapping[1], mapping[2] = 2, 7  # Active one-step chain 1 -> 2 -> 7.
            shapes = [_rotate(s, int(rng.integers(4))) for s in shapes]
            specs = [(1, shapes[0], False, bool(rng.integers(2))),
                     (2, shapes[1], False, bool(rng.integers(2)))]
            destinations = [_rotate(tuple((r, -c) for r, c in shapes[0]), int(rng.integers(4))),
                            _rotate(shapes[1], int(rng.integers(1, 4)))]
            target_groups = [destinations]
        elif lesson == "topology":
            diamond_hole = ((0, 0),) if rng.random() < .5 else ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
            shapes = [diamond_hole]
            open_shape = _grow(rng, int(rng.integers(2, 5)), 3)
            mapping[1] = 1
            specs = [(1, shapes[0], False, False), (2, open_shape, True, True)]
            destinations = [shapes[0], shapes[0], _rotate(open_shape, int(rng.integers(4)))]
            target_groups = [destinations]
        else:
            count = int(rng.integers(1, 4))
            shapes = []
            for i in range(count):
                for _ in range(60):
                    s = _grow(rng, int(rng.integers(1, 10)))
                    if _canonical(s) not in {_canonical(t) for t in shapes}:
                        break
                else:
                    break
                shapes.append(s)
            if not shapes:
                continue
            specs = [(i+1, s, False, bool(rng.integers(2))) for i, s in enumerate(shapes)]
            destinations = []
            for s in shapes:
                opts = _orientations(s)
                if reflected:
                    mirror = [o for o in opts if o not in _orientations(s, False)]
                    opts = mirror or opts
                destinations.append(opts[int(rng.integers(len(opts)))])
            if rng.random() < .45:
                destinations.append(destinations[int(rng.integers(len(destinations)))])
            if rng.random() < .55:
                for _ in range(40):
                    spare = _grow(rng, int(rng.integers(1, 8)))
                    if _canonical(spare) not in {_canonical(s) for s in shapes}:
                        break
                else:
                    continue
                destinations.append(spare)
                if rng.random() < .65:
                    specs.append((n, spare, True, True))
            if group or len(destinations) == 1 or rng.random() < .6:
                target_groups = [destinations]
            else:
                rng.shuffle(destinations)
                cut = int(rng.integers(1, len(destinations)))
                target_groups = [destinations[:cut], destinations[cut:]]
        objects = []
        try:
            for color, shape, opened, diagonal in specs:
                objects.append((_source_mask(rng, shape, opened, diagonal), color))
        except AssertionError:
            continue
        target_indices = []
        for shapes_in_frame in target_groups:
            mask = _target_mask(rng, shapes_in_frame, scale)
            if mask is None:
                break
            target_indices.append(len(objects))
            objects.append((mask, wall))
        else:
            # Some clean examples contain an unlisted small closed contour.
            if lesson is None and rng.random() < .25:
                objects.append((_source_mask(rng, ((0, 0),), False, True), 8))
            packed = _pack(rng, objects, mapping, side)
            if packed is None:
                continue
            grid, placements = packed
            # Construct the label from the placed design, before invoking input
            # recovery. Construction knows which masks are targets; solve does not.
            expected = grid.copy()
            shape_palette = {_canonical(s): mapping[color] for color, s, opened, diagonal in specs if not opened}
            for j in target_indices:
                mask, color = objects[j]
                r0, c0 = placements[j]
                logical = mask[::scale, ::scale].astype(np.int8)
                for cells, edge, boundary in _regions(logical, 0):
                    if edge:
                        continue
                    fill = shape_palette.get(_canonical(cells), 0)
                    for r, c in cells:
                        expected[r0+r*scale:r0+(r+1)*scale, c0+c*scale:c0+(c+1)*scale] = fill
            palette = rng.permutation(10).astype(np.int8)
            grid = palette[grid]
            expected = palette[expected]
            try:
                out, info = _recover(grid)
            except AssertionError:
                continue
            if info["scale"] != scale or info["target_holes"] != len(destinations):
                continue
            assert np.array_equal(out, expected), "construction and input recovery disagree"
            if lesson == "matching" and not (info["reflection"] and info["chain"] and info["equal_area"]
                                               and info["multicolor_frame"]):
                continue
            if lesson == "topology" and not (info["diagonal_source"] and info["open"] and info["unmatched"]
                                               and info["self_map"] and info["broadcast"]):
                continue
            return {"input": grid, "output": expected}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    sides = list(map(str, rng.permutation(["top", "left", "bottom", "right"])))
    mode = int(rng.integers(4))
    if mode == 0:
        train_scales, test_scales = [2]*num_train, [3]*num_test
    elif mode == 1:
        train_scales, test_scales = [3]*num_train, [2]*num_test
    elif mode == 2:
        train_scales = [2, 3] + [int(rng.choice([2, 3])) for _ in range(num_train-2)]
        test_scales = [int(rng.choice([2, 3])) for _ in range(num_test)]
    else:
        k = int(rng.choice([2, 3]))
        train_scales, test_scales = [k]*num_train, [k]*num_test
    puzzle, seen = {"train": [], "test": []}, set()
    for split, scales in (("train", train_scales), ("test", test_scales)):
        for i, scale in enumerate(scales):
            # Two directions in the first two demonstrations support novel-side
            # tests; additional training directions never reveal the held-out side.
            side = sides[i % 2] if split == "train" else sides[2+i % 2]
            lesson = ("matching" if i == 0 else "topology" if i == 1 else None) if split == "train" else None
            for _ in range(20):
                pair = _make_pair(rng, scale, side, lesson, group=split == "test", reflected=split == "test")
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "incorrect casting"
            assert inp.shape == out.shape and info["changed"] > 0
            mask = info["change_mask"]
            assert np.array_equal(inp[~mask], out[~mask]), "non-target pixels changed"
            assert np.all(inp[mask] == info["background"])
            assert np.all(out[mask] != info["background"])
            assert not np.any(mask & (inp != info["background"]))
            infos[split].append(info)
    train = infos["train"]
    # Demonstration sufficiency is semantic, reconstructed from the actual grids.
    for feature in ("reflection", "chain", "self_map", "equal_area_different_fill", "diagonal_source",
                    "open", "unmatched", "broadcast", "multicolor_frame", "irregular"):
        assert any(i[feature] for i in train), "missing training witness: " + feature
    assert len({i["side"] for i in train}) >= 2
    assert len({i["source_shapes"] for i in train}) >= 2


def _puzzle_key(puzzle):
    return tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """A bounded 42-seed suite, geometric invariants, and negative diagnostics."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    infos = [_recover(pair["input"])[1] for p in puzzles for s in ("train", "test") for pair in p[s]]
    assert {i["scale"] for i in infos} == {2, 3}
    assert {i["side"] for i in infos} == {"top", "bottom", "left", "right"}
    assert len({i["background"] for i in infos}) == 10
    assert len({i["wall"] for i in infos}) == 10
    assert len({i["input_shape"] for i in infos}) >= 30
    assert len({s for i in infos for s in i["source_shapes"]}) >= 30
    assert len({i["target_holes"] for i in infos}) >= 4
    assert any(i["target_count"] > 1 for i in infos)
    for p in puzzles:
        for s in ("train", "test"):
            for pair in p[s]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp.copy()), out)
        # Equivariance is an input-only test of every boundary direction and D4
        # matching. A full color permutation also changes the background role.
        pair = p["test"][0]
        for q in range(4):
            assert np.array_equal(solve(np.rot90(pair["input"], q)), np.rot90(pair["output"], q))
        assert np.array_equal(solve(np.fliplr(pair["input"])), np.fliplr(pair["output"]))
        perm = np.array([4, 7, 1, 9, 0, 6, 3, 2, 8, 5], dtype=np.int8)
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
        lesson = p["train"][0]
        _, detail = _recover(lesson["input"])
        entries = list(detail["shape_palette"].items())
        assert any(len(a) == len(b) and a != b and ca != cb
                   for a, ca in entries for b, cb in entries), "area-only matching is not defeated"
        # Remove one closed source wall, retaining its key entry. Exactly its
        # geometric address vanishes: a key alone cannot color that large hole.
        src, cells, boundary = detail["source_regions"][0]
        erased = lesson["input"].copy()
        side, mapping, key_cells = _key(erased, detail["background"])
        objects = _components((erased == src) & ~np.array(
            [[(r, c) in key_cells for c in range(erased.shape[1])] for r in range(erased.shape[0])]), True)
        owner = next(obj for obj in objects if boundary <= set(obj))
        for q in owner:
            erased[q] = detail["background"]
        cast = solve(erased)
        assert any(cast[q] != lesson["output"][q] for q in zip(*np.nonzero(detail["change_mask"]))), \
            "geometric source omission did not affect any target"
        # Replace the edge lookup with identity entries. Source-color copying
        # must differ from the chain lesson's specified remote colors.
        identity = lesson["input"].copy()
        for r, c in key_cells:
            if side == "top" and r == 1:
                identity[r, c] = identity[0, c]
            elif side == "bottom" and r == identity.shape[0]-2:
                identity[r, c] = identity[-1, c]
            elif side == "left" and c == 1:
                identity[r, c] = identity[r, 0]
            elif side == "right" and c == identity.shape[1]-2:
                identity[r, c] = identity[r, -1]
        assert not np.array_equal(solve(identity), lesson["output"]), "lookup omission had no effect"
    # All explicit legal count combinations, including the minimal two lessons.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(100+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    info = _recover(broken["test"][0]["input"])[1]
    p = tuple(np.argwhere(info["change_mask"])[0])
    broken["test"][0]["output"][p] = info["background"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong target label accepted")
    # Shape-equivalent compact sources cannot prescribe different colors.
    rng = np.random.default_rng(123)
    s = ((0, 0), (0, 1), (1, 0))
    objects = [(_source_mask(rng, s, False, True), 1),
               (_source_mask(rng, _rotate(s, 1), False, False), 2),
               (_target_mask(rng, [s], 2), 9)]
    diagnostic, positions = _pack(rng, objects, {1: 4, 2: 5, 3: 6}, "top")
    try:
        solve(diagnostic)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting shape colors accepted")
    side, mapping, key_cells = _key(diagnostic, 0)
    for r, c in key_cells:
        if r == 1 and diagnostic[0, c] == 2:
            diagnostic[r, c] = 4
    accepted = solve(diagnostic)
    assert np.any(accepted != diagnostic), "same-color shape tie should be legal"
    # Corrupt one macro wall pixel; the remaining ink no longer decodes fully.
    r0, c0 = positions[-1]
    local = tuple(np.argwhere(objects[-1][0])[0])
    diagnostic[r0+local[0], c0+local[1]] = 0
    try:
        solve(diagnostic)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial macroblock accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all legal counts)")
