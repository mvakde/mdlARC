"""Register arrow-bearing pieces, then count shots by receiver region.

The solver reads only pixels.  Construction uses logical objects, whereas
recovery segments and block-decodes the returned canvas before registering them.
Only Python's standard library and NumPy are needed.
"""

from collections import Counter, defaultdict
import numpy as np


_DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
         (1, 0), (1, -1), (0, -1), (-1, -1))
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DISTANCE = 5
_BACKGROUND, _INK, _COLLISION = 0, 1, 2


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _normalise(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _transform(cells, flip, turns):
    result = [(r, -c if flip else c) for r, c in cells]
    for _ in range(turns):
        result = [(c, -r) for r, c in result]
    return result


def _variants(cells, reflected=True):
    return {_normalise(_transform(cells, f, q))
            for f in range(2 if reflected else 1) for q in range(4)}


def _canonical(cells):
    return min(_variants(cells))


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return {(r+a, c+b) for a, b in offsets}


def _read_arrow(cells):
    matches = [(p, d) for p in sorted(cells) for d in _DIRS
               if _arrow(p, d) == set(cells)]
    assert len(matches) == 1, "Expected one three-cell directional arrow"
    return matches[0]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                nxt = r+dr, c+dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    cells.add(nxt)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells)+1,
            min(c for r, c in cells), max(c for r, c in cells)+1)


def _move(obj, offset):
    a, b = offset
    return dict(obj, cells={(r+a, c+b) for r, c in obj["cells"]},
                arrow={(r+a, c+b) for r, c in obj["arrow"]},
                tip=(obj["tip"][0]+a, obj["tip"][1]+b))


def _endpoint(obj, direction=None):
    dr, dc = obj["direction"] if direction is None else direction
    return obj["tip"][0]+_DISTANCE*dr, obj["tip"][1]+_DISTANCE*dc


def _assemble(board, objects):
    result = board.copy()
    for obj in objects:
        for p in obj["cells"]:
            result[p] = obj["color"]
        for p in obj["arrow"]:
            result[p] = _INK
    return result


def _fire(board, objects, *, point_collisions=False, loose_directions=False):
    """A body is one receiver; each structural cell is a separate receiver."""
    result = _assemble(board, objects)
    owner = {p: i for i, obj in enumerate(objects) for p in obj["cells"]}
    arrivals = defaultdict(list)
    for obj in objects:
        direction = obj.get("loose_direction") if loose_directions else None
        p = _endpoint(obj, direction)
        if not (0 <= p[0] < board.shape[0] and 0 <= p[1] < board.shape[1]):
            assert loose_directions, "All intended endpoints must be in bounds"
            continue  # Diagnostic alternative: an incorrectly aimed shot misses.
        key = ("body", owner[p]) if p in owner else ("cell", *p)
        arrivals[key].append(p)
    for key, points in arrivals.items():
        collision = max(Counter(points).values()) >= 2 if point_collisions else len(points) >= 2
        color = _COLLISION if collision else _INK
        cells = objects[key[1]]["cells"] if key[0] == "body" else {key[1:]}
        for p in cells:
            result[p] = color
    return result, dict(arrivals)


def _expand(grid, scale):
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1).astype(np.int8)


def _decode_piece(grid, cells, scale, structural):
    r0, r1, c0, c1 = _bbox(cells)
    if (r1-r0) % scale or (c1-c0) % scale:
        return None
    crop = np.zeros((r1-r0, c1-c0), dtype=grid.dtype)
    for r, c in cells:
        crop[r-r0, c-c0] = grid[r, c]
    h, w = (r1-r0)//scale, (c1-c0)//scale
    logical = crop[::scale, ::scale]
    if not np.array_equal(crop, _expand(logical, scale)):
        return None
    colors = set(map(int, np.unique(logical))) - {0, 1}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    if color < 3 or color == structural:
        return None
    shape = set(map(tuple, np.argwhere(logical != 0).tolist()))
    if len(shape)*scale*scale != len(cells) or not 4 <= len(shape) <= 12:
        return None
    if len(_variants(shape)) != 8:
        return None  # A symmetric silhouette would leave the command pose ambiguous.
    ink = set(map(tuple, np.argwhere(logical == 1).tolist()))
    try:
        tip, direction = _read_arrow(ink)
    except AssertionError:
        return None
    return dict(cells=shape, arrow=ink, tip=tip, direction=direction, color=color,
                raster_bounds=(r0, r1, c0, c1))


def _register(piece, hole):
    target = _normalise(hole)
    r0, _, c0, _ = _bbox(hole)
    matches = []
    for flip in range(2):
        for turns in range(4):
            transformed = _transform(piece["cells"], flip, turns)
            if _normalise(transformed) != target:
                continue
            a = min(r for r, c in transformed)
            b = min(c for r, c in transformed)
            ink = {(r-a+r0, c-b+c0)
                   for r, c in _transform(piece["arrow"], flip, turns)}
            tip, direction = _read_arrow(ink)
            matches.append(dict(cells=set(hole), arrow=ink, tip=tip,
                                direction=direction, color=piece["color"],
                                loose_direction=piece["direction"],
                                reflected=bool(flip), turns=turns))
    assert len(matches) == 1, "A silhouette must uniquely determine the rigid registration"
    return matches[0]


def _recover(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    assert not np.any(grid == _COLLISION), "Color 2 is reserved for output collisions"
    candidates = []
    for structural in range(3, 10):
        for component in _components(grid == structural):
            r0, r1, c0, c1 = _bbox(component)
            if min(r1-r0, c1-c0) < 5 or max(r1-r0, c1-c0) > 15:
                continue
            board = grid[r0:r1, c0:c1]
            if not (np.all(board[0] == structural) and np.all(board[-1] == structural)
                    and np.all(board[:, 0] == structural) and np.all(board[:, -1] == structural)
                    and np.all((board == 0) | (board == structural))):
                continue
            holes = _components(board == 0)
            if not 1 <= len(holes) <= 5:
                continue
            keys = [_canonical(h) for h in holes]
            if len(set(keys)) != len(keys) or any(len(_variants(h)) != 8 for h in holes):
                continue
            outside = grid.copy()
            outside[r0:r1, c0:c1] = 0
            # Every loose object has a blank moat from the board.
            if np.any(outside[max(0, r0-1):min(grid.shape[0], r1+1),
                              max(0, c0-1):min(grid.shape[1], c1+1)]):
                continue
            components = _components(outside != 0)
            pieces = [c for c in components if len(c) >= 16]
            noise = [c for c in components if len(c) <= 3]
            if len(pieces) != len(holes) or len(pieces)+len(noise) != len(components):
                continue
            for scale in (2, 3):
                if max(board.shape)*scale > 30:
                    continue
                decoded = [_decode_piece(outside, c, scale, structural) for c in pieces]
                if any(p is None for p in decoded):
                    continue
                lookup = {_canonical(p["cells"]): p for p in decoded}
                if len(lookup) != len(decoded) or set(lookup) != set(keys):
                    continue
                fitted = [_register(lookup[key], hole) for key, hole in zip(keys, holes)]
                if any(not (0 <= _endpoint(o)[0] < board.shape[0] and
                            0 <= _endpoint(o)[1] < board.shape[1]) for o in fitted):
                    continue
                if any(_endpoint(o) in o["cells"] for o in fitted):
                    continue
                logical, arrivals = _fire(board, fitted)
                info = dict(board=board.copy(), structural=structural, scale=scale,
                            bounds=(r0, r1, c0, c1), objects=fitted,
                            classes=tuple(sorted(keys)), arrivals=arrivals,
                            noise=sum(map(len, noise)), input_shape=grid.shape)
                candidates.append((_expand(logical, scale), info))
    assert len(candidates) == 1, "Expected exactly one blueprint, scale and registration"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve a member of the family from that input alone, without training data."""
    return _recover(np.asarray(input_grid))[0]


def _body(rng, direction, area=None, compact=False):
    """Procedural connected, asymmetric silhouette containing an arrow command."""
    marker = _arrow((0, 0), direction)
    for _ in range(100):
        cells = set(marker)
        if direction[0]*direction[1] == 0:
            cells.add((-direction[0], -direction[1]))
        size = area if area is not None else int(rng.integers(max(4, len(cells)+1), 11))
        limit = 4 if compact else int(_pick(rng, (3, 4, 4, 5)))
        while len(cells) < size:
            frontier = sorted({(r+a, c+b) for r, c in cells for a, b in _N4} - cells)
            frontier = [p for p in frontier if all(
                max(q[k] for q in cells | {p})-min(q[k] for q in cells | {p}) < limit
                for k in (0, 1))]
            if not frontier:
                break
            cells.add(_pick(rng, frontier))
        if len(cells) == size and len(_variants(cells)) == 8:
            return dict(cells=cells, arrow=set(marker), tip=(0, 0), direction=direction)
    raise RuntimeError("Unable to grow an asymmetric body")


def _separate(objects):
    occupied = set()
    for obj in objects:
        if any((r+a, c+b) in occupied for r, c in obj["cells"]
               for a, b in ((0, 0),) + _N4):
            return False
        occupied.update(obj["cells"])
    return True


def _bounds(objects):
    # A structural border around the bodies, but endpoints may lie on that border.
    points = [(r+a, c+b) for o in objects for r, c in o["cells"]
              for a, b in ((-1, -1), (1, 1))]
    points += [_endpoint(o) for o in objects]
    return (min(r for r, c in points), max(r for r, c in points)+1,
            min(c for r, c in points), max(c for r, c in points)+1)


def _event_features(objects, shape):
    board = np.full(shape, 3, dtype=np.int8)
    colored = [dict(o, color=4) for o in objects]
    _, arrivals = _fire(board, colored)
    return dict(
        single_body=any(k[0] == "body" and len(v) == 1 for k, v in arrivals.items()),
        region_collision=any(k[0] == "body" and len(v) >= 2 and len(set(v)) >= 2
                             for k, v in arrivals.items()),
        point_collision=any(k[0] == "cell" and len(v) >= 2 for k, v in arrivals.items()),
        floor_single=any(k[0] == "cell" and len(v) == 1 for k, v in arrivals.items()))


def _layout(rng, kind, directions, desired, excluded=()):
    # Essential interactions are solved algebraically, then receive random
    # procedural bodies, padding, positions, and optional additional engines.
    for attempt in range(600):
        equal_area = int(rng.integers(6, 10)) if kind == "strike" else None
        target_dir = _pick(rng, directions)
        if kind == "strike":
            target_dir = _pick(rng, [d for d in directions if 0 in d])
        target = _body(rng, target_dir, equal_area)
        objects = [target]
        if kind in ("strike", "region"):
            hit_cells = sorted(target["cells"])
            chosen = rng.choice(len(hit_cells), size=2, replace=False)
            for j in range(1 if kind == "strike" else 2):
                pool = [d for d in directions if 0 not in d] if kind == "strike" else directions
                direction = _pick(rng, pool)
                shooter = _body(rng, direction, equal_area)
                p = hit_cells[int(chosen[j])]
                objects.append(_move(shooter, (p[0]-5*direction[0], p[1]-5*direction[1])))
        elif kind == "point":
            ds = rng.choice(len(directions), size=2, replace=False)
            objects = [_move(_body(rng, directions[int(i)]),
                             (-5*directions[int(i)][0], -5*directions[int(i)][1])) for i in ds]
        if not _separate(objects):
            continue
        keys = [_canonical(o["cells"]) for o in objects]
        if len(set(keys)) != len(keys) or set(keys) & set(excluded):
            continue
        if any(_endpoint(o) in o["cells"] for o in objects):
            continue
        r0, r1, c0, c1 = _bounds(objects)
        if max(r1-r0, c1-c0) > 15:
            continue
        h = int(rng.integers(max(9, r1-r0), 16))
        w = int(rng.integers(max(9, c1-c0), 16))
        offset = (-r0+int(rng.integers(h-(r1-r0)+1)),
                  -c0+int(rng.integers(w-(c1-c0)+1)))
        objects = [_move(o, offset) for o in objects]
        features = _event_features(objects, (h, w))
        if kind == "strike" and not (features["single_body"] and features["floor_single"]):
            continue
        if kind == "region" and not (features["region_collision"] and features["floor_single"]):
            continue
        if kind == "point" and not features["point_collision"]:
            continue
        # Fillers use the same placement law in both splits. They must preserve
        # the intended witness, rather than silently turning a single into a tie.
        for _ in range(140):
            if len(objects) >= desired:
                break
            obj = _move(_body(rng, _pick(rng, directions), compact=True),
                        (int(rng.integers(1, h-1)), int(rng.integers(1, w-1))))
            if any(not (0 < r < h-1 and 0 < c < w-1) for r, c in obj["cells"]):
                continue
            p = _endpoint(obj)
            if not (0 <= p[0] < h and 0 <= p[1] < w) or p in obj["cells"]:
                continue
            key = _canonical(obj["cells"])
            if key in keys or key in excluded or not _separate(objects+[obj]):
                continue
            after = _event_features(objects+[obj], (h, w))
            if kind == "strike" and not (after["single_body"] and after["floor_single"]):
                continue
            if kind == "region" and not (after["region_collision"] and after["floor_single"]):
                continue
            objects.append(obj)
            keys.append(key)
        if len(objects) < desired:
            continue
        return objects, (h, w)
    raise RuntimeError("Bounded event construction exhausted")


def _small_layout(rng, directions, excluded):
    # Scale-three scenes retain genuine pose and scale reasoning in a smaller
    # blueprint. A single engine also supplies especially clear distance evidence.
    for _ in range(150):
        obj = _body(rng, _pick(rng, directions))
        if _canonical(obj["cells"]) in excluded or _endpoint(obj) in obj["cells"]:
            continue
        r0, r1, c0, c1 = _bounds([obj])
        if max(r1-r0, c1-c0) > 10:
            continue
        shape = (int(rng.integers(max(7, r1-r0), 11)),
                 int(rng.integers(max(7, c1-c0), 11)))
        offset = (-r0+int(rng.integers(shape[0]-(r1-r0)+1)),
                  -c0+int(rng.integers(shape[1]-(c1-c0)+1)))
        return [_move(obj, offset)], shape
    raise RuntimeError("Bounded small-board construction exhausted")


def _loose_raster(rng, obj, scale, pose):
    flip = int(rng.integers(2)) if pose is None else int(pose == "reflection")
    turns = int(rng.integers(4)) if pose != "rotation" else int(rng.integers(1, 4))
    body = _transform(obj["cells"], flip, turns)
    ink = _transform(obj["arrow"], flip, turns)
    a, b = min(r for r, c in body), min(c for r, c in body)
    body = {(r-a, c-b) for r, c in body}
    ink = {(r-a, c-b) for r, c in ink}
    obj["loose_direction"] = _read_arrow(ink)[1]
    h, w = max(r for r, c in body)+1, max(c for r, c in body)+1
    raster = np.zeros((h, w), dtype=np.int8)
    for p in body:
        raster[p] = obj["color"]
    for p in ink:
        raster[p] = _INK
    return _expand(raster, scale)


def _free_origins(blocked, mask):
    h, w = mask.shape
    H, W = blocked.shape
    if h > H or w > W:
        return []
    free = np.ones((H-h+1, W-w+1), dtype=bool)
    for r, c in np.argwhere(mask):
        free &= ~blocked[r:r+H-h+1, c:c+W-w+1]
    return np.argwhere(free)


def _pack(rng, board, loose):
    rasters = [board] + loose
    masks = [np.ones(board.shape, dtype=bool)] + [p != 0 for p in loose]
    order = [0] + sorted(range(1, len(rasters)), key=lambda i: -np.count_nonzero(masks[i]))
    for attempt in range(35):
        shape = (int(rng.integers(max(17, board.shape[0]), 31)),
                 int(rng.integers(max(17, board.shape[1]), 31))) if attempt < 12 else (30, 30)
        grid = np.zeros(shape, dtype=np.int8)
        blocked = np.zeros(shape, dtype=bool)
        for i in order:
            origins = _free_origins(blocked, masks[i])
            if len(origins) == 0:
                break
            a, b = map(int, origins[int(rng.integers(len(origins)))])
            raster, mask = rasters[i], masks[i]
            h, w = raster.shape
            grid[a:a+h, b:b+w][mask] = raster[mask]
            for r, c in np.argwhere(mask):
                blocked[max(0, a+r-1):min(shape[0], a+r+2),
                        max(0, b+c-1):min(shape[1], b+c+2)] = True
        else:
            if rng.random() < .3:
                # Optional sub-macrocell noise, always disconnected and size <4.
                for _ in range(int(rng.integers(1, 7))):
                    free = np.argwhere(~blocked)
                    if not len(free):
                        break
                    r, c = map(int, free[int(rng.integers(len(free)))])
                    grid[r, c] = int(_pick(rng, (1, 3, 4, 6, 7, 8, 9)))
                    blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
            if rng.random() < .8:
                r0, r1, c0, c1 = _bbox(set(map(tuple, np.argwhere(grid != 0).tolist())))
                r0, c0 = max(0, r0-int(rng.integers(3))), max(0, c0-int(rng.integers(3)))
                r1, c1 = min(shape[0], r1+int(rng.integers(3))), min(shape[1], c1+int(rng.integers(3)))
                grid = grid[r0:r1, c0:c1].copy()
            return grid
    raise RuntimeError("Bounded two-dimensional packing exhausted")


def _make_pair(rng, kind, directions, desired, scale, pose, excluded):
    for _ in range(15):
        if scale == 3:
            objects, shape = _small_layout(rng, directions, excluded)
        else:
            objects, shape = _layout(rng, kind, directions, desired, excluded)
        palette = list(map(int, rng.permutation(np.arange(3, 10))))
        structural, available = palette[0], palette[1:]
        for i, obj in enumerate(objects):
            obj["color"] = (available[0] if pose == "rotation" else
                            available[i] if kind == "strike" else int(_pick(rng, available)))
        board = np.full(shape, structural, dtype=np.int8)
        for obj in objects:
            for p in obj["cells"]:
                board[p] = 0
        loose = [_loose_raster(rng, o, scale, pose if i == 0 else None)
                 for i, o in enumerate(objects)]
        logical, _ = _fire(board, objects)
        if pose is not None and np.array_equal(
                _fire(board, objects, loose_directions=True)[0], logical):
            continue  # Essential examples must visibly teach command transport.
        expected = _expand(logical, scale)
        try:
            grid = _pack(rng, board, loose)
        except RuntimeError:
            continue
        recovered, info = _recover(grid)
        assert np.array_equal(recovered, expected), "Construction and input recovery disagree"
        return {"input": grid, "output": expected}, info
    raise RuntimeError("Bounded pair packing exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    cards = [d for d in _DIRS if 0 in d]
    diagonals = [d for d in _DIRS if 0 not in d]
    if regime == 0:
        train_dirs = tuple(cards[int(i)] for i in rng.choice(4, 2, replace=False)) + \
                     tuple(diagonals[int(i)] for i in rng.choice(4, 2, replace=False))
        test_dirs = tuple(d for d in _DIRS if d not in train_dirs)
    else:
        train_dirs, test_dirs = _DIRS, _DIRS
    specs = [("strike", 2, 2, "reflection"), ("region", 3, 2, "rotation")]
    for _ in range(num_train-2):
        scale = 3 if regime == 2 and rng.random() < .4 else 2
        kind = str(_pick(rng, ("simple", "strike", "region", "point")))
        count = 1 if scale == 3 else max(3 if kind == "region" else 2 if kind != "simple" else 1,
                                       int(rng.integers(1, 4)))
        specs.append((kind, count, scale, None))
    rng.shuffle(specs)
    puzzle, seen, train_classes = {"train": [], "test": []}, set(), set()
    for split, count, directions in (("train", num_train, train_dirs), ("test", num_test, test_dirs)):
        for i in range(count):
            if split == "train":
                kind, desired, scale, pose = specs[i]
                excluded = ()
            else:
                scale = 3 if regime == 1 else 2
                kind = str(_pick(rng, ("simple", "strike", "region", "point")))
                desired = 1 if scale == 3 else 4 if regime == 2 else \
                    max(3 if kind == "region" else 2 if kind != "simple" else 1,
                        int(rng.integers(1, 5)))
                pose, excluded = None, train_classes
            for _ in range(20):
                pair, info = _make_pair(rng, kind, directions, desired, scale, pose, excluded)
                key = _grid_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_classes.update(info["classes"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    objects, arrivals = info["objects"], info["arrivals"]
    board = info["board"]
    struck = {k[1] for k in arrivals if k[0] == "body"}
    return dict(
        scale=info["scale"], count=len(objects), board_shape=board.shape,
        input_shape=info["input_shape"], structural=info["structural"], noise=info["noise"],
        classes=info["classes"], directions={o["direction"] for o in objects},
        reflection=any(o["reflected"] for o in objects),
        rotation=any(not o["reflected"] and o["turns"] for o in objects),
        equal_area=len({len(o["cells"]) for o in objects}) < len(objects),
        shared_colors=len({o["color"] for o in objects}) < len(objects),
        same_color_unhit=any(objects[i]["color"] == o["color"]
                             for i in struck for j, o in enumerate(objects) if j not in struck),
        single_body=any(k[0] == "body" and len(v) == 1 for k, v in arrivals.items()),
        region_collision=any(k[0] == "body" and len(v) >= 2 and len(set(v)) >= 2
                             for k, v in arrivals.items()),
        point_collision=any(k[0] == "cell" and len(v) >= 2 for k, v in arrivals.items()),
        floor_single=any(k[0] == "cell" and len(v) == 1 for k, v in arrivals.items()),
        no_fire_diff=not np.array_equal(_assemble(board, objects), _fire(board, objects)[0]),
        no_pose_diff=not np.array_equal(_fire(board, objects, loose_directions=True)[0],
                                      _fire(board, objects)[0]),
        point_only_diff=not np.array_equal(_fire(board, objects, point_collisions=True)[0],
                                         _fire(board, objects)[0]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features, classes = set(), {"train": [], "test": []}, {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for value in pair.values():
                grid = np.asarray(value)
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            expected, info = _recover(pair["input"])
            assert np.array_equal(expected, np.asarray(pair["output"])), "Incorrect output"
            assert not np.any(expected == 0), "Every socket and structural cell is retained and filled"
            assert len(_components(info["board"] == 0)) == len(info["objects"])
            features[split].append(_features(info))
            classes[split].update(info["classes"])
    train = features["train"]
    for name in ("reflection", "rotation", "single_body", "region_collision", "floor_single",
                 "equal_area", "same_color_unhit", "no_fire_diff", "no_pose_diff", "point_only_diff"):
        assert any(f[name] for f in train), f"Missing training evidence: {name}"
    assert any(0 in d for f in train for d in f["directions"])
    assert any(0 not in d for f in train for d in f["directions"])
    assert classes["test"].isdisjoint(classes["train"]), "Queries use unseen silhouette classes"
    queries = features["test"]
    train_directions = set().union(*(f["directions"] for f in train))
    query_directions = set().union(*(f["directions"] for f in queries))
    train_scales, query_scales = {f["scale"] for f in train}, {f["scale"] for f in queries}
    assert (query_directions-train_directions or query_scales-train_scales or
            max(f["count"] for f in queries) > max(f["count"] for f in train)), \
        "Queries need a direction, scale or interaction-count shift"


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded diversity audit, pixel invariants, ablations and invalid requests."""
    keys, rows, shifts = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        keys.add(_puzzle_key(puzzle))
        grouped = {s: [_features(_recover(p["input"])[1]) for p in puzzle[s]] for s in puzzle}
        for split in grouped:
            rows.extend((split, f) for f in grouped[split])
        td = set().union(*(f["directions"] for f in grouped["train"]))
        qd = set().union(*(f["directions"] for f in grouped["test"]))
        ts = {f["scale"] for f in grouped["train"]}
        qs = {f["scale"] for f in grouped["test"]}
        tc = max(f["count"] for f in grouped["train"])
        qc = max(f["count"] for f in grouped["test"])
        assert qd-td or qs-ts or qc > tc, "Each puzzle has a visible structural shift"
        shifts.add((bool(qd-td), bool(qs-ts), qc > tc))
    assert len(keys) == 42 and len(shifts) >= 3
    all_rows = [f for s, f in rows]
    assert {f["scale"] for f in all_rows} == {2, 3}
    assert {f["count"] for f in all_rows} >= {1, 2, 3, 4}
    assert len({f["input_shape"] for f in all_rows}) >= 35
    assert len({f["board_shape"] for f in all_rows}) >= 20
    assert len({k for f in all_rows for k in f["classes"]}) >= 100
    assert set().union(*(f["directions"] for f in all_rows)) == set(_DIRS)
    assert {f["structural"] for f in all_rows} == set(range(3, 10))
    assert all(any(f[n] for f in all_rows) for n in
               ("shared_colors", "point_collision", "region_collision", "single_body", "reflection", "rotation"))
    first = generate_puzzle(19, 2, 1)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(19, 2, 1))
    # Spatial conventions must survive global rotations and reflections; the
    # solver cannot accidentally rely on a shelf, orientation or canvas corner.
    for pair in first["train"]+first["test"]:
        for transform in (np.rot90, np.fliplr):
            assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
    # Whole-object simultaneous reception is different from coordinate-only
    # collision counting. The collided engine still fires its own visible shot.
    for pair in first["train"]:
        expected, info = _recover(pair["input"])
        f = _features(info)
        if f["region_collision"]:
            assert f["point_only_diff"] and f["same_color_unhit"] and f["floor_single"]
            assert np.any(expected == 2)
    # Exercise the specified domain beyond the ordinary sampler's count/area
    # choices, without changing the input-only solver or introducing exceptions.
    probe_rng = np.random.default_rng(8102)
    pair, info = _make_pair(probe_rng, "region", _DIRS, 5, 2, None, set())
    assert len(info["objects"]) == 5
    assert np.array_equal(solve(pair["input"]), pair["output"])
    for area in (11, 12):
        obj = _body(probe_rng, (1, 1), area=area)
        a, b, c, d = _bounds([obj])
        obj = _move(obj, (-a, -c))
        obj["color"] = 4
        board = np.full((b-a, d-c), 3, dtype=np.int8)
        for p in obj["cells"]:
            board[p] = 0
        loose = _loose_raster(probe_rng, obj, 2, "reflection")
        inp = _pack(probe_rng, board, [loose])
        assert np.array_equal(solve(inp), _expand(_fire(board, [obj])[0], 2))
    # An incomplete macrocell must never be silently accepted as a smaller scale.
    _, info = _recover(first["train"][0]["input"])
    damaged = first["train"][0]["input"].copy()
    a, b, c, d = info["bounds"]
    outside = damaged.copy()
    outside[a:b, c:d] = 0
    large = next(cells for cells in _components(outside != 0) if len(cells) >= 16)
    damaged[min(large)] = 0
    try:
        solve(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incomplete macrocell accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(100+a*4+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (3, 4), (True, 1),
                 (4.0, 1), (None, 1), (4, None), (4, False), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
