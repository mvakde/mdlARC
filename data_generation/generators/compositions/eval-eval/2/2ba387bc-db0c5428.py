"""Rank-paired telescopes: ordinal anchors, inward signals, radial half-turns.

Only Python's standard library and NumPy are used. The solver parses its input;
sampling profiles and RNG state never enter solving or label validation.
"""

from itertools import combinations
import numpy as np


HOLLOW, SOLID = 0, 1
SIZES = (3, 5, 7)
PAIRING_ALTERNATIVES = (
    "horizontal", "horizontal_reverse", "vertical_reverse", "nearest", "greedy"
)


def _check_grid(value):
    grid = np.asarray(value)
    if grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("grid cells must be integers")
    if not np.all((0 <= grid) & (grid <= 9)):
        raise ValueError("grid colors must be 0..9")
    return grid


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _queues(objects, mode="rank"):
    queues = [[o for o in objects if o["kind"] == k] for k in (HOLLOW, SOLID)]
    if mode.startswith("horizontal"):
        key = lambda o: (o["left"], o["top"])
    else:
        key = lambda o: (o["top"], o["left"])
    reverse = mode in ("horizontal_reverse", "vertical_reverse")
    return [sorted(q, key=key, reverse=reverse) for q in queues]


def _pairs(objects, mode="rank"):
    hollows, solids = _queues(objects, mode)
    if mode == "nearest":
        # A plausible one-to-one distance shortcut, with visible tie breakers.
        result = []
        while hollows and solids:
            h, s = min(((h, s) for h in hollows for s in solids), key=lambda p: (
                (p[0]["top"]-p[1]["top"])**2 + (p[0]["left"]-p[1]["left"])**2,
                p[0]["top"], p[0]["left"], p[1]["top"], p[1]["left"]))
            result.append((h, s))
            hollows.remove(h)
            solids.remove(s)
        return result
    if mode == "greedy":
        rows = [[None, None]]
        for obj in sorted(objects, key=lambda o: (o["top"], o["left"])):
            kind = obj["kind"]
            if rows[-1][kind] is not None:
                rows.append([None, None])
            rows[-1][kind] = obj
        return [(h, s) for h, s in rows if h is not None and s is not None]
    return list(zip(hollows, solids))


def _destination(h, s):
    return 2*s["top"]-h["top"], 2*s["left"]-h["left"]


def _inward(h, s, size):
    dy, dx = s["top"]-h["top"], s["left"]-h["left"]
    mid = size//2
    # Project the frame's pixel-center onto the source's closed footprint.
    # For shallow rays at larger sizes the closest pixel need not be a corner.
    return max(0, min(size-1, mid-dy)), max(0, min(size-1, mid-dx))


def _compass_inward(h, s, size):
    dy, dx = s["top"]-h["top"], s["left"]-h["left"]
    mid = size//2
    return (0 if dy > 0 else size-1 if dy < 0 else mid,
            0 if dx > 0 else size-1 if dx < 0 else mid)


def _separated(positions, size):
    return all(abs(a[0]-b[0]) > size or abs(a[1]-b[1]) > size
               for a, b in combinations(positions, 2))


def _position_objects(positions):
    return [dict(id=i, kind=int(k), top=int(r), left=int(c))
            for i, (k, r, c) in enumerate(positions)]


def _layout_ok(positions, size, shape, *, cardinal=False, radius_max=None,
               radius_min=None, shallow=False):
    height, width = shape
    if any(not (1 <= r <= height-size-1 and 1 <= c <= width-size-1)
           for _, r, c in positions):
        return False
    for kind in (HOLLOW, SOLID):
        rows = [r for k, r, _ in positions if k == kind]
        if not rows or len(rows) != len(set(rows)):
            return False
    objects = _position_objects(positions)
    pairs = _pairs(objects)
    destinations = [_destination(h, s) for h, s in pairs]
    if any(not (0 <= r <= height-size and 0 <= c <= width-size)
           for r, c in destinations):
        return False
    if not _separated([(r, c) for _, r, c in positions] + destinations, size):
        return False
    vectors = [(s["top"]-h["top"], s["left"]-h["left"]) for h, s in pairs]
    if cardinal and not any(dy == 0 or dx == 0 for dy, dx in vectors):
        return False
    if shallow and not any(0 < abs(dy) < size//2 or 0 < abs(dx) < size//2
                           for dy, dx in vectors):
        return False
    radius = max(max(abs(dy), abs(dx)) for dy, dx in vectors)
    return ((radius_max is None or radius <= radius_max) and
            (radius_min is None or radius >= radius_min))


def _analyze(input_grid):
    grid = _check_grid(input_grid)
    background = int(grid[0, 0])
    if not all(np.all(edge == background)
               for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("input requires a uniform background border")
    todo = set(map(tuple, np.argwhere(grid != background).tolist()))
    objects = []
    sizes = set()
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, component = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    component.append(q)
                    stack.append(q)
        r = min(p[0] for p in component)
        c = min(p[1] for p in component)
        bottom = max(p[0] for p in component)+1
        right = max(p[1] for p in component)+1
        size = bottom-r
        if size not in SIZES or right-c != size:
            raise ValueError("expected complete odd square tiles of size 3, 5, or 7")
        tile = grid[r:bottom, c:right].copy()
        if np.all(tile != background):
            kind = SOLID
        else:
            color = tile[0, 0]
            if color == background or not all(np.all(edge == color)
                    for edge in (tile[0], tile[-1], tile[:, 0], tile[:, -1])):
                raise ValueError("hollow tile needs a monochrome one-cell rim")
            if not np.all(tile[1:-1, 1:-1] == background):
                raise ValueError("hollow tile interior must be background")
            kind = HOLLOW
        sizes.add(size)
        objects.append(dict(id=len(objects), kind=kind, top=r, left=c, tile=tile))
    if len(sizes) != 1:
        raise ValueError("each input needs one common tile size")
    size = sizes.pop()
    positions = [(o["kind"], o["top"], o["left"]) for o in objects]
    if not _layout_ok(positions, size, grid.shape):
        raise ValueError("invalid queues, overlapping footprints, or clipped projection")
    return dict(grid=grid, background=background, size=size, objects=objects,
                pairs=_pairs(objects))


def _render(info):
    output = info["grid"].copy()
    size = info["size"]
    for h, s in info["pairs"]:
        y, x = _inward(h, s, size)
        r, c = h["top"], h["left"]
        output[r+1:r+size-1, c+1:c+size-1] = s["tile"][y, x]
        r, c = _destination(h, s)
        output[r:r+size, c:c+size] = s["tile"][::-1, ::-1]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible objects and coordinates only."""
    return _render(_analyze(input_grid))


def _dihedral(tile):
    return [np.rot90(tile, k) for k in range(4)] + [
        np.rot90(tile[:, ::-1], k) for k in range(4)]


def _alternative(info, *, pairing="rank", sample="inward", operation=2,
                 project=True, single_pixel=False):
    """Finite authored shortcut checks, not a second solver or learned rule."""
    size, bg = info["size"], info["background"]
    pairs = _pairs(info["objects"], pairing)
    destinations = [_destination(h, s) for h, s in pairs] if project else []
    if any(not (0 <= r <= info["grid"].shape[0]-size and
                0 <= c <= info["grid"].shape[1]-size) for r, c in destinations):
        return None
    if not _separated([(o["top"], o["left"]) for o in info["objects"]] + destinations, size):
        return None
    output = info["grid"].copy()
    for h, s in pairs:
        tile = s["tile"]
        y, x = _inward(h, s, size)
        if sample == "center":
            value = tile[size//2, size//2]
        elif sample == "outward":
            value = tile[size-1-y, size-1-x]
        elif sample == "compass":
            value = tile[_compass_inward(h, s, size)]
        elif sample in ("rare", "majority"):
            colors, counts = np.unique(tile, return_counts=True)
            carrier = int(colors[np.argmax(counts)])
            if sample == "majority" or len(colors) == 1:
                value = carrier
            else:
                choices = [(int(n), int(c)) for c, n in zip(colors, counts) if c != carrier]
                value = min(choices)[1]
        else:
            value = tile[y, x]
        r, c = h["top"], h["left"]
        if single_pixel:
            output[r+size//2, c+size//2] = value
        else:
            output[r+1:r+size-1, c+1:c+size-1] = value
        if project:
            if operation == "radial":
                dy, dx = s["top"]-h["top"], s["left"]-h["left"]
                view = tile[::-1, :] if dx == 0 else tile[:, ::-1] if dy == 0 else tile[::-1, ::-1]
            else:
                view = _dihedral(tile)[operation]
            r, c = _destination(h, s)
            output[r:r+size, c:c+size] = view
    return output


def _alternatives():
    result = [(name, dict(pairing=name)) for name in PAIRING_ALTERNATIVES]
    result += [("sample_"+name, dict(sample=name))
               for name in ("center", "outward", "rare", "majority", "compass")]
    result += [("D4_"+str(k), dict(operation=k)) for k in range(8) if k != 2]
    result += [("radial_reflection", dict(operation="radial")),
               ("no_projection", dict(project=False)),
               ("single_aperture_pixel", dict(single_pixel=True))]
    return result


def _agrees(info, output, configuration):
    alternative = _alternative(info, **configuration)
    return alternative is not None and np.array_equal(alternative, output)


def _sample_layout(rng, size, counts, *, witness=False, radius_max=None,
                   radius_min=None, shallow=False):
    n, m = counts
    step = size+1 if radius_min is None else radius_min
    horizontal = bool(rng.integers(2))
    positions = []
    for kind, count in ((HOLLOW, n), (SOLID, m)):
        for rank in range(count):
            offset = rank*(size+3) if shallow else rank*(size+1)
            # Size-five's fourth (surplus) tile uses the last inset coordinate.
            if shallow and rank == 3:
                offset -= 1
            if horizontal:
                positions.append((kind, 1+offset+(1 if shallow and kind == SOLID and rank == 0 else 0),
                                  1+kind*step))
            else:
                positions.append((kind, 1+kind*step+rank,
                                  1+offset+(1 if shallow and kind == SOLID and rank == 0 else 0)))
    objects = _position_objects(positions)
    footprints = [(r, c) for _, r, c in positions] + [
        _destination(h, s) for h, s in _pairs(objects)]
    min_h = max(r for r, _ in footprints)+size+1
    min_w = max(c for _, c in footprints)+size+1
    if witness:
        min_h, min_w = max(22, min_h), max(23, min_w)
    assert min_h <= 30 and min_w <= 30
    mode = int(rng.integers(6))
    if mode == 0:
        height = int(rng.integers(min_h, min(30, min_h+2)+1))
        width = int(rng.integers(min_w, min(30, min_w+2)+1))
    elif mode == 1:
        height = width = int(rng.integers(max(min_h, min_w), 31))
    elif mode == 2:
        height, width = 30, int(rng.integers(min_w, 31))
    elif mode == 3:
        height, width = int(rng.integers(min_h, 31)), 30
    elif mode == 4:
        height = width = 30
    else:
        height, width = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    shape = height, width
    dr = int(rng.integers(height-min_h+1))
    dc = int(rng.integers(width-min_w+1))
    positions = [(k, r+dr, c+dc) for k, r, c in positions]
    options = dict(cardinal=witness, radius_max=radius_max, radius_min=radius_min, shallow=shallow)
    # Reflections are accepted only after recomputing the rank associations.
    for axis in rng.permutation(2):
        if rng.random() < .5:
            reflected = [(k, height-size-r if axis == 0 else r,
                          width-size-c if axis == 1 else c) for k, r, c in positions]
            if _layout_ok(reflected, size, shape, **options):
                positions = reflected
    assert _layout_ok(positions, size, shape, **options)
    # Free relocation checks all derived destinations, including changed ranks.
    for _ in range(4):
        for idx in rng.permutation(len(positions)):
            kind = positions[int(idx)][0]
            for _ in range(24):
                r = int(rng.integers(1, height-size))
                c = int(rng.integers(1, width-size))
                proposal = positions.copy()
                proposal[int(idx)] = kind, r, c
                if _layout_ok(proposal, size, shape, **options):
                    positions = proposal
                    break
    return shape, positions


def _mapping_signature(objects, mode="rank"):
    return frozenset((h["id"], s["id"]) for h, s in _pairs(objects, mode))


def _fallback_layout(rng, counts, radius_max):
    """A guaranteed small-radius witness, followed by safe free relocation.

    The first pair always has size three, two or three matches, and one surplus
    object. This construction handles either surplus side without exceptions
    to the rule. It is used only if random witness sampling exhausts its bound.
    """
    n, m = counts
    assert min(n, m) in (2, 3) and abs(n-m) == 1
    frames = [(4, 9), (7, 13), (12, 3)][:min(n, m)]
    sources = [(9, 9), (11, 15), (17, 3)][:min(n, m)]
    positions = [(HOLLOW, r, c) for r, c in frames] + [
        (SOLID, r, c) for r, c in sources]
    positions.append((HOLLOW if n > m else SOLID, 22, 11))
    height, width = int(rng.integers(26, 31)), int(rng.integers(21, 31))
    dr, dc = int(rng.integers(height-25)), int(rng.integers(width-20))
    positions = [(k, r+dr, c+dc) for k, r, c in positions]
    if rng.integers(2):
        positions = [(k, r, width-3-c) for k, r, c in positions]
    shape = height, width

    def acceptable(proposal):
        if not _layout_ok(proposal, 3, shape, cardinal=True, radius_max=radius_max):
            return False
        objects = _position_objects(proposal)
        signature = _mapping_signature(objects)
        return all(_mapping_signature(objects, mode) != signature
                   for mode in PAIRING_ALTERNATIVES)

    assert acceptable(positions)
    for _ in range(2):
        for idx in rng.permutation(len(positions)):
            for _ in range(8):
                proposal = positions.copy()
                proposal[int(idx)] = (positions[int(idx)][0],
                                      int(rng.integers(1, height-3)),
                                      int(rng.integers(1, width-3)))
                if acceptable(proposal):
                    positions = proposal
                    break
    return shape, positions


def _witness_source(rng, size, carrier, details, inward, repeated, forbidden, protected=()):
    center = (size//2, size//2)
    outward = (size-1-inward[0], size-1-inward[1])
    available = [(r, c) for r in range(size) for c in range(size)
                 if (r, c) not in (center, inward, outward) and (r, c) not in protected]
    for _ in range(24):
        sample, auxiliary = map(int, rng.choice(details, size=2, replace=False))
        shuffled = [available[int(i)] for i in rng.permutation(len(available))]
        repeat_sites = shuffled if repeated else [None]
        for repeat in repeat_sites:
            for extra in shuffled:
                if extra == repeat:
                    continue
                tile = np.full((size, size), carrier, dtype=np.int8)
                tile[inward] = sample
                if repeat is not None:
                    tile[repeat] = sample
                tile[extra] = auxiliary
                if len({t.tobytes() for t in _dihedral(tile)}) == 8 and tile.tobytes() not in forbidden:
                    return tile
    raise RuntimeError("bounded asymmetric source construction exhausted")


def _ordinary_source(rng, size, carrier, details, inward, richness):
    tile = np.full((size, size), carrier, dtype=np.int8)
    if rng.random() < (.35 if richness == 0 else .16):
        return tile
    budget = int(rng.integers(1, min((size*size-1)//2, size+2+richness*3)+1))
    mode = int(rng.integers(4))
    cells = [(r, c) for r in range(size) for c in range(size)]
    if mode == 0:
        chosen = [cells[int(i)] for i in rng.permutation(len(cells))[:budget]]
    elif mode == 1:
        axis, fixed = int(rng.integers(2)), int(rng.integers(size))
        shuffled = [cells[int(i)] for i in rng.permutation(len(cells))]
        chosen = sorted(shuffled, key=lambda p: p[axis] != fixed)[:budget]
    elif mode == 2:
        start = inward if inward is not None and rng.random() < .5 else cells[int(rng.integers(len(cells)))]
        cluster = {start}
        while len(cluster) < budget:
            frontier = sorted({q for r, c in cluster for q in
                ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                if 0 <= q[0] < size and 0 <= q[1] < size} - cluster)
            cluster.add(frontier[int(rng.integers(len(frontier)))])
        chosen = sorted(cluster)
    else:
        rng.shuffle(cells)
        cluster = set()
        for r, c in cells:
            pair = {(r, c), (size-1-r, size-1-c)}
            if len(cluster | pair) <= budget:
                cluster |= pair
        chosen = sorted(cluster)
    color = int(rng.choice(details))
    for p in chosen:
        tile[p] = int(rng.choice(details)) if mode == 0 and rng.random() < .4 else color
    return tile


def _make_pair(rng, size, counts, background, richness, *, first=False,
               second=False, radius_max=None, radius_min=None):
    for attempt in range(65):
        if first and attempt == 64:
            shape, positions = _fallback_layout(rng, counts, radius_max)
        else:
            shape, positions = _sample_layout(rng, size, counts, witness=first,
                                              radius_max=radius_max, radius_min=radius_min,
                                              shallow=second and size > 3)
        objects = _position_objects(positions)
        if first:
            signature = _mapping_signature(objects)
            if any(_mapping_signature(objects, mode) == signature for mode in PAIRING_ALTERNATIVES):
                continue
        palette = [int(c) for c in rng.permutation([c for c in range(10) if c != background])]
        carrier, details = palette[0], palette[1:]
        paired = {s["id"]: _inward(h, s, size) for h, s in _pairs(objects)}
        compass = {s["id"]: _compass_inward(h, s, size) for h, s in _pairs(objects)}
        grid = np.full(shape, background, dtype=np.int8)
        source_keys = set()
        frame_index = 0
        for obj in objects:
            if obj["kind"] == HOLLOW:
                color = carrier if frame_index == 0 else int(rng.choice(palette[:int(rng.integers(2, 10))]))
                tile = np.full((size, size), color, dtype=np.int8)
                tile[1:-1, 1:-1] = background
                frame_index += 1
            else:
                inward = paired.get(obj["id"])
                if (first or second) and inward is not None:
                    protected = (compass[obj["id"]],) if second and compass[obj["id"]] != inward else ()
                    tile = _witness_source(rng, size, carrier, details, inward, second, source_keys, protected)
                else:
                    tile = _ordinary_source(rng, size, carrier, details, inward, richness)
                source_keys.add(tile.tobytes())
            r, c = obj["top"], obj["left"]
            grid[r:r+size, c:c+size] = tile
        info = _analyze(grid)
        output = _render(info)
        if first:
            required = [(name, config) for name, config in _alternatives()
                        if name not in ("sample_rare", "single_aperture_pixel")]
            required = [(name, config) for name, config in required if name != "sample_compass"]
            if any(_agrees(info, output, config) for _, config in required):
                continue
        return {"input": grid, "output": output}
    raise RuntimeError("bounded teaching-witness construction exhausted")


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name+" is out of range")
    return value


def _capacity(size):
    return {3: 6, 5: 4, 7: 3}[size]


def _counts(rng, size, *, sign=None, large_tail=False, small_gap=False):
    limit = _capacity(size)
    if sign is not None:
        small = int(rng.integers(2, min(3, limit-1)+1))
        return (small+1, small) if sign < 0 else (small, small+1)
    if large_tail:
        small = int(rng.integers(1, limit-1))
        large = int(rng.integers(small+2, limit+1))
        return (small, large) if rng.integers(2) else (large, small)
    if small_gap:
        small = int(rng.integers(1, limit+1))
        gap = int(rng.integers(-1, 2))
        return small, int(np.clip(small+gap, 1, limit))
    if rng.random() < .25:
        same = int(rng.integers(1, limit+1))
        return same, same
    return tuple(map(int, rng.integers(1, limit+1, size=2)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("tail", "scale", "reach"), p=(.5, .25, .25)))
    constant_size = regime == "reach" or (regime == "tail" and rng.random() < .3)
    fixed_background = bool(rng.random() < .25)
    base_background = int(rng.integers(10))
    richness = int(rng.integers(3))
    first_sign = 1 if rng.integers(2) else -1
    puzzle, seen = {"train": [], "test": []}, set()
    for index in range(num_train+num_test):
        train = index < num_train
        first, second = index == 0, index == 1
        first_query = index == num_train
        if first or constant_size:
            size = 3
        elif second:
            size = 5
        elif regime == "scale" and train:
            size = int(rng.choice((3, 5)))
        elif regime == "scale" and first_query:
            size = 7
        else:
            size = int(rng.choice(SIZES, p=(.55, .30, .15)))
        if first or second:
            counts = _counts(rng, size, sign=first_sign if first else -first_sign)
        elif regime == "tail" and first_query:
            counts = _counts(rng, size, large_tail=True)
        else:
            counts = _counts(rng, size, small_gap=(regime == "tail" and train))
        bg = base_background if fixed_background else int(rng.integers(10))
        for _ in range(32):
            pair = _make_pair(rng, size, counts, bg, richness, first=first, second=second,
                              radius_max=7 if regime == "reach" and train else None,
                              radius_min=8 if regime == "reach" and first_query else None)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded distinct-input construction exhausted")
        seen.add(key)
        puzzle["train" if train else "test"].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys"
    try:
        a, b = _check_grid(pair["input"]), _check_grid(pair["output"])
        info = _analyze(a)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    assert a.shape == b.shape, "output dimensions changed"
    size, bg = info["size"], info["background"]
    assigned = np.zeros(a.shape, dtype=bool)
    # Scalar label checks deliberately do not call the slice-based solve/render.
    for h, s in info["pairs"]:
        y, x = _inward(h, s, size)
        mid = size//2
        distances = [((s["top"]+yy-h["top"]-mid)**2+
                      (s["left"]+xx-h["left"]-mid)**2, yy, xx)
                     for yy in range(size) for xx in range(size)]
        nearest = min(distances)
        assert sum(d == nearest[0] for d, _, _ in distances) == 1, "sample-distance tie"
        assert (y, x) == nearest[1:], "sample is not the nearest source pixel"
        sample = int(a[s["top"]+y, s["left"]+x])
        for yy in range(1, size-1):
            for xx in range(1, size-1):
                r, c = h["top"]+yy, h["left"]+xx
                assert a[r, c] == bg and b[r, c] == sample, "incorrect aperture signal"
                assert not assigned[r, c], "assignment collision"
                assigned[r, c] = True
        for yy in range(size):
            for xx in range(size):
                # Compute the projected center and rotated offset directly.
                tr = 2*(s["top"]+mid)-(h["top"]+mid)+(mid-yy)
                tc = 2*(s["left"]+mid)-(h["left"]+mid)+(mid-xx)
                assert a[tr, tc] == bg, "destination was occupied"
                assert b[tr, tc] == a[s["top"]+yy, s["left"]+xx], "incorrect projected half-turn"
                assert not assigned[tr, tc], "assignment collision"
                assigned[tr, tc] = True
    expected_changes = len(info["pairs"])*((size-2)**2+size**2)
    assert int(assigned.sum()) == expected_changes
    assert int(np.count_nonzero(a != b)) == expected_changes, "incorrect change count"
    assert np.array_equal(a[~assigned], b[~assigned]), "original or unused cells changed"
    assert np.array_equal(np.unique(a), np.unique(b)), "palette changed"
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys"
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid example counts"
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    differences = [len(_queues(info["objects"])[1])-len(_queues(info["objects"])[0])
                   for info in infos["train"]]
    assert min(differences) < 0 < max(differences), "missing opposite surplus-tail demonstrations"
    assert any(int(h["tile"][0, 0]) in set(map(int, s["tile"].flat))
               for info in infos["train"] for h in _queues(info["objects"])[0]
               for s in _queues(info["objects"])[1]), "shape/color-role witness absent"
    repeated_sample = False
    axial_asymmetry = False
    for info in infos["train"]:
        for h, s in info["pairs"]:
            y, x = _inward(h, s, info["size"])
            values, counts = np.unique(s["tile"], return_counts=True)
            carrier = values[np.argmax(counts)]
            sample = s["tile"][y, x]
            repeated_sample |= sample != carrier and np.count_nonzero(s["tile"] == sample) > 1
            dy, dx = s["top"]-h["top"], s["left"]-h["left"]
            if dy == 0 or dx == 0:
                radial = s["tile"][:, ::-1] if dy == 0 else s["tile"][::-1, :]
                axial_asymmetry |= not np.array_equal(radial, s["tile"][::-1, ::-1])
    assert repeated_sample, "spatial sampling of repeated detail colors is not taught"
    assert axial_asymmetry, "half-turn versus radial reflection is not taught"
    mandatory = set(PAIRING_ALTERNATIVES) | {"sample_center", "sample_outward", "sample_rare",
                                          "sample_majority", "D4_0", "radial_reflection", "no_projection"}
    for name, configuration in _alternatives():
        survives = all(_agrees(info, pair["output"], configuration)
                       for info, pair in zip(infos["train"], puzzle["train"]))
        if name in mandatory:
            assert not survives, "missing training counterexample: "+name
        if survives:
            assert all(_agrees(info, pair["output"], configuration)
                       for info, pair in zip(infos["test"], puzzle["test"])), \
                "query requires an undemonstrated distinction: "+name
    # Verify a visible structural train/test shift, without a hidden profile tag.
    train_sizes = {info["size"] for info in infos["train"]}
    max_imbalance = max(abs(d) for d in differences)
    max_radius = max(max(abs(s["top"]-h["top"]), abs(s["left"]-h["left"]))
                     for info in infos["train"] for h, s in info["pairs"])
    assert any(info["size"] not in train_sizes or
               abs(len(_queues(info["objects"])[1])-len(_queues(info["objects"])[0])) > max_imbalance or
               max(max(abs(s["top"]-h["top"]), abs(s["left"]-h["left"]))
                   for h, s in info["pairs"]) > max_radius
               for info in infos["test"]), "missing visible structural train/test shift"


def measure_pair(pair):
    """Observable family measurements; no sampler metadata is required."""
    info = _analyze(pair["input"])
    size = info["size"]
    hollows, solids = _queues(info["objects"])
    rays = [(s["top"]-h["top"], s["left"]-h["left"]) for h, s in info["pairs"]]
    directions = [(int(np.sign(dy)), int(np.sign(dx))) for dy, dx in rays]
    background = info["background"]
    roles, normalized = {background: 0}, []
    for value in info["grid"].flat:
        value = int(value)
        roles.setdefault(value, len(roles))
        normalized.append(roles[value])
    copies = [_destination(h, s) for h, s in info["pairs"]]
    repeated = sum(np.count_nonzero(s["tile"] == s["tile"][_inward(h, s, size)]) > 1
                   for h, s in info["pairs"])
    return dict(shape=tuple(info["grid"].shape), size=size, background=background,
                counts=(len(hollows), len(solids)), matched=len(rays), rays=rays,
                directions=directions, repeated_samples=int(repeated),
                copy_edge=any(r == 0 or c == 0 or r+size == info["grid"].shape[0] or
                              c+size == info["grid"].shape[1] for r, c in copies),
                oblique=any(dy and dx and abs(dy) != abs(dx) for dy, dx in rays),
                layout=(size, tuple((o["kind"], o["top"], o["left"]) for o in info["objects"])),
                structure=tuple(normalized),
                rejected_shortcuts=[name for name, config in _alternatives()
                                    if not _agrees(info, pair["output"], config)])


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _expect_rejection(puzzle):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError("validator accepted damaged evidence or labels")


def _coordinate_fixture():
    a = np.zeros((19, 23), dtype=np.int8)
    for r, c, color in ((2, 2, 2), (9, 11, 4), (15, 2, 2)):
        a[r:r+3, c:c+3] = color
        a[r+1, c+1] = 0
    a[2:5, 9:12] = [[3, 3, 5], [6, 3, 3], [3, 7, 3]]
    a[10:13, 15:18] = [[8, 1, 1], [1, 1, 9], [1, 1, 1]]
    b = a.copy()
    b[3, 3], b[10, 12] = 6, 8
    b[2:5, 16:19] = [[3, 7, 3], [3, 3, 6], [5, 3, 3]]
    b[11:14, 19:22] = [[1, 1, 1], [9, 1, 1], [1, 1, 8]]
    return {"input": a, "output": b}


def self_test():
    """Bounded authored tests and structural measurements over exactly 42 seeds."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    features = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                features.append(measure_pair(pair))
    assert {f["size"] for f in features} == set(SIZES)
    assert {f["background"] for f in features} == set(range(10))
    assert len({f["shape"] for f in features}) >= 35
    assert len({f["layout"] for f in features}) >= 150
    assert len({f["structure"] for f in features}) >= 150
    assert len({f["counts"] for f in features}) >= 18
    assert len({ray for f in features for ray in f["rays"]}) >= 90
    assert {d for f in features for d in f["directions"]} == {
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx}
    assert sum(f["oblique"] for f in features) >= 40
    assert sum(f["copy_edge"] for f in features) >= 3
    assert any(f["counts"][0] == f["counts"][1] for f in features)
    assert any(f["counts"][0]-f["counts"][1] >= 2 for f in features)
    assert any(f["counts"][1]-f["counts"][0] >= 2 for f in features)
    assert any(min(f["counts"]) == 1 for f in features)
    assert max(f["matched"] for f in features) >= 4
    fixture = _coordinate_fixture()
    _check_pair(fixture)
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    # Palette permutations and translations preserve all visible conventions.
    permutation = np.array([6, 0, 9, 2, 8, 1, 4, 7, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[fixture["input"]]), permutation[fixture["output"]])
    a, b = np.zeros((22, 27), dtype=int), np.zeros((22, 27), dtype=int)
    a[2:21, 3:26], b[2:21, 3:26] = fixture["input"], fixture["output"]
    assert np.array_equal(solve(a), b)
    # Every legal explicit count combination, including the two-demo minimum.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            puzzle = generate_puzzle(1000+n*10+m, n, m)
            assert len(puzzle["train"]) == n and len(puzzle["test"]) == m
    assert _signature(generate_puzzle(12345, 2, 1)) == _signature(generate_puzzle(12345, 2, 1))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, None), (None, 1), (4, 1.0)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    for where in ("aperture", "projection", "original", "unused"):
        damaged = {split: [{k: np.array(v, copy=True) for k, v in pair.items()}
                           for pair in puzzles[0][split]] for split in ("train", "test")}
        pair = damaged["test"][0]
        info = _analyze(pair["input"])
        h, s = info["pairs"][0]
        point = {"aperture": (h["top"]+1, h["left"]+1),
                 "projection": _destination(h, s),
                 "original": (s["top"], s["left"]), "unused": (0, 0)}[where]
        pair["output"][point] = (int(pair["output"][point])+1) % 10
        _expect_rejection(damaged)
    # A correct label with insufficient teaching evidence is also rejected.
    bad = {"train": [fixture, {k: permutation[v] for k, v in fixture.items()}],
           "test": puzzles[0]["test"]}
    _expect_rejection(bad)
    return dict(seeds_checked=42, pairs_checked=len(features),
                distinct_shapes=len({f["shape"] for f in features}),
                distinct_layouts=len({f["layout"] for f in features}),
                normalized_structures=len({f["structure"] for f in features}),
                count_combinations=len({f["counts"] for f in features}),
                distinct_vectors=len({ray for f in features for ray in f["rays"]}),
                oblique_pairs=sum(bool(f["oblique"]) for f in features),
                edge_copy_pairs=sum(f["copy_edge"] for f in features),
                independent_solver=False, blind_prediction=False)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
