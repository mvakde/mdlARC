"""Blueprint silhouettes slide toward their matching rails.

The input carries every convention. No examples, seed, or generation metadata
are used by solve. Generation and validation use different packing calculations.
All diagnostics are in memory; this module writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 706594240, 797007194, 953402178, 439797827, 859581606,
              241978937, 776238541, 907451037, 666744944, 637056101,
              614456174, 22629730, 81977587, 173336637, 311937143,
              34563190, 537310837, 42151959, 398926178, 682218750,
              912277223, 387483012, 275832773, 791243897, 199054367,
              943860402, 843303804, 215250782, 283324547, 780496962,
              805038518, 46027077, 107990856, 648808456, 111272053,
              379253199, 515076558, 71114674, 972497620, 465391196,
              319567018)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(rng, a, b):
    if a is not None:
        a = _integer(a, "num_train")
        if not 2 <= a <= 10:
            raise ValueError("num_train must be in 2..10")
    if b is not None:
        b = _integer(b, "num_test")
        if not 1 <= b <= 4:
            raise ValueError("num_test must be in 1..4")
    if a is None and b is None:
        legal = [(x, y) for x in range(2, 11) for y in range(1, min(4, x-1)+1)]
        return legal[int(rng.integers(len(legal)))]
    if a is None:
        a = int(rng.integers(b+1, 11))
    if b is None:
        b = int(rng.integers(1, min(4, a-1)+1))
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _key(grid):
    g = np.asarray(grid)
    return g.shape, g.astype(np.uint8).tobytes()


def _array(grid):
    g = np.asarray(grid)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "integer colors required"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors required"
    return g.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _square(part):
    rs, cs = zip(*part)
    y, x = min(rs), min(cs)
    u = max(rs)-y+1
    assert max(cs)-x+1 == u and len(part) == u*u, "core must be a solid square"
    return y, x, u


def _collar(board, y, x, u, frame, core):
    assert u <= y <= board.shape[0]-2*u and u <= x <= board.shape[1]-2*u
    expected = np.full((3*u, 3*u), frame, dtype=np.int8)
    expected[u:2*u, u:2*u] = core
    assert np.array_equal(board[y-u:y+2*u, x-u:x+2*u], expected), "invalid seed collar"


def _scale(source, core_position, source_unit, destination_unit, bg, reflected):
    """Exact rational scaling, including negative anchor-relative offsets.

    Refine source pixels, then collapse only monochromatic cells. Thus a
    reduction cannot quietly round a boundary or lose a one-pixel feature.
    """
    ay, ax = core_position
    h, w = source.shape
    top = (-ay)//source_unit*source_unit
    left = (-ax)//source_unit*source_unit
    bottom = (h-ay+source_unit-1)//source_unit*source_unit
    right = (w-ax+source_unit-1)//source_unit*source_unit
    raw = np.full((bottom-top, right-left), bg, dtype=np.int8)
    raw[-ay-top:h-ay-top, -ax-left:w-ax-left] = source
    fine = np.repeat(np.repeat(raw, destination_unit, 0), destination_unit, 1)
    nh, nw = fine.shape[0]//source_unit, fine.shape[1]//source_unit
    blocks = fine.reshape(nh, source_unit, nw, source_unit)
    small = blocks.min(axis=(1, 3))
    assert np.array_equal(small, blocks.max(axis=(1, 3))), "nonintegral color boundary"
    offset = [-top*destination_unit//source_unit, -left*destination_unit//source_unit]
    if reflected:
        small = small[:, ::-1]
        offset[1] = nw-offset[1]-destination_unit
    coords = np.argwhere(small != bg)
    lo, hi = coords.min(0), coords.max(0)+1
    return small[lo[0]:hi[0], lo[1]:hi[1]].copy(), (offset[0]-int(lo[0]), offset[1]-int(lo[1]))


def _boxes_overlap(a, b):
    return (max(a["y"], b["y"]) < min(a["y"]+a["h"], b["y"]+b["h"])
            and max(a["x"], b["x"]) < min(a["x"]+a["w"], b["x"]+b["w"]))


def _order(objects, direction):
    return sorted((i for i, o in enumerate(objects) if o["d"] == direction),
                  key=lambda i: (objects[i]["x"] if direction == 0 else
                                 -objects[i]["x"]-objects[i]["w"],
                                 objects[i]["y"], i))


def _unit_pack(objects, shape, *, boxes=False, jump=False):
    """Input-only sliding: a blocked intermediate position cannot be crossed."""
    result = [dict(o) for o in objects]
    for d in (0, 1):
        occupied = np.zeros(shape, dtype=bool)
        for i in _order(objects, d):
            o = result[i]
            y, x, h, w = (o[k] for k in ("y", "x", "h", "w"))
            mask = np.ones((h, w), dtype=bool) if boxes else o["mask"]
            assert not np.any(occupied[y:y+h, x:x+w] & mask), "blocked starting footprint"
            if jump:
                candidates = range(shape[1]-w+1)
                if d:
                    candidates = reversed(candidates)
                for nx in candidates:
                    if not np.any(occupied[y:y+h, nx:nx+w] & mask):
                        x = nx
                        break
            else:
                step = -1 if d == 0 else 1
                while 0 <= x+step <= shape[1]-w:
                    nx = x+step
                    if np.any(occupied[y:y+h, nx:nx+w] & mask):
                        break
                    x = nx
            o["x"] = x
            occupied[y:y+h, x:x+w] |= mask
    # Opposite colors are transparent during motion, but must finish disjoint.
    _check_footprints(result, shape)
    return result


def _check_footprints(objects, shape):
    occupied = np.zeros(shape, dtype=bool)
    for o in objects:
        y, x, h, w = (o[k] for k in ("y", "x", "h", "w"))
        assert 0 <= y <= shape[0]-h and 0 <= x <= shape[1]-w, "footprint outside arena"
        assert not np.any(occupied[y:y+h, x:x+w] & o["mask"]), "overlapping final silhouettes"
        occupied[y:y+h, x:x+w] |= o["mask"]


def _forbidden_pack(objects, shape):
    """Construction/check calculation using forbidden translation coordinates.

    Every pair of occupied cells on a shared row forbids one translation.
    The closest such coordinate on the journey is the barrier. This checks
    gaps and interlocking without using the solver's one-cell motion loop.
    """
    result = [dict(o) for o in objects]
    for d in (0, 1):
        settled_rows = {}
        for i in _order(objects, d):
            o = result[i]
            pixels = np.argwhere(o["mask"]).tolist()
            forbidden = {px-c for r, c in pixels for px in settled_rows.get(o["y"]+r, ())}
            x = o["x"]
            assert x not in forbidden
            if d == 0:
                barrier = max((p for p in forbidden if 0 <= p < x), default=-1)
                o["x"] = barrier+1
            else:
                end = shape[1]-o["w"]
                barrier = min((p for p in forbidden if x < p <= end), default=end+1)
                o["x"] = barrier-1
            for r, c in pixels:
                settled_rows.setdefault(o["y"]+r, set()).add(o["x"]+c)
    _check_footprints(result, shape)
    return result


def _decode(input_grid):
    grid = _array(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg]*2 > grid.size, "input background must be a strict majority"
    candidates = []
    for k in range(4):
        g = np.rot90(grid, k)
        for row in range(3, g.shape[0]-3):
            divider = int(g[row, 0])
            if divider == bg or not np.all(g[row] == divider):
                continue
            low, high = int(g[row+1, 0]), int(g[row+1, -1])
            if len({bg, divider, low, high}) != 4:
                continue
            if (np.all(g[row+1:, 0] == low) and np.all(g[row+1:, -1] == high)
                    and np.all(g[:row, 0] == bg) and np.all(g[:row, -1] == bg)
                    and np.all(g[0] == bg) and np.all(g[row-1] == bg)):
                candidates.append((k, g.copy(), row, divider, (low, high)))
    assert len(candidates) == 1, "expected one recipe separator with two colored arena rails"
    rotation, canonical, row, divider, frames = candidates[0]
    board = canonical[row+1:, 1:-1]
    header = canonical[:row]
    neutral = set(map(int, np.unique(board)))-{bg, *frames}
    assert len(neutral) == 1, "seed cores identify one payload color"
    core = next(iter(neutral))
    assert core != divider
    source_cores = _components(header == core)
    assert len(source_cores) == 1
    ay, ax, source_unit = _square(source_cores[0])
    source_frame = int(header[ay-source_unit, ax-source_unit])
    assert source_frame in frames
    _collar(header, ay, ax, source_unit, source_frame, core)
    assert len(_components(header != bg)) == 1, "one connected reference motif required"
    coords = np.argwhere(header != bg)
    lo, hi = coords.min(0), coords.max(0)+1
    source = header[lo[0]:hi[0], lo[1]:hi[1]].copy()
    source_anchor = ay-int(lo[0]), ax-int(lo[1])
    details = set(map(int, np.unique(source)))-{bg, source_frame, core}
    assert details and not details & {divider, *frames}, "detail colors must have distinct roles"
    # The collar is the only directional-color region in the reference.
    expected_frame = np.zeros(header.shape, dtype=bool)
    expected_frame[ay-source_unit:ay+2*source_unit, ax-source_unit:ax+2*source_unit] = True
    expected_frame[ay:ay+source_unit, ax:ax+source_unit] = False
    assert np.array_equal(header == source_frame, expected_frame)
    objects, covered = [], board == bg
    for part in _components(board == core):
        y, x, u = _square(part)
        assert y >= u and x >= u
        frame = int(board[y-u, x-u])
        assert frame in frames
        _collar(board, y, x, u, frame, core)
        assert not np.any(covered[y-u:y+2*u, x-u:x+2*u])
        covered[y-u:y+2*u, x-u:x+2*u] = True
        patch, offset = _scale(source, source_anchor, source_unit, u, bg,
                               frame != source_frame)
        patch[patch == source_frame] = frame
        h, w = patch.shape
        objects.append(dict(y=y-offset[0], x=x-offset[1], h=h, w=w,
                            d=frames.index(frame), u=u, core_y=y, core_x=x,
                            offset=offset, frame=frame, patch=patch, mask=patch != bg))
    assert objects and np.all(covered), "arena contains only complete square seeds"
    _check_footprints(objects, board.shape)
    assert not any(_boxes_overlap(a, b) for i, a in enumerate(objects) for b in objects[i+1:]), \
        "initial reconstructed bounding boxes must be disjoint"
    return dict(canonical=canonical, rotation=rotation, row=row, bg=bg,
                divider=divider, frames=frames, core=core, source=source,
                source_unit=source_unit, source_frame=source_frame,
                source_anchor=source_anchor, objects=objects, shape=board.shape)


def _paint(data, settled):
    out = data["canonical"].copy()
    arena = out[data["row"]+1:, 1:-1]
    arena[:] = data["bg"]
    for o in settled:
        y, x, h, w = (o[k] for k in ("y", "x", "h", "w"))
        target = arena[y:y+h, x:x+w]
        target[o["mask"]] = o["patch"][o["mask"]]
    return np.rot90(out, -data["rotation"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _decode(input_grid)
    return _paint(data, _unit_pack(data["objects"], data["shape"]))


def _palette(rng):
    colors = [int(c) for c in rng.permutation(10)]
    return dict(bg=colors[0], divider=colors[1], core=colors[2],
                frames=tuple(colors[3:5]), details=colors[5:5+int(rng.integers(1, 6))])


def _model(rng, palette, source_direction, simple=False):
    """Grow an asymmetric connected motif outside a 3x3 frame-and-core collar."""
    bg, core = palette["bg"], palette["core"]
    frame, details = palette["frames"][source_direction], palette["details"]
    cells = {(r, c): frame for r in range(3) for c in range(3)}
    cells[1, 1] = core
    style = 0 if simple else int(rng.integers(5))
    if style == 0:
        row = int(rng.choice([0, 2]))
        length = 1 if simple else int(rng.integers(1, 4))
        for i in range(length):
            cells[row, 3+i] = details[i % len(details)]
    elif style == 1:
        col = int(rng.integers(3))
        length = int(rng.integers(2, 5))
        for i in range(1, length+1):
            cells[-i, col] = details[0]
        for level in range(1, length+1, 2):
            direction = int(rng.choice([-1, 1]))
            for j in range(1, int(rng.integers(2, 4))):
                cells[-level, col+direction*j] = details[level % len(details)]
    else:
        steps = int(rng.integers(2, 13))
        for i in range(steps):
            frontier = set()
            for r, c in cells:
                for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if p not in cells and -3 <= p[0] <= 5 and -3 <= p[1] <= 5:
                        frontier.add(p)
            possible = []
            for p in sorted(frontier):
                rs, cs = zip(*(set(cells) | {p}))
                if max(rs)-min(rs) < 7 and max(cs)-min(cs) < 7:
                    possible.append(p)
            if not possible:
                break
            point = possible[int(rng.integers(len(possible)))]
            cells[point] = details[i % len(details)]
    rs, cs = zip(*cells)
    arr = np.full((max(rs)-min(rs)+1, max(cs)-min(cs)+1), bg, dtype=np.int8)
    for (r, c), color in cells.items():
        arr[r-min(rs), c-min(cs)] = color
    arr = np.rot90(arr, int(rng.integers(4))).copy()
    if rng.random() < .5:
        arr = arr[:, ::-1].copy()
    anchor = tuple(map(int, np.argwhere(arr == core)[0]))
    return arr, anchor


def _object(model, anchor, palette, ref_d, u, d, y, x):
    arr = model if d == ref_d else model[:, ::-1]
    offset = (anchor[0]*u, (anchor[1] if d == ref_d else model.shape[1]-anchor[1]-1)*u)
    patch = np.repeat(np.repeat(arr, u, 0), u, 1).copy()
    patch[patch == palette["frames"][ref_d]] = palette["frames"][d]
    h, w = patch.shape
    return dict(y=y, x=x, h=h, w=w, d=d, u=u, offset=offset,
                frame=palette["frames"][d], patch=patch, mask=patch != palette["bg"])


def _positions(objects):
    return tuple(o["x"] for o in objects)


def _witness_layout(rng, model, anchor, p, ref_d, purpose):
    h, w = model.shape
    if max(h, w) > 6:
        return None
    d = int(rng.integers(2))
    if purpose == "interlock":
        if 5*w+3 > 28:
            return None
        choices = []
        for dy in range(1, h):
            for reverse in (False, True):
                ys = [dy, 0] if reverse else [0, dy]
                objs = [_object(model, anchor, p, ref_d, 1, d, ys[i], i*w) for i in range(2)]
                if d:
                    for o in objs:
                        o["x"] = 2*w-o["x"]-w
                shape = (h+dy, 2*w)
                packed = _forbidden_pack(objs, shape)
                boxed = _unit_pack(objs, shape, boxes=True)
                if _positions(packed) != _positions(boxed):
                    choices.append(ys)
        if not choices:
            return None
        ys = choices[int(rng.integers(len(choices)))]
        source_unit = 1
        board_h = int(rng.integers(max(h+max(ys), 2*h), 28-h))
        board_w = int(rng.integers(5*w+3, 29))
        directions = [d, d, 1-d, 1-d]
        units = [1, 1, 2, 1]
        # Randomize which class starts nearer each rail, preserving the chosen
        # destination-side order of the two interacting same-class objects.
        other_group = [2, 3] if rng.random() < .5 else [3, 2]
        block_order = [0, 1]+other_group if rng.random() < .5 else other_group+[0, 1]
        if d:
            block_order = block_order[::-1]
        gaps = rng.multinomial(board_w-5*w, np.full(5, .2))
        starts, cursor = {}, int(gaps[0])
        for j, i in enumerate(block_order):
            starts[i] = cursor
            cursor += units[i]*w + int(gaps[j+1])
        shift = int(rng.integers(board_h-h-max(ys)+1))
        rows = [ys[0]+shift, ys[1]+shift, int(rng.integers(board_h-2*h+1)),
                int(rng.integers(board_h-h+1))]
        objects = [_object(model, anchor, p, ref_d, units[i], directions[i], rows[i], starts[i])
                   for i in range(4)]
    else:
        choices = []
        for dy in range((h+1)//2, h):
            for reverse in (False, True):
                ys = [2*dy, dy, 0] if reverse else [0, dy, 2*dy]
                objs = [_object(model, anchor, p, ref_d, 1, d, ys[i], i*w) for i in range(3)]
                if d:
                    for o in objs:
                        o["x"] = 3*w-o["x"]-w
                shape = (h+2*dy, 3*w)
                if _positions(_forbidden_pack(objs, shape)) != _positions(_unit_pack(objs, shape, jump=True)):
                    choices.append(ys)
        if not choices:
            return None
        ys = choices[int(rng.integers(len(choices)))]
        source_unit = 2 if 2*h+2+1+h+max(ys) <= 30 else 1
        max_h = 27-source_unit*h
        board_h = int(rng.integers(h+max(ys), max_h+1))
        board_w = int(rng.integers(3*w+2, 29))
        free = board_w-3*w
        gaps = rng.multinomial(free, np.full(4, .25))
        cursor = int(gaps[0])
        shift = int(rng.integers(board_h-h-max(ys)+1))
        objects = []
        for i in range(3):
            x = cursor if d == 0 else board_w-cursor-w
            objects.append(_object(model, anchor, p, ref_d, 1, d, ys[i]+shift, x))
            cursor += w+int(gaps[i+1])
    _check_footprints(_forbidden_pack(objects, (board_h, board_w)), (board_h, board_w))
    return source_unit, (board_h, board_w), objects


def _free_layout(rng, model, anchor, p, ref_d, test):
    h, w = model.shape
    if h > 6 or w > 7:
        return None
    source_unit = 1 if test else int(rng.integers(1, 3))
    first_unit = 3 if test else int(rng.integers(1, 3))
    max_h = 27-source_unit*h
    minimum_w = (4*w+2 if test else max(first_unit*w+2, source_unit*w))
    if first_unit*h > max_h or minimum_w > 28:
        return None
    board_h = int(rng.integers(first_unit*h, max_h+1))
    board_w = int(rng.integers(minimum_w, 29))
    requested = int(rng.integers(1 if not test else 2, 8))
    bias = float(rng.choice([0, .15, .5, .5, .85, 1]))
    objects = []
    if test:
        # A large and small copy must interact in each held-out example. A
        # solitary enlargement would test scaling without the new coupling.
        d = int(rng.integers(2))
        gaps = rng.multinomial(board_w-4*w, np.full(3, 1/3))
        if rng.random() < .5:
            starts = [int(gaps[0]), int(gaps[0]+3*w+gaps[1])]
        else:
            starts = [int(gaps[0]+w+gaps[1]), int(gaps[0])]
        for u, x in zip((3, 1), starts):
            y = int(rng.integers(board_h-u*h+1))
            objects.append(_object(model, anchor, p, ref_d, u, d, y, x))
    for index in range(len(objects), requested):
        for attempt in range(56):
            u = first_unit if index == 0 else int(rng.integers(1, 3))
            if u*h > board_h or u*w > board_w:
                continue
            y = int(rng.integers(board_h-u*h+1))
            x = int(rng.integers(board_w-u*w+1))
            if index == 0:
                x = int(rng.integers(1, board_w-u*w))
            d = int(rng.random() >= bias)
            o = _object(model, anchor, p, ref_d, u, d, y, x)
            if any(_boxes_overlap(o, prev) for prev in objects):
                continue
            try:
                _forbidden_pack(objects+[o], (board_h, board_w))
            except AssertionError:
                continue
            objects.append(o)
            break
        else:
            if not objects:
                return None
            break
    return source_unit, (board_h, board_w), objects


def _assemble(rng, p, model, anchor, ref_d, layout, rotation):
    source_unit, shape, objects = layout
    source = np.repeat(np.repeat(model, source_unit, 0), source_unit, 1)
    header_h = source.shape[0]+2
    height, width = header_h+1+shape[0], shape[1]+2
    if height > 30 or width > 30 or source.shape[1]+2 > width:
        return None
    inp = np.full((height, width), p["bg"], dtype=np.int8)
    left = int(rng.integers(1, width-source.shape[1]))
    inp[1:header_h-1, left:left+source.shape[1]] = source
    inp[header_h] = p["divider"]
    inp[header_h+1:, 0], inp[header_h+1:, -1] = p["frames"]
    arena = inp[header_h+1:, 1:-1]
    for o in objects:
        u = o["u"]
        y, x = o["y"]+o["offset"][0], o["x"]+o["offset"][1]
        arena[y-u:y+2*u, x-u:x+2*u] = o["frame"]
        arena[y:y+u, x:x+u] = p["core"]
    if np.count_nonzero(inp == p["bg"])*2 <= inp.size:
        return None
    settled = _forbidden_pack(objects, shape)
    data = dict(canonical=inp, row=header_h, bg=p["bg"], rotation=0)
    out = _paint(data, settled)
    if np.array_equal(_positions(objects), _positions(settled)):
        return None
    return {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _features(data, settled):
    objects, shape = data["objects"], data["shape"]
    try:
        boxed = _unit_pack(objects, shape, boxes=True)
        interlock = _positions(boxed) != _positions(settled)
    except AssertionError:
        interlock = True
    try:
        jumped = _unit_pack(objects, shape, jump=True)
        chain = _positions(jumped) != _positions(settled)
    except AssertionError:
        chain = True
    core_objects = []
    for o in objects:
        u = o["u"]
        core_objects.append(dict(o, y=o["core_y"]-u, x=o["core_x"]-u,
                                 h=3*u, w=3*u, mask=np.ones((3*u, 3*u), dtype=bool)))
    core_settled = _unit_pack(core_objects, shape)
    core_destinations = tuple(p["x"]+o["u"]-o["offset"][1] for o, p in zip(objects, core_settled))
    sy, sx = data["source_anchor"]
    horizontal_asymmetry = (2*sx+data["source_unit"] != data["source"].shape[1]
                            or not np.array_equal(data["source"], data["source"][:, ::-1]))
    large_small_contact = False
    for a in settled:
        if a["u"] != 3:
            continue
        a_pixels = {(a["y"]+int(r), a["x"]+int(c)) for r, c in np.argwhere(a["mask"])}
        for b in settled:
            if b["u"] == 3 or a["d"] != b["d"]:
                continue
            b_pixels = {(b["y"]+int(r), b["x"]+int(c)) for r, c in np.argwhere(b["mask"])}
            if any((r, c-1) in b_pixels or (r, c+1) in b_pixels for r, c in a_pixels):
                large_small_contact = True
    return dict(count=len(objects), units=sorted({o["u"] for o in objects}),
                same_scale_both_directions=bool({o["u"] for o in objects if o["d"] == 0}
                                                & {o["u"] for o in objects if o["d"] == 1}),
                source_unit=data["source_unit"], rotation=data["rotation"],
                horizontal_asymmetry=bool(horizontal_asymmetry),
                large_small_contact=bool(large_small_contact),
                shape=list(data["canonical"].shape), frames=list(data["frames"]),
                motif_shape=list(data["source"].shape), interlock=bool(interlock),
                chain=bool(chain), core_only_differs=core_destinations != _positions(settled),
                movement=any(a["x"] != b["x"] for a, b in zip(objects, settled)),
                moving_directions=sorted({a["d"] for a, b in zip(objects, settled) if a["x"] != b["x"]}),
                directions=sorted({o["d"] for o in objects}),
                stationary=sum(a["x"] == b["x"] for a, b in zip(objects, settled)),
                blocked=sum(o["x"] != (0 if o["d"] == 0 else shape[1]-o["w"])
                            for o in settled),
                bbox_intersections=sum(_boxes_overlap(a, b) for i, a in enumerate(settled)
                                       for b in settled[i+1:]))


def _make_pair(rng, purpose, rotation):
    for attempt in range(96):
        p = _palette(rng)
        ref_d = int(rng.integers(2))
        # A finite simple fallback still exposes both intended distinctions.
        model, anchor = _model(rng, p, ref_d, simple=attempt >= 72)
        try:
            layout = (_witness_layout(rng, model, anchor, p, ref_d, purpose)
                      if purpose in ("interlock", "chain") else
                      _free_layout(rng, model, anchor, p, ref_d, purpose == "test"))
            if layout is None:
                continue
            pair = _assemble(rng, p, model, anchor, ref_d, layout, rotation)
            if pair is None:
                continue
            data = _decode(pair["input"])
            packed = _unit_pack(data["objects"], data["shape"])
            f = _features(data, packed)
            horizontal_asymmetry = (anchor[1] != model.shape[1]-anchor[1]-1
                                    or not np.array_equal(model, model[:, ::-1]))
            if purpose == "interlock" and not (f["interlock"] and f["core_only_differs"]
                                               and f["bbox_intersections"] and horizontal_asymmetry
                                               and f["moving_directions"] == [0, 1]):
                continue
            if purpose == "chain" and not (f["chain"] and f["source_unit"] == 2):
                continue
            if purpose == "test" and not (f["core_only_differs"] and f["blocked"]
                                          and f["large_small_contact"]):
                continue
            assert np.array_equal(_paint(data, packed), pair["output"])
            return pair
        except AssertionError:
            continue
    raise RuntimeError("bounded construction failed for " + purpose)


def _fresh(pair, seen):
    if _key(pair["input"]) not in seen:
        return pair
    # Ninety injective role assignments versus at most thirteen used inputs.
    vals = list(map(int, np.unique(pair["input"])))
    first, second = vals[:2]
    old = [first, second]+[c for c in range(10) if c not in (first, second)]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            new = [a, b]+[c for c in range(10) if c not in (a, b)]
            mapping = np.zeros(10, dtype=np.int8)
            mapping[old] = new
            candidate = {k: mapping[g] for k, g in pair.items()}
            if _key(candidate["input"]) not in seen:
                return candidate
    raise AssertionError("finite color fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    a, b = _counts(rng, num_train, num_test)
    rotation = int(rng.integers(4))
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            purpose = ("test" if split == "test" else "interlock" if i == 0
                       else "chain" if i == 1 else "free")
            pair = _fresh(_make_pair(rng, purpose, rotation), seen)
            seen.add(_key(pair["input"]))
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _array(pair["input"]), _array(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            data = _decode(inp)
            objects, shape = data["objects"], data["shape"]
            simulated = _unit_pack(objects, shape)
            checked = _forbidden_pack(objects, shape)
            assert _positions(simulated) == _positions(checked), "packing calculations disagree"
            assert np.array_equal(out, _paint(data, checked)), "incorrect completion/packing"
            # Every translated pattern, including its holes, is preserved.
            for before, after in zip(objects, checked):
                assert before["y"] == after["y"] and np.array_equal(before["patch"], after["patch"])
                assert after["x"] <= before["x"] if before["d"] == 0 else after["x"] >= before["x"]
            f = _features(data, checked)
            assert f["movement"], "each generated example must expose compaction"
            assert not np.array_equal(out, inp)
            features[split].append(f)
    train = features["train"]
    assert any(f["interlock"] and f["core_only_differs"] and f["bbox_intersections"]
               and f["horizontal_asymmetry"]
               for f in train), "missing silhouette witness"
    assert any(f["chain"] for f in train), "missing no-jumping witness"
    assert any(f["source_unit"] == 1 and 2 in f["units"] and f["moving_directions"] == [0, 1]
               and f["same_scale_both_directions"]
               for f in train), "training must expose enlargement and both rails"
    assert any(f["source_unit"] == 2 and 1 in f["units"] for f in train), "missing reduction witness"
    assert len({f["rotation"] for f in train+features["test"]}) == 1
    # The generator's documented curriculum, not a condition required by solve.
    assert all(max(f["units"]) <= 2 for f in train)
    assert all(3 in f["units"] and f["source_unit"] == 1 for f in features["test"])
    assert all(f["core_only_differs"] and f["blocked"] and f["large_small_contact"]
               for f in features["test"])


def self_test():
    """Bounded 42-seed corpus, mutation checks, and all explicit count pairs."""
    fingerprints, shapes, roles, sources, counts, orientations = set(), set(), set(), set(), set(), set()
    total, interlocks, chains, solo, mixed, bbox_hits = 0, 0, 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                settled = _unit_pack(data["objects"], data["shape"])
                f = _features(data, settled)
                total += 1
                interlocks += f["interlock"]
                chains += f["chain"]
                solo += f["count"] == 1
                mixed += len(f["units"]) > 1
                bbox_hits += f["bbox_intersections"] > 0
                shapes.add(pair["input"].shape)
                roles.add((data["bg"], data["divider"], data["core"], *data["frames"]))
                sources.add(_key(data["source"]))
                counts.add(f["count"])
                orientations.add(f["rotation"])
                for k in ("input", "output"):
                    digest.update(bytes(pair[k].shape))
                    digest.update(pair[k].tobytes())
        fingerprints.add(digest.digest())
        if seed == 0:
            repeated = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], repeated[s]) for k in a)
    assert len(fingerprints) == 42 and orientations == {0, 1, 2, 3}
    assert len(shapes) >= 50 and len(roles) >= 150 and len(sources) >= 180
    assert {1, 2, 3, 4} <= counts and solo and mixed >= 42
    assert interlocks >= 42 and chains >= 42 and bbox_hits >= 42
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(73, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.0, 1), (4, False), (4, 1.0)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a, b in ((None, None), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(99, a, b)
        assert a is None or len(p["train"]) == a
        assert b is None or len(p["test"]) == b
    # Small explicit shape examples distinguish interlocking from rectangles,
    # and continuous sliding from the old edge-first jump convention.
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=bool)
    objs = [dict(y=y, x=x, h=3, w=5, d=0, mask=mask)
            for y, x in ((4, 1), (2, 9), (0, 17))]
    assert _positions(_unit_pack(objs, (7, 24))) == (0, 5, 10)
    assert _positions(_unit_pack(objs, (7, 24), jump=True)) == (0, 5, 0)
    assert _positions(_forbidden_pack(objs, (7, 24))) == (0, 5, 10)
    objs = [dict(y=y, x=x, h=3, w=5, d=0, mask=mask) for y, x in ((0, 1), (2, 9))]
    assert _positions(_unit_pack(objs, (5, 16))) == (0, 3)
    assert _positions(_unit_pack(objs, (5, 16), boxes=True)) == (0, 5)
    # Dropping compaction gives complete motifs at the input anchors, a
    # concrete plausible wrong rule. It must fail on the training witnesses.
    no_motion = generate_puzzle(0, 2, 1)
    for pair in no_motion["train"]:
        data = _decode(pair["input"])
        pair["output"] = _paint(data, data["objects"])
    try:
        validate_generated_puzzle(no_motion)
    except AssertionError:
        pass
    else:
        raise AssertionError("completion without compaction accepted")
    # An actual wrong label must be rejected, rather than merely exercising
    # agreement between routines on their own constructed data.
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=total, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                role_assignments=len(roles), source_rasters=len(sources), counts=sorted(counts),
                orientations=sorted(orientations), interlock_pairs=interlocks,
                no_jumping_pairs=chains, single_object_pairs=solo, mixed_scale_pairs=mixed,
                interpenetrating_bounding_box_pairs=bbox_hits)
