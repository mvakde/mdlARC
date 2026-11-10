"""Jigsaw cargo queues: exact-profile, stable, frame-directed compaction.

All semantic information is visible in the input. No parent code is imported.
The solver operates on the original masks; construction builds a gapless frontier
and scatters its layers while preserving their source-side partial order.
"""
from __future__ import annotations

import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid ARC color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    """Four-connected components of occupied cells, irrespective of payload color."""
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.asarray(sorted(cells), dtype=int))
    return result


def _dilate(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r, c in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))])


def _boundary(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    interior = mask.copy()
    for r, c in ((0, 1), (2, 1), (1, 0), (1, 2)):
        interior &= p[r:r+h, c:c+w]
    return mask & ~interior


def _decode(value):
    g = _grid(value)
    hist = np.bincount(g.ravel(), minlength=10)
    bg = int(hist.argmax())
    _require(2*hist[bg] > g.size, "background must occupy more than half the input")
    comps = _components(g != bg)
    spanning = []
    for i, cells in enumerate(comps):
        for axis in (0, 1):
            if len(np.unique(cells[:, axis])) == g.shape[axis]:
                spanning.append((i, axis))
    _require(len(spanning) == 2 and spanning[0][1] == spanning[1][1],
             "expected two parallel spanning edge rails")
    transpose = spanning[0][1] == 1
    canonical = g.T.copy() if transpose else g
    h, w = canonical.shape
    rail_indices = {i for i, _ in spanning}
    rails, rail_mask = {}, np.zeros((h, w), dtype=bool)
    objects = []
    for i, original_cells in enumerate(comps):
        cells = original_cells[:, ::-1] if transpose else original_cells
        if i in rail_indices:
            colors = np.unique(canonical[cells[:, 0], cells[:, 1]])
            _require(len(colors) == 1, "rail must be monochrome")
            color = int(colors[0])
            lo, hi = [], []
            for r in range(h):
                cols = np.sort(cells[cells[:, 0] == r, 1])
                _require(len(cols) and np.array_equal(cols, np.arange(cols[0], cols[-1]+1)),
                         "rail must be contiguous in every row")
                lo.append(int(cols[0]))
                hi.append(int(cols[-1])+1)
            side = 0 if all(x == 0 for x in lo) else 1
            _require(side == 0 or all(x == w for x in hi), "rail must adjoin a canvas edge")
            _require(side not in rails, "two rails occupy the same edge")
            rails[side] = dict(color=color, edge=np.array(hi if side == 0 else lo))
            rail_mask[cells[:, 0], cells[:, 1]] = True
        else:
            r, c = map(int, cells.min(axis=0))
            oh, ow = map(int, cells.max(axis=0)-cells.min(axis=0)+1)
            local = cells - (r, c)
            mask = np.zeros((oh, ow), dtype=bool)
            mask[local[:, 0], local[:, 1]] = True
            _require(oh >= 3 and ow >= 3, "cargo is too thin to contain a payload")
            lo, hi = [], []
            for rr in range(oh):
                cols = np.flatnonzero(mask[rr])
                _require(len(cols) and np.array_equal(cols, np.arange(cols[0], cols[-1]+1)),
                         "cargo must have one occupied interval in every row")
                lo.append(int(cols[0]))
                hi.append(int(cols[-1])+1)
            patch = canonical[r:r+oh, c:c+ow].copy()
            border = _boundary(mask)
            colors = np.unique(patch[border])
            _require(len(colors) == 1, "cargo contour must have a uniform frame")
            frame = int(colors[0])
            inside = mask & ~border
            _require(np.any(inside) and np.all(patch[inside] != frame), "missing or erased payload")
            objects.append(dict(r=r, c=c, h=oh, w=ow, mask=mask, patch=patch,
                                lo=np.array(lo), hi=np.array(hi), frame=frame))
    _require(set(rails) == {0, 1} and rails[0]["color"] != rails[1]["color"], "ambiguous rails")
    _require(np.all(rails[0]["edge"] < rails[1]["edge"]), "rails overlap")
    frame_map = {rail["color"]: side for side, rail in rails.items()}
    _require(bool(objects), "no cargo")
    occupied = rail_mask.copy()
    for o in objects:
        _require(o["frame"] in frame_map, "cargo frame has no matching rail")
        o["side"] = frame_map[o["frame"]]
        cells = np.argwhere(o["mask"]) + (o["r"], o["c"])
        _require(not np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]), "input cargo must be detached")
        occupied[cells[:, 0], cells[:, 1]] = True
    base = np.full_like(canonical, bg)
    base[rail_mask] = canonical[rail_mask]
    return dict(grid=canonical, background=bg, transpose=transpose, rails=rails,
                rail_mask=rail_mask, base=base, objects=objects)


def _shared_rows(a, b):
    start = max(a["r"], b["r"])
    end = min(a["r"]+a["h"], b["r"]+b["h"])
    return start, end


def _source_orders(scene):
    """Partial order from occupied intervals, not bounding-box origin sorting.

    An input pair sharing rows is entirely separated on one side in ALL of
    those rows. Disjoint-row pieces commute, so raster ties do not affect output.
    """
    objects = scene["objects"]
    orders, edges = {}, []
    for side in (0, 1):
        indices = [i for i, o in enumerate(objects) if o["side"] == side]
        previous = {i: set() for i in indices}
        for pos, i in enumerate(indices):
            a = objects[i]
            for j in indices[pos+1:]:
                b = objects[j]
                s, e = _shared_rows(a, b)
                if s >= e:
                    continue
                aa, bb = slice(s-a["r"], e-a["r"]), slice(s-b["r"], e-b["r"])
                ab = np.all(a["c"]+a["hi"][aa] <= b["c"]+b["lo"][bb])
                ba = np.all(b["c"]+b["hi"][bb] <= a["c"]+a["lo"][aa])
                _require(bool(ab) != bool(ba), "source cargo has ambiguous side order")
                left, right = (i, j) if ab else (j, i)
                ahead, behind = (left, right) if side == 0 else (right, left)
                previous[behind].add(ahead)
                edges.append((ahead, behind))
        order, done = [], set()
        while len(order) < len(indices):
            ready = [i for i in indices if i not in done and previous[i] <= done]
            _require(bool(ready), "cyclic source order")
            i = min(ready, key=lambda j: (objects[j]["r"], objects[j]["c"]))
            done.add(i)
            order.append(i)
        orders[side] = order
    return orders, edges


def _place(scene, positions):
    out = scene["base"].copy()
    occupied = scene["rail_mask"].copy()
    for o, c in zip(scene["objects"], positions):
        cells = np.argwhere(o["mask"]) + (o["r"], c)
        _require(np.all(cells >= 0) and np.all(cells < np.array(out.shape)), "cargo leaves canvas")
        _require(not np.any(occupied[cells[:, 0], cells[:, 1]]), "final occupied masks overlap")
        occupied[cells[:, 0], cells[:, 1]] = True
        patch = out[o["r"]:o["r"]+o["h"], c:c+o["w"]]
        patch[o["mask"]] = o["patch"][o["mask"]]
    return out


def _settle(scene):
    orders, edges = _source_orders(scene)
    positions = [None]*len(scene["objects"])
    for side in (0, 1):
        frontier = scene["rails"][side]["edge"].copy()
        for i in orders[side]:
            o = scene["objects"][i]
            rows = slice(o["r"], o["r"]+o["h"])
            offsets = frontier[rows] - (o["lo"] if side == 0 else o["hi"])
            c = int(offsets.max() if side == 0 else offsets.min())
            _require(np.all(offsets == c), "cargo does not exactly complement its queue frontier")
            _require(c <= o["c"] if side == 0 else c >= o["c"], "cargo would move against its rail")
            positions[i] = c
            frontier[rows] = c + (o["hi"] if side == 0 else o["lo"])
    return _place(scene, positions), positions, orders, edges


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read both rails, frame roles, masks, and stable source order from one input."""
    scene = _decode(input_grid)
    out = _settle(scene)[0]
    return (out.T if scene["transpose"] else out).copy()


def _simulate(scene):
    """Supplemental unit-step collision check, independent of frontier recurrence.

    Other-direction cargo is absent during each sweep. A rigid mask cannot jump
    an occupied tooth. The two settled classes are then checked together.
    """
    objects = scene["objects"]
    positions = [o["c"] for o in objects]
    h, w = scene["grid"].shape
    for side, delta in ((0, -1), (1, 1)):
        indices = [i for i, o in enumerate(objects) if o["side"] == side]
        occupied = scene["rail_mask"].copy()
        for i in indices:
            o = objects[i]
            cells = np.argwhere(o["mask"])+(o["r"], positions[i])
            occupied[cells[:, 0], cells[:, 1]] = True
        for _ in range(w*max(1, len(indices))+1):
            moved = False
            for i in indices:
                o = objects[i]
                local = np.argwhere(o["mask"])+(o["r"], 0)
                cells = local+(0, positions[i])
                occupied[cells[:, 0], cells[:, 1]] = False
                nxt = local+(0, positions[i]+delta)
                if (np.all(nxt[:, 1] >= 0) and np.all(nxt[:, 1] < w)
                        and not np.any(occupied[nxt[:, 0], nxt[:, 1]])):
                    positions[i] += delta
                    moved = True
                cells = local+(0, positions[i])
                occupied[cells[:, 0], cells[:, 1]] = True
            if not moved:
                break
        else:
            raise ValueError("unit-step motion did not terminate")
    return _place(scene, positions), positions


def _payload(rng, mask, frame, bg, color, allowed, patterned):
    patch = np.full(mask.shape, bg, dtype=np.int8)
    patch[mask] = frame
    inside = mask & ~_boundary(mask)
    _require(np.any(inside), "generated shape has no interior")
    patch[inside] = color
    if patterned and np.count_nonzero(inside) > 1:
        other = int(rng.choice([c for c in allowed if c != color]))
        rr, cc = np.indices(mask.shape)
        stripe = (rr+cc) % 2 == 0 if rng.random() < .5 else cc % 2 == 0
        patch[inside & stripe] = other
    return patch


def _add_layer(rng, objects, frontier, side, row, height, frame, bg, colors,
               *, thickness=3, shape="flat", color=None, patterned=False):
    lower = frontier[side][row:row+height].copy()
    if shape == "parallel":
        upper = lower+thickness
    else:
        upper = np.full(height, int(lower.max())+thickness)
        if shape == "teeth":
            upper += rng.integers(0, 3, size=height)
    w_canvas = frontier[2]
    if side == 0:
        lo, hi = lower, upper
    else:
        lo, hi = w_canvas-upper, w_canvas-lower
    origin = int(lo.min())
    local_lo, local_hi = lo-origin, hi-origin
    width = int(local_hi.max())
    _require(width <= 16, "cargo too wide to scatter")
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        mask[r, local_lo[r]:local_hi[r]] = True
    _require(len(_components(mask)) == 1, "generated contour is disconnected")
    neutral = [c for c in colors if c != frame]
    color = int(rng.choice(neutral)) if color is None else int(color)
    patch = _payload(rng, mask, frame, bg, color, neutral, patterned)
    objects.append(dict(r=row, c=origin, h=height, w=width, mask=mask, patch=patch,
                        lo=local_lo, hi=local_hi, side=side, frame=frame))
    frontier[side][row:row+height] = upper


def _scatter(rng, base, bg, objects, *, require_crossing=False):
    """Bounded backtracking over translations, preserving every shared-row order."""
    h, w = base.shape
    rail_mask = base != bg
    placed, nodes = [], 0

    def visit(index, occupied):
        nonlocal nodes
        nodes += 1
        if nodes > 3500:
            return None
        if index == len(objects):
            if require_crossing and not _crossing([placed[0], placed[-1]]):
                return None
            return [dict(o) for o in placed]
        o = objects[index]
        direction = 1 if o["side"] == 0 else -1
        maximum = w-o["w"]-o["c"] if direction == 1 else o["c"]
        minimum = objects[-1]["w"]+3 if require_crossing and index == 0 else 1
        choices = list(range(minimum, maximum+1))
        rng.shuffle(choices)
        # Usually preserve enough slack for the follower, while still sampling
        # far origins and opposite-class crossings on many accepted scenes.
        if rng.random() < .7:
            choices.sort(key=lambda d: d + float(rng.random())*7)
        forbidden = _dilate(occupied)
        local = np.argwhere(o["mask"])+(o["r"], 0)
        for distance in choices:
            c = o["c"]+direction*distance
            valid = True
            for p in placed:
                s, e = _shared_rows(o, p)
                if p["side"] != o["side"] or s >= e:
                    continue
                a, b = slice(s-o["r"], e-o["r"]), slice(s-p["r"], e-p["r"])
                valid &= bool(np.all(c+o["lo"][a] > p["c"]+p["hi"][b]) if direction == 1
                              else np.all(c+o["hi"][a] < p["c"]+p["lo"][b]))
            if not valid:
                continue
            cells = local+(0, c)
            if np.any(forbidden[cells[:, 0], cells[:, 1]]):
                continue
            next_occupied = occupied.copy()
            next_occupied[cells[:, 0], cells[:, 1]] = True
            placed.append(dict(o, c=c))
            found = visit(index+1, next_occupied)
            placed.pop()
            if found is not None:
                return found
        return None

    sources = visit(0, rail_mask)
    _require(sources is not None, "no detached ordered scattering fits")
    grid = base.copy()
    for o in sources:
        block = grid[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]]
        block[o["mask"]] = o["patch"][o["mask"]]
    return grid


def _crossing(objects):
    return any(a["side"] == 0 and b["side"] == 1 and _shared_rows(a, b)[0] < _shared_rows(a, b)[1]
               and a["c"] > b["c"] for a in objects for b in objects)


def _make_pair(rng, kind, transpose, palette):
    bg, low, high = palette
    colors = [c for c in range(10) if c != bg]
    neutral = [c for c in colors if c not in (low, high)]
    main = int(rng.integers(2))
    frames = (low, high)
    height = int(rng.integers(16 if kind in ("exchange", "teach", "test") else 12, 31))
    width = int(rng.integers(28 if kind == "teach" else 26 if kind.startswith("test") else 20, 31))
    depths = {side: rng.integers(1, 4, size=height) for side in (0, 1)}
    plans = []
    shared_color = int(rng.choice(neutral))
    if kind == "exchange":
        hh = int(rng.integers(4, 7))
        row = int(rng.integers(1, height-hh-5))
        p = rng.integers(1, 4, size=hh)
        if p.max() == p.min():
            p[1] = 1 if p[0] != 1 else 2
        depths[main][row:row+hh] = p
        thickness = int(rng.integers(3, 5))
        second = int(rng.choice([c for c in neutral if c != shared_color]))
        plans = [(main, row, hh, thickness, "parallel", shared_color, False),
                 (main, row, hh, thickness, "parallel", second, False),
                 (main, row, hh, 3, "flat", None, bool(rng.integers(2))),
                 (1-main, row+hh+2, 3, 3, "flat", shared_color, False)]
    elif kind == "teach":
        hh = int(rng.integers(3, 5))
        step = hh-1
        span = hh+2*step
        row = int(rng.integers(1, height-span))
        depth = int(rng.integers(1, 3))
        for side in (0, 1):
            depths[side][row:row+span] = depth
        thickness = 3
        plans = [(main, row, hh, thickness, "flat", shared_color, False),
                 (main, row+step, hh, 3, "flat", None, bool(rng.integers(2))),
                 (main, row+2*step, hh, 3, "flat", None, bool(rng.integers(2))),
                 (1-main, row, hh, thickness, "flat", shared_color, False)]
    else:
        count = 4 if kind == "test" else 3 if kind == "test-rich" else int(rng.integers(1, 5))
        mode = int(rng.integers(3))
        hh = int(rng.integers(3, 6))
        step = 0 if mode == 0 else 1 if mode == 1 else hh-1
        span = hh+(count-1)*step
        _require(span+2 <= height, "row plan exceeds canvas")
        row = int(rng.integers(1, height-span))
        if kind == "test-rich":
            depths[main][row:row+hh] = rng.integers(1, 5, size=hh)
            depths[main][row] = 1
            depths[main][row+hh-1] = 4
        for i in range(count):
            shape = "flat" if i == count-1 else ("parallel" if step == 0 and kind.startswith("test")
                                                else "teeth" if step == 0 else "flat")
            plans.append((main, row+i*step, hh, 3, shape, None, rng.random() < .4))
        # Secondary objects may interact across classes or occupy independent
        # row bands. Additional training can also be a single-class scene.
        others = int(rng.integers(0, 3)) if kind.startswith("test") else int(rng.integers(0, 4))
        for _ in range(others):
            hh2 = int(rng.integers(3, 6))
            r2 = int(rng.integers(1, height-hh2))
            plans.append((1-main, r2, hh2, 3, "flat", None, rng.random() < .4))
    base = np.full((height, width), bg, dtype=np.int8)
    for r in range(height):
        base[r, :depths[0][r]] = low
        base[r, width-depths[1][r]:] = high
    frontier = {0: depths[0].copy(), 1: depths[1].copy(), 2: width}
    objects = []
    for side, row, hh, thick, shape, color, pattern in plans:
        _add_layer(rng, objects, frontier, side, row, hh, frames[side], bg, colors,
                   thickness=thick, shape=shape, color=color, patterned=pattern)
    _require(np.all(frontier[0]+frontier[1] < width), "opposite settled terraces collide")
    output = base.copy()
    for o in objects:
        block = output[o["r"]:o["r"]+o["h"], o["c"]:o["c"]+o["w"]]
        _require(block.shape == o["mask"].shape and np.all(block[o["mask"]] == bg), "invalid target")
        block[o["mask"]] = o["patch"][o["mask"]]
    _require(2*np.count_nonzero(output == bg) > output.size, "insufficient background")
    inp = _scatter(rng, base, bg, objects, require_crossing=kind == "teach")
    scene = _decode(inp)
    if kind == "teach":
        _require(_crossing(scene["objects"]), "teaching pair must show opposite-class passage")
    predicted = _settle(scene)[0]
    _require(np.array_equal(predicted, output), "construction/order mismatch")
    if transpose:
        inp, output = inp.T, output.T
    return {"input": inp.copy(), "output": output.copy()}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    _require(low <= int(value) <= high, f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    mode = int(rng.integers(5))
    stable_palette = rng.random() < .35
    shared_palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            transpose = (mode == 1 if mode <= 1 else (split == "test") if mode == 2
                         else (split == "train") if mode == 3 else bool(i % 2))
            palette = shared_palette if stable_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
            kind = ("exchange" if split == "train" and i == 0 else "teach"
                    if split == "train" and i == 1 else str(rng.choice(("test", "test-rich")))
                    if split == "test" else "free")
            for _ in range(180):
                try:
                    pair = _make_pair(rng, kind, transpose, palette)
                except ValueError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError(f"could not construct {kind} example within bounded attempts")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    scene = _decode(pair["input"])
    out, positions, orders, edges = _settle(scene)
    objects = scene["objects"]
    bbox_interlock, exchange, shared, shared_crossing = False, False, False, False
    for i, a in enumerate(objects):
        for j, b in enumerate(objects[i+1:], i+1):
            s, e = _shared_rows(a, b)
            if s >= e:
                continue
            bbox_interlock |= (max(positions[i], positions[j]) < min(positions[i]+a["w"], positions[j]+b["w"]))
            same_mask = a["r"] == b["r"] and np.array_equal(a["mask"], b["mask"])
            if same_mask and a["side"] == b["side"] and not np.array_equal(a["patch"], b["patch"]):
                # Shape-only assembly admits swapping these colored cargo items.
                alternative = out.copy()
                for src, dest in ((a, j), (b, i)):
                    target = objects[dest]
                    block = alternative[target["r"]:target["r"]+target["h"],
                                        positions[dest]:positions[dest]+target["w"]]
                    block[target["mask"]] = src["patch"][src["mask"]]
                exchange |= not np.array_equal(alternative, out)
            ia = a["mask"] & ~_boundary(a["mask"])
            ib = b["mask"] & ~_boundary(b["mask"])
            pair_shared = (same_mask and a["side"] != b["side"]
                           and np.array_equal(a["patch"][ia], b["patch"][ib]))
            shared |= pair_shared
            shared_crossing |= pair_shared and _crossing([a, b])
    # An induced three-item dependency path shows indirect blocking and teaches
    # preservation of the perpendicular coordinate, not sorting into global bins.
    graph = set(edges)
    chain = any((a, b) in graph and (b, c) in graph
                and _shared_rows(objects[a], objects[c])[0] >= _shared_rows(objects[a], objects[c])[1]
                for a in range(len(objects)) for b in range(len(objects)) for c in range(len(objects)))
    return dict(shape=tuple(np.asarray(pair["input"]).shape), orientation=int(scene["transpose"]),
                roles=(scene["background"], scene["rails"][0]["color"], scene["rails"][1]["color"]),
                count=len(objects), classes=tuple(len(orders[s]) for s in (0, 1)),
                footprints=tuple(sorted({(o["h"], o["w"]) for o in objects})),
                contour=int(max(max(max(o["lo"])-min(o["lo"]), max(o["hi"])-min(o["hi"]))
                                for o in objects)),
                patterned=any(len(np.unique(o["patch"][o["mask"] & ~_boundary(o["mask"])])) > 1 for o in objects),
                crossing=_crossing(objects), chain=chain, interlock=bool(bbox_interlock), exchange=bool(exchange),
                shared=bool(shared), shared_crossing=bool(shared_crossing), edges=len(edges),
                travel=tuple(abs(o["c"]-c) for o, c in zip(objects, positions)))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, "num_train", 2, 10)
    _count(b, "num_test", 1, 4)
    _require(a > b, "illegal split counts")
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, output = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == output.shape, "shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _decode(inp)
            expected, positions, _, _ = _settle(scene)
            actual = output.T if scene["transpose"] else output
            _require(np.array_equal(expected, actual), "wrong cargo destination or pixels")
            physical, physical_positions = _simulate(scene)
            _require(np.array_equal(physical, actual) and physical_positions == positions, "unit-step collision mismatch")
            _require(np.array_equal(scene["grid"][scene["rail_mask"]], actual[scene["rail_mask"]]), "fixed rail changed")
            _require(np.array_equal(np.bincount(inp.ravel(), minlength=10), np.bincount(output.ravel(), minlength=10)),
                     "pixel colors or area were not conserved")
            _require(all(o["c"] != c for o, c in zip(scene["objects"], positions)), "generated cargo must visibly move")
            # Every row is a gapless terrace extending from each fixed rail;
            # exactly complementary outlines are stronger than first contact.
            h, w = actual.shape
            for side in (0, 1):
                for r in range(h):
                    ranges = []
                    for o, c in zip(scene["objects"], positions):
                        if o["side"] == side and o["r"] <= r < o["r"]+o["h"]:
                            rr = r-o["r"]
                            ranges.append((c+int(o["lo"][rr]), c+int(o["hi"][rr])))
                    ranges.sort(reverse=bool(side))
                    edge = int(scene["rails"][side]["edge"][r])
                    for lo, hi in ranges:
                        _require(lo == edge if side == 0 else hi == edge, "gap in an assembled terrace")
                        edge = hi if side == 0 else lo
            if split == "train":
                train_features.append(_features(pair))
    _require(any(f["interlock"] for f in train_features), "training must defeat rectangular-footprint packing")
    _require(any(f["exchange"] for f in train_features), "training must resolve shape-only cargo exchange")
    _require(any(f["shared_crossing"] for f in train_features), "training must expose frame-selected passage")
    _require(any(f["chain"] for f in train_features), "training must expose indirect overlap blocking")


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed variation audit, count contract, mutations, and ablations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "no cross-seed diversity")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "seed is not deterministic")
    features = [_features(pair) for p in puzzles for s in ("train", "test") for pair in p[s]]
    shapes = {f["shape"] for f in features}
    _require(len(shapes) >= 40, "too little canvas variation")
    _require({f["orientation"] for f in features} == {0, 1}, "missing an axis")
    _require(len({f["roles"] for f in features}) >= 30, "too little role variation")
    _require(len({f["footprints"] for f in features}) >= 50, "too little mask geometry variation")
    _require(len({f["count"] for f in features}) >= 5, "too little count variation")
    _require(all(any(_features(q)["exchange"] for q in p["train"]) for p in puzzles), "missing B ablation witness")
    _require(all(any(_features(q)["interlock"] for q in p["train"]) for p in puzzles), "missing A ablation witness")
    _require(any(min(f["classes"]) == 0 for f in features), "single-class extension absent")
    _require(any(f["patterned"] for f in features), "no copied pattern variation")
    _require({f["crossing"] for f in features} == {False, True}, "no crossing variation")
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            p = generate_puzzle(12000+10*a+b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = puzzles[0]
    changed = {s: [{k: v.copy() for k, v in q.items()} for q in first[s]] for s in first}
    changed["test"][0]["output"] = changed["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("mislabeled output accepted")
    # Shape-only assembly can swap equal molds, yet the family validator must
    # reject that physically different answer because stable cargo order matters.
    changed = {s: [{k: v.copy() for k, v in q.items()} for q in first[s]] for s in first}
    scene = _decode(changed["train"][0]["input"])
    _, positions, _, _ = _settle(scene)
    objs = scene["objects"]
    for i, a in enumerate(objs):
        for j, b in enumerate(objs[i+1:], i+1):
            if (a["side"] == b["side"] and a["r"] == b["r"] and np.array_equal(a["mask"], b["mask"])
                    and not np.array_equal(a["patch"], b["patch"])):
                objs[i]["patch"], objs[j]["patch"] = b["patch"], a["patch"]
                swapped = _place(scene, positions)
                changed["train"][0]["output"] = swapped.T.copy() if scene["transpose"] else swapped
                break
        else:
            continue
        break
    try:
        validate_generated_puzzle(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("shape-only swapped answer accepted")
    return {"seeds": 42, "pairs": len(features), "shapes": len(shapes),
            "footprint_sets": len({f["footprints"] for f in features}),
            "counts": sorted({f["count"] for f in features}),
            "roles": len({f["roles"] for f in features}),
            "orientations": sorted({f["orientation"] for f in features}),
            "interlocking_pairs": sum(f["interlock"] for f in features),
            "exchange_witnesses": sum(f["exchange"] for f in features),
            "crossing_pairs": sum(f["crossing"] for f in features),
            "patterned_pairs": sum(f["patterned"] for f in features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
