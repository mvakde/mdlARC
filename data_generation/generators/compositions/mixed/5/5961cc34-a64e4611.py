"""Relay-conditioned empty centers. No parent modules or saved examples are used.

The input cap supplies the highlight color. Routing is completed before the
clipped 3x3 test is applied. Only centers, never entire windows, are recolored.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 99809288, 572258437, 767696098, 509563202, 792596859,
              936995421, 914746744, 599684922, 892574778, 889828845,
              376334530, 293622610, 359972564, 439312284, 707946211,
              868039030, 181795908, 144483921, 93895836, 101652413,
              784080664, 815040267, 324617499, 81315281, 865136289,
              140918227, 145350478, 876086666, 224506397, 493173511,
              579800482, 184363145, 774941742, 869415635, 189785889,
              33521956, 109119398, 183121488, 5259419, 943301773, 383322316)


def _step(cell, direction, distance=1):
    dr, dc = DIRS[direction]
    return cell[0] + distance * dr, cell[1] + distance * dc


def _components(cells):
    remaining, groups = set(cells), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, group = [first], {first}
        while todo:
            cell = todo.pop()
            for d in range(4):
                other = _step(cell, d)
                if other in remaining:
                    remaining.remove(other)
                    group.add(other)
                    todo.append(other)
        groups.append(group)
    return groups


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=True)


def _decode(inp):
    h, w = inp.shape
    bg = int(inp[0, 0])
    assert all(int(inp[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
    groups = _components(map(tuple, np.argwhere(inp != bg)))
    boundary = [g for g in groups if any(r in (0, h-1) or c in (0, w-1) for r, c in g)]
    assert len(boundary) == 1, "one boundary launcher is required"
    launcher = boundary[0]
    ends = [p for p in launcher if p[0] in (0, h-1) or p[1] in (0, w-1)]
    assert len(ends) == 1
    start = ends[0]
    direction = 2 if start[0] == 0 else 0 if start[0] == h-1 else 1 if start[1] == 0 else 3
    line = [_step(start, direction, k) for k in range(len(launcher))]
    assert 3 <= len(line) <= 6 and set(line) == launcher
    signal, highlight = int(inp[start]), int(inp[line[-1]])
    assert signal != highlight and all(int(inp[p]) == signal for p in line[:-1])
    objects, port_colors, bodies = [], set(), set()
    for group in groups:
        if group == launcher:
            continue
        colors, counts = np.unique([inp[p] for p in group], return_counts=True)
        assert len(colors) == 2 and counts[0] != counts[1]
        pc, body = int(colors[np.argmin(counts)]), int(colors[np.argmax(counts)])
        port = {p for p in group if inp[p] == pc}
        assert 1 <= len(port) <= 3 and len(group - port) > len(port)
        assert len(_components(group - port)) == 1
        r0, r1 = min(r for r, c in group), max(r for r, c in group)
        c0, c1 = min(c for r, c in group), max(c for r, c in group)
        assert 0 < r0 <= r1 < h-1 and 0 < c0 <= c1 < w-1
        sides = [d for d, ok in enumerate((all(r == r0 for r, c in port),
                  all(c == c1 for r, c in port), all(r == r1 for r, c in port),
                  all(c == c0 for r, c in port))) if ok]
        assert len(sides) == 1, "port must have one extreme side"
        side = sides[0]
        span = sorted(c if side % 2 == 0 else r for r, c in port)
        assert span == list(range(span[0], span[0] + len(span)))
        assert all(_step(p, (side + 2) % 4) in group - port for p in port)
        objects.append({"cells": group, "port": port, "direction": side})
        port_colors.add(pc)
        bodies.add(body)
    assert objects and len(port_colors) == 1
    pc = next(iter(port_colors))
    assert len({bg, signal, highlight, pc}) == 4
    assert not bodies & {bg, signal, highlight, pc}
    assert np.count_nonzero(inp == bg) > inp.size // 2
    # Object components, including the launcher, have an eight-neighbor gap.
    owner = {p: i for i, g in enumerate(groups) for p in g}
    assert all(owner.get((r+dr, c+dc), i) == i for (r, c), i in owner.items()
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    return bg, signal, highlight, set(line[:-1]), line[-1], direction, objects


def _project_route(shape, tail, cap, direction, objects):
    """Input solver: nearest-object projection along each beam lane."""
    h, w = shape
    foreground = np.zeros(shape, dtype=bool)
    for p in tail | {cap}:
        foreground[p] = True
    origins, reached, transitions, gaps = [cap], [], [], []
    for _ in range(len(objects) + 1):
        dr, dc = DIRS[direction]
        stops = []
        for r, c in origins:
            edge = r+1 if direction == 0 else w-c if direction == 1 else h-r if direction == 2 else c+1
            hits = [(edge, -1)]
            for i, obj in enumerate(objects):
                for rr, cc in obj["cells"]:
                    distance = (rr-r)*dr + (cc-c)*dc
                    if distance > 0 and (rr-r)*dc - (cc-c)*dr == 0:
                        hits.append((distance, i))
            stops.append(min(hits))
        assert len(set(stops)) == 1, "partial-width or simultaneous different hits"
        distance, hit = stops[0]
        for origin in origins:
            for k in range(1, distance):
                foreground[_step(origin, direction, k)] = True
        if hit == -1:
            return foreground, reached, transitions, gaps
        assert hit not in reached, "cyclic relay route"
        obj = objects[hit]
        assert obj["direction"] != (direction+2) % 4
        reached.append(hit)
        transitions.append((direction, obj["direction"], len(obj["port"])))
        gaps.append(distance-1)
        for p in obj["cells"]:
            foreground[p] = True
        origins, direction = sorted(obj["port"]), obj["direction"]
    raise AssertionError("route does not terminate")


def _near(foreground):
    h, w = foreground.shape
    padded = np.pad(foreground, 1, constant_values=False)
    near = np.zeros_like(foreground)
    for r in range(3):
        for c in range(3):
            near |= padded[r:r+h, c:c+w]
    return near


def _paint(foreground, bg, signal, highlight):
    out = np.full(foreground.shape, bg, dtype=np.int8)
    out[~_near(foreground)] = highlight
    out[foreground] = signal
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    bg, signal, highlight, tail, cap, direction, objects = _decode(inp)
    foreground, _, _, _ = _project_route(inp.shape, tail, cap, direction, objects)
    return _paint(foreground, bg, signal, highlight)


def _walk_route(shape, tail, cap, direction, objects):
    """Construction check: advance the front one cell at a time."""
    h, w = shape
    owner = {p: i for i, obj in enumerate(objects) for p in obj["cells"]}
    trace, reached = set(tail) | {cap}, []
    front = [_step(cap, direction)]
    for _ in range((len(objects)+1)*max(h, w) + len(objects)):
        inside = [0 <= r < h and 0 <= c < w for r, c in front]
        if not any(inside):
            return trace, reached
        if not all(inside):
            return None
        hits = {owner[p] for p in front if p in owner}
        if hits:
            if len(hits) != 1:
                return None
            hit = next(iter(hits))
            if hit in reached or any(owner.get(p) != hit for p in front):
                return None
            obj = objects[hit]
            if obj["direction"] == (direction+2) % 4:
                return None
            trace |= obj["cells"]
            reached.append(hit)
            direction = obj["direction"]
            front = [_step(p, direction) for p in sorted(obj["port"])]
        else:
            trace.update(front)
            front = [_step(p, direction) for p in front]
    return None


def _move(cells, r0, c0):
    return {(r+r0, c+c0) for r, c in cells}


def _separated(cells, occupied):
    return not any((r+dr, c+dc) in occupied for r, c in cells
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _edge_runs(cells, h, w, side, width, port=False):
    result = []
    length = w if side % 2 == 0 else h
    for start in range(1 if port else 0, length-width+1):
        if port and start+width == length:
            continue
        strip = ({(0, start+k) for k in range(width)} if side == 0 else
                 {(h-1, start+k) for k in range(width)} if side == 2 else
                 {(start+k, w-1) for k in range(width)} if side == 1 else
                 {(start+k, 0) for k in range(width)})
        if not strip <= cells:
            continue
        if port and (not all(_step(p, (side+2) % 4) in cells-strip for p in strip)
                     or len(_components(cells-strip)) != 1):
            continue
        result.append(strip)
    return result


def _shape(rng, incoming, outgoing, in_width, out_width, hollow, extended):
    if hollow:
        hole_h, hole_w = (3, int(rng.integers(4, 6))) if extended else (3, 3)
        if rng.random() < .5:
            hole_h, hole_w = hole_w, hole_h
        h, w = hole_h+4, hole_w+4
        cells = {(r, c) for r in range(h) for c in range(w)
                 if not (2 <= r < h-2 and 2 <= c < w-2)}
    else:
        h, w = int(rng.integers(4, 8)), int(rng.integers(4, 8))
        h = max(h, out_width+2 if outgoing % 2 else in_width)
        w = max(w, out_width+2 if not outgoing % 2 else in_width)
        kind = int(rng.integers(4))
        cells = {(r, c) for r in range(h) for c in range(w)}
        if kind == 1:
            cells -= {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
        elif kind == 2:
            cells -= {(0, 0), (0, 1), (1, 0)}
        elif kind == 3:
            # A one-cell boundary notch; its footprint, not its bbox, survives.
            cells.remove((0, w//2))
    ports = _edge_runs(cells, h, w, outgoing, out_width, port=True)
    entries = _edge_runs(cells, h, w, (incoming+2) % 4, in_width)
    if not ports or not entries:
        return None
    return cells, ports[int(rng.integers(len(ports)))], entries[int(rng.integers(len(entries)))], h, w


def _random_layout(rng, n, widths, hollow_index, extended, decoys):
    lower = 18 if n <= 2 else 25
    h, w = map(int, rng.integers(lower, 31, size=2))
    if n <= 1 and hollow_index is None:
        h = int(rng.integers(11, 24))
    if rng.random() < .12:
        h = w = 30
    initial = int(rng.integers(4))
    transverse = int(rng.integers(2, (w if initial % 2 == 0 else h)-2))
    start = (h-1, transverse) if initial == 0 else (0, transverse) if initial == 2 else (transverse, 0) if initial == 1 else (transverse, w-1)
    tail = {_step(start, initial, k) for k in range(int(rng.integers(2, 5)))}
    cap = _step(start, initial, len(tail))
    budget = [160]

    def grow(objects, occupied, rays, origins, incoming):
        index = len(objects)
        if index == n:
            result = _walk_route((h, w), tail, cap, initial, objects)
            if result is not None and result[1] == list(range(n)):
                turns = [(o["direction"]-(initial if i == 0 else objects[i-1]["direction"])) % 4 for i, o in enumerate(objects)]
                if n >= 2 and (0 not in turns or not any(t in (1, 3) for t in turns)):
                    return None
                return objects, result[0]
            return None
        for _ in range(28):
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            outgoing = (incoming + int(rng.choice([-1, 0, 1]))) % 4
            shape = _shape(rng, incoming, outgoing, len(origins), widths[index],
                           index == hollow_index, extended)
            if shape is None:
                continue
            cells, port, entry, oh, ow = shape
            anchor = min(entry)
            options = []
            for distance in range(2, max(h, w)):
                target = _step(min(origins), incoming, distance)
                r0, c0 = target[0]-anchor[0], target[1]-anchor[1]
                if r0 < 1 or c0 < 1 or r0+oh >= h or c0+ow >= w:
                    continue
                placed = _move(cells, r0, c0)
                approach = {_step(p, incoming, k) for p in origins for k in range(1, distance)}
                if placed & rays or not _separated(placed, occupied) or approach & occupied:
                    continue
                options.append((placed, _move(port, r0, c0), approach))
            if not options:
                continue
            placed, placed_port, approach = options[int(rng.integers(len(options)))]
            obj = {"cells": placed, "port": placed_port, "direction": outgoing}
            result = grow(objects+[obj], occupied | placed, rays | approach,
                          sorted(placed_port), outgoing)
            if result is not None:
                return result
        return None

    route = grow([], tail | {cap}, tail | {cap}, [cap], initial)
    if route is None:
        return None
    objects, trace = route
    occupied = tail | {cap} | set().union(*(o["cells"] for o in objects))
    for _ in range(decoys):
        for _attempt in range(70):
            side = int(rng.integers(4))
            shape = _shape(rng, side, side, 1, 1, False, False)
            if shape is None:
                continue
            cells, port, _, oh, ow = shape
            r0, c0 = int(rng.integers(1, h-oh)), int(rng.integers(1, w-ow))
            placed = _move(cells, r0, c0)
            if placed & trace or not _separated(placed, occupied):
                continue
            objects.append({"cells": placed, "port": _move(port, r0, c0), "direction": side})
            occupied |= placed
            break
        else:
            return None
    if len(occupied) >= h*w//2:
        return None
    return (h, w), tail, cap, initial, objects


def _fallback(rng, n, widths, hollow_index, extended, decoys):
    """Guaranteed bounded packing: straight relays followed by a turn."""
    row, col = int(rng.integers(2, 5)), 5
    tail, cap, objects = {(row+1, 0), (row+1, 1)}, (row+1, 2), []
    for i in range(n):
        last, width = i == n-1, widths[i]
        hollow = i == hollow_index
        oh, ow = (7, 7) if hollow else (5, max(3, width+2) if last else 3)
        if hollow and extended:
            oh = 9
        cells = {(r, c) for r in range(row, row+oh) for c in range(col, col+ow)}
        if hollow:
            cells -= {(r, c) for r in range(row+2, row+oh-2) for c in range(col+2, col+ow-2)}
        elif rng.random() < .5:
            cells.remove((row+oh-1, col))
        direction = 0 if last else 1
        port = ({(row, col+1+k) for k in range(width)} if last else
                {(row+1+k, col+ow-1) for k in range(width)})
        objects.append({"cells": cells, "port": port, "direction": direction})
        col += ow+1
    for i in range(decoys):
        r0, c0 = 19, 2+8*i
        cells = {(r, c) for r in range(r0, r0+4) for c in range(c0, c0+4)}
        objects.append({"cells": cells, "port": {(r0, c0+1)}, "direction": 0})
    # Dihedral symmetries change orientation without changing the semantics.
    initial = 1
    for _ in range(int(rng.integers(4))):
        rotate = lambda p: (29-p[1], p[0])
        tail, cap = {rotate(p) for p in tail}, rotate(cap)
        objects = [{"cells": {rotate(p) for p in o["cells"]},
                    "port": {rotate(p) for p in o["port"]},
                    "direction": (o["direction"]-1) % 4} for o in objects]
        initial = (initial-1) % 4
    return (30, 30), tail, cap, initial, objects


def _make_pair(rng, n, widths, hollow_index, extended, decoys, colors,
               fallback_only=False):
    layout = None
    for _ in range(0 if fallback_only else 8):
        layout = _random_layout(rng, n, widths, hollow_index, extended, decoys)
        if layout is not None:
            break
    if layout is None:
        layout = _fallback(rng, n, widths, hollow_index, extended, decoys)
    shape, tail, cap, initial, objects = layout
    walked = _walk_route(shape, tail, cap, initial, objects)
    assert walked is not None and walked[1] == list(range(n))
    trace = walked[0]
    bg, signal, highlight, pc, *bodies = colors
    inp = np.full(shape, bg, dtype=np.int8)
    for i, obj in enumerate(objects):
        for p in obj["cells"]:
            inp[p] = pc if p in obj["port"] else bodies[i % len(bodies)]
    for p in tail:
        inp[p] = signal
    inp[cap] = highlight
    # A direct neighbor predicate is separate from solve's shifted-array test.
    out = np.full(shape, bg, dtype=np.int8)
    h, w = shape
    for r in range(h):
        for c in range(w):
            if (r, c) in trace:
                out[r, c] = signal
            elif not any((rr, cc) in trace for rr in range(max(0, r-1), min(h, r+2))
                         for cc in range(max(0, c-1), min(w, c+2))):
                out[r, c] = highlight
    return {"input": inp, "output": out}


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _count(seed, "seed", 0, 2**128-1)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    varying = bool(rng.integers(2))
    max_width = int(rng.integers(2, 4))
    shift = int(rng.integers(2))  # longer route, or larger hollow core
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            demo = split == "train" and i < 2
            if demo:
                n, decoys = 2, 1 if i == 0 else int(rng.integers(3))
                widths = [1, max_width] if i == 0 else [max_width, 1]
                hollow_index = 0 if i == 0 else None
            elif split == "test":
                n = int(rng.choice([3, 4])) if shift == 0 else int(rng.choice([2, 3]))
                widths = list(map(int, rng.choice([1, max_width], size=n)))
                decoys = int(rng.integers(4))
                hollow_index = int(rng.integers(n)) if shift == 1 else None
            else:
                n, decoys = int(rng.integers(3)), int(rng.integers(4))
                decoys = max(1, decoys) if n == 0 else decoys
                widths = list(map(int, rng.choice([1, max_width], size=n)))
                hollow_index = int(rng.integers(n)) if n and rng.random() < .3 else None
            colors = (list(np.roll(palette, 1)) if demo and i == 1 else
                      list(map(int, rng.permutation(10)))) if varying and not (demo and i == 0) else palette
            colors = colors[:int(rng.integers(5, 7))]
            for _ in range(6):
                pair = _make_pair(rng, n, widths, hollow_index,
                                  split == "test" and shift == 1, decoys, colors)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    return puzzle


def _facts(inp, out):
    bg, signal, highlight, tail, cap, direction, objects = _decode(inp)
    fg, reached, transitions, gaps = _project_route(inp.shape, tail, cap, direction, objects)
    near = _near(fg)
    assert np.array_equal(out, _paint(fg, bg, signal, highlight))
    assert set(map(int, np.unique(out))) == {bg, signal, highlight}
    assert np.all(out[fg] == signal) and np.all(out[near & ~fg] == bg)
    assert np.all(out[~near] == highlight)
    # Omitting A: use raw input obstacles, with cap retained as an obstacle.
    raw_empty = ~_near(inp != bg)
    routing_changes_centers = bool(np.any(raw_empty != ~near))
    # There is a background cell blocked only diagonally, not orthogonally.
    orthogonal = fg.copy()
    padded = np.pad(fg, 1)
    h, w = fg.shape
    for dr, dc in DIRS:
        orthogonal |= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    diagonal = near & ~orthogonal
    assert np.all(out[diagonal] == bg)
    assert not reached or np.any(diagonal)
    edge = np.zeros_like(fg)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    assert np.any(edge & ~near), "clipped boundary must be observable"
    tiny = [g for g in _components(map(tuple, np.argwhere(~near))) if len(g) < 5]
    # Whole-window union would also recolor fringe belonging to clear windows.
    window_union = _near(~near) & ~fg
    assert np.any(window_union & near), "center-vs-window distinction must be visible"
    decoy_cells = set().union(*(o["cells"] for i, o in enumerate(objects) if i not in reached))
    # Read enclosed holes from actual reached footprints, without generation labels.
    hole_sizes = []
    for i in reached:
        cells = objects[i]["cells"]
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        blank = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)} - cells
        for hole in _components(blank):
            if all(r0 < r < r1 and c0 < c < c1 for r, c in hole):
                hole_sizes.append(len(hole))
    return {"shape": inp.shape, "initial": direction, "active": len(reached),
            "decoys": len(objects)-len(reached), "transitions": transitions,
            "gaps": gaps, "routing_changes_centers": routing_changes_centers,
            "diagonal_fringe": int(np.sum(diagonal)),
            "tiny_cores": len(tiny), "hole_sizes": hole_sizes,
            "highlight": int(np.sum(~near)),
            "erased_highlight": sum(out[p] == highlight for p in decoy_cells),
            "palette": (bg, signal, highlight),
            "object_shapes": [(max(r for r, c in o["cells"])-min(r for r, c in o["cells"])+1,
                               max(c for r, c in o["cells"])-min(c for r, c in o["cells"])+1) for o in objects]}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            records[split].append(_facts(inp, out))
    train = records["train"]
    assert any(f["active"] >= 2 for f in train)
    assert any(f["routing_changes_centers"] for f in train)
    assert any(f["erased_highlight"] > 0 for f in train)
    assert any(f["tiny_cores"] > 0 for f in train), "show no component-size cutoff"
    transitions = [t for f in train for t in f["transitions"]]
    assert any(a == b for a, b, _ in transitions)
    assert any(a != b for a, b, _ in transitions)
    widths = {width for _, _, width in transitions}
    assert 1 in widths and len(widths) >= 2
    assert all({t[2] for t in f["transitions"]} <= widths for f in records["test"])
    max_depth = max(f["active"] for f in train)
    max_hole = max(size for f in train for size in f["hole_sizes"])
    assert all(f["active"] > max_depth or max(f["hole_sizes"], default=0) > max_hole
               for f in records["test"]), "each test must extend route depth or hollow area"


def self_test():
    """42 streamed seeds, all legal counts, no retained grids or disk output."""
    signatures, shapes, structures, directions, active_counts = set(), set(), set(), set(), set()
    palettes, object_shapes, turn_kinds, widths = set(), set(), set(), set()
    total = tiny = full_size = long_gap = erased = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                facts = _facts(inp, out)
                digest.update(bytes(inp.shape)); digest.update(inp.tobytes()); digest.update(out.tobytes())
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, normalized))
                shapes.add(inp.shape); directions.add(facts["initial"])
                active_counts.add(facts["active"]); palettes.add(facts["palette"])
                object_shapes.update(facts["object_shapes"])
                turn_kinds.update((b-a) % 4 for a, b, _ in facts["transitions"])
                widths.update(t[2] for t in facts["transitions"])
                total += 1; tiny += facts["tiny_cores"] > 0
                full_size += inp.shape == (30, 30)
                long_gap += max(facts["gaps"], default=0) >= 8
                erased += facts["erased_highlight"] > 0
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(structures) >= 160
    assert len(shapes) >= 25 and len(object_shapes) >= 12 and len(palettes) >= 20
    assert directions == {0, 1, 2, 3} and {0, 1, 2, 3} <= active_counts
    assert turn_kinds == {0, 1, 3} and widths == {1, 2, 3}
    assert tiny >= 42 and full_size and long_gap and erased >= 42
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            puzzle = generate_puzzle(75, tr, te)
            validate_generated_puzzle(puzzle)
    for tr, te in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    repeated_a, repeated_b = generate_puzzle(2026), generate_puzzle(2026)
    for split in ("train", "test"):
        for a, b in zip(repeated_a[split], repeated_b[split]):
            assert all(np.array_equal(a[k], b[k]) for k in a)
    # Exercise bounded packing directly; ordinary seeds need not reach it.
    fallback_rng = np.random.default_rng(48105)
    fallback_cases = 0
    for n in range(5):
        for width in (1, 2, 3):
            for hollow in (False, True) if n else (False,):
                for extended in (False, True) if hollow else (False,):
                    pair = _make_pair(fallback_rng, n, [1 if i % 2 else width for i in range(n)],
                                      n-1 if hollow else None, extended, 3,
                                      [0, 2, 4, 3, 1, 5], fallback_only=True)
                    facts = _facts(pair["input"], pair["output"])
                    assert facts["active"] == n and facts["decoys"] == 3
                    assert np.array_equal(solve(pair["input"]), pair["output"])
                    if hollow:
                        assert facts["hole_sizes"] == [15 if extended else 9]
                    fallback_cases += 1
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": total, "distinct_structures": len(structures),
            "shapes": len(shapes), "object_shapes": len(object_shapes),
            "tiny_core_pairs": tiny, "full_size_pairs": full_size,
            "long_approach_pairs": long_gap, "erased_highlight_pairs": erased,
            "fallback_cases": fallback_cases}
