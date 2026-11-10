"""Mirror membranes: topology controls anchor-relative motif completion.

Only NumPy and the standard library are used. No parent files, saved puzzles,
seed state, or training examples are consulted by the input-only solver.
"""
from __future__ import annotations

from math import gcd
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=False)


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    padded = np.pad(mask, 1)
    h, w = padded.shape
    seen = np.zeros_like(padded)
    seen[0, 0] = True
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        for y, x in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    return (~seen & ~padded)[1:-1, 1:-1]


def _decode(grid):
    grid = _grid(grid)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "one background border required")
    objects, loops, leaves = [], [], []
    for cells in _components(grid != bg):
        lo, hi = cells.min(0), cells.max(0)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _holes(mask)
        colors, counts = np.unique(grid[mask], return_counts=True)
        i = len(objects)
        obj = dict(cells=cells, lo=lo, hi=hi, mask=mask, hole=hole)
        if hole.any():
            _require(len(_components(hole)) == 1 and len(colors) == 2, "one marked simple loop")
            marker_candidates = colors[counts == 2]
            _require(len(marker_candidates) == 1, "one marker domino")
            mark = int(marker_candidates[0])
            wall = int(colors[colors != mark][0])
            pts = np.argwhere(mask & (grid == mark))
            delta = np.abs(pts[0]-pts[1])
            _require(int(delta.sum()) == 1, "straight marker domino")
            # A horizontal mirror flips the vertical coordinate, and vice versa.
            obj.update(kind="C", axis=0 if delta[1] else 1, wall=wall, mark=mark)
            loops.append(i)
        else:
            obj.update(kind="L", colors=tuple(map(int, colors)))
            leaves.append(i)
        objects.append(obj)
    _require(loops and len(leaves) >= 3, "need membranes and at least three leaves")
    walls = {objects[i]["wall"] for i in loops}
    marks = {objects[i]["mark"] for i in loops}
    _require(len(walls) == len(marks) == 1, "consistent membrane colors")
    wall, mark = next(iter(walls)), next(iter(marks))
    _require(wall != mark and bg not in (wall, mark), "distinct roles")
    # Distinct objects are separated even diagonally.
    labels = np.zeros(grid.shape, dtype=int)
    for i, o in enumerate(objects):
        labels[o["mask"]] = i+1
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a > 0) & (b > 0) & (a != b)), "touching objects")
    refs = [i for i in leaves if len(objects[i]["colors"]) > 1]
    _require(len(refs) == 1, "one multicolor reference")
    ref = refs[0]
    anchor_colors = {objects[i]["colors"][0] for i in leaves if i != ref}
    _require(len(anchor_colors) == 1, "matching anchors")
    anchor = next(iter(anchor_colors))
    _require(anchor not in (bg, wall, mark), "distinct anchor role")
    for i in leaves:
        o = objects[i]
        _require(not set(o["colors"]) & {wall, mark}, "motif cannot use membrane colors")
        pts = np.argwhere(o["mask"] & (grid == anchor))
        _require(len(pts) > 0, "missing anchor")
        low, high = pts.min(0), pts.max(0)+1
        side = int(high[0]-low[0])
        _require(side == high[1]-low[1] and len(pts) == side*side, "solid square anchor")
        o.update(anchor_lo=low, side=side)
    children = {i: [] for i in loops}
    roots, parents = [], []
    for i, o in enumerate(objects):
        containing = []
        for j in loops:
            if i == j:
                continue
            inside = objects[j]["hole"][o["cells"][:, 0], o["cells"][:, 1]]
            _require(inside.all() or not inside.any(), "crossing boundary")
            if inside.all():
                containing.append(j)
        parent = min(containing, key=lambda j: int(objects[j]["hole"].sum())) if containing else None
        parents.append(parent)
        (roots if parent is None else children[parent]).append(i)
    _require(all(children[i] for i in loops), "empty membrane")
    states = []
    for i in range(len(objects)):
        state, depth, p = [0, 0], 0, parents[i]
        while p is not None:
            state[objects[p]["axis"]] ^= 1
            depth += 1
            p = parents[p]
        states.append((tuple(state), depth))
    source = objects[ref]
    motif = {(int(r-source["anchor_lo"][0]), int(c-source["anchor_lo"][1])): int(grid[r, c])
             for r, c in source["cells"]}
    quantum = source["side"]
    # Only real color boundaries matter; identical adjacent source pixels may
    # coalesce when a reference is reduced.
    for (r, c), color in motif.items():
        for dr, dc, coordinate in ((-1, 0, r), (1, 0, r+1), (0, -1, c), (0, 1, c+1)):
            if motif.get((r+dr, c+dc), bg) != color:
                quantum = gcd(quantum, abs(coordinate))
    _require(quantum > 0, "template resolution")
    base = {}
    for (r, c), color in motif.items():
        key = (r//quantum, c//quantum)
        _require(key not in base or base[key] == color, "unaligned source boundary")
        base[key] = color
    _require(len(motif) == len(base)*quantum*quantum, "incomplete source blocks")
    return dict(grid=grid, bg=bg, wall=wall, mark=mark, anchor=anchor,
                objects=objects, loops=loops, leaves=leaves, ref=ref, parents=parents,
                children=children, roots=roots, states=states, base=base,
                a=source["side"]//quantum, quantum=quantum, source_side=source["side"])


def _motif_tile(base, a, unit, bg, flips=(0, 0)):
    transformed = {(a-1-r if flips[0] else r, a-1-c if flips[1] else c): color
                   for (r, c), color in base.items()}
    rs, cs = zip(*transformed)
    tile = np.full(((max(rs)-min(rs)+1)*unit, (max(cs)-min(cs)+1)*unit), bg, dtype=np.int8)
    for (r, c), color in transformed.items():
        y, x = (r-min(rs))*unit, (c-min(cs))*unit
        tile[y:y+unit, x:x+unit] = color
    return tile


def _join(tiles, axis, gap, bg):
    cross = 1-axis
    shape = [0, 0]
    shape[axis] = sum(t.shape[axis] for t in tiles)+gap*(len(tiles)-1)
    shape[cross] = max(t.shape[cross] for t in tiles)
    out = np.full(shape, bg, dtype=np.int8)
    position = 0
    for tile in tiles:
        start = [0, 0]
        start[axis] = position
        start[cross] = (shape[cross]-1)//2-(tile.shape[cross]-1)//2
        y, x = start
        out[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
        position += tile.shape[axis]+gap
    return out


def _frame(content, bg, wall):
    out = np.pad(content, 2, constant_values=bg)
    out[0] = out[-1] = wall
    out[:, 0] = out[:, -1] = wall
    return out


def _render_decoded(data, *, ignore_mirrors=False, anchors_only=False):
    objects, bg = data["objects"], data["bg"]
    reference_state = data["states"][data["ref"]][0]
    cache = {}

    def object_tile(i):
        o = objects[i]
        if o["kind"] == "C":
            return _frame(group(data["children"][i]), bg, data["wall"])
        if anchors_only and i != data["ref"]:
            return np.full((o["side"], o["side"]), data["anchor"], dtype=np.int8)
        numerator = data["quantum"]*o["side"]
        _require(numerator % data["source_side"] == 0, "nonintegral target color boundary")
        unit = numerator//data["source_side"]
        flips = tuple(s ^ r for s, r in zip(data["states"][i][0], reference_state))
        if ignore_mirrors:
            flips = (0, 0)
        return _motif_tile(data["base"], data["a"], unit, bg, flips)

    def group(ids):
        key = tuple(sorted(ids))
        if key in cache:
            return cache[key]
        if len(ids) == 1:
            result = object_tile(ids[0])
        else:
            candidates = []
            for axis in (0, 1):
                ordered = sorted(ids, key=lambda i: int(objects[i]["lo"][axis]))
                parts, end = [], -100
                for i in ordered:
                    start = int(objects[i]["lo"][axis])
                    if start > end+1:
                        parts.append([])
                    parts[-1].append(i)
                    end = max(end, int(objects[i]["hi"][axis]))
                if len(parts) < 2:
                    continue
                cross = 1-axis
                centers = [(min(int(objects[i]["lo"][cross]) for i in part)+
                            max(int(objects[i]["hi"][cross]) for i in part))//2 for part in parts]
                if len(set(centers)) != 1:
                    continue
                try:
                    candidates.append(_join([group(part) for part in parts], axis, 1, bg))
                except ValueError:
                    continue
            _require(bool(candidates), "no centered cardinal layout")
            result = candidates[0]
            _require(all(np.array_equal(result, other) for other in candidates[1:]), "ambiguous layout")
        cache[key] = result
        return result

    result = group(data["roots"])
    _require(all(n <= 30 for n in result.shape), "canonical scene exceeds 30")
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the actual enclosing regions, cancel shared mirrors, and redraw."""
    return _render_decoded(_decode(input_grid))


# Abstract construction below is separate from visible-input decoding. A group
# adds spatial arrangement only; C adds a real boundary and toggles an axis.
def _group(rng, children, axis=None, mixed=False):
    if len(children) == 1:
        return children[0]
    rng.shuffle(children)
    axis = int(rng.integers(2)) if axis is None else axis
    if mixed and len(children) >= 3:
        cut = int(rng.integers(1, len(children)))
        return ("G", axis, (_group(rng, children[:cut], 1-axis),
                             _group(rng, children[cut:], 1-axis)))
    return ("G", axis, tuple(children))


def _scene(rng, mode, witness_axis=None):
    def leaf(i):
        return ("L", i)
    def room(axis, children, mixed=False):
        return ("C", axis, _group(rng, children, mixed=mixed))
    axis = int(rng.integers(2)) if witness_axis is None else witness_axis
    if mode == "cancel":
        # Reference outside: one reflection at the middle anchor, two at deep.
        inner = room(axis, [leaf(2)])
        outer = room(axis, [leaf(1), inner])
        scene, ref, n = _group(rng, [leaf(0), outer]), 0, 3
    elif mode == "relative":
        # Reference and one target share BOTH boundaries. Common ancestry must
        # cancel; the two remaining targets demonstrate one and two axes.
        inner = room(1-axis, [leaf(0), leaf(1)])
        outer = room(axis, [leaf(2), inner])
        scene, ref, n = _group(rng, [leaf(3), outer]), 0, 4
    elif mode == "deep":
        inside = room(int(rng.integers(2)), [leaf(0), leaf(1)])
        middle = room(1-axis, [inside])
        outside = room(axis, [middle, leaf(2)])
        scene, ref, n = _group(rng, [outside, leaf(3)]), int(rng.integers(4)), 4
    elif mode == "branch":
        left = room(axis, [leaf(0), leaf(1)])
        right = room(1-axis, [leaf(2)])
        outer = room(int(rng.integers(2)), [left, right, leaf(3)], mixed=True)
        scene, ref, n = outer, int(rng.integers(4)), 4
    else:
        n = int(rng.integers(3, 6))
        pool = [leaf(i) for i in range(n)]
        rng.shuffle(pool)
        for _ in range(int(rng.integers(1, 4))):
            count = int(rng.integers(1, min(3, len(pool))+1))
            selected, pool = pool[:count], pool[count:]
            pool.append(room(int(rng.integers(2)), selected, mixed=True))
            rng.shuffle(pool)
        scene, ref = _group(rng, pool, mixed=True), int(rng.integers(n))
    return scene, ref, n


def _template(rng, anchor, details, simple=False):
    a = 1 if simple else int(rng.choice([1, 1, 1, 2]))
    points = {(r, c): anchor for r in range(a) for c in range(a)}
    count = int(rng.integers(2, 5 if simple else 8))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Growth supports branches, long arms, thin details, and repeated colors;
    # it is not confined to an anchor-sized 3x3 mosaic.
    for step in range(count):
        for _ in range(30):
            cells = sorted(points)
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = directions[int(rng.integers(4))]
            p = (r+dr, c+dc)
            if p in points:
                continue
            candidate = dict(points)
            candidate[p] = int(details[step % len(details)])
            rs, cs = zip(*candidate)
            limit = 3 if simple else 5
            if max(rs)-min(rs)+1 > limit or max(cs)-min(cs)+1 > limit:
                continue
            mask = np.zeros((max(rs)-min(rs)+1, max(cs)-min(cs)+1), dtype=bool)
            for y, x in candidate:
                mask[y-min(rs), x-min(cs)] = True
            if _holes(mask).any():
                continue
            points = candidate
            break
    for _ in range(int(rng.integers(4))):
        points = {(c, a-1-r): color for (r, c), color in points.items()}
    if rng.random() < .5:
        points = {(r, a-1-c): color for (r, c), color in points.items()}
    asymmetric = all({(a-1-r if v else r, a-1-c if h else c): color
                      for (r, c), color in points.items()} != points
                     for v, h in ((1, 0), (0, 1), (1, 1)))
    return points, a, asymmetric


def _membrane(content, axis, rng, bg, wall, mark, compact, plain=False):
    if plain:
        out = _frame(content, bg, wall)
        if axis == 0:
            out[0, 1:3] = mark
        else:
            out[1:3, 0] = mark
        return out
    padding = [int(rng.integers(2, 4 if compact else 6)) for _ in range(4)]
    top, bottom, left, right = padding
    h, w = content.shape[0]+top+bottom, content.shape[1]+left+right
    filled = np.ones((h, w), dtype=bool)
    # Independent corner cuts and bays distort the boundary while leaving a
    # guaranteed moat around the actual contents.
    if rng.random() < .9:
        for corner in range(4):
            if rng.random() < .65:
                ph = top if corner < 2 else bottom
                pw = left if corner % 2 == 0 else right
                if ph < 3 or pw < 3:
                    continue
                dh = int(rng.integers(1, ph-1))
                dw = int(rng.integers(1, pw-1))
                ys = slice(0, dh) if corner < 2 else slice(h-dh, h)
                xs = slice(0, dw) if corner % 2 == 0 else slice(w-dw, w)
                filled[ys, xs] = False
        if not compact and rng.random() < .6:
            side = int(rng.integers(4))
            if side < 2 and padding[side] >= 3:
                start = int(rng.integers(left, w-right))
                depth = int(rng.integers(1, padding[side]-1))
                ys = slice(0, depth) if side == 0 else slice(h-depth, h)
                filled[ys, start:start+1] = False
            elif side >= 2 and padding[side] >= 3:
                start = int(rng.integers(top, h-bottom))
                depth = int(rng.integers(1, padding[side]-1))
                xs = slice(0, depth) if side == 2 else slice(w-depth, w)
                filled[start:start+1, xs] = False
    padded = np.pad(filled, 1)
    interior = filled.copy()
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy+h, dx:dx+w]
    boundary = filled & ~interior
    out = np.full((h, w), bg, dtype=np.int8)
    out[boundary] = wall
    candidates = []
    dr, dc = (0, 1) if axis == 0 else (1, 0)
    for r, c in np.argwhere(boundary):
        positions = [(int(r+k*dr), int(c+k*dc)) for k in (-1, 0, 1, 2)]
        if all(0 <= y < h and 0 <= x < w and boundary[y, x] for y, x in positions):
            candidates.append(positions[1:3])
    if not candidates:
        raise ValueError("no straight mirror segment")
    for r, c in candidates[int(rng.integers(len(candidates)))]:
        out[r, c] = mark
    view = out[top:top+content.shape[0], left:left+content.shape[1]]
    _require(np.all(view == bg), "boundary touches contents")
    view[:] = content
    return out


def _fallback_scene(mode, variant, witness_axis):
    """Small, proven-fit realization; no saved or parent puzzle is used."""
    axis = variant & 1 if witness_axis is None else witness_axis
    other, layout_axis = (variant >> 1) & 1, (variant >> 2) & 1
    def leaf(i):
        return ("L", i)
    def group(*children):
        return children[0] if len(children) == 1 else ("G", layout_axis, children)
    def room(mirror, *children):
        return ("C", mirror, group(*children))
    if mode in ("cancel", "free"):
        return group(leaf(0), room(axis, leaf(1), room(axis, leaf(2)))), 0, 3
    if mode == "relative":
        return group(leaf(3), room(axis, leaf(2), room(1-axis, leaf(0), leaf(1)))), 0, 4
    if mode == "deep":
        scene = group(leaf(3), room(axis, leaf(2), room(1-axis, room(other, leaf(0), leaf(1)))))
    else:
        scene = room(axis, room(other, leaf(0), leaf(1)), room(1-other, leaf(2)), leaf(3))
    return scene, (variant >> 3) % 4, 4


def _construct(rng, mode, compact=False, witness_axis=None, fallback_variant=None):
    colors = [int(c) for c in rng.permutation(10)]
    bg, wall, mark, anchor = colors[:4]
    details = colors[4:4+int(rng.integers(1, 4))]
    scene, ref, n = _scene(rng, mode, witness_axis)
    base, a, asym = _template(rng, anchor, details, simple=compact)
    if fallback_variant is not None:
        scene, ref, n = _fallback_scene(mode, fallback_variant, witness_axis)
        a, asym = 1, True
        base = {(0, 0): anchor, (0, 1): colors[4], (1, 0): colors[5]}
    if not asym:
        raise ValueError("symmetric prototype")
    units = [int(rng.choice([1, 1, 1, 1, 2, 2, 3])) for _ in range(n)]
    if mode in ("cancel", "relative"):
        # Both demonstrations must expose a genuine size change.
        units[ref] = 1 if a == 2 else int(rng.integers(1, 3))
        units[(ref+1) % n] = 2 if units[ref] == 1 else 1
    if compact:
        units = [1]*n
        if mode in ("cancel", "relative"):
            units[(ref+1) % n] = 2
    states = {}

    def state_walk(node, state):
        if node[0] == "L":
            states[node[1]] = state
        elif node[0] == "C":
            s = list(state)
            s[node[1]] ^= 1
            state_walk(node[2], tuple(s))
        else:
            for child in node[2]:
                state_walk(child, state)
    state_walk(scene, (0, 0))

    def draw(node):
        if node[0] == "L":
            i = node[1]
            inp = (_motif_tile(base, a, units[i], bg) if i == ref else
                   np.full((a*units[i], a*units[i]), anchor, dtype=np.int8))
            flips = tuple(x ^ y for x, y in zip(states[i], states[ref]))
            return inp, _motif_tile(base, a, units[i], bg, flips)
        if node[0] == "C":
            inp, out = draw(node[2])
            return _membrane(inp, node[1], rng, bg, wall, mark, compact,
                             plain=fallback_variant is not None), _frame(out, bg, wall)
        parts = [draw(child) for child in node[2]]
        gap = 1 if fallback_variant is not None else int(rng.integers(1, 3 if compact else 5))
        return (_join([p[0] for p in parts], node[1], gap, bg),
                _join([p[1] for p in parts], node[1], 1, bg))
    inp, out = draw(scene)
    _require(max(inp.shape) <= 28 and max(out.shape) <= 30, "space for border")
    # Cropped canonical output has no exterior halo; all input borders are blank.
    room_h, room_w = 30-inp.shape[0], 30-inp.shape[1]
    ph = int(rng.integers(2, min(room_h, 7)+1))
    pw = int(rng.integers(2, min(room_w, 7)+1))
    t, l = int(rng.integers(1, ph)), int(rng.integers(1, pw))
    inp = np.pad(inp, ((t, ph-t), (l, pw-l)), constant_values=bg)
    _require(np.array_equal(solve(inp), out), "visible reconstruction differs")
    data = _decode(inp)
    _require(not np.array_equal(out, _render_decoded(data, ignore_mirrors=True)), "no observable mirror effect")
    return dict(input=inp, output=out)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+" outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(data):
    """Containment, marked axes and source location, without colors or size."""
    def visit(i):
        if data["objects"][i]["kind"] == "L":
            return ("R",) if i == data["ref"] else ("T",)
        return ("C", data["objects"][i]["axis"],
                tuple(sorted(visit(j) for j in data["children"][i])))
    return tuple(sorted(visit(i) for i in data["roots"]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    witness_axis = int(rng.integers(2))
    puzzle, seen, training_structures = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = (("cancel", "relative")[i] if split == "train" and i < 2 else
                    str(rng.choice(["deep", "branch"])) if split == "test" else "free")
            for attempt in range(128+64):
                try:
                    pair = _construct(rng, mode, compact=attempt >= 12,
                                      witness_axis=witness_axis if mode in ("cancel", "relative") else None,
                                      fallback_variant=attempt-128 if attempt >= 128 else None)
                except ValueError:
                    continue
                key = _key(pair["input"])
                signature = _signature(_decode(pair["input"]))
                if key not in seen and (split == "train" or signature not in training_structures):
                    seen.add(key)
                    if split == "train":
                        training_structures.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted for "+mode)
        rng.shuffle(puzzle[split])
    return puzzle


def _pair_features(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    data = _decode(inp)
    _require(np.array_equal(out, _render_decoded(data)), "incorrect canonical completion")
    objs, states, ref = data["objects"], data["states"], data["ref"]
    flips = {tuple(x ^ y for x, y in zip(states[i][0], states[ref][0])) for i in data["leaves"] if i != ref}
    ratios = {(objs[i]["side"], objs[ref]["side"]) for i in data["leaves"] if i != ref}
    depths = [states[i][1] for i in data["leaves"]]
    repeated = any(states[i][1] >= 2 and states[i][0] == (0, 0) for i in data["leaves"])
    raw = _render_decoded(data, ignore_mirrors=True)
    bare = _render_decoded(data, anchors_only=True)
    _require(not np.array_equal(out, raw), "mirror ablation unchanged")
    _require(not np.array_equal(out, bare), "completion ablation unchanged")
    _require(not np.any(out == data["mark"]), "mirror marks retained")
    _require(set(map(int, np.unique(out))) == set(map(int, np.unique(inp)))-{data["mark"]}, "color loss")
    # Frames are re-parsed from actual output wall cells, independently of the
    # template completion. Count, exact rectangularity, and interior moat hold.
    frames = _components(out == data["wall"])
    _require(len(frames) == len(data["loops"]), "frame count")
    for cells in frames:
        lo, hi = cells.min(0), cells.max(0)
        y, x = lo
        yy, xx = hi
        _require(len(cells) == 2*(yy-y+xx-x), "nonrectangular frame")
        _require(np.all(out[y, x:xx+1] == data["wall"]) and np.all(out[yy, x:xx+1] == data["wall"])
                 and np.all(out[y:yy+1, x] == data["wall"]) and np.all(out[y:yy+1, xx] == data["wall"]), "frame perimeter")
        _require(np.all(out[y+1, x+1:xx] == data["bg"]) and np.all(out[yy-1, x+1:xx] == data["bg"])
                 and np.all(out[y+1:yy, x+1] == data["bg"]) and np.all(out[y+1:yy, xx-1] == data["bg"]), "one-cell moat")
    anchor_parts = _components(out == data["anchor"])
    _require(sorted(len(c) for c in anchor_parts) == sorted(objs[i]["side"]**2 for i in data["leaves"]), "anchor size multiset")
    roles = {data["anchor"]: 0}
    normalized_motif = tuple((r, c, roles.setdefault(color, len(roles))) for (r, c), color in sorted(data["base"].items()))
    return dict(loops=len(data["loops"]), leaves=len(data["leaves"]), depth=max(depths),
                ref_depth=states[ref][1], repeated=repeated, flips=flips, ratios=ratios,
                axes={objs[i]["axis"] for i in data["loops"]},
                palette=(data["bg"], data["wall"], data["mark"], data["anchor"]),
                motif=(data["a"], normalized_motif), shape=inp.shape, signature=_signature(data),
                fine=data["a"] > 1)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            key = _key(_grid(pair["input"]))
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            features = _pair_features(pair)
            if split == "train":
                training.append(features)
            else:
                _require(features["loops"] >= 3, "test should recombine more membranes")
                tests.append(features)
    _require(any(f["repeated"] and f["ref_depth"] == 0 and len(f["axes"]) == 1 and
                 any(n != d for n, d in f["ratios"]) for f in training), "cancellation demonstration")
    _require(any(f["ref_depth"] == 2 and len(f["axes"]) == 2 and (0, 0) in f["flips"] and (1, 1) in f["flips"] and
                 any(n != d for n, d in f["ratios"]) for f in training), "relative mirror demonstration")
    _require({(1, 0), (0, 1), (1, 1)} <= set.union(*(f["flips"] for f in training)), "both mirror axes must be taught")
    _require(not {f["signature"] for f in training} & {f["signature"] for f in tests}, "held-out test structure")


def self_test():
    """Bounded streaming suite: 42 seeds, no saved puzzle bank or logs."""
    signatures, shapes, palettes, motifs, topologies = set(), set(), set(), set(), set()
    axes, flips, ratios, reference_depths = set(), set(), set(), set()
    fine = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_features(pair)
                shapes.add(f["shape"])
                palettes.add(f["palette"])
                motifs.add(f["motif"])
                topologies.add(f["signature"])
                axes.update(f["axes"])
                flips.update(f["flips"])
                ratios.update((n//gcd(n, d), d//gcd(n, d)) for n, d in f["ratios"])
                reference_depths.add(f["ref_depth"])
                fine += f["fine"]
                for role in ("input", "output"):
                    digest.update(str(pair[role].shape).encode())
                    digest.update(pair[role].tobytes())
        signatures.add(digest.digest())
        if seed < 2:
            repeat = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s]) for k in a), "determinism")
        # Color-role equivariance uses only visible grids.
        p = puzzle["test"][0]
        rng = np.random.default_rng(seed+900)
        palette = rng.permutation(10)
        _require(np.array_equal(solve(palette[p["input"]]), palette[p["output"]]), "recoloring")
    _require(len(signatures) == 42 and len(shapes) >= 45 and len(palettes) >= 100, "42-seed variety")
    _require(len(motifs) >= 80 and len(topologies) >= 10 and fine >= 5, "structural variety")
    _require(axes == {0, 1} and len(flips) == 4 and len(ratios) >= 3 and {0, 1, 2, 3} <= reference_depths, "operator coverage")
    # Explicitly exercise every finite-fallback variant, including both marked
    # axes, both layout axes and all possible reference leaves.
    rng = np.random.default_rng(8722)
    for mode in ("cancel", "relative", "deep", "branch", "free"):
        for variant in range(32):
            _pair_features(_construct(rng, mode, compact=True, fallback_variant=variant))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(8100+train*10+test, train, test)
            validate_generated_puzzle(p)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(7, 2, 1)
    p["test"][0]["output"] = p["test"][0]["output"].copy()
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
