"""Register two ink-carrying terminals, then connect their registration ports.

Only standard Python and NumPy are used. Generation keeps candidates in memory;
there are no saved grids, execution traces, caches, or construction-state solvers.
"""
from collections import deque
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 796021773, 226814060, 559815301, 225664571, 899311778,
              375017894, 218889321, 489977761, 41913865, 600383199,
              155588762, 278264836, 721951979, 754126551, 169033052,
              988603104, 821881134, 854417580, 186783688, 593318684,
              496568088, 668798403, 94211174, 874098408, 49112723,
              469568320, 2488607, 933235198, 127641572, 144178117,
              788353341, 688577144, 153531242, 134914643, 17693139,
              468947179, 837379677, 971262992, 894852970, 618503744,
              959661294)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _decode(a):
    """Decode colors, guides and terminal components from this input alone."""
    h, w = a.shape
    _require(min(h, w) >= 5, "no room for unambiguous guides")
    bg = int(a[0, 0])
    _require(all(int(a[p]) == bg for p in ((0, w - 1), (h - 1, 0), (h - 1, w - 1))),
             "all corners must show background")
    edge = np.zeros(a.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    anchors = {tuple(map(int, p)): int(a[tuple(p)])
               for p in np.argwhere(edge & (a != bg))}
    marker_colors = sorted(set(anchors.values()))
    _require(1 <= len(marker_colors) <= 2, "one or two registration identities required")
    systems = {}
    for color in marker_colors:
        points = [p for p in anchors if anchors[p] == color]
        horizontal = (len({r for r, c in points}) == 1
                      and all(c in (0, w - 1) for r, c in points))
        vertical = (len({c for r, c in points}) == 1
                    and all(r in (0, h - 1) for r, c in points))
        _require(horizontal != vertical and 1 <= len(points) <= 2,
                 "anchors must define one consistent noncorner guide")
        axis = 0 if horizontal else 1  # coordinate changed by translation
        guide = points[0][axis]
        _require(0 < guide < a.shape[axis] - 1, "corner anchors are ambiguous")
        for r, c in points:
            _require(all(not (0 <= r + dr < h and 0 <= c + dc < w)
                         or a[r + dr, c + dc] == bg for dr, dc in DIRS),
                     "boundary anchors must be isolated")
        systems[color] = (axis, guide)
    interior_marks = {tuple(map(int, p)) for p in
                      np.argwhere(np.isin(a, marker_colors) & ~edge)}
    _require(len(interior_marks) == 2, "exactly two interior registration ports required")
    _require({int(a[p]) for p in interior_marks} == set(marker_colors),
             "every identity needs a matching terminal")
    other = sorted(set(map(int, np.unique(a))) - {bg} - set(marker_colors))
    _require(len(other) == 2, "one shared ink and one wall color required")
    candidates = []
    for ink in other:
        mask = a == ink
        for p in interior_marks:
            mask[p] = True
        comps = _components(mask)
        if (len(comps) == 2 and all(2 <= len(c) <= 8 and len(c & interior_marks) == 1
                                    for c in comps)):
            candidates.append((ink, comps))
    _require(len(candidates) == 1, "terminal ink must be visually unambiguous")
    ink, comps = candidates[0]
    wall = next(c for c in other if c != ink)
    _require(np.count_nonzero(a == wall) >= 3, "walls missing")
    movers, claimed = [], set(anchors)
    for comp in comps:
        reg = next(iter(comp & interior_marks))
        axis, guide = systems[int(a[reg])]
        delta = (guide - reg[0], 0) if axis == 0 else (0, guide - reg[1])
        target = {(r + delta[0], c + delta[1]): int(a[r, c]) for r, c in comp}
        _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp),
                 "source terminals must be interior")
        _require(all(0 < r < h - 1 and 0 < c < w - 1 and a[r, c] != wall
                     for r, c in target), "terminal destination hits a wall or boundary")
        _require(not claimed.intersection(target), "terminal destinations overlap")
        claimed.update(target)
        movers.append({"source": comp, "reg": reg, "axis": axis, "delta": delta,
                       "target": target,
                       "port": (reg[0] + delta[0], reg[1] + delta[1])})
    return {"bg": bg, "ink": ink, "wall": wall, "anchors": anchors,
            "systems": systems, "movers": movers}


def _register(a, info):
    registered = a.copy()
    # Erase every source before any target is placed: movement is simultaneous.
    for m in info["movers"]:
        for p in m["source"]:
            registered[p] = info["bg"]
    for m in info["movers"]:
        for p, color in m["target"].items():
            registered[p] = color
    return registered


def _shortest(free, start, goal):
    """Breadth-first distances and number of minimum paths, capped at two."""
    h, w = free.shape
    dist = np.full((h, w), -1, dtype=np.int16)
    ways = np.zeros((h, w), dtype=np.uint8)
    parent = {}
    dist[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        nd = int(dist[r, c]) + 1
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if not (0 <= q[0] < h and 0 <= q[1] < w and free[q]):
                continue
            if dist[q] == -1:
                dist[q], ways[q], parent[q] = nd, ways[r, c], (r, c)
                queue.append(q)
            elif dist[q] == nd:
                ways[q] = min(2, int(ways[q]) + int(ways[r, c]))
    if dist[goal] < 0:
        return [], 0
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return path[::-1], int(ways[goal])


def _route(registered, info):
    ports = [m["port"] for m in info["movers"]]
    free = registered == info["bg"]
    for p in ports:
        free[p] = True
    path, ways = _shortest(free, *ports)
    _require(ways == 1 and len(path) > 2, "minimum route must exist and be unique")
    return path, free


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _decode(a)
    registered = _register(a, info)
    path, _ = _route(registered, info)
    result = registered.copy()
    for p in path[1:-1]:
        result[p] = info["ink"]
    return result


def _turns(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _longer_alternative(free, path):
    # A path through a cycle has a replacement after at least one path cell is
    # blocked. Early exit keeps this bounded by at most 898 small BFS calls.
    for p in path[1:-1]:
        trial = free.copy()
        trial[p] = False
        other, _ = _shortest(trial, path[0], path[-1])
        if other:
            _require(len(other) > len(path), "shortest path was not unique")
            return True
    return False


def _features(pair, *, check_alternative=True):
    a, b = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == b.shape, "input/output shape mismatch")
    info = _decode(a)
    registered = _register(a, info)
    path, free = _route(registered, info)
    expected = registered.copy()
    for p in path[1:-1]:
        _require(registered[p] == info["bg"], "route crosses terminal body")
        expected[p] = info["ink"]
    _require(np.array_equal(expected, b), "output violates registration and routing")
    _require(all(m["delta"] != (0, 0) for m in info["movers"]),
             "each generated terminal must move")
    _require(np.array_equal(a[a == info["wall"]], b[a == info["wall"]]), "wall moved")
    _require(all(b[p] == color for p, color in info["anchors"].items()), "anchor moved")
    detour = len(path) - 1 - sum(abs(path[0][k] - path[-1][k]) for k in (0, 1))
    _require(detour > 0 and _turns(path) >= 2, "walls must force an orthogonal detour")
    _require(np.count_nonzero(registered == info["bg"]) > len(path), "no off-route background")
    alternative = _longer_alternative(free, path) if check_alternative else False
    if check_alternative:
        _require(alternative, "no longer alternative route to distinguish shortest routing")
    # Omitting A changes the endpoint set and the completed grid. Omitting B
    # returns 'registered', which lacks every intermediate route cell.
    _require({m["reg"] for m in info["movers"]} != set((path[0], path[-1])),
             "registration did not change route endpoints")
    _require(not np.array_equal(registered, b), "routing contribution missing")
    eccentric = any(any(2 * m["reg"][k] != min(p[k] for p in m["source"])
                         + max(p[k] for p in m["source"]) for k in (0, 1))
                    for m in info["movers"] if len(m["source"]) >= 3)
    wall_points = np.argwhere(a == info["wall"])
    low, high = wall_points.min(axis=0), wall_points.max(axis=0)
    return {"shape": a.shape, "palette": (info["bg"], info["wall"], info["ink"]),
            "axes": tuple(sorted({m["axis"] for m in info["movers"]})),
            "guides": len(info["systems"]), "anchors": len(info["anchors"]),
            "sizes": tuple(sorted(len(m["source"]) for m in info["movers"])),
            "turns": _turns(path), "length": len(path) - 1, "detour": detour,
            "wall_shape": tuple(map(int, high - low + 1)),
            "alternative": alternative, "eccentric": eccentric,
            "source_on_route": any(m["reg"] in set(path[1:-1]) for m in info["movers"]),
            "internal_source": any(all(low[0] < p[0] < high[0]
                                       and low[1] < p[1] < high[1]
                                       for p in m["source"]) for m in info["movers"])}


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split type")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check_counts(a, b)
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            sig = _signature(_grid(pair["input"]))
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            features[split].append(_features(pair))
    train = features["train"]
    _require(any(f["axes"] == (0, 1) for f in train), "mixed guides need a demonstration")
    _require(any(f["eccentric"] for f in train), "rigid eccentric carrier demonstration missing")
    _require(any(f["source_on_route"] for f in train), "erasure-before-routing witness missing")
    _require(len({f["shape"] for f in train}) >= 2, "training geometry did not vary")
    _require(all(all(f["wall_shape"][k] > max(t["wall_shape"][k] for t in train)
                     for k in (0, 1)) for f in features["test"]),
             "tests must enlarge the wall field along both axes")
    return None


def _check_counts(num_train, num_test):
    for name, x, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not low <= x <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    _require(num_train > num_test, "num_train must exceed num_test")


def _maze(rng, rows, cols):
    """A random spanning-tree maze with additional cycle-forming openings."""
    mask = np.ones((2 * rows + 1, 2 * cols + 1), dtype=bool)
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    seen, stack = {start}, [start]
    mask[2 * start[0] + 1, 2 * start[1] + 1] = False
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in DIRS
                   if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in seen]
        if not options:
            stack.pop()
            continue
        nr, nc = options[int(rng.integers(len(options)))]
        mask[r + nr + 1, c + nc + 1] = False
        mask[2 * nr + 1, 2 * nc + 1] = False
        seen.add((nr, nc))
        stack.append((nr, nc))
    openings = [(r, c) for r in range(1, 2 * rows) for c in range(1, 2 * cols)
                if (r % 2 != c % 2) and mask[r, c]]
    rng.shuffle(openings)
    for p in openings[:int(rng.integers(1, min(5, len(openings)) + 1))]:
        mask[p] = False
    return mask


def _target(rng, free, size):
    cells = [tuple(map(int, p)) for p in np.argwhere(free)]
    reg = cells[int(rng.integers(len(cells)))]
    shape = {reg}
    for _ in range(size - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in shape for dr, dc in DIRS
                           if 0 <= r + dr < free.shape[0] and 0 <= c + dc < free.shape[1]
                           and free[r + dr, c + dc]} - shape)
        frontier = [p for p in frontier if max(abs(p[k] - reg[k]) for k in (0, 1)) <= 3]
        if not frontier:
            return None
        shape.add(frontier[int(rng.integers(len(frontier)))])
    return reg, shape


def _source_options(walls, reg, shape, axis):
    h, w = walls.shape
    options = []
    for shift in range(-walls.shape[axis], walls.shape[axis]):
        if shift == 0:
            continue
        delta = (shift, 0) if axis == 0 else (0, shift)
        points = {(r + delta[0], c + delta[1]) for r, c in shape}
        if all(0 < r < h - 1 and 0 < c < w - 1 and not walls[r, c] for r, c in points):
            options.append(((reg[0] + delta[0], reg[1] + delta[1]), points))
    return options


def _put_anchors(rng, grid, targets, axes, colors, bg, end_codes=None):
    h, w = grid.shape
    anchors = {}
    for j, ((reg, shape), axis, color) in enumerate(zip(targets, axes, colors)):
        if end_codes is None:
            ends = (0, 1) if rng.random() < .35 else (int(rng.integers(2)),)
        else:
            ends = ((0,), (1,), (0, 1))[end_codes[j]]
        for end in ends:
            p = (reg[0], 0 if end == 0 else w - 1) if axis == 0 else (0 if end == 0 else h - 1, reg[1])
            if p in anchors and anchors[p] != color:
                return False
            anchors[p] = color
    for p, color in anchors.items():
        grid[p] = color
    # Source placement can approach the border but must leave anchors isolated.
    for r, c in anchors:
        if any(0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr, c + dc] != bg
               for dr, dc in DIRS):
            return False
    return True


def _candidate(rng, plan, palette):
    bg, wall, ink, mark_a, mark_b = palette
    coarse = (int(rng.integers(5, 9)), int(rng.integers(5, 9))) if plan["test"] else (
        int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    local_walls = _maze(rng, *coarse)
    pads = tuple(map(int, rng.integers(4, 7, size=4)))
    h, w = local_walls.shape[0] + pads[0] + pads[1], local_walls.shape[1] + pads[2] + pads[3]
    walls = np.zeros((h, w), dtype=bool)
    walls[pads[0]:h - pads[1], pads[2]:w - pads[3]] = local_walls
    inside = np.zeros_like(walls)
    inside[pads[0]:h - pads[1], pads[2]:w - pads[3]] = ~local_walls
    targets = []
    for _ in range(2):
        size = int(rng.integers(3 if plan["test"] else 2, 8 if plan["test"] else 6))
        target = _target(rng, inside, size)
        if target is None:
            return None
        targets.append(target)
    if any(abs(r - rr) + abs(c - cc) <= 1 for r, c in targets[0][1] for rr, cc in targets[1][1]):
        return None
    free = ~walls
    for reg, shape in targets:
        for p in shape - {reg}:
            free[p] = False
    path, ways = _shortest(free, targets[0][0], targets[1][0])
    if ways != 1 or len(path) < 7 or _turns(path) < 2:
        return None
    if len(path) - 1 <= sum(abs(path[0][k] - path[-1][k]) for k in (0, 1)):
        return None
    if not _longer_alternative(free, path):
        return None
    axes = plan["axes"]
    if plan["on_route"]:
        axes = (0, 1)  # both orientations are taught by the smallest training set
    colors = [mark_a, mark_b]
    # Two components can also share a single guide when their ports visibly align.
    aligned_axes = [k for k in (0, 1) if targets[0][0][k] == targets[1][0][k]]
    if not plan["on_route"] and aligned_axes and rng.random() < .55:
        axes = (aligned_axes[0], aligned_axes[0])
        colors = [mark_a, mark_a]
    options = [_source_options(walls, reg, shape, axis)
               for (reg, shape), axis in zip(targets, axes)]
    for opts in options:
        rng.shuffle(opts)
    if plan["on_route"]:
        options[0] = [s for s in options[0] if s[0] in set(path[1:-1])]
    choices = None
    for first in options[0]:
        for second in options[1]:
            if not any(abs(r - rr) + abs(c - cc) <= 1 for r, c in first[1] for rr, cc in second[1]):
                choices = first, second
                break
        if choices:
            break
    if choices is None:
        return None
    a = np.full((h, w), bg, dtype=np.int8)
    a[walls] = wall
    for (reg, points), marker in zip(choices, colors):
        for p in points:
            a[p] = marker if p == reg else ink
    if not _put_anchors(rng, a, targets, axes, colors, bg):
        return None
    try:
        b = solve(a)
        feature = _features({"input": a, "output": b}, check_alternative=False)
    except ValueError:
        return None
    if plan["on_route"] and not (feature["source_on_route"] and feature["eccentric"]):
        return None
    # This check also guards against implementation mistakes in source placement.
    expected_ports = {x[0] for x in targets}
    if {m["port"] for m in _decode(a)["movers"]} != expected_ports:
        return None
    return {"input": a, "output": b}


def _fallback(rng, plan, palette, variant):
    """Bounded constructive backup: unequal routes around a rectangular island.

    The wall ring has two holes occupied by terminal bodies after registration.
    An optional third hole is a blind exterior branch, exposing source erasure.
    """
    bg, wall, ink, mark_a, mark_b = palette
    bh = (12 if plan["test"] else 8) + variant % 3
    bw = (11 if plan["test"] else 7) + (variant // 3) % 3
    h, w, pad = bh + 10, bw + 10, 5
    walls = np.zeros((h, w), bool)
    walls[pad:pad + bh, pad:pad + bw] = True
    top, bottom, left, right = pad + 1, pad + bh - 2, pad + 1, pad + bw - 2
    walls[top, left:right + 1] = False
    walls[bottom, left:right + 1] = False
    walls[top:bottom + 1, left] = False
    walls[top:bottom + 1, right] = False
    targets = [((top + 1, left), {(top + 1, left - j) for j in range(3)}),
               ((top + 3, right), {(top + 3, right + j) for j in range(3)})]
    for reg, shape in targets:
        for p in shape:
            walls[p] = False
    axes = (0, 1) if plan["on_route"] else plan["axes"]
    sources = []
    for j, ((reg_r, reg_c), shape) in enumerate(targets):
        axis = axes[j]
        sr = top if j == 0 and plan["on_route"] else (2 if j == 0 else h - 3)
        sc = 3 if j == 0 else w - 4
        reg = (sr, reg_c) if axis == 0 else (reg_r, sc)
        points = {(r + reg[0] - reg_r, c + reg[1] - reg_c) for r, c in shape}
        for p in points:
            walls[p] = False
        sources.append((reg, points))
    a = np.full((h, w), bg, np.int8)
    a[walls] = wall
    for (reg, points), marker in zip(sources, (mark_a, mark_b)):
        for p in points:
            a[p] = marker if p == reg else ink
    # Nine shapes times four explicit end-mark patterns give 36 distinct
    # inputs for a fixed plan, more than the 13 inputs that can precede it.
    code = variant // 9
    _require(_put_anchors(rng, a, targets, axes, (mark_a, mark_b), bg,
                          end_codes=(code % 3, code // 3)), "fallback anchors")
    return {"input": a, "output": solve(a)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)[:5]))
    plans = [{"test": False, "on_route": j == 0,
              "axes": (0, 1) if j == 0 else (j % 2, j % 2)} for j in range(num_train)]
    rng.shuffle(plans)
    plans += [{"test": True, "on_route": False,
               "axes": tuple(map(int, rng.integers(2, size=2)))} for _ in range(num_test)]
    puzzle, seen, train_shapes = {"train": [], "test": []}, set(), set()
    for i, plan in enumerate(plans):
        split = "train" if i < num_train else "test"
        pair = None
        for _ in range(240):
            candidate = _candidate(rng, plan, palette)
            if candidate is None:
                continue
            sig = _signature(candidate["input"])
            # The second training shape differs even with a requested 2/1 set.
            if sig in seen or (split == "train" and len(train_shapes) == 1
                               and candidate["input"].shape in train_shapes):
                continue
            pair = candidate
            break
        if pair is None:
            for variant in range(36):
                candidate = _fallback(rng, plan, palette, variant)
                sig = _signature(candidate["input"])
                if sig not in seen and not (split == "train" and len(train_shapes) == 1
                                            and candidate["input"].shape in train_shapes):
                    pair = candidate
                    break
        _require(pair is not None, "bounded construction exhausted")
        seen.add(_signature(pair["input"]))
        if split == "train":
            train_shapes.add(pair["input"].shape)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = _grid(pair[key])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 supplied seeds, semantic checks, count contract, and metamorphisms.

    Only small feature sets and digests survive each iteration. No files are
    written and at most two puzzles plus one candidate are held simultaneously.
    """
    digests, shapes, structures, palettes, sizes, lengths, turns, guide_counts = (set() for _ in range(8))
    on_route = internal = bilateral = pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                a = _grid(pair["input"])
                info = _decode(a)
                normal = np.full(a.shape, 0, np.int8)
                for role, color in enumerate((info["wall"], info["ink"], *sorted(info["systems"])), 1):
                    normal[a == color] = role
                structures.add(hashlib.sha256(bytes(a.shape) + normal.tobytes()).hexdigest())
                shapes.add(f["shape"])
                palettes.add(f["palette"])
                sizes.update(f["sizes"])
                lengths.add(f["length"])
                turns.add(f["turns"])
                guide_counts.add(f["guides"])
                on_route += f["source_on_route"]
                internal += f["internal_source"]
                bilateral += f["anchors"] > f["guides"]
                pairs += 1
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic seed")
    del puzzle
    _require(len(digests) == 42 and len(structures) >= 180, "structural variation missing")
    _require(len(shapes) >= 35 and len(palettes) >= 20, "geometry/palette variation missing")
    _require({2, 3, 4, 5, 6, 7} <= sizes and len(lengths) >= 20 and len(turns) >= 6,
             "terminal or route geometry too narrow")
    _require(guide_counts == {1, 2} and on_route >= 42 and bilateral >= 30,
             "guide and erasure variation missing")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7300 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(2026, 2, 1)
    rng = np.random.default_rng(26023)
    for k in (1, 2, 3):
        q = {split: [] for split in ("train", "test")}
        for split in q:
            for pair in p[split]:
                palette = rng.permutation(10)
                q[split].append({key: palette[np.rot90(value, k)[:, ::-1]].astype(np.int8)
                                 for key, value in pair.items()})
        validate_generated_puzzle(q)
    # Negative checks reject an omitted route cell and a moved boundary marker.
    pair = p["test"][0]
    info = _decode(pair["input"])
    registered = _register(pair["input"], info)
    path, _ = _route(registered, info)
    for position, color in ((path[len(path) // 2], info["bg"]),
                            (next(iter(info["anchors"])), info["bg"])):
        bad = {"input": pair["input"], "output": pair["output"].copy()}
        bad["output"][position] = color
        try:
            _features(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("damaged output accepted")
    # Exercise the constructive backup without waiting for a random rejection.
    for test in (False, True):
        for axes in ((0, 0), (1, 1), (0, 1), (1, 0)):
            for witness in (False, True):
                plan = {"test": test, "axes": axes, "on_route": witness}
                _features(_fallback(rng, plan, (0, 1, 4, 2, 3), 4))
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "shapes": len(shapes), "color_normalized_structures": len(structures),
            "palettes": len(palettes), "terminal_sizes": sorted(sizes),
            "route_lengths": [min(lengths), max(lengths)], "turn_counts": sorted(turns),
            "guide_counts": sorted(guide_counts), "source_on_route_pairs": on_route,
            "internal_source_pairs": internal, "bilateral_pairs": bilateral}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
