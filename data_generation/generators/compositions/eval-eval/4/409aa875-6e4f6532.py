"""Virtual docking: compass-oriented carriers aim at a shared receiver pattern.

Only standard-library modules and NumPy are used. The solver reads the input
pixels, enumerates rigid placements, and checks simultaneous endpoint counts.
It never receives a seed, demonstrations, or construction metadata.
"""
from collections import Counter
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


SIDES = {"top": (-1, 0), "right": (0, 1),
         "bottom": (1, 0), "left": (0, -1)}
DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                   if r or c)
DISTANCE = 5


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            new = {(r + dr, c + dc) for dr, dc in DIRECTIONS} & remaining
            remaining -= new
            component |= new
            stack.extend(sorted(new))
        result.append(component)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return {(r + a, c + b) for a, b in offsets}


def _decode_arrows(points):
    result = []
    for component in _components(points):
        if len(component) != 3:
            raise ValueError("Arrow marks must be separated three-cell motifs")
        matches = [(p, d) for p in sorted(component) for d in DIRECTIONS
                   if _arrow(p, d) == component]
        if len(matches) != 1:
            raise ValueError("Unrecognized arrowhead")
        result.append(matches[0])
    if not result:
        raise ValueError("Each carrier needs arrowheads")
    return result


def _transform(cells, t):
    moved = []
    for (r, c), v in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        moved.append((r, c, v))
    r0 = min(p[0] for p in moved)
    c0 = min(p[1] for p in moved)
    return {(r - r0, c - c0): v for r, c, v in moved}


def _shift(cells, offset):
    return {(r + offset[0], c + offset[1]): v for (r, c), v in cells.items()}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _facing(cells, band_colors):
    core = [p for p, v in cells.items() if v not in band_colors]
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    labels = []
    for (r, c), v in cells.items():
        if v not in band_colors:
            continue
        options = [side for side, condition in (
            ("top", r < r0), ("bottom", r > r1),
            ("left", c < c0), ("right", c > c1)) if condition]
        if len(options) != 1:
            raise ValueError("Labels must project beyond exactly one core side")
        side = options[0]
        separation = {"top": r0 - r, "bottom": r - r1,
                      "left": c0 - c, "right": c - c1}[side]
        if separation != 1:
            raise ValueError("Labels must extend exactly one cell beyond the core")
        labels.append((options[0], v))
    for side in SIDES:
        identities = [value for facing, value in labels if facing == side]
        if len(identities) > 3 or len(set(identities)) > 1:
            raise ValueError("Each labeled side needs one identity and at most three cells")
    directions = {SIDES[s] for s, _ in labels}
    if not any(a[0] * b[0] + a[1] * b[1] == 0
               for a in directions for b in directions):
        raise ValueError("Every carrier needs two adjacent compass labels")
    return labels


def _structure(grid, background):
    height, width = grid.shape
    structures = []
    for thicknesses in product(range(3), repeat=4):
        thickness = dict(zip(SIDES, thicknesses))
        marked = [side for side in SIDES if thickness[side]]
        if len(marked) < 2 or not any(
                sum(first * second for first, second in zip(SIDES[left], SIDES[right])) == 0
                for left in marked for right in marked):
            continue
        top, bottom = thickness["top"], height - thickness["bottom"]
        left, right = thickness["left"], width - thickness["right"]
        if min(bottom - top, right - left) < 4:
            continue
        spans = {"top": grid[:top, left:right],
                 "bottom": grid[bottom:, left:right],
                 "left": grid[top:bottom, :left],
                 "right": grid[top:bottom, right:]}
        inner_edges = {"top": grid[top, left:right],
                       "bottom": grid[bottom - 1, left:right],
                       "left": grid[top:bottom, left],
                       "right": grid[top:bottom, right - 1]}
        sides = {}
        for side in marked:
            span = spans[side]
            identity = int(span.flat[0])
            if identity == background or not np.all(span == identity):
                break
            sides[side] = identity
        if len(sides) != len(marked) or len(set(sides.values())) != len(sides):
            continue
        if any(np.all(edge == edge[0]) and edge[0] != background
               and (side not in sides or edge[0] == sides[side])
               for side, edge in inner_edges.items()):
            continue
        structural = np.ones(grid.shape, dtype=bool)
        structural[top:bottom, left:right] = False
        allowed = np.zeros(grid.shape, dtype=bool)
        for side, identity in sides.items():
            matching = grid == identity
            if side == "top":
                allowed[:top, :] |= matching[:top, :]
            elif side == "bottom":
                allowed[bottom:, :] |= matching[bottom:, :]
            elif side == "left":
                allowed[:, :left] |= matching[:, :left]
            else:
                allowed[:, right:] |= matching[:, right:]
        if np.any(structural & ~allowed):
            continue
        structures.append((structural, sides, (top, bottom - 1, left, right - 1)))
    if len(structures) != 1:
        raise ValueError("Outer strips must determine one rectangular free field")
    return structures[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("Background must be the unique most frequent color")
    structural, sides, bounds = _structure(grid, bg)
    band_colors = set(sides.values())
    if len(band_colors) != len(sides):
        raise ValueError("Compass sides must have different identities")
    free = set(map(tuple, np.argwhere(~structural).tolist()))
    objects = _components(p for p in free if grid[p] != bg)
    sources, receivers = [], {}
    for component in objects:
        cells = {p: int(grid[p]) for p in component}
        if set(cells.values()) & band_colors:
            sources.append(cells)
        else:
            receivers.update(cells)
    if not sources or not receivers:
        raise ValueError("Need carriers and detached receivers")
    core_colors = set().union(*(set(s.values()) - band_colors for s in sources))
    if len(core_colors) != 2 or bg in core_colors:
        raise ValueError("Carriers need common body and arrow colors")
    totals = {v: sum(list(s.values()).count(v) for s in sources) for v in core_colors}
    body, signal = sorted(core_colors, key=lambda v: (-totals[v], v))
    for s in sources:
        if (set(s.values()) - band_colors != core_colors
                or list(s.values()).count(body) <= list(s.values()).count(signal)):
            raise ValueError("Each carrier must have more body than arrow cells")
        _decode_arrows(p for p, v in s.items() if v == signal)
        _facing(s, band_colors)
    if not set(receivers.values()) <= {body, signal}:
        raise ValueError("Receivers use the body/collision and arrow/single colors")
    return {"grid": grid, "background": bg, "body": body, "signal": signal,
            "structural": structural, "sides": sides, "bands": band_colors,
            "bounds": bounds, "sources": sources,
            "receivers": receivers}


def _shots(cells, signal, distance=DISTANCE):
    arrows = _decode_arrows(p for p, v in cells.items() if v == signal)
    return Counter((r + distance * dr, c + distance * dc)
                   for (r, c), (dr, dc) in arrows)


def _fits(points, bounds):
    r0, r1, c0, c1 = bounds
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in points)


def _candidates(source, info, use_compass=True, use_multiplicity=True,
                distance=DISTANCE):
    candidates, seen = [], set()
    receivers = info["receivers"]
    for t in range(8):
        cells = _transform(source, t)
        if use_compass and any(info["sides"].get(side) != value
                               for side, value in _facing(cells, info["bands"])):
            continue
        shots = _shots(cells, info["signal"], distance)
        first = min(shots)
        for target in sorted(receivers):
            delta = (target[0] - first[0], target[1] - first[1])
            endpoints = Counter({(r + delta[0], c + delta[1]): n
                                 for (r, c), n in shots.items()})
            if not endpoints.keys() <= receivers.keys():
                continue
            if use_multiplicity and any(n > 1 and receivers[p] == info["signal"]
                                       for p, n in endpoints.items()):
                continue
            placed = _shift(cells, delta)
            if not _fits(placed, info["bounds"]) or placed.keys() & receivers.keys():
                continue
            key = tuple(sorted(placed.items()))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((placed, endpoints))
    return candidates


def _configurations(info, use_compass=True, use_multiplicity=True,
                    distance=DISTANCE, limit=2):
    choices = [_candidates(s, info, use_compass, use_multiplicity, distance)
               for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    receivers, solutions = info["receivers"], {}

    def search(depth, occupied, counts, placed):
        if len(solutions) >= limit:
            return
        if depth == len(order):
            if set(counts) != set(receivers):
                return
            if use_multiplicity and any((counts[p] == 1) != (v == info["signal"])
                                       for p, v in receivers.items()):
                return
            signature = tuple(sorted(occupied.items()))
            solutions[signature] = (occupied.copy(), counts.copy(), placed.copy())
            return
        for cells, endpoints in choices[order[depth]]:
            if cells.keys() & occupied.keys():
                continue
            new_counts = counts + endpoints
            if use_multiplicity and any(new_counts[p] > 1 and receivers[p] == info["signal"]
                                       for p in endpoints):
                continue
            # Prune an uncovered receiver if no remaining carrier can reach it.
            reachable = set(new_counts)
            for i in order[depth + 1:]:
                reachable.update(p for _, shots in choices[i] for p in shots)
            if not receivers.keys() <= reachable:
                continue
            search(depth + 1, {**occupied, **cells}, new_counts,
                   placed + [(order[depth], cells, endpoints)])
    search(0, {}, Counter(), [])
    return list(solutions.values())


def _render(info, solution):
    output = np.full_like(info["grid"], info["background"])
    output[info["structural"]] = info["grid"][info["structural"]]
    for p, v in {**info["receivers"], **solution[0]}.items():
        output[p] = v
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    solutions = _configurations(info)
    if len(solutions) != 1:
        raise ValueError("Input must determine exactly one resulting pixel map")
    return _render(info, solutions[0])


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _path(rng, start, end):
    r, c = start
    points = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < .5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        points.add((r, c))
    return points


def _make_piece(rng, sides, body, signal, count, parallel=False, first_direction=None):
    arrows = []
    same_direction = first_direction or _pick(rng, DIRECTIONS)
    for i in range(count):
        for _ in range(100):
            d = same_direction if parallel or (i == 0 and first_direction) else _pick(rng, DIRECTIONS)
            if parallel and count == 2:
                step = int(rng.integers(4, 7))
                # A repeated pair gives a symmetric *virtual* registration pattern.
                tip = (0, 0) if not i else (step, 0) if d[0] == 0 else (0, step)
            else:
                tip = (int(rng.integers(7)), int(rng.integers(7)))
            cells = _arrow(tip, d)
            old = set().union(*[a[2] for a in arrows]) if arrows else set()
            union = old | cells
            if cells & _halo(old):
                continue
            if any(max(p[a] for p in union) - min(p[a] for p in union) > 8 for a in (0, 1)):
                continue
            arrows.append((tip, d, cells))
            break
        else:
            return None
    marks = set().union(*(a[2] for a in arrows))
    min_r, min_c = min(r for r, _ in marks), min(c for _, c in marks)
    marks = {(r - min_r + 1, c - min_c + 1) for r, c in marks}
    h, w = max(r for r, _ in marks) + 2, max(c for _, c in marks) + 2
    available = list(sides)
    adjacent = [(a, b) for i, a in enumerate(available) for b in available[i + 1:]
                if sum(x * y for x, y in zip(SIDES[a], SIDES[b])) == 0]
    if not adjacent:
        return None
    selected = list(_pick(rng, adjacent))
    for s in available:
        if s not in selected and rng.random() < .2:
            selected.append(s)
    labels, supports = {}, []
    for side in selected:
        length = w if side in ("top", "bottom") else h
        patch = int(rng.integers(1, min(3, length) + 1))
        positions = sorted(int(v) for v in rng.choice(length, patch, replace=False))
        for i in positions:
            label, support = {"top": ((-1, i), (0, i)),
                              "bottom": ((h, i), (h - 1, i)),
                              "left": ((i, -1), (i, 0)),
                              "right": ((i, w), (i, w - 1))}[side]
            labels[label] = sides[side]
            supports.append(support)
    core = set(marks)
    endpoints = supports + [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                            (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
    # Connect arrows and label supports, then grow variable irregular body branches.
    anchors = [min(component) for component in _components(marks)]
    root = anchors[0]
    for p in anchors[1:] + endpoints:
        core.update(_path(rng, root if rng.random() < .35 else
                          min(core, key=lambda q: (abs(q[0] - p[0]) + abs(q[1] - p[1]), q)), p))
    desired_body = max(len(marks) + 1, int(h * w * rng.uniform(.28, .65)))
    for _ in range(h * w):
        if len(core - marks) >= desired_body:
            break
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in SIDES.values()
                           if 0 <= r + dr < h and 0 <= c + dc < w} - core)
        if not frontier:
            return None
        core.add(_pick(rng, frontier))
    if len(core - marks) <= len(marks):
        return None
    return _transform({**{p: body for p in core}, **{p: signal for p in marks}, **labels}, 0)


def _origins(points, bounds):
    r0, r1, c0, c1 = bounds
    low_r, high_r = min(r for r, _ in points), max(r for r, _ in points)
    low_c, high_c = min(c for _, c in points), max(c for _, c in points)
    return [(r, c) for r in range(r0 - low_r, r1 - high_r + 1)
            for c in range(c0 - low_c, c1 - high_c + 1)]


def _scene(rng, palette, large=False):
    bg, body, signal = palette[:3]
    h = int(rng.integers(24 if large else 18, 31))
    w = int(rng.integers(24 if large else 18, 31))
    base = np.full((h, w), bg, dtype=np.int8)
    side_order = list(SIDES)
    # At least an L-shaped pair of strips, with optional third/fourth strips.
    first = int(rng.integers(4))
    chosen = {side_order[first], side_order[(first + 1) % 4]}
    chosen.update(s for s in side_order if rng.random() < .65)
    sides = {s: palette[3 + i] for i, s in enumerate(side_order) if s in chosen}
    thickness = {side: int(rng.integers(1, 3)) for side in side_order if side in chosen}
    for s in ("left", "right", "top", "bottom"):
        if s not in chosen:
            continue
        d = thickness[s]
        if s == "left":
            base[:, :d] = sides[s]
        elif s == "right":
            base[:, -d:] = sides[s]
        elif s == "top":
            base[:d, :] = sides[s]
        else:
            base[-d:, :] = sides[s]
    bounds = (thickness.get("top", 0), h - thickness.get("bottom", 0) - 1,
              thickness.get("left", 0), w - thickness.get("right", 0) - 1)
    return base, bounds, sides


def _construct(rng, palette, counts, teaching=None, require_collision=False):
    bg, body, signal = palette[:3]
    base, bounds, sides = _scene(rng, palette, len(counts) >= 2 or max(counts) >= 3)
    finals, shots, occupied = [], Counter(), set()
    for i, count in enumerate(counts):
        found = False
        for _ in range(45):
            want_collision = bool(i and (require_collision or rng.random() < .45))
            direction = None
            anchor = None
            if want_collision:
                previous = _pick(rng, finals)
                tip, old_direction = _pick(rng, _decode_arrows(p for p, v in previous.items() if v == signal))
                anchor = (tip[0] + 5 * old_direction[0], tip[1] + 5 * old_direction[1])
                direction = tuple(-v for v in old_direction)
            piece = _make_piece(rng, sides, body, signal, count,
                                parallel=teaching == "compass" and i == 0,
                                first_direction=direction)
            if piece is None:
                continue
            local_shots = _shots(piece, signal)
            if want_collision:
                origins = [(anchor[0] - p[0], anchor[1] - p[1]) for p in local_shots]
            else:
                origins = _origins(set(piece) | set(local_shots), bounds)
            rng.shuffle(origins)
            for origin in origins:
                placed = _shift(piece, origin)
                ends = Counter({(r + origin[0], c + origin[1]): n for (r, c), n in local_shots.items()})
                if not _fits(set(placed) | set(ends), bounds):
                    continue
                if placed.keys() & _halo(occupied) or placed.keys() & shots.keys():
                    continue
                if ends.keys() & (occupied | placed.keys()):
                    continue
                finals.append(placed)
                occupied.update(placed)
                shots += ends
                found = True
                break
            if found:
                break
        if not found:
            return None
    if require_collision and not any(n >= 2 for n in shots.values()):
        return None
    receivers = {p: signal if n == 1 else body for p, n in shots.items()}
    inp, out = base.copy(), base.copy()
    for p, v in receivers.items():
        inp[p] = out[p] = v
    for piece in finals:
        for p, v in piece.items():
            out[p] = v
    blocked = _halo(receivers)
    sources = []
    prefer_overlap = bool(rng.integers(2))
    for i, final in enumerate(finals):
        transforms = list(int(t) for t in rng.permutation(8))
        if i == 0 and teaching == "compass":
            transforms = [t for t in transforms if 0 < t < 4]
        elif i == 0 and teaching == "collision":
            transforms = [t for t in transforms if t >= 4]
        found = False
        for t in transforms:
            source = _transform(final, t)
            if i == 0 and teaching == "collision" and any(
                    _signature(_transform(source, j)) == _signature(final) for j in range(4)):
                continue
            origins = _origins(source, bounds)
            rng.shuffle(origins)
            eligible = []
            for origin in origins:
                placed = _shift(source, origin)
                if placed.keys() & blocked:
                    continue
                eligible.append(placed)
                if bool(placed.keys() & occupied) == prefer_overlap:
                    break
            if eligible:
                placed = eligible[-1]
                sources.append(placed)
                blocked.update(_halo(placed))
                found = True
                break
        if not found:
            return None
    for source in sources:
        for p, v in source.items():
            inp[p] = v
    try:
        info = _parse(inp)
        solutions = _configurations(info)
        if len(solutions) != 1 or not np.array_equal(_render(info, solutions[0]), out):
            return None
        if teaching == "compass" and len(_configurations(info, use_compass=False)) < 2:
            return None
        if teaching == "collision" and len(_configurations(info, use_multiplicity=False)) < 2:
            return None
        if np.array_equal(inp, out):
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_roles = bool(rng.integers(2))
    roles = [int(v) for v in rng.choice(10, 2, replace=False)]
    shift = _pick(rng, ("carriers", "arrowheads"))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            palette = [int(v) for v in rng.permutation(10)]
            if shared_roles:
                palette = [v for v in palette if v not in roles]
                palette = [palette[0], *roles, *palette[1:]]
            teaching = ("compass" if index == 0 else "collision" if index == 1 else None) if split == "train" else None
            if teaching == "compass":
                counts = [2]
            elif teaching == "collision":
                counts = [1, 2]
            elif split == "train":
                counts = [int(rng.integers(1, 3)) for _ in range(int(rng.integers(1, 3)))]
            elif shift == "carriers":
                counts = [int(rng.integers(1, 3)) for _ in range(3)]
                # Several single-arrow carriers would often be exchangeable.
                # Two multi-arrow constellations anchor the larger test scene.
                counts[0] = counts[1] = 2
            else:
                counts = [3 for _ in range(int(rng.integers(1, 3)))]
            for _ in range(160):
                pair = _construct(rng, palette, counts, teaching,
                                  require_collision=teaching == "collision")
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded carrier construction exhausted")
    # Teaching content is guaranteed, but its displayed order is not a cue.
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    solutions = _configurations(info)
    assert len(solutions) == 1, "Placement is not uniquely determined"
    solution = solutions[0]
    assert np.array_equal(_render(info, solution), out), "Output violates virtual docking"
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert np.array_equal(inp[info["structural"]], out[info["structural"]])
    assert all(out[p] == v for p, v in info["receivers"].items())
    assert sum(solution[1].values()) == sum(
        len(_decode_arrows(p for p, v in s.items() if v == info["signal"]))
        for s in info["sources"])
    moves, arrow_counts, directions, body_shapes = [], [], set(), []
    for source_index, final, _ in solution[2]:
        source = info["sources"][source_index]
        transforms = [t for t in range(8)
                      if _signature(_transform(source, t)) == _signature(final)]
        assert transforms, "Carrier colors or internal geometry changed"
        moves.append("translate" if 0 in transforms else "rotate" if any(t < 4 for t in transforms) else "reflect")
        arrows = _decode_arrows(p for p, v in final.items() if v == info["signal"])
        arrow_counts.append(len(arrows))
        directions.update(d for _, d in arrows)
        body_shapes.append(_signature({p: int(v == info["signal"]) + 1
                                      for p, v in final.items() if v not in info["bands"]}))
    source_union = set().union(*(set(s) for s in info["sources"]))
    collision_points = [p for p, n in solution[1].items() if n > 1]
    cross_collisions = sum(sum(p in ends for _, _, ends in solution[2]) > 1 for p in collision_points)
    return {"shape": inp.shape, "background": info["background"],
            "roles": (info["body"], info["signal"]), "sides": tuple(info["sides"]),
            "pieces": len(info["sources"]), "arrows": tuple(sorted(arrow_counts)),
            "directions": tuple(sorted(directions)), "moves": moves,
            "collisions": len(collision_points), "cross_collisions": cross_collisions,
            "convergence": max(solution[1].values()),
            "body_shapes": body_shapes, "overlap": len(source_union & solution[0].keys()),
            "compass_needed": len(_configurations(info, use_compass=False)) > 1,
            "multiplicity_needed": len(_configurations(info, use_multiplicity=False)) > 1,
            "zero_distance_solutions": len(_configurations(info, distance=0))}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = _grid(pair["input"])
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            features = _features(pair)
            assert features["zero_distance_solutions"] == 0
            if split == "train":
                training.append(features)
            else:
                queries.append(features)
    assert any(f["compass_needed"] for f in training), "Teach compass disambiguation"
    assert any(f["cross_collisions"] for f in training), "Teach simultaneous shared receivers"
    assert any(f["multiplicity_needed"] for f in training), "Teach collision-dependent placement"
    assert any("rotate" in f["moves"] for f in training), "Teach rotation"
    assert any("reflect" in f["moves"] for f in training), "Teach reflection"
    assert all(features["pieces"] <= 2 and max(features["arrows"]) <= 2
               for features in training), "Training must teach the lower-count regime"
    carrier_shift = all(features["pieces"] == 3 for features in queries)
    arrow_shift = all(max(features["arrows"]) == 3 for features in queries)
    assert carrier_shift or arrow_shift, "Queries must extend one demonstrated count axis"


def _projection_regressions():
    """Check complete scenes beyond the constructor's usual two-shot collisions."""
    def exercise(finals, origins, poses):
        base = np.zeros((28, 28), dtype=np.int8)
        base[:, 0], base[:, -1] = 6, 4
        base[0, :], base[-1, :] = 3, 5
        inp, expected = base.copy(), base.copy()
        counts = Counter()
        for final in finals:
            counts += _shots(final, 2)
        for p, n in counts.items():
            inp[p] = expected[p] = 2 if n == 1 else 1
        for final, origin, pose in zip(finals, origins, poses):
            for p, value in final.items():
                assert expected[p] == 0
                expected[p] = value
            for p, value in _shift(_transform(final, pose), origin).items():
                assert inp[p] == 0
                inp[p] = value
        original = inp.copy()
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(inp, original)
        return expected, counts

    # Three different carriers share one receiver; no exact collision numeral
    # or two-shooter ceiling is encoded in the binary receiver-color convention.
    finals = []
    for tip, direction, body, labels in (
        ((12, 7), (0, 1), {(11, 5), (12, 5), (12, 6), (13, 5)},
         {(10, 6): 3, (12, 4): 6}),
        ((12, 17), (0, -1), {(11, 19), (12, 18), (12, 19), (13, 19)},
         {(10, 18): 3, (12, 16): 6}),
        ((7, 12), (1, 0), {(5, 11), (5, 12), (5, 13), (6, 12)},
         {(4, 12): 3, (6, 10): 6}),
    ):
        finals.append({**{p: 1 for p in body},
                       **{p: 2 for p in _arrow(tip, direction)}, **labels})
    expected, counts = exercise(finals, [(2, 2), (20, 2), (20, 20)], [1, 5, 3])
    assert counts == Counter({(12, 12): 3}) and expected[12, 12] == 1

    # Endpoint multiplicity also includes arrows belonging to the same carrier.
    marks = _arrow((10, 8), (-1, 0)) | _arrow((10, 13), (-1, -1))
    body = {(r, c) for r in (9, 12) for c in range(6, 16)} | {(10, 6), (11, 6)}
    final = {**{p: 1 for p in body}, **{p: 2 for p in marks},
             (8, 10): 3, (10, 5): 6}
    expected, counts = exercise([final], [(15, 8)], [7])
    assert counts == Counter({(5, 8): 2}) and expected[5, 8] == 1

    # A shot passes through body pixels and leaves empty intermediate cells blank.
    marks = _arrow((12, 5), (0, 1)) | _arrow((10, 7), (1, 0))
    body = {(r, c) for r in (8, 14) for c in range(3, 9)}
    body |= {(r, 3) for r in range(8, 15)} | {(r, 7) for r in range(11, 15)}
    body.add((12, 6))
    final = {**{p: 1 for p in body}, **{p: 2 for p in marks},
             (7, 5): 3, (11, 2): 6}
    expected, counts = exercise([final], [(2, 16)], [2])
    assert counts == Counter({(12, 10): 1, (15, 7): 1})
    assert expected[12, 7] == 1 and expected[12, 9] == 0
    assert expected[12, 10] == expected[15, 7] == 2


def self_test() -> None:
    _projection_regressions()
    records, puzzles = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        encoded = json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                                  for p in puzzle[s]] for s in puzzle}, separators=(",", ":"))
        puzzles.add(encoded)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                record = _features(pair)
                record["split"] = split
                records.append(record)
                copy = pair["input"].copy()
                assert np.array_equal(solve(copy), pair["output"])
                assert np.array_equal(copy, pair["input"]), "solve modified its input"
    assert len(puzzles) == 42
    assert len({r["shape"] for r in records}) >= 30
    assert {r["background"] for r in records} == set(range(10))
    assert len({r["roles"] for r in records}) >= 20
    assert {r["pieces"] for r in records} >= {1, 2, 3}
    assert {n for r in records for n in r["arrows"]} >= {1, 2, 3}
    assert {d for r in records for d in r["directions"]} == set(DIRECTIONS)
    assert {m for r in records for m in r["moves"]} == {"translate", "rotate", "reflect"}
    assert len({r["sides"] for r in records}) >= 6
    assert len({s for r in records for s in r["body_shapes"]}) >= 100
    assert sum(r["cross_collisions"] > 0 for r in records) >= 42
    assert any(r["overlap"] for r in records)
    # Exact manual vector and multiplicity fixtures, independent of RNG placement.
    cells = {p: 8 for p in _arrow((6, 2), (0, 1)) | _arrow((1, 7), (1, 0))}
    assert _shots(cells, 8) == Counter({(6, 7): 2})
    assert _shots({p: 8 for p in _arrow((5, 6), (-1, -1))}, 8) == Counter({(0, 1): 1})
    p = generate_puzzle(99, 2, 1)
    q = generate_puzzle(99, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Global D4 and palette permutations change neither rule nor role inference.
    colors = np.array([9, 4, 7, 2, 8, 0, 6, 1, 5, 3], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        info = _parse(pair["input"])
        corner_input, corner_output = pair["input"].copy(), pair["output"].copy()
        top, bottom, left, right = info["bounds"]
        ownership = {("top", "left"): "left", ("top", "right"): "top",
                     ("bottom", "right"): "right", ("bottom", "left"): "bottom"}
        for row, column in np.ndindex(corner_input.shape):
            vertical = "top" if row < top else "bottom" if row > bottom else None
            horizontal = "left" if column < left else "right" if column > right else None
            if vertical is not None and horizontal is not None:
                identity = info["sides"][ownership[vertical, horizontal]]
                corner_input[row, column] = corner_output[row, column] = identity
        assert np.array_equal(solve(corner_input), corner_output)
        for t in range(8):
            inp, out = corner_input, corner_output
            if t >= 4:
                inp, out = np.fliplr(inp), np.fliplr(out)
            inp, out = np.rot90(inp, t % 4), np.rot90(out, t % 4)
            assert np.array_equal(solve(colors[inp]), colors[out])
    border_fixture = np.zeros((17, 19), dtype=np.int8)
    border_fixture[0, :], border_fixture[-1, :] = 1, 3
    border_fixture[:, 0], border_fixture[:, -1] = 4, 2
    border_fixture[0, -1], border_fixture[-1, 0] = 1, 3
    structural, sides, bounds = _structure(border_fixture, 0)
    assert bounds == (1, 15, 1, 17)
    assert sides == {"top": 1, "right": 2, "bottom": 3, "left": 4}
    assert np.count_nonzero(structural) == border_fixture.size - 15 * 17
    digest_code = (
        "import hashlib,importlib.util,json,sys; "
        "spec=importlib.util.spec_from_file_location('family',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "puzzle=module.generate_puzzle(0,2,1); "
        "data={split:[{key:value.tolist() for key,value in pair.items()} "
        "for pair in pairs] for split,pairs in puzzle.items()}; "
        "print(hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest())"
    )
    fingerprints = []
    for hash_seed in ("0", "1"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        fingerprints.append(subprocess.check_output(
            [sys.executable, "-c", digest_code, str(Path(__file__).resolve())],
            env=environment, timeout=30, text=True).strip())
    assert len(set(fingerprints)) == 1, "A seed must reproduce across interpreter processes"
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Incorrect output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
