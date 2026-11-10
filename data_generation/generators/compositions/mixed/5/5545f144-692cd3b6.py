"""Tip-led shortest-route painting. Standard library and NumPy only.

The public solver decodes all roles, geometry and waypoint order from the input.
Generation never imports parent code or reads files. No artifacts are written.
"""
from collections import deque
import hashlib

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 730138776, 805600679, 559629041, 87634533, 681166767,
              121234702, 851244963, 941006771, 517421826, 140752143,
              191581366, 947736998, 132791634, 870052401, 468299094,
              923171271, 386373604, 737043415, 188812112, 386639134,
              868172041, 567997371, 331869089, 73099897, 6184261,
              489336681, 698539514, 636713189, 120191370, 28925564,
              638757871, 113665144, 88429536, 736656550, 508773225,
              257340810, 349303557, 850238768, 228276047, 82714123, 292617424)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _turn(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _footprint(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_turn(p, direction - 1) for p in mask)}


def _inside(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "leg terminals must be distinct and aligned")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        root = min(todo)
        todo.remove(root)
        component, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in todo:
                        todo.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _pointed_mask(cells, tip):
    """Read the visible bitmap; there is no recognition lookup by area or seed."""
    matches = []
    for direction in range(4):
        offsets = {_turn((r - tip[0], c - tip[1]), 1 - direction) for r, c in cells}
        if (all(c <= 0 for r, c in offsets)
                and {p for p in offsets if p[1] == 0} == {(0, 0)}
                and any(r != 0 for r, c in offsets)
                and {(-r, c) for r, c in offsets} == offsets):
            matches.append((tuple(sorted(offsets)), direction))
    _require(len(matches) == 1, "mover must have one geometric pointing direction")
    return matches[0]


def _ordered_points(tip, waypoints):
    remaining, points, directions = set(waypoints), [tip], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        _require(len(rays) == 1, "each tip needs exactly one occupied waypoint ray")
        direction, candidates = next(iter(rays.items()))
        target = min(candidates, key=lambda q: abs(q[0]-tip[0]) + abs(q[1]-tip[1]))
        remaining.remove(target)
        points.append(target)
        directions.append(direction)
        tip = target
    return points, directions


def _decode(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 4, "input needs background, wall, body and tip colors")
    bg = int(colors[np.argmax(counts)])
    _require(int(max(counts)) > a.size // 2, "background must occupy a majority")
    single = [int(c) for c, n in zip(colors, counts) if n == 1]
    _require(len(single) == 1, "exactly the accent tip color must occur once")
    trace = single[0]
    tip = tuple(map(int, np.argwhere(a == trace)[0]))
    candidates = []
    for ink in (int(c) for c in colors if c not in (bg, trace)):
        cells = {tuple(map(int, q)) for q in np.argwhere((a == ink) | (a == trace))}
        components = _components(cells)
        mover = next(c for c in components if tip in c)
        others = [c for c in components if c is not mover]
        if len(mover) < 4 or len(others) < 2 or any(len(c) != 1 for c in others):
            continue
        try:
            mask, initial = _pointed_mask(mover, tip)
            points, directions = _ordered_points(tip, {next(iter(c)) for c in others})
            if initial != directions[0]:
                continue
            wall = next(int(c) for c in colors if c not in (bg, trace, ink))
            candidates.append(dict(bg=bg, ink=ink, trace=trace, wall=wall,
                                   mask=mask, initial=initial, points=points,
                                   directions=directions, shape=a.shape))
        except AssertionError:
            continue
    _require(len(candidates) == 1, "input roles or waypoint order are ambiguous")
    info = candidates[0]
    _require(int(np.count_nonzero(a == info["wall"])) >= 2, "need visible obstacles")
    for i, p in enumerate(info["points"]):
        d = info["initial"] if i == 0 else info["directions"][i-1]
        cells = _footprint(info["mask"], p, d)
        _require(_inside(cells, a.shape), "arrival footprint outside canvas")
        _require(all(a[q] != info["wall"] for q in cells), "arrival intersects a wall")
        _require(all(max(abs(q[0]-r), abs(q[1]-c)) > 1
                     for q in info["points"][i+1:] for r, c in cells),
                 "future waypoint touches an arrival footprint")
    return info


def _distances(blocked, start):
    h, w = blocked.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    _require(not blocked[start], "blocked terminal")
    distance[start] = 0
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        nd = int(distance[r, c]) + 1
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if (0 <= q[0] < h and 0 <= q[1] < w
                    and not blocked[q] and distance[q] < 0):
                distance[q] = nd
                queue.append(q)
    return distance


def _route_union(blocked, start, target):
    left, right = _distances(blocked, start), _distances(blocked, target)
    length = int(left[target])
    _require(length >= 0, "unreachable waypoint")
    union = (left >= 0) & (right >= 0) & (left + right == length)
    return union, length, right


def _single_path(right, start, target):
    """Deliberately wrong single-route alternative, used only for ablation checks."""
    path = np.zeros(right.shape, dtype=bool)
    q = start
    path[q] = True
    while q != target:
        q = next((q[0]+dr, q[1]+dc) for dr, dc in DIRS
                 if 0 <= q[0]+dr < right.shape[0] and 0 <= q[1]+dc < right.shape[1]
                 and right[q[0]+dr, q[1]+dc] == right[q] - 1)
        path[q] = True
    return path


def _draw(a, mask, tip, direction, ink, trace):
    for q in _footprint(mask, tip, direction):
        a[q] = ink
    a[tip] = trace


def _execute(a, info, mode="all"):
    """Paint current background, clear the old mask, rotate and place.

    Only walls block the point-route calculation. The rigid mask is placed at
    terminals; it does not sweep along one arbitrarily selected detour. Trace
    underneath a departing mask is erased, rather than restored from a layer.
    Other modes are deliberately wrong rules used for causal validation only.
    """
    board = a.copy()
    blocked = (a == info["wall"]) if mode != "no_walls" else np.zeros(a.shape, dtype=bool)
    direct = (_route_union(blocked, info["points"][0], info["points"][-1])[0]
              if mode == "direct" else None)
    union_total = np.zeros(a.shape, dtype=bool)
    legs = []
    p = info["points"][0]
    facing = info["initial"]
    for target, direction in zip(info["points"][1:], info["directions"]):
        union, length, right = _route_union(blocked, p, target)
        selected = _single_path(right, p, target) if mode == "single" else union
        if mode == "direct":
            # Grant correct mover placements and waypoint consumption, but
            # omit the leg-defined terminal dependency for the painted field.
            selected = direct
        board[selected & (board == info["bg"])] = info["trace"]
        for q in _footprint(info["mask"], p, facing):
            board[q] = info["bg"]
        _draw(board, info["mask"], target, direction, info["ink"], info["trace"])
        union_total |= union
        legs.append(dict(length=length, manhattan=abs(p[0]-target[0])+abs(p[1]-target[1]),
                         cells=int(union.sum())))
        p, facing = target, direction
    return board, legs, union_total


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the supplied input. Does not mutate it or access generation."""
    a = _grid(input_grid)
    return _execute(a, _decode(a))[0]


def _arrow(span, extension, tails):
    cells = {(0, -c) for c in range(extension)}
    for depth in range(1, span + 1):
        cells.update((r, -extension + 1 - depth) for r in range(-depth, depth+1))
    for j in range(1, tails + 1):
        cells.update((r, -extension + 1 - span - j) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(_arrow(span, extension, tails)
              for span in (1, 2) for extension in (1, 2) for tails in (0, 1))


def _extend_options(points, footprints, mask, shape):
    """Reject new junctions at every earlier terminal, not merely the newest one."""
    options = []
    p = points[-1]
    for d, (dr, dc) in enumerate(DIRS):
        initial = _footprint(mask, p, d) if len(points) == 1 else None
        if initial is not None and not _inside(initial, shape):
            continue
        protected = footprints if initial is None else [initial]
        for length in range(9 if len(points) == 1 else 4, max(shape)):
            q = p[0] + dr*length, p[1] + dc*length
            cells = _footprint(mask, q, d)
            if not _inside(cells, shape):
                continue
            if any(max(abs(q[0]-r), abs(q[1]-c)) <= 1 for f in protected for r, c in f):
                continue
            if any(q[0] == old[0] or q[1] == old[1] for old in points[:-1]):
                # Repeated coordinates are allowed only farther along the same
                # ray as that earlier terminal's already chosen next waypoint.
                valid = True
                for old, nxt in zip(points[:-1], points[1:]):
                    if q[0] == old[0] or q[1] == old[1]:
                        cross = (q[0]-old[0])*(nxt[1]-old[1]) - (q[1]-old[1])*(nxt[0]-old[0])
                        dot = (q[0]-nxt[0])*(nxt[0]-old[0]) + (q[1]-nxt[1])*(nxt[1]-old[1])
                        if cross != 0 or dot <= 0:
                            valid = False
                            break
                if not valid:
                    continue
            options.append((d, q, cells, initial))
    return options


def _random_route(rng, mask, shape, count, need_rotation=False):
    for _ in range(45):
        p = (int(rng.integers(2, shape[0]-2)), int(rng.integers(2, shape[1]-2)))
        points, footprints, directions = [p], [], []
        while len(directions) < count:
            options = _extend_options(points, footprints, mask, shape)
            if not options:
                break
            ds = sorted({v[0] for v in options})
            # Draw direction and distance separately; no lattice or route bank.
            weights = np.array([0.35 if directions and d == directions[-1] else 1.0 for d in ds])
            d = int(rng.choice(ds, p=weights/weights.sum()))
            choices = [v for v in options if v[0] == d]
            d, q, cells, initial = choices[int(rng.integers(len(choices)))]
            if initial is not None:
                footprints.append(initial)
            footprints.append(cells)
            points.append(q)
            directions.append(d)
        if len(directions) == count and (not need_rotation or directions[0] != directions[-1]):
            return points, directions, footprints
    return None


def _primary_obstacle(rng, points, directions, shape, protected, tied):
    p, target, d = points[0], points[1], directions[0]
    dr, dc = DIRS[d]
    nr, nc = -dc, dr
    length = abs(p[0]-target[0]) + abs(p[1]-target[1])
    choices = []
    for along in range(3, length-2):
        for radius in (1, 2):
            for thickness in (1, 2):
                for sign in (-1, 1):
                    low, high = -radius, radius if tied else radius+1
                    wall = {(p[0]+dr*(along+j)+nr*k*sign,
                             p[1]+dc*(along+j)+nc*k*sign)
                            for j in range(thickness) for k in range(low, high+1)}
                    if _inside(wall, shape) and wall.isdisjoint(protected):
                        # Both bypasses have at least one free side cell.
                        rim = {(r+nr, c+nc) for r, c in wall} | {(r-nr, c-nc) for r, c in wall}
                        if _inside(rim, shape):
                            choices.append(wall)
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _make_pair(rng, mask, count, tied, large, palette, need_rotation=False):
    bg, wall, ink, trace = palette
    for _ in range(70):
        shape = tuple(int(rng.integers(23 if large else 16, 31 if large else 26)) for _ in range(2))
        if rng.random() < 0.13:
            shape = (shape[0], 30)
        grown = _random_route(rng, mask, shape, count, need_rotation)
        if grown is None:
            continue
        points, directions, footprints = grown
        protected = set().union(*footprints)
        walls = _primary_obstacle(rng, points, directions, shape, protected, tied)
        if walls is None:
            continue
        # Additional bars, blocks or elbows share the same obstacle semantics.
        for _ in range(int(rng.integers(0, 5))):
            r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            kind = int(rng.integers(3))
            cells = {(r+i, c+j) for i in range(h) for j in range(w)
                     if kind == 0 or (kind == 1 and (i == 0 or j == 0))
                     or (kind == 2 and (i in (0, h-1) or j == 0))}
            if _inside(cells, shape) and cells.isdisjoint(protected):
                walls |= cells
        if len(walls) < 2:
            continue
        a = np.full(shape, bg, dtype=np.int8)
        for q in walls:
            a[q] = wall
        for q in points[1:]:
            a[q] = ink
        _draw(a, mask, points[0], directions[0], ink, trace)
        info = dict(bg=bg, wall=wall, ink=ink, trace=trace, mask=mask, points=points,
                    directions=directions, initial=directions[0], shape=shape)
        try:
            out, legs, unions = _execute(a, info)
            first, length, _ = _route_union(a == wall, points[0], points[1])
            # Primary bar must remain causally active, not turn into scenery.
            if legs[0]["length"] <= legs[0]["manhattan"]:
                continue
            # Test equal bypasses vs the rejected longer bypass geometrically.
            dr, dc = DIRS[directions[0]]
            nr, nc = -dc, dr
            side = [(r-points[0][0])*nr + (c-points[0][1])*nc for r, c in zip(*np.where(first))]
            both_sides = min(side) < 0 < max(side)
            if tied != both_sides:
                continue
            if tied and int(first.sum()) <= length + 1:
                continue
            direct = _route_union(a == wall, points[0], points[-1])[0]
            if np.array_equal(unions, direct):
                continue
            if np.array_equal(out, _execute(a, info, "direct")[0]):
                continue
            if np.array_equal(out, _execute(a, info, "single")[0]):
                continue
            if np.array_equal(out, _execute(a, info, "no_walls")[0]):
                continue
            decoded = _decode(a)
            _require(decoded["points"] == points and decoded["mask"] == mask, "input construction lost identity")
            _require(np.array_equal(solve(a), out), "construction and decoder disagree")
        except AssertionError:
            continue
        return {"input": a, "output": out}
    raise RuntimeError("bounded route/obstacle construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return distinct examples; first two teach ties, obstacle asymmetry and motion."""
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    first_mask = int(rng.integers(len(MASKS)))
    first_palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    for split, n in (("train", nt), ("test", ne)):
        for i in range(n):
            mi = first_mask if split == "train" and i == 0 else int(rng.integers(len(MASKS)))
            if split == "train" and i == 1 and mi == first_mask:
                mi = (mi + 1 + int(rng.integers(len(MASKS)-1))) % len(MASKS)
            palette = first_palette if split == "train" and i == 0 else tuple(map(int, rng.choice(10, 4, replace=False)))
            if split == "train" and i == 1 and palette == first_palette:
                palette = palette[1:] + palette[:1]
            count = (2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 5))) if split == "train" else int(rng.integers(5, 8))
            tied = (i == 0) if split == "train" and i < 2 else bool(rng.integers(2))
            for _ in range(12):
                pair = _make_pair(rng, MASKS[mi], count, tied, split == "test", palette,
                                  need_rotation=(split == "train" and i == 0))
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
    a, out = _grid(pair["input"]), _grid(pair["output"])
    _require(a.shape == out.shape, "canvas changed")
    info = _decode(a)
    expected, legs, unions = _execute(a, info)
    _require(np.array_equal(expected, out), "incorrect per-leg simulation")
    _require(np.array_equal(a == info["wall"], out == info["wall"]), "walls changed")
    _require(int(np.count_nonzero(out == info["ink"])) == len(info["mask"])-1,
             "body geometry or waypoint consumption failed")
    final = _footprint(info["mask"], info["points"][-1], info["directions"][-1])
    _require({tuple(map(int, q)) for q in np.argwhere(out == info["ink"])} == final - {info["points"][-1]},
             "terminal mask is not the rotated input mask")
    _require(not np.array_equal(a, out), "identity pair")
    _require(any(v["length"] > v["manhattan"] for v in legs), "obstacles never force detours")
    _require(not np.array_equal(out, _execute(a, info, "single")[0]), "one route suffices")
    _require(not np.array_equal(out, _execute(a, info, "no_walls")[0]), "obstacles do not affect output")
    direct = _route_union(a == info["wall"], info["points"][0], info["points"][-1])[0]
    _require(not np.array_equal(unions, direct), "start-to-end shortcut matches leg unions")
    _require(not np.array_equal(out, _execute(a, info, "direct")[0]),
             "omitting leg-defined terminals leaves the same rendered answer")
    info["legs"] = legs
    return info


def validate_generated_puzzle(puzzle):
    """Check grids, recoverability, actual rigid-mask motion and both causal ablations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            rows[split].append(info)
    train, test = rows["train"], rows["test"]
    _require(len({a["mask"] for a in train}) >= 2, "training needs two actual masks")
    _require(len({tuple(a[c] for c in ("bg", "wall", "ink", "trace")) for a in train}) >= 2,
             "training needs visible role recoloring")
    _require(len({len(a["legs"]) for a in train}) >= 2, "training needs repeated-leg evidence")
    _require(any(a["initial"] != a["directions"][-1] for a in train), "training never shows rotation")
    _require(min(len(a["legs"]) for a in test) > max(len(a["legs"]) for a in train),
             "test must extend demonstrated route lengths")
    kinds = set()
    for a, pair in zip(train, puzzle["train"]):
        union = _route_union(np.asarray(pair["input"]) == a["wall"], a["points"][0], a["points"][1])[0]
        dr, dc = DIRS[a["directions"][0]]
        nr, nc = -dc, dr
        side = [(r-a["points"][0][0])*nr + (c-a["points"][0][1])*nc for r, c in zip(*np.where(union))]
        kinds.add("both" if min(side) < 0 < max(side) else "one")
    _require(kinds == {"both", "one"}, "training needs equal and unequal obstacle bypasses")


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed audit; retain hashes and small feature sets, never puzzles."""
    signatures, geometries, shapes, routes, palettes = (set() for _ in range(5))
    directions, finals, counts, wall_counts, lengths = (set() for _ in range(5))
    mixed = edge_routes = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_hash(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                _, legs, unions = _execute(pair["input"], info)
                geometries.add(info["mask"])
                shapes.add(info["shape"])
                p0 = info["points"][0]
                # Remove translation, rotations and reflections for route diversity.
                variants = []
                for sign in (-1, 1):
                    for turn in range(4):
                        variants.append(tuple(_turn((r-p0[0], sign*(c-p0[1])), turn) for r, c in info["points"]))
                routes.add(min(variants))
                palettes.add(tuple(info[c] for c in ("bg", "wall", "ink", "trace")))
                directions.add(info["initial"])
                finals.add(info["directions"][-1])
                counts.add(len(legs))
                wall_counts.add(int(np.count_nonzero(pair["input"] == info["wall"])))
                lengths.update(v["length"] for v in legs)
                turns = {(b-a) % 4 for a, b in zip(info["directions"], info["directions"][1:])}
                mixed += int(1 in turns and 3 in turns)
                edge_routes += int(np.any(unions[0]) or np.any(unions[-1]) or np.any(unions[:, 0]) or np.any(unions[:, -1]))
    _require(len(signatures) == 42, "puzzles repeated across seeds")
    _require(len(geometries) == len(MASKS), "mask variation missing")
    _require(len(shapes) >= 50 and len(routes) >= 150, "geometric diversity too small")
    _require(len(palettes) >= 100 and len(wall_counts) >= 15, "role/obstacle diversity too small")
    _require(directions == finals == set(range(4)), "direction coverage missing")
    _require(counts == set(range(2, 8)) and len(lengths) >= 20, "leg variation missing")
    _require(mixed >= 10, "mixed-turn routes missing")
    _require(edge_routes >= 1, "boundary route coverage missing")
    _require(_puzzle_hash(generate_puzzle(73, 3, 2)) == _puzzle_hash(generate_puzzle(73, 3, 2)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(19+a*7+b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-check path semantics: centered bar has two optimal bypasses;
    # asymmetric bar excludes the strictly longer side.
    blocked = np.zeros((9, 13), dtype=bool)
    blocked[3:6, 6] = True
    union, length, right = _route_union(blocked, (4, 2), (4, 10))
    _require(length == 12 and union[2, 6] and union[6, 6] and not union[1, 6], "equal detours failed")
    _require(int(union.sum()) > int(_single_path(right, (4, 2), (4, 10)).sum()), "ties lost")
    blocked[6, 6] = True
    union, length, _ = _route_union(blocked, (4, 2), (4, 10))
    _require(length == 12 and union[2, 6] and not union[7, 6], "longer bypass retained")
    # A hand-constructed complete two-leg result checks the sequencing as well
    # as distances: painting precedes clearing, and the second leg turns north.
    fixture = np.zeros((9, 14), dtype=np.int8)
    fixture[5:8, 6] = 8
    fixture[5:8, 1] = 3
    fixture[6, 2] = 2
    fixture[6, 11] = fixture[2, 11] = 3
    expected = np.zeros((9, 14), dtype=np.int8)
    expected[4:9, 2:12] = 2
    expected[5:8, 6] = 8
    expected[6, 2] = expected[6, 11] = 0
    expected[5:8, 10] = 0
    expected[2, 11] = 2
    expected[3, 10:13] = 3
    _require(np.array_equal(solve(fixture), expected), "per-leg painting/clearing/rotation order failed")
    try:
        _ordered_points((5, 5), {(5, 9), (9, 5)})
    except AssertionError:
        pass
    else:
        raise AssertionError("junction accepted")
    puzzle = generate_puzzle(0, 2, 1)
    source = puzzle["test"][0]["input"].copy()
    _require(np.array_equal(solve(source), puzzle["test"][0]["output"]), "array input unsupported")
    _require(np.array_equal(solve(source.tolist()), puzzle["test"][0]["output"]), "list input unsupported")
    _require(np.array_equal(source, puzzle["test"][0]["input"]), "solver mutated input")
    for mode in ("single", "direct", "no_walls"):
        puzzle["test"][0]["output"] = _execute(source, _decode(source), mode)[0]
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"{mode} corruption accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), masks=len(geometries),
                grid_shapes=len(shapes), normalized_routes=len(routes), palettes=len(palettes),
                obstacle_cell_counts=len(wall_counts), leg_counts=sorted(counts),
                shortest_lengths=len(lengths), mixed_turn_pairs=mixed, boundary_union_pairs=edge_routes)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
