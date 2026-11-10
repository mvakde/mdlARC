"""Live-socket grafting. Standard library + NumPy; all solution evidence is visible.

Socket centers are isolated registration dots, bracketed by three colored rails.
A two-cell diagonal scar activates a socket. Detached parts supply the complete
replacement geometry; their adjacent colored ports are a local D4 compass.
"""
from __future__ import annotations

import itertools
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAGS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(bool(((0 <= g) & (g <= 9)).all()), "invalid grid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _add(a, b, k=1):
    return a[0] + k*b[0], a[1] + k*b[1]


def _at(g, p, default=-1):
    return int(g[p]) if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1] else default


def _transform(cells, t):
    """D4 about the registration origin: reflection first, then quarter turns."""
    result = {}
    for (r, c), color in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[(r, c)] = color
    return result


def _shift(cells, p):
    return {_add(q, p): v for q, v in cells.items()}


def _signature(cells):
    return tuple(sorted(cells.items()))


def _components(points, diagonal=False):
    remaining = set(points)
    result = []
    steps = DIRS + DIAGS if diagonal else DIRS
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], {first}
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _socket_at(g, p, bg):
    """A dot with empty unit neighbors and three differently colored rails.

    Each rail starts two cells away and has >=3 consecutive same-color cells.
    The fourth direction has no such rail. No bounding rectangle is a cue.
    """
    reg = _at(g, p)
    if reg == bg or any(_at(g, _add(p, d)) != bg for d in DIRS):
        return None
    rails = {}
    for d in DIRS:
        color = _at(g, _add(p, d, 2))
        if color not in (-1, bg, reg) and all(
                _at(g, _add(p, d, k)) == color for k in (3, 4)):
            rails[d] = color
    if len(rails) != 3 or len(set(rails.values())) != 3:
        return None
    missing = next(d for d in DIRS if d not in rails)
    axis = next(d for d in rails if (-d[0], -d[1]) in rails)
    # Canonical positive direction, so every bracket is examined once.
    if axis[0] < 0 or axis[1] < 0:
        axis = (-axis[0], -axis[1])
    scars = []
    for sign in (-1, 1):
        anchor_dir = (sign*axis[0], sign*axis[1])
        anchor = _add(p, anchor_dir, 2)
        step = (-anchor_dir[0] + missing[0], -anchor_dir[1] + missing[1])
        spur = (_add(anchor, step), _add(anchor, step, 2))
        if all(_at(g, q) == rails[anchor_dir] for q in spur):
            # Evidence must be an off-axis pair, not part of an intact stroke.
            if all(not any(_at(g, _add(q, d)) not in (-1, bg) for d in DIRS)
                   for q in spur):
                scars.append(spur)
    return {"center": p, "rails": rails, "missing": missing,
            "axis": axis, "scars": scars, "active": bool(scars)}


def _parse(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(int(np.count_nonzero(counts == counts[bg])) == 1, "ambiguous background")
    nonbg = {tuple(map(int, p)) for p in np.argwhere(g != bg)}
    sockets = [s for p in sorted(nonbg) if (s := _socket_at(g, p, bg)) is not None]
    _require(len(sockets) >= 2, "need live and intentional sockets")
    regs = {_at(g, s["center"]) for s in sockets}
    _require(len(regs) == 1, "socket registration colors disagree")
    reg = next(iter(regs))
    centers = {s["center"] for s in sockets}
    sources = []
    for component in _components(nonbg):
        marks = {p for p in component if _at(g, p) == reg}
        if marks and component not in [{p} for p in centers]:
            _require(len(marks) == 1 and len({_at(g, p) for p in component}) >= 3,
                     "malformed connector")
            origin = next(iter(marks))
            cells = {(p[0]-origin[0], p[1]-origin[1]): _at(g, p) for p in component}
            sources.append({"origin": origin, "absolute": component, "cells": cells})
    _require(bool(sources), "no connectors")
    source_cells = set().union(*(s["absolute"] for s in sources))
    external = nonbg - source_cells - centers
    rail_colors = {_at(g, p) for p in external}
    _require(reg not in rail_colors, "extra registration marks")
    bodies = set().union(*(set(s["cells"].values()) for s in sources)) - rail_colors - {reg}
    _require(len(bodies) == 1, "body role not uniquely visible")
    body = next(iter(bodies))
    for source in sources:
        cells = source["cells"]
        labels = {p: v for p, v in cells.items() if v in rail_colors}
        _require(all(d in cells for d in DIRS), "connector must bridge its four unit neighbors")
        _require(len(labels) in (2, 3) and set(labels) <= set(DIRS), "invalid compass ports")
        _require(len(set(labels.values())) == len(labels), "port identities must differ")
        if len(labels) == 2:
            a, b = labels
            _require(a[0]*b[0] + a[1]*b[1] == 0, "two labels must be adjacent")
        _require(sum(v == body for v in cells.values()) >= 3, "connector body is too small")
        _require(all(max(abs(r), abs(c)) <= 3 for r, c in cells), "connector exceeds radius three")
    _require(all(len(s["scars"]) <= 1 for s in sockets), "ambiguous scar")
    active = [s for s in sockets if s["active"]]
    _require(len(active) == len(sources) and len(active) < len(sockets), "socket/source count mismatch")
    # External diagonal scraps have no orthogonal neighbor. Mark only chains;
    # arbitrary isolated points are outside this family's generated domain.
    weak = {p for p in external if not any(_add(p, d) in external for d in DIRS)}
    removed = set()
    for component in _components(weak, diagonal=True):
        if len(component) >= 2:
            ordered = sorted(component)
            _require(len(ordered) in (2, 3)
                     and len({_at(g, p) for p in ordered}) == 1
                     and all(b[0]-a[0] == 1 and abs(b[1]-a[1]) == 1
                             for a, b in zip(ordered, ordered[1:]))
                     and len({b[1]-a[1] for a, b in zip(ordered, ordered[1:])}) == 1,
                     "external scrap must be a straight diagonal of length two or three")
            removed.update(component)
    _require(weak == removed, "unexplained isolated rail cell")
    _require(all(set(s["scars"][0]) <= removed for s in active), "scar cannot be deleted cleanly")
    return {"bg": bg, "reg": reg, "body": body, "sockets": sockets,
            "sources": sources, "source_cells": source_cells,
            "external": external, "removed": removed, "rail_colors": rail_colors}


def _base(g, info):
    base = g.copy()
    for p in info["source_cells"] | info["removed"]:
        base[p] = info["bg"]
    return base


def _choices(g, info, use_compass=True, use_scars=True):
    base = _base(g, info)
    sockets = [s for s in info["sockets"] if s["active"] or not use_scars]
    choices = []
    for source in info["sources"]:
        variants, seen = [], set()
        for t in range(8):
            cells = _transform(source["cells"], t)
            labels = {p: v for p, v in cells.items() if v in info["rail_colors"]}
            for index, socket in enumerate(sockets):
                if use_compass and any(socket["rails"].get(d) != color for d, color in labels.items()):
                    continue
                placed = _shift(cells, socket["center"])
                if any(_at(base, p) == -1 for p in placed):
                    continue
                if any(_at(base, p) not in (info["bg"], color) for p, color in placed.items()):
                    continue
                # Other registration dots may never be swallowed by a graft.
                if any(s["center"] in placed for s in info["sockets"] if s is not socket):
                    continue
                key = index, _signature(placed)
                if key not in seen:
                    seen.add(key)
                    variants.append((index, placed, t))
        choices.append(variants)
    return sockets, choices


def _solutions(g, info, use_compass=True, use_scars=True, limit=2):
    sockets, choices = _choices(g, info, use_compass, use_scars)
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    maps = {}

    def search(depth, used, occupied, assignments):
        if len(maps) >= limit:
            return
        if depth == len(order):
            key = _signature(occupied)
            maps[key] = (dict(occupied), list(assignments))
            return
        index = order[depth]
        for target, cells, t in choices[index]:
            if target in used or occupied.keys() & cells.keys():
                continue
            search(depth+1, used | {target}, {**occupied, **cells},
                   assignments + [(index, sockets[target], cells, t)])
    search(0, set(), {}, [])
    return list(maps.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Determine every output cell from the input, with no examples or RNG."""
    g = _grid(input_grid)
    info = _parse(g)
    solutions = _solutions(g, info)
    _require(len(solutions) == 1, "input does not determine one grafted output")
    out = _base(g, info)
    for p, color in solutions[0][0].items():
        out[p] = color
    return out


def _piece(rng, colors, ports, radius):
    bg, reg, body = colors[:3]
    # Canonical station: N/W/E rails, empty S. The fourth neighbor is body.
    cells = {(0, 0): reg, (1, 0): body}
    for d, color in zip((DIRS[0], DIRS[1], DIRS[3]), ports):
        cells[d] = color
    if rng.random() < .4:
        # Keep N and one adjacent label; the unlabeled opposite endpoint is body.
        cells[(0, int(rng.choice([-1, 1])))] = body
    available = {(r, c) for r in range(-radius, radius+1) for c in range(-radius, radius+1)
                 if not ((c == 0 and r <= -2) or (r == 0 and abs(c) >= 2))}
    target = int(rng.integers(2, min(len(available)-5, 4+radius*3)+1))
    for _ in range(target):
        frontier = sorted({_add(p, d) for p in cells for d in DIRS} & available - cells.keys())
        if not frontier:
            break
        cells[frontier[int(rng.integers(len(frontier)))]] = body
    # Guarantee branch extent rather than only nominal radius variation.
    if radius >= 2 and rng.random() < .65:
        side = int(rng.choice([-1, 1]))
        for c in range(0, side*(radius+1), side):
            cells[(1, c)] = body
    if radius == 3:
        cells[(1, 0)] = cells[(2, 0)] = cells[(3, 0)] = body
    return cells


def _station(rng, colors, ports, final, active, compact=False, near_miss=False):
    reg = colors[1]
    cells = {(0, 0): reg}
    for d, color in zip((DIRS[0], DIRS[1], DIRS[3]), ports):
        length = 3 if compact else int(rng.choice([3, 3, 3, 4, 5]))
        for k in range(2, length+2):
            cells[_add((0, 0), d, k)] = color
        if rng.random() < .4:
            # Side stubs at a remote rail end diversify open/branching networks.
            end = _add((0, 0), d, length+1)
            perpendicular = (d[1], -d[0])
            side = int(rng.choice([-1, 1]))
            for k in range(1, int(rng.integers(2, 4))):
                cells[_add(end, perpendicular, side*k)] = color
    if active:
        side = int(rng.choice([-1, 1]))
        color = ports[1] if side == 1 else ports[2]
        cells[(1, side)] = cells[(2, 0)] = color
    elif near_miss:
        # Same bend, displaced one row: its extrapolated anchor is absent.
        side = int(rng.choice([-1, 1]))
        color = ports[1] if side == 1 else ports[2]
        cells[(2, side)] = cells[(3, 0)] = color
    out = {p: v for p, v in cells.items() if not (
        (active and p in {(1, -1), (1, 1), (2, 0)} and v != reg)
        or (near_miss and p in {(2, -1), (2, 1), (3, 0)} and v != reg))}
    if active:
        out.update(final)
    return cells, out


def _tile(inp, out, tag):
    points = inp.keys() | out.keys()
    r0, c0 = min(r for r, _ in points), min(c for _, c in points)
    return {"input": _shift(inp, (-r0, -c0)), "output": _shift(out, (-r0, -c0)),
            "h": max(r for r, _ in points)-r0+1,
            "w": max(c for _, c in points)-c0+1, "tag": tag}


def _pack(rng, tiles, height, width, boundary=False):
    """Random rectangle packing, with no semantic slots or compartment bands."""
    for _ in range(35):
        blocked = np.zeros((height, width), bool)
        order = sorted(range(len(tiles)), key=lambda i: -(tiles[i]["h"]*tiles[i]["w"] + rng.uniform(0, 25)))
        if boundary:
            station = next(i for i in order if tiles[i]["tag"] == "live")
            order.remove(station)
            order.insert(0, station)
        placements = {}
        for position, i in enumerate(order):
            tile = tiles[i]
            h, w = tile["h"], tile["w"]
            inset = 0 if boundary else 1
            options = [(r, c) for r in range(inset, height-h+1-inset)
                       for c in range(inset, width-w+1-inset)]
            if boundary and position == 0:
                # Touch an edge with actual surviving railwork, not just a scar.
                edge_cells = [p for p in tile["input"] if p in tile["output"]
                              and tile["input"][p] == tile["output"][p]]
                options = [(r, c) for r, c in options if any(
                    r+p[0] in (0, height-1) or c+p[1] in (0, width-1) for p in edge_cells)]
            rng.shuffle(options)
            for r, c in options:
                if blocked[r:r+h, c:c+w].any():
                    continue
                placements[i] = (r, c)
                blocked[max(0, r-1):min(height, r+h+1), max(0, c-1):min(width, c+w+1)] = True
                break
            else:
                break
        if len(placements) == len(tiles):
            return placements
    return None


def _make_pair(rng, palette, count, radius, demonstration=None, boundary=False):
    bg, reg, body = palette[:3]
    identities = palette[3:]
    triples = list(itertools.combinations(identities, 3))
    rng.shuffle(triples)
    tiles = []
    for i in range(count):
        ports = list(triples[i])
        rng.shuffle(ports)
        piece = _piece(rng, palette, ports, radius if i == 0 else int(rng.integers(1, radius+1)))
        pose = int(rng.integers(8))
        final = _transform(piece, pose)
        st_in, st_out = _station(rng, palette, ports, piece, True, compact=count >= 2)
        tiles.append(_tile(_transform(st_in, pose), _transform(st_out, pose), "live"))
        motion = int(rng.integers(1, 4) if demonstration == "rotation" and i == 0
                     else rng.integers(4, 8) if demonstration == "reflection" and i == 0
                     else rng.integers(8))
        source = _transform(final, motion)
        tiles.append(_tile(source, {}, "part"))
        if i == 0:
            # A compass-compatible spare socket makes B necessary for docking.
            sp_in, sp_out = _station(rng, palette, ports, piece, False,
                                     compact=count >= 2, near_miss=True)
            spare_pose = int(rng.integers(8))
            # Include the would-be part footprint in packing bounds without
            # actually drawing it: both ablation alternatives must fit visibly.
            tile = _tile(_transform(sp_in, spare_pose),
                         {**_transform(sp_out, spare_pose), **_transform(piece, spare_pose)}, "spare")
            # Packing union retained; desired spare output is only its rails/dot.
            points = _transform(sp_in, spare_pose).keys() | _transform(piece, spare_pose).keys()
            r0, c0 = min(r for r, _ in points), min(c for _, c in points)
            tile["output"] = _shift(_transform(sp_out, spare_pose), (-r0, -c0))
            tiles.append(tile)
    # Shape variation reflects packing load but does not encode socket identities.
    low = 18 if count == 1 else 25 if count == 2 else 28
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if count == 1 and rng.random() < .35:
        h, w = int(rng.integers(15, 22)), int(rng.integers(26, 31))
        if rng.integers(2):
            h, w = w, h
    placements = _pack(rng, tiles, h, w, boundary)
    if placements is None:
        return None
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    for i, tile in enumerate(tiles):
        for p, v in _shift(tile["input"], placements[i]).items():
            inp[p] = v
        for p, v in _shift(tile["output"], placements[i]).items():
            out[p] = v
    # Free diagonal fragments vary independently of real and near-miss scars.
    occupied = inp != bg
    for _ in range(int(rng.integers(1, 4))):
        length = int(rng.integers(2, 4))
        for _ in range(30):
            start = (int(rng.integers(h)), int(rng.integers(w)))
            direction = DIAGS[int(rng.integers(4))]
            points = [_add(start, direction, k) for k in range(length)]
            if all(_at(inp, p) == bg and _at(out, p) == bg
                   and all(not (0 <= (q := _add(p, d))[0] < h and 0 <= q[1] < w and occupied[q])
                           for d in DIRS + DIAGS) for p in points):
                color = int(rng.choice(identities))
                # Use an already visible rail identity: role parsing is stable.
                visible = set(int(v) for v in inp.ravel())
                if color not in visible:
                    color = int(rng.choice(sorted(visible - {bg, reg, body})))
                for p in points:
                    inp[p] = color
                    occupied[p] = True
                break
    try:
        info = _parse(inp)
        solutions = _solutions(inp, info)
        if len(solutions) != 1 or not np.array_equal(solve(inp), out):
            return None
        if len(_solutions(inp, info, use_compass=False)) < 2:
            return None
        if len(_solutions(inp, info, use_scars=False)) < 2:
            return None
        if demonstration:
            moves = _motion_types(info, solutions[0][1])
            if demonstration not in moves:
                return None
        if demonstration == "rotation":
            # The closest stencil is a spare, so nearest-target docking fails.
            origin = info["sources"][0]["origin"]
            distance = lambda p: abs(p[0]-origin[0]) + abs(p[1]-origin[1])
            live_distance = min(distance(s["center"]) for s in info["sockets"] if s["active"])
            spare_distance = min(distance(s["center"]) for s in info["sockets"] if not s["active"])
            if spare_distance >= live_distance:
                return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _motion_types(info, assignments):
    result = []
    for i, socket, placed, _ in assignments:
        center = socket["center"]
        relative = {(r-center[0], c-center[1]): v for (r, c), v in placed.items()}
        source = info["sources"][i]["cells"]
        ts = [t for t in range(8) if _transform(source, t) == relative]
        result.append("translation" if 0 in ts else "rotation" if any(t < 4 for t in ts) else "reflection")
    return result


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    _require(low <= int(value) <= high, name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)[:8]))
    varied_palette = bool(rng.integers(2))
    shift = str(rng.choice(["count", "boundary", "extent"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for index in range(number):
            local_palette = list(map(int, rng.permutation(10)[:8])) if varied_palette else palette
            if split == "train":
                count = 1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 3))
                radius = 1 if index == 0 else 2
                demonstration = "rotation" if index == 0 else "reflection" if index == 1 else None
                boundary = False
            else:
                count = 3 if shift == "count" else int(rng.integers(1, 3))
                radius = 3 if shift == "extent" else 2
                demonstration = None
                boundary = shift == "boundary"
            for _ in range(150):
                pair = _make_pair(rng, local_palette, count, radius, demonstration, boundary)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded packing/certification failed")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(inp, out):
    inp, out = _grid(inp), _grid(out)
    _require(inp.shape == out.shape, "output dimensions changed")
    info = _parse(inp)
    solutions = _solutions(inp, info)
    _require(len(solutions) == 1 and np.array_equal(solve(inp), out), "incorrect or incomplete grafting")
    _require(len(_solutions(inp, info, use_compass=False)) > 1, "compass has no effect")
    _require(len(_solutions(inp, info, use_scars=False)) > 1, "scar evidence has no effect")
    installed, assignments = solutions[0]
    preserved = info["external"] - info["removed"]
    _require(all(inp[p] == out[p] for p in preserved), "intact rail changed")
    _require(all(out[s["center"]] == info["reg"] for s in info["sockets"]), "registration dot lost")
    _require(all(out[p] == info["bg"] for p in info["source_cells"] - installed.keys()), "old part not erased")
    _require(all(out[p] == installed.get(p, info["bg"]) for p in info["removed"]), "scrap survived grafting")
    for _, socket, cells, _ in assignments:
        center, axis = socket["center"], socket["axis"]
        _require(all(_at(out, _add(center, axis, k)) != info["bg"] for k in range(-2, 3)),
                 "bracketed break not bridged")
    for socket in info["sockets"]:
        if not socket["active"]:
            _require(all(_at(out, _add(socket["center"], d)) == info["bg"] for d in DIRS),
                     "intentional socket closed")
    shapes, geometry, ports = [], [], []
    for source in info["sources"]:
        cells = source["cells"]
        shapes.append((max(r for r, _ in cells)-min(r for r, _ in cells)+1,
                       max(c for _, c in cells)-min(c for _, c in cells)+1))
        geometry.append(min(tuple(sorted(_transform({p: 1 for p in cells}, t))) for t in range(8)))
        ports.append(sum(v in info["rail_colors"] for v in cells.values()))
    return {"shape": inp.shape, "background": info["bg"], "active": len(assignments),
            "spares": len(info["sockets"])-len(assignments), "ports": ports,
            "piece_shapes": shapes, "geometry": geometry,
            "moves": _motion_types(info, assignments), "deleted": len(info["removed"]),
            "overwritten_scar": len(info["removed"] & installed.keys()),
            "axes": tuple(sorted({"h" if s["axis"][0] == 0 else "v" for s in info["sockets"] if s["active"]})),
            "edge": any(p[0] in (0, inp.shape[0]-1) or p[1] in (0, inp.shape[1]-1) for p in preserved)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, t = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, "illegal example counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            record = _inspect_pair(pair["input"], pair["output"])
            _require(1 <= record["active"] <= 3 and record["spares"] >= 1, "invalid family counts")
            if split == "train":
                training.append(record)
    _require(any("rotation" in r["moves"] for r in training), "rotation not demonstrated")
    _require(any("reflection" in r["moves"] for r in training), "reflection not demonstrated")
    _require(any(r["active"] >= 2 for r in training), "joint assignment not demonstrated")


def self_test():
    """Bounded 42-seed semantic/diversity sweep and negative regressions."""
    records, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                snapshot = np.asarray(pair["input"]).copy()
                _require(np.array_equal(solve(snapshot), pair["output"]), "solver mismatch")
                _require(np.array_equal(snapshot, pair["input"]), "solver mutated input")
                records.append(_inspect_pair(pair["input"], pair["output"]))
    puzzle_key = lambda p: tuple((_key(x["input"]), _key(x["output"])) for s in ("train", "test") for x in p[s])
    _require(len({puzzle_key(p) for p in puzzles}) == 42, "seed diversity failed")
    _require(puzzle_key(puzzles[0]) == puzzle_key(generate_puzzle(0)), "nondeterministic puzzle")
    _require(len({r["shape"] for r in records}) >= 35, "canvas diversity failed")
    _require(len({g for r in records for g in r["geometry"]}) >= 40, "body diversity failed")
    _require({r["active"] for r in records} == {1, 2, 3}, "graft count diversity failed")
    _require({n for r in records for n in r["ports"]} == {2, 3}, "port diversity failed")
    _require({r["background"] for r in records} == set(range(10)), "background diversity failed")
    _require({m for r in records for m in r["moves"]} == {"translation", "rotation", "reflection"}, "motion diversity failed")
    _require({a for r in records for a in r["axes"]} == {"h", "v"}, "axis coverage failed")
    _require(any(len(r["axes"]) == 2 for r in records), "mixed axes absent")
    _require(any(r["edge"] for r in records) and any(not r["edge"] for r in records), "boundary diversity failed")
    _require(any(r["overwritten_scar"] for r in records), "scar/graft overlap absent")
    # D4 equivariance on a complete puzzle, including non-square canvases.
    for t in range(8):
        for pair in puzzles[0]["train"] + puzzles[0]["test"]:
            inp, out = pair["input"], pair["output"]
            if t >= 4:
                inp, out = inp[:, ::-1], out[:, ::-1]
            inp, out = np.rot90(inp, t % 4), np.rot90(out, t % 4)
            _require(np.array_equal(solve(inp), out), "D4 rule changed")
    # Validation must reject missing even ONE installation or filling a spare.
    for mutation in ("omit", "spare"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ("train", "test")}
        pair = bad["train"][1]
        info = _parse(pair["input"])
        _, assignments = _solutions(pair["input"], info)[0]
        if mutation == "omit":
            for p in assignments[0][2]:
                pair["output"][p] = _at(_base(pair["input"], info), p)
        else:
            spare = next(s for s in info["sockets"] if not s["active"])
            pair["output"][_add(spare["center"], spare["axis"])] = info["body"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect output accepted: " + mutation)
    for a, b in ((2, 1), (10, 4)):
        generate_puzzle(421, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({r["shape"] for r in records}),
            "body_geometries": len({g for r in records for g in r["geometry"]}),
            "scar_overlaps": sum(bool(r["overwritten_scar"]) for r in records)}


if __name__ == "__main__":
    print(self_test())
