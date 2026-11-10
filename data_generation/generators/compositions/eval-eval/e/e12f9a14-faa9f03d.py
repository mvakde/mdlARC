"""Beam grafting: incomplete-frame sources splice into damaged terminal wires.

Only NumPy and the standard library are used. No parent code, stored puzzle,
seed-dependent solving convention, filesystem write, or execution trace is used.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PORTS = {
    "N0": ((-1, 0), (-1, 0)), "N1": ((-1, 1), (-1, 0)),
    "E0": ((0, 2), (0, 1)), "E1": ((1, 2), (0, 1)),
    "S0": ((2, 0), (1, 0)), "S1": ((2, 1), (1, 0)),
    "W0": ((0, -1), (0, -1)), "W1": ((1, -1), (0, -1)),
}
RING = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
             if r in (-1, 2) or c in (-1, 2))
SIDE_OFFSETS = {offset: d for offset, d in PORTS.values()}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _ray(p, d, shape):
    result = []
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
    return [_add(a, (i*d[0], i*d[1]))
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1)]


def _bends(walk):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _footprint(core, shape):
    r, c = core
    return {(i, j) for i in range(r-1, r+3) for j in range(c-1, c+3)
            if _inside((i, j), shape)}


def _parse_frames(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > a.size, "background must be a strict majority")
    frames = []
    # Wires have no solid 2x2; this signature identifies payload cores.
    mask = ((a[:-1, :-1] == a[1:, :-1]) & (a[:-1, :-1] == a[:-1, 1:])
            & (a[:-1, :-1] == a[1:, 1:]) & (a[:-1, :-1] != bg))
    for raw in np.argwhere(mask):
        core = tuple(map(int, raw))
        color = int(a[core])
        ring = [(off, _add(core, off)) for off in RING if _inside(_add(core, off), a.shape)]
        nonbg = [int(a[p]) for _, p in ring if a[p] != bg]
        if len(nonbg) < 3 or len(set(nonbg)) != 1 or nonbg[0] == color:
            continue
        ports = [(p, SIDE_OFFSETS[off]) for off, p in ring
                 if a[p] == bg and off in SIDE_OFFSETS]
        if not ports or any(a[p] == bg and off not in SIDE_OFFSETS for off, p in ring):
            continue
        frames.append({"core": core, "color": color, "structure": nonbg[0],
                       "ports": ports, "footprint": _footprint(core, a.shape)})
    _require(bool(frames), "no incomplete-frame source found")
    structures = {f["structure"] for f in frames}
    _require(len(structures) == 1, "frame structure colors disagree")
    structure = next(iter(structures))
    used = set()
    for f in frames:
        _require(not used & f["footprint"], "overlapping frames")
        used.update(f["footprint"])
    return bg, structure, frames, used


def _reconstruct(a, bg, structure, frames, excluded):
    payloads = {f["color"] for f in frames}
    cells = {(r, c): int(a[r, c]) for r in range(a.shape[0]) for c in range(a.shape[1])
             if (r, c) not in excluded and a[r, c] not in (bg, structure)}
    _require(cells and not (set(cells.values()) & payloads), "wire/source palettes overlap")
    markers = {tuple(map(int, p)) for p in np.argwhere(a == structure)
               if tuple(map(int, p)) not in excluded}
    pending = set(markers)
    for _ in range(len(markers) + 1):
        changed = False
        for p in sorted(pending):
            colors = {cells[q] for q in _neighbors(p, cells)}
            _require(len(colors) <= 1, "ambiguous repair bead")
            if colors:
                cells[p] = next(iter(colors))
                pending.remove(p)
                changed = True
        if not changed:
            break
    _require(not pending, "unwitnessed repair beads")
    before_gaps = dict(cells)
    proposals = {}
    for p, color in sorted(before_gaps.items()):
        for d in ((1, 0), (0, 1)):
            for distance in (2, 3):
                q = _add(p, (distance*d[0], distance*d[1]))
                if before_gaps.get(q) != color:
                    continue
                between = _line(p, q)[1:-1]
                if not all(x not in excluded and a[x] == bg for x in between):
                    continue
                outer_a = _add(p, (-d[0], -d[1]))
                outer_b = _add(q, d)
                if before_gaps.get(outer_a) != color and before_gaps.get(outer_b) != color:
                    continue
                for x in between:
                    proposals.setdefault(x, set()).add(color)
    for p, colors in proposals.items():
        _require(len(colors) == 1, "competing gap completions")
        cells[p] = next(iter(colors))
    # Components have no crossings, junctions, or touching colors.
    remaining, components = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        comp, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                comp.add(q)
                stack.append(q)
        _require(len({cells[p] for p in comp}) == 1, "wire components touch")
        ends = [p for p in comp if len(_neighbors(p, comp)) == 1]
        _require(len(comp) >= 4 and len(ends) == 2
                 and all(1 <= len(_neighbors(p, comp)) <= 2 for p in comp), "wire is not a simple path")
        edge = {p for p in comp if _boundary(p, a.shape)}
        _require(len(edge) == 1 and next(iter(edge)) in ends, "wire needs one boundary endpoint")
        components.append(comp)
    return cells, components, markers, set(proposals)


def _walk_arm(pivot, start, comp):
    walk, previous, current = [pivot], pivot, start
    for _ in range(len(comp)):
        walk.append(current)
        nxt = [p for p in _neighbors(current, comp) if p != previous]
        if not nxt:
            return walk
        _require(len(nxt) == 1, "branch is not uniquely traceable")
        previous, current = current, nxt[0]
    raise AssertionError("wire contains a cycle")


def _analyze(a):
    bg, structure, frames, excluded = _parse_frames(a)
    cells, components, markers, gaps = _reconstruct(a, bg, structure, frames, excluded)
    owner = {p: i for i, comp in enumerate(components) for p in comp}
    beams, hits, hit_components, occupied = [], [], set(), set()
    for f in frames:
        for port, d in f["ports"]:
            beam, contact = [], None
            for p in _ray(port, d, a.shape):
                _require(p == port or p not in excluded, "beam encounters a frame")
                beam.append(p)
                if p in cells:
                    contact = p
                    break
            _require(not occupied & set(beam), "beams intersect")
            occupied.update(beam)
            record = {"cells": beam, "port": port, "d": d, "color": f["color"], "contact": contact}
            beams.append(record)
            if contact is None:
                continue
            index = owner[contact]
            _require(index not in hit_components, "two beams contact one wire")
            hit_components.add(index)
            comp = components[index]
            ns = _neighbors(contact, comp)
            forward = _add(contact, d)
            _require(len(ns) == 2 and forward in ns, "contact has no forward tail")
            other = next(p for p in ns if p != forward)
            _require((other[0]-contact[0])*d[0] + (other[1]-contact[1])*d[1] == 0,
                     "first contact must be a corner")
            old_walk = _walk_arm(contact, forward, comp)
            kept_walk = _walk_arm(contact, other, comp)
            _require(_boundary(old_walk[-1], a.shape) and not _boundary(kept_walk[-1], a.shape),
                     "contact does not select the boundary tail")
            hits.append({**record, "component": index, "old": set(old_walk[1:]),
                         "kept": set(kept_walk), "old_walk": old_walk, "kept_walk": kept_walk})
    _require(bool(hits), "scene has no graft")
    return {"shape": a.shape, "bg": bg, "structure": structure, "frames": frames,
            "excluded": excluded, "wires": cells, "components": components,
            "markers": markers, "gaps": gaps, "beams": beams, "hits": hits}


def _render(a, info, *, emit=True, surgery=True, erase=True, recolor=True, stop=True, global_color=False):
    out = a.copy()
    for p, color in info["wires"].items():
        out[p] = color
    if not emit:
        return out
    if surgery:
        for h in info["hits"]:
            if erase:
                for p in h["old"]:
                    out[p] = info["bg"]
            if recolor:
                for p in h["kept"]:
                    out[p] = h["color"]
        if global_color:
            touched = {h["component"] for h in info["hits"]}
            for i, comp in enumerate(info["components"]):
                if i not in touched:
                    for p in comp:
                        out[p] = info["hits"][0]["color"]
    for b in info["beams"]:
        points = b["cells"] if stop and surgery else _ray(b["port"], b["d"], a.shape)
        for p in points:
            if p not in info["excluded"] or p == b["port"]:
                out[p] = b["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role and edit from this input alone; do not mutate it."""
    a = _array(input_grid)
    return _render(a, _analyze(a))


def _valid_walk(walk):
    s = set(walk)
    return len(s) == len(walk) and len(walk) >= 2 and all(
        len(_neighbors(p, s)) == (1 if p in (walk[0], walk[-1]) else 2) for p in s)


def _branch(rng, start, direction, turns, shape, boundary):
    walk, point, d = [start], start, direction
    for segment in range(turns + 1):
        if boundary and segment == turns:
            rest = _ray(point, d, shape)
            if len(rest) < 3:
                return None
            end = rest[-1]
        else:
            options = [_add(point, (n*d[0], n*d[1])) for n in range(2, 10)]
            options = [p for p in options if _inside(p, shape) and not _boundary(p, shape)]
            if not options:
                return None
            end = options[int(rng.integers(len(options)))]
        walk.extend(_line(point, end)[1:])
        if not _valid_walk(walk):
            return None
        point = end
        sign = 1 if rng.integers(2) else -1
        d = (sign*d[1], sign*d[0])
    return walk


def _compatible(cells, occupied, footprints):
    return not (cells & footprints) and not any(_neighbors(p, footprints) for p in cells) and not (
        cells & occupied or any(_neighbors(p, occupied) for p in cells))


def _paint_scene(shape, bg, structure, sources, wires):
    a = np.full(shape, bg, dtype=np.int8)
    for f in sources:
        core = f["core"]
        for off in RING:
            p = _add(core, off)
            if _inside(p, shape) and off not in {PORTS[n][0] for n in f["names"]}:
                a[p] = structure
        r, c = core
        a[r:r+2, c:c+2] = f["color"]
    for p, color in wires.items():
        a[p] = color
    return a


def _constructed_output(a, bg, wires, plans, misses):
    """Labels from construction paths, before input reconstruction is consulted."""
    out = a.copy()
    for p, color in wires.items():
        out[p] = color
    for plan in plans:
        for p in plan["old"][1:]:
            out[p] = bg
        for p in plan["kept"]:
            out[p] = plan["color"]
    for plan in plans + misses:
        for p in plan["beam"]:
            out[p] = plan["color"]
    return out


def _damage(rng, full, wires, plans, complex_witness):
    bg, structure, frames, excluded = _parse_frames(full)
    a = full.copy()
    forced = set()
    if complex_witness:
        forced.add(plans[0]["kept"][0])
        forced.add(plans[0]["kept"][-1])
        old = plans[0]["old"]
        forced.update(p for i, p in enumerate(old[1:-1], 1)
                      if old[i-1][0] != old[i+1][0] and old[i-1][1] != old[i+1][1])
        hit_cells = set().union(*(set(p["old"]) | set(p["kept"]) for p in plans))
        unhit = sorted(set(wires) - hit_cells)
        if unhit:
            forced.add(unhit[int(rng.integers(len(unhit)))])
    for p in forced:
        a[p] = structure
    candidates = sorted(set(wires) - forced)
    rng.shuffle(candidates)
    for p in candidates[:int(rng.integers(1, 7))]:
        a[p] = structure
    recovered = _reconstruct(a, bg, structure, frames, excluded)[0]
    if recovered != wires:
        return None
    # Try a few sparse omissions. Each accepted change must preserve the
    # original complete graph under the public, input-only repair convention.
    gap_count = int(rng.integers(1 if complex_witness else 0, 4))
    successes = 0
    rng.shuffle(candidates)
    for p in candidates:
        if successes >= gap_count:
            break
        if a[p] == structure:
            continue
        ns = _neighbors(p, wires)
        if len(ns) != 2 or ns[0][0]+ns[1][0] != 2*p[0] or ns[0][1]+ns[1][1] != 2*p[1]:
            continue
        previous = int(a[p])
        a[p] = bg
        try:
            correct = _reconstruct(a, bg, structure, frames, excluded)[0] == wires
        except AssertionError:
            correct = False
        if correct:
            successes += 1
        else:
            a[p] = previous
    if complex_witness and not successes:
        return None
    return a


def _attempt_scene(rng, palette, directions, required, mode, desired_hits):
    complex_witness = mode == "bent"
    low = 22 if desired_hits > 1 else (18 if complex_witness else 13)
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    bg, structure = palette[:2]
    source_count = 1 if desired_hits == 1 else int(rng.integers(1, min(2, desired_hits) + 1))
    sources, footprints = [], set()
    for i in range(source_count):
        found = False
        for _ in range(40):
            core = tuple(map(int, (rng.integers(shape[0]-1), rng.integers(shape[1]-1))))
            if rng.random() < .25:
                axis = int(rng.integers(2))
                core = tuple((int(rng.choice((0, shape[j]-2))) if j == axis else core[j]) for j in range(2))
            fp = _footprint(core, shape)
            if fp & footprints or any(_neighbors(p, footprints) for p in fp):
                continue
            sources.append({"core": core, "color": palette[2+i], "names": []})
            footprints.update(fp)
            found = True
            break
        if not found:
            return None
    wire_palette = palette[2+source_count:4+source_count]
    wires, plans, beam_cells = {}, [], set()
    for j in range(desired_hits):
        found = False
        for _ in range(100):
            index = j if j < source_count else int(rng.integers(source_count))
            source = sources[index]
            allowed = [n for n, (off, d) in PORTS.items() if n not in source["names"]
                       and d in (directions if j or required is None else (required,))
                       and _inside(_add(source["core"], off), shape)]
            if not allowed:
                continue
            name = allowed[int(rng.integers(len(allowed)))]
            off, d = PORTS[name]
            port = _add(source["core"], off)
            ray = _ray(port, d, shape)
            if len(ray) < 8:
                continue
            pivot = ray[int(rng.integers(3, len(ray)-3))]
            old_turns = int(rng.integers(1, 4)) if complex_witness and j == 0 else int(rng.integers(4))
            kept_turns = int(rng.integers(1, 3)) if complex_witness and j == 0 else int(rng.integers(4))
            if mode == "simple":
                old_turns = kept_turns = 0
            side = (-d[1], d[0])
            if rng.integers(2):
                side = (-side[0], -side[1])
            old = _branch(rng, pivot, d, old_turns, shape, True)
            kept = _branch(rng, pivot, side, kept_turns, shape, False)
            if old is None or kept is None:
                continue
            whole = list(reversed(old)) + kept[1:]
            comp = set(whole)
            beam = ray[:ray.index(pivot)+1]
            if not _valid_walk(whole) or not _compatible(comp, set(wires), footprints):
                continue
            if comp & beam_cells or set(beam) & (set(wires) | beam_cells) or set(beam[:-1]) & comp:
                continue
            if any(p in footprints for p in beam[1:]):
                continue
            color = wire_palette[int(rng.integers(len(wire_palette)))]
            wires.update({p: color for p in comp})
            plans.append({"beam": beam, "old": old, "kept": kept, "color": source["color"]})
            source["names"].append(name)
            beam_cells.update(beam)
            found = True
            break
        if not found:
            return None
    unhit_count = 1 if complex_witness else int(rng.integers(3))
    for _ in range(unhit_count):
        found = False
        for _ in range(100):
            start = tuple(map(int, (rng.integers(2, shape[0]-2), rng.integers(2, shape[1]-2))))
            walk = _branch(rng, start, DIRS[int(rng.integers(4))], int(rng.integers(4)), shape, True)
            if walk is None or len(walk) < 5:
                continue
            comp = set(walk)
            if comp & beam_cells or not _compatible(comp, set(wires), footprints):
                continue
            color = wire_palette[int(rng.integers(len(wire_palette)))]
            wires.update({p: color for p in comp})
            found = True
            break
        if not found:
            return None
    # Additional ports belong to the same sources; miss beams teach stopping.
    desired_misses = 1 if mode == "simple" else int(rng.integers(3))
    misses = []
    options = [(i, n) for i, f in enumerate(sources) for n in PORTS if n not in f["names"]]
    rng.shuffle(options)
    for i, name in options:
        if len(misses) == desired_misses:
            break
        source = sources[i]
        off, d = PORTS[name]
        port = _add(source["core"], off)
        if not _inside(port, shape):
            continue
        beam = _ray(port, d, shape)
        if set(beam) & (set(wires) | beam_cells) or any(p in footprints for p in beam[1:]):
            continue
        source["names"].append(name)
        beam_cells.update(beam)
        misses.append({"beam": beam, "color": source["color"]})
    if mode == "simple" and not misses:
        return None
    full = _paint_scene(shape, bg, structure, sources, wires)
    # Check bead-free reconstruction too: an intentional near-gap is illegal.
    try:
        info = _analyze(full)
        if info["wires"] != wires or len(info["hits"]) != desired_hits:
            return None
        a = _damage(rng, full, wires, plans, complex_witness)
        if a is None:
            return None
        expected = _constructed_output(full, bg, wires, plans, misses)
        if not np.array_equal(solve(a), expected):
            return None
        return {"input": a, "output": expected}
    except AssertionError:
        return None


def _fallback(rng, palette, direction, mode):
    # Safe geometry, subsequently transformed and recolored. It obeys the
    # same public rule and provides every teaching witness if needed.
    shape = (24, 25)
    sources = [{"core": (12, 11), "color": 1, "names": ["N0", "E1"]}]
    pivot = (7, 11)
    old = _line(pivot, (3, 11)) + _line((3, 11), (3, 18))[1:] + _line((3, 18), (0, 18))[1:]
    kept = _line(pivot, (7, 4)) + _line((7, 4), (10, 4))[1:]
    other = _line((23, 6), (18, 6)) + _line((18, 6), (18, 9))[1:]
    wires = {p: 2 for p in old + kept}
    wires.update({p: 3 for p in other})
    plans = [{"old": old, "kept": kept, "beam": _line((11, 11), pivot), "color": 1}]
    misses = [{"beam": _line((13, 13), (13, 24)), "color": 1}]
    full = _paint_scene(shape, 0, 8, sources, wires)
    a = _damage(rng, full, wires, plans, True)
    _require(a is not None, "fallback damage failed")
    b = _constructed_output(full, 0, wires, plans, misses)
    lut = np.arange(10, dtype=np.int8)
    for old_c, new_c in zip((0, 8, 1, 2, 3), palette[:5]):
        lut[old_c] = new_c
    rotation = (-DIRS.index(direction)) % 4
    a, b = np.rot90(lut[a], rotation).copy(), np.rot90(lut[b], rotation).copy()
    _require(np.array_equal(solve(a), b), "fallback disagrees with solver")
    return {"input": a, "output": b}


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    held_out = DIRS[int(rng.integers(4))]
    train_dirs = tuple(d for d in DIRS if d != held_out)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            mode = ("simple" if i == 0 else "bent" if i == 1 else "varied") if split == "train" else "varied"
            required = held_out if split == "test" else train_dirs[int(rng.integers(3))]
            desired = 1 if mode == "simple" else int(rng.choice((1, 2, 3), p=(.30, .50, .20)))
            for attempt in range(48):
                roles = palette if rng.random() < .35 else list(map(int, rng.permutation(10)))
                pair = _attempt_scene(rng, roles, train_dirs if split == "train" else DIRS, required, mode, desired)
                if pair is None:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                # Palette retries are bounded; 10! role assignments provide
                # ample distinctness even for the maximum fourteen pairs.
                for _ in range(32):
                    roles = list(map(int, rng.permutation(10)))
                    pair = _fallback(rng, roles, required, mode)
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("bounded uniqueness retries exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    a = _array(pair["input"])
    info = _analyze(a)
    touched = {h["component"] for h in info["hits"]}
    old = set().union(*(h["old"] for h in info["hits"]))
    return {"shape": list(a.shape), "sources": len(info["frames"]),
            "ports": len(info["beams"]), "hits": len(info["hits"]),
            "misses": sum(b["contact"] is None for b in info["beams"]),
            "components": len(info["components"]), "unhit": len(info["components"])-len(touched),
            "wire_colors": len(set(info["wires"].values())),
            "markers": len(info["markers"]), "gaps": len(info["gaps"]),
            "marked_contacts": sum(h["contact"] in info["markers"] for h in info["hits"]),
            "deleted_markers": len(old & info["markers"]),
            "retained_markers": len(info["markers"] - old),
            "old_bends": [_bends(h["old_walk"]) for h in info["hits"]],
            "kept_bends": [_bends(h["kept_walk"]) for h in info["hits"]],
            "directions": [list(h["d"]) for h in info["hits"]],
            "beam_lengths": [len(b["cells"]) for b in info["beams"]],
            "clipped": sum(len(f["footprint"]) < 16 for f in info["frames"]),
            "roles": [info["bg"], info["structure"], *[f["color"] for f in info["frames"]]]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses = set(), set()
    alternatives = {"omit_A": False, "omit_B": False, "keep_tail": False,
                    "keep_wire_color": False, "global_recolor": False, "ignore_first_contact": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _array(pair["input"]), _array(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            info = _analyze(a)
            predicted = _render(a, info)
            _require(np.array_equal(predicted, b), "output violates beam grafting")
            _require(not np.array_equal(a, b), "identity pair")
            fixed = [p for p in info["excluded"] if a[p] != info["bg"]]
            _require(all(a[p] == b[p] for p in fixed), "frame material changed")
            _require(all(b[p] != info["structure"] for p in info["markers"]), "repair bead remains")
            if split != "train":
                continue
            touched = {h["component"] for h in info["hits"]}
            if any(x["contact"] is None for x in info["beams"]):
                witnesses.add("miss")
            if len(touched) < len(info["components"]):
                witnesses.add("unhit")
            for h in info["hits"]:
                if _bends(h["old_walk"]):
                    witnesses.add("bent_tail")
                if _bends(h["kept_walk"]):
                    witnesses.add("bent_kept")
                if h["contact"] in info["markers"]:
                    witnesses.add("marked_contact")
            variants = {"omit_A": {"emit": False}, "omit_B": {"surgery": False},
                        "keep_tail": {"erase": False}, "keep_wire_color": {"recolor": False},
                        "global_recolor": {"global_color": True}, "ignore_first_contact": {"stop": False}}
            for name, kwargs in variants.items():
                alternatives[name] |= not np.array_equal(_render(a, info, **kwargs), b)
    _require({"miss", "unhit", "bent_tail", "bent_kept", "marked_contact"} <= witnesses,
             "training lacks a required distinguishing witness")
    _require(all(alternatives.values()), "training fails to distinguish a documented alternative")
    return True


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def _contact_regression():
    """Small explicit geometry: the hidden corner is the first contact.

    This fixture is new composition evidence, not a copied parent example.
    Its asserted coordinates exercise decisions separately from generation.
    """
    sources = [{"core": (6, 2), "color": 1, "names": ["E0", "N1"]}]
    old = _line((6, 9), (6, 13)) + _line((6, 13), (0, 13))[1:]
    kept = _line((6, 9), (11, 9)) + _line((11, 9), (11, 14))[1:]
    unhit = _line((16, 1), (13, 1)) + _line((13, 1), (13, 5))[1:]
    wires = {p: 2 for p in old + kept}
    wires.update({p: 3 for p in unhit})
    a = _paint_scene((17, 19), 0, 8, sources, wires)
    for p in ((6, 9), (6, 13), (11, 14), (13, 5)):
        a[p] = 8
    a[3, 13] = 0
    before = a.copy()
    predicted = solve(a)
    _require(np.array_equal(a, before), "solve modified its input")
    _require(all(predicted[p] == 0 for p in old[1:]), "bent obsolete tail was not erased")
    _require(all(predicted[p] == 1 for p in kept), "payload stopped before the retained endpoint")
    _require(all(predicted[p] == 3 for p in unhit), "unhit wire was not repaired and preserved")
    _require(predicted[0, 3] == 1 and predicted[6, 7] == 1, "miss or connecting beam failed")
    raw_first = next(p for p in _ray((6, 4), (0, 1), a.shape) if a[p] in (2, 3))
    _require(raw_first == (6, 10) and _analyze(a)["hits"][0]["contact"] == (6, 9),
             "marked first-contact regression no longer distinguishes skipped reconstruction")


def self_test():
    """42 seeds, streamed measurements; no stored grids, files, or unbounded retries."""
    _contact_regression()
    shapes, structures, signatures, directions, palettes = set(), set(), set(), set(), set()
    hits, sources, bends, kept_bends, beam_lengths = set(), set(), set(), set(), []
    totals = Counter()
    for seed in range(42):
        nt = 2 + seed % 9
        nq = 1 + seed % min(4, nt-1)
        p = generate_puzzle(seed, nt, nq)
        digest = hashlib.sha256()
        train_dirs, query_dirs = set(), set()
        for split in ("train", "test"):
            for pair in p[split]:
                a, b = pair["input"], pair["output"]
                f = measure_pair(pair)
                shapes.add(tuple(f["shape"]))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
                digest.update(a.tobytes())
                palettes.add(tuple(f["roles"]))
                hits.add(f["hits"])
                sources.add(f["sources"])
                bends.update(f["old_bends"])
                kept_bends.update(f["kept_bends"])
                beam_lengths.extend(f["beam_lengths"])
                local_dirs = {tuple(d) for d in f["directions"]}
                directions.update(local_dirs)
                (train_dirs if split == "train" else query_dirs).update(local_dirs)
                for name in ("misses", "unhit", "gaps", "marked_contacts", "deleted_markers", "retained_markers", "clipped"):
                    totals[name] += f[name]
                # Geometry and palette equivariance, on actual returned grids.
                if seed < 3:
                    transformed = np.rot90(np.fliplr(a))
                    _require(np.array_equal(solve(transformed), np.rot90(np.fliplr(b))), "symmetry failed")
                    lut = np.array([9-i for i in range(10)], dtype=np.int8)
                    _require(np.array_equal(solve(lut[a]), lut[b]), "palette equivariance failed")
                    info = _analyze(a)
                    reversed_info = dict(info, hits=list(reversed(info["hits"])), beams=list(reversed(info["beams"])))
                    _require(np.array_equal(_render(a, reversed_info), b), "simultaneous edit order changed output")
        _require(bool(query_dirs-train_dirs), "held-out orientation shift disappeared")
        signatures.add(digest.digest())
        if seed == 0:
            _require(_equal(p, generate_puzzle(seed, nt, nq)), "nondeterministic regeneration")
            # Mutations include preserved cells and semantic sites, not just
            # changes already made by the transformation.
            query = p["test"][0]
            info = _analyze(query["input"])
            sites = [(0, 0), info["hits"][0]["contact"], min(info["hits"][0]["old"])]
            for site in sites:
                bad = {"train": p["train"], "test": [dict(q) for q in p["test"]]}
                bad["test"][0]["output"] = query["output"].copy()
                bad["test"][0]["output"][site] = (int(query["output"][site])+1) % 10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("corrupt output accepted")
    _require(len(signatures) == 42 and len(shapes) >= 40 and len(structures) >= 150, "structural diversity collapsed")
    _require(directions == set(DIRS) and hits >= {1, 2} and sources >= {1, 2}, "source/contact variation collapsed")
    _require(bends >= {0, 1, 2} and kept_bends >= {0, 1, 2}, "path variation collapsed")
    _require(len(palettes) >= 30 and min(beam_lengths) <= 2 and max(beam_lengths) >= 20, "palette/length variation collapsed")
    _require(all(totals[name] > 0 for name in totals), "damage or clipping mode missing")
    for nt, nq in ((2, 1), (10, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(12345, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "structures": len(structures), "payload_role_assignments": len(palettes),
            "hit_counts": sorted(hits), "source_counts": sorted(sources),
            "old_bends": sorted(bends), "retained_bends": sorted(kept_bends),
            "beam_length_range": [min(beam_lengths), max(beam_lengths)], "totals": dict(totals)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
