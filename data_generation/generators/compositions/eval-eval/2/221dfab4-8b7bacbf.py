"""Pulse reservoirs: an edge clock travels through a branching paint circuit.

Only standard-library modules and NumPy are used. No files are read or written.
Distances and reservoir ownership are reconstructed from the input by solve().
"""

from collections import deque
import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
SIDES = ("top", "right", "bottom", "left")


def _neighbors(cells, shape, steps=D4):
    h, w = shape
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - cells


def _components(cells):
    left = set(cells)
    result = []
    while left:
        first = min(left)
        left.remove(first)
        group = {first}
        queue = [first]
        for r, c in queue:
            for dr, dc in D4:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        result.append(group)
    return result


def _border(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _distance(cells, roots):
    distances = {p: 0 for p in roots}
    queue = deque(sorted(roots))
    while queue:
        r, c = queue.popleft()
        for dr, dc in D4:
            q = (r + dr, c + dc)
            if q in cells and q not in distances:
                distances[q] = distances[(r, c)] + 1
                queue.append(q)
    return distances


def _array(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _analyze(input_grid):
    """Find cavities, edge sources, monochromatic wires, and actual inlets."""
    a = _array(input_grid)
    h, w = a.shape
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    assert np.count_nonzero(a == bg) > a.size / 2, "evident majority background"
    cavities = [g for g in _components(zip(*np.where(a == bg)))
                if not any(_border(p, a.shape) for p in g)]
    assert cavities, "at least one fully enclosed reservoir"
    shell_colors = set()
    loops = []
    for cells in cavities:
        edge = _neighbors(cells, a.shape)
        colors = {int(a[p]) for p in edge}
        assert len(colors) == 1 and bg not in colors, "monochromatic closed shell"
        color = next(iter(colors))
        shell_colors.add(color)
        shell = {p for p in _neighbors(cells, a.shape, D8) if a[p] == color}
        loops.append({"cells": cells, "shell": shell})
    assert len(shell_colors) == 1, "one visible shell role per example"
    shell_color = next(iter(shell_colors))

    # A source is an edge run with a differently colored, single-cell end tip.
    # All other pixels of its wire component are strictly inside the canvas.
    sources = []
    for color in sorted(set(map(int, np.unique(a))) - {bg, shell_color}):
        for cells in _components(zip(*np.where(a == color))):
            boundary = {p for p in cells if _border(p, a.shape)}
            if len(boundary) < 2:
                continue
            for side in range(4):
                coords = [(0, c) for c in range(w)] if side == 0 else (
                    [(r, w - 1) for r in range(h)] if side == 1 else (
                    [(h - 1, c) for c in range(w)] if side == 2 else
                    [(r, 0) for r in range(h)]))
                indices = [i for i, p in enumerate(coords) if p in boundary]
                if (len(indices) != len(boundary) or not indices
                        or indices != list(range(indices[0], indices[-1] + 1))):
                    continue
                if any(p in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)) for p in boundary):
                    continue
                tips = []
                for index in (indices[0] - 1, indices[-1] + 1):
                    if not 0 < index < len(coords) - 1:
                        continue
                    p = coords[index]
                    paint = int(a[p])
                    if paint in (bg, shell_color, color):
                        continue
                    if any(a[q] == paint for q in _neighbors({p}, a.shape)):
                        continue
                    if len(_neighbors({p}, a.shape) & cells) == 1:
                        tips.append((p, paint))
                if len(tips) == 1 and cells - boundary:
                    p, paint = tips[0]
                    sources.append({"bar": boundary, "tip": p, "paint": paint,
                                    "wire": color, "side": side, "cells": cells,
                                    "dist": _distance(cells, boundary)})
    assert 1 <= len(sources) <= 2, "one or two edge-rooted paint circuits"
    assert len({s["wire"] for s in sources}) == len(sources)
    assert len({s["paint"] for s in sources}) == len(sources)
    assert all(2 <= len(s["bar"]) <= 5 for s in sources)
    tips = {s["tip"] for s in sources}
    assert len(tips) == len(sources)
    wire_colors = set(map(int, np.unique(a))) - {bg, shell_color} - {s["paint"] for s in sources}
    assert all(s["wire"] in wire_colors for s in sources), "separate paint/wire roles"
    wires = []
    for color in sorted(wire_colors):
        for cells in _components(zip(*np.where(a == color))):
            edges = sum(len(_neighbors({p}, a.shape) & cells) for p in cells) // 2
            assert edges == len(cells) - 1, "wire components are trees"
            wires.append((cells, color))
    allwire = set().union(*(g for g, _ in wires))
    # Neither a contour nor a different-colored wire relays the clock.
    for loop in loops:
        inlet = _neighbors(loop["shell"], a.shape) & allwire
        assert len(inlet) <= 1, "a reservoir has zero or one wire inlet"
        loop["inlet"] = next(iter(inlet)) if inlet else None
        matches = [s for s in sources if inlet & s["cells"]]
        assert len(matches) <= 1
        loop["source"] = matches[0] if matches else None
        loop["depth"] = matches[0]["dist"][loop["inlet"]] if matches else None
    return {"background": bg, "shell_color": shell_color, "loops": loops,
            "sources": sources, "wires": wires}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repaint the clocked wires and fill reservoirs sampled on paint beats."""
    a = _array(input_grid)
    info = _analyze(a)
    out = a.copy()
    for source in info["sources"]:
        for p, distance in source["dist"].items():
            out[p] = (info["background"] if distance % 2 else
                      source["paint"] if distance % 6 == 4 else source["wire"])
    for loop in info["loops"]:
        if loop["depth"] is not None and loop["depth"] % 6 == 4:
            for p in loop["cells"]:
                out[p] = loop["source"]["paint"]
    return out


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pick(rng, values):
    return values[_ri(rng, 0, len(values) - 1)]


def _shape(rng, small):
    """Connected cavities with independent extents and asymmetric row profiles."""
    h, w = _ri(rng, 1, 3 if small else 5), _ri(rng, 1, 3 if small else 5)
    kind = _pick(rng, ("box", "box", "steps", "waist", "slant", "diamond", "long"))
    if kind == "diamond":
        radius = _ri(rng, 1, 1 if small else 2)
        inner = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    else:
        if kind == "long":
            h, w = 1, _ri(rng, 2, 4 if small else 8)
        inner = set()
        left = 0
        for r in range(h):
            if kind in ("steps", "slant") and r:
                left += _pick(rng, (0, 1) if kind == "slant" else (-1, 0, 1))
            width = max(1, w - 1) if kind == "waist" and 0 < r < h - 1 else w
            if kind == "steps":
                width = max(2, width + _ri(rng, -1, 1))
            inner.update((r, c) for c in range(left, left + width))
        for r in range(1, h):
            previous = {c for rr, c in inner if rr == r - 1}
            current = {c for rr, c in inner if rr == r}
            if not previous & current:
                a, b = sorted((min(previous), min(current)))
                inner.update((r, c) for c in range(a, b + 1))
    if rng.random() < .5:
        inner = {(c, r) for r, c in inner}
    # The minimal orthogonal shell also permits diagonal shell corners.
    shell = {(r + dr, c + dc) for r, c in inner for dr, dc in D4} - inner
    r0, c0 = min(r for r, c in shell), min(c for r, c in shell)
    return ({(r - r0, c - c0) for r, c in inner},
            {(r - r0, c - c0) for r, c in shell})


def _source(rng, shape, side, wire, paint):
    h, w = shape
    span = w if side % 2 == 0 else h
    length = _ri(rng, 2, min(5, span - 5))
    start = _ri(rng, 2, span - length - 2)
    def pos(d, lateral):
        return ((d, lateral) if side == 0 else (lateral, w - 1 - d)
                if side == 1 else (h - 1 - d, lateral) if side == 2 else (lateral, d))
    bar = {pos(0, c) for c in range(start, start + length)}
    tip = pos(0, start - 1 if rng.random() < .5 else start + length)
    lateral = _ri(rng, start, start + length - 1)
    stem = {pos(d, lateral) for d in range(1, _ri(rng, 2, 4) + 1)}
    if len(_neighbors({tip}, shape) & (bar | stem)) != 1:
        return None
    return {"bar": bar, "tip": tip, "paint": paint, "wire": wire,
            "cells": bar | stem, "side": side}


def _path(rng, start, goals, allowed, shape):
    """A bounded randomized shortest route through the currently free cells."""
    queue = deque([start])
    prev = {start: None}
    directions = [D4[int(i)] for i in rng.permutation(4)]
    while queue:
        p = queue.popleft()
        if p in goals:
            path = []
            while p is not None:
                path.append(p)
                p = prev[p]
            return path
        for dr, dc in directions:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in prev:
                prev[q] = p
                queue.append(q)
    return None


def _edge_distance(p, source, shape):
    return (p[0], shape[1] - 1 - p[1], shape[0] - 1 - p[0], p[1])[source["side"]]


def _scene(rng, sides, lesson):
    nsource = len(sides)
    compact = not lesson and nsource == 1 and rng.random() < .4
    h, w = ((_ri(rng, 25, 30), _ri(rng, 25, 30)) if nsource == 2 else
            (_ri(rng, 12, 20), _ri(rng, 12, 22)) if compact else
            (_ri(rng, 18, 30), _ri(rng, 16, 30)))
    shape = (h, w)
    canvas = {(r, c) for r in range(h) for c in range(w)}
    interior = {p for p in canvas if not _border(p, shape)}
    colors = list(map(int, rng.permutation(10)))
    bg, shell_color = colors[:2]
    sources = []
    for i, side in enumerate(sides):
        placed = False
        for _ in range(20):
            source = _source(rng, shape, side, colors[2 + 2 * i], colors[3 + 2 * i])
            if source is None:
                continue
            occupied = set().union(*(s["cells"] | {s["tip"]} for s in sources))
            if (source["cells"] | {source["tip"]}) & (occupied | _neighbors(occupied, shape, D8)):
                continue
            sources.append(source)
            placed = True
            break
        if not placed:
            return None
    counts = ([_ri(rng, 2, 3) for _ in sources] if lesson and nsource == 2 else
              [_ri(rng, 3, 5)] if lesson else
              [_ri(rng, 1, 2 if compact else 4) for _ in sources])
    owners = [i for i, count in enumerate(counts) for _ in range(count)]
    # An unsupplied reservoir can have an open tail or be completely detached.
    owners += [None] * (_ri(rng, 1, 2) if lesson else _ri(rng, 0, 2))
    rng.shuffle(owners)
    regions = []
    occupied = set().union(*(s["cells"] | {s["tip"]} for s in sources))
    for owner in owners:
        placed = False
        inner, shell = _shape(rng, compact or nsource == 2 or len(owners) > 6)
        hh, ww = max(r for r, c in shell) + 1, max(c for r, c in shell) + 1
        if hh + 2 > h or ww + 2 > w:
            return None
        for placement_attempt in range(60):
            r0, c0 = _ri(rng, 1, h - hh - 1), _ri(rng, 1, w - ww - 1)
            inside = {(r + r0, c + c0) for r, c in inner}
            edge = {(r + r0, c + c0) for r, c in shell}
            cells = inside | edge
            if nsource == 2 and owner is not None and placement_attempt < 50:
                center = (sum(r for r, c in inside) / len(inside),
                          sum(c for r, c in inside) / len(inside))
                distances = [min(abs(center[0] - r) + abs(center[1] - c)
                                 for r, c in s["bar"]) for s in sources]
                if distances[owner] > distances[1 - owner] + 1:
                    continue
            if cells & (occupied | _neighbors(occupied, shape, D8)):
                continue
            occupied |= cells
            regions.append({"cells": inside, "shell": edge, "owner": owner})
            placed = True
            break
        if not placed:
            return None
    all_regions = set().union(*(q["cells"] | q["shell"] for q in regions))
    shells = set().union(*(q["shell"] for q in regions))
    tips = {s["tip"] for s in sources}
    blocked_regions = all_regions | _neighbors(shells, shape, D8)
    for owner in map(int, rng.permutation(nsource)):
        source = sources[owner]
        own = source["cells"]
        other = set().union(*(s["cells"] for i, s in enumerate(sources) if i != owner))
        for index in rng.permutation(len(regions)):
            loop = regions[int(index)]
            if loop["owner"] != owner:
                continue
            ports = sorted(p for p in _neighbors(loop["shell"], shape) & interior
                           if p not in all_regions and len(_neighbors({p}, shape) & shells) == 1)
            rng.shuffle(ports)
            route = None
            foreign = other | _neighbors(other, shape, D8) | tips | _neighbors(tips, shape)
            forbidden = blocked_regions | foreign
            for port in ports[:24]:
                if port in foreign or port in own:
                    continue
                allowed = (interior - forbidden - own) | {port}
                goals = _neighbors(own, shape) & allowed
                allowed -= _neighbors(own, shape, D8) - goals
                candidate = _path(rng, port, goals, allowed, shape)
                if candidate is None:
                    continue
                contacts = sum(len(_neighbors({p}, shape) & own) for p in candidate)
                if contacts != 1:
                    continue
                route = candidate
                break
            if route is None:
                return None
            own.update(route)
            # _path returns goal-to-start; the actual port is its last cell.
            loop["inlet"] = route[-1]
        source["dist"] = _distance(own, source["bar"])

    # Optional unsourced tails use a separate visible wire color.
    inactive = set()
    usedwire = set().union(*(s["cells"] for s in sources))
    for loop in regions:
        if loop["owner"] is not None or rng.random() < .45:
            continue
        ports = sorted(p for p in _neighbors(loop["shell"], shape) & interior
                       if p not in all_regions and len(_neighbors({p}, shape) & shells) == 1)
        rng.shuffle(ports)
        for port in ports:
            blocked = all_regions | usedwire | _neighbors(usedwire, shape, D8)
            blocked |= tips | _neighbors(tips, shape, D8) | inactive | _neighbors(inactive, shape, D8)
            blocked |= _neighbors(shells, shape, D8) - {port}
            if port in blocked:
                continue
            tail = {port}
            current = port
            for _ in range(_ri(rng, 2, 6)):
                choices = sorted(q for q in _neighbors({current}, shape) & interior
                                 if q not in blocked and q not in tail
                                 and len(_neighbors({q}, shape) & tail) == 1)
                if not choices:
                    break
                current = _pick(rng, choices)
                tail.add(current)
            if len(tail) >= 2:
                inactive |= tail
                break

    a = np.full(shape, bg, dtype=np.int8)
    for loop in regions:
        for p in loop["shell"]:
            a[p] = shell_color
    for source in sources:
        for p in source["cells"]:
            a[p] = source["wire"]
        a[source["tip"]] = source["paint"]
    for p in inactive:
        a[p] = colors[2 + 2 * nsource]
    # Construct labels from intended owners and paths, before input analysis.
    b = a.copy()
    for source in sources:
        for p, d in source["dist"].items():
            b[p] = bg if d % 2 else source["paint"] if d % 6 == 4 else source["wire"]
    for loop in regions:
        if loop["owner"] is None:
            continue
        source = sources[loop["owner"]]
        if source["dist"][loop["inlet"]] % 6 == 4:
            for p in loop["cells"]:
                b[p] = source["paint"]
    try:
        info = _analyze(a)
    except AssertionError:
        return None
    if len(info["loops"]) != len(regions) or not np.array_equal(solve(a), b):
        return None
    features = _features(a, info)
    if lesson and (not features["filled"] or not features["cold_connected"]):
        return None
    # Every productive source copies its own paint into at least one reservoir.
    if lesson and len({q["source"]["tip"] for q in info["loops"]
                       if q["depth"] is not None and q["depth"] % 6 == 4}) != nsource:
        return None
    if lesson and (features["max_depth"] < 12 or not features["bends"]
                   or not features["branches"] or not features["geometric_disagreement"]
                   or max(features["filled_areas"], default=0) < 3):
        return None
    return {"input": a, "output": b}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or not low <= value <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    base = _ri(rng, 0, 3)
    mixed = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_paint = None
    lesson_depths = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            side = (base + (index % 2 if mixed else 0)) % 4 if split == "train" else (
                base + 2 + index % (2 if mixed else 1)) % 4
            nsource = 1 if index == 0 and split == "train" else (
                2 if index == 1 and split == "train" else _pick(rng, (1, 1, 2)))
            sides = [side] * nsource
            if nsource == 2 and mixed and rng.random() < .5:
                sides[1] = (base + 1 - index % 2) % 4 if split == "train" else (
                    base + 3 - index % 2) % 4
            accepted = None
            for _ in range(320):
                pair = _scene(rng, sides, split == "train" and index < 2)
                if pair is None:
                    continue
                key = _key(pair["input"])
                if key in seen:
                    continue
                paints = {s["paint"] for s in _analyze(pair["input"])["sources"]}
                if split == "train" and index == 1 and paints == {first_paint}:
                    continue
                hot_depths = set(_features(pair["input"])["hot_depths"])
                if split == "train" and index == 1 and len(lesson_depths | hot_depths) < 2:
                    continue
                accepted = pair
                if split == "train" and index == 0:
                    first_paint = next(iter(paints))
                    lesson_depths = hot_depths
                break
            if accepted is None:
                raise RuntimeError("bounded circuit construction exhausted")
            seen.add(_key(accepted["input"]))
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _features(a, info=None):
    info = _analyze(a) if info is None else info
    loops = info["loops"]
    supplied = [q for q in loops if q["depth"] is not None]
    filled = [q for q in supplied if q["depth"] % 6 == 4]
    branches = bends = 0
    for s in info["sources"]:
        for r, c in s["cells"] - s["bar"]:
            near = _neighbors({(r, c)}, a.shape) & s["cells"]
            branches += len(near) >= 3
            bends += any(p[0] == r for p in near) and any(p[1] == c for p in near)
    disagreement = sum((q["depth"] % 6 == 4) != (
        _edge_distance(q["inlet"], q["source"], a.shape) % 6 == 4) for q in supplied)
    shapes = []
    for q in loops:
        r0, c0 = min(r for r, c in q["cells"]), min(c for r, c in q["cells"])
        shapes.append(tuple(sorted((r - r0, c - c0) for r, c in q["cells"])))
    return {"shape": tuple(a.shape), "background": info["background"],
            "sources": len(info["sources"]), "sides": tuple(s["side"] for s in info["sources"]),
            "bars": tuple(len(s["bar"]) for s in info["sources"]),
            "paints": tuple(s["paint"] for s in info["sources"]),
            "loops": len(loops), "filled": len(filled), "cold_connected": len(supplied) - len(filled),
            "unsupplied": len(loops) - len(supplied), "branches": int(branches), "bends": int(bends),
            "max_depth": max(max(s["dist"].values()) for s in info["sources"]),
            "geometric_disagreement": int(disagreement), "cavity_shapes": shapes,
            "filled_areas": tuple(len(q["cells"]) for q in filled),
            "hot_depths": tuple(q["depth"] for q in filled),
            "cold_phases": tuple(q["depth"] % 6 for q in supplied if q not in filled),
            "wire_cells": sum(len(s["cells"]) for s in info["sources"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _array(pair["input"]), _array(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(solve(a), b), "input-derived prediction mismatch"
            assert not np.array_equal(a, b)
            wire = set().union(*(s["cells"] for s in info["sources"]))
            hot = set().union(*(q["cells"] for q in info["loops"]
                                if q["depth"] is not None and q["depth"] % 6 == 4))
            allowed = wire | hot
            assert all(tuple(p) in allowed for p in np.argwhere(a != b))
            for s in info["sources"]:
                assert b[s["tip"]] == s["paint"]
                assert all(b[p] == s["wire"] for p in s["bar"])
            for q in info["loops"]:
                assert all(a[p] == b[p] for p in q["shell"])
                if q["cells"] & hot:
                    assert all(b[p] == q["source"]["paint"] for p in q["cells"])
                else:
                    assert all(b[p] == info["background"] for p in q["cells"])
            f = _features(a, info)
            records[split].append(f)
    lessons = records["train"][:2]
    assert all(f["max_depth"] >= 12 and f["branches"] and f["bends"]
               and f["geometric_disagreement"] and f["filled"]
               and f["cold_connected"] and f["unsupplied"]
               and max(f["filled_areas"]) >= 3 for f in lessons)
    assert {f["sources"] for f in lessons} == {1, 2}
    assert len({c for f in lessons for c in f["paints"]}) >= 2
    assert len({d for f in lessons for d in f["hot_depths"]}) >= 2
    second = _analyze(np.asarray(puzzle["train"][1]["input"]))
    assert {q["source"]["paint"] for q in second["loops"]
            if q["depth"] is not None and q["depth"] % 6 == 4} == {
                s["paint"] for s in second["sources"]}
    train_sides = {side for f in records["train"] for side in f["sides"]}
    assert all(not train_sides.intersection(f["sides"]) for f in records["test"])


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = pair[role]
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded streaming audit: 42 seeds, no stored puzzle archive or traces."""
    structures, shapes, cavities, hot_depths, cold_phases = set(), set(), set(), set(), set()
    sides, bars, backgrounds, paints, counts = set(), set(), set(), set(), set()
    digests = set()
    branching = multi = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest not in digests
        digests.add(digest)
        if seed in (0, 21, 41):
            assert digest == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                shapes.add(f["shape"])
                cavities.update(f["cavity_shapes"])
                hot_depths.update(f["hot_depths"])
                cold_phases.update(f["cold_phases"])
                sides.update(f["sides"])
                bars.update(f["bars"])
                backgrounds.add(f["background"])
                paints.update(f["paints"])
                counts.add(f["loops"])
                branching += f["branches"] > 0
                multi += f["sources"] == 2
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)
                structures.add((a.shape, hashlib.sha256(normalized).digest()))
    assert len(shapes) >= 45 and len(cavities) >= 40 and len(structures) >= 200
    assert sides == set(range(4)) and bars >= {2, 3, 4, 5}
    assert len(backgrounds) >= 8 and len(paints) >= 8
    assert len(hot_depths) >= 4 and cold_phases >= {0, 1, 2, 3, 5}
    assert len(counts) >= 4 and branching >= 100 and multi >= 42
    for nt in range(2, 11):
        for ns in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    # These wrong simplifications must change the initial lessons' answers.
    for pair in p["train"][:2]:
        a, b = pair["input"], pair["output"]
        info = _analyze(a)
        without_clock, without_fill, straight_clock = b.copy(), b.copy(), b.copy()
        for q in info["loops"]:
            if q["source"] is None:
                continue
            for cell in q["cells"]:
                without_clock[cell] = q["source"]["paint"]
                without_fill[cell] = info["background"]
                straight_clock[cell] = (q["source"]["paint"] if
                    _edge_distance(q["inlet"], q["source"], a.shape) % 6 == 4
                    else info["background"])
        assert not np.array_equal(b, without_clock)
        assert not np.array_equal(b, without_fill)
        assert not np.array_equal(b, straight_clock)
    # Spatial and literal-color changes cannot alter the fixed rule.
    permutation = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for pair in p["train"][:2] + p["test"]:
        for turns in range(4):
            a = np.rot90(pair["input"], turns)
            b = np.rot90(pair["output"], turns)
            assert np.array_equal(solve(a), b)
            assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
            assert np.array_equal(solve(permutation[a]), permutation[b])
    pair = p["test"][0]
    cells = np.argwhere(pair["input"] != pair["output"])
    r, c = cells[0]
    pair["output"][r, c] = pair["input"][r, c]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
