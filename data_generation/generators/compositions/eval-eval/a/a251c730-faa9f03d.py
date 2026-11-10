"""Threaded motif spools: keyed transparent tiles follow repaired, rerouted paths.

All solving conventions are fixed or visible. No parent code is imported.
"""

from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BACKGROUND, REPAIR, DIRECTION = 0, 8, 9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, left):
                left.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _simple(path):
    cells = set(path)
    return (len(path) >= 2 and len(cells) == len(path)
            and all(len(_neighbors(p, cells)) == (1 if p in (path[0], path[-1]) else 2)
                    for p in path))


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    d = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i*d[0], a[1] + i*d[1])
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _cards(inp):
    """The 8 border is recognized geometrically, before any color inference."""
    cards, mask = [], np.zeros(inp.shape, dtype=bool)
    cells = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
    for component in _components(cells):
        top = min(p[0] for p in component)
        bottom = max(p[0] for p in component)
        left = min(p[1] for p in component)
        right = max(p[1] for p in component)
        h, w = bottom-top+1, right-left+1
        if not ((h == 3 and 5 <= w <= 9) or (w == 3 and 5 <= h <= 9)):
            continue
        border = {(r, c) for r in range(top, bottom+1) for c in range(left, right+1)
                  if r in (top, bottom) or c in (left, right)}
        if component != border:
            continue
        body = inp[top+1:bottom, left+1:right].ravel()
        _require(not np.any(np.isin(body, [REPAIR, DIRECTION])), "annotations inside spool")
        _require(body[0] != 0 and body[-1] != 0, "spool must have visible endpoints")
        _require(not np.any(mask[top:bottom+1, left:right+1]), "overlapping spools")
        mask[top:bottom+1, left:right+1] = True
        cards.append({"box": (top, left, h, w), "body": tuple(map(int, body))})
    _require(bool(cards), "missing spools")
    return cards, mask


def _analyze(input_grid):
    inp = _grid(input_grid)
    cards, card_mask = _cards(inp)
    outside = set(map(int, np.unique(inp[~card_mask]))) - {0, 8, 9}
    printed = {c for card in cards for c in card["body"] if c != 0}
    carriers = outside - printed
    _require(len(carriers) == 1, "carrier color must be absent from spool interiors")
    carrier = next(iter(carriers))
    keys = outside - {carrier}
    _require(bool(keys), "missing keyed heads")
    templates = {}
    repetitions = {}
    for card in cards:
        body = card["body"]
        positions = [i for i, c in enumerate(body) if c in keys]
        if not positions:  # An unused entry need not be interpreted.
            continue
        _require(len(positions) == 1 and positions[0] in (0, len(body)-1),
                 "a spool key must occur once, at an endpoint")
        tile = body if positions[0] == 0 else tuple(reversed(body))
        key = tile[0]
        _require(key not in templates or templates[key] == tile, "conflicting spool copies")
        templates[key] = tile
        repetitions[key] = repetitions.get(key, 0) + 1
    _require(set(templates) == keys, "every requested key needs a spool")

    heads = {tuple(map(int, p)): int(inp[tuple(p)])
             for p in np.argwhere(np.isin(inp, sorted(keys)) & ~card_mask)}
    repaired = {tuple(map(int, p)) for p in np.argwhere((inp == REPAIR) & ~card_mask)}
    observed = {tuple(map(int, p)) for p in np.argwhere((inp == carrier) & ~card_mask)}
    cells = observed | repaired | set(heads)
    # One omitted straight cell, witnessed by both flanks and at least one
    # further collinear cell. Evaluate all bridges from the unchanged evidence.
    gaps = set()
    for p in sorted(cells):
        for d in ((1, 0), (0, 1)):
            q = _add(p, d)
            far = _add(q, d)
            if not _inside(q, inp.shape) or far not in cells or q in cells:
                continue
            if card_mask[q] or inp[q] != BACKGROUND:
                continue
            back = _add(p, (-d[0], -d[1]))
            forward = _add(far, d)
            if back in cells or forward in cells:
                gaps.add(q)
    cells |= gaps
    paths = []
    for component in _components(cells):
        component_heads = sorted(set(heads) & component)
        _require(len(component_heads) == 1, "each track needs exactly one keyed head")
        head = component_heads[0]
        _require(len(_neighbors(head, component)) == 1, "key must be a path endpoint")
        path, previous, current = [head], None, head
        while True:
            following = [q for q in _neighbors(current, component) if q != previous]
            _require(len(following) <= 1, "branching or touching input track")
            if not following:
                break
            previous, current = current, following[0]
            _require(current not in path, "cyclic track")
            path.append(current)
        _require(set(path) == component and _simple(path), "invalid path")
        paths.append({"key": heads[head], "old": path, "final": list(path), "edit": None})

    arrows = sorted(tuple(map(int, p)) for p in np.argwhere(inp == DIRECTION))
    for q in arrows:
        candidates = []
        for index, route in enumerate(paths):
            path = route["old"]
            for d in DIRS:
                pivot = q[0]-d[0], q[1]-d[1]
                if pivot not in path:
                    continue
                i = path.index(pivot)
                if not 0 < i < len(path)-1:
                    continue
                if path[i+1] != (pivot[0]-d[0], pivot[1]-d[1]):
                    continue
                if not _boundary(path[-1], inp.shape):
                    continue
                previous = path[i-1][0]-pivot[0], path[i-1][1]-pivot[1]
                if previous[0]*d[0] + previous[1]*d[1] != 0:
                    continue
                candidates.append((index, i, d))
        _require(len(candidates) == 1, "direction marker must have one old boundary branch")
        index, i, d = candidates[0]
        route = paths[index]
        _require(route["edit"] is None, "only one redirect per input track")
        pivot = route["old"][i]
        ray = _ray(pivot, d, inp.shape)
        route["final"] = route["old"][:i+1] + ray
        route["edit"] = {"pivot": pivot, "direction": d, "removed": route["old"][i+1:],
                         "ray": ray, "marker": q, "index": i}

    occupied = set()
    for route in paths:
        final = route["final"]
        _require(_simple(final), "ray touches retained part of its own track")
        _require(not any(p in occupied or _neighbors(p, occupied) for p in final),
                 "final tracks must be separated")
        _require(not any(card_mask[p] for p in route["old"] + final), "track intersects spool")
        occupied.update(final)
    _require(all(len(_neighbors(q, cells)) == 1 for q in arrows), "arrow must touch only its pivot")
    return {"input": inp, "cards": cards, "card_mask": card_mask, "carrier": carrier,
            "keys": keys, "templates": templates, "repetitions": repetitions,
            "heads": heads, "paths": paths, "repairs": repaired, "gaps": gaps}


def _paint(shape, carrier, routes, templates):
    out = np.zeros(shape, dtype=np.int8)
    for key, path in routes:
        tile = templates[key]
        for distance, p in enumerate(path):
            color = tile[distance % len(tile)]
            out[p] = color if color != BACKGROUND else carrier
    return out


def _turn_indices(path):
    return [i for i in range(1, len(path)-1)
            if (path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]) !=
               (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])]


def _phase_witnesses(info):
    """Visible cells that distinguish three plausible phase shortcuts."""
    bend, gap, ray = False, False, False
    for route in info["paths"]:
        path, tile = route["final"], info["templates"][route["key"]]
        # A reset at a corner would put the key there instead of this tile cell.
        bend |= any(tile[i % len(tile)] != tile[0] for i in _turn_indices(path))
        skipped = 0
        for i, p in enumerate(path):
            if p in info["gaps"]:
                skipped += 1
            elif skipped:
                gap |= tile[i % len(tile)] != tile[(i-skipped) % len(tile)]
        e = route["edit"]
        if e:
            ray |= any(tile[(e["index"]+j) % len(tile)] != tile[j % len(tile)]
                       for j in range(1, len(e["ray"])+1))
    return bend, gap, ray


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read spools, recover geometry, then unfold each tile along its final path."""
    info = _analyze(input_grid)
    return _paint(info["input"].shape, info["carrier"],
                  [(r["key"], r["final"]) for r in info["paths"]], info["templates"])


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _random_vertices(rng, shape, terminal, minimum_segments=2):
    h, w = shape
    if terminal:
        side = int(rng.integers(4))
        if side < 2:
            start = (0 if side == 0 else h-1, int(rng.integers(1, w-1)))
            axis = 0
        else:
            start = (int(rng.integers(1, h-1)), 0 if side == 2 else w-1)
            axis = 1
    else:
        start = (int(rng.integers(1, h-1)), int(rng.integers(1, w-1)))
        axis = int(rng.integers(2))
    vertices = [start]
    desired = int(rng.integers(minimum_segments, 7))
    for _ in range(desired):
        current = vertices[-1]
        choices = [v for v in range(1, shape[axis]-1) if abs(v-current[axis]) >= 2]
        rng.shuffle(choices)
        for v in choices:
            point = list(current)
            point[axis] = v
            candidate = vertices + [tuple(point)]
            if _simple(_polyline(candidate)):
                vertices = candidate
                break
        else:
            break
        axis = 1-axis
    return vertices


def _separated(path, occupied):
    return not any(p in occupied or _neighbors(p, occupied) for p in path)


def _route_candidate(rng, shape, redirect, allowed, bent_old=False):
    vertices = _random_vertices(rng, shape, redirect, 3 if bent_old else 2)
    path = _polyline(vertices)
    if len(path) < 11 or len(vertices) < 3:
        return None
    if not redirect:
        return {"old": path, "final": path, "edit": None}
    options = []
    for j in range(2 if bent_old else 1, len(vertices)-1):
        pivot, previous = vertices[j], vertices[j-1]
        d = int(np.sign(pivot[0]-previous[0])), int(np.sign(pivot[1]-previous[1]))
        if d not in allowed:
            continue
        at = path.index(pivot)
        old = list(reversed(path))
        i = len(path)-at-1
        ray = _ray(pivot, d, shape)
        final = old[:i+1] + ray
        q = _add(pivot, d)
        if _simple(final) and q not in path and len(_neighbors(q, set(path))) == 1:
            options.append({"old": old, "final": final,
                            "edit": {"pivot": pivot, "direction": d, "index": i,
                                     "removed": old[i+1:], "ray": ray, "marker": q}})
    return _pick(rng, options) if options else None


def _tile(rng, key, decorations, length=None):
    n = int(rng.integers(3, 8)) if length is None else length
    body = [key] + [int(_pick(rng, decorations)) for _ in range(n-1)]
    # Holes are transparent mask positions, not omitted cells of the reference.
    holes = list(rng.choice(np.arange(1, n-1), size=int(rng.integers(1, n-1)), replace=False))
    for i in holes:
        body[int(i)] = 0
    return tuple(body)


def _moat(cells, shape):
    return {q for p in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if _inside(q := (p[0]+dr, p[1]+dc), shape)}


def _place_cards(rng, inp, templates, card_keys, forbidden):
    blocked = _moat(forbidden, inp.shape)
    boxes = []
    for key in card_keys:
        tile = templates[key]
        vertical = bool(rng.integers(2))
        h, w = (len(tile)+2, 3) if vertical else (3, len(tile)+2)
        candidates = [(r, c) for r in range(inp.shape[0]-h+1)
                      for c in range(inp.shape[1]-w+1)]
        rng.shuffle(candidates)
        for r, c in candidates:
            rectangle = {(y, x) for y in range(r, r+h) for x in range(c, c+w)}
            if rectangle & blocked:
                continue
            inp[r:r+h, c:c+w] = REPAIR
            body = list(reversed(tile)) if rng.integers(2) else tile
            if vertical:
                inp[r+1:r+h-1, c+1] = body
            else:
                inp[r+1, c+1:c+w-1] = body
            blocked |= _moat(rectangle, inp.shape)
            boxes.append((r, c, h, w))
            break
        else:
            return None
    return boxes


def _make_pair(rng, config, kind, index):
    first = kind == "train" and index == 0
    second = kind == "train" and index == 1
    query = kind == "test"
    for attempt in range(160):
        # Independent aspect ratios; richer queries share the same size range.
        low = 20 if query else 17
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        n_routes = 2 if first or second else int(rng.integers(2, 4) if query else rng.integers(1, 4))
        n_types = min(n_routes, 2 if first or second else len(config["keys"]))
        used_keys = list(map(int, rng.choice(config["keys"], n_types, replace=False)))
        route_keys = used_keys + [int(_pick(rng, used_keys)) for _ in range(n_routes-n_types)]
        rng.shuffle(route_keys)
        n_redirect = 0 if first else 1 if second else 2 if query else int(rng.integers(0, min(2, n_routes)+1))
        allowed = config["test_dirs"] if query else config["train_dirs"]
        old_occupied, final_occupied, routes = set(), set(), []
        for i, key in enumerate(route_keys):
            for _ in range(110):
                route = _route_candidate(rng, shape, i < n_redirect, allowed,
                                         bent_old=second and i == 0)
                if route is None:
                    continue
                if not _separated(route["old"], old_occupied) or not _separated(route["final"], final_occupied):
                    continue
                if route["edit"] and not _separated([route["edit"]["marker"]], old_occupied):
                    continue
                # Existing off-track annotations cannot touch a later old track.
                markers = [r["edit"]["marker"] for r in routes if r["edit"]]
                if any(p in route["old"] or _neighbors(p, set(route["old"])) for p in markers):
                    continue
                route["key"] = key
                routes.append(route)
                old_occupied.update(route["old"])
                final_occupied.update(route["final"])
                break
            else:
                break
        if len(routes) != n_routes:
            continue
        templates = {key: _tile(rng, key, config["decorations"]) for key in used_keys}
        # Contrasting tile lengths in the first demonstration defeat one shared
        # period. These are visible entries, not a puzzle-level hidden code.
        if first and len({len(t) for t in templates.values()}) == 1:
            templates[used_keys[1]] = _tile(rng, used_keys[1], config["decorations"],
                                           3 if len(templates[used_keys[0]]) != 3 else 5)
        card_keys = list(used_keys)
        unused = [k for k in config["keys"] if k not in used_keys]
        if unused and rng.random() < .25:
            key = int(_pick(rng, unused))
            templates[key] = _tile(rng, key, config["decorations"])
            card_keys.append(key)
        if first or (len(card_keys) < 4 and rng.random() < .25):
            card_keys.append(int(_pick(rng, used_keys)))
        inp = np.zeros(shape, dtype=np.int8)
        carrier = config["carrier"]
        for route in routes:
            for p in route["old"]:
                inp[p] = carrier
            inp[route["old"][0]] = route["key"]
            if route["edit"]:
                inp[route["edit"]["marker"]] = DIRECTION
        forbidden = old_occupied | final_occupied | {
            r["edit"]["marker"] for r in routes if r["edit"]}
        if _place_cards(rng, inp, templates, card_keys, forbidden) is None:
            continue
        # Labels use constructed full routes, not the input parser.
        out = _paint(shape, carrier, [(r["key"], r["final"]) for r in routes], templates)
        complete = inp.copy()
        try:
            info = _analyze(inp)
            if {tuple(r["old"]) for r in info["paths"]} != {tuple(r["old"]) for r in routes}:
                continue
            if not np.array_equal(solve(inp), out):
                continue
        except ValueError:
            continue

        # Damage is accepted only if input-only geometry remains exact. Include
        # marked turns and pivots as well as straight sites; no head is erased.
        final_cells = set().union(*(set(r["final"]) for r in routes))
        priority = []
        if second:
            e = routes[0]["edit"]
            priority += [e["pivot"], e["removed"][len(e["removed"])//2]]
        priority += [r["old"][i] for r in routes for i in _turn_indices(r["old"])]
        remaining = [p for r in routes for p in r["old"][1:]]
        rng.shuffle(priority)
        rng.shuffle(remaining)
        priority = list(dict.fromkeys(priority + remaining))
        desired = int(rng.integers(2, 7))
        if second:
            # Guarantee a marked pivot and a removed marker before random sites.
            e = routes[0]["edit"]
            priority = list(dict.fromkeys([e["pivot"], e["removed"][len(e["removed"])//2]] + priority))
        successes = 0
        for p in priority:
            if any(inp[q] == REPAIR for q in _neighbors(p, old_occupied)):
                continue
            previous = int(inp[p])
            inp[p] = REPAIR
            try:
                candidate = _analyze(inp)
                _require({tuple(r["old"]) for r in candidate["paths"]} ==
                         {tuple(r["old"]) for r in routes}, "damage changed geometry")
                _require(np.array_equal(solve(inp), out), "damage changed solution")
            except ValueError:
                inp[p] = previous
            else:
                successes += 1
                if successes >= desired:
                    break
        # One or two isolated unmarked straight gaps; first train must have one
        # on a retained route, so counting only observed cells is demonstrably wrong.
        candidates = [p for r in routes for p in r["old"][2:-2]
                      if inp[p] == carrier and p in final_cells]
        rng.shuffle(candidates)
        gap_count = 0
        for p in candidates:
            if any(inp[q] != carrier for q in _neighbors(p, old_occupied)):
                continue
            ns = _neighbors(p, old_occupied)
            if len(ns) != 2 or (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) != (2*p[0], 2*p[1]):
                continue
            inp[p] = BACKGROUND
            try:
                candidate = _analyze(inp)
                _require({tuple(r["old"]) for r in candidate["paths"]} ==
                         {tuple(r["old"]) for r in routes}, "gap changed geometry")
                _require(np.array_equal(solve(inp), out), "gap changed solution")
            except ValueError:
                inp[p] = carrier
            else:
                gap_count += 1
                if gap_count >= (1 if first or second else 2):
                    break
        if successes < 2 or ((first or second) and gap_count == 0):
            continue
        if second:
            e = routes[0]["edit"]
            if inp[e["pivot"]] != REPAIR or not any(inp[p] == REPAIR for p in e["removed"]):
                continue
        bend_witness, gap_witness, ray_witness = _phase_witnesses(_analyze(inp))
        if first and not (bend_witness and gap_witness):
            continue
        if second and not ray_witness:
            continue
        # Complete input is retained only locally to guard the exact recovery;
        # it is never returned, cached, or consulted by solve.
        _require(all(complete[p] == carrier for p in old_occupied
                     if p not in {r["old"][0] for r in routes}), "construction error")
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n_train = _count(num_train, "num_train", 2, 10)
    n_test = _count(num_test, "num_test", 1, 4)
    if n_train <= n_test:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(np.arange(1, 8))))
    n_keys = int(rng.integers(2, 4))
    first_direction = _pick(rng, DIRS)
    novel_direction = _pick(rng, [d for d in DIRS if d != first_direction])
    held_out = bool(rng.integers(2))
    config = {"carrier": palette[0], "keys": palette[1:1+n_keys],
              "decorations": palette[1+n_keys:3+n_keys],
              "train_dirs": [first_direction] if held_out else [first_direction, novel_direction],
              "test_dirs": [novel_direction] if held_out else [first_direction, novel_direction]}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", n_train), ("test", n_test)):
        for index in range(count):
            for _ in range(8):
                pair = _make_pair(rng, config, split, index)
                key = _grid_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct scene construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _turns(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def pair_features(pair):
    """Family measurements derived from public grids, not generation metadata."""
    info = _analyze(pair["input"])
    edits = [r["edit"] for r in info["paths"] if r["edit"]]
    old_cells = set().union(*(set(r["old"]) for r in info["paths"]))
    retained = set().union(*(set(r["final"]) for r in info["paths"]))
    removed = set().union(*(set(e["removed"]) for e in edits)) if edits else set()
    geometry = tuple(tuple((p[0]-r["old"][0][0], p[1]-r["old"][0][1])
                           for p in r["old"]) for r in info["paths"])
    return {"shape": info["input"].shape, "carrier": info["carrier"],
            "keys": tuple(sorted(info["keys"])), "tracks": len(info["paths"]),
            "spools": len(info["cards"]), "redirects": len(edits),
            "directions": tuple(e["direction"] for e in edits),
            "periods": tuple(sorted(len(t) for t in info["templates"].values())),
            "tiles": tuple(sorted(info["templates"].values())),
            "turns": tuple(_turns(r["old"]) for r in info["paths"]),
            "old_branch_turns": tuple(_turns([e["pivot"]] + e["removed"]) for e in edits),
            "ray_lengths": tuple(len(e["ray"]) for e in edits),
            "repair_count": len(info["repairs"]), "gaps": len(info["gaps"]),
            "retained_repairs": len(info["repairs"] & retained),
            "removed_repairs": len(info["repairs"] & removed),
            "marked_pivots": sum(e["pivot"] in info["repairs"] for e in edits),
            "new_old_intersections": sum(len(set(e["ray"]) & old_cells) for e in edits),
            "geometries": geometry}


def validate_generated_puzzle(puzzle) -> None:
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
        _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
        _count(len(puzzle["train"]), "num_train", 2, 10)
        _count(len(puzzle["test"]), "num_test", 1, 4)
        _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
        seen, train_info = set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
                inp, out = _grid(pair["input"]), _grid(pair["output"])
                _require(inp.shape == out.shape, "output must retain grid dimensions")
                key = _grid_key(inp)
                _require(key not in seen, "duplicate input")
                seen.add(key)
                info = _analyze(inp)
                prediction = solve(inp)
                _require(np.array_equal(prediction, out), "incorrect routing, phase, stamp, or preservation")
                _require(not np.any(np.isin(out, [8, 9])), "annotations must disappear")
                _require(np.all(out[info["card_mask"]] == 0), "spools must be discarded")
                # Ablate A: repair/reroute but retain monochrome tracks.
                no_a = np.zeros(inp.shape, dtype=np.int8)
                for route in info["paths"]:
                    for p in route["final"]:
                        no_a[p] = info["carrier"]
                    no_a[route["final"][0]] = route["key"]
                _require(not np.array_equal(no_a, out), "spool copying must affect answer")
                if split == "train":
                    train_info.append(info)
        # Even two demonstrations jointly establish transparency, key selection,
        # repairing a retained hole, and deleting a bent, marked old branch.
        _require(any(len(i["keys"]) >= 2 and len({len(t) for t in i["templates"].values()}) >= 2
                     for i in train_info), "training must distinguish keyed periods")
        _require(any(i["gaps"] for i in train_info), "training must witness gap repair")
        transparent = False
        rerouting = False
        phase_evidence = [False, False, False]
        for info in train_info:
            phase_evidence = [a or b for a, b in zip(phase_evidence, _phase_witnesses(info))]
            out = solve(info["input"])
            for route in info["paths"]:
                tile = info["templates"][route["key"]]
                transparent |= any(tile[n % len(tile)] == 0 and out[p] == info["carrier"]
                                   for n, p in enumerate(route["final"]))
                e = route["edit"]
                if e and _turns([e["pivot"]] + e["removed"]) >= 1:
                    no_b = _paint(out.shape, info["carrier"],
                                  [(r["key"], r["old"]) for r in info["paths"]], info["templates"])
                    rerouting |= (bool(set(e["removed"]) & info["repairs"])
                                  and e["pivot"] in info["repairs"]
                                  and not np.array_equal(no_b, out))
        _require(transparent and rerouting, "training must expose transparent copying and bent marked deletion")
        _require(all(phase_evidence), "training must distinguish phase resets and skipping missing cells")
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise AssertionError(str(error)) from error


def self_test() -> None:
    """Bounded semantics, equivariance, negative-label and 42-seed coverage checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    fingerprints = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_grid_key(pair[role]) for split in ("train", "test")
                               for pair in puzzle[split] for role in ("input", "output")))
        fs = [pair_features(pair) for split in ("train", "test") for pair in puzzle[split]]
        records.extend(fs)
        assert len({f["geometries"] for f in fs}) >= 3
        assert len({f["tiles"] for f in fs}) >= 3
        for pair in puzzle["test"]:
            assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
    assert len(fingerprints) == 42
    assert len({r["shape"] for r in records}) >= 45
    assert len({r["geometries"] for r in records}) >= 150
    assert len({r["tiles"] for r in records}) >= 150
    assert {r["carrier"] for r in records} == set(range(1, 8))
    assert {r["tracks"] for r in records} >= {1, 2, 3}
    assert {r["redirects"] for r in records} >= {0, 1, 2}
    assert {d for r in records for d in r["directions"]} == set(DIRS)
    assert {p for r in records for p in r["periods"]} == set(range(3, 8))
    assert max(b for r in records for b in r["old_branch_turns"]) >= 3
    assert min(n for r in records for n in r["ray_lengths"]) == 1
    assert max(n for r in records for n in r["ray_lengths"]) >= 20
    assert all(sum(r[name] for r in records) > 0 for name in
               ("gaps", "retained_repairs", "removed_repairs", "marked_pivots"))
    first = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(np.array_equal(a[role], b[role]) for split in first
               for a, b in zip(first[split], again[split]) for role in a)
    # Whole-scene rotation/reflection must preserve path order and local arrows.
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"] + puzzle["test"]:
            for k in range(4):
                inp = np.rot90(np.fliplr(pair["input"]), k).copy()
                out = np.rot90(np.fliplr(pair["output"]), k).copy()
                assert np.array_equal(solve(inp), out)
    # Explicit mutations exercise phase, transparent slots, removed bends, rays,
    # ordinary background, and reference erasure. Full-grid comparison rejects all.
    query = first["test"][0]
    info = _analyze(query["input"])
    sites = {tuple(map(int, np.argwhere(info["card_mask"])[0])), (0, 0)}
    for route in info["paths"]:
        sites.update(route["final"][:10])
        if route["edit"]:
            sites.add(route["edit"]["removed"][-1])
            sites.add(route["edit"]["ray"][-1])
    for p in sites:
        bad = {"train": first["train"], "test": [dict(query)]}
        bad["test"][0]["output"] = query["output"].copy()
        bad["test"][0]["output"][p] = (int(query["output"][p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted label accepted")
    for a, b in ((2, 1), (10, 4), (7, 3)):
        p = generate_puzzle(107+a+b, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (5, 5),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
