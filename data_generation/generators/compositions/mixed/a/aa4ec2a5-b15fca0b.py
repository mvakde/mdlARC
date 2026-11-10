"""Route-reserved topological wall decoration. Standard library + NumPy only.

No parent modules, files, examples, generation metadata, or RNG are used by solve.
All output-only colors are fixed family conventions (see task_spec.md).
"""
from collections import deque
import hashlib
import json
import time

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
TERMINAL, ROUTE, HALO, HOLLOW, FILL = 3, 4, 2, 8, 6
INPUT_COLORS = (0, 1, 5, 7, 9)
HOST_SEEDS = (0, 830991509, 191203878, 372706148, 667434243, 92959161,
              40141266, 65964304, 386526678, 916832219, 223682637, 112350744,
              66006146, 970512201, 358413419, 107153096, 740491972, 70259032,
              453469379, 495612551, 762027188, 168109805, 837626776, 858307371,
              520875039, 597759904, 779861982, 599212839, 663703979, 39122067,
              114742862, 119408989, 411339190, 916523777, 76542256, 841474876,
              467021351, 713925259, 253311642, 289328243, 336538913, 430807145)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _enclosed(wall, neighbors=N4):
    # Padding is outside the canvas, including when a wall touches an edge.
    blocked = np.pad(wall, 1)
    outside = np.zeros_like(blocked)
    outside[0, 0] = True
    todo = [(0, 0)]
    h, w = blocked.shape
    while todo:
        r, c = todo.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                todo.append((rr, cc))
    return (~blocked & ~outside)[1:-1, 1:-1]


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr, dc in N8])


def _unambiguous(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _bfs(wall, start):
    """Distances, capped shortest-path multiplicities, and one predecessor."""
    h, w = wall.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    ways = np.zeros((h, w), dtype=np.int8)
    previous = {}
    distance[start], ways[start] = 0, 1
    todo = deque([start])
    while todo:
        r, c = todo.popleft()
        nd = int(distance[r, c]) + 1
        for dr, dc in N4:
            q = (r + dr, c + dc)
            rr, cc = q
            if not (0 <= rr < h and 0 <= cc < w) or wall[q]:
                continue
            if distance[q] < 0:
                distance[q], ways[q] = nd, ways[r, c]
                previous[q] = (r, c)
                todo.append(q)
            elif distance[q] == nd:
                ways[q] = min(2, int(ways[q]) + int(ways[r, c]))
    return distance, ways, previous


def _route(wall, start, goal):
    distance, ways, previous = _bfs(wall, start)
    if distance[goal] < 2 or ways[goal] != 1:
        raise ValueError("terminals need a unique shortest route with intermediate cells")
    path = [goal]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or np.count_nonzero(a == TERMINAL) != 2:
        raise ValueError("input needs two color-3 terminals and two scene colors")
    others = [(int(n), int(c)) for c, n in zip(colors, counts) if c != TERMINAL]
    if len(others) != 2 or any(c not in INPUT_COLORS for n, c in others):
        raise ValueError("scene colors must avoid the fixed terminal and decoration roles")
    others.sort(reverse=True)
    if others[0][0] <= others[1][0] or others[1][0] < 3:
        raise ValueError("background must outnumber wall cells; at least three wall cells")
    return others[0][1], others[1][1]


def _analyze(a):
    bg, ink = _roles(a)
    wall, terminal = a == ink, a == TERMINAL
    if not _unambiguous(wall):
        raise ValueError("diagonal-only contacts or leaks are outside this family")
    bodies = _components(wall)
    if len(bodies) != len(_components(wall, N8)):
        raise ValueError("distinct walls cannot touch diagonally")
    labels = np.full(a.shape, -1, dtype=np.int16)
    for i, cells in enumerate(bodies):
        rr, cc = np.array(sorted(cells)).T
        labels[rr, cc] = i
    holes = _enclosed(wall)
    if not np.array_equal(holes, _enclosed(wall, N8)) or np.any(holes & terminal):
        raise ValueError("ambiguous cavities or an enclosed terminal")
    hollow_ids = set()
    cavities = _components(holes)
    for cells in cavities:
        owners = {int(labels[r + dr, c + dc]) for r, c in cells for dr, dc in N4
                  if 0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1]
                  and labels[r + dr, c + dc] >= 0}
        if len(owners) != 1:
            raise ValueError("each cavity must belong to exactly one original wall body")
        hollow_ids.update(owners)
    hollow = np.isin(labels, sorted(hollow_ids))
    endpoints = list(map(tuple, np.argwhere(terminal).tolist()))
    path = _route(wall, *endpoints)
    route = np.zeros(a.shape, dtype=bool)
    rr, cc = np.array(path[1:-1]).T
    route[rr, cc] = True
    # This protected set parameterizes the decoration, rather than becoming ink.
    halo_base = _dilate(wall) & ~wall & ~holes
    halo = halo_base & ~terminal & ~route
    return dict(bg=bg, ink=ink, wall=wall, terminal=terminal, bodies=bodies,
                labels=labels, holes=holes, cavities=cavities, hollow=hollow,
                solid=wall & ~hollow, path=path, route=route, halo=halo,
                halo_base=halo_base, far=(a == bg) & ~holes & ~route & ~halo)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from the input alone, without mutation or latent state."""
    a = _grid(input_grid)
    z = _analyze(a)
    out = a.copy()
    out[z["hollow"]] = HOLLOW
    out[z["holes"]] = FILL
    out[z["halo"]] = HALO
    out[z["route"]] = ROUTE
    return out


def _body(rng, height, width, kind, multiple=False, irregular=False, sculpt=True):
    """A navigational obstacle and a topological specimen in the same mask."""
    a = np.ones((height, width), dtype=bool)
    if kind in ("hollow", "open"):
        if multiple:
            a[1:3, 1:width - 1] = False
            a[height - 3:height - 1, 1:width - 1] = False
            if irregular:
                a[2, width - 2] = True
        else:
            rh = int(rng.integers(2 if irregular else 1, height - 2))
            cw = int(rng.integers(2 if irregular else 1, width - 1))
            r = int(rng.integers(1, height - rh))
            c = int(rng.integers(1, width - cw))
            a[r:r + rh, c:c + cw] = False
            if irregular:
                a[r + rh - 1, c + cw - 1] = True
        if kind == "open":
            # Exactly one orthogonal door, leaving a connected C/U-like body.
            r, c = np.argwhere(~a)[0]
            if rng.integers(2):
                a[r, :c + 1] = False
            else:
                right = np.flatnonzero(~a[r])[-1]
                a[r, right:] = False
    if sculpt:
        protected_holes = _enclosed(a)
        for _ in range(int(rng.integers(1, 5))):
            b = a.copy()
            rh, cw = int(rng.integers(1, 4)), int(rng.integers(1, width - 1))
            top, left = bool(rng.integers(2)), bool(rng.integers(2))
            rs = slice(0, rh) if top else slice(height - rh, height)
            cs = slice(0, cw) if left else slice(width - cw, width)
            b[rs, cs] = False
            if (len(_components(b)) == 1 and _unambiguous(b)
                    and np.array_equal(_enclosed(b), protected_holes)):
                a = b
    return a


def _bars(rng, n, anchor=None, fallback_code=None):
    h = 15 if fallback_code is not None else int(rng.integers(11, 25))
    widths = [5] * n if fallback_code is not None else list(map(int, rng.integers(4, 7, size=n)))
    gaps = [1] * (n - 1) if fallback_code is not None else list(map(int, rng.choice([1, 1, 1, 2, 3], size=n - 1)))
    margin = 2 if fallback_code is not None else 1
    w = 2 * margin + sum(widths) + sum(gaps)
    if w > 30:
        return None
    wall = np.zeros((h, w), dtype=bool)
    top_first = True if fallback_code is not None else bool(rng.integers(2))
    kinds = [str(rng.choice(["solid", "open", "hollow"])) for _ in range(n)]
    if anchor == "contrast":
        kinds = ["open", "hollow"]
    elif n >= 2:
        kinds[0], kinds[-1] = "open", "hollow"
    else:
        kinds[0] = "hollow"
    c = margin
    for i, width in enumerate(widths):
        top = top_first == (i % 2 == 0)
        clearance = 1 if fallback_code is not None else int(rng.integers(1, 4))
        length = h - clearance
        # Training anchor 2 and every test expose multiple/irregular cavities.
        multi = anchor == "complex" and kinds[i] == "hollow"
        irregular = multi or (anchor is None and bool(rng.random() < .4))
        if fallback_code is None:
            body = _body(rng, length, width, kinds[i], multi, irregular, sculpt=True)
        else:
            # Thin walls guarantee background dominance, even with four barriers.
            body = np.ones((length, width), dtype=bool)
            body[1:-1, 1:-1] = False
            if i == 0 and n > 1:
                body[3, 0] = False
            if i == n - 1:
                if anchor == "complex":
                    body[6, 1:-1] = True
                    body[2, -2] = True
                r = 1 + fallback_code // 2
                cc = 1 if fallback_code % 2 else width - 2
                body[r, cc] = True
        rs = slice(0, length) if top else slice(h - length, h)
        wall[rs, c:c + width] = body
        c += width + (gaps[i] if i < n - 1 else 0)
    if fallback_code is not None:
        start = (2 if n == 1 else 0, margin - 1)
        goal = (3 if n == 1 else 0, w - margin)
    else:
        start = (int(rng.integers(h)), 0)
        goal = (int(rng.integers(h)), w - 1)
    return wall, start, goal


def _islands(rng, n, complex_body=False):
    # Free placement breaks the parallel-bar layout. Endpoints are adjacent
    # to opposite sides of one sampled body; other bodies may steer its route.
    h, w = int(rng.integers(17, 29)), int(rng.integers(17, 30))
    occupied = np.zeros((h, w), dtype=bool)
    wall = occupied.copy()
    boxes = []
    for i in range(n):
        bh, bw = int(rng.integers(8, 12)), int(rng.integers(4, 8))
        kind = "hollow" if i == 0 else str(rng.choice(["solid", "open", "hollow"]))
        body = _body(rng, bh, bw, kind, complex_body and i == 0,
                     complex_body and i == 0, True)
        for _ in range(32):
            r, c = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
            if not occupied[r:r + bh, c:c + bw].any():
                wall[r:r + bh, c:c + bw] = body
                occupied[r - 1:r + bh + 1, c - 1:c + bw + 1] = True
                boxes.append((r, c, bh, bw))
                break
        else:
            return None
    r, c, bh, bw = boxes[0]
    sr, gr = r + int(rng.integers(1, bh - 1)), r + int(rng.integers(1, bh - 1))
    return wall, (sr, c - 1), (gr, c + bw)


def _render_input(candidate, bg, ink, rng, orient=True):
    if candidate is None:
        return None
    wall, start, goal = candidate
    if wall[start] or wall[goal] or wall.sum() >= wall.size - wall.sum() - 2:
        return None
    a = np.where(wall, ink, bg).astype(np.int8)
    a[start] = a[goal] = TERMINAL
    if orient:
        a = np.rot90(a, int(rng.integers(4)))
        if rng.integers(2):
            a = a[:, ::-1]
    return a.copy()


def _features(a, z=None):
    z = _analyze(a) if z is None else z
    path = z["path"]
    steps = [(v[0] - u[0], v[1] - u[1]) for u, v in zip(path, path[1:])]
    hole_counts = [0] * len(z["bodies"])
    for cells in z["cavities"]:
        for r, c in sorted(cells):
            adjacent = [int(z["labels"][r + dr, c + dc]) for dr, dc in N4
                        if 0 <= r + dr < a.shape[0] and 0 <= c + dc < a.shape[1]
                        and z["labels"][r + dr, c + dc] >= 0]
            if adjacent:
                hole_counts[adjacent[0]] += 1
                break
    irregular = any(len(cells) != (max(r for r, c in cells) - min(r for r, c in cells) + 1)
                    * (max(c for r, c in cells) - min(c for r, c in cells) + 1)
                    for cells in z["cavities"])
    open_bay = False
    for body in z["bodies"]:
        rr, cc = np.array(sorted(body)).T
        if z["hollow"][rr[0], cc[0]]:
            continue
        for axis in (0, 1):
            groups = {}
            for r, c in body:
                groups.setdefault(r if axis == 0 else c, []).append(c if axis == 0 else r)
            open_bay |= any(max(v) - min(v) + 1 > len(v) for v in groups.values())
    return {"shape": list(a.shape), "objects": len(z["bodies"]),
            "hollow_objects": int(sum(any(z["hollow"][r, c] for r, c in b) for b in z["bodies"])),
            "holes": len(z["cavities"]), "multiple": max(hole_counts, default=0) > 1,
            "irregular": bool(irregular), "open_bay": bool(open_bay),
            "route_cells": len(path) - 2, "turns": sum(u != v for u, v in zip(steps, steps[1:])),
            "detour": len(path) - 1 - sum(abs(x - y) for x, y in zip(path[0], path[-1])),
            "route_halo_overlap": int(np.count_nonzero(z["route"] & z["halo_base"])),
            "route_as_wall_extra_halo": bool((_dilate(z["route"]) & z["far"]).any()),
            "terminal_halo_overlap": int(np.count_nonzero(z["terminal"] & z["halo_base"])),
            "clipped": bool(z["wall"][0].any() or z["wall"][-1].any()
                            or z["wall"][:, 0].any() or z["wall"][:, -1].any()),
            "interior_terminals": any(0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1
                                      for r, c in (path[0], path[-1]))}


def _signature(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(name + " has an illegal count")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, "num_train"), _count(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg, ink = map(int, rng.choice(INPUT_COLORS, 2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            anchor = ("contrast" if i == 0 else "complex" if i == 1 else None) if split == "train" else "complex"
            for attempt in range(128):
                if anchor == "contrast":
                    candidate = _bars(rng, 2, anchor)
                elif anchor == "complex" and split == "train":
                    candidate = _islands(rng, 1, True)
                else:
                    n = int(rng.integers(3, 5)) if split == "test" else int(rng.integers(1, 4))
                    candidate = (_islands(rng, n, True) if rng.random() < .25
                                 else _bars(rng, n, anchor))
                a = _render_input(candidate, bg, ink, rng)
                if a is None or _signature(a) in seen:
                    continue
                try:
                    z = _analyze(a)
                except ValueError:
                    continue
                f = _features(a, z)
                if not z["halo"].any() or not z["far"].any() or f["turns"] < 2 or f["detour"] < 2:
                    continue
                if not f["route_halo_overlap"] or not f["terminal_halo_overlap"]:
                    continue
                if anchor == "contrast" and not (z["solid"].any() and z["hollow"].any() and f["open_bay"]):
                    continue
                if anchor == "complex" and not (f["multiple"] and f["irregular"]):
                    continue
                break
            else:
                # Finite, rule-preserving fallback. At most 14 pairs are requested.
                # Different reflected/oriented bodies and pockets supply >14 inputs.
                for code in range(64):
                    n = 2 if anchor == "contrast" else 1 if split == "train" and i == 1 else 3 + code % 2
                    candidate = _bars(rng, n, anchor, code % 16)
                    a = _render_input(candidate, bg, ink, rng, orient=False)
                    a = np.rot90(a, code // 16).copy()
                    if _signature(a) not in seen:
                        break
                else:
                    raise RuntimeError("exhausted finite distinct-input fallback")
            seen.add(_signature(a))
            puzzle[split].append({"input": a, "output": solve(a)})
    # Teach both anchors, but do not expose their construction order as a cue.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, roles, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "dimensions changed")
            signature = _signature(a)
            _require(signature not in seen, "duplicate inputs")
            seen.add(signature)
            z = _analyze(a)
            roles.add((z["bg"], z["ink"]))
            f = _features(a, z)
            features[split].append(f)
            # Object-local enclosure cross-check catches collective/nested holes.
            local_union = np.zeros(a.shape, dtype=bool)
            for body in z["bodies"]:
                rr, cc = np.array(sorted(body)).T
                r, c = int(rr.min()), int(cc.min())
                mask = np.zeros((int(rr.max()) - r + 1, int(cc.max()) - c + 1), dtype=bool)
                mask[rr - r, cc - c] = True
                local = _enclosed(mask)
                hr, hc = np.nonzero(local)
                _require(not z["wall"][hr + r, hc + c].any(), "nested wall")
                _require(not local_union[hr + r, hc + c].any(), "overlapping cavity ownership")
                local_union[hr + r, hc + c] = True
                _require(bool(local.any()) == bool(z["hollow"][rr[0], cc[0]]), "whole-body classification")
            _require(np.array_equal(local_union, z["holes"]), "collective enclosure")
            # Dual distance characterization checks the exact selected set.
            ds, _, _ = _bfs(z["wall"], z["path"][0])
            dg, _, _ = _bfs(z["wall"], z["path"][-1])
            shortest = (ds >= 0) & (dg >= 0) & (ds + dg == ds[z["path"][-1]])
            _require(int(shortest.sum()) == len(z["path"]), "shortest set is not a unique chain")
            _require(np.array_equal(shortest & ~z["terminal"], z["route"]), "route mask mismatch")
            for role, color in (("solid", z["ink"]), ("hollow", HOLLOW), ("holes", FILL),
                                ("halo", HALO), ("route", ROUTE), ("terminal", TERMINAL), ("far", z["bg"])):
                _require(np.all(out[z[role]] == color), "wrong output zone: " + role)
            _require(z["far"].any() and z["halo"].any(), "need visible off-route exterior")
            _require(f["turns"] >= 2 and f["detour"] >= 2, "walls must steer the route")
            _require(f["route_halo_overlap"] > 0 and f["terminal_halo_overlap"] > 0, "priority must matter")
    _require(len(roles) == 1, "scene colors changed within puzzle")
    train = features["train"]
    _require(any(f["open_bay"] and f["hollow_objects"] for f in train), "open/closed teaching contrast missing")
    _require(any(f["multiple"] and f["irregular"] for f in train), "complex cavities not demonstrated")
    _require(any(f["route_as_wall_extra_halo"] for f in train), "training must distinguish treating route as wall")
    _require(min(f["objects"] for f in train) < min(f["objects"] for f in features["test"]), "count recombination missing")


def _puzzle_hash(p):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in p[split]:
            for key in ("input", "output"):
                a = np.asarray(pair[key], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _regressions():
    # Tied and absent routes fail closed, rather than selecting a BFS tie by order.
    for rows in (("030", "010", "030"), ("131", "111", "131")):
        a = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        try:
            _route(a == 1, (0, 1), (2, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("tied/unreachable route was accepted")
    # Closing an originally open wall bay with its halo does not fill the bay.
    opened = np.ones((9, 7), dtype=bool)
    opened[1:8, 1:6] = False
    opened[4, 0] = False
    _require(not _enclosed(opened).any(), "open door incorrectly enclosed")
    _require(_enclosed(opened | _dilate(opened)).any(), "fixture should close after halo")
    # Exercise the two essential fallback anchors directly, without RNG retries.
    rng = np.random.default_rng(123)
    for anchor, n in (("contrast", 2), ("complex", 1), ("complex", 4)):
        a = _render_input(_bars(rng, n, anchor, 0), 0, 1, rng, orient=False)
        z = _analyze(a)
        f = _features(a, z)
        _require(z["halo"].any() and z["far"].any(), "fallback lost visible output zones")
        _require(f["detour"] > 0 and f["route_halo_overlap"] > 0, "fallback lost interaction")
        if anchor == "complex":
            _require(f["multiple"] and f["irregular"], "fallback lost teaching cavities")


def self_test():
    """Stream 42 seeds; no payload files, caches, or growing execution logs."""
    started = time.perf_counter()
    _regressions()
    hashes, shapes, layouts, palettes, counts = (set() for _ in range(5))
    measured = {k: 0 for k in ("multiple", "irregular", "open_bay", "clipped", "interior_terminals")}
    pair_count = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        digest = _puzzle_hash(p)
        _require(digest == _puzzle_hash(generate_puzzle(seed)), "nondeterministic seed")
        hashes.add(digest)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                before = a.copy()
                _require(np.array_equal(solve(a), pair["output"]), "solve disagreement")
                _require(np.array_equal(a, before), "solve mutated input")
                z = _analyze(a)
                f = _features(a, z)
                shapes.add(a.shape)
                layouts.add(_signature(a == z["ink"]))
                palettes.add((z["bg"], z["ink"]))
                counts.add(f["objects"])
                for k in measured:
                    measured[k] += int(f[k])
                pair_count += 1
    _require(len(hashes) == 42 and len(layouts) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 30 and len(palettes) >= 12 and {1, 2, 3, 4} <= counts, "missing variation axis")
    _require(all(v >= 10 for v in measured.values()), "missing measured family features")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123456 + nt * 10 + nq, nt, nq)
            _require((len(p["train"]), len(p["test"])) == (nt, nq), "count request ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Reject corruption in each zone, including route/halo conflicts.
    p = generate_puzzle(0)
    for role in ("solid", "hollow", "holes", "halo", "route", "terminal", "far"):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        for pair in q["train"]:
            mask = _analyze(pair["input"])[role]
            if mask.any():
                r, c = np.argwhere(mask)[0]
                pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
                break
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted corrupted " + role)
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(hashes),
            "shapes": len(shapes), "wall_layouts": len(layouts), "palettes": len(palettes),
            "object_counts": sorted(counts), "features": measured,
            "seconds": round(time.perf_counter() - started, 3)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
