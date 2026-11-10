"""Frame-powered five-step arrow networks. Standard library and NumPy only.

No files, network, saved traces, or parent modules are used at runtime.
"""
from __future__ import annotations

from collections import deque
from itertools import product
import hashlib
import numpy as np


DISTANCE = 5
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
PORTS = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
              if r in (-1, 2) or c in (-1, 2))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _direction(offset):
    return (-1 if offset[0] == -1 else 1 if offset[0] == 2 else 0,
            -1 if offset[1] == -1 else 1 if offset[1] == 2 else 0)


def _endpoint(point, direction):
    return point[0] + 5 * direction[0], point[1] + 5 * direction[1]


def _arrow(tip, direction):
    dr, dc = direction
    if not dr:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif not dc:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return {"tip": tip, "direction": direction,
            "cells": tuple(sorted(_add(tip, p) for p in offsets))}


def _at_hit(point, direction, part):
    relative = _arrow((0, 0), direction)["cells"][part]
    return _arrow((point[0] - relative[0], point[1] - relative[1]), direction)


def _footprint(frame):
    r, c = frame["core"]
    return tuple((r + dr, c + dc) for dr in range(-1, 3) for dc in range(-1, 3))


def _frame_endpoints(frame, all_ports=False):
    r, c = frame["core"]
    return [_endpoint((r + p[0], c + p[1]), _direction(p))
            for p in (PORTS if all_ports else frame["ports"])]


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _parse(input_grid):
    """Recover vocabulary, roles and geometry using only the input pixels."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(counts.argmax())
    _require(int(counts[background]) > grid.size // 2, "background must be a strict majority")
    h, w = grid.shape
    frames = []
    covered = set()
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            color = int(grid[r, c])
            if color == background or not np.all(grid[r:r + 2, c:c + 2] == color):
                continue
            values = [int(grid[r + dr, c + dc]) for dr, dc in PORTS]
            structure = set(values) - {background}
            if len(structure) != 1 or color in structure:
                continue
            ring = next(iter(structure))
            ports = tuple(p for p, v in zip(PORTS, values) if v == background)
            if not 1 <= len(ports) <= 4:
                continue
            frame = {"core": (r, c), "color": color, "ring": ring, "ports": ports}
            footprint = set(_footprint(frame))
            _require(not covered & footprint, "overlapping frames")
            covered |= footprint
            frames.append(frame)
    _require(frames and len({f["ring"] for f in frames}) == 1, "need common frame structure")
    ring = frames[0]["ring"]
    _require(all(f["color"] not in (background, ring) for f in frames), "invalid payload role")
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)} - covered
    arrows = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            neighbors = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)} & remaining
            remaining -= neighbors
            cells |= neighbors
            todo.extend(neighbors)
        _require(len(cells) == 3 and len({int(grid[p]) for p in cells}) == 1,
                 "relays must be separated monochrome arrowheads")
        matches = [_arrow(p, d) for p in sorted(cells) for d in DIRECTIONS
                   if set(_arrow(p, d)["cells"]) == cells]
        _require(len(matches) == 1, "ambiguous arrow")
        arrows.append(matches[0])
    _require(arrows, "no arrow relays")
    ink = {int(grid[a["tip"]]) for a in arrows}
    _require(len(ink) == 1 and not ink & ({background, ring} | {f["color"] for f in frames}),
             "relay ink must be distinct from the visible source roles")
    # Full footprints, including the blank ports, have a one-cell separation.
    objects = [set(_footprint(f)) for f in frames] + [set(a["cells"]) for a in arrows]
    occupied = set()
    for cells in objects:
        _require(not any((r + dr, c + dc) in occupied for r, c in cells
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1)), "touching objects")
        occupied |= cells
    return {"grid": grid, "background": background, "ring": ring,
            "ink": next(iter(ink)), "frames": frames, "arrows": arrows}


def _network(info, all_ports=False):
    arrows = info["arrows"]
    owner = {p: i for i, a in enumerate(arrows) for p in a["cells"]}
    edges = [owner.get(_endpoint(a["tip"], a["direction"])) for a in arrows]
    injections = []
    for f in info["frames"]:
        injections.append([owner[p] for p in _frame_endpoints(f, all_ports) if p in owner])
    return owner, edges, injections


def _reach(info, *, all_ports=False, one_hop=False, undirected=False):
    """Monotone union on source-ID bitsets, with colors interpreted at display."""
    owner, edges, injections = _network(info, all_ports)
    n = len(edges)
    masks = [0] * n
    queue = deque()
    for source, targets in enumerate(injections):
        for target in targets:
            value = masks[target] | (1 << source)
            if value != masks[target]:
                masks[target] = value
                queue.append(target)
    adjacency = [[] if j is None else [j] for j in edges]
    if undirected:
        for i, j in enumerate(edges):
            if j is not None:
                adjacency[j].append(i)
    if not one_hop:
        while queue:
            i = queue.popleft()
            for j in adjacency[i]:
                value = masks[j] | masks[i]
                if value != masks[j]:
                    masks[j] = value
                    queue.append(j)
    return masks, edges, injections


def _colors(info, mask):
    return {f["color"] for i, f in enumerate(info["frames"]) if mask & (1 << i)}


def _display(info, masks, count_origins=False):
    out = info["grid"].copy()
    for arrow, mask in zip(info["arrows"], masks):
        colors = _colors(info, mask)
        if colors:
            conflict = mask.bit_count() > 1 if count_origins else len(colors) > 1
            color = info["ring"] if conflict else next(iter(colors))
            for p in arrow["cells"]:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions are fixed or visible in this one input."""
    info = _parse(input_grid)
    masks, _, _ = _reach(info)
    return _display(info, masks)


def _occupied(arrows, frames):
    return {p for a in arrows for p in a["cells"]} | {p for f in frames for p in _footprint(f)}


def _fits(cells, arrows, frames, limit=28):
    occupied = _occupied(arrows, frames)
    if any((r + dr, c + dc) in occupied for r, c in cells
           for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
        return False
    points = list(occupied) + list(cells)
    return all(max(p[k] for p in points) - min(p[k] for p in points) + 1 <= limit for k in (0, 1))


def _shuffled(rng, values):
    return [values[int(i)] for i in rng.permutation(len(values))]


def _incoming_options(rng, point):
    return _shuffled(rng, [_arrow((point[0] - 5 * d[0], point[1] - 5 * d[1]), d)
                           for d in DIRECTIONS])


def _source_options(rng, arrow, color, kind):
    options = []
    for hit in arrow["cells"]:
        for port in PORTS:
            d = _direction(port)
            if (sum(map(abs, d)) == 1) != (kind == "side"):
                continue
            core = (hit[0] - 5 * d[0] - port[0], hit[1] - 5 * d[1] - port[1])
            options.append({"core": core, "color": color, "ports": (port,)})
    return _shuffled(rng, options)


def _attach_source(rng, arrows, frames, target, hit, length, color, kind):
    """A bounded backtracking branch ending at a chosen cell of target."""
    budget = [360]

    def search(current, point, left):
        if budget[0] <= 0:
            return None
        if not left:
            for frame in _source_options(rng, current[-1], color, kind):
                budget[0] -= 1
                if _fits(_footprint(frame), arrows + current, frames):
                    return current, frame
            return None
        for candidate in _incoming_options(rng, point):
            budget[0] -= 1
            if budget[0] <= 0:
                return None
            if _fits(candidate["cells"], arrows + current, frames):
                for next_point in _shuffled(rng, candidate["cells"]):
                    result = search(current + [candidate], next_point, left - 1)
                    if result is not None:
                        return result
        return None

    return search([], target["cells"][hit], length)


def _make_spine(rng, cycle, length):
    if cycle:
        # Enumerate short exact vector loops; this does not store puzzle grids.
        sequences = [ds for ds in product(DIRECTIONS, repeat=length)
                     if all(sum(d[k] for d in ds) == 0 for k in (0, 1))]
        for ds in _shuffled(rng, sequences):
            arrows, point = [], (0, 0)
            for d in ds:
                candidate = _arrow(point, d)
                if not _fits(candidate["cells"], arrows, []):
                    break
                arrows.append(candidate)
                point = _endpoint(point, d)
            else:
                return arrows
        return None
    arrows = [_arrow((0, 0), DIRECTIONS[int(rng.integers(8))])]
    for _ in range(length - 1):
        point = _endpoint(arrows[-1]["tip"], arrows[-1]["direction"])
        candidates = [_at_hit(point, d, part) for d in DIRECTIONS for part in range(3)]
        for candidate in _shuffled(rng, candidates):
            if _fits(candidate["cells"], arrows, []):
                arrows.append(candidate)
                break
        else:
            return None
    return arrows


def _emitter_points(arrows, frames):
    return {_endpoint(a["tip"], a["direction"]) for a in arrows} | {
        p for f in frames for p in _frame_endpoints(f)}


def _add_decoy(rng, arrows, frames, closed=False):
    """A dormant feeder or an arrow beyond a closed (non-emitting) ring cell."""
    emitted = _emitter_points(arrows, frames)
    candidates = []
    if closed:
        for frame in frames:
            for p in PORTS:
                if p not in frame["ports"]:
                    point = _endpoint(_add(frame["core"], p), _direction(p))
                    candidates.extend(_at_hit(point, d, part) for d in DIRECTIONS for part in range(3))
    else:
        for target in arrows:
            for point in target["cells"]:
                candidates.extend(_incoming_options(rng, point))
    for candidate in _shuffled(rng, candidates):
        if not set(candidate["cells"]) & emitted and _fits(candidate["cells"], arrows, frames):
            arrows.append(candidate)
            return True
    return False


def _extra_port(rng, arrows, frames, index):
    frame = frames[index]
    for port in _shuffled(rng, [p for p in PORTS if p not in frame["ports"]]):
        point = _endpoint(_add(frame["core"], port), _direction(port))
        # An extra opening always powers a visible receiver, not an inert badge.
        candidates = [_at_hit(point, d, part) for d in DIRECTIONS for part in range(3)]
        for candidate in _shuffled(rng, candidates):
            if _fits(candidate["cells"], arrows, frames):
                frame["ports"] += (port,)
                arrows.append(candidate)
                # Continue a small independent arm, where geometry allows.
                for _ in range(int(rng.integers(3))):
                    point = _endpoint(arrows[-1]["tip"], arrows[-1]["direction"])
                    if any(point in a["cells"] for a in arrows):
                        break
                    choices = [_at_hit(point, d, part) for d in DIRECTIONS for part in range(3)]
                    for nxt in _shuffled(rng, choices):
                        if _fits(nxt["cells"], arrows, frames):
                            arrows.append(nxt)
                            break
                    else:
                        break
                return True
    return False


def _render(rng, arrows, frames, roles):
    points = list(_occupied(arrows, frames))
    low = tuple(min(p[k] for p in points) for k in (0, 1))
    extent = tuple(max(p[k] for p in points) - low[k] + 1 for k in (0, 1))
    shape = tuple(int(rng.integers(max(12, n), 31)) for n in extent)
    shift = tuple(int(rng.integers(shape[k] - extent[k] + 1)) - low[k] for k in (0, 1))
    grid = np.full(shape, 0, dtype=np.int8)
    for frame in frames:
        r, c = _add(frame["core"], shift)
        grid[r - 1:r + 3, c - 1:c + 3] = 8
        grid[r:r + 2, c:c + 2] = frame["color"]
        for dr, dc in frame["ports"]:
            grid[r + dr, c + dc] = 0
    for arrow in arrows:
        for p in arrow["cells"]:
            grid[_add(p, shift)] = 7
    grid = np.asarray(roles, dtype=np.int8)[grid]
    grid = np.rot90(grid, int(rng.integers(4)))
    if rng.integers(2):
        grid = np.fliplr(grid)
    return np.ascontiguousarray(grid)


def _features(input_grid):
    info = _parse(input_grid)
    masks, edges, injections = _reach(info)
    owner, _, _ = _network(info)
    incoming_cells = {}
    for i, a in enumerate(info["arrows"]):
        point = _endpoint(a["tip"], a["direction"])
        if point in owner and masks[i]:
            incoming_cells.setdefault(owner[point], set()).add(point)
    cycle_lengths = set()
    for start in range(len(edges)):
        visited, i = {}, start
        while i is not None and i not in visited:
            visited[i] = len(visited)
            i = edges[i]
        if i is not None:
            cycle_lengths.add(len(visited) - visited[i])
    active = {i for i, m in enumerate(masks) if m}
    conflict = {i for i, m in enumerate(masks) if len(_colors(info, m)) > 1}
    shortest = [[None] * len(edges) for _ in info["frames"]]
    for source, targets in enumerate(injections):
        queue = deque((i, 0) for i in targets)
        while queue:
            i, distance = queue.popleft()
            if shortest[source][i] is not None:
                continue
            shortest[source][i] = distance
            if edges[i] is not None:
                queue.append((edges[i], distance + 1))
    unequal_merge = any(len({shortest[s][i] for s in range(len(injections))
                            if shortest[s][i] is not None}) > 1 for i in conflict)
    return info, {"frames": len(info["frames"]), "arrows": len(edges),
                  "shape": tuple(info["grid"].shape), "active": len(active),
                  "dormant": len(edges) - len(active), "conflicts": len(conflict),
                  "same_color_merge": any(m.bit_count() > 1 and len(_colors(info, m)) == 1 for m in masks),
                  "object_merge": any(i in conflict and len(ps) > 1 for i, ps in incoming_cells.items()),
                  "conflict_forward": any(edges[i] in conflict for i in conflict),
                  "dormant_feeder": any(i not in active and edges[i] in active for i in range(len(edges))),
                  "unequal_merge": unequal_merge, "cycles": tuple(sorted(cycle_lengths)),
                  "max_hops": max((d for ds in shortest for d in ds if d is not None), default=0),
                  "port_counts": tuple(len(f["ports"]) for f in info["frames"]),
                  "source_types": tuple(sorted({sum(map(abs, _direction(p)))
                                                for f in info["frames"] for p in f["ports"]})),
                  "directions": tuple(sorted({a["direction"] for a in info["arrows"]})),
                  "masks": masks, "edges": edges, "injections": injections}


def _pair(rng, mode, sources, roles, essential=False):
    # No infinite retries and no retained candidate grids. A candidate has at
    # most 35 temporary arrows and three full 4x4 frames inside a 28x28
    # footprint. Returned scenes are capped at 30 arrows.
    for attempt in range(100):
        cycle = mode == "same" or (mode == "query" and bool(rng.integers(2)))
        spine_length = int(rng.integers(2, 5)) if cycle else int(rng.integers(2, 4))
        arrows = _make_spine(rng, cycle, spine_length)
        if arrows is None:
            continue
        spine = tuple(arrows)
        root, frames = arrows[0], []
        hits = _shuffled(rng, list(range(3)))
        colors = [1] * sources if mode == "same" else list(range(1, sources + 1))
        if mode == "query" and sources == 3:
            # The held-out third source combines taught equal/different-color
            # behavior; it introduces no third distinct-color display category.
            colors = [1, 2, int(rng.integers(1, 3))]
            rng.shuffle(colors)
        lengths = [1, 2] if essential else [int(rng.integers(1, 4)) for _ in range(sources)]
        if sources == 1:
            lengths = lengths[:1]
        success = True
        for source in range(sources):
            target = root
            if not essential and source and rng.random() < .55:
                # Sources may join at successive points along a route instead
                # of every input having one common fan-in site.
                target = spine[int(rng.integers(1, len(spine)))]
            result = _attach_source(rng, arrows, frames, target, hits[source], lengths[source],
                                    colors[source], "side" if source % 2 == 0 else "corner")
            if result is None:
                success = False
                break
            branch, frame = result
            arrows += branch
            frames.append(frame)
        if not success:
            continue
        for i in range(sources):
            for _ in range(int(rng.integers(0, 3))):
                _extra_port(rng, arrows, frames, i)
        if not _add_decoy(rng, arrows, frames, closed=True):
            continue
        if not _add_decoy(rng, arrows, frames):
            continue
        for _ in range(int(rng.integers(3))):
            _add_decoy(rng, arrows, frames, closed=bool(rng.integers(2)))
        if len(arrows) > 30:
            continue
        grid = _render(rng, arrows, frames, roles)
        info, features = _features(grid)
        if features["active"] < 3 or not features["dormant_feeder"]:
            continue
        expected = _display(info, features["masks"])
        # Reconstruct every teaching witness from final pixels, not the builder.
        if essential:
            if mode == "same" and not (features["same_color_merge"] and features["cycles"]):
                continue
            if mode == "mixed" and not (features["object_merge"] and features["conflict_forward"]
                                        and features["unequal_merge"]):
                continue
            alternatives = (("count_origins", "all_ports", "one_hop", "undirected")
                            if mode == "same" else
                            ("all_ports", "one_hop", "undirected", "shortest_only",
                             "coordinate_conflict", "absorb_conflicts"))
            if any(np.array_equal(expected, _alternative(info, name)) for name in alternatives):
                continue
        return {"input": grid, "output": expected}
    raise RuntimeError(f"bounded construction exhausted for {mode}/{sources}")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("example counts must be integers")
        if not low <= value <= high:
            raise ValueError("example count outside allowed range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    preferred_roles = rng.permutation(10)
    specs = [("same", 2, True), ("mixed", 2, True)]
    specs += [(str(rng.choice(("single", "same", "mixed"))), int(rng.integers(1, 3)), False)
              for _ in range(num_train - 2)]
    specs = _shuffled(rng, specs)
    queries = [("query", 3 if i == 0 else int(rng.integers(1, 4)), False) for i in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, items in (("train", specs), ("test", queries)):
        for mode, sources, essential in items:
            for _ in range(12):
                roles = preferred_roles if rng.random() < .3 else rng.permutation(10)
                pair = _pair(rng, mode, sources, roles, essential)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, name):
    if name == "count_origins":
        return _display(info, _reach(info)[0], count_origins=True)
    if name == "absorb_conflicts":
        full, edges, injections = _reach(info)
        masks = [0] * len(edges)
        queue = deque()
        for source, targets in enumerate(injections):
            for i in targets:
                masks[i] |= 1 << source
                queue.append(i)
        while queue:
            i = queue.popleft()
            j = edges[i]
            if j is None or len(_colors(info, full[i])) > 1:
                continue
            value = masks[j] | masks[i]
            if value != masks[j]:
                masks[j] = value
                queue.append(j)
        return _display(info, masks)
    if name == "shortest_only":
        _, edges, injections = _network(info)
        best, masks = [1000] * len(edges), [0] * len(edges)
        queue = deque((i, 0, 1 << source) for source, targets in enumerate(injections) for i in targets)
        while queue:
            i, distance, mask = queue.popleft()
            if distance > best[i]:
                continue
            new = mask if distance < best[i] else masks[i] | mask
            if distance == best[i] and new == masks[i]:
                continue
            best[i], masks[i] = distance, new
            if edges[i] is not None:
                queue.append((edges[i], distance + 1, new))
        return _display(info, masks)
    if name == "coordinate_conflict":
        # A local endpoint-counting interpretation misses union on different
        # cells. Preserve one arriving source at such merges, then propagate.
        owner, edges, injections = _network(info)
        masks = _reach(info)[0]
        point_sources = {}
        for source, f in enumerate(info["frames"]):
            for p in _frame_endpoints(f):
                if p in owner:
                    point_sources[p] = point_sources.get(p, 0) | (1 << source)
        for i, arrow in enumerate(info["arrows"]):
            p = _endpoint(arrow["tip"], arrow["direction"])
            if p in owner:
                point_sources[p] = point_sources.get(p, 0) | masks[i]
        wrong = masks.copy()
        for i, arrow in enumerate(info["arrows"]):
            parts = [point_sources.get(p, 0) for p in arrow["cells"]]
            if len(_colors(info, masks[i])) > 1 and all(len(_colors(info, m)) <= 1 for m in parts):
                wrong[i] = next((m for m in parts if m), 0)
        return _display(info, wrong)
    return _display(info, _reach(info, **{name: True})[0])


ALTERNATIVES = ("one_hop", "undirected", "all_ports", "count_origins",
                "shortest_only", "coordinate_conflict", "absorb_conflicts")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    rejected = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info, f = _features(inp)
            expected = _display(info, f["masks"])
            _require(np.array_equal(expected, out), "incorrect signal union output")
            _require(not np.array_equal(inp, out), "unchanged pair")
            arrow_cells = {p for a in info["arrows"] for p in a["cells"]}
            changed = {tuple(map(int, p)) for p in np.argwhere(inp != out)}
            _require(changed <= arrow_cells and len(changed) == 3 * f["active"], "nonlocal recoloring")
            _require(f["dormant"] > 0 and f["dormant_feeder"], "need unpowered incoming decoys")
            _require(1 <= f["frames"] <= 3 and 1 <= f["arrows"] <= 30, "object budget")
            for i, j in enumerate(f["edges"]):
                if j is not None:
                    _require(f["masks"][i] & ~f["masks"][j] == 0, "signal lost across edge")
            if split == "train":
                rejected |= {name for name in ALTERNATIVES
                             if not np.array_equal(_alternative(info, name), out)}
            records[split].append(f)
    train = records["train"]
    _require(any(f["same_color_merge"] and f["cycles"] for f in train), "teach duplicate colors and feedback")
    _require(any(f["object_merge"] and f["conflict_forward"] and f["unequal_merge"] for f in train),
             "teach object-level union and onward conflict")
    _require(set(ALTERNATIVES) <= rejected,
             f"training leaves shortcuts unchallenged: {sorted(set(ALTERNATIVES) - rejected)}")
    _require(set().union(*(set(f["source_types"]) for f in train)) == {1, 2}, "teach side and corner ports")
    _require(max(f["frames"] for f in train) == 2 and records["test"][0]["frames"] == 3,
             "source-count shift missing")


def _semantic_regressions():
    """Small fixtures with manually assigned expected relay colors."""
    for payload in (2, 3):
        for feedback in (False, True):
            grid = np.zeros((18, 18), dtype=np.int8)
            for core, color, port in (((2, 2), 2, (0, 2)), ((14, 1), payload, (-1, 2))):
                r, c = core
                grid[r - 1:r + 3, c - 1:c + 3] = 8
                grid[r:r + 2, c:c + 2] = color
                grid[r + port[0], c + port[1]] = 0
            arrows = [_arrow((2, 9), (1, 0)), _arrow((7, 9), (0, 1)),
                      _arrow((7, 14), (0, -1) if feedback else (-1, 0)),
                      _arrow((12, 9), (-1, 0)),
                      _arrow((16, 8), (0, 1)), _arrow((16, 13), (0, -1))]
            for a in arrows:
                for p in a["cells"]:
                    grid[p] = 7
            expected = grid.copy()
            for a, color in zip(arrows[:3], (2, 2 if payload == 2 else 8, 2 if payload == 2 else 8)):
                for p in a["cells"]:
                    expected[p] = color
            _require(np.array_equal(solve(grid), expected), "manual relay/merge/feedback fixture")
            _require(expected[4, 9] == 0 and expected[12, 9] == 7,
                     "intermediate space or dormant feeder changed")


def self_test():
    """42 seeds, streaming metrics; no saved grids, traces, logs or file writes."""
    _semantic_regressions()
    shapes, directions, counts, cycles, structures, puzzles = set(), set(), set(), set(), set(), set()
    port_counts, role_colors = set(), set()
    totals = {"pairs": 0, "conflicts": 0, "same_color_merges": 0, "dormant_feeders": 0}
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = hashlib.sha256()
        if seed == 0:
            first = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info, f = _features(grid)
                _require(np.array_equal(solve(grid.copy()), pair["output"]), "solver disagreement")
                shapes.add(f["shape"])
                counts.add((f["frames"], f["arrows"], f["active"], f["dormant"]))
                directions.update(f["directions"])
                cycles.update(f["cycles"])
                port_counts.update(f["port_counts"])
                role_colors.add((info["background"], info["ring"], info["ink"]))
                # A coarse invariant graph profile, not a graph-isomorphism
                # certificate. It ignores palette, canvas size and pose.
                structures.add((tuple(sorted((sum(j == i for j in f["edges"]),
                                              f["masks"][i].bit_count(), f["edges"][i] is None)
                                             for i in range(f["arrows"]))), f["cycles"],
                                tuple(sorted(f["port_counts"]))))
                totals["pairs"] += 1
                totals["conflicts"] += f["conflicts"]
                totals["same_color_merges"] += int(f["same_color_merge"])
                totals["dormant_feeders"] += int(f["dormant_feeder"])
                fingerprint.update(str(grid.shape).encode())
                fingerprint.update(grid.tobytes())
        puzzles.add(fingerprint.hexdigest())
    _require(len(puzzles) == 42 and len(shapes) >= 50 and len(structures) >= 60, "variation collapsed")
    _require(directions == set(DIRECTIONS) and {2, 3, 4} <= cycles, "direction or loop coverage")
    _require(len(counts) >= 50 and {1, 2, 3} <= port_counts and len(role_colors) >= 60, "structural/palette variation")
    repeat = generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], repeat[s])
                 for k in ("input", "output")), "nondeterminism")
    # Palette and spatial covariance, including directions absent from any
    # particular demonstration, follow from visible geometry alone.
    permutation = np.array([9, 5, 1, 7, 0, 8, 3, 2, 6, 4], dtype=np.int8)
    for pair in first["train"] + first["test"]:
        for rotation in range(4):
            inp = np.fliplr(np.rot90(permutation[pair["input"]], rotation))
            out = np.fliplr(np.rot90(permutation[pair["output"]], rotation))
            _require(np.array_equal(solve(inp), out), "geometry/palette covariance failed")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {**totals, "seeds": 42, "distinct_puzzles": len(puzzles), "shapes": len(shapes),
            "graph_signatures": len(structures), "object_count_profiles": len(counts),
            "cycle_lengths": sorted(cycles), "port_counts": sorted(port_counts),
            "role_palettes": len(role_colors), "independent_semantic_verification": False}
