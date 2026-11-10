"""Crossing rails: sparse reconstruction controls a rigid arrow's journey.

All semantic decisions are made from the input bitmap. Generation samples
geometry; it neither imports parent generators nor supplies solver metadata.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr == 0) != (dc == 0), "nonorthogonal leg")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _line(a, b):
    d = DIRECTIONS[_direction(a, b)]
    length = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + k*d[0], a[1] + k*d[1]) for k in range(length + 1)]


def _path(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    _require(len(result) == len(set(result)), "self-intersecting rail")
    return result


def _components(points):
    remaining, groups = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        group.add(q)
                        stack.append(q)
        groups.append(group)
    return groups


def _arrow_geometry(cells):
    """Find a symmetric pointed mask from its cells, without a motif lookup."""
    matches = []
    for tip in sorted(cells):
        for facing in range(4):
            offsets = {_rotate((r-tip[0], c-tip[1]), 1-facing) for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if not any(r for r, c in offsets):
                continue
            if {(-r, c) for r, c in offsets} == offsets:
                matches.append((tuple(sorted(offsets)), tip, facing))
    _require(len(matches) == 1, "arrow needs a unique symmetric axis and projecting tip")
    return matches[0]


def _mask_cells(mask, tip, facing):
    return {(tip[0]+r, tip[1]+c) for r, c in
            (_rotate(p, facing-1) for p in mask)}


def _candidate_edges(points):
    vertices = set(points)
    edges = []
    for i, a in enumerate(points):
        for j in range(i+1, len(points)):
            b = points[j]
            if a[0] != b[0] and a[1] != b[1]:
                continue
            pixels = set(_line(a, b))
            if not (pixels - {a, b}) & vertices:
                edges.append((i, j, pixels))
    return edges


def _infer_chain(points, entry):
    """Unique simple full open path from the visibly selected entry landmark.

    Only nearest collinear landmarks are candidate neighbors. Nonconsecutive
    chords remain candidates for the search, but cannot be drawn by default.
    Orthogonal continuous crossings coincide with integer raster cells.
    """
    points = sorted(points)
    n = len(points)
    _require(2 <= n <= 10 and entry in points, "bad landmark set")
    edges = _candidate_edges(points)
    adjacency = [[] for _ in points]
    for k, (i, j, _) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    solutions, budget = [], [0]

    def visit(order, selected):
        budget[0] += 1
        _require(budget[0] <= 20000, "topology search budget exceeded")
        if len(solutions) > 1:
            return
        if len(order) == n:
            solutions.append((order, selected))
            return
        i = order[-1]
        for j, k in adjacency[i]:
            if j in order:
                continue
            pixels = edges[k][2]
            valid = True
            for old in selected:
                a, b, prior = edges[old]
                shared = {points[i], points[j]} & {points[a], points[b]}
                if pixels & prior != shared:
                    valid = False
                    break
            if valid:
                visit(order + [j], selected + [k])

    visit([points.index(entry)], [])
    _require(len(solutions) == 1, "landmarks do not give one open rail from the entry")
    order, selected = solutions[0]
    ordered = [points[i] for i in order]
    rail = _path(ordered)
    all_pixels = set().union(*(e[2] for e in edges))
    return ordered, len(edges)-len(selected), all_pixels-set(rail)


def _make_tracks(points, offset):
    source = _path(points)
    tracks = [source, [(r+offset[0], c+offset[1]) for r, c in source]]
    indices = [{p: i for i, p in enumerate(t)} for t in tracks]
    crossings = set(tracks[0]) & set(tracks[1])
    for p in crossings:
        tangents = []
        for t, index in zip(tracks, indices):
            i = index[p]
            _require(0 < i < len(t)-1, "an endpoint touches the other rail")
            before = (p[0]-t[i-1][0], p[1]-t[i-1][1])
            after = (t[i+1][0]-p[0], t[i+1][1]-p[1])
            _require(before == after, "a corner touches the other rail")
            tangents.append(after)
        _require(sum(a*b for a, b in zip(*tangents)) == 0,
                 "rails overlap instead of crossing transversely")
    return tracks, indices, crossings


def _travel(points, offset, clockwise=True):
    """Traverse the reconstructed two-rail network, including reverse travel."""
    tracks, indices, crossings = _make_tracks(points, offset)
    rail, index, step = 0, 0, 1
    direction = _direction(points[0], points[1])
    states, seen, switches = [(tracks[0][0], direction)], set(), []
    while 0 <= index+step < len(tracks[rail]):
        state = rail, index, step
        _require(state not in seen, "nonterminating journey")
        seen.add(state)
        p, q = tracks[rail][index], tracks[rail][index+step]
        direction = _direction(p, q)
        # Rotations happen around the tip before the next unit move.
        states.extend(((p, direction), (q, direction)))
        index += step
        if q in crossings:
            wanted = (direction + (1 if clockwise else -1)) % 4
            rail = 1-rail
            index = indices[rail][q]
            choices = [s for s in (-1, 1)
                       if _direction(q, tracks[rail][index+s]) == wanted]
            _require(len(choices) == 1, "ambiguous crossing")
            step = choices[0]
            switches.append(q)
    return {"tip": tracks[rail][index], "facing": direction,
            "exit": (rail, step), "states": states, "switches": switches,
            "tracks": tracks, "crossings": crossings}


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4 and int(max(counts)) > grid.size//2,
             "input needs a majority background and three distinct visible roles")
    background = int(colors[np.argmax(counts)])
    groups = {int(c): {tuple(map(int, p)) for p in np.argwhere(grid == c)}
              for c in colors if c != background}
    movers = [(c, ps) for c, ps in groups.items()
              if len(ps) >= 3 and len(_components(ps)) == 1]
    _require(len(movers) == 1, "mover must be the unique compact ink component")
    ink, mover = movers[0]
    mask, tip, facing = _arrow_geometry(mover)
    remaining = {c: ps for c, ps in groups.items() if c != ink}
    singleton = [(c, ps) for c, ps in remaining.items() if len(ps) == 1]
    _require(len(singleton) == 1, "companion marker must have its own singleton color")
    companion, marker_group = singleton[0]
    marker = next(iter(marker_group))
    source = next(c for c in remaining if c != companion)
    vertices = remaining[source]
    _require(2 <= len(vertices) <= 10, "bad source vertex count")
    _require(all(max(abs(a[0]-b[0]), abs(a[1]-b[1])) >= 3
                 for a in vertices for b in vertices if a != b),
             "source landmarks must be separated")
    anchors = [v for v in vertices
               if max(abs(marker[0]-v[0]), abs(marker[1]-v[1])) == 1]
    _require(len(anchors) == 1, "marker association is ambiguous")
    anchor = anchors[0]
    offset = marker[0]-anchor[0], marker[1]-anchor[1]
    _require(offset in OFFSETS, "companion uses a diagonal unit displacement")
    dr, dc = DIRECTIONS[facing]
    ray = [v for v in vertices
           if (v[0]-tip[0])*dc == (v[1]-tip[1])*dr
           and (v[0]-tip[0])*dr+(v[1]-tip[1])*dc > 0]
    _require(bool(ray), "arrow does not point toward an entry landmark")
    entry = min(ray, key=lambda p: abs(p[0]-tip[0])+abs(p[1]-tip[1]))
    points, chords, rejected = _infer_chain(vertices, entry)
    _require(_direction(points[0], points[1]) == facing,
             "first rail leg must continue the launch direction")
    trip = _travel(points, offset)
    tracks = trip["tracks"]
    _require(marker in set(tracks[1])-set(tracks[0]), "marker is hidden by source rail")
    launch = _line(tip, entry)
    _require(not set(launch[:-1]) & (set(tracks[0]) | set(tracks[1])),
             "launch meets a rail before its entry landmark")
    _require(not mover & (set(tracks[0]) | set(tracks[1])), "initial arrow covers a rail")
    occupied = set(tracks[0]) | set(tracks[1])
    states = [(p, facing) for p in launch] + trip["states"]
    for p, d in states:
        occupied.update(_mask_cells(mask, p, d))
    _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in occupied),
             "rail or moving mask goes out of bounds")
    return dict(grid=grid, background=background, source=source, companion=companion,
                ink=ink, marker=marker, offset=offset, points=points, chords=chords,
                rejected=rejected, mask=mask, initial_tip=tip, initial_facing=facing,
                trip=trip)


def _render(info, tip=None, facing=None):
    out = np.full(info["grid"].shape, info["background"], dtype=np.int8)
    tracks = info["trip"]["tracks"]
    for p in tracks[1]:
        out[p] = info["companion"]
    for p in tracks[0]:
        out[p] = info["source"]
    if tip is None:
        tip, facing = info["trip"]["tip"], info["trip"]["facing"]
    for p in _mask_cells(info["mask"], tip, facing):
        out[p] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete both rails and return the rigid mover at its reachable rail end."""
    return _render(_decode(input_grid))


def _make_mask(span, extension, solid, tail):
    cells = {(0, -k) for k in range(extension)}
    for depth in range(1, span+1):
        rows = range(-depth, depth+1) if solid else (-depth, depth)
        cells.update((r, -extension+1-depth) for r in rows)
    if tail:
        cells.update((r, -extension-span) for r in (-span, span))
    return tuple(sorted(cells))


MASKS = tuple(dict.fromkeys(
    _make_mask(span, extension, solid, tail)
    for span in (1, 2) for extension in (1, 2)
    for solid in (False, True) for tail in (0, 1)
    if span+extension-1+tail <= 3
))


def _transform(points, turns, reflection=False):
    return [_rotate((r, -c if reflection else c), turns) for r, c in points]


def _random_chain(rng, count):
    """Incremental nonlattice orthogonal walks, with optional chord alignments."""
    points, pixels = [(0, 0)], {(0, 0)}
    for _ in range(count-1):
        options = []
        for d, (dr, dc) in enumerate(DIRECTIONS):
            if len(points) > 1 and d == (_direction(points[-2], points[-1])+2) % 4:
                continue
            lengths = set(range(3, 9))
            # Deliberately offer old coordinates: chords can be informative.
            lengths.update((q[0]-points[-1][0])*dr+(q[1]-points[-1][1])*dc
                           for q in points[:-1])
            for length in sorted(lengths):
                if not 3 <= length <= 9:
                    continue
                p = points[-1][0]+dr*length, points[-1][1]+dc*length
                if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) < 3 for q in points):
                    continue
                link = set(_line(points[-1], p))
                if link & pixels != {points[-1]}:
                    continue
                proposed = points + [p]
                if any(max(q[axis] for q in proposed)-min(q[axis] for q in proposed) > 22
                       for axis in (0, 1)):
                    continue
                options.append((p, link))
        if not options:
            return None
        p, link = options[int(rng.integers(len(options)))]
        points.append(p)
        pixels.update(link)
    return points


def _witness_chain(rng):
    w, h = int(rng.integers(7, 12)), int(rng.integers(7, 12))
    middle = int(rng.integers(3, h-2))
    inward = int(rng.integers(3, w-2))
    return [(0, 0), (0, w), (h, w), (h, 0), (middle, 0), (middle, inward)]


def _prepare_pair(rng, points, offset, mask):
    ordered, _, _ = _infer_chain(points, points[0])
    _require(ordered == points, "sampled rail differs from visible inferred rail")
    trip = _travel(points, offset)
    facing = _direction(points[0], points[1])
    dr, dc = DIRECTIONS[facing]
    gap = int(rng.integers(3, 5))
    tip = points[0][0]-gap*dr, points[0][1]-gap*dc
    mover = _mask_cells(mask, tip, facing)
    rails = set(trip["tracks"][0]) | set(trip["tracks"][1])
    _require(not rails & mover, "initial mover covers rails")
    launch = _line(tip, points[0])
    _require(not rails & set(launch[:-1]), "blocked launch")
    markers = [(p[0]+offset[0], p[1]+offset[1]) for p in points
               if (p[0]+offset[0], p[1]+offset[1]) not in set(trip["tracks"][0])
               and (p[0]+offset[0], p[1]+offset[1]) not in mover]
    _require(bool(markers), "no visible companion marker")
    marker = markers[int(rng.integers(len(markers)))]
    occupied = rails | mover
    for p, d in [(p, facing) for p in launch] + trip["states"]:
        occupied.update(_mask_cells(mask, p, d))
    lo = tuple(min(p[a] for p in occupied) for a in (0, 1))
    extent = tuple(max(p[a] for p in occupied)-lo[a]+1 for a in (0, 1))
    _require(max(extent) <= 30, "moving geometry too large")
    extras = [int(rng.integers(0, min(6, 30-n)+1)) for n in extent]
    before = [int(rng.integers(0, e+1)) for e in extras]
    shift = before[0]-lo[0], before[1]-lo[1]
    moved = lambda p: (p[0]+shift[0], p[1]+shift[1])
    shape = extent[0]+extras[0], extent[1]+extras[1]
    background, source, companion, ink = map(int, rng.choice(10, 4, replace=False))
    grid = np.full(shape, background, dtype=np.int8)
    for p in points:
        grid[moved(p)] = source
    grid[moved(marker)] = companion
    for p in mover:
        grid[moved(p)] = ink
    # Labels are rasterized from sampled track geometry and the sampled mask.
    # The decoder separately reconstructs topology, palette and arrow geometry.
    out = np.full(shape, background, dtype=np.int8)
    for p in trip["tracks"][1]:
        out[moved(p)] = companion
    for p in trip["tracks"][0]:
        out[moved(p)] = source
    for p in _mask_cells(mask, trip["tip"], trip["facing"]):
        out[moved(p)] = ink
    info = _decode(grid)
    _require(np.array_equal(_render(info), out), "input reconstruction disagrees")
    return {"input": grid, "output": out}, info


def _build_pair(rng, kind, mask):
    count = int(rng.integers(7, 11)) if kind == "query" else int(rng.integers(2, 7))
    reverse_query = kind == "query" and rng.random() < .35
    for attempt in range(240):
        if kind == "reverse_demo":
            a, b = map(int, rng.integers(4, 10, size=2))
            points, offset = [(0, 0), (0, a), (b, a)], (1, 1)
            reflection = False
        elif kind == "chord_demo":
            points, offset = _witness_chain(rng), (-1, int(rng.choice((-1, 1))))
            reflection = False
        else:
            points = _random_chain(rng, count)
            if points is None:
                continue
            offset = OFFSETS[int(rng.integers(4))]
            reflection = bool(rng.integers(2))
        turns = int(rng.integers(4))
        points = _transform(points, turns, reflection)
        offset = _transform([offset], turns, reflection)[0]
        try:
            pair, info = _prepare_pair(rng, points, offset, mask)
            trip = info["trip"]
            if kind == "reverse_demo":
                _require(trip["exit"] == (1, -1) and len(trip["switches"]) == 1,
                         "reverse demonstration must exit the companion's entry end")
            if kind == "chord_demo":
                _require(bool(info["rejected"]) and len(trip["switches"]) >= 2
                         and trip["exit"] == (0, 1), "need chord rejection and a source exit")
            if kind == "query":
                ds = [_direction(a, b) for a, b in zip(points, points[1:])]
                _require(sum(a != b for a, b in zip(ds, ds[1:])) >= 5,
                         "query needs a longer turning rail")
                if reverse_query:
                    _require(trip["exit"] == (1, -1) and len(trip["switches"]) == 1
                             and len(trip["crossings"]) >= 2,
                             "reverse query needs unused crossings")
                else:
                    _require(len(trip["switches"]) >= 2, "query needs multiple switches")
            return pair
        except ValueError:
            continue
    # A guaranteed spacious staircase fallback for ordinary/query examples.
    # It is used only after bounded attempts at irregular walks.
    _require(kind in ("ordinary", "query"), "teaching demonstration construction exhausted")
    for attempt in range(24):
        points = [(0, 0)]
        for k in range(count-1):
            length = int(rng.integers(3, 5))
            dr, dc = (0, 1) if k % 2 == 0 else (1, 0)
            points.append((points[-1][0]+dr*length, points[-1][1]+dc*length))
        turns = int(rng.integers(4))
        try:
            fallback_offset = (1, 1) if reverse_query else (-1, -1)
            pair, info = _prepare_pair(rng, _transform(points, turns), _rotate(fallback_offset, turns), mask)
            if kind == "query":
                _require(len(info["trip"]["switches"]) >= (1 if reverse_query else 2),
                         "insufficient fallback switches")
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded geometric construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    demo_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    first_mask = int(rng.integers(len(MASKS)))
    second_mask = int(rng.choice([i for i in range(len(MASKS)) if i != first_mask]))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = "query" if split == "test" else "ordinary"
            if split == "train" and i == demo_slots[0]:
                kind, mask_index = "reverse_demo", first_mask
            elif split == "train" and i == demo_slots[1]:
                kind, mask_index = "chord_demo", second_mask
            else:
                mask_index = int(rng.integers(len(MASKS)))
            for _ in range(12):
                pair = _build_pair(rng, kind, MASKS[mask_index])
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_differences(info):
    points, trip = info["points"], info["trip"]
    # Same completed rails, but ignore them when routing: the source's far end.
    expected = _mask_cells(info["mask"], trip["tip"], trip["facing"])
    no_companion_routing = _mask_cells(info["mask"], points[-1], _direction(points[-2], points[-1]))
    opposite = _travel(points, info["offset"], clockwise=False)
    left_turns = _mask_cells(info["mask"], opposite["tip"], opposite["facing"])
    unchanged_facing = _mask_cells(info["mask"], trip["tip"], info["initial_facing"])
    # Ink is a distinct layer, so equality of complete outputs with identical
    # rails is exactly equality of their ink cell sets. Wrong placements may
    # extend out of bounds; they are still demonstrably different predictions.
    return dict(translation_matters=expected != no_companion_routing,
                clockwise_matters=expected != left_turns,
                rotation_matters=expected != unchanged_facing)


def validate_generated_puzzle(puzzle):
    """Check visible geometry, rigid masks, teaching witnesses and the shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad example counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output must retain board dimensions")
            key = _fingerprint(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            _require(np.array_equal(out, _render(info)), "incorrect terminal rail state")
            ink, trip = info["ink"], info["trip"]
            _require(set(map(tuple, np.argwhere(out == ink))) ==
                     _mask_cells(info["mask"], trip["tip"], trip["facing"]),
                     "output does not preserve the actual rigid mask")
            _require(not np.array_equal(inp, out), "identity example")
            # Source stays in front of the companion at every unoccluded crossing.
            terminal = _mask_cells(info["mask"], trip["tip"], trip["facing"])
            _require(all(out[p] == info["source"] for p in trip["crossings"]-terminal),
                     "wrong crossing occlusion")
            _require(all(out[p] == info["background"] for p in info["rejected"]
                         if p not in set(trip["tracks"][1]) | terminal), "a chord was filled")
            info["ablations"] = _ablation_differences(info)
            infos[split].append(info)
    train = infos["train"]
    _require(len({i["mask"] for i in train}) >= 2, "need different visible mover masks")
    _require(any(i["rejected"] for i in train), "training must reject an accidental chord")
    _require(any(i["trip"]["exit"] == (1, -1) and len(i["trip"]["switches"]) == 1 for i in train),
             "training must teach reverse travel to the companion's entry end")
    _require(any(i["trip"]["exit"] == (0, 1) and len(i["trip"]["switches"]) >= 2 for i in train),
             "training must teach multiple switches with a source exit")
    _require(any(i["trip"]["crossings"] - _mask_cells(i["mask"], i["trip"]["tip"], i["trip"]["facing"])
                 for i in train), "training must expose source-over-companion occlusion")
    _require(any(len(set(i["trip"]["tracks"][1]) - set(i["trip"]["tracks"][0])
                         - _mask_cells(i["mask"], i["trip"]["tip"], i["trip"]["facing"])
                         - {i["marker"]}) >= 2 for i in train),
             "training must visibly extend the companion marker")
    for name in ("translation_matters", "clockwise_matters", "rotation_matters"):
        _require(any(i["ablations"][name] for i in train), f"training does not expose {name}")
    for i in infos["test"]:
        ds = [_direction(a, b) for a, b in zip(i["points"], i["points"][1:])]
        _require(len(i["points"]) > max(len(t["points"]) for t in train), "missing vertex-count shift")
        _require(sum(a != b for a, b in zip(ds, ds[1:])) >= 5
                 and len(i["trip"]["switches"]) >= 1
                 and len(i["trip"]["crossings"]) >= 2, "missing rail-complexity shift")
    return None


def measure_pair(pair):
    """Compact grid-derived measurements useful for reviewing variation."""
    info = _decode(pair["input"])
    _require(np.array_equal(_render(info), _grid(pair["output"])), "bad label")
    points, trip = info["points"], info["trip"]
    ds = [_direction(a, b) for a, b in zip(points, points[1:])]
    normalized = tuple((r-points[0][0], c-points[0][1]) for r, c in points)
    geometry = min(tuple(_transform(normalized, k, f)) for k in range(4) for f in (False, True))
    return dict(shape=list(info["grid"].shape), vertices=len(points),
                bends=sum(a != b for a, b in zip(ds, ds[1:])),
                lengths=[abs(a[0]-b[0])+abs(a[1]-b[1]) for a, b in zip(points, points[1:])],
                chords=info["chords"], rejected=len(info["rejected"]),
                crossings=len(trip["crossings"]), switches=len(trip["switches"]),
                unused_crossings=len(trip["crossings"]-set(trip["switches"])),
                exit=list(trip["exit"]), initial=info["initial_facing"], final=trip["facing"],
                offset=list(info["offset"]), area=len(info["mask"]), mask=info["mask"],
                palette=[info[k] for k in ("background", "source", "companion", "ink")],
                geometry=geometry, **_ablation_differences(info))


def _puzzle_key(puzzle):
    return tuple(tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep plus counts, determinism and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "puzzles did not vary")
    rows = {s: [measure_pair(pair) for p in puzzles for pair in p[s]] for s in ("train", "test")}
    for split, features in rows.items():
        _require({m["initial"] for m in features} == set(range(4)), "missing launch directions")
        _require({m["final"] for m in features} == set(range(4)), "missing terminal directions")
        _require({tuple(m["offset"]) for m in features} == set(OFFSETS), "missing marker offsets")
        _require(len({tuple(m["shape"]) for m in features}) >= 25, "canvas variation too low")
        _require(len({m["geometry"] for m in features}) >= (40 if split == "train" else 25),
                 "geometry variation too low")
        _require(len({m["mask"] for m in features}) >= 6, "mover geometry variation too low")
        _require(any(m["unused_crossings"] for m in features), "all-crossing parity shortcut untested")
        _require(any(len(set(m["lengths"])) > 1 for m in features), "all links have equal length")
        for role in range(4):
            _require(len({m["palette"][role] for m in features}) >= 8, "color role did not vary")
    _require(len({tuple(m["exit"]) for m in rows["train"]+rows["test"]}) >= 3,
             "reachable endpoints too restricted")
    _require(any(m["crossings"] == 0 for m in rows["train"]), "no simple uncrossed example")
    _require(max(m["switches"] for m in rows["test"]) >= 4, "query journeys too short")
    _require(any(len({m["mask"] for m in rows["train"] if m["area"] == a}) > 1
                 for a in range(3, 13)), "equal-area masks missing")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(9700+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = copy.deepcopy(puzzles[0])
    info = _decode(damaged["test"][0]["input"])
    damaged["test"][0]["output"][info["trip"]["tip"]] = info["background"]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted terminal mask accepted")
    # A rectangle has two full open orders from a corner; no latent order may
    # decide which one to draw. The source-marker vector must be diagonal.
    try:
        _infer_chain([(0, 0), (0, 6), (6, 6), (6, 0)], (0, 0))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous source topology accepted")
    pair = puzzles[0]["train"][0]
    info = _decode(pair["input"])
    malformed = pair["input"].copy()
    marker = info["marker"]
    anchor = marker[0]-info["offset"][0], marker[1]-info["offset"][1]
    malformed[marker] = info["background"]
    malformed[(anchor[0], marker[1])] = info["companion"]
    try:
        _decode(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("cardinal companion displacement accepted")
    # Translation covariance and clockwise rotation covariance, input only.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for k in range(4):
            _require(np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k)),
                     "rotation covariance failed")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in p),
            "geometry_counts": {s: len({m["geometry"] for m in rows[s]}) for s in rows},
            "exit_histogram": dict(Counter(str(m["exit"]) for s in rows for m in rows[s]))}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
