"""Absence-operated relays: plate defects supply implicit wire directions.

Only Python's standard library and NumPy are used. No parent module is imported.
Generation renders explicit construction masks; solve reconstructs the input.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _boundary(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _points(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _mask(shape, cells):
    out = np.zeros(shape, dtype=bool)
    for p in cells:
        out[p] = True
    return out


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _components(mask):
    remaining = _points(mask)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _block(mask):
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1]
                       & mask[:-1, 1:] & mask[1:, 1:]))


def _close_short_gaps(mask, maximum):
    """One simultaneous pass, using only endpoints present before the pass."""
    out = mask.copy()
    for axis in (0, 1):
        a, b = (mask, out) if axis == 0 else (mask.T, out.T)
        for distance in range(2, maximum + 2):
            endpoints = a[:-distance] & a[distance:]
            for offset in range(1, distance):
                b[offset:len(a)-distance+offset] |= endpoints
    return out


def _recover_plate(observed):
    # Exterior specks are components of one or two cells, separated from plates.
    supported = np.zeros_like(observed)
    for component in _components(observed):
        if len(component) > 3:
            for p in component:
                supported[p] = True
    # Missing runs are at most two cells and have tangible collinear support.
    return _close_short_gaps(supported, 2)


def _roles(inp):
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 3:
        raise ValueError("An input must contain field, plate and wire colors")
    bg = int(colors[int(counts.argmax())])
    if int(counts.max()) * 2 <= inp.size:
        raise ValueError("The field must occupy more than half the input")
    thick = [int(c) for c in colors if c != bg and _block(inp == c)]
    if len(thick) != 1:
        raise ValueError("Exactly one non-field color must have a 2x2 block")
    solid = thick[0]
    wire = next(int(c) for c in colors if c not in (bg, solid))
    return bg, solid, wire


def _edits(base, holes, wires):
    """Find hole-derived directions and follow original terminal branches."""
    shape = base.shape
    edits = []
    for h in sorted(_points(holes)):
        candidates = []
        for d in DIRS:
            tangent = (-d[1], d[0])
            p, q = _add(h, d, -2), _add(h, d, -1)
            evidence = (_add(h, d), _add(h, tangent), _add(h, tangent, -1))
            if not all(_inside(x, shape) and base[x] for x in evidence):
                continue
            if not _inside(p, shape) or base[q] or base[p] or p not in wires or q in wires:
                continue
            old_start = _add(p, d, -1)
            if old_start not in wires:
                continue
            if not any(_add(p, tangent, s) in wires for s in (-1, 1)):
                continue
            previous, current, old = p, old_start, []
            while current not in old:
                old.append(current)
                following = [x for x in _neighbors(current, wires) if x != previous]
                if not following:
                    if _boundary(current, shape):
                        ray, point = set(), _add(p, d)
                        while _inside(point, shape):
                            ray.add(point)
                            point = _add(point, d)
                        candidates.append({"hole": h, "pivot": p, "direction": d,
                                           "old": set(old), "walk": (p, *old), "ray": ray})
                    break
                if len(following) != 1:
                    break
                previous, current = current, following[0]
        if len(candidates) > 1:
            raise ValueError("A defect has several competing relay pivots")
        edits.extend(candidates)
    for i, a in enumerate(edits):
        for b in edits[i+1:]:
            if (a["pivot"] == b["pivot"] or a["old"] & b["old"]
                    or a["pivot"] in b["old"] or b["pivot"] in a["old"]):
                raise ValueError("Relay terminal branches are not independent")
    return edits


def _decode(input_grid):
    inp = np.asarray(input_grid)
    if (inp.ndim != 2 or not all(1 <= n <= 30 for n in inp.shape)
            or not np.issubdtype(inp.dtype, np.integer)
            or not np.all((inp >= 0) & (inp <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    bg, solid, wire = _roles(inp)
    base = _recover_plate(inp == solid)
    holes = base & (inp == bg)
    extras = ~base & (inp == solid)
    wires = _points(_close_short_gaps(inp == wire, 1))
    if any(base[p] for p in wires) or np.any(base & (inp == wire)):
        raise ValueError("Input wires must be outside plates")
    edits = _edits(base, holes, wires)
    return inp, (bg, solid, wire), base, holes, extras, wires, edits


def _predict(decoded, *, ignore_holes=False, ignore_routing=False):
    inp, (bg, solid, wire), base, holes, extras, wires, edits = decoded
    out = inp.copy()
    out[extras] = bg
    if ignore_holes:
        out[holes] = solid
        edits = []
    old = set().union(*(e["old"] for e in edits)) if edits and not ignore_routing else set()
    for p in wires - old:
        out[p] = wire
    for p in old:
        out[p] = bg
    if not ignore_holes:
        out[_dilate(holes) & ~holes] = wire
        if not ignore_routing:
            for edit in edits:
                for p in edit["ray"]:
                    out[p] = wire
        # Empty defect centers win even where a ray crosses them.
        out[holes] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; no RNG, training pairs or generated state."""
    return _predict(_decode(input_grid))


def _line(a, b):
    if a[0] != b[0] and a[1] != b[1]:
        raise ValueError("Segments must be orthogonal")
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [_add(a, d, n) for n in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(vertices):
    path = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        path.extend(_line(a, b)[1:])
    return path


def _simple(path):
    cells = set(path)
    return len(path) == len(cells) and all(len(_neighbors(p, cells)) <= 2 for p in cells)


def _path(rng, shape, both_boundary, complexity):
    h, w = shape
    side = int(rng.integers(4))
    if side < 2:
        start, axis = (0 if side == 0 else h-1, int(rng.integers(2, w-2))), 0
    else:
        start, axis = (int(rng.integers(2, h-2)), 0 if side == 2 else w-1), 1
    vertices = [start]
    segments = int(rng.integers(3, 5 + complexity))
    for index in range(segments):
        current = vertices[-1]
        if index == segments-1 and both_boundary:
            choices = [0, shape[axis]-1]
        else:
            choices = list(range(2, shape[axis]-2))
        rng.shuffle(choices)
        for value in choices:
            if abs(value-current[axis]) < 4:
                continue
            point = list(current)
            point[axis] = value
            candidate = vertices + [tuple(point)]
            if _simple(_polyline(candidate)):
                vertices = candidate
                break
        else:
            return None
        axis = 1-axis
    path = _polyline(vertices)
    if both_boundary and not _boundary(path[-1], shape):
        return None
    mask = _mask(shape, path)
    if not np.array_equal(_close_short_gaps(mask, 1), mask):
        return None
    return vertices, path


def _plate(rng, shape, pivot, d):
    h = _add(pivot, d, 2)
    t = (-d[1], d[0])
    depth, low, high = int(rng.integers(4, 9)), int(rng.integers(1, 5)), int(rng.integers(2, 6))
    cells = {_add(_add(h, d, a), t, b) for a in range(depth) for b in range(-low, high+1)}
    if not all(_inside(p, shape) for p in cells):
        return None
    return h, cells


def _construction_output(shape, colors, base, holes, wires, edits):
    """Render explicit latent masks, without calling the decoder or predictor."""
    bg, solid, wire = colors
    out = np.where(base, solid, bg).astype(np.int8)
    removed = set()
    for e in edits:
        removed.update(e["old"])
    for p in wires:
        if p not in removed:
            out[p] = wire
    for h in _points(holes):
        y, x = h
        out[max(0, y-1):y+2, max(0, x-1):x+2] = wire
    for e in edits:
        p = _add(e["pivot"], e["direction"])
        while _inside(p, shape):
            out[p] = wire
            p = _add(p, e["direction"])
    out[holes] = bg
    return out


def _attempt(rng, colors, relay_count, rich, cluster):
    shape = (int(rng.integers(16, 31)), int(rng.integers(16, 31)))
    route = _path(rng, shape, relay_count == 2 or bool(rng.integers(2)), 1 if rich else 0)
    if route is None:
        return None
    vertices, path = route
    wires = set(path)
    # A shared path has two removable ends; its middle remains a single network.
    if relay_count == 1:
        indices = [int(rng.integers(1, len(vertices)-1))]
    else:
        first = int(rng.integers(1, len(vertices)-2))
        second = int(rng.integers(first+1, len(vertices)-1))
        indices = [first, second]
    base = np.zeros(shape, dtype=bool)
    holes = np.zeros(shape, dtype=bool)
    edits = []
    for number, index in enumerate(indices):
        p = vertices[index]
        endpoint_side = vertices[index+1] if relay_count == 2 and number == 1 else vertices[index-1]
        d = (int(np.sign(p[0]-endpoint_side[0])), int(np.sign(p[1]-endpoint_side[1])))
        plate = _plate(rng, shape, p, d)
        if plate is None:
            return None
        h, cells = plate
        if cells & wires:
            return None
        for point in cells:
            base[point] = True
        holes[h] = True
        at = path.index(p)
        old = set(path[at+1:]) if relay_count == 2 and number == 1 else set(path[:at])
        edits.append({"hole": h, "pivot": p, "direction": d, "old": old})
    # Grow attached offset rectangles, producing L/T/stepped plate unions.
    if rich:
        for _ in range(int(rng.integers(1, 4))):
            for _ in range(12):
                height, width = int(rng.integers(4, 10)), int(rng.integers(4, 10))
                y, x = int(rng.integers(shape[0]-height+1)), int(rng.integers(shape[1]-width+1))
                rectangle = np.zeros(shape, dtype=bool)
                rectangle[y:y+height, x:x+width] = True
                if int((rectangle & base).sum()) < 4 or int((rectangle & ~base).sum()) < 4:
                    continue
                grown = base | rectangle
                if any(grown[p] for p in wires):
                    continue
                if any(grown[_add(e["hole"], e["direction"], -1)] for e in edits):
                    continue
                if grown.sum() > grown.size * .36:
                    continue
                base = grown
                break
    # A boundary dimer demonstrates merging halos and exclusion of all centers.
    if cluster:
        e = edits[0]
        t = (-e["direction"][1], e["direction"][0])
        choices = [_add(e["hole"], t, s) for s in (-1, 1)]
        rng.shuffle(choices)
        for p in choices:
            if _inside(p, shape) and base[p]:
                holes[p] = True
                break
    # Interior holes always show that a defect need not activate a relay.
    interior = base.copy()
    for d in DIRS:
        for p in list(_points(interior)):
            q = _add(p, d)
            if not _inside(q, shape) or not base[q]:
                interior[p] = False
    choices = sorted(_points(interior & ~_dilate(holes)))
    if not choices:
        return None
    interior_hole = choices[int(rng.integers(len(choices)))]
    holes[interior_hole] = True
    if rich:
        more = sorted(_points(interior & ~_dilate(holes)))
        rng.shuffle(more)
        for p in more[:int(rng.integers(0, 4))]:
            holes[p] = True
    if not np.array_equal(_recover_plate(base), base):
        return None
    bg, solid, wire = colors
    inp = np.where(base, solid, bg).astype(np.int8)
    for p in wires:
        inp[p] = wire
    inp[holes] = bg
    out = _construction_output(shape, colors, base, holes, wires, edits)
    # At least one retained gap makes repair observable, even with two trains.
    removed = set().union(*(e["old"] for e in edits))
    ray_cells = set()
    for e in edits:
        q = _add(e["pivot"], e["direction"])
        while _inside(q, shape):
            ray_cells.add(q)
            q = _add(q, e["direction"])
    halo = _dilate(holes)
    gap_candidates = [p for p in path[1:-1] if p not in removed and p not in ray_cells
                      and p not in {e["pivot"] for e in edits}
                      and len(_neighbors(p, wires)) == 2
                      and all(p != v for v in vertices)
                      and not halo[p]]
    if not gap_candidates:
        return None
    rng.shuffle(gap_candidates)
    gap_count = int(rng.integers(1, min(5, len(gap_candidates))+1))
    damaged = set()
    for p in gap_candidates:
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in damaged):
            continue
        inp[p] = bg
        damaged.add(p)
        if len(damaged) >= gap_count:
            break
    # Occasionally damage a branch which will be deleted; repair precedes tracing.
    old_options = [p for p in sorted(removed) if p not in vertices and not _boundary(p, shape)
                   and len(_neighbors(p, wires)) == 2
                   and all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) > 2 for q in damaged)]
    if old_options and rng.random() < .65:
        inp[old_options[int(rng.integers(len(old_options)))]] = bg
    extras = np.zeros(shape, dtype=bool)
    free = (inp == bg) & ~_dilate(base) & (out == bg) & ~_mask(shape, wires)
    options = sorted(_points(free))
    rng.shuffle(options)
    wanted = int(rng.integers(1, 6))
    for p in options:
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in _points(extras)):
            continue
        inp[p] = solid
        extras[p] = True
        if extras.sum() < wanted and rng.random() < .3:
            neighbors = [_add(p, d) for d in DIRS]
            rng.shuffle(neighbors)
            for q in neighbors:
                if not _inside(q, shape) or not free[q] or extras[q]:
                    continue
                if any(max(abs(q[0]-r[0]), abs(q[1]-r[1])) <= 1
                       for r in _points(extras) if r != p):
                    continue
                inp[q] = solid
                extras[q] = True
                break
        if extras.sum() == wanted:
            break
    # Keep the field identifiable in the filled-defect counterfactual too.
    if not extras.any() or np.count_nonzero(inp == bg)*2 <= inp.size + 2*len(edits):
        return None
    try:
        decoded = _decode(inp)
        if (not np.array_equal(decoded[2], base) or not np.array_equal(decoded[3], holes)
                or not np.array_equal(decoded[4], extras) or decoded[5] != wires):
            return None
        observed = {(e["hole"], e["pivot"], frozenset(e["old"])) for e in decoded[6]}
        intended = {(e["hole"], e["pivot"], frozenset(e["old"])) for e in edits}
        if observed != intended or not np.array_equal(_predict(decoded), out):
            return None
        features = _features(decoded)
        if not features["visible_repair"] or not features["visible_erasure"]:
            return None
        if not features["omitting_A_changes"] or not features["omitting_B_changes"]:
            return None
    except ValueError:
        return None
    return {"input": inp, "output": out}


def _features(decoded):
    inp, (bg, solid, wire), base, holes, extras, wires, edits = decoded
    out = _predict(decoded)
    old = set().union(*(e["old"] for e in edits)) if edits else set()
    rays = set().union(*(e["ray"] for e in edits)) if edits else set()
    active = {e["hole"] for e in edits}
    halo = _dilate(holes)
    padded = np.pad(base.astype(np.int16), 1)
    corners = int(np.abs(padded[1:, 1:] - padded[1:, :-1]
                         - padded[:-1, 1:] + padded[:-1, :-1]).sum())
    bends = []
    for e in edits:
        walk = e["walk"]
        ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
        bends.append(sum(a != b for a, b in zip(ds, ds[1:])))
    return {"shape": list(inp.shape), "colors": [bg, solid, wire],
            "relays": len(edits), "holes": int(holes.sum()), "extras": int(extras.sum()),
            "plate_cells": int(base.sum()), "plate_components": len(_components(base)),
            "plate_corners": corners,
            "speck_clusters": sum(len(c) > 1 for c in _components(extras)),
            "halo_only_holes": len(_points(holes)-active),
            "directions": [list(e["direction"]) for e in edits], "old_bends": bends,
            "old_lengths": [len(e["old"]) for e in edits],
            "ray_lengths": [len(e["ray"]) for e in edits],
            "visible_repair": any(inp[p] == bg and out[p] == wire and p not in old
                                  and p not in rays and not halo[p] for p in wires),
            "deleted_gaps": sum(int(inp[p] == bg) for p in old),
            "visible_erasure": int(np.count_nonzero(extras & (out == bg))),
            "ray_meets_retained": len(rays & (wires-old)),
            "ray_meets_other_old": sum(len(a["ray"] & b["old"]) for a in edits for b in edits if a is not b),
            "omitting_A_changes": not np.array_equal(out, _predict(decoded, ignore_holes=True)),
            "omitting_B_changes": not np.array_equal(out, _predict(decoded, ignore_routing=True)),
            "plate_signature": hashlib.sha256(np.packbits(base).tobytes()).hexdigest()}


def _counts(num_train, num_test):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral)
           for n in (num_train, num_test)):
        raise ValueError("Example counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.choice(10, 3, replace=False)))
    regime = int(rng.integers(3))
    held_index = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                relays = 2 if regime == 2 and index >= 1 else 1
                rich = index >= 1
                cluster = index == 1 or (index > 1 and bool(rng.integers(2)))
            else:
                relays = 2 if regime in (0, 2) else 1
                rich = True
                cluster = bool(rng.integers(2))
            for _ in range(1800):
                pair = _attempt(rng, colors, relays, rich, cluster)
                if pair is None:
                    continue
                if regime == 1:
                    # Teach two different normals; queries use a third normal.
                    target = DIRS[(held_index + 1 + index % 2) % 4] if split == "train" else DIRS[held_index]
                    current = _decode(pair["input"])[6][0]["direction"]
                    k = 0
                    while current != target:
                        current = (-current[1], current[0])
                        k += 1
                    pair = {role: np.rot90(grid, k).copy() for role, grid in pair.items()}
                if regime == 2 and relays == 2:
                    directions = {e["direction"] for e in _decode(pair["input"])[6]}
                    if (len(directions) > 1) != (split == "test"):
                        continue
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded relay construction exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, roles, records = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen, "Repeated input"
            seen.add(key)
            decoded = _decode(inp)
            base, holes, extras, wires = decoded[2:6]
            assert np.array_equal(_recover_plate(base), base), "Plate has an unsupported narrow cavity"
            assert len(_components(_mask(inp.shape, wires))) == 1, "The wire must form one shared path"
            assert _simple(wires) and sum(len(_neighbors(p, wires)) == 1 for p in wires) == 2
            assert any(_boundary(p, inp.shape) for p in wires)
            assert not np.any(_dilate(base) & extras), "Specks must be separated from plate material"
            assert all(len(c) <= 2 for c in _components(extras)), "Specks must be singletons or dimers"
            roles.add(decoded[1])
            assert np.array_equal(out, _predict(decoded)), "Full-grid relay rule mismatch"
            assert not np.array_equal(inp, out)
            f = _features(decoded)
            assert f["relays"] in (1, 2) and f["halo_only_holes"] >= 1
            assert f["visible_repair"] and f["visible_erasure"]
            assert f["omitting_A_changes"] and f["omitting_B_changes"]
            assert np.all(out[decoded[3]] == decoded[1][0]), "A hole center was filled"
            records[split].append(f)
    assert len(roles) == 1, "Color conventions must agree within a puzzle"
    assert any(f["relays"] == 1 for f in records["train"])
    assert any(f["holes"] > f["relays"] for f in records["train"])
    # Removing the defect evidence must alter routing beyond the local halos.
    for pair in puzzle["train"][:2]:
        decoded = _decode(np.asarray(pair["input"]))
        filled = np.asarray(pair["input"]).copy()
        for e in decoded[6]:
            filled[e["hole"]] = decoded[1][1]
        changed = solve(filled) != np.asarray(pair["output"])
        outside = ~_dilate(decoded[3])
        assert np.any(changed & outside), "Defects must control remote wire edits"
    return None


def self_test():
    """42 seeds, streaming metrics and bounded metamorphic/negative checks."""
    shapes, structures, palettes, directions, relay_counts, bends, hole_counts = (set() for _ in range(7))
    counts = {"pairs": 0, "repairs": 0, "erasures": 0, "ray_contacts": 0, "deleted_gaps": 0}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                decoded = _decode(inp)
                f = _features(decoded)
                shapes.add(inp.shape)
                structures.add((inp.shape, f["plate_signature"]))
                palettes.add(tuple(f["colors"]))
                directions.update(map(tuple, f["directions"]))
                relay_counts.add(f["relays"])
                bends.update(f["old_bends"])
                hole_counts.add(f["holes"])
                counts["pairs"] += 1
                counts["repairs"] += int(f["visible_repair"])
                counts["erasures"] += f["visible_erasure"]
                counts["ray_contacts"] += f["ray_meets_retained"]
                counts["deleted_gaps"] += f["deleted_gaps"]
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                if seed < 3:
                    for k in range(4):
                        transformed = np.rot90(np.fliplr(inp), k).copy()
                        expected = np.rot90(np.fliplr(out), k)
                        assert np.array_equal(solve(transformed), expected)
        signatures.add(digest.hexdigest())
        if seed < 2:
            replay = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], replay[s]) for k in a)
            # Mutate preserved cells, holes, deleted branches and new rays.
            pair = puzzle["test"][0]
            decoded = _decode(pair["input"])
            sites = {tuple(map(int, np.argwhere(pair["input"] == pair["output"])[0]))}
            sites.update(_points(decoded[3]))
            for e in decoded[6]:
                sites.update((min(e["old"]), max(e["ray"]), e["pivot"]))
            for site in sites:
                corrupted = pair["output"].copy()
                corrupted[site] = (int(corrupted[site])+1) % 10
                mutant = {"train": puzzle["train"], "test": [{"input": pair["input"], "output": corrupted}]}
                try:
                    validate_generated_puzzle(mutant)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("Corrupted output was accepted")
    assert len(signatures) == 42 and len(shapes) >= 60 and len(structures) >= 150
    assert len(palettes) >= 30 and directions == set(DIRS) and relay_counts == {1, 2}
    assert {0, 1, 2} <= bends and len(hole_counts) >= 4
    assert counts["repairs"] == counts["pairs"] and counts["ray_contacts"] > 0 and counts["deleted_gaps"] > 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(42, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "plate_structures": len(structures), "palettes": len(palettes),
            "directions": sorted(directions), "old_bends": sorted(bends),
            "hole_counts": sorted(hole_counts), **counts}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
