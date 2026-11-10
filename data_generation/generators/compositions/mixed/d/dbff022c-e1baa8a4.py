"""Restore a run-expanded scene, then fill cavities using its inward edge key.

No parent module, file, seed or generation metadata is consulted by solve().
Generation and self_test keep only bounded in-memory scenes and small digests.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "left", "bottom", "right")
HOST_SEEDS = (0, 28401597, 307635342, 691214437, 953571650, 80312219,
              495128436, 107072989, 168091465, 464448487, 875480028,
              389261376, 940482865, 901754103, 23497915, 619468002,
              105029216, 469411350, 657084303, 231810517, 689035249,
              320441955, 99718238, 2014286, 798538343, 721672597,
              51607890, 406156243, 896431068, 736652172, 993930154,
              874041991, 335702260, 350134396, 182505415, 178548020,
              85615149, 704510159, 189100633, 421110933, 643823015, 965104562)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must be two dimensional, with sides in 1..30")
    _require(np.issubdtype(a.dtype, np.integer)
             and bool(np.all((0 <= a) & (a <= 9))), "invalid color or dtype")
    return a.astype(np.int8, copy=False)


def _starts(a, axis):
    lines = a if axis == 0 else a.T
    return np.r_[0, np.flatnonzero(np.any(lines[1:] != lines[:-1], axis=1)) + 1]


def _compact(a):
    # Representatives of maximal CONSECUTIVE equal full vectors, on both axes.
    return a[np.ix_(_starts(a, 0), _starts(a, 1))].copy()


def _runs(a, axis):
    return tuple(map(int, np.diff(np.r_[_starts(a, axis), a.shape[axis]])))


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    steps = STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                q = stack.pop()
                cells.append(q)
                for dr, dc in steps:
                    rr, cc = q[0] + dr, q[1] + dc
                    if (0 <= rr < h and 0 <= cc < w
                            and mask[rr, cc] and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            yield cells


def _parse_scene(a):
    """Decode a compact scene. Background/ownership/key ambiguity is invalid."""
    colors, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1,
             "compact scene has no unique modal background")
    bg = int(colors[np.argmax(counts)])
    h, w = a.shape
    candidates = []
    ink_components = list(_components(a != bg, diagonal=True))
    for cells in ink_components:
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) != sh * sw or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 7:
            if r0 == 0:
                lanes.append(("top", a[r0, c0:c1+1], a[r1, c0:c1+1]))
            if r1 == h - 1:
                lanes.append(("bottom", a[r1, c0:c1+1], a[r0, c0:c1+1]))
        if sw == 2 and 2 <= sh <= 7:
            if c0 == 0:
                lanes.append(("left", a[r0:r1+1, c0], a[r0:r1+1, c1]))
            if c1 == w - 1:
                lanes.append(("right", a[r0:r1+1, c1], a[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            src, dst = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(src)) == len(src):
                candidates.append((side, dict(zip(src, dst)), frozenset(cells)))
    _require(len(candidates) == 1, "need exactly one unambiguous two-lane edge key")
    side, mapping, key_cells = candidates[0]
    ids = np.full(a.shape, -1, dtype=np.int16)
    objects = []
    for cells in ink_components:
        if frozenset(cells) == key_cells:
            continue
        ink = {int(a[q]) for q in cells}
        _require(len(ink) == 1, "contour components must be monochromatic")
        obj = {"color": ink.pop(), "cells": cells, "holes": []}
        for q in cells:
            ids[q] = len(objects)
        objects.append(obj)
    regions = []
    for cells in _components(a == bg):
        edge = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        boundary = set()
        for r, c in cells:
            for dr, dc in STEPS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and a[rr, cc] != bg:
                    boundary.add((rr, cc))
        owner = None
        if not edge:
            owners = {int(ids[q]) for q in boundary}
            _require(len(owners) == 1 and -1 not in owners,
                     "cavity must have exactly one contour owner, excluding key")
            owner = owners.pop()
            objects[owner]["holes"].append(cells)
        regions.append({"cells": cells, "edge": edge, "owner": owner})
    return {"background": bg, "side": side, "mapping": mapping,
            "key_cells": key_cells, "objects": objects, "regions": regions}


def _fill(a, scene):
    out = a.copy()
    for region in scene["regions"]:
        if region["owner"] is None:
            continue
        color = scene["objects"][region["owner"]]["color"]
        if color in scene["mapping"]:
            for q in region["cells"]:
                out[q] = scene["mapping"][color]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Canonicalize FIRST; read and apply the restored key exactly once."""
    compact = _compact(_grid(input_grid))
    return _fill(compact, _parse_scene(compact))


def _dilate(mask, diagonal=False):
    result = mask.copy()
    result[1:] |= mask[:-1]
    result[:-1] |= mask[1:]
    result[:, 1:] |= mask[:, :-1]
    result[:, :-1] |= mask[:, 1:]
    if diagonal:
        result[1:, 1:] |= mask[:-1, :-1]
        result[1:, :-1] |= mask[:-1, 1:]
        result[:-1, 1:] |= mask[1:, :-1]
        result[:-1, :-1] |= mask[1:, 1:]
    return result


def _template_holes(wall):
    """Construction-only local flood, independent of scene/key decoding."""
    holes = np.zeros(wall.shape, dtype=bool)
    for cells in _components(~wall):
        if not any(r in (0, wall.shape[0]-1) or c in (0, wall.shape[1]-1)
                   for r, c in cells):
            for q in cells:
                holes[q] = True
    return holes


def _shape(rng, kind):
    if kind == "self":
        wall = np.ones((3, 3), dtype=bool)
        wall[1, 1] = False
    elif kind == "diamond":
        radius = int(rng.integers(1, 4))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        wall = abs(rr-radius) + abs(cc-radius) == radius
    elif kind in ("rectangle", "open", "multi"):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(5 if kind == "multi" else 3, 9))
        wall = np.ones((h, w), dtype=bool)
        wall[1:-1, 1:-1] = False
        if kind == "open":
            wall[0, int(rng.integers(1, w-1))] = False
        elif kind == "multi":
            wall[:, int(rng.integers(2, w-2))] = True
    elif kind == "profile":
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        cavity = np.ones((h, w), dtype=bool)
        cavity[0, -1] = cavity[-1, 0] = False
        if rng.random() < .5:
            cavity[-1, -1] = False
        cavity = np.pad(cavity, 1)
        wall = _dilate(cavity, diagonal=bool(rng.integers(2))) & ~cavity
    else:
        raise ValueError("unknown shape")
    # No redundant padding; contours can be placed directly on the border.
    rr, cc = np.nonzero(wall)
    wall = wall[rr.min():rr.max()+1, cc.min():cc.max()+1]
    wall = np.rot90(wall, int(rng.integers(4))).copy()
    if rng.random() < .5:
        wall = np.fliplr(wall).copy()
    return wall, _template_holes(wall)


def _mapping_and_specs(rng, lesson):
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    n = int(rng.integers(2, 8))
    sources, outside = colors[:n], colors[n:]
    fills = list(map(int, rng.choice(colors, n)))
    # At least one nontrivial mapping keeps the two key lanes distinguishable.
    fills[0] = outside[0]
    if lesson == 0:
        fills[0], fills[1] = sources[1], outside[0]
        # Identical-size open/closed pair makes leakage visible. The profile
        # also supplies a nonrectangular cavity owned by the chain destination.
        specs = [(sources[0], "rectangle"), (sources[0], "open"),
                 (sources[1], "profile"), (outside[0], "rectangle")]
    elif lesson == 1:
        fills[1] = sources[1]
        fills = [outside[0] if c == outside[-1] else c for c in fills]
        specs = [(sources[0], "multi"), (sources[0], "diamond"),
                 (outside[-1], "rectangle"), (sources[1], "self")]
    else:
        style = int(rng.integers(4))
        if style == 0:
            fills = sources[1:] + [outside[0]]
        elif style == 1:
            fills = [outside[0]] * n
        elif style == 2:
            fills[1] = sources[1]
        count = int(rng.integers(1, 7))
        kinds = ("rectangle", "profile", "diamond", "multi", "open")
        specs = [(sources[0], str(rng.choice(kinds[:-1])))]
        for _ in range(1, count):
            color = int(rng.choice(sources if rng.random() < .7 else outside))
            specs.append((color, str(rng.choice(kinds))))
    order = rng.permutation(n)
    return [sources[i] for i in order], [fills[i] for i in order], specs


def _pack_scene(rng, sources, fills, specs, lesson):
    shapes = [_shape(rng, kind) for color, kind in specs]
    if lesson == 0:
        # Match the first closed ring's geometry, opening just one wall pixel.
        closed = shapes[0][0]
        opened = closed.copy()
        # Any rectangle orientation still has a straight top wall.
        opened[0, 1] = False
        shapes[1] = opened, _template_holes(opened)
    area = sum((wall.shape[0]+1)*(wall.shape[1]+1) for wall, holes in shapes)
    for attempt in range(32):
        if len(specs) <= 2:
            h, w = int(rng.integers(8, 17)), int(rng.integers(8, 19))
        else:
            h, w = int(rng.integers(14, 24)), int(rng.integers(14, 24))
        if attempt >= 20:
            h = w = 24
        if area + 3*len(sources) > h*w:
            continue
        a = np.zeros((h, w), dtype=np.int8)
        start = int(rng.integers(w-len(sources)+1))
        if len(sources) == 2:
            start = int(rng.integers(1, w-2))  # exclude ambiguous corner 2x2
        a[0, start:start+len(sources)] = sources
        a[1, start:start+len(sources)] = fills
        blocked = _dilate(a != 0, diagonal=True)
        placements = {}
        if lesson == 1:
            # Reserve an isolated three-row self ring. Filling it creates equal
            # adjacent full rows; the answer must NOT be compressed again.
            i = len(specs)-1
            placements[i] = (h-3, int(rng.integers(w-2)))
            blocked[h-4:] = True
        for i in sorted(range(len(specs)), key=lambda j: -shapes[j][0].size):
            if i in placements:
                continue
            mh, mw = shapes[i][0].shape
            positions = [(r, c) for r in range(h-mh+1) for c in range(w-mw+1)
                         if not np.any(blocked[r:r+mh, c:c+mw])]
            if not positions:
                break
            borders = [(r, c) for r, c in positions
                       if r == 0 or c == 0 or r+mh == h or c+mw == w]
            choices = borders if borders and rng.random() < .4 else positions
            r, c = choices[int(rng.integers(len(choices)))]
            placements[i] = r, c
            blocked[max(0, r-1):min(h, r+mh+1),
                    max(0, c-1):min(w, c+mw+1)] = True
        else:
            expected = a.copy()
            mapping = dict(zip(sources, fills))
            for i, (color, kind) in enumerate(specs):
                wall, holes = shapes[i]
                r, c = placements[i]
                sh, sw = wall.shape
                a[r:r+sh, c:c+sw][wall] = color
                expected[r:r+sh, c:c+sw][wall] = color
                if color in mapping:
                    expected[r:r+sh, c:c+sw][holes] = mapping[color]
            # Remove redundant design coordinates using INPUT representatives.
            # Never re-compress the filled labels: equal lines created by fill
            # are part of the answer. Topology is preserved by whole-line runs.
            ri, ci = _starts(a, 0), _starts(a, 1)
            return a[np.ix_(ri, ci)], expected[np.ix_(ri, ci)]
    raise RuntimeError("bounded random packing exhausted")


def _sample_runs(rng, length, expand, key_thickness=False):
    values = np.ones(length, dtype=int)
    if not expand:
        return values
    if not key_thickness and length <= 14 and rng.random() < .12:
        return values * 2
    # The compact side is at most 24, so there is always room for expansion.
    budget = min(30-length, int(rng.integers(2, 12)))
    if key_thickness:
        values[0] += 1
        budget -= 1
    if rng.random() < .25:
        q = int(rng.integers(2 if key_thickness else 0, length))
        values[q] += budget
    else:
        choices = rng.integers(2 if key_thickness else 0, length, size=budget)
        for q in choices:
            values[q] += 1
    return values


def _make_pair(rng, side, lesson):
    for _ in range(24):
        sources, fills, specs = _mapping_and_specs(rng, lesson)
        try:
            latent, target = _pack_scene(rng, sources, fills, specs, lesson)
        except RuntimeError:
            continue
        palette = rng.permutation(np.arange(10, dtype=np.int8))
        latent, target = palette[latent], palette[target]
        # Both axes for the lessons; later scenes sometimes expand just one.
        axes = 0 if lesson in (0, 1) else int(rng.choice(3, p=[.7, .15, .15]))
        rows = _sample_runs(rng, latent.shape[0], axes != 2, key_thickness=lesson == 0)
        cols = _sample_runs(rng, latent.shape[1], axes != 1)
        inp = np.repeat(np.repeat(latent, rows, axis=0), cols, axis=1)
        turns = SIDES.index(side)
        inp, target = np.rot90(inp, turns).copy(), np.rot90(target, turns).copy()
        try:
            f = _analyze_pair(inp, target)
        except ValueError:
            continue
        if lesson == 0 and not (f["chain"] and f["open_active"]
                                and f["fill_only_inert"] and f["irregular"]):
            continue
        if lesson == 1 and not (f["self_map"] and f["diagonal"]
                                and f["absent_inert"] and f["postfill_equal"]):
            continue
        return {"input": inp, "output": target}
    raise RuntimeError("bounded scene construction exhausted")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _fingerprint(a):
    return (a.shape, a.tobytes())


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    order = list(map(str, rng.permutation(SIDES)))
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            side = order[i % 2] if split == "train" else order[2 + i % 2]
            lesson = i if split == "train" and i < 2 else -1
            for _ in range(24):
                pair = _make_pair(rng, side, lesson)
                fp = _fingerprint(pair["input"])
                if fp in seen or (lesson == 1 and pair["output"].shape == first_shape):
                    continue
                seen.add(fp)
                puzzle[split].append(pair)
                if lesson == 0:
                    first_shape = pair["output"].shape
                break
            else:
                raise RuntimeError("could not produce a distinct input/lesson shape")
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_separated_repeat(a):
    return any(np.array_equal(lines[i], lines[j])
               for lines in (a, a.T) for i in range(len(lines))
               for j in range(i+2, len(lines)))


def _analyze_pair(inp, out):
    a, out = _compact(_grid(inp)), _grid(out)
    scene = _parse_scene(a)
    _require(a.shape == out.shape, "output must have canonical INPUT dimensions")
    _require(np.array_equal(_fill(a, scene), out), "incorrect keyed cavity fill")
    bg, mapping = scene["background"], scene["mapping"]
    _require(np.array_equal(a[a != bg], out[a != bg]), "original ink/key changed")
    _require(np.any(a != out), "scene must expose at least one cavity fill")
    row_runs, col_runs = _runs(inp, 0), _runs(inp, 1)
    reconstructed = np.repeat(np.repeat(a, row_runs, axis=0), col_runs, axis=1)
    _require(np.array_equal(reconstructed, inp), "invalid rectangular expansion")
    _require(inp.shape != a.shape, "each generated scene must have redundant bands")
    closed = [o for o in scene["objects"] if o["holes"]]
    active = [o for o in closed if o["color"] in mapping]
    _require(active, "no listed closed contour")
    irregular = False
    diagonal = False
    for obj in closed:
        mask = np.zeros(a.shape, dtype=bool)
        for q in obj["cells"]:
            mask[q] = True
        diagonal |= len(list(_components(mask))) > 1 and obj["color"] in mapping
        for cells in obj["holes"]:
            rr, cc = zip(*cells)
            irregular |= len(cells) != (max(rr)-min(rr)+1)*(max(cc)-min(cc)+1)
    return {"side": scene["side"], "background": bg, "key_length": len(mapping),
            "object_count": len(scene["objects"]),
            "hole_count": sum(len(o["holes"]) for o in closed),
            "max_holes": max(len(o["holes"]) for o in closed),
            "mapping": tuple(mapping.items()), "row_runs": row_runs, "col_runs": col_runs,
            "chain": any(mapping[o["color"]] in mapping
                         and mapping[mapping[o["color"]]] != mapping[o["color"]]
                         for o in active),
            "self_map": any(mapping[o["color"]] == o["color"] for o in active),
            "open_active": any(o["color"] in mapping and not o["holes"]
                               for o in scene["objects"]),
            "fill_only_inert": any(o["color"] in set(mapping.values())-set(mapping)
                                   for o in closed),
            "absent_inert": any(o["color"] not in set(mapping) | set(mapping.values())
                                for o in closed),
            "irregular": bool(irregular), "diagonal": bool(diagonal),
            "postfill_equal": _compact(out).shape != out.shape,
            "separated_repeat": _has_separated_repeat(a),
            "changed_cells": int(np.count_nonzero(a != out))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, facts = set(), {"train": [], "test": []}
    shapes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must contain exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            fp = _fingerprint(inp)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            facts[split].append(_analyze_pair(inp, out))
            if split == "train":
                shapes.add(out.shape)
    train = facts["train"]
    for feature in ("chain", "self_map", "open_active", "fill_only_inert",
                    "absent_inert", "irregular", "diagonal", "postfill_equal",
                    "separated_repeat"):
        _require(any(f[feature] for f in train), "missing training witness: " + feature)
    _require(any(max(f["row_runs"]) > 1 and max(f["col_runs"]) > 1
                 and len(set(f["row_runs"])) > 1 and len(set(f["col_runs"])) > 1
                 for f in train), "missing nonuniform two-axis training witness")
    _require(len(shapes) >= 2, "training lacks compact shape variation")
    _require(any(f["max_holes"] >= 2 for f in train),
             "training must show all cavities and repeated listed colors")
    sides = {f["side"] for f in train}
    _require(len(sides) >= 2, "training must illustrate more than one edge direction")
    _require(all(f["side"] not in sides for f in facts["test"]),
             "generated tests must transfer the rule to an unseen key side")


def self_test():
    """42 supplied seeds, streamed; no saved scenes, traces, logs or JSON files."""
    puzzles, shapes, compact_shapes, schedules = set(), set(), set(), set()
    sides, backgrounds, key_lengths, counts, hole_counts = set(), set(), set(), set(), set()
    features = {k: 0 for k in ("chain", "self_map", "open_active", "irregular",
                              "diagonal", "postfill_equal", "separated_repeat")}
    single_axis = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_digest(puzzle))
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)),
                 "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                before = inp.copy()
                _require(np.array_equal(solve(inp), out), "solver mismatch")
                _require(np.array_equal(before, inp), "solve mutated input")
                f = _analyze_pair(inp, out)
                shapes.add(inp.shape)
                compact_shapes.add(out.shape)
                schedules.add((f["row_runs"], f["col_runs"]))
                sides.add(f["side"])
                backgrounds.add(f["background"])
                key_lengths.add(f["key_length"])
                counts.add(f["object_count"])
                hole_counts.add(f["max_holes"])
                single_axis += (max(f["row_runs"]) == 1) != (max(f["col_runs"]) == 1)
                for name in features:
                    features[name] += bool(f[name])
                # Equivariance across every side and arbitrary color roles.
                shifted = (np.rot90(inp, 1).astype(int) + 3) % 10
                _require(np.array_equal(solve(shifted), (np.rot90(out, 1)+3) % 10),
                         "rotation/color-role regression")
    _require(len(puzzles) == 42 and len(shapes) >= 25
             and len(compact_shapes) >= 20 and len(schedules) >= 100,
             "insufficient structural variation")
    _require(sides == set(SIDES) and backgrounds == set(range(10)),
             "missing direction/background variation")
    _require(len(key_lengths) >= 4 and len(counts) >= 4 and 2 in hole_counts
             and single_axis > 0 and all(features.values()), "missing family variation")
    # Explicit local witnesses: consecutive collapse preserves separated repeats.
    motif = np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]], dtype=np.int8)
    enlarged = np.repeat(np.repeat(motif, [2, 1, 3], axis=0), [1, 4, 2], axis=1)
    _require(np.array_equal(_compact(enlarged), motif), "separated repeats lost")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(0, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "count mismatch")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(0)
    broken["test"][0]["output"] = _compact(broken["test"][0]["input"])
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("compression-only answer accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "compact_shapes": len(compact_shapes),
            "schedules": len(schedules), "key_lengths": sorted(key_lengths),
            "object_counts": sorted(counts), "single_axis_pairs": single_axis,
            "witness_counts": features}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
