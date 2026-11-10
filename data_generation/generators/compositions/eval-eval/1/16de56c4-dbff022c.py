"""Cadence woven through cavities. Standard library + NumPy; no file I/O.

The key supplies both a one-step contour lookup and a coordinate orientation.
Two contour-colored interior beads specify arithmetic period and phase in the
serpentine ordering of each cavity. Construction labels use sampled cavities;
solve reconstructs walls, cavities, key and beads solely from input pixels.
"""
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = _D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape):
        raise ValueError("grids must be 2D with dimensions 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid cells must be integers in 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(int(r), int(c))], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in (_D8 if diagonal else _D4):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(tuple(sorted(cells)))
    return result


def _holes(wall):
    """Interior components of a single wall, treating every other cell as air."""
    h, w = wall.shape
    holes = []
    for cells in _components(~wall):
        if not any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            holes.append(cells)
    return holes


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("input background must occupy more than half the grid")
    return int(colors[np.argmax(counts)])


def _find_key(a, bg):
    candidates = []
    h, w = a.shape
    for cells in _components(a != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1-r0+1, c1-c0+1
        if len(cells) != sh*sw or len({int(a[q]) for q in cells}) < 2:
            continue
        lanes = []
        # Length >=3 removes the two-direction ambiguity of a corner 2x2 key.
        if sh == 2 and sw >= 3:
            if r0 == 0:
                lanes.append(("top", a[r0, c0:c1+1], a[r1, c0:c1+1]))
            if r1 == h-1:
                lanes.append(("bottom", a[r1, c0:c1+1], a[r0, c0:c1+1]))
        if sw == 2 and sh >= 3:
            if c0 == 0:
                lanes.append(("left", a[r0:r1+1, c0], a[r0:r1+1, c1]))
            if c1 == w-1:
                lanes.append(("right", a[r0:r1+1, c1], a[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            source, fill = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(source)) == len(source):
                candidates.append({"side": side, "mapping": dict(zip(source, fill)),
                                   "cells": frozenset(cells), "sources": source,
                                   "fills": fill})
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous local two-lane edge key")
    return candidates[0]


def _order(cells, shape, side, mode="snake"):
    """Rotate coordinates to a top key; first row runs left to right."""
    h, w = shape
    k = _TURNS[side]
    def coordinate(q):
        r, c = q
        return ((r, c), (w-1-c, r), (h-1-r, w-1-c), (c, h-1-r))[k]
    rows = {}
    for q in cells:
        r, c = coordinate(q)
        rows.setdefault(r, []).append((c, q))
    result = []
    for i, r in enumerate(sorted(rows)):
        row = sorted(rows[r], reverse=(mode == "snake" and i % 2 == 1))
        result.extend(q for _, q in row)
    return result


def _scene(a):
    """Recover complete contour components before examining their seed beads."""
    bg = _background(a)
    key = _find_key(a, bg)
    wall_records = []
    wall_cells, interior_cells = set(), set()
    # A bead/dimer does not enclose any cell and cannot become a wall.
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        mask = a == color
        for q in key["cells"]:
            mask[q] = False
        for cells in _components(mask, diagonal=True):
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            local = np.zeros((r1-r0+3, c1-c0+3), dtype=bool)
            for r, c in cells:
                local[r-r0+1, c-c0+1] = True
            holes = [tuple((r+r0-1, c+c0-1) for r, c in region) for region in _holes(local)]
            if not holes:
                continue
            wall_records.append({"color": color, "wall": frozenset(cells), "holes": holes})
            wall_cells.update(cells)
            for hole in holes:
                if interior_cells.intersection(hole):
                    raise ValueError("nested or overlapping cavities are excluded")
                interior_cells.update(hole)
    if not wall_records:
        raise ValueError("no closed contour")
    if wall_cells.intersection(interior_cells) or key["cells"].intersection(interior_cells):
        raise ValueError("nested objects or enclosed key")
    records = []
    for wall in wall_records:
        color = wall["color"]
        for hole in wall["holes"]:
            ordered = _order(hole, a.shape, key["side"])
            beads = [i for i, q in enumerate(ordered) if int(a[q]) == color]
            if len(beads) != 2:
                raise ValueError("each cavity needs exactly two contour-colored beads")
            first, second = beads
            period = second-first
            for i in beads:
                r, c = ordered[i]
                if any((r+dy, c+dx) in wall["wall"] for dy, dx in _D8):
                    raise ValueError("reference beads must be separate from the wall")
            extras = [i for i, q in enumerate(ordered) if int(a[q]) not in (bg, color)]
            if any((i-first) % period == 0 for i in extras):
                raise ValueError("singletons must be off phase")
            if len({int(a[ordered[i]]) for i in extras}) != len(extras):
                raise ValueError("extra bead colors must be locally unique")
            records.append({"color": color, "wall": wall["wall"], "cells": tuple(ordered),
                            "a": first, "b": second, "period": period, "extras": tuple(extras),
                            "active": color in key["mapping"]})
    return bg, key, records


def _paint(a, key, records, policy="rule"):
    out = a.copy()
    for record in records:
        color = record["color"]
        if policy != "all_listed" and not record["active"]:
            continue
        fill = key["mapping"].get(color, color)
        if policy == "reference_color":
            fill = color
        if policy == "recursive":
            visited = {color}
            while fill in key["mapping"] and fill not in visited:
                visited.add(fill)
                fill = key["mapping"][fill]
        cells = record["cells"]
        if policy == "raster":
            cells = _order(cells, a.shape, key["side"], "raster")
        elif policy == "fixed_top":
            cells = _order(cells, a.shape, "top")
        indices = [i for i, q in enumerate(cells) if int(a[q]) == color]
        first, second = indices
        period = second-first
        local_indices = {}
        if policy == "row_reset":
            k = _TURNS[key["side"]]
            def depth(q):
                r, c = q
                return (r, a.shape[1]-1-c, a.shape[0]-1-r, c)[k]
            counter = {}
            for q in cells:
                d = depth(q)
                local_indices[q] = counter.get(d, 0)
                counter[d] = local_indices[q]+1
        for i, q in enumerate(cells):
            choose = (i-first) % period == 0
            if policy == "whole_hole":
                choose = True
            elif policy == "row_reset":
                choose = (local_indices[q]-first) % period == 0
            elif policy == "forward_only":
                choose = choose and i >= first
            elif policy == "backward_only":
                choose = choose and i <= second
            if choose:
                out[q] = fill
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    _, key, records = _scene(a)
    return _paint(a, key, records)


def _dilate(mask, diagonal=False):
    result = mask.copy()
    h, w = mask.shape
    for dy, dx in (_D8 if diagonal else _D4):
        r0, r1 = max(0, dy), min(h, h+dy)
        c0, c1 = max(0, dx), min(w, w+dx)
        result[r0:r1, c0:c1] |= mask[r0-dy:r1-dy, c0-dx:c1-dx]
    return result


def _safe(cavity):
    result = cavity.copy()
    h, w = cavity.shape
    for r, c in zip(*np.nonzero(cavity)):
        if any(not (0 <= r+dy < h and 0 <= c+dx < w) or not cavity[r+dy, c+dx] for dy, dx in _D8):
            result[r, c] = False
    return result


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_shape(rng, kind, small=False):
    """Return one wall and its exact intended cavities in local coordinates."""
    h = int(rng.integers(5, 8 if small else 12))
    w = int(rng.integers(6, 9 if small else 14))
    if kind == "diamond":
        radius = int(rng.integers(3, 5 if small else 6))
        y, x = np.indices((2*radius+1, 2*radius+1))
        cavity = abs(y-radius) + abs(x-radius) < radius
    else:
        cavity = np.ones((h, w), dtype=bool)
        if kind == "profile":
            # Row-convex, connected irregular shapes with a substantial common core.
            for r in range(h):
                left = int(rng.integers(0, min(3, w//3)))
                right = int(rng.integers(0, min(3, w//3)))
                cavity[r, :left] = False
                if right:
                    cavity[r, -right:] = False
        elif kind == "dumbbell":
            mid = h//2
            cavity[mid, :] = False
            neck = int(rng.integers(1, max(2, w-2)))
            cavity[mid, neck:neck+2] = True
    cavity = np.pad(cavity, 2)
    thickness = 1 if small or rng.random() < .7 else 2
    wall = cavity.copy()
    for _ in range(thickness):
        wall = _dilate(wall)
    wall &= ~cavity
    rr, cc = np.nonzero(wall)
    wall = wall[rr.min():rr.max()+1, cc.min():cc.max()+1]
    cavity = cavity[rr.min():rr.max()+1, cc.min():cc.max()+1]
    if kind == "multi":
        # Two unequal holes joined by a solid bridge. Both reset their own ruler.
        other_h = int(rng.integers(3, 6))
        other_w = int(rng.integers(4, 8))
        other = np.zeros((other_h+2, other_w+2), dtype=bool)
        other[0, :] = other[-1, :] = True
        other[:, 0] = other[:, -1] = True
        big = np.zeros((max(wall.shape[0], other.shape[0]), wall.shape[1]+other.shape[1]+1), dtype=bool)
        big[:wall.shape[0], :wall.shape[1]] = wall
        big[:other.shape[0], wall.shape[1]+1:] = other
        big[1, wall.shape[1]-1:wall.shape[1]+2] = True
        wall = big
    turn = int(rng.integers(4))
    wall = np.rot90(wall, turn).copy()
    # Padding ensures that local edge contact is not confused with leakage.
    holes = _holes(np.pad(wall, 1))
    holes = [tuple((r-1, c-1) for r, c in hole) for hole in holes]
    return wall, holes


def _beads(rng, cells, shape, side, harder=False, witness=False, bidirectional=False):
    ordered = _order(cells, shape, side)
    cavity = np.zeros(shape, dtype=bool)
    for q in cells:
        cavity[q] = True
    safe = _safe(cavity)
    options = [i for i, q in enumerate(ordered) if safe[q]]
    candidates = []
    for ai, first in enumerate(options):
        for second in options[ai+1:]:
            p = second-first
            positions = list(range(first % p, len(ordered), p))
            if len(positions) < 3:
                continue
            if witness and (p < 2 or len(positions) == len(ordered)):
                continue
            if bidirectional and (first < p or second+p >= len(ordered)):
                continue
            candidates.append((first, second))
    if not candidates:
        return None
    if harder:
        preferred = [v for v in candidates if 9 <= v[1]-v[0] <= max(9, len(ordered)//3)]
    else:
        preferred = [v for v in candidates if 1 <= v[1]-v[0] <= 8]
        if not preferred:
            return None
    first, second = _pick(rng, preferred or candidates)
    return ordered, first, second


def _put_key(a, side, sources, fills, start):
    for i, (s, f) in enumerate(zip(sources, fills)):
        p = start+i
        if side == "top":
            a[0, p], a[1, p] = s, f
        elif side == "bottom":
            a[-1, p], a[-2, p] = s, f
        elif side == "left":
            a[p, 0], a[p, 1] = s, f
        else:
            a[p, -1], a[p, -2] = s, f


def _build_pair(rng, side, lesson=None, test=False, rescue=False):
    # Canonical role colors are permuted together after geometry is complete.
    length = int(rng.integers(3, 6))
    sources = list(range(1, length+1))
    fills = [int(x) for x in rng.integers(1, 9, length)]
    fills[0] = 2
    fills[1] = 6  # a visible nontrivial one-step chain 1 -> 2 -> 6
    if rng.random() < .35:
        fills[-1] = sources[-1]
    order = rng.permutation(length)
    sources, fills = [sources[i] for i in order], [fills[i] for i in order]
    mapping = dict(zip(sources, fills))
    if lesson == "inert":
        # Both fill-only and absent-from-both-lanes identities can be inert.
        colors = [1, int(_pick(rng, [6, 9]))]
    else:
        count = 1 if lesson == "weave" else int(rng.integers(1, 4))
        colors = [1] + [int(_pick(rng, sources+[6, 9])) for _ in range(count-1)]
    kinds = []
    for i in range(len(colors)):
        if lesson == "weave" and i == 0:
            kinds.append(str(_pick(rng, ["profile", "diamond", "dumbbell"])))
        else:
            kinds.append(str(_pick(rng, ["rectangle", "profile", "diamond", "dumbbell", "multi"])))
    if rescue:
        kinds = ["profile" if lesson == "weave" else "rectangle"] * len(colors)
    shapes = [_make_shape(rng, kind, small=len(colors) > 1 or rescue) for kind in kinds]
    if any(max(wall.shape) > 26 for wall, _ in shapes):
        return None
    h = 30 if rescue else int(rng.integers(max(14, max(wall.shape[0] for wall, _ in shapes)+4), 31))
    w = 30 if rescue else int(rng.integers(max(14, max(wall.shape[1] for wall, _ in shapes)+4), 31))
    if not rescue and rng.random() < .2:
        h = w = max(h, w)
    a = np.zeros((h, w), dtype=np.int8)
    axis = w if side in ("top", "bottom") else h
    start = int(_pick(rng, [0, axis-length])) if rng.random() < .25 else int(rng.integers(axis-length+1))
    _put_key(a, side, sources, fills, start)
    blocked = _dilate(a != 0, True)
    placed = []
    # Placement has no fixed slots. Reserve each complete bounding box + halo.
    for (wall, holes), color in sorted(zip(shapes, colors), key=lambda v: -v[0][0].size):
        sh, sw = wall.shape
        options = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)
                   if not np.any(blocked[r:r+sh, c:c+sw])]
        if not options:
            return None
        edge_options = [(r, c) for r, c in options if r == 0 or c == 0 or r+sh == h or c+sw == w]
        r, c = _pick(rng, edge_options if edge_options and rng.random() < .35 else options)
        patch = a[r:r+sh, c:c+sw]
        patch[wall] = color
        blocked[max(0, r-1):min(h, r+sh+1), max(0, c-1):min(w, c+sw+1)] = True
        placed.append((color, [(tuple((y+r, x+c) for y, x in hole)) for hole in holes]))
    out = a.copy()
    for color, holes in placed:
        for cells in holes:
            selection = _beads(rng, cells, a.shape, side, harder=test,
                               witness=lesson in ("weave", "inert"),
                               bidirectional=lesson == "weave")
            if selection is None:
                return None
            ordered, first, second = selection
            a[ordered[first]] = a[ordered[second]] = color
            period = second-first
            # Optional locally unique off-phase beads; taught in the inert lesson.
            extras = [i for i, q in enumerate(ordered) if (i-first) % period and i not in (first, second)]
            noise = 1 if lesson == "inert" and extras else int(rng.integers(0, 3)) if rng.random() < .4 else 0
            noise = min(noise, len(extras))
            extra_colors = [x for x in range(1, 10) if x != color]
            rng.shuffle(extra_colors)
            for i, extra in zip(rng.choice(extras, noise, replace=False), extra_colors):
                a[ordered[int(i)]] = extra
            # Labels are painted directly from this sampled cavity and its beads.
            for q in ordered:
                out[q] = a[q]
            if color in mapping:
                for i in range(first % period, len(ordered), period):
                    out[ordered[i]] = mapping[color]
    if np.count_nonzero(a == 0) * 2 <= a.size:
        return None
    palette = np.asarray(rng.permutation(10), dtype=np.int8)
    a, out = palette[a], palette[out]
    try:
        bg, key, records = _scene(a)
    except ValueError:
        return None
    if not np.array_equal(_paint(a, key, records), out):
        return None
    if test and not any(r["active"] and r["period"] > 8 for r in records):
        return None
    if lesson == "weave":
        if not all(not np.array_equal(_paint(a, key, records, policy), out)
                   for policy in ("whole_hole", "reference_color", "raster", "row_reset", "recursive",
                                  "forward_only", "backward_only")):
            return None
    if lesson == "inert":
        if not any(not r["active"] for r in records) or not any(r["active"] and r["extras"] for r in records):
            return None
    if lesson and side != "top" and np.array_equal(_paint(a, key, records, "fixed_top"), out):
        return None
    return {"input": a, "output": out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    sides = list(map(str, rng.permutation(["top", "right", "bottom", "left"])))
    familiar = bool(rng.integers(2))
    train_sides = [sides[i % 2] for i in range(num_train)]
    test_sides = [sides[int(rng.integers(2))] if familiar else sides[2+i % 2] for i in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, side_list in (("train", train_sides), ("test", test_sides)):
        for i, side in enumerate(side_list):
            lesson = ("weave" if i == 0 else "inert" if i == 1 else None) if split == "train" else None
            for attempt in range(160):
                pair = _build_pair(rng, side, lesson, split == "test", rescue=attempt >= 120)
                if pair is None:
                    continue
                signature = (pair["input"].shape, pair["input"].tobytes())
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, out=None):
    bg, key, records = _scene(a)
    if out is None:
        out = _paint(a, key, records)
    policies = ("whole_hole", "reference_color", "raster", "row_reset", "fixed_top", "recursive", "all_listed",
                "forward_only", "backward_only")
    distinguishes = {p: not np.array_equal(_paint(a, key, records, p), out) for p in policies}
    return {"side": key["side"], "background": bg, "key_length": len(key["mapping"]),
            "walls": len({r["wall"] for r in records}), "cavities": len(records),
            "periods": tuple(r["period"] for r in records),
            "sizes": tuple(len(r["cells"]) for r in records),
            "extras": sum(len(r["extras"]) for r in records),
            "active_extras": sum(len(r["extras"]) for r in records if r["active"]),
            "inert": sum(not r["active"] for r in records),
            "irregular": any(len(r["cells"]) != (max(q[0] for q in r["cells"])-min(q[0] for q in r["cells"])+1) *
                             (max(q[1] for q in r["cells"])-min(q[1] for q in r["cells"])+1) for r in records),
            "multi": len(records) > len({r["wall"] for r in records}),
            "distinguishes": distinguishes}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            signature = (a.shape, a.tobytes())
            assert signature not in seen, "duplicate input"
            seen.add(signature)
            bg, key, records = _scene(a)
            assert np.array_equal(_paint(a, key, records), out), "incorrect weave"
            assert np.any(a != out), "each example must have a visible change"
            walls = set().union(*(r["wall"] for r in records))
            assert all(a[q] == out[q] for q in walls | key["cells"]), "key or wall altered"
            eligible = {q for r in records if r["active"] for q in r["cells"]}
            assert all(tuple(q) in eligible for q in np.argwhere(a != out)), "paint outside eligible cavities"
            for r in records:
                period = r["period"]
                positions = list(range(r["a"] % period, len(r["cells"]), period))
                assert len(positions) >= 3, "no visible extrapolation"
                assert all(a[r["cells"][i]] == out[r["cells"][i]] for i in r["extras"]), "singleton changed"
            features[split].append(_features(a, out))
    train = features["train"]
    for policy in ("whole_hole", "reference_color", "raster", "row_reset", "fixed_top", "recursive", "all_listed",
                   "forward_only", "backward_only"):
        assert any(f["distinguishes"][policy] for f in train), "missing demonstration: " + policy
    assert any(f["irregular"] for f in train)
    assert any(f["active_extras"] for f in train) and any(f["inert"] for f in train)
    assert len({f["side"] for f in train}) >= 2


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, geometric regressions, symmetry, and mutation checks."""
    signatures, shapes, periods, sides, bgs, counts = set(), set(), set(), set(), set(), set()
    irregular = multi = noise = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                f = _features(a, out)
                shapes.add(a.shape)
                periods.update(f["periods"])
                sides.add(f["side"])
                bgs.add(f["background"])
                counts.add(f["walls"])
                irregular += f["irregular"]
                multi += f["multi"]
                noise += f["extras"] > 0
                assert np.array_equal(solve(a.copy()), out)
        if seed in (0, 17, 41):
            assert _fingerprint(generate_puzzle(seed)) == _fingerprint(puzzle)
            pair = puzzle["train"][0]
            for turn in range(4):
                a, b = np.rot90(pair["input"], turn), np.rot90(pair["output"], turn)
                assert np.array_equal(solve(a), b)
                assert np.array_equal(solve(9-a), 9-b)
    assert len(signatures) == 42 and len(shapes) >= 30
    assert sides == {"top", "right", "bottom", "left"} and bgs == set(range(10))
    assert len(periods) >= 12 and {1, 2, 3} <= periods
    assert {1, 2, 3} <= counts and irregular >= 42 and multi >= 5 and noise >= 42
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1000+tr*10+te, tr, te)
            assert len(p["train"]) == tr and len(p["test"]) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact small fixture: top key, 5x6 interior, p=3 across folded rows.
    a = np.zeros((20, 24), dtype=np.int8)
    a[:2, 1:4] = [[1, 2, 3], [2, 6, 3]]
    a[4, 2:10] = a[10, 2:10] = 1
    a[4:11, 2] = a[4:11, 9] = 1
    cells = [(r, c) for r in range(5, 10) for c in range(3, 9)]
    ordered = _order(cells, a.shape, "top")
    a[ordered[10]] = a[ordered[13]] = 1
    # A same-color open outline with its own beads is preserved in full.
    a[13, 15:23] = a[18, 15:23] = 1
    a[13:19, 15] = a[13:19, 22] = 1
    a[13, 17] = 0
    a[16, 18] = a[16, 20] = 1
    expected = a.copy()
    for i in (1, 4, 7, 10, 13, 16, 19, 22, 25, 28):
        expected[ordered[i]] = 2
    assert np.array_equal(solve(a), expected)
    # A diagonal outline closes a four-connected cavity, despite eight-neighbor
    # background leakage. Expected paint locations are specified by hand.
    diamond = np.zeros((14, 16), dtype=np.int8)
    diamond[:2, 1:4] = [[1, 2, 3], [2, 6, 3]]
    yy, xx = np.indices((9, 9))
    diamond[4:13, 5:14][abs(yy-4)+abs(xx-4) == 4] = 1
    diamond[7, 9] = diamond[8, 8] = 1
    expected_diamond = diamond.copy()
    for q in ((7, 9), (8, 8), (9, 11)):
        expected_diamond[q] = 2
    assert np.array_equal(solve(diamond), expected_diamond)
    broken = generate_puzzle(0)
    mutated = {s: [{k: v.copy() for k, v in p.items()} for p in broken[s]] for s in ("train", "test")}
    q = next(iter(_scene(mutated["test"][0]["input"])[1]["cells"]))
    mutated["test"][0]["output"][q] = (int(mutated["test"][0]["output"][q])+1) % 10
    try:
        validate_generated_puzzle(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("key mutation accepted")
