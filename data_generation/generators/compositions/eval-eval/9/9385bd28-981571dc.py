"""Local two-mirror stencil completion with ordered, opaque compositing.

No parent modules, files, labels, random state, or puzzle context enter solve.
All table swatches are literal colors; blank parts of every box are opaque.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _signature(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _corners(bounds, diagonal):
    r0, c0, r1, c1 = bounds
    if diagonal == 0:
        return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
                (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}
    return {(r0, c1), (r0, c1 - 1), (r0 + 1, c1),
            (r1, c0), (r1, c0 + 1), (r1 - 1, c0)}


def _inside(point, bounds):
    r, c = point
    r0, c0, r1, c1 = bounds
    return r0 <= r <= r1 and c0 <= c <= c1


def _interior(point, bounds):
    r, c = point
    r0, c0, r1, c1 = bounds
    return r0 < r < r1 and c0 < c < c1


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _orbit(point, bounds):
    r, c = point
    r0, c0, r1, c1 = bounds
    return tuple(sorted({(y, x) for y in (r, r0 + r1 - r)
                         for x in (c, c0 + c1 - c)}))


def _orbits(bounds):
    r0, c0, r1, c1 = bounds
    return [_orbit((r, c), bounds)
            for r in range(r0 + 1, (r0 + r1) // 2 + 1)
            for c in range(c0 + 1, (c0 + c1) // 2 + 1)]


def _occupied(obj):
    return {p for seed in obj["seeds"] for p in _orbit(seed, obj["bounds"])}


def _parse(input_grid):
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("expected a two-dimensional integer grid")
    if not all(1 <= s <= 30 for s in grid.shape) or np.any((grid < 0) | (grid > 9)):
        raise ValueError("invalid grid size or color")
    h, w = grid.shape
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= grid.size // 2:
        raise ValueError("the canvas background must occupy more than half the input")
    candidates = []
    for c in range(w - 1):
        rows = np.flatnonzero(grid[:, c] != bg)
        runs = np.split(rows, np.flatnonzero(np.diff(rows) != 1) + 1)
        for run in runs:
            if not 1 <= len(run) <= 4:
                continue
            r0, r1 = int(run[0]), int(run[-1])
            cues = list(map(int, grid[r0:r1 + 1, c]))
            if len(set(cues)) != len(cues):
                continue
            table = (r0, c, r1, c + 1)
            moat = (max(0, r0 - 1), max(0, c - 1), min(h - 1, r1 + 1), min(w - 1, c + 2))
            if any(grid[r, cc] != bg
                   for r in range(max(0, r0 - 1), min(h, r1 + 2))
                   for cc in range(max(0, c - 1), min(w, c + 3))
                   if not _inside((r, cc), table)):
                continue
            scene = grid.copy()
            scene[r0:r1 + 1, c:c + 2] = bg
            if set(map(int, np.unique(scene))) != {bg, *cues}:
                continue
            objects = []
            valid = True
            for rank, cue in enumerate(cues):
                points = {tuple(map(int, p)) for p in np.argwhere(scene == cue)}
                if len(points) < 7:
                    valid = False
                    break
                bound = (min(p[0] for p in points), min(p[1] for p in points),
                         max(p[0] for p in points), max(p[1] for p in points))
                if min(bound[2] - bound[0], bound[3] - bound[1]) < 4 or _overlap(bound, moat):
                    valid = False
                    break
                edge = {p for p in points if not _interior(p, bound)}
                diagonals = [d for d in (0, 1) if edge == _corners(bound, d)]
                if len(diagonals) != 1:
                    valid = False
                    break
                seeds = points - edge
                if not seeds:
                    valid = False
                    break
                objects.append(dict(cue=cue, paint=int(grid[r0 + rank, c + 1]),
                                    bounds=bound, markers=edge, seeds=seeds,
                                    diagonal=diagonals[0]))
            if valid:
                candidates.append(dict(background=bg, table=table, objects=objects))
    if len(candidates) != 1:
        raise ValueError("input must identify exactly one table and stencil scene")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local boxes, close seed orbits, and choose the foremost layer per cell."""
    grid = np.asarray(input_grid)
    parsed = _parse(grid)
    result = grid.copy()
    objects = [dict(obj, occupied=_occupied(obj)) for obj in parsed["objects"]]
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            p = (r, c)
            for obj in objects:
                if _inside(p, obj["bounds"]):
                    if p in obj["markers"]:
                        result[p] = obj["cue"]
                    elif p in obj["occupied"]:
                        result[p] = obj["paint"]
                    else:
                        result[p] = parsed["background"]
                    break
    return result


def _compose(grid, objects, bg, *, solid=False, reverse=False, use_cues=False, transparent=False):
    """Generation compositor and explicit simpler-rule ablations."""
    result = grid.copy()
    ordered = list(reversed(objects)) if reverse else objects
    for obj in reversed(ordered):
        r0, c0, r1, c1 = obj["bounds"]
        if not transparent:
            result[r0:r1 + 1, c0:c1 + 1] = bg
        occupied = (_occupied(obj) if not solid else
                    {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)})
        for p in occupied:
            result[p] = obj["cue"] if use_cues else obj["paint"]
        for p in obj["markers"]:
            result[p] = obj["cue"]
    return result


def _features(grid, parsed, output):
    bg, objs = parsed["background"], parsed["objects"]
    occupied = [_occupied(o) for o in objs]
    f = Counter()
    edges = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))
             if _overlap(objs[i]["bounds"], objs[j]["bounds"])]
    def contains(a, b):
        return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]
    f["overlap_edges"] = len(edges)
    f["disjoint_scene"] = int(len(objs) > 1 and not edges)
    f["nested_scene"] = int(any(contains(objs[i]["bounds"], objs[j]["bounds"]) or
                                  contains(objs[j]["bounds"], objs[i]["bounds"]) for i, j in edges))
    f["partial_overlap"] = int(any(not contains(objs[i]["bounds"], objs[j]["bounds"]) and
                                    not contains(objs[j]["bounds"], objs[i]["bounds"]) for i, j in edges))
    f["shared_paint"] = int(len({o["paint"] for o in objs}) < len(objs))
    f["mixed_diagonals"] = int(len({o["diagonal"] for o in objs}) > 1)
    tr0, tc0, tr1, tc1 = parsed["table"]
    f["interior_table"] = int(tr0 > 0 and tc0 > 0 and tr1 < grid.shape[0] - 1 and tc1 < grid.shape[1] - 1)
    for i, obj in enumerate(objs):
        b = obj["bounds"]
        f["layers"] += 1
        f["seeds"] += len(obj["seeds"])
        f["background_paint"] += int(obj["paint"] == bg)
        f["self_paint"] += int(obj["paint"] == obj["cue"])
        f["other_cue_paint"] += int(obj["paint"] != obj["cue"] and obj["paint"] in {o["cue"] for o in objs})
        hh, ww = b[2] - b[0] + 1, b[3] - b[1] + 1
        f["square_box"] += int(hh == ww)
        f["horizontal_box"] += int(ww >= 2 * hh)
        f["vertical_box"] += int(hh >= 2 * ww)
        quadrants = {(int(np.sign(2 * r - b[0] - b[2])), int(np.sign(2 * c - b[1] - b[3])))
                     for r, c in obj["seeds"]
                     if 2 * r != b[0] + b[2] and 2 * c != b[1] + b[3]}
        f["source_spread"] += int(len(quadrants) > 1)
        for orbit in _orbits(b):
            seeds = set(orbit) & obj["seeds"]
            f["redundant_orbit"] += int(len(seeds) > 1)
            f["axis_seed"] += int(bool(seeds) and len(orbit) < 4)
            if len(orbit) == 4 and len(seeds) == 1 and obj["paint"] != bg:
                p = next(iter(seeds))
                rr, cc = b[0] + b[2] - p[0], b[1] + b[3] - p[1]
                targets = {(rr, p[1]), (p[0], cc), (rr, cc)}
                if all(not any(_inside(q, upper["bounds"]) for upper in objs[:i])
                       and output[q] == obj["paint"] for q in targets):
                    f["two_folds"] += 1
            if not seeds:
                f["empty_orbits"] += 1
        for p in obj["markers"]:
            covering = [j for j in range(i) if _inside(p, objs[j]["bounds"])]
            if covering and output[p] != obj["cue"]:
                f["corner_occlusion"] += 1
            elif not covering and output[p] == obj["cue"]:
                f["corner_retention"] += 1
        for p in occupied[i]:
            if not any(_inside(p, up["bounds"]) for up in objs[:i]):
                if output[p] == obj["paint"] and obj["paint"] != obj["cue"]:
                    f["recolor"] += 1
        for p in occupied[i]:
            if obj["paint"] == bg:
                continue
            covering = [j for j in range(i) if _inside(p, objs[j]["bounds"])]
            if covering:
                top = covering[0]
                if p not in occupied[top] and p not in objs[top]["markers"] and output[p] == bg:
                    f["opaque_blank"] += 1
    for name, kwargs in (("priority", dict(reverse=True)), ("needs_mirrors", dict(solid=True)),
                         ("needs_mapping", dict(use_cues=True)), ("needs_opacity", dict(transparent=True))):
        f[name] = int(not np.array_equal(output, _compose(grid, objs, bg, **kwargs)))
    return f


def _sample_bounds(rng, h, w):
    style = int(rng.integers(5))
    if style == 0:
        hh, ww = int(rng.integers(5, min(h, 8) + 1)), int(rng.integers(7, w + 1))
    elif style == 1:
        hh, ww = int(rng.integers(7, h + 1)), int(rng.integers(5, min(w, 8) + 1))
    elif style == 2:
        hh = ww = int(rng.integers(5, min(h, w, 15) + 1))
    else:
        hh, ww = int(rng.integers(5, h + 1)), int(rng.integers(5, w + 1))
    r, c = int(rng.integers(h - hh + 1)), int(rng.integers(w - ww + 1))
    return (r, c, r + hh - 1, c + ww - 1)


def _seed_motif(rng, bounds, forbidden, distributed):
    orbits = _orbits(bounds)
    available = [o for o in orbits if any(p not in forbidden for p in o)]
    if not available:
        return None
    representatives = [o[0] for o in available]
    style = int(rng.integers(5))
    center = representatives[int(rng.integers(len(representatives)))]
    if style == 0:  # Sparse fragments.
        scores = rng.random(len(available))
    elif style == 1:  # Compact patches, then reflected.
        scores = [abs(r - center[0]) + abs(c - center[1]) + rng.random() * .8
                  for r, c in representatives]
    elif style == 2:  # Bars with occasional branches.
        vertical = bool(rng.integers(2))
        scores = [3 * abs((c if vertical else r) - (center[1] if vertical else center[0]))
                  + rng.random() for r, c in representatives]
    elif style == 3:  # Steps / diagonal fragments.
        scores = [abs((r - center[0]) - (c - center[1])) + rng.random()
                  for r, c in representatives]
    else:  # Hollow outlines in a fragment.
        radius = int(rng.integers(1, 4))
        scores = [abs(max(abs(r - center[0]), abs(c - center[1])) - radius) + rng.random()
                  for r, c in representatives]
    count = int(rng.integers(1, min(11, max(1, len(available) - 1)) + 1))
    chosen = [available[int(i)] for i in np.argsort(scores)[:count]]
    quadrant = int(rng.integers(4))
    def source_side(choices):
        return max(choices, key=lambda p: ((1 if quadrant // 2 else -1) * p[0],
                                          (1 if quadrant % 2 else -1) * p[1]))
    seeds = set()
    for orbit in chosen:
        choices = [p for p in orbit if p not in forbidden]
        seeds.add(choices[int(rng.integers(len(choices)))] if distributed else source_side(choices))
        if distributed and len(choices) > 1 and len(seeds) < 14 and rng.random() < .24:
            seeds.add(choices[int(rng.integers(len(choices)))])
    # Deliberately retain a singly seeded four-way orbit as a mirror witness.
    four = [o for o in available if len(o) == 4]
    if four:
        orbit = four[int(rng.integers(len(four)))]
        seeds.difference_update(orbit)
        choices = [p for p in orbit if p not in forbidden]
        seeds.add(choices[int(rng.integers(len(choices)))] if distributed else source_side(choices))
    return seeds


def _make_pair(rng, n, size_range, distributed, require, background_paint):
    """Bounded spatial rejection; all semantic checks use returned arrays."""
    for _ in range(300):
        lo, hi = size_range
        h, w = map(int, rng.integers(max(lo, 10 if n == 1 else 14), hi + 1, size=2))
        bg = int(rng.integers(10))
        cues = list(map(int, rng.choice([c for c in range(10) if c != bg], n, replace=False)))
        table_r, table_c = int(rng.integers(h - n + 1)), int(rng.integers(w - 1))
        table = (table_r, table_c, table_r + n - 1, table_c + 1)
        moat = (max(0, table[0] - 1), max(0, table[1] - 1),
                min(h - 1, table[2] + 1), min(w - 1, table[3] + 1))
        objects, used = [], set()
        topology = str(rng.choice(["free", "free", "nested", "chain", "disjoint"]))
        if "priority" in require:
            topology = str(rng.choice(["free", "nested", "chain"]))
        for i in range(n):
            found = None
            for _ in range(100):
                b = _sample_bounds(rng, h, w)
                if _overlap(b, moat):
                    continue
                if i and topology == "nested":
                    a = objects[0]["bounds"]
                    if not (a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]):
                        continue
                if i and topology == "disjoint" and any(_overlap(b, o["bounds"]) for o in objects):
                    continue
                if i and topology == "chain" and not _overlap(b, objects[-1]["bounds"]):
                    continue
                if any(b == o["bounds"] for o in objects):
                    continue
                d = int(rng.integers(2))
                markers = _corners(b, d)
                if markers & used:
                    continue
                found = dict(bounds=b, markers=markers, diagonal=d)
                break
            if found is None:
                break
            used.update(found["markers"])
            objects.append(found)
        if len(objects) != n:
            continue
        objects = [objects[int(i)] for i in rng.permutation(n)]
        used = set().union(*(o["markers"] for o in objects))
        for cue, obj in zip(cues, objects):
            obj["cue"] = cue
            obj["paint"] = int(rng.choice([c for c in range(10) if c not in (bg, cue)]))
            obj["seeds"] = _seed_motif(rng, obj["bounds"], used, distributed)
            if not obj["seeds"]:
                break
            used.update(obj["seeds"])
        if any(not o.get("seeds") for o in objects):
            continue
        if n >= 2 and background_paint:
            objects[0]["paint"] = bg
        elif not require:
            for obj in objects:
                if rng.random() < .18:
                    obj["paint"] = obj["cue"]
                if background_paint and rng.random() < .13:
                    obj["paint"] = bg
            if n > 1 and rng.random() < .22:
                objects[-1]["paint"] = objects[0]["paint"]
        if "priority" in require:
            upper, lower = objects[:2]
            if lower["paint"] == upper["paint"]:
                lower["paint"] = int(rng.choice([c for c in range(10)
                                                 if c not in (bg, lower["cue"], upper["paint"])]))
            overlap = [p for orbit in _orbits(lower["bounds"]) for p in orbit
                       if _interior(p, upper["bounds"]) and p not in used - lower["seeds"]]
            if not overlap:
                continue
            p = overlap[int(rng.integers(len(overlap)))]
            upper["seeds"].difference_update(_orbit(p, upper["bounds"]))
            lower["seeds"].add(p)
        grid = np.full((h, w), bg, dtype=np.int8)
        for obj in objects:
            for p in obj["markers"] | obj["seeds"]:
                grid[p] = obj["cue"]
        for i, obj in enumerate(objects):
            grid[table_r + i, table_c:table_c + 2] = obj["cue"], obj["paint"]
        output = _compose(grid, objects, bg)
        try:
            parsed = _parse(grid)
        except ValueError:
            continue
        f = _features(grid, parsed, output)
        if all(f[k] > 0 for k in require) and not np.array_equal(grid, output):
            return dict(input=grid, output=output)
    raise RuntimeError("bounded stencil sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 to 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1..4 and smaller than num_train")
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime = int(rng.integers(3))  # Count, source-fragment placement, or scale shift.
    background_paint = bool(rng.random() < .25)
    basic, interaction = map(int, rng.choice(num_train, 2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            require = set()
            if split == "train" and index == basic:
                n = 1
                require = {"two_folds", "recolor", "empty_orbits", "needs_mirrors", "corner_retention"}
            elif split == "train" and index == interaction:
                n = 2
                require = {"priority", "opaque_blank", "corner_occlusion", "needs_opacity"}
            else:
                high = 3 if regime == 0 and split == "train" else 4
                n = int(rng.integers(3, 5)) if split == "test" and regime == 0 else int(rng.integers(1, high))
            if split == "test":
                require = {"two_folds", "needs_mirrors", "needs_mapping"}
                if regime == 1:
                    require.add("source_spread")
            sizes = (22, 30) if regime == 2 and split == "test" else ((10, 21) if regime == 2 else (10, 30))
            distributed = bool(split == "test" and regime == 1 or rng.random() < .23)
            if regime == 1 and split == "train":
                distributed = False
            for _ in range(20):
                pair = _make_pair(rng, n, sizes, distributed, require, background_paint)
                key = _signature(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), Counter()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for a in (inp, out):
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            assert inp.shape == out.shape
            key = _signature(inp)
            assert key not in seen
            seen.add(key)
            p = _parse(inp)
            assert np.array_equal(solve(inp.copy()), out), "wrong stencil completion"
            assert not np.array_equal(inp, out)
            covered = np.zeros(inp.shape, dtype=bool)
            for obj in p["objects"]:
                r0, c0, r1, c1 = obj["bounds"]
                covered[r0:r1 + 1, c0:c1 + 1] = True
                occupancy = _occupied(obj)
                for orbit in _orbits(obj["bounds"]):
                    assert all((q in occupancy) == bool(set(orbit) & obj["seeds"]) for q in orbit)
                assert all(_interior(q, obj["bounds"]) for q in obj["seeds"])
            assert np.array_equal(inp[~covered], out[~covered]), "uncovered cells changed"
            r0, c0, r1, c1 = p["table"]
            assert np.array_equal(inp[r0:r1 + 1, c0:c1 + 1], out[r0:r1 + 1, c0:c1 + 1])
            if split == "train":
                training.update(_features(inp, p, out))
    for concept in ("two_folds", "empty_orbits", "corner_retention", "corner_occlusion", "recolor",
                    "priority", "opaque_blank", "needs_mirrors", "needs_mapping", "needs_opacity"):
        assert training[concept] > 0, f"missing training witness: {concept}"


def self_test():
    """Bounded 42-seed sweep, teaching checks, counts, mutations and equivariance."""
    shapes, layouts, palettes, backgrounds, parities, diagonals, counts = set(), set(), set(), set(), set(), set(), set()
    statistics = Counter()
    for seed in range(42):
        nt, ns = 2 + seed % 9, 1 + seed % min(4, 1 + seed % 9)
        puzzle = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                p = _parse(inp)
                shapes.add(inp.shape)
                backgrounds.add(p["background"])
                palettes.add(tuple((o["cue"], o["paint"]) for o in p["objects"]))
                counts.add(len(p["objects"]))
                layouts.add(tuple((o["bounds"], tuple(sorted(o["seeds"]))) for o in p["objects"]))
                statistics.update(_features(inp, p, out))
                for obj in p["objects"]:
                    b = obj["bounds"]
                    parities.add(((b[0] + b[2]) % 2, (b[1] + b[3]) % 2))
                    diagonals.add(obj["diagonal"])
                    statistics["edge_box"] += int(b[0] == 0 or b[1] == 0 or b[2] == inp.shape[0] - 1 or b[3] == inp.shape[1] - 1)
                # All literal color values are interchangeable semantic roles.
                perm = np.roll(np.arange(10), seed % 10)
                assert np.array_equal(solve(perm[inp]), perm[out]), "palette equivariance failed"
    assert len(shapes) >= 60 and len(layouts) >= 200 and len(palettes) >= 200
    assert backgrounds == set(range(10)) and parities == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert diagonals == {0, 1} and counts == {1, 2, 3, 4}
    assert statistics["edge_box"] > 10 and statistics["two_folds"] > 42
    for feature in ("disjoint_scene", "nested_scene", "partial_overlap", "shared_paint", "mixed_diagonals",
                    "interior_table", "background_paint", "self_paint", "other_cue_paint", "square_box",
                    "horizontal_box", "vertical_box", "redundant_orbit", "axis_seed", "source_spread"):
        assert statistics[feature] > 0, f"unobserved variation: {feature}"
    first = generate_puzzle(0)
    second = generate_puzzle(0)
    assert np.array_equal(solve(first["train"][0]["input"].astype(np.uint64)), first["train"][0]["output"])
    assert all(_signature(a[k]) == _signature(b[k]) for s in first
               for a, b in zip(first[s], second[s]) for k in a)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9000 + nt * 10 + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for location in ("table", "motif"):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in first[s]] for s in first}
        pair = bad["test"][0]
        p = _parse(pair["input"])
        point = p["table"][:2] if location == "table" else next(iter(_occupied(p["objects"][0])))
        pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupted output")
    return {"seeds": 42, "shapes": len(shapes), "layouts": len(layouts),
            "mappings": len(palettes), "axis_parities": len(parities),
            "layer_counts": sorted(counts), "witnesses": dict(statistics)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
