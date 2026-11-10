"""Crossed-guide docking: a spatially typed, simultaneous relational join.

Only standard library and NumPy are used. No files are read or written by this
module. Construction records are discarded; solving and validation parse grids.
"""
from __future__ import annotations

import hashlib
import numpy as np

_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape), "grid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "color range")
    return grid.astype(np.int8, copy=False)


def _components(grid, background):
    todo = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _decode(input_grid):
    grid = _array(input_grid)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    _require(counts.max() > grid.size / 2, "background must occupy more than half")
    bg = int(colors[np.argmax(counts)])
    anchors, tokens = {}, []
    for cells in _components(grid, bg):
        p = min(cells)
        if len(cells) == 1 and (p[0] in (0, h - 1) or p[1] in (0, w - 1)):
            _require(not (p[0] in (0, h - 1) and p[1] in (0, w - 1)), "corner anchor")
            anchors.setdefault(int(grid[p]), set()).add(p)
            continue
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        _require(3 <= r1 - r0 + 1 <= 7 and 3 <= c1 - c0 + 1 <= 7,
                 "token bounding box must span 3, 5 or 7 on each axis")
        _require((r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "odd token spans")
        center = (r0 + r1) // 2, (c0 + c1) // 2
        _require(center in cells, "bounding-box center must be occupied")
        bodies = {int(grid[q]) for q in cells if q != center}
        _require(len(bodies) == 1, "uniform body except geometric center")
        tokens.append({"cells": cells, "center": center, "body": bodies.pop(),
                       "value": int(grid[center]), "span": (r1-r0+1, c1-c0+1)})
    _require(anchors and tokens, "anchors and tokens required")
    guides = {}
    for color, cells in anchors.items():
        rows, cols = {r for r, c in cells}, {c for r, c in cells}
        horizontal = len(rows) == 1 and all(c in (0, w-1) for r, c in cells)
        vertical = len(cols) == 1 and all(r in (0, h-1) for r, c in cells)
        _require(horizontal != vertical, "one unambiguous guide per anchor color")
        axis = 0 if horizontal else 1
        guides[color] = (axis, next(iter(rows if horizontal else cols)))
    mapping = {}
    for t in tokens:
        b, v = t["body"], t["value"]
        _require(b not in mapping or mapping[b] == v, "conflicting repeated donors")
        mapping[b] = v
    for i, t in enumerate(tokens):
        donors = [j for j, q in enumerate(tokens) if j != i and q["body"] == t["value"]]
        t["donors"] = donors
        z = tokens[donors[0]]["value"] if donors else None
        t["resolved"] = z
        if z is None:
            reason = "missing_donor"
        elif t["body"] not in guides:
            reason = "missing_body_guide"
        elif z not in guides:
            reason = "missing_result_guide"
        elif guides[t["body"]][0] == guides[z][0]:
            reason = "parallel"
        else:
            reason = "retained"
        t["reason"] = reason
        t["target"] = None
        if reason == "retained":
            target = [0, 0]
            for color in (t["body"], z):
                axis, coordinate = guides[color]
                target[axis] = coordinate
            t["target"] = tuple(target)
    return grid, bg, anchors, guides, tokens


def _paint(grid, bg, anchors, tokens, *, variant=None):
    result = np.full_like(grid, bg)
    occupied = set()
    for color, cells in anchors.items():
        for p in cells:
            result[p] = color
            occupied.add(p)
    # All reads are completed in _decode before any writes take place.
    guides = {}
    h, w = grid.shape
    for color, cells in anchors.items():
        p = min(cells)
        guides[color] = (0, p[0]) if p[1] in (0, w-1) else (1, p[1])
    for t in tokens:
        z, target = t["resolved"], t["target"]
        if variant == "no_A":
            target = t["center"] if z is not None else None
        elif variant == "no_B":
            z = t["value"]
            target = None
            if t["body"] in guides and z in guides and guides[t["body"]][0] != guides[z][0]:
                q = [0, 0]
                for color in (t["body"], z):
                    axis, coordinate = guides[color]
                    q[axis] = coordinate
                target = tuple(q)
        elif variant == "updated_donors":
            z = tokens[t["donors"][0]]["resolved"] if t["donors"] else None
            target = None
            if z is not None and t["body"] in guides and z in guides and guides[t["body"]][0] != guides[z][0]:
                q = [0, 0]
                for color in (t["body"], z):
                    axis, coordinate = guides[color]
                    q[axis] = coordinate
                target = tuple(q)
        if target is None:
            continue
        dr, dc = target[0] - t["center"][0], target[1] - t["center"][1]
        for r, c in t["cells"]:
            q = r + dr, c + dc
            if 0 <= q[0] < h and 0 <= q[1] < w:
                if variant is None:
                    _require(q not in occupied, "destination footprints overlap")
                occupied.add(q)
                result[q] = z if (r, c) == t["center"] else t["body"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the simultaneous result from this input alone."""
    grid, bg, anchors, guides, tokens = _decode(input_grid)
    return _paint(grid, bg, anchors, tokens)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _features(input_grid):
    grid, bg, anchors, guides, tokens = _decode(input_grid)
    retained = [t for t in tokens if t["target"] is not None]
    h, w = grid.shape
    clipped = 0
    clip_sides = set()
    for t in retained:
        dr, dc = t["target"][0]-t["center"][0], t["target"][1]-t["center"][1]
        outside = [(r+dr, c+dc) for r, c in t["cells"]
                   if not (0 <= r+dr < h and 0 <= c+dc < w)]
        clipped += bool(outside)
        for r, c in outside:
            if r < 0:
                clip_sides.add("top")
            if r >= h:
                clip_sides.add("bottom")
            if c < 0:
                clip_sides.add("left")
            if c >= w:
                clip_sides.add("right")
    return {"shape": grid.shape, "background": bg, "tokens": len(tokens),
            "guides": len(guides), "survivors": len(retained), "clipped": clipped,
            "clip_sides": tuple(sorted(clip_sides)),
            "sizes": tuple(sorted(len(t["cells"]) for t in tokens)),
            "spans": tuple(sorted(set(t["span"] for t in tokens))),
            "axes": tuple(sorted({guides[t["body"]][0] for t in retained})),
            "bilateral": sum(len(cells) == 2 for cells in anchors.values()),
            "repeated": len(tokens) - len({t["body"] for t in tokens}),
            "monochrome": sum(t["body"] == t["value"] for t in tokens),
            "unchanged_center": sum(t["resolved"] == t["value"] for t in retained),
            "shared_donor": any(sum(j in t["donors"] for t in retained) >= 2 for j in range(len(tokens))),
            "moving_donor": any(any(tokens[j]["target"] is not None and
                                      tokens[j]["resolved"] != tokens[j]["value"]
                                      for j in t["donors"]) for t in retained),
            "two_axis_motion": sum(all(t["target"][k] != t["center"][k] for k in (0, 1)) for t in retained),
            "input_edge": sum(any(r in (0,h-1) or c in (0,w-1) for r,c in t["cells"]) for t in tokens),
            "reasons": tuple(sorted(t["reason"] for t in tokens)),
            "layout": tuple(sorted((t["center"], t["span"], tuple(sorted(t["cells"]))) for t in tokens))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, train_features, test_features = set(), [], []
    training_ablations = {name: False for name in ("no_A", "no_B", "updated_donors")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(inp.shape == out.shape, "canvas preservation")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            decoded = _decode(inp)
            _require(np.array_equal(_paint(decoded[0], decoded[1], decoded[2], decoded[4]), out),
                     "wrong relational crossing, rigid movement, clipping or anchor preservation")
            f = _features(inp)
            _require(1 <= f["survivors"] < f["tokens"], "must retain and discard tokens")
            _require(not np.array_equal(inp, out), "unchanged pair")
            if split == "train":
                train_features.append(f)
                for name in training_ablations:
                    alternative = _paint(decoded[0], decoded[1], decoded[2], decoded[4], variant=name)
                    training_ablations[name] |= not np.array_equal(alternative, out)
            else:
                test_features.append(f)
    _require(all(training_ablations.values()), "training does not distinguish inherited mechanisms or simultaneous reads")
    _require(any(f["moving_donor"] and f["two_axis_motion"] and len(f["axes"]) == 2 for f in train_features),
             "need a simultaneous, moving-donor witness in both orientations")
    _require(any(f["shared_donor"] and f["unchanged_center"] for f in train_features), "shared fixed-point witness")
    for reason in ("missing_donor", "missing_body_guide", "parallel"):
        _require(any(reason in f["reasons"] for f in train_features), "missing rejection demonstration: " + reason)
    if any(f["clipped"] for f in test_features):
        _require(any(f["clipped"] for f in train_features), "test clipping lacks a demonstration")


def _support(rng, regime):
    """Odd bounding-box centers are visible even for monochrome/asymmetric tokens."""
    kinds = ("square", "cross", "rectangle", "diamond", "spines", "grown")
    kind = kinds[int(rng.integers(len(kinds)))]
    radius = 1 if regime == "small" else int(rng.choice([1, 2], p=[.6, .4]))
    if regime == "large":
        radius = int(rng.choice([1, 2, 3], p=[.2, .65, .15]))
    if kind == "square":
        cells = {(r, c) for r in range(-radius, radius+1) for c in range(-radius, radius+1)}
    elif kind == "cross":
        cells = {(r, 0) for r in range(-radius, radius+1)} | {(0, c) for c in range(-radius, radius+1)}
    elif kind == "rectangle":
        a, b = radius, int(rng.integers(1, radius+1))
        if rng.random() < .5:
            a, b = b, a
        cells = {(r, c) for r in range(-a,a+1) for c in range(-b,b+1)}
    elif kind == "diamond":
        cells = {(r,c) for r in range(-radius,radius+1) for c in range(-radius,radius+1)
                 if abs(r)+abs(c) <= radius}
    else:
        a, b = radius, int(rng.integers(1, radius+1))
        cells = {(r,0) for r in range(-a,a+1)} | {(0,c) for c in range(-b,b+1)}
        # Random branches preserve a centered odd bounding box, not symmetry.
        goal = int(rng.integers(len(cells), (2*a+1)*(2*b+1)+1))
        if kind == "spines":
            goal = min(goal, len(cells) + int(rng.integers(1, 4)))
        while len(cells) < goal:
            frontier = sorted({(r+dr,c+dc) for r,c in cells for dr,dc in _D4
                               if -a <= r+dr <= a and -b <= c+dc <= b} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    return tuple(sorted(cells))


def _relations(rng, mode):
    """Numbers are temporary identity roles, recolored independently of geometry."""
    if mode == "chain":
        # A->B->C->D, E lacks a guide, F and C have parallel guides.
        rel = [(0,1), (1,2), (2,3), (4,1), (5,1)]
        axes = {0:0, 1:1, 2:1, 3:0, 5:1}
        if rng.random() < .45:
            rel.append((2,3))  # consistent repeated donors
        return rel, axes
    if mode == "shared":
        rel = [(0,1), (2,1), (1,1), (3,1)]
        axes = {0:0, 2:0, 1:1}
        if rng.random() < .45:
            rel.append((1,1))
        if rng.random() < .4:
            rel.append((4,5))
            axes[4] = 0  # selected-looking query without a donor
        return rel, axes
    for _ in range(96):
        k = int(rng.integers(4, 9))
        n = int(rng.integers(3, min(k, 7)+1))
        bodies = list(map(int, rng.choice(k, n, replace=False)))
        mapping = {body:int(rng.integers(k)) for body in bodies}
        axes = {i:int(rng.integers(2)) for i in range(k) if rng.random() < .82}
        rel = [(body, mapping[body]) for body in bodies]
        if rng.random() < .35 and len(rel) < 8:
            rel.append(rel[int(rng.integers(len(rel)))])
        # Repeated eligible bodies would occupy the same destination; only
        # nonretained duplicate donors are allowed in this collision-free family.
        retained = []
        for i, (body, value) in enumerate(rel):
            donors = [v for j,(b,v) in enumerate(rel) if i != j and b == value]
            z = donors[0] if donors else None
            retained.append(z is not None and body in axes and z in axes and axes[body] != axes[z])
        successful_bodies = [body for (body,v),ok in zip(rel,retained) if ok]
        low, high = (3, 4) if mode == "transfer" else (1, 2)
        if (low <= sum(retained) <= high and sum(retained) < len(rel)
                and len(successful_bodies) == len(set(successful_bodies))
                and 2 <= len(axes) <= 7):
            return rel, axes
    # A finite, semantically identical fallback for an unusually unlucky graph draw.
    if mode == "transfer":
        return [(0,1),(1,2),(2,3),(3,1),(4,1)], {0:0,1:1,2:1,3:0}
    return _relations(rng, "chain")


def _construction_targets(rel, axes, coordinates):
    result = []
    for i, (body, value) in enumerate(rel):
        donors = [v for j, (b,v) in enumerate(rel) if i != j and b == value]
        z = donors[0] if donors else None
        target = None
        if z is not None and body in axes and z in axes and axes[body] != axes[z]:
            point = [0, 0]
            point[axes[body]] = coordinates[body]
            point[axes[z]] = coordinates[z]
            target = tuple(point)
        result.append((z, target))
    return result


def _layout(rng, rel, axes, supports, palette, shift, force_motion=False, clip_witness=False):
    # Each attempt allocates two <=900-byte grids. Attempts never write to disk.
    for attempt in range(48):
        min_h = max(12, 2 + 3*sum(v == 0 for v in axes.values()))
        min_w = max(12, 2 + 3*sum(v == 1 for v in axes.values()))
        area_need = sum(len(s) for s in supports) * 3 + 40
        h = int(rng.integers(min(29,min_h), 31))
        w = int(rng.integers(min(29,min_w), 31))
        if h*w < area_need:
            h = w = 30
        if shift and rng.random() < .3:
            h = w = 30
        coordinates = {}
        # Separating every same-axis guide by 3 is sufficient for small supports.
        # Actual destination footprints, including larger shapes, are checked below.
        for axis, limit in ((0,h),(1,w)):
            used = []
            for color in sorted(i for i,v in axes.items() if v == axis):
                choices = [x for x in range(1,limit-1) if all(abs(x-y) >= 3 for y in used)]
                if not choices:
                    break
                near = [x for x in choices if x in (1,limit-2)]
                if clip_witness and color == 0 and near:
                    choices = near
                x = choices[int(rng.integers(len(choices)))]
                coordinates[color] = x
                used.append(x)
        if len(coordinates) != len(axes):
            continue
        bg = palette[0]
        inp, out = np.full((h,w),bg,np.int8), np.full((h,w),bg,np.int8)
        anchor_cells, blocked_target = set(), set()
        for body, axis in axes.items():
            coordinate = coordinates[body]
            ends = (0,1) if rng.random() < .3 else (int(rng.integers(2)),)
            for end in ends:
                p = (coordinate, 0 if end == 0 else w-1) if axis == 0 else (0 if end == 0 else h-1, coordinate)
                anchor_cells.add(p)
                inp[p] = out[p] = palette[body+1]
                blocked_target.add(p)
        targets = _construction_targets(rel, axes, coordinates)
        destinations = []
        valid = True
        for (body, value), support, (z, target) in zip(rel,supports,targets):
            cells = set() if target is None else {(target[0]+r,target[1]+c) for r,c in support
                                                  if 0 <= target[0]+r < h and 0 <= target[1]+c < w}
            if cells & blocked_target:
                valid = False
                break
            blocked_target.update(cells)
            destinations.append(cells)
        if not valid:
            continue
        occupied = set(anchor_cells)
        centers = [None]*len(rel)
        # Large tokens first; source placement is independent of relation direction.
        for i in sorted(range(len(rel)), key=lambda j:-len(supports[j])):
            support = supports[i]
            a = max(abs(r) for r,c in support)
            b = max(abs(c) for r,c in support)
            candidates = []
            for r in range(a,h-a):
                for c in range(b,w-b):
                    if force_motion and targets[i][1] is not None and (r == targets[i][1][0] or c == targets[i][1][1]):
                        continue
                    cells = {(r+dr,c+dc) for dr,dc in support}
                    if any(p in occupied or any((p[0]+dr,p[1]+dc) in occupied for dr,dc in _D4) for p in cells):
                        continue
                    candidates.append((r,c))
            if not candidates:
                valid = False
                break
            center = candidates[int(rng.integers(len(candidates)))]
            centers[i] = center
            occupied.update((center[0]+r,center[1]+c) for r,c in support)
        if not valid or len(occupied)*2 >= h*w:
            continue
        for i, ((body,value),support,(z,target)) in enumerate(zip(rel,supports,targets)):
            r,c = centers[i]
            for dr,dc in support:
                inp[r+dr,c+dc] = palette[(value if (dr,dc) == (0,0) else body)+1]
                if target is not None:
                    tr,tc = target[0]+dr,target[1]+dc
                    if 0 <= tr < h and 0 <= tc < w:
                        out[tr,tc] = palette[(z if (dr,dc) == (0,0) else body)+1]
        return {"input":inp,"output":out}
    return None


def _make_pair(rng, mode, parameters, test):
    for attempt in range(96):
        rel, axes = _relations(rng, mode)
        # Exchange the two axes freely; neither body identity nor lookup depth
        # has a fixed association with horizontal or vertical geometry.
        if rng.random() < .5:
            axes = {c:1-axis for c,axis in axes.items()}
        if parameters["stable_palette"]:
            palette = parameters["palette"]
        else:
            palette = tuple(map(int,rng.permutation(10)))
        regime = ("large" if test and parameters["scale_shift"] else parameters["morphology"])
        supports = [_support(rng,regime) for _ in rel]
        clip_witness = mode == "chain" and parameters["teach_clipping"]
        if clip_witness:
            supports[0] = tuple((r,c) for r in range(-2,3) for c in range(-2,3))
        if test and parameters["scale_shift"]:
            supports[0] = tuple(sorted({(r,0) for r in range(-3,4)} | {(0,c) for c in range(-3,4)}))
        pair = _layout(rng,rel,axes,supports,palette,test,
                       force_motion=mode == "chain",clip_witness=clip_witness)
        if pair is None:
            continue
        # Reflection changes no guide-axis roles and is sampled independently.
        if rng.random() < .5:
            pair = {k:np.fliplr(v).copy() for k,v in pair.items()}
        if rng.random() < .5:
            pair = {k:np.flipud(v).copy() for k,v in pair.items()}
        try:
            _require(np.array_equal(solve(pair["input"]),pair["output"]), "construction/parse mismatch")
            f = _features(pair["input"])
            if not parameters["teach_clipping"]:
                _require(f["clipped"] == 0, "clipping reserved for demonstrated regimes")
            if clip_witness:
                _require(f["clipped"] > 0, "clipping witness missing")
            if mode == "chain":
                _require(f["moving_donor"] and len(f["axes"]) == 2 and f["two_axis_motion"] >= 2,
                         "chain witness missing")
            if mode == "shared":
                _require(f["shared_donor"] and f["unchanged_center"], "shared witness missing")
        except AssertionError:
            continue
        return pair
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name,value,low,high in (("num_train",num_train,2,10),("num_test",num_test,1,4)):
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
                or not low <= int(value) <= high):
            raise ValueError(name + " outside legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    parameters = {"stable_palette":bool(rng.random() < .3),
                  "palette":tuple(map(int,rng.permutation(10))),
                  "morphology":str(rng.choice(["small","mixed"],p=[.35,.65])),
                  "teach_clipping":bool(rng.random() < .65),
                  "scale_shift":bool(rng.random() < .55)}
    modes = ["chain","shared"] + ["random"]*(int(num_train)-2)
    rng.shuffle(modes)
    puzzle,seen = {"train":[],"test":[]},set()
    for split,count in (("train",int(num_train)),("test",int(num_test))):
        for i in range(count):
            for retry in range(16):
                pair = _make_pair(rng,modes[i] if split == "train" else "transfer",parameters,split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Stream 42 seeds; no sample files, trace payloads or unbounded caches."""
    fingerprints, dimensions, layouts, sizes, spans = set(),set(),set(),set(),set()
    rows = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train","test"):
            for pair in puzzle[split]:
                for grid in pair.values():
                    digest.update(bytes(grid.shape))
                    digest.update(grid.tobytes())
                f = _features(pair["input"])
                rows.append({k:v for k,v in f.items() if k != "layout"})
                dimensions.add(f["shape"])
                layouts.add(f["layout"])
                sizes.update(f["sizes"])
                spans.update(f["spans"])
        fingerprints.add(digest.hexdigest())
        if seed == 0:
            original = puzzle
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(dimensions) >= 60 and len(layouts) >= 190, "geometric diversity")
    _require(len(sizes) >= 10 and len(spans) >= 6, "support diversity")
    _require(len({f["tokens"] for f in rows}) >= 4, "token count diversity")
    _require(len({f["survivors"] for f in rows}) >= 3, "survivor diversity")
    _require(len({f["guides"] for f in rows}) >= 4, "guide count diversity")
    _require({f["background"] for f in rows} == set(range(10)), "background roles")
    for feature in ("clipped","input_edge","bilateral","repeated","monochrome","unchanged_center","shared_donor","moving_donor"):
        _require(sum(bool(f[feature]) for f in rows) >= 5, "coverage: " + feature)
    _require(set().union(*(set(f["clip_sides"]) for f in rows)) == {"top","bottom","left","right"}, "clipping sides")
    # Rotation, reflection and arbitrary palette permutations preserve the rule.
    rng = np.random.default_rng(1937)
    for turns in (1,2,3):
        transformed = {"train":[],"test":[]}
        for split in transformed:
            for pair in original[split]:
                colors = rng.permutation(10)
                transformed[split].append({k:colors[np.rot90(v,turns)].astype(np.int8) for k,v in pair.items()})
        validate_generated_puzzle(transformed)
    repeated = generate_puzzle(0)
    _require(all(np.array_equal(p[k],q[k]) for s in original for p,q in zip(original[s],repeated[s])
                 for k in ("input","output")), "determinism")
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(137+a*10+b,a,b)
            _require(len(p["train"]) == a and len(p["test"]) == b,"explicit counts")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Negative tests operate on returned grids; labels and anchor cells matter.
    bad = {s:[{k:v.copy() for k,v in p.items()} for p in original[s]] for s in original}
    cell = min(next(iter(_decode(bad["test"][0]["input"])[2].values())))
    bad["test"][0]["output"][cell] = (int(bad["test"][0]["output"][cell])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("anchor corruption accepted")
