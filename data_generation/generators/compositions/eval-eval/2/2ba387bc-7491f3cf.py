"""Color ferry: topology-filtered shape queues with consumptive color transfer.

Only Python's standard library and NumPy are used. No parent data are loaded.
"""
from __future__ import annotations

import copy
import json
from collections import Counter

import numpy as np


_KINDS = ("hollow", "solid")
_CUTS = ("vertical", "horizontal", "diagonal", "elbow", "stepped")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask)))
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mask(shape, cells):
    a = np.zeros(shape, dtype=bool)
    for p in cells:
        a[p] = True
    return a


def _regions(cut):
    """Check a simple digital path; label the two four-connected shores."""
    cells = set(map(tuple, np.argwhere(cut)))
    _require(len(cells) >= 3, "cut too short")
    graph = {}
    for r, c in cells:
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r + dr, c + dc
                if (dr or dc) and p in cells:
                    # Orthogonal elbows have no diagonal shortcut edge.
                    if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                        continue
                    neighbors.append(p)
        graph[r, c] = neighbors
    ends = [p for p in cells if len(graph[p]) == 1]
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()),
             "cut must be a path, without branches or loops")
    h, w = cut.shape
    boundary = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require({p for p in cells if boundary(p)} == set(ends),
             "only the two cut endpoints may touch the boundary")
    visited, stack = set(), [ends[0]]
    while stack:
        p = stack.pop()
        if p not in visited:
            visited.add(p)
            stack.extend(graph[p])
    _require(visited == cells, "disconnected cut")
    regions = _components(~cut)
    _require(len(regions) == 2, "cut must leave exactly two shores")
    return [_mask(cut.shape, r) for r in regions]


def _decode(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(h >= 7 and w >= 7, "canvas too small for this family")
    border = np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))
    bg = int(np.argmax(np.bincount(border, minlength=10)))
    _require(np.count_nonzero(border != bg) == 2, "boundary must be background except cut endpoints")
    ctrl = set(map(int, border)) - {bg}
    _require(len(ctrl) == 1, "endpoints must have one control color")
    ctrl = ctrl.pop()
    control_components = sorted(_components(a == ctrl, True), key=len)
    _require(len(control_components) == 2 and len(control_components[0]) == 1,
             "control needs one isolated dot and one cut")
    marker = control_components[0][0]
    cut = _mask(a.shape, control_components[1])
    regions = _regions(cut)
    side = next((i for i, region in enumerate(regions) if region[marker]), None)
    _require(side is not None, "dot lies on cut")
    r, c = marker
    _require(not np.any(cut[max(0, r-1):r+2, max(0, c-1):c+2]), "dot touches cut")
    objects, sizes = [], set()
    for cells in _components((a != bg) & (a != ctrl)):
        rs, cs = zip(*cells)
        top, left = min(rs), min(cs)
        s = max(rs) - top + 1
        _require(s == max(cs) - left + 1 and 3 <= s <= 6, "non-square or illegal tile scale")
        _require(1 <= top and 1 <= left and top+s < h and left+s < w, "tile touches canvas edge")
        tile = a[top:top+s, left:left+s]
        color = int(tile[0, 0])
        _require(color not in (bg, ctrl), "unreadable tile color")
        rim = np.ones((s, s), dtype=bool)
        rim[1:-1, 1:-1] = False
        _require(np.all(tile[rim] == color), "broken tile rim")
        if np.all(tile[1:-1, 1:-1] == bg):
            kind = "hollow"
        else:
            _require(np.all(tile == color), "tile must be hollow or solid")
            kind = "solid"
        halo = a[top-1:top+s+1, left-1:left+s+1].copy()
        halo[1:-1, 1:-1] = bg
        _require(np.all(halo == bg), "tiles need a background gap from objects and controls")
        shore = next((i for i, region in enumerate(regions)
                      if np.all(region[top:top+s, left:left+s])), None)
        _require(shore is not None, "tile straddles cut")
        objects.append(dict(top=top, left=left, size=s, color=color, kind=kind, shore=shore))
        sizes.add(s)
    _require(len(sizes) == 1, "one common tile size required")
    for i in (0, 1):
        for kind in _KINDS:
            tops = [o["top"] for o in objects if o["shore"] == i and o["kind"] == kind]
            _require(len(tops) == len(set(tops)), "same-shore same-shape top-row tie")
    receivers = sorted([o for o in objects if o["shore"] == side and o["kind"] == "hollow"],
                       key=lambda o: o["top"])
    donors = sorted([o for o in objects if o["shore"] != side and o["kind"] == "solid"],
                    key=lambda o: o["top"])
    _require(receivers and donors, "at least one active frame and opposite donor required")
    return dict(grid=a, bg=bg, ctrl=ctrl, cut=cut, marker=marker, regions=regions,
                side=side, objects=objects, size=sizes.pop(), receivers=receivers, donors=donors)


def _apply(d, receivers=None, donors=None):
    out = d["grid"].copy()
    s, bg = d["size"], d["bg"]
    receivers = d["receivers"] if receivers is None else receivers
    donors = d["donors"] if donors is None else donors
    for frame, donor in zip(receivers, donors):
        r, c = frame["top"], frame["left"]
        out[r+1:r+s-1, c+1:c+s-1] = donor["color"]
        r, c = donor["top"], donor["left"]
        out[r:r+s, c:c+s] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Parse only visible input pixels; no examples, metadata, RNG or seed."""
    return _apply(_decode(input_grid))


def _alternative(d, mode):
    """Authored shortcut checks, not an independent semantic oracle."""
    hs, ss = list(d["receivers"]), list(d["donors"])
    if mode == "global":
        hs = sorted([o for o in d["objects"] if o["kind"] == "hollow"], key=lambda o: (o["top"], o["left"]))
        ss = sorted([o for o in d["objects"] if o["kind"] == "solid"], key=lambda o: (o["top"], o["left"]))
    elif mode in ("horizontal", "reverse_horizontal", "reverse_vertical"):
        if mode == "reverse_vertical":
            hs.reverse()
            ss.reverse()
        else:
            reverse = mode == "reverse_horizontal"
            hs.sort(key=lambda o: (o["left"], o["top"]), reverse=reverse)
            ss.sort(key=lambda o: (o["left"], o["top"]), reverse=reverse)
    elif mode == "nearest":
        available, ss = ss, []
        for frame in hs[:len(available)]:
            donor = min(available, key=lambda o: (
                (o["top"]-frame["top"])**2 + (o["left"]-frame["left"])**2,
                o["top"], o["left"]))
            ss.append(donor)
            available.remove(donor)
    elif mode == "greedy":
        rows, slot = [], [None, None]
        objects = sorted([(o, 0) for o in hs] + [(o, 1) for o in ss],
                         key=lambda x: (x[0]["top"], x[0]["left"]))
        for obj, col in objects:
            if slot[col] is not None:
                rows.append(slot)
                slot = [None, None]
            slot[col] = obj
        rows.append(slot)
        hs, ss = [], []
        for frame, donor in rows:
            if frame is not None and donor is not None:
                hs.append(frame)
                ss.append(donor)
    else:
        raise ValueError("unknown alternative")
    return _apply(d, hs, ss)


_SHORTCUTS = ("global", "horizontal", "reverse_horizontal", "reverse_vertical", "nearest", "greedy")


def _cut_family(cut):
    rs, cs = np.nonzero(cut)
    if len(set(cs)) == 1:
        return "vertical"
    if len(set(rs)) == 1:
        return "horizontal"
    # A diagonal has no orthogonal runs longer than one on both axes.
    if np.all(cut.sum(axis=1) == 1) or np.all(cut.sum(axis=0) == 1):
        return "diagonal"
    cells = set(map(tuple, np.argwhere(cut)))
    horizontal_rows = {r for r, c in cells if (r, c+1) in cells}
    vertical_cols = {c for r, c in cells if (r+1, c) in cells}
    return "elbow" if len(horizontal_rows) == len(vertical_cols) == 1 else "stepped"


def _sample_cut(rng, shape, family):
    h, w = shape
    for _ in range(32):
        cut = np.zeros(shape, dtype=bool)
        if family == "vertical":
            cut[:, int(rng.integers(w//3, 2*w//3+1))] = True
        elif family == "horizontal":
            cut[int(rng.integers(h//3, 2*h//3+1)), :] = True
        elif family == "diagonal":
            transpose = bool(rng.integers(2))
            ch, cw = (w, h) if transpose else (h, w)
            p = int(rng.integers(max(1, cw//4), cw//2))
            q = int(rng.integers(cw//2+1, min(cw-1, 3*cw//4+1)))
            if abs(p-q) > ch-1:
                continue
            trial = np.zeros((ch, cw), dtype=bool)
            trial[np.arange(ch), np.rint(np.linspace(p, q, ch)).astype(int)] = True
            cut = trial.T.copy() if transpose else trial
        elif family == "elbow":
            r = int(rng.integers(h//3, 2*h//3+1))
            c = int(rng.integers(w//3, 2*w//3+1))
            cut[:r+1, c] = True
            cut[r, c:] = True
        else:
            transpose = bool(rng.integers(2))
            ch, cw = (w, h) if transpose else (h, w)
            trial = np.zeros((ch, cw), dtype=bool)
            row_options = np.arange(4, ch-4, 4)
            bend_count = int(rng.integers(1, min(3, len(row_options))+1))
            rows = sorted(map(int, rng.choice(row_options, bend_count, replace=False)))
            col, last = int(rng.integers(cw//3, 2*cw//3+1)), 0
            for row in rows:
                trial[last:row+1, col] = True
                choices = [c for c in range(cw//3, 2*cw//3+1) if abs(c-col) >= 2]
                new_col = int(rng.choice(choices))
                trial[row, min(col, new_col):max(col, new_col)+1] = True
                col, last = new_col, row
            trial[last:, col] = True
            cut = trial.T.copy() if transpose else trial
        if rng.integers(2):
            cut = cut[::-1].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        try:
            _regions(cut)
            return cut
        except AssertionError:
            continue
    raise RuntimeError("could not sample a simple cut")


def _sites(region, size):
    h, w = region.shape
    # Summed-area rectangles make the full background halo constraint cheap.
    integral = np.pad(region.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for r in range(1, h-size):
        for c in range(1, w-size):
            area = (integral[r+size+1, c+size+1] - integral[r-1, c+size+1]
                    - integral[r+size+1, c-1] + integral[r-1, c-1])
            if area == (size+2)**2:
                result.append((r, c))
    return result


def _place(rng, regions, size, counts):
    sites = [_sites(region, size) for region in regions]
    if any(len(sites[i]) < sum(counts[i]) for i in (0, 1)):
        return None
    for _ in range(20):
        objects = []
        for shore in rng.permutation(2):
            kinds = list(rng.permutation(["hollow"]*counts[shore][0] + ["solid"]*counts[shore][1]))
            for kind in kinds:
                valid = [(r, c) for r, c in sites[shore] if all(
                    (abs(r-o["top"]) > size or abs(c-o["left"]) > size)
                    and not (o["shore"] == shore and o["kind"] == kind and r == o["top"])
                    for o in objects)]
                if not valid:
                    break
                r, c = valid[int(rng.integers(len(valid)))]
                objects.append(dict(top=r, left=c, kind=str(kind), shore=int(shore), size=size))
            else:
                continue
            break
        else:
            return objects
    return None


def _paint_objects(rng, objects, bg, ctrl, witness):
    palette = [int(c) for c in rng.permutation(10) if c not in (bg, ctrl)]
    for shore in (0, 1):
        solids = sorted([o for o in objects if o["shore"] == shore and o["kind"] == "solid"],
                        key=lambda o: o["top"])
        colors = list(rng.permutation(palette))
        for i, obj in enumerate(solids):
            obj["color"] = int(colors[i % len(colors)] if witness or rng.random() < .7 else rng.choice(palette))
    for shore in (0, 1):
        frames = sorted([o for o in objects if o["shore"] == shore and o["kind"] == "hollow"],
                        key=lambda o: o["top"])
        donors = sorted([o for o in objects if o["shore"] != shore and o["kind"] == "solid"],
                        key=lambda o: o["top"])
        for i, obj in enumerate(frames):
            choices = [c for c in palette if not (witness and i < len(donors) and c == donors[i]["color"])]
            obj["color"] = int(rng.choice(choices))
    if witness:
        # Same color can denote either geometry on the same shore, without
        # hiding any donor-to-recipient contrast on the opposite shore.
        for shore in (0, 1):
            frames = sorted([o for o in objects if o["shore"] == shore and o["kind"] == "hollow"], key=lambda o: o["top"])
            solids = sorted([o for o in objects if o["shore"] == shore and o["kind"] == "solid"], key=lambda o: o["top"])
            donors = sorted([o for o in objects if o["shore"] != shore and o["kind"] == "solid"], key=lambda o: o["top"])
            choices = [o["color"] for o in solids if o["color"] != donors[0]["color"]]
            frames[0]["color"] = choices[0]


def _scene(rng, cut, objects, bg, ctrl):
    grid = np.full(cut.shape, bg, dtype=np.int8)
    grid[cut] = ctrl
    for o in objects:
        r, c, s = o["top"], o["left"], o["size"]
        grid[r:r+s, c:c+s] = o["color"]
        if o["kind"] == "hollow":
            grid[r+1:r+s-1, c+1:c+s-1] = bg
    regions = _regions(cut)
    sites = []
    for region in regions:
        options = []
        for r, c in np.argwhere(region & (grid == bg)):
            r, c = int(r), int(c)
            if any(o["top"] <= r < o["top"]+o["size"] and
                   o["left"] <= c < o["left"]+o["size"] for o in objects):
                continue
            if 0 < r < grid.shape[0]-1 and 0 < c < grid.shape[1]-1 and np.all(grid[r-1:r+2, c-1:c+2] == bg):
                options.append((r, c))
        if not options:
            return None
        sites.append(options[int(rng.integers(len(options)))])
    return grid, sites


def _make_pair(base, marker, ctrl):
    a = base.copy()
    a[marker] = ctrl
    return dict(input=a, output=solve(a))


def _candidate(rng, family, size, counts, bg, ctrl, roomy=False, witness=False):
    # Entire objects, rather than noisy repair pixels, are resampled to obtain
    # readable control regions and adequate packing space.
    for _ in range(24):
        low = max(16, 2*size+7)
        if roomy:
            low = max(low, 25)
        elif size >= 5:
            low = max(low, 23)
        shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        cut = _sample_cut(rng, shape, family)
        regions = _regions(cut)
        objects = _place(rng, regions, size, counts)
        if objects is None:
            continue
        _paint_objects(rng, objects, bg, ctrl, witness)
        scene = _scene(rng, cut, objects, bg, ctrl)
        if scene is not None:
            return scene
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("cut", "scale", "supply")))
    fixed_palette = bool(rng.random() < .3)
    bg, ctrl = map(int, rng.choice(10, 2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    # Controlled opposite-direction witnesses, always supplied even at (2,1).
    # Both shores contain both shapes. Side 0 has a donor shortage; side 1
    # has surplus donors. Their output effects are visible with identical tiles.
    for attempt in range(160):
        s = int(rng.integers(3, 5))
        gap_h = 1 if regime == "supply" else int(rng.integers(1, 3))
        gap_s = 1 if regime == "supply" else int(rng.integers(1, 3))
        counts = ((2+gap_h, 2+gap_s), (2, 2))
        family = str(rng.choice(_CUTS[:3]))
        scene = _candidate(rng, family, s, counts, bg, ctrl, roomy=True, witness=True)
        if scene is None:
            continue
        base, sites = scene
        first_side = int(rng.integers(2))
        twins = [_make_pair(base, sites[i], ctrl) for i in (first_side, 1-first_side)]
        analyses = [_decode(p["input"]) for p in twins]
        if all(any(not np.array_equal(_alternative(d, shortcut), p["output"])
                   for d, p in zip(analyses, twins)) for shortcut in _SHORTCUTS):
            puzzle["train"].extend(twins)
            seen.update(_key(p["input"]) for p in twins)
            break
    else:
        raise RuntimeError("could not construct the controlled queue witnesses")
    for split, count in (("train", nt-2), ("test", nq)):
        for index in range(count):
            for attempt in range(128):
                if not fixed_palette:
                    bg, ctrl = map(int, rng.choice(10, 2, replace=False))
                testing = split == "test"
                if regime == "cut":
                    family = str(rng.choice(_CUTS[3:] if testing else _CUTS[:3]))
                else:
                    family = str(rng.choice(_CUTS))
                size = int(rng.integers(5, 7)) if regime == "scale" and testing else int(rng.integers(3, 5))
                # Outside scale-shift puzzles, occasional larger ordinary tiles.
                if regime == "cut" and rng.random() < .2:
                    size = int(rng.integers(5, 7))
                side = int(rng.integers(2))
                limit = 2 if size >= 5 else 4
                if regime == "supply" and testing:
                    short = int(rng.integers(1, 3))
                    long = min(5, short + int(rng.integers(2, 4)))
                    nh, ns = (short, long) if rng.integers(2) else (long, short)
                elif regime == "supply":
                    nh = int(rng.integers(1, 4))
                    ns = max(1, nh + int(rng.integers(-1, 2)))
                else:
                    nh, ns = map(int, rng.integers(1, limit+1, size=2))
                counts = [[0, 0], [0, 0]]
                counts[side][0], counts[1-side][1] = nh, ns
                counts[side][1] = int(rng.integers(0, min(3, limit+1)))
                counts[1-side][0] = int(rng.integers(0, min(3, limit+1)))
                scene = _candidate(rng, family, size, counts, bg, ctrl)
                if scene is None:
                    continue
                base, sites = scene
                pair = _make_pair(base, sites[side], ctrl)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not sample a distinct compatible scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Structural, cellwise transfer, conservation and teaching checks."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output must retain the canvas")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            # Reconstruct changed coordinates cell by cell, including the exact
            # protected rims and unmatched tails, without calling solve/_apply.
            edits = {}
            s, k = d["size"], min(len(d["receivers"]), len(d["donors"]))
            used_colors = Counter()
            for rank in range(k):
                frame, donor = d["receivers"][rank], d["donors"][rank]
                for r in range(frame["top"]+1, frame["top"]+s-1):
                    for c in range(frame["left"]+1, frame["left"]+s-1):
                        edits[r, c] = donor["color"]
                for r in range(donor["top"], donor["top"]+s):
                    for c in range(donor["left"], donor["left"]+s):
                        edits[r, c] = d["bg"]
                used_colors[donor["color"]] += 1
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    _require(out[r, c] == edits.get((r, c), int(inp[r, c])), "incorrect transfer or changed protected cell")
            delta = np.bincount(out.ravel(), minlength=10) - np.bincount(inp.ravel(), minlength=10)
            expected_delta = np.zeros(10, dtype=int)
            expected_delta[d["bg"]] = k*(4*s-4)
            for color, number in used_colors.items():
                expected_delta[color] -= number*(4*s-4)
            _require(np.array_equal(delta, expected_delta), "donor/material conservation failed")
            _require(np.count_nonzero(inp != out) == k*(s*s+(s-2)**2), "wrong transfer footprint")
            (training if split == "train" else testing).append((d, out))
    _require({d["side"] for d, _ in training} == {0, 1}, "both marker shores must be taught")
    differences = [len(d["receivers"])-len(d["donors"]) for d, _ in training]
    _require(min(differences) < 0 < max(differences), "teach both unmatched-tail directions")
    for shortcut in _SHORTCUTS:
        _require(any(not np.array_equal(_alternative(d, shortcut), out) for d, out in training),
                 f"training admits {shortcut} shortcut")
    roles = {}
    for d, _ in training:
        for obj in d["objects"]:
            roles.setdefault(obj["color"], set()).add(obj["kind"])
    _require(any(v == set(_KINDS) for v in roles.values()), "geometry/color roles not demonstrated")
    _require(all(any(len({o["color"] for o in d["donors"]}) >= 2 for d, _ in training if d["side"] == side)
                 for side in (0, 1)), "donor order must be observable in both directions")
    first, second = training[0][0], training[1][0]
    a, b = first["grid"].copy(), second["grid"].copy()
    a[first["marker"]], b[second["marker"]] = first["bg"], second["bg"]
    _require(first["side"] != second["side"] and np.array_equal(a, b), "first demonstrations must isolate marker effect")
    families = {_cut_family(d["cut"]) for d, _ in training}
    sizes = {d["size"] for d, _ in training}
    largest_imbalance = max(abs(n) for n in differences)
    _require(all(_cut_family(d["cut"]) not in families or d["size"] > max(sizes)
                 or abs(len(d["receivers"])-len(d["donors"])) > largest_imbalance
                 for d, _ in testing), "test shift was not constructed")


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed diversity audit plus targeted metamorphic controls."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, layouts = [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                features.append((split, d["grid"].shape, d["size"], d["bg"], _cut_family(d["cut"]),
                                 len(d["receivers"]), len(d["donors"]), len(d["objects"])))
                layouts.add((d["grid"].shape, d["cut"].tobytes(),
                             tuple(sorted((o["top"], o["left"], o["kind"]) for o in d["objects"]))))
        # Horizontal reflection preserves vertical ranks; top-bottom reflection
        # does not generally preserve a partial queue's matched end.
        p = puzzle["train"][0]
        _require(np.array_equal(solve(p["input"][:, ::-1]), p["output"][:, ::-1]), "horizontal reflection failed")
        # Every bijection of literal colors preserves structural color roles.
        palette = np.random.default_rng(900).permutation(10)
        _require(np.array_equal(solve(palette[p["input"]]), palette[p["output"]]), "palette equivariance failed")
        # Opposite dot controls activate different shape subsets on the same scene.
        _require(not np.array_equal(puzzle["train"][0]["output"], puzzle["train"][1]["output"]), "ineffective marker twins")
    _require(len({_signature(p) for p in puzzles}) == 42, "collapsed puzzles")
    _require(len(layouts) >= 140, "insufficient structural variation")
    _require(len({f[1] for f in features}) >= 35, "insufficient canvas variation")
    _require({f[2] for f in features} == {3, 4, 5, 6}, "scale coverage")
    _require({f[3] for f in features} == set(range(10)), "background coverage")
    _require({f[4] for f in features} == set(_CUTS), "cut coverage")
    _require(any(f[5] == 1 or f[6] == 1 for f in features), "singleton queue missing")
    _require(any(abs(f[5]-f[6]) >= 3 for f in features), "long unmatched tail missing")
    _require(len({(f[5], f[6], f[7]) for f in features}) >= 20, "count diversity")
    _require(_signature(generate_puzzle(0)) == _signature(puzzles[0]), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(73, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "count handling")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = copy.deepcopy(puzzles[0])
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed protected border accepted")
    bad = copy.deepcopy(puzzles[0])
    d = _decode(bad["train"][0]["input"])
    frame = d["receivers"][0]
    bad["train"][0]["output"][frame["top"], frame["left"]] = d["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("destroyed container rim accepted")
    for style in ("branch", "loop", "extra_shore"):
        cut = np.zeros((15, 15), dtype=bool)
        if style == "branch":
            cut[:, 7] = True
            cut[7, 7:11] = True
        elif style == "loop":
            cut[4:11, 4] = cut[4:11, 10] = True
            cut[4, 4:11] = cut[10, 4:11] = True
        else:
            cut[:, 4] = cut[:, 10] = True
        try:
            _regions(cut)
        except AssertionError:
            pass
        else:
            raise AssertionError("malformed control accepted")
    return {"seeds": 42, "pairs": len(features), "layouts": len(layouts),
            "canvas_shapes": len({f[1] for f in features}), "cut_families": sorted({f[4] for f in features}),
            "scales": sorted({f[2] for f in features}), "backgrounds": sorted({f[3] for f in features})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
