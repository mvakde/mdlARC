"""Mirror-addressed ink: supported sculpture pixels are paired write addresses.

Only standard-library modules and NumPy are used. Generation builds labels from
known mirrored masks and geometric routes; solve reconstructs everything from
the input. No parent code, examples, random state, or learned palette is used.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np


PORTS = {
    "NW": ((-1, -1), (-1, -1)),
    "N0": ((-1, 0), (-1, 0)), "N1": ((-1, 1), (-1, 0)),
    "NE": ((-1, 2), (-1, 1)),
    "W0": ((0, -1), (0, -1)), "W1": ((1, -1), (0, -1)),
    "E0": ((0, 2), (0, 1)), "E1": ((1, 2), (0, 1)),
    "SW": ((2, -1), (1, -1)),
    "S0": ((2, 0), (1, 0)), "S1": ((2, 1), (1, 0)),
    "SE": ((2, 2), (1, 1)),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _cells(grid, color):
    return set(map(tuple, np.argwhere(grid == color).tolist()))


def _key(grid):
    return grid.shape, grid.tobytes()


def _mate(p, orientation, axis2):
    # orientation 0: vertical axis, 1: horizontal axis.
    return (p[0], axis2 - p[1]) if orientation == 0 else (axis2 - p[0], p[1])


def _symmetry(cells, shape):
    """Unique greatest support over BOTH orientations, from row convolutions."""
    mask = np.zeros(shape, dtype=np.int16)
    for p in cells:
        mask[p] = 1
    candidates = []
    for orientation, work in ((0, mask), (1, mask.T)):
        scores = np.zeros(2 * work.shape[1] - 1, dtype=int)
        for row in work:
            scores += np.convolve(row, row)
        candidates.extend((int(score), orientation, axis)
                          for axis, score in enumerate(scores))
    maximum = max(c[0] for c in candidates)
    winners = [c for c in candidates if c[0] == maximum]
    _require(len(winners) == 1, "sculpture has a tied strongest reflection")
    _, orientation, axis2 = winners[0]
    supported = {p for p in cells if _mate(p, orientation, axis2) in cells}
    return orientation, axis2, supported


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > grid.size / 2, "input needs a majority background")
    candidates = {}
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            payload = int(grid[r, c])
            if payload == bg or not np.all(grid[r:r+2, c:c+2] == payload):
                continue
            ring = [((r+off[0], c+off[1]), name)
                    for name, (off, _) in PORTS.items()
                    if _inside((r+off[0], c+off[1]), grid.shape)]
            foreground = [int(grid[p]) for p, _ in ring if grid[p] != bg]
            colors = set(foreground)
            if len(colors) != 1 or len(foreground) < 3 or len(foreground) == len(ring):
                continue
            structure = next(iter(colors))
            if structure == payload:
                continue
            candidates.setdefault(structure, []).append({
                "core": (r, c), "color": payload,
                "ports": [(p, PORTS[name][1]) for p, name in ring if grid[p] == bg],
                "footprint": {p for p, _ in ring} |
                             {(r+dr, c+dc) for dr in (0, 1) for dc in (0, 1)},
            })
    _require(len(candidates) == 1, "launcher structure must be uniquely identifiable")
    structure, launchers = next(iter(candidates.items()))
    launchers.sort(key=lambda x: x["core"])
    footprints = set()
    payloads = {l["color"] for l in launchers}
    _require(structure not in payloads, "structure cannot also be payload")
    for launcher in launchers:
        _require(not footprints & launcher["footprint"], "launcher footprints overlap")
        footprints |= launcher["footprint"]
    _require(all(p in footprints for color in payloads | {structure}
                 for p in _cells(grid, color)), "launcher colors occur outside launchers")
    sculpture_colors = set(map(int, np.unique(grid))) - {bg, structure} - payloads
    _require(bool(sculpture_colors), "at least one sculpture is needed")
    sculptures = []
    for color in sorted(sculpture_colors):
        original = _cells(grid, color)
        orientation, axis2, core = _symmetry(original, grid.shape)
        _require(len(core) >= 10, "too little supporting sculpture geometry")
        sculptures.append({"color": color, "orientation": orientation,
                           "axis2": axis2, "core": core, "noise": original - core})
    return bg, structure, launchers, sculptures


def _route(start, direction, shape, lookup, forbidden):
    """Forecast against original supported pixels, never partially painted ink."""
    path, p = [], start
    while _inside(p, shape):
        _require(p not in forbidden, "ray intersects another launcher footprint")
        path.append(p)
        if p in lookup:
            sculpture = lookup[p]
            return path, p, _mate(p, sculpture["orientation"], sculpture["axis2"]), sculpture
        p = _add(p, direction)
    return path, None, None, None


def _crossings(path):
    return {((p[0]+q[0], p[1]+q[1]), 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1)
            for p, q in zip(path, path[1:]) if p[0] != q[0] and p[1] != q[1]}


def _check_routes(routes):
    occupied, segments = set(), set()
    for route in routes:
        claimed = set(route["path"])
        if route["mate"] is not None:
            claimed.add(route["mate"])
        _require(not occupied & claimed, "rays or mirrored write addresses overlap")
        crossing = _crossings(route["path"])
        _require(not any((mid, -sign) in segments for mid, sign in crossing),
                 "ray segments cross between cells")
        occupied |= claimed
        segments |= crossing


def _reconstruct(input_grid, keep_noise=False, paired=True, draw=True):
    """The optional modes are documented ablations, not generation conventions."""
    grid = _grid(input_grid)
    bg, structure, launchers, sculptures = _parse(grid)
    base = grid.copy()
    lookup = {}
    for sculpture in sculptures:
        core = sculpture["core"] | sculpture["noise"] if keep_noise else sculpture["core"]
        for p in sculpture["noise"]:
            if not keep_noise:
                base[p] = bg
        for p in core:
            lookup[p] = sculpture
    routes = []
    if draw:
        footprints = set().union(*(l["footprint"] for l in launchers))
        for launcher in launchers:
            for start, direction in launcher["ports"]:
                path, hit, mate, target = _route(start, direction, grid.shape, lookup,
                                                footprints - launcher["footprint"])
                routes.append({"path": path, "hit": hit, "mate": mate,
                               "color": launcher["color"], "direction": direction,
                               "target": target})
        if not keep_noise:
            _check_routes(routes)
    output = base.copy()
    for route in routes:
        for p in route["path"]:
            output[p] = route["color"]
        if paired and route["mate"] is not None and _inside(route["mate"], grid.shape):
            output[route["mate"]] = route["color"]
    return output, (bg, structure, launchers, sculptures, routes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver for every member of the specified family."""
    return _reconstruct(input_grid)[0]


def _sample_mask(rng, parity):
    """Procedural profiles, hollow contours, branches, tips and paired islands."""
    for _ in range(150):
        h = int(rng.integers(4, 11))
        w = int(rng.choice([v for v in range(3, 10) if (v-1) % 2 == parity]))
        radius, odd = w // 2, w % 2
        mask = np.zeros((h, w), dtype=bool)
        family = int(rng.integers(4))
        levels = rng.integers(1, radius+1, h)
        if family == 0:
            level = int(rng.integers(1, radius+1))
            for row in range(h):
                level = int(np.clip(level + rng.integers(-1, 2), 1, radius))
                levels[row] = level
        levels[int(rng.integers(h))] = radius
        for row, level in enumerate(levels):
            if family == 3 and row not in (0, h-1) and rng.random() < .16:
                continue
            for distance in range(1, int(level)+1):
                present = (family == 0 or distance == level or
                           (family == 1 and (row in (0, h-1) or rng.random() < .18)) or
                           (family >= 2 and rng.random() < .6))
                if present:
                    left = radius - distance
                    mask[row, left] = mask[row, w-1-left] = True
            if odd:
                mask[row, radius] = family == 0 or rng.random() < .6
            if family == 3 and odd and row == 0 and rng.random() < .5:
                mask[row] = False
                mask[row, radius] = True
        rows, cols = np.nonzero(mask)
        mask = mask[rows.min():rows.max()+1, cols.min():cols.max()+1]
        cells = set(map(tuple, np.argwhere(mask).tolist()))
        if len(cells) < 10 or mask.shape[1] < 2:
            continue
        try:
            orientation, axis2, supported = _symmetry(cells, mask.shape)
        except ValueError:
            continue
        if orientation == 0 and axis2 == mask.shape[1]-1 and supported == cells:
            return mask
    # Deliberately asymmetric row profile; both parities have a unique full axis.
    w = 5 if parity == 0 else 6
    mask = np.ones((5, w), dtype=bool)
    mask[0, 1:-1] = False
    mask[1, 0] = mask[1, -1] = False
    mask[4] = False
    mask[4, w//2-1:w//2+1] = True
    if parity == 0:
        mask[4] = False
        mask[4, w//2] = True
    return mask


def _place_sculptures(rng, shape, colors, anchor_orientation, anchor_parity):
    sculptures, boxes = [], set()
    for index, color in enumerate(colors):
        for _ in range(100):
            orientation = anchor_orientation if index == 0 else int(rng.integers(2))
            parity = anchor_parity if index == 0 else int(rng.integers(2))
            mask = _sample_mask(rng, parity)
            if orientation:
                mask = mask.T
            h, w = mask.shape
            top = int(rng.integers(shape[0]-h+1))
            left = int(rng.integers(shape[1]-w+1))
            box = {(r, c) for r in range(top, top+h) for c in range(left, left+w)}
            if box & boxes:
                continue
            core = {(top+int(r), left+int(c)) for r, c in np.argwhere(mask)}
            ori, axis, supported = _symmetry(core, shape)
            if ori != orientation or supported != core:
                continue
            sculptures.append({"color": color, "orientation": ori, "axis2": axis,
                               "core": core, "noise": set()})
            # Empty margins prevent a neighboring sculpture from imitating a ring.
            boxes |= {(r, c) for r in range(max(0, top-1), min(shape[0], top+h+1))
                      for c in range(max(0, left-1), min(shape[1], left+w+1))}
            break
        else:
            raise ValueError("sculpture placement exhausted")
    return sculptures, boxes


def _make_launchers(rng, shape, payloads, sculptures, boxes, distance_range,
                    port_kind, shared, teach_miss, restrict_distance=False):
    lookup = {p: s for s in sculptures for p in s["core"]}
    launchers, routes, footprints = [], [], set()
    for index, color in enumerate(payloads):
        for _ in range(220):
            target = sculptures[0] if index == 0 or shared else sculptures[int(rng.integers(len(sculptures)))]
            hits = sorted(target["core"])
            axis_hits = [p for p in hits if p == _mate(p, target["orientation"], target["axis2"])]
            if index > 0 and axis_hits and rng.random() < .35:
                hits = axis_hits
            hit = hits[int(rng.integers(len(hits)))]
            if index == 0 and hit == _mate(hit, target["orientation"], target["axis2"]):
                continue  # The first beam shows a visibly separate twin address.
            names = [name for name, (_, d) in PORTS.items()
                     if index > 0 or port_kind is None or sum(map(abs, d)) == port_kind]
            name = str(rng.choice(names))
            offset, direction = PORTS[name]
            distance = int(rng.integers(distance_range[0], distance_range[1]+1))
            start = (hit[0]-distance*direction[0], hit[1]-distance*direction[1])
            origin = (start[0]-offset[0], start[1]-offset[1])
            if not (0 <= origin[0] < shape[0]-1 and 0 <= origin[1] < shape[1]-1):
                continue
            footprint = {(origin[0]+dr, origin[1]+dc)
                         for dr in range(-1, 3) for dc in range(-1, 3)
                         if _inside((origin[0]+dr, origin[1]+dc), shape)}
            claims = set().union(*(set(r["path"]) | {r["mate"]} for r in routes)) if routes else set()
            if footprint & (boxes | footprints | claims) or len(footprint)-5 < 3:
                continue
            try:
                path, actual_hit, mate, actual_target = _route(start, direction, shape, lookup, footprints)
                if actual_hit != hit:
                    continue
                route = {"path": path, "hit": hit, "mate": mate, "target": actual_target,
                         "color": color, "direction": direction}
                _check_routes(routes + [route])
            except ValueError:
                continue
            launchers.append({"core": origin, "color": color, "names": [name],
                              "footprint": footprint})
            footprints |= footprint
            routes.append(route)
            break
        else:
            raise ValueError("targeted launcher placement exhausted")
    # Extra ports are evaluated after ALL footprints are placed. Some miss,
    # some hit a different sculpture or an axis cell; none conflict with ink.
    added_miss = False
    order = list(rng.permutation(len(launchers)))
    for index in order:
        launcher = launchers[int(index)]
        names = list(PORTS)
        rng.shuffle(names)
        desired = int(rng.integers(0, 3))
        if teach_miss and not added_miss:
            desired = max(1, desired)
        for name in names:
            if desired == 0:
                break
            if name in launcher["names"] or len(launcher["names"]) >= 3:
                continue
            offset, direction = PORTS[name]
            start = _add(launcher["core"], offset)
            if not _inside(start, shape) or len(launcher["footprint"])-4-len(launcher["names"])-1 < 3:
                continue
            try:
                path, hit, mate, target = _route(start, direction, shape, lookup,
                                                footprints - launcher["footprint"])
                if (restrict_distance and hit is not None and
                        not distance_range[0] <= len(path)-1 <= distance_range[1]):
                    continue
                if teach_miss and not added_miss and hit is not None:
                    continue
                route = {"path": path, "hit": hit, "mate": mate, "target": target,
                         "color": launcher["color"], "direction": direction}
                _check_routes(routes + [route])
            except ValueError:
                continue
            launcher["names"].append(name)
            routes.append(route)
            desired -= 1
            added_miss |= hit is None
    _require(not teach_miss or added_miss, "no boundary-reaching teaching port")
    return launchers, routes, footprints


def _noise_candidate(sculpture, point, shape):
    cells = sculpture["core"] | sculpture["noise"] | {point}
    try:
        ori, axis, supported = _symmetry(cells, shape)
    except ValueError:
        return False
    return (ori, axis) == (sculpture["orientation"], sculpture["axis2"]) and supported == sculpture["core"]


def _make_pair(rng, parameters, anchor=None, testing=False):
    shift = parameters["shift"]
    for _ in range(100):
        h, w = map(int, rng.integers(16, 31, 2))
        shape = (h, w)
        sculpture_count = int(rng.integers(1, 4))
        if shift == "count":
            sculpture_count = int(rng.integers(3, 5) if testing else rng.integers(1, 3))
        if anchor is not None and anchor["shared"]:
            sculpture_count = max(2, sculpture_count)
        launcher_count = (2 if anchor and anchor["shared"] else 1) if anchor else int(rng.integers(1, 4))
        palette = list(map(int, rng.choice(10, 2+sculpture_count+launcher_count, replace=False)))
        bg, structure = palette[:2]
        sculpture_colors = palette[2:2+sculpture_count]
        payloads = palette[2+sculpture_count:]
        ori = anchor["orientation"] if anchor else int(rng.integers(2))
        parity = anchor["parity"] if anchor else int(rng.integers(2))
        distances = (6, 10) if testing and shift == "distance" else ((2, 5) if shift == "distance" else (2, 9))
        try:
            sculptures, boxes = _place_sculptures(rng, shape, sculpture_colors, ori, parity)
            launchers, routes, footprints = _make_launchers(
                rng, shape, payloads, sculptures, boxes, distances,
                anchor["port_kind"] if anchor else None,
                bool(anchor and anchor["shared"]) or rng.random() < .6,
                bool(anchor and anchor["miss"]), restrict_distance=shift == "distance")
            occupied = footprints | set().union(*(s["core"] for s in sculptures))
            # Both teaching scenes contain debris in front of a genuine hit.
            main = routes[0]
            candidates = list(reversed(main["path"][:-1]))
            if anchor:
                blocked = False
                for p in candidates:
                    if p not in occupied and _noise_candidate(main["target"], p, shape):
                        main["target"]["noise"].add(p)
                        occupied.add(p)
                        blocked = True
                        break
                _require(blocked, "cannot add a uniquely unsupported ray blocker")
            ink_cells = set().union(*(set(r["path"]) | ({r["mate"]} if r["mate"] else set()) for r in routes))
            noise_budget = int(rng.integers(8, 13)) if testing and shift == "noise" else (
                           int(rng.integers(2, 5)) if shift == "noise" else int(rng.integers(3, 11)))
            all_positions = [(r, c) for r in range(h) for c in range(w) if (r, c) not in occupied]
            rng.shuffle(all_positions)
            initial_noise = sum(len(s["noise"]) for s in sculptures)
            # Mix local holes/protrusions with detached debris. Unilluminated
            # noise is mandatory in teaching scenes and remains erased in output.
            selected = initial_noise
            for p in all_positions:
                if selected >= noise_budget:
                    break
                if p in ink_cells and (anchor or rng.random() < .75):
                    continue
                ranked = sorted(sculptures, key=lambda s: min(max(abs(p[0]-q[0]), abs(p[1]-q[1])) for q in s["core"]))
                target = ranked[0] if rng.random() < .8 else sculptures[int(rng.integers(len(sculptures)))]
                if _noise_candidate(target, p, shape):
                    target["noise"].add(p)
                    occupied.add(p)
                    selected += 1
            _require(selected == noise_budget, "noise budget cannot preserve unique axes")
            _require(len(occupied) < h*w/2, "background majority lost")
            clean = np.full(shape, bg, dtype=np.int8)
            for sculpture in sculptures:
                for p in sculpture["core"]:
                    clean[p] = sculpture["color"]
            for launcher in launchers:
                r, c = launcher["core"]
                clean[r:r+2, c:c+2] = launcher["color"]
                for name, (offset, _) in PORTS.items():
                    p = _add((r, c), offset)
                    if _inside(p, shape) and name not in launcher["names"]:
                        clean[p] = structure
            inp, out = clean.copy(), clean.copy()
            for sculpture in sculptures:
                for p in sculpture["noise"]:
                    inp[p] = sculpture["color"]
            # Labels use construction geometry, not solve(inp).
            for route in routes:
                for p in route["path"]:
                    out[p] = route["color"]
                if route["mate"] is not None:
                    out[route["mate"]] = route["color"]
            predicted, info = _reconstruct(inp)
            _require(np.array_equal(predicted, out), "construction and input reconstruction disagree")
            if anchor:
                metrics = _measure_info(inp, out, info)
                _require(metrics["blocked_rays"] and metrics["erased_noise"] and metrics["remote_hits"],
                         "teaching scene lacks the connected mechanisms")
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    parameters = {"shift": str(rng.choice(["count", "distance", "noise"]))}
    orientations, parities, kinds = list(rng.permutation(2)), list(rng.permutation(2)), list(rng.permutation([1, 2]))
    anchors = [{"orientation": int(orientations[i]), "parity": int(parities[i]),
                "port_kind": int(kinds[i]), "shared": i == 1, "miss": i == 0}
               for i in range(2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(int(count)):
            for _ in range(30):
                pair = _make_pair(rng, parameters, anchors[index] if split == "train" and index < 2 else None,
                                  testing=split == "test")
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded duplicate rejection exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure_info(inp, out, info):
    bg, structure, launchers, sculptures, routes = info
    noise = set().union(*(s["noise"] for s in sculptures))
    hit_counts = Counter(r["target"]["color"] for r in routes if r["target"] is not None)
    return {
        "shape": tuple(inp.shape), "background": bg, "structure": structure,
        "sculptures": len(sculptures), "launchers": len(launchers),
        "port_counts": tuple(len(l["ports"]) for l in launchers),
        "orientations": tuple(sorted({s["orientation"] for s in sculptures})),
        "parities": tuple(sorted({s["axis2"] % 2 for s in sculptures})),
        "directions": tuple(sorted({r["direction"] for r in routes})),
        "noise": len(noise), "erased_noise": sum(out[p] == bg for p in noise),
        "blocked_rays": sum(bool(set(r["path"][:-1]) & noise) for r in routes if r["hit"]),
        "remote_hits": sum(r["hit"] != r["mate"] for r in routes if r["hit"] is not None),
        "axis_hits": sum(r["hit"] == r["mate"] for r in routes if r["hit"] is not None),
        "misses": sum(r["hit"] is None for r in routes),
        "shared_target": max(hit_counts.values(), default=0),
        "clipped_launchers": sum(len(l["footprint"]) < 16 for l in launchers),
        "hit_distances": tuple(len(r["path"])-1 for r in routes if r["hit"] is not None),
        "bbox_bias": sum((min(p[1] for p in s["core"] | s["noise"]) + max(p[1] for p in s["core"] | s["noise"]) != s["axis2"])
                         if s["orientation"] == 0 else
                         (min(p[0] for p in s["core"] | s["noise"]) + max(p[0] for p in s["core"] | s["noise"]) != s["axis2"])
                         for s in sculptures),
    }


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "illegal example counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output dimensions changed")
            _require(_key(inp) not in seen, "duplicate inputs")
            seen.add(_key(inp))
            expected, info = _reconstruct(inp)
            _require(np.array_equal(expected, out), "output violates mirror-addressed ink")
            bg, structure, launchers, sculptures, routes = info
            _require(1 <= len(launchers) <= 3 and 1 <= len(sculptures) <= 4, "object counts out of domain")
            fixed = (inp == structure) | np.isin(inp, [l["color"] for l in launchers])
            _require(np.array_equal(inp[fixed], out[fixed]), "launcher material changed")
            _require(any(r["hit"] is not None for r in routes), "no supported contact")
            for sculpture in sculptures:
                # Only symmetric ink pairs are allowed to recolor the core.
                for p in sculpture["core"]:
                    _require(out[p] == out[_mate(p, sculpture["orientation"], sculpture["axis2"])],
                             "paired sculpture pixels have different ink")
            metrics = _measure_info(inp, out, info)
            if split == "train":
                training.append(metrics)
            else:
                testing.append(metrics)
    _require({o for m in training for o in m["orientations"]} == {0, 1}, "both reflection orientations need evidence")
    _require({p for m in training for p in m["parities"]} == {0, 1}, "both axis parities need evidence")
    _require({sum(map(abs, d)) for m in training for d in m["directions"]} == {1, 2}, "side and corner emission need evidence")
    _require(sum(m["blocked_rays"] > 0 and m["erased_noise"] > 0 and m["remote_hits"] > 0 for m in training) >= 2,
             "two informative connected demonstrations are required")
    _require(any(m["misses"] for m in training), "boundary continuation needs evidence")
    _require(any(m["shared_target"] >= 2 for m in training), "multiple writes to one sculpture need evidence")
    count_shift = max(m["sculptures"] for m in training) <= 2 and min(m["sculptures"] for m in testing) >= 3
    noise_shift = max(m["noise"] for m in training) <= 4 and min(m["noise"] for m in testing) >= 8
    distance_shift = (max(d for m in training for d in m["hit_distances"]) <= 5 and
                      min(d for m in testing for d in m["hit_distances"]) >= 6)
    _require(count_shift or noise_shift or distance_shift, "no measurable train/test shift")
    # Omission checks are actual competing outputs, not sampled parameter tags.
    for mode in ("no_symmetry", "no_emission", "no_twin"):
        differs = False
        for pair in puzzle["train"]:
            try:
                alternative = _reconstruct(pair["input"], keep_noise=mode == "no_symmetry",
                                           draw=mode != "no_emission", paired=mode != "no_twin")[0]
                differs |= not np.array_equal(alternative, pair["output"])
            except ValueError:
                differs = True
        _require(differs, f"training does not reject {mode}")


def self_test():
    """Bounded, grid-derived variation and semantic invariants across 42 seeds."""
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                predicted, info = _reconstruct(inp)
                assert np.array_equal(predicted, out)
                # Coordinate-set scoring checks convolution support independently.
                for s in info[3]:
                    cells = s["core"] | s["noise"]
                    scored = [(sum(_mate(p, ori, a) in cells for p in cells), ori, a)
                              for ori in (0, 1) for a in range(2*inp.shape[1-ori]-1)]
                    best = max(c[0] for c in scored)
                    winners = [c for c in scored if c[0] == best]
                    assert winners == [(len(s["core"]), s["orientation"], s["axis2"])]
                metrics = _measure_info(inp, out, info)
                metrics["split"] = split
                records.append(metrics)
        if seed < 3:
            repeated = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeated[s])
                       for k in ("input", "output"))
    assert len(signatures) == 42
    assert len({r["shape"] for r in records}) >= 40
    assert {r["background"] for r in records} == set(range(10))
    assert {r["structure"] for r in records} == set(range(10))
    assert {r["sculptures"] for r in records} == {1, 2, 3, 4}
    assert {r["launchers"] for r in records} == {1, 2, 3}
    assert len({d for r in records for d in r["directions"]}) == 8
    assert any(len(r["orientations"]) == 2 for r in records)
    assert any(r["axis_hits"] for r in records)
    assert any(r["clipped_launchers"] for r in records)
    assert any(r["bbox_bias"] for r in records)
    assert max(max(r["port_counts"]) for r in records) == 3
    assert any(r["shared_target"] >= 3 for r in records)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(700+a*10+b, a, b)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(5, 2, 1)
    out = corrupt["test"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    # Return descriptive evidence; these are generator-owned, finite-seed checks.
    return {"seeds": 42, "pairs": len(records), "shapes": len({r["shape"] for r in records}),
            "sculpture_counts": sorted({r["sculptures"] for r in records}),
            "launcher_counts": sorted({r["launchers"] for r in records}),
            "blocked_rays": sum(r["blocked_rays"] for r in records),
            "remote_hits": sum(r["remote_hits"] for r in records),
            "axis_hits": sum(r["axis_hits"] for r in records),
            "clipped_launchers": sum(r["clipped_launchers"] for r in records)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
