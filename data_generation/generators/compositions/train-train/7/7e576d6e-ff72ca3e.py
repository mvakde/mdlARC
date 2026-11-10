"""The widest square brush that can negotiate a portal slalom.

Only the input determines the palette, route, and maximum brush radius.
No parent code, examples, or generation parameters are used by solve().
"""

from collections import Counter, deque

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    x = np.asarray(value)
    _require(x.ndim == 2 and all(1 <= n <= 30 for n in x.shape),
             "Grids must have dimensions 1..30.")
    _require(np.issubdtype(x.dtype, np.integer) and
             np.all((0 <= x) & (x <= 9)), "Illegal cell values.")
    return x


def _positions(x, color):
    return [tuple(map(int, p)) for p in np.argwhere(x == color)]


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _analyze(value):
    """Read roles geometrically, and normalize progress to left-to-right."""
    original = _grid(value)
    colors, counts = np.unique(original, return_counts=True)
    _require(len(colors) == 5, "Expected five visible roles.")
    _require(sum(counts == max(counts)) == 1, "Background must be unique modal.")
    background = int(colors[np.argmax(counts)])
    groups = {}
    for transposed in (False, True):
        x = original.T if transposed else original
        for b in range(x.shape[1]):
            line = x[:, b]
            wall = int(line[0])
            if wall == background or int(line[-1]) != wall:
                continue
            other = np.flatnonzero(line != wall)
            if len(other) < 3 or len(other) % 2 != 1:
                continue
            if other[0] == 0 or other[-1] == len(line) - 1:
                continue
            if not np.all(np.diff(other) == 1):
                continue
            portal = int(line[other[0]])
            if portal == background or not np.all(line[other] == portal):
                continue
            key = (transposed, wall, portal)
            groups.setdefault(key, []).append(
                (b, int(other[len(other) // 2]), len(other) // 2))
    _require(len(groups) == 1, "Ambiguous barrier geometry.")
    (transposed, wall, portal), gates = next(iter(groups.items()))
    gates.sort()
    x = original.T if transposed else original
    remaining = set(map(int, colors)) - {background, wall, portal}
    route_candidates = [c for c in remaining if np.count_nonzero(x == c) == 2]
    _require(len(route_candidates) == 1, "Endpoints must uniquely identify the brush color.")
    route = route_candidates[0]
    stop = (remaining - {route}).pop()
    endpoints = sorted(_positions(x, route), key=lambda p: p[1])
    lead, tail = endpoints
    _require(lead[1] < gates[0][0] and tail[1] > gates[-1][0],
             "Endpoints must be in opposite exterior chambers.")
    stops = _positions(x, stop)
    _require(len(stops) >= 3, "At least three isolated stopping markers are required.")
    markers = stops + endpoints
    _require(all(_cheb(a, b) > 1 for i, a in enumerate(markers)
                 for b in markers[i + 1:]), "Markers must be isolated.")
    rebuilt = np.full(x.shape, background, dtype=int)
    for b, p, q in gates:
        rebuilt[:, b] = wall
        rebuilt[p - q:p + q + 1, b] = portal
    for p in stops:
        rebuilt[p] = stop
    for p in endpoints:
        rebuilt[p] = route
    _require(np.array_equal(rebuilt, x), "Unexpected input objects.")
    return dict(x=x, transposed=transposed, background=background, wall=wall,
                portal=portal, route=route, stop=stop, gates=gates,
                lead=lead, tail=tail, stops=stops)


def _segments(info, radius, *, clearance=True):
    current = info["lead"]
    segments = []
    forward = True
    for b, p, _ in info["gates"]:
        turn = b - (radius + 1 if clearance else 1)
        forward = forward and turn >= current[1]
        for target in ((current[0], turn), (p, turn), (p, b)):
            segments.append((current, target))
            current = target
    for target in ((current[0], info["tail"][1]), info["tail"]):
        segments.append((current, target))
        current = target
    return segments, forward


def _sweep(info, radius, *, clearance=True):
    """Union square stamps along segments, using their exact rectangles."""
    h, w = info["x"].shape
    mask = np.zeros((h, w), dtype=bool)
    segments, forward = _segments(info, radius, clearance=clearance)
    inside = True
    for a, b in segments:
        lo_r, hi_r = min(a[0], b[0]) - radius, max(a[0], b[0]) + radius
        lo_c, hi_c = min(a[1], b[1]) - radius, max(a[1], b[1]) + radius
        inside = inside and lo_r >= 0 and lo_c >= 0 and hi_r < h and hi_c < w
        if max(lo_r, 0) <= min(hi_r, h - 1) and max(lo_c, 0) <= min(hi_c, w - 1):
            mask[max(0, lo_r):min(h, hi_r + 1), max(0, lo_c):min(w, hi_c + 1)] = True
    return mask, inside, forward


def _failures(info, radius, *, ignore_stops=False, ignore_walls=False):
    mask, inside, forward = _sweep(info, radius)
    reasons = set()
    if not inside:
        reasons.add("edge")
    if not forward:
        reasons.add("progress")
    if not ignore_walls and np.any(mask & (info["x"] == info["wall"])):
        reasons.add("wall")
    if not ignore_stops and np.any(mask & (info["x"] == info["stop"])):
        reasons.add("stop")
    return reasons


def _maximum(info, **kwargs):
    for radius in range((min(info["x"].shape) - 1) // 2, -1, -1):
        if not _failures(info, radius, **kwargs):
            return radius
    raise ValueError("No brush can follow the prescribed route.")


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    radius = _maximum(info)
    mask, _, _ = _sweep(info, radius)
    y = info["x"].copy()
    y[mask] = info["route"]
    return y.T.copy() if info["transposed"] else y


def _key(x):
    return x.shape, np.asarray(x, dtype=np.uint8).tobytes()


def _path_cells(info, radius):
    cells = set()
    for a, b in _segments(info, radius)[0]:
        dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
        for i in range(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1):
            cells.add((a[0] + dr * i, a[1] + dc * i))
    return cells


def _construct(rng, palette, mode, transposed, rich, test):
    """Construct clearance geometry, then select blockers from swept layers."""
    has_stop_limit = mode in ("marker", "mixed", "mixed_edge")
    has_gate_limit = mode in ("gate", "mixed", "mixed_edge")
    has_edge_limit = mode in ("edge", "mixed_edge")
    expected_causes = ({"stop"} if has_stop_limit else set()) | (
        {"wall"} if has_gate_limit else set()) | ({"edge"} if has_edge_limit else set())
    background, wall, portal, route, stop = palette
    radii = (1, 2, 3) if rich else (1, 2, 3, 4)
    for _ in range(240):
        radius = int(rng.choice(radii))
        s = radius + 1  # Leave geometric room for the next larger candidate.
        max_count = min(4 if test else 3, 31 // (2 * s + 2) - 1)
        min_count = 2 if rich else 1
        if max_count < min_count:
            continue
        n = int(rng.integers(min_count, max_count + 1))
        h = int(rng.integers(max(13, 2 * radius + 9), 31))
        minimum_w = (n + 1) * (2 * s + 2) - 1
        w = int(rng.integers(minimum_w, 31))
        intervals = np.array([s, s + 1] + [2 * s + 2] * (n - 1) + [s + 1, s])
        intervals += rng.multinomial(w - minimum_w, np.ones(n + 3) / (n + 3))
        lead_c = int(intervals[0])
        barriers = [lead_c + int(intervals[1])]
        for gap in intervals[2:n + 1]:
            barriers.append(barriers[-1] + int(gap))
        tail_c = barriers[-1] + int(intervals[-2])
        q_max = min(radius + 3, (h - 3) // 2)
        halfwidths = [int(rng.integers(s, q_max + 1)) for _ in range(n)]
        if has_gate_limit:
            halfwidths[int(rng.integers(n))] = radius
            if n > 2 and rng.random() < 0.3:
                halfwidths[int(rng.integers(n))] = radius
        portal_centers = [int(rng.integers(q + 1, h - q - 1))
                          if h - 2 * q > 3 else h // 2 for q in halfwidths]
        # Apertures always leave a wall cell at both canvas ends.
        if any(p - q < 1 or p + q >= h - 1 for p, q in zip(portal_centers, halfwidths)):
            continue
        lead_r = int(rng.integers(s, h - s))
        tail_r = int(rng.integers(s, h - s))
        if has_edge_limit:
            lead_r = radius if rng.random() < 0.5 else h - 1 - radius
        lead, tail = (lead_r, lead_c), (tail_r, tail_c)
        if rich and max(abs(b - a) for a, b in zip(
                [lead_r] + portal_centers, portal_centers + [tail_r])) < 3:
            continue
        x = np.full((h, w), background, dtype=np.int8)
        gates = list(zip(barriers, portal_centers, halfwidths))
        for b, p, q in gates:
            x[:, b] = wall
            x[p - q:p + q + 1, b] = portal
        x[lead] = x[tail] = route
        info = dict(x=x, gates=gates, lead=lead, tail=tail, wall=wall,
                    portal=portal, stop=stop, route=route, background=background)
        small = _sweep(info, radius)[0]
        large = _sweep(info, s)[0]
        eligible = (x == background)
        markers = [lead, tail]
        stops = []

        def choose(pool):
            candidates = [tuple(map(int, p)) for p in np.argwhere(pool)
                          if all(_cheb(p, m) > 1 for m in markers + stops)]
            if not candidates:
                return False
            stops.append(candidates[int(rng.integers(len(candidates)))])
            return True

        if has_stop_limit:
            ring = large & ~small & eligible
            # For rich demonstrations, use a square-corner obstruction that
            # a Manhattan-radius brush at the next size would miss.
            if rich:
                rows, cols = np.indices(x.shape)
                manhattan = np.full(x.shape, 100, dtype=int)
                for pr, pc in _path_cells(info, s):
                    manhattan = np.minimum(manhattan, abs(rows - pr) + abs(cols - pc))
                ring &= manhattan > s
            if not choose(ring):
                continue
            if rng.random() < 0.35:
                choose(ring)
        extras = eligible & ~large
        wanted_stops = int(rng.integers(3, 10 if test else 8))
        while len(stops) < wanted_stops and choose(extras):
            pass
        if len(stops) < 3:
            continue
        for p in stops:
            x[p] = stop
        if _failures(info, radius) or _failures(info, s) != expected_causes:
            continue
        if _maximum(info) != radius:
            continue
        if rich:
            wrong = _sweep(info, radius, clearance=False)[0]
            paintable = (x == background) | (x == portal)
            if np.array_equal(wrong & paintable, small & paintable):
                continue
        original = x.T.copy() if transposed else x
        return {"input": original, "output": solve(original)}
    raise RuntimeError("Could not construct the requested clearance demonstration.")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer.")
    if not low <= int(value) <= high:
        raise ValueError(name + " is out of range.")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, size=5, replace=False)))
    first_orientation = bool(rng.integers(2))
    plans = [("gate", first_orientation, True), ("marker", not first_orientation, True)]
    if num_train >= 3:
        plans.append(("edge", bool(rng.integers(2)), False))
    while len(plans) < num_train:
        plans.append((str(rng.choice(("gate", "marker", "edge"))),
                      bool(rng.integers(2)), bool(rng.random() < 0.45)))
    rng.shuffle(plans)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        if split == "test":
            plans = [("mixed_edge" if num_train >= 3 and rng.random() < 0.3 else "mixed",
                      bool(rng.integers(2)), True) for _ in range(num_test)]
        for mode, orientation, rich in plans:
            for _ in range(20):
                pair = _construct(rng, palette, mode, orientation, rich, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Unable to generate distinct inputs.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _connected(mask):
    start = tuple(map(int, np.argwhere(mask)[0]))
    seen, todo = {start}, deque([start])
    h, w = mask.shape
    while todo:
        r, c = todo.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == int(mask.sum())


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Exactly train and test are required.")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "Splits must be lists.")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Illegal pair counts.")
    seen, palettes, train_causes, train_orientations = set(), set(), set(), set()
    clearance_causes, metric_evidence = set(), False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "A pair must have exactly input and output.")
            x, y = _grid(pair["input"]), _grid(pair["output"])
            _require(x.shape == y.shape and _key(x) not in seen, "Shape change or duplicate input.")
            seen.add(_key(x))
            info = _analyze(x)
            palettes.add(tuple(info[k] for k in ("background", "wall", "portal", "route", "stop")))
            radius = _maximum(info)
            _require(1 <= radius <= 4, "Generated brushes must have radius 1..4.")
            mask, inside, forward = _sweep(info, radius)
            _require(inside and forward and _connected(mask), "The trail must be continuous and contained.")
            yn = y.T if info["transposed"] else y
            expected = info["x"].copy()
            expected[mask] = info["route"]
            _require(np.array_equal(yn, expected), "Incorrect swept brush output.")
            _require(np.array_equal(solve(x), y), "Input-only solve disagrees.")
            protected = (info["x"] == info["wall"]) | (info["x"] == info["stop"])
            _require(np.array_equal(yn[protected], info["x"][protected]), "Protected object changed.")
            _require(not np.array_equal(x, y), "Every pair must visibly change.")
            for b, p, q in info["gates"]:
                crossed = np.flatnonzero(yn[:, b] == info["route"])
                _require(np.array_equal(crossed, np.arange(p - radius, p + radius + 1)),
                         "Each barrier must be crossed by a centered brush band.")
                _require(radius <= q, "The brush does not fit a portal.")
            # Both inherited mechanisms must visibly matter in every pair.
            thin = info["x"].copy()
            thin[_sweep(info, 0)[0]] = info["route"]
            _require(not np.array_equal(thin, yn), "Ignoring square growth leaves the answer unchanged.")
            endpoint_only = np.zeros(mask.shape, dtype=bool)
            for r, c in (info["lead"], info["tail"]):
                endpoint_only[r - radius:r + radius + 1, c - radius:c + radius + 1] = True
            _require(np.any(mask & ~endpoint_only), "Ignoring portal traversal leaves the answer unchanged.")
            causes = frozenset(_failures(info, radius + 1))
            if split == "train":
                train_causes.add(causes)
                train_orientations.add(info["transposed"])
                if causes in (frozenset({"wall"}), frozenset({"stop"})):
                    wrong = _sweep(info, radius, clearance=False)[0]
                    paintable = (info["x"] == info["background"]) | (info["x"] == info["portal"])
                    if np.any((wrong ^ mask) & paintable):
                        clearance_causes.add(causes)
                if causes == frozenset({"stop"}):
                    _require(_maximum(info, ignore_stops=True) > radius,
                             "Stopping markers must actively constrain a demonstration.")
                    s = radius + 1
                    cells = _path_cells(info, s)
                    metric_evidence |= all(min(abs(p[0] - q[0]) + abs(p[1] - q[1])
                                               for q in cells) > s for p in info["stops"])
                if causes == frozenset({"wall"}):
                    _require(_maximum(info, ignore_walls=True) > radius,
                             "A portal must actively constrain a demonstration.")
            else:
                _require({"wall", "stop"} <= causes, "Tests must combine gate and marker bottlenecks.")
                _require("edge" not in causes or frozenset({"edge"}) in train_causes,
                         "An edge-limited test needs a boundary demonstration.")
    _require(len(palettes) == 1, "Palette roles must be consistent within a puzzle.")
    _require({False, True} == train_orientations, "Demonstrate both orientations.")
    _require({frozenset({"wall"}), frozenset({"stop"})} <= train_causes,
             "Demonstrate aperture and stopper limits separately.")
    _require({frozenset({"wall"}), frozenset({"stop"})} <= clearance_causes and metric_evidence,
             "Demonstrations must expose turn clearance and off-axis square obstruction.")
    if len(train) >= 3:
        _require(frozenset({"edge"}) in train_causes, "Missing boundary demonstration.")


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded diversity, counterfactual, count, and cell-stamp checks."""
    shapes, palettes, layouts, radii, counts, widths, causes = (set() for _ in range(7))
    orientations = {"train": set(), "test": set()}
    structures = set()
    stats = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        structures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = pair["input"]
                info = _analyze(x)
                radius = _maximum(info)
                shapes.add(x.shape)
                palettes.add(tuple(info[k] for k in ("background", "wall", "portal", "route", "stop")))
                layouts.add((x.shape, tuple(info["gates"]), info["lead"], info["tail"], tuple(info["stops"])))
                radii.add(radius)
                counts.add(len(info["gates"]))
                widths.update(2 * q + 1 for _, _, q in info["gates"])
                causes.add(frozenset(_failures(info, radius + 1)))
                orientations[split].add(info["transposed"])
                stats["stops"] = max(stats["stops"], len(info["stops"]))
                ps = [info["lead"][0]] + [p for _, p, _ in info["gates"]] + [info["tail"][0]]
                delta = np.diff(ps)
                stats["aligned"] += int(np.any(delta == 0))
                stats["reversal"] += int(any(a * b < 0 for a, b in zip(delta, delta[1:])))
                stats["wide"] += int(x.shape[1] > x.shape[0] + 5)
                stats["tall"] += int(x.shape[0] > x.shape[1] + 5)
                # A separate cell-by-cell stamp construction verifies the
                # rectangle union, including every corner and gate crossing.
                stamped = np.zeros(info["x"].shape, dtype=bool)
                for r, c in _path_cells(info, radius):
                    stamped[r - radius:r + radius + 1, c - radius:c + radius + 1] = True
                _require(np.array_equal(stamped, _sweep(info, radius)[0]), "Sweep geometry mismatch.")
    _require(len(structures) == 42 and len(palettes) >= 30 and len(layouts) >= 200,
             "Insufficient seed or palette diversity.")
    _require(len(shapes) >= 60 and {1, 2, 3, 4} <= radii and {1, 2, 3, 4} <= counts,
             "Insufficient geometric variation.")
    _require({3, 5, 7, 9} <= widths and all(v == {False, True} for v in orientations.values()),
             "Missing aperture widths or orientations.")
    _require(all(frozenset(v) in causes for v in ({"wall"}, {"stop"}, {"edge"},
                                                {"wall", "stop"}, {"wall", "stop", "edge"})),
             "Missing bottleneck regimes.")
    _require(all(stats[k] > 0 for k in ("aligned", "reversal", "wide", "tall")) and stats["stops"] >= 8,
             "Missing route patterns or densities.")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(781 + a * 7 + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "Counts ignored.")
    _require(_puzzle_key(generate_puzzle(67, 5, 2)) == _puzzle_key(generate_puzzle(67, 5, 2)),
             "Generation is nondeterministic.")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted.")
    return {"seeds": 42, "shapes": len(shapes), "palettes": len(palettes),
            "layouts": len(layouts), "radii": sorted(radii), "barrier_counts": sorted(counts),
            "portal_widths": sorted(widths), "observations": dict(stats)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
