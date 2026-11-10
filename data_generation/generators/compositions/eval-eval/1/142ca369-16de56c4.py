"""Folded pulse trails: arithmetic phase lives on a reflected route.

Only NumPy is required. Generation plans geometry; solve reconstructs every
object, route, reference pair, and terminal from the supplied input alone.
"""

import numpy as np


_HOST_SEEDS = (0, 659027227, 964360945, 473996326, 186333670, 881710579,
               630868482, 563179158, 39935560, 53227609, 108049925, 62497349,
               157762524, 484926111, 787819146, 616181376, 233095250, 145847307,
               238970462, 194674147, 492784377, 493458781, 895091500, 927442345,
               888431339, 99085528, 469504278, 635275071, 497714206, 67449607,
               900348952, 805956505, 284518026, 949585751, 952935853, 146409336,
               765038062, 231160902, 329957678, 981447461, 696565275, 80847181)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer colors required")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _source(elbow, direction):
    r, c = elbow
    dr, dc = direction
    return {"cells": {elbow, (r - dr, c), (r, c - dc)},
            "elbow": elbow, "direction": direction}


def _mirror(midpoint, axis, length):
    r, c = midpoint
    cells = {(r, c + k) if axis == 0 else (r + k, c)
             for k in range(-(length // 2), length // 2 + 1)}
    return {"cells": cells, "midpoint": midpoint, "axis": axis}


def _parse_objects(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > grid.size, "majority background required")
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    sources, mirrors, beads = [], [], {}
    # Connectivity is color-sensitive; distinct-color touching shapes remain
    # separate objects. No arbitrary decomposition of same-color shapes.
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(grid[first])
        cells, stack = {first}, [first]
        while stack:
            for p in _neighbors(stack.pop()):
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        if len(cells) == 1:
            beads[first] = color
        elif len(cells) == 3 and r1 - r0 == c1 - c0 == 1:
            missing = next(iter({(r, c) for r in (r0, r1)
                                  for c in (c0, c1)} - cells))
            elbow = (r0 + r1 - missing[0], c0 + c1 - missing[1])
            sources.append(dict(_source(elbow, (elbow[0] - missing[0],
                                                 elbow[1] - missing[1])), color=color))
        elif (len(cells) >= 3 and len(cells) % 2 == 1
              and (r0 == r1 or c0 == c1)
              and max(r1 - r0, c1 - c0) + 1 == len(cells)):
            mirrors.append(dict(_mirror(((r0 + r1) // 2, (c0 + c1) // 2),
                                         0 if r0 == r1 else 1, len(cells)), color=color))
        else:
            raise ValueError("foreground is not an L, odd bar, or singleton")
    _require(bool(sources), "an L launch is required")
    return background, sources, mirrors, beads


def _trace(shape, sources, mirrors):
    """Trace transparent routes, before interpreting any pulse samples."""
    objects = [o["cells"] for o in sources + mirrors]
    occupied = set().union(*objects)
    _require(sum(map(len, objects)) == len(occupied), "overlapping objects")
    routes, all_cells, used = [], set(), set()
    for source in sources:
        dr, dc = source["direction"]
        r, c = source["elbow"]
        p = (r + dr, c + dc)
        path, outgoing, contacts, visited = [], [], [], set()
        while _inside(p, shape):
            _require(p not in occupied, "route strikes a structural object")
            _require(p not in visited, "route revisits a cell")
            visited.add(p)
            hits = []
            for index, mirror in enumerate(mirrors):
                mr, mc = mirror["midpoint"]
                axis = mirror["axis"]
                if ((axis == 0 and p == (mr - dr, mc))
                        or (axis == 1 and p == (mr, mc - dc))):
                    hits.append((index, axis))
            _require(len(hits) <= 1, "two mirrors compete for one contact")
            path.append(p)
            if hits:
                index, axis = hits[0]
                contacts.append((len(path), axis, index))
                used.add(index)
                if axis == 0:
                    dr = -dr
                else:
                    dc = -dc
            outgoing.append((dr, dc))
            p = (p[0] + dr, p[1] + dc)
        _require(not all_cells.intersection(visited), "routes share a cell")
        all_cells.update(visited)
        routes.append({"path": path, "outgoing": outgoing, "contacts": contacts})
    _require(used == set(range(len(mirrors))), "bar is not on any virtual route")
    return routes


def _analyze(input_grid):
    grid = _array(input_grid)
    background, sources, mirrors, beads = _parse_objects(grid)
    routes = _trace(grid.shape, sources, mirrors)
    accounted = set()
    for route in routes:
        samples = {t: beads[p] for t, p in enumerate(route["path"], 1) if p in beads}
        frequencies = {}
        for shade in samples.values():
            frequencies[shade] = frequencies.get(shade, 0) + 1
        repeated = [shade for shade, count in frequencies.items() if count > 1]
        _require(len(repeated) == 1 and frequencies[repeated[0]] == 2,
                 "each route needs exactly one equal-colored singleton pair")
        reference = repeated[0]
        a, b = sorted(t for t, shade in samples.items() if shade == reference)
        period = b - a
        aligned = [t for t in samples if t not in (a, b) and (t - a) % period == 0]
        _require(len(aligned) <= 1, "multiple on-phase terminals")
        terminal = aligned[0] if aligned else None
        _require(terminal is None or terminal < a or terminal > b,
                 "terminal must lie outside the reference pair")
        noise = tuple(t for t in samples if t not in (a, b, terminal))
        route.update(a=a, b=b, period=period, reference=reference,
                     terminal=terminal, samples=samples, noise=noise,
                     paint=reference if terminal is None else samples[terminal])
        accounted.update(route["path"][t - 1] for t in samples)
    _require(accounted == set(beads), "singleton outside all virtual routes")
    return {"grid": grid, "background": background, "sources": sources,
            "mirrors": mirrors, "beads": beads, "routes": routes}


def _positions(route):
    a, b, p, terminal = (route[k] for k in ("a", "b", "period", "terminal"))
    low, high = ((1, len(route["path"])) if terminal is None
                 else (min(a, terminal), max(b, terminal)))
    return [t for t in range(low, high + 1) if (t - a) % p == 0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete pulses using input geometry and input singleton samples only."""
    info = _analyze(input_grid)
    output = info["grid"].copy()
    for route in info["routes"]:
        for t in _positions(route):
            output[route["path"][t - 1]] = route["paint"]
    return output


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _clear_object(obj, shape, occupied):
    return all(_inside(p, shape) and p not in occupied
               and not any(q in occupied for q in _neighbors(p))
               for p in obj["cells"])


def _add_source(rng, shape, sources, mirrors, minimum):
    occupied = set().union(*(o["cells"] for o in sources + mirrors)) if sources + mirrors else set()
    for _ in range(90):
        dr, dc = _pick(rng, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
        # An edge bias allows longer routes without fixing a border template.
        r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        if rng.random() < .6:
            if rng.integers(2):
                r = int(rng.integers(1, max(2, shape[0] // 3)))
                if dr < 0:
                    r = shape[0] - 1 - r
            else:
                c = int(rng.integers(1, max(2, shape[1] // 3)))
                if dc < 0:
                    c = shape[1] - 1 - c
        source = _source((r, c), (dr, dc))
        if not _clear_object(source, shape, occupied):
            continue
        try:
            routes = _trace(shape, sources + [source], mirrors)
        except ValueError:
            continue
        if len(routes[-1]["path"]) >= minimum:
            return sources + [source], routes
    return None


def _add_mirror(rng, shape, sources, mirrors, first_axis=None):
    before = _trace(shape, sources, mirrors)[0]
    last = before["contacts"][-1][0] if before["contacts"] else 0
    occupied = set().union(*(o["cells"] for o in sources + mirrors))
    # First-flight length is broad; teaching filters subsequently choose
    # reference placements that actually exhibit the desired distinctions.
    candidates = [(t, axis, length)
                  for t in range(last + (7 if not mirrors else 3), len(before["path"]) + 1)
                  for axis in ((first_axis,) if first_axis is not None else (0, 1))
                  for length in (3, 5, 7, 9)]
    rng.shuffle(candidates)
    for t, axis, length in candidates[:160]:
        contact = before["path"][t - 1]
        dr, dc = before["outgoing"][t - 1]
        midpoint = (contact[0] + (dr if axis == 0 else 0),
                    contact[1] + (dc if axis == 1 else 0))
        mirror = _mirror(midpoint, axis, length)
        if not _clear_object(mirror, shape, occupied):
            continue
        try:
            after = _trace(shape, sources, mirrors + [mirror])[0]
        except ValueError:
            continue
        if (after["contacts"] == before["contacts"] + [(t, axis, len(mirrors))]
                and len(after["path"]) >= t + 6):
            return mirrors + [mirror]
    return None


def _usable(route, structural):
    return {t for t, p in enumerate(route["path"], 1)
            if not any(q in structural for q in _neighbors(p))}


def _signal_options(route, usable, kind):
    """Enumerate visible, informative reference/terminal arrangements."""
    length = len(route["path"])
    first = route["contacts"][0][0] if route["contacts"] else length + 1
    periods = (1,) if kind == "early_solid" else range(
        2 if kind in ("open_demo", "stop_demo", "cross") else 1,
        min(16, max(2, length // 3)) + 1)
    options = []
    for period in periods:
        for a in sorted(usable):
            b = a + period
            if b not in usable:
                continue
            phase = list(range(1 + (a - 1) % period, length + 1, period))
            if kind in ("open_demo", "stop_demo"):
                if not (a - period >= 1 and b < first and b + period <= length):
                    continue
                # The contact is silent but still bends the route. Continuing
                # an unchanged clock and restarting it give different cells.
                if (first - a) % period == 0 or first % period == 0:
                    continue
                if not any(t > first for t in phase):
                    continue
            if kind == "cross":
                pa, pb = route["path"][a - 1], route["path"][b - 1]
                if not any(a < t < b for t, _, _ in route["contacts"]):
                    continue
                if abs(pa[0] - pb[0]) == abs(pa[1] - pb[1]):
                    continue
            if kind == "open_demo":
                options.append((a, b, None))
                continue
            if kind not in ("stop_demo", "early_solid") and len(phase) >= 3:
                options.append((a, b, None))
            for terminal in phase:
                if terminal not in usable or a <= terminal <= b:
                    continue
                low, high = min(a, terminal), max(b, terminal)
                # Bounded examples must add at least one new pulse and have a
                # visible potential pulse outside their bounded interval.
                if sum(low <= t <= high for t in phase) < 4:
                    continue
                if not any(t < low or t > high for t in phase):
                    continue
                if kind == "stop_demo" and not (terminal > first and terminal > b):
                    continue
                if kind == "early_solid" and not terminal < a:
                    continue
                options.append((a, b, terminal))
    return options


def _make_pair(rng, kind, axis, palette_setting):
    low = 18 if kind in ("open", "stop", "test") else 11
    h, w = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    if rng.random() < .18:
        w = h
    shape = (h, w)
    added = _add_source(rng, shape, [], [], 12 if kind != "random" else 7)
    if added is None:
        return None
    sources, _ = added
    mirrors = []
    bounces = (int(rng.integers(1, 3)) if kind in ("open", "stop")
               else int(rng.integers(2, 5)) if kind == "test"
               else int(_pick(rng, (0, 0, 1, 2, 3, 4))))
    for index in range(bounces):
        extended = _add_mirror(rng, shape, sources, mirrors, axis if index == 0 else None)
        if extended is None:
            return None
        mirrors = extended
    count = (1 if kind == "open" else 2 if kind == "stop"
             else int(rng.integers(1, 4 if kind == "test" else 5)))
    for _ in range(count - 1):
        added = _add_source(rng, shape, sources, mirrors, 7)
        if added is None:
            if kind == "stop":
                return None
            break
        sources, _ = added
    routes = _trace(shape, sources, mirrors)
    if palette_setting is None:
        background = int(rng.integers(10))
        palette = [int(c) for c in rng.choice([c for c in range(10) if c != background],
                                             int(rng.integers(3, 10)), replace=False)]
    else:
        background, palette = palette_setting
    grid = np.full(shape, background, dtype=np.int8)
    reference = int(_pick(rng, palette))
    for obj in sources + mirrors:
        colors = ([c for c in palette if c != reference]
                  if obj in mirrors and kind in ("open", "stop") else palette)
        obj["color"] = int(_pick(rng, colors))
        for p in obj["cells"]:
            grid[p] = obj["color"]
    structural = set().union(*(o["cells"] for o in sources + mirrors))
    for index, route in enumerate(routes):
        signal_kind = ("open_demo" if kind == "open" else "stop_demo" if kind == "stop"
                       else "cross" if kind == "test" else "general") if index == 0 else (
                           "early_solid" if kind == "stop" else "general")
        options = _signal_options(route, _usable(route, structural), signal_kind)
        if not options:
            return None
        a, b, terminal = _pick(rng, options)
        shade = reference if index == 0 else int(_pick(rng, palette))
        other = [c for c in palette if c != shade]
        paint = shade if terminal is None else int(_pick(rng, other))
        samples = {a: shade, b: shade}
        if terminal is not None:
            samples[terminal] = paint
        # Off-phase singletons never count as mirrors. The stop demonstration
        # puts a nearer decoy before the true terminal whenever feasible.
        available = [t for t in _usable(route, structural)
                     if (t - a) % (b - a) != 0]
        decoy_colors = [c for c in palette if c not in (shade, paint)]
        rng.shuffle(available)
        rng.shuffle(decoy_colors)
        noise_count = (1 if signal_kind == "stop_demo" else
                       int(rng.integers(0, min(3, len(decoy_colors)) + 1)))
        if signal_kind == "stop_demo":
            nearer = [t for t in available if b < t < terminal]
            if not nearer:
                return None
            available = nearer + [t for t in available if t not in nearer]
        if noise_count > min(len(available), len(decoy_colors)):
            return None
        samples.update(dict(zip(available[:noise_count], decoy_colors[:noise_count])))
        for t, color in samples.items():
            p = route["path"][t - 1]
            if grid[p] != background or any(_inside(q, shape) and grid[q] == color
                                             for q in _neighbors(p)):
                return None
            grid[p] = color
        route.update(a=a, b=b, period=b - a, terminal=terminal, reference=shade,
                     paint=paint, samples=samples,
                     noise=tuple(t for t in samples if t not in (a, b, terminal)))
    output = grid.copy()
    for route in routes:
        for t in _positions(route):
            output[route["path"][t - 1]] = route["paint"]
    try:
        # Reconstruct from actual components: close approaches can change how
        # a colored object is perceived even when geometric planning was valid.
        info = _analyze(grid)
    except ValueError:
        return None
    if not np.array_equal(solve(grid), output) or np.array_equal(grid, output):
        return None
    return {"input": grid, "output": output}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return value


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    palette_setting = None
    if rng.random() < .25:
        bg = int(rng.integers(10))
        palette_setting = (bg, [int(c) for c in rng.choice(
            [c for c in range(10) if c != bg], int(rng.integers(3, 6)), replace=False)])
    puzzle, seen = {"train": [], "test": []}, set()
    kinds = [("train", "open", axis), ("train", "stop", 1 - axis)]
    kinds += [("train", "random", None)] * (num_train - 2)
    kinds += [("test", "test", None)] * num_test
    for split, kind, first_axis in kinds:
        for _ in range(200):
            pair = _make_pair(rng, kind, first_axis, palette_setting)
            if pair is not None and _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("bounded folded-trail construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    routes = info["routes"]
    open_routes = [r for r in routes if r["terminal"] is None]
    stopped = [r for r in routes if r["terminal"] is not None]
    return {
        "shape": info["grid"].shape, "background": info["background"],
        "source_count": len(routes), "mirror_count": len(info["mirrors"]),
        "axes": {axis for r in routes for _, axis, _ in r["contacts"]},
        "directions": {s["direction"] for s in info["sources"]},
        "periods": tuple(r["period"] for r in routes),
        "noise": sum(len(r["noise"]) for r in routes),
        "open": bool(open_routes), "bounded": bool(stopped),
        "both_extensions": any(r["a"] - r["period"] >= 1
                               and r["b"] + r["period"] <= len(r["path"])
                               for r in open_routes),
        "stopping": any(any(t < min(r["a"], r["terminal"])
                            or t > max(r["b"], r["terminal"])
                            for t in range(1, len(r["path"]) + 1)
                            if (t - r["a"]) % r["period"] == 0) for r in stopped),
        "early_terminal": any(r["terminal"] < r["a"] for r in stopped),
        "late_terminal": any(r["terminal"] > r["b"] for r in stopped),
        "cross_references": any(any(r["a"] < t < r["b"] for t, _, _ in r["contacts"])
                                and abs(r["path"][r["a"] - 1][0] - r["path"][r["b"] - 1][0])
                                != abs(r["path"][r["a"] - 1][1] - r["path"][r["b"] - 1][1])
                                for r in routes),
        "silent_bend": any((t - r["a"]) % r["period"] != 0
                           and any(x > t for x in _positions(r))
                           for r in routes for t, _, _ in r["contacts"]),
    }


def _shortcut(info, mode):
    """Explicit ablations, for demonstration coverage rather than solving."""
    output = info["grid"].copy()
    for source, route in zip(info["sources"], info["routes"]):
        if mode == "dense":
            times = range(1, len(route["path"]) + 1)
        elif mode == "unbounded":
            times = [t for t in range(1, len(route["path"]) + 1)
                     if (t - route["a"]) % route["period"] == 0]
        elif mode == "reset":
            low, high = ((1, len(route["path"])) if route["terminal"] is None
                         else (min(route["a"], route["terminal"]),
                               max(route["b"], route["terminal"])))
            times = []
            for t in range(low, high + 1):
                last = max([0] + [x for x, _, _ in route["contacts"] if x < t])
                if (t - last - route["a"]) % route["period"] == 0:
                    times.append(t)
        else:
            times = _positions(route)
        for t in times:
            if mode == "forward" and t < route["a"]:
                continue
            p = route["path"][t - 1]
            if mode == "straight":
                p = tuple(source["elbow"][i] + t * source["direction"][i] for i in (0, 1))
                if not _inside(p, output.shape) or p in set().union(
                        *(o["cells"] for o in info["sources"] + info["mirrors"])):
                    continue
            output[p] = route["paint"]
    return output


def validate_generated_puzzle(puzzle):
    """Validate schema, route grammar, semantics and training witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _array(pair["input"]), _array(pair["output"])
            _require(inp.shape == out.shape, "output shape changes")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(solve(inp), out), "wrong pulse output")
            _require(not np.array_equal(inp, out), "identity pair")
            structural = set().union(*(o["cells"] for o in info["sources"] + info["mirrors"]))
            _require(all(inp[p] == out[p] for p in structural), "modified L or bar")
            _require(all(inp[r["path"][t - 1]] == out[r["path"][t - 1]]
                         for r in info["routes"] for t in r["noise"]), "modified off-phase dot")
            _require(set(np.unique(out)) <= set(np.unique(inp)), "invented color")
            f = _features(info)
            if split == "train":
                training.append((info, f, out))
            else:
                _require(f["cross_references"], "test lacks cross-fold reference shift")
    fs = [f for _, f, _ in training]
    for name in ("open", "bounded", "both_extensions", "stopping", "early_terminal",
                 "late_terminal", "silent_bend"):
        _require(any(f[name] for f in fs), "missing training witness: " + name)
    _require(set().union(*(f["axes"] for f in fs)) == {0, 1}, "both mirror normals required")
    _require(any(1 in f["periods"] for f in fs) and any(max(f["periods"]) > 1 for f in fs),
             "solid and dotted periods must be taught")
    _require(any(f["noise"] for f in fs), "off-phase preservation must be taught")
    _require(len({f["source_count"] for f in fs}) >= 2, "source-count variation")
    for mode in ("straight", "dense", "reset", "forward", "unbounded"):
        _require(any(not np.array_equal(_shortcut(info, mode), out)
                     for info, _, out in training), "parent ablation does not change answer")


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded 42-host-seed sweep, direct rule probes and API count checks."""
    puzzles = [generate_puzzle(seed) for seed in _HOST_SEEDS]
    features, structures, lengths, mirrors = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                features.append(_features(info))
                lengths.update(len(r["path"]) for r in info["routes"])
                mirrors.update(len(m["cells"]) for m in info["mirrors"])
                mapping = {info["background"]: 0}
                structure = tuple(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, structure))
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    _require(len(structures) >= 150, "structural diversity")
    _require(len({f["shape"] for f in features}) >= 50, "shape diversity")
    _require({np.sign(f["shape"][0] - f["shape"][1]) for f in features} == {-1, 0, 1},
             "aspect-ratio diversity")
    _require({f["background"] for f in features} == set(range(10)), "background diversity")
    _require({1, 2, 3} <= {f["source_count"] for f in features}, "source-count diversity")
    _require({1, 2, 3, 4, 5, 6} <= {p for f in features for p in f["periods"]}, "period diversity")
    _require({3, 5, 7, 9} <= mirrors and len(lengths) >= 20, "route/bar diversity")
    _require(set().union(*(f["directions"] for f in features)) ==
             {(1, 1), (1, -1), (-1, 1), (-1, -1)}, "launch rotations")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "determinism")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(314159, train, test)
            _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _rule_probes()


def _rule_probes():
    # One manually enumerated folded route: contact at t=5, no clock reset.
    inp = np.zeros((12, 13), dtype=np.int8)
    inp[0, 1] = inp[1, 0] = inp[1, 1] = 3
    inp[7, 5:8] = 6
    path = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            (5, 7), (4, 8), (3, 9), (2, 10), (1, 11), (0, 12)]
    inp[path[1]] = inp[path[3]] = 2  # t=2,4; period two, even phase.
    inp[path[6]] = 9                # t=7 is transparent, off phase.
    expected = inp.copy()
    for t in (2, 4, 6, 8, 10):
        expected[path[t - 1]] = 2
    _require(np.array_equal(solve(inp), expected), "folded even-phase probe")
    terminal = inp.copy()
    terminal[path[7]] = 5           # t=8 supplies color and upper bound.
    expected = terminal.copy()
    for t in (2, 4, 6, 8):
        expected[path[t - 1]] = 5
    _require(np.array_equal(solve(terminal), expected), "terminal span probe")
    early = inp.copy()
    early[path[1]] = 5             # terminal t=2; references move to t=6,8.
    early[path[3]] = 0
    early[path[5]] = early[path[7]] = 2
    expected = early.copy()
    for t in (2, 4, 6, 8):
        expected[path[t - 1]] = 5
    _require(np.array_equal(solve(early), expected), "early-terminal probe")
    cross = inp.copy()
    cross[path[3]] = 0
    cross[path[5]] = 2              # references t=2,6 straddle the bend.
    expected_cross = cross.copy()
    expected_cross[path[9]] = 2     # global period four continues to t=10.
    _require(np.array_equal(solve(cross), expected_cross), "cross-fold references probe")
    for grid, expected_grid in ((inp, solve(inp)), (terminal, solve(terminal)), (early, solve(early))):
        for turns in range(4):
            _require(np.array_equal(solve(9 - np.rot90(grid, turns)),
                                    9 - np.rot90(expected_grid, turns)), "rotation/palette equivariance")
