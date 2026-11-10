"""Braided tuning combs: sort directed, cable-local span templates.

No files, parent modules, generation metadata, or demonstrations are read by
solve. Generation paints explicit paths; validation recovers a graph from cells.
"""
from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND
AUDIT_SEEDS = (0, 232521771, 661185339, 10040536, 30037972, 951844751,
               478715635, 331936137, 156509870, 826522206, 582627698,
               299348008, 823725065, 953672990, 891765667, 756635058,
               11346059, 930732328, 200204329, 318665984, 185037948,
               346684446, 332288096, 521312865, 85425874, 670716257,
               84527702, 132614701, 287580190, 963568644, 68086037,
               226923601, 555021277, 180323362, 51895543, 85516558,
               516466284, 674356114, 351093837, 332587562, 769222918, 856441142)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= x <= 30 for x in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _add(p, d, amount=1):
    return p[0] + amount * d[0], p[1] + amount * d[1]


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, component = [p], {p}
        while stack:
            p = stack.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _terminals(g, bg):
    """Find the unique stencil color and its common connector color."""
    windows = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    possibilities = []
    for marker in sorted(set(map(int, np.unique(g))) - {bg}):
        found, covered, footprints = [], set(), set()
        for kind, pattern in (("source", DIAMOND), ("sink", CHECKER)):
            stencil = np.where(pattern, marker, bg)
            hits = np.argwhere(np.all(windows == stencil, axis=(-2, -1)))
            for r, c in hits:
                r, c = int(r), int(c)
                center = r + 1, c + 1
                footprint = {(r + a, c + b) for a in range(3) for b in range(3)}
                # One connector outside the 3x3 footprint; the other three
                # compass ports are background. Stencils do not touch.
                ports = [_add(center, d, 2) for d in DIRS]
                occupied = [p for p in ports if _inside(g, p) and g[p] != bg]
                if len(occupied) != 1 or footprints & footprint:
                    break
                port = occupied[0]
                covered.update((r + a, c + b) for a, b in np.argwhere(pattern))
                footprints.update(footprint)
                found.append(dict(kind=kind, center=center, port=port,
                                  carrier=int(g[port]), footprint=footprint))
            else:
                continue
            break
        else:
            sources = [t for t in found if t["kind"] == "source"]
            sinks = [t for t in found if t["kind"] == "sink"]
            if (2 <= len(sources) == len(sinks) <= 3
                    and covered == set(map(tuple, np.argwhere(g == marker)))
                    and len({t["carrier"] for t in found}) == 1
                    and found[0]["carrier"] not in (bg, marker)):
                possibilities.append((marker, found[0]["carrier"], found))
    assert len(possibilities) == 1, "terminal grammar must have one interpretation"
    return possibilities[0]


def _span_at(station, base, direction):
    normal = (-direction[1], direction[0])  # positive is the cable's left
    r0, r1, c0, c1 = station["box"]
    values = [(r - base[0]) * normal[0] + (c - base[1]) * normal[1]
              for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))]
    return int(min(values)), int(max(values))


def _station_cells(station, span):
    d, p = station["direction"], station["base"]
    n = -d[1], d[0]
    return {_add(_add(p, d, t), n, q)
            for t in range(station["width"])
            for q in range(span[0], span[1] + 1)}


def _parse(input_grid):
    g = _grid(input_grid)
    assert min(g.shape) >= 3
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    marker, carrier, terminals = _terminals(g, bg)
    terminal_cells = set().union(*(t["footprint"] for t in terminals))
    wire = set(map(tuple, np.argwhere(g == carrier)))
    assert not wire & terminal_cells
    stations, crossing = [], set()
    cross_color = None
    for color in sorted(set(map(int, np.unique(g))) - {bg, marker, carrier}):
        for cells in _components(map(tuple, np.argwhere(g == color))):
            p = min(cells)
            if len(cells) == 1 and all(_add(p, d) in wire for d in DIRS):
                assert cross_color in (None, color)
                cross_color = color
                crossing.add(p)
                continue
            r0, c0 = map(int, np.min(list(cells), axis=0))
            r1, c1 = map(int, np.max(list(cells), axis=0))
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "nonsolid bar"
            contacts = []
            for r in range(r0, r1 + 1):
                for c in (c0 - 1, c1 + 1):
                    if (r, c) in wire:
                        contacts.append((r, c))
            for c in range(c0, c1 + 1):
                for r in (r0 - 1, r1 + 1):
                    if (r, c) in wire:
                        contacts.append((r, c))
            assert len(contacts) == 2, "bar needs two exposed, aligned cable ports"
            a, b = sorted(contacts)
            if a[0] == b[0] and a[1] == c0 - 1 and b[1] == c1 + 1:
                line = [(a[0], c) for c in range(c0, c1 + 1)]
            else:
                assert a[1] == b[1] and a[0] == r0 - 1 and b[0] == r1 + 1
                line = [(r, a[1]) for r in range(r0, r1 + 1)]
            stations.append(dict(box=(r0, r1, c0, c1), color=color,
                                 line=line, cells=cells, width=len(line)))
    assert all(s["color"] != cross_color for s in stations)
    visible = wire | crossing
    station_at = {}
    for i, station in enumerate(stations):
        for p in station["line"]:
            assert p not in visible and p not in station_at
            station_at[p] = i
            visible.add(p)
    endpoint = {t["port"]: t for t in terminals}
    assert len(endpoint) == len(terminals)
    graph = {p: [_add(p, d) for d in DIRS if _add(p, d) in visible] for p in visible}
    for p, neighbors in graph.items():
        assert len(neighbors) == (1 if p in endpoint else 4 if p in crossing else 2)
    routes, all_seen, used_stations, sinks = [], set(), set(), set()
    for terminal in sorted((t for t in terminals if t["kind"] == "source"),
                           key=lambda t: t["center"]):
        p, previous = terminal["port"], None
        path, order, states = [], [], set()
        while True:
            assert (previous, p) not in states, "cable cycle"
            states.add((previous, p))
            path.append(p)
            all_seen.add(p)
            if p in station_at:
                index = station_at[p]
                if not order or order[-1] != index:
                    assert index not in used_stations
                    used_stations.add(index)
                    order.append(index)
                    d = p[0] - previous[0], p[1] - previous[1]
                    s = stations[index]
                    s.update(base=p, direction=d, span=_span_at(s, p, d))
                    assert s["span"][0] <= 0 <= s["span"][1]
            if previous is not None and p in endpoint:
                assert endpoint[p]["kind"] == "sink" and p not in sinks
                sinks.add(p)
                break
            if p in crossing:
                next_p = 2 * p[0] - previous[0], 2 * p[1] - previous[1]
                assert next_p in graph[p]
            else:
                candidates = [q for q in graph[p] if q != previous]
                assert len(candidates) == 1
                next_p = candidates[0]
            previous, p = p, next_p
        assert len(order) >= 2, "every cable needs at least two bars"
        templates = {}
        for index in order:
            lo, hi = stations[index]["span"]
            assert templates.setdefault(hi - lo + 1, (lo, hi)) == (lo, hi), "ambiguous tie"
        routes.append(dict(path=path, stations=[stations[i] for i in order],
                           source=terminal["center"], sink=endpoint[p]["center"]))
    assert all_seen == visible and len(used_stations) == len(stations)
    assert len(sinks) * 2 == len(terminals)
    # Full span transfer is always legal, including for templates wider than
    # the receiving station's input span. Envelopes cannot touch other bars.
    reserved = set()
    all_bar_cells = set().union(*(s["cells"] for s in stations))
    for route in routes:
        span = (min(s["span"][0] for s in route["stations"]),
                max(s["span"][1] for s in route["stations"]))
        for s in route["stations"]:
            envelope = _station_cells(s, span)
            assert all(_inside(g, p) for p in envelope), "template would leave canvas"
            assert not envelope & reserved, "template envelopes overlap"
            assert all(p in s["cells"] or int(g[p]) == bg for p in envelope)
            # Same-color bars must remain separate even after reassignment.
            assert not any(_add(p, d) in reserved for p in envelope for d in DIRS)
            reserved.update(envelope)
    return dict(grid=g, background=bg, marker=marker, carrier=carrier,
                cross_color=cross_color, crossings=crossing, stations=stations,
                routes=routes, bar_cells=all_bar_cells)


def _render(info, assignments):
    out = info["grid"].copy()
    for p in info["bar_cells"]:
        out[p] = info["background"]
    for station, span in assignments:
        for p in _station_cells(station, span):
            assert _inside(out, p) and out[p] == info["background"]
            out[p] = station["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Sort each cable's spans from diamond to checker in its moving frame."""
    info = _parse(input_grid)
    assignments = []
    for route in info["routes"]:
        bars = route["stations"]
        spans = sorted((s["span"] for s in bars), key=lambda v: v[1] - v[0])
        assignments.extend(zip(bars, spans))
    return _render(info, assignments)


def _ablation(info, mode):
    assignments = []
    if mode == "global":
        bars = sorted(info["stations"], key=lambda s: s["base"])
        spans = sorted((s["span"] for s in bars), key=lambda v: v[1] - v[0])
        assignments.extend(zip(bars, spans))
    else:
        for route in info["routes"]:
            bars = route["stations"]
            if mode == "raster":
                bars = sorted(bars, key=lambda s: s["base"])
            if mode == "reverse":
                spans = [s["span"] for s in bars[::-1]]
            else:
                key = (lambda s: s["width"] * (s["span"][1] - s["span"][0] + 1)) if mode == "area" else (
                    lambda s: s["span"][1] - s["span"][0])
                spans = [s["span"] for s in sorted(bars, key=key)]
                if mode == "frame":
                    ranked = sorted(bars, key=key)
                    converted = []
                    for destination, source in zip(bars, ranked):
                        a, b = source["direction"], destination["direction"]
                        sign_a = a[0] if a[0] else -a[1]
                        sign_b = b[0] if b[0] else -b[1]
                        lo, hi = source["span"]
                        converted.append((lo, hi) if sign_a == sign_b else (-hi, -lo))
                    spans = converted
                if mode == "center":
                    spans = [(-(hi - lo) // 2, (hi - lo) // 2) for lo, hi in spans]
            assignments.extend(zip(bars, spans))
    # Ablations can violate envelope constraints. A collision or out-of-bounds
    # attempt is itself a failed alternative, represented by None.
    try:
        return _render(info, assignments)
    except AssertionError:
        return None


def _pair_features(pair, info=None):
    info = _parse(pair["input"]) if info is None else info
    out = np.asarray(pair["output"])
    failures = {mode: not np.array_equal(_ablation(info, mode), out)
                for mode in ("global", "raster", "area", "center", "frame", "reverse")}
    cross_axis = False
    for route in info["routes"]:
        bars = route["stations"]
        ranked = sorted(bars, key=lambda s: s["span"][1] - s["span"][0])
        for a, b in zip(bars, ranked):
            if (a["direction"][0] == 0) != (b["direction"][0] == 0) and b["span"][0] != -b["span"][1]:
                cross_axis = True
    return dict(shape=tuple(info["grid"].shape), cables=len(info["routes"]),
                counts=tuple(len(r["stations"]) for r in info["routes"]),
                crossings=len(info["crossings"]),
                widths=tuple(s["width"] for s in info["stations"]),
                lengths=tuple(s["span"][1] - s["span"][0] + 1 for s in info["stations"]),
                directions=tuple(sorted({tuple(map(int, s["direction"])) for s in info["stations"]})),
                roles=(info["background"], info["marker"], info["carrier"], info["cross_color"]),
                ties=any(len({s["span"][1] - s["span"][0] for s in r["stations"]}) < len(r["stations"])
                         for r in info["routes"]), cross_axis=cross_axis,
                changed=not np.array_equal(info["grid"], out), **failures)


def _random_path(rng, h, w):
    boundary = [(r, c) for r in range(h) for c in range(w)
                if r in (0, h - 1) or c in (0, w - 1)]
    path = [boundary[int(rng.integers(len(boundary)))]]
    for _ in range(11):
        choices = [_add(path[-1], d) for d in DIRS]
        choices = [p for p in choices if 0 <= p[0] < h and 0 <= p[1] < w and p not in path]
        if not choices:
            return None
        path.append(choices[int(rng.integers(len(choices)))])
        if len(path) >= 4 and path[-1] in boundary and rng.random() < .55:
            return tuple(path)
    return None


def _compatible(a, b):
    for p in set(a) & set(b):
        i, j = a.index(p), b.index(p)
        if i in (0, len(a) - 1) or j in (0, len(b) - 1):
            return False
        da = a[i + 1][0] - a[i - 1][0], a[i + 1][1] - a[i - 1][1]
        db = b[j + 1][0] - b[j - 1][0], b[j + 1][1] - b[j - 1][1]
        if 2 not in map(abs, da) or 2 not in map(abs, db) or da[0] * db[0] + da[1] * db[1]:
            return False
    return True


FALLBACK_TWO = (((0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2), (3, 3)),
                ((3, 0), (2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (0, 3)))
FALLBACK_THREE = (((0, 1), (1, 1), (2, 1), (3, 1)),
                  ((3, 0), (2, 0), (1, 0), (1, 1), (1, 2), (0, 2)),
                  ((3, 3), (2, 3), (2, 2), (3, 2)))


def _geometry(rng, pools, shape, n, min_edges, crossing):
    if shape not in pools:
        pool = list(dict.fromkeys(p for _ in range(320)
                                 if (p := _random_path(rng, *shape)) is not None))
        if shape == (4, 4):
            pool.extend(FALLBACK_TWO + FALLBACK_THREE)
        pools[shape] = pool
    pool = [p for p in pools[shape] if len(p) > min_edges]
    for _ in range(100):
        first = pool[int(rng.integers(len(pool)))]
        group = [first]
        for i in rng.permutation(len(pool)):
            path = pool[int(i)]
            if all(_compatible(path, q) for q in group):
                group.append(path)
            if len(group) == n:
                break
        if len(group) != n:
            continue
        if crossing and not any(set(a) & set(b) for i, a in enumerate(group) for b in group[i + 1:]):
            continue
        if crossing and not all(len({(b[0] - a[0], b[1] - a[1]) for a, b in zip(p, p[1:])}) >= 2 for p in group):
            continue
        return group, shape
    return (FALLBACK_TWO if n == 2 else FALLBACK_THREE), (4, 4)


def _template(rng, length):
    lo = int(rng.integers(max(-2, 1 - length), min(0, 3 - length) + 1))
    return lo, lo + length - 1


def _make_pair(rng, pools, n, count_range, witness=False, compact=False, fallback=False):
    shape = (3, int(rng.integers(3, 5))) if compact and n == 2 else (4, 4)
    if fallback:
        paths, shape = FALLBACK_TWO, (4, 4)
    else:
        paths, shape = _geometry(rng, pools, shape, n, count_range[0], witness)
    pitch = int(rng.choice([7, 8] if witness or count_range[0] >= 4 else [6, 7, 8]))
    # Four independently chosen exterior margins fit the 30-cell bound.
    margins = []
    for size in shape:
        budget = 30 - (size - 1) * pitch - 1
        before = int(rng.integers(2, min(4, budget - 2) + 1))
        after = int(rng.integers(2, min(4, budget - before) + 1))
        margins.extend([before, after])
    top, bottom, left, right = margins
    h, w = top + (shape[0] - 1) * pitch + 1 + bottom, left + (shape[1] - 1) * pitch + 1 + right
    palette = list(map(int, rng.permutation(10)))
    bg, marker, carrier, cross = palette[:4]
    accent_pool = palette[4:4 + int(rng.integers(2, 7))]
    inp = np.full((h, w), bg, dtype=np.int8)
    pixel_paths, visits = [], Counter()
    for path in paths:
        points = [(top + r * pitch, left + c * pitch) for r, c in path]
        pixels = [points[0]]
        for a, b in zip(points, points[1:]):
            d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
            pixels.extend(_add(a, d, t) for t in range(1, pitch + 1))
        pixel_paths.append((points, pixels))
        visits.update(pixels[2:-2])
    for p, count in visits.items():
        inp[p] = cross if count == 2 else carrier
    assignments, raw_stations = [], []
    for cable, (points, pixels) in enumerate(pixel_paths):
        for center, pattern in ((points[0], DIAMOND), (points[-1], CHECKER)):
            r, c = center
            inp[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, marker, bg)
        edges = len(points) - 1
        k = 3 if witness and cable == 0 else int(rng.integers(count_range[0], min(count_range[1], edges) + 1))
        if witness and cable == 0:
            horizontal = [i for i in range(edges) if points[i][0] == points[i + 1][0]]
            vertical = [i for i in range(edges) if points[i][1] == points[i + 1][1]]
            if not horizontal or not vertical:
                raise AssertionError("witness lacks a turn")
            chosen = {int(rng.choice(horizontal)), int(rng.choice(vertical))}
            remaining = [i for i in range(edges) if i not in chosen]
            chosen.add(int(rng.choice(remaining)))
            selected = sorted(chosen)
            lengths = [4, 1, 3]
            widths = [1, int(rng.integers(1, 3)), 2]
            templates = {1: (0, 0), 3: (0, 2), 4: (-2, 1)}
        else:
            selected = sorted(map(int, rng.choice(edges, k, replace=False)))
            lengths = list(map(int, rng.integers(1, 6, size=k)))
            if len(set(lengths)) == 1:
                lengths[-1] = 1 if lengths[0] != 1 else 5
            templates = {length: _template(rng, length) for length in set(lengths)}
            widths = list(map(int, rng.integers(1, min(3, pitch - 5) + 1, size=k)))
        cable_stations = []
        for i, length, width in zip(selected, lengths, widths):
            a, b = points[i], points[i + 1]
            d = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
            base = _add(a, d, (pitch - width + 1) // 2)
            station = dict(base=base, direction=d, width=width,
                           span=templates[length], color=int(rng.choice(accent_pool)))
            for p in _station_cells(station, station["span"]):
                inp[p] = station["color"]
            cable_stations.append(station)
        raw_stations.extend(cable_stations)
        ranked = sorted((s["span"] for s in cable_stations), key=lambda v: v[1] - v[0])
        assignments.extend(zip(cable_stations, ranked))
    out = inp.copy()
    for s in raw_stations:
        for p in _station_cells(s, s["span"]):
            out[p] = bg
    for s, span in assignments:
        for p in _station_cells(s, span):
            out[p] = s["color"]
    # Rotations/reflections move the whole rule, including its handed frame.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
    if reflect:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["cables", "stations"]))
    puzzle, seen, pools = {"train": [], "test": []}, set(), {}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = split == "train" and index == 0
            n = 3 if split == "test" and shift == "cables" else 2
            count_range = (4, 5) if split == "test" and shift == "stations" else (2, 3 if shift == "stations" else 4)
            if witness:
                count_range = (3, 3)
            for attempt in range(80):
                pair = _make_pair(rng, pools, n, count_range, witness,
                                  compact=split == "train" and index % 2 == 1,
                                  fallback=witness and attempt >= 64)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    continue
                info = _parse(pair["input"])
                features = _pair_features(pair, info)
                if witness and not (features["cross_axis"] and features["crossings"]
                                    and all(features[m] for m in ("global", "raster", "area", "center", "frame", "reverse"))):
                    continue
                break
            else:
                raise RuntimeError("bounded construction exhausted")
            puzzle[split].append(pair)
            seen.add(key)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    a, b = _parse(pair["input"]), _parse(pair["output"])
    assert a["grid"].shape == b["grid"].shape
    for key in ("background", "marker", "carrier", "cross_color", "crossings"):
        assert a[key] == b[key]
    changed_zone = a["bar_cells"] | b["bar_cells"]
    assert all(a["grid"][r, c] == b["grid"][r, c]
               for r in range(a["grid"].shape[0]) for c in range(a["grid"].shape[1])
               if (r, c) not in changed_zone), "non-bar field changed"
    assert len(a["routes"]) == len(b["routes"])
    for before, after in zip(a["routes"], b["routes"]):
        assert before["path"] == after["path"]
        assert before["source"] == after["source"] and before["sink"] == after["sink"]
        assert [(s["base"], s["direction"], s["width"], s["color"]) for s in before["stations"]] == [
            (s["base"], s["direction"], s["width"], s["color"]) for s in after["stations"]]
        spans = [s["span"] for s in after["stations"]]
        lengths = [hi - lo + 1 for lo, hi in spans]
        assert lengths == sorted(lengths), "cable lengths not ordered"
        assert Counter(spans) == Counter(s["span"] for s in before["stations"]), "span template lost"
    return a


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, features = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = info["grid"].shape, info["grid"].tobytes()
            assert key not in seen
            seen.add(key)
            f = _pair_features(pair, info)
            features[split].append(f)
            if split == "train":
                evidence.update(m for m in ("global", "raster", "area", "center", "frame", "reverse") if f[m])
    assert evidence == {"global", "raster", "area", "center", "frame", "reverse"}
    assert any(f["cross_axis"] and f["crossings"] for f in features["train"])
    assert any(f["changed"] for f in features["train"])
    # One visible complexity axis is held out; neither convention is hidden.
    train, test = features["train"], features["test"]
    assert (max(f["cables"] for f in train) < min(f["cables"] for f in test)
            or max(max(f["counts"]) for f in train) < min(min(f["counts"]) for f in test))
    return True


def _puzzle_digest(puzzle):
    encoded = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Bounded streaming 42-seed audit, plus mutation and count checks."""
    shapes, roles, counts, directions, widths, lengths, digests = set(), set(), set(), set(), set(), set(), set()
    crossing_counts, ties, cross_axis, total = set(), set(), 0, 0
    for seed in AUDIT_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = pair["input"].copy()
                assert np.array_equal(solve(pair["input"]), pair["output"])
                assert np.array_equal(before, pair["input"]), "solve mutates input"
                f = _pair_features(pair)
                shapes.add(f["shape"]); roles.add(f["roles"]); counts.update(f["counts"])
                directions.update(f["directions"]); widths.update(f["widths"]); lengths.update(f["lengths"])
                crossing_counts.add(f["crossings"]); ties.add(f["ties"])
                cross_axis += int(f["cross_axis"]); total += 1
    assert len(digests) == 42 and len(shapes) >= 20 and len(roles) >= 100
    assert counts >= {2, 3, 4, 5} and widths >= {1, 2, 3} and lengths == {1, 2, 3, 4, 5}
    assert directions == set(DIRS) and ties == {False, True} and 0 in crossing_counts and max(crossing_counts) >= 2
    assert cross_axis >= 42
    first = generate_puzzle(0)
    assert _puzzle_digest(first) == _puzzle_digest(generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Relational validation rejects answers that ignore inherited mechanisms.
    for mode in ("global", "raster", "area", "center", "frame", "reverse"):
        for pair in first["train"]:
            info = _parse(pair["input"])
            bad = _ablation(info, mode)
            if bad is not None and not np.array_equal(bad, pair["output"]):
                try:
                    _check_pair({"input": pair["input"], "output": bad})
                except AssertionError:
                    break
                raise AssertionError(f"accepted {mode} mutation")
        else:
            raise AssertionError(f"missing concrete {mode} mutation")
    pair = first["train"][0]
    corrupt = pair["output"].copy()
    corrupt[0, 0] = (int(corrupt[0, 0]) + 1) % 10
    try:
        _check_pair({"input": pair["input"], "output": corrupt})
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted routing/background corruption")
    return dict(seeds=42, pairs=total, distinct_puzzles=len(digests), shapes=len(shapes),
                role_assignments=len(roles), station_counts=sorted(counts), widths=sorted(widths),
                lengths=sorted(lengths), crossing_counts=sorted(crossing_counts),
                cross_axis_pairs=cross_axis, note="Authored invariants, not independent semantic approval")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
