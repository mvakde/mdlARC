"""Shadow docks: rank-matched signals project stops for filling vehicles.

Only NumPy and the standard library are used. Generation has no file I/O;
all retries, diagnostics, and the 42-seed self-test are bounded and in memory.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ("U", "D", "L", "R")
_DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_TURNS = {"U": 0, "D": 2, "L": 3, "R": 1}


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _need(np.all((0 <= g) & (g <= 9)), "ARC colors required")
    return g


def _key(g):
    return tuple(g.shape), np.asarray(g, dtype=np.uint8).tobytes()


def _apart(a, b):
    """Half-open boxes with at least one intervening background cell."""
    t, l, z, r = a
    u, c, v, d = b
    return z < u or v < t or r < c or d < l


def _square_box(o, size):
    return o["top"], o["left"], o["top"] + size, o["left"] + size


def _nose(v, size):
    t, l = v["top"], v["left"]
    d = v["direction"]
    if d == "U":
        return (t - 1, l, t, l + size)
    if d == "D":
        return (t + size, l, t + size + 1, l + size)
    if d == "L":
        return (t, l - 1, t + size, l)
    return (t, l + size, t + size, l + size + 1)


def _distance(v, signal, size):
    if v["direction"] == "U":
        return v["top"] - signal["top"] - size
    if v["direction"] == "D":
        return signal["top"] - v["top"] - size
    if v["direction"] == "L":
        return v["left"] - signal["left"] - size
    return signal["left"] - v["left"] - size


def _sweep(v, size, distance):
    dy, dx = _DELTA[v["direction"]]
    nose = _nose(v, size)
    t, l, b, r = _square_box(v, size)
    return (min(t, nose[0], t + dy * distance, nose[0] + dy * distance),
            min(l, nose[1], l + dx * distance, nose[1] + dx * distance),
            max(b, nose[2], b + dy * distance, nose[2] + dy * distance),
            max(r, nose[3], r + dx * distance, nose[3] + dx * distance))


def _block(size, frame, bg, advanced, direction):
    canonical = np.full((size, size), frame, dtype=np.int8)
    interior = np.full((size - 2, size - 2), bg, dtype=np.int8)
    interior.flat[:advanced] = frame
    canonical[1:-1, 1:-1] = interior
    return np.rot90(canonical, -_TURNS[direction]).copy()


def _paint_vehicle(g, v, size, bg, movement=0):
    dy, dx = _DELTA[v["direction"]]
    moved = dict(v, top=v["top"] + dy * movement, left=v["left"] + dx * movement)
    t, l = moved["top"], moved["left"]
    g[t:t + size, l:l + size] = _block(size, v["color"], bg,
                                                  v["advanced"] + movement, v["direction"])
    t, l, b, r = _nose(moved, size)
    g[t:b, l:r] = v["marker"]


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, points = [start], []
        while stack:
            y, x = stack.pop()
            points.append((int(y), int(x)))
            for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(points)
    return result


def _parse(input_grid):
    """Read roles through geometry; no seed, examples, or global palette."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _need(np.all(g[0] == bg) and np.all(g[-1] == bg)
          and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "background border")
    vehicles, signals, boxes, sizes = [], [], [], set()
    for component in _components(g != bg):
        ys, xs = zip(*component)
        t, l, b, r = min(ys), min(xs), max(ys) + 1, max(xs) + 1
        box, h, w = (t, l, b, r), b - t, r - l
        _need(all(_apart(box, other) for other in boxes), "objects need a background gap")
        boxes.append(box)
        if h == w:
            s = h
            tile = g[t:b, l:r]
            _need(len(component) == s * s and np.all(tile == tile[0, 0]), "solid signal")
            signals.append(dict(top=t, left=l, color=int(tile[0, 0])))
        else:
            candidates = []
            if h == w + 1:
                choices, s = ("U", "D"), w
            elif w == h + 1:
                choices, s = ("L", "R"), h
            else:
                raise AssertionError("square with a one-cell nose required")
            for direction in choices:
                vt = t + int(direction == "U")
                vl = l + int(direction == "L")
                v = dict(top=vt, left=vl, direction=direction)
                nt, nl, nb, nr = _nose(v, s)
                bar = g[nt:nb, nl:nr]
                tile = g[vt:vt + s, vl:vl + s]
                frame, marker = int(tile[0, 0]), int(bar[0, 0])
                if frame == bg or marker in (bg, frame) or not np.all(bar == marker):
                    continue
                canonical = np.rot90(tile, _TURNS[direction])
                rim = np.concatenate((canonical[0], canonical[-1], canonical[:, 0], canonical[:, -1]))
                interior = canonical[1:-1, 1:-1].ravel()
                a = int(np.count_nonzero(interior == frame))
                if (not np.all(rim == frame) or a >= (s - 2) ** 2
                        or not np.array_equal(interior, [frame] * a + [bg] * (len(interior) - a))):
                    continue
                v.update(color=frame, marker=marker, advanced=a, capacity=len(interior))
                candidates.append(v)
            _need(len(candidates) == 1, "ambiguous vehicle")
            vehicles.append(candidates[0])
        _need(3 <= s <= 6, "square size must be 3..6")
        sizes.add(s)
    _need(vehicles and signals and len(sizes) == 1, "both classes and one scale required")
    s = sizes.pop()
    for objects in (vehicles, signals):
        objects.sort(key=lambda o: o["top"])
        _need(len({o["top"] for o in objects}) == len(objects), "same-class top tie")
    sweeps = []
    for i, v in enumerate(vehicles):
        d = _distance(v, signals[i], s) if i < len(signals) else 0
        _need(i >= len(signals) or d > 0, "signal must be strictly ahead")
        v["distance"] = d
        box = _sweep(v, s, d)
        _need(1 <= box[0] < box[2] <= g.shape[0] - 1
              and 1 <= box[1] < box[3] <= g.shape[1] - 1, "sweep outside border")
        _need(all(_apart(box, _square_box(o, s)) for o in signals), "signal in travel corridor")
        _need(all(_apart(box, other) for other in sweeps), "interacting travel corridors")
        sweeps.append(box)
    return dict(background=bg, size=s, vehicles=vehicles, signals=signals)


def _movement(v):
    return min(v["capacity"] - v["advanced"], v["distance"])


def _transform(g, analysis):
    out = g.copy()
    bg, s = analysis["background"], analysis["size"]
    for v in analysis["vehicles"]:
        t, l, b, r = _sweep(v, s, 0)
        out[t:b, l:r] = bg
    for v in analysis["vehicles"]:
        _paint_vehicle(out, v, s, bg, _movement(v))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Determine every role, ordinal match, stop, and counter tick from input."""
    g = _grid(input_grid)
    return _transform(g, _parse(g))


def _alternative_moves(a, mode):
    """Explicit shortcut probes, reported as per-vehicle travel distances."""
    vs, ss, s = a["vehicles"], a["signals"], a["size"]
    mapping = {}
    if mode == "nearest":
        for i, v in enumerate(vs):
            candidates = [j for j, o in enumerate(ss) if _distance(v, o, s) > 0]
            if candidates:
                mapping[i] = min(candidates, key=lambda j: (abs(v["top"] - ss[j]["top"])
                                                            + abs(v["left"] - ss[j]["left"]), j))
    elif mode == "reverse":
        mapping = {i: len(ss) - 1 - i for i in range(min(len(vs), len(ss)))}
    elif mode in ("horizontal", "horizontal_reverse"):
        rev = mode == "horizontal_reverse"
        vi = sorted(range(len(vs)), key=lambda i: (vs[i]["left"], vs[i]["top"]), reverse=rev)
        si = sorted(range(len(ss)), key=lambda i: (ss[i]["left"], ss[i]["top"]), reverse=rev)
        mapping = dict(zip(vi, si))
    elif mode == "greedy":
        objects = [(v["top"], v["left"], 0, i) for i, v in enumerate(vs)]
        objects += [(o["top"], o["left"], 1, i) for i, o in enumerate(ss)]
        rows = [{}]
        for _, _, kind, index in sorted(objects):
            if kind in rows[-1]:
                rows.append({})
            rows[-1][kind] = index
        mapping = {row[0]: row[1] for row in rows if 0 in row and 1 in row}
    else:
        raise ValueError(mode)
    return tuple(min(v["capacity"] - v["advanced"], max(0, _distance(v, ss[mapping[i]], s)))
                 if i in mapping else 0 for i, v in enumerate(vs))


def _witnesses(a):
    actual = tuple(_movement(v) for v in a["vehicles"])
    matched = min(len(a["vehicles"]), len(a["signals"]))
    result = {m: actual != _alternative_moves(a, m)
              for m in ("reverse", "horizontal", "horizontal_reverse", "greedy")}
    result["nearest"] = actual[:matched] != _alternative_moves(a, "nearest")[:matched]
    result["fuel"] = any(0 < _movement(v) < v["distance"] for v in a["vehicles"])
    result["dock"] = any(0 < v["distance"] < v["capacity"] - v["advanced"] for v in a["vehicles"])
    width = a["size"] - 2
    result["scan"] = any(0 < v["advanced"] < v["advanced"] + _movement(v) < v["capacity"]
                         and v["advanced"] // width != (v["advanced"] + _movement(v)) // width
                         and (v["advanced"] + _movement(v)) % width != 0 for v in a["vehicles"])
    return result


def _positions(rng, s, nv, ns, h, w):
    """Free rectangle packing, with no shelves or fixed spatial slots."""
    yy, xx = np.indices((h - s + 1, w - s + 1))
    objects = []
    kinds = list(map(int, rng.permutation([0] * nv + [1] * ns)))
    for kind in kinds:
        legal = np.ones(yy.shape, dtype=bool)
        for o in objects:
            legal &= (np.abs(yy - o["top"]) > s) | (np.abs(xx - o["left"]) > s)
            if o["kind"] == kind:
                legal &= yy != o["top"]
        sites = np.argwhere(legal)
        if not len(sites):
            return None
        y, x = map(int, sites[int(rng.integers(len(sites)))])
        objects.append(dict(top=y, left=x, kind=kind))
    vs = sorted((o for o in objects if o["kind"] == 0), key=lambda o: o["top"])
    ss = sorted((o for o in objects if o["kind"] == 1), key=lambda o: o["top"])
    return vs, ss


def _scene(rng, size, counts, demo, core, palette, direction_pool=None):
    nv, ns = counts
    cap, cross = (size - 2) ** 2, size - 2
    regimes = list(map(str, rng.permutation(["dock", "fuel"]))) if core else []
    # Each attempt holds at most one <=30x30 scene. No attempt is stored.
    for attempt in range(2400):
        h, w = (28, 28) if attempt >= 600 else (int(rng.integers(21, 29)), int(rng.integers(21, 29)))
        positioned = _positions(rng, size, nv, ns, h, w)
        if positioned is None:
            continue
        vs, ss = positioned
        options = []
        for i, v in enumerate(vs):
            choices = []
            for direction in rng.permutation(direction_pool or _DIRS):
                q = dict(v, direction=str(direction))
                d = _distance(q, ss[i], size) if i < ns else 0
                if i < ns and d <= 0:
                    continue
                if core and i < 2:
                    if regimes[i] == "dock":
                        advances = [k for k in range(1, cap - d)
                                    if k // cross != (k + d) // cross and (k + d) % cross]
                        if not advances:
                            continue
                    elif d < 2:
                        continue
                box = _sweep(q, size, d)
                if not (0 <= box[0] < box[2] <= h and 0 <= box[1] < box[3] <= w):
                    continue
                others = ss + vs[:i] + vs[i + 1:]
                if not all(_apart(box, _square_box(o, size)) for o in others):
                    continue
                choices.append((q, d, box))
            if not choices:
                break
            options.append(choices)
        if len(options) != nv:
            continue

        def choose(index, selected):
            if index == nv:
                return selected
            for q, d, box in options[index]:
                if all(_apart(box, old[2]) for old in selected):
                    found = choose(index + 1, selected + [(q, d, box)])
                    if found is not None:
                        return found
            return None

        selected = choose(0, [])
        if selected is None:
            continue
        bg = int(rng.choice([c for c in palette if c != demo]))
        fg = [c for c in palette if c != bg]
        grid = np.full((h, w), bg, dtype=np.int8)
        for o in ss:
            o["color"] = int(rng.choice(fg))
            grid[o["top"]:o["top"] + size, o["left"]:o["left"] + size] = o["color"]
        if core == 2:
            ss[0]["color"] = demo
            o = ss[0]
            grid[o["top"]:o["top"] + size, o["left"]:o["left"] + size] = demo
        for i, (v, d, _) in enumerate(selected):
            frame = demo if core == 1 and i == 0 else int(rng.choice(fg))
            marker = int(rng.choice([c for c in fg if c != frame]))
            if core and i < 2 and regimes[i] == "dock":
                advances = [k for k in range(1, cap - d)
                            if k // cross != (k + d) // cross and (k + d) % cross]
                advanced = int(rng.choice(advances))
            elif core and i < 2:
                remaining = int(rng.integers(1, min(cap - 1, d - 1) + 1))
                advanced = cap - remaining
            else:
                mode = int(rng.integers(4))
                if mode == 0:
                    advanced = 0
                elif mode == 1 and 0 < d <= cap:
                    advanced = cap - d
                else:
                    advanced = int(rng.integers(cap))
            v.update(color=frame, marker=marker, advanced=advanced, capacity=cap, distance=d)
            _paint_vehicle(grid, v, size, bg)
        # Randomly crop blank exterior, then install an explicit background rim.
        ys, xs = np.nonzero(grid != bg)
        t = int(rng.integers(0, int(ys.min()) + 1))
        l = int(rng.integers(0, int(xs.min()) + 1))
        b = int(rng.integers(int(ys.max()) + 1, h + 1))
        r = int(rng.integers(int(xs.max()) + 1, w + 1))
        grid = np.pad(grid[t:b, l:r], 1, constant_values=bg)
        a = _parse(grid)
        if core == 1 and not _witnesses(a)["nearest"]:
            continue
        return dict(input=grid, output=_transform(grid, a))
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not minimum <= int(value) <= maximum:
        raise ValueError(name + " outside legal range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    nt = _count(num_train, 2, 10, "num_train")
    ne = _count(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    demo = int(rng.integers(10))
    first_counts = [(2, 3), (3, 2)]
    if rng.integers(2):
        first_counts.reverse()
    # Retry just the informative two-scene teaching set, never an unbounded loop.
    for _ in range(24):
        training = [_scene(rng, int(rng.choice((5, 6))), counts, demo, i + 1, palette)
                    for i, counts in enumerate(first_counts)]
        analyses = [_parse(p["input"]) for p in training]
        dirs = sorted({v["direction"] for a in analyses for v in a["vehicles"] if _movement(v)})
        if len(dirs) < 2:
            continue
        if all(any(_witnesses(a)[m] for a in analyses)
               for m in ("nearest", "reverse", "horizontal", "horizontal_reverse", "greedy")):
            break
    else:
        raise RuntimeError("bounded teaching-set construction exhausted")
    seen = {_key(p["input"]) for p in training}
    puzzle = dict(train=training, test=[])
    for split, count, start in (("train", nt - 2, 2), ("test", ne, 0)):
        for i in range(start, start + count):
            for _ in range(32):
                size = int(rng.integers(3, 7))
                if split == "test" and i == 0:
                    small = int(rng.integers(1, 3))
                    tail = int(rng.integers(2, 4))
                    counts = (small, small + tail)
                    if rng.integers(2):
                        counts = counts[::-1]
                    if size == 6:
                        counts = (1, 3) if counts[0] < counts[1] else (3, 1)
                else:
                    counts = tuple(map(int, rng.choice(((1, 1), (1, 2), (2, 1),
                                                       (2, 2), (2, 3), (3, 2), (3, 3)))))
                try:
                    pair = _scene(rng, size, counts, demo, 0, palette, dirs if split == "test" else None)
                except RuntimeError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-pair construction exhausted")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                dirs = sorted(set(dirs) | {v["direction"] for v in _parse(pair["input"])["vehicles"]
                                          if _movement(v)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _need(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
          and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape and _key(inp) not in seen, "shape or duplicate input")
            seen.add(_key(inp))
            a = _parse(inp)
            _need(np.array_equal(out, _transform(inp, a)), "incorrect answer")
            s, bg = a["size"], a["background"]
            # Check semantic conservation beyond merely invoking solve again.
            for o in a["signals"]:
                t, l = o["top"], o["left"]
                _need(np.all(out[t:t+s, l:l+s] == o["color"]), "signal changed")
            for v in a["vehicles"]:
                m = _movement(v)
                dy, dx = _DELTA[v["direction"]]
                t, l = v["top"] + m * dy, v["left"] + m * dx
                final = out[t:t+s, l:l+s]
                # Count newly painted interior cells: exactly one per step.
                _need(np.count_nonzero(final[1:-1, 1:-1] == v["color"])
                      == v["advanced"] + m, "counter conservation")
                canonical = np.rot90(final, _TURNS[v["direction"]])[1:-1, 1:-1].ravel()
                expected = [v["color"]] * (v["advanced"] + m) + [bg] * (v["capacity"] - v["advanced"] - m)
                _need(np.array_equal(canonical, expected), "rotated prefix update")
            (training if split == "train" else testing).append(a)
    for m in ("nearest", "reverse", "horizontal", "horizontal_reverse", "greedy", "fuel", "dock", "scan"):
        _need(any(_witnesses(a)[m] for a in training), "missing training witness: " + m)
    imbalances = [len(a["vehicles"]) - len(a["signals"]) for a in training]
    _need(min(imbalances) < 0 < max(imbalances), "teach both unmatched tails")
    _need(all(abs(x) <= 1 for x in imbalances), "training tail range")
    _need(abs(len(testing[0]["vehicles"]) - len(testing[0]["signals"])) >= 2, "missing longer-tail shift")
    directions = {v["direction"] for a in training for v in a["vehicles"] if _movement(v)}
    _need(len(directions) >= 2, "teach more than one counter orientation")
    _need(all(v["direction"] in directions for a in testing for v in a["vehicles"]),
          "untaught test orientation")
    vehicle_colors = {v["color"] for a in training for v in a["vehicles"]}
    signal_colors = {o["color"] for a in training for o in a["signals"]}
    _need(bool(vehicle_colors & signal_colors), "teach geometric rather than color classes")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """42 serial seeds; retain scalar features/hashes, never a sample archive."""
    hashes, layouts, shapes, sizes, backgrounds, counts, directions = set(), set(), set(), set(), set(), set(), set()
    regime_direction, tails, prefixes, distances = set(), set(), set(), set()
    fresh = repeats = reordered = 0
    first = None
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        hashes.add(_puzzle_digest(p))
        if seed == 0:
            first = p
        for split in ("train", "test"):
            for pair in p[split]:
                a = _parse(pair["input"])
                _need(np.array_equal(solve(pair["input"].copy()), pair["output"]), "input-only solve")
                sizes.add(a["size"])
                backgrounds.add(a["background"])
                shapes.add(pair["input"].shape)
                counts.add((len(a["vehicles"]), len(a["signals"])))
                tails.add(len(a["vehicles"]) - len(a["signals"]))
                layouts.add((a["size"], tuple((v["top"], v["left"], v["direction"]) for v in a["vehicles"]),
                             tuple((o["top"], o["left"]) for o in a["signals"])))
                colors = [v["color"] for v in a["vehicles"]] + [o["color"] for o in a["signals"]]
                repeats += len(set(colors)) < len(colors)
                moved_tops = []
                for v in a["vehicles"]:
                    directions.add(v["direction"])
                    prefixes.add((a["size"], v["advanced"]))
                    fresh += v["advanced"] == 0
                    m, d, rem = _movement(v), v["distance"], v["capacity"] - v["advanced"]
                    distances.add(d)
                    regime = "unmatched" if not d else "dock" if d < rem else "exact" if d == rem else "fuel"
                    regime_direction.add((v["direction"], regime))
                    moved_tops.append(v["top"] + _DELTA[v["direction"]][0] * m)
                reordered += moved_tops != sorted(moved_tops)
    _need(len(hashes) == 42 and len(layouts) >= 180, "structural diversity")
    _need(len(shapes) >= 35 and sizes == {3, 4, 5, 6}, "scale/canvas diversity")
    _need(len(backgrounds) >= 8 and len(counts) >= 12, "palette/count diversity")
    _need(directions == set(_DIRS) and fresh > 0 and repeats > 20, "direction/fill/color diversity")
    _need(all((d, r) in regime_direction for d in _DIRS for r in ("dock", "fuel", "unmatched")),
          "direction does not determine stopping")
    _need(min(tails) <= -2 and max(tails) >= 2 and len(distances) >= 10 and len(prefixes) >= 20,
          "tail/distance/counter diversity")
    _need(reordered > 0, "static input ranking must survive output reordering")
    _need(_puzzle_digest(first) == _puzzle_digest(generate_puzzle(0)), "determinism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17, nt, ne)
            _need(len(p["train"]) == nt and len(p["test"]) == ne, "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.array(g, copy=True) for k, g in pair.items()} for pair in first[s]] for s in first}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "layouts": len(layouts), "input_shapes": len(shapes),
            "sizes": sorted(sizes), "backgrounds": sorted(backgrounds), "count_pairs": sorted(counts),
            "direction_regimes": sorted(regime_direction), "fresh_vehicles": fresh, "repeated_color_pairs": repeats,
            "output_reordering_pairs": reordered, "distance_values": sorted(distances)}
