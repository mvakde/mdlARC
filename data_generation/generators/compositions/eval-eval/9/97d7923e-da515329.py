"""Ranked pinwheel terminals: unary spoke keys choose winding destinations.

Only standard-library modules and NumPy are used. The solver decodes pixels;
there is no generation metadata, parent import, or puzzle-level hidden state.
"""
from itertools import combinations
from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _signature(a):
    a = np.ascontiguousarray(a)
    return a.shape, a.tobytes()


def _rotate(y, x, arm):
    for _ in range(arm):
        y, x = x, -y
    return y, x


def _path(radius, arm, stages=24):
    """Infinite-plane prefix and first-forward-run index intervals.

    North initially turns east. Stage k walks forward 2+4k, right R-2,
    forward R, then turns right. Other arms are clockwise quarter turns.
    Six full cycles suffice for every 30x30 canvas: each cycle moves eight
    cells outward, including the inward jog. Never stop at a boundary exit.
    """
    coords = [_rotate(-d, 0, arm) for d in range(1, radius + 1)]
    y, x, dy, dx = -radius, 0, 0, 1
    runs = []
    for k in range(stages):
        start = len(coords) - 1
        for vy, vx, steps in ((dy, dx, 2 + 4*k),
                              (dx, -dy, radius - 2), (dy, dx, radius)):
            for _ in range(steps):
                y, x = y + vy, x + vx
                coords.append(_rotate(y, x, arm))
        runs.append((k, start, start + 2 + 4*k))
        dy, dx = dx, -dy
    return coords, runs


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], []
        while todo:
            y, x = todo.pop()
            component.append((y, x))
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(component)
    return result


def _parse_at(a, bg, cy, cx):
    h, w = a.shape
    vectors = ((-1, 0), (0, 1), (1, 0), (0, -1))
    spokes = []
    for dy, dx in vectors:
        values = []
        for d in range(1, 7):
            y, x = cy + d*dy, cx + d*dx
            if not (0 <= y < h and 0 <= x < w) or a[y, x] == bg:
                break
            values.append(int(a[y, x]))
        spokes.append(values)
    radius = len(spokes[0])
    _require(3 <= radius <= 5 and all(len(s) == radius for s in spokes),
             "four fully visible equal spokes of length 3..5 required")
    payload = spokes[0][-1]
    _require(payload != bg and all(s[-1] == payload for s in spokes),
             "spoke tails must share a payload color")
    seed_mask = np.zeros(a.shape, dtype=bool)
    active = {}
    for arm, (s, (dy, dx)) in enumerate(zip(spokes, vectors)):
        for d in range(1, radius + 1):
            seed_mask[cy + d*dy, cx + d*dx] = True
        if all(c == payload for c in s):
            continue
        cap = s[0]
        rank = next((i for i, c in enumerate(s) if c != cap), radius)
        _require(cap not in (bg, payload) and 1 <= rank < radius
                 and s[rank:] == [payload]*(radius-rank), "malformed unary spoke")
        _require(cap not in active, "active spokes need distinct identities")
        active[cap] = {"arm": arm, "rank": rank}
    _require(1 <= len(active) <= 4, "no active spoke")
    y0, y1 = cy-radius, cy+radius+1
    x0, x1 = cx-radius, cx+radius+1
    _require(0 <= y0 and y1 <= h and 0 <= x0 and x1 <= w, "clipped cross")
    _require(np.array_equal(a[y0:y1, x0:x1] != bg,
                            seed_mask[y0:y1, x0:x1]), "cross box contains a gate")
    residue = (a != bg) & ~seed_mask
    bars = {c: [] for c in active}
    path_data = {}
    for cap, instruction in active.items():
        coords, runs = _path(radius, instruction["arm"])
        index = {(cy+y, cx+x): i for i, (y, x) in enumerate(coords)}
        _require(len(index) == len(coords), "winding self-intersection")
        path_data[cap] = (coords, runs, index)
    for component in _components(residue):
        points = sorted(component)
        ys, xs = {y for y, _ in points}, {x for _, x in points}
        _require(len(points) >= 3 and (len(ys) == 1 or len(xs) == 1),
                 "each gate must be a straight one-cell-wide capped bar")
        values = [int(a[q]) for q in points]
        cap = values[0]
        _require(cap in active and values[-1] == cap
                 and values[1:-1] == [payload]*(len(values)-2), "invalid gate colors")
        coords, runs, index = path_data[cap]
        _require(all(q in index for q in points), "gate is on the wrong winding phase")
        ids = sorted(index[q] for q in points)
        _require(ids == list(range(ids[0], ids[-1]+1)), "gate is not a route interval")
        stages = [k for k, lo, hi in runs if k >= 1 and lo < ids[0] <= ids[-1] < hi]
        _require(len(stages) == 1, "gate must lie strictly inside a long forward run")
        bars[cap].append({"length": len(points), "start": ids[0], "end": ids[-1],
                          "stage": stages[0], "points": points})
    for cap, family in bars.items():
        _require(2 <= len(family) <= 5, "each active spoke needs 2..5 gates")
        _require(len({b["length"] for b in family}) == len(family), "tied gate lengths")
        _require(len({b["stage"] for b in family}) == len(family), "one gate per forward run")
        _require(active[cap]["rank"] <= len(family), "requested rank does not exist")
        family.sort(key=lambda b: b["start"])
        active[cap]["bars"] = family
        active[cap]["path"] = path_data[cap][0]
    return {"background": bg, "payload": payload, "radius": radius,
            "center": (cy, cx), "active": active, "seed_mask": seed_mask,
            "shape": a.shape}


def _decode(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.sum(counts == counts[bg]) == 1, "background must be uniquely modal")
    occupied = a != bg
    possible = np.zeros(a.shape, dtype=bool)
    possible[1:-1, 1:-1] = (~occupied[1:-1, 1:-1] & occupied[:-2, 1:-1]
                            & occupied[2:, 1:-1] & occupied[1:-1, :-2]
                            & occupied[1:-1, 2:])
    scenes = []
    for cy, cx in np.argwhere(possible):
        try:
            scenes.append(_parse_at(a, bg, int(cy), int(cx)))
        except AssertionError:
            continue
    _require(len(scenes) == 1, "input must encode one unambiguous keyed hollow cross")
    return scenes[0]


def _choose(instruction, mode="rank"):
    bars, n = instruction["bars"], instruction["rank"]
    if mode == "nearest":
        return bars[0]
    if mode == "farthest":
        return bars[-1]
    if mode == "route_rank":
        return bars[n-1]
    ordered = sorted(bars, key=lambda b: b["length"], reverse=mode != "ascending")
    if mode == "longest":
        return ordered[0]
    if mode == "shortest":
        return ordered[-1]
    return ordered[n-1]


def _answer(scene, mode="rank"):
    out = np.full(scene["shape"], scene["background"], dtype=np.int8)
    cy, cx = scene["center"]
    for cap, instruction in scene["active"].items():
        chosen = _choose(instruction, mode if mode not in ("full", "bars_only") else "rank")
        if mode == "bars_only":
            for q in chosen["points"]:
                out[q] = cap
            continue
        stop = len(instruction["path"]) - 1 if mode == "full" else chosen["end"]
        for y, x in instruction["path"][:stop+1]:
            y, x = cy + y, cx + x
            if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
                _require(out[y, x] in (scene["background"], cap), "phase collision")
                out[y, x] = cap
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all identities, ranks, routes, and terminal indices from input."""
    a = _grid(input_grid)
    return _answer(_decode(a))


def _available(radius, shape, center, arm):
    coords, runs = _path(radius, arm)
    cy, cx = center
    result = []
    for k, lo, hi in runs:
        if k == 0:
            continue
        inside = [i for i in range(lo+1, hi)
                  if 0 <= cy+coords[i][0] < shape[0] and 0 <= cx+coords[i][1] < shape[1]]
        if len(inside) >= 3:
            result.append({"stage": k, "lo": inside[0], "hi": inside[-1],
                           "capacity": min(12, len(inside))})
    return coords, result


def _gate_plan(rng, available, forced_count=None, anchor=False):
    if anchor:
        runs = [next(b for b in available if b["stage"] == k) for k in (1, 2, 3)]
        # The longest is an intermediate destination; rank two reaches farther.
        lengths = [3, int(rng.integers(6, 10)), int(rng.integers(4, 6))]
    else:
        count_options = []
        combos_by_count = {}
        for count in range(2, min(5, len(available))+1):
            good = [combo for combo in combinations(available, count)
                    if all(cap >= 3+i for i, cap in enumerate(sorted(b["capacity"] for b in combo)))]
            if good:
                count_options.append(count)
                combos_by_count[count] = good
        _require(count_options, "no feasible distinct-length gate family")
        count = forced_count if forced_count in count_options else int(rng.choice(count_options))
        choices = combos_by_count[count]
        runs = list(choices[int(rng.integers(len(choices)))])
        lengths = [None]*count
        used = set()
        style = int(rng.integers(3))
        for i in sorted(range(count), key=lambda i: runs[i]["capacity"]):
            choices = [n for n in range(3, runs[i]["capacity"]+1) if n not in used]
            length = min(choices) if style == 0 else (max(choices) if style == 1 else int(rng.choice(choices)))
            lengths[i] = length
            used.add(length)
    plans = []
    for run, length in zip(runs, lengths):
        _require(length <= run["capacity"], "gate does not fit")
        start = int(rng.integers(run["lo"], run["hi"]-length+2))
        plans.append((start, start+length-1))
    return plans


def _make_scene(rng, radius, families, palette, geometry="roomy", anchor_arm=None,
                rank_limit=2):
    bg, payload = palette[:2]
    if geometry == "anchor":
        shape = tuple(int(v) for v in rng.integers(28, 31, size=2))
        center = tuple(int(rng.integers(radius+6, side-radius-6)) for side in shape)
    elif geometry == "edge":
        shape = tuple(int(v) for v in rng.integers(max(21, 2*radius+3), 31, size=2))
        center = [int(rng.integers(radius, side-radius)) for side in shape]
        axis = int(rng.integers(2))
        center[axis] = int(rng.choice([radius, shape[axis]-1-radius]))
        center = tuple(center)
    else:
        shape = tuple(int(v) for v in rng.integers(2*radius+11, 31, size=2))
        center = tuple(int(rng.integers(radius+4, side-radius-4)) for side in shape)
    a = np.full(shape, bg, dtype=np.int8)
    cy, cx = center
    for arm in range(4):
        for d in range(1, radius+1):
            y, x = _rotate(-d, 0, arm)
            a[cy+y, cx+x] = payload
    eligible = []
    for arm in range(4):
        coords, available = _available(radius, shape, center, arm)
        if len(available) >= 2 and max(b["capacity"] for b in available) >= 4:
            eligible.append(arm)
    _require(len(eligible) >= families, "not enough visible candidate routes")
    if anchor_arm is not None:
        arms = [anchor_arm] + [int(v) for v in rng.permutation([a for a in eligible if a != anchor_arm])][:families-1]
    else:
        arms = [int(v) for v in rng.choice(eligible, families, replace=False)]
    keys = {}
    for j, arm in enumerate(arms):
        cap = int(palette[2+j])
        coords, available = _available(radius, shape, center, arm)
        plans = _gate_plan(rng, available, anchor=arm == anchor_arm)
        rank = 1 if arm == anchor_arm else int(rng.integers(1, min(rank_limit, radius-1, len(plans))+1))
        keys[cap] = (arm, rank)
        for d in range(1, rank+1):
            y, x = _rotate(-d, 0, arm)
            a[cy+y, cx+x] = cap
        for start, end in plans:
            for i in range(start, end+1):
                y, x = coords[i]
                q = (cy+y, cx+x)
                _require(a[q] == bg, "gate overlaps another object")
                a[q] = cap if i in (start, end) else payload
    return a, keys


def _set_rank(a, cap, rank):
    scene = _decode(a)
    instruction = scene["active"][cap]
    cy, cx = scene["center"]
    b = a.copy()
    for d in range(1, scene["radius"]+1):
        y, x = _rotate(-d, 0, instruction["arm"])
        b[cy+y, cx+x] = cap if d <= rank else scene["payload"]
    return b


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regimes = ["edge", "multiplex"] + (["scale"] if ntrain >= 3 else [])
    regime = str(rng.choice(regimes))
    radius = 3 if regime == "scale" else int(rng.integers(3, 6))
    palette = [int(c) for c in rng.permutation(10)]
    common_palette = bool(rng.integers(2))
    anchor_arm = int(rng.integers(4))
    families = int(rng.integers(1, 3 if regime == "multiplex" else 4))
    first, keys = _make_scene(rng, radius, families, palette, "anchor", anchor_arm)
    cap = next(c for c, (arm, _) in keys.items() if arm == anchor_arm)
    second = _set_rank(first, cap, 2)
    puzzle = {"train": [], "test": []}
    seen = set()

    def add(split, a):
        if _signature(a) in seen:
            # Forty visible color/rotation variants; each background has four
            # distinct rotations because an active spoke has a unique identity.
            original = a
            for offset in range(10):
                for turn in range(4):
                    variant = np.ascontiguousarray((np.rot90(original, turn)+offset) % 10)
                    if _signature(variant) not in seen:
                        a = variant
                        break
                else:
                    continue
                break
            else:
                raise AssertionError("uniqueness fallback exhausted")
        seen.add(_signature(a))
        puzzle[split].append({"input": a.copy(), "output": solve(a)})

    add("train", first)
    add("train", second)
    for i in range(2, ntrain):
        r = 4 if regime == "scale" else int(rng.integers(3, 6))
        f = int(rng.integers(1, 3 if regime == "multiplex" else 5))
        colors = palette if common_palette else [int(c) for c in rng.permutation(10)]
        limit = 2 if ntrain < 4 else min(4, r-1)
        a, _ = _make_scene(rng, r, f, colors, rank_limit=limit)
        add("train", a)
    train_scenes = [_decode(p["input"]) for p in puzzle["train"]]
    known_ranks = {v["rank"] for s in train_scenes for v in s["active"].values()}
    rank_limit = max(known_ranks)  # Test uses demonstrated ordinals only.
    known_radii = sorted({s["radius"] for s in train_scenes})
    for _ in range(ntest):
        r = 5 if regime == "scale" else int(rng.choice(known_radii))
        f = int(rng.integers(3, 5)) if regime == "multiplex" else int(rng.integers(1, 5))
        colors = palette if common_palette else [int(c) for c in rng.permutation(10)]
        a, _ = _make_scene(rng, r, f, colors, "edge" if regime == "edge" else "roomy",
                           rank_limit=rank_limit)
        add("test", a)
    # Random demonstration order does not change the deliberate key contrast.
    order = rng.permutation(ntrain)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split types")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, scenes = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            _require(_signature(a) not in seen, "duplicate input")
            seen.add(_signature(a))
            s = _decode(a)
            expected = _answer(s)
            _require(np.array_equal(b, expected), "incorrect ranked terminal completion")
            _require(np.any(a != b), "identity example")
            _require(b[s["center"]] == s["background"], "center hole filled")
            _require(s["payload"] not in np.unique(b), "instruction payload was not consumed")
            _require(set(map(int, np.unique(b))) <= {s["background"], *s["active"]}, "new color")
            for cap, instruction in s["active"].items():
                chosen = _choose(instruction)
                _require(all(b[q] == cap for q in chosen["points"]), "selected gate not solid")
                for bar in instruction["bars"]:
                    if bar["start"] > chosen["end"]:
                        _require(all(b[q] == s["background"] for q in bar["points"]), "unused gate retained")
            scenes[split].append(s)
    ranks = {v["rank"] for s in scenes["train"] for v in s["active"].values()}
    _require({1, 2} <= ranks, "training must show ordinal progression")
    _require(any(1 < v["rank"] < len(v["bars"]) for s in scenes["train"]
                 for v in s["active"].values()), "training lacks an intermediate selection")
    for mode in ("nearest", "farthest", "longest", "shortest", "ascending", "route_rank", "full", "bars_only"):
        _require(any(not np.array_equal(_answer(s, mode), p["output"])
                     for s, p in zip(scenes["train"], puzzle["train"])), f"shortcut survives: {mode}")
    test_ranks = {v["rank"] for s in scenes["test"] for v in s["active"].values()}
    _require(all(r in ranks or set(range(1, r)) <= ranks for r in test_ranks), "unsupported test ordinal")
    radii = {s["radius"] for s in scenes["train"]}
    _require(all(s["radius"] in radii or (s["radius"] == max(radii)+1 and len(radii) >= 2)
                 for s in scenes["test"]), "unsupported scale shift")


def _measure(puzzles):
    shapes, layouts, palettes, signatures = set(), set(), set(), set()
    hist = {name: Counter() for name in ("families", "radius", "rank", "population", "length", "arm", "background")}
    features = Counter()
    for puzzle in puzzles:
        signatures.add(tuple(_signature(p[k]) for split in ("train", "test")
                             for p in puzzle[split] for k in ("input", "output")))
        local_families, local_shapes = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _decode(pair["input"])
                h, w = s["shape"]
                cy, cx = s["center"]
                shapes.add(s["shape"])
                local_shapes.add(s["shape"])
                local_families.add(len(s["active"]))
                hist["families"][len(s["active"])] += 1
                hist["radius"][s["radius"]] += 1
                hist["background"][s["background"]] += 1
                palettes.add((s["background"], s["payload"], tuple(s["active"])))
                margins = (cy-s["radius"], h-1-cy-s["radius"], cx-s["radius"], w-1-cx-s["radius"])
                features["edge_cross"] += int(min(margins) == 0)
                features["rectangles"] += int(h != w)
                layout = []
                for cap, v in s["active"].items():
                    chosen = _choose(v)
                    hist["rank"][v["rank"]] += 1
                    hist["population"][len(v["bars"])] += 1
                    hist["arm"][v["arm"]] += 1
                    for bar in v["bars"]:
                        hist["length"][bar["length"]] += 1
                    features["shortest_selected"] += int(chosen["length"] == min(b["length"] for b in v["bars"]))
                    features["intermediate_selected"] += int(1 < v["rank"] < len(v["bars"]))
                    features["late_stage_gate"] += sum(b["stage"] >= 5 for b in v["bars"])
                    prefix = v["path"][:chosen["end"]+1]
                    flags = [0 <= cy+y < h and 0 <= cx+x < w for y, x in prefix]
                    features["reentry"] += int(any(not flags[i-1] and flags[i] for i in range(1, len(flags))))
                    layout.append((v["arm"], v["rank"], tuple((b["length"], b["start"], b["end"]) for b in v["bars"])))
                layouts.add((s["radius"], s["shape"], s["center"], tuple(layout)))
        features["within_puzzle_family_variation"] += int(len(local_families) > 1)
        features["within_puzzle_shape_variation"] += int(len(local_shapes) > 1)
    return {"puzzles": len(signatures), "shapes": len(shapes), "layouts": len(layouts),
            "palettes": len(palettes), "histograms": {k: dict(v) for k, v in hist.items()},
            "features": dict(features)}


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    measured = _measure(puzzles)
    _require(measured["puzzles"] == 42, "seed diversity")
    _require(measured["shapes"] >= 35 and measured["layouts"] >= 130, "structural diversity")
    _require(measured["palettes"] >= 30, "palette diversity")
    hist = measured["histograms"]
    _require(set(hist["families"]) == {1, 2, 3, 4}, "active phase diversity")
    _require(set(hist["radius"]) == {3, 4, 5}, "scale diversity")
    _require(set(hist["population"]) == {2, 3, 4, 5}, "independent population coverage")
    _require({1, 2, 3} <= set(hist["rank"]), "rank diversity")
    _require(set(hist["arm"]) == {0, 1, 2, 3}, "phase diversity")
    for name in ("edge_cross", "rectangles", "shortest_selected", "intermediate_selected", "late_stage_gate", "reentry"):
        _require(measured["features"][name] > 0, f"missing {name}")
    _require(measured["features"]["within_puzzle_family_variation"] >= 20, "internal family variation")
    _require(measured["features"]["within_puzzle_shape_variation"] == 42, "internal shape variation")
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            a = pair["input"]
            before = a.copy()
            _require(np.array_equal(solve(a), pair["output"]), "input-only prediction")
            _require(np.array_equal(a, before), "solver mutated its input")
    # Rotation and role renaming preserve the input-encoded convention.
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        for turn in range(4):
            a, b = np.rot90(pair["input"], turn), np.rot90(pair["output"], turn)
            _require(np.array_equal(solve(a), b), "rotation equivariance")
            mapping = np.array([4, 9, 2, 8, 6, 0, 3, 5, 1, 7], dtype=np.int8)
            _require(np.array_equal(solve(mapping[a]), mapping[b]), "color equivariance")
    # Extending the infinite-plane calculation cannot add a missed late reentry.
    for radius in range(3, 6):
        paths = [set(_path(radius, arm)[0]) for arm in range(4)]
        _require(all(not paths[a] & paths[b] for a in range(4) for b in range(a)), "phase intersection")
        for arm in range(4):
            short = {(y, x) for y, x in _path(radius, arm, 24)[0] if -29 <= y <= 29 and -29 <= x <= 29}
            long = {(y, x) for y, x in _path(radius, arm, 40)[0] if -29 <= y <= 29 and -29 <= x <= 29}
            _require(short == long, "insufficient continuation bound")
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(7100+ntrain, ntrain, ntest)
            _require(len(p["train"]) == ntrain and len(p["test"]) == ntest, "explicit counts")
    a, b = generate_puzzle(888633603, 10, 4), generate_puzzle(888633603, 10, 4)
    _require(all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s])
                 for k in ("input", "output")), "determinism")
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    pair = bad["test"][0]
    q = tuple(np.argwhere(pair["input"] != pair["output"])[0])
    pair["output"][q] = pair["input"][q]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        return
    raise AssertionError("wrong output was accepted")
