"""Generate solved rectangle-attraction ARC puzzles."""

import numpy as np


SIDES = ("N", "E", "S", "W")


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def counts(seed, train, test):
    for name, value, low, high in (("train", train, 2, 10), ("test", test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"invalid {name} count")
    if train is None and test is None:
        test = 1 + seed % 4
        first = max(2, test + 1)
        train = first + (seed // 4) % (11 - first)
    elif train is None:
        first = max(2, test + 1)
        train = first + seed % (11 - first)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("train count must exceed test count")
    return int(train), int(test)


def grid(rows):
    return np.asarray([[int(c) for c in row] for row in rows.split()], dtype=np.int8)


SOURCE = (
    (grid("""1111777777 1111777797 1111777777 1111777777 7777777777
             7977777777 7777777777 7797777777 7777777777 7777777777"""),
     grid("""2222777777 2222977777 2222777777 2922777777 7797777777
             7777777777 7777777777 7777777777 7777777777 7777777777""")),
    (grid("""77777777 77777777 77977777 77777777 71117777 71117797
             71117777 77777777"""),
     grid("""77777777 77777777 77777777 77777777 72927777 72229777
             72227777 77777777""")),
    (grid("""777777777777 777777797777 777777777777 777777777777
             777777711117 777977711117 777777711117 977777711117
             777777777777 777777777777 777777777777 777777777797"""),
     grid("""777777777777 777777777777 777777777777 777777777777
             777777792227 777777922227 777777722227 777777922227
             777777777797 777777777777 777777777777 777777777777""")),
    (grid("""7777777797777777 7777777777777777 7777777777777777
             7777777777777777 7777777777777777 9777711111777797
             7777711111777777 7777711111777977 7777711111777779
             7777711111777777 7777777777777777 7777777777777777
             7777777777777777 7777777777777777 7777777797777777
             7777779777777777"""),
     grid("""7777777777777777 7777777777777777 7777777777777777
             7777777777777777 7777777797777777 7777922222977777
             7777722222777777 7777722229777777 7777722222977777
             7777722222777777 7777779797777777 7777777777777777
             7777777777777777 7777777777777777 7777777777777777
             7777777777777777""")),
)


def route(point, bounds, diagonal=False):
    row, col = point
    r0, r1, c0, c1 = bounds
    v = r0 - row if row < r0 else row - r1 if row > r1 else 0
    h = c0 - col if col < c0 else col - c1 if col > c1 else 0
    need(v or h, "marker in rectangle")
    if v and h:
        need(diagonal and v != h, "diagonal outside core family")
    if v > h:
        return ("N" if row < r0 else "S", max(c0, min(c1, col)), v)
    return ("W" if col < c0 else "E", max(r0, min(r1, row)), h)


def destination(side, tangent, bounds, perimeter):
    r0, r1, c0, c1 = bounds
    if side == "N":
        return (r0 if perimeter else r0 - 1, tangent)
    if side == "S":
        return (r1 if perimeter else r1 + 1, tangent)
    if side == "W":
        return (tangent, c0 if perimeter else c0 - 1)
    return (tangent, c1 if perimeter else c1 + 1)


def inspect_input(inp, diagonal=False):
    values, counts_ = np.unique(inp, return_counts=True)
    need(len(values) == 3 and np.count_nonzero(counts_ == max(counts_)) == 1,
         "input needs three identifiable roles")
    bg = int(values[np.argmax(counts_)])
    rectangles = []
    for color in values:
        if color == bg:
            continue
        pts = np.argwhere(inp == color)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bounds = tuple(map(int, (r0, r1, c0, c1)))
        if r1 - r0 >= 2 and c1 - c0 >= 2 and len(pts) == (r1-r0+1)*(c1-c0+1):
            rectangles.append((int(color), bounds))
    need(len(rectangles) == 1, "one solid rectangle required")
    source, bounds = rectangles[0]
    mark = (set(map(int, values)) - {bg, source}).pop()
    points = [tuple(map(int, p)) for p in np.argwhere(inp == mark)]
    need(len(points) >= 2, "two or more markers required")
    for i, a in enumerate(points):
        for b in points[i+1:]:
            need(max(abs(a[0]-b[0]), abs(a[1]-b[1])) > 1, "markers must be isolated")
    routes = [route(p, bounds, diagonal) for p in points]
    gaps = [r[2] for r in routes]
    near = min(gaps)
    need(near >= 2 and gaps.count(near) == 1, "unique nearest marker required")
    need(len({(s, t) for s, t, _ in routes}) == len(routes), "projection collision")
    dests = [destination(s, t, bounds, d == near) for s, t, d in routes]
    need(len(set(dests)) == len(dests), "destination collision")
    need(all(0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] for r, c in dests),
         "destination outside grid")
    return {"background": bg, "source": source, "marker": mark, "bounds": bounds,
            "points": points, "routes": routes, "destinations": dests,
            "nearest_gap": near, "nearest_side": routes[gaps.index(near)][0],
            "sides": {r[0] for r in routes}}


def transform(inp, info, target):
    out = inp.copy()
    for p in info["points"]:
        out[p] = info["background"]
    r0, r1, c0, c1 = info["bounds"]
    out[r0:r1+1, c0:c1+1] = target
    for p in info["destinations"]:
        out[p] = info["marker"]
    return out


def palette(rng, target, baseline=None):
    if baseline is not None:
        return baseline
    return tuple(map(int, rng.choice([c for c in range(10) if c != target], 3, replace=False)))


def geometry(rng, regime):
    if regime == "train":
        rh, rw = map(int, rng.integers(3, 5, 2))
        h, w = map(int, rng.integers(8, 17, 2))
    else:
        sizes = (3, 4, 4, 5, 5, 5, 5, 6, 6, 7)
        rh, rw = int(rng.choice(sizes)), int(rng.choice(sizes))
        h = (16 if rng.random() < .22 else
             int(rng.integers(max(8, rh+3), 17 if rng.random() < .60 else 25)))
        w = (16 if rng.random() < .22 else
             int(rng.integers(max(8, rw+3), 17 if rng.random() < .60 else 25)))
    def offset(extent):
        return int(rng.choice((0, extent))) if rng.random() < .28 else int(rng.integers(extent+1))
    r0, c0 = offset(h-rh), offset(w-rw)
    return (h, w), (r0, r0+rh-1, c0, c0+rw-1)


def candidates(shape, bounds):
    h, w = shape
    r0, r1, c0, c1 = bounds
    margins = {"N": r0, "S": h-1-r1, "W": c0, "E": w-1-c1}
    result = []
    for side in SIDES:
        tangents = range(c0, c1+1) if side in ("N", "S") else range(r0, r1+1)
        for tangent in tangents:
            for gap in range(2, margins[side]+1):
                if side == "N": p = (r0-gap, tangent)
                elif side == "S": p = (r1+gap, tangent)
                elif side == "W": p = (tangent, c0-gap)
                else: p = (tangent, c1+gap)
                result.append((side, tangent, gap, p))
    return result


def pick_markers(rng, pool, amount, required):
    if not pool:
        return None
    maxgap = max(x[2] for x in pool)
    if maxgap < 3:
        return None
    for _ in range(16):
        nearest_gap = int(rng.integers(2, min(7, maxgap-1)+1))
        nearest = [x for x in pool if x[2] == nearest_gap]
        rng.shuffle(nearest)
        for first in nearest[:10]:
            chosen = [first]
            projections = {(first[0], first[1])}
            def compatible(x):
                p = x[3]
                return (x[2] > nearest_gap and (x[0], x[1]) not in projections
                        and all(max(abs(p[0]-y[3][0]), abs(p[1]-y[3][1])) > 1 for y in chosen))
            missing = list(required - {first[0]})
            rng.shuffle(missing)
            for side in missing:
                choices = [x for x in pool if x[0] == side and compatible(x)]
                if not choices:
                    break
                x = choices[int(rng.integers(len(choices)))]
                chosen.append(x)
                projections.add((x[0], x[1]))
            if not required.issubset({x[0] for x in chosen}):
                continue
            weights = rng.dirichlet(np.full(4, float(rng.uniform(.35, 1.7))))
            ranked = [(float(rng.random()) ** (1/max(weights[SIDES.index(x[0])], .01)), x)
                      for x in pool if x[2] > nearest_gap]
            ranked.sort(key=lambda entry: entry[0], reverse=True)
            for _, x in ranked:
                if len(chosen) == amount:
                    break
                if compatible(x):
                    chosen.append(x)
                    projections.add((x[0], x[1]))
            if len(chosen) == amount:
                return [x[3] for x in chosen]
    return None


def make_pair(rng, regime, amount, required, colors, target):
    for attempt in range(250):
        if attempt < 200:
            shape, bounds = geometry(rng, regime)
        elif regime == "train":
            shape, bounds = (12, 13), (4, 6, 4, 7)
        else:
            shape, bounds = (18, 18), (6, 10, 5, 10)
        points = pick_markers(rng, candidates(shape, bounds), amount, required)
        if points is None:
            continue
        bg, src, mark = colors
        inp = np.full(shape, bg, dtype=np.int8)
        r0, r1, c0, c1 = bounds
        inp[r0:r1+1, c0:c1+1] = src
        for p in points:
            inp[p] = mark
        if np.count_nonzero(inp == bg) <= np.count_nonzero(inp == src):
            continue
        if any(inp.shape == old.shape and np.array_equal(inp, old) for old, _ in SOURCE):
            continue
        info = inspect_input(inp)
        return {"input": inp, "output": transform(inp, info, target)}
    raise RuntimeError("layout construction failed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    nt, ne = counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    target = int(rng.integers(10))
    baseline = palette(rng, target)
    constant = bool(rng.random() < .32)
    directions = list(SIDES)
    rng.shuffle(directions)
    required = [set() for _ in range(nt)]
    required[0], required[1] = set(directions[:2]), set(directions[2:])
    sparse, dense = map(int, rng.choice(nt, size=2, replace=False))
    train_amounts = [int(rng.integers(2, 6)) for _ in range(nt)]
    train_amounts[sparse] = int(rng.integers(2, 4))
    train_amounts[dense] = int(rng.integers(4, 6))
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, total in (("train", nt), ("test", ne)):
        for index in range(total):
            amount = train_amounts[index] if regime == "train" else int(rng.integers(3, 10))
            needed_sides = required[index] if regime == "train" else set()
            for _ in range(50):
                pair_rng = np.random.default_rng(int(rng.integers(1 << 63)))
                colors = palette(pair_rng, target, baseline if constant else None)
                pair = make_pair(pair_rng, regime, amount, needed_sides, colors, target)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise RuntimeError("duplicate layouts could not be avoided")
    validate_generated_puzzle(puzzle)
    return puzzle


def inspect_pair(pair, regime):
    need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
    inp, out = pair["input"], pair["output"]
    for arr in (inp, out):
        need(isinstance(arr, np.ndarray) and arr.ndim == 2, "grid array")
        need(np.issubdtype(arr.dtype, np.integer), "grid integer dtype")
        need(all(1 <= d <= 30 for d in arr.shape) and bool(np.all((0 <= arr) & (arr <= 9))),
             "grid dimensions or colors")
    need(inp.shape == out.shape, "shape must stay fixed")
    info = inspect_input(inp)
    r0, r1, c0, c1 = info["bounds"]
    rh, rw = r1-r0+1, c1-c0+1
    h, w = inp.shape
    n = len(info["points"])
    if regime == "train":
        need(8 <= h <= 16 and 8 <= w <= 16 and 3 <= rh <= 4 and 3 <= rw <= 4
             and 2 <= n <= 5, "train regime")
    else:
        need(8 <= h <= 24 and 8 <= w <= 24 and 3 <= rh <= 7 and 3 <= rw <= 7
             and 3 <= n <= 9, "test regime")
    colors = set(map(int, np.unique(out)))
    need(info["source"] not in colors, "source color remains")
    choices = colors - {info["background"], info["marker"]}
    need(len(choices) == 1, "target ambiguous")
    target = choices.pop()
    need(len({info["background"], info["source"], info["marker"], target}) == 4,
         "colors not distinct")
    need(np.array_equal(out, transform(inp, info, target)), "incorrect output")
    need(not np.array_equal(inp, out), "unchanged output")
    need(int(np.count_nonzero(out == info["marker"])) == n, "marker count changed")
    need(sum(r0 <= r <= r1 and c0 <= c <= c1 for r, c in info["destinations"]) == 1,
         "perimeter marker count")
    info.update(target=target, shape=(h, w), rect_shape=(rh, rw))
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    need(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "section schema")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            info = inspect_pair(pair, regime)
            infos[regime].append(info)
            inp, out = pair["input"], pair["output"]
            key_in = (inp.shape, inp.tobytes())
            key = (key_in, out.shape, out.tobytes())
            need(key_in not in seen_inputs and key not in seen_pairs, "duplicate input or pair")
            seen_inputs.add(key_in)
            seen_pairs.add(key)
            need(not any(inp.shape == old.shape and np.array_equal(inp, old)
                         and np.array_equal(out, answer) for old, answer in SOURCE),
                 "source pair copied")
    need(len({x["target"] for sec in infos.values() for x in sec}) == 1,
         "inconsistent target fill")
    need(set().union(*(x["sides"] for x in infos["train"])) == set(SIDES),
         "training must show all four sides")
    sizes = [len(x["points"]) for x in infos["train"]]
    need(min(sizes) <= 3 and max(sizes) >= 4, "training must show sparse and dense cases")


def signature(puzzle):
    return tuple((section, tuple((p["input"].shape, p["input"].tobytes(),
                                  p["output"].tobytes()) for p in puzzle[section]))
                 for section in ("train", "test"))


def predict(train, inp):
    # Only demonstrations and the test input are available here.
    targets = []
    for pair in train:
        info = inspect_input(pair["input"])
        values = set(map(int, np.unique(pair["output"]))) - {info["background"], info["marker"]}
        need(len(values) == 1, "training target ambiguous")
        targets.append(values.pop())
    need(len(set(targets)) == 1, "training targets disagree")
    return transform(inp, inspect_input(inp), targets[0])


def self_test() -> None:
    for inp, expected in SOURCE:
        need(np.array_equal(transform(inp, inspect_input(inp, True), 2), expected),
             "source rule reproduction failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    shapes, rects, markers, gaps, sides, edges, layouts = (set() for _ in range(7))
    constant = varied = sparse_test = dense_test = small_test = large_test = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        need(signature(puzzle) == signature(generate_puzzle(seed)), "nondeterminism")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                info = inspect_pair(pair, regime)
                shapes.add(info["shape"])
                rects.add(info["rect_shape"])
                markers.add(len(info["points"]))
                gaps.add(info["nearest_gap"])
                sides.add(info["nearest_side"])
                palettes.add((info["background"], info["source"], info["marker"]))
                r0, r1, c0, c1 = info["bounds"]
                h, w = info["shape"]
                if r0 == 0: edges.add("N")
                if r1 == h-1: edges.add("S")
                if c0 == 0: edges.add("W")
                if c1 == w-1: edges.add("E")
                inp = pair["input"]
                pts = np.argwhere(inp != info["background"])
                crop = inp[pts[:,0].min():pts[:,0].max()+1, pts[:,1].min():pts[:,1].max()+1]
                normalized = np.where(crop == info["background"], 0,
                                      np.where(crop == info["source"], 1, 2))
                layouts.add((crop.shape, normalized.tobytes()))
                if regime == "test":
                    sparse_test += len(info["points"]) <= 4
                    dense_test += len(info["points"]) >= 7
                    small_test += h <= 16 and w <= 16
                    large_test += h >= 17 or w >= 17
                    need(np.array_equal(predict(puzzle["train"], inp), pair["output"]),
                         "test prediction failed")
        constant += len(palettes) == 1
        varied += len(palettes) > 1
    need(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    need(len({signature(p) for p in puzzles}) >= 40, "duplicate puzzles")
    need(len(shapes) >= 40 and len(rects) >= 10 and len(layouts) >= 100,
         "structural diversity")
    need(markers == set(range(2, 10)), "marker count coverage")
    need(sides == set(SIDES) and len(gaps) >= 4 and edges == set(SIDES),
         "direction, distance or edge coverage")
    need(constant >= 4 and varied >= 12, "palette scope coverage")
    need(min(sparse_test, dense_test, small_test, large_test) >= 5, "test scale coverage")
    for nt, ne in ((2,1), (10,1), (10,4), (None,1), (None,4), (2,None), (10,None)):
        p = generate_puzzle(93071, nt, ne)
        validate_generated_puzzle(p)
        need(nt is None or len(p["train"]) == nt, "train count ignored")
        need(ne is None or len(p["test"]) == ne, "test count ignored")
        need(signature(p) == signature(generate_puzzle(93071, nt, ne)), "count nondeterminism")
    for nt, ne in ((1,None), (11,None), (None,0), (None,5), (2,2), (3,4), (4,4)):
        try:
            generate_puzzle(93071, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
