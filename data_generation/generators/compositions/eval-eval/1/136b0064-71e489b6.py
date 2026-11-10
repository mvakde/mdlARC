"""Negative-space relay: damaged plates are a spatial, colored program.

Only NumPy and the standard library are used. No parent module, seed, examples,
or generation metadata are consulted by solve(). See task_spec.md for the rule.
"""
from __future__ import annotations

import numpy as np


# The tip is (0, 0). The other two holes form the back of the chevron.
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
HOLES = {
    (0, 1): ((0, 0), (-1, -1), (1, -1)),
    (1, 0): ((0, 0), (-1, -1), (-1, 1)),
    (0, -1): ((0, 0), (-1, 1), (1, 1)),
    (-1, 0): ((0, 0), (1, -1), (1, 1)),
    None: ((0, 0),),
}


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _dilate(a):
    h, w = a.shape
    padded = np.pad(a, 1)
    return np.logical_or.reduce([padded[y:y+h, x:x+w]
                                 for y in range(3) for x in range(3)])


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    offsets = DIRECTIONS + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if diagonal else ())
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in offsets:
                p = (y + dy, x + dx)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.append(p)
        groups.append(group)
    return groups


def _decode(input_grid):
    """Recover rectangles and negative glyphs solely from observed pixels.

    Plate ink has large four-connected components. Tiny positive corruption
    components cannot supply a plate. Bounding rectangles recover even a missing
    corner; a small isolated piece of genuine plate ink remains inside that box.
    """
    inp = _grid(input_grid)
    colors, counts = np.unique(inp[inp != 0], return_counts=True)
    singletons = colors[counts == 1]
    if len(singletons) != 1:
        raise ValueError("one unique singleton color must mark the initial tip")
    marker = int(singletons[0])
    start = tuple(map(int, np.argwhere(inp == marker)[0]))
    base = np.zeros_like(inp, dtype=bool)
    holes = np.zeros_like(base)
    plates, tips = [], {}
    for color in map(int, colors):
        if color == marker:
            continue
        for component in _components(inp == color):
            if len(component) < 12:
                continue
            pts = np.asarray(component)
            y, x = map(int, pts.min(axis=0))
            v, u = map(int, pts.max(axis=0) + 1)
            if v-y < 5 or u-x < 5 or base[y:v, x:u].any():
                raise ValueError("invalid or overlapping intended plates")
            block = inp[y:v, x:u]
            if not np.isin(block, (0, color, marker)).all():
                raise ValueError("a plate contains another ink color")
            missing = set((int(r+y), int(c+x))
                          for r, c in np.argwhere(block != color))
            matches = []
            for tip in sorted(missing):
                for direction, offsets in HOLES.items():
                    pattern = {(tip[0]+dy, tip[1]+dx) for dy, dx in offsets}
                    if missing == pattern:
                        matches.append((tip, direction))
            if len(matches) != 1:
                raise ValueError("plate does not have one readable negative glyph")
            tip, direction = matches[0]
            plate = {"box": (y, v, x, u), "color": color,
                     "tip": tip, "direction": direction}
            if tip in tips:
                raise ValueError("coincident command tips")
            tips[tip] = len(plates)
            plates.append(plate)
            base[y:v, x:u] = True
            for p in missing:
                holes[p] = True
    if not plates or start not in tips or plates[tips[start]]["direction"] is None:
        raise ValueError("marker must occupy the tip of a chevron")
    if np.any((inp == marker) & ~holes):
        raise ValueError("marker outside a genuine missing cell")
    return {"input": inp, "marker": marker, "start": start,
            "plates": plates, "tips": tips, "base": base, "holes": holes}


def _trace(info, *, farthest=False, all_arrows=False):
    """Execute pointers by nearest positive distance, never by raster order."""
    plates = info["plates"]
    current = info["tips"][info["start"]]
    visited, edges = [], []

    def target(index):
        a = plates[index]
        if a["direction"] is None:
            return None
        dy, dx = a["direction"]
        y, x = a["tip"]
        choices = []
        for j, b in enumerate(plates):
            ry, rx = b["tip"][0]-y, b["tip"][1]-x
            distance = ry*dy + rx*dx
            if distance > 0 and ry*dx == rx*dy:
                choices.append((distance, j))
        if not choices:
            raise ValueError("active chevron has no forward target")
        return (max(choices) if farthest else min(choices))[1]

    if all_arrows:
        for i, plate in enumerate(plates):
            if plate["direction"] is not None:
                try:
                    edges.append((i, target(i)))
                except ValueError:
                    pass
        return list(range(len(plates))), edges
    while True:
        if current in visited:
            raise ValueError("active relay cycles instead of stopping")
        visited.append(current)
        nxt = target(current)
        if nxt is None:
            break
        edges.append((current, nxt))
        current = nxt
    return visited, edges


def _predict(info, edges, *, use_halos=True):
    # First reconstruct the scaffold, thereby removing every extra speck.
    out = np.zeros_like(info["input"])
    for plate in info["plates"]:
        y, v, x, u = plate["box"]
        out[y:v, x:u] = plate["color"]
    bridge = np.zeros_like(info["base"])
    for a, b in edges:
        source, dest = info["plates"][a], info["plates"][b]
        y, x = source["tip"]
        ty, tx = dest["tip"]
        dy, dx = int(np.sign(ty-y)), int(np.sign(tx-x))
        while (y, x) != (ty, tx):
            y, x = y+dy, x+dx
            if not info["base"][y, x]:
                out[y, x] = source["color"]
                bridge[y, x] = True
    if use_halos:
        out[_dilate(info["holes"]) & ~info["holes"]] = info["marker"]
    out[info["holes"]] = 0
    out[info["start"]] = info["marker"]
    return out, bridge


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; palette roles and every command are visible."""
    info = _decode(input_grid)
    _, edges = _trace(info)
    return _predict(info, edges)[0]


def _axis(rng, n, corner=False):
    # Nonuniform spacing, with a roomy first gap for readable positive decoys.
    lengths, gaps, margins = [5]*n, [4]+[2]*(n-2), [0, 0]
    minimum = sum(lengths)+sum(gaps)
    capacity = n*7+(n-1)*5+(2 if corner else 4)
    target = int(rng.integers(minimum, min(30, minimum+9, capacity)+1))
    for _ in range(target-minimum):
        options = ([(0, i) for i, size in enumerate(lengths) if size < 7]
                   + [(1, i) for i, gap in enumerate(gaps) if gap < 5]
                   + [(2, i) for i, m in enumerate(margins)
                      if m < 2 and not (corner and i == 0)])
        kind, i = options[int(rng.integers(len(options)))]
        (lengths, gaps, margins)[kind][i] += 1
    windows, anchors = [], []
    pos = margins[0]
    for i, length in enumerate(lengths):
        windows.append((pos, pos+length))
        offset = 1 if corner else int(rng.integers(1, length-1))
        anchors.append(pos+offset)
        pos += length+(gaps[i] if i < n-1 else 0)
    return windows, anchors, pos+margins[1]


def _walk(rng, rows, cols, length):
    """Finite randomized self-avoiding walks, including occasional skipped slots."""
    vertices = {(r, c) for r in range(rows) for c in range(cols)}
    visits = 0

    def search(path, reserved):
        nonlocal visits
        visits += 1
        if visits > 6000:
            return None
        if len(path) == length:
            return (path, reserved) if vertices-set(path)-reserved else None
        y, x = path[-1]
        choices = [(d, step) for d in DIRECTIONS for step in (1, 2)]
        scores = rng.random(len(choices))
        order = sorted(range(len(choices)), key=lambda i: scores[i]+.35*(choices[i][1]-1))
        for i in order:
            (dy, dx), step = choices[i]
            dest = y+dy*step, x+dx*step
            intermediate = {(y+dy, x+dx)} if step == 2 else set()
            if (dest not in vertices or dest in path or dest in reserved
                    or intermediate & (set(path) | reserved)):
                continue
            answer = search(path+[dest], reserved | intermediate)
            if answer is not None:
                return answer
        return None

    starts = sorted(vertices)
    for i in rng.permutation(len(starts)):
        result = search([starts[int(i)]], set())
        if result is not None:
            return result
        if visits > 6000:
            break
    # A bounded, always feasible contiguous walk; geometry still varies.
    snake = [(r, c) for r in range(rows)
             for c in (range(cols) if r % 2 == 0 else reversed(range(cols)))]
    return snake[:length], set()


def _interval(rng, window, values):
    low, high = window
    candidates = [(a, a+size) for size in range(5, high-low+1)
                  for a in range(low, high-size+1)
                  if a <= min(values) and max(values) < a+size]
    return candidates[int(rng.integers(len(candidates)))]


def _paint_case(rng, palette, marker, split, index, rotation, reflection):
    if split == "train" and index == 0:
        rows, cols = 2, 3
        path, reserved = [(0, 0), (0, 1), (1, 1), (1, 0)], set()
        forced = {(0, 2)}
    elif split == "train" and index == 1:
        rows, cols = 3, 2
        path, reserved = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1)], set()
        forced = {(0, 1)}
    else:
        shapes = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2)) if split == "train" else (
            (3, 3), (3, 4), (4, 3), (4, 4))
        rows, cols = shapes[int(rng.integers(len(shapes)))]
        length = int(rng.integers(4, 6)) if split == "train" else int(
            rng.integers(6, min(10, rows*cols-1)+1))
        path, reserved = _walk(rng, rows, cols, length)
        forced = set()
    unused = sorted({(r, c) for r in range(rows) for c in range(cols)}-set(path)-reserved-forced)
    decoys = set(forced)
    desired = int(rng.integers(1, min(3, len(unused)+len(forced))+1))
    for i in rng.permutation(len(unused))[:max(0, desired-len(forced))]:
        decoys.add(unused[int(i)])
    corner = split == "train" and index == 0
    yw, ya, h = _axis(rng, rows, corner)
    xw, xa, w = _axis(rng, cols, corner)
    directions = {}
    for a, b in zip(path, path[1:]):
        directions[a] = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    directions[path[-1]] = None
    for station in sorted(decoys):
        # The first off-route plate has a valid outgoing connection. This
        # makes 'execute every arrow' observably different in demonstration 1.
        directions[station] = (0, -1) if corner and station == (0, 2) else (
            DIRECTIONS[int(rng.integers(4))] if rng.random() < .85 else None)
    shuffled = list(map(int, rng.permutation(palette)))
    colors = {}
    for i, station in enumerate(path):
        choices = [c for c in palette if i == 0 or c != colors[path[i-1]]]
        colors[station] = shuffled[i] if i < len(shuffled) else int(rng.choice(choices))
    for station in sorted(decoys):
        colors[station] = int(rng.choice(palette))
    inp = np.zeros((h, w), dtype=np.int64)
    clean = np.zeros_like(inp)
    base = np.zeros_like(inp, dtype=bool)
    missing = np.zeros_like(base)
    boxes = {}
    for r, c in sorted(directions):
        points = [(ya[r]+dy, xa[c]+dx) for dy, dx in HOLES[directions[r, c]]]
        y, v = _interval(rng, yw[r], [p[0] for p in points])
        x, u = _interval(rng, xw[c], [p[1] for p in points])
        boxes[r, c] = (y, v, x, u)
        clean[y:v, x:u] = colors[r, c]
        base[y:v, x:u] = True
        for p in points:
            missing[p] = True
    inp[:] = clean
    inp[missing] = 0
    start = (ya[path[0][0]], xa[path[0][1]])
    inp[start] = marker
    # Construction labels use explicit scaffold, route, and hole masks. They
    # do not call the input decoder, tracer, or prediction renderer.
    out = clean.copy()
    bridges = np.zeros_like(base)
    for a, b in zip(path, path[1:]):
        sy, sx, ty, tx = ya[a[0]], xa[a[1]], ya[b[0]], xa[b[1]]
        if sy == ty:
            cells = [(sy, x) for x in range(min(sx, tx), max(sx, tx)+1)]
        else:
            cells = [(y, sx) for y in range(min(sy, ty), max(sy, ty)+1)]
        for p in cells:
            if not base[p]:
                out[p] = colors[a]
                bridges[p] = True
    # Explicit, local Moore painting, independent of the solver's dilation.
    for y, x in np.argwhere(missing):
        out[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)] = marker
    out[missing] = 0
    out[start] = marker
    extras = np.zeros_like(base)
    safe = ~_dilate(base)
    if corner:
        # A positive chevron lies on the first outgoing ray, before its true
        # target. Its polarity and absence of a plate make it removable noise.
        end = boxes[0, 0][3]
        begin = boxes[0, 1][2]
        choices = []
        for tipx in range(end+1, begin):
            pts = [(ya[0]+dy, tipx+dx) for dy, dx in HOLES[0, -1]]
            if all(0 <= y < h and 0 <= x < w and safe[y, x] for y, x in pts):
                choices.append(pts)
        if not choices:
            raise RuntimeError("first gap cannot hold its positive decoy")
        for p in choices[int(rng.integers(len(choices)))]:
            extras[p] = True
            inp[p] = colors[path[0]]
    # Every pair contains visible erasure away from halos and routed bridges.
    far = np.argwhere(safe & ~_dilate(extras) & ~bridges)
    if not len(far):
        raise RuntimeError("no readable field area for an extra speck")
    p = tuple(map(int, far[int(rng.integers(len(far)))]))
    extras[p] = True
    inp[p] = int(rng.choice(palette))
    motifs = [((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1))] + list(HOLES.values())[:4]
    for _ in range(int(rng.integers(0, 5))):
        motif = motifs[int(rng.integers(len(motifs)))]
        available = safe & ~_dilate(extras)
        choices = []
        for y, x in np.argwhere(available):
            pts = [(int(y+dy), int(x+dx)) for dy, dx in motif]
            if all(0 <= r < h and 0 <= c < w and available[r, c] for r, c in pts):
                choices.append(pts)
        if choices:
            color = int(rng.choice(palette))
            for p in choices[int(rng.integers(len(choices)))]:
                extras[p] = True
                inp[p] = color
    def transform(a):
        return np.rot90(np.fliplr(a) if reflection else a, rotation).copy()
    return {"input": transform(inp), "output": transform(out)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.permutation(np.arange(1, 10))))
    marker = roles[0]
    palette = roles[1:1+int(rng.integers(2, 5))]
    anchor_rotation, anchor_reflection = int(rng.integers(4)), bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for _ in range(30):
                if split == "train" and index < 2:
                    rotation, reflection = anchor_rotation, anchor_reflection
                else:
                    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
                pair = _paint_case(rng, palette, marker, split, index, rotation, reflection)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    visited, edges = _trace(info)
    prediction, bridge = _predict(info, edges)
    halo = _dilate(info["holes"]) & ~info["holes"]
    extras = (info["input"] != 0) & (info["input"] != info["marker"]) & ~info["base"]
    boundary = np.zeros_like(halo)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    return {"visited": len(visited), "plates": len(info["plates"]),
            "holes": int(info["holes"].sum()), "extras": int(extras.sum()),
            "turns": tuple(info["plates"][i]["direction"] for i in visited[:-1]),
            "visible_bridge": bool((bridge & ~halo).any()),
            "far_erasure": bool((extras & ~halo & ~bridge).any()),
            "cross_boundary": bool((halo & ~info["base"]).any()),
            "clipped": bool((info["holes"] & boundary).any()),
            "shape": info["input"].shape,
            "plate_sizes": tuple((p["box"][1]-p["box"][0], p["box"][3]-p["box"][2])
                                 for p in info["plates"]),
            "prediction": prediction}


def validate_generated_puzzle(puzzle):
    """Validate actual arrays, input-only reconstruction, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, markers, palettes = set(), set(), set()
    features, training_infos = {"train": [], "test": []}, []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            local_palette = frozenset(p["color"] for p in info["plates"])
            assert 2 <= len(local_palette) <= 4
            palettes.add(local_palette)
            extra_mask = (inp != 0) & (inp != info["marker"]) & ~info["base"]
            assert not (extra_mask & _dilate(info["base"])).any()
            assert all(len(group) <= 3 for group in _components(extra_mask, diagonal=True))
            route, _ = _trace(info)
            assert all(info["plates"][a]["color"] != info["plates"][b]["color"]
                       for a, b in zip(route, route[1:]))
            f = _features(info)
            assert np.array_equal(out, f.pop("prediction")), "wrong relay or halo output"
            assert f["plates"] > f["visited"] and f["visible_bridge"]
            assert f["extras"] and f["far_erasure"]
            assert np.all(out[info["holes"] & (inp != info["marker"])] == 0)
            assert out[info["start"]] == info["marker"]
            # Parent ablations must change an actual answer, not just metadata.
            assert not np.array_equal(out, _predict(info, [])[0]), "A is inessential"
            assert not np.array_equal(out, _predict(info, _trace(info)[1], use_halos=False)[0]), "B is inessential"
            markers.add(info["marker"])
            features[split].append(f)
            if split == "train":
                training_infos.append(info)
    assert len(markers) == len(palettes) == 1
    assert [f["visited"] for f in features["train"][:2]] == [4, 5]
    assert max(f["visited"] for f in features["train"]) <= 5
    assert min(f["visited"] for f in features["test"]) >= 6
    assert set(d for f in features["train"] for d in f["turns"]) == set(DIRECTIONS)
    assert features["train"][0]["clipped"] and features["train"][0]["cross_boundary"]
    first = training_infos[0]
    # A polarity-blind reader would hit an opposite-pointing positive chevron
    # before the real target, then return to the start and cycle.
    first_visit, _ = _trace(first)
    source = first["plates"][first_visit[0]]
    target = first["plates"][first_visit[1]]
    dy, dx = source["direction"]
    sy, sx = source["tip"]
    real_distance = (target["tip"][0]-sy)*dy + (target["tip"][1]-sx)*dx
    extras = (first["input"] != 0) & ~first["base"]
    trap = False
    for color in {p["color"] for p in first["plates"]}:
        for group in _components(extras & (first["input"] == color), diagonal=True):
            cells = set(group)
            for tip in cells:
                pattern = {(tip[0]+r, tip[1]+c) for r, c in HOLES[-dy, -dx]}
                ry, rx = tip[0]-sy, tip[1]-sx
                if cells == pattern and ry*dx == rx*dy and 0 < ry*dy+rx*dx < real_distance:
                    trap = True
    assert trap, "no consequential positive/negative instruction distinction"
    # The third collinear plate is an inactive arrow. Both common alternatives
    # must be separated by observed output pixels in the first demonstration.
    expected = _predict(first, _trace(first)[1])[0]
    all_edges = _trace(first, all_arrows=True)[1]
    assert not np.array_equal(expected, _predict(first, all_edges)[0]), "execute-all shortcut survives"
    try:
        wrong_edges = _trace(first, farthest=True)[1]
    except ValueError:
        pass
    else:
        assert not np.array_equal(expected, _predict(first, wrong_edges)[0]), "farthest-target shortcut survives"


def self_test():
    """Bounded semantic checks and structural diversity across 42 full puzzles."""
    shapes, routes, plate_sizes, palettes, structures = set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        p = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                f = _features(info)
                records.append(f)
                shapes.add(f["shape"])
                routes.add(f["turns"])
                plate_sizes.update(f["plate_sizes"])
                palettes.add((info["marker"], tuple(sorted({s["color"] for s in info["plates"]}))))
                structures.add((info["base"].shape, info["base"].tobytes(), info["holes"].tobytes()))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                # Rule and scaffold recovery are equivariant to all D4 actions.
                if seed < 2:
                    for k in range(4):
                        for reflection in (False, True):
                            a = np.fliplr(pair["input"]) if reflection else pair["input"]
                            b = np.fliplr(pair["output"]) if reflection else pair["output"]
                            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p
                       for a, b in zip(p[s], q[s]) for k in a)
    assert len(shapes) >= 40 and len(routes) >= 55
    assert len(plate_sizes) >= 8 and len(palettes) >= 30 and len(structures) >= 180
    assert len({f["plates"] for f in records}) >= 6
    assert len({f["holes"] for f in records}) >= 10
    assert len({f["extras"] for f in records}) >= 7
    assert {f["visited"] for f in records} >= {4, 5, 6, 7, 8, 9, 10}
    # Every explicit count boundary and representative type errors.
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        generate_puzzle(1000+nt+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # An actual output mutation must be rejected by the family validator.
    p = generate_puzzle(123)
    info = _decode(p["test"][0]["input"])
    halo = _dilate(info["holes"]) & ~info["holes"]
    where = tuple(map(int, np.argwhere(halo)[0]))
    p["test"][0]["output"][where] = 0
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("semantic output mutation accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
