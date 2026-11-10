"""Room-powered wavefronts: repaired topology controls competing exterior growth.

Only NumPy and the standard library are used. No files are read or written.
Colors: 0 exterior, 9 damage/noise, 8 equal-arrival collision, 1..7 objects.
"""
from collections import deque
import hashlib
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return g.astype(np.int8, copy=True)


def _regions(mask):
    seen = np.zeros_like(mask, bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in N4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _holes(mask):
    """Enclosure relative to an explicit padded outside, including edge objects."""
    p = np.pad(mask, 1)
    exterior = np.zeros_like(p)
    exterior[0, 0] = True
    stack = [(0, 0)]
    h, w = p.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not p[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                stack.append((rr, cc))
    return (~p & ~exterior)[1:-1, 1:-1]


def _dilate(mask, neighbors=N8):
    p = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dr, dc in neighbors:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _repair(grid):
    """One simultaneous pass; true background openings are never repaired."""
    if np.any(grid == 8):
        raise ValueError("8 is an output-only collision color")
    out = grid.copy()
    out[out == 9] = 0
    h, w = grid.shape
    for y, x in np.argwhere(grid == 9):
        supports = set()
        if 0 < y < h - 1 and grid[y - 1, x] == grid[y + 1, x] and 1 <= grid[y - 1, x] <= 7:
            supports.add(int(grid[y - 1, x]))
        if 0 < x < w - 1 and grid[y, x - 1] == grid[y, x + 1] and 1 <= grid[y, x - 1] <= 7:
            supports.add(int(grid[y, x - 1]))
        if len(supports) > 1:
            raise ValueError("ambiguous repair")
        if supports:
            out[y, x] = supports.pop()
    return out


def _objects(clean):
    objects = []
    cavity_union = np.zeros_like(clean, bool)
    for color in np.unique(clean):
        if color == 0:
            continue
        body = clean == color
        if len(_regions(body)) != 1:
            raise ValueError("each object color must identify one orthogonal component")
        rr, cc = np.nonzero(body)
        t, l, b, r = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
        local = body[t:b + 1, l:r + 1]
        cavities = np.zeros_like(body)
        cavities[t:b + 1, l:r + 1] = _holes(local)
        if np.any(cavities & (clean != 0)) or np.any(cavity_union & cavities):
            raise ValueError("nested or multiply owned cavity")
        cavity_union |= cavities
        objects.append({"color": int(color), "body": body, "holes": cavities,
                        "count": len(_regions(cavities)), "box": (t, l, b, r)})
    if not objects:
        raise ValueError("no objects")
    if not np.array_equal(_holes(clean != 0), cavity_union):
        raise ValueError("collective enclosure is outside this family")
    return objects, (clean == 0) & ~cavity_union


def _distances(body, exterior, budget, neighbors=N8):
    """Bounded breadth-first distance through the static original exterior."""
    distances = np.full(body.shape, -1, np.int16)
    front = _dilate(body, neighbors) & exterior
    distances[front] = 1
    queue = deque(map(tuple, np.argwhere(front).tolist()))
    h, w = body.shape
    while queue:
        r, c = queue.popleft()
        d = int(distances[r, c])
        if d == budget:
            continue
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and exterior[rr, cc] and distances[rr, cc] < 0:
                distances[rr, cc] = d + 1
                queue.append((rr, cc))
    return distances


def _transform(clean, objects=None, exterior=None, *, unit=False, binary=False,
               fill=True, neighbors=N8, direct=False):
    if objects is None:
        objects, exterior = _objects(clean)
    out = clean.copy()
    best = np.full(clean.shape, 32767, np.int16)
    distances = []
    for obj in objects:
        if fill:
            out[obj["holes"]] = obj["color"]
        budget = 1 if unit else min(obj["count"], 1) + 1 if binary else obj["count"] + 1
        if direct:
            # Diagnostic alternative: disregard intervening bodies/cavities.
            reach = obj["body"].copy()
            d = np.full(clean.shape, -1, np.int16)
            for step in range(1, budget + 1):
                reach |= _dilate(reach, neighbors)
                d[reach & exterior & (d < 0)] = step
        else:
            d = _distances(obj["body"], exterior, budget, neighbors)
        active = d > 0
        earlier = active & (d < best)
        tied = active & (d == best)
        out[earlier] = obj["color"]
        out[tied] = 8
        best[earlier] = d[earlier]
        distances.append(d)
    return out, distances


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from this input alone; never mutate the caller."""
    clean = _repair(_grid(input_grid))
    return _transform(clean)[0]


def _partition(rng, count, compact=False):
    low = 7 if count >= 3 else 5
    high = 10 if compact else 12
    h, w = int(rng.integers(low, high)), int(rng.integers(low, high))
    boxes = [(0, 0, h - 1, w - 1)]
    first_axis = None
    for step in range(count - 1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum(((bb - tt) // 2) * ((rr - ll) // 2)
                                   for j, (tt, ll, bb, rr) in enumerate(boxes) if j != i)
                    capacity += sum(((bb - tt) // 2) * ((rr - ll) // 2) for tt, ll, bb, rr in halves)
                    if capacity >= count:
                        choices.append((i, axis, halves))
        if not choices:
            raise AssertionError("partition capacity exhausted")
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    mask = np.zeros((h, w), bool)
    for t, l, b, r in boxes:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _mask(rng, count, *, bay=False, compact=False):
    if count:
        mask = _partition(rng, count, compact)
        # A one-room thick body with an L-shaped, off-center sealed cavity.
        if count == 1 and rng.random() < .45:
            h, w = mask.shape
            mask[:] = True
            mask[1:-1, 1:3] = False
            mask[1:3, 1:-1] = False
    elif bay or rng.random() < .55:
        h, w = int(rng.integers(7, 10)), int(rng.integers(7, 10))
        mask = np.zeros((h, w), bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
        door = int(rng.integers(2, w - 2))
        mask[0, door:door + int(rng.integers(1, 3))] = False
    else:
        h, w = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        mask = np.ones((h, w), bool)
        if h >= 3 and w >= 3:
            mask[:int(rng.integers(1, h)), 1:int(rng.integers(2, w))] = False
    # Independent protrusions change silhouettes without changing the rooms.
    if count and not compact and rng.random() < .65:
        mask = np.pad(mask, ((0, 2), (0, 0)))
        width = mask.shape[1]
        left = int(rng.integers(width))
        mask[-3:, left:min(width, left + int(rng.integers(1, 5)))] = True
    mask = np.rot90(mask, int(rng.integers(4)))
    if rng.random() < .5:
        mask = np.fliplr(mask)
    assert len(_regions(mask)) == 1 and len(_regions(_holes(mask))) == count
    return mask.copy()


def _pack(rng, masks, *, near=False, fallback=False):
    n = len(masks)
    if fallback:
        # Compact masks are <=9x9. Five fit into a 2 by 3 shelf (<=19x29).
        order = list(map(int, rng.permutation(n)))
        rows = (n + 1) // 2
        h, w = rows * 10 - 1, 19 if n > 1 else 9
        origins = [None] * n
        for slot, i in enumerate(order):
            origins[i] = (10 * (slot // 2), 10 * (slot % 2))
        return (h, w), origins
    h, w = int(rng.integers(19, 31)), int(rng.integers(19, 31))
    for _ in range(12):
        occupied = np.zeros((h, w), bool)
        origins, placed = [None] * n, []
        order = sorted(range(n), key=lambda i: -masks[i].size)
        for i in order:
            mh, mw = masks[i].shape
            prefix = np.pad(occupied.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            if mh > h or mw > w:
                break
            sums = prefix[mh:, mw:] - prefix[:-mh, mw:] - prefix[mh:, :-mw] + prefix[:-mh, :-mw]
            sites = np.argwhere(sums == 0)
            if not len(sites):
                break
            if near and placed:
                distance = np.full(len(sites), 100)
                for r, c, ph, pw in placed:
                    dr = np.maximum(0, np.maximum(r - (sites[:, 0] + mh - 1), sites[:, 0] - (r + ph - 1)))
                    dc = np.maximum(0, np.maximum(c - (sites[:, 1] + mw - 1), sites[:, 1] - (c + pw - 1)))
                    distance = np.minimum(distance, np.maximum(dr, dc))
                sites = sites[distance == distance.min()]
            r, c = map(int, sites[int(rng.integers(len(sites)))])
            origins[i] = r, c
            placed.append((r, c, mh, mw))
            occupied[max(0, r - 1):min(h, r + mh + 1), max(0, c - 1):min(w, c + mw + 1)] = True
        else:
            return (h, w), origins
    return None


def _damage(rng, clean, objects, witness):
    grid = clean.copy()
    supported = np.zeros_like(clean, bool)
    supported[1:-1] |= (clean[:-2] == clean[2:]) & (clean[:-2] > 0)
    supported[:, 1:-1] |= (clean[:, :-2] == clean[:, 2:]) & (clean[:, :-2] > 0)
    supported &= clean > 0
    # Require an actually topology-changing interruption in the teaching scene.
    if witness:
        candidates = []
        for obj in objects:
            if obj["count"] < 2:
                continue
            boundary = _dilate(obj["holes"], N4)
            for r, c in np.argwhere(supported & obj["body"] & boundary):
                candidate = obj["body"].copy()
                candidate[r, c] = False
                if len(_regions(_holes(candidate))) < obj["count"]:
                    candidates.append((int(r), int(c)))
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        grid[r, c] = 9
    for r, c in rng.permutation(np.argwhere(supported))[:int(rng.integers(0, 5))]:
        if not np.any(grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2] == 9):
            grid[r, c] = 9
    # Unbracketed 9s are irrelevant specks, even inside a room.
    eligible = np.argwhere(clean == 0)
    for r, c in rng.permutation(eligible)[:int(rng.integers(1, 7))]:
        trial = grid.copy()
        trial[r, c] = 9
        if np.array_equal(_repair(trial), clean):
            grid = trial
    assert np.array_equal(_repair(grid), clean)
    return grid


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, np.uint8).tobytes()


def _features(inp, out=None):
    clean = _repair(inp)
    objects, exterior = _objects(clean)
    expected, distances = _transform(clean, objects, exterior)
    if out is not None:
        assert np.array_equal(out, expected), "incorrect output"
    strict = False
    crosses_territory = False
    for obj, d in zip(objects, distances):
        allowed = exterior & ((expected == 0) | (expected == obj["color"]))
        restricted = _distances(obj["body"], allowed, obj["count"] + 1)
        crosses_territory |= bool(np.any(exterior & (expected == obj["color"])
                                        & (d > 0) & (restricted < 0)))
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if a["count"] < b["count"]:
                da, db = distances[i], distances[j]
                strict |= bool(np.any((da > 0) & (db > da) & (expected == a["color"])))
    bay = False
    full_reach = set()
    shapes = []
    partial = False
    for obj, d in zip(objects, distances):
        t, l, b, r = obj["box"]
        local = obj["body"][t:b + 1, l:r + 1]
        shapes.append(hashlib.sha256(bytes(local.shape) + local.tobytes()).hexdigest())
        if not obj["count"]:
            bay |= bool(np.any((clean[t:b + 1, l:r + 1] == 0) & (expected[t:b + 1, l:r + 1] == 0)))
        if np.any((d == obj["count"] + 1) & (expected == obj["color"])):
            full_reach.add(obj["count"])
        degree = sum(np.roll(obj["body"], shift, axis=axis).astype(int)
                     for shift, axis in ((-1, 0), (1, 0), (-1, 1), (1, 1)))
        if b - t > 2 and r - l > 2:
            partial |= bool(np.any((degree[t + 1:b, l + 1:r] == 3) & obj["body"][t + 1:b, l + 1:r]))
    raw = inp.copy()
    raw[raw == 9] = 0
    raw_count = sum(len(_regions(_holes(raw == o["color"]))) for o in objects)
    return {"objects": objects, "distances": distances, "exterior": exterior,
            "counts": tuple(o["count"] for o in objects), "collision": bool(np.any(expected == 8)),
            "strict": strict, "bay": bay, "partial": partial, "full_reach": full_reach,
            "crosses_territory": crosses_territory,
            "repair_changes_count": raw_count != sum(o["count"] for o in objects),
            "shapes": shapes, "clean": clean,
            "unit_diff": not np.array_equal(expected, _transform(clean, objects, exterior, unit=True)[0]),
            "binary_diff": not np.array_equal(expected, _transform(clean, objects, exterior, binary=True)[0]),
            "direct_diff": not np.array_equal(expected, _transform(clean, objects, exterior, direct=True)[0]),
            "four_diff": not np.array_equal(expected, _transform(clean, objects, exterior, neighbors=N4)[0])}


def _make_pair(rng, counts, witness=False):
    for attempt in range(70):
        compact = attempt >= 45
        masks = [_mask(rng, k, bay=(witness and k == 0), compact=compact) for k in counts]
        layout = _pack(rng, masks, near=witness or rng.random() < .6, fallback=attempt >= 60)
        if layout is None:
            continue
        shape, origins = layout
        colors = list(map(int, rng.choice(np.arange(1, 8), len(counts), replace=False)))
        clean = np.zeros(shape, np.int8)
        for color, mask, (r, c) in zip(colors, masks, origins):
            clean[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = color
        objects, exterior = _objects(clean)
        grid = _damage(rng, clean, objects, witness)
        if grid is None:
            continue
        output = _transform(clean, objects, exterior)[0]
        if witness:
            f = _features(grid, output)
            required = set(counts)
            if not (f["collision"] and f["strict"] and required <= f["full_reach"]
                    and f["repair_changes_count"] and (0 not in counts or f["bay"])):
                continue
            if 3 in counts and 0 in counts and not (f["direct_diff"] and f["crosses_territory"]):
                continue
        return {"input": grid, "output": output}
    raise AssertionError("bounded teaching-scene construction exhausted")


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            if split == "train" and i == 0:
                counts = [3, 0] + ([int(rng.integers(1, 3))] if rng.random() < .3 else [])
            elif split == "train" and i == 1:
                counts = [2, 1] + ([0] if rng.random() < .4 else [])
            elif split == "train":
                counts = list(map(int, rng.integers(0, 4, size=int(rng.integers(1, 5)))))
            else:
                counts = [int(rng.integers(4, 6))]
                counts += list(map(int, rng.integers(0, 5, size=int(rng.integers(1, 5)))))
            for _ in range(12):
                pair = _make_pair(rng, counts, witness=(split == "train" and i < 2))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("duplicate input after bounded retries")
    rng.shuffle(puzzle["train"])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    records, seen = {"train": [], "test": []}, set()
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            assert not np.any(out == 9)
            f = _features(inp, out)
            clean, exterior = f["clean"], f["exterior"]
            # Original bodies are protected, and cavities are owned exclusively.
            assert np.array_equal(out[clean > 0], clean[clean > 0])
            for obj in f["objects"]:
                assert np.all(out[obj["holes"]] == obj["color"])
                for other in f["objects"]:
                    if obj is not other:
                        assert not np.any(_dilate(obj["body"]) & other["body"])
            active = np.logical_or.reduce([d > 0 for d in f["distances"]])
            assert np.all(out[exterior & ~active] == 0)
            assert np.all((out != 8) | exterior)
            records[split].append(f)
    tr, te = records["train"], records["test"]
    assert {0, 1, 2, 3} <= set(k for f in tr for k in f["counts"])
    assert {0, 1, 2, 3} <= set().union(*(f["full_reach"] for f in tr))
    for feature in ("collision", "strict", "bay", "partial", "repair_changes_count",
                    "unit_diff", "binary_diff", "four_diff", "direct_diff", "crosses_territory"):
        assert any(f[feature] for f in tr), "missing teaching evidence: " + feature
    assert max(k for f in tr for k in f["counts"]) == 3
    assert all(max(f["counts"]) >= 4 for f in te), "missing room-count extrapolation"


def self_test():
    """42 seeds, pixel-derived diversity, metamorphic and rejection checks; no I/O."""
    sizes, shapes, multisets, scenes = set(), set(), set(), set()
    collision = clear = partial = noise = bay = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in p:
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp, out)
                sizes.add(inp.shape)
                shapes.update(f["shapes"])
                multisets.add(tuple(sorted(f["counts"])))
                scenes.add(_key(f["clean"]))
                collision += f["collision"]
                clear += not f["collision"]
                partial += f["partial"]
                bay += f["bay"]
                noise += f["repair_changes_count"]
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], again[s]) for k in a)
            for pair in p["train"] + p["test"]:
                inp, out = pair["input"], pair["output"]
                for op in (np.rot90, np.fliplr, np.transpose):
                    assert np.array_equal(solve(op(inp)), op(out))
                colors = np.array([0, 7, 3, 1, 6, 2, 4, 5, 8, 9], np.int8)
                assert np.array_equal(solve(colors[inp]), colors[out])
                unchanged = inp.copy()
                solve(inp)
                assert np.array_equal(inp, unchanged)
    assert len(sizes) >= 35 and len(shapes) >= 150 and len(multisets) >= 35
    assert len(scenes) >= 200 and min(collision, clear, partial, noise, bay) >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1700 + nt * 10 + nq, nt, nq)
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-check a closed 3x3 loop: one room, radius two, filled center.
    g = np.zeros((9, 9), np.int8)
    g[3:6, 3:6] = 2
    g[4, 4] = 0
    expected = np.zeros_like(g)
    expected[1:8, 1:8] = 2
    assert np.array_equal(solve(g), expected)
    # A one-cell door is real background. Its deep bay stays exterior.
    g = np.zeros((11, 11), np.int8)
    g[2:9, 2:9] = 3
    g[3:8, 3:8] = 0
    g[2, 5] = 0
    assert solve(g)[5, 5] == 0
    damaged = g.copy()
    damaged[2, 5] = 9
    assert solve(damaged)[5, 5] == 3
    # Straight distance would cross the C's sealed back wall. The actual
    # route must enter its distant door and is longer than the fuel budget.
    g = np.zeros((17, 25), np.int8)
    g[4:11, 3:10] = 1
    g[5:10, 4:9] = 0
    g[4:11, 6] = 1
    g[7, 3:7] = 1
    g[3:12, 11:21] = 2
    g[4:11, 12:20] = 0
    g[7, 20] = 0
    assert solve(g)[7, 13] == 0
    assert _transform(g, direct=True)[0][7, 13] == 1
    # A competing one-round outline does not become an obstacle. The
    # longer-lived wave crosses its territory and enters the deep bay.
    g = np.zeros((17, 25), np.int8)
    g[4:13, 14:21] = 1
    g[5:12, 15:20] = 0
    g[4:13, 17] = 1
    g[8, 14:18] = 1
    g[3:12, 4:13] = 2
    g[4:11, 5:12] = 0
    g[7, 12] = 0
    result = solve(g)
    assert tuple(result[7, 10:14]) == (1, 2, 2, 8)
    # Validator must reject a wrong cavity fill or an exterior ownership error.
    p = generate_puzzle(800)
    for target in ("cavity", "collision"):
        q = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
        changed = False
        for pair in q["train"]:
            f = _features(pair["input"])
            mask = (np.logical_or.reduce([o["holes"] for o in f["objects"]])
                    if target == "cavity" else pair["output"] == 8)
            if mask.any():
                r, c = np.argwhere(mask)[0]
                pair["output"][r, c] = 0
                changed = True
                break
        assert changed
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted altered " + target)
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(sizes),
            "object_silhouettes": len(shapes), "count_multisets": len(multisets),
            "clean_scenes": len(scenes), "collision_scenes": collision,
            "no_collision_scenes": clear, "partial_scenes": partial,
            "topology_changing_damage_scenes": noise, "open_bay_scenes": bay}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
