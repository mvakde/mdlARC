"""Count through visible bands, then contract identical whole-grid vectors.

Only NumPy and the standard library are used. No source fixtures, files, seed,
or generation records are consulted by solve. Generation records independently
locate edits for construction checks, and are discarded before returning pairs.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


HOST_SEEDS = (0,772473637,216049449,309452819,590809668,266040304,838799608,
              107829239,2967521,384928522,483326375,119903619,833589379,
              145656413,650673332,119298740,387293864,59458557,824767258,
              743212559,606904095,634138286,76194146,94904110,901318910,
              18972497,457467093,804345898,161343284,282331990,862639038,
              186294265,148279056,365497540,727571567,964367369,428619865,
              860050724,657255679,935476392,619185724,573544534)
_DIRECTIONS = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _background(a):
    colors, counts = np.unique(a, return_counts=True)
    index = int(np.argmax(counts))
    if int(counts[index]) * 2 <= a.size:
        raise ValueError("family background must occupy more than half the input")
    return int(colors[index])


def _components(cells):
    left = set(cells)
    found = []
    while left:
        start = min(left)
        left.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in left:
                        left.remove(p)
                        component.add(p)
                        stack.append(p)
        found.append(component)
    return found


def _local(p, direction):
    r, c = p
    return {"U": (r, c), "R": (-c, r), "D": (-r, -c), "L": (c, -r)}[direction]


def _world(p, direction):
    y, x = p
    return {"U": (y, x), "R": (x, -y), "D": (-y, -x), "L": (-x, y)}[direction]


def _geometry(cells, direction):
    """Recognize a thin pointed envelope from its boundary, not a stencil."""
    local = {_local(p, direction) for p in cells}
    first, last = min(y for y, _ in local), max(y for y, _ in local)
    if last - first < 2:
        return None
    sections = [sorted(x for y, x in local if y == level)
                for level in range(first, last + 1)]
    if len(sections[0]) != 1:
        return None
    tip_x = sections[0][0]
    lo = hi = tip_x
    split = False
    cavity = set()
    for level, xs in enumerate(sections[1:], first + 1):
        if xs == [tip_x] and not split:
            continue
        if len(xs) != 2:
            return None
        left, right = xs
        if not (0 <= lo - left <= 1 and 0 <= right - hi <= 1):
            return None
        lo, hi, split = left, right, True
        cavity.update(_world((level, x), direction) for x in range(lo + 1, hi))
    if not cavity or not lo < tip_x < hi:
        return None
    return _world((first, tip_x), direction), cavity


def _ray(shape, tip, direction, count):
    dr, dc = _DIRECTIONS[direction]
    points = []
    for distance in range(1, count + 1):
        p = (tip[0] + distance * dr, tip[1] + distance * dc)
        if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]):
            break
        points.append(p)
    return points


def _objects(a):
    bg = _background(a)
    objects = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        for frame in _components(map(tuple, np.argwhere(a == color))):
            if len(frame) < 5:
                continue
            candidates = []
            for direction in _DIRECTIONS:
                geom = _geometry(frame, direction)
                if geom is None:
                    continue
                tip, cavity = geom
                if not all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in cavity):
                    continue
                values = {int(a[p]) for p in cavity} - {bg}
                if len(values) != 1 or color in values:
                    continue
                payload_color = next(iter(values))
                payload = {p for p in cavity if a[p] == payload_color}
                if not 1 <= len(payload) < len(cavity):
                    continue
                rs, cs = zip(*frame)
                box = {(r, c) for r in range(min(rs), max(rs) + 1)
                       for c in range(min(cs), max(cs) + 1)}
                candidates.append(dict(frame=frame, cavity=cavity, tip=tip,
                    direction=direction, frame_color=color, color=payload_color,
                    payload=payload, count=len(payload), box=box))
            if len(candidates) > 1:
                raise ValueError("ambiguous frame direction")
            objects.extend(candidates)
    if not objects:
        raise ValueError("no hollow pointed frame")
    objects.sort(key=lambda o: min(o["frame"]))
    boxes, rays = set(), set()
    for obj in objects:
        ray = set(_ray(a.shape, obj["tip"], obj["direction"], obj["count"]))
        if boxes & obj["box"] or rays & ray:
            raise ValueError("overlapping frame regions or tally rays")
        boxes |= obj["box"]
        rays |= ray
    if boxes & rays:
        raise ValueError("tally ray enters a frame")
    return objects


def _stamp(a, objects, counts=None):
    out = a.copy()
    bg = _background(a)
    for obj in objects:
        for p in obj["payload"]:
            out[p] = bg
    for index, obj in enumerate(objects):
        count = obj["count"] if counts is None else counts[index]
        for p in _ray(a.shape, obj["tip"], obj["direction"], count):
            out[p] = obj["color"]
    return out


def _indices(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _compress(a):
    reduced = a[_indices(a, 0)]
    return reduced[:, _indices(reduced, 1)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all payloads in original coordinates, stamp, then compress."""
    a = _grid(input_grid)
    return _compress(_stamp(a, _objects(a)))


def _profile(rng, capacity):
    for _ in range(40):
        reach_l, reach_r = (int(x) for x in rng.integers(1, 4, size=2))
        stem, tail = int(rng.integers(0, 2)), int(rng.integers(2, 5))
        boundaries = [(0, 0)] * (stem + 1)
        lo = hi = 0
        while lo > -reach_l or hi < reach_r:
            dl = lo > -reach_l and (hi == reach_r or rng.random() < .8)
            dr = hi < reach_r and (lo == -reach_l or rng.random() < .8)
            if not dl and not dr:
                dl = lo > -reach_l
                dr = not dl
            lo, hi = lo - int(dl), hi + int(dr)
            boundaries.append((lo, hi))
        boundaries += [(lo, hi)] * tail
        frame, cavity = set(), set()
        for y, (left, right) in enumerate(boundaries):
            frame.update({(y, left + reach_l), (y, right + reach_l)})
            cavity.update((y, x + reach_l) for x in range(left + 1, right))
        if len(cavity) > capacity:
            return dict(frame=frame, cavity=cavity, tip=(0, reach_l))
    # A finite, roomy fallback; this is geometry, never a stored example.
    frame = {(0, 3)}
    for y in range(1, 8):
        extent = min(y, 3)
        frame |= {(y, 3 - extent), (y, 3 + extent)}
    cavity = {(y, x) for y in range(1, 8) for x in range(4-min(y, 3), 3+min(y, 3))}
    return dict(frame=frame, cavity=cavity, tip=(0, 3))


def _orient(profile, direction):
    transformed = {p: _world(p, direction)
                   for p in profile["frame"] | profile["cavity"]}
    r0 = min(r for r, _ in transformed.values())
    c0 = min(c for _, c in transformed.values())
    def move(p):
        r, c = transformed[p]
        return r - r0, c - c0
    frame = {move(p) for p in profile["frame"]}
    return dict(frame=frame, cavity={move(p) for p in profile["cavity"]},
                tip=move(profile["tip"]), direction=direction,
                height=max(r for r, _ in frame) + 1,
                width=max(c for _, c in frame) + 1)


def _payload_order(rng, cavity, compact):
    cells = sorted(cavity)
    if not compact:
        return [cells[int(i)] for i in rng.permutation(len(cells))]
    chosen = [cells[int(rng.integers(len(cells)))]]
    remaining = set(cells) - set(chosen)
    while remaining:
        frontier = sorted(p for p in remaining
                          if any(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1 for q in chosen))
        choices = frontier or sorted(remaining)
        p = choices[int(rng.integers(len(choices)))]
        remaining.remove(p)
        chosen.append(p)
    return chosen


def _place(spec, r0, c0, frame_color, color, payload):
    def move(points):
        return {(r + r0, c + c0) for r, c in points}
    return dict(frame=move(spec["frame"]), cavity=move(spec["cavity"]),
                payload=move(payload), tip=(r0+spec["tip"][0], c0+spec["tip"][1]),
                direction=spec["direction"], frame_color=frame_color, color=color,
                count=len(payload), box={(r, c) for r in range(r0, r0+spec["height"])
                                       for c in range(c0, c0+spec["width"])})


def _paint(a, obj):
    for p in obj["frame"]:
        a[p] = obj["frame_color"]
    for p in obj["payload"]:
        a[p] = obj["color"]


def _scene(rng, palette, count, witness=False, test=False):
    """Build one scene; lines and frames are sampled together, not chained tasks."""
    bg, frame_color, payload0, payload1, hc, vc = palette
    primary = _orient(_profile(rng, max(count, 4)), "U")
    # A rail at exactly the primary endpoint makes a unit-count change visible.
    tip_r = count + int(rng.integers(4, 7))
    height_min = tip_r + primary["height"] + 1
    height = int(rng.integers(height_min, min(30, height_min + 5) + 1))
    width = int(rng.integers(max(14, primary["width"]+7), 31))
    c0 = int(rng.integers(3, width-primary["width"]-2))
    a = np.full((height, width), bg, dtype=np.int8)
    # Repeated equal-color bands separated by gaps are deliberately retained.
    hrows = {tip_r-count, tip_r-count-3}
    if witness:
        # The high-count demonstration must pass a band and continue through
        # background, not merely stop on its first band. Its two initial equal
        # ray rows also expose contraction of vectors created by the tally.
        hrows.add(tip_r-3)
    elif count >= 6:
        hrows.add(tip_r-int(rng.integers(2, count-2)))
    if test and count >= 8:
        hrows.add(tip_r-count+2)
    if not witness and rng.random() < .5 and min(hrows) >= 2:
        hrows.add(min(hrows)-1)  # A thick band, compressed only by exact equality.
    vcols = {int(rng.integers(1, c0-1)),
             int(rng.integers(c0+primary["width"]+1, width-1))}
    if not witness and rng.random() < .35:
        vcols.add(0)  # Occasionally a band touches the canvas edge.
    for c in vcols:
        a[:, c] = vc
    for r in hrows:
        a[r, :] = hc
    # Identical crossing ownership in neighboring equal bands allows thick runs.
    # Ownership varies between columns and scenes, never by a global priority.
    ownership = {c: bool(rng.integers(0, 2)) for c in sorted(vcols)}
    for r in hrows:
        for c in vcols:
            if ownership[c]:
                a[r, c] = vc
    order = _payload_order(rng, primary["cavity"], compact=witness or rng.random() < .5)
    main = _place(primary, tip_r, c0, frame_color, payload0, order[:count])
    objects = [main]
    _paint(a, main)
    target = 1 if witness else int(rng.integers(1 if not test else 2, 4))
    for _ in range(target-1):
        q = int(rng.integers(1, 8 if test else 6))
        spec = _orient(_profile(rng, q), str(rng.choice(list(_DIRECTIONS))))
        candidates = [(r, c) for r in range(height-spec["height"]+1)
                      for c in range(width-spec["width"]+1)]
        accepted = None
        boxes = set().union(*(o["box"] for o in objects))
        guarded = {(r+dr, c+dc) for r, c in boxes for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        rays = set().union(*(set(_ray(a.shape, o["tip"], o["direction"], o["count"])) for o in objects))
        payload = _payload_order(rng, spec["cavity"], compact=rng.random() < .5)[:q]
        for index in rng.permutation(len(candidates)):
            r, c = candidates[int(index)]
            obj = _place(spec, r, c, frame_color, int(rng.choice([payload0, payload1])), payload)
            if obj["box"] & (guarded | rays):
                continue
            if any(a[p] != bg for p in obj["box"]):
                continue
            ray = set(_ray(a.shape, obj["tip"], obj["direction"], q))
            if ray & (boxes | rays):
                continue
            accepted = obj
            break
        if accepted is not None:
            objects.append(accepted)
            _paint(a, accepted)
    expected = _compress(_stamp(a, objects))
    # Construction records and input geometry must identify exactly the same edits.
    observed = _objects(a)
    signature = lambda o: (o["tip"], o["direction"], o["count"], o["color"], frozenset(o["frame"]))
    assert {signature(o) for o in objects} == {signature(o) for o in observed}
    assert np.array_equal(solve(a), expected)
    return {"input": a, "output": expected}, main


def _transform(pair, rotation, reflect):
    return {role: np.rot90(a[:, ::-1] if reflect else a, rotation).copy()
            for role, a in pair.items()}


def _checked_count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice(10, 6, replace=False))
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    high, main = _scene(rng, palette, 5, witness=True)
    low_input = high["input"].copy()
    # Grow a connected two-cell subset of the high-count cluster. Only payload
    # occupancy changes: bands, frame, palette and original coordinates match.
    low_payload = {min(main["payload"])}
    low_payload.add(min(p for p in main["payload"] - low_payload
                        if any(abs(p[0]-q[0]) + abs(p[1]-q[1]) == 1
                               for q in low_payload)))
    for p in main["payload"] - low_payload:
        low_input[p] = palette[0]
    low_objects = [dict(main, payload=low_payload, count=2)]
    low = {"input": low_input, "output": _compress(_stamp(low_input, low_objects))}
    anchors = [_transform(low, rotation, reflect), _transform(high, rotation, reflect)]
    if rng.random() < .5:
        anchors.reverse()
    puzzle = {"train": anchors, "test": []}
    seen = {_key(p["input"]) for p in anchors}
    for split, remaining in (("train", num_train-2), ("test", num_test)):
        for index in range(remaining):
            for _ in range(16):
                q = int(rng.integers(4, 7)) if split == "train" else int(rng.integers(7, 11))
                pair, _ = _scene(rng, palette, q, test=split == "test")
                pair = _transform(pair, int(rng.integers(4)), bool(rng.integers(2)))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-scene sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(a, objects, counts):
    return _compress(_stamp(a, objects, counts))


def _has_count_witness(a, objects, expected):
    counts = [o["count"] for o in objects]
    for index, obj in enumerate(objects):
        for quantity in sorted({1, 2, obj["count"]-1, obj["count"]+1, len(obj["cavity"])-1}):
            if not 1 <= quantity < len(obj["cavity"]) or quantity == obj["count"]:
                continue
            trial = counts.copy()
            trial[index] = quantity
            # Exclude counterfactual lengths that would run into another object/ray.
            rays = [set(_ray(a.shape, o["tip"], o["direction"], q))
                    for o, q in zip(objects, trial)]
            boxes = set().union(*(o["box"] for o in objects))
            if any(ray & boxes for ray in rays):
                continue
            if any(rays[i] & rays[j] for i in range(len(rays)) for j in range(i)):
                continue
            if not np.array_equal(_alternative(a, objects, trial), expected):
                return True
    return False


def _run_lengths(a, axis):
    starts = np.flatnonzero(_indices(a, axis))
    return np.diff(np.r_[starts, a.shape[axis]])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen = set()
    component_counterexample = False
    extend_counterexample = False
    band_stop_counterexample = False
    repeated_tally_vectors = False
    matched = {}
    train_max = 0
    test_maxima = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            objects = _objects(a)
            assert 1 <= len(objects) <= 3
            middle = _stamp(a, objects)
            assert np.array_equal(_compress(middle), out), "incorrect stamping/compression"
            assert np.array_equal(solve(a), out)
            assert out.shape != a.shape, "compression absent"
            assert not np.array_equal(_compress(a), out), "omitting A has no effect"
            assert _has_count_witness(a, objects, out), "all tally counts erased by B"
            assert np.all(_indices(out, 0)) and np.all(_indices(out, 1))
            # Simultaneous-axis reconstruction checks retained crossings literally.
            rr, cc = _run_lengths(middle, 0), _run_lengths(middle, 1)
            assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cc, axis=1), middle)
            reverse = middle[:, _indices(middle, 1)]
            reverse = reverse[_indices(reverse, 0)]
            assert np.array_equal(reverse, out), "axis order changed answer"
            edited = set().union(*(o["payload"] | set(_ray(a.shape, o["tip"], o["direction"], o["count"])) for o in objects))
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    if (r, c) not in edited:
                        assert middle[r, c] == a[r, c]
            if split == "train":
                train_max = max(train_max, max(o["count"] for o in objects))
                components = [len(_components(o["payload"])) for o in objects]
                component_counterexample |= not np.array_equal(_alternative(a, objects, components), out)
                extend_counterexample |= not np.array_equal(_alternative(a, objects, [60]*len(objects)), out)
                stop_counts = []
                for obj in objects:
                    ray = _ray(a.shape, obj["tip"], obj["direction"], obj["count"])
                    stop_counts.append(next((i for i, p in enumerate(ray, 1)
                                             if a[p] != _background(a)), obj["count"]))
                    vectors = middle if obj["direction"] in ("U", "D") else middle.T
                    axis = 0 if obj["direction"] in ("U", "D") else 1
                    repeated_tally_vectors |= any(
                        np.array_equal(vectors[p[axis]], vectors[q[axis]])
                        for p, q in zip(ray, ray[1:]))
                band_stop_counterexample |= not np.array_equal(
                    _alternative(a, objects, stop_counts), out)
                # Compare scenes after clearing payloads: geometry, palette, rails,
                # position and direction are all held fixed in this witness.
                cleared = a.copy()
                for o in objects:
                    for p in o["payload"]:
                        cleared[p] = _background(a)
                matched.setdefault(_key(cleared), []).append((tuple(o["count"] for o in objects), _key(out)))
            else:
                test_maxima.append(max(o["count"] for o in objects))
    assert component_counterexample, "training admits component counting"
    assert extend_counterexample, "training admits extending every ray to the edge"
    assert band_stop_counterexample, "training admits stopping at the first band"
    assert repeated_tally_vectors, "training never contracts a repeated tally vector"
    assert any(len({c for c, _ in group}) >= 2 and len({o for _, o in group}) >= 2
               for group in matched.values()), "missing count-sensitive matched demonstrations"
    assert train_max < min(test_maxima), "a test lacks the larger-count shift"


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(pair["input"])), _key(np.asarray(pair["output"])))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Sequential 42-seed audit; at most one 14-pair puzzle retained, no disk I/O."""
    fingerprints, shapes, outputs, profiles, counts = set(), set(), set(), set(), set()
    directions, backgrounds, inventories = set(), set(), set()
    clipped = mixed = separated = crossings = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(hashlib.sha256(repr(_fingerprint(puzzle)).encode()).hexdigest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                objs = _objects(a)
                shapes.add(a.shape)
                outputs.add(pair["output"].shape)
                backgrounds.add(_background(a))
                inventories.add(len(objs))
                mixed += len({o["direction"] for o in objs}) > 1
                middle = _stamp(a, objs)
                # A repeated row separated by another row must survive twice.
                row_keys = [row.tobytes() for row in middle]
                separated += len(set(row_keys)) < int(_indices(middle, 0).sum())
                for obj in objs:
                    directions.add(obj["direction"])
                    counts.add(obj["count"])
                    local = [_local(p, obj["direction"]) for p in obj["frame"]]
                    y0, x0 = min(y for y, _ in local), min(x for _, x in local)
                    profiles.add(tuple(sorted((y-y0, x-x0) for y, x in local)))
                    ray = _ray(a.shape, obj["tip"], obj["direction"], obj["count"])
                    clipped += len(ray) < obj["count"]
                    crossings += any(a[p] != _background(a) for p in ray)
                assert np.array_equal(_compress(_compress(middle)), pair["output"])
        del puzzle
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(outputs) >= 20 and len(profiles) >= 30
    assert directions == set(_DIRECTIONS) and len(backgrounds) >= 8
    assert {1, 2, 3} <= inventories and set(range(1, 11)) <= counts
    assert mixed >= 10 and clipped >= 1 and separated >= 42 and crossings >= 84
    # All 30 legal explicit count combinations, with deterministic repetition.
    legal_calls = 0
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            a = generate_puzzle(1234+n*10+m, n, m)
            first = hashlib.sha256(repr(_fingerprint(a)).encode()).digest()
            assert len(a["train"]) == n and len(a["test"]) == m
            del a
            b = generate_puzzle(1234+n*10+m, n, m)
            assert first == hashlib.sha256(repr(_fingerprint(b)).encode()).digest()
            del b
            legal_calls += 1
    for n, m in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Small independent B-only fixture: adjacency, separated repeats, crossings.
    fixture = np.array([[1,1,2,1,1],[1,1,2,1,1],[3,3,4,3,3],[1,1,2,1,1]])
    assert np.array_equal(_compress(fixture), np.array([[1,2,1],[3,4,3],[1,2,1]]))
    # Hand-calculated composition regressions, unrelated to sampled layouts:
    # two-cell export into a band, and zero outward clearance. Literal expected
    # grids check clearing, clipping, frame contraction and band crossings.
    for visible in (False, True):
        inp = np.zeros((7, 9), dtype=np.int8)
        inp[:, 7] = 8
        inp[1 if visible else 5, :] = 2
        tip = 3 if visible else 0
        inp[tip, 4] = 4
        inp[tip+1:tip+4, 3] = inp[tip+1:tip+4, 5] = 4
        inp[tip+1:tip+3, 4] = 6
        rows = (['0000080','2262222','0060080','0040080','0404080']
                if visible else
                ['0040080','0404080','0000080','2222222','0000080'])
        expected = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
        for rotation in range(4):
            for offset in (0, 3):
                transformed = (np.rot90(inp, rotation) + offset) % 10
                before = transformed.copy()
                target = (np.rot90(expected, rotation) + offset) % 10
                assert np.array_equal(solve(transformed), target)
                assert np.array_equal(transformed, before), "solve mutated its input"
    return dict(seeds=42, distinct_puzzles=len(fingerprints), input_shapes=len(shapes),
                output_shapes=len(outputs), frame_profiles=len(profiles),
                payload_counts=sorted(counts), directions=sorted(directions),
                backgrounds=sorted(backgrounds), object_counts=sorted(inventories),
                mixed_direction_pairs=mixed, clipped_rays=clipped,
                separated_repeat_pairs=separated, crossing_rays=crossings,
                legal_count_combinations=legal_calls,
                hand_calculated_regressions=16,
                limitations="Generator-owned checks; no independent solver or blind prediction.")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
