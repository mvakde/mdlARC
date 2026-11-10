"""Corner loom: reflected stencils are the pulses of a shared run-length clock.

Standard library + NumPy. Generation paints translated stencils; input-only
prediction instead tests cell membership algebraically. Neither imports a
parent generator nor reads any evidence at runtime.
"""
from __future__ import annotations

import math
import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_HOST_SEEDS = (0, 624951095, 547557781, 46861200, 634772752, 993890622,
               794404067, 289476094, 103662753, 625085105, 89147527,
               615709849, 864932072, 485556099, 622109568, 363625017,
               291918030, 940876304, 233369604, 334287753, 553354452,
               249190686, 717927584, 583704446, 910622422, 156343632,
               821287120, 82844015, 7782157, 152692842, 22137595,
               222019011, 81823729, 430407374, 958716093, 431154680,
               160441918, 288394020, 80030683, 162140418, 616392118,
               589728225)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                               ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int64, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(points):
    points = set(points)
    seen = {min(points)}
    todo = list(seen)
    while todo:
        u, v = todo.pop()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                p = (u + du, v + dv)
                if p in points and p not in seen:
                    seen.add(p)
                    todo.append(p)
    return seen == points


def _corners(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _parse_candidate(a, bg, frame):
    ys, xs = np.nonzero(a == frame)
    assert len(ys) >= 16
    t, l, b, r = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert b - t >= 6 and r - l >= 6
    assert all(y in (t, b) or x in (l, r) for y, x in zip(ys, xs))
    assert np.all(a[t, l + 1:r] == frame) and np.all(a[b, l + 1:r] == frame)
    assert np.all(a[t + 1:b, l] == frame) and np.all(a[t + 1:b, r] == frame)
    rect = (t, l, b, r)
    corners = _corners(rect)
    inside = a[t + 1:b, l + 1:r]
    reference = [q for q, p in enumerate(corners)
                 if a[p] not in (bg, frame) and np.any(inside == a[p])]
    assert len(reference) == 1
    ref = reference[0]
    ry, rx = corners[ref]
    blueprint = int(a[ry, rx])
    dy, dx = _DIRS[ref]
    template_cells = {tuple(map(int, p)) for p in np.argwhere(a == blueprint)} - {(ry, rx)}
    stencil = frozenset(((y - ry) * dy, (x - rx) * dx) for y, x in template_cells)
    assert 3 <= len(stencil) <= 12 and _connected(stencil)
    assert all(1 <= u <= 4 and 1 <= v <= 4 for u, v in stencil)
    launches = [q for q, p in enumerate(corners) if q != ref and a[p] != frame]
    assert launches and all(a[corners[q]] not in (bg, blueprint) for q in launches)
    markers = {int(a[corners[q]]) for q in launches}
    assert len(markers) == 1
    marker = markers.pop()
    assert {tuple(map(int, p)) for p in np.argwhere(a == marker)} == {corners[q] for q in launches}
    roles = {bg, frame, blueprint, marker}
    assert len(roles) == 4
    exterior = [(int(y), int(x)) for y, x in np.argwhere(a != bg)
                if not (t <= y <= b and l <= x <= r)]
    assert exterior
    horizontal = all(y == ry and (x - rx) * dx < 0 for y, x in exterior)
    vertical = all(x == rx and (y - ry) * dy < 0 for y, x in exterior)
    assert horizontal != vertical
    step = (0, -dx) if horizontal else (-dy, 0)
    length = (rx if step[1] == -1 else a.shape[1] - 1 - rx) if horizontal else (
        ry if step[0] == -1 else a.shape[0] - 1 - ry)
    runs = []
    d = 1
    while d <= length:
        color = int(a[ry + step[0] * d, rx + step[1] * d])
        if color == bg:
            d += 1
            continue
        assert color not in roles
        start = d
        while d <= length and a[ry + step[0] * d, rx + step[1] * d] == color:
            d += 1
        runs.append((d - start, color, start))
    assert 1 <= len(runs) <= 4 and all(1 <= period <= 8 for period, _, _ in runs)
    # All colored interior clues must be partial first copies, in the nearest
    # source's color. Geometry, not connected-component/color identity, assigns them.
    first = runs[0][1]
    clue_domain = set()
    clue_counts = []
    for q in launches:
        ay, ax = corners[q]
        sy, sx = _DIRS[q]
        points = {(ay + sy * u, ax + sx * v) for u, v in stencil}
        assert all(t < y < b and l < x < r for y, x in points)
        clue_domain |= points
        clue_counts.append(sum(int(a[p]) == first for p in points))
    accents = {(y, x) for y in range(t + 1, b) for x in range(l + 1, r)
               if int(a[y, x]) not in (bg, blueprint)}
    assert accents <= clue_domain and all(int(a[p]) == first for p in accents)
    assert all(1 <= n < len(stencil) for n in clue_counts)
    # No unknown non-background pixels are allowed inside the bounding rectangle.
    assert all(int(a[p]) in roles for p in corners)
    return dict(background=bg, frame=frame, blueprint=blueprint, marker=marker,
                rect=rect, reference=ref, corners=corners, launches=tuple(launches),
                stencil=stencil, template_cells=template_cells, runs=tuple(runs),
                code_axis="row" if horizontal else "column", clue_counts=tuple(clue_counts))


def _analyze(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely dominant"
    bg = int(counts.argmax())
    candidates = []
    for frame in range(10):
        if frame == bg:
            continue
        try:
            candidates.append(_parse_candidate(a, bg, frame))
        except AssertionError:
            pass
    assert len(candidates) == 1, "expected one unambiguous loom"
    return a, candidates[0]


def _prediction(a, info, alternative=None):
    """Cell-wise membership equation; no generated geometry or outputs are read."""
    out = a.copy()
    t, l, b, r = info["rect"]
    runs = list(info["runs"])
    if alternative == "short":
        runs.sort(key=lambda run: (run[0], run[2]))
    elif alternative == "long":
        runs.sort(key=lambda run: (-run[0], run[2]))
    elif alternative == "far":
        runs.reverse()
    stencil = info["stencil"] if alternative != "point" else {(1, 1)}
    for y in range(t + 1, b):
        for x in range(l + 1, r):
            if (y, x) in info["template_cells"]:
                continue
            clocks = []
            for q in info["launches"]:
                ay, ax = info["corners"][q]
                dy, dx = _DIRS[q]
                u, v = (y - ay) * dy, (x - ax) * dx
                local = stencil
                if alternative == "rotation" and (q ^ info["reference"]) in (1, 2):
                    local = {(bb, aa) for aa, bb in stencil}
                clocks.extend(u - aa for aa, bb in local if u - aa == v - bb and u - aa >= 0)
            for period, color, near in runs:
                if alternative == "first_only":
                    fires = 0 in clocks
                elif alternative == "solid":
                    fires = bool(clocks)
                else:
                    p = max(1, period + (1 if alternative == "period_plus" else
                                        -1 if alternative == "period_minus" else 0))
                    phase = near - 1 if alternative == "gap_phase" else 1 if alternative == "phase_one" else 0
                    fires = any(z >= phase and (z - phase) % p == 0 for z in clocks)
                if fires:
                    out[y, x] = color
                    break
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the frame, reference stencil, launches and clock from this input."""
    a, info = _analyze(input_grid)
    return _prediction(a, info)


def _sample_stencil(rng, radius, asymmetric=False):
    domain = {(u, v) for u in range(1, radius + 1) for v in range(1, radius + 1)}
    for _ in range(80):
        size = int(rng.integers(3, min(12, radius * radius) + 1))
        # This extreme diagonal has a singleton cross-section: recurrence can
        # be observed without being filled by another cell of the same stamp.
        points = {(1, radius)}
        while len(points) < size:
            frontier = sorted(p for p in domain - points if any(
                max(abs(p[0] - u), abs(p[1] - v)) <= 1 for u, v in points))
            points.add(frontier[int(rng.integers(len(frontier)))])
        if not asymmetric or points != {(v, u) for u, v in points}:
            return frozenset(points)
    return frozenset({(1, radius), (1, radius - 1), (2, radius)})


def _stamp_output(base, rect, stencil, runs, launches, template_cells):
    """Construction uses translated stamp sets and overwrite order."""
    out = base.copy()
    t, l, b, r = rect
    corners = _corners(rect)
    for period, color, _ in reversed(runs):
        for q in launches:
            ay, ax = corners[q]
            dy, dx = _DIRS[q]
            for tick in range(0, min(b - t, r - l), period):
                for u, v in stencil:
                    p = (ay + dy * (u + tick), ax + dx * (v + tick))
                    if t < p[0] < b and l < p[1] < r and p not in template_cells:
                        out[p] = color
    return out


def _make_pair(rng, roles, clock_colors, lesson=None, shift=None, test=False):
    bg, frame, blueprint, marker = roles
    radius = 4 if test and shift == "stencil" else int(rng.choice((2, 3)))
    stencil = _sample_stencil(rng, radius, asymmetric=lesson is not None)
    if lesson is not None:
        periods = (2, 3) if lesson == 0 else (3, 2)
        colors = clock_colors if lesson == 0 else clock_colors[::-1]
    else:
        count = int(rng.integers(3, 5)) if test and shift == "clock" else int(rng.integers(1, 3))
        # A real source-space budget, rather than a fixed menu of length tuples.
        periods = []
        budget = 11 if count >= 3 else 10
        for i in range(count):
            p = int(rng.integers(1, min(6, budget - sum(periods) - (count - i - 1)) + 1))
            periods.append(p)
        palette = [c for c in range(10) if c not in roles]
        colors = [int(rng.choice(palette)) for _ in periods]
    gaps = [int(rng.integers(0, 3)) for _ in periods]
    if lesson is not None:
        gaps[0] = 1
    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            gaps[i] = max(1, gaps[i])
    # Keep source gaps varied but leave enough room for visible repetition.
    while sum(gaps) + sum(periods) > 14:
        i = next((i for i in reversed(range(len(gaps))) if gaps[i] > (int(i > 0 and colors[i] == colors[i - 1]))), None)
        if i is None:
            raise AssertionError("unpackable source")
        gaps[i] -= 1
    outside = sum(periods) + sum(gaps) + int(rng.integers(0, 3))
    trailing = int(rng.integers(0, 3))
    min_size = max(9, radius + max(periods) + 3)
    if lesson is not None:
        min_size = max(min_size, math.lcm(*periods) + radius + 3)
    max_width = 30 - outside - trailing
    assert min_size <= max_width
    fw = int(rng.integers(min_size, max_width + 1))
    top, bottom = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    fh = int(rng.integers(min_size, min(25, 30 - top - bottom) + 1))
    shape = (top + fh + bottom, outside + fw + trailing)
    rect = (top, outside, top + fh - 1, outside + fw - 1)
    t, l, b, r = rect
    inp = np.full(shape, bg, dtype=np.int8)
    inp[t, l:r + 1] = inp[b, l:r + 1] = frame
    inp[t:b + 1, l] = inp[t:b + 1, r] = frame
    inp[t, l] = blueprint
    template_cells = {(t + u, l + v) for u, v in stencil}
    for p in template_cells:
        inp[p] = blueprint
    count = 3 if test and shift == "launches" else (1 if lesson is not None else int(rng.integers(1, 3)))
    if lesson is not None:
        launches = (int(rng.choice((1, 2))),)
    else:
        launches = tuple(sorted(int(q) for q in rng.choice((1, 2, 3), count, replace=False)))
    corners = _corners(rect)
    for q in launches:
        inp[corners[q]] = marker
    runs, cursor = [], 1
    for period, color, gap in zip(periods, colors, gaps):
        cursor += gap
        runs.append((int(period), int(color), cursor))
        inp[t, l - cursor - period + 1:l - cursor + 1] = color
        cursor += period
    out = _stamp_output(inp, rect, stencil, runs, launches, template_cells)
    # Partial first copies provide B-style repair evidence. They never serve as
    # hidden templates; the differently colored complete reference does that.
    for q in launches:
        ay, ax = corners[q]
        dy, dx = _DIRS[q]
        first_points = sorted((ay + dy * u, ax + dx * v) for u, v in stencil)
        retained = int(rng.integers(1, len(first_points)))
        for i in rng.choice(len(first_points), retained, replace=False):
            inp[first_points[int(i)]] = runs[0][1]
    if rng.random() < .5:
        inp, out = inp.T.copy(), out.T.copy()
    turn = int(rng.integers(4))
    return {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}


_RIVALS = ("short", "long", "far", "point", "first_only", "solid",
           "rotation", "gap_phase", "phase_one", "period_plus", "period_minus")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    clock_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    available = [c for c in range(10) if c not in clock_colors]
    roles = tuple(map(int, rng.choice(available, 4, replace=False)))
    stable_roles = bool(rng.random() < .6)
    shift = str(rng.choice(("launches", "stencil", "clock")))
    lesson_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    lessons = {lesson_slots[0]: 0, lesson_slots[1]: 1}
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = lessons.get(index) if split == "train" else None
            for _ in range(100):
                pair_roles = roles if stable_roles else tuple(map(int, rng.choice(available, 4, replace=False)))
                try:
                    pair = _make_pair(rng, pair_roles, clock_colors, lesson, shift, split == "test")
                    a, info = _analyze(pair["input"])
                    assert _key(a) not in seen
                    assert np.array_equal(_prediction(a, info), pair["output"])
                    assert np.any(a != pair["output"])
                    assert not np.array_equal(_prediction(a, info, "point"), pair["output"])
                    assert not np.array_equal(_prediction(a, info, "first_only"), pair["output"])
                    if lesson is not None:
                        for rival in (("long", "far") if lesson == 0 else ("short", "far")) + ("rotation", "solid", "gap_phase", "phase_one", "period_plus", "period_minus"):
                            assert not np.array_equal(_prediction(a, info, rival), pair["output"])
                        interior = pair["output"][info["rect"][0] + 1:info["rect"][2],
                                                   info["rect"][1] + 1:info["rect"][3]]
                        assert all(color in interior for color in clock_colors)
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded loom construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, fitted = set(), set(_RIVALS)
    wins = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, info = _analyze(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            expected = _prediction(a, info)
            assert np.array_equal(expected, out), "clock, footprint, clipping, priority or preservation failed"
            assert np.any(a != out)
            t, l, b, r = info["rect"]
            changed = a != out
            change_y, change_x = np.nonzero(changed)
            assert np.all((t < change_y) & (change_y < b) & (l < change_x) & (change_x < r))
            assert not any(changed[p] for p in info["template_cells"])
            assert set(map(int, np.unique(a))) == set(map(int, np.unique(out)))
            for alt in ("point", "first_only"):
                assert not np.array_equal(_prediction(a, info, alt), out), "one parent has no visible effect"
            infos[split].append(info)
            if split == "train":
                fitted = {alt for alt in fitted if np.array_equal(_prediction(a, info, alt), out)}
                if len(info["runs"]) >= 2:
                    p0, c0, _ = info["runs"][0]
                    for p1, c1, _ in info["runs"][1:]:
                        if c0 != c1 and c0 in out[t + 1:b, l + 1:r] and c1 in out[t + 1:b, l + 1:r]:
                            wins.add((c0, c1, p0 < p1, p0 > p1))
    assert not fitted, f"simpler rules fit every demonstration: {sorted(fitted)}"
    assert any(w[2] for w in wins) and any(w[3] for w in wins)
    assert any(a == d and b == c for a, b, _, _ in wins for c, d, _, _ in wins), "fixed color priority is not refuted"
    # The three visible, compatible shifts are alternatives in distribution,
    # never changes of the rule. At least one must actually occur.
    assert (max(len(i["launches"]) for i in infos["test"]) > max(len(i["launches"]) for i in infos["train"]) or
            max(max(max(p) for p in i["stencil"]) for i in infos["test"]) > max(max(max(p) for p in i["stencil"]) for i in infos["train"]) or
            max(len(i["runs"]) for i in infos["test"]) > max(len(i["runs"]) for i in infos["train"]))
    return None


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def _interaction_features(info):
    """Measure the composition's collisions, independently of literal colors."""
    t, l, b, r = info["rect"]
    coverage = {}
    clipped = False
    for rank, (period, color, _) in enumerate(info["runs"]):
        for q in info["launches"]:
            ay, ax = info["corners"][q]
            dy, dx = _DIRS[q]
            for tick in range(0, min(b - t, r - l), period):
                positions = {(ay + dy * (u + tick), ax + dx * (v + tick))
                             for u, v in info["stencil"]}
                visible = {p for p in positions if t < p[0] < b and l < p[1] < r}
                clipped |= bool(visible) and visible != positions
                for p in visible - info["template_cells"]:
                    coverage.setdefault(p, []).append((rank, q, tick, color))
    cross_corner = any(len({entry[1] for entry in entries}) > 1 for entries in coverage.values())
    offset_collision = any(any(a[0] != bb[0] and a[2] != bb[2] and a[3] != bb[3]
                               for a in entries for bb in entries)
                           for entries in coverage.values())
    return cross_corner, offset_collision, clipped


def self_test():
    """Bounded semantic regressions, metamorphisms and 42-seed diversity checks."""
    # A hand-specified asymmetric two-pulse fixture, not an output from solve.
    inp = np.zeros((12, 16), dtype=np.int8)
    rect = (1, 6, 10, 15)
    t, l, b, r = rect
    inp[t, l:r + 1] = inp[b, l:r + 1] = 1
    inp[t:b + 1, l] = inp[t:b + 1, r] = 1
    inp[t, l] = 8
    stencil = {(1, 1), (1, 2), (2, 2)}
    for u, v in stencil:
        inp[t + u, l + v] = 8
    inp[t, r] = 9
    inp[t, 0:3] = 3
    inp[t, 4:6] = 2
    inp[t + 1, r - 1] = 2
    expected = inp.copy()
    # Nearest period 2 paints ticks 0,2,4,6; far period 3 paints
    # 0,3,6. Exact coordinates below include shape-overlap collisions.
    expected[2, 13:15] = 2
    expected[3, 13] = 2
    expected[4, 11:13] = 2
    expected[5, 11] = 2
    expected[5, 10] = 3
    expected[6, 10:11] = 3
    expected[6, 9:11] = 2
    expected[7, 9] = 2
    expected[8, 7:9] = 2
    expected[9, 7] = 2
    assert np.array_equal(solve(inp), expected), "hand-computed overlapping stencil fixture"
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn))
        assert np.array_equal(solve(np.rot90(inp.T, turn)), np.rot90(expected.T, turn))
    mapping = np.array((7, 0, 6, 4, 5, 2, 3, 1, 9, 8))
    assert np.array_equal(solve(mapping[inp]), mapping[expected])
    # Periods seven and eight are specified family members beyond the ordinary
    # sampler's cap of six. Their first recurrences overlap at different ticks.
    large = np.zeros((16, 30), dtype=np.int8)
    large[1, 16:30] = large[14, 16:30] = 1
    large[1:15, 16] = large[1:15, 29] = 1
    large[1, 16] = 8
    large[2, 17:19] = 8
    large[3, 18] = 8
    large[1, 29] = 9
    large[1, 8:16] = 2
    large[1, 0:7] = 3
    large[2, 28] = 2
    large_expected = large.copy()
    large_expected[2, 27:29] = 2
    large_expected[3, 27] = 2
    large_expected[9, 20:22] = 3
    large_expected[10, 19:21] = 2
    large_expected[11, 19] = 2
    assert np.array_equal(solve(large), large_expected)
    puzzles = [generate_puzzle(seed) for seed in _HOST_SEEDS]
    features, shapes, stencils, layouts, signatures = [], set(), set(), set(), set()
    interactions = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, info = _analyze(pair["input"])
                shapes.add(a.shape)
                stencils.add(tuple(sorted(info["stencil"])))
                layouts.add((info["rect"], info["launches"], info["runs"], tuple(sorted(info["stencil"]))))
                features.append(info)
                interactions.append(_interaction_features(info))
                assert np.array_equal(solve(a.copy()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 35 and len(stencils) >= 35 and len(layouts) >= 150
    assert {i["reference"] for i in features} == set(range(4))
    assert {i["code_axis"] for i in features} == {"row", "column"}
    assert {len(i["launches"]) for i in features} == {1, 2, 3}
    assert {len(i["runs"]) for i in features} == {1, 2, 3, 4}
    assert {i["background"] for i in features} == set(range(10))
    assert {p for i in features for p, _, _ in i["runs"]} >= set(range(1, 7))
    assert any(len({c for _, c, _ in i["runs"]}) < len(i["runs"]) for i in features)
    assert any(len(i["launches"]) >= 2 for i in features)
    assert all(any(flags[j] for flags in interactions) for j in range(3)), "cross-corner, offset collision, clipping coverage"
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
                 for s in ("train", "test")}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("corner loom self-test passed")
