"""Expandable diagonal reservoir. Standard library + NumPy; no parent imports.

All semantic parsing and solving use the current input only. No files are read
or written by generation, validation, or self_test. Diagnostic alternatives
are deliberately weaker rules, not independent semantic certification.
"""
from collections import deque
import hashlib

import numpy as np


HOST_SEEDS = (0, 740683202, 389391502, 671998534, 404297109, 553005854,
              991519284, 403267960, 921750589, 704323761, 344871372,
              68412660, 537535974, 636031115, 38310413, 383113772,
              477525852, 703950528, 606709251, 552210055, 233966136,
              904567584, 827258662, 666326252, 791689350, 107746783,
              913472799, 189476692, 95151358, 201561434, 750396798,
              721193447, 34104315, 297536640, 697665999, 759380880,
              562986593, 395034981, 479455311, 752491460, 546387857,
              877574241)
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ALTERNATIVES = ("dock_only", "full", "all_brackets", "pixel_centers",
                "consecutive", "dock_distance", "manhattan")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require integer counts 2..10 / 1..4, with num_train > num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "Invalid colors/dtype")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _stroke_candidates(mask):
    results = []
    for slope in (-1, 1):
        lines = {}
        for r, c in np.argwhere(mask):
            r, c = int(r), int(c)
            lines.setdefault(c-slope*r, []).append(r)
        strokes, valid = [], True
        for q, rows in lines.items():
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(r)
            if any(len(run) < 2 for run in runs):
                valid = False
                break
            strokes.extend((q, 2*run[0]+slope*q, 2*run[-1]+slope*q) for run in runs)
        if valid and len(strokes) >= 2:
            results.append((slope, sorted(strokes)))
    return results


def _parse(grid):
    """Dock motif encodes structural roles; the fourth color is the supply."""
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 4 and counts.max() > grid.size//2, "Need four roles and a majority field")
    bg = int(values[np.argmax(counts)])
    h, w = grid.shape
    candidates = []
    for r in range(h-2):
        for c in range(w-2):
            patch = grid[r:r+3, c:c+3]
            guide, wall = int(patch[0, 0]), int(patch[1, 1])
            if guide == bg or wall in (guide, bg):
                continue
            if np.count_nonzero(patch == guide) != 8:
                continue
            dock = np.zeros(grid.shape, bool)
            dock[r:r+3, c:c+3] = True
            guide_mask = (grid == guide) & ~dock
            recognized = _stroke_candidates(guide_mask)
            if len(recognized) != 1:
                continue
            rest = set(int(x) for x in values)-{bg, guide, wall}
            if len(rest) != 1:
                continue
            slope, strokes = recognized[0]
            candidates.append(dict(bg=bg, guide=guide, wall=wall,
                                   payload=rest.pop(), dock=dock, pos=(r, c),
                                   guides=guide_mask, walls=(grid == wall) & ~dock,
                                   slope=slope, strokes=strokes))
    _require(len(candidates) == 1, "Need one uniquely recognizable marked dock and guide slope")
    return candidates[0]


def _consecutive_pairs(strokes):
    remaining = set(range(len(strokes)))
    pairs = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component = [first]
        for i in component:
            for j in sorted(remaining):
                if max(strokes[i][1], strokes[j][1]) <= min(strokes[i][2], strokes[j][2]):
                    remaining.remove(j)
                    component.append(j)
        ordered = sorted(component, key=lambda i: (strokes[i][0], strokes[i][1]))
        pairs.extend(zip(ordered, ordered[1:]))
    return pairs


def _bracket(shape, slope, strokes, alternative=""):
    rows, cols = np.indices(shape)
    q, t = cols-slope*rows, rows+slope*cols
    result = np.zeros(shape, bool)
    if alternative in ("pixel_centers", "consecutive"):
        pairs = (_consecutive_pairs(strokes) if alternative == "consecutive" else
                 [(i, j) for i in range(len(strokes)) for j in range(i+1, len(strokes))])
        for i, j in pairs:
            a, b = strokes[i], strokes[j]
            m = ((t >= max(a[1], b[1])) & (t <= min(a[2], b[2])) &
                 (q > min(a[0], b[0])) & (q < max(a[0], b[0])))
            if alternative == "pixel_centers":
                m &= ((t-a[0]) % 2 == 0) & ((t-b[0]) % 2 == 0)
            result |= m
    else:
        for level in range(int(t.min()), int(t.max())+1):
            active = [a for a, lo, hi in strokes if lo <= level <= hi]
            if len(active) >= 2:
                result |= (t == level) & (q > min(active)) & (q < max(active))
    return result


def _distances(passable, starts):
    h, w = passable.shape
    distance = np.full(passable.shape, -1, np.int16)
    queue = deque((int(r), int(c)) for r, c in np.argwhere(starts))
    distance[starts] = 0
    while queue:
        r, c = queue.popleft()
        nd = int(distance[r, c])+1
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = nd
                queue.append((nr, nc))
    return distance


def _reservoir(grid, info, alternative=""):
    if alternative == "dock_only":
        return info["dock"].copy()
    bracket = _bracket(grid.shape, info["slope"], info["strokes"], alternative)
    eligible = (bracket & ~info["guides"] & ~info["walls"]) | info["dock"]
    if alternative == "all_brackets":
        return eligible
    return _distances(eligible, info["dock"]) >= 0


def _state(grid, info=None):
    info = _parse(grid) if info is None else info
    target = _reservoir(grid, info)
    passable = ~(info["guides"] | info["walls"])
    distance = _distances(passable, target)
    return info, target, passable, distance


def _select(grid, info, target, distance, strict=True):
    source = (grid == info["payload"]) & ~target & (distance >= 0)
    candidates = [int(i) for i in np.flatnonzero(source)]
    candidates.sort(key=lambda i: (int(distance.flat[i]), i))
    capacity = int(target.sum())
    if strict and len(candidates) > capacity:
        _require(distance.flat[candidates[capacity-1]] < distance.flat[candidates[capacity]],
                 "Ambiguous capacity-frontier distance tie")
    selected = np.zeros(grid.shape, bool)
    selected.flat[candidates[:capacity]] = True
    return selected, len(candidates)


def _prediction(grid, info=None, alternative=""):
    info = _parse(grid) if info is None else info
    target = _reservoir(grid, info, alternative)
    passable = ~(info["guides"] | info["walls"])
    starts = info["dock"] if alternative == "dock_distance" else target
    distance = _distances(np.ones(grid.shape, bool) if alternative == "manhattan" else passable, starts)
    if alternative == "manhattan":
        # Isolate geometric ranking from reachability: unreachable packets are
        # excluded under both rules, as in the reviewed parent distinction.
        distance[_distances(passable, target) < 0] = -1
    selected, _ = _select(grid, info, target, distance, strict=not alternative)
    result = grid.copy()
    result[selected] = info["bg"]
    n = int(target.sum()) if alternative == "full" else int(selected.sum())
    result.flat[np.flatnonzero(target)[:n]] = info["payload"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover roles, reservoir, routes, capacity, and prefix from one input."""
    grid = _grid(input_grid)
    info, target, _, _ = _state(grid)
    bracket = _bracket(grid.shape, info["slope"], info["strokes"])
    _require(np.all(bracket[info["dock"]]), "Dock must lie inside the bracket")
    _require(not np.any(target & (grid == info["payload"])), "Reservoir must initially be empty")
    return _prediction(grid, info)


def _differences(pair, info=None):
    grid, expected = pair["input"], pair["output"]
    info = _parse(grid) if info is None else info
    return {name: not np.array_equal(expected, _prediction(grid, info, name)) for name in ALTERNATIVES}


def pair_features(pair):
    """All measurements come from the returned grids, not generation metadata."""
    grid = _grid(pair["input"])
    info, target, passable, distance = _state(grid)
    selected, reachable = _select(grid, info, target, distance)
    source = grid == info["payload"]
    retained = source & (distance >= 0) & ~selected
    direct = _distances(np.ones(grid.shape, bool), target)
    bracket = _bracket(grid.shape, info["slope"], info["strokes"])
    return dict(shape=list(grid.shape), slope=info["slope"],
                strokes=len(info["strokes"]), capacity=int(target.sum()),
                selected=int(selected.sum()), reachable=reachable,
                excess=max(0, reachable-int(target.sum())),
                unreachable=int(np.count_nonzero(source & (distance < 0))),
                walls=int(info["walls"].sum()),
                detached=int(np.count_nonzero(bracket & passable & ~target)),
                dock=list(info["pos"]), palette=[info[k] for k in ("bg", "guide", "wall", "payload")],
                max_distance=int(distance.max()),
                path_reversal=bool(selected.any() and retained.any() and direct[selected].max() > direct[retained].min()),
                alternatives=_differences(pair, info))


def _interval(rng, q, low, high):
    low += (q-low) % 2
    high -= (high-q) % 2
    return int(q), int(low), int(high)


def _abstract_specs(rng, short_interior=False, separate=False):
    count = int(rng.integers(3 if short_interior else 2, 6))
    if count == 2:
        qs = [0, int(rng.integers(10, 17))]
    else:
        gaps = [int(x) for x in rng.integers(3, 7, size=count-1)]
        while sum(gaps) < 10:
            gaps[int(rng.integers(len(gaps)))] += 1
        qs = [0]+list(np.cumsum(gaps).astype(int))
    specs = []
    for index, q in enumerate(qs):
        if index in (0, count-1):
            low, high = int(rng.integers(-10, -5)), int(rng.integers(8, 14))
        else:
            low = int(rng.integers(-9, 4))
            high = low+2*int(rng.integers(2, 10))
        specs.append(_interval(rng, int(q), low, high))
    if short_interior:
        i = int(rng.integers(1, count-1))
        low = int(rng.integers(-2, 3))
        low += (qs[i]-low) % 2
        specs[i] = (int(qs[i]), low, low+2*int(rng.integers(1, 3)))
    primary = len(specs)
    if separate:
        # Earlier in longitudinal order, so a partial prefix can distinguish
        # the rooted receiver from every bracket on the canvas.
        high = min(lo for _, lo, _ in specs)-int(rng.integers(4, 7))
        width = int(rng.integers(5, 11))
        startq = int(rng.integers(-3, 6))
        for q in (startq, startq+width):
            low = high-2*int(rng.integers(2, 5))
            specs.append(_interval(rng, q, low, high))
    return specs, primary


def _wall_geometry(rng, grid, bg, guide, wall, dock, raw_bracket):
    h, w = grid.shape
    wm = np.zeros(grid.shape, bool)
    # Segments and turns cross both the receiving area and surrounding field.
    for _ in range(int(rng.integers(1, 7))):
        r, c = int(rng.integers(h-2)), int(rng.integers(w-2))
        r1, c1 = int(rng.integers(r+2, h+1)), int(rng.integers(c+2, w+1))
        kind = int(rng.integers(4))
        if kind != 1:
            wm[r, c:c1] = True
        if kind != 0:
            wm[r:r1, c] = True
        if kind == 3:
            wm[r:r1, c1-1] = True
    # A closed pocket supplies genuine unreachable distraction, independently
    # of how many reachable packets remain after capacity is reached.
    for _ in range(32):
        r, c = int(rng.integers(h-4)), int(rng.integers(w-4))
        hh, ww = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        if r+hh > h or c+ww > w:
            continue
        patch = np.s_[r:r+hh, c:c+ww]
        if np.any(raw_bracket[patch] | dock[patch] | (grid[patch] == guide)):
            continue
        wm[patch] = False
        wm[r, c:c+ww] = wm[r+hh-1, c:c+ww] = True
        wm[r:r+hh, c] = wm[r:r+hh, c+ww-1] = True
        break
    wm &= ~dock & (grid != guide)
    grid[wm] = wall


def _geometry(rng, slope, palette, test, short_interior, separate):
    bg, guide, wall, payload = palette
    specs, primary = _abstract_specs(rng, short_interior, separate)
    cells = []
    groups = []
    for q, lo, hi in specs:
        run = [((t-slope*q)//2, (q+slope*t)//2) for t in range(lo, hi+1, 2)]
        groups.append(run)
        cells.extend(run)
    coords = np.array(cells, dtype=int)
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    spans = upper-lower+1
    cap = 30 if test else 26
    if np.any(spans > cap):
        return None
    floor = 27 if test else 17
    shape = tuple(int(rng.integers(max(floor, int(n)), cap+1)) for n in spans)
    shift = np.array([int(rng.integers(shape[i]-int(spans[i])+1)) for i in range(2)])-lower
    grid = np.full(shape, bg, dtype=np.int8)
    visible = []
    for run in groups:
        pts = np.array(run)+shift
        grid[pts[:, 0], pts[:, 1]] = guide
        q = int(pts[0, 1]-slope*pts[0, 0])
        ts = pts[:, 0]+slope*pts[:, 1]
        visible.append((q, int(ts.min()), int(ts.max())))
    raw = _bracket(shape, slope, visible)
    primary_raw = _bracket(shape, slope, visible[:primary])
    slots = raw & primary_raw & (grid == bg)
    possible = []
    for r in range(shape[0]-2):
        for c in range(shape[1]-2):
            if slots[r:r+3, c:c+3].all():
                possible.append((r, c))
    if not possible:
        return None
    r, c = possible[int(rng.integers(len(possible)))]
    dock = np.zeros(shape, bool)
    dock[r:r+3, c:c+3] = True
    _wall_geometry(rng, grid, bg, guide, wall, dock, raw)
    grid[dock] = guide
    grid[r+1, c+1] = wall
    # Temporary fourth-role sentinel lets the exact same parser inspect the
    # finished geometry. It is removed before source placement.
    choices = np.flatnonzero((grid == bg) & ~dock & ~raw)
    if not len(choices):
        return None
    sentinel = int(choices[0])
    grid.flat[sentinel] = payload
    try:
        info = _parse(grid)
    except AssertionError:
        return None
    grid.flat[sentinel] = bg
    info, target, passable, distance = _state(grid, info)
    if not 12 <= target.sum() <= 110:
        return None
    return grid, info, target, passable, distance


def _pick(rng, pool, count, style, forced=()):
    cells = np.array(sorted(set(int(i) for i in pool)-set(forced)), dtype=int)
    chosen = list(forced)
    if len(cells) < count-len(chosen):
        return None
    if count == len(chosen):
        return chosen
    if style == "dispersed":
        chosen.extend(int(i) for i in rng.choice(cells, count-len(chosen), replace=False))
    else:
        # A compact or straight-biased supply can be split by the frontier.
        width = style[1]
        kind = style[0]
        anchor = chosen[0] if chosen else int(rng.choice(cells))
        ar, ac = divmod(anchor, width)
        rr, cc = cells//width, cells % width
        score = (rr-ar)**2+(cc-ac)**2 if kind == "compact" else np.abs(rr-ar)*width+np.abs(cc-ac)
        order = np.lexsort((rng.random(len(cells)), score))
        chosen.extend(int(i) for i in cells[order[:count-len(chosen)]])
    return chosen


def _sources(rng, grid, info, target, passable, distance, mode, require_path):
    capacity = int(target.sum())
    available = passable & ~target
    active = available & (distance >= 0)
    inactive = available & (distance < 0)
    styles = ["dispersed", ("compact", grid.shape[1]), ("line", grid.shape[1])]
    style = styles[int(rng.integers(3))]
    if mode == "partial":
        n = int(rng.choice([2, 4]))
    elif mode == "zero":
        n = 0
    elif mode == "full":
        n = capacity
    elif mode == "excess":
        n = capacity+int(rng.integers(1, 19))
    else:
        n = int(rng.integers(1, capacity+1))
    if active.sum() < n or (n == 0 and not inactive.any()):
        return None
    if n <= capacity:
        chosen = _pick(rng, np.flatnonzero(active), n, style)
    else:
        direct = _distances(np.ones(grid.shape, bool), target)
        chosen = None
        anchors = rng.permutation(np.flatnonzero(active))
        for ai in anchors:
            a = int(ai)
            near = active & (distance <= distance.flat[a])
            far = active & (distance > distance.flat[a])
            if near.sum() < capacity or far.sum() < n-capacity:
                continue
            eligible = far & (direct < direct.flat[a]) if require_path else far
            if not eligible.any():
                continue
            b = int(rng.choice(np.flatnonzero(eligible)))
            left = _pick(rng, np.flatnonzero(near), capacity, style, [a])
            right = _pick(rng, np.flatnonzero(far), n-capacity, style, [b])
            chosen = left+right
            break
        if chosen is None:
            return None
    max_distractors = min(12, int(inactive.sum()))
    distractors = int(rng.integers(1 if n == 0 else 0, max_distractors+1)) if max_distractors else 0
    other = _pick(rng, np.flatnonzero(inactive), distractors, styles[int(rng.integers(3))])
    result = grid.copy()
    result.flat[chosen+other] = info["payload"]
    if np.count_nonzero(result == info["bg"]) <= result.size//2:
        return None
    return result


def _make_pair(rng, slope, palette, test, mode, required, seen):
    short = "consecutive" in required or (rng.random() < .5)
    separate = "all_brackets" in required or (rng.random() < (.45 if test else .3))
    for _ in range(240):
        geo = _geometry(rng, slope, palette, test, short, separate)
        if geo is None:
            continue
        grid, info, target, passable, distance = geo
        populated = _sources(rng, grid, info, target, passable, distance, mode,
                             "manhattan" in required)
        if populated is None or _key(populated) in seen:
            continue
        try:
            output = solve(populated)
            pair = {"input": populated, "output": output}
            differences = _differences(pair)
            if not all(differences[name] for name in required):
                continue
            if "manhattan" in required and not pair_features(pair)["path_reversal"]:
                continue
        except AssertionError:
            continue
        return pair
    raise RuntimeError("Bounded reservoir construction exhausted for " + repr(sorted(required)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    palette = [int(x) for x in rng.choice(10, 4, replace=False)]
    changing_palette = bool(rng.integers(2))
    slope = int(rng.choice([-1, 1]))
    puzzle, seen = {"train": [], "test": []}, set()

    def add(split, mode, required):
        colors = [int(x) for x in rng.choice(10, 4, replace=False)] if changing_palette else palette
        pair = _make_pair(rng, slope, colors, split == "test", mode, required, seen)
        puzzle[split].append(pair)
        seen.add(_key(pair["input"]))

    # Actual tests determine evidence obligations; their labels are never
    # visible to solve. Tests have no fixed load at a given index.
    for _ in range(num_test):
        add("test", str(rng.choice(["random", "excess", "zero", "full"], p=[.52, .35, .05, .08])), set())
    needed = {name for pair in puzzle["test"] for name, differs in _differences(pair).items() if differs}
    low = {"dock_only", "full"} | (needed & {"all_brackets", "pixel_centers"})
    high = {"manhattan"} | (needed & {"consecutive", "dock_distance"})
    requests = [("partial", low), ("excess", high)]
    requests.extend((str(rng.choice(["random", "excess", "zero", "full"], p=[.6, .25, .05, .1])), set())
                    for _ in range(num_train-2))
    for i in rng.permutation(num_train):
        add("train", *requests[int(i)])
    # Some required distinctions can be witnessed by either demonstration.
    # The fixed allocation above covers each diagnostic, not just mode names.
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Shape changed")
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            info, target, _, distance = _state(a)
            selected, reachable = _select(a, info, target, distance)
            _require(np.all(_bracket(a.shape, info["slope"], info["strokes"])[info["dock"]]), "Dock outside bracket")
            _require(np.all(a[target & ~info["dock"]] == info["bg"]), "Receiving slots not empty")
            expected = _prediction(a, info)
            _require(np.array_equal(expected, b), "Incorrect reservoir/path/capacity/prefix output")
            _require(np.count_nonzero(a == info["payload"]) == np.count_nonzero(b == info["payload"]), "Material not conserved")
            obstacles = info["guides"] | info["walls"]
            _require(np.array_equal(a[obstacles], b[obstacles]), "Structural cells changed")
            untouched = ~(selected | target)
            _require(np.array_equal(a[untouched], b[untouched]), "Unselected source or background changed")
            record = pair_features({"input": a, "output": b})
            stats[split].append(record)
            if not selected.any():
                _require(record["unreachable"] > 0 and np.array_equal(a, b), "Uninformative empty-supply case")
    train = stats["train"]
    _require(any(f["selected"] in (2, 4) and f["selected"] < f["capacity"] for f in train), "Need partial-prefix teaching")
    _require(any(f["excess"] > 0 and f["path_reversal"] for f in train), "Need strict path/capacity teaching")
    for name in ALTERNATIVES:
        if any(f["alternatives"][name] for f in stats["test"]):
            _require(any(f["alternatives"][name] for f in train), "Missing training witness: "+name)
    _require(any(f["alternatives"]["dock_only"] for f in train), "A contributes no visible evidence")
    _require(any(f["alternatives"]["full"] for f in train), "B contributes no visible evidence")
    return None


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for k in ("input", "output"):
                a = np.asarray(p[k], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds in memory; no saved puzzles, traces, or unbounded logs."""
    # Analytic fixture: q=-6,+6, t=-8..8, shifted by (7,8).
    # Nine even t levels each contain five slots; eight odd levels contain
    # six: capacity 93. Reading order begins (1,6),(1,7),(2,5).
    fixture = np.zeros((15, 17), np.int8)
    for q in (-6, 6):
        for t in range(-8, 9, 2):
            fixture[(t-q)//2+7, (q+t)//2+8] = 8
    fixture[6:9, 7:10] = 8
    fixture[7, 8] = 5
    fixture[0, 0] = fixture[0, 1] = fixture[14, 16] = 2
    expected = fixture.copy()
    expected[0, 0] = expected[0, 1] = expected[14, 16] = 0
    expected[1, 6] = expected[1, 7] = expected[2, 5] = 2
    _require(_reservoir(fixture, _parse(fixture)).sum() == 93, "Analytic diagonal capacity failed")
    _require(np.array_equal(solve(fixture), expected), "Analytic three-cell prefix failed")
    _require(not np.array_equal(_prediction(fixture, alternative="pixel_centers"), expected), "Parity fixture ineffective")
    # A wall through row 4 separates twelve eligible upper cells, removes
    # eight slots on the wall, and leaves a 73-cell rooted reservoir. Only
    # the lower source is reachable; the upper two must remain in place.
    divided = fixture.copy()
    divided[4, divided[4] != 8] = 5
    expected = divided.copy()
    expected[14, 16] = 0
    expected[5, 2] = 2
    _require(_reservoir(divided, _parse(divided)).sum() == 73, "Rooted capacity fixture failed")
    _require(np.array_equal(solve(divided), expected), "Rooted connectivity fixture failed")
    # A tie at the capacity frontier has no sanctioned queue-order answer.
    tied = np.zeros((7, 7), np.int8)
    target = np.zeros(tied.shape, bool)
    target[2:5, 2:5] = True
    tied.flat[:10] = 2
    distance = np.ones(tied.shape, np.int16)
    distance[target] = 0
    try:
        _select(tied, {"payload": 2}, target, distance)
    except AssertionError:
        pass
    else:
        raise AssertionError("Ambiguous capacity tie accepted")
    signatures, normalized, shapes = set(), set(), set()
    records = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for p in puzzle[split]:
                a = p["input"]
                shapes.add(a.shape)
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)
                normalized.add(hashlib.sha256(bytes(a.shape)+norm).hexdigest())
                records.append(pair_features(p))
                before = a.copy()
                _require(np.array_equal(solve(a), p["output"]), "Solver disagreement")
                _require(np.array_equal(a, before), "Solver mutated input")
    _require(len(signatures) == 42 and len(normalized) >= 180 and len(shapes) >= 35, "Structural diversity collapsed")
    for field, minimum in (("capacity", 25), ("selected", 12), ("strokes", 5), ("max_distance", 15)):
        _require(len({f[field] for f in records}) >= minimum, "Variation missing: "+field)
    _require({f["slope"] for f in records} == {-1, 1}, "Slope diversity missing")
    _require(any(f["unreachable"] for f in records) and any(f["detached"] for f in records), "Disconnected cases missing")
    _require(sum(f["path_reversal"] for f in records) >= 42, "Path-distance witnesses missing")
    for role in range(4):
        _require(len({f["palette"][role] for f in records}) >= 9, "Palette role collapsed")
    for seed, a, b in ((17, 2, 1), (18, 10, 4)):
        p = generate_puzzle(seed, a, b)
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed, a, b)), "Determinism failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (True, 1), (3, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    broken = generate_puzzle(0, 2, 1)
    p = broken["test"][0]
    p["output"] = p["output"].copy()
    p["output"][0, 0] = (int(p["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")
    return {"seeds": 42, "pairs": len(records), "distinct_puzzles": len(signatures),
            "normalized_inputs": len(normalized), "shapes": len(shapes),
            "capacities": sorted({f["capacity"] for f in records}),
            "path_reversal_examples": sum(f["path_reversal"] for f in records)}
