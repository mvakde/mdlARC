"""Repair a two-fill rectangle scene, quotient equal lines, then mark erasures.

No evidence files, examples, random state, or generation metadata are used by
solve. All generation and validation are in memory; this module performs no I/O.
"""
from itertools import permutations
import hashlib
import numpy as np

BACKGROUND = 0
HIGHLIGHT = 8
FILL_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(a, color):
    """Return (cells, bounding box) for four-connected components."""
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    h, w = a.shape
    for rr, cc in np.argwhere(a == color):
        p = (int(rr), int(cc))
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and not seen[q] and a[q] == color):
                    seen[q] = True
                    stack.append(q)
        rows, cols = zip(*cells)
        result.append((cells, (min(rows), min(cols), max(rows)+1,
                               max(cols)+1, int(color))))
    return result


def _paint(shape, boxes):
    a = np.zeros(shape, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        a[r0:r1, c0:c1] = color
    return a


def _overlap(a, b):
    return (max(0, min(a[2], b[2])-max(a[0], b[0]))
            * max(0, min(a[3], b[3])-max(a[1], b[1])))


def _path_geometry(boxes):
    n = len(boxes)
    if not 2 <= n <= 4:
        return False
    degree = [0] * n
    for i in range(n):
        if min(boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]) < 5:
            return False
        for j in range(i):
            if _overlap(boxes[i], boxes[j]):
                if boxes[i][4] == boxes[j][4]:
                    return False
                degree[i] += 1
                degree[j] += 1
    return sorted(degree) == [1, 1] + [2] * (n-2)


def _recover(a):
    """Fit paint order to component-derived rectangle bounds, input only.

    In this family each visible rectangle remains connected, retains all four
    bounding extents, and has >=18 observed cells. Ordinary noise components
    have <=3 cells and are detached from all genuine components.
    """
    colors = tuple(int(c) for c in np.unique(a) if c != BACKGROUND)
    _require(0 in a and len(colors) == 2 and HIGHLIGHT not in colors,
             "input requires background 0 and two non-8 fill colors")
    boxes = []
    for color in colors:
        for cells, box in _components(a, color):
            if len(cells) >= 18:
                boxes.append(box)
            else:
                _require(len(cells) <= 3, "ambiguous medium-sized component")
    boxes.sort()
    _require(_path_geometry(boxes), "unsupported rectangle overlap path")
    scenes = {}
    for order in permutations(boxes):
        clean = _paint(a.shape, order)
        cost = int(np.count_nonzero(clean != a))
        key = clean.tobytes()
        if key not in scenes:
            scenes[key] = (cost, clean, order)
    ranked = sorted(scenes.values(), key=lambda x: x[0])
    cost, clean, order = ranked[0]
    _require(cost <= max(8, int(a.size * .10)), "damage too dense")
    _require(len(ranked) == 1 or ranked[1][0] >= cost + 3,
             "covering order is ambiguous")
    _require(np.count_nonzero(clean == 0) >= 8, "insufficient exterior evidence")
    for box in boxes:
        r0, c0, r1, c1, color = box
        visible = clean[r0:r1, c0:c1] == color
        observed = a[r0:r1, c0:c1] == color
        _require(visible.sum() >= 18 and (visible & observed).sum() >= .80*visible.sum(),
                 "insufficient visible rectangle evidence")
    return clean, colors, tuple(order)


def _quotient(clean):
    """Independent maximal full-vector runs, with total coordinate maps."""
    row_change = np.r_[True, np.any(clean[1:] != clean[:-1], axis=1)]
    col_change = np.r_[True, np.any(clean[:, 1:] != clean[:, :-1], axis=0)]
    rows, cols = np.flatnonzero(row_change), np.flatnonzero(col_change)
    row_map, col_map = np.cumsum(row_change)-1, np.cumsum(col_change)-1
    compact = clean[np.ix_(rows, cols)].copy()
    return compact, row_map, col_map


def _recurs(a):
    return (any(np.array_equal(a[i], a[j]) for i in range(a.shape[0])
                for j in range(i+2, a.shape[0]))
            or any(np.array_equal(a[:, i], a[:, j]) for i in range(a.shape[1])
                   for j in range(i+2, a.shape[1])))


def _wrong_noise_witness(x):
    wrong = ((x["input"] != x["clean"]) & (x["input"] != 0)
             & (x["clean"] != 0))
    return any((int(x["row_map"][r]), int(x["col_map"][c])) not in x["centers"]
               for r, c in np.argwhere(wrong))


def _neighbors(shape, p):
    return [(r, c) for r in range(max(0, p[0]-1), min(shape[0], p[0]+2))
            for c in range(max(0, p[1]-1), min(shape[1], p[1]+2)) if (r, c) != p]


def _stamp(compact, centers, fills):
    """Set-valued center transfer; simultaneous, compatible ring constraints."""
    centers = set(centers)
    demands = {}
    for p in sorted(centers):
        _require(int(compact[p]) in fills, "center must belong to a fill")
        opposite = fills[1] if compact[p] == fills[0] else fills[0]
        for q in _neighbors(compact.shape, p):
            if q in centers:
                continue
            _require(q not in demands or demands[q] == opposite,
                     "opposite-fill rings conflict")
            demands[q] = opposite
    result = compact.copy()
    for p, color in demands.items():
        result[p] = color
    for p in centers:
        result[p] = HIGHLIGHT
    return result


def _analyze(value):
    a = _grid(value)
    clean, fills, boxes = _recover(a)
    compact, rm, cm = _quotient(clean)
    holes = [tuple(map(int, p)) for p in np.argwhere((a == 0) & (clean != 0))]
    _require(bool(holes), "no true erasures")
    centers = {(int(rm[r]), int(cm[c])) for r, c in holes}
    result = _stamp(compact, centers, fills)
    return {"input": a, "clean": clean, "compact": compact, "row_map": rm,
            "col_map": cm, "holes": holes, "centers": centers, "fills": fills,
            "boxes": boxes, "expected": result}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input alone, under fixed 0/8 conventions."""
    return _analyze(input_grid)["expected"]


def _knots(rng, length, count):
    for _ in range(60):
        gaps = rng.integers(2, 7, size=2*count-1)
        span = int(gaps.sum())
        if span <= length and gaps[0]+gaps[1] >= 5 and gaps[-2]+gaps[-1] >= 5:
            slack = length-span
            start = int(rng.choice([0, slack, int(rng.integers(slack+1))]))
            return np.r_[start, start+np.cumsum(gaps)]
    # Feasible for every requested generation canvas/count.
    gaps = np.full(2*count-1, 3)
    start = int(rng.integers(length-int(gaps.sum())+1))
    return np.r_[start, start+np.cumsum(gaps)]


def _scene(rng, shape, count, axis):
    main, cross = shape[axis], shape[1-axis]
    k = _knots(rng, main, count)
    intervals = [(int(k[0]), int(k[2]))]
    intervals += [(int(k[2*i-1]), int(k[2*i+2])) for i in range(1, count-1)]
    intervals += [(int(k[-3]), int(k[-1]))]
    boxes = []
    for i, (lo, hi) in enumerate(intervals):
        width = int(rng.integers(6, max(7, min(cross, int(cross*.85)))+1))
        start = int(rng.integers(cross-width+1))
        if rng.random() < .4:
            start = int(rng.choice([0, cross-width]))
        if axis == 0:
            box = (lo, start, hi, start+width, 1+i % 2)
        else:
            box = (start, lo, start+width, hi, 1+i % 2)
        if boxes and _overlap(box, boxes[-1]) < 4:
            return None
        boxes.append(box)
    order = [boxes[int(i)] for i in rng.permutation(count)]
    clean = _paint(shape, order)
    # The reconstruction convention requires all true bounds to survive.
    actual = [b for color in (1, 2) for cells, b in _components(clean, color)]
    if sorted(actual) != sorted(boxes):
        return None
    if any(len(cells) < 22 for color in (1, 2) for cells, _ in _components(clean, color)):
        return None
    if np.count_nonzero(clean == 0) < 12:
        return None
    return clean


def _choose_centers(rng, compact, mode):
    points = [tuple(map(int, p)) for p in np.argwhere(compact != 0)]
    rng.shuffle(points)
    selected = []
    if mode == "both":
        candidates = [(p, q) for i, p in enumerate(points) for q in points[:i]
                      if compact[p] != compact[q]
                      and max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 3]
        if not candidates:
            return None
        selected = list(candidates[int(rng.integers(len(candidates)))])
    elif mode == "merge":
        candidates = [(p, q) for i, p in enumerate(points) for q in points[:i]
                      if compact[p] == compact[q]
                      and max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1]
        if not candidates:
            return None
        selected = list(candidates[int(rng.integers(len(candidates)))])
    target = int(rng.integers(max(1, len(selected)), 6))
    for p in points:
        if len(selected) >= target:
            break
        if p in selected:
            continue
        try:
            _stamp(compact, selected+[p], (1, 2))
        except ValueError:
            continue
        selected.append(p)
    return selected


def _damage(rng, clean, centers, rm, cm, merge):
    a = clean.copy()
    holes = []
    # Select a collision block first; two distinct physical holes then map to
    # exactly one logical center. Large blocks make this a genuine erasure.
    blocks = {p: [(int(r), int(c)) for r in np.flatnonzero(rm == p[0])
                  for c in np.flatnonzero(cm == p[1])] for p in centers}
    eligible = [p for p in centers if len(blocks[p]) >= 6]
    if merge and not eligible:
        return None
    collision = eligible[int(rng.integers(len(eligible)))] if merge else None
    for p in centers:
        pool = blocks[p]
        total = 2 if p == collision else 1
        if rng.random() < .25 and len(pool) >= 8:
            total = max(total, 2)
        for i in rng.choice(len(pool), total, replace=False):
            q = pool[int(i)]
            a[q] = 0
            holes.append(q)
    # Ordinary wrong-fill errors are isolated within a homogeneous interior;
    # speckles are isolated within true background. Neither is an erasure.
    occupied = set(holes)
    for exterior in (False, True):
        pool = []
        for r in range(clean.shape[0]):
            for c in range(clean.shape[1]):
                p = (r, c)
                color = int(clean[p])
                if (color == 0) != exterior or p in occupied:
                    continue
                local = _neighbors(clean.shape, p)
                if len(local) < 8 or not all(clean[q] == color for q in local):
                    continue
                pool.append(p)
        rng.shuffle(pool)
        target = int(rng.integers(1, 4))
        added = 0
        for p in pool:
            if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in occupied):
                continue
            a[p] = int(rng.integers(1, 3)) if exterior else 3-int(clean[p])
            occupied.add(p)
            added += 1
            if added >= target:
                break
    return a


def _pair(rng, shape, count, axis, mode, palette, forbidden_shape=None):
    for _ in range(240):
        clean = _scene(rng, shape, count, axis)
        if clean is None:
            continue
        compact, rm, cm = _quotient(clean)
        if compact.shape == forbidden_shape:
            continue
        if mode == "merge" and not _recurs(compact):
            continue
        centers = _choose_centers(rng, compact, mode)
        if not centers:
            continue
        corrupted = _damage(rng, clean, centers, rm, cm,
                            mode == "merge" or rng.random() < .45)
        if corrupted is None:
            continue
        mapping = np.array([0, *palette], dtype=np.int8)
        inp = mapping[corrupted]
        try:
            analysis = _analyze(inp)
        except ValueError:
            continue
        if not np.array_equal(analysis["clean"], mapping[clean]):
            continue
        # Output is built from construction-time geometry, never from a call
        # to solve. The input-only analysis above is an acceptance check.
        expected_centers = set(centers)
        expected = _stamp(mapping[compact], expected_centers, tuple(palette))
        if not np.array_equal(analysis["expected"], expected):
            continue
        if mode == "both":
            if not _wrong_noise_witness(analysis):
                continue
            if not np.any((inp != 0) & (analysis["clean"] == 0)):
                continue
            if _quotient(inp)[0].shape == compact.shape:
                continue
        if mode == "merge" and len(analysis["holes"]) <= len(analysis["centers"]):
            continue
        return {"input": inp, "output": expected}
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback")


def _count(value, low, high, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)) and low <= value <= high,
             f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    palette = tuple(map(int, rng.choice(FILL_COLORS, 2, replace=False)))
    fixed_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    first_shape = None
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            mode = "both" if split == "train" and i == 0 else (
                "merge" if split == "train" and i == 1 else "free")
            count = 3 if mode == "both" else int(rng.integers(2, 4))
            if regime == 0:
                shape = tuple(map(int, rng.integers(22, 31, size=2)))
                axis = int(rng.integers(2))
                if split == "test":
                    count = 4
            elif regime == 1:
                low, high = (19, 25) if split == "train" else (27, 31)
                shape = tuple(map(int, rng.integers(low, high, size=2)))
                axis = int(rng.integers(2))
            else:
                short, long = int(rng.integers(19, 24)), int(rng.integers(25, 31))
                shape = (short, long) if split == "train" else (long, short)
                axis = 1 if split == "train" else 0
                if mode != "both":
                    count = int(rng.integers(2, 5))
            if not fixed_palette:
                palette = tuple(map(int, rng.choice(FILL_COLORS, 2, replace=False)))
            for _ in range(8):
                pair = _pair(rng, shape, count, axis, mode, palette,
                             first_shape if split == "train" and i == 1 else None)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("unable to construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and i == 0:
                first_shape = pair["output"].shape
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, 2, 10, "num_train")
    _count(ne, 1, 4, "num_test")
    _require(nt > ne, "train count must exceed test count")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input/output")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = (a.shape, a.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            x = _analyze(a)
            _require(np.array_equal(out, x["expected"]), "incorrect output")
            compact, rm, cm = x["compact"], x["row_map"], x["col_map"]
            _require(np.array_equal(compact[rm[:, None], cm[None, :]], x["clean"]),
                     "quotient does not reconstruct scene")
            _require(out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1],
                     "both clean axes must collapse")
            _require(set(map(tuple, np.argwhere(out == 8))) == x["centers"],
                     "center transfer is not set-valued")
            _require(np.array_equal(_stamp(compact, reversed(sorted(x["centers"])),
                                          x["fills"]), out), "marker order dependence")
            if split == "train":
                training.append(x)
    _require(any({int(x["compact"][p]) for p in x["centers"]} == set(x["fills"])
                 for x in training), "training must show both opposite fills")
    _require(any(len(x["holes"]) > len(x["centers"]) for x in training),
             "training must show merged erasures")
    _require(any(any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                         for p in x["centers"] for q in x["centers"] if p != q)
                 for x in training), "training must show overlapping marker centers")
    _require(len({x["compact"].shape for x in training}) >= 2,
             "training must vary compressed shape")
    _require(any(np.any((x["input"] != x["clean"]) & (x["input"] != 0)
                        & (x["clean"] != 0)) for x in training),
             "training must show ordinary wrong-fill repair")
    _require(any(np.any((x["input"] != 0) & (x["clean"] == 0)) for x in training),
             "training must show exterior speckle removal")
    _require(any(_wrong_noise_witness(x) for x in training),
             "ordinary noise must distinguish an extra-marker shortcut")
    _require(any(_recurs(x["compact"]) for x in training),
             "training must retain a separated whole-vector recurrence")
    _require(any(_quotient(x["input"])[0].shape != x["compact"].shape for x in training),
             "training must distinguish raw-input compression")


def self_test():
    """Bounded, streaming 42-seed checks plus small semantic probes."""
    shapes, clean_layouts, palettes, schedules, counts = set(), set(), set(), set(), set()
    metrics = {"seeds": 42, "pairs": 0, "merged_pairs": 0, "clipped_pairs": 0,
               "dual_fill_pairs": 0, "separated_recurrence_pairs": 0,
               "raw_compression_wrong_shape": 0, "ring_into_background_pairs": 0,
               "mark_before_compress_differs": 0}
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fp = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, out = pair["input"], pair["output"]
                x = _analyze(a)
                _require(np.array_equal(solve(a), out), "solver mismatch")
                _require(np.array_equal(pair["input"], a), "input mutated")
                fp.update(a.tobytes())
                fp.update(out.tobytes())
                shapes.add((a.shape, out.shape))
                comp = x["compact"]
                normalized = np.where(comp == 0, 0, np.where(comp == x["fills"][0], 1, 2))
                clean_layouts.add((comp.shape, normalized.tobytes()))
                palettes.add(x["fills"])
                counts.add(len(x["boxes"]))
                rm, cm = x["row_map"], x["col_map"]
                schedules.add((tuple(np.bincount(rm)), tuple(np.bincount(cm))))
                metrics["pairs"] += 1
                metrics["merged_pairs"] += len(x["holes"]) > len(x["centers"])
                metrics["dual_fill_pairs"] += len({int(comp[p]) for p in x["centers"]}) == 2
                metrics["clipped_pairs"] += any(p[0] in (0, comp.shape[0]-1)
                                                  or p[1] in (0, comp.shape[1]-1)
                                                  for p in x["centers"])
                metrics["raw_compression_wrong_shape"] += _quotient(a)[0].shape != comp.shape
                metrics["ring_into_background_pairs"] += any(comp[q] == 0 for p in x["centers"]
                                                               for q in _neighbors(comp.shape, p))
                metrics["separated_recurrence_pairs"] += _recurs(comp)
                # Parent-omission/order probes use the same recovered clean
                # geometry but deliberately perform the proposed shortcuts.
                _require(not np.array_equal(comp, out), "omitting A has no effect")
                physical = _stamp(x["clean"], x["holes"], x["fills"])
                _require(physical.shape != out.shape, "omitting B has no effect")
                metrics["mark_before_compress_differs"] += not np.array_equal(
                    _quotient(physical)[0], out)
                # Separate cellwise output check: union transferred masks, then
                # inspect every output cell against all neighboring centers.
                transferred = np.zeros(comp.shape, dtype=bool)
                for r, c in x["holes"]:
                    transferred[rm[r], cm[c]] = True
                checked = comp.copy()
                for r in range(comp.shape[0]):
                    for c in range(comp.shape[1]):
                        if transferred[r, c]:
                            checked[r, c] = 8
                        else:
                            demands = {next(v for v in x["fills"] if v != comp[p])
                                       for p in _neighbors(comp.shape, (r, c)) if transferred[p]}
                            _require(len(demands) <= 1, "ring conflict")
                            if demands:
                                checked[r, c] = demands.pop()
                _require(np.array_equal(checked, out), "cellwise semantic check failed")
        fingerprints.add(fp.digest())
    _require(len(fingerprints) == 42, "seeds do not vary puzzles")
    _require(len(clean_layouts) >= 80 and len(shapes) >= 80 and len(schedules) >= 100,
             "insufficient geometric variation")
    _require(counts == {2, 3, 4} and len(palettes) >= 12, "insufficient role/count variation")
    for key in ("merged_pairs", "clipped_pairs", "dual_fill_pairs",
                "separated_recurrence_pairs", "raw_compression_wrong_shape",
                "ring_into_background_pairs", "mark_before_compress_differs"):
        _require(metrics[key] >= 10, f"insufficient {key}")
    one, again = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in one for p, q in zip(one[s], again[s])
                 for k in ("input", "output")), "nondeterministic seed")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(101+nt*7+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A separated recurrence must survive; equal adjacent copies disappear.
    a = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [2, 2, 1, 2], [1, 1, 2, 1]])
    _require(_quotient(a)[0].tolist() == [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
             "global rather than consecutive deduplication")
    scene = np.ones((5, 7), dtype=np.int8)
    scene[:, 4:] = 2
    try:
        _stamp(scene, [(2, 3), (2, 4)], (1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting rings accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in one[s]] for s in one}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect label accepted")
    metrics.update({"distinct_clean_layouts": len(clean_layouts), "shape_pairs": len(shapes),
                    "fill_pairs": len(palettes), "band_schedules": len(schedules),
                    "rectangle_counts": sorted(counts)})
    return metrics
