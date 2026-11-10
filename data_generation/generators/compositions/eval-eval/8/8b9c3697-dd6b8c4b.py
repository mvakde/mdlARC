"""Fork magazines: width-aware shortest-path collection and finite packing.

Only standard library and NumPy. No files, hidden palettes, or generation state
are used by solve. All geometry is measured again from the visible input.
"""

import hashlib
from collections import deque

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "Invalid colors")
    return a


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in _STEPS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _fork(cells):
    """Parse a downward-open three-prong component, with a full base row."""
    rows, cols = zip(*cells)
    r0, c0 = min(rows), min(cols)
    width, depth = max(cols) - c0 + 1, max(rows) - r0
    if width < 5 or depth < 2:
        return None
    local = {(r - r0, c - c0) for r, c in cells}
    if {c for r, c in local if r == 0} != set(range(width)):
        return None
    lengths = [sum((r, c) in local for r in range(1, depth + 1)) for c in range(width)]
    expected = {(0, c) for c in range(width)}
    expected |= {(r, c) for c, n in enumerate(lengths) for r in range(1, n + 1)}
    if expected != local or not lengths[0] or not lengths[-1]:
        return None
    middle = [c for c in range(1, width - 1) if lengths[c]]
    if not middle:
        return None
    q, span = middle[0], len(middle)
    if middle != list(range(q, q + span)) or q < 2 or q + span > width - 2:
        return None
    if 2 * q + span != width or len({lengths[c] for c in middle}) != 1:
        return None
    center = lengths[q]
    capacity = min(lengths[0], lengths[-1]) - center
    if capacity < 1:
        return None
    return dict(r=r0, c=c0, width=width, span=span, clearance=q - 1,
                center=center, guards=(lengths[0], lengths[-1]), capacity=capacity,
                target=(r0 + center + 1, c0 + q), cells=cells)


def _parse(input_grid):
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    _require(len(values) == 3 and sum(counts == counts.max()) == 1, "Need three clear color roles")
    background = int(values[np.argmax(counts)])
    parses = []
    for turn in range(4):
        a = np.rot90(original, turn)
        for color in values:
            if color == background:
                continue
            for cells in _components(a == color):
                f = _fork(cells)
                if f is not None:
                    parses.append((turn, int(color), f))
    _require(len(parses) == 1, "Need one unambiguous isolated fork")
    turn, structure, f = parses[0]
    a = np.rot90(original, turn)
    payload = next(int(c) for c in values if c not in (background, structure))
    bars = []
    for cells in _components(a == payload):
        rows, cols = zip(*cells)
        r, c, n = min(rows), min(cols), len(cells)
        _require(min(rows) == max(rows) and max(cols) - c + 1 == n,
                 "Payload components must be separated parallel bars")
        bars.append(dict(r=r, c=c, width=n, cells=cells))
    _require(bars, "No bars")
    r, c = f["target"]
    cap, span = f["capacity"], f["span"]
    _require(r + cap <= a.shape[0] and c + span <= a.shape[1], "Magazine leaves canvas")
    _require(np.all(a[r:r + cap, c:c + span] == background), "Magazine must initially be empty")
    return dict(grid=a, turn=turn, background=background, structure=structure,
                payload=payload, fork=f, bars=bars)


def _distances(obstacle, span, target):
    """BFS on left-edge anchors of a horizontal, nonrotating rigid bar."""
    h, w = obstacle.shape
    if not 1 <= span <= w:
        return np.empty((h, 0), dtype=np.int16)
    prefix = np.pad(np.cumsum(obstacle, axis=1), ((0, 0), (1, 0)))
    valid = (prefix[:, span:] - prefix[:, :-span]) == 0
    d = np.full(valid.shape, -1, dtype=np.int16)
    if not (0 <= target[0] < h and 0 <= target[1] < valid.shape[1] and valid[target]):
        return d
    d[target] = 0
    queue = deque([target])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < valid.shape[1] and valid[rr, cc] and d[rr, cc] < 0:
                d[rr, cc] = d[r, c] + 1
                queue.append((rr, cc))
    return d


def _selection(analysis):
    a, f = analysis["grid"], analysis["fork"]
    d = _distances(a == analysis["structure"], f["span"], f["target"])
    eligible = []
    for i, bar in enumerate(analysis["bars"]):
        if bar["width"] == f["span"] and d[bar["r"], bar["c"]] >= 0:
            eligible.append((int(d[bar["r"], bar["c"]]), i))
    eligible.sort()
    cap = f["capacity"]
    if len(eligible) > cap:
        _require(eligible[cap - 1][0] < eligible[cap][0], "Ambiguous distance tie at capacity")
    return [i for _, i in eligible[:cap]], eligible, d


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collect fitting reachable bars, then pack a stop-relative unary prefix."""
    analysis = _parse(input_grid)
    selected, _, _ = _selection(analysis)
    output = analysis["grid"].copy()
    for i in selected:
        for p in analysis["bars"][i]["cells"]:
            output[p] = analysis["background"]
    r, c = analysis["fork"]["target"]
    output[r:r + len(selected), c:c + analysis["fork"]["span"]] = analysis["payload"]
    return np.rot90(output, -analysis["turn"]).copy()


def _geometry(rng, span, cap, kind, test, fallback=False, unequal=False, mobile=False):
    h = int(rng.integers(26, 31) if test else rng.integers(19, 24))
    w = int(rng.integers(26, 31) if test else rng.integers(22, 26))
    if fallback:
        h, w = (30, 30) if test else (23, 25)
    r0, c0 = (1, 1) if fallback else (int(rng.integers(1, 4)), int(rng.integers(1, 3)))
    clearance = 1 if fallback else int(rng.integers(1, 3))
    center = 1 if fallback else int(rng.integers(1, 4))
    width = span + 2 * clearance + 2
    guards = [center + cap, center + cap + int(rng.integers(1 if unequal else 0, 3))]
    if rng.integers(2):
        guards.reverse()
    if mobile and not fallback:
        r0 = int(rng.integers(1, h - max(guards) - 3))
        c0 = int(rng.integers(1, w - width - 3))
    structure = np.zeros((h, w), dtype=bool)
    structure[r0, c0:c0 + width] = True
    structure[r0 + 1:r0 + guards[0] + 1, c0] = True
    structure[r0 + 1:r0 + guards[1] + 1, c0 + width - 1] = True
    q = c0 + clearance + 1
    structure[r0 + 1:r0 + center + 1, q:q + span] = True
    fork_mask = structure.copy()
    top = r0 + center + 1
    wall = np.zeros_like(structure)
    if kind in ("detour", "mixed"):
        col = c0 + width + 1
        end = h - (4 if fallback else int(rng.integers(3, 7)))
        wall[:end, col] = True
    if kind in ("bottleneck", "mixed"):
        # An open point doorway narrower than the bar's full cross-section.
        ra, rb, ca, cb = h - 8, h - 2, w - span - 5, w - 2
        wall[ra, ca:cb + 1] = wall[rb, ca:cb + 1] = True
        wall[ra:rb + 1, ca] = wall[ra:rb + 1, cb] = True
        if span > 1:
            wall[ra, ca + 1:ca + span] = False
    if kind in ("baffles", "mixed", "partitions") and not fallback:
        amount = int(rng.integers(2, 6 if test else 4))
        for _ in range(amount):
            if rng.integers(2):
                rr = int(rng.integers(r0 + max(guards) + 2, h - 1))
                lo = int(rng.integers(0, w - 5))
                hi = int(rng.integers(lo + 3, w + 1))
                wall[rr, lo:hi] = True
                if kind == "partitions":
                    wall[rr, int(rng.integers(lo + 1, hi - 1))] = False
            else:
                cc = int(rng.integers(c0 + width + 1, w - 1))
                lo = int(rng.integers(0, h - 5))
                hi = int(rng.integers(lo + 3, h + 1))
                wall[lo:hi, cc] = True
                if kind == "partitions":
                    wall[int(rng.integers(lo + 1, hi - 1)), cc] = False
    # A blank moat makes the receiver a unique independent component.
    wall[max(0, r0 - 1):r0 + max(guards) + 2, max(0, c0 - 1):c0 + width + 1] = False
    obstacle = structure | wall
    forbidden = np.zeros_like(obstacle)
    forbidden[max(0, r0 - 1):r0 + max(guards) + 2, max(0, c0 - 1):c0 + width + 1] = True
    return obstacle, fork_mask, forbidden, (top, q)


def _anchors(obstacle, forbidden, width):
    h, w = obstacle.shape
    prefix = np.pad(np.cumsum(obstacle | forbidden, axis=1), ((0, 0), (1, 0)))
    valid = prefix[:, width:] == prefix[:, :-width]
    return [tuple(map(int, p)) for p in np.argwhere(valid)]


def _place(rng, pool, count, width, occupied, forced=()):
    """Keep monochrome bars as distinct 4-connected components."""
    chosen = []
    pool = [tuple(p) for p in pool]
    order = list(forced) + [pool[int(i)] for i in rng.permutation(len(pool))]
    for r, c in order:
        if len(chosen) == count:
            break
        cells = {(r, cc) for cc in range(c, c + width)}
        if cells & occupied:
            continue
        chosen.append((r, c, width))
        occupied.update((rr, cc) for rr in range(r - 1, r + 2)
                        for cc in range(c - 1, c + width + 1))
    return chosen if len(chosen) == count else None


def _construct(rng, spec, colors, test, turns):
    span, cap, obligation = spec["span"], spec["cap"], spec["obligation"]
    bg, sc, pc = colors
    for attempt in range(48):
        fallback = attempt == 47
        kind = spec["kind"]
        if fallback and obligation not in ("partial", "overflow"):
            kind = "mixed" if span > 1 else "detour"
        obstacle, _, forbidden, target = _geometry(
            rng, span, cap, kind, test, fallback, unequal=obligation == "overflow",
            mobile=obligation == "free")
        d = _distances(obstacle, span, target)
        point = _distances(obstacle, 1, target)
        pool = _anchors(obstacle, forbidden, span)
        reachable = [p for p in pool if d[p] >= 0]
        inaccessible = [p for p in pool if d[p] < 0]
        bottleneck = [p for p in inaccessible if point[p] >= 0]
        occupied, bars = set(), []
        if obligation == "partial":
            load = int(rng.integers(2, cap))
        elif obligation == "overflow":
            load = cap + int(rng.integers(1, 5))
        else:
            load = int(rng.integers(1, cap + 5))
        selected_forced, retained_forced = (), ()
        if load > cap:
            # Choose a strict path frontier. The evidence example must reverse
            # direct Manhattan ordering, not merely contain a decorative wall.
            frontiers = [p for p in reachable if sum(abs(p[j] - target[j]) for j in (0, 1)) >= 8]
            rng.shuffle(frontiers)
            accepted = False
            for a in frontiers[:120]:
                near = [p for p in reachable if d[p] <= d[a]]
                far = [p for p in reachable if d[p] > d[a]]
                if obligation == "overflow":
                    direct_a = sum(abs(a[j] - target[j]) for j in (0, 1))
                    reversed_far = [p for p in far if sum(abs(p[j] - target[j]) for j in (0, 1)) < direct_a]
                    if not reversed_far:
                        continue
                    b = reversed_far[int(rng.integers(len(reversed_far)))]
                    selected_forced, retained_forced = (a,), (b,)
                local = set()
                first = _place(rng, near, cap, span, local, selected_forced)
                second = _place(rng, far, load - cap, span, local, retained_forced) if first else None
                if first and second:
                    bars, occupied = first + second, local
                    accepted = True
                    break
            if not accepted:
                continue
        else:
            bars = _place(rng, reachable, load, span, occupied)
            if bars is None:
                continue
        unreachable_count = int(rng.integers(0, 4))
        if obligation == "partial":
            if not bottleneck:
                continue
            unreachable_count = max(1, unreachable_count)
        extras = _place(rng, inaccessible, unreachable_count, span, occupied,
                        (bottleneck[int(rng.integers(len(bottleneck)))],) if obligation == "partial" else ())
        if extras is None:
            continue
        bars += extras
        wrong_count = 1 if obligation in ("partial", "overflow") else int(rng.integers(0, 4))
        wrong_width = span + 1 if span == 1 else int(rng.choice([span - 1, span + 1]))
        wrong_pool = _anchors(obstacle, forbidden, wrong_width)
        # They could physically reach the first resting anchor at their own
        # width; rejection must therefore depend on the fork's geometric key.
        wrong_distance = _distances(obstacle, wrong_width, target)
        wrong_pool = [p for p in wrong_pool if wrong_distance[p] >= 0]
        wrong = _place(rng, wrong_pool, wrong_count, wrong_width, occupied)
        if wrong is None:
            continue
        bars += wrong
        grid = np.full(obstacle.shape, bg, dtype=np.int8)
        grid[obstacle] = sc
        for r, c, width in bars:
            grid[r, c:c + width] = pc
        # Labels are painted from construction choices, before the input parser
        # is used to validate geometry and the routing decision.
        output = grid.copy()
        chosen = sorted(bars[:load], key=lambda b: int(d[b[0], b[1]]))[:cap]
        for r, c, width in chosen:
            output[r, c:c + width] = bg
        r, c = target
        output[r:r + len(chosen), c:c + span] = pc
        turn = int(rng.choice(turns))
        grid, output = np.rot90(grid, turn).copy(), np.rot90(output, turn).copy()
        try:
            analysis = _parse(grid)
            _selection(analysis)
            if not np.array_equal(solve(grid), output):
                continue
            if obligation == "overflow" and not pair_features(grid)["path_reversal"]:
                continue
            if obligation == "partial" and not pair_features(grid)["point_only"]:
                continue
        except AssertionError:
            continue
        return {"input": grid, "output": output}
    raise AssertionError("Bounded magazine construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be in 1..4 and smaller than num_train")
    rng = np.random.default_rng(seed)
    spans = [2, 3]
    rng.shuffle(spans)
    train_specs = [dict(span=spans[0], cap=int(rng.integers(3, 6)), kind="bottleneck", obligation="partial"),
                   dict(span=spans[1], cap=int(rng.integers(2, 6)), kind="detour", obligation="overflow")]
    kinds = ["open", "bottleneck", "detour", "baffles", "partitions", "mixed"]
    for _ in range(num_train - 2):
        train_specs.append(dict(span=int(rng.integers(1, 4)), cap=int(rng.integers(1, 7)),
                                kind=str(rng.choice(kinds)), obligation="free"))
    rng.shuffle(train_specs)
    taught_spans = sorted({s["span"] for s in train_specs})
    test_specs = [dict(span=int(rng.choice(taught_spans)), cap=int(rng.integers(1, 7)),
                       kind=str(rng.choice(kinds[1:])), obligation="free") for _ in range(num_test)]
    # Three allowed training orientations; the fourth is held out in tests.
    held_out = int(rng.integers(4))
    familiar = [k for k in range(4) if k != held_out]
    rng.shuffle(familiar)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, spec in enumerate(specs):
            for _ in range(8):
                colors = tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _construct(rng, spec, colors, split == "test",
                                  [held_out] if split == "test" else [familiar[index % 3]])
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Duplicate inputs after bounded retries")
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    a = _parse(input_grid)
    f, bars = a["fork"], a["bars"]
    selected, eligible, d = _selection(a)
    point = _distances(a["grid"] == a["structure"], 1, f["target"])
    selected_set = set(selected)
    retained = [i for _, i in eligible if i not in selected_set]
    direct = lambda i: abs(bars[i]["r"] - f["target"][0]) + abs(bars[i]["c"] - f["target"][1])
    other_maps = {b["width"]: _distances(a["grid"] == a["structure"], b["width"], f["target"])
                  for b in bars if b["width"] != f["span"]}
    return dict(shape=tuple(map(int, np.asarray(input_grid).shape)), span=f["span"],
                capacity=f["capacity"], selected=len(selected), eligible=len(eligible),
                excess=len(retained), turn=a["turn"], clearance=f["clearance"],
                center=f["center"], guards=f["guards"],
                wrong=sum(b["width"] != f["span"] for b in bars),
                wrong_reachable=sum(b["width"] != f["span"] and
                                    bool(other_maps[b["width"]][b["r"], b["c"]] >= 0) for b in bars),
                unreachable=int(sum(b["width"] == f["span"] and d[b["r"], b["c"]] < 0 for b in bars)),
                point_only=int(sum(b["width"] == f["span"] and d[b["r"], b["c"]] < 0
                                   and point[b["r"], b["c"]] >= 0 for b in bars)),
                path_reversal=bool(selected and retained and max(map(direct, selected)) > min(map(direct, retained))),
                max_distance=max((x[0] for x in eligible), default=0),
                frontier_gap=eligible[f["capacity"]][0] - eligible[f["capacity"] - 1][0] if retained else None,
                wall_cells=int(np.count_nonzero(a["grid"] == a["structure"])) - len(f["cells"]),
                palette=(a["background"], a["structure"], a["payload"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Split types")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Pair counts")
    seen, stats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "Dimensions changed")
            key = (grid.shape, grid.astype(np.int8).tobytes())
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            analysis = _parse(grid)
            selected, eligible, _ = _selection(analysis)
            predicted = solve(grid)
            _require(np.array_equal(predicted, output), "Incorrect magazine output")
            canonical_out = np.rot90(output, analysis["turn"])
            original = analysis["grid"]
            pc, sc = analysis["payload"], analysis["structure"]
            _require(np.array_equal(original == sc, canonical_out == sc), "Structure changed")
            _require(np.count_nonzero(original == pc) == np.count_nonzero(canonical_out == pc), "Payload mass changed")
            r, c = analysis["fork"]["target"]
            span, cap = analysis["fork"]["span"], analysis["fork"]["capacity"]
            expected_stack = np.full((cap, span), analysis["background"], dtype=original.dtype)
            expected_stack[:len(selected)] = pc
            _require(np.array_equal(canonical_out[r:r + cap, c:c + span], expected_stack), "Non-prefix stack")
            for i, b in enumerate(analysis["bars"]):
                if i not in selected:
                    _require(all(canonical_out[p] == pc for p in b["cells"]), "Unselected bar erased")
            _require(len(selected) == min(cap, len(eligible)), "Wrong capacity")
            stats[split].append(pair_features(grid))
    train = stats["train"]
    _require(any(2 <= f["selected"] < f["capacity"] for f in train), "Missing partial prefix demonstration")
    _require(any(f["excess"] and f["path_reversal"] and f["guards"][0] != f["guards"][1]
                 for f in train), "Missing distance/capacity/shorter-guard demonstration")
    _require(any(f["point_only"] for f in train), "Missing footprint bottleneck demonstration")
    _require(any(f["wrong_reachable"] for f in train), "Missing accessible width mismatch demonstration")
    _require({2, 3} <= {f["span"] for f in train}, "Need two demonstrated bar widths")
    train_turns = {f["turn"] for f in train}
    _require(len(train_turns) >= 2, "Need two demonstrated orientations")
    _require(all(f["turn"] not in train_turns and min(f["shape"]) >= 26 for f in stats["test"]),
             "Missing size/orientation shift")


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds streamed in memory; no files, retained puzzles, or trace logs."""
    signatures, features = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        features.extend(pair_features(pair["input"]) for split in p for pair in p[split])
    _require(len(signatures) == 42, "Seed diversity")
    for field, minimum in (("shape", 25), ("span", 3), ("capacity", 6), ("selected", 6),
                           ("turn", 4), ("palette", 30), ("max_distance", 20)):
        _require(len({f[field] for f in features}) >= minimum, "Missing variation: " + field)
    _require(any(f["wall_cells"] == 0 for f in features), "Open fields missing")
    _require(any(f["guards"][0] != f["guards"][1] for f in features), "Unequal guards missing")
    _require(any(f["point_only"] and f["excess"] for f in features), "Combined constraints missing")
    _require(_signature(generate_puzzle(82615, 5, 3)) == _signature(generate_puzzle(82615, 5, 3)), "Determinism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(10000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "Explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (4, True), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
    # Hand-check a narrow doorway: point access does not imply rigid-bar access.
    wall = np.zeros((9, 12), dtype=bool)
    wall[:, 5] = True
    wall[4, 5] = False
    _require(_distances(wall, 1, (2, 1))[2, 8] >= 0, "Point doorway")
    # A horizontal bar can slide sideways through a vertical-wall doorway.
    _require(_distances(wall, 3, (2, 1))[2, 8] >= 0, "Lateral motion must be allowed")
    wall = wall.T.copy()
    _require(_distances(wall, 3, (1, 2))[8, 2] < 0, "Narrow cross-section must block")
    # A strict frontier is essential: selection cannot depend on queue order.
    p = generate_puzzle(41, 2, 1)
    analysis = _parse(p["train"][0]["input"])
    analysis["fork"]["capacity"] = 1
    # Duplicate an eligible anchor in a private analysis to exercise equal rank.
    _, eligible, _ = _selection(_parse(p["train"][0]["input"]))
    eligible_bar = analysis["bars"][eligible[0][1]].copy()
    analysis["bars"] = [eligible_bar, eligible_bar.copy()]
    try:
        _selection(analysis)
    except AssertionError:
        pass
    else:
        raise AssertionError("Capacity tie accepted")
    # Corrupt an output at a packed cell; validation must reject it.
    p = generate_puzzle(99, 2, 1)
    p["test"][0]["output"] = p["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted answer accepted")
