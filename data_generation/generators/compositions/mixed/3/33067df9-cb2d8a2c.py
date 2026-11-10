"""Sparse run stacks compile into unary-controlled routing obstacles.

Only standard-library and NumPy code is used. No files, examples, or generator
state are consulted by solve. The module never writes diagnostics or traces.
"""
from __future__ import annotations

import math
import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} must be in {low}..{high}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _row_runs(coarse, bg):
    """At most one monochrome interval in each occupied forward row."""
    rows = []
    for r, row in enumerate(coarse):
        cells = np.flatnonzero(row != bg)
        if not len(cells):
            rows.append(None)
            continue
        lo, hi = int(cells[0]), int(cells[-1]) + 1
        color = int(row[lo])
        _require(len(cells) == hi - lo and bool(np.all(row[lo:hi] == color)),
                 "each occupied coarse row must contain one constant-color run")
        _require(hi - lo >= 2, "coarse runs must have width at least two")
        rows.append((color, lo, hi))
    return rows


def _read(input_grid):
    """Decode the visible seed, pure payload lattice, and off-lattice tabs."""
    raw = _grid(input_grid)
    h, w = raw.shape
    _require(h % 2 == w % 2 == 1 and min(h, w) >= 5, "odd sparse-lattice dimensions required")
    bg = int(raw[0, 0])
    _require(all(int(raw[r, c]) == bg for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))),
             "corners must be background")
    boundary = np.zeros(raw.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    seeds = np.argwhere(boundary & (raw != bg))
    _require(len(seeds) == 1, "exactly one noncorner boundary seed required")
    r, c = map(int, seeds[0])
    k = 0 if r == 0 else 1 if c == w-1 else 2 if r == h-1 else 3
    a = np.rot90(raw, k).copy()
    height, width = a.shape
    route = int(raw[r, c])
    _require(np.count_nonzero(a == route) == 1, "route color belongs only to the seed")
    initial = int(np.argwhere(a == route)[0, 1])
    coarse = a[1::2, 1::2].copy()
    rows = _row_runs(coarse, bg)
    _require(rows[0] is None and any(x is not None for x in rows), "leading blank coarse row required")
    tabs = a[::2, 1::2].copy()
    tabs[0, :] = bg
    marker_colors = set(map(int, np.unique(tabs))) - {bg}
    _require(len(marker_colors) == 1, "one tab color must be visible off the payload lattice")
    marker = marker_colors.pop()
    payload = set(map(int, np.unique(coarse))) - {bg}
    _require(marker not in payload and route not in payload and marker != route,
             "background, seed, tabs and payload labels must be distinct")
    # Reconstruct the entire allowed input, rejecting noise at any other parity.
    allowed = np.full(a.shape, bg, dtype=np.int8)
    allowed[0, initial] = route
    allowed[1::2, 1::2] = coarse
    allowed[::2, 1::2][1:] = tabs[1:]
    _require(np.array_equal(a, allowed), "unexpected cells outside lattice and tabs")
    _require(bool(np.all(tabs[-1] == bg)), "tabs need a payload row immediately below")
    bars = []
    for row, run in enumerate(rows):
        columns = list(map(int, np.flatnonzero(tabs[row] == marker)))
        if run is None:
            _require(not columns, "a tab cannot label an empty coarse row")
            continue
        color, lo, hi = run
        _require(columns and all(lo <= x < hi for x in columns), "every run row needs its own tabs")
        if bars and bars[-1]["end"] == row and (bars[-1]["color"], bars[-1]["lo"], bars[-1]["hi"]) == run:
            bars[-1]["end"] += 1
            bars[-1]["tabs"].extend((row, x) for x in columns)
        else:
            bars.append({"top": row, "end": row+1, "lo": lo, "hi": hi,
                         "color": color, "tabs": [(row, x) for x in columns]})
    for bar in bars:
        cols = sorted(c for _, c in bar["tabs"])
        n = len(cols)
        _require(1 <= n <= 4 and len(set(cols)) == n, "a bar needs 1..4 tabs in distinct columns")
        lo, hi = bar["lo"], bar["hi"]
        low = cols == list(range(lo, lo+n))
        high = cols == list(range(hi-n, hi))
        candidates = []
        if low and 2*hi-1+n+1 < width:
            candidates.append(1)
        if high and 2*lo+1-n-1 >= 0:
            candidates.append(-1)
        _require(len(candidates) == 1, "endpoint pattern must determine one in-bounds bypass")
        bar["n"], bar["side"] = n, candidates[0]
        bar["symmetric"] = bool(low and high)
    return {"raw": raw, "a": a, "coarse": coarse, "rows": rows, "bars": bars,
            "bg": bg, "route": route, "marker": marker, "initial": initial, "rotation": k}


def _obstacles(info, bars=None):
    out = np.full(info["a"].shape, info["bg"], dtype=np.int8)
    for b in info["bars"] if bars is None else bars:
        out[2*b["top"]+1:2*b["end"], 2*b["lo"]+1:2*b["hi"]] = b["color"]
    return out


def _path(info, bars=None, count_override=None, strict=True):
    """Trace in fine-grid cells; relaxed mode is only for omission tests."""
    shape = info["a"].shape
    path = np.zeros(shape, dtype=bool)
    start, current, previous_turn = 0, info["initial"], -1
    turns = []
    for b in info["bars"] if bars is None else bars:
        n = b["n"] if count_override is None else count_override
        d, side = n+1, b["side"]
        front, lo, hi = 2*b["top"]+1, 2*b["lo"]+1, 2*b["hi"]-1
        turn = front-d
        target = hi+d if side == 1 else lo-d
        _require(0 <= turn < shape[0] and 0 <= target < shape[1], "route leaves canvas")
        if strict:
            _require(turn >= 1 and turn >= previous_turn+2, "turns must advance with a blank row between")
            _require(lo <= current <= hi, "every bar must intercept the incoming ray")
        _require(turn >= start, "route cannot reverse its forward heading")
        for r in range(start, turn+1):
            path[r, current] = True
        for c in range(min(current, target), max(current, target)+1):
            path[turn, c] = True
        start, current, previous_turn = turn, target, turn
        turns.append(turn)
    path[start:, current] = True
    if strict:
        _require(not bool(np.any(path & (_obstacles(info) != info["bg"]))), "route collides with a decoded bar")
    return path, turns


def _predict_info(input_grid):
    info = _read(input_grid)
    path, turns = _path(info)
    out = _obstacles(info)
    out[path] = info["route"]
    info["path"], info["turns"] = path, turns
    return np.rot90(out, -info["rotation"]).copy(), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input alone, with all scale/role conventions visibly encoded."""
    return _predict_info(input_grid)[0]


def _span_options(cols, current, n, side, previous=None):
    width = 2*cols+1
    options = []
    # n < span width gives an asymmetric endpoint pattern in sampled puzzles.
    for lo in range(cols):
        for hi in range(lo+max(2, n+1), cols+1):
            if not 2*lo+1 <= current <= 2*hi-1:
                continue
            target = 2*hi+n if side == 1 else 2*lo-n
            if not 0 <= target < width:
                continue
            if previous is not None:
                pl, ph = previous
                if (lo, hi) == previous or max(lo, pl) >= min(hi, ph):
                    continue
            options.append((lo, hi, target))
    return options


def _lateral_plan(rng, cols, counts, sides, partial):
    """Bounded memoized search over small interval/position states."""
    failed = set()

    def search(i, current, previous):
        if i == len(counts):
            return []
        overlap = previous if partial and i == 1 else None
        state = (i, current, overlap)
        if state in failed:
            return None
        options = _span_options(cols, current, counts[i], sides[i], overlap)
        for j in rng.permutation(len(options)):
            lo, hi, target = options[int(j)]
            rest = search(i+1, target, (lo, hi))
            if rest is not None:
                return [(lo, hi, target)] + rest
        failed.add(state)
        return None

    for initial in rng.permutation(np.arange(1, 2*cols)):
        plan = search(0, int(initial), None)
        if plan is not None:
            return int(initial), plan
    return None


def _sample_pair(rng, maximum, kind, rotation, first_side=None):
    # At most 128 geometric attempts, no retained retry payloads.
    for _ in range(128):
        if kind == "merge":
            counts = list(map(int, rng.permutation(np.arange(2, maximum+1))))
            if rng.random() < .3:
                counts.append(int(rng.integers(1, maximum+1)))
        elif kind == "partial":
            counts = [1, 1]
            if rng.random() < .4:
                counts.append(int(rng.integers(1, maximum+1)))
        else:
            count = int(rng.choice((1, 2, 3, 4, 5), p=(.2, .27, .25, .23, .05)))
            counts = list(map(int, rng.integers(1, maximum+1, size=count)))
        sides = list(map(int, rng.choice((-1, 1), size=len(counts))))
        if first_side is not None:
            sides[0] = first_side
        if kind == "partial":
            sides[1] = sides[0]
        depths = [int(rng.integers(1, n+1)) if rng.random() < .55 else 1 for n in counts]
        if kind == "merge":
            candidates = [i for i, n in enumerate(counts) if n >= 2]
            depths[int(rng.choice(candidates))] = 2
        if kind == "partial":
            depths[0] = depths[1] = 1
        tops = [(counts[0]+2)//2 + int(rng.integers(2))]
        for i in range(1, len(counts)):
            if sides[i] == sides[i-1]:
                blank = max(0, math.ceil((2-2*depths[i-1]-counts[i-1]+counts[i])/2))
            else:
                blank = (counts[i]+1)//2
            if kind == "partial" and i == 1:
                blank = 0
            else:
                blank += int(rng.integers(3))
            tops.append(tops[-1]+depths[i-1]+blank)
        minimum_rows = tops[-1]+depths[-1]
        if minimum_rows > 14:
            continue
        rows = minimum_rows + int(rng.integers(min(3, 14-minimum_rows)+1))
        cols = int(rng.integers(6, 15))
        plan = _lateral_plan(rng, cols, counts, sides, kind == "partial")
        if plan is None:
            continue
        initial, spans = plan
        bg, route, marker = map(int, rng.choice(10, 3, replace=False))
        available = [c for c in range(10) if c not in (bg, route, marker)]
        labels = list(map(int, rng.choice(available, int(rng.integers(1, min(4, len(counts))+1)), replace=False)))
        colors = list(map(int, rng.choice(labels, len(counts))))
        if kind == "partial":
            colors[1] = colors[0]
        a = np.full((2*rows+1, 2*cols+1), bg, dtype=np.int8)
        a[0, initial] = route
        out = np.full(a.shape, bg, dtype=np.int8)
        path = np.zeros(a.shape, dtype=bool)
        start, current = 0, initial
        for top, depth, n, side, (lo, hi, target), color in zip(tops, depths, counts, sides, spans, colors):
            for r in range(top, top+depth):
                a[2*r+1, 2*lo+1:2*hi:2] = color
            tab_cols = list(range(lo, lo+n)) if side == 1 else list(range(hi-n, hi))
            # Partition the projected alternating endpoint run over the stack.
            order = list(map(int, rng.permutation(n)))
            assignments = list(range(depth)) + list(map(int, rng.integers(depth, size=n-depth)))
            rng.shuffle(assignments)
            for j, row_offset in zip(order, assignments):
                a[2*(top+row_offset), 2*tab_cols[j]+1] = marker
            out[2*top+1:2*(top+depth), 2*lo+1:2*hi] = color
            turn = 2*top+1-n-1
            path[start:turn+1, current] = True
            path[turn, min(current, target):max(current, target)+1] = True
            start, current = turn, target
        path[start:, current] = True
        if np.any(path & (out != bg)):
            continue
        out[path] = route
        pair = {"input": np.rot90(a, -rotation).copy(), "output": np.rot90(out, -rotation).copy()}
        # This check reads the constructed grid rather than the geometric plan.
        predicted, info = _predict_info(pair["input"])
        assert len(info["bars"]) == len(counts)
        assert np.array_equal(predicted, pair["output"])
        return pair
    raise RuntimeError("bounded geometric construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    _require(nt > ns, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.choice((2, 3, 4)))
    rotations = list(map(int, rng.permutation(4)))
    partial_side = int(rng.choice((-1, 1)))
    specs = [("merge", rotations[0], -partial_side), ("partial", rotations[1], partial_side)]
    specs += [("random", int(rng.choice(rotations[:3])), None) for _ in range(nt-2)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, local_specs in (("train", specs), ("test", [("random", rotations[3], None)]*ns)):
        for kind, rotation, side in local_specs:
            for _ in range(16):
                pair = _sample_pair(rng, maximum, kind, rotation, side)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _bridge_obstacles(info):
    """An authored local check: paint samples, row bridges, then exact-run bridges."""
    a = np.full(info["a"].shape, info["bg"], dtype=np.int8)
    for r, run in enumerate(info["rows"]):
        if run is None:
            continue
        color, lo, hi = run
        a[2*r+1, 2*lo+1:2*hi] = color
        if r and info["rows"][r-1] == run:
            a[2*r, 2*lo+1:2*hi] = color
    return a


def _split_ablation(info):
    fragments = []
    for bar in info["bars"]:
        for r in range(bar["top"], bar["end"]):
            b = dict(bar)
            b["top"], b["end"] = r, r+1
            b["n"] = sum(tr == r for tr, _ in bar["tabs"])
            fragments.append(b)
    try:
        return _path(info, fragments, strict=False)[0]
    except ValueError:
        # Treating every sample row as a command can itself produce an invalid
        # encounter sequence. Invalid counterfactuals are recorded, not clamped.
        return None


def _partial(info):
    return any(a is not None and b is not None and a[0] == b[0] and a[1:] != b[1:]
               and max(a[1], b[1]) < min(a[2], b[2])
               for a, b in zip(info["rows"], info["rows"][1:]))


def _literal_regressions():
    # One literal fixture combines a depth-two exact stack with an adjacent,
    # shifted same-color bar. Expected bends are hand-specified, not generated.
    a = np.zeros((15, 19), dtype=np.int8)
    a[0, 5] = 3
    a[5, 3:10:2] = a[7, 3:10:2] = a[9, 7:14:2] = 2
    a[4, 3] = a[6, 5] = a[8, 7] = 8
    expected = np.zeros(a.shape, dtype=np.int8)
    expected[5:8, 3:10] = expected[9, 7:14] = 2
    expected[:3, 5] = expected[2, 5:13] = expected[2:8, 12] = 3
    expected[7, 12:16] = expected[7:, 15] = 3
    for rotation in range(4):
        for palette in (np.arange(10), np.array([6, 5, 0, 9, 3, 1, 7, 8, 2, 4])):
            query = palette[np.rot90(a, rotation)]
            original = query.copy()
            assert np.array_equal(solve(query), palette[np.rot90(expected, rotation)])
            assert np.array_equal(query, original)
    info = _read(a)
    assert len(info["bars"]) == 2 and _partial(info)
    assert [b["n"] for b in info["bars"]] == [2, 1]
    split = _split_ablation(info)
    assert split is not None and not np.array_equal(split, expected == 3)
    # Same projected markers, different distribution among exact matching rows.
    redistributed = a.copy()
    redistributed[4, 3] = redistributed[6, 5] = 0
    redistributed[4, 5] = redistributed[6, 3] = 8
    assert np.array_equal(solve(redistributed), expected)
    # Count is not stack depth: one more projected tab changes only the route.
    extra = a.copy()
    extra[4, 7] = 8
    changed = solve(extra)
    assert not np.array_equal(changed, expected)
    assert np.array_equal(changed == 2, expected == 2)
    # Incorrect overlap joining would bridge the deliberately empty gap row.
    assert np.all(expected[8, 7:10] == 0)
    # Fully alternating projected endpoints: only one exit is feasible.
    symmetric = np.zeros((13, 13), dtype=np.int8)
    symmetric[0, 3] = 3
    symmetric[7, 1:4:2] = 2
    symmetric[6, 1:4:2] = 8
    predicted, decoded = _predict_info(symmetric)
    assert decoded["bars"][0]["symmetric"] and decoded["bars"][0]["side"] == 1
    assert np.all(predicted[4:, 6] == 3)
    ambiguous = np.zeros((13, 13), dtype=np.int8)
    ambiguous[0, 6] = 3
    ambiguous[7, 5:8:2] = 2
    ambiguous[6, 5:8:2] = 8
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("two feasible symmetric exits accepted")


def _check_path_graph(path, initial):
    degree = np.zeros(path.shape, dtype=np.int8)
    degree[1:] += path[:-1]
    degree[:-1] += path[1:]
    degree[:, 1:] += path[:, :-1]
    degree[:, :-1] += path[:, 1:]
    assert np.count_nonzero(path & (degree == 1)) == 2
    assert degree[0, initial] == 1
    assert np.count_nonzero(path[-1]) == 1
    assert bool(np.all((degree[path] == 1) | (degree[path] == 2)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            assert key not in seen
            seen.add(key)
            predicted, info = _predict_info(a)
            assert b.shape == a.shape and np.array_equal(b, predicted)
            assert np.array_equal(_obstacles(info), _bridge_obstacles(info))
            _check_path_graph(info["path"], info["initial"])
            canonical_out = np.rot90(b, info["rotation"])
            payload = info["coarse"] != info["bg"]
            assert np.array_equal(canonical_out[1::2, 1::2][payload], info["coarse"][payload])
            assert info["marker"] not in set(map(int, np.unique(b)))
            assert np.count_nonzero(b == info["route"]) > 1
            if any(x["end"]-x["top"] > 1 for x in info["bars"]):
                ablated = _split_ablation(info)
                assert ablated is None or not np.array_equal(ablated, info["path"])
            analyses[split].append(info)
    train = analyses["train"]
    assert any(any(b["end"]-b["top"] >= 2 for b in i["bars"]) for i in train)
    assert any(_partial(i) for i in train)
    counts = {b["n"] for i in train for b in i["bars"]}
    assert {1, 2} <= counts
    assert {b["side"] for i in train for b in i["bars"]} == {-1, 1}
    assert {b["n"] for i in analyses["test"] for b in i["bars"]} <= counts
    assert {i["rotation"] for i in train}.isdisjoint({i["rotation"] for i in analyses["test"]})
    # Constant distance and obstacle-only shortcuts must fail on demonstrations.
    assert any(not np.array_equal(_path(i, count_override=1, strict=False)[0], i["path"]) for i in train)


def self_test():
    """Stream 42 seeds; do not keep grids or write files across iterations."""
    _literal_regressions()
    shapes, layouts, directions, counts, depths, backgrounds, labels = set(), set(), set(), set(), set(), set(), set()
    signatures = set()
    pairs = merged = partials = repeated = zero_payload = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        signature = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output"))
        assert signature == tuple(_key(p[k]) for s in ("train", "test") for p in again[s] for k in ("input", "output"))
        # Keep hashes rather than reconstructible puzzle payloads across seeds.
        signatures.add(hashlib.sha256(repr(signature).encode()).hexdigest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                pairs += 1
                shapes.add(pair["input"].shape)
                directions.add(info["rotation"])
                backgrounds.add(info["bg"])
                bars = info["bars"]
                counts.add(len(bars))
                depths.update(b["end"]-b["top"] for b in bars)
                labels.update(b["color"] for b in bars)
                merged += any(b["end"]-b["top"] > 1 for b in bars)
                partials += _partial(info)
                repeated += any(a["side"] == b["side"] for a, b in zip(bars, bars[1:]))
                zero_payload += any(b["color"] == 0 for b in bars)
                renaming = {info["bg"]: 0, info["route"]: 1, info["marker"]: 2}
                normalized = []
                for c in info["a"].flat:
                    c = int(c)
                    normalized.append(renaming.setdefault(c, len(renaming)))
                layouts.add(hashlib.sha256(bytes(normalized)).hexdigest())
    assert len(signatures) == 42 and len(layouts) >= 150 and len(shapes) >= 35
    assert directions == {0, 1, 2, 3} and {1, 2, 3, 4} <= counts
    assert {1, 2, 3} <= depths and len(backgrounds) == len(labels) == 10
    assert merged >= 42 and partials >= 42 and repeated >= 42 and zero_payload >= 5
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corruption check: missing a route cell is rejected.
    damaged = generate_puzzle(12, 2, 1)
    pair = damaged["test"][0]
    expected, info = _predict_info(pair["input"])
    candidate = np.argwhere((expected == info["route"]) & (pair["input"] != info["route"]))[0]
    pair["output"][tuple(candidate)] = info["bg"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds_checked": 42, "pairs_checked": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_layouts": len(layouts),
            "directions": sorted(directions), "bar_counts": sorted(counts), "stack_depths": sorted(depths),
            "merged_pairs": merged, "partial_overlap_pairs": partials, "repeated_side_pairs": repeated,
            "zero_payload_pairs": zero_payload, "independent_solver": False}
