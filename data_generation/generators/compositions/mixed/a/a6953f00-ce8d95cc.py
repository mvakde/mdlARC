"""Collapse orthogonal line spacing, then crop by the reduced square's parity.

Only NumPy and the standard library are used. No filesystem access or saved
examples participate in generation or solving; self_test streams its samples.
"""
from __future__ import annotations

import hashlib

import numpy as np


HOST_SEEDS = (
    0, 737793532, 182875081, 846102594, 389907451, 446982410, 927105183,
    240389494, 513168884, 568924412, 608578704, 245997234, 252198873,
    800066124, 282687663, 478730634, 253995532, 601889231, 765611639,
    692267026, 139270796, 917929145, 954365801, 676181885, 883867868,
    801631608, 447960877, 154285643, 252699111, 806556570, 64905362,
    212698886, 554763600, 361268858, 312692417, 299964961, 839466480,
    198133954, 297713879, 654393422, 190402095, 523955070,
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("a grid must be a 1..30 integer rectangle using colors 0..9")
    return array


def _collapse(grid):
    # Equality is of entire vectors, never just colors or local segments.
    row_keep = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    reduced = grid[row_keep]
    col_keep = np.r_[True, np.any(reduced[:, 1:] != reduced[:, :-1], axis=0)]
    return reduced[:, col_keep].copy()


def _corner(grid, parity=None):
    if parity is None:
        parity = grid.shape[0] % 2
    return grid[:2, :2].copy() if parity else grid[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input: consecutive-vector collapse, then odd L / even R."""
    reduced = _collapse(_grid(input_grid))
    if reduced.shape[0] != reduced.shape[1] or reduced.shape[0] < 2:
        raise ValueError("the collapsed grid must be a square of side at least two")
    return _corner(reduced)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _phases(n, row_phase, col_phase):
    # phase 0 = background at index zero; phase 1 = a line at index zero.
    return ((np.arange(n) + row_phase) % 2 == 1,
            (np.arange(n) + col_phase) % 2 == 1)


def _canonical(rng, n, row_phase, col_phase, background, witness, owner):
    row_line, col_line = _phases(n, row_phase, col_phase)
    ri, ci = np.flatnonzero(row_line), np.flatnonzero(col_line)
    palette = np.array([c for c in range(10) if c != background], dtype=np.int8)
    count = len(ri) + len(ci)
    if witness or (count <= 9 and rng.random() < .30):
        colors = rng.choice(palette, size=count, replace=False)
    else:
        # Small reusable palettes, including the two-color whole-grid regime.
        k = int(rng.integers(1, min(9, count) + 1))
        colors = rng.choice(rng.choice(palette, size=k, replace=False), size=count)
    horizontal, vertical = colors[:len(ri)], colors[len(ri):]
    result = np.full((n, n), background, dtype=np.int8)
    result[:, ci] = vertical
    result[ri, :] = horizontal[:, None]
    mode = int(rng.integers(5))
    if mode < 2:
        bits = np.full((len(ri), len(ci)), mode, dtype=int)
    elif mode == 2:
        bits = np.broadcast_to(rng.integers(2, size=(len(ri), 1)), (len(ri), len(ci))).copy()
    elif mode == 3:
        bits = np.broadcast_to(rng.integers(2, size=(1, len(ci))), (len(ri), len(ci))).copy()
    else:
        bits = rng.integers(2, size=(len(ri), len(ci)))
    if owner is not None:
        # Exactly one crossing lies in the selected 2x2. Expose both literal
        # crossing choices across the two required training witnesses. Give
        # the opposite candidate the same owner, preventing an ownership cue
        # from deciding between the two corners in these demonstrations.
        for candidate_cols in ((0, 1), (n - 2, n - 1)):
            j = next(j for j, col in enumerate(ci) if col in candidate_cols)
            bits[0, j] = owner
    for i, row in enumerate(ri):
        for j, col in enumerate(ci):
            result[row, col] = vertical[j] if bits[i, j] else horizontal[i]
    return result


def _runs_for(rng, n, phase, total, forced=None, bulk=False):
    line = (np.arange(n) + phase) % 2 == 1
    slots = np.flatnonzero(~line)
    runs = np.ones(n, dtype=int)
    extra = total - n
    if forced is not None:
        _require(not line[forced] and extra >= 1, "impossible forced band")
        runs[forced] += 1
        extra -= 1
    if extra:
        if bulk or rng.random() < .5:
            # A single border or interior band may hold most of the spacing.
            mass = int(rng.integers((extra + 1) // 2, extra + 1))
            runs[int(rng.choice(slots))] += mass
            extra -= mass
        runs[slots] += rng.multinomial(extra, np.full(len(slots), 1 / len(slots)))
    return runs


def _dimension(rng, n, high, parity=None, low=None):
    low = n if low is None else max(n, low)
    options = [x for x in range(low, high + 1) if parity is None or x % 2 == parity]
    return int(rng.choice(options))


def _make_pair(rng, spec, background):
    n, rp, cp = spec["n"], spec["rp"], spec["cp"]
    witness = spec.get("witness", False)
    canonical = _canonical(rng, n, rp, cp, background, witness, spec.get("owner"))
    if witness:
        height = _dimension(rng, n, 22, spec["raw_parity"], n + 2)
        width = (height if rng.random() < .30 else
                 _dimension(rng, n, 22, spec["raw_parity"], n + 2))
        forced_row = 0
        edge = 0 if n % 2 else n - 1
        # Expand a background band at the selected corner, or immediately
        # beside its outer line. The phase is shared by the two parity
        # witnesses, so an outer background cannot itself select the corner.
        col_line = _phases(n, rp, cp)[1]
        forced_col = edge if not col_line[edge] else (1 if n % 2 else n - 2)
    else:
        forced_row = forced_col = None
        regime = spec.get("regime", "train")
        if regime == "spacing":
            height, width = int(rng.integers(24, 31)), int(rng.integers(24, 31))
        elif regime == "sparse":
            if rng.random() < .5:
                height, width = n, int(rng.integers(24, 31))
            else:
                height, width = int(rng.integers(24, 31)), n
        else:
            high = 22 if regime == "train" else 30
            height, width = _dimension(rng, n, high), _dimension(rng, n, high)
            if rng.random() < .18:
                if rng.random() < .5:
                    height = n
                else:
                    width = n
            if height == width == n:
                height += 1
    rr = _runs_for(rng, n, rp, height, forced_row, spec.get("regime") == "spacing")
    cr = _runs_for(rng, n, cp, width, forced_col, spec.get("regime") == "spacing")
    source = np.repeat(np.repeat(canonical, rr, axis=0), cr, axis=1)
    _require(np.array_equal(_collapse(source), canonical), "canonical construction failed")
    # Ground truth is derived directly from the constructed canonical grid.
    return {"input": source, "output": _corner(canonical)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, 2, 10, "num_train")
    num_test = _integer(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    odd, even = int(rng.choice([5, 7, 9])), int(rng.choice([4, 6, 8]))
    raw_parity = int(rng.integers(2))
    witness_cp = int(rng.integers(2))
    specs = [
        {"n": odd, "rp": 0, "cp": witness_cp, "witness": True, "owner": 0, "raw_parity": raw_parity},
        {"n": even, "rp": 0, "cp": witness_cp, "witness": True, "owner": 1, "raw_parity": raw_parity},
    ]
    if num_train >= 4:
        specs.extend([
            {"n": int(rng.choice([n for n in (3, 5, 7, 9) if n != odd])), "rp": 1, "cp": 1},
            {"n": int(rng.choice([n for n in (4, 6, 8, 10) if n != even])), "rp": 1, "cp": 0},
        ])
    while len(specs) < num_train:
        specs.append({"n": int(rng.integers(3, 11)), "rp": int(rng.integers(2)), "cp": int(rng.integers(2))})
    # Different examples can change every color role. No palette is a key.
    background = int(rng.integers(10))
    backgrounds = [background, (background + int(rng.integers(1, 10))) % 10]
    backgrounds.extend(int(rng.integers(10)) for _ in range(num_train - 2))
    puzzle, used = {"train": [], "test": []}, set()

    def append(split, spec, bg):
        for _ in range(96):
            pair = _make_pair(rng, spec, bg)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("bounded uniqueness sampling exhausted")

    for spec, bg in zip(specs, backgrounds):
        append("train", spec, bg)
    rng.shuffle(puzzle["train"])
    first_parity = int(rng.integers(2))
    regimes = list(rng.permutation(["inventory", "spacing", "sparse"]))
    for i in range(num_test):
        parity = (first_parity + i) % 2
        regime = str(regimes[i % 3])
        if regime == "inventory":
            n = int(rng.choice([x for x in range(11, 21) if x % 2 == parity]))
        elif regime == "spacing":
            n = int(rng.choice([s["n"] for s in specs if s["n"] % 2 == parity]))
        else:
            n = int(rng.choice([2, 4])) if parity == 0 else 3
        if num_train < 4:
            # Both parity demonstrations expose this boundary phase; the
            # shift is scale or spacing, without an unseen boundary phase.
            rp, cp = 0, witness_cp
        else:
            rp, cp = int(rng.integers(2)), int(rng.integers(2))
        append("test", {"n": n, "rp": rp, "cp": cp, "regime": regime}, int(rng.integers(10)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _axis_runs(grid, axis):
    vectors = grid if axis == 0 else grid.T
    starts, lengths = [0], [1]
    for index in range(1, len(vectors)):
        if np.array_equal(vectors[index], vectors[index - 1]):
            lengths[-1] += 1
        else:
            starts.append(index)
            lengths.append(1)
    return np.array(starts), np.array(lengths)


def _line_analysis(canonical, rr, cr):
    n = len(canonical)
    for rp in (0, 1):
        for cp in (0, 1):
            rl, cl = _phases(n, rp, cp)
            rb, cb = np.flatnonzero(~rl), np.flatnonzero(~cl)
            ri, ci = np.flatnonzero(rl), np.flatnonzero(cl)
            bg = int(canonical[rb[0], cb[0]])
            hs, vs = canonical[ri, cb[0]], canonical[rb[0], ci]
            if (not np.all(canonical[np.ix_(rb, cb)] == bg)
                    or np.any(hs == bg) or np.any(vs == bg)
                    or not np.all(canonical[np.ix_(ri, cb)] == hs[:, None])
                    or not np.all(canonical[np.ix_(rb, ci)] == vs[None, :])
                    or not np.all(rr[ri] == 1) or not np.all(cr[ci] == 1)):
                continue
            crossings = canonical[np.ix_(ri, ci)]
            if not np.all((crossings == hs[:, None]) | (crossings == vs[None, :])):
                continue
            j = next(j for j, c in enumerate(ci) if c in ((0, 1) if n % 2 else (n - 2, n - 1)))
            owner = None if hs[0] == vs[j] else int(crossings[0, j] == vs[j])
            return {"n": n, "rp": rp, "cp": cp, "background": bg,
                    "h": len(ri), "v": len(ci), "selected_owner": owner,
                    "parallel_reuse": len(set(hs)) < len(hs) or len(set(vs)) < len(vs),
                    "cross_reuse": bool(set(hs) & set(vs))}
    raise AssertionError("input is not a one-cell orthogonal line arrangement")


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    g, out = _grid(pair["input"]), _grid(pair["output"])
    rows, rr = _axis_runs(g, 0)
    cols, cr = _axis_runs(g, 1)
    c = g[np.ix_(rows, cols)]
    _require(c.shape[0] == c.shape[1] and len(c) >= 2, "collapsed square required")
    _require(np.array_equal(np.repeat(np.repeat(c, rr, axis=0), cr, axis=1), g), "lossless runs")
    _require(np.array_equal(_collapse(c), c), "collapse must be idempotent")
    _require(np.array_equal(_collapse(g.T).T, c), "axis order must commute")
    _require(out.shape == (2, 2) and np.array_equal(out, _corner(c)), "wrong reduced parity crop")
    _require(np.array_equal(solve(g), out), "solve disagreement")
    _require(g.shape != c.shape, "generated pairs must contain some compression")
    info = _line_analysis(c, rr, cr)
    info.update({"grid": g, "canonical": c, "rr": rr, "cr": cr, "output": out})
    return info


def _global_dedup(grid):
    # A deliberately wrong alternative used only for witness checks.
    rows, seen = [], set()
    for row in grid:
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            rows.append(row)
    reduced = np.array(rows)
    cols, seen = [], set()
    for col in reduced.T:
        key = col.tobytes()
        if key not in seen:
            seen.add(key)
            cols.append(col)
    return np.array(cols).T


def validate_generated_puzzle(puzzle):
    """Check schema, line/run invariants, literal crops, and training witnesses."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(info)
    train = analyses["train"]
    for p in (0, 1):
        _require(any(x["n"] % 2 == p and not np.array_equal(x["canonical"][:2, :2], x["canonical"][:2, -2:])
                     for x in train), "need a contrasting corner witness for each parity")
        if a >= 4:
            _require(len({x["n"] for x in train if x["n"] % 2 == p}) >= 2,
                     "need two reduced sizes per parity with four demonstrations")
    _require({x["selected_owner"] for x in train} >= {0, 1}, "literal crossing choices not exposed")
    _require(len({x["background"] for x in train}) >= 2, "vary background roles in training")
    _require(any(x["grid"].shape[0] > x["n"] and x["grid"].shape[1] > x["n"]
                 for x in train), "need a two-axis compression demonstration")
    _require(any(x["n"] % 2 != y["n"] % 2 and (x["rp"], x["cp"]) == (y["rp"], y["cp"])
                 for x in train for y in train), "boundary phase must not decide parity branch")
    max_train_side = max(x["n"] for x in train)
    max_train_extent = max(max(x["grid"].shape) for x in train)
    _require(all(x["n"] > max_train_side or max(x["grid"].shape) > max_train_extent
                 for x in analyses["test"]), "test needs an inventory or spacing shift")
    for alternative in ("raw_height", "raw_width", "inverse_raw", "no_rows", "no_cols", "global"):
        failures = 0
        for x in train:
            g, c, out = x["grid"], x["canonical"], x["output"]
            if alternative == "raw_height":
                pred = _corner(c, g.shape[0] % 2)
            elif alternative == "raw_width":
                pred = _corner(c, g.shape[1] % 2)
            elif alternative == "inverse_raw":
                pred = _corner(c, 1 - g.shape[0] % 2)
            elif alternative == "no_rows":
                cols, _ = _axis_runs(g, 1)
                pred = _corner(g[:, cols])
            elif alternative == "no_cols":
                rows, _ = _axis_runs(g, 0)
                pred = _corner(g[rows])
            else:
                pred = _corner(_global_dedup(g))
            failures += not np.array_equal(pred, out)
        _require(failures > 0, f"training fails to distinguish {alternative}")
    _require(any(not np.array_equal(_corner(x["grid"]), x["output"]) for x in train), "raw crop shortcut")
    if b >= 2:
        _require({x["n"] % 2 for x in analyses["test"]} == {0, 1}, "test parity coverage")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.astype(np.uint8).tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed audit plus all legal counts and focused boundary tests.

    Retains only sets of small features and hashes, never puzzles or traces.
    Returns descriptive metrics; these are not an independent semantic review.
    """
    fingerprints, shapes, normalized, sides, phases, backgrounds, layouts = (set() for _ in range(7))
    metrics = {"pairs": 0, "one_axis_only": 0, "both_axes": 0, "rectangular": 0,
               "parallel_reuse": 0, "cross_reuse": 0, "odd_tests": 0, "even_tests": 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signature = _fingerprint(puzzle)
        _require(signature == _fingerprint(generate_puzzle(seed)), "determinism")
        fingerprints.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = _analyze(pair)
                g, n = x["grid"], x["n"]
                metrics["pairs"] += 1
                shapes.add(g.shape)
                sides.add(n)
                phases.add((x["rp"], x["cp"]))
                backgrounds.add(x["background"])
                layouts.add((n, x["rp"], x["cp"], tuple(x["rr"]), tuple(x["cr"])))
                labels = {}
                norm = bytes(labels.setdefault(int(c), len(labels)) for c in g.flat)
                normalized.add(hashlib.sha256(bytes(g.shape) + norm).hexdigest())
                shrink = (g.shape[0] > n, g.shape[1] > n)
                metrics["one_axis_only"] += shrink[0] != shrink[1]
                metrics["both_axes"] += all(shrink)
                metrics["rectangular"] += g.shape[0] != g.shape[1]
                metrics["parallel_reuse"] += int(x["parallel_reuse"])
                metrics["cross_reuse"] += int(x["cross_reuse"])
                if split == "test":
                    metrics["odd_tests" if n % 2 else "even_tests"] += 1
    _require(len(fingerprints) == 42, "seed diversity")
    _require(len(shapes) >= 70 and len(normalized) >= 150, "geometry diversity")
    _require(len(sides) >= 12 and len(phases) == 4 and len(backgrounds) == 10, "structural roles")
    _require(len(layouts) >= 150, "spacing diversity")
    _require(metrics["one_axis_only"] >= 10 and metrics["both_axes"] >= 84, "axis regimes")
    _require(metrics["parallel_reuse"] >= 10 and metrics["cross_reuse"] >= 10, "line color reuse")
    _require(metrics["odd_tests"] >= 10 and metrics["even_tests"] >= 10, "test parity diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(314159, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Solve beyond the generator's preferred sizes, including both no-op axes.
    local_rng = np.random.default_rng(24680)
    for n in (2, 3, 29, 30):
        for rp in (0, 1):
            for cp in (0, 1):
                c = _canonical(local_rng, n, rp, cp, 0, False, None)
                _require(np.array_equal(solve(c), _corner(c)), "canonical boundary support")
    c = np.array([[0, 1, 0, 2, 0], [3, 3, 3, 2, 3], [0, 1, 0, 2, 0],
                  [4, 1, 4, 4, 4], [0, 1, 0, 2, 0]], dtype=int)
    g = np.repeat(np.repeat(c, [3, 1, 2, 1, 2], axis=0), [2, 1, 3, 1, 2], axis=1)
    _require(np.array_equal(solve(g), [[0, 1], [3, 3]]), "explicit separated repeats")
    _require(_global_dedup(g).shape == (3, 3), "separated vectors must survive")
    c = np.array([[1, 0, 2, 0], [1, 3, 2, 3], [1, 0, 2, 0], [4, 4, 4, 4]])
    g = np.repeat(np.repeat(c, [3, 1, 2, 1], axis=0), [1, 2, 1, 3], axis=1)
    _require(np.array_equal(solve(g), [[2, 0], [2, 3]]), "explicit parity switch")
    _require(not np.array_equal(_corner(_global_dedup(g)), solve(g)), "global dedup witness")
    for g in (np.zeros((3, 3), dtype=int), np.array([[0, 1, 2], [3, 4, 5]])):
        try:
            solve(g)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid reduced shape accepted")
    metrics.update({"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
                    "normalized_inputs": len(normalized), "reduced_sides": sorted(sides),
                    "boundary_phases": len(phases), "backgrounds": len(backgrounds), "layouts": len(layouts)})
    return metrics
