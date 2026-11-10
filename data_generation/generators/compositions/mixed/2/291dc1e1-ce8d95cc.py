"""L-key strip normalization followed by independent equality-run contraction.

Only standard library and NumPy are used. No parent module or external artifact
is loaded. The tick on the fast arm makes every input self describing.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np


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
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    """Find the corner and recognize the origin-colored tick on the fast arm."""
    found = []
    if min(a.shape) < 5:
        raise ValueError("No separated L-key scene")
    for bottom in (False, True):
        for right in (False, True):
            g = a[::-1] if bottom else a
            g = g[:, ::-1] if right else g
            origin, bg = int(g[0, 0]), int(g[-1, -1])
            if not (np.all(g[-1, 1:] == bg) and np.all(g[1:, -1] == bg)
                    and np.all(g[1, 1:] == bg) and np.all(g[1:, 1] == bg)):
                continue
            for axis in (0, 1):
                fast_arm = g[0, 1:] if axis == 0 else g[1:, 0]
                slow_arm = g[1:, 0] if axis == 0 else g[0, 1:]
                fast, slow = int(fast_arm[0]), int(slow_arm[0])
                if len({origin, bg, fast, slow}) != 4:
                    continue
                if (len(fast_arm) >= 4 and fast_arm[1] == origin
                        and np.all(fast_arm[2:] == fast)
                        and np.all(slow_arm == slow)):
                    found.append((axis, bottom, right, bg, origin, fast, slow))
    if len(found) != 1:
        raise ValueError("Input must have one ticked L-key")
    return found[0]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    boxes = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, points = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    points.append(q)
        r0, r1 = min(r for r, c in points), max(r for r, c in points) + 1
        c0, c1 = min(c for r, c in points), max(c for r, c in points) + 1
        if len(points) != (r1 - r0) * (c1 - c0):
            raise ValueError("Strips must be full rectangles without background holes")
        boxes.append(tuple(map(int, (r0, r1, c0, c1))))
    return boxes


def _read(input_grid):
    a = _grid(input_grid)
    axis, bottom, right, bg, origin, fast, slow = _key(a)
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _components(mask)
    if not 1 <= len(boxes) <= 15:
        raise ValueError("Expected 1..15 strips")
    lanes = {}
    for box in boxes:
        r0, r1, c0, c1 = box
        if not (0 < r0 < r1 < a.shape[0] and 0 < c0 < c1 < a.shape[1]):
            raise ValueError("Payload touches the outer edge")
        thickness = c1 - c0 if axis else r1 - r0
        length = r1 - r0 if axis else c1 - c0
        if thickness != 2 or length < 2:
            raise ValueError("Expected two-cell-thick strips parallel to the fast arm")
        lanes.setdefault(c0 if axis else r0, []).append(box)
    positions = sorted(lanes)
    if any(b - a < 3 for a, b in zip(positions, positions[1:])):
        raise ValueError("Lane bands must be aligned and separated")
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    blocks, ordered, groups = [], [], []
    for v in sorted(lanes, reverse=right if axis else bottom):
        group = sorted(lanes[v], key=lambda b: b[0] if axis else b[2],
                       reverse=bottom if axis else right)
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in group)
        if any(b[0] - a[1] < 1 for a, b in zip(intervals, intervals[1:])):
            raise ValueError("Strips require a positive gap")
        groups.append(list(range(len(blocks), len(blocks) + len(group))))
        for r0, r1, c0, c1 in group:
            blocks.append(np.rot90(a[r0:r1, c0:c1], rotation).copy())
            ordered.append((r0, r1, c0, c1))
    lengths = [b.shape[1] for b in blocks]
    if len({n % 2 for n in lengths}) != 1:
        raise ValueError("Lengths must have common parity for exact centering")
    return {"blocks": blocks, "groups": groups, "boxes": ordered,
            "configuration": (axis, bottom, right), "background": bg,
            "roles": (bg, origin, fast, slow), "lengths": lengths}


def _stack(blocks, background):
    width = max(b.shape[1] for b in blocks)
    result = np.full((2 * len(blocks), width), background, dtype=np.int8)
    for i, b in enumerate(blocks):
        pad = (width - b.shape[1]) // 2
        result[2 * i:2 * i + 2, pad:pad + b.shape[1]] = b
    return result


def _keep(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(grid):
    # Compute BOTH masks from the complete stack. No intersections are repainted.
    return grid[np.ix_(_keep(grid, 0), _keep(grid, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this input; build the entire stack, then contract both axes."""
    d = _read(input_grid)
    return _collapse(_stack(d["blocks"], d["background"]))


def _digest(a):
    a = np.asarray(a, dtype=np.int8)
    return hashlib.sha256(bytes(a.shape) + a.tobytes()).hexdigest()


def _recurrent(grid, axis):
    vectors = grid if axis == 0 else grid.T
    reduced = vectors[_keep(grid, axis)]
    return len({_digest(v.reshape(1, -1)) for v in reduced}) < len(reduced)


def _flags(d):
    blocks, bg = d["blocks"], d["background"]
    stack = _stack(blocks, bg)
    output = _collapse(stack)
    def changed(order):
        return not np.array_equal(output, _collapse(_stack([blocks[i] for i in order], bg)))
    fast_reverse = [i for group in d["groups"] for i in group[::-1]]
    slow_reverse = [i for group in d["groups"][::-1] for i in group]
    axis, bottom, right = d["configuration"]
    fast_major = sorted(range(len(blocks)), key=lambda i: d["boxes"][i][0 if axis else 2],
                        reverse=bottom if axis else right)
    # A plausible incorrect method: contract rows in each padded block, then
    # concatenate, without contracting rows newly adjacent at block boundaries.
    local_rows = np.concatenate([b[_keep(b, 0)] for b in
                                (stack[2*i:2*i+2] for i in range(len(blocks)))])
    local_output = local_rows[:, _keep(local_rows, 1)]
    col_diff = np.count_nonzero(stack[:, 1:] != stack[:, :-1], axis=0)
    reflected = _collapse(_stack([b[::-1] for b in blocks], bg))
    return {"row_collapse": output.shape[0] < stack.shape[0],
            "column_collapse": output.shape[1] < stack.shape[1],
            "separated_rows": _recurrent(stack, 0),
            "separated_columns": _recurrent(stack, 1),
            "cross_boundary": not np.array_equal(local_output, output),
            "single_cell_exception": bool(np.any(col_diff == 1)),
            "centering": len(set(d["lengths"])) > 1,
            "fast_order": changed(fast_reverse), "lane_order": changed(slow_reverse),
            "lane_major": changed(fast_major),
            "proper_rotation": not np.array_equal(reflected, output)}


def _gaps(rng, occupied, count, extent):
    extra = extent - occupied - count - 1
    weights = rng.dirichlet(np.full(count + 1, float(rng.choice([0.4, 1, 3]))))
    return (1 + rng.multinomial(extra, weights)).tolist()


def _layout(rng, lengths, witness):
    """Pack the chosen sequence into feasible contiguous lane populations."""
    n = len(lengths)
    @lru_cache(None)
    def feasible(start, lanes):
        if lanes == 0:
            return start == n
        if n - start < lanes:
            return False
        occupied = 0
        for end in range(start + 1, n + 1):
            occupied += lengths[end - 1]
            if occupied + end - start + 1 > 29:
                break
            if feasible(end, lanes - 1):
                return True
        return False
    possible = [k for k in range(2 if witness else 1, min(9, n - int(witness)) + 1)
                if feasible(0, k)]
    if not possible:
        return None
    weights = np.array([3.0 if 2 <= k <= 4 else 1.0 for k in possible])
    lane_count = int(rng.choice(possible, p=weights / weights.sum()))
    counts, start = [], 0
    for left in range(lane_count, 0, -1):
        options = [end - start for end in range(start + 1, n + 1)
                   if sum(lengths[start:end]) + end - start + 1 <= 29
                   and feasible(end, left - 1)]
        c = int(rng.choice(options))
        counts.append(c)
        start += c
    minimum_fast, start = 0, 0
    for c in counts:
        minimum_fast = max(minimum_fast, sum(lengths[start:start+c]) + c + 1)
        start += c
    fast_extent = minimum_fast + int(rng.integers(30 - minimum_fast))
    minimum_slow = 3 * lane_count + 1
    slow_extent = minimum_slow + int(rng.integers(30 - minimum_slow))
    slow_gaps = _gaps(rng, 2 * lane_count, lane_count, slow_extent)
    placements, start, v = [], 0, slow_gaps[0]
    for lane, c in enumerate(counts):
        gaps = _gaps(rng, sum(lengths[start:start+c]), c, fast_extent)
        u = gaps[0]
        for j in range(c):
            length = lengths[start+j]
            placements.append((v, u, length))
            u += length + gaps[j+1]
        start += c
        v += 2 + slow_gaps[lane+1]
    return placements, (slow_extent, fast_extent)


def _paint(blocks, placements, shape, config, roles):
    axis, bottom, right = config
    bg, origin, fast, slow = roles
    g = np.full((shape[0] + 1, shape[1] + 1), bg, dtype=np.int8)
    reverse = (bottom != right) if axis == 0 else (bottom == right)
    for block, (v, u, length) in zip(blocks, placements):
        g[1+v:3+v, 1+u:1+u+length] = block[::-1] if reverse else block
    g[0, 0], g[0, 1:], g[1:, 0] = origin, fast, slow
    g[0, 2] = origin  # the second cell away from the corner is the fast tick
    if axis:
        g = g.T
    if bottom:
        g = g[::-1]
    if right:
        g = g[:, ::-1]
    return g.copy()


def _pattern(rng, length, palette):
    mode = int(rng.integers(5))
    if mode == 0:
        return np.full((2, length), rng.choice(palette), dtype=np.int8)
    if mode == 1:
        return np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
    if mode == 2:
        return np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
    if mode == 3:
        motif = rng.choice(palette, size=(2, int(rng.integers(1, min(4, length) + 1))))
        return np.tile(motif, (1, length))[:, :length].astype(np.int8)
    return rng.choice(palette, size=(2, length)).astype(np.int8)


def _blocks(rng, palette, profile):
    witness = profile in (0, 1)
    n = int(rng.integers(5, 9) if witness else rng.integers(1, 16))
    if not witness and rng.random() < 0.4:
        # Native cell-level patterns need not have any particular expansion.
        base = int(rng.choice([2, 3]))
        ceiling = int(rng.integers(1, 14))
        lengths = [base + 2 * int(rng.integers(ceiling)) for _ in range(n)]
        blocks = [_pattern(rng, length, palette) for length in lengths]
        if n >= 3 and rng.random() < 0.5:
            a, b = map(int, rng.choice(n, size=2, replace=False))
            blocks[b] = blocks[a].copy()
        return blocks
    # Shared variable-width columns supply GLOBAL runs even through centering.
    # Palindromic widths make symmetric crops have exact physical padding.
    groups = int(rng.integers(5, 10))
    half = rng.integers(1, 4, size=(groups + 1) // 2).tolist()
    half[0] = 1
    half[1] = int(rng.integers(2, 4))
    widths = half + (half[:-1] if groups % 2 else half)[::-1]
    candidates = [k for k in range((groups + 1) // 2)
                  if sum(widths[k:groups-k]) >= 2]
    crops = [int(rng.choice(candidates)) for _ in range(n)]
    special = None
    if witness:
        short = [k for k in candidates if 1 <= k and 3 <= sum(widths[k:groups-k]) <= 8
                 and groups - 2*k >= 2]
        if not short:
            return None
        a = int(rng.integers(n - 1))
        b = a + 1
        choices = [i for i in range(n) if i < a - 1 or i > b + 1]
        c = int(rng.choice(choices))
        d = int(rng.choice([i for i in range(n) if i not in (a, b, c)]))
        crops[a] = crops[b] = crops[c] = int(rng.choice(short))
        crops[d] = 0
        special = (a, b, c, d)
    virtual = [_pattern(rng, groups - 2*k, palette) for k in crops]
    if witness:
        a, b, c, d = special
        length = virtual[a].shape[1]
        row = rng.choice(palette, size=length).astype(np.int8)
        row[0], row[-1] = palette[0], palette[1]
        other = rng.choice(palette, size=length).astype(np.int8)
        other[0] = palette[2]
        virtual[a][0] = virtual[a][1] = row
        virtual[b][0], virtual[b][1] = row, other
        virtual[c][int(rng.integers(2))] = row
    # Symmetrically placed outer columns recur but are separated by other data.
    for block, crop in zip(virtual, crops):
        if crop == 0:
            block[:, -1] = block[:, 0]
    blocks = [np.repeat(block, widths[k:groups-k], axis=1).astype(np.int8)
              for block, k in zip(virtual, crops)]
    if profile == 1:
        # One exceptional crossing splits a formerly equal column pair. Retain
        # the literal cell instead of inferring a majority or line ownership.
        d = special[3]
        col = sum(widths[:1]) + 1  # second cell of the >=2-cell second run
        row = int(rng.integers(2))
        old = int(blocks[d][row, col])
        blocks[d][row, col] = next(int(c) for c in palette if int(c) != old)
    return blocks


def _make_pair(rng, config, profile, used):
    required = {"row_collapse", "column_collapse", "separated_rows", "separated_columns",
                "cross_boundary", "centering", "fast_order", "lane_order", "lane_major",
                "proper_rotation"}
    if profile == 1:
        required.add("single_cell_exception")
    for _ in range(128):
        roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
        pool = [c for c in range(10) if c != roles[0]]
        size = int(rng.integers(3 if profile in (0, 1) else 1, 10))
        palette = rng.choice(pool, size=size, replace=False).astype(np.int8)
        blocks = _blocks(rng, palette, profile)
        if blocks is None:
            continue
        layout = _layout(rng, [b.shape[1] for b in blocks], profile in (0, 1))
        if layout is None:
            continue
        source = _paint(blocks, *layout, config, roles)
        if _digest(source) in used:
            continue
        descriptor = _read(source)
        if not all(np.array_equal(a, b) for a, b in zip(blocks, descriptor["blocks"])):
            raise AssertionError("Inverse rotation failed")
        if profile in (0, 1):
            flags = _flags(descriptor)
            if not all(flags[name] for name in required):
                continue
        output = _collapse(_stack(blocks, roles[0]))
        return {"input": source, "output": output}
    raise RuntimeError("Bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first = (int(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1 - first[0], not first[1], not first[2])
    held = (first[0], second[1], second[2])
    configurations = [(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)]
    train_pool = [c for c in configurations if c != held]
    train_configs = [first, second] + [train_pool[int(rng.integers(7))]
                                      for _ in range(num_train - 2)]
    test_configs = [held] + [configurations[int(rng.integers(8))] for _ in range(num_test - 1)]
    puzzle, used = {"train": [], "test": []}, set()
    for split, configs in (("train", train_configs), ("test", test_configs)):
        for i, config in enumerate(configs):
            profile = i if split == "train" and i < 2 else 2
            pair = _make_pair(rng, config, profile, used)
            used.add(_digest(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Schema, geometry, exact contraction, lossless expansion, and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, training, test_configs = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, output = _grid(pair["input"]), _grid(pair["output"])
            key = _digest(source)
            assert key not in used, "Duplicate input"
            used.add(key)
            d = _read(source)
            stack = _stack(d["blocks"], d["background"])
            rows, cols = _keep(stack, 0), _keep(stack, 1)
            assert np.array_equal(output, stack[np.ix_(rows, cols)]), "Wrong contraction"
            assert np.array_equal(solve(source.copy()), output), "Input-only solve disagrees"
            row_lengths = np.diff(np.r_[np.flatnonzero(rows), stack.shape[0]])
            col_lengths = np.diff(np.r_[np.flatnonzero(cols), stack.shape[1]])
            rebuilt = np.repeat(np.repeat(output, row_lengths, axis=0), col_lengths, axis=1)
            assert np.array_equal(rebuilt, stack), "Cell values or multiplicities lost"
            assert np.array_equal(_collapse(output), output), "Output still has adjacent equal vectors"
            assert np.array_equal(stack[rows][:, _keep(stack[rows], 1)], output)
            assert np.array_equal(stack[:, cols][_keep(stack[:, cols], 0)], output)
            if split == "train":
                training.append((d, _flags(d)))
            else:
                test_configs.append(d["configuration"])
    evidence = {name: any(f[name] for d, f in training) for name in training[0][1]}
    assert all(evidence.values()), f"Missing training evidence: {evidence}"
    configs = {d["configuration"] for d, f in training}
    assert {c[0] for c in configs} == {0, 1}
    assert {c[1] for c in configs} == {False, True}
    assert {c[2] for c in configs} == {False, True}
    assert test_configs[0] not in configs, "Expected a held-out combination of taught directions"


def _puzzle_digest(puzzle):
    return hashlib.sha256("".join(_digest(p[k]) for s in ("train", "test")
                                  for p in puzzle[s] for k in ("input", "output")).encode()).hexdigest()


def self_test():
    """42 bounded seeds; only small sets of digests/features are retained."""
    puzzles, shapes, outputs, configs, lengths, lanes, palettes = (set() for _ in range(7))
    row_runs, col_runs, all_colors = set(), set(), set()
    contractions = {"both": 0, "row_only": 0, "column_only": 0, "neither": 0}
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_digest(puzzle))
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _read(pair["input"])
                stack = _stack(d["blocks"], d["background"])
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                configs.add(d["configuration"])
                lengths.update(d["lengths"])
                lanes.add(tuple(map(len, d["groups"])))
                palettes.add(len(set(map(int, stack.flat))))
                all_colors.update(d["roles"])
                for axis, seen in ((0, row_runs), (1, col_runs)):
                    vector_count = stack.shape[axis]
                    seen.update(map(int, np.diff(np.r_[np.flatnonzero(_keep(stack, axis)), vector_count])))
                r = pair["output"].shape[0] < stack.shape[0]
                c = pair["output"].shape[1] < stack.shape[1]
                contractions["both" if r and c else "row_only" if r else "column_only" if c else "neither"] += 1
                pairs += 1
    assert len(puzzles) == 42 and len(configs) == 8
    assert len(shapes) >= 50 and len(outputs) >= 30 and len(lanes) >= 30
    assert len(lengths) >= 15 and {0, 1} == {n % 2 for n in lengths}
    assert min(palettes) <= 3 and max(palettes) >= 8 and all_colors == set(range(10))
    assert {1, 2, 3} <= row_runs and {1, 2, 3} <= col_runs
    assert contractions["both"] >= 84
    assert all(contractions[name] > 0 for name in ("row_only", "column_only", "neither"))
    # Hand-built equality cases: preserve separated recurrence and a crossing.
    example = np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3],
                        [1, 1, 2, 1]], dtype=np.int8)
    assert np.array_equal(_collapse(example), np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]]))
    exception = example.copy()
    exception[2, 1] = 5
    assert _collapse(exception).shape == (3, 4)
    # Stack adjacency is causal: moving the final repeated block before the
    # different row changes THREE retained rows to TWO.
    r = np.array([3, 3, 4, 3], dtype=np.int8)
    s = np.array([5, 5, 6, 5], dtype=np.int8)
    blocks = [np.vstack((r, r)), np.vstack((r, s)), np.vstack((r, r))]
    assert np.array_equal(_collapse(_stack(blocks, 0)),
                          np.array([[3, 4, 3], [5, 6, 5], [3, 4, 3]]))
    assert _collapse(_stack([blocks[0], blocks[2], blocks[1]], 0)).shape == (2, 3)
    # Every physical axis/corner combination, all-square payloads, odd lengths,
    # maximum width, maximum raw height, and literal reuse of key colors.
    for blocks, placements, shape in (
        ([np.array([[1, 2], [3, 4]], dtype=np.int8)] * 4,
         [(1, 1, 2), (1, 4, 2), (4, 1, 2), (4, 4, 2)], (7, 7)),
        ([np.full((2, length), 1, dtype=np.int8) for length in (3, 7, 27)],
         [(1, 1, 3), (4, 1, 7), (7, 1, 27)], (10, 29)),
        ([np.full((2, 2), 1 + i % 3, dtype=np.int8) for i in range(15)],
         [(1 + 3*(i//5), 1 + 3*(i%5), 2) for i in range(15)], (10, 16)),
    ):
        expected = _collapse(_stack(blocks, 0))
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    source = _paint(blocks, placements, shape, (axis, bottom, right), (0, 3, 1, 2))
                    for rotation in range(4):
                        assert np.array_equal(solve(np.rot90(source, rotation)), expected)
    # Invalid geometry is rejected using the input alone.
    for invalid_lengths in ([2, 3], [2, 2]):
        blocks = [np.ones((2, length), dtype=np.int8) for length in invalid_lengths]
        source = _paint(blocks, [(1, 1, invalid_lengths[0]), (4, 1, invalid_lengths[1])],
                        (7, 6), (0, False, False), (0, 3, 1, 2))
        if invalid_lengths == [2, 2]:
            source[2, 2] = 0  # a background hole in a full rectangle
        try:
            solve(source)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid strip geometry accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(1000 + 10*a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Bad labels must be rejected even when all geometry remains valid.
    bad = generate_puzzle(999)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "input_shapes": len(shapes), "output_shapes": len(outputs),
            "configurations": len(configs), "lengths": sorted(lengths),
            "lane_population_vectors": len(lanes), "palette_sizes": sorted(palettes),
            "row_run_lengths": sorted(row_runs), "column_run_lengths": sorted(col_runs),
            "contraction_regimes": contractions}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
