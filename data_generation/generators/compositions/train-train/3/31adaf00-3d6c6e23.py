"""Local gravity in maximal square chambers; only stdlib and NumPy are used."""

from collections import Counter
from math import isqrt
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid color")
    return a


def _wall(a):
    w = int(a[0, 0])
    _require(w != 0, "zero is reserved for empty space")
    _require(bool(np.all(a[0] == w) and np.all(a[-1] == w)
                  and np.all(a[:, 0] == w) and np.all(a[:, -1] == w)),
             "outer frame must be uniform wall")
    return w


def _squares(mask):
    """Enumerate squares, excluding any with a complete open side extension.

    Coordinates are (top, left, side). Particle colors are absent from this
    calculation: the mask is all non-wall cells, including particles.
    """
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = []
    for s in range(2, min(h, w) + 1):
        area = integral[s:, s:] - integral[:-s, s:] - integral[s:, :-s] + integral[:-s, :-s]
        for rr, cc in np.argwhere(area == s * s):
            r, c = int(rr), int(cc)
            if r and mask[r - 1, c:c + s].all():
                continue
            if r + s < h and mask[r + s, c:c + s].all():
                continue
            if c and mask[r:r + s, c - 1].all():
                continue
            if c + s < w and mask[r:r + s, c + s].all():
                continue
            result.append((r, c, s))
    return sorted(result)


def _overlap(a, b):
    r, c, s = a
    t, d, z = b
    return max(r, t) < min(r + s, t + z) and max(c, d) < min(c + s, d + z)


def _chambers(a):
    wall = _wall(a)
    squares = _squares(a != wall)
    _require(not any(_overlap(x, y) for i, x in enumerate(squares)
                     for y in squares[i + 1:]), "overlapping maximal squares outside family")
    streams = []
    for r, c, s in squares:
        local = a[r:r + s, c:c + s]
        rows, cols = np.nonzero(local != 0)
        _require(len(rows) > 0 and len(set(map(int, cols))) == 1,
                 "each square must contain exactly one vertical particle stream")
        k = isqrt(len(rows))
        _require(k * k == len(rows), "local particle count must be a perfect square")
        col = c + int(cols[0])
        _require(c <= col - k + 1 and col + k <= c + s and k <= s,
                 "pyramid does not fit its chamber")
        streams.append({"square": (r, c, s), "column": col, "rows": r + rows,
                        "sequence": local[rows, cols].copy(), "k": k})
    return wall, streams


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input alone; frame color identifies all wall cells."""
    a = _grid(input_grid)
    _, streams = _chambers(a)
    out = a.copy()
    for stream in streams:
        r, c, s = stream["square"]
        k, col, seq = stream["k"], stream["column"], stream["sequence"]
        out[r:r + s, c:c + s] = 0
        for level in range(k):
            row = r + s - k + level
            out[row, col - level:col + level + 1] = seq[level * level:(level + 1) ** 2]
    return out


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _spaced_rect(a, b):
    """Room interiors must be separated by at least one cell before paths."""
    r, c, h, w = a
    t, d, z, v = b
    return r + h < t or t + z < r or c + w < d or d + v < c


def _place(rng, dims, h, w, fixed=()):
    rooms = list(fixed)
    ordered = {}
    # Small rooms should not occupy the only space available to a large one.
    for index in sorted(range(len(dims)), key=lambda i: -dims[i][0] * dims[i][1]):
        rh, rw = dims[index]
        placed = False
        for _ in range(160):
            r = int(rng.integers(1, h - rh))
            c = int(rng.integers(1, w - rw))
            rect = (r, c, rh, rw)
            if all(_spaced_rect(rect, other) for other in rooms):
                rooms.append(rect)
                ordered[index] = rect
                placed = True
                break
        if not placed:
            return None
    return list(fixed) + [ordered[i] for i in range(len(dims))]


def _path(mask, a, b, horizontal_first):
    r, c = a
    t, d = b
    if horizontal_first:
        mask[r, min(c, d):max(c, d) + 1] = True
        mask[min(r, t):max(r, t) + 1, d] = True
    else:
        mask[min(r, t):max(r, t) + 1, c] = True
        mask[t, min(c, d):max(c, d) + 1] = True


def _rows(rng, side, n):
    mode = int(rng.integers(3))
    if mode == 0:
        start = int(rng.integers(0, side - n + 1))
        return np.arange(start, start + n)
    if mode == 1:
        step = max(1, (side - 1) // max(1, n - 1))
        span = (n - 1) * step
        start = int(rng.integers(0, side - span))
        return start + step * np.arange(n)
    return np.sort(rng.choice(side, n, replace=False))


def _sequence(rng, k, palette, diagnostic=False, mono=False):
    colors = list(rng.permutation(palette))
    n = k * k
    if diagnostic:
        a, b, c, d = colors[:4]
        return np.array([a, b, b, c, c, a, d, d, d]
                        + list(rng.choice(colors, n - 9)), dtype=np.int8)
    mode = 0 if mono else int(rng.integers(5))
    if mode == 0:
        return np.full(n, colors[0], dtype=np.int8)
    if mode == 1:
        return np.array([colors[level % len(colors)] for level in range(k)
                         for _ in range(2 * level + 1)], dtype=np.int8)
    if mode == 2:
        return np.array([colors[i % len(colors)] for i in range(n)], dtype=np.int8)
    if mode == 3:
        return np.array([colors[0] if i < n // 3 or i >= 2 * n // 3 else colors[1]
                         for i in range(n)], dtype=np.int8)
    return np.array(rng.choice(colors, n), dtype=np.int8)


def _make_pair(rng, wall, palette, largest_k, witness_side, kind):
    """Compose terrain and particles directly, never call either parent."""
    if kind == "first":
        ks = [2]
    elif kind == "diagnostic":
        ks = [largest_k, 2]
    elif kind == "test":
        count = int(rng.choice([3, 4], p=[0.7, 0.3]))
        ks = [int(rng.choice([2, largest_k]))] + [2] * (count - 1)
        rng.shuffle(ks)
    else:
        count = int(rng.integers(1, 5))
        ks = [int(rng.choice([2, largest_k], p=[0.75, 0.25])) for _ in range(count)]
        if sum(k >= 4 for k in ks) > 1:
            ks = [largest_k] + [2] * (count - 1)
    for attempt in range(220):
        sides = [k * k + int(rng.integers(0, 4 if k < 4 else 3)) for k in ks]
        if kind in ("first", "diagnostic"):
            # Equal room size but unequal populations refutes size-based stamps.
            sides[0] = witness_side
        if kind == "diagnostic":
            # The shared column makes segmentation by chamber essential.
            gap = int(rng.integers(1, 4))
            min_h = sum(sides) + gap + 2
            min_w = max(sides) + 7
            if min_h > 30 or min_w > 30:
                continue
        else:
            min_h = max(12 if len(ks) == 1 else 20, max(sides) + 2)
            min_w = max(13 if len(ks) == 1 else 20, max(sides) + 2)
            if max(ks) == 4 and len(ks) >= 3:
                min_h = min_w = 28
        h = int(rng.integers(min_h, 31))
        w = int(rng.integers(min_w, 31))
        if kind == "diagnostic":
            big, small = sides
            top = int(rng.integers(1, h - big - small - gap))
            left = int(rng.integers(1, w - big))
            small_left = int(rng.integers(left, left + big - small + 1))
            fixed = [(top, left, big, big), (top + big + gap, small_left, small, small)]
            if rng.integers(2):
                fixed = [(h - r - rh, c, rh, cw) for r, c, rh, cw in fixed]
        elif len(ks) >= 2 and rng.random() < 0.30 and sum(sides[:2]) + 2 <= min(h, w):
            # Staggered side contact does not supply a complete extension side.
            first, second = sides[:2]
            top = int(rng.integers(1, h - first - second + 1))
            left = int(rng.integers(1, w - first - second))
            fixed = [(top, left, first, first),
                     (top + first - 1, left + first, second, second)]
            if rng.integers(2):
                fixed = [(h - r - rh, c, rh, cw) for r, c, rh, cw in fixed]
            if rng.integers(2):
                fixed = [(r, w - c - cw, rh, cw) for r, c, rh, cw in fixed]
        else:
            fixed = ()
        # First two demonstrations always have an ignored rectangular room.
        add_decoy = kind in ("first", "diagnostic") or rng.random() < 0.6
        decoy_shape = (int(rng.integers(5, 9)), int(rng.integers(2, 4)))
        dims = [(s, s) for s in sides[len(fixed):]] + ([decoy_shape] if add_decoy else [])
        rooms = _place(rng, dims, h, w, fixed)
        if rooms is None:
            continue
        squares = [(r, c, rh) for r, c, rh, _ in rooms[:len(ks)]]
        mask = np.zeros((h, w), dtype=bool)
        core = np.zeros_like(mask)
        for r, c, rh, cw in rooms:
            mask[r:r + rh, c:c + cw] = True
            core[r:r + rh, c:c + cw] = True
        # A connected region can contain several separately maximal squares.
        if kind == "diagnostic" or (len(ks) > 1 and rng.random() < 0.8):
            for i in range(1, len(ks)):
                j = int(rng.integers(i))
                r, c, s = squares[i]
                t, d, z = squares[j]
                _path(mask, (r + s // 2, c + s // 2),
                      (t + z // 2, d + z // 2), bool(rng.integers(2)))
        # Thin protrusions teach that open connected components are not rooms.
        for r, c, s in squares:
            for _ in range(int(rng.integers(1 if kind == "first" else 0, 4))):
                direction = int(rng.integers(4))
                length = int(rng.integers(1, 6))
                pos = int(rng.integers(s))
                if direction == 0:
                    mask[max(1, r - length):r, c + pos] = True
                elif direction == 1:
                    mask[r + s:min(h - 1, r + s + length), c + pos] = True
                elif direction == 2:
                    mask[r + pos, max(1, c - length):c] = True
                else:
                    mask[r + pos, c + s:min(w - 1, c + s + length)] = True
        if _squares(mask) != sorted(squares):
            continue
        a = np.full((h, w), wall, dtype=np.int8)
        a[mask] = 0
        common_col = None
        if kind == "diagnostic":
            lo = max(c + k - 1 for (_, c, _), k in zip(squares, ks))
            hi = min(c + s - k for (_, c, s), k in zip(squares, ks))
            if lo > hi:
                continue
            common_col = int(rng.integers(lo, hi + 1))
        for i, ((r, c, s), k) in enumerate(zip(squares, ks)):
            col = common_col if common_col is not None else int(rng.integers(c + k - 1, c + s - k + 1))
            seq = _sequence(rng, k, palette, diagnostic=kind == "diagnostic" and i == 0,
                            mono=kind == "first")
            a[r + _rows(rng, s, k * k), col] = seq
        if add_decoy:
            r, c, rh, cw = rooms[-1]
            col = c + int(rng.integers(cw))
            a[r + _rows(rng, rh, 4), col] = _sequence(rng, 2, palette)
        branches = np.argwhere(mask & ~core)
        if kind == "first" and len(branches) == 0:
            continue
        if len(branches) and (kind == "first" or rng.random() < 0.6):
            for index in rng.choice(len(branches), min(len(branches), int(rng.integers(1, 4))), replace=False):
                rr, cc = branches[index]
                a[rr, cc] = int(rng.choice(palette))
        # Build expected output using construction records, separately from solve.
        b = a.copy()
        for r, c, s in squares:
            b[r:r + s, c:c + s] = 0
            local = a[r:r + s, c:c + s]
            rr, cc = np.nonzero(local)
            seq = local[rr, cc]
            k = isqrt(len(seq))
            col = c + int(cc[0])
            offset = 0
            for level in range(k):
                width = 2 * level + 1
                b[r + s - k + level, col - level:col + level + 1] = seq[offset:offset + width]
                offset += width
        if np.array_equal(a, b):
            continue
        # Usually remove large unused wall margins, retaining a visible frame.
        if kind == "first" or rng.random() < 0.80:
            rr, cc = np.nonzero(mask)
            top = max(0, int(rr.min()) - int(rng.integers(1, 4)))
            left = max(0, int(cc.min()) - int(rng.integers(1, 4)))
            bottom = min(h, int(rr.max()) + int(rng.integers(2, 5)))
            right = min(w, int(cc.max()) + int(rng.integers(2, 5)))
            a, b = a[top:bottom, left:right].copy(), b[top:bottom, left:right].copy()
        return {"input": a, "output": b}
    raise RuntimeError(f"bounded terrain construction exhausted ({kind})")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nv = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    wall = int(rng.integers(1, 10))
    palette = list(rng.choice([x for x in range(1, 10) if x != wall], 4, replace=False))
    largest_k = int(rng.choice([3, 4], p=[0.8, 0.2]))
    witness_side = largest_k * largest_k + int(rng.integers(0, 3))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", nv)):
        for i in range(count):
            kind = "test" if split == "test" else "first" if i == 0 else "diagnostic" if i == 1 else "extra"
            for _ in range(20):
                pair = _make_pair(rng, wall, palette, largest_k, witness_side, kind)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _scan_witness(seq, k):
    """Refute reverse, alternating, and the two outside-in row traversals."""
    contradicted = [False] * 4
    for level in range(1, k):
        row = list(map(int, seq[level * level:(level + 1) ** 2]))
        n = len(row)
        outer_left = [j for i in range(n // 2) for j in (i, n - i - 1)] + [n // 2]
        outer_right = [j for i in range(n // 2) for j in (n - i - 1, i)] + [n // 2]
        orders = [list(range(n))[::-1], list(range(n))[::-1] if level % 2 else list(range(n)),
                  outer_left, outer_right]
        for i, order in enumerate(orders):
            alternative = [0] * n
            for value, pos in zip(row, order):
                alternative[pos] = value
            contradicted[i] |= alternative != row
    return contradicted


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, walls, training_ks = set(), set(), set()
    populations_by_side = {}
    evidence = Counter()
    scans = [False] * 4
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "shape changed")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            wall, streams = _chambers(a)
            walls.add(wall)
            _require(bool(streams), "no square chamber")
            selected = np.zeros_like(a, dtype=bool)
            for stream in streams:
                r, c, s = stream["square"]
                selected[r:r + s, c:c + s] = True
                k, seq = stream["k"], stream["sequence"]
                patch = b[r:r + s, c:c + s]
                _require(Counter(map(int, patch[patch != 0])) == Counter(map(int, seq)),
                         "local particle colors not conserved")
                extracted = []
                for level in range(k):
                    row = r + s - k + level
                    col = stream["column"]
                    extracted.extend(map(int, b[row, col - level:col + level + 1]))
                _require(extracted == list(map(int, seq)), "particle order or local floor wrong")
                if split == "train":
                    training_ks.add(k)
                    populations_by_side.setdefault(s, set()).add(k)
                    for i, flag in enumerate(_scan_witness(seq, k)):
                        scans[i] |= flag
                    values = list(map(int, seq))
                    evidence["color_return"] += any(
                        x == z and x != y for x, y, z in zip(values, values[1:], values[2:]))
                    # Also catch returns separated by whole bands.
                    evidence["color_return"] += any(
                        len(set(values[values.index(x):len(values) - values[::-1].index(x)])) > 1
                        for x in set(values))
                else:
                    _require(k in training_ks, "test uses an undemonstrated particle count")
            _require(np.array_equal(a[~selected], b[~selected]), "changed cells outside squares")
            _require(np.array_equal(a == wall, b == wall), "wall geometry changed")
            _require(Counter(map(int, a.ravel())) == Counter(map(int, b.ravel())),
                     "global colors not conserved")
            _require(np.array_equal(solve(a), b), "solve disagrees")
            _require(not np.array_equal(a, b), "identity example")
            if split == "train":
                outside_particles = (~selected) & (a != wall) & (a != 0)
                evidence["ignored_particles"] += int(np.any(outside_particles))
                cols = [s["column"] for s in streams]
                evidence["shared_column"] += len(cols) != len(set(cols))
                evidence["local_floor"] += any(s["square"][0] + s["square"][2] < a.shape[0] - 2 for s in streams)
                # At least one square has a partial open side attachment.
                for r, c, s in [x["square"] for x in streams]:
                    evidence["attachment"] += int(any((
                        np.any(a[r - 1, c:c + s] != wall),
                        np.any(a[r + s, c:c + s] != wall),
                        np.any(a[r:r + s, c - 1] != wall),
                        np.any(a[r:r + s, c + s] != wall))))
    _require(len(walls) == 1, "wall color changes within generated puzzle")
    _require(2 in training_ks and len(training_ks) >= 2, "missing size evidence")
    _require(any(len(values) > 1 for values in populations_by_side.values()),
             "training does not distinguish room size from particle population")
    _require(all(scans), "training fails to distinguish simple lateral scans")
    for name in ("color_return", "ignored_particles", "shared_column", "local_floor", "attachment"):
        _require(evidence[name] > 0, f"missing teaching evidence: {name}")


def _hand_cases():
    # A five-wide square with a particle-bearing partial extension, beside a
    # non-square room. Literal expected pixels do not use the solver's painter.
    a = np.full((10, 14), 5, dtype=np.int8)
    a[1:6, 1:6] = 0
    a[6:8, 3] = 0
    a[7, 3] = 9
    a[1:5, 3] = [1, 2, 3, 4]
    a[2:8, 9:12] = 0
    a[2:6, 10] = [4, 3, 2, 1]
    expected = a.copy()
    expected[1:6, 1:6] = 0
    expected[4, 3] = 1
    expected[5, 2:5] = [2, 3, 4]
    _require(_squares(a != 5) == [(1, 1, 5)], "full-side extension/subwindow bug")
    _require(np.array_equal(solve(a), expected), "hand square/decoy answer wrong")
    no_a = expected.copy()
    no_a[2:8, 9:12] = 0
    no_a[6, 10] = 4
    no_a[7, 9:12] = [3, 2, 1]
    _require(not np.array_equal(no_a, expected), "square gate has no causal effect")
    no_b = expected.copy()
    no_b[1:6, 1:6] = 0
    no_b[2:6, 3] = [1, 2, 3, 4]
    _require(not np.array_equal(no_b, expected), "odd-width packing has no causal effect")

    a = np.full((16, 9), 8, dtype=np.int8)
    a[1:6, 1:6] = 0
    a[9:13, 2:6] = 0
    a[6:9, 3] = 0
    a[1:5, 3] = [1, 2, 2, 3]
    a[9:13, 3] = [4, 6, 7, 9]
    expected = a.copy()
    expected[1:6, 1:6] = 0
    expected[9:13, 2:6] = 0
    expected[4, 3] = 1
    expected[5, 2:5] = [2, 2, 3]
    expected[11, 3] = 4
    expected[12, 2:5] = [6, 7, 9]
    _require(np.array_equal(solve(a), expected), "shared column/local floors wrong")

    # The solver's geometric domain exceeds the sampler's size preferences.
    for k, side in ((1, 2), (5, 25)):
        a = np.full((side + 2, side + 2), 6, dtype=np.int8)
        a[1:side + 1, 1:side + 1] = 0
        col = 1 + side // 2
        seq = np.array([1 + i % 4 for i in range(k * k)], dtype=np.int8)
        a[1:1 + k * k, col] = seq
        expected = a.copy()
        expected[1:side + 1, 1:side + 1] = 0
        for level in range(k):
            expected[side - k + 1 + level, col - level:col + level + 1] = seq[level ** 2:(level + 1) ** 2]
        _require(np.array_equal(solve(a), expected), "solver size boundary wrong")


def self_test() -> None:
    """Bounded 42-seed diversity and semantic checks, plus API boundaries."""
    _hand_cases()
    shapes, layouts, palettes, walls, sides, ks, counts = (set() for _ in range(7))
    features = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                wall, streams = _chambers(a)
                shapes.add(a.shape)
                walls.add(wall)
                palettes.add(tuple(map(int, np.unique(a))))
                layouts.add((a.shape, tuple((s["square"], s["column"], tuple(s["rows"])) for s in streams)))
                counts.add(len(streams))
                sides.update(s["square"][2] for s in streams)
                ks.update(s["k"] for s in streams)
                features["dimension30"] += 30 in a.shape
                features["noncentered"] += any(2 * (s["column"] - s["square"][1]) != s["square"][2] - 1 for s in streams)
                features["gaps"] += any(np.any(np.diff(s["rows"]) > 1) for s in streams)
                features["contiguous"] += any(np.all(np.diff(s["rows"]) == 1) for s in streams)
                features["bottom_particle"] += any(s["rows"][-1] == s["square"][0] + s["square"][2] - 1 for s in streams)
    _require(len(shapes) >= 40 and len(layouts) >= 200, "insufficient geometry variation")
    _require(len(palettes) >= 25 and len(walls) >= 8, "insufficient palette variation")
    _require({2, 3, 4} <= ks and {1, 2, 3, 4} <= counts and len(sides) >= 10,
             "insufficient size/count variation")
    _require(all(features[x] >= 10 for x in features), "missing structural variation")
    for nt in range(2, 11):
        for nv in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(42000 + nt * 10 + nv, nt, nv)
            _require(len(p["train"]) == nt and len(p["test"]) == nv, "counts ignored")
    for nt, nv in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nv)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(x[role], y[role]) for split in ("train", "test")
                 for x, y in zip(a[split], b[split]) for role in ("input", "output")),
             "nondeterministic")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
