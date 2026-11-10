"""Clockwise ring compaction followed by a crop chosen by canvas parity.

Only NumPy and the standard library are used. Generation never writes files.
Point order is outer NW, NE, SE, SW; inner NW, NE, SE, SW; center.
"""
from __future__ import annotations

import hashlib
import numpy as np


_COMPACT_ORDER = np.array(((0, 4, 1), (7, 8, 5), (3, 6, 2)))
_CROP_SOURCES = {1: (0, 4, 7, 8), 0: (4, 1, 8, 5)}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape) or
            not np.issubdtype(array.dtype, np.integer) or
            not np.all((0 <= array) & (array <= 9))):
        raise ValueError("grids need integer colors 0..9 and dimensions 1..30")
    return array


def _positions(geometry):
    n, oy, ox, iy, ix, top, left = geometry
    cy, cx = top + oy, left + ox
    return ((cy-oy, cx-ox), (cy-oy, cx+ox),
            (cy+oy, cx+ox), (cy+oy, cx-ox),
            (cy-iy, cx-ix), (cy-iy, cx+ix),
            (cy+iy, cx+ix), (cy+iy, cx-ix), (cy, cx))


def _recover(value):
    """Read geometry and literal colors from the input, including repeats."""
    grid = _grid(value)
    n, width = grid.shape
    if n != width or n < 5:
        raise ValueError("input must be a square of side 5..30")
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= grid.size // 2:
        raise ValueError("background must be the unique strict majority")
    rows, cols = np.nonzero(grid != background)
    if len(rows) != 9:
        raise ValueError("exactly nine visible points are required")
    ys, xs = sorted(set(map(int, rows))), sorted(set(map(int, cols)))
    if len(ys) != 5 or len(xs) != 5:
        raise ValueError("two nested rings need five row and column levels")
    cy, cx = ys[2], xs[2]
    oy, ox, iy, ix = cy-ys[0], cx-xs[0], cy-ys[1], cx-xs[1]
    if not (0 < iy < oy and 0 < ix < ox):
        raise ValueError("rings must be strictly nested")
    if (ys != [cy-oy, cy-iy, cy, cy+iy, cy+oy] or
            xs != [cx-ox, cx-ix, cx, cx+ix, cx+ox]):
        raise ValueError("rings must have a common center")
    geometry = (n, oy, ox, iy, ix, ys[0], xs[0])
    positions = _positions(geometry)
    if set(zip(map(int, rows), map(int, cols))) != set(positions):
        raise ValueError("foreground must consist only of both rings and center")
    points = tuple(int(grid[r, c]) for r, c in positions)
    return geometry, background, points


def _compact(points):
    return np.asarray(points, dtype=np.int8)[_COMPACT_ORDER].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compact all nine points, then crop using ORIGINAL square side parity."""
    geometry, _, points = _recover(input_grid)
    compact = _compact(points)
    return (compact[:2, :2] if geometry[0] % 2 else compact[:2, 1:]).copy()


def _expected(points, side):
    # Direct four-source check, separate from solve's intermediate 3x3 board.
    return np.array([points[i] for i in _CROP_SOURCES[side % 2]],
                    dtype=np.int8).reshape(2, 2)


def _build(geometry, background, points):
    n = geometry[0]
    inp = np.full((n, n), background, dtype=np.int8)
    for (r, c), color in zip(_positions(geometry), points):
        inp[r, c] = color
    return {"input": inp, "output": _expected(points, n)}


def _is_rectangle(geometry):
    _, oy, ox, iy, ix, _, _ = geometry
    return oy != ox or iy != ix


def _sample_geometry(rng, side, rectangular, cap=14):
    cap = min(cap, (side-1)//2)
    if rectangular and cap < 3:
        raise ValueError("rectangular rings require a canvas of side at least 7")
    oy = int(rng.integers(2, cap+1))
    ox = int(rng.integers(2, cap+1)) if rectangular else oy
    iy = int(rng.integers(1, oy))
    ix = int(rng.integers(1, ox)) if rectangular else iy
    if rectangular and oy == ox and iy == ix:
        ox = 2 if oy != 2 else 3
        ix = int(rng.integers(1, ox))
    top = int(rng.integers(0, side-2*oy))
    left = int(rng.integers(0, side-2*ox))
    return side, oy, ox, iy, ix, top, left


def _colors(rng, background, mode):
    palette = rng.permutation([c for c in range(10) if c != background])
    if mode == "distinct":
        return tuple(map(int, palette))
    if mode == "diagonal":
        a, b, c, d, e = map(int, palette[:5])
        return a, b, a, b, c, d, c, d, e
    if mode == "rings":
        a, b, c = map(int, palette[:3])
        return (a,)*4 + (b,)*4 + (c,)
    if mode == "solid":
        return (int(palette[0]),)*9
    if mode == "reuse":
        count = int(rng.integers(2, 7))
        points = list(map(int, rng.choice(palette[:count], size=9)))
        points[4], points[8] = points[0], points[5]
        return tuple(points)
    count = int(rng.integers(2, 10))
    points = list(map(int, palette[:count]))
    points += list(map(int, rng.choice(palette[:count], size=9-count)))
    rng.shuffle(points)
    return tuple(points)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _pad_pair(pair):
    geometry, background, points = _recover(pair["input"])
    return _build((geometry[0]+1, *geometry[1:]), background, points)


def _random_pair(rng, side, rectangular, cap, seen_inputs, seen_geometry):
    modes = ("distinct", "diagonal", "rings", "solid", "reuse", "partial")
    for _ in range(64):
        geometry = _sample_geometry(rng, side, rectangular, cap)
        if geometry in seen_geometry:
            continue
        background = int(rng.integers(10))
        points = _colors(rng, background, str(rng.choice(modes)))
        pair = _build(geometry, background, points)
        if _key(pair["input"]) not in seen_inputs:
            return pair
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Two distinctly colored witnesses cover both parity branches even at 2/1.
    n = int(rng.integers(5, 11))
    # At four train pairs, at least one even witness must be 2 modulo 4;
    # otherwise a divisible-by-four substitute can agree with both evens.
    p_even = n if n % 2 == 0 else n+1
    candidates = [v for v in range(n+3, 19) if v % 2 != n % 2]
    if p_even % 4 == 0:
        candidates = [v for v in candidates if (v if v % 2 == 0 else v+1) % 4 == 2]
    m = int(rng.choice(candidates))
    p_geometry = _sample_geometry(rng, n, False, cap=4)
    rectangles = bool(rng.random() < .65)
    for _ in range(64):
        q_geometry = _sample_geometry(rng, m, rectangles, cap=6)
        if q_geometry[1:5] != p_geometry[1:5]:
            break
    else:
        # A deterministic alternative scale fits m >= 8.
        radii = (3, 2, 1, 1) if rectangles else (3, 3, 1, 1)
        if radii == p_geometry[1:5]:
            radii = (3, 3, 2, 2)
        q_geometry = (m, *radii, 0, 0)
    p_background = int(rng.integers(10))
    q_background = (p_background+int(rng.integers(1, 10))) % 10
    p = _build(p_geometry, p_background, _colors(rng, p_background, "distinct"))
    q = _build(q_geometry, q_background, _colors(rng, q_background, "distinct"))
    train = [p, q]
    if num_train >= 3:
        train.append(_pad_pair(p))
    if num_train >= 4:
        train.append(_pad_pair(q))
    seen_inputs = {_key(pair["input"]) for pair in train}
    seen_geometry = {_recover(pair["input"])[0] for pair in train}
    rectangles = any(_is_rectangle(g) for g in seen_geometry)
    while len(train) < num_train:
        for _ in range(64):
            side = int(rng.integers(5, 21))
            rectangular = rectangles and side >= 7 and rng.random() < .5
            pair = _random_pair(rng, side, rectangular, 6, seen_inputs, seen_geometry)
            if pair is not None:
                break
        else:
            raise RuntimeError("bounded training construction exhausted")
        train.append(pair)
        seen_inputs.add(_key(pair["input"]))
        seen_geometry.add(_recover(pair["input"])[0])
    rng.shuffle(train)
    test = []
    first_parity = int(rng.integers(2))
    for index in range(num_test):
        parity = (first_parity+index) % 2
        for _ in range(64):
            lo, hi = ((5, 12), (13, 20), (21, 30))[int(rng.choice(3, p=[.3, .35, .35]))]
            choices = [v for v in range(lo, hi+1) if v % 2 == parity]
            side = int(rng.choice(choices))
            rectangular = rectangles and side >= 7 and rng.random() < .6
            pair = _random_pair(rng, side, rectangular, 14, seen_inputs, seen_geometry)
            if pair is not None:
                break
        else:
            raise RuntimeError("bounded test construction exhausted")
        test.append(pair)
        seen_inputs.add(_key(pair["input"]))
        seen_geometry.add(_recover(pair["input"])[0])
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _padding_contrasts(train):
    count = 0
    for small in train:
        g = np.asarray(small["input"])
        n = g.shape[0]
        _, bg, _ = _recover(g)
        for large in train:
            h = np.asarray(large["input"])
            if (h.shape == (n+1, n+1) and np.array_equal(h[:n, :n], g) and
                    np.all(h[-1, :] == bg) and np.all(h[:, -1] == bg) and
                    not np.array_equal(small["output"], large["output"])):
                count += 1
    return count


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, records = set(), {"train": [], "test": []}
    learned = {0: set(), 1: set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must have exactly input/output")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            geometry, bg, points = _recover(grid)
            _require(out.shape == (2, 2), "outputs must be 2x2")
            _require(np.array_equal(out, _expected(points, geometry[0])),
                     "wrong literal source colors or parity crop")
            _require(np.array_equal(out, solve(grid)), "solve disagrees")
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append((geometry, bg, points))
            if split == "train" and len(set(points)) == 9:
                lookup = {color: i for i, color in enumerate(points)}
                learned[geometry[0] % 2].add(tuple(lookup[int(v)] for v in out.flat))
                # Concrete ablations for both inherited contributions.
                raw = grid[:2, :2] if geometry[0] % 2 else grid[:2, -2:]
                _require(not np.array_equal(raw, out), "raw-canvas shortcut survived")
                wrong = _compact(points)[:2, 1:] if geometry[0] % 2 else _compact(points)[:2, :2]
                _require(not np.array_equal(wrong, out), "parity is not observable")
                cc_order = np.array(((0, 5, 1), (4, 8, 6), (3, 7, 2)))
                counterclockwise = np.array(points)[cc_order]
                cc_crop = counterclockwise[:2, :2] if geometry[0] % 2 else counterclockwise[:2, 1:]
                _require(not np.array_equal(cc_crop, out), "clockwise direction is not observable")
    _require(learned == {0: {_CROP_SOURCES[0]}, 1: {_CROP_SOURCES[1]}},
             "both parities need fully asymmetric training witnesses")
    training = records["train"]
    _require(len({bg for _, bg, _ in training}) >= 2, "backgrounds must vary in train")
    _require(len({g[1:5] for g, _, _ in training}) >= 2, "train must vary ring geometry")
    allow_rectangles = any(_is_rectangle(g) for g, _, _ in training)
    train_geometry = {g for g, _, _ in training}
    for pair, (geometry, _, points) in zip(puzzle["test"], records["test"]):
        _require(geometry not in train_geometry, "test geometry must be a new combination")
        _require(not _is_rectangle(geometry) or allow_rectangles,
                 "rectangular tests need rectangular training")
        order = next(iter(learned[geometry[0] % 2]))
        prediction = np.array([points[i] for i in order]).reshape(2, 2)
        _require(np.array_equal(prediction, pair["output"]), "train-derived transfer failed")
    if nq >= 2:
        _require({g[0] % 2 for g, _, _ in records["test"]} == {0, 1}, "test parity coverage")
    if nt >= 3:
        _require(_padding_contrasts(puzzle["train"]) >= min(2, nt-2),
                 "missing padding-only parity contrasts")
    if nt >= 4:
        _require(any(g[0] % 4 == 2 for g, _, _ in training),
                 "training must distinguish even from divisible by four")
        for parity in (0, 1):
            _require(len({g[0] for g, _, _ in training if g[0] % 2 == parity}) >= 2,
                     "default train must separate parity from scale")


def self_test():
    """42 streamed seeds, all legal counts, metamorphic checks; no disk output."""
    digests, geometries, sides, backgrounds, richness, structures = (set() for _ in range(6))
    counts = dict(pairs=0, rectangular=0, offset=0, outer_gap=0, inner_gap=0,
                  reused=0, uniform=0, odd_tests=0, even_tests=0,
                  large_tests=0, overlapping_tests=0, combined_tests=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "determinism")
        digests.add(_puzzle_digest(puzzle))
        max_train_side = max(np.asarray(p["input"]).shape[0] for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                g, bg, points = _recover(grid)
                n, oy, ox, iy, ix, top, left = g
                geometries.add(g)
                sides.add(n)
                backgrounds.add(bg)
                richness.add(len(set(points)))
                mask = np.asarray(grid != bg, dtype=np.uint8)
                structures.add((n, hashlib.sha256(mask.tobytes()).hexdigest()))
                rect = _is_rectangle(g)
                offset = len({top, n-1-top-2*oy, left, n-1-left-2*ox}) > 1
                reuse = bool(set(points[:4]) & set(points[4:8])) or points[8] in points[:8]
                counts["pairs"] += 1
                counts["rectangular"] += rect
                counts["offset"] += offset
                counts["outer_gap"] += oy-iy > 1 or ox-ix > 1
                counts["inner_gap"] += iy > 1 or ix > 1
                counts["reused"] += reuse
                counts["uniform"] += len(set(points)) == 1
                if split == "test":
                    counts["odd_tests" if n % 2 else "even_tests"] += 1
                    counts["large_tests"] += n >= 21
                    counts["overlapping_tests"] += n <= max_train_side
                    counts["combined_tests"] += rect and offset and reuse
        pair = puzzle["train"][0]
        g, bg, points = _recover(pair["input"])
        n, oy, ox, iy, ix, top, left = g
        # Recoloring is equivariant, even when zero changes semantic role.
        permutation = np.random.default_rng(seed+104729).permutation(10)
        _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                 "literal color equivariance")
        once = np.pad(pair["input"], ((0, 1), (0, 1)), constant_values=bg)
        twice = np.pad(pair["input"], ((0, 2), (0, 2)), constant_values=bg)
        _require(np.array_equal(solve(once), _expected(points, n+1)), "padding must flip branch")
        _require(not np.array_equal(solve(once), pair["output"]), "padding flip lost contrast")
        _require(np.array_equal(solve(twice), pair["output"]), "two padding layers preserve branch")
        translated = _build((n, oy, ox, iy, ix, n-1-2*oy-top, n-1-2*ox-left), bg, points)
        _require(np.array_equal(solve(translated["input"]), pair["output"]), "translation invariance")
        # These three southern slots are always discarded by the required crop.
        changed = list(points)
        for index in (2, 3, 6):
            changed[index] = next(c for c in range(10) if c not in (bg, points[index]))
        discarded = _build(g, bg, changed)
        _require(np.array_equal(solve(discarded["input"]), pair["output"]), "discarded slots leaked")
    _require(len(digests) == 42, "seed diversity")
    _require(len(sides) >= 20 and len(geometries) >= 100 and len(structures) >= 100,
             "structural diversity")
    _require(len(backgrounds) == 10 and len(richness) >= 5, "palette diversity")
    _require(min(counts[k] for k in ("rectangular", "offset", "outer_gap", "inner_gap")) >= 15,
             "geometry axes missing")
    _require(counts["reused"] >= 10 and counts["uniform"] >= 2, "color equality cases missing")
    _require(min(counts[k] for k in ("odd_tests", "even_tests", "large_tests", "overlapping_tests")) >= 5,
             "test shift or parity collapsed")
    _require(counts["combined_tests"] >= 2, "combined structural/color variation missing")
    legal_counts = 0
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(7000+a*10+b, a, b)
            legal_counts += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Tight 5x5 and maximal 30x30 boundaries, including repeated foreground.
    for g in ((5, 2, 2, 1, 1, 0, 0), (30, 14, 13, 13, 1, 1, 3)):
        for points in (tuple(range(1, 10)), (7,)*9):
            pair = _build(g, 0, points)
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "boundary behavior")
    return {"seeds": 42, "distinct_puzzles": len(digests), "legal_counts": legal_counts,
            "sides": sorted(sides), "geometries": len(geometries),
            "foreground_color_counts": sorted(richness), "backgrounds": sorted(backgrounds),
            "foreground_masks": len(structures), **counts}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
