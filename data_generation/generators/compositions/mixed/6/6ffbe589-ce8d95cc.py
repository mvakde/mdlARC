"""Color-controlled square rotations followed by full-vector run contraction.

Only NumPy and the standard library are used. No parent generator is imported.
The solver reads the square and unary bars from the input alone.
"""
from __future__ import annotations

import hashlib
import numpy as np


HOST_SEEDS = (0, 387161888, 1854403, 72762896, 881664183, 74676800,
              470512899, 479665948, 433502498, 807027808, 189858320,
              81524493, 171423815, 552205602, 671069047, 882810605,
              629066976, 941711553, 636011935, 169943390, 826311244,
              863727475, 569299353, 471208171, 494320935, 394245316,
              945515706, 417742625, 976539907, 846233601, 266377795,
              433774600, 474448360, 825124977, 307165338, 770829375,
              539734878, 5681394, 857338831, 896518737, 987820431,
              466968338)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _equal_edges(a, axis):
    vectors = a if axis == 0 else a.T
    return np.all(vectors[1:] == vectors[:-1], axis=1)


def _keep(a, axis):
    return np.r_[True, ~_equal_edges(a, axis)]


def _compress(a):
    # Column equality is unchanged by removal of duplicate full rows.
    return a[np.ix_(_keep(a, 0), _keep(a, 1))].copy()


def _runs(a, axis):
    starts = np.flatnonzero(_keep(a, axis))
    return tuple(int(x) for x in np.diff(np.r_[starts, a.shape[axis]]))


def _separated_repeat(a, axis):
    vectors = a if axis == 0 else a.T
    return len({v.tobytes() for v in vectors}) < len(vectors)


def _nonbackground_repeat(a, axis, bg):
    vectors = a if axis == 0 else a.T
    return any(np.array_equal(x, y) and np.any(x != bg)
               for x, y in zip(vectors, vectors[1:]))


def _locate(a):
    """Enumerate tight squares with clear halos and matched exterior bars.

    This is an input-only geometric grammar, adapted from parent A's final
    locator. Neither output dimensions nor a special frame color are needed.
    """
    values, frequencies = np.unique(a, return_counts=True)
    if np.sum(frequencies == frequencies.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(values[frequencies.argmax()])
    occupied = a != bg
    h, w = a.shape
    prefix = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def mass(r, c, y, x):
        return prefix[y, x] - prefix[r, x] - prefix[y, c] + prefix[r, c]

    candidates = []
    total = int(occupied.sum())
    for n in range(3, min(h, w) + 1):
        rr, cc = np.indices((h - n + 1, w - n + 1))
        y, x = rr + n, cc + n
        inside = mass(rr, cc, y, x)
        valid = (inside >= 5) & (inside < total)
        valid &= mass(rr, cc, rr + 1, x) > 0
        valid &= mass(y - 1, cc, y, x) > 0
        valid &= mass(rr, cc, y, cc + 1) > 0
        valid &= mass(rr, x - 1, y, x) > 0
        valid &= mass(np.maximum(0, rr - 1), np.maximum(0, cc - 1),
                      np.minimum(h, y + 1), np.minimum(w, x + 1)) == inside
        for r, c in np.argwhere(valid):
            r, c = int(r), int(c)
            square = a[r:r+n, c:c+n]
            colors = [int(v) for v in np.unique(square) if v != bg]
            if len(colors) < 2:
                continue
            exterior = occupied.copy()
            exterior[r:r+n, c:c+n] = False
            counts, bars = {}, {}
            good = True
            for v in np.unique(a[exterior]):
                color = int(v)
                coordinates = np.argwhere(exterior & (a == color))
                rows, cols = coordinates.T
                length = len(coordinates)
                horizontal = np.ptp(rows) == 0 and np.ptp(cols) + 1 == length
                vertical = np.ptp(cols) == 0 and np.ptp(rows) + 1 == length
                if color not in colors or not (horizontal or vertical):
                    good = False
                    break
                counts[color] = length
                bars[color] = coordinates
            if not good or not counts:
                continue
            points = [(v, p) for v, coords in bars.items() for p in coords]
            if any(v != u and np.max(np.abs(p-q)) <= 1
                   for i, (v, p) in enumerate(points) for u, q in points[i+1:]):
                continue
            candidates.append((square.copy(), bg, counts, (r, c), bars))
    if len(candidates) != 1:
        raise ValueError("input must have exactly one identifiable source square")
    return candidates[0]


def _rotate_layers(square, bg, counts, direction=1, absent=0):
    result = np.full_like(square, bg)
    for value in np.unique(square):
        color = int(value)
        if color == bg:
            continue
        mask = np.rot90(square == color, -direction * counts.get(color, absent))
        if np.any(mask & (result != bg)):
            raise ValueError("rotated color masks overlap; no overwrite convention")
        result[mask] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Count bars, rotate independent masks clockwise, contract equal vectors."""
    square, bg, counts, _, _ = _locate(_grid(input_grid))
    return _compress(_rotate_layers(square, bg, counts))


def _coordinate_rotation(square, bg, counts):
    """A coordinate implementation for validation, separate from np.rot90."""
    n = len(square)
    result = np.full_like(square, bg)
    for r, c in np.argwhere(square != bg):
        color = int(square[r, c])
        y, x = int(r), int(c)
        for _ in range(counts.get(color, 0) % 4):
            y, x = x, n - 1 - y
        assert result[y, x] == bg, "overlapping layers"
        result[y, x] = color
    return result


def _reference_compression(a):
    rows = [0] + [i for i in range(1, len(a)) if not np.array_equal(a[i], a[i-1])]
    columns = [0] + [j for j in range(1, a.shape[1])
                     if not np.array_equal(a[:, j], a[:, j-1])]
    return a[np.ix_(rows, columns)].copy()


def _wrong_result(square, bg, counts, *, direction=1, absent=0):
    try:
        return _compress(_rotate_layers(square, bg, counts, direction, absent))
    except ValueError:
        return None


def _differs(a, b):
    return a is None or not np.array_equal(a, b)


def _analyze(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    square, bg, counts, position, bars = _locate(inp)
    rotated = _coordinate_rotation(square, bg, counts)
    expected = _reference_compression(rotated)
    assert np.array_equal(expected, out), "incorrect rotation or contraction"
    row_runs, col_runs = _runs(rotated, 0), _runs(rotated, 1)
    assert np.array_equal(np.repeat(np.repeat(out, row_runs, 0), col_runs, 1), rotated)
    for color in np.unique(square):
        if color != bg:
            assert np.sum(square == color) == np.sum(rotated == color)
    assert not np.array_equal(out, _compress(square)), "rotation must matter"
    assert out.shape != rotated.shape, "compression must matter"
    gained = [int(np.sum(_equal_edges(rotated, axis) & ~_equal_edges(square, axis)))
              for axis in (0, 1)]
    assert sum(gained) > 0, "rotation must create equality consumed by compression"
    return {"square": square, "rotated": rotated, "background": bg,
            "counts": counts, "position": position, "bars": bars,
            "row_runs": row_runs, "column_runs": col_runs, "gained": gained}


def _motif(rng, h, w):
    mode = int(rng.integers(0, 5))
    m = np.ones((h, w), dtype=bool)
    if mode == 1 and min(h, w) >= 3:
        m[1:-1, 1:-1] = False  # Hollow rectangular loop.
    elif mode == 2:
        m[:] = False
        m[:, 0] = True
        m[-1, :] = True  # Bent ribbon.
    elif mode == 3:
        m[:] = False
        m[int(rng.integers(h)), :] = True
        m[:, int(rng.integers(w))] = True  # Cross; its values remain literal.
    elif mode == 4:
        m[:] = False
        m[::2, :] = True  # Separated parallel strips of one layer.
    return np.rot90(m, int(rng.integers(4))).copy()


def _make_square(rng, n, bg, colors, counts):
    """Place procedural target motifs only where their inverse masks also fit.

    This builds a new joint scene, not a call to either whole-puzzle parent.
    The corner orbit establishes the crop; extra motifs make its color layer
    nontrivial. Other layers can have several connected or separated pieces.
    """
    target = np.full((n, n), bg, dtype=np.int8)
    source = target.copy()
    owner = int(rng.choice(colors))
    thickness = int(rng.integers(1, 3))
    for r in (0, n-thickness):
        for c in (0, n-thickness):
            target[r:r+thickness, c:c+thickness] = owner
            source[r:r+thickness, c:c+thickness] = owner
    for color in rng.permutation(colors):
        color = int(color)
        wanted = int(rng.integers(1, 4))
        placed = 0
        for _ in range(100):
            h, w = (int(x) for x in rng.integers(1, min(7, n-3), size=2))
            motif = _motif(rng, h, w)
            h, w = motif.shape
            r = int(rng.integers(thickness, n-thickness-h+1)) if n-2*thickness >= h else -1
            c = int(rng.integers(thickness, n-thickness-w+1)) if n-2*thickness >= w else -1
            if r < 0 or c < 0:
                continue
            mask = np.zeros((n, n), dtype=bool)
            mask[r:r+h, c:c+w] = motif
            inverse = np.rot90(mask, counts.get(color, 0))
            if np.any(mask & (target != bg) & (target != color)):
                continue
            if np.any(inverse & (source != bg) & (source != color)):
                continue
            target[mask] = color
            source[inverse] = color
            placed += 1
            if placed == wanted:
                break
        if not placed:
            return None
    return source, target


def _bar_starts(blocked, length, vertical):
    h, w = blocked.shape
    bh, bw = (length, 1) if vertical else (1, length)
    rr, cc = np.indices((h-bh+1, w-bw+1))
    clear = np.ones(rr.shape, dtype=bool)
    for k in range(length):
        clear &= ~blocked[rr + (k if vertical else 0), cc + (0 if vertical else k)]
    return np.argwhere(clear)


def _canvas(rng, source, bg, counts, force_mixed=False):
    n = len(source)
    for _ in range(16):
        h, w = (int(x) for x in rng.integers(max(n+4, 18), 31, size=2))
        r = int(rng.integers(h-n+1))
        c = int(rng.integers(w-n+1))
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[r:r+n, c:c+n] = source
        blocked = np.zeros((h, w), dtype=bool)
        blocked[max(0, r-1):min(h, r+n+1), max(0, c-1):min(w, c+n+1)] = True
        marked = sorted(counts, key=lambda color: -counts[color])
        success = True
        for j, color in enumerate(marked):
            length = counts[color]
            orientations = [bool(rng.integers(2))]
            orientations.append(not orientations[0])
            if force_mixed and length == 2:
                # The two equal-count bars of training example 1 differ in direction.
                orientations = [sum(counts[v] == 2 for v in marked[:j]) % 2 == 0]
            for vertical in orientations:
                starts = _bar_starts(blocked, length, vertical)
                if len(starts):
                    y, x = starts[int(rng.integers(len(starts)))]
                    yy = y + (np.arange(length) if vertical else np.zeros(length, dtype=int))
                    xx = x + (np.zeros(length, dtype=int) if vertical else np.arange(length))
                    inp[yy, xx] = color
                    for u, v in zip(yy, xx):
                        blocked[max(0, u-1):min(h, u+2), max(0, v-1):min(w, v+2)] = True
                    break
            else:
                success = False
                break
        if not success:
            continue
        try:
            found = _locate(inp)
        except ValueError:
            continue
        if found[3] == (r, c) and np.array_equal(found[0], source):
            return inp
    return None


def _make_pair(rng, split, index):
    # A bounded rejection loop enforces semantic witnesses, not a finite bank.
    for _ in range(180):
        n = int(rng.integers(10, 23))
        layer_count = 4 if split == "train" and index < 2 else int(rng.integers(2, 7))
        palette = [int(x) for x in rng.choice(10, layer_count+1, replace=False)]
        bg, colors = palette[0], palette[1:]
        if split == "train" and index == 0:
            lengths = [1, 2, 2, 0]
        elif split == "train" and index == 1:
            lengths = [3, 4, 5, 0]
        else:
            lengths = [int(x) for x in rng.integers(0, 6 if split == "train" else 10, layer_count)]
            if split == "test":
                lengths[int(rng.integers(layer_count))] = int(rng.integers(6, 10))
        counts = {color: k for color, k in zip(colors, lengths) if k}
        if not counts:
            continue
        made = _make_square(rng, n, bg, colors, counts)
        if made is None:
            continue
        source, target = made
        out = _compress(target)
        if np.array_equal(out, _compress(source)) or out.shape == target.shape:
            continue
        if not any(np.any(_equal_edges(target, axis) & ~_equal_edges(source, axis)) for axis in (0, 1)):
            continue
        if not all(_separated_repeat(out, axis) for axis in (0, 1)):
            continue
        if split == "train" and index < 2:
            if not all(len(_runs(target, axis)) < n for axis in (0, 1)):
                continue
            if not all(_nonbackground_repeat(target, axis, bg) for axis in (0, 1)):
                continue
            if not _differs(_wrong_result(source, bg, counts, direction=-1), out):
                continue
            if not _differs(_wrong_result(source, bg, counts, absent=1), out):
                continue
            if any(not _differs(_wrong_result(source, bg, {v:k for v in colors}), out)
                   for k in range(4)):
                continue
            # Every nonzero, non-full-cycle mask must actually move. Compression
            # must not disguise the absent-counter teaching witness either.
            if any(k % 4 and np.array_equal(source == v, target == v)
                   for v, k in counts.items()):
                continue
            if index == 1:
                ranked = {v: sorted(set(counts.values())).index(k)+1 for v, k in counts.items()}
                clamped = {v: min(k, 3) for v, k in counts.items()}
                if any(not _differs(_wrong_result(source, bg, alternative), out)
                       for alternative in (ranked, clamped)):
                    continue
        inp = _canvas(rng, source, bg, counts, split == "train" and index == 0)
        if inp is not None:
            return {"input": inp, "output": out}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(20):
                pair = _make_pair(rng, split, i)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            assert 10 <= len(info["square"]) <= 22
            assert 2 <= len(np.unique(info["square"])) - 1 <= 6
            assert all(1 <= k <= 9 for k in info["counts"].values())
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert np.array_equal(solve(pair["input"]), pair["output"])
            if split == "train":
                assert max(info["counts"].values()) <= 5
                train_info.append(info)
            else:
                assert max(info["counts"].values()) >= 6
    for pair, info in zip(puzzle["train"][:2], train_info[:2]):
        s, b, counts = info["square"], info["background"], info["counts"]
        out = np.asarray(pair["output"])
        assert max(info["row_runs"]) > 1 and max(info["column_runs"]) > 1
        assert all(_nonbackground_repeat(info["rotated"], axis, b) for axis in (0, 1))
        assert all(_separated_repeat(out, axis) for axis in (0, 1))
        assert _differs(_wrong_result(s, b, counts, direction=-1), out)
        assert _differs(_wrong_result(s, b, counts, absent=1), out)
        assert all(_differs(_wrong_result(s, b, {int(v): k for v in np.unique(s) if v != b}), out)
                   for k in range(4)), "a whole-square rotation fits a demonstration"
    assert {1, 2, 3, 4, 5} <= {k for info in train_info for k in info["counts"].values()}
    repeated = [points for color, points in train_info[0]["bars"].items()
                if train_info[0]["counts"][color] == 2]
    assert len(repeated) == 2
    assert {int(np.ptp(points[:, 0]) > 0) for points in repeated} == {0, 1}
    second = train_info[1]
    counts = second["counts"]
    ranked = {v: sorted(set(counts.values())).index(k)+1 for v, k in counts.items()}
    clamped = {v: min(k, 3) for v, k in counts.items()}
    assert all(_differs(_wrong_result(second["square"], second["background"], alternative),
                        puzzle["train"][1]["output"]) for alternative in (ranked, clamped))


def self_test():
    """Bounded, in-memory 42-seed diversity and semantic regression checks."""
    shapes, output_shapes, normalized, backgrounds, sizes, layers, puzzles = (set() for _ in range(7))
    rectangular = nonbackground_runs = gained_rows = gained_columns = 0
    orientations = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair)
                a, out = pair["input"], pair["output"]
                digest.update(a.tobytes())
                shapes.add(a.shape)
                output_shapes.add(out.shape)
                sizes.add(len(info["square"]))
                layers.add(len(np.unique(info["square"]))-1)
                backgrounds.add(info["background"])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in out.flat)
                normalized.add((out.shape, norm))
                rectangular += out.shape[0] != out.shape[1]
                gained_rows += info["gained"][0] > 0
                gained_columns += info["gained"][1] > 0
                for axis in (0, 1):
                    vectors = info["rotated"] if axis == 0 else info["rotated"].T
                    nonbackground_runs += any(np.array_equal(x, y) and np.any(x != info["background"])
                                              for x, y in zip(vectors, vectors[1:]))
                for color, points in info["bars"].items():
                    if len(points) > 1:
                        orientations.add((info["counts"][color], int(np.ptp(points[:, 0]) > 0)))
        puzzles.add(digest.digest())
    assert len(puzzles) == 42 and len(normalized) >= 150
    assert len(shapes) >= 40 and len(output_shapes) >= 25 and len(sizes) >= 10
    assert backgrounds == set(range(10)) and layers == set(range(2, 7))
    assert rectangular >= 40 and nonbackground_runs >= 50
    assert gained_rows >= 30 and gained_columns >= 30
    assert {(2, 0), (2, 1), (5, 0), (5, 1)} <= orientations
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(125, a, b)
            validate_generated_puzzle(p)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    first, second = generate_puzzle(710, 2, 1), generate_puzzle(710, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for split in first
               for x, y in zip(first[split], second[split]) for k in x)
    # Nonadjacent repeats survive; row/column order commutes; full-cycle count.
    fixture = np.array([[1,1,0,1], [1,1,0,1], [0,0,2,0], [1,1,0,1]], dtype=np.int8)
    expected = np.array([[1,0,1], [0,2,0], [1,0,1]], dtype=np.int8)
    assert np.array_equal(_compress(fixture), expected)
    assert np.array_equal(_compress(fixture.T).T, expected)
    assert np.array_equal(_rotate_layers(fixture, 0, {1:4, 2:8}), fixture)
    # The input-only locator also handles a tight square without corner marks,
    # and visibly equivalent unary counts 1 and 5 on the same source geometry.
    source = np.array([[0,0,1,0,0], [0,2,2,0,0], [1,0,0,0,1],
                       [0,0,0,0,0], [0,0,1,0,0]], dtype=np.int8)
    predictions = []
    for length in (1, 5):
        inp = np.zeros((11, 14), dtype=np.int8)
        inp[1:6, 1:6] = source
        inp[:4, 12] = 1
        inp[9, 1:1+length] = 2
        predictions.append(solve(inp))
    assert np.array_equal(predictions[0], predictions[1])
    assert np.array_equal(predictions[0], _rotate_layers(source, 0, {1:4, 2:1}))
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad = corrupted["test"][0]["output"]
    bad[0, 0] = (int(bad[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
