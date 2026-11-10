"""Recover logical cells before frequency packing; no files or hidden solver state.

Generation and tests retain no puzzles, traces, logs, or caches on disk.
"""

from collections import Counter
import hashlib

import numpy as np


_SHAPES = tuple((h, w) for h in range(2, 9) for w in range(2, 9))
_WITNESS_SHAPES = ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4),
                   (4, 5), (5, 4), (5, 5))
_HOST_SEEDS = (0, 710353724, 317418005, 888056782, 737631949, 338925971,
               832238350, 45995125, 394753452, 378063292, 140106892,
               939309873, 854991147, 920733436, 249263900, 583415301,
               712782692, 30678736, 296466070, 253462411, 107465678,
               203274875, 200750726, 8016308, 95481543, 90641781, 63814848,
               276087319, 471256253, 374296018, 6946348, 105524442,
               252352976, 71598576, 402277484, 171978225, 888697317,
               377984723, 428154957, 184903416, 865200846, 979485848)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid with colors 0..9")
    return array.astype(np.int8, copy=False)


def _recover(value):
    """Crop the visibly uniform exterior, then collapse maximal adjacent runs."""
    grid = _grid(value)
    background = int(grid[0, 0])
    if not grid[0, -1] == grid[-1, 0] == grid[-1, -1] == background:
        raise ValueError("All corners must identify the exterior color")
    occupied = grid != background
    rr, cc = np.nonzero(occupied)
    if not len(rr):
        raise ValueError("Missing mosaic")
    top, bottom, left, right = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    crop = grid[top:bottom, left:right]
    margins = (top, grid.shape[0] - bottom, left, grid.shape[1] - right)
    if np.any(crop == background) or margins.count(0) > 1:
        raise ValueError("Expected one filled rectangular mosaic with at most one edge contact")
    row_starts = np.r_[0, np.flatnonzero(np.any(crop[1:] != crop[:-1], axis=1)) + 1]
    col_starts = np.r_[0, np.flatnonzero(np.any(crop[:, 1:] != crop[:, :-1], axis=0)) + 1]
    mosaic = crop[np.ix_(row_starts, col_starts)].copy()
    rows = np.diff(np.r_[row_starts, crop.shape[0]])
    columns = np.diff(np.r_[col_starts, crop.shape[1]])
    return mosaic, rows, columns, background, margins


def _rank(mosaic):
    colors, counts = np.unique(mosaic, return_counts=True)
    if len(colors) < 2 or len(set(map(int, counts))) != len(colors):
        raise ValueError("Logical color frequencies must be positive and pairwise distinct")
    order = np.argsort(-counts)
    return colors[order], counts[order]


def _pack(colors, counts, shape):
    height, width = shape
    return np.repeat(colors, counts).reshape(width, height)[:, ::-1].T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """The output dimensions and exact inventory both come from run collapse."""
    mosaic, _, _, _, _ = _recover(input_grid)
    return _pack(*_rank(mosaic), mosaic.shape)


def _visible(mosaic):
    return (not np.any(np.all(mosaic[1:] == mosaic[:-1], axis=1))
            and not np.any(np.all(mosaic[:, 1:] == mosaic[:, :-1], axis=0)))


def _seam(mosaic):
    """A logical boundary exposed by exactly one cell along its full vector."""
    return any(np.any(np.count_nonzero(np.diff(a, axis=1), axis=0) == 1)
               for a in (mosaic, mosaic.T))


def _returning_vector(mosaic):
    return any(np.array_equal(a[i], a[j]) for a in (mosaic, mosaic.T)
               for i in range(a.shape[0]) for j in range(i + 2, a.shape[0]))


def _pixel_counts(mosaic, rows, columns):
    weights = rows[:, None] * columns[None, :]
    return np.bincount(mosaic.ravel(), weights=weights.ravel(), minlength=10).astype(int)


def _reversal(mosaic, rows, columns):
    logical = np.bincount(mosaic.ravel(), minlength=10)
    pixels = _pixel_counts(mosaic, rows, columns)
    colors = np.flatnonzero(logical)
    return any(logical[a] > logical[b] and pixels[a] < pixels[b]
               for a in colors for b in colors)


def _traversal_pack(mosaic, major, reverse_major, reverse_minor):
    """Explicit coordinates for finite alternative-rule diagnostics, no inference."""
    colors, counts = _rank(mosaic)
    height, width = mosaic.shape
    outer, inner = (width, height) if major == "column" else (height, width)
    result = np.empty_like(mosaic)
    positions = []
    for a in (range(outer - 1, -1, -1) if reverse_major else range(outer)):
        for b in (range(inner - 1, -1, -1) if reverse_minor else range(inner)):
            positions.append((b, a) if major == "column" else (a, b))
    for position, color in zip(positions, np.repeat(colors, counts)):
        result[position] = color
    return result


def _direction_witness(mosaic):
    expected = _pack(*_rank(mosaic), mosaic.shape)
    return all(not np.array_equal(expected, _traversal_pack(mosaic, major, rev_a, rev_b))
               for major in ("column", "row") for rev_a in (False, True)
               for rev_b in (False, True)
               if (major, rev_a, rev_b) != ("column", False, True))


def _draw_mosaic(rng, shape, kind, background):
    """One bounded candidate draw. Structured copies are checked after insertion."""
    total = int(np.prod(shape))
    maximum = min(9, int((np.sqrt(1 + 8 * total) - 1) // 2))
    minimum = 3 if kind == "seam" else 2
    size = int(rng.integers(minimum, min(maximum, 6 if kind == "return" else 9) + 1))
    palette = rng.choice([c for c in range(10) if c != background], size=size, replace=False)
    alpha = float(rng.choice((.3, 1.5, 12.0)))
    # A decreasing staircase plus a weakly decreasing partition is always strict.
    extra = rng.multinomial(total - size * (size + 1) // 2, rng.dirichlet(np.full(size, alpha)))
    counts = np.arange(size, 0, -1) + np.sort(extra)[::-1]
    stream = np.repeat(palette, counts).astype(np.int8)
    mode = int(rng.integers(5))
    rr, cc = np.indices(shape)
    if mode == 0:
        rng.shuffle(stream)
        mosaic = stream.reshape(shape).copy()
    elif mode == 1:
        key = rr * rng.uniform(-1, 1) + cc * rng.uniform(-1, 1) + rng.normal(0, .8, shape)
        mosaic = np.empty(total, dtype=np.int8)
        mosaic[np.argsort(key.ravel())] = stream
        mosaic = mosaic.reshape(shape)
    elif mode == 2:
        patches = rng.random((3, 3))
        key = patches[rr * 3 // shape[0], cc * 3 // shape[1]] + rng.normal(0, .25, shape)
        mosaic = np.empty(total, dtype=np.int8)
        mosaic[np.argsort(key.ravel())] = stream
        mosaic = mosaic.reshape(shape)
    elif mode == 3:
        key = (rr - rng.uniform(0, shape[0])) ** 2 + (cc - rng.uniform(0, shape[1])) ** 2
        key = key + rng.normal(0, 1.5, shape)
        mosaic = np.empty(total, dtype=np.int8)
        mosaic[np.argsort(key.ravel())] = stream
        mosaic = mosaic.reshape(shape)
    else:
        mosaic = _pack(palette, counts, shape)
        for _ in range(max(2, total // 3)):
            a, b = rng.choice(total, 2, replace=False)
            ar, ac = divmod(int(a), shape[1])
            br, bc = divmod(int(b), shape[1])
            mosaic[ar, ac], mosaic[br, bc] = mosaic[br, bc], mosaic[ar, ac]
    if kind == "seam":
        view = mosaic if rng.random() < .5 else mosaic.T
        column = int(rng.integers(1, view.shape[1]))
        witness = int(rng.integers(view.shape[0]))
        view[:, column] = view[:, column - 1]
        view[witness, column] = rng.choice([c for c in palette if c != view[witness, column - 1]])
    elif kind == "return":
        choices = [a for a in (mosaic, mosaic.T) if a.shape[0] >= 3]
        view = choices[int(rng.integers(len(choices)))]
        possibilities = [(a, b) for a in range(view.shape[0]) for b in range(a + 2, view.shape[0])]
        a, b = possibilities[int(rng.integers(len(possibilities)))]
        view[b] = view[a]
    return mosaic


def _axis(rng, length, style):
    if style == "uniform":
        return np.full(length, int(rng.integers(2, min(6, 28 // length) + 1)))
    if style == "fat":
        bands = rng.integers(1, 3, size=length)
        peak = int(rng.integers(length))
        bands[peak] = int(rng.integers(8, min(14, 28 - int(bands.sum()) + int(bands[peak])) + 1))
        return bands
    bands = rng.integers(1, min(5, 28 // length) + 1, size=length)
    if np.max(bands) == 1:
        bands[int(rng.integers(length))] = 2
    return bands


def _scales(rng, mosaic, require_reversal):
    colors, counts = _rank(mosaic)
    for _ in range(96 if require_reversal else 1):
        if require_reversal and rng.random() < .65:
            # Magnify a currently less frequent color, then test the visible effect.
            color = int(rng.choice(colors[1:]))
            points = np.argwhere(mosaic == color)
            r, c = points[int(rng.integers(len(points)))]
            rows = rng.integers(1, 3, size=mosaic.shape[0])
            columns = rng.integers(1, 3, size=mosaic.shape[1])
            rows[r], columns[c] = int(rng.integers(8, 15)), int(rng.integers(8, 15))
        else:
            styles = ("uniform", "mild", "fat")
            rows = _axis(rng, mosaic.shape[0], str(rng.choice(styles)))
            columns = _axis(rng, mosaic.shape[1], str(rng.choice(styles)))
        if max(rows.sum(), columns.sum()) <= 28 and (
                not require_reversal or _reversal(mosaic, rows, columns)):
            return rows, columns
    return None


def _embed(rng, mosaic, rows, columns, background):
    crop = np.repeat(np.repeat(mosaic, rows, axis=0), columns, axis=1)
    extra_h = int(rng.integers(2, min(10, 30 - crop.shape[0]) + 1))
    extra_w = int(rng.integers(2, min(10, 30 - crop.shape[1]) + 1))
    top, left = int(rng.integers(1, extra_h)), int(rng.integers(1, extra_w))
    bottom, right = extra_h - top, extra_w - left
    if rng.random() < .3:
        side = int(rng.integers(4))
        if side == 0:
            top, bottom = 0, extra_h
        elif side == 1:
            top, bottom = extra_h, 0
        elif side == 2:
            left, right = 0, extra_w
        else:
            left, right = extra_w, 0
    canvas = np.full((crop.shape[0] + extra_h, crop.shape[1] + extra_w), background, dtype=np.int8)
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return canvas


def _new_pair(rng, shape, kind, background, seen, forbidden_dominant=None, reversal=False):
    for _ in range(1024):
        mosaic = _draw_mosaic(rng, shape, kind, background)
        if not _visible(mosaic):
            continue
        try:
            colors, counts = _rank(mosaic)
        except ValueError:
            continue
        if int(colors[0]) == forbidden_dominant:
            continue
        if kind == "seam" and (len(colors) < 3 or not _seam(mosaic) or not _direction_witness(mosaic)):
            continue
        if kind == "seam" and (np.all(np.diff(colors.astype(int)) > 0) or np.all(np.diff(colors.astype(int)) < 0)):
            continue
        if kind == "return" and not _returning_vector(mosaic):
            continue
        output = _pack(colors, counts, shape)
        if np.array_equal(mosaic, output):
            continue
        scales = _scales(rng, mosaic, reversal)
        if scales is None:
            continue
        source = _embed(rng, mosaic, *scales, background)
        key = (source.shape, source.tobytes())
        if key not in seen:
            seen.add(key)
            return {"input": source, "output": output}
    raise RuntimeError("Bounded construction could not find a compatible mosaic")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1..4 and smaller than num_train")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_background = bool(rng.integers(2))
    background0 = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen, train_shapes = set(), set()
    first_dominant = None
    for i in range(int(num_train)):
        kind = "seam" if i == 0 else "return" if i == 1 else "free"
        options = _WITNESS_SHAPES if i == 0 else tuple(
            s for s in _SHAPES if (i != 1 or s != (2, 2)) and (i != 1 or s not in train_shapes))
        shape = options[int(rng.integers(len(options)))]
        background = background0 if shared_background or i == 0 else int(
            rng.choice([c for c in range(10) if i != 1 or c != background0]))
        pair = _new_pair(rng, shape, kind, background, seen,
                         forbidden_dominant=first_dominant if i == 1 else None, reversal=i == 0)
        puzzle["train"].append(pair)
        train_shapes.add(shape)
        if i == 0:
            first_dominant = int(_rank(_recover(pair["input"])[0])[0][0])
    # New logical dimensions, often a new column height; the shape pool is shared.
    held_out_shapes = [s for s in _SHAPES if s not in train_shapes]
    for i in range(int(num_test)):
        shape = held_out_shapes[int(rng.integers(len(held_out_shapes)))]
        background = background0 if shared_background else int(rng.integers(10))
        puzzle["test"].append(_new_pair(rng, shape, "free", background, seen, reversal=i == 0))
    # Teaching examples have no fixed position; their constraints concern evidence.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    train_shapes, test_shapes = set(), set()
    test_reversal = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = _grid(pair["input"]), _grid(pair["output"])
            key = (source.shape, source.tobytes())
            assert key not in seen, "Inputs must be distinct within a puzzle"
            seen.add(key)
            mosaic, rows, columns, background, _ = _recover(source)
            colors, counts = _rank(mosaic)
            assert min(mosaic.shape) >= 2 and 2 <= len(colors) <= 9
            assert _visible(mosaic) and background not in target
            assert rows.max() >= 2 and columns.max() >= 2
            expanded = np.repeat(np.repeat(mosaic, rows, axis=0), columns, axis=1)
            mask = source != background
            assert expanded.size == int(mask.sum())
            assert np.array_equal(expanded.ravel(), source[mask])
            assert target.shape == mosaic.shape
            assert Counter(map(int, target.ravel())) == Counter(map(int, mosaic.ravel()))
            assert not np.array_equal(mosaic, target), "Collapse alone must not solve a generated pair"
            # Coordinate enumeration checks the packing invariant separately from reshape.
            expected = _traversal_pack(mosaic, "column", False, True)
            assert np.array_equal(target, expected) and np.array_equal(target, solve(source))
            reversal = _reversal(mosaic, rows, columns)
            if split == "train":
                train_shapes.add(mosaic.shape)
                training.append((mosaic, reversal, int(colors[0])))
            else:
                test_shapes.add(mosaic.shape)
                test_reversal |= reversal
    assert len(train_shapes) >= 2 and train_shapes.isdisjoint(test_shapes)
    assert test_reversal
    def teaching_witness(mosaic, reversal):
        colors, _ = _rank(mosaic)
        differences = np.diff(colors.astype(int))
        nonnumeric = not (np.all(differences > 0) or np.all(differences < 0))
        return (_seam(mosaic) and reversal and len(colors) >= 3
                and _direction_witness(mosaic) and nonnumeric)
    assert any(teaching_witness(m, rev) for m, rev, _ in training), (
        "Missing joint compression/counting demonstration")
    assert any(_returning_vector(m) for m, _, _ in training)
    assert len({lead for _, _, lead in training}) >= 2


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 fixed seeds and counts; no disk output or retained puzzle corpus."""
    signatures, shapes, profiles, histograms, structures = set(), set(), set(), set(), set()
    backgrounds, palette_sizes, dominant_colors, scale_kinds = set(), set(), set(), set()
    stats = Counter()
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                mosaic, rows, columns, background, margins = _recover(pair["input"])
                # Slow full-vector equality gives a second compression check.
                source = pair["input"]
                rr, cc = np.nonzero(source != background)
                crop = source[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                keep_r = [0] + [i for i in range(1, len(crop)) if not np.array_equal(crop[i], crop[i - 1])]
                keep_c = [0] + [j for j in range(1, crop.shape[1]) if not np.array_equal(crop[:, j], crop[:, j - 1])]
                assert np.array_equal(mosaic, crop[np.ix_(keep_r, keep_c)])
                shapes.add((split, mosaic.shape))
                backgrounds.add(background)
                colors, counts = _rank(mosaic)
                palette_sizes.add(len(colors))
                dominant_colors.add(int(colors[0]))
                profiles.add((tuple(map(int, rows)), tuple(map(int, columns))))
                histograms.add((mosaic.shape, tuple(map(int, counts))))
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in mosaic.ravel())
                structures.add((mosaic.shape, norm))
                scale_kinds.add((len(set(rows)) == 1, len(set(columns)) == 1))
                stats["edge"] += 0 in margins
                stats["one_pixel"] += 1 in rows or 1 in columns
                stats["large_band"] += max(rows.max(), columns.max()) >= 8
                stats[split + "_reverse"] += _reversal(mosaic, rows, columns)
                stats["return"] += _returning_vector(mosaic)
                stats["seam"] += _seam(mosaic)
                stats["aligned"] += bool(np.any(np.cumsum(counts)[:-1] % mosaic.shape[0] == 0))
                stats["partial"] += bool(np.any(np.cumsum(counts)[:-1] % mosaic.shape[0] != 0))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(profiles) >= 100
    assert len(histograms) >= 100 and len(structures) >= 150
    assert len(backgrounds) == len(dominant_colors) == 10 and len(palette_sizes) >= 6
    assert len(scale_kinds) == 4 and all(stats[k] > 0 for k in (
        "edge", "one_pixel", "large_band", "train_reverse", "test_reverse", "return", "seam", "aligned", "partial"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(20261023, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    assert _signature(generate_puzzle(12, 7, 3)) == _signature(generate_puzzle(12, 7, 3))
    assert _signature(generate_puzzle(12)) != _signature(generate_puzzle(13))
    # Explicit semantic fixture: separated identical rows survive; packing is final.
    logical = np.array([[1, 1, 2], [1, 3, 3], [1, 1, 2]], dtype=np.int8)
    expanded = np.repeat(np.repeat(logical, [1, 8, 2], axis=0), [2, 1, 9], axis=1)
    source = np.pad(expanded, ((1, 2), (2, 1)), constant_values=0)
    # This fixture has tied tails and must fail closed, rather than invent a tie rule.
    try:
        solve(source)
    except ValueError:
        pass
    else:
        raise AssertionError("Tied frequencies accepted")
    logical[1, 2] = 1
    source = np.pad(np.repeat(np.repeat(logical, [1, 8, 2], axis=0), [2, 1, 9], axis=1),
                    ((1, 2), (2, 1)), constant_values=0)
    expected = np.array([[1, 1, 3], [1, 1, 2], [1, 1, 2]], dtype=np.int8)
    assert np.array_equal(solve(source), expected)
    logical = np.array([[1, 1, 2], [1, 3, 3], [1, 2, 2]], dtype=np.int8)
    source = np.pad(np.repeat(np.repeat(logical, [1, 8, 2], axis=0), [2, 1, 9], axis=1),
                    ((1, 2), (2, 1)), constant_values=0)
    expected = np.array([[1, 2, 3], [1, 2, 3], [1, 1, 2]], dtype=np.int8)
    assert np.array_equal(_pixel_counts(logical, np.array([1, 8, 2]), np.array([2, 1, 9])),
                          np.array([0, 23, 29, 80, 0, 0, 0, 0, 0, 0]))
    assert np.array_equal(solve(source), expected)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
