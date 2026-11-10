"""Mark negative space at pixel scale, then collapse equal adjacent vectors.

Family conventions: background 0, new highlight 3, every other color foreground.
No files, examples, random state, or generation metadata are read by solve().
"""

import hashlib

import numpy as np


BACKGROUND = 0
HIGHLIGHT = 3
FOREGROUND_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=np.int8)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((grid >= 0) & (grid <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return grid.astype(np.int8, copy=True)


def _empty_neighborhood(grid):
    """All original in-grid cells in the clipped 3x3 window must be zero."""
    height, width = grid.shape
    obstacles = np.pad(grid != BACKGROUND, 1, constant_values=False)
    near = np.zeros(grid.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= obstacles[dy:dy + height, dx:dx + width]
    return ~near


def _mark(grid):
    marked = grid.copy()
    marked[_empty_neighborhood(grid)] = HIGHLIGHT
    return marked


def _collapse(grid):
    """Use complete vectors of the marked grid; retain separated recurrences."""
    rows = np.r_[0, np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1)) + 1]
    columns = np.r_[0, np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0)) + 1]
    result = grid[np.ix_(rows, columns)].copy()
    heights = np.diff(np.r_[rows, grid.shape[0]])
    widths = np.diff(np.r_[columns, grid.shape[1]])
    return result, rows, columns, heights, widths


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Evaluate the fixed family rule using only the given pixels.

    The whole input is the mosaic: no crop, frame, legend, or exterior is removed.
    Even unusual valid members (one row, no foreground, no eligible cells) work.
    """
    grid = _grid(input_grid)
    if np.any(grid == HIGHLIGHT):
        raise ValueError("color 3 is reserved for newly introduced highlights")
    return _collapse(_mark(grid))[0]


def _components(mask):
    """Sizes of four-connected components, used only for diagnostics."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for row, column in np.argwhere(mask):
        row, column = int(row), int(column)
        if seen[row, column]:
            continue
        todo = [(row, column)]
        seen[row, column] = True
        size = 0
        while todo:
            y, x = todo.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < height and 0 <= xx < width
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        sizes.append(size)
    return sizes


def _separated_recurrence(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return any(np.array_equal(vectors[i], vectors[j])
               for i in range(len(vectors)) for j in range(i + 2, len(vectors)))


def _features(grid):
    mask = _empty_neighborhood(grid)
    marked = _mark(grid)
    output, rows, columns, heights, widths = _collapse(marked)
    foreground = grid != BACKGROUND
    padded = np.pad(foreground, 1, constant_values=False)
    h, w = grid.shape
    orthogonal = (padded[1:1 + h, 1:1 + w] | padded[:h, 1:1 + w]
                  | padded[2:2 + h, 1:1 + w] | padded[1:1 + h, :w]
                  | padded[1:1 + h, 2:2 + w])
    diagonal_only = (grid == BACKGROUND) & ~orthogonal & ~mask
    edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    broken_rows = np.all(grid[1:] == grid[:-1], axis=1) & np.any(marked[1:] != marked[:-1], axis=1)
    broken_columns = np.all(grid[:, 1:] == grid[:, :-1], axis=0) & np.any(marked[:, 1:] != marked[:, :-1], axis=0)
    return {"mask": mask, "marked": marked, "output": output,
            "rows": rows, "columns": columns, "heights": heights, "widths": widths,
            "component_sizes": _components(mask), "edge": edge,
            "diagonal": bool(diagonal_only.any()),
            "breaks_rows": bool(broken_rows.any()), "breaks_columns": bool(broken_columns.any()),
            "recurs_rows": _separated_recurrence(output, 0),
            "recurs_columns": _separated_recurrence(output, 1),
            "order_matters": not np.array_equal(output, _mark(_collapse(grid)[0]))}


def _foreground_mosaic(rng, height, width):
    count = int(rng.choice([1, 2, 3, 4, 5, 6], p=[.19, .22, .21, .16, .13, .09]))
    palette = rng.choice(FOREGROUND_COLORS, count, replace=False)
    style = int(rng.integers(4))
    if style == 0:
        cells = rng.choice(palette, (height, width))
    elif style == 1:
        cells = np.repeat(rng.choice(palette, (height, 1)), width, axis=1)
    elif style == 2:
        cells = np.repeat(rng.choice(palette, (1, width)), height, axis=0)
    else:
        cells = rng.choice(palette, (height, width))
        # Coherent patches, with global boundaries sometimes hidden locally.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            cells[r:min(height, r + 2), c:min(width, c + 2)] = int(rng.choice(palette))
    return cells.astype(np.int8)


def _openings(rng, height, width):
    """Several compatible negative-space geometries, not separate subpuzzles."""
    opening = np.zeros((height, width), dtype=bool)
    mode = int(rng.integers(5))
    if mode == 0:  # Multiple rectangular cutouts, sometimes meeting the edge.
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            opening[r:min(height, r + dh), c:min(width, c + dw)] = True
    elif mode == 1:  # A bent or branched corridor.
        r, c = int(rng.integers(height)), int(rng.integers(width))
        for _ in range(int(rng.integers(5, 13))):
            opening[r:min(height, r + int(rng.integers(1, 3))),
                    c:min(width, c + int(rng.integers(1, 3)))] = True
            dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
            r, c = int(np.clip(r + dy, 0, height - 1)), int(np.clip(c + dx, 0, width - 1))
    elif mode == 2:  # Crossing bands with independently varying thickness.
        r, c = int(rng.integers(height)), int(rng.integers(width))
        opening[r:min(height, r + int(rng.integers(1, 3))), :] = True
        opening[:, c:min(width, c + int(rng.integers(1, 3)))] = True
    elif mode == 3:  # Broad clear area containing a foreground island.
        r, c = int(rng.integers(0, height - 2)), int(rng.integers(0, width - 2))
        bottom = int(rng.integers(r + 3, height + 1))
        right = int(rng.integers(c + 3, width + 1))
        opening[r:bottom, c:right] = True
        opening[int(rng.integers(r + 1, bottom - 1)), int(rng.integers(c + 1, right - 1))] = False
    else:  # Stepped channel; different rows can have different lateral spans.
        left = int(rng.integers(width - 1))
        for r in range(height):
            left = int(np.clip(left + int(rng.integers(-1, 2)), 0, width - 2))
            opening[r, left:min(width, left + int(rng.integers(1, 4)))] = True
    return opening


def _bands(rng, count, fixed):
    if rng.random() < .20:
        values = np.full(count, int(rng.integers(2, 5)), dtype=int)
    else:
        values = rng.integers(1, 6, count)
    for index, value in fixed.items():
        values[index] = value
    # At most eight bands and two fixed bands: this always has ample room.
    while values.sum() > 30:
        candidates = [i for i in range(count) if i not in fixed and values[i] > 1]
        if not candidates:
            raise RuntimeError("incompatible expansion profile")
        values[int(rng.choice(candidates))] -= 1
    return values


def _make_pair(rng, kind, is_test):
    low, high = (6, 9) if is_test else (5, 8)
    height, width = int(rng.integers(low, high)), int(rng.integers(low, high))
    source = _foreground_mosaic(rng, height, width)
    opening = _openings(rng, height, width)
    # Random anchor within the lower-right region; a later dihedral transform
    # varies its quadrant. Its foreground halo isolates the small cavity.
    r, c = int(rng.integers(3, height - 1)), int(rng.integers(3, width - 1))
    opening[r - 1:r + 2, c - 1:c + 2] = False
    opening[r, c] = True
    row_fixed, column_fixed = {}, {}
    if kind == "pocket":
        row_fixed[r] = 3
        column_fixed[c] = 3
        # The closed perimeter supplies separated recurring vectors, which
        # must not be globally deduplicated. There can still be other holes.
        opening[[0, -1], :] = False
        opening[:, [0, -1]] = False
        source[-1] = source[0]
        source[:, -1] = source[:, 0]
    else:
        # An open L at a corner provides a clipped-edge highlight AND a
        # background pixel with foreground only diagonally in its 3x3 window.
        opening[0, :int(rng.integers(2, min(5, width) + 1))] = True
        opening[:int(rng.integers(2, min(5, height) + 1)), 0] = True
        opening[1, 1] = False
        row_fixed[0] = int(rng.integers(2, 6))
        column_fixed[0] = int(rng.integers(2, 6))
        if kind == "edge":
            # An isolated one- or two-pixel-wide slit has no clear core.
            row_fixed[r] = int(rng.integers(3, 6))
            column_fixed[c] = int(rng.integers(1, 3))
        else:
            # Tests recombine the already demonstrated bounded pocket and L.
            row_fixed[r] = int(rng.integers(3, 7))
            column_fixed[c] = int(rng.integers(3, 7))
    source[opening] = BACKGROUND
    rows = _bands(rng, height, row_fixed)
    columns = _bands(rng, width, column_fixed)
    expanded = np.repeat(np.repeat(source, rows, axis=0), columns, axis=1)
    expanded = np.rot90(expanded, int(rng.integers(4)))
    if rng.random() < .5:
        expanded = expanded[:, ::-1]
    expanded = expanded.copy()
    return {"input": expanded, "output": _collapse(_mark(expanded))[0]}


def _signature(grid):
    return tuple(grid.shape), grid.astype(np.uint8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return a deterministic puzzle; no sampled conventions are hidden."""
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    train_kinds = ["pocket", "edge"]
    rng.shuffle(train_kinds)
    train_kinds.extend(str(rng.choice(["pocket", "edge", "mixed"])) for _ in range(ntrain - 2))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, kinds in (("train", train_kinds), ("test", ["mixed"] * ntest)):
        for kind in kinds:
            for _ in range(64):
                pair = _make_pair(rng, kind, split == "test")
                info = _features(pair["input"])
                key = _signature(pair["input"])
                if (key in used or not info["mask"].any() or not info["order_matters"]
                        or not (info["breaks_rows"] and info["breaks_columns"])
                        or not all(a < b for a, b in zip(pair["output"].shape, pair["input"].shape))):
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not construct a distinct diagnostic pair in 64 attempts")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check labels, local eligibility, lossless run reduction, and teaching cases.

    These checks are generator-owned, not an independent semantic review.
    No generation metadata is used: all diagnostics come from the supplied grids.
    """
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    training = {key: False for key in ("singleton", "edge", "diagonal", "row_recurrence", "column_recurrence")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert HIGHLIGHT not in inp and BACKGROUND in inp
            assert np.any((inp != BACKGROUND) & (inp != HIGHLIGHT))
            key = _signature(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _features(inp)
            marked = info["marked"]
            # A scalar check expresses the all-cells predicate separately from
            # the vectorized padded implementation; diagonal neighbors count.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    window = inp[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    assert bool(info["mask"][r, c]) == bool(np.all(window == BACKGROUND))
            assert np.array_equal(out, info["output"])
            assert np.array_equal(out, solve(inp))
            assert all(a < b for a, b in zip(out.shape, inp.shape)), "both axes must really compress"
            assert set(np.unique(out)) == set(np.unique(inp)) | {HIGHLIGHT}
            assert np.all(marked[inp != BACKGROUND] == inp[inp != BACKGROUND])
            assert np.any((marked == BACKGROUND) & (inp == BACKGROUND)), "missing background fringe"
            assert info["breaks_rows"] and info["breaks_columns"]
            assert info["order_matters"], "compress-first shortcut must give a different result"
            # Re-expansion proves every marked and unmarked pixel state survives
            # as the value of its run intersection; multiplicities are reduced.
            reconstruction = np.repeat(np.repeat(out, info["heights"], axis=0), info["widths"], axis=1)
            assert np.array_equal(reconstruction, marked)
            assert not np.any(np.all(out[1:] == out[:-1], axis=1))
            assert not np.any(np.all(out[:, 1:] == out[:, :-1], axis=0))
            # Explicit omission controls for both inherited operations.
            assert not np.array_equal(_collapse(inp)[0], out), "omitting B did not change the answer"
            assert not np.array_equal(marked, out), "omitting A did not change the answer"
            if split == "train":
                training["singleton"] |= 1 in info["component_sizes"]
                training["edge"] |= info["edge"]
                training["diagonal"] |= info["diagonal"]
                training["row_recurrence"] |= info["recurs_rows"]
                training["column_recurrence"] |= info["recurs_columns"]
            else:
                assert info["edge"] and len(info["component_sizes"]) >= 2
    assert all(training.values()), training


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                grid = np.asarray(pair[name], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds and legal counts, retaining hashes and aggregate sets only."""
    # Hand-specified witnesses for marking before compression and recurrence.
    enclosed = np.ones((7, 7), dtype=np.int8)
    enclosed[2:5, 2:5] = BACKGROUND
    expected = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 3, 0, 1],
                         [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=np.int8)
    assert np.array_equal(solve(enclosed), expected)
    assert not np.array_equal(_mark(_collapse(enclosed)[0]), expected)
    assert np.array_equal(solve(np.array([[0, 0, 1, 1, 0, 0]])), [[3, 0, 1, 0, 3]])
    assert np.array_equal(solve(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 2]])),
                          [[3, 3, 3], [3, 0, 0], [3, 0, 2]])
    assert np.array_equal(solve(np.zeros((1, 1), dtype=int)), [[3]])
    assert np.array_equal(solve(np.array([[2, 2], [2, 2]])), [[2]])
    assert np.array_equal(solve(np.array([[1, 2, 1], [1, 2, 1]])), [[1, 2, 1]])
    signatures, shapes, outputs, structures, palettes, profiles = set(), set(), set(), set(), set(), set()
    stats = {key: 0 for key in ("pairs", "singleton", "edge", "diagonal", "multiple_cores",
                               "row_recurrence", "column_recurrence", "order_matters", "full_axis")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _features(inp)
                shapes.add(inp.shape)
                outputs.add(out.shape)
                roles = {BACKGROUND: 0, HIGHLIGHT: 1}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape) + normalized).hexdigest())
                palettes.add(tuple(map(int, np.unique(inp))))
                profiles.add((tuple(map(int, info["heights"])), tuple(map(int, info["widths"]))))
                stats["pairs"] += 1
                stats["singleton"] += 1 in info["component_sizes"]
                stats["edge"] += info["edge"]
                stats["diagonal"] += info["diagonal"]
                stats["multiple_cores"] += len(info["component_sizes"]) >= 2
                stats["row_recurrence"] += info["recurs_rows"]
                stats["column_recurrence"] += info["recurs_columns"]
                stats["order_matters"] += info["order_matters"]
                stats["full_axis"] += 30 in inp.shape
    assert len(signatures) == 42 and len(shapes) >= 30 and len(outputs) >= 25
    assert len(structures) >= 150 and len(palettes) >= 20 and len(profiles) >= 100
    assert all(stats[key] > 0 for key in stats)
    # No output or sampling depends on prior calls or a mutated input.
    assert _puzzle_digest(generate_puzzle(12345)) == _puzzle_digest(generate_puzzle(12345))
    saved = enclosed.copy()
    solve(enclosed)
    assert np.array_equal(enclosed, saved)
    count_calls = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(100000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            count_calls += 1
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": stats.pop("pairs"), "legal_count_combinations": count_calls,
            "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "output_shapes": len(outputs), "color_normalized_inputs": len(structures),
            "palettes": len(palettes), "marked_run_profiles": len(profiles), "witnesses": stats,
            "independent_semantic_review": False}
