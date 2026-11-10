"""Generate divider-lattice ARC puzzles with a left-to-right collapse rule."""

import numpy as np


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, seed, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _transforms(grid):
    reflected = grid[:, ::-1]
    return [np.rot90(grid, k) for k in range(4)] + [
        np.rot90(reflected, k) for k in range(4)
    ]


def _distinct_reflections(grid):
    return len({_key(item) for item in _transforms(grid)}) == 8


def _expand(macro, block, divider):
    rows, cols = macro.shape
    result = np.full((rows * (block + 1) - 1, cols * (block + 1) - 1),
                     divider, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            result[r * (block + 1):r * (block + 1) + block,
                   c * (block + 1):c * (block + 1) + block] = macro[r, c]
    return result


def _decode(grid):
    """Find the unique full-line divider and its uniform square compartments."""
    height, width = grid.shape
    found = []
    for raw in np.unique(grid):
        color = int(raw)
        line_rows = np.flatnonzero(np.all(grid == color, axis=1))
        line_cols = np.flatnonzero(np.all(grid == color, axis=0))
        if not len(line_rows) or not len(line_cols):
            continue
        rs = np.diff(np.r_[-1, line_rows, height]) - 1
        cs = np.diff(np.r_[-1, line_cols, width]) - 1
        if (np.any(rs < 2) or np.any(cs < 2) or
                not np.all(rs == rs[0]) or not np.all(cs == cs[0]) or
                rs[0] != cs[0]):
            continue
        block = int(rs[0])
        macro = np.empty((len(rs), len(cs)), dtype=np.int8)
        okay = True
        for r in range(len(rs)):
            for c in range(len(cs)):
                patch = grid[r * (block + 1):r * (block + 1) + block,
                             c * (block + 1):c * (block + 1) + block]
                value = int(patch[0, 0])
                if value == color or not np.all(patch == value):
                    okay = False
                    break
                macro[r, c] = value
            if not okay:
                break
        if okay and np.array_equal(grid, _expand(macro, block, color)):
            found.append((macro, block, color))
    assert len(found) == 1, "expected one unambiguous divider lattice"
    return found[0]


_SOURCE = (
    (np.array([[0, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0],
               [0, 0, 0, 3]], dtype=np.int8), 4, 8),
    (np.array([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0],
               [3, 3, 3, 0]], dtype=np.int8), 4, 2),
    (np.array([[0, 8, 0], [8, 8, 0], [0, 0, 4]], dtype=np.int8), 3, 2),
    (np.array([[1, 3, 1, 1], [0, 3, 0, 0], [2, 3, 0, 2],
               [0, 3, 0, 0]], dtype=np.int8), 5, 8),
)


def _source_copy(input_grid, output_grid):
    return any(np.array_equal(input_grid, _expand(macro, block, divider)) and
               np.array_equal(output_grid, macro[:, ::-1])
               for macro, block, divider in _SOURCE)


def _shapes():
    return [(r, c) for r in range(2, 7) for c in range(2, 7)]


def _scales(shape):
    return [b for b in range(2, 7) if max(shape) * (b + 1) - 1 <= 30]


def _background(rng, divider, mandatory):
    options = [c for c in range(10) if c != divider]
    if 0 in options and rng.random() < 0.72:
        return 0
    if mandatory in options and rng.random() < 0.12:
        return mandatory
    return int(rng.choice(options))


def _density(rng, kind):
    spans = {"sparse": (0.15, 0.39), "medium": (0.39, 0.61),
             "dense": (0.61, 0.86)}
    lo, hi = spans[kind]
    return float(rng.uniform(lo, hi))


def _occupancy(rng, shape, fraction, motif):
    rows, cols = shape
    area = rows * cols
    desired = min(area - 1, max(1, int(round(fraction * area))))
    mask = np.zeros(shape, dtype=bool)
    fixed = np.zeros(shape, dtype=bool)
    if motif == "mirror":
        for r in range(rows):
            for c in range((cols + 1) // 2):
                if rng.random() < fraction:
                    mask[r, c] = mask[r, cols - 1 - c] = True
        if not np.any(mask):
            r = int(rng.integers(rows))
            mask[r, 0] = mask[r, cols - 1] = True
        if np.all(mask):
            mask[0, 0] = mask[0, cols - 1] = False
        return mask, fixed
    if motif == "column":
        c = int(rng.integers(cols))
        mask[:, c] = fixed[:, c] = True
    elif motif == "row":
        r = int(rng.integers(rows))
        mask[r, :] = fixed[r, :] = True
    elif motif == "diagonal":
        offset = int(rng.integers(max(1, cols - rows + 1)))
        for r in range(min(rows, cols)):
            c = min(cols - 1, r + offset)
            mask[r, c] = fixed[r, c] = True
        if rows >= 3 and rng.random() < 0.5:
            mask[-1, :] = fixed[-1, :] = True
    elif motif == "patch":
        r, c = int(rng.integers(rows)), int(rng.integers(cols))
        mask[r, c] = True
        while int(mask.sum()) < desired:
            frontier = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                        frontier.append((nr, nc))
            r, c = frontier[int(rng.integers(len(frontier)))]
            mask[r, c] = True
        return mask, fixed
    needed = max(0, desired - int(mask.sum()))
    if needed:
        choices = np.flatnonzero(~mask)
        for index in rng.choice(choices, size=min(needed, len(choices)), replace=False):
            mask.flat[int(index)] = True
    return mask, fixed


def _macro(rng, shape, divider, mandatory, density_kind, motif):
    base = _background(rng, divider, mandatory)
    mask, fixed = _occupancy(rng, shape, _density(rng, density_kind), motif)
    if np.all(mask):
        spare = np.flatnonzero(~fixed)
        if len(spare):
            mask.flat[int(spare[0])] = False
    available = [c for c in range(10) if c not in (divider, base)]
    if mandatory is not None and mandatory != base:
        primary = mandatory
    else:
        primary = int(rng.choice(available))
    if motif == "mirror":
        macro = np.full(shape, base, dtype=np.int8)
        macro[mask] = primary
        return macro
    free = np.flatnonzero(mask & ~fixed)
    max_extra = min(4, len(free), len(available) - 1)
    extra_count = 0 if max_extra == 0 or rng.random() < 0.38 else int(
        rng.integers(1, max_extra + 1))
    other = [c for c in available if c != primary]
    extras = [int(v) for v in rng.choice(other, size=extra_count, replace=False)]
    macro = np.full(shape, base, dtype=np.int8)
    macro[mask] = primary
    if extra_count:
        positions = rng.choice(free, size=extra_count, replace=False)
        for pos, color in zip(positions, extras):
            macro.flat[int(pos)] = color
        # Extra color patches, rather than isolated mandatory singleton colors.
        for pos in free:
            if rng.random() < 0.18:
                macro.flat[int(pos)] = int(rng.choice(extras))
    if mandatory is not None and mandatory not in macro:
        candidates = np.flatnonzero(mask & ~fixed)
        if not len(candidates):
            candidates = np.flatnonzero(mask)
        macro.flat[int(candidates[0])] = mandatory
    return macro


def _pair(rng, shape, block, divider, mandatory, density, motif,
          require_asymmetry=False):
    for _ in range(160):
        macro = _macro(rng, shape, divider, mandatory, density, motif)
        if require_asymmetry and not _distinct_reflections(macro):
            continue
        input_grid = _expand(macro, block, divider)
        output_grid = macro[:, ::-1].copy()
        if not _source_copy(input_grid, output_grid):
            return {"input": input_grid, "output": output_grid}
    # A two-color irregular pattern provides a bounded fallback for the
    # single training witness that must distinguish all eight orientations.
    base = 0 if divider != 0 and mandatory != 0 else next(
        c for c in range(10) if c not in (divider, mandatory))
    accent = mandatory if mandatory is not None and mandatory != base else next(
        c for c in range(10) if c not in (divider, base))
    for bits in range(1, 1 << min(shape[0] * shape[1], 12)):
        macro = np.full(shape, base, dtype=np.int8)
        for index in range(min(macro.size, 12)):
            if bits & (1 << index):
                macro.flat[index] = accent
        if _distinct_reflections(macro):
            input_grid = _expand(macro, block, divider)
            output_grid = macro[:, ::-1].copy()
            if not _source_copy(input_grid, output_grid):
                return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct an asymmetric witness")


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, seed, num_train, num_test)
    # Reflection is the sole puzzle-wide convention. Color identities and
    # compartment geometry are visible, pair-local properties.
    mode = _choice(rng, ("source_like", "same", "smaller", "mixed"))
    anchor_shape = (4, 4) if mode == "source_like" else _choice(
        rng, ((3, 3), (4, 4), (3, 4), (4, 3), (3, 5), (5, 3)))
    if mode == "source_like":
        anchor_scales = (3, 4)
    elif mode == "smaller":
        anchor_scales = tuple(int(v) for v in rng.choice(
            [3, 4] if max(anchor_shape) == 5 else [3, 4, 5],
            size=2, replace=False))
    else:
        anchor_scales = tuple(int(v) for v in rng.choice(
            _scales(anchor_shape), size=2, replace=False))
    d1, d2 = (int(v) for v in rng.choice(10, size=2, replace=False))
    specs = []
    motifs = ("random", "column", "row", "diagonal", "patch", "mirror")
    specs.append((anchor_shape, anchor_scales[0], d1, d2,
                  "sparse" if mode == "source_like" else _choice(rng, ("sparse", "medium")),
                  "random", True))
    second_density = "sparse" if mode == "source_like" else "dense"
    specs.append((anchor_shape, anchor_scales[1], d2, d1,
                  second_density, _choice(rng, motifs), False))
    for _ in range(num_train - 2):
        shape = _choice(rng, _shapes())
        scales = [b for b in _scales(shape) if mode != "source_like" or b <= 4]
        block = _choice(rng, scales)
        divider = int(rng.integers(10))
        density = _choice(rng, ("sparse", "medium", "dense"))
        specs.append((shape, block, divider, None, density,
                      _choice(rng, motifs), False))
    train_configs = {(shape, block) for shape, block, *_ in specs}
    test_specs = []
    for i in range(num_test):
        possible = [(shape, b) for shape in _shapes() for b in _scales(shape)
                    if (shape, b) not in train_configs]
        if mode == "source_like":
            preferred = [(shape, b) for shape, b in possible
                         if shape == (4, 4) and b in (5, 6)]
        elif mode == "same":
            preferred = [(shape, b) for shape, b in possible
                         if b in anchor_scales and shape != anchor_shape]
        elif mode == "smaller":
            preferred = [(shape, b) for shape, b in possible if b == 2]
        else:
            preferred = possible
        shape, block = _choice(rng, preferred or possible)
        divider = int(_choice(rng, (d1, d2))) if rng.random() < 0.7 else int(
            rng.integers(10))
        density = "dense" if mode == "source_like" else _choice(
            rng, ("sparse", "medium", "dense"))
        test_specs.append((shape, block, divider, None, density,
                           _choice(rng, motifs), False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, plan in (("train", specs), ("test", test_specs)):
        for shape, block, divider, mandatory, density, motif, asym in plan:
            for _ in range(80):
                pair = _pair(rng, shape, block, divider, mandatory,
                             density, motif, asym)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_key, output_key = _key(pair["input"]), _key(pair["output"])
            assert input_key not in inputs and (input_key, output_key) not in pairs
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            macro, block, divider = _decode(pair["input"])
            assert 2 <= macro.shape[0] <= 6 and 2 <= macro.shape[1] <= 6
            assert 2 <= block <= 6
            assert np.array_equal(pair["output"], macro[:, ::-1])
            assert divider not in macro and len(np.unique(macro)) >= 2
            assert not _source_copy(pair["input"], pair["output"])
            records[split].append((macro, block, divider))
    train = records["train"]
    # The same macro dimensions at two physical scales demonstrate that
    # block scale is not the output size.
    assert any(a.shape == b.shape and sa != sb for i, (a, sa, _) in enumerate(train)
               for b, sb, _ in train[i + 1:])
    assert train[0][2] != train[1][2]
    assert train[1][2] in train[0][0] and train[0][2] in train[1][0]
    # A complete orbit in training excludes identity, vertical reflection,
    # rotations and transposes as competing output conventions.
    assert any(_distinct_reflections(macro) for macro, _, _ in train)
    train_configs = {(macro.shape, block) for macro, block, _ in train}
    assert all((macro.shape, block) not in train_configs
               for macro, block, _ in records["test"])


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[name], y[name])
        for x, y in zip(a[split], b[split]) for name in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_key(pair["input"]) for split in ("train", "test")
                      for pair in p[split]) for p in puzzles}) == 42
    shapes, scales, dividers, palettes, motifs = set(), set(), set(), set(), set()
    sparse_train = dense_train = sparse_test = dense_test = 0
    two_color = solid_row = solid_col = symmetric = zero_divider = 0
    different_density = rectangular = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        densities = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                macro, block, divider = _decode(pair["input"])
                _, counts = np.unique(macro, return_counts=True)
                occupancy = 1 - max(counts) / macro.size
                densities.append(occupancy)
                shapes.add(macro.shape)
                scales.add(block)
                dividers.add(divider)
                palettes.add(tuple(np.unique(macro)))
                motifs.add(_key(macro))
                rectangular += macro.shape[0] != macro.shape[1]
                two_color += len(np.unique(macro)) == 2
                solid_row += any(np.all(row == row[0]) for row in macro)
                solid_col += any(np.all(col == col[0]) for col in macro.T)
                symmetric += np.array_equal(macro, macro[:, ::-1])
                zero_divider += divider == 0
                if split == "train":
                    sparse_train += occupancy <= 0.4
                    dense_train += occupancy >= 0.55
                else:
                    sparse_test += occupancy <= 0.4
                    dense_test += occupancy >= 0.55
        different_density += max(densities) - min(densities) >= 0.15
    assert len(shapes) >= 12 and len(scales) == 5 and len(dividers) == 10
    assert len(palettes) >= 30 and len(motifs) >= 0.8 * sum(
        len(p["train"]) + len(p["test"]) for p in puzzles)
    assert min(sparse_train, dense_train, sparse_test, dense_test) >= 5
    assert min(two_color, solid_row, solid_col, symmetric, zero_divider) >= 3
    assert rectangular >= 20 and different_density >= 30
    for seed, counts in ((1001, (2, 1)), (1002, (10, 1)), (1003, (10, 4)),
                         (2001, (None, 1)), (2002, (None, 4)),
                         (2003, (2, None)), (2004, (10, None)),
                         (3001, (7, 3))):
        first = generate_puzzle(seed, *counts)
        second = generate_puzzle(seed, *counts)
        validate_generated_puzzle(first)
        assert _equal_puzzles(first, second)
        if counts[0] is not None:
            assert len(first["train"]) == counts[0]
        if counts[1] is not None:
            assert len(first["test"]) == counts[1]
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
