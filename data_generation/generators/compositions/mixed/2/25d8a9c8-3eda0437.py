"""Whole-row equality creates the field for unique-largest-rectangle selection.

Fixed output roles: uniform row = 5, mixed row = 0, selected rectangle = 6.
The required proposal necessarily makes the rectangle a full-width row band.
No parent code, external data, persistent state, or file I/O is used here.
"""

from collections import Counter
import hashlib
import numpy as np


POSITIVE, BACKGROUND, HIGHLIGHT = 5, 0, 6


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _grid(value):
    array = np.asarray(value)
    if array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape):
        raise ValueError("grids must have two dimensions in 1..30")
    if array.dtype.kind not in "iu" or not np.all((0 <= array) & (array <= 9)):
        raise ValueError("grids must contain integer colors 0..9")
    return array


def _runs(bits):
    """Half-open intervals of true values, including boundary intervals."""
    result, start = [], None
    for i, bit in enumerate(bits):
        if bit and start is None:
            start = i
        elif not bit and start is not None:
            result.append((start, i))
            start = None
    if start is not None:
        result.append((start, len(bits)))
    return result


def _largest_empty_rectangle(empty):
    """Area search on a 2D boolean field; no row-band assumption here."""
    height, width = empty.shape
    best_area, winners = 0, []
    for top in range(height - 1):
        common = np.ones(width, dtype=bool)
        for bottom in range(top, height):
            common &= empty[bottom]
            rh = bottom - top + 1
            if rh < 2:
                continue
            for left, right in _runs(common):
                rw = right - left
                if rw < 2:
                    continue
                area = rh * rw
                rectangle = (top, bottom + 1, left, right)
                if area > best_area:
                    best_area, winners = area, [rectangle]
                elif area == best_area:
                    winners.append(rectangle)
    if len(winners) != 1:
        raise ValueError("input must yield one unique largest rectangle of size at least 2x2")
    return winners[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the two stages from the input alone; reject missing/tied winners."""
    grid = _grid(input_grid)
    uniform = np.all(grid == grid[:, :1], axis=1)
    field = np.repeat(np.where(uniform, POSITIVE, BACKGROUND)[:, None],
                      grid.shape[1], axis=1).astype(np.int8)
    top, bottom, left, right = _largest_empty_rectangle(field == BACKGROUND)
    field[top:bottom, left:right] = HIGHLIGHT
    return field


def _output_for_mask(uniform, width):
    """Construction/validation path using the full-width-band theorem."""
    bands = _runs(~np.asarray(uniform, dtype=bool))
    eligible = [(end - start, start, end) for start, end in bands if end - start >= 2]
    if width < 2 or not eligible:
        raise ValueError("no eligible mixed-row band")
    maximum = max(length for length, _, _ in eligible)
    winners = [(start, end) for length, start, end in eligible if length == maximum]
    if len(winners) != 1:
        raise ValueError("largest mixed-row band is tied")
    start, end = winners[0]
    output = np.repeat(np.where(uniform, POSITIVE, BACKGROUND)[:, None],
                       width, axis=1).astype(np.int8)
    output[start:end] = HIGHLIGHT
    return output


def _mask_from_lengths(lengths, gaps, before, after):
    values = [True] * before
    for index, length in enumerate(lengths):
        if index:
            values.extend([True] * gaps[index - 1])
        values.extend([False] * length)
    values.extend([True] * after)
    return np.asarray(values, dtype=bool)


def _plan(rng, profile, forced_boundary=None):
    """Construct a unique winner directly; never reject random entire grids."""
    if profile == "witness":
        winner = int(rng.integers(3, 6))
        lengths = [int(rng.integers(2, winner)), winner,
                   int(rng.integers(2, winner))]
        # A singleton is ineligible and remains background. Keep eligible
        # competitors on both sides of the winner after insertion.
        lengths.insert(int(rng.integers(0, 4)), 1)
        gaps = [int(rng.integers(1, 4)) for _ in range(3)]
        before, after = (int(x) for x in rng.integers(0, 3, size=2))
    else:
        if profile == "test":
            count = int(rng.integers(5, 8))
            winner = int(rng.integers(3, 7))
        elif profile == "boundary":
            count = int(rng.integers(2, 4))
            winner = int(rng.integers(2, 7))
        elif profile == "simple":
            count, winner = 1, int(rng.integers(2, 16))
        else:
            count = int(rng.integers(2, 5))
            winner = int(rng.integers(2, 7))
        position = int(rng.integers(count))
        if forced_boundary is not None:
            position = 0 if forced_boundary == "top" else count - 1
        lengths = [1] * count
        lengths[position] = winner
        if count > 1:
            other = [i for i in range(count) if i != position]
            # A close runner-up in tests; smaller genuine competitors where
            # possible in training. Height-two winners only have singletons.
            rival = other[int(rng.integers(len(other)))]
            lengths[rival] = winner - 1
        gaps = [1] * (count - 1)
        before = 0 if forced_boundary == "top" else int(rng.integers(0, 3))
        after = 0 if forced_boundary == "bottom" else int(rng.integers(0, 3))
        # Spend a bounded height budget on independent run and gap variation.
        # This also permits 30-row canvases without large retry loops.
        total = sum(lengths) + sum(gaps) + before + after
        budget = int(rng.integers(0, 31 - total))
        for _ in range(budget):
            options = [("run", i) for i in range(count)
                       if i != position and lengths[i] < winner - 1]
            options += [("gap", i) for i in range(len(gaps)) if gaps[i] < 4]
            if forced_boundary != "top" and before < 4:
                options.append(("before", 0))
            if forced_boundary != "bottom" and after < 4:
                options.append(("after", 0))
            if not options:
                break
            kind, index = options[int(rng.integers(len(options)))]
            if kind == "run":
                lengths[index] += 1
            elif kind == "gap":
                gaps[index] += 1
            elif kind == "before":
                before += 1
            else:
                after += 1
    return _mask_from_lengths(lengths, gaps, before, after)


def _mixed_row(rng, palette, width, style=None):
    a, b = (int(x) for x in rng.choice(palette, size=2, replace=False))
    style = style or str(rng.choice(("exception", "alternating", "blocks", "random", "balanced")))
    if style in ("first", "middle", "last", "exception"):
        row = np.full(width, a, dtype=np.int8)
        position = (0 if style == "first" else width - 1 if style == "last"
                    else int(rng.integers(1, width - 1)) if style == "middle"
                    else int(rng.integers(width)))
        row[position] = b
    elif style == "alternating":
        row = np.asarray([a if i % 2 else b for i in range(width)], dtype=np.int8)
    elif style in ("blocks", "balanced"):
        cut = int(rng.integers(1, width)) if style == "blocks" else width // 2
        row = np.asarray([a] * cut + [b] * (width - cut), dtype=np.int8)
        if style == "balanced":
            rng.shuffle(row)
    else:
        row = np.asarray(rng.choice(palette, size=width), dtype=np.int8)
        # Guarantee mixed status, including on two-column grids.
        if np.all(row == row[0]):
            row[-1] = a if row[0] != a else b
    return row


def _paint(rng, uniform, width, palette, witness=False):
    grid = np.empty((len(uniform), width), dtype=np.int8)
    previous_mixed = []
    for index, is_uniform in enumerate(uniform):
        if is_uniform:
            grid[index] = int(rng.choice(palette))
        elif previous_mixed and rng.random() < 0.15:
            grid[index] = previous_mixed[int(rng.integers(len(previous_mixed)))]
        else:
            grid[index] = _mixed_row(rng, palette, width)
            previous_mixed.append(grid[index].copy())
    if witness:
        winner = max(_runs(~uniform), key=lambda band: band[1] - band[0])
        slots = np.arange(*winner)
        rng.shuffle(slots)
        styles = ["first", "middle", "last"]
        rng.shuffle(styles)
        bases = np.asarray([0, 5, 6])
        rng.shuffle(bases)
        for index, style, base in zip(slots, styles, bases):
            row = _mixed_row(rng, palette, width, style)
            # Recolor the repeated value to a role color. A permutation keeps
            # the exception distinct, even if it originally equals the base.
            repeated = int(row[1] if style == "first" else row[0])
            other = int(row[np.flatnonzero(row != repeated)[0]])
            different = int(rng.choice([c for c in palette if c != base]))
            grid[index] = np.where(row == other, different, base)
        positives = np.flatnonzero(uniform)
        rng.shuffle(positives)
        for index, color in zip(positives, (0, 5, 6)):
            grid[index] = color
    return grid


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    width_limit = int(rng.choice((8, 16, 30)))
    boundary = str(rng.choice(("top", "bottom")))
    puzzle, used, test_structures = {"train": [], "test": []}, set(), set()
    training_widths = []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            profile = ("test" if split == "test" else "witness" if index == 0
                       else "boundary" if index == 1
                       else str(rng.choice(("simple", "random", "random"))))
            for _ in range(64):
                uniform = _plan(rng, profile, boundary if profile == "boundary" else None)
                if split == "test" and rng.random() < 0.8:
                    width = int(rng.choice(training_widths))
                else:
                    width = int(rng.integers(3 if profile == "witness" else 2, width_limit + 1))
                structure = (width, uniform.tobytes())
                if split == "test" and structure in test_structures:
                    continue
                if profile == "witness":
                    extra = rng.permutation([c for c in range(10) if c not in (0, 5, 6)])
                    palette = np.asarray([0, 5, 6, *extra[:int(rng.integers(0, 8))]])
                else:
                    palette = rng.choice(10, size=int(rng.integers(2, 11)), replace=False)
                grid = _paint(rng, uniform, width, palette, profile == "witness")
                if _key(grid) in used:
                    continue
                used.add(_key(grid))
                if split == "test":
                    test_structures.add(structure)
                else:
                    training_widths.append(width)
                puzzle[split].append({"input": grid, "output": _output_for_mask(uniform, width)})
                break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _exception_styles(row):
    if len(row) < 3:
        return set()
    counts = Counter(int(x) for x in row)
    if len(counts) != 2 or sorted(counts.values()) != [1, len(row) - 1]:
        return set()
    exceptional = next(c for c, count in counts.items() if count == 1)
    index = int(np.flatnonzero(row == exceptional)[0])
    return {"first" if index == 0 else "last" if index == len(row) - 1 else "middle"}


def validate_generated_puzzle(puzzle):
    """Check semantics plus the sampler's explicit demonstration/shift contract."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal pair counts")
    seen, styles, train_counts, test_counts = set(), set(), [], []
    has_competition = has_boundary = has_singleton = False
    uniform_role_colors = set()
    test_structures = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == output.shape, "shape changed")
            key = _key(grid)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            uniform = np.all(grid == grid[:, :1], axis=1)
            bands = _runs(~uniform)
            expected = _output_for_mask(uniform, grid.shape[1])
            _require(np.array_equal(output, expected), "wrong row mask or selected rectangle")
            highlight_rows = np.flatnonzero(output[:, 0] == HIGHLIGHT)
            start, end = int(highlight_rows[0]), int(highlight_rows[-1]) + 1
            _require(end - start >= 2 and np.all(~uniform[start:end]), "invalid highlighted band")
            _require(np.all(output[uniform] == POSITIVE), "uniform rows must be positive")
            unselected = ~uniform.copy()
            unselected[start:end] = False
            _require(np.all(output[unselected] == BACKGROUND), "unselected mixed rows must be background")
            if split == "train":
                train_counts.append(len(bands))
                has_singleton |= any(b - a == 1 for a, b in bands)
                has_boundary |= start == 0 or end == len(uniform)
                has_competition |= (any(b <= start and b - a >= 2 for a, b in bands)
                                    and any(a >= end and b - a >= 2 for a, b in bands))
                uniform_role_colors.update(int(x) for x in grid[uniform, 0])
                for row in grid[~uniform]:
                    styles.update(_exception_styles(row))
            else:
                test_counts.append(len(bands))
                structure = (grid.shape[1], uniform.tobytes())
                _require(structure not in test_structures, "repeated test answer structure")
                test_structures.add(structure)
    _require(styles == {"first", "middle", "last"}, "missing full-row equality contrasts")
    _require({0, 5, 6} <= uniform_role_colors, "training must show literal colors do not determine roles")
    _require(has_competition and has_singleton and has_boundary, "missing selection/boundary demonstrations")
    _require(max(train_counts) <= 4 and min(test_counts) >= 5, "missing candidate-count shift")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role])
                digest.update(bytes(grid.shape))
                digest.update(grid.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming 42-seed checks. No files or execution traces saved."""
    fingerprints, shapes, masks, palettes, targets, band_counts = set(), set(), set(), set(), set(), set()
    features = Counter()
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                original = grid.copy()
                _require(np.array_equal(solve(grid), output), "2D rectangle search disagrees with band construction")
                _require(np.array_equal(grid, original), "solve mutated input")
                uniform = np.all(grid == grid[:, :1], axis=1)
                bands = _runs(~uniform)
                start, end = max(bands, key=lambda band: band[1] - band[0])
                shapes.add(grid.shape)
                masks.add(tuple(bool(x) for x in uniform))
                palettes.add(tuple(int(x) for x in np.unique(grid)))
                targets.add((end - start, grid.shape[1]))
                band_counts.add(len(bands))
                features["top"] += start == 0
                features["bottom"] += end == len(grid)
                features["interior"] += start > 0 and end < len(grid)
                features["width_two"] += grid.shape[1] == 2
                features["height_30"] += grid.shape[0] == 30
                features["width_30"] += grid.shape[1] == 30
                features["singleton"] += any(b - a == 1 for a, b in bands)
                features["all_mixed"] += not np.any(uniform)
                pairs_checked += 1
    _require(len(fingerprints) == 42, "identical puzzles across seeds")
    _require(len(shapes) >= 60 and len(masks) >= 140 and len(targets) >= 35, "insufficient structural variation")
    _require(len(palettes) >= 60 and {4, 5, 6, 7} <= band_counts, "insufficient palette/count variation")
    _require(all(features[name] > 0 for name in ("top", "bottom", "interior", "width_two", "height_30")),
             "missing boundary or minimal-width coverage")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + train * 10 + test, train, test)
            _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test, "counts changed")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exercise valid members beyond the sampler: a 2x2 winner, all-mixed
    # inputs, 30x30, all first/interior/last exceptions, and arbitrary colors.
    for width in (2, 3, 30):
        grid = np.tile(np.arange(width) % 10, (2, 1))
        _require(np.all(solve(grid) == HIGHLIGHT), "minimal/all-mixed member failed")
    grid = np.tile(np.arange(30) % 10, (30, 1))
    _require(np.all(solve(grid) == HIGHLIGHT), "30x30 all-mixed member failed")
    # Undefined cases are rejected rather than receiving an arbitrary tie rule.
    for rows in (("111", "222"), ("121", "111", "232"),
                 ("121", "232", "111", "343", "454"), ("1", "2")):
        try:
            solve(np.asarray([[int(x) for x in row] for row in rows]))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid or tied rectangle accepted")
    # Column permutations and bijective input recoloring preserve every output.
    puzzle = generate_puzzle(17)
    recolor = np.asarray([8, 4, 9, 1, 6, 0, 7, 2, 5, 3])
    for pair in puzzle["train"] + puzzle["test"]:
        _require(np.array_equal(solve(pair["input"][:, ::-1]), pair["output"]), "column-order dependence")
        _require(np.array_equal(solve(recolor[pair["input"]]), pair["output"]), "literal-color dependence")
        _require(np.array_equal(solve(pair["input"][::-1]), pair["output"][::-1]), "vertical-reflection dependence")
    witness = puzzle["train"][0]
    grid, output = witness["input"], witness["output"]
    uniform = np.all(grid == grid[:, :1], axis=1)
    no_b = np.repeat(np.where(uniform, POSITIVE, BACKGROUND)[:, None], grid.shape[1], axis=1)
    all_mixed = no_b.copy()
    all_mixed[~uniform] = HIGHLIGHT
    _require(not np.array_equal(no_b, output), "omitting B did not change answer")
    _require(not np.array_equal(all_mixed, output), "highlight-all shortcut was not distinguished")
    # Omitting A and searching literal zero rectangles cannot recolor a
    # uniform input-6 row to 5. It also often has no eligible raw rectangle.
    raw_b = grid.copy()
    try:
        top, bottom, left, right = _largest_empty_rectangle(grid == BACKGROUND)
        raw_b[top:bottom, left:right] = HIGHLIGHT
    except ValueError:
        pass
    _require(not np.array_equal(raw_b, output), "omitting A did not change answer")
    for weak_uniform in (grid[:, 0] == grid[:, -1], grid[:, 0] == grid[:, 1],
                         grid[:, -2] == grid[:, -1]):
        _require(np.any(weak_uniform & ~uniform), "two-cell equality shortcut survived training")
    return {"seeds": 42, "pairs_checked": pairs_checked, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "row_masks": len(masks), "input_palettes": len(palettes),
            "highlight_shapes": len(targets), "mixed_band_counts": sorted(band_counts),
            "features": dict(features), "independent_solver": False}
