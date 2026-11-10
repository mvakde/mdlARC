import numpy as np


_ORIGINAL_INPUTS = {
    ("0202020", "3233323", "0202020", "3233323", "0202020", "3233323", "0202020"),
    ("0400400", "0400400", "1111411", "0400400", "1111411", "0400400", "0400400"),
    ("0009000", "6666666", "0009000", "0009000", "0009000", "0009000", "0009000"),
    ("0101001", "0101001", "5555551", "0101001", "0101001", "5555551", "0101001"),
    ("000000000", "003300000", "999999999", "999999999", "003300000", "003300000", "003300000", "003300000", "000000000"),
    ("0000000000", "0006060000", "5556555555", "0006060000", "5556555555", "0006060000", "0000000000", "0000000000", "0000000000"),
}


def _key(grid):
    return grid.shape, tuple(int(v) for v in grid.flat)


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _blocks(indices):
    values = list(indices)
    if not values:
        return []
    result = []
    first = last = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value != last + 1:
            result.append((first, last))
            first = value
        last = value
    result.append((first, last))
    return result


def _checked_count(value, low, high, name):
    if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _counts(rng, num_train, num_test):
    _checked_count(num_train, 2, 10, "num_train")
    _checked_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _size(rng, regime):
    if regime == "source_train":
        ranges = ((6, 8), (9, 14), (15, 19))
        probabilities = (0.32, 0.53, 0.15)
    else:
        ranges = ((6, 8), (9, 14), (15, 24))
        probabilities = (0.22, 0.36, 0.42)
    category = int(rng.choice(3, p=probabilities))
    low, high = ranges[category]
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    return height, width


def _bands(rng, height, simple=False):
    count = int(rng.integers(1, min(2 if simple else 3, (height - 1) // 2) + 1))
    thicknesses = [1] * count
    capacity = height - (2 * count + 1)
    extras = 0 if simple else int(rng.integers(0, min(2 * count, capacity) + 1))
    for _ in range(extras):
        available = [i for i, thick in enumerate(thicknesses) if thick < 3]
        thicknesses[int(rng.choice(available))] += 1
    gaps = [2] + [1] * (count - 1) + [2] if simple else [1] * (count + 1)
    slack = height - sum(thicknesses) - sum(gaps)
    if slack:
        allocation = rng.multinomial(slack, np.ones(count + 1) / (count + 1))
        gaps = [gap + int(extra) for gap, extra in zip(gaps, allocation)]
    bands = []
    row = gaps[0]
    for i, thick in enumerate(thicknesses):
        bands.append((row, thick))
        row += thick + gaps[i + 1]
    return bands


def _columns(rng, width, count, simple=False):
    widths = [1] * count
    maximum_extra = min(2 * count, width - (2 * count))
    extras = 0 if simple else int(rng.integers(0, maximum_extra + 1))
    for _ in range(extras):
        available = [i for i, thick in enumerate(widths) if thick < 3]
        widths[int(rng.choice(available))] += 1
    slack = width - sum(widths) - (count - 1)
    gaps = [0] + [1] * (count - 1) + [0]
    fixed = [False] * (count + 1)
    fixed[0] = bool(rng.random() < 0.30)
    fixed[-1] = bool(rng.random() < 0.30)
    if count == 1 and fixed[0] and fixed[-1]:
        fixed[int(rng.integers(0, 2))] = False
    available = [i for i in range(count + 1) if not fixed[i]]
    allocation = rng.multinomial(slack, np.ones(len(available)) / len(available))
    for i, extra in zip(available, allocation):
        gaps[i] += int(extra)
    columns = []
    column = gaps[0]
    for i, thick in enumerate(widths):
        columns.append((column, column + thick - 1))
        column += thick + gaps[i + 1]
    return columns


def _construct(rng, regime, colors, mode):
    height, width = _size(rng, regime)
    simple = regime == "target_test" and bool(rng.random() < 0.25)
    if simple:
        height = int(rng.integers(7, 12))
        width = int(rng.integers(7, 12))
    bands = _bands(rng, height, simple)
    maximum = min(5, width // 2)
    minimum = 2 if mode == "mixed" else 1
    count = int(rng.integers(minimum, (2 if simple else maximum) + 1))
    columns = _columns(rng, width, count, simple)
    if mode == "mixed":
        keeps = [False, True] + [bool(v) for v in rng.integers(0, 2, size=count - 2)]
        rng.shuffle(keeps)
    elif mode == "erased":
        keeps = [False] * count
    else:
        keeps = [True] * count
    first_band = bands[0][0]
    last_band = bands[-1][0] + bands[-1][1] - 1
    starts = [int(rng.integers(0, first_band)) for _ in columns]
    ends = [int(rng.integers(last_band + 1, height)) for _ in columns]
    background, band_color, stripe_color = colors
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    band_rows = np.zeros(height, dtype=bool)
    for start, thick in bands:
        band_rows[start:start + thick] = True
    inp[band_rows, :] = band_color
    out[band_rows, :] = band_color
    for (left, right), start, end, keep in zip(columns, starts, ends, keeps):
        for row in range(start, end + 1):
            if not band_rows[row]:
                inp[row, left:right + 1] = stripe_color
                if keep:
                    out[row, left:right + 1] = stripe_color
            elif keep:
                inp[row, left:right + 1] = stripe_color
                out[row, left:right + 1] = stripe_color
    return inp, out


def _interpret(inp, colors):
    background, band, stripe = colors
    height, width = inp.shape
    band_rows = np.flatnonzero(np.any(inp == band, axis=1))
    bands = _blocks(band_rows)
    if not 1 <= len(bands) <= 3 or bands[0][0] < 1 or bands[-1][1] >= height - 1:
        return None
    if any(right - left + 1 > 3 for left, right in bands):
        return None
    band_mask = np.zeros(height, dtype=bool)
    band_mask[band_rows] = True
    columns = np.flatnonzero(np.any((inp == stripe) & ~band_mask[:, None], axis=0))
    runs = _blocks(columns)
    if not 1 <= len(runs) <= 5 or any(right - left + 1 > 3 for left, right in runs):
        return None
    if any(left - previous <= 1 for (_, previous), (left, _) in zip(runs, runs[1:])):
        return None
    expected = np.full_like(inp, background)
    expected[band_mask, :] = band
    reconstructed = expected.copy()
    statuses = []
    spans = []
    for left, right in runs:
        visible = np.flatnonzero((inp[:, left] == stripe) & ~band_mask)
        if len(visible) < 2:
            return None
        start, end = int(visible[0]), int(visible[-1])
        if start >= bands[0][0] or end <= bands[-1][1]:
            return None
        crossing = inp[np.ix_(band_rows, np.arange(left, right + 1))]
        if np.all(crossing == stripe):
            keep = True
        elif np.all(crossing == band):
            keep = False
        else:
            return None
        statuses.append(keep)
        spans.append((start, end))
        for row in range(start, end + 1):
            if not band_mask[row]:
                reconstructed[row, left:right + 1] = stripe
                if keep:
                    expected[row, left:right + 1] = stripe
            elif keep:
                reconstructed[row, left:right + 1] = stripe
                expected[row, left:right + 1] = stripe
    if not np.array_equal(inp, reconstructed):
        return None
    return expected, {"colors": colors, "bands": bands, "runs": runs, "statuses": statuses, "spans": spans}


def _possibilities(inp):
    palette = tuple(int(v) for v in np.unique(inp))
    if len(palette) != 3:
        return []
    candidates = []
    for background in palette:
        for band in palette:
            if band == background:
                continue
            stripe = next(v for v in palette if v not in (background, band))
            result = _interpret(inp, (background, band, stripe))
            if result is not None:
                candidates.append(result)
    return candidates


def _unique_interpretation(inp):
    candidates = _possibilities(inp)
    if len(candidates) != 1:
        return None
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and 6 <= inp.shape[0] <= (19 if split == "train" else 24)
            assert 6 <= inp.shape[1] <= (19 if split == "train" else 24)
            key = _key(inp)
            assert key not in seen and _rows(inp) not in _ORIGINAL_INPUTS
            seen.add(key)
            analysis = _unique_interpretation(inp)
            assert analysis is not None
            expected, info = analysis
            assert np.array_equal(out, expected)
            infos[split].append(info)
    train_statuses = [info["statuses"] for info in infos["train"]]
    assert any(any(x) and not all(x) for x in train_statuses)
    assert any(not any(x) for x in train_statuses)
    role_tuples = [info["colors"] for split in ("train", "test") for info in infos[split]]
    assert len(set(role_tuples)) > 1
    assert len({tuple(sorted(role)) for role in role_tuples}) > 1
    assert any(a != b for a, b in zip(role_tuples[0], role_tuples[1]))
    assert any(
        len({position for role in role_tuples[:ntrain] for position, color in enumerate(role) if color == digit}) > 1
        for digit in range(10)
    )


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(rng, num_train, num_test)
    anchor = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    permutation = tuple(int(v) for v in rng.permutation(3))
    if permutation == (0, 1, 2):
        permutation = (1, 2, 0)
    swapped = tuple(anchor[i] for i in permutation)
    modes = ["mixed", "erased"] + [str(rng.choice(("mixed", "erased", "kept"), p=(0.55, 0.30, 0.15))) for _ in range(ntrain - 2)]
    rng.shuffle(modes)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            if split == "train" and index == 0:
                colors = anchor
            elif split == "train" and index == 1:
                colors = swapped
            else:
                for _ in range(20):
                    colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
                    if tuple(sorted(colors)) != tuple(sorted(anchor)):
                        break
            mode = modes[index] if split == "train" else str(rng.choice(("mixed", "erased", "kept"), p=(0.55, 0.33, 0.12)))
            for _ in range(300):
                inp, out = _construct(rng, "source_train" if split == "train" else "target_test", colors, mode)
                key = _key(inp)
                if key not in seen and _rows(inp) not in _ORIGINAL_INPUTS and _unique_interpretation(inp) is not None:
                    break
            else:
                raise RuntimeError("Could not construct a unique interpretable pair")
            seen.add(key)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(left, right):
    return all(len(left[split]) == len(right[split]) and all(np.array_equal(a[k], b[k]) for a, b in zip(left[split], right[split]) for k in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, band_counts, band_thicks = set(), set(), set(), set(), set()
    stripe_counts, stripe_widths, modes, backgrounds, color_roles = set(), set(), set(), set(), set()
    clipped = edge_erased = edge_kept = wide_train = thick_train = short_train = unchanged = simple_test = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        within_roles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                candidates = _possibilities(inp)
                assert len(candidates) == 1 and np.array_equal(candidates[0][0], out)
                info = candidates[0][1]
                sizes.add(inp.shape)
                band_counts.add(len(info["bands"]))
                band_thicks.update(b - a + 1 for a, b in info["bands"])
                stripe_counts.add(len(info["runs"]))
                stripe_widths.update(b - a + 1 for a, b in info["runs"])
                backgrounds.add(info["colors"][0])
                color_roles.add(info["colors"])
                within_roles.add(info["colors"])
                statuses = info["statuses"]
                modes.add("mixed" if any(statuses) and not all(statuses) else "kept" if all(statuses) else "erased")
                unchanged += int(np.array_equal(inp, out))
                if split == "test":
                    simple_test += int(
                        len(info["runs"]) == 2
                        and all(a == b for a, b in info["runs"])
                        and all(a == b for a, b in info["bands"])
                        and all(a > 0 or b < inp.shape[0] - 1 for a, b in info["spans"])
                    )
                for (left, right), (start, end), keep in zip(info["runs"], info["spans"], statuses):
                    clipped += int(start > 0 or end < inp.shape[0] - 1)
                    if left == 0 or right == inp.shape[1] - 1:
                        edge_kept += int(keep)
                        edge_erased += int(not keep)
                    if split == "train":
                        wide_train += int(right > left)
                        short_train += int(start > 0 or end < inp.shape[0] - 1)
                if split == "train":
                    thick_train += int(any(b > a for a, b in info["bands"]))
        assert len(within_roles) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(fingerprints) == 42
    assert len(sizes) >= 35 and len(backgrounds) >= 8 and len(color_roles) >= 60
    assert band_counts == {1, 2, 3} and band_thicks == {1, 2, 3}
    assert stripe_counts == {1, 2, 3, 4, 5} and stripe_widths == {1, 2, 3}
    assert modes == {"mixed", "erased", "kept"}
    assert min(clipped, edge_erased, edge_kept, wide_train, thick_train, short_train, unchanged, simple_test) > 0
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + i, *counts)
        validate_generated_puzzle(puzzle)
        assert _same(puzzle, generate_puzzle(1000 + i, *counts))
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 4)):
        try:
            generate_puzzle(999, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {counts}")
