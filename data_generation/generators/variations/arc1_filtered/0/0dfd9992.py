"""Generate square-periodic ARC grids with recoverable erased cells."""

import hashlib

import numpy as np


_SOURCE_DIGESTS = {
    "743d718af3de29088b8a0cd25961c6c1a6ab1329772ee1976708558b1cc13e48",
    "e90bea0c8cdaa205a286f0f60e43fae6360d80117091fd5b55b215c68416319d",
    "8d293bd35f521ce363c2b21a1edfbdfe6c882366f64a0a227fdecc414524af51",
    "a5ba3bd2174d80a4681122a46fcb31748e998c9ada4c6f35732d0799f96d47f3",
}


def _choose_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("invalid pair count")
    if num_train is None and num_test is None:
        choices = [(a, b) for b in range(1, 5) for a in range(max(2, b + 1), 11)]
        return choices[int(rng.integers(len(choices)))]
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("train count must exceed test count")
    return int(num_train), int(num_test)


def _axis_period(grid, axis):
    length = grid.shape[axis]
    for offset in range(1, length):
        if axis == 0:
            equal = np.array_equal(grid[:-offset, :], grid[offset:, :])
        else:
            equal = np.array_equal(grid[:, :-offset], grid[:, offset:])
        if equal:
            return offset
    return length


def _visible_tile(grid, blank, period, complete=True):
    tile = np.full((period, period), -1, dtype=np.int16)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            color = int(grid[row, col])
            if color == blank:
                continue
            r, c = row % period, col % period
            if tile[r, c] >= 0 and tile[r, c] != color:
                return None
            tile[r, c] = color
    if complete and np.any(tile < 0):
        return None
    return tile


def _make_tile(rng, period, palette):
    ncolors = len(palette)
    for _ in range(300):
        mode = int(rng.choice(3, p=[0.72, 0.15, 0.13]))
        if mode == 0:
            bands = int(rng.integers(3, min(5, period - 1) + 1))
            positions = [(r, c) for r in range(bands) for c in range(r, bands)]
            if ncolors > len(positions):
                continue
            codes = np.concatenate((np.arange(bands), rng.integers(bands, size=period - bands)))
            rng.shuffle(codes)
            entries = rng.integers(ncolors, size=len(positions))
            entries[rng.choice(len(positions), size=ncolors, replace=False)] = np.arange(ncolors)
            lookup = np.zeros((bands, bands), dtype=np.int8)
            for (r, c), value in zip(positions, entries):
                lookup[r, c] = lookup[c, r] = value
            indexes = lookup[codes[:, None], codes[None, :]]
        elif mode == 1:
            indexes = rng.integers(ncolors, size=(period, period))
            indexes = np.triu(indexes) + np.triu(indexes, 1).T
        else:
            indexes = rng.integers(ncolors, size=(period, period))
        tile = np.asarray(palette, dtype=np.int8)[indexes]
        if (len(np.unique(tile)) == ncolors
                and _axis_period(tile, 0) == period
                and _axis_period(tile, 1) == period):
            return tile
    raise RuntimeError("could not build a full-period tile")


def _canvas(rng, period, regime):
    minimum = max(14 if regime == "source_train" else 15, 2 * period)
    multiples = [n for n in range(minimum, 31) if n % period == 0]
    clipped = [n for n in range(minimum, 31) if n % period != 0]
    form = int(rng.integers(4))
    if form == 0:
        return int(rng.choice(multiples)), int(rng.choice(multiples))
    if form == 1:
        exact, clip = int(rng.choice(multiples)), int(rng.choice(clipped))
        return (exact, clip) if rng.random() < 0.5 else (clip, exact)
    if form == 2:
        return int(rng.choice(clipped)), int(rng.choice(clipped))
    if regime == "target_test" and period == 9 and rng.random() < 0.45:
        return 21, 21
    return int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))


def _patch(rng, height, width):
    kind = int(rng.integers(7))
    if kind == 0:
        ph, pw = 1, 1
    elif kind == 1:
        ph, pw = 1, int(rng.integers(2, 7))
    elif kind == 2:
        ph, pw = int(rng.integers(2, 7)), 1
    elif kind == 3:
        ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    elif kind == 4:
        ph, pw = int(rng.integers(3, 7)), int(rng.integers(2, 6))
    else:
        ph, pw = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    ph, pw = min(ph, height), min(pw, width)
    local = np.ones((ph, pw), dtype=bool)
    if kind == 5 and ph > 2 and pw > 2:
        local[1:, 1:] = False
    elif kind == 6 and ph > 2 and pw > 2:
        local[0, 0] = False
        local[-1, -1] = False
    return local


def _component_sizes(mask):
    remaining = mask.copy()
    sizes = []
    for row, col in np.argwhere(mask):
        if not remaining[row, col]:
            continue
        stack = [(int(row), int(col))]
        remaining[row, col] = False
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and remaining[rr, cc]):
                    remaining[rr, cc] = False
                    stack.append((rr, cc))
        sizes.append(size)
    return sizes


def _make_input(rng, output, blank, period, regime):
    height, width = output.shape
    rows, cols = np.indices(output.shape)
    phases = (rows % period) * period + cols % period
    for _ in range(240):
        mask = np.zeros(output.shape, dtype=bool)
        style = int(rng.choice(3, p=[0.55, 0.3, 0.15]))
        count = int(rng.integers(4, 7)) if style == 0 else int(rng.integers(2, 7))
        placed = 0
        for _ in range(count):
            local = _patch(rng, height, width)
            ph, pw = local.shape
            for _ in range(90):
                if rng.random() < 0.18:
                    top = int(rng.choice([0, height - ph]))
                else:
                    top = int(rng.integers(height - ph + 1))
                if rng.random() < 0.18:
                    left = int(rng.choice([0, width - pw]))
                else:
                    left = int(rng.integers(width - pw + 1))
                if style == 0:
                    r0, r1 = max(0, top - 1), min(height, top + ph + 1)
                    c0, c1 = max(0, left - 1), min(width, left + pw + 1)
                    if np.any(mask[r0:r1, c0:c1]):
                        continue
                mask[top:top + ph, left:left + pw] |= local
                placed += 1
                break
        if style == 0 and placed < count:
            continue
        erased = int(mask.sum())
        if not 10 <= erased <= min(115, height * width // 3):
            continue
        if len(np.unique(phases[mask])) < min(8, period * period // 2):
            continue
        visible = np.bincount(phases[~mask].ravel(), minlength=period * period)
        if visible.min() < 1:
            continue
        sizes = _component_sizes(mask)
        if style == 0 and len(sizes) < 4:
            continue
        inp = output.copy()
        inp[mask] = blank
        if any(_visible_tile(inp, blank, smaller, False) is not None
               for smaller in range(1, period)):
            continue
        return inp
    raise RuntimeError("could not build recoverable erasures")


def _pair(rng, blank, palette, period, regime):
    height, width = _canvas(rng, period, regime)
    tile = _make_tile(rng, period, palette)
    output = tile[np.arange(height)[:, None] % period,
                  np.arange(width)[None, :] % period].copy()
    inp = _make_input(rng, output, blank, period, regime)
    return {"input": inp.astype(np.int8), "output": output.astype(np.int8)}


def _digest_pair(inp, output):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.uint8).tobytes()
                          + output.astype(np.uint8).tobytes()).hexdigest()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    blanks = set()
    train_periods = set()
    train_colors = set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, output = pair["input"], pair["output"]
            for grid in (inp, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == output.shape
            key = (inp.shape, inp.astype(np.int8).tobytes())
            pair_key = (key, output.astype(np.int8).tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _digest_pair(inp, output) not in _SOURCE_DIGESTS
            erased = inp != output
            assert erased.any()
            blank_values = np.unique(inp[erased])
            assert len(blank_values) == 1
            blank = int(blank_values[0])
            blanks.add(blank)
            assert np.array_equal(erased, inp == blank)
            assert not np.any(output == blank)
            period = _axis_period(output, 0)
            assert period == _axis_period(output, 1)
            assert period <= 10 and output.shape[0] >= 2 * period
            assert output.shape[1] >= 2 * period
            if regime == "train":
                assert period in (4, 5, 6, 7)
                train_periods.add(period)
                train_colors.update(map(int, np.unique(output)))
            else:
                assert period in (4, 5, 6, 7, 8, 9, 10)
            tile = _visible_tile(inp, blank, period)
            assert tile is not None
            restored = tile[np.arange(inp.shape[0])[:, None] % period,
                            np.arange(inp.shape[1])[None, :] % period]
            assert np.array_equal(output, restored)
            assert all(_visible_tile(inp, blank, smaller, False) is None
                       for smaller in range(1, period))
            assert len(_component_sizes(erased)) >= 1
            assert len(np.unique(output)) >= 4
    assert len(blanks) == 1
    assert len(train_periods) >= min(2, len(puzzle["train"]))
    assert len(train_colors) >= 4


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    blank = int(rng.integers(10))
    colors = np.asarray([c for c in range(10) if c != blank], dtype=np.int8)
    rng.shuffle(colors)
    train_pool = colors[:int(rng.integers(5, 8))]
    period_order = list(map(int, rng.permutation([4, 6, 7])))
    train_periods = period_order[:num_train]
    while len(train_periods) < num_train:
        train_periods.append(int(rng.integers(4, 8)))
    puzzle = {"train": [], "test": []}
    seen = set()
    for period in train_periods:
        for _ in range(30):
            count = int(rng.integers(4, min(7, len(train_pool)) + 1))
            palette = rng.choice(train_pool, size=count, replace=False)
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            pair = _pair(pair_rng, blank, palette, period, "source_train")
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and _digest_pair(pair["input"], pair["output"]) not in _SOURCE_DIGESTS:
                break
        else:
            raise RuntimeError("could not build unique training pair")
        seen.add(key)
        puzzle["train"].append(pair)
    seen_colors = set().union(*(set(map(int, np.unique(p["output"]))) for p in puzzle["train"]))
    novel_colors = np.asarray(sorted(set(map(int, colors)) - seen_colors), dtype=np.int8)
    familiar_colors = np.asarray(sorted(seen_colors), dtype=np.int8)
    for _ in range(num_test):
        for _ in range(30):
            period = int(rng.choice([4, 5, 6, 7, 8, 9, 10], p=[.09, .09, .11, .12, .18, .25, .16]))
            novelty = int(rng.integers(0, min(3, len(novel_colors)) + 1))
            total = int(rng.integers(max(4, novelty + 1), 9))
            total = min(total, novelty + len(familiar_colors))
            novel = rng.choice(novel_colors, size=novelty, replace=False)
            familiar = rng.choice(familiar_colors, size=total - novelty, replace=False)
            palette = np.concatenate((novel, familiar))
            pair_rng = np.random.default_rng(int(rng.integers(2**63)))
            pair = _pair(pair_rng, blank, palette, period, "target_test")
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and _digest_pair(pair["input"], pair["output"]) not in _SOURCE_DIGESTS:
                break
        else:
            raise RuntimeError("could not build unique test pair")
        seen.add(key)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple((pair["input"].shape, pair["input"].tobytes(), pair["output"].tobytes())
                 for regime in ("train", "test") for pair in puzzle[regime])


def self_test() -> None:
    train_counts, test_counts, blanks, shapes, mask_sizes = set(), set(), set(), set(), set()
    fingerprints, periods, test_novelties, components = set(), set(), set(), set()
    test_periods, palette_sizes, band_counts, test_shapes = set(), set(), set(), set()
    symmetric = 0
    banded_symmetric = 0
    edge_masks = 0
    interior_masks = 0
    repeated_erased_phases = 0
    clipped_forms = set()
    combined = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))
        train_colors = set().union(*(set(map(int, np.unique(p["output"]))) for p in puzzle["train"]))
        local_periods = set()
        local_shapes = set()
        local_blanks = set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, output = pair["input"], pair["output"]
                period = _axis_period(output, 0)
                assert period == _axis_period(output, 1)
                local_periods.add(period)
                local_shapes.add(output.shape)
                periods.add(period)
                shapes.add(output.shape)
                palette_sizes.add(len(np.unique(output)))
                erased = inp != output
                mask_sizes.add(int(erased.sum()))
                edge = bool(erased[0].any() or erased[-1].any()
                            or erased[:, 0].any() or erased[:, -1].any())
                edge_masks += int(edge)
                interior_masks += int(not edge)
                phases = (np.indices(output.shape)[0] % period) * period + np.indices(output.shape)[1] % period
                repeated_erased_phases += int(len(np.unique(phases[erased])) < int(erased.sum()))
                ncomp = len(_component_sizes(erased))
                components.add(ncomp)
                blank = int(inp[erased][0])
                local_blanks.add(blank)
                blanks.add(blank)
                tile = _visible_tile(inp, blank, period)
                assert tile is not None
                assert np.array_equal(output, tile[np.arange(output.shape[0])[:, None] % period,
                                                   np.arange(output.shape[1])[None, :] % period])
                sym = np.array_equal(output[:period, :period], output[:period, :period].T)
                symmetric += int(sym)
                bands = len(np.unique(output[:period, :period], axis=0))
                band_counts.add(bands)
                banded_symmetric += int(sym and 3 <= bands <= 5)
                total += 1
                form = (output.shape[0] % period == 0, output.shape[1] % period == 0)
                clipped_forms.add(form)
                if regime == "test":
                    test_periods.add(period)
                    test_shapes.add(output.shape)
                    novelty = len(set(map(int, np.unique(output))) - train_colors)
                    test_novelties.add(novelty)
                    combined += int(sym and ncomp >= 4 and not all(form))
        assert len(local_blanks) == 1 and len(local_periods) >= 2
        assert len(local_shapes) >= 2
        if seed < 5:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(blanks) >= 5 and len(shapes) >= 20 and len(mask_sizes) >= 25
    assert len(fingerprints) == 42 and len(periods) >= 6
    assert {4, 5, 6, 7, 8, 9, 10}.issubset(test_periods)
    assert len(test_shapes) >= 25 and {4, 5, 6, 7}.issubset(palette_sizes)
    assert {3, 4, 5}.issubset(band_counts) and banded_symmetric / total > 0.45
    assert edge_masks >= 20 and interior_masks >= 5
    assert repeated_erased_phases >= 20
    assert {0, 1, 2}.issubset(test_novelties)
    assert {(True, True), (True, False), (False, True), (False, False)}.issubset(clipped_forms)
    assert max(components) >= 5 and min(components) <= 2
    assert symmetric / total > 0.55 and combined >= 3
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 2)):
        first, second = generate_puzzle(137, a, b), generate_puzzle(137, a, b)
        assert (len(first["train"]), len(first["test"])) == (a, b)
        assert _fingerprint(first) == _fingerprint(second)
        validate_generated_puzzle(first)
    for request in ({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10}):
        puzzle = generate_puzzle(231, **request)
        validate_generated_puzzle(puzzle)
        for key, value in request.items():
            assert len(puzzle["train" if key == "num_train" else "test"]) == value
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
