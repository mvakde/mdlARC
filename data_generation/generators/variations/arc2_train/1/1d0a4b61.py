"""Generate periodic-wallpaper restoration puzzles with a shared erasure color."""

import hashlib
import numpy as np


_SOURCE_HASHES = {
    "db395273e39ea9206158b1d00a89d0079e2fa1d0f707912022640a0f62dbb787",
    "a6b3e5fc510a8f1ed4c0b41bee5e02f0fbd9049166775b00ac6c66475f52cbc1",
    "1b5dee5eac320a9bc1626ca1d0c9448d99666a181b3eb3bc27bf96d4630a24b6",
    "081da3810673564fe84da74e0146caf024ec71a052f36775ee7fe882eb6ea04d",
}


def _normalize_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _normalize_count("num_train", num_train, 2, 10)
    num_test = _normalize_count("num_test", num_test, 1, 4)
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


def _axis_period(grid, axis):
    length = grid.shape[axis]
    for period in range(1, length // 2 + 1):
        if axis == 0:
            same = np.array_equal(grid[:-period], grid[period:])
        else:
            same = np.array_equal(grid[:, :-period], grid[:, period:])
        if same:
            return period
    return length


def _period_choices(rng, count, low, high):
    result = []
    for index in range(count):
        for _ in range(20):
            if low <= 2 and high >= 4 and rng.random() < 0.12:
                choice = (2, 4) if rng.random() < 0.5 else (4, 2)
            elif rng.random() < 0.30:
                size = int(rng.integers(low, high + 1))
                choice = (size, size)
            else:
                choice = (int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1)))
            if index != 1 or choice != result[0]:
                break
        result.append(choice)
    return result


def _sample_shape(rng, period):
    sizes = []
    for p in period:
        minimum = max(2 * p, 6)
        if rng.random() < 0.32:
            upper = min(30, minimum + max(4, p))
        else:
            upper = 30
        if rng.random() < 0.34:
            multiples = [n for n in range(minimum, upper + 1) if n % p == 0]
            size = int(rng.choice(multiples)) if multiples else minimum
        else:
            size = int(rng.integers(minimum, upper + 1))
        sizes.append(size)
    if rng.random() < 0.06 and all(p <= 12 for p in period):
        sizes = [25, 25]
    return tuple(sizes)


def _motif(rng, period, palette, style):
    height, width = period
    colors = np.asarray(palette, dtype=np.int8)
    k = len(colors)
    for _ in range(100):
        if style == "mirror":
            half_height, half_width = (height + 1) // 2, (width + 1) // 2
            if k > half_height * half_width or height < 3 or width < 3:
                style = "random"
                continue
            quadrant = rng.choice(colors, size=(half_height, half_width))
            places = rng.choice(half_height * half_width, k, replace=False)
            quadrant.flat[places] = rng.permutation(colors)
            rows = np.minimum(np.arange(height), height - 1 - np.arange(height))
            cols = np.minimum(np.arange(width), width - 1 - np.arange(width))
            tile = quadrant[rows[:, None], cols[None, :]].copy()
        elif style == "bands":
            row_colors = rng.integers(0, k, size=height)
            col_colors = rng.integers(0, k, size=width)
            tile = colors[(row_colors[:, None] + col_colors[None, :]) % k].copy()
            if rng.random() < 0.5:
                tile[::int(rng.integers(2, 5)), :] = colors[0]
            if rng.random() < 0.5:
                tile[:, ::int(rng.integers(2, 5))] = colors[0]
        elif style == "shapes":
            tile = np.full(period, colors[0], dtype=np.int8)
            for color in rng.permutation(colors[1:]):
                top = int(rng.integers(0, height))
                left = int(rng.integers(0, width))
                dh = int(rng.integers(1, min(4, height - top) + 1))
                dw = int(rng.integers(1, min(4, width - left) + 1))
                tile[top:top + dh, left:left + dw] = color
            for _ in range(int(rng.integers(1, 5))):
                top = int(rng.integers(0, height))
                left = int(rng.integers(0, width))
                tile[top:min(height, top + int(rng.integers(1, 4))),
                     left:min(width, left + int(rng.integers(1, 4)))] = rng.choice(colors)
        elif style == "separators":
            spacing = int(rng.integers(2, 4))
            tile = rng.choice(colors, size=period).astype(np.int8)
            tile[::spacing, :] = colors[0]
            tile[:, ::spacing] = colors[0]
            # Block interiors change color across the full cycle, even though
            # separator lines recur more often.
            for r in range(1, height):
                for c in range(1, width):
                    if r % spacing and c % spacing:
                        tile[r, c] = colors[(r // spacing + 2 * (c // spacing)) % k]
        else:
            if rng.random() < 0.45:
                weights = np.full(k, 1 / k)
            else:
                weights = rng.dirichlet(np.full(k, float(rng.uniform(0.8, 2.5))))
            tile = rng.choice(colors, size=period, p=weights).astype(np.int8)
            places = rng.choice(height * width, k, replace=False)
            tile.flat[places] = rng.permutation(colors)
        if set(map(int, np.unique(tile))) != set(map(int, colors)):
            continue
        if _axis_period(tile, 0) == height and _axis_period(tile, 1) == width:
            return tile
    # Rare structured configurations cannot carry every sampled color with
    # their requested period. A random tile remains faithful to the rule.
    if style != "random":
        return _motif(rng, period, palette, "random")
    raise RuntimeError("failed to construct motif")


def _clean_grid(tile, shape, rng):
    p, q = tile.shape
    row_phase = int(rng.integers(0, p))
    col_phase = int(rng.integers(0, q))
    rows = (np.arange(shape[0]) + row_phase) % p
    cols = (np.arange(shape[1]) + col_phase) % q
    return tile[rows[:, None], cols[None, :]].astype(np.int8, copy=True)


def _damage_mask(rng, shape, period):
    height, width = shape
    interior = bool(rng.random() < 0.60)
    top_limit = 1 if interior else 0
    bottom_limit = height - 1 if interior else height
    left_limit = 1 if interior else 0
    right_limit = width - 1 if interior else width
    tier = int(rng.choice(4, p=[0.26, 0.35, 0.31, 0.08]))
    ranges = ((0.005, 0.055), (0.055, 0.16), (0.16, 0.32), (0.32, 0.46))
    fraction = float(rng.uniform(*ranges[tier]))
    target = max(1, int(round(fraction * height * width)))
    style = int(rng.integers(0, 5))
    mask = np.zeros(shape, dtype=bool)
    area_height = bottom_limit - top_limit
    area_width = right_limit - left_limit

    if style == 3:
        # A broad patch may be light or heavy; its size is sampled apart from
        # period and train/test split.
        patch_area = max(2, int(target * rng.uniform(0.45, 0.85)))
        patch_height = max(1, min(area_height, int(round(np.sqrt(patch_area) * rng.uniform(0.7, 1.5)))))
        patch_width = max(1, min(area_width, int(np.ceil(patch_area / patch_height))))
        top = int(rng.integers(top_limit, bottom_limit - patch_height + 1))
        left = int(rng.integers(left_limit, right_limit - patch_width + 1))
        mask[top:top + patch_height, left:left + patch_width] = True

    for _ in range(max(25, target * 3)):
        if int(mask.sum()) >= target:
            break
        row = int(rng.integers(top_limit, bottom_limit))
        col = int(rng.integers(left_limit, right_limit))
        if style == 0:
            mask[row, col] = True
        elif style == 1:
            length = int(rng.integers(2, max(3, min(max(period) + 5, 14))))
            if rng.random() < 0.5:
                mask[row:min(bottom_limit, row + length), col] = True
            else:
                mask[row, col:min(right_limit, col + length)] = True
        else:
            maximum = 4 if style == 4 else max(4, max(period))
            dh = int(rng.integers(1, min(maximum, bottom_limit - row) + 1))
            dw = int(rng.integers(1, min(maximum, right_limit - col) + 1))
            mask[row:row + dh, col:col + dw] = True

    # Preserve two independent witnesses at every motif phase. This keeps
    # even masks that cross many tiles recoverable from the visible input.
    p, q = period
    for r in range(p):
        for c in range(q):
            phase = mask[r::p, c::q]
            deficit = 2 - (phase.size - int(phase.sum()))
            if deficit > 0:
                positions = np.argwhere(phase)
                rng.shuffle(positions)
                for local_r, local_c in positions[:deficit]:
                    mask[r + int(local_r) * p, c + int(local_c) * q] = False
    return mask


def _candidate_periods(grid, marker, axis):
    length = grid.shape[axis]
    result = []
    for period in range(1, length // 2 + 1):
        if axis == 0:
            a, b = grid[:-period], grid[period:]
        else:
            a, b = grid[:, :-period], grid[:, period:]
        reliable = (a != marker) & (b != marker)
        if int(reliable.sum()) < max(3, int(0.06 * reliable.size)):
            continue
        if np.array_equal(a[reliable], b[reliable]):
            result.append(period)
    return result


def _completion(grid, marker, period):
    p, q = period
    completed = grid.copy()
    for r in range(p):
        for c in range(q):
            phase = grid[r::p, c::q]
            visible = np.unique(phase[phase != marker])
            if visible.size != 1:
                return None
            completed[r::p, c::q][phase == marker] = visible[0]
    return completed


def _infer_completion(grid, marker):
    rows = _candidate_periods(grid, marker, 0)
    cols = _candidate_periods(grid, marker, 1)
    valid = []
    for p in rows:
        for q in cols:
            completed = _completion(grid, marker, (p, q))
            if completed is not None:
                valid.append(((p, q), completed))
    if not valid:
        return None, None
    first = valid[0][1]
    if any(not np.array_equal(first, candidate) for _, candidate in valid[1:]):
        return None, None
    return first, valid[0][0]


def _make_pair(rng, marker, period, shared_palette, regime):
    for _ in range(100):
        shape = _sample_shape(rng, period)
        maximum_colors = min(6, period[0] * period[1])
        if len(shared_palette) <= maximum_colors and rng.random() < 0.26:
            palette = shared_palette
        else:
            color_count = int(rng.integers(2, maximum_colors + 1))
            available = [color for color in range(10) if color != marker]
            palette = tuple(int(v) for v in rng.choice(available, color_count, replace=False))
        style = str(rng.choice(("random", "mirror", "bands", "shapes", "separators"),
                               p=(0.25, 0.24, 0.18, 0.17, 0.16)))
        tile = _motif(rng, period, palette, style)
        output = _clean_grid(tile, shape, rng)
        if (_axis_period(output, 0), _axis_period(output, 1)) != period:
            continue
        if set(map(int, np.unique(output))) != set(palette):
            continue
        mask = _damage_mask(rng, shape, period)
        if not np.any(mask):
            continue
        input_grid = output.copy()
        input_grid[mask] = marker
        recovered, inferred = _infer_completion(input_grid, marker)
        if recovered is None or inferred != period or not np.array_equal(recovered, output):
            continue
        return {"input": input_grid, "output": output}
    raise RuntimeError(f"unable to construct {regime} pair with period {period}")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    marker = int(rng.integers(0, 10))
    available = [color for color in range(10) if color != marker]
    shared_palette = tuple(int(v) for v in rng.choice(available, int(rng.integers(2, 6)), replace=False))
    regime = int(rng.integers(0, 4))
    if regime == 0:  # retain the source's larger-test possibility
        train_periods = _period_choices(rng, num_train, 2, 7)
        test_periods = _period_choices(rng, num_test, 8, 10)
    elif regime == 1:  # exact period overlap with different motif contents
        train_periods = _period_choices(rng, num_train, 2, 10)
        test_periods = [train_periods[int(rng.integers(0, num_train))]]
        test_periods += _period_choices(rng, num_test - 1, 2, 10)
    elif regime == 2:  # smaller target motifs
        train_periods = _period_choices(rng, num_train, 6, 10)
        test_periods = _period_choices(rng, num_test, 2, 4)
    else:
        train_periods = _period_choices(rng, num_train, 2, 10)
        test_periods = _period_choices(rng, num_test, 2, 10)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, periods in (("train", train_periods), ("test", test_periods)):
        for period in periods:
            pair_rng = np.random.default_rng(int(pair_seeds[len(seen)]))
            for _ in range(20):
                pair = _make_pair(pair_rng, marker, period, shared_palette,
                                  "source_train" if split == "train" else "target_test")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("duplicate input rejection exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_metadata(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    difference = inp != out
    assert np.any(difference)
    markers = np.unique(inp[difference])
    assert markers.size == 1
    marker = int(markers[0])
    assert not np.any(out == marker)
    assert np.array_equal(difference, inp == marker)
    assert np.array_equal(inp[~difference], out[~difference])
    period = (_axis_period(out, 0), _axis_period(out, 1))
    assert period[0] <= inp.shape[0] // 2 and period[1] <= inp.shape[1] // 2
    for r in range(period[0]):
        for c in range(period[1]):
            phase = out[r::period[0], c::period[1]]
            assert np.all(phase == phase.flat[0])
            visible = inp[r::period[0], c::period[1]]
            assert np.count_nonzero(visible != marker) >= 2
    recovered, inferred = _infer_completion(inp, marker)
    assert recovered is not None and inferred == period
    assert np.array_equal(recovered, out)
    if inp.shape == (25, 25):
        digest = hashlib.sha256(inp.astype(np.uint8).tobytes() + out.astype(np.uint8).tobytes()).hexdigest()
        assert digest not in _SOURCE_HASHES
    return marker, period


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    markers = set()
    seen_inputs, seen_pairs = set(), set()
    periods = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            marker, period = _pair_metadata(pair)
            markers.add(marker)
            periods[split].append(period)
            inp, out = pair["input"], pair["output"]
            input_key = (inp.shape, inp.astype(np.uint8).tobytes())
            pair_key = (input_key, out.astype(np.uint8).tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    assert len(markers) == 1
    assert len(set(periods["train"])) >= 2
    # Every training example visibly demonstrates removal of the same marker;
    # reconstruction of each test then depends only on its own visible phases.
    marker = next(iter(markers))
    for pair in puzzle["test"]:
        predicted, _ = _infer_completion(pair["input"], marker)
        assert predicted is not None and np.array_equal(predicted, pair["output"])


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[role], y[role])
        for x, y in zip(a[s], b[s]) for role in ("input", "output")
    ) for s in ("train", "test"))


def _mask_components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for start_r, start_c in np.argwhere(mask):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        rows, cols = zip(*cells)
        components.append((len(cells), max(rows) - min(rows) + 1,
                           max(cols) - min(cols) + 1))
    return components


def _tile_structure(output, period):
    p, q = period
    tile = output[:p, :q]
    row_mirror = any(np.array_equal(tile, np.roll(tile[::-1], shift, axis=0))
                     for shift in range(p))
    col_mirror = any(np.array_equal(tile, np.roll(tile[:, ::-1], shift, axis=1))
                     for shift in range(q))
    row_lines = {r for r in range(p) if np.all(tile[r] == tile[r, 0])}
    col_lines = {c for c in range(q) if np.all(tile[:, c] == tile[0, c])}
    short_spacing = any(len(range(0, p, step)) >= 2 and
                        all(r in row_lines for r in range(0, p, step))
                        for step in range(2, p)) or any(
                        len(range(0, q, step)) >= 2 and
                        all(c in col_lines for c in range(0, q, step))
                        for step in range(2, q))
    return row_mirror and col_mirror, short_spacing


def self_test() -> None:
    train_counts, test_counts, markers = set(), set(), set()
    periods, shapes, palette_sizes, signatures = set(), set(), set(), set()
    counts = {name: 0 for name in (
        "pairs", "square", "rectangle", "thin", "complete", "partial",
        "original_size", "compact", "interior", "edge", "light_train",
        "light_test", "heavy_train", "heavy_test", "isolated", "streak",
        "multiple", "broad", "no_intact", "mirror", "large_mirror",
        "test_mirror", "separators", "equal_top", "larger", "smaller",
        "equal_period", "familiar_palette", "novel_palette",
        "varying_palette", "varying_palette_size", "varying_mode",
        "compact_two_complete", "large_mirror_light_interior_test",
        "smaller_heavy_test",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_periods = []
        test_periods = []
        train_palettes = []
        test_palettes = []
        modes = set()
        pair_palettes = set()
        pair_palette_sizes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                marker, period = _pair_metadata(pair)
                p, q = period
                h, w = inp.shape
                mask = inp == marker
                fraction = float(np.mean(mask))
                palette = frozenset(map(int, np.unique(out)))
                values, frequencies = np.unique(out, return_counts=True)
                modes.add(int(values[np.argmax(frequencies)]))
                pair_palettes.add(palette)
                pair_palette_sizes.add(len(palette))
                if split == "train":
                    train_periods.append(period)
                    train_palettes.append(palette)
                else:
                    test_periods.append(period)
                    test_palettes.append(palette)
                markers.add(marker)
                periods.add(period)
                shapes.add(inp.shape)
                palette_sizes.add(len(palette))
                signatures.add((inp.shape, inp.tobytes()))
                counts["pairs"] += 1
                counts["square" if p == q else "rectangle"] += 1
                counts["thin"] += period in ((2, 4), (4, 2))
                complete = h % p == 0 and w % q == 0
                counts["complete" if complete else "partial"] += 1
                counts["original_size"] += (h, w) == (25, 25)
                counts["compact"] += min(h, w) <= 12
                interior = not (mask[0].any() or mask[-1].any() or
                                mask[:, 0].any() or mask[:, -1].any())
                counts["interior" if interior else "edge"] += 1
                if fraction < 0.12:
                    counts["light_" + split] += 1
                if fraction >= 0.20:
                    counts["heavy_" + split] += 1
                components = _mask_components(mask)
                counts["isolated"] += any(size == 1 for size, _, _ in components)
                counts["streak"] += any((rh == 1 and cw >= 3) or
                                         (cw == 1 and rh >= 3)
                                         for _, rh, cw in components)
                counts["multiple"] += len(components) >= 2
                counts["broad"] += any(rh >= p and cw >= q for _, rh, cw in components)
                counts["no_intact"] += not any(
                    not mask[r:r + p, c:c + q].any()
                    for r in range(h - p + 1) for c in range(w - q + 1))
                mirror, separator = _tile_structure(out, period)
                counts["mirror"] += mirror
                counts["large_mirror"] += mirror and p >= 5 and q >= 5
                counts["test_mirror"] += mirror and split == "test"
                counts["separators"] += separator
                counts["equal_top"] += np.count_nonzero(frequencies == frequencies.max()) >= 2
                counts["compact_two_complete"] += (len(palette) == 2 and complete
                                                   and min(h, w) <= 12)
                counts["large_mirror_light_interior_test"] += (
                    split == "test" and mirror and p >= 5 and q >= 5
                    and fraction < 0.12 and interior)
                if split == "test":
                    # Prediction receives only the training examples and this
                    # test input; no seed or hidden construction parameters.
                    learned_marker = int(np.unique(puzzle["train"][0]["input"][
                        puzzle["train"][0]["input"] != puzzle["train"][0]["output"]])[0])
                    predicted, _ = _infer_completion(inp, learned_marker)
                    assert predicted is not None and np.array_equal(predicted, out)
        counts["varying_palette"] += len(pair_palettes) > 1
        counts["varying_palette_size"] += len(pair_palette_sizes) > 1
        counts["varying_mode"] += len(modes) > 1
        train_areas = [p * q for p, q in train_periods]
        for pair, period, palette in zip(puzzle["test"], test_periods, test_palettes):
            area = period[0] * period[1]
            counts["larger"] += area > max(train_areas)
            counts["smaller"] += area < min(train_areas)
            counts["equal_period"] += period in train_periods
            counts["familiar_palette" if palette in train_palettes else "novel_palette"] += 1
            counts["smaller_heavy_test"] += (area < min(train_areas) and
                                             np.mean(pair["input"] == marker) >= 0.20)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(markers) >= 6 and len(periods) >= 30 and len(shapes) >= 70
    assert {2, 3, 4, 5, 6}.issubset(palette_sizes)
    assert all(counts[name] > 0 for name in counts if name not in ("pairs",))
    assert counts["mirror"] >= 20 and counts["test_mirror"] >= 5
    assert counts["separators"] >= 10 and counts["interior"] >= 40
    assert counts["varying_palette"] >= 35 and counts["varying_mode"] >= 35
    assert len(signatures) > 0.98 * counts["pairs"]
    for seed, nt, ne in ((7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
                         (8001, None, 1), (8002, None, 4), (8003, 2, None), (8004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ne)
        validate_generated_puzzle(puzzle)
        assert nt is None or len(puzzle["train"]) == nt
        assert ne is None or len(puzzle["test"]) == ne
    a = generate_puzzle(123456, 7, 3)
    b = generate_puzzle(123456, 7, 3)
    assert _equal_puzzles(a, b)
    assert not _equal_puzzles(generate_puzzle(5000, 6, 2), generate_puzzle(5001, 6, 2))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(9000, **args)
        except ValueError:
            continue
        raise AssertionError(f"invalid counts accepted: {args}")
