"""Sparse clues to a three-color, period-three diagonal lattice."""

import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (("0000000", "0000000", "0000001", "0000012", "0000124", "0001240", "0012400"),
     ("2412412", "4124124", "1241241", "2412412", "4124124", "1241241", "2412412")),
    (("2830000", "8300000", "3000000", "0000000", "0000000", "0000000", "0000000"),
     ("2832832", "8328328", "3283283", "2832832", "8328328", "3283283", "2832832")),
    (("0000830", "0008300", "0083000", "0830004", "8300040", "3000400", "0004000"),
     ("4834834", "8348348", "3483483", "4834834", "8348348", "3483483", "4834834")),
    (("0100002", "1000020", "0000200", "0002000", "0020000", "0200004", "2000040"),
     ("2142142", "1421421", "4214214", "2142142", "1421421", "4214214", "2142142")),
}


def _array_key(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _is_source_pair(inp, out):
    def lines(grid):
        return tuple("".join(str(int(v)) for v in row) for row in grid)
    return (lines(inp), lines(out)) in _SOURCE_PAIR_SIGNATURES


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    key = int(seed)
    if num_train is None and num_test is None:
        num_test = 1 + key % 4
        num_train = num_test + 1 + (key // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + key % (10 - num_test)
    elif num_test is None:
        num_test = 1 + key % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _output(height, width, colors, slope):
    rows, cols = np.indices((height, width))
    return np.asarray(colors, dtype=np.int8)[(rows + slope * cols) % 3]


def _diagonal(height, width, index, slope):
    first = max(0, index - width + 1)
    last = min(height - 1, index)
    if slope == 1:
        return [(r, index - r) for r in range(first, last + 1)]
    return [(r, width - 1 - (index - r)) for r in range(first, last + 1)]


def _dimensions(rng):
    mode = float(rng.random())
    if mode < 0.18:
        return int(rng.integers(3, 8)), int(rng.integers(3, 8))
    if mode < 0.40:
        long = int(rng.integers(19, 31))
        short = int(rng.integers(3, 9))
        return (long, short) if rng.random() < 0.5 else (short, long)
    if mode < 0.70:
        return int(rng.integers(6, 19)), int(rng.integers(6, 19))
    return int(rng.integers(8, 31)), int(rng.integers(8, 31))


def _base_diagonals(rng, height, width, layout, no_full):
    maximum = height + width - 2
    eligible = [d for d in range(maximum + 1)
                if not no_full or len(_diagonal(height, width, d, 1)) >= 2]
    placement = int(rng.integers(0, 4))  # low, high, internal, unrestricted
    candidates = []
    for ai, a in enumerate(eligible):
        for bi in range(ai + 1, len(eligible)):
            b = eligible[bi]
            if layout == "adjacent" and b != a + 1:
                continue
            if layout == "separated" and b < a + 2:
                continue
            for c in eligible[bi + 1:]:
                if layout == "adjacent" and c != b + 1:
                    continue
                if layout == "separated" and c < b + 2:
                    continue
                if len({a % 3, b % 3, c % 3}) != 3:
                    continue
                center = (a + b + c) / 3
                keep = (placement == 3
                        or placement == 0 and center <= maximum * 0.43
                        or placement == 1 and center >= maximum * 0.57
                        or placement == 2 and a >= maximum * 0.12 and c <= maximum * 0.88)
                if keep:
                    candidates.append((a, b, c))
    if not candidates:
        return _base_diagonals_unrestricted(rng, eligible, layout)
    return candidates[int(rng.integers(len(candidates)))]


def _base_diagonals_unrestricted(rng, eligible, layout):
    candidates = [(a, b, c)
                  for i, a in enumerate(eligible)
                  for j, b in enumerate(eligible[i + 1:], i + 1)
                  for c in eligible[j + 1:]
                  if len({a % 3, b % 3, c % 3}) == 3
                  and ((b == a + 1 and c == b + 1) if layout == "adjacent"
                       else (b >= a + 2 and c >= b + 2))]
    assert candidates
    return candidates[int(rng.integers(len(candidates)))]


def _mask_line(rng, coordinates, mode, no_full):
    length = len(coordinates)
    upper = length - 1 if no_full else length
    if upper <= 1:
        return [coordinates[int(rng.integers(length))]]
    if mode == "full" and not no_full:
        return coordinates
    if mode == "long":
        lower = min(upper, max(2, length // 2))
        count = int(rng.integers(lower, upper + 1))
        start = int(rng.integers(length - count + 1))
        return coordinates[start:start + count]
    if mode == "short":
        count = int(rng.integers(1, min(upper, 4) + 1))
        start = int(rng.integers(length - count + 1))
        return coordinates[start:start + count]
    if mode == "fragments" and length >= 5 and upper >= 2:
        left = int(rng.integers(0, max(1, length // 3)))
        right = int(rng.integers(max(left + 2, length // 2), length))
        chosen = [coordinates[left], coordinates[right]]
        if upper > 2 and rng.random() < 0.5 and right + 1 < length:
            chosen.append(coordinates[right + 1])
        return chosen
    count = int(rng.integers(1, upper + 1))
    indices = rng.choice(length, size=count, replace=False)
    return [coordinates[int(i)] for i in indices]


def _line_count(rng, maximum):
    total = maximum + 1
    mode = float(rng.random())
    if mode < 0.27:
        return 3
    if mode < 0.66:
        return int(rng.integers(4, min(9, total) + 1)) if total >= 4 else 3
    low = min(total, max(6, int(total * 0.3)))
    high = min(total, max(low, int(total * 0.8)))
    return int(rng.integers(low, high + 1))


def _make_mask(rng, height, width, slope, layout, difficulty):
    maximum = height + width - 2
    no_full = difficulty == "broken" and min(height, width) >= 5
    base = list(_base_diagonals(rng, height, width, layout, no_full))
    eligible = [d for d in range(maximum + 1)
                if not no_full or len(_diagonal(height, width, d, slope)) >= 2]
    remaining = [d for d in eligible if d not in base]
    extra = min(len(remaining), max(0, _line_count(rng, maximum) - 3))
    rng.shuffle(remaining)
    for d in remaining:
        if extra <= 0:
            break
        if layout == "separated" and any(abs(d - chosen) == 1 for chosen in base):
            continue
        base.append(int(d))
        extra -= 1
    mask = np.zeros((height, width), dtype=bool)
    mandatory = np.zeros_like(mask)
    witness_options = [d for d in base if len(_diagonal(height, width, d, slope)) >= (3 if no_full else 2)]
    witness = int(rng.choice(witness_options))
    witness_coords = _diagonal(height, width, witness, slope)
    start = int(rng.integers(len(witness_coords) - 1))
    for r, c in witness_coords[start:start + 2]:
        mandatory[r, c] = True
    for residue in range(3):
        if residue == witness % 3:
            continue
        d = next(d for d in base if d % 3 == residue)
        coords = _diagonal(height, width, d, slope)
        r, c = coords[int(rng.integers(len(coords)))]
        mandatory[r, c] = True
    mask |= mandatory
    line_modes = ("full", "long", "short", "fragments", "scattered")
    proposals = []
    for d in base:
        coords = _diagonal(height, width, d, slope)
        if difficulty == "broken":
            choices = line_modes[1:]
        elif difficulty == "clear":
            choices = line_modes[:3]
        else:
            choices = line_modes
        mode = choices[int(rng.integers(len(choices)))]
        proposals.extend(_mask_line(rng, coords, mode, no_full))
    # Background remains the unique majority, so it is identifiable from input.
    cap = (height * width - 1) // 2
    rng.shuffle(proposals)
    for r, c in proposals:
        if int(mask.sum()) >= cap:
            break
        if no_full:
            d = r + c if slope == 1 else r + width - 1 - c
            coords = _diagonal(height, width, d, slope)
            if sum(mask[y, x] for y, x in coords) >= len(coords) - 1:
                continue
        mask[r, c] = True
    return mask


def _simple_mask(rng, height, width, slope, layout):
    eligible = [d for d in range(height + width - 1)
                if len(_diagonal(height, width, d, slope)) >= 3]
    diagonals = _base_diagonals_unrestricted(rng, eligible, layout)
    mask = np.zeros((height, width), dtype=bool)
    for d in diagonals:
        coords = _diagonal(height, width, d, slope)
        count = int(rng.integers(2, min(6, len(coords)) + 1))
        start = int(rng.integers(len(coords) - count + 1))
        for r, c in coords[start:start + count]:
            mask[r, c] = True
    return mask


def _candidate_outputs(inp):
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 4:
        return []
    order = np.argsort(counts)
    if counts[order[-1]] <= inp.size // 2:
        return []
    background = int(values[order[-1]])
    clues = inp != background
    rows, cols = np.indices(inp.shape)
    answers = []
    for slope in (1, -1):
        phases = (rows + slope * cols) % 3
        colors = []
        for residue in range(3):
            local = np.unique(inp[clues & (phases == residue)])
            if len(local) != 1:
                break
            colors.append(int(local[0]))
        if len(colors) == 3 and len(set(colors)) == 3:
            answers.append((slope, np.asarray(colors, dtype=np.int8)[phases]))
    return answers


def _pair_details(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    answers = _candidate_outputs(inp)
    assert answers
    assert all(np.array_equal(answer, answers[0][1]) for _, answer in answers)
    assert np.array_equal(out, answers[0][1])
    assert len(np.unique(out)) == 3
    assert np.any(inp != out)
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    clues = inp != background
    assert np.all(inp[clues] == out[clues])
    rows, cols = np.indices(inp.shape)
    phases = (rows + answers[0][0] * cols) % 3
    assert all(len(np.unique(out[phases == residue])) == 1 for residue in range(3))
    return answers[0][0]


def _choose_colors(rng, background, earlier_palettes, earlier_orders):
    compatible = [i for i, palette in enumerate(earlier_palettes) if background not in palette]
    if compatible and rng.random() < 0.43:
        prior = int(rng.choice(compatible))
        palette = earlier_palettes[prior]
        if rng.random() < 0.48:
            return palette, earlier_orders[prior]
    else:
        palette = tuple(sorted(int(x) for x in rng.choice(
            [x for x in range(10) if x != background], size=3, replace=False)))
    colors = tuple(int(x) for x in rng.permutation(palette))
    return palette, colors


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # These are the only puzzle-level regimes. Phase colors and background are local.
    # Stratify the finite direction regimes across consecutive seeds.
    slope_regime = ("plus", "minus", "mixed")[int(seed) % 3]
    layout_regime = "source_shift" if int(seed) % 4 == 0 else "mixed"
    simple_regime = int(seed) % 12 in (0, 1)
    background_regime = "fixed" if simple_regime or rng.random() < 0.34 else "variable"
    fixed_background = int(rng.integers(10))
    total = num_train + num_test
    child_seeds = rng.integers(0, np.iinfo(np.int64).max, size=total, dtype=np.int64)
    if layout_regime == "source_shift":
        train_layouts = ["adjacent"] * num_train
        test_layouts = ["separated"] * num_test
    else:
        train_layouts = ["adjacent", "separated"] + [
            ("adjacent" if rng.random() < 0.5 else "separated") for _ in range(num_train - 2)]
        rng.shuffle(train_layouts)
        test_layouts = [("adjacent" if rng.random() < 0.5 else "separated")
                        for _ in range(num_test)]
    if slope_regime == "mixed":
        train_slopes = [1, -1] + [1 if rng.random() < 0.5 else -1
                                 for _ in range(num_train - 2)]
        rng.shuffle(train_slopes)
        test_slopes = [1 if rng.random() < 0.5 else -1 for _ in range(num_test)]
    else:
        slope = 1 if slope_regime == "plus" else -1
        train_slopes = [slope] * num_train
        test_slopes = [slope] * num_test
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    palettes, orders = [], []
    backgrounds = []
    for index in range(total):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        slope = train_slopes[local_index] if split == "train" else test_slopes[local_index]
        layout = train_layouts[local_index] if split == "train" else test_layouts[local_index]
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        for attempt in range(12):
            if background_regime == "fixed":
                background = fixed_background
            elif index == 1:
                background = int(pair_rng.choice([c for c in range(10) if c != backgrounds[0]]))
            else:
                background = int(pair_rng.integers(10))
            palette, colors = _choose_colors(pair_rng, background, palettes, orders)
            if simple_regime:
                height = int(pair_rng.integers(7, 16))
                width = int(pair_rng.integers(7, 16))
            else:
                height, width = _dimensions(pair_rng)
            out = _output(height, width, colors, slope)
            if simple_regime:
                mask = _simple_mask(pair_rng, height, width, slope, layout)
            else:
                difficulty = ("clear", "varied", "broken")[int(pair_rng.integers(3))]
                mask = _make_mask(pair_rng, height, width, slope, layout, difficulty)
            inp = np.full((height, width), background, dtype=np.int8)
            inp[mask] = out[mask]
            key = _array_key(inp)
            if key in used_inputs or _is_source_pair(inp, out):
                continue
            # The orientation witness and three phase clues must make this unique.
            answers = _candidate_outputs(inp)
            if not answers or any(not np.array_equal(a, out) for _, a in answers):
                continue
            used_inputs.add(key)
            palettes.append(palette)
            orders.append(colors)
            backgrounds.append(background)
            puzzle[split].append({"input": inp, "output": out})
            break
        else:
            raise ValueError(f"could not construct pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
        assert 2 <= len(puzzle["train"]) <= 10
        assert 1 <= len(puzzle["test"]) <= 4
        assert len(puzzle["train"]) > len(puzzle["test"])
        seen_inputs, seen_pairs = set(), set()
        slopes = {"train": set(), "test": set()}
        adjacency = {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in (pair["input"], pair["output"]):
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= n <= 30 for n in grid.shape)
                    assert bool(np.all((0 <= grid) & (grid <= 9)))
                inp_key = _array_key(pair["input"])
                pair_key = inp_key, _array_key(pair["output"])
                assert inp_key not in seen_inputs and pair_key not in seen_pairs
                seen_inputs.add(inp_key)
                seen_pairs.add(pair_key)
                assert not _is_source_pair(pair["input"], pair["output"])
                slope = _pair_details(pair)
                slopes[split].add(slope)
                inp = pair["input"]
                colors, counts = np.unique(inp, return_counts=True)
                background = int(colors[np.argmax(counts)])
                width = inp.shape[1]
                diagonals = sorted({int(r + c if slope == 1 else r + width - 1 - c)
                                    for r, c in np.argwhere(inp != background)})
                adjacency[split].add(any(b == a + 1 for a, b in
                                         zip(diagonals, diagonals[1:])))
        # A direction used by a query is demonstrated by a training pair.
        assert slopes["test"] <= slopes["train"]
        assert (adjacency["train"] == {True, False}
                or adjacency["train"] == {True} and adjacency["test"] == {False})
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed or invalid puzzle") from exc


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _observed_features(pair):
    inp, out = pair["input"], pair["output"]
    slope = _pair_details(pair)
    background = int(np.unique(inp, return_counts=True)[0][
        np.argmax(np.unique(inp, return_counts=True)[1])])
    positions = np.argwhere(inp != background)
    h, w = inp.shape
    indices = sorted({int(r + c if slope == 1 else r + w - 1 - c) for r, c in positions})
    line_counts = []
    line_lengths = []
    fragmented = False
    for d in indices:
        coords = _diagonal(h, w, d, slope)
        selected = [i for i, (r, c) in enumerate(coords) if inp[r, c] != background]
        line_counts.append(len(selected))
        line_lengths.append(len(coords))
        fragmented |= any(b > a + 1 for a, b in zip(selected, selected[1:]))
    return {
        "slope": slope,
        "shape": (h, w),
        "background": background,
        "palette": frozenset(int(x) for x in np.unique(out)),
        "phase": tuple(int(out[r, 0]) for r in range(3)),
        "lines": len(indices),
        "min_line_clues": min(line_counts),
        "repeated": len(indices) > 3,
        "fragmented": fragmented,
        "full": any(a == b for a, b in zip(line_counts, line_lengths)),
        "adjacent": any(b == a + 1 for a, b in zip(indices, indices[1:])),
        "density": len(positions) / inp.size,
        "internal": all(0 < r < h - 1 and 0 < c < w - 1 for r, c in positions),
    }


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIR_SIGNATURES:
        inp = np.asarray([[int(v) for v in row] for row in source_input], dtype=np.int8)
        expected = np.asarray([[int(v) for v in row] for row in source_output], dtype=np.int8)
        answers = _candidate_outputs(inp)
        assert answers and all(np.array_equal(answer, expected) for _, answer in answers)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    features = {"train": [], "test": []}
    variable_backgrounds = 0
    mixed_slopes = 0
    repeated_palettes = 0
    test_fresh_palettes = 0
    source_shifts = 0
    mixed_layouts = 0
    simple_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        observed = {s: [_observed_features(p) for p in puzzle[s]]
                    for s in ("train", "test")}
        for split in ("train", "test"):
            features[split].extend(observed[split])
        assert {f["slope"] for f in observed["test"]} <= {
            f["slope"] for f in observed["train"]}
        train_layouts = {f["adjacent"] for f in observed["train"]}
        test_layouts = {f["adjacent"] for f in observed["test"]}
        source_shifts += train_layouts == {True} and test_layouts == {False}
        mixed_layouts += train_layouts == {True, False}
        variable_backgrounds += len({f["background"] for f in
                                     observed["train"] + observed["test"]}) > 1
        mixed_slopes += len({f["slope"] for f in observed["train"]}) > 1
        all_observed = observed["train"] + observed["test"]
        simple_puzzles += (len({f["background"] for f in all_observed}) == 1
                           and len({f["slope"] for f in all_observed}) == 1
                           and all(f["lines"] == 3 and f["min_line_clues"] >= 2
                                   and not f["fragmented"] for f in all_observed))
        train_palettes = [f["palette"] for f in observed["train"]]
        repeated_palettes += len(set(train_palettes)) < len(train_palettes)
        test_fresh_palettes += any(f["palette"] not in train_palettes
                                   for f in observed["test"])
    all_features = features["train"] + features["test"]
    assert len({f["shape"] for f in all_features}) >= 60
    assert len({f["palette"] for f in all_features}) >= 35
    assert len({f["background"] for f in all_features}) == 10
    assert any(f["shape"][0] >= 25 and f["shape"][1] <= 8 for f in all_features)
    assert any(f["shape"][1] >= 25 and f["shape"][0] <= 8 for f in all_features)
    assert any(max(f["shape"]) <= 7 for f in all_features)
    assert any(f["lines"] >= 7 for f in all_features)
    assert any(f["lines"] == 3 for f in all_features)
    assert any(f["fragmented"] for f in features["train"])
    assert any(f["fragmented"] for f in features["test"])
    assert any(not f["full"] for f in features["test"])
    assert any(f["full"] for f in features["test"])
    assert {f["adjacent"] for f in features["test"]} == {True, False}
    assert any(f["density"] < 0.10 for f in all_features)
    assert any(f["density"] > 0.30 for f in all_features)
    assert variable_backgrounds >= 8 and mixed_slopes >= 6
    assert repeated_palettes >= 5 and test_fresh_palettes >= 5
    assert source_shifts >= 8 and mixed_layouts >= 20
    assert simple_puzzles >= 5
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000 + train_count + test_count, train_count, test_count)
        assert len(p["train"]) == train_count and len(p["test"]) == test_count
    for request in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9012, *request)
        assert request[0] is None or len(p["train"]) == request[0]
        assert request[1] is None or len(p["test"]) == request[1]
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4)):
        try:
            generate_puzzle(7, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")
    assert _puzzle_key(generate_puzzle(2718, 8, 3)) == _puzzle_key(
        generate_puzzle(2718, 8, 3))
