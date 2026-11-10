import numpy as np


_AUTO_COUNTS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 4),
    (7, 1),
    (8, 2),
    (9, 3),
    (10, 4),
)


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIR_ROWS = (
    (
        (
            "000000000000",
            "355550000000",
            "355550055555",
            "455550055555",
            "455550055555",
            "300000055555",
            "400055555555",
            "300055555555",
            "300055555555",
            "300055555555",
            "400055500000",
            "400055500000",
        ),
        (
            "000000000000",
            "333330000000",
            "333330033333",
            "444440044444",
            "444440044444",
            "300000033333",
            "400044444444",
            "300033333333",
            "300033333333",
            "300033333333",
            "400044400000",
            "400044400000",
        ),
    ),
    (
        (
            "000000000000",
            "105500000000",
            "205500000000",
            "105500000000",
            "105500000000",
            "105500005500",
            "205500005500",
            "205500005500",
            "100000005500",
            "100055505500",
            "100055505500",
            "200055505500",
        ),
        (
            "000000000000",
            "101100000000",
            "202200000000",
            "101100000000",
            "101100000000",
            "101100001100",
            "202200002200",
            "202200002200",
            "100000001100",
            "100011101100",
            "100011101100",
            "200022202200",
        ),
    ),
    (
        (
            "105555550000",
            "805555550000",
            "105555550000",
            "105555550555",
            "705555550555",
            "705555550555",
            "700000555555",
            "700000555000",
            "800000555000",
            "805550555000",
            "805550555000",
            "805550555000",
        ),
        (
            "101111110000",
            "808888880000",
            "101111110000",
            "101111110111",
            "707777770777",
            "707777770777",
            "700000777777",
            "700000777000",
            "800000888000",
            "808880888000",
            "808880888000",
            "808880888000",
        ),
    ),
)

_SOURCE_PAIRS = tuple(
    (_rows_to_array(input_rows), _rows_to_array(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _checked_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed_value


def _checked_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count("num_train", num_train, 2, 10)
    test_count = _checked_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]

    if train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))

    if test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_puzzle_parameters(rng):
    roles = rng.choice(10, size=2, replace=False)
    return {
        "background": int(roles[0]),
        "placeholder": int(roles[1]),
        "vary_placeholder": bool(rng.random() < 0.30),
        "rotation": int(rng.integers(0, 4)),
        "shared_shape": (int(rng.integers(6, 23)), int(rng.integers(7, 27))),
        "reuse_shape": bool(rng.random() < 0.27),
    }


def _sample_dimensions(rng, parameters, pair_index, split):
    if parameters["reuse_shape"] and pair_index == 0:
        return parameters["shared_shape"]
    if rng.random() < 0.10:
        return 12, 12
    return int(rng.integers(6, 23)), int(rng.integers(7, 27))


def _positive_composition(total, parts, rng):
    if parts == 1:
        return np.array([total], dtype=int)
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total])))


def _row_pattern(rng, width, target, max_runs, min_runs=1):
    n = width - 1
    total = max(1, min(n - 1, int(round(n * target + rng.normal(0, 0.7)))))
    feasible = [r for r in range(min_runs, min(max_runs, 5, n) + 1)
                if r <= total and total + r - 1 <= n]
    if not feasible:
        feasible = [1]
    runs = int(rng.choice(feasible))
    lengths = _positive_composition(total, runs, rng)
    free = n - total - (runs - 1)
    gaps = rng.multinomial(free, np.ones(runs + 1) / (runs + 1))
    pattern = np.zeros(n, dtype=bool)
    cursor = int(gaps[0])
    for i, length in enumerate(lengths):
        pattern[cursor:cursor + length] = True
        cursor += int(length)
        if i < runs - 1:
            cursor += 1 + int(gaps[i + 1])
    return pattern


def _shift_pattern(pattern, shift):
    result = np.zeros_like(pattern)
    if shift >= 0:
        result[shift:] = pattern[:len(pattern) - shift]
    else:
        result[:shift] = pattern[-shift:]
    return result if np.any(result) else pattern.copy()


def _make_mask(rng, height, width, target, style, require_two_runs):
    states = rng.choice(3, size=height, p=[0.10, 0.12, 0.78])
    if rng.random() < 0.20:
        states[:] = 2
    if require_two_runs:
        states[:3] = 2
    while np.count_nonzero(states == 2) < 3:
        states[int(rng.integers(0, height))] = 2

    mask = np.zeros((height, width), dtype=bool)
    filled = np.flatnonzero(states == 2)
    if not require_two_runs and rng.random() < 0.16:
        cap = 1
    elif style == "coherent":
        cap = int(rng.integers(2, 4))
    elif style == "stepped":
        cap = int(rng.integers(1, 3))
    else:
        cap = int(rng.integers(2, 6))
    templates = [_row_pattern(rng, width, target, cap,
                              2 if require_two_runs else 1)]
    if style == "alternating":
        templates.extend(_row_pattern(rng, width, target, cap)
                         for _ in range(int(rng.integers(1, 4))))
    previous = None
    band_remaining = 0
    for j, row in enumerate(filled):
        if require_two_runs and j < min(4, len(filled)):
            pattern = templates[0].copy()
        elif style == "coherent":
            if previous is not None and band_remaining > 0:
                pattern = previous.copy()
                band_remaining -= 1
            else:
                pattern = _row_pattern(rng, width, target, cap)
                band_remaining = int(rng.integers(1, 5))
        elif style == "stepped":
            base = templates[0]
            pattern = _shift_pattern(base, int(np.sin(j / 2) * min(3, width // 5)))
            if rng.random() < 0.18:
                pattern = _row_pattern(rng, width, target, cap)
        elif style == "alternating":
            pattern = templates[j % len(templates)].copy()
        else:
            pattern = _row_pattern(rng, width, target, cap)
        mask[row, 1:] = pattern
        previous = pattern
    return mask, states


def _row_labels(rng, filled, key_rows, palette):
    mode = int(rng.integers(0, 4))
    count = len(palette)
    indices = np.zeros(len(filled), dtype=int)
    if mode == 0:
        indices = np.minimum(count - 1, np.arange(len(filled)) * count // len(filled))
    elif mode == 1:
        indices = np.arange(len(filled)) % count
    elif mode == 2:
        indices = rng.integers(0, count, size=len(filled))
    else:
        indices = rng.choice(count, size=len(filled), p=np.array([0.58] +
            [0.42 / (count - 1)] * (count - 1)) if count > 1 else [1.0])
    if mode in (2, 3):
        for i in range(count):
            indices[i] = i
        rng.shuffle(indices)
    labels = np.full(int(max(key_rows)) + 1, -1, dtype=int)
    for row, index in zip(filled, indices):
        labels[row] = int(palette[index])
    for row in key_rows:
        if labels[row] < 0:
            labels[row] = int(rng.choice(palette))
    return labels


def _generate_pair(rng, parameters, split, pair_index, target, placeholder):
    height, width = _sample_dimensions(rng, parameters, pair_index, split)
    style = "coherent" if split == "train" and pair_index == 0 else str(
        rng.choice(["coherent", "stepped", "alternating", "irregular"]))
    require_two_runs = split == "train" and pair_index == 0
    mask, states = _make_mask(rng, height, width, target, style, require_two_runs)
    filled = np.flatnonzero(states == 2)
    key_rows = np.flatnonzero(states != 0)
    color_count = int(rng.integers(2 if require_two_runs else 1,
                                   min(5, len(filled)) + 1))
    available = [c for c in range(10) if c not in
                 (parameters["background"], placeholder)]
    palette = rng.choice(available, size=color_count, replace=False)
    labels = _row_labels(rng, filled, key_rows, palette)
    inp = np.full((height, width), parameters["background"], dtype=np.int8)
    inp[mask] = placeholder
    for row in key_rows:
        inp[row, 0] = labels[row]
    out = inp.copy()
    for row in filled:
        out[row, mask[row]] = labels[row]
    return np.rot90(inp, parameters["rotation"]).copy(), np.rot90(
        out, parameters["rotation"]).copy()


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _matches_source_pair(inp, out):
    return any(inp.shape == a.shape and np.array_equal(inp, a) and
               np.array_equal(out, b) for a, b in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _checked_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(rng, seed_value, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    broad_first = bool(rng.integers(0, 2))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_placeholder = parameters["placeholder"]
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if not parameters["vary_placeholder"] or (split == "train" and index == 0):
                placeholder = first_placeholder
            elif split == "train" and index == 1:
                placeholder = int(rng.choice([c for c in range(10) if c not in
                                              (parameters["background"], first_placeholder)]))
            else:
                placeholder = int(rng.choice([c for c in range(10) if c !=
                                              parameters["background"]]))
            if split == "train" and index < 2:
                broad = broad_first if index == 0 else not broad_first
                target = float(rng.uniform(0.58, 0.80) if broad else rng.uniform(0.22, 0.40))
            else:
                target = float(rng.uniform(0.19, 0.82))
            for _ in range(50):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = _generate_pair(pair_rng, parameters, split, index,
                                          target, placeholder)
                key = _grid_key(inp)
                if key in seen or _matches_source_pair(inp, out):
                    continue
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _count_runs(bits):
    return int(np.count_nonzero(bits & np.concatenate(([True], ~bits[:-1]))))


def _inspect_pair(inp, out):
    assert inp.shape == out.shape
    changed = inp != out
    assert np.any(changed)
    values = np.unique(inp[changed])
    assert len(values) == 1
    placeholder = int(values[0])
    assert np.array_equal(changed, inp == placeholder)
    candidates = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        b = np.rot90(out, -rotation)
        if np.any(a[:, 0] == placeholder):
            continue
        other = a[:, 1:]
        backgrounds = np.unique(other[other != placeholder])
        if len(backgrounds) != 1:
            continue
        bg = int(backgrounds[0])
        if bg == placeholder:
            continue
        rows = np.any(a[:, 1:] == placeholder, axis=1)
        if np.count_nonzero(rows) < 3 or np.any(a[rows, 0] == bg):
            continue
        if np.any((a[~rows, 0] != bg) & ~np.isin(a[~rows, 0], a[rows, 0])):
            continue
        expected = a.copy()
        for row in np.flatnonzero(rows):
            expected[row, a[row] == placeholder] = a[row, 0]
        if not np.array_equal(expected, b):
            continue
        labels = frozenset(int(c) for c in a[rows, 0])
        runs = [_count_runs(a[row, 1:] == placeholder) for row in np.flatnonzero(rows)]
        repeated = any(np.array_equal(a[r, 1:] == placeholder,
                                     a[r + 1, 1:] == placeholder)
                       for r in range(len(rows) - 1) if rows[r] and rows[r + 1])
        density = float(np.count_nonzero(a == placeholder) /
                        (np.count_nonzero(rows) * (a.shape[1] - 1)))
        candidates.append({"rotation": rotation, "background": bg,
                           "placeholder": placeholder, "labels": labels,
                           "runs": runs, "repeated": repeated,
                           "density": density, "filled": int(np.count_nonzero(rows)),
                           "blank": int(np.count_nonzero(np.all(a == bg, axis=1))),
                           "key_only": int(np.count_nonzero(~rows & (a[:, 0] != bg))),
                           "shape": a.shape})
    assert len(candidates) == 1
    return candidates[0]


def _predict(train, test_input):
    analyses = [_inspect_pair(p["input"], p["output"]) for p in train]
    rotation = analyses[0]["rotation"]
    background = analyses[0]["background"]
    a = np.rot90(test_input, -rotation).copy()
    non_background = np.unique(a[:, 1:][a[:, 1:] != background])
    assert len(non_background) == 1
    placeholder = int(non_background[0])
    for row in range(a.shape[0]):
        bits = a[row] == placeholder
        if np.any(bits):
            assert a[row, 0] not in (background, placeholder)
            a[row, bits] = a[row, 0]
    return np.rot90(a, rotation).copy()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and not _matches_source_pair(inp, out)
            ik, pk = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            analyses[split].append(_inspect_pair(inp, out))
    all_a = analyses["train"] + analyses["test"]
    for role in ("rotation", "background"):
        assert len({a[role] for a in all_a}) == 1
    trained_placeholders = {a["placeholder"] for a in analyses["train"]}
    if len(trained_placeholders) == 1:
        assert all(a["placeholder"] in trained_placeholders for a in analyses["test"])
    assert any(len(a["labels"]) >= 2 for a in analyses["train"])
    assert any(max(a["runs"]) >= 2 for a in analyses["train"])
    assert any(a["repeated"] for a in analyses["train"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _require_value_error(**kwargs):
    try:
        generate_puzzle(987654321, **kwargs)
    except ValueError:
        return
    raise AssertionError("invalid count accepted")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert not _puzzles_equal(puzzles[0], puzzles[1])
    roles, orientations, shapes, masks, label_counts = set(), set(), set(), set(), set()
    densities = {"train": [], "test": []}
    blank_pairs = key_only_pairs = one_run_pairs = rectangle_pairs = 0
    interior_blank_pairs = three_aligned_pairs = multi_run_pairs = 0
    stepped_pairs = label_band_pairs = alternating_label_pairs = rare_label_pairs = 0
    same_shape_puzzles = same_label_count_puzzles = 0
    reversed_density = overlapping_density = varying_placeholder_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        a_train = [_inspect_pair(p["input"], p["output"]) for p in puzzle["train"]]
        a_test = [_inspect_pair(p["input"], p["output"]) for p in puzzle["test"]]
        if len({a["placeholder"] for a in a_train}) > 1:
            varying_placeholder_puzzles += 1
        if puzzle["train"][0]["input"].shape == puzzle["test"][0]["input"].shape:
            same_shape_puzzles += 1
        if any(len(a["labels"]) == len(b["labels"]) for a in a_train for b in a_test):
            same_label_count_puzzles += 1
        if max(a["density"] for a in a_train) > min(a["density"] for a in a_test):
            reversed_density += 1
        if min(a["density"] for a in a_train) <= min(a["density"] for a in a_test) <= max(a["density"] for a in a_train):
            overlapping_density += 1
        for split, analyses in (("train", a_train), ("test", a_test)):
            for pair, a in zip(puzzle[split], analyses):
                roles.add((a["background"], a["placeholder"]))
                orientations.add(a["rotation"])
                shapes.add(pair["input"].shape)
                masks.add((pair["input"].shape, (pair["input"] == a["placeholder"]).tobytes()))
                label_counts.add(len(a["labels"]))
                densities[split].append(a["density"])
                blank_pairs += a["blank"] > 0
                key_only_pairs += a["key_only"] > 0
                one_run_pairs += max(a["runs"]) == 1
                rectangle_pairs += a["repeated"]
                multi_run_pairs += max(a["runs"]) >= 3
                canonical = np.rot90(pair["input"], -a["rotation"])
                mask = canonical[:, 1:] == a["placeholder"]
                interior_blank_pairs += bool(np.any(np.all(
                    canonical[1:-1] == a["background"], axis=1)))
                longest_aligned = 1
                current_aligned = 1
                stepped = False
                for row in range(1, len(mask)):
                    if np.any(mask[row]) and np.array_equal(mask[row], mask[row - 1]):
                        current_aligned += 1
                    else:
                        current_aligned = 1
                    longest_aligned = max(longest_aligned, current_aligned)
                    if np.any(mask[row]) and np.any(mask[row - 1]) and not np.array_equal(mask[row], mask[row - 1]):
                        stepped |= (np.array_equal(mask[row], _shift_pattern(mask[row - 1], 1))
                                    or np.array_equal(mask[row], _shift_pattern(mask[row - 1], -1)))
                three_aligned_pairs += longest_aligned >= 3
                stepped_pairs += stepped
                filled_rows = np.flatnonzero(np.any(mask, axis=1))
                keys = canonical[filled_rows, 0]
                label_band_pairs += any(np.all(keys[i:i + 3] == keys[i])
                                        for i in range(len(keys) - 2))
                alternating_label_pairs += any(keys[i] == keys[i + 2]
                    and keys[i + 1] == keys[i + 3] and keys[i] != keys[i + 1]
                    for i in range(len(keys) - 3))
                if len(a["labels"]) >= 3:
                    counts = [np.count_nonzero(keys == color) for color in a["labels"]]
                    rare_label_pairs += min(counts) == 1 and max(counts) >= 3
    assert len(roles) >= 18 and len(orientations) == 4
    assert varying_placeholder_puzzles >= 7
    assert same_shape_puzzles >= 3 and same_label_count_puzzles >= 20
    assert len(shapes) >= 45 and len(masks) >= 170
    assert len(label_counts) >= 4
    assert min(densities["train"]) < 0.40 and max(densities["train"]) > 0.55
    assert min(densities["test"]) < 0.45 and max(densities["test"]) > 0.55
    assert reversed_density >= 15 and overlapping_density >= 8
    assert blank_pairs >= 25 and key_only_pairs >= 25
    assert one_run_pairs >= 3 and rectangle_pairs >= 30
    assert interior_blank_pairs >= 40 and three_aligned_pairs >= 30
    assert multi_run_pairs >= 30 and stepped_pairs >= 20
    assert label_band_pairs >= 30 and alternating_label_pairs >= 15
    assert rare_label_pairs >= 10
    assert any(p["input"].shape == (12, 12) for puzzle in puzzles
               for split in ("train", "test") for p in puzzle[split])
    for seed, train, test in ((1001, 3, 1), (1002, 6, 2), (1003, 9, 4),
                              (2001, 2, 1), (2002, 10, 1), (2003, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for seed, train, test in ((3001, None, 1), (3002, None, 4),
                              (3003, 2, None), (3004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert (train is None or len(puzzle["train"]) == train)
        assert (test is None or len(puzzle["test"]) == test)
    assert _puzzles_equal(generate_puzzle(314159, 8, 3), generate_puzzle(314159, 8, 3))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        _require_value_error(**args)
