"""Compositional examples of connecting orthogonally aligned point markers."""

import numpy as np


_SOURCE_INPUT_ROWS = {
    ("000000", "008000", "000000", "000000", "000000", "000000", "000000", "008000", "000000"),
    ("000", "080", "000"),
    ("000000", "000800", "000000", "080000", "000000", "000000", "000800"),
    ("00000000000", "00080000000", "00000000000", "00000000000", "00000080008", "08000000000", "00000000000", "00000000000", "00000000000", "00080000000", "00000000000", "00000000000"),
    ("00000000000", "08000000080", "00000000000", "00000000000", "00000000000", "00000000000", "00000000000", "00000000000", "00800008000", "00000000000", "00000000000", "00000000000"),
    ("000000", "080000", "000000", "000080", "000000"),
    ("0000000000000", "0000000000000", "0000000000000", "0080000008000", "0000000000000", "0000000000000", "0000000000000"),
    ("00000000000", "00008000000", "00000000800", "00000000000", "00000000000", "00000000000", "00000000800", "00008000000", "00000000000", "00000000000"),
    ("0000000000000", "0080000000800", "0000000000000", "0000000000000", "0000000000000", "0000008000008", "0000000000000", "0800000000000", "0000000000000", "0000000080000", "0800000000000", "0000000000000"),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _source_input(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid) in _SOURCE_INPUT_ROWS


def _signature(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _roles(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    order = np.argsort(counts)
    assert counts[order[0]] < counts[order[1]]
    return int(colors[order[1]]), int(colors[order[0]])


def _apply_rule(grid, marker, connector):
    out = grid.copy()
    for row in range(grid.shape[0]):
        columns = np.flatnonzero(grid[row] == marker)
        for left, right in zip(columns, columns[1:]):
            out[row, left + 1:right] = connector
    for column in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, column] == marker)
        for top, bottom in zip(rows, rows[1:]):
            out[top + 1:bottom, column] = connector
    return out


def _features(grid, marker):
    positions = {(int(r), int(c)) for r, c in np.argwhere(grid == marker)}
    rows = {r: sorted(c for rr, c in positions if rr == r) for r, _ in positions}
    cols = {c: sorted(r for r, cc in positions if cc == c) for _, c in positions}
    horizontal = [(r, values) for r, values in rows.items() if len(values) >= 2]
    vertical = [(c, values) for c, values in cols.items() if len(values) >= 2]
    hpoints = {(r, c) for r, values in horizontal for c in values}
    vpoints = {(r, c) for c, values in vertical for r in values}
    hmask = np.zeros(grid.shape, dtype=bool)
    vmask = np.zeros(grid.shape, dtype=bool)
    gaps = []
    for r, values in horizontal:
        for left, right in zip(values, values[1:]):
            hmask[r, left + 1:right] = True
            gaps.append(right - left - 1)
    for c, values in vertical:
        for top, bottom in zip(values, values[1:]):
            vmask[top + 1:bottom, c] = True
            gaps.append(bottom - top - 1)
    return {
        "markers": len(positions),
        "horizontal": len(horizontal),
        "vertical": len(vertical),
        "segments": sum(len(v) - 1 for _, v in horizontal + vertical),
        "unmatched": len(positions - hpoints - vpoints),
        "triple_h": any(len(v) >= 3 for _, v in horizontal),
        "triple_v": any(len(v) >= 3 for _, v in vertical),
        "dual": bool(hpoints & vpoints),
        "cross": bool(np.any(hmask & vmask & (grid != marker))),
        "painted": int(np.count_nonzero((hmask | vmask) & (grid != marker))),
        "adjacent": any(gap == 0 for gap in gaps),
        "max_gap": max(gaps, default=0),
    }


def _draw_dimension(rng, minimum, is_test, maximum):
    if minimum > maximum:
        raise ValueError("scene exceeds its size regime")
    choice = int(rng.integers(0, 10))
    if choice < (2 if is_test else 4):
        high = min(maximum, max(minimum, 8))
    elif choice < 7:
        high = min(maximum, max(minimum, 16))
    elif choice < 9:
        high = maximum
    else:
        return maximum
    return int(rng.integers(minimum, high + 1))


def _random_lines(rng, minimum, maximum):
    count = int(rng.integers(minimum, maximum + 1))
    return [("h" if rng.random() < 0.5 else "v", 2) for _ in range(count)]


def _training_recipe(rng, index, capability, required_dots):
    if index == 0:
        return [("h", 2)] + [("dot", 1)] * required_dots[0]
    if index == 1:
        return [("v", 2)] + [("dot", 1)] * required_dots[1]
    if index == 2 and capability != "basic":
        core = {
            "triple_h": [("h", int(rng.integers(3, 5)))],
            "triple_v": [("v", int(rng.integers(3, 5)))],
            "dual": [("dual", 1)],
            "cross": [("cross", 1)],
        }[capability]
        return core + [("dot", 1)] * int(rng.integers(0, 3))
    mode = int(rng.integers(0, 8))
    if mode == 0:
        lines = []
    elif mode == 1:
        lines = [("h", 2)]
    elif mode == 2:
        lines = [("v", 2)]
    elif mode == 3:
        lines = [("h", 2), ("h", 2)]
    elif mode == 4:
        lines = [("v", 2), ("v", 2)]
    elif mode == 5:
        lines = [("h", 2), ("v", 2)]
    else:
        lines = _random_lines(rng, 3, 5)
    dots = int(rng.integers(1, 6)) if not lines else int(rng.integers(0, 5))
    return lines + [("dot", 1)] * dots


def _test_recipe(rng, capability, force_network=False):
    modes = ("unchanged", "single", "horizontal", "vertical", "mixed", "dense", "network")
    mode = "network" if force_network else modes[int(rng.integers(0, len(modes)))]
    if mode == "unchanged":
        lines = []
    elif mode == "single":
        lines = [("h" if rng.random() < 0.5 else "v", 2)]
    elif mode == "horizontal":
        lines = [("h", 2) for _ in range(int(rng.integers(1, 5)))]
    elif mode == "vertical":
        lines = [("v", 2) for _ in range(int(rng.integers(1, 5)))]
    elif mode == "mixed":
        lines = [("h", 2), ("v", 2)]
        lines += _random_lines(rng, 0, 2)
    elif mode == "network" and capability != "basic":
        lines = [{"triple_h": ("h", int(rng.integers(3, 5))), "triple_v": ("v", int(rng.integers(3, 5))), "dual": ("dual", 1), "cross": ("cross", 1)}[capability]]
        lines += _random_lines(rng, 0, 3)
    else:
        lines = _random_lines(rng, 3, 5)
    dots = int(rng.integers(1, 7)) if not lines else int(rng.integers(0, 5))
    return lines + [("dot", 1)] * dots


def _slot_counts(recipe):
    nr = nc = markers = 0
    for kind, length in recipe:
        if kind == "h":
            nr += 1
            nc += length
            markers += length
        elif kind == "v":
            nr += length
            nc += 1
            markers += length
        elif kind == "dot":
            nr += 1
            nc += 1
            markers += 1
        elif kind == "dual":
            nr += 2
            nc += 2
            markers += 3
        else:
            nr += 3
            nc += 3
            markers += 4
    return nr, nc, markers


def _place_recipe(rng, recipe, height, width):
    nr, nc, _ = _slot_counts(recipe)
    row_slots = [int(x) for x in rng.choice(height, nr, replace=False)]
    col_slots = [int(x) for x in rng.choice(width, nc, replace=False)]
    rng.shuffle(row_slots)
    rng.shuffle(col_slots)
    ri = ci = 0
    markers = []
    for kind, length in recipe:
        if kind == "h":
            r = row_slots[ri]
            markers.extend((r, c) for c in col_slots[ci:ci + length])
            ri += 1
            ci += length
        elif kind == "v":
            c = col_slots[ci]
            markers.extend((r, c) for r in row_slots[ri:ri + length])
            ri += length
            ci += 1
        elif kind == "dot":
            markers.append((row_slots[ri], col_slots[ci]))
            ri += 1
            ci += 1
        elif kind == "dual":
            r0, r1 = row_slots[ri:ri + 2]
            c0, c1 = col_slots[ci:ci + 2]
            markers.extend(((r0, c0), (r0, c1), (r1, c0)))
            ri += 2
            ci += 2
        else:
            low_r, mid_r, high_r = sorted(row_slots[ri:ri + 3])
            low_c, mid_c, high_c = sorted(col_slots[ci:ci + 3])
            markers.extend(((mid_r, low_c), (mid_r, high_c), (low_r, mid_c), (high_r, mid_c)))
            ri += 3
            ci += 3
    return markers


def _sample_pair(rng, recipe, connector, fixed_palette, is_test, min_height=3, min_width=3, max_size=30):
    nr, nc, marker_count = _slot_counts(recipe)
    minimum_h = max(min_height, nr + (1 if any(kind == "v" for kind, _ in recipe) else 0))
    minimum_w = max(min_width, nc + (1 if any(kind == "h" for kind, _ in recipe) else 0))
    if minimum_h > max_size or minimum_w > max_size:
        raise ValueError("recipe too large")
    if fixed_palette is None:
        colors = [x for x in range(10) if x != connector]
        background, marker = (int(x) for x in rng.choice(colors, 2, replace=False))
    else:
        background, marker = fixed_palette
    for _ in range(300):
        height = _draw_dimension(rng, minimum_h, is_test, max_size)
        width = _draw_dimension(rng, minimum_w, is_test, max_size)
        if height * width <= marker_count * 2:
            continue
        positions = _place_recipe(rng, recipe, height, width)
        grid = np.full((height, width), background, dtype=np.int8)
        for r, c in positions:
            grid[r, c] = marker
        features = _features(grid, marker)
        if features["markers"] != marker_count:
            continue
        if any(kind in ("h", "v", "dual", "cross") for kind, _ in recipe) and features["painted"] == 0:
            continue
        if any(kind == "dual" for kind, _ in recipe) and not features["dual"]:
            continue
        if any(kind == "cross" for kind, _ in recipe) and not features["cross"]:
            continue
        if not any(kind == "cross" for kind, _ in recipe) and features["cross"]:
            continue
        if _source_input(grid):
            continue
        return {"input": grid, "output": _apply_rule(grid, marker, connector)}
    raise RuntimeError("could not place a valid scene")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    connector = int(rng.integers(0, 10))
    palette_mode = "variable" if rng.random() < 0.8 else "fixed"
    if palette_mode == "fixed":
        choices = [x for x in range(10) if x != connector]
        fixed_palette = tuple(int(x) for x in rng.choice(choices, 2, replace=False))
    else:
        fixed_palette = None
    capability = ("basic", "basic", "basic", "triple_h", "triple_v", "dual", "cross")[int(rng.integers(0, 7))] if num_train >= 3 else "basic"
    growth_regime = rng.random() < 0.25
    required_dots = [int(x) for x in rng.choice(4, size=2, replace=False)]
    pair_rngs = [np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max))) for _ in range(num_train + num_test)]
    train = []
    test = []
    seen = set()
    for index in range(num_train):
        pair_rng = pair_rngs[index]
        recipe = _training_recipe(pair_rng, index, capability, required_dots)
        for _ in range(50):
            pair = _sample_pair(pair_rng, recipe, connector, fixed_palette, False, max_size=18 if growth_regime else 30)
            sig = _signature(pair["input"])
            if sig not in seen:
                seen.add(sig)
                train.append(pair)
                break
        else:
            raise RuntimeError("could not make distinct training input")
    rng.shuffle(train)
    test_min_height = max(p["input"].shape[0] for p in train) + 1 if growth_regime else 3
    test_min_width = max(p["input"].shape[1] for p in train) + 1 if growth_regime else 3
    for index in range(num_test):
        pair_rng = pair_rngs[num_train + index]
        recipe = ([('h', 2), ('v', 2)] + [('dot', 1)] * int(pair_rng.integers(0, 3))
                  if num_train == 2 else
                  _test_recipe(pair_rng, capability, index == 0 and capability != "basic"))
        for _ in range(50):
            pair = _sample_pair(pair_rng, recipe, connector, fixed_palette, True, test_min_height, test_min_width)
            sig = _signature(pair["input"])
            if sig not in seen:
                seen.add(sig)
                test.append(pair)
                break
        else:
            raise RuntimeError("could not make distinct test input")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_demonstrations(train, test_input):
    changes = {
        int(value)
        for pair in train
        for value in np.unique(pair["output"][pair["input"] != pair["output"]])
    }
    assert len(changes) == 1
    connector = next(iter(changes))
    _, marker = _roles(test_input)
    out = test_input.copy()
    for r in range(test_input.shape[0]):
        for c in range(test_input.shape[1]):
            if test_input[r, c] == marker:
                continue
            horizontal = np.any(test_input[r, :c] == marker) and np.any(test_input[r, c + 1:] == marker)
            vertical = np.any(test_input[:r, c] == marker) and np.any(test_input[r + 1:, c] == marker)
            if horizontal or vertical:
                out[r, c] = connector
    return out


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    training_features = []
    connector_colors = set()
    backgrounds = set()
    markers = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            assert not _source_input(inp)
            background, marker = _roles(inp)
            backgrounds.add(background)
            markers.add(marker)
            features = _features(inp, marker)
            assert 1 <= features["markers"] <= 29
            assert np.array_equal(inp == marker, out == marker)
            changed = out[inp != out]
            if changed.size:
                assert np.all(inp[inp != out] == background)
                assert len(np.unique(changed)) == 1
                connector_colors.add(int(changed[0]))
            if split == "train":
                training_features.append(features)
    assert len(connector_colors) == 1
    connector = next(iter(connector_colors))
    assert connector not in backgrounds and connector not in markers
    for pair in train + test:
        _, marker = _roles(pair["input"])
        assert np.array_equal(pair["output"], _apply_rule(pair["input"], marker, connector))
    assert any(f["horizontal"] and f["painted"] for f in training_features)
    assert any(f["vertical"] and f["painted"] for f in training_features)
    assert any(f["unmatched"] for f in training_features)
    train_triple_h = any(f["triple_h"] for f in training_features)
    train_triple_v = any(f["triple_v"] for f in training_features)
    train_dual = any(f["dual"] for f in training_features)
    train_cross = any(f["cross"] for f in training_features)
    for pair in test:
        _, marker = _roles(pair["input"])
        f = _features(pair["input"], marker)
        assert not f["triple_h"] or train_triple_h
        assert not f["triple_v"] or train_triple_v
        assert not f["dual"] or train_dual
        assert not f["cross"] or train_cross
        assert np.array_equal(_predict_from_demonstrations(train, pair["input"]), pair["output"])


def _puzzle_signature(puzzle):
    return tuple((_signature(pair["input"]), _signature(pair["output"])) for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    all_pairs = [pair for p in puzzles for split in ("train", "test") for pair in p[split]]
    all_features = [_features(pair["input"], _roles(pair["input"])[1]) for pair in all_pairs]
    train_pairs = [pair for p in puzzles for pair in p["train"]]
    test_pairs = [pair for p in puzzles for pair in p["test"]]
    train_features = [_features(pair["input"], _roles(pair["input"])[1]) for pair in train_pairs]
    test_features = [_features(pair["input"], _roles(pair["input"])[1]) for pair in test_pairs]
    assert len({pair["input"].shape for pair in all_pairs}) >= 50
    assert len({_signature(pair["input"]) for pair in all_pairs}) >= 200
    assert {0, 1, 2, 3, 4}.issubset({min(f["segments"], 4) for f in all_features})
    assert any(f["triple_h"] or f["triple_v"] for f in all_features)
    assert any(f["dual"] for f in all_features)
    assert any(f["cross"] for f in all_features)
    assert any(f["segments"] >= 3 for p in puzzles for pair in p["train"] for f in [_features(pair["input"], _roles(pair["input"])[1])])
    assert any(f["segments"] == 2 and f["markers"] == 4 and f["unmatched"] == 0 for f in all_features)
    assert any(f["unmatched"] == 0 for f in train_features)
    assert any(f["unmatched"] >= 2 for f in train_features)
    assert len({(f["segments"], f["unmatched"]) for f in all_features}) >= 20
    assert all(len({_features(pair["input"], _roles(pair["input"])[1])["segments"] for split in ("train", "test") for pair in p[split]}) > 1 for p in puzzles)
    assert all(len({_features(pair["input"], _roles(pair["input"])[1])["unmatched"] for split in ("train", "test") for pair in p[split]}) > 1 for p in puzzles)
    for features in (train_features, test_features):
        assert any(f["triple_h"] for f in features)
        assert any(f["triple_v"] for f in features)
        assert any(f["dual"] for f in features)
        assert any(f["cross"] for f in features)
        assert any(f["adjacent"] for f in features)
    assert any(f["painted"] == 0 for f in test_features)
    assert any(f["segments"] == 1 for f in test_features)
    assert any(f["segments"] >= 3 for f in test_features)
    assert any(f["horizontal"] and not f["vertical"] for f in test_features)
    assert any(f["vertical"] and not f["horizontal"] for f in test_features)
    assert any(f["horizontal"] and f["vertical"] for f in test_features)
    first_features = [_features(p["train"][0]["input"], _roles(p["train"][0]["input"])[1]) for p in puzzles]
    assert any(f["horizontal"] and not f["vertical"] for f in first_features)
    assert any(f["vertical"] and not f["horizontal"] for f in first_features)
    assert any(_roles(pair["input"])[0] == 0 for pair in all_pairs)
    assert len({_roles(pair["input"])[0] for pair in all_pairs}) == 10
    assert len({_roles(pair["input"])[1] for pair in all_pairs}) == 10
    assert any(len({_roles(pair["input"])[0] for split in ("train", "test") for pair in p[split]}) > 1 for p in puzzles)
    assert any(len({_roles(pair["input"])[1] for split in ("train", "test") for pair in p[split]}) > 1 for p in puzzles)
    assert any(any(np.array_equal(pair["input"], pair["output"]) for pair in p["test"]) for p in puzzles)
    assert any(30 in pair["input"].shape for pair in train_pairs)
    assert any(30 in pair["input"].shape for pair in test_pairs)
    assert any(pair["input"].shape[0] >= 2 * pair["input"].shape[1] for pair in all_pairs)
    assert any(pair["input"].shape[1] >= 2 * pair["input"].shape[0] for pair in all_pairs)
    assert any(f["max_gap"] >= 10 for f in all_features)
    assert any(1 <= f["max_gap"] <= 2 for f in all_features)
    assert any(any(t["input"].shape[0] > max(x["input"].shape[0] for x in p["train"]) and t["input"].shape[1] > max(x["input"].shape[1] for x in p["train"]) for t in p["test"]) for p in puzzles)
    assert any(any(t["input"].shape[0] <= max(x["input"].shape[0] for x in p["train"]) and t["input"].shape[1] <= max(x["input"].shape[1] for x in p["train"]) for t in p["test"]) for p in puzzles)
    for p in puzzles:
        validate_generated_puzzle(p)
        for index, pair in enumerate(p["train"]):
            demonstrations = p["train"][:index] + p["train"][index + 1:]
            if any(np.any(other["input"] != other["output"]) for other in demonstrations):
                assert np.array_equal(_predict_from_demonstrations(demonstrations, pair["input"]), pair["output"])
    for args in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + args[0] * 10 + args[1], *args)
        assert (len(p["train"]), len(p["test"])) == args
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(2000 + (args[0] or 0) * 10 + (args[1] or 0), *args)
        assert args[0] is None or len(p["train"]) == args[0]
        assert args[1] is None or len(p["test"]) == args[1]
    assert _puzzle_signature(generate_puzzle(98765, 7, 3)) == _puzzle_signature(generate_puzzle(98765, 7, 3))
    assert _puzzle_signature(generate_puzzle(24680)) == _puzzle_signature(generate_puzzle(24680))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
