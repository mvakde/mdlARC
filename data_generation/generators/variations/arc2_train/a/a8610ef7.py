import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (("080888", "888880", "808080", "088808", "880880", "880008"),
     ("020552", "225220", "502050", "052505", "220220", "520002")),
    (("880880", "808880", "008888", "080088", "880808", "800808"),
     ("250250", "205250", "005522", "050022", "250205", "200205")),
    (("088008", "888000", "888080", "808808", "880000", "888880"),
     ("022005", "225000", "252050", "202505", "220000", "522550")),
    (("888000", "008808", "080000", "880088", "808888", "000088"),
     ("555000", "002202", "020000", "520055", "502252", "000055")),
    (("000808", "888088", "888808", "800088", "080008", "888088"),
     ("000502", "525052", "255502", "200052", "020002", "555052")),
}


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _is_source_pair(inp, out):
    return (_rows(inp), _rows(out)) in _SOURCE_PAIR_SIGNATURES


def _array_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _validate_count("num_train", num_train, 2, 10)
    test = _validate_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def sample_puzzle_parameters(rng):
    colors = [int(v) for v in rng.permutation(10)]
    mode = ("stable", "vary_foreground", "vary_background")[int(rng.integers(3))]
    return {
        "mode": mode,
        "background": colors[0],
        "foreground": colors[1],
        "matched": colors[2],
        "unmatched": colors[3],
        "alternate": colors[4],
    }


def _role_masks(mask):
    partner = mask[::-1, :]
    return mask & partner, mask & ~partner


def _axis_is_unambiguous(mask):
    intended = mask[::-1, :][mask]
    for alternative in (mask[:, ::-1][mask], mask[::-1, ::-1][mask]):
        if np.array_equal(intended, alternative) or np.array_equal(intended, ~alternative):
            return False
    return True


def _sample_shape(rng, kind=None):
    if kind == "even":
        height = int(rng.choice(np.arange(4, 19, 2)))
    elif kind == "odd":
        height = int(rng.choice(np.arange(5, 18, 2)))
    else:
        height = int(rng.integers(4, 19))
    return height, int(rng.integers(4, 19))


def _add_strokes(rng, mask):
    height, width = mask.shape
    for _ in range(int(rng.integers(1, 5))):
        if rng.random() < 0.5:
            row = int(rng.integers(height))
            start = int(rng.integers(width - 1))
            end = int(rng.integers(start + 2, width + 1))
            mask[row, start:end] = True
        else:
            col = int(rng.integers(width))
            start = int(rng.integers(height - 1))
            end = int(rng.integers(start + 2, height + 1))
            mask[start:end, col] = True


def _sample_mask(rng, height, width, style, center_required=False):
    if style == "sparse":
        mask = np.zeros((height, width), dtype=bool)
        count = int(rng.integers(1, max(3, min(10, height * width // 8)) + 1))
        mask.flat[rng.choice(height * width, size=count, replace=False)] = True
        if rng.random() < 0.45:
            _add_strokes(rng, mask)
    elif style == "dense":
        mask = rng.random((height, width)) < rng.uniform(0.83, 0.97)
    elif style in ("top_heavy", "bottom_heavy"):
        high, low = rng.uniform(0.65, 0.92), rng.uniform(0.03, 0.23)
        upper, lower = (high, low) if style == "top_heavy" else (low, high)
        mask = np.zeros((height, width), dtype=bool)
        mask[:height // 2] = rng.random((height // 2, width)) < upper
        mask[(height + 1) // 2:] = rng.random((height // 2, width)) < lower
        if height % 2:
            mask[height // 2] = rng.random(width) < rng.uniform(0.2, 0.8)
    elif style == "symmetric":
        mask = np.zeros((height, width), dtype=bool)
        upper = rng.random((height // 2, width)) < rng.uniform(0.22, 0.82)
        mask[:height // 2] = upper
        mask[(height + 1) // 2:] = upper[::-1]
        if height % 2:
            mask[height // 2] = rng.random(width) < rng.uniform(0.2, 0.8)
    elif style == "unmatched_only":
        mask = np.zeros((height, width), dtype=bool)
        choices = rng.choice(3, size=(height // 2, width), p=[0.4, 0.3, 0.3])
        mask[:height // 2] = choices == 1
        mask[(height + 1) // 2:] = (choices == 2)[::-1]
    else:
        mask = rng.random((height, width)) < rng.uniform(0.28, 0.72)
        if rng.random() < 0.65:
            _add_strokes(rng, mask)

    # Complete columns, rows, and partial boundary runs can occur in either split.
    if style in ("balanced", "dense", "top_heavy", "bottom_heavy"):
        if rng.random() < 0.27:
            n = int(rng.integers(1, min(3, width) + 1))
            mask[:, rng.choice(width, size=n, replace=False)] = True
        if rng.random() < 0.20:
            mask[int(rng.integers(height)), :] = True
        if rng.random() < 0.35:
            row = int(rng.choice([0, height - 1]))
            start = int(rng.integers(width - 2))
            length = int(rng.integers(2, width - start))
            mask[row, start:start + length] = True
    if center_required:
        mask[height // 2, int(rng.integers(width))] = True
    return mask


def _pair_colors(parameters, index):
    bg, fg = parameters["background"], parameters["foreground"]
    if index % 2:
        if parameters["mode"] == "vary_foreground":
            fg = parameters["alternate"]
        elif parameters["mode"] == "vary_background":
            bg = parameters["alternate"]
    return bg, fg


def generate_pair(rng, regime, puzzle_parameters, index=0, style=None, shape_override=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown pair regime: {regime}")
    if regime == "source_train" and index == 0:
        shape_kind, style, decisive, center = "even", "balanced", True, False
    elif regime == "source_train" and index == 1:
        shape_kind, style, decisive, center = "odd", "balanced", True, True
    else:
        shape_kind, decisive, center = None, False, False
        if style is None:
            styles = ("balanced", "sparse", "dense", "top_heavy",
                      "bottom_heavy", "symmetric", "unmatched_only")
            style = styles[int(rng.integers(len(styles)))]
    bg, fg = _pair_colors(puzzle_parameters, index)
    for _ in range(200):
        height, width = shape_override if shape_override is not None else _sample_shape(rng, shape_kind)
        mask = _sample_mask(rng, height, width, style, center)
        matched, unmatched = _role_masks(mask)
        if not np.any(mask) or np.all(mask):
            continue
        if decisive and (not np.any(matched) or not np.any(unmatched)
                         or not _axis_is_unambiguous(mask)):
            continue
        inp = np.full(mask.shape, bg, dtype=np.int8)
        inp[mask] = fg
        out = np.full(mask.shape, bg, dtype=np.int8)
        out[matched] = puzzle_parameters["matched"]
        out[unmatched] = puzzle_parameters["unmatched"]
        if not _is_source_pair(inp, out):
            return inp, out
    raise RuntimeError("could not construct a valid pair")


def _learn_conventions(train):
    first = train[0]
    inp, out = first["input"], first["output"]
    shared = set(np.unique(inp).tolist()) & set(np.unique(out).tolist())
    assert len(shared) == 1
    bg = int(next(iter(shared)))
    fg = int(next(iter(set(np.unique(inp).tolist()) - {bg})))
    mask = inp == fg
    matched, unmatched = _role_masks(mask)
    assert np.any(matched) and np.any(unmatched) and _axis_is_unambiguous(mask)
    matched_color = int(out[matched][0])
    unmatched_color = int(out[unmatched][0])
    assert len({bg, fg, matched_color, unmatched_color}) == 4
    roles = []
    for pair in train:
        colors = set(np.unique(pair["input"]).tolist())
        common = colors & set(np.unique(pair["output"]).tolist())
        assert len(colors) == 2 and len(common) == 1
        local_bg = int(next(iter(common)))
        local_fg = int(next(iter(colors - {local_bg})))
        roles.append((local_bg, local_fg))
    assert len({b for b, _ in roles}) == 1 or len({f for _, f in roles}) == 1
    return matched_color, unmatched_color, roles


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    matched_color, unmatched_color, roles = _learn_conventions(puzzle["train"])
    assert matched_color != unmatched_color
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and len(np.unique(inp)) == 2
            if split == "train":
                bg, fg = roles[index]
            else:
                colors = set(np.unique(inp).tolist())
                candidates = [(b, f) for b, f in roles if colors == {b, f}]
                assert candidates
                bg, fg = candidates[0]
            assert len({bg, fg, matched_color, unmatched_color}) == 4
            mask = inp == fg
            matched, unmatched = _role_masks(mask)
            expected = np.full(inp.shape, bg, dtype=np.int8)
            expected[matched] = matched_color
            expected[unmatched] = unmatched_color
            assert np.array_equal(out, expected)
            assert not np.array_equal(inp, out)
            assert not _is_source_pair(inp, out)
            ikey = _array_key(inp)
            pkey = (ikey, _array_key(out))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
    # First training pair fixes both output colors and distinguishes the axis.
    first = puzzle["train"][0]
    first_mask = first["input"] == roles[0][1]
    first_matched, first_unmatched = _role_masks(first_mask)
    assert np.any(first_matched) and np.any(first_unmatched)
    assert _axis_is_unambiguous(first_mask)
    # Middle-row self-pairing is demonstrated in training.
    second = puzzle["train"][1]
    second_mask = second["input"] == roles[1][1]
    height = second_mask.shape[0]
    assert height % 2 == 1 and np.any(second_mask[height // 2])
    assert np.all(second["output"][height // 2][second_mask[height // 2]] == matched_color)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    styles = ("balanced", "sparse", "dense", "top_heavy",
              "bottom_heavy", "symmetric", "unmatched_only")
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_index = index if split == "train" else int(rng.integers(0, 2))
            style = None if split == "train" and index < 2 else str(rng.choice(styles))
            shape_override = None
            if split == "test" and rng.random() < 0.35:
                shape_override = puzzle["train"][int(rng.integers(train_count))]["input"].shape
            for _ in range(100):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out = generate_pair(child_rng, "source_train" if split == "train" else "target_test",
                                         parameters, pair_index, style, shape_override)
                key = _array_key(inp)
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not generate unique pairs")
            seen.add(key)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k])
        for a, b in zip(left[s], right[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _has_incomplete_edge_run(mask):
    for line in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        run = 0
        for value in line:
            if value:
                run += 1
            else:
                if 2 <= run < len(line):
                    return True
                run = 0
        if 2 <= run < len(line):
            return True
    return False


def self_test() -> None:
    train_counts, test_counts, shapes, palettes, layouts = set(), set(), set(), set(), set()
    full_train, full_test, densities, role_cases = set(), set(), set(), set()
    full_rows_train, full_rows_test = set(), set()
    heights, widths, palette_modes = set(), set(), set()
    same_shape_tests = new_shape_tests = 0
    top_heavy = bottom_heavy = edge_runs = odd_center = 0
    sparse = dense = 0
    combined_palette_and_extreme = 0
    changed_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        matched_color, unmatched_color, roles = _learn_conventions(puzzle["train"])
        palettes.add((matched_color, unmatched_color, tuple(roles)))
        changed_palettes += len(set(roles)) > 1
        if len({b for b, _ in roles}) > 1:
            palette_modes.add("vary_background")
        elif len({f for _, f in roles}) > 1:
            palette_modes.add("vary_foreground")
        else:
            palette_modes.add("stable")
        assert len({pair["input"].shape for pair in puzzle["train"]}) > 1
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                inp, out = pair["input"], pair["output"]
                bg, fg = roles[index] if split == "train" else next(
                    (b, f) for b, f in roles if set(np.unique(inp).tolist()) == {b, f})
                mask = inp == fg
                matched, unmatched = _role_masks(mask)
                reconstruction = np.full(inp.shape, bg, dtype=np.int8)
                reconstruction[matched] = matched_color
                reconstruction[unmatched] = unmatched_color
                assert np.array_equal(out, reconstruction)
                shapes.add(inp.shape)
                heights.add(inp.shape[0])
                widths.add(inp.shape[1])
                layouts.add((inp.shape, tuple(mask.flat)))
                densities.add(int(mask.mean() * 20))
                role_cases.add((bool(np.any(matched)), bool(np.any(unmatched))))
                (full_train if split == "train" else full_test).add(
                    int(np.count_nonzero(np.all(mask, axis=0))))
                (full_rows_train if split == "train" else full_rows_test).add(
                    int(np.count_nonzero(np.all(mask, axis=1))))
                height = mask.shape[0]
                top = mask[:height // 2].mean()
                bottom = mask[(height + 1) // 2:].mean()
                top_heavy += top - bottom > 0.3
                bottom_heavy += bottom - top > 0.3
                edge_runs += _has_incomplete_edge_run(mask)
                odd_center += height % 2 and np.any(mask[height // 2])
                sparse += mask.mean() < 0.15
                dense += mask.mean() > 0.85
                if split == "test":
                    same_shape_tests += inp.shape in train_shapes
                    new_shape_tests += inp.shape not in train_shapes
                    combined_palette_and_extreme += (
                        len(set(roles)) > 1
                        and (not np.any(matched) or not np.any(unmatched))
                    )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 50 and len(palettes) >= 30 and len(layouts) >= 150
    assert len(densities) >= 15 and changed_palettes >= 10
    assert {0, 1, 2} <= full_train and {0, 1, 2} <= full_test
    assert {0, 1, 2} <= full_rows_train and {0, 1, 2} <= full_rows_test
    assert {(True, True), (True, False), (False, True)} <= role_cases
    assert {4, 5, 17, 18} <= heights and {4, 18} <= widths
    assert palette_modes == {"stable", "vary_foreground", "vary_background"}
    assert same_shape_tests > 0 and new_shape_tests > 0
    assert min(top_heavy, bottom_heavy, edge_runs, odd_center, sparse, dense) > 0
    assert combined_palette_and_extreme > 0
    assert _puzzles_equal(generate_puzzle(987654, 8, 3), generate_puzzle(987654, 8, 3))
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {args}")
