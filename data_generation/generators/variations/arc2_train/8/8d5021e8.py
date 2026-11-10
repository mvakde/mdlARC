"""Whole-grid mirror tilings for the reviewed ARC family."""

import numpy as np


_SOURCE_INPUTS = {
    ((2, 0), (2, 2), (2, 0)),
    ((0, 8), (0, 0), (0, 8)),
    ((0, 0), (0, 5), (5, 0)),
    ((3, 3), (3, 0), (3, 3)),
}
_SIZES = ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5),
          (4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (6, 2))


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _value_key(grid):
    return tuple(tuple(int(x) for x in row) for row in grid)


def _canonical_key(grid):
    labels = {}
    return grid.shape, tuple(labels.setdefault(int(x), len(labels)) for x in grid.flat)


def _transform(grid):
    middle = np.concatenate((np.fliplr(grid), grid), axis=1)
    outer = np.flipud(middle)
    return np.concatenate((outer, middle, outer), axis=0).copy()


def _symmetry(grid):
    vertical = np.array_equal(grid, np.flipud(grid))
    horizontal = np.array_equal(grid, np.fliplr(grid))
    if vertical and horizontal:
        return "both"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "none"


def _validate_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _validate_optional_count("num_train", num_train, 2, 10)
    num_test = _validate_optional_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        return minimum + (seed // 4) % (11 - minimum), num_test
    if num_train is None:
        return int(rng.integers(max(2, num_test + 1), 11)), num_test
    return num_train, int(rng.integers(1, min(4, num_train - 1) + 1))


def _draw_mask(rng, shape, symmetry, density, minimum, maximum=None):
    h, w = shape
    size = h * w
    maximum = size - 1 if maximum is None else maximum
    for _ in range(128):
        mode = int(rng.integers(0, 8))
        if mode == 0:
            mask = rng.random(shape) < density
        elif mode == 1:
            mask = np.broadcast_to(rng.random((h, 1)) < density, shape).copy()
        elif mode == 2:
            mask = np.broadcast_to(rng.random((1, w)) < density, shape).copy()
        elif mode == 3:
            mask = np.zeros(shape, dtype=bool)
            for i in range(h):
                mask[i, (i + int(rng.integers(0, w))) % w] = True
        elif mode == 4:
            mask = np.zeros(shape, dtype=bool)
            chosen = rng.choice(size, size=int(rng.integers(1, min(size, 5))), replace=False)
            mask.flat[chosen] = True
        elif mode == 5:
            mask = np.ones(shape, dtype=bool)
            chosen = rng.choice(size, size=int(rng.integers(1, min(size, 5))), replace=False)
            mask.flat[chosen] = False
        elif mode == 6:
            mask = np.zeros(shape, dtype=bool)
            top = int(rng.integers(0, h))
            left = int(rng.integers(0, w))
            bottom = int(rng.integers(top + 1, h + 1))
            right = int(rng.integers(left + 1, w + 1))
            mask[top:bottom, left:right] = True
        else:
            mask = rng.random(shape) < density
            mask.flat[int(rng.integers(size))] = True
        if symmetry in ("vertical", "both"):
            for row in range(h // 2):
                mask[h - 1 - row, :] = mask[row, :]
        if symmetry in ("horizontal", "both"):
            for col in range(w // 2):
                mask[:, w - 1 - col] = mask[:, col]
        count = int(mask.sum())
        if minimum <= count <= maximum and _symmetry(mask) == symmetry:
            return mask
    # Every listed size has enough cells for this diagnostic pattern.
    for _ in range(1024):
        mask = rng.random(shape) < 0.5
        if symmetry in ("vertical", "both"):
            for row in range(h // 2):
                mask[h - 1 - row, :] = mask[row, :]
        if symmetry in ("horizontal", "both"):
            for col in range(w // 2):
                mask[:, w - 1 - col] = mask[:, col]
        if minimum <= int(mask.sum()) <= maximum and _symmetry(mask) == symmetry:
            return mask
    raise AssertionError("unable to draw a nonconstant motif")


def _sample_palette(rng, multi, excluded=(), mandatory=(), forced_style=None):
    mandatory = list(dict.fromkeys(int(x) for x in mandatory))
    excluded = set(excluded)
    nonzero = [x for x in range(1, 10) if x not in excluded]
    count = (int(rng.integers(3, 5)) if multi else 2)
    count = max(count, len(mandatory) + 1)
    style = int(rng.integers(0, 3)) if forced_style is None else forced_style
    if style == 0:  # zero background
        base = 0
        accents = list(mandatory)
    elif style == 1:  # nonzero background, zero accent
        base = mandatory.pop(0) if mandatory else int(rng.choice(nonzero))
        accents = [0] + mandatory
    else:  # two or more nonzero colors
        base = mandatory.pop(0) if mandatory else int(rng.choice(nonzero))
        accents = list(mandatory)
    available = [x for x in nonzero if x != base and x not in accents]
    rng.shuffle(available)
    accents.extend(available[:count - 1 - len(accents)])
    assert len(accents) == count - 1
    return int(base), [int(x) for x in accents]


def _colorize(rng, mask, base, accents):
    grid = np.full(mask.shape, base, dtype=np.int8)
    positions = np.flatnonzero(mask.ravel())
    rng.shuffle(positions)
    for position, color in zip(positions, accents):
        grid.flat[int(position)] = color
    for position in positions[len(accents):]:
        grid.flat[int(position)] = int(rng.choice(accents))
    return grid


def _sample_pair(rng, shape, symmetry, density, multi, excluded,
                 mandatory, used, prior_masks, minimum_active=1,
                 maximum_active=None, forced_style=None, forbidden_canonical=()):
    for _ in range(128):
        base, accents = _sample_palette(rng, multi, excluded, mandatory,
                                        forced_style)
        eligible = [mask for mask in prior_masks if mask.shape == shape
                    and _symmetry(mask) == symmetry
                    and max(len(accents), minimum_active) <= int(mask.sum())
                    and (maximum_active is None or int(mask.sum()) <= maximum_active)]
        if eligible and rng.random() < 0.28:
            mask = eligible[int(rng.integers(len(eligible)))].copy()
        else:
            mask = _draw_mask(rng, shape, symmetry, density,
                              max(len(accents), minimum_active), maximum_active)
        grid = _colorize(rng, mask, base, accents)
        key = _grid_key(grid)
        if (key in used or _value_key(grid) in _SOURCE_INPUTS
                or _canonical_key(grid) in forbidden_canonical):
            continue
        return {"input": grid, "output": _transform(grid)}, mask
    raise AssertionError("unable to construct a distinct pair")


def _candidate_output(grid, original_on_right, reflected_outer):
    middle = np.concatenate((np.fliplr(grid), grid), axis=1) if original_on_right else np.concatenate((grid, np.fliplr(grid)), axis=1)
    outer = np.flipud(middle) if reflected_outer else middle
    return np.concatenate((outer, middle, outer), axis=0)


def _predict_from_training(train, test_input):
    candidates = []
    for right in (False, True):
        for reflected in (False, True):
            if all(np.array_equal(pair["output"],
                                  _candidate_output(pair["input"], right, reflected))
                   for pair in train):
                candidates.append((right, reflected))
    assert candidates == [(True, True)]
    return _candidate_output(test_input, *candidates[0])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)

    # The reflection convention has no latent color map. Size regime and palette
    # sharing are puzzle-level sampling choices; colors and motifs remain local.
    mixed_sizes = bool(rng.random() < 0.74)
    share_color = bool(rng.random() < 0.65)
    introduce_color = bool(rng.random() < 0.55)
    common = int(rng.integers(1, 10))
    novel = int(rng.choice([x for x in range(1, 10) if x != common]))
    excluded = (novel,) if introduce_color else ()
    test_multi = bool(rng.random() < 0.43) or (share_color and introduce_color and num_test == 1)

    train_shapes = [(3, 2)]
    train_shapes.append(_SIZES[int(rng.integers(len(_SIZES)))] if mixed_sizes else (3, 2))
    for _ in range(2, num_train):
        train_shapes.append(_SIZES[int(rng.integers(len(_SIZES)))]
                            if mixed_sizes and rng.random() >= 0.45 else (3, 2))
    test_shapes = [(_SIZES[int(rng.integers(len(_SIZES)))]
                    if mixed_sizes and rng.random() >= 0.35 else (3, 2))
                   for _ in range(num_test)]

    symmetry_values = ("none", "vertical", "horizontal", "both")
    symmetry_probabilities = (0.49, 0.25, 0.17, 0.09)
    train_symmetries = ["none"] + [str(rng.choice(symmetry_values, p=symmetry_probabilities))
                                    for _ in range(num_train - 1)]
    test_symmetries = [str(rng.choice(symmetry_values, p=symmetry_probabilities))
                       for _ in range(num_test)]
    if num_test >= 2 and rng.random() < 0.65:
        test_symmetries[0] = "none"
        test_symmetries[1] = str(rng.choice(symmetry_values[1:]))
    dense_test_index = None
    if rng.random() < 0.28 and (num_test >= 2 or not (share_color and introduce_color)):
        dense_test_index = num_test - 1
        test_shapes[dense_test_index] = (3, 2)
        test_symmetries[dense_test_index] = "vertical"

    train_multi = [False] * num_train
    train_multi[0] = test_multi or (share_color and introduce_color) or bool(rng.random() < 0.3)
    for index in range(1, num_train):
        train_multi[index] = bool(rng.random() < 0.3)
    test_multi_flags = [test_multi] + [bool(rng.random() < 0.3) for _ in range(num_test - 1)]
    if dense_test_index is not None:
        test_multi_flags[dense_test_index] = False
    if any(test_multi_flags):
        train_multi[0] = True

    used = set()
    train_canonical = set()
    prior_masks = []
    puzzle = {"train": [], "test": []}
    for split in ("train", "test"):
        count = num_train if split == "train" else num_test
        shapes = train_shapes if split == "train" else test_shapes
        symmetries = train_symmetries if split == "train" else test_symmetries
        multi_flags = train_multi if split == "train" else test_multi_flags
        for index in range(count):
            mandatory = []
            if index == 0 and share_color:
                mandatory.append(common)
            if split == "test" and index == 0 and introduce_color:
                mandatory.append(novel)
            density = float(rng.uniform(0.54, 0.82)) if split == "train" and index == 0 else float(rng.uniform(0.18, 0.82))
            dense = split == "test" and index == dense_test_index
            if dense:
                density = float(rng.uniform(0.68, 0.86))
            pair, mask = _sample_pair(rng, shapes[index], symmetries[index],
                                      density, multi_flags[index],
                                      excluded if split == "train" else (),
                                      mandatory, used, prior_masks,
                                      minimum_active=4 if dense else 1,
                                      maximum_active=5 if dense else None,
                                      forced_style=0 if dense else None,
                                      forbidden_canonical=train_canonical
                                      if split == "train" and num_train <= 3 else ())
            puzzle[split].append(pair)
            used.add(_grid_key(pair["input"]))
            if split == "train":
                train_canonical.add(_canonical_key(pair["input"]))
            prior_masks.append(mask)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs = set(), set()
    train_sizes, train_colors = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            h, w = source.shape
            assert h >= 2 and w >= 2 and target.shape == (3 * h, 2 * w)
            assert len(np.unique(source)) >= 2
            assert np.array_equal(np.unique(source), np.unique(target))
            assert np.array_equal(target, _transform(source))
            assert _value_key(source) not in _SOURCE_INPUTS
            source_key, target_key = _grid_key(source), _grid_key(target)
            assert source_key not in inputs and (source_key, target_key) not in pairs
            inputs.add(source_key)
            pairs.add((source_key, target_key))
            if split == "train":
                train_sizes.add(source.shape)
                train_colors.add(len(np.unique(source)))
    assert any(_symmetry(pair["input"]) == "none" for pair in train)
    assert all(pair["input"].shape == (3, 2) for pair in test) or len(train_sizes) >= 2
    assert all(len(np.unique(pair["input"])) <= 2 for pair in test) or max(train_colors) >= 3
    for pair in test:
        assert np.array_equal(pair["output"], _predict_from_training(train, pair["input"]))


def _signature(puzzle):
    return tuple((split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    import collections

    def expect_error(**kwargs):
        try:
            generate_puzzle(9001, **kwargs)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError for {kwargs}")

    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        expect_error(**arguments)
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                                          (201, None, 1), (202, None, 4),
                                          (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train_count, test_count))

    metrics = collections.Counter()
    signatures, input_signatures, masks, sizes, palettes = set(), set(), set(), set(), set()
    edge_signatures, combinations, train_symmetry_mixes = set(), set(), set()
    train_counts, test_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [pair["input"] for pair in puzzle["train"]]
        test = [pair["input"] for pair in puzzle["test"]]
        train_symmetry_mixes.add(tuple(sum(_symmetry(grid) == name for grid in train)
                                       for name in ("none", "vertical", "horizontal", "both")))
        train_masks = {(grid.shape, tuple((grid != grid.flat[0]).flat)) for grid in train}
        test_masks = {(grid.shape, tuple((grid != grid.flat[0]).flat)) for grid in test}
        if train_masks & test_masks:
            metrics["recolored_layout"] += 1
        train_palette = set(int(x) for grid in train for x in np.unique(grid))
        test_palette = set(int(x) for grid in test for x in np.unique(grid))
        if train_palette & test_palette:
            metrics["color_overlap"] += 1
        if test_palette - train_palette:
            metrics["new_test_color"] += 1
        if len({grid.shape for grid in train}) >= 2:
            metrics["mixed_train_sizes"] += 1
        if any(grid.shape != (3, 2) for grid in test):
            metrics["changed_test_size"] += 1
        if any(_symmetry(grid) == "none" for grid in test):
            metrics["asymmetric_test"] += 1
        if any(_symmetry(grid) != "none" for grid in test):
            metrics["symmetric_test"] += 1
        if len(test) >= 2 and any(_symmetry(grid) == "none" for grid in test) and any(_symmetry(grid) != "none" for grid in test):
            metrics["mixed_test_symmetry"] += 1
        if np.mean([np.count_nonzero(grid != grid.flat[0]) / grid.size for grid in test]) < np.mean([np.count_nonzero(grid != grid.flat[0]) / grid.size for grid in train]):
            metrics["sparser_test"] += 1
        if any(len(np.unique(grid)) >= 3 for grid in test):
            metrics["multicolor_test"] += 1
        all_grids = train + test
        if any(0 not in grid for grid in all_grids):
            metrics["no_zero"] += 1
        if any(0 in grid and np.count_nonzero(grid == 0) < grid.size / 2 for grid in all_grids):
            metrics["zero_accent"] += 1
        if any(0 not in grid and len(np.unique(grid)) == 2 for grid in all_grids):
            metrics["two_nonzero"] += 1
        if any(grid.shape != (3, 2) and len(np.unique(grid)) >= 3
               and _symmetry(grid) == "none" for grid in test):
            metrics["combined_axes"] += 1
        if any(grid.shape == (3, 2) and len(np.unique(grid)) == 2 and 0 in grid
               and np.count_nonzero(grid) >= 4 and _symmetry(grid) != "none"
               for grid in test):
            metrics["dense_symmetric_test"] += 1
        if any(grid.shape == (3, 2) and len(np.unique(grid)) == 2 and 0 in grid
               and np.count_nonzero(grid) <= 3 and _symmetry(grid) == "none"
               for grid in test):
            metrics["sparse_asymmetric_test"] += 1
        for grid in all_grids:
            input_signatures.add(_grid_key(grid))
            masks.add((grid.shape, tuple((grid != grid.flat[0]).flat)))
            sizes.add(grid.shape)
            palettes.add(tuple(int(x) for x in np.unique(grid)))
            binary = grid != grid.flat[0]
            edge_signatures.add((grid.shape, tuple(binary[0]), tuple(binary[-1]),
                                 tuple(binary[:, 0]), tuple(binary[:, -1])))
            combinations.add((grid.shape, _symmetry(grid), len(np.unique(grid)), 0 in grid))
        for pair in puzzle["train"] + puzzle["test"]:
            source, target = pair["input"], pair["output"]
            h, w = source.shape
            middle = np.hstack((source[:, ::-1], source))
            assert target.shape == (3 * h, 2 * w)
            assert np.array_equal(target[:h], middle[::-1])
            assert np.array_equal(target[h:2*h], middle)
            assert np.array_equal(target[2*h:], middle[::-1])
            assert _value_key(source) not in _SOURCE_INPUTS
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(input_signatures) >= 150
    assert len(masks) >= 90 and len(sizes) >= 9 and len(palettes) >= 25
    assert len(edge_signatures) >= 90 and len(combinations) >= 60
    assert len(train_symmetry_mixes) >= 15
    for name, minimum in (("asymmetric_test", 15), ("symmetric_test", 12),
                          ("mixed_test_symmetry", 5), ("sparser_test", 8),
                          ("mixed_train_sizes", 20), ("changed_test_size", 15),
                          ("color_overlap", 25), ("new_test_color", 12),
                          ("multicolor_test", 12), ("no_zero", 12),
                          ("zero_accent", 12), ("two_nonzero", 12),
                          ("recolored_layout", 5), ("combined_axes", 5),
                          ("dense_symmetric_test", 5),
                          ("sparse_asymmetric_test", 2)):
        assert metrics[name] >= minimum, (name, metrics[name])
