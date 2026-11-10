"""Generator for mosaics enlarged by independent row and column replication."""

import hashlib

import numpy as np


_SHAPES = ((2, 2), (3, 3), (3, 2), (2, 3), (4, 2),
           (2, 4), (4, 3), (3, 4), (5, 2), (2, 5))
_WEIGHTS = np.array((15, 18, 12, 12, 9, 9, 8, 8, 5, 4), dtype=float) / 100
_SOURCE_DIGESTS = {
    "c67235a30ffc1b17b5ba3bfbce383d70255c509e133ec5b151f8c31432976abd",
    "8db74c77ca88d36d7843871e16bf12b2d85de07e537abc2c576031b72a64516d",
    "be4f21901417badbf7cab99c825bb2f88850055cbe1b766bdf570607db13767a",
    "c8d7e03bfa1200f183089fc4b03cc97f1ae6bea3ff764b550b3abd8f11dcb951",
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, seed, train, test):
    if train is not None:
        if not _integer(train) or not 2 <= train <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        train = int(train)
    if test is not None:
        if not _integer(test) or not 1 <= test <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        test = int(test)
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _visible_boundaries(grid):
    return (all(not np.array_equal(grid[i - 1], grid[i])
                for i in range(1, grid.shape[0]))
            and all(not np.array_equal(grid[:, j - 1], grid[:, j])
                    for j in range(1, grid.shape[1])))


def _single_witness_seam(grid):
    for j in range(1, grid.shape[1]):
        equal = grid[:, j - 1] == grid[:, j]
        if bool(np.any(equal)) and np.count_nonzero(~equal) == 1:
            return True
    for i in range(1, grid.shape[0]):
        equal = grid[i - 1] == grid[i]
        if bool(np.any(equal)) and np.count_nonzero(~equal) == 1:
            return True
    return False


def _symmetric(grid):
    return (np.array_equal(grid, grid[::-1, :])
            or np.array_equal(grid, grid[:, ::-1])
            or np.array_equal(grid, grid[::-1, ::-1]))


def _latent(rng, shape, background, kind):
    height, width = shape
    available = np.array([c for c in range(10) if c != background], dtype=np.int8)
    if kind == "single_seam":
        sizes = (2, 3)
    elif kind == "low":
        sizes = (2, 3, 4)
    elif kind == "high":
        sizes = tuple(range(min(6, height * width), min(9, height * width) + 1))
    elif kind == "symmetric":
        sizes = (2, 3, 4, 5)
    else:
        sizes = tuple(range(2, min(9, height * width) + 1))
    for _ in range(600):
        if kind == "symmetric":
            symmetry = int(rng.integers(3))
            orbits, seen = [], set()
            for i in range(height):
                for j in range(width):
                    if (i, j) in seen:
                        continue
                    if symmetry == 0:
                        orbit = {(i, j), (height - 1 - i, j)}
                    elif symmetry == 1:
                        orbit = {(i, j), (i, width - 1 - j)}
                    else:
                        orbit = {(i, j), (height - 1 - i, width - 1 - j)}
                    seen.update(orbit)
                    orbits.append(orbit)
            possible = [size for size in sizes if size <= len(orbits)]
            palette = rng.choice(available, size=int(rng.choice(possible)), replace=False)
            values = np.concatenate((palette, rng.choice(palette, len(orbits) - len(palette))))
            rng.shuffle(values)
            grid = np.empty(shape, dtype=np.int8)
            for value, orbit in zip(values, orbits):
                for position in orbit:
                    grid[position] = value
        else:
            palette = rng.choice(available, size=int(rng.choice(sizes)), replace=False)
            values = np.concatenate((palette, rng.choice(palette, height * width - len(palette))))
            rng.shuffle(values)
            grid = values.reshape(shape).astype(np.int8, copy=True)
        if _visible_boundaries(grid) and (kind != "single_seam" or _single_witness_seam(grid)):
            return grid
    raise ValueError("Could not sample a mosaic with visible logical boundaries")


def _shape(rng, exclude=None, small=False):
    choices = [i for i, shape in enumerate(_SHAPES)
               if shape != exclude and
               (not small or shape in ((2, 2), (3, 3), (3, 2), (2, 3)))]
    weights = _WEIGHTS[choices]
    return _SHAPES[int(rng.choice(choices, p=weights / weights.sum()))]


def _profile(rng, shape, regime, uneven=False):
    names = ["uniform", "row_uniform", "column_uniform", "mild",
             "strong", "double_peak", "one_pixel"]
    weights = ([.12, .12, .12, .23, .18, .15, .08] if regime == "source_train"
               else [.07, .10, .10, .17, .24, .22, .10])
    if shape == (3, 3):
        names.append("center_heavy")
        weights = [w * .82 for w in weights] + [.18]
    if uneven:
        weights[0] = 0
    weights = np.array(weights) / sum(weights)
    return str(rng.choice(names, p=weights))


def _axis(rng, length, mode, peak=None):
    if mode == "uniform":
        return np.full(length, int(rng.integers(2, 7)), dtype=np.int64)
    if mode == "peak":
        bands = rng.integers(1, 5, size=length, dtype=np.int64)
        bands[int(rng.integers(length)) if peak is None else peak] = int(rng.integers(8, 12))
        return bands
    return rng.integers(2, 8, size=length, dtype=np.int64)


def _scales(rng, shape, profile):
    height, width = shape
    for _ in range(300):
        if profile == "uniform":
            rows, columns = _axis(rng, height, "uniform"), _axis(rng, width, "uniform")
        elif profile == "row_uniform":
            rows, columns = _axis(rng, height, "uniform"), _axis(rng, width, "mild")
        elif profile == "column_uniform":
            rows, columns = _axis(rng, height, "mild"), _axis(rng, width, "uniform")
        elif profile == "mild":
            rows, columns = _axis(rng, height, "mild"), _axis(rng, width, "mild")
        elif profile == "strong":
            if rng.random() < .5:
                rows, columns = _axis(rng, height, "peak"), _axis(rng, width, "mild")
            else:
                rows, columns = _axis(rng, height, "mild"), _axis(rng, width, "peak")
        elif profile in ("double_peak", "center_heavy"):
            center = profile == "center_heavy"
            rows = _axis(rng, height, "peak", 1 if center else None)
            columns = _axis(rng, width, "peak", 1 if center else None)
        else:
            rows, columns = _axis(rng, height, "mild"), _axis(rng, width, "mild")
            if rng.random() < .5:
                rows[int(rng.integers(height))] = 1
            else:
                columns[int(rng.integers(width))] = 1
        if rows.sum() > 28 or columns.sum() > 28:
            continue
        if profile == "row_uniform" and len(set(columns)) == 1:
            continue
        if profile == "column_uniform" and len(set(rows)) == 1:
            continue
        if profile == "mild" and len(set(rows)) == len(set(columns)) == 1:
            continue
        return rows, columns
    raise ValueError("Could not fit enlarged mosaic")


def _embed(rng, latent, rows, columns, background):
    expanded = np.repeat(np.repeat(latent, rows, axis=0), columns, axis=1)
    height, width = expanded.shape
    if height <= 19 and width <= 19 and rng.random() < .13:
        extra_height, extra_width = 21 - height, 21 - width
    else:
        extra_height = int(rng.integers(2, min(12, 30 - height) + 1))
        extra_width = int(rng.integers(2, min(12, 30 - width) + 1))
    top = int(rng.integers(1, extra_height))
    left = int(rng.integers(1, extra_width))
    bottom, right = extra_height - top, extra_width - left
    if rng.random() < .29:
        side = int(rng.integers(4))
        if side == 0:
            top, bottom = 0, extra_height
        elif side == 1:
            top, bottom = extra_height, 0
        elif side == 2:
            left, right = 0, extra_width
        else:
            left, right = extra_width, 0
    canvas = np.full((top + height + bottom, left + width + right),
                     background, dtype=np.int8)
    canvas[top:top + height, left:left + width] = expanded
    return canvas


def _digest(pair):
    source = np.asarray(pair["input"], dtype=np.uint8)
    target = np.asarray(pair["output"], dtype=np.uint8)
    return hashlib.sha256(bytes(source.shape) + source.tobytes()
                          + bytes(target.shape) + target.tobytes()).hexdigest()


def _signature(grid):
    return grid.shape, grid.tobytes()


def _recover(input_grid):
    """Infer the output using input pixels only."""
    background = int(input_grid[0, 0])
    assert input_grid[0, -1] == input_grid[-1, 0] == input_grid[-1, -1] == background
    mask = input_grid != background
    assert bool(np.any(mask))
    occupied_rows, occupied_columns = np.nonzero(mask)
    top, bottom = int(occupied_rows.min()), int(occupied_rows.max()) + 1
    left, right = int(occupied_columns.min()), int(occupied_columns.max()) + 1
    crop = input_grid[top:bottom, left:right]
    assert bool(np.all(crop != background)) and np.count_nonzero(mask) == crop.size
    margins = (top, input_grid.shape[0] - bottom, left, input_grid.shape[1] - right)
    assert sum(m == 0 for m in margins) <= 1
    row_starts = np.r_[0, np.flatnonzero(np.any(crop[1:] != crop[:-1], axis=1)) + 1]
    col_starts = np.r_[0, np.flatnonzero(np.any(crop[:, 1:] != crop[:, :-1], axis=0)) + 1]
    output = crop[np.ix_(row_starts, col_starts)]
    row_bands = np.diff(np.r_[row_starts, crop.shape[0]])
    col_bands = np.diff(np.r_[col_starts, crop.shape[1]])
    assert np.array_equal(np.repeat(np.repeat(output, row_bands, axis=0),
                                    col_bands, axis=1), crop)
    assert _visible_boundaries(output)
    return output, background, tuple(map(int, row_bands)), tuple(map(int, col_bands)), margins


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    shared_background = bool(rng.random() < .45)
    puzzle_background = int(rng.integers(10)) if shared_background else None
    special_index = int(rng.integers(train_count))
    train_shapes = [_shape(rng, small=i == special_index) for i in range(train_count)]
    if train_shapes[1] == train_shapes[0]:
        train_shapes[1] = _shape(rng, exclude=train_shapes[0])
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    first_background = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            shape = train_shapes[index] if split == "train" else _shape(rng)
            special = split == "train" and index == special_index
            profile = _profile(rng, shape, "source_train" if split == "train" else "target_test",
                               uneven=special)
            kind = ("single_seam" if special else
                    str(rng.choice(("low", "random", "high", "symmetric"),
                                   p=(.27, .47, .20, .06) if split == "train"
                                   else (.33, .37, .18, .12))))
            if kind == "symmetric" and min(shape) != 3:
                kind = "random"
            for _ in range(64):
                if shared_background:
                    background = puzzle_background
                elif split == "train" and index == 1:
                    background = int(rng.choice([c for c in range(10) if c != first_background]))
                else:
                    background = int(rng.integers(10))
                latent = _latent(rng, shape, background, kind)
                rows, columns = _scales(rng, shape, profile)
                pair = {"input": _embed(rng, latent, rows, columns, background),
                        "output": latent.copy()}
                signature = _signature(pair["input"])
                if signature in used_inputs or _digest(pair) in _SOURCE_DIGESTS:
                    continue
                used_inputs.add(signature)
                puzzle[split].append(pair)
                if split == "train" and index == 0:
                    first_background = background
                break
            else:
                raise ValueError("Could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, train_shapes = set(), set(), set()
    train_seam, train_uneven, train_seam_uneven = False, False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape != output_grid.shape
            assert _digest(pair) not in _SOURCE_DIGESTS
            input_signature = _signature(input_grid)
            pair_signature = (input_signature, _signature(output_grid))
            assert input_signature not in inputs and pair_signature not in pairs
            inputs.add(input_signature)
            pairs.add(pair_signature)
            recovered, background, rows, columns, _ = _recover(input_grid)
            assert np.array_equal(recovered, output_grid)
            assert background not in output_grid
            assert output_grid.shape in _SHAPES
            assert max(rows) <= 11 and max(columns) <= 11
            if split == "train":
                train_shapes.add(output_grid.shape)
                seam = _single_witness_seam(output_grid)
                uneven = len(set(rows)) > 1 or len(set(columns)) > 1
                train_seam |= seam
                train_uneven |= uneven
                train_seam_uneven |= seam and uneven
    assert len(train_shapes) >= 2
    assert train_seam and train_uneven and train_seam_uneven


def _equal_puzzles(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[split], second[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    first = generate_puzzle(314159)
    assert _equal_puzzles(first, generate_puzzle(314159))
    assert not _equal_puzzles(first, generate_puzzle(314160))
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None), (7, 3))):
        puzzle = generate_puzzle(20000 + offset, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for offset, (train, test) in enumerate(((1, None), (11, None),
                                            (None, 0), (None, 5),
                                            (2, 2), (3, 3), (2, 4), (4, 4))):
        try:
            generate_puzzle(30000 + offset, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")

    counts, shapes, backgrounds, palettes = set(), set(), set(), set()
    split_shapes = {"train": set(), "test": set()}
    split_palettes = {"train": set(), "test": set()}
    split_scales = {"train": set(), "test": set()}
    canvases, profiles, outputs = set(), set(), set()
    stats = {name: 0 for name in ("shared", "varying", "zero_margin", "size21",
                                  "train_seam", "train_one_pixel", "train_center",
                                  "test_center", "test_symmetric", "test_asymmetric",
                                  "test_high_palette", "train_low_3x3", "large_band")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        local_backgrounds = set()
        local_palettes = set()
        local_profiles = set()
        local_shapes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                recovered, bg, rows, columns, margins = _recover(pair["input"])
                assert np.array_equal(recovered, pair["output"])
                local_backgrounds.add(bg)
                backgrounds.add(bg)
                shapes.add(recovered.shape)
                palette_size = len(np.unique(recovered))
                palettes.add(palette_size)
                local_palettes.add(palette_size)
                local_profiles.add((rows, columns))
                local_shapes.add(recovered.shape)
                split_shapes[split].add(recovered.shape)
                split_palettes[split].add(palette_size)
                canvases.add(pair["input"].shape)
                outputs.add(_signature(recovered))
                profiles.add((rows, columns))
                row_uniform, column_uniform = len(set(rows)) == 1, len(set(columns)) == 1
                split_scales[split].add((row_uniform, column_uniform))
                stats["zero_margin"] += 0 in margins
                stats["size21"] += pair["input"].shape == (21, 21)
                stats["large_band"] += max(rows + columns) >= 8
                if split == "train":
                    stats["train_seam"] += _single_witness_seam(recovered)
                    stats["train_one_pixel"] += 1 in rows + columns
                    stats["train_low_3x3"] += recovered.shape == (3, 3) and len(np.unique(recovered)) <= 4
                    stats["train_center"] += (recovered.shape == (3, 3) and
                                               rows[1] > max(rows[0], rows[2]) and
                                               columns[1] > max(columns[0], columns[2]))
                else:
                    stats["test_symmetric"] += _symmetric(recovered)
                    stats["test_asymmetric"] += not _symmetric(recovered)
                    stats["test_high_palette"] += len(np.unique(recovered)) >= 6
                    stats["test_center"] += (recovered.shape == (3, 3) and
                                              rows[1] > max(rows[0], rows[2]) and
                                              columns[1] > max(columns[0], columns[2]))
        stats["shared" if len(local_backgrounds) == 1 else "varying"] += 1
        assert len(local_shapes) >= 2 and len(local_profiles) >= 2
        assert len(local_palettes) >= 2
    assert len({a for a, _ in counts}) >= 4 and len({b for _, b in counts}) >= 3
    assert len(shapes) == len(_SHAPES) and len(backgrounds) >= 8
    assert len(palettes) >= 7 and len(canvases) >= 60
    assert len(profiles) >= 80 and len(outputs) >= 100
    assert all(split_shapes[split] == set(_SHAPES) for split in ("train", "test"))
    assert all(len(split_palettes[split]) >= 7 for split in ("train", "test"))
    assert all(len(split_scales[split]) == 4 for split in ("train", "test"))
    assert all(value > 0 for value in stats.values()), stats
