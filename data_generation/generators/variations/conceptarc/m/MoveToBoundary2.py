import numpy as np


# Exact source inputs. Source dimensions, colors, and structures remain legal.
_SOURCE_INPUTS = {
    "0000000000000000000000000700000700070000070007007007000700700700070000000007000000000700000000000000",
    "0000000000000000000007777700000000000000000007777700000777700000000000077777777000000000000000000000",
    "0000000000000700000007070070000707007000070000700007000070000700007000070700700007070000000000000000",
    "0000000000000000000000000000000777777700000000000007007770000000000000000000000000000000000000000000",
    "0000000000070000000700000000070000000000070000000007000000000000070000000007000000000000000000000000",
    "0000000000000000000000000000000707770000000000000000000000000070070077000000000000000000000000000000",
}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("Illegal pair count")
    if num_test is None:
        upper = 4 if num_train is None else min(4, num_train - 1)
        num_test = int(rng.integers(1, upper + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("The train count must exceed the test count")
    return int(num_train), int(num_test)


def _palette(rng):
    if rng.random() < 0.18:
        return 0, 7
    background = int(rng.integers(0, 10))
    foreground = int(rng.integers(0, 9))
    if foreground >= background:
        foreground += 1
    return background, foreground


def _dimensions(rng):
    def side():
        tier = int(rng.choice(3, p=(0.40, 0.40, 0.20)))
        return int(rng.integers(*((7, 14), (14, 23), (23, 31))[tier]))
    return side(), side()


def _runs(cells):
    return np.split(cells, np.flatnonzero(np.diff(cells) > 1) + 1)


def _orientation(mask):
    vertical = int(np.sum(mask[:-1, :] & mask[1:, :]))
    horizontal = int(np.sum(mask[:, :-1] & mask[:, 1:]))
    if vertical >= 1.35 * horizontal + 2:
        return "vertical"
    if horizontal >= 1.35 * vertical + 2:
        return "horizontal"
    return None


def _has_gap(mask, mode):
    axis = 0 if mode == "vertical" else 1
    for line in np.flatnonzero(mask.any(axis=axis)):
        cells = np.flatnonzero(mask[:, line] if mode == "vertical" else mask[line, :])
        if len(_runs(cells)) > 1:
            return True
    return False


def _source_copy(grid):
    return (
        grid.shape == (10, 10)
        and set(map(int, np.unique(grid))) == {0, 7}
        and "".join(str(int(value)) for value in grid.flat) in _SOURCE_INPUTS
    )


def _solid(rng, length, anchor=False, packed=False):
    minimum = max(4, (length + 2) // 3) if anchor else 2
    maximum = max(minimum, min(length - 1, int(np.ceil(0.85 * length))))
    count = int(rng.integers(minimum, maximum + 1))
    start = length - count if packed else int(
        rng.integers(0, length - count + (0 if anchor else 1))
    )
    return np.arange(start, start + count)


def _broken(rng, length):
    for _ in range(100):
        pieces = int(rng.integers(2, 4))
        max_piece = max(2, min(9, (length + 1) // 2))
        lengths = rng.integers(1, max_piece + 1, size=pieces)
        if rng.random() < 0.55:
            lengths[int(rng.integers(pieces))] = min(max_piece, max(3, length // 3))
        gaps = rng.integers(1, min(7, length - 2) + 1, size=pieces - 1)
        span = int(sum(lengths) + sum(gaps))
        if span > length:
            continue
        cursor = int(rng.integers(0, length - span + 1))
        cells = []
        for index, run_length in enumerate(lengths):
            cells.extend(range(cursor, cursor + int(run_length)))
            cursor += int(run_length)
            if index < pieces - 1:
                cursor += int(gaps[index])
        return np.asarray(cells, dtype=int)
    return np.asarray((0, 1, length - 2, length - 1), dtype=int)


def _line_cells(rng, length, texture):
    if texture == "broken":
        return _broken(rng, length)
    if texture == "short":
        count = int(rng.integers(1, 3))
        start = int(rng.integers(0, length - count + 1))
        return np.arange(start, start + count)
    return _solid(rng, length, packed=texture == "packed")


def _pack(grid, background, foreground, mode):
    output = np.full_like(grid, background)
    mask = grid == foreground
    for line, count in enumerate(mask.sum(axis=0 if mode == "vertical" else 1)):
        if count:
            if mode == "vertical":
                output[-int(count):, line] = foreground
            else:
                output[line, -int(count):] = foreground
    return output


def _pair(rng, regime, palette, mode, shape, require_closure):
    background, foreground = palette
    height, width = shape
    cross_length = width if mode == "vertical" else height
    along_length = height if mode == "vertical" else width
    maximum = max(2, min(cross_length - 1, int(np.ceil(0.58 * cross_length))))
    for _ in range(300):
        grid = np.full(shape, background, dtype=np.int8)
        count = int(rng.integers(2, maximum + 1))
        want_adjacent = rng.random() < 0.58
        for _ in range(80):
            occupied = np.sort(rng.choice(cross_length, size=count, replace=False))
            if not want_adjacent or np.any(np.diff(occupied) == 1):
                break
        else:
            continue
        style = str(rng.choice(
            ("solid", "mixed", "fragmented"),
            p=(0.27, 0.51, 0.22) if regime == "source_train"
            else (0.14, 0.44, 0.42),
        ))
        if require_closure and style == "solid":
            style = "mixed"
        anchor = int(rng.integers(count))
        others = [i for i in range(count) if i != anchor]
        forced_broken = int(rng.choice(others)) if style != "solid" else -1
        weights = {
            "solid": (0.70, 0.30, 0.0, 0.0),
            "mixed": (0.28, 0.19, 0.41, 0.12),
            "fragmented": (0.12, 0.10, 0.68, 0.10),
        }[style]
        for index, line in enumerate(occupied):
            if index == anchor:
                cells = _solid(rng, along_length, anchor=True)
            elif index == forced_broken:
                cells = _broken(rng, along_length)
            else:
                texture = str(rng.choice(("solid", "packed", "broken", "short"), p=weights))
                cells = _line_cells(rng, along_length, texture)
            if mode == "vertical":
                grid[cells, line] = foreground
            else:
                grid[line, cells] = foreground
        mask = grid == foreground
        if mask.mean() >= 0.45 or _orientation(mask) != mode:
            continue
        if require_closure and not _has_gap(mask, mode):
            continue
        output = _pack(grid, background, foreground, mode)
        if np.array_equal(grid, output) or _source_copy(grid):
            continue
        return grid, output
    raise RuntimeError("Could not construct an unambiguous pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    schedule = str(rng.choice(("vertical", "horizontal", "mixed"), p=(0.25, 0.25, 0.50)))
    if schedule == "mixed":
        train_modes = ["vertical", "horizontal"] + [
            str(rng.choice(("vertical", "horizontal"))) for _ in range(train_count - 2)
        ]
        rng.shuffle(train_modes)
    else:
        train_modes = [schedule] * train_count
    test_modes = [str(rng.choice(tuple(sorted(set(train_modes))))) for _ in range(test_count)]

    palette_policy = str(rng.choice(("fixed", "per_pair"), p=(0.58, 0.42)))
    fixed_palette = _palette(rng) if palette_policy == "fixed" else None
    canvas_policy = str(rng.choice(("source_10", "fixed_other", "variable"), p=(0.21, 0.19, 0.60)))
    fixed_shape = (
        (10, 10) if canvas_policy == "source_10" else
        _dimensions(rng) if canvas_policy == "fixed_other" else None
    )
    if canvas_policy == "fixed_other" and fixed_shape == (10, 10):
        fixed_shape = (10, 11)

    puzzle = {"train": [], "test": []}
    seen_inputs, used_shapes, demonstrated = set(), set(), set()
    for regime, modes, destination in (
        ("source_train", train_modes, "train"),
        ("target_test", test_modes, "test"),
    ):
        for mode in modes:
            first_train_mode = regime == "source_train" and mode not in demonstrated
            require_closure = first_train_mode or (regime == "target_test" and not puzzle["test"])
            if first_train_mode:
                demonstrated.add(mode)
            for _ in range(300):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                shape = fixed_shape if fixed_shape is not None else _dimensions(pair_rng)
                if canvas_policy == "variable" and shape in used_shapes:
                    continue
                palette = fixed_palette if fixed_palette is not None else _palette(pair_rng)
                grid, output = _pair(pair_rng, regime, palette, mode, shape, require_closure)
                key = (grid.shape, grid.tobytes())
                if key in seen_inputs:
                    continue
                seen_inputs.add(key)
                used_shapes.add(shape)
                puzzle[destination].append({"input": grid, "output": output})
                break
            else:
                raise RuntimeError("Could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    assert len(values) == 2
    background = int(values[np.argmax(frequencies)])
    foreground = int(values[np.argmin(frequencies)])
    assert int(np.min(frequencies)) < grid.size * 0.45
    return background, foreground


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, train_modes, train_closure, test_modes = (
        set(), set(), set(), set(), set()
    )
    test_closure = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for image in (grid, output):
                assert isinstance(image, np.ndarray) and image.ndim == 2
                assert 1 <= image.shape[0] <= 30 and 1 <= image.shape[1] <= 30
                assert np.issubdtype(image.dtype, np.integer)
                assert np.all((0 <= image) & (image <= 9))
            assert grid.shape == output.shape and not np.array_equal(grid, output)
            background, foreground = _roles(grid)
            assert set(map(int, np.unique(output))) == {background, foreground}
            assert not _source_copy(grid)
            input_key = (grid.shape, grid.tobytes())
            pair_key = (input_key, output.shape, output.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            mask = grid == foreground
            mode = _orientation(mask)
            assert mode is not None
            if split == "train":
                train_modes.add(mode)
                if _has_gap(mask, mode):
                    train_closure.add(mode)
            else:
                test_modes.add(mode)
                test_closure |= _has_gap(mask, mode)
            assert np.array_equal(output, _pack(grid, background, foreground, mode))
            axis = 0 if mode == "vertical" else 1
            occupied = np.flatnonzero(mask.any(axis=axis))
            assert 2 <= len(occupied) <= max(
                2, int(np.ceil(0.58 * mask.shape[1 - axis]))
            )
            assert any(
                len(_runs(np.flatnonzero(mask[:, line] if mode == "vertical" else mask[line, :]))) == 1
                and int(mask[:, line].sum() if mode == "vertical" else mask[line, :].sum()) >= 4
                for line in occupied
            )
    assert test_modes <= train_modes and test_modes <= train_closure
    assert test_closure


def _same_puzzle(first, second):
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(pair_a[key], pair_b[key])
            for pair_a, pair_b in zip(first[split], second[split])
            for key in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes, fingerprints = set(), set(), set(), set()
    schedules, canvases = set(), set()
    adjacent = thick = long_gaps = solid_train = mixed_test = singleton = 0
    large_line_counts = recolored = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        modes, puzzle_palettes, puzzle_shapes, fingerprint = [], set(), set(), []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, foreground = _roles(grid)
                palette = (background, foreground)
                palettes.add(palette)
                puzzle_palettes.add(palette)
                shapes.add(grid.shape)
                puzzle_shapes.add(grid.shape)
                fingerprint.append((grid.shape, grid.tobytes()))
                mask = grid == foreground
                mode = _orientation(mask)
                modes.append(mode)
                axis = 0 if mode == "vertical" else 1
                occupied = np.flatnonzero(mask.any(axis=axis))
                layouts.add((mode, tuple(occupied), tuple(int(x) for x in mask.sum(axis=axis))))
                adjacent += int(np.any(np.diff(occupied) == 1))
                large_line_counts += int(len(occupied) >= 6)
                line_runs = [
                    _runs(np.flatnonzero(mask[:, line] if mode == "vertical" else mask[line, :]))
                    for line in occupied
                ]
                solid_train += int(split == "train" and all(len(runs) == 1 for runs in line_runs))
                mixed_test += int(split == "test" and any(len(runs) == 1 for runs in line_runs)
                                  and any(len(runs) > 1 for runs in line_runs))
                singleton += sum(any(len(run) == 1 for run in runs) for runs in line_runs)
                long_gaps += sum(
                    any(runs[j + 1][0] - runs[j][-1] - 1 >= 3 for j in range(len(runs) - 1))
                    for runs in line_runs
                )
                if np.any(np.diff(occupied) == 1):
                    thick += int(any(
                        np.any(mask[:, a] & mask[:, a + 1]) if mode == "vertical"
                        else np.any(mask[a, :] & mask[a + 1, :])
                        for a in occupied[:-1] if a + 1 in occupied
                    ))
                expected = np.full_like(grid, background)
                counts = mask.sum(axis=axis)
                for line, count in enumerate(counts):
                    if count:
                        if mode == "vertical":
                            expected[-int(count):, line] = foreground
                        else:
                            expected[line, -int(count):] = foreground
                assert np.array_equal(expected, pair["output"])
        schedules.add(tuple(sorted(set(modes[:len(puzzle["train"])]))))
        canvases.add("fixed" if len(puzzle_shapes) == 1 else "variable")
        recolored += int(len(puzzle_palettes) > 1)
        fingerprints.add(tuple(fingerprint))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(layouts) >= 100
    assert len(palettes) >= 12 and len(fingerprints) == 42
    assert {("horizontal",), ("vertical",), ("horizontal", "vertical")} <= schedules
    assert canvases == {"fixed", "variable"}
    assert min(adjacent, thick, long_gaps, solid_train, mixed_test, singleton,
               large_line_counts, recolored) > 0

    cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))
    for index, (num_train, num_test) in enumerate(cases):
        puzzle = generate_puzzle(1000 + index, num_train, num_test)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + index, num_train, num_test))
        if num_train is not None:
            assert len(puzzle["train"]) == num_train
        if num_test is not None:
            assert len(puzzle["test"]) == num_test
    for args in ((0, 1, None), (0, 11, None), (0, None, 0),
                 (0, None, 5), (0, 2, 2), (0, 3, 4)):
        try:
            generate_puzzle(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
