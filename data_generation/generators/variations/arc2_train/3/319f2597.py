"""Generate conditional two-lane cross projection ARC puzzles."""

import hashlib

import numpy as np


_COLORS = tuple(range(10))
_SOURCE_INPUT_HASHES = {
    "e14241778636a53075f9699ba64382963613ea903bb8fb5b3949f3b594506f1f",
    "18cc5f39fe0a25c925d31b04e0711aa3a196aa165956472b0bbc1a1184b3c1d6",
    "a079d8c2cb34bbed1200c41dd0a610cc89a129a653c351a0f32e9f45a86dbc37",
    "2404efeee2f8646369f79315d80d0f07c4f000a23dd0a9d866e839a9194edb45",
}


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
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


def _cross_mask(shape, row, col):
    mask = np.zeros(shape, dtype=bool)
    mask[row:row + 2, :] = True
    mask[:, col:col + 2] = True
    return mask


def _marker_candidates(grid):
    """A marker color occurs exactly four times, all in one 2x2 square."""
    candidates = []
    for color in np.unique(grid):
        rr, cc = np.where(grid == color)
        if len(rr) != 4:
            continue
        row, col = int(rr.min()), int(cc.min())
        if int(rr.max()) == row + 1 and int(cc.max()) == col + 1:
            if np.all(grid[row:row + 2, col:col + 2] == color):
                candidates.append((int(color), row, col))
    return candidates


def _apply_rule(grid, marker, protected, row, col):
    output = grid.copy()
    cross = _cross_mask(grid.shape, row, col)
    output[cross & (grid != protected)] = marker
    return output


def _source_copy(grid):
    return grid.shape == (20, 20) and hashlib.sha256(
        np.asarray(grid, dtype=np.int8).tobytes()
    ).hexdigest() in _SOURCE_INPUT_HASHES


def _sample_shape(rng, split):
    if rng.random() < 0.08:
        return 20, 20
    kind = int(rng.choice(4, p=(0.24, 0.24, 0.20, 0.32) if split == "test"
                          else (0.20, 0.17, 0.25, 0.38)))
    if kind == 0:
        return int(rng.integers(6, 14)), int(rng.integers(6, 14))
    if kind == 1:
        short, long = int(rng.integers(6, 11)), int(rng.integers(19, 31))
        return (short, long) if rng.integers(2) else (long, short)
    if kind == 2:
        return int(rng.integers(18, 31)), int(rng.integers(18, 31))
    return int(rng.integers(7, 31)), int(rng.integers(7, 31))


def _zone(top, length):
    return min(2, 3 * top // (length - 1))


def _sample_top(rng, length, zone):
    choices = [top for top in range(length - 1) if _zone(top, length) == zone]
    if rng.random() < 0.38:
        return int(choices[0] if rng.integers(2) else choices[-1])
    return int(rng.choice(choices))


def _stamp_texture(grid, rng, colors):
    height, width = grid.shape
    for _ in range(int(rng.integers(0, 6))):
        color = int(rng.choice(colors))
        kind = int(rng.integers(4))
        if kind == 0:  # patch
            hh = int(rng.integers(2, min(6, height) + 1))
            ww = int(rng.integers(2, min(6, width) + 1))
            rr = int(rng.integers(height - hh + 1))
            cc = int(rng.integers(width - ww + 1))
            grid[rr:rr + hh, cc:cc + ww] = color
        elif kind == 1:  # short run
            rr, cc = int(rng.integers(height)), int(rng.integers(width))
            if rng.integers(2):
                grid[rr, cc:min(width, cc + int(rng.integers(3, 10)))] = color
            else:
                grid[rr:min(height, rr + int(rng.integers(3, 10))), cc] = color
        elif kind == 2:  # stripe, sometimes broken
            if rng.integers(2):
                grid[int(rng.integers(height)), ::int(rng.integers(1, 4))] = color
            else:
                grid[::int(rng.integers(1, 4)), int(rng.integers(width))] = color
        else:  # irregular cluster
            rr, cc = int(rng.integers(height)), int(rng.integers(width))
            for _ in range(int(rng.integers(4, 14))):
                grid[rr, cc] = color
                rr = min(height - 1, max(0, rr + int(rng.integers(-1, 2))))
                cc = min(width - 1, max(0, cc + int(rng.integers(-1, 2))))


def _stamp_squares(grid, rng, row, col, color):
    height, width = grid.shape
    other_rows = [r for r in range(height - 1) if r + 1 < row or r > row + 1]
    other_cols = [c for c in range(width - 1) if c + 1 < col or c > col + 1]
    if other_rows and other_cols and rng.random() < 0.48:
        rr, cc = int(rng.choice(other_rows)), int(rng.choice(other_cols))
        grid[rr:rr + 2, cc:cc + 2] = color
    if other_rows and rng.random() < 0.46:
        rr = int(rng.choice(other_rows))
        # Only one column of this square lies in the vertical lane.
        cc = col + 1 if col + 2 < width and (col == 0 or rng.integers(2)) else col - 1
        grid[rr:rr + 2, cc:cc + 2] = color


def _try_frequency_tie(grid, rng, protected, marker, cross):
    """Sometimes tie protection with the dominant distractor using off-cross cells."""
    trial = grid.copy()
    counts = np.bincount(trial.ravel(), minlength=10)
    distractors = [c for c in _COLORS if c not in (marker, protected) and counts[c]]
    if len(distractors) < 2:
        return grid
    dominant = max(distractors, key=lambda c: counts[c])
    gap = int(counts[dominant] - counts[protected])
    if abs(gap) > trial.size // 3:
        return grid
    if gap % 2:
        extra = np.argwhere(~cross & (trial != marker) & (trial != protected) & (trial != dominant))
        if len(extra) == 0:
            return grid
        rr, cc = extra[int(rng.integers(len(extra)))]
        trial[rr, cc] = protected if gap > 0 else dominant
        gap += -1 if gap > 0 else 1
    source = dominant if gap > 0 else protected
    destination = protected if gap > 0 else dominant
    eligible = np.argwhere(~cross & (trial == source))
    needed = abs(gap) // 2
    if len(eligible) < needed:
        return grid
    if needed:
        for rr, cc in eligible[rng.choice(len(eligible), size=needed, replace=False)]:
            trial[rr, cc] = destination
    revised = np.bincount(trial.ravel(), minlength=10)
    if revised[protected] == revised[dominant] == revised.max():
        return trial
    return grid


def _make_pair(rng, shape, position_zone, marker, protected, split, index):
    height, width = shape
    row = _sample_top(rng, height, position_zone[0])
    col = _sample_top(rng, width, position_zone[1])
    cross = _cross_mask(shape, row, col)
    block = np.zeros(shape, dtype=bool)
    block[row:row + 2, col:col + 2] = True
    distractors = tuple(int(x) for x in rng.choice(
        [c for c in _COLORS if c not in (marker, protected)],
        size=int(rng.integers(2, 9)), replace=False
    ))
    cross_only = len(distractors) >= 3 and rng.random() < 0.31
    cross_color = distractors[-1] if cross_only else None
    base_colors = distractors[:-1] if cross_only else distractors
    modes = ("sparse", "medium", "abundant") if split == "train" and index == 0 else (
        "absent", "offcross", "sparse", "medium", "abundant"
    )
    mode = str(rng.choice(modes))
    for _ in range(16):
        protected_probability = {
            "absent": 0.0,
            "offcross": 0.0,
            "sparse": float(rng.uniform(0.005, 0.035)),
            "medium": float(rng.uniform(0.065, 0.19)),
            "abundant": float(rng.uniform(0.22, 0.39)),
        }[mode]
        weights = rng.dirichlet(np.full(len(base_colors), float(rng.uniform(0.35, 1.7))))
        palette = list(base_colors)
        probabilities = list((1 - protected_probability) * weights)
        if protected_probability:
            palette.append(protected)
            probabilities.append(protected_probability)
        grid = rng.choice(palette, size=shape, p=probabilities).astype(np.int8)
        _stamp_texture(grid, rng, palette)
        _stamp_squares(grid, rng, row, col, int(base_colors[0]))
        if mode == "offcross":
            grid[grid == protected] = base_colors[0]
            off = np.argwhere(~cross)
            for rr, cc in off[rng.choice(len(off), size=min(3, len(off)), replace=False)]:
                grid[rr, cc] = protected
        elif mode == "absent":
            grid[grid == protected] = base_colors[0]
        elif rng.random() < 0.37:  # a longer protected run, never a barrier
            if rng.integers(2):
                rr, cc = row + int(rng.integers(2)), int(rng.integers(width))
                grid[rr, cc:min(width, cc + int(rng.integers(4, 10)))] = protected
            else:
                rr, cc = int(rng.integers(height)), col + int(rng.integers(2))
                grid[rr:min(height, rr + int(rng.integers(4, 10))), cc] = protected
        if split == "train" and index == 0:
            # One demonstration identifies protection in both directions.
            row_cells = [(r, c) for r in (row, row + 1) for c in range(width)
                         if not block[r, c]]
            col_cells = [(r, c) for c in (col, col + 1) for r in range(height)
                         if not block[r, c]]
            for cells in (row_cells, col_cells):
                for take in rng.choice(len(cells), size=2, replace=False):
                    grid[cells[int(take)]] = protected
        if cross_color is not None:
            available = np.argwhere(cross & ~block & (grid != protected))
            if len(available) >= 6:
                for rr, cc in available[rng.choice(len(available), size=min(9, len(available)), replace=False)]:
                    grid[rr, cc] = cross_color
        grid[block] = marker
        if mode in ("medium", "abundant") and rng.random() < 0.18:
            grid = _try_frequency_tie(grid, rng, protected, marker, cross)
        if _marker_candidates(grid) != [(marker, row, col)]:
            continue
        output = _apply_rule(grid, marker, protected, row, col)
        if np.array_equal(grid, output) or _source_copy(grid):
            continue
        return grid, output
    raise RuntimeError("failed to construct a unique marker pair")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _learn_protected(train):
    surviving = set()
    for pair in train:
        candidates = _marker_candidates(pair["input"])
        assert len(candidates) == 1
        marker, row, col = candidates[0]
        cross = _cross_mask(pair["input"].shape, row, col)
        unchanged = cross & (pair["input"] != marker) & (pair["input"] == pair["output"])
        surviving.update(int(c) for c in np.unique(pair["input"][unchanged]))
    assert len(surviving) == 1, "training must identify one protected color"
    return next(iter(surviving))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    protected = int(rng.integers(10))
    marker_colors = [c for c in _COLORS if c != protected]
    rng.shuffle(marker_colors)
    varying_markers = bool(rng.integers(2))
    markers = ([marker_colors[i % len(marker_colors)] for i in range(num_train + num_test)]
               if varying_markers else [marker_colors[0]] * (num_train + num_test))
    held_out = bool(rng.integers(2))
    if held_out:
        target = (int(rng.integers(3)), int(rng.integers(3)))
        other_row = (target[0] + int(rng.integers(1, 3))) % 3
        other_col = (target[1] + int(rng.integers(1, 3))) % 3
        train_zones = [(target[0], other_col), (other_row, target[1])]
        choices = [(r, c) for r in range(3) for c in range(3) if (r, c) != target]
        train_zones += [choices[int(rng.integers(len(choices)))] for _ in range(num_train - 2)]
        test_zones = [target] * num_test
    else:
        choices = [(r, c) for r in range(3) for c in range(3)]
        chosen = rng.choice(len(choices), size=2, replace=False)
        train_zones = [choices[int(i)] for i in chosen]
        train_zones += [choices[int(rng.integers(9))] for _ in range(num_train - 2)]
        test_zones = []
        for _ in range(num_test):
            if rng.random() < 0.72:
                test_zones.append(train_zones[int(rng.integers(num_train))])
            else:
                test_zones.append((train_zones[int(rng.integers(num_train))][0],
                                   train_zones[int(rng.integers(num_train))][1]))
    same_shape = bool(rng.random() < 0.28)
    common_shape = _sample_shape(rng, "train") if same_shape else None
    puzzle = {"train": [], "test": []}
    used = set()
    pair_index = 0
    for split, zones in (("train", train_zones), ("test", test_zones)):
        for index, zone in enumerate(zones):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(12):
                shape = common_shape if same_shape else _sample_shape(pair_rng, split)
                if not same_shape and split == "train" and index == 1:
                    if shape == puzzle["train"][0]["input"].shape:
                        continue
                input_grid, output_grid = _make_pair(
                    pair_rng, shape, zone, markers[pair_index], protected, split, index
                )
                key = _grid_key(input_grid)
                if key not in used:
                    used.add(key)
                    puzzle[split].append({"input": input_grid, "output": output_grid})
                    break
            else:
                raise RuntimeError("failed to construct a distinct pair")
            pair_index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _source_copy(inp)
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in inputs and (ikey, okey) not in pairs
            inputs.add(ikey)
            pairs.add((ikey, okey))
            assert len(_marker_candidates(inp)) == 1
    protected = _learn_protected(puzzle["train"])
    train_markers, train_zones = set(), set()
    for pair in puzzle["train"]:
        marker, row, col = _marker_candidates(pair["input"])[0]
        assert marker != protected
        train_markers.add(marker)
        train_zones.add((_zone(row, pair["input"].shape[0]),
                         _zone(col, pair["input"].shape[1])))
    assert len(train_zones) >= 2
    train_row_zones = {z[0] for z in train_zones}
    train_col_zones = {z[1] for z in train_zones}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            marker, row, col = _marker_candidates(inp)[0]
            assert marker != protected
            assert np.array_equal(_apply_rule(inp, marker, protected, row, col), out)
            if split == "test":
                zone = (_zone(row, inp.shape[0]), _zone(col, inp.shape[1]))
                assert zone[0] in train_row_zones and zone[1] in train_col_zones
                assert marker in train_markers or len(train_markers) >= 2


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, markers, protected_colors = set(), set(), set(), set()
    edge_count = corner_count = square_count = absent_count = cross_only_count = 0
    source_shape_count = extreme_count = on_square_count = off_square_count = 0
    protected_top_count = protected_tie_count = long_run_count = pooled_top_count = 0
    patch_count = distractor_run_count = 0
    heldout_puzzles = overlap_puzzles = varied_palette_puzzles = varied_density_puzzles = 0
    variable_marker_puzzles = constant_marker_puzzles = 0
    same_size_puzzles = varied_size_puzzles = 0
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        protected = _learn_protected(puzzle["train"])
        protected_colors.add(protected)
        pair_shapes, pair_markers = set(), set()
        puzzle_palettes, puzzle_densities = set(), set()
        train_zones, test_zones = set(), set()
        pooled_counts = np.zeros(10, dtype=int)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                marker, row, col = _marker_candidates(inp)[0]
                assert np.array_equal(out, _apply_rule(inp, marker, protected, row, col))
                total_pairs += 1
                pair_shapes.add(inp.shape)
                pair_markers.add(marker)
                shapes.add(inp.shape)
                palettes.add(len(np.unique(inp)))
                puzzle_palettes.add(len(np.unique(inp)))
                markers.add(marker)
                cross = _cross_mask(inp.shape, row, col)
                density = np.count_nonzero(inp == protected) / inp.size
                puzzle_densities.add(
                    "absent" if density == 0 else "rare" if density < .04
                    else "medium" if density < .2 else "abundant"
                )
                position_zone = (_zone(row, inp.shape[0]), _zone(col, inp.shape[1]))
                (train_zones if split == "train" else test_zones).add(position_zone)
                if split == "train":
                    pooled_counts += np.bincount(inp.ravel(), minlength=10)
                edge_count += row in (0, inp.shape[0] - 2) or col in (0, inp.shape[1] - 2)
                corner_count += row in (0, inp.shape[0] - 2) and col in (0, inp.shape[1] - 2)
                source_shape_count += inp.shape == (20, 20)
                extreme_count += max(inp.shape) / min(inp.shape) >= 2.5
                frequencies = np.bincount(inp.ravel(), minlength=10)
                protected_top_count += (
                    frequencies[protected] == frequencies.max()
                    and np.count_nonzero(frequencies == frequencies.max()) == 1
                )
                protected_tie_count += (
                    frequencies[protected] == frequencies.max()
                    and np.count_nonzero(frequencies == frequencies.max()) > 1
                )
                solid = (inp[:-1, :-1] == inp[1:, :-1]) & (inp[:-1, :-1] == inp[:-1, 1:]) & (inp[:-1, :-1] == inp[1:, 1:])
                square_count += int(np.count_nonzero(solid)) > 1
                patch_count += any(
                    np.all(inp[rr:rr + 3, cc:cc + 3] == inp[rr, cc])
                    for rr in range(inp.shape[0] - 2)
                    for cc in range(inp.shape[1] - 2)
                )
                for rr, cc in np.argwhere(solid):
                    if (int(rr), int(cc)) == (row, col):
                        continue
                    if inp[rr, cc] == protected:
                        continue
                    patch = np.zeros(inp.shape, dtype=bool)
                    patch[rr:rr + 2, cc:cc + 2] = True
                    if np.any(patch & cross):
                        on_square_count += 1
                    else:
                        off_square_count += 1
                absent_count += not np.any(inp == protected)
                cross_only_count += len(set(np.unique(inp)) - set(np.unique(out))) > 0
                longest = 0
                for line in (inp[row, :], inp[row + 1, :], inp[:, col], inp[:, col + 1]):
                    run = 0
                    for value in line:
                        run = run + 1 if value == protected else 0
                        longest = max(longest, run)
                long_run_count += longest >= 4
                distractor_run_count += any(
                    np.any((line[:-4] == line[1:-3]) & (line[:-4] == line[2:-2])
                           & (line[:-4] == line[3:-1]) & (line[:-4] == line[4:])
                           & (line[:-4] != marker) & (line[:-4] != protected))
                    for line in list(inp) + list(inp.T)
                )
        variable_marker_puzzles += len(pair_markers) > 1
        constant_marker_puzzles += len(pair_markers) == 1
        same_size_puzzles += len(pair_shapes) == 1
        varied_size_puzzles += len(pair_shapes) > 1
        varied_palette_puzzles += len(puzzle_palettes) > 1
        varied_density_puzzles += len(puzzle_densities) > 1
        heldout_puzzles += any(zone not in train_zones for zone in test_zones)
        overlap_puzzles += all(zone in train_zones for zone in test_zones)
        pooled_top_count += (
            pooled_counts[protected] == pooled_counts.max()
            and np.count_nonzero(pooled_counts == pooled_counts.max()) == 1
        )
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(palettes) >= 5
    assert len(markers) >= 8 and len(protected_colors) >= 8
    assert edge_count > 10 and corner_count > 0 and square_count > 10
    assert absent_count > 0 and cross_only_count > 0
    assert source_shape_count > 0 and extreme_count > 10
    assert on_square_count > 0 and off_square_count > 0 and long_run_count > 0
    assert patch_count > 0 and distractor_run_count > 0
    assert 0 < protected_top_count < total_pairs // 2
    assert protected_tie_count > 0
    assert pooled_top_count < 42 // 2
    assert heldout_puzzles > 5 and overlap_puzzles > 5
    assert varied_palette_puzzles > 20 and varied_density_puzzles > 20
    assert variable_marker_puzzles > 5 and constant_marker_puzzles > 5
    assert same_size_puzzles > 2 and varied_size_puzzles > 10
    for seed, train_count, test_count in ((1234, 2, 1), (1235, 10, 1), (1236, 10, 4), (1237, 7, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train_count, test_count))
    for arguments in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(4321, *arguments)
        assert (arguments[0] is None or len(puzzle["train"]) == arguments[0])
        assert (arguments[1] is None or len(puzzle["test"]) == arguments[1])
    assert _puzzles_equal(generate_puzzle(24680), generate_puzzle(24680))
    assert not _puzzles_equal(generate_puzzle(444, 5, 2), generate_puzzle(445, 5, 2))
    for arguments in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
