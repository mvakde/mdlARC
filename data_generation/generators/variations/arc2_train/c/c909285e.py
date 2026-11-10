"""Synthetic puzzles for inclusive cropping of an added square border."""

import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    "3e63dd0bc54fa4a0fea1f65ed542b7bd792a9020978c55f6a639343e4056740e",
    "b9a58ede01452b08e30fef31ba239f47bd7c1ddefac8656cb44ffba4cbaf8b3c",
    "3fe8680a1c772980ed92db7d0efccb097ceacb8334152fce5ce4ee10cdb283ee",
    "104c4d508cc9f4cbbf499ba96776a08ce6a71585f00da4631f73d69fd93faabe",
}


def _seed_value(seed):
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed_value, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        num_train = num_test + 1 + (seed_value // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (seed_value // 4) % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed_value % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _source_digest(pair):
    return hashlib.sha256(
        pair["input"].astype(np.int8, copy=False).tobytes()
        + pair["output"].astype(np.int8, copy=False).tobytes()
    ).hexdigest()


def _frames(grid):
    """All complete, one-cell monochrome square perimeters, side >= 4."""
    height, width = grid.shape
    horizontal = np.ones(grid.shape, dtype=np.int16)
    vertical = np.ones(grid.shape, dtype=np.int16)
    for column in range(width - 2, -1, -1):
        horizontal[:, column] += (grid[:, column] == grid[:, column + 1]) * horizontal[:, column + 1]
    for row in range(height - 2, -1, -1):
        vertical[row, :] += (grid[row, :] == grid[row + 1, :]) * vertical[row + 1, :]
    result = []
    for side in range(4, min(height, width) + 1):
        for row in range(height - side + 1):
            bottom = row + side - 1
            for column in range(width - side + 1):
                right = column + side - 1
                if (horizontal[row, column] >= side
                        and horizontal[bottom, column] >= side
                        and vertical[row, column] >= side
                        and vertical[row, right] >= side):
                    result.append((row, column, side, int(grid[row, column])))
    return result


def _perimeter(shape, frame):
    row, column, side, _ = frame
    mask = np.zeros(shape, dtype=bool)
    mask[row, column:column + side] = True
    mask[row + side - 1, column:column + side] = True
    mask[row:row + side, column] = True
    mask[row:row + side, column + side - 1] = True
    return mask


def _color_outside_perimeter(grid, frame):
    return bool(np.any(grid[~_perimeter(grid.shape, frame)] == frame[3]))


def _fits_repeated_field(grid, mask):
    """Can visible cells be explained by a short, fully observed 2-D rhythm?"""
    height, width = grid.shape
    for rows in range(4, 8):
        for columns in range(4, 8):
            if rows > height // 2 or columns > width // 2:
                continue
            okay = True
            for row in range(rows):
                for column in range(columns):
                    cells = grid[row::rows, column::columns]
                    visible = ~mask[row::rows, column::columns]
                    values = cells[visible]
                    if values.size < 2 or np.any(values != values[0]):
                        okay = False
                        break
                if not okay:
                    break
            if okay:
                return True
    return False


def _plausible_frames(grid, frames):
    """Candidates supported by color novelty or a uniquely repaired rhythm."""
    novel = [f for f in frames if not _color_outside_perimeter(grid, f)]
    if novel:
        return novel
    return [f for f in frames if _fits_repeated_field(grid, _perimeter(grid.shape, f))]


def _palette(rng, marker, previous_marker, reuse):
    size = int(rng.integers(2, 8))
    required = {marker} if reuse else set()
    if previous_marker is not None and previous_marker != marker:
        required.add(previous_marker)
    available = [c for c in range(10) if c != marker or reuse]
    size = max(size, len(required))
    remaining = [c for c in available if c not in required]
    colors = list(required) + list(rng.choice(remaining, size - len(required), replace=False))
    rng.shuffle(colors)
    return np.array(colors, dtype=np.int8)


def _rhythm_field(rng, shape, palette, mode, reused):
    if mode == "outline":
        rows = int(rng.integers(9, 13))
        columns = int(rng.integers(9, 13))
    elif reused:
        rows = int(rng.integers(4, 8))
        columns = int(rng.integers(4, 8))
    else:
        rows = int(rng.integers(6, 11))
        columns = int(rng.integers(6, 11))
    colors = len(palette)
    row_codes = rng.integers(0, colors, size=rows)
    col_codes = rng.integers(0, colors, size=columns)
    if rows > 5:
        row_codes[1] = row_codes[0]
    if columns > 5:
        col_codes[1] = col_codes[0]
    tile = palette[(row_codes[:, None] + col_codes[None, :]) % colors].copy()
    for row in rng.choice(rows, size=max(1, rows // 5), replace=False):
        tile[row, :] = palette[int(row_codes[row])]
    if mode == "outline":
        side = int(rng.integers(4, 7))
        row = int(rng.integers(1, rows - side + 1))
        column = int(rng.integers(1, columns - side + 1))
        color = int(rng.choice(palette))
        tile[row, column:column + side] = color
        tile[row + side - 1, column:column + side] = color
        tile[row:row + side, column] = color
        tile[row:row + side, column + side - 1] = color
    row_phase = int(rng.integers(0, rows))
    col_phase = int(rng.integers(0, columns))
    rr = (np.arange(shape[0]) + row_phase) % rows
    cc = (np.arange(shape[1]) + col_phase) % columns
    return tile[rr[:, None], cc[None, :]].copy()


def _block_field(rng, shape, palette):
    """Irregular wide row and column bands with recurring color roles."""
    height, width = shape
    row_sizes = []
    column_sizes = []
    while sum(row_sizes) < height:
        row_sizes.append(int(rng.integers(6, 11)))
    while sum(column_sizes) < width:
        column_sizes.append(int(rng.integers(6, 11)))
    row_index = np.repeat(np.arange(len(row_sizes)), row_sizes)[:height]
    col_index = np.repeat(np.arange(len(column_sizes)), column_sizes)[:width]
    table = rng.choice(palette, size=(len(row_sizes), len(column_sizes)))
    for row in range(len(row_sizes)):
        table[row, row % len(column_sizes)] = palette[row % len(palette)]
    return table[row_index[:, None], col_index[None, :]].copy()


def _dimensions(rng):
    if rng.random() < 0.46:
        side = int(rng.integers(18, 31))
        return side, side
    return int(rng.integers(18, 31)), int(rng.integers(18, 31))


def _start(rng, maximum, sector):
    if sector == 0:
        return int(rng.integers(0, max(1, maximum // 3 + 1)))
    if sector == 2:
        return int(rng.integers(maximum * 2 // 3, maximum + 1))
    return int(rng.integers(maximum // 3, maximum * 2 // 3 + 1))


def _place(rng, background, side, uniform):
    height, width = background.shape
    if uniform:
        starts = [(r, c) for r in range(height - side + 1)
                  for c in range(width - side + 1)
                  if np.unique(background[r + 1:r + side - 1,
                                          c + 1:c + side - 1]).size == 1]
        if not starts:
            return None
        return starts[int(rng.integers(len(starts)))]
    r = _start(rng, height - side, int(rng.integers(3)))
    c = _start(rng, width - side, int(rng.integers(3)))
    return r, c


def _make_pair(rng, side, marker, previous_marker, reuse, mode):
    for _ in range(90):
        shape = _dimensions(rng)
        palette = _palette(rng, marker, previous_marker, reuse)
        background = (_block_field(rng, shape, palette) if mode == "blocks"
                      else _rhythm_field(rng, shape, palette, mode, reuse))
        if np.unique(background).size < 2:
            continue
        placement = _place(rng, background, side, mode == "blocks")
        if placement is None:
            continue
        row, column = placement
        frame = (row, column, side, marker)
        grid = background.copy()
        grid[_perimeter(shape, frame)] = marker
        output = grid[row:row + side, column:column + side].copy()
        frames = _frames(grid)
        if frame not in frames or _plausible_frames(grid, frames) != [frame]:
            continue
        if mode == "outline" and len(frames) < 2:
            continue
        if reuse != _color_outside_perimeter(grid, frame):
            continue
        if (previous_marker is not None and previous_marker != marker
                and not np.any(grid[~_perimeter(shape, frame)] == previous_marker)):
            continue
        if np.count_nonzero(grid != background) < max(4, side):
            continue
        return {"input": grid.astype(np.int8), "output": output.astype(np.int8)}
    raise RuntimeError("could not construct an unambiguous border")


def _analyze_pair(pair):
    grid, output = pair["input"], pair["output"]
    assert output.shape[0] == output.shape[1] and 4 <= output.shape[0] <= 10
    side = output.shape[0]
    frames = _frames(grid)
    candidates = []
    for frame in frames:
        row, column, candidate_side, _ = frame
        if candidate_side == side and np.array_equal(
            output, grid[row:row + side, column:column + side]
        ):
            candidates.append(frame)
    assert len(candidates) == 1, "the output must identify one unchanged crop"
    frame = candidates[0]
    assert _plausible_frames(grid, frames) == [frame], "ambiguous added border"
    return frame, _color_outside_perimeter(grid, frame)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _seed_value(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    markers = list(map(int, rng.permutation(10)))
    while len(markers) < train_count + test_count:
        markers.extend(map(int, rng.permutation(10)))
    first_side = int(rng.integers(5, 8))
    second_side = int(rng.integers(first_side + 1, 10))
    train_sides = [first_side, second_side]
    train_sides.extend(int(rng.integers(5, 10)) for _ in range(train_count - 2))
    relation = seed_value % 3
    if relation == 0:
        first_test_side = int(rng.integers(4, min(train_sides)))
    elif relation == 1:
        first_test_side = int(rng.choice(train_sides))
    else:
        first_test_side = int(rng.integers(max(train_sides) + 1, 11)) if max(train_sides) < 10 else 10
    test_sides = [first_test_side]
    test_sides.extend(int(rng.integers(4, 11)) for _ in range(test_count - 1))
    have_reuse_demo = seed_value % 2 == 0
    have_outline_demo = True
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for index in range(train_count + test_count):
        training = index < train_count
        side = train_sides[index] if training else test_sides[index - train_count]
        reuse = ((index == 1 and have_reuse_demo)
                 or (index > 1 and have_reuse_demo and rng.random() < 0.30))
        if not training and not have_reuse_demo:
            reuse = False
        if reuse:
            mode = "rhythm"
        elif index == 0 and have_outline_demo:
            mode = "outline"
        elif index < 2:
            mode = "rhythm"
        else:
            modes = ["rhythm", "blocks"]
            if training or have_outline_demo:
                modes.append("outline")
            mode = str(rng.choice(modes))
        if mode == "blocks" and side > 6:
            mode = "rhythm"
        marker = markers[index]
        for _ in range(20):
            pair = _make_pair(rng, side, marker, previous, reuse, mode)
            if _key(pair["input"]) not in seen and _source_digest(pair) not in _SOURCE_PAIR_DIGESTS:
                break
        else:
            raise RuntimeError("could not construct distinct pairs")
        split = "train" if training else "test"
        puzzle[split].append(pair)
        seen.add(_key(pair["input"]))
        previous = marker
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_reuse = False
    test_reuse = False
    train_ordinary_outline = False
    test_ordinary_outline = False
    train_sides = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= length <= 30 for length in grid.shape)
                assert grid.size and 0 <= int(grid.min()) <= int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            inp_key, out_key = _key(inp), _key(out)
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            assert not np.array_equal(inp, out)
            assert _source_digest(pair) not in _SOURCE_PAIR_DIGESTS
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            frame, reused = _analyze_pair(pair)
            assert inp.shape[0] >= frame[2] + 4 and inp.shape[1] >= frame[2] + 4
            if split == "train":
                train_reuse |= reused
                train_ordinary_outline |= len(_frames(inp)) > 1
                train_sides.add(frame[2])
            else:
                test_reuse |= reused
                test_ordinary_outline |= len(_frames(inp)) > 1
    assert len(train_sides) >= 2
    assert not test_reuse or train_reuse, "training must demonstrate reused marker detection"
    assert not test_ordinary_outline or train_ordinary_outline, "training must show an ordinary square outline"


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[k], b[k])
        for a, b in zip(first[split], second[split]) for k in ("input", "output")
    ) for split in ("train", "test"))


def self_test() -> None:
    assert _equal(generate_puzzle(11), generate_puzzle(11))
    assert not _equal(generate_puzzle(11), generate_puzzle(12))
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        puzzle = generate_puzzle(300 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for kwargs in ({"num_test": 1}, {"num_test": 4},
                   {"num_train": 2}, {"num_train": 10}):
        puzzle = generate_puzzle(444, **kwargs)
        for key, value in kwargs.items():
            assert len(puzzle["train" if key == "num_train" else "test"]) == value
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    train_counts, test_counts = set(), set()
    shapes, sides, markers, palettes, layouts = set(), set(), set(), set(), set()
    novel = reused = natural = uniform = square = 0
    zero_absent = source_dimensions = edge = 0
    natural_nonshort = reused_rectangular = uniform_rectangular = 0
    train_regions, test_regions = set(), set()
    test_size_relations = set()
    varied_field_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_side_values = [pair["output"].shape[0] for pair in puzzle["train"]]
        first_test_side = puzzle["test"][0]["output"].shape[0]
        if first_test_side < min(train_side_values):
            test_size_relations.add("smaller")
        elif first_test_side > max(train_side_values):
            test_size_relations.add("larger")
        elif first_test_side in train_side_values:
            test_size_relations.add("same")
        else:
            test_size_relations.add("between")
        field_palettes = set()
        pair_markers = set()
        for index, pair in enumerate(puzzle["train"] + puzzle["test"]):
            grid, output = pair["input"], pair["output"]
            frame, is_reused = _analyze_pair(pair)
            row, column, side, marker = frame
            mask = _perimeter(grid.shape, frame)
            assert np.array_equal(output, grid[row:row + side, column:column + side])
            assert all(np.all(edge == marker) for edge in
                       (output[0], output[-1], output[:, 0], output[:, -1]))
            field_palettes.add(tuple(map(int, np.unique(grid[~mask]))))
            pair_markers.add(marker)
            if index:
                previous_marker = (puzzle["train"] + puzzle["test"])[index - 1]["output"][0, 0]
                if previous_marker != marker:
                    assert previous_marker in grid[~mask]
            shapes.add(grid.shape)
            sides.add(side)
            markers.add(marker)
            palettes.add(tuple(map(int, np.unique(grid))))
            layouts.add((grid.shape, row, column, side))
            novel += not is_reused
            reused += is_reused
            natural += len(_frames(grid)) > 1
            uniform += np.unique(output[1:-1, 1:-1]).size == 1
            square += grid.shape[0] == grid.shape[1]
            zero_absent += 0 not in grid
            source_dimensions += grid.shape in {(24, 24), (26, 26), (28, 28)}
            edge += (row == 0 or column == 0 or row + side == grid.shape[0]
                     or column + side == grid.shape[1])
            if index == 0:
                natural_nonshort += (len(_frames(grid)) > 1
                                     and not _fits_repeated_field(grid, mask))
            reused_rectangular += is_reused and grid.shape[0] != grid.shape[1]
            uniform_rectangular += (np.unique(output[1:-1, 1:-1]).size == 1
                                    and grid.shape[0] != grid.shape[1])
            region = (min(2, 3 * row // (grid.shape[0] - side + 1)),
                      min(2, 3 * column // (grid.shape[1] - side + 1)))
            (train_regions if index < len(puzzle["train"]) else test_regions).add(region)
        assert len(pair_markers) >= 2
        varied_field_palettes += len(field_palettes) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(layouts) >= 80
    assert len(sides) >= 6 and len(markers) >= 8 and len(palettes) >= 20
    assert novel >= 30 and reused >= 8 and natural >= 6 and uniform >= 4
    assert square >= 25
    assert zero_absent >= 20 and source_dimensions >= 3 and edge >= 8
    assert natural_nonshort >= 30 and reused_rectangular >= 3
    assert uniform_rectangular >= 3 and varied_field_palettes >= 30
    assert len(train_regions) == 9 and len(test_regions) == 9
    assert {"smaller", "larger", "same"} <= test_size_relations
