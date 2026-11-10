import numpy as np


_SOURCE_INPUTS = frozenset({
    "00000000000/00000000000/00000000000/40000000000/00000004000/00000004000/04440000040/00000000040/00000000040/00000000040",
    "0000000000/0000000000/0000060060/0000060060/0060060000/0060060000/0060060000/0000000060/6600000000",
    "000000000000/000000000000/007000000000/000000007000/007770007000/000000007000/000000007000/777777007000/000000000000/000000000000",
    "0000000000/0110000000/0000011110/0000000000/0000000000/0010000000/0010000000/0000001000/0000000000/0000000000",
    "00000050000/00000050000/05555050000/00000050000/05555050000/00000000000/00000000000/00005500000/00000000555",
})


def _grid_text(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _inspect_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    bg_index = int(np.argmax(counts))
    assert counts[bg_index] > counts[1 - bg_index]
    background = int(colors[bg_index])
    foreground = int(colors[1 - bg_index])
    mask = grid == foreground
    seen = np.zeros(grid.shape, dtype=bool)
    height, width = grid.shape
    bars = []
    for raw_row, raw_col in np.argwhere(mask):
        row, col = int(raw_row), int(raw_col)
        if seen[row, col]:
            continue
        stack = [(row, col)]
        seen[row, col] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr or dc) and 0 <= nr < height and 0 <= nc < width:
                        if mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
        rows, cols = {r for r, _ in cells}, {c for _, c in cells}
        length = len(cells)
        if len(rows) == 1:
            assert len(cols) == length and max(cols) - min(cols) + 1 == length
            orientation = "s" if length == 1 else "h"
        else:
            assert len(cols) == 1 and len(rows) == length
            assert max(rows) - min(rows) + 1 == length
            orientation = "v"
        bars.append((length, orientation, min(cols), min(rows)))
    return background, foreground, bars


def _draw_output(shape, lengths, background, foreground):
    height, width = shape
    assert len(lengths) <= width and all(1 <= x <= height for x in lengths)
    output = np.full(shape, background, dtype=np.int8)
    for col, length in enumerate(sorted(lengths)):
        output[height - length:, col] = foreground
    return output


def _dimensions(rng, anchor, ordered):
    if anchor or ordered:
        return int(rng.integers(9, 17)), int(rng.integers(11, 22))
    height, width = int(rng.integers(7, 27)), int(rng.integers(7, 29))
    if rng.random() < 0.25:
        height, width = int(rng.integers(8, 14)), int(rng.integers(9, 16))
    return height, width


def _sample_bars(rng, count, height, width, mode, anchor):
    common_cap = min(height, width)
    tall_possible = mode != "h" and height >= 9
    profile = int(rng.choice(3, p=[0.42, 0.38, 0.20]))
    if not tall_possible and profile == 2:
        profile = 1
    cap = min(common_cap, int(rng.integers(4, 8))) if profile == 0 else common_cap
    cap = max(cap, count if anchor == 1 else 4 if anchor == 2 else 1)
    if anchor == 1:
        lengths = [int(x) for x in rng.choice(np.arange(1, cap + 1), size=count, replace=False)]
    elif anchor == 2:
        lengths = [int(x) for x in rng.integers(1, cap + 1, size=count)]
        lengths[1] = lengths[0]
    else:
        pattern = int(rng.integers(0, 3))
        if pattern == 0 and cap >= count:
            lengths = [int(x) for x in rng.choice(np.arange(1, cap + 1), size=count, replace=False)]
        elif pattern == 1:
            lengths = [int(x) for x in rng.integers(1, cap + 1, size=count)]
            lengths[1] = lengths[0]
        else:
            base = int(rng.integers(1, cap + 1))
            repeat = int(rng.integers(2, count + 1))
            lengths = [base] * repeat
            lengths += [int(x) for x in rng.integers(1, cap + 1, size=count - repeat)]
    if profile == 2:
        lengths[int(rng.integers(count))] = height - int(rng.integers(0, 2))
    if anchor == 1 and len(set(lengths)) != count:
        return None
    if anchor == 2:
        lengths[1] = lengths[0]
    orientations = []
    for length in lengths:
        if length > width:
            orientations.append("v")
        elif mode == "m":
            orientations.append("h" if rng.random() < 0.5 else "v")
        else:
            orientations.append(mode)
    if mode == "m":
        eligible_h = [i for i, length in enumerate(lengths) if 1 < length <= width]
        eligible_v = [i for i, length in enumerate(lengths) if length > 1]
        if not eligible_h or len(eligible_v) < 2:
            return None
        h_index = int(rng.choice(eligible_h))
        v_index = int(rng.choice([i for i in eligible_v if i != h_index]))
        orientations[h_index], orientations[v_index] = "h", "v"
    if not any(length > 1 for length in lengths):
        return None
    return lengths, orientations


def _place_segments(rng, height, width, lengths, orientations, background, foreground, style):
    occupied = np.zeros((height, width), dtype=bool)
    for index in sorted(range(len(lengths)), key=lambda i: -lengths[i]):
        length = lengths[index]
        bh = length if orientations[index] == "v" else 1
        bw = length if orientations[index] == "h" else 1
        candidates = []
        for row in range(height - bh + 1):
            for col in range(width - bw + 1):
                near = occupied[max(0, row - 1):min(height, row + bh + 1),
                                max(0, col - 1):min(width, col + bw + 1)]
                if not near.any():
                    candidates.append((row, col))
        if not candidates:
            return None
        if style == "free" or not occupied.any():
            row, col = candidates[int(rng.integers(len(candidates)))]
        else:
            rr, cc = np.nonzero(occupied)
            center_r, center_c = float(rr.mean()), float(cc.mean())
            scored = []
            for row, col in candidates:
                distance = abs(row + (bh - 1) / 2 - center_r) + abs(col + (bw - 1) / 2 - center_c)
                edge = min(row, col, height - row - bh, width - col - bw)
                value = distance if style == "cluster" else -distance if style == "spread" else edge
                scored.append((value + float(rng.random()) * 0.6, row, col))
            scored.sort()
            choice = scored[int(rng.integers(min(6, len(scored))))]
            row, col = choice[1], choice[2]
        occupied[row:row + bh, col:col + bw] = True
    if int(occupied.sum()) * 2 >= height * width:
        return None
    return np.where(occupied, foreground, background).astype(np.int8)


def _generate_pair(rng, anchor, target, first_test, background, foreground):
    for _ in range(280):
        ordered = not anchor and rng.random() < 0.24
        height, width = _dimensions(rng, bool(anchor), ordered)
        if anchor == 1:
            count = int(rng.integers(4, 6))
        elif anchor == 2:
            count = int(rng.integers(4, 7))
        elif ordered:
            count = int(rng.integers(2, 5))
        else:
            count = int(rng.integers(2, min(9, width + 1)))
        if anchor:
            mode = "m"
        elif first_test:
            mode = str(rng.choice(["h", "v"]))
        else:
            mode = str(rng.choice(["m", "h", "v"],
                                  p=[0.38, 0.31, 0.31] if target else [0.54, 0.23, 0.23]))
        sampled = _sample_bars(rng, count, height, width, mode, anchor)
        if sampled is None:
            continue
        lengths, orientations = sampled
        style = str(rng.choice(["free", "cluster", "spread", "edge"],
                               p=[0.35, 0.24, 0.22, 0.19]))
        input_grid = _place_segments(rng, height, width, lengths, orientations,
                                     background, foreground, style)
        if input_grid is None or _grid_text(input_grid) in _SOURCE_INPUTS:
            continue
        bg, fg, bars = _inspect_input(input_grid)
        if bg != background or fg != foreground or len(bars) != count:
            continue
        left_order = [b[0] for b in sorted(bars, key=lambda b: (b[2], b[3]))]
        if anchor == 1 and left_order == sorted(left_order):
            continue
        if ordered and left_order != sorted(left_order):
            continue
        output_grid = _draw_output(input_grid.shape, lengths, background, foreground)
        if np.array_equal(input_grid, output_grid):
            continue
        return input_grid, output_grid
    raise RuntimeError(f"could not construct separated bars (anchor={anchor}, target={target})")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    base_background = 0 if rng.random() < 0.62 else int(rng.integers(1, 10))
    palette_mode = int(rng.integers(0, 3))
    first_foreground = int(rng.choice([x for x in range(10) if x != base_background]))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs, train_foregrounds = set(), []
    for position in range(train_count + test_count):
        target = position >= train_count
        index = position - train_count if target else position
        if position == 0 or palette_mode == 0:
            background = base_background
        elif palette_mode == 2 and position == 1:
            background = first_foreground
        elif palette_mode == 1 and position == 1:
            background = int(rng.choice([x for x in range(10) if x != base_background]))
        elif rng.random() < 0.48:
            background = base_background
        else:
            background = int(rng.integers(0, 10))
        if position == 0:
            foreground = first_foreground
        elif palette_mode == 2 and position == 1:
            foreground = base_background
        elif position == 1:
            foreground = int(rng.choice([x for x in range(10)
                                         if x != background and x != first_foreground]))
        elif target and rng.random() < 0.53:
            choices = [x for x in train_foregrounds if x != background]
            foreground = int(rng.choice(choices)) if choices else int(rng.choice([x for x in range(10) if x != background]))
        else:
            foreground = int(rng.choice([x for x in range(10) if x != background]))
        if not target:
            train_foregrounds.append(foreground)
        pair_rng = np.random.default_rng(int(pair_seeds[position]))
        anchor = index + 1 if not target and index < 2 else 0
        for _ in range(12):
            input_grid, output_grid = _generate_pair(pair_rng, anchor, target,
                                                      target and index == 0,
                                                      background, foreground)
            key = _grid_key(input_grid)
            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["test" if target else "train"].append({"input": input_grid, "output": output_grid})
                break
        else:
            raise RuntimeError("could not avoid duplicate input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    first_train_foreground = None
    train_backgrounds, test_backgrounds = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            assert _grid_text(input_grid) not in _SOURCE_INPUTS
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            bg, fg, bars = _inspect_input(input_grid)
            if split == "train":
                train_backgrounds.add(bg)
            else:
                test_backgrounds.add(bg)
            if split == "train" and index == 0:
                first_train_foreground = fg
            elif split == "train" and index == 1:
                assert fg != first_train_foreground
            lengths = [bar[0] for bar in bars]
            orientations = {bar[1] for bar in bars}
            assert 2 <= len(bars) <= 8 and len(bars) <= input_grid.shape[1]
            assert all(length <= input_grid.shape[0] for length in lengths)
            for col in range(output_grid.shape[1]):
                expected_length = sorted(lengths)[col] if col < len(lengths) else 0
                expected_column = np.full(output_grid.shape[0], bg, dtype=np.int8)
                if expected_length:
                    expected_column[-expected_length:] = fg
                assert np.array_equal(output_grid[:, col], expected_column)
            assert not np.array_equal(input_grid, output_grid)
            if split == "train" and index < 2:
                assert 4 <= len(bars) <= 6 and "h" in orientations and "v" in orientations
                if index == 0:
                    assert len(set(lengths)) == len(lengths)
                    left_order = [b[0] for b in sorted(bars, key=lambda b: (b[2], b[3]))]
                    assert left_order != sorted(left_order)
                else:
                    assert len(set(lengths)) < len(lengths)
            if split == "test" and index == 0:
                assert not ("h" in orientations and "v" in orientations)
    if not test_backgrounds.issubset(train_backgrounds):
        assert len(train_backgrounds) > 1


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, counts, profiles, orientations, backgrounds = set(), set(), set(), set(), set()
    near_top = no_singleton = all_horizontal = all_vertical = ascending = reuse = bg_change = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        train_colors = {_inspect_input(p["input"])[1] for p in puzzle["train"]}
        all_bgs = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg, fg, bars = _inspect_input(grid)
                all_bgs.append(bg)
                lengths = [b[0] for b in bars]
                left_order = [b[0] for b in sorted(bars, key=lambda b: (b[2], b[3]))]
                kinds = frozenset(b[1] for b in bars if b[0] > 1)
                shapes.add(grid.shape)
                counts.add(len(bars))
                profiles.add(tuple(sorted(lengths)))
                orientations.add(kinds)
                backgrounds.add(bg)
                near_top += max(lengths) >= grid.shape[0] - 1
                no_singleton += 1 not in lengths
                all_horizontal += kinds == {"h"}
                all_vertical += kinds == {"v"}
                ascending += left_order == sorted(left_order)
                if split == "test":
                    reuse += fg in train_colors
                assert np.array_equal(pair["output"], _draw_output(grid.shape, lengths, bg, fg))
        bg_change += len(set(all_bgs)) > 1
    assert len(shapes) >= 65 and len(counts) >= 6 and len(profiles) >= 80
    assert len(orientations) == 3 and len(backgrounds) >= 7
    assert min(near_top, no_singleton, all_horizontal, all_vertical, ascending, reuse, bg_change) > 0
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, train, test)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(1000 + index, train, test))
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (3, 4), (True, 1), (2, False)):
        try:
            generate_puzzle(2026, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
