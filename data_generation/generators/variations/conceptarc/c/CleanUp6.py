"""Repeated stepped silhouettes with sparse, recoverable damage."""

import numpy as np


_SOURCE_INPUTS = {
    ("0000000003", "3033333000", "0003330000", "0100300001", "0000000000",
     "0000001100", "0100300003", "0003330000", "0033333001", "3000000000"),
    ("00000000000800000", "08000000000000000", "00600000000000600",
     "00680008000006800", "00666000000066600", "00866600000666600",
     "00686000080066600", "00660080008006600", "00600000000000600",
     "00000000000000000"),
    ("0707000700", "0100000010", "0111001110", "0111001110", "7170070010", "0000000000"),
    ("000002000", "444444444", "044244240", "204424400", "000444000",
     "020040002", "000040000", "000000020", "000000000", "444424444",
     "044444440", "004242400", "000444000", "020040000", "020040000",
     "000000202", "022002000"),
    ("050060700603006", "055000770003300", "655500777003330",
     "056550677603633", "055500777006330", "065000770063300",
     "050060706003060"),
}


def _choose_counts(rng, num_train, num_test):
    for value, name, lower, upper in ((num_train, "num_train", 2, 10),
                                      (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not lower <= value <= upper):
            raise ValueError(f"{name} is outside its legal integer range")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _mask_from_widths(widths, centered):
    width = max(widths)
    mask = np.zeros((len(widths), width), dtype=bool)
    for row, span in enumerate(widths):
        start = (width - span) // 2 if centered else 0
        mask[row, start:start + span] = True
    return mask


def _vertical_mask(rng):
    levels = int(rng.integers(3, 7))
    tail = int(rng.integers(0, 3))
    widths = list(range(2 * levels - 1, 0, -2)) + [1] * tail
    return _mask_from_widths(widths, True)


def _side_mask(rng):
    peak = int(rng.integers(3, 7))
    rise = [1]
    while rise[-1] < peak:
        rise.append(min(peak, rise[-1] + int(rng.integers(1, 3))))
    plateau = int(rng.integers(1, 4))
    if len(rise) == 2 and plateau == 1:
        plateau = 2
    widths = rise + [peak] * (plateau - 1) + rise[-2::-1]
    return _mask_from_widths(widths, False)


def _margin(rng):
    return int(rng.choice((0, 0, 1, 1, 2, 3)))


def _offset(rng):
    return int(rng.choice((0, 0, 0, 0, -2, -1, 1, 2)))


def _geometry(rng, mode):
    stack = mode.startswith("stack")
    compact = bool(rng.random() < 0.17)
    if compact and not stack:
        if mode == "triple_same":
            base = _mask_from_widths([1, 2, 3, 4, 3, 2, 1], False)
            positions = [(0, 0), (0, 5), (0, 10)]
            return base, positions, (7, 15)
        base = _mask_from_widths([1, 3, 3, 1], False)
        gap = int(rng.integers(1, 4))
        left = int(rng.integers(0, 5 - gap))
        top = int(rng.integers(0, 3))
        return base, [(top, left), (top, left + 3 + gap)], (6, 10)

    base = _vertical_mask(rng) if stack else _side_mask(rng)
    height, width = base.shape
    if stack:
        gap = int(rng.integers(1, 7))
        positions = [(0, 0), (height + gap, _offset(rng))]
    elif mode == "triple_same":
        first_gap = int(rng.integers(1, 5))
        second_gap = int(rng.integers(1, 5))
        positions = [(0, 0), (int(rng.integers(-1, 2)), width + first_gap),
                     (int(rng.integers(-1, 2)), 2 * width + first_gap + second_gap)]
    else:
        gap = int(rng.integers(1, 7))
        positions = [(0, 0), (_offset(rng), width + gap)]

    top = min(row for row, _ in positions)
    left = min(col for _, col in positions)
    left_margin = _margin(rng) if mode != "triple_same" else int(rng.integers(0, 3))
    shift_row, shift_col = _margin(rng) - top, left_margin - left
    positions = [(row + shift_row, col + shift_col) for row, col in positions]
    grid_height = max(row + height for row, _ in positions) + _margin(rng)
    right_margin = _margin(rng) if mode != "triple_same" else int(rng.integers(0, 3))
    grid_width = max(col + width for _, col in positions) + right_margin
    assert grid_height <= 30 and grid_width <= 30
    return base, positions, (grid_height, grid_width)


def _oriented_masks(base, mode):
    if mode == "stack_reflect":
        return [base, np.flipud(base)]
    if mode == "side_reflect":
        return [base, np.fliplr(base)]
    return [base] * (3 if mode == "triple_same" else 2)


def _make_pair(rng, mode, force_multicolor=False, excluded_background=None):
    base, positions, grid_shape = _geometry(rng, mode)
    masks = _oriented_masks(base, mode)
    palette = [int(x) for x in rng.permutation(10)]
    background = palette[0]
    if excluded_background is not None and background == excluded_background:
        background = palette[1]
    colors = [color for color in palette if color != background]
    if len(masks) == 3:
        if force_multicolor or rng.random() < 0.5:
            fills = colors[:3]
        else:
            fills = [colors[0], colors[0], colors[1]]
            fills = [fills[int(index)] for index in rng.permutation(3)]
    elif force_multicolor or rng.random() < 0.42:
        fills = colors[:2]
    else:
        fills = [colors[0], colors[0]]
    noise = next(color for color in colors if color not in fills)

    height, width = grid_shape
    output = np.full((height, width), background, dtype=np.int8)
    shape_height, shape_width = base.shape
    for mask, (row, col), fill in zip(masks, positions, fills):
        window = output[row:row + shape_height, col:col + shape_width]
        window[mask] = fill
    # The background must be recognizable even with repeated fill colors.
    while np.count_nonzero(output == background) <= max(np.count_nonzero(output == fill) for fill in set(fills)) + 6:
        if output.shape[0] < 30:
            output = np.pad(output, ((0, 1), (0, 0)), constant_values=background)
        else:
            output = np.pad(output, ((0, 0), (0, 1)), constant_values=background)

    input_grid = output.copy()
    regime = ("exterior_only", "foreign_only", "missing_only", "mixed")[int(rng.integers(0, 4))]
    coordinates = [(int(r), int(c)) for r, c in np.argwhere(base)]
    rng.shuffle(coordinates)
    if regime == "exterior_only":
        interior_count = 0
    else:
        upper = max(2, min(len(coordinates), int(np.ceil(len(coordinates) * 0.31))))
        interior_count = int(rng.integers(2 if regime == "mixed" else 1, upper + 1))
    landmarks = [point for point in coordinates if point[0] in (0, shape_height - 1)
                 or int(base[point[0]].sum()) == shape_width]
    selected = []
    if interior_count and landmarks and rng.random() < 0.48:
        selected.append(landmarks[int(rng.integers(0, len(landmarks)))])
    selected.extend(point for point in coordinates if point not in selected)
    selected = selected[:interior_count]
    first_copy = int(rng.integers(0, len(masks)))
    for index, (row, col) in enumerate(selected):
        copy_index = (first_copy + index) % len(masks)
        if mode == "stack_reflect" and copy_index == 1:
            row = shape_height - 1 - row
        if mode == "side_reflect" and copy_index == 1:
            col = shape_width - 1 - col
        top, left = positions[copy_index]
        replacement = (background if regime == "missing_only" else noise
                       if regime == "foreign_only" else (background if index % 2 == 0 else noise))
        input_grid[top + row, left + col] = replacement

    outside_count = int(rng.integers(1, 8)) if regime == "exterior_only" else int(rng.integers(0, 7))
    if outside_count:
        exterior = [(int(r), int(c)) for r, c in np.argwhere(output == background)]
        rng.shuffle(exterior)
        edges = [p for p in exterior if p[0] in (0, output.shape[0] - 1)
                 or p[1] in (0, output.shape[1] - 1)]
        corners = [p for p in exterior if p[0] in (0, output.shape[0] - 1)
                   and p[1] in (0, output.shape[1] - 1)]
        adjacent = [p for p in exterior if np.any(output[max(0, p[0] - 1):p[0] + 2,
                                                       max(0, p[1] - 1):p[1] + 2] != background)]
        chosen = []
        for pool, chance in ((corners, 0.28), (edges, 0.32), (adjacent, 0.34)):
            if pool and len(chosen) < outside_count and rng.random() < chance:
                option = pool[int(rng.integers(0, len(pool)))]
                if option not in chosen:
                    chosen.append(option)
        chosen.extend(p for p in exterior if p not in chosen)
        chosen = chosen[:outside_count]
        fill_count = int(rng.integers(0, min(3, outside_count) + 1))
        rng.shuffle(chosen)
        for index, (row, col) in enumerate(chosen):
            input_grid[row, col] = fills[int(rng.integers(0, len(fills)))] if index < fill_count else noise
    while np.count_nonzero(input_grid == background) <= max(
        np.count_nonzero(input_grid == color) for color in range(10) if color != background
    ):
        if input_grid.shape[0] < 30:
            input_grid = np.pad(input_grid, ((0, 1), (0, 0)), constant_values=background)
            output = np.pad(output, ((0, 1), (0, 0)), constant_values=background)
        elif input_grid.shape[1] < 30:
            input_grid = np.pad(input_grid, ((0, 0), (0, 1)), constant_values=background)
            output = np.pad(output, ((0, 0), (0, 1)), constant_values=background)
        else:
            raise AssertionError("background is not identifiable")
    return input_grid, output


def _components(output, background):
    remaining = set((int(r), int(c)) for r, c in np.argwhere(output != background))
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        bottom = max(r for r, _ in cells)
        right = max(c for _, c in cells)
        color = int(output[start])
        assert all(int(output[r, c]) == color for r, c in cells)
        mask = np.zeros((bottom - top + 1, right - left + 1), dtype=bool)
        for row, col in cells:
            mask[row - top, col - left] = True
        components.append((top, left, color, mask))
    return components


def _vertical_profile_ok(mask):
    height, width = mask.shape
    if not 3 <= height <= 10 or width < 5 or width % 2 != 1:
        return False
    widths = []
    for row in mask:
        positions = np.flatnonzero(row)
        span = len(positions)
        if not span or span % 2 != 1 or positions[0] != (width - span) // 2 or not np.all(row[positions[0]:positions[-1] + 1]):
            return False
        widths.append(span)
    return widths[0] == width and widths[-1] == 1 and all(a - b in (0, 2) for a, b in zip(widths, widths[1:]))


def _side_profile_ok(mask):
    height, width = mask.shape
    if not 4 <= height <= 14 or not 3 <= width <= 6:
        return False
    widths = []
    for row in mask:
        positions = np.flatnonzero(row)
        if not len(positions) or not np.array_equal(positions, np.arange(len(positions))):
            return False
        widths.append(len(positions))
    if widths != widths[::-1] or widths[0] != 1 or max(widths) != width:
        return False
    peak = widths.index(width)
    return all(b - a in (1, 2) for a, b in zip(widths[:peak], widths[1:peak + 1])) and all(x == width for x in widths[peak:height - peak])


def _inspect_pair(input_grid, output):
    assert input_grid.shape == output.shape and not np.array_equal(input_grid, output)
    colors, counts = np.unique(output, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert np.count_nonzero(output == background) > output.size // 3
    assert np.count_nonzero(input_grid == background) > max(
        np.count_nonzero(input_grid == color) for color in range(10) if color != background
    )
    components = _components(output, background)
    assert len(components) in (2, 3)
    components.sort(key=lambda x: (x[0], x[1]))
    first = components[0]
    shape = first[3].shape
    assert all(component[3].shape == shape for component in components)
    if len(components) == 3:
        components.sort(key=lambda x: x[1])
        assert all(components[i + 1][1] >= components[i][1] + shape[1] + 1 for i in range(2))
        assert max(c[0] for c in components) - min(c[0] for c in components) <= 2
        assert all(np.array_equal(c[3], components[0][3]) for c in components)
        mode = "triple_same"
    elif components[1][0] >= components[0][0] + shape[0] + 1:
        assert abs(components[1][1] - components[0][1]) <= 2
        mode = "stack_same" if np.array_equal(components[1][3], first[3]) else "stack_reflect"
        assert np.array_equal(components[1][3], first[3] if mode == "stack_same" else np.flipud(first[3]))
    else:
        components.sort(key=lambda x: x[1])
        assert components[1][1] >= components[0][1] + shape[1] + 1
        assert abs(components[1][0] - components[0][0]) <= 2
        mode = "side_same" if np.array_equal(components[1][3], components[0][3]) else "side_reflect"
        assert np.array_equal(components[1][3], components[0][3] if mode == "side_same" else np.fliplr(components[0][3]))
    base = components[0][3]
    assert (_vertical_profile_ok(base) if mode.startswith("stack") else _side_profile_ok(base)), (mode, base.shape, [int(row.sum()) for row in base])
    fills = [c[2] for c in components]
    extras = set(int(x) for x in np.unique(input_grid)) - set(fills) - {background}
    assert len(extras) <= 1
    inside = output != background
    changed = input_grid != output
    assert np.any(changed)
    assert np.count_nonzero(changed & ~inside) <= 7
    assert np.count_nonzero(changed & inside) <= max(2, int(np.ceil(base.sum() * 0.31)))
    for top, left, fill, mask in components:
        window = input_grid[top:top + shape[0], left:left + shape[1]]
        visible = window[mask] == fill
        assert np.count_nonzero(visible) >= max(1, int(np.ceil(mask.sum() * 0.65)))
        assert set(int(x) for x in np.unique(window[mask])) <= {fill, background} | extras
    aligned = []
    for index, (top, left, fill, mask) in enumerate(components):
        window = input_grid[top:top + shape[0], left:left + shape[1]]
        visible = window == fill
        if mode == "stack_reflect" and index == 1:
            visible = np.flipud(visible)
        if mode == "side_reflect" and index == 1:
            visible = np.fliplr(visible)
        aligned.append(visible)
    assert np.all(np.logical_or.reduce(aligned)[base])
    return {"mode": mode, "background": background, "components": components,
            "inside_missing": int(np.count_nonzero(inside & (input_grid == background))),
            "inside_foreign": int(np.count_nonzero(inside & changed & (input_grid != background))),
            "outside_fill": int(np.count_nonzero(~inside & np.isin(input_grid, fills))),
            "outside_foreign": int(np.count_nonzero(~inside & changed & ~np.isin(input_grid, fills))),
            "landmark_damage": any(input_grid[top + r, left + c] != fill
                                   for top, left, fill, mask in components
                                   for r, c in np.argwhere(mask)
                                   if r in (0, shape[0] - 1) or int(mask[r].sum()) == shape[1]),
            "border_noise": bool(np.any(changed[0]) or np.any(changed[-1]) or np.any(changed[:, 0]) or np.any(changed[:, -1])),
            "corner_noise": bool(any(output[r, c] == background and input_grid[r, c] != background
                                     for r in (0, output.shape[0] - 1)
                                     for c in (0, output.shape[1] - 1)))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    train_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_grid, output = pair["input"], pair["output"]
            key = (input_grid.shape, input_grid.tobytes())
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert tuple("".join(str(int(x)) for x in row) for row in input_grid) not in _SOURCE_INPUTS
            features = _inspect_pair(input_grid, output)
            if split == "train":
                train_features.append(features)
    assert any(f["mode"].startswith("stack") for f in train_features)
    assert any(f["mode"].startswith("side") or f["mode"] == "triple_same" for f in train_features)
    assert any(len({c[2] for c in f["components"]}) >= 2 for f in train_features)
    assert len({f["background"] for f in train_features}) >= 2


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, num_train, num_test)
    emphasis = int(rng.integers(0, 3))
    if emphasis == 0:
        leading = ["stack_reflect", "side_reflect"]
        test_pool = ("stack_same", "side_same", "triple_same", "stack_same", "triple_same")
    elif emphasis == 1:
        leading = ["stack_same", "triple_same" if rng.random() < 0.5 else "side_same"]
        test_pool = ("stack_reflect", "side_reflect", "side_reflect", "stack_same", "triple_same")
    else:
        leading = ["stack_reflect" if rng.random() < 0.5 else "stack_same",
                   "side_same" if rng.random() < 0.5 else "side_reflect"]
        test_pool = ("stack_reflect", "side_reflect", "stack_same", "side_same", "triple_same")
    all_modes = ("stack_reflect", "stack_same", "side_reflect", "side_same", "triple_same")
    train_pool = (("stack_reflect", "side_reflect") if emphasis == 0 else
                  ("stack_same", "side_same", "triple_same") if emphasis == 1 else all_modes)
    train_modes = leading + [train_pool[int(rng.integers(0, len(train_pool)))] for _ in range(train_count - 2)]
    test_modes = [test_pool[int(rng.integers(0, len(test_pool)))] for _ in range(test_count)]
    multi_index = int(rng.integers(0, train_count))
    puzzle = {"train": [], "test": []}
    previous_background = None
    seen = set()
    for split, modes in (("train", train_modes), ("test", test_modes)):
        for index, mode in enumerate(modes):
            for _ in range(24):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, mode, split == "train" and index == multi_index,
                                  previous_background if split == "train" and index == 1 else None)
                key = (pair[0].shape, pair[0].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            seen.add(key)
            puzzle[split].append({"input": pair[0], "output": pair[1]})
            if split == "train" and index == 0:
                previous_background = _inspect_pair(*pair)["background"]
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(first, second):
    return all(np.array_equal(a[role], b[role])
               for split in ("train", "test")
               for a, b in zip(first[split], second[split])
               for role in ("input", "output")) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, profiles, layouts, sizes, colors = set(), set(), set(), set(), set(), set()
    modes, regimes = set(), set()
    compact_6x10 = compact_7x15 = landmark = border = corner = mixed = 0
    edge = one_margin = gap_one = gap_wide = unequal_gaps = offset = 0
    two_different = triple_repeated = short_steep = gradual = steep_jump = 0
    shared_layout = source_shift = reverse_shift = changing_background = 0
    combined = 0
    seen_puzzles = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        per_split_modes = {"train": set(), "test": set()}
        puzzle_backgrounds = set()
        fingerprint = tuple((p["input"].shape, p["input"].tobytes()) for split in ("train", "test") for p in puzzle[split])
        seen_puzzles.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _inspect_pair(pair["input"], pair["output"])
                mode = f["mode"]
                modes.add(mode)
                per_split_modes[split].add(mode)
                puzzle_backgrounds.add(f["background"])
                shape = f["components"][0][3]
                widths = tuple(int(row.sum()) for row in shape)
                short_steep += widths == (1, 3, 3, 1)
                gradual += widths == (1, 2, 3, 4, 3, 2, 1)
                steep_jump += not mode.startswith("stack") and any(abs(a - b) == 2 for a, b in zip(widths, widths[1:]))
                profiles.add((shape.shape, shape.tobytes()))
                layouts.add((mode, pair["output"].shape,
                             tuple((c[0], c[1]) for c in f["components"])))
                sizes.add(pair["output"].shape)
                colors.add((f["background"], tuple(c[2] for c in f["components"])))
                compact_6x10 += pair["output"].shape == (6, 10)
                compact_7x15 += pair["output"].shape == (7, 15)
                landmark += f["landmark_damage"]
                border += f["border_noise"]
                corner += f["corner_noise"]
                mixed += f["inside_missing"] > 0 and f["inside_foreign"] > 0
                components = f["components"]
                fill_count = len({c[2] for c in components})
                two_different += len(components) == 2 and fill_count == 2
                triple_repeated += len(components) == 3 and fill_count == 2
                tops = [c[0] for c in components]
                lefts = [c[1] for c in components]
                margins = (min(tops), min(lefts),
                           pair["output"].shape[0] - max(t + shape.shape[0] for t in tops),
                           pair["output"].shape[1] - max(c + shape.shape[1] for c in lefts))
                edge += 0 in margins
                one_margin += 1 in margins
                if mode.startswith("stack"):
                    gaps = [tops[i + 1] - tops[i] - shape.shape[0] for i in range(len(components) - 1)]
                    offset += len(set(lefts)) > 1
                else:
                    gaps = [lefts[i + 1] - lefts[i] - shape.shape[1] for i in range(len(components) - 1)]
                    offset += len(set(tops)) > 1
                gap_one += 1 in gaps
                gap_wide += any(g >= 5 for g in gaps)
                unequal_gaps += len(gaps) == 2 and gaps[0] != gaps[1]
                combined += widths == (1, 3, 3, 1) and 0 in margins and f["inside_missing"] > 0 and f["inside_foreign"] > 0
                regime = (f["inside_missing"] > 0, f["inside_foreign"] > 0,
                          f["outside_fill"] > 0, f["outside_foreign"] > 0)
                regimes.add(regime)
        changing_background += len(puzzle_backgrounds) > 1
        shared_layout += bool(per_split_modes["train"] & per_split_modes["test"])
        source_shift += per_split_modes["train"] <= {"stack_reflect", "side_reflect"} and bool(per_split_modes["test"] & {"stack_same", "side_same", "triple_same"})
        reverse_shift += bool(per_split_modes["train"] & {"stack_same", "side_same", "triple_same"}) and bool(per_split_modes["test"] & {"stack_reflect", "side_reflect"})
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and len(seen_puzzles) == 42
    assert len(profiles) >= 15 and len(layouts) >= 100 and len(sizes) >= 20 and len(colors) >= 50
    assert modes == {"stack_reflect", "stack_same", "side_reflect", "side_same", "triple_same"}
    assert compact_6x10 and compact_7x15 and landmark and border and corner and mixed
    assert edge and one_margin and gap_one and gap_wide and unequal_gaps and offset
    assert two_different and triple_repeated and short_steep and gradual and steep_jump
    assert shared_layout and source_shift and reverse_shift and changing_background == 42
    assert combined
    assert (False, False, False, True) in regimes
    assert len(regimes) >= 10
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (2, 3),
                                          (None, 1), (None, 4), (2, None), (10, None))):
        if train == 2 and test == 3:
            try:
                generate_puzzle(1000 + index, train, test)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid counts accepted")
            continue
        puzzle = generate_puzzle(1000 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(1000 + index, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
                        (True, 1), (2, False)):
        try:
            generate_puzzle(2000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
