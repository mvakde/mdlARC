"""Noisy four-quadrant enlargement of a three-by-three cross."""

import hashlib
import numpy as np


_MIN_SIDE = 17
_MAX_SIDE = 30
_MIN_CORE = 4
_SOURCE_INPUT_HASHES = {
    "18ef6c056824f609db51eb7742d5d1e399dbf4488933639a52002094c79e91de",
    "40529415cc9b90fd373e137d899f9a801d52e30600e322dfb760727f48900e0b",
    "6cfc1c39bc9c8d998b3ce2a72c5bd2d2401fc8369ee4af22a868a41ab60eb480",
    "5639753eebe01de21f83fe557d1df52cad4fe329e3a7ddfd9b4a3de1be68a2ac",
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(rng, seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (not _integer(value) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        return test + 1 + (seed // 4) % (10 - test), test
    if num_train is None:
        return int(rng.integers(num_test + 1, 11)), int(num_test)
    return int(num_train), int(rng.integers(1, min(4, num_train - 1) + 1))


def _key(grid):
    return (grid.shape, grid.tobytes())


def _cross(grid):
    rows = np.flatnonzero(np.all(grid == grid[:, :1], axis=1))
    cols = np.flatnonzero(np.all(grid == grid[:1, :], axis=0))
    assert len(rows) == len(cols) == 1
    row, col = int(rows[0]), int(cols[0])
    color = int(grid[row, col])
    assert np.all(grid[row, :] == color) and np.all(grid[:, col] == color)
    return row, col, color


def _quadrants(grid, row, col):
    return (grid[:row, :col], grid[:row, col + 1:],
            grid[row + 1:, :col], grid[row + 1:, col + 1:])


def _mode(block):
    counts = np.bincount(block.reshape(-1), minlength=10)
    order = np.argsort(counts)
    color = int(order[-1])
    assert counts[color] >= 1.5 * max(1, counts[int(order[-2])])
    return color


def _output_from_input(grid):
    row, col, separator = _cross(grid)
    colors = [_mode(q) for q in _quadrants(grid, row, col)]
    assert separator not in colors
    return np.asarray([[colors[0], separator, colors[1]],
                       [separator, separator, separator],
                       [colors[2], separator, colors[3]]], dtype=np.int64)


def _largest_rectangle(block, color):
    """Area, dimensions, and origin of the largest solid rectangle."""
    heights = [0] * block.shape[1]
    best = (0, 0, 0, 0, 0)
    for bottom, row in enumerate(block):
        for col, value in enumerate(row):
            heights[col] = heights[col] + 1 if value == color else 0
        stack = []
        for col in range(len(heights) + 1):
            current = heights[col] if col < len(heights) else 0
            start = col
            while stack and stack[-1][1] > current:
                left, height = stack.pop()
                width = col - left
                if height * width > best[0]:
                    best = (height * width, height, width, bottom-height+1, left)
                start = left
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def _sample_panels(rng, separator, repeat, required=None, allowed=None):
    available = [x for x in (range(10) if allowed is None else allowed) if x != separator]
    if required is not None:
        assert required in available
        available.remove(required)
        colors = [required] + [int(x) for x in rng.choice(available, size=2 if repeat else 3, replace=False)]
    else:
        colors = [int(x) for x in rng.choice(available, size=3 if repeat else 4, replace=False)]
    if repeat:
        colors.append(colors[0])
        if rng.integers(2):
            colors = [colors[1], colors[0], colors[0], colors[2]]
    else:
        rng.shuffle(colors)
    return tuple(colors)


def _sample_bands(rng, wide):
    bands = [1] * 8
    if wide:
        for i in range(8):
            if rng.random() < 0.33:
                bands[i] = 2
        if sum(x == 2 for x in bands) < 2:
            for i in rng.permutation(8)[:2]:
                bands[int(i)] = 2
    return bands  # top outer/inner, bottom inner/outer, left outer/inner, right inner/outer


def _axis_split(rng, side, outer_a, inner_a, inner_b, outer_b, forced=None):
    low = outer_a + inner_a + _MIN_CORE
    high = side - 1 - inner_b - outer_b - _MIN_CORE
    if low > high:
        return None
    if forced is not None:
        return forced if low <= forced <= high else None
    # Endpoints provide strong offsets on large grids; the remaining draws
    # cover every legal intermediate position. Rows and columns draw alone.
    if rng.random() < 0.4:
        return low if rng.integers(2) == 0 else high
    return int(rng.integers(low, high + 1))


def _noise_palette(rng, allowed):
    size = int(rng.integers(2, len(allowed) + 1))
    palette = rng.choice(allowed, size=size, replace=False)
    concentration = float(rng.uniform(0.35, 2.5))
    probabilities = rng.dirichlet(np.full(size, concentration))
    return palette, probabilities


def _paint_noise(rng, grid, bands, style, signal):
    height, width = grid.shape
    top_outer, _, _, bottom_outer, left_outer, _, _, right_outer = bands
    outer = np.zeros(grid.shape, dtype=bool)
    outer[:top_outer, :] = True
    outer[height - bottom_outer:, :] = True
    outer[:, :left_outer] = True
    outer[:, width - right_outer:] = True
    outer &= grid == -1
    inner = (grid == -1) & ~outer
    extras = [color for color in range(10) if color not in signal]
    if style == 0:
        allowed = extras
    elif style == 1:
        selected = rng.choice(sorted(signal), size=int(rng.integers(1, 3)), replace=False)
        allowed = extras + [int(x) for x in selected]
    else:
        allowed = list(range(10))
    for mask in (outer, inner):
        palette, probabilities = _noise_palette(rng, allowed)
        grid[mask] = rng.choice(palette, size=int(mask.sum()), p=probabilities)
        # Short runs along either type of margin, without touching a core.
        for axis in (0, 1):
            for line in range(grid.shape[axis]):
                if rng.random() >= 0.14:
                    continue
                indices = np.flatnonzero(mask[line, :] if axis == 0 else mask[:, line])
                if len(indices) < 2:
                    continue
                start = int(rng.choice(indices))
                length = int(rng.integers(2, 5))
                value = int(rng.choice(palette, p=probabilities))
                for position in range(start, min(start + length, grid.shape[1 - axis])):
                    if axis == 0 and mask[line, position]:
                        grid[line, position] = value
                    elif axis == 1 and mask[position, line]:
                        grid[position, line] = value
    assert not np.any(grid == -1)


def _generate_pair(rng, spec, repeated_layout=None):
    for attempt in range(120):
        if repeated_layout is None:
            height = int(rng.integers(_MIN_SIDE, _MAX_SIDE + 1))
            width = int(rng.integers(_MIN_SIDE, _MAX_SIDE + 1))
        else:
            height, width = repeated_layout[:2]
        bands = _sample_bands(rng, spec["wide"])
        row = _axis_split(rng, height, *bands[:4],
                          forced=repeated_layout[2] if repeated_layout is not None else None)
        col = _axis_split(rng, width, *bands[4:],
                          forced=repeated_layout[3] if repeated_layout is not None else None)
        if row is None or col is None:
            continue
        if spec.get("equal_split"):
            common_low = max(bands[0] + bands[1] + _MIN_CORE,
                             bands[4] + bands[5] + _MIN_CORE)
            common_high = min(height - 1 - bands[2] - bands[3] - _MIN_CORE,
                              width - 1 - bands[6] - bands[7] - _MIN_CORE)
            if common_low > common_high:
                continue
            row = col = int(rng.integers(common_low, common_high + 1))
        if spec.get("shifted") and row == col:
            continue
        grid = np.full((height, width), -1, dtype=np.int64)
        t_outer, t_inner, b_inner, b_outer, l_outer, l_inner, r_inner, r_outer = bands
        slices = (
            (slice(t_outer, row - t_inner), slice(l_outer, col - l_inner)),
            (slice(t_outer, row - t_inner), slice(col + 1 + r_inner, width - r_outer)),
            (slice(row + 1 + b_inner, height - b_outer), slice(l_outer, col - l_inner)),
            (slice(row + 1 + b_inner, height - b_outer), slice(col + 1 + r_inner, width - r_outer)),
        )
        for region, color in zip(slices, spec["panels"]):
            grid[region] = color
        grid[row, :] = spec["separator"]
        grid[:, col] = spec["separator"]
        _paint_noise(rng, grid, bands, spec["noise_style"],
                     {spec["separator"], *spec["panels"]})
        try:
            output = _output_from_input(grid)
            info = _analyze_pair(grid, output)
            if spec["wide"] and not _has_wide(info):
                continue
        except AssertionError:
            continue
        if height == width == 19 and hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest() in _SOURCE_INPUT_HASHES:
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct a clear cross and four cores")


def _analyze_pair(grid, output):
    height, width = grid.shape
    assert _MIN_SIDE <= height <= _MAX_SIDE and _MIN_SIDE <= width <= _MAX_SIDE
    assert output.shape == (3, 3)
    row, col, separator = _cross(grid)
    quadrants = _quadrants(grid, row, col)
    colors = []
    core_dims = []
    for quadrant in quadrants:
        color = _mode(quadrant)
        area, core_height, core_width, _, _ = _largest_rectangle(quadrant, color)
        assert core_height >= _MIN_CORE and core_width >= _MIN_CORE
        assert area >= 0.20 * quadrant.size
        assert area > max(_largest_rectangle(quadrant, other)[0]
                          for other in range(10) if other != color)
        colors.append(color)
        core_dims.append((core_height, core_width))
    assert separator not in colors
    assert len(set(colors)) in (3, 4)
    if len(set(colors)) == 3:
        assert (colors[0] == colors[3]) != (colors[1] == colors[2])
    expected = np.asarray([[colors[0], separator, colors[1]],
                           [separator, separator, separator],
                           [colors[2], separator, colors[3]]], dtype=np.int64)
    assert np.array_equal(output, expected)
    if height == width == 19:
        assert hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest() not in _SOURCE_INPUT_HASHES
    return {"shape": (height, width), "split": (row, col), "separator": separator,
            "panels": tuple(colors), "core_dims": tuple(core_dims),
            "repeated": len(set(colors)) == 3}


def _has_wide(info):
    height, width = info["shape"]
    row, col = info["split"]
    sizes = ((row, col), (row, width-col-1),
             (height-row-1, col), (height-row-1, width-col-1))
    return any(qh-ch >= 3 or qw-cw >= 3
               for (qh, qw), (ch, cw) in zip(sizes, info["core_dims"]))


def _noise_colors(grid):
    row, col, _ = _cross(grid)
    mask = np.ones(grid.shape, dtype=bool)
    mask[row, :] = False
    mask[:, col] = False
    offsets = ((0, 0), (0, col+1), (row+1, 0), (row+1, col+1))
    for block, (row0, col0) in zip(_quadrants(grid, row, col), offsets):
        color = _mode(block)
        _, height, width, top, left = _largest_rectangle(block, color)
        mask[row0+top:row0+top+height, col0+left:col0+left+width] = False
    return {int(x) for x in np.unique(grid[mask])}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys = set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(pair["input"]) not in input_keys
            input_keys.add(_key(pair["input"]))
            assert pair["input"].shape != pair["output"].shape
            analyses[section].append(_analyze_pair(pair["input"], pair["output"]))
    # All conventions are visible in a training input/output: cross color,
    # corner order, compression, and any requested extension in a test.
    train = analyses["train"]
    test = analyses["test"]
    assert any(not item["repeated"] for item in train)
    if any(item["repeated"] for item in test):
        assert any(item["repeated"] for item in train)
    if any(_has_wide(info) for info in test):
        assert any(_has_wide(info) for info in train)
    for pair in puzzle["test"]:
        assert np.array_equal(_output_from_input(pair["input"].copy()), pair["output"])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    # These are construction preferences, not hidden rules. Every color role
    # and split is explicitly recoverable from its own input.
    palette = [int(x) for x in rng.permutation(10)]
    role_style = int(rng.integers(3))  # unrestricted, training swap, or test swap
    separator_pool = palette[:3]
    panel_pool = palette[3:]
    train_repeat = [False] + [bool(rng.random() < 0.18) for _ in range(train_count - 1)]
    test_repeat = [bool(rng.random() < 0.16) for _ in range(test_count)]
    train_wide = [False] + [bool(rng.random() < 0.30) for _ in range(train_count - 1)]
    test_wide = [bool(rng.random() < 0.24) for _ in range(test_count)]
    if any(test_repeat):
        train_repeat[1] = True
    if any(test_wide):
        train_wide[1] = True
    specs = {"train": [], "test": []}
    for index in range(train_count):
        if index == 1 and role_style == 1:
            separator = specs["train"][0]["panels"][int(rng.integers(4))]
            required = specs["train"][0]["separator"]
        else:
            separator = int(rng.choice(palette if role_style == 1 else separator_pool))
            required = None
        specs["train"].append({"separator": separator,
                               "panels": _sample_panels(rng, separator, train_repeat[index], required,
                                                        None if role_style == 1 else panel_pool),
                               "wide": train_wide[index],
                               "noise_style": int(rng.choice(3, p=(0.18, 0.32, 0.50)))})
    train_panel_colors = sorted({color for spec in specs["train"] for color in spec["panels"]})
    for index in range(test_count):
        if index == 0 and role_style == 2:
            separator = int(rng.choice(train_panel_colors))
            required = specs["train"][0]["separator"]
            if separator == required:
                alternatives = [x for x in train_panel_colors if x != required]
                separator = int(rng.choice(alternatives))
        else:
            separator = int(rng.choice(palette if role_style == 1 else separator_pool))
            required = None
        specs["test"].append({"separator": separator,
                              "panels": _sample_panels(rng, separator, test_repeat[index], required,
                                                       None if required is not None or role_style == 1 else panel_pool),
                              "wide": test_wide[index],
                              "noise_style": int(rng.choice(3, p=(0.18, 0.32, 0.50)))})
    puzzle = {"train": [], "test": []}
    used = set()
    train_layouts = []
    for section in ("train", "test"):
        for index, spec in enumerate(specs[section]):
            if section == "test":
                style = int(rng.integers(4))
                spec["equal_split"] = style == 1
                spec["shifted"] = style == 0
                repeated = train_layouts[int(rng.integers(len(train_layouts)))] if style == 2 else None
            else:
                repeated = None
            for _ in range(30):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _generate_pair(child, spec, repeated)
                if _key(pair["input"]) not in used:
                    break
            else:
                raise RuntimeError("failed to create a unique input")
            used.add(_key(pair["input"]))
            puzzle[section].append(pair)
            if section == "train":
                info = _analyze_pair(pair["input"], pair["output"])
                train_layouts.append(info["shape"] + info["split"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_using_training(train_pairs, test_input):
    # Check the demonstrated correspondence, then extract the same spatial
    # slots from the test with a rectangle criterion independent of the mode.
    for pair in train_pairs:
        grid, output = pair["input"], pair["output"]
        row, col, sep = _cross(grid)
        assert np.all(output[1, :] == sep) and np.all(output[:, 1] == sep)
        for quadrant, target in zip(_quadrants(grid, row, col),
                                    (output[0, 0], output[0, 2], output[2, 0], output[2, 2])):
            areas = [_largest_rectangle(quadrant, color)[0] for color in range(10)]
            assert int(np.argmax(areas)) == int(target)
    row, col, sep = _cross(test_input)
    colors = []
    for quadrant in _quadrants(test_input, row, col):
        areas = [_largest_rectangle(quadrant, color)[0] for color in range(10)]
        assert np.count_nonzero(areas == np.max(areas)) == 1
        colors.append(int(np.argmax(areas)))
    return np.asarray([[colors[0], sep, colors[1]], [sep, sep, sep],
                       [colors[2], sep, colors[3]]], dtype=np.int64)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, splits, separators, input_keys = set(), set(), set(), set()
    repeated, width_extensions, combined, source_size = 0, 0, 0, 0
    all_noise_sizes, no_signal_noise, signal_noise = set(), 0, 0
    different_band_palettes, zero_absent_from_noise = 0, 0
    outer_dominances = []
    train_exchanges, test_only_exchanges, no_exchanges = 0, 0, 0
    reused_test_layouts, equal_test_splits, shifted_test_layouts = 0, 0, 0
    strong_splits, balanced_splits = 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_info = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_exchange = any(a["separator"] in b["panels"]
                             for i, a in enumerate(train_info) for j, b in enumerate(train_info) if i != j)
        test_exchange = any(a["separator"] in b["panels"] or b["separator"] in a["panels"]
                            for a in test_info for b in train_info)
        train_exchanges += train_exchange
        test_only_exchanges += test_exchange and not train_exchange
        no_exchanges += not train_exchange and not test_exchange
        train_layouts = {m["shape"] + m["split"] for m in train_info}
        reused_test_layouts += sum(m["shape"] + m["split"] in train_layouts for m in test_info)
        equal_test_splits += sum(m["split"][0] == m["split"][1] for m in test_info)
        shifted_test_layouts += sum(m["split"][0] != m["split"][1]
                                    and m["shape"] + m["split"] not in train_layouts
                                    for m in test_info)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                shapes.add(info["shape"])
                splits.add(info["shape"] + info["split"])
                separators.add(info["separator"])
                input_keys.add(_key(pair["input"]))
                repeated += int(info["repeated"])
                combined += int(info["repeated"] and _has_wide(info))
                source_size += int(info["shape"] == (19, 19))
                width_extensions += int(_has_wide(info))
                h, w = info["shape"]
                row, col = info["split"]
                strong_splits += int(abs(row/(h-1)-0.5) > 0.26 or abs(col/(w-1)-0.5) > 0.26)
                balanced_splits += int(abs(row/(h-1)-0.5) < 0.12 and abs(col/(w-1)-0.5) < 0.12)
                noise = _noise_colors(pair["input"])
                all_noise_sizes.add(len(noise))
                signal = {info["separator"], *info["panels"]}
                no_signal_noise += int(not noise.intersection(signal))
                signal_noise += int(bool(noise.intersection(signal)))
                zero_absent_from_noise += int(0 in signal and 0 not in noise)
                outer_mask = np.ones((h, w), dtype=bool)
                outer_mask[1:-1, 1:-1] = False
                outer_mask[row, :] = False
                outer_mask[:, col] = False
                inner_mask = np.zeros((h, w), dtype=bool)
                inner_mask[row-1, :] = True
                inner_mask[row+1, :] = True
                inner_mask[:, col-1] = True
                inner_mask[:, col+1] = True
                inner_mask[0, :] = False
                inner_mask[-1, :] = False
                inner_mask[:, 0] = False
                inner_mask[:, -1] = False
                inner_mask[row, :] = False
                inner_mask[:, col] = False
                outer_values = pair["input"][outer_mask]
                inner_values = pair["input"][inner_mask]
                different_band_palettes += int(set(outer_values) != set(inner_values))
                outer_dominances.append(np.bincount(outer_values, minlength=10).max() / len(outer_values))
        train_copies = [{"input": p["input"].copy(), "output": p["output"].copy()} for p in puzzle["train"]]
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_using_training(train_copies, pair["input"].copy()), pair["output"])
    assert len(shapes) >= 40 and len(splits) >= 70 and len(separators) == 10
    assert len(input_keys) == sum(len(p[s]) for p in puzzles for s in ("train", "test"))
    assert repeated >= 8 and width_extensions >= 8 and combined >= 3 and source_size >= 1
    assert train_exchanges >= 3 and test_only_exchanges >= 3 and no_exchanges >= 3
    assert reused_test_layouts >= 3 and equal_test_splits >= 3 and shifted_test_layouts >= 3
    assert strong_splits >= 12 and balanced_splits >= 12
    assert len(all_noise_sizes) >= 5 and no_signal_noise >= 3 and signal_noise >= 3
    assert zero_absent_from_noise >= 3 and different_band_palettes >= 20
    assert min(outer_dominances) < 0.3 and max(outer_dominances) > 0.7
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1101, None, 1), (1102, None, 4),
                              (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    first = generate_puzzle(918273, 7, 3)
    second = generate_puzzle(918273, 7, 3)
    assert all(np.array_equal(a[k], b[k]) for s in ("train", "test")
               for a, b in zip(first[s], second[s]) for k in ("input", "output"))
    different = generate_puzzle(918274, 7, 3)
    assert not np.array_equal(first["train"][0]["input"], different["train"][0]["input"])
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError: {kwargs}")
