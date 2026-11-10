"""Aligned two-panel symmetric-difference ARC puzzle generator."""

import numpy as np


_SOURCE = (
    ("0100/1011/1110/1110/4444/0000/0202/2202/0200", "0300/3330/0033/3030"),
    ("1100/1010/1101/0110/4444/0222/2020/2222/2222", "3033/0000/0030/3003"),
    ("1011/0001/1100/0011/4444/0222/0222/2022/2222", "3300/0330/0333/3300"),
    ("0101/0001/1010/0001/4444/0202/0002/2002/2220", "0000/0000/0033/3333"),
    ("1011/0111/0010/1011/4444/2202/0020/2002/0202", "0330/0303/3033/3330"),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _array_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _pair_key(pair: dict) -> tuple:
    return (_array_key(pair["input"]), _array_key(pair["output"]))


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    a = "/".join("".join(str(int(x)) for x in row) for row in input_grid)
    b = "/".join("".join(str(int(x)) for x in row) for row in output_grid)
    return (a, b) in _SOURCE


def _validate_optional_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _validate_optional_count(num_train, "num_train", 2, 10)
    test = _validate_optional_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if test is None and train is None:
        test = int(rng.integers(1, 5))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _sample_layout(rng: np.random.Generator, orientation: str | None = None) -> tuple:
    if orientation is None:
        orientation = ("horizontal", "vertical")[int(rng.integers(2))]
    mode = int(rng.integers(5))
    if mode == 0:  # original-scale, including 4 by 4
        height = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
    elif mode == 1:  # narrow and elongated
        if orientation == "horizontal":
            height = int(rng.integers(2, 5))
            width = int(rng.integers(9, 26))
        else:
            height = int(rng.integers(9, 26))
            width = int(rng.integers(2, 5))
    elif mode == 2:  # tall or wide, with the long side along the divider
        if orientation == "horizontal":
            height = int(rng.integers(8, 15))
            width = int(rng.integers(2, 5))
        else:
            height = int(rng.integers(2, 5))
            width = int(rng.integers(8, 15))
    elif mode == 3:
        height = int(rng.integers(9, 15))
        width = int(rng.integers(9, 15))
    else:
        height = int(rng.integers(3, 15))
        width = int(rng.integers(3, 15))
    return orientation, height, width


def _sample_palette(rng: np.random.Generator, result: int) -> tuple:
    choices = np.array([c for c in range(10) if c != result], dtype=np.int8)
    colors = rng.choice(choices, size=4, replace=False)
    return tuple(int(c) for c in colors)  # background, first ink, second ink, divider


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    result = int(rng.integers(10))
    simple = bool(rng.random() < 0.16)
    return {
        "result": result,
        "fixed_layout": simple or bool(rng.random() < 0.28),
        "fixed_palette": simple or bool(rng.random() < 0.32),
        "base_layout": ("horizontal", int(rng.integers(3, 7)),
                        int(rng.integers(3, 7))) if simple else _sample_layout(rng),
        "base_palette": _sample_palette(rng, result),
    }


def _flat_masks(rng: np.random.Generator, shape: tuple[int, int],
                teaching: bool) -> tuple[np.ndarray, np.ndarray]:
    count = shape[0] * shape[1]
    if teaching:
        xor_count = int(rng.integers(max(2, count // 4), min(count - 2, 3 * count // 4) + 1))
        first_only = int(rng.integers(1, xor_count))
        both = int(rng.integers(1, count - xor_count))
    else:
        fraction = float(rng.uniform(0.06, 0.94))
        xor_count = max(1, min(count - 1, round(fraction * count)))
        first_only = int(rng.integers(0, xor_count + 1))
        both = int(rng.integers(0, count - xor_count + 1))
    categories = np.repeat(np.arange(4, dtype=np.int8),
                           [count - xor_count - both, both,
                            first_only, xor_count - first_only])
    rng.shuffle(categories)
    categories = categories.reshape(shape)
    return np.isin(categories, (1, 2)), np.isin(categories, (1, 3))


def _shape_mask(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(int(rng.integers(1, 5))):
        style = int(rng.integers(4))
        if style == 0:  # block
            rh, rw = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
            row, col = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
            mask[row:row + rh, col:col + rw] = True
        elif style == 1:  # horizontal bar or run
            row = int(rng.integers(h))
            start = int(rng.integers(w))
            end = int(rng.integers(start + 1, w + 1))
            mask[row, start:end] = True
        elif style == 2:  # vertical bar or run
            col = int(rng.integers(w))
            start = int(rng.integers(h))
            end = int(rng.integers(start + 1, h + 1))
            mask[start:end, col] = True
        else:  # stepped silhouette
            row, col = int(rng.integers(h)), int(rng.integers(w))
            for step in range(int(rng.integers(2, max(h, w) + 2))):
                mask[row, col] = True
                if step % 2:
                    row = min(h - 1, row + 1)
                else:
                    col = min(w - 1, col + 1)
    if rng.random() < 0.30:
        mask = ~mask
    return mask


def _sample_masks(rng: np.random.Generator, shape: tuple[int, int],
                  teaching: bool, allow_mono: bool) -> tuple[np.ndarray, np.ndarray]:
    count = shape[0] * shape[1]
    for _ in range(120):
        mode = "flat" if teaching else rng.choice(
            ["flat", "silhouette", "edge"], p=[0.48, 0.32, 0.20])
        if mode == "flat":
            first, second = _flat_masks(rng, shape, teaching)
        elif mode == "silhouette":
            first = _shape_mask(rng, shape)
            second = (np.logical_xor(first, _shape_mask(rng, shape))
                      if rng.random() < 0.65 else _shape_mask(rng, shape))
        else:
            first = _shape_mask(rng, shape)
            kind = int(rng.integers(4))
            if kind == 0:  # identical
                second = first.copy()
            elif kind == 1:  # complementary
                second = ~first
            elif kind == 2:  # near-identical
                second = first.copy()
                indices = rng.choice(count, size=int(rng.integers(1, max(2, count // 9 + 1))), replace=False)
                second.flat[indices] = ~second.flat[indices]
            else:  # one empty or full panel
                second = np.full(shape, bool(rng.integers(2)), dtype=bool)
                if rng.random() < 0.5:
                    first, second = second, first
        a, b = int(first.sum()), int(second.sum())
        if teaching:
            both = int(np.count_nonzero(first & second))
            neither = int(np.count_nonzero(~first & ~second))
            only_a = int(np.count_nonzero(first & ~second))
            only_b = int(np.count_nonzero(~first & second))
            if min(both, neither, only_a, only_b) == 0:
                continue
        if not allow_mono and (a in (0, count) or b in (0, count)):
            continue
        if a in (0, count) and b in (0, count):
            continue
        return first, second
    raise RuntimeError("unable to sample decodable masks")


def _construct_grids(first: np.ndarray, second: np.ndarray, layout: tuple,
                     palette: tuple, result: int) -> tuple[np.ndarray, np.ndarray]:
    orientation, height, width = layout
    background, first_ink, second_ink, separator = palette
    first_panel = np.where(first, first_ink, background).astype(np.int8)
    second_panel = np.where(second, second_ink, background).astype(np.int8)
    if orientation == "horizontal":
        grid = np.full((2 * height + 1, width), separator, dtype=np.int8)
        grid[:height] = first_panel
        grid[height + 1:] = second_panel
    else:
        grid = np.full((height, 2 * width + 1), separator, dtype=np.int8)
        grid[:, :width] = first_panel
        grid[:, width + 1:] = second_panel
    output = np.where(first ^ second, result, background).astype(np.int8)
    return grid, output


def generate_pair(rng: np.random.Generator, regime: str, parameters: dict,
                  forbidden_inputs: set[tuple] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    if forbidden_inputs is None:
        forbidden_inputs = set()
    layout = parameters["layout"]
    for _ in range(128):
        first, second = _sample_masks(rng, layout[1:], parameters["teaching"],
                                      parameters["allow_mono"])
        grid, output = _construct_grids(first, second, layout,
                                        parameters["palette"], parameters["result"])
        if _array_key(grid) not in forbidden_inputs and not _is_source_pair(grid, output):
            return grid, output
    raise RuntimeError("unable to generate a unique pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    conventions = sample_puzzle_parameters(rng)
    result = conventions["result"]
    base_layout = conventions["base_layout"]
    base_palette = conventions["base_palette"]
    puzzle = {"train": [], "test": []}
    used: set[tuple] = set()
    layouts: list[tuple] = []
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if conventions["fixed_layout"]:
                layout = base_layout
            elif section == "train" and index == 0:
                layout = base_layout
            elif section == "train" and index == 1:
                orientation = ("vertical" if base_layout[0] == "horizontal"
                               else "horizontal")
                for _ in range(20):
                    layout = _sample_layout(rng, orientation)
                    if layout[1:] != base_layout[1:]:
                        break
            elif section == "test" and index == 0 and rng.random() < 0.55:
                layout = (layouts[0][0], layouts[1][1], layouts[1][2])
                if ((layout[0] == "horizontal" and 2 * layout[1] + 1 > 30)
                        or (layout[0] == "vertical" and 2 * layout[2] + 1 > 30)):
                    layout = _sample_layout(rng, layouts[0][0])
            else:
                layout = _sample_layout(rng)
            if conventions["fixed_palette"] or (section == "train" and index == 0):
                palette = base_palette
            else:
                for _ in range(20):
                    palette = _sample_palette(rng, result)
                    if section != "train" or index != 1 or sum(
                            a != b for a, b in zip(palette, base_palette)) >= 3:
                        break
            params = {
                "layout": layout,
                "palette": palette,
                "result": result,
                "teaching": section == "train" and index == 0,
                "allow_mono": conventions["fixed_palette"],
            }
            grid, output = generate_pair(rng, "source_train" if section == "train"
                                         else "target_test", params, used)
            used.add(_array_key(grid))
            layouts.append(layout)
            puzzle[section].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _split_input(grid: np.ndarray) -> tuple:
    h, w = grid.shape
    candidates = []
    if h % 2:
        center = h // 2
        color = int(grid[center, 0])
        if np.all(grid[center] == color) and np.count_nonzero(grid == color) == w:
            candidates.append(("horizontal", center, color))
    if w % 2:
        center = w // 2
        color = int(grid[0, center])
        if np.all(grid[:, center] == color) and np.count_nonzero(grid == color) == h:
            candidates.append(("vertical", center, color))
    _require(len(candidates) == 1, "input needs one exclusive central divider")
    orientation, center, divider = candidates[0]
    if orientation == "horizontal":
        first, second = grid[:center], grid[center + 1:]
    else:
        first, second = grid[:, :center], grid[:, center + 1:]
    _require(first.shape == second.shape, "panels must be congruent")
    return orientation, divider, first, second


def _infer_roles(first: np.ndarray, second: np.ndarray,
                 stable_palette: tuple | None) -> tuple:
    a = set(int(c) for c in np.unique(first))
    b = set(int(c) for c in np.unique(second))
    if len(a) == len(b) == 2 and len(a & b) == 1:
        background = next(iter(a & b))
        ink_a = next(iter(a - {background}))
        ink_b = next(iter(b - {background}))
        _require(ink_a != ink_b, "panel inks must differ")
        return background, ink_a, ink_b
    _require(stable_palette is not None, "monochrome panel needs learned stable palette")
    background, ink_a, ink_b = stable_palette[:3]
    _require(a <= {background, ink_a} and b <= {background, ink_b},
             "monochrome panel conflicts with learned palette")
    return background, ink_a, ink_b


def _decode_and_check_pair(input_grid: np.ndarray, output_grid: np.ndarray,
                           result: int, stable_palette: tuple | None) -> dict:
    orientation, divider, first, second = _split_input(input_grid)
    background, ink_a, ink_b = _infer_roles(first, second, stable_palette)
    _require(len({background, ink_a, ink_b, divider, result}) == 5,
             "semantic colors must be distinct within a pair")
    _require(output_grid.shape == first.shape, "output must be panel sized")
    mask_a, mask_b = first == ink_a, second == ink_b
    expected = np.where(mask_a ^ mask_b, result, background)
    _require(np.array_equal(output_grid, expected), "output is not aligned XOR")
    counts = (
        int(np.count_nonzero(~mask_a & ~mask_b)),
        int(np.count_nonzero(mask_a & mask_b)),
        int(np.count_nonzero(mask_a & ~mask_b)),
        int(np.count_nonzero(~mask_a & mask_b)),
    )
    return {
        "layout": (orientation, *first.shape),
        "palette": (background, ink_a, ink_b, divider),
        "counts": counts,
        "density": (counts[2] + counts[3]) / first.size,
        "mono": len(np.unique(first)) == 1 or len(np.unique(second)) == 1,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    inputs, pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must use integers")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid dimensions outside 1..30")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors outside 0..9")
            key = _array_key(pair["input"])
            pair_key = _pair_key(pair)
            _require(key not in inputs and pair_key not in pairs, "duplicate pair or input")
            inputs.add(key)
            pairs.add(pair_key)
            _require(not _is_source_pair(pair["input"], pair["output"]),
                     "generated pair copies the source")
    anchor = puzzle["train"][0]
    _, divider, first, second = _split_input(anchor["input"])
    background, ink_a, ink_b = _infer_roles(first, second, None)
    initial_palette = (background, ink_a, ink_b, divider)
    output_colors = set(int(c) for c in np.unique(anchor["output"]))
    _require(len(output_colors) == 2 and background in output_colors,
             "first training pair must reveal result and background")
    result = next(iter(output_colors - {background}))
    _require(len({*initial_palette, result}) == 5, "result role must be distinct")
    anchor_metadata = _decode_and_check_pair(anchor["input"], anchor["output"],
                                              result, None)
    _require(min(anchor_metadata["counts"]) > 0,
             "first training pair must teach all four XOR cases")
    decoded = {"train": [anchor_metadata], "test": []}
    for pair in puzzle["train"][1:]:
        decoded["train"].append(_decode_and_check_pair(
            pair["input"], pair["output"], result, initial_palette))
    stable = all(m["palette"] == initial_palette for m in decoded["train"])
    for pair in puzzle["test"]:
        decoded["test"].append(_decode_and_check_pair(
            pair["input"], pair["output"], result,
            initial_palette if stable else None))
    if not stable:
        _require(not any(m["mono"] for m in decoded["train"] + decoded["test"]),
                 "changing palettes cannot support monochrome panels")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds repeat puzzles")
    layouts, palettes, results = set(), set(), set()
    variable_layout_train = variable_palette_train = 0
    close = reverse = forward = 0
    identical = complementary = near = mono = 0
    narrow = large = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        anchor = puzzle["train"][0]
        _, _, a, b = _split_input(anchor["input"])
        bg, _, _ = _infer_roles(a, b, None)
        result = next(iter(set(int(x) for x in np.unique(anchor["output"])) - {bg}))
        results.add(result)
        train_meta, test_meta = [], []
        for section, dest in (("train", train_meta), ("test", test_meta)):
            for pair in puzzle[section]:
                _, _, x, y = _split_input(pair["input"])
                colors = set(int(v) for v in np.unique(pair["input"]))
                _require(result not in colors, "result appears in input")
                # Check the transformation again from visible grids, not sampling metadata.
                known = None
                if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
                    known = _decode_and_check_pair(anchor["input"], anchor["output"], result, None)["palette"]
                meta = _decode_and_check_pair(pair["input"], pair["output"], result, known)
                dest.append(meta)
                layouts.add(meta["layout"])
                palettes.add(meta["palette"])
                h, w = meta["layout"][1:]
                narrow += min(h, w) <= 3 and max(h, w) >= 9
                large += h >= 9 and w >= 9
                count = sum(meta["counts"])
                xor = meta["counts"][2] + meta["counts"][3]
                identical += xor == 0
                complementary += xor == count
                near += 0 < xor <= max(1, count // 9)
                mono += meta["mono"]
        variable_layout_train += len({m["layout"] for m in train_meta}) > 1
        variable_palette_train += len({m["palette"] for m in train_meta}) > 1
        train_mean = sum(m["density"] for m in train_meta) / len(train_meta)
        test_mean = sum(m["density"] for m in test_meta) / len(test_meta)
        close += abs(train_mean - test_mean) < 0.12
        reverse += train_mean > test_mean + 0.12
        forward += test_mean > train_mean + 0.12
    _require({x[0] for x in layouts} == {"horizontal", "vertical"}, "missing orientation")
    _require(len(layouts) >= 30 and len(palettes) >= 30 and len(results) >= 7,
             "layout or color diversity is weak")
    _require(variable_layout_train >= 15 and variable_palette_train >= 15,
             "within-training variation is weak")
    _require(min(close, reverse, forward) >= 3, "train/test densities lack regimes")
    _require(min(identical, complementary, near, mono, narrow, large) >= 2,
             "edge masks or panel dimensions lack coverage")
    _require(_puzzle_key(generate_puzzle(20260827)) == _puzzle_key(generate_puzzle(20260827)),
             "automatic generation is not deterministic")
    for seed, counts in ((314159, (7, 3)), (5000, (2, 1)),
                         (5001, (10, 1)), (5002, (10, 4))):
        a = generate_puzzle(seed, *counts)
        b = generate_puzzle(seed, *counts)
        _require(_puzzle_key(a) == _puzzle_key(b), "explicit generation is not deterministic")
        _require((len(a["train"]), len(a["test"])) == counts, "explicit counts changed")
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 6000):
        puzzle = generate_puzzle(seed, *counts)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "partial train count changed")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "partial test count changed")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4), (2, 4)):
        try:
            generate_puzzle(7000, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal count request did not raise ValueError")
