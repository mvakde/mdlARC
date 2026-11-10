"""Synthetic puzzles for the fixed 2-by-2 cell substitution task."""

import numpy as np


_SOURCE_INPUTS = (
    ((5, 0, 0), (0, 5, 0), (0, 0, 5)),
    ((0, 5, 0), (5, 5, 5), (0, 5, 0)),
    ((0, 5, 0), (0, 5, 5), (5, 5, 0)),
    ((0, 0, 0), (0, 5, 0), (5, 5, 5)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    bounds = (2, 10) if name == "num_train" else (1, 4)
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _validate_count("num_train", num_train)
    test = _validate_count("num_test", num_test)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.ravel()))


def _make_output(input_grid: np.ndarray, background: int, marker: int,
                 accent_a: int, accent_b: int) -> np.ndarray:
    height, width = input_grid.shape
    output = np.full((2 * height, 2 * width), background, dtype=np.int8)
    rows, cols = np.nonzero(input_grid == marker)
    output[2 * rows, 2 * cols] = accent_a
    output[2 * rows, 2 * cols + 1] = accent_b
    output[2 * rows + 1, 2 * cols] = accent_b
    output[2 * rows + 1, 2 * cols + 1] = accent_a
    return output


def _source_pair_signatures() -> set[tuple]:
    result = set()
    for source in _SOURCE_INPUTS:
        inp = np.asarray(source, dtype=np.int8)
        out = _make_output(inp, 0, 5, 1, 2)
        result.add((_signature(inp), _signature(out)))
    return result


_SOURCE_PAIRS = _source_pair_signatures()


def _sample_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "background": colors[0], "marker": colors[1],
        "accent_a": colors[2], "accent_b": colors[3],
        "color_mode": ("fixed", "background", "marker")[int(rng.choice(3, p=(.38, .31, .31)))],
        "layout_mode": ("source_shift", "mixed", "size_shift")[int(rng.choice(3, p=(.22, .55, .23)))],
    }


def _sample_shape(rng: np.random.Generator, regime: str, layout_mode: str,
                  demonstration: bool = False) -> tuple[int, int]:
    if layout_mode == "size_shift":
        if regime == "source_train":
            height, width = int(rng.integers(2, 9)), int(rng.integers(2, 9))
        else:
            height, width = int(rng.integers(9, 16)), int(rng.integers(9, 16))
    else:
        kind = int(rng.choice(6, p=(.20, .17, .30, .18, .11, .04)))
        if kind == 0:
            height, width = 3, 3
        elif kind == 1:
            height, width = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        elif kind == 2:
            height, width = int(rng.integers(2, 11)), int(rng.integers(2, 11))
        elif kind == 3:
            height, width = int(rng.integers(6, 16)), int(rng.integers(6, 16))
        elif kind == 4:
            length = int(rng.integers(2, 16))
            height, width = (1, length) if rng.random() < .5 else (length, 1)
        else:
            height, width = 15, 15
    if demonstration:
        height, width = max(height, 4 if layout_mode == "mixed" else 2), max(width, 2)
    if layout_mode == "source_shift":
        if regime == "source_train":
            height, width = max(height, 2), max(width, 2)
        else:
            height = max(height, 3)
    return height, width


def _sample_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    # Motifs are drawn in a randomly inset window. Its four margins are sampled
    # separately, so position does not determine size, density, or connectivity.
    if rng.random() < .45:
        top = bottom = left = right = 0
    else:
        top = int(rng.integers(0, min(4, height)))
        left = int(rng.integers(0, min(4, width)))
        bottom = int(rng.integers(0, min(4, height - top)))
        right = int(rng.integers(0, min(4, width - left)))
    h, w = height - top - bottom, width - left - right
    area = h * w
    motif = np.zeros((h, w), dtype=bool)
    style = int(rng.integers(0, 9))
    if style == 0:  # scattered components, sometimes a single mark
        count = int(rng.integers(1, max(2, min(area, area // 3 + 1) + 1)))
        motif.ravel()[rng.choice(area, size=min(count, area), replace=False)] = True
    elif style in (1, 2):  # sparse and dense independent occupancies
        probability = float(rng.uniform(.06, .38) if style == 1 else rng.uniform(.4, .88))
        motif = rng.random((h, w)) < probability
    elif style == 3:  # row, column, or diagonal segment
        direction = int(rng.integers(0, 4))
        span = (w, h, min(h, w), min(h, w))[direction]
        length = int(rng.integers(1, span + 1))
        if direction == 0:
            row, col = int(rng.integers(0, h)), int(rng.integers(0, w - length + 1))
        elif direction == 1:
            row, col = int(rng.integers(0, h - length + 1)), int(rng.integers(0, w))
        else:
            row = int(rng.integers(0, h - length + 1))
            col = int(rng.integers(0, w - length + 1)) if direction == 2 else int(rng.integers(length - 1, w))
        dr, dc = ((0, 1), (1, 0), (1, 1), (1, -1))[direction]
        for offset in range(length):
            motif[row + dr * offset, col + dc * offset] = True
    elif style == 4:  # filled, irregular rectangle
        r0, r1 = sorted((int(rng.integers(0, h)), int(rng.integers(0, h))))
        c0, c1 = sorted((int(rng.integers(0, w)), int(rng.integers(0, w))))
        motif[r0:r1 + 1, c0:c1 + 1] = True
        motif &= rng.random((h, w)) >= float(rng.uniform(0, .28))
    elif style == 5:  # connected random walk
        row, col = int(rng.integers(0, h)), int(rng.integers(0, w))
        for _ in range(int(rng.integers(1, max(3, 2 * area + 1)))):
            motif[row, col] = True
            dr, dc = ((0, 1), (1, 0), (0, -1), (-1, 0))[int(rng.integers(0, 4))]
            row, col = int(np.clip(row + dr, 0, h - 1)), int(np.clip(col + dc, 0, w - 1))
    elif style == 6:  # cross or rectangular ring
        if rng.random() < .5 or h < 3 or w < 3:
            row, col = int(rng.integers(0, h)), int(rng.integers(0, w))
            motif[row, :] = True
            motif[:, col] = True
        else:
            r0 = int(rng.integers(0, h - 2))
            r1 = int(rng.integers(r0 + 2, h))
            c0 = int(rng.integers(0, w - 2))
            c1 = int(rng.integers(c0 + 2, w))
            motif[r0:r1 + 1, c0] = True
            motif[r0:r1 + 1, c1] = True
            motif[r0, c0:c1 + 1] = True
            motif[r1, c0:c1 + 1] = True
    elif style == 7:  # nearly filled window
        motif[:] = True
        holes = int(rng.integers(1, max(2, area // 6 + 1)))
        motif.ravel()[rng.choice(area, size=min(holes, area), replace=False)] = False
    else:  # several separately placed horizontal or vertical bands
        horizontal = rng.random() < .5
        for _ in range(int(rng.integers(1, 5))):
            if horizontal:
                row = int(rng.integers(0, h))
                start = int(rng.integers(0, w))
                end = int(rng.integers(start + 1, w + 1))
                motif[row, start:end] = True
            else:
                col = int(rng.integers(0, w))
                start = int(rng.integers(0, h))
                end = int(rng.integers(start + 1, h + 1))
                motif[start:end, col] = True
    if area > 3 and rng.random() < .23:
        flips = int(rng.integers(1, max(2, area // 10 + 1)))
        motif.ravel()[rng.choice(area, size=flips, replace=False)] ^= True
    mask = np.zeros((height, width), dtype=bool)
    mask[top:height - bottom, left:width - right] = motif
    return mask


def _sample_pair_colors(rng: np.random.Generator, parameters: dict,
                        index: int, training: bool) -> tuple[int, int]:
    background, marker = parameters["background"], parameters["marker"]
    mode = parameters["color_mode"]
    if mode == "fixed":
        return background, marker
    fixed = marker if mode == "background" else background
    choices = [c for c in range(10) if c not in
               (fixed, parameters["accent_a"], parameters["accent_b"])]
    base = background if mode == "background" else marker
    if index == 0 and training:
        variable = base
    elif index == 1 and training:
        variable = int(rng.choice([c for c in choices if c != base]))
    else:
        variable = int(rng.choice(choices))
    return (variable, marker) if mode == "background" else (background, variable)


def _sample_pair(rng: np.random.Generator, regime: str, parameters: dict,
                 pair_index: int, adjacency_index: int, empty_demo_index: int) -> dict:
    mode = parameters["layout_mode"]
    demonstration = regime == "source_train" and pair_index in (adjacency_index, empty_demo_index)
    height, width = _sample_shape(rng, regime, mode, demonstration)
    mask = _sample_mask(rng, height, width)
    empty_row = None
    if regime == "source_train":
        if mode == "source_shift":
            for row in range(height):
                if not np.any(mask[row]):
                    mask[row, int(rng.integers(0, width))] = True
        elif mode == "mixed" and pair_index == empty_demo_index:
            empty_row = int(rng.integers(1, height - 1))
            mask[empty_row, :] = False
            mask[int(rng.integers(0, empty_row)), int(rng.integers(0, width))] = True
            mask[int(rng.integers(empty_row + 1, height)), int(rng.integers(0, width))] = True
        if pair_index == adjacency_index:
            eligible = [r for r in range(height) if r != empty_row]
            row = int(rng.choice(eligible))
            col = int(rng.integers(0, width - 1))
            mask[row, col:col + 2] = True
        if not np.any(mask):
            mask[int(rng.integers(0, height)), int(rng.integers(0, width))] = True
        if np.all(mask):
            row = int(rng.integers(0, height))
            col = int(rng.integers(0, width))
            mask[row, col] = False
    else:
        test_choice = rng.random()
        if mode != "source_shift" and test_choice < .05:
            mask[:] = False
        elif mode != "source_shift" and test_choice < .10:
            mask[:] = True
        elif mode == "source_shift" or (mode == "mixed" and test_choice < .35):
            height = max(height, 3)
            if mask.shape[0] != height:
                extended = np.zeros((height, width), dtype=bool)
                extended[:mask.shape[0], :] = mask
                mask = extended
            empty_row = int(rng.integers(0, height))
            mask[empty_row, :] = False
            if not np.any(mask):
                row = int(rng.choice([r for r in range(height) if r != empty_row]))
                mask[row, int(rng.integers(0, width))] = True
        elif (mode == "mixed" and test_choice < .70) or (mode == "size_shift" and test_choice < .55):
            for row in range(height):
                if not np.any(mask[row]):
                    mask[row, int(rng.integers(0, width))] = True
    background, marker = _sample_pair_colors(rng, parameters, pair_index, regime == "source_train")
    input_grid = np.full(mask.shape, background, dtype=np.int8)
    input_grid[mask] = marker
    return {"input": input_grid,
            "output": _make_output(input_grid, background, marker,
                                   parameters["accent_a"], parameters["accent_b"])}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    parameters = _sample_parameters(rng)
    adjacency_index = int(rng.integers(0, train_count))
    empty_demo_index = int(rng.integers(0, train_count))
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_layouts = set(), set()
    for regime, count, destination in (("source_train", train_count, "train"),
                                       ("target_test", test_count, "test")):
        for index in range(count):
            for _ in range(128):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _sample_pair(pair_rng, regime, parameters, index,
                                    adjacency_index, empty_demo_index)
                inp, out = pair["input"], pair["output"]
                input_sig = _signature(inp)
                layout_sig = (inp.shape, tuple(int(x) for x in (inp == inp[0, 0]).ravel()))
                pair_sig = (input_sig, _signature(out))
                if (input_sig in seen_inputs or layout_sig in seen_layouts
                        or pair_sig in _SOURCE_PAIRS):
                    continue
                seen_inputs.add(input_sig)
                seen_layouts.add(layout_sig)
                puzzle[destination].append(pair)
                break
            else:
                raise ValueError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_train_pair(pair: dict) -> tuple[int, int, int, int]:
    inp, out = pair["input"], pair["output"]
    solid, check = [], []
    for row in range(inp.shape[0]):
        for col in range(inp.shape[1]):
            block = out[2*row:2*row+2, 2*col:2*col+2]
            if np.all(block == block[0, 0]):
                solid.append((int(inp[row, col]), int(block[0, 0])))
            else:
                check.append((int(inp[row, col]), tuple(int(x) for x in block.ravel())))
    _require(bool(solid) and bool(check), "every training pair must show both tiles")
    background, output_background = solid[0]
    marker, tile = check[0]
    accent_a, accent_b = tile[0], tile[1]
    _require(background == output_background, "background color is not preserved")
    _require(len({background, marker, accent_a, accent_b}) == 4,
             "color roles are not distinct")
    _assert_pair_rule(inp, out, background, marker, accent_a, accent_b)
    return background, marker, accent_a, accent_b


def _assert_pair_rule(inp: np.ndarray, out: np.ndarray, background: int,
                      marker: int, accent_a: int, accent_b: int) -> None:
    _require(out.shape == (2 * inp.shape[0], 2 * inp.shape[1]), "dimensions are not doubled")
    _require(bool(np.all((inp == background) | (inp == marker))), "unexpected input color")
    expected = _make_output(inp, background, marker, accent_a, accent_b)
    _require(bool(np.array_equal(out, expected)), "incorrect cellwise checkerboard expansion")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                     "grid dimensions outside ARC bounds")
            _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
            _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid color outside ARC palette")
        inp, out = pair["input"], pair["output"]
        _require(out.shape == (2 * inp.shape[0], 2 * inp.shape[1]), "invalid output shape")
        input_sig, pair_sig = _signature(inp), (_signature(inp), _signature(out))
        _require(input_sig not in seen_inputs and pair_sig not in seen_pairs, "duplicate pair")
        _require(pair_sig not in _SOURCE_PAIRS, "exact source pair copied")
        seen_inputs.add(input_sig)
        seen_pairs.add(pair_sig)
    conventions = [_infer_train_pair(pair) for pair in train]
    backgrounds = {c[0] for c in conventions}
    markers = {c[1] for c in conventions}
    accents = {(c[2], c[3]) for c in conventions}
    _require(len(accents) == 1, "checkerboard colors or phase differ within puzzle")
    _require(len(backgrounds) == 1 or len(markers) == 1,
             "both input color roles vary between pairs")
    accent_a, accent_b = next(iter(accents))
    _require(all(c[0] not in (accent_a, accent_b) and c[1] not in (accent_a, accent_b)
                 for c in conventions), "input role overlaps an accent")
    _require(any(np.any(mask[:, :-1] & mask[:, 1:]) or
                 np.any(mask[:-1, :] & mask[1:, :])
                 for mask in (pair["input"] == role[1] for pair, role in zip(train, conventions))),
             "training lacks adjacent markers")
    base_background, base_marker = conventions[0][:2]
    for pair in test:
        inp = pair["input"]
        values = set(int(x) for x in np.unique(inp))
        if len(backgrounds) > 1:
            marker = base_marker
            other = values - {marker}
            _require(len(other) <= 1, "test has multiple background colors")
            background = next(iter(other)) if other else base_background
        elif len(markers) > 1:
            background = base_background
            other = values - {background}
            _require(len(other) <= 1, "test has multiple marker colors")
            marker = next(iter(other)) if other else base_marker
        else:
            background, marker = base_background, base_marker
            _require(values <= {background, marker}, "unexplained test color")
        _require(len({background, marker, accent_a, accent_b}) == 4,
                 "test color roles overlap")
        _assert_pair_rule(inp, pair["output"], background, marker, accent_a, accent_b)


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def _component_count(mask: np.ndarray) -> int:
    seen = set()
    count = 0
    for row, col in np.argwhere(mask):
        start = (int(row), int(col))
        if start in seen:
            continue
        count += 1
        pending = [start]
        seen.add(start)
        while pending:
            r, c = pending.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (r + dr, c + dc)
                if (0 <= neighbor[0] < mask.shape[0]
                        and 0 <= neighbor[1] < mask.shape[1]
                        and mask[neighbor] and neighbor not in seen):
                    seen.add(neighbor)
                    pending.append(neighbor)
    return count


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40,
             "insufficient puzzle diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts do not vary")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts do not vary")
    shapes, layouts, palettes = set(), set(), set()
    empty_train = empty_test = isolated_test = narrow = large = all_one = three = 0
    multi_component = empty_column = all_background = all_marker = 0
    max_size = horizontal_one = vertical_one = near_full = rings = 0
    source_like_shift = mixed_empty_layouts = 0
    varying_background = varying_marker = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles = [_infer_train_pair(pair) for pair in puzzle["train"]]
        puzzle_train_empty = puzzle_test_empty = False
        varying_background += len({r[0] for r in roles}) > 1
        varying_marker += len({r[1] for r in roles}) > 1
        palettes.add(roles[0])
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp = pair["input"]
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(int(v) for v in (inp == inp[0, 0]).ravel())))
                marker_mask = inp != roles[0][0] if len({r[0] for r in roles}) == 1 else inp == roles[0][1]
                empty = bool(np.any(~np.any(marker_mask, axis=1)))
                if section == "train":
                    empty_train += empty
                    puzzle_train_empty |= empty
                else:
                    empty_test += empty
                    puzzle_test_empty |= empty
                    isolated_test += int(np.count_nonzero(marker_mask) == 1)
                    all_one += int(len(np.unique(inp)) == 1)
                    multi_component += int(_component_count(marker_mask) >= 2)
                    empty_column += int(np.any(~np.any(marker_mask, axis=0)))
                    all_background += int(not np.any(marker_mask))
                    all_marker += int(np.all(marker_mask))
                    max_size += int(inp.shape == (15, 15))
                    horizontal_one += int(inp.shape[0] == 1)
                    vertical_one += int(inp.shape[1] == 1)
                    near_full += int(np.mean(marker_mask) >= .8)
                    rings += int(any(
                        not marker_mask[r, c]
                        and np.count_nonzero(marker_mask[r-1:r+2, c-1:c+2]) == 8
                        for r in range(1, inp.shape[0] - 1)
                        for c in range(1, inp.shape[1] - 1)
                    ))
                narrow += int(1 in inp.shape)
                large += int(max(inp.shape) >= 14)
                three += int(inp.shape == (3, 3))
        source_like_shift += int(not puzzle_train_empty and puzzle_test_empty)
        mixed_empty_layouts += int(puzzle_train_empty and puzzle_test_empty)
    _require(len(shapes) >= 35 and len(layouts) >= 150 and len(palettes) >= 30,
             "insufficient shape, layout, or palette diversity")
    _require(empty_train >= 4 and empty_test >= 10 and isolated_test >= 2,
             "empty-row or single-marker coverage is insufficient")
    _require(narrow >= 2 and large >= 10 and all_one >= 2 and three >= 8,
             "size or occupancy boundaries are underrepresented")
    _require(varying_background >= 5 and varying_marker >= 5,
             "within-puzzle color variation is insufficient")
    _require(multi_component >= 10 and empty_column >= 10 and rings >= 2,
             "test geometry lacks separated objects, empty columns, or rings")
    _require(all_background >= 2 and all_marker >= 2 and near_full >= 4,
             "test occupancy boundaries are underrepresented")
    _require(max_size >= 2 and horizontal_one >= 1 and vertical_one >= 1,
             "maximum and one-dimensional sizes are underrepresented")
    _require(source_like_shift >= 3 and mixed_empty_layouts >= 10,
             "train/test layout relationships do not vary enough")
    first = generate_puzzle(712345, 7, 3)
    _require(_puzzle_signature(first) == _puzzle_signature(generate_puzzle(712345, 7, 3)),
             "same seed is not deterministic")
    _require(_puzzle_signature(first) != _puzzle_signature(generate_puzzle(712346, 7, 3)),
             "different seeds produced equal puzzles")
    for offset, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4),
                                                (None, 1), (None, 4),
                                                (2, None), (10, None))):
        puzzle = generate_puzzle(800000 + offset, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            _require(len(puzzle["train"]) == ntrain, "explicit train count changed")
        if ntest is not None:
            _require(len(puzzle["test"]) == ntest, "explicit test count changed")
    for offset, args in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                   (2, 2), (3, 4), (4, 4), (True, 1), (3.0, 1))):
        try:
            generate_puzzle(900000 + offset, *args)
        except ValueError:
            continue
        raise AssertionError(f"illegal count request accepted: {args}")
