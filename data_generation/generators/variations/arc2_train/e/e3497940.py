"""Generate transparent vertical-fold ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("000050000", "000250000", "000252600", "000250000",
     "000252220", "006656000", "000250000", "022052000",
     "000250000", "000050000"),
    ("000050000", "000050000", "000450000", "000454400",
     "003350000", "000350000", "000353330", "000350000",
     "000050000", "000050000"),
    ("000050000", "000057000", "000850000", "000850000",
     "078850000", "000058800", "000850000", "000850000",
     "000058700", "000050000"),
    ("000050000", "000150000", "000151000", "011151116",
     "000656600", "000051110", "000150000", "000151600",
     "000056000", "000050000"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _shape(rng, kind, height, width, allowed_first):
    """Return one connected piece in folded coordinates."""
    low = allowed_first
    row = int(rng.integers(height))
    col = int(rng.integers(low, width))
    if kind == "arm":
        # Some arms meet the divider; others start elsewhere.
        if rng.random() < 0.65:
            end = width
            start = int(rng.integers(low, end))
        else:
            start = col
            end = int(rng.integers(start + 1, width + 1))
        return {(row, c) for c in range(start, end)}
    if kind == "bar":
        length = int(rng.integers(1, width - col + 1))
        return {(row, c) for c in range(col, col + length)}
    if kind == "block":
        rh = int(rng.integers(1, min(4, height - row) + 1))
        cw = int(rng.integers(1, min(4, width - col) + 1))
        return {(r, c) for r in range(row, row + rh)
                for c in range(col, col + cw)}
    if kind == "stroke":
        length = int(rng.integers(2, min(7, height - row) + 1)) if row < height - 1 else 1
        return {(r, col) for r in range(row, row + length)}
    if kind == "step":
        length = int(rng.integers(2, min(6, height - row) + 1)) if row < height - 1 else 1
        direction = -1 if rng.random() < 0.5 else 1
        cells = set()
        for offset in range(length):
            c = max(low, min(width - 1, col + direction * (offset // 2)))
            cells.add((row + offset, c))
            if c + 1 < width and rng.random() < 0.55:
                cells.add((row + offset, c + 1))
        return cells
    # An L joins a vertical stroke and horizontal run.
    rh = int(rng.integers(1, min(5, height - row) + 1))
    cw = int(rng.integers(1, width - col + 1))
    return {(r, col) for r in range(row, row + rh)} | {
        (row + rh - 1, c) for c in range(col, col + cw)}


def _piece_palette(rng, background, separator, count):
    available = [c for c in range(10) if c not in (background, separator)]
    return [int(c) for c in rng.choice(available, size=count, replace=False)]


def _pair(rng, regime, source_shift, demonstration, previous):
    height = int(rng.integers(7, 24))
    width = int(rng.integers(3, 12))
    if demonstration:
        height = max(height, 9)
    if previous is not None and (height, width) == previous[0]:
        height = 7 + (height - 6) % 17
    if rng.random() < 0.48:
        background = 0
    else:
        background = int(rng.integers(1, 10))
    if previous is not None and background == previous[1]:
        background = (background + 1) % 10
    separator = int(rng.choice([c for c in range(10) if c != background]))
    if previous is not None and separator == previous[2]:
        separator = next(c for c in range(10) if c not in (background, previous[2]))

    color_count = int(rng.choice([1, 2, 3, 4], p=[0.18, 0.55, 0.22, 0.05]))
    colors = _piece_palette(rng, background, separator, color_count)
    left = np.full((height, width), background, dtype=np.int8)
    right = np.full((height, width), background, dtype=np.int8)
    first_col = int(source_shift and regime == "source_train")
    overlap_mode = int(rng.choice(4, p=[0.23, 0.30, 0.30, 0.17]))
    if demonstration:
        overlap_mode = 2
    motif = int(rng.integers(4))
    area = height * width
    pieces = int(rng.integers(2, 5 if area < 65 else 8))
    if regime == "target_test" and source_shift:
        pieces += 2
    if motif == 0:
        pieces = max(3, pieces)
    # Each primitive is placed on a flank as a coherent piece. The output is
    # derived only after both flank canvases have been composed.
    placed = []
    initial_side = int(rng.integers(2))
    for index in range(pieces):
        if motif == 0 and index < pieces - 1:
            kind = "arm"
        else:
            kind = ("arm", "bar", "block", "stroke", "step", "ell")[int(rng.integers(6))]
        cells = _shape(rng, kind, height, width, first_col)
        color = colors[index] if index < len(colors) else int(rng.choice(colors))
        if motif == 2 and index >= len(colors):
            color = colors[0]
        if motif == 3 and index >= len(colors):
            color = colors[(index // 2) % len(colors)]
        side = initial_side ^ (index % 2) if index < 2 else int(rng.integers(2))
        own, other = (left, right) if side == 0 else (right, left)
        accepted = set()
        for r, c in cells:
            if own[r, c] not in (background, color):
                continue
            if other[r, c] != background:
                if overlap_mode == 0 or other[r, c] != color:
                    continue
            own[r, c] = color
            accepted.add((r, c))
        if accepted:
            placed.append((side, accepted))

    # Occasionally let a coherent piece touch a canvas boundary.
    if rng.random() < 0.36:
        r = 0 if rng.random() < 0.5 else height - 1
        c = int(rng.integers(first_col, width))
        side = int(rng.integers(2))
        own, other = (left, right) if side == 0 else (right, left)
        color = int(rng.choice(colors))
        if other[r, c] in (background, color):
            own[r, c] = color
            placed.append((side, {(r, c)}))

    if regime == "target_test" and source_shift:
        # The source's outer-column shift is one possible challenge.
        r = int(rng.integers(height))
        c = 0
        color = int(rng.choice(colors))
        left[r, c], right[r, c] = background, color
        placed.append((1, {(r, c)}))

    if overlap_mode:
        copy_count = {1: 1, 2: max(1, len(placed) // 2),
                      3: len(placed)}[overlap_mode]
        order = rng.permutation(len(placed))
        for index in order[:copy_count]:
            side, cells = placed[int(index)]
            own, other = (left, right) if side == 0 else (right, left)
            for r, c in cells:
                if own[r, c] != background and other[r, c] == background:
                    other[r, c] = own[r, c]

    if demonstration:
        # Training evidence distinguishes folding from cropping and shows
        # same-color collapse. These anchors occur in only one training pair.
        vacancies = [(r, c) for r in range(height) for c in range(first_col, width)
                     if left[r, c] == background and right[r, c] == background]
        rng.shuffle(vacancies)
        if vacancies:
            r, c = next(((r, c) for r, c in vacancies if c < width - 1), vacancies[0])
            right[r, c] = colors[0]
        vacancies = [(r, c) for r in range(height) for c in range(first_col, width)
                     if left[r, c] == background and right[r, c] == background]
        if vacancies:
            r, c = vacancies[int(rng.integers(len(vacancies)))]
            left[r, c] = colors[0]
        candidates = np.argwhere((left != background) & (right == background))
        if len(candidates) == 0:
            candidates = np.argwhere((right != background) & (left == background))
            if len(candidates):
                r, c = map(int, candidates[0])
                left[r, c] = right[r, c]
        else:
            r, c = map(int, candidates[0])
            right[r, c] = left[r, c]

    output = np.where(left != background, left, right)
    occupied = int(np.count_nonzero(output != background))
    if occupied < 3 or occupied >= area * 0.48:
        return None
    if not np.any(left != background) or not np.any(right != background):
        return None
    if demonstration:
        left_only = (left != background) & (right == background)
        right_only = (left == background) & (right != background)
        both = (left != background) & (right != background)
        if not (np.any(left_only) and np.any(right_only[:, :-1]) and np.any(both)):
            return None
    if source_shift and regime == "source_train" and np.any(output[:, 0] != background):
        return None
    if source_shift and regime == "target_test" and not np.any(output[:, 0] != background):
        return None
    input_grid = np.concatenate(
        (left, np.full((height, 1), separator, dtype=np.int8), right[:, ::-1]),
        axis=1,
    )
    for column in range(input_grid.shape[1]):
        if column == width:
            continue
        value = int(input_grid[0, column])
        if (value != background and np.all(input_grid[:, column] == value)
                and np.count_nonzero(input_grid == value) == height):
            return None
    return {"input": input_grid, "output": output}


def _fallback_pair(regime, source_shift, demonstration, previous, ordinal):
    """A sparse, visibly folded construction if random proposals are exhausted."""
    height = 7 + ordinal % 17
    width = 3 + (ordinal // 17) % 9
    if previous is not None and (height, width) == previous[0]:
        height = 7 + (height - 6) % 17
    background = ordinal % 10
    if previous is not None and background == previous[1]:
        background = (background + 1) % 10
    forbidden = {background}
    if previous is not None:
        forbidden.add(previous[2])
    separator = next(color for color in range(10) if color not in forbidden)
    payload = next(color for color in range(10)
                   if color not in (background, separator))
    left = np.full((height, width), background, dtype=np.int8)
    right = left.copy()
    left[1, width - 1] = payload
    right[2, width - 2] = payload
    left[5, width - 1] = payload
    if demonstration:
        left[3, width - 1] = right[3, width - 1] = payload
    if source_shift and regime == "target_test":
        right[4, 0] = payload
    output = np.where(left != background, left, right)
    inp = np.concatenate(
        (left, np.full((height, 1), separator, dtype=np.int8), right[:, ::-1]),
        axis=1,
    )
    return {"input": inp, "output": output}


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(pair):
    return tuple("".join(str(int(v)) for v in row) for row in pair["input"]) in _SOURCE_INPUTS


def _background(input_grid, width):
    flanks = np.concatenate((input_grid[:, :width].ravel(),
                             input_grid[:, width + 1:].ravel()))
    colors, counts = np.unique(flanks, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "expected train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "pair groups must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen = set()
    training_evidence = [False, False, False]
    for group in ("train", "test"):
        for pair in puzzle[group]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                         and np.issubdtype(grid.dtype, np.integer), "invalid array")
                _require(all(1 <= d <= 30 for d in grid.shape), "invalid size")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            height, width = out.shape
            _require(inp.shape == (height, 2 * width + 1), "fold dimensions differ")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(not _source_copy(pair), "source pair copied")
            separator = int(inp[0, width])
            _require(bool(np.all(inp[:, width] == separator)), "incomplete separator")
            background = _background(inp, width)
            _require(background != separator, "separator equals background")
            left, right = inp[:, :width], inp[:, width + 1:][:, ::-1]
            _require(not np.any(left == separator) and not np.any(right == separator),
                     "separator leaked into payload")
            _require(not np.any(out == separator), "separator leaked into output")
            _require(np.count_nonzero((left != background) | (right != background))
                     < height * width * 0.5, "background is not identifiable")
            _require(not np.any((left != background) & (right != background)
                                & (left != right)), "opposing colors conflict")
            _require(np.array_equal(out, np.where(left != background, left, right)),
                     "output is not the transparent fold")
            _require(np.any(out != background), "empty payload")
            unique_dividers = []
            for col in range(inp.shape[1]):
                value = int(inp[0, col])
                if np.all(inp[:, col] == value) and value != background:
                    if np.count_nonzero(inp == value) == height:
                        unique_dividers.append(col)
            _require(unique_dividers == [width], "ambiguous divider")
            if group == "train":
                training_evidence[0] |= bool(np.any((left != background)
                                                    & (right == background)))
                training_evidence[1] |= bool(np.any((left == background)
                                                    & (right != background)
                                                    & (np.indices(right.shape)[1] < width - 1)))
                training_evidence[2] |= bool(np.any((left != background)
                                                    & (right != background)))
    _require(all(training_evidence), "training omits fold evidence")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    # The only puzzle-level convention is the vertical, left-oriented fold.
    # The source-like reach shift is a puzzle-level sampling regime, not a rule.
    source_shift = bool(rng.random() < 0.27)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for group, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if group == "train" else "target_test"
        for index in range(count):
            for attempt in range(100):
                pair = _pair(rng, regime, source_shift,
                             group == "train" and index == 0, previous)
                if pair is not None and _grid_key(pair["input"]) not in seen and not _source_copy(pair):
                    break
            else:
                for offset in range(100):
                    pair = _fallback_pair(regime, source_shift,
                                          group == "train" and index == 0,
                                          previous, len(seen) + offset)
                    if _grid_key(pair["input"]) not in seen and not _source_copy(pair):
                        break
                else:
                    raise RuntimeError("could not construct a valid unique fold pair")
            seen.add(_grid_key(pair["input"]))
            puzzle[group].append(pair)
            width = pair["output"].shape[1]
            previous = ((pair["output"].shape[0], width),
                        _background(pair["input"], width), int(pair["input"][0, width]))
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(first, second):
    return all(
        len(first[group]) == len(second[group])
        and all(np.array_equal(a[key], b[key])
                for a, b in zip(first[group], second[group])
                for key in ("input", "output"))
        for group in ("train", "test")
    )


def _solve_from_examples(train, test_input):
    # Infer the divider and transparent background from grids only.
    for pair in train:
        inp, out = pair["input"], pair["output"]
        if inp.shape[1] != 2 * out.shape[1] + 1:
            raise AssertionError("training does not show a vertical fold")
    width = test_input.shape[1] // 2
    background = _background(test_input, width)
    left = test_input[:, :width]
    right = test_input[:, width + 1:][:, ::-1]
    return np.where(left != background, left, right)


def self_test() -> None:
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_equal(puzzle, generate_puzzle(seed)), "not deterministic")
        for pair in puzzle["train"] + puzzle["test"]:
            prediction = _solve_from_examples(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]], pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]),
                     "input-derived prediction differs")
        puzzles.append(puzzle)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    _require(len({(_grid_key(p["train"][0]["input"]),
                   _grid_key(p["test"][0]["input"])) for p in puzzles}) >= 40,
             "whole-puzzle structures lack variety")
    features = []
    for puzzle in puzzles:
        for group in ("train", "test"):
            for pair in puzzle[group]:
                inp, out = pair["input"], pair["output"]
                width = out.shape[1]
                bg = _background(inp, width)
                left, right = inp[:, :width], inp[:, width + 1:][:, ::-1]
                features.append((out.shape, bg, int(inp[0, width]),
                                 len(set(int(x) for x in np.unique(out)) - {bg}),
                                 int(np.count_nonzero((left != bg) & (right != bg))),
                                 bool(np.any(out[0] != bg) or np.any(out[-1] != bg)),
                                 bool(np.any((out[:, :-1] != bg) & (out[:, 1:] == bg)))))
    _require(len({f[0] for f in features}) >= 25, "dimension diversity missing")
    _require(len({f[1] for f in features}) >= 5, "background diversity missing")
    _require(len({f[2] for f in features}) >= 5, "separator diversity missing")
    _require({1, 2, 3}.issubset({f[3] for f in features}), "palette diversity missing")
    _require(any(f[4] == 0 for f in features) and any(f[4] >= 3 for f in features),
             "overlap regimes missing")
    _require(any(f[5] for f in features), "boundary contact missing")
    _require(any(f[6] for f in features), "off-divider geometry missing")
    varied_background = varied_separator = varied_size = 0
    no_shared_payload = no_overlap = several_overlap = 0
    training_outer = test_without_outer = non_suffix = multi_long = short_only = 0
    for puzzle in puzzles:
        local = []
        palettes = []
        for group in ("train", "test"):
            for pair in puzzle[group]:
                inp, out = pair["input"], pair["output"]
                height, width = out.shape
                bg = _background(inp, width)
                separator = int(inp[0, width])
                left = inp[:, :width]
                right = inp[:, width + 1:][:, ::-1]
                both = int(np.count_nonzero((left != bg) & (right != bg)))
                no_overlap += both == 0
                several_overlap += both >= 3
                palette = set(int(c) for c in np.unique(out)) - {bg}
                palettes.append(palette)
                local.append((bg, separator, out.shape))
                occupied = out != bg
                training_outer += group == "train" and bool(np.any(occupied[:, 0]))
                test_without_outer += group == "test" and not bool(np.any(occupied[:, 0]))
                long_runs = 0
                for row in occupied:
                    columns = np.flatnonzero(row)
                    if len(columns) and (columns[-1] != width - 1
                                         or len(columns) != width - columns[0]):
                        non_suffix += 1
                    run = 0
                    for cell in row:
                        run = run + 1 if cell else 0
                        if run == 3:
                            long_runs += 1
                multi_long += long_runs >= 2
                short_only += long_runs == 0
        varied_background += len({x[0] for x in local}) > 1
        varied_separator += len({x[1] for x in local}) > 1
        varied_size += len({x[2] for x in local}) > 1
        no_shared_payload += not set.intersection(*palettes)
    _require(min(varied_background, varied_separator, varied_size) >= 35,
             "pair-level conventions or dimensions did not vary within puzzles")
    _require(no_shared_payload >= 25, "fixed shared payload color remains")
    _require(no_overlap >= 15 and several_overlap >= 30,
             "overlap distribution collapsed")
    _require(training_outer >= 15 and test_without_outer >= 5,
             "train/test reach schedule is too rigid")
    _require(non_suffix >= 30 and multi_long >= 15 and short_only >= 15,
             "folded shape distribution is too narrow")
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        puzzle = generate_puzzle(787, *counts)
        validate_generated_puzzle(puzzle)
        for actual, expected in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(expected is None or actual == expected, "requested count ignored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(787, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
    bad = generate_puzzle(808, 3, 1)
    pair = bad["train"][0]
    width = pair["output"].shape[1]
    bg = _background(pair["input"], width)
    r, c = map(int, np.argwhere(pair["output"] != bg)[0])
    pair["output"][r, c] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label was accepted")
