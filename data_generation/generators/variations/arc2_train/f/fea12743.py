"""Six aligned masks: recolor the two strict-union operands and their result."""

import hashlib
import numpy as np


_SIZES = (3, 4, 5, 6, 7)
_ARRANGEMENTS = ((3, 2), (2, 3))
_SOURCE_PAIR_DIGESTS = frozenset((
    "8ca703fd0a92d5d6a99df9557b5062d162f135c151b966adbae7cabdcb49340c",
    "f7c993ad784448d6c384369e676ad4dd6d0bbe6a9825a798271af19192c1a6fb",
    "ffe3a64f0b3c373577333b7c30e6eca2d8f32b4724967c2b1044ae346ed769c4",
    "4c18bc36bbfe8a9a66b8135f86ab7466c4ea963d05a7dc7cb037f3feffa7321f",
))


def _counts(seed, num_train, num_test):
    for value, name, lo, hi in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= value <= hi
        ):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _slot(slot, height, width, columns):
    row, col = divmod(slot, columns)
    top = 1 + row * (height + 1)
    left = 1 + col * (width + 1)
    return slice(top, top + height), slice(left, left + width)


def _relations(masks):
    relations = []
    for first in range(6):
        for second in range(first + 1, 6):
            a, b = masks[first], masks[second]
            if not (np.any(a & ~b) and np.any(b & ~a)):
                continue
            union = a | b
            for result in range(6):
                if result not in (first, second) and np.array_equal(union, masks[result]):
                    relations.append((first, second, result))
    return relations


def _chosen_relation(masks):
    relations = _relations(masks)
    if not relations:
        return None
    signatures = set()
    results = set()
    for first, second, result in relations:
        operands = tuple(sorted((masks[first].tobytes(), masks[second].tobytes())))
        signatures.add((operands, masks[result].tobytes()))
        results.add(result)
    if len(signatures) != 1 or len(results) != 1:
        return None
    return min(relations)


def _orientation(pair, columns):
    (a, b) = pair
    ar, ac = divmod(a, columns)
    br, bc = divmod(b, columns)
    if ar == br:
        return "horizontal"
    if ac == bc:
        return "vertical"
    return "diagonal"


def _choose_slots(rng, columns, orientation):
    choices = [
        (a, b) for a in range(6) for b in range(a + 1, 6)
        if _orientation((a, b), columns) == orientation
    ]
    pair = choices[int(rng.integers(len(choices)))]
    remaining = [i for i in range(6) if i not in pair]
    return pair, remaining[int(rng.integers(len(remaining)))]


def _random_path(rng, height, width):
    mask = np.zeros((height, width), dtype=bool)
    row = int(rng.integers(height))
    col = int(rng.integers(width))
    for _ in range(int(rng.integers(height + width, 2 * height * width + 1))):
        mask[row, col] = True
        neighbors = [
            (r, c) for r, c in ((row - 1, col), (row + 1, col),
                                  (row, col - 1), (row, col + 1))
            if 0 <= r < height and 0 <= c < width
        ]
        row, col = neighbors[int(rng.integers(len(neighbors)))]
    return mask


def _base(rng, height, width, style):
    mask = np.zeros((height, width), dtype=bool)
    if style == "frame":
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        # Complete horizontal bars are especially characteristic of the source.
        if rng.random() < 0.35:
            mask[1:-1, int(rng.integers(width))] = True
    elif style == "bars":
        mask[0, :] = True
        mask[-1, :] = True
        for col in rng.choice(width, size=int(rng.integers(1, min(width, 3) + 1)), replace=False):
            mask[:, int(col)] = True
    elif style == "partial":
        mask[0, :] = True
        mask[:, 0] = True
        if rng.random() < 0.6:
            mask[-1, :int(rng.integers(1, width + 1))] = True
        if rng.random() < 0.6:
            mask[:int(rng.integers(1, height + 1)), -1] = True
    elif style == "path":
        mask = _random_path(rng, height, width)
    else:
        mask = rng.random((height, width)) < float(rng.uniform(0.14, 0.48))
    if np.count_nonzero(~mask) < 2:
        removable = np.flatnonzero(mask)
        if style in ("frame", "bars"):
            interior_rows = np.flatnonzero(mask[1:-1, :])
            if len(interior_rows) >= 2:
                removable = np.flatnonzero(mask &
                                            (np.arange(height)[:, None] > 0) &
                                            (np.arange(height)[:, None] < height - 1))
        drop = rng.choice(removable, size=2 - np.count_nonzero(~mask), replace=False)
        mask.flat[drop] = False
    return mask


def _pick_extra(rng, available, amount):
    positions = np.flatnonzero(available)
    if len(positions) < amount:
        return None
    picked = rng.choice(positions, size=amount, replace=False)
    mask = np.zeros(available.shape, dtype=bool)
    mask.flat[picked] = True
    return mask


def _operands(rng, height, width, style):
    for _ in range(80):
        base = _base(rng, height, width, style)
        empty = ~base
        if np.count_nonzero(empty) < 2:
            continue
        # Thin local differences and broader contributions both occur.
        maximum = min(5, max(1, np.count_nonzero(empty) // 3))
        n_a = 1 if rng.random() < 0.35 else int(rng.integers(1, maximum + 1))
        n_b = 1 if rng.random() < 0.35 else int(rng.integers(1, maximum + 1))
        if n_a + n_b > np.count_nonzero(empty):
            continue
        extra_a = _pick_extra(rng, empty, n_a)
        extra_b = _pick_extra(rng, empty & ~extra_a, n_b)
        a, b = base | extra_a, base | extra_b
        if style in ("open", "path", "partial") and rng.random() < 0.35:
            # Move some shared cells to only one operand; overlap is not fixed.
            shared = np.flatnonzero(base)
            if len(shared):
                selected = rng.choice(shared, size=min(len(shared), int(rng.integers(1, 4))), replace=False)
                if rng.random() < 0.5:
                    a.flat[selected] = False
                else:
                    b.flat[selected] = False
        if np.any(a & ~b) and np.any(b & ~a) and np.any(a & b):
            return a, b
    raise RuntimeError("could not make operand masks")


def _candidate_distractor(rng, masks, height, width, style):
    choice = int(rng.integers(5))
    preserve_bars = style in ("frame", "bars") and height >= 4
    if choice <= 2:
        source = masks[int(rng.integers(len(masks)))].copy()
        positions = (np.arange(height * width).reshape(height, width)[1:-1, :].ravel()
                     if preserve_bars else np.arange(height * width))
        flips = int(rng.integers(1, min(4, len(positions)) + 1))
        sites = rng.choice(positions, size=flips, replace=False)
        source.flat[sites] = ~source.flat[sites]
        return source
    if choice == 3 or preserve_bars:
        return _base(rng, height, width, style)
    return _base(rng, height, width, ("frame", "bars", "partial", "path", "open")[int(rng.integers(5))])


def _make_masks(rng, height, width, operands, result, style, duplicate):
    first, second = operands
    # Construct whole candidates in bounded attempts; reject only genuine
    # competing unions, never equivalent duplicate operand choices.
    for _ in range(120):
        a, b = _operands(rng, height, width, style)
        masks = [None] * 6
        masks[first], masks[second], masks[result] = a, b, a | b
        empties = [slot for slot in range(6) if masks[slot] is None]
        if duplicate == "operand":
            eligible = [slot for slot in empties if slot > first]
            if not eligible:
                continue
            duplicate_slot = eligible[int(rng.integers(len(eligible)))]
            masks[duplicate_slot] = a.copy()
        elif duplicate == "distractor":
            duplicate_slot = None
        else:
            duplicate_slot = None
        free = [slot for slot in empties if masks[slot] is None]
        for slot in free:
            for _ in range(90):
                existing = [mask for mask in masks if mask is not None]
                candidate = _candidate_distractor(rng, existing, height, width, style)
                if not np.any(candidate) or any(np.array_equal(candidate, other) for other in existing):
                    continue
                trial = masks.copy()
                trial[slot] = candidate
                if len([mask for mask in trial if mask is not None]) == 6 and _chosen_relation(trial) != (first, second, result):
                    continue
                masks[slot] = candidate
                break
            else:
                break
        if any(mask is None for mask in masks):
            continue
        if duplicate == "distractor":
            distractors = [slot for slot in range(6) if slot not in (first, second, result)]
            source, target = distractors[:2]
            masks[target] = masks[source].copy()
        if _chosen_relation(masks) == (first, second, result):
            return masks
    raise RuntimeError("could not construct an unambiguous mask set")


def _compose(masks, height, width, rows, columns, background, foreground):
    grid = np.full((rows * height + rows + 1, columns * width + columns + 1),
                   background, dtype=np.int8)
    for slot, mask in enumerate(masks):
        block = grid[_slot(slot, height, width, columns)]
        block[mask] = foreground
    return grid


def _solve(input_grid, height, width, rows, columns, foreground, operand_color, result_color):
    masks = [input_grid[_slot(i, height, width, columns)] == foreground for i in range(6)]
    relation = _chosen_relation(masks)
    assert relation is not None
    first, second, result = relation
    output = input_grid.copy()
    for slot, color in ((first, operand_color), (second, operand_color),
                        (result, result_color)):
        block = output[_slot(slot, height, width, columns)]
        block[masks[slot]] = color
    return output, relation


def _dimensions(rng, arrangement):
    if rng.random() < 0.22:
        return 4, 4
    # A 13x13 canvas can be divided as either 3x2 of 3x5 motifs or
    # 2x3 of 5x3 motifs. Keep the visible scaffold unambiguous.
    choices = [(h, w) for h in _SIZES for w in _SIZES
               if not (arrangement == (3, 2) and (h, w) == (3, 5))
               and not (arrangement == (2, 3) and (h, w) == (5, 3))]
    return choices[int(rng.integers(len(choices)))]


def _pair(rng, parameters, orientation, style, duplicate, palette, dimensions):
    rows, columns = parameters["arrangement"]
    height, width = dimensions
    operands, result = _choose_slots(rng, columns, orientation)
    masks = _make_masks(rng, height, width, operands, result, style, duplicate)
    background, foreground = palette
    input_grid = _compose(masks, height, width, rows, columns, background, foreground)
    output_grid, relation = _solve(input_grid, height, width, rows, columns,
                                   foreground, parameters["operand_color"],
                                   parameters["result_color"])
    assert relation == (*operands, result)
    return {"input": input_grid, "output": output_grid}


def _key(grid):
    return grid.shape, grid.tobytes()


def _pair_digest(pair):
    digest = hashlib.sha256()
    for role in ("input", "output"):
        grid = np.asarray(pair[role], dtype=np.uint8)
        digest.update(bytes(grid.shape))
        digest.update(grid.tobytes())
    return digest.hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    arrangement = _ARRANGEMENTS[int(rng.random() < 0.24)]
    parameters = {"arrangement": arrangement,
                  "operand_color": colors[2], "result_color": colors[3]}
    available = [c for c in range(10) if c not in colors[2:]]
    palettes = [(colors[0], colors[1])]
    if rng.random() < 0.65:
        other = tuple(int(x) for x in rng.choice(available, size=2, replace=False))
        if other != palettes[0]:
            palettes.append(other)
    # Every test palette and highlight convention is shown by training.
    train_palettes = [palettes[i % len(palettes)] for i in range(train_count)]
    rng.shuffle(train_palettes)
    train_dimensions = [_dimensions(rng, arrangement) for _ in range(train_count)]
    if train_count >= 2 and len(set(train_dimensions)) == 1:
        train_dimensions[-1] = (5, 4) if train_dimensions[0] != (5, 4) else (4, 5)
    orientations = ["horizontal", "vertical", "diagonal"]
    offset = int(rng.integers(3))
    holdout_orientation = bool(rng.random() < 0.35) or train_count == 2
    demonstrated = 2 if holdout_orientation else 3
    train_orientations = [orientations[(offset + i % demonstrated) % 3]
                          for i in range(train_count)]
    novel_orientation = orientations[(offset + 2) % 3]
    rng.shuffle(train_orientations)
    styles = ("frame", "bars", "partial", "path", "open")
    train_styles = [styles[(int(rng.integers(5)) + i) % 5] for i in range(train_count)]
    rng.shuffle(train_styles)
    tie_enabled = bool(rng.random() < 0.42)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(80):
                if split == "train":
                    orientation = train_orientations[index]
                    style = train_styles[index]
                    palette = train_palettes[index]
                    dimensions = train_dimensions[index]
                    duplicate = ("operand" if tie_enabled and index == 0 else
                                 "distractor" if index == 1 and rng.random() < 0.7 else
                                 ("operand", "distractor", "none", "none")[int(rng.integers(4))]
                                 if tie_enabled else
                                 ("distractor", "none", "none")[int(rng.integers(3))])
                else:
                    orientation = (novel_orientation if holdout_orientation and index == 0
                                   else orientations[int(rng.integers(3))])
                    style = styles[int(rng.integers(5))]
                    palette = train_palettes[int(rng.integers(train_count))]
                    dimensions = _dimensions(rng, arrangement)
                    duplicate = ("operand", "distractor", "none", "none")[int(rng.integers(4))] if tie_enabled else ("distractor", "none", "none")[int(rng.integers(3))]
                try:
                    pair = _pair(rng, parameters, orientation, style, duplicate, palette, dimensions)
                except RuntimeError:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background = int(input_grid[0, 0])
    colors = set(int(x) for x in np.unique(input_grid))
    assert len(colors) == 2 and background in colors
    foreground = next(iter(colors - {background}))
    output_colors = set(int(x) for x in np.unique(output_grid))
    highlights = output_colors - colors
    assert len(highlights) == 2 and len(output_colors) == 4
    candidates = []
    for rows, columns in _ARRANGEMENTS:
        h_raw = input_grid.shape[0] - rows - 1
        w_raw = input_grid.shape[1] - columns - 1
        if h_raw % rows or w_raw % columns:
            continue
        height, width = h_raw // rows, w_raw // columns
        if height not in _SIZES or width not in _SIZES:
            continue
        gutter = np.ones(input_grid.shape, dtype=bool)
        masks = []
        for slot in range(6):
            slices = _slot(slot, height, width, columns)
            gutter[slices] = False
            masks.append(input_grid[slices] == foreground)
        if not np.all(input_grid[gutter] == background) or not all(np.any(mask) for mask in masks):
            continue
        relation = _chosen_relation(masks)
        if relation is None:
            continue
        for operand_color, result_color in (
            tuple(highlights), tuple(reversed(tuple(highlights)))
        ):
            expected, found = _solve(input_grid, height, width, rows, columns,
                                     foreground, operand_color, result_color)
            if np.array_equal(expected, output_grid):
                candidates.append((rows, columns, height, width, background,
                                   foreground, operand_color, result_color,
                                   relation, len(_relations(masks)),
                                   len(set(mask.tobytes() for mask in masks))))
    assert len(candidates) == 1
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyzed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(np.min(grid)) <= int(np.max(grid)) <= 9
            assert not np.array_equal(pair["input"], pair["output"])
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _pair_digest(pair) not in _SOURCE_PAIR_DIGESTS
            analyzed[split].append(_analyze(pair["input"], pair["output"]))
    train = analyzed["train"]
    highlights = {(a[6], a[7]) for a in train}
    assert len(highlights) == 1
    arrangement = {(a[0], a[1]) for a in train}
    assert len(arrangement) == 1
    train_palettes = {(a[4], a[5]) for a in train}
    assert len({ _orientation(a[8][:2], a[1]) for a in train }) >= 2
    if any(a[9] > 1 for a in analyzed["test"]):
        assert any(a[9] > 1 for a in train)
    for analysis in analyzed["test"]:
        assert (analysis[6], analysis[7]) in highlights
        assert (analysis[0], analysis[1]) in arrangement
        assert (analysis[4], analysis[5]) in train_palettes


def _equal(first, second):
    return all(
        len(first[split]) == len(second[split]) and
        all(np.array_equal(a[role], b[role])
            for a, b in zip(first[split], second[split]) for role in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(p["input"]) for split in ("train", "test")
                      for p in puzzle[split]) for puzzle in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    train_analyses = [_analyze(pair["input"], pair["output"])
                      for puzzle in puzzles for pair in puzzle["train"]]
    test_analyses = [_analyze(pair["input"], pair["output"])
                     for puzzle in puzzles for pair in puzzle["test"]]
    analyses = train_analyses + test_analyses
    assert {a[:2] for a in analyses} == set(_ARRANGEMENTS)
    assert (4, 4) in {(a[2], a[3]) for a in analyses}
    assert len({(a[2], a[3]) for a in analyses}) >= 15
    for subset in (train_analyses, test_analyses):
        assert {_orientation(a[8][:2], a[1]) for a in subset} == {
            "horizontal", "vertical", "diagonal"}
        assert {a[8][2] for a in subset} == set(range(6))
        assert any(a[9] > 1 for a in subset)
        assert any(a[10] < 6 for a in subset)
    assert any(
        _orientation(test[8][:2], test[1]) not in
        {_orientation(train[8][:2], train[1]) for train in
         [_analyze(pair["input"], pair["output"]) for pair in puzzle["train"]]}
        for puzzle in puzzles for test in
        [_analyze(pair["input"], pair["output"]) for pair in puzzle["test"]]
    )
    assert any(len({(a[4], a[5]) for a in
                    [_analyze(p["input"], p["output"]) for p in puzzle["train"]]}) > 1
               for puzzle in puzzles)
    assert all(len({(_analyze(p["input"], p["output"])[2:4])
                    for p in puzzle["train"]}) >= 2 for puzzle in puzzles)
    full_bars = 0
    small_contributions = 0
    broad_contributions = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                rows, columns, height, width, background, foreground, operand_color, result_color, relation, _, _ = _analyze(
                    pair["input"], pair["output"])
                masks = [pair["input"][_slot(slot, height, width, columns)] == foreground
                         for slot in range(6)]
                first, second, result = relation
                assert np.array_equal(masks[first] | masks[second], masks[result])
                first_only = int(np.count_nonzero(masks[first] & ~masks[second]))
                second_only = int(np.count_nonzero(masks[second] & ~masks[first]))
                small_contributions += first_only == second_only == 1
                broad_contributions += max(first_only, second_only) >= 4
                full_bars += all(np.all(mask[0]) and np.all(mask[-1]) for mask in masks)
                for slot, mask in enumerate(masks):
                    input_block = pair["input"][_slot(slot, height, width, columns)]
                    output_block = pair["output"][_slot(slot, height, width, columns)]
                    color = (result_color if slot == result else
                             operand_color if slot in (first, second) else foreground)
                    assert np.array_equal(output_block, np.where(mask, color, background))
                    assert np.all(input_block[~mask] == background)
    assert full_bars >= 15
    assert small_contributions >= 20
    assert broad_contributions >= 10
    for seed, train, test in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4)):
        puzzle = generate_puzzle(seed, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for seed, train, test in ((9101, None, 1), (9102, None, 4),
                              (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(31, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    assert _equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
