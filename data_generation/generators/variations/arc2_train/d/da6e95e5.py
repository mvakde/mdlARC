"""Generate panels whose unique self-colored frame selects an exact inset."""

import hashlib

import numpy as np


_SOURCE_INPUT_DIGESTS = frozenset({
    "7fdf040f3f0734a062c2f281e9c82d7d7e067a83ed566ee7fd88d23e271bf2dc",
    "d812027e17d0c61b5cf9120a731750f015fbf4a7448c95e7ebdcd19c44499031",
    "bdb3a04962e0e1a681e2dfe2474c4e26ea175957e995e5851f067f9b58e1f023",
    "297773b5f26ccd945e685376de2322d672bde1cca800160f6491f0a79298924d",
    "7c158598f6ca2338f68d291e7fd585e64cf08995d407a929776d1cc661734c4d",
})

_SHAPES = (
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 2), (3, 3), (3, 4), (3, 5),
    (4, 2), (4, 3), (4, 4), (4, 5),
    (5, 2), (5, 3), (5, 4),
)


def _count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _counts(seed, rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
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


def _frame_mask(height, width, arm):
    mask = np.zeros((height, width), dtype=bool)
    for distance in range(arm):
        for row, col, row_step, col_step in (
            (0, 0, 1, 1), (0, width - 1, 1, -1),
            (height - 1, 0, -1, 1),
            (height - 1, width - 1, -1, -1),
        ):
            mask[row + row_step * distance, col] = True
            mask[row, col + col_step * distance] = True
    return mask


def _spaces(rng, minimums, extra):
    if extra == 0:
        return np.asarray(minimums, dtype=int)
    return np.asarray(minimums, dtype=int) + rng.multinomial(
        extra, np.full(len(minimums), 1 / len(minimums))
    )


def _layout(rng, count, outer_height, outer_width):
    options = []
    for rows in range(1, count + 1):
        for columns in range(1, count + 1):
            if not count <= rows * columns <= count + 2:
                continue
            for gap in (2, 3, 4):
                min_height = 2 + rows * outer_height + (rows - 1) * gap
                min_width = 2 + columns * outer_width + (columns - 1) * gap
                if min_height <= 30 and min_width <= 30:
                    options.append((rows, columns, gap, min_height, min_width))
    if not options:
        raise ValueError("panels do not fit on a 30 by 30 canvas")

    # Favor multiple columns so separate columns can have independently
    # staggered vertical positions. One-row arrangements remain possible.
    weights = np.asarray([3 if option[1] >= 2 and option[0] >= 2 else 1
                          for option in options], dtype=float)
    choice = int(rng.choice(len(options), p=weights / weights.sum()))
    rows, columns, gap, min_height, min_width = options[choice]
    extent_mode = int(rng.integers(0, 5))
    if extent_mode == 0:
        height, width = 30, 30
    elif extent_mode == 1:
        height = min(30, min_height + int(rng.integers(0, 4)))
        width = min(30, min_width + int(rng.integers(0, 4)))
    else:
        height = int(rng.integers(min_height, 31))
        width = int(rng.integers(min_width, 31))

    horizontal = _spaces(rng, [1] + [gap] * (columns - 1) + [1],
                         width - min_width)
    column_lefts = []
    cursor = int(horizontal[0])
    for column in range(columns):
        column_lefts.append(cursor)
        if column + 1 < columns:
            cursor += outer_width + int(horizontal[column + 1])

    staggered = columns >= 2 and bool(rng.integers(0, 4))
    best = None
    best_distinct = -1
    # Separate vertical allocation per column gives different top coordinates
    # without ever bringing two panels in one column together.
    for _ in range(16 if staggered else 1):
        column_tops = []
        shared = _spaces(rng, [1] + [gap] * (rows - 1) + [1],
                         height - min_height)
        for column in range(columns):
            vertical = (_spaces(rng, [1] + [gap] * (rows - 1) + [1],
                                height - min_height) if staggered else shared)
            tops = []
            cursor = int(vertical[0])
            for row in range(rows):
                tops.append(cursor)
                if row + 1 < rows:
                    cursor += outer_height + int(vertical[row + 1])
            column_tops.append(tops)
        slots = [(column_tops[column][row], column_lefts[column])
                 for row in range(rows) for column in range(columns)]
        chosen = rng.choice(len(slots), size=count, replace=False)
        placements = sorted(slots[int(index)] for index in chosen)
        distinct = len({top for top, _ in placements})
        if distinct > best_distinct:
            best, best_distinct = placements, distinct
        if distinct == count:
            break
    return height, width, best


def _glyph_mask(rng, height, width, allow_full=True):
    rows, columns = np.indices((height, width))
    mode = int(rng.integers(0, 9 if allow_full else 8))
    if mode == 0:
        mask = rng.random((height, width)) < rng.uniform(.2, .8)
    elif mode == 1:
        mask = (rows + columns + int(rng.integers(0, 2))) % 2 == 0
    elif mode == 2:
        mask = ((rows == 0) | (rows == height - 1) |
                (columns == 0) | (columns == width - 1))
    elif mode == 3:
        mask = ((rows == int(rng.integers(0, height))) |
                (columns == int(rng.integers(0, width))))
    elif mode == 4:
        mask = np.zeros((height, width), dtype=bool)
        mask.flat[rng.choice(height * width,
                             size=int(rng.integers(1, height * width)),
                             replace=False)] = True
    elif mode == 5:
        mask = ((1 + int(rng.integers(0, 3))) * rows +
                (1 + int(rng.integers(0, 3))) * columns +
                int(rng.integers(0, 4))) % int(rng.integers(2, 5)) == 0
    elif mode == 6:
        mask = np.zeros((height, width), dtype=bool)
        r0 = int(rng.integers(0, height))
        c0 = int(rng.integers(0, width))
        mask[r0:, c0:] = True
    elif mode == 7:
        mask = np.ones((height, width), dtype=bool)
        holes = int(rng.integers(1, max(2, height * width // 2)))
        mask.flat[rng.choice(height * width, size=holes, replace=False)] = False
    else:
        mask = np.ones((height, width), dtype=bool)
    if not np.any(mask) or (not allow_full and np.all(mask)):
        mask = np.zeros((height, width), dtype=bool)
        mask.flat[int(rng.integers(0, height * width))] = True
    if rng.integers(0, 2):
        mask = np.flipud(mask)
    if rng.integers(0, 2):
        mask = np.fliplr(mask)
    if height == width and rng.integers(0, 2):
        mask = mask.T
    return mask


def _colors(rng, background, count, target, mode, split):
    available = np.asarray([color for color in range(10) if color != background],
                           dtype=np.int8)
    distractors = [index for index in range(count) if index != target]
    if mode == "same_frame":
        size = int(rng.integers(2, min(count, 6) + 1))
    else:
        high = min(8 if split == "test" else 6, 2 * count - 1, 9)
        low = 3 if split == "test" else 2
        size = int(rng.integers(low, high + 1))
    palette = rng.choice(available, size=size, replace=False)
    chosen = int(palette[0])
    frames = np.full(count, -1, dtype=np.int8)
    glyphs = np.full(count, -1, dtype=np.int8)
    frames[target] = glyphs[target] = chosen
    if mode == "same_frame":
        frames[:] = chosen
        glyphs[distractors] = rng.choice(palette[1:], size=len(distractors))
        for index, color in enumerate(palette[1:]):
            glyphs[distractors[index]] = color
    else:
        roles = [(index, role) for index in distractors for role in (0, 1)]
        rng.shuffle(roles)
        for (index, role), color in zip(roles, palette[1:]):
            (frames if role == 0 else glyphs)[index] = color
        if mode == "reuse_target":
            # Put the chosen color on a distractor, then repair its other role.
            index = distractors[int(rng.integers(0, len(distractors)))]
            role = int(rng.integers(0, 2))
            (frames if role == 0 else glyphs)[index] = chosen
        if mode == "shared_glyph":
            first, second = rng.choice(distractors, size=2, replace=False)
            common = int(palette[int(rng.integers(0, len(palette)))])
            glyphs[first] = glyphs[second] = common
        for index in distractors:
            if frames[index] < 0:
                choices = palette[palette != glyphs[index]]
                frames[index] = rng.choice(choices)
            if glyphs[index] < 0 or glyphs[index] == frames[index]:
                choices = palette[palette != frames[index]]
                glyphs[index] = rng.choice(choices)
        if mode == "shared_glyph":
            # The previous repairs might have changed one shared role.
            common = int(glyphs[first])
            glyphs[second] = common
            if frames[second] == common:
                frames[second] = rng.choice(palette[palette != common])
            if frames[first] == common:
                frames[first] = rng.choice(palette[palette != common])
    assert sum(int(frames[i] == glyphs[i]) for i in range(count)) == 1
    return frames, glyphs


def _make_pair(rng, count, shape, arm, background, target, mode, split,
               force_holed_target=False):
    payload_height, payload_width = shape
    outer_height, outer_width = payload_height + 4, payload_width + 4
    assert 2 * arm < min(outer_height, outer_width)
    height, width, positions = _layout(rng, count, outer_height, outer_width)
    frames, glyphs = _colors(rng, background, count, target, mode, split)
    grid = np.full((height, width), background, dtype=np.int8)
    frame_mask = _frame_mask(outer_height, outer_width, arm)
    output = None
    for index, (top, left) in enumerate(positions):
        panel = grid[top:top + outer_height, left:left + outer_width]
        panel[frame_mask] = frames[index]
        payload = np.full(shape, background, dtype=np.int8)
        mask = _glyph_mask(rng, payload_height, payload_width,
                           allow_full=not (force_holed_target and index == target))
        payload[mask] = glyphs[index]
        panel[2:-2, 2:-2] = payload
        if index == target:
            output = payload.copy()
    return {"input": grid, "output": output}


def _grid_key(grid):
    return (tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes())


def _source_copy(grid):
    return (grid.shape == (30, 30) and
            hashlib.sha256(np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()
            in _SOURCE_INPUT_DIGESTS)


def _find_panels(grid, shape, background, arm):
    payload_height, payload_width = shape
    outer_height, outer_width = payload_height + 4, payload_width + 4
    if 2 * arm >= min(outer_height, outer_width):
        return []
    frame_mask = _frame_mask(outer_height, outer_width, arm)
    payload_mask = np.zeros((outer_height, outer_width), dtype=bool)
    payload_mask[2:-2, 2:-2] = True
    moat = ~(frame_mask | payload_mask)
    panels = []
    for top, left in np.argwhere(grid != background):
        top, left = int(top), int(left)
        if top + outer_height > grid.shape[0] or left + outer_width > grid.shape[1]:
            continue
        region = grid[top:top + outer_height, left:left + outer_width]
        color = int(grid[top, left])
        if not np.all(region[frame_mask] == color):
            continue
        if not np.all(region[moat] == background):
            continue
        payload = region[2:-2, 2:-2].copy()
        glyphs = np.unique(payload[payload != background])
        if len(glyphs) != 1:
            continue
        panels.append((top, left, color, int(glyphs[0]), payload))
    return panels


def _interpret(grid, shape):
    background = int(grid[0, 0])
    assert np.all(grid[0, :] == background)
    assert np.all(grid[-1, :] == background)
    assert np.all(grid[:, 0] == background)
    assert np.all(grid[:, -1] == background)
    interpretations = []
    for arm in (2, 3):
        panels = _find_panels(grid, shape, background, arm)
        if not 4 <= len(panels) <= 6:
            continue
        occupied = np.zeros(grid.shape, dtype=bool)
        separated = True
        for top, left, _, _, _ in panels:
            outer_height, outer_width = shape[0] + 4, shape[1] + 4
            neighborhood = occupied[max(0, top - 1):top + outer_height + 1,
                                    max(0, left - 1):left + outer_width + 1]
            if np.any(neighborhood):
                separated = False
                break
            occupied[top:top + outer_height, left:left + outer_width] = True
        if not separated or not np.all((grid == background) | occupied):
            continue
        matches = [index for index, panel in enumerate(panels)
                   if panel[2] == panel[3]]
        if len(matches) == 1:
            interpretations.append((arm, panels, matches[0]))
    assert len(interpretations) == 1
    return background, interpretations[0]


def _analysis(pair):
    grid, output = pair["input"], pair["output"]
    background, (arm, panels, target) = _interpret(grid, output.shape)
    assert np.array_equal(output, panels[target][4])
    assert set(int(value) for value in np.unique(output)) in (
        {panels[target][3]}, {background, panels[target][3]}
    )
    frames = [panel[2] for panel in panels]
    glyphs = [panel[3] for panel in panels]
    return {
        "background": background, "arm": arm, "count": len(panels),
        "shape": tuple(output.shape), "canvas": tuple(grid.shape),
        "palette": len(np.unique(grid)), "target": target,
        "tops": tuple(panel[0] for panel in panels),
        "placements": tuple((panel[0], panel[1]) for panel in panels),
        "full_glyphs": sum(bool(np.all(panel[4] != background)) for panel in panels),
        "same_frames": len(set(frames)) == 1,
        "shared_glyph": len(set(glyphs)) < len(glyphs),
        "target_reused": any(index != target and
                             (frames[index] == frames[target] or
                              glyphs[index] == frames[target])
                             for index in range(len(panels))),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            assert key not in seen and not _source_copy(pair["input"])
            seen.add(key)
            assert not np.array_equal(pair["input"], pair["output"])
            analyses[split].append(_analysis(pair))
    train, test = analyses["train"], analyses["test"]
    assert all(item["shape"] in _SHAPES for item in train + test)
    assert all(3 <= item["palette"] <= 9 for item in train + test)
    assert {item["count"] for item in train}.issuperset({4, 5})
    assert (3, 3) in {item["shape"] for item in train}
    assert len({item["shape"] for item in train}) >= 2
    assert {item["arm"] for item in train} == {2, 3}
    assert any(pair["output"].size > np.count_nonzero(
        pair["output"] != item["background"])
        for pair, item in zip(puzzle["train"], train))
    assert len({item["target"] for item in train}) >= 2
    # A test arm or scale is directly visible in its own input. Both arm
    # conventions and at least two scales have already been demonstrated.
    assert all(item["arm"] in {2, 3} for item in test)


def _solve_from_input(train_pairs, test_input):
    # Infer the invariant from demonstrations, then use only the test input.
    for pair in train_pairs:
        _, (_, panels, target) = _interpret(pair["input"], pair["output"].shape)
        assert np.array_equal(pair["output"], panels[target][4])
    answers = []
    for height in range(1, test_input.shape[0] - 3):
        for width in range(1, test_input.shape[1] - 3):
            try:
                _, (_, panels, target) = _interpret(test_input, (height, width))
            except AssertionError:
                continue
            answers.append(panels[target][4])
    assert len(answers) == 1
    return answers[0]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # The semantic convention is fixed across the puzzle: four corners, a
    # two-cell inset, and selection by local color equality. This color order
    # merely distributes literal backgrounds over pairs.
    color_order = rng.permutation(10)
    puzzle = {"train": [], "test": []}
    train_shapes = [(3, 3), _SHAPES[int(rng.integers(0, len(_SHAPES)))]]
    if train_shapes[1] == (3, 3):
        train_shapes[1] = (4, 5)
    while len(train_shapes) < train_count:
        train_shapes.append(_SHAPES[int(rng.integers(0, len(_SHAPES)))])
    train_counts = [4, 5] + [int(rng.integers(4, 7)) for _ in range(train_count - 2)]
    train_arms = [3, 2]
    if min(train_shapes[0]) < 3:
        train_arms = [2, 3 if min(train_shapes[1]) >= 3 else 2]
    if min(train_shapes[1]) < 3:
        train_arms = [3, 2]
    for index in range(2, train_count):
        train_arms.append(int(rng.integers(2, 4)) if min(train_shapes[index]) >= 3 else 2)
    if set(train_arms) != {2, 3}:
        train_shapes[1], train_arms[1] = (4, 5), 3
        train_arms[0] = 2
    first_target = int(rng.integers(0, 4))
    targets = [first_target, (first_target + 1) % 5]
    targets += [int(rng.integers(0, count)) for count in train_counts[2:]]

    requests = []
    for index in range(train_count):
        requests.append(("train", train_counts[index], train_shapes[index],
                         train_arms[index], targets[index]))
    known_shapes = list(dict.fromkeys(train_shapes))
    for index in range(test_count):
        if rng.integers(0, 4) == 0:
            shape = known_shapes[int(rng.integers(0, len(known_shapes)))]
        else:
            shape = _SHAPES[int(rng.integers(0, len(_SHAPES)))]
        count = int(rng.choice((4, 5, 6), p=(.1, .2, .7)))
        arm = int(rng.integers(2, 4)) if min(shape) >= 3 else 2
        requests.append(("test", count, shape, arm,
                         int(rng.integers(0, count))))

    seen = set()
    for ordinal, (split, count, shape, arm, target) in enumerate(requests):
        mode = ("same_frame", "reuse_target", "shared_glyph", "mixed")[
            int(rng.integers(0, 4))]
        background = int(color_order[ordinal % 10])
        for _ in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            try:
                pair = _make_pair(pair_rng, count, shape, arm, background,
                                  target, mode, split,
                                  force_holed_target=(split == "train" and ordinal == 0))
                key = _grid_key(pair["input"])
                if key in seen or _source_copy(pair["input"]):
                    continue
                _analysis(pair)
            except (AssertionError, ValueError):
                continue
            seen.add(key)
            puzzle[split].append(pair)
            break
        else:
            raise AssertionError(f"could not construct pair {ordinal}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _component_count(mask):
    visited = set()
    count = 0
    for row, column in np.argwhere(mask):
        start = (int(row), int(column))
        if start in visited:
            continue
        count += 1
        pending = [start]
        visited.add(start)
        while pending:
            y, x = pending.pop()
            for neighbor in ((y - 1, x), (y + 1, x),
                             (y, x - 1), (y, x + 1)):
                ny, nx = neighbor
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[neighbor] and neighbor not in visited):
                    visited.add(neighbor)
                    pending.append(neighbor)
    return count


def self_test() -> None:
    first = generate_puzzle(123456, 7, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(123456, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + index, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for index, arguments in enumerate((
        {"num_test": 1}, {"num_test": 4},
        {"num_train": 2}, {"num_train": 10},
    )):
        validate_generated_puzzle(generate_puzzle(21000 + index, **arguments))
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    )):
        try:
            generate_puzzle(22000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    analyses = []
    per_puzzle = []
    measured = {name: 0 for name in (
        "full_target", "full_distractor", "shared_different_frames",
        "disconnected", "asymmetric", "test_known", "test_new",
        "train_six", "test_non_six", "test_palette_seven",
        "combined", "tight", "spacious", "palette_overlap", "count_overlap",
    )}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = []
        training = [_analysis(pair) for pair in puzzle["train"]]
        test = [_analysis(pair) for pair in puzzle["test"]]
        measured["palette_overlap"] += bool(
            {item["palette"] for item in training} &
            {item["palette"] for item in test}
        )
        measured["count_overlap"] += bool(
            {item["count"] for item in training} &
            {item["count"] for item in test}
        )
        training_shapes = {item["shape"] for item in training}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analysis(pair)
                item["split"] = split
                analyses.append(item)
                local.append(item)
                background, (_, panels, target) = _interpret(
                    pair["input"], pair["output"].shape
                )
                measured["full_target"] += bool(np.all(
                    panels[target][4] != background
                ))
                measured["full_distractor"] += sum(
                    bool(np.all(panel[4] != background))
                    for index, panel in enumerate(panels) if index != target
                )
                measured["shared_different_frames"] += any(
                    panels[first][3] == panels[second][3]
                    and panels[first][2] != panels[second][2]
                    for first in range(len(panels))
                    for second in range(first + 1, len(panels))
                )
                for panel in panels:
                    mask = panel[4] != background
                    measured["disconnected"] += _component_count(mask) > 1
                    measured["asymmetric"] += not (
                        np.array_equal(mask, np.flipud(mask))
                        and np.array_equal(mask, np.fliplr(mask))
                    )
                measured["train_six"] += split == "train" and item["count"] == 6
                measured["test_non_six"] += split == "test" and item["count"] != 6
                measured["test_palette_seven"] += (
                    split == "test" and item["palette"] == 7
                )
                measured["test_known"] += (
                    split == "test" and item["shape"] in training_shapes
                )
                measured["test_new"] += (
                    split == "test" and item["shape"] not in training_shapes
                )
                measured["combined"] += (
                    len(set(item["tops"])) == item["count"]
                    and item["target_reused"]
                    and item["shape"] not in ((2, 2), (3, 3), (4, 5))
                )
                density = (item["count"] * (item["shape"][0] + 4)
                           * (item["shape"][1] + 4)
                           / (item["canvas"][0] * item["canvas"][1]))
                measured["tight"] += density >= .4
                measured["spacious"] += density <= .3
        per_puzzle.append(local)
    assert len({item["canvas"] for item in analyses}) >= 20
    assert len({item["shape"] for item in analyses}) >= 9
    assert len({(item["canvas"], item["placements"]) for item in analyses}) >= 100
    assert {item["arm"] for item in analyses} == {2, 3}
    assert {item["count"] for item in analyses} == {4, 5, 6}
    assert len({item["background"] for item in analyses}) == 10
    assert sum(item["canvas"] == (30, 30) for item in analyses) >= 10
    assert sum(len(set(item["tops"])) == item["count"] for item in analyses) >= 10
    assert sum(item["full_glyphs"] > 0 for item in analyses) >= 10
    assert sum(item["same_frames"] for item in analyses) >= 10
    assert sum(item["target_reused"] for item in analyses) >= 30
    assert sum(item["shared_glyph"] for item in analyses) >= 30
    assert sum(len({item["arm"] for item in local}) > 1 for local in per_puzzle) == 42
    assert sum(len({item["shape"] for item in local}) > 1 for local in per_puzzle) == 42
    for name, minimum in (
        ("full_target", 10), ("full_distractor", 50),
        ("shared_different_frames", 50), ("disconnected", 50),
        ("asymmetric", 100), ("test_known", 10), ("test_new", 10),
        ("train_six", 10), ("test_non_six", 5),
        ("test_palette_seven", 3), ("combined", 30),
        ("tight", 30), ("spacious", 20),
        ("palette_overlap", 5), ("count_overlap", 5),
    ):
        assert measured[name] >= minimum, (name, measured[name])
    # Predict held-out outputs using demonstrations and the test input only.
    for puzzle in puzzles[:8]:
        for pair in puzzle["test"]:
            predicted = _solve_from_input(puzzle["train"], pair["input"])
            assert np.array_equal(predicted, pair["output"])
