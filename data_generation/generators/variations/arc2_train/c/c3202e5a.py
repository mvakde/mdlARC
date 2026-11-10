"""Generate compartment-selection ARC puzzles.

The complete separator lines delimit equal rectangular interiors.  The answer is
exactly the unique nonempty interior with one color besides its background.
"""

import numpy as np


_SOURCE_OUTPUTS = (
    np.array([[0, 7, 0, 0, 7], [0, 0, 0, 7, 0], [0, 0, 0, 7, 0],
              [0, 0, 7, 0, 0], [0, 0, 0, 0, 7]], dtype=np.int8),
    np.array([[0, 2, 0, 0, 0], [0, 0, 0, 0, 2], [0, 0, 0, 2, 2],
              [0, 2, 0, 2, 2], [0, 0, 0, 2, 0]], dtype=np.int8),
    np.array([[0, 0, 4], [4, 4, 0], [0, 4, 0]], dtype=np.int8),
    np.array([[0, 0, 8, 0, 8], [0, 0, 0, 0, 0], [8, 0, 0, 0, 0],
              [0, 0, 8, 8, 0], [0, 0, 0, 0, 8]], dtype=np.int8),
)
_SOURCE_LAYOUT_A = (4, 4, 5, 5)
_SOURCE_LAYOUT_B = (6, 6, 3, 3)
_SOURCE_TEST_LAYOUT = (5, 5, 5, 5)
_LAYOUTS = tuple(
    (rows, cols, height, width)
    for rows in range(3, 7)
    for cols in range(3, 7)
    for height in range(3, 7)
    for width in range(3, 7)
    if rows * (height + 1) - 1 <= 30
    and cols * (width + 1) - 1 <= 30
)


def _normalise_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _choose_counts(rng, seed, num_train, num_test):
    train = _normalise_count(num_train, "num_train")
    test = _normalise_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _source_output(grid):
    return any(grid.shape == old.shape and np.array_equal(grid, old)
               for old in _SOURCE_OUTPUTS)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _layout(rng, regime, index, context, train_layouts):
    if context["source_shift"]:
        if regime == "source_train" and index < 3:
            return (_SOURCE_LAYOUT_A, _SOURCE_LAYOUT_A, _SOURCE_LAYOUT_B)[index]
        if regime == "target_test" and int(rng.integers(4)) != 0:
            return _SOURCE_TEST_LAYOUT
    if regime == "source_train" and index and int(rng.integers(4)) == 0:
        return _pick(rng, train_layouts)
    if regime == "target_test" and int(rng.integers(4)) == 0:
        return _pick(rng, train_layouts)
    # Draw the two axes independently, subject only to the ARC size limit.
    return _pick(rng, _LAYOUTS)


def _roles(rng, regime, index, context, train_roles):
    if context["shared_background"]:
        background = context["background"]
    elif regime == "source_train" and index == 1:
        background = _pick(rng, [c for c in range(10) if c != train_roles[0][0]])
    elif regime == "target_test" and context["source_shift"] and index == 0:
        background = _pick(rng, [c for c in range(10)
                                 if c not in train_roles[1][1:]])
    else:
        background = int(rng.integers(10))
    available = [c for c in range(10) if c != background]
    if regime == "source_train" or not train_roles:
        separator = _pick(rng, available)
        same_color = int(rng.integers(9)) == 0 and not (
            context["source_shift"] and index == 1)
        foreground = (separator if same_color else
                      _pick(rng, [c for c in available if c != separator]))
    else:
        mode = 0 if context["source_shift"] and index == 0 else int(rng.integers(4))
        anchor = train_roles[1] if mode == 0 and len(train_roles) > 1 else _pick(rng, train_roles)
        if mode == 0 or mode == 1:  # a swap, including the original train-2 swap
            separator, foreground = anchor[2], anchor[1]
        elif mode == 2:  # retained roles
            separator, foreground = anchor[1], anchor[2]
        else:  # foreground identity unseen as a training payload, if available
            separator = _pick(rng, available)
            used = {role[2] for role in train_roles}
            fresh = [c for c in available if c != separator and c not in used]
            foreground = _pick(rng, fresh or [c for c in available if c != separator])
        if background in (separator, foreground):
            separator = _pick(rng, available)
            foreground = _pick(rng, [c for c in available if c != separator])
    return background, separator, foreground


def _connected_mask(rng, height, width, count):
    mask = np.zeros((height, width), dtype=bool)
    cell = (int(rng.integers(height)), int(rng.integers(width)))
    mask[cell] = True
    while int(mask.sum()) < count:
        frontier = []
        for row, col in zip(*np.where(mask)):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            break
        mask[_pick(rng, frontier)] = True
    return mask


def _payload(rng, height, width, background, foreground):
    area = height * width
    mode = int(rng.integers(7))
    mask = np.zeros((height, width), dtype=bool)
    if mode == 0:  # occasional tiny answer
        count = int(rng.integers(1, min(4, area)))
        mask.flat[rng.choice(area, count, replace=False)] = True
    elif mode == 1:  # connected object
        count = int(rng.integers(3, max(4, int(area * .65) + 1)))
        mask = _connected_mask(rng, height, width, min(count, area - 1))
    elif mode == 2:  # row or column line, possibly incomplete
        if int(rng.integers(2)):
            mask[int(rng.integers(height)), :] = True
        else:
            mask[:, int(rng.integers(width))] = True
        if int(rng.integers(2)):
            mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif mode == 3:  # reflection-symmetric pattern
        half = rng.random((height, (width + 1) // 2)) < float(rng.uniform(.25, .7))
        mask[:, :half.shape[1]] = half
        mask[:, width // 2 + width % 2:] = np.fliplr(half[:, :width // 2])
    elif mode == 4:  # explicitly touch every boundary
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
        extra = int(rng.integers(0, max(2, area // 3)))
        mask.flat[rng.choice(area, extra, replace=False)] = True
    else:  # scattered, including some denser than every decoy
        low = 2 if mode == 5 else max(3, int(area * .72))
        high = max(low, int(area * (.55 if mode == 5 else .93)))
        count = int(rng.integers(low, min(area - 1, high) + 1))
        mask.flat[rng.choice(area, count, replace=False)] = True
    if not mask.any():
        mask.flat[int(rng.integers(area))] = True
    if mask.all():
        mask.flat[int(rng.integers(area))] = False
    block = np.full((height, width), background, dtype=np.int8)
    block[mask] = foreground
    return block


def _decoy(rng, height, width, background, palette):
    area = height * width
    upper = min(area - 1, max(3, int(area * .78)))
    count = int(rng.integers(2, upper + 1))
    distinct = int(rng.integers(2, min(len(palette), count, 7) + 1))
    chosen = list(rng.choice(palette, distinct, replace=False))
    colors = chosen + [int(_pick(rng, chosen)) for _ in range(count - distinct)]
    rng.shuffle(colors)
    block = np.full((height, width), background, dtype=np.int8)
    if int(rng.integers(4)) == 0:
        # Locally clustered marks, with no change to the color-set rule.
        positions = list(zip(*np.where(_connected_mask(rng, height, width, count))))
        positions = [int(row * width + col) for row, col in positions]
    else:
        positions = rng.choice(area, count, replace=False)
    for pos, color in zip(positions, colors):
        block.flat[int(pos)] = color
    return block


def _make_pair(rng, layout, roles, blank_policy=0):
    rows, cols, height, width = layout
    background, separator, foreground = roles
    palette_size = 10 if int(rng.integers(3)) == 0 else int(rng.integers(4, 10))
    role_colors = set((separator, foreground))
    other = [c for c in range(10) if c != background and c not in role_colors]
    extra = list(rng.choice(other, palette_size - 1 - len(role_colors), replace=False))
    # Local appearances of the structural and target colors are optional.
    # This varies their usefulness as clues without changing the pure-block
    # criterion.  Retain at least two decoy colors in every case.
    decoy_palette = list(dict.fromkeys([separator, foreground] + [int(c) for c in extra]))
    if int(rng.integers(3)) == 0 and len(decoy_palette) > 2:
        decoy_palette = [c for c in decoy_palette if c != separator]
    if int(rng.integers(3)) == 0 and len(decoy_palette) > 2:
        decoy_palette = [c for c in decoy_palette if c != foreground]
    canvas = np.full((rows * (height + 1) - 1, cols * (width + 1) - 1),
                     background, dtype=np.int8)
    for row in range(height, canvas.shape[0], height + 1):
        canvas[row, :] = separator
    for col in range(width, canvas.shape[1], width + 1):
        canvas[:, col] = separator
    payload_position = int(rng.integers(rows * cols))
    forced_blank = (int(rng.integers(rows * cols - 1)) if blank_policy == 2 else -1)
    decoy_index = 0
    payload = _payload(rng, height, width, background, foreground)
    for row in range(rows):
        for col in range(cols):
            if row * cols + col == payload_position:
                block = payload
            else:
                blank = (decoy_index == forced_blank or
                         (blank_policy and int(rng.integers(18)) == 0))
                block = (np.full((height, width), background, dtype=np.int8)
                         if blank else _decoy(rng, height, width, background, decoy_palette))
                decoy_index += 1
            rr, cc = row * (height + 1), col * (width + 1)
            canvas[rr:rr + height, cc:cc + width] = block
    return canvas, payload.copy()


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _geometry(grid):
    height, width = grid.shape
    row_lines = [r for r in range(height) if np.all(grid[r] == grid[r, 0])]
    col_lines = [c for c in range(width) if np.all(grid[:, c] == grid[0, c])]
    assert row_lines and col_lines
    separator = int(grid[row_lines[0], 0])
    assert all(np.all(grid[r] == separator) for r in row_lines)
    assert all(np.all(grid[:, c] == separator) for c in col_lines)
    row_edges = (-1, *row_lines, height)
    col_edges = (-1, *col_lines, width)
    row_sizes = [row_edges[i + 1] - row_edges[i] - 1 for i in range(len(row_edges) - 1)]
    col_sizes = [col_edges[i + 1] - col_edges[i] - 1 for i in range(len(col_edges) - 1)]
    assert len(set(row_sizes)) == len(set(col_sizes)) == 1
    assert row_sizes[0] >= 3 and col_sizes[0] >= 3
    assert 3 <= len(row_sizes) <= 6 and 3 <= len(col_sizes) <= 6
    return separator, row_lines, col_lines, row_sizes[0], col_sizes[0]


def _inspect(input_grid, output_grid=None):
    separator, row_lines, col_lines, height, width = _geometry(input_grid)
    interiors = np.delete(np.delete(input_grid, row_lines, axis=0), col_lines, axis=1)
    values, counts = np.unique(interiors, return_counts=True)
    order = np.argsort(counts)
    assert len(order) >= 2 and counts[order[-1]] > counts[order[-2]]
    background = int(values[order[-1]])
    assert background != separator
    rows = len(row_lines) + 1
    cols = len(col_lines) + 1
    blocks = []
    color_sets = []
    counts_by_block = []
    for row in range(rows):
        for col in range(cols):
            rr, cc = row * (height + 1), col * (width + 1)
            block = input_grid[rr:rr + height, cc:cc + width]
            colors = set(int(c) for c in np.unique(block)) - {background}
            blocks.append(block)
            color_sets.append(colors)
            counts_by_block.append(int(np.count_nonzero(block != background)))
    pure = [i for i, colors in enumerate(color_sets) if len(colors) == 1]
    assert len(pure) == 1
    target = blocks[pure[0]]
    foreground = next(iter(color_sets[pure[0]]))
    assert 0 < counts_by_block[pure[0]] < height * width
    assert all(len(colors) == 0 or len(colors) >= 2 for i, colors in enumerate(color_sets)
               if i != pure[0])
    if output_grid is not None:
        assert output_grid.shape == target.shape
        assert np.array_equal(output_grid, target)
        assert not _source_output(output_grid)
    return {
        "layout": (rows, cols, height, width),
        "background": background,
        "separator": separator,
        "foreground": foreground,
        "position": divmod(pure[0], cols),
        "payload_count": counts_by_block[pure[0]],
        "decoy_counts": [n for i, n in enumerate(counts_by_block) if i != pure[0]],
        "blank_count": sum(not colors for colors in color_sets),
        "palette_size": len(np.unique(input_grid)),
        "separator_in_decoys": any(separator in colors for i, colors in enumerate(color_sets) if i != pure[0]),
        "payload_in_decoys": any(foreground in colors for i, colors in enumerate(color_sets) if i != pure[0]),
        "target": target.copy(),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert 0 <= grid.min() <= grid.max() <= 9
        input_key, output_key = _grid_key(pair["input"]), _grid_key(pair["output"])
        assert input_key not in seen_inputs and (input_key, output_key) not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add((input_key, output_key))
        assert not np.array_equal(pair["input"], pair["output"])
        infos.append(_inspect(pair["input"], pair["output"]))
    # No hidden color convention is required. The common selection rule is
    # demonstrated in every training pair, including the input-local background.
    assert all(info["target"].shape == pair["output"].shape
               for info, pair in zip(infos, puzzle["train"] + puzzle["test"]))
    if any(info["blank_count"] for info in infos[len(puzzle["train"]):]):
        assert any(info["blank_count"] for info in infos[:len(puzzle["train"])])
    training_backgrounds = {info["background"] for info in infos[:len(puzzle["train"])]}
    if any(info["background"] not in training_backgrounds
           for info in infos[len(puzzle["train"]):]):
        assert len(training_backgrounds) >= 2


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(rng, seed, num_train, num_test)
    context = {
        "background": int(rng.integers(10)),
        "shared_background": int(rng.integers(3)) != 0,
        "source_shift": int(rng.integers(5)) == 0,
        "blank_exposure": int(rng.integers(3)) == 0,
    }
    puzzle = {"train": [], "test": []}
    used = set()
    train_layouts, train_roles = [], []
    for regime, count, destination in (("source_train", train_count, "train"),
                                       ("target_test", test_count, "test")):
        for index in range(count):
            for _ in range(64):
                layout = _layout(rng, regime, index, context, train_layouts)
                roles = _roles(rng, regime, index, context, train_roles)
                blank_policy = (2 if context["blank_exposure"] and regime == "source_train"
                                and index == 0 else 1 if context["blank_exposure"] else 0)
                input_grid, output_grid = _make_pair(rng, layout, roles, blank_policy)
                key = _grid_key(input_grid)
                if key in used or _source_output(output_grid):
                    continue
                try:
                    _inspect(input_grid, output_grid)
                except AssertionError:
                    continue
                used.add(key)
                puzzle[destination].append({"input": input_grid, "output": output_grid})
                if regime == "source_train":
                    train_layouts.append(layout)
                    train_roles.append(roles)
                break
            else:
                raise RuntimeError("could not construct an unambiguous pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(left, right):
    return all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(left[split], right[split]) for k in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    puzzles += [generate_puzzle(101, 2, 1), generate_puzzle(102, 10, 1),
                generate_puzzle(103, 10, 4), generate_puzzle(201, None, 1),
                generate_puzzle(202, None, 4), generate_puzzle(203, 2, None),
                generate_puzzle(204, 10, None)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            info = _inspect(pair["input"])
            assert np.array_equal(info["target"], pair["output"])
    assert _equal_puzzles(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _equal_puzzles(generate_puzzle(314159), generate_puzzle(314160))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {args}")
    automatic = puzzles[:42]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    infos = [_inspect(pair["input"], pair["output"])
             for puzzle in automatic for pair in puzzle["train"] + puzzle["test"]]
    assert len({i["layout"] for i in infos}) >= 20
    assert len({(i["layout"][2], i["layout"][3]) for i in infos}) >= 8
    assert len({i["background"] for i in infos}) >= 8
    assert len({i["separator"] for i in infos}) >= 8
    assert len({i["foreground"] for i in infos}) >= 8
    assert any(i["layout"][0] != i["layout"][1] for i in infos)
    assert any(i["layout"][2] != i["layout"][3] for i in infos)
    assert any(i["palette_size"] < 10 for i in infos)
    assert any(i["palette_size"] == 10 for i in infos)
    assert any(i["payload_count"] < 3 for i in infos)
    assert any(i["payload_count"] > 10 for i in infos)
    assert any(max(i["decoy_counts"]) > i["payload_count"] for i in infos)
    assert any(i["payload_count"] > max(i["decoy_counts"]) for i in infos)
    assert any(not i["separator_in_decoys"] for i in infos)
    assert any(not i["payload_in_decoys"] for i in infos)
    assert any(i["blank_count"] for i in infos)
    assert any(i["foreground"] == i["separator"] for i in infos)
    masks = [(i["target"] != i["background"]) for i in infos]
    assert any(m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()
               for m in masks)
    assert any(np.array_equal(m, np.fliplr(m)) for m in masks)
    assert any(np.all(m, axis=0).any() or np.all(m, axis=1).any() for m in masks)
    repeated_train = same_layout = changed_size = changed_count = False
    swapped_second = retained = fresh_payload = varied_background = False
    blank_test_covered = False
    for puzzle in automatic:
        train = [_inspect(p["input"]) for p in puzzle["train"]]
        test = [_inspect(p["input"]) for p in puzzle["test"]]
        train_layouts = {i["layout"] for i in train}
        train_sizes = {i["layout"][2:] for i in train}
        train_counts = {i["layout"][:2] for i in train}
        train_payloads = {i["foreground"] for i in train}
        repeated_train |= len(train_layouts) < len(train)
        same_layout |= any(i["layout"] in train_layouts for i in test)
        changed_size |= any(i["layout"][2:] not in train_sizes for i in test)
        changed_count |= any(i["layout"][:2] not in train_counts for i in test)
        swapped_second |= any((i["separator"], i["foreground"]) ==
                              (train[1]["foreground"], train[1]["separator"])
                              for i in test)
        retained |= any((i["separator"], i["foreground"]) ==
                        (t["separator"], t["foreground"])
                        for i in test for t in train)
        fresh_payload |= any(i["foreground"] not in train_payloads for i in test)
        varied_background |= len({i["background"] for i in train + test}) > 1
        blank_test_covered |= (any(i["blank_count"] for i in test)
                               and any(i["blank_count"] for i in train))
    assert all((repeated_train, same_layout, changed_size, changed_count,
                swapped_second, retained, fresh_payload, varied_background,
                blank_test_covered))
