"""Self-copy block substitution puzzles.

The two color roles are fixed within a puzzle.  All geometry and the other
colors belong to individual examples.
"""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[0, 5, 0, 3], [5, 5, 2, 0], [0, 2, 5, 5], [3, 0, 5, 0]], dtype=np.int8),
    np.array([[2, 5, 1], [0, 5, 0], [2, 5, 1]], dtype=np.int8),
    np.array([[5, 5, 5, 5], [5, 2, 3, 5], [5, 3, 3, 5], [5, 5, 5, 5]], dtype=np.int8),
    np.array([[5, 0, 1], [5, 2, 0], [5, 5, 5]], dtype=np.int8),
    np.array([[1, 0, 5, 0, 1], [0, 2, 2, 2, 0], [5, 0, 5, 0, 5],
              [0, 2, 2, 2, 0], [1, 0, 5, 0, 1]], dtype=np.int8),
)


def _checked_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _build_output(seed_grid, trigger, blank):
    n = seed_grid.shape[0]
    output = np.full((n * n, n * n), blank, dtype=np.int8)
    for row, col in np.argwhere(seed_grid == trigger):
        output[row * n:(row + 1) * n, col * n:(col + 1) * n] = seed_grid
    return output


def _mask(rng, n, kind):
    cells = n * n
    mask = np.zeros((n, n), dtype=bool)
    if kind == "mixed" and n == 2:
        kind = "diagonal"
    if kind == "minority":
        kind = ("sparse", "line", "diagonal", "scattered")[int(rng.integers(0, 4))]
        limit = max(1, (cells - 1) // 3)
    else:
        limit = cells - 1

    if kind == "sparse":
        count = int(rng.integers(1, min(2, limit) + 1))
        mask.flat[rng.choice(cells, count, replace=False)] = True
    elif kind == "line":
        index = int(rng.integers(0, n))
        if int(rng.integers(0, 2)):
            mask[index, :] = True
        else:
            mask[:, index] = True
    elif kind == "border":
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if n >= 4 and int(rng.integers(0, 2)):
            edge = np.argwhere(mask)
            for row, col in edge[rng.choice(len(edge), size=int(rng.integers(1, 3)), replace=False)]:
                mask[row, col] = False
    elif kind == "source_cross":
        assert n == 5
        mask[2, 2] = True
        mask[0, 2] = mask[4, 2] = True
        mask[2, 0] = mask[2, 4] = True
    elif kind == "cross":
        if n == 5 and int(rng.integers(0, 4)) == 0:
            # Retain the source's sparse, centered five-cell cross sometimes.
            mask[2, 2] = True
            mask[0, 2] = mask[4, 2] = True
            mask[2, 0] = mask[2, 4] = True
        else:
            row = int(rng.integers(0, n))
            col = int(rng.integers(0, n))
            mask[row, :] = True
            mask[:, col] = True
            if int(rng.integers(0, 3)) == 0:
                mask[row, col] = False
    elif kind == "near_full":
        mask[:] = True
        mask.flat[rng.choice(cells, size=int(rng.integers(1, min(2, cells - 1) + 1)), replace=False)] = False
    elif kind == "diagonal":
        if int(rng.integers(0, 2)):
            np.fill_diagonal(mask, True)
        else:
            mask[np.arange(n), np.arange(n - 1, -1, -1)] = True
        if limit == cells - 1 and int(rng.integers(0, 2)):
            mask.flat[int(rng.integers(0, cells))] = True
    elif kind == "cluster":
        count = int(rng.integers(3, max(4, cells - 1)))
        row, col = (int(x) for x in rng.integers(0, n, size=2))
        mask[row, col] = True
        for _ in range(count * 12):
            if int(mask.sum()) >= count:
                break
            direction = int(rng.integers(0, 4))
            row = min(n - 1, max(0, row + (direction == 0) - (direction == 1)))
            col = min(n - 1, max(0, col + (direction == 2) - (direction == 3)))
            mask[row, col] = True
        if int(mask.sum()) < count:
            available = np.flatnonzero(~mask)
            mask.flat[rng.choice(available, count - int(mask.sum()), replace=False)] = True
    elif kind == "mixed":
        # A touching pair plus a separated cell, in arbitrary orientation.
        if int(rng.integers(0, 2)):
            row, col = int(rng.integers(0, n)), int(rng.integers(0, n - 1))
            neighbor = (row, col + 1)
        else:
            row, col = int(rng.integers(0, n - 1)), int(rng.integers(0, n))
            neighbor = (row + 1, col)
        pair = ((row, col), neighbor)
        isolated = [(r, c) for r in range(n) for c in range(n)
                    if all(abs(r - pr) + abs(c - pc) > 1 for pr, pc in pair)]
        extra = isolated[int(rng.integers(0, len(isolated)))]
        mask[row, col] = mask[neighbor] = mask[extra] = True
        if n >= 4 and int(rng.integers(0, 2)):
            extensions = [(r, c) for r in range(n) for c in range(n)
                          if not mask[r, c] and abs(r - extra[0]) + abs(c - extra[1]) > 1
                          and any(abs(r - pr) + abs(c - pc) == 1 for pr, pc in pair)]
            if extensions:
                mask[extensions[int(rng.integers(0, len(extensions)))]] = True
    elif kind == "checker":
        rows, cols = np.indices((n, n))
        mask = (rows + cols) % 2 == int(rng.integers(0, 2))
        for index in rng.choice(cells, size=int(rng.integers(0, max(2, n))), replace=False):
            mask.flat[index] = ~mask.flat[index]
    elif kind == "scattered":
        upper = min(limit, cells - 2)
        lower = min(3, upper)
        count = int(rng.integers(lower, upper + 1))
        mask.flat[rng.choice(cells, count, replace=False)] = True
    else:
        raise ValueError("unknown mask kind")

    if not mask.any():
        mask.flat[int(rng.integers(0, cells))] = True
    if mask.all():
        mask.flat[int(rng.integers(0, cells))] = False
    return mask


def _fill_seed(rng, mask, trigger, blank, content_pool, blank_present):
    n = mask.shape[0]
    free = np.argwhere(~mask)
    free = free[rng.permutation(len(free))]
    free_count = len(free)
    if blank_present:
        if free_count == 1:
            blank_count = 1
        else:
            fraction = float(rng.uniform(0.08, 0.72))
            blank_count = min(free_count - 1, max(1, int(round(free_count * fraction))))
    else:
        blank_count = 0
    blanks, content_positions = free[:blank_count], free[blank_count:]
    grid = np.full((n, n), trigger, dtype=np.int8)
    for row, col in blanks:
        grid[row, col] = blank
    if not len(content_positions):
        return grid

    maximum = min(len(content_pool), len(content_positions))
    tier = int(rng.integers(0, 5))
    if tier == 0:
        color_count = 1
    elif tier <= 2:
        color_count = int(rng.integers(1, min(3, maximum) + 1))
    else:
        color_count = int(rng.integers(min(3, maximum), maximum + 1))
    colors = rng.choice(content_pool, size=color_count, replace=False)
    style = int(rng.integers(0, 6))
    phase = int(rng.integers(0, color_count))
    for row, col in content_positions:
        row, col = int(row), int(col)
        if style == 0:
            index = int(rng.integers(0, color_count))
        elif style == 1:
            index = (row + phase) % color_count
        elif style == 2:
            index = (col + phase) % color_count
        elif style == 3:
            index = (row // 2 + col // 2 + phase) % color_count
        else:
            index = 0
        grid[row, col] = colors[index]

    if style == 4:
        # Uneven small patches and short objects over a base color.
        for _ in range(int(rng.integers(1, n + 2))):
            row, col = (int(x) for x in rng.integers(0, n, size=2))
            height = int(rng.integers(1, min(3, n - row) + 1))
            width = int(rng.integers(1, min(3, n - col) + 1))
            color = int(colors[int(rng.integers(0, color_count))])
            patch = grid[row:row + height, col:col + width]
            patch[patch != trigger] = color
            # A blank cell stays a blank gap, regardless of a patch crossing it.
            for br, bc in blanks:
                grid[br, bc] = blank
    elif style == 5:
        # Scattered accents can make asymmetric motifs without changing roles.
        for row, col in content_positions[rng.permutation(len(content_positions))[:max(1, len(content_positions) // 3)]]:
            grid[row, col] = colors[int(rng.integers(0, color_count))]

    # Ensure each selected content color is visible even in small tiles.
    for color, position in zip(colors, content_positions):
        grid[position[0], position[1]] = color
    return grid


def _grid_key(grid):
    return tuple(grid.shape), grid.astype(np.int8, copy=False).tobytes()


def _is_source_pair(seed_grid, output):
    for source in _SOURCE_INPUTS:
        if seed_grid.shape == source.shape and np.array_equal(seed_grid, source):
            if np.array_equal(output, _build_output(source, 5, 0)):
                return True
    return False


def _infer_pair_roles(seed_grid, output):
    n = seed_grid.shape[0]
    assert output.shape == (n * n, n * n)
    active = np.zeros((n, n), dtype=bool)
    blanks = set()
    for row in range(n):
        for col in range(n):
            block = output[row * n:(row + 1) * n, col * n:(col + 1) * n]
            if np.array_equal(block, seed_grid):
                active[row, col] = True
            else:
                assert np.all(block == block[0, 0]), "inactive block must be uniform"
                blanks.add(int(block[0, 0]))
    assert active.any() and not active.all()
    assert len(blanks) == 1, "all inactive blocks use one blank role"
    trigger_values = np.unique(seed_grid[active])
    assert len(trigger_values) == 1
    trigger, blank = int(trigger_values[0]), next(iter(blanks))
    assert trigger != blank
    assert np.array_equal(active, seed_grid == trigger)
    assert np.array_equal(output, _build_output(seed_grid, trigger, blank))
    return trigger, blank, active


def _make_pair(rng, n, kind, blank_present, roles, seen, used_masks=None):
    trigger, blank, content_pool = roles
    fallback_kinds = ("sparse", "line", "diagonal", "scattered", "cross",
                      "cluster", "checker", "border", "near_full")
    for attempt in range(128):
        if used_masks is not None and attempt and attempt % 64 == 0:
            n = 3 + (n - 2) % 3
        if attempt < 16:
            trial_kind = kind
        else:
            trial_kind = fallback_kinds[(attempt // 16 - 1) % len(fallback_kinds)]
        mask = _mask(rng, n, trial_kind)
        mask_key = (n, mask.tobytes())
        if used_masks is not None and mask_key in used_masks:
            continue
        seed_grid = _fill_seed(rng, mask, trigger, blank, content_pool, blank_present)
        output = _build_output(seed_grid, trigger, blank)
        key = _grid_key(seed_grid)
        if key not in seen and not _is_source_pair(seed_grid, output):
            seen.add(key)
            if used_masks is not None:
                used_masks.add(mask_key)
            return {"input": seed_grid, "output": output}
    raise RuntimeError("unable to construct a unique pair")


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
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    roles = (int(palette[0]), int(palette[1]), palette[2:])
    seen = set()
    source_shift = bool(rng.integers(0, 5) == 0)

    first_size = int(rng.choice([3, 4]))
    if source_shift:
        second_size = 7 - first_size
    else:
        other_sizes = [n for n in (2, 3, 4, 5) if n != first_size]
        second_size = int(rng.choice(other_sizes, p=[0.15, 0.30, 0.55]))
    train_specs = [(first_size, "minority", True)]
    train_kinds = ("line", "border", "cross", "near_full", "diagonal", "cluster", "checker", "scattered", "sparse", "mixed")
    train_specs.append((second_size, str(rng.choice(train_kinds[1:])), False))
    for _ in range(2, train_count):
        n = int(rng.choice([3, 4])) if source_shift else int(rng.choice([2, 3, 4, 5], p=[0.07, 0.28, 0.32, 0.33]))
        train_specs.append((n, str(rng.choice(train_kinds)), bool(rng.integers(0, 2))))
    rng.shuffle(train_specs)

    # Distinct challenge bands keep multiple test cases useful, while their
    # order and sizes vary independently across puzzles.
    bands = ["sparse", "structured", "dense", "irregular"]
    rng.shuffle(bands)
    test_specs = []
    for index in range(test_count):
        band = bands[index]
        if band == "structured":
            kind = str(rng.choice(("line", "border", "cross", "diagonal")))
        elif band == "irregular":
            kind = "mixed" if test_count == 4 else str(rng.choice(("mixed", "scattered", "cluster", "checker")))
        elif band == "dense":
            kind = "near_full"
        else:
            kind = "sparse"
        n = 5 if source_shift else int(rng.choice([3, 4, 5], p=[0.22, 0.25, 0.53]))
        present = bool(rng.integers(0, 2)) if test_count == 1 else index % 2 == 0
        test_specs.append((n, kind, present))
    if source_shift and int(rng.integers(0, 2)):
        source_index = bands.index("structured") if "structured" in bands[:test_count] else int(rng.integers(0, test_count))
        _, _, present = test_specs[source_index]
        test_specs[source_index] = (5, "source_cross", present)
    rng.shuffle(test_specs)

    puzzle = {"train": [], "test": []}
    used_test_masks = set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for n, kind, present in specs:
            masks = used_test_masks if split == "test" else None
            puzzle[split].append(_make_pair(rng, n, kind, present, roles, seen, masks))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    roles = set()
    train_sizes = set()
    train_blank_presence = set()
    train_minority = False
    test_masks = set()
    test_counts = set()
    test_components = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            seed_grid, output = pair["input"], pair["output"]
            for grid in (seed_grid, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            n = seed_grid.shape[0]
            assert seed_grid.shape == (n, n)
            assert (2 <= n <= 5) if split == "train" else (3 <= n <= 5)
            key = _grid_key(seed_grid)
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert not _is_source_pair(seed_grid, output)
            trigger, blank, active = _infer_pair_roles(seed_grid, output)
            roles.add((trigger, blank))
            assert len(np.unique(seed_grid)) >= 2
            if split == "train":
                train_sizes.add(n)
                train_blank_presence.add(bool(np.any(seed_grid == blank)))
                train_minority |= int(active.sum()) < n * n / 2
            else:
                mask_key = (n, active.tobytes())
                assert mask_key not in test_masks
                test_masks.add(mask_key)
                test_counts.add(int(active.sum()))
                test_components.add(_component_count(active))
    assert len(roles) == 1, "the two color roles must agree across the puzzle"
    assert len(train_sizes) >= 2
    assert train_blank_presence == {False, True}
    assert train_minority
    if test_count == 4:
        assert len(test_counts) >= 3 and len(test_components) >= 2
    trigger, blank = next(iter(roles))
    # Infer the roles again using training alone, then reconstruct each test.
    train_roles = {_infer_pair_roles(p["input"], p["output"])[:2] for p in puzzle["train"]}
    assert train_roles == {(trigger, blank)}
    for pair in puzzle["test"]:
        assert np.array_equal(_build_output(pair["input"], trigger, blank), pair["output"])


def _signature(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzles, inputs, masks, role_pairs = set(), set(), set(), set()
    train_sizes, test_sizes, trigger_counts = set(), set(), set()
    palettes, components, blank_presence = set(), set(), set()
    minority_train = dense_test = sparse_test = inactive_center = corner_test = 0
    source_shift = source_cross = offcenter_test = mixed_test = 0
    train_blank_gaps = test_blank_gaps = test_lines = test_borders = 0
    four_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        trigger, blank, _ = _infer_pair_roles(puzzle["train"][0]["input"], puzzle["train"][0]["output"])
        role_pairs.add((trigger, blank))
        train_sizes_here = {len(p["input"]) for p in puzzle["train"]}
        test_sizes_here = {len(p["input"]) for p in puzzle["test"]}
        source_shift += int(train_sizes_here == {3, 4} and test_sizes_here == {5})
        if len(puzzle["test"]) == 4:
            four_tests += 1
            test_features = []
            for pair in puzzle["test"]:
                active = pair["input"] == trigger
                test_features.append((int(active.sum()), _component_count(active)))
            assert len({feature[0] for feature in test_features}) >= 3
            assert len({feature[1] for feature in test_features}) >= 2
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                n = len(grid)
                active = grid == trigger
                inputs.add(_grid_key(grid))
                masks.add((n, active.tobytes()))
                trigger_counts.add((n, int(active.sum())))
                palettes.add(len(np.unique(grid)))
                components.add(_component_count(active))
                blank_presence.add(bool(np.any(grid == blank)))
                tiled = np.tile(grid, (n, n))
                block_mask = np.repeat(np.repeat(active, n, axis=0), n, axis=1)
                independent_expected = np.where(block_mask, tiled, blank)
                assert np.array_equal(output, independent_expected)
                assert not _is_source_pair(grid, output)
                if split == "train":
                    train_sizes.add(n)
                    minority_train += int(active.sum() < n * n / 2)
                    train_blank_gaps += int(np.any(grid == blank) and len(np.unique(grid)) >= 4)
                else:
                    test_sizes.add(n)
                    sparse_test += int(active.sum() <= 2)
                    dense_test += int((~active).sum() <= 2)
                    inactive_center += int(not active[n // 2, n // 2])
                    corner_test += int(active[0, 0] or active[0, -1] or active[-1, 0] or active[-1, -1])
                    offcenter_test += int(active.sum() == 1 and not active[n // 2, n // 2])
                    mixed_test += int(_component_count(active) > 1 and active.sum() > _component_count(active))
                    test_blank_gaps += int(np.any(grid == blank) and len(np.unique(grid)) >= 4)
                    test_lines += int(any(active.all(axis=0)) or any(active.all(axis=1)))
                    test_borders += int(active[0, :].all() and active[-1, :].all()
                                        and active[:, 0].all() and active[:, -1].all())
                    source_cross += int(n == 5 and active.sum() == 5 and
                                        all(active[row, col] for row, col in
                                            ((2, 2), (0, 2), (4, 2), (2, 0), (2, 4))))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) >= 40 and len(inputs) >= 180 and len(masks) >= 90
    assert len(role_pairs) >= 12 and len(trigger_counts) >= 20
    assert train_sizes == {2, 3, 4, 5} and test_sizes == {3, 4, 5}
    assert min(palettes) == 2 and max(palettes) >= 6
    assert len(components) >= 3 and blank_presence == {False, True}
    assert minority_train >= 42 and sparse_test >= 8 and dense_test >= 8
    assert inactive_center >= 15 and corner_test >= 15
    assert source_shift >= 3 and source_cross >= 2 and four_tests >= 5
    assert offcenter_test >= 3 and mixed_test >= 8
    assert train_blank_gaps >= 10 and test_blank_gaps >= 5
    assert test_lines >= 8 and test_borders >= 2
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1004, None, 1), (1005, None, 4),
                              (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _signature(generate_puzzle(314159, 7, 3)) == _signature(generate_puzzle(314159, 7, 3))
    assert _signature(generate_puzzle(41)) != _signature(generate_puzzle(42))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 2.0}, {"num_test": True}):
        try:
            generate_puzzle(900, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
