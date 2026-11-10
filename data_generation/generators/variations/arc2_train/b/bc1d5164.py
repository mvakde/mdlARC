"""Transparent union of four panels separated by a one-cell cross."""

import numpy as np


_SOURCE_PAIR_TEXT = frozenset({
    (("4000004", "0000000", "0000000", "0000000", "4000044"),
     ("404", "000", "444")),
    (("2200022", "0000002", "0000000", "0200020", "2000002"),
     ("222", "022", "202")),
    (("4400040", "0000044", "0000000", "0000000", "4000004"),
     ("440", "044", "404")),
    (("0300030", "3000003", "0000000", "0000000", "0000003"),
     ("030", "303", "003")),
    (("0800080", "8800088", "0000000", "8800088", "0800080"),
     ("080", "888", "080")),
    (("0000011", "1000000", "0000000", "0000000", "0100001"),
     ("011", "100", "011")),
})


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _parameters(rng, num_train):
    # Orientation and dimensions are learned once from this puzzle's training set.
    if rng.random() < 0.36:
        h, w = 2, 3
    else:
        h = int(rng.integers(2, 7))
        w = int(rng.integers(2, 8))
    rotated = bool(rng.random() < 0.22)

    # The separator, rather than a fixed color number, identifies background.
    ink_size = 1 if rng.random() < 0.18 else int(rng.integers(3, 6))
    ink = tuple(int(x) for x in rng.choice(10, size=ink_size, replace=False))
    training_ink_count = 1 if ink_size == 1 else int(rng.integers(2, min(4, ink_size)))
    training_inks = ink[:training_ink_count]
    novel_inks = ink[training_ink_count:]
    available = [x for x in range(10) if x not in ink]
    bg_size = int(rng.integers(1, min(4, len(available)) + 1))
    background = []
    if 0 in available and rng.random() < 0.62:
        background.append(0)
        available.remove(0)
    remaining = bg_size - len(background)
    if remaining:
        background.extend(int(x) for x in rng.choice(available, size=remaining, replace=False))
    return h, w, rotated, training_inks, novel_inks, tuple(background)


def _eligible(row, h):
    if row < h - 1:
        return (0, 1)
    if row == h - 1:
        return (0, 1, 2, 3)
    return (2, 3)


def _put(layers, panel, row, col, h):
    local_row = row if panel < 2 else row - h + 1
    layers[panel, local_row, col] = True


def _mask(rng, h, w, panels):
    height = 2 * h - 1
    allowed = np.zeros((height, w), dtype=bool)
    if any(p < 2 for p in panels):
        allowed[:h] = True
    if any(p >= 2 for p in panels):
        allowed[h - 1:] = True
    points = np.argwhere(allowed)
    category = int(rng.integers(0, 10))
    mask = np.zeros_like(allowed)

    if category == 0:  # moderate random
        mask = allowed & (rng.random(allowed.shape) < rng.uniform(0.20, 0.68))
    elif category == 1:  # one or two foreground cells
        chosen = rng.choice(len(points), size=min(len(points), int(rng.integers(1, 3))), replace=False)
        for index in np.atleast_1d(chosen):
            r, c = points[int(index)]
            mask[r, c] = True
    elif category == 2:  # dense, with many possible panel allocations
        mask = allowed & (rng.random(allowed.shape) < rng.uniform(0.62, 0.92))
    elif category == 3:  # omit a boundary or shared row
        blank = int(rng.choice((0, h - 1, height - 1)))
        allowed[blank] = False
        mask = allowed & (rng.random(allowed.shape) < rng.uniform(0.30, 0.80))
    elif category == 4:  # one or two output columns
        cols = rng.choice(w, size=min(w, int(rng.integers(1, 3))), replace=False)
        mask[:, cols] = allowed[:, cols] & (rng.random((height, len(cols))) < 0.78)
    elif category == 5:  # compact rectangle
        r0 = int(rng.integers(0, height))
        c0 = int(rng.integers(0, w))
        r1 = min(height, r0 + int(rng.integers(1, max(3, h + 1))))
        c1 = min(w, c0 + int(rng.integers(1, max(3, w))))
        mask[r0:r1, c0:c1] = allowed[r0:r1, c0:c1]
    elif category == 6:  # line
        if rng.random() < 0.5:
            row = int(rng.choice(np.flatnonzero(np.any(allowed, axis=1))))
            mask[row] = allowed[row]
        else:
            col = int(rng.integers(0, w))
            mask[:, col] = allowed[:, col]
    elif category == 7:  # intersecting lines
        row = int(rng.choice(np.flatnonzero(np.any(allowed, axis=1))))
        col = int(rng.integers(0, w))
        mask[row] = allowed[row]
        mask[:, col] = allowed[:, col]
    elif category == 8:  # separated marks
        first = points[int(rng.integers(0, len(points)))]
        distances = np.abs(points - first).sum(axis=1)
        farthest = points[int(np.argmax(distances))]
        mask[tuple(first)] = True
        mask[tuple(farthest)] = True
        if len(points) > 3 and rng.random() < 0.35:
            third = points[int(rng.integers(0, len(points)))]
            mask[tuple(third)] = True
    else:  # complete permitted region, including full output when both strips exist
        mask = allowed.copy()

    if not np.any(mask):
        r, c = points[int(rng.integers(0, len(points)))]
        mask[r, c] = True
    return mask


def _layers_for_mask(rng, mask, h, w, panels, multiplicity):
    layers = np.zeros((4, h, w), dtype=bool)
    positions = np.argwhere(mask)
    forced = None
    if multiplicity > 1:
        possibilities = [
            (int(r), int(c))
            for r, c in positions
            if len(set(_eligible(int(r), h)) & set(panels)) >= multiplicity
        ]
        if not possibilities:
            col = int(rng.integers(0, w))
            mask[h - 1, col] = True
            positions = np.argwhere(mask)
            possibilities = [(h - 1, col)]
        forced = possibilities[int(rng.integers(0, len(possibilities)))]

    for r, c in positions:
        r, c = int(r), int(c)
        choices = [p for p in panels if p in _eligible(r, h)]
        if (r, c) == forced:
            selected = rng.choice(choices, size=multiplicity, replace=False)
        else:
            count = 1
            while count < min(multiplicity, len(choices)) and rng.random() < 0.22:
                count += 1
            selected = rng.choice(choices, size=count, replace=False)
        for panel in np.atleast_1d(selected):
            _put(layers, int(panel), r, c, h)

    # Use selected panels when possible without changing the requested overlap class.
    for panel in panels:
        if np.any(layers[panel]):
            continue
        choices = []
        for r, c in positions:
            r, c = int(r), int(c)
            if panel in _eligible(r, h):
                current = sum(
                    bool(layers[p, r if p < 2 else r - h + 1, c])
                    for p in _eligible(r, h)
                )
                if current < multiplicity:
                    choices.append((r, c))
        if choices:
            r, c = choices[int(rng.integers(0, len(choices)))]
            _put(layers, panel, r, c, h)
    return layers


def _anchor(rng, h, w):
    """One training pair demonstrates all panels and shared-row union."""
    layers = np.zeros((4, h, w), dtype=bool)
    mask = np.zeros((2 * h - 1, w), dtype=bool)
    top = [(r, c) for r in range(h - 1) for c in range(w)]
    bottom = [(r, c) for r in range(h, 2 * h - 1) for c in range(w)]
    top_indices = rng.choice(len(top), size=2, replace=False)
    bottom_indices = rng.choice(len(bottom), size=2, replace=False)
    protected = set()
    for panel, position in enumerate((
        top[int(top_indices[0])], top[int(top_indices[1])],
        bottom[int(bottom_indices[0])], bottom[int(bottom_indices[1])],
    )):
        r, c = position
        protected.add(position)
        mask[r, c] = True
        _put(layers, panel, r, c, h)

    shared_col = int(rng.integers(0, w))
    shared = (h - 1, shared_col)
    mask[shared] = True
    for panel in rng.choice(4, size=int(rng.integers(2, 5)), replace=False):
        _put(layers, int(panel), *shared, h)

    # Optional additional cells make the anchor structurally variable.
    for r in range(2 * h - 1):
        for c in range(w):
            if (r, c) in protected or (r, c) == shared:
                continue
            if rng.random() < rng.uniform(0.12, 0.38):
                mask[r, c] = True
                choices = _eligible(r, h)
                count = 2 if rng.random() < 0.25 else 1
                for panel in rng.choice(choices, size=count, replace=False):
                    _put(layers, int(panel), r, c, h)
    return mask, layers


def _sample_pair(rng, h, w, rotated, ink, background, anchor=False, reused_mask=None):
    if anchor:
        mask, layers = _anchor(rng, h, w)
    else:
        multiplicity = int(rng.choice((1, 2, 3, 4), p=(0.34, 0.31, 0.21, 0.14)))
        count = int(rng.integers(multiplicity, 5))
        panels = tuple(int(p) for p in rng.choice(4, size=count, replace=False))
        if reused_mask is None:
            mask = _mask(rng, h, w, panels)
        else:
            mask = reused_mask.copy()
            # The reused anchor spans upper and lower nonshared rows.
            if not (any(p < 2 for p in panels) and any(p >= 2 for p in panels)):
                panels = (0, 1, 2, 3)
        layers = _layers_for_mask(rng, mask, h, w, panels, multiplicity)

    input_grid = np.full((2 * h + 1, 2 * w + 1), background, dtype=np.int8)
    for panel in range(4):
        rows = slice(0, h) if panel < 2 else slice(h + 1, 2 * h + 1)
        cols = slice(0, w) if panel % 2 == 0 else slice(w + 1, 2 * w + 1)
        input_grid[rows, cols] = np.where(layers[panel], ink, background)
    output_grid = np.where(mask, ink, background).astype(np.int8)
    if rotated:
        input_grid = input_grid.T.copy()
        output_grid = output_grid.T.copy()
    return input_grid, output_grid


def _canonical(input_grid, rotated):
    return input_grid.T if rotated else input_grid


def _contributions(input_grid, rotated):
    grid = _canonical(input_grid, rotated)
    h, w = (grid.shape[0] - 1) // 2, (grid.shape[1] - 1) // 2
    bg = int(grid[h, w])
    layers = (
        grid[:h, :w] != bg,
        grid[:h, w + 1:] != bg,
        grid[h + 1:, :w] != bg,
        grid[h + 1:, w + 1:] != bg,
    )
    counts = np.zeros((2 * h - 1, w), dtype=np.int8)
    for panel, layer in enumerate(layers):
        rows = slice(0, h) if panel < 2 else slice(h - 1, 2 * h - 1)
        counts[rows] += layer.astype(np.int8)
    return layers, counts, bg


def _reconstruct(input_grid, rotated):
    layers, counts, bg = _contributions(input_grid, rotated)
    other = np.unique(input_grid[input_grid != bg])
    assert len(other) == 1
    result = np.where(counts > 0, int(other[0]), bg).astype(np.int8)
    return result.T.copy() if rotated else result


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _is_source_pair(input_grid, output_grid):
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(input_grid), rows(output_grid)) in _SOURCE_PAIR_TEXT


def _anchor_evidence(input_grid, rotated):
    layers, counts, _ = _contributions(input_grid, rotated)
    h = layers[0].shape[0]
    for panel, layer in enumerate(layers):
        output_rows = slice(0, h - 1) if panel < 2 else slice(h, 2 * h - 1)
        local_rows = slice(0, h - 1) if panel < 2 else slice(1, h)
        if not np.any(layer[local_rows] & (counts[output_rows] == 1)):
            return False
    return bool(np.any(counts[h - 1] >= 2))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    geometry = None
    anchor_found = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(inp, out)

            options = []
            for rotated in (False, True):
                canonical = _canonical(inp, rotated)
                rows, cols = canonical.shape
                if rows >= 5 and cols >= 5 and rows % 2 == 1 and cols % 2 == 1:
                    h, w = (rows - 1) // 2, (cols - 1) // 2
                    expected = (2 * h - 1, w)
                    if rotated:
                        expected = expected[::-1]
                    if out.shape == expected:
                        options.append((rotated, h, w))
            assert len(options) == 1, "panel geometry must be identifiable"
            this_geometry = options[0]
            if geometry is None:
                geometry = this_geometry
            assert this_geometry == geometry, "geometry must be stable within a puzzle"
            rotated, h, w = this_geometry
            canonical = _canonical(inp, rotated)
            bg = int(canonical[h, w])
            assert np.all(canonical[h] == bg)
            assert np.all(canonical[:, w] == bg)
            colors = set(int(x) for x in np.unique(inp))
            assert len(colors) == 2 and bg in colors
            ink = next(x for x in colors if x != bg)
            assert set(int(x) for x in np.unique(out)) <= {bg, ink}
            assert np.any(out == ink)
            assert np.array_equal(out, _reconstruct(inp, rotated)), "incorrect transparent union"
            if split == "train":
                anchor_found |= _anchor_evidence(inp, rotated)
    assert anchor_found, "training must demonstrate every panel and shared-row overlap"


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
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    h, w, rotated, training_inks, novel_inks, backgrounds = _parameters(rng, num_train)
    anchor_index = int(rng.integers(0, num_train - 1 if num_train >= 4 else num_train))
    repeat_index = None
    if num_train >= 4 and rng.random() < 0.48:
        choices = list(range(anchor_index + 1, num_train))
        repeat_index = int(rng.choice(choices))

    colors = []
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            palette = (
                novel_inks
                if split == "test" and novel_inks and rng.random() < 0.45
                else training_inks
            )
            ink = int(rng.choice(palette))
            bg = int(rng.choice(backgrounds))
            colors.append((ink, bg))
    if len(training_inks) > 1:
        colors[0] = (training_inks[0], colors[0][1])
        colors[1] = (training_inks[1], colors[1][1])
    if len(backgrounds) > 1:
        colors[0] = (colors[0][0], backgrounds[0])
        colors[1] = (colors[1][0], backgrounds[1])

    puzzle = {"train": [], "test": []}
    seen = set()
    anchor_mask = None
    ordinal = 0
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            ink, bg = colors[ordinal]
            is_anchor = split == "train" and index == anchor_index
            for _ in range(200):
                if is_anchor:
                    # Capture the output mask after constructing the anchor.
                    inp, out = _sample_pair(rng, h, w, rotated, ink, bg, anchor=True)
                    anchor_mask = (out.T if rotated else out) == ink
                elif split == "train" and index == repeat_index and anchor_mask is not None:
                    inp, out = _sample_pair(rng, h, w, rotated, ink, bg, reused_mask=anchor_mask)
                else:
                    inp, out = _sample_pair(rng, h, w, rotated, ink, bg)
                key = _grid_key(inp)
                if key not in seen and not _is_source_pair(inp, out):
                    break
            else:
                raise RuntimeError("could not construct a distinct non-source pair")
            seen.add(key)
            puzzle[split].append({"input": inp, "output": out})
            ordinal += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(
        len(a[split]) == len(b[split]) and all(
            np.array_equal(x[role], y[role])
            for x, y in zip(a[split], b[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes = set()
    orientations = set()
    test_multiplicities = set()
    test_panel_counts = set()
    test_densities = set()
    blank_rows = set()
    backgrounds = set()
    zero_as_ink = False
    reused_test_ink = False
    varying_background = False
    fixed_ink_puzzles = 0
    structures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_inks = set()
        puzzle_bgs = set()
        puzzle_inks = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                rotated = next(
                    orientation for orientation in (False, True)
                    if out.shape == (
                        (((inp.T if orientation else inp).shape[0] - 2),
                         ((inp.T if orientation else inp).shape[1] - 1) // 2)[::-1 if orientation else 1]
                    )
                )
                layers, counts, bg = _contributions(inp, rotated)
                ink = next(int(x) for x in np.unique(inp) if x != bg)
                puzzle_bgs.add(bg)
                puzzle_inks.add(ink)
                shapes.add(tuple(inp.shape))
                orientations.add(rotated)
                backgrounds.add(bg)
                zero_as_ink |= ink == 0
                structures.add((tuple(inp.shape), tuple((inp != bg).flat)))
                if split == "train":
                    train_inks.add(ink)
                else:
                    reused_test_ink |= ink in train_inks
                    test_multiplicities.add(int(counts.max()))
                    test_panel_counts.add(sum(bool(np.any(layer)) for layer in layers))
                    test_densities.add(int(np.count_nonzero(out != bg)))
                    for row in (0, counts.shape[0] // 2, counts.shape[0] - 1):
                        if not np.any(counts[row]):
                            blank_rows.add(("top" if row == 0 else "bottom" if row == counts.shape[0] - 1 else "middle"))
        varying_background |= len(puzzle_bgs) > 1
        fixed_ink_puzzles += len(puzzle_inks) == 1
    assert len(shapes) >= 12 and orientations == {False, True}
    assert test_multiplicities == {1, 2, 3, 4}
    assert test_panel_counts == {1, 2, 3, 4}
    assert {1, 2, 9} <= test_densities
    assert blank_rows == {"top", "middle", "bottom"}
    assert len(backgrounds) >= 5 and zero_as_ink
    assert reused_test_ink and varying_background and fixed_ink_puzzles >= 2
    assert len(structures) >= 120
    assert _puzzles_equal(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4), (103, 6, 3),
                              (200, None, 1), (201, None, 4), (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(300, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
