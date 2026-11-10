"""Synthetic framed-tile motif propagation puzzles."""

import numpy as np


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _counts(seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer from {lo} to {hi}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _source_style(rng):
    # A construction regime is useful for keeping some original-sized examples;
    # it has no hidden meaning for the solver.
    stable_palette = bool(rng.random() < 0.18)
    if stable_palette:
        background = int(rng.integers(0, 10))
        colors = tuple(int(c) for c in rng.choice([c for c in range(10) if c != background], size=int(rng.integers(2, 4)), replace=False))
        anchor_color = colors[0]
    else:
        background, colors = None, None
        anchor_color = int(rng.integers(0, 10))
    return {"expansion": bool(rng.random() < 0.24), "anchor_color": anchor_color,
            "stable_background": background, "stable_colors": colors}


def _geometry(rng, regime, pair_index, conventions):
    if conventions["expansion"] and pair_index == 0:
        rows, cols, ih, iw, gap, pad = ((4, 3, 3, 3, 1, 1) if regime == "source_train" else (4, 4, 3, 3, 1, 1))
        return rows, cols, ih, iw, gap, pad
    if regime == "source_train" and pair_index == 0:
        ih = iw = 3  # Give every puzzle an orientation-sensitive teaching example.
    else:
        choices = [(3, 3)] * 4 + [(2, 2), (4, 4), (5, 5), (2, 3), (3, 2), (3, 4), (4, 3), (2, 5), (5, 2)]
        ih, iw = choices[int(rng.integers(0, len(choices)))]
    for _ in range(200):
        gap = int(rng.integers(1, 3))
        pad = int(rng.integers(1, 3))
        if regime == "source_train" and pair_index == 0:
            rows = int(rng.integers(2, 5))
            cols = int(rng.integers(2, 5))
        else:
            rows = int(rng.integers(1, 6))
            cols = int(rng.integers(1, 6))
        if rows * cols < 2:
            continue
        h = 2 * pad + rows * (ih + 2) + (rows - 1) * gap
        w = 2 * pad + cols * (iw + 2) + (cols - 1) * gap
        if h <= 30 and w <= 30:
            return rows, cols, ih, iw, gap, pad
    # Large interiors and padding still permit a compact pair.
    return 2, 2, ih, iw, 1, 1


def _motif(rng, height, width, asymmetric=False):
    if asymmetric:
        # A corner plus an off-axis neighbor distinguishes every fixed
        # reflection/rotation of the original 3-by-3 teaching motif.
        mask = np.zeros((height, width), dtype=bool)
        mask[0, 0] = mask[0, 1] = mask[1, 0] = True
        if width > 2:
            mask[1, 2] = True
        return mask
    mask = np.zeros((height, width), dtype=bool)
    mode = int(rng.integers(0, 12))
    if mode == 0:  # central single cell
        mask[height // 2, width // 2] = True
    elif mode == 1:  # any single cell
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    elif mode == 2:
        mask[int(rng.integers(height)), :] = True
    elif mode == 3:
        mask[:, int(rng.integers(width))] = True
    elif mode == 4:
        mask[height // 2, :] = True
        mask[:, width // 2] = True
    elif mode == 5:
        mask[:] = True
    elif mode == 6:
        mask[:] = True
        mask[int(rng.integers(height)), int(rng.integers(width))] = False
    elif mode == 7:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    else:
        n = height * width
        occupied = int(rng.integers(1, n + 1))
        mask.ravel()[rng.choice(n, size=occupied, replace=False)] = True
    return mask


def _group_counts(rng, tile_count, force_active, many_active=False, fixed_group_count=None):
    minimum_repeated = 4 if many_active else int(force_active)
    upper = min(tile_count - minimum_repeated, 9)
    if upper < 1:
        raise ValueError("too few tiles")
    group_count = int(fixed_group_count) if fixed_group_count is not None else int(rng.integers(4 if many_active else 1, upper + 1))
    assert 1 <= group_count <= upper
    if fixed_group_count is None and force_active and not many_active and tile_count >= 4 and rng.random() < 0.75:
        group_count = max(2, group_count)
    counts = np.ones(group_count, dtype=int)
    counts[:minimum_repeated] += 1
    remaining = tile_count - int(counts.sum())
    concentration = bool(rng.random() < 0.28)
    for _ in range(remaining):
        index = 0 if concentration and rng.random() < 0.7 else int(rng.integers(group_count))
        counts[index] += 1
    return counts


def _render(rows, cols, ih, iw, gap, pad, background, colors, motifs):
    th, tw = ih + 2, iw + 2
    grid = np.full((2 * pad + rows * th + (rows - 1) * gap, 2 * pad + cols * tw + (cols - 1) * gap), background, dtype=np.int8)
    for index, (color, motif) in enumerate(zip(colors, motifs)):
        r, c = divmod(index, cols)
        top, left = pad + r * (th + gap), pad + c * (tw + gap)
        tile = grid[top:top + th, left:left + tw]
        tile[0, :] = tile[-1, :] = color
        tile[:, 0] = tile[:, -1] = color
        tile[1:-1, 1:-1][motif] = color
    return grid


def _make_pair(rng, regime, pair_index, conventions, previous_anchor_mask):
    anchor_demonstration = regime == "source_train" and pair_index < 2
    make_unchanged = regime == "target_test" and rng.random() < 0.13
    require_active = not make_unchanged
    fixed_colors = conventions["stable_colors"]
    for _ in range(100):
        rows, cols, ih, iw, gap, pad = _geometry(rng, regime, pair_index, conventions)
        count = rows * cols
        if fixed_colors is None or count >= len(fixed_colors) + int(require_active):
            break
    else:
        raise AssertionError("no layout fits the stable palette")
    many_active = regime == "target_test" and require_active and fixed_colors is None and count >= 8 and rng.random() < 0.25
    group_counts = _group_counts(rng, count, require_active, many_active,
                                 None if fixed_colors is None else len(fixed_colors))
    background = int(conventions["stable_background"] if fixed_colors is not None else rng.integers(0, 10))
    if fixed_colors is None and anchor_demonstration and background == conventions["anchor_color"]:
        background = (background + 1) % 10
    available = [c for c in range(10) if c != background]
    if fixed_colors is not None:
        palette = ([conventions["anchor_color"]] + [int(c) for c in rng.permutation([c for c in fixed_colors if c != conventions["anchor_color"]])]
                   if anchor_demonstration else [int(c) for c in rng.permutation(fixed_colors)])
    elif anchor_demonstration:
        anchor = conventions["anchor_color"]
        palette = [anchor] + [int(c) for c in rng.choice([c for c in available if c != anchor], size=len(group_counts) - 1, replace=False)]
    else:
        palette = [int(c) for c in rng.choice(available, size=len(group_counts), replace=False)]
    tile_colors = np.repeat(np.asarray(palette, dtype=np.int8), group_counts)
    rng.shuffle(tile_colors)
    motifs = [np.zeros((ih, iw), dtype=bool) for _ in range(count)]
    prior_motifs = []
    anchor_mask = None
    for group_index, (color, occurrences) in enumerate(zip(palette, group_counts)):
        indices = np.flatnonzero(tile_colors == color)
        if occurrences == 1:
            donor_count = int(rng.integers(0, 2))
        elif require_active and (group_index == 0 or (many_active and group_index < 4)):
            donor_count = int(rng.integers(1, occurrences))
        elif make_unchanged:
            donor_count = int(rng.choice([0, occurrences]))
        else:
            category = int(rng.choice(3, p=[0.57, 0.22, 0.21]))
            donor_count = (int(rng.integers(1, occurrences)) if category == 0 else occurrences if category == 1 else 0)
        if donor_count == 0:
            continue
        if anchor_demonstration and group_index == 0 and pair_index == 0:
            motif = _motif(rng, ih, iw, asymmetric=True)
        elif anchor_demonstration and group_index == 0 and previous_anchor_mask is not None and previous_anchor_mask.shape == (ih, iw):
            motif = _motif(rng, ih, iw)
            for _ in range(20):
                if not np.array_equal(motif, previous_anchor_mask):
                    break
                motif = _motif(rng, ih, iw)
        elif prior_motifs and rng.random() < 0.25:
            motif = prior_motifs[int(rng.integers(len(prior_motifs)))].copy()
        else:
            motif = _motif(rng, ih, iw)
        prior_motifs.append(motif)
        if anchor_demonstration and group_index == 0:
            anchor_mask = motif.copy()
        for index in np.atleast_1d(rng.choice(indices, size=donor_count, replace=False)):
            motifs[int(index)] = motif.copy()
    inp = _render(rows, cols, ih, iw, gap, pad, background, tile_colors, motifs)
    out = _solve_grid(inp)
    return {"input": inp, "output": out}, anchor_mask


def _segments(flags):
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False]) + 1
    return list(zip((int(x) for x in starts), (int(x) for x in ends)))


def _analyze_grid(grid):
    background = int(grid[0, 0])
    row_runs = _segments(np.any(grid != background, axis=1))
    col_runs = _segments(np.any(grid != background, axis=0))
    assert row_runs and col_runs
    assert row_runs[0][0] >= 1 and col_runs[0][0] >= 1
    assert row_runs[-1][1] < grid.shape[0] and col_runs[-1][1] < grid.shape[1]
    assert all(b < c for (_, b), (c, _) in zip(row_runs, row_runs[1:]))
    assert all(b < c for (_, b), (c, _) in zip(col_runs, col_runs[1:]))
    heights = {b - a for a, b in row_runs}
    widths = {b - a for a, b in col_runs}
    assert len(heights) == len(widths) == 1
    assert min(heights) >= 4 and min(widths) >= 4
    colors, masks, boxes = [], [], []
    for top, bottom in row_runs:
        for left, right in col_runs:
            tile = grid[top:bottom, left:right]
            color = int(tile[0, 0])
            assert color != background
            assert np.all(tile[0, :] == color) and np.all(tile[-1, :] == color)
            assert np.all(tile[:, 0] == color) and np.all(tile[:, -1] == color)
            interior = tile[1:-1, 1:-1]
            assert np.all((interior == background) | (interior == color))
            colors.append(color)
            masks.append((interior == color).copy())
            boxes.append((top, bottom, left, right))
    return background, colors, masks, boxes, (len(row_runs), len(col_runs)), (next(iter(heights)) - 2, next(iter(widths)) - 2)


def _solve_grid(input_grid):
    _, colors, masks, boxes, _, _ = _analyze_grid(input_grid)
    templates = {}
    for color, mask in zip(colors, masks):
        if np.any(mask):
            if color in templates:
                assert np.array_equal(mask, templates[color])
            else:
                templates[color] = mask
    output = input_grid.copy()
    for color, mask, (top, bottom, left, right) in zip(colors, masks, boxes):
        if color in templates and not np.any(mask):
            output[top + 1:bottom - 1, left + 1:right - 1][templates[color]] = color
    return output


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    active_training = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape and inp.dtype == out.dtype
            assert np.array_equal(out, _solve_grid(inp))
            background, colors, masks, boxes, layout, interior = _analyze_grid(inp)
            ob, oc, om, oboxes, olayout, ointerior = _analyze_grid(out)
            assert (background, colors, boxes, layout, interior) == (ob, oc, oboxes, olayout, ointerior)
            assert len(masks) == len(om)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            if split == "train" and not np.array_equal(inp, out):
                active_training += 1
    assert active_training >= 2


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    conventions = _source_style(rng)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_anchor = None
    for split, amount, regime in (("train", train_count, "source_train"), ("test", test_count, "target_test")):
        for index in range(amount):
            for _ in range(200):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair, anchor = _make_pair(pair_rng, regime, index, conventions, first_anchor)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_anchor = anchor
                    break
            else:
                raise AssertionError("could not generate unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    signatures, layouts, geometries, backgrounds, motif_sizes = set(), set(), set(), set(), set()
    one_axis = rectangular = unchanged = symmetric_active = multi_donor = rich_test = stable_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_grid_key(p["input"]) for split in ("train", "test") for p in puzzle[split]))
        pair_palettes = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, colors, masks, boxes, layout, geometry = _analyze_grid(pair["input"])
                pair_palettes.append((bg, tuple(sorted(set(colors)))))
                backgrounds.add(bg)
                layouts.add(layout)
                geometries.add(geometry)
                motif_sizes.update(int(mask.sum()) for mask in masks if np.any(mask))
                one_axis += int(1 in layout)
                rectangular += int(geometry[0] != geometry[1])
                unchanged += int(np.array_equal(pair["input"], pair["output"]))
                active_here = 0
                for color in set(colors):
                    candidates = [mask for c, mask in zip(colors, masks) if c == color and np.any(mask)]
                    blank_count = sum(c == color and not np.any(mask) for c, mask in zip(colors, masks))
                    if candidates and blank_count:
                        active_here += 1
                        multi_donor += int(len(candidates) > 1)
                        template = candidates[0]
                        symmetric_active += int(np.array_equal(template, np.flipud(template)) or
                                                np.array_equal(template, np.fliplr(template)) or
                                                np.array_equal(template, np.rot90(template, 2)))
                    target = candidates[0] if candidates else None
                    for c, mask, box in zip(colors, masks, boxes):
                        if c != color:
                            continue
                        top, bottom, left, right = box
                        actual = pair["output"][top + 1:bottom - 1, left + 1:right - 1] == color
                        assert np.array_equal(actual, target if target is not None else mask)
                if split == "test" and active_here >= 4:
                    rich_test += 1
        stable_palettes += int(len(set(pair_palettes)) == 1)
    assert len(signatures) == 42
    assert len(layouts) >= 10 and len(geometries) >= 6
    assert len(backgrounds) >= 8
    assert {1, 8, 9} <= motif_sizes
    assert min(one_axis, rectangular, unchanged, symmetric_active, multi_donor, rich_test, stable_palettes) > 0
    assert all(np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
               for split in ("train", "test") for a, b in zip(generate_puzzle(12345)[split], generate_puzzle(12345)[split]))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 4), (2, None)):
        puzzle = generate_puzzle(875, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(20, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts accepted")
