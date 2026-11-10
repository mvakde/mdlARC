"""Cellwise lookup-matrix recoloring of a regular stencil lattice."""

import hashlib

import numpy as np


_PLACEMENTS = ("left", "right", "above", "below")
_MASK_SIZES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4),
               (4, 2), (3, 4), (4, 3), (4, 4), (1, 3), (3, 1))
_MASK_WEIGHTS = (.12, .10, .10, .20, .11, .11, .07, .07, .05, .035, .035)
_SOURCE_MASKS = (
    ((1, 1, 1), (0, 1, 0), (1, 1, 1)),
    ((1, 1, 1), (1, 0, 1), (1, 1, 1)),
    ((1, 1, 1), (1, 0, 1), (1, 0, 1)),
)
_SOURCE_DIGESTS = frozenset((
    "98b3e6d436638c9239e84439c6d5346dbc652ff17fdcc77d4",
    "f0286eb890dd896cbfb00defab19c24b48fbf685e34bc81b614170c47146fc4d",
    "a1f5cfd6b214b6be718447aa193eccb5f3642570d57608f28a4e9dc7bb863676",
))


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4 + 3 * seed + 1) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (3 * seed + seed // 7 + 1) % (10 - num_test)
    elif num_test is None:
        num_test = 1 + (5 * seed + seed // 11 + 2) % min(4, num_train - 1)
    return num_train, num_test


def _connected(mask):
    occupied = set(zip(*np.where(mask)))
    if not occupied:
        return False
    found = {next(iter(occupied))}
    frontier = list(found)
    while frontier:
        y, x = frontier.pop()
        for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if neighbor in occupied and neighbor not in found:
                found.add(neighbor)
                frontier.append(neighbor)
    return found == occupied


def _sample_mask(rng, size):
    height, width = size
    mode = int(rng.choice(6, p=(.18, .08, .12, .24, .28, .10)))
    if mode == 0 and size == (3, 3):
        return np.asarray(_SOURCE_MASKS[int(rng.integers(0, 3))], dtype=bool)
    if mode == 1:
        return np.ones(size, dtype=bool)
    if mode == 2:
        mask = np.zeros(size, dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask
    if mode == 3:
        mask = np.ones(size, dtype=bool)
        target = int(rng.integers(max(height, width), height * width + 1))
        for index in rng.permutation(height * width):
            if int(mask.sum()) <= target:
                break
            y, x = divmod(int(index), width)
            mask[y, x] = False
            if (not mask.any(axis=0).all() or not mask.any(axis=1).all()
                    or not _connected(mask)):
                mask[y, x] = True
        return mask
    if mode == 5:
        mask = np.zeros(size, dtype=bool)
        mask[int(rng.integers(height)), :] = True
        mask[:, int(rng.integers(width))] = True
        return mask
    density = float(rng.uniform(.30, .76))
    mask = rng.random(size) < density
    for y in range(height):
        if not mask[y].any():
            mask[y, int(rng.integers(width))] = True
    for x in range(width):
        if not mask[:, x].any():
            mask[int(rng.integers(height)), x] = True
    return mask


def _sample_dimensions(rng, regime, style, anchor):
    if anchor:
        # Two-dimensional, unequal axes make ordering visible in training.
        return ((3, 4), (4, 3), (3, 5), (5, 3))[int(rng.integers(4))]
    u = float(rng.random())
    large_cut = .22 if regime == "source_train" else (.38 if style == 2 else .28)
    if u < .11:
        extent = int(rng.integers(3, 8))
        return (1, extent) if int(rng.integers(2)) else (extent, 1)
    if u < .35:
        return int(rng.integers(2, 4)), int(rng.integers(2, 5))
    if u < .35 + large_cut:
        return int(rng.integers(4, 8)), int(rng.integers(4, 8))
    return int(rng.integers(2, 6)), int(rng.integers(2, 6))


def _geometry(rng, rows, cols, mask_shape, placement):
    mh, mw = mask_shape
    gap_y = int(rng.integers(1, 5))
    gap_x = int(rng.integers(1, 5))
    separator = int(rng.integers(1, 9))
    lattice_h = rows * mh + (rows - 1) * gap_y
    lattice_w = cols * mw + (cols - 1) * gap_x
    if placement in ("left", "right"):
        offset = int(rng.integers(1 - rows, lattice_h))
        ly, ky = max(0, -offset), max(0, offset)
        if placement == "left":
            lx, kx = cols + separator, 0
        else:
            lx, kx = 0, lattice_w + separator
    else:
        offset = int(rng.integers(1 - cols, lattice_w))
        lx, kx = max(0, -offset), max(0, offset)
        if placement == "above":
            ly, ky = rows + separator, 0
        else:
            ly, ky = 0, lattice_h + separator
    content_h = max(ly + lattice_h, ky + rows)
    content_w = max(lx + lattice_w, kx + cols)
    if content_h > 30 or content_w > 30:
        return None
    extra_h = int(rng.integers(0, 31 - content_h))
    extra_w = int(rng.integers(0, 31 - content_w))
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    return ((content_h + extra_h, content_w + extra_w),
            (ly + top, lx + left), (ky + top, kx + left), gap_y, gap_x)


def _sample_lookup(rng, rows, cols, colors, anchor):
    if len(colors) == 1:
        mode = "uniform"
    elif anchor:
        mode = "random"
    else:
        mode = ("random", "row_repeat", "col_repeat", "stripes",
                "checker", "symmetric", "uniform")[int(rng.choice(
                    7, p=(.42, .12, .10, .11, .08, .12, .05)))]
    if mode == "uniform":
        return np.full((rows, cols), int(colors[0]), dtype=np.int8)
    if mode == "random":
        count = min(len(colors), rows * cols)
        indices = rng.integers(0, count, size=rows * cols)
        indices[:count] = rng.permutation(count)
        rng.shuffle(indices)
        lookup = np.asarray(colors, dtype=np.int8)[indices].reshape(rows, cols)
        if anchor and rows > 1 and cols > 1:
            for _ in range(32):
                if not (np.array_equal(lookup, np.flipud(lookup))
                        or np.array_equal(lookup, np.fliplr(lookup))):
                    break
                rng.shuffle(indices)
                lookup = np.asarray(colors, dtype=np.int8)[indices].reshape(rows, cols)
            if (np.array_equal(lookup, np.flipud(lookup))
                    or np.array_equal(lookup, np.fliplr(lookup))):
                lookup[0, 0] = colors[0]
                lookup[0, -1] = colors[1]
                lookup[-1, 0] = colors[1]
        return lookup
    if mode == "row_repeat":
        row = rng.choice(colors, size=cols)
        return np.tile(row, (rows, 1)).astype(np.int8)
    if mode == "col_repeat":
        column = rng.choice(colors, size=rows)
        return np.tile(column[:, None], (1, cols)).astype(np.int8)
    if mode == "stripes":
        if int(rng.integers(2)):
            sequence = rng.choice(colors, size=rows)
            return np.tile(sequence[:, None], (1, cols)).astype(np.int8)
        sequence = rng.choice(colors, size=cols)
        return np.tile(sequence, (rows, 1)).astype(np.int8)
    if mode == "checker":
        a, b = colors[:2]
        return np.fromfunction(lambda y, x: np.where((x + y) % 2, a, b),
                               (rows, cols), dtype=int).astype(np.int8)
    # Reflect one axis while allowing irregular content on the other.
    lookup = rng.choice(colors, size=(rows, cols)).astype(np.int8)
    if int(rng.integers(2)):
        for c in range(cols // 2):
            lookup[:, -1 - c] = lookup[:, c]
    else:
        for r in range(rows // 2):
            lookup[-1 - r, :] = lookup[r, :]
    return lookup


def _make_pair(rng, regime, style, anchor, placement, familiar_mask=None,
               previous_roles=None, fixed_roles=None, familiar_palette=None):
    for attempt in range(240):
        rows, cols = _sample_dimensions(rng, regime, style, anchor)
        if familiar_mask is not None and attempt < 100:
            mask = familiar_mask.copy()
        else:
            if regime == "target_test" and rows * cols >= 20 and float(rng.random()) < .75:
                size = ((2, 2), (2, 3), (3, 2), (1, 3), (3, 1))[
                    int(rng.integers(5))]
            else:
                size = _MASK_SIZES[int(rng.choice(len(_MASK_SIZES), p=_MASK_WEIGHTS))]
            mask = _sample_mask(rng, size)
        layout = _geometry(rng, rows, cols, mask.shape, placement)
        if layout is not None:
            (height, width) = layout[0]
            background_cells = height * width - rows * cols * (int(mask.sum()) + 1)
            if background_cells <= rows * cols * int(mask.sum()):
                continue
            break
    else:
        # A guaranteed feasible construction for every placement and count request.
        rows, cols = 3, 4
        mask = _sample_mask(rng, (2, 2))
        if placement == "left":
            layout = ((8, 16), (0, 5), (0, 0), 1, 1)
        elif placement == "right":
            layout = ((8, 16), (0, 0), (0, 12), 1, 1)
        elif placement == "above":
            layout = ((12, 11), (4, 0), (0, 0), 1, 1)
        else:
            layout = ((12, 11), (0, 0), (9, 0), 1, 1)
    (height, width), (ly, lx), (ky, kx), gap_y, gap_x = layout
    role_order = list(map(int, rng.permutation(10)))
    if fixed_roles is not None:
        background, carrier = fixed_roles
    else:
        if previous_roles is not None and role_order[0] == previous_roles[0]:
            role_order[0], role_order[2] = role_order[2], role_order[0]
        if previous_roles is not None and role_order[1] == previous_roles[1]:
            role_order[1], role_order[3] = role_order[3], role_order[1]
        background, carrier = role_order[:2]
    available = tuple(c for c in role_order if c not in (background, carrier))
    max_colors = min(8, rows * cols)
    if anchor:
        color_count = int(rng.integers(2, min(5, max_colors) + 1))
    else:
        color_count = int(rng.integers(2, max_colors + 1))
    colors = (tuple(familiar_palette) if familiar_palette is not None
              else available[:color_count])
    lookup = _sample_lookup(rng, rows, cols, colors, anchor)
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[ky:ky + rows, kx:kx + cols] = lookup
    output_grid = input_grid.copy()
    mh, mw = mask.shape
    for r in range(rows):
        for c in range(cols):
            y = ly + r * (mh + gap_y)
            x = lx + c * (mw + gap_x)
            input_grid[y:y + mh, x:x + mw][mask] = carrier
            output_grid[y:y + mh, x:x + mw][mask] = lookup[r, c]
    return {"input": input_grid, "output": output_grid}, mask, (background, carrier)


def _groups(values):
    assert values.size
    return list(np.split(values, np.flatnonzero(np.diff(values) > 1) + 1))


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert changed.any()
    changed_from = np.unique(input_grid[changed])
    assert changed_from.size == 1
    carrier = int(changed_from[0])
    counts = np.bincount(input_grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > max(counts[c] for c in range(10) if c != background)
    assert carrier != background
    assert np.array_equal(changed, input_grid == carrier)
    lookup_cells = (input_grid != background) & (input_grid != carrier)
    assert lookup_cells.any()
    ys, xs = np.where(lookup_cells)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    rows, cols = y1 - y0 + 1, x1 - x0 + 1
    assert lookup_cells.sum() == rows * cols
    assert lookup_cells[y0:y1 + 1, x0:x1 + 1].all()
    lookup = input_grid[y0:y1 + 1, x0:x1 + 1]
    assert np.array_equal(output_grid[y0:y1 + 1, x0:x1 + 1], lookup)
    carrier_cells = input_grid == carrier
    row_groups = _groups(np.flatnonzero(carrier_cells.any(axis=1)))
    col_groups = _groups(np.flatnonzero(carrier_cells.any(axis=0)))
    assert len(row_groups) == rows and len(col_groups) == cols
    mh, mw = len(row_groups[0]), len(col_groups[0])
    assert all(len(g) == mh for g in row_groups)
    assert all(len(g) == mw for g in col_groups)
    row_starts = [int(g[0]) for g in row_groups]
    col_starts = [int(g[0]) for g in col_groups]
    gy = [b - a - mh for a, b in zip(row_starts, row_starts[1:])]
    gx = [b - a - mw for a, b in zip(col_starts, col_starts[1:])]
    assert (not gy or 1 <= gy[0] <= 4 and len(set(gy)) == 1)
    assert (not gx or 1 <= gx[0] <= 4 and len(set(gx)) == 1)
    mask = carrier_cells[row_starts[0]:row_starts[0] + mh,
                         col_starts[0]:col_starts[0] + mw]
    assert mask.any(axis=0).all() and mask.any(axis=1).all()
    assert mask.sum() >= 2
    expected = input_grid.copy()
    for r, y in enumerate(row_starts):
        for c, x in enumerate(col_starts):
            slot = input_grid[y:y + mh, x:x + mw]
            assert np.array_equal(slot == carrier, mask)
            assert np.all((slot == carrier) | (slot == background))
            expected[y:y + mh, x:x + mw][mask] = lookup[r, c]
    assert np.array_equal(output_grid, expected)
    lattice_y0, lattice_y1 = row_starts[0], row_starts[-1] + mh - 1
    lattice_x0, lattice_x1 = col_starts[0], col_starts[-1] + mw - 1
    if x1 < lattice_x0 and max(y0, lattice_y0) <= min(y1, lattice_y1):
        placement, separation = "left", lattice_x0 - x1 - 1
    elif x0 > lattice_x1 and max(y0, lattice_y0) <= min(y1, lattice_y1):
        placement, separation = "right", x0 - lattice_x1 - 1
    elif y1 < lattice_y0 and max(x0, lattice_x0) <= min(x1, lattice_x1):
        placement, separation = "above", lattice_y0 - y1 - 1
    elif y0 > lattice_y1 and max(x0, lattice_x0) <= min(x1, lattice_x1):
        placement, separation = "below", y0 - lattice_y1 - 1
    else:
        raise AssertionError("lookup and lattice overlap or do not align")
    assert 1 <= separation <= 8
    mask_key = (mask.shape, mask.tobytes())
    return {"background": background, "carrier": carrier,
            "rows": rows, "cols": cols, "palette": frozenset(map(int, np.unique(lookup))),
            "lookup": lookup.copy(), "mask": mask.copy(), "mask_key": mask_key,
            "placement": placement, "separation": separation,
            "gap_y": gy[0] if gy else None, "gap_x": gx[0] if gx else None,
            "slot_count": rows * cols, "grid_shape": input_grid.shape,
            "lookup_bounds": (y0, y1, x0, x1),
            "lattice_bounds": (lattice_y0, lattice_y1, lattice_x0, lattice_x1)}


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _source_digest(pair):
    digest = hashlib.sha256()
    for name in ("input", "output"):
        grid = pair[name]
        digest.update(bytes(grid.shape))
        digest.update(grid.astype(np.int8, copy=False).tobytes())
    return digest.hexdigest()


def _predict_from_input(grid):
    """Recover the mapping using visible input roles and regular spacing."""
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    counts[background] = -1
    carrier = int(np.argmax(counts))
    other = (grid != background) & (grid != carrier)
    ys, xs = np.where(other)
    lookup = grid[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    carrier_y, carrier_x = np.where(grid == carrier)
    row_groups = _groups(np.unique(carrier_y))
    col_groups = _groups(np.unique(carrier_x))
    assert lookup.shape == (len(row_groups), len(col_groups))
    height, width = len(row_groups[0]), len(col_groups[0])
    y0, x0 = int(row_groups[0][0]), int(col_groups[0][0])
    mask = grid[y0:y0 + height, x0:x0 + width] == carrier
    answer = grid.copy()
    for row, row_group in enumerate(row_groups):
        for col, col_group in enumerate(col_groups):
            y, x = int(row_group[0]), int(col_group[0])
            answer[y:y + height, x:x + width][mask] = lookup[row, col]
    return answer


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            ik, pk = _grid_key(pair["input"]), _pair_key(pair)
            assert ik not in input_keys and pk not in pair_keys
            input_keys.add(ik)
            pair_keys.add(pk)
            assert _source_digest(pair) not in _SOURCE_DIGESTS
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    # The first two examples teach indexing on two axes and on more than one side.
    anchors = analyses["train"][:2]
    assert all(a["rows"] > 1 and a["cols"] > 1 and a["rows"] != a["cols"]
               for a in anchors)
    assert anchors[0]["mask_key"] != anchors[1]["mask_key"]
    assert anchors[0]["background"] != anchors[1]["background"]
    assert anchors[0]["carrier"] != anchors[1]["carrier"]
    for a in anchors:
        assert len(a["palette"]) >= 2
        assert not np.array_equal(a["lookup"], np.flipud(a["lookup"]))
        assert not np.array_equal(a["lookup"], np.fliplr(a["lookup"]))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    resolved_train, resolved_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The only learned convention is the fixed row-column rule. Style controls
    # how often test pairs use the spacious and larger end of the same family.
    style = int(rng.integers(3))
    one_placement = style == 0 and float(rng.random()) < .40
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=resolved_train + resolved_test, dtype=np.int64)
    placements = rng.permutation(_PLACEMENTS)
    puzzle = {"train": [], "test": []}
    masks, roles = [], []
    for i, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        regime = "source_train" if i < resolved_train else "target_test"
        anchor = i < 2
        if one_placement:
            placement = str(placements[0])
        elif anchor:
            placement = str(placements[i])
        else:
            placement = _PLACEMENTS[int(pair_rng.integers(4))]
        familiar = None
        if not anchor and masks and float(pair_rng.random()) < (.40 if style == 0 else .22):
            familiar = masks[int(pair_rng.integers(len(masks)))].copy()
        fixed_roles = None
        familiar_palette = None
        if not anchor and style == 0 and float(pair_rng.random()) < .55:
            fixed_roles = roles[-1]
            if float(pair_rng.random()) < .75:
                prior_split = "train" if i <= resolved_train else "test"
                prior = puzzle[prior_split][-1]
                prior_colors = np.unique(prior["input"])
                familiar_palette = tuple(int(c) for c in prior_colors
                                        if int(c) not in fixed_roles)
        for _ in range(64):
            pair, mask, role = _make_pair(pair_rng, regime, style, anchor,
                                          placement, familiar,
                                          roles[-1] if anchor and roles else None,
                                          fixed_roles, familiar_palette)
            if not anchor or not masks or (mask.shape, mask.tobytes()) != (
                    masks[0].shape, masks[0].tobytes()):
                break
        target = puzzle["test" if regime == "target_test" else "train"]
        target.append(pair)
        masks.append(mask)
        roles.append(role)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    counts_train, counts_test = set(), set()
    layouts, masks, shapes, palettes, sizes, placements = set(), set(), set(), set(), set(), set()
    puzzle_keys = set()
    mask_sizes, palette_sizes, gaps_x, gaps_y, separators = set(), set(), set(), set(), set()
    review = {name: 0 for name in (
        "rectangular_mask", "solid_mask", "sparse_mask", "source_mask",
        "small_matrix", "one_dimensional", "large_array", "repeated_row",
        "repeated_column", "symmetric_lookup", "uniform_lookup",
        "large_outer_margin", "edge_touch", "role_change", "stable_role",
        "palette_change", "one_placement", "familiar_test_mask",
        "new_test_color", "test_count_growth", "test_count_shrink",
        "test_row_growth", "test_column_growth", "combined_variation")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        puzzle_keys.add(tuple(_pair_key(p) for split in ("train", "test")
                              for p in puzzle[split]))
        pair_analyses = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                pair_analyses[split].append(a)
                assert np.array_equal(_predict_from_input(pair["input"]),
                                      pair["output"])
                masks.add(a["mask_key"])
                shapes.add(a["grid_shape"])
                sizes.add((a["rows"], a["cols"]))
                palettes.add(tuple(sorted(a["palette"])))
                placements.add(a["placement"])
                mask_sizes.add(a["mask"].shape)
                palette_sizes.add(len(a["palette"]))
                if a["gap_x"] is not None:
                    gaps_x.add(a["gap_x"])
                if a["gap_y"] is not None:
                    gaps_y.add(a["gap_y"])
                separators.add(a["separation"])
                layouts.add((a["rows"], a["cols"], a["mask"].shape,
                             a["gap_y"], a["gap_x"], a["placement"],
                             a["separation"], a["grid_shape"]))
                mask, lookup = a["mask"], a["lookup"]
                ly0, ly1, lx0, lx1 = a["lattice_bounds"]
                ky0, ky1, kx0, kx1 = a["lookup_bounds"]
                height, width = a["grid_shape"]
                margins = (min(ly0, ky0), height - 1 - max(ly1, ky1),
                           min(lx0, kx0), width - 1 - max(lx1, kx1))
                flags = {
                    "rectangular_mask": mask.shape[0] != mask.shape[1],
                    "solid_mask": bool(mask.all()),
                    "sparse_mask": float(mask.mean()) < .6,
                    "source_mask": mask.shape == (3, 3) and any(
                        np.array_equal(mask, source) for source in _SOURCE_MASKS),
                    "small_matrix": (a["rows"], a["cols"]) == (2, 2),
                    "one_dimensional": a["rows"] == 1 or a["cols"] == 1,
                    "large_array": a["slot_count"] >= 20,
                    "repeated_row": len({tuple(row) for row in lookup}) < a["rows"],
                    "repeated_column": len({tuple(col) for col in lookup.T}) < a["cols"],
                    "symmetric_lookup": np.array_equal(lookup, np.flipud(lookup))
                    or np.array_equal(lookup, np.fliplr(lookup)),
                    "uniform_lookup": len(a["palette"]) == 1,
                    "large_outer_margin": max(margins) >= 7,
                    "edge_touch": min(margins) == 0,
                    "combined_variation": a["placement"] == "right"
                    and mask.shape[0] != mask.shape[1],
                }
                for name, present in flags.items():
                    review[name] += bool(present)
        all_a = pair_analyses["train"] + pair_analyses["test"]
        train_a, test_a = pair_analyses["train"], pair_analyses["test"]
        train_colors = set().union(*(a["palette"] for a in train_a))
        train_masks = {a["mask_key"] for a in train_a}
        max_train_count = max(a["slot_count"] for a in train_a)
        review["role_change"] += len({(a["background"], a["carrier"])
                                      for a in all_a}) > 1
        review["stable_role"] += any(
            (a["background"], a["carrier"]) == (b["background"], b["carrier"])
            for a, b in zip(all_a, all_a[1:]))
        review["palette_change"] += len({tuple(sorted(a["palette"]))
                                         for a in all_a}) > 1
        review["one_placement"] += len({a["placement"] for a in all_a}) == 1
        review["familiar_test_mask"] += any(a["mask_key"] in train_masks
                                             for a in test_a)
        review["new_test_color"] += any(a["palette"] - train_colors
                                         for a in test_a)
        review["test_count_growth"] += any(a["slot_count"] > max_train_count
                                           for a in test_a)
        review["test_count_shrink"] += any(a["slot_count"] < max_train_count
                                           for a in test_a)
        review["test_row_growth"] += any(a["rows"] > max(b["rows"] for b in train_a)
                                         for a in test_a)
        review["test_column_growth"] += any(
            a["cols"] > max(b["cols"] for b in train_a) for a in test_a)
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(puzzle_keys) == 42
    assert len(layouts) >= 100 and len(masks) >= 25
    assert len(shapes) >= 20 and len(sizes) >= 15
    assert len(palettes) >= 30 and placements == set(_PLACEMENTS)
    assert mask_sizes == set(_MASK_SIZES)
    assert palette_sizes == set(range(1, 9))
    assert gaps_x == {1, 2, 3, 4} and gaps_y == {1, 2, 3, 4}
    assert separators == set(range(1, 9))
    assert all(value > 0 for value in review.values())
    same_a, same_b, different = (generate_puzzle(seed) for seed in (777, 777, 778))
    key = lambda p: tuple(_pair_key(pair) for split in ("train", "test")
                          for pair in p[split])
    assert key(same_a) == key(same_b)
    assert key(same_a) != key(different)
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (98765, 7, 3),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(3001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid seed accepted")
