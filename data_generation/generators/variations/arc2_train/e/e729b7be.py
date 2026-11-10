"""Generate half-turn completion puzzles with visible center cues."""

import numpy as np


_SOURCE_INPUTS = {
    tuple("""77777777477777777 77777777477777777 77777777477777777
77777766477777777 77777611477777777 77776113477777777
77761139477777777 77761399477777777 87761399877777778
77611399477777777 77761399477777777 77761139477777777
77776113477777777 77776161477777777 77777676477777777
77777777477777777 77777777477777777""".split()),
    tuple("""777777747777777 777777747777777 777771147777777
777712247777777 777129947777777 777129947777777
771299947777777 877129987777778 777129947777777
777129947777777 777712247777777 777771147777777
777777747777777 777777747777777 777777747777777""".split()),
    tuple("""77777777477777777 77777777477777777 77777777477777777
77777777477777777 77777777477777777 77777777477777777
77777755477777777 77775599477777777 87559911877777778
77591111477777777 77591111477777777 77596666477777777
77596666477777777 77559966477777777 77775599477777777
77777755477777777 77777777477777777""".split()),
    tuple("""7777777774777777777 7777777774777777777 7777777334777777777
7777733394777777777 7777730094777777777 7777330094777777777
7773322294777777777 7733222294777777777 7732222294777777777
8735555598777777778 7735555594777777777 7731111194777777777
7731111194777777777 7733111194777777777 7773366694777777777
7777333794777777777 7777773334777777777 7777777774777777777
7777777774777777777""".split()),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        if rng.random() < 0.30:
            train, test = 3, 1
        else:
            test = int(rng.integers(1, 5))
            train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _size(rng, regime, classic, training_sizes):
    if classic:
        if regime == "source_train":
            return int(rng.choice((15, 17))), int(rng.choice((15, 17)))
        return int(rng.choice((19, 21, 23))), int(rng.choice((19, 21, 23)))
    if regime == "target_test" and training_sizes and rng.random() < 0.34:
        return training_sizes[int(rng.integers(len(training_sizes)))]
    height = int(rng.choice((13, 15, 17, 19, 21, 23, 25, 27, 29)))
    width = height if rng.random() < 0.62 else int(rng.choice((13, 15, 17, 19, 21, 23, 25, 27, 29)))
    return height, width


def _row_interval(rng, height, vertical_position, large):
    center = height // 2
    if vertical_position == "above":
        high = center
        low = min(2, high)
        length = int(rng.integers(low, high + 1))
        if large:
            length = max(length, max(3, high - 2))
        start = int(rng.integers(0, center - length + 1))
    elif vertical_position == "below":
        high = height - center - 1
        length = int(rng.integers(2, high + 1))
        if large:
            length = max(length, max(3, high - 2))
        start = int(rng.integers(center + 1, height - length + 1))
    else:
        length = int(rng.integers(4, height + 1))
        if large:
            length = max(length, max(7, height - 5))
        low = max(0, center - length + 1)
        high = min(center, height - length)
        start = int(rng.integers(low, high + 1))
    return start, start + length


def _mask(rng, height, depth, mode, position, large):
    mask = np.zeros((height, depth), dtype=bool)
    r0, r1 = _row_interval(rng, height, position, large)
    span = r1 - r0
    width = int(rng.integers(2, depth + 1))
    if large:
        width = max(width, max(4, depth - 2))
    width = min(width, depth)
    if rng.random() < 0.48:
        d0 = 0
    elif rng.random() < 0.25:
        d0 = depth - width
    else:
        d0 = int(rng.integers(0, depth - width + 1))
    d1 = d0 + width

    if mode in ("solid", "broken"):
        peak = int(rng.integers(r0, r1))
        slope = int(rng.integers(1, max(2, span // 3 + 1)))
        for row in range(r0, r1):
            taper = abs(row - peak) // slope
            reach = max(1, width - taper + int(rng.integers(-1, 2)))
            mask[row, d0:min(d1, d0 + reach)] = True
        if mode == "broken":
            for row in range(r0, r1):
                if rng.random() < 0.20:
                    mask[row, :] = False
            holes = rng.random(mask.shape) < float(rng.uniform(0.08, 0.27))
            mask &= ~holes
    elif mode == "islands":
        for _ in range(int(rng.integers(2, 6))):
            patch_h = int(rng.integers(1, max(3, span // 2 + 2)))
            patch_w = int(rng.integers(1, max(3, width // 2 + 2)))
            patch_h, patch_w = min(patch_h, span), min(patch_w, width)
            rr = int(rng.integers(r0, r1 - patch_h + 1))
            dd = int(rng.integers(d0, d1 - patch_w + 1))
            mask[rr:rr + patch_h, dd:dd + patch_w] = True
        if large:
            rr = int(rng.integers(r0, r1))
            mask[rr:min(r1, rr + 2), d0:d1] = True
    elif mode == "hollow":
        thickness = int(rng.integers(1, 3))
        mask[r0:min(r1, r0 + thickness), d0:d1] = True
        mask[max(r0, r1 - thickness):r1, d0:d1] = True
        mask[r0:r1, d0:min(d1, d0 + thickness)] = True
        mask[r0:r1, max(d0, d1 - thickness):d1] = True
        if rng.random() < 0.42 and span >= 5:
            rr = int(rng.integers(r0 + 2, r1 - 1))
            mask[rr, d0:d1] = True
    elif mode == "thin":
        for _ in range(int(rng.integers(2, 5))):
            rr = int(rng.integers(r0, r1))
            dd = int(rng.integers(d0, d1))
            if rng.random() < 0.5:
                mask[rr, dd:d1] = True
            else:
                mask[rr:r1, dd] = True
    else:  # staggered short bands, with occasional blank rows
        for rr in range(r0, r1):
            if rng.random() < 0.14:
                continue
            for _ in range(int(rng.integers(1, 4))):
                maximum = width if large else max(2, width // 2 + 1)
                length = int(rng.integers(1, maximum + 1))
                left = int(rng.integers(d0, d1 - length + 1))
                mask[rr, left:left + length] = True
            if rng.random() < 0.18:
                gap = int(rng.integers(d0, d1))
                mask[rr, gap:min(d1, gap + 2)] = False

    # The outer center marker is an immutable cue, including when the shape
    # reaches the outer edge on other rows.
    mask[height // 2, depth - 1] = False
    if np.count_nonzero(mask) < 7:
        candidates = [(r, d) for r in range(height) for d in range(depth)
                      if not mask[r, d] and not (r == height // 2 and d == depth - 1)]
        candidates.sort(key=lambda cell: (max(r0 - cell[0], 0, cell[0] - r1 + 1)
                                          + max(d0 - cell[1], 0, cell[1] - d1 + 1)))
        for row, col in candidates[:7 - int(np.count_nonzero(mask))]:
            mask[row, col] = True
    return mask


def _color_indices(rng, mask, count, style):
    cells = np.argwhere(mask)
    row, col = cells[:, 0], cells[:, 1]
    if style == "blobs":
        centers = cells[rng.choice(len(cells), size=count, replace=False)]
        distances = ((row[:, None] - centers[None, :, 0]) ** 2
                     + (col[:, None] - centers[None, :, 1]) ** 2)
        indices = np.argmin(distances, axis=1)
        change = rng.random(len(cells)) < 0.07
        indices[change] = rng.integers(0, count, size=int(np.count_nonzero(change)))
    elif style == "stripes":
        period = int(rng.integers(1, 4))
        indices = ((row + col // int(rng.integers(2, 5))) // period) % count
    elif style == "staggered":
        indices = (row // int(rng.integers(1, 3))
                   + col // int(rng.integers(1, 4)) + row % 2) % count
    elif style == "patches":
        tile_h = int(rng.integers(2, 5))
        tile_w = int(rng.integers(2, 5))
        indices = (row // tile_h + 2 * (col // tile_w)) % count
        change = rng.random(len(cells)) < 0.24
        indices[change] = rng.integers(0, count, size=int(np.count_nonzero(change)))
    else:
        indices = (2 * row + 3 * col + (row * col) % 3) % count
    indices = np.asarray(indices, dtype=np.int64)
    # Every requested palette color is visible, even in small thin examples.
    for color, place in enumerate(rng.choice(len(cells), size=count, replace=False)):
        indices[int(place)] = color
    return cells, indices


def _completion(grid, side):
    height, width = grid.shape
    center = width // 2
    result = grid.copy()
    if side == "left":
        result[:, center + 1:] = np.rot90(grid[:, :center], 2)
    else:
        result[:, :center] = np.rot90(grid[:, center + 1:], 2)
    return result


def _reflection(grid, side):
    center = grid.shape[1] // 2
    result = grid.copy()
    if side == "left":
        result[:, center + 1:] = grid[:, :center][:, ::-1]
    else:
        result[:, :center] = grid[:, center + 1:][:, ::-1]
    return result


def _make_pair(rng, height, width, orientation, side, roles, palette, shape, position, large, color_style):
    background, guide, marker = roles
    # Canonical coordinates always have a vertical guide. A transpose rotates
    # the whole marked construction to the horizontal-guide family.
    if orientation == "horizontal":
        height, width = width, height
    depth = width // 2
    mask = _mask(rng, height, depth, shape, position, large)
    grid = np.full((height, width), background, dtype=np.int64)
    grid[:, depth] = guide
    grid[height // 2, (0, depth, width - 1)] = marker
    cells, indices = _color_indices(rng, mask, len(palette), color_style)
    for (row, distance), index in zip(cells, indices):
        col = depth - 1 - int(distance) if side == "left" else depth + 1 + int(distance)
        grid[int(row), col] = palette[int(index)]
    result = _completion(grid, side)
    if orientation == "horizontal":
        grid, result = grid.T.copy(), result.T.copy()
    return {"input": grid, "output": result}


def _signature(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    colors = [int(value) for value in rng.permutation(10)]
    orientation = "horizontal" if rng.random() < 0.19 else "vertical"
    mixed_sides = rng.random() < 0.48 and train_count >= 3
    fixed_side = "right" if rng.random() < 0.29 else "left"
    varying_roles = rng.random() < 0.42
    novel_test_color = colors[-1] if rng.random() < 0.42 else None
    classic = rng.random() < 0.22
    stable_roles = tuple(colors[:3])
    puzzle = {"train": [], "test": []}
    seen = set()
    training_sizes = []
    training_payload_colors = set()
    train_sides = [fixed_side] * train_count
    if mixed_sides:
        train_sides[0], train_sides[1] = "left", "right"
        for index in range(2, train_count):
            train_sides[index] = "right" if rng.random() < 0.5 else "left"
        rng.shuffle(train_sides)
    test_sides = [train_sides[int(rng.integers(train_count))] for _ in range(test_count)]
    if mixed_sides and test_count >= 2:
        test_sides[0], test_sides[1] = "left", "right"

    for split, count, sides in (("train", train_count, train_sides),
                                ("test", test_count, test_sides)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            for attempt in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                size = _size(pair_rng, regime, classic, training_sizes)
                if classic:
                    if split == "train":
                        original_size = 15 if index % 2 == 0 else 17
                        size = (original_size, original_size)
                    elif index == 0:
                        size = (19, 19)
                if varying_roles:
                    role_pool = [value for value in colors if value != novel_test_color]
                    roles = tuple(int(value) for value in pair_rng.choice(role_pool, size=3, replace=False))
                else:
                    roles = stable_roles
                eligible = [value for value in colors if value not in roles
                            and (split == "test" or value != novel_test_color)]
                if split == "test" and novel_test_color is None:
                    familiar = [value for value in eligible if value in training_payload_colors]
                    if familiar:
                        eligible = familiar
                if split == "train":
                    palette_width = int(pair_rng.choice((1, 2, 3, 3, 4, 4, 5, 6)))
                else:
                    palette_width = int(pair_rng.choice((1, 2, 3, 4, 5, 6, 7, 7)))
                palette_width = min(palette_width, len(eligible))
                palette = [int(value) for value in pair_rng.choice(eligible, size=palette_width, replace=False)]
                if split == "test" and novel_test_color is not None:
                    palette[-1] = novel_test_color
                shape = str(pair_rng.choice(("solid", "broken", "islands", "hollow", "thin", "mosaic"),
                                             p=(0.25, 0.18, 0.16, 0.14, 0.10, 0.17)))
                position = str(pair_rng.choice(("above", "across", "below"), p=(0.22, 0.56, 0.22)))
                large = bool(pair_rng.random() < (0.58 if split == "train" else 0.66))
                style = str(pair_rng.choice(("blobs", "stripes", "staggered", "patches", "interleaved"),
                                            p=(0.23, 0.22, 0.20, 0.21, 0.14)))
                pair = _make_pair(pair_rng, *size, orientation, sides[index], roles,
                                  palette, shape, position, large, style)
                signature = _signature(pair["input"])
                if signature in seen:
                    continue
                if split == "train" and index == 0:
                    canonical = pair["input"] if orientation == "vertical" else pair["input"].T
                    if np.array_equal(_completion(canonical, sides[index]), _reflection(canonical, sides[index])):
                        continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == "train":
                    training_sizes.append(size)
                    training_payload_colors.update(palette)
                break
            else:
                raise ValueError("could not make a unique informative pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
    for key in ("input", "output"):
        grid = pair[key]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
        _require(all(1 <= dimension <= 30 for dimension in grid.shape), "grid bounds")
        _require(bool(np.all((grid >= 0) & (grid <= 9))), "ARC colors")
    original, output = pair["input"], pair["output"]
    _require(original.shape == output.shape, "shape changed")
    _require(all(dimension % 2 == 1 and dimension >= 13 for dimension in original.shape), "odd marked grid")
    _require(tuple("".join(str(int(x)) for x in row) for row in original) not in _SOURCE_INPUTS,
             "source input copied")
    counts = np.bincount(original.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    _require(int(np.count_nonzero(counts == counts[background])) == 1, "dominant background")
    candidates = []
    for orientation in ("vertical", "horizontal"):
        grid = original if orientation == "vertical" else original.T
        h, w = grid.shape
        middle_row, middle_col = h // 2, w // 2
        guide, marker = int(grid[0, middle_col]), int(grid[middle_row, 0])
        if len({background, guide, marker}) != 3:
            continue
        expected = np.full(h, guide, dtype=grid.dtype)
        expected[middle_row] = marker
        if not np.array_equal(grid[:, middle_col], expected):
            continue
        if int(grid[middle_row, -1]) != marker:
            continue
        if int(np.count_nonzero(grid == guide)) != h - 1 or int(np.count_nonzero(grid == marker)) != 3:
            continue
        left = grid[:, :middle_col].copy()
        right = grid[:, middle_col + 1:].copy()
        left[middle_row, 0] = background
        right[middle_row, -1] = background
        left_count = int(np.count_nonzero(left != background))
        right_count = int(np.count_nonzero(right != background))
        if (left_count >= 5 and right_count == 0) or (right_count >= 5 and left_count == 0):
            candidates.append((orientation, "left" if left_count else "right", guide, marker))
    _require(len(candidates) == 1, "unique guide, markers, and blank destination")
    orientation, side, guide, marker = candidates[0]
    grid = original if orientation == "vertical" else original.T
    completed = _completion(grid, side)
    expected = completed if orientation == "vertical" else completed.T
    _require(np.array_equal(output, expected), "incorrect half-turn output")
    _require(np.array_equal(output, np.rot90(output, 2)), "output lacks point symmetry")
    _require(not np.array_equal(original, output), "degenerate pair")
    color_mask = (grid != background) & (grid != guide) & (grid != marker)
    rows, columns = np.where(color_mask)
    payload = {int(x) for x in grid[color_mask]}
    reflection = _reflection(grid, side)
    return {"orientation": orientation, "side": side, "roles": (background, guide, marker),
            "size": original.shape, "area": len(rows), "colors": payload,
            "rows": set(int(x) for x in rows), "columns": set(int(x) for x in columns),
            "distinguishes": not np.array_equal(completed, reflection),
            "outer": bool(np.any(columns == (0 if side == "left" else grid.shape[1] - 1))),
            "guide_touch": bool(np.any(columns == (grid.shape[1] // 2 - 1 if side == "left" else grid.shape[1] // 2 + 1))),
            "center_cross": min(rows) < grid.shape[0] // 2 < max(rows),
            "center_absent": grid.shape[0] // 2 not in rows}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    signatures = set()
    train_info, test_info = [], []
    for split, pairs, infos in (("train", train, train_info), ("test", test, test_info)):
        for pair in pairs:
            info = _pair_info(pair)
            signature = _signature(pair["input"])
            _require(signature not in signatures, "duplicate input or pair")
            signatures.add(signature)
            infos.append(info)
    _require(len({x["orientation"] for x in train_info + test_info}) == 1, "inconsistent guide orientation")
    _require({x["side"] for x in test_info} <= {x["side"] for x in train_info}, "unshown source side")
    _require(any(x["distinguishes"] for x in train_info), "training does not distinguish a half-turn")


def _same(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[key], b[key]) for a, b in zip(first[split], second[split])
        for key in ("input", "output")) for split in ("train", "test"))


def _expect_error(**kwargs):
    try:
        generate_puzzle(123, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def _component_count(mask):
    seen = set()
    count = 0
    for row, col in np.argwhere(mask):
        cell = (int(row), int(col))
        if cell in seen:
            continue
        count += 1
        seen.add(cell)
        stack = [cell]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and neighbor not in seen):
                    seen.add(neighbor)
                    stack.append(neighbor)
    return count


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _require(_same(generate_puzzle(991, 7, 2), generate_puzzle(991, 7, 2)), "determinism")
    _require(len({tuple(_signature(pair["input"]) for split in ("train", "test")
                        for pair in puzzle[split]) for puzzle in puzzles}) >= 40, "seed diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(700 + train + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts")
    for args in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(800 + (args[0] or 0) + (args[1] or 0), *args)
        _require((args[0] is None or len(puzzle["train"]) == args[0]) and
                 (args[1] is None or len(puzzle["test"]) == args[1]), "partial counts")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
                   {"num_train": True}):
        _expect_error(**kwargs)
    infos = [[_pair_info(pair) for split in ("train", "test") for pair in p[split]] for p in puzzles]
    flat = [info for group in infos for info in group]
    _require({x["orientation"] for x in flat} == {"vertical", "horizontal"}, "orientation diversity")
    _require({x["side"] for x in flat} == {"left", "right"}, "side diversity")
    _require(any(len({x["side"] for x in group}) == 2 for group in infos), "mixed sides")
    _require(any(len({x["side"] for x in group}) == 1 for group in infos), "single side")
    _require(any(len({x["roles"] for x in group}) > 1 for group in infos), "role changes")
    _require(any(len({x["roles"] for x in group}) == 1 for group in infos), "stable roles")
    _require(any(x["center_absent"] for x in flat) and any(x["center_cross"] for x in flat), "vertical placement")
    _require(any(x["outer"] for x in flat) and any(not x["guide_touch"] for x in flat), "depth placement")
    _require(any(x["area"] >= 35 for group, p in zip(infos, puzzles)
                 for x in group[:len(p["train"])]), "substantial training")
    _require({len(x["colors"]) for x in flat} >= {1, 2, 3, 4, 5, 6}, "palette range")
    _require(any(any(t["size"] == r["size"] for r in group[:len(p["train"])])
                 for group, p in zip(infos, puzzles) for t in group[len(p["train"]):]), "same-size tests")
    _require(any(len(p["train"]) == 3 and len(p["test"]) == 1 for p in puzzles), "compact episodes")
    row_gaps = detached = solid_suffixes = rich_colors = substantial = 0
    original_area_range = 0
    wholly_above = wholly_below = 0
    size_palette = {}
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                source = pair["input"] if info["orientation"] == "vertical" else pair["input"].T
                result = pair["output"] if info["orientation"] == "vertical" else pair["output"].T
                h, w = source.shape
                mid = w // 2
                source_columns = range(mid) if info["side"] == "left" else range(mid + 1, w)
                destination_columns = range(mid + 1, w) if info["side"] == "left" else range(mid)
                for row in range(h):
                    for col in source_columns:
                        _require(int(result[h - 1 - row, w - 1 - col]) == int(source[row, col]),
                                 "independent coordinate reversal")
                        _require(int(result[row, col]) == int(source[row, col]), "source preservation")
                    for col in destination_columns:
                        _require(int(source[row, col]) in (info["roles"][0], info["roles"][2]),
                                 "blank destination")
                bg, guide, marker = info["roles"]
                mask = (source != bg) & (source != guide) & (source != marker)
                rows = np.flatnonzero(np.any(mask, axis=1))
                row_gaps += len(rows) < rows[-1] - rows[0] + 1
                detached += _component_count(mask) > 1
                wholly_above += int(rows[-1] < h // 2)
                wholly_below += int(rows[0] > h // 2)
                color_components = sum(_component_count(source == color) for color in info["colors"])
                rich_colors += color_components >= 12
                if split == "train":
                    substantial += info["area"] >= 35
                    original_area_range += 35 <= info["area"] <= 50
                    size_palette.setdefault(info["size"], set()).add(len(info["colors"]))
                suffix = True
                for row in rows:
                    columns = np.flatnonzero(mask[int(row)])
                    if info["side"] == "left":
                        suffix &= bool(columns[-1] == mid - 1 and
                                       np.array_equal(columns, np.arange(columns[0], mid)))
                    else:
                        suffix &= bool(columns[0] == mid + 1 and
                                       np.array_equal(columns, np.arange(mid + 1, columns[-1] + 1)))
                solid_suffixes += suffix
    _require(row_gaps >= 30 and detached >= 30 and solid_suffixes >= 20,
             "outline regimes")
    _require(wholly_above >= 10 and wholly_below >= 10, "one-sided vertical placement")
    _require(rich_colors >= 60, "interleaved color boundaries")
    _require(substantial >= 50 and original_area_range >= 15, "source-scale training areas")
    _require(len({shape for shape in size_palette}) >= 20 and
             any(len(widths) >= 4 for widths in size_palette.values()), "size and palette independence")
    role_vary = novel = familiar = same_size = smaller = classic = combined = 0
    for puzzle in puzzles:
        train_info = [_pair_info(pair) for pair in puzzle["train"]]
        test_info = [_pair_info(pair) for pair in puzzle["test"]]
        role_vary += len({info["roles"] for info in train_info + test_info}) > 1
        train_colors = set().union(*(info["colors"] for info in train_info))
        test_colors = set().union(*(info["colors"] for info in test_info))
        novel += bool(test_colors - train_colors)
        familiar += test_colors <= train_colors
        same_size += any(t["size"] == r["size"] for t in test_info for r in train_info)
        smaller += any(np.prod(t["size"]) < np.prod(r["size"])
                       for t in test_info for r in train_info)
        classic += ({info["size"] for info in train_info} <= {(15, 15), (17, 17)}
                    and any(info["size"] == (19, 19) for info in test_info))
        combined += (len({info["side"] for info in train_info}) == 2
                     and len({info["roles"] for info in train_info + test_info}) > 1
                     and any(len(info["rows"]) < max(info["rows"]) - min(info["rows"]) + 1
                             for info in train_info + test_info))
    _require(role_vary >= 5 and novel >= 5 and familiar >= 5, "color-role and novelty regimes")
    _require(same_size >= 5 and smaller >= 5 and classic >= 2, "test-size regimes")
    _require(combined >= 2, "combined variation axes")
    damaged = {split: [{key: value.copy() for key, value in pair.items()}
                       for pair in puzzles[0][split]] for split in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted wrong output")
