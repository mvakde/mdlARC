import numpy as np


# Exact source inputs; original dimensions and palettes remain allowed.
_SOURCE_INPUTS = {
    "0000000000000|0000000000000|0000000000000|0000222220000|0000200020000|0000220020000|0000020022000|0000020002000|0000022222000|0000000000000|0000000000000",
    "00000000000|00000000000|00000000000|00006660000|00006060000|00006060000|00006060000|00006660000|00000000000|00000000000",
    "0000000000000|0000000000000|0000000000000|0000000333000|0000000303000|0000003303300|0000003000300|0000003333300|0000000000000|0000000000000",
    "0000000|0888000|0888000|0000000|0000000|0000000|0000000",
    "00000000000|00000000000|00888880000|00800080000|00800080000|00800080000|00888880000|00000000000|00000000000|00000000000",
    "0000000000000|0000000000000|0000000000000|0000007770000|0000777070000|0000700070000|0000770070000|0000070077000|0000070007000|0000077777000|0000000000000",
}


def _valid_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be from {low} to {high}")


def _counts(rng, num_train, num_test):
    if num_train is not None:
        _valid_count(num_train, "num_train", 2, 10)
    if num_test is not None:
        _valid_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(int(v) for v in points[0])
    seen, stack = {start}, [start]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == len(points)


def _shape(rng, height, width, kind, clipping):
    """Compose a connected figure with independently varied row extents."""
    if clipping == "middle":
        mask = np.zeros((height, width), dtype=bool)
        mask[0, 2:int(rng.integers(4, width + 1))] = True
        mask[1, :] = True
        previous = int(rng.integers(1, width))
        for r in range(2, height):
            center = int(np.clip(previous + int(rng.integers(-1, 2)), 0, width - 1))
            lo = max(0, min(previous, center) - int(rng.integers(0, 3)))
            hi = min(width, max(previous, center) + int(rng.integers(1, 4)))
            mask[r, lo:hi] = True
            previous = center
        return mask
    if kind == 0:
        mask = np.ones((height, width), dtype=bool)
    elif kind == 1 and height >= 3 and width >= 3:
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    else:
        mask = np.zeros((height, width), dtype=bool)
        center = int(rng.integers(0, width))
        for r in range(height):
            previous = center
            center = int(np.clip(center + int(rng.integers(-1, 2)), 0, width - 1))
            if kind == 4:
                left_extension = int(rng.choice((0, 0, 1, 2, width // 2)))
                right_extension = int(rng.choice((0, 0, 1, 2, width // 2)))
            else:
                left_extension = int(rng.integers(0, width // 2 + 2))
                right_extension = int(rng.integers(0, width // 2 + 2))
            lo = max(0, min(previous, center) - left_extension)
            hi = min(width, max(previous, center) + right_extension + 1)
            mask[r, lo:hi] = True
        if kind in (3, 5):
            for index in rng.permutation(height * width):
                r, c = divmod(int(index), width)
                if not mask[r, c] or rng.random() > (0.36 if kind == 3 else 0.55):
                    continue
                if np.count_nonzero(mask[r]) <= 1:
                    continue
                mask[r, c] = False
                if not _connected(mask):
                    mask[r, c] = True
        if kind == 2 and height >= 4 and width >= 5:
            r = int(rng.integers(1, height - 1))
            first = int(np.flatnonzero(mask[r])[0])
            mask[r, :first + 1] = True
    if clipping in ("top", "multi"):
        for r in ((0, 1) if clipping == "multi" else (0,)):
            first = int(np.flatnonzero(mask[r])[0])
            mask[r, :max(3, first + 1)] = True
    elif rng.random() < 0.5:
        mask = mask[:, ::-1].copy()
    cols = np.flatnonzero(np.any(mask, axis=0))
    mask = mask[:, cols[0]:cols[-1] + 1]
    assert _connected(mask)
    return mask


def _shear(input_grid, background):
    output = np.full(input_grid.shape, background, dtype=input_grid.dtype)
    rows, cols = np.nonzero(input_grid != background)
    bottom = int(rows.max())
    target_cols = cols - (bottom - rows)
    retained = target_cols >= 0
    output[rows[retained], target_cols[retained]] = input_grid[rows[retained], cols[retained]]
    return output


def _placement(rng, mask, clipping):
    height, width = mask.shape
    if clipping == "middle":
        left = height - 3
    elif clipping == "top":
        left = int(rng.integers(0, height - 1))
    elif clipping == "multi":
        left = int(rng.integers(0, height - 2))
    else:
        left = max(height - 1 - r - int(np.flatnonzero(mask[r])[0]) for r in range(height))
        left = max(0, left) + int(rng.integers(0, 8))
    left = min(left, 30 - width)
    top_margin = int(rng.integers(0, 7))
    bottom_margin = int(rng.integers(0, 7))
    right_margin = int(rng.integers(0, 10))
    canvas_height = height + top_margin + bottom_margin
    canvas_width = left + width + right_margin
    aspect = int(rng.integers(0, 5))
    if aspect == 0:
        side = max(canvas_height, left + width)
        canvas_height = canvas_width = side
    elif aspect == 1:
        canvas_height = max(canvas_height, canvas_width + int(rng.integers(1, 5)))
    elif aspect == 2:
        canvas_width = max(canvas_width, canvas_height + int(rng.integers(1, 6)))
    canvas_height = min(30, canvas_height)
    canvas_width = min(30, canvas_width)
    top = top_margin if canvas_height > height else 0
    mass = int(np.count_nonzero(mask))
    if canvas_height * canvas_width <= 2 * mass:
        canvas_width = min(30, max(canvas_width, 2 * mass // canvas_height + 1))
    if canvas_height * canvas_width <= 2 * mass:
        canvas_height = min(30, max(canvas_height, 2 * mass // canvas_width + 1))
    return canvas_height, canvas_width, top, left


def _make_pair(rng, height, clipping, kind, background, foreground):
    width = int(rng.integers(4 if clipping == "middle" else 3, 12))
    mask = _shape(rng, height, width, kind, clipping)
    canvas_height, canvas_width, top, left = _placement(rng, mask, clipping)
    input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    patch = input_grid[top:top + height, left:left + mask.shape[1]]
    patch[mask] = foreground
    return {"input": input_grid, "output": _shear(input_grid, background)}


def _colors(rng, num_train, num_test):
    variable_background = bool(rng.integers(0, 2))
    backgrounds = [int(rng.integers(0, 10))]
    if variable_background:
        backgrounds.append(int(rng.choice([c for c in range(10) if c != backgrounds[0]])))
        backgrounds += [int(rng.integers(0, 10)) for _ in range(num_train + num_test - 2)]
    else:
        backgrounds *= num_train + num_test
    foregrounds = []
    for index, background in enumerate(backgrounds):
        previous = [c for c in foregrounds[:min(index, num_train)] if c != background]
        unused = [c for c in range(10) if c != background and c not in foregrounds[:min(index, num_train)]]
        if index == 1:
            choices = [c for c in range(10) if c != background and c != foregrounds[0]]
        elif index == 0 or (unused and (not previous or rng.random() < 0.43)):
            choices = unused
        else:
            choices = previous or [c for c in range(10) if c != background]
        foregrounds.append(int(rng.choice(choices)))
    return list(zip(backgrounds, foregrounds))


def _vertical_partner(rng, source, replaced):
    source_values, source_counts = np.unique(source["input"], return_counts=True)
    source_fg = int(source_values[int(np.argmin(source_counts))])
    rows, cols = np.nonzero(source["input"] == source_fg)
    top, bottom, left, right = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    shape = source["input"][top:bottom + 1, left:right + 1] == source_fg
    new_top = (top + int(rng.integers(1, 8))) % 8
    height = max(source["input"].shape[0], new_top + shape.shape[0] + 1)
    width = source["input"].shape[1]
    values, counts = np.unique(replaced["input"], return_counts=True)
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    input_grid = np.full((height, width), background, dtype=np.int8)
    patch = input_grid[new_top:new_top + shape.shape[0], left:left + shape.shape[1]]
    patch[shape] = foreground
    return {"input": input_grid, "output": _shear(input_grid, background)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    # The only puzzle-level geometric convention is the source's leftward unit
    # shear. Each pair visibly supplies its own background and foreground.
    short = int(rng.integers(2, 5))
    tall = int(rng.integers(max(4, short + 1), 8))
    clipped_style = str(rng.choice(("top", "middle", "multi")))
    train_specs = [(short, "none", int(rng.integers(0, 6))),
                   (tall, clipped_style, int(rng.integers(0, 6)))]
    if rng.random() < 0.5:
        short_style = clipped_style if short >= 3 or clipped_style == "top" else "top"
        train_specs[0] = (short, short_style, train_specs[0][2])
        train_specs[1] = (tall, "none", train_specs[1][2])
    for _ in range(num_train - 2):
        height = int(rng.integers(2, 9))
        style = str(rng.choice(("none", "none", "top", "middle", "multi")))
        if height == 2 and style in ("middle", "multi"):
            style = "top"
        train_specs.append((height, style, int(rng.integers(0, 6))))
    rng.shuffle(train_specs)
    max_train_height = max(spec[0] for spec in train_specs)
    test_specs = []
    for _ in range(num_test):
        height_mode = int(rng.integers(0, 3))
        if height_mode == 0:
            height = int(rng.integers(2, max_train_height + 1))
        elif height_mode == 1:
            height = min(10, max_train_height + 1)
        else:
            height = int(rng.integers(min(10, max_train_height + 1), 11))
        style = str(rng.choice(("none", "none", "top", "middle", "multi")))
        if height == 2 and style in ("middle", "multi"):
            style = "top"
        test_specs.append((height, style, int(rng.integers(0, 6))))
    colors = _colors(rng, num_train, num_test)
    puzzle, seen = {"train": [], "test": []}, set()
    for index, (height, style, kind) in enumerate(train_specs + test_specs):
        background, foreground = colors[index]
        for _ in range(32):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, height, style, kind, background, foreground)
            signature = _signature(pair["input"])
            if signature not in seen and _source_text(pair["input"]) not in _SOURCE_INPUTS and not np.array_equal(pair["input"], pair["output"]):
                seen.add(signature)
                puzzle["train" if index < num_train else "test"].append(pair)
                break
        else:
            raise RuntimeError("could not construct a distinct sheared figure")
    if num_train >= 4 and rng.random() < 0.4:
        source_index, target_index = (int(v) for v in rng.choice(num_train, size=2, replace=False))
        original = puzzle["train"][target_index]
        puzzle["train"][target_index] = _vertical_partner(rng, puzzle["train"][source_index], original)
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            puzzle["train"][target_index] = original
    return puzzle


def _signature(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _source_text(grid):
    return "|".join("".join(str(int(v)) for v in row) for row in grid)


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2 and counts.max() > counts.min()
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    mask = input_grid == foreground
    assert _connected(mask)
    rows, _ = np.nonzero(mask)
    height = int(rows.max() - rows.min() + 1)
    assert height >= 2
    assert all(np.any(mask[r]) for r in range(int(rows.min()), int(rows.max()) + 1))
    assert np.array_equal(output_grid, _shear(input_grid, background))
    assert np.any(output_grid == foreground)
    assert not np.array_equal(input_grid, output_grid)
    assert _source_text(input_grid) not in _SOURCE_INPUTS
    out_rows, _ = np.nonzero(output_grid == foreground)
    assert int(out_rows.max()) == int(rows.max())
    clipped = int(np.count_nonzero(mask)) > int(np.count_nonzero(output_grid == foreground))
    row_losses = tuple(int(np.count_nonzero(mask[r])) - int(np.count_nonzero(output_grid[r] == foreground))
                       for r in range(int(rows.min()), int(rows.max()) + 1))
    return background, foreground, height, clipped, row_losses


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    facts = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _signature(pair["input"])
            pair_key = (input_key, _signature(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key); seen_pairs.add(pair_key)
            facts[regime].append(_inspect_pair(pair))
    train = facts["train"]
    assert len({fact[1] for fact in train}) >= 2
    assert len({fact[2] for fact in train}) >= 2
    assert max(fact[2] for fact in train) >= 4
    assert any(fact[3] for fact in train) and any(not fact[3] for fact in train)
    assert all(fact[2] <= 10 for fact in facts["test"])


def _same_puzzle(a, b):
    return all(len(a[key]) == len(b[key]) and all(
        np.array_equal(pa[field], pb[field])
        for pa, pb in zip(a[key], b[key]) for field in ("input", "output")
    ) for key in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    canvases, crops, foregrounds, backgrounds = set(), set(), set(), set()
    clipped_styles, test_relations, aspect_ratios = set(), set(), set()
    within_clip, within_background, within_foreground, vertical_partners = 0, 0, 0, 0
    clip_by_split = {"train": set(), "test": set()}
    first_train_clip, first_test_clip, palette_modes = set(), set(), set()
    test_color_status, geometry = set(), set()
    top_inset_count = strong_clips = edge_contact = bottom_differs = 0
    top_edge = bottom_edge = left_edge = right_edge = 0
    inset_on_left = inset_on_right = 0
    mixed_aspects = mixed_geometry = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"])); test_counts.add(len(puzzle["test"]))
        train_facts = [_inspect_pair(p) for p in puzzle["train"]]
        all_facts = train_facts + [_inspect_pair(p) for p in puzzle["test"]]
        within_clip += len({fact[3] for fact in all_facts}) > 1
        within_background += len({fact[0] for fact in all_facts}) > 1
        within_foreground += len({fact[1] for fact in all_facts}) < len(all_facts)
        palette_modes.add("varied" if len({fact[0] for fact in all_facts}) > 1 else "fixed")
        first_train_clip.add(train_facts[0][3])
        first_test_clip.add(all_facts[len(train_facts)][3])
        train_colors = {fact[1] for fact in train_facts}
        test_color_status.update("repeat" if fact[1] in train_colors else "new" for fact in all_facts[len(train_facts):])
        mixed_aspects += len({
            "square" if p["input"].shape[0] == p["input"].shape[1] else
            "tall" if p["input"].shape[0] > p["input"].shape[1] else "wide"
            for p in puzzle["train"] + puzzle["test"]
        }) > 1
        train_crops = []
        for pair, fact in zip(puzzle["train"], train_facts):
            grid = pair["input"]
            rows, cols = np.nonzero(grid == fact[1])
            train_crops.append(((grid == fact[1])[rows.min():rows.max() + 1, cols.min():cols.max() + 1],
                                int(rows.min())))
        vertical_partners += any(
            np.array_equal(a[0], b[0]) and a[1] != b[1]
            for i, a in enumerate(train_crops) for b in train_crops[i + 1:]
        )
        max_train = max(fact[2] for fact in train_facts)
        for fact in all_facts[len(train_facts):]:
            test_relations.add("shorter" if fact[2] < max_train else "equal" if fact[2] == max_train else "taller")
        puzzle_geometries = set()
        for pair_index, (pair, fact) in enumerate(zip(puzzle["train"] + puzzle["test"], all_facts)):
            grid = pair["input"]
            independent_output = np.full(grid.shape, fact[0], dtype=grid.dtype)
            occupied_rows, occupied_cols = np.nonzero(grid == fact[1])
            anchor = int(occupied_rows.max())
            for row, col in zip(occupied_rows, occupied_cols):
                new_col = int(col) - (anchor - int(row))
                if new_col >= 0:
                    independent_output[int(row), new_col] = fact[1]
            assert np.array_equal(independent_output, pair["output"])
            backgrounds.add(fact[0]); foregrounds.add(fact[1])
            canvases.add(grid.shape)
            aspect_ratios.add("square" if grid.shape[0] == grid.shape[1] else "tall" if grid.shape[0] > grid.shape[1] else "wide")
            rows, cols = np.nonzero(grid == fact[1])
            crop = (grid == fact[1])[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
            crops.add(_signature(crop.astype(np.int8)))
            if np.all(crop):
                shape_class = "solid"
            elif (crop.shape[0] >= 3 and crop.shape[1] >= 3
                  and np.all(crop[0]) and np.all(crop[-1])
                  and np.all(crop[:, 0]) and np.all(crop[:, -1])):
                shape_class = "ring"
            elif np.count_nonzero(crop) <= 2.2 * crop.shape[0]:
                shape_class = "sparse"
            else:
                shape_class = "irregular"
            geometry.add(shape_class); puzzle_geometries.add(shape_class)
            top_inset_count += int(np.flatnonzero(crop[0])[0]) > 0
            inset_on_left += int(np.flatnonzero(crop[0])[0]) > 0
            inset_on_right += crop.shape[1] - 1 - int(np.flatnonzero(crop[0])[-1]) > 0
            bottom_differs += not np.array_equal(crop[0], crop[-1])
            top_edge += int(rows.min()) == 0
            bottom_edge += int(rows.max()) == grid.shape[0] - 1
            left_edge += int(cols.min()) == 0
            right_edge += int(cols.max()) == grid.shape[1] - 1
            edge_contact += (int(rows.min()) == 0 or int(rows.max()) == grid.shape[0] - 1
                             or int(cols.min()) == 0 or int(cols.max()) == grid.shape[1] - 1)
            losses = fact[4]
            strong_clips += sum(losses) >= 4
            if not any(losses):
                label = "none"
            elif losses[0] == 0 and sum(v > 0 for v in losses) == 1:
                label = "middle_only"
            elif losses[0] > 0 and sum(v > 0 for v in losses) == 1:
                label = "top_only"
            else:
                label = "multiple"
            clipped_styles.add(label)
            clip_by_split["train" if pair_index < len(train_facts) else "test"].add(label)
        mixed_geometry += len(puzzle_geometries) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(canvases) >= 35 and len(crops) >= 80
    assert len(backgrounds) >= 8 and len(foregrounds) >= 8
    assert aspect_ratios == {"square", "tall", "wide"}
    assert clipped_styles == {"none", "middle_only", "top_only", "multiple"}
    assert clip_by_split["train"] == clipped_styles and clip_by_split["test"] == clipped_styles
    assert test_relations == {"shorter", "equal", "taller"}
    assert within_clip == 42 and within_background >= 10 and within_foreground >= 10
    assert vertical_partners >= 3
    assert first_train_clip == first_test_clip == {False, True}
    assert palette_modes == {"fixed", "varied"} and test_color_status == {"repeat", "new"}
    assert geometry == {"solid", "ring", "sparse", "irregular"}
    assert top_inset_count >= 30 and inset_on_left >= 30 and inset_on_right >= 10
    assert strong_clips >= 20 and bottom_differs >= 50
    assert edge_contact >= 50 and min(top_edge, bottom_edge, left_edge, right_edge) >= 10
    assert mixed_aspects >= 20 and mixed_geometry >= 20
    assert not _same_puzzle(generate_puzzle(0), generate_puzzle(1))
    for counts in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(141, *counts)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        assert _same_puzzle(puzzle, generate_puzzle(141, *counts))
    for kwargs in ({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10}):
        puzzle = generate_puzzle(142, **kwargs)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(142, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(143, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
