import numpy as np


_SOURCE_INPUT_ROWS = (
    ("555555000000", "550005030030", "555005030030", "555505033330",
     "555555000000", "500055022200", "500555022000", "505555020000",
     "555555000000", "505505001110", "505505000110", "500005000010",
     "555555000000"),
    ("111111000000", "110011002200", "100001002200", "111111000000",
     "100001003300", "110011033330", "111111000000", "110011066660",
     "110011006600", "111111000000"),
    ("8888800000", "8000802220", "8808800020", "8000800020",
     "8888800000", "8000804040", "8880804040", "8880804440",
     "8888800000", "8080803330", "8080800300", "8000803330",
     "8888800000", "8800800770", "8000807770", "8008807700",
     "8888800000"),
)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


_SOURCE_INPUT_KEYS = frozenset(
    ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))
    for rows in _SOURCE_INPUT_ROWS
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
    return count


def _connected_mask(rng, height, width, area):
    """Grow a joined silhouette from a wandering vertical spine."""
    mask = np.zeros((height, width), dtype=bool)
    col = int(rng.integers(1, width - 1))
    spare = area - height
    for row in range(height):
        mask[row, col] = True
        if row and spare and rng.random() < 0.55:
            choices = [step for step in (-1, 1)
                       if 1 <= col + step <= width - 2]
            col += int(rng.choice(choices))
            mask[row, col] = True
            spare -= 1
    while int(mask.sum()) < area:
        frontier = []
        for row in range(height):
            for col in range(1, width - 1):
                if mask[row, col]:
                    continue
                if ((row > 0 and mask[row - 1, col]) or
                    (row + 1 < height and mask[row + 1, col]) or
                    mask[row, col - 1] or mask[row, col + 1]):
                    frontier.append((row, col))
        row, col = frontier[int(rng.integers(len(frontier)))]
        mask[row, col] = True
    return mask


def _disconnected_mask(rng, height, width, area):
    inner = [(r, c) for r in range(height) for c in range(1, width - 1)]
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        chosen = rng.choice(len(inner), size=area, replace=False)
        for index in chosen:
            mask[inner[int(index)]] = True
        if np.all(mask.any(axis=1)) and _components(mask) > 1:
            return mask
    return None


def _one_mask(rng, height, width, area=None, allow_disconnected=True):
    capacity = height * (width - 2)
    if area is None:
        upper = max(height + 1, min(capacity - 1, int(capacity * 0.8)))
        area = int(rng.integers(height, upper + 1))
    if allow_disconnected and rng.random() < 0.10:
        result = _disconnected_mask(rng, height, width, area)
        if result is not None:
            return result
    return _connected_mask(rng, height, width, area)


def _sample_masks(rng, count, height, width):
    masks = []
    used = set()
    equal_area = rng.random() < 0.82
    for index in range(count):
        area = int(masks[0].sum()) if index == 1 and equal_area else None
        result = None
        # Reflections give equal-area lookalikes with identical bounding-box
        # dimensions; exact coordinates still distinguish their labels.
        if index == 1 and equal_area and rng.random() < 0.60:
            variants = (masks[0][:, ::-1], masks[0][::-1, :],
                        masks[0][::-1, ::-1])
            for choice in rng.permutation(len(variants)):
                candidate = variants[int(choice)].copy()
                key = tuple(candidate.flat)
                if key not in used:
                    result = candidate
                    break
        if result is None:
            for _ in range(200):
                candidate = _one_mask(rng, height, width, area)
                key = tuple(candidate.flat)
                if key not in used:
                    result = candidate
                    break
        if result is None:
            raise RuntimeError("could not make distinct silhouettes")
        masks.append(result)
        used.add(tuple(result.flat))
    return masks


def _layout(rng, regime, train_layouts):
    if regime == "target_test" and train_layouts and rng.random() < 0.25:
        return train_layouts[int(rng.integers(len(train_layouts)))]
    count = int(rng.integers(2, 6 if regime == "source_train" else 7))
    legal_heights = [height for height in range(2, 6)
                     if count * (height + 1) + 1 <= 30]
    height = int(rng.choice(legal_heights))
    width = int(rng.integers(5, 10))
    return count, height, width


def _permutation(rng, count, force_displaced):
    if not force_displaced and rng.random() < 0.14:
        return np.arange(count, dtype=int)
    if count == 4:
        mode = int(rng.choice(3, p=(0.44, 0.28, 0.28)))
        positions = rng.permutation(4)
        if mode == 1:
            order = np.arange(4, dtype=int)
            for index in range(4):
                order[positions[index]] = positions[(index + 1) % 4]
            return order
        if mode == 2:
            order = np.arange(4, dtype=int)
            order[positions[0]], order[positions[1]] = positions[1], positions[0]
            order[positions[2]], order[positions[3]] = positions[3], positions[2]
            return order
        for _ in range(100):
            order = rng.permutation(4)
            fixed = sum(int(order[index] == index) for index in range(4))
            if fixed in (1, 2):
                return order
        raise RuntimeError("could not make mixed permutation")
    for _ in range(100):
        order = rng.permutation(count)
        if np.any(order != np.arange(count)):
            return order
    raise RuntimeError("could not permute bands")


def _make_pair(rng, layout, background, force_displaced):
    count, height, width = layout
    palette = rng.permutation([color for color in range(10)
                               if color != background])
    structural = int(palette[0])
    exemplar_colors = [int(x) for x in palette[1:count + 1]]
    masks = _sample_masks(rng, count, height, width)
    order = _permutation(rng, count, force_displaced)
    grid_height = count * (height + 1) + 1
    inp = np.full((grid_height, width * 2), background, dtype=np.int8)
    left, right = inp[:, :width], inp[:, width:]
    left[:] = structural
    out = np.full((grid_height, width), structural, dtype=np.int8)
    for band, mask in enumerate(masks):
        start = band * (height + 1) + 1
        left[start:start + height][mask] = background
    for exemplar_band, cavity_band in enumerate(order):
        cavity_band = int(cavity_band)
        start = exemplar_band * (height + 1) + 1
        right[start:start + height][masks[cavity_band]] = exemplar_colors[exemplar_band]
        target = cavity_band * (height + 1) + 1
        out[target:target + height][masks[cavity_band]] = exemplar_colors[exemplar_band]
    return {"input": inp, "output": out}


def _inspect_pair(pair, regime):
    inp, out = pair["input"], pair["output"]
    rows, full_width = inp.shape
    assert full_width % 2 == 0
    width = full_width // 2
    assert 5 <= width <= 9 and out.shape == (rows, width)
    left, right = inp[:, :width], inp[:, width:]
    structural, background = int(left[0, 0]), int(right[0, 0])
    assert structural != background
    assert np.all(left[0] == structural) and np.all(right[0] == background)
    assert np.all((left == structural) | (left == background))
    boundaries = [r for r in range(rows) if np.all(left[r] == structural)]
    assert len(boundaries) >= 3 and boundaries[0] == 0 and boundaries[-1] == rows - 1
    count = len(boundaries) - 1
    assert 2 <= count <= (5 if regime == "source_train" else 6)
    heights = [b - a - 1 for a, b in zip(boundaries, boundaries[1:])]
    assert len(set(heights)) == 1 and 2 <= heights[0] <= 5
    height = heights[0]
    assert np.all(right[boundaries] == background)
    masks = []
    keys = []
    connected = 0
    boxes = []
    row_counts = []
    for a, b in zip(boundaries, boundaries[1:]):
        mask = left[a + 1:b] == background
        assert np.all(mask.any(axis=1))
        assert not np.any(mask[:, 0]) and not np.any(mask[:, -1])
        masks.append(mask)
        keys.append(tuple(mask.flat))
        connected += _components(mask) == 1
        occupied_rows, occupied_cols = np.nonzero(mask)
        boxes.append((int(occupied_rows.min()), int(occupied_rows.max()),
                      int(occupied_cols.min()), int(occupied_cols.max())))
        row_counts.append(tuple(int(x) for x in mask.sum(axis=1)))
    assert len(set(keys)) == count
    lookup = {key: index for index, key in enumerate(keys)}
    order = []
    colors = []
    for a, b in zip(boundaries, boundaries[1:]):
        band = right[a + 1:b]
        occupied = band != background
        assert np.all(occupied.any(axis=1))
        key = tuple(occupied.flat)
        assert key in lookup
        values = np.unique(band[occupied])
        assert len(values) == 1
        color = int(values[0])
        assert color not in (background, structural)
        order.append(lookup[key])
        colors.append(color)
    assert sorted(order) == list(range(count)) and len(set(colors)) == count
    expected = left.copy()
    for exemplar_band, cavity_band in enumerate(order):
        a = boundaries[cavity_band]
        expected[a + 1:a + 1 + height][masks[cavity_band]] = colors[exemplar_band]
    assert np.array_equal(out, expected)
    assert background not in out
    return {"layout": (count, height, width), "background": background,
            "structural": structural, "order": tuple(order),
            "areas": tuple(int(mask.sum()) for mask in masks),
            "connected": connected, "total": count,
            "boxes": tuple(boxes), "row_counts": tuple(row_counts),
            "mask_keys": tuple(keys)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    input_keys, pair_keys = set(), set()
    train_info = []
    for split, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in input_keys and pair_key not in pair_keys
            assert input_key not in _SOURCE_INPUT_KEYS
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            info = _inspect_pair(pair, regime)
            if split == "train":
                train_info.append(info)
    assert len({x["layout"] for x in train_info}) >= 2
    assert any(any(i != j for i, j in enumerate(x["order"])) for x in train_info)


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
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed)
    if ntrain is None and ntest is None:
        ntest = 1 + seed % 4
        minimum = max(2, ntest + 1)
        ntrain = minimum + ((seed + seed // 4) % (11 - minimum))
    elif ntrain is None:
        ntrain = int(rng.integers(max(2, ntest + 1), 11))
    elif ntest is None:
        ntest = int(rng.integers(1, min(4, ntrain - 1) + 1))
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")

    # Only the left-to-right panel convention and exact local-mask matching
    # persist across pairs. Color identities are local semantic roles.
    constant_background = bool(rng.integers(0, 2))
    base_background = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    layouts = []
    used = set()
    for split, regime, count in (("train", "source_train", ntrain),
                                 ("test", "target_test", ntest)):
        for index in range(count):
            pair = None
            for attempt in range(100):
                layout = _layout(rng, regime, layouts)
                if split == "train" and index == 1 and layout == layouts[0]:
                    continue
                if constant_background or (split == "train" and index == 0):
                    background = base_background
                elif split == "train" and index == 1:
                    background = int(rng.choice([x for x in range(10)
                                                 if x != base_background]))
                else:
                    background = int(rng.integers(0, 10))
                pair = _make_pair(rng, layout, background,
                                  force_displaced=(split == "train" and index == 0))
                key = _grid_key(pair["input"])
                if key not in used and key not in _SOURCE_INPUT_KEYS:
                    used.add(key)
                    break
                pair = None
            if pair is None:
                raise RuntimeError("could not make a unique pair")
            puzzle[split].append(pair)
            if split == "train":
                layouts.append(layout)
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None),
                                    (10, None))):
        first = generate_puzzle(2000 + index, *counts)
        second = generate_puzzle(2000 + index, *counts)
        assert _same_puzzle(first, second)
        validate_generated_puzzle(first)
        if counts[0] is not None:
            assert len(first["train"]) == counts[0]
        if counts[1] is not None:
            assert len(first["test"]) == counts[1]
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4},
                 {"num_train": 2.0}, {"num_test": "1"}):
        try:
            generate_puzzle(99, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal seed: {seed}")
    train_counts, test_counts, layouts, backgrounds = set(), set(), set(), set()
    widths, heights, band_counts, structural_colors = set(), set(), set(), set()
    orders, input_keys, pair_count = set(), set(), 0
    connected = 0
    total_masks = 0
    equal_area_pairs = 0
    same_box_pairs = same_row_count_pairs = 0
    aligned_train = aligned_test = 0
    four_cycles_test = two_swaps_test = 0
    combined = 0
    same_layout_tests = larger_tests = smaller_tests = 0
    variable_background_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_inspect_pair(p, "source_train") for p in puzzle["train"]]
        test_info = [_inspect_pair(p, "target_test") for p in puzzle["test"]]
        infos = train_info + test_info
        variable_background_puzzles += len({x["background"] for x in infos}) > 1
        for info in infos:
            layouts.add(info["layout"])
            band_count, height, width = info["layout"]
            band_counts.add(band_count)
            heights.add(height)
            widths.add(width)
            backgrounds.add(info["background"])
            structural_colors.add(info["structural"])
            orders.add(info["order"])
            connected += info["connected"]
            total_masks += info["total"]
            equal_area_pairs += len(set(info["areas"])) < len(info["areas"])
            same_box_pairs += len(set(info["boxes"])) < band_count
            same_row_count_pairs += len(set(info["row_counts"])) < band_count
            if (band_count >= 4 and info["connected"] == band_count
                    and len(set(info["areas"])) < band_count
                    and any(i != j for i, j in enumerate(info["order"]))):
                combined += 1
        aligned_train += sum(x["order"] == tuple(range(x["layout"][0]))
                             for x in train_info)
        for info in test_info:
            aligned_test += info["order"] == tuple(range(info["layout"][0]))
            if info["layout"][0] == 4 and all(
                    index != value for index, value in enumerate(info["order"])):
                seen = {0}
                value = info["order"][0]
                while value not in seen:
                    seen.add(value)
                    value = info["order"][value]
                four_cycles_test += len(seen) == 4
                two_swaps_test += len(seen) == 2
            training_layouts = {x["layout"] for x in train_info}
            same_layout_tests += info["layout"] in training_layouts
            larger_tests += info["layout"][0] > max(x["layout"][0] for x in train_info)
            smaller_tests += info["layout"][0] < min(x["layout"][0] for x in train_info)
        for pair in puzzle["train"] + puzzle["test"]:
            key = _grid_key(pair["input"])
            input_keys.add(key)
            pair_count += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 35 and len(backgrounds) == 10
    assert widths == {5, 6, 7, 8, 9} and heights == {2, 3, 4, 5}
    assert band_counts == {2, 3, 4, 5, 6} and len(structural_colors) == 10
    assert len(orders) >= 20 and len(input_keys) * 10 >= pair_count * 9
    assert connected / total_masks >= 0.75
    assert connected < total_masks
    assert equal_area_pairs >= pair_count * 0.60
    assert same_box_pairs >= 30 and same_row_count_pairs >= 20
    assert combined >= 20
    assert aligned_train > 0 and aligned_test > 0
    assert four_cycles_test > 0 and two_swaps_test > 0
    assert variable_background_puzzles >= 10
    assert same_layout_tests > 0 and larger_tests > 0 and smaller_tests > 0
