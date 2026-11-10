import numpy as np


_SOURCE_INPUTS = {
    tuple(s.split()) for s in (
        "000000000000 033333333300 033373333300 033333333300 "
        "033333333300 033333373300 033833333300 033333333300 "
        "033333333300 033333333300 033333333300 000000000000",

        "00000000000 05555555550 05555555550 05355355350 "
        "05555555550 05555556550 00000000000",

        "000000000000000 066666666666660 066666666666660 "
        "066666666666660 066646666665660 066666666666660 "
        "066666666666660 066667666666660 066666666666660 "
        "066666666666660 066666666666660 066166666666660 "
        "066666666666660 066666666666690 000000000000000",

        "0000003333300000000 0000033333330000000 "
        "0000333333333000000 0003333333333300000 "
        "0033333733333330000 0333333333333333000 "
        "3333333333333333300 3333333333333333300 "
        "3333333333333333300 0000000000000000000 "
        "0000000000000000000",

        "00000000 09599590 09999990 09999990 09999990 00000000",
    )
}


def _rows(array):
    return tuple("".join(str(int(x)) for x in row) for row in array)


def _count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _rectangle(mask):
    rows, cols = np.where(mask)
    return bool(mask[rows.min():rows.max()+1, cols.min():cols.max()+1].all())


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(map(int, points[0]))
    seen, stack = {start}, [start]
    height, width = mask.shape
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row+dr, col+dc
            if (0 <= nr < height and 0 <= nc < width and mask[nr, nc]
                    and (nr, nc) not in seen):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _local_shape(rng, height, width, kind):
    if kind == "rectangle":
        return np.ones((height, width), dtype=bool)
    mask = np.zeros((height, width), dtype=bool)
    style = int(rng.integers(0, 4))
    depth = int(rng.integers(1, min(4, width-2)))
    step = int(rng.integers(2, 5))
    for row in range(height):
        if style == 0:
            left, right = min(depth, row//step), width
        elif style == 1:
            left = depth if height//3 <= row < 2*height//3 else 0
            right = width
        elif style == 2:
            left = depth if height//4 <= row < height//2 else 0
            right = width - (depth if 2*height//3 <= row else 0)
        else:
            left = max(0, depth-row//step)
            right = min(width, width-depth+row//step)
        mask[row, left:right] = True
    return np.rot90(mask, int(rng.integers(0, 4))).copy()


def _canvas_mask(rng, height, width, kind, train):
    shape = _local_shape(rng, height, width, kind)
    h, w = shape.shape
    margins = [int(x) for x in rng.integers(1, 6 if train else 7, size=4)]
    if not train and int(rng.integers(0, 3)) == 0:
        margins[int(rng.integers(0, 4))] = 0
    top, bottom, left, right = margins
    mask = np.zeros((h+top+bottom, w+left+right), dtype=bool)
    mask[top:top+h, left:left+w] = shape
    return mask


def _dimensions(rng, band):
    if band == "small":
        height, width = int(rng.integers(4, 10)), int(rng.integers(6, 11))
    elif band == "large":
        height, width = int(rng.integers(13, 21)), int(rng.integers(13, 21))
    else:
        height, width = int(rng.integers(5, 15)), int(rng.integers(6, 16))
    return (width, height) if int(rng.integers(0, 3)) == 0 else (height, width)


def _accent_colors(rng, background, foreground, count, pattern, forced=None):
    pool = [c for c in range(10) if c not in (background, foreground, 2)]
    if forced is not None and forced not in pool:
        forced = None
    first = int(forced if forced is not None else rng.choice(pool))
    if pattern == "same":
        colors = [first]*count
    elif pattern == "distinct":
        others = [c for c in pool if c != first]
        colors = [first] + list(map(int, rng.choice(others, size=count-1, replace=False)))
    else:
        second = int(rng.choice([c for c in pool if c != first]))
        repeated = count-1 if pattern == "source" else int(rng.integers(1, count))
        colors = [first]*repeated + [second]*(count-repeated)
    rng.shuffle(colors)
    return colors


def _positions(rng, mask, count, mode):
    coordinates = [tuple(map(int, p)) for p in np.argwhere(mask)]
    coordinate_set = set(coordinates)
    for _ in range(24):
        chosen = []
        if mode != "separate" and count >= 2:
            starts = [p for p in coordinates if (p[0], p[1]+1) in coordinate_set]
            start = starts[int(rng.integers(len(starts)))]
            chosen = [start, (start[0], start[1]+1)]
            if mode == "cluster" and count >= 3:
                neighbors = [p for p in coordinates if p not in chosen and
                             any(abs(p[0]-q[0])+abs(p[1]-q[1]) == 1 for q in chosen)]
                if neighbors:
                    chosen.append(neighbors[int(rng.integers(len(neighbors)))])
        if len(chosen) == count:
            return chosen
        for index in rng.permutation(len(coordinates)):
            point = coordinates[int(index)]
            if point in chosen or any(abs(point[0]-q[0])+abs(point[1]-q[1]) == 1 for q in chosen):
                continue
            chosen.append(point)
            if len(chosen) == count:
                return chosen
    if mode == "separate":
        raise RuntimeError("Could not separate accents")
    # Preserve the mandatory touching pair if spacing becomes too tight.
    for point in coordinates:
        if point not in chosen:
            chosen.append(point)
            if len(chosen) == count:
                return chosen
    raise RuntimeError("Insufficient foreground cells")


def _pair(rng, count, background, foreground, kind, band, train,
          pattern, mode, forced=None, dimensions=None):
    for _ in range(16):
        height, width = dimensions if dimensions is not None else _dimensions(rng, band)
        mask = _canvas_mask(rng, height, width, kind, train)
        if (max(mask.shape) <= 30 and mask.sum() >= max(24, 4*count)
                and np.count_nonzero(~mask) >= max(8, int(mask.size*0.12))
                and _connected(mask) and (kind != "irregular" or not _rectangle(mask))):
            break
    else:
        raise RuntimeError("Could not construct a clear foreground")
    grid = np.full(mask.shape, background, dtype=np.int8)
    grid[mask] = foreground
    colors = _accent_colors(rng, background, foreground, count, pattern, forced)
    positions = _positions(rng, mask, count, mode)
    if mode == "same_touch" and len(colors) >= 2:
        shared = next((c for c in colors if colors.count(c) >= 2), None)
        if shared is not None:
            colors.remove(shared)
            colors.remove(shared)
            colors = [shared, shared] + colors
    elif mode == "different_touch" and len(set(colors)) >= 2:
        first = colors.pop()
        second = next(c for c in colors if c != first)
        colors.remove(second)
        colors = [first, second] + colors
    for (row, col), color in zip(positions, colors):
        grid[row, col] = color
    return {"input": grid, "output": np.full((count, 1), 2, dtype=np.int8)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count_arg(num_train, "num_train", 2, 10)
    num_test = _count_arg(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test+1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test+1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    palette = [c for c in range(10) if c != 2]
    variable_background = bool(rng.integers(0, 2))
    base_background = int(rng.choice(palette))
    first_foreground = int(rng.choice([c for c in palette if c != base_background]))
    train_counts = list(map(int, rng.permutation([3, 4])))
    train_counts += [int(rng.choice([3, 4])) for _ in range(num_train-2)]
    joined_index = int(rng.integers(0, 2))
    first_dimensions = _dimensions(rng, "medium")
    second_dimensions = first_dimensions if bool(rng.integers(0, 2)) else _dimensions(rng, "medium")
    puzzle = {"train": [], "test": []}
    seen = set()

    def add_pair(regime, index, count, kind, band, pattern, mode, dimensions=None):
        for _ in range(20):
            if regime == "train" and index == 0:
                background, foreground = base_background, first_foreground
            else:
                background_pool = palette if variable_background else [base_background]
                if regime == "train" and index == 1 and variable_background:
                    background_pool = [c for c in palette if c not in (base_background, first_foreground)]
                background = int(rng.choice(background_pool))
                foreground_pool = [c for c in palette if c != background]
                if regime == "train" and index == 1:
                    foreground_pool = [c for c in foreground_pool if c != first_foreground]
                foreground = int(rng.choice(foreground_pool))
            forced = first_foreground if regime == "train" and index == 1 else None
            pair = _pair(rng, count, background, foreground, kind, band,
                         regime == "train", pattern, mode, forced, dimensions)
            signature = _rows(pair["input"])
            if signature not in seen and signature not in _SOURCE_INPUTS:
                seen.add(signature)
                puzzle[regime].append(pair)
                return
        raise RuntimeError("Could not make a distinct pair")

    for index, count in enumerate(train_counts):
        if index < 2:
            pattern = "source"
            mode = "same_touch" if index == joined_index else "separate"
            dimensions = first_dimensions if index == 0 else second_dimensions
        else:
            pattern = str(rng.choice(["source", "same", "distinct", "mixed"]))
            mode = str(rng.choice(["separate", "same_touch", "different_touch", "cluster"]))
            dimensions = None
        add_pair("train", index, count, "rectangle", "medium", pattern, mode, dimensions)

    if num_test >= 3:
        kinds = ["irregular", "rectangle", "rectangle"]
        bands = ["medium", "small", "large"]
        for _ in range(num_test-3):
            kinds.append(str(rng.choice(["rectangle", "irregular"])))
            bands.append(str(rng.choice(["small", "medium", "large"])))
    elif num_test == 2:
        kinds = ["irregular", "rectangle"]
        bands = [str(rng.choice(["small", "medium", "large"])),
                 str(rng.choice(["small", "large"]))]
    else:
        kinds = [str(rng.choice(["rectangle", "irregular"]))]
        bands = [str(rng.choice(["small", "large"]))]
    order = rng.permutation(num_test)
    kinds = [kinds[int(i)] for i in order]
    bands = [bands[int(i)] for i in order]
    if num_test >= 2:
        test_counts = [int(rng.choice([3, 4])), int(rng.choice([1, 2, 5, 6, 7]))]
        test_counts += [int(rng.integers(1, 8)) for _ in range(num_test-2)]
        rng.shuffle(test_counts)
    else:
        test_counts = [int(rng.integers(1, 8))]
    for index, (count, kind, band) in enumerate(zip(test_counts, kinds, bands)):
        pattern = str(rng.choice(["same", "distinct", "mixed"])) if count >= 2 else "same"
        mode = str(rng.choice(["separate", "same_touch", "different_touch", "cluster"]))
        add_pair("test", index, count, kind, band, pattern, mode)
    return puzzle


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for array in (pair["input"], pair["output"]):
        assert isinstance(array, np.ndarray) and array.ndim == 2
        assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
        assert np.issubdtype(array.dtype, np.integer)
        assert np.all((0 <= array) & (array <= 9))
    grid, output = pair["input"], pair["output"]
    corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
    assert len(set(map(int, corners))) == 1
    background = int(corners[0])
    assert background != 2 and not np.any(grid == 2)
    mask = grid != background
    assert mask.any() and (~mask).any() and _connected(mask)
    assert np.count_nonzero(~mask) >= max(8, int(mask.size*0.12))
    colors, sizes = np.unique(grid[mask], return_counts=True)
    foreground = int(colors[int(np.argmax(sizes))])
    accent_mask = mask & (grid != foreground)
    count = int(accent_mask.sum())
    assert 1 <= count <= 7 and int(np.max(sizes)) >= 3*count
    assert output.shape == (count, 1) and np.all(output == 2)
    return background, foreground, mask, grid[accent_mask]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training_counts, source_like = set(), set(), set()
    train_foregrounds, train_accent_colors, test_counts = [], [], set()
    test_irregular = False
    joined = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            background, foreground, mask, accents = _inspect(pair)
            signature = _rows(pair["input"])
            assert signature not in seen and signature not in _SOURCE_INPUTS
            seen.add(signature)
            if regime == "train":
                assert _rectangle(mask)
                assert not (mask[0].any() or mask[-1].any()
                            or mask[:, 0].any() or mask[:, -1].any())
                assert len(accents) in (3, 4)
                training_counts.add(len(accents))
                if len(np.unique(accents)) == 2:
                    source_like.add(len(accents))
                train_foregrounds.append(foreground)
                train_accent_colors.append(set(map(int, accents)))
                positions = np.argwhere(mask & (pair["input"] != foreground))
                joined |= any(
                    abs(int(a[0])-int(b[0])) + abs(int(a[1])-int(b[1])) == 1
                    and pair["input"][tuple(a)] == pair["input"][tuple(b)]
                    for i, a in enumerate(positions) for b in positions[i+1:]
                )
            else:
                test_counts.add(len(accents))
                test_irregular |= not _rectangle(mask)
    assert training_counts == {3, 4} and source_like == {3, 4} and joined
    assert any(
        foreground in train_accent_colors[j]
        for i, foreground in enumerate(train_foregrounds)
        for j in range(len(train_accent_colors)) if i != j
    )
    if len(puzzle["test"]) >= 2:
        assert test_counts & training_counts
        assert test_counts - training_counts
        assert test_irregular


def _same_puzzle(first, second):
    return (all(len(first[k]) == len(second[k]) for k in ("train", "test"))
            and all(np.array_equal(a[key], b[key])
                    for regime in ("train", "test")
                    for a, b in zip(first[regime], second[regime])
                    for key in ("input", "output")))


def self_test() -> None:
    train_lengths, test_lengths = set(), set()
    backgrounds, sizes, layouts, signatures = set(), set(), set(), set()
    shapes, counts, multiplicities = set(), set(), set()
    within_different_count = 0
    review = {name: 0 for name in (
        "variable_background", "stable_background", "role_reuse", "repeat_and_extra",
        "close_area_different_count", "same_count_distant_area", "small_many",
        "large_few", "irregular_many", "same_touch", "different_touch",
        "cluster_three", "boundary_accent", "edge_touch", "height_four",
        "wide_margin", "portrait", "landscape")}
    touching_sides = set()
    joined_indices = set()
    for seed in range(42):
        puzzle = generate_puzzle(10000+seed)
        validate_generated_puzzle(puzzle)
        demonstrated_markers = {int(value) for pair in puzzle["train"]
                                for value in pair["output"].flat}
        assert demonstrated_markers == {2}
        for pair in puzzle["test"]:
            test_input = pair["input"].copy()
            inferred_background = int(test_input[0, 0])
            local_colors, local_sizes = np.unique(
                test_input[test_input != inferred_background], return_counts=True)
            inferred_foreground = int(local_colors[int(np.argmax(local_sizes))])
            inferred_count = int(np.count_nonzero(
                (test_input != inferred_background) & (test_input != inferred_foreground)))
            prediction = np.full((inferred_count, 1), 2, dtype=np.int8)
            assert np.array_equal(prediction, pair["output"])
        assert _same_puzzle(puzzle, generate_puzzle(10000+seed))
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        signatures.add(tuple(_rows(p["input"]) for r in ("train", "test") for p in puzzle[r]))
        pair_counts = set()
        puzzle_backgrounds = set()
        area_counts = []
        train_foregrounds = []
        train_accent_colors = []
        train_joined = []
        train_output_counts = {len(p["output"]) for p in puzzle["train"]}
        test_output_counts = {len(p["output"]) for p in puzzle["test"]}
        review["repeat_and_extra"] += bool(train_output_counts & test_output_counts) and bool(test_output_counts - train_output_counts)
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                background, foreground, mask, accents = _inspect(pair)
                assert pair["output"].shape == (len(accents), 1)
                backgrounds.add(background)
                puzzle_backgrounds.add(background)
                sizes.add(pair["input"].shape)
                layouts.add((mask.shape, mask.tobytes()))
                rectangle = _rectangle(mask)
                shapes.add("rectangle" if rectangle else "irregular")
                counts.add(len(accents))
                multiplicities.add((len(accents), len(np.unique(accents))))
                pair_counts.add(len(accents))
                area = int(mask.sum())
                area_counts.append((area, len(accents)))
                coords = [tuple(map(int, p)) for p in np.argwhere(mask & (pair["input"] != foreground))]
                neighbors = [(a, b) for i, a in enumerate(coords) for b in coords[i+1:]
                             if abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
                same = any(pair["input"][a] == pair["input"][b] for a, b in neighbors)
                different = any(pair["input"][a] != pair["input"][b] for a, b in neighbors)
                review["same_touch"] += same
                review["different_touch"] += different
                review["cluster_three"] += any(
                    sum(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for b in coords) >= 2
                    for a in coords)
                review["boundary_accent"] += any(
                    any(not (0 <= a[0]+dr < mask.shape[0] and 0 <= a[1]+dc < mask.shape[1]
                             and mask[a[0]+dr, a[1]+dc]) for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)))
                    for a in coords)
                rows, cols = np.where(mask)
                h, w = int(rows.max()-rows.min()+1), int(cols.max()-cols.min()+1)
                review["portrait"] += h >= 2*w
                review["landscape"] += w >= 2*h
                margins = (int(rows.min()), int(mask.shape[0]-1-rows.max()),
                           int(cols.min()), int(mask.shape[1]-1-cols.max()))
                review["wide_margin"] += max(margins) >= 4
                if regime == "train":
                    train_foregrounds.append(foreground)
                    train_accent_colors.append(set(map(int, accents)))
                    train_joined.append(same)
                else:
                    review["small_many"] += rectangle and area <= 70 and len(accents) > 2
                    review["large_few"] += rectangle and area >= 150 and len(accents) < 5
                    review["irregular_many"] += not rectangle and len(accents) > 1
                    review["height_four"] += rectangle and h == 4
                    for side, hit in (("top", mask[0].any()), ("bottom", mask[-1].any()),
                                      ("left", mask[:,0].any()), ("right", mask[:,-1].any())):
                        if hit:
                            touching_sides.add(side)
                            review["edge_touch"] += 1
        review["variable_background" if len(puzzle_backgrounds) > 1 else "stable_background"] += 1
        review["role_reuse"] += any(
            fg in train_accent_colors[j] for i, fg in enumerate(train_foregrounds)
            for j in range(len(train_accent_colors)) if i != j)
        joined_indices.add(0 if train_joined[0] else 1)
        assert train_joined[0] != train_joined[1]
        review["close_area_different_count"] += any(
            abs(a-b) <= 12 and c != d for i, (a, c) in enumerate(area_counts)
            for b, d in area_counts[i+1:])
        review["same_count_distant_area"] += any(
            abs(a-b) >= 50 and c == d for i, (a, c) in enumerate(area_counts)
            for b, d in area_counts[i+1:])
        within_different_count += len(pair_counts) >= 2
    assert len(train_lengths) >= 4 and len(test_lengths) >= 3
    assert len(backgrounds) >= 7 and len(sizes) >= 30
    assert len(layouts) >= 100 and len(signatures) >= 40
    assert shapes == {"rectangle", "irregular"}
    assert counts == set(range(1, 8))
    assert (4, 2) in multiplicities and (4, 4) in multiplicities and (3, 1) in multiplicities
    assert within_different_count >= 40
    assert joined_indices == {0, 1}
    assert review["variable_background"] >= 10 and review["stable_background"] >= 10
    assert review["role_reuse"] == 42
    assert review["repeat_and_extra"] >= 15
    assert review["close_area_different_count"] >= 20
    assert review["same_count_distant_area"] >= 20
    assert all(review[name] >= 3 for name in (
        "small_many", "large_few", "irregular_many", "same_touch",
        "different_touch", "cluster_three", "boundary_accent", "edge_touch",
        "wide_margin", "portrait", "landscape"))
    assert review["height_four"] >= 1
    assert touching_sides == {"top", "bottom", "left", "right"}
    for n, m in ((2, 1), (10, 1), (10, 4), (3, 2)):
        puzzle = generate_puzzle(777, n, m)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (n, m)
        assert _same_puzzle(puzzle, generate_puzzle(777, n, m))
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(888, n, m)
        validate_generated_puzzle(puzzle)
        assert n is None or len(puzzle["train"]) == n
        assert m is None or len(puzzle["test"]) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(999, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
