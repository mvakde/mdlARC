"""Generate square-frame filling puzzles with a learned size-to-color key."""

import numpy as np


_ODD_SIZES = (5, 7, 9, 11, 13)
_SOURCE_INPUT_TEXTS = (
    """000000222222222
000000200000002
000000200000002
000000200000002
000000200020002
000000200000002
000000200000002
000000200000002
000000222222222
000000000000000
022222000000000
020002000000000
020202000000000
020002000000000
022222000000000""",
    """222222200
200000200
200000200
200200200
200000200
200000200
222222200
000000000
000000000""",
    """0222220000000
0200020000000
0202020000000
0200020000000
0222220000000
0000000000000
0000022222220
0000020000020
0000020000020
0000020020020
0000020000020
0000020000020
0000022222220""",
    """2222200
2000200
2020200
2000200
2222200
0000000
0000000""",
    """00000000000222220000
02222222220200020000
02000000020202020000
02000000020200020000
02000000020222220000
02000200020000000000
02000000020000000000
02000000020002222200
02000000020002000200
02222222220002020200
00000000000002000200
00000000000002222200
00000222222200000000
00000200000200000000
00000200000200000000
00000200200200000000
00000200000200000000
00000200000200000000
00000222222200000000
00000000000000000000""",
)
_SOURCE_INPUT_SIGNATURES = frozenset(
    (len(rows), len(rows[0]), "".join(rows))
    for rows in (text.splitlines() for text in _SOURCE_INPUT_TEXTS)
)


def _signature(grid):
    return grid.shape + (grid.tobytes(),)


def _source_signature(grid):
    return grid.shape + ("".join(str(int(x)) for x in grid.flat),)


def _valid_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(seed, num_train, num_test):
    if not _valid_integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            not _valid_integer(value) or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(num_test) + 1 + int(seed) % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + int(seed) % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _separated(a, b):
    ar, ac, az = a[:3]
    br, bc, bz = b[:3]
    return ar + az < br or br + bz < ar or ac + az < bc or bc + bz < ac


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)):
                    if (0 <= rr < height and 0 <= cc < width
                            and mask[rr, cc] and not seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            yield cells


def _parse_for_background(grid, background):
    frames = []
    markers = set()
    for color in (int(x) for x in np.unique(grid) if int(x) != background):
        for cells in _components(grid == color):
            if len(cells) == 1:
                markers.add((cells[0][0], cells[0][1], color))
                continue
            top = min(r for r, _ in cells)
            left = min(c for _, c in cells)
            bottom = max(r for r, _ in cells)
            right = max(c for _, c in cells)
            size = bottom - top + 1
            assert size == right - left + 1 and size in _ODD_SIZES
            boundary = {
                (r, c)
                for r in range(top, bottom + 1)
                for c in range(left, right + 1)
                if r in (top, bottom) or c in (left, right)
            }
            assert set(cells) == boundary
            interior = grid[top + 1:bottom, left + 1:right].copy()
            interior[size // 2 - 1, size // 2 - 1] = background
            assert np.all(interior == background)
            frames.append((top, left, size, color))
    assert frames
    assert markers == {
        (r + size // 2, c + size // 2, color)
        for r, c, size, color in frames
    }
    assert all(_separated(a, b)
               for i, a in enumerate(frames) for b in frames[i + 1:])
    return sorted(frames)


def _parse(grid):
    parses = []
    for background in (int(x) for x in np.unique(grid)):
        try:
            frames = _parse_for_background(grid, background)
        except AssertionError:
            continue
        parses.append((background, frames))
    assert len(parses) == 1
    return parses[0]


def _predict(input_grid, mapping, frames=None):
    if frames is None:
        _, frames = _parse(input_grid)
    output = input_grid.copy()
    for r, c, size, _ in frames:
        assert size in mapping
        output[r + 1:r + size - 1, c + 1:c + size - 1] = mapping[size]
        output[r + size // 2, c + size // 2] = input_grid[
            r + size // 2, c + size // 2
        ]
    return output


def _learn(train):
    mapping = {}
    for pair in train:
        input_grid, output_grid = pair["input"], pair["output"]
        _, frames = _parse(input_grid)
        for r, c, size, _ in frames:
            interior = output_grid[
                r + 1:r + size - 1, c + 1:c + size - 1
            ].copy()
            center = size // 2 - 1
            interior[center, center] = -1
            accents = set(int(x) for x in np.unique(interior)) - {-1}
            assert len(accents) == 1
            accent = accents.pop()
            assert accent != int(input_grid[r + 1, c + 1])
            if size in mapping:
                assert mapping[size] == accent
            mapping[size] = accent
        assert np.array_equal(output_grid, _predict(input_grid, mapping, frames))
    assert len(set(mapping.values())) == len(mapping)
    return mapping


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    used = set()
    pair_features = []
    input_colors = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= x <= 30 for x in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            sig = _signature(inp)
            assert sig not in used
            used.add(sig)
            assert _source_signature(inp) not in _SOURCE_INPUT_SIGNATURES
            background, frames = _parse(inp)
            assert all(size in _ODD_SIZES for _, _, size, _ in frames)
            assert not np.array_equal(inp, out)
            input_colors.update(int(x) for x in np.unique(inp))
            pair_features.append((split, background, frames, inp, out))
    mapping = _learn(puzzle["train"])
    assert set(mapping.values()).isdisjoint(input_colors)
    train_sizes = {
        size for split, _, frames, _, _ in pair_features
        if split == "train" for _, _, size, _ in frames
    }
    assert train_sizes == set(mapping)
    for _, _, frames, inp, out in pair_features:
        assert all(size in mapping for _, _, size, _ in frames)
        assert np.array_equal(out, _predict(inp, mapping, frames))


def _sample_conventions(rng, n_train):
    max_categories = 3 if n_train == 2 else 4
    category_count = int(rng.integers(1, max_categories + 1))
    sizes = tuple(sorted(int(x) for x in rng.choice(
        _ODD_SIZES, size=category_count, replace=False
    )))
    colors = [int(x) for x in rng.permutation(10)]
    accents = colors[:category_count]
    backgrounds = colors[category_count:category_count + 2]
    structures = colors[category_count + 2:category_count + 5]
    return {
        "sizes": sizes,
        "mapping": dict(zip(sizes, accents)),
        "backgrounds": backgrounds,
        "structures": structures,
    }


def _training_specs(rng, sizes, n_train):
    ordered = [int(sizes[i]) for i in rng.permutation(len(sizes))]
    a = ordered[0]
    if len(ordered) == 1:
        specs = [[a], [a, a]]
    elif len(ordered) == 2:
        specs = [[a, ordered[1]], [a, ordered[1]]]
    else:
        specs = [[a, ordered[1]], [a, ordered[2]]]
        if len(ordered) == 4:
            specs.append([ordered[3]])
    while len(specs) < n_train:
        count = int(rng.choice((1, 2, 3), p=(0.32, 0.48, 0.20)))
        specs.append([int(x) for x in rng.choice(sizes, count)])
    return specs


def _pack(rng, height, width, sizes):
    for _ in range(5):
        boxes = []
        for index, size in sorted(enumerate(sizes), key=lambda item: -item[1]):
            positions = [
                (r, c, size)
                for r in range(height - size + 1)
                for c in range(width - size + 1)
                if all(_separated((r, c, size), box) for _, box in boxes)
            ]
            if not positions:
                break
            boxes.append((index, positions[int(rng.integers(len(positions)))]))
        if len(boxes) == len(sizes):
            return [box for _, box in sorted(boxes)]
    return None


def _geometry(rng, sizes, regime):
    low = max(sizes) + 2
    for _ in range(48):
        if regime == "source_train" and rng.random() < 0.65:
            high = min(30, low + 7)
        else:
            high = 30
        lower = low if regime == "source_train" else max(low, 10)
        height = int(rng.integers(lower, high + 1))
        width = int(rng.integers(lower, high + 1))
        if regime == "target_test" and rng.random() < 0.06:
            height = width = 30
        elif rng.random() < 0.25:
            width = height
        elif rng.random() < 0.08:
            width = 30
        boxes = _pack(rng, height, width, sizes)
        if boxes is not None:
            return height, width, boxes
    boxes = _pack(rng, 30, 30, sizes)
    if boxes is not None:
        return 30, 30, boxes
    raise ValueError("size combination does not fit")


def _draw_pair(height, width, boxes, colors, background, mapping):
    grid = np.full((height, width), background, dtype=np.int8)
    for (r, c, size), color in zip(boxes, colors):
        grid[r, c:c + size] = color
        grid[r + size - 1, c:c + size] = color
        grid[r:r + size, c] = color
        grid[r:r + size, c + size - 1] = color
        grid[r + size // 2, c + size // 2] = color
    output = grid.copy()
    for r, c, size in boxes:
        output[r + 1:r + size - 1, c + 1:c + size - 1] = mapping[size]
        output[r + size // 2, c + size // 2] = grid[
            r + size // 2, c + size // 2
        ]
    return {"input": grid, "output": output}


def _make_pair(rng, regime, sizes, colors, background, mapping):
    height, width, boxes = _geometry(rng, sizes, regime)
    # Geometry is sampled without color bias; keep the size/color association.
    return _draw_pair(height, width, boxes, colors, background, mapping)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    n_train, n_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(int(seed))
    conventions = _sample_conventions(rng, n_train)
    sizes = conventions["sizes"]
    mapping = conventions["mapping"]
    backgrounds = conventions["backgrounds"]
    structures = conventions["structures"]
    specs = _training_specs(rng, sizes, n_train)
    puzzle = {"train": [], "test": []}
    used = set()
    for index, spec in enumerate(specs):
        if index == 0:
            colors = [structures[0]] * len(spec)
        elif index == 1:
            colors = [structures[1]] + [structures[0]] * (len(spec) - 1)
        else:
            colors = [structures[int(rng.integers(len(structures)))]
                      for _ in spec]
        background = backgrounds[index % 2]
        for _ in range(12):
            pair = _make_pair(rng, "source_train", spec, colors,
                              background, mapping)
            sig = _signature(pair["input"])
            if sig not in used and _source_signature(
                    pair["input"]) not in _SOURCE_INPUT_SIGNATURES:
                used.add(sig)
                puzzle["train"].append(pair)
                break
        else:
            raise ValueError("could not make unique training pair")
    for index in range(n_test):
        for _ in range(40):
            count = int(rng.choice((1, 2, 3, 4, 5),
                                   p=(0.14, 0.19, 0.25, 0.28, 0.14)))
            spec = [int(x) for x in rng.choice(sizes, count)]
            colors = [structures[int(rng.integers(len(structures)))]
                      for _ in spec]
            if count >= 2 and index == 0:
                colors[0], colors[1] = structures[:2]
            background = backgrounds[int(rng.integers(2))]
            try:
                pair = _make_pair(rng, "target_test", spec, colors,
                                  background, mapping)
            except ValueError:
                continue
            sig = _signature(pair["input"])
            if sig not in used and _source_signature(
                    pair["input"]) not in _SOURCE_INPUT_SIGNATURES:
                used.add(sig)
                puzzle["test"].append(pair)
                break
        else:
            raise ValueError("could not make unique test pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(
        len(a[split]) == len(b[split])
        and all(np.array_equal(pa[key], pb[key])
                for pa, pb in zip(a[split], b[split])
                for key in ("input", "output"))
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert _equal(generate_puzzle(777, 8, 3),
                  generate_puzzle(777, 8, 3))
    assert not _equal(generate_puzzle(900, 6, 2),
                      generate_puzzle(901, 6, 2))
    for i, (n_train, n_test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + i, n_train, n_test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (
            n_train, n_test)
    for i, (n_train, n_test) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + i, n_train, n_test)
        if n_train is not None:
            assert len(puzzle["train"]) == n_train
        if n_test is not None:
            assert len(puzzle["test"]) == n_test
    for n_train, n_test in ((1, None), (11, None), (None, 0),
                            (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(4000, n_train, n_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, layouts, maps, category_counts = set(), set(), set(), set()
    aspects, object_counts, test_counts = set(), set(), set()
    backgrounds, structural_colors, sizes = set(), set(), set()
    repeated_medium_or_large = False
    mixed_frame_grid = False
    changed_background = False
    changed_structure = False
    mixed_aspects = False
    same_size_different_color = False
    different_sizes_same_color = False
    compact = square_30 = edge = interior = False
    side_by_side = stacked = staggered = False
    test_more = test_fewer = False
    for puzzle in puzzles:
        mapping = _learn(puzzle["train"])
        maps.add(tuple(sorted(mapping.items())))
        category_counts.add(len(mapping))
        local_backgrounds, local_structures = set(), set()
        local_aspects = set()
        training_color_by_size = {}
        training_size_by_color = {}
        training_counts = []
        testing_counts = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg, frames = _parse(inp)
                height, width = inp.shape
                local_backgrounds.add(bg)
                local_structures.update(color for *_, color in frames)
                backgrounds.add(bg)
                structural_colors.update(color for *_, color in frames)
                sizes.update(size for _, _, size, _ in frames)
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(frames)))
                aspect = ("square" if height == width else
                          "wide" if width > height else "tall")
                aspects.add(aspect)
                local_aspects.add(aspect)
                object_counts.add(len(frames))
                mixed_frame_grid |= len({x[3] for x in frames}) > 1
                compact |= height <= 15 and width <= 15
                square_30 |= height == width == 30
                edge |= any(r == 0 or c == 0 or r + size == height
                            or c + size == width
                            for r, c, size, _ in frames)
                interior |= all(r > 0 and c > 0 and r + size < height
                                and c + size < width
                                for r, c, size, _ in frames)
                if len(frames) == 2:
                    a, b = frames
                    row_overlap = (a[0] < b[0] + b[2]
                                   and b[0] < a[0] + a[2])
                    col_overlap = (a[1] < b[1] + b[2]
                                   and b[1] < a[1] + a[2])
                    side_by_side |= row_overlap and not col_overlap
                    stacked |= col_overlap and not row_overlap
                    staggered |= not row_overlap and not col_overlap
                if split == "train":
                    training_counts.append(len(frames))
                    for _, _, size, color in frames:
                        training_color_by_size.setdefault(size, set()).add(color)
                        training_size_by_color.setdefault(color, set()).add(size)
                if split == "test":
                    testing_counts.append(len(frames))
                    test_counts.add(len(frames))
                    counts = [sum(f[2] == size for f in frames)
                              for size in (7, 9, 11, 13)]
                    repeated_medium_or_large |= max(counts) >= 2
        changed_background |= len(local_backgrounds) > 1
        changed_structure |= len(local_structures) > 1
        mixed_aspects |= len(local_aspects) > 1
        same_size_different_color |= any(
            len(colors) > 1 for colors in training_color_by_size.values())
        if len(mapping) > 1:
            different_sizes_same_color |= any(
                len(observed_sizes) > 1
                for observed_sizes in training_size_by_color.values())
        test_more |= max(testing_counts) > max(training_counts)
        test_fewer |= min(testing_counts) < max(training_counts)
    assert len(shapes) >= 35 and len(layouts) >= 100
    assert aspects == {"square", "wide", "tall"}
    assert set(_ODD_SIZES) <= sizes and category_counts >= {1, 2, 3}
    assert len(maps) >= 35 and len(backgrounds) >= 6
    assert len(structural_colors) >= 6 and object_counts >= {1, 2, 3, 4}
    assert test_counts >= {1, 2, 3, 4}
    assert repeated_medium_or_large and mixed_frame_grid
    assert changed_background and changed_structure
    assert mixed_aspects and same_size_different_color
    assert different_sizes_same_color
    assert compact and square_30 and edge and interior
    assert side_by_side and stacked and staggered
    assert test_more and test_fewer
