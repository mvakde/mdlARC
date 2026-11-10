import numpy as np


_OUTPUT_SHAPES = (
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
    (5, 5),
    (3, 6),
    (6, 3),
    (4, 6),
    (6, 4),
    (5, 6),
    (6, 5),
    (6, 6),
)


_SOURCE_SIGNATURES = (
    (
        (
            "0000000040",
            "0000000044",
            "0006660000",
            "0000660000",
            "0000060000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        ("666", "466", "446"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0330000000",
            "0300000000",
            "0300000000",
            "0000000000",
            "0000000000",
            "0000000007",
            "0000000077",
            "0000000077",
        ),
        ("337", "377", "377"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000300000",
            "0003330000",
            "0000000000",
            "0000000000",
            "0111000000",
            "0101000000",
        ),
        ("111", "131", "333"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000200000",
            "0002200000",
            "0000000000",
            "0000000000",
            "0000000880",
            "0000000800",
            "0000000888",
            "0000000000",
        ),
        ("882", "822", "888"),
    ),
)


def _validated_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train_count = _validated_count("num_train", num_train, 2, 10)
    test_count = _validated_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif test_count is None:
        largest_test_count = min(4, train_count - 1)
        test_count = int(rng.integers(1, largest_test_count + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _bbox(coords):
    assert coords
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_for_value(grid, value):
    rows, cols = np.where(grid == value)
    coords = {(int(r), int(c)) for r, c in zip(rows, cols)}
    return _bbox(coords)


def _normalize(coords):
    coords = {(int(r), int(c)) for r, c in coords}
    min_row = min(r for r, _ in coords)
    min_col = min(c for _, c in coords)
    return tuple(
        sorted((r - min_row, c - min_col) for r, c in coords)
    )


def _connected(coords):
    coords = {(int(r), int(c)) for r, c in coords}
    if not coords:
        return False

    remaining = set(coords)
    start = remaining.pop()
    stack = [start]

    while stack:
        row, col = stack.pop()
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor in remaining:
                remaining.remove(neighbor)
                stack.append(neighbor)

    return not remaining


def _packing_solutions(shape_a, shape_b, max_solutions=2):
    """Enumerate all colored rectangular packings, including other factor sizes."""
    shape_a = _normalize(shape_a)
    shape_b = _normalize(shape_b)
    area = len(shape_a) + len(shape_b)
    height_a, width_a = _shape_dimensions(shape_a)
    height_b, width_b = _shape_dimensions(shape_b)
    solutions = []

    for height in range(1, min(30, area) + 1):
        if area % height:
            continue
        width = area // height
        if width > 30 or height_a > height or height_b > height:
            continue
        if width_a > width or width_b > width:
            continue
        rectangle = {(r, c) for r in range(height) for c in range(width)}
        for row in range(height - height_a + 1):
            for col in range(width - width_a + 1):
                placed_a = frozenset((r + row, c + col) for r, c in shape_a)
                remainder = rectangle - placed_a
                if _normalize(remainder) == shape_b:
                    solutions.append((height, width, placed_a))
                    if len(solutions) >= max_solutions:
                        return solutions
    return solutions


def _shape_dimensions(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _partition_is_unique(mask, height, width):
    rectangle = {(r, c) for r in range(height) for c in range(width)}
    mask = set(mask)
    other = rectangle - mask
    if not mask or not other or not _connected(mask) or not _connected(other):
        return False
    solutions = _packing_solutions(mask, other)
    return (len(solutions) == 1 and solutions[0][:2] == (height, width)
            and solutions[0][2] == frozenset(mask))


def _boundary_mask(cuts, height, width, orientation):
    if orientation == "vertical":
        return {(r, c) for r in range(height) for c in range(int(cuts[r]))}
    if orientation == "horizontal":
        return {(r, c) for c in range(width) for r in range(int(cuts[c]))}
    raise ValueError("unknown orientation")


def _sample_join(rng, height, width, hard=False):
    orientations = []
    if width >= 3:
        orientations.append("vertical")
    if height >= 3:
        orientations.append("horizontal")
    orientations = list(rng.permutation(orientations))
    for orientation in orientations:
        length = height if orientation == "vertical" else width
        upper = (width if orientation == "vertical" else height) - 1
        for _ in range(80):
            base = int(rng.integers(1, upper + 1))
            if rng.random() < (0.16 if hard else 0.60):
                cuts = [base] * length
                index = int(rng.integers(0, length))
                alternatives = [x for x in range(1, upper + 1) if x != base]
                cuts[index] = int(rng.choice(alternatives))
            else:
                cuts = [int(x) for x in rng.integers(1, upper + 1, size=length)]
            if len(set(cuts)) < 2:
                continue
            mask = _boundary_mask(cuts, height, width, orientation)
            if _partition_is_unique(mask, height, width):
                return mask
        for base in range(1, upper + 1):
            for alt in range(1, upper + 1):
                if alt == base:
                    continue
                cuts = [base] * length
                cuts[0] = alt
                mask = _boundary_mask(cuts, height, width, orientation)
                if _partition_is_unique(mask, height, width):
                    return mask
    raise RuntimeError("no unique join")


def _socket_position(rng, height, width, patch_h, patch_w, kind):
    if kind == "corner":
        return (int(rng.choice((0, height - patch_h))),
                int(rng.choice((0, width - patch_w))))
    if kind == "edge":
        if int(rng.integers(0, 2)):
            return (int(rng.integers(1, height - patch_h)),
                    int(rng.choice((0, width - patch_w))))
        return (int(rng.choice((0, height - patch_h))),
                int(rng.integers(1, width - patch_w)))
    if kind == "interior":
        return (int(rng.integers(1, height - patch_h)),
                int(rng.integers(1, width - patch_w)))
    raise ValueError("unknown socket kind")


def _socket_contact_kind(mask, height, width):
    top, left, bottom, right = _bbox(mask)
    contacts = sum((top == 0, bottom == height - 1,
                    left == 0, right == width - 1))
    return "interior" if contacts == 0 else "edge" if contacts == 1 else "corner"


def _sample_socket(rng, height, width, kind, hard=False,
                   source_like=False):
    if kind in ("edge", "interior") and min(height, width) < 4:
        raise ValueError("edge and interior sockets require at least 4 by 4")
    if source_like:
        assert kind == "corner"
        square = {(r, width - 2 + c) for r in range(2) for c in range(2)}
        cells = sorted(square)
        for index in rng.permutation(len(cells)):
            mask = square - {cells[int(index)]}
            if (_socket_contact_kind(mask, height, width) == "corner"
                    and _partition_is_unique(mask, height, width)):
                return mask
        raise RuntimeError("no unique upper-right L socket")
    max_h = min(3, height - (2 if kind != "corner" else 1))
    max_w = min(3, width - (2 if kind != "corner" else 1))
    for _ in range(160):
        patch_h = int(rng.integers(2, max_h + 1))
        patch_w = int(rng.integers(2, max_w + 1))
        row, col = _socket_position(rng, height, width, patch_h, patch_w, kind)
        local = {(r, c) for r in range(patch_h) for c in range(patch_w)}
        style = int(rng.choice((1, 2, 3, 3) if hard else (0, 0, 1, 2)))
        if style == 1:
            corners = ((0, 0), (0, patch_w - 1),
                       (patch_h - 1, 0), (patch_h - 1, patch_w - 1))
            local.remove(corners[int(rng.integers(0, 4))])
        elif style >= 2:
            deletions = int(rng.integers(1, min(4, patch_h * patch_w - 2)))
            for _ in range(deletions):
                removable = [cell for cell in local
                             if _connected(local - {cell})]
                if not removable:
                    break
                local.remove(removable[int(rng.integers(0, len(removable)))])
        if len(local) >= height * width / 2:
            continue
        mask = {(row + r, col + c) for r, c in local}
        if (_socket_contact_kind(mask, height, width) == kind
                and _partition_is_unique(mask, height, width)):
            return mask

    # A compact patch gives a deterministic exit for every supported size.
    if kind == "corner" and min(height, width) == 2:
        positions = [(0, 0), (0, width - 2),
                     (height - 2, 0), (height - 2, width - 2)]
        for row, col in positions:
            square = {(row + r, col + c) for r in range(2) for c in range(2)}
            for cell in sorted(square):
                mask = square - {cell}
                if _partition_is_unique(mask, height, width):
                    return mask
    else:
        if kind == "corner":
            row, col = 0, 0
        elif kind == "edge":
            row, col = 1, 0
        else:
            row, col = 1, 1
        mask = {(row + r, col + c) for r in range(2) for c in range(2)}
        if _partition_is_unique(mask, height, width):
            return mask
    raise RuntimeError("no unique socket")


def _sample_partition(rng, kind, height, width, hard=False,
                      source_like=False):
    if kind == "join":
        return _sample_join(rng, height, width, hard)
    return _sample_socket(rng, height, width, kind, hard, source_like)


def _scatter_fragments(rng, pieces, background):
    items = list(pieces)
    if int(rng.integers(0, 2)):
        items.reverse()
    first, second = items
    shape_a, color_a = _normalize(first[0]), int(first[1])
    shape_b, color_b = _normalize(second[0]), int(second[1])
    ha, wa = _shape_dimensions(shape_a)
    hb, wb = _shape_dimensions(shape_b)
    gap = int(rng.integers(1, 5))
    if int(rng.integers(0, 2)):
        left = int(rng.integers(1, 5))
        right = int(rng.integers(1, 5))
        height = max(ha, hb) + int(rng.integers(2, 7))
        width = left + wa + gap + wb + right
        ar = int(rng.integers(0, height - ha + 1))
        br = int(rng.integers(0, height - hb + 1))
        ac, bc = left, left + wa + gap
    else:
        top = int(rng.integers(1, 5))
        bottom = int(rng.integers(1, 5))
        height = top + ha + gap + hb + bottom
        width = max(wa, wb) + int(rng.integers(2, 7))
        ar, br = top, top + ha + gap
        ac = int(rng.integers(0, width - wa + 1))
        bc = int(rng.integers(0, width - wb + 1))
    grid = np.full((height, width), int(background), dtype=np.int8)
    for r, c in shape_a:
        grid[ar + r, ac + c] = color_a
    for r, c in shape_b:
        grid[br + r, bc + c] = color_b
    return grid


def generate_pair(rng, kind, rectangle_shape, background,
                  forced_color=None, hard=False, source_like=False):
    height, width = rectangle_shape
    first_region = _sample_partition(rng, kind, height, width,
                                     hard, source_like)
    rectangle = {(r, c) for r in range(height) for c in range(width)}
    second_region = rectangle - first_region
    if int(rng.integers(0, 2)):
        first_region, second_region = second_region, first_region
    choices = [c for c in range(10) if c != background]
    if forced_color is not None and forced_color != background:
        color_a = int(forced_color)
        choices.remove(color_a)
        color_b = int(rng.choice(choices))
    else:
        selected = rng.choice(choices, size=2, replace=False)
        color_a, color_b = int(selected[0]), int(selected[1])
    output = np.full((height, width), color_b, dtype=np.int8)
    for r, c in first_region:
        output[r, c] = color_a
    input_grid = _scatter_fragments(rng,
                                    ((first_region, color_a),
                                     (second_region, color_b)),
                                    background)
    return input_grid, output


def _grid_rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _is_source_pair(input_grid, output_grid):
    return (_grid_rows(input_grid), _grid_rows(output_grid)) in _SOURCE_SIGNATURES


def _grid_key(grid):
    return (tuple(int(x) for x in grid.shape),
            tuple(int(x) for x in grid.ravel()))


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _puzzle_key(puzzle):
    return (tuple(_pair_key(p) for p in puzzle["train"]),
            tuple(_pair_key(p) for p in puzzle["test"]))


def _background_color(input_grid, output_grid):
    background = set(int(x) for x in np.unique(input_grid)) - set(
        int(x) for x in np.unique(output_grid))
    assert len(background) == 1
    return next(iter(background))


def _visible_kind(output_grid):
    height, width = output_grid.shape
    colors = [int(x) for x in np.unique(output_grid)]
    boxes = [_bbox_for_value(output_grid, color) for color in colors]
    if all(box[0] == 0 and box[2] == height - 1 for box in boxes):
        return "join"
    if all(box[1] == 0 and box[3] == width - 1 for box in boxes):
        return "join"
    small_color = min(colors, key=lambda c: int(np.count_nonzero(output_grid == c)))
    top, left, bottom, right = _bbox_for_value(output_grid, small_color)
    contacts = sum((top == 0, bottom == height - 1,
                    left == 0, right == width - 1))
    if contacts == 0:
        return "interior"
    if contacts == 1:
        return "edge"
    return "corner"


def _input_layout_axes(input_grid, colors):
    a = _bbox_for_value(input_grid, colors[0])
    b = _bbox_for_value(input_grid, colors[1])
    axes = set()
    if a[2] + 1 < b[0] or b[2] + 1 < a[0]:
        axes.add("vertical")
    if a[3] + 1 < b[1] or b[3] + 1 < a[1]:
        axes.add("horizontal")
    return axes


def _assert_pair_invariants(input_grid, output_grid):
    input_values = set(int(x) for x in np.unique(input_grid))
    output_values = set(int(x) for x in np.unique(output_grid))
    assert len(input_values) == 3 and len(output_values) == 2
    assert output_values < input_values
    background = _background_color(input_grid, output_grid)
    assert background not in output_values
    assert int(np.count_nonzero(input_grid == background)) > output_grid.size
    assert input_grid.shape != output_grid.shape
    shapes = []
    for color in sorted(output_values):
        input_coords = {(int(r), int(c)) for r, c in zip(*np.where(input_grid == color))}
        output_coords = {(int(r), int(c)) for r, c in zip(*np.where(output_grid == color))}
        assert input_coords and output_coords
        assert _connected(input_coords) and _connected(output_coords)
        assert _normalize(input_coords) == _normalize(output_coords)
        shapes.append(_normalize(input_coords))
    assert _input_layout_axes(input_grid, sorted(output_values))
    assert sum(len(shape) for shape in shapes) == output_grid.size
    solutions = _packing_solutions(shapes[0], shapes[1])
    assert len(solutions) == 1
    height, width, placed_a = solutions[0]
    assert (height, width) == output_grid.shape
    actual_a = frozenset((int(r), int(c)) for r, c in zip(
        *np.where(output_grid == min(output_values))))
    assert placed_a == actual_a
    assert _visible_kind(output_grid) in {"corner", "edge", "interior", "join"}


def _shape_pool(kind):
    if kind in ("edge", "interior"):
        return tuple(s for s in _OUTPUT_SHAPES if min(s) >= 4)
    if kind == "corner":
        return tuple(s for s in _OUTPUT_SHAPES if min(s) >= 3)
    return _OUTPUT_SHAPES


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    kinds = ("corner", "edge", "interior", "join")
    test_kinds = [str(rng.choice(kinds)) for _ in range(test_count)]
    training_kinds = list(dict.fromkeys(test_kinds))
    other_kinds = list(rng.permutation(kinds))
    for kind in other_kinds:
        if len(training_kinds) >= train_count:
            break
        if kind not in training_kinds:
            training_kinds.append(kind)
    while len(training_kinds) < train_count:
        training_kinds.append(str(rng.choice(kinds)))
    training_kinds = list(rng.permutation(training_kinds))

    fixed_shape = bool(rng.random() < 0.38)
    fixed_background = bool(rng.random() < 0.42)
    if fixed_shape:
        shape_pool = set(_OUTPUT_SHAPES)
        for kind in training_kinds + test_kinds:
            shape_pool &= set(_shape_pool(kind))
        ordered_shapes = sorted(shape_pool)
        common_shape = ordered_shapes[int(rng.integers(0, len(ordered_shapes)))]
    else:
        common_shape = None
    common_background = int(rng.integers(0, 10)) if fixed_background else None

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    first_shape = None
    first_background = None

    def add_pair(split, kind):
        nonlocal first_shape, first_background
        for _ in range(80):
            shape_choices = list(_shape_pool(kind))
            if common_shape is not None:
                shape = common_shape
            else:
                if kind == "join":
                    narrow_shapes = [s for s in shape_choices if min(s) == 2]
                    shape_choices.extend(narrow_shapes * 2)
                if split == "train" and len(puzzle["train"]) == 1:
                    shape_choices = [s for s in shape_choices if s != first_shape]
                shape = shape_choices[int(rng.integers(0, len(shape_choices)))]
            if common_background is not None:
                background = common_background
            elif split == "train" and len(puzzle["train"]) == 1:
                background = int(rng.choice([c for c in range(10)
                                             if c != first_background]))
            else:
                background = int(rng.integers(0, 10))
            forced_color = (first_background if common_background is None
                            and split == "train" and len(puzzle["train"]) == 1
                            else None)
            child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(child_seed)
            input_grid, output_grid = generate_pair(
                pair_rng, kind, shape, background, forced_color,
                hard=(split == "test" and rng.random() < 0.72),
                source_like=(split == "test" and kind == "corner"
                             and rng.random() < 0.48))
            pair = {"input": input_grid, "output": output_grid}
            key = _grid_key(input_grid)
            if key in used_inputs or _is_source_pair(input_grid, output_grid):
                continue
            used_inputs.add(key)
            puzzle[split].append(pair)
            if first_shape is None:
                first_shape = shape
            if first_background is None:
                first_background = background
            return
        raise RuntimeError("could not produce a distinct pair")

    for kind in training_kinds:
        add_pair("train", kind)
    for kind in test_kinds:
        add_pair("test", kind)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    used_inputs, used_pairs = set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
        assert input_key not in used_inputs and pair_key not in used_pairs
        used_inputs.add(input_key)
        used_pairs.add(pair_key)
        assert not _is_source_pair(pair["input"], pair["output"])
        _assert_pair_invariants(pair["input"], pair["output"])
    train_kinds = {_visible_kind(pair["output"]) for pair in puzzle["train"]}
    test_kinds = {_visible_kind(pair["output"]) for pair in puzzle["test"]}
    assert len(train_kinds) >= 2
    assert test_kinds <= train_kinds


def _canonical_mask_layout(output_grid):
    colors = [int(x) for x in np.unique(output_grid)]
    mask = tuple(int(x == colors[0]) for x in output_grid.ravel())
    return (tuple(output_grid.shape), min(mask, tuple(1 - x for x in mask)))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        assert all(_visible_kind(p["output"]) in
                   {_visible_kind(t["output"]) for t in puzzle["train"]}
                   for p in puzzle["test"])
    a = generate_puzzle(987654321, 7, 3)
    b = generate_puzzle(987654321, 7, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    assert len({_puzzle_key(p) for p in automatic}) >= 40
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1004, 5, 3), (1005, 9, 2)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        validate_generated_puzzle(puzzle)
    for seed, nt, ns in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ns is not None:
            assert len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11},
                    {"num_test": 0}, {"num_test": 5},
                    {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4},
                    {"num_train": 4, "num_test": 4},
                    {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(3001, **request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    shapes, layouts, canvases, palettes, backgrounds = set(), set(), set(), set(), set()
    kinds, test_kinds, axes = set(), set(), set()
    patch_sides, join_axes = set(), set()
    mixed_shapes = mixed_backgrounds = repeated_palettes = cross_role = 0
    large_patches = irregular_patches = thin_outputs = 0
    simple_joins = complex_joins = upper_right_test_l = 0
    for puzzle in automatic:
        pairs = puzzle["train"] + puzzle["test"]
        test_pair_ids = {id(pair) for pair in puzzle["test"]}
        puzzle_shapes = set()
        puzzle_backgrounds = set()
        color_pairs = []
        for pair in pairs:
            x, y = pair["input"], pair["output"]
            shape = tuple(y.shape)
            kind = _visible_kind(y)
            thin_outputs += min(shape) == 2
            if kind == "join":
                boxes = [_bbox_for_value(y, int(c)) for c in np.unique(y)]
                vertical = all(box[0] == 0 and box[2] == shape[0] - 1
                               for box in boxes)
                join_axes.add("vertical" if vertical else "horizontal")
                first_color = int(np.unique(y)[0])
                cuts = ([int(np.count_nonzero(y[row] == first_color))
                         for row in range(shape[0])] if vertical else
                        [int(np.count_nonzero(y[:, col] == first_color))
                         for col in range(shape[1])])
                counts = [cuts.count(value) for value in set(cuts)]
                simple_joins += len(set(cuts)) == 2 and min(counts) == 1
                complex_joins += (len(set(cuts)) >= 3 or
                                  sum(a != b for a, b in zip(cuts, cuts[1:])) >= 2)
            else:
                small_color = min((int(c) for c in np.unique(y)),
                                  key=lambda c: int(np.count_nonzero(y == c)))
                top, left, bottom, right = _bbox_for_value(y, small_color)
                patch_size = int(np.count_nonzero(y == small_color))
                large_patches += patch_size >= 4
                irregular_patches += patch_size < ((bottom - top + 1) *
                                                   (right - left + 1))
                if top == 0:
                    patch_sides.add("top")
                if bottom == shape[0] - 1:
                    patch_sides.add("bottom")
                if left == 0:
                    patch_sides.add("left")
                if right == shape[1] - 1:
                    patch_sides.add("right")
                if id(pair) in test_pair_ids:
                    upper_right_test_l += (patch_size == 3 and
                                           bottom - top == 1 and
                                           right - left == 1 and
                                           top == 0 and left == shape[1] - 2)
            bg = _background_color(x, y)
            color_pair = tuple(sorted(int(c) for c in np.unique(y)))
            shapes.add(shape)
            layouts.add(_canonical_mask_layout(y))
            canvases.add(tuple(x.shape))
            palettes.add(color_pair)
            backgrounds.add(bg)
            puzzle_shapes.add(shape)
            puzzle_backgrounds.add(bg)
            color_pairs.append(color_pair)
            axes.update(_input_layout_axes(x, color_pair))
        kinds.update(_visible_kind(pair["output"]) for pair in puzzle["train"])
        test_kinds.update(_visible_kind(pair["output"]) for pair in puzzle["test"])
        mixed_shapes += len(puzzle_shapes) > 1
        mixed_backgrounds += len(puzzle_backgrounds) > 1
        repeated_palettes += len(color_pairs) != len(set(color_pairs))
        cross_role += any(_background_color(a["input"], a["output"])
                          in set(int(c) for c in np.unique(b["output"]))
                          for a in pairs for b in pairs if a is not b)
    assert len(shapes) >= 12 and len(layouts) >= 50 and len(canvases) >= 20
    assert len(palettes) >= 20 and backgrounds == set(range(10))
    assert kinds == test_kinds == {"corner", "edge", "interior", "join"}
    assert axes == {"horizontal", "vertical"}
    assert patch_sides == {"top", "bottom", "left", "right"}
    assert join_axes == {"horizontal", "vertical"}
    assert large_patches >= 30 and irregular_patches >= 30
    assert thin_outputs >= 5 and upper_right_test_l >= 1
    assert simple_joins >= 5 and complex_joins >= 5
    assert mixed_shapes >= 15 and mixed_backgrounds >= 15
    assert repeated_palettes >= 1 and cross_role >= 15
