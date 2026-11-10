import numpy as np


_SOURCE_ROWS = (
    ("0000660000", "0000000000", "0000000000", "0002379518",
     "4000000000", "4000000000", "0000000000", "0000000000",
     "0000000000", "0000000000"),
    ("0000770000", "0000000000", "0000000000", "0000000000",
     "0000800009", "0000400009", "0000600000", "0000300000",
     "0000200000", "0000100000"),
    ("0000", "8030", "8020", "0040"),
    ("00099000", "00000000", "00000000", "00200007",
     "00400007", "00800000", "00600000", "00300000"),
)
_SOURCE_INPUTS = tuple(
    np.array([[int(digit) for digit in row] for row in rows], dtype=np.int8)
    for rows in _SOURCE_ROWS
)
_DIRECTIONS = {"up": (-1, 0), "down": (1, 0),
               "left": (0, -1), "right": (0, 1)}


def _components(scaffold, background):
    height, width = scaffold.shape
    labels = np.full((height, width), -1, dtype=np.int16)
    sizes = []
    for row in range(height):
        for col in range(width):
            if scaffold[row, col] != background or labels[row, col] >= 0:
                continue
            label = len(sizes)
            labels[row, col] = label
            stack = [(row, col)]
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for nr, nc in ((r - 1, c), (r + 1, c),
                               (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and scaffold[nr, nc] == background
                            and labels[nr, nc] < 0):
                        labels[nr, nc] = label
                        stack.append((nr, nc))
            sizes.append(size)
    return labels, sizes


def _scaffold(shape, spine, background, backward_count=2):
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    for row, col, color in spine:
        grid[row, col] = color
    dr = spine[-1][0] - spine[-2][0]
    dc = spine[-1][1] - spine[-2][1]
    for index, (row, col, color) in enumerate(spine):
        for forward in (1, -1) if index < backward_count else (1,):
            for side in (-1, 1):
                sr = forward * dr + side * dc
                sc = forward * dc - side * dr
                r, c = row + sr, col + sc
                while 0 <= r < height and 0 <= c < width:
                    assert grid[r, c] in (background, color)
                    grid[r, c] = color
                    r += sr
                    c += sc
    return grid


def _domino_candidates(labels, sizes, minimum_size=4):
    height, width = labels.shape
    candidates = {}
    for row in range(height):
        for col in range(width):
            label = int(labels[row, col])
            if label < 0 or sizes[label] < minimum_size:
                continue
            for nr, nc in ((row + 1, col), (row, col + 1)):
                if nr >= height or nc >= width or labels[nr, nc] != label:
                    continue
                boundary = (row == 0 or col == 0 or nr == height - 1
                            or nc == width - 1)
                candidates.setdefault(label, []).append(
                    (((row, col), (nr, nc)), boundary)
                )
    return candidates


def _later_backward_would_add(scaffold, spine, background):
    height, width = scaffold.shape
    dr = spine[-1][0] - spine[-2][0]
    dc = spine[-1][1] - spine[-2][1]
    for row, col, _ in spine[2:]:
        for side in (-1, 1):
            r, c = row - dr + side * dc, col - dc - side * dr
            while 0 <= r < height and 0 <= c < width:
                if scaffold[r, c] == background:
                    return True
                r += -dr + side * dc
                c += -dc - side * dr
    return False


def _resolve_counts(rng, num_train, num_test):
    for value in (num_train, num_test):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))):
            raise ValueError("Pair counts must be integers")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _pick_shape(rng, scale):
    if scale == "compact":
        return int(rng.integers(4, 7)), int(rng.integers(4, 8))
    if scale == "medium":
        return int(rng.integers(7, 13)), int(rng.integers(7, 14))
    return int(rng.integers(12, 20)), int(rng.integers(12, 21))


def _make_pair(rng, direction, scale, desired_markers, demonstration):
    dr, dc = _DIRECTIONS[direction]
    for _ in range(240):
        height, width = _pick_shape(rng, scale)
        axis = height if dr else width
        perpendicular = width if dr else height
        maximum_length = min(7, axis - (3 if demonstration else 1))
        if maximum_length < 3:
            continue
        length = int(rng.integers(3, maximum_length + 1))
        if desired_markers == 3 and length > 6:
            continue
        cross = int(rng.integers(1, perpendicular - 1))
        if direction == "down":
            positions = [(height - length + i, cross) for i in range(length)]
        elif direction == "up":
            positions = [(length - 1 - i, cross) for i in range(length)]
        elif direction == "right":
            positions = [(cross, width - length + i) for i in range(length)]
        else:
            positions = [(cross, length - 1 - i) for i in range(length)]
        palette = [int(value) for value in rng.permutation(10)]
        background = palette[0]
        spine = [(row, col, palette[i + 1])
                 for i, (row, col) in enumerate(positions)]
        scaffold = _scaffold((height, width), spine, background)
        labels, sizes = _components(scaffold, background)
        candidates = _domino_candidates(labels, sizes)
        if len(candidates) < desired_markers:
            continue
        if demonstration:
            forward_only = _scaffold((height, width), spine, background, 0)
            if (np.array_equal(scaffold, forward_only)
                    or not _later_backward_would_add(scaffold, spine,
                                                     background)):
                continue
        component_ids = [int(v) for v in rng.permutation(list(candidates))
                         [:desired_markers]]
        marker_colors = palette[length + 1:length + 1 + desired_markers]
        input_grid = np.full((height, width), background, dtype=np.int8)
        output_grid = scaffold.copy()
        for row, col, color in spine:
            input_grid[row, col] = color
        for label, color in zip(component_ids, marker_colors):
            choices = candidates[label]
            want_boundary = bool(rng.integers(2))
            preferred = [entry for entry in choices if entry[1] == want_boundary]
            cells, _ = preferred[int(rng.integers(len(preferred)))] if preferred \
                else choices[int(rng.integers(len(choices)))]
            for row, col in cells:
                input_grid[row, col] = color
            output_grid[labels == label] = color
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("Could not construct a seeded diagonal scaffold")


def _pair_key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # The source transformation is the only puzzle-level convention. Each
    # pair's input identifies its color roles; no hidden map is shared.
    horizontal = "left" if rng.integers(2) else "right"
    vertical = "up" if rng.integers(2) else "down"
    first_directions = [horizontal, vertical]
    if rng.integers(2):
        first_directions.reverse()
    train_directions = first_directions + [
        tuple(_DIRECTIONS)[int(rng.integers(4))]
        for _ in range(train_count - 2)
    ]
    seen_directions = set(train_directions)
    unseen_directions = [d for d in _DIRECTIONS if d not in seen_directions]
    test_directions = []
    for _ in range(test_count):
        if unseen_directions and rng.random() < 0.45:
            test_directions.append(unseen_directions[int(
                rng.integers(len(unseen_directions)))])
        else:
            test_directions.append(tuple(_DIRECTIONS)[int(rng.integers(4))])

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, directions in (("train", train_directions),
                              ("test", test_directions)):
        for index, direction in enumerate(directions):
            demonstration = split == "train" and index < 2
            if demonstration:
                scale = "medium" if rng.integers(2) else "large"
                desired_markers = 2
            else:
                scales = ("compact", "medium", "large")
                scale = scales[int(rng.choice(3, p=(0.18, 0.46, 0.36)
                                              if split == "train"
                                              else (0.10, 0.38, 0.52)))]
                desired_markers = int(rng.choice((1, 2, 3),
                                                 p=(0.30, 0.42, 0.28)))
                if scale == "compact":
                    desired_markers = min(2, desired_markers)
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(30):
                pair = _make_pair(pair_rng, direction, scale,
                                  desired_markers, demonstration)
                key = _pair_key(pair["input"])
                if key not in seen_inputs and all(
                    not np.array_equal(pair["input"], original)
                    for original in _SOURCE_INPUTS
                ):
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(input_grid):
    height, width = input_grid.shape
    counts = np.bincount(input_grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] > 2
    colors = [color for color in range(10)
              if color != background and counts[color] > 0]
    assert all(counts[color] in (1, 2) for color in colors)
    spine_colors = [color for color in colors if counts[color] == 1]
    marker_colors = [color for color in colors if counts[color] == 2]
    assert 3 <= len(spine_colors) <= 7
    assert 1 <= len(marker_colors) <= 3
    coordinates = [tuple(map(int, np.argwhere(input_grid == color)[0]))
                   for color in spine_colors]
    rows = {row for row, _ in coordinates}
    cols = {col for _, col in coordinates}
    if len(rows) == 1:
        assert len(cols) == len(coordinates)
        assert 0 < next(iter(rows)) < height - 1
        ordered = sorted(coordinates, key=lambda cell: cell[1])
        assert [cell[1] for cell in ordered] == list(
            range(ordered[0][1], ordered[-1][1] + 1))
        low_edge = ordered[0][1] == 0
        high_edge = ordered[-1][1] == width - 1
        direction = "left" if low_edge else "right"
    else:
        assert len(cols) == 1 and len(rows) == len(coordinates)
        assert 0 < next(iter(cols)) < width - 1
        ordered = sorted(coordinates, key=lambda cell: cell[0])
        assert [cell[0] for cell in ordered] == list(
            range(ordered[0][0], ordered[-1][0] + 1))
        low_edge = ordered[0][0] == 0
        high_edge = ordered[-1][0] == height - 1
        direction = "up" if low_edge else "down"
    assert low_edge != high_edge
    if low_edge:
        ordered.reverse()
    spine = [(row, col, int(input_grid[row, col])) for row, col in ordered]
    scaffold = _scaffold(input_grid.shape, spine, background)
    labels, sizes = _components(scaffold, background)
    expected = scaffold.copy()
    markers = []
    seeded = set()
    for color in marker_colors:
        cells = [tuple(map(int, cell)) for cell in np.argwhere(input_grid == color)]
        assert (abs(cells[0][0] - cells[1][0])
                + abs(cells[0][1] - cells[1][1]) == 1)
        assert all(scaffold[cell] == background for cell in cells)
        label = int(labels[cells[0]])
        assert label >= 0 and labels[cells[1]] == label
        assert label not in seeded and sizes[label] >= 4
        seeded.add(label)
        expected[labels == label] = color
        markers.append((color, cells, label))
    assert not np.array_equal(input_grid, expected)
    return {"background": background, "direction": direction,
            "spine": spine, "markers": markers, "labels": labels,
            "sizes": sizes, "expected": expected}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_details = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _pair_key(inp)
            pair_key = (key, out.astype(np.uint8).tobytes())
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert all(not np.array_equal(inp, original)
                       for original in _SOURCE_INPUTS)
            details = _inspect_pair(inp)
            assert np.array_equal(out, details["expected"])
            if split == "train":
                train_details.append(details)

    assert any(d["direction"] in ("left", "right") for d in train_details)
    assert any(d["direction"] in ("up", "down") for d in train_details)
    assert sum(len(d["markers"]) >= 2 for d in train_details) >= 2
    assert any(not np.array_equal(
        _scaffold(d["labels"].shape, d["spine"], d["background"], 0),
        _scaffold(d["labels"].shape, d["spine"], d["background"], 2))
        for d in train_details)
    assert any(_later_backward_would_add(
        _scaffold(d["labels"].shape, d["spine"], d["background"], 2),
        d["spine"], d["background"]) for d in train_details)


def _equal_puzzles(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(one[key], two[key])
        for one, two in zip(first[split], second[split])
        for key in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    directions, sizes, marker_counts, backgrounds = set(), set(), set(), set()
    orientations, sites, shapes, layouts = set(), set(), set(), set()
    interior = boundary = three_markers = role_changes = 0
    unseen_test_directions = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _equal_puzzles(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_directions = set()
        pair_roles = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                details = _inspect_pair(grid)
                assert np.array_equal(details["expected"], pair["output"])
                direction = details["direction"]
                if split == "train":
                    train_directions.add(direction)
                directions.add(direction)
                sizes.add(grid.shape)
                backgrounds.add(details["background"])
                marker_counts.add(len(details["markers"]))
                shapes.add((grid.shape[0] // 3, grid.shape[1] // 3))
                layouts.add((grid.shape, direction, len(details["spine"]),
                             details["spine"][0][:2]))
                spine_colors = {color for _, _, color in details["spine"]}
                marker_colors = {color for color, _, _ in details["markers"]}
                pair_roles.append((spine_colors, marker_colors))
                if len(details["markers"]) == 3:
                    three_markers += 1
                for _, cells, _ in details["markers"]:
                    is_boundary = any(r in (0, grid.shape[0] - 1)
                                      or c in (0, grid.shape[1] - 1)
                                      for r, c in cells)
                    boundary += is_boundary
                    interior += not is_boundary
                    orientations.add("vertical" if cells[0][1] == cells[1][1]
                                     else "horizontal")
                    sites.add((grid.shape, tuple(cells)))
            if split == "test":
                unseen_test_directions += any(
                    _inspect_pair(pair["input"])["direction"]
                    not in train_directions for pair in puzzle["test"])
        role_changes += any(a[1] & b[0] or a[0] & b[1]
                            for i, a in enumerate(pair_roles)
                            for b in pair_roles[i + 1:])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert directions == set(_DIRECTIONS)
    assert len(sizes) >= 60 and len(shapes) >= 12
    assert len(layouts) >= 100 and len(sites) >= 100
    assert marker_counts == {1, 2, 3}
    assert len(backgrounds) >= 8 and backgrounds - {0}
    assert orientations == {"vertical", "horizontal"}
    assert interior >= 25 and boundary >= 25
    assert three_markers >= 10 and role_changes >= 25
    assert unseen_test_directions >= 5

    for ntrain, ntest in ((2, 1), (10, 1), (10, 4),
                          (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(103, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert _equal_puzzles(puzzle, generate_puzzle(103, ntrain, ntest))
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
    for ntrain, ntest in ((1, None), (11, None), (None, 0),
                          (None, 5), (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(103, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
