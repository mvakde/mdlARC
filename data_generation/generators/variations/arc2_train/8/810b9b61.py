"""Generate puzzles about exact, hollow rectangular ink components."""

import numpy as np


# The source inputs are kept here solely to reject exact source copies.
_SOURCE_INPUTS = (
    ("000000000000000", "000000000000000", "000000000000000",
     "000011100010000", "000010100010000", "000011100000000",
     "000000000000000", "000000000000000", "000010000101100",
     "000000000100100", "000000000111100", "000000000000000",
     "000000000000000", "000000000000000", "000000000000000"),
    ("000000000000000", "000000000011100", "001111000010100",
     "001001000010100", "001111000000000", "000000000000000",
     "000000000000110", "000100111100100", "000000100100000",
     "000000100100000", "111000111100000", "101000000000000",
     "101000000011110", "111001100010010", "000000000011110"),
    ("000000000", "000000000", "011111000", "010001000",
     "011111000", "000000000", "110000000", "010011000",
     "010000000"),
    ("000000011111", "011110010001", "010010010001",
     "011110010001", "000000011011", "000000000000",
     "011000000000", "000011111000", "010010001001",
     "000010001000", "000011111000", "000000000000"),
)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _perimeter(top, left, height, width):
    return frozenset(
        (row, col)
        for row in range(top, top + height)
        for col in range(left, left + width)
        if row in (top, top + height - 1) or col in (left, left + width - 1)
    )


def _components(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            cells = set()
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and not seen[nr, nc]
                            and int(grid[nr, nc]) != background):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(frozenset(cells))
    return result


def _box(cells):
    rows, cols = zip(*cells)
    top, left = min(rows), min(cols)
    return top, left, max(rows) - top + 1, max(cols) - left + 1


def _frames(grid, background):
    result = []
    for cells in _components(grid, background):
        top, left, height, width = _box(cells)
        if height >= 3 and width >= 3 and cells == _perimeter(top, left, height, width):
            result.append((cells, (top, left, height, width)))
    return result


def _one_missing(grid, background):
    result = []
    for cells in _components(grid, background):
        top, left, height, width = _box(cells)
        if height < 3 or width < 3:
            continue
        border = _perimeter(top, left, height, width)
        if cells.issubset(border) and len(border - cells) == 1:
            result.append((cells, (top, left, height, width)))
    return result


def _answer(grid, background, highlight):
    output = grid.copy()
    for cells, _ in _frames(grid, background):
        for row, col in cells:
            output[row, col] = highlight
    return output


def _same_size_training_specs(index, rng):
    if index == 0:
        # Positive and negative 5x5 examples, regardless of requested counts.
        extra = ((3, 3), (4, 4), (3, 6))[int(rng.integers(0, 3))]
        return [("frame", 5, 5, 0), ("broken", 5, 5, 1),
                ("frame", *extra, 0)]
    if index == 1:
        # The remaining two source-like sizes also have both outcomes.
        long_size = ((3, 8), (8, 3), (6, 9), (9, 6))[int(rng.integers(0, 4))]
        return [("frame", 4, 5, 0), ("broken", 4, 5, 1),
                ("frame", 3, 4, 0), ("broken", 3, 4, 1),
                ("frame", *long_size, 0)]
    return None


_SIZES = ((3, 3), (3, 4), (4, 4), (4, 5), (5, 5),
          (3, 6), (3, 8), (6, 9))
_WEIGHTS = (15, 18, 13, 18, 13, 9, 8, 6)


def _size(rng):
    index = int(rng.choice(len(_SIZES), p=np.array(_WEIGHTS) / sum(_WEIGHTS)))
    height, width = _SIZES[index]
    if height != width and bool(rng.integers(0, 2)):
        return width, height
    return height, width


def _random_specs(rng, regime):
    nframe = int(rng.integers(1, 5))
    nbroken = int(rng.integers(0, 4))
    # Some pairs contain only frames. The independent draws also permit
    # several near frames alongside a single positive frame.
    specs = [("frame", *_size(rng), 0) for _ in range(nframe)]
    for _ in range(nbroken):
        height, width = _size(rng)
        missing = int(rng.choice((1, 1, 1, 2, 3, 4)))
        specs.append(("broken", height, width, missing))
    if regime == "target_test" and bool(rng.integers(0, 3) == 0):
        # Test may use the simplest regime already available in training.
        specs = specs[:nframe]
    return specs


def _overlap_with_halo(a, b):
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    return not (at + ah <= bt - 1 or bt + bh <= at - 1
                or al + aw <= bl - 1 or bl + bw <= al - 1)


def _place_boxes(rng, height, width, specs):
    # Largest first improves packing; object roles do not determine position.
    order = sorted(range(len(specs)), key=lambda i: specs[i][1] * specs[i][2], reverse=True)
    boxes = [None] * len(specs)
    occupied = []
    for i in order:
        _, h, w, _ = specs[i]
        if h > height or w > width:
            return None
        candidates = [(r, c, h, w) for r in range(height - h + 1)
                      for c in range(width - w + 1)]
        # Boundary and interior locations are sampled for either label.
        prefer_boundary = bool(rng.integers(0, 2))
        boundary = [box for box in candidates if box[0] == 0 or box[1] == 0
                    or box[0] + h == height or box[1] + w == width]
        interior = [box for box in candidates if box not in boundary]
        groups = (boundary, interior) if prefer_boundary else (interior, boundary)
        chosen = None
        for group in groups:
            for j in rng.permutation(len(group)):
                box = group[int(j)]
                if all(not _overlap_with_halo(box, old) for old in occupied):
                    chosen = box
                    break
            if chosen is not None:
                break
        if chosen is None:
            return None
        boxes[i] = chosen
        occupied.append(chosen)
    return boxes


def _fragment_cells(kind, top, left, height, width):
    if kind == "block":
        return {(r, c) for r in range(top, top + height)
                for c in range(left, left + width)}
    if kind == "line":
        return {(top + r, left + c) for r in range(height) for c in range(width)}
    if kind == "L":
        return {(top + r, left) for r in range(height)} | {
            (top + height - 1, left + c) for c in range(width)}
    return {(top, left + c) for c in range(width)} | {
        (top + r, left + width // 2) for r in range(height)}


def _draw_scene(rng, shape, specs, background, ink, fragment_count):
    height, width = shape
    boxes = _place_boxes(rng, height, width, specs)
    if boxes is None:
        return None
    grid = np.full(shape, background, dtype=np.int8)
    for (kind, _, _, missing), (top, left, h, w) in zip(specs, boxes):
        cells = sorted(_perimeter(top, left, h, w))
        if kind == "broken":
            # One-cell omissions cover sides and corners; larger omissions
            # may form more open C/U outlines or multiple gaps.
            if missing == 1:
                omitted = {int(rng.integers(0, len(cells)))}
            elif bool(rng.integers(0, 2)):
                start = int(rng.integers(0, len(cells) - missing + 1))
                omitted = set(range(start, start + missing))
            else:
                omitted = {int(v) for v in rng.choice(len(cells),
                            size=min(missing, len(cells) - 3), replace=False)}
            cells = [cell for j, cell in enumerate(cells) if j not in omitted]
        for row, col in cells:
            grid[row, col] = ink

    motifs = (("line", 1, 1), ("line", 1, 2), ("line", 1, 4),
              ("line", 4, 1), ("L", 3, 3), ("T", 3, 3),
              ("block", 2, 2), ("block", 3, 3))
    for _ in range(fragment_count):
        kind, h, w = motifs[int(rng.integers(0, len(motifs)))]
        if kind in ("line", "L", "T") and bool(rng.integers(0, 2)):
            h, w = w, h
        candidates = [(r, c, h, w) for r in range(height - h + 1)
                      for c in range(width - w + 1)]
        for j in rng.permutation(len(candidates)):
            box = candidates[int(j)]
            if all(not _overlap_with_halo(box, old) for old in boxes):
                boxes.append(box)
                for row, col in _fragment_cells(kind, *box):
                    grid[row, col] = ink
                break
    return grid


def _sample_shape(rng, specs, attempt):
    if attempt >= 35:
        return (30, 30)
    tier = int(rng.choice(3, p=(0.3, 0.45, 0.25)))
    ranges = ((9, 16), (15, 24), (22, 31))
    low, high = ranges[tier]
    min_h = max(spec[1] for spec in specs)
    min_w = max(spec[2] for spec in specs)
    return (int(rng.integers(max(low, min_h), high)),
            int(rng.integers(max(low, min_w), high)))


def _make_input(rng, regime, train_index, background, ink):
    forced = _same_size_training_specs(train_index, rng) if regime == "source_train" else None
    for attempt in range(90):
        specs = forced if forced is not None else _random_specs(rng, regime)
        height, width = _sample_shape(rng, specs, attempt)
        fragments = int(rng.integers(0, 4)) if forced is None else int(rng.integers(0, 3))
        grid = _draw_scene(rng, (height, width), specs, background, ink, fragments)
        if grid is None:
            continue
        # A unique, dominant background makes input roles visible.
        if int(np.count_nonzero(grid == background)) <= grid.size // 2:
            continue
        if len(_frames(grid, background)) != sum(kind == "frame" for kind, *_ in specs):
            continue
        return grid
    raise RuntimeError("could not pack a separated scene")


def _key(grid):
    return (grid.shape, bytes(grid.astype(np.uint8).ravel()))


def _is_source_input(grid):
    if set(int(v) for v in np.unique(grid)) != {0, 1}:
        return False
    rows = tuple("".join(str(int(v)) for v in row) for row in grid)
    return rows in _SOURCE_INPUTS


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    highlight = int(rng.integers(0, 10))
    colors = [color for color in range(10) if color != highlight]
    stable_palette = bool(rng.integers(0, 3) == 0)
    pair_colors = tuple(int(v) for v in rng.choice(colors, size=2, replace=False))
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for global_index in range(train_count + test_count):
        regime = "source_train" if global_index < train_count else "target_test"
        section = "train" if regime == "source_train" else "test"
        local_index = global_index if regime == "source_train" else global_index - train_count
        if not stable_palette and global_index:
            # The highlight is a learned puzzle convention; input colors are
            # pair-level appearances, and can change even within training.
            choices = [pair for pair in ((a, b) for a in colors for b in colors if a != b)
                       if pair != pair_colors]
            pair_colors = choices[int(rng.integers(0, len(choices)))]
        background, ink = pair_colors
        for attempt in range(20):
            seq = np.random.SeedSequence((int(pair_seeds[global_index]) & 0xffffffff,
                                          int(pair_seeds[global_index]) >> 32,
                                          global_index, attempt))
            pair_rng = np.random.default_rng(seq)
            grid = _make_input(pair_rng, regime, local_index, background, ink)
            key = _key(grid)
            if key in seen or _is_source_input(grid):
                continue
            seen.add(key)
            puzzle[section].append({"input": grid, "output": _answer(grid, background, highlight)})
            break
        else:
            raise RuntimeError("could not generate distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _input_roles(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    index = int(np.argmax(counts))
    assert counts[index] > grid.size // 2
    return int(values[index]), int(values[1 - index])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    highlights = set()
    train_frame_sizes = set()
    train_broken_sizes = set()
    train_frame_counts = set()
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((0 <= array) & (array <= 9))
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            assert not _is_source_input(grid)
            background, ink = _input_roles(grid)
            output_colors = set(int(v) for v in np.unique(output))
            assert background in output_colors
            new = output_colors - {background, ink}
            assert len(new) == 1
            highlight = next(iter(new))
            highlights.add(highlight)
            frames = _frames(grid, background)
            assert frames
            assert np.array_equal(output, _answer(grid, background, highlight))
            assert all(grid[r, c] == ink for cells, _ in frames for r, c in cells)
            # Every qualifying frame is isolated from other ink, including
            # at diagonal corners. Broken outlines may have several pieces.
            owner = {cell: i for i, cells in enumerate(_components(grid, background)) for cell in cells}
            for cells, _ in frames:
                for r, c in cells:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if (dr or dc) and (r + dr, c + dc) in owner:
                                assert (r + dr, c + dc) in cells
            if section == "train":
                train_frame_counts.add(len(frames))
                train_frame_sizes.update(tuple(sorted(box[2:])) for _, box in frames)
                train_broken_sizes.update(tuple(sorted(box[2:])) for _, box in _one_missing(grid, background))
    assert len(highlights) == 1
    # Both outcomes at identical sizes are demonstrated before testing.
    for size in ((3, 4), (4, 5), (5, 5)):
        assert size in train_frame_sizes and size in train_broken_sizes
    assert {2, 3}.issubset(train_frame_counts)


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(54321)
    assert _equal(a, generate_puzzle(54321))
    assert not _equal(a, generate_puzzle(54322))
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4),
                          (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (ntrain or 0) * 10 + (ntest or 0), ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kwargs}")

    train_counts, test_counts, shapes, frame_counts = set(), set(), set(), set()
    frame_sizes, broken_sizes, palettes, layouts = set(), set(), set(), set()
    boundary_frame = boundary_broken = only_frames = 0
    long_frame = large_frame = same_palette = changing_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        backgrounds = set()
        for pair in puzzle["train"] + puzzle["test"]:
            grid, output = pair["input"], pair["output"]
            background, ink = _input_roles(grid)
            backgrounds.add((background, ink))
            highlight = next(iter(set(int(v) for v in np.unique(output)) - {background, ink}))
            palettes.add((background, ink, highlight))
            shapes.add(grid.shape)
            layouts.add(_key((grid != background).astype(np.int8)))
            frames = _frames(grid, background)
            broken = _one_missing(grid, background)
            frame_counts.add(len(frames))
            frame_sizes.update(tuple(sorted(box[2:])) for _, box in frames)
            broken_sizes.update(tuple(sorted(box[2:])) for _, box in broken)
            if len(_components(grid, background)) == len(frames):
                only_frames += 1
            for _, (top, left, h, w) in frames:
                boundary_frame += int(top == 0 or left == 0 or top + h == grid.shape[0] or left + w == grid.shape[1])
                long_frame += int(min(h, w) == 3 and max(h, w) >= 8)
                large_frame += int(min(h, w) >= 6)
            for _, (top, left, h, w) in broken:
                boundary_broken += int(top == 0 or left == 0 or top + h == grid.shape[0] or left + w == grid.shape[1])
        same_palette += int(len(backgrounds) == 1)
        changing_palette += int(len(backgrounds) > 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(layouts) >= 150 and len(palettes) >= 20
    assert frame_counts == {1, 2, 3, 4}
    assert {(3, 4), (4, 5), (5, 5), (3, 8), (6, 9)}.issubset(frame_sizes)
    assert {(3, 4), (4, 5), (5, 5)}.issubset(broken_sizes)
    assert boundary_frame >= 20 and boundary_broken >= 10
    assert only_frames >= 5 and long_frame >= 10 and large_frame >= 5
    assert same_palette >= 5 and changing_palette >= 5
