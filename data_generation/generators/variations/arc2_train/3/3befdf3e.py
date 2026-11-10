"""Generate framed-square ARC puzzles with local colors and full plus expansions."""

import numpy as np


def _rows(*lines):
    return np.array([[int(cell) for cell in line] for line in lines], dtype=np.int8)


_SOURCE_PAIRS = (
    (
        _rows("0000000000", "0000000000", "0000000000", "0002222000",
              "0002772000", "0002772000", "0002222000", "0000000000",
              "0000000000", "0000000000"),
        _rows("0000000000", "0002222000", "0002222000", "0227777220",
              "0227227220", "0227227220", "0227777220", "0002222000",
              "0002222000", "0000000000"),
    ),
    (
        _rows("0000000000", "0000000000", "0000000000", "0004440000",
              "0004640000", "0004440000", "0000000000", "0000000000",
              "0000000000", "0000000000"),
        _rows("0000000000", "0000000000", "0004440000", "0046664000",
              "0046464000", "0046664000", "0004440000", "0000000000",
              "0000000000", "0000000000"),
    ),
    (
        _rows("0000000000", "0000000000", "0000000000", "0033330000",
              "0031130000", "0031130000", "0033330000", "0000000000",
              "0000000000", "0000000000"),
        _rows("0000000000", "0033330000", "0033330000", "3311113300",
              "3313313300", "3313313300", "3311113300", "0033330000",
              "0033330000", "0000000000"),
    ),
    (
        _rows("000000000000", "088800000000", "083800000000",
              "088800000000", "000000000000", "000000000000",
              "000000888800", "000000833800", "000000833800",
              "000000888800", "000000000000", "000000000000"),
        _rows("088800000000", "833380000000", "838380000000",
              "833380000000", "088800888800", "000000888800",
              "000088333388", "000088388388", "000088388388",
              "000088333388", "000000888800", "000000888800"),
    ),
)


def _count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid_key(grid):
    grid = np.asarray(grid, dtype=np.int16)
    return grid.shape, grid.tobytes()


def _is_source_pair(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def _mask(shape, top, left, k):
    """Exact painted plus, including its original (k+2)-square footprint."""
    size = k + 2
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + size, left:left + size] = True
    mask[top - k:top, left:left + size] = True
    mask[top + size:top + size + k, left:left + size] = True
    mask[top:top + size, left - k:left] = True
    mask[top:top + size, left + size:left + size + k] = True
    return mask


def _inspect_input(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    assert np.all(grid[0] == background)
    assert np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background)
    assert np.all(grid[:, -1] == background)
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == background:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width and
                            not seen[nr, nc] and int(grid[nr, nc]) != background):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            top = min(r for r, _ in cells)
            bottom = max(r for r, _ in cells)
            left = min(c for _, c in cells)
            right = max(c for _, c in cells)
            size = bottom - top + 1
            assert size >= 3 and right - left + 1 == size
            assert len(cells) == size * size
            patch = grid[top:bottom + 1, left:right + 1]
            frame = int(patch[0, 0])
            center = int(patch[1, 1])
            assert len({background, frame, center}) == 3
            assert np.all(patch[0] == frame) and np.all(patch[-1] == frame)
            assert np.all(patch[:, 0] == frame) and np.all(patch[:, -1] == frame)
            assert np.all(patch[1:-1, 1:-1] == center)
            k = size - 2
            assert top >= k and left >= k
            assert bottom + k < height and right + k < width
            objects.append({"top": top, "left": left, "k": k, "size": size,
                            "frame": frame, "center": center})
    assert objects
    occupied = np.zeros(grid.shape, dtype=bool)
    for obj in objects:
        painted = _mask(grid.shape, obj["top"], obj["left"], obj["k"])
        assert not np.any(occupied & painted)
        occupied |= painted
    return {"background": background,
            "objects": sorted(objects, key=lambda x: (x["top"], x["left"]))}


def _transform(inp, inspection):
    out = inp.copy()
    for obj in inspection["objects"]:
        top, left, k = obj["top"], obj["left"], obj["k"]
        size = k + 2
        frame, center = obj["frame"], obj["center"]
        out[top:top + size, left:left + size] = center
        out[top + 1:top + size - 1, left + 1:left + size - 1] = frame
        out[top - k:top, left:left + size] = frame
        out[top + size:top + size + k, left:left + size] = frame
        out[top:top + size, left - k:left] = frame
        out[top:top + size, left + size:left + size + k] = frame
    return out


def _pick_sizes(rng, count, required_k, mode, train_max):
    if required_k is not None:
        sizes = [required_k]
    else:
        sizes = []
    while len(sizes) < count:
        if count == 1:
            size = int(rng.integers(1, min(9, train_max + 1) + 1))
        elif count == 2:
            size = int(rng.choice([1, 2, 3, 4], p=[.32, .34, .24, .10]))
        else:
            size = int(rng.choice([1, 2, 3], p=[.47, .39, .14]))
        if mode in ("row", "column") and count == 4:
            size = min(size, 2)
        sizes.append(size)
    rng.shuffle(sizes)
    return sizes


def _dimensions(rng, sizes, mode):
    count = len(sizes)
    minimum = 3 * max(sizes) + 2
    if count == 1:
        lower_h = lower_w = minimum
    elif mode == "row":
        lower_h, lower_w = minimum, max(19, minimum, 5 * count + 3)
    elif mode == "column":
        lower_h, lower_w = max(19, minimum, 5 * count + 3), minimum
    else:
        lower_h = lower_w = max(16 if count == 2 else 19, minimum)
    if count == 1 and rng.random() < .35:
        height, width = lower_h, lower_w
    elif count == 1 and rng.random() < .27:
        height = min(30, lower_h + int(rng.integers(0, 4)))
        width = min(30, lower_w + int(rng.integers(0, 4)))
    else:
        height = int(rng.integers(lower_h, 31))
        width = int(rng.integers(lower_w, 31))
    if rng.random() < .09:
        if rng.random() < .5:
            height = 30
        else:
            width = 30
    return height, width


def _fractions(rng, count, mode):
    if count == 1:
        positions = [(float(rng.random()), float(rng.random()))]
    elif mode == "row":
        y = float(rng.random())
        positions = [(y, (i + .5) / count) for i in range(count)]
    elif mode == "column":
        x = float(rng.random())
        positions = [((i + .5) / count, x) for i in range(count)]
    elif mode == "grid":
        positions = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)][:count]
    else:
        positions = [(float(rng.random()), float(rng.random()))
                     for _ in range(count)]
    rng.shuffle(positions)
    # Exact extremes produce expansions touching one or more outer edges.
    if mode == "free" or count == 1:
        positions = [((float(rng.integers(0, 2)) if rng.random() < .27 else y),
                      (float(rng.integers(0, 2)) if rng.random() < .27 else x))
                     for y, x in positions]
    return positions


def _place(rng, height, width, sizes, mode):
    positions = _fractions(rng, len(sizes), mode)
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i])
    placed = {}
    occupied = np.zeros((height, width), dtype=bool)
    for index in order:
        k = sizes[index]
        last_top = height - (3 * k + 2)
        last_left = width - (3 * k + 2)
        y, x = positions[index]
        target_top = k + y * last_top
        target_left = k + x * last_left
        candidates = [(r, c) for r in range(k, k + last_top + 1)
                      for c in range(k, k + last_left + 1)]
        rng.shuffle(candidates)
        candidates.sort(key=lambda rc: abs(rc[0] - target_top) +
                        abs(rc[1] - target_left))
        chosen = None
        for top, left in candidates:
            size = k + 2
            # Keep at least one background cell between input footprints.
            if any(top <= old_top + old_k + 2 and
                   top + size >= old_top and
                   left <= old_left + old_k + 2 and
                   left + size >= old_left
                   for old_top, old_left, old_k in placed.values()):
                continue
            painted = _mask((height, width), top, left, k)
            if np.any(painted & occupied):
                continue
            chosen = (top, left, k)
            occupied |= painted
            break
        if chosen is None:
            return None
        placed[index] = chosen
    return [placed[i] for i in range(len(sizes))]


def _palettes(rng, count, background):
    available = [color for color in range(10) if color != background]
    first = tuple(int(v) for v in rng.choice(available, 2, replace=False))
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return [first] * count
    palettes = [first]
    if mode == 1 and count > 1:
        palettes.append((first[1], first[0]))
    while len(palettes) < count:
        choices = tuple(int(v) for v in rng.choice(available, 2, replace=False))
        if count > 1 and len(palettes) == 1 and choices == first:
            continue
        palettes.append(choices)
    return palettes


def _make_pair(rng, count, required_k, train_max, used_inputs,
               prior_background, require_slack=False):
    modes = ("free", "row", "column", "grid")
    for attempt in range(100):
        mode = "free" if count == 1 else modes[int(rng.integers(0, 4))]
        sizes = _pick_sizes(rng, count, required_k, mode, train_max)
        height, width = _dimensions(rng, sizes, mode)
        if require_slack:
            height = max(height, 3 * max(sizes) + 3)
        placements = _place(rng, height, width, sizes, mode)
        if placements is None:
            continue
        backgrounds = [v for v in range(10) if v != prior_background]
        background = int(rng.choice(backgrounds))
        colors = _palettes(rng, count, background)
        inp = np.full((height, width), background, dtype=np.int8)
        for (top, left, k), (frame, center) in zip(placements, colors):
            size = k + 2
            inp[top:top + size, left:left + size] = frame
            inp[top + 1:top + size - 1, left + 1:left + size - 1] = center
        inspected = _inspect_input(inp)
        assert len(inspected["objects"]) == count
        out = _transform(inp, inspected)
        if _grid_key(inp) in used_inputs or _is_source_pair(inp, out):
            continue
        return inp, out, background
    raise RuntimeError("unable to place separate, full plus expansions")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Geometry and local color roles are the whole shared convention. No hidden
    # palette, orientation, or split-specific rule is sampled.
    large_k = int(rng.choice([3, 4, 5, 6, 7, 8, 9],
                             p=[.31, .24, .17, .11, .08, .05, .04]))
    small_k = int(rng.integers(1, 3))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    previous_background = None
    for split, total in (("train", train_count), ("test", test_count)):
        for ordinal in range(total):
            if split == "train" and ordinal < 2:
                count = 1
                required = small_k if ordinal == 0 else large_k
            else:
                probabilities = ([.42, .32, .19, .07] if split == "train"
                                 else [.19, .40, .28, .13])
                count = int(rng.choice([1, 2, 3, 4], p=probabilities))
                required = None
            inp, out, previous_background = _make_pair(
                rng, count, required, large_k, used_inputs,
                previous_background, split == "train" and ordinal == 1)
            used_inputs.add(_grid_key(inp))
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_sizes = set()
    test_sizes = set()
    demonstrates_stopped_arm = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert not _is_source_pair(inp, out)
            inspected = _inspect_input(inp)
            assert np.array_equal(out, _transform(inp, inspected))
            objects = inspected["objects"]
            assert 1 <= len(objects) <= 4
            for obj in objects:
                (train_sizes if split == "train" else test_sizes).add(obj["k"])
                if split == "train" and (
                    inp.shape[0] > 3 * obj["k"] + 2 or
                    inp.shape[1] > 3 * obj["k"] + 2
                ):
                    demonstrates_stopped_arm = True
                # The plus mask contains one center footprint and four full arms.
                expected_area = (obj["k"] + 2) ** 2 + 4 * obj["k"] * (obj["k"] + 2)
                assert int(_mask(inp.shape, obj["top"], obj["left"],
                                 obj["k"]).sum()) == expected_area
    assert any(k <= 2 for k in train_sizes)
    assert any(k >= 3 for k in train_sizes)
    assert demonstrates_stopped_arm
    assert max(test_sizes) <= max(train_sizes) + 1


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[split], second[split])
                   for role in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    train_objects, test_objects = set(), set()
    sizes, shapes, palettes = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _inspect_input(inp)
                (train_objects if split == "train" else test_objects).add(
                    len(info["objects"]))
                shapes.add(inp.shape)
                for obj in info["objects"]:
                    sizes.add(obj["k"])
                    palettes.add((info["background"], obj["frame"], obj["center"]))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(train_objects) >= 3 and len(test_objects) == 4
    assert set(range(1, 6)).issubset(sizes)
    assert len(shapes) >= 30 and len(palettes) >= 40
    for seed, train, test in ((91357, 7, 3), (2000, 2, 1),
                              (2001, 10, 1), (2002, 10, 4),
                              (3000, None, 1), (3001, None, 4),
                              (3002, 2, None), (3003, 10, None)):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        assert _puzzles_equal(first, second)
        if train is not None:
            assert len(first["train"]) == train
        if test is not None:
            assert len(first["test"]) == test
    assert not _puzzles_equal(generate_puzzle(7), generate_puzzle(8))
    for index, counts in enumerate(((1, None), (11, None), (None, 0),
                                     (None, 5), (2, 2), (3, 4),
                                     (True, 1), (4.0, 1))):
        try:
            generate_puzzle(4000 + index, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
