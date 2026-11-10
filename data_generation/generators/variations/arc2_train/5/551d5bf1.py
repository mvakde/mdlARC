"""Generate scenes of filled rectangular frames with directional outlets."""

import hashlib

import numpy as np


_DIRECTIONS = ("U", "D", "L", "R")
_SOURCE_PAIR_HASHES = {
    "c53a4ef7b05a5ff2c8ac298bf5cd4bab527c0c2bda4dc2bd44f8038ac6694bc3",
    "c796e90b766a478f6bf78553d1dcf0f39720a3f93ff54725b8103efe90da39aa",
    "ae4f2509ad4da1f90a9d6b721d29731432644bfaaddb7f522bad3a91c26ae9d7",
}


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _palette_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    paint, base_bg, base_outline = colors[:3]
    mode = int(rng.integers(4))
    alt_bg = base_bg if mode in (0, 2) else colors[3]
    if mode in (0, 1):
        alt_outline = base_outline
    else:
        alt_outline = next(color for color in colors[3:] if color != alt_bg)
    base = (base_bg, base_outline)
    alternate = (alt_bg, alt_outline)
    choices = [base]
    if alternate != base:
        choices.append(alternate)
    if mode == 3:
        choices += [
            pair for pair in ((base_bg, alt_outline), (alt_bg, base_outline))
            if pair[0] != pair[1]
        ]
    return paint, choices, mode


def _separated(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _frame_size(rng, height, width, count):
    cap_h, cap_w = height - 2, width - 2
    mode = int(rng.integers(5))
    dense = count >= 6
    if mode == 0:  # shallow
        return 3, int(rng.integers(3, min(10 if not dense else 7, cap_w) + 1))
    if mode == 1:  # narrow and tall
        return int(rng.integers(6, min(16 if not dense else 10, cap_h) + 1)), int(rng.integers(3, min(5, cap_w) + 1))
    if mode == 2:  # long and wide
        return int(rng.integers(3, min(5, cap_h) + 1)), int(rng.integers(6, min(16 if not dense else 10, cap_w) + 1))
    if mode == 3:  # medium
        return int(rng.integers(4, min(9 if not dense else 7, cap_h) + 1)), int(rng.integers(4, min(10 if not dense else 7, cap_w) + 1))
    return int(rng.integers(3, min(7, cap_h) + 1)), int(rng.integers(3, min(7, cap_w) + 1))


def _place_boxes(rng, height, width, count):
    boxes = []
    for _ in range(count):
        candidates = []
        for _ in range(35):
            box_h, box_w = _frame_size(rng, height, width, count)
            positions = []
            for r0 in range(1, height - box_h):
                for c0 in range(1, width - box_w):
                    box = (r0, r0 + box_h - 1, c0, c0 + box_w - 1)
                    if all(_separated(box, other) for other in boxes):
                        positions.append(box)
            if positions:
                candidates = positions
                break
        if not candidates:
            return None
        boxes.append(candidates[int(rng.integers(len(candidates)))])
    return boxes


def _gap_options(box, boxes, height, width):
    r0, r1, c0, c1 = box
    result = {direction: [] for direction in _DIRECTIONS}
    for col in range(c0 + 1, c1):
        if all(not (b < r0 and x <= col <= y)
               for a, b, x, y in boxes if (a, b, x, y) != box):
            result["U"].append((r0, col))
        if all(not (a > r1 and x <= col <= y)
               for a, b, x, y in boxes if (a, b, x, y) != box):
            result["D"].append((r1, col))
    for row in range(r0 + 1, r1):
        if all(not (y < c0 and a <= row <= b)
               for a, b, x, y in boxes if (a, b, x, y) != box):
            result["L"].append((row, c0))
        if all(not (x > c1 and a <= row <= b)
               for a, b, x, y in boxes if (a, b, x, y) != box):
            result["R"].append((row, c1))
    return result


def _assign_openings(rng, boxes, height, width, opened_count, required):
    options = [_gap_options(box, boxes, height, width) for box in boxes]
    assignments = {}
    used = set()
    for direction in rng.permutation(required):
        candidates = [i for i, item in enumerate(options)
                      if i not in used and item[str(direction)]]
        if not candidates:
            return None
        index = int(rng.choice(candidates))
        used.add(index)
        assignments[index] = str(direction)
    available = [i for i, item in enumerate(options)
                 if i not in used and any(item[d] for d in _DIRECTIONS)]
    extra = opened_count - len(required)
    if extra > len(available):
        return None
    if extra:
        for index in rng.choice(available, size=extra, replace=False):
            index = int(index)
            choices = [d for d in _DIRECTIONS if options[index][d]]
            assignments[index] = str(rng.choice(choices))
    frames = []
    for index, box in enumerate(boxes):
        direction = assignments.get(index)
        gap = None
        if direction is not None:
            gap = options[index][direction][int(rng.integers(len(options[index][direction])))]
        frames.append((box, direction, gap))
    return frames


def _draw_pair(height, width, frames, background, outline, paint):
    inp = np.full((height, width), background, dtype=np.int8)
    for (r0, r1, c0, c1), _, gap in frames:
        inp[r0, c0:c1 + 1] = outline
        inp[r1, c0:c1 + 1] = outline
        inp[r0:r1 + 1, c0] = outline
        inp[r0:r1 + 1, c1] = outline
        if gap is not None:
            inp[gap] = background
    out = inp.copy()
    for (r0, r1, c0, c1), _, _ in frames:
        out[r0 + 1:r1, c0 + 1:c1] = paint
    for (r0, r1, c0, c1), direction, gap in frames:
        if direction is None:
            continue
        row, col = gap
        if direction == "U":
            out[:r0 + 1, col] = paint
        elif direction == "D":
            out[r1:, col] = paint
        elif direction == "L":
            out[row, :c0 + 1] = paint
        else:
            out[row, c1:] = paint
    return inp, out


def _make_pair(rng, count, opened_count, required, palette, paint, anchor=None):
    minimum = 17 if count <= 3 else 20 if count <= 5 else 25
    for attempt in range(120):
        if anchor is None:
            height = int(rng.integers(minimum, 31))
            width = int(rng.integers(minimum, 31))
        else:
            height = int(rng.integers(max(minimum, anchor[0] - 2),
                                      min(30, anchor[0] + 2) + 1))
            width = int(rng.integers(max(minimum, anchor[1] - 2),
                                     min(30, anchor[1] + 2) + 1))
        boxes = _place_boxes(rng, height, width, count)
        if boxes is None:
            continue
        frames = _assign_openings(rng, boxes, height, width, opened_count, required)
        if frames is None:
            continue
        return _draw_pair(height, width, frames, palette[0], palette[1], paint)
    raise RuntimeError("could not place a valid scene")


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_hash(inp, out):
    return hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes()
                          + out.astype(np.int8).tobytes()).hexdigest()


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            cells, stack = [], [(r, c)]
            seen[r, c] = True
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for rr, cc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)):
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            found.append(cells)
    return found


def _perimeter(box):
    r0, r1, c0, c1 = box
    return ({(r, c0) for r in range(r0, r1 + 1)}
            | {(r, c1) for r in range(r0, r1 + 1)}
            | {(r0, c) for c in range(c0, c1 + 1)}
            | {(r1, c) for c in range(c0, c1 + 1)})


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    height, width = inp.shape
    background = int(inp[0, 0])
    assert np.all(inp[0, :] == background) and np.all(inp[-1, :] == background)
    assert np.all(inp[:, 0] == background) and np.all(inp[:, -1] == background)
    input_colors = set(map(int, np.unique(inp)))
    assert len(input_colors) == 2 and background in input_colors
    outline = next(iter(input_colors - {background}))
    output_colors = set(map(int, np.unique(out)))
    assert len(output_colors) == 3 and input_colors.issubset(output_colors)
    paint = next(iter(output_colors - input_colors))
    frames = []
    for cells in _components(inp == outline):
        rows, cols = zip(*cells)
        box = min(rows), max(rows), min(cols), max(cols)
        r0, r1, c0, c1 = box
        assert r1 - r0 >= 2 and c1 - c0 >= 2
        nominal = _perimeter(box)
        assert set(cells).issubset(nominal)
        missing = nominal - set(cells)
        assert len(missing) <= 1
        assert np.all(inp[r0 + 1:r1, c0 + 1:c1] == background)
        direction, gap = None, None
        if missing:
            gap = next(iter(missing))
            row, col = gap
            assert (row in (r0, r1)) != (col in (c0, c1))
            if row == r0:
                direction = "U"
                assert np.all(inp[:r0 + 1, col] == background)
            elif row == r1:
                direction = "D"
                assert np.all(inp[r1:, col] == background)
            elif col == c0:
                direction = "L"
                assert np.all(inp[row, :c0 + 1] == background)
            else:
                direction = "R"
                assert np.all(inp[row, c1:] == background)
        frames.append((box, direction, gap))
    assert 2 <= len(frames) <= 7
    for i, first in enumerate(frames):
        for second in frames[i + 1:]:
            assert _separated(first[0], second[0])
    expected = inp.copy()
    for (r0, r1, c0, c1), _, _ in frames:
        expected[r0 + 1:r1, c0 + 1:c1] = paint
    for (r0, r1, c0, c1), direction, gap in frames:
        if direction is None:
            continue
        row, col = gap
        if direction == "U":
            expected[:r0 + 1, col] = paint
        elif direction == "D":
            expected[r1:, col] = paint
        elif direction == "L":
            expected[row, :c0 + 1] = paint
        else:
            expected[row, c1:] = paint
    assert np.array_equal(expected, out)
    return background, outline, paint, frames


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    paints, training_directions, test_directions = set(), set(), set()
    training_closed = False
    all_counts = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            inp_key, pair_key = _grid_key(inp), (_grid_key(inp), _grid_key(out))
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            assert _pair_hash(inp, out) not in _SOURCE_PAIR_HASHES
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            background, outline, paint, frames = _analyze_pair(inp, out)
            assert len({background, outline, paint}) == 3
            paints.add(paint)
            all_counts.add(len(frames))
            directions = {direction for _, direction, _ in frames if direction is not None}
            if split == "train":
                training_directions.update(directions)
                training_closed |= any(direction is None for _, direction, _ in frames)
            else:
                test_directions.update(directions)
    assert len(paints) == 1
    assert training_directions == set(_DIRECTIONS) and training_closed
    assert test_directions.issubset(training_directions)
    assert len(all_counts) >= 2


def _puzzles_equal(a, b):
    if set(a) != set(b):
        return False
    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False
        for x, y in zip(a[split], b[split]):
            if not np.array_equal(x["input"], y["input"]) or not np.array_equal(x["output"], y["output"]):
                return False
    return True


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed % (1 << 64))
    train_count, test_count = _pair_counts(seed, rng, num_train, num_test)
    paint, palettes, palette_mode = _palette_parameters(rng)
    low_variation = bool(rng.random() < 0.14)
    if low_variation:
        palettes, palette_mode = palettes[:1], 0
        anchor = (int(rng.integers(23, 28)), int(rng.integers(23, 28)))
    else:
        anchor = None
    direction_order = [str(x) for x in rng.permutation(_DIRECTIONS)]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    frame_counts = []
    for split, pair_count in (("train", train_count), ("test", test_count)):
        for index in range(pair_count):
            required = direction_order[2 * index:2 * index + 2] if split == "train" and index < 2 else []
            for retry in range(60):
                if required:
                    count = int(rng.integers(3, 6 if low_variation else 7))
                    opened = int(rng.integers(2, count))
                else:
                    count = int(rng.integers(3, 6)) if low_variation else int(rng.integers(2, 8))
                    opened = int(rng.integers(count + 1))
                if split == "test" and index == pair_count - 1 and len(set(frame_counts)) == 1 and count == frame_counts[0]:
                    count = 2 if count != 2 else 3
                    opened = min(opened, count)
                if palette_mode == 0 or index == 0:
                    palette = palettes[0]
                elif split == "train" and index == 1:
                    palette = palettes[1]
                else:
                    palette = palettes[int(rng.integers(len(palettes)))]
                try:
                    inp, out = _make_pair(rng, count, opened, required, palette, paint, anchor)
                except RuntimeError:
                    continue
                key = _grid_key(inp)
                if key in seen_inputs or _pair_hash(inp, out) in _SOURCE_PAIR_HASHES:
                    continue
                seen_inputs.add(key)
                frame_counts.append(count)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("could not generate a distinct valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    train_counts, test_counts, frame_counts, opened_counts = set(), set(), set(), set()
    shapes, sizes, palettes, layouts = set(), set(), set(), set()
    coverage = {
        "palette_changes": 0, "fewer": 0, "same": 0, "more": 0,
        "test_shorter": 0, "test_taller": 0, "test_same_height": 0,
        "test_right": 0, "test_repeated": 0, "train_shallow": 0,
        "shallow_closed": 0, "shallow_other_direction": 0,
        "long_ray": 0, "short_ray": 0, "inner_outlet": 0,
        "crossing": 0, "staggered": 0, "column_overlap": 0,
        "offset_gap": 0, "near_end_gap": 0, "low_variation": 0,
        "train_simple": 0, "train_dense": 0,
        "test_simple": 0, "test_dense": 0,
    }

    def exterior(frame, shape):
        (r0, r1, c0, c1), direction, gap = frame
        if direction == "U":
            return {(r, gap[1]) for r in range(r0)}
        if direction == "D":
            return {(r, gap[1]) for r in range(r1 + 1, shape[0])}
        if direction == "L":
            return {(gap[0], c) for c in range(c0)}
        if direction == "R":
            return {(gap[0], c) for c in range(c1 + 1, shape[1])}
        return set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        scene_counts = {"train": [], "test": []}
        scene_heights = {"train": [], "test": []}
        scene_widths = {"train": [], "test": []}
        scene_palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, outline, paint, frames = _analyze_pair(inp, out)
                scene_palettes.add((bg, outline))
                palettes.add((bg, outline, paint))
                shapes.add(inp.shape)
                frame_counts.add(len(frames))
                opened_counts.add(sum(direction is not None for _, direction, _ in frames))
                layouts.add((inp.shape, tuple(sorted(frames))))
                scene_counts[split].append(len(frames))
                scene_heights[split].append(inp.shape[0])
                scene_widths[split].append(inp.shape[1])
                coverage[f"{split}_simple"] += len(frames) <= 3
                coverage[f"{split}_dense"] += len(frames) >= 6
                coverage["staggered"] += len({f[0][0] for f in frames}) >= 3
                directions = [direction for _, direction, _ in frames if direction is not None]
                if split == "test":
                    coverage["test_right"] += "R" in directions
                    coverage["test_repeated"] += len(directions) > len(set(directions))
                for index, ((r0, r1, c0, c1), direction, gap) in enumerate(frames):
                    sizes.add((r1 - r0 + 1, c1 - c0 + 1))
                    if split == "train" and r1 - r0 + 1 == 3:
                        coverage["train_shallow"] += 1
                    if r1 - r0 + 1 == 3:
                        coverage["shallow_closed"] += direction is None
                        coverage["shallow_other_direction"] += direction not in (None, "D")
                    if direction:
                        position = gap[1] - c0 if direction in "UD" else gap[0] - r0
                        side = c1 - c0 + 1 if direction in "UD" else r1 - r0 + 1
                        coverage["near_end_gap"] += position in (1, side - 2)
                        coverage["offset_gap"] += position not in (1, side - 2)
                        length = len(exterior(frames[index], inp.shape))
                        coverage["short_ray"] += length <= 2
                        coverage["long_ray"] += length >= 12
                        for other_index, other in enumerate(frames):
                            if index == other_index:
                                continue
                            a, b, x, y = other[0]
                            if ((direction == "U" and a < r0)
                                or (direction == "D" and b > r1)
                                or (direction == "L" and x < c0)
                                or (direction == "R" and y > c1)):
                                coverage["inner_outlet"] += 1
                                break
                for index, first in enumerate(frames):
                    for second in frames[index + 1:]:
                        ar0, ar1, ac0, ac1 = first[0]
                        br0, br1, bc0, bc1 = second[0]
                        coverage["column_overlap"] += max(ac0, bc0) <= min(ac1, bc1) and (ar1 < br0 or br1 < ar0)
                        if first[1] and second[1] and (first[1] in "UD") != (second[1] in "UD"):
                            coverage["crossing"] += bool(exterior(first, inp.shape) & exterior(second, inp.shape))
        coverage["palette_changes"] += len(scene_palettes) > 1
        all_counts = scene_counts["train"] + scene_counts["test"]
        all_heights = scene_heights["train"] + scene_heights["test"]
        all_widths = scene_widths["train"] + scene_widths["test"]
        coverage["low_variation"] += (
            len(scene_palettes) == 1 and max(all_counts) - min(all_counts) <= 2
            and max(all_heights) - min(all_heights) <= 4
            and max(all_widths) - min(all_widths) <= 4
        )
        coverage["fewer"] += any(t < n for t in scene_counts["test"] for n in scene_counts["train"])
        coverage["same"] += any(t == n for t in scene_counts["test"] for n in scene_counts["train"])
        coverage["more"] += any(t > n for t in scene_counts["test"] for n in scene_counts["train"])
        coverage["test_shorter"] += any(t < n for t in scene_heights["test"] for n in scene_heights["train"])
        coverage["test_same_height"] += any(t == n for t in scene_heights["test"] for n in scene_heights["train"])
        coverage["test_taller"] += any(t > n for t in scene_heights["test"] for n in scene_heights["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(frame_counts) >= 5 and len(opened_counts) >= 5
    assert len(shapes) >= 30 and len(sizes) >= 25
    assert len(palettes) >= 15 and len(layouts) >= 100
    for property_name, minimum in {
        "palette_changes": 10, "fewer": 10, "same": 10, "more": 10,
        "test_shorter": 10, "test_taller": 10, "test_same_height": 5,
        "test_right": 5, "test_repeated": 5, "train_shallow": 20,
        "shallow_closed": 10, "shallow_other_direction": 10,
        "long_ray": 20, "short_ray": 20, "inner_outlet": 20,
        "crossing": 1, "staggered": 30, "column_overlap": 30,
        "offset_gap": 20, "near_end_gap": 20, "low_variation": 1,
        "train_simple": 10, "train_dense": 10,
        "test_simple": 5, "test_dense": 5,
    }.items():
        assert coverage[property_name] >= minimum, property_name
    a = generate_puzzle(314159, 6, 3)
    b = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(a, b) and not _puzzles_equal(a, generate_puzzle(314160, 6, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(5000 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for index, (train, test) in enumerate(((1, None), (11, None),
                                           (None, 0), (None, 5),
                                           (2, 2), (3, 4), (4, 4))):
        try:
            generate_puzzle(7000 + index, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count was accepted")
