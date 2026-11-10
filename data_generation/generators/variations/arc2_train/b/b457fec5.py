"""Generate diagonal chains of overlapping, cyclically colored squares."""

import hashlib
import numpy as np


_SOURCE_INPUTS = {
    ((17, 18), "9eb83ad41360d3d310fe991dafc808b51ced9fc1a6f60e91c5b5f80d28081262"),
    ((23, 19), "debcf6c9e5588f97ec304b58eb2183a4ebe8e9c8aece274219358ee0a"),
    ((14, 18), "e358053a3a53c61546c596ff25a938aab1afcddcc207a78c436c8319da94f5dc"),
    ((27, 23), "1ef878dbcf9127fc544a0e7d920603d7dfb64502f64c2e79668d221b83699a19"),
}
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 2), (8, 3), (9, 4), (10, 1))


def _counts(seed, rng, num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[int(seed) % len(_AUTO_COUNTS)]
    if num_train is None:
        return int(rng.integers(int(num_test) + 1, 11)), int(num_test)
    if num_test is None:
        return int(num_train), int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_roles(rng, base):
    if int(rng.integers(0, 2)):
        return base
    roles = rng.choice(10, size=2, replace=False)
    return int(roles[0]), int(roles[1])


def _palette(rng, size, background, temporary):
    available = [c for c in range(10) if c not in (background, temporary)]
    return tuple(int(c) for c in rng.choice(available, size=size, replace=False))


def _length(rng, side, key_size, category=None, cap=13):
    if category is None:
        category = int(rng.integers(0, 4))
    if category == 0:  # shorter than the square side, often without wrap
        low, high = 2, min(cap, side - 1)
    elif category == 1:  # no palette wrap
        low, high = 2, min(cap, key_size)
    elif category == 2:  # one or two traversals of the palette
        low, high = key_size + 1, min(cap, 2 * key_size)
    else:  # several traversals where the canvas permits them
        low, high = 2 * key_size + 1, min(cap, max(9, 3 * key_size + 2))
    if low > high:
        low, high = 2, min(cap, max(side, key_size))
    return int(rng.integers(low, high + 1))


def _specifications(rng, regime, index):
    if regime == "source_train" and index == 0:
        count = 1
    elif regime == "source_train" and index == 1:
        count = 2
    else:
        choices = (1, 2, 2, 3) if regime == "source_train" else (1, 2, 3, 3, 4)
        count = int(choices[int(rng.integers(len(choices)))])
    key_size = int(rng.choice((3, 4))) if regime == "source_train" and index < 2 else int(rng.choice((2, 3, 3, 4, 4, 5)))
    base_side = int(rng.choice((3, 4, 5))) if regime == "source_train" else int(rng.choice((3, 4, 5, 5, 6)))
    mixed_sides = count > 1 and regime != "source_train" or count > 1 and index > 1
    mixed_sides = mixed_sides and int(rng.integers(0, 4)) == 0
    directions = [1 if int(rng.integers(0, 2)) else -1 for _ in range(count)]
    if regime == "source_train" and index == 1:
        directions[1] = -directions[0]
    specs = []
    for object_index in range(count):
        side = int(rng.choice((3, 4, 5, 6))) if mixed_sides else base_side
        cap = 9 if count >= 3 else 13
        if regime == "source_train" and index == 0:
            length = _length(rng, side, key_size, 3 if int(rng.integers(0, 2)) else 2, cap)
        elif regime == "source_train" and index == 1:
            length = _length(rng, side, key_size, 0 if object_index == 0 else 2, cap)
            # Either object can appear first spatially. Make both offsets
            # from a hypothetical continuous palette cycle observable.
            if length % key_size == 0:
                length -= 1
        else:
            length = _length(rng, side, key_size, cap=cap)
        specs.append((side, length, directions[object_index]))
    return key_size, specs


def _boxes_separate(top, left, span, other):
    row, column, other_span = other
    return (top + span < row or row + other_span < top
            or left + span < column or column + other_span < left)


def _place(rng, specs, key_size):
    spans = [side + length - 1 for side, length, _ in specs]
    largest = max(spans)
    count = len(spans)
    minimum = max(10, largest + 2, 17 if count >= 3 else 0)
    if minimum > 30:
        return None
    for _ in range(70):
        height = int(rng.integers(minimum, 31))
        width = int(rng.integers(minimum, 31))
        order = sorted(range(count), key=lambda i: spans[i], reverse=True)
        boxes = [None] * count
        occupied = []
        for i in order:
            span = spans[i]
            positions = [(r, c) for r in range(height - span + 1)
                         for c in range(width - span + 1)]
            rng.shuffle(positions)
            chosen = next(((r, c) for r, c in positions
                           if all(_boxes_separate(r, c, span, box)
                                  for box in occupied)), None)
            if chosen is None:
                break
            box = (chosen[0], chosen[1], span)
            boxes[i] = box
            occupied.append(box)
        if any(box is None for box in boxes):
            continue
        key_positions = []
        for row in range(1, height - 1):
            for column in range(1, width - key_size):
                if all(row < top - 1 or row > top + span
                       or column + key_size - 1 < left - 1
                       or column > left + span
                       for top, left, span in occupied):
                    key_positions.append((row, column))
        if not key_positions:
            continue
        if int(rng.integers(0, 3)) == 0:
            same_rows = [point for point in key_positions
                         if any(point[0] == box[0] for box in occupied)]
            if same_rows:
                key_positions = same_rows
        elif int(rng.integers(0, 4)) == 0:
            near_top = [point for point in key_positions if point[0] <= 3]
            if near_top:
                key_positions = near_top
        key_row, key_column = key_positions[int(rng.integers(len(key_positions)))]
        minimum_row = min([key_row - 1] + [box[0] for box in occupied])
        minimum_column = min([key_column - 1] + [box[1] for box in occupied])
        maximum_row = max([key_row + 1] + [box[0] + box[2] - 1 for box in occupied])
        maximum_column = max([key_column + key_size] + [box[1] + box[2] - 1 for box in occupied])
        crop_top = max(0, minimum_row - int(rng.integers(0, 4)))
        crop_left = max(0, minimum_column - int(rng.integers(0, 4)))
        crop_bottom = min(height, maximum_row + 1 + int(rng.integers(0, 4)))
        crop_right = min(width, maximum_column + 1 + int(rng.integers(0, 4)))
        shifted = [(r - crop_top, c - crop_left) for r, c, _ in boxes]
        return (crop_bottom - crop_top, crop_right - crop_left, shifted,
                key_row - crop_top, key_column - crop_left)
    return None


def _construct(background, temporary, palette, specs, placement):
    height, width, positions, key_row, key_column = placement
    source = np.full((height, width), background, dtype=np.int8)
    for offset, color in enumerate(palette):
        source[key_row, key_column + offset] = color
    result = source.copy()
    for (side, length, direction), (top, left) in zip(specs, positions):
        start = left if direction == 1 else left + length - 1
        for layer in range(length):
            row, column = top + layer, start + direction * layer
            source[row:row + side, column:column + side] = temporary
            result[row:row + side, column:column + side] = palette[layer % len(palette)]
    return source, result


def _generate_pair(rng, regime, index, base_roles):
    for _ in range(80):
        key_size, specs = _specifications(rng, regime, index)
        placement = _place(rng, specs, key_size)
        if placement is None:
            continue
        background, temporary = _sample_roles(rng, base_roles)
        palette = _palette(rng, key_size, background, temporary)
        source, result = _construct(background, temporary, palette, specs, placement)
        try:
            _analyze_pair(source, result)
        except AssertionError:
            continue
        return source, result
    raise RuntimeError("could not place an unambiguous pair")


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(c) for c in grid.flat)


def _components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    found = []
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or visited[row, column]:
                continue
            component = np.zeros(mask.shape, dtype=bool)
            stack = [(row, column)]
            visited[row, column] = True
            while stack:
                r, c = stack.pop()
                component[r, c] = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and mask[nr, nc] and not visited[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            found.append(component)
    return found


def _infer_chain(component):
    rows, columns = np.where(component)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(columns.min()), int(columns.max())
    height, width = bottom - top + 1, right - left + 1
    candidates = []
    for side in range(2, min(height, width) + 1):
        length = height - side + 1
        if length < 2 or width != side + length - 1:
            continue
        for direction in (-1, 1):
            start = left if direction == 1 else left + length - 1
            expected = np.zeros_like(component)
            for layer in range(length):
                row, column = top + layer, start + direction * layer
                expected[row:row + side, column:column + side] = True
            if np.array_equal(expected, component):
                candidates.append((side, length, direction, top, start))
    assert len(candidates) == 1
    return candidates[0]


def _analyze_pair(source, result):
    assert source.shape == result.shape
    height, width = source.shape
    counts = np.bincount(source.ravel().astype(np.int64), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(np.argmax(counts))
    changed = source != result
    assert np.any(changed)
    values = np.unique(source[changed])
    assert len(values) == 1
    temporary = int(values[0])
    assert temporary != background
    object_mask = source == temporary
    assert np.array_equal(changed, object_mask)
    assert not np.any(result == temporary)
    key_mask = (source != background) & ~object_mask
    rows, columns = np.where(key_mask)
    assert 2 <= len(rows) <= 5 and len(set(rows)) == 1
    key_row = int(rows[0])
    first, last = int(columns.min()), int(columns.max())
    assert 0 < key_row < height - 1 and 0 < first and last < width - 1
    assert np.array_equal(np.sort(columns), np.arange(first, last + 1))
    palette = tuple(int(source[key_row, column]) for column in range(first, last + 1))
    assert len(set(palette)) == len(palette)
    assert background not in palette and temporary not in palette
    assert np.all(source[key_row - 1, first - 1:last + 2] == background)
    assert np.all(source[key_row + 1, first - 1:last + 2] == background)
    assert source[key_row, first - 1] == background
    assert source[key_row, last + 1] == background
    assert set(np.unique(source)) == {background, temporary, *palette}
    assert set(np.unique(result)) == {background, *palette}
    components = _components(object_mask)
    assert 1 <= len(components) <= 4
    objects = [_infer_chain(component) for component in components]
    expected = source.copy()
    for side, length, direction, top, start in objects:
        for layer in range(length):
            row, column = top + layer, start + direction * layer
            expected[row:row + side, column:column + side] = palette[layer % len(palette)]
    assert np.array_equal(expected, result)
    return {"shape": (height, width), "background": background,
            "temporary": temporary, "palette": palette,
            "key_row": key_row, "key_column": first, "objects": objects}


def _predict_from_input(source):
    """Rebuild a pair without inspecting its output or construction choices."""
    colors, counts = np.unique(source, return_counts=True)
    background = int(colors[np.argmax(counts)])
    singletons = {int(color) for color, count in zip(colors, counts)
                  if count == 1}
    temporary_colors = [int(color) for color, count in zip(colors, counts)
                        if color != background and count > 1]
    assert len(temporary_colors) == 1
    temporary = temporary_colors[0]
    key_rows, key_columns = np.where(np.isin(source, list(singletons)))
    assert len(set(key_rows)) == 1
    palette = [int(source[int(key_rows[0]), column])
               for column in sorted(int(c) for c in key_columns)]
    answer = source.copy()
    for component in _components(source == temporary):
        rows, columns = np.where(component)
        top, bottom = int(rows.min()), int(rows.max())
        top_columns = columns[rows == top]
        bottom_columns = columns[rows == bottom]
        side = len(top_columns)
        length = bottom - top - side + 2
        direction = 1 if int(bottom_columns.min()) > int(top_columns.min()) else -1
        start = int(top_columns.min())
        for layer in range(length):
            row, column = top + layer, start + direction * layer
            answer[row:row + side, column:column + side] = palette[layer % len(palette)]
    return answer


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, result = pair["input"], pair["output"]
            for grid in (source, result):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert source.shape == result.shape
            signature = _grid_key(source)
            pair_signature = signature, _grid_key(result)
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            source_digest = hashlib.sha256(source.astype(np.uint8).tobytes()).hexdigest()
            assert (tuple(source.shape), source_digest) not in _SOURCE_INPUTS
            analyses[section].append(_analyze_pair(source, result))
    training = analyses["train"]
    assert any(any(obj[1] > len(item["palette"]) for obj in item["objects"])
               for item in training)
    assert any(len(item["objects"]) >= 2 for item in training)
    assert any(len(item["objects"]) >= 2
               and all(obj[1] % len(item["palette"]) != 0
                       for obj in item["objects"])
               for item in training)
    assert {obj[2] for item in training for obj in item["objects"]} == {-1, 1}
    assert all(3 <= obj[0] <= 6 for section in analyses.values()
               for item in section for obj in item["objects"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    base_colors = rng.choice(10, size=2, replace=False)
    base_roles = int(base_colors[0]), int(base_colors[1])
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, count, section in (("source_train", train_count, "train"),
                                   ("target_test", test_count, "test")):
        for index in range(count):
            for _ in range(64):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                source, result = _generate_pair(child_rng, regime, index, base_roles)
                signature = _grid_key(source)
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[section].append({"input": source, "output": result})
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(np.array_equal(a[name], b[name])
               for section in ("train", "test")
               for a, b in zip(first[section], second[section])
               for name in ("input", "output"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(314159, 6, 3),
                          generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(generate_puzzle(1001, 5, 2),
                              generate_puzzle(1002, 5, 2))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4),
                                            (2, None), (10, None))):
        puzzle = generate_puzzle(7000 + index, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (2, 4), (4, 4)):
        try:
            generate_puzzle(99173, train, test)
        except ValueError:
            continue
        raise AssertionError("invalid counts accepted")
    train_counts, test_counts = set(), set()
    sides, keys, lengths, object_counts, shapes = set(), set(), set(), set(), set()
    combos, layouts, roles = set(), set(), set()
    short, no_wrap, one_wrap, several_wrap = 0, 0, 0, 0
    mixed_sides, same_slope, mixed_slope, edge, same_key_row = 0, 0, 0, 0, 0
    role_changing_puzzles = 0
    train_side_variation = train_key_variation = 0
    reused_test_side = novel_test_side = 0
    late_key = source_shape_reuse = repeated_silhouette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_roles = set()
        puzzle_train_sides = set()
        puzzle_train_keys = set()
        puzzle_test_sides = set()
        fingerprint = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                item = _analyze_pair(pair["input"], pair["output"])
                assert np.array_equal(_predict_from_input(pair["input"]),
                                      pair["output"])
                shape, palette, objects = item["shape"], item["palette"], item["objects"]
                shapes.add(shape)
                keys.add(len(palette))
                source_shape_reuse += shape in {(17, 18), (23, 19), (14, 18), (27, 23)}
                late_key += item["key_row"] > 3
                object_counts.add(len(objects))
                puzzle_roles.add((item["background"], item["temporary"]))
                roles.add((item["background"], item["temporary"]))
                signature = tuple(sorted((s, length, direction) for s, length, direction, _, _ in objects))
                fingerprint.append((section, shape, len(palette), signature))
                if len({obj[0] for obj in objects}) > 1:
                    mixed_sides += 1
                if len(objects) > 1:
                    object_signatures = [(obj[0], obj[1], obj[2]) for obj in objects]
                    repeated_silhouette += len(set(object_signatures)) < len(object_signatures)
                    if len({obj[2] for obj in objects}) == 1:
                        same_slope += 1
                    else:
                        mixed_slope += 1
                if any(top == 0 or top + side + length - 1 == shape[0]
                       or min(start, start + direction * (length - 1)) == 0
                       or max(start, start + direction * (length - 1)) + side == shape[1]
                       for side, length, direction, top, start in objects):
                    edge += 1
                if any(top == item["key_row"] for _, _, _, top, _ in objects):
                    same_key_row += 1
                for side, length, direction, _, _ in objects:
                    sides.add(side)
                    if section == "train":
                        puzzle_train_sides.add(side)
                        puzzle_train_keys.add(len(palette))
                    else:
                        puzzle_test_sides.add(side)
                    lengths.add(length)
                    combos.add((side, len(palette)))
                    short += length < side
                    no_wrap += length <= len(palette)
                    one_wrap += len(palette) < length <= 2 * len(palette)
                    several_wrap += length > 2 * len(palette)
        layouts.add(tuple(fingerprint))
        role_changing_puzzles += len(puzzle_roles) > 1
        train_side_variation += len(puzzle_train_sides) > 1
        train_key_variation += len(puzzle_train_keys) > 1
        reused_test_side += bool(puzzle_test_sides & puzzle_train_sides)
        novel_test_side += bool(puzzle_test_sides - puzzle_train_sides)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sides.issuperset({3, 4, 5}) and keys.issuperset({2, 3, 4, 5})
    assert max(lengths) >= 10 and min(lengths) == 2
    assert object_counts.issuperset({1, 2, 3})
    assert len(shapes) >= 30 and len(combos) >= 10 and len(layouts) >= 35
    assert len(roles) >= 15 and role_changing_puzzles >= 15
    assert train_side_variation >= 20 and train_key_variation >= 20
    assert reused_test_side >= 20 and novel_test_side >= 5
    assert min(short, no_wrap, one_wrap, several_wrap) >= 15
    assert min(mixed_sides, same_slope, mixed_slope, edge, same_key_row) >= 5
    assert late_key >= 20 and source_shape_reuse >= 1
    assert repeated_silhouette >= 5
