"""Two-color silhouettes: rectangular components select the fill color."""

import numpy as np


_SOURCE_INPUTS = {
    "0000000000/0333333330/1113333111/1113333111/1113333111/"
    "0333333330/0333333330/0333333330/0333333330/0000000000",
    "00000000000000/00444444444400/00444444444400/00444444444400/"
    "22444444444422/22220000002222/22220000002222/22220000002222/"
    "00000000000000/00000000000000/00000000000000",
    "00000000000/00066666660/00066666660/00066666660/00066666660/"
    "00066666660/00066668888/00066668888/00066668888/00000008888/"
    "00000000000/00000000000/00000000000/00000000000",
    "00000000000000/00000000000000/01111111111110/01111111111110/"
    "01111111111110/01111155511110/01111155511110/00000055500000/"
    "00000000000000/00000000000000",
    "0000000/1111000/1111333/1111333/1111333/0033333/0033333/"
    "0000000/0000000",
    "000000000000/000000000000/555550000000/555550000000/"
    "555550000000/555550000000/555550000000/555550000000/"
    "555333000000/555333000000/000000000000/000000000000",
}


def _grid_key(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _paint(mask, row, col, height, width):
    mask[row:row + height, col:col + width] = True


def _components(mask):
    remaining = mask.copy()
    result = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if not remaining[row, col]:
            continue
        stack = [(row, col)]
        remaining[row, col] = False
        points = []
        while stack:
            rr, cc = stack.pop()
            points.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc),
                           (rr, cc - 1), (rr, cc + 1)):
                if (0 <= nr < height and 0 <= nc < width
                        and remaining[nr, nc]):
                    remaining[nr, nc] = False
                    stack.append((nr, nc))
        component = np.zeros_like(mask)
        for rr, cc in points:
            component[rr, cc] = True
        result.append(component)
    return result


def _bbox(mask):
    points = np.argwhere(mask)
    assert len(points)
    low = points.min(axis=0)
    high = points.max(axis=0) + 1
    return tuple(int(v) for v in (*low, *high))


def _box_area(mask):
    top, left, bottom, right = _bbox(mask)
    return (bottom - top) * (right - left)


def _is_rectangle(mask):
    return bool(np.any(mask)) and int(mask.sum()) == _box_area(mask)


def _all_rectangles(mask):
    components = _components(mask)
    return bool(components) and all(_is_rectangle(c) for c in components)


def _new_masks(rng, mode):
    rear = np.zeros((40, 40), dtype=bool)
    front = np.zeros_like(rear)
    row, col = 12, 12
    if mode == "bridge":
        cap_width = int(rng.integers(6, 13))
        cap_height = int(rng.integers(3, 6))
        left, right = (int(v) for v in rng.integers(2, 5, size=2))
        leg_height = int(rng.integers(3, 8))
        span = left + cap_width + right
        _paint(front, row, col + left, cap_height, cap_width)
        shoulder = row + cap_height - 1
        _paint(rear, shoulder, col, 1, span)
        _paint(rear, shoulder + 1, col,
               leg_height, left + int(rng.integers(1, 3)))
        right_extension = int(rng.integers(1, 3))
        _paint(rear, shoulder + 1, col + span - right - right_extension,
               leg_height - int(rng.integers(0, 2)),
               right + right_extension)
    else:
        rear_height = int(rng.integers(7, 15))
        rear_width = int(rng.integers(7, 15))
        _paint(rear, row, col, rear_height, rear_width)
        if mode == "multi":
            for side in ("left", "right"):
                patch_height = int(rng.integers(2, 5))
                patch_width = int(rng.integers(2, 5))
                overlap = int(rng.integers(1, patch_width))
                offset = int(rng.integers(-2, 3))
                patch_top = row + (rear_height - patch_height) // 2 + offset
                patch_top = max(row + 1,
                                min(patch_top, row + rear_height - patch_height - 1))
                patch_left = (col - patch_width + overlap if side == "left"
                              else col + rear_width - overlap)
                _paint(front, patch_top, patch_left, patch_height, patch_width)
        else:
            patch_height = int(rng.integers(2, min(8, rear_height - 1)))
            patch_width = int(rng.integers(2, min(8, rear_width - 1)))
            if mode == "embedded":
                patch_top = int(rng.integers(row + 1,
                                             row + rear_height - patch_height))
                patch_left = int(rng.integers(col + 1,
                                              col + rear_width - patch_width))
            elif mode == "side":
                overlap = int(rng.integers(1, patch_height))
                patch_top = row - patch_height + overlap
                patch_left = int(rng.integers(col + 1,
                                              col + rear_width - patch_width))
            elif mode == "corner":
                overlap_height = int(rng.integers(1, patch_height))
                overlap_width = int(rng.integers(1, patch_width))
                patch_top = row - patch_height + overlap_height
                patch_left = col - patch_width + overlap_width
            else:
                raise ValueError("unknown geometry")
            _paint(front, patch_top, patch_left, patch_height, patch_width)
    return rear, front


def _decorate(rng, rear, front, mode, regime):
    if mode == "bridge":
        return rear
    top, left, bottom, right = _bbox(rear)
    probabilities = ((.28, .37, .27, .08) if regime == "source_train"
                     else (.15, .30, .35, .20))
    arm_count = int(rng.choice((0, 1, 2, 3), p=probabilities))
    for _ in range(arm_count):
        side = int(rng.integers(0, 4))
        depth = int(rng.integers(1, 4))
        if side in (0, 1):
            span = int(rng.integers(2, min(6, right - left)))
            start = int(rng.integers(left + 1, right - span))
            _paint(rear, top - depth if side == 0 else bottom,
                   start, depth, span)
        else:
            span = int(rng.integers(2, min(6, bottom - top)))
            start = int(rng.integers(top + 1, bottom - span))
            _paint(rear, start, left - depth if side == 2 else right,
                   span, depth)
    if rng.random() < (.22 if regime == "source_train" else .43):
        candidates = []
        for _ in range(28):
            hole_height = int(rng.integers(1, 3))
            hole_width = int(rng.integers(1, 3))
            enclosed = bool(rng.integers(0, 2))
            if enclosed:
                hr = int(rng.integers(top + 1, bottom - hole_height))
                hc = int(rng.integers(left + 1, right - hole_width))
            else:
                edge = int(rng.integers(0, 4))
                hr = (top if edge == 0 else bottom - hole_height if edge == 1
                      else int(rng.integers(top + 1, bottom - hole_height)))
                hc = (left if edge == 2 else right - hole_width if edge == 3
                      else int(rng.integers(left + 1, right - hole_width)))
            area = rear[hr:hr + hole_height, hc:hc + hole_width]
            if np.all(area) and not np.any(front[hr:hr + hole_height,
                                               hc:hc + hole_width]):
                candidates.append((hr, hc, hole_height, hole_width))
        if candidates:
            hr, hc, hh, hw = candidates[int(rng.integers(len(candidates)))]
            rear[hr:hr + hh, hc:hc + hw] = False
    return rear


def _crop_and_turn(rng, rear, front):
    top, left, bottom, right = _bbox(rear | front)
    rear = rear[top:bottom, left:right]
    front = front[top:bottom, left:right]
    turns = int(rng.integers(0, 4))
    rear, front = np.rot90(rear, turns), np.rot90(front, turns)
    if rng.random() < .5:
        rear, front = np.fliplr(rear), np.fliplr(front)
    return rear, front


def _valid_masks(rear, front, mode):
    visible_rear = rear & ~front
    if not np.any(visible_rear) or not np.any(front):
        return False
    parts = _components(front)
    if len(parts) != (2 if mode == "multi" else 1):
        return False
    if not _all_rectangles(front) or _all_rectangles(visible_rear):
        return False
    if _box_area(front) >= _box_area(visible_rear):
        return False
    if len(_components(rear | front)) != 1:
        return False
    return True


def _make_pair(rng, mode, regime, background, front_color, rear_color):
    for _ in range(250):
        rear, front = _new_masks(rng, mode)
        rear = _decorate(rng, rear, front, mode, regime)
        rear, front = _crop_and_turn(rng, rear, front)
        if not _valid_masks(rear, front, mode):
            continue
        height, width = rear.shape
        if height > 26 or width > 26:
            continue
        limits = (30 - height, 30 - width)
        top = int(rng.integers(1, min(5, limits[0])))
        left = int(rng.integers(1, min(5, limits[1])))
        bottom = int(rng.integers(1, min(5, limits[0] - top + 1)))
        right = int(rng.integers(1, min(5, limits[1] - left + 1)))
        grid = np.full((top + height + bottom, left + width + right),
                       background, dtype=np.int8)
        region = grid[top:top + height, left:left + width]
        region[rear] = rear_color
        region[front] = front_color
        output = np.where(grid == background, background,
                          front_color).astype(np.int8)
        return grid, output
    raise RuntimeError("could not construct an unambiguous pair")


def _valid_count(value, minimum, maximum, name):
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")


def _colors(rng, background, previous=None, reverse=False):
    available = [color for color in range(10) if color != background]
    if reverse and previous is not None and previous[0] != background:
        rear_color = previous[0]
        available.remove(rear_color)
        if previous[1] in available:
            available.remove(previous[1])
        front_color = int(rng.choice(available))
    else:
        front_color, rear_color = (int(v) for v in
                                   rng.choice(available, size=2, replace=False))
        if previous is not None and {front_color, rear_color} == set(previous):
            alternatives = [c for c in available if c not in previous]
            rear_color = int(rng.choice(alternatives))
    return front_color, rear_color


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    _valid_count(num_train, 2, 10, "num_train")
    _valid_count(num_test, 1, 4, "num_test")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    # The only puzzle-wide convention is the shape rule. Colors are local.
    modes = ["multi", str(rng.choice(("embedded", "side", "corner", "bridge")))]
    modes += [str(rng.choice(("multi", "embedded", "side", "corner", "bridge"),
                             p=(.22, .16, .24, .24, .14)))
              for _ in range(num_train - 2)]
    rng.shuffle(modes)
    test_modes = [str(rng.choice(("multi", "embedded", "side", "corner", "bridge"),
                                 p=(.21, .15, .24, .24, .16)))
                  for _ in range(num_test)]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_colors = None
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        split = "train" if index < num_train else "test"
        mode = modes[index] if split == "train" else test_modes[index - num_train]
        regime = "source_train" if split == "train" else "target_test"
        background = (0 if pair_rng.random() < .75
                      else int(pair_rng.integers(1, 10)))
        reverse = index > 0 and pair_rng.random() < .38
        front_color, rear_color = _colors(pair_rng, background,
                                          previous_colors, reverse)
        previous_colors = (front_color, rear_color)
        for _ in range(20):
            input_grid, output_grid = _make_pair(
                pair_rng, mode, regime, background, front_color, rear_color)
            key = _grid_key(input_grid)
            if key not in seen and key not in _SOURCE_INPUTS:
                seen.add(key)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                break
        else:
            raise RuntimeError("could not sample a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_properties(pair):
    grid, output = pair["input"], pair["output"]
    background = int(grid[0, 0])
    assert all(int(grid[row, col]) == background for row, col in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    colors = set(int(v) for v in np.unique(grid))
    assert len(colors) == 3 and background in colors
    output_colors = set(int(v) for v in np.unique(output))
    assert len(output_colors) == 2 and background in output_colors
    front_color = next(iter(output_colors - {background}))
    rear_color = next(iter(colors - {background, front_color}))
    front = grid == front_color
    rear = grid == rear_color
    assert np.array_equal(output, np.where(grid == background,
                                            background, front_color))
    assert np.array_equal(grid == background, output == background)
    assert len(_components(front)) in (1, 2)
    assert _all_rectangles(front) and not _all_rectangles(rear)
    assert _box_area(front) < _box_area(rear)
    assert len(_components(grid != background)) == 1
    assert not np.array_equal(grid, output)
    return background, front_color, rear_color, len(_components(front))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    seen = set()
    train_component_counts = set()
    train_palettes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = _grid_key(pair["input"])
            assert key not in seen and key not in _SOURCE_INPUTS
            seen.add(key)
            background, front, rear, count = _pair_properties(pair)
            if split == "train":
                train_component_counts.add(count)
                train_palettes.add(frozenset((front, rear)))
    assert train_component_counts == {1, 2}
    assert len(train_palettes) >= 2


def _same_puzzles(first, second):
    return (all(len(first[s]) == len(second[s]) for s in ("train", "test"))
            and all(np.array_equal(a[k], b[k])
                    for split in ("train", "test")
                    for a, b in zip(first[split], second[split])
                    for k in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, silhouettes, palettes, backgrounds = set(), set(), set(), set()
    component_counts, rear_parts = set(), set()
    role_reversal_puzzles = 0
    solid_count = hole_count = one_cell_overlap = multi_hole_count = 0
    shared_orientation_puzzles = novel_orientation_puzzles = 0
    directions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = {}
        orientations = {}
        for split in ("train", "test"):
            orientations[split] = set()
            for pair in puzzle[split]:
                grid = pair["input"]
                background, front, rear, count = _pair_properties(pair)
                front_mask, rear_mask = grid == front, grid == rear
                component_counts.add(count)
                rear_parts.add(len(_components(rear_mask)))
                sizes.add(grid.shape)
                palettes.add(frozenset((front, rear)))
                backgrounds.add(background)
                roles.setdefault(front, set()).add("front")
                roles.setdefault(rear, set()).add("rear")
                top, left, bottom, right = _bbox(grid != background)
                crop = grid[top:bottom, left:right]
                silhouettes.add((crop.shape, tuple(int(v != background)
                                                for v in crop.flat)))
                occupied = grid != background
                solid_count += _is_rectangle(occupied)
                holes = []
                for component in _components(~occupied[top:bottom, left:right]):
                    points = np.argwhere(component)
                    if (len(points) and np.all(points.min(axis=0) > 0)
                            and np.all(points.max(axis=0)
                                       < np.array(component.shape) - 1)):
                        holes.append(component)
                hole_count += bool(holes)
                multi_hole_count += count == 2 and bool(holes)
                a, b = _bbox(front_mask), _bbox(rear_mask)
                row_overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                col_overlap = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                one_cell_overlap += row_overlap == 1 or col_overlap == 1
                direction = "".join(label for label, condition in (
                    ("T", a[0] < b[0]), ("B", a[2] > b[2]),
                    ("L", a[1] < b[1]), ("R", a[3] > b[3])) if condition) or "I"
                orientations[split].add(direction)
                directions.add(direction)
        role_reversal_puzzles += any(len(v) == 2 for v in roles.values())
        shared_orientation_puzzles += bool(orientations["train"]
                                           & orientations["test"])
        novel_orientation_puzzles += bool(orientations["test"]
                                          - orientations["train"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 45 and len(silhouettes) >= 120
    assert len(palettes) >= 22 and len(backgrounds) >= 4
    assert component_counts == {1, 2} and len(rear_parts) >= 2
    assert role_reversal_puzzles >= 18
    assert solid_count >= 3 and hole_count >= 15 and multi_hole_count >= 5
    assert one_cell_overlap >= 35
    assert {"T", "B", "L", "R", "TL", "TR", "BL", "BR", "I"} <= directions
    assert shared_orientation_puzzles >= 15 and novel_orientation_puzzles >= 10

    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                          (None, 1), (None, 4),
                                          (2, None), (10, None))):
        puzzle = generate_puzzle(100 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(100 + index, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(999, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert not _same_puzzles(generate_puzzle(42), generate_puzzle(43))
