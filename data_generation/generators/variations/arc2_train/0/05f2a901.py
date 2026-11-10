import numpy as np


_DIRECTIONS = ("right", "down", "up", "left")

_SOURCE_PAIR_SIGNATURES = frozenset({
    (
        (
            "0000000000",
            "0220000000",
            "0220000000",
            "2220000000",
            "0220008800",
            "0000008800",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000220000",
            "0000220000",
            "0002220000",
            "0000228800",
            "0000008800",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "022200000",
            "220200000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000880000",
            "000880000",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "000000000",
            "022200000",
            "220200000",
            "000880000",
            "000880000",
            "000000000",
            "000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0008800000",
            "0008800000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0002220000",
            "0222220000",
            "0022000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0008800000",
            "0008800000",
            "0002220000",
            "0222220000",
            "0022000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000020000",
            "0000022000",
            "0880022000",
            "0880002000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0002000000",
            "0002200000",
            "0882200000",
            "0880200000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
})


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid.tolist()
    )


def _pair_is_source_copy(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _rows_signature(input_grid),
        _rows_signature(output_grid),
    ) in _SOURCE_PAIR_SIGNATURES

def _bbox(points):
    return (int(points[:, 0].min()), int(points[:, 0].max()),
            int(points[:, 1].min()), int(points[:, 1].max()))


def _connected(points):
    cells = {tuple(map(int, point)) for point in points}
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == cells


def _rectangle(points):
    t, b, l, r = _bbox(points)
    return len(points) == (b-t+1)*(r-l+1)


def _touch(first, second, diagonal=False):
    distance = np.abs(first[:, None, :] - second[None, :, :])
    if diagonal:
        return bool(np.any(np.max(distance, axis=2) <= 1))
    return bool(np.any(np.sum(distance, axis=2) == 1))


def _direction_and_gap(mover, marker):
    mt, mb, ml, mr = _bbox(mover)
    kt, kb, kl, kr = _bbox(marker)
    row_sep = mb < kt or kb < mt
    col_sep = mr < kl or kr < ml
    assert row_sep != col_sep
    if mr < kl:
        return "right", kl-mr-1
    if kr < ml:
        return "left", ml-kr-1
    if mb < kt:
        return "down", kt-mb-1
    return "up", mt-kb-1


def _delta(direction, gap):
    return {"right": (0, gap), "left": (0, -gap),
            "down": (gap, 0), "up": (-gap, 0)}[direction]


def _further_eight_contact(mover, marker, direction):
    dr, dc = _delta(direction, 1)
    for step in range(1, 11):
        shifted = mover + np.array((dr*step, dc*step))
        if _touch(shifted, marker, diagonal=True):
            return True
    return False


def _inspect_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 3
    background = int(values[np.argmax(counts)])
    assert int(counts.max()) > input_grid.size // 2
    assert set(map(int, np.unique(output_grid))) == set(map(int, values))
    colors = [int(x) for x in values if int(x) != background]
    candidates = []
    for color in colors:
        points = np.argwhere(input_grid == color)
        t, b, l, r = _bbox(points)
        if _rectangle(points) and 2 <= b-t+1 <= 4 and 2 <= r-l+1 <= 4:
            candidates.append(color)
    assert len(candidates) == 1
    marker_color = candidates[0]
    mover_color = colors[0] if colors[1] == marker_color else colors[1]
    marker = np.argwhere(input_grid == marker_color)
    mover = np.argwhere(input_grid == mover_color)
    output_marker = np.argwhere(output_grid == marker_color)
    output_mover = np.argwhere(output_grid == mover_color)
    assert np.array_equal(marker, output_marker)
    assert len(mover) >= 3 and not _rectangle(mover) and _connected(mover)
    assert len(output_mover) == len(mover) and _connected(output_mover)
    direction, gap = _direction_and_gap(mover, marker)
    assert gap >= 1
    dr, dc = _delta(direction, gap)
    shifted = mover + np.array((dr, dc))
    assert np.all(shifted >= 0)
    assert np.all(shifted[:, 0] < input_grid.shape[0])
    assert np.all(shifted[:, 1] < input_grid.shape[1])
    expected = input_grid.copy()
    expected[mover[:, 0], mover[:, 1]] = background
    assert np.all(expected[shifted[:, 0], shifted[:, 1]] == background)
    expected[shifted[:, 0], shifted[:, 1]] = mover_color
    assert np.array_equal(output_grid, expected)
    assert not np.array_equal(input_grid, output_grid)
    shape = _bbox(mover)
    normalized = tuple(sorted((int(r-shape[0]), int(c-shape[2])) for r, c in mover))
    marker_box = _bbox(marker)
    return {
        "background": background, "colors": (background, mover_color, marker_color),
        "marker_size": (marker_box[1]-marker_box[0]+1, marker_box[3]-marker_box[2]+1),
        "mover_area": len(mover), "mover_size": (shape[1]-shape[0]+1, shape[3]-shape[2]+1),
        "mover_shape": normalized, "direction": direction, "gap": gap,
        "size": input_grid.shape, "mover_box": shape, "marker_box": marker_box,
        "orth_contact": _touch(output_mover, marker),
        "eight_contact": _touch(output_mover, marker, diagonal=True),
        "further_contact": _further_eight_contact(output_mover, marker, direction),
    }


def _counts(rng, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test+1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test)+1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)
def _mask(rng):
    style = int(rng.integers(0, 4))
    if style == 0:
        h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
    elif style == 1:
        h, w = int(rng.integers(4, 10)), int(rng.integers(4, 10))
    elif style == 2:
        h, w = int(rng.integers(2, 5)), int(rng.integers(5, 10))
        if rng.integers(2):
            h, w = w, h
    else:
        h, w = int(rng.integers(3, 9)), int(rng.integers(3, 9))
    if style == 1:
        mask = np.ones((h, w), dtype=bool)
        boundary = [(r, c) for r in range(h) for c in range(w)
                    if r in (0, h-1) or c in (0, w-1)]
        for index in rng.permutation(len(boundary))[:int(rng.integers(1, max(2, len(boundary)//2)))]:
            r, c = boundary[int(index)]
            mask[r, c] = False
            if not _connected(np.argwhere(mask)):
                mask[r, c] = True
    elif style == 2:
        mask = np.zeros((h, w), dtype=bool)
        if w >= h:
            row = int(rng.integers(h))
            for col in range(w):
                if col:
                    next_row = min(h-1, max(0, row+int(rng.integers(-1, 2))))
                    mask[min(row, next_row):max(row, next_row)+1, col] = True
                    row = next_row
                mask[row, col] = True
                if rng.random() < 0.55:
                    mask[min(h-1, max(0, row+int(rng.choice((-1, 1))))), col] = True
        else:
            col = int(rng.integers(w))
            for row in range(h):
                if row:
                    next_col = min(w-1, max(0, col+int(rng.integers(-1, 2))))
                    mask[row, min(col, next_col):max(col, next_col)+1] = True
                    col = next_col
                mask[row, col] = True
                if rng.random() < 0.55:
                    mask[row, min(w-1, max(0, col+int(rng.choice((-1, 1)))))] = True
    else:
        max_cells = min(h*w, max(4, int(h*w*(0.48 if style == 0 else 0.72))))
        target = int(rng.integers(3, max_cells+1))
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        while len(cells) < target:
            frontier = sorted({(nr, nc) for r, c in cells
                               for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                               if 0 <= nr < h and 0 <= nc < w and (nr,nc) not in cells})
            cells.add(frontier[int(rng.integers(len(frontier)))])
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
    points = np.argwhere(mask)
    t, b, l, r = _bbox(points)
    mask = mask[t:b+1, l:r+1].copy()
    if mask.shape[0] < 2 or mask.shape[1] < 2 or mask.all():
        return None
    return mask


def _oriented_fallback(direction):
    mask = np.array([[0,0,0,1], [1,1,1,1], [1,1,0,0], [1,0,0,0]], dtype=bool)
    if direction == "left":
        return np.fliplr(mask).copy()
    if direction == "down":
        return mask.T.copy()
    if direction == "up":
        return np.flipud(mask.T).copy()
    return mask


def _offsets(mask, marker_size, direction, contact_mode):
    h, w = mask.shape
    mh, mw = marker_size
    offsets = []
    offset_range = range(-mh+1, h) if direction in ("right", "left") else range(-mw+1, w)
    mover = np.argwhere(mask)
    for offset in offset_range:
        if direction == "right":
            marker = np.array([(r, c) for r in range(offset, offset+mh)
                               for c in range(w, w+mw)])
        elif direction == "left":
            marker = np.array([(r, c) for r in range(offset, offset+mh)
                               for c in range(-mw, 0)])
        elif direction == "down":
            marker = np.array([(r, c) for r in range(h, h+mh)
                               for c in range(offset, offset+mw)])
        else:
            marker = np.array([(r, c) for r in range(-mh, 0)
                               for c in range(offset, offset+mw)])
        orth = _touch(mover, marker)
        eight = _touch(mover, marker, diagonal=True)
        if contact_mode == "touch" and orth:
            offsets.append(offset)
        elif contact_mode == "near" and eight and not orth:
            offsets.append(offset)
        elif contact_mode == "clear" and not eight and _further_eight_contact(mover, marker, direction):
            offsets.append(offset)
        elif contact_mode == "any":
            offsets.append(offset)
    return offsets


def _canvas_extent(rng, extent):
    choice = int(rng.integers(0, 5))
    if choice == 0:
        upper = min(30, extent+3)
    elif choice in (1, 2):
        upper = min(30, extent+int(rng.integers(4, 12)))
    else:
        upper = 30
    lower = max(extent, 5)
    if choice == 4:
        lower = max(lower, 23)
    return int(rng.integers(lower, max(lower, upper)+1))


def _make_pair(rng, direction, gap, marker_size, colors, mode, repeated_mask=None):
    mh, mw = marker_size
    for attempt in range(180):
        mask = (repeated_mask if repeated_mask is not None else
                (_mask(rng) if attempt < 155 else _oriented_fallback(direction)))
        if mask is None:
            continue
        offsets = _offsets(mask, marker_size, direction, mode)
        if not offsets:
            continue
        offset = offsets[int(rng.integers(len(offsets)))]
        h, w = mask.shape
        if direction == "right":
            mr, mc, kr, kc = 0, 0, offset, w+gap
        elif direction == "left":
            mr, mc, kr, kc = 0, mw+gap, offset, 0
        elif direction == "down":
            mr, mc, kr, kc = 0, 0, h+gap, offset
        else:
            mr, mc, kr, kc = mh+gap, 0, 0, offset
        top, left = min(mr, kr), min(mc, kc)
        bottom, right = max(mr+h, kr+mh), max(mc+w, kc+mw)
        extent_h, extent_w = bottom-top, right-left
        if extent_h > 30 or extent_w > 30:
            continue
        gh, gw = _canvas_extent(rng, extent_h), _canvas_extent(rng, extent_w)
        free_h, free_w = gh-extent_h, gw-extent_w
        row_pad = int(rng.integers(free_h+1)) if rng.random() < 0.6 else int(rng.choice((0, free_h)))
        col_pad = int(rng.integers(free_w+1)) if rng.random() < 0.6 else int(rng.choice((0, free_w)))
        mr, kr = mr+row_pad-top, kr+row_pad-top
        mc, kc = mc+col_pad-left, kc+col_pad-left
        background, mover_color, marker_color = colors
        grid = np.full((gh, gw), background, dtype=np.int8)
        grid[kr:kr+mh, kc:kc+mw] = marker_color
        rows, cols = np.nonzero(mask)
        rows, cols = rows+mr, cols+mc
        grid[rows, cols] = mover_color
        if int(np.count_nonzero(grid == background)) <= grid.size//2:
            continue
        dr, dc = _delta(direction, gap)
        answer = grid.copy()
        answer[rows, cols] = background
        answer[rows+dr, cols+dc] = mover_color
        pair = {"input": grid, "output": answer}
        if not _pair_is_source_copy(grid, answer):
            return pair
    raise AssertionError("bounded construction failed")


def _palette(rng):
    background = int(rng.integers(10))
    others = [x for x in range(10) if x != background]
    chosen = rng.choice(others, 2, replace=False)
    return background, int(chosen[0]), int(chosen[1])
def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    marker_size = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4))[
        int(rng.integers(7))]
    variable_palette = bool(rng.integers(2))
    palette = _palette(rng)
    box_extension = bool(rng.integers(4))
    direction_order = list(rng.permutation(_DIRECTIONS))
    if train_count >= 4 and rng.random() < 0.5:
        train_directions = direction_order[:4]
    else:
        train_directions = direction_order[:2]
    while len(train_directions) < train_count:
        train_directions.append(_DIRECTIONS[int(rng.integers(4))])
    train_directions = [train_directions[int(i)] for i in rng.permutation(train_count)]
    held_out = [d for d in _DIRECTIONS if d not in train_directions]
    test_directions = []
    for index in range(test_count):
        if index == 0 and held_out and rng.random() < 0.65:
            test_directions.append(held_out[int(rng.integers(len(held_out)))])
        else:
            test_directions.append(_DIRECTIONS[int(rng.integers(4))])
    train_modes = ["touch"] * train_count
    if box_extension:
        train_modes[int(rng.integers(train_count))] = "clear"
        for i in range(train_count):
            if train_modes[i] == "touch" and rng.random() < 0.4:
                train_modes[i] = "near" if rng.random() < 0.4 else "clear"
    test_modes = []
    for _ in range(test_count):
        if box_extension:
            test_modes.append(("touch", "near", "clear")[int(rng.integers(3))])
        else:
            test_modes.append("touch")
    puzzle = {"train": [], "test": []}
    used = set()
    repeat_shape = train_count >= 3 and bool(rng.integers(2))
    if repeat_shape:
        train_modes[-1] = "touch"
        if box_extension and "clear" not in train_modes:
            train_modes[int(rng.integers(train_count-1))] = "clear"
    saved_mask = None
    for section, directions, modes in (("train", train_directions, train_modes),
                                       ("test", test_directions, test_modes)):
        for index, (direction, mode) in enumerate(zip(directions, modes)):
            for _ in range(30):
                colors = _palette(rng) if variable_palette else palette
                gap = int(rng.integers(1, 11))
                repeated = saved_mask if section == "train" and index == train_count-1 and repeat_shape else None
                pair = _make_pair(rng, direction, gap, marker_size, colors, mode, repeated)
                signature = _grid_signature(pair["input"])
                if signature not in used:
                    used.add(signature)
                    puzzle[section].append(pair)
                    if section == "train" and index == 0:
                        bg = max(((int(np.count_nonzero(pair["input"] == color)), color)
                                  for color in np.unique(pair["input"])))[1]
                        marker_color = next(int(color) for color in np.unique(pair["input"])
                                            if int(color) != bg and _rectangle(np.argwhere(pair["input"] == color)))
                        mover_color = next(int(color) for color in np.unique(pair["input"])
                                           if int(color) not in (bg, marker_color))
                        points = np.argwhere(pair["input"] == mover_color)
                        t, b, l, r = _bbox(points)
                        saved_mask = (pair["input"][t:b+1, l:r+1] == mover_color)
                    break
            else:
                raise AssertionError("duplicate construction")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    infos = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            first, second = pair["input"], pair["output"]
            si, so = _grid_signature(first), _grid_signature(second)
            assert si not in seen_inputs and (si, so) not in seen_pairs
            assert not _pair_is_source_copy(first, second)
            seen_inputs.add(si)
            seen_pairs.add((si, so))
            infos[section].append(_inspect_pair(first, second))
    all_infos = infos["train"] + infos["test"]
    assert len({info["marker_size"] for info in all_infos}) == 1
    assert len({info["direction"] for info in infos["train"]}) >= 2
    if any(not info["orth_contact"] for info in infos["test"]):
        assert any(not info["orth_contact"] for info in infos["train"])
    if any(not info["eight_contact"] for info in infos["test"]):
        assert any(not info["eight_contact"] and info["further_contact"]
                   for info in infos["train"])


def _same_puzzle(first, second):
    return all(_grid_signature(a[name]) == _grid_signature(b[name])
               for section in ("train", "test")
               for a, b in zip(first[section], second[section])
               for name in ("input", "output"))


def self_test() -> None:
    train_counts, test_counts, backgrounds = set(), set(), set()
    directions, marker_sizes, dimensions, shapes, gaps = set(), set(), set(), set(), set()
    edge = corner = large = small = mixed_palette = fixed_palette = larger_marker = 0
    box_witness = clear_test = touching_only = repeated_shape = 0
    changing_training_palette = color_role_swap = repeated_shape_new_direction = 0
    four_direction_train = mixed_test_directions = held_out_test = 0
    square = portrait = landscape = narrow_overlap = overlapping_footprint = 0
    near_contact = source_marker = one_cell_test = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_inspect_pair(pair["input"], pair["output"])
                 for section in ("train", "test") for pair in puzzle[section]]
        training_directions = {info["direction"] for info in infos[:len(puzzle["train"])]}
        testing_directions = {info["direction"] for info in infos[len(puzzle["train"]):]}
        train_infos = infos[:len(puzzle["train"])]
        test_infos = infos[len(puzzle["train"]):]
        four_direction_train += len(training_directions) == 4
        mixed_test_directions += len(testing_directions) > 1
        held_out_test += bool(testing_directions-training_directions)
        mixed_palette += len({info["colors"] for info in infos}) > 1
        fixed_palette += len({info["colors"] for info in infos}) == 1
        changing_training_palette += len({info["colors"] for info in train_infos}) > 1
        color_roles = {color: set() for color in range(10)}
        for info in infos:
            for role, color in enumerate(info["colors"]):
                color_roles[color].add(role)
        color_role_swap += any(len(roles) > 1 for roles in color_roles.values())
        repeated_shape += len({info["mover_shape"] for info in infos}) < len(infos)
        repeated_shape_new_direction += any(
            first["mover_shape"] == second["mover_shape"]
            and first["direction"] != second["direction"]
            for index, first in enumerate(infos) for second in infos[index+1:]
        )
        touching_only += all(info["orth_contact"] for info in infos)
        box_witness += any(not info["eight_contact"] and info["further_contact"]
                           for info in train_infos)
        clear_test += any(not info["eight_contact"] for info in test_infos)
        one_cell_test += any(info["gap"] == 1 for info in test_infos)
        if any(not info["eight_contact"] for info in test_infos):
            assert any(not info["eight_contact"] and info["further_contact"]
                       for info in train_infos)
        if any(not info["orth_contact"] for info in test_infos):
            assert any(not info["orth_contact"] for info in train_infos)
        for info in infos:
            backgrounds.add(info["background"])
            directions.add(info["direction"])
            marker_sizes.add(info["marker_size"])
            dimensions.add(info["size"])
            shapes.add(info["mover_shape"])
            gaps.add(info["gap"])
            large += max(info["size"]) >= 25
            small += max(info["size"]) <= 14
            larger_marker += info["marker_size"][0]*info["marker_size"][1] > info["mover_area"]
            source_marker += info["marker_size"] == (2, 2)
            near_contact += info["eight_contact"] and not info["orth_contact"]
            square += abs(info["size"][0]-info["size"][1]) <= 2
            portrait += info["size"][0] >= info["size"][1]+5
            landscape += info["size"][1] >= info["size"][0]+5
            t, b, l, r = info["mover_box"]
            kt, kb, kl, kr = info["marker_box"]
            edge += t == 0 or l == 0 or b == info["size"][0]-1 or r == info["size"][1]-1
            corner += ((t == 0 or b == info["size"][0]-1) and
                       (l == 0 or r == info["size"][1]-1))
            overlap = (min(b, kb)-max(t, kt)+1 if info["direction"] in ("left", "right")
                       else min(r, kr)-max(l, kl)+1)
            narrow_overlap += overlap == 1
            signatures.add((info["size"], info["direction"], info["gap"],
                            info["mover_shape"], info["mover_box"], info["marker_box"]))
        for pair, info in zip(puzzle["train"]+puzzle["test"], infos):
            color = info["colors"][1]
            first = set(map(tuple, np.argwhere(pair["input"] == color)))
            second = set(map(tuple, np.argwhere(pair["output"] == color)))
            overlapping_footprint += bool(first & second)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert backgrounds == set(range(10)) and directions == set(_DIRECTIONS)
    assert len(marker_sizes) >= 4 and len(dimensions) >= 60 and len(shapes) >= 100
    assert gaps == set(range(1, 11)) and len(signatures) >= 250
    assert edge >= 10 and large >= 25 and small >= 20
    assert corner >= 2 and square >= 5 and portrait >= 5 and landscape >= 5
    assert narrow_overlap >= 10 and overlapping_footprint >= 10
    assert mixed_palette >= 8 and fixed_palette >= 8
    assert changing_training_palette >= 8 and color_role_swap >= 8
    assert touching_only >= 2 and box_witness >= 8 and clear_test >= 2
    assert near_contact >= 2 and source_marker >= 2 and one_cell_test >= 2
    assert repeated_shape >= 8 and four_direction_train >= 3
    assert repeated_shape_new_direction >= 5
    assert mixed_test_directions >= 3 and held_out_test >= 3
    assert larger_marker >= 2
    for i, (nt, ns) in enumerate(((2,1), (10,1), (10,4),
                                  (None,1), (None,4), (2,None), (10,None))):
        puzzle = generate_puzzle(1000+i, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
    assert _same_puzzle(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _same_puzzle(generate_puzzle(314159, 7, 3), generate_puzzle(314160, 7, 3))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(99, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    damaged = generate_puzzle(271828, 3, 1)
    changed = damaged["test"][0]["output"].copy()
    changed[0, 0] = (int(changed[0, 0])+1) % 10
    damaged["test"][0]["output"] = changed
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
