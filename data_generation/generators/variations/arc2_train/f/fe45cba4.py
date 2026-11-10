import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _array_from_rows(rows):
    return np.array(
        [[ord(ch) - ord("0") for ch in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _array_from_rows((
            "77777999",
            "77777799",
            "27777999",
            "22277777",
            "22277222",
            "27777772",
            "77777772",
            "77777222",
        )),
        _array_from_rows((
            "77777999",
            "77777799",
            "77777999",
            "77777777",
            "77772222",
            "77772222",
            "77772222",
            "77772222",
        )),
    ),
    (
        _array_from_rows((
            "77777999",
            "97777799",
            "99777999",
            "97777777",
            "77777222",
            "77777772",
            "77777772",
            "77777222",
        )),
        _array_from_rows((
            "77779999",
            "77779999",
            "77779999",
            "77777777",
            "77777222",
            "77777772",
            "77777772",
            "77777222",
        )),
    ),
    (
        _array_from_rows((
            "7777777777999999",
            "7777777777799999",
            "7777777777779999",
            "7777777777799999",
            "7777777777999999",
            "8777777777777777",
            "8878777777777777",
            "8888877777777777",
            "8878777777888888",
            "8777777777787888",
            "7777777777777788",
            "7777777777787888",
            "7777777777888888",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
        )),
        _array_from_rows((
            "7777777777999999",
            "7777777777799999",
            "7777777777779999",
            "7777777777799999",
            "7777777777999999",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777778888888",
            "7777777778888888",
            "7777777778888888",
            "7777777778888888",
            "7777777778888888",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
        )),
    ),
)


def _validate_requested_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(seed, num_train, num_test):
    _validate_requested_count("num_train", num_train, 2, 10)
    _validate_requested_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low_train = max(2, num_test + 1)
        span = 11 - low_train
        num_train = low_train + (seed + seed // 4) % span
    elif num_train is None:
        num_test = int(num_test)
        low_train = max(2, num_test + 1)
        span = 11 - low_train
        num_train = low_train + (seed + 3 * num_test) % span
    elif num_test is None:
        num_train = int(num_train)
        high_test = min(4, num_train - 1)
        if high_test < 1:
            raise ValueError("no legal test count exists")
        num_test = 1 + seed % high_test
    else:
        num_train = int(num_train)
        num_test = int(num_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _components_of_mask(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            cells = set()

            while stack:
                current_row, current_col = stack.pop()
                cells.add((current_row, current_col))

                for delta_row, delta_col in _NEIGHBORS:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

            components.append(frozenset(cells))

    return components


def _components(grid, color):
    return _components_of_mask(grid == color)


def _mask_is_connected(mask):
    return bool(np.any(mask)) and len(_components_of_mask(mask)) == 1


def _bbox(cells):
    assert cells
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _rect_cells(box):
    row_min, row_max, col_min, col_max = box
    return frozenset(
        (row, col)
        for row in range(row_min, row_max + 1)
        for col in range(col_min, col_max + 1)
    )


def _is_rectangle(cells):
    return bool(cells) and frozenset(cells) == _rect_cells(_bbox(cells))


def _normalized_shape(cells):
    row_min, _, col_min, _ = _bbox(cells)
    return tuple(
        sorted((row - row_min, col - col_min) for row, col in cells)
    )


def _cells(mask, top=0, left=0):
    return frozenset((top + int(r), left + int(c)) for r, c in np.argwhere(mask))


def _translate(cells, dr, dc):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _make_partition(rng, height, width, side, intricate):
    # A variable boundary gives two connected pieces; local changes make bays.
    if side in ("top", "bottom"):
        base_h, base_w = width, height
    else:
        base_h, base_w = height, width
    right_gap = int(rng.integers(1, min(4, base_w)))
    maximum = base_w - right_gap
    minimum = int(rng.integers(1, min(4, maximum + 1)))
    boundaries = rng.integers(minimum, maximum + 1, size=base_h)
    boundaries[int(rng.integers(base_h))] = minimum
    boundaries[int(rng.integers(base_h))] = maximum
    fixed = np.zeros((base_h, base_w), dtype=bool)
    for r, b in enumerate(boundaries):
        fixed[r, int(b):] = True
    if intricate:
        n_changes = int(rng.integers(1, max(2, base_h // 2 + 1)))
        for _ in range(n_changes):
            candidates = []
            for r in range(base_h):
                row_start = next(c for c in range(base_w) if fixed[r, c])
                for c in range(1, base_w - 1):
                    if not fixed[r, c] or c <= row_start:
                        continue
                    if any(0 <= r + dr < base_h and not fixed[r + dr, c]
                           for dr in (-1, 1)):
                        candidates.append((r, c))
            rng.shuffle(candidates)
            for r, c in candidates:
                trial = fixed.copy()
                trial[r, c] = False
                if _mask_is_connected(trial) and _mask_is_connected(~trial):
                    fixed = trial
                    break
    if side == "right":
        fixed = np.fliplr(fixed)
    elif side == "top":
        fixed = fixed.T
    elif side == "bottom":
        fixed = np.flipud(fixed.T)
    moving = ~fixed
    assert fixed.shape == (height, width)
    assert _mask_is_connected(fixed) and _mask_is_connected(moving)
    return fixed, moving


def _make_other_mask(rng, height, width):
    # A randomized connected walk spans its box, followed by irregular growth.
    mask = np.zeros((height, width), dtype=bool)
    steps = [0] * (height - 1) + [1] * (width - 1)
    rng.shuffle(steps)
    r = c = 0
    mask[r, c] = True
    for direction in steps:
        if direction == 0:
            r += 1
        else:
            c += 1
        mask[r, c] = True
    wanted = int(rng.integers(mask.sum(), height * width + 1))
    while int(mask.sum()) < wanted:
        frontier = [(r, c) for r in range(height) for c in range(width)
                    if not mask[r, c] and any(
                        0 <= r + dr < height and 0 <= c + dc < width
                        and mask[r + dr, c + dc] for dr, dc in _NEIGHBORS)]
        if not frontier:
            break
        r, c = frontier[int(rng.integers(len(frontier)))]
        mask[r, c] = True
    return mask


def _packings(shape, fixed, moving, other):
    # Enumerate every rectangle and every in-bounds translation, with no
    # directional or fixed-edge convention. Its area determines candidates.
    height, width = shape
    area = len(fixed) + len(moving)
    fr0, fr1, fc0, fc1 = _bbox(fixed)
    mr0, _, mc0, _ = _bbox(moving)
    found = []
    for rect_h in range(1, min(height, area) + 1):
        if area % rect_h:
            continue
        rect_w = area // rect_h
        if rect_w > width:
            continue
        for top in range(max(0, fr1 - rect_h + 1), min(fr0, height - rect_h) + 1):
            for left in range(max(0, fc1 - rect_w + 1), min(fc0, width - rect_w) + 1):
                rectangle = _rect_cells((top, top + rect_h - 1, left, left + rect_w - 1))
                complement = rectangle - fixed
                if not complement or not complement.isdisjoint(other):
                    continue
                cr0, _, cc0, _ = _bbox(complement)
                dr, dc = cr0 - mr0, cc0 - mc0
                if dr == 0 and dc == 0:
                    continue
                if _translate(moving, dr, dc) == complement:
                    found.append((dr, dc, complement, rectangle))
    return found


def _lane_axis(fixed, other):
    fr0, fr1, fc0, fc1 = _bbox(fixed)
    or0, or1, oc0, oc1 = _bbox(other)
    vertical = fc0 == oc0 and fc1 == oc1 and (fr1 + 1 < or0 or or1 + 1 < fr0)
    horizontal = fr0 == or0 and fr1 == or1 and (fc1 + 1 < oc0 or oc1 + 1 < fc0)
    if vertical == horizontal:
        return None
    return "vertical" if vertical else "horizontal"


def _solve_input(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) == 3 and np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[int(np.argmax(counts))])
    foreground = [int(c) for c in colors if int(c) != background]
    possible = []
    for assembly in foreground:
        same = _components(input_grid, assembly)
        other_color = next(c for c in foreground if c != assembly)
        other_components = _components(input_grid, other_color)
        if len(same) != 2 or len(other_components) != 1:
            continue
        other = other_components[0]
        for fixed, moving in ((same[0], same[1]), (same[1], same[0])):
            lane = _lane_axis(fixed, other)
            if lane is None:
                continue
            for dr, dc, added, rectangle in _packings(input_grid.shape, fixed, moving, other):
                output = input_grid.copy()
                for r, c in moving:
                    output[r, c] = background
                for r, c in added:
                    output[r, c] = assembly
                possible.append({
                    "output": output, "background": background,
                    "assembly": assembly, "other_color": other_color,
                    "fixed": fixed, "moving": moving, "other": other,
                    "added": added, "rectangle": rectangle,
                    "lane": lane, "shift": (dr, dc),
                })
    # Even two identical output grids can encode competing object explanations.
    assert len(possible) == 1
    return possible[0]


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    solution = _solve_input(input_grid)
    assert len(_components_of_mask(input_grid != solution["background"])) == 3
    assert np.array_equal(solution["output"], output_grid)
    assert len(_components_of_mask(output_grid != solution["background"])) == 2
    assert set(map(int, np.unique(input_grid))) == set(map(int, np.unique(output_grid)))
    fixed, moving, other = (solution[k] for k in ("fixed", "moving", "other"))
    rectangle = solution["rectangle"]
    rb = _bbox(rectangle)
    fb = _bbox(fixed)
    mb = _bbox(moving)
    ob = _bbox(other)
    if solution["lane"] == "vertical":
        assert mb[3] < min(fb[2], ob[2]) or mb[2] > max(fb[3], ob[3])
    else:
        assert mb[1] < min(fb[0], ob[0]) or mb[0] > max(fb[1], ob[1])
    return {
        **solution,
        "shape": input_grid.shape,
        "colors": frozenset(map(int, np.unique(input_grid))),
        "rect_box": rb,
        "fixed_box": fb,
        "moving_box": mb,
        "other_box": ob,
        "rect_size": (rb[1] - rb[0] + 1, rb[3] - rb[2] + 1),
        "target_before": (ob[0] > fb[1] if solution["lane"] == "vertical" else ob[2] > fb[3]),
        "completion_side": (
            "left" if rb[2] < fb[2] else
            "right" if rb[3] > fb[3] else
            "top" if rb[0] < fb[0] else "bottom"
        ),
    }


def _construct_pair(rng, regime, palette, assembly_index, lane, target_before):
    if regime == "source_train":
        rect_h = int(rng.integers(3, 8))
        rect_w = int(rng.integers(3, 8))
        intricate = bool(rng.integers(0, 3))
    else:
        rect_h = int(rng.integers(4, 10))
        rect_w = int(rng.integers(4, 10))
        intricate = bool(rng.integers(0, 4))
    completion_side = ("left", "right", "top", "bottom")[int(rng.integers(4))]
    fixed_mask, moving_mask = _make_partition(rng, rect_h, rect_w, completion_side, intricate)
    fixed = _cells(fixed_mask)
    destination = _cells(moving_mask)
    fb = _bbox(fixed)
    gap = int(rng.integers(1, 4))
    if lane == "vertical":
        other_h = int(rng.integers(2, 7))
        other_w = fb[3] - fb[2] + 1
        other_mask = _make_other_mask(rng, other_h, other_w)
        other_top = -gap - other_h if target_before else rect_h + gap
        other = _cells(other_mask, other_top, fb[2])
    else:
        other_h = fb[1] - fb[0] + 1
        other_w = int(rng.integers(2, 7))
        other_mask = _make_other_mask(rng, other_h, other_w)
        other_left = -gap - other_w if target_before else rect_w + gap
        other = _cells(other_mask, fb[0], other_left)
    if not destination.isdisjoint(other):
        return None
    moving_box = _bbox(destination)
    mh, mw = moving_box[1] - moving_box[0] + 1, moving_box[3] - moving_box[2] + 1
    lane_cells = fixed | destination | other
    l0, l1, c0, c1 = _bbox(lane_cells)
    detached_side = int(rng.integers(2))
    travel_gap = int(rng.integers(1, 6))
    if lane == "vertical":
        origin_left = c0 - travel_gap - mw if detached_side == 0 else c1 + travel_gap + 1
        origin_top = int(rng.integers(l0 - 2, max(l0 - 1, l1 - mh + 4)))
    else:
        origin_top = l0 - travel_gap - mh if detached_side == 0 else l1 + travel_gap + 1
        origin_left = int(rng.integers(c0 - 2, max(c0 - 1, c1 - mw + 4)))
    origin = _translate(destination, origin_top - moving_box[0], origin_left - moving_box[2])
    if not origin.isdisjoint(lane_cells):
        return None
    all_cells = lane_cells | origin
    aminr, amaxr, aminc, amaxc = _bbox(all_cells)
    top_margin = int(rng.integers(0, 5))
    left_margin = int(rng.integers(0, 5))
    bottom_margin = int(rng.integers(0, 5))
    right_margin = int(rng.integers(0, 5))
    height = amaxr - aminr + 1 + top_margin + bottom_margin
    width = amaxc - aminc + 1 + left_margin + right_margin
    if height > 30 or width > 30:
        return None
    dr, dc = top_margin - aminr, left_margin - aminc
    fixed, destination, other, origin = (_translate(x, dr, dc) for x in (fixed, destination, other, origin))
    background, first, second = palette
    assembly = (first, second)[assembly_index]
    other_color = (second, first)[assembly_index]
    grid = np.full((height, width), background, dtype=np.int8)
    for r, c in fixed | origin:
        grid[r, c] = assembly
    for r, c in other:
        grid[r, c] = other_color
    output = grid.copy()
    for r, c in origin:
        output[r, c] = background
    for r, c in destination:
        output[r, c] = assembly
    return {"input": grid, "output": output}


def _matches_source_pair(input_grid, output_grid):
    return any(input_grid.shape == original_input.shape
               and np.array_equal(input_grid, original_input)
               and np.array_equal(output_grid, original_output)
               for original_input, original_output in _SOURCE_PAIRS)


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _sample_palette(rng, base, regime, index, mixed):
    if not mixed or regime == "source_train" and index < 2:
        return base
    mode = int(rng.integers(5))
    if mode == 0:
        return base
    available = [c for c in range(10) if c not in base]
    rng.shuffle(available)
    if mode == 1:
        return (base[0], available[0], base[2])
    if mode == 2:
        return (base[0], base[1], available[0])
    if mode == 3:
        return (available[0], base[1], base[2])
    return tuple(int(c) for c in rng.choice(10, 3, replace=False))


def _first_test_palette(rng, base, role, style):
    if style == 0:
        return base
    choices = [c for c in range(10) if c not in base]
    replacement = int(choices[int(rng.integers(len(choices)))])
    result = list(base)
    index = (1 + role) if style == 1 else (2 - role) if style == 2 else 0
    result[index] = replacement
    return tuple(result)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    palette_style = seed % 4
    mixed_palette = palette_style != 0
    mixed_axis = bool(rng.integers(2))
    base_axis = ("vertical", "horizontal")[int(rng.integers(2))]
    first_role = int(rng.integers(2))
    first_position = bool(rng.integers(2))
    pair_seeds = rng.integers(0, 2**63, size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_number = index if split == "train" else num_train + index
            pair_rng = np.random.default_rng(int(pair_seeds[pair_number]))
            if split == "train" and index < 2:
                role = first_role ^ index
                position = first_position ^ bool(index)
            elif split == "train" and index == 2:
                role = first_role
                position = not first_position
            else:
                role = int(pair_rng.integers(2))
                position = bool(pair_rng.integers(2))
            axis = (base_axis if not mixed_axis or index % 2 == 0 else
                    ("horizontal" if base_axis == "vertical" else "vertical")) if split == "train" else (
                    base_axis if not mixed_axis else ("vertical", "horizontal")[int(pair_rng.integers(2))])
            palette = (base if split == "train" and index == 2 else
                       _first_test_palette(pair_rng, base, role, palette_style)
                       if split == "test" and index == 0 else
                       _sample_palette(pair_rng, base, "source_train" if split == "train" else "target_test", index, mixed_palette))
            for _ in range(2500):
                pair = _construct_pair(pair_rng, "source_train" if split == "train" else "target_test", palette, role, axis, position)
                if pair is None or _grid_key(pair["input"]) in used_inputs:
                    continue
                if _matches_source_pair(pair["input"], pair["output"]):
                    continue
                try:
                    _analyze_pair(pair["input"], pair["output"])
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                used_inputs.add(_grid_key(pair["input"]))
                break
            else:
                raise ValueError(f"unable to generate {split} pair for seed {seed}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
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
            assert not _matches_source_pair(inp, out)
            assert _grid_key(inp) not in seen_inputs and _pair_key(pair) not in seen_pairs
            seen_inputs.add(_grid_key(inp))
            seen_pairs.add(_pair_key(pair))
            a = _analyze_pair(inp, out)
            rh, rw = a["rect_size"]
            if split == "train":
                assert 3 <= rh <= 7 and 3 <= rw <= 7
            else:
                assert 4 <= rh <= 9 and 4 <= rw <= 9
            analyses[split].append(a)
    train = analyses["train"]
    assert train[0]["colors"] == train[1]["colors"]
    assert train[0]["background"] == train[1]["background"]
    assert train[0]["assembly"] != train[1]["assembly"]
    assert train[0]["target_before"] != train[1]["target_before"]
    if len(train) >= 3:
        assert train[2]["colors"] == train[0]["colors"]
        assert train[2]["assembly"] == train[0]["assembly"]
        assert train[2]["target_before"] == train[1]["target_before"]
    train_axes = {a["lane"] for a in train}
    assert {a["lane"] for a in analyses["test"]} <= train_axes


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[k], b[k])) for k in ("train", "test"))


def self_test() -> None:
    samples = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in samples}) >= 4
    assert len({len(p["test"]) for p in samples}) >= 3
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert not _puzzles_equal(generate_puzzle(17), generate_puzzle(18))
    for p in samples:
        validate_generated_puzzle(p)
    for n, m in ((2, 1), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(200 + n * 10 + m, n, m)
        assert len(p["train"]) == n and len(p["test"]) == m
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(300 + (n or 0) + (m or 0), n, m)
        assert (n is None or len(p["train"]) == n) and (m is None or len(p["test"]) == m)
    for n, m in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(400, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
    all_a = [_analyze_pair(pair["input"], pair["output"])
             for p in samples for split in ("train", "test") for pair in p[split]]
    assert len({_grid_key(pair["input"]) for p in samples for split in ("train", "test") for pair in p[split]}) > 150
    assert {a["lane"] for a in all_a} == {"vertical", "horizontal"}
    assert {a["completion_side"] for a in all_a} == {"left", "right", "top", "bottom"}
    assert len({a["rect_size"] for a in all_a}) >= 20
    assert len({a["shape"] for a in all_a}) >= 30
    assert len({a["colors"] for a in all_a}) >= 25
    assert len({(a["lane"], a["completion_side"]) for a in all_a}) == 8
    assert any(a["rect_box"][3] == a["shape"][1] - 1 for a in all_a)
    assert any(a["rect_box"][3] < a["shape"][1] - 1 for a in all_a)
    assert any(a["rect_size"][1] - (a["moving_box"][3] - a["moving_box"][2] + 1) >= 2
               for a in all_a)
    assert any(abs(a["shift"][0]) + abs(a["shift"][1]) >= 12 for a in all_a)
    assert len({_normalized_shape(a["other"]) for a in all_a}) >= 100
    assert {len({_analyze_pair(pair["input"], pair["output"])["lane"]
                 for pair in p["train"]}) for p in samples} == {1, 2}
    assert {len({frozenset(map(int, np.unique(pair["input"])))
                 for split in ("train", "test") for pair in p[split]}) == 1
            for p in samples} == {False, True}
    train_dims = {_analyze_pair(pair["input"], pair["output"])["rect_size"]
                  for p in samples for pair in p["train"]}
    test_dims = {_analyze_pair(pair["input"], pair["output"])["rect_size"]
                 for p in samples for pair in p["test"]}
    assert len(train_dims & test_dims) >= 5
    def has_bay(a):
        fixed = a["fixed"]
        box = a["fixed_box"]
        profiles = (
            (sorted(c for r, c in fixed if r == row)
             for row in range(box[0], box[1] + 1))
            if a["completion_side"] in ("left", "right") else
            (sorted(r for r, c in fixed if c == col)
             for col in range(box[2], box[3] + 1))
        )
        return any(p and len(p) < p[-1] - p[0] + 1 for p in profiles)
    assert any(has_bay(_analyze_pair(pair["input"], pair["output"]))
               for p in samples for pair in p["train"])
    assert any(has_bay(_analyze_pair(pair["input"], pair["output"]))
               for p in samples for pair in p["test"])
    first_test_styles = set()
    for p in samples:
        train_palette = _analyze_pair(p["train"][0]["input"], p["train"][0]["output"])["colors"]
        test = _analyze_pair(p["test"][0]["input"], p["test"][0]["output"])
        if test["colors"] == train_palette:
            first_test_styles.add("familiar")
        elif test["assembly"] not in train_palette:
            first_test_styles.add("new_assembly")
        elif test["other_color"] not in train_palette:
            first_test_styles.add("new_other")
        else:
            first_test_styles.add("new_background")
    assert first_test_styles == {"familiar", "new_assembly", "new_other", "new_background"}
