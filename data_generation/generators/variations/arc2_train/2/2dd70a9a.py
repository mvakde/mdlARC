import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

_SOURCE_INPUT_ROWS = (
    (
        "0000000008",
        "0380000000",
        "0300000800",
        "0000008008",
        "0808000000",
        "0008000000",
        "0880020000",
        "0080020000",
        "0080000000",
        "8000000000",
    ),
    (
        "00008888800808880808",
        "08000008000008080000",
        "88888080800088800200",
        "80880000008888880200",
        "80088000000808000000",
        "00080000000008000000",
        "80000000000000000080",
        "00800000000000000088",
        "80000000000080000088",
        "00000000080080000088",
        "08000088808008088000",
        "80000800008888888888",
        "00000088800888080088",
        "00000088000080008008",
        "00030008080800800808",
        "00030088800088880000",
        "08000008088080808000",
        "00000008000800000880",
        "00080008080088800008",
        "00008888008000088800",
    ),
    (
        "000008080080080",
        "000800800008088",
        "800088800008880",
        "000008080000000",
        "000000000800080",
        "033000008000800",
        "000000000008080",
        "088008008088000",
        "080000000000000",
        "822000000000080",
        "800000088800000",
        "080080800088880",
        "000008080000000",
        "000008808008008",
        "080088000000000",
    ),
    (
        "8888000008800",
        "8000082200000",
        "0800880000000",
        "0080000080008",
        "0080008000008",
        "0008000080800",
        "0000000000800",
        "8083300000800",
        "0880080000800",
        "0000008800000",
        "0880008000000",
        "0008000000000",
        "0000000000080",
    ),
)


def _turn(direction, hand):
    dr, dc = direction
    if hand == "R":
        return dc, -dr
    assert hand == "L"
    return -dc, dr


def _turn_hand(before, after):
    if after == _turn(before, "R"):
        return "R"
    if after == _turn(before, "L"):
        return "L"
    raise AssertionError("directions are not perpendicular")


def _add(point, direction, amount=1):
    return (
        point[0] + amount * direction[0],
        point[1] + amount * direction[1],
    )


def _inside(shape, point):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.ravel())


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _is_original_source_input(grid):
    rows = tuple(
        "".join(str(int(value)) for value in grid[row])
        for row in range(grid.shape[0])
    )
    return rows in _SOURCE_INPUT_ROWS


def _bar(grid, color):
    cells = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
    if not 2 <= len(cells) <= 5:
        return None
    rows = {p[0] for p in cells}
    cols = {p[1] for p in cells}
    if len(rows) == 1:
        ordered = sorted(cells, key=lambda p: p[1])
        axis = "H"
    elif len(cols) == 1:
        ordered = sorted(cells, key=lambda p: p[0])
        axis = "V"
    else:
        return None
    if any(_add(a, (0, 1) if axis == "H" else (1, 0)) != b
           for a, b in zip(ordered, ordered[1:])):
        return None
    return ordered, axis


def _simple_chain(chain):
    if len(set(chain)) != len(chain):
        return False
    locations = {point: index for index, point in enumerate(chain)}
    for index, point in enumerate(chain):
        for direction in _DIRS:
            neighbor = _add(point, direction)
            if neighbor in locations and abs(index - locations[neighbor]) != 1:
                return False
    return True


def _find_routes(grid, background, start, target, blocker, max_bends=8):
    start_bar = _bar(grid, start)
    target_bar = _bar(grid, target)
    if start_bar is None or target_bar is None or start_bar[1] != target_bar[1]:
        return []
    starts, _ = start_bar
    targets, _ = target_bar
    routes = []

    def walk(position, direction, path, directions, lengths, tip):
        local = []
        seen = set(path)
        step = _add(position, direction)
        while _inside(grid.shape, step):
            value = int(grid[step])
            if value == background:
                if step in seen:
                    return
                local.append(step)
                seen.add(step)
                step = _add(step, direction)
                continue
            if value == blocker:
                if not local or len(directions) - 1 >= max_bends:
                    return
                for hand in ("L", "R"):
                    next_direction = _turn(direction, hand)
                    walk(local[-1], next_direction, path + tuple(local),
                         directions + (next_direction,), lengths + (len(local),), tip)
                return
            if value == target:
                bends = len(directions) - 1
                if bends not in (0, 2, 4, 6, 8) or not local:
                    return
                expected = {_add(step, direction, k) for k in range(len(targets))}
                if expected != set(targets):
                    return
                full_path = path + tuple(local)
                start_line = sorted(starts, key=lambda p: p[0] * direction0[0] + p[1] * direction0[1])
                target_line = [_add(step, direction, k) for k in range(len(targets))]
                if not _simple_chain(start_line + list(full_path) + target_line):
                    return
                routes.append({"path": full_path, "directions": directions,
                               "lengths": lengths + (len(local),),
                               "start_tip": tip, "target_tip": step})
                return
            return

    for tip, inner in ((starts[0], starts[1]), (starts[-1], starts[-2])):
        direction0 = (tip[0] - inner[0], tip[1] - inner[1])
        walk(tip, direction0, (), (direction0,), (), tip)
    unique = {}
    for route in routes:
        unique[(route["path"], route["directions"])] = route
    return list(unique.values())


def _analyze_pair(input_grid, output_grid, known_markers=None):
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert bool(np.any(changed))
    old = {int(x) for x in np.unique(input_grid[changed])}
    new = {int(x) for x in np.unique(output_grid[changed])}
    assert len(old) == len(new) == 1
    background, start = next(iter(old)), next(iter(new))
    assert background != start
    colors = {int(x) for x in np.unique(input_grid)}
    assert len(colors) == 4 and {int(x) for x in np.unique(output_grid)} == colors
    assert background in colors and start in colors
    start_bar = _bar(input_grid, start)
    assert start_bar is not None
    candidates = [c for c in colors - {background, start}
                  if (bar := _bar(input_grid, c)) is not None
                  and bar[1] == start_bar[1]]
    assert len(candidates) == 1
    target = candidates[0]
    if known_markers is not None:
        assert (start, target) == known_markers
    blocker = next(iter(colors - {background, start, target}))
    assert int(np.count_nonzero(input_grid == blocker)) >= 5
    routes = _find_routes(input_grid, background, start, target, blocker)
    assert len(routes) == 1
    route = routes[0]
    assert len(route["directions"]) in (3, 5)
    assert all(length >= 1 for length in route["lengths"])
    changed_cells = {tuple(map(int, p)) for p in np.argwhere(changed)}
    assert changed_cells == set(route["path"])
    assert all(int(input_grid[p]) == background and int(output_grid[p]) == start
               for p in route["path"])
    hands = tuple(_turn_hand(a, b) for a, b in
                  zip(route["directions"], route["directions"][1:]))
    return {"roles": (background, start, target, blocker),
            "orientation": start_bar[1], "directions": route["directions"],
            "hands": hands, "lengths": route["lengths"],
            "path": route["path"], "marker_lengths": (len(start_bar[0]), len(_bar(input_grid, target)[0])),
            "bends": len(hands)}


def _resolve_counts(rng, num_train, num_test):
    def checked(value, name, low, high):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} is outside its legal range")
        return int(value)
    if num_train is not None:
        num_train = checked(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = checked(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _length(rng, bends):
    choice = int(rng.integers(10))
    if choice < 2:
        return 1
    if choice < 5:
        return int(rng.integers(2, 6))
    if choice < 9 or bends == 4:
        return int(rng.integers(6, 11))
    return int(rng.integers(11, 24))


def _canvas_dimension(rng, extent):
    if extent > 30:
        return None
    choice = int(rng.integers(10))
    if choice < 2:
        extra = int(rng.integers(0, min(4, 30 - extent) + 1))
    elif choice < 6:
        low, high = min(4, 30 - extent), min(12, 30 - extent)
        extra = int(rng.integers(low, high + 1))
    elif choice < 9:
        low = min(13, 30 - extent)
        extra = int(rng.integers(low, 31 - extent))
    else:
        extra = 30 - extent
    leading = int(rng.integers(extra + 1))
    return extent + extra, leading


def _make_pair_attempt(rng, colors, spec):
    bends, direction0, hands, marker_lengths = spec
    directions = [direction0]
    for hand in hands:
        directions.append(_turn(directions[-1], hand))
    lengths = tuple(_length(rng, bends) for _ in directions)
    start_tip = (0, 0)
    start_cells = tuple(_add(start_tip, direction0, -k)
                        for k in range(marker_lengths[0]))
    current = start_tip
    path, key_blockers = [], []
    for index, (direction, length) in enumerate(zip(directions, lengths)):
        for _ in range(length):
            current = _add(current, direction)
            path.append(current)
        if index < bends:
            key_blockers.append(_add(current, direction))
    target_cells = tuple(_add(current, directions[-1], k)
                         for k in range(1, marker_lengths[1] + 1))
    chain = list(reversed(start_cells)) + path + list(target_cells)
    if not _simple_chain(chain):
        return None
    occupied = set(chain)
    if len(set(key_blockers)) != bends or any(p in occupied for p in key_blockers):
        return None
    structural = occupied | set(key_blockers)
    rows, cols = [p[0] for p in structural], [p[1] for p in structural]
    height_data = _canvas_dimension(rng, max(rows) - min(rows) + 1)
    width_data = _canvas_dimension(rng, max(cols) - min(cols) + 1)
    if height_data is None or width_data is None:
        return None
    height, top = height_data
    width, left = width_data
    shift = (-min(rows) + top, -min(cols) + left)
    def placed(p):
        return p[0] + shift[0], p[1] + shift[1]
    path = tuple(placed(p) for p in path)
    key_blockers = tuple(placed(p) for p in key_blockers)
    start_cells = tuple(placed(p) for p in start_cells)
    target_cells = tuple(placed(p) for p in target_cells)
    background, start, target, blocker = colors
    grid = np.full((height, width), background, dtype=np.int8)
    for p in key_blockers:
        grid[p] = blocker
    for p in start_cells:
        grid[p] = start
    for p in target_cells:
        grid[p] = target
    protected = set(path) | set(key_blockers) | set(start_cells) | set(target_cells)
    available = [(r, c) for r in range(height) for c in range(width)
                 if (r, c) not in protected]
    positions = {p: k for k, p in enumerate(available)}
    chosen = list(key_blockers)
    density_mode = int(rng.integers(10))
    if density_mode < 3:
        density = float(rng.uniform(0.02, 0.09))
    elif density_mode < 7:
        density = float(rng.uniform(0.09, 0.21))
    else:
        density = float(rng.uniform(0.22, 0.37))
    desired = max(5, int(round(height * width * density)))
    cluster_probability = float(rng.uniform(0.15, 0.85))
    while len(chosen) < desired and available:
        point = None
        if chosen and float(rng.random()) < cluster_probability:
            anchor = chosen[int(rng.integers(len(chosen)))]
            neighbors = [_add(anchor, direction) for direction in _DIRS
                         if _add(anchor, direction) in positions]
            if neighbors:
                point = neighbors[int(rng.integers(len(neighbors)))]
        if point is None:
            point = available[int(rng.integers(len(available)))]
        index = positions.pop(point)
        last = available.pop()
        if index < len(available):
            available[index] = last
            positions[last] = index
        chosen.append(point)
        grid[point] = blocker
    output = grid.copy()
    for point in path:
        output[point] = start
    if _is_original_source_input(grid):
        return None
    try:
        info = _analyze_pair(grid, output, (start, target))
        assert info["directions"] == tuple(directions)
        assert info["lengths"] == lengths
    except AssertionError:
        return None
    return {"input": grid, "output": output}


def _make_specs(rng, train_count, test_count):
    mode = int(rng.integers(4))
    if mode == 0:
        fixed = "H" if int(rng.integers(2)) == 0 else "V"
        train_axes = [fixed] * train_count
        test_axes = [fixed] * test_count
    elif mode == 1:
        train_axes = ["H", "V"] + ["H" if int(rng.integers(2)) == 0 else "V"
                                     for _ in range(train_count - 2)]
        test_axes = ["H" if int(rng.integers(2)) == 0 else "V" for _ in range(test_count)]
    else:
        majority = "H" if mode == 2 else "V"
        minority = "V" if mode == 2 else "H"
        train_axes = [minority] + [majority] * (train_count - 1)
        test_axes = [minority if float(rng.random()) < 0.8 else majority
                     for _ in range(test_count)]
    rng.shuffle(train_axes)
    extended = float(rng.random()) < 0.65
    four = train_count >= 3 and float(rng.random()) < 0.65
    train_bends = [2] * train_count
    if four:
        train_bends[int(rng.integers(train_count))] = 4
        if train_count >= 6 and float(rng.random()) < 0.35:
            train_bends[int(rng.integers(train_count))] = 4
    rng.shuffle(train_bends)
    test_bends = [4 if four and float(rng.random()) < 0.38 else 2
                  for _ in range(test_count)]
    train_markers = [(2, 2)] * train_count
    if extended:
        index = int(rng.integers(train_count))
        train_markers[index] = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        if train_markers[index] == (2, 2):
            train_markers[index] = (3, 2)
        for k in range(train_count):
            if k != index and float(rng.random()) < 0.45:
                train_markers[k] = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    test_markers = [(int(rng.integers(2, 5)), int(rng.integers(2, 5)))
                    if extended and float(rng.random()) < 0.55 else (2, 2)
                    for _ in range(test_count)]
    def build(axis, bends, marker):
        if axis == "H":
            direction = (0, 1) if int(rng.integers(2)) == 0 else (0, -1)
        else:
            direction = (1, 0) if int(rng.integers(2)) == 0 else (-1, 0)
        hands = tuple("L" if int(rng.integers(2)) == 0 else "R" for _ in range(bends))
        return bends, direction, hands, marker
    train = [build(a, b, m) for a, b, m in zip(train_axes, train_bends, train_markers)]
    test = [build(a, b, m) for a, b, m in zip(test_axes, test_bends, test_markers)]
    return train, test


def _predict_from_training(train_pairs, test_input):
    analyses = [_analyze_pair(p["input"], p["output"]) for p in train_pairs]
    start = analyses[0]["roles"][1]
    target = analyses[0]["roles"][2]
    assert all(info["roles"][1:3] == (start, target) for info in analyses)
    colors = {int(x) for x in np.unique(test_input)}
    other = colors - {start, target}
    assert len(other) == 2
    background = max(other, key=lambda color: int(np.count_nonzero(test_input == color)))
    blocker = next(iter(other - {background}))
    routes = _find_routes(test_input, background, start, target, blocker)
    assert len(routes) == 1 and len(routes[0]["directions"]) in (3, 5)
    result = test_input.copy()
    for p in routes[0]["path"]:
        result[p] = start
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    marker_colors = [int(x) for x in rng.permutation(10)[:2]]
    remaining = [c for c in range(10) if c not in marker_colors]
    shared_palette = float(rng.random()) < 0.35
    base_background, base_blocker = [int(x) for x in rng.choice(remaining, 2, replace=False)]
    train_specs, test_specs = _make_specs(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    palettes = set()
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for index, spec in enumerate(specs):
            for _ in range(120):
                if shared_palette:
                    background, blocker = base_background, base_blocker
                else:
                    background, blocker = [int(x) for x in rng.choice(remaining, 2, replace=False)]
                    if section == "train" and index == 1 and (background, blocker) in palettes:
                        continue
                colors = (background, marker_colors[0], marker_colors[1], blocker)
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = None
                for _ in range(300):
                    pair = _make_pair_attempt(pair_rng, colors, spec)
                    if pair is not None:
                        break
                if pair is None:
                    continue
                key = _grid_key(pair["input"])
                if key in used_inputs:
                    continue
                used_inputs.add(key)
                palettes.add((background, blocker))
                puzzle[section].append(pair)
                break
            else:
                raise ValueError("could not construct distinct, unambiguous pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert pair["input"].shape == pair["output"].shape
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            assert not _is_original_source_input(pair["input"])
            infos[section].append(_analyze_pair(pair["input"], pair["output"]))
    start, target = infos["train"][0]["roles"][1:3]
    assert all(i["roles"][1:3] == (start, target)
               for section in ("train", "test") for i in infos[section])
    train_axes = {i["orientation"] for i in infos["train"]}
    train_bends = {i["bends"] for i in infos["train"]}
    has_long = any(max(i["marker_lengths"]) > 2 for i in infos["train"])
    for info in infos["test"]:
        assert info["orientation"] in train_axes
        assert info["bends"] in train_bends
        assert max(info["marker_lengths"]) == 2 or has_long
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                              pair["output"])


def _puzzles_equal(first, second):
    return all(np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
               for section in ("train", "test")
               for a, b in zip(first[section], second[section])
               for name in ("input", "output")) and all(
                   len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    shapes, lengths, directions, palettes, bends, marker_lengths = set(), set(), set(), set(), set(), set()
    density_bins, orientations, hands = set(), set(), set()
    zero_margin = edge_blocker = large_margin = 0
    tall = wide = full = one_leg = long_leg = dense = sparse = 0
    isolated = clustered = wrong_departure = rich = 0
    mixed_puzzles = consistent_puzzles = varied_backgrounds = varied_blockers = 0
    heldout_patterns = reused_patterns = four_train = four_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_pair(pair["input"], pair["output"])
                      for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair["input"], pair["output"])
                     for pair in puzzle["test"]]
        mixed_puzzles += len({i["orientation"] for i in train_info}) > 1
        consistent_puzzles += len({i["orientation"] for i in train_info}) == 1
        varied_backgrounds += len({i["roles"][0] for i in train_info}) > 1
        varied_blockers += len({i["roles"][3] for i in train_info}) > 1
        seen = {(i["directions"][0], i["hands"]) for i in train_info}
        novel = any((i["directions"][0], i["hands"]) not in seen for i in test_info)
        heldout_patterns += novel
        reused_patterns += not novel
        four_train += any(i["bends"] == 4 for i in train_info)
        four_test += any(i["bends"] == 4 for i in test_info)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                info = _analyze_pair(grid, pair["output"])
                shapes.add(grid.shape)
                lengths.add(info["lengths"])
                directions.add(info["directions"])
                palettes.add(info["roles"])
                bends.add(info["bends"])
                marker_lengths.add(info["marker_lengths"])
                orientations.add(info["orientation"])
                hands.add(info["hands"])
                density = float(np.count_nonzero(grid == info["roles"][3])) / grid.size
                density_bins.add(int(density * 10))
                sparse += density < 0.09
                dense += density >= 0.22
                one_leg += 1 in info["lengths"]
                long_leg += max(info["lengths"]) >= 12
                tall += grid.shape[0] >= 1.4 * grid.shape[1]
                wide += grid.shape[1] >= 1.4 * grid.shape[0]
                full += 30 in grid.shape
                start_bar = _bar(grid, info["roles"][1])[0]
                target_bar = _bar(grid, info["roles"][2])[0]
                structural = set(info["path"]) | set(start_bar) | set(target_bar)
                count = 0
                key_blockers = []
                for direction, length in zip(info["directions"][:-1], info["lengths"][:-1]):
                    count += length
                    key_blockers.append(_add(info["path"][count - 1], direction))
                structural |= set(key_blockers)
                rows = [p[0] for p in structural]
                cols = [p[1] for p in structural]
                margins = (min(rows), grid.shape[0] - 1 - max(rows),
                           min(cols), grid.shape[1] - 1 - max(cols))
                zero_margin += 0 in margins
                large_margin += max(margins) >= 8
                edge_blocker += any(p[0] in (0, grid.shape[0] - 1) or
                                    p[1] in (0, grid.shape[1] - 1)
                                    for p in key_blockers)
                tip = start_bar[-1] if info["directions"][0] in ((0, 1), (1, 0)) else start_bar[0]
                other_tip = start_bar[0] if tip == start_bar[-1] else start_bar[-1]
                reverse = tuple(-x for x in info["directions"][0])
                point = _add(other_tip, reverse)
                free = 0
                while _inside(grid.shape, point) and int(grid[point]) == info["roles"][0]:
                    free += 1
                    point = _add(point, reverse)
                wrong_departure += (free >= 1 and _inside(grid.shape, point)
                                    and int(grid[point]) == info["roles"][3])
                blockers = {tuple(map(int, p)) for p in np.argwhere(grid == info["roles"][3])}
                while blockers:
                    component = {blockers.pop()}
                    frontier = list(component)
                    while frontier:
                        current = frontier.pop()
                        for direction in _DIRS:
                            neighbor = _add(current, direction)
                            if neighbor in blockers:
                                blockers.remove(neighbor)
                                component.add(neighbor)
                                frontier.append(neighbor)
                    isolated += len(component) == 1
                    clustered += len(component) >= 3
                rich += (info["bends"] == 4 and density >= 0.22
                         and max(info["marker_lengths"]) > 2)
    assert len(shapes) >= 50 and len(lengths) >= 80 and len(directions) >= 12
    assert len(palettes) >= 25 and bends == {2, 4}
    assert len(marker_lengths) >= 4 and orientations == {"H", "V"}
    assert len(hands) >= 6 and len(density_bins) >= 3
    assert sparse >= 10 and dense >= 10 and one_leg >= 10 and long_leg >= 10
    assert tall >= 5 and wide >= 5 and full >= 10
    assert zero_margin >= 5 and edge_blocker >= 5 and large_margin >= 5
    assert isolated >= 10 and clustered >= 10 and wrong_departure >= 5
    assert rich >= 1 and four_train >= 5 and four_test >= 2
    assert mixed_puzzles >= 5 and consistent_puzzles >= 5
    assert varied_backgrounds >= 5 and varied_blockers >= 5
    assert heldout_patterns >= 5 and reused_patterns >= 1
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(271828, 7, 3), generate_puzzle(271829, 7, 3))
    for t, q in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + 10 * t + q, t, q)
        assert (len(p["train"]), len(p["test"])) == (t, q)
    for seed, t, q in ((10001, None, 1), (10002, None, 4),
                       (10003, 2, None), (10004, 10, None)):
        p = generate_puzzle(seed, t, q)
        if t is not None:
            assert len(p["train"]) == t
        if q is not None:
            assert len(p["test"]) == q
    for t, q in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, t, q)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts were accepted")
