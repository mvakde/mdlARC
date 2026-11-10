import numpy as np


_HEIGHT = 20
_WIDTH = 20
_TARGET_SIZE = 3
_BLANK = 0
_HIGHLIGHT = 1
_TEXTURE_COLORS = tuple(range(2, 10))


_SOURCE_INPUT_ROWS = (
    (
        "33330303033003330300",
        "00330030330303300330",
        "33333003000303303333",
        "30330000303303330330",
        "00030303330333033300",
        "33003303333003033330",
        "03000033030030003030",
        "30300000033303333333",
        "03300003033033003333",
        "00033003333303030333",
        "30330333003030003303",
        "30030003333003030333",
        "03300033033330030033",
        "00303333000333003030",
        "30333033033333333003",
        "00303300303033033300",
        "33033000000030300033",
        "03030033303330033000",
        "30030330300300333333",
        "30330333000303033303",
    ),
    (
        "50050055555500555500",
        "50555555005555055055",
        "55555555555555550055",
        "05555500050550550005",
        "55555505005550005505",
        "05050500505050505555",
        "55555555500505550005",
        "00055500055505050005",
        "55005005555005050005",
        "00500055055550555055",
        "55505555500500555555",
        "50555550555505055505",
        "50055550055505555555",
        "55055555555055505505",
        "00555505505555055505",
        "50050500055505055050",
        "05050550050050500050",
        "55555055555000505505",
        "55555555555550055055",
        "55500555505505050005",
    ),
    (
        "70777707700770070777",
        "00707070777000077007",
        "00000700777707000070",
        "70707077000770077070",
        "00700700707777007007",
        "77777777077077000707",
        "00070700770707000077",
        "07777070700777000007",
        "00070000777007700077",
        "77077707007077077070",
        "70000000070070000770",
        "77007770777707007777",
        "07077700077077070077",
        "00770777770770777077",
        "00777070770707770777",
        "70777070777770077700",
        "77700077707707070000",
        "77700070770700700000",
        "70000077070007077707",
        "07707707007777007077",
    ),
    (
        "04044044400040444440",
        "00440040440000444440",
        "44400404040444444040",
        "44040040000004440400",
        "40044044444440440404",
        "44004004444440044040",
        "00040000444440404404",
        "40440004400004400000",
        "04440004440040444000",
        "40004400400404440404",
        "00040404440040444044",
        "04400444400044444040",
        "00400400400404400044",
        "40440400444000044400",
        "04444004040044044444",
        "40044040400404040400",
        "44040404400444000044",
        "40000440440404000444",
        "00000444404004000000",
        "44000004400040404044",
    ),
)

_SOURCE_INPUT_SIGNATURES = frozenset(
    "".join(rows) for rows in _SOURCE_INPUT_ROWS
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        train = int(rng.integers(2, 11))
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _window_counts(mask):
    height, width = mask.shape
    counts = np.zeros((height - 2, width - 2), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            counts += mask[dr:height - 2 + dr, dc:width - 2 + dc]
    return counts


def _starts(grid, color):
    return [tuple(map(int, point)) for point in np.argwhere(_window_counts(grid == color) == 9)]


def _footprint(shape, starts):
    mask = np.zeros(shape, dtype=bool)
    for row, col in starts:
        mask[row:row + 3, col:col + 3] = True
    return mask


def _separate(starts):
    return all(max(abs(r - s), abs(c - d)) >= 4
               for i, (r, c) in enumerate(starts)
               for s, d in starts[i + 1:])


def _ring(shape, start):
    row, col = start
    height, width = shape
    sides = []
    if row:
        sides.append([(row - 1, col + x) for x in range(3)])
    if row + 3 < height:
        sides.append([(row + 3, col + x) for x in range(3)])
    if col:
        sides.append([(row + x, col - 1) for x in range(3)])
    if col + 3 < width:
        sides.append([(row + x, col + 3) for x in range(3)])
    return sides


def _choose_starts(rng, shape, count, mode):
    height, width = shape
    all_starts = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    edges = [(r, c) for r, c in all_starts
             if r == 0 or c == 0 or r == height - 3 or c == width - 3]
    starts = []
    if mode == "edge":
        starts.append(edges[int(rng.integers(len(edges)))])
    elif mode == "close" and count >= 2:
        for _ in range(100):
            first = all_starts[int(rng.integers(len(all_starts)))]
            candidates = [(r, c) for r, c in all_starts
                          if max(abs(r - first[0]), abs(c - first[1])) == 4
                          and (abs(r - first[0]) <= 2 or abs(c - first[1]) <= 2)]
            if candidates:
                starts = [first, candidates[int(rng.integers(len(candidates)))]]
                break
    for _ in range(1000):
        if len(starts) == count:
            return starts
        candidate = all_starts[int(rng.integers(len(all_starts)))]
        if all(max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 4 for r, c in starts):
            starts.append(candidate)
    return None


def _stamp_near_miss(rng, grid, fixed, blank, texture, kind):
    height, width = grid.shape
    options = [(r, c) for r in range(height - 2) for c in range(width - 2)
               if not np.any(fixed[max(0, r - 1):min(height, r + 4),
                                   max(0, c - 1):min(width, c + 4)])]
    if not options:
        return False
    row, col = options[int(rng.integers(len(options)))]
    patch = np.zeros((3, 3), dtype=bool)
    if kind == "2x3":
        if int(rng.integers(2)):
            patch[1:, :] = True
        else:
            patch[:2, :] = True
    elif kind == "3x2":
        if int(rng.integers(2)):
            patch[:, 1:] = True
        else:
            patch[:, :2] = True
    else:
        patch[:, :] = True
        patch[int(rng.integers(3)), int(rng.integers(3))] = False
    grid[row:row + 3, col:col + 3] = np.where(patch, blank, texture)
    fixed[row:row + 3, col:col + 3] = True
    return True


def _repair(grid, fixed, blank, texture, intended, rng):
    for _ in range(grid.size * 3):
        bad_blank = [tuple(map(int, point)) for point in
                     np.argwhere(_window_counts(grid == blank) == 9)
                     if tuple(map(int, point)) not in intended]
        bad_texture = np.argwhere(_window_counts(grid == texture) == 9)
        if not len(bad_blank) and not len(bad_texture):
            return True
        if len(bad_blank):
            row, col = bad_blank[0]
            replacement = texture
        else:
            row, col = map(int, bad_texture[0])
            replacement = blank
        candidates = [(r, c) for r in range(row, row + 3)
                      for c in range(col, col + 3) if not fixed[r, c]]
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(len(candidates)))]
        grid[r, c] = replacement
    return False


def _make_pair(rng, blank, highlight, texture, count, regime, style):
    for attempt in range(80):
        if int(rng.integers(7)) == 0:
            shape = (20, 20)
        else:
            shape = (int(rng.integers(12, 31)), int(rng.integers(12, 31)))
        if count >= 4 and min(shape) < 15:
            continue
        mode = style if attempt == 0 else ("close" if int(rng.integers(4)) == 0
                                          else "edge" if int(rng.integers(3)) == 0 else "free")
        starts = _choose_starts(rng, shape, count, mode)
        if starts is None:
            continue
        probability = float(rng.uniform(0.25, 0.63))
        grid = np.where(rng.random(shape) < probability, blank, texture).astype(np.int8)
        fixed = _footprint(shape, starts)
        protected_targets = fixed.copy()
        grid[fixed] = blank
        # Set each immediate perimeter before repairing accidental windows.
        for index, start in enumerate(starts):
            sides = _ring(shape, start)
            ring_cells = [p for side in sides for p in side if not fixed[p]]
            for point in ring_cells:
                grid[point] = texture
                fixed[point] = True
            local_style = ("isolated" if index == 0 and style == "isolated"
                           else "attached" if index == 0 and style == "attached"
                           else "isolated" if int(rng.integers(3)) == 0 else "attached")
            if local_style == "attached" and sides:
                order = rng.permutation(len(sides))
                for side_index in order[:int(rng.integers(1, min(3, len(sides)) + 1))]:
                    side = sides[int(side_index)]
                    available = [p for p in side if not protected_targets[p]]
                    if available:
                        chosen = rng.choice(len(available), size=min(len(available), int(rng.integers(1, 3))), replace=False)
                        for choice in np.atleast_1d(chosen):
                            grid[available[int(choice)]] = blank
        for index, (row, col) in enumerate(starts):
            for other_row, other_col in starts[index + 1:]:
                if abs(row - other_row) == 4 and abs(col - other_col) <= 2:
                    gap_row = min(row, other_row) + 3
                    left, right = min(col, other_col), max(col, other_col) + 3
                    grid[gap_row, left:right] = texture
                    fixed[gap_row, left:right] = True
                if abs(col - other_col) == 4 and abs(row - other_row) <= 2:
                    gap_col = min(col, other_col) + 3
                    top, bottom = min(row, other_row), max(row, other_row) + 3
                    grid[top:bottom, gap_col] = texture
                    fixed[top:bottom, gap_col] = True
        if not _stamp_near_miss(rng, grid, fixed, blank, texture,
                                ("2x3", "3x2", "8/9")[int(rng.integers(3))]):
            continue
        # Repair every non-target monochrome window while protecting intended
        # squares and designed near misses. This also prevents color-role ambiguity.
        if not _repair(grid, fixed, blank, texture, set(starts), rng):
            continue
        if set(_starts(grid, blank)) != set(starts) or _starts(grid, texture):
            continue
        output = grid.copy()
        output[protected_targets] = highlight
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(_validate_seed(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    if int(rng.integers(6)) == 0:
        colors = [0, 1] + list(map(int, rng.permutation(np.arange(2, 10))))
    else:
        colors = list(map(int, rng.permutation(10)))
    blank, highlight = colors[:2]
    textures = colors[2:]
    train_targets = [1, int(rng.integers(2, 5))]
    train_targets.extend(int(rng.integers(1, 6)) for _ in range(train_count - 2))
    test_targets = [int(rng.integers(1, 6)) for _ in range(test_count)]
    train_textures = [textures[0], textures[1]]
    train_textures.extend(textures[int(rng.integers(len(textures)))] for _ in range(train_count - 2))
    test_textures = [textures[int(rng.integers(len(textures)))] for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, regime, counts, palette in (
        ("train", "source_train", train_targets, train_textures),
        ("test", "target_test", test_targets, test_textures),
    ):
        for index, (target_count, texture) in enumerate(zip(counts, palette)):
            for attempt in range(40):
                if section == "train" and index == 0:
                    style = "isolated" if int(rng.integers(2)) else "attached"
                elif section == "train" and index == 1:
                    style = "close" if int(rng.integers(2)) else "edge"
                else:
                    style = ("close", "edge", "free", "isolated", "attached")[int(rng.integers(5))]
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, blank, highlight, texture, target_count, regime, style)
                key = _grid_key(pair["input"])
                if key in seen or (pair["input"].shape == (20, 20)
                                   and "".join(map(str, pair["input"].flat)) in _SOURCE_INPUT_SIGNATURES):
                    continue
                seen.add(key)
                puzzle[section].append(pair)
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_roles(pair):
    inp, out = pair["input"], pair["output"]
    changed = inp != out
    before = set(map(int, np.unique(inp[changed])))
    after = set(map(int, np.unique(out[changed])))
    assert len(before) == len(after) == 1
    return before.pop(), after.pop()


def _validate_pair(pair, blank, highlight):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 12 <= grid.shape[0] <= 30 and 12 <= grid.shape[1] <= 30
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    colors = set(map(int, np.unique(inp)))
    assert len(colors) == 2 and blank in colors and highlight not in colors
    texture = (colors - {blank}).pop()
    starts = _starts(inp, blank)
    assert starts and _separate(starts) and not _starts(inp, texture)
    mask = _footprint(inp.shape, starts)
    assert int(mask.sum()) == 9 * len(starts)
    assert np.array_equal(inp != out, mask)
    assert np.all(out[mask] == highlight)
    assert np.all(inp[mask] == blank)
    assert np.array_equal(inp[~mask], out[~mask])
    assert set(map(int, np.unique(out))) == {blank, highlight, texture}
    assert not (inp.shape == (20, 20) and "".join(map(str, inp.flat)) in _SOURCE_INPUT_SIGNATURES)
    return len(starts), texture


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    all_pairs = puzzle["train"] + puzzle["test"]
    blank, highlight = _infer_roles(puzzle["train"][0])
    assert blank != highlight
    seen_inputs, seen_pairs = set(), set()
    train_target_counts = []
    train_textures = []
    for index, pair in enumerate(all_pairs):
        assert _infer_roles(pair) == (blank, highlight)
        targets, texture = _validate_pair(pair, blank, highlight)
        key = _grid_key(pair["input"])
        pair_key = (key, _grid_key(pair["output"]))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        if index < train_count:
            train_target_counts.append(targets)
            train_textures.append(texture)
    assert 1 in train_target_counts and max(train_target_counts) >= 2
    assert len(set(train_textures)) >= 2


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[split], second[split]) for role in ("input", "output"))
               for split in ("train", "test"))


def _expect_value_error(thunk):
    try:
        thunk()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, target_counts, palettes, layouts, fractions = set(), set(), set(), set(), set()
    isolated = attached = train_edge = test_upper = test_right = close = 0
    near_8 = near_2x3 = near_3x2 = rectangular = small = large = classic = combined = 0
    fewer = same = more = repeated = held_out = varying_shapes = varying_counts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _infer_roles(puzzle["train"][0])
        palettes.add(roles)
        puzzle_shapes, puzzle_counts = set(), set()
        train_target_counts, test_target_counts = [], []
        train_textures, test_textures = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                height, width = inp.shape
                blank = roles[0]
                starts = [(row, col) for row in range(height - 2)
                          for col in range(width - 2)
                          if np.all(inp[row:row + 3, col:col + 3] == blank)]
                mask = _footprint(inp.shape, starts)
                assert np.array_equal(inp != out, mask)
                assert np.all(out[mask] == roles[1])
                assert np.array_equal(inp[~mask], out[~mask])
                texture = (set(map(int, np.unique(inp))) - {blank}).pop()
                if section == "train":
                    train_target_counts.append(len(starts))
                    train_textures.add(texture)
                else:
                    test_target_counts.append(len(starts))
                    test_textures.add(texture)
                target_counts.add(len(starts))
                shapes.add(inp.shape)
                puzzle_shapes.add(inp.shape)
                puzzle_counts.add(len(starts))
                layouts.add((inp.shape, tuple(starts)))
                fraction = round(float(np.mean(inp == blank)), 2)
                fractions.add(fraction)
                rectangular += height != width
                small += min(inp.shape) < 20
                large += max(inp.shape) > 20
                classic += inp.shape == (20, 20) and roles == (0, 1)
                for start in starts:
                    neighbors = [p for side in _ring(inp.shape, start) for p in side]
                    if neighbors:
                        touched = any(inp[p] == blank for p in neighbors)
                        isolated += not touched
                        attached += touched
                    if section == "train":
                        train_edge += start[0] in (0, height - 3) or start[1] in (0, width - 3)
                    else:
                        test_upper += start[0] < (height - 3) / 3
                        test_right += start[1] > 2 * (width - 3) / 3
                has_close = any(max(abs(r - s), abs(c - d)) == 4 and
                                (abs(r - s) <= 2 or abs(c - d) <= 2)
                                for i, (r, c) in enumerate(starts) for s, d in starts[i + 1:])
                close += has_close
                combined += has_close and height != width and fraction > .55
                for row in range(height - 2):
                    for col in range(width - 2):
                        if np.any(mask[row:row + 3, col:col + 3]):
                            continue
                        patch = inp[row:row + 3, col:col + 3] == blank
                        near_8 += int(np.count_nonzero(patch) == 8)
                        near_2x3 += int(np.all(patch[:2]) or np.all(patch[1:]))
                        near_3x2 += int(np.all(patch[:, :2]) or np.all(patch[:, 1:]))
        varying_shapes += len(puzzle_shapes) > 1
        varying_counts += len(puzzle_counts) > 1
        fewer += any(n < max(train_target_counts) for n in test_target_counts)
        same += any(n in train_target_counts for n in test_target_counts)
        more += any(n > max(train_target_counts) for n in test_target_counts)
        repeated += bool(test_textures & train_textures)
        held_out += bool(test_textures - train_textures)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 50 and len(layouts) >= 100 and len(target_counts) >= 4
    assert len(palettes) >= 15 and len(fractions) >= 15
    assert varying_shapes >= 35 and varying_counts >= 35
    assert fewer > 0 and same > 0 and more > 0 and repeated > 0 and held_out > 0
    assert isolated > 0 and attached > 0 and train_edge > 0 and test_upper > 0 and test_right > 0
    assert close > 0 and near_8 > 0 and near_2x3 > 0 and near_3x2 > 0
    assert rectangular > 0 and small > 0 and large > 0 and classic > 0 and combined > 0
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                      (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + i, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3))
    assert not _puzzles_equal(generate_puzzle(8001, 6, 2), generate_puzzle(8002, 6, 2))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        _expect_value_error(lambda train=train, test=test: generate_puzzle(1, train, test))
