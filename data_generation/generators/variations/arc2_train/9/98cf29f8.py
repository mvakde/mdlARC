import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "000000000000000000",
        "000000000000000000",
        "000000000000000000",
        "000000022222220000",
        "000000022222220000",
        "044400022222220000",
        "044444422222220000",
        "044400022222220000",
        "000000022222220000",
        "000000022222220000",
        "000000000000000000",
        "000000000000000000",
        "000000000000000000",
    ),
    (
        "00000000000000",
        "04444444000000",
        "04444444000000",
        "04444444000000",
        "00050000000000",
        "00050000000000",
        "00050000000000",
        "00050000000000",
        "00555500000000",
        "00555500000000",
        "00555500000000",
        "00000000000000",
        "00000000000000",
        "00000000000000",
        "00000000000000",
        "00000000000000",
        "00000000000000",
    ),
    (
        "0000000000000000000",
        "0000002222200000000",
        "0000002222200000000",
        "0000000300000000000",
        "0000000300000000000",
        "0000000300000000000",
        "0000000300000000000",
        "0003333333300000000",
        "0003333333300000000",
        "0003333333300000000",
        "0003333333300000000",
        "0003333333300000000",
        "0003333333300000000",
        "0000000000000000000",
        "0000000000000000000",
    ),
    (
        "00000000000000000",
        "00000000000000000",
        "00000000000000000",
        "03333300000000000",
        "03333377777777700",
        "03333300000777700",
        "03333300000777700",
        "03333300000000000",
        "03333300000000000",
        "03333300000000000",
        "03333300000000000",
        "00000000000000000",
    ),
)


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    data = np.ascontiguousarray(grid, dtype=np.int8)
    return tuple(data.shape), data.tobytes()


def _rows_key(rows: tuple[str, ...]) -> tuple:
    height = len(rows)
    width = len(rows[0])
    values = bytes(int(ch) for row in rows for ch in row)
    return (height, width), values


# Rejecting an exact source input is sufficient to reject an exact source pair.
_SOURCE_INPUT_KEYS = frozenset(_rows_key(rows) for rows in _SOURCE_INPUT_ROWS)


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _randint(
    rng: np.random.Generator,
    low: int,
    high_inclusive: int,
) -> int:
    return int(rng.integers(low, high_inclusive + 1))


def _normalize_seed(seed: int) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _normalize_optional_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _normalize_optional_count(
        num_train, "num_train", 2, 10
    )
    num_test = _normalize_optional_count(
        num_test, "num_test", 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + ((seed + seed // 4) % span)
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(x) for x in rng.permutation(10))
    return {
        "colors": colors,
        "stable_background": bool(rng.integers(0, 2)),
        "stable_objects": bool(rng.integers(0, 3) == 0),
        "multi": bool(rng.integers(0, 3) == 0),
        "direction_mode": _randint(rng, 0, 3),
    }


def _solid_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = np.argwhere(mask)
    if len(points) == 0:
        return None
    r0, c0 = (int(v) for v in points.min(axis=0))
    r1, c1 = (int(v) + 1 for v in points.max(axis=0))
    if int(mask.sum()) != (r1 - r0) * (c1 - c0):
        return None
    return r0, r1, c0, c1


def _border_is(grid: np.ndarray, color: int) -> bool:
    return bool(np.all(grid[0] == color) and np.all(grid[-1] == color)
                and np.all(grid[:, 0] == color) and np.all(grid[:, -1] == color))


def _sample_padding(rng: np.random.Generator, extent: int,
                    compact: bool) -> tuple[int, int]:
    maximum = min(1 if compact else 8, 28 - extent)
    if maximum < 0:
        raise ValueError("geometry exceeds ARC canvas")
    extra = _randint(rng, 0, maximum)
    before = _randint(rng, 0, extra)
    return 1 + before, 1 + extra - before


def _make_pair(rng: np.random.Generator, direction: str,
               background: int, stationary_color: int, movable_color: int,
               compact: bool, clear_example: bool) -> tuple[np.ndarray, np.ndarray]:
    high = 4 if compact else 9
    sh, sw, ph, pw = (_randint(rng, 2, high) for _ in range(4))
    bridge = _randint(rng, 2 if clear_example else 1,
                     3 if compact else 8)
    if direction in ("right", "left"):
        pr = 0
        sr = _randint(rng, 1 - sh, ph - 1)
        if direction == "right":
            pc, sc = 0, pw + bridge
        else:
            sc, pc = 0, sw + bridge
        attachment = _randint(rng, max(pr, sr), min(pr + ph, sr + sh) - 1)
    elif direction in ("up", "down"):
        sc = 0
        pc = _randint(rng, 1 - pw, sw - 1)
        if direction == "up":
            sr, pr = 0, sh + bridge
        else:
            pr, sr = 0, ph + bridge
        attachment = _randint(rng, max(pc, sc), min(pc + pw, sc + sw) - 1)
    else:
        raise ValueError("invalid direction")

    top = min(pr, sr)
    bottom = max(pr + ph, sr + sh)
    left = min(pc, sc)
    right = max(pc + pw, sc + sw)
    top_pad, bottom_pad = _sample_padding(rng, bottom - top, compact)
    left_pad, right_pad = _sample_padding(rng, right - left, compact)
    height = bottom - top + top_pad + bottom_pad
    width = right - left + left_pad + right_pad
    dr, dc = top_pad - top, left_pad - left
    pr, sr, pc, sc = pr + dr, sr + dr, pc + dc, sc + dc
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[sr:sr + sh, sc:sc + sw] = stationary_color
    out[sr:sr + sh, sc:sc + sw] = stationary_color
    inp[pr:pr + ph, pc:pc + pw] = movable_color
    if direction == "right":
        inp[attachment + dr, pc + pw:sc] = movable_color
        nr, nc = pr, sc - pw
    elif direction == "left":
        inp[attachment + dr, sc + sw:pc] = movable_color
        nr, nc = pr, sc + sw
    elif direction == "up":
        inp[sr + sh:pr, attachment + dc] = movable_color
        nr, nc = sr + sh, pc
    else:
        inp[pr + ph:sr, attachment + dc] = movable_color
        nr, nc = sr - ph, pc
    out[nr:nr + ph, nc:nc + pw] = movable_color
    return inp, out


def _trim_pair(pair: tuple[np.ndarray, np.ndarray], background: int
               ) -> tuple[np.ndarray, np.ndarray]:
    inp, out = pair
    points = np.argwhere((inp != background) | (out != background))
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    return (inp[r0 - 1:r1 + 1, c0 - 1:c1 + 1].copy(),
            out[r0 - 1:r1 + 1, c0 - 1:c1 + 1].copy())


def _combine(pieces: list[tuple[np.ndarray, np.ndarray]],
             background: int, rng: np.random.Generator
             ) -> tuple[np.ndarray, np.ndarray]:
    if len(pieces) == 1:
        return pieces[0]
    first, second = (_trim_pair(piece, background) for piece in pieces)
    vertical = bool(rng.integers(0, 2))
    gap = _randint(rng, 1, 3)
    if vertical:
        height = first[0].shape[0] + second[0].shape[0] + gap
        width = max(first[0].shape[1], second[0].shape[1])
        if height > 30:
            raise ValueError("multi geometry too tall")
        offsets = [(0, _randint(rng, 0, width - first[0].shape[1])),
                   (first[0].shape[0] + gap,
                    _randint(rng, 0, width - second[0].shape[1]))]
    else:
        height = max(first[0].shape[0], second[0].shape[0])
        width = first[0].shape[1] + second[0].shape[1] + gap
        if width > 30:
            raise ValueError("multi geometry too wide")
        offsets = [(_randint(rng, 0, height - first[0].shape[0]), 0),
                   (_randint(rng, 0, height - second[0].shape[0]),
                    first[0].shape[1] + gap)]
    output = []
    for layer in (0, 1):
        grid = np.full((height, width), background, dtype=np.int8)
        for piece, (r, c) in zip((first, second), offsets):
            source = piece[layer]
            h, w = source.shape
            grid[r:r + h, c:c + w] = source
        output.append(grid)
    return output[0], output[1]


def _candidate(input_grid: np.ndarray, stationary: int, movable: int,
               direction: str, cross: int, bridge: int,
               stationary_box: tuple[int, int, int, int]) -> dict | None:
    sr0, sr1, sc0, sc1 = stationary_box
    mask = input_grid == movable
    remaining = mask.copy()
    if direction == "right":
        edge = sc0
        if edge - bridge < 0 or not np.all(mask[cross, edge - bridge:edge]):
            return None
        remaining[cross, edge - bridge:edge] = False
    elif direction == "left":
        edge = sc1
        if edge + bridge > input_grid.shape[1] or not np.all(mask[cross, edge:edge + bridge]):
            return None
        remaining[cross, edge:edge + bridge] = False
    elif direction == "up":
        edge = sr1
        if edge + bridge > input_grid.shape[0] or not np.all(mask[edge:edge + bridge, cross]):
            return None
        remaining[edge:edge + bridge, cross] = False
    else:
        edge = sr0
        if edge - bridge < 0 or not np.all(mask[edge - bridge:edge, cross]):
            return None
        remaining[edge - bridge:edge, cross] = False
    payload = _solid_bbox(remaining)
    if payload is None:
        return None
    pr0, pr1, pc0, pc1 = payload
    ph, pw = pr1 - pr0, pc1 - pc0
    if ph < 2 or pw < 2:
        return None
    if direction == "right":
        if pc1 != sc0 - bridge or not pr0 <= cross < pr1:
            return None
        target = (pr0, pr1, sc0 - pw, sc0)
    elif direction == "left":
        if pc0 != sc1 + bridge or not pr0 <= cross < pr1:
            return None
        target = (pr0, pr1, sc1, sc1 + pw)
    elif direction == "up":
        if pr0 != sr1 + bridge or not pc0 <= cross < pc1:
            return None
        target = (sr1, sr1 + ph, pc0, pc1)
    else:
        if pr1 != sr0 - bridge or not pc0 <= cross < pc1:
            return None
        target = (sr0 - ph, sr0, pc0, pc1)
    r0, r1, c0, c1 = target
    h, w = input_grid.shape
    if not (1 <= r0 < r1 < h and 1 <= c0 < c1 < w):
        return None
    destination = input_grid[r0:r1, c0:c1]
    if not np.all((destination == int(input_grid[0, 0])) |
                  (destination == movable)):
        return None
    return {"direction": direction, "stationary_color": stationary,
            "movable_color": movable, "stationary_box": stationary_box,
            "payload_box": payload, "output_box": target,
            "stationary_shape": (sr1 - sr0, sc1 - sc0),
            "payload_shape": (ph, pw), "bridge_length": bridge,
            "background": int(input_grid[0, 0]), "grid_shape": (h, w),
            "overlap": (min(pr1, sr1) - max(pr0, sr0)) if direction in ("left", "right")
                       else (min(pc1, sc1) - max(pc0, sc0))}


def _input_solutions(input_grid: np.ndarray) -> list[tuple[list[dict], np.ndarray]]:
    background = int(input_grid[0, 0])
    if not _border_is(input_grid, background):
        return []
    colors = [int(c) for c in np.unique(input_grid) if int(c) != background]
    if len(colors) not in (2, 4):
        return []
    candidates = []
    for stationary in colors:
        box = _solid_bbox(input_grid == stationary)
        if box is None or min(box[1] - box[0], box[3] - box[2]) < 2:
            continue
        for movable in colors:
            if movable == stationary:
                continue
            for direction in ("right", "left", "up", "down"):
                crosses = (range(box[0], box[1]) if direction in ("right", "left")
                           else range(box[2], box[3]))
                for cross in crosses:
                    for bridge in range(1, 9):
                        candidate = _candidate(input_grid, stationary, movable,
                                               direction, cross, bridge, box)
                        if candidate is not None:
                            key = (stationary, movable, direction,
                                   candidate["payload_box"], candidate["output_box"])
                            if not any(existing[0] == key for existing in candidates):
                                candidates.append((key, candidate))
    candidates = [candidate for _, candidate in candidates]
    solutions = []
    for first in candidates:
        groups = [[first]] if len(colors) == 2 else [
            [first, second] for second in candidates
            if first["stationary_color"] < second["stationary_color"]
            and len({first["stationary_color"], first["movable_color"],
                     second["stationary_color"], second["movable_color"]}) == 4]
        for group in groups:
            output = input_grid.copy()
            for item in group:
                output[input_grid == item["movable_color"]] = background
            occupied = output != background
            valid = True
            for item in group:
                r0, r1, c0, c1 = item["output_box"]
                if np.any(occupied[r0:r1, c0:c1]):
                    valid = False
                    break
                occupied[r0:r1, c0:c1] = True
                output[r0:r1, c0:c1] = item["movable_color"]
            if valid and not np.array_equal(output, input_grid):
                solutions.append((group, output))
    return solutions


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> list[dict]:
    _check(input_grid.shape == output_grid.shape, "dimensions changed")
    _check(_border_is(output_grid, int(input_grid[0, 0])), "output border changed")
    _check(set(np.unique(input_grid)) == set(np.unique(output_grid)),
           "palette changed")
    solutions = _input_solutions(input_grid)
    _check(len(solutions) == 1, "input has no unique bridge interpretation")
    info, predicted = solutions[0]
    _check(np.array_equal(predicted, output_grid), "output violates bridge rule")
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "split schema")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    keys, pairs = set(), set()
    train_infos = []
    test_infos = []
    for split, destination in (("train", train_infos), ("test", test_infos)):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "invalid grid")
                h, w = grid.shape
                _check(1 <= h <= 30 and 1 <= w <= 30 and
                       np.issubdtype(grid.dtype, np.integer) and
                       bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC cells")
            key = _grid_key(pair["input"])
            full = _pair_key(pair)
            _check(key not in keys and full not in pairs, "duplicate pair")
            _check(key not in _SOURCE_INPUT_KEYS, "source copy")
            keys.add(key)
            pairs.add(full)
            destination.append(_analyze_pair(pair["input"], pair["output"]))
    # A two-move test requires a demonstrated two-move training example.
    if any(len(items) == 2 for items in test_infos):
        _check(any(len(items) == 2 for items in train_infos),
               "test composition absent from training")
    _check(any(len(items) == 1 for items in train_infos),
           "training needs a clear single-move demonstration")


def _choose_colors(rng: np.random.Generator, parameters: dict,
                   pair_index: int, moves: int) -> tuple[int, list[tuple[int, int]]]:
    palette = parameters["colors"]
    background = palette[0] if parameters["stable_background"] else _randint(rng, 0, 9)
    if parameters["stable_objects"]:
        choices = [color for color in palette[1:5] if color != background]
        if len(choices) < 2 * moves:
            choices += [color for color in palette[5:] if color != background]
        if moves == 1:
            selected = choices[:2]
            if pair_index % 2:
                selected.reverse()
        else:
            selected = choices[:4]
    else:
        available = [color for color in range(10) if color != background]
        selected = [int(color) for color in rng.choice(available, 2 * moves,
                                                     replace=False)]
    return background, [(selected[2 * i], selected[2 * i + 1])
                        for i in range(moves)]


def _generate_valid_pair(rng: np.random.Generator, directions: list[str],
                         parameters: dict, pair_index: int,
                         clear_example: bool) -> dict:
    moves = len(directions)
    for _ in range(120):
        background, roles = _choose_colors(rng, parameters, pair_index, moves)
        pieces = [_make_pair(rng, direction, background, stationary, movable,
                             moves == 2, clear_example and i == 0)
                  for i, (direction, (stationary, movable))
                  in enumerate(zip(directions, roles))]
        try:
            inp, out = _combine(pieces, background, rng)
        except ValueError:
            continue
        if _grid_key(inp) in _SOURCE_INPUT_KEYS:
            continue
        solutions = _input_solutions(inp)
        if len(solutions) == 1 and np.array_equal(solutions[0][1], out):
            return {"input": inp, "output": out}
    raise ValueError("could not construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    directions = ["up", "right", "down", "left"]
    rng.shuffle(directions)
    mode = parameters["direction_mode"]
    cross_axis = [next(d for d in directions if d in ("left", "right")),
                  next(d for d in directions if d in ("up", "down"))]
    remaining = [d for d in directions if d not in cross_axis]
    if mode == 0:
        train_set = directions[:1]
        test_set = train_set
    elif mode == 1:
        train_set = directions[:min(2, num_train)]
        test_set = train_set
    elif mode == 2:
        train_set = cross_axis + remaining[:min(3, num_train) - 2]
        test_set = [next(d for d in directions if d not in train_set)]
    else:
        train_set = cross_axis + remaining[:_randint(rng, 2, min(4, num_train)) - 2]
        test_set = directions
    train_directions = list(train_set)
    while len(train_directions) < num_train:
        train_directions.append(train_set[_randint(rng, 0, len(train_set) - 1)])
    rng.shuffle(train_directions)
    test_directions = [test_set[_randint(rng, 0, len(test_set) - 1)]
                       for _ in range(num_test)]
    if mode == 2:
        test_directions[0] = test_set[0]
    multi = parameters["multi"]
    train_multi_index = 1 if multi else -1
    seen = set()
    result = {"train": [], "test": []}
    for split, listed in (("train", train_directions),
                          ("test", test_directions)):
        for index, direction in enumerate(listed):
            moves = 2 if ((split == "train" and index == train_multi_index)
                          or (split == "test" and multi and
                              (index == 0 or bool(rng.integers(0, 3) == 0)))) else 1
            allowed = train_set if split == "train" else test_set
            move_directions = [direction] + [allowed[_randint(rng, 0, len(allowed) - 1)]
                                                for _ in range(moves - 1)]
            for attempt in range(80):
                pair = _generate_valid_pair(rng, move_directions, parameters,
                                            index + (0 if split == "train" else 13),
                                            split == "train" and index == 0)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise ValueError("could not construct distinct pairs")
    validate_generated_puzzle(result)
    return result


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name])
                   for a, b in zip(first[split], second[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, geometries, directions, bridge_lengths = set(), set(), set(), set()
    stationary_shapes, payload_shapes, backgrounds = set(), set(), set()
    single_direction, heldout, multi_train, multi_test = 0, 0, 0, 0
    test_square_stationary = test_square_payload = test_thin = 0
    test_wide_stationary = test_partial = test_large_payload = 0
    varied_background = varied_roles = varied_bridges = varied_shapes = 0
    vertical_one_cell = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_pair_key(pair) for split in ("train", "test")
                             for pair in puzzle[split]))
        train_infos = [_analyze_pair(pair["input"], pair["output"])
                       for pair in puzzle["train"]]
        test_infos = [_analyze_pair(pair["input"], pair["output"])
                      for pair in puzzle["test"]]
        train_dirs = {item["direction"] for infos in train_infos for item in infos}
        test_dirs = {item["direction"] for infos in test_infos for item in infos}
        single_direction += len(train_dirs | test_dirs) == 1
        heldout += bool(test_dirs - train_dirs)
        multi_train += any(len(infos) == 2 for infos in train_infos)
        multi_test += any(len(infos) == 2 for infos in test_infos)
        flattened = [item for infos in train_infos + test_infos for item in infos]
        varied_background += len({item["background"] for item in flattened}) > 1
        varied_roles += len({(item["stationary_color"], item["movable_color"])
                             for item in flattened}) > 1
        varied_bridges += len({item["bridge_length"] for item in flattened}) > 1
        varied_shapes += len({(item["stationary_shape"], item["payload_shape"])
                              for item in flattened}) > 1
        for split, infos_list in (("train", train_infos), ("test", test_infos)):
            for infos in infos_list:
                for item in infos:
                    directions.add(item["direction"])
                    bridge_lengths.add(item["bridge_length"])
                    backgrounds.add(item["background"])
                    vertical_one_cell += (item["bridge_length"] == 1 and
                                          item["direction"] in {"up", "down"})
                    shapes.add(item["grid_shape"])
                    stationary_shapes.add(item["stationary_shape"])
                    payload_shapes.add(item["payload_shape"])
                    geometries.add((item["direction"], item["grid_shape"],
                                    item["stationary_box"], item["payload_box"],
                                    item["bridge_length"]))
                    if split == "test":
                        sh, sw = item["stationary_shape"]
                        ph, pw = item["payload_shape"]
                        test_square_stationary += sh == sw
                        test_square_payload += ph == pw
                        test_thin += min(sh, sw, ph, pw) == 2
                        test_wide_stationary += sw > sh
                        test_partial += item["overlap"] < min(
                            ph if item["direction"] in {"left", "right"} else pw,
                            sh if item["direction"] in {"left", "right"} else sw)
                        test_large_payload += ph * pw > sh * sw
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "automatic counts lack variety")
    _check(len(signatures) == 42 and len(shapes) >= 30 and
           len(geometries) >= 100, "layouts lack variety")
    _check(directions == {"up", "right", "down", "left"} and
           {1, 2, 3, 4, 5}.issubset(bridge_lengths), "directions/gaps lack variety")
    _check(len(stationary_shapes) >= 25 and len(payload_shapes) >= 25 and
           len(backgrounds) >= 7, "shapes/colors lack variety")
    _check(single_direction >= 2 and heldout >= 3 and
           multi_train >= 3 and multi_test >= 3, "regime diversity lacking")
    _check(test_square_stationary >= 2 and test_square_payload >= 2 and
           test_thin >= 5, "test silhouettes lack variety")
    _check(test_wide_stationary >= 5 and test_partial >= 10 and
           test_large_payload >= 5 and vertical_one_cell >= 2,
           "overlap, size, or vertical short bridges lack variety")
    _check(varied_background >= 5 and varied_roles >= 20 and
           varied_bridges >= 35 and varied_shapes == 42,
           "within-puzzle variation lacks breadth")
    _check(_puzzles_equal(generate_puzzle(17), generate_puzzle(17)),
           "generation is not deterministic")
    _check(_puzzles_equal(generate_puzzle(12345, 7, 3),
                         generate_puzzle(12345, 7, 3)),
           "explicit generation is not deterministic")
    for seed, nt, ns in ((3000, 2, 1), (3001, 10, 1), (3002, 10, 4),
                         (4000, None, 1), (4001, None, 4),
                         (4002, 2, None), (4003, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        if nt is not None:
            _check(len(puzzle["train"]) == nt, "train count not honored")
        if ns is not None:
            _check(len(puzzle["test"]) == ns, "test count not honored")
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(5000, **kwargs)
        except ValueError:
            continue
        raise AssertionError("illegal counts were accepted")
