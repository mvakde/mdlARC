"""Generate variations of the detached-tile, holed-board ARC task."""

import hashlib

import numpy as np


# SHA-256 of the four source input arrays, in row-major int8 order.  A
# comparison of full input contents is enough to rule out an exact pair copy.
_SOURCE_INPUTS = {
    ((20, 10), "7ef818e687bb62fbe069f1010066c55d6e1c61b6b85c5189e0eee318e9719607"),
    ((24, 19), "e57d45523bcc72d936dad6632afa69e675e8bd5f6dda812cdf0bb44aa5cdba1c"),
    ((16, 12), "d45f8563f91a9968fa52c849ec86334e69d472f96b13a5df624c7ee4ebcfd738"),
    ((19, 15), "49c370e6f1d57c5bbc9a8f75050a7b3f8c58dd8e50ebe90cbf94d14ee64f41e7"),
}


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or (value is not None and not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    return None if value is None else int(value)


def _resolve_counts(rng, num_train, num_test):
    num_train, num_test = _count(num_train, "num_train"), _count(num_test, "num_test")
    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be 2..10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be 1..4")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _transform(array, operation):
    if operation < 4:
        return np.rot90(array, operation).copy()
    return np.rot90(np.fliplr(array), operation - 4).copy()


def _variants(array):
    seen = set()
    for operation in range(8):
        transformed = _transform(array, operation)
        key = (transformed.shape, transformed.tobytes())
        if key not in seen:
            seen.add(key)
            yield operation, transformed


def _mask_orbit(mask):
    return min(transformed.tobytes() for _, transformed in _variants(mask.astype(np.uint8)))


def _connected_components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            stack, cells = [(r, c)], []
            visited[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
            components.append(cells)
    return components


def _analyze_pair(input_grid, output_grid=None):
    """Reconstruct the board and all distinct completed outputs from pixels."""
    h, w = input_grid.shape
    border = np.concatenate((input_grid[0], input_grid[-1], input_grid[1:-1, 0], input_grid[1:-1, -1]))
    assert border.size and np.all(border == border[0])
    background = int(border[0])
    components = _connected_components(input_grid != background)
    assert components
    main = max(components, key=len)
    assert len(main) > 9 and sum(len(x) == len(main) for x in components) == 1
    top = min(r for r, _ in main)
    bottom = max(r for r, _ in main) + 1
    left = min(c for _, c in main)
    right = max(c for _, c in main) + 1
    board = input_grid[top:bottom, left:right]
    bh, bw = board.shape
    assert bh >= 5 and bw >= 5
    assert 0 < top and 0 < left and bottom < h and right < w
    colors = set(int(v) for v in np.unique(board)) - {background}
    assert len(colors) == 1
    carrier = colors.pop()
    assert np.all(board[0] == carrier) and np.all(board[-1] == carrier)
    assert np.all(board[:, 0] == carrier) and np.all(board[:, -1] == carrier)
    assert set(int(v) for v in np.unique(board)) == {background, carrier}

    outside = input_grid != background
    outside[top:bottom, left:right] = False
    tile_components = _connected_components(outside)
    assert tile_components
    tiles = []
    tile_origins = []
    for component in tile_components:
        assert len(component) == 9
        tr = min(r for r, _ in component)
        tc = min(c for _, c in component)
        tile = input_grid[tr:tr + 3, tc:tc + 3]
        assert tile.shape == (3, 3) and np.all(tile != background)
        assert len(set(int(v) for v in np.unique(tile))) == 2
        assert carrier in tile and 1 <= int(np.count_nonzero(tile == carrier)) <= 8
        tiles.append(tile.copy())
        tile_origins.append((tr, tc))
    assert 1 <= len(tiles) <= 7

    hole_positions = np.argwhere(board == background)
    assert len(hole_positions) > 0
    all_holes = sum(1 << (int(r) * bw + int(c)) for r, c in hole_positions)
    positions_by_mask = {}
    weights = 1 << np.arange(9, dtype=np.int64)
    for r in range(bh - 2):
        for c in range(bw - 2):
            mask = int(np.dot((board[r:r + 3, c:c + 3] == background).ravel(), weights))
            positions_by_mask.setdefault(mask, []).append((r, c))

    candidate_lists = []
    for tile in tiles:
        candidates = []
        for operation, transformed in _variants(tile):
            carrier_mask = transformed == carrier
            mask_code = int(np.dot(carrier_mask.ravel(), weights))
            for r, c in positions_by_mask.get(mask_code, ()):
                cells = sum(1 << ((r + dy) * bw + c + dx) for dy in range(3) for dx in range(3))
                holes = sum(1 << ((r + int(dy)) * bw + c + int(dx)) for dy, dx in np.argwhere(carrier_mask))
                candidates.append((r, c, transformed, cells, holes, operation))
        assert candidates
        candidate_lists.append(candidates)

    order = sorted(range(len(tiles)), key=lambda i: len(candidate_lists[i]))
    outputs = {}
    assignment = {}

    def search(index, occupied, covered):
        if len(outputs) >= 2:
            return
        if index == len(order):
            if covered != all_holes:
                return
            result = np.full(board.shape, carrier, dtype=np.int8)
            for candidate in assignment.values():
                r, c, transformed = candidate[:3]
                result[r:r + 3, c:c + 3] = transformed
            outputs.setdefault(result.tobytes(), (result, dict(assignment)))
            return
        tile_index = order[index]
        for candidate in candidate_lists[tile_index]:
            if occupied & candidate[3]:
                continue
            assignment[tile_index] = candidate
            search(index + 1, occupied | candidate[3], covered | candidate[4])
            del assignment[tile_index]
            if len(outputs) >= 2:
                return

    search(0, 0, 0)
    assert len(outputs) == 1
    expected, solution = next(iter(outputs.values()))
    if output_grid is not None:
        assert output_grid.shape == board.shape and np.array_equal(output_grid, expected)
    windows = tuple((solution[i][0], solution[i][1]) for i in range(len(tiles)))
    visible = tuple(sum(1 for _ in _variants(tile)) for tile in tiles)
    edge = any(r == 0 or c == 0 or r == bh - 3 or c == bw - 3 for r, c in windows)
    accent_border = bool(np.any(expected[0] != carrier) or np.any(expected[-1] != carrier) or
                         np.any(expected[:, 0] != carrier) or np.any(expected[:, -1] != carrier))
    island = any(top <= min(r for r, _ in component) and max(r for r, _ in component) < bottom and
                 left <= min(c for _, c in component) and max(c for _, c in component) < right and
                 component is not main for component in components)
    rotations = []
    for i, tile in enumerate(tiles):
        transformed = solution[i][2]
        rotations.append(any(np.array_equal(_transform(tile, op), transformed) for op in range(1, 4)) and
                         not np.array_equal(tile, transformed))
    # A reflection is visually necessary only if rotations cannot make it.
    reflections = [not any(np.array_equal(_transform(tile, op), solution[i][2]) for op in range(4))
                   for i, tile in enumerate(tiles)]
    return {
        "background": background, "carrier": carrier, "board_shape": board.shape,
        "board_origin": (top, left), "canvas_shape": input_grid.shape,
        "tile_count": len(tiles), "tile_origins": tuple(tile_origins), "windows": windows,
        "accents": tuple(next(int(v) for v in np.unique(tile) if v != carrier) for tile in tiles),
        "carrier_counts": tuple(int(np.count_nonzero(tile == carrier)) for tile in tiles),
        "orientation_counts": visible, "edge_window": edge, "accent_border": accent_border,
        "island": island, "visible_rotation": any(rotations), "necessary_reflection": any(reflections),
        "output": expected,
    }


def _too_close(first, second, gap=1):
    r, c, h, w = first
    rr, cc, hh, ww = second
    return not (r + h + gap <= rr or rr + hh + gap <= r or
                c + w + gap <= cc or cc + ww + gap <= c)


def _sample_windows(rng, height, width, count, edge_policy=None):
    slots = [(r, c) for r in range(height - 2) for c in range(width - 2)]
    windows = []
    for index in range(count):
        available = [(r, c) for r, c in slots if all(not _too_close((r, c, 3, 3), (rr, cc, 3, 3), 0)
                                                     for rr, cc in windows)]
        if edge_policy == "interior":
            available = [(r, c) for r, c in available if 0 < r < height - 3 and 0 < c < width - 3]
        if not available:
            return None
        if index == 0 and (edge_policy == "edge" or (edge_policy is None and rng.random() < 0.48)):
            edge = [(r, c) for r, c in available if r == 0 or c == 0 or r == height - 3 or c == width - 3]
            if edge:
                available = edge
        elif index > 0 and rng.random() < 0.25:
            adjacent = [(r, c) for r, c in available if any(
                (r == rr and abs(c - cc) == 3) or (c == cc and abs(r - rr) == 3)
                for rr, cc in windows)]
            if adjacent:
                available = adjacent
        windows.append(available[int(rng.integers(len(available)))])
    return windows


def _sample_dimensions(rng, count, regime):
    minimum = 7 if count <= 3 else (9 if count <= 5 else 11)
    if regime == "target_test" and rng.random() < 0.35:
        minimum += 1
    height = int(rng.integers(minimum, 19))
    width = int(rng.integers(minimum, 19))
    return height, width


def _template_masks():
    center = np.zeros((3, 3), dtype=bool)
    center[1, 1] = True
    bar = np.zeros((3, 3), dtype=bool)
    bar[1, :] = True
    cross = np.zeros((3, 3), dtype=bool)
    cross[0, 1] = cross[1, 0] = cross[1, 2] = cross[2, 1] = True
    corners = np.zeros((3, 3), dtype=bool)
    corners[0, 0] = corners[0, 2] = corners[2, 0] = corners[2, 2] = True
    block = np.zeros((3, 3), dtype=bool)
    block[:2, :2] = True
    diagonal = np.eye(3, dtype=bool)
    dense = np.ones((3, 3), dtype=bool)
    dense[1, 1] = False
    return (center, bar, cross, corners, block, diagonal, dense)


def _sample_mask(rng, forbidden, used_orbits, requested=None):
    allowed = [index for index in range(9) if index not in forbidden]
    templates = _template_masks()
    choices = []
    if requested == "asymmetric":
        modes = ["asymmetric"]
    elif requested == "island":
        modes = ["island"]
    else:
        modes = ["template", "template", "template", "random", "random", "asymmetric"]
    for _ in range(150):
        mode = modes[int(rng.integers(len(modes)))]
        if mode == "template":
            mask = templates[int(rng.integers(len(templates)))].copy()
            mask = _transform(mask, int(rng.integers(8)))
        elif mode == "island":
            mask = templates[2].copy()
        else:
            if not allowed:
                continue
            count = int(rng.integers(1, min(8, len(allowed)) + 1))
            mask = np.zeros((3, 3), dtype=bool)
            mask.flat[rng.choice(allowed, size=count, replace=False)] = True
        if np.any(mask.ravel()[list(forbidden)]):
            continue
        orientation_count = sum(1 for _ in _variants(mask))
        if mode == "asymmetric" and orientation_count != 8:
            continue
        orbit = _mask_orbit(mask)
        if orbit in used_orbits:
            continue
        choices.append((mask, orbit))
        if requested is not None or rng.random() < 0.75:
            break
    if not choices:
        return None
    mask, orbit = choices[int(rng.integers(len(choices)))]
    used_orbits.add(orbit)
    return mask


def _compose_input(rng, board, tiles, background):
    bh, bw = board.shape
    count = len(tiles)
    min_extra = 6 if count <= 3 else 9
    for _ in range(60):
        ch = min(30, bh + int(rng.integers(min_extra, 14)))
        cw = min(30, bw + int(rng.integers(min_extra, 14)))
        br = int(rng.integers(1, ch - bh))
        bc = int(rng.integers(1, cw - bw))
        board_box = (br, bc, bh, bw)
        slots = [(r, c) for r in range(1, ch - 3) for c in range(1, cw - 3)
                 if not _too_close((r, c, 3, 3), board_box)]
        rng.shuffle(slots)
        chosen = []
        for r, c in slots:
            if all(not _too_close((r, c, 3, 3), (rr, cc, 3, 3)) for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    break
        if len(chosen) < count:
            continue
        canvas = np.full((ch, cw), background, dtype=np.int8)
        canvas[br:br + bh, bc:bc + bw] = board
        for index in rng.permutation(count):
            r, c = chosen.pop()
            canvas[r:r + 3, c:c + 3] = tiles[int(index)]
        return canvas
    return None


def _sample_colors(rng, fixed):
    if fixed is not None:
        return fixed
    background = int(rng.integers(10))
    carrier = int(rng.choice([c for c in range(10) if c != background]))
    pool = tuple(int(c) for c in rng.permutation([c for c in range(10) if c not in (background, carrier)]))
    return background, carrier, pool


def _generate_pair(rng, regime, colors, count, forced=None, edge_policy=None, allowed_operations=None):
    background, carrier, pool = colors
    for _ in range(280):
        bh, bw = _sample_dimensions(rng, count, regime)
        windows = _sample_windows(rng, bh, bw, count, edge_policy)
        if windows is None:
            continue
        source_tiles = []
        target_tiles = []
        used_orbits = set()
        valid = True
        # Force some observable demonstrations, but let all other tile types
        # and dihedral operations vary independently.
        for index, (r, c) in enumerate(windows):
            forbidden = set()
            if r == 0:
                forbidden.update((0, 1, 2))
            if r == bh - 3:
                forbidden.update((6, 7, 8))
            if c == 0:
                forbidden.update((0, 3, 6))
            if c == bw - 3:
                forbidden.update((2, 5, 8))
            special = forced if index == 0 else None
            if special in ("rotate", "reflect", "plain"):
                mask = _sample_mask(rng, forbidden, used_orbits, "asymmetric")
            elif special == "island":
                mask = _sample_mask(rng, forbidden, used_orbits, "island")
            else:
                mask = _sample_mask(rng, forbidden, used_orbits)
            if mask is None:
                valid = False
                break
            if special == "rotate":
                operation = int(rng.integers(1, 4))
            elif special == "reflect":
                operation = int(rng.integers(4, 8))
            elif special == "plain":
                operation = 0
            else:
                operation = int(rng.choice(allowed_operations)) if allowed_operations is not None else int(rng.integers(8))
            accent = int(pool[int(rng.integers(min(len(pool), max(2, count + 1))))])
            target = np.where(mask, carrier, accent).astype(np.int8)
            inverse = (4 - operation) % 4 if operation < 4 else operation
            source = _transform(target, inverse)
            source_tiles.append(source)
            target_tiles.append(target)
        if not valid:
            continue
        input_board = np.full((bh, bw), carrier, dtype=np.int8)
        output_board = np.full((bh, bw), carrier, dtype=np.int8)
        for (r, c), target in zip(windows, target_tiles):
            input_board[r:r + 3, c:c + 3][target == carrier] = background
            output_board[r:r + 3, c:c + 3] = target
        canvas = _compose_input(rng, input_board, source_tiles, background)
        if canvas is None:
            continue
        try:
            metadata = _analyze_pair(canvas, output_board)
        except AssertionError:
            continue
        if forced == "rotate" and not metadata["visible_rotation"]:
            continue
        if forced == "reflect" and not metadata["necessary_reflection"]:
            continue
        return {"input": canvas, "output": output_board}
    raise RuntimeError("could not make a uniquely completable pair")


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    stable_palette = bool(rng.random() < 0.57)
    shared_colors = _sample_colors(rng, None) if stable_palette else None
    regime_mode = str(rng.choice(("progressive", "balanced", "modest_test"), p=(0.27, 0.53, 0.20)))
    rotation_plan = "rotate" if rng.random() < 0.5 else "reflect"
    used_inputs = set()
    train, test = [], []
    train_roles = []
    demonstrated_rotation = False
    demonstrated_reflection = False
    for index in range(num_train + num_test):
        regime = "source_train" if index < num_train else "target_test"
        if index == 0:
            count = int(rng.integers(1, 3))
            forced = "plain" if rng.random() < 0.55 else None
        elif index == 1:
            count = int(rng.integers(4, 6)) if regime_mode == "modest_test" else int(rng.integers(2, 5))
            forced = rotation_plan
        else:
            count = int(rng.choice((1, 2, 3, 4, 5, 6, 7), p=(0.08, 0.14, 0.19, 0.22, 0.19, 0.12, 0.06)))
            forced = "island" if rng.random() < 0.12 else None
        if regime_mode == "progressive" and regime == "source_train" and index >= 2:
            count = min(count, 3)
        if regime_mode == "progressive" and regime == "target_test":
            count = max(count, int(rng.integers(4, 7)))
        if regime_mode == "modest_test" and regime == "target_test":
            count = min(count, 3)
        if regime_mode == "balanced" and regime == "target_test" and index == num_train and rng.random() < 0.5:
            count = max(count, int(rng.integers(3, 6)))
        edge_policy = ("interior" if regime == "source_train" else "edge") if regime_mode == "progressive" else None
        if edge_policy == "edge" and forced == "island":
            forced = None
        allowed_operations = None
        if regime == "target_test" and not demonstrated_reflection:
            allowed_operations = (0, 1, 2, 3) if demonstrated_rotation else (0,)
        elif regime == "target_test" and not demonstrated_rotation:
            allowed_operations = (0,)
        for _ in range(30):
            colors = _sample_colors(rng, shared_colors)
            if not stable_palette and index == 1 and colors[:2] == train_roles[0]:
                continue
            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            pair = _generate_pair(np.random.default_rng(pair_seed), regime, colors, count, forced,
                                  edge_policy, allowed_operations)
            data = _analyze_pair(pair["input"])
            if regime == "target_test" and ((data["visible_rotation"] and not demonstrated_rotation) or
                                            (data["necessary_reflection"] and not demonstrated_reflection)):
                continue
            key = _grid_key(pair["input"])
            if key not in used_inputs:
                used_inputs.add(key)
                break
        else:
            raise RuntimeError("could not create a distinct pair")
        if regime == "source_train":
            train.append(pair)
            train_roles.append(colors[:2])
            demonstrated_rotation |= data["visible_rotation"]
            demonstrated_reflection |= data["necessary_reflection"]
        else:
            test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            assert input_key not in keys
            keys.add(input_key)
            source_hash = hashlib.sha256(pair["input"].astype(np.int8).tobytes()).hexdigest()
            assert (pair["input"].shape, source_hash) not in _SOURCE_INPUTS
            metadata[split].append(_analyze_pair(pair["input"], pair["output"]))
    train_roles = {(x["background"], x["carrier"]) for x in metadata["train"]}
    test_roles = {(x["background"], x["carrier"]) for x in metadata["test"]}
    if len(train_roles) == 1:
        assert test_roles <= train_roles
    assert any(x["visible_rotation"] or x["necessary_reflection"] for x in metadata["train"])
    assert not any(x["visible_rotation"] for x in metadata["test"]) or any(
        x["visible_rotation"] for x in metadata["train"])
    assert not any(x["necessary_reflection"] for x in metadata["test"]) or any(
        x["necessary_reflection"] for x in metadata["train"])


def _equal_puzzles(first, second):
    return all(np.array_equal(a[k], b[k]) for split in ("train", "test")
               for a, b in zip(first[split], second[split]) for k in ("input", "output")) and all(
                   len(first[split]) == len(second[split]) for split in ("train", "test"))


def _must_raise(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    for seed, train_count, test_count in ((1101, 2, 1), (1102, 10, 1), (1103, 10, 4),
                                          (1201, None, 1), (1202, None, 4),
                                          (1203, 2, None), (1204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    _must_raise(lambda: generate_puzzle(1, num_train=1))
    _must_raise(lambda: generate_puzzle(2, num_train=11))
    _must_raise(lambda: generate_puzzle(3, num_test=0))
    _must_raise(lambda: generate_puzzle(4, num_test=5))
    _must_raise(lambda: generate_puzzle(5, 2, 2))
    _must_raise(lambda: generate_puzzle(6, 3, 4))
    assert _equal_puzzles(generate_puzzle(99173, 4, 2), generate_puzzle(99173, 4, 2))
    assert not _equal_puzzles(generate_puzzle(7001, 4, 2), generate_puzzle(7002, 4, 2))
    train_counts, test_counts, shapes, layouts, signatures, tile_counts = set(), set(), set(), set(), set(), set()
    symmetric = edge = border = island = landscape_train = portrait_test = staggered = 0
    singleton = high_carrier = one_orientation = adjacent = multiside = background_zero = 0
    repeated_accent = rotations = reflections = edge_symmetric = 0
    variable_puzzles = stable_puzzles = five_accent_pairs = high_tile_pairs = 0
    mixed_shapes = mixed_counts = harder_tiles = harder_size = progressive_edges = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = set()
        local_shapes, local_counts = set(), set()
        train_data, test_data = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analyze_pair(pair["input"], pair["output"])
                roles.add((data["background"], data["carrier"]))
                local_shapes.add(data["board_shape"])
                local_counts.add(data["tile_count"])
                (train_data if split == "train" else test_data).append(data)
                shapes.add((split, data["board_shape"]))
                layouts.add((data["board_shape"], tuple(sorted(data["windows"]))))
                signatures.add(_grid_key(pair["input"]))
                tile_counts.add(data["tile_count"])
                symmetric += any(x < 8 for x in data["orientation_counts"])
                one_orientation += any(x == 1 for x in data["orientation_counts"])
                singleton += 1 in data["carrier_counts"]
                high_carrier += any(x >= 7 for x in data["carrier_counts"])
                edge += data["edge_window"]
                border += data["accent_border"]
                island += data["island"]
                edge_symmetric += data["edge_window"] and any(x < 8 for x in data["orientation_counts"])
                background_zero += data["background"] == 0
                repeated_accent += len(set(data["accents"])) < data["tile_count"]
                rotations += data["visible_rotation"]
                reflections += data["necessary_reflection"]
                adjacent += any((r == rr and abs(c - cc) == 3) or (c == cc and abs(r - rr) == 3)
                                for i, (r, c) in enumerate(data["windows"]) for rr, cc in data["windows"][i + 1:])
                top, left = data["board_origin"]
                height, width = data["board_shape"]
                sides = set()
                for r, c in data["tile_origins"]:
                    if r + 3 <= top:
                        sides.add("above")
                    if r >= top + height:
                        sides.add("below")
                    if c + 3 <= left:
                        sides.add("left")
                    if c >= left + width:
                        sides.add("right")
                multiside += len(sides) >= 2
                landscape_train += split == "train" and data["board_shape"][1] > data["board_shape"][0]
                portrait_test += split == "test" and data["board_shape"][0] >= data["board_shape"][1]
                staggered += len({r % 3 for r, _ in data["windows"]}) > 1 or len({c % 3 for _, c in data["windows"]}) > 1
                five_accent_pairs += len(set(data["accents"])) >= 5
                high_tile_pairs += data["tile_count"] >= 6
        variable_puzzles += len(roles) > 1
        stable_puzzles += len(roles) == 1
        mixed_shapes += len(local_shapes) > 1
        mixed_counts += len(local_counts) > 1
        harder_tiles += any(x["tile_count"] > max(y["tile_count"] for y in train_data) for x in test_data)
        harder_size += any(x["board_shape"][0] * x["board_shape"][1] >
                           max(y["board_shape"][0] * y["board_shape"][1] for y in train_data) for x in test_data)
        progressive_edges += (not any(x["edge_window"] for x in train_data) and
                              any(x["edge_window"] for x in test_data))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(layouts) >= 100 and len(signatures) >= 150
    assert {1, 2, 3, 4, 5, 6} <= tile_counts
    assert symmetric >= 30 and edge >= 10 and border >= 5 and island >= 2
    assert landscape_train >= 10 and portrait_test >= 5 and staggered >= 15
    assert variable_puzzles >= 5 and five_accent_pairs >= 1 and high_tile_pairs >= 1
    assert stable_puzzles >= 5 and mixed_shapes >= 30 and mixed_counts >= 30
    assert singleton >= 10 and high_carrier >= 10 and one_orientation >= 20
    assert adjacent >= 10 and multiside >= 10 and background_zero >= 5
    assert repeated_accent >= 10 and rotations >= 10 and reflections >= 10
    assert edge_symmetric >= 5 and harder_tiles >= 2 and harder_size >= 2 and progressive_edges >= 2
