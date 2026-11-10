"""Generate solved rightward rigid-square packing puzzles."""

import numpy as np


_SOURCE_INPUT_CODES = frozenset(
    "".join(rows)
    for rows in (
        (
            "000000000000000", "000000000000000", "001200000000000",
            "002100058000000", "000000085000000", "000000000000000",
            "000000000000000", "000006400000000", "089004600000000",
            "098000000000000", "000000000210000", "000000000120000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
        (
            "000000000000000", "083800000000000", "038300000000000",
            "083800000000000", "000000000000000", "000212000000000",
            "000121000000000", "000212001410000", "000000004140000",
            "076700001410000", "067600000000000", "076700000000000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
        (
            "000000000000000", "009200000000000", "002900000373000",
            "000000000737000", "000000000373000", "000000000000000",
            "000000000000000", "000006860000000", "000008680000000",
            "000006860008500", "000000000005800", "000000000000000",
            "002500000000000", "005200000000000", "000000000000000",
        ),
        (
            "000686000000000", "000868000000000", "000686025252000",
            "000000052525000", "000000025252000", "002100052525000",
            "001200025252000", "000000000000000", "000000000000000",
            "035000000000000", "053000084800000", "000000048400000",
            "000000084800000", "001700000000000", "007100000000000",
        ),
    )
)


def _is_count(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        key = int(seed)
        num_test = 1 + key % 4
        low = num_test + 1
        num_train = low + (key // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _row_overlap(a, b):
    return a["r"] < b["r"] + b["s"] and b["r"] < a["r"] + a["s"]


def _placement_conflict(a, b):
    row_overlap = _row_overlap(a, b)
    col_overlap = a["c"] < b["c"] + b["s"] and b["c"] < a["c"] + a["s"]
    if row_overlap and col_overlap:
        return True
    if row_overlap and (
        a["c"] + a["s"] == b["c"] or b["c"] + b["s"] == a["c"]
    ):
        return True
    if col_overlap and (
        a["r"] + a["s"] == b["r"] or b["r"] + b["s"] == a["r"]
    ):
        return True
    return False


def _gravity_columns(tiles, width):
    """Pack right to left, considering only tiles whose rows intersect."""
    targets = [None] * len(tiles)
    order = sorted(
        range(len(tiles)),
        key=lambda i: (tiles[i]["c"] + tiles[i]["s"], tiles[i]["c"]),
        reverse=True,
    )
    for position, i in enumerate(order):
        target = width - tiles[i]["s"]
        for j in order[:position]:
            if _row_overlap(tiles[i], tiles[j]):
                target = min(target, targets[j] - tiles[i]["s"])
        assert target >= tiles[i]["c"]
        targets[i] = target
    return targets


def _sample_dimensions(rng, avoid=None):
    for _ in range(20):
        kind = int(rng.integers(0, 5))
        if kind == 0:
            height = width = 15
        elif kind == 1:
            height = width = int(rng.integers(13, 24))
        elif kind == 2:
            height = int(rng.integers(13, 22))
            width = height + int(rng.integers(2, 7))
        elif kind == 3:
            width = int(rng.integers(13, 22))
            height = width + int(rng.integers(2, 7))
        else:
            height = int(rng.integers(13, 25))
            width = int(rng.integers(13, 25))
        if (height, width) != avoid:
            return height, width
    return 15, 16 if avoid == (15, 15) else 15


def _sample_chain(rng, height, width, length, stable, regime, first_train):
    """Seed one collision group, optionally with a large tile in any role."""
    for _ in range(120):
        if first_train:
            sizes = [2, 3, int(rng.choice((2, 3)))]
            rng.shuffle(sizes)
        else:
            sizes = [int(rng.choice((2, 3))) for _ in range(length)]
        big_chance = 0.32 if regime == "source_train" else 0.56
        if rng.random() < big_chance:
            big_index = int(rng.integers(0, length))
            sizes[big_index] = int(rng.choice((4, 5)))
            if first_train:
                others = [i for i in range(length) if i != big_index]
                sizes[others[0]], sizes[others[1]] = 2, 3
        gaps = [int(rng.integers(1, 4)) for _ in range(length - 1)]
        span = sum(sizes) + sum(gaps)
        free = width - span
        if free < (0 if stable else 1):
            continue
        right_margin = 0 if stable else int(rng.integers(1, min(5, free + 1)))
        left_margin = free if stable else int(rng.integers(0, free - right_margin + 1))
        spine = int(rng.integers(1, height - 1))
        layout = []
        column = left_margin
        for i, size in enumerate(sizes):
            low = max(0, spine - size + 1)
            high = min(spine, height - size)
            row = int(rng.integers(low, high + 1))
            layout.append({"r": row, "c": column, "s": size})
            column += size + (gaps[i] if i < length - 1 else 0)
        return layout
    sizes = [2, 3] if length == 2 else [2, 3, 2]
    span = sum(sizes) + length - 1
    column = width - span if stable else 0
    spine = height // 2
    layout = []
    for size in sizes:
        layout.append({"r": spine - size // 2, "c": column, "s": size})
        column += size + 1
    return layout


def _sample_independent(rng, height, width):
    """Make a whole pair of row-disjoint, independently moving tiles."""
    large_canvas = height * width >= 400
    maximum = min(7, (height + 1) // (4 if large_canvas else 3))
    minimum = max(3, maximum - 1)
    for _ in range(80):
        count = int(rng.integers(minimum, maximum + 1))
        choices = (3, 3, 4) if large_canvas else (2, 3, 3)
        sizes = [int(rng.choice(choices)) for _ in range(count)]
        slack = height - sum(sizes) - (count - 1)
        if slack < 0:
            continue
        spaces = [0] * (count + 1)
        for _ in range(slack):
            spaces[int(rng.integers(0, count + 1))] += 1
        row = spaces[0]
        layout = []
        for i, size in enumerate(sizes):
            column = int(rng.integers(0, width - size))
            layout.append({"r": row, "c": column, "s": size})
            row += size + 1 + spaces[i + 1]
        return layout
    count = min(5, (height + 1) // 4)
    return [
        {"r": 4 * i, "c": 3 * i % (width - 3), "s": 3}
        for i in range(count)
    ]


def _fill_layout(rng, layout, height, width, target_count, protect_edge):
    """Add separated tiles without prescribing their collision roles."""
    for _ in range(max(0, target_count - len(layout))):
        added = False
        for _ in range(350):
            size = int(rng.choice((2, 2, 3, 3, 4, 5)))
            row = int(rng.integers(0, height - size + 1))
            if rng.random() < 0.24:
                column = width - size - int(rng.integers(0, min(3, width - size + 1)))
            else:
                column = int(rng.integers(0, width - size + 1))
            candidate = {"r": row, "c": column, "s": size}
            if protect_edge is not None and (
                _row_overlap(candidate, protect_edge)
                and candidate["c"] > protect_edge["c"]
            ):
                continue
            if all(not _placement_conflict(candidate, other) for other in layout):
                layout.append(candidate)
                added = True
                break
        if not added:
            break
    return layout


def _sample_layout(rng, regime, index, height, width):
    if regime == "source_train" and index == 0:
        mode = "chain"
    elif regime == "source_train" and index == 1:
        mode = "stable"
    else:
        mode = str(rng.choice(("chain", "chain", "short", "stable", "independent")))
    if mode == "independent":
        return _sample_independent(rng, height, width)
    length = 2 if mode in ("short", "stable") else 3
    layout = _sample_chain(
        rng, height, width, length, mode == "stable", regime,
        regime == "source_train" and index == 0,
    )
    area = height * width
    target_count = 4 + max(0, (area - 225) // 105)
    target_count += int(rng.integers(-1, 3))
    if regime == "target_test":
        target_count += int(rng.integers(0, 3))
    target_count = max(length, min(10, target_count))
    # The first training example must visibly show a tile reach the edge.
    protected = layout[-1] if regime == "source_train" and index == 0 else None
    return _fill_layout(rng, layout, height, width, target_count, protected)


def _make_patterns(rng, layout, background):
    available = [color for color in range(10) if color != background]
    rng.shuffle(available)
    pool = available[:int(rng.integers(3, 7))]
    bank = []
    bank_count = min(3, max(2, len(layout) // 2))
    for _ in range(bank_count):
        pair = rng.choice(pool, size=2, replace=False)
        bank.append((int(pair[0]), int(pair[1])))
    patterns = []
    for i, tile in enumerate(layout):
        pair = bank[0] if i < 2 else bank[1] if i == 2 else bank[int(rng.integers(0, bank_count))]
        first, second = pair if rng.random() < 0.5 else pair[::-1]
        size = tile["s"]
        rows, cols = np.indices((size, size))
        if i == 0 or rng.random() < 0.83:
            selector = (rows + cols) % 2
        elif rng.random() < 0.5:
            selector = rows % 2
        else:
            selector = cols % 2
        patterns.append(np.where(selector == 0, first, second).astype(np.int8))
    return patterns


def _generate_pair(rng, regime, index, background, avoid_shape=None):
    height, width = _sample_dimensions(rng, avoid_shape)
    layout = _sample_layout(rng, regime, index, height, width)
    patterns = _make_patterns(rng, layout, background)
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = np.full((height, width), background, dtype=np.int8)
    targets = _gravity_columns(layout, width)
    for tile, pattern, target in zip(layout, patterns, targets):
        row, column, size = tile["r"], tile["c"], tile["s"]
        input_grid[row:row + size, column:column + size] = pattern
        destination = output_grid[row:row + size, target:target + size]
        assert np.all(destination == background)
        destination[:] = pattern
    return {"input": input_grid, "output": output_grid}


def _grid_signature(grid):
    return tuple(grid.shape), np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _flat_code(grid):
    return "".join(str(int(value)) for value in grid.flat)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2
    return int(colors[index])


def _extract_tiles(grid, background):
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    tiles = []
    for start_r in range(height):
        for start_c in range(width):
            if visited[start_r, start_c] or grid[start_r, start_c] == background:
                continue
            stack = [(start_r, start_c)]
            visited[start_r, start_c] = True
            cells = []
            while stack:
                row, column = stack.pop()
                cells.append((row, column))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, column + dc
                    if (
                        0 <= nr < height and 0 <= nc < width
                        and not visited[nr, nc] and grid[nr, nc] != background
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            side = r1 - r0 + 1
            assert side == c1 - c0 + 1
            assert len(cells) == side * side
            pattern = grid[r0:r1 + 1, c0:c1 + 1].copy()
            assert np.all(pattern != background)
            assert len(np.unique(pattern)) == 2
            tiles.append({"r": r0, "c": c0, "s": side, "pattern": pattern})
    assert len(tiles) >= 2
    return tiles


def _apply_gravity(input_grid):
    background = _background(input_grid)
    tiles = _extract_tiles(input_grid, background)
    targets = _gravity_columns(tiles, input_grid.shape[1])
    result = np.full_like(input_grid, background)
    for tile, target in zip(tiles, targets):
        row, size = tile["r"], tile["s"]
        destination = result[row:row + size, target:target + size]
        assert np.all(destination == background)
        destination[:] = tile["pattern"]
    return result


def _pair_info(grid):
    background = _background(grid)
    tiles = _extract_tiles(grid, background)
    targets = _gravity_columns(tiles, grid.shape[1])
    collision_count = sum(
        _row_overlap(a, b)
        for i, a in enumerate(tiles)
        for b in tiles[i + 1:]
    )
    stable = sum(target == tile["c"] for tile, target in zip(tiles, targets))
    blocked = sum(
        target > tile["c"] and target + tile["s"] < grid.shape[1]
        for tile, target in zip(tiles, targets)
    )
    moving_edge = sum(
        target > tile["c"] and target + tile["s"] == grid.shape[1]
        for tile, target in zip(tiles, targets)
    )
    checker = 0
    for tile in tiles:
        pattern = tile["pattern"]
        rows, cols = np.indices(pattern.shape)
        if np.all(pattern == np.where(
            (rows + cols) % 2 == 0, pattern[0, 0], pattern[0, 1]
        )):
            checker += 1
    return {
        "background": background,
        "tiles": tiles,
        "targets": targets,
        "collisions": collision_count,
        "stable": stable,
        "blocked": blocked,
        "moving_edge": moving_edge,
        "checker": checker,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[part], list) for part in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    train_info, test_info = [], []
    shapes = set()
    for part, infos in (("train", train_info), ("test", test_info)):
        for pair in puzzle[part]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            assert not np.array_equal(inp, out)
            assert np.array_equal(out, _apply_gravity(inp))
            assert _background(inp) == _background(out)
            key = _grid_signature(inp)
            pair_key = key, _grid_signature(out)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            if inp.shape == (15, 15):
                assert _flat_code(inp) not in _SOURCE_INPUT_CODES
            info = _pair_info(inp)
            assert info["checker"] >= 1
            infos.append(info)
            shapes.add(inp.shape)
    assert len(shapes) >= 2
    assert len({len(info["tiles"]) for info in train_info + test_info}) >= 2
    assert any(info["collisions"] >= 2 for info in train_info)
    assert any(info["blocked"] >= 1 for info in train_info)
    assert any(info["moving_edge"] >= 1 for info in train_info)
    assert any(info["stable"] >= 1 for info in train_info)
    assert {info["background"] == 0 for info in train_info} == {False, True}
    assert all(2 <= tile["s"] <= 5 for info in train_info + test_info for tile in info["tiles"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # The only rule convention is rightward motion. Blank color is visible in
    # each input, so foreground and background values are pair-level choices.
    first_background = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_shape = None
    first_tile_count = None
    for part, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if part == "train" else "target_test"
        for index in range(count):
            if part == "train" and index == 0:
                background = first_background
            elif part == "train" and index == 1:
                background = int(rng.integers(1, 10)) if first_background == 0 else 0
            else:
                background = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
            for _ in range(30):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                pair = _generate_pair(
                    pair_rng, regime, index, background,
                    first_shape if part == "train" and index == 1 else None,
                )
                tile_count = len(_extract_tiles(pair["input"], background))
                if part == "train" and index == 1 and tile_count == first_tile_count:
                    continue
                key = _grid_signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[part].append(pair)
                    if part == "train" and index == 0:
                        first_shape = pair["input"].shape
                        first_tile_count = tile_count
                    break
            else:
                raise RuntimeError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple(
        tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
              for pair in puzzle[part])
        for part in ("train", "test")
    )


def _simulate_gravity(input_grid):
    """Independent one-cell movement check using only the input grid."""
    background = _background(input_grid)
    tiles = _extract_tiles(input_grid, background)
    order = sorted(
        range(len(tiles)),
        key=lambda i: (tiles[i]["c"] + tiles[i]["s"], tiles[i]["c"]),
        reverse=True,
    )
    positions = [tile["c"] for tile in tiles]
    for i in order:
        tile = tiles[i]
        while positions[i] + tile["s"] < input_grid.shape[1]:
            next_column = positions[i] + 1
            if any(
                j != i and _row_overlap(tile, other)
                and next_column < positions[j] + other["s"]
                and positions[j] < next_column + tile["s"]
                for j, other in enumerate(tiles)
            ):
                break
            positions[i] = next_column
    result = np.full_like(input_grid, background)
    for tile, column in zip(tiles, positions):
        row, size = tile["r"], tile["s"]
        result[row:row + size, column:column + size] = tile["pattern"]
    return result


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, layouts, puzzles = set(), set(), set(), set()
    checker_tiles = total_tiles = all_checker_pairs = 0
    square = wide = tall = fifteen = stable_test = independent = 0
    large_train = large_test = large_edge = large_blocked = 0
    five_edge = five_blocked = familiar_test = 0
    short_collision = long_collision = repeated_pairs = nonzero = 0
    size_values, gap_values, large_occupancy = set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_signature(puzzle))
        assert {_pair_info(p["input"])["background"] == 0 for p in puzzle["train"]} == {False, True}
        assert len({p["input"].shape for part in puzzle.values() for p in part}) >= 2
        assert len({len(_pair_info(p["input"])["tiles"]) for part in puzzle.values() for p in part}) >= 2
        for part in ("train", "test"):
            for pair in puzzle[part]:
                inp = pair["input"]
                assert np.array_equal(pair["output"], _simulate_gravity(inp))
                info = _pair_info(inp)
                checker_tiles += info["checker"]
                total_tiles += len(info["tiles"])
                all_checker_pairs += info["checker"] == len(info["tiles"])
                nonzero += info["background"] != 0
                short_collision += info["collisions"] == 1
                long_collision += info["collisions"] >= 3
                color_pairs = {
                    tuple(sorted(int(c) for c in np.unique(tile["pattern"])))
                    for tile in info["tiles"]
                }
                repeated_pairs += len(color_pairs) < len(info["tiles"])
                height, width = inp.shape
                if height * width >= 400:
                    large_occupancy.append(
                        sum(tile["s"] ** 2 for tile in info["tiles"])
                        / (height * width)
                    )
                shapes.add((height, width))
                square += height == width
                wide += width > height
                tall += height > width
                fifteen += (height, width) == (15, 15)
                palettes.add(tuple(sorted(int(c) for c in np.unique(inp))))
                layouts.add((inp.shape, np.ascontiguousarray(inp != info["background"]).tobytes()))
                independent += info["collisions"] == 0
                if part == "test":
                    stable_test += info["stable"] > 0
                    familiar_test += all(tile["s"] <= 3 for tile in info["tiles"])
                for i, left in enumerate(info["tiles"]):
                    for right in info["tiles"][i + 1:]:
                        if _row_overlap(left, right):
                            gap_values.add(abs(left["c"] - right["c"]) - (
                                left["s"] if left["c"] < right["c"] else right["s"]
                            ))
                for tile, target in zip(info["tiles"], info["targets"]):
                    size_values.add(tile["s"])
                    if tile["s"] >= 4:
                        if part == "train":
                            large_train += 1
                        else:
                            large_test += 1
                            large_edge += target + tile["s"] == width
                            large_blocked += target + tile["s"] < width
                    if part == "test" and tile["s"] == 5:
                        five_edge += target + 5 == width
                        five_blocked += target + 5 < width
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(palettes) >= 30
    assert len(layouts) >= 150 and len(puzzles) >= 40
    assert square >= 15 and wide >= 15 and tall >= 15 and fifteen >= 8
    assert checker_tiles / total_tiles >= 0.7
    assert all_checker_pairs >= 15
    assert repeated_pairs >= 300 and nonzero >= 40
    assert stable_test >= 3 and independent >= 3
    assert short_collision >= 5 and long_collision >= 50
    assert familiar_test >= 10
    assert large_train >= 12 and large_test >= 12
    assert large_edge >= 3 and large_blocked >= 3
    assert five_edge >= 3 and five_blocked >= 3
    assert size_values == {2, 3, 4, 5}
    assert len(gap_values) >= 5 and max(gap_values) >= 5
    assert len(large_occupancy) >= 20 and np.median(large_occupancy) >= 0.08
    for i, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + i, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for i, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(2000 + i, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    assert _puzzle_signature(generate_puzzle(314159, 7, 3)) == _puzzle_signature(
        generate_puzzle(314159, 7, 3)
    )
    assert len({_puzzle_signature(generate_puzzle(3000 + i, 5, 2)) for i in range(12)}) >= 11
    for i, (train_count, test_count) in enumerate(
        ((1, None), (11, None), (None, 0), (None, 5),
         (2, 2), (3, 4), (2.0, 1))
    ):
        try:
            generate_puzzle(4000 + i, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count arguments did not raise ValueError")
