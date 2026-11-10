import numpy as np


def _grid(*rows: str) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (_array_key(pair["input"]), _array_key(pair["output"]))


_SOURCE_PAIRS = (
    (
        _grid(
            "0000000",
            "0585000",
            "0585000",
            "0888000",
            "0000000",
            "0000000",
            "0000000",
        ),
        _grid(
            "0000000",
            "0858000",
            "0858000",
            "0555000",
            "0000000",
            "0000000",
            "0000000",
        ),
    ),
    (
        _grid(
            "0000000000",
            "0884000000",
            "0484000000",
            "0884000000",
            "0000003220",
            "0000003320",
            "0000003220",
            "0000000000",
            "0036300000",
            "0036300000",
            "0033300000",
            "0000000000",
            "0000000000",
        ),
        _grid(
            "0000000000",
            "0448000000",
            "0848000000",
            "0448000000",
            "0000002330",
            "0000002230",
            "0000002330",
            "0000000000",
            "0063600000",
            "0063600000",
            "0066600000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        _grid(
            "000000000000",
            "011100000000",
            "081800032300",
            "088800022200",
            "000000023200",
            "000000000000",
            "000611000000",
            "000166000000",
            "000166004450",
            "000000004550",
            "000000005550",
            "000000000000",
        ),
        _grid(
            "000000000000",
            "088800000000",
            "018100023200",
            "011100033300",
            "000000032300",
            "000000000000",
            "000166000000",
            "000611000000",
            "000611005540",
            "000000005440",
            "000000004440",
            "000000000000",
        ),
    ),
    (
        _grid(
            "000000000",
            "032200000",
            "033200000",
            "032200000",
            "000006660",
            "000001110",
            "000001660",
            "000000000",
        ),
        _grid(
            "000000000",
            "023300000",
            "022300000",
            "023300000",
            "000001110",
            "000006660",
            "000006110",
            "000000000",
        ),
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    (_array_key(input_grid), _array_key(output_grid))
    for input_grid, output_grid in _SOURCE_PAIRS
)



def _validate_seed(seed: int) -> int:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    return seed


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test
    # Count randomness is seed-derived, so the same request always has the same result.
    rng = np.random.default_rng(seed)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _object_count(rng: np.random.Generator, regime: str) -> int:
    values = np.arange(1, 8)
    weights = ([20, 18, 18, 17, 13, 9, 5] if regime == "source_train"
               else [9, 15, 19, 20, 17, 13, 7])
    return int(rng.choice(values, p=np.asarray(weights) / 100))


def _tile_size(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    chance = float(rng.random())
    if chance < (0.55 if regime == "source_train" else 0.40):
        return 3, 3
    if chance < (0.73 if regime == "source_train" else 0.60):
        side = int(rng.choice(np.asarray([2, 4, 5])))
        return side, side
    height = int(rng.integers(2, 7))
    width = int(rng.integers(2, 7))
    if height == width:
        width = 2 + (width - 1) % 5
    return height, width


def _sample_dimension(rng: np.random.Generator, minimum: int) -> int:
    if rng.random() < 0.10:
        return 30
    roll = float(rng.random())
    if roll < 0.45:
        upper = min(30, minimum + 8)
        return int(rng.integers(minimum, upper + 1))
    if roll < 0.80:
        return int(rng.integers(minimum, 31))
    return int(rng.integers(max(minimum, 19), 31))


def _separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return (ar + ah < br or br + bh < ar or
            ac + aw < bc or bc + bw < ac)


def _stagger_proposal(rng: np.random.Generator, prior: tuple[int, int, int, int],
                      height: int, width: int) -> tuple[int, int]:
    row, col, old_h, old_w = prior
    if int(rng.integers(0, 2)):
        # Partly overlapping row projections with a horizontal background gap.
        delta = int(rng.integers(1, min(height, old_h)))
        top = row + delta * (-1 if int(rng.integers(0, 2)) else 1)
        gap = int(rng.integers(1, 5))
        left = col + old_w + gap if int(rng.integers(0, 2)) else col - width - gap
    else:
        delta = int(rng.integers(1, min(width, old_w)))
        left = col + delta * (-1 if int(rng.integers(0, 2)) else 1)
        gap = int(rng.integers(1, 5))
        top = row + old_h + gap if int(rng.integers(0, 2)) else row - height - gap
    return top, left


def _place_tiles(rng: np.random.Generator, sizes: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    area = sum(h * w for h, w in sizes)
    minimum = max(5, max(max(size) for size in sizes), int(np.ceil(np.sqrt(area * 2.3))))
    for _ in range(100):
        height = _sample_dimension(rng, minimum)
        width = _sample_dimension(rng, minimum)
        if height * width <= 2 * area:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        for tile_h, tile_w in sizes:
            placed = False
            for proposal in range(110):
                if boxes and proposal < 35:
                    top, left = _stagger_proposal(rng, boxes[int(rng.integers(0, len(boxes)))], tile_h, tile_w)
                elif proposal < 53:
                    side = int(rng.integers(0, 4))
                    top = (0 if side == 0 else height - tile_h if side == 1
                           else int(rng.integers(0, height - tile_h + 1)))
                    left = (0 if side == 2 else width - tile_w if side == 3
                            else int(rng.integers(0, width - tile_w + 1)))
                else:
                    top = int(rng.integers(0, height - tile_h + 1))
                    left = int(rng.integers(0, width - tile_w + 1))
                if not (0 <= top <= height - tile_h and 0 <= left <= width - tile_w):
                    continue
                candidate = (top, left, tile_h, tile_w)
                if all(_separate(candidate, previous) for previous in boxes):
                    boxes.append(candidate)
                    placed = True
                    break
            if not placed:
                break
        if len(boxes) == len(sizes):
            return boxes, (height, width)
    # A guaranteed bounded packing for at most seven objects of side at most six.
    boxes = [(7 * (index // 4), 7 * (index % 4), h, w)
             for index, (h, w) in enumerate(sizes)]
    return boxes, (30, 30)


def _color_pairs(rng: np.random.Generator, background: int, count: int,
                 witness: bool) -> list[tuple[int, int]]:
    colors = [color for color in range(10) if color != background]
    rng.shuffle(colors)
    if count == 1:
        return [(colors[0], colors[1])]
    if witness:
        mode = "shared"
    else:
        choices = ["shared", "disjoint", "repeated"]
        probabilities = [0.37, 0.38, 0.25]
        if count >= 3:
            choices.append("mixed")
            probabilities = [0.29, 0.29, 0.20, 0.22]
        mode = str(rng.choice(choices, p=probabilities))
    if mode == "shared":
        pairs = [(colors[0], colors[1]), (colors[0], colors[2])]
        used = 3
    elif mode == "disjoint":
        pairs = [(colors[0], colors[1]), (colors[2], colors[3])]
        used = 4
    elif mode == "repeated":
        pairs = [(colors[0], colors[1]), (colors[0], colors[1])]
        used = 2
    else:
        pairs = [(colors[0], colors[1]), (colors[0], colors[2]), (colors[3], colors[4])]
        used = 5
    maximum = min(9, used + 2 * (count - len(pairs)))
    palette_size = (maximum if maximum == 9 and rng.random() < 0.24
                    else int(rng.integers(used, maximum + 1)))
    while len(pairs) < count:
        remaining = colors[used:palette_size]
        if len(remaining) >= 2:
            pairs.append((remaining[0], remaining[1]))
            used += 2
        elif remaining:
            pairs.append((remaining[0], colors[int(rng.integers(0, used))]))
            used += 1
        else:
            selected = rng.choice(np.asarray(colors[:used]), size=2, replace=False)
            pairs.append((int(selected[0]), int(selected[1])))
    rng.shuffle(pairs)
    return [(a, b) if rng.random() < 0.5 else (b, a) for a, b in pairs]


def _binary_pattern(rng: np.random.Generator, height: int, width: int,
                    regime: str) -> np.ndarray:
    irregular = rng.random() < (0.68 if regime == "source_train" else 0.82)
    if irregular:
        count = int(rng.integers(1, height * width))
        mask = np.zeros(height * width, dtype=bool)
        mask[rng.choice(height * width, size=count, replace=False)] = True
        return mask.reshape(height, width)
    mode = int(rng.integers(0, 5))
    rows, cols = np.indices((height, width))
    if mode == 0:
        mask = (rows + cols) % 2 == 0
    elif mode == 1:
        mask = rows == int(rng.integers(0, height))
    elif mode == 2:
        mask = cols == int(rng.integers(0, width))
    elif mode == 3:
        mask = (rows == 0) | (cols == 0)
    else:
        mask = (rows == height // 2) & (cols == width // 2)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = ~mask
    return mask.copy()


def _generate_pair(rng: np.random.Generator, regime: str, object_count: int,
                   witness: bool) -> dict:
    background = int(rng.integers(0, 10))
    sizes = [_tile_size(rng, regime) for _ in range(object_count)]
    if witness:
        sizes[0] = (3, 3)
    boxes, shape = _place_tiles(rng, sizes)
    colors = _color_pairs(rng, background, object_count, witness)
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = input_grid.copy()
    saved_patterns: dict[tuple[int, int], list[np.ndarray]] = {}
    for (top, left, height, width), (first, second) in zip(boxes, colors):
        prior = saved_patterns.get((height, width), [])
        pattern = (prior[int(rng.integers(0, len(prior)))].copy()
                   if prior and rng.random() < 0.25 else
                   _binary_pattern(rng, height, width, regime))
        saved_patterns.setdefault((height, width), []).append(pattern)
        input_grid[top:top + height, left:left + width] = np.where(pattern, first, second)
        output_grid[top:top + height, left:left + width] = np.where(pattern, second, first)
    return {"input": input_grid, "output": output_grid}


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for row, col in zip(*np.nonzero(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        component = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(component)
    return result


def _analyze_pair(pair: dict) -> dict:
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape, "input/output dimensions changed"
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    foreground = inp != background
    assert np.any(foreground), "empty foreground"
    assert int(np.count_nonzero(~foreground)) > int(np.count_nonzero(foreground)), "background is not dominant"
    assert np.array_equal(foreground, out != background), "foreground support changed"
    expected = inp.copy()
    boxes = []
    for component in _connected_components(foreground):
        rows, cols = zip(*component)
        top, left = min(rows), min(cols)
        height, width = max(rows) - top + 1, max(cols) - left + 1
        assert 2 <= height <= 6 and 2 <= width <= 6, "invalid tile dimensions"
        assert len(component) == height * width, "tile is not a filled rectangle"
        tile = inp[top:top + height, left:left + width]
        colors = np.unique(tile)
        assert len(colors) == 2 and background not in colors, "tile needs two visible foreground colors"
        a, b = int(colors[0]), int(colors[1])
        expected[top:top + height, left:left + width] = np.where(tile == a, b, a)
        boxes.append((top, left, height, width, (a, b)))
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            assert _separate(first[:4], second[:4]), "tiles are not background-separated"
    assert np.array_equal(expected, out), "output is not the independent local color exchange"
    return {"background": background, "tiles": boxes}


def _validate_generated_puzzle_impl(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys"
    assert all(isinstance(puzzle[part], list) for part in ("train", "test")), "splits must be lists"
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts"
    inputs, pairs = set(), set()
    train_metadata = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys"
            for role in ("input", "output"):
                grid = pair[role]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid"
                assert all(1 <= side <= 30 for side in grid.shape), "grid size outside ARC limits"
                assert np.issubdtype(grid.dtype, np.integer), "noninteger grid"
                assert bool(np.all((0 <= grid) & (grid <= 9))), "illegal color"
            input_key, pair_key = _array_key(pair["input"]), _pair_key(pair)
            assert input_key not in inputs and pair_key not in pairs, "duplicate pair or input"
            assert pair_key not in _SOURCE_PAIR_KEYS, "exact source pair copied"
            inputs.add(input_key)
            pairs.add(pair_key)
            metadata = _analyze_pair(pair)
            if split == "train":
                train_metadata.append(metadata)
    # A literal color must have different partners in one demonstrated scene.
    assert any(any(len(partners) >= 2 for partners in _partner_sets(meta).values())
               for meta in train_metadata), "training lacks a local-pair witness"
    assert any(any(h == 3 and w == 3 for _, _, h, w, _ in meta["tiles"])
               for meta in train_metadata), "training lacks a source-size tile"


def _partner_sets(metadata: dict) -> dict[int, set[int]]:
    partners: dict[int, set[int]] = {}
    for _, _, _, _, (a, b) in metadata["tiles"]:
        partners.setdefault(a, set()).add(b)
        partners.setdefault(b, set()).add(a)
    return partners


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError(f"malformed generated puzzle: {error}") from error


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The puzzle-level convention is fixed: each filled object's two colors exchange.
    # Literal colors, backgrounds, geometry, and counts are visible pair properties.
    witness_index = int(rng.integers(0, train_count))
    train_objects = [_object_count(rng, "source_train") for _ in range(train_count)]
    train_objects[witness_index] = int(rng.integers(2, 5))
    test_objects = []
    for _ in range(test_count):
        absent = [count for count in range(1, 8) if count not in train_objects]
        if absent and rng.random() < 0.35:
            test_objects.append(int(rng.choice(np.asarray(absent))))
        else:
            test_objects.append(_object_count(rng, "target_test"))
    puzzle = {"train": [], "test": []}
    inputs, pairs = set(), set()
    plan = [("train", "source_train", count, index == witness_index)
            for index, count in enumerate(train_objects)]
    plan += [("test", "target_test", count, False) for count in test_objects]
    for split, regime, count, witness in plan:
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        for _ in range(40):
            pair = _generate_pair(pair_rng, regime, count, witness)
            input_key, pair_key = _array_key(pair["input"]), _pair_key(pair)
            if input_key not in inputs and pair_key not in pairs and pair_key not in _SOURCE_PAIR_KEYS:
                puzzle[split].append(pair)
                inputs.add(input_key)
                pairs.add(pair_key)
                break
        else:
            raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    def patch_count(mask: np.ndarray) -> int:
        remaining = set(zip(*np.nonzero(mask)))
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                row, column = stack.pop()
                for neighbor in ((row - 1, column), (row + 1, column),
                                 (row, column - 1), (row, column + 1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        return count

    familiar = set()
    rows, columns = np.indices((3, 3))
    for base in (np.eye(3, dtype=bool), (rows + columns) % 2 == 0,
                 (rows == 1) & (columns == 1), rows == 0,
                 (rows == 0) | (columns == 0)):
        for rotation in range(4):
            rotated = np.rot90(base, rotation)
            for variant in (rotated, np.fliplr(rotated)):
                familiar.add(tuple(bool(value) for value in variant.flat))
                familiar.add(tuple(bool(value) for value in (~variant).flat))

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    features = {"counts": set(), "sizes": set(), "shapes": set(), "backgrounds": set(),
                "palettes": set(), "edge": set(), "partial_rows": set(), "partial_cols": set(),
                "relations": set(), "patterns": set()}
    within_background = 0
    within_count = 0
    within_palette = 0
    nine_color_pairs = 0
    mixed_size_pairs = 0
    unfamiliar_patterns = 0
    disconnected_colors = 0
    reused_motifs = 0
    rectangles = 0
    other_squares = 0
    full_size = 0
    wide = tall = crowded = sparse = 0
    heldout_counts = new_test_colors = new_test_pairs = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_meta = [_analyze_pair(pair) for pair in puzzle["train"]]
        train_counts = {len(meta["tiles"]) for meta in train_meta}
        train_colors = {color for meta in train_meta for tile in meta["tiles"]
                        for color in tile[4]}
        train_pairs = {tile[4] for meta in train_meta for tile in meta["tiles"]}
        all_meta = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                metadata = _analyze_pair(pair)
                all_meta.append(metadata)
                grid = pair["input"]
                tiles = metadata["tiles"]
                features["counts"].add(len(tiles))
                features["shapes"].add(grid.shape)
                features["backgrounds"].add(metadata["background"])
                full_size += grid.shape == (30, 30)
                wide += grid.shape[1] > 2 * grid.shape[0]
                tall += grid.shape[0] > 2 * grid.shape[1]
                occupancy = sum(tile[2] * tile[3] for tile in tiles) / grid.size
                crowded += occupancy > 0.30
                sparse += occupancy < 0.10 and max(grid.shape) >= 20
                if split == "test":
                    heldout_counts += len(tiles) not in train_counts
                    new_test_colors += bool({color for tile in tiles for color in tile[4]} - train_colors)
                    new_test_pairs += any(tile[4] not in train_pairs for tile in tiles)
                tile_sizes = {(h, w) for _, _, h, w, _ in tiles}
                if len(tile_sizes) > 1:
                    mixed_size_pairs += 1
                palette = set()
                motifs = {}
                for top, left, h, w, colors in tiles:
                    features["sizes"].add((h, w))
                    rectangles += h != w
                    other_squares += h == w and h != 3
                    palette.update(colors)
                    if top == 0 or left == 0 or top + h == grid.shape[0] or left + w == grid.shape[1]:
                        features["edge"].add(True)
                    tile = grid[top:top + h, left:left + w]
                    mask = tile == colors[0]
                    code = tuple(bool(value) for value in mask.flat)
                    features["patterns"].add((h, w, code))
                    if (h, w) == (3, 3) and code not in familiar:
                        unfamiliar_patterns += 1
                    if patch_count(mask) > 1 and patch_count(~mask) > 1:
                        disconnected_colors += 1
                    motif = (h, w, min(code, tuple(not value for value in code)))
                    if motif in motifs and motifs[motif] != colors:
                        reused_motifs += 1
                    motifs[motif] = colors
                features["palettes"].add(len(palette))
                if len(palette) == 9:
                    nine_color_pairs += 1
                color_sets = [set(tile[4]) for tile in tiles]
                for i, first in enumerate(tiles):
                    for second in tiles[i + 1:]:
                        shared = len(color_sets[i] & set(second[4]))
                        features["relations"].add(shared)
                        ar, ac, ah, aw = first[:4]
                        br, bc, bh, bw = second[:4]
                        overlap_rows = max(0, min(ar + ah, br + bh) - max(ar, br))
                        overlap_cols = max(0, min(ac + aw, bc + bw) - max(ac, bc))
                        if 0 < overlap_rows < min(ah, bh):
                            features["partial_rows"].add(True)
                        if 0 < overlap_cols < min(aw, bw):
                            features["partial_cols"].add(True)
        if len({meta["background"] for meta in all_meta}) > 1:
            within_background += 1
        if len({len(meta["tiles"]) for meta in all_meta}) > 1:
            within_count += 1
        if len({len({color for tile in meta["tiles"] for color in tile[4]})
                for meta in all_meta}) > 1:
            within_palette += 1
    assert features["counts"] == set(range(1, 8))
    assert len(features["sizes"]) >= 12 and (3, 3) in features["sizes"]
    assert len(features["shapes"]) >= 80
    assert len(features["backgrounds"]) == 10
    assert features["palettes"] == set(range(2, 10))
    assert {0, 1, 2} <= features["relations"]
    assert features["edge"] and features["partial_rows"] and features["partial_cols"]
    assert len(features["patterns"]) >= 300
    assert within_background >= 30 and within_count >= 30 and within_palette >= 30
    assert mixed_size_pairs >= 40 and nine_color_pairs >= 1
    assert rectangles >= 100 and other_squares >= 50
    assert unfamiliar_patterns >= 100 and disconnected_colors >= 50
    assert reused_motifs >= 10
    assert full_size >= 1 and wide >= 5 and tall >= 5
    assert crowded >= 1 and sparse >= 10
    assert heldout_counts >= 10 and new_test_colors >= 1 and new_test_pairs >= 10
    assert _puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(generate_puzzle(12345, 7, 3))
    assert _puzzle_key(generate_puzzle(12345, 7, 3)) != _puzzle_key(generate_puzzle(12346, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                           (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
