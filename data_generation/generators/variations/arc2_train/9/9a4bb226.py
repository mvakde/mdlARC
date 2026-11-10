"""Generate puzzles that copy the sole locally tricolored solid tile."""

import numpy as np


_SOURCE_INPUTS = frozenset({
    "000000000000000/000000000000000/000555000000000/000688000000000/000655009990000/000000004490000/000000004990000/000000000000000/000000000000000/000000017100000/000322071700000/000222077700000/000323000000000/000000000000000/000000000000000",
    "000000000000000/000000000000000/009990007440000/008890007470000/008890007440000/000000000000000/000000001330000/000000001220000/000000001320000/003660000000000/003330000000000/006630000000000/000000000000000/000000000000000/000000000000000",
    "000000000000000/007550000000000/005750001620000/005770006110000/000000002610000/000000000000000/000000000000000/008330000000000/003380000000000/008330009440000/000000004940000/000000004490000/000000000000000/000000000000000/000000000000000",
    "000000000000000/003660007170000/003330001770000/006360007110000/000000000000000/000000000000000/000000008850000/000000005550000/009440005580000/004440000000000/002220000000000/000000000000000/000000000000000/000000000000000/000000000000000",
})

_SHAPES = ((3, 3), (3, 3), (3, 3), (4, 4), (5, 5),
           (3, 4), (4, 3), (4, 5), (5, 4), (3, 5), (5, 3))


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    code = seed + seed // 4
    if test is None and train is None:
        test = 1 + seed % 4
    if train is None:
        low = max(2, test + 1)
        train = low + code % (11 - low)
    if test is None:
        test = 1 + code % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid):
    return (grid.shape, np.asarray(grid, dtype=np.uint8).tobytes())


def _grid_text(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in grid)


def _components(grid, background):
    mask = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    found = []
    for raw_r, raw_c in np.argwhere(mask):
        r, c = int(raw_r), int(raw_c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo = [(r, c)]
        cells = []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        found.append(cells)
    return found


def _inspect(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    assert int(counts.max()) > input_grid.size // 2
    objects = []
    for cells in _components(input_grid, background):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows) + 1, min(cols), max(cols) + 1
        tile = input_grid[r0:r1, c0:c1]
        assert len(cells) == tile.size
        assert np.all(tile != background)
        assert tile.shape[0] >= 3 and tile.shape[1] >= 3
        assert len(cells) >= 9
        objects.append({"box": (r0, r1, c0, c1), "tile": tile.copy(),
                        "colors": int(np.unique(tile).size)})
    objects.sort(key=lambda o: o["box"])
    assert 3 <= len(objects) <= 7
    shapes = {o["tile"].shape for o in objects}
    assert len(shapes) == 1
    assert sorted(o["colors"] for o in objects) == [2] * (len(objects) - 1) + [3]
    for i, first in enumerate(objects):
        a, b, c, d = first["box"]
        for second in objects[i + 1:]:
            e, f, g, h = second["box"]
            assert b < e or f < a or d < g or h < c
    target = next(o for o in objects if o["colors"] == 3)
    return background, objects, target


def _quadrant(box, shape):
    r0, r1, c0, c1 = box
    return 2 * int(r0 + r1 >= shape[0]) + int(c0 + c1 >= shape[1])


def _row_banded(tile):
    return sum(np.unique(row).size == 1 for row in tile) >= 2


def _make_mosaic(rng, shape, colors, texture):
    h, w = shape
    n = len(colors)
    if texture == "bands":
        horizontal = bool(rng.integers(0, 2))
        length = h if horizontal else w
        order = rng.permutation(n)
        runs = np.asarray([int(order[i % n]) for i in range(length)])
        if horizontal:
            indices = np.repeat(runs[:, None], w, axis=1)
        else:
            indices = np.repeat(runs[None, :], h, axis=0)
        for _ in range(int(rng.integers(0, 4))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            indices[r, c] = int(rng.integers(n))
    elif texture == "clusters":
        points = [(int(rng.integers(h)), int(rng.integers(w))) for _ in colors]
        indices = np.empty(shape, dtype=np.int8)
        for r in range(h):
            for c in range(w):
                distances = [(r - y) ** 2 + (c - x) ** 2 + float(rng.random()) * 1.5
                             for y, x in points]
                indices[r, c] = int(np.argmin(distances))
    elif texture == "sparse":
        indices = rng.integers(0, max(1, n - 1), size=shape, dtype=np.int8)
        if n == 2:
            indices.fill(0)
        positions = rng.choice(h * w, size=int(rng.integers(1, 3)), replace=False)
        indices.flat[positions] = n - 1
    else:
        weights = rng.dirichlet(np.full(n, float(rng.choice((0.6, 1.3, 3.0)))))
        indices = rng.choice(n, size=shape, p=weights).astype(np.int8)
    missing = set(range(n)) - set(int(x) for x in np.unique(indices))
    if missing:
        chosen = rng.choice(h * w, size=n, replace=False)
        for place, value in zip(chosen, range(n)):
            indices.flat[int(place)] = value
    return np.asarray(colors, dtype=np.int8)[indices]


def _make_row_banded(rng, shape, colors):
    h, w = shape
    indices = rng.integers(0, 3, size=shape, dtype=np.int8)
    rows = rng.choice(h, size=2, replace=False)
    order = rng.permutation(3)
    indices[int(rows[0]), :] = int(order[0])
    indices[int(rows[1]), :] = int(order[1])
    remaining = [r for r in range(h) if r not in rows]
    indices[remaining[0], int(rng.integers(w))] = int(order[2])
    return np.asarray(colors, dtype=np.int8)[indices]


def _palette_pairs(rng, background, count):
    available = [int(x) for x in rng.permutation([c for c in range(10) if c != background])]
    target = tuple(available[:3])
    mode_options = ["independent", "multi_overlap", "repeated"]
    if count <= 4:
        mode_options += ["disjoint", "one_overlap"]
    mode = mode_options[int(rng.integers(len(mode_options)))]
    distractors = []
    if mode == "disjoint":
        for i in range(count - 1):
            distractors.append(tuple(available[3 + 2 * i:5 + 2 * i]))
    elif mode == "one_overlap":
        distractors.append((target[0], available[3]))
        for i in range(count - 2):
            distractors.append(tuple(available[4 + 2 * i:6 + 2 * i]))
    elif mode == "repeated":
        pair = tuple(int(x) for x in rng.choice(available, size=2, replace=False))
        distractors.extend((pair, pair))
        for _ in range(count - 3):
            distractors.append(tuple(int(x) for x in rng.choice(available, size=2, replace=False)))
    elif mode == "multi_overlap":
        for i in range(count - 1):
            shared = target[i % 3]
            other = int(rng.choice([v for v in available if v != shared]))
            distractors.append((shared, other))
    else:
        for _ in range(count - 1):
            distractors.append(tuple(int(x) for x in rng.choice(available, size=2, replace=False)))
    assert all(len(set(pair)) == 2 for pair in distractors)
    return target, distractors


def _canvas_options(tile_shape, count):
    th, tw = tile_shape
    result = []
    for height in range(10, 31):
        for width in range(10, 31):
            rows = (height - 1) // (th + 1)
            cols = (width - 1) // (tw + 1)
            if rows * cols >= count and height * width > 2 * count * th * tw:
                result.append((height, width))
    return result


def _clear(box, boxes):
    r0, r1, c0, c1 = box
    return all(r1 < a or b < r0 or c1 < d or e < c0
               for a, b, d, e in boxes)


def _free_positions(rng, canvas, tile_shape, count):
    height, width = canvas
    th, tw = tile_shape
    candidates = [(r, c) for r in range(1, height - th)
                  for c in range(1, width - tw)]
    for _ in range(24):
        positions = []
        boxes = []
        for index in rng.permutation(len(candidates)):
            r, c = candidates[int(index)]
            box = (r, r + th, c, c + tw)
            if _clear(box, boxes):
                positions.append((r, c))
                boxes.append(box)
                if len(positions) == count:
                    return positions
    return None


def _cluster_positions(rng, canvas, tile_shape, count):
    height, width = canvas
    th, tw = tile_shape
    options = []
    for rows in range(2, count + 1):
        cols = (count + rows - 1) // rows
        if 2 + rows * th + rows - 1 <= height and 2 + cols * tw + cols - 1 <= width:
            options.append((rows, cols))
    if not options:
        return None
    rows, cols = options[int(rng.integers(len(options)))]
    extent_h, extent_w = rows * th + rows - 1, cols * tw + cols - 1
    top = int(rng.integers(1, height - extent_h))
    left = int(rng.integers(1, width - extent_w))
    slots = [(top + r * (th + 1), left + c * (tw + 1))
             for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(slots), size=count, replace=False)
    return [slots[int(i)] for i in chosen]


def _line_positions(rng, canvas, tile_shape, count):
    height, width = canvas
    th, tw = tile_shape
    directions = []
    if count * tw + count + 1 <= width:
        directions.append("horizontal")
    if count * th + count + 1 <= height:
        directions.append("vertical")
    if not directions:
        return None
    direction = directions[int(rng.integers(len(directions)))]
    if direction == "horizontal":
        spare = width - (count * tw + count + 1)
        extras = rng.multinomial(spare, np.full(count + 1, 1 / (count + 1)))
        row = int(rng.integers(1, height - th))
        col = 1 + int(extras[0])
        positions = []
        for i in range(count):
            positions.append((row, col))
            col += tw + 1 + int(extras[i + 1])
    else:
        spare = height - (count * th + count + 1)
        extras = rng.multinomial(spare, np.full(count + 1, 1 / (count + 1)))
        col = int(rng.integers(1, width - tw))
        row = 1 + int(extras[0])
        positions = []
        for i in range(count):
            positions.append((row, col))
            row += th + 1 + int(extras[i + 1])
    return positions


def _positions(rng, canvas, tile_shape, count):
    methods = (_free_positions, _free_positions, _free_positions,
               _cluster_positions, _cluster_positions, _line_positions)
    for _ in range(12):
        method = methods[int(rng.integers(len(methods)))]
        result = method(rng, canvas, tile_shape, count)
        if result is not None:
            return result
    # A compact lattice always fits because _canvas_options checks capacity.
    th, tw = tile_shape
    slots = [(1 + r * (th + 1), 1 + c * (tw + 1))
             for r in range((canvas[0] - 1) // (th + 1))
             for c in range((canvas[1] - 1) // (tw + 1))]
    return [slots[int(i)] for i in rng.choice(len(slots), size=count, replace=False)]


def _generate_pair(rng, allowed_quadrants, forbidden_quadrant, texture_restriction):
    for _ in range(80):
        count = int(rng.integers(3, 8))
        tile_shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
        canvas_options = _canvas_options(tile_shape, count)
        if count == 4 and tile_shape == (3, 3) and rng.random() < 0.5:
            canvas = (15, 15)
        else:
            canvas = canvas_options[int(rng.integers(len(canvas_options)))]
        positions = _positions(rng, canvas, tile_shape, count)
        quadrants = [_quadrant((r, r + tile_shape[0], c, c + tile_shape[1]), canvas)
                     for r, c in positions]
        eligible = [i for i, q in enumerate(quadrants)
                    if q in allowed_quadrants and q != forbidden_quadrant]
        if eligible:
            break
    else:
        raise AssertionError("could not place a target in the requested region")
    target_index = int(rng.choice(eligible))
    background = int(rng.integers(10))
    target_colors, distractor_colors = _palette_pairs(rng, background, count)
    textures = ("scattered", "bands", "clusters", "sparse")
    tiles = [None] * count
    if texture_restriction == "banded":
        tiles[target_index] = _make_row_banded(rng, tile_shape, target_colors)
    else:
        for _ in range(32):
            target_texture = textures[int(rng.choice((0, 2, 3)))] if texture_restriction == "irregular" else textures[int(rng.integers(len(textures)))]
            tile = _make_mosaic(rng, tile_shape, target_colors, target_texture)
            if texture_restriction != "irregular" or not _row_banded(tile):
                break
        else:
            indices = np.fromfunction(lambda r, c: (r + c) % 3, tile_shape, dtype=int)
            tile = np.asarray(target_colors, dtype=np.int8)[indices]
        tiles[target_index] = tile
    for index, colors in zip((i for i in range(count) if i != target_index), distractor_colors):
        texture = textures[int(rng.integers(len(textures)))]
        tiles[index] = _make_mosaic(rng, tile_shape, colors, texture)
    grid = np.full(canvas, background, dtype=np.int8)
    for (r, c), tile in zip(positions, tiles):
        grid[r:r + tile_shape[0], c:c + tile_shape[1]] = tile
    _, _, target = _inspect(grid)
    return {"input": grid, "output": target["tile"].copy()}


def _pair_features(pair):
    background, objects, target = _inspect(pair["input"])
    palettes = [set(int(v) for v in np.unique(o["tile"])) for o in objects]
    overlap = sum(bool(a & b) for i, a in enumerate(palettes)
                  for b in palettes[i + 1:])
    distractor_palettes = [p for p, o in zip(palettes, objects) if o["colors"] == 2]
    repeated = len({tuple(sorted(p)) for p in distractor_palettes}) < len(distractor_palettes)
    return {"background": background, "count": len(objects),
            "canvas": pair["input"].shape, "shape": target["tile"].shape,
            "quadrant": _quadrant(target["box"], pair["input"].shape),
            "row_banded": _row_banded(target["tile"]),
            "profile": tuple(sorted((int(v) for v in np.unique(target["tile"], return_counts=True)[1]), reverse=True)),
            "overlap": overlap, "repeated_palette": repeated,
            "positions": tuple(o["box"] for o in objects)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    train_quadrants = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= value <= 30 for value in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert input_grid.shape != output_grid.shape
            key = _grid_key(input_grid)
            pair_key = (key, _grid_key(output_grid))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _grid_text(input_grid) not in _SOURCE_INPUTS
            background, objects, target = _inspect(input_grid)
            assert np.array_equal(output_grid, target["tile"])
            assert background not in output_grid
            assert all(o["colors"] == 2 for o in objects if o is not target)
            if split == "train":
                train_quadrants.add(_quadrant(target["box"], input_grid.shape))
    assert len(train_quadrants) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # There is no hidden color or orientation convention in this family. These
    # independent schedules control which visible cases are demonstrated.
    position_regime = ("source", "reverse", "shared", "open")[int(rng.integers(4))]
    texture_regime = ("source", "reverse", "shared", "open")[int(rng.integers(4))]
    puzzle = {"train": [], "test": []}
    seen = set()
    train_quadrants = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if position_regime == "source":
                allowed = (0, 1) if split == "train" else (2, 3)
            elif position_regime == "reverse":
                allowed = (2, 3) if split == "train" else (0, 1)
            elif position_regime == "shared" and split == "test":
                allowed = tuple(sorted(train_quadrants))
            else:
                allowed = (0, 1, 2, 3)
            if texture_regime == "source":
                texture = "irregular" if split == "train" else "banded"
            elif texture_regime == "reverse":
                texture = "banded" if split == "train" else "irregular"
            else:
                texture = None
            forbidden = next(iter(train_quadrants)) if split == "train" and index == 1 else None
            for _ in range(80):
                pair = _generate_pair(rng, allowed, forbidden, texture)
                key = _grid_key(pair["input"])
                if key not in seen and _grid_text(pair["input"]) not in _SOURCE_INPUTS:
                    break
            else:
                raise AssertionError("could not make a fresh pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_quadrants.add(_pair_features(pair)["quadrant"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _normalized_pattern(tile):
    mapping = {}
    return tuple(mapping.setdefault(int(value), len(mapping)) for value in tile.flat)


def _predict_from_demonstrations(train_pairs, test_input):
    # This separate eight-neighbor segmentation uses only the demonstrations
    # and the proposed test input, not generation parameters or a test label.
    def crops(grid):
        background = int(np.argmax(np.bincount(grid.ravel().astype(int), minlength=10)))
        remaining = {tuple(map(int, cell)) for cell in np.argwhere(grid != background)}
        result = []
        while remaining:
            start = remaining.pop()
            component = [start]
            pending = [start]
            while pending:
                r, c = pending.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        neighbor = (r + dr, c + dc)
                        if neighbor in remaining:
                            remaining.remove(neighbor)
                            pending.append(neighbor)
                            component.append(neighbor)
            rows, cols = zip(*component)
            result.append(grid[min(rows):max(rows) + 1, min(cols):max(cols) + 1])
        return result

    learned_counts = set()
    for pair in train_pairs:
        candidates = crops(pair["input"])
        matched = [tile for tile in candidates if np.array_equal(tile, pair["output"])]
        assert len(matched) == 1
        count = int(np.unique(matched[0]).size)
        assert sum(np.unique(tile).size == count for tile in candidates) == 1
        learned_counts.add(count)
    assert len(learned_counts) == 1
    count = learned_counts.pop()
    matches = [tile for tile in crops(test_input) if np.unique(tile).size == count]
    assert len(matches) == 1
    return matches[0].copy()


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(301 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(17, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {args}")
    assert _puzzle_key(generate_puzzle(12345, 6, 3)) == _puzzle_key(generate_puzzle(12345, 6, 3))
    assert _puzzle_key(generate_puzzle(23)) != _puzzle_key(generate_puzzle(24))
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, canvases, counts, backgrounds, positions, profiles = set(), set(), set(), set(), set(), set()
    motifs, overlap_counts, repeated, banded, simple = set(), set(), set(), set(), 0
    within_background, within_count, within_shape, within_overlap = 0, 0, 0, 0
    source_position = reverse_position = shared_position = 0
    source_texture = reverse_texture = shared_texture = 0
    lines = profile_531 = disjoint = multi_overlap = combined = source_canvas = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            predicted = _predict_from_demonstrations(puzzle["train"], pair["input"])
            assert np.array_equal(predicted, pair["output"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                background, objects, target = _inspect(pair["input"])
                assert np.array_equal(pair["output"], target["tile"])
                f = _pair_features(pair)
                local.append(f)
                shapes.add(f["shape"])
                canvases.add(f["canvas"])
                counts.add(f["count"])
                backgrounds.add(background)
                positions.add(f["positions"])
                profiles.add(f["profile"])
                motifs.add((f["shape"], _normalized_pattern(target["tile"])))
                overlap_counts.add(f["overlap"])
                repeated.add(f["repeated_palette"])
                banded.add(f["row_banded"])
                simple += f["shape"] == (3, 3) and f["count"] == 4
                source_canvas += f["canvas"] == (15, 15)
                profile_531 += f["shape"] == (3, 3) and f["profile"] == (5, 3, 1)
                disjoint += f["overlap"] == 0
                multi_overlap += f["overlap"] >= 2
                combined += (f["shape"][0] != f["shape"][1] and
                             f["overlap"] > 0 and f["background"] != 0 and f["row_banded"])
                boxes = f["positions"]
                lines += len({box[0] for box in boxes}) == 1 or len({box[2] for box in boxes}) == 1
        train_features = [_pair_features(pair) for pair in puzzle["train"]]
        test_features = [_pair_features(pair) for pair in puzzle["test"]]
        train_quadrants = {f["quadrant"] for f in train_features}
        test_quadrants = {f["quadrant"] for f in test_features}
        if train_quadrants <= {0, 1} and test_quadrants <= {2, 3}:
            source_position += 1
        elif train_quadrants <= {2, 3} and test_quadrants <= {0, 1}:
            reverse_position += 1
        elif test_quadrants <= train_quadrants:
            shared_position += 1
        train_banded = {f["row_banded"] for f in train_features}
        test_banded = {f["row_banded"] for f in test_features}
        if train_banded == {False} and test_banded == {True}:
            source_texture += 1
        elif train_banded == {True} and test_banded == {False}:
            reverse_texture += 1
        elif train_banded & test_banded:
            shared_texture += 1
        within_background += len({f["background"] for f in local}) > 1
        within_count += len({f["count"] for f in local}) > 1
        within_shape += len({f["shape"] for f in local}) > 1
        within_overlap += len({f["overlap"] for f in local}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40
    assert len(shapes) >= 8 and len(canvases) >= 80 and counts == set(range(3, 8))
    assert len(backgrounds) == 10 and len(positions) >= 100 and len(profiles) >= 20
    assert len(motifs) >= 100 and len(overlap_counts) >= 4 and repeated == {False, True}
    assert banded == {False, True} and simple >= 4
    assert min(within_background, within_count, within_shape, within_overlap) >= 24
    assert min(source_position, reverse_position, shared_position) >= 3
    assert min(source_texture, reverse_texture, shared_texture) >= 3
    assert min(lines, profile_531, disjoint, source_canvas) >= 3
    assert multi_overlap >= 30 and combined >= 10
