import numpy as np


# Exact source inputs are excluded by content, without excluding their colors
# or their 15 by 15 canvas.
_SOURCE_INPUTS = {
    (
        "888888888888888", "888888886668888", "888888886668888",
        "886666886668888", "886866888888888", "886866888888888",
        "886666888888888", "888888888888888", "888888886666668",
        "888888886888868", "888888886888868", "888888886888868",
        "888888886888868", "888888886666668", "888888888888888",
    ),
    (
        "888888888888888", "888888888888888", "888888886666888",
        "888888886886888", "888668886886888", "888668886886888",
        "888888886666888", "888888888888888", "888888888888888",
        "888888888888888", "888886666888888", "888886666888888",
        "888886666888888", "888886666888888", "888888888888888",
    ),
    (
        "888888888888888", "888888888888888", "888888888666888",
        "886666888666888", "886886888686888", "886886888686888",
        "886666888686888", "888888888666888", "888888888888888",
        "888888888888888", "888888888888888", "888866666668888",
        "888866886668888", "888866666668888", "888888888888888",
    ),
}


def _components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) != color:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and not seen[nr, nc] and grid[nr, nc] == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(component)
    return result


def _bbox(component: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*component)
    return min(rows), max(rows), min(cols), max(cols)


def _render(input_grid: np.ndarray, colors: tuple[int, int, int, int]) -> np.ndarray:
    background, body, fill, outline = colors
    output = input_grid.copy()
    boxes = [_bbox(part) for part in _components(input_grid, body)]
    for r0, r1, c0, c1 in boxes:
        block = output[r0:r1 + 1, c0:c1 + 1]
        block[block == background] = fill
        output[r0 - 1, c0 - 1:c1 + 2] = outline
        output[r1 + 1, c0 - 1:c1 + 2] = outline
        output[r0 - 1:r1 + 2, c0 - 1] = outline
        output[r0 - 1:r1 + 2, c1 + 1] = outline
    return output


def _checked_count(value: int, low: int, high: int, name: str) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def _dimensions(rng: np.random.Generator, first_train: bool) -> tuple[int, int]:
    low = 13 if first_train else 11
    height = int(rng.integers(low, 31))
    width = height if rng.random() < 0.18 else int(rng.integers(low, 31))
    return height, width


def _object_count(rng: np.random.Generator, height: int, width: int,
                  regime: str, first_train: bool) -> int:
    ceiling = min(5, max(2, height * width // 75))
    floor = 2 if first_train else 1
    if regime == "target_test" and ceiling >= 3 and rng.random() < 0.55:
        floor = 3
    return int(rng.integers(floor, ceiling + 1))


def _shape(rng: np.random.Generator, height: int, width: int,
           count: int, regime: str, holdout: bool,
           forced: bool) -> tuple[int, int]:
    cap = 6 if count >= 4 else (8 if count == 3 else 12)
    max_h = min(height - 2, cap)
    max_w = min(width - 2, cap)
    if forced:
        vertical = bool(rng.integers(2))
        long_max = max_h if vertical else max_w
        long = int(rng.integers(7, long_max + 1))
        short = 3 if long == 7 else int(rng.integers(3, 5))
        return (long, short) if vertical else (short, long)
    extreme_weight = 0.38 if regime == "target_test" else 0.17
    roll = rng.random()
    if roll < 0.28:
        side = int(rng.integers(3, min(max_h, max_w) + 1))
        return side, side
    if roll < 0.28 + extreme_weight and not (
            holdout and regime == "source_train"):
        long_vertical = bool(rng.integers(2))
        long_max = max_h if long_vertical else max_w
        short_max = max_w if long_vertical else max_h
        if long_max >= 7:
            long = int(rng.integers(7, long_max + 1))
            short = int(rng.integers(3, min(5, short_max) + 1))
            return (long, short) if long_vertical else (short, long)
    for _ in range(16):
        h = int(rng.integers(3, max_h + 1))
        w = int(rng.integers(3, max_w + 1))
        if not (holdout and regime == "source_train"
                and max(h, w) >= 7 and max(h, w) >= 2 * min(h, w)):
            return h, w
    return 4, 4


def _make_tile(rng: np.random.Generator, height: int, width: int,
               hollow: bool, background: int, body: int,
               regime: str) -> np.ndarray:
    tile = np.full((height, width), body, dtype=np.int8)
    if not hollow:
        return tile
    inner_h, inner_w = height - 2, width - 2
    mode = rng.random()
    if mode < (0.32 if regime == "target_test" else 0.22) and (
            inner_h >= 2 and inner_w >= 5 or inner_w >= 2 and inner_h >= 5):
        # Two separated openings, each at least two cells when space permits.
        if inner_w >= 5 and (inner_w >= inner_h or inner_h < 5):
            split = int(rng.integers(2, inner_w - 2))
            left_w = int(rng.integers(2, split + 1))
            right_w = int(rng.integers(2, inner_w - split))
            h1 = int(rng.integers(2, inner_h + 1))
            h2 = int(rng.integers(2, inner_h + 1))
            r1 = int(rng.integers(1, height - h1))
            r2 = int(rng.integers(1, height - h2))
            tile[r1:r1 + h1, 1:1 + left_w] = background
            tile[r2:r2 + h2, width - 1 - right_w:width - 1] = background
        else:
            transpose = _make_tile(rng, width, height, True, background,
                                   body, regime).T
            return transpose
    elif mode < (0.63 if regime == "target_test" else 0.47) and (
            inner_h >= 3 and inner_w >= 3):
        # A rectangle with a shorter offset extension forms an L or step.
        hole_h = int(rng.integers(2, inner_h + 1))
        hole_w = int(rng.integers(2, inner_w + 1))
        r = int(rng.integers(1, height - hole_h))
        c = int(rng.integers(1, width - hole_w))
        tile[r:r + hole_h, c:c + hole_w] = background
        if c + hole_w < width - 1:
            tile[r:r + max(1, hole_h // 2), c + hole_w] = background
        elif c > 1:
            tile[r + hole_h // 2:r + hole_h, c - 1] = background
        elif r + hole_h < height - 1:
            tile[r + hole_h, c:c + max(1, hole_w // 2)] = background
        elif r > 1:
            tile[r - 1, c + hole_w // 2:c + hole_w] = background
    else:
        # Full open centers and tiny gaps are both common.
        broad = rng.random() < 0.48
        min_h = max(1, (inner_h + 1) // 2) if broad else 1
        min_w = max(1, (inner_w + 1) // 2) if broad else 1
        hole_h = int(rng.integers(min_h, inner_h + 1))
        hole_w = int(rng.integers(min_w, inner_w + 1))
        r = int(rng.integers(1, height - hole_h))
        c = int(rng.integers(1, width - hole_w))
        tile[r:r + hole_h, c:c + hole_w] = background
    assert len(_components(tile, body)) == 1
    return tile


def _place(rng: np.random.Generator, specs: list[tuple[int, int, bool]],
           height: int, width: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(specs)), key=lambda i: specs[i][0] * specs[i][1],
                   reverse=True)
    for _ in range(5):
        occupied = np.zeros((height, width), dtype=bool)
        positions = [(-1, -1)] * len(specs)
        for i in order:
            h, w, _ = specs[i]
            if h + 2 > height or w + 2 > width:
                break
            candidates = rng.permutation((height - h - 1) * (width - w - 1))
            found = False
            for value in candidates:
                r, c = divmod(int(value), width - w - 1)
                frame = occupied[r:r + h + 2, c:c + w + 2]
                if not np.any(frame):
                    frame[:] = True
                    positions[i] = (r, c)
                    found = True
                    break
            if not found:
                break
        else:
            return positions
    return None


def _pair(rng: np.random.Generator, regime: str,
          colors: tuple[int, int, int, int], first_train: bool,
          holdout: bool, first_test: bool) -> dict:
    background, body, _, _ = colors
    for _ in range(70):
        height, width = _dimensions(rng, first_train)
        if holdout and first_test:
            height, width = max(16, height), max(16, width)
        count = _object_count(rng, height, width, regime, first_train)
        if holdout and first_test:
            count = min(count, 3)
        if regime == "target_test" and rng.random() < 0.14:
            hollow_flags = [False] * count
        else:
            hollow_flags = [rng.random() < 0.58 for _ in range(count)]
        if first_train:
            hollow_flags[0], hollow_flags[1] = True, False
        specs = [(*_shape(rng, height, width, count, regime, holdout,
                          holdout and first_test and i == 0), flag)
                 for i, flag in enumerate(hollow_flags)]
        positions = _place(rng, specs, height, width)
        if positions is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for (h, w, hollow), (r, c) in zip(specs, positions):
            grid[r + 1:r + h + 1, c + 1:c + w + 1] = _make_tile(
                rng, h, w, hollow, background, body, regime)
        return {"input": grid, "output": _render(grid, colors)}
    # A deterministic geometric fallback keeps even unlucky legal seeds bounded.
    height, width = (14, 17) if first_train else (
        (16, 16) if holdout and first_test else (12, 13))
    specs = [(4, 4, True), (3, 5, False)] if first_train else (
        [(8, 3, True)] if holdout and first_test else [(5, 4, True)])
    positions = [(1, 1), (7, 9)] if first_train else [(3, 4)]
    grid = np.full((height, width), background, dtype=np.int8)
    for (h, w, hollow), (r, c) in zip(specs, positions):
        grid[r + 1:r + h + 1, c + 1:c + w + 1] = _make_tile(
            rng, h, w, hollow, background, body, regime)
    return {"input": grid, "output": _render(grid, colors)}


def _signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _source_copy(grid: np.ndarray) -> bool:
    if grid.shape != (15, 15):
        return False
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is not None:
        num_train = _checked_count(num_train, 2, 10, "num_train")
    if num_test is not None:
        num_test = _checked_count(num_test, 1, 4, "num_test")
    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    palette = [int(v) for v in rng.permutation(10)]
    fill, outline = palette[:2]
    input_colors = palette[2:]
    fixed = rng.random() < 0.55
    holdout_extremes = rng.random() < 0.30
    base = tuple(input_colors[:2])
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, regime, count in (
            ("train", "source_train", num_train),
            ("test", "target_test", num_test)):
        for index in range(count):
            if fixed or (section == "train" and index == 0):
                background, body = base
            elif section == "train" and index == 1:
                background, body = input_colors[2:4]
            else:
                selected = rng.choice(input_colors, size=2, replace=False)
                background, body = int(selected[0]), int(selected[1])
            colors = (background, body, fill, outline)
            for _ in range(12):
                pair = _pair(
                    rng, regime, colors, section == "train" and index == 0,
                    holdout_extremes, section == "test" and index == 0)
                signature = _signature(pair["input"])
                if signature not in seen and not _source_copy(pair["input"]):
                    seen.add(signature)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair: dict) -> tuple[tuple[int, int], int, list[dict]]:
    source, target = pair["input"], pair["output"]
    height, width = source.shape
    background = int(source[0, 0])
    assert np.all(source[0, :] == background)
    assert np.all(source[-1, :] == background)
    assert np.all(source[:, 0] == background)
    assert np.all(source[:, -1] == background)
    input_colors = {int(v) for v in np.unique(source)}
    assert len(input_colors) == 2
    body = next(iter(input_colors - {background}))
    components = _components(source, body)
    assert 1 <= len(components) <= 5
    occupied = np.zeros(source.shape, dtype=bool)
    infos = []
    outline = None
    for component in components:
        r0, r1, c0, c1 = _bbox(component)
        assert r0 >= 1 and c0 >= 1 and r1 < height - 1 and c1 < width - 1
        frame = occupied[r0 - 1:r1 + 2, c0 - 1:c1 + 2]
        assert not np.any(frame)
        frame[:] = True
        block = source[r0:r1 + 1, c0:c1 + 1]
        assert np.all(block[0, :] == body) and np.all(block[-1, :] == body)
        assert np.all(block[:, 0] == body) and np.all(block[:, -1] == body)
        assert len(component) == int(np.count_nonzero(block == body))
        border_color = int(target[r0 - 1, c0 - 1])
        if outline is None:
            outline = border_color
        assert border_color == outline
        holes = np.argwhere(block == background)
        infos.append({
            "box": (r0, r1, c0, c1),
            "h": r1 - r0 + 1, "w": c1 - c0 + 1,
            "holes": int(len(holes)),
            "hole_cell": (r0 + int(holes[0, 0]), c0 + int(holes[0, 1]))
            if len(holes) else None,
        })
    return (background, body), outline, infos


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    observations = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            source, target = pair["input"], pair["output"]
            assert source.shape == target.shape and not np.array_equal(source, target)
            key = _signature(source)
            assert key not in seen and not _source_copy(source)
            seen.add(key)
            input_roles, outline, infos = _inspect(pair)
            observations.append((section, pair, input_roles, outline, infos))

    train = [row for row in observations if row[0] == "train"]
    assert any(info["holes"] for row in train for info in row[4])
    assert any(not info["holes"] for row in train for info in row[4])
    fill_cells = [int(row[1]["output"][info["hole_cell"]])
                  for row in train for info in row[4] if info["holes"]]
    assert len(set(fill_cells)) == 1
    fill = fill_cells[0]
    outlines = {row[3] for row in observations}
    assert len(outlines) == 1
    outline = next(iter(outlines))
    assert fill != outline
    train_palettes = {row[2] for row in train}
    test_palettes = {row[2] for row in observations if row[0] == "test"}
    if not test_palettes.issubset(train_palettes):
        assert len(train_palettes) >= 2
    for _, pair, (background, body), local_outline, infos in observations:
        assert len({background, body, fill, outline}) == 4
        assert local_outline == outline
        source, target = pair["input"], pair["output"]
        expected = source.copy()
        for info in infos:
            r0, r1, c0, c1 = info["box"]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if source[r, c] == background:
                        expected[r, c] = fill
            for c in range(c0 - 1, c1 + 2):
                expected[r0 - 1, c] = outline
                expected[r1 + 1, c] = outline
            for r in range(r0 - 1, r1 + 2):
                expected[r, c0 - 1] = outline
                expected[r, c1 + 1] = outline
        assert np.array_equal(target, expected)


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_signature(p["input"]), _signature(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    sizes, object_counts, shapes, layouts = set(), set(), set(), set()
    varied_sizes = varied_palettes = varied_counts = 0
    solid_tests = mixed_tests = hollow_tests = extreme_holdouts = 0
    irregular = large_holes = multi_gaps = multi_substantial = 0
    multi_substantial_objects = 0
    edge_pairs = interior_pairs = abutting_pairs = square_canvases = 0
    oblong_canvases = source_palette_pairs = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        observations = []
        train_extreme = test_extreme = False
        for section in ("train", "test"):
            for pair in puzzle[section]:
                roles, _, infos = _inspect(pair)
                observations.append((section, pair, roles, infos))
                height, width = pair["input"].shape
                square_canvases += height == width
                oblong_canvases += max(height, width) * 2 >= 3 * min(height, width)
                source_palette_pairs += roles == (8, 6)
                sizes.add(pair["input"].shape)
                object_counts.add(len(infos))
                shapes.update((info["h"], info["w"], bool(info["holes"]))
                              for info in infos)
                layouts.add((pair["input"].shape, tuple(sorted(
                    (i["box"], i["holes"]) for i in infos))))
                large_holes += any(i["holes"] >= 9 for i in infos)
                frames = []
                substantial = 0
                for info in infos:
                    r0, r1, c0, c1 = info["box"]
                    frames.append((r0 - 1, r1 + 1, c0 - 1, c1 + 1))
                    if max(info["h"], info["w"]) >= 7 and (
                            max(info["h"], info["w"]) >=
                            2 * min(info["h"], info["w"])):
                        if section == "train":
                            train_extreme = True
                        else:
                            test_extreme = True
                    mask = pair["input"][r0:r1 + 1, c0:c1 + 1] == roles[0]
                    cavities = _components(mask.astype(np.int8), 1)
                    multi_gaps += len(cavities) >= 2
                    object_substantial = sum(len(cavity) >= 4
                                             for cavity in cavities)
                    substantial += object_substantial
                    multi_substantial_objects += object_substantial >= 2
                    if np.any(mask):
                        rr, cc = np.where(mask)
                        irregular += int(int(mask.sum()) !=
                                         (int(rr.max() - rr.min() + 1) *
                                          int(cc.max() - cc.min() + 1)))
                multi_substantial += substantial >= 2
                edge = any(a == 0 or c == 0 or b == height - 1 or d == width - 1
                           for a, b, c, d in frames)
                edge_pairs += edge
                interior_pairs += not edge
                abutting_pairs += any(
                    (((a[1] + 1 == b[0] or b[1] + 1 == a[0]) and
                      max(a[2], b[2]) <= min(a[3], b[3])) or
                     ((a[3] + 1 == b[2] or b[3] + 1 == a[2]) and
                      max(a[0], b[0]) <= min(a[1], b[1])))
                    for i, a in enumerate(frames) for b in frames[i + 1:])
        varied_sizes += len({x[1]["input"].shape for x in observations}) > 1
        varied_palettes += len({x[2] for x in observations}) > 1
        varied_counts += len({len(x[3]) for x in observations}) > 1
        solid_tests += any(all(not i["holes"] for i in x[3])
                           for x in observations if x[0] == "test")
        hollow_tests += any(all(i["holes"] for i in x[3])
                            for x in observations if x[0] == "test")
        mixed_tests += any(any(i["holes"] for i in x[3])
                           and any(not i["holes"] for i in x[3])
                           for x in observations if x[0] == "test")
        extreme_holdouts += test_extreme and not train_extreme
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(sizes) >= 45
    assert object_counts == {1, 2, 3, 4, 5}
    assert len(shapes) >= 80 and len(layouts) >= 200
    assert varied_sizes >= 35 and varied_palettes >= 8 and varied_counts >= 35
    assert solid_tests >= 2 and mixed_tests >= 5 and hollow_tests >= 2
    assert extreme_holdouts >= 5
    assert irregular >= 10 and large_holes >= 20
    assert multi_gaps >= 10 and multi_substantial >= 10
    assert multi_substantial_objects >= 5
    assert edge_pairs >= 20 and interior_pairs >= 20 and abutting_pairs >= 5
    assert square_canvases >= 15 and oblong_canvases >= 30
    assert source_palette_pairs >= 1
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(500 + train_count * 10 + test_count,
                                 train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (
            train_count, test_count)
    for index, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(700 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4)):
        try:
            generate_puzzle(900, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {args}")
    for args in ((None, None), (7, 3)):
        assert _puzzle_signature(generate_puzzle(12345, *args)) == (
            _puzzle_signature(generate_puzzle(12345, *args)))
    assert _puzzle_signature(generate_puzzle(12345)) != (
        _puzzle_signature(generate_puzzle(12346)))
