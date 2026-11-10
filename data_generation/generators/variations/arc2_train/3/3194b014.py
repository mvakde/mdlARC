import numpy as np


_SOURCE_INPUT_STRINGS = (
    "88888010118111000001"
    "10000811000805555581"
    "00000810081115555580"
    "18018008888185555510"
    "08099998000005555580"
    "81899998110110808808"
    "00099990118833811001"
    "81181801000333108188"
    "01881108833333800810"
    "01101000833333118810"
    "80808000033333111088"
    "00008111133333110181"
    "08808818033381100000"
    "00880080100100081110"
    "00101018810080101100"
    "04444484000108080180"
    "14444444011080000818"
    "14444411010110000108"
    "01000181080100800810"
    "80100108011018080010",

    "66677700600600660066"
    "00006000700700676070"
    "60076000070066000676"
    "70722220707707006706"
    "07022227670067607060"
    "70022227000067000000"
    "66622220700999770770"
    "70007007606999706000"
    "76066706067999600000"
    "00776076676999000000"
    "07760000706060070700"
    "06770777000666000066"
    "67707606000006000007"
    "76606706067600670076"
    "60060744444007606000"
    "70700044444707600007"
    "66000044444000000600"
    "67666044444677670006"
    "70006644444606000067"
    "00776006766060607700",

    "03200000203223320000"
    "22002000332203003223"
    "02888888000233026602"
    "38888888300323666662"
    "08888888320239666663"
    "20888888800200666660"
    "02088888822066666662"
    "00238888880306666663"
    "00020020020336660000"
    "00220303300000000300"
    "02322332202202002022"
    "30032030024440002023"
    "00002020044440030202"
    "32000300444443232020"
    "32202000444443203022"
    "23002203044443200032"
    "22030203023222022303"
    "22000003230200020222"
    "00000033233320030233"
    "00320022203002030300",

    "09970097770700709000"
    "00733330770000666609"
    "07933333309970666607"
    "09033333337076666600"
    "90070977077000966677"
    "00970990070090066607"
    "09090070090000099000"
    "09009700097099070000"
    "07888899070097700977"
    "90988877070090090700"
    "00988809090075500909"
    "00988890000955570009"
    "90088807997055550007"
    "99979880970555559070"
    "00770780007555555909"
    "97709079709555555009"
    "07700790709755555979"
    "00777009090055555000"
    "00000790009955509090"
    "00907009700700790000",
)


def _components(mask: np.ndarray) -> list[int]:
    seen = np.zeros(mask.shape, dtype=bool)
    sizes = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        size = 0
        while stack:
            x, y = stack.pop()
            size += 1
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    stack.append((nx, ny))
        sizes.append(size)
    return sizes


def _analyze_input(grid: np.ndarray) -> dict:
    """Classify colors by visible spread, then compare compact object areas."""
    height, width = grid.shape
    broad, objects, speckles = {}, {}, {}
    for color in np.unique(grid):
        color = int(color)
        mask = grid == color
        rr, cc = np.nonzero(mask)
        r0, r1 = int(rr.min()), int(rr.max())
        c0, c1 = int(cc.min()), int(cc.max())
        area = len(rr)
        box_h, box_w = r1 - r0 + 1, c1 - c0 + 1
        components = _components(mask)
        stats = {"area": area, "bbox": (r0, r1, c0, c1),
                 "height": box_h, "width": box_w,
                 "box_area": box_h * box_w, "components": len(components)}
        if (box_h >= 0.70 * height and box_w >= 0.70 * width
                and area >= max(12, int(0.055 * height * width))):
            broad[color] = stats
        elif area <= 2:
            speckles[color] = stats
        else:
            assert area >= 8
            assert len(components) == 1
            assert box_h <= 11 and box_w <= 11
            assert area / (box_h * box_w) >= 0.32
            objects[color] = stats
    assert 2 <= len(broad) <= 4
    assert 2 <= len(objects) <= 5
    ordered = sorted(objects, key=lambda color: objects[color]["area"], reverse=True)
    assert objects[ordered[0]]["area"] > objects[ordered[1]]["area"]
    return {"broad": broad, "objects": objects, "speckles": speckles,
            "winner": ordered[0]}


def _trim_rectangle(rng: np.random.Generator, height: int, width: int,
                    target: int) -> np.ndarray | None:
    shape = np.ones((height, width), dtype=bool)
    if target < 8 or target > height * width:
        return None
    for _ in range(height * width - target):
        candidates = [tuple(map(int, x)) for x in np.argwhere(shape)]
        rng.shuffle(candidates)
        for r, c in candidates:
            shape[r, c] = False
            rr, cc = np.nonzero(shape)
            if (int(rr.min()) == 0 and int(rr.max()) == height - 1
                    and int(cc.min()) == 0 and int(cc.max()) == width - 1
                    and len(_components(shape)) == 1):
                break
            shape[r, c] = True
        else:
            return None
    return shape


def _make_shape(rng: np.random.Generator, height: int, width: int,
                style: int | None = None) -> np.ndarray:
    if style is None:
        style = int(rng.integers(0, 6))
    shape = np.ones((height, width), dtype=bool)
    if style == 1:
        target = height * width - int(rng.integers(1, max(3, height * width // 4)))
        candidate = _trim_rectangle(rng, height, width, target)
        if candidate is not None:
            shape = candidate
    elif style == 2:
        cut = int(rng.integers(1, max(2, width // 2)))
        pivot = int(rng.integers(1, height))
        shape[:pivot, width - cut:] = False
        shape[pivot:, :cut] = False
    elif style == 3:
        pivot = int(rng.integers(1, height - 1))
        for r in range(height):
            span = max(2, width - abs(r - pivot) * int(rng.integers(1, 3)))
            left = (width - span) // 2
            shape[r, :left] = False
            shape[r, left + span:] = False
    elif style == 4:
        top = int(rng.integers(1, max(2, height // 2)))
        bottom = int(rng.integers(1, max(2, height // 2)))
        left_cut = int(rng.integers(1, max(2, width // 2)))
        right_cut = int(rng.integers(1, max(2, width // 2)))
        shape[:top, :left_cut] = False
        shape[height - bottom:, width - right_cut:] = False
    elif style == 5:
        tail = int(rng.integers(1, max(2, height // 2 + 1)))
        span = int(rng.integers(1, min(3, width) + 1))
        left = int(rng.integers(0, width - span + 1))
        shape[height - tail:, :left] = False
        shape[height - tail:, left + span:] = False
    if int(rng.integers(0, 2)):
        shape = shape[::-1, :]
    if int(rng.integers(0, 2)):
        shape = shape[:, ::-1]
    if (int(shape.sum()) < 8 or len(_components(shape)) != 1):
        shape = np.ones((height, width), dtype=bool)
    return shape.copy()


def _random_shape(rng: np.random.Generator, limit: int) -> np.ndarray:
    height = int(rng.integers(3, limit + 1))
    width = int(rng.integers(3, limit + 1))
    return _make_shape(rng, height, width)


def _shapes_for_pair(rng: np.random.Generator, count: int, limit: int,
                     mode: str) -> list[np.ndarray] | None:
    if mode == "natural":
        shapes = [_random_shape(rng, limit) for _ in range(count)]
        shapes.sort(key=lambda shape: int(shape.sum()), reverse=True)
        if int(shapes[0].sum()) == int(shapes[1].sum()):
            return None
        return shapes
    if mode == "easy":
        h, w = int(rng.integers(5, limit + 1)), int(rng.integers(5, limit + 1))
        winner = _make_shape(rng, h, w, int(rng.integers(0, 3)))
        shapes = [winner]
        for _ in range(count - 1):
            for _ in range(30):
                loser = _random_shape(rng, max(4, limit - 2))
                if int(loser.sum()) <= int(winner.sum()) - 5:
                    break
            else:
                loser = np.ones((3, 3), dtype=bool)
            shapes.append(loser)
        return shapes
    h = int(rng.integers(4, max(5, limit)))
    w = int(rng.integers(4, max(5, limit)))
    winner_area = h * w - int(rng.integers(0, 3))
    winner = _trim_rectangle(rng, h, w, winner_area)
    if winner is None:
        return None
    if mode == "inverse":
        lh, lw = h + 1, w + 1
        if lh > limit or lw > limit:
            return None
        difference = int(rng.integers(1, 4))
    elif mode == "equal_box":
        lh, lw = h, w
        difference = int(rng.integers(3, min(9, winner_area - 7)))
    else:  # close, with no fixed bounding-box relation
        lh, lw = h, w
        difference = int(rng.integers(1, 4))
    competitor = _trim_rectangle(rng, lh, lw, winner_area - difference)
    if competitor is None:
        return None
    shapes = [winner, competitor]
    for _ in range(count - 2):
        for _ in range(40):
            loser = _random_shape(rng, limit)
            if int(loser.sum()) < int(winner.sum()) - 3:
                break
        else:
            loser = np.ones((3, 3), dtype=bool)
        shapes.append(loser)
    return shapes


def _positions(rng: np.random.Generator, shapes: list[np.ndarray],
               height: int, width: int, bias: str) -> list[tuple[int, int]] | None:
    placed = []
    for index, shape in enumerate(shapes):
        h, w = shape.shape
        if h > height or w > width:
            return None
        candidates = []
        if index == 0:
            for _ in range(24):
                if bias == "center":
                    r = int(np.clip(height // 2 - h // 2 + rng.integers(-2, 3), 0, height - h))
                    c = int(np.clip(width // 2 - w // 2 + rng.integers(-2, 3), 0, width - w))
                elif bias == "edge":
                    side = int(rng.integers(0, 4))
                    r = (0 if side == 0 else height - h) if side < 2 else int(rng.integers(0, height - h + 1))
                    c = (0 if side == 2 else width - w) if side >= 2 else int(rng.integers(0, width - w + 1))
                elif bias == "lower_right":
                    r = int(rng.integers(max(0, height // 2), height - h + 1)) if height // 2 <= height - h else height - h
                    c = int(rng.integers(max(0, width // 2), width - w + 1)) if width // 2 <= width - w else width - w
                else:
                    r, c = int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1))
                candidates.append((r, c))
        candidates.extend((int(rng.integers(0, height - h + 1)),
                           int(rng.integers(0, width - w + 1))) for _ in range(200))
        for r, c in candidates:
            if all(r + h + 1 <= pr or pr + ph + 1 <= r
                   or c + w + 1 <= pc or pc + pw + 1 <= c
                   for pr, pc, ph, pw in placed):
                placed.append((r, c, h, w))
                break
        else:
            return None
    return [(r, c) for r, c, _, _ in placed]


def _generate_pair(rng: np.random.Generator, palette: tuple[int, ...],
                   regime: str, winner_color: int | None = None,
                   mode: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(80):
        count = int(rng.integers(2, 6))
        if regime == "source_train" and mode == "inverse":
            count = int(rng.integers(3, 5))
        min_dim = 21 if count == 5 else 17
        height = 20 if rng.random() < 0.22 else int(rng.integers(min_dim, 31))
        width = 20 if rng.random() < 0.22 else int(rng.integers(min_dim, 31))
        limit = min(9 if count <= 3 else 8 if count == 4 else 6,
                    max(5, min(height, width) // 2))
        local_mode = mode or str(rng.choice(("natural", "easy", "close", "equal_box", "inverse"),
                                             p=(0.34, 0.17, 0.17, 0.14, 0.18)))
        shapes = _shapes_for_pair(rng, count, limit, local_mode)
        if shapes is None:
            continue
        noise_count = int(rng.integers(2, min(4, len(palette) - count) + 1))
        if winner_color is None:
            colors = [int(x) for x in rng.choice(palette, size=count + noise_count, replace=False)]
        else:
            others = [c for c in palette if c != winner_color]
            colors = [int(winner_color)] + [int(x) for x in rng.choice(others, size=count + noise_count - 1, replace=False)]
        object_colors, noise_colors = colors[:count], colors[count:]
        if regime == "target_test" and rng.random() < 0.23:
            bias = "lower_right"
        else:
            bias = str(rng.choice(("random", "center", "edge"), p=(0.48, 0.28, 0.24)))
        positions = _positions(rng, shapes, height, width, bias)
        if positions is None:
            continue
        weights = rng.uniform(0.7, 1.8, size=noise_count)
        if rng.random() < 0.35:
            weights[int(rng.integers(0, noise_count))] *= 2.0
        weights /= weights.sum()
        for _ in range(10):
            grid = rng.choice(np.array(noise_colors, dtype=np.int8),
                              size=(height, width), p=weights).astype(np.int8)
            for color, shape, (r, c) in zip(object_colors, shapes, positions):
                grid[r:r + shape.shape[0], c:c + shape.shape[1]][shape] = color
            spare = [c for c in palette if c not in colors]
            if spare and rng.random() < 0.16:
                free = np.argwhere(np.isin(grid, noise_colors))
                if len(free):
                    r, c = map(int, free[int(rng.integers(0, len(free)))])
                    grid[r, c] = int(rng.choice(spare))
            try:
                analysis = _analyze_input(grid)
            except AssertionError:
                continue
            if (analysis["winner"] == object_colors[0]
                    and set(analysis["objects"]) == set(object_colors)
                    and set(analysis["broad"]) == set(noise_colors)):
                return grid, np.full((3, 3), object_colors[0], dtype=np.int8)
    # Deterministic construction if random packing is unusually unlucky.
    if winner_color is None:
        colors = [int(x) for x in rng.choice(palette, size=5, replace=False)]
    else:
        colors = [int(winner_color)] + [int(x) for x in rng.choice(
            [c for c in palette if c != winner_color], size=4, replace=False)]
    rr, cc = np.indices((24, 24))
    grid = np.where((rr + cc + int(rng.integers(0, 2))) % 2,
                    colors[3], colors[4]).astype(np.int8)
    dr, dc = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    grid[10 + dr:15 + dr, 9 + dc:15 + dc] = colors[0]
    competitor = np.ones((6, 6), dtype=bool)
    competitor[0, :3] = False
    competitor[1, :2] = False
    competitor[5, 4:] = False
    grid[2 + dr:8 + dr, 14 + dc:20 + dc][competitor] = colors[1]
    grid[16 + dr:19 + dr, 2 + dc:6 + dc] = colors[2]
    analysis = _analyze_input(grid)
    assert analysis["winner"] == colors[0]
    return grid, np.full((3, 3), colors[0], dtype=np.int8)


def _checked_optional_count(name: str, value: int | None,
                            low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train = _checked_optional_count("num_train", num_train, 2, 10)
    num_test = _checked_optional_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # Palette availability is a puzzle-level tendency, not a color-role rule.
    palette_size = int(rng.integers(7, 11))
    palette = tuple(int(x) for x in rng.choice(np.arange(10), size=palette_size, replace=False))
    puzzle = {"train": [], "test": []}
    train_winners = []
    for index in range(num_train):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        preferred = None
        if index == 1:
            preferred = int(pair_rng.choice([c for c in palette if c != train_winners[0]]))
        mode = "inverse" if index == 0 else None
        input_grid, output_grid = _generate_pair(pair_rng, palette, "source_train", preferred, mode)
        puzzle["train"].append({"input": input_grid, "output": output_grid})
        train_winners.append(int(output_grid[0, 0]))

    train_objects = []
    for pair in puzzle["train"]:
        analysis = _analyze_input(pair["input"])
        train_objects.extend(c for c in analysis["objects"] if c != analysis["winner"])
    unused_object_colors = [c for c in palette if c not in train_winners
                            and c not in train_objects]
    for _ in range(num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        role = int(pair_rng.integers(0, 3))
        if role == 0:
            preferred = int(pair_rng.choice(train_winners))
        elif role == 1 and train_objects:
            preferred = int(pair_rng.choice(train_objects))
        elif role == 2 and unused_object_colors:
            preferred = int(pair_rng.choice(unused_object_colors))
        else:
            preferred = None
        input_grid, output_grid = _generate_pair(pair_rng, palette, "target_test", preferred)
        puzzle["test"].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            key = (grid.shape, grid.astype(np.int64).tobytes())
            assert key not in seen
            seen.add(key)
            assert output.shape == (3, 3)
            assert grid.shape != output.shape
            if grid.shape == (20, 20):
                assert "".join(str(int(x)) for x in grid.ravel()) not in _SOURCE_INPUT_STRINGS
            analysis = _analyze_input(grid)
            assert np.all(output == analysis["winner"])
            assert len(analysis["broad"]) + len(analysis["objects"]) + len(analysis["speckles"]) == len(np.unique(grid))
            analyses[split].append(analysis)
    assert len({a["winner"] for a in analyses["train"]}) >= 2
    # A training demonstration requires true cell counts: a larger box loses.
    assert any(any(s["box_area"] > a["objects"][a["winner"]]["box_area"]
                   and s["area"] < a["objects"][a["winner"]]["area"]
                   for c, s in a["objects"].items() if c != a["winner"])
               for a in analyses["train"])
    # The output convention is demonstrated by every training pair. There is
    # no hidden color dictionary: each target color is visible in its input.
    assert all(a["winner"] in a["objects"] for a in analyses["test"])


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s])
        for k in ("input", "output")) for s in ("train", "test"))


def _independent_winner(grid: np.ndarray) -> int:
    areas = {}
    height, width = grid.shape
    for color in np.unique(grid):
        positions = np.argwhere(grid == color)
        extent = positions.max(axis=0) - positions.min(axis=0) + 1
        if len(positions) < 8 or (extent[0] >= 0.7 * height and extent[1] >= 0.7 * width):
            continue
        if len(_components(grid == color)) == 1:
            areas[int(color)] = len(positions)
    ordered = sorted(areas, key=areas.get, reverse=True)
    assert len(ordered) >= 2 and areas[ordered[0]] > areas[ordered[1]]
    return ordered[0]


def self_test() -> None:
    originals = [np.array([[int(x) for x in row[i:i + 20]]
                           for i in range(0, 400, 20)], dtype=np.int8)
                 for row in _SOURCE_INPUT_STRINGS]
    assert [_independent_winner(grid) for grid in originals] == [3, 4, 8, 5]
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    profiles, sizes, counts, colors = set(), set(), set(), set()
    noise_counts = set()
    metrics = {name: 0 for name in (
        "close", "inverse", "equal_box", "sparse", "center", "edge",
        "zero_object", "zero_broad", "zero_winner", "speckle",
        "small_winner", "tall_lower_right_test", "repeat_test_winner",
        "prior_loser_test", "novel_test_winner", "intersection_shortcut",
        "vary_size", "vary_palette", "vary_count", "vary_noise")}
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        within = {"size": set(), "palette": set(), "count": set(), "noise": set()}
        train_winners, train_losers = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                assert np.all(output == _independent_winner(grid))
                analysis = _analyze_input(grid)
                sizes.add(grid.shape)
                counts.add(len(analysis["objects"]))
                noise_counts.add(len(analysis["broad"]))
                colors.add(analysis["winner"])
                within["size"].add(grid.shape)
                within["palette"].add(tuple(sorted(int(c) for c in np.unique(grid))))
                within["count"].add(len(analysis["objects"]))
                within["noise"].add(len(analysis["broad"]))
                features = tuple(sorted((s["height"], s["width"], s["area"])
                                        for s in analysis["objects"].values()))
                profiles.add(features)
                winner = analysis["objects"][analysis["winner"]]
                losers = [s for c, s in analysis["objects"].items() if c != analysis["winner"]]
                bbox = winner["bbox"]
                height, width = grid.shape
                metrics["close"] += any(1 <= winner["area"] - s["area"] <= 3 for s in losers)
                metrics["inverse"] += any(s["box_area"] > winner["box_area"] for s in losers)
                metrics["equal_box"] += any(s["height"] == winner["height"] and s["width"] == winner["width"] for s in losers)
                metrics["sparse"] += any(s["area"] / s["box_area"] < .65 for s in analysis["objects"].values())
                metrics["center"] += bbox[0] < height // 2 <= bbox[1] and bbox[2] < width // 2 <= bbox[3]
                metrics["edge"] += any(s["bbox"][0] == 0 or s["bbox"][1] == height - 1
                                       or s["bbox"][2] == 0 or s["bbox"][3] == width - 1
                                       for s in analysis["objects"].values())
                metrics["zero_object"] += 0 in analysis["objects"]
                metrics["zero_broad"] += 0 in analysis["broad"]
                metrics["zero_winner"] += analysis["winner"] == 0
                metrics["speckle"] += bool(analysis["speckles"])
                metrics["small_winner"] += max(winner["height"], winner["width"]) <= 5
                if split == "train":
                    train_winners.append(analysis["winner"])
                    train_losers.append(set(analysis["objects"]) - {analysis["winner"]})
                else:
                    metrics["tall_lower_right_test"] += (winner["height"] > winner["width"]
                        and bbox[0] >= height // 2 and bbox[2] >= width // 2)
                    metrics["repeat_test_winner"] += analysis["winner"] in train_winners
                    loser_union = set().union(*train_losers)
                    metrics["prior_loser_test"] += analysis["winner"] in loser_union
                    metrics["novel_test_winner"] += analysis["winner"] not in loser_union and analysis["winner"] not in train_winners
                    metrics["intersection_shortcut"] += set.intersection(*train_losers) == {analysis["winner"]}
        for key, values in within.items():
            metrics["vary_" + key] += len(values) > 1
    assert len(profiles) >= 80 and len(sizes) >= 20
    assert counts == {2, 3, 4, 5} and len(colors) >= 8
    assert noise_counts == {2, 3, 4}
    for name in ("close", "inverse", "equal_box", "sparse", "center", "edge",
                 "zero_object", "zero_broad", "zero_winner", "speckle",
                 "small_winner", "tall_lower_right_test", "repeat_test_winner",
                 "prior_loser_test", "novel_test_winner"):
        assert metrics[name] > 0, name
    assert metrics["close"] >= 20 and metrics["inverse"] >= 20
    assert metrics["equal_box"] >= 10 and metrics["sparse"] >= 10
    assert all(metrics["vary_" + name] >= 35 for name in ("size", "palette", "count", "noise"))
    assert metrics["intersection_shortcut"] < sum(len(p["test"]) for p in automatic) // 4
    for seed, nt, nv in ((91, 2, 1), (92, 10, 1), (93, 10, 4), (94, 7, 3),
                         (101, None, 1), (102, None, 4), (103, 2, None), (104, 10, None)):
        p = generate_puzzle(seed, nt, nv)
        if nt is not None:
            assert len(p["train"]) == nt
        if nv is not None:
            assert len(p["test"]) == nv
        validate_generated_puzzle(p)
    assert _puzzles_equal(generate_puzzle(777), generate_puzzle(777))
    assert _puzzles_equal(generate_puzzle(778, 8, 3), generate_puzzle(778, 8, 3))
    assert not _puzzles_equal(generate_puzzle(779, 8, 3), generate_puzzle(780, 8, 3))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
