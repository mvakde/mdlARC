import numpy as np


# Exact source pairs are excluded; their dimensions and layout categories remain legal.
_SOURCE_PAIRS = {
    (("005005000", "005555000", "555000000", "050000000", "050005555", "055555000", "550005555", "000005000", "555555000", "000005000"),
     ("005775888", "005555888", "555888888", "058888888", "058885555", "055555000", "550005555", "000005000", "555555000", "000005000")),
    (("50050", "50050", "50555", "55500", "00500", "00555", "00050", "55550", "05000", "05000", "05550", "00050", "05555", "55000", "05000"),
     ("50050", "50050", "50555", "55500", "00500", "00555", "00058", "55558", "05888", "05888", "05558", "00058", "05555", "55000", "75000")),
    (("050050000500", "050055500500", "055550500500", "005000555500", "555000500555", "000000500000", "000555500000", "555500500000", "000500555000", "000500005000"),
     ("057750000500", "057755500500", "055550500500", "005000555500", "555000588555", "000000588888", "000555588888", "555500588888", "000500555888", "000500005888")),
    (("050000050050", "050005550550", "555005055500", "005555050550", "000500050050"),
     ("758888850050", "758885550550", "555885055500", "005555050550", "000500050050")),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.astype(np.uint8, copy=False).tobytes())


def _components(mask: np.ndarray) -> list[frozenset[tuple[int, int]]]:
    coordinates = list(zip(*np.nonzero(mask)))
    remaining = set(coordinates)
    found = []
    for start in coordinates:
        if start not in remaining:
            continue
        remaining.remove(start)
        stack = [start]
        cells = {start}
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
        found.append(frozenset(cells))
    return found


def _aspect(shape: tuple[int, int]) -> str:
    h, w = shape
    if h >= w + 3:
        return "tall"
    if w >= h + 3:
        return "wide"
    return "balanced"


def _zone(cells: frozenset[tuple[int, int]], shape: tuple[int, int]) -> str:
    h, w = shape
    edges = set()
    for r, c in cells:
        if r == 0:
            edges.add("top")
        if r == h - 1:
            edges.add("bottom")
        if c == 0:
            edges.add("left")
        if c == w - 1:
            edges.add("right")
    if not edges:
        return "interior"
    if ("top" in edges or "bottom" in edges) and ("left" in edges or "right" in edges):
        return "corner"
    return "edge"


def _dimensions(rng: np.random.Generator, profile: str) -> tuple[int, int]:
    if profile == "shallow":
        return int(rng.integers(5, 8)), int(rng.integers(11, 21))
    if profile == "tall":
        width = int(rng.integers(5, 12))
        return int(rng.integers(max(10, width + 4), 23)), width
    if profile == "wide":
        height = int(rng.integers(6, 13))
        return height, int(rng.integers(max(11, height + 4), 23))
    height = int(rng.integers(8, 18))
    width = int(np.clip(height + int(rng.integers(-2, 3)), 7, 20))
    return height, width


def _first_stroke(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    wall = np.zeros((height, width), dtype=bool)
    vertical = bool(rng.integers(0, 2))
    length = height if vertical else width
    breadth = width if vertical else height
    position = int(rng.integers(2, breadth - 2))
    for step in range(length):
        if step and rng.random() < 0.28:
            proposed = position + int(rng.choice((-1, 1)))
            proposed = int(np.clip(proposed, 2, breadth - 3))
        else:
            proposed = position
        lo, hi = sorted((position, proposed))
        for across in range(lo, hi + 1):
            if vertical:
                wall[step, across] = True
            else:
                wall[across, step] = True
        position = proposed
    return wall


def _runs(component: frozenset[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[bool, int, int, int]]:
    height, width = shape
    runs = []
    for horizontal in (True, False):
        outer = height if horizontal else width
        inner = width if horizontal else height
        for fixed in range(outer):
            start = None
            for pos in range(inner + 1):
                cell = (fixed, pos) if horizontal else (pos, fixed)
                present = pos < inner and cell in component
                if present and start is None:
                    start = pos
                elif not present and start is not None:
                    if pos - start >= 2:
                        runs.append((horizontal, fixed, start, pos - 1))
                    start = None
    return runs


def _path_for_run(
    rng: np.random.Generator,
    run: tuple[bool, int, int, int],
    component: frozenset[tuple[int, int]],
) -> set[tuple[int, int]]:
    horizontal, fixed, start, end = run
    path = {((fixed, p) if horizontal else (p, fixed)) for p in range(start, end + 1)}
    if end - start < 5 or rng.random() >= 0.55:
        return path
    a = int(rng.integers(start + 1, end - 2))
    b = int(rng.integers(a + 2, end))
    shift = int(rng.choice((-2, -1, 1, 2)))
    jog = set()
    for p in range(a, b + 1):
        jog.add((fixed + shift, p) if horizontal else (p, fixed + shift))
    for offset in range(min(0, shift), max(0, shift) + 1):
        jog.add((fixed + offset, a) if horizontal else (a, fixed + offset))
        jog.add((fixed + offset, b) if horizontal else (b, fixed + offset))
    if not jog <= component:
        return path
    for p in range(a + 1, b):
        path.remove((fixed, p) if horizontal else (p, fixed))
    return path | jog


def _try_split(
    rng: np.random.Generator,
    wall: np.ndarray,
    component: frozenset[tuple[int, int]],
    component_count: int,
    minimum: int,
) -> np.ndarray | None:
    runs = _runs(component, wall.shape)
    if not runs:
        return None
    for index in rng.permutation(len(runs))[:min(45, len(runs))]:
        cells = _path_for_run(rng, runs[int(index)], component)
        if not any(
            0 <= rr < wall.shape[0] and 0 <= cc < wall.shape[1] and wall[rr, cc]
            for r, c in cells
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            continue
        trial = wall.copy()
        for r, c in cells:
            trial[r, c] = True
        if np.mean(trial) > 0.46:
            continue
        regions = _components(~trial)
        if len(regions) != component_count + 1:
            continue
        if min(map(len, regions)) < minimum:
            continue
        return trial
    return None


def _partition(
    rng: np.random.Generator,
    shape: tuple[int, int],
    target_regions: int,
    minimum: int,
) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]]] | None:
    wall = _first_stroke(rng, *shape)
    for _ in range(target_regions - 2):
        regions = _components(~wall)
        order = sorted(range(len(regions)), key=lambda i: len(regions[i]), reverse=True)
        if len(order) > 1 and rng.random() < 0.45:
            rng.shuffle(order)
        result = None
        for index in order:
            if len(regions[index]) < max(2 * minimum + 2, 8):
                continue
            result = _try_split(rng, wall, regions[index], len(regions), minimum)
            if result is not None:
                break
        if result is None:
            return None
        wall = result
    return wall, _components(~wall)


def _pair_features(wall: np.ndarray, regions: list[frozenset[tuple[int, int]]]) -> dict:
    areas = [len(region) for region in regions]
    small = regions[areas.index(min(areas))]
    large = regions[areas.index(max(areas))]
    junctions = 0
    h, w = wall.shape
    for r, c in zip(*np.nonzero(wall)):
        neighbors = ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if sum(0 <= rr < h and 0 <= cc < w and wall[rr, cc] for rr, cc in neighbors) >= 3:
            junctions += 1
    return {
        "areas": areas,
        "small": small,
        "large": large,
        "small_zone": _zone(small, wall.shape),
        "large_zone": _zone(large, wall.shape),
        "largest_fraction": max(areas) / (h * w),
        "junctions": junctions,
        "wall_components": len(_components(wall)),
    }


def _fallback_partition(
    rng: np.random.Generator, shape: tuple[int, int]
) -> tuple[np.ndarray, list[frozenset[tuple[int, int]]]]:
    height, width = shape
    for col in rng.permutation(np.arange(1, width - 1)):
        for left_row in rng.permutation(np.arange(1, height - 1)):
            for right_row in rng.permutation(np.arange(1, height - 1)):
                if left_row == right_row:
                    continue
                areas = [
                    int(col * left_row),
                    int(col * (height - left_row - 1)),
                    int((width - col - 1) * right_row),
                    int((width - col - 1) * (height - right_row - 1)),
                ]
                if areas.count(min(areas)) != 1 or areas.count(max(areas)) != 1:
                    continue
                if max(areas) / (height * width) > .52:
                    continue
                wall = np.zeros(shape, dtype=bool)
                wall[:, col] = True
                wall[left_row, :col] = True
                wall[right_row, col + 1:] = True
                if np.mean(wall) <= .46:
                    return wall, _components(~wall)
    raise RuntimeError("no valid fallback partition")


def _make_pair(
    rng: np.random.Generator,
    colors: tuple[int, int, int, int],
    profile: str,
    desired_regions: int,
    minimum_target: int,
    prefer: str,
) -> dict:
    best = None
    best_score = -1e9
    valid_candidates = 0
    for _ in range(12):
        shape = _dimensions(rng, profile)
        target = min(desired_regions, max(4, shape[0] * shape[1] // 8))
        minimum = min(minimum_target, max(1, shape[0] * shape[1] // (target * 3)))
        result = _partition(rng, shape, target, minimum)
        if result is None:
            continue
        wall, regions = result
        areas = [len(region) for region in regions]
        if len(regions) < 4 or areas.count(min(areas)) != 1 or areas.count(max(areas)) != 1:
            continue
        features = _pair_features(wall, regions)
        if features["largest_fraction"] > 0.52:
            continue
        valid_candidates += 1
        score = float(rng.random())
        if prefer == "interior_small":
            score += 3 * (features["small_zone"] == "interior")
        elif prefer == "interior_large":
            score += 3 * (features["large_zone"] == "interior")
        elif prefer == "close":
            sorted_areas = sorted(areas)
            score += 2 * (sorted_areas[-1] - sorted_areas[-2] <= 5)
            score += 1 * (sorted_areas[1] - sorted_areas[0] <= 3)
        elif prefer == "corner_small":
            score += 2 * (features["small_zone"] == "corner")
        score += 0.5 * (features["junctions"] >= 3)
        if score > best_score:
            best = (wall, regions, features)
            best_score = score
        if prefer == "none" or score >= 3.5 or valid_candidates >= 3:
            break
    if best is None:
        wall, regions = _fallback_partition(rng, _dimensions(rng, profile))
        best = (wall, regions, _pair_features(wall, regions))
    wall, regions, features = best
    background, wall_color, small_color, large_color = colors
    inp = np.where(wall, wall_color, background).astype(np.int8)
    out = inp.copy()
    for r, c in features["small"]:
        out[r, c] = small_color
    for r, c in features["large"]:
        out[r, c] = large_color
    return {"input": inp, "output": out}


def _check_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _counts(rng: np.random.Generator, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _check_count(num_train, "num_train", 2, 10)
    test = _check_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _analyze_pair(pair: dict) -> tuple[tuple[int, int, int, int], dict]:
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "input/output shapes differ")
    colors = set(map(int, np.unique(inp)))
    _require(len(colors) == 2, "input must contain background and wall")
    changed = inp != out
    _require(bool(np.any(changed)), "output must recolor regions")
    from_colors = set(map(int, np.unique(inp[changed])))
    _require(len(from_colors) == 1, "only background may change")
    background = next(iter(from_colors))
    wall_color = next(iter(colors - {background}))
    wall = inp == wall_color
    regions = _components(inp == background)
    areas = [len(region) for region in regions]
    _require(len(regions) >= 4, "too few background regions")
    _require(areas.count(min(areas)) == 1 and areas.count(max(areas)) == 1, "area extrema are tied")
    features = _pair_features(wall, regions)
    small_values = {int(out[r, c]) for r, c in features["small"]}
    large_values = {int(out[r, c]) for r, c in features["large"]}
    _require(len(small_values) == len(large_values) == 1, "highlight is not uniform")
    small_color, large_color = next(iter(small_values)), next(iter(large_values))
    roles = (background, wall_color, small_color, large_color)
    _require(len(set(roles)) == 4, "semantic colors must be distinct")
    expected = inp.copy()
    for r, c in features["small"]:
        expected[r, c] = small_color
    for r, c in features["large"]:
        expected[r, c] = large_color
    _require(np.array_equal(out, expected), "incorrect area-based recoloring")
    _require(features["largest_fraction"] <= 0.52, "largest region dominates the grid")
    return roles, features


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    roles = [_analyze_pair(pair)[0] for pair in train]
    _require(len(set(roles)) == 1, "training color conventions disagree")
    background, wall, small_color, large_color = roles[0]
    _require(set(map(int, np.unique(test_input))) == {background, wall}, "test input changes the color convention")
    regions = _components(test_input == background)
    areas = [len(region) for region in regions]
    _require(len(regions) >= 4 and areas.count(min(areas)) == areas.count(max(areas)) == 1, "ambiguous test input")
    prediction = test_input.copy()
    for r, c in regions[areas.index(min(areas))]:
        prediction[r, c] = small_color
    for r, c in regions[areas.index(max(areas))]:
        prediction[r, c] = large_color
    return prediction


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, 4, replace=False))
    profile_order = ["balanced", "tall", "wide"]
    rng.shuffle(profile_order)
    shallow_shift = bool(rng.random() < 0.38)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                profile = profile_order[index % 3]
            elif shallow_shift and index == 0:
                profile = "shallow"
            elif index == 0 and train_count == 2:
                profile = profile_order[2]
            else:
                profile = str(rng.choice(profile_order))
            desired_regions = int(rng.choice((4, 5, 6, 7, 8, 9, 10), p=(.08, .1, .13, .26, .23, .14, .06)))
            previous_counts = [
                len(_components(pair["input"] == colors[0])) for pair in puzzle["train"]
            ] if split == "train" else []
            if split == "train" and index == count - 1 and len(set(previous_counts)) == 1:
                desired_regions = 4 if previous_counts[0] >= 7 else 9
            minimum_target = int(rng.choice((1, 2, 3, 4, 5, 6), p=(.12, .18, .18, .23, .17, .12)))
            prefer = ("none", "interior_small", "close", "corner_small", "interior_large")[(index + int(seed) % 5) % 5]
            for _ in range(12):
                pair = _make_pair(rng, colors, profile, desired_regions, minimum_target, prefer)
                key = _key(pair["input"])
                source_key = (_rows(pair["input"]), _rows(pair["output"]))
                if (split == "train" and index == count - 1 and len(set(previous_counts)) == 1
                        and len(_components(pair["input"] == colors[0])) == previous_counts[0]):
                    continue
                if key not in seen and source_key not in _SOURCE_PAIRS:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split schema")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen_inputs, seen_pairs = set(), set()
    common_roles = None
    train_features = []
    test_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "ARC colors")
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require((_rows(pair["input"]), _rows(pair["output"])) not in _SOURCE_PAIRS, "source copy")
            roles, features = _analyze_pair(pair)
            if common_roles is None:
                common_roles = roles
            _require(roles == common_roles, "color conventions differ")
            _require(features["wall_components"] == 1, "wall network is disconnected")
            _require(features["junctions"] >= 1, "no wall branch")
            (train_features if split == "train" else test_features).append(features)
    _require(len({_aspect(pair["input"].shape) for pair in puzzle["train"]}) >= 2, "training aspect contrast")
    _require(len({len(feature["areas"]) for feature in train_features}) >= 2, "training region-count contrast")
    for pair in puzzle["test"]:
        prediction = _predict_from_training(puzzle["train"], pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]), "test cannot be solved from training")


def _puzzles_equal(a: dict, b: dict) -> bool:
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and np.array_equal(x["output"], y["output"])
        for x, y in zip(a[split], b[split])) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, palettes, shapes, layouts = set(), set(), set(), set(), set()
    region_counts, min_areas, small_zones, large_zones, aspects = set(), set(), set(), set(), set()
    close_count = interior_large = shallow_test = junction_cases = 0
    broad_loses = enclosed_cases = irregular_large = simpler_cases = 0
    narrow_large = bent_small = 0
    paired_variations = shallow_shift_puzzles = same_aspect_tests = 0
    tall_tests = balanced_tests = 0
    largest_shares = []
    varied_regions = varied_locations = varied_min_areas = varied_large_zones = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = [_analyze_pair(pair)[0] for pair in puzzle["train"] + puzzle["test"]]
        assert len(set(roles)) == 1
        palettes.add(roles[0])
        trains = [_analyze_pair(pair)[1] for pair in puzzle["train"]]
        varied_regions += len({len(x["areas"]) for x in trains}) > 1
        varied_locations += len({(x["small_zone"], x["large_zone"]) for x in trains}) > 1
        varied_min_areas += len({min(x["areas"]) for x in trains}) > 1
        varied_large_zones += len({x["large_zone"] for x in trains}) > 1
        shallow_shift_puzzles += (
            all(pair["input"].shape[0] >= 8 for pair in puzzle["train"])
            and any(pair["input"].shape[0] <= 7 and _aspect(pair["input"].shape) == "wide"
                    for pair in puzzle["test"])
        )
        same_aspect_tests += any(
            _aspect(test["input"].shape) in {_aspect(train["input"].shape) for train in puzzle["train"]}
            for test in puzzle["test"]
        )
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                feature = _analyze_pair(pair)[1]
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(sorted(feature["areas"])), _key(inp == roles[0][1])))
                aspects.add(_aspect(inp.shape))
                region_counts.add(len(feature["areas"]))
                min_areas.add(min(feature["areas"]))
                small_zones.add(feature["small_zone"])
                large_zones.add(feature["large_zone"])
                areas = sorted(feature["areas"])
                close_count += areas[-1] - areas[-2] <= 5
                interior_large += feature["large_zone"] == "interior"
                junction_cases += feature["junctions"] >= 3
                shallow_test += split == "test" and inp.shape[0] <= 7 and _aspect(inp.shape) == "wide"
                tall_tests += split == "test" and _aspect(inp.shape) == "tall"
                balanced_tests += split == "test" and _aspect(inp.shape) == "balanced"
                simpler_cases += len(areas) <= 5
                largest_shares.append(feature["largest_fraction"])
                regions = _components(inp == roles[0][0])
                boxes = []
                lengths = []
                for region in regions:
                    rows, cols = zip(*region)
                    row_span, col_span = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
                    boxes.append(row_span * col_span)
                    lengths.append(max(row_span, col_span))
                largest_index = [len(region) for region in regions].index(areas[-1])
                smallest_index = [len(region) for region in regions].index(areas[0])
                broad_loses += any(
                    box > boxes[largest_index] and len(region) < areas[-1]
                    for box, region in zip(boxes, regions)
                )
                enclosed_cases += any(_zone(region, inp.shape) == "interior" for region in regions)
                irregular_large += areas[-1] / boxes[largest_index] < .75
                narrow_large += areas[-1] / lengths[largest_index] <= 2.5
                bent_small += areas[0] / boxes[smallest_index] < 1
                paired_variations += feature["small_zone"] == "interior" and areas[-1] - areas[-2] <= 5
                assert inp.shape == out.shape and not np.array_equal(inp, out)
                assert np.array_equal(_predict_from_training(puzzle["train"], inp.copy()), out)
                assert (_rows(inp), _rows(out)) not in _SOURCE_PAIRS
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 30 and len(shapes) >= 30 and len(layouts) >= 150
    assert len(region_counts) >= 5 and max(min_areas) >= 4
    assert small_zones == {"corner", "edge", "interior"}
    assert large_zones == {"corner", "edge", "interior"}
    assert aspects == {"balanced", "tall", "wide"}
    assert close_count >= 8 and interior_large >= 2 and shallow_test >= 5
    assert junction_cases >= 100 and varied_regions >= 20 and varied_locations >= 20
    assert varied_min_areas >= 20 and varied_large_zones >= 15
    assert broad_loses >= 5 and enclosed_cases >= 20 and irregular_large >= 8
    assert narrow_large >= 8 and bent_small >= 8
    assert simpler_cases >= 20 and paired_variations >= 3
    assert np.median(largest_shares) <= .3 and max(largest_shares) <= .52
    assert shallow_shift_puzzles >= 4 and same_aspect_tests >= 20
    assert tall_tests >= 5 and balanced_tests >= 5
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _puzzles_equal(generate_puzzle(777, 5, 2), generate_puzzle(777, 5, 2))
    assert not _puzzles_equal(generate_puzzle(777, 5, 2), generate_puzzle(778, 5, 2))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
