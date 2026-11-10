import numpy as np


_SOURCE_PAIRS = (
    (("1118000000", "1811018818", "8281081828", "1118088881", "8188081282", "0000088818", "0000011818", "0822081182", "0221000000", "0218000000"), ("822", "221", "218")),
    (("0000008888", "0888808228", "0818808888", "0882808218", "0888808888", "0000000000", "0008888880", "0008882880", "0008281880", "0008188880"), ("8888", "8228", "8888", "8218", "8888")),
    (("2888000000", "8818000000", "1888000000", "8882001882", "8281008818", "8188008288", "0000008881", "0000001888", "0000008818", "0000000000"), ("2888", "8818", "1888", "8882", "8281", "8188")),
    (("2888000000", "8818002810", "1281008880", "8888002180", "0000008820", "0000002810", "0128201880", "0881800000", "0128100000", "0000000000"), ("281", "888", "218", "882", "281", "188")),
)


def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + (seed // 7 + num_test) % (11 - low)
    elif num_test is None:
        num_test = 1 + (seed // 11) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    mode = int(rng.integers(0, 4))
    return {
        "backgrounds": (colors[0], colors[4]) if mode in (1, 3) else (colors[0],),
        "selector": colors[1],
        "anchor": colors[2],
        "companions": (colors[3], colors[5]) if mode in (2, 3) else (colors[3],),
    }


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _area(dim):
    return dim[0] * dim[1]


def _other_counts(rng, area, selector_count, allow_omission):
    remaining = area - selector_count
    assert remaining >= 2
    if allow_omission and remaining >= 3 and rng.random() < 0.15:
        return (selector_count, remaining, 0)
    if rng.random() < 0.48:
        primary = int(rng.integers(max(1, remaining * 2 // 3), remaining))
    else:
        primary = int(rng.integers(1, remaining))
    return (selector_count, primary, remaining - primary)


def _witness_specs(rng):
    dim = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
    area = _area(dim)
    # Both other foreground colors increase in the losing equal-area panel.
    winner_selector = int(rng.integers(4, min(7, area - 2) + 1))
    loser_selector = int(rng.integers(1, winner_selector - 1))
    difference = winner_selector - loser_selector
    first_increment = int(rng.integers(1, difference))
    winner_first = int(rng.integers(1, area - winner_selector))
    winner_second = area - winner_selector - winner_first
    winner_counts = (winner_selector, winner_first, winner_second)
    loser_counts = (loser_selector, winner_first + first_increment,
                    winner_second + difference - first_increment)
    return [(dim, winner_counts), (dim, loser_counts)]


def _challenge_specs(rng):
    # One smaller and one larger decoy; the smaller decoy is denser in selector.
    for _ in range(400):
        dims = [(int(rng.integers(2, 8)), int(rng.integers(2, 8))) for _ in range(3)]
        areas = [_area(dim) for dim in dims]
        order = sorted(range(3), key=lambda i: areas[i])
        small, winner, large = order
        if not (areas[small] + 3 <= areas[winner] <= areas[large] - 3):
            continue
        winner_selector = int(rng.integers(4, min(8, areas[winner] - 2) + 1))
        eligible = [n for n in range(1, min(winner_selector, areas[small] - 1))
                    if n * areas[winner] > winner_selector * areas[small]]
        if not eligible:
            continue
        small_selector = _pick(rng, eligible)
        large_selector = int(rng.integers(0, winner_selector))
        return [(dims[winner], _other_counts(rng, areas[winner], winner_selector, False)),
                (dims[small], _other_counts(rng, areas[small], small_selector, False)),
                (dims[large], _other_counts(rng, areas[large], large_selector, False))]
    raise RuntimeError("Could not construct count-versus-density demonstration")


def _free_specs(rng, profile, size_mode):
    if profile == "intermediate":
        count = int(rng.integers(3, 6))
    else:
        count = int(_pick(rng, (2, 2, 3, 3, 3, 4, 4, 5)))
    limit = (5, 7, 9)[size_mode]
    for _ in range(400):
        dims = [(int(rng.integers(2, limit + 1)),
                 int(rng.integers(2, limit + 1))) for _ in range(count)]
        if profile == "equal":
            dims[1] = dims[0]
            winner = 0
        else:
            areas = [_area(dim) for dim in dims]
            if profile == "smallest":
                minima = [i for i, a in enumerate(areas) if a == min(areas)]
                if len(minima) != 1:
                    continue
                winner = minima[0]
            elif profile == "largest":
                maxima = [i for i, a in enumerate(areas) if a == max(areas)]
                if len(maxima) != 1:
                    continue
                winner = maxima[0]
            else:
                candidates = [i for i, a in enumerate(areas)
                              if min(areas) < a < max(areas)]
                if not candidates:
                    continue
                winner = _pick(rng, candidates)
        winner_area = _area(dims[winner])
        if winner_area < 5:
            continue
        high = min(winner_area - 2, max(4, winner_area // 2 + 1))
        if high < 3:
            continue
        winner_selector = int(rng.integers(3, high + 1))
        specs = [(dims[winner], _other_counts(rng, winner_area, winner_selector, False))]
        for i, dim in enumerate(dims):
            if i == winner:
                continue
            area = _area(dim)
            selector_count = int(rng.integers(0, min(winner_selector, area - 1)))
            specs.append((dim, _other_counts(rng, area, selector_count, True)))
        return specs
    raise RuntimeError("Could not construct free panel profile")


def _make_panel(rng, dim, counts, colors):
    height, width = dim
    selector, anchor, companion = colors
    values = [selector] * counts[0] + [anchor] * counts[1] + [companion] * counts[2]
    mode = int(rng.integers(0, 5))
    rows, cols = np.indices(dim)
    noise = rng.random(dim)
    if mode == 0:
        score = noise
    elif mode == 1:
        cr, cc = int(rng.integers(height)), int(rng.integers(width))
        score = abs(rows - cr) + abs(cols - cc) + noise * 0.65
    elif mode == 2:
        score = (rows if rng.random() < 0.5 else cols) + noise * 0.7
    elif mode == 3:
        score = (rows + cols if rng.random() < 0.5 else rows - cols) + noise * 1.5
    else:
        score = ((rows + cols) % 2) + noise * 1.2
    order = np.argsort(score.ravel(), kind="stable")
    rng.shuffle(values)
    if mode in (1, 2, 3):
        color_order = [selector, anchor, companion]
        rng.shuffle(color_order)
        values = [color for color in color_order for _ in range(counts[colors.index(color)])]
    panel = np.empty(height * width, dtype=np.int8)
    panel[order] = values
    return panel.reshape(dim)


def _layout(rng, dims, size_mode):
    count = len(dims)
    candidates = []
    for _ in range(110):
        vertical = bool(rng.integers(0, 2))
        working_dims = [(w, h) for h, w in dims] if vertical else dims
        order = [int(x) for x in rng.permutation(count)]
        rows = int(rng.integers(1, count + 1))
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, count), size=rows - 1, replace=False)) if rows > 1 else []
        groups = []
        starts = [0] + cuts
        ends = cuts + [count]
        for a, b in zip(starts, ends):
            groups.append(order[a:b])
        gap = 1 if size_mode == 0 or rng.random() < 0.65 else 2
        y = 0
        positions = [None] * count
        width = 0
        for group in groups:
            x = 0
            row_height = max(working_dims[i][0] for i in group)
            for i in group:
                positions[i] = (y, x)
                x += working_dims[i][1] + gap
            width = max(width, x - gap)
            y += row_height + gap
        height = y - gap
        if vertical:
            positions = [(x, y) for y, x in positions]
            height, width = width, height
        if height + 1 <= 30 and width <= 30:
            candidates.append((height, width, positions))
    if not candidates:
        raise RuntimeError("No legal panel layout")
    upper = (13, 20, 30)[size_mode]
    lower = (8, 14, 21)[size_mode]
    eligible = [item for item in candidates if item[0] + 1 <= upper and item[1] <= upper]
    if not eligible:
        raise RuntimeError("Panel layout does not fit requested canvas class")
    if size_mode == 0:
        source_scale = [item for item in eligible if item[0] + 1 <= 10 and item[1] <= 10]
        if source_scale and rng.random() < 0.26:
            base_h, base_w, positions = _pick(rng, source_scale)
            top = int(rng.integers(1, 10 - base_h + 1))
            left = int(rng.integers(0, 10 - base_w + 1))
            return (10, 10), [(r + top, c + left) for r, c in positions]
        eligible.sort(key=lambda item: item[0] * item[1])
        eligible = eligible[:max(1, len(eligible) // 3)]
    base_h, base_w, positions = _pick(rng, eligible)
    if size_mode == 0 and rng.random() < 0.55:
        canvas_h = min(upper, max(lower, base_h + 1) + int(rng.integers(0, 2)))
        canvas_w = min(upper, max(lower, base_w) + int(rng.integers(0, 2)))
    else:
        canvas_h = int(rng.integers(max(lower, base_h + 1), upper + 1))
        canvas_w = int(rng.integers(max(lower, base_w), upper + 1))
    top = int(rng.integers(1, canvas_h - base_h + 1))
    left = int(rng.integers(0, canvas_w - base_w + 1))
    return (canvas_h, canvas_w), [(r + top, c + left) for r, c in positions]


def _make_pair(rng, profile, size_mode, background, colors):
    if profile == "witness":
        specs = _witness_specs(rng)
    elif profile == "challenge":
        specs = _challenge_specs(rng)
    else:
        specs = _free_specs(rng, profile, size_mode)
    dims = [dim for dim, _ in specs]
    canvas_shape, positions = _layout(rng, dims, size_mode)
    canvas = np.full(canvas_shape, background, dtype=np.int8)
    panels = [_make_panel(rng, dim, counts, colors) for dim, counts in specs]
    for panel, (row, col) in zip(panels, positions):
        h, w = panel.shape
        canvas[row:row + h, col:col + w] = panel
    return canvas, panels[0].copy()


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _assert_not_source_pair(input_grid, output_grid):
    for input_rows, output_rows in _SOURCE_PAIRS:
        if input_grid.shape == (len(input_rows), len(input_rows[0])) and output_grid.shape == (len(output_rows), len(output_rows[0])):
            original_input = np.array([[int(c) for c in row] for row in input_rows], dtype=np.int8)
            original_output = np.array([[int(c) for c in row] for row in output_rows], dtype=np.int8)
            assert not (np.array_equal(input_grid, original_input) and np.array_equal(output_grid, original_output)), "Source pair copied"


def _component_records(grid, background):
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    records = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or grid[row, col] == background:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc] and grid[rr, cc] != background:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            r0 = min(r for r, c in cells)
            r1 = max(r for r, c in cells) + 1
            c0 = min(c for r, c in cells)
            c1 = max(c for r, c in cells) + 1
            panel = grid[r0:r1, c0:c1].copy()
            assert len(cells) == panel.size and np.all(panel != background), "Panel is not a filled rectangle"
            assert panel.shape[0] >= 2 and panel.shape[1] >= 2
            records.append({"panel": panel, "box": (r0, c0, r1, c1)})
    return records


def _analyze(puzzle):
    analyses = {"train": [], "test": []}
    training_colors = set()
    backgrounds = set()
    foreground_sets = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid, output = pair["input"], pair["output"]
            background = int(grid[0, 0])
            assert np.all(grid[0] == background), "Background field is not clear along the top"
            records = _component_records(grid, background)
            assert 2 <= len(records) <= 5
            matches = [i for i, record in enumerate(records) if np.array_equal(record["panel"], output)]
            assert len(matches) == 1, "Output must be one exact, uniquely matching panel"
            colors = set(int(x) for x in np.unique(grid)) - {background}
            assert 2 <= len(colors) <= 3
            backgrounds.add(background)
            foreground_sets.append(colors)
            if split == "train":
                training_colors |= colors
            analyses[split].append({"background": background, "records": records,
                                    "winner": matches[0], "colors": colors,
                                    "input_shape": grid.shape, "output_shape": output.shape})
    assert not backgrounds.intersection(set.union(*foreground_sets)), "Background and foreground roles conflict"
    candidates = []
    for color in training_colors:
        if all((lambda counts, winner: counts[winner] > max(n for i, n in enumerate(counts) if i != winner))
               ([int(np.count_nonzero(record["panel"] == color)) for record in a["records"]], a["winner"])
               for a in analyses["train"]):
            candidates.append(color)
    assert len(candidates) == 1, "Training does not uniquely identify a selector"
    selector = candidates[0]
    assert len(set.intersection(*foreground_sets) - {selector}) >= 1, "Pairs have no shared companion color"
    for split in ("train", "test"):
        for analysis in analyses[split]:
            counts = [int(np.count_nonzero(record["panel"] == selector)) for record in analysis["records"]]
            assert counts.count(max(counts)) == 1
            assert analysis["winner"] == counts.index(max(counts))
            analysis["selector_counts"] = counts
    return selector, analyses


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not np.array_equal(pair["input"], pair["output"])
            _assert_not_source_pair(pair["input"], pair["output"])
    selector, analyses = _analyze(puzzle)
    assert all(selector in a["colors"] for split in analyses for a in analyses[split])
    train = analyses["train"]
    assert any(len(a["records"]) == 2 for a in train)
    assert any(len(a["records"]) >= 3 for a in train)
    assert any(any(record["panel"].size >= a["records"][a["winner"]]["panel"].size
                   for i, record in enumerate(a["records"]) if i != a["winner"])
               for a in train)
    assert any(any(a["selector_counts"][i] * a["records"][a["winner"]]["panel"].size
                   > a["selector_counts"][a["winner"]] * record["panel"].size
                   for i, record in enumerate(a["records"]) if i != a["winner"])
               for a in train), "Training must distinguish count from density"


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    params = sample_puzzle_parameters(rng)
    profiles = ["witness", "challenge"] + [str(_pick(rng, ("smallest", "intermediate", "equal", "largest"))) for _ in range(train_count - 2)]
    rng.shuffle(profiles)
    test_profiles = [str(_pick(rng, ("smallest", "intermediate", "equal", "largest"))) for _ in range(test_count)]
    # Canvas classes vary across both splits; test uses the same support with a
    # mild bias toward configurations less frequent in demonstrations.
    size_modes = [0, 1, 2] * ((train_count + test_count + 2) // 3)
    rng.shuffle(size_modes)
    size_modes = size_modes[:train_count + test_count]
    if train_count == 2:
        for i, profile in enumerate(profiles):
            if profile == "challenge":
                size_modes[i] = max(1, size_modes[i])
    palette_choices = [(i % len(params["backgrounds"]), i % len(params["companions"]))
                       for i in range(train_count + test_count)]
    rng.shuffle(palette_choices)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (split, profile) in enumerate(
            [("train", p) for p in profiles] + [("test", p) for p in test_profiles]):
        background_index, companion_index = palette_choices[index]
        background = params["backgrounds"][background_index]
        colors = (params["selector"], params["anchor"], params["companions"][companion_index])
        preferred_mode = size_modes[index]
        pair = None
        for attempt in range(160):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            mode = preferred_mode if attempt < 65 else int(pair_rng.integers(0, 3))
            try:
                input_grid, output_grid = _make_pair(pair_rng, profile, mode, background, colors)
            except RuntimeError:
                continue
            key = _grid_key(input_grid)
            if key in seen:
                continue
            seen.add(key)
            pair = {"input": input_grid, "output": output_grid}
            break
        if pair is None:
            raise RuntimeError("Unable to generate a unique pair")
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
                   for name in ("input", "output")) for split in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures = set()
    palettes = set()
    shapes = set()
    panel_counts = set()
    ranks = {"train": set(), "test": set()}
    orientations = {"train": set(), "test": set()}
    coverages = set()
    review_counts = {"source_scale": 0, "rich": 0, "roomy": 0,
                     "five_panels": 0, "loser_tie": 0, "loser_unequal": 0,
                     "density_reversal": 0, "missing_selector": 0,
                     "missing_companion": 0, "runs": 0,
                     "background_changed": 0, "companion_changed": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        selector, analyses = _analyze(puzzle)
        backgrounds = set()
        companion_sets = set()
        signatures.add(tuple(_grid_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        palettes.add(selector)
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], analyses[split]):
                shapes.add(pair["input"].shape)
                panel_counts.add(len(a["records"]))
                backgrounds.add(a["background"])
                companion_sets.add(tuple(sorted(a["colors"] - {selector})))
                h, w = pair["output"].shape
                orientations[split].add("square" if h == w else "portrait" if h > w else "landscape")
                areas = [record["panel"].size for record in a["records"]]
                counts = a["selector_counts"]
                winner_area = areas[a["winner"]]
                ranks[split].add("smallest" if winner_area == min(areas) else "largest" if winner_area == max(areas) else "intermediate")
                foreground = sum(areas)
                coverages.add(round(foreground / pair["input"].size, 1))
                review_counts["source_scale"] += pair["input"].shape == (10, 10)
                review_counts["rich"] += foreground / pair["input"].size >= 0.4
                review_counts["roomy"] += max(pair["input"].shape) >= 25
                review_counts["five_panels"] += len(areas) == 5
                losing = [n for i, n in enumerate(counts) if i != a["winner"]]
                review_counts["loser_tie"] += len(losing) >= 2 and len(set(losing)) < len(losing)
                review_counts["loser_unequal"] += len(set(losing)) > 1
                review_counts["density_reversal"] += any(
                    counts[i] * winner_area > counts[a["winner"]] * area
                    for i, area in enumerate(areas) if i != a["winner"])
                review_counts["missing_selector"] += any(n == 0 for n in losing)
                review_counts["missing_companion"] += any(
                    len(np.unique(record["panel"])) < 3
                    for i, record in enumerate(a["records"]) if i != a["winner"])
                review_counts["runs"] += any(
                    any(any(np.all(row[j:j + 4] == row[j]) for j in range(len(row) - 3))
                        for row in record["panel"])
                    for record in a["records"])
                assert np.array_equal(pair["output"], a["records"][a["winner"]]["panel"])
        review_counts["background_changed"] += len(backgrounds) > 1
        review_counts["companion_changed"] += len(companion_sets) > 1
    assert len(signatures) == 42 and len(palettes) >= 8 and len(shapes) >= 30
    assert {2, 3, 4}.issubset(panel_counts)
    assert all(len(ranks[split]) == 3 and len(orientations[split]) == 3 for split in ("train", "test"))
    assert len(coverages) >= 4
    assert review_counts["source_scale"] >= 10
    assert review_counts["rich"] >= 20 and review_counts["roomy"] >= 30
    assert review_counts["five_panels"] >= 10
    assert review_counts["loser_tie"] >= 20 and review_counts["loser_unequal"] >= 20
    assert review_counts["density_reversal"] >= 30
    assert review_counts["missing_selector"] >= 15
    assert review_counts["missing_companion"] >= 15
    assert review_counts["runs"] >= 30
    assert review_counts["background_changed"] >= 5
    assert review_counts["companion_changed"] >= 5
    for seed, nt, nv in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2)):
        assert len(generate_puzzle(seed, nt, nv)["train"]) == nt
    for kwargs in ({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
                   {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None}):
        validate_generated_puzzle(generate_puzzle(2001, **kwargs))
    assert _puzzles_equal(generate_puzzle(314159, 7, 3), generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(generate_puzzle(271828, 5, 2), generate_puzzle(271829, 5, 2))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
