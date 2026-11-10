import numpy as np


# Exact source pairs are excluded, without excluding their dimensions or colors.
_SOURCE_ROWS = (
    (("03300", "03300", "00000", "22002", "22000", "00022", "00022"), ("100", "000", "000")),
    (("0330003", "0330000", "0000200", "3000330", "0030330"), ("100", "010", "000")),
    (("00022", "33022", "33000", "00220", "30220"), ("100", "000", "000")),
    (("0033000", "2033030", "0000000", "3300220", "3300220", "0033000", "0033000"), ("100", "010", "001")),
    (("2203003", "2202200", "0002200", "2330022", "0330022", "0000330", "3020330"), ("100", "010", "000")),
    (("03300000", "03300320", "20000003", "00220000", "30220330", "00000330", "03300000", "03302030"), ("100", "010", "001")),
    (("00000", "03300", "03300", "20000", "00003", "33000", "33022", "00022"), ("100", "010", "000")),
)


def _rows_key(rows):
    return ((len(rows), len(rows[0])), tuple(int(c) for row in rows for c in row))


_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.ravel())


def _pair_key(input_grid, output_grid):
    return _grid_key(input_grid), _grid_key(output_grid)


def _check_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_pair_counts(rng, num_train, num_test):
    num_train = _check_optional_count(num_train, "num_train", 2, 10)
    num_test = _check_optional_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _profile(rng, target_squares):
    return {
        "target_squares": target_squares,
        "distractor_squares": int(rng.integers(0, 6)),
        "target_singletons": int(rng.integers(0, 8)),
        "distractor_singletons": int(rng.integers(0, 8)),
        "target_l": 0,
        "distractor_l": 0,
    }


def _profiles(rng, num_train, num_test):
    # Two demonstrations can show either a novel count-two or count-three test.
    if num_train == 2:
        train_counts = [1, int(rng.choice([2, 3]))]
        progression = set(train_counts) == {1, 2}
    elif rng.random() < 0.22:
        # A larger training set can also demonstrate only the first two
        # diagonal positions before asking for the third in the test.
        train_counts = [1, 2]
        train_counts += [int(rng.choice([1, 2])) for _ in range(num_train - 2)]
        progression = True
    else:
        train_counts = [1, 2, 3]
        train_counts += [int(rng.integers(1, 4)) for _ in range(num_train - 3)]
        if num_train >= 4 and rng.random() < 0.30:
            train_counts[int(rng.integers(3, num_train))] = 0
        progression = False
    rng.shuffle(train_counts)
    train = [_profile(rng, count) for count in train_counts]

    # In every puzzle a training output contradicts the pixel-area shortcut.
    high = next(i for i, p in enumerate(train) if p["target_squares"] == 1)
    train[high]["target_singletons"] = int(rng.integers(4, 9))
    train[high]["distractor_squares"] = int(rng.integers(1, 5))

    # Some puzzles put equal target-pixel areas under different answers.
    if rng.random() < 0.38:
        other = next(i for i, p in enumerate(train) if p["target_squares"] >= 2)
        train[high]["target_singletons"] = 8
        train[other]["target_singletons"] = 12 - 4 * train[other]["target_squares"]
    else:
        low = next(i for i, p in enumerate(train) if i != high)
        train[low]["target_singletons"] = int(rng.integers(0, 2))

    # Reproduce source-level clutter occasionally, independently of answer.
    if num_train >= 3 and rng.random() < 0.48:
        clutter = next(i for i, p in enumerate(train) if p["target_squares"] == 2)
        train[clutter]["distractor_squares"] = int(rng.integers(3, 6))
        train[clutter]["distractor_singletons"] = int(rng.integers(3, 7))

    # A sparse contrast prevents "count all squares" and frequency rules.
    sparse = next(i for i, p in enumerate(train) if i != high)
    if rng.random() < 0.55 and not (rng.random() < 0.35 and num_train == 2):
        train[sparse]["distractor_squares"] = 0
        train[sparse]["distractor_singletons"] = int(rng.integers(0, 3))

    # Near-squares are optional negatives; a test uses them only after training.
    has_l = num_train >= 3 and rng.random() < 0.23
    if has_l:
        eligible = [i for i, p in enumerate(train) if p["target_squares"] > 0 and i != high]
        if eligible:
            train[int(rng.choice(eligible))]["target_l"] = 1

    # Test counts may be familiar or the next diagonal count after training 1,2.
    demonstrated = sorted(set(train_counts))
    test = []
    for j in range(num_test):
        if progression and j == 0:
            count = 3
        elif 0 in demonstrated and rng.random() < 0.15:
            count = 0
        else:
            count = int(rng.integers(1, 4))
        p = _profile(rng, count)
        p["target_singletons"] = int(rng.integers(0, 10)) if count else int(rng.integers(1, 6))
        p["distractor_singletons"] = int(rng.integers(0, 10))
        if has_l and rng.random() < 0.42:
            p["target_l"] = 1
        test.append(p)
    return train, test


_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)
_L_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _objects(rng, profile, target, distractor_colors):
    total_distractor = (
        profile["distractor_squares"]
        + profile["distractor_singletons"]
        + profile["distractor_l"]
    )
    colors = []
    if total_distractor:
        colors = [int(x) for x in distractor_colors]
        assigned = colors[:]
        assigned.extend(int(rng.choice(colors)) for _ in range(total_distractor - len(colors)))
        rng.shuffle(assigned)
    else:
        assigned = []
    objects = []
    objects.extend((target, _SQUARE) for _ in range(profile["target_squares"]))
    objects.extend((target, _SINGLE) for _ in range(profile["target_singletons"]))
    objects.extend((target, _L_SHAPES[int(rng.integers(0, 4))]) for _ in range(profile["target_l"]))
    distractor_shapes = (
        [_SQUARE] * profile["distractor_squares"]
        + [_SINGLE] * profile["distractor_singletons"]
        + [_L_SHAPES[int(rng.integers(0, 4))] for _ in range(profile["distractor_l"])]
    )
    objects.extend(zip(assigned, distractor_shapes))
    # Large shapes first, without fixing the foreground-color order.
    rng.shuffle(objects)
    objects.sort(key=lambda object_: -len(object_[1]))
    return objects


def _dimensions(rng, regime, attempt):
    probabilities = (0.40, 0.43, 0.17) if regime == "source_train" else (0.23, 0.42, 0.35)
    band = int(rng.choice(3, p=probabilities))
    if attempt >= 28:
        band = max(band, 1)
    if attempt >= 48:
        band = 2
    ranges = ((5, 9), (8, 17), (16, 30))
    low, high = ranges[band]
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    if band > 0 and rng.random() < 0.28:
        side = int(rng.integers(5, 11))
        if rng.random() < 0.5:
            height = side
        else:
            width = side
    return height, width


def _place_objects(rng, height, width, objects, background, seek_contact):
    occupied = {}
    contact_exists = False
    placements = {}
    for _, shape in objects:
        if shape not in placements:
            max_row = max(row for row, _ in shape)
            max_col = max(col for _, col in shape)
            placements[shape] = [
                tuple((row + dr, col + dc) for dr, dc in shape)
                for row in range(height - max_row)
                for col in range(width - max_col)
            ]

    for color, shape in objects:
        candidates = placements[shape]
        first_valid = None
        selected = None
        for raw_index in rng.permutation(len(candidates)):
            cells = candidates[int(raw_index)]
            local = set(cells)
            if any(cell in occupied for cell in cells):
                continue
            touches_same = False
            touches_other = False
            for row, col in cells:
                for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if neighbor in local:
                        continue
                    neighbor_color = occupied.get(neighbor)
                    if neighbor_color == color:
                        touches_same = True
                        break
                    if neighbor_color is not None:
                        touches_other = True
                if touches_same:
                    break
            if touches_same:
                continue
            if first_valid is None:
                first_valid = (cells, touches_other)
            if seek_contact and not contact_exists and touches_other:
                selected = (cells, True)
                break
            if not seek_contact or contact_exists:
                selected = (cells, touches_other)
                break
        if selected is None:
            selected = first_valid
        if selected is None:
            return None
        cells, touches_other = selected
        contact_exists |= touches_other
        for cell in cells:
            occupied[cell] = color

    if seek_contact and not contact_exists and len({color for color, _ in objects}) > 1:
        return None
    grid = np.full((height, width), background, dtype=np.int8)
    for (row, col), color in occupied.items():
        grid[row, col] = color
    return grid


def _make_output(count, background, indicator):
    output = np.full((3, 3), background, dtype=np.int8)
    for i in range(count):
        output[i, i] = indicator
    return output


def _generate_pair(rng, regime, profile, target, indicator, background):
    total_distractor = (
        profile["distractor_squares"]
        + profile["distractor_singletons"]
        + profile["distractor_l"]
    )
    palette = [color for color in range(10) if color not in (target, indicator, background)]
    ncolors = min(total_distractor, int(rng.choice([1, 1, 1, 2, 2, 3])))
    distractors = rng.choice(palette, size=ncolors, replace=False) if ncolors else []
    objects = _objects(rng, profile, target, distractors)
    foreground_cells = sum(len(shape) for _, shape in objects)
    seek_contact = total_distractor > 0 and rng.random() < 0.58
    for attempt in range(65):
        height, width = _dimensions(rng, regime, attempt)
        if foreground_cells > 0.58 * height * width:
            continue
        grid = _place_objects(rng, height, width, objects, background, seek_contact)
        if grid is None:
            continue
        colors, counts = np.unique(grid, return_counts=True)
        if int(colors[int(np.argmax(counts))]) != background:
            continue
        return {"input": grid, "output": _make_output(profile["target_squares"], background, indicator)}
    # This is a deterministic, spacious fallback for legal boundary requests.
    grid = _place_objects(rng, 30, 30, objects, background, seek_contact)
    if grid is None:
        raise RuntimeError("could not place requested objects")
    return {"input": grid, "output": _make_output(profile["target_squares"], background, indicator)}


def _color_components(grid, color):
    height, width = grid.shape
    seen = set()
    components = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) != color or (row, col) in seen:
                continue
            component = set()
            pending = [(row, col)]
            seen.add((row, col))
            while pending:
                r, c = pending.pop()
                component.add((r, c))
                for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    nr, nc = neighbor
                    if 0 <= nr < height and 0 <= nc < width and neighbor not in seen and int(grid[nr, nc]) == color:
                        seen.add(neighbor)
                        pending.append(neighbor)
            components.append(component)
    return components


def _analyze_color(grid, color):
    result = {"squares": 0, "singletons": 0, "l_shapes": 0, "components": []}
    for component in _color_components(grid, color):
        result["components"].append(component)
        if len(component) == 1:
            result["singletons"] += 1
            continue
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        assert max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1
        bounding = {(r, c) for r in range(min(rows), max(rows) + 1) for c in range(min(cols), max(cols) + 1)}
        assert component.issubset(bounding)
        if len(component) == 4:
            assert component == bounding
            result["squares"] += 1
        else:
            assert len(component) == 3
            result["l_shapes"] += 1
    return result


def _input_background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(np.max(counts))
    assert int(np.count_nonzero(counts == largest)) == 1, "background must be uniquely most frequent"
    return int(colors[int(np.argmax(counts))])


def _cross_color_contact(grid, background):
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            color = int(grid[row, col])
            if color == background:
                continue
            for neighbor in ((row + 1, col), (row, col + 1)):
                nr, nc = neighbor
                if nr < grid.shape[0] and nc < grid.shape[1]:
                    other = int(grid[nr, nc])
                    if other != background and other != color:
                        return True
    return False


def _inspect_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    records = []
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9
            input_grid, output = pair["input"], pair["output"]
            assert output.shape == (3, 3)
            key = _grid_key(input_grid)
            pair_key = _pair_key(input_grid, output)
            assert key not in used_inputs and pair_key not in used_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            assert not np.array_equal(input_grid, output)
            used_inputs.add(key)
            used_pairs.add(pair_key)
            background = _input_background(input_grid)
            assert int(np.count_nonzero(output == background)) >= 6
            nonbackground = {int(x) for x in np.unique(output)} - {background}
            assert len(nonbackground) <= 1
            indicator = next(iter(nonbackground)) if nonbackground else None
            count = 0 if indicator is None else int(np.count_nonzero(output == indicator))
            assert 0 <= count <= 3
            if indicator is not None:
                assert np.array_equal(output, _make_output(count, background, indicator))
            else:
                assert np.all(output == background)
            records.append({
                "section": section, "index": index, "input": input_grid,
                "background": background, "indicator": indicator, "count": count,
                "contact": _cross_color_contact(input_grid, background),
            })
    train = [r for r in records if r["section"] == "train"]
    positive_indicators = {r["indicator"] for r in train if r["indicator"] is not None}
    assert len(positive_indicators) == 1
    indicator = next(iter(positive_indicators))
    assert all(r["indicator"] in (None, indicator) for r in records)
    assert all(indicator not in np.unique(r["input"]) for r in records)
    foreground = set()
    for r in train:
        foreground.update(int(x) for x in np.unique(r["input"]) if int(x) != r["background"])
    # A pair's background may have served as a distractor in another pair.
    # Such a color cannot be the puzzle's fixed target role.
    foreground.difference_update(r["background"] for r in train)
    assert foreground
    matching_targets = []
    for color in foreground:
        if all(_analyze_color(r["input"], color)["squares"] == r["count"] for r in train):
            matching_targets.append(color)
    assert len(matching_targets) == 1, "training must identify the counted color"
    target = matching_targets[0]
    assert target != indicator
    for r in records:
        grid = r["input"]
        assert r["background"] != target
        assert target in np.unique(grid)
        assert len(np.unique(grid)) <= 6
        analyses = {}
        for color in (int(x) for x in np.unique(grid) if int(x) != r["background"]):
            analyses[color] = _analyze_color(grid, color)
        assert 0 <= analyses[target]["squares"] <= 3
        assert analyses[target]["squares"] == r["count"]
        assert sum(a["l_shapes"] for a in analyses.values()) <= 1
        r["target_squares"] = analyses[target]["squares"]
        r["target_singletons"] = analyses[target]["singletons"]
        r["target_l"] = analyses[target]["l_shapes"]
        r["distractor_squares"] = sum(a["squares"] for c, a in analyses.items() if c != target)
        r["distractor_singletons"] = sum(a["singletons"] for c, a in analyses.items() if c != target)
        r["distractor_colors"] = len(analyses) - 1
        assert r["distractor_squares"] <= 5
        assert all(c not in (indicator, r["background"], target) for c in analyses if c != target)
        # This reconstructed prediction uses training-derived roles only.
        assert np.array_equal(puzzle[r["section"]][r["index"]]["output"], _make_output(analyses[target]["squares"], r["background"], indicator))
    assert any(r["target_squares"] in (1, 2) and r["target_singletons"] >= 4 for r in train)
    assert any(min(3, (4 * r["target_squares"] + r["target_singletons"] + 3 * r["target_l"]) // 4) != r["count"] for r in train)
    assert any(r["distractor_squares"] > 0 for r in train)
    assert len({r["target_squares"] for r in train}) >= 2
    if any(r["target_l"] for r in records if r["section"] == "test"):
        assert any(r["target_l"] for r in train)
    if any(r["count"] == 0 for r in records if r["section"] == "test"):
        assert any(r["count"] == 0 for r in train)
    return {"target": target, "indicator": indicator, "records": records}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_pair_counts(rng, num_train, num_test)
    target, indicator = (int(x) for x in rng.choice(10, size=2, replace=False))
    available_backgrounds = [c for c in range(10) if c not in (target, indicator)]
    for _ in range(12):
        train_profiles, test_profiles = _profiles(rng, num_train, num_test)
        backgrounds = [int(x) for x in rng.choice(available_backgrounds, size=num_train + num_test, replace=True)]
        if backgrounds[0] == backgrounds[1]:
            backgrounds[1] = int(rng.choice([c for c in available_backgrounds if c != backgrounds[0]]))
        puzzle = {"train": [], "test": []}
        used_inputs = set()
        index = 0
        try:
            for section, regime, profiles in (("train", "source_train", train_profiles), ("test", "target_test", test_profiles)):
                for profile in profiles:
                    for _ in range(12):
                        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                        pair = _generate_pair(pair_rng, regime, profile, target, indicator, backgrounds[index])
                        if _grid_key(pair["input"]) in used_inputs or _pair_key(pair["input"], pair["output"]) in _SOURCE_PAIR_KEYS:
                            continue
                        used_inputs.add(_grid_key(pair["input"]))
                        puzzle[section].append(pair)
                        break
                    else:
                        raise AssertionError("duplicate or source pair")
                    index += 1
            validate_generated_puzzle(puzzle)
            return puzzle
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("could not construct a valid puzzle")


def validate_generated_puzzle(puzzle: dict) -> None:
    _inspect_puzzle(puzzle)


def _puzzles_equal(left, right):
    return all(
        len(left[section]) == len(right[section])
        and all(np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"]) for a, b in zip(left[section], right[section]))
        for section in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(10000, 10042)]
    assert _puzzles_equal(generate_puzzle(31001, 7, 3), generate_puzzle(31001, 7, 3))
    assert _puzzles_equal(generate_puzzle(31002), generate_puzzle(31002))
    for seed, nt, nx in ((33001, 2, 1), (33002, 10, 1), (33003, 10, 4), (34001, None, 1), (34002, None, 4), (34003, 2, None), (34004, 10, None)):
        puzzle = generate_puzzle(seed, nt, nx)
        validate_generated_puzzle(puzzle)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if nx is not None:
            assert len(puzzle["test"]) == nx
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 4}, {"num_train": True}, {"num_train": 4.0}):
        try:
            generate_puzzle(35001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts, shapes, target_counts = set(), set(), set(), set()
    backgrounds, palette_roles, signatures, profiles = set(), set(), set(), set()
    total = high_noise = contact = many_distractors = many_singletons = equal_area = 0
    large_train = small_test = multicolor = zero = l_shapes = 0
    for puzzle in puzzles:
        metadata = _inspect_puzzle(puzzle)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palette_roles.add((metadata["target"], metadata["indicator"]))
        train_records = [r for r in metadata["records"] if r["section"] == "train"]
        areas = {}
        for r in train_records:
            area = 4 * r["target_squares"] + r["target_singletons"] + 3 * r["target_l"]
            areas.setdefault(area, set()).add(r["target_squares"])
        equal_area += any(len(counts) > 1 for counts in areas.values())
        for r in metadata["records"]:
            total += 1
            shape = r["input"].shape
            shapes.add(shape)
            backgrounds.add(r["background"])
            target_counts.add(r["target_squares"])
            profiles.add((r["section"], shape, r["target_squares"], r["distractor_squares"], r["target_singletons"], r["distractor_singletons"], r["distractor_colors"]))
            signatures.add(_grid_key(r["input"]))
            high_noise += r["section"] == "train" and r["target_singletons"] >= 4
            contact += r["contact"]
            many_distractors += r["distractor_squares"] > r["target_squares"]
            many_singletons += r["target_singletons"] + r["distractor_singletons"] >= 5
            multicolor += r["distractor_colors"] >= 2
            zero += r["target_squares"] == 0
            l_shapes += r["target_l"] > 0
            large_train += r["section"] == "train" and max(shape) >= 16
            small_test += r["section"] == "test" and max(shape) <= 9
        assert len({r["background"] for r in metadata["records"]}) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palette_roles) >= 30 and len(backgrounds) == 10
    assert len(signatures) >= int(0.95 * total)
    assert len(shapes) >= 45 and len(profiles) >= 100
    assert len(target_counts) >= 3
    assert high_noise >= 42 and equal_area >= 5
    assert contact >= 35 and many_distractors >= 35 and many_singletons >= 60
    assert multicolor >= 25 and large_train >= 20 and small_test >= 8
    assert zero >= 2 and l_shapes >= 2
