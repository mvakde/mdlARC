import numpy as np


_DIRECTIONS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_TURN_SIDE = {"U": "left", "D": "right", "L": "bottom", "R": "top"}
_SOURCE_INPUTS = (
    ((16, 16), ((3, 11), (4, 5), (9, 10), (11, 4)),
     ((0, 5), (3, 14), (8, 10), (15, 4))),
    ((10, 10), ((2, 4), (5, 6), (6, 2)),
     ((1, 4), (6, 0), (6, 6))),
    ((8, 8), ((0, 0), (4, 5), (5, 2)),
     ((0, 1), (2, 5), (6, 2))),
    ((16, 16), ((3, 0), (3, 13), (4, 6), (9, 11), (14, 2), (14, 7), (14, 13)),
     ((2, 0), (4, 8), (4, 13), (8, 11), (11, 7), (13, 2), (14, 10))),
)


def _checked_count(value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("pair counts must be integers or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError("pair count outside its legal range")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, 2, 10)
    num_test = _checked_count(num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _inside(cell, height, width):
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _side(cell, height, width):
    if cell[0] < 0:
        return "top"
    if cell[0] >= height:
        return "bottom"
    if cell[1] < 0:
        return "left"
    if cell[1] >= width:
        return "right"
    return None


def _aligned(first, second):
    return first[0] == second[0] or first[1] == second[1]


def _corridor(pivot, source):
    if pivot[0] == source[0]:
        step = 1 if source[1] > pivot[1] else -1
        return tuple((pivot[0], c) for c in range(pivot[1] + step, source[1], step))
    step = 1 if source[0] > pivot[0] else -1
    return tuple((r, pivot[1]) for r in range(pivot[0] + step, source[0], step))


def _destination(pivot, source):
    dr, dc = source[0] - pivot[0], source[1] - pivot[1]
    return pivot[0] - dc, pivot[1] + dr


def _matching(pivots, sources, occupied):
    """Find the sole one-to-one alignment through empty corridors."""
    options = []
    for source in sources:
        options.append([i for i, pivot in enumerate(pivots)
                        if _aligned(source, pivot) and not any(
                            cell in occupied for cell in _corridor(pivot, source))])
    order = sorted(range(len(sources)), key=lambda i: len(options[i]))
    answers = []
    assignment = [-1] * len(sources)

    def search(depth, used):
        if len(answers) > 1:
            return
        if depth == len(order):
            answers.append(tuple(assignment))
            return
        source_index = order[depth]
        for pivot_index in options[source_index]:
            if pivot_index not in used:
                assignment[source_index] = pivot_index
                search(depth + 1, used | {pivot_index})

    search(0, set())
    return answers[0] if len(answers) == 1 else None


def _candidates(height, width, spec):
    direction, distance, kind = spec
    dr, dc = _DIRECTIONS[direction]
    side = _TURN_SIDE[direction]
    candidates = []
    for pr in range(height):
        for pc in range(width):
            pivot = (pr, pc)
            source = (pr + dr * distance, pc + dc * distance)
            if not _inside(source, height, width):
                continue
            destination = _destination(pivot, source)
            clipped = _side(destination, height, width)
            on_edge = ((side == "top" and destination[0] == 0) or
                       (side == "bottom" and destination[0] == height - 1) or
                       (side == "left" and destination[1] == 0) or
                       (side == "right" and destination[1] == width - 1))
            one_beyond = ((side == "top" and destination[0] == -1) or
                          (side == "bottom" and destination[0] == height) or
                          (side == "left" and destination[1] == -1) or
                          (side == "right" and destination[1] == width))
            if kind == "inside" and clipped is not None:
                continue
            if kind == "edge" and (clipped is not None or not on_edge):
                continue
            if kind == "off" and clipped != side:
                continue
            if kind == "off1" and (clipped != side or not one_beyond):
                continue
            candidates.append((pivot, source, destination, _corridor(pivot, source)))
    return candidates


def _can_add(candidate, placed, height, width):
    pivot, source, destination, corridor = candidate
    points = {point for item in placed for point in item[:2]}
    corridors = {cell for item in placed for cell in item[3]}
    destinations = {item[2] for item in placed if _inside(item[2], height, width)}
    if pivot in points or source in points or pivot in corridors or source in corridors:
        return False
    if any(cell in points for cell in corridor):
        return False
    if pivot in destinations or source in destinations:
        return False
    if _inside(destination, height, width):
        if destination in points or destination in destinations or destination in (pivot, source):
            return False
    return True


def _candidate_degrees(pivots, sources):
    occupied = set(pivots + sources)
    return [sum(_aligned(source, pivot) and not any(
        cell in occupied for cell in _corridor(pivot, source))
    for pivot in pivots) for source in sources]


def _has_equal_distance_alternative(pivots, sources):
    occupied = set(pivots + sources)
    for source in sources:
        distances = [abs(source[0] - pivot[0]) + abs(source[1] - pivot[1])
                     for pivot in pivots
                     if _aligned(source, pivot) and not any(
                         cell in occupied for cell in _corridor(pivot, source))]
        if len(distances) > len(set(distances)):
            return True
    return False


def _place_specs(rng, height, width, specs, association):
    pools = [_candidates(height, width, spec) for spec in specs]
    if any(not pool for pool in pools):
        return None
    for _ in range(36):
        placed = []
        for index, pool in enumerate(pools):
            accepted = None
            for choice in rng.permutation(len(pool))[:350]:
                candidate = pool[int(choice)]
                if association in ("cross", "tie") and index == 1:
                    first_pivot, first_source = placed[0][:2]
                    if not _aligned(first_source, candidate[0]) or _aligned(candidate[1], first_pivot):
                        continue
                    if association == "tie":
                        d1 = abs(first_source[0] - first_pivot[0]) + abs(first_source[1] - first_pivot[1])
                        d2 = abs(first_source[0] - candidate[0][0]) + abs(first_source[1] - candidate[0][1])
                        if d1 != d2:
                            continue
                if not _can_add(candidate, placed, height, width):
                    continue
                trial = placed + [candidate]
                pivots = [item[0] for item in trial]
                sources = [item[1] for item in trial]
                if _matching(pivots, sources, set(pivots + sources)) is None:
                    continue
                if association == "isolated" and any(d != 1 for d in _candidate_degrees(pivots, sources)):
                    continue
                accepted = candidate
                break
            if accepted is None:
                break
            placed.append(accepted)
        if len(placed) == len(specs):
            pivots = [item[0] for item in placed]
            sources = [item[1] for item in placed]
            if association == "cross" and max(_candidate_degrees(pivots, sources)) == 1:
                continue
            if association == "tie" and not _has_equal_distance_alternative(pivots, sources):
                continue
            return placed
    return None


def _dimensions(rng, count):
    minimum = max(6, 4 + count // 2)
    style = int(rng.integers(0, 4))
    if style == 0:
        return (int(rng.integers(minimum, min(30, minimum + 6) + 1)),
                int(rng.integers(minimum, min(30, minimum + 6) + 1)))
    if style == 1:
        return (int(rng.integers(max(minimum, 9), 23)),
                int(rng.integers(max(minimum, 9), 23)))
    if style == 2:
        return (int(rng.integers(max(minimum, 18), 31)),
                int(rng.integers(max(minimum, 18), 31)))
    short = int(rng.integers(minimum, min(18, minimum + 7) + 1))
    long = int(rng.integers(max(short + 3, 19), 31))
    return (short, long) if rng.integers(0, 2) else (long, short)


def _distance(rng, height, width, direction):
    maximum = (height if direction in "UD" else width) - 1
    tier = int(rng.integers(0, 10))
    if tier < 6:
        low, high = 1, min(4, maximum)
    elif tier < 9 and maximum >= 5:
        low, high = 5, min(8, maximum)
    elif maximum >= 9:
        low, high = 9, maximum
    else:
        low, high = 1, maximum
    return int(rng.integers(low, high + 1))


def _specs(rng, height, width, count, *, cover=False, clipped=None,
           association="free", long_visible=False):
    if cover:
        vertical = str(rng.choice(("U", "D")))
        horizontal = str(rng.choice(("L", "R")))
        directions = [vertical, horizontal,
                      "D" if vertical == "U" else "U",
                      "R" if horizontal == "L" else "L"]
        distances = [int(rng.integers(1, 5)) for _ in range(4)]
        kinds = ["inside"] * 4
        if rng.integers(0, 2):
            kinds[int(rng.integers(2, 4))] = "edge"
        return list(zip(directions, distances, kinds))
    directions = [str(rng.choice(tuple(_DIRECTIONS))) for _ in range(count)]
    if association in ("cross", "tie") and count >= 2:
        directions[:2] = [str(rng.choice(("U", "D"))),
                          str(rng.choice(("L", "R")))]
    distances = [_distance(rng, height, width, direction) for direction in directions]
    kinds = ["inside"] * count
    minimum_visible = 2 if association in ("cross", "tie") else 1
    if clipped is None:
        clipped = int(rng.integers(0, min(3, count - minimum_visible) + 1))
    clipped = min(clipped, count - minimum_visible)
    available = list(range(minimum_visible if association in ("cross", "tie") else 0, count))
    chosen = [int(x) for x in rng.choice(available, size=clipped, replace=False)] if clipped else []
    for index in chosen:
        kinds[index] = "off1" if rng.integers(0, 2) else "off"
    visible = [i for i in range(count) if i not in chosen]
    if long_visible and visible:
        eligible = [i for i in visible if
                    (height if directions[i] in "UD" else width) >= 6]
        if eligible:
            index = int(rng.choice(eligible))
            maximum = (height if directions[index] in "UD" else width) - 1
            distances[index] = int(rng.integers(5, min(maximum, 12) + 1))
    if visible and rng.integers(0, 3) == 0:
        kinds[int(rng.choice(visible))] = "edge"
    return list(zip(directions, distances, kinds))


def _make_pair(rng, count, colors, background, *, cover=False,
               forced_clip=None, association="free", long_visible=False):
    for _ in range(80):
        height, width = _dimensions(rng, count)
        specs = _specs(rng, height, width, count, cover=cover,
                       clipped=forced_clip, association=association,
                       long_visible=long_visible)
        placed = _place_specs(rng, height, width, specs, association)
        if placed is None:
            continue
        pivot_color, source_color, result_color = colors
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full((height, width), background, dtype=np.int8)
        for pivot, source, _, _ in placed:
            inp[pivot] = pivot_color
            inp[source] = source_color
            out[pivot] = pivot_color
        for _, _, destination, _ in placed:
            if _inside(destination, height, width):
                out[destination] = result_color
        return {"input": inp, "output": out}
    raise RuntimeError("could not construct a uniquely paired scene")


def _grid_key(grid):
    return grid.shape, grid.astype(np.int16, copy=False).tobytes()


def _choose_test_count(rng, train_counts):
    high, low = max(train_counts), min(train_counts)
    style = int(rng.integers(0, 4))
    if style == 0:
        return int(rng.integers(1, 4))
    if style == 1:
        return int(rng.integers(max(1, high - 1), min(12, high + 1) + 1))
    if style == 2:
        return int(rng.integers(min(12, high + 1), 13))
    return int(rng.integers(max(1, low - 1), 12))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    backgrounds = [int(x) for x in rng.permutation(
        [c for c in range(10) if c not in colors])]
    train_objects = [4, int(rng.integers(2, 8))]
    train_objects.extend(int(rng.integers(1, 11)) for _ in range(train_count - 2))
    test_objects = [_choose_test_count(rng, train_objects) for _ in range(test_count)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, count in enumerate(train_objects + test_objects):
        if index == 0:
            settings = {"cover": True, "association": "tie", "forced_clip": 0}
        elif index == 1:
            settings = {"forced_clip": 1}
        else:
            association = (str(rng.choice(("cross", "tie"))) if
                           count >= 2 and rng.integers(0, 2) else
                           ("isolated" if rng.integers(0, 3) == 0 else "free"))
            settings = {"association": association,
                        "long_visible": index >= train_count and bool(rng.integers(0, 2))}
        background = backgrounds[index % len(backgrounds)]
        for _ in range(12):
            pair = _make_pair(rng, count, colors, background, **settings)
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("could not obtain a unique input")
        seen.add(key)
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles_from_train(train):
    pair = train[0]
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    input_colors = set(int(x) for x in colors)
    output_colors = set(int(x) for x in np.unique(out))
    assert len(input_colors) == 3
    assert len(input_colors - output_colors) == 1
    assert len(output_colors - input_colors) == 1
    source_color = (input_colors - output_colors).pop()
    result_color = (output_colors - input_colors).pop()
    pivot_color = (input_colors & output_colors) - {background}
    assert len(pivot_color) == 1
    return pivot_color.pop(), source_color, result_color


def _analyze_pair(pair, colors):
    inp, out = pair["input"], pair["output"]
    height, width = inp.shape
    pivot_color, source_color, result_color = colors
    palette, frequencies = np.unique(inp, return_counts=True)
    background = int(palette[np.argmax(frequencies)])
    assert background not in colors
    assert set(int(x) for x in palette) == {background, pivot_color, source_color}
    pivots = [tuple(map(int, p)) for p in np.argwhere(inp == pivot_color)]
    sources = [tuple(map(int, p)) for p in np.argwhere(inp == source_color)]
    assert len(pivots) == len(sources) >= 1
    occupied = set(pivots + sources)
    matching = _matching(pivots, sources, occupied)
    assert matching is not None
    expected = np.full_like(inp, background)
    for pivot in pivots:
        expected[pivot] = pivot_color
    records, destinations = [], []
    for source, pivot_index in zip(sources, matching):
        pivot = pivots[pivot_index]
        dr, dc = source[0] - pivot[0], source[1] - pivot[1]
        assert (dr == 0) != (dc == 0)
        assert not any(cell in occupied for cell in _corridor(pivot, source))
        direction = "U" if dr < 0 else "D" if dr > 0 else "L" if dc < 0 else "R"
        distance = abs(dr) + abs(dc)
        destination = _destination(pivot, source)
        clip = _side(destination, height, width)
        if clip is None:
            destinations.append(destination)
        records.append((pivot, source, destination, direction, distance, clip))
    assert len(set(destinations)) == len(destinations)
    assert not (set(destinations) & occupied)
    for destination in destinations:
        expected[destination] = result_color
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {"height": height, "width": width, "count": len(records),
            "directions": [r[3] for r in records],
            "visible_directions": [r[3] for r in records if r[5] is None],
            "distances": [r[4] for r in records],
            "visible_distances": [r[4] for r in records if r[5] is None],
            "clips": [r[5] for r in records if r[5] is not None],
            "candidate_degrees": _candidate_degrees(pivots, sources),
            "destinations": destinations}


def _predict_from_training(train, test_input):
    pivot_color, source_color, result_color = _roles_from_train(train)
    palette, frequencies = np.unique(test_input, return_counts=True)
    background = int(palette[np.argmax(frequencies)])
    pivots = [tuple(map(int, p)) for p in np.argwhere(test_input == pivot_color)]
    sources = [tuple(map(int, p)) for p in np.argwhere(test_input == source_color)]
    matching = _matching(pivots, sources, set(pivots + sources))
    assert matching is not None
    result = np.full_like(test_input, background)
    for pivot in pivots:
        result[pivot] = pivot_color
    for source, index in zip(sources, matching):
        destination = _destination(pivots[index], source)
        if _inside(destination, *test_input.shape):
            result[destination] = result_color
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
        assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
        nt, nv = len(puzzle["train"]), len(puzzle["test"])
        assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv
        seen_inputs = set()
        colors = _roles_from_train(puzzle["train"])
        infos = {"train": [], "test": []}
        backgrounds = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {"input", "output"}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= size <= 30 for size in grid.shape)
                    assert np.all((grid >= 0) & (grid <= 9))
                assert pair["input"].shape == pair["output"].shape
                key = _grid_key(pair["input"])
                assert key not in seen_inputs
                seen_inputs.add(key)
                info = _analyze_pair(pair, colors)
                infos[split].append(info)
                palette, frequencies = np.unique(pair["input"], return_counts=True)
                background = int(palette[np.argmax(frequencies)])
                backgrounds.append(background)
                if (background, *colors) == (8, 1, 6, 7):
                    pivots = tuple(tuple(map(int, p)) for p in np.argwhere(pair["input"] == 1))
                    sources = tuple(tuple(map(int, p)) for p in np.argwhere(pair["input"] == 6))
                    assert (pair["input"].shape, pivots, sources) not in _SOURCE_INPUTS
        assert len(set(backgrounds)) >= 2
        train = infos["train"]
        assert {d for info in train for d in info["visible_directions"]} == set(_DIRECTIONS)
        assert any(info["clips"] for info in train)
        assert any(max(info["candidate_degrees"]) > 1 for info in train)
        assert any(info["destinations"] for info in train)
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                  pair["output"])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("invalid generated puzzle") from exc


def _puzzles_equal(a, b):
    return (all(len(a[split]) == len(b[split]) for split in ("train", "test")) and
            all(np.array_equal(pa[role], pb[role])
                for split in ("train", "test")
                for pa, pb in zip(a[split], b[split])
                for role in ("input", "output")))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, object_counts, palettes, layouts = set(), set(), set(), set()
    complex_train = complex_test = no_clip = multi_clip = long_visible = 0
    clip_sides, edge_landing, one_beyond = set(), 0, 0
    equal_ties = repeated_side = thirty_side = one_pair = five_six = 0
    varied_backgrounds = 0
    sparse_test = crowded_test = subset_test = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        colors = _roles_from_train(puzzle["train"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes.add(colors)
        infos = {split: [_analyze_pair(pair, colors) for pair in puzzle[split]]
                 for split in ("train", "test")}
        puzzle_backgrounds = set()
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                shapes.add(pair["input"].shape)
                object_counts.add(info["count"])
                palette, frequencies = np.unique(pair["input"], return_counts=True)
                background = int(palette[np.argmax(frequencies)])
                puzzle_backgrounds.add(background)
                normalized = np.zeros(pair["input"].shape, dtype=np.int8)
                normalized[pair["input"] == colors[0]] = 1
                normalized[pair["input"] == colors[1]] = 2
                layouts.add((pair["input"].shape, normalized.tobytes()))
                no_clip += not info["clips"]
                multi_clip += len(info["clips"]) >= 2
                long_visible += any(d >= 5 for d in info["visible_distances"])
                clip_sides.update(info["clips"])
                repeated_side += len(info["clips"]) > len(set(info["clips"]))
                thirty_side += 30 in pair["input"].shape
                one_pair += info["count"] == 1
                five_six += info["count"] in (5, 6)
                edge_landing += any(r in (0, info["height"] - 1) or
                                    c in (0, info["width"] - 1)
                                    for r, c in info["destinations"])
                pivots = [tuple(map(int, p)) for p in np.argwhere(pair["input"] == colors[0])]
                sources = [tuple(map(int, p)) for p in np.argwhere(pair["input"] == colors[1])]
                equal_ties += _has_equal_distance_alternative(pivots, sources)
                matching = _matching(pivots, sources, set(pivots + sources))
                for source, pivot_index in zip(sources, matching):
                    dest = _destination(pivots[pivot_index], source)
                    one_beyond += (dest[0] == -1 or dest[0] == info["height"] or
                                   dest[1] == -1 or dest[1] == info["width"])
                if max(info["candidate_degrees"]) > 1:
                    if split == "train":
                        complex_train += 1
                    else:
                        complex_test += 1
        max_train = max(info["count"] for info in infos["train"])
        sparse_test += any(info["count"] < max_train for info in infos["test"])
        crowded_test += any(info["count"] > max_train for info in infos["test"])
        subset_test += any(len(set(info["directions"])) < 4 for info in infos["test"])
        varied_backgrounds += len(puzzle_backgrounds) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(object_counts) >= 10
    assert len(palettes) >= 30 and len(layouts) >= 150
    assert complex_train >= 42 and complex_test >= 5
    assert no_clip >= 30 and multi_clip >= 10 and long_visible >= 30
    assert clip_sides == {"top", "bottom", "left", "right"}
    assert edge_landing >= 10 and one_beyond >= 20
    assert equal_ties >= 42 and repeated_side >= 5
    assert thirty_side >= 5 and one_pair >= 5 and five_six >= 10
    assert varied_backgrounds == 42
    assert sparse_test >= 8 and crowded_test >= 8 and subset_test >= 8
    explicit = generate_puzzle(123456, 7, 3)
    assert _puzzles_equal(explicit, generate_puzzle(123456, 7, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(123457, 7, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(200000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for index, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(300000 + index, *counts)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            assert requested is None or actual == requested
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (2.0, 1), (3, True)):
        try:
            generate_puzzle(400000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
