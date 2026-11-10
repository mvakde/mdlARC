"""Generate variations of the corner-ray and clockwise-staircase ARC task."""

import numpy as np


DIAGONALS = ("nw", "ne", "se", "sw")
STAIRS = ("top", "right", "bottom", "left")
NAMES = DIAGONALS + STAIRS
SOURCE_ROLES = (0, 8, 5, 2, 4)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _inside(shape, point):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _walk(shape, origin, cycle):
    point = origin
    path = []
    for step in range(4 * (shape[0] + shape[1] + 2)):
        dr, dc = cycle[step % len(cycle)]
        point = (point[0] + dr, point[1] + dc)
        if not _inside(shape, point):
            return path
        path.append(point)
    raise AssertionError("route did not leave the grid")


def _routes(shape, anchor):
    r, c = anchor
    return {
        "nw": _walk(shape, (r, c), ((-1, -1),)),
        "ne": _walk(shape, (r, c + 2), ((-1, 1),)),
        "se": _walk(shape, (r + 2, c + 2), ((1, 1),)),
        "sw": _walk(shape, (r + 2, c), ((1, -1),)),
        "top": _walk(shape, (r, c + 1), ((-1, 0), (-1, 0), (0, 1), (0, 1))),
        "right": _walk(shape, (r + 1, c + 2), ((0, 1), (0, 1), (1, 0), (1, 0))),
        "bottom": _walk(shape, (r + 2, c + 1), ((1, 0), (1, 0), (0, -1), (0, -1))),
        "left": _walk(shape, (r + 1, c), ((0, -1), (0, -1), (-1, 0), (-1, 0))),
    }


def _flanks(previous, destination):
    return (previous[0], destination[1]), (destination[0], previous[1])


def _trace(routes, blockers, anchor):
    origins = {
        "nw": anchor,
        "ne": (anchor[0], anchor[1] + 2),
        "se": (anchor[0] + 2, anchor[1] + 2),
        "sw": (anchor[0] + 2, anchor[1]),
    }
    visible = {}
    stops = {}
    causes = {}
    for name, path in routes.items():
        prefix = []
        previous = origins.get(name)
        stop = None
        cause = None
        for index, point in enumerate(path):
            if point in blockers:
                stop, cause = index, "destination"
                break
            if name in DIAGONALS:
                a, b = _flanks(previous, point)
                if a in blockers and b in blockers:
                    stop, cause = index, "corner"
                    break
                previous = point
            prefix.append(point)
        visible[name] = prefix
        stops[name] = stop
        causes[name] = cause
    return visible, stops, causes


def _solid_blocker_square(blockers, shape):
    for r in range(shape[0] - 2):
        for c in range(shape[1] - 2):
            if all((y, x) in blockers for y in range(r, r + 3) for x in range(c, c + 3)):
                return True
    return False


def _geometry(rng, mode, split):
    if mode in ("demo", "corner"):
        h = int(rng.integers(19, 27))
        w = int(rng.integers(19, 27))
        r = int(rng.integers(7, h - 9))
        c = int(rng.integers(7, w - 9))
        return (h, w), (r, c)

    large = split == "test" and rng.random() < 0.42
    size_mode = int(rng.integers(0, 6))
    if large:
        h = int(rng.integers(21, 31))
        w = int(rng.integers(21, 31))
    elif size_mode == 0:
        h = int(rng.integers(7, 16))
        w = int(rng.integers(7, 16))
    elif size_mode == 1:
        h = int(rng.integers(7, 17))
        w = int(rng.integers(19, 31))
    elif size_mode == 2:
        h = int(rng.integers(19, 31))
        w = int(rng.integers(7, 17))
    elif size_mode == 3:
        h = int(rng.integers(10, 31))
        w = h
    elif size_mode == 4:
        h = 30
        w = int(rng.integers(10, 31))
    else:
        h = int(rng.integers(10, 31))
        w = 30

    placement = int(rng.integers(0, 6))
    if placement == 0:
        r = int(rng.integers(max(0, h // 3 - 1), min(h - 2, 2 * h // 3)))
        c = int(rng.integers(max(0, w // 3 - 1), min(w - 2, 2 * w // 3)))
    elif placement == 1:
        r = int(rng.choice((0, h - 3)))
        c = int(rng.integers(0, w - 2))
    elif placement == 2:
        r = int(rng.integers(0, h - 2))
        c = int(rng.choice((0, w - 3)))
    else:
        r = int(rng.integers(0, h - 2))
        c = int(rng.integers(0, w - 2))
    return (h, w), (r, c)


def _stop_index(rng, length):
    band = int(rng.integers(0, 4))
    if band == 0:
        candidates = range(0, min(3, length))
    elif band == 1:
        candidates = range(min(3, length), min(6, length))
    elif band == 2:
        candidates = range(min(6, length), length)
    else:
        candidates = range(length)
    candidates = list(candidates)
    if not candidates:
        candidates = list(range(length))
    return int(rng.choice(candidates))


def _blocker_plan(rng, routes, mode):
    plan = {name: None for name in NAMES}
    if mode == "demo":
        return plan
    if mode == "corner":
        turn = int(rng.integers(0, 4))
        plan[DIAGONALS[turn]] = ("corner", 2)
        plan[DIAGONALS[(turn + 2) % 4]] = ("destination", 0)
        plan[STAIRS[turn]] = ("destination", 0)
        plan[STAIRS[(turn + 2) % 4]] = ("destination", 1)
        return plan
    special = float(rng.random())
    if special < 0.045:
        return {name: ("destination", 0) if routes[name] else None
                for name in NAMES}
    if special < 0.12:
        silent = DIAGONALS if rng.random() < 0.5 else STAIRS
        for name in silent:
            if routes[name]:
                plan[name] = ("destination", 0)
        return plan
    available = [name for name in NAMES if routes[name]]
    n = int(rng.integers(0, len(available) + 1))
    for name in rng.choice(available, size=n, replace=False):
        name = str(name)
        plan[name] = ("destination", _stop_index(rng, len(routes[name])))
    return plan


def _blockers(rng, shape, anchor, routes, plan, density):
    h, w = shape
    emitter = {(r, c) for r in range(anchor[0], anchor[0] + 3)
               for c in range(anchor[1], anchor[1] + 3)}
    protected = set(emitter)
    blockers = set()
    flank_pairs = []
    origins = {
        "nw": anchor, "ne": (anchor[0], anchor[1] + 2),
        "se": (anchor[0] + 2, anchor[1] + 2),
        "sw": (anchor[0] + 2, anchor[1]),
    }
    for name, path in routes.items():
        stopping = plan[name]
        end = len(path) if stopping is None else stopping[1]
        protected.update(path[:end])
        if stopping is None or stopping[0] == "corner":
            protected.update(path[end:end + 1] if stopping is not None else ())
        if name in DIAGONALS:
            prev = origins[name]
            for point in path[:end]:
                flank_pairs.append(_flanks(prev, point))
                prev = point
        if stopping is not None:
            kind, index = stopping
            if kind == "destination":
                blockers.add(path[index])
            else:
                prev = origins[name] if index == 0 else path[index - 1]
                blockers.update(_flanks(prev, path[index]))

    if blockers & protected or blockers & emitter:
        raise ValueError("incompatible strategic blockers")

    # Extra cells behind first stops make the first-collision rule observable.
    stopped = [name for name in NAMES if plan[name] is not None
               and plan[name][1] + 1 < len(routes[name])]
    rng.shuffle(stopped)
    for name in stopped[:int(rng.integers(0, min(4, len(stopped)) + 1))]:
        later = routes[name][plan[name][1] + 1:]
        point = later[int(rng.integers(0, len(later)))]
        if point not in protected:
            blockers.add(point)

    target = max(len(blockers), int(round(density * h * w)))
    target = min(target, h * w - len(protected))
    candidates = [(r, c) for r in range(h) for c in range(w)
                  if (r, c) not in protected and (r, c) not in blockers]
    rng.shuffle(candidates)

    def allowed(point):
        for a, b in flank_pairs:
            if point == a and b in blockers or point == b and a in blockers:
                return False
        blockers.add(point)
        bad = _solid_blocker_square(blockers, shape)
        blockers.remove(point)
        return not bad

    while len(blockers) < target and candidates:
        if blockers and rng.random() < 0.45:
            neighbor = rng.choice(tuple(blockers))
            r, c = int(neighbor[0]), int(neighbor[1])
            near = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            rng.shuffle(near)
            point = next((p for p in near if p in candidates and allowed(p)), None)
            if point is not None:
                candidates.remove(point)
                blockers.add(point)
                continue
        point = candidates.pop()
        if allowed(point):
            blockers.add(point)
    if not blockers or _solid_blocker_square(blockers, shape):
        raise ValueError("could not make a unique blocker layout")
    return blockers


def _colors(rng, traces, previous=None):
    available = [c for c in range(10) if c not in traces]
    for _ in range(20):
        visible = tuple(int(c) for c in rng.choice(available, size=3, replace=False))
        if visible != previous and visible + traces != SOURCE_ROLES:
            return visible
    raise AssertionError("could not assign visible colors")


def _draw(shape, anchor, blockers, visible, traces):
    background, emitter, marker = visible
    grid = np.full(shape, background, dtype=np.int8)
    r, c = anchor
    grid[r:r + 3, c:c + 3] = emitter
    for point in blockers:
        grid[point] = marker
    routes = _routes(shape, anchor)
    paths, _, _ = _trace(routes, blockers, anchor)
    output = grid.copy()
    for name in DIAGONALS:
        for point in paths[name]:
            output[point] = traces[0]
    for name in STAIRS:
        for point in paths[name]:
            output[point] = traces[1]
    return {"input": grid, "output": output}


def _make_pair(rng, mode, split, traces, previous_colors):
    for _ in range(30):
        shape, anchor = _geometry(rng, mode, split)
        routes = _routes(shape, anchor)
        plan = _blocker_plan(rng, routes, mode)
        if mode == "corner":
            gap_name = next(name for name in DIAGONALS
                            if plan[name] is not None and plan[name][0] == "corner")
            near_name = DIAGONALS[(DIAGONALS.index(gap_name) + 1) % 4]
            short_stair = next(name for name in STAIRS
                               if plan[name] == ("destination", 1))
            if (len(routes[gap_name]) < 3 or len(routes[near_name]) < 3
                    or len(routes[short_stair]) < 2):
                continue
        if mode == "demo":
            density = float(rng.uniform(0.025, 0.075))
        elif mode == "corner":
            density = float(rng.uniform(0.065, 0.13))
        elif split == "test" and rng.random() < 0.4:
            density = float(rng.uniform(0.10, 0.22))
        else:
            density = float(rng.uniform(0.015, 0.19))
        try:
            blockers = _blockers(rng, shape, anchor, routes, plan, density)
        except ValueError:
            continue
        actual, stops, causes = _trace(routes, blockers, anchor)
        if mode == "demo" and any(stops[name] is not None for name in NAMES):
            continue
        if mode == "corner":
            if causes[gap_name] != "corner" or stops[gap_name] != 2:
                continue
            if any(stops[name] != entry[1]
                   for name, entry in plan.items()
                   if entry is not None and name != gap_name):
                continue
            # A blocker on only one orthogonal flank must leave this
            # different diagonal passable.
            previous = routes[near_name][1]
            point = routes[near_name][2]
            a, b = _flanks(previous, point)
            if a not in blockers and b not in blockers:
                for flank in (a, b):
                    trial = blockers | {flank}
                    trial_paths, trial_stops, _ = _trace(routes, trial, anchor)
                    if (not _solid_blocker_square(trial, shape)
                            and len(trial_paths[near_name]) >= 3
                            and trial_stops[gap_name] == 2):
                        blockers = trial
                        break
            if not (a in blockers) != (b in blockers):
                continue
            if len(_trace(routes, blockers, anchor)[0][near_name]) < 3:
                continue
        visible = _colors(rng, traces, previous_colors)
        return _draw(shape, anchor, blockers, visible, traces)
    raise ValueError("pair construction exhausted its bounded attempts")


def _key(grid):
    return grid.shape, grid.astype(np.int16, copy=False).tobytes()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    traces = tuple(int(c) for c in rng.choice(10, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = ("demo" if index == 0 else "corner" if index == 1 else "random") if split == "train" else "random"
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(pair_rng, mode, split, traces, previous)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise ValueError("could not produce distinct inputs")
            seen.add(key)
            previous = _analyze_input(pair["input"])["roles"]
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 3
    background = int(colors[int(np.argmax(counts))])
    h, w = grid.shape
    candidates = []
    for value in colors:
        color = int(value)
        if color == background:
            continue
        for r in range(h - 2):
            for c in range(w - 2):
                if np.all(grid[r:r + 3, c:c + 3] == color):
                    candidates.append((r, c, color))
    assert len(candidates) == 1
    r, c, emitter = candidates[0]
    assert int(np.count_nonzero(grid == emitter)) == 9
    marker = next(int(value) for value in colors
                  if value != background and value != emitter)
    blockers = {(int(y), int(x)) for y, x in np.argwhere(grid == marker)}
    assert blockers and not _solid_blocker_square(blockers, grid.shape)
    routes = _routes(grid.shape, (r, c))
    all_route_cells = [point for path in routes.values() for point in path]
    assert len(all_route_cells) == len(set(all_route_cells))
    actual, stops, causes = _trace(routes, blockers, (r, c))
    return {
        "shape": grid.shape, "anchor": (r, c),
        "roles": (background, emitter, marker),
        "blockers": blockers, "routes": routes,
        "actual": actual, "stops": stops, "causes": causes,
    }


def _family_cells(info, names):
    return {point for name in names for point in info["actual"][name]}


def _predict_from_input(grid, traces):
    info = _analyze_input(grid)
    expected = grid.copy()
    for name in DIAGONALS:
        for point in info["actual"][name]:
            expected[point] = traces[0]
    for name in STAIRS:
        for point in info["actual"][name]:
            expected[point] = traces[1]
    return expected


def _has_one_flank_pass(info):
    blockers = info["blockers"]
    origins = {
        "nw": info["anchor"],
        "ne": (info["anchor"][0], info["anchor"][1] + 2),
        "se": (info["anchor"][0] + 2, info["anchor"][1] + 2),
        "sw": (info["anchor"][0] + 2, info["anchor"][1]),
    }
    for name in DIAGONALS:
        previous = origins[name]
        for point in info["actual"][name]:
            a, b = _flanks(previous, point)
            if (a in blockers) != (b in blockers):
                return True
            previous = point
    return False


def _later_blocker_count(info):
    result = 0
    for name in NAMES:
        stop = info["stops"][name]
        if stop is not None:
            result += sum(point in info["blockers"]
                          for point in info["routes"][name][stop + 1:])
    return result


def _check_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            input_key = _key(inp)
            pair_key = (input_key, _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_input(inp))

    # The first training example visibly supplies both output color roles.
    exemplar = puzzle["train"][0]
    first = analyses["train"][0]
    diagonal = _family_cells(first, DIAGONALS)
    staircase = _family_cells(first, STAIRS)
    assert diagonal and staircase
    dcolors = {int(exemplar["output"][point]) for point in diagonal}
    scolors = {int(exemplar["output"][point]) for point in staircase}
    assert len(dcolors) == len(scolors) == 1
    traces = (dcolors.pop(), scolors.pop())
    assert traces[0] != traces[1]
    assert all(color not in set(int(c) for c in np.unique(exemplar["input"]))
               for color in traces)

    for split in ("train", "test"):
        for pair, info in zip(puzzle[split], analyses[split]):
            assert all(color not in info["roles"] for color in traces)
            assert info["roles"] + traces != SOURCE_ROLES
            expected = _predict_from_input(pair["input"], traces)
            assert np.array_equal(pair["output"], expected)
            assert all(pair["output"][point] == pair["input"][point]
                       for point in info["blockers"])
            assert np.all(pair["output"][info["anchor"][0]:info["anchor"][0] + 3,
                                          info["anchor"][1]:info["anchor"][1] + 3]
                          == info["roles"][1])

    # Every origin, clockwise turn, second cycle, and both output colors are
    # demonstrated by training, even for a requested two-example puzzle.
    assert all(any(len(info["actual"][name]) >= 2 for info in analyses["train"])
               for name in DIAGONALS)
    assert all(any(len(info["actual"][name]) >= 8 for info in analyses["train"])
               for name in STAIRS)
    assert any(info["causes"][name] == "corner"
               for info in analyses["train"] for name in DIAGONALS)
    assert any(_has_one_flank_pass(info) for info in analyses["train"])
    assert any(info["causes"][name] == "destination"
               for info in analyses["train"] for name in DIAGONALS)
    assert any(info["causes"][name] == "destination"
               for info in analyses["train"] for name in STAIRS)
    assert any(info["stops"][name] is None
               for info in analyses["train"] for name in DIAGONALS)
    assert any(info["stops"][name] is None
               for info in analyses["train"] for name in STAIRS)
    return analyses, traces


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _check_puzzle(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid puzzle: {exc}") from exc


def _equal_puzzles(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[split], b[split]))
        for split in ("train", "test"))


def self_test() -> None:
    # This constructed case distinguishes corner closure from a blocked
    # destination. The northwest destination is clear, but both flanks are 5.
    sample_routes = _routes((12, 12), (5, 5))
    prev, dest = sample_routes["nw"][0], sample_routes["nw"][1]
    a, b = _flanks(prev, dest)
    paths, stops, causes = _trace(sample_routes, {a, b}, (5, 5))
    assert stops["nw"] == 1 and causes["nw"] == "corner"
    assert paths["nw"] == [prev]
    paths, stops, _ = _trace(sample_routes, {a}, (5, 5))
    assert len(paths["nw"]) > 1 and stops["nw"] is None

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures = set()
    role_pairs = set()
    shapes = set()
    positions = set()
    blocked_counts = set()
    densities = []
    later = 0
    coverage = {
        "zero": 0, "one": 0, "mid": 0, "late": 0,
        "corner": 0, "one_flank": 0, "multi": 0,
        "cluster": 0, "no_diagonal": 0, "no_stair": 0,
        "unchanged": 0, "border": 0, "square": 0,
        "wide": 0, "tall": 0, "full": 0,
        "large_dense_test": 0, "dense_long": 0,
        "sparse_blocked": 0,
    }
    quadrants = set()
    smaller_tests = 0
    larger_tests = 0
    sparser_tests = 0
    denser_tests = 0
    for puzzle in puzzles:
        analyses, traces = _check_puzzle(puzzle)
        role_pairs.add(traces)
        signatures.add(tuple(_key(pair["input"]) for split in ("train", "test")
                             for pair in puzzle[split]))
        all_infos = analyses["train"] + analyses["test"]
        assert len({info["roles"] for info in all_infos}) >= 2
        train_areas = [info["shape"][0] * info["shape"][1]
                       for info in analyses["train"]]
        test_areas = [info["shape"][0] * info["shape"][1]
                      for info in analyses["test"]]
        train_density = [len(info["blockers"]) /
                         (info["shape"][0] * info["shape"][1])
                         for info in analyses["train"]]
        test_density = [len(info["blockers"]) /
                        (info["shape"][0] * info["shape"][1])
                        for info in analyses["test"]]
        smaller_tests += any(area < max(train_areas) for area in test_areas)
        larger_tests += any(area > max(train_areas) for area in test_areas)
        sparser_tests += any(d < min(train_density) for d in test_density)
        denser_tests += any(d > max(train_density) for d in test_density)
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], analyses[split]):
                h, w = info["shape"]
                r, c = info["anchor"]
                shapes.add(info["shape"])
                positions.add((info["shape"], info["anchor"]))
                quadrants.add((r > (h - 3) / 2, c > (w - 3) / 2))
                blocked_counts.add(sum(info["stops"][name] is not None
                                       for name in NAMES))
                density = len(info["blockers"]) / (h * w)
                densities.append(density)
                later += _later_blocker_count(info)
                lengths = [len(info["actual"][name]) for name in NAMES]
                stops = info["stops"].values()
                coverage["zero"] += any(length == 0 for length in lengths)
                coverage["one"] += any(length == 1 for length in lengths)
                coverage["mid"] += any(stop is not None and 3 <= stop <= 5
                                       for stop in stops)
                coverage["late"] += any(stop is not None and stop >= 6
                                        for stop in info["stops"].values())
                coverage["corner"] += any(cause == "corner"
                                          for cause in info["causes"].values())
                coverage["one_flank"] += _has_one_flank_pass(info)
                coverage["multi"] += any(sum(point in info["blockers"]
                                              for point in path) >= 2
                                          for path in info["routes"].values())
                coverage["cluster"] += any(
                    (y + 1, x) in info["blockers"] or
                    (y, x + 1) in info["blockers"]
                    for y, x in info["blockers"])
                coverage["no_diagonal"] += all(
                    not info["actual"][name] for name in DIAGONALS)
                coverage["no_stair"] += all(
                    not info["actual"][name] for name in STAIRS)
                coverage["unchanged"] += sum(lengths) == 0
                coverage["border"] += r == 0 or c == 0 or r == h - 3 or c == w - 3
                coverage["square"] += h == w
                coverage["wide"] += w >= h + 8
                coverage["tall"] += h >= w + 8
                coverage["full"] += 30 in (h, w)
                coverage["large_dense_test"] += (split == "test"
                                                  and min(h, w) >= 21
                                                  and density >= 0.10)
                coverage["dense_long"] += density >= 0.12 and sum(lengths) >= 20
                coverage["sparse_blocked"] += (density <= 0.06 and
                                                sum(stop is not None for stop
                                                    in info["stops"].values()) >= 5)
                # Reconstruct using only the input and colors learned from
                # the training labels, with no test metadata or test output.
                assert np.array_equal(_predict_from_input(pair["input"], traces),
                                      pair["output"])
    assert len(signatures) >= 40
    assert len(role_pairs) >= 20
    assert len(shapes) >= 50 and len(positions) >= 100
    assert len(quadrants) == 4
    assert min(densities) < 0.07 and max(densities) > 0.14
    assert later > 0
    assert blocked_counts == set(range(9))
    assert all(value > 0 for value in coverage.values())
    assert smaller_tests and larger_tests and sparser_tests and denser_tests
    assert _equal_puzzles(generate_puzzle(314159, 6, 2),
                          generate_puzzle(314159, 6, 2))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + index, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for index, (train, test) in enumerate(((None, 1), (None, 4),
                                            (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + index, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11},
                 {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
