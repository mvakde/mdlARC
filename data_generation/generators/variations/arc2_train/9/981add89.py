"""Synthetic boundary-stencil puzzles with conditional, full-length rays."""

import hashlib
import numpy as np


_ORIENTATIONS = ("top", "right", "bottom", "left")
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))
_SOURCE_PAIR_DIGESTS = {
    ((20, 20), "d0a78ecc91667407682466b63ea5ccb52a9bde9afdd741662076aa24312c9568"),
    ((30, 30), "431c9dd51315b2e9ac0d90ac8477eda138b8be8ade69dbc7e55fdb2a051dc447"),
    ((30, 30), "5662536ee101a110982b2f117b4f83b8bec6d05b90962c3e27ed85de1121196c"),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = test + 1 + seed % (10 - test)
    if test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _rotate(grid, orientation):
    return np.rot90(grid, {"top": 0, "right": 3, "bottom": 2, "left": 1}[orientation]).copy()


def _edge(grid, orientation):
    if orientation == "top":
        return grid[0]
    if orientation == "bottom":
        return grid[-1]
    if orientation == "left":
        return grid[:, 0]
    return grid[:, -1]


def _background(grid):
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    maximum = int(counts.max())
    _require(np.count_nonzero(counts == maximum) == 1, "background is not uniquely modal")
    return int(np.argmax(counts))


def _project(grid, orientation, background):
    """The marker itself is fixed; matching cells beyond it are erased."""
    result = grid.copy()
    edge = _edge(grid, orientation)
    for position in np.flatnonzero(edge != background):
        color = int(edge[position])
        if orientation == "top":
            old = grid[1:, position]
            result[1:, position] = np.where(old == color, background, color)
        elif orientation == "bottom":
            old = grid[:-1, position]
            result[:-1, position] = np.where(old == color, background, color)
        elif orientation == "left":
            old = grid[position, 1:]
            result[position, 1:] = np.where(old == color, background, color)
        else:
            old = grid[position, :-1]
            result[position, :-1] = np.where(old == color, background, color)
    return result


def _dimension(rng, simple):
    if simple:
        choices = (20, 20, 20, 18, 22, 24)
        if rng.random() < 0.38:
            return int(rng.choice(choices))
        return int(rng.integers(16, 27))
    if rng.random() < 0.28:
        return int(rng.choice((20, 20, 30, 30)))
    return int(rng.integers(18, 31))


def _dimensions(rng, simple, old_shapes, regime):
    if regime == "target_test" and old_shapes and rng.random() < 0.32:
        return old_shapes[int(rng.integers(len(old_shapes)))]
    depth = _dimension(rng, simple)
    if rng.random() < 0.23:
        span = depth
    else:
        span = _dimension(rng, simple)
    return depth, span


def _positions(rng, span, count):
    places = list(range(1, span - 1))
    chosen = set(int(x) for x in rng.choice(places, size=count, replace=False))
    if count >= 2 and rng.random() < 0.28:
        start = int(rng.integers(1, span - 2))
        chosen = {start, start + 1}
        rest = [p for p in places if p not in chosen]
        chosen.update(int(x) for x in rng.choice(rest, size=count - 2, replace=False))
    if rng.random() < 0.30 and 1 not in chosen:
        chosen.remove(int(rng.choice(tuple(chosen))))
        chosen.add(1)
    elif rng.random() < 0.30 and span - 2 not in chosen:
        chosen.remove(int(rng.choice(tuple(chosen))))
        chosen.add(span - 2)
    return sorted(chosen)


def _extent(rng, length, mode):
    if mode == 0:
        return int(rng.integers(2, min(7, length) + 1))
    if mode == 1:
        return int(rng.integers(max(3, length // 3), length + 1))
    return int(rng.integers(1, min(3, length) + 1))


def _complex_scene(grid, rng, colors, positions, background, regime):
    depth, span = grid.shape
    palette_size = int(rng.integers(2, 8))
    palette = rng.choice([c for c in range(10) if c != background],
                         size=palette_size, replace=False)
    target = float(rng.uniform(0.18, 0.78))
    if regime == "target_test" and rng.random() < 0.50:
        target = float(rng.uniform(0.50, 0.80))
    boxes = []
    minimum = max(3, palette_size)
    for index in range(26):
        mode = int(rng.choice((0, 0, 1, 1, 1, 2, 3)))
        if mode == 2:  # thin horizontal strip
            rh, rw = _extent(rng, depth - 1, 2), _extent(rng, span, 1)
        elif mode == 3:  # thin vertical strip
            rh, rw = _extent(rng, depth - 1, 1), _extent(rng, span, 2)
        else:
            rh = _extent(rng, depth - 1, mode)
            rw = _extent(rng, span, mode)
        rh = min(rh, depth - 1)
        rw = min(rw, span)
        if boxes and rng.random() < 0.32:
            old_row, old_col, old_h, old_w = boxes[int(rng.integers(len(boxes)))]
            row = int(np.clip(old_row + rng.integers(-rh + 1, old_h), 1, depth - rh))
            col = int(np.clip(old_col + rng.integers(-rw + 1, old_w), 0, span - rw))
        else:
            row = int(rng.integers(1, depth - rh + 1))
            col = int(rng.integers(0, span - rw + 1))
        if rng.random() < 0.14:
            col = int(rng.choice((0, span - rw)))
        if rng.random() < 0.10:
            row = depth - rh
        color = int(palette[index % palette_size]) if index < palette_size else int(rng.choice(palette))
        grid[row:row + rh, col:col + rw] = color
        boxes.append((row, col, rh, rw))
        density = float(np.mean(grid[1:] != background))
        if index + 1 >= minimum and density >= target:
            break
    if boxes and rng.random() < 0.33:
        row, col, rh, rw = boxes[int(rng.integers(len(boxes)))]
        if rh >= 4 and rw >= 4:
            hole_h = int(rng.integers(1, min(4, rh - 1)))
            hole_w = int(rng.integers(1, min(4, rw - 1)))
            hr = int(rng.integers(row + 1, row + rh - hole_h))
            hc = int(rng.integers(col + 1, col + rw - hole_w))
            grid[hr:hr + hole_h, hc:hc + hole_w] = background
    if rng.random() < 0.48:
        # A real region, rather than a single diagnostic pixel, meets a ray.
        index = int(rng.integers(len(positions)))
        position = positions[index]
        ray_color = colors[index]
        patch_h = int(rng.integers(2, min(6, depth)))
        patch_w = int(rng.integers(2, min(6, span)))
        row = int(rng.integers(1, depth - patch_h + 1))
        col = int(np.clip(position - rng.integers(0, patch_w), 0, span - patch_w))
        grid[row:row + patch_h, col:col + patch_w] = ray_color
    if rng.random() < 0.16:
        # Some emitters traverse only background, even in a busy scene.
        position = int(rng.choice(positions))
        grid[1:, position] = background


def _make_pair(rng, regime, orientation, simple, old_shapes, background):
    for _ in range(80):
        depth, span = _dimensions(rng, simple, old_shapes, regime)
        count = int(rng.choice((2, 2, 2, 3, 3, 4))) if simple else int(rng.integers(2, min(9, (span - 2) // 2 + 2)))
        if simple:
            width = int(rng.integers(5, min(13, span - 1)))
            col = int(rng.integers(1, span - width))
            core = rng.choice(np.arange(col, col + width), size=2, replace=False)
            extra = [p for p in range(1, span - 1) if p not in core]
            positions = sorted([int(x) for x in core] +
                               [int(x) for x in rng.choice(extra, size=count - 2, replace=False)])
            # The painter samples its own box; pass the chosen crossing interval.
            scene_span = (col, width)
        else:
            positions = _positions(rng, span, count)
            scene_span = None
        colors = [int(x) for x in rng.choice([c for c in range(10) if c != background],
                                              size=count, replace=True)]
        canonical = np.full((depth, span), background, dtype=np.int8)
        if simple:
            _simple_scene_with_span(canonical, rng, positions, colors, background, scene_span)
        else:
            _complex_scene(canonical, rng, colors, positions, background, regime)
        for position, color in zip(positions, colors):
            canonical[0, position] = color
        inp = _rotate(canonical, orientation)
        try:
            if _background(inp) != background:
                continue
            out = _project(inp, orientation, background)
            info = _analyze_pair(inp, out)
            if info["orientation"] != orientation:
                continue
            if simple and not (info["matching"] and info["contrasting"]):
                continue
            if not simple and len(info["passive_colors"]) < 2:
                continue
            return inp, out
        except AssertionError:
            continue
    raise RuntimeError("could not construct a valid pair")


def _simple_scene_with_span(grid, rng, positions, colors, background, scene_span):
    depth, _ = grid.shape
    col, width = scene_span
    height = int(rng.integers(3, min(11, depth - 1)))
    row = int(rng.integers(1, depth - height))
    inside = [p for p in positions if col <= p < col + width]
    matching, contrasting = rng.choice(inside, size=2, replace=False)
    color = int(rng.choice([c for c in range(10) if c != background]))
    grid[row:row + height, col:col + width] = color
    colors[positions.index(int(matching))] = color
    colors[positions.index(int(contrasting))] = int(rng.choice(
        [c for c in range(10) if c not in (background, color)]))


def _source_digest(inp, out):
    return (inp.shape, hashlib.sha256(inp.astype(np.int8).tobytes() +
                                       out.astype(np.int8).tobytes()).hexdigest())


def _control_first(grid, direction):
    return np.rot90(grid, {"top": 0, "right": 1,
                           "bottom": 2, "left": 3}[direction])


def _analyze_pair(inp, out):
    _require(inp.shape == out.shape, "shape changed")
    bg = _background(inp)
    _require(not np.array_equal(inp, out), "unchanged pair")
    possibilities = []
    for direction in _ORIENTATIONS:
        edge = _edge(inp, direction)
        points = np.flatnonzero(edge != bg)
        if len(points) < 2 or len(points) > min(9, (len(edge) - 1) // 2):
            continue
        if points[0] == 0 or points[-1] == len(edge) - 1:
            continue
        if np.array_equal(_project(inp, direction, bg), out):
            possibilities.append((direction, points))
    _require(len(possibilities) == 1, "control edge cannot be recovered uniquely")
    direction, points = possibilities[0]
    matching = contrasting = background_hits = object_free_rays = 0
    for p in points:
        color = int(_edge(inp, direction)[p])
        if direction == "top":
            ray = inp[1:, p]
        elif direction == "bottom":
            ray = inp[:-1, p]
        elif direction == "left":
            ray = inp[p, 1:]
        else:
            ray = inp[p, :-1]
        matching += int(np.count_nonzero(ray == color))
        contrasting += int(np.count_nonzero((ray != bg) & (ray != color)))
        background_hits += int(np.count_nonzero(ray == bg))
        object_free_rays += int(not np.any(ray != bg))
    canonical = _control_first(inp, direction)
    passive = canonical[1:]
    return {"orientation": direction, "background": bg,
            "emitters": tuple((int(p), int(_edge(inp, direction)[p])) for p in points),
            "matching": matching, "contrasting": contrasting,
            "background_hits": background_hits, "object_free_rays": object_free_rays,
            "passive_colors": tuple(int(c) for c in np.unique(passive) if c != bg),
            "density": float(np.mean(passive != bg)),
            "side_reach": bool(np.any(passive[:, 0] != bg) or np.any(passive[:, -1] != bg)),
            "opposite_reach": bool(np.any(passive[-1] != bg)),
            "adjacent": bool(np.any(np.diff(points) == 1)),
            "edge_near": bool(points[0] == 1 or points[-1] == len(_edge(inp, direction)) - 2),
            "square": inp.shape[0] == inp.shape[1]}


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation = _ORIENTATIONS[int(rng.integers(4))]
    simple_index = 0 if rng.random() < 0.60 else int(rng.integers(1, train_count))
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    shapes = []
    seen = set()
    # Only the direction is a learned puzzle convention. Colors, geometry, and
    # stencil density are visible local properties and vary from pair to pair.
    for index, pair_seed in enumerate(pair_seeds):
        split = "train" if index < train_count else "test"
        simple = index == simple_index
        pair_rng = np.random.default_rng(int(pair_seed))
        for _ in range(20):
            background = int(pair_rng.integers(10))
            inp, out = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                                  orientation, simple, shapes, background)
            key = _grid_key(inp)
            if key not in seen and _source_digest(inp, out) not in _SOURCE_PAIR_DIGESTS:
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        seen.add(key)
        puzzle[split].append({"input": inp, "output": out})
        if split == "train":
            shapes.append((inp.shape[0], inp.shape[1]) if orientation in ("top", "bottom")
                          else (inp.shape[1], inp.shape[0]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid array")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid size out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype is not integral")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal grid color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "dimensions changed")
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            _require(key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(_source_digest(inp, out) not in _SOURCE_PAIR_DIGESTS, "source pair copied")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            infos[split].append(_analyze_pair(inp, out))
    directions = {x["orientation"] for s in infos.values() for x in s}
    _require(len(directions) == 1, "direction changed within puzzle")
    _require(any(x["matching"] > 0 for x in infos["train"]), "training omits same-color erasure")
    _require(any(x["contrasting"] > 0 for x in infos["train"]), "training omits contrasting recolor")
    _require(any(x["background_hits"] > 0 for x in infos["train"]), "training omits extension")
    simple_demonstrations = 0
    for pair, information in zip(puzzle["train"], infos["train"]):
        if len(information["passive_colors"]) != 1:
            continue
        scene = _control_first(pair["input"], information["orientation"])
        mask = scene[1:] != information["background"]
        coordinates = np.argwhere(mask)
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        if (int(mask.sum()) == int((bottom - top + 1) * (right - left + 1))
                and information["matching"] > 0 and information["contrasting"] > 0):
            simple_demonstrations += 1
    _require(simple_demonstrations >= 1,
             "training needs a simple rectangle with erasure and recoloring")
    for pair in puzzle["test"]:
        prediction = _prediction_from_training(puzzle["train"], pair["input"])
        _require(np.array_equal(prediction, pair["output"]),
                 "test output cannot be inferred from training and input")


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(p["input"]), _grid_key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _prediction_from_training(train, test_input):
    """Infer the edge from labeled training grids, then read test markers."""
    possible = set(_ORIENTATIONS)
    for pair in train:
        inp, out = pair["input"], pair["output"]
        background = _background(inp)
        supported = set()
        for direction in _ORIENTATIONS:
            edge = _edge(inp, direction)
            if 2 <= np.count_nonzero(edge != background) <= 9 and np.array_equal(
                    _project(inp, direction, background), out):
                supported.add(direction)
        possible &= supported
    _require(len(possible) == 1, "training fails to identify the direction")
    direction = next(iter(possible))
    background = _background(test_input)
    output = test_input.copy()
    edge = _edge(test_input, direction)
    for position in np.flatnonzero(edge != background):
        color = int(edge[position])
        if direction == "top":
            coordinates = ((r, int(position)) for r in range(1, test_input.shape[0]))
        elif direction == "bottom":
            coordinates = ((r, int(position)) for r in range(test_input.shape[0] - 1))
        elif direction == "left":
            coordinates = ((int(position), c) for c in range(1, test_input.shape[1]))
        else:
            coordinates = ((int(position), c) for c in range(test_input.shape[1] - 1))
        for row, col in coordinates:
            output[row, col] = background if test_input[row, col] == color else color
    return output


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            prediction = _prediction_from_training(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]], pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]),
                     "a test output is not inferred from training and its input")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "seed diversity failed")
    infos = [[_analyze_pair(p["input"], p["output"])
              for split in ("train", "test") for p in puzzle[split]] for puzzle in puzzles]
    _require({p[0]["orientation"] for p in infos} == set(_ORIENTATIONS), "direction diversity failed")
    _require(len({pair["input"].shape for puzzle in puzzles for split in ("train", "test")
                  for pair in puzzle[split]}) >= 45, "shape diversity failed")
    _require(len({i["emitters"] for group in infos for i in group}) >= 100,
             "stencil diversity failed")
    _require(any(i["square"] for group in infos for i in group), "no square grid")
    _require(any(i["side_reach"] for group in infos for i in group), "no side-reaching scene")
    _require(any(i["density"] >= 0.60 for group in infos for i in group), "no dense scene")
    _require(any(i["adjacent"] for group in infos for i in group), "no adjacent markers")
    _require(any(i["edge_near"] for group in infos for i in group), "no edge-near markers")
    _require(any(i["object_free_rays"] for group in infos for i in group),
             "no emitter crosses only background")
    _require(any(len({c for _, c in i["emitters"]}) < len(i["emitters"])
                 for group in infos for i in group), "no repeated emitter colors")
    _require(any(i["opposite_reach"] for group in infos for i in group),
             "no opposite-edge passive scene")
    _require(any(i["matching"] == 0 for group in infos for i in group[1:]),
             "every complex scene forces same-color erasure")
    _require(any(i["density"] < 0.25 for group in infos for i in group) and
             any(i["density"] > 0.65 for group in infos for i in group),
             "passive densities lack range")
    _require(any(p["input"].shape == (20, 20) for puzzle in puzzles
                 for split in ("train", "test") for p in puzzle[split]),
             "missing source-like 20-square scale")
    _require(any(p["input"].shape == (30, 30) for puzzle in puzzles
                 for split in ("train", "test") for p in puzzle[split]),
             "missing source-like 30-square scale")
    _require(all(len({len(i["emitters"]) for i in group}) > 1 for group in infos),
             "emitter count is fixed inside a puzzle")
    _require(all(len({i["background"] for i in group}) > 1 for group in infos),
             "background is fixed inside a puzzle")
    _require(all(len({p["input"].shape for split in ("train", "test")
                      for p in puzzle[split]}) > 1 for puzzle in puzzles),
             "grid shape is fixed inside a puzzle")
    _require(any(len({i["background"] for i in group}) > 1 for group in infos),
             "background roles never vary within a puzzle")
    _require(any(min(len(i["emitters"]) for i in group[2:]) <=
                 max(len(i["emitters"]) for i in group[:2]) for group in infos),
             "tests are always the busiest")
    _require(all(any(i["matching"] > 0 for i in group[:len(puzzle["train"])])
                 for group, puzzle in zip(infos, puzzles)),
             "a training set omits same-color erasure")
    _require(any(set(p["input"].shape for p in puzzle["train"]) &
                 set(p["input"].shape for p in puzzle["test"])
                 for puzzle in puzzles), "train and test never share a scale")
    simple_indices = [next(index for index, information in enumerate(
        group[:len(puzzle["train"])]) if len(information["passive_colors"]) == 1)
        for group, puzzle in zip(infos, puzzles)]
    _require(len(set(simple_indices)) >= 3, "simple demonstration order lacks diversity")
    shared_backgrounds = new_backgrounds = swapped_backgrounds = reverse_swaps = 0
    for group, puzzle in zip(infos, puzzles):
        training = group[:len(puzzle["train"])]
        testing = group[len(puzzle["train"]):]
        training_backgrounds = {i["background"] for i in training}
        training_emitter_colors = {color for i in training for _, color in i["emitters"]}
        shared_backgrounds += int(any(i["background"] in training_backgrounds for i in testing))
        new_backgrounds += int(any(i["background"] not in training_backgrounds for i in testing))
        swapped_backgrounds += int(any(i["background"] in training_emitter_colors for i in testing))
        reverse_swaps += int(any(any(color in training_backgrounds for _, color in i["emitters"])
                                 for i in testing))
    _require(min(shared_backgrounds, new_backgrounds, swapped_backgrounds, reverse_swaps) >= 5,
             "palette role schedules lack diversity")
    _require(any(i["density"] >= 0.60 and i["side_reach"] and i["adjacent"]
                 for group in infos for i in group),
             "dense scenes never combine with boundary and stencil variation")
    sample = generate_puzzle(314159, 7, 3)
    _require(_puzzle_key(sample) == _puzzle_key(generate_puzzle(314159, 7, 3)),
             "non-deterministic generation")
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(8000 + train * 10 + test, train, test)
        _require((len(p["train"]), len(p["test"])) == (train, test), "explicit count ignored")
    for index, kwargs in enumerate(({ "num_train": None, "num_test": 1},
                                    {"num_train": None, "num_test": 4},
                                    {"num_train": 2, "num_test": None},
                                    {"num_train": 10, "num_test": None})):
        p = generate_puzzle(9000 + index, **kwargs)
        _require(all(kwargs[name] is None or len(p[split]) == kwargs[name]
                     for name, split in (("num_train", "train"), ("num_test", "test"))),
                 "partial count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
