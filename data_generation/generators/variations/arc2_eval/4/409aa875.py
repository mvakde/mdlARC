"""Three-cell arrows fire simultaneously; conventions are learned per puzzle."""
import numpy as np


_DIRECTION_VECTORS = {
    "N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1),
    "NW": (-1, -1), "NE": (-1, 1), "SW": (1, -1), "SE": (1, 1),
}
_DIRECTIONS = tuple(_DIRECTION_VECTORS)

# Exact source pairs, used only for regression and exclusion.
_SOURCE_PAIRS = (('7777777777777777777777777777777777777777777777777777777777779777777777777777997777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777779777779777777979797779797777979', '7777777777777777777777777777777777777777777777777777777777779777777777777777997777777777777777777777777777777777777777777777777777777777777777777977777177777797777777777777777777777777777777777777777777777777777777777777777779777779777777979797779797777979'), ('7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777707777777777777707077777077777777777077070777777777070777770777770777770770707770707770707777777', '7777777777777777777777777777777777777777777777777777777777777777777777777777777779777777777777777777777797777777777797777777777777777777777977777977777977777777707777777777777707077777077777777777077070777777777070777770777770777770770707770707770707777777'), ('7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777722722722722777772772772772777777777777777777777777777777777777777777777777777777777777772277777777777777277777777777777777777777227227227227777727727727727', '7777777777777777977977977977777777777777777777777777777777777777777777777777777777777777777777777777722799722722777772779772772777777777777777779779779779777777777777777777777777777777777772277777777777777277777777777777777777777227227227227777727727727727'), ('7777777777777777777777777777777777677777676777777667777776777767777777777777767777777777767777677777777767677777776777777777777776767777777777777777777777777777777777777777777777777777777777777777777777777777777767777777777777767677667777677777777776777676', '7777777779777777777777777777777777977777676777777997777776777767777777779777767777777777767777677777777767677777776777777777777776769779797777777777777777777717777777777777777777777777777777777777777777777777777767777777777777767677667777677777777776777676'))


def _check_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    return num_train, num_test


def _arrow(tip, direction):
    r, c = tip
    dr, dc = _DIRECTION_VECTORS[direction]
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (r, c, direction, tuple(sorted((r+a, c+b) for a, b in offsets)))


def _endpoint(obj, distance):
    dr, dc = _DIRECTION_VECTORS[obj[2]]
    return (obj[0] + distance * dr, obj[1] + distance * dc)


def _compatible(objects):
    # A blank king-neighbor gap makes decomposition unique, even across colors.
    occupied = set()
    for obj in objects:
        if any((r+a, c+b) in occupied for r, c in obj[3]
               for a in (-1, 0, 1) for b in (-1, 0, 1)):
            return False
        occupied.update(obj[3])
    return True


def _events(objects, distance, shape=None):
    owner = {cell: i for i, obj in enumerate(objects) for cell in obj[3]}
    assert len(owner) == 3 * len(objects), "Overlapping arrows"
    counts = {}
    for obj in objects:
        point = _endpoint(obj, distance)
        counts[point] = counts.get(point, 0) + 1
    if shape is not None:
        assert all(0 <= r < shape[0] and 0 <= c < shape[1]
                   for r, c in list(owner) + list(counts)), "Out-of-bounds shot"
    hits = {}
    for point, count in counts.items():
        if point in owner:
            hits.setdefault(owner[point], []).append((point, count))
    # Two separate single endpoints do not constitute a collision.
    assert all(len({n > 1 for _, n in points}) == 1 for points in hits.values()), \
        "Incompatible colors assigned to one arrow"
    paint = ({}, {})
    for point, count in counts.items():
        cells = objects[owner[point]][3] if point in owner else (point,)
        for cell in cells:
            paint[int(count > 1)][cell] = True
    return {"owner": owner, "counts": counts, "hits": hits, "paint": paint,
            "single_background": [p for p, n in counts.items() if n == 1 and p not in owner],
            "collision_sites": sum(n > 1 for n in counts.values()),
            "struck": len(hits)}


def _background(grid):
    return int(np.bincount(grid.ravel(), minlength=10).argmax())


def _parse_input(grid):
    background = _background(grid)
    remaining = set(map(tuple, np.argwhere(grid != background).tolist()))
    assert remaining and len(remaining) < grid.size / 2
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            neighbors = {(r+a, c+b) for a in (-1, 0, 1) for b in (-1, 0, 1)} & remaining
            remaining -= neighbors
            component |= neighbors
            todo.extend(neighbors)
        assert len(component) == 3, "Input motifs must be separated triominoes"
        assert len({int(grid[p]) for p in component}) == 1, "Each arrow is monochrome"
        matches = [_arrow(tip, direction) for tip in sorted(component)
                   for direction in _DIRECTIONS
                   if set(_arrow(tip, direction)[3]) == component]
        assert len(matches) == 1, "Ambiguous arrow geometry"
        objects.append(matches[0])
    return background, objects


def _apply(grid, objects, distance, single, collision):
    stats = _events(objects, distance, grid.shape)
    out = grid.copy()
    for color, cells in zip((single, collision), stats["paint"]):
        for p in cells:
            out[p] = color
    return out


def _models(train):
    """Recover all distance/color explanations, without query labels or state."""
    parsed = [(pair, *_parse_input(pair["input"])) for pair in train]
    backgrounds = {bg for _, bg, _ in parsed}
    possibilities = []
    for distance in range(1, 30):
        roles = [set(), set()]
        valid = True
        for pair, _, objects in parsed:
            inp, out = pair["input"], pair["output"]
            try:
                stats = _events(objects, distance, inp.shape)
            except AssertionError:
                valid = False
                break
            untouched = np.ones(inp.shape, dtype=bool)
            for role, cells in enumerate(stats["paint"]):
                roles[role].update(int(out[p]) for p in cells)
                for p in cells:
                    untouched[p] = False
            if any(len(colors) > 1 for colors in roles) or not np.array_equal(inp[untouched], out[untouched]):
                valid = False
                break
        if not valid:
            continue
        choices = [sorted(colors) if colors else list(range(10)) for colors in roles]
        possibilities.extend((distance, a, b) for a in choices[0] for b in choices[1]
                             if a != b and a not in backgrounds and b not in backgrounds)
    assert possibilities, "Training does not fit the endpoint rule"
    return possibilities


def _predict_models(models, test_input):
    background, objects = _parse_input(test_input)
    outputs = []
    for d, single, collision in models:
        if background in (single, collision):
            continue
        try:
            outputs.append(_apply(test_input, objects, d, single, collision))
        except AssertionError:
            continue
    assert outputs and all(np.array_equal(outputs[0], o) for o in outputs[1:]), \
        "Competing training explanations disagree on the query"
    return outputs[0]


def solve(train_pairs, test_input):
    return _predict_models(_models(train_pairs), test_input)


def _allowed(objects, distance, shape, mode):
    if not _compatible(objects):
        return False
    try:
        stats = _events(objects, distance, shape)
    except AssertionError:
        return False
    if mode == "simple" and (stats["collision_sites"] or stats["struck"]):
        return False
    if mode == "collision" and stats["struck"]:
        return False
    if mode in ("strike", "multi", "mutual", "through", "cross") and stats["collision_sites"]:
        return False
    return True


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _module(rng, kind, distance, directions):
    """Build an event by solving tip + distance * direction = destination."""
    if kind in ("simple", "formation"):
        return [_arrow((0, 0), _choose(rng, directions))]
    # Enumerate finite geometric choices, then sample them in random order. There
    # are no direction-specific templates and no chance of missing a valid strike.
    options = []
    if kind in ("collision", "triple"):
        count = min(len(directions), 3 if kind == "triple" else 2)
        for _ in range(40):
            chosen = rng.choice(directions, size=count, replace=False)
            options.append([_arrow(tuple(-distance * v for v in _DIRECTION_VECTORS[di]), str(di))
                            for di in chosen])
    else:
        for target_direction in directions:
            target = _arrow((0, 0), target_direction)
            for hit in target[3]:
                for shooter_direction in directions:
                    dr, dc = _DIRECTION_VECTORS[shooter_direction]
                    shooter = _arrow((hit[0]-distance*dr, hit[1]-distance*dc), shooter_direction)
                    if kind in ("strike", "both"):
                        options.append([target, shooter])
                    elif kind == "mutual":
                        if _endpoint(target, distance) in shooter[3]:
                            options.append([target, shooter])
                    elif kind == "collision_strike":
                        for other_direction in directions:
                            if other_direction == shooter_direction:
                                continue
                            vr, vc = _DIRECTION_VECTORS[other_direction]
                            other = _arrow((hit[0]-distance*vr, hit[1]-distance*vc), other_direction)
                            options.append([target, shooter, other])
                    elif kind == "multi":
                        for second_hit in target[3]:
                            if second_hit == hit:
                                continue
                            for other_direction in directions:
                                vr, vc = _DIRECTION_VECTORS[other_direction]
                                other = _arrow((second_hit[0]-distance*vr, second_hit[1]-distance*vc), other_direction)
                                options.append([target, shooter, other])
                    elif kind == "through":
                        for step in range(2, distance):
                            other = _arrow((hit[0]-step*dr, hit[1]-step*dc), shooter_direction)
                            options.append([target, other])
                    elif kind == "cross":
                        # The two rays meet at the origin before either endpoint.
                        ar, ac = _DIRECTION_VECTORS[target_direction]
                        if ar*dc == ac*dr:
                            continue
                        for k in range(2, distance):
                            options.append([
                                _arrow((-k*ar, -k*ac), target_direction),
                                _arrow((-(distance-1)*dr, -(distance-1)*dc), shooter_direction)])
    if not options:
        return None
    for index in rng.permutation(len(options)):
        objects = options[int(index)]
        mode = kind if kind not in ("triple", "collision_strike") else "both"
        if not _allowed(objects, distance, None, mode):
            continue
        stats = _events(objects, distance)
        if kind == "multi" and not any(len(hits) >= 2 for hits in stats["hits"].values()):
            continue
        if kind == "collision_strike" and not any(n > 1 and p in stats["owner"] for p, n in stats["counts"].items()):
            continue
        return objects
    return None


def _bounds(objects, distance):
    points = [p for obj in objects for p in obj[3]] + [_endpoint(obj, distance) for obj in objects]
    return tuple(min(p[axis] for p in points) for axis in (0, 1)), \
        tuple(max(p[axis] for p in points) for axis in (0, 1))


def _move(objects, offset):
    return [_arrow((o[0]+offset[0], o[1]+offset[1]), o[2]) for o in objects]


def _place(rng, module, selected, distance, shape, mode):
    low, high = _bounds(module, distance)
    ranges = [(-low[a], shape[a]-high[a]) for a in (0, 1)]
    if any(a >= b for a, b in ranges):
        return None
    for _ in range(65):
        offset = tuple(int(rng.integers(a, b)) if rng.random() > .25
                       else _choose(rng, (a, b-1)) for a, b in ranges)
        moved = _move(module, offset)
        if _allowed(selected + moved, distance, shape, mode):
            return moved
    # Deterministic complete placement search is the bounded fallback.
    for r in range(*ranges[0]):
        for c in range(*ranges[1]):
            moved = _move(module, (r, c))
            if _allowed(selected + moved, distance, shape, mode):
                return moved
    return None


def _layout(rng, distance, directions, kind, teaching=False):
    mode = "both" if kind in ("both", "collision_strike", "triple") else kind
    if mode == "formation":
        mode = "simple"
    primary = _module(rng, kind, distance, directions)
    if primary is None:
        # Some direction pools cannot realize mutual/crossing events. A strike
        # remains possible for every direction, including a one-direction puzzle.
        kind, mode = "strike", "strike"
        primary = _module(rng, kind, distance, directions)
    low, high = _bounds(primary, distance)
    minimum = [high[a]-low[a]+1 for a in (0, 1)]
    richness = int(rng.choice(3, p=(.23, .46, .31)))
    desired = int(rng.integers(*((1, 3), (3, 10), (10, 25))[richness]))
    desired = max(desired, len(primary))
    # Event extent sets only a lower bound. Size, count, aspect and arrangement
    # are otherwise independent pair-level choices shared by both splits.
    floor = (4, 9, 18)[richness]
    ceiling = (18, 27, 30)[richness]
    shape = tuple(int(rng.integers(max(floor, n), max(ceiling, n)+1)) for n in minimum)
    if rng.random() < .3:
        shape = (max(shape), max(shape))
    placed = _place(rng, primary, [], distance, shape, mode)
    assert placed is not None
    selected = placed
    if kind == "both":
        secondary = _module(rng, "collision", distance, directions)
        added = _place(rng, secondary, selected, distance, shape, mode)
        if added is None:
            shape = (30, 30)
            added = _place(rng, secondary, selected, distance, shape, mode)
        if added is not None:
            selected += added
    # Several independent event sites, when space permits.
    if len(directions) > 1 and rng.random() < .35 and kind in ("both", "collision", "strike", "collision_strike", "triple"):
        extra = _module(rng, _choose(rng, ("strike", "collision")) if mode == "both" else mode,
                        distance, directions)
        if extra is not None:
            added = _place(rng, extra, selected, distance, shape, mode)
            if added is not None:
                selected += added
    desired = max(desired, len(selected))
    if kind == "formation" or rng.random() < .65:
        # Build rows or staggered lattices around an existing arrow; mixing in
        # other independently generated events lets formations interact.
        base = _choose(rng, selected)
        row_step = max(r for r, _ in base[3])-min(r for r, _ in base[3])+2+int(rng.integers(3))
        col_step = max(c for _, c in base[3])-min(c for _, c in base[3])+2+int(rng.integers(3))
        stagger = int(rng.integers(0, col_step))
        offsets = [(r*row_step, c*col_step+(r % 2)*stagger)
                   for r in range(-8, 9) for c in range(-8, 9) if r or c]
        if rng.random() < .5:
            offsets.sort(key=lambda p: (abs(p[0]), abs(p[1])))
        else:
            offsets.sort(key=lambda p: (abs(p[0])+abs(p[1]), p))
        for offset in offsets:
            if len(selected) >= desired:
                break
            obj = _move([base], offset)[0]
            if _allowed(selected+[obj], distance, shape, mode):
                selected.append(obj)
    # Random local clusters and globally scattered arrows fill the same count
    # range. Crowding can reduce the requested count; actual counts are measured.
    for _ in range(400):
        if len(selected) >= desired:
            break
        direction = _choose(rng, directions)
        if rng.random() < .32 and selected:
            base = _choose(rng, selected)
            tip = (base[0]+int(rng.integers(-6, 7)), base[1]+int(rng.integers(-6, 7)))
        else:
            tip = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        obj = _arrow(tip, direction)
        if _allowed(selected+[obj], distance, shape, mode):
            selected.append(obj)
    if teaching and not _events(selected, distance)["single_background"]:
        for direction in directions:
            added = _place(rng, [_arrow((0, 0), direction)], selected, distance, shape, mode)
            if added and _events(selected+added, distance)["single_background"]:
                selected += added
                break
        else:
            # Add a separate visible distance witness in a spacious canvas.
            shape = (30, 30)
            for direction in directions:
                added = _place(rng, [_arrow((0, 0), direction)], selected, distance, shape, "simple")
                if added:
                    selected += added
                    break
    return selected, shape


def _direction_pools(rng):
    regime = int(rng.choice(5, p=(.13, .22, .30, .25, .10)))
    perm = tuple(str(d) for d in rng.permutation(_DIRECTIONS))
    if regime == 0:
        return perm[:1], perm[:1]
    if regime == 1:
        common = perm[:int(rng.integers(2, 7))]
        return common, common
    if regime == 2:
        return _DIRECTIONS, _DIRECTIONS
    if regime == 3:
        size = int(rng.integers(2, 6))
        return perm[:size], perm[size:]+perm[:int(rng.integers(0, 2))]
    # Occasional rotated/reflected analogues of the original shift.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def transform(direction):
        r, c = _DIRECTION_VECTORS[direction]
        if reflect:
            c = -c
        for _ in range(turns):
            r, c = -c, r
        return next(d for d, v in _DIRECTION_VECTORS.items() if v == (r, c))
    return tuple(map(transform, ("N", "NW", "SW"))), tuple(map(transform, ("N", "S", "W", "NE", "SE")))


def _pair(rng, distance, single, collision, background, directions, kind, teaching):
    # Single-direction and repeated-direction inputs occur in each split, not
    # just in puzzles restricted to one direction.
    if kind in ("simple", "formation", "strike") and rng.random() < .5:
        directions = (_choose(rng, directions),)
    objects, shape = _layout(rng, distance, directions, kind, teaching)
    palette = [c for c in range(10) if c != background and (not teaching or c not in (single, collision))]
    grid = np.full(shape, background, dtype=np.int8)
    mono = bool(rng.random() < .45 or len(objects) == 1)
    base = _choose(rng, palette)
    colors = [_choose(rng, palette) for _ in objects] if not mono else [base]*len(objects)
    if not mono and len(set(colors)) == 1 and len(objects) > 1:
        colors[-1] = next(c for c in palette if c != base)
    for obj, color in zip(objects, colors):
        for p in obj[3]:
            grid[p] = color
    return {"input": grid, "output": _apply(grid, objects, distance, single, collision)}


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _is_source_pair(pair):
    if pair["input"].shape != (16, 16):
        return False
    return tuple("".join(str(int(v)) for v in pair[role].flat) for role in ("input", "output")) in _SOURCE_PAIRS


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    single, collision = map(int, rng.choice(10, size=2, replace=False))
    distance = int(rng.choice((3, 4, 5, 6, 7), p=(.17, .10, .46, .10, .17)))
    train_dirs, test_dirs = _direction_pools(rng)
    has_collision = len(train_dirs) > 1
    ordinary = ["simple", "simple", "formation", "strike", "multi", "mutual", "through", "cross"]
    complex_modes = ["collision", "both", "triple", "collision_strike"] if has_collision else []
    # Every puzzle teaches a visible strike. The unused collision identity in
    # a one-direction puzzle need not be shown.
    teaching = [_choose(rng, ("strike", "multi")),
                _choose(rng, ("collision", "triple", "both", "collision_strike")) if has_collision else "simple"]
    train_kinds = teaching + [_choose(rng, ordinary+complex_modes) for _ in range(train_count-2)]
    order = rng.permutation(train_count)
    train_specs = [(train_kinds[int(i)], int(i) < 2) for i in order]
    test_specs = [(_choose(rng, ordinary+complex_modes), False) for _ in range(test_count)]
    backgrounds = [c for c in range(10) if c not in (single, collision)]
    background_order = [int(c) for c in rng.permutation(backgrounds)]
    puzzle, seen = {"train": [], "test": []}, set()
    pair_number = 0
    for split, regime, specs, directions in (("train", "source_train", train_specs, train_dirs),
                                              ("test", "target_test", test_specs, test_dirs)):
        for kind, essential in specs:
            background = background_order[pair_number % len(background_order)]
            pair_number += 1
            for attempt in range(12):
                pair = _pair(rng, distance, single, collision, background, directions, kind, essential)
                key = _grid_key(pair["input"])
                if key not in seen and not _is_source_pair(pair) and not np.array_equal(pair["input"], pair["output"]):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                # A single shot with a different canvas is always distinct and
                # changed; only nonessential crowded configurations need this.
                assert not essential
                pair = _pair(rng, distance, single, collision, background, directions, "simple", False)
                assert _grid_key(pair["input"]) not in seen
                puzzle[split].append(pair)
                seen.add(_grid_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer) and np.all((grid >= 0) & (grid <= 9))
        assert pair["input"].shape == pair["output"].shape
        assert not np.array_equal(pair["input"], pair["output"])
        assert not _is_source_pair(pair)
        key = _grid_key(pair["input"])
        assert key not in seen
        seen.add(key)
        _parse_input(pair["input"])
    train = [{k: p[k].copy() for k in ("input", "output")} for p in puzzle["train"]]
    models = _models(train)
    assert len({d for d, _, _ in models}) == 1, "Distance must be taught unambiguously"
    distance, single, collision = models[0]
    training_single = False
    visible_strike = False
    training_collision = False
    for pair in train:
        _, objects = _parse_input(pair["input"])
        stats = _events(objects, distance, pair["input"].shape)
        training_single |= bool(stats["single_background"])
        training_collision |= bool(stats["collision_sites"])
        visible_strike |= any(all(pair["input"][p] != pair["output"][p] for p in objects[i][3])
                              for i in stats["hits"])
        assert np.array_equal(_predict_models(models, pair["input"]), pair["output"])
    assert training_single, "Need a clear distance and single-impact witness"
    for pair in puzzle["test"]:
        _, objects = _parse_input(pair["input"])
        stats = _events(objects, distance, pair["input"].shape)
        assert not stats["struck"] or visible_strike, "Teach whole-object recoloring"
        assert not stats["collision_sites"] or training_collision, "Teach the collision role when queried"
        expected = _predict_models(models, pair["input"].copy())
        assert np.array_equal(expected, pair["output"])


def _features(pair, distance):
    grid = pair["input"]
    bg, objects = _parse_input(grid)
    stats = _events(objects, distance, grid.shape)
    endpoints = [_endpoint(o, distance) for o in objects]
    paths = [{_endpoint(o, step) for step in range(1, distance)} for o in objects]
    edges = {(i, stats["owner"][p]) for i, p in enumerate(endpoints) if p in stats["owner"]}
    repeated = False
    for i, a in enumerate(objects):
        for b in objects[i+1:]:
            if a[2] == b[2]:
                repeated |= any(c[2] == a[2] and (c[0], c[1]) == (2*b[0]-a[0], 2*b[1]-a[1]) for c in objects)
                repeated |= any(c[2] == a[2] and (c[0], c[1]) == (2*a[0]-b[0], 2*a[1]-b[1]) for c in objects)
    rows = {}
    for r, c, direction, _ in objects:
        rows.setdefault((direction, r), set()).add(c)
    regular_rows = any(c+step in cols and c+2*step in cols
                       for cols in rows.values() for c in cols for step in range(3, 10))
    staggered = any(direction == other_direction and r != other_r and
                    c+step in cols and other_c+step in other_cols and (other_c-c) % step != 0
                    for (direction, r), cols in rows.items()
                    for (other_direction, other_r), other_cols in rows.items()
                    for c in cols for other_c in other_cols for step in range(3, 10))
    vectors = [_DIRECTION_VECTORS[o[2]] for o in objects]
    return {
        "shape": grid.shape, "background": bg, "distance": distance,
        "arrows": len(objects), "directions": tuple(sorted({o[2] for o in objects})),
        "arrow_colors": len({int(grid[o[3][0]]) for o in objects}),
        "repeated": bool(repeated), "regular_rows": regular_rows, "staggered": staggered,
        "mixed_compass": any(a*b == 0 for a, b in vectors) and any(a*b != 0 for a, b in vectors),
        "density": 3*len(objects)/grid.size, "collision_sites": stats["collision_sites"],
        "max_convergence": max(stats["counts"].values()), "struck": stats["struck"],
        "collision_strike": any(n > 1 and p in stats["owner"] for p, n in stats["counts"].items()),
        "multiple_hits": any(len(hits) > 1 for hits in stats["hits"].values()),
        "tip_strike": any(p == objects[i][:2] for i, hits in stats["hits"].items() for p, _ in hits),
        "wing_strike": any(p != objects[i][:2] for i, hits in stats["hits"].items() for p, _ in hits),
        "through": any(p in stats["owner"] and stats["owner"][p] != i for i, path in enumerate(paths) for p in path),
        "crossing": any(paths[i] & paths[j] and endpoints[i] != endpoints[j]
                        and vectors[i][0]*vectors[j][1] != vectors[i][1]*vectors[j][0]
                        for i in range(len(objects)) for j in range(i)),
        "mutual": any((j, i) in edges for i, j in edges),
        "edge_arrow": any(r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1) for o in objects for r, c in o[3]),
        "edge_endpoint": any(r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1) for r, c in endpoints),
    }


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _regression_tests():
    source = [{role: np.array([int(v) for v in text], dtype=np.int8).reshape(16, 16)
               for role, text in zip(("input", "output"), strings)} for strings in _SOURCE_PAIRS]
    for pair in source:
        _, objects = _parse_input(pair["input"])
        assert np.array_equal(_apply(pair["input"], objects, 5, 9, 1), pair["output"])
    assert np.array_equal(solve(source[:3], source[3]["input"].copy()), source[3]["output"])
    # Review's exact compatible two-cell hit; no collision may be invented.
    objects = [_arrow((8, 8), "E"), _arrow((13, 8), "N"), _arrow((12, 2), "NE")]
    stats = _events(objects, 5, (18, 18))
    assert len(stats["hits"][0]) == 2 and stats["collision_sites"] == 0
    grid = np.full((18, 18), 7, dtype=np.int8)
    for o in objects:
        for p in o[3]:
            grid[p] = 6
    out = _apply(grid, objects, 5, 9, 1)
    assert all(out[p] == 9 for p in objects[0][3]) and out[8, 13] == 9
    incompatible = objects + [_arrow((3, 8), "S")]
    try:
        _events(incompatible, 5, (18, 18))
    except AssertionError:
        pass
    else:
        raise AssertionError("Contradictory target roles accepted")
    # A struck arrow still fires, and rays pass through objects without stopping.
    objects = [_arrow((9, 3), "E"), _arrow((9, 8), "W"), _arrow((4, 6), "S")]
    grid = np.full((16, 16), 7, dtype=np.int8)
    for i, o in enumerate(objects):
        for p in o[3]:
            grid[p] = (2, 3, 4)[i]
    out = _apply(grid, objects, 5, 9, 1)
    assert all(out[p] == 9 for o in objects[:2] for p in o[3])
    assert out[9, 6] == 9 and out[9, 5] == 7
    assert np.array_equal(out, _apply(grid, objects[::-1], 5, 9, 1))
    for specifications, marks, unchanged in (
        ((((8, 2), "E"), ((8, 5), "N")), ((8, 7), (3, 5)), (8, 5)),
        ((((8, 5), "E"), ((5, 8), "S")), ((8, 10), (10, 8)), (8, 8)),
    ):
        objects = [_arrow(tip, direction) for tip, direction in specifications]
        grid = np.full((16, 16), 7, dtype=np.int8)
        for obj in objects:
            for p in obj[3]:
                grid[p] = 6
        out = _apply(grid, objects, 5, 9, 1)
        assert all(out[p] == 9 for p in marks)
        assert out[unchanged] == grid[unchanged]
    objects = [_arrow((13, 8), "N"), _arrow((8, 3), "E"),
               _arrow((3, 3), "SE"), _arrow((8, 8), "W")]
    grid = np.full((20, 20), 7, dtype=np.int8)
    for obj in objects:
        for p in obj[3]:
            grid[p] = 6
    out = _apply(grid, objects, 5, 9, 1)
    assert all(out[p] == 1 for p in objects[3][3])
    assert all(out[p] == 9 for p in objects[1][3])
    assert np.array_equal(out, _apply(grid, objects[::-1], 5, 9, 1))
    # Mutating a label or distance must fail validation.
    puzzle = generate_puzzle(200, 2, 1)
    rotated = {s: [{k: np.rot90(p[k]).copy() for k in p} for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(rotated)
    broken = {s: [{k: p[k].copy() for k in p} for p in puzzle[s]] for s in puzzle}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect query label accepted")


def self_test() -> None:
    _regression_tests()
    puzzles = [generate_puzzle(seed) for seed in range(4200, 4242)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    records, conventions = [], set()
    coverage = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        models = _models(puzzle["train"])
        distance = models[0][0]
        conventions.add(tuple(models))
        pairs = puzzle["train"] + puzzle["test"]
        features = [dict(_features(pair, distance), split=split)
                    for split in ("train", "test") for pair in puzzle[split]]
        assert len({f["background"] for f in features}) >= 2
        assert len({f["shape"] for f in features}) >= 2
        train_directions = {d for f in features if f["split"] == "train" for d in f["directions"]}
        test_directions = {d for f in features if f["split"] == "test" for d in f["directions"]}
        coverage.append((train_directions, test_directions))
        records.extend(features)
        for pair in pairs:
            # The pixel reader reconstructs arrows independently of placement.
            assert np.array_equal(solve(puzzle["train"], pair["input"].copy()), pair["output"])
    assert len(conventions) >= 25
    assert sum(a == b for a, b in coverage) >= 4
    assert sum(bool(b-a) for a, b in coverage) >= 4
    assert sum(len(a) == len(b) == 1 for a, b in coverage) >= 2
    assert {f["distance"] for f in records} == {3, 4, 5, 6, 7}
    assert len({f["shape"] for f in records}) >= 80
    assert len({f["arrows"] for f in records}) >= 18
    assert min(f["arrows"] for f in records) == 1 and max(f["arrows"] for f in records) >= 20
    assert len({f["directions"] for f in records}) >= 35
    assert any(len(f["directions"]) == 1 for f in records)
    assert {d for f in records for d in f["directions"]} == set(_DIRECTIONS)
    for name in ("repeated", "regular_rows", "staggered", "mixed_compass", "collision_strike", "multiple_hits", "tip_strike", "wing_strike", "through", "crossing", "mutual", "edge_arrow", "edge_endpoint"):
        assert sum(bool(f[name]) for f in records) >= 4, name
    assert sum(f["max_convergence"] >= 3 for f in records) >= 8
    assert sum(f["collision_sites"] >= 2 for f in records) >= 4
    assert sum(f["struck"] >= 2 for f in records) >= 8
    assert sum(f["arrow_colors"] >= 2 for f in records) >= 70
    assert max(f["density"] for f in records) > .15
    assert min(f["density"] for f in records) < .02
    for split in ("train", "test"):
        selected = [f for f in records if f["split"] == split]
        assert {(bool(f["collision_sites"]), bool(f["struck"])) for f in selected} == {
            (False, False), (False, True), (True, False), (True, True)}
        assert sum(len(f["directions"]) == 1 for f in selected) >= 8
        assert sum(f["mixed_compass"] for f in selected) >= 8
        assert any(f["arrows"] <= 2 for f in selected)
        assert any(f["arrows"] >= 20 for f in selected)
        assert any(f["collision_strike"] for f in selected)
    assert sum(f["shape"][0] != f["shape"][1] and max(f["shape"]) >= 24 and
               f["arrow_colors"] >= 2 and f["repeated"] and f["struck"] and f["collision_sites"] for f in records) >= 2
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        a = generate_puzzle(91, *args)
        b = generate_puzzle(91, *args)
        validate_generated_puzzle(a)
        assert _puzzle_key(a) == _puzzle_key(b)
        assert all(n is None or len(a[s]) == n for s, n in zip(("train", "test"), args))
    assert _puzzle_key(generate_puzzle(17)) != _puzzle_key(generate_puzzle(18))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (2.0, 1)):
        try:
            generate_puzzle(9, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
