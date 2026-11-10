"""Solved synthetic puzzles for the four fixed ARC trace stencils."""

from itertools import permutations

import numpy as np


HOOK, WEST, SOUTH, NORTH = 1, 2, 3, 6
TRAIL, COLLISION, BACKGROUND = 5, 4, 0
ROLES = (HOOK, WEST, SOUTH, NORTH)
SPECS = {
    HOOK: ((-1, 2), ((0, 1), (0, 2))),
    WEST: ((0, -4), ((0, -1), (0, -2), (0, -3))),
    SOUTH: ((3, 0), ((1, 0), (2, 0))),
    NORTH: ((-6, 0), ((-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0))),
}
CROSS_KINDS = ((HOOK, SOUTH), (HOOK, NORTH), (WEST, SOUTH), (WEST, NORTH))
COMPLEMENTARY = (((HOOK, SOUTH), (WEST, NORTH)),
                 ((HOOK, NORTH), (WEST, SOUTH)))

# Source input positions, including the solved test input. These are checked
# by content after decoding a seed's palette, not by size or palette alone.
SOURCE_INPUTS = frozenset({
    ((11, 11), ((1, 9, 7), (2, 4, 5), (6, 6, 3))),
    ((10, 10), ((1, 5, 6), (3, 2, 3))),
    ((10, 10), ((2, 4, 6), (3, 2, 4))),
    ((11, 11), ((1, 2, 3), (2, 4, 5), (3, 5, 8))),
    ((10, 10), ((1, 4, 3), (2, 7, 5))),
    ((9, 9), ((6, 7, 3),)),
    ((11, 11), ((1, 2, 1), (2, 6, 7), (3, 7, 9), (6, 7, 5))),
})


def _count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng):
    # Six semantic colors are stable; the canvas may use any unused color.
    values = [int(x) for x in rng.permutation(10)[:7]]
    return dict(zip((BACKGROUND, HOOK, WEST, SOUTH, COLLISION, TRAIL, NORTH), values))


def _tokens(grid):
    return [(int(grid[r, c]), int(r), int(c))
            for r, c in np.argwhere(grid != BACKGROUND)]


def _signature(grid):
    return (tuple(grid.shape), tuple(sorted(_tokens(grid))))


def _key(grid):
    return (tuple(grid.shape), grid.tobytes())


def _scene(tokens, height, width):
    """Inspect instance geometry and reject every ambiguous overwrite."""
    if not tokens or any(role not in ROLES for role, _, _ in tokens):
        return None
    sources = [(r, c) for _, r, c in tokens]
    if len(set(sources)) != len(sources):
        return None
    if any(not (0 <= r < height and 0 <= c < width) for r, c in sources):
        return None
    for i, (r, c) in enumerate(sources):
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in sources[i + 1:]):
            return None

    endpoints = []
    owners = {}
    for index, (role, r, c) in enumerate(tokens):
        endpoint_offset, connector_offsets = SPECS[role]
        er, ec = r + endpoint_offset[0], c + endpoint_offset[1]
        if not (0 <= er < height and 0 <= ec < width):
            return None
        endpoints.append((er, ec))
        for dr, dc in connector_offsets:
            point = (r + dr, c + dc)
            if not (0 <= point[0] < height and 0 <= point[1] < width):
                return None
            owners.setdefault(point, []).append(index)

    source_set, endpoint_set = set(sources), set(endpoints)
    connector_set = set(owners)
    if (len(endpoint_set) != len(endpoints) or source_set & endpoint_set
            or source_set & connector_set or endpoint_set & connector_set):
        return None
    if any(len(ids) > 2 for ids in owners.values()):
        return None

    collisions = []
    rays = {}
    occupied = source_set | endpoint_set | connector_set
    ray_seen = set()
    for point, ids in owners.items():
        if len(ids) != 2:
            continue
        pair = frozenset((tokens[ids[0]][0], tokens[ids[1]][0]))
        if len(pair) != 2 or pair not in (frozenset(x) for x in CROSS_KINDS):
            return None
        collisions.append((point, pair))
        r, c = point
        ray = []
        while r + 1 < height and c - 1 >= 0:
            r += 1
            c -= 1
            if (r, c) in occupied or (r, c) in ray_seen:
                return None
            ray.append((r, c))
            ray_seen.add((r, c))
        if not ray:
            return None
        rays[point] = tuple(ray)
    return {"endpoints": tuple(endpoints), "owners": owners,
            "collisions": tuple(sorted(collisions, key=lambda x: x[0])), "rays": rays}


def _rule(grid):
    info = _scene(_tokens(grid), *grid.shape)
    if info is None:
        raise ValueError("invalid trace scene")
    out = grid.copy()
    for (role, _, _), point in zip(_tokens(grid), info["endpoints"]):
        out[point] = role
    for point, ids in info["owners"].items():
        out[point] = TRAIL if len(ids) == 1 else COLLISION
    for point, _ in info["collisions"]:
        for cell in info["rays"][point]:
            out[cell] = COLLISION
    return out


def _grid(tokens, shape):
    grid = np.zeros(shape, dtype=np.int8)
    for role, r, c in tokens:
        grid[r, c] = role
    return grid


def _encode(grid, palette):
    out = np.empty(grid.shape, dtype=np.int8)
    for canonical, actual in palette.items():
        out[grid == canonical] = actual
    return out


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _pair_palette(palette, grid):
    result = dict(palette)
    result[BACKGROUND] = _background(grid)
    return result


def _shape(rng, mode):
    if mode == "multi":
        return int(rng.integers(20, 31)), int(rng.integers(20, 31))
    if mode in ("repeat", "crowded"):
        low = 16 if mode == "repeat" else 20
        return int(rng.integers(low, 31)), int(rng.integers(low, 31))
    band = int(rng.choice(3, p=(0.35, 0.40, 0.25)))
    low, high = ((9, 13), (12, 20), (19, 31))[band]
    if mode == "single":
        low = 8
    return int(rng.integers(low, high)), int(rng.integers(low, high))


def _base(rng, shape, kind, short=False):
    height, width = shape
    horizontal, vertical = kind
    hgaps = (1, 2) if horizontal == HOOK else (1, 2, 3)
    vgaps = (1, 2) if vertical == SOUTH else (1, 2, 3, 4, 5)
    # Random centers give different edge distances, gaps and ray lengths.
    for _ in range(320):
        row = int(rng.integers(1, height - 1))
        col = 1 if short else int(rng.integers(2, width - 1))
        hgap = int(rng.choice(hgaps))
        vgap = int(rng.choice(vgaps))
        hc = col - hgap if horizontal == HOOK else col + hgap
        vr = row - vgap if vertical == SOUTH else row + vgap
        tokens = [(horizontal, row, hc), (vertical, vr, col)]
        info = _scene(tokens, height, width)
        if info is None or len(info["collisions"]) != 1:
            continue
        point, pair = info["collisions"][0]
        length = len(info["rays"][point])
        if pair == frozenset(kind) and (length == 1 if short else length >= 2):
            return tokens
    return None


def _multi_base(rng, shape):
    height, width = shape
    for _ in range(240):
        row = int(rng.integers(8, height - 4))
        left = int(rng.integers(2, width - 11))
        right = int(rng.integers(left + 7, width - 1))
        # Two independent overlaps, with independently clipped rays.
        hgap = int(rng.integers(1, 4))
        vgap = int(rng.integers(1, 6))
        hg2 = int(rng.integers(1, 3))
        vg2 = int(rng.integers(1, 3))
        tokens = [(WEST, row, left + hgap), (NORTH, row + vgap, left),
                  (HOOK, row, right - hg2), (SOUTH, row - vg2, right)]
        info = _scene(tokens, height, width)
        if info is not None and len(info["collisions"]) == 2:
            return tokens
    return None


def _positions(role, shape):
    height, width = shape
    offsets = (SPECS[role][0],) + SPECS[role][1]
    return [(r, c) for r in range(height) for c in range(width)
            if all(0 <= r + dr < height and 0 <= c + dc < width
                   for dr, dc in offsets)]


def _add_tokens(rng, tokens, roles, shape, collision_count):
    tokens = list(tokens)
    for role in roles:
        positions = _positions(role, shape)
        order = rng.permutation(len(positions))
        placed = False
        for index in order[:min(160, len(order))]:
            r, c = positions[int(index)]
            candidate = tokens + [(role, r, c)]
            info = _scene(candidate, *shape)
            if info is not None and len(info["collisions"]) == collision_count:
                tokens = candidate
                placed = True
                break
        if not placed:
            return None
    return tokens


def _choose_roles(rng, mode, trained_repeat):
    if mode == "single":
        return [int(rng.choice(ROLES))]
    if mode == "four":
        return list(ROLES)
    if mode == "repeat":
        repeated = int(rng.choice(ROLES))
        count = int(rng.integers(3, 6))
        roles = [repeated, repeated]
        roles.extend(int(x) for x in rng.choice(ROLES, size=count - 2))
        return roles
    if mode == "crowded":
        count = int(rng.integers(5 if trained_repeat else 3,
                                 8 if trained_repeat else 5))
        if trained_repeat:
            return [int(x) for x in rng.choice(ROLES, size=count)]
        return [int(x) for x in rng.choice(ROLES, size=count, replace=False)]
    count = int(rng.integers(2, 5))
    return [int(x) for x in rng.choice(ROLES, size=count, replace=False)]


def _construct(rng, shape, mode, kind=None, trained_repeat=False):
    if mode == "multi":
        return _multi_base(rng, shape)
    if mode in ("cross", "short", "four"):
        base = _base(rng, shape, kind, mode == "short")
        if base is None:
            return None
        if mode == "four":
            extras = [role for role in ROLES if role not in kind]
            return _add_tokens(rng, base, extras, shape, 1)
        # Some crossings coexist with independent same-color or other tokens.
        if mode == "cross" and shape[0] >= 15 and shape[1] >= 15 and rng.random() < 0.80:
            if trained_repeat:
                choices = kind if rng.random() < 0.70 else ROLES
            else:
                choices = tuple(role for role in ROLES if role not in kind)
            extra = int(rng.choice(choices))
            trial = _add_tokens(rng, base, [extra], shape, 1)
            if trial is not None:
                return trial
        return base
    roles = _choose_roles(rng, mode, trained_repeat)
    return _add_tokens(rng, [], roles, shape, 0)


def _pair(rng, mode, palette, used, kind=None, trained_repeat=False,
          different_shape=None, different_aspect=None,
          different_background=None):
    for _ in range(90):
        shape = _shape(rng, mode)
        if shape == different_shape:
            continue
        aspect = 0 if shape[0] == shape[1] else 1 if shape[1] > shape[0] else 2
        if aspect == different_aspect:
            continue
        palette_here = dict(palette)
        semantic_colors = {palette[role] for role in
                           (HOOK, WEST, SOUTH, NORTH, TRAIL, COLLISION)}
        backgrounds = [color for color in range(10)
                       if color not in semantic_colors and color != different_background]
        palette_here[BACKGROUND] = int(rng.choice(backgrounds))
        tokens = _construct(rng, shape, mode, kind, trained_repeat)
        if tokens is None:
            continue
        inp = _grid(tokens, shape)
        if _signature(inp) in SOURCE_INPUTS:
            continue
        actual = _encode(inp, palette_here)
        key = _key(actual)
        if key in used:
            continue
        used.add(key)
        return {"input": actual, "output": _encode(_rule(inp), palette_here)}
    raise ValueError(f"cannot construct {mode} pair")


def _decode(grid, inverse):
    out = np.empty(grid.shape, dtype=np.int8)
    for actual, canonical in inverse.items():
        out[grid == actual] = canonical
    return out


def _infer_palette(train):
    # Infer conventions from demonstrations only. The dominant input color is
    # the canvas; first two training pairs expose all four source roles and
    # both output-only colors, even for the minimum legal train count.
    backgrounds = [_background(p["input"]) for p in train]
    sources = sorted({int(color) for pair, background in zip(train, backgrounds)
                      for color in np.unique(pair["input"]) if color != background})
    outputs = set(int(x) for p in train for x in np.unique(p["output"]))
    extras = sorted(outputs - set(sources) - set(backgrounds))
    assert len(sources) == 4 and len(extras) == 2
    matches = []
    for role_colors in permutations(sources):
        for trail, collision in permutations(extras):
            palette = dict(zip(ROLES, role_colors))
            palette.update({TRAIL: trail, COLLISION: collision})
            valid = True
            for pair, background in zip(train, backgrounds):
                palette_here = dict(palette)
                palette_here[BACKGROUND] = background
                inverse = {v: k for k, v in palette_here.items()}
                canonical = _decode(pair["input"], inverse)
                try:
                    expected = _encode(_rule(canonical), palette_here)
                except ValueError:
                    valid = False
                    break
                if not np.array_equal(expected, pair["output"]):
                    valid = False
                    break
            if valid:
                matches.append(palette)
    assert len(matches) == 1, "training must identify one stable color convention"
    return matches[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, int(seed), num_train, num_test)
    palette = _palette(rng)
    pairing = COMPLEMENTARY[int(rng.integers(0, 2))]
    pairing = tuple(pairing[int(i)] for i in rng.permutation(2))
    puzzle = {"train": [], "test": []}
    used = set()
    child_rngs = [np.random.default_rng(int(x)) for x in
                  rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)]

    for index, kind in enumerate(pairing):
        first_shape = puzzle["train"][0]["input"].shape if index else None
        first_aspect = (0 if first_shape[0] == first_shape[1]
                        else 1 if first_shape[1] > first_shape[0] else 2) if index else None
        pair = _pair(child_rngs[index], "cross", palette, used, kind,
                     different_shape=first_shape, different_aspect=first_aspect,
                     different_background=_background(puzzle["train"][0]["input"]) if index else None)
        puzzle["train"].append(pair)

    extra_modes = ("single", "disjoint", "cross", "repeat", "short", "multi", "crowded")
    offset = int(rng.integers(0, len(extra_modes)))
    for index in range(2, train_count):
        mode = extra_modes[(offset + index - 2) % len(extra_modes)]
        kind = CROSS_KINDS[int(rng.integers(0, len(CROSS_KINDS)))] if mode in ("cross", "short") else None
        puzzle["train"].append(_pair(child_rngs[index], mode, palette, used, kind,
                                     trained_repeat=True))

    train_info = []
    for pair in puzzle["train"]:
        inverse = {v: k for k, v in _pair_palette(palette, pair["input"]).items()}
        grid = _decode(pair["input"], inverse)
        train_info.append((_tokens(grid), _scene(_tokens(grid), *grid.shape)))
    collision_kinds = sorted({tuple(sorted(kind)) for _, info in train_info
                              for _, kind in info["collisions"]})
    has_repeat = any(len({role for role, _, _ in tokens}) < len(tokens)
                     for tokens, _ in train_info)
    has_multi = any(len(info["collisions"]) > 1 for _, info in train_info)
    test_modes = ["single", "disjoint", "cross", "short", "four", "crowded"]
    if has_repeat:
        test_modes.append("repeat")
    if has_multi:
        test_modes.append("multi")
    for index in range(test_count):
        mode = test_modes[int(child_rngs[train_count + index].integers(0, len(test_modes)))]
        kind = collision_kinds[int(child_rngs[train_count + index].integers(0, len(collision_kinds)))] \
            if mode in ("cross", "short", "four") else None
        puzzle["test"].append(_pair(child_rngs[train_count + index], mode, palette,
                                    used, kind, trained_repeat=has_repeat))
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _check_grid(pair["input"])
            _check_grid(pair["output"])
            assert pair["input"].shape == pair["output"].shape
    palette = _infer_palette(puzzle["train"])
    seen = set()
    trained_kinds = set()
    trained_repeat = False
    trained_multi = False
    demonstrated = set()
    shapes = set()
    train_scenes = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            palette_here = _pair_palette(palette, inp)
            inverse = {v: k for k, v in palette_here.items()}
            assert set(int(x) for x in np.unique(inp)) <= set(palette_here.values())
            assert set(int(x) for x in np.unique(out)) <= set(palette_here.values())
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            canonical = _decode(inp, inverse)
            assert _signature(canonical) not in SOURCE_INPUTS
            tokens = _tokens(canonical)
            info = _scene(tokens, *inp.shape)
            assert info is not None
            assert np.array_equal(out, _encode(_rule(canonical), palette_here))
            kinds = {tuple(sorted(kind)) for _, kind in info["collisions"]}
            repeat = len(set(role for role, _, _ in tokens)) < len(tokens)
            if split == "train":
                train_scenes.append((tokens, info))
                demonstrated.update(role for role, _, _ in tokens)
                trained_kinds.update(kinds)
                trained_repeat |= repeat
                trained_multi |= len(info["collisions"]) > 1
            else:
                assert kinds <= trained_kinds
                assert not repeat or trained_repeat
                assert len(info["collisions"]) <= 1 or trained_multi
            shapes.add(inp.shape)
    assert demonstrated == set(ROLES)
    assert len(shapes) >= 2
    assert len(trained_kinds) >= 2
    first_tokens, first_info = train_scenes[0]
    second_tokens, second_info = train_scenes[1]
    assert len(first_info["collisions"]) == len(second_info["collisions"]) == 1
    core_kinds = {tuple(sorted(first_info["collisions"][0][1])),
                  tuple(sorted(second_info["collisions"][0][1]))}
    assert core_kinds in ({(HOOK, SOUTH), (WEST, NORTH)},
                          {(HOOK, NORTH), (WEST, SOUTH)})
    assert {role for role, _, _ in first_tokens + second_tokens} == set(ROLES)
    first, second = (pair["input"] for pair in puzzle["train"][:2])
    first_aspect = 0 if first.shape[0] == first.shape[1] else (1 if first.shape[1] > first.shape[0] else 2)
    second_aspect = 0 if second.shape[0] == second.shape[1] else (1 if second.shape[1] > second.shape[0] else 2)
    assert first_aspect != second_aspect
    assert _background(first) != _background(second)


def _same(a, b):
    return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))


def self_test() -> None:
    counts = set()
    shapes = set()
    aspects = set()
    role_subsets = set()
    kinds = set()
    ray_lengths = set()
    source_counts = set()
    palettes = set()
    backgrounds = set()
    repeats = multis = short = no_cross = 0
    test_kinds = set()
    test_no_cross = test_repeated = test_many = test_short = 0
    test_smaller = repeated_cross = multi_rectangular = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        largest_train = max(pair["input"].size for pair in puzzle["train"])
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        palette = _infer_palette(puzzle["train"])
        palettes.add(tuple(palette[k] for k in (1, 2, 3, 4, 5, 6)))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inverse = {v: k for k, v in
                           _pair_palette(palette, pair["input"]).items()}
                backgrounds.add(_background(pair["input"]))
                grid = _decode(pair["input"], inverse)
                info = _scene(_tokens(grid), *grid.shape)
                assert info is not None
                assert np.array_equal(_decode(pair["output"], inverse), _rule(grid))
                signatures.add(_signature(grid))
                shapes.add(grid.shape)
                aspects.add(np.sign(grid.shape[0] - grid.shape[1]))
                tokens = _tokens(grid)
                source_counts.add(len(tokens))
                role_subsets.add(frozenset(role for role, _, _ in tokens))
                repeats += len({role for role, _, _ in tokens}) < len(tokens)
                multis += len(info["collisions"]) > 1
                no_cross += len(info["collisions"]) == 0
                repeated = len({role for role, _, _ in tokens}) < len(tokens)
                repeated_cross += repeated and bool(info["collisions"])
                multi_rectangular += len(info["collisions"]) > 1 and grid.shape[0] != grid.shape[1]
                if split == "test":
                    test_no_cross += not info["collisions"]
                    test_repeated += repeated
                    test_many += len(tokens) > 4
                    test_smaller += grid.size < largest_train
                for point, kind in info["collisions"]:
                    kinds.add(tuple(sorted(kind)))
                    if split == "test":
                        test_kinds.add(tuple(sorted(kind)))
                    length = len(info["rays"][point])
                    ray_lengths.add(length)
                    short += length == 1
                    if split == "test":
                        test_short += length == 1
    assert len({x for x, _ in counts}) >= 4 and len({x for _, x in counts}) >= 3
    assert len(signatures) >= 150 and len(shapes) >= 30
    assert aspects == {-1, 0, 1} and len(role_subsets) >= 10
    assert kinds == {tuple(sorted(x)) for x in CROSS_KINDS}
    assert len(ray_lengths) >= 7 and min(ray_lengths) == 1
    assert max(source_counts) >= 5 and repeats > 0 and multis > 0
    assert short > 0 and no_cross > 0 and len(palettes) >= 35
    assert len(backgrounds) == 10
    assert test_kinds == kinds and test_no_cross > 0 and test_repeated > 0
    assert test_many > 0 and test_short > 0 and test_smaller > 0
    assert repeated_cross > 0 and multi_rectangular > 0

    a, b = generate_puzzle(314159), generate_puzzle(314159)
    assert _same(a, b)
    a, b = generate_puzzle(271828, 7, 3), generate_puzzle(271828, 7, 3)
    assert _same(a, b)
    for seed, train, test in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4),
                              (9101, None, 1), (9102, None, 4),
                              (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    # With only the two core demonstrations, no repeated identity has been
    # taught; test constructions must not introduce one by accident.
    for seed in range(10000, 10048):
        puzzle = generate_puzzle(seed, 2, 1)
        palette = _infer_palette(puzzle["train"])
        for pair in puzzle["test"]:
            local = _pair_palette(palette, pair["input"])
            tokens = _tokens(_decode(pair["input"], {v: k for k, v in local.items()}))
            assert len({role for role, _, _ in tokens}) == len(tokens)
    validate_generated_puzzle(generate_puzzle(580412868844527154))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1},
                   {"num_train": 3, "num_test": 1.5}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
