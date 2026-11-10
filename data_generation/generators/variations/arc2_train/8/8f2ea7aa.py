import numpy as np


_SOURCE_INPUTS = (
    ("000000000", "000000000", "000000000", "000007000", "000077000", "000700000", "000000000", "000000000", "000000000"),
    ("880000000", "008000000", "800000000", "000000000", "000000000", "000000000", "000000000", "000000000", "000000000"),
    ("000060000", "000606000", "000660000", "000000000", "000000000", "000000000", "000000000", "000000000", "000000000"),
    ("000000000", "000000000", "000000000", "200000000", "220000000", "022000000", "000000000", "000000000", "000000000"),
)
_SOURCE_KEYS = frozenset(
    ((9, 9), tuple(int(cell) for row in rows for cell in row))
    for rows in _SOURCE_INPUTS
)


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _mask_key(mask):
    return tuple(int(x) for x in mask.flat)


def _count_arg(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count_arg(num_train, "num_train", 2, 10)
    test = _count_arg(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train = (3, 4, 5, 6, 7, 8, 9, 10)[seed % 8]
        test = 1 + (seed // 8) % min(4, train - 1)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _valid_mask(mask):
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        return False
    n = mask.shape[0]
    if not 2 <= n <= 5 or not np.all((mask == 0) | (mask == 1)):
        return False
    count = int(mask.sum())
    if n == 2:
        if count not in (2, 3):
            return False
    elif not n <= count <= int(np.floor(0.8 * n * n)):
        return False
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask == 1)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
    return components


def _traits(mask):
    n = mask.shape[0]
    count = int(mask.sum())
    branch = any(
        mask[r, c] and sum(
            mask[rr, cc]
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= rr < n and 0 <= cc < n
        ) >= 3
        for r in range(n) for c in range(n)
    )
    diagonal = bool(np.all(np.diag(mask) == 1) or np.all(np.diag(np.fliplr(mask)) == 1))
    symmetric = bool(np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)))
    hollow = bool(n >= 4 and any(
        mask[r, c] == 0 and mask[r - 1, c] and mask[r + 1, c]
        and mask[r, c - 1] and mask[r, c + 1]
        for r in range(1, n - 1) for c in range(1, n - 1)
    ))
    return {
        "branched": branch,
        "diagonal": diagonal,
        "disconnected": _components(mask) > 1,
        "symmetric": symmetric,
        "hollow": hollow,
        "sparse": count / mask.size <= 0.4,
        "dense": count / mask.size >= 0.65,
    }


def _small_masks():
    result = []
    for code in range(16):
        mask = np.array([(code >> bit) & 1 for bit in range(3, -1, -1)], dtype=np.int8).reshape(2, 2)
        if _valid_mask(mask):
            result.append(mask)
    return result


_SMALL_MASKS = _small_masks()


def _random_mask(rng, n, kind, required):
    if kind == "diagonal" or kind == "disconnected":
        mask = np.eye(n, dtype=np.int8)
        if int(rng.integers(2)):
            mask = np.fliplr(mask)
        if kind == "diagonal":
            extras = int(rng.integers(0, max(2, n // 2 + 1)))
            for _ in range(extras):
                mask[int(rng.integers(n)), int(rng.integers(n))] = 1
    elif kind == "hollow":
        mask = np.zeros((n, n), dtype=np.int8)
        mask[0] = mask[-1] = 1
        mask[:, 0] = mask[:, -1] = 1
    elif kind == "symmetric":
        mask = np.zeros((n, n), dtype=np.int8)
        for r in range(n):
            for c in range((n + 1) // 2):
                bit = int(rng.random() < 0.48)
                mask[r, c] = mask[r, n - 1 - c] = bit
        for r in (0, n - 1):
            c = int(rng.integers((n + 1) // 2))
            mask[r, c] = mask[r, n - 1 - c] = 1
        r = int(rng.integers(n))
        mask[r, 0] = mask[r, -1] = 1
    else:
        probability = {
            "branched": 0.55, "sparse": 0.28, "dense": 0.68,
            "random": 0.46,
        }[kind]
        mask = (rng.random((n, n)) < probability).astype(np.int8)
        mask[0, int(rng.integers(n))] = 1
        mask[-1, int(rng.integers(n))] = 1
        mask[int(rng.integers(n)), 0] = 1
        mask[int(rng.integers(n)), -1] = 1
    if required is not None:
        r, c = required
        mask[r, c] = 1
        if kind == "symmetric":
            mask[r, n - 1 - c] = 1
    return mask


def _draw_mask(rng, n, kind, required, excluded):
    if n == 2:
        masks = [m for m in _SMALL_MASKS if kind == "any" or int(m.sum()) == kind]
        if required is not None:
            masks = [m for m in masks if m[required]]
        order = rng.permutation(len(masks))
        for index in order:
            mask = masks[int(index)]
            if _mask_key(mask) not in excluded:
                return mask.copy()
        return masks[int(order[0])].copy()
    for _ in range(500):
        mask = _random_mask(rng, n, kind, required)
        if _valid_mask(mask) and _mask_key(mask) not in excluded:
            if kind == "random" or _traits(mask).get(kind, False):
                return mask
    for _ in range(500):
        mask = _random_mask(rng, n, "random", required)
        if _valid_mask(mask) and _mask_key(mask) not in excluded:
            return mask
    raise RuntimeError("unable to draw a valid motif")


def _make_pair(n, background, foreground, mask, location):
    side = n * n
    source = np.full((side, side), background, dtype=np.int8)
    r, c = location
    source[r * n:(r + 1) * n, c * n:(c + 1) * n] = np.where(mask, foreground, background)
    target = np.where(np.kron(mask, mask), foreground, background).astype(np.int8)
    return {"input": source, "output": target}


def _facts(pair):
    source, target = pair["input"], pair["output"]
    assert source.shape == target.shape and source.shape[0] == source.shape[1]
    n = int(round(source.shape[0] ** 0.5))
    assert 2 <= n <= 5 and n * n == source.shape[0]
    values, counts = np.unique(source, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    background = int(values[int(np.argmax(counts))])
    foreground = int(values[int(np.argmin(counts))])
    coords = np.argwhere(source == foreground)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    assert np.all(hi - lo + 1 == n) and np.all(lo % n == 0)
    location = tuple(int(x // n) for x in lo)
    block = source[lo[0]:lo[0] + n, lo[1]:lo[1] + n]
    mask = (block == foreground).astype(np.int8)
    assert _valid_mask(mask) and mask[location]
    rebuilt = _make_pair(n, background, foreground, mask, location)
    assert np.array_equal(source, rebuilt["input"])
    assert np.array_equal(target, rebuilt["output"])
    assert int(np.count_nonzero(target == foreground)) == int(mask.sum()) ** 2
    return {
        "n": n, "background": background, "foreground": foreground,
        "location": location, "mask": mask, "density": int(mask.sum()),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys = set()
    facts = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            key = _grid_key(pair["input"])
            assert key not in keys and key not in _SOURCE_KEYS
            keys.add(key)
            facts[regime].append(_facts(pair))
    train = facts["train"]
    test = facts["test"]
    assert len({_mask_key(f["mask"]) + (f["n"],) for f in train}) >= 2
    assert len({f["location"] for f in train}) >= 2
    train_sizes = {f["n"] for f in train}
    train_backgrounds = {f["background"] for f in train}
    two_densities = {f["density"] for f in train if f["n"] == 2}
    for f in test:
        if f["n"] not in train_sizes:
            assert len(train_sizes) >= 2
        if f["background"] not in train_backgrounds:
            assert len(train_backgrounds) >= 2
        if f["n"] == 2 and f["density"] not in two_densities:
            assert two_densities == {2, 3}


def _size_plan(rng, train_count, test_count):
    mode = int(rng.choice(5, p=(0.19, 0.25, 0.16, 0.13, 0.27)))
    if mode < 4:
        n = (2, 3, 4, 5)[mode]
        return [n] * train_count, [n] * test_count
    train = [2, 3]
    train += [int(rng.integers(2, 6)) for _ in range(train_count - 2)]
    rng.shuffle(train)
    test = []
    for index in range(test_count):
        if index == 0 and int(rng.integers(2)):
            choices = [n for n in (4, 5) if n not in train]
            test.append(int(rng.choice(choices if choices else (4, 5))))
        else:
            test.append(int(rng.choice((2, 3, 4, 5))))
    return train, test


def _palette_plan(rng, train_count, test_count, allow_constant):
    colors = [int(x) for x in rng.permutation(10)]
    mode = int(rng.integers(3)) if allow_constant else 1 + int(rng.integers(2))
    if mode == 0:
        return [(colors[0], colors[1])] * train_count, [(colors[0], colors[1])] * test_count
    if mode == 1:
        background = colors[0]
        train_fg = [colors[1], colors[2]]
        train_fg += [int(rng.choice(colors[1:6])) for _ in range(train_count - 2)]
        train = [(background, fg) for fg in train_fg]
        test = []
        for index in range(test_count):
            fg = int(rng.choice(train_fg)) if (index % 2 == 0 and int(rng.integers(2))) else int(rng.choice(colors[6:]))
            test.append((background, fg))
        return train, test
    train = [(colors[0], colors[1]), (colors[1], colors[0])]
    for _ in range(train_count - 2):
        bg, fg = rng.choice(colors[:7], size=2, replace=False)
        train.append((int(bg), int(fg)))
    test = []
    for index in range(test_count):
        bg = int(rng.choice(colors[:7] if index % 2 else colors))
        fg_pool = list({fg for _, fg in train}) if (index % 2 == 0 and int(rng.integers(2))) else colors
        fg_pool = [color for color in fg_pool if color != bg]
        test.append((bg, int(rng.choice(fg_pool))))
    return train, test


def _location_candidates(mask, index, train_locations, test_mode):
    positions = [tuple(map(int, point)) for point in np.argwhere(mask)]
    n = mask.shape[0]
    if test_mode is None:
        if index == 0:
            preferred = [p for p in positions if p[0] == 0]
        elif index == 1:
            preferred = [p for p in positions if p[0] == n - 1]
        elif index == 2:
            preferred = [p for p in positions if p[1] == 0]
        elif index == 3:
            preferred = [p for p in positions if p[1] == n - 1]
        else:
            preferred = [p for p in positions if p not in train_locations]
    elif test_mode == "reuse":
        preferred = [p for p in positions if p in train_locations]
    else:
        preferred = [p for p in positions if p not in train_locations]
    return preferred if preferred else positions


def _choose_pair(rng, n, palette, kind, index, train_locations, test_mode, used, excluded):
    background, foreground = palette
    if n == 2:
        options = []
        for candidate_foreground in [foreground] + [c for c in range(10) if c != background and c != foreground]:
            for preferred_only in (True, False):
                for mask in _SMALL_MASKS:
                    if kind != "any" and int(mask.sum()) != kind:
                        continue
                    locations = (_location_candidates(mask, index, train_locations, test_mode)
                                 if preferred_only else list(map(tuple, np.argwhere(mask))))
                    for location in locations:
                        pair = _make_pair(n, background, candidate_foreground, mask, location)
                        if _grid_key(pair["input"]) not in used:
                            options.append((pair, location, mask))
                if options:
                    break
            if options:
                break
        if not options:
            raise RuntimeError("exhausted 2 by 2 inputs")
        pair, location, mask = options[int(rng.integers(len(options)))]
        return pair, location, mask
    for _ in range(1000):
        required = None
        if test_mode is not None:
            pool = [p for p in ((r, c) for r in range(n) for c in range(n))
                    if (p in train_locations) == (test_mode == "reuse")]
            if pool:
                required = pool[int(rng.integers(len(pool)))]
        mask = _draw_mask(rng, n, kind, required, excluded)
        positions = _location_candidates(mask, index, train_locations, test_mode)
        if required is not None:
            positions = [required]
        location = positions[int(rng.integers(len(positions)))]
        pair = _make_pair(n, background, foreground, mask, location)
        if _grid_key(pair["input"]) not in used and _grid_key(pair["input"]) not in _SOURCE_KEYS:
            return pair, location, mask
    raise RuntimeError("unable to produce a distinct input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    train_sizes, test_sizes = _size_plan(rng, train_count, test_count)
    allow_constant = not (all(n == 2 for n in train_sizes + test_sizes) and train_count + test_count > 4)
    train_palettes, test_palettes = _palette_plan(rng, train_count, test_count, allow_constant)
    kinds = ["branched", "diagonal", "disconnected", "symmetric", "hollow", "sparse", "dense", "random"]
    rng.shuffle(kinds)
    edge_order = ([int(x) for x in rng.permutation(2)]
                  + [int(x) + 2 for x in rng.permutation(2)])
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    pair_rngs = [np.random.default_rng(int(s)) for s in pair_seeds]
    used = set()
    train_locations = set()
    train_masks = set()
    two_densities = set()
    pairs = {"train": [], "test": []}
    for index, (n, palette) in enumerate(zip(train_sizes, train_palettes)):
        prng = pair_rngs[index]
        if n == 2:
            density = 2 if not two_densities else (3 if two_densities == {2} else int(prng.choice((2, 3))))
            if not two_densities and int(prng.integers(2)):
                density = 3
            if two_densities == {3}:
                density = 2
            kind = density
        else:
            kind = kinds[index % len(kinds)]
            if kind == "hollow" and n == 3:
                kind = "branched"
        position_hint = edge_order[index] if index < 4 else index
        pair, location, mask = _choose_pair(prng, n, palette, kind, position_hint, train_locations, None, used, train_masks)
        pairs["train"].append(pair)
        used.add(_grid_key(pair["input"]))
        train_locations.add(location)
        train_masks.add(_mask_key(mask))
        if n == 2:
            two_densities.add(int(mask.sum()))
    for offset, (n, palette) in enumerate(zip(test_sizes, test_palettes)):
        prng = pair_rngs[train_count + offset]
        mode = "reuse" if int(prng.integers(2)) else "holdout"
        if n == 2:
            kind = int(prng.choice(sorted(two_densities))) if two_densities else "any"
        else:
            kind = kinds[(train_count + offset) % len(kinds)]
            if kind == "hollow" and n == 3:
                kind = "diagonal"
        pair, _, _ = _choose_pair(prng, n, palette, kind, offset, train_locations, mode, used, set())
        pairs["test"].append(pair)
        used.add(_grid_key(pair["input"]))
    validate_generated_puzzle(pairs)
    return pairs


def _puzzle_key(puzzle):
    return tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                 for regime in ("train", "test") for p in puzzle[regime])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for seed, train, test in ((200, 2, 1), (201, 10, 1), (202, 10, 4),
                              (210, None, 1), (211, None, 4),
                              (212, 2, None), (213, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(300, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _puzzle_key(generate_puzzle(12345, 8, 3)) == _puzzle_key(generate_puzzle(12345, 8, 3))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    sizes = set()
    locations = set()
    masks = set()
    traits = set()
    palette_modes = set()
    location_modes = set()
    mixed_sizes = 0
    fixed_three = 0
    compact = 0
    varied_background = 0
    constant_palette = 0
    train_role_swap = 0
    train_both_sides = 0
    small_density_coverage = 0
    novel_test_size = 0
    reused_test_color = 0
    new_test_color = 0
    combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_facts(pair) for pair in puzzle["train"]]
        test = [_facts(pair) for pair in puzzle["test"]]
        train_sizes = {f["n"] for f in train}
        mixed_sizes += len(train_sizes) > 1
        fixed_three += train_sizes == {3} and {f["n"] for f in test} == {3}
        compact += (len(train), len(test)) == (3, 1)
        train_backgrounds = {f["background"] for f in train}
        train_colors = {f["foreground"] for f in train}
        varied_background += len(train_backgrounds) > 1
        constant_palette += len({(f["background"], f["foreground"]) for f in train + test}) == 1
        train_role_swap += any(f["background"] in train_colors and f["foreground"] in train_backgrounds for f in train)
        train_both_sides += (
            any(f["location"][0] == 0 for f in train)
            and any(f["location"][0] == f["n"] - 1 for f in train)
            and any(f["location"][1] == 0 for f in train)
            and any(f["location"][1] == f["n"] - 1 for f in train)
        )
        small_train = {f["density"] for f in train if f["n"] == 2}
        small_density_coverage += small_train == {2, 3}
        novel_test_size += any(f["n"] not in train_sizes for f in test)
        reused_test_color += any(f["foreground"] in train_colors for f in test)
        new_test_color += any(f["foreground"] not in train_colors for f in test)
        palette_modes.add((len({f["background"] for f in train}),
                           len({f["foreground"] for f in train})))
        train_locs = {f["location"] for f in train}
        combined += (len(train_sizes) > 1 and len(train_backgrounds) > 1
                     and any(f["location"] not in train_locs for f in test))
        for fact in test:
            location_modes.add("reuse" if fact["location"] in train_locs else "holdout")
        for fact in train + test:
            sizes.add(fact["n"])
            locations.add((fact["n"], fact["location"]))
            masks.add((fact["n"], _mask_key(fact["mask"])))
            traits.update(name for name, value in _traits(fact["mask"]).items() if value)
    assert sizes == {2, 3, 4, 5}
    assert len(locations) >= 20 and len(masks) >= 90
    assert len(traits) == 7
    assert mixed_sizes >= 3 and fixed_three >= 2 and compact >= 1
    assert any(bg == 1 and fg == 1 for bg, fg in palette_modes)
    assert any(bg >= 2 for bg, _ in palette_modes)
    assert location_modes == {"reuse", "holdout"}
    assert varied_background >= 3 and constant_palette >= 2
    assert train_role_swap >= 3 and train_both_sides >= 25
    assert small_density_coverage >= 3 and novel_test_size >= 1
    assert reused_test_color >= 3 and new_test_color >= 3
    assert combined >= 1
