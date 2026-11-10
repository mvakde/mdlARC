"""Synthetic ARC puzzles: a 2x2 color code grows a downward unary bar."""

import numpy as np


_SOURCE_PAIRS = (
    (("0000000000", "0048000000", "0094000000", "0000000000",
      "0000002100", "0000001200", "0000000000", "0000000000",
      "0000000000", "0000000000"),
     ("0000000000", "0048000000", "0094000000", "0033000000",
      "0033002100", "0033001200", "0000003300", "0000003300",
      "0000000000", "0000000000")),
    (("0000000000", "0000000000", "0990000000", "0660000000",
      "0000000000", "0000084000", "0000077000", "0000000000",
      "0000000000", "0000000000"),
     ("0000000000", "0000000000", "0990000000", "0660000000",
      "0330000000", "0330084000", "0000077000", "0000033000",
      "0000033000", "0000033000")),
    (("0000000000", "0024009800", "0067008900", "0000000000",
      "0000000000", "0000760000", "0000660000", "0000000000",
      "0000000000", "0000000000"),
     ("0000000000", "0024009800", "0067008900", "0033003300",
      "0033003300", "0033760000", "0033660000", "0000330000",
      "0000330000", "0000000000")),
    (("0000000000", "1100002900", "2100001600", "0004700000",
      "0008400000", "0000000000", "0000000000", "0000000000",
      "0000000000", "0000000000"),
     ("0000000000", "1100002900", "2100001600", "3304703300",
      "3308403300", "0003303300", "0003303300", "0003300000",
      "0000000000", "0000000000")),
)


def _integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(rng, num_train, num_test):
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_test is None:
        options = np.arange(1, min(4, (num_train or 10) - 1) + 1)
        weights = np.asarray([4, 3, 2, 1], dtype=float)[:len(options)]
        num_test = int(rng.choice(options, p=weights / weights.sum()))
    if num_train is None:
        options = np.arange(max(2, num_test + 1), 11)
        weights = np.asarray([5, 6, 5, 4, 3, 2, 1, 1, 1], dtype=float)[options - 2]
        num_train = int(rng.choice(options, p=weights / weights.sum()))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.ravel()))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _source_copy(pair):
    rows = lambda grid: tuple("".join(str(int(x)) for x in row) for row in grid)
    return (rows(pair["input"]), rows(pair["output"])) in _SOURCE_PAIRS


def _compatible(candidate, count, positions, counts):
    r, c = candidate
    for (other_r, other_c), other_count in zip(positions, counts):
        # A blank row or column separates input tiles. Their downward footprints
        # must also be disjoint and must never pass through another code tile.
        if not (r + 2 < other_r or other_r + 2 < r or
                c + 2 < other_c or other_c + 2 < c):
            return False
        if c < other_c + 2 and other_c < c + 2:
            if r < other_r + 2 + other_count and other_r < r + 2 + count:
                return False
    return True


def _dimensions(rng, counts, mode):
    k = len(counts)
    if mode == "row":
        min_width = 3 * k - 1
        width = int(rng.integers(max(7, min_width), 27))
        height = int(rng.integers(max(7, 2 + max(counts)), 24))
    elif mode == "column":
        min_height = sum(2 + n for n in counts) + k - 1
        height = int(rng.integers(max(7, min_height), 31))
        width = int(rng.integers(7, 20))
    else:
        shape_mode = int(rng.integers(4))
        if shape_mode == 0:  # compact, including source-sized grids
            height = int(rng.integers(7, 12))
            width = int(rng.integers(7, 12))
        elif shape_mode == 1:
            height = int(rng.integers(7, 14))
            width = int(rng.integers(13, 26))
        elif shape_mode == 2:
            height = int(rng.integers(13, 26))
            width = int(rng.integers(7, 14))
        else:
            height = int(rng.integers(12, 25))
            width = int(rng.integers(12, 25))
    return height, width


def _positions(rng, height, width, counts, mode):
    k = len(counts)
    if mode == "row":
        slack = width - (3 * k - 1)
        gaps = rng.multinomial(slack, [1 / (k + 1)] * (k + 1))
        row = int(rng.integers(0, height - 1 - max(counts)))
        column = int(gaps[0])
        result = []
        for i in range(k):
            result.append((row, column))
            column += 3 + int(gaps[i + 1])
        return result
    if mode == "column":
        slack = height - (sum(2 + n for n in counts) + k - 1)
        gaps = rng.multinomial(slack, [1 / (k + 1)] * (k + 1))
        column = int(rng.integers(0, width - 1))
        row = int(gaps[0])
        result = []
        for i, count in enumerate(counts):
            result.append((row, column))
            row += 3 + count + int(gaps[i + 1])
        return result

    result = []
    placed_counts = []
    for count in counts:
        candidates = [
            (r, c)
            for r in range(height - 1 - count)
            for c in range(width - 1)
            if _compatible((r, c), count, result, placed_counts)
            and (mode != "stagger" or not result or
                 (all(r != rr for rr, _ in result) and
                  all(c != cc for _, cc in result)))
            and (mode != "edge" or result or
                 r == 0 or c == 0 or c == width - 2)
        ]
        if not candidates:
            return None
        result.append(candidates[int(rng.integers(len(candidates)))])
        placed_counts.append(count)
    return result


def _tile_arrays(rng, counts, payload):
    mode = str(rng.choice(["shared", "disjoint", "random"]))
    sets = []
    if mode == "disjoint" and sum(counts) <= len(payload):
        order = rng.permutation(payload)
        offset = 0
        for n in counts:
            sets.append(list(order[offset:offset + n]))
            offset += n
    else:
        pool = payload
        if mode == "shared":
            pool_size = int(rng.integers(max(counts), min(len(payload), max(counts) + 2) + 1))
            pool = list(rng.choice(payload, size=pool_size, replace=False))
        for n in counts:
            sets.append(list(rng.choice(pool, size=n, replace=False)))

    # Near-identical color sets make a one-color change visible as one extra row.
    consecutive = [(i, j) for i, n in enumerate(counts)
                   for j, m in enumerate(counts) if m == n + 1]
    if mode != "disjoint" and consecutive and rng.random() < 0.35:
        i, j = consecutive[int(rng.integers(len(consecutive)))]
        base = list(rng.choice(payload, size=counts[i], replace=False))
        extra = int(rng.choice([x for x in payload if x not in base]))
        sets[i], sets[j] = base, base + [extra]

    arrays = []
    for chosen in sets:
        n = len(chosen)
        if n == 1:
            values = chosen * 4
        elif n == 2:
            values = ([chosen[0]] * 2 + [chosen[1]] * 2 if rng.random() < 0.5
                      else [chosen[0]] * 3 + [chosen[1]])
        elif n == 3:
            values = chosen + [chosen[int(rng.integers(3))]]
        else:
            values = chosen
        rng.shuffle(values)
        arrays.append(np.asarray(values, dtype=np.int8).reshape(2, 2))
    return arrays


def _make_pair(rng, counts, background, construction, payload):
    modes = ["free", "row", "column", "stagger", "edge"]
    for _ in range(80):
        mode = str(rng.choice(modes))
        height, width = _dimensions(rng, counts, mode)
        positions = _positions(rng, height, width, counts, mode)
        if positions is None:
            continue
        tiles = _tile_arrays(rng, counts, payload)
        input_grid = np.full((height, width), background, dtype=np.int8)
        for (r, c), tile in zip(positions, tiles):
            input_grid[r:r + 2, c:c + 2] = tile
        output_grid = input_grid.copy()
        for (r, c), n in zip(positions, counts):
            assert np.all(input_grid[r + 2:r + 2 + n, c:c + 2] == background)
            output_grid[r + 2:r + 2 + n, c:c + 2] = construction
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not place the requested tiles")


def _train_plans(rng, num_train, include_one):
    if include_one:
        plans = [[1, 2], [3, 4, int(rng.choice([2, 3, 4]))]]
        repertoire = [1, 2, 3, 4]
    else:
        plans = [[2, 3], [4, int(rng.choice([2, 3])),
                          int(rng.choice([2, 3, 4]))]]
        repertoire = [2, 3, 4]
    while len(plans) < num_train:
        k = int(rng.integers(1, 5))
        plans.append([int(x) for x in rng.choice(repertoire, size=k)])
    rng.shuffle(plans)
    for plan in plans:
        rng.shuffle(plan)
    return plans, repertoire


def _test_counts(rng, repertoire):
    k = int(rng.integers(1, 5))
    # Give these useful repeated-count contrasts a visible share of the
    # distribution without tying them to any particular placement or palette.
    if k == 2 and rng.random() < 0.25:
        return [3, 3]
    if k == 3 and rng.random() < 0.25:
        counts = [2, 2, 4]
        rng.shuffle(counts)
        return counts
    style = int(rng.integers(3))
    if style == 0:
        counts = [int(rng.choice(repertoire))] * k
    elif style == 1 and k >= 3:
        two = rng.choice(repertoire, size=2, replace=False)
        counts = [int(two[0])] * (k - 1) + [int(two[1])]
    else:
        counts = [int(x) for x in rng.choice(repertoire, size=k)]
    rng.shuffle(counts)
    return counts


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _pair_counts(rng, num_train, num_test)

    # The construction color is learned once per puzzle from the demonstrations.
    background = int(rng.choice(10, p=[0.25] + [0.75 / 9] * 9))
    construction = int(rng.choice([x for x in range(10) if x != background]))
    payload = [x for x in range(10) if x not in (background, construction)]
    include_one = bool(rng.random() < 0.35)
    train_plans, repertoire = _train_plans(rng, num_train, include_one)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, plans in (("train", train_plans),
                         ("test", [_test_counts(rng, repertoire) for _ in range(num_test)])):
        for counts in plans:
            for _ in range(30):
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _make_pair(child, counts, background, construction, payload)
                key = _grid_key(pair["input"])
                if key not in seen and not _source_copy(pair):
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not generate a new pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(component)
    return result


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    difference = inp != out
    assert difference.any()
    old, new = np.unique(inp[difference]), np.unique(out[difference])
    assert len(old) == len(new) == 1
    background, construction = int(old[0]), int(new[0])
    assert background != construction and not np.any(inp == construction)
    assert np.count_nonzero(inp == background) > inp.size // 2
    components = _components(inp != background)
    assert 1 <= len(components) <= 4
    tiles = []
    for comp in components:
        assert len(comp) == 4
        rr = [r for r, _ in comp]
        cc = [c for _, c in comp]
        r, c = min(rr), min(cc)
        assert max(rr) == r + 1 and max(cc) == c + 1
        assert np.all(inp[r:r + 2, c:c + 2] != background)
        n = len(np.unique(inp[r:r + 2, c:c + 2]))
        assert 1 <= n <= 4
        tiles.append((r, c, n))
    expected = inp.copy()
    mask = np.zeros(inp.shape, dtype=bool)
    for i, (r, c, n) in enumerate(tiles):
        for other_r, other_c, other_n in tiles[:i]:
            assert _compatible((r, c), n, [(other_r, other_c)], [other_n])
        assert r + 2 + n <= inp.shape[0]
        region = (slice(r + 2, r + 2 + n), slice(c, c + 2))
        assert np.all(inp[region] == background)
        assert not np.any(mask[region])
        mask[region] = True
        expected[region] = construction
    assert np.array_equal(mask, difference)
    assert np.array_equal(expected, out)
    return background, construction, tiles, len(set(int(x) for x in np.unique(inp)) - {background})


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _check_grid(pair["input"])
            _check_grid(pair["output"])
            key = _grid_key(pair["input"])
            pair_key = _pair_key(pair)
            assert key not in keys and pair_key not in pair_keys
            assert not _source_copy(pair)
            keys.add(key)
            pair_keys.add(pair_key)
            analyses[split].append(_analyze(pair))
    roles = {(bg, ink) for split in analyses for bg, ink, _, _ in analyses[split]}
    assert len(roles) == 1
    train_counts = {n for _, _, tiles, _ in analyses["train"] for _, _, n in tiles}
    test_counts = {n for _, _, tiles, _ in analyses["test"] for _, _, n in tiles}
    assert {2, 3, 4}.issubset(train_counts)
    assert test_counts <= train_counts
    # The inferred one-row extension must be explicitly demonstrated.
    if 1 in test_counts:
        assert 1 in train_counts


def _puzzle_key(puzzle):
    return tuple((split, tuple(_pair_key(p) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    roles, shapes, layouts, train_edges, test_edges = set(), set(), set(), 0, 0
    tile_counts = {"train": set(), "test": set()}
    palettes = {"train": set(), "test": set()}
    regimes = {"train": set(), "test": set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, ink, tiles, palette = _analyze(pair)
                roles.add((bg, ink))
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, tuple(sorted(tiles))))
                tile_counts[split].add(len(tiles))
                palettes[split].add(palette)
                regimes[split].add(tuple(sorted(n for _, _, n in tiles)))
                edge = any(r == 0 or c == 0 or c == pair["input"].shape[1] - 2
                           for r, c, _ in tiles)
                if split == "train":
                    train_edges += edge
                else:
                    test_edges += edge
    assert len(roles) >= 15 and len(shapes) >= 30 and len(layouts) >= 80
    assert {1, 2, 3, 4} <= tile_counts["test"]
    assert {1, 2, 3, 4} <= tile_counts["train"]
    assert len(palettes["train"] & palettes["test"]) >= 3
    assert len(regimes["test"]) >= 8
    assert train_edges >= 5 and test_edges >= 5
    assert all(len({len(_analyze(pair)[2]) for pair in puzzle["train"]}) >= 2
               for puzzle in puzzles)
    for seed, nt, ns in ((500, 2, 1), (501, 10, 1), (502, 10, 4),
                         (600, None, 1), (601, None, 4),
                         (602, 2, None), (603, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    assert _puzzle_key(generate_puzzle(700, 7, 3)) == _puzzle_key(generate_puzzle(700, 7, 3))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(991, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count accepted")
