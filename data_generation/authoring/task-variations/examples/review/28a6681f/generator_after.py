import numpy as np


_SOURCE_TEXT = (
    (
        (
            "0000000011",
            "0000000011",
            "0000000311",
            "0000000311",
            "0000000311",
            "0000030311",
            "0000030311",
            "0000230311",
            "0000230311",
            "0222233311",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000311",
            "0000000311",
            "0000000311",
            "0000031311",
            "0000031311",
            "0000231311",
            "0000231311",
            "0222233311",
        ),
    ),
    (
        (
            "0030000000",
            "0030000001",
            "0030000111",
            "0030000111",
            "0030001111",
            "0030031111",
            "0030035115",
            "0031135115",
            "0031135115",
            "0033335555",
        ),
        (
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0031111111",
            "0031131111",
            "0031135115",
            "0031135115",
            "0031135115",
            "0033335555",
        ),
    ),
    (
        (
            "0000000000",
            "9911000099",
            "0991000990",
            "0099000900",
            "0009009900",
            "0000099000",
            "0000040000",
            "2004040000",
            "2024040000",
            "2224440000",
        ),
        (
            "0000000000",
            "9900000099",
            "0990000990",
            "0099000900",
            "0009009900",
            "0000099000",
            "0000040000",
            "2004140000",
            "2124140000",
            "2224440000",
        ),
    ),
    (
        (
            "1100000000",
            "9911000000",
            "0991000000",
            "0099900005",
            "0000999005",
            "0000000505",
            "3000000005",
            "3300005505",
            "3330055505",
            "3333555505",
        ),
        (
            "0000000000",
            "9900000000",
            "0990000000",
            "0099900005",
            "0000999005",
            "0000000505",
            "3000000005",
            "3300005515",
            "3331155515",
            "3333555515",
        ),
    ),
)


def _text_grid(lines: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in lines],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_text_grid(input_lines), _text_grid(output_lines))
    for input_lines, output_lines in _SOURCE_TEXT
)


def _component(
    mask: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [start]
    seen[start] = True

    while stack:
        row, column = stack.pop()

        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and mask[next_row, next_column]
                and not seen[next_row, next_column]
            ):
                seen[next_row, next_column] = True
                stack.append((next_row, next_column))

    return seen


def _component_sizes(mask: np.ndarray) -> list[int]:
    unseen = mask.copy()
    sizes: list[int] = []

    while np.any(unseen):
        row, column = np.argwhere(unseen)[0]
        component = _component(
            unseen,
            (int(row), int(column)),
        )
        sizes.append(int(np.count_nonzero(component)))
        unseen[component] = False

    return sorted(sizes)


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        point = tuple(map(int, np.argwhere(remaining)[0]))
        part = _component(remaining, point)
        remaining[part] = False
        result.append(part)
    return result


def _settle(grid, background, water, reverse=False, top_first=False):
    """Pour unit cells downwards, spreading only when supported.

    Construction accepts only complete, contained horizontal levels. The
    ordering switches are adversarial checks, not puzzle conventions.
    """
    result = grid.copy()
    points = sorted(map(tuple, np.argwhere(grid == water)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    result[result == water] = background
    height, width = grid.shape
    for row, col in points:
        row, col = int(row), int(col)
        while row > 0 and result[row, col] == water:
            row -= 1
        if result[row, col] != background:
            raise ValueError("a source is trapped beneath a full ceiling")
        # Every fall increases row; each horizontal move ends in a fall.
        for _ in range(height + 1):
            if row + 1 < height and result[row + 1, col] == background:
                row += 1
                continue
            left = right = col
            while left > 0 and result[row, left - 1] == background:
                left -= 1
            while right + 1 < width and result[row, right + 1] == background:
                right += 1
            exits = [] if row + 1 == height else [
                c for c in range(left, right + 1)
                if result[row + 1, c] == background]
            if not exits:
                result[row, col] = water
                break
            col = min(exits, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
        else:
            raise AssertionError("downward motion did not terminate")
    return result


def _capture(terrain, pools):
    """All possible dry downward routes, with pools as absorbing regions.

    A bit mask records EVERY lateral outlet, not one preferred tie break.
    Thus a source is accepted only if every route enters its assigned pool.
    This is deliberately separate from the cell-by-cell pouring algorithm.
    """
    height, width = terrain.shape
    unknown = 1 << len(pools)
    labels = np.zeros(terrain.shape, dtype=np.int64)
    fixed = np.zeros(terrain.shape, dtype=np.int64)
    for index, pool in enumerate(pools):
        fixed[pool] = 1 << index
    for row in range(height - 1, -1, -1):
        col = 0
        while col < width:
            if terrain[row, col]:
                col += 1
                continue
            end = col
            while end + 1 < width and not terrain[row, end + 1]:
                end += 1
            outlets = 0
            for c in range(col, end + 1):
                outlets |= int(fixed[row, c])
                if row + 1 < height and not terrain[row + 1, c]:
                    outlets |= int(labels[row + 1, c])
            for c in range(col, end + 1):
                if fixed[row, c]:
                    labels[row, c] = fixed[row, c]
                elif row + 1 < height and not terrain[row + 1, c]:
                    labels[row, c] = labels[row + 1, c]
                else:
                    labels[row, c] = outlets or unknown
            col = end + 1
    return labels


def _branches(pool):
    """Maximum number of lower pockets joined by this water body."""
    top = int(np.nonzero(pool)[0].min())
    return max([1] + [len(_components(pool &
                (np.arange(pool.shape[0])[:, None] >= row)))
                for row in range(top + 1, pool.shape[0])])


def _terrain(rng, height, width, simple=False):
    """Compose grounded masses, open cups, stepped walls and deflectors."""
    solid = np.zeros((height, width), dtype=bool)
    if simple:
        # A guaranteed open comb, still with random spacing and elevations.
        profile = np.full(width, height - 1, dtype=int)
        profile[1::3] = height - 3
        for c, floor in enumerate(profile):
            solid[floor:, c] = True
        return solid
    # Independently placed solid masses with flat, sloping or jagged tops.
    for _ in range(int(rng.integers(1, 5))):
        left = int(rng.integers(0, width - 1))
        right = int(rng.integers(left + 1, width + 1))
        floor = int(rng.integers(max(3, height * 2 // 3), height + 1))
        style = int(rng.integers(0, 4))
        slope = int(rng.choice([-1, 1]))
        for c in range(left, right):
            if style == 1:
                floor += slope
            elif style == 2:
                floor += int(rng.integers(-2, 3))
            elif style == 3:
                floor += slope
                if c % 3 == 0:
                    slope *= -1
            floor = int(np.clip(floor, height // 2, height))
            solid[floor:, c] = True
    # Cups may share supports, sit at different heights, or use canvas edges.
    for _ in range(int(rng.integers(1, min(6, width // 3 + 1)))):
        span = int(rng.integers(3, min(width, 12) + 1))
        left = int(rng.integers(0, width - span + 1))
        right = left + span - 1
        bottom = int(rng.integers(max(4, height // 2), height))
        previous = bottom
        for c in range(left, right + 1):
            floor = int(np.clip(previous + int(rng.choice([-1, 0, 0, 0, 1])),
                                max(3, bottom - 2), min(height - 1, bottom + 1)))
            solid[min(previous, floor):max(previous, floor) + 1, c] = True
            previous = floor
        for c in (left, right):
            base = int(np.nonzero(solid[:, c])[0].max())
            top = max(1, base - int(rng.integers(2, min(height - 1, 10))))
            thickness = int(rng.choice([1, 1, 1, 2]))
            solid[top:base + 1, max(0, c - thickness + 1):c + 1] = True
    # Thin independent shelves, diagonals and stepped deflectors, never noise.
    for _ in range(int(rng.integers(0, 5))):
        row = int(rng.integers(1, max(2, height - 3)))
        col = int(rng.integers(0, width - 2))
        length = int(rng.integers(2, min(9, width - col) + 1))
        direction = int(rng.integers(-1, 2))
        for c in range(col, col + length):
            next_row = int(np.clip(row + direction * int(rng.integers(0, 2)),
                                   1, height - 2))
            solid[min(row, next_row):max(row, next_row) + 1, c] = True
            row = next_row
    if rng.integers(0, 2):
        solid = solid[:, ::-1].copy()
    return solid


def _pool_candidates(terrain):
    height, width = terrain.shape
    candidates = []
    rows = np.arange(height)[:, None]
    for top in range(max(2, height // 3), height):
        for pool in _components(~terrain & (rows >= top)):
            volume = int(pool.sum())
            if not 2 <= volume <= min(100, height * width // 4):
                continue
            if not pool[top].any():
                continue
            # A visible open mouth permits liquid to arrive from above.
            if not np.any(pool[top] & ~terrain[top - 1]):
                continue
            candidates.append(pool)
    return candidates


def _select_pools(rng, terrain, mode):
    candidates = _pool_candidates(terrain)
    rng.shuffle(candidates)
    if mode == 0:
        candidates.sort(key=lambda p: _branches(p) != 1)
    elif mode == 1:
        candidates.sort(key=lambda p: _branches(p) < 2)
    if not candidates:
        return []
    first = candidates[0]
    if mode == 0 and _branches(first) != 1:
        return []
    if mode == 1 and _branches(first) < 2:
        return []
    selected = [first]
    wanted = 1 if mode == 0 else int(rng.integers(2, 5))
    occupied = first.copy()
    for pool in candidates[1:]:
        if len(selected) == wanted:
            break
        touching = occupied.copy()
        touching[1:] |= occupied[:-1]
        touching[:-1] |= occupied[1:]
        touching[:, 1:] |= occupied[:, :-1]
        touching[:, :-1] |= occupied[:, 1:]
        if not np.any(pool & touching):
            selected.append(pool)
            occupied |= pool
    return selected


def _source(rng, eligible, target, style, tier):
    """Compact, scattered, banded, filamentary, and partly settled supplies."""
    volume = int(target.sum())
    height, _ = eligible.shape
    points = np.argwhere(eligible)
    outside = np.argwhere(eligible & ~target)
    if not len(outside):
        raise ValueError("no moving source in catchment")
    row_min = (0, height // 4, height // 2, 0)[tier]
    preferred = outside[outside[:, 0] >= row_min]
    if len(preferred):
        outside = preferred
    anchor = outside[int(rng.integers(len(outside)))]
    dr = np.abs(points[:, 0] - anchor[0])
    dc = np.abs(points[:, 1] - anchor[1])
    jitter = rng.random(len(points))
    if style == 0:
        score = dr + dc + jitter
    elif style == 1:
        score = jitter
    elif style == 2:
        score = 3 * dr + dc * 0.1 + jitter
    elif style == 3:
        score = 3 * dc + dr * 0.2 + jitter
    else:
        score = (~target[points[:, 0], points[:, 1]]) * 3 + jitter
    score += (points[:, 0] < row_min) * (height * 10)
    chosen = points[np.argsort(score)[:volume]].copy()
    if not np.any(np.all(chosen == anchor, axis=1)):
        chosen[-1] = anchor
    source = np.zeros(eligible.shape, dtype=bool)
    source[chosen[:, 0], chosen[:, 1]] = True
    assert int(source.sum()) == volume
    return source


def _paint(terrain, palette, background, rng):
    grid = np.full(terrain.shape, background, dtype=np.int8)
    cells = np.argwhere(terrain)
    if len(cells) < len(palette):
        raise ValueError("too little terrain for palette")
    # Spatial color regions keep structures readable even when they touch.
    anchors = cells[rng.choice(len(cells), size=len(palette), replace=False)]
    distances = ((cells[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
    colors = np.asarray(palette, dtype=np.int8)[np.argmin(distances, axis=1)]
    grid[cells[:, 0], cells[:, 1]] = colors
    return grid


def _make_pair(rng, background, water, palette, mode, serial, regime):
    for attempt in range(25):
        simple = attempt == 24
        if simple:
            height, width = 12 + serial % 7, 12 + serial
        else:
            height = int(rng.choice([10, int(rng.integers(8, 31))]))
            width = int(rng.choice([10, int(rng.integers(8, 31))]))
        terrain = _terrain(rng, height, width, simple)
        pools = _select_pools(rng, terrain, mode)
        if not pools:
            continue
        if regime == "target_test" and len(pools) < 2 and _branches(pools[0]) < 2:
            continue
        labels = _capture(terrain, pools)
        source = np.zeros(terrain.shape, dtype=bool)
        try:
            for index, pool in enumerate(pools):
                source |= _source(rng, labels == (1 << index), pool,
                                  int(rng.integers(5)), int(rng.integers(4)))
            inp = _paint(terrain, palette, background, rng)
            out = inp.copy()
            inp[source] = water
            for pool in pools:
                out[pool] = water
            # Four orders must agree; full basin closure and catchment
            # containment additionally rule out partially resolved splits.
            if all(np.array_equal(_settle(inp, background, water, rev, order), out)
                   for rev in (False, True) for order in (False, True)):
                return {"input": inp, "output": out}
        except ValueError:
            continue
    raise AssertionError("constructive fallback failed")


def _is_integer_argument(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (not _is_integer_argument(value) or not low <= int(value) <= high):
            raise ValueError("invalid pair count")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not _is_integer_argument(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    background, water = map(int, rng.choice(10, size=2, replace=False))
    colors = [c for c in range(10) if c not in (background, water)]
    # There are 162 subsets of sizes 1..4: every pair gets a distinct palette.
    palette_masks = [m for m in range(1, 256) if m.bit_count() <= 4]
    rng.shuffle(palette_masks)
    puzzle = {"train": [], "test": []}
    serial = 0
    for split, count, regime in (("train", num_train, "source_train"),
                                ("test", num_test, "target_test")):
        for index in range(count):
            mask = palette_masks[serial]
            palette = [colors[k] for k in range(8) if mask & (1 << k)]
            # The first two demonstrations explicitly show retention and
            # joining/overflow. Tests recombine those cases with extra pools.
            mode = (0 if index == 0 else 1 if index == 1 else
                    int(rng.integers(3))) if split == "train" else 2
            puzzle[split].append(_make_pair(rng, background, water, palette,
                                             mode, serial, regime))
            serial += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _roles(train):
    """Infer roles from visible changes and downward displacement only."""
    candidates = None
    for pair in train:
        inp, out = pair["input"], pair["output"]
        changed = inp != out
        values = list(map(int, np.unique(np.concatenate((inp[changed], out[changed])))))
        assert len(values) == 2
        possible = set()
        for water in values:
            background = values[1] if water == values[0] else values[0]
            before, after = np.argwhere(inp == water), np.argwhere(out == water)
            if len(before) == len(after) and len(before) and after[:, 0].sum() > before[:, 0].sum():
                possible.add((background, water))
        candidates = possible if candidates is None else candidates & possible
    assert candidates is not None and len(candidates) == 1
    return next(iter(candidates))


def predict(train_pairs, test_input):
    """Input-only prediction entry point; no seed or held-out output is read."""
    background, water = _roles(train_pairs)
    return _settle(test_input, background, water)


def _inspect_pair(inp, out, background, water):
    assert inp.shape == out.shape and np.any(inp != out)
    terrain = (inp != background) & (inp != water)
    assert np.array_equal(inp[terrain], out[terrain])
    assert np.array_equal(terrain, (out != background) & (out != water))
    assert int((inp == water).sum()) == int((out == water).sum()) > 0
    assert (np.argwhere(out == water)[:, 0].sum() >
            np.argwhere(inp == water)[:, 0].sum())
    pools = _components(out == water)
    levels = []
    for pool in pools:
        top = int(np.nonzero(pool)[0].min())
        levels.append(top)
        # A pool must fill its entire basin at its surface level. This tests
        # support, horizontal menisci, and no sideways/downward escape.
        available = ~terrain & (np.arange(inp.shape[0])[:, None] >= top)
        closure = _component(available, tuple(map(int, np.argwhere(pool)[0])))
        assert np.array_equal(closure, pool), "unfilled level or leaking basin"
    labels = _capture(terrain, pools)
    assigned = np.zeros(inp.shape, dtype=bool)
    for index, pool in enumerate(pools):
        supply = (inp == water) & (labels == (1 << index))
        assert int(supply.sum()) == int(pool.sum()), "ambiguous basin allocation"
        assigned |= supply
    assert np.array_equal(assigned, inp == water)
    assert np.array_equal(_settle(inp, background, water), out)
    return {
        "background": background, "water": water,
        "shape": inp.shape,
        "palette": tuple(map(int, np.unique(inp[terrain]))),
        "terrain_components": len(_components(terrain)),
        "source_components": len(_components(inp == water)),
        "pools": len(pools), "levels": tuple(sorted(set(levels))),
        "branches": max(_branches(pool) for pool in pools),
        "volume": int((inp == water).sum()),
        "source_min_row": int(np.nonzero(inp == water)[0].min()),
        "source_max_row": int(np.nonzero(inp == water)[0].max()),
        "retained_cells": int(((inp == water) & (out == water)).sum()),
        "lateral_redistribution": bool(np.any((inp == water).sum(axis=0) !=
                                               (out == water).sum(axis=0))),
        "boundary_pool": bool(np.any(out[-1] == water) or
                              np.any(out[:, 0] == water) or np.any(out[:, -1] == water)),
    }


def _key(grid):
    return grid.shape, grid.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
    background, water = _roles(puzzle["train"])
    seen, palettes, train_stats = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            assert _key(inp) not in seen
            seen.add(_key(inp))
            assert not any(np.array_equal(inp, a) and np.array_equal(out, b)
                           for a, b in _SOURCE_PAIRS)
            stats = _inspect_pair(inp, out, background, water)
            assert stats["palette"] and stats["palette"] not in palettes
            palettes.add(stats["palette"])
            assert np.array_equal(predict(puzzle["train"], inp.copy()), out)
            if split == "train":
                train_stats.append(stats)
            else:
                assert stats["pools"] >= 2 or stats["branches"] >= 2
    assert train_stats[0]["pools"] == 1 and train_stats[0]["branches"] == 1
    assert train_stats[1]["branches"] >= 2


def self_test() -> None:
    # Exact source evidence is a regression, not an independent reference solver.
    for inp, out in _SOURCE_PAIRS:
        for reverse in (False, True):
            for order in (False, True):
                assert np.array_equal(_settle(inp, 0, 1, reverse, order), out)
    # Exercise the bounded fallback directly at every possible pair index.
    for serial in range(14):
        for mode in range(3):
            rng = np.random.default_rng(100 * serial + mode)
            terrain = _terrain(rng, 12 + serial % 7, 12 + serial, True)
            pools = _select_pools(rng, terrain, mode)
            assert pools
            assert mode != 1 or _branches(pools[0]) >= 2
            assert mode != 2 or len(pools) > 1 or _branches(pools[0]) >= 2
            labels = _capture(terrain, pools)
            source = np.zeros(terrain.shape, dtype=bool)
            for index, pool in enumerate(pools):
                source |= _source(rng, labels == (1 << index), pool,
                                  serial % 5, mode)
            inp = _paint(terrain, [2, 3, 4, 5], 0, rng)
            out = inp.copy()
            inp[source] = 1
            for pool in pools:
                out[pool] = 1
            assert np.array_equal(_settle(inp, 0, 1), out)
    signatures, layouts, train_counts, test_counts = set(), set(), set(), set()
    backgrounds, waters, shapes, volumes, records = set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        background, water = _roles(puzzle["train"])
        backgrounds.add(background)
        waters.add(water)
        local = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                stats = _inspect_pair(inp, out, background, water)
                records.append(stats)
                local.append(stats)
                shapes.add(inp.shape)
                volumes.add(stats["volume"])
                normalized = np.where(inp == background, 0, np.where(inp == water, 2, 1))
                layouts.add(_key(normalized.astype(np.int8)))
                for reverse in (False, True):
                    for order in (False, True):
                        assert np.array_equal(_settle(inp, background, water, reverse, order), out)
        assert len({s["palette"] for s in local}) == len(local)
        assert {s["branches"] > 1 for s in local} == {False, True}
    assert len(signatures) == 42 and len(layouts) >= 200
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 8 and len(waters) >= 8
    assert len(shapes) >= 50 and len(volumes) >= 30
    assert sum(s["pools"] > 1 for s in records) >= 50
    assert sum(len(s["levels"]) > 1 for s in records) >= 30
    assert sum(s["source_components"] > 1 for s in records) >= 100
    assert sum(s["source_min_row"] >= s["shape"][0] // 2 for s in records) >= 30
    assert sum(s["retained_cells"] > 0 for s in records) >= 100
    assert sum(s["lateral_redistribution"] for s in records) >= 100
    assert sum(s["boundary_pool"] for s in records) >= 30
    assert len({s["terrain_components"] for s in records}) >= 5
    assert sum(s["pools"] > 1 and len(s["levels"]) > 1 and
               s["source_components"] > 1 and s["background"] != 0
               for s in records) >= 20
    for counts in ((None, None), (2, 1), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        first = generate_puzzle(2026, *counts)
        second = generate_puzzle(2026, *counts)
        assert _signature(first) == _signature(second)
        for split, count in zip(("train", "test"), counts):
            assert count is None or len(first[split]) == count
    for counts in ((1, 1), (11, 1), (3, 0), (6, 5), (2, 2), (3, 4),
                   (2.5, 1), (True, 1), (2, False)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
