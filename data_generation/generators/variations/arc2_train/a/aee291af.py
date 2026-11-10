import hashlib

import numpy as np


SOURCE_DIGESTS = {
    "d62424fa9ed160e37bb851f2b0d55776339f037cac7223909003b9492cc70159",
    "1bbb92adcc61d7e6f87d888cb85bd0b1e13b148d3ce3d2a6063d0c99e6222b25",
    "244fdf9a367ad970663efc7114292522b439edbb94fcec64c33b901ae97aa538",
    "4c7de3d115861431e4175d4ffc49cbed12df9d84b2062b6893eaa029afb48935",
}


def _is_integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(seed, rng, num_train, num_test):
    if num_train is not None:
        if not _is_integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be from 1 through 4")
        num_test = int(num_test)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _palette(rng):
    if rng.random() < 0.18:
        return (1, 8, 2)
    return tuple(int(x) for x in rng.choice(10, size=3, replace=False))


def _mask(rng, interior_side, count):
    area = interior_side * interior_side
    result = np.zeros(area, dtype=bool)
    result[rng.choice(area, size=count, replace=False)] = True
    return result.reshape(interior_side, interior_side)


def _motifs(rng, side, mode):
    inner = side - 2
    area = inner * inner
    for _ in range(80):
        if mode == "full_repeat":
            repeated = np.ones((inner, inner), dtype=bool)
            unique = _mask(rng, inner, int(rng.integers(1, area)))
        elif mode == "full_unique":
            repeated = _mask(rng, inner, int(rng.integers(1, area)))
            unique = np.ones((inner, inner), dtype=bool)
        elif mode == "sparse":
            repeated = _mask(rng, inner, 1)
            unique = _mask(rng, inner, int(rng.integers(1, area + 1)))
        elif mode == "one_cell":
            repeated = _mask(rng, inner, int(rng.integers(1, area + 1)))
            unique = repeated.copy()
            candidates = np.flatnonzero(repeated.ravel()) if repeated.all() else np.arange(area)
            index = int(rng.choice(candidates))
            unique.ravel()[index] = ~unique.ravel()[index]
            if not unique.any():
                continue
        elif mode in ("orientation", "equal_count"):
            count = int(rng.integers(1, area))
            repeated = _mask(rng, inner, count)
            if mode == "equal_count":
                unique = _mask(rng, inner, count)
            else:
                alternatives = []
                for k in range(4):
                    rotated = np.rot90(repeated, k)
                    alternatives.extend((rotated, np.fliplr(rotated)))
                alternatives = [
                    a for a in alternatives if not np.array_equal(a, repeated)
                ]
                if not alternatives:
                    continue
                unique = alternatives[int(rng.integers(len(alternatives)))].copy()
        else:
            repeated = _mask(rng, inner, int(rng.integers(1, area + 1)))
            unique = _mask(rng, inner, int(rng.integers(1, area + 1)))
        if repeated.any() and unique.any() and not np.array_equal(repeated, unique):
            return repeated, unique
    raise RuntimeError("could not sample distinct motifs")


def _tile(mask, structural, accent):
    side = mask.shape[0] + 2
    result = np.full((side, side), structural, dtype=np.int8)
    result[1:-1, 1:-1] = np.where(mask, accent, structural)
    return result


def _anchors(rng, length, side, number, edge=None, clear=False):
    slack = length - number * side - (number - 1)
    if slack < 0:
        raise ValueError("canvas does not fit tiles")
    extras = np.zeros(number + 1, dtype=int)
    if clear:
        if slack < 2:
            raise ValueError("canvas lacks clear margins")
        extras[0] = extras[-1] = 1
        slack -= 2
    available = [i for i in range(number + 1) if
                 not ((edge == "first" and i == 0) or
                      (edge == "last" and i == number))]
    if slack:
        extras[available] += rng.multinomial(
            slack, np.full(len(available), 1 / len(available))
        )
    return [
        int(sum(extras[:i + 1]) + i * (side + 1))
        for i in range(number)
    ]


def _positions(rng, height, width, side, count, rows, cols, edge):
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    if edge is None:
        chosen = rng.choice(len(slots), size=count, replace=False)
    else:
        axis, direction = edge
        boundary = [
            i for i, slot in enumerate(slots)
            if slot[axis] == (0 if direction == "first"
                              else (rows - 1 if axis == 0 else cols - 1))
        ]
        required = int(rng.choice(boundary))
        others = [i for i in range(len(slots)) if i != required]
        chosen = [required] + list(rng.choice(others, size=count - 1, replace=False))
    row_edge = edge[1] if edge is not None and edge[0] == 0 else None
    col_edge = edge[1] if edge is not None and edge[0] == 1 else None
    row_anchors = _anchors(rng, height, side, rows, row_edge, edge is None)
    col_anchors = _anchors(rng, width, side, cols, col_edge, edge is None)
    return [(row_anchors[slots[int(i)][0]], col_anchors[slots[int(i)][1]])
            for i in chosen]


def _ring(position, side, height, width):
    r, c = position
    cells = []
    for cc in range(c, c + side):
        if r:
            cells.append((r - 1, cc))
        if r + side < height:
            cells.append((r + side, cc))
    for rr in range(r, r + side):
        if c:
            cells.append((rr, c - 1))
        if c + side < width:
            cells.append((rr, c + side))
    return cells


def _find_tiles(grid, structural, accent, side):
    height, width = grid.shape
    found = []
    for r in range(height - side + 1):
        for c in range(width - side + 1):
            w = grid[r:r + side, c:c + side]
            if not (np.all(w[0] == structural) and
                    np.all(w[-1] == structural) and
                    np.all(w[:, 0] == structural) and
                    np.all(w[:, -1] == structural)):
                continue
            inside = w[1:-1, 1:-1]
            if np.any(inside == accent) and np.all(
                (inside == structural) | (inside == accent)
            ):
                found.append((r, c, w.copy()))
    return found


def _all_tiles(grid, structural, accent):
    return {
        side: found
        for side in range(3, min(grid.shape) + 1)
        if (found := _find_tiles(grid, structural, accent, side))
    }


def _related(a, b):
    a, b = a[1:-1, 1:-1], b[1:-1, 1:-1]
    if np.array_equal(a, b):
        return False
    return any(
        np.array_equal(x, b)
        for k in range(4)
        for x in (np.rot90(a, k), np.fliplr(np.rot90(a, k)))
    )


def _extract(grid, structural, accent):
    found = _all_tiles(grid, structural, accent)
    answers = []
    for side, candidates in found.items():
        info = _extract_side(grid, structural, accent, side, candidates)
        if info is not None:
            answers.append(info)
    return answers[0] if len(answers) == 1 else None


def _extract_side(grid, structural, accent, side, candidates):
    if not 3 <= len(candidates) <= 5:
        return None
    groups = {}
    for item in candidates:
        groups.setdefault(item[2].tobytes(), []).append(item)
    if sorted(map(len, groups.values())) != [1, len(candidates) - 1]:
        return None
    occupied = np.zeros(grid.shape, dtype=bool)
    for r, c, _ in candidates:
        area = occupied[r:r + side, c:c + side]
        if np.any(area):
            return None
        area[:] = True
    if np.any((grid == accent) & ~occupied):
        return None
    unique = next(v[0] for v in groups.values() if len(v) == 1)
    repeated = next(v[0] for v in groups.values() if len(v) > 1)
    height, width = grid.shape
    contacts = []
    edges = []
    for r, c, _ in candidates:
        contacts.append(any(grid[y, x] == structural
                            for y, x in _ring((r, c), side, height, width)))
        edges.append(r == 0 or c == 0 or r + side == height or c + side == width)
    return {
        "output": unique[2],
        "side": side,
        "count": len(candidates),
        "positions": tuple(sorted((r, c) for r, c, _ in candidates)),
        "contacts": tuple(contacts),
        "edges": tuple(edges),
        "full_repeat": bool(np.all(repeated[2][1:-1, 1:-1] == accent)),
        "full_unique": bool(np.all(unique[2][1:-1, 1:-1] == accent)),
        "sparse": (int(np.count_nonzero(repeated[2] == accent)) == 1 or
                   int(np.count_nonzero(unique[2] == accent)) == 1),
        "equal_count": int(np.count_nonzero(repeated[2] == accent)) ==
                       int(np.count_nonzero(unique[2] == accent)),
        "one_cell": int(np.count_nonzero(repeated[2] != unique[2])) == 1,
        "related": _related(repeated[2], unique[2]),
        "noise": int(np.count_nonzero((grid == structural) & ~occupied)),
    }


def _infer(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 3:
        return None
    background = int(colors[int(np.argmax(counts))])
    answers = []
    for structural in colors:
        if structural == background:
            continue
        accent = int(next(x for x in colors if x not in (background, structural)))
        info = _extract(grid, int(structural), accent)
        if info is not None:
            answers.append((background, int(structural), accent, info))
    if len(answers) != 1:
        return None
    return answers[0]


def _generate_pair(rng, palette, side, count, mode, edge_request):
    background, structural, accent = palette
    rows = int(rng.choice((2, 3)))
    cols = (count + rows - 1) // rows
    if rng.random() < 0.5:
        rows, cols = cols, rows
    min_height = rows * side + rows - 1
    min_width = cols * side + cols - 1
    low_height = max(16, min_height + (0 if edge_request else 2))
    low_width = max(16, min_width + (0 if edge_request else 2))
    if low_height > 30 or low_width > 30:
        raise RuntimeError("unsupported layout")
    for _ in range(100):
        repeated_mask, unique_mask = _motifs(rng, side, mode)
        repeated = _tile(repeated_mask, structural, accent)
        unique = _tile(unique_mask, structural, accent)
        height = int(rng.integers(low_height, min(30, low_height + 8) + 1))
        width = int(rng.integers(low_width, min(30, low_width + 8) + 1))
        edge = None
        if edge_request:
            edge = (int(rng.integers(0, 2)),
                    "first" if rng.random() < 0.5 else "last")
        positions = _positions(rng, height, width, side, count, rows, cols, edge)
        grid = np.full((height, width), background, dtype=np.int8)
        occupied = np.zeros(grid.shape, dtype=bool)
        unique_index = int(rng.integers(count))
        for i, (r, c) in enumerate(positions):
            grid[r:r + side, c:c + side] = unique if i == unique_index else repeated
            occupied[r:r + side, c:c + side] = True

        # Keep one or more exterior rings clean in most pairs. Other rings
        # deliberately receive same-color clutter.
        contact_order = rng.permutation(count)
        clean_count = int(rng.choice((0, 1, 1, 2, count)))
        clean_count = min(clean_count, count)
        clean = set(int(i) for i in contact_order[:clean_count])
        blocked = occupied.copy()
        for i, position in enumerate(positions):
            ring = _ring(position, side, height, width)
            if i in clean:
                for r, c in ring:
                    blocked[r, c] = True
            elif ring:
                for index in rng.choice(len(ring), size=min(len(ring),
                                      int(rng.integers(1, 4))), replace=False):
                    r, c = ring[int(index)]
                    if not occupied[r, c]:
                        grid[r, c] = structural
        target_noise = int(rng.integers(
            max(8, int(0.04 * height * width)),
            max(10, int(0.11 * height * width)) + 1,
        ))
        current = int(np.count_nonzero((grid == structural) & ~occupied))
        available = np.argwhere((grid == background) & ~blocked)
        add = min(max(0, target_noise - current), len(available))
        if add:
            for index in rng.choice(len(available), size=add, replace=False):
                r, c = available[int(index)]
                grid[r, c] = structural
        inferred = _infer(grid)
        if inferred is None:
            continue
        _, detected_structural, detected_accent, info = inferred
        if (detected_structural == structural and detected_accent == accent and
            info["side"] == side and info["count"] == count and
            info["positions"] == tuple(sorted(positions)) and
            np.array_equal(info["output"], unique)):
            return grid, unique.copy()
    raise RuntimeError("could not construct an unambiguous pair")


def _grid_key(grid):
    return (grid.shape, np.asarray(grid, dtype=np.uint8).tobytes())


def _signature(puzzle):
    return tuple(
        (_grid_key(p["input"]), _grid_key(p["output"]))
        for section in ("train", "test")
        for p in puzzle[section]
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)

    fixed_palette = rng.random() < 0.4
    first_palette = _palette(rng)
    palettes = [first_palette]
    for index in range(1, num_train + num_test):
        if fixed_palette:
            palettes.append(first_palette)
        else:
            candidate = _palette(rng)
            if index == 1:
                while candidate == first_palette:
                    candidate = _palette(rng)
            palettes.append(candidate)

    first_side = int(rng.integers(4, 7))
    second_side = int(rng.choice([x for x in (4, 5, 6, 7) if x != first_side]))
    train_specs = [
        (first_side, int(rng.integers(3, 6)), "orientation"),
        (second_side, int(rng.integers(3, 6)),
         "full_repeat" if rng.random() < 0.5 else "full_unique"),
    ]
    modes = ("regular", "equal_count", "one_cell", "orientation",
             "full_repeat", "full_unique", "sparse")
    for _ in range(num_train - 2):
        train_specs.append((int(rng.integers(4, 8)),
                            int(rng.integers(3, 6)),
                            modes[int(rng.integers(len(modes)))]))
    test_modes = ("orientation", "equal_count", "one_cell",
                  "full_repeat", "full_unique", "sparse", "regular")
    test_specs = [
        (int(rng.integers(4, 8)), int(rng.integers(3, 6)),
         test_modes[(seed + i) % len(test_modes)])
        for i in range(num_test)
    ]
    specs = train_specs + test_specs
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=len(specs), dtype=np.int64)
    train, test = [], []
    used = set()
    for i, (side, count, mode) in enumerate(specs):
        pair_rng = np.random.default_rng(pair_seeds[i])
        for _ in range(20):
            edge_request = pair_rng.random() < (0.55 if i >= num_train else 0.4)
            input_grid, output_grid = _generate_pair(
                pair_rng, palettes[i], side, count, mode, edge_request
            )
            key = _grid_key(input_grid)
            if key not in used:
                used.add(key)
                pair = {"input": input_grid, "output": output_grid}
                (train if i < num_train else test).append(pair)
                break
        else:
            raise RuntimeError("could not avoid duplicate input")
    train = [train[int(i)] for i in rng.permutation(len(train))]
    test = [test[int(i)] for i in rng.permutation(len(test))]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_infos = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            key = _grid_key(input_grid)
            assert key not in seen
            seen.add(key)
            assert not np.array_equal(input_grid, output_grid)
            if (input_grid.shape, output_grid.shape) in (
                ((16, 16), (4, 4)), ((18, 18), (5, 5))
            ):
                digest = hashlib.sha256(
                    np.asarray(input_grid, dtype=np.uint8).tobytes() +
                    np.asarray(output_grid, dtype=np.uint8).tobytes()
                ).hexdigest()
                assert digest not in SOURCE_DIGESTS
            inferred = _infer(input_grid)
            assert inferred is not None
            background, structural, accent, info = inferred
            assert np.array_equal(info["output"], output_grid)
            assert output_grid.shape == (info["side"], info["side"])
            assert 4 <= info["side"] <= 7 and 3 <= info["count"] <= 5
            assert info["noise"] >= 8
            assert len({background, structural, accent}) == 3
            if section == "train":
                train_infos.append(info)
    # The smallest valid puzzle still demonstrates exact orientation and
    # scale variation. Color roles are visible afresh in every input.
    assert any(info["related"] for info in train_infos)
    assert len({info["side"] for info in train_infos}) >= 2


def self_test() -> None:
    counts_train, counts_test = set(), set()
    palette_modes, palettes, sides, counts, shapes, layouts = (
        set(), set(), set(), set(), set(), set()
    )
    full_repeat = full_unique = sparse = one_cell = equal = related = 0
    edge = clean = dirty = mixed = strong = 0
    test_five = test_large = test_related = test_one_cell = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        pair_palettes = []
        pair_sides, pair_counts = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inferred = _infer(pair["input"])
                assert inferred is not None
                background, structural, accent, info = inferred
                assert np.array_equal(info["output"], pair["output"])
                pair_palettes.append((background, structural, accent))
                palettes.add((background, structural, accent))
                sides.add(info["side"])
                counts.add(info["count"])
                pair_sides.add(info["side"])
                pair_counts.add(info["count"])
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, info["positions"]))
                full_repeat += info["full_repeat"]
                full_unique += info["full_unique"]
                sparse += info["sparse"]
                one_cell += info["one_cell"]
                equal += info["equal_count"]
                related += info["related"]
                edge += any(info["edges"])
                clean += not all(info["contacts"])
                dirty += any(info["contacts"])
                mixed += any(info["contacts"]) and not all(info["contacts"])
                strong += (info["count"] == 5 and any(info["edges"]) and
                           any(info["contacts"]) and not all(info["contacts"]))
                if section == "test":
                    test_five += info["count"] == 5
                    test_large += info["side"] >= 6
                    test_related += info["related"]
                    test_one_cell += info["one_cell"]
        palette_modes.add(len(set(pair_palettes)) == 1)
        assert len(pair_sides) >= 2
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(signatures) == 42
    assert palette_modes == {False, True} and (1, 8, 2) in palettes
    assert len(palettes) >= 20 and sides == {4, 5, 6, 7}
    assert counts == {3, 4, 5} and len(shapes) >= 30
    assert len(layouts) >= 150
    assert min(full_repeat, full_unique, sparse, one_cell, equal, related) >= 5
    assert min(edge, clean, dirty, mixed, strong) >= 5
    assert min(test_five, test_large, test_related, test_one_cell) >= 5
    for seed, ntrain, ntest in (
        (1201, 2, 1), (1202, 10, 1), (1203, 10, 4),
        (1301, None, 1), (1302, None, 4),
        (1303, 2, None), (1304, 10, None),
    ):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for args in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(1400, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
