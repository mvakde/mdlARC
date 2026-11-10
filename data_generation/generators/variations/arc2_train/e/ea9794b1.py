import numpy as np
from itertools import permutations


_PRIORITY = (1, 2, 3, 0)  # source order
_ALL_ORDERS = tuple(permutations(range(4)))


def _grid_from_rows(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIR_ROWS = (
    (
        (
            "0444000033",
            "4400030330",
            "4004403330",
            "0040430030",
            "0044433333",
            "0909900080",
            "9009908808",
            "0009000880",
            "0099080800",
            "9909008088",
        ),
        ("09433", "38339", "43334", "30934", "33333"),
    ),
    (
        (
            "0400400003",
            "0444433333",
            "0400003330",
            "4400033303",
            "0044033000",
            "9090900880",
            "0009000080",
            "9909008880",
            "0099900000",
            "9909088880",
        ),
        ("94983", "33333", "93330", "33393", "33890"),
    ),
    (
        (
            "4000030003",
            "0044433333",
            "4404033333",
            "4440030000",
            "0040433000",
            "0000908088",
            "9090980800",
            "0090080880",
            "9999080008",
            "0990088888",
        ),
        ("38083", "33333", "33333", "39998", "33988"),
    ),
    (
        (
            "0444003030",
            "0400003003",
            "0000000333",
            "0044030333",
            "0444433333",
            "9099000000",
            "9000908080",
            "0090000008",
            "0009088000",
            "0000900888",
        ),
        ("93930", "93083", "00333", "38333", "33333"),
    ),
    (
        (
            "4444433033",
            "4044000303",
            "0040400030",
            "4440030033",
            "4444030303",
            "9990900888",
            "9900980000",
            "0009000000",
            "0900080800",
            "0000908080",
        ),
        ("33933", "99343", "00434", "39833", "38383"),
    ),
    (
        (
            "4000400330",
            "4000033330",
            "0440430033",
            "0440400300",
            "0000400003",
            "0999908008",
            "0099980008",
            "9990088080",
            "9990908888",
            "0990908008",
        ),
        ("49339", "33339", "39933", "99389", "09903"),
    ),
    (
        (
            "4040030330",
            "4000033003",
            "0044400030",
            "0040433330",
            "4444030000",
            "0009900808",
            "0909980008",
            "0009900880",
            "0099980000",
            "9090000880",
        ),
        ("30339", "33093", "00839", "33339", "34980"),
    ),
)


_SOURCE_PAIRS = tuple(
    (_grid_from_rows(input_rows), _grid_from_rows(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _checked_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _checked_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (seed // 4) % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for the requested train count")
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng):
    base_palette = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    if rng.random() < 0.55:
        priority = _PRIORITY
    else:
        alternatives = [p for p in _ALL_ORDERS if p != _PRIORITY]
        priority = alternatives[int(rng.integers(len(alternatives)))]
    return {
        "base_palette": base_palette,
        "priority": priority,
        "variable_palette": bool(rng.random() < 0.62),
    }


def _pair_palette(rng, params, index):
    base = params["base_palette"]
    if not params["variable_palette"] or index == 0:
        return base
    if index == 1:
        # Demonstrate both transparency recoloring and positional, rather
        # than literal-color, priority in the second training example.
        new_background = int(rng.choice([x for x in range(10) if x not in base]))
        inks = list(base[1:])
        shift = int(rng.integers(1, 4))
        return (new_background,) + tuple(inks[shift:] + inks[:shift])
    if rng.random() < 0.28:
        return base
    return tuple(int(x) for x in rng.choice(10, size=5, replace=False))


def _pair_shape(rng, index, shapes):
    if index == 0:
        height = int(rng.integers(4, 10))
        width = int(rng.integers(4, 10))
    elif index == 1:
        height = int(rng.integers(9, 16))
        width = int(rng.integers(7, 16))
    else:
        height = int(rng.integers(4, 16))
        width = int(rng.integers(4, 16))
    if index > 0 and (height, width) == shapes[0]:
        width = 4 + (width - 3) % 12
    return height, width


def _ranked_mask(rng, height, width, density):
    count = max(2, min(height * width - 2, int(round(density * height * width))))
    rows, cols = np.indices((height, width))
    noise = rng.random((height, width))
    style = int(rng.integers(0, 9))

    if style == 0:
        scores = noise
    elif style == 1:
        scores = np.full((height, width), float(height * height + width * width))
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(height)), int(rng.integers(width))
            scores = np.minimum(scores, (rows - r) ** 2 + (cols - c) ** 2)
        scores = scores + 0.3 * noise
    elif style == 2:
        band = rng.random(height)[:, None] if rng.random() < 0.5 else rng.random(width)[None, :]
        scores = band + 0.22 * noise
    elif style == 3:
        coordinate = rows - cols if rng.random() < 0.5 else rows + cols
        offset = int(rng.integers(-width, height + width))
        scores = np.abs(coordinate - offset) + 0.35 * noise
    elif style == 4:
        r0, r1 = sorted(int(x) for x in rng.choice(height, size=2, replace=False))
        c0, c1 = sorted(int(x) for x in rng.choice(width, size=2, replace=False))
        distance = np.maximum.reduce((r0 - rows, rows - r1, c0 - cols, cols - c1))
        inside = np.minimum.reduce((rows - r0, r1 - rows, cols - c0, c1 - cols))
        scores = np.where(distance <= 0, -inside, distance).astype(float) + 0.2 * noise
    elif style == 5:
        r0, r1 = sorted(int(x) for x in rng.choice(height, size=2, replace=False))
        c0, c1 = sorted(int(x) for x in rng.choice(width, size=2, replace=False))
        border = np.minimum.reduce((np.abs(rows-r0), np.abs(rows-r1), np.abs(cols-c0), np.abs(cols-c1)))
        outside = np.maximum.reduce((r0-rows, rows-r1, c0-cols, cols-c1, np.zeros_like(rows)))
        scores = border + 2 * outside + 0.3 * noise
    elif style == 6:
        scores = np.full((height, width), float(max(height, width)))
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < 0.5:
                rr = int(rng.integers(height))
                scores = np.minimum(scores, np.abs(rows - rr))
            else:
                cc = int(rng.integers(width))
                scores = np.minimum(scores, np.abs(cols - cc))
        scores = scores + 0.4 * noise
    elif style == 7:
        scores = noise.copy()
        for _ in range(2):
            scores = (scores + np.roll(scores, 1, 0) + np.roll(scores, -1, 0)
                      + np.roll(scores, 1, 1) + np.roll(scores, -1, 1)) / 5
        scores = scores + 0.12 * noise
    else:
        block_h = int(rng.integers(2, 5))
        block_w = int(rng.integers(2, 5))
        coarse = rng.random(((height + block_h - 1) // block_h,
                             (width + block_w - 1) // block_w))
        scores = coarse[rows // block_h, cols // block_w] + 0.08 * noise

    chosen = np.argsort(scores.ravel(), kind="stable")[:count]
    mask = np.zeros(height * width, dtype=bool)
    mask[chosen] = True
    return mask.reshape(height, width)


def _shift_mask(mask, dr, dc):
    height, width = mask.shape
    result = np.zeros_like(mask)
    source_r = slice(max(0, -dr), min(height, height-dr))
    source_c = slice(max(0, -dc), min(width, width-dc))
    dest_r = slice(max(0, dr), min(height, height+dr))
    dest_c = slice(max(0, dc), min(width, width+dc))
    result[dest_r, dest_c] = mask[source_r, source_c]
    return result


def _masks(rng, height, width, regime):
    profile = int(rng.choice(4, p=(.23, .34, .26, .17) if regime == "source_train"
                             else (.21, .31, .31, .17)))
    if profile == 0:
        densities = rng.uniform(.17, .4, size=4)
    elif profile == 1:
        densities = rng.uniform(.28, .61, size=4)
    elif profile == 2:
        densities = rng.uniform(.48, .77, size=4)
    else:
        densities = rng.uniform(.13, .74, size=4)
    masks = np.stack([_ranked_mask(rng, height, width, d) for d in densities])
    mode = int(rng.choice(3, p=(.52, .23, .25)))
    if mode:
        a, b = (int(x) for x in rng.choice(4, size=2, replace=False))
        if mode == 1:
            dr, dc = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
            related = _shift_mask(masks[a], dr, dc)
            flips = rng.random((height, width)) < rng.uniform(.05, .22)
            masks[b] = np.logical_xor(related, flips)
        else:
            complement = ~masks[a]
            if rng.random() < .38:
                masks[b] = complement
            else:
                flips = rng.random((height, width)) < rng.uniform(.03, .16)
                masks[b] = np.logical_xor(complement, flips)
        if not np.any(masks[b]) or np.all(masks[b]):
            masks[b] = _ranked_mask(rng, height, width, densities[b])
    return masks


def _cap_to_pairwise(rng, masks):
    height, width = masks.shape[1:]
    for row in range(height):
        for col in range(width):
            active = np.flatnonzero(masks[:, row, col])
            if len(active) <= 2:
                continue
            neighbors = []
            for layer in active:
                score = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row+dr, col+dc
                    if 0 <= rr < height and 0 <= cc < width:
                        score += int(masks[layer, rr, cc])
                neighbors.append(score + .01 * rng.random())
            keep = active[np.argsort(neighbors)[-2:]]
            masks[active, row, col] = False
            masks[keep, row, col] = True


def _isolate_layer(rng, masks, layer):
    height, width = masks.shape[1:]
    # One panel contributes a local stencil but has no conflicts in this
    # demonstration. Its priority is established by another training pair.
    stencil = _ranked_mask(rng, height, width, rng.uniform(.16, .36))
    masks[layer] = stencil
    for other in range(4):
        if other != layer:
            masks[other][stencil] = False


def _embed_specs(rng, masks, specs):
    flat = masks.reshape(4, -1)
    used = np.zeros(flat.shape[1], dtype=bool)
    for active in specs:
        target = np.array([layer in active for layer in range(4)], dtype=bool)
        costs = np.count_nonzero(flat != target[:, None], axis=0).astype(float)
        costs[used] = 10.0
        costs += rng.random(flat.shape[1]) * .1
        position = int(np.argmin(costs))
        flat[:, position] = target
        used[position] = True


def _compose(masks, palette, priority):
    background, *inks = palette
    height, width = masks.shape[1:]
    panels = np.full((4, height, width), background, dtype=np.int8)
    for layer in range(4):
        panels[layer][masks[layer]] = inks[layer]
    input_grid = np.concatenate((np.concatenate((panels[0], panels[1]), axis=1),
                                 np.concatenate((panels[2], panels[3]), axis=1)), axis=0)
    output = np.full((height, width), background, dtype=np.int8)
    for layer in reversed(priority):
        output[masks[layer]] = inks[layer]
    return input_grid, output


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def _source_copy(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def _witness_assignments(rng, train_count, priority):
    assignments = [[] for _ in range(train_count)]
    isolated = [None] * train_count
    if rng.random() < .52:
        for i in range(train_count):
            isolated[i] = priority[0] if i % 2 == 0 else priority[3]
        for first in range(4):
            for second in range(first+1, 4):
                pair = (first, second)
                if isolated[0] not in pair:
                    assignments[0].append(pair)
                elif isolated[1] not in pair:
                    assignments[1].append(pair)
        # Top versus bottom need not be observed directly: adjacent
        # comparisons across the two demonstrations determine their order.
    else:
        spread = int(rng.integers(1, min(4, train_count) + 1))
        pairs = [(a, b) for a in range(4) for b in range(a+1, 4)]
        rng.shuffle(pairs)
        for i, pair in enumerate(pairs):
            assignments[i % spread].append(pair)
    roles = [(), (0,), (1,), (2,), (3,)]
    rng.shuffle(roles)
    for i, role in enumerate(roles):
        assignments[i % train_count].append(role)
    return assignments, isolated


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _checked_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    assignments, isolated = _witness_assignments(rng, train_count, params["priority"])
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count+test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    shapes = []
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        regime = "source_train" if index < train_count else "target_test"
        palette = _pair_palette(pair_rng, params, index)
        height, width = _pair_shape(pair_rng, index, shapes)
        shapes.append((height, width))
        specs = assignments[index] if index < train_count else []
        for _ in range(40):
            masks = _masks(pair_rng, height, width, regime)
            if pair_rng.random() < .22:
                _cap_to_pairwise(pair_rng, masks)
            if index < train_count and isolated[index] is not None:
                _isolate_layer(pair_rng, masks, isolated[index])
            if specs:
                _embed_specs(pair_rng, masks, specs)
            if np.any(masks.sum(axis=(1, 2)) == 0) or np.any(masks.sum(axis=(1, 2)) == height*width):
                continue
            inp, out = _compose(masks, palette, params["priority"])
            key = _grid_key(inp)
            if key not in seen and not _source_copy(inp, out):
                break
        else:
            raise RuntimeError("could not construct a distinct valid pair")
        seen.add(key)
        puzzle["train" if index < train_count else "test"].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.all((grid >= 0) & (grid <= 9))


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    h, w = out.shape
    assert inp.shape == (2*h, 2*w)
    panels = (inp[:h, :w], inp[:h, w:], inp[h:, :w], inp[h:, w:])
    color_sets = [set(int(x) for x in np.unique(panel)) for panel in panels]
    assert all(len(s) == 2 for s in color_sets)
    common = set.intersection(*color_sets)
    assert len(common) == 1
    background = next(iter(common))
    inks = tuple(next(iter(s-{background})) for s in color_sets)
    assert len(set(inks)) == 4
    masks = np.stack([panel != background for panel in panels])
    counts = masks.sum(axis=0)
    return (h, w), (background,) + inks, masks, counts


def _predicted(masks, palette, priority):
    output = np.full(masks.shape[1:], palette[0], dtype=np.int8)
    for layer in reversed(priority):
        output[masks[layer]] = palette[layer+1]
    return output


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _assert_grid(pair["input"])
            _assert_grid(pair["output"])
            ikey = _grid_key(pair["input"])
            pkey = (ikey, _grid_key(pair["output"]))
            assert ikey not in seen_inputs and pkey not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            assert not _source_copy(pair["input"], pair["output"])
            analyses[split].append(_analyze_pair(pair))

    # Enumerate all positional orders using only the returned training data.
    possible = []
    for priority in _ALL_ORDERS:
        if all(np.array_equal(_predicted(masks, palette, priority), pair["output"])
               for (_, palette, masks, _), pair in zip(analyses["train"], puzzle["train"])):
            possible.append(priority)
    assert len(possible) == 1, "training does not identify one positional order"
    priority = possible[0]
    for split in ("train", "test"):
        for (_, palette, masks, _), pair in zip(analyses[split], puzzle[split]):
            assert np.array_equal(_predicted(masks, palette, priority), pair["output"])

    train_palettes = [item[1] for item in analyses["train"]]
    varied = len(set(train_palettes)) > 1
    if varied:
        assert len({p[0] for p in train_palettes}) > 1
        positions = {}
        for palette in train_palettes:
            for layer, ink in enumerate(palette[1:]):
                positions.setdefault(ink, set()).add(layer)
        assert any(len(v) > 1 for v in positions.values())
    else:
        assert all(item[1] == train_palettes[0] for item in analyses["test"])

    # The training set, rather than each pair, exposes transparency and the
    # contribution of every layer. Pairwise examples above expose the order.
    assert any(np.any(counts == 0) for _, _, _, counts in analyses["train"])
    for layer in range(4):
        assert any(np.any((counts == 1) & masks[layer])
                   for _, _, masks, counts in analyses["train"])
    assert len({item[0] for item in analyses["train"]}) > 1


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _expect_value_error(train_count, test_count):
    try:
        generate_puzzle(123456, train_count, test_count)
    except ValueError:
        return
    raise AssertionError("illegal count request was accepted")


def self_test() -> None:
    a = generate_puzzle(91001, 7, 3)
    assert _puzzles_equal(a, generate_puzzle(91001, 7, 3))
    assert not _puzzles_equal(a, generate_puzzle(91002, 7, 3))
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(30000+i, tr, te)
        validate_generated_puzzle(puzzle)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        _expect_value_error(tr, te)

    train_counts, test_counts, shapes, palettes, orders = set(), set(), set(), set(), set()
    overlap_patterns, geometry = set(), set()
    variable_palette = constant_palette = 0
    rich_train = rich_test = simple_train = simple_test = zero_test_background = 0
    hidden_train_layer = distributed_order = square_pairs = rectangular_pairs = 0
    source_order = other_order = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_analyses = [_analyze_pair(p) for p in puzzle["train"]]
        test_analyses = [_analyze_pair(p) for p in puzzle["test"]]
        palettes.add(train_analyses[0][1])
        varied = len({a[1] for a in train_analyses}) > 1
        variable_palette += varied
        constant_palette += not varied
        distributed_order += all(
            sum(np.array_equal(_predicted(m, p, order), pair["output"])
                for order in _ALL_ORDERS) > 1
            for (_, p, m, _), pair in zip(train_analyses, puzzle["train"]))
        for order in _ALL_ORDERS:
            if all(np.array_equal(_predicted(m, p, order), pair["output"])
                   for (_, p, m, _), pair in zip(train_analyses, puzzle["train"])):
                orders.add(order)
                source_order += order == _PRIORITY
                other_order += order != _PRIORITY
        rich_train += any(np.any(c == 3) and np.any(c == 4) for _, _, _, c in train_analyses)
        rich_test += any(np.any(c == 3) and np.any(c == 4) for _, _, _, c in test_analyses)
        simple_train += any(np.max(c) <= 2 for _, _, _, c in train_analyses)
        simple_test += any(np.max(c) <= 2 for _, _, _, c in test_analyses)
        zero_test_background += any(np.all(c > 0) for _, _, _, c in test_analyses)
        hidden_train_layer += any(
            any(ink not in np.unique(pair["output"]) for ink in palette[1:])
            for (_, palette, _, _), pair in zip(train_analyses, puzzle["train"]))
        for split, items in (("train", train_analyses), ("test", test_analyses)):
            for shape, palette, masks, counts in items:
                total += 1
                shapes.add(shape)
                square_pairs += shape[0] == shape[1]
                rectangular_pairs += shape[0] != shape[1]
                histogram = tuple(int(np.count_nonzero(counts == n)) for n in range(5))
                overlap_patterns.add((split, shape, histogram))
                geometry.add((split, shape, histogram, tuple(int(m.sum()) for m in masks)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(palettes) >= 30 and len(orders) >= 5
    assert variable_palette >= 12 and constant_palette >= 5
    assert source_order >= 5 and other_order >= 5 and distributed_order >= 5
    assert rich_train >= 5 and rich_test >= 5
    assert simple_train >= 5 and simple_test >= 2
    assert zero_test_background >= 2
    assert hidden_train_layer >= 5
    assert square_pairs >= 5 and rectangular_pairs >= 50
    assert len(overlap_patterns) >= int(.60*total)
    assert len(geometry) >= int(.85*total)
