"""Generators for aligned bars whose colors encode relative length."""

import numpy as np


# Exact source pairs, retained solely for the source-copy check.
_SOURCE_PAIR_ROWS = (
    (
        ("000000000", "000000050", "000000050", "000000050", "000500050",
         "000505050", "000505050", "050505050", "050505050"),
        ("000000000", "000000010", "000000010", "000000010", "000200010",
         "000203010", "000203010", "040203010", "040203010"),
    ),
    (
        ("000005000", "050005000", "050005000", "050505000", "050505000",
         "050505000", "050505050", "050505050", "050505050"),
        ("000001000", "020001000", "020001000", "020301000", "020301000",
         "020301000", "020301040", "020301040", "020301040"),
    ),
    (
        ("000000000", "050000000", "050005000", "050005000", "050005050",
         "050005050", "050505050", "050505050", "050505050"),
        ("000000000", "010000000", "010002000", "010002000", "010002030",
         "010002030", "010402030", "010402030", "010402030"),
    ),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _signature(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _pair_signature(pair):
    return _signature(pair["input"]), _signature(pair["output"])


def _source_copy(pair):
    rows = tuple(tuple("".join(str(int(v)) for v in row) for row in pair[key])
                 for key in ("input", "output"))
    return rows in _SOURCE_PAIR_ROWS


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        cells = [start]
        while stack:
            r, c = stack.pop()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    cells.append(neighbor)
        result.append(cells)
    return result


def _geometry(mask):
    """Reconstruct rectangles, direction, shared baseline and length ranks."""
    components = _components(mask)
    _require(2 <= len(components) <= 8, "Need two to eight disconnected bars")
    boxes = []
    for cells in components:
        rr = [p[0] for p in cells]
        cc = [p[1] for p in cells]
        box = (min(rr), max(rr), min(cc), max(cc))
        _require(len(cells) == (box[1] - box[0] + 1) * (box[3] - box[2] + 1),
                 "Bars must be solid rectangles")
        boxes.append(box)

    candidates = []
    for axis in (0, 1):
        for side in (0, 1):
            lengths = [b[1] - b[0] + 1 if axis == 0 else b[3] - b[2] + 1
                       for b in boxes]
            widths = [b[3] - b[2] + 1 if axis == 0 else b[1] - b[0] + 1
                      for b in boxes]
            if len(set(lengths)) != len(lengths):
                continue
            if any(w > 2 or (length == 1 and w != 1) or
                   (length > 1 and length <= w)
                   for length, w in zip(lengths, widths)):
                continue
            baseline = [b[side] if axis == 0 else b[2 + side] for b in boxes]
            if len(set(baseline)) != 1:
                continue
            tracks = [(b[2], b[3]) if axis == 0 else (b[0], b[1])
                      for b in boxes]
            order = sorted(range(len(boxes)), key=lambda j: tracks[j][0])
            if any(tracks[a][1] + 1 >= tracks[b][0]
                   for a, b in zip(order, order[1:])):
                continue
            candidates.append((axis, side, baseline[0], lengths, widths, order))
    _require(len(candidates) == 1, "Bar orientation and common baseline must be unique")
    axis, side, baseline, lengths, widths, order = candidates[0]
    descending = sorted(lengths, reverse=True)
    bars = []
    for i in order:
        bars.append({
            "cells": tuple(components[i]),
            "length": lengths[i],
            "width": widths[i],
            "rank": descending.index(lengths[i]) + 1,
            "box": boxes[i],
        })
    return {"axis": axis, "side": side, "baseline": baseline, "bars": bars,
            "order": tuple(bar["rank"] for bar in bars)}


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    _require(inp.shape == out.shape, "Geometry and dimensions must be preserved")
    input_colors = set(int(x) for x in np.unique(inp))
    output_colors = set(int(x) for x in np.unique(out))
    _require(len(input_colors) == 2, "Input requires one background and one source color")
    common = input_colors & output_colors
    _require(len(common) == 1, "Only the background may retain its input color")
    background = next(iter(common))
    source = next(iter(input_colors - common))
    mask = inp == source
    _require(np.array_equal(mask, out != background), "Bar footprints must be preserved")
    _require(int(np.count_nonzero(mask)) < inp.size / 2,
             "Background must be identifiable as the majority input color")
    geometry = _geometry(mask)
    rank_colors = [None] * len(geometry["bars"])
    for bar in geometry["bars"]:
        observed = {int(out[r, c]) for r, c in bar["cells"]}
        _require(len(observed) == 1, "A bar must have one output color")
        rank_colors[bar["rank"] - 1] = observed.pop()
    _require(len(set(rank_colors)) == len(rank_colors),
             "Each displayed rank needs a distinct color")
    _require(background not in rank_colors and source not in rank_colors,
             "Rank colors must be distinct from pair-level role colors")
    _require(output_colors == {background, *rank_colors}, "Unexpected output colors")
    return {"background": background, "source": source,
            "rank_colors": tuple(rank_colors), "geometry": geometry,
            "shape": tuple(inp.shape), "count": len(rank_colors)}


def _training_palette(train):
    mapping = {}
    for pair in train:
        analysis = _analyze(pair)
        for rank, color in enumerate(analysis["rank_colors"], 1):
            if rank in mapping:
                _require(mapping[rank] == color, "Conflicting rank color in training")
            mapping[rank] = color
    _require(set(mapping) == set(range(1, max(mapping) + 1)),
             "Training rank palette must be contiguous")
    _require(len(set(mapping.values())) == len(mapping), "Rank colors must be distinct")
    return tuple(mapping[i] for i in range(1, len(mapping) + 1))


def _predict_from_training(train, test_input):
    """Uses training labels and an input only; never reads a test label."""
    palette = _training_palette(train)
    colors, counts = np.unique(test_input, return_counts=True)
    _require(len(colors) == 2, "Test input must have two colors")
    background = int(colors[int(np.argmax(counts))])
    source = int(colors[int(np.argmin(counts))])
    geometry = _geometry(test_input == source)
    _require(len(geometry["bars"]) <= len(palette), "Unseen rank in test")
    prediction = np.full(test_input.shape, background, dtype=np.int8)
    for bar in geometry["bars"]:
        for r, c in bar["cells"]:
            prediction[r, c] = palette[bar["rank"] - 1]
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "Both splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair must contain exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grid must be a 2D NumPy array")
                _require(all(1 <= dim <= 30 for dim in grid.shape),
                         "Grid dimension outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid must contain integers")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "Illegal ARC color")
            key = _pair_signature(pair)
            _require(key[0] not in seen_inputs and key not in seen_pairs,
                     "Duplicate input or pair")
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            _require(not _source_copy(pair), "Exact source pair copied")
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "Input and output are identical")
            analyses[split].append(_analyze(pair))

    palette = _training_palette(train)
    _require(max(a["count"] for a in analyses["train"]) == len(palette),
             "Training must show every rank")
    for split in ("train", "test"):
        for pair, a in zip(puzzle[split], analyses[split]):
            _require(a["rank_colors"] == palette[:a["count"]],
                     "Rank palette differs across pairs")
            _require(a["background"] not in palette and a["source"] not in palette,
                     "Pair role color collides with rank palette")
            predicted = _predict_from_training(train, pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]),
                     "Output is not implied by training and input")

    # The first two demonstrations must defeat both absolute-length and
    # fixed-lateral-position explanations, including with only two examples.
    train_lengths = {}
    for a in analyses["train"]:
        for bar in a["geometry"]["bars"]:
            train_lengths.setdefault(bar["length"], set()).add(bar["rank"])
    _require(any(len(ranks) >= 2 for ranks in train_lengths.values()),
             "Training lacks a same-length, different-rank witness")
    _require(any(
        x["count"] == y["count"] and
        all(a != b for a, b in zip(x["geometry"]["order"], y["geometry"]["order"]))
        for i, x in enumerate(analyses["train"])
        for y in analyses["train"][i + 1:]),
        "Training must move ranks between all lateral positions")
    if any(len({bar["width"] for bar in a["geometry"]["bars"]}) > 1
           for a in analyses["train"] + analyses["test"]):
        _require(any(
            any(b["rank"] > 1 and b["length"] * b["width"] >
                a["geometry"]["bars"][next(i for i, v in enumerate(a["geometry"]["bars"])
                                          if v["rank"] == 1)]["length"]
                for b in a["geometry"]["bars"])
            for a in analyses["train"]),
            "Mixed widths need a training example where area and length disagree")


def _lengths_with_anchor(rng, count, anchor, anchor_rank, near):
    if near:
        # The first training pair has close competitors. A width-two
        # second-ranked bar then has more area than the longest bar.
        return list(range(anchor + anchor_rank - 1,
                          anchor + anchor_rank - 1 - count, -1))
    ceiling = min(29, max(anchor + count + 2, 24))
    lower, upper = 1, ceiling
    above = np.arange(anchor + 1, upper + 1)
    below = np.arange(lower, anchor)
    need_above, need_below = anchor_rank - 1, count - anchor_rank
    if len(above) < need_above:
        above = np.arange(anchor + 1, 30)
    if len(below) < need_below:
        below = np.arange(1, anchor)
    sampled = [anchor]
    sampled.extend(int(x) for x in rng.choice(above, need_above, replace=False))
    sampled.extend(int(x) for x in rng.choice(below, need_below, replace=False))
    return sorted(sampled, reverse=True)


def _free_lengths(rng, count):
    mode = int(rng.integers(0, 3))
    if mode == 0:
        low = int(rng.integers(1, max(2, 27 - count)))
        pool = np.arange(low, min(30, low + count + int(rng.integers(1, 5))))
    elif mode == 1:
        pool = np.arange(1, int(rng.integers(max(count + 1, 10), 30)))
    else:
        pool = np.arange(1, 30)
    return sorted((int(x) for x in rng.choice(pool, count, replace=False)), reverse=True)


def _widths(rng, count, lengths, style, first_train):
    widths = [1] * count
    if style == "simple":
        return widths
    if first_train:
        widths[1] = 2
        return widths
    if rng.random() < (0.3 if style == "moderate" else 0.65):
        eligible = [i for i, length in enumerate(lengths) if length >= 3]
        rng.shuffle(eligible)
        for i in eligible[:min(len(eligible), count // 2)]:
            if rng.random() < 0.7:
                widths[i] = 2
    return widths


def _draw_pair(rng, rank_colors, count, order, lengths, orientation,
               internal, style, roles, first_train):
    widths_by_rank = _widths(rng, count, lengths, style, first_train)
    track_widths = [widths_by_rank[rank - 1] for rank in order]
    total_width = sum(track_widths)
    min_lateral = max(total_width + count - 1, 2 * total_width + 1)
    if style == "simple" and rng.random() < 0.5:
        lateral = int(rng.integers(min_lateral, min(30, min_lateral + 6) + 1))
    else:
        lateral = int(rng.integers(min_lateral, 31))
    max_length = lengths[0]
    if internal:
        after = int(rng.integers(1, min(5, 31 - max_length)))
    else:
        after = 0
    minimum_longitudinal = max_length + after
    if style == "simple" and rng.random() < 0.65:
        longitudinal = int(rng.integers(
            minimum_longitudinal, min(30, minimum_longitudinal + 5) + 1))
    else:
        longitudinal = int(rng.integers(minimum_longitudinal, 31))
    baseline = longitudinal - 1 - after

    # Distribute slack over both margins and the spaces between tracks.
    slack = lateral - (total_width + count - 1)
    probabilities = rng.dirichlet(np.ones(count + 1))
    extras = rng.multinomial(slack, probabilities)
    positions = []
    cursor = int(extras[0])
    for i, width in enumerate(track_widths):
        positions.append(cursor)
        cursor += width + 1 + int(extras[i + 1])

    background, source = roles
    inp = np.full((longitudinal, lateral), background, dtype=np.int8)
    out = inp.copy()
    for position, rank in zip(positions, order):
        length = lengths[rank - 1]
        width = widths_by_rank[rank - 1]
        rows = slice(baseline - length + 1, baseline + 1)
        cols = slice(position, position + width)
        inp[rows, cols] = source
        out[rows, cols] = rank_colors[rank - 1]
    return {"input": np.rot90(inp, orientation).copy(),
            "output": np.rot90(out, orientation).copy()}


def _deranged_order(rng, first):
    count = len(first)
    shift = int(rng.integers(1, count))
    return tuple(first[(i + shift) % count] for i in range(count))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    style = str(rng.choice(("simple", "moderate", "rich"), p=(0.20, 0.36, 0.44)))
    if style == "simple":
        max_count = 4
    else:
        max_count = int(rng.choice(np.arange(2, 9),
                                   p=(0.10, 0.13, 0.29, 0.15, 0.13, 0.11, 0.09)))
    color_order = [int(v) for v in rng.permutation(10)]
    rank_colors = tuple(color_order[:max_count])
    free_colors = color_order[max_count:]
    role_options = [(a, b) for a in free_colors for b in free_colors if a != b]
    rng.shuffle(role_options)
    # For the first two pairs, change both roles even with only two free colors.
    first_roles = role_options[0]
    second_roles = next(role for role in role_options
                        if role[0] != first_roles[0] and role[1] != first_roles[1])
    roles = [first_roles, second_roles]
    roles.extend(role_options[(i + 2) % len(role_options)]
                 for i in range(train_count + test_count - 2))

    total = train_count + test_count
    orient_mode = str(rng.choice(("homogeneous", "mixed", "shift"),
                                 p=(0.24, 0.43, 0.33)))
    base = int(rng.integers(0, 4))
    orientations = [base] * total
    if orient_mode == "mixed":
        orientations[1] = (base + int(rng.integers(1, 4))) % 4
        for i in range(2, total):
            orientations[i] = int(rng.integers(0, 4))
    elif orient_mode == "shift":
        for i in range(train_count, total):
            orientations[i] = (base + int(rng.integers(1, 4))) % 4

    if style == "simple":
        internal_flags = [False] * total
    else:
        internal_flags = [bool(rng.random() < 0.48) for _ in range(total)]
        if style == "rich":
            internal_flags[0], internal_flags[1] = False, True

    counts = [max_count, max_count]
    for i in range(2, total):
        if max_count == 2:
            counts.append(2)
        elif style == "simple" and rng.random() < 0.75:
            counts.append(max_count)
        else:
            counts.append(int(rng.integers(2, max_count + 1)))
    if style == "rich" and max_count > 2 and total > 2:
        counts[-1] = max_count - 1

    first_order = tuple(int(v) for v in rng.permutation(np.arange(1, max_count + 1)))
    first_observed = first_order[::-1] if orientations[0] in (1, 2) else first_order
    second_observed = _deranged_order(rng, first_observed)
    second_order = second_observed[::-1] if orientations[1] in (1, 2) else second_observed
    orders = [first_order, second_order]
    train_orders = {max_count: {first_order, second_order}}
    for i in range(2, total):
        n = counts[i]
        if i >= train_count and rng.random() < 0.35 and n in train_orders:
            pool = tuple(sorted(train_orders[n]))
            order = pool[int(rng.integers(0, len(pool)))]
        else:
            order = tuple(int(v) for v in rng.permutation(np.arange(1, n + 1)))
        orders.append(order)
        if i < train_count:
            train_orders.setdefault(n, set()).add(order)

    # The same absolute length changes rank across the first two examples.
    anchor = int(rng.integers(max_count + 2, 19))
    first_rank = int(rng.integers(1, max_count + 1))
    second_rank = 1 + first_rank % max_count
    lengths = [
        _lengths_with_anchor(rng, max_count, anchor, first_rank, True),
        _lengths_with_anchor(rng, max_count, anchor, second_rank, False),
    ]
    lengths.extend(_free_lengths(rng, counts[i]) for i in range(2, total))
    puzzle = {"train": [], "test": []}
    seen = set()
    for i in range(total):
        for attempt in range(12):
            pair = _draw_pair(rng, rank_colors, counts[i], orders[i], lengths[i],
                              orientations[i], internal_flags[i], style, roles[i], i == 0)
            if _signature(pair["input"]) not in seen and not _source_copy(pair):
                break
        else:
            raise AssertionError("Could not make a distinct pair")
        seen.add(_signature(pair["input"]))
        puzzle["train" if i < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(np.array_equal(x[k], y[k])
                   for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(12345)
    _require(_same_puzzle(automatic, generate_puzzle(12345)),
             "Automatic generation is not deterministic")
    _require(not _same_puzzle(automatic, generate_puzzle(12346)),
             "Different seeds produced identical puzzles")
    for seed, tr, te in ((20000, 2, 1), (20001, 10, 1), (20002, 10, 4),
                         (20003, 7, 3), (20004, None, 1), (20005, None, 4),
                         (20006, 2, None), (20007, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        if tr is not None:
            _require(len(puzzle["train"]) == tr, "Explicit train count ignored")
        if te is not None:
            _require(len(puzzle["test"]) == te, "Explicit test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(22000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, shapes, structures = set(), set(), set(), set()
    max_counts, orientations, roles = set(), set(), set()
    all_orders = {"train": set(), "test": set()}
    interior = edge = singleton = thick = mixed_width = 0
    within_shapes = within_backgrounds = within_sources = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        local = [_analyze(pair) for split in ("train", "test") for pair in puzzle[split]]
        max_counts.add(max(a["count"] for a in local))
        within_shapes += len({a["shape"] for a in local}) > 1
        within_backgrounds += len({a["background"] for a in local}) > 1
        within_sources += len({a["source"] for a in local}) > 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                shapes.add(a["shape"])
                orientations.add((a["geometry"]["axis"], a["geometry"]["side"]))
                roles.add((a["background"], a["source"]))
                all_orders[split].add(a["geometry"]["order"])
                bars = a["geometry"]["bars"]
                singleton += any(b["length"] == 1 for b in bars)
                thick += any(b["width"] == 2 for b in bars)
                mixed_width += len({b["width"] for b in bars}) > 1
                axis, side = a["geometry"]["axis"], a["geometry"]["side"]
                dim = a["shape"][axis]
                interior += 0 < a["geometry"]["baseline"] < dim - 1
                edge += a["geometry"]["baseline"] in (0, dim - 1)
                structures.add((a["shape"], axis, side, a["geometry"]["baseline"],
                                tuple((b["length"], b["width"], b["box"]) for b in bars)))
                _require(np.array_equal(_predict_from_training(puzzle["train"],
                                                                pair["input"].copy()),
                                        pair["output"]), "Training-only prediction failed")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Count diversity collapsed")
    _require(max_counts == set(range(2, 9)), "Bar-count range collapsed")
    _require(len(shapes) >= 50 and len(structures) >= 150, "Geometry diversity collapsed")
    _require(len(orientations) == 4 and interior > 20 and edge > 20,
             "Orientation or baseline diversity collapsed")
    _require(singleton > 5 and thick > 10 and mixed_width > 10,
             "Length or thickness diversity collapsed")
    _require(within_shapes > 35 and within_backgrounds == 42 and within_sources == 42,
             "Within-puzzle pair variation collapsed")
    _require(len(roles) >= 20 and len(all_orders["train"]) >= 20 and
             len(all_orders["test"]) >= 15, "Colors or rank permutations collapsed")
