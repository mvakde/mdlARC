"""Corner-controlled in-place square rotations."""

import numpy as np


_DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
_COMBOS = tuple((s, d) for s in range(4) for d in range(4) if s != d)
_SOURCE_INPUTS = frozenset({
    "0000000000/0000000000/0040000000/0008180000/0001880000/0401110000/0000000000/0404000000/0000000000/0000000000/0000000000",
    "0000000000/0404000000/0000004000/0401110000/0008180000/0008110000/0000000000/0000000000/0000000000/0000000000",
    "00000000000/00000000000/00000000000/00000004000/00001110000/00008880000/00001810400/00000000000/00000040400/00000000000/00000000000",
    "0000000000/0000000000/0000000000/0040400000/0000000000/0040818000/0000818000/0000118000/0000000400/0000000000/0000000000/0000000000",
    "00000000000/00000000000/00000040400/00000000000/00001110400/00001880000/00001810000/00040000000/00000000000/00000000000/00000000000",
})


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is outside {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _geometry(top, left, size, corner):
    dr, dc = _DIRECTIONS[corner]
    cr = top + (size - 1 if dr == 1 else 0)
    cc = left + (size - 1 if dc == 1 else 0)
    center = (cr + dr, cc + dc)
    socket = frozenset(((cr + 2 * dr, cc),
                        (cr, cc + 2 * dc),
                        (cr + 2 * dr, cc + 2 * dc)))
    return center, socket


def _training_combos(rng, count):
    if count >= 3:
        covering = ((0, 2, 1, 3), (0, 2, 3, 1),
                    (1, 3, 0, 2), (1, 3, 2, 0),
                    (2, 0, 1, 3), (2, 0, 3, 1),
                    (3, 1, 0, 2), (3, 1, 2, 0))
        corners = covering[int(rng.integers(0, len(covering)))]
    else:
        corners = [int(x) for x in rng.permutation(4)]
    source, destination_a, destination_b, second_source = corners
    result = [(source, destination_a), (source, destination_b)]
    if count >= 3:
        result.append((second_source, destination_a))
    reserved_source = destination_b if count <= 7 and rng.integers(0, 4) != 0 else None
    reserved_destination = second_source if count <= 7 and rng.integers(0, 4) != 0 else None
    if 8 <= count <= 9 and rng.integers(0, 4) != 0:
        if rng.integers(0, 2):
            reserved_source = destination_b
        else:
            reserved_destination = second_source
    while len(result) < count:
        unused = [x for x in _COMBOS if x not in result
                  and x[0] != reserved_source and x[1] != reserved_destination]
        sources = {x[0] for x in result}
        destinations = {x[1] for x in result}
        turns = {(d - s) % 4 for s, d in result}
        scores = [4 * (s not in sources) + 4 * (d not in destinations)
                  + 3 * (((d - s) % 4) not in turns) + int(rng.integers(0, 3))
                  for s, d in unused]
        result.append(unused[int(np.argmax(scores))])
    return [result[int(i)] for i in rng.permutation(len(result))]


def _test_combos(rng, train, count):
    if len(train) == 2:
        return [train[int(rng.integers(0, 2))]]
    known_sources = {s for s, _ in train}
    known_destinations = {d for _, d in train}
    repeated = list(train)
    recombined = [x for x in _COMBOS if x not in train
                  and x[0] in known_sources and x[1] in known_destinations]
    heldout = [x for x in _COMBOS if x not in train
               and (x[0] not in known_sources or x[1] not in known_destinations)]
    buckets = [repeated, recombined, heldout]
    result = []
    if count == 1:
        order = [int(x) for x in rng.permutation(3)]
    elif heldout:
        order = [2] + [int(x) for x in rng.permutation(2)]
    else:
        order = [int(x) for x in rng.permutation(2)]
    for i in order:
        candidates = [x for x in buckets[int(i)] if x not in result]
        if candidates and len(result) < count:
            result.append(candidates[int(rng.integers(0, len(candidates)))])
    remaining = [x for x in _COMBOS if x not in result]
    while len(result) < count:
        result.append(remaining.pop(int(rng.integers(0, len(remaining)))))
    return [result[int(i)] for i in rng.permutation(len(result))]


def _sample_pattern(rng, size, colors, turns):
    kinds = ("accent", "band", "diagonal", "corner", "irregular")
    for _ in range(200):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        labels = np.zeros((size, size), dtype=np.int8)
        if kind == "accent":
            labels.ravel()[int(rng.integers(0, size * size))] = 1
        elif kind == "band":
            index = int(rng.integers(0, size))
            if rng.integers(0, 2):
                labels[index, :] = 1
            else:
                labels[:, index] = 1
        elif kind == "diagonal":
            main = bool(rng.integers(0, 2))
            for i in range(size):
                labels[i, i if main else size - 1 - i] = 1
        elif kind == "corner":
            labels[0, :] = 1
            labels[:, 0] = 1
        else:
            labels = rng.integers(0, len(colors), (size, size), dtype=np.int8)
        if kind != "irregular":
            labels = np.rot90(labels, int(rng.integers(0, 4)))
            if len(colors) > 2:
                free = [i for i in range(size * size) if labels.ravel()[i] == 0]
                if len(free) < len(colors) - 2:
                    continue
                chosen = rng.choice(free, size=len(colors) - 2, replace=False)
                for label, position in enumerate(chosen, start=2):
                    labels.ravel()[int(position)] = label
        if set(int(x) for x in np.unique(labels)) != set(range(len(colors))):
            continue
        pattern = np.asarray(colors, dtype=np.int8)[labels]
        if not np.array_equal(pattern, np.rot90(pattern, -turns)):
            return pattern
    raise RuntimeError("could not sample a visible motif")


def _axis_layout(rng, minimum, maximum):
    footprint = maximum - minimum + 1
    kind = int(rng.integers(0, 3))
    if kind == 0:
        extent = min(30, footprint + int(rng.integers(0, 4)))
    elif kind == 1:
        extent = int(rng.integers(max(footprint, 9), max(footprint, 18) + 1))
    else:
        extent = int(rng.integers(max(footprint, 19), 31))
    before = int(rng.integers(0, extent - footprint + 1))
    return extent, before - minimum


def _make_pair(rng, regime, combo, size, palette):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    source, destination = combo
    turns = (destination - source) % 4
    assert turns in (1, 2, 3)
    background, marker, *colors = palette
    source_cell, _ = _geometry(0, 0, size, source)
    destination_cell, socket = _geometry(0, 0, size, destination)
    marker_cells = set(socket) | {source_cell}
    assert len(marker_cells) == 4
    rows = [0, size - 1] + [r for r, _ in marker_cells]
    cols = [0, size - 1] + [c for _, c in marker_cells]
    height, top = _axis_layout(rng, min(rows), max(rows))
    width, left = _axis_layout(rng, min(cols), max(cols))
    pattern = _sample_pattern(rng, size, colors, turns)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[top:top + size, left:left + size] = pattern
    for row, col in marker_cells:
        grid[top + row, left + col] = marker
    output = grid.copy()
    for row, col in marker_cells:
        output[top + row, left + col] = background
    output[top:top + size, left:left + size] = np.rot90(pattern, -turns)
    output[top + destination_cell[0], left + destination_cell[1]] = marker
    return grid, output


def _signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def _source_string(grid):
    return "/".join("".join(str(int(x)) for x in row) for row in grid)


def _input_predictions(grid):
    """Reconstruct possible answers using only the visible input geometry."""
    palette = set(int(x) for x in np.unique(grid))
    predictions = []
    for background in palette:
        for marker in palette - {background}:
            if int(np.count_nonzero(grid == marker)) != 4:
                continue
            colors = palette - {background, marker}
            if not 2 <= len(colors) <= 4:
                continue
            coordinates = np.argwhere(np.isin(grid, list(colors)))
            top, left = coordinates.min(axis=0)
            bottom, right = coordinates.max(axis=0)
            size = int(bottom - top + 1)
            if not (2 <= size <= 5 and right - left + 1 == size
                    and len(coordinates) == size * size):
                continue
            if not bool(np.isin(grid[top:top + size, left:left + size], list(colors)).all()):
                continue
            marker_cells = set(map(tuple, np.argwhere(grid == marker)))
            for source, destination in _COMBOS:
                source_cell, _ = _geometry(int(top), int(left), size, source)
                destination_cell, socket = _geometry(int(top), int(left), size, destination)
                if len(set(socket) | {source_cell}) != 4:
                    continue
                if marker_cells != set(socket) | {source_cell}:
                    continue
                if grid[destination_cell] != background:
                    continue
                output = grid.copy()
                for cell in marker_cells:
                    output[cell] = background
                output[top:top + size, left:left + size] = np.rot90(
                    grid[top:top + size, left:left + size],
                    -(destination - source) % 4,
                )
                output[destination_cell] = marker
                predictions.append(output)
    return predictions


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    counts_in = np.bincount(input_grid.ravel(), minlength=10)
    counts_out = np.bincount(output_grid.ravel(), minlength=10)
    delta = counts_out - counts_in
    backgrounds = np.flatnonzero(delta == 3)
    markers = np.flatnonzero(delta == -3)
    assert len(backgrounds) == len(markers) == 1
    assert int(np.count_nonzero(delta)) == 2
    background, marker = int(backgrounds[0]), int(markers[0])
    assert int(counts_in[marker]) == 4 and int(counts_out[marker]) == 1
    used = set(int(x) for x in np.unique(input_grid))
    assert used == set(int(x) for x in np.unique(output_grid))
    colors = used - {background, marker}
    assert 2 <= len(colors) <= 4
    object_mask = np.isin(input_grid, list(colors))
    assert np.array_equal(object_mask, np.isin(output_grid, list(colors)))
    rows, cols = np.where(object_mask)
    assert len(rows) >= 4
    top, left = int(rows.min()), int(cols.min())
    size = int(rows.max() - top + 1)
    assert 2 <= size <= 5 and int(cols.max() - left + 1) == size
    assert len(rows) == size * size and bool(object_mask[top:top + size, left:left + size].all())
    marker_positions = set(zip(*[x.tolist() for x in np.where(input_grid == marker)]))
    assert len(marker_positions) == 4
    candidates = []
    for source, destination in _COMBOS:
        source_cell, _ = _geometry(top, left, size, source)
        destination_cell, socket = _geometry(top, left, size, destination)
        if marker_positions == set(socket) | {source_cell} and len(set(socket) | {source_cell}) == 4:
            candidates.append((source, destination, destination_cell))
    assert len(candidates) == 1
    source, destination, destination_cell = candidates[0]
    assert input_grid[destination_cell] == background
    pattern = input_grid[top:top + size, left:left + size]
    turns = (destination - source) % 4
    rotated = np.rot90(pattern, -turns)
    assert not np.array_equal(pattern, rotated)
    expected = input_grid.copy()
    for cell in marker_positions:
        expected[cell] = background
    expected[top:top + size, left:left + size] = rotated
    expected[destination_cell] = marker
    assert np.array_equal(expected, output_grid)
    return {"source": source, "destination": destination, "turn": turns,
            "size": size, "palette": (background, marker, tuple(sorted(colors))),
            "shape": input_grid.shape, "top": top, "left": left,
            "pattern": tuple(int(x) for x in pattern.ravel())}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            key = _signature(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _source_string(pair["input"]) not in _SOURCE_INPUTS
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
            predictions = _input_predictions(pair["input"])
            assert len(predictions) == 1
            assert np.array_equal(predictions[0], pair["output"])
    training = analyses["train"]
    tests = analyses["test"]
    assert 3 in {a["size"] for a in training}
    assert len({a["size"] for a in training}) >= 2
    assert {a["size"] for a in tests} <= {a["size"] for a in training}
    train_combos = {(a["source"], a["destination"]) for a in training}
    assert len(train_combos) == len(training)
    if len(training) == 2:
        assert all((a["source"], a["destination"]) in train_combos for a in tests)
    else:
        assert {a["turn"] for a in training} == {1, 2, 3}
        assert any(a["source"] == b["source"] and a["destination"] != b["destination"]
                   for a in training for b in training)
        assert any(a["destination"] == b["destination"] and a["source"] != b["source"]
                   for a in training for b in training)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    # Marker/socket meaning and corner ordering are shared by all pairs.
    # Palette mode controls only whether literal colors recur within a puzzle.
    constant_palette = bool(rng.integers(0, 4) == 0)
    shared_palette = tuple(int(x) for x in rng.choice(10, size=int(rng.integers(4, 6)), replace=False))
    train_combos = _training_combos(rng, train_count)
    test_combos = _test_combos(rng, train_combos, test_count)
    alternative = int(rng.choice((2, 4, 5)))
    train_sizes = [3, alternative]
    train_sizes += [int(rng.integers(2, 6)) for _ in range(train_count - 2)]
    train_sizes = [train_sizes[int(i)] for i in rng.permutation(train_count)]
    test_sizes = [train_sizes[int(rng.integers(0, train_count))] for _ in range(test_count)]
    if test_count > 1 and len(set(test_sizes)) == 1:
        test_sizes[-1] = alternative if test_sizes[0] == 3 else 3
    puzzle = {"train": [], "test": []}
    seen = set()
    palettes = set()
    for split, combos, sizes in (("train", train_combos, train_sizes),
                                  ("test", test_combos, test_sizes)):
        for combo, size in zip(combos, sizes):
            for _ in range(200):
                if constant_palette:
                    palette = shared_palette
                else:
                    color_count = int(rng.integers(2, min(4, size * size) + 1))
                    palette = tuple(int(x) for x in rng.choice(10, size=color_count + 2, replace=False))
                    if palette in palettes:
                        continue
                child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                grid, output = _make_pair(child, "source_train" if split == "train" else "target_test",
                                          combo, size, palette)
                signature = _signature(grid)
                if signature in seen or _source_string(grid) in _SOURCE_INPUTS:
                    continue
                _analyze_pair(grid, output)
                seen.add(signature)
                palettes.add(palette)
                puzzle[split].append({"input": grid, "output": output})
                break
            else:
                raise RuntimeError("could not create a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    for seed, train, test in ((8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
                              (8004, None, 1), (8005, None, 4),
                              (8006, 2, None), (8007, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(9000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    a, b = generate_puzzle(123456, 7, 3), generate_puzzle(123456, 7, 3)
    assert all(np.array_equal(x[key], y[key]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for key in ("input", "output"))
    a, b = generate_puzzle(654321), generate_puzzle(654321)
    assert all(np.array_equal(x[key], y[key]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for key in ("input", "output"))
    assert _signature(generate_puzzle(11111)["train"][0]["input"]) != _signature(
        generate_puzzle(22222)["train"][0]["input"])
    train_counts, test_counts, sizes, turns, sources, destinations = (set() for _ in range(6))
    shapes, palettes, motifs, relations, combinations = (set() for _ in range(5))
    varied_palettes = 0
    varied_sizes = 0
    varied_test_relations = 0
    edge_touch = 0
    wide = tall = one_cell_accent = 0
    test_sources, test_destinations, test_regimes, color_counts = (set() for _ in range(4))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        analyses = [_analyze_pair(p["input"], p["output"])
                    for split in ("train", "test") for p in puzzle[split]]
        varied_palettes += len({a["palette"] for a in analyses}) > 1
        varied_sizes += len({a["size"] for a in analyses}) > 1
        training = analyses[:len(puzzle["train"])]
        testing = analyses[len(puzzle["train"]):]
        train_combos = {(a["source"], a["destination"]) for a in training}
        train_sources = {a["source"] for a in training}
        train_destinations = {a["destination"] for a in training}
        varied_test_relations += len({(a["source"], a["destination"]) for a in testing}) > 1
        for a in testing:
            test_sources.add(a["source"])
            test_destinations.add(a["destination"])
            combo = (a["source"], a["destination"])
            regime = ("repeated" if combo in train_combos else
                      "recombined" if a["source"] in train_sources and
                      a["destination"] in train_destinations else "heldout")
            test_regimes.add(regime)
        for a in analyses:
            sizes.add(a["size"])
            turns.add(a["turn"])
            sources.add(a["source"])
            destinations.add(a["destination"])
            shapes.add(a["shape"])
            palettes.add(a["palette"])
            motifs.add(a["pattern"])
            relations.add((a["source"], a["destination"]))
            combinations.add((a["size"], len(a["palette"][2]), a["turn"]))
            color_counts.add(len(a["palette"][2]))
            height, width = a["shape"]
            edge_touch += (a["top"] == 0 or a["left"] == 0 or
                           a["top"] + a["size"] == height or
                           a["left"] + a["size"] == width)
            wide += width >= 2 * height
            tall += height >= 2 * width
            unique, counts = np.unique(a["pattern"], return_counts=True)
            one_cell_accent += int(counts.min() == 1)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sizes == {2, 3, 4, 5} and turns == {1, 2, 3}
    assert sources == destinations == {0, 1, 2, 3} and len(relations) == 12
    assert len(shapes) >= 60 and len(palettes) >= 50 and len(motifs) >= 100
    assert varied_palettes >= 20 and varied_sizes == 42 and len(combinations) >= 25
    assert test_sources == test_destinations == {0, 1, 2, 3}
    assert test_regimes == {"repeated", "recombined", "heldout"}
    assert color_counts == {2, 3, 4}
    assert varied_test_relations >= 10 and edge_touch >= 20
    assert wide >= 20 and tall >= 20 and one_cell_accent >= 30
