"""Generate ARC puzzles that complete line segments into tapering diamonds."""

import numpy as np


SOURCE_OBJECTS = (
    ((19, 21), (("V", 2, 4, 5), ("H", 12, 9, 9))),
    ((17, 19), (("H", 7, 2, 14),)),
    ((21, 22), (("H", 4, 2, 6), ("H", 9, 11, 10), ("V", 11, 5, 7))),
)
SOURCE_TYPES = (("H", 1), ("V", 1), ("H", 0))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _line_mask(shape, obj):
    direction, row, col, length = obj
    mask = np.zeros(shape, dtype=bool)
    if direction == "H":
        mask[row, col:col + length] = True
    else:
        mask[row:row + length, col] = True
    return mask


def _diamond_mask(shape, obj):
    direction, row, col, length = obj
    mask = _line_mask(shape, obj)
    for distance in range(1, (length - 1) // 2 + 1):
        if direction == "H":
            mask[row - distance, col + distance:col + length - distance] = True
            mask[row + distance, col + distance:col + length - distance] = True
        else:
            mask[row + distance:row + length - distance, col - distance] = True
            mask[row + distance:row + length - distance, col + distance] = True
    return mask


def _fits(shape, obj):
    height, width = shape
    direction, row, col, length = obj
    radius = (length - 1) // 2
    if direction == "H":
        return row - radius >= 0 and row + radius < height and col >= 0 and col + length <= width
    return row >= 0 and row + length <= height and col - radius >= 0 and col + radius < width


def _touch4(mask):
    expanded = mask.copy()
    expanded[1:] |= mask[:-1]
    expanded[:-1] |= mask[1:]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    return expanded


def _positions(shape, direction, length):
    height, width = shape
    radius = (length - 1) // 2
    if direction == "H":
        return [(r, c) for r in range(radius, height - radius) for c in range(width - length + 1)]
    return [(r, c) for r in range(height - length + 1) for c in range(radius, width - radius)]


def _on_edge(shape, mask):
    return bool(np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]))


def _choose_lengths(rng, shape, specs, long_hint):
    lengths = []
    for direction, parity in specs:
        possible = [length for length in range(3, 17) if length % 2 == parity and _positions(shape, direction, length)]
        if not possible:
            return None
        if long_hint and direction == "H" and parity == 0 and 14 in possible:
            length = 14
            long_hint = False
        else:
            if len(specs) >= 3 and rng.random() < 0.8:
                possible = [length for length in possible if length <= 9] or possible
            elif len(specs) == 2 and rng.random() < 0.55:
                possible = [length for length in possible if length <= 11] or possible
            length = int(rng.choice(possible))
        lengths.append((direction, length))
    return lengths


def _place(rng, shape, lengths, edge_hint):
    occupied = np.zeros(shape, dtype=bool)
    seeds = np.zeros(shape, dtype=bool)
    objects = []
    for direction, length in sorted(lengths, key=lambda item: -item[1]):
        positions = _positions(shape, direction, length)
        rng.shuffle(positions)
        if edge_hint and not objects:
            edge_positions = [(r, c) for r, c in positions if _on_edge(shape, _diamond_mask(shape, (direction, r, c, length)))]
            if edge_positions:
                positions = edge_positions + positions
        found = False
        for row, col in positions:
            obj = (direction, row, col, length)
            diamond = _diamond_mask(shape, obj)
            line = _line_mask(shape, obj)
            if np.any(diamond & occupied) or np.any(line & _touch4(seeds)):
                continue
            objects.append(obj)
            occupied |= diamond
            seeds |= line
            found = True
            break
        if not found:
            return None
    return objects


def _sample_objects(rng, regime, specs, long_hint=False):
    for attempt in range(70):
        if long_hint and attempt < 25:
            shape = (int(rng.integers(14, 29)), int(rng.integers(17, 31)))
        else:
            low = 9 if len(specs) <= 2 else 16
            shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        lengths = _choose_lengths(rng, shape, specs, long_hint)
        if lengths is None:
            continue
        edge_hint = rng.random() < (0.35 if regime == "target_test" else 0.22)
        objects = _place(rng, shape, lengths, edge_hint)
        if objects is not None:
            return shape, objects
    shape = (30, 30)
    for _ in range(20):
        lengths = [(direction, 3 if parity else 4) for direction, parity in specs]
        objects = _place(rng, shape, lengths, False)
        if objects is not None:
            return shape, objects
    raise RuntimeError("could not place isolated line segments")


def _render(shape, objects, background, seed_colors, completion):
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for obj, seed_color in zip(objects, seed_colors):
        line = _line_mask(shape, obj)
        diamond = _diamond_mask(shape, obj)
        inp[line] = seed_color
        out[line] = seed_color
        out[diamond & ~line] = completion
    return {"input": inp, "output": out}


def _colors(rng, mode):
    colors = [int(c) for c in rng.choice(10, size=3 if mode == "fixed" else 5, replace=False)]
    if mode == "fixed":
        return colors[0], (colors[1],), (colors[2],)
    return colors[0], (colors[1], colors[2]), (colors[3], colors[4])


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = ("fixed", "vary_pairs", "multi_line")[int(rng.integers(0, 3))]
    completion, backgrounds, seed_palette = _colors(rng, mode)
    types = list(SOURCE_TYPES)
    if rng.random() < 0.55:
        types.append(("V", 0))
    rng.shuffle(types)
    first, second = types[:2], types[2:]
    if len(second) == 1 and rng.random() < 0.5:
        first, second = second, first
    train_specs = [first, second]
    for _ in range(num_train - 2):
        count = int(rng.integers(1, 5))
        train_specs.append([types[int(rng.integers(len(types)))] for _ in range(count)])
    rng.shuffle(train_specs)
    if mode == "multi_line":
        train_specs.sort(key=lambda specs: len(specs) < 2)
    test_specs = []
    for _ in range(num_test):
        count = int(rng.integers(1, 5))
        test_specs.append([types[int(rng.integers(len(types)))] for _ in range(count)])
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, all_specs in (("train", train_specs), ("test", test_specs)):
        for index, specs in enumerate(all_specs):
            for _ in range(12):
                long_hint = split == "train" and len(specs) == 1 and specs[0] == ("H", 0) and rng.random() < 0.48
                shape, objects = _sample_objects(rng, "source_train" if split == "train" else "target_test", specs, long_hint)
                if mode == "fixed":
                    background = backgrounds[0]
                    seed_colors = [seed_palette[0]] * len(objects)
                elif mode == "vary_pairs":
                    choice = index % 2 if split == "train" and index < 2 else int(rng.integers(0, 2))
                    background = backgrounds[choice]
                    seed_colors = [seed_palette[choice]] * len(objects)
                else:
                    background = backgrounds[index % 2] if split == "train" and index < 2 else backgrounds[int(rng.integers(0, 2))]
                    seed_colors = [seed_palette[j % 2] for j in range(len(objects))] if split == "train" and index == 0 else [seed_palette[int(rng.integers(0, 2))] for _ in objects]
                pair = _render(shape, objects, background, seed_colors, completion)
                key = _key(pair["input"])
                if key not in seen and not _is_source_pair(pair):
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, background):
    mask = grid != background
    visited = np.zeros(grid.shape, dtype=bool)
    objects = []
    for sr, sc in zip(*np.nonzero(mask)):
        sr, sc = int(sr), int(sc)
        if visited[sr, sc]:
            continue
        stack = [(sr, sc)]
        visited[sr, sc] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and mask[rr, cc] and not visited[rr, cc]:
                    visited[rr, cc] = True
                    stack.append((rr, cc))
        colors = {int(grid[r, c]) for r, c in cells}
        assert len(colors) == 1
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) == 1:
            obj = ("H", min(rows), min(cols), len(cells))
        else:
            assert len(cols) == 1
            obj = ("V", min(rows), min(cols), len(cells))
        line = _line_mask(grid.shape, obj)
        assert len(cells) >= 3 and all(line[r, c] for r, c in cells)
        assert np.count_nonzero(line) == len(cells)
        objects.append((obj, colors.pop()))
    assert objects
    return objects


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert counts[np.argmax(counts)] > inp.size // 2
    components = _components(inp, background)
    completion_colors = set(int(x) for x in np.unique(out)) - set(int(x) for x in values)
    assert len(completion_colors) == 1
    completion = completion_colors.pop()
    expected = inp.copy()
    occupied = np.zeros(inp.shape, dtype=bool)
    seed_mask = inp != background
    touching = False
    for obj, _ in components:
        assert _fits(inp.shape, obj)
        diamond = _diamond_mask(inp.shape, obj)
        line = _line_mask(inp.shape, obj)
        assert not np.any(diamond & occupied)
        assert not np.any(diamond & seed_mask & ~line)
        touching |= bool(np.any(diamond & _touch4(occupied)))
        occupied |= diamond
        expected[diamond & ~line] = completion
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {"background": background, "seeds": {color for _, color in components}, "completion": completion,
            "components": components, "edge": _on_edge(inp.shape, occupied), "touching": touching}


def _is_source_pair(pair):
    for shape, raw_objects in SOURCE_OBJECTS:
        if pair["input"].shape != shape:
            continue
        source = _render(shape, raw_objects, 0, [2] * len(raw_objects), 8)
        if np.array_equal(pair["input"], source["input"]) and np.array_equal(pair["output"], source["output"]):
            return True
    return False


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair)
            records[split].append(_analyze(pair))
    completions = {record["completion"] for split in records.values() for record in split}
    assert len(completions) == 1
    completion = completions.pop()
    assert all(completion != rec["background"] and completion not in rec["seeds"] for split in records.values() for rec in split)
    train_types = {(obj[0], obj[3] % 2) for rec in records["train"] for obj, _ in rec["components"]}
    test_types = {(obj[0], obj[3] % 2) for rec in records["test"] for obj, _ in rec["components"]}
    assert set(SOURCE_TYPES) <= train_types and test_types <= train_types
    train_backgrounds = {rec["background"] for rec in records["train"]}
    train_seeds = set().union(*(rec["seeds"] for rec in records["train"]))
    assert {rec["background"] for rec in records["test"]} <= train_backgrounds
    assert set().union(*(rec["seeds"] for rec in records["test"])) <= train_seeds
    assert any(len(rec["components"]) >= 2 for rec in records["train"])


def _puzzle_key(puzzle):
    return tuple((_key(pair["input"]), _key(pair["output"])) for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    a = generate_puzzle(777, 6, 3)
    b = generate_puzzle(777, 6, 3)
    assert _puzzle_key(a) == _puzzle_key(b)
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(900 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        validate_generated_puzzle(puzzle)
    for index, kwargs in enumerate(({"num_test": 1}, {"num_test": 4}, {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(950 + index, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(1001, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    train_counts, test_counts, fingerprints, shapes, role_triples = set(), set(), set(), set(), set()
    types, lengths, object_counts = set(), set(), set()
    mixed_colors = varied_backgrounds = edge_pairs = smaller_tests = touching_pairs = 0
    parallel_tests = odd_dominated_tests = vertical_dominated_tests = no_count_increase = 0
    varied_train_counts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_key(puzzle))
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        test_shapes = [pair["input"].shape for pair in puzzle["test"]]
        smaller_tests += any(th <= max(h for h, _ in train_shapes) or tw <= max(w for _, w in train_shapes) for th, tw in test_shapes)
        recs = [_analyze(pair) for split in ("train", "test") for pair in puzzle[split]]
        train_object_counts = [len(rec["components"]) for rec in recs[:len(puzzle["train"])]]
        test_object_counts = [len(rec["components"]) for rec in recs[len(puzzle["train"]):]]
        varied_train_counts += len(set(train_object_counts)) > 1
        no_count_increase += max(test_object_counts) <= max(train_object_counts)
        varied_backgrounds += len({rec["background"] for rec in recs}) > 1
        for pair_index, (pair, rec) in enumerate(zip(puzzle["train"] + puzzle["test"], recs)):
            shapes.add(pair["input"].shape)
            role_triples.add((rec["background"], tuple(sorted(rec["seeds"])), rec["completion"]))
            object_counts.add(len(rec["components"]))
            mixed_colors += len(rec["seeds"]) > 1
            edge_pairs += rec["edge"]
            touching_pairs += rec["touching"]
            if pair_index >= len(puzzle["train"]):
                lines = [obj for obj, _ in rec["components"]]
                parallel_tests += len(lines) >= 2 and len({obj[0] for obj in lines}) == 1
                odd_dominated_tests += sum(obj[3] % 2 for obj in lines) > len(lines) / 2
                vertical_dominated_tests += sum(obj[0] == "V" for obj in lines) > len(lines) / 2
            for obj, _ in rec["components"]:
                types.add((obj[0], obj[3] % 2))
                lengths.add(obj[3])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(role_triples) >= 25
    assert set(SOURCE_TYPES) <= types and ("V", 0) in types
    assert {1, 2, 3, 4} <= object_counts and {3, 4, 9, 14} <= lengths
    assert mixed_colors >= 4 and varied_backgrounds >= 5 and edge_pairs >= 10 and smaller_tests >= 15
    assert touching_pairs >= 8 and varied_train_counts >= 20 and no_count_increase >= 15
    assert parallel_tests >= 8 and odd_dominated_tests >= 15 and vertical_dominated_tests >= 8
