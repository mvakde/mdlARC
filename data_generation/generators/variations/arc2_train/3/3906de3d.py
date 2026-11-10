import numpy as np


_SOURCE_EXACT = {
    (
        (
            "0011111110",
            "0011011110",
            "0011010110",
            "0011010110",
            "0000000110",
            "0000000000",
            "0000002000",
            "0000002000",
            "0000002000",
            "0000202000",
        ),
        (
            "0011111110",
            "0011211110",
            "0011012110",
            "0011012110",
            "0000002110",
            "0000002000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0011111000",
            "0011111000",
            "0011011000",
            "0011011000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000200000",
            "0000200000",
            "0000200000",
        ),
        (
            "0011111000",
            "0011111000",
            "0011211000",
            "0011211000",
            "0000200000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0111111111",
            "0110111111",
            "0110111101",
            "0110010101",
            "0000000000",
            "0000000000",
            "0000002000",
            "0002002020",
            "0002202020",
            "0002202020",
        ),
        (
            "0111111111",
            "0112111111",
            "0112111121",
            "0112212121",
            "0000202020",
            "0000002000",
            "0000002000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "0111111111",
            "0101111101",
            "0101010101",
            "0101000101",
            "0001000021",
            "0000000020",
            "0000020020",
            "0020020020",
            "0020220020",
            "0020222020",
        ),
        (
            "0111111111",
            "0121111121",
            "0121212121",
            "0121220121",
            "0001020021",
            "0000020020",
            "0000020020",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
}


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _rows_key(input_grid),
        _rows_key(output_grid),
    ) in _SOURCE_EXACT


def _is_four_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    seen = {tuple(int(v) for v in cells[0])}
    todo = list(seen)
    h, w = mask.shape
    while todo:
        y, x = todo.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            q = (y + dy, x + dx)
            if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and q not in seen:
                seen.add(q)
                todo.append(q)
    return len(seen) == int(mask.sum())


def _canonical_parse(grid: np.ndarray, bg: int, scaffold: int) -> dict | None:
    h, w = grid.shape
    mask = grid == scaffold
    if not _is_four_connected(mask):
        return None
    ys, xs = np.where(mask)
    roof = int(ys.min())
    left, right = int(xs.min()), int(xs.max())
    if right - left + 1 < 4 or not np.all(mask[roof, left:right + 1]):
        return None
    if np.any(mask[:, :left]) or np.any(mask[:, right + 1:]):
        return None
    movers = (grid != bg) & ~mask
    if not movers.any():
        return None
    lanes = []
    for x in range(w):
        rows = np.flatnonzero(movers[:, x])
        if not len(rows):
            continue
        if x < left or x > right:
            return None
        tip_rows = np.flatnonzero(mask[:, x])
        if not len(tip_rows):
            return None
        tip = int(tip_rows.max())
        start, length = int(rows[0]), len(rows)
        if not np.array_equal(rows, np.arange(start, start + length)):
            return None
        if start < tip + 1 or tip + length >= h:
            return None
        if not np.all(grid[tip + 1:start, x] == bg):
            return None
        colors = np.unique(grid[rows, x])
        if len(colors) != 1:
            return None
        lanes.append((x, tip, start, length, int(colors[0])))
    if not lanes:
        return None
    return {"bg": bg, "scaffold": scaffold, "roof": roof,
            "span": (left, right), "lanes": tuple(lanes),
            "scaffold_count": int(mask.sum()), "shape": (h, w),
            "overhang": any(not np.array_equal(np.flatnonzero(mask[:, x]),
                             np.arange(roof, np.flatnonzero(mask[:, x]).max() + 1))
                            for x in range(left, right + 1))}


def _settle_canonical(grid: np.ndarray, parsed: dict) -> np.ndarray:
    result = grid.copy()
    for x, tip, start, length, color in parsed["lanes"]:
        result[start:start + length, x] = parsed["bg"]
        result[tip + 1:tip + 1 + length, x] = color
    return result


def _infer(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid)]
    candidates = []
    for turn in range(4):
        canonical = np.rot90(grid, turn)
        for bg in colors:
            for scaffold in colors:
                if bg == scaffold:
                    continue
                parsed = _canonical_parse(canonical, bg, scaffold)
                if parsed is None:
                    continue
                parsed["turn"] = turn
                parsed["palette"] = (bg, scaffold)
                parsed["predicted"] = np.rot90(_settle_canonical(canonical, parsed), -turn)
                candidates.append(parsed)
    return candidates


def _size(rng: np.random.Generator) -> int:
    band = int(rng.integers(0, 3))
    return int(rng.integers(*((9, 12), (12, 20), (20, 31))[band]))


def _profile(rng: np.random.Generator, span: int, maximum: int) -> list[int]:
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return [int(rng.integers(0, maximum + 1))] * span
    if mode == 1:
        cuts = sorted(int(v) for v in rng.choice(np.arange(1, span),
                         size=min(span - 1, int(rng.integers(1, 4))), replace=False))
        values = [int(v) for v in rng.integers(0, maximum + 1, size=len(cuts) + 1)]
        return [values[sum(i >= cut for cut in cuts)] for i in range(span)]
    if mode == 2:
        base = int(rng.integers(0, maximum + 1))
        result = [base] * span
        width = int(rng.integers(1, min(span, 5) + 1))
        begin = int(rng.integers(0, span - width + 1))
        value = int(rng.integers(0, maximum + 1))
        result[begin:begin + width] = [value] * width
        return result
    if mode == 3:
        levels = sorted(int(v) for v in rng.integers(0, maximum + 1, size=span))
        return levels if rng.integers(0, 2) else levels[::-1]
    return [int(v) for v in rng.integers(0, maximum + 1, size=span)]


def _occupied(rng: np.random.Generator, span: int, proof: bool, regime: str,
              min_lanes: int) -> list[int]:
    limit = min(span, 10 if regime == "target_test" else 8)
    minimum = max(min_lanes, 3 if proof else 1)
    count = int(rng.integers(minimum, limit + 1))
    if proof or (count >= 2 and rng.random() < 0.45):
        run = min(count, int(rng.integers(2, min(5, count) + 1)))
        begin = int(rng.integers(0, span - run + 1))
        chosen = set(range(begin, begin + run))
        remaining = [x for x in range(span) if x not in chosen]
        if count > run:
            chosen.update(int(x) for x in rng.choice(remaining, size=count - run, replace=False))
        return sorted(chosen)
    return sorted(int(x) for x in rng.choice(span, size=count, replace=False))


def _build_pair(rng: np.random.Generator, direction: int, palette: tuple[int, ...],
                proof: bool, force_zero: bool, regime: str, multicolor: bool,
                min_lanes: int = 1) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        h, w = _size(rng), _size(rng)
        if w < min_lanes:
            continue
        span = int(rng.integers(max(4, min_lanes), w + 1))
        left = int(rng.integers(0, w - span + 1))
        roof = int(rng.integers(0, min(8, h // 3) + 1))
        free = h - roof - 1
        maximum = min(free - 3, max(2, h // 3 + 1))
        if maximum < 0:
            continue
        depths = _profile(rng, span, maximum)
        occupied = _occupied(rng, span, proof, regime, min_lanes)
        if multicolor and len(occupied) < 2:
            continue
        if proof:
            adjacent = [(a, b) for a, b in zip(occupied, occupied[1:]) if b == a + 1]
            if not adjacent:
                continue
            proof_lanes = adjacent[int(rng.integers(0, len(adjacent)))]
            a, b = proof_lanes
            max_tip_depth = min(maximum, free - 4)
            if max_tip_depth < 1:
                continue
            low = int(rng.integers(0, max_tip_depth))
            high = int(rng.integers(low + 1, max_tip_depth + 1))
            if rng.integers(0, 2):
                low, high = high, low
            depths[a], depths[b] = low, high
        else:
            proof_lanes = None
        lengths = {}
        for x in occupied:
            room = free - depths[x]
            upper = min(10, room - 1)
            if upper < 1:
                break
            lengths[x] = int(rng.integers(1, upper + 1))
        if len(lengths) != len(occupied):
            continue
        if len(occupied) >= 2 and rng.random() < 0.22:
            common = min(8, *(free - depths[x] - 1 for x in occupied[:2]))
            if common >= 2:
                value = int(rng.integers(2, common + 1))
                for x in occupied[:2]:
                    lengths[x] = value
        starts = {}
        arrangement = int(rng.integers(0, 3))
        for x in occupied:
            target = roof + depths[x] + 1
            last = h - lengths[x]
            if arrangement == 0:
                starts[x] = last
            elif arrangement == 1:
                starts[x] = int(rng.choice((target, target + 1, last)))
            else:
                starts[x] = int(rng.integers(target, last + 1))
        if proof_lanes is not None:
            a, b = proof_lanes
            earliest = roof + max(depths[a], depths[b]) + 2
            latest = min(h - lengths[a], h - lengths[b])
            if earliest > latest:
                continue
            same_start = int(rng.integers(earliest, latest + 1))
            starts[a] = starts[b] = same_start
        if force_zero:
            candidates = [x for x in occupied if proof_lanes is None or x not in proof_lanes]
            if not candidates:
                candidates = [proof_lanes[0]]
            x = int(rng.choice(candidates))
            starts[x] = roof + depths[x] + 1
        if not any(starts[x] > roof + depths[x] + 1 for x in occupied):
            continue
        bg, scaffold = palette[:2]
        mover_colors = palette[2:]
        grid = np.full((h, w), bg, dtype=np.int8)
        for offset, depth in enumerate(depths):
            grid[roof:roof + depth + 1, left + offset] = scaffold
        if rng.random() < 0.24:
            for _hole in range(int(rng.integers(1, 4))):
                x = int(rng.integers(left + 1, left + span - 1)) if span > 2 else left
                tip = roof + depths[x - left]
                if tip >= roof + 2:
                    y = int(rng.integers(roof + 1, tip))
                    old = grid[y, x]
                    grid[y, x] = bg
                    if not _is_four_connected(grid == scaffold):
                        grid[y, x] = old
        for i, x in enumerate(occupied):
            color = mover_colors[i % len(mover_colors)] if multicolor else mover_colors[0]
            start = starts[x]
            grid[start:start + lengths[x], left + x] = color
        candidates = _infer(np.rot90(grid, -direction))
        if len(candidates) != 1:
            continue
        input_grid = np.rot90(grid, -direction).copy()
        output_grid = candidates[0]["predicted"].copy()
        if np.array_equal(input_grid, output_grid) or _is_exact_source_pair(input_grid, output_grid):
            continue
        parsed = candidates[0]
        if proof and not _touching_unequal(parsed):
            continue
        return input_grid, output_grid
    raise RuntimeError("Unable to construct an unambiguous pair")


def _touching_unequal(parsed: dict) -> bool:
    lanes = parsed["lanes"]
    for (x1, tip1, start1, len1, _), (x2, tip2, start2, len2, _) in zip(lanes, lanes[1:]):
        if x2 == x1 + 1 and max(start1, start2) < min(start1 + len1, start2 + len2):
            if start1 - tip1 != start2 - tip2:
                return True
    return False


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        choices = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        a, b = choices[int(rng.integers(0, len(choices)))]
    elif num_train is None:
        b = int(num_test)
        a = int(rng.integers(max(2, b + 1), 11))
    elif num_test is None:
        a = int(num_train)
        b = int(rng.integers(1, min(4, a - 1) + 1))
    else:
        a, b = int(num_train), int(num_test)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _palette(rng: np.random.Generator, multicolor: bool) -> tuple[int, ...]:
    amount = int(rng.integers(2, 6)) if multicolor else 1
    return tuple(int(v) for v in rng.choice(10, size=amount + 2, replace=False))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    mixed = bool(rng.random() < 0.23)
    local_palette = bool(rng.random() < 0.42)
    multicolor = bool(rng.random() < 0.25)
    if mixed:
        directions = [int(x) for x in rng.choice(4, size=2, replace=False)]
        train_directions = [directions[i % 2] for i in range(train_count)]
        rng.shuffle(train_directions)
    else:
        direction = 0 if rng.random() < 0.54 else int(rng.integers(1, 4))
        train_directions = [direction] * train_count
    test_directions = [int(rng.choice(train_directions)) for _ in range(test_count)]
    fixed_palette = _palette(rng, multicolor)
    train_palettes = [_palette(rng, multicolor) if local_palette else fixed_palette
                      for _ in range(train_count)]
    if local_palette and train_palettes[0][:2] == train_palettes[1][:2]:
        train_palettes[1] = _palette(rng, multicolor)
        if train_palettes[0][:2] == train_palettes[1][:2]:
            p = list(train_palettes[1])
            p[0], p[1] = p[1], p[0]
            train_palettes[1] = tuple(p)
    test_palettes = [_palette(rng, multicolor) if local_palette else fixed_palette
                     for _ in range(test_count)]
    proof_index = int(rng.integers(0, train_count))
    zero_index = int(rng.integers(0, train_count))
    harder_test = bool(rng.random() < 0.5)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count, directions, palettes in (("train", train_count, train_directions, train_palettes),
                                                  ("test", test_count, test_directions, test_palettes)):
        for index in range(count):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                pair = _build_pair(pair_rng, directions[index], palettes[index],
                                   section == "train" and index == proof_index,
                                   section == "train" and index == zero_index or
                                   section == "test" and rng.random() < 0.27,
                                   "source_train" if section == "train" else "target_test",
                                   multicolor and (section == "test" or index == 0),
                                   min(10, max(len(_infer(p["input"])[0]["lanes"])
                                               for p in puzzle["train"]) + 1)
                                   if section == "test" and index == 0 and harder_test else 1)
                key = _grid_key(pair[0])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append({"input": pair[0], "output": pair[1]})
                    break
            else:
                raise RuntimeError("Unable to avoid duplicate pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            key, pair_key = _grid_key(a), (_grid_key(a), _grid_key(b))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _is_exact_source_pair(a, b)
            candidates = _infer(a)
            assert len(candidates) == 1
            parsed = candidates[0]
            assert np.array_equal(parsed["predicted"], b)
            for x, tip, start, length, color in parsed["lanes"]:
                assert start >= tip + 1 and tip + length < parsed["shape"][0]
                assert color not in parsed["palette"]
                assert np.count_nonzero(np.rot90(a, parsed["turn"])[:, x] == color) >= length
            analyses[section].append(parsed)
    train = analyses["train"]
    test = analyses["test"]
    assert any(_touching_unequal(p) for p in train)
    assert any(any(start == tip + 1 for _, tip, start, _, _ in p["lanes"]) for p in train)
    train_dirs = {p["turn"] for p in train}
    assert all(p["turn"] in train_dirs for p in test)
    train_palettes = {p["palette"] for p in train}
    if len(train_palettes) == 1:
        assert all(p["palette"] in train_palettes for p in test)
    if any(len({lane[4] for lane in p["lanes"]}) > 1 for p in test):
        assert any(len({lane[4] for lane in p["lanes"]}) > 1 for p in train)


def _equal_puzzles(a: dict, b: dict) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, directions, palettes = set(), set(), set(), set(), set()
    lane_counts, lengths, shift_counts, overhangs, signatures = set(), set(), set(), set(), set()
    local_count = mixed_count = baseline_count = zero_count = 0
    harder_lanes = comparable_lanes = lower_density = higher_density = 0
    flat_faces = inset_scaffolds = side_margins = repeated_long = 0
    test_one_lane = test_zero = multi_color_train = 0
    first_train_lanes, test_lanes = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        parsed = {s: [_infer(p["input"])[0] for p in puzzle[s]] for s in ("train", "test")}
        local_count += len({p["palette"] for p in parsed["train"]}) > 1
        mixed_count += len({p["turn"] for p in parsed["train"]}) > 1
        first_train_lanes.add(len(parsed["train"][0]["lanes"]))
        harder_lanes += max(len(x["lanes"]) for x in parsed["test"]) > max(len(x["lanes"]) for x in parsed["train"])
        comparable_lanes += min(len(x["lanes"]) for x in parsed["test"]) <= max(len(x["lanes"]) for x in parsed["train"])
        lower_density += min(x["scaffold_count"] for x in parsed["test"]) <= max(x["scaffold_count"] for x in parsed["train"])
        higher_density += max(x["scaffold_count"] for x in parsed["test"]) > max(x["scaffold_count"] for x in parsed["train"])
        multi_color_train += any(len({lane[4] for lane in p["lanes"]}) > 1 for p in parsed["train"])
        for s in ("train", "test"):
            for pair, p in zip(puzzle[s], parsed[s]):
                shapes.add(pair["input"].shape)
                directions.add(p["turn"])
                palettes.add(p["palette"])
                lane_counts.add(len(p["lanes"]))
                lengths.update(lane[3] for lane in p["lanes"])
                shift_counts.add(tuple(start - tip - 1 for _, tip, start, _, _ in p["lanes"]))
                overhangs.add(p["overhang"])
                h = p["shape"][0]
                baseline_count += len(p["lanes"]) > 1 and all(start + length == h for _, _, start, length, _ in p["lanes"])
                zero_count += any(start == tip + 1 for _, tip, start, _, _ in p["lanes"])
                flat_faces += len({tip for _, tip, _, _, _ in p["lanes"]}) == 1
                inset_scaffolds += p["roof"] > 0
                side_margins += p["span"][0] > 0 and p["span"][1] < p["shape"][1] - 1
                repeated_long += sum(length >= 5 for _, _, _, length, _ in p["lanes"]) >= 2
                if s == "test":
                    test_lanes.add(len(p["lanes"]))
                    test_one_lane += len(p["lanes"]) == 1
                    test_zero += any(start == tip + 1 for _, tip, start, _, _ in p["lanes"])
                signatures.add((_grid_key(pair["input"]), _grid_key(pair["output"])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 60 and len(directions) == 4 and len(palettes) >= 25
    assert min(lane_counts) == 1 and max(lane_counts) >= 7
    assert min(lengths) == 1 and max(lengths) >= 7
    assert 5 in lengths and len(test_lanes) >= 7 and test_one_lane >= 1
    assert len(first_train_lanes) >= 4
    assert len(shift_counts) >= 100 and len(signatures) >= 150
    assert local_count >= 8 and mixed_count >= 3
    assert baseline_count >= 10 and zero_count >= 20 and overhangs == {False, True}
    assert harder_lanes >= 8 and comparable_lanes >= 15
    assert lower_density >= 15 and higher_density >= 1
    assert flat_faces >= 40 and inset_scaffolds >= 50 and side_margins >= 30
    assert repeated_long >= 30 and test_zero >= 10 and multi_color_train >= 5
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(1000 + i, a, b)
        assert (len(p["train"]), len(p["test"])) == (a, b)
    for i, (a, b) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1100 + i, a, b)
        assert (a is None or len(p["train"]) == a) and (b is None or len(p["test"]) == b)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal count request: {args}")
    assert _equal_puzzles(generate_puzzle(1300, 7, 3), generate_puzzle(1300, 7, 3))
    assert not _equal_puzzles(generate_puzzle(1300, 7, 3), generate_puzzle(1301, 7, 3))
