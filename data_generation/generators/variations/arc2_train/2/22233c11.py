"""Generate the opposite-diagonal completion of touching square pairs."""

import numpy as np


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("Illegal pair count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, 9 if num_train is None else num_train - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _cells(row, col, side, height, width):
    return {(r, c) for r in range(max(0, row), min(height, row + side))
            for c in range(max(0, col), min(width, col + side))}


def _completion_origins(row, col, direction, side):
    return ((row - side, col + 2 * direction * side),
            (row + 2 * side, col - direction * side))


def _blocks(pair, height, width):
    row, col, direction, side, _ = pair
    source = [_cells(row, col, side, height, width),
              _cells(row + side, col + direction * side, side, height, width)]
    added = [_cells(r, c, side, height, width)
             for r, c in _completion_origins(row, col, direction, side)]
    return source, added


def _layout_ok(height, width, pairs):
    if not pairs or not 1 <= height <= 30 or not 1 <= width <= 30:
        return False
    sources, additions, footprints, origins = [], [], [], []
    for row, col, direction, side, color in pairs:
        if direction not in (-1, 1) or not 1 <= side <= 5 or not 0 <= color <= 9:
            return False
        if row < 0 or row + 2 * side > height or min(col, col + direction * side) < 0:
            return False
        if max(col, col + direction * side) + side > width:
            return False
        source, added = _blocks((row, col, direction, side, color), height, width)
        if sum(map(len, added)) < side:
            return False
        sources.extend(source)
        additions.extend(added)
        footprints.append(set().union(*source, *added))
        origins.extend(((row, col, side, color),
                        (row + side, col + direction * side, side, color)))
    for i, first in enumerate(sources):
        for second in sources[i + 1:]:
            if any((r + dr, c + dc) in second for r, c in first
                   for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))):
                return False
    for i, (r, c, side, color) in enumerate(origins):
        neighbors = [j for j, (rr, cc, ss, co) in enumerate(origins)
                     if i != j and side == ss and color == co
                     and abs(r - rr) == side and abs(c - cc) == side]
        if neighbors != [i + 1 if i % 2 == 0 else i - 1]:
            return False
    all_sources = set().union(*sources)
    all_added = set().union(*additions)
    if all_sources & all_added or len(all_added) != sum(map(len, additions)):
        return False
    for i, first in enumerate(footprints):
        for second in footprints[i + 1:]:
            if any((r + dr, c + dc) in second for r, c in first
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                return False
    return True


def _render(height, width, pairs, background, completion):
    assert _layout_ok(height, width, pairs)
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for pair in pairs:
        source, added = _blocks(pair, height, width)
        for block in source:
            for r, c in block:
                inp[r, c] = out[r, c] = pair[4]
        for block in added:
            for r, c in block:
                out[r, c] = completion
    return inp, out


def _clip_kind(pair, height, width):
    _, added = _blocks(pair, height, width)
    counts = [len(block) for block in added]
    if 0 in counts:
        return "invisible"
    if any(count < pair[3] ** 2 for count in counts):
        return "partial"
    return "full"


def _dimension(rng, minimum):
    if minimum > 30:
        return None
    band = int(rng.integers(0, 3))
    lo, hi = ((minimum, min(30, max(11, minimum + 2))) if band == 0 else
              (max(minimum, 10), 22) if band == 1 else
              (max(minimum, 22), 30))
    if lo > hi:
        lo, hi = minimum, 30
    return int(rng.integers(lo, hi + 1))


def _sample_spec(rng, regime, index, directions, train_sides, colors):
    if regime == "train" and index < 2:
        side = int(rng.integers(1, 3)) if index == 0 else int(rng.integers(2, 4))
        return [(side, directions[index], colors[index % len(colors)],
                 "full" if index == 0 else "partial")]
    count = int(rng.choice([1, 2, 3], p=([.39, .48, .13] if regime == "train"
                                              else [.42, .48, .10])))
    if count == 3:
        sides = [int(rng.choice([1, 2])) for _ in range(3)]
    elif count == 2:
        sides = ([2, 3] if rng.random() < .32 else
                 [int(rng.choice([1, 2, 3])) for _ in range(2)])
        rng.shuffle(sides)
    elif regime == "test" and rng.random() < .38:
        sides = [int(rng.choice(train_sides))]
    else:
        sides = [int(rng.choice([1, 2, 3, 4, 5],
                                p=([.20, .28, .32, .16, .04] if regime == "train"
                                   else [.12, .23, .30, .27, .08])))]
    result = []
    for side in sides:
        mode = str(rng.choice(["full", "partial", "invisible"],
                              p=([.56, .41, .03] if regime == "train"
                                 else [.37, .58, .05])))
        if side == 1 and mode == "partial":
            mode = "full"
        result.append((side, int(rng.choice([-1, 1])),
                       int(rng.choice(colors)), mode))
    return result


def _make_grid(rng, specs, background, completion):
    count = len(specs)
    max_side = max(spec[0] for spec in specs)
    for attempt in range(90):
        if count == 1:
            minimum = 4 * max_side if specs[0][3] == "full" else 2 * max_side + 1
        else:
            minimum = max(13, 6 * max_side + (4 if count == 3 else 1))
        height, width = _dimension(rng, minimum), _dimension(rng, minimum)
        if height is None or width is None:
            continue
        pairs = []
        for side, direction, source, mode in specs:
            candidates = [(row, col, direction, side, source)
                          for row in range(height - 2 * side + 1)
                          for col in range(0 if direction == 1 else side,
                                           (width - 2 * side if direction == 1 else width - side) + 1)]
            rng.shuffle(candidates)
            for candidate in candidates:
                if _clip_kind(candidate, height, width) == mode and _layout_ok(height, width, pairs + [candidate]):
                    pairs.append(candidate)
                    break
            else:
                break
        if len(pairs) == count:
            return _render(height, width, pairs, background, completion)
    # Bounded fallback: retain sizes, directions, and colors while relaxing clipping.
    for height, width in ((30, 30), (26, 30), (30, 26)):
        pairs = []
        for side, direction, source, _ in specs:
            candidates = [(row, col, direction, side, source)
                          for row in range(height - 2 * side + 1)
                          for col in range(0 if direction == 1 else side,
                                           (width - 2 * side if direction == 1 else width - side) + 1)]
            rng.shuffle(candidates)
            for candidate in candidates:
                if _layout_ok(height, width, pairs + [candidate]):
                    pairs.append(candidate)
                    break
            else:
                break
        if len(pairs) == count:
            return _render(height, width, pairs, background, completion)
    raise RuntimeError("Could not place square pairs")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    background, completion = map(int, rng.choice(10, size=2, replace=False))
    palette = [color for color in range(10) if color not in (background, completion)]
    rng.shuffle(palette)
    colors = palette[:int(rng.choice([1, 2], p=[.58, .42]))]
    directions = [-1, 1]
    rng.shuffle(directions)
    puzzle = {"train": [], "test": []}
    seen, train_sides = set(), []
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(30):
                specs = _sample_spec(rng, regime, index, directions, train_sides, colors)
                try:
                    inp, out = _make_grid(rng, specs, background, completion)
                except RuntimeError:
                    continue
                if _grid_key(inp) not in seen:
                    seen.add(_grid_key(inp))
                    puzzle[regime].append({"input": inp, "output": out})
                    if regime == "train":
                        train_sides.extend(spec[0] for spec in specs)
                    break
            else:
                raise RuntimeError("Could not generate distinct example")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    result = []
    while remaining:
        start = remaining.pop()
        component, stack = {start}, [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row + 1, col), (row - 1, col),
                             (row, col + 1), (row, col - 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _parse_input(inp, background):
    blocks = []
    for color in set(map(int, np.unique(inp))) - {background}:
        for component in _components(inp, color):
            row, col = min(r for r, _ in component), min(c for _, c in component)
            side = max(r for r, _ in component) - row + 1
            assert 1 <= side <= 5
            assert component == _cells(row, col, side, *inp.shape)
            blocks.append((row, col, side, color))
    assert len(blocks) >= 2 and len(blocks) % 2 == 0
    pairs, used = [], set()
    for i, (row, col, side, color) in enumerate(blocks):
        matches = [j for j, (rr, cc, ss, co) in enumerate(blocks)
                   if j != i and side == ss and color == co
                   and abs(row - rr) == side and abs(col - cc) == side]
        assert len(matches) == 1
        j = matches[0]
        if i in used:
            assert j in used
            continue
        assert j not in used
        rr, cc, _, _ = blocks[j]
        upper_row, upper_col = (row, col) if row < rr else (rr, cc)
        lower_col = cc if row < rr else col
        pairs.append((upper_row, upper_col, 1 if lower_col > upper_col else -1,
                      side, color))
        used.update((i, j))
    assert len(used) == len(blocks) and _layout_ok(*inp.shape, pairs)
    return pairs


_ORIGINAL_INPUTS = (
    ("0000000000", "0000000000", "0000000000", "0003300000", "0003300000",
     "0330000000", "0330000000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0000000000", "0030000000", "0003000000",
     "0000000000", "0000000300", "0000003000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0000000000", "0003000000", "0000300000",
     "0000000000", "0000000000", "0000000000", "0000000000", "0000000000"),
    ("0000000000", "0000000000", "0000003330", "0000003330", "0000003330",
     "0003330000", "0003330000", "0003330000", "0000000000", "0000000000"),
)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs, records = set(), set(), []
    for regime, group in (("train", train), ("test", test)):
        for pair in group:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_key, output_key = _grid_key(inp), _grid_key(out)
            assert input_key not in seen_inputs and (input_key, output_key) not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            if inp.shape == (10, 10):
                assert tuple("".join(map(str, row)) for row in inp) not in _ORIGINAL_INPUTS
            values, counts = np.unique(inp, return_counts=True)
            background = int(values[int(np.argmax(counts))])
            assert counts.max() > inp.size // 2
            pairs = _parse_input(inp, background)
            changed = inp != out
            assert np.any(changed) and np.all(inp[changed] == background)
            additions = set(map(int, np.unique(out[changed])))
            assert len(additions) == 1
            completion = additions.pop()
            assert completion not in set(map(int, values))
            expected_input, expected_output = _render(*inp.shape, pairs, background, completion)
            assert np.array_equal(inp, expected_input) and np.array_equal(out, expected_output)
            records.append((regime, background, completion, pairs, inp.shape))
    assert len({r[1] for r in records}) == len({r[2] for r in records}) == 1
    train_colors = {p[4] for regime, _, _, pairs, _ in records
                    if regime == "train" for p in pairs}
    test_colors = {p[4] for regime, _, _, pairs, _ in records
                   if regime == "test" for p in pairs}
    assert test_colors <= train_colors
    assert {p[2] for regime, _, _, pairs, _ in records
            if regime == "train" for p in pairs} == {-1, 1}


def _same_puzzle(first, second):
    return all(len(first[regime]) == len(second[regime]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[regime], second[regime])
                   for key in ("input", "output"))
               for regime in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        for regime, group in puzzle.items():
            for item in group:
                inp, out = item["input"], item["output"]
                values, counts = np.unique(inp, return_counts=True)
                pairs = _parse_input(inp, int(values[np.argmax(counts)]))
                changed = inp != out
                for pair in pairs:
                    source, added = _blocks(pair, *inp.shape)
                    assert all(inp[r, c] == out[r, c] == pair[4]
                               for block in source for r, c in block)
                    assert all(changed[r, c] for block in added for r, c in block)
                records.append((regime, inp.shape, pairs,
                                tuple(_clip_kind(pair, *inp.shape) for pair in pairs)))
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({shape for _, shape, _, _ in records}) >= 35
    assert {p[2] for _, _, pairs, _ in records for p in pairs} == {-1, 1}
    assert {p[3] for _, _, pairs, _ in records for p in pairs} == {1, 2, 3, 4, 5}
    assert any(len(pairs) >= 2 and min(p[3] for p in pairs) >= 2
               for _, _, pairs, _ in records)
    assert any(len({p[3] for p in pairs}) > 1 for _, _, pairs, _ in records)
    assert {kind for _, _, _, kinds in records for kind in kinds} >= {"full", "partial"}
    assert any(regime == "test" and len(pairs) >= 2 for regime, _, pairs, _ in records)
    for regime in ("train", "test"):
        group = [(shape, pairs, kinds) for split, shape, pairs, kinds in records
                 if split == regime]
        assert any(len(pairs) >= 2 and min(p[3] for p in pairs) >= 2
                   for _, pairs, _ in group)
        assert any(len({p[3] for p in pairs}) > 1 for _, pairs, _ in group)
        assert any(len({p[4] for p in pairs}) > 1 for _, pairs, _ in group)
        assert any(any(p[3] >= 3 and kind == "full"
                       for p, kind in zip(pairs, kinds))
                   for _, pairs, kinds in group)
        assert any("invisible" in kinds for _, _, kinds in group)
        assert any(min(shape) <= 9 for shape, _, _ in group)
        assert any(max(shape) >= 28 for shape, _, _ in group)
        assert any(h > 1.4 * w for (h, w), _, _ in group)
        assert any(w > 1.4 * h for (h, w), _, _ in group)
        clipped_edges = set()
        for (height, width), pairs, _ in group:
            for row, col, direction, side, _ in pairs:
                for rr, cc in _completion_origins(row, col, direction, side):
                    if rr < 0:
                        clipped_edges.add("top")
                    if rr + side > height:
                        clipped_edges.add("bottom")
                    if cc < 0:
                        clipped_edges.add("left")
                    if cc + side > width:
                        clipped_edges.add("right")
        assert clipped_edges == {"top", "bottom", "left", "right"}
    def scales(group):
        result = set()
        for example in group:
            inp = example["input"]
            colors, counts = np.unique(inp, return_counts=True)
            result.update(p[3] for p in _parse_input(inp, int(colors[np.argmax(counts)])))
        return result

    reused_scale, novel_scale = 0, 0
    for puzzle in puzzles:
        train_scales, test_scales = scales(puzzle["train"]), scales(puzzle["test"])
        reused_scale += bool(test_scales & train_scales)
        novel_scale += bool(test_scales - train_scales)
    assert reused_scale >= 25 and novel_scale >= 2
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (None, 1),
                                    (None, 4), (2, None), (10, None), (4, 2)):
        puzzle = generate_puzzle(991, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(991, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(17, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError((train_count, test_count))
