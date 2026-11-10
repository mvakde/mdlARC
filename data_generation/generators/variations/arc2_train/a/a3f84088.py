"""Hollow-square completion puzzles with a common concentric color cycle."""
import numpy as np


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _counts(seed, train, test):
    for name, value, low, high in (("num_train", train, 2, 10), ("num_test", test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be in {low}..{high}")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(height, width, background, accent, squares):
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for top, left, side, frame in squares:
        bottom, right = top + side - 1, left + side - 1
        inp[top, left:right + 1] = frame
        inp[bottom, left:right + 1] = frame
        inp[top:bottom + 1, left] = frame
        inp[top:bottom + 1, right] = frame
        axis = np.arange(side)
        d = np.minimum(axis, axis[::-1])
        depth = np.minimum(d[:, None], d[None, :])
        out[top:top + side, left:left + side] = np.array(
            [frame, accent, frame, background], dtype=np.int8)[depth % 4]
    return {"input": inp, "output": out}


def _single_layout(rng, side, maximum, edge_mode):
    if edge_mode == "inset":
        contacts = 0
    elif edge_mode == "edge":
        contacts = int(rng.integers(1, 5))
    else:
        contacts = int(rng.integers(0, 5))
    choices = []
    for bits in range(16):
        free_v = 2 - bool(bits & 1) - bool(bits & 2)
        free_h = 2 - bool(bits & 4) - bool(bits & 8)
        if bits.bit_count() == contacts and side + max(free_v, free_h) <= maximum:
            choices.append(bits)
    if not choices:
        # This can only arise for a very large square and an infeasible contact count.
        choices = [bits for bits in range(16) if side + max(
            2 - bool(bits & 1) - bool(bits & 2),
            2 - bool(bits & 4) - bool(bits & 8)) <= maximum]
    bits = choices[int(rng.integers(len(choices)))]
    margins = [0 if bits & (1 << i) else int(rng.integers(1, min(5, maximum - side) + 1))
               for i in range(4)]
    for a, b in ((0, 1), (2, 3)):
        while side + margins[a] + margins[b] > maximum:
            i = a if margins[a] > margins[b] else b
            if margins[i] <= (0 if bits & (1 << i) else 1):
                i = b if i == a else a
            margins[i] -= 1
    top, bottom, left, right = margins
    return side + top + bottom, side + left + right, top, left


def _multi_layout(rng, frames, count):
    if count == 2:
        sides = [int(rng.choice(np.array([6, 8, 10, 12]))),
                 int(rng.choice(np.array([7, 9, 11])))]
        rng.shuffle(sides)
    else:
        sides = [6, 7, 8]
        rng.shuffle(sides)
    gap = int(rng.integers(1, 4))
    before = int(rng.integers(0, 3))
    after = int(rng.integers(0, 3))
    long_dim = before + sum(sides) + gap * (count - 1) + after
    if long_dim > 30:
        reduction = min(after, long_dim - 30)
        after -= reduction
        long_dim -= reduction
    if long_dim > 30:
        before -= long_dim - 30
        long_dim = 30
    cross_dim = max(sides) + int(rng.integers(0, 5))
    horizontal = bool(rng.integers(0, 2))
    cursor = before
    squares = []
    for side, frame in zip(sides, frames):
        cross = int(rng.integers(0, cross_dim - side + 1))
        top, left = (cross, cursor) if horizontal else (cursor, cross)
        squares.append((top, left, side, frame))
        cursor += side + gap
    height, width = (cross_dim, long_dim) if horizontal else (long_dim, cross_dim)
    return height, width, squares


def _components(mask):
    unseen = set(zip(*np.nonzero(mask)))
    groups = []
    while unseen:
        start = unseen.pop()
        group = [start]
        for row, col in group:
            for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if point in unseen:
                    unseen.remove(point)
                    group.append(point)
        groups.append(group)
    return groups


def _parse_input(grid):
    candidates = []
    for background in (int(v) for v in np.unique(grid)):
        groups = _components(grid != background)
        squares = []
        valid = bool(groups)
        for group in groups:
            rows, cols = zip(*group)
            top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
            side = bottom - top + 1
            if side < 6 or right - left + 1 != side or len(group) != 4 * side - 4:
                valid = False
                break
            frame = int(grid[top, left])
            border = {(top, left + i) for i in range(side)}
            border |= {(bottom, left + i) for i in range(side)}
            border |= {(top + i, left) for i in range(side)}
            border |= {(top + i, right) for i in range(side)}
            if set(group) != border or any(int(grid[r, c]) != frame for r, c in group):
                valid = False
                break
            if not np.all(grid[top + 1:bottom, left + 1:right] == background):
                valid = False
                break
            squares.append((top, left, side, frame))
        if valid and 1 <= len(squares) <= 3:
            candidates.append((background, sorted(squares)))
    _check(len(candidates) == 1, "input must contain identifiable hollow squares")
    return candidates[0]


def _source_inputs():
    specs = ((16, 16, 1, 3, 12), (6, 6, 0, 0, 6),
             (19, 19, 1, 1, 17), (13, 13, 1, 1, 9), (30, 30, 1, 1, 25))
    return {_key(_make_pair(h, w, 0, 2, [(t, l, s, 5)])["input"])
            for h, w, t, l, s in specs}


def _contact(shape, square):
    top, left, side, _ = square
    return sum((top == 0, left == 0, top + side == shape[0], left + side == shape[1]))


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "sections must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, accents = set(), set()
    records = {"train": [], "test": []}
    source = _source_inputs()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid type")
                _check(all(1 <= n <= 30 for n in grid.shape), "invalid grid size")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid color")
            _check(inp.shape == out.shape and not np.array_equal(inp, out), "output must change interior")
            key = _key(inp)
            _check(key not in seen and key not in source, "duplicate or source input")
            seen.add(key)
            background, squares = _parse_input(inp)
            addition = {int(v) for v in np.unique(out)} - {int(v) for v in np.unique(inp)}
            _check(len(addition) == 1, "output must introduce one accent")
            accent = next(iter(addition))
            accents.add(accent)
            expected = inp.copy()
            for top, left, side, frame in squares:
                for row in range(top, top + side):
                    for col in range(left, left + side):
                        depth = min(row - top, col - left, top + side - 1 - row,
                                    left + side - 1 - col)
                        expected[row, col] = (frame, accent, frame, background)[depth % 4]
            _check(np.array_equal(out, expected), "incorrect concentric cycle")
            records[split].append((inp.shape, background, squares))
    _check(len(accents) == 1, "accent must be consistent across the puzzle")
    train_sides = [s[2] for _, _, sq in records["train"] for s in sq]
    _check(max(train_sides) >= 15, "training must show two complete cycles")
    _check({s % 2 for s in train_sides} == {0, 1},
           "training must demonstrate odd and even centers")
    for shape, _, squares in records["train"]:
        if len(squares) == 1:
            _check(6 <= squares[0][2] <= 19 and max(shape) <= 22,
                   "single-square training example exceeds its size regime")
        else:
            _check(2 <= len(squares) <= 3 and all(6 <= s[2] <= 12 for s in squares),
                   "multi-square training example exceeds its size regime")
    for _, _, squares in records["test"]:
        if len(squares) == 1:
            _check(6 <= squares[0][2] <= 28, "single-square test exceeds its size regime")
        else:
            _check(2 <= len(squares) <= 3 and all(6 <= s[2] <= 12 for s in squares),
                   "multi-square test exceeds its size regime")
    if any(len(sq) > 1 for _, _, sq in records["test"]):
        _check(any(len(sq) > 1 for _, _, sq in records["train"]),
               "multi-square test needs a training demonstration")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    accent = int(rng.integers(0, 10))
    available = [c for c in range(10) if c != accent]
    rng.shuffle(available)
    varying = seed % 5 in (1, 2)
    pool = available[:5] if varying else available[:2]
    multi = seed % 4 == 0
    size_mode = ("large", "same", "smaller")[seed % 3]
    edge_mode = ("inset", "edge", "mixed", "mixed", "edge", "inset", "mixed")[seed % 7]
    puzzle = {"train": [], "test": []}
    used = set()
    train_max = None
    for split, count in (("train", train_count), ("test", test_count)):
        for slot in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for attempt in range(128):
                if varying:
                    bg_index = (slot + (0 if split == "train" else train_count)) % len(pool)
                    background = pool[bg_index]
                    candidates = [c for c in pool if c != background]
                    pair_rng.shuffle(candidates)
                    frame = candidates[0]
                else:
                    background, frame = pool
                is_multi = multi and ((split == "train" and slot == 1) or
                                      (split == "test" and slot == 0) or
                                      (slot >= 2 and bool(pair_rng.integers(0, 3) == 0)))
                if is_multi:
                    count_objects = 2 + int(pair_rng.integers(0, 2))
                    frames = [frame] * count_objects
                    if varying and bool(pair_rng.integers(0, 2)):
                        options = [c for c in pool if c != background]
                        pair_rng.shuffle(options)
                        frames = options[:count_objects]
                    h, w, squares = _multi_layout(pair_rng, frames, count_objects)
                else:
                    if split == "train":
                        if slot == 0:
                            side = int(pair_rng.integers(15, 20))
                        elif slot == 1:
                            choices = [s for s in range(6, 20) if s % 2 != train_max % 2]
                            side = int(pair_rng.choice(np.array(choices)))
                        else:
                            side = int(pair_rng.integers(6, 20))
                        maximum = 22
                    else:
                        if size_mode == "large":
                            side = int(pair_rng.integers(20, 29))
                        elif size_mode == "same":
                            side = train_max if slot == 0 else int(pair_rng.integers(train_max - 2,
                                                                                      train_max + 1))
                        else:
                            side = int(pair_rng.integers(6, train_max))
                        maximum = 30
                    h, w, top, left = _single_layout(pair_rng, side, maximum, edge_mode)
                    squares = [(top, left, side, frame)]
                pair = _make_pair(h, w, background, accent, squares)
                key = _key(pair["input"])
                if key not in used and key not in _source_inputs():
                    break
            else:
                raise RuntimeError("could not produce a unique pair")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                side_max = max(s[2] for s in squares)
                train_max = side_max if train_max is None else max(train_max, side_max)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    accents = []
    for pair in train_pairs:
        input_colors = {int(v) for v in np.unique(pair["input"])}
        new_colors = {int(v) for v in np.unique(pair["output"])} - input_colors
        _check(len(new_colors) == 1, "training accent is unclear")
        accents.append(next(iter(new_colors)))
    _check(len(set(accents)) == 1, "training does not establish one accent")
    accent = accents[0]
    _check(accent not in test_input, "test input already contains the accent")
    background, squares = _parse_input(test_input)
    predicted = test_input.copy()
    for top, left, side, frame in squares:
        for row in range(top, top + side):
            for col in range(left, left + side):
                depth = min(row - top, col - left, top + side - 1 - row,
                            left + side - 1 - col)
                predicted[row, col] = (frame, accent, frame, background)[depth % 4]
    return predicted


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None), (7, 3)):
        p = generate_puzzle(1001, train, test)
        validate_generated_puzzle(p)
        _check(train is None or len(p["train"]) == train, "train count ignored")
        _check(test is None or len(p["test"]) == test, "test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1001, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    _check(_puzzle_key(generate_puzzle(12345)) == _puzzle_key(generate_puzzle(12345)),
           "generation is nondeterministic")
    train_counts, test_counts, keys = set(), set(), set()
    sides, shapes, placements, palettes, contacts = set(), set(), set(), set(), set()
    relations, multi_puzzles, color_puzzles, only_inset, only_edge = set(), 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        training = [{role: np.array(pair[role], copy=True) for role in ("input", "output")}
                    for pair in p["train"]]
        for pair in p["test"]:
            predicted = _predict_from_training(training, np.array(pair["input"], copy=True))
            _check(np.array_equal(predicted, pair["output"]),
                   "test output is not recoverable from training and its input")
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        keys.add(_puzzle_key(p))
        local_palettes, local_contacts = set(), []
        train_sides, test_sides = [], []
        has_multi = False
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                bg, squares = _parse_input(inp)
                accent = next(iter({int(v) for v in np.unique(out)} -
                                   {int(v) for v in np.unique(inp)}))
                has_multi |= len(squares) > 1
                shapes.add(inp.shape)
                for square in squares:
                    top, left, side, frame = square
                    sides.add(side)
                    placements.add((inp.shape, top, left, side))
                    palettes.add((bg, frame, accent))
                    local_palettes.add((bg, frame))
                    contact = _contact(inp.shape, square)
                    contacts.add(contact)
                    local_contacts.append(contact)
                    (train_sides if split == "train" else test_sides).append(side)
        multi_puzzles += has_multi
        color_puzzles += len(local_palettes) > 1
        only_inset += bool(local_contacts) and all(c == 0 for c in local_contacts)
        only_edge += bool(local_contacts) and all(c > 0 for c in local_contacts)
        relations.add("larger" if max(test_sides) > max(train_sides) else
                      "smaller" if max(test_sides) < max(train_sides) else "equal")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3 and len(keys) == 42,
           "count or cross-seed diversity collapsed")
    _check(len(sides) >= 18 and len(shapes) >= 45 and len(placements) >= 90,
           "geometric diversity collapsed")
    _check(len(palettes) >= 25 and color_puzzles >= 10, "palette diversity collapsed")
    _check(contacts == {0, 1, 2, 3, 4} and only_inset > 0 and only_edge > 0,
           "edge-contact diversity collapsed")
    _check(relations == {"larger", "smaller", "equal"} and multi_puzzles >= 6,
           "size or object-count regimes collapsed")
