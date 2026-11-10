"""Compositional instances of the fixed 2-by-2 block attraction task."""

import numpy as np


# The four original solved examples are regression evidence, never templates.
_SOURCE = (
    (
        ("000000000000000", "000100000000000", "000000030055000",
         "000000000055000", "000550200000000", "000550000009000",
         "000000000002010", "000000000000003", "000000000000000",
         "002000000000000", "000000006055000", "005500000055000",
         "005500200000000", "000000000000000", "000000004000700"),
        ("000000000000000", "000000000000000", "000000000355000",
         "000100000055000", "000552000009100", "000550000000000",
         "000000000000000", "000000000000000", "000000000000000",
         "000000000002300", "002000000655000", "005500000055000",
         "005520000040700", "000000000000000", "000000000000000"),
    ),
    (
        ("0000000000", "0550300550", "0550000550", "0000000000",
         "0000400000", "0080000600", "0000000000", "3000200550",
         "0070000550", "0000000000"),
        ("0000000000", "0553000550", "0550000550", "3084000000",
         "0000000000", "0000000000", "0000000600", "0000002550",
         "0000007550", "0000000000"),
    ),
    (
        ("000000000000000", "005500000005500", "005508000005500",
         "000000000000000", "000000000000006", "000000550000000",
         "000000550000055", "000000000000055", "000100009000000",
         "000000000000000", "005500000005500", "005500000005500",
         "000000000600000", "000000000000000", "000000000000000"),
        ("000000000000000", "005500000005500", "005580000005500",
         "000000000000000", "000000000000000", "000000550000006",
         "000000550000055", "000000009000055", "000000000000000",
         "000100000000000", "005500000005500", "005500000005500",
         "000000000060000", "000000000000000", "000000000000000"),
    ),
    (
        ("000000004000000", "005500000009000", "005500000000550",
         "000000000000550", "000000000000000", "070000000000000",
         "000550000000000", "000550002000080", "000000000000000",
         "000000008009000", "000000000000000", "000000005500000",
         "000000005500000", "550030000000060", "550000000000000"),
        ("000040000000000", "005500000009000", "005500000000550",
         "000000000000550", "000000000000080", "007000000000000",
         "000550000000000", "000552000000000", "000000000000000",
         "000000000000000", "000000008090000", "000000005500000",
         "000000005500000", "553000000060000", "550000000000000"),
    ),
)
_SOURCE_INPUTS = {pair[0] for pair in _SOURCE}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    tr = _checked_count(num_train, "num_train", 2, 10)
    te = _checked_count(num_test, "num_test", 1, 4)
    if tr is None and te is None:
        te = 1 + seed % 4
        tr = te + 1 + (seed // 4) % (10 - te)
    elif tr is None:
        tr = te + 1 + seed % (10 - te)
    elif te is None:
        te = 1 + seed % min(4, tr - 1)
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _signature(grid):
    return grid.shape, grid.tobytes()


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _rect_cells(rect):
    r, c = rect
    return {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}


def _halo(rect, shape):
    ar, ac = rect
    h, w = shape
    return [(r, c) for r in range(max(0, ar - 1), min(h, ar + 3))
            for c in range(max(0, ac - 1), min(w, ac + 3))
            if not (ar <= r <= ar + 1 and ac <= c <= ac + 1)]


def _parse(grid):
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    squares = {}
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color != bg and bool(np.all(grid[r:r + 2, c:c + 2] == color)):
                squares.setdefault(color, []).append((r, c))
    assert len(squares) == 1
    block_color, blocks = next(iter(squares.items()))
    covered = set()
    for rect in blocks:
        cells = _rect_cells(rect)
        assert not covered.intersection(cells)
        covered.update(cells)
    assert len(covered) == int(np.count_nonzero(grid == block_color))
    tokens = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w)
              if int(grid[r, c]) != bg and (r, c) not in covered]
    assert tokens
    assert len(tokens) + len(covered) < h * w / 2
    return bg, block_color, sorted(blocks), tokens


def _distance(point, rect):
    r, c = point
    ar, ac = rect
    return max(ar - 1 - r, r - ar - 2, ac - 1 - c, c - ac - 2, 0)


def _landing(point, rect):
    r, c = point
    ar, ac = rect
    return (min(max(r, ar - 1), ar + 2),
            min(max(c, ac - 1), ac + 2))


def _predict(grid):
    bg, block_color, blocks, tokens = _parse(grid)
    output = np.full(grid.shape, bg, dtype=np.int8)
    for ar, ac in blocks:
        output[ar:ar + 2, ac:ac + 2] = block_color
    landings = set()
    moves = []
    for r, c, color in tokens:
        distances = [_distance((r, c), block) for block in blocks]
        best = min(distances)
        assert distances.count(best) == 1
        block = blocks[distances.index(best)]
        end = _landing((r, c), block)
        assert end in _halo(block, grid.shape)
        assert end not in landings
        assert int(output[end]) == bg
        landings.add(end)
        output[end] = color
        moves.append(((r, c), end, block, color))
    return output, moves


def _place_blocks(rng, shape, wanted):
    h, w = shape
    positions = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    rng.shuffle(positions)
    if rng.random() < 0.25:
        corners = [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]
        positions.insert(0, corners[int(rng.integers(0, 4))])
    elif rng.random() < 0.3:
        edges = [p for p in positions if p[0] in (0, h - 2) or p[1] in (0, w - 2)]
        positions.insert(0, edges[int(rng.integers(0, len(edges)))])
    blocks = []
    for rect in positions:
        if rect in blocks:
            continue
        if all(abs(rect[0] - other[0]) >= 4 or abs(rect[1] - other[1]) >= 4
               for other in blocks):
            blocks.append(rect)
            if len(blocks) == wanted:
                break
    return sorted(blocks)


def _size(rng, regime):
    weights = (0.30, 0.48, 0.22) if regime == "source_train" else (0.18, 0.47, 0.35)
    band = int(rng.choice(3, p=weights))
    if band == 0:
        return int(rng.integers(8, 13)), int(rng.integers(8, 14))
    if band == 1:
        return int(rng.integers(11, 21)), int(rng.integers(10, 22))
    return int(rng.integers(16, 31)), int(rng.integers(12, 31))


def _candidates(blocks, shape):
    h, w = shape
    block_cells = set().union(*(_rect_cells(b) for b in blocks))
    candidates = []
    for block in blocks:
        for end in _halo(block, shape):
            ar, ac = block
            er, ec = end
            vertical = -1 if er < ar else (1 if er > ar + 1 else 0)
            horizontal = -1 if ec < ac else (1 if ec > ac + 1 else 0)
            directions = []
            if vertical:
                directions.append((vertical, 0, "vertical"))
            if horizontal:
                directions.append((0, horizontal, "horizontal"))
            if vertical and horizontal:
                directions.append((vertical, horizontal, "diagonal"))
            candidates.append((end, end, block, "stationary", 0))
            for dr, dc, orientation in directions:
                for length in range(1, max(h, w)):
                    start = (er + dr * length, ec + dc * length)
                    if not (0 <= start[0] < h and 0 <= start[1] < w):
                        break
                    if start in block_cells:
                        continue
                    if _landing(start, block) != end:
                        continue
                    distances = [_distance(start, other) for other in blocks]
                    if distances.count(min(distances)) != 1 or blocks[distances.index(min(distances))] != block:
                        continue
                    candidates.append((start, end, block, orientation, length))
    return candidates


def _choose_tokens(rng, candidates, count, required):
    selected = []
    used_starts = set()
    used_ends = set()
    block_weight = {}
    for candidate in candidates:
        block_weight[candidate[2]] = float(rng.uniform(0.5, 2.0))

    def take(orientation=None, travel=None):
        pool = [x for x in candidates if (orientation is None or x[3] == orientation)
                and (travel is None or (x[4] == 0 if travel == 0 else
                                           x[4] <= 2 if travel == 1 else x[4] >= 3))]
        if not pool:
            return False
        order = rng.permutation(len(pool))
        for index in order:
            item = pool[int(index)]
            start, end, block, _, _ = item
            if (start in used_starts or end in used_ends or
                    start in used_ends or end in used_starts):
                continue
            if rng.random() > min(1.0, block_weight[block] / 2):
                continue
            selected.append(item)
            used_starts.add(start)
            used_ends.add(end)
            return True
        return False

    for orientation in required:
        if not take(orientation):
            return None
    attempts = 0
    while len(selected) < count and attempts < 100:
        attempts += 1
        travel = int(rng.choice(3, p=(0.16, 0.42, 0.42)))
        if not take(travel=travel):
            take()
    if len(selected) < count or all(x[4] == 0 for x in selected):
        return None
    return selected


def _make_pair(rng, regime, bg, block_color, reserved, required, force_unseen):
    for _ in range(30):
        shape = _size(rng, regime)
        h, w = shape
        maximum = min(7, max(1, (h // 4) * (w // 4)))
        wanted_blocks = int(rng.integers(1, maximum + 1))
        blocks = _place_blocks(rng, shape, wanted_blocks)
        candidates = _candidates(blocks, shape)
        maximum_tokens = min(20, len(blocks) * 5 + 3, len({x[1] for x in candidates}))
        if maximum_tokens < max(1, len(required)):
            continue
        low = max(1, len(required))
        if regime == "target_test" and rng.random() < 0.55:
            low = min(maximum_tokens, max(low, 4))
        count = int(rng.integers(low, maximum_tokens + 1))
        selected = _choose_tokens(rng, candidates, count, required)
        if selected is None:
            continue
        grid = np.full(shape, bg, dtype=np.int8)
        for ar, ac in blocks:
            grid[ar:ar + 2, ac:ac + 2] = block_color
        available = [c for c in range(10) if c not in (bg, block_color)
                     and (regime == "target_test" or c != reserved)]
        if force_unseen and reserved not in available:
            continue
        palette_size = int(rng.integers(1, min(len(available), count) + 1))
        palette = list(map(int, rng.choice(available, size=palette_size, replace=False)))
        if force_unseen and reserved not in palette:
            palette[0] = reserved
        colors = palette + [palette[int(rng.integers(0, len(palette)))]
                            for _ in range(count - len(palette))]
        rng.shuffle(colors)
        for item, color in zip(selected, colors):
            r, c = item[0]
            grid[r, c] = color
        try:
            output, moves = _predict(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, output):
            continue
        return {"input": grid, "output": output}
    raise AssertionError("Could not construct an unambiguous scene")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    reserved = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_bg = None
    previous_block = None
    pair_features = []
    for split, regime, amount in (("train", "source_train", train_count),
                                  ("test", "target_test", test_count)):
        for index in range(amount):
            for _ in range(20):
                force_unseen = split == "test" and index == 0 and seed % 2 == 0
                colors = [c for c in range(10) if c != reserved or
                          (split == "test" and not force_unseen)]
                backgrounds = [c for c in colors if c != previous_bg]
                bg = int(rng.choice(backgrounds))
                block_colors = [c for c in colors if c not in (bg, previous_block)]
                block_color = int(rng.choice(block_colors))
                required = ("vertical", "horizontal", "diagonal", "stationary") if split == "train" and index == 0 else ()
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _make_pair(np.random.default_rng(child_seed), regime, bg,
                                  block_color, reserved, required, force_unseen)
                key = _signature(pair["input"])
                if key in seen or _rows(pair["input"]) in _SOURCE_INPUTS:
                    continue
                _, _, blocks, tokens = _parse(pair["input"])
                _, moves = _predict(pair["input"])
                features = (len(blocks), len(tokens),
                            sum(start == end for start, end, _, _ in moves),
                            max(max(abs(start[0] - end[0]),
                                    abs(start[1] - end[1])) for start, end, _, _ in moves))
                if (split == "test" and index == amount - 1 and
                        any(len({old[k] for old in pair_features} | {features[k]}) < 2
                            for k in range(4))):
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                pair_features.append(features)
                previous_bg, previous_block = bg, block_color
                break
            else:
                raise AssertionError("Could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_orientations = set()
    all_orientations = set()
    backgrounds = set()
    block_colors = set()
    block_counts = set()
    token_counts = set()
    stationary_counts = set()
    travel_lengths = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert a.shape == b.shape and not np.array_equal(a, b)
            sig = _signature(a)
            assert sig not in seen and _rows(a) not in _SOURCE_INPUTS
            seen.add(sig)
            bg, block_color, blocks, tokens = _parse(a)
            backgrounds.add(bg)
            block_colors.add(block_color)
            block_counts.add(len(blocks))
            token_counts.add(len(tokens))
            expected, moves = _predict(a)
            assert np.array_equal(expected, b)
            assert len(moves) == len(tokens)
            stationary_counts.add(sum(start == end for start, end, _, _ in moves))
            travel_lengths.add(max(max(abs(start[0] - end[0]),
                                       abs(start[1] - end[1]))
                                   for start, end, _, _ in moves))
            for start, end, block, color in moves:
                assert int(a[start]) == int(b[end]) == color
                assert end in _halo(block, a.shape)
                dr, dc = end[0] - start[0], end[1] - start[1]
                orientation = ("stationary" if dr == dc == 0 else
                               "diagonal" if dr and dc else
                               "vertical" if dr else "horizontal")
                all_orientations.add(orientation)
                if split == "train":
                    train_orientations.add(orientation)
    assert len(backgrounds) >= 2 and len(block_colors) >= 2
    assert all(len(values) >= 2 for values in
               (block_counts, token_counts, stationary_counts, travel_lengths))
    assert all_orientations <= train_orientations
    assert {"vertical", "horizontal", "diagonal", "stationary"} <= train_orientations


def _source_regression():
    # Independently check the source labels' invariant facts, including cases
    # whose crowded destination choice is not specified by this generator.
    for input_rows, output_rows in _SOURCE:
        a = np.array([[int(ch) for ch in row] for row in input_rows], dtype=np.int8)
        b = np.array([[int(ch) for ch in row] for row in output_rows], dtype=np.int8)
        assert a.shape == b.shape
        assert sorted(a.ravel()) == sorted(b.ravel())
        block_cells = set()
        blocks = []
        for r in range(a.shape[0] - 1):
            for c in range(a.shape[1] - 1):
                if (np.all(a[r:r + 2, c:c + 2] == 5) and
                        (r == 0 or a[r - 1, c] != 5) and
                        (c == 0 or a[r, c - 1] != 5)):
                    blocks.append((r, c))
                    block_cells.update(_rect_cells((r, c)))
        assert blocks
        for r, c in block_cells:
            assert a[r, c] == b[r, c] == 5
        for r in range(b.shape[0]):
            for c in range(b.shape[1]):
                if (r, c) not in block_cells and b[r, c] != 0:
                    assert any((r, c) in _halo(block, b.shape) for block in blocks)
        assert np.count_nonzero(a != 5) == np.count_nonzero(b != 5)
    # Exact coordinates from the review guard against reintroducing queuing.
    assert _SOURCE[1][1][3][2] == "8"
    assert _SOURCE[1][1][8][6] == "7"
    assert _SOURCE[2][1][7][8] == "9"
    assert _SOURCE[3][1][0][4] == "4"


def self_test() -> None:
    _source_regression()
    puzzles = []
    counts = (set(), set())
    shapes = set()
    block_counts = set()
    token_counts = set()
    stationary_counts = set()
    palettes = set()
    layouts = set()
    corner = 0
    long_move = 0
    unseen = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        counts[0].add(len(puzzle["train"]))
        counts[1].add(len(puzzle["test"]))
        train_colors = set()
        for pair in puzzle["train"]:
            bg, block_color, _, tokens = _parse(pair["input"])
            train_colors.update(color for _, _, color in tokens)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                bg, block_color, blocks, tokens = _parse(a)
                expected, moves = _predict(a)
                assert np.array_equal(expected, pair["output"])
                shapes.add(a.shape)
                block_counts.add(len(blocks))
                token_counts.add(len(tokens))
                stationary_counts.add(sum(start == end for start, end, _, _ in moves))
                palettes.add(len({color for _, _, color in tokens}))
                layouts.add((a.shape, tuple(blocks), tuple((r, c) for r, c, _ in tokens)))
                corner += sum(block in ((0, 0), (0, a.shape[1] - 2),
                                        (a.shape[0] - 2, 0),
                                        (a.shape[0] - 2, a.shape[1] - 2)) for block in blocks)
                long_move += sum(max(abs(start[0] - end[0]), abs(start[1] - end[1])) >= 4
                                 for start, end, _, _ in moves)
                if split == "test" and any(color not in train_colors for _, _, color in tokens):
                    unseen += 1
    assert len(counts[0]) >= 4 and len(counts[1]) >= 3
    assert len(shapes) >= 30 and len(block_counts) >= 4
    assert len(token_counts) >= 8 and len(stationary_counts) >= 3
    assert len(palettes) >= 5 and len(layouts) >= 150
    assert corner >= 10 and long_move >= 25 and unseen >= 10
    assert all(np.array_equal(x["input"], y["input"]) and
               np.array_equal(x["output"], y["output"])
               for split in ("train", "test")
               for x, y in zip(generate_puzzle(77)[split], generate_puzzle(77)[split]))
    assert _signature(generate_puzzle(77)["train"][0]["input"]) != _signature(generate_puzzle(78)["train"][0]["input"])
    for tr, te in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(8123, tr, te)
        assert (len(p["train"]), len(p["test"])) == (tr, te)
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(9000, tr, te)
        assert tr is None or len(p["train"]) == tr
        assert te is None or len(p["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4)):
        try:
            generate_puzzle(1, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
