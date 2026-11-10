import numpy as np


_SOURCE_INPUTS = (
    (
        "0000000000", "0000600000", "0006060000", "0060006000",
        "0006060000", "0000600200", "0000002220", "0300022222",
        "3330002220", "0300000200",
    ),
    (
        "0000000700", "0000007770", "0000400777", "0004040070",
        "0000400000", "0000000100", "0800001110", "8880011100",
        "0800001000", "0000000000",
    ),
    (
        "0000400000", "0004440000", "0044400000", "0444000000",
        "4440000900", "0400009090", "0008000900", "0088800000",
        "0008880000", "0000800000",
    ),
    (
        "0000000300", "0000003030", "0050000303", "0555000030",
        "5555500000", "0555000000", "0050007000", "0000077700",
        "0000007000", "0000000000",
    ),
    (
        "0000000000004000", "0088000000044400", "0088000000004440",
        "8800880000000440", "8800880000000000", "0088000000000000",
        "0088000000000000", "0000001100002000", "0000001100022200",
        "0000111111002000", "0000111111000000", "0000001100000000",
        "0000001100000000",
    ),
    (
        "0000000000000000", "0000000000000000", "0000000000000000",
        "0020002000200020", "0222022202220222", "0020002000200020",
        "0000000000000000", "0007000100000700", "0077701010007770",
        "0007000100000700", "0000000000000000", "0000000000000000",
    ),
)


def _components(mask: np.ndarray, diagonal: bool = False) -> list[list[tuple[int, int]]]:
    """Connected components of a Boolean mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diagonal:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    result = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        stack = [(int(sy), int(sx))]
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height and 0 <= nx < width and
                        mask[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(component)
    return result


def _pockets(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    """Background regions blocked from the exterior by one color alone."""
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    stack = []
    for y in range(height):
        for x in (0, width - 1):
            if grid[y, x] != color and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    for x in range(width):
        for y in (0, height - 1):
            if grid[y, x] != color and not seen[y, x]:
                seen[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and
                    grid[ny, nx] != color and not seen[ny, nx]):
                seen[ny, nx] = True
                stack.append((ny, nx))
    return _components((grid == 0) & ~seen)


def _diamond(radius: int) -> np.ndarray:
    y, x = np.indices((2 * radius + 1, 2 * radius + 1))
    return abs(y - radius) + abs(x - radius) == radius


def _target_mask(rng: np.random.Generator, family: str) -> np.ndarray:
    if family == "diamond":
        mask = _diamond(int(rng.choice((1, 1, 2, 3))))
    elif family == "scaled":
        # The source test's six-by-six, 16-pixel outline is this mask.
        mask = np.kron(_diamond(1), np.ones((2, 2), dtype=bool))
    elif family == "staggered":
        mask = np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                         [0, 1, 0, 1], [0, 0, 1, 0]], dtype=bool)
    elif family in ("linked", "multi"):
        count = 2 if family == "linked" else int(rng.integers(3, 5))
        mask = np.zeros((3, 2 * count + 1), dtype=bool)
        for center in range(1, 2 * count, 2):
            mask[0, center] = mask[2, center] = True
            mask[1, center - 1] = mask[1, center + 1] = True
    elif family == "frame":
        height = int(rng.integers(4, 10))
        width = int(rng.integers(4, 11))
        stroke = int(rng.integers(1, min(3, (height - 1) // 2 + 1,
                                             (width - 1) // 2 + 1)))
        mask = np.ones((height, width), dtype=bool)
        mask[stroke:-stroke, stroke:-stroke] = False
    else:
        raise ValueError("unknown target family")
    return np.rot90(mask, int(rng.integers(0, 4)))


def _distractor_mask(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "plus":
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    elif kind == "solid":
        mask = np.ones((int(rng.integers(2, 7)),
                        int(rng.integers(2, 7))), dtype=bool)
    elif kind == "line":
        mask = np.ones((1, int(rng.integers(2, 8))), dtype=bool)
    elif kind == "ell":
        mask = np.zeros((int(rng.integers(3, 7)),
                         int(rng.integers(3, 7))), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == "tee":
        mask = np.zeros((int(rng.integers(3, 6)),
                         int(rng.integers(3, 7))), dtype=bool)
        mask[0, :] = True
        mask[:, mask.shape[1] // 2] = True
    elif kind == "open_diamond":
        mask = _diamond(int(rng.integers(1, 4)))
        mask[0, mask.shape[1] // 2] = False
    elif kind == "open_frame":
        mask = np.ones((int(rng.integers(4, 8)),
                        int(rng.integers(4, 9))), dtype=bool)
        mask[1:-1, 1:-1] = False
        mask[0, 1:3] = False
    elif kind == "stairs":
        length = int(rng.integers(3, 7))
        mask = np.zeros((length, length + 1), dtype=bool)
        for i in range(length):
            mask[i, i:i + 2] = True
    else:
        raise ValueError("unknown distractor kind")
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.rot90(mask, int(rng.integers(0, 4)))


def _place(grid: np.ndarray, mask: np.ndarray, color: int,
           rng: np.random.Generator, edge: bool = False) -> bool:
    height, width = grid.shape
    mh, mw = mask.shape
    if mh > height or mw > width:
        return False
    positions = [(y, x) for y in range(height - mh + 1)
                       for x in range(width - mw + 1)]
    rng.shuffle(positions)
    if edge:
        positions.sort(key=lambda pos: not (
            pos[0] == 0 or pos[1] == 0 or
            pos[0] + mh == height or pos[1] + mw == width))
    ys, xs = np.nonzero(mask)
    for top, left in positions:
        if any(np.any(grid[max(0, top + int(y) - 1):
                           min(height, top + int(y) + 2),
                           max(0, left + int(x) - 1):
                           min(width, left + int(x) + 2)])
               for y, x in zip(ys, xs)):
            continue
        grid[top:top + mh, left:left + mw][mask] = color
        return True
    return False


def _resolve_counts(rng: np.random.Generator,
                    num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or
            not low <= value <= high
        ):
            raise ValueError("invalid pair count")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, (num_train or 10) - 1) + 1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _source_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _cropped(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _is_simple_diamond(mask: np.ndarray) -> bool:
    crop = _cropped(mask)
    return any(np.array_equal(crop, _diamond(radius)) for radius in (1, 2, 3))


def _make_pair(rng: np.random.Generator, family: str, target_color: int,
               regime: str, complexity: str) -> dict:
    kinds = ("plus", "solid", "line", "ell", "tee", "open_diamond",
             "open_frame", "stairs")
    for attempt in range(90):
        target = _target_mask(rng, family)
        repeat = complexity == "repeat"
        large = complexity == "large"
        compact = (not repeat and not large and
                   family in ("diamond", "staggered", "linked") and
                   rng.random() < 0.25)
        if compact:
            height = width = 10
        else:
            min_side = 14 if repeat else 10
            high = 24 if regime == "target_test" else 21
            height = int(rng.integers(max(min_side, target.shape[0] + 4), high + 1))
            width = int(rng.integers(max(min_side, target.shape[1] + 4), high + 1))
        grid = np.zeros((height, width), dtype=np.int8)
        if not _place(grid, target, target_color, rng,
                      edge=bool(rng.random() < 0.3)):
            continue
        n_colors = 3 if compact else int(rng.integers(3, 6))
        available = [c for c in range(1, 10) if c != target_color]
        other_colors = [int(c) for c in rng.choice(available,
                                                  size=n_colors - 1,
                                                  replace=False)]
        n_objects = (2 if compact else
                     max(n_colors - 1, int(rng.integers(2, 7))))
        if repeat:
            n_objects = max(n_objects, 4)
        colors = other_colors[:]
        while len(colors) < n_objects:
            colors.append(int(rng.choice(other_colors)))
        if repeat:
            colors[0:3] = [other_colors[0]] * 3
            colors[-1] = other_colors[-1]
        motifs = []
        repeated_motif = _distractor_mask(rng, "plus") if repeat else None
        for index in range(n_objects):
            if repeat and index < 3:
                motif = repeated_motif
            elif large and index == 0:
                motif = np.ones((int(rng.integers(4, 7)),
                                 int(rng.integers(4, 7))), dtype=bool)
            elif compact:
                motif = _distractor_mask(rng, str(rng.choice((
                    "plus", "line", "ell", "open_diamond"))))
            else:
                motif = _distractor_mask(rng, str(rng.choice(kinds)))
            motifs.append(motif)
        if not all(_place(grid, motif, color, rng)
                   for motif, color in zip(motifs, colors)):
            continue
        foreground = [int(c) for c in np.unique(grid) if c != 0]
        if (len(foreground) != n_colors or
                [c for c in foreground if _pockets(grid, c)] != [target_color] or
                _source_key(grid) in _SOURCE_INPUTS):
            continue
        return {"input": grid, "output": np.array([[target_color]], dtype=np.int8)}
    raise RuntimeError("could not construct a separated enclosure puzzle")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The shared convention is zero background and the color that encloses it.
    # Literal foreground colors have no fixed role across pairs.
    puzzle = {"train": [], "test": []}
    seen = set()
    train_answers = set()
    families = ("diamond", "staggered", "scaled", "linked", "multi", "frame")
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train" and index == 0:
                family = "diamond"
            elif split == "train" and index == 1:
                family = str(rng.choice(families[:4]))
            else:
                family = str(rng.choice(families))
            if split == "train" and index == 1:
                possible = [c for c in range(1, 10) if c not in train_answers]
            elif split == "test" and index == 0:
                possible = [c for c in range(1, 10) if c not in train_answers]
            else:
                possible = list(range(1, 10))
            if not possible:
                possible = list(range(1, 10))
            target_color = int(rng.choice(possible))
            complexity = str(rng.choice(("ordinary", "ordinary", "repeat", "large")))
            for _ in range(8):
                pair = _make_pair(rng, family, target_color,
                                  "source_train" if split == "train" else "target_test",
                                  complexity)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_answers.add(target_color)
                    break
            else:
                raise RuntimeError("could not make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    seen_inputs = set()
    seen_pairs = set()
    train_answers = set()
    simple_training = False
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for array in (grid, output):
                assert isinstance(array, np.ndarray) and array.ndim == 2
                assert 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30
                assert np.issubdtype(array.dtype, np.integer)
                assert np.all((0 <= array) & (array <= 9))
            assert output.shape == (1, 1) and grid.shape != (1, 1)
            key = (grid.shape, tuple(int(v) for v in grid.flat))
            pair_key = (key, tuple(int(v) for v in output.flat))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert _source_key(grid) not in _SOURCE_INPUTS
            values, counts = np.unique(grid, return_counts=True)
            assert 0 in values and int(values[np.argmax(counts)]) == 0
            colors = [int(c) for c in values if c != 0]
            assert 3 <= len(colors) <= 5
            enclosing = [c for c in colors if _pockets(grid, c)]
            assert enclosing == [int(output[0, 0])]
            assert len(_components(grid == enclosing[0], diagonal=True)) == 1
            if split == "train":
                train_answers.add(enclosing[0])
                simple_training |= _is_simple_diamond(grid == enclosing[0])
    assert len(train_answers) >= 2
    assert simple_training
    # Solve from visible training pairs and each test input, without metadata.
    assert all([c for c in np.unique(p["input"]) if c != 0 and
                _pockets(p["input"], int(c))] == [int(p["output"][0, 0])]
               for p in train)
    for pair in test:
        prediction = [int(c) for c in np.unique(pair["input"])
                      if c != 0 and _pockets(pair["input"], int(c))]
        assert prediction == [int(pair["output"][0, 0])]


def _same_puzzle(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[f], b[f])
                   for a, b in zip(first[s], second[s])
                   for f in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, dimensions, palettes = set(), set(), set(), set()
    signatures, small_answers = set(), 0
    edge_train = edge_test = repeated_train = repeated_test = 0
    coverage = {"train": {"small": 0, "multi_pocket": 0, "many_colors": 0,
                          "answer_smallest": 0, "large_distractor": 0},
                "test": {"small": 0, "multi_pocket": 0, "many_colors": 0,
                         "answer_smallest": 0, "large_distractor": 0}}
    unseen_test_answers = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_answers = {int(p["output"][0, 0]) for p in puzzle["train"]}
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple((p["input"].shape, p["input"].tobytes())
                             for p in puzzle["train"] + puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                answer = int(pair["output"][0, 0])
                dimensions.add(grid.shape)
                palettes.add(tuple(int(c) for c in np.unique(grid) if c != 0))
                pixels = {int(c): int(np.sum(grid == c))
                          for c in np.unique(grid) if c != 0}
                small_answers += pixels[answer] < max(pixels.values())
                coverage[split]["small"] += grid.shape == (10, 10)
                coverage[split]["multi_pocket"] += len(_pockets(grid, answer)) >= 2
                coverage[split]["many_colors"] += len(pixels) >= 4
                coverage[split]["answer_smallest"] += (
                    pixels[answer] < min(size for color, size in pixels.items()
                                         if color != answer))
                coverage[split]["large_distractor"] += any(
                    len(component) >= 16 for color in pixels if color != answer
                    for component in _components(grid == color, diagonal=True))
                if split == "test":
                    unseen_test_answers += answer not in training_answers
                ys, xs = np.nonzero(grid == answer)
                edge = (ys.min() == 0 or xs.min() == 0 or
                        ys.max() == grid.shape[0] - 1 or
                        xs.max() == grid.shape[1] - 1)
                repeated = any(len(_components(grid == c, diagonal=True)) >= 2
                               for c in pixels if c != answer)
                if split == "train":
                    edge_train += edge
                    repeated_train += repeated
                else:
                    edge_test += edge
                    repeated_test += repeated
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 30 and len(palettes) >= 20
    assert len(signatures) >= 40 and small_answers >= 10
    assert edge_train and edge_test and repeated_train and repeated_test
    assert all(all(value > 0 for value in axes.values())
               for axes in coverage.values())
    assert unseen_test_answers > 0
    assert _same_puzzle(generate_puzzle(931), generate_puzzle(931))
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                   (2, None), (10, None)):
        p = generate_puzzle(211, *counts)
        validate_generated_puzzle(p)
        if counts[0] is not None:
            assert len(p["train"]) == counts[0]
        if counts[1] is not None:
            assert len(p["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (True, None), (2, True)):
        try:
            generate_puzzle(1, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
