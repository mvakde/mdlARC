"""Modal-color stencil puzzles with a learned, puzzle-wide background."""

import numpy as np


_SOURCE_INPUTS = {
    (4, 5, 4, 2, 2, 5, 5, 5, 4),
    (7, 7, 1, 4, 7, 1, 3, 3, 7),
    (1, 2, 3, 9, 9, 1, 2, 9, 4),
    (8, 8, 1, 8, 6, 1, 4, 9, 6),
    (9, 6, 7, 8, 7, 7, 2, 8, 7),
}
_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (4, 4), (5, 5),
           (4, 5), (5, 4), (2, 5), (5, 2), (3, 4), (4, 3))
_SOURCE_ABSTRACT_MASK = np.asarray(((0, 0, 1), (0, 1, 1), (0, 0, 1)), dtype=bool)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    a = _count_value("num_train", num_train, 2, 10)
    b = _count_value("num_test", num_test, 1, 4)
    if a is not None and b is not None and a <= b:
        raise ValueError("num_train must exceed num_test")
    if a is None and b is None:
        b = 1 + seed % 4
        a = max(2, b + 1) + (seed // 4) % (11 - max(2, b + 1))
    elif a is None:
        a = b + 1 + seed % (10 - b)
    elif b is None:
        b = 1 + seed % min(4, a - 1)
    return int(a), int(b)


def _mode(grid):
    colors, counts = np.unique(grid, return_counts=True)
    top = int(counts.max())
    _require(int(np.count_nonzero(counts == top)) == 1, "mode must be unique")
    return int(colors[int(np.argmax(counts))]), top


def _fingerprint(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _construct_output(tile, background):
    h, w = tile.shape
    color, _ = _mode(tile)
    out = np.full((h * h, w * w), background, dtype=np.int8)
    for r, c in np.argwhere(tile == color):
        out[int(r) * h:(int(r) + 1) * h,
            int(c) * w:(int(c) + 1) * w] = tile
    return out


def _choose_counts(rng, n, modal_count=None, palette_size=None):
    legal_m = [m for m in range(2, n) if n - m <= 8 * (m - 1)]
    if modal_count is None:
        anchors = [legal_m[0], legal_m[0] + 1, 3, 4, 5,
                   n // 3, n // 2, n - 2, n - 1]
        anchors = [m for m in anchors if m in legal_m]
        modal_count = int(rng.choice(anchors if rng.random() < .7 else legal_m))
    if modal_count not in legal_m:
        raise ValueError("infeasible modal count")
    rest = n - modal_count
    legal_k = [k for k in range(1, min(8, rest) + 1)
               if rest <= k * (modal_count - 1)]
    if palette_size is None:
        k = int(rng.choice(legal_k))
    else:
        k = palette_size - 1
        if k not in legal_k:
            raise ValueError("infeasible palette size")
    counts = [1] * k
    for _ in range(rest - k):
        choices = [i for i, x in enumerate(counts) if x < modal_count - 1]
        counts[int(rng.choice(choices))] += 1
    rng.shuffle(counts)
    return modal_count, counts


def _sample_pair(rng, background, shape, modal_count=None,
                 palette_size=None, forbidden=None, exclude_mode=None,
                 edge_decoy=False, forced_mask=None):
    h, w = shape
    n = h * w
    if forced_mask is not None:
        _require(forced_mask.shape == shape, "forced mask has wrong dimensions")
        modal_count = int(forced_mask.sum())
    if edge_decoy:
        _require(shape == (3, 3), "edge decoy uses a 3x3 seed")
        modal_count, counts = 4, [2, 1, 1, 1]
    else:
        if modal_count is None and min(h, w) >= 4 and rng.random() < .4:
            interior_capacity = (h - 2) * (w - 2)
            interior_counts = [m for m in range(2, min(n, interior_capacity + 1))
                               if n - m <= 8 * (m - 1)]
            if interior_counts:
                modal_count = int(rng.choice(interior_counts))
        modal_count, counts = _choose_counts(rng, n, modal_count, palette_size)
    available = [x for x in range(10) if x != background and x != exclude_mode]
    modal = int(rng.choice(available))
    other_pool = [x for x in range(10) if x != background and x != modal]
    others = [int(x) for x in rng.choice(other_pool, size=len(counts), replace=False)]
    if edge_decoy:
        edge = int(rng.integers(0, 4))
        if edge == 0:
            decoy_positions = [(0, 0), (1, 0)]
        elif edge == 1:
            decoy_positions = [(0, 2), (1, 2)]
        elif edge == 2:
            decoy_positions = [(0, 0), (0, 1)]
        else:
            decoy_positions = [(2, 0), (2, 1)]
    else:
        decoy_positions = []
    banned = np.zeros(shape, dtype=bool)
    if forbidden is not None:
        bh = min(h, forbidden.shape[0])
        bw = min(w, forbidden.shape[1])
        banned[:bh, :bw] = forbidden[:bh, :bw]
    for r, c in decoy_positions:
        banned[r, c] = True
    eligible = np.flatnonzero(~banned.ravel())
    if len(eligible) < modal_count:
        raise ValueError("insufficient cells for requested disjoint stencil")
    styles = ["uniform", "uniform", "uniform"]
    if forbidden is None and not edge_decoy:
        if modal_count <= max(h, w):
            styles.extend(("line", "edge"))
        if h > 2 and w > 2 and modal_count <= (h - 2) * (w - 2):
            styles.extend(("interior", "interior", "interior"))
        styles.append("cluster")
    style = str(rng.choice(styles))
    if forced_mask is not None:
        chosen = np.flatnonzero(forced_mask.ravel())
        _require(bool(np.all(~banned.ravel()[chosen])), "forced mask overlaps forbidden cells")
    elif style == "interior":
        interior = np.zeros(shape, dtype=bool)
        interior[1:-1, 1:-1] = True
        choices = np.flatnonzero(interior.ravel())
        chosen = rng.choice(choices, size=modal_count, replace=False)
    elif style in ("line", "edge"):
        lines = []
        for r in range(h):
            if style == "line" or r in (0, h - 1):
                lines.append([r * w + c for c in range(w)])
        for c in range(w):
            if style == "line" or c in (0, w - 1):
                lines.append([r * w + c for r in range(h)])
        lines = [line for line in lines if len(line) >= modal_count]
        line = lines[int(rng.integers(0, len(lines)))]
        start = int(rng.integers(0, len(line) - modal_count + 1))
        chosen = np.asarray(line[start:start + modal_count])
    elif style == "cluster":
        cells = {int(rng.choice(eligible))}
        while len(cells) < modal_count:
            frontier = set()
            for index in cells:
                r, c = divmod(index, w)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w:
                        candidate = nr * w + nc
                        if candidate not in cells and candidate in eligible:
                            frontier.add(candidate)
            cells.add(int(rng.choice(sorted(frontier))))
        chosen = np.asarray(sorted(cells))
    else:
        chosen = rng.choice(eligible, size=modal_count, replace=False)
    grid = np.empty(shape, dtype=np.int8)
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True
    grid.ravel()[mask] = modal
    if edge_decoy:
        for r, c in decoy_positions:
            grid[r, c] = others[0]
        values = [color for color, count in zip(others[1:], counts[1:])
                  for _ in range(count)]
        empty = np.flatnonzero(~mask & ~banned.ravel())
    else:
        values = [color for color, count in zip(others, counts)
                  for _ in range(count)]
        empty = np.flatnonzero(~mask)
    rng.shuffle(values)
    grid.ravel()[empty] = values
    return {"input": grid, "output": _construct_output(grid, background)}


def _draw_unique_pair(rng, background, used, shape, **kwargs):
    for _ in range(100):
        pair = _sample_pair(rng, background, shape, **kwargs)
        inp = pair["input"]
        fp = _fingerprint(inp)
        if fp not in used and not (inp.shape == (3, 3) and tuple(int(x) for x in inp.flat) in _SOURCE_INPUTS):
            used.add(fp)
            return pair
    raise RuntimeError("could not draw a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    background = int(rng.integers(0, 10))
    regime = seed % 4  # source recombination, same-size overlap, size transfer, broad overlap
    used = set()
    train = []
    test = []
    if regime == 0:
        first_shape, second_shape = (3, 3), (3, 3)
        first_spec = {"modal_count": 4, "palette_size": int(rng.choice([3, 4]))}
        second_spec = {"modal_count": 3, "palette_size": 5}
    elif regime == 2:
        first_shape, second_shape = ((2, 3), (3, 2)) if rng.random() < .5 else ((3, 2), (2, 3))
        first_spec = {"modal_count": 2}
        second_spec = {"modal_count": 2}
    else:
        first_shape, second_shape = (3, 3), (3, 3)
        first_spec = {"modal_count": 3}
        second_spec = {"modal_count": 3}
    first = _draw_unique_pair(rng, background, used, first_shape, **first_spec)
    train.append(first)
    first_mode, _ = _mode(first["input"])
    second = _draw_unique_pair(rng, background, used, second_shape,
                               forbidden=first["input"] == first_mode,
                               exclude_mode=first_mode, **second_spec)
    train.append(second)
    for index in range(2, train_count):
        if regime in (0, 1):
            shape = (3, 3)
        else:
            shape = _SHAPES[int(rng.integers(0, len(_SHAPES)))]
        kwargs = {}
        if regime == 1 and index == 2 and rng.random() < .5:
            kwargs["forced_mask"] = _SOURCE_ABSTRACT_MASK
        train.append(_draw_unique_pair(rng, background, used, shape, **kwargs))
    training_shapes = {pair["input"].shape for pair in train}
    for index in range(test_count):
        if regime in (0, 1):
            shape = (3, 3)
        elif regime == 2 and index == 0:
            shape = _SHAPES[(seed // 4) % len(_SHAPES)]
        elif regime == 3 and len(training_shapes) == 1:
            shape = (3, 3)
        else:
            shape = _SHAPES[int(rng.integers(0, len(_SHAPES)))]
        kwargs = {}
        if regime == 0 and index == 0:
            kwargs = {"edge_decoy": True}
        elif regime == 1 and index == 0 and rng.random() < .5:
            kwargs = {"forced_mask": _SOURCE_ABSTRACT_MASK}
        test.append(_draw_unique_pair(rng, background, used, shape, **kwargs))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid, name):
    _require(isinstance(grid, np.ndarray), f"{name} must be an array")
    _require(grid.ndim == 2, f"{name} must be two dimensional")
    _require(all(1 <= d <= 30 for d in grid.shape), f"{name} dimensions outside 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must be integer")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{name} colors outside 0..9")


def _analyze_pair(pair):
    tile, out = pair["input"], pair["output"]
    h, w = tile.shape
    _require(2 <= h <= 5 and 2 <= w <= 5, "seed size must be 2..5")
    _require(out.shape == (h * h, w * w), "output dimensions must be squared seed dimensions")
    input_colors = set(int(x) for x in np.unique(tile))
    output_colors = set(int(x) for x in np.unique(out))
    _require(2 <= len(input_colors) <= 9, "tile must be multicolor")
    extra = output_colors - input_colors
    _require(len(extra) == 1 and input_colors <= output_colors,
             "output must have exactly one new background color")
    background = next(iter(extra))
    modal, count = _mode(tile)
    _require(count < tile.size, "at least one blank block is required")
    expected = np.full(out.shape, background, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if tile[r, c] == modal:
                expected[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    _require(np.array_equal(out, expected), "incorrect block substitution")
    return background, modal, count


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    used = set()
    backgrounds = set()
    train_modes = []
    for split_name, split in (("train", train), ("test", test)):
        for index, pair in enumerate(split):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            _check_grid(pair["input"], f"{split_name} {index} input")
            _check_grid(pair["output"], f"{split_name} {index} output")
            fp = _fingerprint(pair["input"])
            _require(fp not in used, "duplicate input")
            used.add(fp)
            inp = pair["input"]
            _require(not (inp.shape == (3, 3) and tuple(int(x) for x in inp.flat) in _SOURCE_INPUTS),
                     "exact source input copied")
            background, modal, _ = _analyze_pair(pair)
            backgrounds.add(background)
            if split_name == "train":
                train_modes.append(modal)
    _require(len(backgrounds) == 1, "background must be learned and puzzle-wide")
    _require(train_modes[0] != train_modes[1], "first demonstrations must change modal color")
    a, b = train[0]["input"], train[1]["input"]
    ma, _ = _mode(a)
    mb, _ = _mode(b)
    common = (a[:min(a.shape[0], b.shape[0]), :min(a.shape[1], b.shape[1])] == ma) & (
        b[:min(a.shape[0], b.shape[0]), :min(a.shape[1], b.shape[1])] == mb)
    _require(not bool(np.any(common)), "training leaves a fixed-coordinate selector")
    training_shapes = {pair["input"].shape for pair in train}
    if any(pair["input"].shape not in training_shapes for pair in test):
        _require(len(training_shapes) >= 2,
                 "new test dimensions require differently sized demonstrations")


def _puzzle_fingerprint(puzzle):
    return tuple(tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    for request in ((2, 1), (10, 1), (10, 4), (None, 1),
                    (None, 4), (2, None), (10, None), (7, 3)):
        p = generate_puzzle(2031 + len(str(request)), *request)
        validate_generated_puzzle(p)
        if request[0] is not None:
            _require(len(p["train"]) == request[0], "train count ignored")
        if request[1] is not None:
            _require(len(p["test"]) == request[1], "test count ignored")
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(10, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, "seed diversity")
    _require(_puzzle_fingerprint(generate_puzzle(17)) ==
             _puzzle_fingerprint(generate_puzzle(17)), "nondeterministic")
    shapes = set()
    modes = set()
    backgrounds = set()
    masks = set()
    palettes = set()
    counts = set()
    disconnected = 0
    edge_decoys = 0
    two_color = 0
    sparse_three = 0
    dense_three = 0
    interior_only = 0
    straight_lines = 0
    edge_lines = 0
    original_abstract_mask = 0
    novel_size_tests = 0
    right_column_counterexamples = 0
    combined_axes = 0
    train_partitions = set()
    test_partitions = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training_shapes = {pair["input"].shape for pair in puzzle["train"]}
        novel_size_tests += sum(pair["input"].shape not in training_shapes
                                for pair in puzzle["test"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                tile, out = pair["input"], pair["output"]
                bg, mode, count = _analyze_pair(pair)
                shapes.add(tile.shape)
                modes.add(mode)
                backgrounds.add(bg)
                counts.add((tile.shape, count))
                palettes.add((tile.shape, len(np.unique(tile))))
                partition = tuple(sorted((int(x) for x in np.unique(tile, return_counts=True)[1]),
                                         reverse=True))
                (train_partitions if split == "train" else test_partitions).add(partition)
                mask = tile == mode
                masks.add((tile.shape, tuple(bool(x) for x in mask.flat)))
                positions = set(map(tuple, np.argwhere(mask)))
                coords = np.argwhere(mask)
                two_color += len(np.unique(tile)) == 2
                sparse_three += tile.shape == (3, 3) and partition == (2, 1, 1, 1, 1, 1, 1, 1)
                dense_three += tile.shape == (3, 3) and partition == (7, 1, 1)
                h, w = tile.shape
                interior_only += all(0 < r < h - 1 and 0 < c < w - 1 for r, c in coords)
                row_line = len(set(coords[:, 0])) == 1
                col_line = len(set(coords[:, 1])) == 1
                straight_lines += row_line or col_line
                edge_lines += (row_line and int(coords[0, 0]) in (0, h - 1)) or (
                    col_line and int(coords[0, 1]) in (0, w - 1))
                original_abstract_mask += tile.shape == (3, 3) and np.array_equal(
                    mask, _SOURCE_ABSTRACT_MASK)
                if split == "test" and tile.shape == (3, 3):
                    right_column_counterexamples += mode not in tile[:, -1]
                seen = {next(iter(positions))}
                frontier = list(seen)
                while frontier:
                    r, c = frontier.pop()
                    for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if neighbor in positions and neighbor not in seen:
                            seen.add(neighbor)
                            frontier.append(neighbor)
                disconnected += seen != positions
                combined_axes += (h != w and seen != positions and
                                  len(np.unique(tile)) >= 5 and count * 2 <= tile.size)
                for edge in (tile[0], tile[-1], tile[:, 0], tile[:, -1]):
                    for color in np.unique(edge):
                        if color != mode and int(np.count_nonzero(edge == color)) >= 2:
                            edge_decoys += 1
                # Separate coordinate reconstruction also checks orientation and blank blocks.
                h, w = tile.shape
                for r in range(h):
                    for c in range(w):
                        block = out[r*h:(r+1)*h, c*w:(c+1)*w]
                        target = tile if tile[r, c] == mode else np.full(tile.shape, bg)
                        _require(np.array_equal(block, target), "independent block check")
    _require(len(shapes) >= 9 and (2, 2) in shapes and (5, 5) in shapes,
             "size and aspect diversity")
    _require(len(modes) >= 8 and len(backgrounds) >= 8, "color role diversity")
    _require(len(masks) >= 80 and len(palettes) >= 16 and len(counts) >= 25,
             "mask, palette, or count diversity")
    _require(disconnected >= 20 and edge_decoys >= 5,
             "disconnected masks or misleading edges missing")
    _require(two_color >= 10 and sparse_three >= 1 and dense_three >= 1,
             "sparse, dense, or two-color cases missing")
    _require(interior_only >= 1 and straight_lines >= 5 and edge_lines >= 5,
             "interior or line geometries missing")
    _require(original_abstract_mask >= 1 and right_column_counterexamples >= 2,
             "source stencil reuse or right-column counterexamples missing")
    _require(novel_size_tests >= 5 and combined_axes >= 5 and
             len(train_partitions & test_partitions) >= 5,
             "size transfer, compatible combinations, or train/test overlap missing")
