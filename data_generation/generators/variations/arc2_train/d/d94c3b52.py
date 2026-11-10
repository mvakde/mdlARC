import numpy as np


_SOURCE_KEYS = frozenset({
    ((0, 3), 186, ((0, 1), (0, 3), (2, 0), (2, 3), (3, 1), (3, 5))),
    ((0, 2), 495, ((0, 2), (0, 4), (1, 3), (2, 1), (2, 5), (3, 4))),
    ((2, 1), 367, ((0, 4), (2, 1), (2, 4))),
    ((0, 1), 282, ((0, 1), (0, 5), (2, 2), (3, 1), (3, 5))),
})


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (not _integer(value) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(test + 1, 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _code(mask):
    return sum(int(bit) << i for i, bit in enumerate(mask.flat))


def _mask(code, size):
    return np.array([(code >> i) & 1 for i in range(size * size)],
                    dtype=bool).reshape(size, size)


def _connected(mask):
    occupied = set(map(tuple, np.argwhere(mask)))
    if not occupied:
        return False
    seen = {next(iter(occupied))}
    stack = list(seen)
    while stack:
        row, col = stack.pop()
        for neighbor in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1)):
            if neighbor in occupied and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen == occupied


def _symmetric(mask):
    return any(np.array_equal(mask, transform) for transform in
               (np.rot90(mask), np.rot90(mask, 2), np.fliplr(mask), np.flipud(mask)))


def _template(rng, size, kind):
    if kind == "hollow":
        result = np.ones((size, size), dtype=bool)
        result[1:-1, 1:-1] = False
        return result
    for _ in range(200):
        density = int(rng.integers(2, size * size))
        cells = rng.choice(size * size, size=density, replace=False)
        result = np.zeros(size * size, dtype=bool)
        result[cells] = True
        result = result.reshape(size, size)
        if kind == "connected" and not _connected(result):
            continue
        if kind == "disconnected" and _connected(result):
            continue
        if kind == "sparse" and density > max(3, size):
            continue
        return result
    result = np.zeros((size, size), dtype=bool)
    if kind == "connected":
        result[0, :] = True
        result[1, 0] = True
    else:
        result[0, 0] = result[-1, -1] = True
    return result


def _geometry(rng, classic=False, size=None):
    if classic:
        return 4, 6, 3
    if size is None:
        size = 3 if rng.random() < 0.68 else 4
    legal = [(rows, cols) for rows in range(3, 7) for cols in range(4, 8)
             if rows * (size + 1) + 1 <= 30 and cols * (size + 1) + 1 <= 30]
    rows, cols = legal[int(rng.integers(len(legal)))]
    return rows, cols, size


def _bridges(matches, rows, cols):
    horizontal, vertical = set(), set()
    for row in range(rows):
        hits = [col for col in range(cols) if (row, col) in matches]
        if len(hits) >= 2:
            horizontal.update((row, col) for col in range(min(hits) + 1, max(hits))
                              if (row, col) not in matches)
    for col in range(cols):
        hits = [row for row in range(rows) if (row, col) in matches]
        if len(hits) >= 2:
            vertical.update((row, col) for row in range(min(hits) + 1, max(hits))
                            if (row, col) not in matches)
    return horizontal, vertical


def _crossing(matches, rows, cols):
    horizontal, vertical = _bridges(matches, rows, cols)
    return bool(horizontal & vertical)


def _middle_match(matches, rows, cols):
    return any(sum((row, col) in matches for col in range(cols)) >= 3
               for row in range(rows)) or any(
                   sum((row, col) in matches for row in range(rows)) >= 3
                   for col in range(cols))


def _precedence_evidence(matches, rows, cols):
    for row in range(rows):
        hits = [col for col in range(cols) if (row, col) in matches]
        if len(hits) >= 3 and len(hits) < max(hits) - min(hits) + 1:
            return True
    for col in range(cols):
        hits = [row for row in range(rows) if (row, col) in matches]
        if len(hits) >= 3 and len(hits) < max(hits) - min(hits) + 1:
            return True
    return False


def _anchors(rng, rows, cols, mode):
    if mode == "training_cross":
        assert (rows, cols) == (4, 6)
        row = int(rng.integers(1, 3))
        possibilities = [(left, col, right) for left in range(cols)
                         for col in range(left + 1, cols)
                         for right in range(col + 1, cols)
                         if right - left >= 3]
        left, col, right = possibilities[int(rng.integers(len(possibilities)))]
        matches = {(row, left), (row, right), (0, col), (3, col)}
        forbidden = {(row, c) for c in range(left + 1, right)}
        forbidden.update((r, col) for r in range(1, 3))
    elif mode == "crossing":
        row = int(rng.integers(1, rows - 1))
        col = int(rng.integers(1, cols - 1))
        left = int(rng.integers(0, col))
        right = int(rng.integers(col + 1, cols))
        top = int(rng.integers(0, row))
        bottom = int(rng.integers(row + 1, rows))
        matches = {(row, left), (row, right), (top, col), (bottom, col)}
        forbidden = {(row, col)}
    elif mode == "precedence":
        row = int(rng.integers(rows))
        choices = [(a, b, c) for a in range(cols) for b in range(a + 1, cols)
                   for c in range(b + 1, cols) if c - a >= 3]
        columns = choices[int(rng.integers(len(choices)))]
        matches = {(row, col) for col in columns}
        forbidden = {(row, col) for col in range(columns[0] + 1, columns[-1])
                     if col != columns[1]}
    elif mode == "row":
        row = int(rng.integers(rows))
        count = int(rng.integers(2, min(cols, 5) + 1))
        matches = {(row, int(col)) for col in rng.choice(cols, size=count, replace=False)}
        forbidden = set()
    elif mode == "column":
        col = int(rng.integers(cols))
        count = int(rng.integers(2, min(rows, 5) + 1))
        matches = {(int(row), col) for row in rng.choice(rows, size=count, replace=False)}
        forbidden = set()
    elif mode == "rectangle":
        top, bottom = sorted(int(x) for x in rng.choice(rows, size=2, replace=False))
        left, right = sorted(int(x) for x in rng.choice(cols, size=2, replace=False))
        matches = {(top, left), (top, right), (bottom, left), (bottom, right)}
        forbidden = set()
    elif mode == "ell":
        row, col = int(rng.integers(rows)), int(rng.integers(cols))
        other_row = int(rng.choice([x for x in range(rows) if x != row]))
        other_col = int(rng.choice([x for x in range(cols) if x != col]))
        matches = {(row, col), (row, other_col), (other_row, col)}
        forbidden = set()
    elif mode == "two_runs":
        row1, row2 = sorted(int(x) for x in rng.choice(rows, size=2, replace=False))
        columns = rng.choice(cols, size=4, replace=False)
        matches = {(row1, int(columns[0])), (row1, int(columns[1])),
                   (row2, int(columns[2])), (row2, int(columns[3]))}
        forbidden = set()
    else:
        count = int(rng.integers(2, min(9, rows * cols - 6) + 1))
        indices = rng.choice(rows * cols, size=count, replace=False)
        matches = {(int(x) // cols, int(x) % cols) for x in indices}
        forbidden = set()
    if mode in ("training_cross", "crossing", "precedence", "rectangle", "ell", "two_runs"):
        extras = 0 if mode == "training_cross" else int(rng.integers(0, 3 if mode == "precedence" else 4))
        available = [(r, c) for r in range(rows) for c in range(cols)
                     if (r, c) not in matches and (r, c) not in forbidden]
        for index in rng.choice(len(available), size=min(extras, len(available)), replace=False):
            matches.add(available[int(index)])
    if mode in ("row", "column") and rng.random() < 0.4:
        if mode == "row":
            row = next(iter(matches))[0]
            available = [(r, c) for r in range(rows) for c in range(cols)
                         if r != row and all(c != hit[1] for hit in matches)]
        else:
            col = next(iter(matches))[1]
            available = [(r, c) for r in range(rows) for c in range(cols)
                         if c != col and all(r != hit[0] for hit in matches)]
        if available:
            matches.add(available[int(rng.integers(len(available)))])
    return matches


def _lookalike_codes(template, rng, count):
    size = template.shape[0]
    reference = _code(template)
    maximum = (1 << (size * size)) - 1
    chosen = []
    categories = (
        [_code(x) for x in (np.rot90(template), np.rot90(template, 2),
                            np.rot90(template, 3))],
        [_code(x) for x in (np.fliplr(template), np.flipud(template),
                            template.T, np.fliplr(template.T))],
    )
    near = [reference ^ (1 << i) for i in range(size * size)
            if reference ^ (1 << i) not in (0, maximum, reference)]
    weight = reference.bit_count()
    equal = [value for value in range(1, maximum)
             if value.bit_count() == weight and value != reference]
    for group in (categories[0], categories[1], near, equal):
        options = [value for value in group
                   if value not in (0, maximum, reference) and value not in chosen]
        if options and len(chosen) < count:
            chosen.append(options[int(rng.integers(len(options)))])
    while len(chosen) < count:
        value = int(rng.integers(1, maximum))
        if value != reference and value not in chosen:
            chosen.append(value)
    rng.shuffle(chosen)
    return chosen


def _panel_masks(rng, rows, cols, template, matches):
    size = template.shape[0]
    masks = np.empty((rows, cols, size, size), dtype=bool)
    for position in matches:
        masks[position] = template
    positions = [(r, c) for r in range(rows) for c in range(cols)
                 if (r, c) not in matches]
    n = len(positions)
    maximum = min(n // 2, 5)
    vocabulary_size = int(rng.integers(min(3, maximum), maximum + 1))
    codes = _lookalike_codes(template, rng, vocabulary_size)
    counts = np.full(vocabulary_size, 2, dtype=int)
    if rng.random() < 0.5:
        dominant = int(rng.integers(vocabulary_size))
        counts[dominant] += max(0, min(n - int(counts.sum()), len(matches) - 1))
    for _ in range(n - int(counts.sum())):
        counts[int(rng.integers(vocabulary_size))] += 1
    assigned = [code for code, count in zip(codes, counts) for _ in range(int(count))]
    rng.shuffle(assigned)
    for position, code in zip(positions, assigned):
        masks[position] = _mask(code, size)
    return masks


def _palette(rng, bridge, ordinary_count, previous=None, simple=False):
    for _ in range(30):
        available = [color for color in range(10) if color != bridge]
        background = int(rng.choice(available))
        available.remove(background)
        query = int(rng.choice(available))
        available.remove(query)
        n_colors = 1 if simple else int(rng.integers(1, min(3, ordinary_count // 2) + 1))
        ordinary = [int(x) for x in rng.choice(available, size=n_colors, replace=False)]
        if previous is None or (background, query) != previous:
            return background, query, ordinary
    raise RuntimeError("could not sample pair palette")


def _panel_slice(row, col, size):
    step = size + 1
    return (slice(1 + row * step, 1 + row * step + size),
            slice(1 + col * step, 1 + col * step + size))


def _render(rng, masks, query_position, palette):
    rows, cols, size, _ = masks.shape
    background, query, ordinary = palette
    grid = np.full((rows * (size + 1) + 1, cols * (size + 1) + 1),
                   background, dtype=np.int8)
    positions = [(r, c) for r in range(rows) for c in range(cols)
                 if (r, c) != query_position]
    colors = [color for color in ordinary for _ in range(2)]
    colors.extend(int(rng.choice(ordinary)) for _ in range(len(positions) - len(colors)))
    rng.shuffle(colors)
    for position, color in zip(positions, colors):
        panel = grid[_panel_slice(*position, size)]
        panel[masks[position]] = color
    panel = grid[_panel_slice(*query_position, size)]
    panel[masks[query_position]] = query
    return grid


def _analyze(grid):
    background = int(grid[0, 0])
    rows_sep = np.flatnonzero(np.all(grid == background, axis=1))
    cols_sep = np.flatnonzero(np.all(grid == background, axis=0))
    assert rows_sep[0] == cols_sep[0] == 0
    assert rows_sep[-1] == grid.shape[0] - 1
    assert cols_sep[-1] == grid.shape[1] - 1
    row_gaps, col_gaps = np.diff(rows_sep), np.diff(cols_sep)
    assert len(set(row_gaps)) == len(set(col_gaps)) == 1
    size = int(row_gaps[0]) - 1
    assert size in (3, 4) and np.all(col_gaps == size + 1)
    rows, cols = len(row_gaps), len(col_gaps)
    assert 3 <= rows <= 6 and 4 <= cols <= 7
    masks = np.empty((rows, cols, size, size), dtype=bool)
    panel_colors = {}
    counts = {}
    for row in range(rows):
        for col in range(cols):
            panel = grid[_panel_slice(row, col, size)]
            occupied = panel != background
            assert np.any(occupied)
            colors = np.unique(panel[occupied])
            assert len(colors) == 1
            color = int(colors[0])
            masks[row, col] = occupied
            panel_colors[row, col] = color
            counts[color] = counts.get(color, 0) + 1
    query_colors = [color for color, count in counts.items() if count == 1]
    assert len(query_colors) == 1
    query = query_colors[0]
    assert all(count >= 2 for color, count in counts.items() if color != query)
    query_position = next(pos for pos, color in panel_colors.items() if color == query)
    template = masks[query_position]
    matches = {(row, col) for row in range(rows) for col in range(cols)
               if np.array_equal(masks[row, col], template)}
    frequencies = {}
    for row in range(rows):
        for col in range(cols):
            code = _code(masks[row, col])
            frequencies[code] = frequencies.get(code, 0) + 1
    return dict(rows=rows, cols=cols, size=size, background=background,
                query=query, query_position=query_position, masks=masks,
                matches=matches, frequencies=frequencies, panel_colors=panel_colors)


def _solve(grid, bridge, analysis=None):
    if analysis is None:
        analysis = _analyze(grid)
    result = grid.copy()
    matches = analysis["matches"]
    size = analysis["size"]
    horizontal, vertical = _bridges(matches, analysis["rows"], analysis["cols"])
    for position in matches:
        panel = result[_panel_slice(*position, size)]
        panel[analysis["masks"][position]] = analysis["query"]
    for position in horizontal | vertical:
        panel = result[_panel_slice(*position, size)]
        panel[analysis["masks"][position]] = bridge
    return result


def _source_key(analysis):
    return (analysis["query_position"], _code(analysis["masks"][analysis["query_position"]]),
            tuple(sorted(analysis["matches"])))


def _pair(rng, geometry, mode, bridge, previous_palette=None, template=None,
          simple=False):
    rows, cols, size = geometry
    for _ in range(100):
        if template is None:
            kind = ("connected", "disconnected", "hollow", "sparse", "random")[
                int(rng.integers(5))]
            query_mask = _template(rng, size, kind)
        else:
            query_mask = template
        matches = _anchors(rng, rows, cols, mode)
        if len(matches) > rows * cols - 6:
            continue
        if mode in ("training_cross", "crossing") and not _crossing(matches, rows, cols):
            continue
        if mode == "precedence" and not _precedence_evidence(matches, rows, cols):
            continue
        masks = _panel_masks(rng, rows, cols, query_mask, matches)
        query_position = sorted(matches)[int(rng.integers(len(matches)))]
        palette = _palette(rng, bridge, rows * cols - 1, previous_palette, simple)
        inp = _render(rng, masks, query_position, palette)
        try:
            analysis = _analyze(inp)
        except AssertionError:
            continue
        if geometry == (4, 6, 3) and _source_key(analysis) in _SOURCE_KEYS:
            continue
        out = _solve(inp, bridge, analysis)
        if not np.array_equal(inp, out) and (mode not in ("training_cross", "crossing", "precedence") or
                                             bridge in out):
            return {"input": inp, "output": out}, analysis
    if template is not None:
        return _pair(rng, geometry, mode, bridge, previous_palette, None, simple)
    raise RuntimeError(f"could not construct pair: geometry={geometry}, mode={mode}")


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= length <= 30 for length in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, analyses = set(), set(), {"train": [], "test": []}
    bridge = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            analysis = _analyze(inp)
            analyses[split].append(analysis)
            assert _source_key(analysis) not in _SOURCE_KEYS or (analysis["rows"], analysis["cols"], analysis["size"]) != (4, 6, 3)
            input_colors = set(map(int, np.unique(inp)))
            output_colors = set(map(int, np.unique(out)))
            new = output_colors - input_colors
            assert len(new) <= 1
            if split == "train":
                assert len(new) == 1
                learned = next(iter(new))
                if bridge is None:
                    bridge = learned
                else:
                    assert learned == bridge
            elif new:
                assert new == {bridge}
            assert bridge is not None and bridge not in input_colors
            assert np.array_equal(out, _solve(inp, bridge, analysis))
            frequencies = analysis["frequencies"]
            assert 3 <= len(frequencies) <= 7
            code = _code(analysis["masks"][analysis["query_position"]])
            assert sum(count >= 2 for value, count in frequencies.items() if value != code) >= 2
    first = analyses["train"][0]
    first_h, first_v = _bridges(first["matches"], first["rows"], first["cols"])
    assert first_h & first_v and first_h - first_v and first_v - first_h
    assert (first["rows"], first["cols"], first["size"]) == (4, 6, 3)
    assert len(set(first["panel_colors"].values())) == 2
    assert any(_precedence_evidence(a["matches"], a["rows"], a["cols"])
               for a in analyses["train"])
    assert len({(a["background"], a["query"]) for a in analyses["train"]}) >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _counts(seed_value, rng, num_train, num_test)
    bridge = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    seen, training_masks, training_sizes = set(), [], []
    previous = None
    modes = ("row", "column", "rectangle", "ell", "two_runs", "random", "crossing")
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        if index == 0:
            geometry, mode = (4, 6, 3), "training_cross"
        elif index == 1:
            geometry, mode = _geometry(rng), "precedence"
        elif split == "train":
            geometry = _geometry(rng)
            mode = modes[int(rng.integers(len(modes)))]
        else:
            size = training_sizes[int(rng.integers(len(training_sizes)))]
            geometry = _geometry(rng, size=size)
            mode = modes[int(rng.integers(len(modes)))]
        for _ in range(30):
            reused = None
            if split == "test" and rng.random() < 0.32:
                candidates = [mask for mask in training_masks if mask.shape[0] == geometry[2]]
                if candidates:
                    reused = candidates[int(rng.integers(len(candidates)))]
            child_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
            try:
                pair, analysis = _pair(child_rng, geometry, mode, bridge, previous,
                                       reused, index == 0)
            except RuntimeError:
                continue
            if split == "train" and bridge not in pair["output"]:
                continue
            key = _grid_key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                previous = (analysis["background"], analysis["query"])
                if split == "train":
                    training_masks.append(analysis["masks"][analysis["query_position"]].copy())
                    training_sizes.append(analysis["size"])
                break
        else:
            raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(91723, 6, 2)) == _puzzle_key(generate_puzzle(91723, 6, 2))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ((1, None), (11, None), (None, 0), (None, 5),
                      (2, 2), (3, 4), (True, 1), (3, 1.0)):
        try:
            generate_puzzle(4000, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    shapes, lattices, templates, backgrounds, bridges = set(), set(), set(), set(), set()
    topologies, glyph_counts, match_counts, interval_lengths = set(), set(), set(), set()
    frequencies = {"train": set(), "test": set()}
    morphologies = {"train": set(), "test": set()}
    axis_cases = {"train": set(), "test": set()}
    lookalikes = {"train": [0, 0, 0, 0], "test": [0, 0, 0, 0]}
    crossing, isolated, multicolor, repeats, combinations = 0, 0, 0, 0, 0
    for puzzle in puzzles:
        trained = set()
        for pair in puzzle["train"]:
            a = _analyze(pair["input"])
            trained.add((a["size"], _code(a["masks"][a["query_position"]])))
        bridge = next(iter(set(map(int, np.unique(puzzle["train"][0]["output"]))) -
                           set(map(int, np.unique(puzzle["train"][0]["input"])))))
        bridges.add(bridge)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                a = _analyze(inp)
                h, v = _bridges(a["matches"], a["rows"], a["cols"])
                code = _code(a["masks"][a["query_position"]])
                shapes.add(inp.shape)
                lattices.add((a["rows"], a["cols"], a["size"]))
                templates.add((a["size"], code))
                backgrounds.add(a["background"])
                topologies.add((bool(h), bool(v), bool(h & v), len(a["matches"])))
                glyph_counts.add(len(a["frequencies"]))
                match_counts.add(len(a["matches"]))
                for row in range(a["rows"]):
                    hits = [col for col in range(a["cols"]) if (row, col) in a["matches"]]
                    if len(hits) >= 2:
                        interval_lengths.add(max(hits) - min(hits))
                for col in range(a["cols"]):
                    hits = [row for row in range(a["rows"]) if (row, col) in a["matches"]]
                    if len(hits) >= 2:
                        interval_lengths.add(max(hits) - min(hits))
                axis_cases[split].add((bool(h), bool(v)))
                crossing += bool(h & v)
                isolated += any(not any(other != pos and
                                        (other[0] == pos[0] or other[1] == pos[1])
                                        for other in a["matches"]) for pos in a["matches"])
                ordinary_count = len(set(a["panel_colors"].values())) - 1
                multicolor += ordinary_count > 1
                combinations += bool(h & v) and not _connected(a["masks"][a["query_position"]]) and ordinary_count > 1
                repeats += split == "test" and (a["size"], code) in trained
                maximum_other = max(value for key, value in a["frequencies"].items()
                                    if key != code)
                frequencies[split].add((a["frequencies"][code] > maximum_other) -
                                       (a["frequencies"][code] < maximum_other))
                query_mask = a["masks"][a["query_position"]]
                connected = _connected(query_mask)
                hollow = (np.all(query_mask[0]) and np.all(query_mask[-1]) and
                          np.all(query_mask[:, 0]) and np.all(query_mask[:, -1]) and
                          not np.all(query_mask[1:-1, 1:-1]))
                morphologies[split].update((
                    "connected" if connected else "disconnected",
                    "symmetric" if _symmetric(query_mask) else "asymmetric",
                    "hollow" if hollow else "nonhollow",
                    "sparse" if np.count_nonzero(query_mask) <= max(3, a["size"]) else "dense",
                ))
                other_masks = [a["masks"][row, col] for row in range(a["rows"])
                               for col in range(a["cols"]) if (row, col) not in a["matches"]]
                lookalikes[split][0] += any(np.count_nonzero(mask != query_mask) == 1
                                            for mask in other_masks)
                lookalikes[split][1] += any(np.count_nonzero(mask) == np.count_nonzero(query_mask)
                                            for mask in other_masks)
                lookalikes[split][2] += any(
                    any(np.array_equal(mask, np.rot90(query_mask, k)) and
                        not np.array_equal(mask, query_mask) for k in (1, 2, 3))
                    for mask in other_masks)
                lookalikes[split][3] += any(
                    any(np.array_equal(mask, reflection) and
                        not np.array_equal(mask, query_mask)
                        for reflection in (np.fliplr(query_mask), np.flipud(query_mask),
                                           query_mask.T, np.fliplr(query_mask.T)))
                    for mask in other_masks)
                # Independently inspect each occupied panel and each empty cell.
                assert np.array_equal(inp == a["background"], out == a["background"])
                for row in range(a["rows"]):
                    for col in range(a["cols"]):
                        pos = (row, col)
                        panel = out[_panel_slice(row, col, a["size"])]
                        expected = (a["query"] if pos in a["matches"] else
                                    bridge if pos in h | v else a["panel_colors"][pos])
                        assert np.all(panel[a["masks"][pos]] == expected)
    assert len(shapes) >= 8 and len(lattices) >= 10 and len(templates) >= 50
    assert {3, 4} <= {size for _, _, size in lattices}
    assert 0 in backgrounds and len(backgrounds) >= 8 and len(bridges) >= 7
    assert len(topologies) >= 12 and len(match_counts) >= 5
    assert {1, 2, 3, 4} <= interval_lengths
    assert crossing >= 40 and isolated >= 10 and multicolor >= 20 and repeats >= 1
    assert combinations >= 1 and len(glyph_counts) >= 3
    assert all(frequencies[split] == {-1, 0, 1} for split in ("train", "test"))
    assert morphologies["train"] == morphologies["test"] == {
        "connected", "disconnected", "symmetric", "asymmetric",
        "hollow", "nonhollow", "sparse", "dense"}
    assert {(True, True), (True, False), (False, True)} <= axis_cases["train"]
    assert {(True, True), (True, False), (False, True), (False, False)} <= axis_cases["test"]
    assert all(lookalikes[split][0] >= 30 and lookalikes[split][1] >= 30 and
               lookalikes[split][2] >= 10 and lookalikes[split][3] >= 10
               for split in ("train", "test"))
