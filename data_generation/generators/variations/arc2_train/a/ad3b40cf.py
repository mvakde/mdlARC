import numpy as np


# Exact source pairs, represented as rows, are rejected by construction
# and independently by validation. Their colors and dimensions remain legal.
_SOURCE_PAIR_ROWS = frozenset(
    (
        (
            (
                "888818888",
                "888818558",
                "888818558",
                "888818888",
                "888818288",
                "888818288",
                "888818288",
                "888818228",
                "888818888",
            ),
            (
                "888818888",
                "855818558",
                "855818558",
                "888818888",
                "888818288",
                "888818288",
                "888818288",
                "888818228",
                "888818888",
            ),
        ),
        (
            (
                "8888888888888881",
                "8778888877788818",
                "8778868887888188",
                "8778866888881888",
                "8888888888818888",
                "8888888888188888",
                "8888878881888888",
                "8887778818888888",
                "8888888188888888",
                "8668881888888888",
                "8668818888888888",
                "8888188888888888",
                "8881888888888888",
                "8818888888888888",
                "8188888888888888",
                "1888888888888888",
            ),
            (
                "8888888888888881",
                "8778888877788818",
                "8778868887888188",
                "8778866888881888",
                "8888888888818888",
                "8888888888188888",
                "8888878881888888",
                "8887778818888888",
                "8888888188888888",
                "8668881888886888",
                "8668818888886688",
                "8888188888888888",
                "8881888888888888",
                "8818866888888888",
                "8188866888888888",
                "1888888888888888",
            ),
        ),
        (
            (
                "888888888888888",
                "882288888888888",
                "882288888866688",
                "888888888866688",
                "868888228866688",
                "866888828888888",
                "888888888888888",
                "111111111111111",
                "888888888888888",
                "888888888888888",
                "888888888888888",
                "888888888888888",
                "888888888888888",
                "888888888888888",
                "888888888888888",
            ),
            (
                "888888888888888",
                "882288888888888",
                "882288888866688",
                "888888888866688",
                "868888228866688",
                "866888828888888",
                "888888888888888",
                "111111111111111",
                "888888888888888",
                "888888828888888",
                "888888228888888",
                "888888888888888",
                "882288888888888",
                "882288888888888",
                "888888888888888",
            ),
        ),
        (
            (
                "18888888888888888",
                "81888888888888888",
                "88188888888888888",
                "88818888888888888",
                "88881888888888888",
                "84448188888888888",
                "88488818888888888",
                "88488881888888888",
                "88888888188888888",
                "87788888818888888",
                "87888888881888888",
                "88887888888188888",
                "88887884448818888",
                "84487888848881888",
                "84487788848888188",
                "84487788848778818",
                "88888888888888881",
            ),
            (
                "18888888888888888",
                "81888888877888888",
                "88188888878888888",
                "88818888888888888",
                "88881888888777778",
                "84448188888888778",
                "88488818888888888",
                "88488881888888888",
                "88888888188888888",
                "87788888818888888",
                "87888888881888888",
                "88887888888188878",
                "88887884448818878",
                "84487888848881888",
                "84487788848888188",
                "84487788848778818",
                "88888888888888881",
            ),
        ),
    )
)


def _validate_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _choose_counts(rng, num_train, num_test):
    num_train = _validate_count("num_train", num_train, 2, 10)
    num_test = _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _sample_conventions(rng):
    # These are visual distribution choices. The only learned convention is
    # the same cell-count-and-reflection rule in every pair.
    base = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    mode = str(rng.choice(("stable", "payload_vary", "all_vary"), p=(.28, .38, .34)))
    return {"base": base, "mode": mode, "density": float(rng.uniform(.17, .30))}


def _axis_cells(shape, orientation, axis_position):
    h, w = shape
    if orientation == "horizontal":
        return {(axis_position, c) for c in range(w)}
    if orientation == "vertical":
        return {(r, axis_position) for r in range(h)}
    if h != w:
        raise ValueError("diagonal axis needs a square grid")
    if orientation == "main":
        return {(i, i) for i in range(h)}
    if orientation == "anti":
        return {(i, h - 1 - i) for i in range(h)}
    raise ValueError("unknown orientation")


def _side_value(r, c, shape, orientation, axis_position):
    if orientation == "horizontal":
        return r - axis_position
    if orientation == "vertical":
        return c - axis_position
    if orientation == "main":
        return r - c
    return r + c - (shape[0] - 1)


def _reflect(r, c, shape, orientation, axis_position):
    if orientation == "horizontal":
        return 2 * axis_position - r, c
    if orientation == "vertical":
        return r, 2 * axis_position - c
    if orientation == "main":
        return c, r
    return shape[0] - 1 - c, shape[1] - 1 - r


def _neighbors(cell, shape):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            yield rr, cc


def _sample_shape_axis(rng, regime, orientation, source_like, allow_offcenter):
    if source_like:
        n = int(rng.integers(17, 20))
        return (n, n), n // 2
    if orientation in ("main", "anti"):
        if regime == "target_test" and rng.random() < .6:
            n = int(rng.integers(15, 27))
        else:
            n = int(rng.integers(9, 23))
        return (n, n), n // 2
    if regime == "target_test" and rng.random() < .55:
        dim = int(rng.integers(15, 27))
    else:
        dim = int(rng.integers(9, 23))
    cross = int(rng.integers(9, 25))
    shape = (dim, cross) if orientation == "horizontal" else (cross, dim)
    choices = [p for p in range(4, dim - 4) if min(p, dim - 1 - p) >= 4]
    if not allow_offcenter or rng.random() < .56:
        position = (dim - 1) // 2
    else:
        position = int(rng.choice(choices))
    return shape, position


def _allowed_cells(shape, orientation, axis_position, side):
    cells = set()
    for r in range(shape[0]):
        for c in range(shape[1]):
            if side * _side_value(r, c, shape, orientation, axis_position) <= 0:
                continue
            rr, cc = _reflect(r, c, shape, orientation, axis_position)
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                cells.add((r, c))
    return cells


def _partition_sizes(rng, total, parts, uneven=False):
    if parts == 1:
        return [total]
    if uneven and total >= parts + 4:
        sizes = [1] * (parts - 1) + [total - parts + 1]
        rng.shuffle(sizes)
        return sizes
    weights = rng.dirichlet(np.full(parts, 1.35))
    counts = rng.multinomial(total - parts, weights)
    sizes = [int(x) + 1 for x in counts]
    rng.shuffle(sizes)
    return sizes


def _component_shape(rng, size):
    style = str(rng.choice(("horizontal", "vertical", "rectangle", "ell", "organic")))
    if style == "horizontal":
        cells = {(0, j) for j in range(size)}
    elif style == "vertical":
        cells = {(i, 0) for i in range(size)}
    elif style == "ell" and size >= 3:
        arm = int(rng.integers(2, size))
        cells = {(0, j) for j in range(arm)} | {(i, 0) for i in range(size - arm + 1)}
    elif style == "rectangle" and size >= 4:
        width = int(rng.integers(2, min(size, 6) + 1))
        cells = {(i // width, i % width) for i in range(size)}
    else:
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _place_shapes(rng, allowed, occupied, prior_same, sizes, shape):
    placed = set()
    for size in sorted(sizes, reverse=True):
        successful = None
        for _ in range(12):
            motif = _component_shape(rng, size)
            mh = max(r for r, _ in motif) + 1
            mw = max(c for _, c in motif) + 1
            origins = [(r, c) for r in range(shape[0] - mh + 1)
                       for c in range(shape[1] - mw + 1)]
            rng.shuffle(origins)
            for r0, c0 in origins:
                cells = {(r0 + r, c0 + c) for r, c in motif}
                if not cells <= allowed or cells & (occupied | placed):
                    continue
                if any(n in prior_same or n in placed
                       for cell in cells for n in _neighbors(cell, shape) if n not in cells):
                    continue
                successful = cells
                break
            if successful is not None:
                break
        if successful is None:
            return None
        placed.update(successful)
    return placed


def _pair_colors(rng, conventions, index, num_train, training_payloads):
    bg, axis, a, b = conventions["base"]
    if index < 2:
        return bg, axis, (a, b), (a if index == 0 else b)
    mode = conventions["mode"]
    if mode == "all_vary" and rng.random() < .65:
        bg, axis = (int(x) for x in rng.choice(10, size=2, replace=False))
    remaining = [c for c in range(10) if c not in (bg, axis)]
    if mode == "stable":
        if a in remaining and b in remaining:
            palette = (a, b)
        else:
            palette = tuple(int(x) for x in rng.choice(remaining, 2, replace=False))
    else:
        if index >= num_train and rng.random() < .55:
            unseen = [c for c in remaining if c not in training_payloads]
            fresh = int(rng.choice(unseen or remaining))
            other = int(rng.choice([c for c in remaining if c != fresh]))
            palette = (fresh, other)
        elif rng.random() < .55:
            anchor = int(rng.choice([c for c in (a, b) if c in remaining] or remaining))
            other = int(rng.choice([c for c in remaining if c != anchor]))
            palette = (anchor, other)
        else:
            palette = tuple(int(x) for x in rng.choice(remaining, 2, replace=False))
    minority = int(rng.choice(palette))
    return bg, axis, palette, minority


def _make_pair(rng, regime, orientation, colors, density, special, allow_offcenter):
    background, axis_color, palette, minority = colors
    majority = palette[1] if minority == palette[0] else palette[0]
    source_like = special == "source_like"
    shape, axis_position = _sample_shape_axis(rng, regime, orientation, source_like,
                                             allow_offcenter)
    side = -1 if rng.random() < .5 else 1
    allowed = _allowed_cells(shape, orientation, axis_position, side)
    capacity = len(allowed)
    if capacity < 20:
        raise RuntimeError("too few usable cells")
    # Dense cases reach source-like occupancy; sparse cases retain small motifs.
    target_total = min(int(capacity * float(rng.uniform(.12, .39)) * density / .23),
                       int(capacity * .42))
    target_total = max(10 if special == "wide_counterexample" else 7, target_total)
    if special in ("close", "count_reversal"):
        gap = 1
    elif special == "wide_counterexample":
        gap = int(rng.integers(6, 11))
    else:
        gap = int(rng.choice((1, 2, 3, 4, 5, 7, 9, 11)))
    gap = min(gap, max(1, target_total - 4))
    minority_total = max(3, (target_total - gap) // 2)
    majority_total = minority_total + gap
    if special == "count_reversal":
        minority_components = int(rng.integers(2, min(5, minority_total) + 1))
        majority_components = 1
    elif special == "wide_counterexample":
        minority_components = 1
        majority_components = 2
    elif source_like:
        minority_components = 3
        majority_components = int(rng.integers(1, 4))
    else:
        minority_components = int(rng.integers(1, min(5, minority_total) + 1))
        majority_components = int(rng.integers(1, min(5, majority_total) + 1))
    if regime == "target_test" and not source_like and rng.random() < .3:
        minority_components = int(rng.choice((1, 2, 4, 5)))
        minority_components = min(minority_components, minority_total)
    minority_sizes = _partition_sizes(rng, minority_total, minority_components,
                                      uneven=(rng.random() < .5))
    majority_sizes = _partition_sizes(rng, majority_total, majority_components,
                                      uneven=(special == "wide_counterexample" or rng.random() < .5))
    axis = _axis_cells(shape, orientation, axis_position)
    for _ in range(32):
        occupied = set(axis)
        minority_cells = _place_shapes(rng, allowed, occupied, set(), minority_sizes, shape)
        if minority_cells is None:
            continue
        occupied.update(minority_cells)
        majority_cells = _place_shapes(rng, allowed, occupied, set(), majority_sizes, shape)
        if majority_cells is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        for cell in axis:
            grid[cell] = axis_color
        for cell in minority_cells:
            grid[cell] = minority
        for cell in majority_cells:
            grid[cell] = majority
        full_lines = sum(int(grid[r, 0] != background and np.all(grid[r, :] == grid[r, 0]))
                         for r in range(shape[0]))
        full_lines += sum(int(grid[0, c] != background and np.all(grid[:, c] == grid[0, c]))
                          for c in range(shape[1]))
        if shape[0] == shape[1]:
            ids = np.arange(shape[0])
            for cells in (grid[ids, ids], grid[ids, shape[0] - 1 - ids]):
                full_lines += int(cells[0] != background and np.all(cells == cells[0]))
        if full_lines != 1:
            continue
        output = grid.copy()
        for r, c in minority_cells:
            output[_reflect(r, c, shape, orientation, axis_position)] = minority
        return {"input": grid, "output": output}
    raise RuntimeError("layout construction failed")


def _grid_rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _signature(pair):
    return (pair["input"].shape, pair["input"].tobytes(),
            pair["output"].shape, pair["output"].tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    conventions = _sample_conventions(rng)
    orientations = ("horizontal", "vertical", "main", "anti")
    axis_mode = str(rng.choice(("source_shift", "shared", "mixed"), p=(.18, .18, .64)))
    if axis_mode == "source_shift":
        classic = ("horizontal", "vertical", "anti")
        train_axes = ["anti", str(rng.choice(("horizontal", "vertical")))]
        train_axes.extend(str(rng.choice(classic)) for _ in range(num_train - 2))
        rng.shuffle(train_axes)
        test_axes = ["main"]
        test_axes.extend(str(rng.choice(orientations)) for _ in range(num_test - 1))
    elif axis_mode == "shared":
        common = str(rng.choice(orientations))
        train_axes = [common] * num_train
        test_axes = [common] * num_test
    else:
        train_axes = [str(rng.choice(orientations))]
        train_axes.append(str(rng.choice([o for o in orientations if o != train_axes[0]])))
        train_axes.extend(str(rng.choice(orientations)) for _ in range(num_train - 2))
        test_axes = [str(rng.choice(train_axes if rng.random() < .48 else orientations))
                     for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    training_payloads = set()
    demonstrated_offset = False
    for index in range(num_train + num_test):
        regime = "source_train" if index < num_train else "target_test"
        orientation = (train_axes + test_axes)[index]
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        colors = _pair_colors(pair_rng, conventions, index, num_train, training_payloads)
        if index == 0:
            special = "count_reversal"
        elif index == 1:
            special = "wide_counterexample"
        elif regime == "target_test" and orientation == "main" and (axis_mode == "source_shift" or pair_rng.random() < .22):
            special = "source_like"
        else:
            special = "close" if pair_rng.random() < .27 else "ordinary"
        for _ in range(12):
            try:
                pair = _make_pair(pair_rng, regime, orientation, colors,
                                  conventions["density"], special,
                                  index < num_train or demonstrated_offset)
            except RuntimeError:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key not in seen and (_grid_rows(pair["input"]), _grid_rows(pair["output"])) not in _SOURCE_PAIR_ROWS:
                seen.add(key)
                puzzle["train" if index < num_train else "test"].append(pair)
                if index < num_train:
                    training_payloads.update(colors[2])
                    info = _analyze_pair(pair)
                    if info["orientation"] in ("horizontal", "vertical"):
                        axis_dimension = info["shape"][0 if info["orientation"] == "horizontal" else 1]
                        demonstrated_offset |= info["position"] not in (
                            (axis_dimension - 1) // 2, axis_dimension // 2)
                break
        else:
            raise RuntimeError(f"could not generate pair {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    parts = []
    while remaining:
        start = remaining.pop()
        part = {start}
        stack = [start]
        while stack:
            for neighbor in _neighbors(stack.pop(), grid.shape):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    part.add(neighbor)
                    stack.append(neighbor)
        parts.append(part)
    return parts


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 4
    order = np.argsort(counts)
    assert counts[order[-1]] > counts[order[-2]]
    bg = int(values[order[-1]])
    h, w = inp.shape
    candidates = []
    for r in range(h):
        color = int(inp[r, 0])
        if color != bg and np.all(inp[r, :] == color):
            candidates.append(("horizontal", r, color))
    for c in range(w):
        color = int(inp[0, c])
        if color != bg and np.all(inp[:, c] == color):
            candidates.append(("vertical", c, color))
    if h == w:
        ids = np.arange(h)
        for orient, cells in (("main", inp[ids, ids]),
                              ("anti", inp[ids, h - 1 - ids])):
            if int(cells[0]) != bg and np.all(cells == cells[0]):
                candidates.append((orient, h // 2, int(cells[0])))
    assert len(candidates) == 1
    orientation, axis_position, axis_color = candidates[0]
    assert set(map(tuple, np.argwhere(inp == axis_color))) == _axis_cells(inp.shape, orientation, axis_position)
    payloads = [int(v) for v in values if v not in (bg, axis_color)]
    totals = {c: int(np.count_nonzero(inp == c)) for c in payloads}
    assert totals[payloads[0]] != totals[payloads[1]]
    minority = min(payloads, key=totals.get)
    majority = max(payloads, key=totals.get)
    all_payload = set(map(tuple, np.argwhere((inp == minority) | (inp == majority))))
    sides = {_side_value(r, c, inp.shape, orientation, axis_position) > 0 for r, c in all_payload}
    assert len(sides) == 1
    assert all(_side_value(r, c, inp.shape, orientation, axis_position) != 0 for r, c in all_payload)
    expected = inp.copy()
    for r, c in np.argwhere(inp == minority):
        rr, cc = _reflect(int(r), int(c), inp.shape, orientation, axis_position)
        assert 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == bg
        expected[rr, cc] = minority
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    assert (_grid_rows(inp), _grid_rows(out)) not in _SOURCE_PAIR_ROWS
    minority_parts = _components(inp, minority)
    majority_parts = _components(inp, majority)
    return {"orientation": orientation, "position": axis_position, "shape": inp.shape,
            "background": bg, "axis_color": axis_color,
            "palette": frozenset(payloads), "minority": minority,
            "minority_total": totals[minority], "majority_total": totals[majority],
            "minority_parts": len(minority_parts), "majority_parts": len(majority_parts),
            "minority_sizes": tuple(sorted(map(len, minority_parts))),
            "majority_sizes": tuple(sorted(map(len, majority_parts))),
            "side": next(iter(sides))}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    analyses = {split: [] for split in ("train", "test")}
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            analyses[split].append(info)
            key = (pair["input"].shape, pair["input"].tobytes())
            sig = _signature(pair)
            assert key not in inputs and sig not in pairs
            inputs.add(key)
            pairs.add(sig)
    train = analyses["train"]
    assert train[0]["palette"] == train[1]["palette"]
    assert train[0]["minority"] != train[1]["minority"]
    assert any(x["minority_parts"] > x["majority_parts"] for x in train)
    assert any(x["majority_total"] - x["minority_total"] == 1 for x in train)
    assert any(x["majority_total"] - x["minority_total"] >= 5 for x in train)
    assert any(min(x["majority_sizes"]) < min(x["minority_sizes"]) for x in train)
    has_offcenter_training = any(
        x["orientation"] in ("horizontal", "vertical") and
        x["position"] not in ((x["shape"][0 if x["orientation"] == "horizontal" else 1] - 1) // 2,
                              x["shape"][0 if x["orientation"] == "horizontal" else 1] // 2)
        for x in train)
    assert has_offcenter_training or all(
        x["orientation"] not in ("horizontal", "vertical") or
        x["position"] in ((x["shape"][0 if x["orientation"] == "horizontal" else 1] - 1) // 2,
                          x["shape"][0 if x["orientation"] == "horizontal" else 1] // 2)
        for x in analyses["test"])
    # Colors and orientations carry no secret puzzle-level lookup: every test
    # pair displays its own axis, background, and two countable payload colors.


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and
               all(_signature(x) == _signature(y) for x, y in zip(a[s], b[s]))
               for s in ("train", "test"))


def self_test() -> None:
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(100 + i, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
    for i, kwargs in enumerate(({"num_train": None, "num_test": 1},
                                 {"num_train": None, "num_test": 4},
                                 {"num_train": 2, "num_test": None},
                                 {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(200 + i, **kwargs)
        for name, split in (("num_train", "train"), ("num_test", "test")):
            if kwargs[name] is not None:
                assert len(puzzle[split]) == kwargs[name]
    a = generate_puzzle(314159, 7, 3)
    assert _same_puzzle(a, generate_puzzle(314159, 7, 3))
    assert not _same_puzzle(a, generate_puzzle(314160, 7, 3))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(7, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts, axes_train, axes_test = set(), set(), set(), set()
    shapes, palettes, profiles, signatures = set(), set(), set(), set()
    dimension_overlap = novel_test_colors = offcenter = even_offcenter = 0
    dense_tests = large_test_components = rectangular_tests = 0
    same_axes = shared_sets = mixed_tests = source_shifts = 0
    palette_changes = background_changes = axis_changes = combined = 0
    test_component_counts = set()
    majority_component_counts = set()
    majority_count_changes = 0
    test_count_reversals = 0
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        tr = [_analyze_pair(p) for p in puzzle["train"]]
        te = [_analyze_pair(p) for p in puzzle["test"]]
        axes_train.update(x["orientation"] for x in tr)
        axes_test.update(x["orientation"] for x in te)
        same_axes += any(x["orientation"] in {y["orientation"] for y in tr} for x in te)
        shared_sets += len({x["orientation"] for x in tr + te}) == 1
        mixed_tests += len({x["orientation"] for x in te}) > 1
        source_shifts += ("main" not in {x["orientation"] for x in tr} and
                          any(x["orientation"] == "main" and x["shape"][0] in (17, 18, 19)
                              and x["minority_parts"] == 3 for x in te))
        dimension_overlap += bool({d for x in tr for d in x["shape"]} &
                                  {d for x in te for d in x["shape"]})
        seen_train_colors = set().union(*(x["palette"] for x in tr))
        novel_test_colors += any(x["palette"] - seen_train_colors for x in te)
        palette_changes += len({x["palette"] for x in tr + te}) > 1
        background_changes += len({x["background"] for x in tr + te}) > 1
        axis_changes += len({x["axis_color"] for x in tr + te}) > 1
        rectangular_tests += any(x["shape"][0] != x["shape"][1] for x in te)
        large_test_components += any(max(x["minority_sizes"]) >= 7 for x in te)
        dense_tests += any((x["minority_total"] + x["majority_total"]) /
                           (x["shape"][0] * x["shape"][1]) >= .09 for x in te)
        test_component_counts.update(x["minority_parts"] for x in te)
        majority_component_counts.update(x["majority_parts"] for x in tr + te)
        majority_count_changes += len({x["majority_parts"] for x in tr + te}) > 1
        test_count_reversals += any(x["minority_parts"] > x["majority_parts"] for x in te)
        combined += (len({x["orientation"] for x in tr + te}) >= 2 and
                     len({x["palette"] for x in tr + te}) >= 2 and
                     any(x["minority_parts"] >= 3 for x in tr + te) and
                     any((x["minority_total"] + x["majority_total"]) /
                         (x["shape"][0] * x["shape"][1]) >= .09 for x in tr + te))
        for info, pair in zip(tr + te, puzzle["train"] + puzzle["test"]):
            shapes.add(info["shape"])
            palettes.add(info["palette"])
            profiles.add(info["minority_sizes"])
            signatures.add(_signature(pair))
            n = info["shape"][0 if info["orientation"] == "horizontal" else 1]
            shifted = (info["orientation"] in ("horizontal", "vertical") and
                       info["position"] not in ((n - 1) // 2, n // 2))
            offcenter += shifted
            even_offcenter += shifted and n % 2 == 0
            inp, out = pair["input"], pair["output"]
            expected = inp.copy()
            for r, c in np.argwhere(inp == info["minority"]):
                if info["orientation"] == "horizontal":
                    dest = (2 * info["position"] - r, c)
                elif info["orientation"] == "vertical":
                    dest = (r, 2 * info["position"] - c)
                elif info["orientation"] == "main":
                    dest = (c, r)
                else:
                    dest = (inp.shape[0] - 1 - c, inp.shape[1] - 1 - r)
                expected[dest] = info["minority"]
            assert np.array_equal(out, expected)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert axes_train == axes_test == {"horizontal", "vertical", "main", "anti"}
    assert same_axes >= 8 and shared_sets >= 2 and mixed_tests >= 3
    assert source_shifts >= 2 and dimension_overlap >= 10
    assert len(shapes) >= 30 and len(palettes) >= 20 and len(profiles) >= 15
    assert len(signatures) >= 150 and offcenter >= 12 and even_offcenter >= 4
    assert novel_test_colors >= 8 and palette_changes >= 8
    assert background_changes >= 4 and axis_changes >= 4
    assert rectangular_tests >= 8 and large_test_components >= 8 and dense_tests >= 8
    assert test_component_counts >= {1, 2, 3, 4} and test_count_reversals >= 5
    assert majority_component_counts >= {1, 2, 3, 4} and majority_count_changes >= 20
    assert combined >= 5
