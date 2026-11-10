import numpy as np


def _rows_key(rows):
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


def _seq_key(seq):
    return ((len(seq), 1), tuple(int(x) for x in seq))


_SOURCE_PAIR_KEYS = frozenset(
    {
        (
            _rows_key(
                (
                    "8888888888888888",
                    "8888881888888888",
                    "8888811888888888",
                    "8888818888888888",
                    "8888818888888888",
                    "8888833388888888",
                    "8888888388888888",
                    "8888888222888888",
                    "8888888882888888",
                    "8888888882888888",
                    "8888884444888888",
                    "8888884888888888",
                    "8888886668888888",
                    "8888888888888888",
                    "8888888888888888",
                    "8888888888888888",
                )
            ),
            _seq_key(
                [1] * 5
                + [3] * 4
                + [2] * 5
                + [4] * 5
                + [6] * 3
            ),
        ),
        (
            _rows_key(
                (
                    "1111111111111111",
                    "1811111111111111",
                    "1811111111111111",
                    "1888811111111111",
                    "1111811111111111",
                    "1111211111111111",
                    "1111211111111111",
                    "1111222211111111",
                    "1111111211111111",
                    "1111111211111111",
                    "1111111311111111",
                    "1111111311111111",
                    "1111111333311111",
                    "1111111111311111",
                    "1111111111331111",
                    "1111111111111111",
                )
            ),
            _seq_key([8] * 7 + [2] * 8 + [3] * 9),
        ),
        (
            _rows_key(
                (
                    "2222222222222222",
                    "2222222222222222",
                    "2111122222222222",
                    "2222122222222222",
                    "2222332222222222",
                    "2222232222222222",
                    "2222282222222222",
                    "2222882222222222",
                    "2222822222222222",
                    "2222822222222222",
                    "2222844422332222",
                    "2222222422122222",
                    "2222222441122222",
                    "2222222222222222",
                    "2222222222222222",
                    "2222222222222222",
                )
            ),
            _seq_key(
                [1] * 5
                + [3] * 3
                + [8] * 6
                + [4] * 6
                + [1] * 3
                + [3] * 2
            ),
        ),
    }
)


def _array_key(array):
    return (
        tuple(array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _pair_key(pair):
    return (
        _array_key(pair["input"]),
        _array_key(pair["output"]),
    )


def _resolve_counts(rng, num_train, num_test):
    def checked(value, name, low, high):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must be an integer or None")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}"
            )
        return value

    num_train = checked(num_train, "num_train", 2, 10)
    num_test = checked(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_test is None:
        num_test = int(
            rng.integers(1, min(4, num_train - 1) + 1)
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _trace_input(grid):
    values, counts = np.unique(
        grid,
        return_counts=True,
    )

    assert len(values) >= 2

    maximum = int(counts.max())
    assert int(
        np.count_nonzero(counts == maximum)
    ) == 1

    background = int(
        values[int(np.argmax(counts))]
    )

    cells = {
        tuple(int(x) for x in coordinates)
        for coordinates in np.argwhere(
            grid != background
        )
    }

    assert 2 <= len(cells) <= 30

    neighbor_map = {}

    for r, c in cells:
        neighbors = []

        for dr, dc in (
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ):
            neighbor = (r + dr, c + dc)
            if neighbor in cells:
                neighbors.append(neighbor)

        neighbor_map[(r, c)] = neighbors

    endpoints = [
        cell
        for cell, neighbors in neighbor_map.items()
        if len(neighbors) == 1
    ]

    assert len(endpoints) == 2
    assert all(
        len(neighbor_map[cell]) in (1, 2)
        for cell in cells
    )
    assert endpoints[0][0] != endpoints[1][0]

    if endpoints[0][0] < endpoints[1][0]:
        start = endpoints[0]
        finish = endpoints[1]
    else:
        start = endpoints[1]
        finish = endpoints[0]

    ordered = []
    visited = set()
    previous = None
    current = start

    while True:
        assert current not in visited

        visited.add(current)
        ordered.append(current)

        choices = [
            neighbor
            for neighbor in neighbor_map[current]
            if neighbor != previous
        ]

        if not choices:
            break

        assert len(choices) == 1
        previous, current = current, choices[0]

    assert current == finish
    assert len(ordered) == len(cells)

    sequence = [
        int(grid[r, c])
        for r, c in ordered
    ]

    return {
        "background": background,
        "path": ordered,
        "sequence": sequence,
        "foreground_colors": set(sequence),
    }


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _steps(path):
    return [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]


def _geometry(path):
    steps = _steps(path)
    return {
        "turns": sum(a != b for a, b in zip(steps, steps[1:])),
        "reversal": (0, 1) in steps and (0, -1) in steps,
        "up": (-1, 0) in steps,
        "internal_up": any(a == (-1, 0) and b != a for a, b in zip(steps, steps[1:])),
        "above": min(r for r, _ in path) < path[0][0],
        "interior_start": min(c for _, c in path) < path[0][1] < max(c for _, c in path),
        "horizontal_start": steps[0][0] == 0,
        "horizontal_finish": steps[-1][0] == 0,
    }


def _walk(rng, length, monotone, persistence):
    """Grow an induced lattice path: a new cell touches only the old tip."""
    path = [(0, 0)]
    occupied = set(path)
    directions = _DIRECTIONS[1:] if monotone else _DIRECTIONS
    previous = None
    for _ in range(length - 1):
        candidates, weights = [], []
        for dr, dc in directions:
            q = (path[-1][0] + dr, path[-1][1] + dc)
            if q in occupied:
                continue
            if sum((q[0] + a, q[1] + b) in occupied for a, b in _DIRECTIONS) != 1:
                continue
            candidates.append((q, (dr, dc)))
            weights.append(persistence if (dr, dc) == previous else 1.0)
        if not candidates:
            return None
        weights = np.asarray(weights, dtype=float)
        q, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        path.append(q)
        occupied.add(q)
    return path


def _orient(rng, path, monotone):
    if not monotone:
        # Reversing after any D4 transform is essential: the readout stays vertical.
        rotation = int(rng.integers(4))
        reflection = int(rng.choice((-1, 1)))
        path = [(r, reflection * c) for r, c in path]
        for _ in range(rotation):
            path = [(c, -r) for r, c in path]
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _fallback_path(length, above, folded, witness):
    # Constructive safety net, not the primary shape vocabulary. Each segment
    # is separated from previous segments; the formulas work at any legal length.
    if above:
        segments = [((-1, 0), 2), ((0, 1), 3), ((1, 0), length - 8), ((0, -1), 2)]
    elif folded:
        segments = [((0, 1), 2), ((1, 0), 4), ((0, -1), length - 9),
                    ((-1, 0), 1), ((0, -1), 1)]
    elif witness:
        segments = [((0, 1), 2), ((1, 0), 2), ((0, -1), length - 6), ((1, 0), 1)]
    else:
        segments = [((1, 0), length - 1)]
    path = [(0, 0)]
    for (dr, dc), count in segments:
        for _ in range(count):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    return path


def _make_path(rng, length, style, allow_above, need_above=False,
               witness=False, folded=False, avoid_turns=None):
    monotone = style == "downward" and not need_above
    persistence = float(rng.uniform(1.0, 5.0))
    if style == "gentle":
        persistence = float(rng.uniform(12.0, 45.0))
    elif style == "winding":
        persistence = float(rng.uniform(0.15, 1.5))
    preferred = None
    for _ in range(160):
        path = _walk(rng, length, monotone, persistence)
        if path is None:
            continue
        path = _orient(rng, path, monotone)
        if path is None:
            continue
        f = _geometry(path)
        if (f["above"] and not allow_above) or (need_above and not f["above"]):
            continue
        if witness and not f["reversal"]:
            continue
        if folded and not f["internal_up"]:
            continue
        preferred = path
        if avoid_turns is None or f["turns"] != avoid_turns:
            return path
    if preferred is not None:
        return preferred
    return _fallback_path(length, need_above, folded, witness)


def _sweep_orders(path):
    # The review's shortcut: visit columns in either order, downward except
    # possibly the last column. Deliberately does not use graph adjacency.
    columns = sorted({c for _, c in path})
    orders = []
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in cs for p in sorted(
                (p for p in path if p[1] == c),
                reverse=reverse_last and c == cs[-1])])
    return orders


def _sweep_matches(path, sequence):
    color = dict(zip(path, sequence))
    return any([color[p] for p in order] == list(sequence) for order in _sweep_orders(path))


def _color_sequence(rng, length, background, style):
    available = [c for c in range(10) if c != background]
    rng.shuffle(available)
    if style == "mono":
        return [available[0]] * length
    if style == "unique":
        count = int(rng.integers(2, min(9, length) + 1))
        colors = available[:count]
    else:
        distinct = 2 if style == "alternate" else int(rng.integers(2, min(9, length) + 1))
        palette = available[:distinct]
        low = max(2, (length + 1) // 2) if style == "fine" else min(4, length)
        count = int(rng.integers(low, length + 1))
        if style == "dominant":
            count = int(rng.integers(3, min(9, length) + 1)) if length >= 3 else 2
        colors = []
        for i in range(count):
            if style in ("alternate", "dominant") and i % 2 == 0:
                color = palette[0]
            else:
                choices = [c for c in palette if not colors or c != colors[-1]]
                if style == "dominant":
                    choices = palette[1:]
                color = int(rng.choice(choices))
            colors.append(color)
    sizes = np.ones(len(colors), dtype=int)
    remaining = length - len(colors)
    if style == "dominant":
        # One long piece interrupted by singleton pieces, at varying positions.
        sizes[int(rng.choice(np.arange(0, len(colors), 2)))] += remaining
    elif remaining:
        alpha = float(rng.choice((0.12, 0.5, 1.5, 4.0)))
        weights = rng.dirichlet(np.full(len(colors), alpha))
        sizes += rng.multinomial(remaining, weights)
    return [int(c) for c, n in zip(colors, sizes) for _ in range(int(n))]


def _witness_colors(rng, path, background, style):
    for _ in range(48):
        sequence = _color_sequence(rng, len(path), background, style)
        if sequence != sequence[::-1] and not _sweep_matches(path, sequence):
            return sequence
    # Deterministic distinguishing partition. Every competing sweep differs
    # somewhere in index order; put a color boundary between those two indices.
    indices = {p: i for i, p in enumerate(path)}
    boundaries = {0, len(path), len(path) // 2}
    for order in _sweep_orders(path):
        mismatch = next((i for i, p in enumerate(order) if indices[p] != i), None)
        assert mismatch is not None
        boundaries.add(min(mismatch, indices[order[mismatch]]) + 1)
    colors = [c for c in range(10) if c != background]
    rng.shuffle(colors)
    cuts = sorted(boundaries)
    return [colors[i] for i, (a, b) in enumerate(zip(cuts, cuts[1:])) for _ in range(b - a)]


def _embed_path(rng, path, colors, background):
    min_r, min_c = min(r for r, _ in path), min(c for _, c in path)
    path = [(r - min_r, c - min_c) for r, c in path]
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    compact = bool(rng.random() < 0.55)
    h = min(30, bh + int(rng.integers(0, 5))) if compact else int(rng.integers(bh, 31))
    w = min(30, bw + int(rng.integers(0, 5))) if compact else int(rng.integers(bw, 31))
    # A strict majority makes background detection independent of path colors,
    # including monochrome paths and border contact on very small grids.
    while h * w <= 2 * len(path):
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    if rng.random() < 0.4:
        if rng.random() < 0.5:
            top = int(rng.choice((0, h - bh)))
        else:
            left = int(rng.choice((0, w - bw)))
    grid = np.full((h, w), background, dtype=np.int8)
    for (r, c), color in zip(path, colors):
        grid[r + top, c + left] = color
    return {"input": grid, "output": np.asarray(colors, dtype=np.int8).reshape(-1, 1)}


def _sample_puzzle_parameters(rng, num_train, num_test):
    # The learned convention is always upper endpoint -> vertical sequence.
    # These are sampling plans, not hidden parameters needed to solve a test.
    mode = str(rng.choice(("source_like", "length", "geometry", "color", "matched", "mixed")))
    bg_mode = str(rng.choice(("shared", "pool", "free", "new_test")))
    focused = mode in ("length", "geometry", "color") and bool(rng.random() < 0.65)
    if focused:
        bg_mode = "shared"
    if mode == "source_like":
        bg_mode = "new_test"
    palette = [int(c) for c in rng.permutation(10)]
    if bg_mode == "shared":
        backgrounds = [palette[0]] * (num_train + num_test)
    elif bg_mode == "pool":
        pool = palette[:int(rng.integers(2, 6))]
        backgrounds = [int(rng.choice(pool)) for _ in range(num_train + num_test)]
    elif bg_mode == "new_test":
        count = int(rng.integers(1, 5))
        backgrounds = [int(rng.choice(palette[:count])) for _ in range(num_train)]
        backgrounds += [int(rng.choice(palette[count:])) for _ in range(num_test)]
    else:
        backgrounds = [int(rng.integers(10)) for _ in range(num_train + num_test)]
    return {
        "mode": mode,
        "focused": focused,
        "backgrounds": backgrounds,
        "allow_above": mode != "source_like" and not (focused and mode == "color")
                       and bool(rng.random() < 0.5),
        "band": ((4, 14), (10, 22), (16, 30))[int(rng.integers(3))],
        "geometry": str(rng.choice(("downward", "walk", "gentle", "winding"))),
        "color": str(rng.choice(("unique", "free", "alternate", "dominant", "fine"))),
    }


def _pair_spec(rng, parameters, regime, index):
    target = regime == "target_test"
    mode = parameters["mode"]
    low, high = parameters["band"]
    geometry, coloring = parameters["geometry"], parameters["color"]
    folded = False
    if mode == "source_like":
        low, high = (25, 30) if target else (12, 24)
        geometry, coloring = ("winding", "free") if target else ("downward", "unique")
        folded = target
    elif mode == "length":
        low, high = (22, 30) if target else (7, 19)
    elif mode == "geometry":
        low, high = 12, 26
        geometry = "winding" if target else str(rng.choice(("downward", "walk", "gentle")))
        folded = target
    elif mode == "color":
        coloring = str(rng.choice(("free", "alternate", "dominant", "fine"))) if target else "unique"
        if parameters["focused"]:
            geometry = "downward"
    elif mode == "mixed":
        low, high = 2, 30
        geometry = str(rng.choice(("downward", "walk", "gentle", "winding")))
        coloring = str(rng.choice(("unique", "free", "alternate", "dominant", "fine", "mono")))
    elif mode == "matched":
        # Same broad family and scale on both sides, with independent local draws.
        if rng.random() < 0.3:
            coloring = str(rng.choice(("free", "alternate", "dominant", "fine", "mono")))
    length = int(rng.integers(low, high + 1))
    witness = index == 0
    above = (index == 0 and parameters["allow_above"]
             and (not target or bool(rng.random() < 0.6)))
    if witness or folded:
        length = max(12, length)
        if coloring == "mono":
            coloring = "free"
    if mode in ("mixed", "matched", "length") and not target and index == 0:
        coloring = str(rng.choice(("free", "alternate", "dominant", "fine")))
    if above:
        geometry = "walk"
    return length, geometry, coloring, witness, folded, above


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    bg_index = 0
    for split, regime, count in (("train", "source_train", num_train), ("test", "target_test", num_test)):
        previous_turns = None
        previous_length = None
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            length, style, coloring, witness, folded, above = _pair_spec(pair_rng, parameters, regime, index)
            if split == "test" and parameters["focused"] and parameters["mode"] in ("color", "geometry"):
                familiar = [len(p["output"]) for p in puzzle["train"] if len(p["output"]) >= 12]
                length = int(pair_rng.choice(familiar))
            if length == previous_length:
                # Keep focused tests at a demonstrated scale. Other modes may
                # move within the supported family to increase local diversity.
                if not (split == "test" and parameters["focused"]):
                    length = length - 1 if length > 12 else length + 1
            path = _make_path(pair_rng, length, style, parameters["allow_above"],
                              above, witness, folded, previous_turns)
            background = parameters["backgrounds"][bg_index]
            bg_index += 1
            if split == "test" and parameters["mode"] == "source_like":
                train_bg = {_trace_input(p["input"])["background"] for p in puzzle["train"]}
                train_fg = set().union(*(set(p["output"].flat) for p in puzzle["train"]))
                choices = sorted(train_fg - train_bg)
                if choices:
                    background = int(pair_rng.choice(choices))
            colors = (_witness_colors(pair_rng, path, background, coloring) if witness else
                      _color_sequence(pair_rng, length, background, coloring))
            if split == "test" and parameters["mode"] in ("source_like", "color"):
                runs = _run_colors(colors)
                if len(runs) == len(set(runs)):
                    colors = (_witness_colors(pair_rng, path, background, "alternate") if witness else
                              _color_sequence(pair_rng, length, background, "alternate"))
            pair = _embed_path(pair_rng, path, colors, background)
            # Bounded deterministic uniqueness fallback: enumerating canvas
            # sizes supplies more choices than all 14 possible pairs.
            if _array_key(pair["input"]) in used_inputs or _pair_key(pair) in _SOURCE_PAIR_KEYS:
                cells = [(int(r), int(c)) for r, c in np.argwhere(pair["input"] != background)]
                origin = (min(r for r, _ in cells), min(c for _, c in cells))
                bh = max(r for r, _ in cells) - origin[0] + 1
                bw = max(c for _, c in cells) - origin[1] + 1
                done = False
                for h in range(bh, 31):
                    for w in range(bw, 31):
                        if h * w <= 2 * length:
                            continue
                        grid = np.full((h, w), background, dtype=np.int8)
                        for r, c in cells:
                            grid[r - origin[0], c - origin[1]] = pair["input"][r, c]
                        candidate = {"input": grid, "output": pair["output"]}
                        if _array_key(grid) not in used_inputs and _pair_key(candidate) not in _SOURCE_PAIR_KEYS:
                            pair, done = candidate, True
                            break
                    if done:
                        break
                assert done
            puzzle[split].append(pair)
            used_inputs.add(_array_key(pair["input"]))
            previous_turns, previous_length = _geometry(path)["turns"], length
    validate_generated_puzzle(puzzle)
    return puzzle


def _run_colors(sequence):
    return [int(c) for i, c in enumerate(sequence) if i == 0 or c != sequence[i - 1]]


def _validate_pair(pair, regime=None):
    """Semantic validation is independent of optional sampling regimes."""
    assert regime in (None, "source_train", "target_test")
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    for grid in pair.values():
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    info = _trace_input(pair["input"])
    expected = np.asarray(info["sequence"], dtype=int).reshape(-1, 1)
    assert np.array_equal(pair["output"], expected)
    assert not np.array_equal(pair["input"], pair["output"])
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[split]:
            info = _validate_pair(pair, regime)
            assert _array_key(pair["input"]) not in seen
            assert _pair_key(pair) not in _SOURCE_PAIR_KEYS
            seen.add(_array_key(pair["input"]))
            infos[split].append(info)
    # Demonstration coverage, not arbitrary regime restrictions. No universal
    # palette, length, run-count, recurrence-position, or upward-tail schedule.
    assert any(i["sequence"] != i["sequence"][::-1] for i in infos["train"])
    assert any(not _sweep_matches(i["path"], i["sequence"]) for i in infos["train"])
    assert any(not _sweep_matches(i["path"], i["sequence"]) for i in infos["test"])
    if any(_geometry(i["path"])["above"] for i in infos["test"]):
        assert any(_geometry(i["path"])["above"] and i["sequence"] != i["sequence"][::-1]
                   for i in infos["train"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def _pixel_prediction(train_pairs, test_input):
    """A second implementation: distance along a pixel adjacency matrix.

    Infer the endpoint direction from demonstrations, with no test label, seed,
    or construction path. This is a local check, not a protected reference solver.
    """
    def candidates(grid):
        histogram = np.bincount(grid.ravel(), minlength=10)
        background = int(histogram.argmax())
        coords = np.argwhere(grid != background)
        distance = np.abs(coords[:, None, :] - coords[None, :, :]).sum(axis=2)
        adjacency = distance == 1
        degrees = adjacency.sum(axis=1)
        endpoints = np.flatnonzero(degrees == 1)
        assert len(endpoints) == 2 and np.all((degrees == 1) | (degrees == 2))
        assert coords[endpoints[0], 0] != coords[endpoints[1], 0]
        start = int(endpoints[np.argmin(coords[endpoints, 0])])
        distances = np.full(len(coords), -1, dtype=int)
        distances[start] = 0
        frontier = [start]
        while frontier:
            current = frontier.pop(0)
            for neighbor in np.flatnonzero(adjacency[current]):
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[current] + 1
                    frontier.append(int(neighbor))
        assert np.all(distances >= 0) and len(set(distances)) == len(coords)
        points = coords[np.argsort(distances)]
        answer = grid[points[:, 0], points[:, 1]].reshape(-1, 1)
        return answer, answer[::-1]
    directions = {0, 1}
    for pair in train_pairs:
        possibilities = candidates(pair["input"])
        directions &= {i for i in directions if np.array_equal(possibilities[i], pair["output"])}
    assert len(directions) == 1
    return candidates(test_input)[next(iter(directions))].copy()


def _features(pair):
    info = _trace_input(pair["input"])
    path, seq = info["path"], info["sequence"]
    f = _geometry(path)
    runs = _run_colors(seq)
    boundaries = [i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]
    cuts = [0] + boundaries + [len(seq)]
    sizes = [b - a for a, b in zip(cuts, cuts[1:])]
    repeat_indices = [i for i, c in enumerate(runs) if c in runs[:i]]
    h, w = pair["input"].shape
    f.update({
        "length": len(path), "runs": len(runs), "distinct": len(set(seq)),
        "run_lengths": tuple(sizes), "repeat_positions": tuple(repeat_indices),
        "foreground_palette": tuple(sorted(set(seq))),
        "background": info["background"], "singleton": 1 in sizes,
        "recurrence": bool(repeat_indices),
        "early_recurrence": any(i < len(runs) - 2 for i in repeat_indices),
        "triple_recurrence": max(runs.count(c) for c in set(runs)) >= 3,
        "alternation": len(set(seq)) == 2 and len(runs) >= 4,
        "dominant": len(runs) >= 3 and max(sizes) >= len(seq) / 2 and min(sizes) == 1,
        "fine": len(runs) >= len(seq) / 2,
        "mono": len(set(seq)) == 1,
        "short": len(path) < 16, "compact": h < 9 and w < 9,
        "border": any(r in (0, h - 1) or c in (0, w - 1) for r, c in path),
        "sweep_failure": not _sweep_matches(path, seq),
        "shape": (h, w),
        "layout": tuple((r - path[0][0], c - path[0][1]) for r, c in path),
        "boundaries_off_bends": any(_steps(path)[i - 1] == _steps(path)[i]
                                     for i in boundaries if i < len(path) - 1),
        "bends_without_boundaries": any(a != b and seq[i] == seq[i + 1]
                                         for i, (a, b) in enumerate(zip(_steps(path), _steps(path)[1:]))),
        "unequal_sections": len(set(_section_lengths(path))) > 1,
        "up_stretches": sum(step == (-1, 0) and (i == 0 or _steps(path)[i - 1] != step)
                            for i, step in enumerate(_steps(path))),
        "down_stretches": sum(step == (1, 0) and (i == 0 or _steps(path)[i - 1] != step)
                              for i, step in enumerate(_steps(path))),
    })
    return f


def _section_lengths(path):
    steps = _steps(path)
    cuts = [0] + [i for i in range(1, len(steps)) if steps[i] != steps[i - 1]] + [len(steps)]
    return [b - a for a, b in zip(cuts, cuts[1:])]


def self_test() -> None:
    # Exact source regressions deliberately bypass source-copy sampling exclusion.
    source_pairs = [{"input": np.asarray(ik[1], dtype=int).reshape(ik[0]),
                     "output": np.asarray(ok[1], dtype=int).reshape(ok[0])}
                    for ik, ok in sorted(_SOURCE_PAIR_KEYS)]
    for pair in source_pairs:
        _validate_pair(pair, "target_test")
        others = [p for p in source_pairs if p is not pair]
        assert np.array_equal(_pixel_prediction(others, pair["input"].copy()), pair["output"])
    # D4 regression: track the original endpoint through the image transform,
    # then check whether its row requires reversing the original color sequence.
    reversed_outputs = 0
    for pair in source_pairs:
        path = _trace_input(pair["input"])["path"]
        endpoints = np.zeros_like(pair["input"])
        endpoints[path[0]], endpoints[path[-1]] = 1, 2
        for rotation in range(4):
            for reflect in (False, True):
                grid = np.rot90(pair["input"], rotation)
                markers = np.rot90(endpoints, rotation)
                if reflect:
                    grid, markers = np.fliplr(grid), np.fliplr(markers)
                start_row = int(np.argwhere(markers == 1)[0, 0])
                finish_row = int(np.argwhere(markers == 2)[0, 0])
                if start_row == finish_row:
                    try:
                        _trace_input(grid)
                    except AssertionError:
                        continue
                    raise AssertionError("rotated equal-row endpoints accepted")
                reverse = start_row > finish_row
                reversed_outputs += reverse
                expected = pair["output"][::-1] if reverse else pair["output"]
                _validate_pair({"input": grid, "output": expected})
                assert np.array_equal(_pixel_prediction(source_pairs, grid.copy()), expected)
    assert reversed_outputs >= 6
    for length in range(12, 31):
        for above, folded, witness in ((True, False, True), (False, True, True),
                                        (False, False, True), (False, False, False)):
            path = _fallback_path(length, above, folded, witness)
            fallback_rng = np.random.default_rng(length)
            colors = (_witness_colors(fallback_rng, path, 0, "unique") if witness else [1] * length)
            pair = _embed_path(fallback_rng, path, colors, 0)
            traced = _validate_pair(pair)
            assert len(traced["path"]) == length
            f = _geometry(traced["path"])
            assert not above or f["above"]
            assert not folded or f["internal_up"]
            assert not witness or not _sweep_matches(traced["path"], traced["sequence"])
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                records.append((seed, split, _features(pair)))
                train = [{k: v.copy() for k, v in x.items()} for x in p["train"]]
                assert np.array_equal(_pixel_prediction(train, pair["input"].copy()), pair["output"])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({f["shape"] for _, _, f in records}) >= 70
    assert len({f["layout"] for _, _, f in records}) >= 150
    assert len({f["length"] for _, _, f in records}) >= 24
    assert len({f["background"] for _, _, f in records}) == 10
    assert len({f["distinct"] for _, _, f in records}) >= 8
    assert len({f["turns"] for _, s, f in records if s == "test"}) >= 9
    for feature in ("reversal", "internal_up", "interior_start", "horizontal_start", "horizontal_finish",
                    "singleton", "early_recurrence", "triple_recurrence", "alternation", "dominant", "fine",
                    "short", "compact", "border", "above", "boundaries_off_bends", "bends_without_boundaries"):
        assert sum(bool(f[feature]) for _, _, f in records) >= 8, feature
    assert any(f["mono"] for _, _, f in records)
    assert any(f["turns"] <= 1 for _, _, f in records)
    assert sum(f["up_stretches"] >= 2 for _, _, f in records) >= 8
    assert sum(f["up_stretches"] >= 2 and f["down_stretches"] >= 2 for _, _, f in records) >= 8
    for feature in ("reversal", "internal_up", "horizontal_start", "horizontal_finish", "singleton",
                    "early_recurrence", "triple_recurrence", "short", "border"):
        assert sum(bool(f[feature]) for _, s, f in records if s == "test") >= 5, feature
    # Run counts and palette sizes must not encode one another; recurrence
    # positions and singleton lengths are measured on reconstructed sequences.
    assert sum(len({f["runs"] for _, _, f in records if f["distinct"] == d}) >= 4
               for d in range(2, 10)) >= 5
    assert len({i for _, _, f in records for i in f["repeat_positions"]}) >= 12
    assert max(n for _, _, f in records for n in f["run_lengths"]) >= 15
    assert sum(f["reversal"] and f["internal_up"] and f["singleton"] and f["recurrence"]
               for _, _, f in records) >= 8
    for seed, p in enumerate(puzzles):
        fs = [f for n, _, f in records if n == seed]
        assert len({f["turns"] for f in fs}) >= 2
        assert len({f["length"] for f in fs}) >= 2
        assert len({f["runs"] for f in fs}) >= 2
        for split in ("train", "test"):
            assert any(f["sweep_failure"] for n, s, f in records if n == seed and s == split)
    # Observe regimes from actual arrays, not the sampled mode name.
    source_like = recurrent_train = comparable = shared = reused = new = 0
    for p in puzzles:
        tr, te = [[_features(x) for x in p[s]] for s in ("train", "test")]
        source_like += (all(not f["recurrence"] and not f["up"] for f in tr)
                        and all(f["recurrence"] and f["up"] for f in te)
                        and min(f["length"] for f in te) > max(f["length"] for f in tr))
        recurrent_train += any(f["recurrence"] for f in tr)
        comparable += any(f["length"] <= max(t["length"] for t in tr) for f in te)
        backgrounds = {f["background"] for f in tr}
        shared += len({f["background"] for f in tr + te}) == 1
        reused += any(f["background"] in backgrounds for f in te)
        new += any(f["background"] not in backgrounds for f in te)
    assert source_like >= 2 and recurrent_train >= 12 and comparable >= 12
    assert shared >= 3 and reused >= 8 and new >= 8
    for counts in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(901, *counts)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(901, *counts))
        for actual, requested in zip((len(p["train"]), len(p["test"])), counts):
            assert requested is None or actual == requested
        validate_generated_puzzle(p)
    assert _puzzle_key(puzzles[17]) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(400, 5, 2)) != _puzzle_key(generate_puzzle(401, 5, 2))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Corrupted answers, tied endpoints, branches, cycles and disconnected cells.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (bad["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    for cells in ([(1, 1), (1, 2), (1, 3)],
                  [(1, 2), (2, 1), (2, 2), (2, 3)],
                  [(1, 1), (1, 2), (2, 1), (2, 2)],
                  [(1, 1), (2, 1), (4, 4)]):
        grid = np.zeros((6, 6), dtype=int)
        for point in cells:
            grid[point] = 1
        try:
            _trace_input(grid)
        except AssertionError:
            continue
        raise AssertionError("malformed path accepted")
