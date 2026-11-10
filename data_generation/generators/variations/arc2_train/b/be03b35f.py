"""Four-panel ARC puzzles whose visible motifs form a quarter-turn sequence."""

import numpy as np


_SOURCE_ROWS = (
    (("10011", "11010", "00000", "11022", "01022"), ("01", "11")),
    (("11011", "10001", "00000", "01022", "11022"), ("10", "11")),
    (("11001", "00001", "00000", "00022", "11022"), ("10", "10")),
    (("11001", "01011", "00000", "10022", "11022"), ("11", "10")),
)
_SOURCE_KEYS = {
    (
        (5, 5, tuple(int(c) for row in inp for c in row)),
        (2, 2, tuple(int(c) for row in out for c in row)),
    )
    for inp, out in _SOURCE_ROWS
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(v) for v in grid.flat))


def _pair_key(pair):
    return (_key(pair["input"]), _key(pair["output"]))


def _orbit(motif):
    return min(_key(np.rot90(motif, k)) for k in range(4))


def _orientation_count(motif):
    return len({_key(np.rot90(motif, k)) for k in range(4)})


def _motif(rng, size, style):
    """Sample a binary motif; all seven styles remain valid at every scale."""
    total = size * size
    motif = np.zeros((size, size), dtype=np.int8)
    if style == 0:  # one cell, including off-center positions
        motif.flat[int(rng.integers(total))] = 1
    elif style == 1:  # one missing cell
        motif[:] = 1
        motif.flat[int(rng.integers(total))] = 0
    elif style == 2:  # connected growth
        target = int(rng.integers(2, max(3, (3 * total) // 5) + 1))
        motif.flat[int(rng.integers(total))] = 1
        while int(motif.sum()) < target:
            frontier = set()
            for row, col in np.argwhere(motif):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = int(row + dr), int(col + dc)
                    if 0 <= rr < size and 0 <= cc < size and motif[rr, cc] == 0:
                        frontier.add((rr, cc))
            if not frontier:
                break
            rr, cc = sorted(frontier)[int(rng.integers(len(frontier)))]
            motif[rr, cc] = 1
    elif style == 3:  # short strokes
        for _ in range(int(rng.integers(1, size + 1))):
            a, b = sorted(int(x) for x in rng.integers(0, size, size=2))
            at = int(rng.integers(size))
            if int(rng.integers(2)):
                motif[at, a:b + 1] = 1
            else:
                motif[a:b + 1, at] = 1
    elif style == 4:  # sparse, separated cells
        count = int(rng.integers(2, max(3, total // 4) + 1))
        motif.flat[rng.choice(total, size=min(total - 1, count), replace=False)] = 1
    elif style == 5:  # dense, with several holes
        motif[:] = 1
        count = int(rng.integers(2, max(3, total // 4) + 1))
        motif.flat[rng.choice(total, size=min(total - 1, count), replace=False)] = 0
    elif style == 6:  # variable-density field
        motif = (rng.random((size, size)) < rng.uniform(.25, .75)).astype(np.int8)
    else:  # half-turn symmetry with a distinct quarter-turn partner
        for r in range(size):
            for c in range(size):
                if (r, c) <= (size - 1 - r, size - 1 - c):
                    v = int(rng.integers(2))
                    motif[r, c] = v
                    motif[size - 1 - r, size - 1 - c] = v
    return motif


def _sample_motif(rng, size, preferred_style=None):
    for attempt in range(300):
        style = int(rng.integers(8)) if preferred_style is None or attempt else preferred_style
        motif = _motif(rng, size, style)
        if 0 < int(motif.sum()) < size * size and _orientation_count(motif) >= 2:
            return motif
    # Deterministic fallback for any size, including 2.
    motif = np.zeros((size, size), dtype=np.int8)
    motif[0, 0] = 1
    return motif


def _make_pair(motif, size, colors, marker_at, turn):
    background, foreground, marker = colors
    panels = [motif]
    panels.append(np.rot90(panels[-1], turn))
    panels.append(np.rot90(panels[-1], turn))
    grid = np.full((2 * size + 1, 2 * size + 1), background, dtype=np.int8)
    locations = ((slice(0, size), slice(0, size)),
                 (slice(0, size), slice(size + 1, None)),
                 (slice(size + 1, None), slice(0, size)),
                 (slice(size + 1, None), slice(size + 1, None)))
    visible = iter(panels)
    for index, region in enumerate(locations):
        if index == marker_at:
            grid[region] = marker
        else:
            grid[region] = np.where(next(visible), foreground, background)
    output = np.where(np.rot90(panels[-1], turn), foreground, background).astype(np.int8)
    return {"input": grid, "output": output}


def _analyze_pair(pair):
    grid, output = pair["input"], pair["output"]
    height, width = grid.shape
    _require(height == width and height % 2 == 1, "input needs four square panels")
    size = (height - 1) // 2
    _require(2 <= size <= 14, "panel size outside 2..14")
    _require(output.shape == (size, size), "output panel dimensions differ")
    background = int(grid[size, size])
    _require(np.all(grid[size, :] == background) and np.all(grid[:, size] == background),
             "separator cross is not uniform")
    panels = (grid[:size, :size], grid[:size, size + 1:],
              grid[size + 1:, :size], grid[size + 1:, size + 1:])
    markers = [i for i, p in enumerate(panels)
               if np.unique(p).size == 1 and int(p[0, 0]) != background]
    _require(len(markers) == 1, "exactly one distinct uniform marker is needed")
    marker_at = markers[0]
    marker = int(panels[marker_at][0, 0])
    visible = [p for i, p in enumerate(panels) if i != marker_at]
    motif_colors = set(int(v) for p in visible for v in np.unique(p))
    _require(len(motif_colors) == 2 and background in motif_colors and marker not in motif_colors,
             "visible panels need distinct binary motif colors")
    foreground = next(v for v in motif_colors if v != background)
    _require(all(set(int(v) for v in np.unique(p)) == motif_colors for p in visible),
             "each visible motif must contain both colors")
    _require(set(int(v) for v in np.unique(grid)) == {background, foreground, marker},
             "input color roles are invalid")
    _require(set(int(v) for v in np.unique(output)) == motif_colors,
             "output must retain the two motif colors")
    matching = [turn for turn in (-1, 1)
                if np.array_equal(visible[1], np.rot90(visible[0], turn))
                and np.array_equal(visible[2], np.rot90(visible[1], turn))]
    _require(bool(matching), "visible panels are not successive quarter-turns")
    possible = [np.rot90(visible[2], turn) for turn in matching]
    _require(all(np.array_equal(possible[0], p) for p in possible[1:]),
             "two directions imply different outputs")
    _require(np.array_equal(output, possible[0]), "output is not the next turn")
    motif = (visible[0] == foreground).astype(np.int8)
    _require(_orientation_count(motif) >= 2, "motif is quarter-turn invariant")
    return {"size": size, "colors": (background, foreground, marker),
            "marker_at": marker_at, "turns": tuple(matching),
            "orbit": _orbit(motif), "motif": motif,
            "foreground_count": int(motif.sum()),
            "orientations": _orientation_count(motif)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    total = train_count + test_count

    # Reading order and the meaning of the uniform marker are shared. The
    # remaining attributes can be inferred from each input's own panels.
    mode = seed % 8
    stable = mode == 0
    source_like = mode == 1
    palette = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
    if source_like and seed % 3 == 1:
        palette = (0, 1, 2)
    base_size = int(rng.integers(2, 7))
    if (stable or source_like) and total > 7 and base_size == 2:
        base_size = 3
    base_marker = 3 if source_like else int(rng.integers(4))
    base_turn = -1 if source_like else (-1 if int(rng.integers(2)) else 1)
    sizes = []
    colors = []
    markers = []
    turns = []
    for index in range(total):
        is_test = index >= train_count
        if stable or source_like:
            size = base_size
            role_colors = palette
            marker_at = base_marker
            turn = base_turn
        else:
            if index == 0:
                size = base_size
            elif index == 1:
                size = 2 + (base_size - 1) % 9
            elif is_test and mode in (5, 6) and index == train_count:
                size = int(rng.integers(10, 15))
            elif is_test and mode in (2, 3, 4) and int(rng.integers(3)) == 0:
                size = sizes[int(rng.integers(train_count))]
            else:
                size = int(rng.integers(2, 10 if not is_test else 15))
            if index == 0:
                role_colors = palette
            elif index == 1:
                role_colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
                if role_colors == palette:
                    role_colors = (palette[1], palette[2], palette[0])
            elif is_test and int(rng.integers(3)) == 0:
                role_colors = colors[int(rng.integers(train_count))]
            else:
                role_colors = tuple(int(v) for v in rng.choice(10, size=3, replace=False))
            if index == 0:
                marker_at = base_marker
            elif index == 1:
                marker_at = (base_marker + 1 + int(rng.integers(3))) % 4
            elif is_test and mode in (5, 6) and index == train_count:
                unseen = [m for m in range(4) if m not in markers[:train_count]]
                marker_at = unseen[int(rng.integers(len(unseen)))] if unseen else int(rng.integers(4))
            else:
                marker_at = int(rng.integers(4))
            turn = base_turn if index == 0 else -base_turn if index == 1 else (-1 if int(rng.integers(2)) else 1)
        sizes.append(size)
        colors.append(role_colors)
        markers.append(marker_at)
        turns.append(turn)

    pairs = []
    used_inputs = set()
    train_motifs = []
    for index in range(total):
        size = sizes[index]
        # Train examples 1 and 2 prove direction and the reading-order rule.
        # Later examples include singletons, holes, and occasional half-turn symmetry.
        preferred = None
        if index >= 2 and index % 7 == 2:
            preferred = 0
        elif index >= 2 and index % 7 == 3:
            preferred = 1
        elif index >= 2 and index % 11 == 4:
            preferred = 7
        for attempt in range(500):
            familiar = (index >= train_count and mode in (2, 3, 4)
                        and (index - train_count) % 2 == 0 and attempt == 0)
            candidates = [m for m in train_motifs if m.shape[0] == size]
            if familiar and candidates:
                motif = np.rot90(candidates[int(rng.integers(len(candidates)))],
                                 int(rng.integers(1, 4))).copy()
            else:
                motif = _sample_motif(rng, size, preferred if attempt == 0 else None)
            if index < 2 and _orientation_count(motif) != 4:
                continue
            pair = _make_pair(motif, size, colors[index], markers[index], turns[index])
            key = _key(pair["input"])
            if key in used_inputs or _pair_key(pair) in _SOURCE_KEYS:
                continue
            used_inputs.add(key)
            pairs.append(pair)
            if index < train_count:
                train_motifs.append(motif)
            break
        else:
            raise RuntimeError("could not construct a distinct valid pair")
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    inputs = set()
    complete = set()
    infos = []
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair needs exactly input and output")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
            _require(all(1 <= v <= 30 for v in grid.shape), "illegal grid size")
            _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
            _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "illegal grid color")
        inp_key = _key(pair["input"])
        both_key = _pair_key(pair)
        _require(inp_key not in inputs and both_key not in complete, "duplicate input or pair")
        _require(both_key not in _SOURCE_KEYS, "exact source pair copied")
        inputs.add(inp_key)
        complete.add(both_key)
        infos.append(_analyze_pair(pair))
    train_info = infos[:len(train)]
    test_info = infos[len(train):]
    _require(all(info["orientations"] == 4 for info in train_info[:2]),
             "first two training motifs must show direction")
    _require(len({_key(info["motif"]) for info in train_info}) >= 2,
             "training needs distinct motif arrangements")
    trained_turns = {info["turns"][0] for info in train_info if len(info["turns"]) == 1}
    _require(all(len(info["turns"]) == 2 or info["turns"][0] in trained_turns
                 for info in test_info), "test turn lacks a training demonstration")


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "cross-seed duplication")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "too few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "too few test counts")
    sizes, markers, palettes, turns, orbits = set(), set(), set(), set(), set()
    size_mixed = marker_mixed = palette_mixed = turn_mixed = 0
    sparse = nearly_full = half_turn = larger_test = familiar_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_pair(p) for p in puzzle["train"]]
        test_info = [_analyze_pair(p) for p in puzzle["test"]]
        info = train_info + test_info
        sizes.update(i["size"] for i in info)
        markers.update(i["marker_at"] for i in info)
        palettes.update(i["colors"] for i in info)
        turns.update(t for i in info for t in i["turns"])
        orbits.update(i["orbit"] for i in info)
        size_mixed += len({i["size"] for i in info}) > 1
        marker_mixed += len({i["marker_at"] for i in info}) > 1
        palette_mixed += len({i["colors"] for i in info}) > 1
        turn_mixed += {-1, 1} <= {i["turns"][0] for i in train_info if len(i["turns"]) == 1}
        sparse += sum(i["foreground_count"] == 1 and i["size"] > 2 for i in info)
        nearly_full += sum(i["foreground_count"] == i["size"] ** 2 - 1 and i["size"] > 2 for i in info)
        half_turn += sum(i["orientations"] == 2 for i in info)
        larger_test += max(i["size"] for i in test_info) > max(i["size"] for i in train_info)
        familiar_test += any(i["orbit"] in {t["orbit"] for t in train_info} for i in test_info)
    _require(sizes == set(range(2, 15)), "panel scales 2..14 not covered")
    _require(markers == set(range(4)), "marker locations not covered")
    _require(turns == {-1, 1}, "turn directions not covered")
    _require(len(palettes) >= 40 and len(orbits) >= 100, "palette or motif diversity too low")
    _require(min(size_mixed, marker_mixed, palette_mixed, turn_mixed) >= 20,
             "insufficient within-puzzle variation")
    _require(min(sparse, nearly_full, half_turn, larger_test, familiar_test) > 0,
             "a requested motif or test regime is absent")
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3)):
        p = generate_puzzle(90 + train + test, train, test)
        _require((len(p["train"]), len(p["test"])) == (train, test), "explicit counts ignored")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(200 + (train or 0) + (test or 0), train, test)
        _require((train is None or len(p["train"]) == train)
                 and (test is None or len(p["test"]) == test), "partial counts ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzle_key(generate_puzzle(314, 8, 3)) == _puzzle_key(generate_puzzle(314, 8, 3)),
             "explicit generation nondeterministic")
    _require(_puzzle_key(generate_puzzle(315)) == _puzzle_key(generate_puzzle(315)),
             "automatic generation nondeterministic")
