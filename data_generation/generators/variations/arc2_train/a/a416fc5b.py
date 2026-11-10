import itertools
import numpy as np


# Boundary compartments in clockwise order. The preceding and following
# compartments beside the target are left and right along the directed path.
_RING = ((0, 0), (0, 1), (0, 2), (1, 2),
         (2, 2), (2, 1), (2, 0), (1, 0))
_RING_INDEX = {place: index for index, place in enumerate(_RING)}
_MOTIFS = {
    "diamond": ((0, 1), (1, 0), (1, 2), (2, 1)),
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "tee": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "ell": ((0, 0), (1, 0), (2, 0), (2, 1)),
}
_GEOMETRIES = tuple((c, d) for c in range(3, 9) for d in range(1, 4)
                    if 3 * c + 2 * d <= 30)
_SOURCE_COLORS = (7, 6, 2, 5, 8)


def _check_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    nt = _check_count(num_train, "num_train", 2, 10)
    ne = _check_count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        ne = 1 + seed % 4
        low = max(2, ne + 1)
        nt = low + (seed // 4) % (11 - low)
    elif nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    elif ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _places(direction):
    target_index = (direction + 4) % 8
    return (_RING[direction], _RING[target_index],
            _RING[(target_index - 1) % 8],
            _RING[(target_index + 1) % 8])


def _starts(c, d):
    return (0, c + d, 2 * (c + d))


def _blank(c, d, background, divider):
    side = 3 * c + 2 * d
    grid = np.full((side, side), background, dtype=np.int8)
    for start in (c, 2 * c + d):
        grid[start:start + d, :] = divider
        grid[:, start:start + d] = divider
    return grid


def _draw(grid, place, color, c, d, offsets):
    starts = _starts(c, d)
    row0 = starts[place[0]]
    col0 = starts[place[1]]
    for dr, dc in offsets:
        grid[row0 + dr, col0 + dc] = color


def _make_pair(direction, terminal, colors, c, d, offsets):
    background, divider, primary, left, right = colors
    source, _, left_place, right_place = _places(direction)
    grid = _blank(c, d, background, divider)
    _draw(grid, (1, 1), primary, c, d, offsets)
    _draw(grid, source, primary, c, d, offsets)
    if terminal:
        _draw(grid, left_place, left, c, d, offsets)
        _draw(grid, right_place, right, c, d, offsets)
        output = np.full((16, 16), background, dtype=np.int8)
    else:
        output = grid.copy()
        _draw(output, left_place, left, c, d, offsets)
        _draw(output, right_place, right, c, d, offsets)
    return {"input": grid, "output": output}


def _schedule(seed, rng, nt, ne):
    mode = seed % 5
    edges = (1, 3, 5, 7)
    corners = (0, 2, 4, 6)
    if mode == 0:
        anchor = [5, 2]
        first_test = 3
    else:
        edge = int(rng.choice(edges))
        corner = int(rng.choice(corners))
        anchor = [edge, corner]
        if mode == 1:  # familiar direction, new geometry
            first_test = int(rng.choice(anchor))
        elif mode == 2:  # cardinal direction absent from training
            first_test = int(rng.choice([x for x in edges if x != edge]))
        elif mode == 3:  # diagonal direction absent from training
            first_test = int(rng.choice([x for x in corners if x != corner]))
        else:  # a completed test; alternate corner and edge sources
            pool = corners if (seed // 5) % 2 == 0 else edges
            first_test = int(rng.choice(pool))
    blocked = first_test if mode in (0, 2, 3) else None
    train = [(anchor[0], False), (anchor[1], False)]
    if nt >= 3:
        terminal_pool = [x for x in range(8) if x != first_test]
        train.append((int(rng.choice(terminal_pool)), True))
    while len(train) < nt:
        pool = [x for x in range(8) if x != blocked]
        train.append((int(rng.choice(pool)), False))
    test = [(first_test, mode == 4 and nt >= 3)]
    remaining = [x for x in range(8) if x != first_test]
    rng.shuffle(remaining)
    for direction in remaining[:ne - 1]:
        test.append((direction, False))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _sample_pair_colors(rng, count, left, right):
    available = [x for x in range(10) if x not in (left, right)]
    triples = list(itertools.permutations(available, 3))
    rng.shuffle(triples)
    first = triples.pop()
    second_index = next(i for i, t in enumerate(triples)
                        if all(t[j] != first[j] for j in range(3)))
    second = triples.pop(second_index)
    return [first, second] + triples[:count - 2]


def _sample_geometry(rng, count):
    geometries = list(_GEOMETRIES)
    rng.shuffle(geometries)
    # Include source-scale cases regularly, while every pair still has its
    # own independent compartment size and divider width.
    if rng.random() < 0.3:
        geometries.remove((3, 1))
        geometries.insert(0, (3, 1))
    return geometries[:count]


def _sample_offsets(rng, c):
    name = str(rng.choice(tuple(_MOTIFS)))
    motif = _MOTIFS[name]
    height = 1 + max(r for r, _ in motif)
    width = 1 + max(col for _, col in motif)
    row = int(rng.integers(0, c - height + 1))
    col = int(rng.integers(0, c - width + 1))
    return tuple((r + row, x + col) for r, x in motif)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, ne = _resolve_counts(seed, rng, num_train, num_test)
    left, right = (int(x) for x in rng.choice(10, size=2, replace=False))
    train_configs, test_configs = _schedule(seed, rng, nt, ne)
    configs = train_configs + test_configs
    triples = _sample_pair_colors(rng, len(configs), left, right)
    geometries = _sample_geometry(rng, len(configs))
    pairs = []
    for (direction, terminal), triple, (c, d) in zip(configs, triples, geometries):
        offsets = _sample_offsets(rng, c)
        # Avoid the four exact original pairs while retaining the original
        # dimensions, palette, motif, and direction as available choices.
        if (triple + (left, right) == _SOURCE_COLORS and c == 3 and d == 1
                and offsets == _MOTIFS["diamond"] and
                (direction, terminal) in ((5, False), (2, False), (0, True), (3, False))):
            offsets = _MOTIFS["square"]
        pairs.append(_make_pair(direction, terminal,
                                triple + (left, right), c, d, offsets))
    puzzle = {"train": pairs[:nt], "test": pairs[nt:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _decode(grid):
    assert grid.ndim == 2 and grid.shape[0] == grid.shape[1]
    side = grid.shape[0]
    candidates = []
    for c, d in _GEOMETRIES:
        if side != 3 * c + 2 * d:
            continue
        divider = int(grid[c, 0])
        bands = (slice(c, c + d), slice(2 * c + d, 2 * c + 2 * d))
        if not all(np.all(grid[band, :] == divider) and np.all(grid[:, band] == divider)
                   for band in bands):
            continue
        starts = _starts(c, d)
        blocks = [grid[r:r+c, x:x+c] for r in starts for x in starts]
        values, counts = np.unique(np.concatenate([b.flat for b in blocks]), return_counts=True)
        background = int(values[int(np.argmax(counts))])
        if background == divider:
            continue
        center = blocks[4]
        active = tuple((int(r), int(x)) for r, x in np.argwhere(center != background))
        if len(active) != 4:
            continue
        primary = int(center[active[0]])
        if primary == divider or any(int(center[pos]) != primary for pos in active):
            continue
        legal = False
        for motif in _MOTIFS.values():
            for dr in range(c - max(r for r, _ in motif)):
                for dc in range(c - max(x for _, x in motif)):
                    if set(active) == {(r + dr, x + dc) for r, x in motif}:
                        legal = True
        if not legal:
            continue
        occupancy = {}
        valid = True
        for i, block in enumerate(blocks):
            other = set(map(tuple, np.argwhere(block != background)))
            if not other:
                continue
            if other != set(active):
                valid = False
                break
            color = int(block[active[0]])
            if color in (background, divider) or any(int(block[pos]) != color for pos in active):
                valid = False
                break
            occupancy[(i // 3, i % 3)] = color
        if valid:
            candidates.append((c, d, background, divider, primary, active, occupancy))
    assert len(candidates) == 1, "board geometry or token shape is invalid"
    c, d, background, divider, primary, offsets, occupancy = candidates[0]
    primary_places = {p for p, color in occupancy.items() if color == primary}
    assert (1, 1) in primary_places and len(primary_places) == 2
    source = next(p for p in primary_places if p != (1, 1))
    assert source in _RING_INDEX
    direction = _RING_INDEX[source]
    _, target, left_place, right_place = _places(direction)
    assert target not in occupancy
    if len(occupancy) == 2:
        assert set(occupancy) == primary_places
        state = "incomplete"
        left = right = None
    else:
        assert len(occupancy) == 4
        assert set(occupancy) == primary_places | {left_place, right_place}
        left = occupancy[left_place]
        right = occupancy[right_place]
        assert len({background, divider, primary, left, right}) == 5
        state = "terminal"
    return {"c": c, "d": d, "background": background, "divider": divider,
            "primary": primary, "offsets": offsets, "occupancy": occupancy,
            "direction": direction, "state": state, "left": left, "right": right,
            "left_place": left_place, "right_place": right_place}


def _inferred_colors(train):
    colors = set()
    incomplete_directions = set()
    for pair in train:
        info = _decode(pair["input"])
        if info["state"] != "incomplete":
            continue
        incomplete_directions.add(info["direction"])
        out = _decode(pair["output"])
        assert out["state"] == "terminal"
        assert (out["c"], out["d"], out["offsets"], out["background"],
                out["divider"], out["primary"]) == (info["c"], info["d"],
                info["offsets"], info["background"], info["divider"], info["primary"])
        assert all(out["occupancy"].get(p) == color for p, color in info["occupancy"].items())
        colors.add((out["left"], out["right"]))
    assert len(incomplete_directions) >= 2
    assert any(x % 2 for x in incomplete_directions)
    assert any(x % 2 == 0 for x in incomplete_directions)
    assert len(colors) == 1, "secondary colors must be consistent"
    return next(iter(colors))


def _expected(grid, info, left, right):
    assert len({info["background"], info["divider"], info["primary"], left, right}) == 5
    if info["state"] == "terminal":
        assert (info["left"], info["right"]) == (left, right)
        return np.full((4 * len(info["occupancy"]),) * 2,
                       info["background"], dtype=grid.dtype)
    output = grid.copy()
    _draw(output, info["left_place"], left, info["c"], info["d"], info["offsets"])
    _draw(output, info["right_place"], right, info["c"], info["d"], info["offsets"])
    return output


def _source_pairs():
    return [_make_pair(direction, terminal, _SOURCE_COLORS, 3, 1,
                       _MOTIFS["diamond"])
            for direction, terminal in ((5, False), (2, False), (0, True), (3, False))]


def _validate_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    pairs = puzzle["train"] + puzzle["test"]
    input_keys = set()
    pair_keys = set()
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
        ik = _grid_key(pair["input"])
        pk = (ik, _grid_key(pair["output"]))
        assert ik not in input_keys and pk not in pair_keys
        input_keys.add(ik)
        pair_keys.add(pk)
        assert not np.array_equal(pair["input"], pair["output"])
    left, right = _inferred_colors(puzzle["train"])
    infos = {split: [] for split in ("train", "test")}
    color_triples = set()
    geometries = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _decode(pair["input"])
            assert np.array_equal(pair["output"], _expected(pair["input"], info, left, right))
            infos[split].append(info)
            triple = (info["background"], info["divider"], info["primary"])
            assert triple not in color_triples
            color_triples.add(triple)
            geometry = (info["c"], info["d"])
            assert geometry not in geometries
            geometries.add(geometry)
    assert sum(info["state"] == "terminal" for info in infos["train"]) <= 1
    assert sum(info["state"] == "incomplete" for info in infos["train"]) >= 2
    if any(info["state"] == "terminal" for info in infos["test"]):
        assert any(info["state"] == "terminal" for info in infos["train"])
    if ne >= 2:
        assert any(info["state"] == "incomplete" for info in infos["test"])
    for pair in pairs:
        for original in _source_pairs():
            assert not (_grid_key(pair["input"]) == _grid_key(original["input"])
                        and _grid_key(pair["output"]) == _grid_key(original["output"]))
    return infos, (left, right)


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed puzzle: {exc}") from exc


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, motifs, directions, states, role_pairs = set(), set(), set(), set(), set()
    heldout_cardinal = heldout_diagonal = familiar = 0
    completed_corner = completed_edge = 0
    for puzzle in puzzles:
        infos, roles = _validate_impl(puzzle)
        role_pairs.add(roles)
        train_dirs = {i["direction"] for i in infos["train"]}
        for split in ("train", "test"):
            for info in infos[split]:
                shapes.add((info["c"], info["d"]))
                motifs.add(tuple((r-min(x for x,_ in info["offsets"]),
                                    c-min(x for _,x in info["offsets"]))
                                   for r,c in info["offsets"]))
                directions.add(info["direction"])
                states.add(info["state"])
        for info in infos["test"]:
            direction = info["direction"]
            if direction in train_dirs:
                familiar += 1
            elif direction % 2:
                heldout_cardinal += 1
            else:
                heldout_diagonal += 1
            if info["state"] == "terminal":
                if direction % 2:
                    completed_edge += 1
                else:
                    completed_corner += 1
    assert len(shapes) >= 15 and len(motifs) == len(_MOTIFS)
    assert directions == set(range(8)) and states == {"incomplete", "terminal"}
    assert len(role_pairs) >= 20
    assert min(heldout_cardinal, heldout_diagonal, familiar) >= 3
    assert completed_corner >= 2 and completed_edge >= 2
    for offset, (nt, ne) in enumerate(((2,1), (6,2), (10,1), (10,4))):
        p = generate_puzzle(1000 + offset, nt, ne)
        assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for offset, (nt, ne) in enumerate(((None,1), (None,4), (2,None), (10,None))):
        p = generate_puzzle(2000 + offset, nt, ne)
        if nt is not None:
            assert len(p["train"]) == nt
        if ne is not None:
            assert len(p["test"]) == ne
    assert _puzzle_key(generate_puzzle(314159,10,4)) == _puzzle_key(generate_puzzle(314159,10,4))
    assert _puzzle_key(generate_puzzle(314159,10,4)) != _puzzle_key(generate_puzzle(314160,10,4))
    for kwargs in ({"num_train":1}, {"num_train":11}, {"num_test":0},
                   {"num_test":5}, {"num_train":2,"num_test":2},
                   {"num_train":3,"num_test":4}):
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
