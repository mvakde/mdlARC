"""Generate motif expansion with a copy beyond each occupied edge cell."""

import numpy as np


_SHAPES = ((2, 3), (3, 2), (2, 4), (4, 2),
           (3, 3), (3, 4), (4, 3), (4, 4))
_SOURCE_INPUTS = (
    np.array([[0, 5, 0], [5, 5, 0], [0, 0, 5]], dtype=np.int8),
    np.array([[0, 5, 0], [5, 5, 5], [0, 5, 0]], dtype=np.int8),
    np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.int8),
    np.array([[0, 5, 0], [0, 5, 0], [5, 0, 5]], dtype=np.int8),
)


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _optional_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _optional_count("num_train", num_train)
    test = _optional_count("num_test", num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
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


def _edge_counts(mask):
    return (int(mask[0].sum()), int(mask[-1].sum()),
            int(mask[:, 0].sum()), int(mask[:, -1].sum()))


def _tile(grid, r, c, h, w):
    return grid[r*h:(r+1)*h, c*w:(c+1)*w]


def _construct_output(input_grid, background, active):
    h, w = input_grid.shape
    mask = input_grid == active
    output = np.full((h*(h+2), w*(w+2)), background, dtype=np.int8)
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                _tile(output, r+1, c+1, h, w)[:] = active
    for c in range(w):
        if mask[0, c]:
            _tile(output, 0, c+1, h, w)[:] = input_grid
        if mask[-1, c]:
            _tile(output, h+1, c+1, h, w)[:] = input_grid
    for r in range(h):
        if mask[r, 0]:
            _tile(output, r+1, 0, h, w)[:] = input_grid
        if mask[r, -1]:
            _tile(output, r+1, w+1, h, w)[:] = input_grid
    return output


def _stabilizer_names(mask):
    transforms = {"identity": mask, "rotate_180": np.rot90(mask, 2),
                  "flip_lr": np.fliplr(mask), "flip_ud": np.flipud(mask)}
    if mask.shape[0] == mask.shape[1]:
        transforms.update({"rotate_90": np.rot90(mask),
                           "rotate_270": np.rot90(mask, 3),
                           "transpose": mask.T,
                           "anti_transpose": np.rot90(mask, 2).T})
    return {name for name, x in transforms.items()
            if x.shape == mask.shape and np.array_equal(mask, x)}


def _stabilizers(mask):
    return len(_stabilizer_names(mask))


def _canonical(mask):
    transforms = [mask, np.rot90(mask, 2), np.fliplr(mask), np.flipud(mask)]
    if mask.shape[0] == mask.shape[1]:
        transforms += [np.rot90(mask), np.rot90(mask, 3), mask.T,
                       np.rot90(mask, 2).T]
    return min(x.tobytes() for x in transforms if x.shape == mask.shape)


def _is_hollow(mask):
    rows, cols = np.where(mask)
    if not len(rows):
        return False
    sub = mask[min(rows):max(rows)+1, min(cols):max(cols)+1]
    return (sub.shape[0] >= 3 and sub.shape[1] >= 3 and
            sub[0].all() and sub[-1].all() and
            sub[:, 0].all() and sub[:, -1].all() and
            not sub[1:-1, 1:-1].any())


def _draw_mask(rng, h, w):
    """Several motif geometries, including free boundary/interior sampling."""
    style = int(rng.integers(10))
    mask = np.zeros((h, w), dtype=bool)
    if style == 0:
        mask = rng.random((h, w)) < rng.uniform(.22, .78)
    elif style == 1:  # connected growth
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, c] = True
        target = int(rng.integers(3, h*w))
        while int(mask.sum()) < target:
            frontier = [(y, x) for y in range(h) for x in range(w)
                        if not mask[y, x] and any(
                            0 <= y+dy < h and 0 <= x+dx < w and mask[y+dy, x+dx]
                            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
            y, x = frontier[int(rng.integers(len(frontier)))]
            mask[y, x] = True
    elif style == 2:  # stem or cross
        if rng.integers(2):
            mask[int(rng.integers(h)), :] = True
            if rng.integers(2):
                mask[:, int(rng.integers(w))] = True
        else:
            mask[:, int(rng.integers(w))] = True
            if rng.integers(2):
                mask[int(rng.integers(h)), :] = True
    elif style == 3:  # bent line, partial cross
        r, c = int(rng.integers(h)), int(rng.integers(w))
        mask[r, c] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if rng.integers(3):
                for step in range(1, int(rng.integers(1, max(h, w)))+1):
                    y, x = r+dy*step, c+dx*step
                    if 0 <= y < h and 0 <= x < w:
                        mask[y, x] = True
    elif style == 4:  # compact block
        rh, rw = int(rng.integers(2, h+1)), int(rng.integers(2, w+1))
        r, c = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        mask[r:r+rh, c:c+rw] = True
    elif style == 5:  # outline
        r0, c0 = int(rng.integers(max(1, h-2))), int(rng.integers(max(1, w-2)))
        r1 = int(rng.integers(r0+2, h+1))
        c1 = int(rng.integers(c0+2, w+1))
        mask[r0, c0:c1] = mask[r1-1, c0:c1] = True
        mask[r0:r1, c0] = mask[r0:r1, c1-1] = True
    elif style == 6:  # checker with small irregularities
        y, x = np.indices((h, w))
        mask = (y+x+int(rng.integers(2))) % 2 == 0
        for _ in range(int(rng.integers(0, max(2, h*w//4)))):
            mask[int(rng.integers(h)), int(rng.integers(w))] ^= True
    elif style == 7:  # dense with holes
        mask = rng.random((h, w)) < rng.uniform(.62, .86)
    elif style == 8:  # sparse fragments
        count = int(rng.integers(2, max(3, h*w//3+1)))
        mask.flat[rng.choice(h*w, count, replace=False)] = True
    else:  # independent boundary and interior densities
        mask = rng.random((h, w)) < rng.uniform(.16, .85)
        if h > 2 and w > 2:
            mask[1:-1, 1:-1] = rng.random((h-2, w-2)) < rng.uniform(.12, .9)
    return mask


def _sample_mask(rng, shape, used, predicate):
    h, w = shape
    for attempt in range(5000):
        mask = _draw_mask(rng, h, w)
        count = int(mask.sum())
        if count < 2 or count >= h*w or not predicate(mask):
            continue
        key = (shape, mask.tobytes())
        if key in used["raw"]:
            continue
        canon = (shape, _canonical(mask))
        if attempt < 900 and canon in used["canonical"]:
            continue
        used["raw"].add(key)
        used["canonical"].add(canon)
        return mask
    # Exact bounded fallback for a rare narrow predicate; no legal seed depends
    # on luck in the rejection sampler.
    candidates = []
    for bits in range(1, (1 << (h*w))-1):
        mask = np.array([(bits >> k) & 1 for k in range(h*w)],
                        dtype=bool).reshape(h, w)
        if (shape, mask.tobytes()) not in used["raw"] and predicate(mask):
            candidates.append(mask)
    if not candidates:
        raise ValueError(f"no distinct motif satisfies {shape} constraints")
    mask = candidates[int(rng.integers(len(candidates)))]
    used["raw"].add((shape, mask.tobytes()))
    used["canonical"].add((shape, _canonical(mask)))
    return mask


def _sample_source_mask(rng, used, predicate):
    candidates = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> k) & 1 for k in range(9)],
                        dtype=bool).reshape(3, 3)
        if ((3, 3), mask.tobytes()) in used["raw"]:
            continue
        if predicate(mask):
            candidates.append(mask)
    if not candidates:
        raise ValueError("source-like configuration exhausted")
    mask = candidates[int(rng.integers(len(candidates)))]
    used["raw"].add(((3, 3), mask.tobytes()))
    used["canonical"].add(((3, 3), _canonical(mask)))
    return mask


def _make_pair(mask, background, active):
    grid = np.where(mask, active, background).astype(np.int8)
    return {"input": grid, "output": _construct_output(grid, background, active)}


def _source_input(grid):
    return any(np.array_equal(grid, source) for source in _SOURCE_INPUTS)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    total = train_count + test_count

    # A shared color anchors one semantic role as the other changes per pair.
    fixed_background = bool(rng.integers(2))
    shared_color = int(rng.integers(10))
    variable_colors = [int(x) for x in rng.permutation(
        [c for c in range(10) if c != shared_color])]
    colors = [variable_colors[i % len(variable_colors)] for i in range(total)]

    # A small training set can mirror the original held-out doubling. Larger
    # training sets draw from the broader rule so they need not recycle 3x3s.
    source_like = 3 <= train_count <= 4 and bool(rng.random() < .22)
    paired_interior = train_count >= 3 and not source_like and rng.random() < .48
    mixed_shapes = not source_like and rng.random() < .58
    if source_like:
        shapes = [(3, 3)] * total
    elif mixed_shapes:
        order = [tuple(x) for x in np.array(_SHAPES)[rng.permutation(len(_SHAPES))]]
        if paired_interior:
            order = [s for s in order if min(s) >= 3] + [s for s in order if min(s) == 2]
        shapes = [order[0], order[1]]
        for i in range(2, train_count):
            shapes.append(order[0] if paired_interior and i == 2 else
                          order[int(rng.integers(len(order)))])
        unseen = [s for s in order if s not in shapes]
        for _ in range(test_count):
            shapes.append(unseen[int(rng.integers(len(unseen)))]
                          if unseen and rng.random() < .55 else
                          order[int(rng.integers(len(order)))])
    else:
        allowed = [s for s in _SHAPES if min(s) >= 3] if paired_interior else _SHAPES
        shape = allowed[int(rng.integers(len(allowed)))]
        shapes = [shape] * total

    used = {"raw": set(), "canonical": set()}
    masks = []
    for i, shape in enumerate(shapes):
        if i == 2 and paired_interior:
            first = masks[0]
            mask = first.copy()
            interior = mask[1:-1, 1:-1]
            positions = rng.choice(interior.size,
                                   size=int(rng.integers(1, interior.size+1)),
                                   replace=False)
            interior.flat[positions] = ~interior.flat[positions]
            if mask.all():
                interior.flat[0] = False
            used["raw"].add((shape, mask.tobytes()))
            used["canonical"].add((shape, _canonical(mask)))
            masks.append(mask)
            continue
        if source_like and i == 0:
            predicate = lambda m: _edge_counts(m) == (1, 1, 1, 1) and _stabilizers(m) == 2
        elif source_like and i == 1:
            first = masks[0]
            predicate = lambda m, f=first: (_edge_counts(m) == (1, 1, 1, 1)
                and int(m.sum()) != int(f.sum()) and
                _stabilizer_names(m).intersection(_stabilizer_names(f)) == {"identity"})
        elif source_like and i < train_count:
            predicate = lambda m: _edge_counts(m) == (1, 1, 1, 1)
        elif source_like:
            predicate = lambda m: sorted(_edge_counts(m)) == [1, 1, 1, 2]
        elif i == 0:
            predicate = lambda m: (min(_edge_counts(m)) >= 1 and
                                   max(_edge_counts(m)) >= 2 and _stabilizers(m) == 1)
        elif i == 1:
            first = masks[0]
            if shape == shapes[0]:
                predicate = lambda m, f=first: (0 in _edge_counts(m) and
                    int(m.sum()) != int(f.sum()) and
                    np.array_equal(m[1:-1, 1:-1], f[1:-1, 1:-1]))
            else:
                predicate = lambda m, f=first: (0 in _edge_counts(m) and
                    int(m.sum()) != int(f.sum()))
        elif i == 3 and train_count >= 4 and min(shape) >= 3 and rng.random() < .55:
            predicate = _is_hollow
        elif i >= train_count and rng.random() < .7:
            if max(shape) == 4 and rng.random() < .4 and train_count >= 3:
                predicate = lambda m: max(_edge_counts(m)) >= 3
            else:
                predicate = lambda m: sum(x >= 2 for x in _edge_counts(m)) >= 2
        else:
            predicate = lambda m: True
        masks.append((_sample_source_mask(rng, used, predicate) if source_like
                      else _sample_mask(rng, shape, used, predicate)))

    pairs = []
    for i, mask in enumerate(masks):
        background, active = ((shared_color, colors[i]) if fixed_background
                              else (colors[i], shared_color))
        if _source_input(np.where(mask, active, background)):
            replacements = [c for c in range(10) if c != shared_color
                            and c not in colors]
            if not replacements:
                replacements = [c for c in range(10) if c not in
                                (shared_color, colors[i]) and c not in colors[:i]]
            if not replacements:
                replacements = [c for c in range(10) if c not in
                                (shared_color, colors[i])]
            colors[i] = replacements[int(rng.integers(len(replacements)))]
            background, active = ((shared_color, colors[i]) if fixed_background
                                  else (colors[i], shared_color))
        pairs.append(_make_pair(mask, background, active))
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, grid.tobytes())


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "puzzle must contain train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen = set()
    infos = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "pair must have input and output")
            grid, output = pair["input"], pair["output"]
            for x in (grid, output):
                _check(isinstance(x, np.ndarray) and x.ndim == 2 and
                       np.issubdtype(x.dtype, np.integer) and
                       1 <= x.shape[0] <= 30 and 1 <= x.shape[1] <= 30 and
                       np.all((x >= 0) & (x <= 9)), "invalid ARC grid")
            h, w = grid.shape
            _check((h, w) in _SHAPES and output.shape == (h*(h+2), w*(w+2)),
                   "incorrect grid dimensions")
            _check(len(np.unique(grid)) == 2 and
                   np.array_equal(np.unique(grid), np.unique(output)),
                   "incorrect color roles")
            background = int(output[0, 0])
            _check(background in grid, "background is absent from input")
            active = next(int(c) for c in np.unique(grid) if c != background)
            mask = grid == active
            _check(mask.sum() >= 2 and (~mask).any(), "degenerate motif")
            _check(np.array_equal(output, _construct_output(grid, background, active)),
                   "incorrect motif expansion or edge copy")
            _check(not _source_input(grid), "exact source input duplicated")
            key = _grid_key(grid)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            infos.append((split, (h, w), background, active, mask))

    train = [x for x in infos if x[0] == "train"]
    backgrounds = {x[2] for x in infos}
    actives = {x[3] for x in infos}
    _check(len(backgrounds) == 1 or len(actives) == 1,
           "no consistent puzzle-level color role")
    _check(len({x[2] for x in train}) > 1 or len({x[3] for x in train}) > 1,
           "training does not demonstrate recoloring")
    common_stabilizers = set.intersection(*[_stabilizer_names(x[4]) for x in train])
    _check(common_stabilizers == {"identity"},
           "training does not reveal copy orientation")
    _check(all(any(_edge_counts(x[4])[side] > 0 for x in train)
               for side in range(4)), "training omits a copy direction")
    _check(len({int(x[4].sum()) for x in train}) >= 2,
           "training lacks occupancy variation")
    train_edges = [_edge_counts(x[4]) for x in train]
    test_edges = [_edge_counts(x[4]) for x in infos if x[0] == "test"]
    if all(edge == (1, 1, 1, 1) for edge in train_edges):
        _check(3 <= len(train) <= 4 and
               all(sorted(edge) == [1, 1, 1, 2] for edge in test_edges),
               "source-like train/test progression is incomplete")
    else:
        _check(any(max(edge) >= 2 for edge in train_edges),
               "training omits per-cell repeated-edge evidence")


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    reached = {next(iter(cells))}
    todo = list(reached)
    while todo:
        r, c = todo.pop()
        for p in ((r+1, c), (r-1, c), (r, c+1), (r, c-1)):
            if p in cells and p not in reached:
                reached.add(p)
                todo.append(p)
    return len(reached) == len(cells)


def _check_tiles_independently(pair):
    grid, output = pair["input"], pair["output"]
    h, w = grid.shape
    background = int(output[0, 0])
    active = next(int(c) for c in np.unique(grid) if c != background)
    mask = grid == active
    for r in range(h+2):
        for c in range(w+2):
            block = _tile(output, r, c, h, w)
            if 1 <= r <= h and 1 <= c <= w:
                expected = active if mask[r-1, c-1] else background
            elif r == 0 and 1 <= c <= w and mask[0, c-1]:
                expected = grid
            elif r == h+1 and 1 <= c <= w and mask[-1, c-1]:
                expected = grid
            elif c == 0 and 1 <= r <= h and mask[r-1, 0]:
                expected = grid
            elif c == w+1 and 1 <= r <= h and mask[r-1, -1]:
                expected = grid
            else:
                expected = background
            _check(np.all(block == expected), "independent tile check failed")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, palettes, edge_profiles = set(), set(), set()
    connected = dense = zero_sides = busy_sides = triple_sides = 0
    adjacent = separated = corner_active = 0
    mixed = recolored = boundary_contrasts = interior_contrasts = 0
    new_test_sizes = source_like_puzzles = 0
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "automatic pair counts lack diversity")
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = puzzle["train"]
        shape_set, palette_set, train_masks = set(), set(), []
        train_shapes = {p["input"].shape for p in train}
        new_test_sizes += any(p["input"].shape not in train_shapes
                              for p in puzzle["test"])
        train_edge_counts = []
        test_edge_counts = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _check_tiles_independently(pair)
                grid, output = pair["input"], pair["output"]
                background = int(output[0, 0])
                active = next(int(x) for x in np.unique(grid) if x != background)
                mask = grid == active
                counts = _edge_counts(mask)
                shapes.add(grid.shape)
                palettes.add((background, active))
                edge_profiles.add(counts)
                shape_set.add(grid.shape)
                palette_set.add((background, active))
                connected += _connected(mask)
                dense += int(mask.mean() >= .65)
                zero_sides += int(0 in counts)
                busy_sides += int(sum(x >= 2 for x in counts) >= 2)
                triple_sides += int(max(counts) >= 3)
                corner_active += int(mask[0, 0] or mask[0, -1] or
                                     mask[-1, 0] or mask[-1, -1])
                for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
                    positions = np.flatnonzero(edge)
                    if len(positions) >= 2:
                        adjacent += int(np.any(np.diff(positions) == 1))
                        separated += int(np.any(np.diff(positions) > 1))
                if split == "train":
                    train_masks.append(mask)
                    train_edge_counts.append(counts)
                else:
                    test_edge_counts.append(counts)
        source_like_puzzles += (all(x == (1, 1, 1, 1)
                                    for x in train_edge_counts) and
                                all(sorted(x) == [1, 1, 1, 2]
                                    for x in test_edge_counts))
        mixed += len(shape_set) > 1
        recolored += len(palette_set) > 1
        for i, a in enumerate(train_masks):
            for b in train_masks[i+1:]:
                if a.shape != b.shape:
                    continue
                same_boundary = (np.array_equal(a[0], b[0]) and
                                 np.array_equal(a[-1], b[-1]) and
                                 np.array_equal(a[:, 0], b[:, 0]) and
                                 np.array_equal(a[:, -1], b[:, -1]))
                interior_contrasts += int(same_boundary and not np.array_equal(a, b))
                boundary_contrasts += int(not same_boundary and
                    np.array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1]))
    _check(shapes == set(_SHAPES), "size range was not exercised")
    _check(len(palettes) >= 25 and len(edge_profiles) >= 30,
           "palette or boundary diversity collapsed")
    _check(mixed >= 8 and recolored == 42,
           "within-puzzle size or color diversity collapsed")
    _check(new_test_sizes >= 6 and source_like_puzzles >= 1,
           "size transfer or original train/test shift absent")
    _check(connected >= 45 and dense >= 25 and zero_sides >= 20 and
           busy_sides >= 25 and triple_sides >= 8,
           "motif or boundary diversity collapsed")
    _check(adjacent >= 25 and separated >= 15 and corner_active >= 25,
           "edge spacing or corner diversity collapsed")
    _check(boundary_contrasts >= 8 and interior_contrasts >= 2,
           "controlled training contrasts were not exercised")
    for args in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2)):
        p = generate_puzzle(*args)
        _check((len(p["train"]), len(p["test"])) == args[1:],
               "explicit counts not respected")
    for seed, requested_train, requested_test in (
            (201, None, 1), (202, None, 4),
            (203, 2, None), (204, 10, None)):
        p = generate_puzzle(seed, requested_train, requested_test)
        if requested_train is not None:
            _check(len(p["train"]) == requested_train,
                   "partial num_train not respected")
        if requested_test is not None:
            _check(len(p["test"]) == requested_test,
                   "partial num_test not respected")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            continue
        raise AssertionError("illegal pair count accepted")
    a, b = generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3)
    _check(all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output")),
           "fixed seed is not deterministic")
    c = generate_puzzle(778, 7, 3)
    _check(any(not np.array_equal(x["input"], y["input"])
               for split in ("train", "test")
               for x, y in zip(a[split], c[split])),
           "different seeds produced the same puzzle")
