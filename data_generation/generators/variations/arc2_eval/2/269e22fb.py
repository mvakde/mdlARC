"""Recognize exact crops of a reference learned from the training outputs.

The reference is composed once per puzzle. Only D4 isometries, bijective
recolorings, and rectangular cropping are allowed between demonstrations.
"""
import numpy as np


def _transform(grid: np.ndarray, index: int) -> np.ndarray:
    if index == 0:
        result = grid
    elif index == 1:
        result = np.rot90(grid, 1)
    elif index == 2:
        result = np.rot90(grid, 2)
    elif index == 3:
        result = np.rot90(grid, 3)
    elif index == 4:
        result = np.fliplr(grid)
    elif index == 5:
        result = np.flipud(grid)
    elif index == 6:
        result = grid.T
    elif index == 7:
        result = np.flipud(np.fliplr(grid.T))
    else:
        raise ValueError("invalid dihedral transform index")

    return np.asarray(result, dtype=np.int8).copy()


def _colorize(
    binary: np.ndarray,
    palette: tuple[int, int],
    swap: int,
) -> np.ndarray:
    if swap:
        role_colors = (palette[1], palette[0])
    else:
        role_colors = palette

    return np.where(
        binary == 0,
        role_colors[0],
        role_colors[1],
    ).astype(np.int8)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        grid.astype(np.int8, copy=False).tobytes(),
    )


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} is outside its legal range")

    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        legal_defaults = (
            (2, 1),
            (3, 1),
            (4, 1),
            (4, 2),
            (5, 2),
            (6, 3),
            (7, 1),
            (7, 3),
            (8, 2),
            (8, 4),
            (9, 3),
            (10, 4),
        )
        index = int(rng.integers(0, len(legal_defaults)))
        return legal_defaults[index]

    if train_count is None:
        assert test_count is not None
        choices = np.arange(
            max(2, test_count + 1),
            11,
            dtype=np.int64,
        )
        if len(choices) == 0:
            raise ValueError(
                "no legal num_train for the requested num_test"
            )

        index = int(rng.integers(0, len(choices)))
        return int(choices[index]), test_count

    choices = np.arange(
        1,
        min(4, train_count - 1) + 1,
        dtype=np.int64,
    )
    if len(choices) == 0:
        raise ValueError(
            "no legal num_test for the requested num_train"
        )

    index = int(rng.integers(0, len(choices)))
    return train_count, int(choices[index])


# Used only for exact source-copy exclusion, never to construct or solve a pair.
_SOURCE_PAIRS = {
    ((8, 10, '77888888887777778888777777788877888777888888887778888888877788888888778888888887'), (20, 20, '7788888888888887788877777788888888877888777777788888888777887788877788888887778888888877788888777778888888877788887887788888888877887778877888888888878878788778888888888777787887788777777777877878877887888888877778788778877777777788787887788787888887887778877877787777778888788778878778888788887777787778788887888778778887877888878887888778777878888788877888778787788878788878777877788887888788777888')),
    ((16, 8, '88788878888787887888788878887888788878888888788888887777778878888788777787777888877877778777788887887888778877888888777888888777'), (20, 20, '8887778878887888877787778788878788877878778887788878888787778778887888788887787888778778887888878777877777888878888778788778878888777777877787788777887888887878877887878877777777788778878777788888887887788787787777777778877887877778888888888778878788788888888887788777887788888888877887888877788888888777778888877788888888777888888877788877887778888888877777778887788888888877777788877888888888888877')),
    ((10, 10, '0000000000000033333300333333333333300000333330000000003333330000003000000000333300000000330000000030'), (20, 20, '0000000000000000030000003333333333303330003333333333333330303333300000000033003333333000000000300003000033333333333303330000003000003003330000000033333330000000000000003330000000030000000030300000003000000333333333333300000033300303030000300003330003030300000300333000030303000000033300000303030000000330000003030333333003300000030330303030033000000303030303033333000003333333333333330000000003030303')),
    ((13, 12, '887778888888887778888888877777888887877887888877877887778877877887878878877887877778877887877877877887877778877887878877877887778878877887888877877777888878'), (20, 20, '8887788888888888887788877888888888777777887778888888877777778877788888887778887787777788888777888888877887888877788888888778877788778888888887788787887888888888877887877778888888888778878778777777777887788787777888888878877887878877777777788778877788788888787887788788887777778777877777888878888778788877877888788887877787788878887888877878778887788878888787778777878887878887787888877788788878888777')),
    ((8, 10, '87878888878787888888878788888887877777788778787878878787878777777777778887878787'), (20, 20, '8888888888888888878888887777777777787778887777777777777778787777788888888877887777777888888888788887888877777777777787778888887888887887778888888877777778888888888888887778888888878888888878788888887888888777777777777788888877788787878888788887778887878788888788777888878787888888877788888787878888888778888887878777777887788888878778787878877888888787878787877777888887777777777777778888888887878787')),
    ((10, 10, '8787878777888888888887878787877878787887788888878777777787877777778787877777878778777787877788888888'), (20, 20, '8787878777777777888888888888888777778888878787878787777778877878787887877777788778888887878777777887777777878787777788877777778787877778887787777787878777888777787777878787788877777788888888888887777778777777787877777777877777777888777777777777777888888877777777888778777778777777888788888888888877778777787777777778888888778877777777788888787888888888888888777888788888888888777777877777777777777777')),
    ((11, 10, '66666555555555556666666566666655556666665566666666656666666655555666665665556666566655566656666555665666665556'), (20, 20, '6656666666666666666665556555555555556666656555555555555555665566556666666665555556666566666666655555555655555555555566666655566566666566666666666665555555666666566666666555666666666566666665656666666666555555555555566666656666565656655566665666665656566655566666666656565666655566666666565656666655566555555656566666655665656565565666666556565656565656666665565555555555566666555556565656666666665555'))
}


def _content_key(grid):
    return (grid.shape[0], grid.shape[1], ''.join(str(int(v)) for v in grid.flat))


def _is_source_pair(pair):
    return (_content_key(pair['input']), _content_key(pair['output'])) in _SOURCE_PAIRS


def _binary(grid):
    colors = np.unique(grid)
    assert len(colors) == 2
    return (grid == colors[1]).astype(np.int8)


def _orbit(reference):
    return [_transform(reference, i) for i in range(8)]


def _canonical(reference):
    return min(_grid_key(g if swap == 0 else 1 - g)
               for g in _orbit(reference) for swap in (0, 1))


def _locations(full, patch):
    """All exact offsets, pruning windows a row at a time."""
    h, w = patch.shape
    H, W = full.shape
    if h > H or w > W:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rows, cols = np.nonzero(np.all(windows[:, :, 0, :] == patch[0], axis=-1))
    for r in range(1, h):
        if not len(rows):
            break
        keep = np.all(full[rows[:, None] + r, cols[:, None] + np.arange(w)] == patch[r], axis=1)
        rows, cols = rows[keep], cols[keep]
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _matches(orbit, patch):
    return [(i, swap, r, c)
            for i, full in enumerate(orbit)
            for swap in (0, 1)
            for r, c in _locations(full, patch if swap == 0 else 1 - patch)]


def _classify(orbit, output):
    return [(i, swap) for i, full in enumerate(orbit) for swap in (0, 1)
            if np.array_equal(full if swap == 0 else 1 - full, output)]


def _compose(a, b):
    probe = np.arange(6, dtype=np.int8).reshape(2, 3)
    target = _transform(_transform(probe, b), a)
    return next(i for i in range(8) if np.array_equal(_transform(probe, i), target))


def _subgroup(indices):
    result = {0, *indices}
    while True:
        expanded = result | {_compose(a, b) for a in result for b in result}
        if expanded == result:
            return result
        result = expanded


def _stroke(grid, a, b, thickness):
    # Bresenham-like lattice path: horizontal, vertical, or stepped diagonal.
    count = max(abs(a[0] - b[0]), abs(a[1] - b[1])) + 1
    rows = np.rint(np.linspace(a[0], b[0], count)).astype(int)
    cols = np.rint(np.linspace(a[1], b[1], count)).astype(int)
    for r, c in zip(rows, cols):
        grid[r:min(grid.shape[0], r + thickness), c:min(grid.shape[1], c + thickness)] = 1


def _motif(rng, h, w, kind):
    tile = np.zeros((h, w), dtype=np.int8)
    thickness = int(rng.integers(1, min(3, (min(h, w) - 1) // 2) + 1))
    bottom, right = h - thickness, w - thickness
    if kind == 0:  # long or short bar
        _stroke(tile, (0, 0), (0, right), thickness)
    elif kind == 1:  # closed loop, sometimes with a projecting branch
        for a, b in [((0, 0), (0, right)), ((0, right), (bottom, right)),
                     ((bottom, right), (bottom, 0)), ((bottom, 0), (0, 0))]:
            _stroke(tile, a, b, thickness)
    elif kind == 2:  # bends: an L or a U
        _stroke(tile, (0, 0), (bottom, 0), thickness)
        _stroke(tile, (bottom, 0), (bottom, right), thickness)
        if rng.random() < .5:
            _stroke(tile, (bottom, right), (0, right), thickness)
    elif kind == 3:  # stepped diagonal, optionally changes direction
        middle = int(rng.integers(1, max(2, bottom)))
        end = int(rng.integers(max(1, right // 2), right + 1))
        _stroke(tile, (0, 0), (middle, end), thickness)
        _stroke(tile, (middle, end), (bottom, right if rng.random() < .5 else 0), thickness)
    elif kind == 4:  # branches with independently varying lengths and gaps
        spine = int(rng.integers(0, right + 1))
        _stroke(tile, (0, spine), (bottom, spine), thickness)
        for r in range(1, bottom + 1, int(rng.integers(2, 5))):
            end = int(rng.choice([0, right]))
            _stroke(tile, (r, spine), (r, end), thickness)
    elif kind == 5:  # local repeated runs, ladder/comb/checkered edge
        pitch = thickness + int(rng.integers(1, 4))
        for r in range(0, h, pitch):
            tile[r:r + thickness, :int(rng.integers(max(2, w - 3), w + 1))] = 1
        if rng.random() < .7:
            tile[:, :thickness] = 1
        if rng.random() < .3:
            tile[:, -thickness:] = 1
    else:  # an irregular orthogonal walk
        position = (int(rng.integers(h)), int(rng.integers(w)))
        for step in range(int(rng.integers(3, 7))):
            end = (int(rng.integers(h)), position[1]) if step % 2 else (position[0], int(rng.integers(w)))
            _stroke(tile, position, end, thickness)
            position = end
    tile = _transform(tile, int(rng.choice([0, 2, 4, 5])))
    rows, cols = np.nonzero(tile)
    return tile[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _large_windows(reference, orbit, required=16):
    """Reserve enough distinguishable proper crops for a guaranteed pair fallback."""
    H, W = reference.shape
    bank, seen = [], set()
    for loss in range(1, 7):
        for dh in range(loss + 1):
            dw = loss - dh
            h, w = H - dh, W - dw
            if min(h, w) < 2:
                continue
            for r in range(dh + 1):
                for c in range(dw + 1):
                    patch = reference[r:r + h, c:c + w]
                    key = _grid_key(patch)
                    if key in seen or len(np.unique(patch)) != 2:
                        continue
                    seen.add(key)
                    if len(_matches(orbit, patch)) == 1:
                        bank.append((r, c, h, w))
                        if len(bank) >= required:
                            return bank
    return bank


def _reference(rng):
    H = int(rng.integers(10, 31))
    W = H if rng.random() < .4 else int(rng.integers(10, 31))
    density = int(rng.integers(0, 3))
    for attempt in range(40):
        grid = np.zeros((H, W), dtype=np.int8)
        count = int(rng.integers(4 + density, 8 + 3 * density))
        kinds = list(rng.permutation(7))
        for j in range(count):
            kind = int(kinds[j % 7])
            # Most motifs are local; some span almost a whole axis.
            max_h = H if rng.random() < .3 else max(4, 2 * H // 3)
            max_w = W if rng.random() < .3 else max(4, 2 * W // 3)
            h, w = int(rng.integers(4, max_h + 1)), int(rng.integers(4, max_w + 1))
            tile = _motif(rng, h, w, kind)
            if rng.random() < .5 and tile.shape[1] <= H and tile.shape[0] <= W:
                tile = tile.T
            h, w = tile.shape
            # Avoid filling every hole through excessive overlap, but allow
            # motifs to meet, cross, and form larger recognizable structures.
            for placement in range(12):
                r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
                region = grid[r:r + h, c:c + w]
                if np.sum(region * tile) <= .4 * np.sum(tile):
                    region[:] |= tile
                    break
        if not .16 <= grid.mean() <= .64:
            continue
        orbit = _orbit(grid)
        if len({_grid_key(g if s == 0 else 1 - g) for g in orbit for s in (0, 1)}) != 16:
            continue
        bank = _large_windows(grid, orbit)
        if len(bank) >= 16:
            return grid, orbit, bank
    return _fallback_reference(H, W)


def _fallback_reference(H, W):
    # Deterministic structured fallback, still at the sampled dimensions.
    # Its unequal loops and staircase have no geometric/color symmetry.
    grid = np.zeros((H, W), dtype=np.int8)
    grid[1:H - 2, 1] = 1
    grid[1, 1:W - 2] = 1
    grid[H - 3, 1:W // 2] = 1
    grid[3:H - 2, W // 2] = 1
    grid[3, W // 2:W - 1] = 1
    grid[3:H - 4, W - 2] = 1
    _stroke(grid, (H // 2, 3), (H - 1, W - 1), 1)
    orbit = _orbit(grid)
    bank = _large_windows(grid, orbit)
    assert len(bank) >= 16, 'fallback reference must expose enough proper crops'
    return grid, orbit, bank


def _orientation_schedule(rng, nt, nq):
    regime = int(rng.integers(4))  # constant, shared pool, held out, mixed
    base = int(rng.integers(8))
    if regime == 0:
        return [base] * nt, [base] * nq
    quarter = int(rng.choice([1, 3]))
    # A quarter turn visibly generates C4. A reflection additionally generates
    # D4, but is introduced only when there is room for a third demonstration.
    anchors = [0, quarter]
    if nt >= 3 and rng.random() < .8:
        anchors.append(int(rng.integers(4, 8)))
    family = sorted(_subgroup(anchors))
    if regime == 2:
        pool = list(anchors)
        available = [i for i in family if i not in pool]
        for i in rng.permutation(available):
            if len(pool) < len(family) - 1 and rng.random() < .45:
                pool.append(int(i))
        train = (anchors + [int(rng.choice(pool)) for _ in range(nt)])[:nt]
        test = [int(rng.choice([i for i in family if i not in train])) for _ in range(nq)]
    else:
        train = (anchors + [int(rng.choice(family)) for _ in range(nt)])[:nt]
        if regime == 1:
            test = [int(rng.choice(train)) for _ in range(nq)]
        else:
            test = [int(rng.choice(train))] + [int(rng.choice(family)) for _ in range(nq - 1)]
    rng.shuffle(train)
    rng.shuffle(test)
    return ([_compose(base, i) for i in train], [_compose(base, i) for i in test])


def _palette_schedule(rng, nt, nq):
    regime = int(rng.integers(4))  # stable, exchange, modest renaming, rich renaming
    all_palettes = [(a, b) for a in range(10) for b in range(a + 1, 10)]
    rng.shuffle(all_palettes)
    first_swap = int(rng.integers(2))
    if regime <= 1:
        train = [(all_palettes[0], first_swap)] * nt
        if regime == 1:
            train[1] = (all_palettes[0], 1 - first_swap)
            train[2:] = [(all_palettes[0], int(rng.integers(2))) for _ in range(nt - 2)]
        test = [(all_palettes[0], int(rng.integers(2)) if regime else first_swap) for _ in range(nq)]
    else:
        count = min(nt, int(rng.integers(2, 4) if regime == 2 else rng.integers(3, 8)))
        pool = all_palettes[:count]
        schedule = pool + [pool[int(rng.integers(count))] for _ in range(nt - count)]
        rng.shuffle(schedule)
        swaps = [first_swap, 1 - first_swap] + [int(rng.integers(2)) for _ in range(nt - 2)]
        train = list(zip(schedule, swaps))
        test = []
        for _ in range(nq):
            # New and familiar palettes can occur at any test position. New
            # palettes may share one color or be wholly disjoint from training.
            palette = all_palettes[int(rng.integers(count, 45))] if rng.random() < .5 else pool[int(rng.integers(count))]
            test.append((palette, int(rng.integers(2))))
    rng.shuffle(train)
    return train, test


def _crop_shape(rng, H, W, category):
    if category == 0:  # small fragments
        h, w = int(rng.integers(2, min(H, 8))), int(rng.integers(2, min(W, 8)))
    elif category == 1:  # short/wide or tall/thin, including full-span strips
        if rng.random() < .5:
            h = int(rng.integers(2, min(H, 5)))
            w = W if rng.random() < .45 else int(rng.integers(max(3, W // 2), W + 1))
        else:
            w = int(rng.integers(2, min(W, 5)))
            h = H if rng.random() < .45 else int(rng.integers(max(3, H // 2), H + 1))
    elif category == 2:  # moderate, independently sampled aspect ratios
        h = int(rng.integers(max(3, H // 3), max(4, 4 * H // 5 + 1)))
        w = int(rng.integers(max(3, W // 3), max(4, 4 * W // 5 + 1)))
    elif category == 3:  # almost complete, still requires completion
        h, w = H - int(rng.integers(0, max(2, H // 8))), W - int(rng.integers(0, max(2, W // 8)))
    else:  # broad continuous range, including all former source dimensions
        h, w = int(rng.integers(2, H + 1)), int(rng.integers(2, W + 1))
    if (h, w) == (H, W):
        if rng.random() < .5:
            h -= 1
        else:
            w -= 1
    return h, w


def _offset(rng, R, C):
    mode = int(rng.integers(5))
    r, c = int(rng.integers(R + 1)), int(rng.integers(C + 1))
    if mode == 0:  # corner
        r, c = int(rng.choice([0, R])), int(rng.choice([0, C]))
    elif mode == 1:  # one edge, when possible
        if rng.random() < .5:
            r = int(rng.choice([0, R]))
        else:
            c = int(rng.choice([0, C]))
    elif mode == 2 and R > 1 and C > 1:
        r, c = int(rng.integers(1, R)), int(rng.integers(1, C))
    return r, c


def _generate_pair(rng, reference, orbit, bank, orientation, colors, category, seen):
    palette, swap = colors
    binary = orbit[orientation]
    output = _colorize(binary, palette, swap)
    H, W = binary.shape
    for _ in range(100):
        h, w = _crop_shape(rng, H, W, category)
        r, c = _offset(rng, H - h, W - w)
        patch = binary[r:r + h, c:c + w]
        if len(np.unique(patch)) != 2:
            continue
        inp = _colorize(patch, palette, swap)
        key = _grid_key(inp)
        if key in seen:
            continue
        if len(_matches(orbit, patch)) == 1:
            pair = {'input': inp, 'output': output.copy()}
            if not _is_source_pair(pair):
                seen.add(key)
                return pair
    # This finite bank contains more independently identifying crops than the
    # maximum number of pairs. Transforming one preserves its unique match.
    for index in rng.permutation(len(bank)):
        r, c, h, w = bank[int(index)]
        patch = _transform(reference[r:r + h, c:c + w], orientation)
        inp = _colorize(patch, palette, swap)
        key = _grid_key(inp)
        pair = {'input': inp, 'output': output.copy()}
        if key not in seen and not _is_source_pair(pair):
            seen.add(key)
            return pair
    raise AssertionError('reserved crop bank exhausted')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    nt, nq = _resolve_counts(rng, num_train, num_test)
    reference, orbit, bank = _reference(rng)
    train_orientations, test_orientations = _orientation_schedule(rng, nt, nq)
    train_colors, test_colors = _palette_schedule(rng, nt, nq)
    # The first two demonstrations have different crop families; later crops
    # vary freely. Test sampling emphasizes fragments and strips independently
    # of geometric and palette novelty.
    categories = list(rng.choice(5, 2, replace=False))
    categories += list(rng.choice(5, nt - 2, p=[.16, .22, .32, .10, .20]))
    test_categories = list(rng.choice(5, nq, p=[.25, .35, .18, .08, .14]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, orientations, colors, crops in (
            ('train', train_orientations, train_colors, categories),
            ('test', test_orientations, test_colors, test_categories)):
        for orientation, palette, category in zip(orientations, colors, crops):
            puzzle[split].append(_generate_pair(rng, reference, orbit, bank, orientation, palette, int(category), seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def solve_from_training(train_pairs, test_input):
    """Prediction uses no held-out output, seed, or generation-time metadata."""
    assert train_pairs
    reference = _binary(np.array(train_pairs[0]['output'], copy=True))
    orbit = _orbit(reference)
    for pair in train_pairs:
        assert len(_classify(orbit, _binary(pair['output']))) == 1
    colors = tuple(int(c) for c in np.unique(test_input))
    assert len(colors) == 2
    matches = _matches(orbit, _binary(test_input))
    answers = {_grid_key(_colorize(orbit[i], colors, swap)): (i, swap)
               for i, swap, _, _ in matches}
    assert len(answers) == 1, 'crop does not uniquely determine its completion'
    i, swap = next(iter(answers.values()))
    return _colorize(orbit[i], colors, swap)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen_inputs, seen_pairs = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert len(np.unique(inp)) == 2 and np.array_equal(np.unique(inp), np.unique(out))
            assert all(a <= b for a, b in zip(inp.shape, out.shape)) and inp.size < out.size
            key = (_grid_key(inp), _grid_key(out))
            assert key[0] not in seen_inputs and key not in seen_pairs
            assert not _is_source_pair(pair)
            seen_inputs.add(key[0])
            seen_pairs.add(key)
    reference = _binary(puzzle['train'][0]['output'])
    orbit = _orbit(reference)
    assert len({_grid_key(g if swap == 0 else 1 - g) for g in orbit for swap in (0, 1)}) == 16
    classes, palettes = {}, {}
    for split in ('train', 'test'):
        classes[split], palettes[split] = [], []
        for index, pair in enumerate(puzzle[split]):
            classification = _classify(orbit, _binary(pair['output']))
            assert len(classification) == 1, 'all outputs must share the learned reference'
            matches = _matches(orbit, _binary(pair['input']))
            assert len(matches) == 1, 'require a distinctive crop under all sixteen candidates'
            i, swap, r, c = matches[0]
            assert classification[0] == (i, swap)
            h, w = pair['input'].shape
            assert np.array_equal(pair['output'][r:r + h, c:c + w], pair['input'])
            # Leave each training label out as well. Predictions never receive
            # the queried answer; this also checks that repeated outputs are OK.
            demonstrations = puzzle['train'] if split == 'test' else [p for j, p in enumerate(puzzle['train']) if j != index]
            predicted = solve_from_training(demonstrations, pair['input'].copy())
            assert np.array_equal(predicted, pair['output'])
            classes[split].append((i, swap))
            palettes[split].append(tuple(int(c) for c in np.unique(pair['input'])))
    # Every novel geometry must be generated by operations visibly demonstrated
    # relative to the first training output, not a hidden test-only convention.
    demonstrated = _subgroup({i for i, _ in classes['train']})
    assert {i for i, _ in classes['test']} <= demonstrated
    if len(set(palettes['train'])) == 1:
        assert set(palettes['test']) <= set(palettes['train'])
    if len({swap for _, swap in classes['train']}) == 1:
        assert {swap for _, swap in classes['test']} <= {swap for _, swap in classes['train']}


def _puzzle_signature(puzzle):
    return tuple((split, _grid_key(p['input']), _grid_key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _reference_features(grid):
    mask = _binary(grid)
    if mask.mean() > .5:
        mask = 1 - mask
    H, W = mask.shape
    runs = []
    for a in (mask, mask.T):
        for row in a:
            boundaries = np.diff(np.r_[0, row, 0])
            runs.extend((np.flatnonzero(boundaries == -1) - np.flatnonzero(boundaries == 1)).tolist())
    blocks = int(np.sum(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]))
    neighbors = sum(np.pad(mask,1)[1+dr:1+dr+H,1+dc:1+dc+W] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
    branches = int(np.sum((neighbors >= 3) & (mask == 1)))
    diagonal = int(np.sum((mask[:-1,:-1] & mask[1:,1:]) & ~(mask[1:,:-1] | mask[:-1,1:])))
    diagonal += int(np.sum((mask[:-1,1:] & mask[1:,:-1]) & ~(mask[1:,1:] | mask[:-1,:-1])))
    remaining = set(zip(*np.nonzero(1-mask)))
    holes = 0
    while remaining:
        start = remaining.pop()
        stack, component = [start], [start]
        while stack:
            r,c = stack.pop()
            for nxt in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if nxt in remaining:
                    remaining.remove(nxt); stack.append(nxt); component.append(nxt)
        if all(0<r<H-1 and 0<c<W-1 for r,c in component):
            holes += 1
    repetition = 0
    for a in (mask,mask.T):
        for row in a:
            for c in range(len(row)-5):
                part = row[c:c+6]
                if 0 < part.sum() < 6 and np.array_equal(part[:4],part[2:]):
                    repetition += 1
    return dict(density=round(float(mask.mean()),4),long_runs=sum(v>=6 for v in runs),short_runs=sum(2<=v<=4 for v in runs),
                max_run=max(runs),blocks=blocks,branches=branches,diagonal=diagonal,holes=holes,repetition=repetition)


def measure_puzzle(puzzle):
    """Reconstruct review features from returned grids, never sampled metadata."""
    orbit = _orbit(_binary(puzzle['train'][0]['output']))
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            inp, out = pair['input'], pair['output']
            i, swap, r, c = _matches(orbit, _binary(inp))[0]
            h, w = inp.shape
            H, W = out.shape
            touches = int(r == 0) + int(c == 0) + int(r + h == H) + int(c + w == W)
            mask = np.zeros((H, W), dtype=np.int8)
            mask[r:r + h, c:c + w] = 1
            inverse = next(k for k in range(8) if _compose(k, i) == 0)
            mask = _transform(mask, inverse)
            rr, cc = np.nonzero(mask)
            focus = (min(2, int(3 * (rr.mean() + .5) / mask.shape[0])),
                     min(2, int(3 * (cc.mean() + .5) / mask.shape[1])))
            records.append(dict(split=split, shape=(h, w), output_shape=(H, W),
                                area=inp.size / out.size, aspect=h / w,
                                narrow=min(h, w), full_span=h == H or w == W,
                                touches=touches, offset=(r, c), focus=focus,
                                orientation=i, swap=swap,
                                palette=tuple(int(v) for v in np.unique(inp))))
    return records


def self_test() -> None:
    corpus = [generate_puzzle(seed) for seed in range(1000, 1042)]
    assert _puzzle_signature(corpus[0]) == _puzzle_signature(generate_puzzle(1000))
    assert _puzzle_signature(generate_puzzle(2718, 6, 2)) != _puzzle_signature(generate_puzzle(2719, 6, 2))
    for j, (nt, nq) in enumerate([(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(400 + j, nt, nq)
        assert nt is None or len(p['train']) == nt
        assert nq is None or len(p['test']) == nq
        assert _puzzle_signature(p) == _puzzle_signature(generate_puzzle(400 + j, nt, nq))
    for nt, nq in [(1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (3.5, 1)]:
        try:
            generate_puzzle(600, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    assert len({len(p['train']) for p in corpus}) >= 4
    assert len({len(p['test']) for p in corpus}) >= 3
    references = {_canonical(_binary(p['train'][0]['output'])) for p in corpus}
    assert len(references) >= 40, 'measure content after BOTH dihedral and color normalization'
    dimensions = {tuple(sorted(p['train'][0]['output'].shape)) for p in corpus}
    assert len(dimensions) >= 22
    assert any(h == w and h < 20 for h, w in dimensions)
    assert any(h == w and h > 20 for h, w in dimensions)
    assert any(h != w for h, w in dimensions)
    assert max(max(s) for s in dimensions) == 30
    # Geometric proxies measured on actual minority-role pixels. Visual QA
    # complements these: they do not claim to identify semantic motif labels.
    feature_rows = [_reference_features(p['train'][0]['output']) for p in corpus]
    assert len({tuple(f.items()) for f in feature_rows}) >= 35
    assert all(f['long_runs'] and f['short_runs'] and f['blocks'] for f in feature_rows)
    assert sum(f['holes'] > 0 for f in feature_rows) >= 20
    assert sum(f['diagonal'] > 0 for f in feature_rows) >= 15
    assert sum(f['repetition'] > 0 for f in feature_rows) >= 10
    assert len({f['branches'] for f in feature_rows}) >= 20
    assert max(f['max_run'] for f in feature_rows) >= 25
    measurements = [measure_puzzle(p) for p in corpus]
    records = [r for rows in measurements for r in rows]
    assert len({r['shape'] for r in records}) >= 100
    assert len({r['offset'] for r in records}) >= 70
    assert {r['orientation'] for r in records} == set(range(8))
    assert len({r['palette'] for r in records}) >= 35
    assert len({r['focus'] for r in records}) == 9
    assert sum(r['area'] < .12 for r in records) >= 30
    assert sum(r['area'] > .8 for r in records) >= 20
    assert sum(r['narrow'] <= 3 for r in records) >= 25
    assert sum(r['full_span'] for r in records) >= 25
    assert sum(r['aspect'] >= 4 for r in records) >= 15
    assert sum(r['aspect'] <= .25 for r in records) >= 15
    assert sum(r['touches'] == 0 for r in records) >= 40
    assert sum(r['touches'] == 1 for r in records) >= 40
    assert sum(r['touches'] >= 2 for r in records) >= 40
    old_shapes = {(8, 10), (16, 8), (10, 10), (13, 12), (11, 10)}
    assert old_shapes & {r['shape'] for r in records}, 'original dimensions are permitted'
    geometry_regimes, palette_regimes, combinations = set(), set(), set()
    novelty_positions, shape_varied, rich, exchanges, shared_colors = set(), 0, 0, 0, 0
    repeated_outputs, nonalternating, size_swapped = 0, 0, 0
    for p, rows in zip(corpus, measurements):
        train, test = [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'test']
        ti, qi = {r['orientation'] for r in train}, {r['orientation'] for r in test}
        tp, qp = {r['palette'] for r in train}, {r['palette'] for r in test}
        geometry_regimes.add('constant' if len(ti | qi) == 1 else 'held_out' if ti.isdisjoint(qi)
                             else 'shared' if qi <= ti else 'mixed')
        palette_regimes.add('stable' if len(tp | qp) == 1 else 'new' if qp.isdisjoint(tp)
                            else 'familiar' if qp <= tp else 'mixed')
        rich += len(tp) > 2
        shared_colors += any(a != b and set(a) & set(b) for a in tp for b in tp)
        exchanges += any(len({r['swap'] for r in rows if r['palette'] == palette}) == 2 for palette in tp | qp)
        shape_varied += len({r['shape'] for r in rows}) >= 3
        repeated_outputs += len({_grid_key(a['output']) for a in p['train'] + p['test']}) < len(rows)
        nonalternating += any(train[j]['palette'] == train[j + 1]['palette'] for j in range(len(train) - 1)) and len(tp) > 1
        size_swapped += len({r['output_shape'] for r in rows}) > 1
        for j, r in enumerate(test):
            novelty_positions.add((j, r['palette'] not in tp))
            combinations.add((r['orientation'] not in ti, r['palette'] not in tp, r['area'] < .2))
    assert geometry_regimes == {'constant', 'held_out', 'shared', 'mixed'}
    assert {'stable', 'new', 'familiar', 'mixed'} <= palette_regimes
    assert len(combinations) == 8, 'geometry, recoloring, and crop scale must combine independently'
    assert all((j, new) in novelty_positions for j in range(3) for new in (False, True))
    assert rich >= 5 and exchanges >= 10 and shared_colors >= 8
    assert shape_varied >= 38 and repeated_outputs >= 10 and nonalternating >= 5 and size_swapped >= 8
    # Corrupt the returned data: validation must inspect content, not metadata.
    bad = {s: [{k: v.copy() for k, v in a.items()} for a in corpus[0][s]] for s in ('train', 'test')}
    out = bad['test'][0]['output']
    colors = np.unique(out)
    out[0, 0] = colors[0] if out[0, 0] == colors[1] else colors[1]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('a test-only pixel edit was accepted')
    # Explicit regression for a crop with several incompatible completions.
    reference = np.array([[int(v) for v in row] for row in [
        '00011000000000000011', '00011000000000111111',
        '00111000000001111111', '00111000000011100011',
        '01111100000111000000', '01100100001110000000',
        '01100111001100000000', '01100101001000000000',
        '01100101111000000000', '01100101101111111110',
        '01100101111000000010', '01100101001111111110',
        '01100111001000001010', '01100100001111110111',
        '01111100001000011010', '00110110001000010111',
        '01100010001000011010', '11000110001000010111',
        '01110100010100011010', '00011100100010000111']], dtype=np.int8)
    for crop in [reference[:4, :4], reference[4:12, :8]]:
        assert len({(i, swap) for i, swap, _, _ in _matches(_orbit(reference), crop)}) > 1
