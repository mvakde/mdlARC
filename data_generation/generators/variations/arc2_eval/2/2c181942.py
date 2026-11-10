"""Color-labelled hub assembly, preserving eight-connected silhouettes.

The inward attachment is the centered two-cell extreme edge. Quarter-turn
choices are equivalent only when they occupy exactly the same final cells.
Background and labels are locally visible; there is no hidden color map.
"""
import numpy as np

_SIDE_TOP, _SIDE_RIGHT, _SIDE_BOTTOM, _SIDE_LEFT = range(4)
_ALL_SIDES = set(range(4))



def _validate_optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_optional_count(num_train, "num_train", 2, 10)
    num_test = _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees broad count diversity across
        # consecutive seeds while remaining entirely seed-dependent.
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        num_train = minimum_train + (
            (seed // 4 + 3 * seed) % span
        )
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists")
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _hub_cells(row, col):
    return (
        (row, col + 1),
        (row, col + 2),
        (row + 1, col + 3),
        (row + 2, col + 3),
        (row + 3, col + 1),
        (row + 3, col + 2),
        (row + 1, col),
        (row + 2, col),
    )


def _draw_hub(grid, row, col, labels):
    grid[row, col + 1 : col + 3] = labels[_SIDE_TOP]
    grid[row + 1 : row + 3, col + 3] = labels[_SIDE_RIGHT]
    grid[row + 3, col + 1 : col + 3] = labels[_SIDE_BOTTOM]
    grid[row + 1 : row + 3, col] = labels[_SIDE_LEFT]


def _target_attachment_cells(side, hub_row, hub_col):
    if side == _SIDE_TOP:
        return np.array(
            [[hub_row - 1, hub_col + 1], [hub_row - 1, hub_col + 2]],
            dtype=np.int64,
        )
    if side == _SIDE_RIGHT:
        return np.array(
            [[hub_row + 1, hub_col + 4], [hub_row + 2, hub_col + 4]],
            dtype=np.int64,
        )
    if side == _SIDE_BOTTOM:
        return np.array(
            [[hub_row + 4, hub_col + 1], [hub_row + 4, hub_col + 2]],
            dtype=np.int64,
        )
    if side == _SIDE_LEFT:
        return np.array(
            [[hub_row + 1, hub_col - 1], [hub_row + 2, hub_col - 1]],
            dtype=np.int64,
        )
    raise AssertionError("invalid side")


def _mark_blocked_around(blocked, rows, cols):
    height, width = blocked.shape
    for row, col in zip(rows, cols):
        row = int(row)
        col = int(col)
        row_start = max(0, row - 1)
        row_stop = min(height, row + 2)
        col_start = max(0, col - 1)
        col_stop = min(width, col + 2)
        blocked[row_start:row_stop, col_start:col_stop] = True


def _mode_color(grid):
    counts = np.bincount(
        grid.astype(np.int64, copy=False).ravel(),
        minlength=10,
    )
    return int(np.argmax(counts))


def _hub_labels_at(grid, row, col, background):
    block = grid[row : row + 4, col : col + 4]
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 1:3] = True
    allowed[3, 1:3] = True
    allowed[1:3, 0] = True
    allowed[1:3, 3] = True

    if np.any(block[~allowed] != background):
        return None

    side_pairs = (
        (int(block[0, 1]), int(block[0, 2])),
        (int(block[1, 3]), int(block[2, 3])),
        (int(block[3, 1]), int(block[3, 2])),
        (int(block[1, 0]), int(block[2, 0])),
    )

    labels = []
    for first, second in side_pairs:
        if first != second or first == background:
            return None
        labels.append(first)

    if len(set(labels)) != 4:
        return None
    return tuple(labels)


def _extract_loose_pieces(grid, background, hub_row, hub_col):
    outside = grid != background
    for row, col in _hub_cells(hub_row, hub_col):
        outside[row, col] = False

    visited = np.zeros(grid.shape, dtype=bool)
    pieces = []
    height, width = grid.shape

    for start_row, start_col in np.argwhere(outside):
        start_row = int(start_row)
        start_col = int(start_col)
        if visited[start_row, start_col]:
            continue

        color = int(grid[start_row, start_col])
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells = []

        while stack:
            row, col = stack.pop()
            cells.append((row, col))

            for drow, dcol in _NEIGHBORS:
                new_row = row + drow
                new_col = col + dcol
                if (
                    0 <= new_row < height
                    and 0 <= new_col < width
                    and outside[new_row, new_col]
                    and not visited[new_row, new_col]
                    and int(grid[new_row, new_col]) == color
                ):
                    visited[new_row, new_col] = True
                    stack.append((new_row, new_col))

        coords = np.asarray(cells, dtype=np.int64)
        min_row = int(coords[:, 0].min())
        max_row = int(coords[:, 0].max())
        min_col = int(coords[:, 1].min())
        max_col = int(coords[:, 1].max())

        mask = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1),
            dtype=bool,
        )
        mask[
            coords[:, 0] - min_row,
            coords[:, 1] - min_col,
        ] = True

        pieces.append(
            {
                "color": color,
                "coords": coords,
                "mask": mask,
            }
        )

    return pieces


def _assert_piece_separation(pieces, hub_row, hub_col):
    coord_sets = []

    for piece in pieces:
        coords = {
            (int(row), int(col))
            for row, col in piece["coords"]
        }
        coord_sets.append(coords)

        for row, col in coords:
            inside_hub_clearance = (
                hub_row - 1 <= row <= hub_row + 4
                and hub_col - 1 <= col <= hub_col + 4
            )
            assert not inside_hub_clearance, (
                "loose pieces must be separated from the hub"
            )

    for first_index in range(len(coord_sets)):
        expanded = set()
        for row, col in coord_sets[first_index]:
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    expanded.add((row + drow, col + dcol))

        for second_index in range(first_index + 1, len(coord_sets)):
            assert expanded.isdisjoint(coord_sets[second_index]), (
                "loose pieces must be separated from one another"
            )


def _grid_value_key(grid):
    normalized = grid.astype(np.int8, copy=False)
    return normalized.shape, normalized.tobytes()


def _puzzles_equal(first, second):
    if set(first.keys()) != set(second.keys()):
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for first_pair, second_pair in zip(
            first[section],
            second[section],
        ):
            for grid_name in ("input", "output"):
                first_grid = first_pair[grid_name]
                second_grid = second_pair[grid_name]
                if first_grid.dtype != second_grid.dtype:
                    return False
                if not np.array_equal(first_grid, second_grid):
                    return False
    return True


def _puzzle_value_key(puzzle):
    return tuple(
        (
            section,
            tuple(
                (
                    _grid_value_key(pair["input"]),
                    _grid_value_key(pair["output"]),
                )
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


# Exact staged grids encoded as (row, start column, nonempty row span).
# These fixtures are used only for regression and exact-content copy checks.
_SOURCE_FIXTURES = (((23, 20),
  ((2, 13, '2222'), (3, 14, '22'), (8, 6, '66'), (9, 5, '3882'), (10, 5, '3882'), (11, 6, '11'),
   (14, 13, '44'), (15, 13, '44'), (16, 12, '4444'), (17, 4, '33'), (18, 3, '3333'),
   (19, 3, '3883')),
  ((8, 2, '338866882'), (9, 3, '33388222'), (10, 3, '33388222'), (11, 2, '338811882'))),
 ((18, 21),
  ((3, 16, '33'), (4, 15, '3883'), (6, 7, '33'), (7, 6, '1882'), (8, 6, '1882'), (9, 7, '44'),
   (12, 1, '1111'), (13, 1, '11118888888882'), (14, 2, '1188888888222'), (15, 12, '222'),
   (16, 14, '2')),
  ((4, 6, '3883'), (5, 7, '33'), (6, 3, '1188338882'), (7, 3, '1111882222'), (8, 3, '1111882222'),
   (9, 3, '1188448882'))),
 ((20, 25),
  ((2, 3, '228888888888888833'), (3, 3, '228888888888888833'), (4, 2, '2882'), (7, 12, '11'),
   (8, 11, '2887'), (9, 11, '2887'), (10, 12, '33'), (14, 1, '7887'),
   (15, 1, '777788888888888881111'), (16, 2, '77888888888888881111')),
  ((3, 12, '11'), (4, 12, '11'), (5, 12, '11'), (6, 12, '11'), (7, 8, '2888118877'),
   (8, 9, '22288777'), (9, 9, '22288777'), (10, 8, '2888338877'), (11, 12, '33'), (12, 12, '33'))),
 ((24, 26),
  ((1, 17, '1'), (2, 17, '111'), (3, 3, '22888888888888111'), (4, 3, '228888888888881'),
   (5, 3, '22'), (6, 2, '2882'), (10, 14, '11'), (11, 13, '4882'), (12, 13, '4882'),
   (13, 4, '338888888866'), (14, 3, '333888888888888884884'), (15, 3, '338888888888888884444'),
   (16, 21, '44'), (17, 12, '6'), (18, 12, '6'), (19, 9, '6666'), (20, 9, '6666'), (21, 12, '6'),
   (22, 12, '6')),
  ((7, 13, '1111'), (8, 14, '11'), (9, 14, '11'), (10, 10, '44881188882'), (11, 11, '444882222'),
   (12, 11, '444882222'), (13, 10, '44886688882'), (14, 14, '66'), (15, 14, '66'), (16, 14, '66'),
   (17, 12, '666666'))))

_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _source_pairs():
    pairs = []
    for shape, inp, out in _SOURCE_FIXTURES:
        pair = {}
        for role, rows in (("input", inp), ("output", out)):
            grid = np.full(shape, 8, dtype=np.int8)
            for row, col, text in rows:
                grid[row, col:col + len(text)] = list(map(int, text))
            pair[role] = grid
        pairs.append(pair)
    return pairs


_SOURCE_KEYS = frozenset((_grid_value_key(p['input']), _grid_value_key(p['output']))
                         for p in _source_pairs())


def _binary_connected(mask, diagonal=True):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    stack = [cells.pop()]
    neighbors = _NEIGHBORS if diagonal else ((-1, 0), (1, 0), (0, -1), (0, 1))
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            point = (r + dr, c + dc)
            if point in cells:
                cells.remove(point)
                stack.append(point)
    return not cells


def _attachment_candidates(mask):
    """Canonical masks face down; keep all turns but deduplicate placements."""
    candidates = {}
    for turns in range(4):
        canonical = np.rot90(mask, turns).copy()
        h, w = canonical.shape
        if w < 2 or w % 2 or h < 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(canonical[-1], neck):
            key = _grid_value_key(canonical)
            if key not in candidates:
                candidates[key] = [canonical, []]
            candidates[key][1].append(turns)
    return list(candidates.values())


def _canonicalize_piece(mask):
    assert _binary_connected(mask), "piece must be eight-connected"
    candidates = _attachment_candidates(mask)
    assert len(candidates) == 1, "competing or missing centered attachment edges"
    return candidates[0]


def _dock_coords(mask, side, row, col):
    """Translate the canonical neck onto the hub's adjacent outside cells."""
    marker = np.zeros_like(mask)
    marker[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    oriented = np.rot90(mask, -side)
    neck = np.argwhere(np.rot90(marker, -side))
    targets = _target_attachment_cells(side, row, col)
    shift = targets.min(axis=0) - neck.min(axis=0)
    return np.argwhere(oriented) + shift


def _find_hubs(grid, background):
    # Candidate top bars first, then check the full 4-by-4 pattern.
    top = (grid[:-3, 1:-2] != background) & (grid[:-3, 1:-2] == grid[:-3, 2:-1])
    hubs = []
    for r, c in np.argwhere(top):
        labels = _hub_labels_at(grid, int(r), int(c), background)
        if labels is not None:
            hubs.append((int(r), int(c), labels))
    return hubs


def _inspect_input(grid):
    bg = _mode_color(grid)
    assert np.count_nonzero(grid == bg) > grid.size // 2
    hubs = _find_hubs(grid, bg)
    assert len(hubs) == 1, "one unambiguous hollow hub is required"
    row, col, labels = hubs[0]
    pieces = _extract_loose_pieces(grid, bg, row, col)
    assert pieces
    _assert_piece_separation(pieces, row, col)
    expected = np.full_like(grid, bg)
    _draw_hub(expected, row, col, labels)
    matched = []
    used_sides = set()
    for piece in pieces:
        color = piece['color']
        assert _binary_connected(piece['mask'])
        piece['side'] = labels.index(color) if color in labels else None
        if piece['side'] is None:
            continue  # Filtering is by color; irregular distractors need no neck.
        side = piece['side']
        assert side not in used_sides, "two matching pieces share a label"
        used_sides.add(side)
        canonical, turns = _canonicalize_piece(piece['mask'])
        coords = _dock_coords(canonical, side, row, col)
        rr, cc = coords.T
        assert np.all((rr >= 0) & (rr < grid.shape[0]))
        assert np.all((cc >= 0) & (cc < grid.shape[1]))
        assert np.all(expected[rr, cc] == bg), "assembled silhouettes overlap"
        expected[rr, cc] = color
        piece['canonical'] = canonical
        piece['rotations'] = {int((t - side) % 4) for t in turns}
        piece['target_coords'] = coords
        matched.append(piece)
    assert np.count_nonzero(expected == bg) > expected.size // 2
    return expected, dict(background=bg, hub_row=row, hub_col=col, labels=labels,
                          pieces=pieces, matched=matched, matched_sides=used_sides)


def solve(train_pairs, test_input):
    """Input-only geometric prediction; demonstrations identify this family."""
    # No seed, test output, construction mask, or sampled convention is consulted.
    return _inspect_input(np.array(test_input, copy=True))[0]


def _mask_text(*rows):
    return np.array([[c == '#' for c in row] for row in rows], dtype=bool)


def _make_piece_mask(rng, complexity=1, directional=False):
    for _ in range(60):
        kind = int(rng.choice(4, p=(.16, .27, .23, .34)))
        width = int(rng.choice((2, 4, 6, 8), p=(.17, .43, .28, .12)))
        if complexity == 0 and rng.random() < .6:
            width = int(rng.choice((2, 4, 6)))
        depth = int(rng.integers(2, (5, 7, 8)[complexity]))
        if kind == 0:
            width = 2
        center = width // 2
        mask = np.zeros((depth, width), dtype=bool)
        mask[:, center - 1:center + 1] = True
        stem = int(rng.integers(1, depth))
        body = depth - stem
        if kind == 1 and width > 2:
            # Flange thickness and neck length are separate parameters.
            mask[:body] = True
        elif kind == 2 and width > 2:
            # Two diagonal wings. Each outward step grows by one cell, so
            # even a missing central flange remains one eight-connected piece.
            body = min(width // 2 - 1, depth - 1)
            mask = np.zeros_like(mask)
            mask[body:, center - 1:center + 1] = True
            for r in range(body):
                reach = body - r
                mask[r, center - 1 - reach] = True
                mask[r, center + reach] = True
                if rng.random() < .35:
                    mask[r, center - 1 - reach:center + reach + 1] = True
            if not mask[:, 0].any():
                continue
        elif width > 2:
            # Independently vary the shoulders on each body row, then punch
            # holes/notches without sacrificing the connected central stem.
            for r in range(body):
                left = int(rng.integers(0, center))
                right = int(rng.integers(center + 1, width + 1))
                mask[r, left:right] = True
            mask[int(rng.integers(body)), :center] = True
            mask[int(rng.integers(body)), center:] = True
            for r, c in np.argwhere(mask[:body]):
                if c not in (center - 1, center) and rng.random() < .18:
                    mask[r, c] = False
            if not mask[:, 0].any() or not mask[:, -1].any():
                continue
        candidates = _attachment_candidates(mask)
        if len(candidates) != 1 or not _binary_connected(mask):
            continue
        canonical, turns = candidates[0]
        if directional and len(turns) != 1:
            continue
        return canonical
    # Deterministic geometric fallback, with one attachment orientation.
    return _mask_text('####', '.##.')


def _make_distractor(rng, complexity, dockable=False):
    if dockable or rng.random() < .5:
        return _make_piece_mask(rng, complexity)
    if rng.random() < .25:
        return _mask_text('.##', '###', '##.')
    h, w = (int(rng.integers(2, 7)) for _ in range(2))
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2, w // 2] = True
    for _ in range(int(rng.integers(3, h * w + 1))):
        options = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _NEIGHBORS:
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    options.add((r + dr, c + dc))
        options = sorted(options)
        mask[options[int(rng.integers(len(options)))]] = True
    rows, cols = np.nonzero(mask)
    return mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def _relative_assembly(pieces):
    occupied = set(_hub_cells(0, 0))
    all_coords = [np.array(_hub_cells(0, 0))]
    for piece in pieces:
        if piece['side'] is None:
            continue
        coords = _dock_coords(piece['mask'], piece['side'], 0, 0)
        cells = set(map(tuple, coords))
        if occupied & cells:
            return None
        occupied.update(cells)
        all_coords.append(coords)
    return np.concatenate(all_coords)


def _choose_dimensions(rng, coords, regime, source_shift):
    min_h, min_w = coords.max(axis=0) - coords.min(axis=0) + 1
    if source_shift and regime == 'target_test':
        h, w = rng.integers(25, 31, size=2)
    elif rng.random() < .3:
        h, w = int(rng.integers(12, 19)), int(rng.integers(23, 31))
        if rng.random() < .5:
            h, w = w, h
    else:
        h, w = rng.integers(12, 31, size=2)
    return max(int(h), int(min_h)), max(int(w), int(min_w))


def _choose_hub_position(rng, height, width, coords):
    low = -coords.min(axis=0)
    high = np.array((height - 1, width - 1)) - coords.max(axis=0)
    assert np.all(low <= high)
    mode = rng.random()
    if mode < .4:
        # Sample the full feasible boundary, including corners and edge touches.
        pos = [int(rng.choice((a, b))) for a, b in zip(low, high)]
        if rng.random() < .45:
            axis = int(rng.integers(2))
            pos[axis] = int(rng.integers(low[axis], high[axis] + 1))
        return tuple(pos)
    return tuple(int(rng.integers(a, b + 1)) for a, b in zip(low, high))


def _place_loose_pieces(rng, shape, bg, hub, labels, pieces):
    grid = np.full(shape, bg, dtype=np.int8)
    row, col = hub
    _draw_hub(grid, row, col, labels)
    blocked = np.zeros(shape, dtype=bool)
    blocked[max(0, row - 1):row + 5, max(0, col - 1):col + 5] = True
    order = list(rng.permutation(len(pieces)))
    order.sort(key=lambda i: -pieces[i]['mask'].size)
    for i in order:
        p = pieces[i]
        oriented = np.rot90(p['mask'], p['input_turns'])
        h, w = oriented.shape
        if h > shape[0] or w > shape[1]:
            return None
        local = np.argwhere(oriented)
        tops, lefts = np.indices((shape[0] - h + 1, shape[1] - w + 1))
        rr = tops.ravel()[:, None] + local[:, 0]
        cc = lefts.ravel()[:, None] + local[:, 1]
        good = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
        if not len(good):
            return None
        k = int(rng.choice(good))
        grid[rr[k], cc[k]] = p['color']
        _mark_blocked_around(blocked, rr[k], cc[k])
    return grid


def _sample_parameters(rng):
    return dict(palette=tuple(map(int, rng.permutation(10))),
                roles=int(rng.choice(3, p=(.2, .55, .25))),
                complexity=int(rng.integers(3)), source_shift=rng.random() < .25)


def _configurations(rng, parameters, ntrain, ntest):
    configs = []
    complexity = parameters['complexity']
    for i in range(ntrain + ntest):
        n = int(rng.choice(5, p=((.06, .34, .3, .2, .1),
                                 (.05, .2, .3, .25, .2),
                                 (.03, .12, .2, .3, .35))[complexity]))
        d = int(rng.choice(4, p=(.35, .35, .2, .1)))
        if parameters['source_shift'] and i >= ntrain:
            n, d = 4, int(rng.integers(1, 4))
        if n == 0:
            d = max(1, d)
        configs.append(dict(sides=set(map(int, rng.choice(4, n, replace=False))),
                            distractors=d, forced={}, swap=None))
    # Randomly distribute witnesses; no fixed 2/3/4 count progression.
    for side in rng.permutation(4):
        if not any(int(side) in c['sides'] for c in configs[:ntrain]):
            configs[int(rng.integers(ntrain))]['sides'].add(int(side))
    for i in rng.choice(ntrain, 2, replace=False):
        if not configs[i]['sides']:
            configs[i]['sides'].add(int(rng.integers(4)))
    slots = [(i, s) for i, c in enumerate(configs[:ntrain]) for s in sorted(c['sides'])]
    rng.shuffle(slots)
    for rotation, (i, side) in enumerate(slots[:4]):
        configs[i]['forced'][side] = rotation
    # Two distinct training silhouettes are visible even with boundary counts.
    for width, (i, side) in zip((4, 6), slots[:2]):
        configs[i].setdefault('witness', {})[side] = width
    nonempty = [i for i, c in enumerate(configs[:ntrain]) if c['sides']]
    a, b = map(int, rng.choice(nonempty, 2, replace=False))
    configs[a]['distractors'] = max(1, configs[a]['distractors'])
    if parameters['roles'] == 1:
        configs[a]['swap'], configs[b]['swap'] = 0, 1
        configs[b]['distractors'] = max(1, configs[b]['distractors'])
    elif parameters['roles'] == 2:
        configs[a]['swap'] = 0
        for c in configs[ntrain:]:
            c['swap'] = 1
            c['distractors'] = max(1, c['distractors'])
            if not c['sides']:
                c['sides'].add(int(rng.integers(4)))
    # Ensure pair-level count configurations vary within every puzzle.
    if len({(tuple(sorted(c['sides'])), c['distractors']) for c in configs}) == 1:
        configs[-1]['distractors'] = (configs[-1]['distractors'] + 1) % 4
    return configs


def _pair_colors(rng, parameters, config, regime, previous_bg):
    pal = parameters['palette']
    roles = parameters['roles']
    swap = config['swap']
    if roles in (0, 2):
        pool = list(pal[:4])
        if roles == 2 and regime == 'target_test':
            pool[0] = pal[4]
    else:
        pool = list(map(int, rng.choice(10, 4, replace=False)))
    if swap is not None:
        match, absent = (pal[0], pal[4]) if swap == 0 else (pal[4], pal[0])
        available = [c for c in range(10) if c not in (match, absent)]
        if roles == 1:
            pool = [match] + list(map(int, rng.choice(available, 3, replace=False)))
        assert match in pool and absent not in pool
    labels = list(map(int, rng.permutation(pool)))
    if swap is not None:
        side = int(rng.choice(sorted(config['sides'])))
        k = labels.index(match)
        labels[k], labels[side] = labels[side], labels[k]
    bg_options = [c for c in range(10) if c not in labels and c != previous_bg
                  and (swap is None or c != absent)
                  and (roles != 2 or c not in (pal[0], pal[4]))]
    bg = int(rng.choice(bg_options))
    distractor_pool = [c for c in range(10) if c not in labels and c != bg]
    colors = list(map(int, rng.choice(distractor_pool, config['distractors'])))
    if swap is not None and colors:
        colors[0] = absent
    return bg, tuple(labels), colors


def _build_pair(rng, parameters, config, regime, previous_bg):
    bg, labels, distractors = _pair_colors(rng, parameters, config, regime, previous_bg)
    pieces = []
    for side in sorted(config['sides']):
        delta = config['forced'].get(side, int(rng.integers(4)))
        width = config.get('witness', {}).get(side)
        for attempt in range(50):
            if width == 4:
                mask = _mask_text('####', '.##.')
            elif width == 6:
                mask = _mask_text('######', '..##..', '..##..')
            else:
                mask = _make_piece_mask(rng, parameters['complexity'], side in config['forced'])
            piece = dict(color=labels[side], side=side, mask=mask,
                         input_turns=int((-side - delta) % 4))
            if _relative_assembly(pieces + [piece]) is not None:
                pieces.append(piece)
                break
        else:
            piece['mask'] = _mask_text('####', '.##.')
            pieces.append(piece)
    for i, color in enumerate(distractors):
        pieces.append(dict(color=color, side=None,
                           mask=_make_distractor(rng, parameters['complexity'], dockable=i == 0),
                           input_turns=int(rng.integers(4))))
    coords = _relative_assembly(pieces)
    assert coords is not None
    shape = _choose_dimensions(rng, coords, regime, parameters['source_shift'])
    for attempt in range(16):
        hub = _choose_hub_position(rng, *shape, coords)
        grid = _place_loose_pieces(rng, shape, bg, hub, labels, pieces)
        if grid is not None and np.count_nonzero(grid == bg) > grid.size // 2:
            output, _ = _inspect_input(grid)
            return dict(input=grid, output=output)
        shape = tuple(min(30, x + 2) for x in shape)
    # Guaranteed separated packing: seven pieces fit ten 6x5 perimeter slots.
    # This rare fallback retains selection and rotation witnesses.
    shape, hub = (30, 30), (13, 13)
    grid = np.full(shape, bg, dtype=np.int8)
    _draw_hub(grid, *hub, labels)
    for i, piece in enumerate(pieces):
        width = config.get('witness', {}).get(piece['side'])
        mask = (_mask_text('######', '..##..', '..##..') if width == 6
                else _mask_text('####', '.##.'))
        oriented = np.rot90(mask, piece['input_turns'])
        r, c = (0 if i < 4 else 23), 7 * (i % 4)
        rr, cc = np.nonzero(oriented)
        grid[rr + r, cc + c] = piece['color']
    return dict(input=grid, output=solve([], grid))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(int(seed), rng, num_train, num_test)
    parameters = _sample_parameters(rng)
    configs = _configurations(rng, parameters, ntrain, ntest)
    puzzle = dict(train=[], test=[])
    seen = set()
    previous_bg = None
    for i, config in enumerate(configs):
        split = 'train' if i < ntrain else 'test'
        regime = 'source_train' if split == 'train' else 'target_test'
        for _ in range(8):
            pair = _build_pair(rng, parameters, config, regime, previous_bg)
            key = _grid_value_key(pair['input'])
            if key not in seen and (key, _grid_value_key(pair['output'])) not in _SOURCE_KEYS:
                break
        else:
            raise AssertionError('could not construct a distinct input')
        seen.add(key)
        previous_bg = _mode_color(pair['input'])
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle

def _piece_signature(mask):
    return min((tuple(np.rot90(mask, t).shape), np.rot90(mask, t).tobytes().hex())
               for t in range(4))


def _features(pair):
    expected, analysis = _inspect_input(pair['input'])
    assert np.array_equal(pair['output'], expected), 'incorrect assembly'
    matched = analysis['matched']
    unmatched = [p for p in analysis['pieces'] if p['side'] is None]
    shapes = []
    for p in matched:
        mask = p['canonical']
        neck = np.zeros(mask.shape[1], dtype=bool)
        neck[mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
        stem = 0
        for row in mask[::-1]:
            if not np.array_equal(row, neck):
                break
            stem += 1
        shapes.append(dict(depth=int(mask.shape[0]), width=int(mask.shape[1]),
                           stem=stem, cells=int(mask.sum()), attachment_width=2,
                           diagonal_only=not _binary_connected(mask, False),
                           equivalent_turns=len(_attachment_candidates(p['mask'])[0][1]),
                           asymmetric=not np.array_equal(mask, np.fliplr(mask)),
                           signature=_piece_signature(mask)))
    h, w = pair['input'].shape
    row, col = analysis['hub_row'], analysis['hub_col']
    coords = np.argwhere(expected != analysis['background'])
    return dict(background=analysis['background'], labels=list(analysis['labels']),
                shape=[h, w], hub=[row, col], matching_count=len(matched),
                distractor_count=len(unmatched), sides=sorted(analysis['matched_sides']),
                shapes=shapes, matching_colors=[p['color'] for p in matched],
                distractor_colors=[p['color'] for p in unmatched],
                distractor_dockable=[len(_attachment_candidates(p['mask'])) == 1 for p in unmatched],
                distractor_irregular=[len(_attachment_candidates(p['mask'])) == 0 for p in unmatched],
                repeated_distractor_color=len({p['color'] for p in unmatched}) < len(unmatched),
                required_rotations=sorted(set().union(*(p['rotations'] for p in matched if len(p['rotations']) == 1))),
                edge_touch=bool(np.any((coords == 0) | (coords == np.array((h - 1, w - 1))))),
                near_corner=min(row, h - row - 4) <= 1 and min(col, w - col - 4) <= 1,
                off_center=abs((row + 1.5) / (h - 1) - .5) > .25 or abs((col + 1.5) / (w - 1) - .5) > .25)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            assert not np.array_equal(pair['input'], pair['output'])
            key = _grid_value_key(pair['input'])
            both = key, _grid_value_key(pair['output'])
            assert key not in seen_inputs and both not in seen_pairs
            assert both not in _SOURCE_KEYS, 'exact source copy'
            seen_inputs.add(key)
            seen_pairs.add(both)
            analyses[split].append(_features(pair))
    train = analyses['train']
    assert set().union(*(set(a['sides']) for a in train)) == _ALL_SIDES
    assert set().union(*(set(a['required_rotations']) for a in train)) == set(range(4))
    assert any(any(a['distractor_dockable']) for a in train), 'demonstrate color-based filtering'
    all_a = train + analyses['test']
    assert len({a['background'] for a in all_a}) >= 2
    assert len({s['width'] for a in all_a for s in a['shapes']}) >= 2
    assert len({s['depth'] for a in all_a for s in a['shapes']}) >= 2
    # Both splits follow the same geometric rule. The optional size/count/color
    # shifts require no hidden mapping or previously unseen movement convention.
    for pair in puzzle['test']:
        predicted = solve([{k: v.copy() for k, v in p.items()} for p in puzzle['train']],
                          pair['input'].copy())
        assert np.array_equal(predicted, pair['output'])


def _diversity_checks(puzzles):
    rows = []
    within_background, within_width, within_depth = 0, 0, 0
    train_flips, stable_roles, heldout_flips, mixed_test_depth = 0, 0, 0, 0
    for puzzle in puzzles:
        train = [_features(p) for p in puzzle['train']]
        test = [_features(p) for p in puzzle['test']]
        all_a = train + test
        within_background += len({a['background'] for a in all_a}) > 1
        within_width += len({s['width'] for a in all_a for s in a['shapes']}) > 1
        within_depth += len({s['depth'] for a in all_a for s in a['shapes']}) > 1
        tm = set(c for a in train for c in a['matching_colors'])
        td = set(c for a in train for c in a['distractor_colors'])
        qm = set(c for a in test for c in a['matching_colors'])
        qd = set(c for a in test for c in a['distractor_colors'])
        train_flips += bool(tm & td)
        stable_roles += not bool((tm | qm) & (td | qd))
        heldout_flips += not bool(tm & td) and bool((tm & qd) or (td & qm))
        mixed_test_depth += len(test) > 1 and len({tuple(sorted(s['depth'] for s in a['shapes'])) for a in test}) > 1
        rows.extend((split, a) for split, aa in (('train', train), ('test', test)) for a in aa)
    shapes = [s for _, a in rows for s in a['shapes']]
    assert within_background == within_width == within_depth == len(puzzles)
    assert train_flips >= 10 and stable_roles >= 3 and heldout_flips >= 3
    assert mixed_test_depth >= 10
    assert len({s['signature'] for s in shapes}) >= 100
    assert {s['width'] for s in shapes} == {2, 4, 6, 8}
    assert set(range(2, 8)) <= {s['depth'] for s in shapes}
    assert set(range(1, 6)) <= {s['stem'] for s in shapes}
    assert {s['attachment_width'] for s in shapes} == {2}
    assert min(s['cells'] for s in shapes) == 4 and max(s['cells'] for s in shapes) >= 30
    assert sum(s['diagonal_only'] for s in shapes) >= 30
    assert sum(s['asymmetric'] for s in shapes) >= 30
    assert {2, 4} <= {s['equivalent_turns'] for s in shapes}
    assert len({(s['depth'], s['width'], s['stem']) for s in shapes}) >= 40
    for split in ('train', 'test'):
        aa = [a for section, a in rows if section == split]
        assert {0, 1, 2, 3, 4} <= {a['matching_count'] for a in aa}
        assert {0, 1, 2, 3} <= {a['distractor_count'] for a in aa}
        assert len({tuple(a['sides']) for a in aa}) >= 12
        assert any(a['matching_count'] == 4 and a['distractor_count'] >= 2 for a in aa)
        assert any(any(s['width'] >= 6 for s in a['shapes']) and a['distractor_count'] >= 2 and a['off_center'] for a in aa)
        assert any(any(a['distractor_irregular']) for a in aa)
        assert any(a['repeated_distractor_color'] for a in aa)
    assert len({tuple(a['shape']) for _, a in rows}) >= 100
    assert min(min(a['shape']) for _, a in rows) <= 14
    assert max(max(a['shape']) for _, a in rows) == 30
    assert any(max(a['shape']) / min(a['shape']) >= 2 for _, a in rows)
    assert sum(a['near_corner'] for _, a in rows) >= 5
    assert sum(a['edge_touch'] for _, a in rows) >= 50
    train_sizes = {tuple(a['shape']) for split, a in rows if split == 'train'}
    test_sizes = {tuple(a['shape']) for split, a in rows if split == 'test'}
    assert len(train_sizes & test_sizes) >= 10


def self_test() -> None:
    # All four evidence outputs, including diagonal wings, 2x4 bar, and square.
    source = _source_pairs()
    for p in source:
        assert np.array_equal(solve(source[:3], p['input'].copy()), p['output'])
    assert sum(len(_inspect_input(p['input'])[1]['pieces']) for p in source) == 15
    ambiguous = _mask_text('.###', '####', '#.##', '.##.')
    assert len(_attachment_candidates(ambiguous)) > 1
    try:
        _canonicalize_piece(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous attachments accepted')
    square = _mask_text('##', '##')
    bar = _mask_text('##', '##', '##', '##')
    assert len(_canonicalize_piece(square)[1]) == 4
    assert len(_canonicalize_piece(bar)[1]) == 2
    assert _binary_connected(_mask_text('#..#', '.##.'))
    assert not _binary_connected(_mask_text('#..#', '.##.'), False)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_value_key(p) for p in puzzles}) == 42
    _diversity_checks(puzzles)
    first = generate_puzzle(991, 6, 3)
    assert _puzzles_equal(first, generate_puzzle(991, 6, 3))
    assert not _puzzles_equal(first, generate_puzzle(992, 6, 3))
    for i, (nt, nq) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                  (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, nt, nq)
        assert nt is None or len(p['train']) == nt
        assert nq is None or len(p['test']) == nq
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (4, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Validator must detect a damaged output, independently of sampler metadata.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]]
               for s in ('train', 'test')}
    grid = damaged['test'][0]['output']
    grid[0, 0] = (int(grid[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
