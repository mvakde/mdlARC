"""Fault-directed support removal in a stack of patterned rigid platforms.

Only the standard library and NumPy are used.  No parent code or examples are
imported.  The decoder, command inference, and solver depend on input cells only.
"""

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(grid, color):
    remaining = set(map(tuple, np.argwhere(grid == color)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    pending.append(p)
        result.append(cells)
    return result


def _piece(row, col, bitmap, kind):
    return dict(row=int(row), col=int(col), bitmap=bitmap.copy(), kind=kind)


def _cells(p, distance=0):
    h, w = p['bitmap'].shape
    return {(p['row'] + distance + r, p['col'] + c)
            for r in range(h) for c in range(w)}


def _settle(pieces, height):
    """Greatest simultaneously movable set; contacts are solid, never bonded."""
    result = [dict(p, bitmap=p['bitmap'].copy()) for p in pieces]
    for _ in range(height * (len(result) + 1)):
        cells = [_cells(p) for p in result]
        targets = [_cells(p, 1) for p in result]
        moving = {i for i, p in enumerate(result)
                  if p['row'] + p['bitmap'].shape[0] < height}
        while True:
            stationary = set(range(len(result))) - moving
            blocked = {i for i in moving
                       if any(targets[i] & cells[j] for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return result
        for i in moving:
            result[i]['row'] += 1
    raise AssertionError("gravity failed to converge")


def _render(shape, bg, pieces):
    result = np.full(shape, bg, dtype=np.int8)
    occupied = set()
    for p in pieces:
        footprint = _cells(p)
        assert not occupied & footprint, "overlapping rigid bodies"
        assert all(0 <= r < shape[0] and 0 <= c < shape[1]
                   for r, c in footprint)
        occupied.update(footprint)
        h, w = p['bitmap'].shape
        result[p['row']:p['row']+h, p['col']:p['col']+w] = p['bitmap']
    return result


def _frame_boxes(grid, color):
    boxes = []
    for cells in _components(grid, color):
        r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
        c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
        a = grid[r0:r1+1, c0:c1+1]
        if a.shape[0] < 4 or a.shape[1] < 10:
            return None
        if not (np.all(a[[0, -1], :] == color)
                and np.all(a[:, [0, -1]] == color)):
            return None
        dividers = np.flatnonzero(np.all(a == color, axis=0))
        if len(dividers) < 4 or not np.all(np.diff(dividers) == dividers[1]):
            return None
        pw = int(dividers[1] - 1)
        if pw < 2 or np.any(a[1:-1, :][:, np.any(a != color, axis=0)] == color):
            return None
        boxes.append((r0, c0, a.copy(), pw, len(dividers)-1))
    return boxes or None


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg) and np.all(grid[:, 0] == bg)
    assert np.all(grid[:, -1] == bg)
    candidates = []
    for color in np.unique(grid):
        if color != bg:
            boxes = _frame_boxes(grid, int(color))
            if boxes is not None:
                candidates.append((int(color), boxes))
    assert len(candidates) == 1, "need one unambiguous platform frame color"
    frame, boxes = candidates[0]
    boxes.sort(key=lambda b: (b[0], b[1]))
    assert len({(a.shape[0]-2, pw) for r, c, a, pw, n in boxes}) == 1
    platforms, panels, locations = [], [], []
    interior_colors = set()
    for r, c, a, pw, n in boxes:
        assert 3 <= n and r >= 1 and c >= 1
        platforms.append(_piece(r, c, a, 'platform'))
        row = [a[1:-1, 1+j*(pw+1):1+j*(pw+1)+pw].copy()
               for j in range(n)]
        panels.append(row)
        locations.append([c+1+j*(pw+1) for j in range(n)])
        interior_colors.update(int(x) for tile in row for x in np.unique(tile))
    assert len(interior_colors) == 2 and not interior_colors & {bg, frame}
    counts = Counter(_key(p) for row in panels for p in row)
    winners = [k for k, n in counts.items() if n == max(counts.values())]
    assert len(winners) == 1, "template must be uniquely modal"
    master = next(p for row in panels for p in row if _key(p) == winners[0])
    assert set(map(int, np.unique(master))) == interior_colors
    rods, rod_colors = [], set(map(int, np.unique(grid))) - {bg, frame} - interior_colors
    for color in sorted(rod_colors):
        for cells in _components(grid, color):
            rows, cols = {r for r, c in cells}, {c for r, c in cells}
            assert len(cells) >= 2 and (len(rows) == 1 or len(cols) == 1)
            r0, c0 = min(rows), min(cols)
            h, w = max(rows)-r0+1, max(cols)-c0+1
            assert len(cells) == h*w, "rod must be contiguous and straight"
            rods.append(_piece(r0, c0, np.full((h, w), color, np.int8), 'rod'))
    assert rods
    assert np.array_equal(_render(grid.shape, bg, platforms+rods), grid)
    commands, faults, exception_columns, edits = set(), [], [], []
    for body, row, xs in zip(platforms, panels, locations):
        exceptional = [j for j, tile in enumerate(row)
                       if not np.array_equal(tile, master)]
        assert len(exceptional) == 1, "one visible exceptional tile per platform"
        j = exceptional[0]
        delta = row[j] != master
        columns = [xs[j]+int(c) for c in np.flatnonzero(np.any(delta, axis=0))]
        contact_row = body['row'] + body['bitmap'].shape[0]
        if contact_row < grid.shape[0]:
            commands.update(int(grid[contact_row, c]) for c in columns
                            if int(grid[contact_row, c]) in rod_colors)
        faults.append(columns)
        exception_columns.append(j)
        colors = sorted(interior_colors)
        # Signed difference counts do not assume any literal color is "ink".
        edits.append((int(np.sum((master == colors[0]) & (row[j] == colors[1]))),
                      int(np.sum((master == colors[1]) & (row[j] == colors[0])))))
    return dict(grid=grid, bg=bg, frame=frame, platforms=platforms, rods=rods,
                commands=commands, faults=faults, master=master, edits=edits,
                exception_columns=exception_columns)


def _answer(info, commands=None, fall=True):
    if commands is None:
        commands = info['commands']
    survivors = info['platforms'] + [p for p in info['rods']
                                     if int(p['bitmap'][0, 0]) not in commands]
    settled = _settle(survivors, info['grid'].shape[0]) if fall else survivors
    return _render(info['grid'].shape, info['bg'], settled)


def _shortcut_commands(info, whole_tile=False, additions_only=False):
    """Deliberately wrong alternatives, used only to test demonstration evidence."""
    master, grid = info['master'], info['grid']
    pw = master.shape[1]
    rod_colors = {int(p['bitmap'][0, 0]) for p in info['rods']}
    commands = set()
    for body, j in zip(info['platforms'], info['exception_columns']):
        start = 1+j*(pw+1)
        tile = body['bitmap'][1:-1, start:start+pw]
        if whole_tile:
            cols = range(pw)
        else:
            color = int(np.max(master))
            delta = (master != color) & (tile == color) if additions_only else tile != master
            cols = np.flatnonzero(np.any(delta, axis=0))
        row = body['row'] + body['bitmap'].shape[0]
        if row < grid.shape[0]:
            commands.update(int(grid[row, body['col']+start+int(c)]) for c in cols
                            if int(grid[row, body['col']+start+int(c)]) in rod_colors)
    return commands


def _local_cut_answer(info):
    contacts = {(p['row']+p['bitmap'].shape[0], x)
                for p, xs in zip(info['platforms'], info['faults']) for x in xs}
    survivors = info['platforms'] + [p for p in info['rods'] if not _cells(p) & contacts]
    return _render(info['grid'].shape, info['bg'], _settle(survivors, info['grid'].shape[0]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read faults once in the initial scene, remove named rods, settle bodies."""
    return _answer(_decode(input_grid))


def _master(rng, ph, pw, carrier, motif):
    style = int(rng.integers(5))
    rr, cc = np.indices((ph, pw))
    if style == 0:
        mask = (rr+cc) % 2 == 0
    elif style == 1:
        mask = rr == rng.integers(ph, size=pw)[None, :]
    elif style == 2:
        mask = rng.random((ph, pw)) < float(rng.uniform(.25, .75))
    elif style == 3:
        mask = (rr+2*cc) % ph == int(rng.integers(ph))
    else:
        mask = rng.random((ph, pw)) < .5
        mask[:, (pw+1)//2:] = mask[:, :pw//2][:, ::-1]
    # Mixed columns make both addition and removal witnesses feasible anywhere.
    for c in range(pw):
        if np.all(mask[:, c]) or not np.any(mask[:, c]):
            mask[int(rng.integers(ph)), c] = not mask[0, c]
    return np.where(mask, motif, carrier).astype(np.int8)


def _variant(rng, master, required_col, allowed_cols, mode):
    colors = sorted(map(int, np.unique(master)))
    mask = master == colors[1]
    changed = mask.copy()
    permitted = [(r, c) for c in allowed_cols for r in range(mask.shape[0])]
    ones = [p for p in permitted if mask[p]]
    zeros = [p for p in permitted if not mask[p]]
    if mode in ('add', 'remove'):
        candidates = zeros if mode == 'add' else ones
        first = next(p for p in candidates if p[1] == required_col)
        changed[first] = mode == 'add'
        rest = [p for p in candidates if p != first]
        limit = min(3, len(candidates)-1)  # preserve both colors in the tile
        if limit >= 2:
            rng.shuffle(rest)
            for p in rest[:int(rng.integers(limit))]:
                changed[p] = mode == 'add'
    elif mode == 'balanced':
        a = [p for p in ones if p[1] == required_col]
        b = [p for p in zeros if p[1] == required_col]
        first = a[int(rng.integers(len(a)))]
        second = b[int(rng.integers(len(b)))]
        changed[first], changed[second] = False, True
        if len(ones) >= 3 and len(zeros) >= 3 and rng.random() < .55:
            a = [p for p in ones if p != first]
            b = [p for p in zeros if p != second]
            changed[a[int(rng.integers(len(a)))]] = False
            changed[b[int(rng.integers(len(b)))]] = True
    else:
        first = (int(rng.integers(mask.shape[0])), required_col)
        changed[first] = not changed[first]
        rest = [p for p in permitted if p != first]
        rng.shuffle(rest)
        for p in rest[:int(rng.integers(min(4, len(rest))+1))]:
            changed[p] = not changed[p]
        if not np.any(changed) or np.all(changed):
            return _variant(rng, master, required_col, allowed_cols, 'balanced')
    return np.where(changed, colors[1], colors[0]).astype(np.int8)


def _surface(pieces, col, height):
    return min([p['row'] for p in pieces
                if p['col'] <= col < p['col'] + p['bitmap'].shape[1]] or [height])


def _platform_bitmap(master, copies, frame, exception_index, variant):
    ph, pw = master.shape
    a = np.full((ph+2, copies*(pw+1)+1), frame, dtype=np.int8)
    for j in range(copies):
        start = 1+j*(pw+1)
        a[1:-1, start:start+pw] = variant if j == exception_index else master
    return a


def _construct(rng, kind, upper_mode):
    """Compose support geometry bottom-up, then encode its failure contacts."""
    levels = 2 if kind == 'cascade' else int(rng.integers(1, 3))
    if kind == 'test':
        levels = int(rng.integers(3, 5))
    ph = 2 if levels == 4 else int(rng.integers(2, 4 if levels == 3 else 5))
    pw = int(rng.integers(3 if kind == 'redundant' else 2, 6))
    body_h = ph+2
    minimum_h = levels*(body_h+2)+2
    if kind == 'cascade':
        minimum_h = max(minimum_h, 2*(body_h+3)+4)
    height = int(rng.integers(min(minimum_h, 30), 31))
    width = int(rng.integers(max(17, 3*(pw+1)+3), 31))
    max_copies = min(5, (width-3)//(pw+1))
    bg, frame, carrier, motif, q, s, landing, foot = map(int, rng.choice(10, 8, replace=False))
    master = _master(rng, ph, pw, carrier, motif)
    pieces, platforms, rods, command_set = [], [], [], set()
    previous_index = None
    for level in range(levels):
        copies = int(rng.integers(3, max_copies+1))
        body_w = copies*(pw+1)+1
        left = int(rng.integers(1, width-body_w))
        attack = int(rng.integers(copies))
        if kind == 'cascade' and previous_index is not None:
            attack = int(rng.choice([j for j in range(copies) if j != previous_index]))
        previous_index = attack
        tile_left = left+1+attack*(pw+1)
        target_x = tile_left+int(rng.choice([0, pw-1]) if kind == 'redundant'
                                  else rng.integers(pw))
        choices = [left+1+j*(pw+1)+c for j in range(copies) for c in range(pw)]
        support_xs = [target_x]
        outside = [x for x in choices if abs(x-target_x) >= 2
                   and not tile_left <= x < tile_left+pw]
        if kind == 'redundant':
            reserved = [tile_left+c for c in range(pw) if abs(tile_left+c-target_x) >= 2]
            support_xs.append(int(rng.choice(reserved)))
        else:
            support_xs.append(int(rng.choice(outside)))
        rng.shuffle(choices)
        target_count = int(rng.integers(2, 6))
        for x in choices:
            if len(support_xs) >= target_count:
                break
            if all(abs(x-y) >= 2 for y in support_xs):
                support_xs.append(x)
        # Consecutive platforms belong to one interacting stack.  At least one
        # post rests on the preceding platform, rather than bypassing every
        # lower platform and accidentally creating a second floor-level scene.
        if platforms:
            lower = platforms[-1]
            def on_lower(x):
                return lower['col'] <= x < lower['col']+lower['bitmap'].shape[1]
            if not any(on_lower(x) for x in support_xs):
                anchors = [x for x in choices if on_lower(x) and abs(x-target_x) >= 2
                           and not tile_left <= x < tile_left+pw]
                if not anchors:
                    return None
                support_xs[1] = int(rng.choice(anchors))
                # A replaced post must still be separated from every other rod.
                if any(abs(a-b) < 2 for i, a in enumerate(support_xs)
                       for b in support_xs[i+1:]):
                    return None
        # A horizontal foot/shelf can overhang lower platforms and connect posts.
        # At four levels, only very compact gaps are feasible.
        if levels < 4 and rng.random() < .55:
            span = int(rng.integers(2, body_w+1))
            x0 = int(rng.integers(left, left+body_w-span+1))
            row = min(_surface(pieces, x, height) for x in range(x0, x0+span))-1
            rod = _piece(row, x0, np.full((1, span), foot, np.int8), 'rod')
            pieces.append(rod)
            rods.append(rod)
        bottoms = [_surface(pieces, x, height)-1 for x in support_xs]
        gap = (2 if levels == 4 else 3 if kind == 'redundant'
               else int(rng.integers(3, 5) if kind == 'cascade' else rng.integers(2, 4)))
        top = min(bottoms)-gap+1
        platform_row = top-body_h
        if platform_row < 1:
            return None
        tall_count = min(len(support_xs), int(rng.integers(2, 4)))
        if kind == 'redundant':
            tall_count = 2
        if kind == 'cascade' and level == 0:
            tall_count = 2
        contact_colors = {}
        for i, (x, bottom) in enumerate(zip(support_xs, bottoms)):
            rod_top = top
            if i >= tall_count and bottom-top+1 >= 3:
                rod_top += int(rng.integers(1, bottom-top))
            is_tall = rod_top == top
            if kind == 'cascade':
                color = (q if level == 0 else s) if i == 0 else (
                    s if level == 0 and i == 1 else int(rng.choice([q, s])))
                if not is_tall:
                    color = landing
            elif kind == 'redundant':
                color = q if i == 0 else int(rng.choice([s, landing]))
            else:
                color = q if i == 0 else int(rng.choice([q, s, landing]))
            rod = _piece(rod_top, x, np.full((bottom-rod_top+1, 1), color, np.int8), 'rod')
            pieces.append(rod)
            rods.append(rod)
            if is_tall:
                contact_colors[x] = color
        # All nonselected contacts in a harmless witness remain outside its stencil.
        allowed = list(range(pw))
        if kind == 'redundant':
            allowed = [c for c in allowed if contact_colors.get(tile_left+c, q) == q]
        if kind == 'cascade':
            mode = 'balanced' if level == 0 else upper_mode
        elif kind == 'redundant':
            mode = 'remove' if upper_mode == 'add' else 'add'
        else:
            mode = str(rng.choice(['add', 'remove', 'balanced', 'scatter']))
        variant = _variant(rng, master, target_x-tile_left, allowed, mode)
        fault_xs = [tile_left+int(c) for c in np.flatnonzero(np.any(variant != master, axis=0))]
        command_set.update(contact_colors[x] for x in fault_xs if x in contact_colors)
        bitmap = _platform_bitmap(master, copies, frame, attack, variant)
        p = _piece(platform_row, left, bitmap, 'platform')
        pieces.append(p)
        platforms.append(p)
    # Rod identity is independent of orientation. A shelf may share a remotely
    # commanded post's color, provided its input component stays an unambiguous
    # straight run. This also permits removal of intermediate horizontal feet.
    for rod in rods:
        if rod['bitmap'].shape[0] != 1 or rng.random() >= .45:
            continue
        cells = _cells(rod)
        neighbors = {(r+dr, c+dc) for r, c in cells
                     for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
        forbidden = {int(other['bitmap'][0, 0]) for other in rods
                     if other is not rod and _cells(other) & neighbors}
        choices = [color for color in (q, s, landing) if color not in forbidden]
        if kind == 'redundant':
            choices = [color for color in choices if color not in command_set]
        if choices:
            rod['bitmap'][:] = int(rng.choice(choices))
    initial = _render((height, width), bg, pieces)
    # Ground truth uses the constructed bodies and contact commands, while solve
    # independently reconstructs both from the drawing (same physics routine).
    survivors = platforms + [p for p in rods if int(p['bitmap'][0, 0]) not in command_set]
    settled = _settle(survivors, height)
    output = _render(initial.shape, bg, settled)
    info = _decode(initial)
    assert command_set == info['commands']
    assert np.array_equal(output, _answer(info))
    if kind == 'cascade':
        if len(command_set) != 2 or not _features(info)['synergy']:
            return None
        if len(set(_features(info)['platform_distances']) - {0}) < 2:
            return None
    if kind == 'redundant' and any(_features(info)['platform_distances']):
        return None
    return {'input': initial, 'output': output}


def _count(value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("pair counts must be integers")
    if not low <= int(value) <= high:
        raise ValueError("pair count out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nt, ne = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    upper_mode = str(rng.choice(['add', 'remove']))
    requests = [('train', 'cascade'), ('train', 'redundant')]
    requests += [('train', str(rng.choice(['redundant', 'mixed']))) for _ in range(nt-2)]
    requests += [('test', 'test') for _ in range(ne)]
    for split, kind in requests:
        for _ in range(150):
            pair = _construct(rng, kind, upper_mode)
            if pair is not None and _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("bounded scene construction exhausted")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    platforms, rods, commands = info['platforms'], info['rods'], info['commands']
    survivors = platforms + [p for p in rods if int(p['bitmap'][0, 0]) not in commands]
    settled = _settle(survivors, info['grid'].shape[0])
    distances = [q['row']-p['row'] for p, q in zip(survivors, settled)]
    synergy = False
    if len(commands) >= 2:
        necessary = [0]*len(platforms)
        for color in sorted(commands):
            alt = platforms + [p for p in rods
                               if int(p['bitmap'][0, 0]) not in commands-{color}]
            alt = _settle(alt, info['grid'].shape[0])
            for i in range(len(platforms)):
                necessary[i] += settled[i]['row'] > alt[i]['row']
        synergy = any(n >= 2 for n in necessary)
    faults = {(p['row']+p['bitmap'].shape[0], x)
              for p, xs in zip(platforms, info['faults']) for x in xs}
    deleted = [p for p in rods if int(p['bitmap'][0, 0]) in commands]
    return dict(levels=len(platforms), commands=len(commands), rods=len(rods),
                platform_distances=distances[:len(platforms)],
                rod_distances=distances[len(platforms):], synergy=synergy,
                deleted=len(deleted), remote=any(not _cells(p) & faults for p in deleted),
                deleted_horizontal=sum(p['bitmap'].shape[0] == 1 for p in deleted),
                balanced=any(a == b > 0 for a, b in info['edits']),
                pure_a=any(a > 0 and b == 0 for a, b in info['edits']),
                pure_b=any(b > 0 and a == 0 for a, b in info['edits']),
                copies=[(p['bitmap'].shape[1]-1)//(info['master'].shape[1]+1)
                        for p in platforms],
                h_lengths=[p['bitmap'].shape[1] for p in rods if p['bitmap'].shape[0] == 1],
                v_lengths=[p['bitmap'].shape[0] for p in rods if p['bitmap'].shape[1] == 1])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            before = info['platforms'] + info['rods']
            after = _settle(before, inp.shape[0])
            assert [p['row'] for p in before] == [p['row'] for p in after], "input must be stable"
            assert np.array_equal(_answer(info), out)
            assert np.all(out[0] == info['bg'])
            # Fully visible, nonoverlapping bodies conserve every uncommanded
            # foreground color. Selected rod identities disappear completely.
            for color in set(map(int, np.unique(inp))) - {info['bg']}:
                if color in info['commands']:
                    assert not np.any(out == color)
                else:
                    assert int(np.sum(inp == color)) == int(np.sum(out == color))
            f = _features(info)
            assert f['deleted'] >= 1 and info['commands']
            # A bypass of deletion or settling cannot explain a moving witness.
            if any(f['platform_distances']):
                assert not np.array_equal(_answer(info, fall=False), out)
                assert not np.array_equal(_answer(info, commands=set()), out)
            (train if split == 'train' else test).append((info, f))
    assert any(f['synergy'] and len(set(f['platform_distances'])-{0}) >= 2 for _, f in train)
    assert any(f['deleted'] and not any(f['platform_distances']) for _, f in train)
    assert any(f['remote'] for _, f in train), "teach identity removal beyond contact"
    assert any(f['balanced'] for _, f in train), "teach bitmap comparison beyond counts"
    assert any(f['pure_a'] for _, f in train) and any(f['pure_b'] for _, f in train)
    assert any(len(set(i['exception_columns'])) > 1 for i, _ in train)
    assert any(not np.array_equal(_answer(i, _shortcut_commands(i, whole_tile=True)), _answer(i))
               for i, _ in train), "distinguish changed columns from whole odd tile"
    assert any(not np.array_equal(_answer(i, _shortcut_commands(i, additions_only=True)), _answer(i))
               for i, _ in train), "removed cells are faults too"
    assert any(not np.array_equal(_local_cut_answer(i), _answer(i)) for i, _ in train)
    assert max(f['levels'] for _, f in train) == 2
    assert all(f['levels'] >= 3 for _, f in test)


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test() -> None:
    # Known mechanical outcomes check shorter-post landing and simultaneous
    # movement of supported bodies, without pretending to be a separate solver.
    beam = _piece(3, 0, np.full((2, 7), 5, np.int8), 'platform')
    leg_a = _piece(5, 1, np.full((4, 1), 1, np.int8), 'rod')
    leg_b = _piece(5, 5, np.full((4, 1), 2, np.int8), 'rod')
    short = _piece(7, 3, np.full((2, 1), 3, np.int8), 'rod')
    assert _settle([beam, leg_a, leg_b, short], 9)[0]['row'] == 3
    assert _settle([beam, leg_b, short], 9)[0]['row'] == 3
    assert _settle([beam, short], 9)[0]['row'] == 5
    assert _settle([beam], 9)[0]['row'] == 7
    free_stack = [_piece(2, 1, np.full((1, 4), 6, np.int8), 'rod'),
                  _piece(3, 2, np.full((3, 1), 7, np.int8), 'rod')]
    assert [p['row'] for p in _settle(free_stack, 10)] == [6, 7]
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    shapes, palettes, structures, records, masters = set(), set(), set(), [], set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train'] + puzzle['test']:
            info = _decode(pair['input'])
            assert np.array_equal(solve(pair['input']), pair['output'])
            # Neither a training list nor a seed is supplied to this call.
            shapes.add(info['grid'].shape)
            palettes.add((info['bg'], info['frame'], tuple(sorted(np.unique(info['master'])))))
            roles = {}
            structures.add((info['grid'].shape, tuple(roles.setdefault(int(c), len(roles))
                                                      for c in info['grid'].flat)))
            roles = {}
            masters.add((info['master'].shape, tuple(roles.setdefault(int(c), len(roles))
                                                    for c in info['master'].flat)))
            records.append(_features(info))
    assert len(shapes) >= 35 and len(structures) >= 150 and len(palettes) >= 80
    assert len(masters) >= 30
    assert {f['levels'] for f in records} == {1, 2, 3, 4}
    assert len({f['rods'] for f in records}) >= 8
    assert {n for f in records for n in f['copies']} >= {3, 4, 5}
    assert max(d for f in records for d in f['platform_distances']) >= 8
    assert sum(f['synergy'] for f in records) >= 42
    assert sum(bool(f['h_lengths']) for f in records) >= 30
    assert sum(bool(f['deleted_horizontal']) for f in records) >= 3
    assert len({n for f in records for n in f['v_lengths']}) >= 8
    assert _fingerprint(generate_puzzle(777, 7, 3)) == _fingerprint(generate_puzzle(777, 7, 3))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert (len(puzzle['train']), len(puzzle['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("accepted illegal counts")
    # Family-specific negative checks: remove the odd tile or corrupt a result.
    base = puzzles[0]
    for defect in ('missing_exception', 'output_cell', 'broken_frame'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in base[s]]
               for s in ('train', 'test')}
        pair = bad['train'][0]
        info = _decode(pair['input'])
        body, j = info['platforms'][0], info['exception_columns'][0]
        ph, pw = info['master'].shape
        r, c = body['row']+1, body['col']+1+j*(pw+1)
        if defect == 'missing_exception':
            pair['input'][r:r+ph, c:c+pw] = info['master']
        elif defect == 'output_cell':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        else:
            pair['input'][body['row'], body['col']] = info['bg']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted " + defect)


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, legal counts, mechanical and mutation checks)')
