"""Relay printer: printed stencil outlets determine the next symbolic instruction.

Only the standard library and NumPy are used. The input contains its complete
codebook; solve never learns from, or consults, generated examples.
"""
from itertools import product
import copy
import numpy as np


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _runs(flags):
    groups = []
    for i in np.flatnonzero(flags):
        i = int(i)
        if groups and groups[-1][1] == i:
            groups[-1][1] += 1
        else:
            groups.append([i, i + 1])
    return groups


def _components(grid, bg, start):
    remaining = set(map(tuple, np.argwhere(grid != bg)))
    remaining.discard(start)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        result.append(cells)
    return result


def _members(field, bg, start, book):
    live, rejected, bare = [], [], []
    for cells in _components(field, bg, start):
        points = [(r, c, int(field[r, c])) for r, c in cells
                  if int(field[r, c]) in book]
        origin = min(r for r, c in cells), min(c for r, c in cells)
        if len(points) == 1 and points[0][:2] == origin:
            live.append(points[0])
        elif points:
            rejected.extend(points)
        else:
            bare.append(cells)
    return sorted(live), sorted(rejected), bare


def _read(input_grid):
    a = np.asarray(input_grid)
    assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
    assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert np.count_nonzero(a == bg) > a.size // 2
    dividers = [c for c in range(1, a.shape[1] - 1)
                if a[0, c] != bg and np.all(a[:, c] == a[0, c])]
    assert len(dividers) == 1, "one solid separator is required"
    sep = dividers[0]
    field = a[:, sep + 1:].copy()
    starts = np.flatnonzero(field[0] != bg)
    assert len(starts) == 1
    start = (0, int(starts[0]))
    wire = int(field[start])
    assert wire not in (bg, int(a[0, sep]))
    panel = a[:, :sep]
    bands = _runs(np.any(panel != bg, axis=1))
    assert 2 <= len(bands) <= 3
    book = {}
    expected = np.full(panel.shape, bg, dtype=np.int8)
    for top, bottom in bands:
        cols = np.flatnonzero(np.any(panel[top:bottom] != bg, axis=0))
        left, right = int(cols[0]), int(cols[-1]) + 1
        stamp = panel[top:bottom, left:right].copy()
        inlet = np.flatnonzero(stamp[0] != bg)
        outlet = np.flatnonzero(stamp[-1] != bg)
        assert len(inlet) == len(outlet) == 1
        entry, exit_col = int(inlet[0]), int(outlet[0])
        code = int(stamp[0, entry])
        assert stamp[-1, exit_col] == wire
        inks = set(map(int, np.unique(stamp[1:-1]))) - {bg}
        assert len(inks) == 1
        ink = inks.pop()
        assert len({bg, wire, int(a[0, sep]), code, ink}) == 5
        assert code not in book
        assert np.all(np.any(stamp != bg, axis=0))
        assert 3 <= stamp.shape[0] <= 4 and 3 <= stamp.shape[1] <= 5
        assert len(_components(stamp, bg, (-1, -1))) == 1
        # Literal body geometry is used; no catalog or puzzle-specific state.
        printed = stamp.copy()
        printed[0, entry] = ink
        book[code] = dict(stamp=stamp, printed=printed, ink=ink,
                          entry=entry, exit=exit_col)
        expected[top:bottom, left:right] = stamp
    assert np.array_equal(panel, expected)
    roles = [bg, int(a[0, sep]), wire] + list(book) + [b['ink'] for b in book.values()]
    assert len(set(roles)) == len(roles)
    assert len({b['stamp'].shape for b in book.values()}) == 1
    assert set(map(int, np.unique(field))) <= set(roles) - {int(a[0, sep])}
    live, rejected, bare = _members(field, bg, start, book)
    return dict(bg=bg, separator=int(a[0, sep]), wire=wire, start=start,
                field=field, panel=panel, book=book, live=live,
                rejected=rejected, bare=bare)


def _trace(info, *, all_codes=False, fixed_column=False):
    """Input-only control flow, independent of output ink or prior printing."""
    field, book = info['field'], info['book']
    if all_codes:
        live = [(int(r), int(c), int(field[r, c]))
                for r, c in np.argwhere(np.isin(field, list(book)))]
    else:
        live = info['live']
    row, col = info['start']
    route = []
    while 0 <= row < field.shape[0] and 0 <= col < field.shape[1]:
        candidates = [p for p in live if p[1] == col and p[0] > row]
        if not candidates:
            break
        r, c, code = min(candidates)
        route.append((r, c, code))
        b = book[code]
        row = r + b['stamp'].shape[0] - 1
        col = c if fixed_column else c + b['exit'] - b['entry']
    return route


def _paste(out, r, c, b, bg):
    h, w = b['printed'].shape
    left = c - b['entry']
    # General intersection also makes the ablation functions well-defined.
    for y in range(h):
        for x in range(w):
            value = int(b['printed'][y, x])
            if value != bg and 0 <= r + y < out.shape[0] and 0 <= left + x < out.shape[1]:
                out[r + y, left + x] = value


def _render(shape, bg, wire, start, book, route, fixed_column=False):
    out = np.full(shape, bg, dtype=np.int8)
    out[start] = wire
    row, col = start
    for r, c, code in route:
        assert c == col and r > row
        if 0 <= col < shape[1]:
            out[max(0, row + 1):min(r, shape[0]), col] = wire
        b = book[code]
        _paste(out, r, c, b, bg)
        row = r + b['stamp'].shape[0] - 1
        col = c if fixed_column else c + b['exit'] - b['entry']
    if 0 <= col < shape[1] and row < shape[0]:
        out[row + 1:, col] = wire
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    return _render(info['field'].shape, info['bg'], info['wire'], info['start'],
                   info['book'], _trace(info))


def _parameters(rng):
    k = int(rng.integers(2, 4))
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 6))
    palette = list(map(int, rng.permutation(10)))
    bg, sep, wire = palette[:3]
    book = {}
    for i in range(k):
        code, ink = palette[3 + i], palette[3 + k + i]
        if i == 0:
            entry, exit_col = 0, int(rng.integers(1, w))
        elif i == 1:
            entry, exit_col = w - 1, int(rng.integers(0, w - 1))
        else:
            entry = exit_col = int(rng.integers(w))
        stamp = np.full((h, w), bg, dtype=np.int8)
        stamp[0, entry], stamp[-1, exit_col] = code, wire
        if h == 3:
            stamp[1, :] = ink
        else:
            # One full interval gives tight width; a second varying interval
            # joins the adjacent port to that interval with eight-connectivity.
            if rng.integers(2):
                stamp[1, :] = ink
                lo = int(rng.integers(0, exit_col + 1))
                hi = int(rng.integers(exit_col + 1, w + 1))
                stamp[2, lo:hi] = ink
            else:
                stamp[2, :] = ink
                lo = int(rng.integers(0, entry + 1))
                hi = int(rng.integers(entry + 1, w + 1))
                stamp[1, lo:hi] = ink
        printed = stamp.copy()
        printed[0, entry] = ink
        book[code] = dict(stamp=stamp, printed=printed, ink=ink,
                          entry=entry, exit=exit_col)
    order = [int(c) for c in rng.permutation(list(book))]
    return bg, sep, wire, book, order


def _clear(field, cells, bg):
    H, W = field.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if np.any(field[max(0, r-1):min(H, r+2), max(0, c-1):min(W, c+2)] != bg):
            return False
    return True


def _artifact_cells(book, code, top, left, bg, attached):
    b = book[code]
    cells = {(top + int(y), left + int(x)): int(b['printed'][y, x])
             for y, x in np.argwhere(b['printed'] != bg)}
    if attached:
        cells[(top + b['stamp'].shape[0], left + b['exit'] + 1)] = code
    return cells


def _assemble(field, bg, sep, book, order):
    h, w = next(iter(book.values()))['stamp'].shape
    panel_w = w + 2
    a = np.full((field.shape[0], panel_w + 1 + field.shape[1]), bg, dtype=np.int8)
    for i, code in enumerate(order):
        top = 1 + i * (h + 2)
        a[top:top+h, 1:1+w] = book[code]['stamp']
    a[:, panel_w] = sep
    a[:, panel_w+1:] = field
    return a


def _make_pair(rng, parameters, n, clipped, on_ray, painted):
    bg, sep, wire, book, order = parameters
    codes = list(book)
    h, w = book[codes[0]]['stamp'].shape
    legend_h = 2 + len(book) * h + (len(book) - 1) * 2
    W = int(rng.integers(12, 30 - w - 3 + 1))
    first_row = h + 3 if on_ray else int(rng.integers(2, 4))
    # A larger first gap holds the old-column continuation trap.
    rows = [first_row]
    for j in range(n - 1):
        rows.append(rows[-1] + h - 1 + (4 if j == 0 else 2))
    room = 30 - rows[-1] - (1 if clipped else h)
    if room < 0:
        return None
    for j in range(1, n):
        extra = int(rng.integers(min(1, room) + 1))
        for q in range(j, n):
            rows[q] += extra
        room -= extra
    if clipped:
        visible = int(rng.integers(1, min(h - 1, 30 - rows[-1]) + 1))
        H = max(legend_h, rows[-1] + visible)
        H = int(rng.integers(H, min(30, H + 3) + 1))
        rows[-1] = H - visible
    else:
        minimum = max(legend_h, rows[-1] + h)
        H = int(rng.integers(minimum, min(30, minimum + 3) + 1))
    possibilities = []
    for program in product(codes, repeat=n):
        if set(program) != set(codes):
            continue
        if painted and program[0] != codes[0]:
            continue
        first = book[program[0]]
        if first['entry'] == first['exit']:
            continue
        cursor, lower, upper = 0, 0, W - 1
        for code in program:
            b = book[code]
            lower = max(lower, b['entry'] - cursor)
            upper = min(upper, W - w + b['entry'] - cursor)
            cursor += b['exit'] - b['entry']
        if on_ray:
            # Use the right-shifting stencil for the obsolete on-ray object.
            b = book[codes[0]]
            lower = max(lower, b['exit'] + 1)
            upper = min(upper, W - w + b['exit'] + 1)
        if lower <= upper:
            possibilities.append((program, lower, upper))
    if not possibilities:
        return None
    program, lower, upper = possibilities[int(rng.integers(len(possibilities)))]
    col = int(rng.integers(lower, upper + 1))
    start = (0, col)
    route = []
    for r, code in zip(rows, program):
        route.append((r, col, code))
        col += book[code]['exit'] - book[code]['entry']
    field = np.full((H, W), bg, dtype=np.int8)
    field[start] = wire
    for r, c, code in route:
        field[r, c] = code
    if painted:
        # A correctly registered origin-code remains executable even though it
        # touches existing material. This distinguishes the bounding-origin
        # rule from rejecting every code that has an ink neighbor.
        r, c, code = route[0]
        b = book[code]
        cells = {(r + int(y), c + int(x)): int(b['stamp'][y, x])
                 for y, x in np.argwhere(b['stamp'] != bg)}
        for p, value in cells.items():
            field[p] = value
    trap = (rows[0] + h + 1, start[1])
    assert _clear(field, [trap], bg)
    field[trap] = int(rng.choice(codes))
    if on_ray:
        artifact_code = codes[0]
        b = book[artifact_code]
        cells = _artifact_cells(book, artifact_code, 1,
                                start[1] - b['exit'] - 1, bg, True)
        if not _clear(field, cells, bg):
            return None
        for p, value in cells.items():
            field[p] = value
    # Additional artifacts are exact legend material, never arbitrary junk.
    for _ in range(int(rng.integers(0, 3))):
        for attempt in range(60):
            code = int(rng.choice(codes))
            attached = bool(rng.integers(2))
            top = int(rng.integers(1, max(2, H - h)))
            left = int(rng.integers(W - w + 1))
            cells = _artifact_cells(book, code, top, left, bg, attached)
            if _clear(field, cells, bg):
                for p, value in cells.items():
                    field[p] = value
                break
    # Free off-route singletons are accepted only if control flow stays fixed.
    for _ in range(int(rng.integers(2, 8))):
        for attempt in range(60):
            p = int(rng.integers(1, H)), int(rng.integers(W))
            if not _clear(field, [p], bg):
                continue
            field[p] = int(rng.choice(codes))
            live, _, _ = _members(field, bg, start, book)
            info = dict(field=field, book=book, start=start, live=live)
            if _trace(info) == route:
                break
            field[p] = bg
    inp = _assemble(field, bg, sep, book, order)
    if np.count_nonzero(inp == bg) <= inp.size // 2:
        return None
    info = _read(inp)
    if _trace(info) != route:
        return None
    out = _render(field.shape, bg, wire, start, book, route)
    return dict(input=inp, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    nt = _integer('num_train', num_train, 2, 10)
    nq = _integer('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    parameters = _parameters(rng)
    k = len(parameters[3])
    puzzle, seen = dict(train=[], test=[]), set()
    for index in range(nt + nq):
        split = 'train' if index < nt else 'test'
        n = k if index < 2 else int(rng.integers(k, 4)) if split == 'train' else int(rng.integers(4, 6))
        clipped = index == 1 or (index >= 2 and bool(rng.integers(2)))
        on_ray = index == 0 or (index >= 2 and n <= 4 and rng.random() < .35)
        painted = index == 1 or (index >= 2 and rng.random() < .3)
        for attempt in range(100):
            pair = _make_pair(rng, parameters, n, clipped, on_ray, painted)
            if pair is not None and _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('bounded placement search exhausted')
    return puzzle


def _signature(info):
    return (info['bg'], info['separator'], info['wire'],
            tuple((code, _key(b['stamp'])) for code, b in info['book'].items()))


def _check_material(info):
    field, bg, book = info['field'], info['bg'], info['book']
    by_ink = {b['ink']: code for code, b in book.items()}
    for cells in _components(field, bg, info['start']):
        if len(cells) == 1 and int(field[cells[0]]) in book:
            continue
        inks = {int(field[p]) for p in cells} & set(by_ink)
        assert len(inks) == 1, 'unexplained material'
        code = by_ink[inks.pop()]
        b = book[code]
        r = min(p[0] for p in cells)
        c = min(p[1] for p in cells)
        points = [p for p in cells if int(field[p]) in book]
        assert len(points) <= 1
        if points == [(r, c)]:
            expected = {(r + int(y), c + int(x)): int(b['stamp'][y, x])
                        for y, x in np.argwhere(b['stamp'] != bg)}
        else:
            expected = _artifact_cells(book, code, r, c, bg, bool(points))
        assert {p: int(field[p]) for p in cells} == expected


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4 and len(puzzle['train']) > len(puzzle['test'])
    infos, lengths, seen, signatures = dict(train=[], test=[]), dict(train=[], test=[]), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                a = np.asarray(value)
                assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            assert _key(pair['input']) not in seen
            seen.add(_key(pair['input']))
            info = _read(pair['input'])
            out = np.asarray(pair['output'])
            assert out.shape == info['field'].shape
            assert np.array_equal(solve(np.asarray(pair['input']).copy()), out), 'incorrect output'
            _check_material(info)
            route = _trace(info)
            assert set(p[2] for p in route) == set(info['book'])
            assert len(info['live']) > len(route), 'missing unused live instruction'
            h, w = next(iter(info['book'].values()))['stamp'].shape
            previous_bottom = 0
            for r, c, code in route:
                b = info['book'][code]
                assert 0 <= c - b['entry'] <= out.shape[1] - w
                assert r > previous_bottom
                previous_bottom = r + h - 1
            # Omitting A: execute every eligible symbol, retaining the true wire.
            stamp_all = out.copy()
            for r, c, code in info['live']:
                _paste(stamp_all, r, c, info['book'][code], info['bg'])
            assert not np.array_equal(stamp_all, out), 'A omission does not change answer'
            # Omitting B's geometric control: remain in the inlet column.
            flat_route = _trace(info, fixed_column=True)
            flat = _render(out.shape, info['bg'], info['wire'], info['start'],
                           info['book'], flat_route, fixed_column=True)
            assert flat_route != route and not np.array_equal(flat, out)
            signatures.add(_signature(info))
            infos[split].append(info)
            lengths[split].append(len(route))
    assert len(signatures) == 1, 'legend or palette changed inside puzzle'
    assert max(lengths['train']) <= 3 and min(lengths['test']) >= 4
    assert min(lengths['test']) > max(lengths['train'])
    first, second = infos['train'][:2]
    h = next(iter(first['book'].values()))['stamp'].shape[0]
    assert _trace(first)[-1][0] + h <= first['field'].shape[0]
    assert _trace(second)[-1][0] + h > second['field'].shape[0]
    assert first['rejected']
    assert _trace(first, all_codes=True) != _trace(first), 'on-ray membership not taught'
    first_live = _trace(second)[0][:2]
    assert any(first_live in cells and len(cells) > 1
               for cells in _components(second['field'], second['bg'], second['start'])), 'valid pre-rendering not taught'


def _puzzle_key(p):
    return tuple((s, _key(pair['input']), _key(pair['output']))
                 for s in ('train', 'test') for pair in p[s])


def self_test():
    puzzles, shapes, masks, shifts, starts, lengths, classes, normalized = [], set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in p:
            for pair in p[split]:
                info = _read(pair['input'])
                shapes.add(pair['input'].shape)
                lengths.add((split, len(_trace(info))))
                starts.add(info['start'][1])
                classes.add(len(info['book']))
                for b in info['book'].values():
                    masks.add(_key(b['stamp'] != info['bg']))
                    shifts.add(b['exit'] - b['entry'])
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(v), len(roles))
                                                           for v in pair['input'].flat)))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(masks) >= 20 and len(normalized) >= 150
    assert classes == {2, 3} and len(starts) >= 8
    assert {s for s in shifts if s > 0} and {s for s in shifts if s < 0} and 0 in shifts
    assert {n for split, n in lengths if split == 'train'} == {2, 3}
    assert {n for split, n in lengths if split == 'test'} == {4, 5}
    assert _puzzle_key(puzzles[9]) == _puzzle_key(generate_puzzle(9))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(314159, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(7, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = copy.deepcopy(puzzles[0])
    query = damaged['test'][0]
    info = _read(query['input'])
    r, c, code = _trace(info)[0]
    b = info['book'][code]
    h, w = b['stamp'].shape
    query['output'][r:r+h, c-b['entry']:c-b['entry']+w] = info['bg']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('deleted complete live stencil accepted')
    # Hand-computed registration: inlet at right edge, outlet at left edge.
    b = dict(printed=np.array([[0, 0, 6], [6, 6, 6], [9, 0, 0]], dtype=np.int8),
             stamp=np.array([[0, 0, 2], [6, 6, 6], [9, 0, 0]], dtype=np.int8),
             entry=2, exit=0, ink=6)
    actual = _render((7, 5), 0, 9, (0, 3), {2: b}, [(2, 3, 2)])
    expected = np.array([[0,0,0,9,0], [0,0,0,9,0], [0,0,0,6,0],
                         [0,6,6,6,0], [0,9,0,0,0], [0,9,0,0,0],
                         [0,9,0,0,0]], dtype=np.int8)
    assert np.array_equal(actual, expected)


if __name__ == '__main__':
    self_test()
    print('42-seed relay-printer self-test passed')
