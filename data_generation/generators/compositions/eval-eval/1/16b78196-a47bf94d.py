"""Printed jigsaw circuit repair, jointly constrained by endpoint labels.

Only NumPy and the standard library are used. The parser, exact-cover search,
and cable tracer receive input cells only. Generation constructs its answer
from the chosen printed fragments, then checks it against that search.
"""
from itertools import combinations
import json
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError('expected an integer ARC grid with dimensions 1..30')
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _runs(values):
    padded = np.r_[False, values, False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded) == 1),
                    np.flatnonzero(np.diff(padded) == -1)))


def _terminal(block, bg):
    if np.all(block == bg):
        return 'blank', None
    colors = set(map(int, np.unique(block))) - {bg}
    if len(colors) != 1:
        raise ValueError('multicolor terminal')
    label = colors.pop()
    if np.all(block == label):
        return 'solid', label
    for kind, mask in (('diamond', DIAMOND), ('checker', CHECKER)):
        if np.array_equal(block, np.where(mask, label, bg)):
            return kind, label
    raise ValueError('invalid terminal stencil')


def _parse(g):
    bg = int(g[0, 0])
    if any(int(g[r, c]) != bg for r, c in
           ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError('background corners disagree')
    candidates = []
    for transpose in (False, True):
        a = g.T if transpose else g
        h, w = a.shape
        for substrate in sorted(set(map(int, np.unique(a))) - {bg}):
            rails = {}
            for c in range(w):
                for r0, r1 in _runs(a[:, c] == substrate):
                    if r1 - r0 >= 12:
                        rails.setdefault((int(r0), int(r1)), []).append(c)
            for (r0, r1), cols in rails.items():
                if r0 < 3 or r1 + 3 > h:
                    continue
                for left, right in combinations(cols, 2):
                    inner_w = right - left - 1
                    if not 7 <= inner_w <= 11:
                        continue
                    try:
                        top = a[r0, left + 1:right]
                        bottom = a[r1 - 1, left + 1:right]
                        wires = set(map(int, np.unique(top))) - {substrate}
                        assert len(wires) == 1 and bg not in wires
                        wire = wires.pop()
                        centers = list(map(int, np.flatnonzero(top == wire) + left + 1))
                        assert len(centers) in (2, 3)
                        assert len(set(np.diff(centers))) == 1
                        assert 4 <= centers[1] - centers[0] <= 8
                        assert centers[0] > left + 1 and centers[-1] < right - 1
                        assert np.array_equal(top, bottom)
                        footprints = np.zeros(a.shape, dtype=bool)
                        banks = []
                        for row in (r0 - 3, r1):
                            bank = []
                            for c in centers:
                                footprints[row:row + 3, c - 1:c + 2] = True
                                bank.append(_terminal(a[row:row + 3, c - 1:c + 2], bg))
                            banks.append(bank)
                        board = np.zeros(a.shape, dtype=bool)
                        board[r0:r1, left:right + 1] = True
                        outside = ~board & ~footprints & (a != bg)
                        pieces = _components(outside)
                        assert 2 <= len(pieces) <= 4
                        assert all(np.ptp(p[:, 1]) + 1 == inner_w for p in pieces)
                        cross_colors = (set(map(int, a[board | outside]))
                                        - {bg, substrate, wire})
                        assert len(cross_colors) == 1
                        cross = cross_colors.pop()
                        structural = {bg, substrate, wire, cross}
                        assert all(v is None or v not in structural
                                   for bank in banks for _, v in bank)
                        phases = [phase for phase in (0, 1)
                                  if all(k in ('blank', 'solid', 'diamond')
                                         for k, _ in banks[phase])
                                  and all(k in ('blank', 'checker')
                                          for k, _ in banks[1 - phase])]
                        assert len(phases) == 1
                        holes = board & (a == bg)
                        assert holes.any()
                        assert np.count_nonzero(holes) == sum(len(p) for p in pieces)
                        # The exterior has only bank clues and separated stock.
                        assert all(substrate in a[p[:, 0], p[:, 1]] for p in pieces)
                        for p in pieces:
                            for r, c in p:
                                for dr, dc in DIRS:
                                    rr, cc = r + dr, c + dc
                                    if 0 <= rr < h and 0 <= cc < w:
                                        assert not board[rr, cc] and not footprints[rr, cc]
                        base = a.copy()
                        for p in pieces:
                            base[p[:, 0], p[:, 1]] = bg
                        candidates.append(dict(grid=a, transpose=transpose, bg=bg,
                                               substrate=substrate, wire=wire, cross=cross,
                                               r0=r0, r1=r1, left=left, right=right,
                                               centers=centers, banks=banks, phase=phases[0],
                                               board=board, footprints=footprints,
                                               holes=holes, pieces=pieces, base=base))
                    except (AssertionError, ValueError):
                        continue
    if len(candidates) != 1:
        raise ValueError(f'expected one board interpretation, got {len(candidates)}')
    return candidates[0]


def _packings(info):
    """Exhaust all translations exact-covering the visible missing region."""
    a, holes = info['grid'], info['holes']
    hole_coords = list(map(tuple, np.argwhere(holes)))
    indices = {p: i for i, p in enumerate(hole_coords)}
    options, by_cell = [], [[] for _ in hole_coords]
    for i, source in enumerate(info['pieces']):
        local = source - source.min(axis=0)
        colors = a[source[:, 0], source[:, 1]]
        hh, ww = local.max(axis=0) + 1
        # Width is a visible family constraint, so all horizontal offsets are
        # tested (not a hidden scatter-column or stored target convention).
        for row in range(info['r0'] + 1, info['r1'] - int(hh) + 1):
            for col in range(info['left'] + 1, info['right'] - int(ww) + 1):
                target = local + (row, col)
                if not np.all(holes[target[:, 0], target[:, 1]]):
                    continue
                bits = sum(1 << indices[tuple(p)] for p in target)
                number = len(options)
                options.append((i, bits, target, colors))
                for p in target:
                    by_cell[indices[tuple(p)]].append(number)
    full = (1 << len(hole_coords)) - 1
    answers = {}

    def search(occupied, used, chosen):
        if occupied == full:
            if used != (1 << len(info['pieces'])) - 1:
                return
            board = info['base'].copy()
            for number in chosen:
                _, _, cells, colors = options[number]
                board[cells[:, 0], cells[:, 1]] = colors
            answers.setdefault(board.tobytes(), (board, [options[k] for k in chosen]))
            return
        best = None
        for k in range(len(hole_coords)):
            if occupied >> k & 1:
                continue
            choices = [q for q in by_cell[k]
                       if not (used >> options[q][0] & 1)
                       and not (occupied & options[q][1])]
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) <= 1:
                    break
        for q in best:
            i, bits, _, _ = options[q]
            search(occupied | bits, used | (1 << i), chosen + [q])

    search(0, 0, [])
    return list(answers.values())


def _trace(board, info):
    wire, cross = info['wire'], info['cross']
    visible = set(map(tuple, np.argwhere(info['board'] &
                                      ((board == wire) | (board == cross)))))
    starts = {(info['r0'], c) for c in info['centers']}
    ends = {(info['r1'] - 1, c) for c in info['centers']}
    graph = {p: {d: (p[0] + dr, p[1] + dc)
                 for d, (dr, dc) in enumerate(DIRS)
                 if (p[0] + dr, p[1] + dc) in visible} for p in visible}
    for p, edges in graph.items():
        if p in starts | ends:
            assert len(edges) == 1
        elif board[p] == cross:
            assert len(edges) == 4, 'crossing must have four orthogonal arms'
        else:
            assert len(edges) == 2, 'broken, branching, or side-touching cable'
    permutation, routes, all_seen = [], [], set()
    for c in info['centers']:
        p, incoming, states, path = (info['r0'], c), 0, set(), []
        while True:
            assert (p, incoming) not in states, 'cable cycle'
            states.add((p, incoming))
            path.append(p)
            all_seen.add(p)
            if p in ends:
                permutation.append(info['centers'].index(p[1]))
                break
            if p in starts:
                assert len(path) == 1, 'route returns to its starting bank'
                outgoing = 2
            elif board[p] == cross:
                outgoing = (incoming + 2) % 4
            else:
                assert incoming in graph[p]
                outgoing = next(d for d in graph[p] if d != incoming)
            p = graph[p][outgoing]
            incoming = (outgoing + 2) % 4
        routes.append(path)
    assert sorted(permutation) == list(range(len(info['centers'])))
    assert all_seen == visible, 'unattached wire fragments'
    return tuple(permutation), routes


def _complete(board, info, permutation):
    answer = board.copy()
    labels, clues = [], []
    for i, j in enumerate(permutation):
        ends = (info['banks'][0][i], info['banks'][1][j])
        colors = {v for _, v in ends if v is not None}
        assert len(colors) == 1, 'missing or contradictory route label'
        color = colors.pop()
        labels.append(color)
        clues.append(ends)
        for bank, k, row in ((0, i, info['r0'] - 3), (1, j, info['r1'])):
            c = info['centers'][k]
            mask = DIAMOND if bank == info['phase'] else CHECKER
            answer[row:row + 3, c - 1:c + 2] = np.where(mask, color, info['bg'])
    return answer, labels, clues


def _resolve(g):
    info = _parse(g)
    layouts = _packings(info)
    answers, traceable = {}, []
    for board, placements in layouts:
        try:
            permutation, routes = _trace(board, info)
            traceable.append(permutation)
            output, labels, clues = _complete(board, info, permutation)
            answers.setdefault(output.tobytes(),
                               (output, placements, permutation, routes, labels, clues))
        except (AssertionError, KeyError, StopIteration):
            continue
    if len(answers) != 1:
        raise ValueError(f'expected one clue-consistent repair, got {len(answers)}')
    output, placements, permutation, routes, labels, clues = next(iter(answers.values()))
    info.update(placements=placements, permutation=permutation, routes=routes,
                labels=labels, clues=clues, geometric_repairs=len(layouts),
                traceable_permutations=tuple(sorted(set(traceable))))
    return (output.T.copy() if info['transpose'] else output.copy()), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using no demonstrations, seed, generator state, or answer."""
    return _resolve(_grid(input_grid))[0]


def _profile(rng, width, ports, exclude=()):
    for _ in range(80):
        p = rng.integers(0, 2, width)
        p[ports] = 0
        if p.any() and tuple(p) not in exclude:
            return p
    # Exhaust the finite profile space; this fallback changes no convention.
    free = [c for c in range(width) if c not in ports]
    for code in range(1, 1 << len(free)):
        p = np.zeros(width, dtype=int)
        for i, c in enumerate(free):
            p[c] = (code >> i) & 1
        if tuple(p) not in exclude:
            return p
    raise RuntimeError('profile space exhausted')


def _module(width, ports, height, swap):
    """Construct the printed wire field and its top-to-bottom permutation."""
    paths = [[(0, c)] for c in ports]
    state = list(range(len(ports)))
    moves = []
    if swap is not None:
        a, b = swap, swap + 1
        # A spare column between lanes permits an orthogonal transposition,
        # with three independently traceable moves and one marked crossing.
        spare = ports[b] + 2 if b < len(ports) - 1 or len(ports) == 2 else ports[a] - 2
        if spare >= width:
            spare = ports[a] - 2
        assert 0 <= spare < width and spare not in ports
        columns = ports + [spare]
        state += [None]
        moves = [(1, a, len(ports)), (3, b, a), (5, len(ports), b)]
    else:
        columns = ports
    previous = 0
    for row, old, new in moves:
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        route = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[route].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((r, columns[lane]) for r in range(previous + 1, height))
    field = {}
    for path in paths:
        for p in path:
            field[p] = field.get(p, 0) + 1
    permutation = tuple(ports.index(path[-1][1]) for path in paths)
    return field, permutation


def _make_pair(rng, n, count, lesson, rotation, reflect):
    palette = list(map(int, rng.permutation(10)))
    bg, substrate, wire, cross = palette[:4]
    if n == 3:
        pitch, width = 4, 11
    else:
        pitch = int(rng.integers(4, 7))
        width = pitch + 5  # room for an exterior spare track
    ports = [1 + i * pitch for i in range(n)]
    common = _profile(rng, width, ports)
    types = [0, 1] if n == 3 else [0]
    if n == 3:
        rng.shuffle(types)
    while len(types) < count:
        types.insert(0 if rng.random() < .5 else len(types), None)
    heights = [7 if s is not None else int(rng.integers(2, 5)) for s in types]
    if sum(heights) < 19 and rng.random() < .45:
        indices = [i for i, s in enumerate(types) if s is not None]
        heights[int(rng.choice(indices))] += 1
    if sum(heights) > 19:
        heights = [7 if s is not None else 2 for s in types]
    profiles = [None] * (count + 1)
    if n == 3:
        for i, s in enumerate(types):
            if s is not None:
                profiles[i] = common.copy()
                profiles[i + 1] = common.copy()
        used = {tuple(common)}
        for i, p in enumerate(profiles):
            if p is None:
                profiles[i] = _profile(rng, width, ports, used)
                used.add(tuple(profiles[i]))
    else:
        used = set()
        for i in range(count + 1):
            profiles[i] = _profile(rng, width, ports, used)
            used.add(tuple(profiles[i]))
    length = sum(heights)
    local_pieces, composite = [], list(range(n))
    offset = 0
    for k, (height, swap) in enumerate(zip(heights, types)):
        cells = np.array([(offset + r, c) for c in range(width)
                          for r in range(int(profiles[k][c]), height + int(profiles[k + 1][c]))], dtype=int)
        colors = np.full(len(cells), substrate, dtype=np.int8)
        field, perm = _module(width, ports, height, swap)
        for j, (r, c) in enumerate(cells):
            visits = field.get((r - offset, c), 0)
            if visits:
                colors[j] = cross if visits > 1 else wire
        # Every constructed wire cell must actually be printed on its tile.
        assert all((r + offset, c) in set(map(tuple, cells)) for r, c in field)
        local_pieces.append((cells, colors))
        composite = [perm[j] for j in composite]
        offset += height
    # Board and inventory share one canvas; neither is an independent puzzle.
    stock_heights = [int(np.ptp(p[:, 0])) + 1 for p, _ in local_pieces]
    stock_min = sum(stock_heights) + count - 1 + 2
    board_min = length + int(profiles[-1].max()) + 10
    minimum = max(stock_min, board_min)
    assert minimum <= 30
    h = int(rng.integers(minimum, 31))
    c0 = width + int(rng.integers(3, 5))
    c1 = c0 + width + 1
    w = c1 + int(rng.integers(2, 4))
    assert w <= 30
    top_margin = int(rng.integers(1, h - (length + int(profiles[-1].max()) + 8) + 1))
    r0 = top_margin + 3
    base_row = r0 + 1
    r1 = base_row + length + int(profiles[-1].max()) + 1
    inp = np.full((h, w), bg, dtype=np.int8)
    inp[r0:r1, c0:c1 + 1] = substrate
    for c in range(width):
        inp[base_row + int(profiles[0][c]):base_row + length + int(profiles[-1][c]), c0 + 1 + c] = bg
    for c in ports:
        inp[r0, c0 + 1 + c] = wire
        inp[base_row + length:r1, c0 + 1 + c] = wire
    out = inp.copy()
    for cells, colors in local_pieces:
        dest = cells + (base_row, c0 + 1)
        assert np.all(out[dest[:, 0], dest[:, 1]] == bg)
        out[dest[:, 0], dest[:, 1]] = colors
    # Scatter the printed fragments in an independently permuted stock column.
    stock_order = list(map(int, rng.permutation(count)))
    extra = h - (sum(stock_heights) + count - 1) - 2
    stock_row = int(rng.integers(1, extra + 2))
    remaining_extra = h - stock_row - (sum(stock_heights) + count - 1) - 1
    for q, i in enumerate(stock_order):
        cells, colors = local_pieces[i]
        local = cells - cells.min(axis=0)
        col = int(rng.integers(1, c0 - width))
        source = local + (stock_row, col)
        inp[source[:, 0], source[:, 1]] = colors
        stock_row += stock_heights[i] + 1
        if q < count - 1 and remaining_extra:
            gap = int(rng.integers(remaining_extra + 1))
            stock_row += gap
            remaining_extra -= gap
    labels = palette[4:4 + n]
    rng.shuffle(labels)
    if n == 2 and lesson == 'random' and rng.random() < .25:
        labels[1] = labels[0]
    near = ['solid'] * n
    far = ['blank'] * n
    if lesson != 'forward':
        near = ['diamond' if rng.random() < .65 else 'solid' for _ in range(n)]
        blank = int(rng.integers(n))
        near[blank] = 'blank'
        far[composite[blank]] = 'checker'
        if lesson == 'reverse':
            near[next(i for i in range(n) if i != blank)] = 'diamond'
    # One agreeing double clue distinguishes the two noncommuting orders.
    if n == 3:
        reverse_types = list(types)
        slots = [i for i, s in enumerate(types) if s is not None]
        reverse_types[slots[0]], reverse_types[slots[1]] = reverse_types[slots[1]], reverse_types[slots[0]]
        alt = list(range(n))
        for swap in reverse_types:
            perm = list(range(n))
            if swap is not None:
                perm[swap], perm[swap + 1] = perm[swap + 1], perm[swap]
            alt = [perm[j] for j in alt]
        choices = [i for i in range(n) if near[i] != 'blank'
                   and near[alt.index(composite[i])] != 'blank'
                   and labels[alt.index(composite[i])] != labels[i]]
        assert choices
        double = int(rng.choice(choices))
    else:
        double = next(i for i in range(n) if near[i] != 'blank')
    far[composite[double]] = 'checker'
    if lesson == 'random':
        for i in range(n):
            if rng.random() < .4:
                far[composite[i]] = 'checker'
    for i, j in enumerate(composite):
        for bank, index, row, mask, mode in (
                (0, i, r0 - 3, DIAMOND, near[i]),
                (1, j, r1, CHECKER, far[j])):
            center = c0 + 1 + ports[index]
            out[row:row + 3, center - 1:center + 2] = np.where(mask, labels[i], bg)
            block = np.full((3, 3), bg if mode == 'blank' else labels[i], dtype=np.int8)
            if mode in ('diamond', 'checker'):
                block = np.where(mask, labels[i], bg)
            inp[row:row + 3, center - 1:center + 2] = block
    if reflect:
        inp, out = inp[:, ::-1], out[:, ::-1]
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    predicted, info = _resolve(inp)
    if not np.array_equal(predicted, out):
        raise ValueError('construction disagrees with joint repair search')
    if n == 3:
        assert info['geometric_repairs'] >= 2
        assert len(info['traceable_permutations']) >= 2
    return {'input': inp, 'output': out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # extra fragments, orientation transfer, recombination
    base_rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, total in (('train', nt), ('test', nq)):
        for i in range(total):
            lesson = ('forward' if i == 0 else 'reverse' if i == 1 else 'random') if split == 'train' else 'random'
            n = 3 if lesson in ('forward', 'reverse') or split == 'test' else int(rng.integers(2, 4))
            if mode == 0:
                count = int(rng.integers(2, 4)) if split == 'train' else 4
            else:
                count = int(rng.integers(2, 5))
            if mode == 1:
                rotation = (base_rotation + (split == 'test')) % 4
            elif mode == 2:
                rotation = int(rng.integers(4))
            else:
                rotation = base_rotation
            for _ in range(100):
                try:
                    pair = _make_pair(rng, n, count, lesson, rotation, bool(rng.integers(2)))
                except (AssertionError, ValueError):
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-repair construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen
            seen.add(key)
            predicted, info = _resolve(inp)
            assert np.array_equal(predicted, out)
            canonical_out = out.T if info['transpose'] else out
            used, destination = set(), np.zeros_like(info['holes'])
            mutable = info['footprints'].copy()
            for i, bits, cells, colors in info['placements']:
                assert i not in used
                used.add(i)
                source = info['pieces'][i]
                assert np.array_equal(source - source.min(axis=0), cells - cells.min(axis=0))
                assert not np.array_equal(source, cells)
                assert np.array_equal(info['grid'][source[:, 0], source[:, 1]], colors)
                assert np.array_equal(canonical_out[cells[:, 0], cells[:, 1]], colors)
                assert np.all(canonical_out[source[:, 0], source[:, 1]] == info['bg'])
                assert not np.any(destination[cells[:, 0], cells[:, 1]])
                destination[cells[:, 0], cells[:, 1]] = True
                mutable[source[:, 0], source[:, 1]] = True
            assert used == set(range(len(info['pieces'])))
            assert np.array_equal(destination, info['holes'])
            mutable |= destination
            assert np.array_equal(info['grid'][~mutable], canonical_out[~mutable])
            assert np.all(canonical_out[info['board']] != info['bg'])
            if split == 'train':
                if info['geometric_repairs'] > 1 and len(info['traceable_permutations']) > 1:
                    evidence.add('joint')
                for ends in info['clues']:
                    source, receiver = ends[info['phase']], ends[1 - info['phase']]
                    if source[0] == 'blank':
                        evidence.add('reverse')
                    if source[0] == 'solid':
                        evidence.add('thin')
                    if source[0] == 'diamond':
                        evidence.add('preserve')
                    if source[0] != 'blank' and receiver[0] == 'blank':
                        evidence.add('forward')
    assert {'joint', 'reverse', 'thin', 'preserve', 'forward'} <= evidence


def _encode(p):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()}
                          for pair in p[s]] for s in ('train', 'test')}, separators=(',', ':'))


def self_test():
    """Bounded 42-seed variation, joint-mechanism, mutation and API checks."""
    signatures, shapes, normalized, counts, pitches, phases = set(), set(), set(), set(), set(), set()
    roles, depths, permutations_seen, regimes = set(), set(), set(), set()
    pairs = ambiguous = two_cable = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_encode(puzzle))
        orientations = []
        for split in ('train', 'test'):
            local = set()
            for pair in puzzle[split]:
                pairs += 1
                _, info = _resolve(pair['input'])
                local.add(info['transpose'])
                g = pair['input']
                lut = {}
                norm = tuple(lut.setdefault(int(c), len(lut)) for c in g.flat)
                normalized.add((g.shape, norm))
                shapes.add(g.shape)
                counts.add(len(info['pieces']))
                pitches.add(info['centers'][1] - info['centers'][0])
                phases.add(info['phase'])
                roles.add((info['bg'], info['substrate'], info['wire'], info['cross']))
                depths.update(int(np.ptp(p[:, 0])) + 1 for p in info['pieces'])
                permutations_seen.add(info['permutation'])
                ambiguous += info['geometric_repairs'] > 1
                two_cable += len(info['centers']) == 2
            orientations.append(tuple(sorted(local)))
        regimes.add(tuple(orientations))
    assert len(signatures) == 42 and len(normalized) >= .95 * pairs
    assert len(shapes) >= 12 and counts == {2, 3, 4}
    assert pitches >= {4, 5, 6} and phases == {0, 1}
    assert len(roles) >= 40 and len(depths) >= 5
    assert ambiguous >= 84 and two_cable > 0
    assert len(permutations_seen) >= 3
    assert ((False,), (True,)) in regimes or ((True,), (False,)) in regimes
    p = generate_puzzle(0)
    assert _encode(p) == _encode(generate_puzzle(0))
    assert _encode(p) != _encode(generate_puzzle(1))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1729 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('fixed background corruption accepted')
    # Actual concept ablations on input-derived reconstructions. Geometry alone
    # admits a different printed repair whose cable labels are contradictory.
    answer, info = _resolve(p['train'][0]['input'])
    assert len(info['traceable_permutations']) >= 2
    rejected_alternative = False
    for board, _ in _packings(info):
        permutation, _ = _trace(board, info)
        if permutation == info['permutation']:
            continue
        try:
            _complete(board, info, permutation)
        except AssertionError:
            rejected_alternative = True
    assert rejected_alternative, 'B must reject a shape-valid alternative repair'
    # Even granting an oracle for the winning cable permutation, omitting A's
    # physical repair leaves stock in place and visible holes in the board.
    no_assembly, _, _ = _complete(info['grid'], info, info['permutation'])
    no_assembly = no_assembly.T if info['transpose'] else no_assembly
    assert not np.array_equal(no_assembly, answer)
    canonical_no_assembly = no_assembly.T if info['transpose'] else no_assembly
    assert np.all(canonical_no_assembly[info['holes']] == info['bg'])
    return {'seeds': 42, 'pairs': pairs, 'shapes': len(shapes),
            'normalized_structures': len(normalized), 'ambiguous_shape_repairs': ambiguous,
            'two_cable_pairs': two_cable, 'piece_counts': sorted(counts),
            'pitches': sorted(pitches), 'fragment_depths': sorted(depths),
            'routing_permutations': len(permutations_seen)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
