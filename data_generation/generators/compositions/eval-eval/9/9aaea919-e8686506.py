"""Keyed counting cartridges: exact assembly determines arithmetic recipients.

Only the visible grid is needed by solve.  Generation constructs labels from
placed cartridges; the parser and exact-cover search recover their placements.
"""
from itertools import combinations
import copy
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(sorted(cells))
    return result


def _piece_data(values, control):
    """A cartridge has one three-cell control and repeated 1x1/1x2 tokens."""
    cells = {(r, c): int(v) for r, c, v in values}
    colors, counts = np.unique(list(cells.values()), return_counts=True)
    carrier = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > len(cells) / 2 and carrier != control,
             "cartridge carrier must be its majority color")
    base = sorted(p for p, v in cells.items() if v == control)
    _require(len(base) == 3 and len({r for r, c in base}) == 1 and
             [c for r, c in base] == list(range(base[0][1], base[0][1]+3)),
             "cartridge needs a three-cell control")
    br, bc = base[0]
    tokens = sorted(p for p, v in cells.items() if v not in (carrier, control))
    _require(bool(tokens), "empty source")
    rows = sorted({r for r, c in tokens}, reverse=True)
    _require(rows == [br-2*k for k in range(1, len(rows)+1)], "source pitch must be two")
    shapes = []
    for r in rows:
        cols = sorted(c for rr, c in tokens if rr == r)
        _require(1 <= len(cols) <= 2 and cols == list(range(cols[0], cols[-1]+1))
                 and bc <= cols[0] <= cols[-1] <= bc+2, "invalid source token")
        shapes.append(tuple(cols))
    _require(len(set(shapes)) == 1 and len({cells[p] for p in tokens}) == 1,
             "source tokens must repeat exactly")
    # The three-wide core makes the carrier a solid object even between tokens.
    top = min(r for r, c in cells)
    _require(all((r, c) in cells for r in range(top, br+1) for c in range(bc, bc+3)),
             "broken cartridge core")
    _require(all(r <= br and bc-1 <= c <= bc+3 for r, c in cells), "invalid cartridge outline")
    _require(not (any(c < bc for r, c in cells) and any(c > bc+2 for r, c in cells)),
             "keys must be on one side")
    return dict(carrier=carrier, base=base, tokens=tokens, count=len(rows),
                center=bc+1, color=cells[tokens[0]])


def _parse(a):
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size / 2, "background must occupy more than half the canvas")
    frames = []
    for color in colors:
        if color == bg:
            continue
        rr, cc = np.where(a == color)
        t, l, b, r = int(rr.min()), int(cc.min()), int(rr.max()+1), int(cc.max()+1)
        if b-t < 14 or r-l < 12:
            continue
        q = a[t:b, l:r]
        if all(bool(np.all(edge == color)) for edge in (q[0], q[-1], q[:, 0], q[:, -1])):
            frames.append((int(color), (t, l, b, r), q.copy()))
    _require(len(frames) == 1, "need one rectangular frame")
    frame, bbox, board = frames[0]
    full = [r for r in range(1, board.shape[0]-1) if np.all(board[r] == frame)]
    _require(bool(full), "missing header divider")
    divider = full[0]
    _require(divider >= 8, "header too short")
    row = board[divider-1]
    bays = _runs(row != frame)
    _require(2 <= len(bays) <= 3 and all(hi-lo >= 5 for lo, hi in bays), "invalid receiver bays")
    control = int(row[bays[0][0]])
    _require(control not in (bg, frame) and all(np.all(row[lo:hi] == control) for lo, hi in bays),
             "controls must share one color")
    expected_header = np.full((divider, board.shape[1]), frame, dtype=np.int8)
    destinations = []
    for lo, hi in bays:
        expected_header[1:, lo:hi] = bg
        expected_header[-1, lo:hi] = control
        block = board[1:divider-2, lo:hi]
        points = [(int(r+1), int(c+lo)) for r, c in np.argwhere(block != bg)]
        _require(bool(points), "empty destination")
        rows = sorted({r for r, c in points}, reverse=True)
        _require(rows == [divider-3-2*k for k in range(len(rows))], "invalid destination pitch")
        offsets = [tuple(sorted(c for rr, c in points if rr == r)) for r in rows]
        cols = offsets[0]
        _require(len(set(offsets)) == 1 and 1 <= len(cols) <= 3 and
                 cols == tuple(range(cols[0], cols[-1]+1)), "invalid destination token")
        payload = {int(board[p]) for p in points}
        _require(len(payload) == 1 and not payload & {bg, frame, control}, "invalid destination color")
        for p in points:
            expected_header[p] = board[p]
        destinations.append(dict(lo=lo, hi=hi, cols=cols, count=len(rows),
                                 color=payload.pop(), bottom=divider-3))
    _require(np.array_equal(board[:divider], expected_header), "unexpected header marks")
    _require(len({d['color'] for d in destinations}) == len(destinations), "receiver colors must differ")
    bed = board[divider+1:-1, 1:-1]
    _require(bed.size > 0 and set(map(int, np.unique(bed))) <= {bg, frame}, "invalid bed")
    exterior = a != bg
    t, l, b, r = bbox
    exterior[t:b, l:r] = False
    pieces = []
    for comp in _components(exterior):
        top, left = min(y for y, x in comp), min(x for y, x in comp)
        values = tuple((y-top, x-left, int(a[y, x])) for y, x in comp)
        _require(all(v not in (bg, frame) for y, x, v in values), "frame color outside frame")
        data = _piece_data(values, control)
        pieces.append(dict(values=values, data=data, origin=(top, left),
                           height=max(y for y, x, v in values)+1,
                           width=max(x for y, x, v in values)+1))
    _require(bool(pieces), "missing inventory")
    carriers = {p['data']['carrier'] for p in pieces}
    _require(len(carriers) == 1 and not carriers & {bg, frame, control}, "inconsistent carrier")
    _require(not carriers & {d['color'] for d in destinations}, "carrier overlaps receiver color")
    _require(all(p['data']['color'] in {d['color'] for d in destinations} for p in pieces),
             "source payload outside receiver palette")
    return dict(bg=bg, frame=frame, bbox=bbox, board=board, divider=divider,
                control=control, destinations=destinations, bed=bed.copy(), pieces=pieces)


def _assemble(bed, bg, pieces):
    """Prove a unique colored exact cover, using every oriented piece once."""
    holes = sorted(map(tuple, np.argwhere(bed == bg).tolist()))
    _require(sum(len(p['values']) for p in pieces) == len(holes), "inventory area mismatch")
    indices = {p: i for i, p in enumerate(holes)}
    by_cell = [[] for p in holes]
    for k, piece in enumerate(pieces):
        for r in range(bed.shape[0]-piece['height']+1):
            for c in range(bed.shape[1]-piece['width']+1):
                coords = [(r+y, c+x) for y, x, v in piece['values']]
                if not all(p in indices for p in coords):
                    continue
                mask = sum(1 << indices[p] for p in coords)
                move = (k, r, c, mask)
                for p in coords:
                    by_cell[indices[p]].append(move)
    answers, nodes = {}, 0
    dead = set()

    def visit(remaining, used, path):
        nonlocal nodes
        nodes += 1
        _require(nodes <= 20000, "exact cover search bound exceeded")
        if len(answers) > 1:
            return True
        if not remaining:
            if used != (1 << len(pieces))-1:
                return False
            result = bed.copy()
            for k, r, c, mask in path:
                for y, x, v in pieces[k]['values']:
                    result[r+y, c+x] = v
            answers.setdefault(result.tobytes(), (result, tuple(path)))
            return True
        state = remaining, used
        if state in dead:
            return False
        best, bits = None, remaining
        while bits:
            bit = bits & -bits
            choices = [m for m in by_cell[bit.bit_length()-1]
                       if not used & (1 << m[0]) and m[3] & remaining == m[3]]
            if not choices:
                dead.add(state)
                return False
            if best is None or len(choices) < len(best):
                best = choices
            if len(best) == 1:
                break
            bits ^= bit
        feasible = False
        for move in best:
            feasible |= visit(remaining ^ move[3], used | (1 << move[0]), path+[move])
            if len(answers) > 1:
                break
        if not feasible:
            dead.add(state)
        return feasible

    visit((1 << len(holes))-1, 0, [])
    _require(len(answers) == 1, "inventory must have exactly one colored assembly")
    result, moves = next(iter(answers.values()))
    return result, moves, nodes


def _analyze(input_grid):
    a = _grid(input_grid)
    p = _parse(a)
    assembled, moves, nodes = _assemble(p['bed'], p['bg'], p['pieces'])
    output = p['board'].copy()
    d = p['divider']
    output[d+1:-1, 1:-1] = assembled
    totals = [0] * len(p['destinations'])
    banks = [[] for _ in totals]
    placements = []
    for k, r, c, mask in moves:
        data = p['pieces'][k]['data']
        center = c+data['center']+1
        owners = [j for j, dest in enumerate(p['destinations']) if dest['lo'] <= center < dest['hi']]
        _require(len(owners) == 1, "source axis must be beneath a control bar")
        j = owners[0]
        totals[j] += data['count']
        banks[j].append(data['count'])
        placements.append((k, r, c, j))
        for y, x in data['tokens']:
            output[d+1+r+y, 1+c+x] = p['control']
        for y, x in data['base']:
            output[d+1+r+y, 1+c+x] = data['carrier']
    for dest, total in zip(p['destinations'], totals):
        output[d-1, dest['lo']:dest['hi']] = p['bg']
        for k in range(dest['count'], dest['count']+total):
            row = dest['bottom']-2*k
            _require(row >= 1 and np.all(output[row, list(dest['cols'])] == p['bg']),
                     "destination extension clips or collides")
            output[row, list(dest['cols'])] = dest['color']
    p.update(output=output, totals=totals, banks=banks, placements=placements, nodes=nodes)
    return p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate the exact inventory, route by fitted axes, and transfer counts."""
    return _analyze(input_grid)['output'].copy()


def _evidence(analyses):
    operands = [(d['count'], b) for p in analyses for d, b in zip(p['destinations'], p['banks'])]
    _require(any(len(b) > 1 and len(set(b)) > 1 for d, b in operands), "need unequal aggregation")
    _require(len({sum(b) for d, b in operands}) >= 2 and len({d for d, b in operands}) >= 2,
             "need independent operands")
    d0, b0 = operands[0]
    _require(any(sum(b)*d0 != sum(b0)*d for d, b in operands), "constant multiplier ambiguity")
    rivals = [lambda d, b: 2*d, lambda d, b: sum(b), lambda d, b: d*sum(b),
              lambda d, b: d+len(b), lambda d, b: d+max(b, default=0)]
    for rival in rivals:
        _require(any(rival(d, b) != d+sum(b) for d, b in operands), "arithmetic shortcut fits training")
    color_witness = order_witness = False
    for p in analyses:
        by_color = [0] * len(p['totals'])
        for piece in p['pieces']:
            owner = next(j for j, dest in enumerate(p['destinations']) if dest['color'] == piece['data']['color'])
            by_color[owner] += piece['data']['count']
        color_witness |= by_color != p['totals']
        exterior_order = sorted(range(len(p['pieces'])), key=lambda k: p['pieces'][k]['origin'][::-1])
        slots = sorted(p['placements'], key=lambda m: m[2]+p['pieces'][m[0]]['data']['center'])
        by_order = [0] * len(p['totals'])
        for k, slot in zip(exterior_order, slots):
            by_order[slot[3]] += p['pieces'][k]['data']['count']
        order_witness |= by_order != p['totals']
    _require(color_witness, "color routing fits training")
    _require(order_witness, "exterior-order routing fits training")
    _require(any(len(_components(p['bed'] == p['bg'])) < len(p['pieces']) for p in analyses),
             "training must show pieces sharing a gap")
    _require({len(q['data']['tokens'])//q['data']['count'] for p in analyses for q in p['pieces']} == {1, 2},
             "training must distinguish token count from source area")
    _require(any(sum(len(p['pieces'][k]['data']['tokens']) for k, r, c, owner in p['placements'] if owner == j)
                 != total*len(dest['cols'])
                 for p in analyses for j, (dest, total) in enumerate(zip(p['destinations'], p['totals']))),
             "area-conserving transfer fits training")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            p = _analyze(a)
            _require(np.array_equal(b, p['output']), "incorrect output")
            _require(all(p['banks']) and any(len(bank) > 1 for bank in p['banks']), "missing routed aggregation")
            d = p['divider']
            _require(not np.any(b[d+1:-1, 1:-1] == p['bg']), "bed not completely filled")
            fixed = p['board'] == p['frame']
            _require(np.array_equal(b[fixed], p['board'][fixed]), "fixed scaffold changed")
            _require(np.count_nonzero(b[d+1:-1] == p['control']) ==
                     sum(len(q['data']['tokens']) for q in p['pieces']), "source highlighting mismatch")
            for dest, total in zip(p['destinations'], p['totals']):
                region = b[1:d-2, dest['lo']:dest['hi']]
                _require(np.count_nonzero(region == dest['color']) ==
                         (dest['count']+total)*len(dest['cols']), "wrong transferred token count")
            if split == 'train':
                training.append(p)
    _evidence(training)


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _geometry(rng, lanes, n, triple):
    widths = [int(rng.integers(6, 9)) for _ in range(2)] if lanes == 2 else [5, 5, 5]
    if triple:
        widths = [8, 8]
    intervals, x = [], 0
    for w in widths:
        intervals.append((x, x+w))
        x += w+1
    bw = x-1
    choices = []
    for xs in combinations(range(bw-2), n):
        if any(b-a < 3 for a, b in zip(xs, xs[1:])):
            continue
        owners = [next((j for j, (lo, hi) in enumerate(intervals) if lo <= x+1 < hi), -1) for x in xs]
        if -1 in owners or set(owners) != set(range(lanes)):
            continue
        counts = [owners.count(j) for j in range(lanes)]
        if (triple and max(counts) != 3) or (not triple and max(counts) != 2):
            continue
        choices.append((xs, owners))
    if not choices:
        return None
    xs, owners = choices[int(rng.integers(len(choices)))]
    bh = int(rng.integers(7, 11))
    shapes, cores, occupied = [], [], set()
    for x in xs:
        h = int(rng.integers(7, min(9, bh)+1))
        y = int(rng.integers(bh-h+1))
        core = {(r, c) for r in range(y, y+h) for c in range(x, x+3)}
        cores.append((y, x, h))
        shapes.append(core.copy())
        occupied |= core
    for k, (y, x, h) in enumerate(cores):
        sides = [-1, 3]
        rng.shuffle(sides)
        for side in sides:
            available = [(r, x+side) for r in range(y, y+h)
                         if 0 <= x+side < bw and (r, x+side) not in occupied]
            if available:
                keys = {p for p in available if rng.random() < 0.45}
                if not keys:
                    keys.add(available[int(rng.integers(len(available)))])
                shapes[k] |= keys
                occupied |= keys
                break
    return widths, intervals, bw, bh, shapes, cores, owners


def _scatter(rng, board, pieces, bg):
    """Two loose inventory columns with randomized order, gaps and offsets."""
    order = rng.permutation(len(pieces)).tolist()
    columns = [order[::2], order[1::2]]
    min_heights = [sum(pieces[k]['height'] for k in col)+len(col)-1 for col in columns]
    ch_min = max(board.shape[0], max(min_heights))
    ch = int(rng.integers(ch_min, 31))
    cw = int(rng.integers(board.shape[1]+10, 31))
    inventory_left = bool(rng.integers(2))
    board_left = cw-board.shape[1] if inventory_left else 0
    bank_left = 0 if inventory_left else board.shape[1]+1
    bank_width = cw-board.shape[1]-1
    board_top = int(rng.integers(ch-board.shape[0]+1))
    canvas = np.full((ch, cw), bg, dtype=np.int8)
    canvas[board_top:board_top+board.shape[0], board_left:board_left+board.shape[1]] = board
    for column, ids in enumerate(columns):
        extra = rng.multinomial(ch-min_heights[column], np.full(len(ids)+1, 1/(len(ids)+1)))
        y = int(extra[0])
        for j, k in enumerate(ids):
            p = pieces[k]
            left = bank_left+column*5
            room = (4 if column == 0 else bank_width-5)-p['width']
            x = left+int(rng.integers(room+1))
            for r, c, v in p['values']:
                canvas[y+r, x+c] = v
            y += p['height']+1+int(extra[j+1])
    return canvas


def _build_pair(rng, lanes, n, triple, palette):
    for attempt in range(160):
        geo = _geometry(rng, lanes, n, triple)
        if geo is None:
            continue
        widths, intervals, bw, bh, shapes, cores, owners = geo
        counts = rng.integers(1, 4, size=n).tolist()
        totals = [sum(counts[k] for k in range(n) if owners[k] == j) for j in range(lanes)]
        if max(totals) > 6 or not any(len({counts[k] for k in range(n) if owners[k] == j}) > 1 for j in range(lanes)):
            continue
        dest_counts = [int(rng.integers(1, min(3, 7-total)+1)) for total in totals]
        final = max(d+t for d, t in zip(dest_counts, totals))
        divider = 2*final+2+int(rng.integers(min(2, 26-2*final-bh)+1))
        bg, frame, carrier, control = map(int, palette[:4])
        colors = list(map(int, palette[4:4+lanes]))
        board = np.full((divider+bh+2, bw+2), frame, dtype=np.int8)
        answer = board.copy()
        for j, ((lo, hi), count) in enumerate(zip(intervals, dest_counts)):
            lo, hi = lo+1, hi+1
            board[1:divider, lo:hi] = bg
            board[divider-1, lo:hi] = control
            token_width = int(rng.integers(1, 4))
            start = (lo+hi-token_width)//2
            for k in range(count):
                board[divider-3-2*k, start:start+token_width] = colors[j]
            answer[:divider, lo:hi] = board[:divider, lo:hi]
            answer[divider-1, lo:hi] = bg
            for k in range(count, count+totals[j]):
                answer[divider-3-2*k, start:start+token_width] = colors[j]
        pieces = []
        for k, (shape, (y, x, h)) in enumerate(zip(shapes, cores)):
            values = {p: carrier for p in shape}
            br = y+h-1
            for c in range(x, x+3):
                values[br, c] = control
            token_width = int(rng.integers(1, 3))
            start = x+1 if token_width == 1 else x+int(rng.integers(2))
            payload = colors[int(rng.integers(lanes))]
            if k == 0:
                payload = colors[(owners[k]+1) % lanes]
            for t in range(1, counts[k]+1):
                for c in range(start, start+token_width):
                    values[br-2*t, c] = payload
            for (r, c), v in values.items():
                board[divider+1+r, 1+c] = bg
                answer[divider+1+r, 1+c] = carrier if v == control else (control if v == payload else v)
            top, left = min(r for r, c in shape), min(c for r, c in shape)
            vals = tuple((r-top, c-left, values[r, c]) for r, c in sorted(shape))
            pieces.append(dict(values=vals, height=max(r for r, c, v in vals)+1,
                               width=max(c for r, c, v in vals)+1))
        bed = board[divider+1:-1, 1:-1]
        try:
            _assemble(bed, bg, pieces)
        except ValueError:
            continue
        canvas = _scatter(rng, board, pieces, bg)
        try:
            p = _analyze(canvas)
        except ValueError:
            continue
        if np.array_equal(p['output'], answer):
            return {'input': canvas, 'output': answer}
    raise RuntimeError("bounded cartridge construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    exchange = bool(rng.integers(2))
    shift = int(rng.integers(2))

    def pair(split, index):
        if split == 'test' and index == 0:
            lanes, n, triple = (3, int(rng.integers(4, 6)), False) if shift == 0 else (2, 5, True)
        else:
            lanes = 2 if shift == 0 or index == 0 else int(rng.choice([2, 3]))
            n = int(rng.integers(3, 5)) if lanes == 2 else int(rng.integers(4, 6))
            triple = False
        return _build_pair(rng, lanes, n, triple, rng.permutation(10) if exchange else palette)

    # Jointly choose informative anchors even for the minimum two demonstrations.
    for attempt in range(80):
        anchors = [pair('train', 0), pair('train', 1)]
        if _key(anchors[0]['input']) == _key(anchors[1]['input']):
            continue
        try:
            _evidence([_analyze(p['input']) for p in anchors])
            break
        except ValueError:
            pass
    else:
        raise RuntimeError("could not construct informative demonstrations")
    puzzle = {'train': anchors, 'test': []}
    seen = {_key(p['input']) for p in anchors}
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(2 if split == 'train' else 0, count):
            for attempt in range(30):
                p = pair(split, index)
                if _key(p['input']) not in seen:
                    seen.add(_key(p['input']))
                    puzzle[split].append(p)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    shapes, silhouettes, layouts, palettes, signatures = set(), set(), set(), set(), set()
    lane_counts, piece_counts, totals, destination_widths = set(), set(), set(), set()
    shared_gaps = triple_tests = new_receiver_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        train = [_analyze(p['input']) for p in puzzle['train']]
        test = [_analyze(p['input']) for p in puzzle['test']]
        triple_tests += max(map(len, test[0]['banks'])) > max(len(b) for p in train for b in p['banks'])
        new_receiver_tests += len(test[0]['banks']) > max(len(p['banks']) for p in train)
        for p in train+test:
            shapes.add(p['board'].shape)
            palettes.add((p['bg'], p['frame'], p['control'], p['pieces'][0]['data']['carrier']))
            lane_counts.add(len(p['banks']))
            piece_counts.add(len(p['pieces']))
            totals.update(p['totals'])
            destination_widths.update(len(d['cols']) for d in p['destinations'])
            layouts.add((p['bed'].shape, (p['bed'] == p['bg']).tobytes()))
            shared_gaps += len(_components(p['bed'] == p['bg'])) < len(p['pieces'])
            for piece in p['pieces']:
                silhouettes.add(tuple((r, c) for r, c, v in piece['values']))
    assert len(signatures) == 42 and len(palettes) >= 25
    assert len(shapes) >= 25 and len(layouts) >= 150 and len(silhouettes) >= 100
    assert lane_counts == {2, 3} and piece_counts == {3, 4, 5}
    assert {1, 2, 3, 4, 5, 6} <= totals and destination_widths == {1, 2, 3}
    assert shared_gaps >= 100 and triple_tests >= 8 and new_receiver_tests >= 8
    # Exact-cover uniqueness concerns colored results, not piece identities.
    singleton = lambda color: dict(values=((0, 0, color),), height=1, width=1)
    slots = np.array([[0, 7, 0]], dtype=np.int8)
    assert np.array_equal(_assemble(slots, 0, [singleton(1), singleton(1)])[0], [[1, 7, 1]])
    try:
        _assemble(slots, 0, [singleton(1), singleton(2)])
    except ValueError:
        pass
    else:
        raise AssertionError('different colored assemblies were accepted')
    try:
        _assemble(np.array([[0, 0, 0], [7, 0, 7]], dtype=np.int8), 0,
                  [dict(values=((0, 1, 1), (1, 0, 1), (1, 1, 1), (1, 2, 1)), height=2, width=3)])
    except ValueError:
        pass
    else:
        raise AssertionError('piece was illegally reflected')
    a = generate_puzzle(123, 2, 1)
    b = generate_puzzle(123, 2, 1)
    assert all(np.array_equal(p[r], q[r]) for s in a for p, q in zip(a[s], b[s]) for r in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Mutations exercise the arithmetic, processing, preservation and inventory.
    original = generate_puzzle(71, 2, 1)
    p = _analyze(original['train'][0]['input'])
    # Counterfactual inputs keep the transport geometry unchanged.  Color has
    # no addressing effect; deleting one source token affects its fitted owner.
    k, r, c, owner = next(m for m in p['placements'] if p['pieces'][m[0]]['data']['count'] > 1)
    piece = p['pieces'][k]
    y0, x0 = piece['origin']
    recolored = original['train'][0]['input'].copy()
    new_color = next(d['color'] for d in p['destinations'] if d['color'] != piece['data']['color'])
    for y, x in piece['data']['tokens']:
        recolored[y0+y, x0+x] = new_color
    assert np.array_equal(solve(recolored), p['output'])
    reduced = original['train'][0]['input'].copy()
    expected = p['output'].copy()
    top = min(y for y, x in piece['data']['tokens'])
    for y, x in piece['data']['tokens']:
        if y == top:
            reduced[y0+y, x0+x] = piece['data']['carrier']
            expected[p['divider']+1+r+y, 1+c+x] = piece['data']['carrier']
    dest = p['destinations'][owner]
    expected[dest['bottom']-2*(dest['count']+p['totals'][owner]-1), list(dest['cols'])] = p['bg']
    assert np.array_equal(solve(reduced), expected)
    for defect in ('growth', 'highlight', 'control', 'fixed', 'inventory'):
        bad = copy.deepcopy(original)
        pair = bad['train'][0]
        if defect == 'inventory':
            source = p['pieces'][0]
            r, c, value = source['values'][0]
            y, x = source['origin']
            pair['input'][y+r, x+c] = p['bg']
        elif defect == 'fixed':
            pair['output'][0, 0] = p['bg']
        elif defect == 'growth':
            dest, total = p['destinations'][0], p['totals'][0]
            pair['output'][dest['bottom']-2*(dest['count']+total-1), list(dest['cols'])] = p['bg']
        elif defect == 'control':
            dest = p['destinations'][0]
            pair['output'][p['divider']-1, dest['lo']:dest['hi']] = p['control']
        else:
            k, r, c, j = p['placements'][0]
            y, x = p['pieces'][k]['data']['tokens'][0]
            pair['output'][p['divider']+1+r+y, 1+c+x] = p['pieces'][k]['data']['color']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("mutation accepted: "+defect)


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, diversity and mutation checks')
