"""Two-hop docking mosaics: relationally typed, exact-inventory tile assembly.

Only the visible input is used by solve. Construction labels are generated from
sampled placements; the validator separately resegments and solves the input.
"""
from collections import Counter
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(cells):
    todo, result = set(cells), []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _key(a):
    return a.shape, a.tobytes()


def _tiles(inp, bg, bbox):
    r, c, h, w = bbox
    exterior = inp != bg
    exterior[r:r+h, c:c+w] = False
    tiles = []
    for cells in _components(map(tuple, np.argwhere(exterior))):
        _require(not any(r <= a+dr < r+h and c <= b+dc < c+w
                         for a, b in cells for dr, dc in D4), "loose tiles need a moat from the board")
        counts = Counter(int(inp[p]) for p in cells)
        _require(len(cells) >= 5 and len(counts) == 2, "a tile needs a body and one mark")
        marks = [v for v, n in counts.items() if n == 1]
        _require(len(marks) == 1, "tile mark must be unique")
        value = marks[0]
        body = next(v for v in counts if v != value)
        anchor = next(p for p in cells if int(inp[p]) == value)
        _require(all((anchor[0]+dr, anchor[1]+dc) in cells for dr, dc in D4),
                 "mark needs a complete cardinal collar")
        shape = tuple(sorted((int(a-anchor[0]), int(b-anchor[1])) for a, b in cells))
        tiles.append(dict(body=body, value=value, shape=shape, cells=cells, anchor=anchor))
    _require(4 <= len(tiles) <= 7, "family uses four to seven actual tiles")
    mapping = {}
    for tile in tiles:
        b, v = tile['body'], tile['value']
        _require(b not in mapping or mapping[b] == v, "conflicting donor values")
        mapping[b] = v
    for tile in tiles:
        _require(tile['value'] in mapping, "missing input donor")
        tile['resolved'] = mapping[tile['value']]
    return tiles


def _sockets(board, bg, frame, wire):
    wires = set(map(tuple, np.argwhere(board == wire)))
    identities = set(map(tuple, np.argwhere((board != bg) & (board != frame) & (board != wire))))
    sockets, pins = [], set()
    for cells in _components(wires):
        _require(len(cells) >= 2, "wire must have at least two cells")
        graph = {p: {(p[0]+dr, p[1]+dc) for dr, dc in D4} & cells for p in cells}
        ends = [p for p in sorted(cells) if len(graph[p]) == 1]
        _require(len(ends) == 2 and all(len(v) in (1, 2) for v in graph.values()),
                 "wire must be an open nonbranching path")
        contacts = [(p, (p[0]+dr, p[1]+dc)) for p in ends for dr, dc in D4
                    if (p[0]+dr, p[1]+dc) in identities]
        _require(len(contacts) == 1, "wire requires one demand pin at one end")
        end, pin = contacts[0]
        _require(pin not in pins, "a pin cannot belong to two sockets")
        _require(sum((pin[0]+dr, pin[1]+dc) in wires for dr, dc in D4) == 1,
                 "pin must touch only its wire end")
        pins.add(pin)
        free = next(p for p in ends if p != end)
        q = next(iter(graph[free]))
        dr, dc = int(free[0]-q[0]), int(free[1]-q[1])
        anchor = (int(free[0]+2*dr), int(free[1]+2*dc))
        collar = (int(free[0]+dr), int(free[1]+dc))
        _require(all(0 <= p[0] < board.shape[0] and 0 <= p[1] < board.shape[1]
                     and board[p] == bg for p in (anchor, collar)), "endpoint must point into holes")
        bends = sum(len(adj) == 2 and tuple(np.add(*sorted(adj))) != (2*p[0], 2*p[1])
                    for p, adj in graph.items())
        sockets.append(dict(anchor=anchor, demand=int(board[pin]), pin=pin,
                            wire=cells, bends=bends, direction=(dr, dc)))
    _require(pins == identities, "every fixed identity cell must be a socket pin")
    _require(len({s['anchor'] for s in sockets}) == len(sockets), "duplicate anchors")
    return sockets


def _interpretations(inp):
    colors, counts = np.unique(inp, return_counts=True)
    _require(int(np.count_nonzero(counts == counts.max())) == 1, "background dominance must be unique")
    bg = int(colors[np.argmax(counts)])
    for frame in map(int, colors):
        if frame == bg:
            continue
        rr, cc = np.where(inp == frame)
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max()-r+1), int(cc.max()-c+1)
        if min(h, w) < 5:
            continue
        board = inp[r:r+h, c:c+w]
        if not (np.all(board[0] == frame) and np.all(board[-1] == frame)
                and np.all(board[:, 0] == frame) and np.all(board[:, -1] == frame)):
            continue
        try:
            tiles = _tiles(inp, bg, (r, c, h, w))
        except ValueError:
            continue
        outside_colors = {t[k] for t in tiles for k in ('body', 'value')}
        for wire in map(int, np.unique(board)):
            if wire in (bg, frame) or wire in outside_colors:
                continue
            try:
                sockets = _sockets(board, bg, frame, wire)
                _require(len(sockets) == len(tiles), "inventory and socket counts differ")
                holes = set(map(tuple, np.argwhere(board == bg)))
                _require(len(holes) == sum(len(t['shape']) for t in tiles), "inventory area differs from holes")
                _require(all(s['demand'] in outside_colors for s in sockets), "pin identity is absent from tiles")
            except ValueError:
                continue
            yield dict(background=bg, frame=frame, wire=wire, board=board.copy(),
                       bbox=(r, c, h, w), tiles=tiles, sockets=sockets, holes=holes)


def _placements(state, mode='composed'):
    """Enumerate anchored translations. The ablation modes are used only in QA."""
    holes, sockets, tiles = state['holes'], state['sockets'], state['tiles']
    anchors = {s['anchor'] for s in sockets}
    hole_index = {p: i for i, p in enumerate(sorted(holes))}
    options = []
    for socket in sockets:
        r, c = socket['anchor']
        choices = []
        for i, tile in enumerate(tiles):
            if mode != 'untyped':
                value = tile['value'] if mode == 'current' else tile['resolved']
                if value != socket['demand']:
                    continue
            cells = {(r+dr, c+dc) for dr, dc in tile['shape']}
            if not cells <= holes or (cells & anchors) != {socket['anchor']}:
                continue
            mask = sum(1 << hole_index[p] for p in cells)
            choices.append((i, mask, cells))
        options.append(choices)
    return options


def _assemblies(state, mode='composed', limit=2):
    """Find distinct final colored grids, ignoring swaps of identical tiles.

    At most seven tiles bounds the search by a small assignment problem. Every
    candidate is a fixed-anchor translation, yet intersecting holes still need
    a global nonoverlap/exact-cover check.
    """
    tiles, sockets = state['tiles'], state['sockets']
    options = _placements(state, mode)
    full = (1 << len(state['holes'])) - 1
    answers, nodes = {}, 0

    def visit(remaining_sockets, used, covered, moves):
        nonlocal nodes
        nodes += 1
        if len(answers) >= limit:
            return
        if not remaining_sockets:
            if covered != full or used != (1 << len(tiles))-1:
                return
            output = state['board'].copy()
            for slot, i, cells in moves:
                tile = tiles[i]
                for cell in cells:
                    output[cell] = tile['body']
                output[sockets[slot]['anchor']] = tile['value'] if mode == 'current' else tile['resolved']
            answers.setdefault(_key(output), (output, tuple(moves)))
            return
        feasible = {}
        for j in remaining_sockets:
            feasible[j] = [x for x in options[j] if not (used & (1 << x[0])) and not (covered & x[1])]
            if not feasible[j]:
                return
        j = min(remaining_sockets, key=lambda k: (len(feasible[k]), k))
        identical = set()
        for i, mask, cells in feasible[j]:
            tile = tiles[i]
            signature = tile['body'], tile['value'], tile['shape']
            if signature in identical:
                continue
            identical.add(signature)
            visit(remaining_sockets-{j}, used | (1 << i), covered | mask, moves+[(j, i, cells)])
            if len(answers) >= limit:
                return

    visit(set(range(len(sockets))), 0, 0, [])
    return list(answers.values()), nodes


def _infer(inp):
    feasible = []
    for state in _interpretations(inp):
        answers, nodes = _assemblies(state)
        if not answers:
            continue
        _require(len(answers) == 1, "different colored assemblies are possible")
        state['output'], state['moves'] = answers[0]
        state['nodes'] = nodes
        feasible.append(state)
    _require(len(feasible) == 1, "input must have one feasible board/socket interpretation")
    return feasible[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover and assemble the mosaic using only its input pixels."""
    return _infer(_grid(input_grid))['output'].copy()


def _rotate_shape(shape, k, flip=False):
    cells = set(shape)
    for _ in range(k):
        cells = {(-c, r) for r, c in cells}
    if flip:
        cells = {(r, -c) for r, c in cells}
    return tuple(sorted(cells))


def _shape(rng):
    kind = int(rng.integers(5))
    cross = {(0, 0), *D4}
    if kind == 0:
        cells = cross
    elif kind == 1:
        cells = {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)}
    elif kind == 2:
        cells = {(r, c) for r in (-1, 0, 1) for c in range(-1, int(rng.integers(2, 4)))}
    elif kind == 3:
        cells = cross | {(r, 0) for r in range(2, int(rng.integers(3, 6)))}
    else:
        cells = cross.copy()
        for _ in range(int(rng.integers(1, 8))):
            # Keep the leftward docking lane open while growing a new silhouette.
            choices = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in D4}
                             - cells)
            choices = [p for p in choices if -2 <= p[0] <= 3 and -1 <= p[1] <= 3]
            cells.add(choices[int(rng.integers(len(choices)))])
    return _rotate_shape(cells, int(rng.integers(4)), bool(rng.integers(2)))


def _relations(rng, mode):
    if mode == 'semantic':
        return [(0, 1), (1, 2), (2, 3), (3, 0)]
    if mode == 'geometric':
        return [(0, 1), (1, 0), (2, 3), (3, 2), (4, 1)]
    if mode == 'combined':
        cycle = int(rng.integers(3, 6))
        rel = [(i, (i+1) % cycle) for i in range(cycle)]
        rel.append((cycle, int(rng.integers(cycle))))
        rel += [rel[int(rng.integers(len(rel)))] for _ in range(int(rng.integers(8-len(rel))))]
        return rel
    n = int(rng.integers(4, 8))
    k = int(rng.integers(3, n+1))
    # The cycle supplies all values; extra keys may point into it or another key.
    cycle = int(rng.integers(2, k+1))
    mapping = {i: (i+1) % cycle for i in range(cycle)}
    for i in range(cycle, k):
        mapping[i] = int(rng.integers(i))
    rel = list(mapping.items())
    rel += [rel[int(rng.integers(k))] for _ in range(n-k)]
    rng.shuffle(rel)
    return rel


def _wire_shape(rng, anchor, direction, bent):
    r, c = anchor
    dr, dc = direction
    length = int(rng.integers(2, 5))
    if not bent:
        wire = {(r+t*dr, c+t*dc) for t in range(2, length+2)}
        pin = (r+(length+2)*dr, c+(length+2)*dc)
    else:
        er, ec = (-dc, dr) if rng.random() < .5 else (dc, -dr)
        wire = {(r+2*dr, c+2*dc), (r+3*dr, c+3*dc)}
        wire |= {(r+3*dr+t*er, c+3*dc+t*ec) for t in range(1, length)}
        pin = (r+3*dr+length*er, c+3*dc+length*ec)
    return wire, pin


def _pack(rng, shapes, h, w, share_pair, bend_probability, force_bend):
    n = len(shapes)
    for _ in range(35):
        placed, occupied, all_wires, all_pins = {}, set(), set(), set()
        order = list(range(n))
        if share_pair is not None:
            a, b = share_pair
            order = [a, b] + [i for i in order if i not in (a, b)]
        success = True
        for stage, i in enumerate(order):
            shape = shapes[i]
            candidates = [(r, c) for r in range(2, h-2) for c in range(2, w-2)]
            rng.shuffle(candidates)
            found = False
            for anchor in candidates:
                holes = {(anchor[0]+dr, anchor[1]+dc) for dr, dc in shape}
                if holes & occupied:
                    continue
                if share_pair is not None and i == share_pair[1]:
                    previous_holes = placed[share_pair[0]]['cells']
                    if not any((r+dr, c+dc) in previous_holes for r, c in holes for dr, dc in D4):
                        continue
                for direction_index in rng.permutation(4):
                    direction = D4[int(direction_index)]
                    bent = force_bend and stage == 0 or bool(rng.random() < bend_probability)
                    wire, pin = _wire_shape(rng, anchor, direction, bent)
                    group = holes | wire | {pin}
                    if len(group) != len(holes)+len(wire)+1 or group & occupied:
                        continue
                    if not all(1 <= r < h-1 and 1 <= c < w-1 for r, c in group):
                        continue
                    if any((r+dr, c+dc) in all_wires | all_pins for r, c in wire for dr, dc in D4):
                        continue
                    if any((pin[0]+dr, pin[1]+dc) in all_wires for dr, dc in D4):
                        continue
                    placed[i] = dict(anchor=anchor, cells=holes, wire=wire, pin=pin)
                    occupied |= group
                    all_wires |= wire
                    all_pins.add(pin)
                    found = True
                    break
                if found:
                    break
            if not found:
                success = False
                break
        if success:
            return [placed[i] for i in range(n)]
    return None


def _scatter(rng, board, shapes, relations, palette):
    bg = int(palette[0])
    h, w = board.shape
    for attempt in range(16):
        ch = int(rng.integers(max(20, h+3), 31)) if attempt < 12 else 30
        cw = int(rng.integers(max(20, w+3), 31)) if attempt < 12 else 30
        r, c = int(rng.integers(ch-h+1)), int(rng.integers(cw-w+1))
        inp = np.full((ch, cw), bg, dtype=np.int8)
        inp[r:r+h, c:c+w] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0, r-1):min(ch, r+h+1), max(0, c-1):min(cw, c+w+1)] = True
        for i in sorted(range(len(shapes)), key=lambda j: (-len(shapes[j]), j)):
            shape = shapes[i]
            choices = [(ar, ac) for ar in range(ch) for ac in range(cw)
                       if all(0 <= ar+dr < ch and 0 <= ac+dc < cw and not blocked[ar+dr, ac+dc]
                              for dr, dc in shape)]
            if not choices:
                break
            ar, ac = choices[int(rng.integers(len(choices)))]
            body, value = relations[i]
            for dr, dc in shape:
                rr, cc = ar+dr, ac+dc
                inp[rr, cc] = int(palette[3+body])
                for er in (-1, 0, 1):
                    for ec in (-1, 0, 1):
                        if 0 <= rr+er < ch and 0 <= cc+ec < cw:
                            blocked[rr+er, cc+ec] = True
            inp[ar, ac] = int(palette[3+value])
        else:
            colors, counts = np.unique(inp, return_counts=True)
            if int(colors[np.argmax(counts)]) == bg and np.count_nonzero(counts == counts.max()) == 1:
                return inp
    return None


def _make_pair(rng, parameters, mode, test):
    for attempt in range(60):
        rel = _relations(rng, mode)
        shapes = [_shape(rng) for _ in rel]
        share = None
        if mode == 'semantic':
            shapes[1] = shapes[0]
        elif mode == 'geometric':
            shapes[0] = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
            shapes[4] = tuple(sorted({(0, 0), *D4}))
            share = (0, 4)
        elif mode == 'combined':
            # Combined proposals list a cycle starting at zero, then its tail.
            cycle = next(i+1 for i, (_, value) in enumerate(rel) if value == 0)
            tail = cycle
            a = (rel[tail][1]-1) % cycle
            twins = [i for i in range(cycle) if i != a][:2]
            shapes[twins[1]] = shapes[twins[0]]
            shapes[a] = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
            shapes[tail] = tuple(sorted({(0, 0), *D4}))
            share = (a, tail)
        palette = rng.permutation(10) if parameters['varied_palette'] else parameters['palette']
        size = parameters['size'] + (1 if test and parameters['larger'] else 0)
        h = int(rng.integers(max(12, size-2), min(19, size+3)))
        w = int(rng.integers(max(13, size-1), min(21, size+5)))
        packed = _pack(rng, shapes, h, w, share, .72 if test else .30, mode == 'semantic')
        if packed is None:
            continue
        bg, frame, wire = map(int, palette[:3])
        board = np.full((h, w), frame, dtype=np.int8)
        output = board.copy()
        mapping = dict(rel)
        for i, group in enumerate(packed):
            b, v = rel[i]
            z = mapping[v]
            for p in group['cells']:
                board[p] = bg
                output[p] = int(palette[3+b])
            output[group['anchor']] = int(palette[3+z])
            for p in group['wire']:
                board[p] = output[p] = wire
            board[group['pin']] = output[group['pin']] = int(palette[3+z])
        inp = _scatter(rng, board, shapes, rel, palette)
        if inp is None:
            continue
        k = int(rng.integers(4))
        inp, output = np.rot90(inp, k).copy(), np.rot90(output, k).copy()
        if rng.random() < .5:
            inp, output = np.fliplr(inp).copy(), np.fliplr(output).copy()
        try:
            state = _infer(inp)
            _require(np.array_equal(state['output'], output), "constructed output disagrees")
            if mode == 'semantic':
                _require(len(_assemblies(state, 'untyped')[0]) == 2, "need geometry-only ambiguity")
            if mode in ('geometric', 'combined'):
                features = _features(state)
                _require(features['shape_needed'] and features['shared_gap'], "need joint geometry witness")
            if mode == 'combined':
                _require(len(_assemblies(state, 'untyped')[0]) == 2, "combined test needs relational ambiguity")
        except ValueError:
            continue
        return dict(input=inp, output=output)
    return _fallback_pair(rng, parameters, mode)


def _fallback_pair(rng, parameters, mode):
    """Constructive, source-independent fallback after bounded random attempts.

    A 15x20 board and a separated 30x30 inventory always fit. The canonical
    relation witnesses are kept; only the random geometry is simplified.
    """
    rel = ([(0, 1), (1, 2), (2, 3), (3, 0), (4, 2)] if mode == 'combined'
           else _relations(rng, mode if mode != 'random' else 'semantic'))
    shapes = [tuple(sorted({(0, 0), *D4})) for _ in rel]
    share = (0, 4) if mode == 'geometric' else ((1, 4) if mode == 'combined' else None)
    if share:
        shapes[share[0]] = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
    order = list(share) + [i for i in range(len(rel)) if i not in share] if share else list(range(len(rel)))
    locations = [(5, 3), (5, 6), (5, 12), (5, 17), (12, 3), (12, 9), (12, 15)]
    palette = rng.permutation(10) if parameters['varied_palette'] else parameters['palette']
    bg, frame, wire = map(int, palette[:3])
    board = np.full((15, 20), frame, dtype=np.int8)
    output = board.copy()
    mapping = dict(rel)
    for i, anchor in zip(order, locations):
        r, c = anchor
        for dr, dc in shapes[i]:
            board[r+dr, c+dc] = bg
            output[r+dr, c+dc] = int(palette[3+rel[i][0]])
        z = int(palette[3+mapping[rel[i][1]]])
        output[anchor] = z
        # The first wire bends left once, while its open end still points down.
        path = {(r-2, c), (r-3, c)}
        pin = (r-3, c-2) if i == order[0] else (r-4, c)
        if i == order[0]:
            path.add((r-3, c-1))
        for p in path:
            board[p] = output[p] = wire
        board[pin] = output[pin] = z
    inp = np.full((30, 30), bg, dtype=np.int8)
    inp[15:30, :20] = board
    exterior = [(2, c) for c in (2, 8, 14, 20, 26)] + [(8, c) for c in (2, 8)]
    for i, (r, c) in enumerate(exterior[:len(rel)]):
        for dr, dc in shapes[i]:
            inp[r+dr, c+dc] = int(palette[3+rel[i][0]])
        inp[r, c] = int(palette[3+rel[i][1]])
    k = int(rng.integers(4))
    inp, output = np.rot90(inp, k).copy(), np.rot90(output, k).copy()
    _require(np.array_equal(solve(inp), output), "constructive fallback must be uniquely solvable")
    return dict(input=inp, output=output)


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    parameters = dict(palette=rng.permutation(10), varied_palette=bool(rng.random() < .65),
                      size=int(rng.integers(14, 17)), larger=bool(rng.random() < .5))
    modes = ['semantic', 'geometric'] + ['random']*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            mode = modes[i] if split == 'train' else ('combined' if i == 0 else 'random')
            for _ in range(12):
                pair = _make_pair(rng, parameters, mode, split == 'test')
                key = _key(pair['input'])
                if parameters['varied_palette'] and split == 'train' and i == 1:
                    previous = _features(_infer(puzzle['train'][0]['input']))['palette']
                    current = _features(_infer(pair['input']))['palette']
                    if previous == current:
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    return puzzle


def _features(state):
    tiles, sockets, moves = state['tiles'], state['sockets'], state['moves']
    by_anchor = {sockets[j]['anchor']: (i, cells) for j, i, cells in moves}
    shape_needed = False
    for anchor, (i, _) in by_anchor.items():
        for k, tile in enumerate(tiles):
            if tile['resolved'] != tiles[i]['resolved'] or tile['shape'] == tiles[i]['shape']:
                continue
            translated = {(anchor[0]+dr, anchor[1]+dc) for dr, dc in tile['shape']}
            if not translated <= state['holes'] or translated & (set(by_anchor)-{anchor}):
                shape_needed = True
    shared_gap = False
    for comp in _components(state['holes']):
        if sum(bool(cells & comp) for _, _, cells in moves) > 1:
            shared_gap = True
    bodies = Counter(t['body'] for t in tiles)
    return dict(board_shape=state['board'].shape, piece_count=len(tiles),
                shape_needed=shape_needed, shared_gap=shared_gap,
                solid=sum(t['resolved'] == t['body'] for t in tiles),
                repeated_body=sum(n-1 for n in bodies.values()),
                shared_donor=any(sum(t['value'] == b for t in tiles) > 1 for b in bodies),
                nonrecursive=any(next(t2['resolved'] for t2 in tiles if t2['body'] == t['value'])
                                 != t['resolved'] for t in tiles),
                bends=sum(s['bends'] for s in sockets),
                shapes=tuple(sorted({t['shape'] for t in tiles})),
                areas=tuple(sorted({len(t['shape']) for t in tiles})),
                directions=tuple(sorted({s['direction'] for s in sockets})),
                palette=(state['background'], state['frame'], state['wire']),
                layout=(state['board'].shape, tuple(sorted(state['holes'])),
                        tuple(sorted(s['anchor'] for s in sockets))))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            state = _infer(inp)
            _require(np.array_equal(out, state['output']), "wrong completed mosaic")
            fixed = state['board'] != state['background']
            _require(np.array_equal(out[fixed], state['board'][fixed]), "fixed cells changed")
            _require(not np.any(out == state['background']), "unfilled board cells")
            _require(sum(len(cells) for _, _, cells in state['moves']) == len(state['holes']), "area not conserved")
            # Audit physical use, each demand and orientation, without recoloring donors.
            _require({i for _, i, _ in state['moves']} == set(range(len(state['tiles']))), "inventory not used once")
            for j, i, cells in state['moves']:
                tile, socket = state['tiles'][i], state['sockets'][j]
                _require(out[socket['anchor']] == socket['demand'] == tile['resolved'], "wrong composed mark")
                _require(all(out[p] == tile['body'] for p in cells-{socket['anchor']}), "body recolored")
            f = _features(state)
            if split == 'train':
                f['untyped_ambiguous'] = len(_assemblies(state, 'untyped')[0]) == 2
                current_answers, _ = _assemblies(state, 'current')
                f['current_fails'] = not any(np.array_equal(answer, out) for answer, _ in current_answers)
                train.append(f)
            else:
                if not test:
                    f['untyped_ambiguous'] = len(_assemblies(state, 'untyped')[0]) == 2
                test.append(f)
    _require(any(f['untyped_ambiguous'] and f['current_fails'] and f['nonrecursive'] for f in train),
             "training must expose the relational contribution")
    _require(any(f['shape_needed'] and f['shared_gap'] and f['shared_donor'] for f in train),
             "training must expose the shape/inventory contribution")
    _require(any(f['solid'] for f in train), "solid-result convention must be demonstrated")
    _require(any(f['bends'] for f in train), "bent endpoint continuation must be demonstrated")
    _require(test[0]['shape_needed'] and test[0]['shared_gap'] and test[0]['untyped_ambiguous'],
             "first test must recombine semantic and geometric witnesses")
    train_palettes = {f['palette'] for f in train}
    if len(train_palettes) == 1:
        _require({f['palette'] for f in test} <= train_palettes, "unexplained structural palette shift")


def self_test():
    """Bounded semantic checks and structural measurements across 42 seeds."""
    signatures, layouts, board_shapes, canvas_shapes, shape_types = set(), set(), set(), set(), set()
    palettes, counts, areas, directions = set(), set(), set(), set()
    metrics = Counter()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        signature = tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))
        signatures.add(signature)
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                state = _infer(pair['input'])
                f = _features(state)
                layouts.add(f['layout'])
                board_shapes.add(f['board_shape'])
                canvas_shapes.add(pair['input'].shape)
                shape_types.update(f['shapes'])
                palettes.add(f['palette'])
                counts.add(f['piece_count'])
                areas.update(f['areas'])
                directions.update(f['directions'])
                for flag in ('shape_needed', 'shared_gap', 'solid', 'repeated_body', 'shared_donor', 'nonrecursive', 'bends'):
                    metrics[flag] += bool(f[flag])
    _require(len(signatures) == 42, "seeds did not vary whole puzzles")
    repeated = generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                 for p, q in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output')), "nondeterministic seed")
    _require(len(layouts) >= 180 and len(board_shapes) >= 25 and len(canvas_shapes) >= 45,
             "insufficient geometric variation")
    _require(len(shape_types) >= 45 and len(areas) >= 7 and counts == {4, 5, 6, 7}, "insufficient tile variation")
    _require(len(palettes) >= 30 and directions == set(D4), "insufficient palette/docking variation")
    _require(all(metrics[k] >= 20 for k in metrics), "missing family-specific variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1800+nt*10+ne, nt, ne)
            validate_generated_puzzle(puzzle)
            _require(len(puzzle['train']) == nt and len(puzzle['test']) == ne, "count ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    fallback_rng = np.random.default_rng(815)
    fallback_parameters = dict(varied_palette=True, palette=np.arange(10))
    for mode in ('semantic', 'geometric', 'combined', 'random'):
        pair = _fallback_pair(fallback_rng, fallback_parameters, mode)
        state = _infer(pair['input'])
        _require(np.array_equal(state['output'], pair['output']), "fallback mismatch")
        if mode in ('geometric', 'combined'):
            _require(_features(state)['shape_needed'] and _features(state)['shared_gap'], "fallback witness missing")
    # Change one of two otherwise consistent same-body donors. Its geometry and
    # minority mark stay valid; only the functional dictionary is contradicted.
    conflicting = None
    for puzzle in puzzles:
        for pair in puzzle['train']:
            state = _infer(pair['input'])
            bodies = Counter(t['body'] for t in state['tiles'])
            if max(bodies.values()) > 1:
                tile = next(t for t in state['tiles'] if bodies[t['body']] > 1)
                replacement = next(t['body'] for t in state['tiles'] if t['body'] not in (tile['body'], tile['value']))
                conflicting = pair['input'].copy()
                conflicting[tile['anchor']] = replacement
                break
        if conflicting is not None:
            break
    _require(conflicting is not None, "no duplicate donor witness")
    try:
        solve(conflicting)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting donors accepted")
    first = puzzles[0]
    for kind in ('fixed', 'inventory', 'donor'):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in first[s]] for s in ('train', 'test')}
        pair = bad['train'][0]
        state = _infer(pair['input'])
        if kind == 'fixed':
            pair['output'][0, 0] = state['background']
        elif kind == 'inventory':
            p = next(iter(state['tiles'][0]['cells']))
            pair['input'][p] = state['background']
        else:
            tile = state['tiles'][0]
            pair['input'][tile['anchor']] = tile['body']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted mosaic accepted: " + kind)
    return dict(seeds=42, pairs=210, layouts=len(layouts), board_shapes=len(board_shapes),
                canvas_shapes=len(canvas_shapes), tile_shapes=len(shape_types), tile_areas=sorted(areas),
                piece_counts=sorted(counts), palettes=len(palettes), witnesses=dict(metrics))


if __name__ == '__main__':
    print(self_test())
