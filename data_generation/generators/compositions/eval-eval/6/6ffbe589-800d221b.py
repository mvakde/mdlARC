"""Shutter-controlled voting circuits. Standard library and NumPy only.

The solver discovers all geometry from the input. Generation builds labels from
construction records; validation uses the input parser and coordinate rotations.
"""
from __future__ import annotations

import heapq
import itertools
import json

import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_grid(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _neighbors(cells, h, w, diagonal=False):
    offsets = tuple(itertools.product((-1, 0, 1), repeat=2)) if diagonal else _DIRS
    return {(r + dr, c + dc) for r, c in cells for dr, dc in offsets
            if 0 <= r + dr < h and 0 <= c + dc < w}


def _rect(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, found = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _turn(p, n, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _socket_map(r, c, n):
    """Outside cable cell -> aligned cell of the n-by-n shutter square."""
    result = {}
    for t in range(1, n - 1):
        result[r - 1, c + 1 + t] = (0, t)
        result[r + 1 + t, c + n + 2] = (t, n - 1)
        result[r + n + 2, c + 1 + t] = (n - 1, t)
        result[r + 1 + t, c - 1] = (t, 0)
    return result


def _majority(values):
    values = list(map(int, values))
    counts = {v: values.count(v) for v in set(values)}
    if not counts:
        return None
    best = max(counts.values())
    winners = [v for v, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else None


def _find_board(a, bg):
    h, w = a.shape
    candidates = []
    for color in map(int, np.unique(a)):
        if color == bg:
            continue
        # A frame begins with a long horizontal run. Testing only such starts
        # avoids enumerating every square on the entire canvas.
        for r, c in np.argwhere(a == color):
            r, c = int(r), int(c)
            if c + 8 >= w or r + 8 >= h or not np.all(a[r, c:c + 9] == color):
                continue
            for d in range(9, min(h - r, w - c) + 1, 2):
                if not (np.all(a[r, c:c+d] == color)
                        and np.all(a[r+d-1, c:c+d] == color)
                        and np.all(a[r:r+d, c] == color)
                        and np.all(a[r:r+d, c+d-1] == color)):
                    continue
                inner = a[r+1:r+d-1, c+1:c+d-1]
                n = d - 2
                mid = n // 2
                sc = inner == color
                expected = np.zeros((n, n), dtype=bool)
                expected[mid-1:mid+2, mid-1:mid+2] = True
                if np.array_equal(sc, expected):
                    candidates.append((r, c, n, color))
    assert len(candidates) == 1, 'exactly one framed board with a central collector is required'
    return candidates[0]


def _read_input(input_grid):
    a = _check_grid(input_grid)
    h, w = a.shape
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    r, c, n, sc = _find_board(a, bg)
    assert r >= 2 and c >= 2 and r + n + 3 < h and c + n + 3 < w
    inner = a[r+1:r+n+1, c+1:c+n+1].copy()
    gates = tuple(v for v in map(int, np.unique(inner)) if v not in (bg, sc))
    assert 2 <= len(gates) <= 4
    board = _rect(r, c, n+2, n+2)
    external = {(y, x) for y in range(h) for x in range(w) if (y, x) not in board}
    counters = {}
    for color in gates:
        cells = {p for p in external if a[p] == color}
        if not cells:
            continue
        rows, cols = zip(*cells)
        assert ((len(set(rows)) == 1 and max(cols)-min(cols)+1 == len(cells))
                or (len(set(cols)) == 1 and max(rows)-min(rows)+1 == len(cells)))
        assert 1 <= len(cells) <= 8
        assert all(a[q] == bg or q in cells for q in _neighbors(cells, h, w, True))
        counters[color] = cells
    assert counters
    signals = tuple(v for v in map(int, np.unique(a)) if v not in (bg, sc, *gates))
    assert len(signals) == 2
    panel_cells = {p for p in external if a[p] in signals}
    panels = _components(panel_cells)
    assert 2 <= len(panels) <= 5
    labels, boxes = [], []
    for cells in panels:
        rr, cc = zip(*cells)
        box = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
        assert len(cells) == box[2]*box[3] and len(cells) >= 3
        values = [int(a[p]) for p in cells]
        assert set(values) == set(signals)
        label = _majority(values)
        assert label is not None
        assert cells.isdisjoint(_neighbors(board, h, w))
        labels.append(label)
        boxes.append(box)
    panel_index = {p: i for i, cells in enumerate(panels) for p in cells}
    routes = _components({p for p in external if a[p] == sc})
    assert routes
    socket_map = _socket_map(r, c, n)
    ports, owners = [], []
    panel_ports = [set() for _ in panels]
    frame_neighbor = _neighbors(board, h, w) - board
    for cells in routes:
        touched = {panel_index[q] for q in _neighbors(cells, h, w) if q in panel_index}
        assert len(touched) == 1, 'each cable component must touch exactly one panel'
        owner = next(iter(touched))
        sockets = cells & frame_neighbor
        assert sockets and sockets <= socket_map.keys(), 'only side sockets may touch the frame'
        owners.append(owner)
        ports.append(sockets)
        panel_ports[owner] |= sockets
    assert all(panel_ports), 'each panel must have at least one physical socket'
    assert 5 <= sum(map(len, ports)) <= 12
    counts = {color: len(counters.get(color, ())) for color in gates}
    rotated = np.full((n, n), bg, dtype=np.int8)
    for color in gates:
        for p in map(tuple, np.argwhere(inner == color).tolist()):
            q = _turn(p, n, counts[color])
            assert rotated[q] == bg, 'shutter layers must remain disjoint after rotation'
            rotated[q] = color
    mid = n//2
    assert np.all(rotated[mid-1:mid+2, mid-1:mid+2] == bg)
    enabled = [{p for p in group if rotated[socket_map[p]] != bg} for group in ports]
    votes = [labels[owner] for group, owner in zip(enabled, owners) for _ in group]
    assert len(votes) >= 3 and set(votes) == set(signals)
    winner = _majority(votes)
    assert winner is not None, 'enabled socket vote must be strict'
    return dict(array=a, background=bg, scaffold=sc, signals=signals, gates=gates,
                row=r, col=c, size=n, inner=inner, rotated=rotated,
                counters=counters, counts=counts, board=board, panels=panels,
                panel_boxes=boxes, labels=labels, routes=routes, owners=owners,
                ports=ports, panel_ports=panel_ports, socket_map=socket_map,
                enabled=enabled, votes=votes, winner=winner)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Rotate shutters, propagate panel majorities, then vote at open sockets."""
    info = _read_input(input_grid)
    a = info['array'].copy()
    r, c, n = info['row'], info['col'], info['size']
    a[r+1:r+n+1, c+1:c+n+1] = info['rotated']
    mid = n//2
    a[r+mid:r+mid+3, c+mid:c+mid+3] = info['scaffold']
    a[r+mid+1, c+mid+1] = info['winner']
    for cells in info['counters'].values():
        for p in cells:
            a[p] = info['background']
    for cells, owner in zip(info['routes'], info['owners']):
        for p in cells:
            a[p] = info['labels'][owner]
    return a


def _path(start, targets, blocked, h, w, rng):
    if start in targets:
        return {start}
    costs = 1 + rng.random((h, w))*float(rng.choice([.2, 2., 7.]))
    distance, previous = {start: 0.}, {}
    queue = [(0., *start)]
    while queue:
        d, r, c = heapq.heappop(queue)
        p = r, c
        if d != distance[p]:
            continue
        if p in targets:
            result = {p}
            while p != start:
                p = previous[p]
                result.add(p)
            return result
        for dr, dc in _DIRS:
            q = r+dr, c+dc
            if 0 < q[0] < h-1 and 0 < q[1] < w-1 and q not in blocked:
                nd = d + costs[q]
                if nd < distance.get(q, float('inf')):
                    distance[q], previous[q] = nd, p
                    heapq.heappush(queue, (nd, *q))
    return None


def _layout(rng, n, weights, witness):
    h, w = int(rng.integers(max(22, n+14), 31)), int(rng.integers(max(22, n+14), 31))
    r = int(rng.integers(4, h-n-5))
    c = int(rng.integers(4, w-n-5))
    board = _rect(r, c, n+2, n+2)
    blocked = board | _neighbors(board, h, w, True)
    panels = []
    for i in range(len(weights)):
        for _ in range(75):
            if witness == 0 and i == 0:
                ph, pw = int(rng.integers(1, 3)), int(rng.integers(3, 5))
            elif witness == 0:
                ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            else:
                ph, pw = int(rng.integers(1, 6)), int(rng.integers(2, 6))
            if ph*pw < 3:
                continue
            side = int(rng.integers(4))
            pad = int(rng.integers(1, 4))
            y = pad if side == 0 else (h-ph-pad if side == 2 else int(rng.integers(1, h-ph)))
            x = w-pw-pad if side == 1 else (pad if side == 3 else int(rng.integers(1, w-pw)))
            cells = _rect(y, x, ph, pw)
            if cells & blocked:
                continue
            panels.append((y, x, ph, pw, cells, i))
            blocked |= cells | _neighbors(cells, h, w, True)
            break
        else:
            return None
    # Circular order and contiguous port groups make connections planar; the
    # rectangles themselves are freshly sampled, not selected from saved scenes.
    angle = lambda p: float(np.arctan2(p[0]-(r+(n+1)/2), p[1]-(c+(n+1)/2)))
    ordered = sorted(range(len(panels)), key=lambda i: angle((panels[i][0]+(panels[i][2]-1)/2,
                                                            panels[i][1]+(panels[i][3]-1)/2)))
    socket_map = _socket_map(r, c, n)
    spaced = [p for p, q in socket_map.items() if (q[0] if q[1] in (0, n-1) else q[1]) % 2 == 1]
    if sum(weights) > len(spaced):
        return None
    selected = sorted([spaced[int(j)] for j in rng.choice(len(spaced), sum(weights), replace=False)], key=angle)
    alternatives = []
    for shift in range(len(selected)):
        sequence = selected[shift:] + selected[:shift]
        groups = [set() for _ in panels]
        cursor, score = 0, 0.
        for i in ordered:
            groups[i] = set(sequence[cursor:cursor+weights[i]])
            cursor += weights[i]
            y, x, ph, pw = panels[i][:4]
            pa = angle((y+(ph-1)/2, x+(pw-1)/2))
            for p in groups[i]:
                delta = abs(angle(p)-pa)
                score += min(delta, 2*np.pi-delta)**2
        alternatives.append((score+float(rng.random())*.6, groups))
    groups = min(alternatives, key=lambda item: item[0])[1]
    allpanels = set().union(*(panel[4] for panel in panels))
    collar = _neighbors(board, h, w) - board
    networks = [set(group) for group in groups]
    for i in sorted(range(len(panels)), key=lambda j: -weights[j]):
        ownpanel = panels[i][4]
        other = (allpanels-ownpanel) | set().union(*(net for j, net in enumerate(networks) if j != i))
        forbidden = board | collar | allpanels | other | _neighbors(other, h, w)
        forbidden -= groups[i]
        border = _neighbors(ownpanel, h, w) - ownpanel - forbidden
        if not border:
            return None
        network = set()
        for port in sorted(groups[i]):
            path = _path(port, border | network, forbidden, h, w, rng)
            if path is None:
                return None
            network |= path
        if network & collar != groups[i]:
            return None
        networks[i] = network
    return h, w, r, c, panels, networks, groups, socket_map


def _panel_pattern(rng, ph, pw, majority, minority, strong=False):
    area = ph*pw
    k = 1 if strong else int(rng.integers(1, (area-1)//2+1))
    style = int(rng.integers(4))
    yy, xx = np.indices((ph, pw))
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        y, x = int(rng.integers(ph)), int(rng.integers(pw))
        order = np.argsort((abs(yy-y)+abs(xx-x)+rng.random((ph, pw))*.2).ravel())
    elif style == 2:
        order = np.argsort(((yy+xx) % 2*area + yy*pw+xx).ravel())
    else:
        order = np.argsort((xx*ph+yy).ravel())
    block = np.full(area, majority, dtype=np.int8)
    block[order[:k]] = minority
    return block.reshape(ph, pw)


def _shutters(rng, n, bg, colors, counts, open_cells, closed_cells, asymmetric):
    center = _rect(n//2-1, n//2-1, 3, 3)
    remaining = _rect(0, 0, n, n) - center
    orbits = []
    while remaining:
        p = min(remaining)
        orbit = [_turn(p, n, k) for k in range(4)]
        remaining.difference_update(orbit)
        orbits.append(orbit)
    interior = [i for i, orbit in enumerate(orbits) if all(0 < y < n-1 and 0 < x < n-1 for y, x in orbit)]
    reserved = list(map(int, rng.choice(interior, len(colors), replace=False)))
    owners = list(map(int, rng.integers(len(colors), size=len(orbits))))
    for j, i in enumerate(reserved):
        owners[i] = j
    styles = list(map(int, rng.integers(5, size=len(colors))))
    densities = rng.uniform(.18, .72, len(colors))
    final = np.full((n, n), bg, dtype=np.int8)
    for i, orbit in enumerate(orbits):
        j = owners[i]
        style = styles[j]
        if style == 3:
            phase = int(rng.integers(2))
            chosen = {orbit[phase], orbit[phase+2]} if rng.random() < densities[j] else set()
        elif style == 4:
            chosen = set(orbit) if rng.random() < densities[j] else set()
        else:
            chosen = {p for p in orbit if rng.random() < densities[j]}
            if style == 1:
                chosen = {p for p in orbit if (p[0]+j) % 3 == 0}
            elif style == 2:
                chosen = {p for p in orbit if p[0] < n//2 and rng.random() < .75}
        if i in reserved:
            chosen = {orbit[int(rng.integers(4))]} if asymmetric else set(orbit[::2] if style == 3 else orbit)
        chosen |= set(orbit) & open_cells
        chosen -= closed_cells
        for p in chosen:
            final[p] = colors[j]
    # Construction deliberately uses NumPy rotation. The solver uses point
    # coordinates and independently discovers counters and geometry.
    source = np.full((n, n), bg, dtype=np.int8)
    for color in colors:
        source[np.rot90(final == color, counts.get(color, 0) % 4)] = color
    return source, final


def _place_counters(rng, inp, board, colors, counts, forced):
    h, w = inp.shape
    bg = int(inp[0, 0])
    unavailable = board | _neighbors({tuple(p) for p in np.argwhere(inp != bg).tolist()}, h, w, True)
    cells_by_color = {}
    for j in sorted(range(len(colors)), key=lambda i: -counts.get(colors[i], 0)):
        color, length = colors[j], counts.get(colors[j], 0)
        if not length:
            continue
        candidates = []
        for orientation in (0, 1):
            if j in forced and orientation != forced[j]:
                continue
            dh, dw = (1, length) if orientation == 0 else (length, 1)
            for y in range(1, h-dh):
                for x in range(1, w-dw):
                    cells = _rect(y, x, dh, dw)
                    if not cells & unavailable:
                        candidates.append(cells)
        if not candidates:
            return None
        cells = candidates[int(rng.integers(len(candidates)))]
        for p in cells:
            inp[p] = color
        cells_by_color[color] = cells
        unavailable |= cells | _neighbors(cells, h, w, True)
    return cells_by_color


def _counterfactuals(info):
    sm = info['socket_map']
    labels, owners = info['labels'], info['owners']
    before = [labels[owner] for group, owner in zip(info['ports'], owners)
              for p in group if info['inner'][sm[p]] in info['gates']]
    allports = [labels[owner] for group, owner in zip(info['ports'], owners) for _ in group]
    active_panels = {owner for group, owner in zip(info['enabled'], owners) if group}
    raw = [int(info['array'][p]) for cells in info['panels'] for p in cells]
    return {'before_rotation': _majority(before), 'ungated': _majority(allports),
            'panel_count': _majority(labels),
            'active_panel_count': _majority(labels[i] for i in active_panels),
            'raw_signal_cells': _majority(raw),
            'panel_area': _majority(label for label, panel in zip(labels, info['panels']) for _ in panel)}


def _make_pair(rng, mode, split, witness, winner_index):
    if witness == 0:
        weights, active = [3, 3, 2], [3, 1, 1]
        bits = [True, False, False]
        lengths = [1, 2, 2] if mode != 1 else [5, 6, 6]
    elif witness == 1:
        weights, active = [3, 2, 2], [2, 1, 2]
        bits = [False, True, True]
        lengths = [3, 4, 0] if mode != 1 else [7, 8, 0]
    else:
        count = int(rng.integers(2, 5 if mode == 2 and split == 'train' else 6))
        if mode == 2 and split == 'test':
            count = 5
        cap = 3 if mode == 2 and split == 'train' else 4
        profiles = [p for p in itertools.product(range(1, cap+1), repeat=count)
                    if 5 <= sum(p) <= 12 and (mode != 2 or split != 'test' or max(p) == 4)]
        weights = list(profiles[int(rng.integers(len(profiles)))])
        for _ in range(200):
            bits = list(map(bool, rng.integers(2, size=count)))
            active = [int(rng.integers(weight+1)) for weight in weights]
            votes = [bit for bit, k in zip(bits, active) for _ in range(k)]
            if len(votes) >= 3 and set(votes) == {False, True} and sum(votes)*2 > len(votes) and sum(active) < sum(weights):
                break
        else:
            # Every feasible weight profile admits this strict, partly closed
            # electorate. The fallback is arithmetic, not a saved grid.
            bits, active = [False]*count, [0]*count
            biggest = int(np.argmax(weights))
            if weights[biggest] >= 2:
                bits[biggest], active[biggest] = True, 2
                active[(biggest+1) % count] = 1
            else:
                bits[:3], active[:4] = [True]*3, [1]*4
        length_cap = 4 if mode == 0 and split == 'train' else 8
        gate_count = int(rng.integers(2, 5))
        lengths = list(map(int, rng.integers(0, length_cap+1, size=gate_count)))
        if not any(lengths):
            lengths[0] = 1
        if mode == 0 and split == 'test':
            lengths[int(rng.integers(gate_count))] = int(rng.integers(5, 9))
    gate_count = len(lengths)
    roles = list(map(int, rng.choice(10, 4+gate_count, replace=False)))
    bg, sc = roles[:2]
    signals = sorted(roles[2:4])
    colors = roles[4:]
    win, lose = signals[winner_index], signals[1-winner_index]
    labels = [win if bit else lose for bit in bits]
    counts = dict(zip(colors, lengths))
    forced = {1: int(rng.integers(2))} if witness == 0 else {}
    if witness == 0:
        forced[2] = 1-forced[1]
    sizes = (7, 9) if mode == 1 and split == 'train' else ((11, 13) if mode == 1 else (7, 9, 11, 13))
    for _ in range(180):
        n = int(rng.choice(sizes))
        layout = _layout(rng, n, weights, witness)
        if layout is None:
            continue
        h, w, r, c, panels, networks, groups, sm = layout
        open_ports = set()
        for group, k in zip(groups, active):
            sequence = sorted(group)
            open_ports |= {sequence[int(j)] for j in rng.choice(len(sequence), k, replace=False)}
        open_cells = {sm[p] for p in open_ports}
        closed_cells = {sm[p] for group in groups for p in group if p not in open_ports}
        source = final = None
        for _ in range(50):
            source, final = _shutters(rng, n, bg, colors, counts, open_cells, closed_cells, witness >= 0)
            before = [label for label, group in zip(labels, groups) for p in group if source[sm[p]] != bg]
            if witness < 0 or _majority(before) == lose:
                break
        else:
            continue
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[r:r+n+2, c:c+n+2] = sc
        inp[r+1:r+n+1, c+1:c+n+1] = source
        mid = n//2
        inp[r+mid:r+mid+3, c+mid:c+mid+3] = sc
        for i, (y, x, ph, pw, cells, _) in enumerate(panels):
            block = _panel_pattern(rng, ph, pw, labels[i], win if labels[i] == lose else lose,
                                   strong=witness == 0 and i > 0)
            inp[y:y+ph, x:x+pw] = block
            for p in networks[i]:
                inp[p] = sc
        counter_cells = _place_counters(rng, inp, _rect(r, c, n+2, n+2), colors, counts, forced)
        if counter_cells is None:
            continue
        out = inp.copy()
        out[r+1:r+n+1, c+1:c+n+1] = final
        out[r+mid:r+mid+3, c+mid:c+mid+3] = sc
        out[r+mid+1, c+mid+1] = win
        for i, net in enumerate(networks):
            for p in net:
                out[p] = labels[i]
        for cells in counter_cells.values():
            for p in cells:
                out[p] = bg
        try:
            info = _read_input(inp)
            assert np.array_equal(solve(inp), out)
            if witness == 0:
                assert all(v == lose for v in _counterfactuals(info).values())
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded circuit construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and smaller than num_train')
    rng = np.random.default_rng(seed)
    mode, first_winner = int(rng.integers(3)), int(rng.integers(2))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = slots.index(index) if split == 'train' and index in slots else -1
            winner_index = first_winner ^ witness if witness >= 0 else int(rng.integers(2))
            for _ in range(8):
                pair = _make_pair(rng, mode, split, witness, winner_index)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    used, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            info = _read_input(inp)
            assert np.array_equal(solve(inp), out)
            # Family-specific conservation and preservation, not just solve equality.
            r, c, n = info['row'], info['col'], info['size']
            output_inner = out[r+1:r+n+1, c+1:c+n+1]
            for color in info['gates']:
                assert np.sum(info['inner'] == color) == np.sum(output_inner == color)
                assert not any(out[p] == color for p in info['counters'].get(color, ()))
            for cells in info['panels']:
                assert all(inp[p] == out[p] for p in cells)
            altered = (info['board'] | set().union(*info['routes'])
                       | set().union(*info['counters'].values()))
            assert all(inp[y, x] == out[y, x] for y in range(inp.shape[0])
                       for x in range(inp.shape[1]) if (y, x) not in altered)
            mid = n//2
            hub = _rect(r+mid, c+mid, 3, 3)
            hub.remove((r+mid+1, c+mid+1))
            frame = info['board'] - _rect(r+1, c+1, n, n)
            assert all(out[p] == info['scaffold'] for p in hub | frame)
            if split == 'train':
                training.append(info)
    assert {i['signals'].index(i['winner']) for i in training} == {0, 1}
    assert any(all(v is not None and v != i['winner'] for v in _counterfactuals(i).values()) for i in training)
    # Training must teach 1/2/3 turns, a cycle, no-marker defaults, independent
    # rotation, and equal counters in different orientations, on asymmetric masks.
    residues, absent, repeated_orientation, independent = set(), False, False, False
    for info in training:
        local = set()
        lengths_to_orientation = {}
        for color in info['gates']:
            mask = info['inner'] == color
            asymmetric = all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))
            if not asymmetric:
                continue
            length = info['counts'][color]
            absent |= length == 0
            if length:
                residues.add(length % 4)
                local.add(length % 4)
                cells = info['counters'][color]
                orientation = len({p[0] for p in cells}) == 1
                lengths_to_orientation.setdefault(length, set()).add(orientation)
        repeated_orientation |= any(len(v) == 2 for v in lengths_to_orientation.values())
        independent |= len(local) >= 2
    assert residues == {0, 1, 2, 3} and absent and repeated_orientation and independent


def self_test():
    """Bounded 42-seed semantic and variation audit; no files or network."""
    shapes, sizes, layouts, normalized, palettes, lengths = set(), set(), set(), set(), set(), set()
    panel_counts, port_counts, open_counts, margins, modes = set(), set(), set(), set(), set()
    zero_weight, shared, multi_side, unchanged = False, False, False, False
    puzzles = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        train_sizes, train_lengths = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                shapes.add(pair['input'].shape)
                sizes.add(info['size'])
                layouts.add((info['row'], info['col'], tuple(info['panel_boxes']),
                             tuple(tuple(sorted(ps)) for ps in info['panel_ports'])))
                roles = {}
                normalized.add(tuple(roles.setdefault(int(v), len(roles)) for v in pair['input'].flat))
                palettes.add((info['background'], info['scaffold'], info['signals'], info['gates']))
                lengths.update(info['counts'].values())
                panel_counts.add(len(info['panels']))
                physical = sum(map(len, info['ports']))
                active = len(info['votes'])
                port_counts.add(physical)
                open_counts.add(active)
                margins.add(abs(2*info['votes'].count(info['winner'])-active))
                for owner, sockets in enumerate(info['panel_ports']):
                    enabled = set().union(*(group for group, own in zip(info['enabled'], info['owners']) if own == owner))
                    zero_weight |= not enabled
                    shared |= len(enabled) >= 2
                    side = lambda p: (0 if p[0] < info['row'] else 2 if p[0] > info['row']+info['size']+1
                                      else 3 if p[1] < info['col'] else 1)
                    multi_side |= len({side(p) for p in sockets}) >= 2
                unchanged |= any(info['counts'][color] and np.array_equal(info['inner'] == color, info['rotated'] == color)
                                 for color in info['gates'])
                if split == 'train':
                    train_sizes.append(info['size'])
                    train_lengths.extend(info['counts'].values())
                else:
                    if max(train_lengths) <= 4 and max(info['counts'].values()) >= 5:
                        modes.add('counts')
                    if max(train_sizes) <= 9 and info['size'] >= 11:
                        modes.add('scale')
                    if len(info['panels']) == 5 and max(map(len, info['panel_ports'])) == 4:
                        modes.add('multiplicity')
    assert len(puzzles) == 42 and len(layouts) >= 180 and len(normalized) >= 180
    assert len(shapes) >= 35 and sizes == {7, 9, 11, 13} and len(palettes) >= 100
    assert lengths == set(range(9)) and panel_counts == {2, 3, 4, 5}
    assert len(port_counts) >= 5 and len(open_counts) >= 5 and len(margins) >= 4
    assert modes == {'counts', 'scale', 'multiplicity'}
    assert zero_weight and shared and multi_side and unchanged
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            validate_generated_puzzle(generate_puzzle(912, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    assert tuple(_key(p['input']) for s in ('train', 'test') for p in generate_puzzle(43)[s]) == tuple(
        _key(p['input']) for s in ('train', 'test') for p in generate_puzzle(43)[s])
    # A wrong collector label or destroyed minority must fail validation.
    puzzle = generate_puzzle(91, 2, 1)
    pair = puzzle['train'][0]
    info = _read_input(pair['input'])
    # Rotating the complete scene changes placement and bar orientation but
    # commutes with this rule. This also checks the four socket alignments.
    for k in (1, 2, 3):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    short = next(iter(info['counters']))
    changed = pair['input'].copy()
    cells = info['counters'][short]
    for p in cells:
        changed[p] = info['background']
    orientation = int(len({p[1] for p in cells}) == 1)
    equivalent_length = len(cells)+4 if len(cells) <= 4 else len(cells)-4
    placed = _place_counters(np.random.default_rng(314), changed, info['board'], [short],
                             {short: equivalent_length}, {0: 1-orientation})
    assert placed is not None
    assert np.array_equal(solve(changed), pair['output']), 'full cycle and counter location must be irrelevant'
    # Reordering minority pixels changes no vote or route. The output must copy
    # the rearranged panel without losing even one original minority cell.
    changed = pair['input'].copy()
    expected = pair['output'].copy()
    y, x, ph, pw = info['panel_boxes'][0]
    block = np.sort(changed[y:y+ph, x:x+pw], axis=None).reshape(ph, pw)
    changed[y:y+ph, x:x+pw] = block
    expected[y:y+ph, x:x+pw] = block
    assert np.array_equal(solve(changed), expected)
    for kind in ('center', 'minority'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        pair = bad['test'][0]
        info = _read_input(pair['input'])
        if kind == 'center':
            p = info['row']+info['size']//2+1, info['col']+info['size']//2+1
        else:
            p = next(p for p in info['panels'][0] if pair['input'][p] != info['labels'][0])
        pair['output'][p] = (int(pair['output'][p])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'layouts': len(layouts),
            'normalized_structures': len(normalized), 'palettes': len(palettes),
            'board_sizes': sorted(sizes), 'counter_lengths_including_absence': sorted(lengths),
            'panel_counts': sorted(panel_counts), 'physical_port_counts': sorted(port_counts),
            'enabled_port_counts': sorted(open_counts), 'vote_margins': sorted(margins),
            'shifts': sorted(modes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
