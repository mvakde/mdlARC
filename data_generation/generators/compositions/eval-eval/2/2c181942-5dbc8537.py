"""Socketed mosaic: color-directed, rotated, finite-inventory exact cover.

Only standard-library modules and NumPy are used. Generation, prediction,
validation, and self-testing do not read files or write diagnostic payloads.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


_NORMALS = ((-1, 0), (0, -1), (1, 0), (0, 1))
_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer color grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _FOUR:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _perimeter_uniform(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _panel(grid):
    """Identify a complete board frame without an output shape or label."""
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        cells = np.argwhere(grid == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0) + 1
        if min(r1 - r0, c1 - c0) < 5:
            continue
        board = grid[r0:r1, c0:c1]
        if not _perimeter_uniform(board, color):
            continue
        if r0 == 0 and r1 == h and c0 == 0 and c1 < w:
            tray, side = grid[:, c1:], "left"
        elif r0 == 0 and r1 == h and c1 == w and c0 > 0:
            tray, side = grid[:, :c0], "right"
        elif c0 == 0 and c1 == w and r0 == 0 and r1 < h:
            tray, side = grid[r1:], "top"
        elif c0 == 0 and c1 == w and r1 == h and r0 > 0:
            tray, side = grid[:r0], "bottom"
        else:
            continue
        if min(tray.shape) < 3:
            continue
        vacancy = int(tray[0, 0])
        if (vacancy == color or np.any(tray == color)
                or not _perimeter_uniform(tray, vacancy)
                or not np.any(board == vacancy)):
            continue
        candidates.append((board.copy(), tray.copy(), color, vacancy, side))
    if len(candidates) != 1:
        raise ValueError("The input must identify one framed board")
    return candidates[0]


def _sockets(board, wall, vacancy):
    sockets = []
    for cells in _components((board != wall) & (board != vacancy)):
        cells = sorted(cells)
        if len(cells) != 2:
            raise ValueError("Each socket must be a separated domino")
        a, b = cells
        color = int(board[a])
        if int(board[b]) != color or abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise ValueError("Socket cells must be equal and adjacent")
        choices = (0, 2) if a[0] == b[0] else (1, 3)
        directions = []
        for k in choices:
            dr, dc = _NORMALS[k]
            target = [(r + dr, c + dc) for r, c in cells]
            back = [(r - dr, c - dc) for r, c in cells]
            if not all(0 <= r < board.shape[0] and 0 <= c < board.shape[1]
                       for r, c in target + back):
                continue
            if all(board[p] == vacancy for p in target) and all(board[p] == wall for p in back):
                directions.append((k, target))
        if len(directions) != 1:
            raise ValueError("A socket must have one vacancy-facing normal")
        k, target = directions[0]
        sockets.append(dict(color=color, k=k, cells=cells, target=np.array(target, dtype=int)))
    if not sockets:
        raise ValueError("The board has no sockets")
    # Prevent two neighboring bars being interpreted as separate terminals.
    for i, s in enumerate(sockets):
        for t in sockets[i + 1:]:
            if any(max(abs(r - y), abs(c - x)) <= 1
                   for r, c in s['cells'] for y, x in t['cells']):
                raise ValueError("Socket bars must be diagonally separated")
    return sockets


def _canonical_piece(piece, vacancy, socket_colors):
    """Infer the terminal from visible geometry and color, including absent tags."""
    candidates = {}
    for undo in range(4):
        a = np.rot90(piece, undo).copy()
        mask = a != vacancy
        h, w = a.shape
        if w < 3 or h < 2 or not np.all(mask[0]):
            continue
        head = 0
        while head < h and np.all(mask[head]):
            head += 1
        if head == h:
            continue
        cols = np.flatnonzero(mask[-1])
        if len(cols) != 2 or cols[1] != cols[0] + 1:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[cols] = True
        if not np.all(mask[head:] == neck):
            continue
        tag = int(a[-1, cols[0]])
        if (tag == vacancy or a[-1, cols[1]] != tag
                or np.count_nonzero(a == tag) != 2):
            continue
        body = a[mask].copy()
        body = body[body != tag]
        if any(np.any(body == c) for c in socket_colors):
            continue
        key = _key(a)
        candidates[key] = dict(pattern=a, mask=mask, tag=tag, undo=undo,
                               geometry=(head, w, h - head, int(cols[0])))
    if len(candidates) != 1:
        raise ValueError("A source needs one geometrically identifiable tagged neck")
    return next(iter(candidates.values()))


def _decode(input_grid):
    grid = _grid(input_grid)
    board, tray, wall, vacancy, side = _panel(grid)
    sockets = _sockets(board, wall, vacancy)
    labels = {s['color'] for s in sockets}
    groups, lookup, pieces = [], {}, []
    blocked = set()
    for cells in _components(tray != vacancy):
        if any(p in blocked for p in cells):
            raise ValueError("Loose source objects must have a diagonal moat")
        for r, c in cells:
            blocked.update((r + dr, c + dc) for dr, dc in _EIGHT)
        rr, cc = np.array(cells).T
        patch = tray[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()
        p = _canonical_piece(patch, vacancy, labels)
        key = _key(p['pattern'])
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append(dict(pattern=p['pattern'], mask=p['mask'], tag=p['tag'],
                               geometry=p['geometry'], count=0, undos=[]))
        g = lookup[key]
        groups[g]['count'] += 1
        groups[g]['undos'].append(p['undo'])
        pieces.append(p)
    if not groups:
        raise ValueError("No loose inventory")
    return dict(board=board, tray=tray, wall=wall, vacancy=vacancy,
                sockets=sockets, groups=groups, pieces=pieces, side=side)


def _placements(a, ignore_colors=False):
    board, vacancy = a['board'], a['vacancy']
    coords = np.argwhere(board == vacancy)
    indices = {tuple(p): i for i, p in enumerate(coords)}
    n = len(coords)
    result = []
    for si, socket in enumerate(a['sockets']):
        for gi, group in enumerate(a['groups']):
            if not ignore_colors and group['tag'] != socket['color']:
                continue
            p = np.rot90(group['pattern'], socket['k'])
            marker = np.argwhere(p == group['tag'])
            shift = socket['target'].min(axis=0) - marker.min(axis=0)
            if {tuple(q) for q in marker + shift} != {tuple(q) for q in socket['target']}:
                continue
            local = np.argwhere(p != vacancy)
            target = local + shift
            points = [tuple(q) for q in target]
            if any(q not in indices for q in points):
                continue
            ix = tuple(indices[q] for q in points)
            bits = (1 << (n + si)) | sum(1 << j for j in ix)
            colors = tuple(int(p[tuple(q)]) for q in local)
            result.append(dict(group=gi, socket=si, bits=bits, ix=ix, colors=colors))
    return coords, result


def _pack(a, ignore_colors=False, unlimited=False, max_nodes=50000):
    """Enumerate up to two distinct images; equal-copy permutations are grouped."""
    coords, placements = _placements(a, ignore_colors)
    n = len(coords)
    total = n + len(a['sockets'])
    by_cell = [[] for _ in range(total)]
    for i, p in enumerate(placements):
        for cell in p['ix']:
            by_cell[cell].append(i)
        by_cell[n + p['socket']].append(i)
    remain = [len(a['sockets']) if unlimited else g['count'] for g in a['groups']]
    full = (1 << total) - 1
    rendered = np.full(n, -1, dtype=np.int8)
    outcomes, path, dead = {}, [], set()
    nodes, complete = 0, True

    def visit(covered):
        nonlocal nodes, complete
        if len(outcomes) > 1 or not complete:
            return False
        nodes += 1
        if nodes > max_nodes:
            complete = False
            return False
        if covered == full:
            outcomes.setdefault(rendered.tobytes(), tuple(path))
            return True
        state = covered, tuple(remain)
        if state in dead:
            return False
        best = None
        # Include sockets as exact-cover columns: one whole object per terminal.
        for cell in list(range(n, total)) + list(range(n)):
            if covered & (1 << cell):
                continue
            options = [i for i in by_cell[cell]
                       if remain[placements[i]['group']] and not (covered & placements[i]['bits'])]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
        found = False
        for i in best:
            p = placements[i]
            remain[p['group']] -= 1
            rendered[list(p['ix'])] = p['colors']
            path.append(i)
            found = visit(covered | p['bits']) or found
            path.pop()
            rendered[list(p['ix'])] = -1
            remain[p['group']] += 1
            if len(outcomes) > 1 or not complete:
                return found
        if not found:
            dead.add(state)
        return found

    visit(0)
    return dict(outcomes=outcomes, complete=complete, nodes=nodes,
                coords=coords, placements=placements)


def _inspect(input_grid):
    a = _decode(input_grid)
    packed = _pack(a)
    if not packed['complete']:
        raise ValueError("Exact-cover search budget exhausted")
    if len(packed['outcomes']) != 1:
        raise ValueError("The input needs exactly one rendered completion")
    colors, path = next(iter(packed['outcomes'].items()))
    output = a['board'].copy()
    rr, cc = packed['coords'].T
    output[rr, cc] = np.frombuffer(colors, dtype=np.int8)
    a.update(packed=packed, path=path, output=output)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the full result from this input alone, without generation state."""
    return _inspect(input_grid)['output']


def _geometry(rng, large=False):
    head = int(rng.integers(2 if large else 1, 5))
    width = int(rng.integers(3, 7))
    stem = int(rng.integers(1, 3))
    offset = int(rng.integers(width - 1))
    return head, width, stem, offset


def _mask(geom):
    h, w, s, x = geom
    mask = np.zeros((h + s, w), dtype=bool)
    mask[:h] = True
    mask[h:, x:x + 2] = True
    return mask


def _rotate_point(r, c, h, w, k):
    return ((r, c), (w - 1 - c, r), (h - 1 - r, w - 1 - c), (c, h - 1 - r))[k]


def _geometry_cells(geom, k):
    mask = _mask(geom)
    h, w = mask.shape
    x = geom[3]
    cells = set(map(tuple, np.argwhere(np.rot90(mask, k))))
    socket = {_rotate_point(h, c, h, w, k) for c in (x, x + 1)}
    back = {_rotate_point(h + 1, c, h, w, k) for c in (x, x + 1)}
    return cells, socket, back


def _shift(cells, r, c):
    return {(y + r, x + c) for y, x in cells}


def _grow(rng, geometries, turns, broad_join=0):
    """Grow a connected irregular cavity; reserve socket bars and their backs."""
    occupied, sockets, backs, records = set(), set(), set(), []
    for index, (geom, k) in enumerate(zip(geometries, turns)):
        local, port, support = _geometry_cells(geom, k)
        options = sorted(local)
        accepted = False
        boundary = sorted({(r + dr, c + dc) for r, c in occupied for dr, dc in _FOUR}
                          - occupied - sockets - backs)
        for _ in range(320 if records and not (index == 1 and broad_join) else 1):
            if index == 1 and broad_join:
                # Two opposite-facing heads share a whole side. Their necks
                # point out of the joined rectangle. Random rotation, tab offset,
                # dimensions, side of contact, and subsequent growth all vary.
                h, w, s, x = geom
                other, _, _ = _geometry_cells(geom, 2)
                other = _shift(other, -s, broad_join * w)
                transformed = {_rotate_point(r, c, h + s, w, turns[0]) for r, c in other}
                dr, dc = min(r for r, c in transformed), min(c for r, c in transformed)
            elif records:
                target = boundary[int(rng.integers(len(boundary)))]
                anchor = options[int(rng.integers(len(options)))]
                dr, dc = target[0] - anchor[0], target[1] - anchor[1]
            else:
                dr, dc = 0, 0
            placed, terminal, backing = (_shift(q, dr, dc) for q in (local, port, support))
            if placed & (occupied | sockets | backs):
                continue
            if terminal & (occupied | sockets | backs) or backing & (occupied | sockets):
                continue
            if any(max(abs(r - y), abs(c - x)) <= 1
                   for r, c in terminal for y, x in sockets):
                continue
            all_cells = occupied | sockets | backs | placed | terminal | backing
            ys, xs = zip(*all_cells)
            if max(ys) - min(ys) + 1 > 26 or max(xs) - min(xs) + 1 > 15:
                continue
            occupied |= placed
            sockets |= terminal
            backs |= backing
            records.append(dict(geom=geom, k=k, origin=(dr, dc),
                                cells=placed, socket=terminal))
            accepted = True
            break
        if not accepted:
            return None
    return records, occupied | sockets | backs


def _pattern(rng, geom, tag, payload, vacancy):
    mask = _mask(geom)
    a = np.full(mask.shape, vacancy, dtype=np.int8)
    base = int(rng.choice(payload))
    a[mask] = base
    mode = int(rng.integers(5))
    rr, cc = np.indices(a.shape)
    if len(payload) > 1:
        other = int(rng.choice([c for c in payload if c != base]))
        if mode == 1:
            a[mask & ((rr + cc) % 2 == 1)] = other
        elif mode == 2:
            a[mask & (rr % 2 == 1)] = other
        elif mode == 3:
            a[mask] = rng.choice(payload, int(mask.sum()))
        elif mode == 4:
            a[0, 0] = other
    a[-1, geom[3]:geom[3] + 2] = tag
    return a


def _scatter(rng, tiles, height, max_width, vacancy):
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    minimum = max(t.shape[1] for t in tiles) + 2
    if minimum > max_width or max(t.shape[0] for t in tiles) + 2 > height:
        return None
    for attempt in range(12):
        width = max_width if attempt >= 8 else int(rng.integers(minimum, max_width + 1))
        tray = np.full((height, width), vacancy, dtype=np.int8)
        blocked = np.zeros(tray.shape, dtype=bool)
        for i in order:
            tile = tiles[i]
            th, tw = tile.shape
            local = np.argwhere(tile != vacancy)
            tops, lefts = np.indices((height - th - 1, width - tw - 1))
            rr = tops.ravel()[:, None] + local[:, 0] + 1
            cc = lefts.ravel()[:, None] + local[:, 1] + 1
            legal = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
            if not len(legal):
                break
            j = int(rng.choice(legal))
            tray[rr[j], cc[j]] = tile[tile != vacancy]
            for r, c in zip(rr[j], cc[j]):
                blocked[r - 1:r + 2, c - 1:c + 2] = True
        else:
            return tray
    return None


def _pair(rng, count, side_turns, witness, reuse=False):
    for attempt in range(160):
        palette = list(map(int, rng.permutation(10)))
        wall, vacancy = palette[:2]
        label_max = min(4, count - (2 if reuse else 1 if witness == 'copies' else 0))
        label_count = int(rng.integers(2 if witness == 'colors' else 1, label_max + 1))
        labels = palette[2:2 + label_count]
        payload = palette[2 + label_count:9]
        payload = payload[:int(rng.integers(1, min(4, len(payload)) + 1))]
        absent_tag = palette[9]
        geoms = [_geometry(rng, large=witness == 'colors' and i < 2) for i in range(count)]
        if witness in ('colors', 'copies') or reuse:
            geoms[1] = geoms[0]
        if reuse:
            geoms[2] = geoms[0]
        repeated = 3 if reuse else 2 if witness in ('colors', 'copies') else 1
        # Training demonstrates two copies; reuse-shift tests use three.
        # Other geometry stays distinct, keeping rotation witnesses attributable.
        for i in range(repeated, count):
            for _ in range(24):
                if geoms[i] not in geoms[:i]:
                    break
                geoms[i] = _geometry(rng)
            else:
                geoms[i] = next((h, w, s, x) for h in range(1, 5)
                                for w in range(3, 7) for s in (1, 2)
                                for x in range(w - 1) if (h, w, s, x) not in geoms[:i])
        turns = list(map(int, rng.integers(0, 4, count)))
        broad_join = int(rng.choice((-1, 1))) if witness == 'colors' else 0
        if witness == 'colors':
            phase = int(rng.integers(4))
            turns[:4] = [(phase + d) % 4 for d in (0, 2, 1, 3)]
            if rng.random() < .5:
                turns[2], turns[3] = turns[3], turns[2]
        grown = _grow(rng, geoms, turns, broad_join)
        if grown is None:
            continue
        records, extent = grown
        ys, xs = zip(*extent)
        bh, bw = max(ys) - min(ys) + 3, max(xs) - min(xs) + 3
        height = max(bh, int(rng.integers(18, 29)))
        width = max(bw, int(rng.integers(9, 16)))
        width = min(18, width + int(rng.integers(0, 2)))
        height = min(30, height + int(rng.integers(0, 3)))
        if width < bw or height < bh:
            continue
        shift_r = 1 - min(ys) + int(rng.integers(height - bh + 1))
        shift_c = 1 - min(xs) + int(rng.integers(width - bw + 1))
        tags = list(map(int, rng.choice(labels, count)))
        if witness == 'colors':
            for i, color in enumerate(labels):
                tags[i] = color
        elif reuse or witness == 'copies':
            copies = 3 if reuse else 2
            tags[:copies] = [labels[0]] * copies
            for i, color in enumerate(labels[1:], copies):
                tags[i] = color
        else:
            for i, color in enumerate(labels):
                tags[i] = color
        patterns, cache = [], {}
        for geom, tag in zip(geoms, tags):
            key = geom, tag
            if key not in cache:
                cache[key] = _pattern(rng, geom, tag, payload, vacancy)
            patterns.append(cache[key].copy())
        board = np.full((height, width), wall, dtype=np.int8)
        completed = board.copy()
        for rec, tag, pattern in zip(records, tags, patterns):
            for r, c in rec['cells']:
                board[r + shift_r, c + shift_c] = vacancy
            for r, c in rec['socket']:
                board[r + shift_r, c + shift_c] = tag
                completed[r + shift_r, c + shift_c] = tag
            oriented = np.rot90(pattern, rec['k'])
            rr, cc = np.nonzero(oriented != vacancy)
            dr, dc = rec['origin']
            completed[rr + dr + shift_r, cc + dc + shift_c] = oriented[rr, cc]
        try:
            _sockets(board, wall, vacancy)
        except ValueError:
            continue
        # A genuinely locally fitting but incomplete same-tag candidate.
        if witness == 'colors' or rng.random() < .45:
            eligible = [i for i, geom in enumerate(geoms) if geom[0] > 1]
            if eligible:
                i = 0 if witness == 'colors' else int(rng.choice(eligible))
                h, w, s, x = geoms[i]
                depths = [d for d in range(1, h)
                          if witness != 'colors' or ((d, w, s, x), tags[i]) not in cache]
                if not depths:
                    continue
                small = (int(rng.choice(depths)), w, s, x)
                cache.setdefault((small, tags[i]), _pattern(rng, small, tags[i], payload, vacancy))
                patterns.append(cache[(small, tags[i])].copy())
        if witness == 'colors':
            # This object fits wholly in the vacancy at the first socket, but
            # steals the second root's head. Its rejection requires joint packing.
            h, w, s, x = geoms[0]
            large = (h, 2 * w, s, x if broad_join > 0 else x + w)
            if (large, tags[0]) in cache:
                continue
            patterns.append(_pattern(rng, large, tags[0], payload, vacancy))
        # Absent tags can have fully legitimate docking geometry, including large heads.
        if rng.random() < .55:
            patterns.append(_pattern(rng, _geometry(rng), absent_tag, payload, vacancy))
        tiles = []
        for i, pattern in enumerate(patterns):
            delta = i if witness == 'colors' and i < 4 else int(rng.integers(4))
            input_turns = (turns[i] - delta) % 4 if i < count else int(rng.integers(4))
            tiles.append(np.rot90(pattern, input_turns).copy())
        tray = _scatter(rng, tiles, height, 30 - width, vacancy)
        if tray is None:
            continue
        inp = np.rot90(np.concatenate((board, tray), axis=1), side_turns).copy()
        out = np.rot90(completed, side_turns).copy()
        try:
            a = _inspect(inp)
        except ValueError:
            continue
        if not np.array_equal(a['output'], out):
            continue
        if witness == 'colors' and not (_smallest_fails(a) and _smallest_fails(a, largest=True)):
            continue
        if witness == 'colors' and _features(a)['rotations'] != set(range(4)):
            continue
        return dict(input=inp, output=out)
    raise RuntimeError("Bounded construction could not find a unique socketed mosaic")


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # more roots; new panel axis; more label reuse
    phase = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(num_train + num_test):
        training = i < num_train
        if training:
            count = int(rng.integers(4 if i == 0 else 3, 6))
        else:
            count = int(rng.integers(6 if regime == 0 else 4, 8))
        if regime == 1:
            side = (phase + 2 * int(rng.integers(2)) + (0 if training else 1)) % 4
        else:
            side = int(rng.integers(4))
        witness = 'colors' if i == 0 else 'copies' if i == 1 else None
        for _ in range(6):
            pair = _pair(rng, count, side, witness, reuse=not training and regime == 2)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError("Could not create distinct inputs within retry budget")
        puzzle['train' if training else 'test'].append(pair)
    return puzzle


def _smallest_fails(a, largest=False):
    """Concrete ablations: smallest/largest local fit at each socket separately."""
    placements = a['packed']['placements']
    covered, counts = set(), Counter()
    out = a['board'].copy()
    coords = a['packed']['coords']
    for si in range(len(a['sockets'])):
        options = [p for p in placements if p['socket'] == si]
        if not options:
            return True
        chooser = max if largest else min
        p = chooser(options, key=lambda q: (len(q['ix']), q['group']))
        if covered.intersection(p['ix']):
            return True
        counts[p['group']] += 1
        if counts[p['group']] > a['groups'][p['group']]['count']:
            return True
        covered.update(p['ix'])
        for cell, color in zip(p['ix'], p['colors']):
            out[tuple(coords[cell])] = color
    return len(covered) != len(coords) or not np.array_equal(out, a['output'])


def _features(a):
    selected = [a['packed']['placements'][i] for i in a['path']]
    used = Counter(p['group'] for p in selected)
    rotations = set()
    for p in selected:
        group = a['groups'][p['group']]
        if group['count'] == 1:
            rotations.add((group['undos'][0] + a['sockets'][p['socket']]['k']) % 4)
    return dict(sockets=len(a['sockets']), side=a['side'], shape=a['board'].shape,
                labels=len({s['color'] for s in a['sockets']}),
                directions={s['k'] for s in a['sockets']}, rotations=rotations,
                copies=any(n >= 2 for n in used.values()),
                max_copies=max(used.values()),
                surplus=sum(g['count'] for g in a['groups']) - len(selected),
                local_surplus=any(p['group'] not in used for p in a['packed']['placements']),
                smallest_fails=_smallest_fails(a),
                largest_fails=_smallest_fails(a, largest=True),
                geometries={a['groups'][p['group']]['geometry'] for p in selected},
                nodes=a['packed']['nodes'])


def validate_generated_puzzle(puzzle) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError("Exactly train and test are required")
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), []
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        key = _key(inp)
        assert key not in seen, "Inputs must be distinct"
        seen.add(key)
        try:
            a = _inspect(inp)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        assert np.array_equal(out, a['output']), "Incorrect socketed packing"
        assert np.all(out[a['board'] != a['vacancy']] == a['board'][a['board'] != a['vacancy']])
        selected = [a['packed']['placements'][i] for i in a['path']]
        counts = Counter(p['group'] for p in selected)
        assert len(selected) == len(a['sockets'])
        assert all(n <= a['groups'][g]['count'] for g, n in counts.items())
        assert len(_components(a['board'] == a['vacancy'])) == 1
        assert len(selected) >= 3, "One merged region must require multiple objects"
        analyses.append(a)
    nt = len(puzzle['train'])
    features = [_features(a) for a in analyses[:nt]]
    assert set().union(*(f['directions'] for f in features)) == set(range(4))
    assert set().union(*(f['rotations'] for f in features)) == set(range(4))
    assert any(f['copies'] for f in features), "Demonstrate repeated physical copies"
    assert any(f['local_surplus'] and f['smallest_fails'] for f in features), "Demonstrate global packing"
    assert any(f['largest_fails'] for f in features), "Reject locally fitting objects that steal another root's space"
    without_a = _pack(analyses[0], ignore_colors=True)
    assert without_a['complete'] and len(without_a['outcomes']) == 2, "Tag equality must change the answer"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(p[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streamed metrics, no files or retained puzzle collection."""
    signatures, shapes, geometries, sides, normalized = set(), set(), set(), set(), set()
    socket_counts, label_counts, surplus_counts = set(), set(), set()
    pairs, repeated, local, largest_search, max_copies = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                a = _inspect(p['input'])
                f = _features(a)
                pairs += 1
                shapes.add(tuple(p['input'].shape))
                geometries.update(f['geometries'])
                sides.add(f['side'])
                socket_counts.add(f['sockets'])
                label_counts.add(f['labels'])
                surplus_counts.add(f['surplus'])
                repeated += f['copies']
                local += f['local_surplus']
                largest_search = max(largest_search, f['nodes'])
                max_copies = max(max_copies, f['max_copies'])
                roles = {}
                image = bytes(roles.setdefault(int(c), len(roles)) for c in p['input'].ravel())
                normalized.add(hashlib.sha256(bytes(p['input'].shape) + image).hexdigest())
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(geometries) >= 50 and len(normalized) == pairs
    assert sides == {'left', 'right', 'top', 'bottom'}
    assert {3, 4, 5, 6, 7} <= socket_counts
    assert {2, 3, 4} <= label_counts and {0, 1, 2} <= surplus_counts
    assert repeated >= 42 and local >= 42
    assert max_copies >= 3
    assert {g[0] for g in geometries} == {1, 2, 3, 4}
    assert {g[1] for g in geometries} == {3, 4, 5, 6}
    assert {g[2] for g in geometries} == {1, 2}
    assert any(g[3] == 0 for g in geometries) and any(g[3] == g[1] - 2 for g in geometries)
    # Whole-puzzle determinism and the full legal count boundary.
    assert _signature(generate_puzzle(991, 2, 1)) == _signature(generate_puzzle(991, 2, 1))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1200 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
    # A genuine finite-copy regression: the same two-root input loses its
    # completion when its required duplicate inventory group is reduced to one.
    a = _inspect(generate_puzzle(73, 2, 1)['train'][1]['input'])
    used = Counter(a['packed']['placements'][i]['group'] for i in a['path'])
    duplicate = next(g for g, n in used.items() if n >= 2)
    saved = a['groups'][duplicate]['count']
    a['groups'][duplicate]['count'] = 1
    missing = _pack(a)
    assert missing['complete'] and not missing['outcomes']
    a['groups'][duplicate]['count'] = saved
    damaged = generate_puzzle(313, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Damaged answer passed validation")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures),
                input_shapes=len(shapes), canonical_geometries=len(geometries),
                normalized_inputs=len(normalized), board_sides=sorted(sides),
                socket_counts=sorted(socket_counts), label_counts=sorted(label_counts),
                surplus_counts=sorted(surplus_counts), repeated_copy_pairs=repeated,
                local_surplus_pairs=local, maximum_search_nodes=largest_search,
                maximum_selected_multiplicity=max_copies)
