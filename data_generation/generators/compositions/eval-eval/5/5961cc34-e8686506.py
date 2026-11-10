"""Assemble ported relays so one signal covers the circuit and reaches its sink.

Only NumPy and the standard library are used. No files, examples, seeds or
construction state are consulted by solve. Search bounds reject uncertainty.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAX_NODES = 12000


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid size")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "Invalid cells")
    return a.astype(np.int8, copy=True)


def _components(cells):
    rest, result = set(cells), []
    while rest:
        start = min(rest)
        rest.remove(start)
        todo, comp = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in rest:
                    rest.remove(p)
                    comp.add(p)
                    todo.append(p)
        result.append(comp)
    return result


def _norm(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shift(cells, r0, c0):
    return {(r + r0, c + c0) for r, c in cells}


def _advance(cell, direction, distance=1):
    dr, dc = DIRS[direction]
    return cell[0] + dr * distance, cell[1] + dc * distance


def _port_options(shape, side, width):
    h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
    options = []
    for start in range(1, (w if side % 2 == 0 else h) - width):
        if side == 0:
            port = {(0, start + k) for k in range(width)}
        elif side == 1:
            port = {(start + k, w - 1) for k in range(width)}
        elif side == 2:
            port = {(h - 1, start + k) for k in range(width)}
        else:
            port = {(start + k, 0) for k in range(width)}
        body = set(shape) - port
        if port <= set(shape) and all(_advance(p, (side + 2) % 4) in body for p in port):
            if len(_components(body)) == 1:
                options.append(tuple(sorted(port)))
    return options


def _piece(shape, port):
    shape, port = tuple(sorted(shape)), tuple(sorted(port))
    h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
    sides = [d for d, ok in enumerate((all(r == 0 for r, c in port),
             all(c == w - 1 for r, c in port), all(r == h - 1 for r, c in port),
             all(c == 0 for r, c in port))) if ok]
    _need(len(sides) == 1 and len(port) in (1, 2), "Port must identify one extreme side")
    side = sides[0]
    _need(port in _port_options(shape, side, len(port)), "Invalid relay port")
    _need(len(shape) > 2 * len(port), "Port cannot dominate its carrier")
    return shape, port, side


def _decode(inp):
    """All role assignments are recovered from the explicit rim and inventory."""
    bg = int(inp[0, 0])
    _need(np.all(inp[0] == bg) and np.all(inp[-1] == bg)
          and np.all(inp[:, 0] == bg) and np.all(inp[:, -1] == bg), "Canvas needs a background border")
    comps = _components(map(tuple, np.argwhere(inp != bg)))
    candidates = [c for c in comps if len({int(inp[p]) for p in c}) >= 3]
    _need(len(candidates) == 1, "Need one circuit board")
    comp = candidates[0]
    r0, r1 = min(r for r, c in comp), max(r for r, c in comp) + 1
    c0, c1 = min(c for r, c in comp), max(c for r, c in comp) + 1
    board = inp[r0:r1, c0:c1].copy()
    h, w = board.shape
    _need(min(h, w) >= 10, "Board is too small")
    edge = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    _need(all(board[p] != bg for p in edge), "Rim must be closed")
    rim = Counter(int(board[p]) for p in edge).most_common(1)[0][0]
    floor = Counter(int(board[r, c]) for r in range(1, h - 1)
                    for c in range(1, w - 1) if board[r, c] != bg).most_common(1)[0][0]
    _need(floor != rim, "Floor and rim differ")
    outside = [c for c in comps if c is not comp]
    # A separated floor island, if any, belongs to the board rather than inventory.
    outside = [c for c in outside if not all(r0 <= r < r1 and c0 <= c1_ < c1 for r, c1_ in c)]
    _need(2 <= len(outside) <= 5, "Need two through five loose relays")
    pieces, carrier_colors, port_colors = [], set(), set()
    for cells in outside:
        _need(all(not (r0 <= r < r1 and c0 <= c < c1) for r, c in cells), "Piece crosses board")
        counts = Counter(int(inp[p]) for p in cells)
        _need(len(counts) == 2, "Loose relays have two colors")
        body, pc = sorted(counts, key=lambda x: -counts[x])
        _need(counts[body] > counts[pc], "Carrier is the majority color")
        rr, cc = min(r for r, c in cells), min(c for r, c in cells)
        shape = _norm(cells)
        port = tuple(sorted((r - rr, c - cc) for r, c in cells if int(inp[r, c]) == pc))
        pieces.append(_piece(shape, port))
        carrier_colors.add(body)
        port_colors.add(pc)
    _need(len(carrier_colors) == len(port_colors) == 1, "Pair-local relay palette must agree")
    carrier, pc = next(iter(carrier_colors)), next(iter(port_colors))
    launchers = []
    for start in sorted(edge):
        r, c = start
        if r in (0, h - 1) and c in (0, w - 1):
            continue
        d = 2 if r == 0 else 0 if r == h - 1 else 1 if c == 0 else 3
        a, b = _advance(start, d), _advance(start, d, 2)
        sig = int(board[start])
        if sig not in (bg, rim, floor, carrier, pc) and board[a] == sig and board[b] == pc:
            launchers.append((start, a, b, d, sig))
    _need(len(launchers) == 1, "Need a unique tail-tail-cap launcher")
    start, middle, cap, initial, signal = launchers[0]
    target = {p for p in edge if int(board[p]) not in (rim, signal)}
    _need(len(target) in (1, 2) and len({int(board[p]) for p in target}) == 1, "Need one receiver strip")
    target_color = int(board[min(target)])
    _need(len({bg, rim, floor, carrier, pc, signal, target_color}) == 7, "Semantic colors must differ")
    _need(len(_components(target)) == 1, "Receiver must be contiguous")
    target_sides = [d for d, ok in enumerate((all(r == 0 for r, c in target),
                    all(c == w - 1 for r, c in target), all(r == h - 1 for r, c in target),
                    all(c == 0 for r, c in target))) if ok]
    _need(len(target_sides) == 1 and not any(r in (0, h - 1) and c in (0, w - 1) for r, c in target), "Receiver avoids corners")
    special = {start, middle, cap} | target
    _need(all(int(board[r, c]) == (rim if (r, c) in edge else floor)
              for r in range(h) for c in range(w) if board[r, c] != bg and (r, c) not in special), "Unexpected fixed markings")
    holes = set(map(tuple, np.argwhere(board == bg)))
    _need(sum(len(p[0]) for p in pieces) == len(holes), "Inventory area differs from missing area")
    _need(all(1 < r < h - 2 and 1 < c < w - 2 for r, c in holes), "Sockets need an interior margin")
    sockets = _components(holes)
    _need(len(sockets) == len(pieces), "Each separated socket holds one relay")
    for i, socket in enumerate(sockets):
        other = holes - socket
        _need(not any((r + dr, c + dc) in other for r, c in socket
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1)), "Sockets need an eight-neighbor gap")
    return dict(board=board, bg=bg, rim=rim, floor=floor, carrier=carrier, pc=pc,
                signal=signal, target_color=target_color, target=target, initial=initial,
                tail={start, middle}, cap=cap, pieces=pieces, holes=holes, bbox=(r0, c0, h, w))


def _route(h, w, tail, cap, direction, objects, target=None):
    """Advance a whole beam front; an incomplete or cyclic circuit is invalid."""
    owner = {}
    for i, (cells, port, side) in enumerate(objects):
        for p in cells:
            if p in owner:
                return None
            owner[p] = i
    trace, reached, widths, transitions, gaps = set(tail) | {cap}, [], [], [], []
    front, gap = {_advance(cap, direction)}, 0
    for _ in range(h * w * (len(objects) + 1)):
        if any(not (0 <= r < h and 0 <= c < w) for r, c in front):
            return None
        boundary = {p for p in front if p[0] in (0, h - 1) or p[1] in (0, w - 1)}
        if boundary:
            if boundary != front or len(reached) != len(objects):
                return None
            if target is not None and front != target:
                return None
            trace |= front
            return dict(trace=trace, reached=tuple(reached), widths=tuple(widths),
                        transitions=tuple(transitions), gaps=tuple(gaps), exit=front)
        hits = {owner[p] for p in front if p in owner}
        if hits:
            if len(hits) != 1:
                return None
            i = next(iter(hits))
            if i in reached or any(owner.get(p) != i for p in front):
                return None
            cells, port, side = objects[i]
            if side == (direction + 2) % 4:
                return None
            reached.append(i)
            gaps.append(gap)
            widths.append(len(port))
            transitions.append((direction, side))
            trace |= set(cells)
            direction, front, gap = side, {_advance(p, side) for p in port}, 0
        else:
            trace |= front
            front = {_advance(p, direction) for p in front}
            gap += 1
    return None


def _assemble(info):
    """Exact inventory cover and routing are checked on the same placements.

    Retain at most two valid marked assemblies. Identical pieces are counted,
    avoiding factorial duplicate permutations. No unbounded memoization/cache.
    """
    holes = sorted(info['holes'])
    index = {p: i for i, p in enumerate(holes)}
    counts = Counter(info['pieces'])
    types = sorted(counts)
    by_cell = [[] for _ in holes]
    h, w = info['board'].shape
    for t, (shape, port, side) in enumerate(types):
        sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                cells = _shift(shape, r0, c0)
                if not cells <= info['holes']:
                    continue
                mask = sum(1 << index[p] for p in cells)
                obj = (frozenset(cells), frozenset(_shift(port, r0, c0)), side)
                move = t, mask, obj
                for p in cells:
                    by_cell[index[p]].append(move)
    solutions, nodes = [], 0
    geometric_answers = 0

    def visit(remaining, inventory, objects):
        nonlocal nodes, geometric_answers
        nodes += 1
        _need(nodes <= MAX_NODES, "Assembly search bound exceeded")
        if not remaining:
            geometric_answers += 1
            route = _route(h, w, info['tail'], info['cap'], info['initial'], objects, info['target'])
            if route is not None:
                marking = tuple(sorted((p, p in port) for cells, port, side in objects for p in cells))
                if not any(s[0] == marking for s in solutions):
                    solutions.append((marking, tuple(objects), route))
            return len(solutions) >= 2
        best = None
        for i in range(len(holes)):
            if remaining & (1 << i):
                choices = [m for m in by_cell[i] if inventory[m[0]] and m[1] & remaining == m[1]]
                if not choices:
                    return False
                if best is None or len(choices) < len(best):
                    best = choices
                    if len(best) == 1:
                        break
        for t, mask, obj in best:
            inventory[t] -= 1
            if visit(remaining ^ mask, inventory, objects + [obj]):
                inventory[t] += 1
                return True
            inventory[t] += 1
        return False

    visit((1 << len(holes)) - 1, [counts[t] for t in types], [])
    _need(len(solutions) == 1, "Circuit must have one valid marked assembly")
    _, objects, route = solutions[0]
    out = info['board'].copy()
    for p in route['trace']:
        out[p] = info['signal']
    _need(not np.any(out == info['bg']), "Every socket must be activated")
    route.update(nodes=nodes, geometry_count=geometric_answers, objects=objects)
    return out, route


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique repaired circuit's trace using only the input."""
    return _assemble(_decode(_grid(input_grid)))[0]


def _random_shape(rng, small=False):
    h, w = map(int, rng.integers(3, 5 if small else 6, size=2))
    shape = {(r, c) for r in range(h) for c in range(w)}
    kind = int(rng.integers(5))
    if kind == 1:
        shape -= {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    elif kind == 2:
        shape.remove((int(rng.choice([0, h - 1])), int(rng.choice([0, w - 1]))))
    elif kind == 3 and min(h, w) >= 4:
        shape.remove((0, int(rng.integers(1, w - 1))))
    elif kind == 4 and h == w == 5:
        shape = {(r, c) for r, c in shape if min(r, h - 1 - r) + min(c, w - 1 - c) >= 2}
    return tuple(sorted(shape))


def _start(h, w, direction, rng):
    line = int(rng.integers(3, (w if direction % 2 == 0 else h) - 3))
    start = (h - 1, line) if direction == 0 else (line, 0) if direction == 1 else (0, line) if direction == 2 else (line, w - 1)
    return {start, _advance(start, direction)}, _advance(start, direction, 2)


def _layout(rng, n):
    h, w = map(int, rng.integers(16 if n <= 3 else 20, 24, size=2))
    initial = int(rng.integers(4))
    tail, cap = _start(h, w, initial, rng)
    common = _random_shape(rng, small=n >= 4)
    shapes = [common, common] + [_random_shape(rng, small=n >= 4) for _ in range(n - 2)]
    rng.shuffle(shapes)
    budget = [350]

    def grow(objects, occupied, traversed, origins, incoming):
        if len(objects) == n:
            route = _route(h, w, tail, cap, initial, objects)
            if route is not None and {1, 2} <= set(route['widths']) and any(a != b for a, b in route['transitions']):
                return objects, route
            return None
        shape = shapes[len(objects)]
        oh, ow = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        choices = []
        for side in ((incoming - 1) % 4, incoming, (incoming + 1) % 4):
            for width in (1, 2):
                for port in _port_options(shape, side, width):
                    choices.append((side, port))
        rng.shuffle(choices)
        for side, port in choices:
            entry_side = (incoming + 2) % 4
            # Alignment need not use an output port on the entry face.
            edge = sorted(c if entry_side % 2 == 0 else r for r, c in shape
                          if (r == 0 if entry_side == 0 else c == ow - 1 if entry_side == 1 else r == oh - 1 if entry_side == 2 else c == 0))
            starts = [v for v in edge if all(v + k in edge for k in range(len(origins)))]
            placements = [(gap, entry) for gap in range(1, 12) for entry in starts]
            rng.shuffle(placements)
            for gap, entry in placements[:24]:
                if budget[0] <= 0:
                    return None
                budget[0] -= 1
                base = min(origins)
                point = _advance(base, incoming, gap + 1)
                if incoming == 0:
                    rr, cc = point[0] - oh + 1, point[1] - entry
                elif incoming == 1:
                    rr, cc = point[0] - entry, point[1]
                elif incoming == 2:
                    rr, cc = point[0], point[1] - entry
                else:
                    rr, cc = point[0] - entry, point[1] - ow + 1
                if rr < 2 or cc < 2 or rr + oh > h - 2 or cc + ow > w - 2:
                    continue
                cells = _shift(shape, rr, cc)
                if cells & traversed or any((r + dr, c + dc) in occupied for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                    continue
                approach = {_advance(p, incoming, k) for p in origins for k in range(1, gap + 1)}
                if approach & occupied:
                    continue
                placed = _shift(port, rr, cc)
                result = grow(objects + [(cells, placed, side)], occupied | cells,
                              traversed | approach, sorted(placed), side)
                if result is not None:
                    return result
        return None

    result = grow([], tail | {cap}, tail | {cap}, [cap], initial)
    if result is None:
        return None
    objects, route = result
    return h, w, tail, cap, initial, objects, route


def _fallback(rng, n):
    """A small monotone staircase, still checked by joint assembly inference."""
    h, w = 20 + int(rng.integers(3)), 21 + int(rng.integers(3))
    positions = [(2, 4), (7, 4), (7, 9), (12, 9), (12, 14)][:n]
    tail, cap, objects = {(3, 0), (3, 1)}, (3, 2), []
    for i, (r0, c0) in enumerate(positions):
        sh, sw = (3, 4) if i == n - 1 and n >= 3 else (4, 4)
        mask = {(r, c) for r in range(sh) for c in range(sw)}
        if rng.random() < .5 and i >= 2:
            mask.remove((sh - 1, 0))
        side = 2 if i % 2 == 0 else 1
        port = {(sh - 1, 1), (sh - 1, 2)} if side == 2 else {(1, sw - 1)}
        objects.append((_shift(mask, r0, c0), _shift(port, r0, c0), side))
    route = _route(h, w, tail, cap, 1, objects)
    _need(route is not None, "Fallback route failed")
    return h, w, tail, cap, 1, objects, route


def _scatter(rng, layout, palette, rotation):
    h, w, tail, cap, initial, objects, route = layout
    bg, rim, floor, carrier, pc, signal, receiver = map(int, palette[:7])
    board = np.full((h, w), floor, dtype=np.int8)
    board[0, :] = board[-1, :] = rim
    board[:, 0] = board[:, -1] = rim
    for cells, port, side in objects:
        for p in cells:
            board[p] = bg
    for p in tail:
        board[p] = signal
    board[cap] = pc
    for p in route['exit']:
        board[p] = receiver
    out = board.copy()
    for p in route['trace']:
        out[p] = signal
    pieces = []
    for cells, port, side in objects:
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        pieces.append((_norm(cells), tuple(sorted((r - r0, c - c0) for r, c in port))))
    # An ample top/bottom or side band fits even five 4x4 pieces. Locations
    # within the free space are sampled; there is no inventory order clue.
    for attempt in range(16):
        ch = int(rng.integers(max(h + 6, 25), 31))
        cw = int(rng.integers(max(w + 2, 25), 31))
        br = 1 if attempt % 2 == 0 else ch - h - 1
        bc = int(rng.integers(1, cw - w))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[br:br + h, bc:bc + w] = board
        blocked = {(r, c) for r in range(br - 1, br + h + 1) for c in range(bc - 1, bc + w + 1)}
        order = list(range(len(pieces)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(pieces[i][0]))
        for i in order:
            shape, port = pieces[i]
            ph, pw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
            options = [(r, c) for r in range(1, ch - ph) for c in range(1, cw - pw)
                       if not (_shift(shape, r, c) & blocked)]
            if not options:
                break
            rr, cc = options[int(rng.integers(len(options)))]
            for dr, dc in shape:
                r, c = rr + dr, cc + dc
                canvas[r, c] = pc if (dr, dc) in port else carrier
                blocked.update((r + ar, c + ac) for ar in (-1, 0, 1) for ac in (-1, 0, 1))
        else:
            return {'input': np.rot90(canvas, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    # Guaranteed for the constructive fallback: a single row retains each
    # piece's orientation and keeps one blank cell between bounding boxes.
    ph = max(max(r for r, c in shape) + 1 for shape, port in pieces)
    total_width = sum(max(c for r, c in shape) + 1 for shape, port in pieces) + len(pieces) - 1
    if h + ph <= 27 and total_width <= 28:
        canvas = np.full((30, 30), bg, dtype=np.int8)
        canvas[1:h + 1, 1:w + 1] = board
        rr, cc = h + 2, 1
        for shape, port in pieces:
            for dr, dc in shape:
                canvas[rr + dr, cc + dc] = pc if (dr, dc) in port else carrier
            cc += max(c for r, c in shape) + 2
        return {'input': np.rot90(canvas, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} through {high}')


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _count(num_train, 'num_train', 2, 10)
    _count(num_test, 'num_test', 1, 4)
    _need(num_train > num_test, 'num_train must exceed num_test')
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette, vary_palette = rng.permutation(10), bool(rng.integers(2))
    orientation = int(rng.integers(4))
    vary_orientation = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            n = 3 if split == 'train' and i < 2 else int(rng.integers(2, 4)) if split == 'train' else int(rng.integers(4, 6))
            colors = rng.permutation(10) if vary_palette else palette
            rotation = int(rng.integers(4)) if vary_orientation else orientation
            for attempt in range(36):
                layout = _layout(rng, n) if attempt < 28 else _fallback(rng, n)
                if layout is None:
                    continue
                # Repeated silhouettes must have visibly different port marks.
                signature_ports = {}
                for cells, port, side in layout[5]:
                    rr, cc = min(r for r, c in cells), min(c for r, c in cells)
                    signature_ports.setdefault(_norm(cells), set()).add(tuple(sorted((r - rr, c - cc) for r, c in port)))
                if not any(len(ps) > 1 for ps in signature_ports.values()):
                    continue
                pair = _scatter(rng, layout, colors, rotation)
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    info = _decode(pair['input'])
                    predicted, facts = _assemble(info)
                except ValueError:
                    continue
                if not np.array_equal(predicted, pair['output']) or facts['geometry_count'] < 2:
                    continue
                break
            else:
                raise RuntimeError('Bounded construction could not produce a unique circuit')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    order = rng.permutation(num_train)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _count(len(puzzle['train']), 'num_train', 2, 10)
    _count(len(puzzle['test']), 'num_test', 1, 4)
    _need(len(puzzle['train']) > len(puzzle['test']), 'Invalid split counts')
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            predicted, route = _assemble(info)
            _need(np.array_equal(predicted, out), 'Output differs from joint solution')
            _need(route['geometry_count'] >= 2, 'Shape-only fitting must be ambiguous')
            _need(len(route['reached']) == len(info['pieces']), 'Every inventory piece must activate')
            _need(set(map(tuple, np.argwhere(out == info['signal']))) == route['trace'], 'Signal footprint differs')
            fixed = ~np.isin(info['board'], [info['bg'], info['signal'], info['pc'], info['target_color']])
            untouched = fixed.copy()
            for p in route['trace']:
                untouched[p] = False
            _need(np.array_equal(out[untouched], info['board'][untouched]), 'Untraversed fixed cells changed')
            _need({1, 2} <= set(route['widths']), 'Both output widths must be exposed')
            _need(any(a != b for a, b in route['transitions']), 'A turn must be exposed')
            # Without assembling the loose relays, tracing on the empty board
            # gives a straight line, even when that line happens to hit the sink.
            direct = _route(*info['board'].shape, info['tail'], info['cap'], info['initial'], [])
            _need(direct is not None and direct['trace'] != route['trace'],
                  'Omitting assembly must change the signal footprint')
            facts[split].append((info, route))
    _need(any(len(info['pieces']) == 3 for info, route in facts['train']), 'Need a multi-relay demonstration')
    _need(min(len(info['pieces']) for info, route in facts['test']) > max(len(info['pieces']) for info, route in facts['train']), 'Tests extend the demonstrated chain')


def self_test():
    """Streaming 42-seed check: no saved grids, persistent caches, or traces."""
    shapes, board_shapes, footprints, widths, directions, counts, palettes = set(), set(), set(), set(), set(), set(), set()
    digests, nodes, geometry, all_sizes = set(), [], [], set()
    within_layout_changes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        local = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _decode(inp)
                before = inp.copy()
                predicted, f = _assemble(info)
                assert np.array_equal(predicted, out) and np.array_equal(solve(inp), out)
                assert np.array_equal(inp, before)
                digest.update(str(inp.shape).encode() + inp.tobytes() + out.tobytes())
                shapes.add(inp.shape)
                board_shapes.add(out.shape)
                footprints.update(p[0] for p in info['pieces'])
                all_sizes.update(len(p[0]) for p in info['pieces'])
                widths.update(f['widths'])
                directions.add(info['initial'])
                counts.add(len(info['pieces']))
                palettes.add((info['bg'], info['rim'], info['floor'], info['signal']))
                local.add((out.shape, tuple(sorted(_norm(c) for c in _components(info['holes'])))))
                nodes.append(f['nodes'])
                geometry.append(f['geometry_count'])
        digests.add(digest.hexdigest())
        within_layout_changes += len(local) >= 3
    assert len(digests) == 42 and len(shapes) >= 20 and len(board_shapes) >= 20
    assert len(footprints) >= 20 and len(all_sizes) >= 8 and len(palettes) >= 30
    assert widths == {1, 2} and directions == set(range(4)) and counts == {2, 3, 4, 5}
    assert within_layout_changes == 42 and min(geometry) >= 2
    fallback_rng = np.random.default_rng(713)
    for n in range(2, 6):
        for rotation in range(4):
            pair = _scatter(fallback_rng, _fallback(fallback_rng, n), range(10), rotation)
            assert pair is not None and np.array_equal(solve(pair['input']), pair['output'])
    # Counts are deliberately explicit: the function's defaults are 4/1.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(101, a, b)
            assert len(p['train']) == a and len(p['test']) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    p = generate_puzzle(314159, 2, 1)
    generate_puzzle(88)
    q = generate_puzzle(314159, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Corrupted trace and deleted inventory both must fail validation.
    for mutation in ('output', 'inventory'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        if mutation == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        else:
            info = _decode(pair['input'])
            r0, c0, h, w = info['bbox']
            for r, c in np.argwhere(pair['input'] == info['carrier']):
                if not (r0 <= r < r0 + h and c0 <= c < c0 + w):
                    pair['input'][r, c] = info['bg']
                    break
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid puzzle accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(digests), canvas_shapes=len(shapes),
                board_shapes=len(board_shapes), oriented_footprints=len(footprints),
                footprint_areas=sorted(all_sizes), relay_counts=sorted(counts), directions=sorted(directions),
                max_search_nodes=max(nodes), min_shape_only_assemblies=min(geometry),
                max_shape_only_assemblies=max(geometry), within_puzzle_layout_variation=within_layout_changes)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
