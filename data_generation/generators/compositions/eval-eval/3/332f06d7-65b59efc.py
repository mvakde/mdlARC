"""Clearance atlas: compete square carriers, then print their keyed motifs.

The public solver recovers everything from pixels. Generation uses weighted
room graphs; solving checks actual k-by-k pixel placements instead. No files,
parent generators, cached examples, or random state are used by solve().
"""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
from itertools import combinations
import json
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=False)


def _neighbors(p, shape):
    r, c = p
    for q in ((r-1, c), (r, c-1), (r, c+1), (r+1, c)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _edge(a, b):
    return tuple(sorted((a, b)))


def _gate_slice(a, b, n):
    if a[0] == b[0]:
        r = 1 + a[0]*(n+2)
        c = 1 + min(a[1], b[1])*(n+2) + n
        return slice(r, r+n), slice(c, c+2)
    r = 1 + min(a[0], b[0])*(n+2) + n
    c = 1 + a[1]*(n+2)
    return slice(r, r+2), slice(c, c+n)


def _paint_map(n, shape, active, edges, source, roles):
    wall, floor, depot, _ = roles
    result = np.full((shape[0]*(n+2), shape[1]*(n+2)), wall, dtype=np.int8)
    for r, c in active:
        top, left = 1+r*(n+2), 1+c*(n+2)
        result[top:top+n, left:left+n] = depot if (r, c) == source else floor
    for (a, b), (width, offset) in edges.items():
        rr, cc = _gate_slice(a, b, n)
        if a[0] == b[0]:
            rr = slice(rr.start+offset, rr.start+offset+width)
        else:
            cc = slice(cc.start+offset, cc.start+offset+width)
        result[rr, cc] = floor
    return result


def _capacities(active, edges, source, n, shortest=False):
    """Graph construction oracle: maximum bottleneck over all routes."""
    adjacency = {p: [] for p in active}
    for (a, b), (width, _) in edges.items():
        adjacency[a].append((b, width))
        adjacency[b].append((a, width))
    caps = dict.fromkeys(active, 0)
    caps[source] = n
    if shortest:
        queue = [source]
        for p in queue:
            for q, width in sorted(adjacency[p]):
                if caps[q] == 0:
                    caps[q] = min(caps[p], width)
                    queue.append(q)
        return caps
    queue = [(-n, source)]
    while queue:
        negative, p = heapq.heappop(queue)
        if -negative != caps[p]:
            continue
        for q, width in adjacency[p]:
            candidate = min(-negative, width)
            if candidate > caps[q]:
                caps[q] = candidate
                heapq.heappush(queue, (-candidate, q))
    return caps


def _parse(input_grid):
    g = _grid(input_grid)
    separator = int(g[0, 0])
    full = np.flatnonzero(np.all(g == separator, axis=1)).tolist()
    _require(len(full) == 3 and full[0] == 0, "three library dividers required")
    n = full[1]-1
    _require(3 <= n <= 5 and full[2] == 2*n+3, "library scale")
    library_height = 2*n+4
    _require(g.shape[0] > library_height, "missing chamber map")
    board = g[library_height:]
    wall = int(board[0, 0])
    _require(wall != separator, "wall equals divider")
    gutters = np.flatnonzero(g[1] == separator).tolist()
    count = len(gutters)-1
    _require(2 <= count <= min(4, n), "library count")
    _require(gutters == [j*(n+1) for j in range(count+1)], "library gutters")
    library_width = count*(n+1)+1
    expected_library = np.full((library_height, g.shape[1]), wall, dtype=np.int8)
    expected_library[[0, n+1, 2*n+3]] = separator
    for c in gutters:
        expected_library[:, c] = separator
    templates, upper = {}, []
    for j in range(count):
        c = 1+j*(n+1)
        panel = g[1:n+1, c:c+n]
        colors = np.unique(panel[panel != wall])
        _require(len(colors) == 1, "template must have one nonwall identity")
        label = int(colors[0])
        _require(label != separator and label not in templates, "template identities")
        templates[label] = panel == label
        upper.append(label)
        expected_library[1:n+1, c:c+n] = panel
    carriers, lower = {}, []
    for j in range(count):
        c = 1+j*(n+1)
        tag_row = g[n+2, c:c+n]
        label = int(tag_row[n//2])
        _require(label in templates and label not in carriers, "gauge identity tag")
        expected_library[n+2, c+n//2] = label
        swatch = g[n+3:2*n+3, c:c+n]
        cells = np.argwhere(swatch != wall)
        _require(len(cells) > 0, "missing size/color swatch")
        size = int(cells[:, 0].max())+1
        target = int(swatch[0, 0])
        expected = np.full((n, n), wall, dtype=np.int8)
        expected[:size, :size] = target
        _require(target not in (wall, separator) and np.array_equal(swatch, expected),
                 "gauge must be a top-left solid square")
        carriers[label] = (size, target)
        lower.append(label)
        expected_library[n+3:2*n+3, c:c+n] = expected
    _require(np.array_equal(g[:library_height], expected_library), "extra library material")
    _require(len({v[0] for v in carriers.values()}) == count, "carrier sizes must be distinct")
    _require(board.shape[0] % (n+2) == 0, "map row registration")
    rows = board.shape[0]//(n+2)
    nonwall = np.argwhere(board != wall)
    _require(len(nonwall) > 0, "empty map")
    cols = (int(nonwall[:, 1].max())+3)//(n+2)
    _require(1 <= cols and cols*(n+2) <= board.shape[1], "map column registration")
    active, room_colors = set(), {}
    for r in range(rows):
        for c in range(cols):
            top, left = 1+r*(n+2), 1+c*(n+2)
            panel = board[top:top+n, left:left+n]
            _require(panel.shape == (n, n) and len(np.unique(panel)) == 1, "room must be solid")
            color = int(panel[0, 0])
            if color != wall:
                active.add((r, c))
                room_colors[(r, c)] = color
    _require(len(active) >= 3, "at least three rooms required")
    colors, counts = np.unique(board[board != wall], return_counts=True)
    _require(len(colors) == 2, "map must have floor and depot roles")
    index = int(np.argmin(counts))
    depot, floor = int(colors[index]), int(colors[1-index])
    _require(int(counts[index]) == n*n and int(counts[1-index]) > n*n, "unique depot room")
    sources = [p for p in active if room_colors[p] == depot]
    _require(len(sources) == 1 and all(v in (floor, depot) for v in room_colors.values()), "room colors")
    source = sources[0]
    _require(len({wall, floor, depot, separator}) == 4, "structural roles must differ")
    _require(not (set(upper) | {v[1] for v in carriers.values()}) & {wall, floor, depot, separator},
             "stamp colors must differ from structural roles")
    _require({r for r, _ in active} >= {0, rows-1} and {c for _, c in active} >= {0, cols-1},
             "empty outer lattice rows/columns")
    edges = {}
    for a in sorted(active):
        for b in ((a[0]+1, a[1]), (a[0], a[1]+1)):
            if b not in active:
                continue
            strip = board[_gate_slice(a, b, n)]
            if np.all(strip == wall):
                continue
            gate = strip == floor
            profile = gate.all(axis=1 if a[0] == b[0] else 0)
            occupied = np.flatnonzero(profile)
            _require(len(occupied) > 0 and np.all(np.diff(occupied) == 1), "rectangular gate")
            edges[_edge(a, b)] = (len(occupied), int(occupied[0]))
    roles = (wall, floor, depot, separator)
    recovered = _paint_map(n, (rows, cols), active, edges, source, roles)
    _require(np.array_equal(board[:, :cols*(n+2)], recovered), "map has malformed gates or stray cells")
    _require(np.all(board[:, cols*(n+2):] == wall), "trailing map material")
    _require(all(v > 0 for v in _capacities(active, edges, source, n).values()), "disconnected rooms")
    return dict(n=n, count=count, wall=wall, roles=roles, templates=templates,
                carriers=carriers, upper=upper, lower=lower, board=board,
                shape=(rows, cols), active=active, edges=edges, source=source)


def _reachable_placements(board, wall, size, start):
    allowed = board != wall
    valid = np.lib.stride_tricks.sliding_window_view(allowed, (size, size)).all(axis=(-1, -2))
    _require(bool(valid[start]), "invalid starting footprint")
    visited = np.zeros(valid.shape, dtype=bool)
    visited[start] = True
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for q in _neighbors(p, valid.shape):
            if valid[q] and not visited[q]:
                visited[q] = True
                queue.append(q)
    return visited


def _pixel_winners(parsed):
    n, source = parsed['n'], parsed['source']
    start = (1+source[0]*(n+2), 1+source[1]*(n+2))
    winners = dict.fromkeys(parsed['active'], None)
    for label, (size, _) in sorted(parsed['carriers'].items(), key=lambda x: x[1][0]):
        reach = _reachable_placements(parsed['board'], parsed['wall'], size, start)
        for r, c in parsed['active']:
            if reach[1+r*(n+2), 1+c*(n+2)]:
                winners[(r, c)] = label
    return winners


def _graph_winners(parsed, *, shortest=False, ignore_clearance=False):
    caps = _capacities(parsed['active'], parsed['edges'], parsed['source'], parsed['n'], shortest)
    sizes = parsed['carriers']
    winners = {}
    for p, cap in caps.items():
        choices = [label for label, (size, _) in sizes.items() if ignore_clearance or size <= cap]
        winners[p] = max(choices, key=lambda x: sizes[x][0]) if choices else None
    return winners


def _transform(mask, code):
    return np.rot90(np.fliplr(mask) if code >= 4 else mask, code % 4)


def _render(parsed, winners, *, column=False, cyclic=False, solid=False, unkeyed=False, transform=0):
    n = parsed['n']
    result = np.full((parsed['shape'][0]*n, parsed['shape'][1]*n), parsed['wall'], dtype=np.int8)
    correspondence = dict(zip(parsed['lower'], parsed['upper'])) if column else dict(zip(parsed['upper'], parsed['upper']))
    if cyclic:
        shift = parsed['upper'].index(parsed['lower'][0])
        correspondence = {label: parsed['upper'][(shift+j) % len(parsed['upper'])]
                          for j, label in enumerate(parsed['lower'])}
    for (r, c), label in winners.items():
        if label is None:
            continue
        motif_label = correspondence[label]
        mask = np.ones((n, n), dtype=bool) if solid else _transform(parsed['templates'][motif_label], transform)
        color = motif_label if unkeyed else parsed['carriers'][label][1]
        result[r*n:(r+1)*n, c*n:(c+1)*n] = np.where(mask, color, parsed['wall'])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return an atlas from only the library, square swatches, and visible map."""
    parsed = _parse(input_grid)
    return _render(parsed, _pixel_winners(parsed))


def _motif(rng, n):
    mask = np.zeros((n, n), dtype=bool)
    style = int(rng.integers(8))
    if style == 0:
        mask[:] = rng.random((n, n)) < rng.uniform(.18, .85)
    elif style == 1:
        mask[[0, -1]] = True
        mask[:, [0, -1]] = True
    elif style == 2:
        mask[int(rng.integers(n)), :] = True
        mask[:, int(rng.integers(n))] = True
    elif style == 3:
        mask = np.indices((n, n)).sum(axis=0) % 2 == int(rng.integers(2))
    elif style == 4:
        mask[:, rng.choice(n, int(rng.integers(1, n)), replace=False)] = True
    elif style == 5:
        r, c = map(int, rng.integers(n, size=2))
        mask[r:, c:] = True
    elif style == 6:
        np.fill_diagonal(mask, True)
        if rng.integers(2):
            mask |= np.fliplr(np.eye(n, dtype=bool))
    else:
        mask.flat[rng.choice(n*n, int(rng.integers(1, n*n+1)), replace=False)] = True
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    return _transform(mask, int(rng.integers(8))).copy()


def _random_network(rng, shape, minimum, maximum, loops):
    rows, cols = shape
    whole = {(r, c) for r in range(rows) for c in range(cols)}
    target = int(rng.integers(minimum, min(maximum, len(whole))+1))
    active = whole.copy()
    # Remove cells while retaining connectivity and the visible bounding box.
    for point in [tuple(map(int, p)) for p in rng.permutation(sorted(whole))]:
        if len(active) <= target:
            break
        candidate = active-{point}
        if not {r for r, _ in candidate} >= {0, rows-1} or not {c for _, c in candidate} >= {0, cols-1}:
            continue
        start = min(candidate)
        seen, queue = {start}, [start]
        for p in queue:
            for q in _neighbors(p, shape):
                if q in candidate and q not in seen:
                    seen.add(q)
                    queue.append(q)
        if seen == candidate:
            active = candidate
    source = sorted(active)[int(rng.integers(len(active)))]
    seen, frontier, edges = {source}, [], set()
    frontier.extend((source, q) for q in _neighbors(source, shape) if q in active)
    while frontier:
        i = int(rng.integers(len(frontier)))
        a, b = frontier.pop(i)
        if b in seen:
            continue
        edges.add(_edge(a, b))
        seen.add(b)
        frontier.extend((b, q) for q in _neighbors(b, shape) if q in active and q not in seen)
    for a in sorted(active):
        for b in _neighbors(a, shape):
            if b in active and a < b and rng.random() < loops:
                edges.add(_edge(a, b))
    return active, edges, source


def _network(rng, n, kind, sizes):
    if kind == 'detour':
        a, b, c, d, e, f = (0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)
        low, middle, high = sizes
        widths = {_edge(a, b): int(rng.integers(1, middle)), _edge(a, c): high,
                  _edge(c, d): middle, _edge(d, b): middle,
                  _edge(b, e): low, _edge(e, f): high}
        shape, active, source = (2, 3), {a, b, c, d, e, f}, a
        # Change the physical presentation of this argument without turning stamps.
        flip_r, flip_c, transpose = map(bool, rng.integers(2, size=3))
        def move(p):
            r, c = (1-p[0] if flip_r else p[0]), (2-p[1] if flip_c else p[1])
            return (c, r) if transpose else (r, c)
        active, source = {move(p) for p in active}, move(source)
        widths = {_edge(move(p), move(q)): w for (p, q), w in widths.items()}
        if transpose:
            shape = shape[::-1]
    else:
        if kind == 'tree':
            shape, minimum, maximum, loops = (2, 3), 6, 6, 0.0
        elif kind == 'test':
            rows = int(rng.choice([3, 4])) if n == 3 else 3 if n == 4 else 2
            cols = int(rng.integers(3, 30//(n+2)+1)) if n < 5 else 4
            shape, minimum, maximum, loops = (rows, cols), 7, rows*cols, float(rng.uniform(.15, .8))
        else:
            shape = (int(rng.integers(1, 3)), int(rng.integers(3, 5)))
            if shape[0]*shape[1] > 6:
                shape = (2, 3)
            minimum, maximum, loops = 3, 6, float(rng.uniform(0, .65))
        active, bare, source = _random_network(rng, shape, minimum, maximum, loops)
        widths = {edge: int(rng.integers(1, n+1)) for edge in sorted(bare)}
    edges = {edge: (w, int(rng.integers(n-w+1))) for edge, w in sorted(widths.items())}
    return shape, active, edges, source


def _encode(n, roles, templates, carriers, upper, lower, shape, active, edges, source):
    wall, _, _, separator = roles
    board = _paint_map(n, shape, active, edges, source, roles)
    width = max(len(upper)*(n+1)+1, board.shape[1])
    lib_height = 2*n+4
    result = np.full((lib_height+board.shape[0], width), wall, dtype=np.int8)
    result[[0, n+1, 2*n+3]] = separator
    for j in range(len(upper)+1):
        result[:lib_height, j*(n+1)] = separator
    for j, label in enumerate(upper):
        c = 1+j*(n+1)
        result[1:n+1, c:c+n] = np.where(templates[label], label, wall)
    for j, label in enumerate(lower):
        c = 1+j*(n+1)
        size, target = carriers[label]
        result[n+2, c+n//2] = label
        result[n+3:n+3+size, c:c+size] = target
    result[lib_height:, :board.shape[1]] = board
    return result


def _pair(rng, roles, n, kind, used):
    pool = [c for c in range(10) if c not in roles]
    sizes = [2, 3, 4] if kind == 'tree' else [1, 2, 3] if kind == 'detour' else sorted(
        map(int, rng.choice(np.arange(1, n+1), int(rng.integers(2, min(4, n)+1)), replace=False)))
    for attempt in range(192):
        shape, active, edges, source = _network(rng, n, kind, sizes)
        if attempt >= 96 and kind in ('tree', 'test'):
            # Guaranteed capacity variation after bounded random rejection.
            # The fallback is a snake on the sampled bounding rectangle.
            path = [(r, c) for r in range(shape[0])
                    for c in (range(shape[1]) if r % 2 == 0 else range(shape[1]-1, -1, -1))]
            active, source = set(path), path[0]
            widths = [4, 3, 2, 1, 4] if kind == 'tree' else [1]+[n]*(len(path)-2)
            edges = {_edge(a, b): (w, int(rng.integers(n-w+1)))
                     for a, b, w in zip(path, path[1:], widths)}
        labels = list(map(int, rng.choice(pool, len(sizes), replace=False)))
        targets = list(map(int, rng.choice(pool, len(sizes), replace=kind not in ('tree', 'detour'))))
        if kind in ('tree', 'detour') and sum(a != b for a, b in zip(labels, targets)) < 2:
            targets = labels[1:]+labels[:1]
        templates = {label: _motif(rng, n) for label in labels}
        if kind == 'detour':
            # No fixed witness silhouette. A random asymmetric motif rules out
            # independently turning the stamp when the physical route turns.
            templates[labels[0]] = rng.random((n, n)) < .48
            if attempt >= 96:
                templates = {labels[0]: np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool),
                             labels[1]: np.eye(3, dtype=bool),
                             labels[2]: np.ones((3, 3), dtype=bool)}
        elif rng.random() < .2:
            templates[labels[-1]] = templates[labels[0]].copy()
        if any(not m.any() for m in templates.values()):
            continue
        carriers = {label: (size, target) for label, size, target in zip(labels, sizes, targets)}
        upper = list(map(int, rng.permutation(labels)))
        lower = upper.copy()
        if kind == 'detour':
            i, j = map(int, rng.choice(3, 2, replace=False))
            lower[i], lower[j] = lower[j], lower[i]
        else:
            rng.shuffle(lower)
        data = dict(n=n, shape=shape, active=active, edges=edges, source=source,
                    roles=roles, wall=roles[0], templates=templates, carriers=carriers,
                    upper=upper, lower=lower)
        winners = _graph_winners(data)
        winning_sizes = {carriers[label][0] if label is not None else 0 for label in winners.values()}
        if kind == 'tree' and winning_sizes != {0, 2, 3, 4}:
            continue
        if kind == 'test' and len(winning_sizes) < 2:
            continue
        output = _render(data, winners)
        if kind == 'detour':
            if any(np.array_equal(output, _render(data, winners, transform=t)) for t in range(1, 8)):
                continue
            if np.array_equal(output, _render(data, winners, column=True)):
                continue
            if np.array_equal(output, _render(data, winners, cyclic=True)):
                continue
        inp = _encode(n, roles, templates, carriers, upper, lower, shape, active, edges, source)
        key = (inp.shape, inp.tobytes())
        if key in used:
            continue
        parsed = _parse(inp)
        _require(_pixel_winners(parsed) == winners, "graph and footprint movement disagree")
        used.add(key)
        return {'input': inp, 'output': output}
    raise RuntimeError(f"bounded {kind} construction exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    used = set()
    train = [_pair(rng, roles, 4, 'tree', used), _pair(rng, roles, 3, 'detour', used)]
    train += [_pair(rng, roles, int(rng.integers(3, 5)), 'extra', used) for _ in range(num_train-2)]
    rng.shuffle(train)
    puzzle = {'train': train, 'test': [_pair(rng, roles, int(rng.integers(3, 6)), 'test', used)
                                       for _ in range(num_test)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, parsed_splits = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, output = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            p = _parse(inp)
            winners = _pixel_winners(p)
            _require(winners == _graph_winners(p), "full footprint versus graph inconsistency")
            _require(np.array_equal(output, _render(p, winners)), "incorrect atlas")
            _require(output.shape == tuple(d*p['n'] for d in p['shape']), "atlas geometry")
            _require(winners[p['source']] == max(p['carriers'], key=lambda x: p['carriers'][x][0]), "depot winner")
            _require(set(map(int, np.unique(output))) <= {p['wall'], *(v[1] for v in p['carriers'].values())}, "atlas colors")
            parsed_splits[split].append((p, winners, output))
    records = parsed_splits['train']+parsed_splits['test']
    _require(len({p['roles'] for p, _, _ in records}) == 1, "structural palette changes within puzzle")
    training = parsed_splits['train']
    _require({p['n'] for p, _, _ in training} >= {3, 4}, "training scale evidence")
    _require(any(None in w.values() and len(set(w.values())-{None}) >= 3 for _, w, _ in training),
             "training must show unreachable rooms and three winners")
    _require(any(not np.array_equal(out, _render(p, _graph_winners(p, shortest=True))) for p, _, out in training),
             "training must refute shortest-route bottlenecks")
    for option in ('column', 'cyclic', 'solid', 'unkeyed'):
        _require(any(not np.array_equal(out, _render(p, w, **{option: True})) for p, w, out in training),
                 "missing training counterfactual: "+option)
    _require(any(not np.array_equal(out, _render(p, _graph_winners(p, ignore_clearance=True)))
                 for p, _, out in training), "clearance is not necessary")
    for t in range(1, 8):
        _require(any(not np.array_equal(out, _render(p, w, transform=t)) for p, w, out in training),
                 "template orientation is not demonstrated")
    _require(all(len(p['active']) > max(len(q['active']) for q, _, _ in training)
                 for p, _, _ in parsed_splits['test']), "test must enlarge the chamber network")
    return None


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                a = np.asarray(pair[key], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, counterexamples, mutations, and all legal counts.

    No saved traces or per-seed payloads; only compact aggregate measurements.
    """
    measurements = {'seeds': 42, 'pairs': 0, 'detour_pairs': 0, 'unreachable_pairs': 0}
    shapes, graphs, motifs, gauges, palettes, signatures, outcomes = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                winners = _pixel_winners(p)
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve")
                measurements['pairs'] += 1
                measurements['detour_pairs'] += winners != _graph_winners(p, shortest=True)
                measurements['unreachable_pairs'] += None in winners.values()
                shapes.add(pair['input'].shape)
                graph = (p['shape'], tuple(sorted(p['active'])), tuple(sorted(p['edges'].items())), p['source'])
                graphs.add(graph)
                motifs.update((m.shape, m.tobytes()) for m in p['templates'].values())
                gauges.add(tuple(sorted(size for size, _ in p['carriers'].values())))
                palettes.add(p['roles'])
                outcomes.add(tuple(sorted({p['carriers'][v][0] if v is not None else 0 for v in winners.values()})))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
    _require(len(signatures) == 42, "seed repetition")
    _require(len(graphs) >= 150 and len(motifs) >= 100 and len(shapes) >= 12, "structural variation")
    _require(len(gauges) >= 8 and len(palettes) >= 35 and len(outcomes) >= 8, "family variation")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(213536340, a, b)
    for counts in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1),
                   (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Analytic micro fixture: thin direct gate, larger three-edge detour,
    # then a low bridge followed by a wide gate. Capacities are stated here.
    roles = (0, 1, 2, 5)
    active = {(r, c) for r in range(2) for c in range(3)}
    edges = {((0, 0), (0, 1)): (1, 1), ((0, 0), (1, 0)): (3, 0),
             ((1, 0), (1, 1)): (2, 1), ((0, 1), (1, 1)): (2, 0),
             ((0, 1), (0, 2)): (1, 0), ((0, 2), (1, 2)): (3, 0)}
    templates = {3: np.eye(3, dtype=bool), 4: np.fliplr(np.eye(3, dtype=bool)),
                 6: np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool)}
    carriers = {3: (1, 7), 4: (2, 8), 6: (3, 9)}
    fixture = _encode(3, roles, templates, carriers, [3, 4, 6], [4, 3, 6],
                      (2, 3), active, edges, (0, 0))
    parsed = _parse(fixture)
    expected = {(0, 0): 6, (1, 0): 6, (0, 1): 4, (1, 1): 4, (0, 2): 3, (1, 2): 3}
    _require(_pixel_winners(parsed) == expected, "analytic detour/bridge fixture")
    _require(np.array_equal(solve(fixture), _render(parsed, expected)), "analytic atlas fixture")
    # Exercise every legal carrier size set at every micro scale on the
    # same stated-capacity argument, including sizes not selected by 0..41.
    analytic_sets = 0
    for n in (3, 4, 5):
        middle = n-1
        edges_n = {edge: (n if width == 3 else middle if width == 2 else 1, 0)
                   for edge, (width, _) in edges.items()}
        stated_caps = {(0, 0): n, (1, 0): n, (0, 1): middle,
                       (1, 1): middle, (0, 2): 1, (1, 2): 1}
        for count in range(2, min(4, n)+1):
            for size_set in combinations(range(1, n+1), count):
                labels = [3, 4, 6, 7][:count]
                carriers_n = {label: (size, target) for label, size, target in
                              zip(labels, size_set, [9, 8, 7, 6])}
                templates_n = {label: _transform(np.eye(n, dtype=bool), j % 2)
                               for j, label in enumerate(labels)}
                encoded = _encode(n, roles, templates_n, carriers_n, labels, labels[::-1],
                                  (2, 3), active, edges_n, (0, 0))
                parsed_n = _parse(encoded)
                expected_n = {}
                for point, cap in stated_caps.items():
                    eligible = [label for label in labels if carriers_n[label][0] <= cap]
                    expected_n[point] = eligible[-1] if eligible else None
                _require(_pixel_winners(parsed_n) == expected_n, "analytic size-set coverage")
                _require(np.array_equal(solve(encoded), _render(parsed_n, expected_n)), "analytic substitution")
                analytic_sets += 1
    asymmetric_fallback = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    _require(all(not np.array_equal(asymmetric_fallback, _transform(asymmetric_fallback, t))
                 for t in range(1, 8)), "fallback orientation evidence")
    # Negative mutations must fail validation, including the very last pixel.
    base = generate_puzzle(0)
    for mutation in ('output', 'gauge', 'gate'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in base[s]] for s in base}
        pair = bad['test'][0]
        p = _parse(pair['input'])
        if mutation == 'output':
            pair['output'][-1, -1] = (int(pair['output'][-1, -1])+1) % 10
        elif mutation == 'gauge':
            pair['input'][p['n']+3, 1] = p['wall']
        else:
            edge = next(iter(p['edges']))
            rr, cc = _gate_slice(*edge, p['n'])
            pair['input'][2*p['n']+4+rr.start, cc.start] = p['roles'][2]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("mutation accepted: "+mutation)
    measurements.update(distinct_puzzles=len(signatures), input_shapes=len(shapes),
                        weighted_graphs=len(graphs), motif_bitmaps=len(motifs),
                        size_sets=len(gauges), structural_palettes=len(palettes),
                        winner_size_sets=len(outcomes), analytic_size_sets=analytic_sets)
    return measurements


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
