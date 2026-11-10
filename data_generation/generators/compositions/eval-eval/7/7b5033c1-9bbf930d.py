"""Folded itineraries for a rail shuttle.

The construction graph is used only to sample valid itineraries and labels.
The solver reconstructs everything from pixels. Reserved roles are 0 (empty),
1 (two itinerary caps), 8 (station corners), and 9 (one shuttle). Colors 2..7
are literal port addresses, with no global color-to-direction dictionary.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CORNERS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
PAYLOAD = frozenset(range(2, 8))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors must be 0..9")
    return g


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _source(g):
    caps = [tuple(map(int, p)) for p in np.argwhere(g == 1)]
    _require(len(caps) == 2 and caps[0][0] != caps[1][0], "two unequal-row caps required")
    start = min(caps)
    cells, pending = {start}, [start]
    while pending:
        p = pending.pop()
        for d in DIRS:
            q = _add(p, d)
            if _inside(g, q) and g[q] != 0 and q not in cells:
                cells.add(q)
                pending.append(q)
    _require(set(caps) <= cells and 3 <= len(cells) <= 30, "one capped itinerary required")
    _require(all(int(g[p]) in PAYLOAD | {1} for p in cells), "itinerary must be separate from rails")
    neighbors = {p: [q for d in DIRS if (q := _add(p, d)) in cells] for p in cells}
    _require({p for p in cells if len(neighbors[p]) == 1} == set(caps), "caps must be the endpoints")
    _require(all(len(neighbors[p]) == (1 if p in caps else 2) for p in cells), "itinerary must be a simple chain")
    order, previous, current = [], None, start
    while True:
        _require(current not in order, "cyclic itinerary")
        order.append(current)
        onward = [p for p in neighbors[current] if p != previous]
        if not onward:
            break
        previous, current = current, onward[0]
    _require(len(order) == len(cells), "disconnected itinerary")
    return order, tuple(int(g[p]) for p in order[1:-1])


def _read_input(value):
    """Decode stations and the two-sided ports without construction metadata."""
    g = _grid(value)
    source, word = _source(g)
    nodes = set()
    for r in range(1, g.shape[0] - 1):
        for c in range(1, g.shape[1] - 1):
            p = (r, c)
            if int(g[p]) not in (0, 9):
                continue
            if (all(g[_add(p, d)] == 8 for d in CORNERS)
                    and all(g[_add(p, d)] == 0 for d in DIRS)):
                nodes.add(p)
    _require(len(nodes) >= 2, "station corner glyphs required")
    tokens = [tuple(map(int, p)) for p in np.argwhere(g == 9)]
    _require(len(tokens) == 1 and tokens[0] in nodes, "one shuttle at a station required")
    corners = {_add(p, d) for p in nodes for d in CORNERS}
    _require(corners == {tuple(map(int, p)) for p in np.argwhere(g == 8)}, "unexplained station marker")
    ports = {p: [] for p in nodes}
    rail_cells = set()
    for p in sorted(nodes):
        for d in DIRS:
            aligned = [q for q in nodes if q != p and
                       ((d[0] and q[1] == p[1] and (q[0] - p[0]) * d[0] > 0)
                        or (d[1] and q[0] == p[0] and (q[1] - p[1]) * d[1] > 0))]
            if not aligned:
                continue
            q = min(aligned, key=lambda t: abs(t[0] - p[0]) + abs(t[1] - p[1]))
            distance = abs(q[0] - p[0]) + abs(q[1] - p[1])
            _require(distance >= 5, "stations need at least five-cell separation")
            n = (-d[1], d[0])
            mouth = _add(p, d, 2)
            flanks = (_add(mouth, n), _add(mouth, n, -1))
            a, b = (int(g[t]) for t in flanks)
            if a == b == 0:
                continue  # No port: proximity alone does not create an edge.
            _require(a in PAYLOAD and b in PAYLOAD, "ports need two guide strands")
            _require(all(g[_add(p, d, j)] == 0 for j in range(1, distance)), "blocked corridor")
            lengths = []
            for flank, color in zip(flanks, (a, b)):
                length = 0
                for j in range((distance - 3) // 2):
                    t = _add(flank, d, j)
                    if g[t] != color:
                        break
                    rail_cells.add(t)
                    length += 1
                lengths.append(length)
            _require(lengths[0] == lengths[1] and lengths[0] >= 1, "parallel equal-length strands required")
            ports[p].append(dict(target=q, direction=d, sides=(a, b), length=lengths[0], distance=distance))
        active = [t['sides'][0] for t in ports[p] if t['sides'][0] == t['sides'][1]]
        _require(len(active) == len(set(active)), "duplicate active address at a station")
    explained = set(source) | nodes | corners | rail_cells
    _require(all(tuple(map(int, p)) in explained for p in np.argwhere(g != 0)), "unexplained foreground")
    return dict(grid=g, source=source, word=word, nodes=nodes, start=tokens[0], ports=ports)


def _execute(ports, start, word):
    current, hops = start, []
    for color in word:
        options = [p for p in ports[current] if p['sides'] == (color, color)]
        _require(len(options) == 1, "each command needs exactly one matching port")
        port = options[0]
        hops.append((current, port))
        current = port['target']
    return current, hops


def _relaxed_endpoints(ports, start, word):
    """Diagnostic alternative: one matching flank is enough, branch all options."""
    states = {start}
    for color in word:
        states = {p['target'] for node in states for p in ports[node] if color in p['sides']}
    return states


def _different_prediction(ports, start, word, correct):
    """A malformed alternative also contradicts the solved demonstration."""
    try:
        predicted, _ = _execute(ports, start, word)
    except ValueError:
        return True
    return predicted != correct


def _collapse(word):
    return tuple(c for i, c in enumerate(word) if i == 0 or c != word[i - 1])


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    destination, _ = _execute(info['ports'], info['start'], info['word'])
    out = info['grid'].copy()
    out[info['start']] = 0
    out[destination] = 9
    return out


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("num_train must be 2..10 and num_test 1..4, both integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_map(rng, rows, cols, palette):
    """An irregularly spaced rectilinear graph with a bidirectional backbone.

    A randomized spanning tree guarantees valid walks. Extra edges can be
    active, absent, or plausible mismatched ports, independently at each end.
    """
    rr = [1]
    cc = [1]
    for _ in range(rows - 1):
        rr.append(rr[-1] + int(rng.integers(5, 8)))
    for _ in range(cols - 1):
        cc.append(cc[-1] + int(rng.integers(5, 8)))
    points = [(r, c) for r in rr for c in cc]
    adjacent = {p: [] for p in points}
    edges = []
    for i, r in enumerate(rr):
        for j, c in enumerate(cc):
            p = (r, c)
            for q in ((rr[i + 1], c) if i + 1 < rows else None,
                      (r, cc[j + 1]) if j + 1 < cols else None):
                if q is not None:
                    adjacent[p].append(q)
                    adjacent[q].append(p)
                    edges.append((p, q))
    start = points[int(rng.integers(len(points)))]
    visited, tree = {start}, set()
    while len(visited) < len(points):
        frontier = [(p, q) for p in sorted(visited) for q in adjacent[p] if q not in visited]
        p, q = frontier[int(rng.integers(len(frontier)))]
        tree.add(frozenset((p, q)))
        visited.add(q)
    extras = [e for e in edges if frozenset(e) not in tree]
    decoy_edge = extras[int(rng.integers(len(extras)))]
    status = {}
    for p, q in edges:
        for a, b in ((p, q), (q, p)):
            if frozenset((p, q)) in tree:
                status[a, b] = 'active'
            elif (p, q) == decoy_edge:
                status[a, b] = 'mismatch'
            else:
                status[a, b] = str(rng.choice(('active', 'mismatch', 'absent'), p=(.45, .40, .15)))
    g = np.zeros((rr[-1] + 2, cc[-1] + 2), dtype=np.int8)
    for p in points:
        for d in CORNERS:
            g[_add(p, d)] = 8
    g[start] = 9
    ports = {p: [] for p in points}
    for p in points:
        active_neighbors = [q for q in adjacent[p] if status[p, q] == 'active']
        colors = [int(x) for x in rng.permutation(palette)[:len(active_neighbors)]]
        addresses = dict(zip(active_neighbors, colors))
        for q in adjacent[p]:
            state = status[p, q]
            if state == 'absent':
                continue
            d = (int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1])))
            distance = abs(q[0] - p[0]) + abs(q[1] - p[1])
            if state == 'active':
                a = b = addresses[q]
            else:
                a = colors[int(rng.integers(len(colors)))]
                others = [c for c in palette if c != a]
                b = int(others[int(rng.integers(len(others)))])
                if rng.integers(2):
                    a, b = b, a
            length = int(rng.integers(1, (distance - 3) // 2 + 1))
            n = (-d[1], d[0])
            for j in range(2, 2 + length):
                mouth = _add(p, d, j)
                g[_add(mouth, n)] = a
                g[_add(mouth, n, -1)] = b
            ports[p].append(dict(target=q, direction=d, sides=(a, b), length=length, distance=distance))
    return g, ports, start


def _contrast(ports, start, length, rng):
    """Find equal-multiset itineraries with different final stations.

    Dynamic programming retains one word per (station, color histogram).
    State space is finite and small (length <= 8, at most 6 addresses).
    """
    states = {(start, (0,) * 6): ()}
    for _ in range(length):
        next_states = {}
        keys = list(states)
        rng.shuffle(keys)
        for node, hist in keys:
            for port in ports[node]:
                a, b = port['sides']
                if a != b:
                    continue
                counts = list(hist)
                counts[a - 2] += 1
                key = port['target'], tuple(counts)
                next_states.setdefault(key, states[node, hist] + (a,))
        states = next_states
    groups = {}
    candidates = list(states.items())
    rng.shuffle(candidates)
    for (end, hist), word in candidates:
        if end == start or len(set(word)) < 2:
            continue
        if hist in groups:
            other_end, other_word = groups[hist]
            if end != other_end:
                first, second = (other_word, other_end), (word, end)
                for pair in ((first, second), (second, first)):
                    (w1, e1), (w2, e2) = pair
                    if (len(_collapse(w1)) < len(w1)
                            and _different_prediction(ports, start, _collapse(w1), e1)
                            and (_different_prediction(ports, start, w1[::-1], e1)
                                 or _different_prediction(ports, start, w2[::-1], e2))
                            and len(_relaxed_endpoints(ports, start, w1) - {e1}) > 0):
                        return w1, e1, w2, e2
        else:
            groups[hist] = end, word
    return None


def _random_word(rng, ports, start, length):
    for _ in range(64):
        node, word = start, []
        for _ in range(length):
            options = [p for p in ports[node] if p['sides'][0] == p['sides'][1]]
            port = options[int(rng.integers(len(options)))]
            word.append(port['sides'][0])
            node = port['target']
        if node != start and len(set(word)) >= 2:
            return tuple(word), node
    # Connected graphs of >=4 stations always admit a non-returning walk of
    # either parity. Enumerate endpoints and retain one word per state.
    states = {start: ()}
    for _ in range(length):
        newer = {}
        for p, word in states.items():
            for port in ports[p]:
                if port['sides'][0] == port['sides'][1]:
                    newer.setdefault(port['target'], word + (port['sides'][0],))
        states = newer
    for p, word in states.items():
        if p != start and len(set(word)) >= 2:
            return word, p
    raise RuntimeError("no informative itinerary on sampled map")


def _path(rng, length, max_h, max_w):
    """Induced walks, not a fixed folded silhouette."""
    for _ in range(300):
        order, occupied = [(0, 0)], {(0, 0)}
        previous = None
        persistence = float(rng.choice((.6, 1.5, 4.0)))
        while len(order) < length:
            candidates, weights = [], []
            for d in DIRS:
                q = _add(order[-1], d)
                if q in occupied or sum(_add(q, a) in occupied for a in DIRS) != 1:
                    continue
                pts = order + [q]
                if (max(r for r, _ in pts) - min(r for r, _ in pts) + 1 > max_h
                        or max(c for _, c in pts) - min(c for _, c in pts) + 1 > max_w):
                    continue
                candidates.append((q, d))
                weights.append(persistence if d == previous else 1.)
            if not candidates:
                break
            probs = np.asarray(weights) / sum(weights)
            q, previous = candidates[int(rng.choice(len(candidates), p=probs))]
            order.append(q)
            occupied.add(q)
        if len(order) == length and order[0][0] != order[-1][0]:
            top, left = min(r for r, _ in order), min(c for _, c in order)
            return [(r - top, c - left) for r, c in order]
    # A separated hairpin fits all generation sizes: maximum available
    # source-strip height is at least 4, and width at least 13.
    for run in range(2, length - 3):
        path = [(0, c) for c in range(run + 1)]
        path += [(1, run), (2, run)]
        path += [(2, run - j) for j in range(1, length - len(path) + 1)]
        top, left = min(r for r, _ in path), min(c for _, c in path)
        path = [(r - top, c - left) for r, c in path]
        if max(r for r, _ in path) < max_h and max(c for _, c in path) < max_w:
            return path
    raise RuntimeError("bounded itinerary construction exhausted")


def _assemble(rng, board, word, destination, *, witness=False, second_destination=None):
    """Pack a separate itinerary near the map, then orient the full scene.

    Assign commands AFTER rotation/reflection, since the higher endpoint is
    always the start; output labels use the known construction destination.
    """
    for attempt in range(200):
        horizontal = bool(rng.integers(2))
        block = board if horizontal else board.T
        bh, bw = block.shape
        max_h = 30 - bh - 2
        if max_h < 3:
            continue
        if witness and attempt >= 40:
            # The random walk can under-sample above-endpoint folds at the
            # shortest lengths. An upward prefix, top run, and longer descent
            # supplies an explicit finite teaching fallback.
            across = int(rng.integers(2, min(4, len(word) - 3) + 1))
            path = [(2, 0), (1, 0), (0, 0)]
            path += [(0, c) for c in range(1, across + 1)]
            path += [(r, across) for r in range(1, len(word) + 2 - len(path) + 1)]
            if max(r for r, c in path) >= max_h:
                continue
        else:
            path = _path(rng, len(word) + 2, min(max_h, 8), max(13, bw))
        ph, pw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
        h, w = bh + ph + 2, max(bw, pw) + int(rng.integers(0, 3))
        if w > 30:
            continue
        g = np.zeros((h, w), dtype=np.int8)
        source_left, map_left = int(rng.integers(w - pw + 1)), int(rng.integers(w - bw + 1))
        map_top = ph + 2
        g[map_top:map_top + bh, map_left:map_left + bw] = block
        for i, (r, c) in enumerate(path):
            g[r, c + source_left] = 1 if i in (0, len(path) - 1) else 2
        out = g.copy()
        out[out == 9] = 0
        dr, dc = destination if horizontal else destination[::-1]
        out[dr + map_top, dc + map_left] = 9
        locator = np.zeros_like(g)
        if second_destination is not None:
            dr, dc = second_destination if horizontal else second_destination[::-1]
            locator[dr + map_top, dc + map_left] = 1
        rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
        g, out = np.rot90(g, rotation).copy(), np.rot90(out, rotation).copy()
        locator = np.rot90(locator, rotation).copy()
        if reflection:
            g, out = np.fliplr(g).copy(), np.fliplr(out).copy()
            locator = np.fliplr(locator).copy()
        try:
            order, _ = _source(g)
        except ValueError:
            continue
        for p, color in zip(order[1:-1], word):
            g[p] = out[p] = color
        if witness:
            # Teach the upper ENDPOINT even if an internal bead is higher.
            if min(r for r, c in order) >= order[0][0]:
                continue
            info = _read_input(g)
            raster = tuple(int(g[p]) for p in sorted(order[1:-1]))
            try:
                wrong, _ = _execute(info['ports'], info['start'], raster)
            except ValueError:
                wrong = None
            correct = tuple(map(int, np.argwhere(out == 9)[0]))
            if raster == word or wrong == correct:
                continue
        pair = dict(input=g, output=out)
        if second_destination is not None:
            return pair, tuple(map(int, np.argwhere(locator == 1)[0]))
        return pair
    raise RuntimeError("bounded scene packing exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('longer', 'larger', 'both')))
    palette = [int(c) for c in rng.permutation(np.arange(2, 8))[:int(rng.integers(4, 7))]]
    puzzle = {'train': [], 'test': []}
    # The first two examples are a controlled contrast, not two sub-puzzles:
    # identical map, start and source geometry; equal histograms, different order.
    for _ in range(80):
        rows, cols = ((2, 3) if rng.integers(2) else (3, 2))
        board, ports, start = _make_map(rng, rows, cols, palette)
        length = int(rng.integers(6, 9))
        contrast = _contrast(ports, start, length, rng)
        if contrast is not None:
            word, end, word2, end2 = contrast
            try:
                first, second_end = _assemble(rng, board, word, end, witness=True, second_destination=end2)
            except RuntimeError:
                continue
            break
    else:
        raise RuntimeError("bounded contrast construction exhausted")
    info = _read_input(first['input'])
    second_input = first['input'].copy()
    for p, color in zip(info['source'][1:-1], word2):
        second_input[p] = color
    # Both labels use sampled graph endpoints carried through the same image
    # transform. Pixel reconstruction is checked separately by the validator.
    second_output = second_input.copy()
    second_output[info['start']] = 0
    second_output[second_end] = 9
    puzzle['train'] = [first, dict(input=second_input, output=second_output)]
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _index in range(count):
            for _attempt in range(100):
                if split == 'test' and shift in ('larger', 'both'):
                    rows, cols = ((3, 3), (3, 4), (4, 3))[int(rng.integers(3))]
                else:
                    rows, cols = ((2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
                length = int(rng.integers(9, 15)) if split == 'test' and shift in ('longer', 'both') else int(rng.integers(4, 9))
                # Local addressing conventions stay fixed; address literals and
                # map wiring can vary independently between examples.
                local_palette = palette if rng.random() < .6 else list(map(int, rng.permutation(np.arange(2, 8))[:int(rng.integers(4, 7))]))
                board, ports, start = _make_map(rng, rows, cols, local_palette)
                word, end = _random_word(rng, ports, start, length)
                pair = _assemble(rng, board, word, end)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _read_input(pair['input'])
    end, hops = _execute(info['ports'], info['start'], info['word'])
    path = info['source']
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return dict(shape=info['grid'].shape, nodes=len(info['nodes']), commands=len(info['word']),
                turns=sum(a != b for a, b in zip(steps, steps[1:])),
                above=min(r for r, c in path) < path[0][0],
                source_geometry=tuple((r - path[0][0], c - path[0][1]) for r, c in path),
                directions={p['direction'] for node, p in hops},
                distances={p['distance'] for ps in info['ports'].values() for p in ps},
                rail_lengths={p['length'] for ps in info['ports'].values() for p in ps},
                mismatches=sum(p['sides'][0] != p['sides'][1] for ps in info['ports'].values() for p in ps),
                revisits=len({info['start']} | {p['target'] for node, p in hops}) < len(hops) + 1,
                local_color_directions={(int(p['sides'][0]), p['direction']) for node, p in hops},
                relaxed_alternative=bool(_relaxed_endpoints(info['ports'], info['start'], info['word']) - {end}))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read_input(inp)
            end, hops = _execute(info['ports'], info['start'], info['word'])
            _require(end != info['start'], "generated examples must move the shuttle")
            expected = inp.copy()
            expected[info['start']] = 0
            expected[end] = 9
            _require(np.array_equal(out, expected), "incorrect endpoint or changed evidence")
            _require(np.count_nonzero(out == 9) == 1 and np.count_nonzero(inp != out) == 2, "shuttle conservation")
            protected = (inp != 9) & (out != 9)
            _require(np.array_equal(inp[protected], out[protected]), "evidence preservation")
            f = _features(pair)
            _require(f['mismatches'] >= 2, "matched and mismatched ports must coexist")
            features[split].append(f)
    first, second = puzzle['train'][:2]
    ia, ib = _read_input(first['input']), _read_input(second['input'])
    _require(Counter(ia['word']) == Counter(ib['word']), "controlled contrast must have equal counts")
    _require(ia['word'] != ib['word'], "controlled contrast must change order")
    mask = np.zeros(ia['grid'].shape, dtype=bool)
    for p in ia['source'][1:-1]:
        mask[p] = True
    _require(ia['grid'].shape == ib['grid'].shape and np.array_equal(ia['grid'][~mask], ib['grid'][~mask]), "contrast must share map and geometry")
    ea, _ = _execute(ia['ports'], ia['start'], ia['word'])
    eb, _ = _execute(ib['ports'], ib['start'], ib['word'])
    _require(ea != eb, "order must change the destination")
    _require(features['train'][0]['above'], "upper-endpoint convention must be taught")
    _require(features['train'][0]['relaxed_alternative'], "two-sided equality must affect the answer")
    _require(len(_collapse(ia['word'])) < len(ia['word'])
             and _different_prediction(ia['ports'], ia['start'], _collapse(ia['word']), ea),
             "each bead, including equal consecutive beads, must matter")
    _require(_different_prediction(ia['ports'], ia['start'], ia['word'][::-1], ea)
             or _different_prediction(ib['ports'], ib['start'], ib['word'][::-1], eb),
             "traversal direction must be distinguished")
    address_directions = (features['train'][0]['local_color_directions']
                          | features['train'][1]['local_color_directions'])
    _require(any(len({d for c2, d in address_directions if c2 == c}) > 1
                 for c, d in address_directions), "addresses must be local, not global directions")
    raster = tuple(int(ia['grid'][p]) for p in sorted(ia['source'][1:-1]))
    try:
        wrong, _ = _execute(ia['ports'], ia['start'], raster)
    except ValueError:
        wrong = None
    _require(wrong != ea, "raster shortcut must fail in training")
    max_nodes = max(f['nodes'] for f in features['train'])
    max_length = max(f['commands'] for f in features['train'])
    _require(all(f['nodes'] > max_nodes or f['commands'] > max_length for f in features['test']), "meaningful test shift required")


def self_test():
    """Bounded 42-seed semantic, mutation, count-contract and diversity checks."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = np.asarray(pair['input']).copy()
                prediction = solve(before)
                assert np.array_equal(prediction, pair['output'])
                assert np.array_equal(before, pair['input']), "solver mutated input"
                records.append(_features(pair))
    assert len({tuple(_key(p['input']) for s in ('train', 'test') for p in task[s]) for task in puzzles}) == 42
    assert len({f['shape'] for f in records}) >= 40
    assert len({f['source_geometry'] for f in records}) >= 100
    assert {4, 6, 9, 12} <= {f['nodes'] for f in records}
    assert set(range(4, 15)) <= {f['commands'] for f in records}
    assert {5, 6, 7} <= set().union(*(f['distances'] for f in records))
    assert {1, 2} <= set().union(*(f['rail_lengths'] for f in records))
    assert set(DIRS) == set().union(*(f['directions'] for f in records))
    assert len({f['turns'] for f in records}) >= 6
    assert sum(f['revisits'] for f in records) >= 30
    assert all(p['input'].shape == p['output'].shape for task in puzzles for s in task for p in task[s])
    # Counts cover every legal explicit combination, with the smallest training
    # set retaining the controlled contrast and the largest retaining its shift.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(73, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    again = generate_puzzle(17)
    assert all(np.array_equal(a[k], b[k]) for s in again for a, b in zip(again[s], puzzles[17][s]) for k in ('input', 'output'))
    # A wrong endpoint or damaged guide must not be accepted as ground truth.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("identity label accepted")
    info = _read_input(puzzles[0]['train'][0]['input'])
    end, hops = _execute(info['ports'], info['start'], info['word'])
    current, port = hops[0]
    normal = (-port['direction'][1], port['direction'][0])
    damaged = info['grid'].copy()
    flank = _add(_add(current, port['direction'], 2), normal)
    damaged[flank] = (int(damaged[flank]) - 2 + 1) % 6 + 2
    try:
        solve(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged commanded matching port accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal count combinations')
