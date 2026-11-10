"""Closed color circuits: broken frames route simultaneous island recoloring.

Only NumPy and the standard library are used. All semantic information is read
from the query grid; generation plans are never consulted by solve().
"""

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
PORTS = ((-1, -1), (-1, 0), (-1, 1), (-1, 2),
         (0, -1), (0, 2), (1, -1), (1, 2),
         (2, -1), (2, 0), (2, 1), (2, 2))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, skip=()):
    labels = np.full(a.shape, -1, dtype=np.int16)
    cells, colors = [], []
    for p in np.ndindex(a.shape):
        if labels[p] >= 0 or int(a[p]) in skip:
            continue
        color, index = int(a[p]), len(cells)
        stack, found = [p], []
        labels[p] = index
        while stack:
            q = stack.pop()
            found.append(q)
            for dr, dc in DIRS:
                z = q[0] + dr, q[1] + dc
                if _inside(z, a.shape) and labels[z] < 0 and a[z] == color:
                    labels[z] = index
                    stack.append(z)
        cells.append(tuple(found))
        colors.append(color)
    return labels, colors, cells


def _port_direction(offset):
    return tuple(-1 if x == -1 else (1 if x == 2 else 0) for x in offset)


def _trace(labels, source, point, direction):
    """A sight line is transparent to its own original island, opaque to others."""
    p, water = point, []
    while _inside(p, labels.shape):
        label = int(labels[p])
        if label >= 0 and label != source:
            return label, tuple(water)
        if label < 0:
            water.append(p)
        p = p[0] + direction[0], p[1] + direction[1]
    return -1, tuple(water)


def _cycles(edges):
    """Return directed cycles, excluding trees that merely lead into them."""
    finished, cycles = set(), []
    for start in range(len(edges)):
        route, positions, node = [], {}, start
        while node >= 0 and node not in finished and node not in positions:
            positions[node] = len(route)
            route.append(node)
            node = edges[node]
        if node in positions:
            cycle = route[positions[node]:]
            _need(len(cycle) >= 2, "self links are not allowed")
            cycles.append(tuple(cycle))
        finished.update(route)
    return cycles


def _decode(input_grid):
    a = _grid(input_grid)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _need(len(set(map(int, border))) == 1, "the water border must be uniform")
    bg = int(border[0])
    _, raw_colors, raw_cells = _components(a, (bg,))
    frames = []
    for ink, component in zip(raw_colors, raw_cells):
        if len(component) != 11:
            continue
        points = set(component)
        r, c = min(p[0] for p in points), min(p[1] for p in points)
        if max(p[0] for p in points) != r + 3 or max(p[1] for p in points) != c + 3:
            continue
        perimeter = {(r+i, c+j) for i in range(4) for j in range(4)
                     if i in (0, 3) or j in (0, 3)}
        if not points <= perimeter or len(perimeter - points) != 1:
            continue
        core = a[r+1:r+3, c+1:c+3]
        color = int(core[0, 0])
        gap = next(iter(perimeter - points))
        if color in (bg, ink) or not np.all(core == color) or a[gap] != color:
            continue
        offset = gap[0] - r - 1, gap[1] - c - 1
        frames.append(dict(cells=points, origin=(r, c), core=(r+1, c+1),
                           gap=gap, direction=_port_direction(offset), offset=offset,
                           ink=ink, color=color))
    _need(bool(frames), "no broken frames")
    inks = {f['ink'] for f in frames}
    _need(len(inks) == 1, "frames must share an ink distinct from island fills")
    ink = next(iter(inks))
    all_ink = set(map(tuple, np.argwhere(a == ink)))
    _need(set().union(*(f['cells'] for f in frames)) == all_ink, "unexplained frame ink")
    clean = a.copy()
    for frame in frames:
        for p in frame['cells']:
            clean[p] = frame['color']
    labels, colors, cells = _components(clean, (bg,))
    _need(len(frames) == len(cells), "each island must have exactly one frame")
    by_id = {}
    for frame in frames:
        label = int(labels[frame['core']])
        _need(label >= 0 and label not in by_id, "duplicate frame on an island")
        r, c = frame['origin']
        _need(np.all(labels[r:r+4, c:c+4] == label), "frame straddles islands")
        _need(len(cells[label]) >= 20, "island needs visible body outside its frame")
        by_id[label] = frame
    # Even equal-colored islands stay distinct; no two islands touch, including diagonally.
    for p in np.ndindex(labels.shape):
        if labels[p] < 0:
            continue
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            q = p[0]+dr, p[1]+dc
            if _inside(q, labels.shape) and labels[q] >= 0:
                _need(labels[q] == labels[p], "islands must have a water gap")
    edges, rays = [], []
    for label in range(len(cells)):
        frame = by_id[label]
        target, water = _trace(labels, label, frame['gap'], frame['direction'])
        edges.append(target)
        rays.append(water)
    loops = _cycles(edges)
    output = clean.copy()
    for loop in loops:
        for source in loop:
            output[labels == edges[source]] = colors[source]
    return output, dict(clean=clean, labels=labels, colors=colors, cells=cells,
                        frames=by_id, edges=edges, cycles=loops, rays=rays,
                        background=bg, ink=ink)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase frames and circulate original colors only around closed sight-line loops."""
    return _decode(input_grid)[0]


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _random_layout(rng, shape, count):
    """Pack unequal rectangles with optional corner notches; no macrocell lattice."""
    labels = np.full(shape, -1, dtype=np.int16)
    cores = []
    for index in range(count):
        for _ in range(180):
            h, w = map(int, rng.integers(4, 9, 2))
            if h*w < 20:
                continue
            r = int(rng.integers(1, shape[0]-h))
            c = int(rng.integers(1, shape[1]-w))
            fr, fc = int(rng.integers(h-3)), int(rng.integers(w-3))
            mask = np.ones((h, w), dtype=bool)
            protected = np.zeros((h, w), dtype=bool)
            protected[fr:fr+4, fc:fc+4] = True
            for corner in rng.permutation(4)[:int(rng.integers(1, 5))]:
                dh, dw = map(int, rng.integers(1, 4, 2))
                rr = slice(0, dh) if corner < 2 else slice(h-dh, h)
                cc = slice(0, dw) if corner % 2 == 0 else slice(w-dw, w)
                if not np.any(protected[rr, cc]):
                    mask[rr, cc] = False
            if int(mask.sum()) < 20:
                continue
            points = [(r+int(y), c+int(x)) for y, x in np.argwhere(mask)]
            if any(labels[y+dy, x+dx] >= 0 for y, x in points
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
                continue
            for p in points:
                labels[p] = index
            cores.append((r+fr+1, c+fc+1))
            break
        else:
            return None
    return labels, cores


def _options(labels, cores):
    options = []
    for i, (r, c) in enumerate(cores):
        targets = {}
        for offset in PORTS:
            p = r+offset[0], c+offset[1]
            try:
                target, _ = _trace(labels, i, p, _port_direction(offset))
            except AssertionError:
                continue
            targets.setdefault(target, []).append(offset)
        options.append(targets)
    return options


def _possible_loops(options, length):
    found = []
    def visit(route):
        if len(route) == length:
            if route[0] in options[route[-1]]:
                found.append(tuple(route))
            return
        for node in sorted(options[route[-1]]):
            if node > route[0] and node not in route:
                visit(route + [node])
    for start in range(len(options)):
        visit([start])
    return found


def _plan_graph(rng, options, length, style):
    loops = _possible_loops(options, length) if length else [()]
    rng.shuffle(loops)
    for loop in loops:
        outside = [i for i in range(len(options)) if i not in loop]
        edges = [-1] * len(options)
        if any(-1 not in options[i] for i in outside):
            continue
        for i, source in enumerate(loop):
            edges[source] = loop[(i+1) % len(loop)]
        if style == 'open_chain':
            chains = [(a, b) for a in outside for b in outside if a != b and b in options[a]]
            if not chains:
                continue
            a, b = chains[int(rng.integers(len(chains)))]
            edges[a] = b
            witness = (a, b)
        elif style == 'tail':
            tails = [(a, b) for a in outside for b in loop if b in options[a]]
            if not tails:
                continue
            a, b = tails[int(rng.integers(len(tails)))]
            edges[a] = b
            witness = (a, b)
        else:
            witness = ()
        # Other nodes may terminate at the border, point into the loop, or form
        # acyclic branches. Do not accidentally introduce another cycle here.
        reserved = set(witness[:1])
        if style == 'open_chain':
            reserved.add(witness[1])
        # Two independent circulating loops can share the same water canvas.
        # The routing rule is unchanged; each loop conserves its own pigments.
        remaining = [i for i in outside if i not in reserved]
        second_loops = [(a, b) for a in remaining for b in remaining
                        if a < b and b in options[a] and a in options[b]]
        if length and second_loops and rng.random() < .45:
            a, b = second_loops[int(rng.integers(len(second_loops)))]
            edges[a], edges[b] = b, a
            reserved.update((a, b))
        expected_lengths = sorted(map(len, _cycles(edges)))
        for node in rng.permutation(outside):
            node = int(node)
            if node in reserved or (style == 'open_chain' and node == witness[1]):
                continue
            choices = [-1] + [x for x in options[node] if x >= 0]
            rng.shuffle(choices)
            for target in choices:
                trial = edges.copy()
                trial[node] = target
                if sorted(map(len, _cycles(trial))) == expected_lengths:
                    edges = trial
                    break
        return loop, edges, witness
    return None


def _paint_scene(rng, labels, cores, options, graph, palette):
    loop, edges, witness = graph
    bg, ink = palette[:2]
    pool = list(palette[2:])
    colors = [int(pool[i % len(pool)]) for i in range(len(cores))]
    # A disconnected non-circuit island shares a circuit fill but must stay put.
    cyclic = set().union(*map(set, _cycles(edges))) if loop else set()
    outside = [i for i in range(len(cores)) if i not in cyclic]
    if loop and outside:
        repeat = (witness[1] if witness and witness[1] in outside else outside[0])
        colors[repeat] = colors[loop[0]]
    if witness and witness[1] not in loop:
        a, b = witness
        if colors[a] == colors[b]:
            colors[a] = next(c for c in pool if c != colors[b])
    clean = np.full(labels.shape, bg, dtype=np.int8)
    for i, color in enumerate(colors):
        clean[labels == i] = color
    inp, out = clean.copy(), clean.copy()
    for i, core in enumerate(cores):
        offset_list = options[i][edges[i]]
        gap = offset_list[int(rng.integers(len(offset_list)))]
        for offset in PORTS:
            if offset != gap:
                inp[core[0]+offset[0], core[1]+offset[1]] = ink
    for circuit in _cycles(edges):
        for source in circuit:
            out[labels == edges[source]] = colors[source]
    # Whole-scene isometries act on both evidence and answers, not on a hidden convention.
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return dict(input=inp.copy(), output=out.copy())


def _make_pair(rng, length, style, palette, test=False):
    for _ in range(320):
        extra = int(rng.integers(2 if style == 'open_chain' else (1 if style == 'tail' else 0), 4))
        count = min(8, max(1, length + extra))
        low = 24 if test else min(24, 10 + 2*count)
        shape = tuple(map(int, rng.integers(low, 31, 2)))
        layout = _random_layout(rng, shape, count)
        if layout is None:
            continue
        labels, cores = layout
        options = _options(labels, cores)
        graph = _plan_graph(rng, options, length, style)
        if graph is None:
            continue
        pair = _paint_scene(rng, labels, cores, options, graph, palette)
        try:
            predicted, info = _decode(pair['input'])
        except AssertionError:
            continue
        _need(np.array_equal(predicted, pair['output']), "construction disagrees with input reader")
        if not test and style in ('open_chain', 'tail'):
            kinds = {sum(map(abs, f['direction'])) for f in info['frames'].values()}
            if kinds != {1, 2}:
                continue
        return pair
    raise RuntimeError("bounded circuit construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if not _integer(value) or not low <= value <= high:
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = tuple(map(int, rng.permutation(10)))
    stable_palette = bool(rng.integers(2))
    plans = [(2, 'open_chain'), (3, 'tail')]
    for _ in range(num_train-2):
        length = int(rng.choice((0, 2, 3)))
        plans.append((length, 'free'))
    rng.shuffle(plans)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            length, style = plans[index] if split == 'train' else (int(rng.choice((4, 5))), 'tail')
            for _ in range(12):
                roles = palette if stable_palette or rng.random() < .35 else tuple(map(int, rng.permutation(10)))
                pair = _make_pair(rng, length, style, roles, split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    loops = set().union(*map(set, info['cycles'])) if info['cycles'] else set()
    colors, edges = info['colors'], info['edges']
    final = colors.copy()
    for source in loops:
        final[edges[source]] = colors[source]
    repeated_locality = any(colors[a] == colors[b] and final[a] != final[b]
                            for a in range(len(colors)) for b in range(a))
    open_link = any(a not in loops and b >= 0 and b not in loops and colors[a] != colors[b]
                    for a, b in enumerate(edges))
    tail = any(a not in loops and b in loops for a, b in enumerate(edges))
    nonrect = sum(len(s) != (max(p[0] for p in s)-min(p[0] for p in s)+1) *
                  (max(p[1] for p in s)-min(p[1] for p in s)+1) for s in info['cells'])
    return dict(islands=len(colors), cycle_lengths=tuple(sorted(map(len, info['cycles']))),
                locality=repeated_locality, open_link=open_link, tail=tail,
                nonrectangular=nonrect, diagonal=sum(abs(f['direction'][0])+abs(f['direction'][1]) == 2
                                                   for f in info['frames'].values()),
                ray_lengths=tuple(len(r) for r in info['rays']),
                roles=(info['background'], info['ink']))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "dimensions must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            expected, info = _decode(inp)
            _need(np.array_equal(expected, out), "incorrect closed-circuit recoloring")
            _need(np.array_equal(inp == info['background'], out == info['background']), "island geometry changed")
            _need(not np.any(out == info['ink']), "frame ink survives")
            loops = set().union(*map(set, info['cycles'])) if info['cycles'] else set()
            for i, cells in enumerate(info['cells']):
                values = {int(out[p]) for p in cells}
                _need(len(values) == 1, "whole island must get one color")
                if i not in loops:
                    _need(values == {info['colors'][i]}, "an open-route island changed")
            # Circulation preserves the multiset of island fills, not color pixel counts.
            final_colors = [int(out[s[0]]) for s in info['cells']]
            _need(Counter(final_colors) == Counter(info['colors']), "island-color inventory changed")
            features[split].append(_features(info))
    train = features['train']
    _need(any(2 in f['cycle_lengths'] and f['open_link'] for f in train), "missing cycle/open-chain contrast")
    _need(any(3 in f['cycle_lengths'] and f['tail'] for f in train), "missing longer loop with an incoming tail")
    _need(any(f['locality'] for f in train), "missing equal-color component contrast")
    _need(any(f['diagonal'] for f in train) and any(f['diagonal'] < f['islands'] for f in train),
          "training must expose side and corner gaps")
    max_train = max((x for f in train for x in f['cycle_lengths']), default=0)
    _need(all(max(f['cycle_lengths'], default=0) > max_train for f in features['test']),
          "test must extend the demonstrated cycle length")


def _hand_fixtures():
    """Small fixed geometries with answers stated independently of graph tracing."""
    def draw(shape, specs, expected):
        inp, out = np.zeros(shape, dtype=np.int8), np.zeros(shape, dtype=np.int8)
        for (r, c, color, gap), final in zip(specs, expected):
            inp[r:r+5, c:c+5] = color
            out[r:r+5, c:c+5] = final
            for offset in PORTS:
                if offset != gap:
                    inp[r+1+offset[0], c+1+offset[1]] = 9
        return inp, out
    # A <-> B, C -> D -> border; A and D originally share fill 1.
    yield draw((23, 23), ((2, 2, 1, (0, 2)), (2, 15, 2, (0, -1)),
                          (15, 2, 3, (0, 2)), (15, 15, 1, (2, 0))), (2, 1, 3, 1))
    # Clockwise four-cycle: every destination gets its predecessor's ORIGINAL color.
    yield draw((23, 23), ((2, 2, 1, (0, 2)), (2, 15, 2, (2, 0)),
                          (15, 15, 3, (1, -1)), (15, 2, 4, (-1, 1))), (4, 1, 2, 3))
    # A's eastward line hits B before C. D points upward into B. Only B <-> C circulates.
    yield draw((25, 29), ((2, 2, 1, (0, 2)), (2, 11, 2, (0, 2)),
                          (2, 20, 3, (0, -1)), (15, 11, 4, (-1, 0))), (1, 3, 2, 4))
    # A triangular circuit needs a corner gap as well as side gaps.
    yield draw((23, 23), ((2, 2, 1, (0, 2)), (2, 15, 2, (2, 0)),
                          (15, 15, 3, (-1, -1))), (3, 1, 2))


def self_test():
    """Bounded 42-seed structural/semantic audit and explicit count regressions."""
    shapes, layouts, palettes, counts, lengths, directions = set(), set(), set(), set(), set(), set()
    nonrect = locality = open_links = tails = multiple = 0
    for inp, expected in _hand_fixtures():
        _need(np.array_equal(solve(inp), expected), "hand-specified circuit fixture failed")
        for k in range(4):
            _need(np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k)), "rotation covariance failed")
        _need(np.array_equal(solve(np.fliplr(inp)), np.fliplr(expected)), "reflection covariance failed")
        permutation = np.random.default_rng(44).permutation(10)
        _need(np.array_equal(solve(permutation[inp]), permutation[expected]), "color-role covariance failed")
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed in (0, 17, 41):
            repeat = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in puzzle for a, b in zip(puzzle[s], repeat[s])
                      for k in ('input', 'output')), "seed nondeterminism")
        for split in puzzle:
            for pair in puzzle[split]:
                output, info = _decode(pair['input'])
                _need(np.array_equal(output, pair['output']), "input-only solve mismatch")
                f = _features(info)
                shapes.add(pair['input'].shape)
                # Geometry signature deliberately ignores literal fills, ink, and water color.
                layouts.add((pair['input'].shape, info['labels'].tobytes(),
                             tuple((i, v['gap'], v['direction']) for i, v in info['frames'].items())))
                palettes.add(f['roles'])
                counts.add(f['islands'])
                lengths.update(f['cycle_lengths'])
                directions.update(v['direction'] for v in info['frames'].values())
                nonrect += f['nonrectangular'] > 0
                locality += f['locality']
                open_links += f['open_link']
                tails += f['tail']
                multiple += len(f['cycle_lengths']) > 1
                # Omitting A's whole-component transfer disagrees on a body cell.
                for loop in info['cycles']:
                    for source in loop:
                        target = info['edges'][source]
                        if info['colors'][source] != info['colors'][target]:
                            r, c = info['frames'][target]['origin']
                            _need(any(output[p] != info['clean'][p] and
                                      not (r <= p[0] < r+4 and c <= p[1] < c+4)
                                      for p in info['cells'][target]), "no substantive flood-fill witness")
    _need(len(shapes) >= 40 and len(layouts) >= 200, "geometry variation collapsed")
    _need(len(palettes) >= 15 and len(counts) >= 5, "palette/count variation collapsed")
    _need({2, 3, 4, 5} <= lengths and len(directions) == 8, "loop/direction variation missing")
    _need(min(nonrect, locality, open_links, tails) >= 42, "family-specific witnesses missing")
    _need(multiple >= 3, "multiple independent circuits missing")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + 10*nt + ne, nt, ne)
            _need((len(p['train']), len(p['test'])) == (nt, ne), "ignored explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(70, 2, 1)
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    return dict(seeds=42, grid_shapes=len(shapes), geometry_signatures=len(layouts),
                role_palettes=len(palettes), island_counts=sorted(counts),
                cycle_lengths=sorted(lengths), directions=len(directions),
                nonrectangular_pairs=nonrect, locality_pairs=locality,
                open_chain_pairs=open_links, tail_pairs=tails, multiple_circuit_pairs=multiple)
