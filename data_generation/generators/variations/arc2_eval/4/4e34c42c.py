"""Translate overlapping fragments, preserve their features, and crop their union.

Color names, canvas positions, and motif identities have no puzzle-wide meaning.
Sampling modes control difficulty only; all answers are reconstructed from pixels.
"""
import numpy as np


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _shape(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _render_tight(cells, background):
    cells = _normalize(cells)
    result = np.full(_shape(cells), background, dtype=np.int8)
    for p, color in cells.items():
        result[p] = color
    return result


def _connected_sets(points):
    remaining = set(points)
    result = []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        group, stack = {point}, [point]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _extract_components(grid, background):
    return [_normalize({p: int(grid[p]) for p in group})
            for group in _connected_sets(zip(*np.nonzero(grid != background)))]


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) > 1 and counts.max()*2 > grid.size, "unclear background"
    return int(colors[counts.argmax()])


def _features(cells):
    """Whole color regions, plus separated tips belonging to one color layer."""
    result = set()
    for color in set(cells.values()):
        layer = {p for p, v in cells.items() if v == color}
        result.add(frozenset(layer))
        result.update(frozenset(group) for group in _connected_sets(layer))
    return result


def _registrations(a, b):
    """Find translations supported by an entire feature and its clipped context.

    Empty cells inside both observed rectangles must agree. A color region or
    separated same-color tips must be fully represented on at least one side.
    A singleton alone never registers a fragment. Partial views of one feature
    are discarded only when BOTH footprints are contained in a fuller match;
    neither area nor total overlap is used to choose between distinct features.
    """
    ah, aw = _shape(a)
    bh, bw = _shape(b)
    fa, fb = _features(a), _features(b)
    shifts = {(r-u, c-v) for (r, c), x in a.items()
              for (u, v), y in b.items() if x == y}
    found = []
    for dr, dc in sorted(shifts):
        hit_a = frozenset(p for p in a if (p[0]-dr, p[1]-dc) in b)
        if len(hit_a) < 2:
            continue
        if any(a[r, c] != b[r-dr, c-dc] for r, c in hit_a):
            continue
        if any(0 <= r-dr < bh and 0 <= c-dc < bw and (r, c) not in hit_a
               for r, c in a):
            continue
        hit_b = frozenset((r-dr, c-dc) for r, c in hit_a)
        if any(0 <= r+dr < ah and 0 <= c+dc < aw and (r, c) not in hit_b
               for r, c in b):
            continue
        multicolor = len({a[p] for p in hit_a}) > 1
        if not any(f <= hit and (len(f) > 1 or multicolor)
                   for features, hit in ((fa, hit_a), (fb, hit_b)) for f in features):
            continue
        found.append(((dr, dc), hit_a, hit_b))
    return [item for item in found if not any(
        item[1] <= other[1] and item[2] <= other[2]
        and (item[1] != other[1] or item[2] != other[2])
        for other in found)]


def _holes(cells):
    cells = _normalize(cells)
    h, w = _shape(cells)
    empty = {(r, c) for r in range(h) for c in range(w) if (r, c) not in cells}
    return [group for group in _connected_sets(empty)
            if all(0 < r < h-1 and 0 < c < w-1 for r, c in group)]


def _has_branch(grid, background):
    points = set(zip(*np.nonzero(grid != background)))
    for r, c in points:
        if sum(p in points for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))) < 3:
            continue
        if sum(len(group) >= 4 for group in _connected_sets(points-{(r, c)})) >= 3:
            return True
    return False


def _reconstruct(components, background):
    """Solve a connected system of observed feature translations, without labels."""
    n = len(components)
    assert n >= 2, "need separated fragments"
    edges, adjacency = [], [[] for _ in components]
    for i, a in enumerate(components):
        for j in range(i+1, n):
            matches = _registrations(a, components[j])
            if matches:
                edges.append((i, j, matches))
                adjacency[i].append((j, [m[0] for m in matches]))
                adjacency[j].append((i, [(-m[0][0], -m[0][1]) for m in matches]))
    offsets = {0: (0, 0)}
    solutions = []
    visits = [0]

    def search():
        visits[0] += 1
        assert visits[0] <= 2000, "unresolved registration search"
        if len(solutions) > 1:
            return
        if len(offsets) == n:
            union = {}
            for i, part in enumerate(components):
                dr, dc = offsets[i]
                for (r, c), color in part.items():
                    p = r+dr, c+dc
                    if p in union and union[p] != color:
                        return
                    union[p] = color
            if max(_shape(_normalize(union))) > 30:
                return
            # Enclosed negative spaces are observed material, not blank canvas.
            for i, part in enumerate(components):
                dr, dc = offsets[i]
                if any((r+dr, c+dc) in union for hole in _holes(part) for r, c in hole):
                    return
            solutions.append((dict(offsets), union))
            return
        options = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, deltas in adjacency[j]:
                if i in offsets:
                    # adjacency[j] stores position_i - position_j.
                    proposed = {(offsets[i][0]-dr, offsets[i][1]-dc) for dr, dc in deltas}
                    domain = proposed if domain is None else domain & proposed
            if domain is not None:
                if not domain:
                    return
                options.append((len(domain), j, domain))
        if not options:
            return
        _, j, domain = min(options, key=lambda item: (item[0], item[1]))
        for position in sorted(domain):
            offsets[j] = position
            search()
            del offsets[j]

    search()
    assert len(solutions) == 1, "disconnected, inconsistent, or ambiguous features"
    offsets, union = solutions[0]
    actual = []
    for i, j, matches in edges:
        delta = offsets[j][0]-offsets[i][0], offsets[j][1]-offsets[i][1]
        selected = [m for m in matches if m[0] == delta]
        assert len(selected) == 1
        actual.append((i, j, selected[0][1]))
    return _render_tight(union, background), offsets, actual


def solve(train_pairs, test_input):
    """The examples teach translation and merging; no glyph dictionary is learned."""
    background = _background(test_input)
    return _reconstruct(_extract_components(test_input, background), background)[0]


def _analyze_pair(pair):
    inp, out = pair['input'], pair['output']
    background = _background(inp)
    components = _extract_components(inp, background)
    expected, offsets, edges = _reconstruct(components, background)
    assert np.array_equal(expected, out), "incorrect union or crop"
    assert set(np.unique(inp))-{background} == set(np.unique(out))-{background}, "colors changed"
    assert not np.array_equal(inp, out)
    placed = [{(r+offsets[i][0], c+offsets[i][1]): v for (r, c), v in part.items()}
              for i, part in enumerate(components)]
    contributions = [len(set(part)-set().union(*(set(q) for j, q in enumerate(placed) if j != i)))
                     for i, part in enumerate(placed)]
    footprints = []
    patterns = []
    for i, j, hit in edges:
        norm = _normalize({p: components[i][p] for p in hit})
        h, w = _shape(norm)
        footprints.append((len(hit), h, w, len(set(norm.values()))))
        labels = {}
        patterns.append(tuple((r, c, labels.setdefault(v, len(labels)))
                              for (r, c), v in sorted(norm.items())))
    out_cells = {(int(r), int(c)): int(out[r, c]) for r, c in zip(*np.nonzero(out != background))}
    return {'background': background, 'component_count': len(components),
            'component_sizes': [len(c) for c in components],
            'component_shapes': [_shape(c) for c in components],
            'contributions': contributions, 'overlaps': footprints,
            'patterns': patterns,
            'holes': len(_holes(out_cells)),
            'hole_sizes': [len(hole) for hole in _holes(out_cells)],
            'offsets': list(offsets.values())}


def _motif(rng, colors):
    """Independent rings, patches, bars, forks, and bent/hollow shapes."""
    h, w = int(rng.integers(2, 6)), int(rng.integers(2, 7))
    kind = int(rng.integers(0, 7))
    if kind == 0:
        h = 1
    if kind == 1:
        w = 1
    cells = {}
    a, b = [int(x) for x in rng.choice(colors, 2, replace=False)]
    style = int(rng.integers(0, 5))
    for r in range(h):
        for c in range(w):
            border = r in (0, h-1) or c in (0, w-1)
            keep = (kind <= 2 or (kind == 3 and border)
                    or (kind == 4 and (r in (0, h-1) or c == 0))
                    or (kind == 5 and (r == h//2 or c in (0, w-1)))
                    or (kind == 6 and (r == h-1 or c == 0)))
            if keep:
                color = a
                if style == 0 and (r+c) % 2:
                    color = b
                elif style == 1 and r == h//2:
                    color = b
                elif style == 2 and c == w//2:
                    color = b
                elif style == 3 and rng.random() < .24:
                    color = int(rng.choice(colors))
                elif style == 4 and (r, c) == (h//2, w//2):
                    color = b
                cells[r, c] = color
    return _normalize(cells)


def _picture(rng, colors, layout, size):
    cells, boxes = {}, []
    attempts = 0
    while len(boxes) < size and attempts < 90:
        attempts += 1
        motif = _motif(rng, colors)
        mh, mw = _shape(motif)
        if not boxes:
            top, left = 0, 0
            path = []
        else:
            parent = len(boxes)-1 if layout in ('horizontal', 'vertical', 'bend') else int(rng.integers(len(boxes)))
            r0, c0, r1, c1 = boxes[parent]
            if layout == 'horizontal':
                direction = 1
            elif layout == 'vertical':
                direction = 2
            elif layout == 'bend':
                direction = 1 if len(boxes) <= size//2 else 2
            else:
                direction = int(rng.integers(4))
            gap = int(rng.integers(1, 6))
            if direction in (1, 3):
                edge = c1 if direction == 1 else c0
                endpoints = [(r, c) for r, c in cells if c == edge and r0 <= r <= r1]
                if not endpoints:
                    continue
                ar, ac = endpoints[int(rng.integers(len(endpoints)))]
                top = ar-int(rng.integers(mh))
                left = c1+gap+1 if direction == 1 else c0-gap-mw
                side = 0 if direction == 1 else mw-1
                tips = [(r, c) for r, c in motif if c == side]
                if not tips:
                    continue
                rr, cc = tips[int(rng.integers(len(tips)))]
                br, bc = top+rr, left+cc
                bend = ac+(1 if direction == 1 else -1)*int(rng.integers(1, gap+1))
                path = [(ar, c) for c in range(min(ac, bend), max(ac, bend)+1)]
                path += [(r, bend) for r in range(min(ar, br), max(ar, br)+1)]
                path += [(br, c) for c in range(min(bend, bc), max(bend, bc)+1)]
            else:
                edge = r1 if direction == 2 else r0
                endpoints = [(r, c) for r, c in cells if r == edge and c0 <= c <= c1]
                if not endpoints:
                    continue
                ar, ac = endpoints[int(rng.integers(len(endpoints)))]
                left = ac-int(rng.integers(mw))
                top = r1+gap+1 if direction == 2 else r0-gap-mh
                side = 0 if direction == 2 else mh-1
                tips = [(r, c) for r, c in motif if r == side]
                if not tips:
                    continue
                rr, cc = tips[int(rng.integers(len(tips)))]
                br, bc = top+rr, left+cc
                bend = ar+(1 if direction == 2 else -1)*int(rng.integers(1, gap+1))
                path = [(r, ac) for r in range(min(ar, bend), max(ar, bend)+1)]
                path += [(bend, c) for c in range(min(ac, bc), max(ac, bc)+1)]
                path += [(r, bc) for r in range(min(bend, br), max(bend, br)+1)]
        box = (top, left, top+mh-1, left+mw-1)
        if any(not (box[2] < r0-1 or box[0] > r1+1 or box[3] < c0-1 or box[1] > c1+1)
               for r0, c0, r1, c1 in boxes):
            continue
        additions = {(r+top, c+left): v for (r, c), v in motif.items()}
        color = int(rng.choice(colors))
        thick = int(rng.random() < .22)+1
        for r, c in path:
            for delta in range(thick):
                p = (r+delta, c)
                # Do not paint inside an observed motif's holes.
                if p not in cells and p not in additions and not any(
                    a <= p[0] <= b and x <= p[1] <= y for a, x, b, y in boxes+[box]):
                    additions[p] = color
        proposed = dict(cells)
        proposed.update(additions)
        if max(_shape(_normalize(proposed))) > 27 or len(_connected_sets(proposed)) != 1:
            continue
        cells = proposed
        boxes.append(box)
    assert len(boxes) >= 2
    return _normalize(cells)


def _cut_fragments(cells, rng, target):
    """Overlapping rectangular views may clip motifs, bars, or whole branches."""
    fragments = [cells]
    for _ in range(target-1):
        proposals = []
        for i, part in enumerate(fragments):
            for axis in (0, 1):
                coords = [p[axis] for p in part]
                lo, hi = min(coords), max(coords)
                for width in (1, 2, 3):
                    for cut in range(lo+1, hi-width+1):
                        a = {p: v for p, v in part.items() if p[axis] < cut+width}
                        b = {p: v for p, v in part.items() if p[axis] >= cut}
                        overlap = set(a) & set(b)
                        if min(len(a), len(b)) < 5 or len(overlap) < 2:
                            continue
                        if len(a) == len(part) or len(b) == len(part):
                            continue
                        if len(_connected_sets(a)) != 1 or len(_connected_sets(b)) != 1:
                            continue
                        # Only cuts with independently recognizable registration evidence.
                        matches = _registrations(_normalize(a), _normalize(b))
                        if len(matches) != 1:
                            continue
                        amin = min(r for r, c in a), min(c for r, c in a)
                        bmin = min(r for r, c in b), min(c for r, c in b)
                        delta = bmin[0]-amin[0], bmin[1]-amin[1]
                        if matches[0][0] != delta:
                            continue
                        proposals.append((i, a, b))
        if not proposals:
            break
        # Widths and orientations compete on equal terms; no compactness score.
        i, a, b = proposals[int(rng.integers(len(proposals)))]
        fragments[i:i+1] = [a, b]
    assert len(fragments) >= 2
    return fragments


def _contained_fragment(cells, rng):
    h, w = _shape(cells)
    for _ in range(35):
        r0, c0 = int(rng.integers(h)), int(rng.integers(w))
        r1 = min(h, r0+int(rng.integers(2, 8)))
        c1 = min(w, c0+int(rng.integers(2, 8)))
        part = {p: v for p, v in cells.items() if r0 <= p[0] < r1 and c0 <= p[1] < c1}
        if 3 <= len(part) < len(cells) and len(_connected_sets(part)) == 1:
            return part
    return None


def _pack_fragments(fragments, background, rng):
    parts = [_normalize(p) for p in fragments]
    shapes = [_shape(p) for p in parts]
    maxh, maxw = max(h for h, w in shapes), max(w for h, w in shapes)
    # Random rectangle sizes, including tight, tall, wide, and 30 by 30 canvases.
    canvases = [(int(rng.integers(maxh, 31)), int(rng.integers(maxw, 31))) for _ in range(22)]
    canvases += [(30, 30)]*8
    for h, w in canvases:
        if h*w <= 2*sum(map(len, parts)):
            continue
        occupied = np.zeros((h, w), dtype=bool)
        grid = np.full((h, w), background, dtype=np.int8)
        order = sorted(range(len(parts)), key=lambda i: -(len(parts[i])+float(rng.random())*15))
        success = True
        for i in order:
            ph, pw = shapes[i]
            points = list(parts[i])
            feasible = []
            for r0 in range(h-ph+1):
                for c0 in range(w-pw+1):
                    if all(not occupied[r+r0, c+c0] for r, c in points):
                        feasible.append((r0, c0))
            if not feasible:
                success = False
                break
            r0, c0 = feasible[int(rng.integers(len(feasible)))]
            for (r, c), value in parts[i].items():
                r, c = r+r0, c+c0
                grid[r, c] = value
                occupied[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        if success:
            assert len(_extract_components(grid, background)) == len(parts)
            return grid
    raise AssertionError('fragments do not fit')


def _generate_pair(rng, regime, parameters, pair_index):
    mode = parameters['mode']
    if mode == 0:
        layout = 'horizontal' if regime == 'source_train' else str(rng.choice(['bend', 'tree']))
    elif mode == 1:
        layout = 'vertical' if regime == 'source_train' else 'horizontal'
    elif mode == 2:
        layout = parameters['layout']
    else:
        layout = str(rng.choice(['horizontal', 'vertical', 'bend', 'tree']))
    size = int(rng.integers(2, 5 if regime == 'source_train' or mode != 0 else 7))
    target = int(rng.integers(2, 5 if size < 4 else 7))
    background = parameters['background'] if rng.random() < parameters['repeat_background'] else int(rng.integers(10))
    palette_size = int(rng.integers(2, 9))
    colors = [int(x) for x in rng.choice([c for c in range(10) if c != background], palette_size, replace=False)]
    for attempt in range(45):
        try:
            cells = _picture(rng, colors, layout, size)
            height, width = _shape(cells)
            if mode == 0 and ((regime == 'source_train' and height > 7)
                              or (regime == 'target_test' and height < 9)):
                continue
            if mode == 1 and ((regime == 'source_train' and height <= width)
                              or (regime == 'target_test' and width <= height)):
                continue
            fragments = _cut_fragments(cells, rng, target)
            # The first demonstration includes absorption; later examples vary it.
            if (regime == 'source_train' and pair_index == 0) or rng.random() < .38:
                contained = _contained_fragment(cells, rng)
                if contained is not None:
                    fragments.append(contained)
                elif regime == 'source_train' and pair_index == 0:
                    continue
            expected = _render_tight(cells, background)
            predicted = _reconstruct([_normalize(p) for p in fragments], background)[0]
            if not np.array_equal(predicted, expected):
                continue
            inp = _pack_fragments(fragments, background, rng)
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    # Bounded constructive fallback: a unique nonperiodic full-color rectangle,
    # clipped into two views, one with an asymmetric extension. No seed reset.
    a, b = colors[:2]
    h = int(rng.integers(3, 6))
    cells = {(r, c): a for r in range(h) for c in range(7)}
    for r in range(h):
        cells[r, 3] = b if r in (0, h-1) else colors[-1]
    cells[h//2, 2] = b
    cells[h//2, 4] = b
    cells[h, 0] = b
    if mode == 0 and regime == 'target_test':
        cells.update({(r, 0): b for r in range(h+1, 10)})
    parts = [{p: v for p, v in cells.items() if p[1] <= 3},
             {p: v for p, v in cells.items() if p[1] >= 2}]
    if layout == 'vertical':
        cells = {(c, r): v for (r, c), v in cells.items()}
        parts = [{(c, r): v for (r, c), v in part.items()} for part in parts]
    if regime == 'source_train' and pair_index == 0:
        parts.append(dict(parts[0]))
    out = _reconstruct([_normalize(p) for p in parts], background)[0]
    assert np.array_equal(out, _render_tight(cells, background))
    return {'input': _pack_fragments(parts, background, rng), 'output': out}


def _grid_key(grid):
    return tuple(grid.shape), bytes(np.asarray(grid, dtype=np.int8).flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p['input']), _grid_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _source_pairs():
    return [{'input': np.array([[int(c) for c in row] for row in inp], dtype=np.int8),
             'output': np.array([[int(c) for c in row] for row in out], dtype=np.int8)}
            for inp, out in _SOURCE_CASES]


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
        nt, nq = len(puzzle['train']), len(puzzle['test'])
        assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
        seen = set()
        sources = {(_grid_key(p['input']), _grid_key(p['output'])) for p in _source_pairs()}
        information = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
                for grid in pair.values():
                    assert isinstance(grid, np.ndarray) and grid.ndim == 2
                    assert np.issubdtype(grid.dtype, np.integer)
                    assert all(1 <= x <= 30 for x in grid.shape)
                    assert np.all((0 <= grid) & (grid <= 9))
                key = _grid_key(pair['input'])
                assert key not in seen
                seen.add(key)
                assert (key, _grid_key(pair['output'])) not in sources
                information[split].append(_analyze_pair(pair))
        # There are no hidden palette, orientation, or glyph conventions.
        # Every split uses the same pixel-level matcher; regimes are sampling
        # distributions, not additional restrictions on a valid assembly.
        train = [{k: v.copy() for k, v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            assert np.array_equal(solve(train, pair['input'].copy()), pair['output'])
        if any(0 in info['contributions'] for info in information['test']):
            assert any(0 in info['contributions'] for info in information['train']), 'absorption not demonstrated'
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('malformed puzzle') from exc


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, nq = _resolve_counts(rng, num_train, num_test)
    parameters = {'mode': int(rng.integers(4)), 'background': int(rng.integers(10)),
                  'repeat_background': float(rng.choice([0., .35, .75, 1.])),
                  'layout': str(rng.choice(['horizontal', 'vertical', 'bend', 'tree']))}
    result = {'train': [], 'test': []}
    seen = set()
    for split, count, regime in (('train', nt, 'source_train'), ('test', nq, 'target_test')):
        for index in range(count):
            child = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(10):
                pair = _generate_pair(child, regime, parameters, index)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise AssertionError('could not create a distinct input')
    return result


def self_test() -> None:
    # These regressions intentionally bypass the synthetic train/test counts.
    for pair in _source_pairs():
        _analyze_pair(pair)
        _analyze_pair({k: v.T.copy() for k, v in pair.items()})
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    pairs = [p for puzzle in puzzles for s in ('train', 'test') for p in puzzle[s]]
    infos = [_analyze_pair(p) for p in pairs]
    assert len({p['input'].shape for p in pairs}) >= 30
    assert len({p['output'].shape for p in pairs}) >= 40
    assert len({i['component_count'] for i in infos}) >= 4
    assert len({len(np.unique(p['input'])) for p in pairs}) >= 5
    assert len({i['background'] for i in infos}) == 10
    assert sum(i['holes'] > 0 for i in infos) >= 10
    assert any(0 in i['contributions'] for i in infos)
    assert any(min(i['component_sizes']) < 9 for i in infos)
    assert any(max(max(s) for s in i['component_shapes']) > 9 for i in infos)
    assert any(p['output'].shape[1] > p['input'].shape[1] for p in pairs)
    assert any(n == 2 for i in infos for n, h, w, c in i['overlaps'])
    assert any(c == 1 for i in infos for n, h, w, c in i['overlaps'])
    assert any(min(h, w) == 1 for i in infos for n, h, w, c in i['overlaps'])
    assert any(n < h*w for i in infos for n, h, w, c in i['overlaps'])
    assert sum(len(set(i['patterns'])) > 1 for i in infos) >= 80
    assert len({pattern for i in infos for pattern in i['patterns']}) >= 80
    assert any(i['holes'] and 0 in i['contributions']
               and any(min(h, w) == 1 for n, h, w, c in i['overlaps']) for i in infos)
    assert any(i['holes'] and i['component_count'] >= 5 for i in infos)
    assert any(_has_branch(p['output'], i['background']) for p, i in zip(pairs, infos))
    assert sum(len({p[0] for p in i['offsets']}) > 1
               and len({p[1] for p in i['offsets']}) > 1 for i in infos) >= 80
    assert sum(any(len(set(np.diff(sorted({p[axis] for p in i['offsets']})))) > 1
                   for axis in (0, 1)) for i in infos) >= 50
    assert sum(len({p['output'].shape for p in puzzle['train']}) > 1 for puzzle in puzzles) >= 38
    assert sum(len({len(np.unique(p['input'])) for p in puzzle['train']}) > 1
               for puzzle in puzzles) >= 30
    assert any(max(p['output'].shape[0] for p in puzzle['train'])
               < min(p['output'].shape[0] for p in puzzle['test']) for puzzle in puzzles)
    assert any(all(p['output'].shape[0] > p['output'].shape[1] for p in puzzle['train'])
               and all(p['output'].shape[1] > p['output'].shape[0] for p in puzzle['test'])
               for puzzle in puzzles)
    # Compact repacking, independent palette changes, and complete-example
    # orientation changes preserve the rule without synthetic split schedules.
    p = _source_pairs()[0]
    bg = _background(p['input'])
    parts = _extract_components(p['input'], bg)
    repacked = _pack_fragments(parts, bg, np.random.default_rng(27))
    _analyze_pair({'input': repacked, 'output': p['output']})
    compact = np.full((19, 11), bg, dtype=np.int8)
    for part, top in zip(parts, (0, 6, 12)):
        for (r, c), color in part.items():
            compact[r+top, c] = color
    _analyze_pair({'input': compact, 'output': p['output']})
    changed = {k: v.copy() for k, v in p.items()}
    motif = np.array([[3, 3, 3], [3, 2, 3], [3, 3, 3]])
    replacements = []
    for grid in changed.values():
        count = 0
        for r in range(grid.shape[0]-2):
            for c in range(grid.shape[1]-2):
                if np.array_equal(grid[r:r+3, c:c+3], motif):
                    grid[r+1, c+1] = 6
                    count += 1
        replacements.append(count)
    assert replacements == [2, 1]
    _analyze_pair(changed)
    mapping = np.array([7, 5, 1, 9, 2, 0, 4, 3, 8, 6], dtype=np.int8)
    _analyze_pair({k: mapping[v] for k, v in p.items()})
    bad = {k: v.copy() for k, v in p.items()}
    bad['output'][0, 0] = bg
    try:
        _analyze_pair(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('lost foreground accepted')
    for seed, nt, nq in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                         (200, None, 1), (201, None, 4), (202, 2, None), (203, 10, None)):
        p = generate_puzzle(seed, nt, nq)
        assert nt is None or len(p['train']) == nt
        assert nq is None or len(p['test']) == nq
        validate_generated_puzzle(p)
    assert _puzzle_key(generate_puzzle(99, 3, 1)) == _puzzle_key(generate_puzzle(99, 3, 1))
    assert _puzzle_key(generate_puzzle(99, 3, 1)) != _puzzle_key(generate_puzzle(98, 3, 1))
    for nt, nq in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (3.2, 1)):
        try:
            generate_puzzle(7, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # A repeated bar with two indistinguishable placements is unresolved.
    a = {(0, c): 1 for c in range(5)}
    b = {(0, c): 1 for c in range(3)}
    try:
        _reconstruct([a, b], 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous registration accepted')


def _check_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        n_test = int(rng.integers(1, 5))
        n_train = int(rng.integers(n_test + 1, 11))
    elif num_train is None:
        n_test = int(num_test)
        n_train = int(rng.integers(max(2, n_test + 1), 11))
    elif num_test is None:
        n_train = int(num_train)
        n_test = int(rng.integers(1, min(4, n_train - 1) + 1))
    else:
        n_train = int(num_train)
        n_test = int(num_test)

    if n_train <= n_test:
        raise ValueError("num_train must be greater than num_test")

    return n_train, n_test



# The four original pairs are rule regressions, separate from synthetic counts.
_SOURCE_CASES = (
    (
        (
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8888889888888888888888888',
            '8888881188282333888888888',
            '8888884111222323888888888',
            '8888881188282333888888888',
            '8888889888888888888888888',
            '8888888888888888888868988',
            '8888888888888888888861188',
            '8888888888888888888861488',
            '8888888888888888888861188',
            '8888888888888888888868988',
            '8888888888888888888888888',
            '8888888888888888888888888',
            '8883338888888888888888888',
            '8883238888888888888888888',
            '8883338888888888888888888',
            '8888888888888888888888888',
            '8888888888888888888888888',
        ),
        (
            '689888888888',
            '611188282333',
            '614111222323',
            '611188282333',
            '689888888888',
        ),
    ),
    (
        (
            '44444444444444444444',
            '44444444444441111444',
            '44444444448111331444',
            '41114444448441441444',
            '41214444448111331444',
            '41114444444441111444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444111444444453544',
            '44444121333333335344',
            '44444111444444453544',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44444444444444444444',
            '44466644444444444444',
            '44535668444444444444',
            '44353668444444444444',
            '44535668444444444444',
            '44466644444444444444',
            '44444444444444444444',
            '44444444444444444444',
        ),
        (
            '4444444444466644441111',
            '1114444444535668111331',
            '1213333333353668441441',
            '1114444444535668111331',
            '4444444444466644441111',
        ),
    ),
    (
        (
            '33333333333333333333',
            '33661111333333336633',
            '33861338883339996833',
            '33661111783339336633',
            '33333338883339333333',
            '33333333333311133333',
            '33333333333317133333',
            '33333333333333333333',
            '33333333333333333333',
            '33333333333333333333',
            '34443333317833333333',
            '33433333388833333333',
            '33433333337333333333',
            '31113333337333333333',
            '31613333337333333333',
            '33333333322233333333',
            '33333333333333333333',
            '33333333333333333333',
            '33333333311133311133',
            '33332323316133317133',
            '33334443338333338333',
            '33344144338333338333',
            '33334443338333338333',
            '33333333338888888333',
            '33333333333323233333',
            '33333333333333333333',
        ),
        (
            '3333333333666111133',
            '4443333999686133888',
            '3433333933666111178',
            '3433333933333333888',
            '1113331113333333373',
            '1613331713333333373',
            '3833333833333333373',
            '3833333833333333222',
            '3833333833333333333',
            '3888888833333333333',
            '3332323333333333333',
            '3334443333333333333',
            '3344144333333333333',
            '3334443333333333333',
        ),
    ),
    (
        (
            '11111111111111111111',
            '11111112221111111111',
            '11111112421111322111',
            '11111112221111121111',
            '11111111111111322111',
            '12221111111111111111',
            '12421111111111111111',
            '12221111144416161888',
            '11311111147488888878',
            '99399111144481118888',
            '19991111111111111111',
            '16161111111111111111',
            '11111111111111111111',
            '11111111188833311111',
            '11111111187833111111',
            '11164441188833311111',
            '16664741111111111111',
            '11164441111111111111',
            '11111111111111111111',
            '11111111111111111111',
        ),
        (
            '1111111222111111111',
            '1111111242111111111',
            '1111111222111111111',
            '1111111131111111111',
            '1111119939911111111',
            '1111111999111111111',
            '1164441616188833322',
            '6664748888887833121',
            '1164448111888833322',
        ),
    ),
)
