"""Census arbitration: local object counts choose simultaneous color transfers.

Only NumPy and the standard library are used. Generation labels are calculated
from the construction plan; solve reconstructs territories and counts from pixels.
"""
from collections import Counter, defaultdict
import numpy as np

INK = 9
D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
SINGLE = ((0, 0),)
HOST_SEEDS = (0, 514118705, 689911282, 61312622, 696177449, 222314107,
              802537478, 786753408, 381150407, 415944164, 367573963,
              931367867, 392516179, 880867977, 245994236, 99309936,
              428065826, 28256679, 842200604, 842827110, 170677779,
              839508087, 2934233, 498499020, 935523884, 68911822,
              673688401, 56591841, 855372908, 929688536, 595722338,
              571517111, 985665017, 597175367, 91364757, 244450770,
              707406423, 428071102, 355140826, 672755602, 185975357,
              321683435)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _need(np.all((0 <= a) & (a <= 9)), 'invalid grid colors')
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _components(mask, diagonal=False):
    """Deterministic flood fill of one Boolean mask."""
    seen = np.zeros(mask.shape, bool)
    parts = []
    directions = D8 if diagonal else D4
    for r, c in np.argwhere(mask):
        p = int(r), int(c)
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], []
        while stack:
            q = stack.pop()
            cells.append(q)
            for d in directions:
                n = _add(q, d)
                if _inside(n, mask.shape) and mask[n] and not seen[n]:
                    seen[n] = True
                    stack.append(n)
        parts.append(cells)
    return parts


def _territories(clean):
    ids = np.full(clean.shape, -1, int)
    colors, cells = [], []
    for color in np.unique(clean):
        for part in _components(clean == color):
            i = len(colors)
            colors.append(int(color))
            cells.append(part)
            for p in part:
                ids[p] = i
    return ids, colors, cells


def _geometry(center, direction, shape):
    perp = -direction[1], direction[0]
    arms = (_add(center, perp), _add(center, perp, -1))
    tip = _add(center, direction)
    targetcell = _add(center, direction, 2)
    flanks = (_add(tip, perp), _add(tip, perp, -1))
    write = {center, tip, *arms}
    if not all(_inside(p, shape) for p in write | {targetcell, *flanks}):
        return None
    return dict(center=center, direction=direction, arms=arms, tip=tip,
                targetcell=targetcell, flanks=flanks, write=write)


def _paint(ids, colors, arrows, strengths=None, payloads=None, aggregate=False):
    """Arbitrate against original component IDs and original source colors."""
    incoming = defaultdict(list)
    for i, arrow in enumerate(arrows):
        strength = arrow['strength'] if strengths is None else strengths[i]
        paint = colors[arrow['source']] if payloads is None else payloads[i]
        incoming[arrow['target']].append((strength, paint))
    final = list(colors)
    for target, proposals in incoming.items():
        if aggregate:
            votes = defaultdict(int)
            for strength, color in proposals:
                votes[color] += strength
            proposals = [(strength, color) for color, strength in votes.items()]
        maximum = max(score for score, _ in proposals)
        winners = {color for score, color in proposals if score == maximum}
        if len(winners) == 1:
            final[target] = next(iter(winners))
    return np.asarray(final, dtype=np.int8)[ids], final


def _inspect(input_grid):
    grid = _grid(input_grid)
    # This is a visible scale convention: glyphs have <=8 cells, while each
    # territory has a surviving connected fill patch containing >=10 cells.
    fills = {int(c) for c in np.unique(grid) if c != INK
             and any(len(part) >= 10 for part in _components(grid == c))}
    _need(len(fills) >= 2, 'territory fills are not recoverable')
    arrows, covered, writes = [], set(), set()
    for r, c in np.argwhere(~np.isin(grid, list(fills | {INK}))):
        center = int(r), int(c)
        for direction in D4:
            geo = _geometry(center, direction, grid.shape)
            if geo is None or not all(grid[p] == INK for p in (*geo['arms'], geo['tip'])):
                continue
            a, b = geo['flanks']
            host = int(grid[a])
            if host not in fills or grid[b] != host or grid[geo['targetcell']] not in fills:
                continue
            if grid[geo['targetcell']] == host:
                continue
            _need(not writes & geo['write'], 'overlapping T annotations')
            writes.update(geo['write'])
            covered.update((*geo['arms'], geo['tip']))
            arrows.append(dict(geo, key=int(grid[center]), host=host))
    _need(arrows and covered == set(map(tuple, np.argwhere(grid == INK))), 'unexplained T ink')
    clean = grid.copy()
    key_mask = np.zeros(grid.shape, bool)
    for arrow in arrows:
        key_mask[arrow['center']] = True
        for p in arrow['write']:
            clean[p] = arrow['host']
    objects = []
    for color in sorted(set(map(int, np.unique(grid))) - fills - {INK}):
        for cells in _components((grid == color) & ~key_mask, diagonal=True):
            _need(1 <= len(cells) <= 8, 'glyph size outside 1..8')
            neighbor_fills = {int(grid[n]) for p in cells for d in D4
                              if _inside(n := _add(p, d), grid.shape) and grid[n] in fills}
            _need(len(neighbor_fills) == 1, 'object host is ambiguous')
            host = next(iter(neighbor_fills))
            for p in cells:
                clean[p] = host
            objects.append(dict(color=color, cells=cells, host=host))
    _need(set(map(int, np.unique(clean))) <= fills, 'uncleaned annotations')
    ids, colors, cells = _territories(clean)
    _need(3 <= len(colors) <= 6, 'family uses three to six territories')
    counts, areas, orth = Counter(), Counter(), Counter()
    object_owner = np.full(grid.shape, -1, int)
    for i, obj in enumerate(objects):
        owners = {int(ids[p]) for p in obj['cells']}
        _need(len(owners) == 1, 'object crosses a territory boundary')
        obj['source'] = source = owners.pop()
        r0, r1 = min(p[0] for p in obj['cells']), max(p[0] for p in obj['cells'])
        c0, c1 = min(p[1] for p in obj['cells']), max(p[1] for p in obj['cells'])
        _need(r0 > 0 and c0 > 0 and r1 < grid.shape[0]-1 and c1 < grid.shape[1]-1,
              'objects need a complete host border')
        _need(np.all(ids[r0-1:r1+2, c0-1:c1+2] == source), 'object lacks homogeneous host halo')
        counts[source, obj['color']] += 1
        areas[source, obj['color']] += len(obj['cells'])
        mask = np.zeros(grid.shape, bool)
        for p in obj['cells']:
            object_owner[p] = i
            mask[p] = True
        orth[source, obj['color']] += len(_components(mask))
    # Different objects and arrows have a Chebyshev-distance separation.
    for obj in objects:
        for p in obj['cells']:
            i = object_owner[p]
            for d in D8:
                n = _add(p, d)
                if _inside(n, grid.shape):
                    _need(object_owner[n] in (-1, i) and n not in writes, 'touching objects/keys')
    directed = set()
    for arrow in arrows:
        source = int(ids[arrow['center']])
        target = int(ids[arrow['targetcell']])
        _need(source != target and all(ids[p] == source for p in arrow['write'] | set(arrow['flanks'])),
              'invalid T crossing')
        _need((source, target) not in directed, 'duplicate directed territory edge')
        directed.add((source, target))
        strength = counts[source, arrow['key']]
        _need(strength >= 1, 'each selected category needs a local object')
        arrow.update(source=source, target=target, strength=strength)
    expected, final = _paint(ids, colors, arrows)
    return dict(grid=grid, clean=clean, ids=ids, colors=colors, cells=cells,
                arrows=arrows, objects=objects, counts=counts, areas=areas,
                orth=orth, final=final, expected=expected)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the input's visible geometry and color equality."""
    return _inspect(input_grid)['expected'].copy()


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _partition_signature(ids):
    """Rename IDs in reading order so palette permutations cannot inflate diversity."""
    names = {}
    normalized = np.array([names.setdefault(int(i), len(names)) for i in ids.ravel()], np.int8)
    return ids.shape, normalized.tobytes()


def _partition(rng, mode, count, rough):
    """Pixel-coordinate bands, independently split bands, or nested rings."""
    if mode == 'rings':
        h, w = int(rng.integers(26, 31)), int(rng.integers(26, 31))
        ids = np.zeros((h, w), int)
        outer = int(rng.integers(5, 7))
        inner = outer + int(rng.integers(4, 6))
        ids[outer:h-outer, outer:w-outer] = 1
        ids[inner:h-inner, inner:w-inner] = 2
        return ids
    bands = 2 if count == 4 else 3
    h = int(rng.integers(23, 31)) if count >= 4 else int(rng.integers(18, 31))
    w = int(rng.integers(24, 31))
    if bands == 2:
        widths = [int(rng.integers(11, w-10)), 0]
        widths[1] = w-widths[0]
    else:
        widths = [7, 7, 7]
        for _ in range(w-21):
            widths[int(rng.integers(3))] += 1
    cuts = np.cumsum(widths)[:-1]
    paths = [np.full(h, cut, int) for cut in cuts]
    if rough:
        for path, cut in zip(paths, cuts):
            step = int(rng.integers(4, 8))
            for r in range(0, h, step):
                path[r:r+step] = cut + int(rng.integers(-1, 2))
    ids = np.zeros((h, w), int)
    for r in range(h):
        for b, path in enumerate(paths):
            ids[r, path[r]:] = b+1
    if count > bands:
        split_bands = list(map(int, rng.choice(bands, count-bands, replace=False)))
        next_id = bands
        for band in split_bands:
            cut = int(rng.integers(10, h-9))
            path = np.full(w, cut)
            if rough and rng.random() < .5:
                path[w//2:] += int(rng.choice((-1, 1)))
            rr, cc = np.indices(ids.shape)
            ids[(ids == band) & (rr >= path[cc])] = next_id
            next_id += 1
    return ids


def _candidates(ids):
    by_edge = defaultdict(list)
    for center in np.ndindex(ids.shape):
        for direction in D4:
            geo = _geometry(center, direction, ids.shape)
            if geo is None:
                continue
            source, target = int(ids[center]), int(ids[geo['targetcell']])
            if source != target and all(ids[p] == source for p in geo['write'] | set(geo['flanks'])):
                by_edge[source, target].append(dict(geo, source=source, target=target))
    return by_edge


def _normalize(cells):
    lo_r, lo_c = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-lo_r, c-lo_c) for r, c in cells))


def _glyph(rng):
    prototypes = (SINGLE, SQUARE, DIAMOND,
                  ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
                  ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
                  ((0, 0), (0, 1)), ((0, 0), (1, 0), (1, 1)))
    if rng.random() < .65:
        cells = prototypes[int(rng.integers(len(prototypes)))]
    else:
        cells = {(1, 1)}
        target = int(rng.integers(2, 9))
        while len(cells) < target:
            frontier = sorted({_add(p, d) for p in cells for d in D8
                               if 0 <= _add(p, d)[0] < 3 and 0 <= _add(p, d)[1] < 3} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(cells)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    if rng.random() < .5:
        cells = tuple((r, -c) for r, c in cells)
    return _normalize(cells)


def _halo(points, shape):
    return {n for p in points for d in ((0, 0), *D8)
            if _inside(n := _add(p, d), shape)}


def _place(rng, ids, blocked, source, shape):
    sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
    candidates = []
    for r in range(1, ids.shape[0]-sh):
        for c in range(1, ids.shape[1]-sw):
            if not np.all(ids[r-1:r+sh+1, c-1:c+sw+1] == source):
                continue
            points = {(r+dr, c+dc) for dr, dc in shape}
            if not points & blocked:
                candidates.append(points)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _build(rng, kind, regime):
    witness = kind in ('winner', 'tie')
    count = 4 if kind == 'winner' else (3 if kind == 'tie' else int(rng.integers(3, 5)))
    rough = bool(rng.random() < .4)
    mode = 'bands'
    cap = 3
    if kind == 'test':
        if regime == 0:
            count, rough = int(rng.integers(5, 7)), True
        elif regime == 1:
            cap, rough = 5, bool(rng.integers(2))
        else:
            mode, count, rough = 'rings', 3, True
    elif kind == 'extra' and regime != 2 and rng.random() < .18:
        mode, count = 'rings', 3
    ids = _partition(rng, mode, count, rough)
    count = int(ids.max())+1
    edges = _candidates(ids)
    neighbors = {i: sorted(t for s, t in edges if s == i) for i in range(count)}
    targets = [i for i in range(count) if len(neighbors[i]) >= 2]
    if kind == 'winner':
        targets = [i for i in targets if any(j != i and j not in neighbors[i] for j in range(count))]
    if not targets:
        return None
    target = int(rng.choice(targets))
    a, b = map(int, rng.choice(neighbors[target], 2, replace=False))
    palette = list(map(int, rng.permutation(9)))
    colors = palette[:count]
    cats = palette[count:count+3]
    if len(cats) < 2:
        return None
    if kind == 'winner':
        untouched = next(j for j in range(count) if j != target and j not in neighbors[target])
        colors[untouched] = colors[target]
    elif not witness and rng.random() < .25:
        pairs = [(i, j) for i in range(count) for j in range(i)
                 if j not in neighbors[i] and (i, j) != (a, b) and (j, i) != (a, b)]
        if pairs:
            i, j = pairs[int(rng.integers(len(pairs)))]
            colors[i] = colors[j]
    # Core: two proposals into one node; that node also sends its original fill.
    planned = [(a, target, cats[0]), (b, target, cats[0]), (target, b, cats[0])]
    if not witness:
        available = [edge for edge in edges if edge not in {(s, t) for s, t, k in planned}]
        rng.shuffle(available)
        desired = int(rng.integers(3, min(8, len(edges))+1))
        for s, t in available[:desired-3]:
            planned.append((s, t, int(rng.choice(cats))))
    counts = {(s, key): int(rng.integers(1, cap+1)) for s, t, key in planned}
    if kind == 'winner':
        counts[a, cats[0]], counts[b, cats[0]], counts[target, cats[0]] = 3, 2, 1
    elif kind == 'tie':
        counts[a, cats[0]] = counts[b, cats[0]] = 2
        counts[target, cats[0]] = 1
    elif kind == 'test' and regime == 1:
        counts[a, cats[0]] = int(rng.integers(4, 6))
    if any(sum(n for (s, k), n in counts.items() if s == i) > 7 for i in range(count)):
        return None
    arrows, blocked, context = [], set(), set()
    for source, dest, key in planned:
        choices = []
        for arrow in edges[source, dest]:
            support = set(arrow['flanks']) | {arrow['targetcell']}
            if not (arrow['write'] & (blocked | context) or support & {p for x in arrows for p in x['write']}):
                choices.append(arrow)
        if not choices:
            return None
        arrow = dict(choices[int(rng.integers(len(choices)))], key=key, strength=counts[source, key])
        arrows.append(arrow)
        blocked.update(_halo(arrow['write'], ids.shape))
        context.update(arrow['flanks'])
        context.add(arrow['targetcell'])
    clean = np.asarray(colors, np.int8)[ids]
    grid = clean.copy()
    for arrow in arrows:
        for p in arrow['write']:
            grid[p] = INK
        grid[arrow['center']] = arrow['key']
    objects = []
    for (source, color), n in counts.items():
        for i in range(n):
            if kind == 'winner' and source == a:
                shape = SINGLE
            elif kind == 'winner' and source == b:
                shape = (DIAMOND, SQUARE)[i]
            else:
                shape = _glyph(rng)
            objects.append((source, color, shape))
    if kind == 'winner':
        # More unkeyed objects in the losing source defeat all-object counting.
        objects.extend((b, cats[1], SINGLE) for _ in range(4))
    else:
        for source in range(count):
            if rng.random() < .4:
                unused = [c for c in cats if (source, c) not in counts]
                if unused:
                    for _ in range(int(rng.integers(1, 3))):
                        objects.append((source, int(rng.choice(unused)), _glyph(rng)))
    rng.shuffle(objects)
    objects.sort(key=lambda obj: -(max(r for r, c in obj[2])+1)*(max(c for r, c in obj[2])+1))
    for source, color, shape in objects:
        points = _place(rng, ids, blocked | context, source, shape)
        if points is None:
            return None
        for p in points:
            grid[p] = color
        blocked.update(_halo(points, ids.shape))
    # Construction labels use sampled object counts, IDs and original colors.
    output, _ = _paint(ids, colors, arrows)
    arrow_tags = np.full(ids.shape, -1, int)
    for index, arrow in enumerate(arrows):
        arrow_tags[arrow['center']] = index
    rotation = int(rng.integers(4))
    if rng.random() < .5:
        grid, output, clean, arrow_tags = map(np.fliplr, (grid, output, clean, arrow_tags))
    grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
    clean, arrow_tags = np.rot90(clean, rotation), np.rot90(arrow_tags, rotation)
    try:
        meta = _inspect(grid)
        if (not np.array_equal(meta['expected'], output) or not np.array_equal(meta['clean'], clean)
                or len(meta['colors']) != count or len(meta['arrows']) != len(arrows)):
            return None
        for decoded in meta['arrows']:
            index = int(arrow_tags[decoded['center']])
            if (index < 0 or decoded['key'] != arrows[index]['key']
                    or decoded['strength'] != arrows[index]['strength']):
                return None
        if kind == 'winner':
            f = _features(meta)
            if not all(f[k] for k in ('four_diff', 'area_diff', 'all_diff', 'global_diff', 'locality', 'chain')):
                return None
        if kind == 'tie' and not _features(meta)['tie']:
            return None
    except AssertionError:
        return None
    return {'input': grid, 'output': output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer from 2 to 10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be 1..4 and smaller than num_train')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    kinds = ['winner', 'tie'] + ['extra']*(int(num_train)-2)
    taught = set()
    for split, requests in (('train', kinds), ('test', ['test']*int(num_test))):
        for kind in requests:
            for _ in range(160):
                pair = _build(rng, kind, regime)
                if pair is not None:
                    features = _features(_inspect(pair['input']))
                    # Max-vs-sum and agreeing ties need a demonstration before
                    # they can become a test distinction in this finite puzzle.
                    if split == 'test' and any(features[k] and k not in taught
                                                for k in ('sum_diff', 'agreement')):
                        continue
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    if split == 'train':
                        taught.update(k for k, v in features.items() if v)
                    break
            else:
                raise RuntimeError('bounded construction could not place a census scene')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(meta):
    arrows, colors = meta['arrows'], meta['colors']
    expected = meta['expected']
    def different(strengths=None, payloads=None, aggregate=False):
        out, _ = _paint(meta['ids'], colors, arrows, strengths, payloads, aggregate)
        return not np.array_equal(out, expected)
    global_counts = Counter()
    totals = Counter()
    for (source, color), n in meta['counts'].items():
        global_counts[color] += n
        totals[source] += n
    incoming = defaultdict(list)
    for arrow in arrows:
        incoming[arrow['target']].append(arrow)
    tie = any(len({colors[a['source']] for a in group
                   if a['strength'] == max(b['strength'] for b in group)}) > 1
              for group in incoming.values())
    agreement = any(len([a for a in group if a['strength'] == max(b['strength'] for b in group)]) > 1
                    and len({colors[a['source']] for a in group
                             if a['strength'] == max(b['strength'] for b in group)}) == 1
                    and meta['final'][target] != colors[target]
                    for target, group in incoming.items())
    chain = any(meta['final'][a['source']] != colors[a['source']]
                and meta['final'][a['target']] == colors[a['source']]
                and colors[a['source']] != colors[a['target']] for a in arrows)
    locality = any(colors[i] == colors[j] and meta['final'][i] != meta['final'][j]
                   for i in range(len(colors)) for j in range(i))
    return dict(tie=tie, agreement=agreement, chain=chain, locality=locality,
                four_diff=different([meta['orth'][a['source'], a['key']] for a in arrows]),
                area_diff=different([meta['areas'][a['source'], a['key']] for a in arrows]),
                all_diff=different([totals[a['source']] for a in arrows]),
                global_diff=different([global_counts[a['key']] for a in arrows]),
                unit_diff=different([1]*len(arrows)),
                junction_diff=different(payloads=[a['key'] for a in arrows]),
                sum_diff=different(aggregate=True),
                without_a=not np.array_equal(meta['clean'], expected),
                nonrect=any(len(cells) != (max(r for r, c in cells)-min(r for r, c in cells)+1)
                            *(max(c for r, c in cells)-min(c for r, c in cells)+1) for cells in meta['cells']))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'invalid pair counts')
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            meta = _inspect(inp)
            _need(inp.shape == out.shape and np.array_equal(meta['expected'], out), 'incorrect census transfer')
            _need(INK not in out and not np.array_equal(inp, out), 'annotation cleanup failed')
            for cells in meta['cells']:
                visible = np.zeros(inp.shape, bool)
                for p in cells:
                    visible[p] = inp[p] == meta['clean'][p]
                _need(any(len(c) >= 10 for c in _components(visible)), 'territory lacks a visible fill patch')
            if split == 'train':
                training.append(_features(meta))
            else:
                testing.append(_features(meta))
    for feature in ('tie', 'chain', 'locality', 'four_diff', 'area_diff', 'all_diff',
                    'global_diff', 'unit_diff', 'junction_diff', 'without_a'):
        _need(any(f[feature] for f in training), 'missing training distinction: '+feature)
    for feature in ('sum_diff', 'agreement'):
        _need(not any(f[feature] for f in testing) or any(f[feature] for f in training),
              'test distinction absent from training: '+feature)
    return None


def _semantic_fixture_checks():
    """Hand-designed pixel fixtures; expectations do not use _paint or solve."""
    ids = np.ones((28, 28), int)
    ids[:, :9], ids[:, 19:] = 0, 2
    ids[14:, :9], ids[14:, 19:] = 3, 4
    original = [1, 3, 2, 2, 4]
    grid = np.asarray(original, np.int8)[ids]
    for center, direction in (((4, 7), (0, 1)), ((5, 20), (0, -1)),
                              ((20, 7), (0, 1)), ((11, 17), (0, 1))):
        geometry = _geometry(center, direction, grid.shape)
        for p in geometry['write']:
            grid[p] = INK
        grid[center] = 0
    for p in ((1, 1), (7, 1), (10, 3), (17, 2), (24, 4), (18, 13)):
        grid[p] = 0
    for r, c in DIAMOND:
        grid[1+r, 23+c] = 0
    for r, c in SQUARE:
        grid[8+r, 24+c] = 0
    for p in ((5, 25), (6, 23), (11, 22), (11, 26)):
        grid[p] = 5
    expected = np.asarray([1, 1, 3, 2, 4], np.int8)[ids]
    meta = _inspect(grid)
    _need(np.array_equal(solve(grid), expected), 'three-way maximum fixture failed')
    _need(_features(meta)['sum_diff'], 'fixture does not distinguish summing')
    _need(_features(meta)['chain'], 'fixture lacks simultaneous dependency')
    # A losing color gets two strongest, equal proposals. They agree on paint.
    agreeing = grid.copy()
    agreeing[7, 1] = agreeing[10, 3] = 1
    expected_agreeing = np.asarray([1, 2, 3, 2, 4], np.int8)[ids]
    _need(np.array_equal(solve(agreeing), expected_agreeing), 'agreeing maximum fixture failed')
    _need(_features(_inspect(agreeing))['agreement'], 'fixture lacks agreeing tie')
    for source, result in ((grid, expected), (agreeing, expected_agreeing)):
        for rotation in range(4):
            # Orientation and color-role changes cannot affect voting.
            _need(np.array_equal(solve(np.rot90(source, rotation)), np.rot90(result, rotation)),
                  'fixture rotation failed')
        palette = np.array([8, 7, 6, 5, 4, 3, 2, 1, 0, 9], np.int8)
        _need(np.array_equal(solve(palette[source]), palette[result]), 'role permutation failed')


def self_test():
    """Bounded family tests over 42 specified seeds; no external semantic oracle."""
    _semantic_fixture_checks()
    signatures, partitions, shapes, glyphs = set(), set(), set(), set()
    counts, arrow_counts, directions = set(), set(), set()
    stats = Counter()
    palettes = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                meta = _inspect(pair['input'])
                _need(np.array_equal(solve(pair['input']), pair['output']), 'input-only solution failed')
                partitions.add(_partition_signature(meta['ids']))
                shapes.add(pair['input'].shape)
                counts.add(len(meta['colors']))
                arrow_counts.add(len(meta['arrows']))
                directions.update(a['direction'] for a in meta['arrows'])
                glyphs.update(_normalize(o['cells']) for o in meta['objects'])
                palettes.add(tuple(sorted(set(meta['colors']))))
                stats.update(k for k, v in _features(meta).items() if v)
    _need(len(signatures) == 42, 'seeds collapsed')
    _need(len(partitions) >= 120 and len(shapes) >= 30, 'insufficient structural variation')
    _need(counts >= {3, 4, 5, 6} and len(arrow_counts) >= 4, 'insufficient count variation')
    _need(directions == set(D4), 'missing arrow directions')
    _need(len(glyphs) >= 35 and len(palettes) >= 30, 'insufficient shape/palette variation')
    _need(stats['nonrect'] >= 60 and stats['tie'] >= 42, 'missing irregular/tied scenes')
    for seed in (0, 514118705):
        p, q = generate_puzzle(seed), generate_puzzle(seed)
        _need(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a), 'not deterministic')
    # Error checks must reject wrong labels, unsupported values and bad counts.
    bad = generate_puzzle(0, 2, 1)
    bad['test'][0]['output'][0, 0] = INK
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'partitions': len(partitions),
            'dimensions': len(shapes), 'glyphs': len(glyphs), 'territory_counts': sorted(counts),
            'arrow_counts': sorted(arrow_counts), 'feature_pairs': dict(stats)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
