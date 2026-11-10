"""Color quorum: collect simultaneous directed messages, then erase full territories.

No parent generator, saved pair, seed, or construction metadata is used by solve.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _view(a, corner):
    if corner & 2:
        a = a[::-1]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _move(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, excluded=None):
    """Equal-color, orthogonally connected components; excluded cells get ID -1."""
    ids = np.full(a.shape, -1, dtype=np.int16)
    forbidden = np.zeros(a.shape, dtype=bool) if excluded is None else excluded
    cells, colors = [], []
    for start in np.ndindex(a.shape):
        if forbidden[start] or ids[start] >= 0:
            continue
        index, color = len(cells), int(a[start])
        queue = [start]
        ids[start] = index
        for p in queue:
            for d in DIRS:
                n = _move(p, d)
                if (_inside(n, a.shape) and not forbidden[n]
                        and ids[n] < 0 and a[n] == color):
                    ids[n] = index
                    queue.append(n)
        cells.append(queue)
        colors.append(color)
    return ids, colors, cells


def _geometry(p, d, shape):
    side = (-d[1], d[0])
    tip = _move(p, d)
    arms = (_move(p, side), _move(p, side, -1))
    flanks = (_move(tip, side), _move(tip, side, -1))
    target = _move(p, d, 2)
    write = frozenset((p, tip, *arms))
    context = frozenset((*flanks, target))
    if not all(_inside(x, shape) for x in write | context):
        return None
    return dict(center=p, direction=d, tip=tip, arms=arms, flanks=flanks,
                target_cell=target, write=write, context=context)


def _decode_view(a, corner):
    """Decode one proposed selector orientation using visible geometry only."""
    v = _view(a, corner)
    _require(min(v.shape) >= 8, "canvas too small for selector")
    ink = int(v[3, 3])
    _require(bool(np.all(v[3, :4] == ink) and np.all(v[:4, 3] == ink)), "no selector L")
    values, counts = np.unique(v[:3, :3], return_counts=True)
    padding = values[counts == 6]
    _require(len(padding) == 1, "selector needs six padding cells")
    bg = int(padding[0])
    query = frozenset(int(x) for x in values if x != bg)
    _require(len(query) == 3 and ink != bg and ink not in query, "invalid query roles")
    reserved = np.zeros(v.shape, dtype=bool)
    reserved[:5, :5] = True
    _require(bool(np.all(v[4, :5] == bg) and np.all(v[:5, 4] == bg)), "missing selector moat")
    # The field is a complete territory partition. Background occurs only in
    # the 5x5 selector reservation before erasure.
    _require(not np.any((v == bg) & ~reserved), "background hole in input field")
    ink_cells = {tuple(map(int, p)) for p in np.argwhere((v == ink) & ~reserved)}
    _require(ink_cells, "no message arrows")
    arrows = []
    for p in sorted({_move(x, d) for x in ink_cells for d in DIRS}):
        for d in DIRS:
            g = _geometry(p, d, v.shape)
            if g is None or any(reserved[x] for x in g['write'] | g['context']):
                continue
            if not all(v[x] == ink for x in (*g['arms'], g['tip'])):
                continue
            left, right = g['flanks']
            host = int(v[left])
            if host in (ink, bg) or v[right] != host:
                continue
            if int(v[g['target_cell']]) in (ink, bg, host):
                continue
            payload = host if v[p] == ink else int(v[p])
            if payload in (bg, ink):
                continue
            arrows.append(dict(g, host=host, payload=payload, implicit=bool(v[p] == ink)))
    used = set()
    for ar in arrows:
        _require(not used & ar['write'], "overlapping arrow interpretations")
        used.update(ar['write'])
    _require(ink_cells == (used & ink_cells), "unexplained arrow ink")
    _require(arrows, "no decodable arrows")
    clean = v.copy()
    for ar in arrows:
        for p in ar['write']:
            clean[p] = ar['host']
    _require(not np.any((clean == ink) & ~reserved), "ink survived cleaning")
    ids, colors, cells = _components(clean, reserved)
    incoming = [[] for _ in cells]
    for ar in arrows:
        source, target = int(ids[ar['center']]), int(ids[ar['target_cell']])
        _require(source >= 0 and target >= 0 and source != target, "noncrossing arrow")
        _require(all(ids[p] == source for p in ar['write'] | frozenset(ar['flanks'])),
                 "arrow crosses its host boundary")
        _require(colors[source] == ar['host'], "incorrect host restoration")
        ar.update(source=source, target=target)
        incoming[target].append(ar['payload'])
    deleted = {i for i, messages in enumerate(incoming) if query <= set(messages)}
    result = clean.copy()
    for i in deleted:
        result[ids == i] = bg
    return _view(result, corner).copy(), dict(
        corner=corner, background=bg, ink=ink, query=query, clean=clean,
        ids=ids, colors=colors, cells=cells, incoming=incoming, arrows=arrows,
        deleted=deleted, reserved=reserved)


def _analyze(input_grid):
    a = _grid(input_grid)
    answers = []
    for corner in range(4):
        try:
            answers.append(_decode_view(a, corner))
        except AssertionError:
            pass
    _require(len(answers) == 1, "input does not have a unique complete interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collect all original-input messages, then erase exactly the quorums."""
    return _analyze(input_grid)[0]


def _connected(mask):
    points = np.argwhere(mask)
    if not len(points):
        return False
    start = tuple(map(int, points[0]))
    seen, queue = {start}, [start]
    for p in queue:
        for d in DIRS:
            n = _move(p, d)
            if _inside(n, mask.shape) and mask[n] and n not in seen:
                seen.add(n)
                queue.append(n)
    return len(seen) == len(points)


def _partition(rng, shape, count, rough, bands=False):
    h, w = shape
    ids = np.zeros(shape, dtype=np.int16)
    ids[:5, :5] = -1
    if bands:
        widths = np.full(count, 4, dtype=int)
        for _ in range(w - 4*count):
            widths[int(rng.integers(count))] += 1
        offset = 0
        for i, width in enumerate(widths):
            ids[:, offset:offset+width] = i
            offset += width
        ids[:5, :5] = -1
        return ids
    rr, cc = np.indices(shape)
    for new in range(1, count):
        sizes = np.array([np.count_nonzero(ids == k) for k in range(new)], dtype=float)
        found = False
        for _ in range(90):
            old = int(rng.choice(new, p=sizes/sizes.sum()))
            mask = ids == old
            rows, cols = np.nonzero(mask)
            axis = int(rng.integers(2))
            lo, hi = (int(rows.min()), int(rows.max())+1) if axis == 0 else (int(cols.min()), int(cols.max())+1)
            if hi-lo < 8:
                continue
            cut = int(rng.integers(lo+4, hi-3))
            coordinate, along = (rr, cc) if axis == 0 else (cc, rr)
            path = np.full(w if axis == 0 else h, cut)
            if rng.random() < rough:
                pivot = int(rng.integers(2, len(path)-2))
                path[pivot:] = int(np.clip(cut+int(rng.choice((-2, -1, 1, 2))), lo+3, hi-3))
            half = coordinate < path[along]
            first, second = mask & half, mask & ~half
            if (min(int(first.sum()), int(second.sum())) < 30
                    or not _connected(first) or not _connected(second)):
                continue
            ids[first] = new
            found = True
            break
        if not found:
            return None
    return ids


def _candidates(ids):
    result = []
    for p in np.ndindex(ids.shape):
        if ids[p] < 0:
            continue
        for d in DIRS:
            g = _geometry(p, d, ids.shape)
            if g is None:
                continue
            source, target = int(ids[p]), int(ids[g['target_cell']])
            if target < 0 or target == source:
                continue
            if all(ids[x] == source for x in g['write'] | frozenset(g['flanks'])):
                result.append(dict(g, source=source, target=target))
    return result


def _legal(g, blocked, context, writes):
    return not (g['write'] & (blocked | context) or g['context'] & writes)


def _take(g, blocked, context, writes):
    writes.update(g['write'])
    context.update(g['context'])
    blocked.update(g['write'])
    blocked.update(_move(p, d) for p in g['write'] for d in DIRS)


def _plan(rng, candidates, count, lists, package, q):
    """Fit message lists to randomly chosen recipients, with one original-fill send."""
    by_target = [[g for g in candidates if g['target'] == i] for i in range(count)]
    capacity_order = sorted(range(count), key=lambda i: len(by_target[i]), reverse=True)
    eligible = [i for i in capacity_order if len(by_target[i]) >= 3*min(map(len, lists))]
    if len(eligible) < len(lists):
        return None
    for _ in range(24):
        targets = [int(x) for x in rng.choice(eligible, len(lists), replace=False)]
        blocked, context, writes, chosen = set(), set(), set(), []
        pending = [(t, list(messages)) for t, messages in zip(targets, lists)]
        forced = {}
        # First teaching package: surviving hard negative sends its original q0
        # to the positive. Second: a deleted positive still sends q0 to another.
        if package in ('identity', 'simultaneous'):
            src = targets[1] if package == 'identity' else targets[0]
            dst = targets[0] if package == 'identity' else targets[1]
            options = [g for g in by_target[dst] if g['source'] == src]
            if not options:
                continue
            g = dict(options[int(rng.integers(len(options)))], payload=q[0])
            chosen.append(g)
            _take(g, blocked, context, writes)
            pending[targets.index(dst)][1].remove(q[0])
            forced[src] = q[0]
            if package == 'simultaneous':
                forced[targets[2]] = q[2]
        pending = [(t, color) for t, messages in pending for color in messages]
        # A random order avoids privileging a direction, region or payload.
        rng.shuffle(pending)
        success = True
        for position, (target, payload) in enumerate(pending):
            options = [g for g in by_target[target] if _legal(g, blocked, context, writes)]
            if not options:
                success = False
                break
            # Prefer candidates that leave the most still-needed recipients
            # usable. The randomized shortlist keeps geometry from freezing.
            rng.shuffle(options)
            best, best_score = None, -1
            future = Counter(t for t, _ in pending[position+1:])
            for g in options[:12]:
                b, c, w = blocked.copy(), context.copy(), writes.copy()
                _take(g, b, c, w)
                score = sum(min(need*8, sum(_legal(x, b, c, w) for x in by_target[t]))
                            for t, need in future.items())
                if score > best_score:
                    best, best_score = g, score
            chosen.append(dict(best, payload=int(payload)))
            _take(best, blocked, context, writes)
        if success:
            return chosen, forced, targets
    return None


def _fills(rng, ids, count, pool, forced, banned=None, repeat_status=None):
    neighbors = [set() for _ in range(count)]
    for a, b in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        mask = (a >= 0) & (b >= 0) & (a != b)
        for x, y in zip(a[mask], b[mask]):
            neighbors[int(x)].add(int(y))
            neighbors[int(y)].add(int(x))
    colors = [-1]*count
    banned = {} if banned is None else banned
    for i, c in forced.items():
        colors[i] = int(c)
    if repeat_status is not None:
        # Teach component identity with equal original fills but different
        # membership outcomes. This is a construction witness, not a rule.
        pairs = [(i, j) for i in range(count) for j in range(i)
                 if (i in repeat_status) != (j in repeat_status) and j not in neighbors[i]]
        rng.shuffle(pairs)
        for i, j in pairs:
            required = {colors[k] for k in (i, j) if colors[k] >= 0}
            if len(required) > 1:
                continue
            options = [c for c in pool if (not required or c in required)
                       and c not in banned.get(i, set()) | banned.get(j, set())
                       and c not in {colors[k] for k in neighbors[i] | neighbors[j]}]
            if options:
                colors[i] = colors[j] = int(rng.choice(options))
                break
        else:
            return None
    for i in range(count):
        if colors[i] >= 0 and (colors[i] in banned.get(i, set())
                              or any(colors[n] == colors[i] for n in neighbors[i])):
            return None
    # Sometimes a query identity exists solely as an explicit payload. The
    # available fill palette is otherwise independent of the recipient rule.
    available = list(pool)
    if rng.random() < .35:
        removable = [c for c in pool if c not in forced.values()]
        if removable:
            available.remove(int(rng.choice(removable)))
    for i in rng.permutation(count):
        i = int(i)
        if colors[i] >= 0:
            continue
        options = [c for c in available if c not in {colors[j] for j in neighbors[i]}
                   and c not in banned.get(i, set())]
        if not options:
            return None
        colors[i] = int(rng.choice(options))
    return colors


def _message_lists(rng, query, extras, package):
    a, b, c = query
    x, y = extras[:2]
    if package == 'identity':
        return [[a, b, c], [b, c, x], [a, c, a, x]]
    if package == 'simultaneous':
        return [[a, b, c, x, a], [a, b, c], [a, b, x, y]]
    number = int(rng.integers(2, 5))
    messages = []
    for i in range(number):
        positive = bool(rng.random() < .5)
        if positive:
            labels = list(query)
            if rng.random() < .6:
                labels.append(int(rng.choice(extras)))
        else:
            labels = list(rng.permutation(query)[:int(rng.integers(3))])
            labels += list(rng.permutation(extras)[:int(rng.integers(3))])
        if not labels:
            labels = [int(rng.choice(extras))]
        limit = 7 if package == 'test' and i == 0 else 5
        size = int(rng.integers(max(len(labels), 6) if package == 'test' and i == 0 else len(labels), limit+1))
        messages.append(labels + [int(rng.choice(labels)) for _ in range(size-len(labels))])
    return messages


def _make_pair(rng, base_palette, vary_palette, corner, package, scale, teaching_query=None):
    for attempt in range(85):
        if vary_palette:
            if teaching_query is None:
                palette = list(map(int, rng.permutation(10)))
            else:
                other = list(map(int, rng.permutation([c for c in range(10) if c not in teaching_query])))
                palette = other[:2] + list(teaching_query) + other[2:]
        else:
            palette = base_palette
        bg, ink = palette[:2]
        pool = palette[2:]
        query = list(teaching_query) if teaching_query is not None else list(map(int, rng.choice(pool, 3, replace=False)))
        extras = [c for c in pool if c not in query]
        rng.shuffle(extras)
        lists = _message_lists(rng, query, extras, package)
        if attempt >= 65:
            count = 5
            shape = (int(rng.integers(26, 31)), int(rng.integers(25, 31)))
            ids = _partition(rng, shape, count, 0, bands=True)
        else:
            count = int(rng.integers(5, 9 if package == 'test' else 8))
            lo, hi = (25, 31) if package == 'test' and scale else (21, 31)
            shape = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
            rough = float(rng.choice((0., .35, .8)))
            ids = _partition(rng, shape, count, rough)
        if ids is None:
            continue
        candidates = _candidates(ids)
        plan = _plan(rng, candidates, count, lists, package, query)
        if plan is None:
            continue
        arrows, forced, targets = plan
        incoming = [set() for _ in range(count)]
        for ar in arrows:
            incoming[ar['target']].add(ar['payload'])
        deleted = {i for i, ms in enumerate(incoming) if set(query) <= ms}
        banned = {}
        if package == 'identity':
            # The exact positive needs b from a literal junction: assuming
            # every arrow carries its host fill must fail on this example.
            for ar in arrows:
                if ar['target'] == targets[0]:
                    banned[ar['source']] = {query[1]}
        colors = _fills(rng, ids, count, pool, forced, banned,
                        deleted if package == 'identity' else None)
        if colors is None:
            continue
        clean = np.full(shape, bg, dtype=np.int8)
        for i, color in enumerate(colors):
            clean[ids == i] = color
        inp = clean.copy()
        for ar in arrows:
            for p in ar['write']:
                inp[p] = ink
            if ar['payload'] != colors[ar['source']]:
                inp[ar['center']] = ar['payload']
        inp[3, :4] = ink
        inp[:4, 3] = ink
        for flat, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
            inp[int(flat)//3, int(flat)%3] = color
        out = clean.copy()
        out[:4, :4] = inp[:4, :4]
        for i in deleted:
            out[ids == i] = bg
        inp, out = _view(inp, corner).copy(), _view(out, corner).copy()
        try:
            predicted, info = _analyze(inp)
        except AssertionError:
            continue
        if not np.array_equal(predicted, out) or len(info['arrows']) != len(arrows):
            continue
        return dict(input=inp, output=out)
    raise AssertionError('bounded scene construction exhausted')


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be an integer in 1..4, below num_train')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    teaching_query = palette[2:5]
    vary_palette = bool(rng.integers(2))
    corners = list(map(int, rng.choice(4, 2, replace=False))) if rng.random() < .65 else [int(rng.integers(4))]
    scale = bool(rng.integers(2))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, total in (('train', num_train), ('test', num_test)):
        for i in range(total):
            package = ('identity' if i == 0 else 'simultaneous' if i == 1 else 'free') if split == 'train' else 'test'
            corner = corners[i % len(corners)] if split == 'train' else int(rng.choice(corners))
            for _ in range(8):
                pair = _make_pair(rng, palette, vary_palette, corner, package, scale,
                                  teaching_query if package in ('identity', 'simultaneous') else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('duplicate construction exhausted')
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_coverage(infos):
    records = [(info, i, ms) for info in infos for i, ms in enumerate(info['incoming'])]
    positives = [(a, i, m) for a, i, m in records if i in a['deleted']]
    negatives = [(a, i, m) for a, i, m in records if i not in a['deleted']]
    _require(positives and negatives, 'need deletion and preservation demonstrations')
    _require(any(len(m) == 3 for _, _, m in positives), 'missing exact-query positive')
    _require(any(len(m) == 3 and len(set(m)) == 3 for _, _, m in negatives), 'missing matched-count negative')
    _require(any(len(set(m)) >= 4 for _, _, m in negatives), 'missing more-colorful negative')
    for group in (positives, negatives):
        _require(any(len(set(m)) < len(m) for _, _, m in group), 'missing duplicate-message contrast')
    _require(any(set(m)-a['query'] for a, _, m in positives), 'missing positive with extras')
    _require(any(not m for _, _, m in negatives), 'missing zero-incoming territory')
    # The same query is used in the two teaching packages, so every literal
    # identity is independently shown to matter even with only two examples.
    _require(any(all(any(b['query'] == a['query'] and set(n) & b['query'] == b['query']-{q}
                        for b, _, n in negatives) for q in a['query'])
                 for a in infos), 'missing an identity-specific hard negative')
    _require(any(a['colors'][i] in a['query']-set(m) and len(set(m) & a['query']) == 2
                 for a, i, m in negatives), 'native fill must not complete the quorum')
    _require(any(a['colors'][i] == a['colors'][j] and ((i in a['deleted']) != (j in a['deleted']))
                 for a in infos for i in range(len(a['cells'])) for j in range(i)),
             'missing equal-fill component-locality demonstration')
    _require(any(not a['query'] <= {ar['host'] for ar in a['arrows'] if ar['target'] == i}
                 for a, i, _ in positives), 'literal payloads are not necessary in training')
    _require(any(ar['implicit'] and ar['source'] in a['deleted'] and ar['target'] in a['deleted']
                 and ar['payload'] in a['query']
                 and not a['query'] <= {b['payload'] for b in a['arrows']
                                      if b['target'] == ar['target'] and b['source'] not in a['deleted']}
                 for a in infos for ar in a['arrows']), 'missing simultaneous original-fill witness')


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, infos = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas dimensions changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            _require(np.array_equal(expected, out), 'incorrect quorum output or annotation cleanup')
            ov = _view(out, info['corner'])
            _require(np.array_equal(_view(inp, info['corner'])[:5, :5], ov[:5, :5]), 'selector changed')
            for i, cells in enumerate(info['cells']):
                expected_color = info['background'] if i in info['deleted'] else info['colors'][i]
                _require(all(ov[p] == expected_color for p in cells), 'partial territory update')
            _require(not np.any((ov == info['ink']) & ~info['reserved']), 'arrow ink retained')
            infos[split].append(info)
    _training_coverage(infos['train'])
    taught_corners = {x['corner'] for x in infos['train']}
    _require(all(x['corner'] in taught_corners for x in infos['test']), 'unshown selector corner')
    _require(all(max(map(len, x['incoming'])) >= 6 for x in infos['test']), 'missing multiplicity shift')
    _require(all(max(map(len, x['incoming'])) <= 5 for x in infos['train']), 'training multiplicity exceeds curriculum')


def _normalized(a):
    names = {}
    return a.shape, tuple(names.setdefault(int(x), len(names)) for x in a.ravel())


def self_test() -> dict:
    """Bounded 42-seed coverage, semantic counterfactuals and corruption checks."""
    structures, shapes, roles, directions, corners = set(), set(), set(), set(), set()
    territories, arrows, quorums, repeated, irregular, singleton = set(), set(), set(), 0, 0, 0
    examples = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 2:
            again = generate_puzzle(seed)
            _require(all(_key(p[k]) == _key(q[k]) for s in puzzle
                         for p, q in zip(puzzle[s], again[s]) for k in p), 'nondeterministic puzzle')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _analyze(pair['input'].copy())
                _require(np.array_equal(predicted, pair['output']), 'input-only solve disagreement')
                structures.add(_normalized(pair['input']))
                shapes.add(pair['input'].shape)
                roles.add((info['background'], info['ink']))
                corners.add(info['corner'])
                directions.update(ar['direction'] for ar in info['arrows'])
                territories.add(len(info['cells']))
                arrows.add(len(info['arrows']))
                quorums.add(len(info['deleted']))
                repeated += len(info['colors']) > len(set(info['colors']))
                singleton += any(ar['payload'] not in info['colors'] for ar in info['arrows'])
                for cells in info['cells']:
                    rs, cs = zip(*cells)
                    irregular += len(cells) != (max(rs)-min(rs)+1)*(max(cs)-min(cs)+1)
                examples += 1
        # Omitting A: no message collection, hence no territory can qualify.
        # Omitting B: erasing every receiver incorrectly deletes hard negatives.
        infos = [_analyze(p['input'])[1] for p in puzzle['train']]
        _require(any(i['deleted'] for i in infos), 'A ablation did not change an answer')
        _require(any(m and j not in i['deleted'] for i in infos for j, m in enumerate(i['incoming'])),
                 'B ablation did not change an answer')
        # Change just the selector query to the equally colorful hard negative.
        for pair, info in zip(puzzle['train'], infos):
            choices = [j for j, m in enumerate(info['incoming'])
                       if j not in info['deleted'] and len(m) == 3 and len(set(m)) == 3]
            if choices:
                changed = pair['input'].copy()
                v = _view(changed, info['corner'])
                points = np.argwhere(v[:3, :3] != info['background'])
                for p, c in zip(points, sorted(set(info['incoming'][choices[0]]))):
                    v[tuple(p)] = c
                _, new = _analyze(changed)
                _require(choices[0] in new['deleted'], 'selector counterfactual had no effect')
                # Query positions are irrelevant; moving them preserves decisions.
                moved = pair['input'].copy()
                mv = _view(moved, info['corner'])
                mv[:3, :3] = np.roll(mv[:3, :3].copy(), 1, axis=0)
                _require(_analyze(moved)[1]['deleted'] == info['deleted'], 'query position affected decisions')
                break
        else:
            raise AssertionError('missing selector counterfactual')
        # Alter one routed payload while leaving geometry and query intact.
        checked_payload = False
        for pair, info in zip(puzzle['train'], infos):
            exact = [j for j in info['deleted'] if len(info['incoming'][j]) == 3]
            for ar in info['arrows']:
                if ar['target'] not in exact:
                    continue
                for replacement in range(10):
                    if replacement in info['query'] | {info['background'], info['ink']}:
                        continue
                    changed = pair['input'].copy()
                    _view(changed, info['corner'])[ar['center']] = replacement
                    try:
                        _, altered = _analyze(changed)
                    except AssertionError:
                        # A token change can make an accidental extra T in
                        # nearby ink. Such an ambiguous scene is outside the
                        # family; use a counterfactual retaining legibility.
                        continue
                    _require(ar['target'] not in altered['deleted'],
                             'changing a payload did not break the exact quorum')
                    checked_payload = True
                    break
                if checked_payload:
                    break
            if checked_payload:
                break
        _require(checked_payload, 'no legible payload counterfactual')
        # Literal color identities carry no hidden numeric meaning.
        if seed < 2:
            pair = puzzle['test'][0]
            renamed = 9-pair['input']
            _require(np.array_equal(solve(renamed), 9-pair['output']), 'palette equivariance failed')
    _require(len(structures) == examples and len(shapes) >= 35 and len(roles) >= 30, 'collapsed variation')
    _require(directions == set(DIRS) and corners == set(range(4)), 'missing orientations')
    _require(len(territories) >= 3 and len(arrows) >= 10 and len(quorums) >= 3, 'collapsed counts')
    _require(repeated >= 30 and irregular >= 30 and singleton >= 30, 'missing structural/payload variation')
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(1234, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Validate labels as well as schema: an incomplete erasure must fail.
    broken = generate_puzzle(0)
    pair = next(p for p in broken['train'] if _analyze(p['input'])[1]['deleted'])
    _, info = _analyze(pair['input'])
    index = min(info['deleted'])
    _view(pair['output'], info['corner'])[info['cells'][index][0]] = info['colors'][index]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial erasure accepted')
    return dict(seeds=42, examples=examples, distinct_structures=len(structures), shapes=len(shapes),
                role_assignments=len(roles), territory_counts=sorted(territories), arrow_counts=sorted(arrows),
                deletion_counts=sorted(quorums), repeated_fill_scenes=repeated,
                nonrectangular_territories=irregular, novel_payload_scenes=singleton)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
