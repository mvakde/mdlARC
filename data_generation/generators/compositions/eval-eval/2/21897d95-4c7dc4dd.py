"""Territory echoes: infer a graph-distance shell from a visible worked map.

Only standard-library modules and NumPy are used. No file reads, caches, or
generation state are involved in solve(). Saved diagnostics are opt-in only.
"""
from collections import deque
import hashlib

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _step(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(a, binary=False):
    """Four-neighbor components; binary=True treats all nonzero cells alike."""
    ids = np.full(a.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(a.shape):
        if ids[start] >= 0 or (binary and not a[start]):
            continue
        label, color = len(cells), int(a[start])
        stack, points = [start], []
        ids[start] = label
        while stack:
            p = stack.pop()
            points.append(p)
            for d in DIRS:
                q = _step(p, d)
                if (_inside(q, a.shape) and ids[q] < 0
                        and (bool(a[q]) if binary else a[q] == color)):
                    ids[q] = label
                    stack.append(q)
        colors.append(color)
        cells.append(set(points))
    return ids, colors, cells


def _adjacency(ids, count):
    graph = [set() for _ in range(count)]
    for a, b in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        different = a != b
        for x, y in zip(a[different], b[different]):
            graph[int(x)].add(int(y))
            graph[int(y)].add(int(x))
    return graph


def _geometry(p, d, shape):
    side = (-d[1], d[0])
    arms = (_step(p, side), _step(p, side, -1))
    tip, target = _step(p, d), _step(p, d, 2)
    flanks = (_step(tip, side), _step(tip, side, -1))
    write = {p, tip, *arms}
    support = {target, *flanks}
    if not all(_inside(q, shape) for q in write | support):
        return None
    return dict(center=p, direction=d, arms=arms, tip=tip, target_cell=target,
                flanks=flanks, write=write, support=support)


def _read_map(a, ink):
    """Restore T annotations, then recover original connected territories."""
    raw_ids, _, raw_cells = _components(a)
    ink_points = set(map(tuple, np.argwhere(a == ink)))
    _need(bool(ink_points), "map has no arrows")
    _need(all(len(raw_cells[int(raw_ids[p])]) <= 4 for p in ink_points), "ink is a territory")
    centers = {_step(p, d) for p in ink_points for d in DIRS}
    arrows = []
    for p in sorted(centers):
        for d in DIRS:
            g = _geometry(p, d, a.shape)
            if g is None or not all(a[q] == ink for q in (*g['arms'], g['tip'])):
                continue
            if any(not raw_cells[int(raw_ids[q])] <= g['write'] for q in (*g['arms'], g['tip'])):
                continue
            x, y = g['flanks']
            if a[x] != a[y] or a[x] == ink or a[g['target_cell']] in (ink, a[x]):
                continue
            arrows.append(dict(g, host=int(a[x]), explicit=bool(a[p] != ink),
                               payload=int(a[p]) if a[p] != ink else int(a[x])))
    _need(bool(arrows), "no recognizable T")
    used = set()
    clean = a.copy()
    for arrow in arrows:
        _need(not used & arrow['write'], "ambiguous or overlapping arrows")
        used.update(arrow['write'])
        for p in arrow['write']:
            clean[p] = arrow['host']
    _need(used & ink_points == ink_points, "unexplained ink")
    _need(ink not in clean, "ink survived restoration")
    ids, colors, cells = _components(clean)
    for arrow in arrows:
        source = int(ids[arrow['center']])
        target = int(ids[arrow['target_cell']])
        _need(source != target and all(ids[p] == source for p in arrow['write']), "T crosses its host")
        arrow.update(source=source, target=target)
    return dict(clean=clean, ids=ids, colors=colors, cells=cells,
                graph=_adjacency(ids, len(colors)), arrows=arrows, ink=ink)


def _shell(graph, source, target, reach):
    """The first crossing is directed; later crossings spread without backwash."""
    distances = {target: 0}
    todo = deque([target])
    while todo:
        v = todo.popleft()
        if distances[v] == reach-1:
            continue
        for w in sorted(graph[v]):
            if w != source and w not in distances:
                distances[w] = distances[v]+1
                todo.append(w)
    return {v for v, distance in distances.items() if distance == reach-1}


def _paint(info, reach):
    assignments, shells = {}, []
    for arrow in info['arrows']:
        shell = _shell(info['graph'], arrow['source'], arrow['target'], reach)
        _need(bool(shell), "echo has no final shell")
        shells.append(shell)
        for v in shell:
            _need(v not in assignments or assignments[v] == arrow['payload'], "conflicting echoes")
            assignments[v] = arrow['payload']
    out = info['clean'].copy()
    for v, color in assignments.items():
        out[info['ids'] == v] = color
    return out, assignments, shells


def _panels(a):
    """Zero gutters separate three solid framed maps; no positional semantics."""
    _, _, groups = _components(a, binary=True)
    _need(len(groups) == 3, "expected three separate framed maps")
    result, frames = [], set()
    for cells in groups:
        r0, c0 = np.min(list(cells), axis=0)
        r1, c1 = np.max(list(cells), axis=0)+1
        box = (int(r0), int(c0), int(r1), int(c1))
        block = a[r0:r1, c0:c1]
        _need(len(cells) == block.size and min(block.shape) >= 5, "map is not a filled rectangle")
        frame = int(block[0, 0])
        _need(all(np.all(edge == frame) for edge in (block[0], block[-1], block[:, 0], block[:, -1])), "broken frame")
        interior = block[1:-1, 1:-1].copy()
        _need(frame not in interior, "frame color used as territory")
        frames.add(frame)
        result.append(dict(box=box, grid=interior))
    _need(len(frames) == 1, "inconsistent frame convention")
    result.sort(key=lambda p: p['grid'].size)
    _need(result[0]['grid'].shape == result[1]['grid'].shape
          and result[1]['grid'].size < result[2]['grid'].size, "reference/query sizes ambiguous")
    return result[:2], result[2], frames.pop()


def _interpret(input_grid):
    a = _grid(input_grid)
    refs, query, frame = _panels(a)
    interpretations = []
    for before_index in (0, 1):
        before, after = refs[before_index]['grid'], refs[1-before_index]['grid']
        for ink0 in sorted(set(map(int, np.unique(before))) & set(map(int, np.unique(query['grid'])))):
            if ink0 in after:
                continue
            try:
                reference = _read_map(before, ink0)
                _need(len(reference['arrows']) == 1, "reference must have one launch")
                info = _read_map(query['grid'], ink0)
                for reach in (1, 2, 3):
                    worked, _, _ = _paint(reference, reach)
                    if not np.array_equal(worked, after):
                        continue
                    output, assignments, shells = _paint(info, reach)
                    interpretations.append(dict(output=output, reach=reach, info=info,
                                                assignments=assignments, shells=shells,
                                                refs=refs, before_index=before_index,
                                                reference=reference, query=query, frame=frame, ink=ink0))
            except AssertionError:
                continue
    _need(len(interpretations) == 1, "worked example does not determine one interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer reach from the two small maps, then echo on the large map."""
    return _interpret(input_grid)['output'].copy()


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        for d in DIRS:
            q = _step(stack[-1], d)
            if q in points:
                points.remove(q)
                stack.append(q)
                break
        else:
            stack.pop()
    return not points


def _partition(rng, shape, count):
    labels = np.zeros(shape, dtype=np.int16)
    rr, cc = np.indices(shape)
    rough = bool(rng.integers(3))
    for new in range(1, count):
        for _ in range(80):
            sizes = np.bincount(labels.ravel(), minlength=new)
            old = int(rng.choice(new, p=sizes/sizes.sum()))
            mask = labels == old
            rows, cols = np.nonzero(mask)
            lo = (int(rows.min()), int(cols.min()))
            hi = (int(rows.max())+1, int(cols.max())+1)
            axis = int(rng.integers(2))
            if hi[axis]-lo[axis] < 5:
                continue
            cut = int(rng.integers(lo[axis]+2, hi[axis]-1))
            coordinate, along = (rr, cc) if axis == 0 else (cc, rr)
            path = np.full(shape[1-axis], cut)
            if rough and rng.random() < .7:
                switch = int(rng.integers(1, len(path)))
                path[switch:] += int(rng.choice([-1, 1]))
            section = coordinate < path[along]
            # Sometimes carve an island: its enclosing room adds graph loops.
            if min(hi[0]-lo[0], hi[1]-lo[1]) >= 7 and rng.random() < .18:
                top = int(rng.integers(lo[0]+1, hi[0]-4))
                left = int(rng.integers(lo[1]+1, hi[1]-4))
                bottom = int(rng.integers(top+3, hi[0]))
                right = int(rng.integers(left+3, hi[1]))
                section = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
            part, rest = mask & section, mask & ~section
            if (min(int(part.sum()), int(rest.sum())) < 10
                    or not _connected(part) or not _connected(rest)):
                continue
            labels[part] = new
            break
        else:
            return None
    return labels


def _fills(rng, labels, palette):
    count = int(labels.max())+1
    graph = _adjacency(labels, count)
    colors = [-1]*count
    available = list(map(int, rng.permutation(palette)))
    nonneighbors = [(x, y) for x in range(count) for y in range(x) if y not in graph[x]]
    if nonneighbors and rng.random() < .8:
        x, y = nonneighbors[int(rng.integers(len(nonneighbors)))]
        colors[x] = colors[y] = available[0]
    for v0 in rng.permutation(count):
        v = int(v0)
        if colors[v] >= 0:
            continue
        options = [c for c in available if c not in {colors[n] for n in graph[v]}]
        if not options:
            return None
        unused = [c for c in options if c not in colors]
        pool = unused if unused else options
        colors[v] = int(rng.choice(pool))
    return np.array(colors, dtype=np.int8)


def _candidates(labels):
    result = []
    for p in np.ndindex(labels.shape):
        for d in DIRS:
            g = _geometry(p, d, labels.shape)
            if g is None:
                continue
            source, target = int(labels[p]), int(labels[g['target_cell']])
            if source != target and all(labels[q] == source for q in g['write'] | set(g['flanks'])):
                result.append(dict(g, source=source, target=target))
    return result


def _plan(rng, labels, colors, palette, reach, desired, chain, explicit, branch):
    candidates = _candidates(labels)
    rng.shuffle(candidates)
    graph = _adjacency(labels, len(colors))
    for a in candidates:
        a['shell'] = _shell(graph, a['source'], a['target'], reach)
    chosen, blocked, support, used, assignments = [], set(), set(), set(), {}

    def legal(a, payload):
        return (bool(a['shell']) and not a['write'] & (blocked | support)
                and not a['support'] & used
                and any(colors[v] != payload for v in a['shell'])
                and all(v not in assignments or assignments[v] == payload for v in a['shell']))

    def take(a, payload, is_explicit):
        chosen.append(dict(a, payload=int(payload), explicit=is_explicit))
        used.update(a['write'])
        support.update(a['support'])
        blocked.update(a['write'])
        blocked.update(_step(p, d) for p in a['write'] for d in DIRS)
        for v in a['shell']:
            assignments[v] = int(payload)

    if chain:
        # An open two-edge chain is a concrete original-color witness.
        for a in candidates:
            for b in candidates:
                if (b['source'] != a['target'] or b['target'] == a['source']
                        or colors[a['source']] == colors[b['source']]):
                    continue
                halo = a['write'] | {_step(p, d) for p in a['write'] for d in DIRS}
                if b['write'] & (halo | a['support']) or b['support'] & a['write']:
                    continue
                if legal(a, int(colors[a['source']])):
                    take(a, int(colors[a['source']]), False)
                    if legal(b, int(colors[b['source']])):
                        take(b, int(colors[b['source']]), False)
                        break
                    return None
            if chosen:
                break
        if len(chosen) != 2:
            return None
    for a in candidates:
        if len(chosen) == desired:
            break
        if branch and not chosen and len(a['shell']) < 2:
            continue
        is_explicit = explicit and (not chosen or rng.random() < .45)
        if is_explicit:
            options = [int(c) for c in palette if c != colors[a['source']] and legal(a, int(c))]
            if not options:
                continue
            payload = int(rng.choice(options))
        else:
            payload = int(colors[a['source']])
            if not legal(a, payload):
                continue
        take(a, payload, is_explicit)
    if len(chosen) != desired:
        return None
    return chosen, assignments


def _draw_arrows(clean, arrows, ink):
    a = clean.copy()
    for arrow in arrows:
        for p in arrow['write']:
            a[p] = ink
        if arrow['explicit']:
            a[arrow['center']] = arrow['payload']
    return a


def _reference(rng, reach, palette, ink):
    count = int(rng.integers(max(4, reach+2), 7))
    widths = [2]+[1]*(count-1)
    total = int(rng.integers(max(8, sum(widths)), 12))
    for _ in range(total-sum(widths)):
        widths[int(rng.integers(count))] += 1
    h = int(rng.integers(3, 6))
    colors = np.array(rng.choice(palette, count, replace=False), dtype=np.int8)
    labels = np.repeat(np.arange(count)[None, :], widths, axis=1)
    labels = np.repeat(labels, h, axis=0)
    clean = colors[labels]
    g = _geometry((int(rng.integers(1, h-1)), widths[0]-2), (0, 1), clean.shape)
    arrow = dict(g, source=0, target=1, payload=int(colors[0]), explicit=False)
    before = _draw_arrows(clean, [arrow], ink)
    after = clean.copy()
    after[labels == reach] = colors[0]
    if rng.integers(2):
        before, after = before[:, ::-1].copy(), after[:, ::-1].copy()
    return before, after


def _frame(a, color):
    out = np.full((a.shape[0]+2, a.shape[1]+2), color, dtype=np.int8)
    out[1:-1, 1:-1] = a
    return out


def _layout(rng, before, after, query, frame):
    refs = [_frame(before, frame), _frame(after, frame)]
    if rng.integers(2):
        refs.reverse()
    q = _frame(query, frame)
    gap = int(rng.integers(1, 3))
    # References are at most 13 columns each: maximum width is 30.
    width = max(sum(x.shape[1] for x in refs)+gap+2, q.shape[1]+2)
    height = refs[0].shape[0]+q.shape[0]+3
    _need(height <= 30 and width <= 30, "layout exceeds ARC bounds")
    a = np.zeros((height, width), dtype=np.int8)
    top_query = bool(rng.integers(2))
    qr = 1 if top_query else refs[0].shape[0]+2
    rr = q.shape[0]+2 if top_query else 1
    qc = int(rng.integers(1, width-q.shape[1]))
    rc = int(rng.integers(1, width-sum(x.shape[1] for x in refs)-gap))
    a[qr:qr+q.shape[0], qc:qc+q.shape[1]] = q
    for ref in refs:
        a[rr:rr+ref.shape[0], rc:rc+ref.shape[1]] = ref
        rc += ref.shape[1]+gap
    return a


def _make_pair(rng, test, reach, chain=False, explicit=False, branch=False):
    for attempt in range(100):
        frame, ink, *palette = map(int, rng.permutation(np.arange(1, 10)))
        h = int(rng.integers(14, 18) if test else rng.integers(9, 13))
        w = int(rng.integers(22, 27) if test else rng.integers(16, 25))
        count = int(rng.integers(9, 12) if test else rng.integers(5, 8))
        labels = _partition(rng, (h, w), count)
        if labels is None:
            continue
        colors = _fills(rng, labels, palette)
        if colors is None:
            continue
        desired = 2 if chain else int(rng.integers(1, 5 if test else 4))
        plan = _plan(rng, labels, colors, palette, reach, desired, chain, explicit, branch)
        if plan is None:
            continue
        arrows, assignments = plan
        clean, output = colors[labels], colors[labels].copy()
        for v, payload in assignments.items():
            output[labels == v] = payload
        query = _draw_arrows(clean, arrows, ink)
        before, after = _reference(rng, reach, palette, ink)
        a = _layout(rng, before, after, query, frame)
        turn = int(rng.integers(4))
        a, output = np.rot90(a, turn).copy(), np.rot90(output, turn).copy()
        try:
            read = _interpret(a)
            if not np.array_equal(read['output'], output):
                continue
            features = _features(read)
            if chain and not features['chain']:
                continue
            if explicit and not features['explicit']:
                continue
            if branch and not (features['branch'] and features['locality']):
                continue
        except AssertionError:
            continue
        return dict(input=a, output=output)
    raise RuntimeError("bounded map construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for n, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(n) or not lo <= n <= hi:
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            chain = split == 'train' and index == 0
            lesson = split == 'train' and index == 1
            explicit = lesson or (not chain and rng.random() < .55)
            branch = lesson
            reach = 1 if chain else int(rng.integers(2 if lesson or split == 'test' else 1, 4))
            for _ in range(8):
                pair = _make_pair(rng, split == 'test', reach, chain, explicit, branch)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    return puzzle


def _features(read):
    info, assignments = read['info'], read['assignments']
    colors, arrows = info['colors'], info['arrows']
    final = [assignments.get(v, c) for v, c in enumerate(colors)]
    chain = any(not a['explicit'] and not b['explicit'] and b['source'] in shell
                and assignments[b['source']] != colors[b['source']]
                and a['payload'] != b['payload']
                for a, shell in zip(arrows, read['shells']) for b in arrows if a is not b)
    boxes = [(max(p[0] for p in s)-min(p[0] for p in s)+1,
              max(p[1] for p in s)-min(p[1] for p in s)+1) for s in info['cells']]
    return dict(reach=read['reach'], rooms=len(colors), arrows=len(arrows),
                explicit=sum(a['explicit'] for a in arrows), chain=chain,
                branch=any(sum(colors[v] != a['payload'] for v in s) >= 2
                           for a, s in zip(arrows, read['shells'])),
                locality=any(colors[v] == colors[w] and final[v] != final[w]
                             for v in range(len(colors)) for w in range(v)),
                irregular=any(len(s) != h*w for s, (h, w) in zip(info['cells'], boxes)),
                cycle=sum(map(len, info['graph']))//2 >= len(colors),
                multiple_incoming=len(assignments) < sum(map(len, read['shells'])),
                shape=info['clean'].shape, directions={a['direction'] for a in arrows},
                changed=sum(c != f for c, f in zip(colors, final)))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal example counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            read = _interpret(inp)
            _need(np.array_equal(out, read['output']), "incorrect echo output")
            _need(not ({0, read['frame'], read['ink']} & set(map(int, np.unique(out)))), "annotation or gutter in answer")
            info = read['info']
            for v, cells in enumerate(info['cells']):
                expected = read['assignments'].get(v, info['colors'][v])
                _need(all(out[p] == expected for p in cells), "a territory was not painted uniformly")
            f = _features(read)
            _need(f['changed'] > 0, "no visible echo")
            features[split].append(f)
    train = features['train']
    _need(any(f['chain'] for f in train), "no simultaneous-transfer demonstration")
    _need(any(f['explicit'] and f['branch'] for f in train), "no explicit branching demonstration")
    _need(any(f['locality'] for f in train), "no repeated-color locality demonstration")
    _need(1 in {f['reach'] for f in train} and any(f['reach'] > 1 for f in train), "reach contrast absent")
    _need(all(f['rooms'] > max(t['rooms'] for t in train) for f in features['test']), "room-count shift absent")
    _need(all(np.prod(f['shape']) > max(np.prod(t['shape']) for t in train) for f in features['test']), "scale shift absent")


def _canonical(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _diagnostics():
    # Same query, different fully visible reference lessons -> different answer.
    rng = np.random.default_rng(991)
    labels = np.repeat(np.arange(5)[None, :], 4, axis=1).repeat(5, axis=0)
    palette, ink, frame = [1, 2, 3, 4, 5, 6, 7], 8, 9
    clean = np.array(palette[:5], dtype=np.int8)[labels]
    arrow = dict(_geometry((2, 2), (0, 1), clean.shape), payload=1, explicit=False)
    query = _draw_arrows(clean, [arrow], ink)
    answers = []
    # Keep both the query AND the reference-before pixels fixed.
    before, _ = _reference(rng, 3, palette, ink)
    ref_info = _read_map(before, ink)
    for reach in (1, 2, 3):
        after = _paint(ref_info, reach)[0]
        a = _layout(np.random.default_rng(12), before, after, query, frame)
        out = solve(a)
        answers.append(_key(out))
        expected = clean.copy()
        expected[labels == reach] = 1
        _need(np.array_equal(out, expected), "reference dependence diagnostic failed")
    _need(len(set(answers)) == 3, "reference can be bypassed")
    # Distances are shortest paths, sources are excluded, cycles do not echo twice.
    graph = [{1}, {0, 2, 3}, {1, 4}, {1, 4}, {2, 3}]
    _need(_shell(graph, 0, 1, 1) == {1}, "first crossing")
    _need(_shell(graph, 0, 1, 2) == {2, 3}, "branch frontier")
    _need(_shell(graph, 0, 1, 3) == {4}, "shortest-path frontier")
    # Malformed maps and conflicting instructions have no invented priority.
    info = dict(clean=clean, ids=labels, colors=palette[:5],
                graph=[{1}, {0, 2}, {1, 3}, {2, 4}, {3}],
                arrows=[dict(source=0, target=1, payload=1), dict(source=2, target=1, payload=3)])
    try:
        _paint(info, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting echoes accepted")


def self_test():
    """42 seeds, bounded in-memory measurements; no persisted puzzles or traces."""
    digests, shapes, partitions, outputs = set(), set(), set(), set()
    reaches, rooms, arrows, inks, directions = set(), set(), set(), set(), set()
    totals = dict(chain=0, explicit=0, branch=0, locality=0, irregular=0, cycle=0, multiple_incoming=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        if seed < 2:
            _need(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input'].copy()
                read = _interpret(inp)
                f = _features(read)
                reaches.add(f['reach'])
                rooms.add(f['rooms'])
                arrows.add(f['arrows'])
                inks.add(read['ink'])
                directions.update(f['directions'])
                shapes.add(inp.shape)
                partitions.add(_key(read['info']['ids']))
                outputs.add(_canonical(pair['output']))
                for k in totals:
                    totals[k] += bool(f[k])
                _need(np.array_equal(solve(inp), pair['output']), "input-only solve failed")
                _need(np.array_equal(inp, pair['input']), "solve mutated input")
                # Counterfactual removing A: restore annotations but send nothing.
                _need(not np.array_equal(read['info']['clean'], pair['output']), "transfer can be omitted")
                # Every orientation and every nonzero palette permutation commute.
                if seed < 3:
                    for turn in (1, 2, 3):
                        _need(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(pair['output'], turn)), "orientation bias")
                    palette = np.r_[0, np.random.default_rng(seed+100).permutation(np.arange(1, 10))]
                    _need(np.array_equal(solve(palette[inp]), palette[pair['output']]), "literal-color bias")
    _need(len(digests) == 42 and len(partitions) >= 150 and len(outputs) >= 150, "collapsed structural diversity")
    _need(len(shapes) >= 20 and reaches == {1, 2, 3} and len(rooms) >= 6, "missing scale/reach variation")
    _need(arrows >= {1, 2, 3, 4} and len(inks) >= 8 and directions == set(DIRS), "missing annotation variation")
    _need(all(v >= 3 for v in totals.values()), "missing family-specific variation")
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            puzzle = generate_puzzle(1042, train, test)
            _need(len(puzzle['train']) == train and len(puzzle['test']) == test, "ignored counts")
            validate_generated_puzzle(puzzle)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(912)
    damaged['test'][0]['output'][0, 0] = 0
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    _diagnostics()
    return dict(seeds=42, distinct_puzzles=len(digests), canvas_shapes=len(shapes),
                territory_partitions=len(partitions), normalized_outputs=len(outputs),
                reaches=sorted(reaches), room_counts=sorted(rooms), arrow_counts=sorted(arrows),
                feature_pairs=totals)
