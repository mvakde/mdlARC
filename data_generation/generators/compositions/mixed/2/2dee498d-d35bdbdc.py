"""Stable column reconstruction followed by disjoint scaffold composition.

Only standard-library and NumPy dependencies. No file I/O, saved examples, or
execution traces: generation and all diagnostic sweeps are bounded in memory.
"""
from __future__ import annotations

import heapq
import hashlib
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0,155685690,321371156,318990702,100469190,513648686,790882569,
              307467901,276350730,450521494,456830862,960836799,628966871,
              780307518,256005493,66586926,439198070,860607537,306656709,
              779491258,458499568,326814948,994274113,358533618,386704866,
              809794521,294576570,327891322,470521789,197837663,828740058,
              571588040,412890230,134301213,732321778,519983346,39263428,
              224662971,614772854,911454384,956032855,158721434)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid grid color')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _dedup(a):
    seen, indices = set(), []
    for j in range(a.shape[1]):
        key = a[:, j].tobytes()
        if key not in seen:
            indices.append(j)
            seen.add(key)
    return a[:, indices].copy(), indices


def _components(cells, directions=D4):
    todo, parts = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, part = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _support(center, kind):
    r, c = center
    offsets = {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if kind == 'cross':
        offsets = {(dr, dc) for dr, dc in offsets if not dr or not dc}
    elif kind == 'bevel0':
        offsets -= {(-1, 1), (1, -1)}
    elif kind == 'bevel1':
        offsets -= {(-1, -1), (1, 1)}
    elif kind != 'square':
        raise ValueError('unknown support')
    return {(r + dr, c + dc) for dr, dc in offsets}


def _graph(cells):
    # Eight-neighbor diagonal lines; suppress the diagonal chord of a pixel L.
    return {p: {q for dr, dc in D8
                if (q := (p[0] + dr, p[1] + dc)) in cells
                and not (dr and dc and
                         ((p[0] + dr, p[1]) in cells or (p[0], p[1] + dc) in cells))}
            for p in cells}


def _parse(a, background, line):
    token_cells = set(zip(*np.where((a != background) & (a != line))))
    tokens = []
    for part in _components(token_cells):
        rows, cols = zip(*part)
        _require(max(rows) - min(rows) == 2 and max(cols) - min(cols) == 2,
                 'token needs a complete 3x3 bounding box')
        center = (min(rows) + 1, min(cols) + 1)
        kinds = [kind for kind in ('square', 'cross', 'bevel0', 'bevel1')
                 if _support(center, kind) == part]
        _require(len(kinds) == 1, 'unrecognized token support')
        bodies = {int(a[p]) for p in part if p != center}
        _require(len(bodies) == 1, 'token body is not uniform')
        body = bodies.pop()
        value = int(a[center])
        _require(body != value, 'input tokens must be two-color')
        tokens.append(dict(cells=part, center=center, body=body, value=value, kind=kinds[0]))
    _require(2 <= len(tokens) <= 8, 'unsupported token count')
    cells = set(zip(*np.where(a == line)))
    _require(bool(cells), 'no scaffold')
    graph = _graph(cells)
    _require(all(len(q) in (1, 2) for q in graph.values()), 'scaffold must be open and nonbranching')
    pairs, used = [], set()
    for part in _components(cells, D8):
        ends = {p for p in part if len(graph[p]) == 1}
        _require(len(ends) == 2, 'each scaffold component needs two ends')
        contacts = []
        for end in sorted(ends):
            attached = [i for i, t in enumerate(tokens)
                        if any(end == (t['center'][0] + 2*dr, t['center'][1] + 2*dc)
                               for dr, dc in D4)]
            _require(len(attached) == 1, 'each end must attach to exactly one token centerline')
            contacts.append(attached[0])
        i, j = contacts
        _require(i != j and i not in used and j not in used, 'partnerships must be disjoint')
        used.update((i, j))
        forward = tokens[i]['value'] == tokens[j]['body']
        backward = tokens[j]['value'] == tokens[i]['body']
        _require(forward != backward, 'pair needs exactly one composition direction')
        pairs.append((i, j) if forward else (j, i))
    _require(1 <= len(pairs) <= 3, 'unsupported scaffold count')
    # Interior contacts cannot create a second spatial interpretation.
    for i, t in enumerate(tokens):
        touching_parts = [part for part in _components(cells, D8)
                          if any((r + dr, c + dc) in part
                                 for r, c in t['cells'] for dr, dc in D4)]
        _require(len(touching_parts) == (1 if i in used else 0), 'unintended scaffold contact')
    return tokens, pairs, cells


def _interpret(a):
    colors, counts = np.unique(a, return_counts=True)
    _require(sum(counts == counts.max()) == 1, 'background must be the unique modal color')
    bg = int(colors[np.argmax(counts)])
    interpretations = []
    for line in colors:
        if int(line) == bg:
            continue
        try:
            tokens, pairs, cells = _parse(a, bg, int(line))
        except ValueError:
            continue
        interpretations.append((bg, int(line), tokens, pairs, cells))
    _require(len(interpretations) == 1, 'scaffold color must be geometrically unique')
    return interpretations[0]


def _compose(a):
    bg, line, tokens, pairs, cells = _interpret(a)
    output = a.copy()
    retained = {i: tokens[j]['value'] for i, j in pairs}
    for i, t in enumerate(tokens):
        if i in retained:
            output[t['center']] = retained[i]
        else:
            for p in t['cells']:
                output[p] = bg
    return output, (bg, line, tokens, pairs, cells)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct exact first-seen columns, then read only the retained grid."""
    a, _ = _dedup(_grid(input_grid))
    _require(a.shape[0] == a.shape[1], 'retained scaffold must be square')
    return _compose(a)[0]


def _route(rng, n, blocked, start, end):
    if start == end or start in blocked or end in blocked:
        return None
    costs = rng.uniform(1., 1.8, size=(n, n))
    queue, distance, previous = [(0., start)], {start: 0.}, {}
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        if p == end:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1] if len(path) >= 2 else None
        for dr, dc in D8:
            q = p[0] + dr, p[1] + dc
            if not (0 <= q[0] < n and 0 <= q[1] < n) or q in blocked:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float('inf')):
                distance[q], previous[q] = new, p
                heapq.heappush(queue, (new, q))
    return None


def _near(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _layout(rng, n, pair_count, singles, mixed):
    """A spanning diagonal partner path makes column identities legible.

    Additional components use independently sampled positions and routed paths.
    Beveled end tokens break duplicate outer-column symmetry without an ID band.
    """
    count = 2*pair_count + singles
    for _ in range(90):
        centers = [(1, 1), (n-2, n-2)]
        kinds = [str(rng.choice(('bevel0', 'bevel1'))) for _ in range(2)]
        main = {(c-2, c) for c in range(3, n-1)}
        good = True
        for i in range(2, count):
            candidates = [(r, c) for r in range(3, n-3) for c in range(3, n-3)
                          if all(max(abs(r-a), abs(c-b)) >= 4 for a, b in centers)
                          and all(max(abs(r-a), abs(c-b)) >= 3 for a, b in main)]
            if not candidates:
                good = False
                break
            centers.append(candidates[int(rng.integers(len(candidates)))])
            kinds.append(str(rng.choice(('square', 'cross', 'bevel0', 'bevel1')))
                         if mixed else str(rng.choice(('bevel0', 'bevel1'))))
        if not good:
            continue
        if mixed and count >= 5:
            # The compulsory multi-pair demonstration teaches both optional
            # centered forms before either can occur in a test.
            kinds[2], kinds[3] = 'square', 'cross'
        supports = [_support(p, k) for p, k in zip(centers, kinds)]
        occupied = set().union(*supports)
        lines = set(main)
        for k in range(1, pair_count):
            i, j = 2*k, 2*k+1
            routed = False
            for _ in range(12):
                dr, dc = D4[int(rng.integers(4))]
                start = centers[i][0] + 2*dr, centers[i][1] + 2*dc
                dr, dc = D4[int(rng.integers(4))]
                end = centers[j][0] + 2*dr, centers[j][1] + 2*dc
                blocked = _near(occupied) | _near(lines)
                # Only the two intended token ports may bypass token margins.
                if start in _near(lines) or end in _near(lines):
                    continue
                blocked -= {start, end}
                path = _route(rng, n, blocked, start, end)
                if path is not None:
                    lines.update(path)
                    routed = True
                    break
            if not routed:
                good = False
                break
        if good:
            return centers, kinds, supports, lines
    raise ValueError('layout retry bound reached')


def _relations(rng, pair_count, singles, witness):
    colors = list(map(int, rng.permutation(8)))
    rel, kept = [], []
    # Conflicting repeated donor bodies distinguish spatial pairing from lookup.
    y = colors[1]
    for k in range(pair_count):
        if witness:
            x, z = colors[(2*k) % 8], colors[(2*k+2) % 8]
            if x == y or z == y:
                raise AssertionError('relation indexing error')
        else:
            x, middle, z = map(int, rng.choice(8, 3, replace=False))
            y = middle
        first, second = (x, y), (y, z)
        reverse = bool(rng.integers(2))
        rel.extend((second, first) if reverse else (first, second))
        kept.append((2*k + int(reverse), z))
    # An isolated duplicate of a valid receiver must still be erased.
    for k in range(singles):
        rel.append(rel[kept[k % pair_count][0]])
    return rel, kept


def _canonical(rng, n, pair_count, singles, mixed, roles, witness):
    for _ in range(32):
        try:
            centers, kinds, supports, lines = _layout(rng, n, pair_count, singles, mixed)
        except ValueError:
            continue
        bg, line = roles
        ids = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
        relations, kept = _relations(rng, pair_count, singles, witness)
        a = np.full((n, n), bg, dtype=np.int8)
        for p in lines:
            a[p] = line
        for p, support, (body, value) in zip(centers, supports, relations):
            for q in support:
                a[q] = ids[body]
            a[p] = ids[value]
        # Labels are constructed from sampled disjoint edges, independently of
        # the input-only segmentation and color-role inference used by solve.
        output = a.copy()
        retained = dict(kept)
        for i, support in enumerate(supports):
            if i in retained:
                output[centers[i]] = ids[retained[i]]
            else:
                for q in support:
                    output[q] = bg
        rotation = int(rng.integers(4))
        a, output = np.rot90(a, rotation), np.rot90(output, rotation)
        if rng.random() < .5:
            a, output = np.fliplr(a), np.fliplr(output)
        a, output = a.copy(), output.copy()
        if len(_dedup(a)[1]) != n:
            continue
        try:
            predicted, _ = _compose(a)
        except ValueError:
            continue
        if np.array_equal(predicted, output):
            return a, output
    raise ValueError('canonical retry bound reached')


def _expand(rng, a, mode, hard=False):
    n = a.shape[1]
    if mode == 'copy':
        # Two complete components, not just repeated individual token pixels.
        indices = list(range(n))*2
    else:
        budget = 30-n
        extra = int(rng.integers(min(budget, 5 if hard else 3), budget+1))
        indices = list(range(n))
        # Both adjacent and nonadjacent early repeats are explicitly sampled.
        source = int(rng.integers(0, n-2))
        insertion = int(rng.integers(source+1, n))
        indices.insert(insertion, source)
        for _ in range(extra-3):
            source = int(rng.integers(n))
            first = indices.index(source)
            insertion = int(rng.integers(first+1, len(indices)+1))
            indices.insert(insertion, source)
        # A late return to the first column distinguishes first-seen ordering
        # from stable last-occurrence filtering as well as periodic cropping.
        indices.extend((n-1, 0))
    return a[:, indices].copy()


def _count(value, lo, hi, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             name + ' must be an integer')
    _require(lo <= int(value) <= hi, name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, 'seed must be a nonnegative integer')
    nt = _count(num_train, 2, 10, 'num_train')
    ne = _count(num_test, 1, 4, 'num_test')
    _require(nt > ne, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    varied_roles = bool(rng.random() < .5)
    max_pairs = int(rng.choice((2, 3), p=(.75, .25)))
    mixed = bool(rng.random() < .65)
    # Minimum counts still teach both necessary contrasts. The order is shuffled.
    schedules = [('copy', int(rng.integers(12, 15)), 1, 1, True),
                 ('early', int(rng.integers(20 if max_pairs == 3 else 17, 24)),
                  max_pairs, 1, True)]
    for _ in range(nt-2):
        pairs = int(rng.integers(1, max_pairs+1))
        schedules.append(('early', int(rng.integers(16 if pairs == 1 else 20, 25)),
                          pairs, int(rng.integers(0, 3)), False))
    rng.shuffle(schedules)
    test_schedule = []
    for _ in range(ne):
        pairs = int(rng.integers(1, max_pairs+1))
        test_schedule.append(('early', int(rng.integers(18 if pairs == 1 else 21, 25)),
                              pairs, int(rng.integers(0, 3)), False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, schedule in (('train', schedules), ('test', test_schedule)):
        for mode, n, pairs, singles, witness in schedule:
            for _ in range(24):
                local_roles = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_roles else roles
                try:
                    canonical, output = _canonical(rng, n, pairs, singles, mixed, local_roles, witness)
                except ValueError:
                    continue
                inp = _expand(rng, canonical, mode, hard=split == 'test')
                if _key(inp) not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-pair generation failed')
            seen.add(_key(inp))
            puzzle[split].append(dict(input=inp, output=output))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    inp, output = _grid(pair['input']), _grid(pair['output'])
    canonical, indices = _dedup(inp)
    predicted, parsed = _compose(canonical)
    bg, line, tokens, pairs, cells = parsed
    _require(canonical.shape[0] == canonical.shape[1], 'canonical must be square')
    _require(output.shape == canonical.shape and np.array_equal(output, predicted), 'incorrect output')
    _require(inp.shape[1] > canonical.shape[1], 'missing column redundancy')
    _require(np.array_equal(output[canonical == line], canonical[canonical == line]), 'scaffold changed')
    token_cells = set().union(*(t['cells'] for t in tokens))
    _require(all(canonical[p] == output[p] for p in np.ndindex(canonical.shape) if p not in token_cells),
             'non-token cell changed')
    for i, j in pairs:
        t, donor = tokens[i], tokens[j]
        _require(output[t['center']] == donor['value'], 'wrong composition result')
        _require(t['value'] != donor['value'], 'composition must visibly change center')
    bodies = {}
    for t in tokens:
        bodies.setdefault(t['body'], set()).add(t['value'])
    n = canonical.shape[1]
    crop_fails = not np.array_equal(inp[:, :n], canonical)
    try:
        crop_composition_fails = not np.array_equal(_compose(inp[:, :n])[0], output)
    except ValueError:
        crop_composition_fails = True
    full_copy = inp.shape[1] == 2*n and np.array_equal(inp[:, :n], canonical) and np.array_equal(inp[:, n:], canonical)
    # Verify the omission-of-A witness really yields twice as many spatial pairs.
    doubled = False
    if full_copy:
        try:
            _, (_, _, raw_tokens, raw_pairs, _) = _compose(inp)
            doubled = len(raw_pairs) == 2*len(pairs) and len(raw_tokens) == 2*len(tokens)
        except ValueError:
            pass
    return dict(shape=inp.shape, size=n, tokens=len(tokens), pairs=len(pairs),
                singles=len(tokens)-2*len(pairs), kinds=tuple(sorted({t['kind'] for t in tokens})),
                roles=(bg, line), crop_fails=crop_fails, full_copy=doubled,
                crop_composition_fails=crop_composition_fails,
                conflicting_donors=any(len(v) > 1 for v in bodies.values()),
                # A color-compatible isolated receiver is still removed.
                isolated_compatible=any(i not in {v for p in pairs for v in p}
                                        and any(t['value'] == q['body'] for q in tokens)
                                        for i, t in enumerate(tokens)),
                extra=inp.shape[1]-n, duplicate_output_columns=len(_dedup(output)[1]) < n,
                layout=tuple(sorted(t['center'] for t in tokens)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'invalid pair counts')
    stats, seen = {}, set()
    for split in ('train', 'test'):
        stats[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp = _grid(pair['input'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            stats[split].append(_features(pair))
    train, test = stats['train'], stats['test']
    for feature in ('crop_fails', 'full_copy', 'conflicting_donors', 'isolated_compatible'):
        _require(any(f[feature] for f in train), 'missing training witness: ' + feature)
    _require(any(f['crop_composition_fails'] for f in train), 'crop-then-compose shortcut survives')
    trained_kinds = set().union(*(set(f['kinds']) for f in train))
    # The two bevel orientations have the same centered support convention.
    _require(all(set(f['kinds']) - {'bevel0', 'bevel1'} <= trained_kinds for f in test),
             'test introduces an undemonstrated support form')
    _require(max(f['pairs'] for f in test) <= max(f['pairs'] for f in train), 'untaught component count')
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 supplied seeds, streaming aggregate metrics, no disk artifacts."""
    signatures, shapes, layouts, roles, kinds, counts = set(), set(), set(), set(), set(), set()
    totals = {k: 0 for k in ('crop_fails', 'full_copy', 'conflicting_donors',
                             'isolated_compatible', 'duplicate_output_columns')}
    all_pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_fingerprint(p))
        _require(_fingerprint(p) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = np.asarray(pair['input'])
                saved = inp.copy()
                _require(np.array_equal(solve(inp), pair['output']), 'solve disagrees')
                _require(np.array_equal(inp, saved), 'solve mutated input')
                f = _features(pair)
                all_pairs += 1
                shapes.add(f['shape'])
                layouts.add((f['size'], f['layout']))
                roles.add(f['roles'])
                kinds.update(f['kinds'])
                counts.add((f['tokens'], f['pairs']))
                for name in totals:
                    totals[name] += bool(f[name])
        # No dependence on generating order, and palette roles are geometric.
        sample = p['train'][0]
        palette = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
        _require(np.array_equal(solve(palette[sample['input']]), palette[sample['output']]),
                 'color permutation changed semantics')
    _require(len(signatures) == 42, 'insufficient puzzle variation')
    _require(len(shapes) >= 25 and len(layouts) >= 100, 'insufficient geometric variation')
    _require(len(roles) >= 30 and len(kinds) == 4 and len(counts) >= 8, 'insufficient structural variation')
    _require(all(totals[k] >= 42 for k in ('crop_fails', 'full_copy', 'conflicting_donors', 'isolated_compatible')),
             'missing contrast coverage')
    _require(totals['duplicate_output_columns'] >= 10, 'second-dedup counterexamples absent')
    legal = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000 + nt*10 + ne, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), 'count not honored')
            legal += 1
    for nt, ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                   (4.,1),(4,1.5),(None,1),(4,None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0)
    bad = {s: [dict(input=q['input'].copy(), output=q['output'].copy()) for q in p[s]]
           for s in ('train','test')}
    c, _ = _dedup(bad['train'][0]['input'])
    _, line, _, _, cells = _interpret(c)
    bad['train'][0]['output'][min(cells)] = (line+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('scaffold corruption accepted')
    return dict(seeds=42, pairs=all_pairs, legal_count_combinations=legal,
                input_shapes=len(shapes), layouts=len(layouts), role_pairs=len(roles),
                token_count_pair_combinations=len(counts), features=totals)
