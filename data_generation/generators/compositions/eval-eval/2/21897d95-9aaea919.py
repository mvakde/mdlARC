"""Counted paint votes: synchronous elections on annotated territories.

Only NumPy and the standard library are used. Solving reads the query pixels;
the sampler's labels, counts, palettes and random state never reach solve().
No files, traces, caches or logs are written by this module.
"""
from collections import defaultdict
import hashlib
import numpy as np


D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _need(bool(np.all((0 <= a) & (a <= 9))), "bad colors")
    return a.astype(np.int8, copy=False)


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(points):
    return set(points) | {_add(p, d) for p in points for d in D}


def _components(a):
    ids = np.full(a.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for p in np.ndindex(a.shape):
        if ids[p] >= 0:
            continue
        label, color = len(cells), int(a[p])
        ids[p] = label
        todo, group = [p], set()
        while todo:
            q = todo.pop()
            group.add(q)
            for d in D:
                r = _add(q, d)
                if _inside(r, a.shape) and ids[r] < 0 and a[r] == color:
                    ids[r] = label
                    todo.append(r)
        colors.append(color)
        cells.append(group)
    return ids, colors, cells


def _geometry(center, d, shape):
    p = (-d[1], d[0])
    tip = _add(center, d)
    arms = (_add(center, p), _add(center, p, -1))
    sides = (_add(tip, p), _add(tip, p, -1))
    target = _add(center, d, 2)
    write = {center, tip, *arms}
    context = write | set(sides) | {target}
    if not all(_inside(q, shape) for q in context):
        return None
    return dict(center=center, d=d, arms=arms, tip=tip, sides=sides,
                targetcell=target, write=write, context=context)


def _tally(colors, counts, arrows):
    incoming = [defaultdict(int) for _ in colors]
    for a in arrows:
        incoming[a['target']][a['payload']] += counts[a['source']]
    fills, final, winning = list(colors), list(counts), [0]*len(colors)
    for i, votes in enumerate(incoming):
        if votes:
            best = max(votes.values())
            winners = [c for c, n in votes.items() if n == best]
            _need(len(winners) == 1, "tied winning dyes")
            fills[i], final[i], winning[i] = winners[0], counts[i]+best, best
    return fills, final, winning, incoming


def _token_points(stack, mask, count):
    offsets = list(zip(*np.nonzero(mask)))
    pitch = mask.shape[0]+1
    return {(stack['bottom']-k*pitch+int(r), stack['x']+int(c))
            for k in range(count) for r, c in offsets}


def _stack_fits(ids, label, stack, mask, count):
    h, w = mask.shape
    top = stack['bottom']-(count-1)*(h+1)
    x, bottom = stack['x'], stack['bottom']+h
    return (top >= 1 and x >= 1 and bottom < ids.shape[0] and x+w < ids.shape[1]
            and np.all(ids[top-1:bottom+1, x-1:x+w+1] == label))


def _render(ids, fills, counts, stacks, mask, token):
    out = np.asarray(fills, dtype=np.int8)[ids].copy()
    for i, stack in enumerate(stacks):
        _need(_stack_fits(ids, i, stack, mask, counts[i]), "stack would cross a boundary")
        for p in _token_points(stack, mask, counts[i]):
            out[p] = token
    return out


def _parse_tokens(a, token, arrow_ink, arrows, token_groups):
    """Recover original fills by the uniform rim around each token component."""
    _need(bool(token_groups), "no tokens")
    clean, objects, masks = a.copy(), [], []
    for group in token_groups:
        r0, r1 = min(p[0] for p in group), max(p[0] for p in group)+1
        c0, c1 = min(p[1] for p in group), max(p[1] for p in group)+1
        _need(1 <= r1-r0 <= 3 and 2 <= c1-c0 <= 3 and 2 <= len(group) <= 9,
              "not a compact token")
        _need(r0 > 0 and c0 > 0 and r1 < a.shape[0] and c1 < a.shape[1], "token at edge")
        rim = _halo(group)-group
        hosts = {int(a[p]) for p in rim}
        _need(len(hosts) == 1 and token not in hosts and arrow_ink not in hosts, "unclear token rim")
        host = hosts.pop()
        mask = a[r0:r1, c0:c1] == token
        masks.append(mask)
        for p in group:
            clean[p] = host
        objects.append(dict(group=group, top=r0, bottom=r1, x=c0, host=host))
    mask = masks[0]
    _need(all(np.array_equal(mask, m) for m in masks), "tokens differ within scene")
    _need(token not in clean and arrow_ink not in clean, "unrestored annotation")
    ids, colors, cells = _components(clean)
    _need(2 <= len(cells) <= 6 and all(len(g) >= 18 for g in cells), "bad territory count/size")
    stacks, counts = [], []
    for i in range(len(cells)):
        parts = [o for o in objects if ids[next(iter(o['group']))] == i]
        _need(bool(parts), "territory has no stack")
        _need(len({o['x'] for o in parts}) == 1, "stack misalignment")
        ys = sorted(o['top'] for o in parts)
        _need(all(b-a == mask.shape[0]+1 for a, b in zip(ys, ys[1:])), "bad token pitch")
        stack = dict(x=parts[0]['x'], bottom=ys[-1])
        _need(_stack_fits(ids, i, stack, mask, len(parts)), "unclear stack room")
        stacks.append(stack)
        counts.append(len(parts))
    decoded = []
    for arow in arrows:
        source, target = int(ids[arow['center']]), int(ids[arow['targetcell']])
        _need(source != target and all(ids[p] == source for p in arow['write']), "bad arrow boundary")
        _need(arow['payload'] not in (token, arrow_ink), "payload uses annotation color")
        decoded.append(dict(arow, source=source, target=target))
    _need(len({(v['source'], v['target']) for v in decoded}) == len(decoded), "duplicate directed edge")
    fills, final, winning, incoming = _tally(colors, counts, decoded)
    output = _render(ids, fills, final, stacks, mask, token)
    info = dict(ids=ids, colors=colors, cells=cells, stacks=stacks, counts=counts,
                fills=fills, final=final, winning=winning, incoming=incoming,
                arrows=decoded, ink=arrow_ink, token=token, mask=mask, clean=clean)
    return output, info


def _decode(input_grid):
    grid = _grid(input_grid)
    raw_ids, raw_colors, raw_cells = _components(grid)
    answers = []
    for ink in sorted(set(raw_colors)):
        groups = [g for c, g in zip(raw_colors, raw_cells) if c == ink]
        if any(len(g) not in (1, 4) for g in groups):
            continue
        ink_cells = set().union(*groups)
        centers = {_add(p, d) for p in ink_cells for d in D}
        arrows = []
        for center in sorted(centers):
            for d in D:
                geo = _geometry(center, d, grid.shape)
                if geo is None or not all(grid[p] == ink for p in (*geo['arms'], geo['tip'])):
                    continue
                if any(not raw_cells[int(raw_ids[p])] <= geo['write'] for p in (*geo['arms'], geo['tip'])):
                    continue
                f, g = geo['sides']
                host = int(grid[f])
                if host == ink or grid[g] != host or grid[geo['targetcell']] in (ink, host):
                    continue
                explicit = int(grid[center]) != ink
                arrows.append(dict(geo, host=host, explicit=explicit,
                                   payload=int(grid[center]) if explicit else host))
        used, covered, valid = set(), set(), bool(arrows)
        for a in arrows:
            if used & a['write']:
                valid = False
            used.update(a['write'])
            covered.update(a['write'] & ink_cells)
        if not valid or covered != ink_cells:
            continue
        restored = grid.copy()
        for a in arrows:
            for p in a['write']:
                restored[p] = a['host']
        _, restored_colors, restored_groups = _components(restored)
        grouped = defaultdict(list)
        for color, group in zip(restored_colors, restored_groups):
            grouped[color].append(group)
        for token, token_groups in sorted(grouped.items()):
            if token == ink or any(not 2 <= len(g) <= 9 for g in token_groups):
                continue
            try:
                answers.append(_parse_tokens(restored, token, ink, arrows, token_groups))
            except AssertionError:
                pass
    _need(len(answers) == 1, "input does not have one complete interpretation")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every role and original component from the input alone."""
    return _decode(input_grid)[0]


def _runs(values):
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _sites(ids, mask):
    h, w = mask.shape
    result = []
    for label in range(int(ids.max())+1):
        choices = []
        for x in range(1, ids.shape[1]-w):
            good = np.all(ids[:, x-1:x+w+1] == label, axis=1)
            for start, stop in _runs(good):
                for bottom in range(int(start)+1, int(stop)-h):
                    cap = 1+(bottom-int(start)-1)//(h+1)
                    choices.append(dict(x=x, bottom=bottom, capacity=cap))
        result.append(choices)
    return result


def _partition(rng, side_counts, *, anchor=False, rotate=False):
    """Unequal, optionally stepped spine with independently subdivided sides."""
    nl, nr = side_counts
    height = int(rng.integers(26, 31)) if anchor else int(rng.integers(max(16, 8*max(nl, nr)), 31))
    widths = [int(rng.integers(7, 11)) if n else 0 for n in (nl, nr)]
    middle = int(rng.integers(6, min(10, 30-sum(widths))+1))
    width = sum(widths)+middle
    left, right = np.full(height, widths[0]), np.full(height, widths[0]+middle)
    rough = bool(rng.random() < .72)
    if rough:
        for profile, enabled in ((left, nl), (right, nr)):
            if enabled:
                step = int(rng.integers(3, 8))
                for r in range(0, height, step):
                    profile[r:r+step] += int(rng.integers(-1, 2))
    ids = np.zeros((height, width), dtype=np.int16)
    sides, next_id = [], 1
    for which, count in enumerate((nl, nr)):
        if not count:
            sides.append([])
            continue
        minimum = 12 if anchor else 7
        lengths = np.full(count, minimum, dtype=int)
        lengths += rng.multinomial(height-count*minimum, np.full(count, 1/count))
        bounds = np.r_[0, np.cumsum(lengths)]
        labels = list(range(next_id, next_id+count))
        for j, label in enumerate(labels):
            for r in range(int(bounds[j]), int(bounds[j+1])):
                if which == 0:
                    ids[r, :int(left[r])] = label
                else:
                    ids[r, int(right[r]):] = label
        sides.append(labels)
        next_id += count
    if rotate:
        ids = np.rot90(ids, int(rng.integers(4))).copy()
    return ids, sides


def _neighbors(ids):
    pairs = set()
    for a, b in ((ids[:-1], ids[1:]), (ids[:, :-1], ids[:, 1:])):
        for x, y in zip(a[a != b], b[a != b]):
            pairs.add((int(x), int(y)))
            pairs.add((int(y), int(x)))
    return pairs


def _colors(rng, ids, pool, repeated=None):
    n = int(ids.max())+1
    adjacent = _neighbors(ids)
    colors = [-1]*n
    if repeated:
        colors[repeated[0]] = colors[repeated[1]] = int(rng.choice(pool))
    for i in rng.permutation(n):
        i = int(i)
        if colors[i] >= 0:
            continue
        available = [c for c in pool if all(colors[j] != c for a, j in adjacent if a == i)]
        colors[i] = int(rng.choice(available))
    return colors


def _boundary_options(ids):
    result = defaultdict(list)
    for center in np.ndindex(ids.shape):
        for d in D:
            geo = _geometry(center, d, ids.shape)
            if geo is None:
                continue
            source, target = int(ids[center]), int(ids[geo['targetcell']])
            if source != target and all(ids[p] == source for p in geo['write'] | set(geo['sides'])):
                result[source, target].append(geo)
    return result


def _pack(rng, params, ids, colors, counts, edges, canonical=False):
    mask, ink, token = params['mask'], params['ink'], params['token']
    arrows = [dict(source=s, target=t, payload=p) for s, t, p in edges]
    fills, final, _, _ = _tally(colors, counts, arrows)
    sites = _sites(ids, mask)
    eligible = [[s for s in choices if s['capacity'] >= need] for choices, need in zip(sites, final)]
    if not all(eligible):
        return None
    candidates = _boundary_options(ids)
    if canonical:
        centers = [np.argwhere(ids == i).mean(axis=0) for i in range(len(colors))]
        for source, target in list(candidates):
            axis = 1 if 0 in (source, target) else 0
            sign = 1 if centers[target][axis] > centers[source][axis] else -1
            direction = (0, sign) if axis else (sign, 0)
            candidates[source, target] = [g for g in candidates[source, target] if g['d'] == direction]
    for _ in range(32):
        stacks = [choices[int(rng.integers(len(choices)))] for choices in eligible]
        reserved = set()
        for stack, n in zip(stacks, final):
            reserved.update(_halo(_token_points(stack, mask, n)))
        available = [[g for g in candidates[a['source'], a['target']] if not g['context'] & reserved]
                     for a in arrows]
        if not all(available):
            continue
        chosen, writes, contexts = {}, set(), set()
        for index in sorted(range(len(arrows)), key=lambda i: len(available[i])):
            opts = [g for g in available[index] if not (g['context'] & _halo(writes) or g['write'] & contexts)]
            if not opts:
                break
            geo = opts[int(rng.integers(len(opts)))]
            chosen[index] = geo
            writes.update(geo['write'])
            contexts.update(geo['context'])
        if len(chosen) != len(arrows):
            continue
        inp = _render(ids, colors, counts, stacks, mask, token)
        out = _render(ids, fills, final, stacks, mask, token)
        for i, arrow in enumerate(arrows):
            geo = chosen[i]
            for p in geo['write']:
                inp[p] = ink
            if arrow['payload'] != colors[arrow['source']]:
                inp[geo['center']] = arrow['payload']
        try:
            predicted, info = _decode(inp)
        except AssertionError:
            continue
        if np.array_equal(predicted, out):
            return dict(input=inp, output=out), info
    return None


def _anchor(rng, params, kind, resident, previous_operand=None):
    for _ in range(100):
        side_counts = (2, 1) if rng.integers(2) else (1, 2)
        ids, sides = _partition(rng, side_counts, anchor=True)
        split = next(s for s in sides if len(s) == 2)
        full = next(s[0] for s in sides if len(s) == 1)
        a, b = split if params['vertical'] == 1 else split[::-1]
        # Opposite-side territories share a fill but retain separate identities.
        colors = _colors(rng, ids, params['pool'], repeated=(a, full))
        capacities = [max((s['capacity'] for s in choices), default=0) for choices in _sites(ids, params['mask'])]
        options = []
        for na in range(1, min(5, capacities[a])+1):
            for nb in range(1, min(5, capacities[b])+1):
                for nc in range(1, min(5, capacities[full])+1):
                    weight = na+nb if kind == 'aggregate' else nc
                    condition = max(na, nb) < nc < na+nb if kind == 'aggregate' else na+nb < nc
                    if (condition and resident+weight <= capacities[0]
                            and nc+resident <= capacities[full]
                            and (kind != 'minority' or nb+na <= capacities[b])
                            and (previous_operand is None or
                                 (weight != previous_operand[1] and
                                  weight*previous_operand[0] != previous_operand[1]*resident))):
                        options.append((na, nb, nc))
        if not options:
            continue
        na, nb, nc = options[int(rng.integers(len(options)))]
        counts = [resident]*4
        counts[a], counts[b], counts[full] = na, nb, nc
        novel = [c for c in params['pool'] if c not in colors]
        # A literal dye absent from all restored territories wins in both anchors.
        g = int(rng.choice(novel))
        h = int(rng.choice([c for c in params['pool'] if c not in (g, colors[0])]))
        dye_ab, dye_c = (g, h) if kind == 'aggregate' else (h, g)
        edges = [(a, 0, dye_ab), (b, 0, dye_ab), (full, 0, dye_c), (0, full, colors[0])]
        if kind == 'minority':
            edges.append((a, b, colors[a]))
        result = _pack(rng, params, ids, colors, counts, edges, canonical=True)
        if result is not None:
            return result
    raise AssertionError('bounded anchor construction exhausted')


def _test_pair(rng, params):
    # Two simultaneous elections, a shared voter, and a changed intermediate.
    for _ in range(100):
        ids, sides = _partition(rng, (2, 2), anchor=True)
        side = int(rng.integers(2))
        a, recipient = sides[side] if params['vertical'] == 1 else sides[side][::-1]
        b, c = sides[1-side]
        if rng.integers(2):
            b, c = c, b
        colors = _colors(rng, ids, params['pool'], repeated=(a, b))
        caps = [max((s['capacity'] for s in choices), default=0) for choices in _sites(ids, params['mask'])]
        options = []
        for n0 in range(2, min(4, caps[0])+1):
            for na in range(1, min(n0-1, caps[a])+1):
                for nb in range(1, min(4, caps[b])+1):
                    for nc in range(1, min(4, caps[c])+1):
                        for nd in range(1, min(3, caps[recipient])+1):
                            if (na+nb != nc and n0+max(na+nb, nc) <= caps[0]
                                    and nd+n0 <= caps[recipient]):
                                options.append((n0, na, nb, nc, nd))
        if not options:
            continue
        n0, na, nb, nc, nd = options[int(rng.integers(len(options)))]
        counts = [1]*5
        counts[0], counts[a], counts[b], counts[c], counts[recipient] = n0, na, nb, nc, nd
        g, h = [int(v) for v in rng.choice([v for v in params['pool'] if v != colors[0]], 2, replace=False)]
        other = int(rng.choice([v for v in params['pool'] if v != colors[0]]))
        edges = [(a, 0, g), (b, 0, g), (c, 0, h),
                 (0, recipient, colors[0]), (a, recipient, other)]
        result = _pack(rng, params, ids, colors, counts, edges, canonical=True)
        if result is not None:
            return result
    raise AssertionError('bounded two-election construction exhausted')


def _extra_pair(rng, params):
    for _ in range(120):
        layouts = ((0, 1), (1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 2))
        side_counts = layouts[int(rng.integers(len(layouts)))]
        ids, _ = _partition(rng, side_counts, rotate=True)
        colors = _colors(rng, ids, params['pool'])
        sites = _sites(ids, params['mask'])
        caps = [max((s['capacity'] for s in choices), default=0) for choices in sites]
        if min(caps) < 1:
            continue
        counts = [int(rng.integers(1, min(4, cap)+1)) for cap in caps]
        adjacent = sorted(_neighbors(ids))
        for _ in range(16):
            size = int(rng.integers(1, min(6, len(adjacent))+1))
            edges = []
            for index in rng.choice(len(adjacent), size, replace=False):
                s, t = adjacent[int(index)]
                payload = colors[s] if rng.random() < .6 else int(rng.choice(params['pool']))
                edges.append((s, t, payload))
            if sum(sum(t == i for s, t, p in edges) > 1 for i in range(len(colors))) > 1:
                continue
            try:
                _, final, _, _ = _tally(colors, counts, [dict(source=s, target=t, payload=p) for s, t, p in edges])
            except AssertionError:
                continue
            if any(n > cap for n, cap in zip(final, caps)):
                continue
            result = _pack(rng, params, ids, colors, counts, edges)
            if result is not None:
                return result
    raise AssertionError('bounded extra construction exhausted')


def _sample_mask(rng):
    h, w = int(rng.integers(1, 4)), int(rng.integers(2, 4))
    mask = np.ones((h, w), dtype=bool)
    if h >= 2 and w == 3 and rng.random() < .55:
        # Full bottom row and left spine keep every procedural silhouette connected.
        for r in range(h-1):
            mask[r, int(rng.integers(1, w+1)):] = False
        if rng.integers(2):
            mask = mask[:, ::-1]
        if rng.integers(2):
            mask = mask[::-1]
    # A four-cell T would be ambiguous with arrow ink: use a block instead.
    pts = set(zip(*np.nonzero(mask)))
    if len(pts) == 4 and any(pts == {p, _add(p, d), _add(p, (-d[1], d[0])), _add(p, (d[1], -d[0]))}
                             for p in pts for d in D):
        mask[:] = True
    return mask.copy()


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not 2 <= num_train <= 10 or
            not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('need integer counts: 2..10 train, 1..4 test, train > test')
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10).tolist()
    params = dict(ink=palette[0], token=palette[1], pool=palette[2:], mask=_sample_mask(rng),
                  vertical=int(rng.choice([-1, 1])))
    residents = rng.permutation([1, 2]).tolist()
    first, info = _anchor(rng, params, 'aggregate', residents[0])
    contested = next(i for i, v in enumerate(info['incoming']) if len(v) > 1)
    second, _ = _anchor(rng, params, 'minority', residents[1],
                        (residents[0], info['winning'][contested]))
    puzzle = dict(train=[first, second], test=[])
    seen = {_key(first['input']), _key(second['input'])}
    _need(len(seen) == 2, 'duplicate anchor inputs')
    for split, count, builder in (('train', int(num_train)-2, _extra_pair), ('test', int(num_test), _test_pair)):
        for _ in range(count):
            for attempt in range(24):
                pair, _ = builder(rng, params)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded uniqueness construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    arrows, counts = info['arrows'], info['counts']
    aggregate = minority = largest_wrong = False
    operands = []
    for i, votes in enumerate(info['incoming']):
        if not votes:
            continue
        incoming = [a for a in arrows if a['target'] == i]
        winner = info['fills'][i]
        bars = defaultdict(int)
        for a in incoming:
            bars[a['payload']] += 1
        aggregate |= len(votes) > 1 and bars[winner] >= 2
        minority |= len(votes) > 1 and bars[winner] < max(bars.values())
        largest = max(counts[a['source']] for a in incoming)
        largest_colors = {a['payload'] for a in incoming if counts[a['source']] == largest}
        largest_wrong |= len(votes) > 1 and winner not in largest_colors
        if len(votes) > 1:
            operands.append((counts[i], info['winning'][i]))
    changed = {i for i in range(len(counts)) if info['fills'][i] != info['colors'][i]
               and info['final'][i] != counts[i]}
    dependency = any(a['source'] in changed and not a['explicit'] for a in arrows)
    locality = any(info['colors'][i] == info['colors'][j] and info['fills'][i] != info['fills'][j]
                   for i in range(len(counts)) for j in range(i))
    return dict(aggregate=aggregate, minority=minority, largest_wrong=largest_wrong,
                dependency=dependency, locality=locality, operands=operands,
                contested=sum(len(v) > 1 for v in info['incoming']),
                novel=any(a['explicit'] and a['payload'] not in info['colors'] for a in arrows),
                explicit=any(a['explicit'] for a in arrows), implicit=any(not a['explicit'] for a in arrows),
                directions={a['d'] for a in arrows}, regions=len(counts), arrows=len(arrows),
                nonrectangular=any(len(g) != (max(r for r, c in g)-min(r for r, c in g)+1)*
                                  (max(c for r, c in g)-min(c for r, c in g)+1) for g in info['cells']))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'bad example counts')
    seen, conventions, features = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape and _key(inp) not in seen, 'shape change or duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'wrong election, growth, restoration or preservation')
            _need(not np.array_equal(inp, out), 'identity pair')
            conventions.add((info['ink'], info['token'], info['mask'].shape, info['mask'].tobytes()))
            features[split].append(_features(info))
    _need(len(conventions) == 1, 'puzzle conventions changed')
    taught = features['train']
    for k in ('aggregate', 'minority', 'largest_wrong', 'dependency', 'locality', 'novel', 'implicit'):
        _need(any(f[k] for f in taught), 'missing informative demonstration: '+k)
    operands = [v for f in taught for v in f['operands']]
    _need(len({d for d, w in operands}) >= 2 and len({w for d, w in operands}) >= 2,
          'constant arithmetic operands')
    _need(any(d*operands[0][1] != w*operands[0][0] for d, w in operands), 'constant destination multiplier')
    _need(any(d+w != 2*d for d, w in operands), 'doubling shortcut')
    _need(any(d+w != d*w for d, w in operands), 'multiplication shortcut')
    train_dirs = set().union(*(f['directions'] for f in taught))
    for f in features['test']:
        _need(f['contested'] == 2 and f['dependency'], 'missing interacting-election test shift')
        _need(f['directions'] <= train_dirs, 'test direction absent from demonstrations')


def self_test():
    """Bounded 42-seed coverage, explicit counts, semantic mutations, no disk I/O."""
    signatures, shapes, masks, roles, regions, edges, directions = set(), set(), set(), set(), set(), set(), set()
    counts = defaultdict(int)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if seed < 2:
            _need(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _decode(pair['input'])
                _need(np.array_equal(out, pair['output']), 'input-only solve failure')
                f = _features(info)
                shapes.add(out.shape)
                masks.add((info['mask'].shape, info['mask'].tobytes()))
                roles.add((info['ink'], info['token']))
                regions.add(f['regions'])
                edges.add(f['arrows'])
                directions.update(f['directions'])
                for name in ('aggregate', 'minority', 'dependency', 'locality', 'novel', 'nonrectangular'):
                    counts[name] += int(f[name])
    _need(len(signatures) == 42 and len(shapes) >= 20 and len(masks) >= 6 and len(roles) >= 20,
          'collapsed seed/geometry/palette variation')
    _need({2, 3, 4, 5} <= regions and len(edges) >= 4 and directions == set(D), 'collapsed graph variation')
    _need(all(n >= 15 for n in counts.values()), 'missing semantic/shape coverage')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(10000+a*10+b, a, b)
            _need((len(p['train']), len(p['test'])) == (a, b), 'explicit counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Full-output mutations: wrong dye, lost copy, retained ink, changed untouched area.
    original = generate_puzzle(0, 2, 1)
    # Losing-group ties are allowed, but a tied maximum has no family answer.
    try:
        _tally([2, 3, 4], [1, 1, 1], [dict(source=1, target=0, payload=5),
                                     dict(source=2, target=0, payload=6)])
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted tied maximum')
    _need(_tally([2, 3, 4, 7], [1, 2, 1, 1],
                 [dict(source=1, target=0, payload=5), dict(source=2, target=0, payload=6),
                  dict(source=3, target=0, payload=8)])[0][0] == 5, 'rejected harmless losing tie')
    for defect in ('dye', 'growth', 'ink', 'untargeted'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in original[s]]
               for s in ('train', 'test')}
        pair = bad['train'][0]
        _, info = _decode(pair['input'])
        target = next(i for i, votes in enumerate(info['incoming']) if len(votes) > 1)
        if defect == 'dye':
            p = next(p for p in info['cells'][target] if pair['output'][p] != info['token'])
            pair['output'][p] = info['colors'][target]
        elif defect == 'growth':
            new = _token_points(info['stacks'][target], info['mask'], info['final'][target])
            old = _token_points(info['stacks'][target], info['mask'], info['counts'][target])
            pair['output'][next(iter(new-old))] = info['fills'][target]
        elif defect == 'ink':
            p = next(iter(info['arrows'][0]['write']))
            pair['output'][p] = info['ink']
        else:
            i = next(i for i, v in enumerate(info['incoming']) if not v)
            p = next(p for p in info['cells'][i] if pair['output'][p] != info['token'])
            pair['output'][p] = info['ink']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted semantic mutation: '+defect)
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes), masks=len(masks),
                role_assignments=len(roles), region_counts=sorted(regions), arrow_counts=sorted(edges),
                feature_pairs=dict(counts))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
