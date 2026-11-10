"""Delegates dissolve a chamber: panel majority -> side election -> set query.

Only standard-library modules and NumPy are used. Construction writes its own
labels; the input parser independently recovers panels, wires and port ballots.
"""
from collections import Counter
import heapq
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _reflect(a, corner):
    if corner & 2:
        a = a[::-1, :]
    if corner & 1:
        a = a[:, ::-1]
    return a


def _cells(box):
    r, c, h, w = box
    return {(x, y) for x in range(r, r+h) for y in range(c, c+w)}


def _neighbors(points, shape):
    h, w = shape
    return {(r+dr, c+dc) for r, c in points for dr, dc in _DIRS
            if 0 <= r+dr < h and 0 <= c+dc < w}


def _components(mask):
    left = set(zip(*np.nonzero(mask)))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r+dr, c+dc)
                if p in left:
                    left.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _box(points):
    rr, cc = zip(*points)
    return min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1


def _windows(mask):
    h, w = mask.shape
    found = np.ones((max(0, h-2), max(0, w-2)), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            found &= mask[dr:dr+h-2, dc:dc+w-2]
    return set(zip(*np.nonzero(found)))


def _ports(box):
    r, c, h, w = box
    # Clockwise, so consecutive owner groups admit planar routing.
    return [[(r-1, c+j) for j in range(w)],
            [(r+i, c+w) for i in range(h)],
            [(r+h, c+j) for j in range(w-1, -1, -1)],
            [(r+i, c-1) for i in range(h-1, -1, -1)]]


def _stamps(box):
    r, c, h, w = box
    return [(r, c+w//2), (r+h//2, c+w-1),
            (r+h-1, c+w//2), (r+h//2, c)]


def _strict_winner(colors):
    counts = Counter(colors)
    if not colors:
        return None
    winner, number = counts.most_common(1)[0]
    assert number * 2 > len(colors), "An election needs a strict majority"
    return int(winner)


def _analyze(input_grid):
    a = _check_grid(input_grid)
    assert min(a.shape) >= 9
    selectors = []
    for corner in range(4):
        v = _reflect(a, corner)
        s = int(v[3, 3])
        if not (np.all(v[3, :4] == s) and np.all(v[:4, 3] == s)):
            continue
        vals, counts = np.unique(v[:3, :3], return_counts=True)
        padding = vals[counts == 6]
        if len(padding) != 1:
            continue
        bg = int(padding[0])
        q = frozenset(int(x) for x in vals if x != bg)
        if len(q) == 3 and s not in q and s != bg:
            # A complete one-cell moat isolates the corner selector.
            if np.all(v[4, :5] == bg) and np.all(v[:5, 4] == bg):
                selectors.append((corner, bg, s, q))
    assert len(selectors) == 1, "Unique corner query required"
    corner, bg, s, query = selectors[0]
    v = _reflect(a, corner)
    structural = v == s
    structural[:4, :4] = False
    windows = _windows(structural)
    assert windows, "Missing chamber"
    wr, wc, wh, ww = _box(windows)
    box = wr, wc, wh+2, ww+2
    r, c, h, w = box
    assert h >= 5 and w >= 5 and h % 2 == w % 2 == 1
    assert 0 < r and 0 < c and r+h < v.shape[0] and c+w < v.shape[1]
    assert windows == _cells((r, c, h-2, w-2))
    assert np.all(v[r:r+h, c:c+w] == s)
    chamber = _cells(box)
    signal_mask = (v != bg) & (v != s)
    signal_mask[:4, :4] = False
    panel_sets = _components(signal_mask)
    assert panel_sets
    panel_ids = {}
    panels = []
    for i, points in enumerate(panel_sets):
        b = _box(points)
        assert points == _cells(b), "Source panels must be solid rectangles"
        count = Counter(int(v[p]) for p in points)
        assert len(count) == 2 and len(points) >= 3
        majority = _strict_winner([int(v[p]) for p in points])
        panels.append(dict(cells=points, box=b, counts=dict(count), majority=majority))
        panel_ids.update((p, i) for p in points)
    # Do not infer scaffold from frequency, or reject solid signal patches.
    routes_mask = structural.copy()
    routes_mask[r:r+h, c:c+w] = False
    routes = _components(routes_mask)
    sides = _ports(box)
    all_ports = set().union(*map(set, sides))
    owner_at = {}
    used = set()
    for route in routes:
        near = _neighbors(route, v.shape)
        owners = {panel_ids[p] for p in near if p in panel_ids}
        assert len(owners) == 1, "Each wire component has one source panel"
        owner = owners.pop()
        touches = route & _neighbors(chamber, v.shape)
        assert touches and touches <= all_ports
        for p in route:
            owner_at[p] = owner
        used.add(owner)
    assert used == set(range(len(panels)))
    votes, port_owners, delegates = [], [], []
    for side in sides:
        occupied = [p for p in side if v[p] == s]
        assert all(v[p] in (bg, s) for p in side)
        owners = [owner_at[p] for p in occupied]
        colors = [panels[i]['majority'] for i in owners]
        votes.append(colors)
        port_owners.append(owners)
        delegates.append(_strict_winner(colors))
    assert sum(d is not None for d in delegates) >= 3
    delegate_set = set(delegates) - {None}
    return dict(view=v, corner=corner, background=bg, scaffold=s, query=query,
                chamber=box, panels=panels, routes=routes, owner_at=owner_at,
                votes=votes, port_owners=port_owners, delegates=delegates,
                erased=query <= delegate_set)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover everything from input cells; no generator metadata is consulted."""
    info = _analyze(input_grid)
    out = np.asarray(input_grid).copy()
    v = _reflect(out, info['corner'])
    for p, owner in info['owner_at'].items():
        v[p] = info['panels'][owner]['majority']
    r, c, h, w = info['chamber']
    if info['erased']:
        v[r:r+h, c:c+w] = info['background']
    else:
        for p, delegate in zip(_stamps(info['chamber']), info['delegates']):
            if delegate is not None:
                v[p] = delegate
    return out


def _path(start, targets, allowed, costs):
    distances = {start: 0.0}
    previous = {}
    heap = [(0.0, start)]
    while heap:
        distance, p = heapq.heappop(heap)
        if distance != distances[p]:
            continue
        if p in targets:
            path = {p}
            while p != start:
                p = previous[p]
                path.add(p)
            return path
        r, c = p
        for dr, dc in _DIRS:
            q = r+dr, c+dc
            if 0 <= q[0] < allowed.shape[0] and 0 <= q[1] < allowed.shape[1] and allowed[q]:
                nd = distance + float(costs[q])
                if nd < distances.get(q, float('inf')):
                    distances[q] = nd
                    previous[q] = p
                    heapq.heappush(heap, (nd, q))
    return None


def _route(rng, shape, box, panel_boxes, groups, roughness):
    chamber = _cells(box)
    collar = _cells((box[0]-1, box[1]-1, box[2]+2, box[3]+2)) - chamber
    panel_cells = [_cells(b) for b in panel_boxes]
    all_panels = set().union(*panel_cells)
    selector = _cells((0, 0, 5, 5))
    networks = [set(g) for g in groups]
    order = sorted(range(len(groups)), key=lambda i: (-len(groups[i]), i))
    for owner in order:
        others = set().union(*(net for i, net in enumerate(networks) if i != owner))
        other_panels = all_panels - panel_cells[owner]
        blocked = (chamber | collar | all_panels | selector | others |
                   _neighbors(other_panels | others, shape)) - groups[owner]
        allowed = np.ones(shape, dtype=bool)
        for p in blocked:
            allowed[p] = False
        targets = _neighbors(panel_cells[owner], shape) - panel_cells[owner]
        targets = {p for p in targets if allowed[p]}
        if not targets:
            return None
        net = set()
        field = rng.random(((shape[0]+3)//4, (shape[1]+3)//4)) * roughness
        costs = 1 + rng.random(shape)*.7 + np.repeat(np.repeat(field, 4, 0), 4, 1)[:shape[0], :shape[1]]
        for p in sorted(groups[owner]):
            path = _path(p, targets | net, allowed, costs)
            if path is None:
                return None
            net |= path
        if net & collar != groups[owner]:
            return None
        networks[owner] = net
    mask = np.zeros(shape, dtype=bool)
    for p in chamber | set().union(*networks):
        mask[p] = True
    if _windows(mask) != _cells((box[0], box[1], box[2]-2, box[3]-2)):
        return None
    return networks


def _place_panels(rng, shape, box, count, small=False):
    h, w = shape
    occupied = (_cells((box[0]-2, box[1]-2, box[2]+4, box[3]+4)) |
                _cells((0, 0, 6, 6)))
    boxes = []
    for _ in range(count):
        for attempt in range(100):
            ph = int(rng.integers(1, 4 if small else 6))
            pw = int(rng.integers(2, 4 if small else 7))
            if ph*pw < 3:
                continue
            side = int(rng.integers(4))
            pad = 0 if rng.random() < .65 else int(rng.integers(1, 3))
            if side in (0, 2):
                r = pad if side == 0 else h-ph-pad
                c = int(rng.integers(w-pw+1))
            else:
                c = w-pw-pad if side == 1 else pad
                r = int(rng.integers(h-ph+1))
            b = r, c, ph, pw
            points = _cells(b)
            if points & occupied:
                continue
            boxes.append(b)
            occupied |= points | _neighbors(points, shape)
            break
        else:
            return None
    center = box[0]+(box[2]-1)/2, box[1]+(box[3]-1)/2
    boxes.sort(key=lambda b: float(np.arctan2(b[0]+(b[2]-1)/2-center[0],
                                             b[1]+(b[3]-1)/2-center[1])))
    return boxes


def _pattern(rng, box, majority, minority, contact=None):
    r, c, h, w = box
    n = h*w
    style = int(rng.integers(4))
    fraction = float(rng.choice((.12, .25, .40, .49)))
    minor_count = min((n-1)//2, max(1, int(n*fraction)))
    if style == 0:
        order = list(map(int, rng.permutation(n)))
    elif style == 1:
        ar, ac = int(rng.integers(h)), int(rng.integers(w))
        order = sorted(range(n), key=lambda i: (abs(i//w-ar)+abs(i%w-ac), i))
    elif style == 2:
        order = sorted(range(n), key=lambda i: ((i//w+i%w)%2, i))
    else:
        order = list(range(n)) if rng.random() < .5 else sorted(range(n), key=lambda i: (i%w, i//w))
    # A local minority at a wire contact defeats copying the touching pixel.
    if contact is not None:
        i = (contact[0]-r)*w + contact[1]-c
        order.remove(i)
        order.insert(0, i)
    flat = np.full(n, majority, dtype=np.int8)
    flat[order[:minor_count]] = minority
    return flat.reshape(h, w)


def _profile(rng, query, extra, box, package, richer):
    q = list(map(int, rng.permutation(list(query))))
    positive = package in ('positive', 'extras') or (package == 'free' and rng.random() < .5)
    if positive:
        fourth = extra if rng.random() < .35 else int(rng.choice(q))
        if package == 'positive':
            fourth = int(rng.choice(q))  # matched distinct-color-count witness
        elif package == 'extras':
            fourth = extra
        winners = q + [fourth]
    else:
        # Missing q[1] is nevertheless sent into the chamber as losing votes.
        winners = [q[0], q[2], extra, q[0]]
    if package not in ('negative', 'extras') and rng.random() < .30:
        winners[3] = None
    rotation = int(rng.integers(4))
    winners = winners[rotation:] + winners[:rotation]
    sides = _ports(box)
    # Training uses one contested side, tests recombine two or three.
    mix_count = int(rng.integers(2, 4)) if richer else 1
    if package == 'negative':
        witness_side = next(i for i, x in enumerate(winners)
                            if x == q[0] and len(sides[i]) >= 9)
    else:
        witness_side = int(rng.integers(4))
    if winners[witness_side] is None:
        witness_side = next(i for i, color in enumerate(winners) if color is not None)
    mixed = {witness_side}
    remaining = [s for s in range(4) if s != witness_side and winners[s] is not None]
    mixed.update(map(int, rng.choice(remaining, mix_count-1, replace=False)))
    groups, labels = [], []
    expected_delegates = winners.copy()
    for side_index, side in enumerate(sides):
        win = winners[side_index]
        if win is None:
            continue
        if package == 'negative' and side_index == witness_side:
            weights, colors = [3, 1, 1], [win, q[1], q[1]]
        elif side_index in mixed:
            loser = q[1] if not positive and win != q[1] else int(rng.choice([x for x in query+[extra] if x != win]))
            options = [[2, 1]]
            if len(side) >= 7:
                options.append([3, 1])
            if len(side) >= 9:
                options.append([3, 2])
            weights = options[int(rng.integers(len(options)))]
            colors = [win, loser]
        else:
            weights, colors = [int(rng.integers(1, min(3, (len(side)+1)//2)+1))], [win]
        total = sum(weights)
        available = list(range(0, len(side), 2))
        positions = sorted(map(int, rng.choice(available, total, replace=False)))
        # Owners may reverse order, without changing the side election.
        if rng.random() < .5:
            weights, colors = weights[::-1], colors[::-1]
        j = 0
        for weight, color in zip(weights, colors):
            groups.append({side[p] for p in positions[j:j+weight]})
            labels.append(color)
            j += weight
    # Adjacent sides can share one source, turning a duplicate delegate into
    # a genuinely branching network rather than two unrelated radial cables.
    if rng.random() < .45:
        center = box[0]+(box[2]-1)/2, box[1]+(box[3]-1)/2
        order = sorted(range(len(groups)), key=lambda i: float(np.arctan2(
            sum(p[0] for p in groups[i])/len(groups[i])-center[0],
            sum(p[1] for p in groups[i])/len(groups[i])-center[1])))
        side_of = {p: i for i, side in enumerate(sides) for p in side}
        candidates = []
        for a, b in zip(order, order[1:]+order[:1]):
            if labels[a] == labels[b] and side_of[next(iter(groups[a]))] != side_of[next(iter(groups[b]))]:
                candidates.append((a, b))
        if candidates:
            a, b = candidates[int(rng.integers(len(candidates)))]
            groups[a] |= groups[b]
            groups.pop(b)
            labels.pop(b)
    return groups, labels, expected_delegates, positive


def _make_pair(rng, roles, corner, package, richer, large):
    bg, s, *labels = roles
    query = labels[:3]
    extra = labels[3]
    for attempt in range(240):
        low = 25 if large else 22
        high = 30 if large else 28
        shape = tuple(map(int, rng.integers(low, high+1, 2)))
        if attempt >= 160:
            shape = (30, 30)
        bh, bw = map(int, rng.choice([5, 7, 9], 2))
        # The weighted training witness needs five separated ports on a side.
        if package == 'negative':
            bh = bw = 9
        r = int(rng.integers(5, shape[0]-bh-3))
        c = int(rng.integers(5, shape[1]-bw-3))
        box = r, c, bh, bw
        groups, colors, delegates, positive = _profile(rng, query, extra, box, package, richer)
        boxes = _place_panels(rng, shape, box, len(groups), small=attempt >= 80)
        if boxes is None:
            continue
        center = r+(bh-1)/2, c+(bw-1)/2
        def angle(points):
            rr = sum(p[0] for p in points)/len(points)
            cc = sum(p[1] for p in points)/len(points)
            return float(np.arctan2(rr-center[0], cc-center[1]))
        order = sorted(range(len(groups)), key=lambda i: angle(groups[i]))
        groups = [groups[i] for i in order]
        colors = [colors[i] for i in order]
        roughness = float(rng.choice([.0, .8, 2.5, 5.0]))
        nets = _route(rng, shape, box, boxes, groups, roughness)
        if nets is None:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        inp[3, :4] = s
        inp[:4, 3] = s
        for pos, color in zip(rng.choice(9, 3, replace=False), rng.permutation(query)):
            inp[int(pos)//3, int(pos)%3] = color
        inp[r:r+bh, c:c+bw] = s
        out = inp.copy()
        for b, network, majority in zip(boxes, nets, colors):
            cells = _cells(b)
            contacts = sorted(cells & _neighbors(network, shape))
            assert contacts
            minority = int(rng.choice([x for x in labels if x != majority]))
            pattern = _pattern(rng, b, majority, minority, contacts[0])
            pr, pc, ph, pw = b
            inp[pr:pr+ph, pc:pc+pw] = pattern
            out[pr:pr+ph, pc:pc+pw] = pattern
            for p in network:
                inp[p] = s
                out[p] = majority
        if positive:
            out[r:r+bh, c:c+bw] = bg
        else:
            for p, color in zip(_stamps(box), delegates):
                if color is not None:
                    out[p] = color
        pair = {'input': _reflect(inp, corner).copy(), 'output': _reflect(out, corner).copy()}
        info = _analyze(pair['input'])
        assert info['erased'] == positive and info['delegates'] == delegates
        assert np.array_equal(solve(pair['input']), pair['output'])
        return pair
    raise RuntimeError('Bounded planar layout search exhausted')


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _teaching_twins(rng, roles, corners):
    """Matched board, different query: geometry and scalar counts cannot decide."""
    negative = _make_pair(rng, roles, corners[0], 'negative', False, False)
    info = _analyze(negative['input'])
    inp = info['view'].copy()
    out = _reflect(negative['output'], corners[0]).copy()
    positions = list(zip(*np.nonzero(inp[:3, :3] != info['background'])))
    for p, color in zip(positions, rng.permutation(sorted(set(info['delegates'])))):
        inp[p] = out[p] = int(color)
    r, c, h, w = info['chamber']
    out[r:r+h, c:c+w] = info['background']
    positive = {'input': _reflect(inp, corners[1]).copy(),
                'output': _reflect(out, corners[1]).copy()}
    assert _analyze(positive['input'])['erased']
    return [negative, positive]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    varying_palette = bool(rng.integers(2))
    corners = list(map(int, rng.choice(4, 2, replace=False)))
    scale_shift = bool(rng.integers(2))
    teacher_roles = list(map(int, rng.choice(10, 6, replace=False))) if varying_palette else palette
    puzzle = {'train': _teaching_twins(rng, teacher_roles, corners), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train-2), ('test', num_test)):
        for i in range(int(count)):
            roles = list(map(int, rng.choice(10, 6, replace=False))) if varying_palette else palette
            corner = corners[i % 2] if split == 'train' else int(rng.choice(corners))
            for _ in range(12):
                package = 'extras' if split == 'train' and i == 0 else 'free'
                pair = _make_pair(rng, roles, corner, package,
                                  richer=split == 'test',
                                  large=split == 'test' and scale_shift)
                if split == 'test' and num_train == 2:
                    info = _analyze(pair['input'])
                    if info['erased'] and len(set(info['delegates']) - {None}) > 3:
                        continue
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Duplicate input search exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    order = rng.permutation(int(num_train))
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(info):
    delegate_set = set(info['delegates']) - {None}
    raw_set = set(x for side in info['votes'] for x in side)
    unweighted = []
    weight_conflict = False
    for votes, owners, delegate in zip(info['votes'], info['port_owners'], info['delegates']):
        source_votes = [info['panels'][i]['majority'] for i in sorted(set(owners))]
        count = Counter(source_votes)
        if count:
            best, n = count.most_common(1)[0]
            if 2*n > len(source_votes):
                unweighted.append(best)
                weight_conflict |= best != delegate
    owner_sides = {}
    for side, owners in enumerate(info['port_owners']):
        for owner in owners:
            owner_sides.setdefault(owner, set()).add(side)
    return dict(erased=info['erased'], raw_match=info['query'] <= raw_set,
                unweighted_match=info['query'] <= set(unweighted),
                weight_conflict=weight_conflict, delegate_count=len(delegate_set),
                mixed_sides=sum(len(set(x)) > 1 for x in info['votes']),
                multi_side=any(len(sides) > 1 for sides in owner_sides.values()),
                occupied_sides=sum(bool(x) for x in info['votes']),
                panel_count=len(info['panels']), ports=sum(map(len, info['votes'])),
                chamber_shape=info['chamber'][2:], route_cells=len(info['owner_at']),
                local_margins=tuple(abs(a-b) for p in info['panels'] for a,b in [tuple(p['counts'].values())]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements, corners = set(), {'train': [], 'test': []}, {'train': set(), 'test': set()}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(solve(inp), out), 'Incorrect vote/filter output'
            # Preserve every source pixel, including wire-touching minorities.
            ov = _reflect(out, info['corner'])
            for panel in info['panels']:
                assert all(ov[p] == info['view'][p] for p in panel['cells'])
            assert np.array_equal(ov[:4, :4], info['view'][:4, :4])
            assert np.all(ov[info['view'] == info['background']] == info['background'])
            assert not np.array_equal(inp, out)
            measurements[split].append(_measure(info))
            corners[split].add(info['corner'])
    train = measurements['train']
    assert {m['erased'] for m in train} == {False, True}
    assert any(not m['erased'] and m['raw_match'] and m['weight_conflict']
               and m['unweighted_match'] for m in train)
    assert any(m['erased'] and m['delegate_count'] == 3 for m in train)
    assert any(not m['erased'] and m['delegate_count'] == 3 for m in train)
    # The minimal demonstrations differ only in the query after reflecting
    # their selectors to the same corner. Both have the same donor and port
    # counts, chamber geometry, local minority patterns and wire lengths.
    witnesses = []
    for pair in puzzle['train']:
        info = _analyze(pair['input'])
        board = info['view'].copy()
        board[:3, :3] = info['background']
        witnesses.append((_key(board), info['erased']))
    assert any(k == j and outcome != other for k, outcome in witnesses
               for j, other in witnesses)
    assert max(m['mixed_sides'] for m in train) == 1
    assert min(m['mixed_sides'] for m in measurements['test']) >= 2
    if len(train) >= 3:
        assert any(m['erased'] and m['delegate_count'] == 4 for m in train)
    if any(m['erased'] and m['delegate_count'] == 4 for m in measurements['test']):
        assert any(m['erased'] and m['delegate_count'] == 4 for m in train)
    assert corners['test'] <= corners['train']


def self_test():
    # A valid family member outside the sampler's chamber and signal bounds.
    # Expected arrays are written directly from this simple hand-built scene.
    manual = np.zeros((21, 29), dtype=np.int8)
    manual[3, :4] = manual[:4, 3] = 1
    manual[0, 0], manual[1, 1], manual[2, 2] = 2, 3, 4
    manual[7:12, 8:19] = 1  # 5 by 11, not a generated 5/7/9 width
    records = [((1, 10, 2, 3), 2, {(r, 11) for r in range(3, 7)}),
               ((7, 24, 2, 3), 3, {(8, c) for c in range(19, 24)}),
               ((17, 14, 2, 4), 4, {(r, 15) for r in range(12, 17)}),
               ((8, 1, 3, 3), 6, {(9, c) for c in range(4, 8)})]
    expected = manual.copy()
    for b, majority, wire in records:
        r,c,h,w = b
        block = np.full(h*w, majority, dtype=np.int8)
        block[:(h*w-1)//3] = 5
        manual[r:r+h, c:c+w] = expected[r:r+h, c:c+w] = block.reshape(h,w)
        for p in wire:
            manual[p], expected[p] = 1, majority
    expected[7:12, 8:19] = 0
    assert np.array_equal(solve(manual), expected)
    manual[2, 2] = 5
    expected[2, 2] = 5
    expected[7:12, 8:19] = 1
    for p, color in zip(_stamps((7,8,5,11)), (2,3,4,6)):
        expected[p] = color
    assert np.array_equal(solve(manual), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, chambers, panels, ports, layouts, palettes, fingerprints = (set() for _ in range(7))
    outcomes, margins, occupied = set(), set(), set()
    multi_side_count = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        if seed < 2:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p,q in zip(puzzle[s], again[s]) for k in p)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                m = _measure(info)
                shapes.add(pair['input'].shape)
                chambers.add(m['chamber_shape'])
                panels.add(m['panel_count']); ports.add(m['ports'])
                layouts.add((pair['input'].shape, info['chamber'],
                             tuple(p['box'] for p in info['panels']),
                             tuple(tuple(x) for x in info['port_owners'])))
                palettes.add((info['background'], info['scaffold'], tuple(sorted(info['query']))))
                outcomes.add(m['erased']); margins.update(m['local_margins'])
                occupied.add(m['occupied_sides'])
                multi_side_count += m['multi_side']
        # Move query positions, preserving identities: decisions stay unchanged.
        pair = puzzle['train'][0]
        info = _analyze(pair['input'])
        changed = pair['input'].copy()
        _reflect(changed, info['corner'])[:3, :3] = np.roll(info['view'][:3, :3], 1, axis=0)
        expected = pair['output'].copy()
        _reflect(expected, info['corner'])[:3, :3] = _reflect(changed, info['corner'])[:3, :3]
        assert np.array_equal(solve(changed), expected)
        # On the deceptive negative, changing only the query to its delegates
        # makes the chamber disappear. Repetitions/positions cannot supply this.
        pair = next(p for p in puzzle['train']
                    if not _analyze(p['input'])['erased']
                    and _measure(_analyze(p['input']))['weight_conflict'])
        info = _analyze(pair['input'])
        colors = sorted(set(info['delegates']))
        assert len(colors) == 3
        changed = pair['input'].copy()
        cv = _reflect(changed, info['corner'])
        positions = list(zip(*np.nonzero(cv[:3, :3] != info['background'])))
        for p, color in zip(positions, colors):
            cv[p] = color
        result = _reflect(solve(changed), info['corner'])
        r,c,h,w = info['chamber']
        assert np.all(result[r:r+h, c:c+w] == info['background'])
        # All three query identities matter; replacing any one accepted
        # identity with the suppressed identity produces a surviving chamber.
        for absent_delegate in colors:
            changed = pair['input'].copy()
            cv = _reflect(changed, info['corner'])
            replacement = next(iter(info['query'] - set(info['delegates'])))
            trial_query = [x for x in colors if x != absent_delegate] + [replacement]
            for p, color in zip(positions, trial_query):
                cv[p] = color
            assert not _analyze(changed)['erased']
            prediction = _reflect(solve(changed), info['corner'])
            assert not np.all(prediction[r:r+h, c:c+w] == info['background'])
        # A local-majority change must propagate through every owned branch.
        # An independent duplicate supplies the former delegate; if it shares
        # this very source, both sides instead change together.
        for side_index, (owners, votes) in enumerate(zip(info['port_owners'], info['votes'])):
            if len(votes) == 5 and len(set(owners)) == 3:
                source_counts = Counter(owners)
                dominant = next(i for i, n in source_counts.items() if n == 3)
                missing = next(iter(info['query'] - set(info['delegates'])))
                changed = pair['input'].copy()
                cv = _reflect(changed, info['corner'])
                panel = info['panels'][dominant]
                former = panel['majority']
                for p in panel['cells']:
                    cv[p] = missing if info['view'][p] == former else former
                changed_info = _analyze(changed)
                prediction = _reflect(solve(changed), info['corner'])
                assert changed_info['delegates'][side_index] == missing
                assert all(prediction[p] == missing for p, owner in info['owner_at'].items()
                           if owner == dominant)
                if sum(dominant in os for os in info['port_owners']) == 1:
                    assert np.all(prediction[r:r+h, c:c+w] == info['background'])
                break
        else:
            raise AssertionError('Missing weighted local-vote witness')
        # Rearrangement changes panel pixels but no wire, election or filter.
        changed = pair['input'].copy()
        expected = pair['output'].copy()
        cv, ev = _reflect(changed, info['corner']), _reflect(expected, info['corner'])
        pr, pc, ph, pw = info['panels'][0]['box']
        block = cv[pr:pr+ph, pc:pc+pw]
        block[:] = np.sort(block, axis=None).reshape(block.shape)
        ev[pr:pr+ph, pc:pc+pw] = block
        assert np.array_equal(solve(changed), expected)
    assert len(fingerprints) == 42 and len(layouts) >= 168
    assert len(shapes) >= 35 and len(chambers) >= 7
    assert len(panels) >= 4 and len(ports) >= 7 and len(palettes) >= 40
    assert outcomes == {False, True} and len(margins) >= 12
    assert occupied == {3, 4} and multi_side_count >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(91, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.0,1),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k,v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    info = _analyze(bad['train'][0]['input'])
    point = next(iter(info['owner_at']))
    _reflect(bad['train'][0]['output'], info['corner'])[point] = info['scaffold']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect route accepted')
