"""Simultaneous, port-weighted composition of majority-encoded relations.

Only NumPy and the standard library are used. Generation records are discarded;
solve reconstructs the complete graph and every color role from its input.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a


def _components(mask):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    parts = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if labels[r, c] >= 0:
            continue
        index = len(parts)
        labels[r, c] = index
        todo, cells = [(r, c)], []
        while todo:
            p = todo.pop()
            cells.append(p)
            for dr, dc in _D4:
                q = p[0] + dr, p[1] + dc
                if (0 <= q[0] < h and 0 <= q[1] < w
                        and mask[q] and labels[q] < 0):
                    labels[q] = index
                    todo.append(q)
        parts.append(cells)
    return parts, labels


def _strict_vote(ballots):
    """None means no votes; -1 means a nonempty electorate without a majority."""
    counts = Counter()
    for color, weight in ballots:
        counts[int(color)] += int(weight)
    if not counts:
        return None
    color, count = counts.most_common(1)[0]
    return color if 2 * count > sum(counts.values()) else -1


def _parse_roles(a, bg, wire):
    parts, labels = _components((a != bg) & (a != wire))
    assert len(parts) >= 2
    nodes = []
    for cells in parts:
        rs, cs = zip(*cells)
        r, c = min(rs), min(cs)
        h, w = max(rs) - r + 1, max(cs) - c + 1
        assert h >= 5 and w >= 5 and h % 2 and w % 2
        assert len(cells) == h * w
        block = a[r:r+h, c:c+w]
        frame = np.concatenate((block[0], block[-1], block[1:-1, 0], block[1:-1, -1]))
        assert np.all(frame == frame[0])
        interior = block[1:-1, 1:-1]
        colors, counts = np.unique(interior, return_counts=True)
        assert len(colors) == 2 and counts[0] != counts[1]
        nodes.append(dict(box=(r, c, h, w), body=int(frame[0]),
                          value=int(colors[np.argmax(counts)]),
                          center=int(a[r+h//2, c+w//2]),
                          minority=int(counts.min()), area=int(interior.size)))
    cables, _ = _components(a == wire)
    assert cables
    edges = []
    neighbors = [[] for _ in nodes]
    for cells in cables:
        contacts = Counter()
        for r, c in cells:
            adjacent = set()
            for dr, dc in _D4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and labels[rr, cc] >= 0:
                    adjacent.add(int(labels[rr, cc]))
            assert len(adjacent) <= 1, "one port may not touch two panels"
            contacts.update(adjacent)
        assert len(contacts) == 2, "each cable must join two panels"
        i, j = sorted(contacts)
        wi, wj = contacts[i], contacts[j]
        edges.append(dict(a=i, b=j, wa=wi, wb=wj, cells=cells))
        neighbors[i].append((j, wi))
        neighbors[j].append((i, wj))
    reached, todo = {0}, [0]
    while todo:
        for j, _ in neighbors[todo.pop()]:
            if j not in reached:
                reached.add(j)
                todo.append(j)
    assert len(reached) == len(nodes), "one shared connected graph is required"
    info = dict(background=bg, wire=wire, nodes=nodes, edges=edges, neighbors=neighbors)
    winners = _evaluate(info)
    assert -1 not in winners, "positive port electorates must have strict majorities"
    info['winners'] = winners
    return info


def _inspect(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) < 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent color")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for wire in colors:
        if wire == bg:
            continue
        try:
            candidates.append(_parse_roles(a, bg, int(wire)))
        except AssertionError:
            continue
    if len(candidates) != 1:
        raise ValueError("expected one geometric cable/panel interpretation")
    return candidates[0]


def _evaluate(info, *, weighted=True, gated=True, centers=False, global_scope=False,
              donor_values=None):
    nodes = info['nodes']
    values = [n['center'] if centers else n['value'] for n in nodes]
    donated = values if donor_values is None else donor_values
    result = []
    for i, node in enumerate(nodes):
        candidates = ([(j, 1) for j in range(len(nodes)) if j != i]
                      if global_scope else info['neighbors'][i])
        ballots = [(donated[j], weight if weighted else 1) for j, weight in candidates
                   if not gated or nodes[j]['body'] == values[i]]
        result.append(_strict_vote(ballots))
    return result


def _paint(a, info):
    out = a.copy()
    for node, winner in zip(info['nodes'], info['winners']):
        r, c, h, w = node['box']
        if winner is None:
            out[r:r+h, c:c+w] = info['background']
        else:
            out[r+1:r+h-1, c+1:c+w-1] = winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _paint(a, _inspect(a))


def _features(a, info=None):
    info = _inspect(a) if info is None else info
    win, nodes = info['winners'], info['nodes']
    unit = _evaluate(info, weighted=False)
    ungated = _evaluate(info, gated=False)
    center = _evaluate(info, centers=True)
    global_w = _evaluate(info, weighted=False, global_scope=True)
    updated = [n['value'] if v is None else v for n, v in zip(nodes, win)]
    premature = _evaluate(info, donor_values=updated)
    differs = lambda alt: any(x is not None and y is not None and y >= 0 and x != y
                             for x, y in zip(win, alt))
    erased_donors = surviving_donors = 0
    for i, neighbors in enumerate(info['neighbors']):
        if win[i] is None:
            continue
        for j, _ in neighbors:
            if nodes[j]['body'] == nodes[i]['value']:
                erased_donors += win[j] is None
                surviving_donors += win[j] is not None and win[j] != nodes[j]['value']
    scope_witness = any(
        v is not None and unit[i] == v and global_w[i] is not None and global_w[i] >= 0
        and global_w[i] != v and any(
            j != i and nodes[j]['body'] == nodes[i]['value']
            and j not in {neighbor for neighbor, _ in info['neighbors'][i]}
            for j in range(len(nodes))) for i, v in enumerate(win))
    contested = sum(len({nodes[j]['value'] for j, _ in neighbors
                         if nodes[j]['body'] == nodes[i]['value']}) >= 2
                    for i, neighbors in enumerate(info['neighbors']))
    return dict(nodes=len(nodes), edges=len(info['edges']), shape=tuple(a.shape),
                survivors=sum(v is not None for v in win),
                weighted_witness=differs(unit), gating_witness=differs(ungated),
                center_witness=center != win, scope_witness=scope_witness,
                contested_receivers=contested,
                simultaneous_witness=differs(premature),
                erased_donors=erased_donors, changed_surviving_donors=surviving_donors,
                solid_results=sum(v == n['body'] for n, v in zip(nodes, win)),
                majority_margins=tuple(n['area'] - 2*n['minority'] for n in nodes),
                core_areas=tuple(n['area'] for n in nodes),
                degrees=tuple(sorted(map(len, info['neighbors']))),
                ports=tuple(w for neighbors in info['neighbors'] for _, w in neighbors),
                palette=(info['background'], info['wire']),
                layout=tuple(n['box'] for n in nodes),
                cycle=len(info['edges']) >= len(nodes))


def _axis(p, q):
    assert p[0] == q[0] or p[1] == q[1]
    if p[0] == q[0]:
        return {(p[0], c) for c in range(min(p[1], q[1]), max(p[1], q[1])+1)}
    return {(r, p[1]) for r in range(min(p[0], q[0]), max(p[0], q[0])+1)}


def _sites(rng, mode, test):
    if mode == 'branch':
        return [(1, 1), (0, 1), (1, 2), (2, 1), (1, 0)], [(0, i) for i in range(1, 5)], 3, 3
    if mode == 'path':
        n = int(rng.integers(5, 7))
        nr, nc = (2, 3) if rng.random() < .5 else (3, 3)
        all_sites = [(r, c) for r in range(nr) for c in range(nc)]
        # At most nine sites: a small randomized depth-first search is bounded.
        def extend(path):
            if len(path) == n:
                return path
            p = path[-1]
            choices = [q for q in all_sites if q not in path
                       and abs(p[0]-q[0])+abs(p[1]-q[1]) == 1]
            rng.shuffle(choices)
            for q in choices:
                answer = extend(path+[q])
                if answer is not None:
                    return answer
            return None
        starts = list(all_sites)
        rng.shuffle(starts)
        for p in starts:
            sites = extend([p])
            if sites is not None:
                return sites, [(i, i+1) for i in range(n-1)], nr, nc
        raise RuntimeError("bounded path construction failed")
    n = int(rng.integers(6, 9) if test else rng.integers(4, 8))
    if n == 4 and rng.random() < .6:
        nr, nc = 2, 2
    elif n <= 6 and rng.random() < .5:
        nr, nc = 2, 3
    else:
        nr, nc = 3, 3
    sites = [(int(rng.integers(nr)), int(rng.integers(nc)))]
    while len(sites) < n:
        frontier = sorted({(r+dr, c+dc) for r, c in sites for dr, dc in _D4
                           if 0 <= r+dr < nr and 0 <= c+dc < nc and (r+dr, c+dc) not in sites})
        sites.append(frontier[int(rng.integers(len(frontier)))])
    possibilities = [(i, j) for i in range(n) for j in range(i+1, n)
                     if sum(abs(x-y) for x, y in zip(sites[i], sites[j])) == 1]
    rng.shuffle(possibilities)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    edges, extras = [], []
    for i, j in possibilities:
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[ri] = rj
            edges.append((i, j))
        else:
            extras.append((i, j))
    for edge in extras:
        if rng.random() < (.65 if test else .25):
            edges.append(edge)
    return sites, edges, nr, nc


def _strips(rng, count):
    sizes = list(map(int, rng.choice([5, 7], count)))
    gaps = [int(rng.integers(3, 6)) for _ in range(count-1)]
    while sum(sizes) + sum(gaps) > 30:
        index = next(i for i, g in enumerate(gaps) if g > 3)
        gaps[index] -= 1
    spare = 30 - sum(sizes) - sum(gaps)
    before = int(rng.integers(min(2, spare)+1))
    after = int(rng.integers(min(2, spare-before)+1))
    starts, current = [], before
    for i, size in enumerate(sizes):
        starts.append(current)
        current += size + (gaps[i] if i < count-1 else 0)
    return starts, sizes, current + after


def _boxes(rng, sites, nr, nc):
    rows, rh, h = _strips(rng, nr)
    cols, cw, w = _strips(rng, nc)
    boxes = []
    for r, c in sites:
        ph = int(rng.choice([5, 7])) if rh[r] == 7 else 5
        pw = int(rng.choice([5, 7])) if cw[c] == 7 else 5
        top = rows[r] + int(rng.integers(rh[r]-ph+1))
        left = cols[c] + int(rng.integers(cw[c]-pw+1))
        boxes.append((top, left, ph, pw))
    return boxes, (h, w)


def _cable(rng, first, second, horizontal, wa, wb):
    if (first[1] if horizontal else first[0]) > (second[1] if horizontal else second[0]):
        return _cable(rng, second, first, horizontal, wb, wa)
    ar, ac, ah, aw = first
    br, bc, bh, bw = second
    if horizontal:
        sa = ar + int(rng.integers(1, ah-wa))
        sb = br + int(rng.integers(1, bh-wb))
        aa, bb = ac+aw, bc-1
        ports = {(r, aa) for r in range(sa, sa+wa)} | {(r, bb) for r in range(sb, sb+wb)}
        ra, rb = sa+int(rng.integers(wa)), sb+int(rng.integers(wb))
        middle = int(rng.integers(aa+1, bb))
        return ports | _axis((ra, aa), (ra, middle)) | _axis((ra, middle), (rb, middle)) | _axis((rb, middle), (rb, bb))
    sa = ac + int(rng.integers(1, aw-wa))
    sb = bc + int(rng.integers(1, bw-wb))
    aa, bb = ar+ah, br-1
    ports = {(aa, c) for c in range(sa, sa+wa)} | {(bb, c) for c in range(sb, sb+wb)}
    ca, cb = sa+int(rng.integers(wa)), sb+int(rng.integers(wb))
    middle = int(rng.integers(aa+1, bb))
    return ports | _axis((aa, ca), (middle, ca)) | _axis((middle, ca), (middle, cb)) | _axis((middle, cb), (bb, cb))


def _relations(rng, n, edges, mode, test=False):
    weights = [[int(rng.integers(1, 4)), int(rng.integers(1, 4))] for _ in edges]
    forced = {}
    if mode == 'branch':
        win, lo1, lo2, foreign = map(int, rng.permutation(np.arange(1, 5)))
        bodies, values = [0]*n, [0]*n
        bodies[0], values[0] = 0, 1
        for j in (win, lo1, lo2):
            bodies[j] = 1
        values[win] = 2
        bodies[foreign] = 3
        for k, (_, j) in enumerate(edges):
            weights[k][0] = 3 if j in (win, foreign) else 1
        forced = {0: 2, win: 0}
    elif mode == 'path':
        bodies = [0, 1, 2] + [1]*(n-3)
        values = [1, 2] + [3]*(n-2)
        forced = {0: 3}
    else:
        k = int(rng.integers(4, 7))
        incident = [[] for _ in range(n)]
        for index, (i, j) in enumerate(edges):
            incident[i].append((j, index, 0))
            incident[j].append((i, index, 1))
        conditioned = test or rng.random() < .5
        for _ in range(128):
            bodies = list(map(int, rng.integers(k, size=n)))
            values = list(map(int, rng.integers(k, size=n)))
            if conditioned:
                focuses = [i for i, ns in enumerate(incident) if len(ns) >= 2]
                richer = [i for i in focuses if len(incident[i]) >= 3]
                focus = int(rng.choice(richer if richer and rng.random() < .7 else focuses))
                key, z, q = map(int, rng.choice(k, 3, replace=False))
                values[focus] = key
                neighbors = list(incident[focus])
                rng.shuffle(neighbors)
                m = int(rng.choice([2, 3])) if len(neighbors) >= 3 else 2
                if m == 2:
                    receiver_weights = list(map(int, rng.choice([1, 2, 3], 2, replace=False)))
                    destinations = [z, q]
                else:
                    # Minority-donor wins, equal weights, and wide majority wins.
                    profiles = ((3, 1, 1), (1, 1, 1), (1, 2, 3))
                    receiver_weights = profiles[int(rng.integers(len(profiles)))]
                    destinations = [z, q, q]
                for index, (j, edge, end) in enumerate(neighbors):
                    if index < m:
                        bodies[j] = key
                        values[j] = destinations[index]
                        weights[edge][end] = int(receiver_weights[index])
                    else:
                        bodies[j] = int(rng.choice([v for v in range(k) if v != key]))
            info = _record_info(bodies, values, edges, weights)
            winners = _evaluate(info)
            if -1 not in winners and any(v is None for v in winners) and any(v is not None for v in winners):
                break
        else:
            # Any connected graph supports this guaranteed one-survivor case.
            bodies, values = [0]*n, [2]*n
            bodies[0], values[0] = 1, 0
    return bodies, values, weights, forced


def _record_info(bodies, values, edges, weights):
    neighbors = [[] for _ in bodies]
    for (i, j), (wi, wj) in zip(edges, weights):
        neighbors[i].append((j, wi))
        neighbors[j].append((i, wj))
    return dict(nodes=[dict(body=b, value=v) for b, v in zip(bodies, values)], neighbors=neighbors)


def _pattern(rng, h, w, majority, minority, force_center):
    area = h*w
    band = int(rng.integers(3))
    nminor = ((area-1)//2 if band == 0 else
              (1 if band == 1 else int(rng.integers(1, (area-1)//2+1))))
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        order = list(map(int, rng.permutation(area)))
    elif style == 1:
        distance = abs(rr-int(rng.integers(h)))+abs(cc-int(rng.integers(w)))
        order = list(map(int, np.argsort((distance+rng.random((h, w))*.4).ravel())))
    elif style == 2:
        order = list(map(int, np.argsort(((rr+cc)%2*area+rr*w+cc).ravel())))
    else:
        order = list(map(int, (rr*w+cc if rng.random() < .5 else cc*h+rr).ravel().argsort()))
    if force_center:
        center = (h//2)*w+w//2
        order.remove(center)
        order.insert(0, center)
    flat = np.full(area, majority, dtype=np.int8)
    flat[order[:nminor]] = minority
    return flat.reshape(h, w)


def _make_pair(rng, mode, test, palette):
    for _ in range(32):
        sites, edge_pairs, nr, nc = _sites(rng, mode, test)
        boxes, shape = _boxes(rng, sites, nr, nc)
        bodies, values, weights, forced = _relations(rng, len(sites), edge_pairs, mode, test)
        bg, wire = palette[:2]
        identities = palette[2:]
        a = np.full(shape, bg, dtype=np.int8)
        cables = []
        for (i, j), (wi, wj) in zip(edge_pairs, weights):
            cable = _cable(rng, boxes[i], boxes[j], sites[i][0] == sites[j][0], wi, wj)
            cables.append(cable)
            for p in cable:
                a[p] = wire
        for i, (r, c, h, w) in enumerate(boxes):
            a[r:r+h, c:c+w] = identities[bodies[i]]
            minor = forced.get(i)
            if minor is None:
                minor = int(rng.choice([j for j in range(6) if j != values[i]]))
            a[r+1:r+h-1, c+1:c+w-1] = _pattern(
                rng, h-2, w-2, identities[values[i]], identities[minor], i in forced)
        record = _record_info(bodies, values, edge_pairs, weights)
        winners = _evaluate(record)
        assert -1 not in winners
        # Labels use pre-render relations and requested port weights, not solve.
        out = a.copy()
        for (r, c, h, w), winner in zip(boxes, winners):
            if winner is None:
                out[r:r+h, c:c+w] = bg
            else:
                out[r+1:r+h-1, c+1:c+w-1] = identities[winner]
        turns = int(rng.integers(4))
        a, out = np.rot90(a, turns), np.rot90(out, turns)
        if rng.random() < .5:
            a, out = np.fliplr(a), np.fliplr(out)
        a, out = a.copy(), out.copy()
        try:
            info = _inspect(a)
            if not np.array_equal(out, _paint(a, info)):
                continue
            f = _features(a, info)
            if mode == 'branch' and not all(f[k] for k in
                    ('weighted_witness', 'gating_witness', 'center_witness', 'simultaneous_witness')):
                continue
            if mode == 'path' and not (f['scope_witness'] and f['simultaneous_witness']):
                continue
            if test and not f['contested_receivers']:
                continue
            return dict(input=a, output=out)
        except ValueError:
            continue
    raise RuntimeError("bounded panel/cable construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    varying = bool(rng.random() < .65)
    modes = ['branch', 'path'] + ['random']*(nt-2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            mode = modes[i] if split == 'train' else 'random'
            for _ in range(8):
                roles = list(map(int, rng.permutation(10))) if varying else palette
                pair = _make_pair(rng, mode, split == 'test', roles)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded duplicate-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _inspect(a)
            expected = _paint(a, info)
            assert np.array_equal(out, expected), "wrong simultaneous consensus rewrite"
            assert np.array_equal(a[a == info['wire']], out[a == info['wire']])
            assert np.array_equal(a[a == info['background']], out[a == info['background']])
            assert any(v is None for v in info['winners'])
            assert any(v is not None for v in info['winners'])
            for node, winner in zip(info['nodes'], info['winners']):
                r, c, h, w = node['box']
                block = out[r:r+h, c:c+w]
                if winner is None:
                    assert np.all(block == info['background'])
                else:
                    assert np.all(block[1:-1, 1:-1] == winner)
                    assert np.all(block[0] == node['body']) and np.all(block[-1] == node['body'])
                    assert np.all(block[:, 0] == node['body']) and np.all(block[:, -1] == node['body'])
            features[split].append(_features(a, info))
    train = features['train']
    for name in ('weighted_witness', 'gating_witness', 'center_witness', 'scope_witness',
                 'simultaneous_witness', 'erased_donors', 'changed_surviving_donors', 'solid_results'):
        assert any(f[name] for f in train), f"missing demonstration for {name}"
    return features


def self_test():
    """Stream 42 seeds; retain aggregate signatures only, with no disk output."""
    signatures, shapes, layouts, palettes = set(), set(), set(), set()
    counts, port_counts, areas, margins, degrees, survivors = set(), set(), set(), set(), set(), set()
    witnesses = Counter()
    cycles = stable = varied = pairs = contested_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fs = validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        role_set = set()
        for split in ('train', 'test'):
            for pair, f in zip(puzzle[split], fs[split]):
                a = pair['input']
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
                digest.update(pair['output'].tobytes())
                pairs += 1
                shapes.add(f['shape'])
                layouts.add((f['shape'], f['layout'], f['degrees']))
                palettes.add(f['palette'])
                role_set.add(f['palette'])
                counts.add(f['nodes'])
                port_counts.update(f['ports'])
                areas.update(f['core_areas'])
                margins.update(f['majority_margins'])
                degrees.update(f['degrees'])
                survivors.add(f['survivors'])
                cycles += f['cycle']
                contested_tests += split == 'test' and f['contested_receivers'] > 0
                for name in ('weighted_witness', 'gating_witness', 'center_witness', 'scope_witness',
                             'simultaneous_witness', 'erased_donors', 'changed_surviving_donors', 'solid_results'):
                    witnesses[name] += bool(f[name])
        signatures.add(digest.digest())
        stable += len(role_set) == 1
        varied += len(role_set) > 1
    assert len(signatures) == 42 and len(shapes) >= 30 and len(layouts) >= 150
    assert {4, 5, 6, 7, 8} <= counts and port_counts == {1, 2, 3}
    assert {9, 15, 25} <= areas and len(margins) >= 7
    assert {1, 2, 3, 4} <= degrees and len(survivors) >= 4
    assert len(palettes) >= 30 and stable >= 3 and varied >= 10 and cycles >= 5
    assert all(witnesses[k] >= 42 for k in witnesses)
    assert witnesses['weighted_witness'] > 42 and contested_tests == 42
    first, second = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s]) for k in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Region-majority invariance, equivariance, and semantic counterfactuals.
    branch = next(p for p in first['train'] if _features(p['input'])['weighted_witness'])
    a = branch['input']
    info = _inspect(a)
    scrambled = a.copy()
    for node in info['nodes']:
        r, c, h, w = node['box']
        core = scrambled[r+1:r+h-1, c+1:c+w-1]
        core[:] = np.sort(core, axis=None).reshape(core.shape)
    assert np.array_equal(solve(scrambled), branch['output'])
    for k in range(4):
        assert np.array_equal(solve(np.rot90(a, k)), np.rot90(branch['output'], k))
    assert np.array_equal(solve(np.fliplr(a)), np.fliplr(branch['output']))
    permutation = np.array([6, 8, 2, 9, 1, 5, 0, 7, 4, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[a]), permutation[branch['output']])
    _counterfactual_checks(branch)
    _larger_panel_check()
    bad = {s: [dict(input=q['input'].copy(), output=q['output'].copy()) for q in first[s]] for s in first}
    target = _inspect(bad['train'][0]['input'])
    r, c, h, w = target['nodes'][0]['box']
    bad['train'][0]['output'][r, c] = (int(bad['train'][0]['output'][r, c])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated panel output accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), shapes=len(shapes),
                layouts=len(layouts), palette_roles=len(palettes), panel_counts=sorted(counts),
                port_counts=sorted(port_counts), core_areas=sorted(areas),
                local_margins=sorted(margins), degrees=sorted(degrees), survivor_counts=sorted(survivors),
                cycle_pairs=cycles, stable_palettes=stable, varied_palettes=varied,
                contested_tests=contested_tests,
                witnesses=dict(witnesses))


def _counterfactual_checks(branch):
    a = branch['input']
    info = _inspect(a)
    # An A-ablation: flip the decisive donor's interior majority, while retaining
    # exactly two core colors and all geometry. Root must switch to the old loser.
    root = next(i for i, ns in enumerate(info['neighbors']) if len(ns) == 4)
    donors = [(j, w) for j, w in info['neighbors'][root]
              if info['nodes'][j]['body'] == info['nodes'][root]['value']]
    decisive = next(j for j, w in donors if w == 3)
    loser = next(info['nodes'][j]['value'] for j, w in donors if w == 1)
    node = info['nodes'][decisive]
    r, c, h, w = node['box']
    changed = a.copy()
    changed[r+1:r+h-1, c+1:c+w-1] = loser
    changed[r+1, c+1] = node['value']
    new = _inspect(changed)
    assert new['winners'][root] == loser != info['winners'][root]
    # Same relations, swapped receiver port weights: consensus must reverse.
    port_swapped = dict(info, neighbors=[list(x) for x in info['neighbors']])
    port_swapped['neighbors'][root] = [(j, 1 if j == decisive else 3)
                                      for j, _ in info['neighbors'][root]]
    assert _evaluate(port_swapped)[root] == loser
    # Repeat the weight intervention on actual input pixels, without metadata.
    rb, cb, rh, cw = info['nodes'][root]['box']
    def ports_on_root(edge):
        return [p for p in edge['cells'] if any(
            rb <= p[0]+dr < rb+rh and cb <= p[1]+dc < cb+cw for dr, dc in _D4)]
    decisive_edge = next(e for e in info['edges'] if {e['a'], e['b']} == {root, decisive})
    ports = ports_on_root(decisive_edge)
    def outward(p):
        if p[0] == rb-1:
            return p[0]-1, p[1]
        if p[0] == rb+rh:
            return p[0]+1, p[1]
        if p[1] == cb-1:
            return p[0], p[1]-1
        return p[0], p[1]+1
    anchor = next(p for p in ports if outward(p) in decisive_edge['cells'])
    edited = a.copy()
    for p in ports:
        if p != anchor:
            edited[p] = info['background']
    losing = next(j for j, w in donors if w == 1)
    losing_edge = next(e for e in info['edges'] if {e['a'], e['b']} == {root, losing})
    p = ports_on_root(losing_edge)[0]
    if p[0] in (rb-1, rb+rh):
        start = max(cb+1, min(p[1]-1, cb+cw-4))
        new_ports = [(p[0], c) for c in range(start, start+3)]
    else:
        start = max(rb+1, min(p[0]-1, rb+rh-4))
        new_ports = [(r, p[1]) for r in range(start, start+3)]
    for p in new_ports:
        edited[p] = info['wire']
    edited_info = _inspect(edited)
    assert edited_info['winners'][root] == loser
    r, c, h, w = edited_info['nodes'][root]['box']
    assert np.all(solve(edited)[r+1:r+h-1, c+1:c+w-1] == loser)
    # Stretch the canvas only in empty panel corridors. A column or row insertion
    # repeats crossing wire pixels, increasing cable length without adding ports.
    stretched = False
    for axis in (0, 1):
        if a.shape[axis] >= 30:
            continue
        for pos in range(1, a.shape[axis]-1):
            line = a[pos, :] if axis == 0 else a[:, pos]
            prev = a[pos-1, :] if axis == 0 else a[:, pos-1]
            if (np.any(line == info['wire']) and np.array_equal(line, prev)
                    and np.all((line == info['background']) | (line == info['wire']))):
                expanded = np.insert(a, pos, line, axis=axis)
                expected_line = branch['output'][pos, :] if axis == 0 else branch['output'][:, pos]
                expected = np.insert(branch['output'], pos, expected_line, axis=axis)
                assert np.array_equal(solve(expanded), expected)
                stretched = True
                break
        if stretched:
            break
    assert stretched, "expected at least one stretchable corridor in seed zero"


def _larger_panel_check():
    """A valid off-distribution 9x7 panel and a two-node graph need no new rule."""
    rng = np.random.default_rng(814)
    a = np.zeros((16, 29), dtype=np.int8)
    boxes = [(2, 1, 9, 7), (6, 17, 5, 9)]
    for (r, c, h, w), body, value, minority in zip(boxes, [1, 2], [2, 3], [3, 1]):
        a[r:r+h, c:c+w] = body
        a[r+1:r+h-1, c+1:c+w-1] = value
        a[r+1, c+1] = minority
    for p in _cable(rng, boxes[0], boxes[1], True, 2, 3):
        a[p] = 5
    expected = a.copy()
    expected[3:10, 2:7] = 3
    expected[6:11, 17:26] = 0
    assert np.array_equal(solve(a), expected)
