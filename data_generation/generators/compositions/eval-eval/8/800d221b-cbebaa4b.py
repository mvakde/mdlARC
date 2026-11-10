"""Signal-gated rigid docking, followed by voting by completed joint pixels.

The collector bead visibly encodes the structural and driver colors. No palette,
anchor convention, ordering, or placement is learned from stored demonstrations.
Construction records draw labels; decoding rediscovers panels, socket banks,
geometric keys, a rooted attachment tree, translations, and votes from input.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(cells):
    left = set(cells)
    answer = []
    while left:
        start = min(left)
        left.remove(start)
        found, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    found.add(q)
                    todo.append(q)
        answer.append(found)
    return answer


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, d):
    return {(r + d[0], c + d[1]) for r, c in cells}


def _normal(cells):
    r, c, _, _ = _box(cells)
    return tuple(sorted((a - r, b - c) for a, b in cells))


def _rectangle(r, c, h, w):
    return {(a, b) for a in range(r, r + h) for b in range(c, c + w)}


def _ring(r, c, h, w):
    return _rectangle(r, c, h, w) - _rectangle(r + 1, c + 1, h - 2, w - 2)


def _checked_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError('expected an integer ARC grid of size 1..30')
    return g.astype(np.int8, copy=True)


def _beads(g):
    """A 3x3 structural ring around a distinct, nonblack driver bead."""
    result = []
    for r in range(g.shape[0] - 2):
        for c in range(g.shape[1] - 2):
            h, m = int(g[r, c]), int(g[r + 1, c + 1])
            if h and m and h != m and all(g[q] == h for q in _ring(r, c, 3, 3)):
                result.append((r, c, h, m))
    return result


def _upright(g, bead):
    """Read one scene with incoming keys above bodies and sockets below them.

    Sparse pins are grouped by their separate continuous structural support
    bars, never by pin connectivity or by area. A whole-scene quarter turn is
    tried by _decode; reflection is already compatible with this grammar.
    """
    br, bc, hull, marker = bead
    colors = set(map(int, np.unique(g)))
    signals = tuple(sorted(colors - {0, hull, marker}))
    assert len(signals) == 2 and len(colors) == 5
    pieces = _components(map(tuple, np.argwhere(g != 0).tolist()))
    assert 3 <= len(pieces) <= 5
    nodes, root = [], None
    for cells in pieces:
        is_root = (br + 1, bc + 1) in cells
        pins = {q for q in cells if g[q] == marker}
        colored = {q for q in cells if g[q] in signals}
        candidates = []
        for comp in _components(colored):
            r, c, b, e = _box(comp)
            if (b - r + 1 >= 2 and e - c + 1 >= 2
                    and comp == _rectangle(r, c, b - r + 1, e - c + 1)
                    and {int(g[q]) for q in comp} == set(signals)
                    and r > 0 and c > 0 and b + 1 < g.shape[0] and e + 1 < g.shape[1]
                    and all(g[q] == hull for q in _ring(r - 1, c - 1, b - r + 3, e - c + 3))):
                candidates.append(comp)
        if is_root:
            assert root is None and not candidates and pins == {(br + 1, bc + 1)}
            root = len(nodes)
            panel, majority, frame_bottom = set(), None, br + 2
            pins = set()
        else:
            assert len(candidates) == 1
            panel = candidates[0]
            pr, pc, pb, pe = _box(panel)
            counts = Counter(int(g[q]) for q in panel)
            assert counts[signals[0]] != counts[signals[1]]
            majority = max(signals, key=lambda s: counts[s])
            assert 2 <= len(pins) <= 5 and max(r for r, c in pins) < pr - 1
            key = _normal(pins)
            assert max(r for r, c in key) <= 1
            assert len({c for r, c in key}) == len(key)
            assert all(c % 2 == 0 for r, c in key)
            frame_bottom = pb + 1
        receivers = colored - panel
        banks = []
        if receivers:
            base = min(r for r, c in receivers)
            assert base >= frame_bottom + 3
            assert all(r in (base, base + 1) for r, c in receivers)
            columns = sorted(c for r, c in cells if r == base - 1 and g[r, c] == hull)
            runs = []
            for c in columns:
                if not runs or c != runs[-1][-1] + 1:
                    runs.append([c])
                else:
                    runs[-1].append(c)
            accounted = set()
            for run in runs:
                points = {q for q in receivers if run[0] <= q[1] <= run[-1]}
                if not points:
                    continue
                by_color = {s: {q for q in points if g[q] == s} for s in signals}
                a, b = (by_color[s] for s in signals)
                assert 2 <= len(a) <= 5 and len(a) == len(b)
                assert _normal(a) == _normal(b)
                ar, ac, _, _ = _box(a)
                zr, zc, _, _ = _box(b)
                assert ar == zr and abs(ac - zc) in (1, 3)
                assert min(c for r, c in points) == run[0]
                assert max(c for r, c in points) == run[-1]
                banks.append({'key': _normal(a), 'sockets': by_color})
                accounted |= points
            assert accounted == receivers and banks
        nodes.append({'cells': cells, 'panel': panel, 'pins': pins,
                      'majority': majority, 'banks': banks, 'root': is_root})
    assert root is not None and nodes[root]['banks']
    keys = [_normal(p['pins']) for p in nodes if not p['root']]
    assert len(set(keys)) == len(keys), 'complete driver geometry must be distinct'
    available = {}
    for i, p in enumerate(nodes):
        for bank in p['banks']:
            assert bank['key'] not in available, 'each geometric key has one receiving bank'
            available[bank['key']] = (i, bank)
    assert set(available) == set(keys)
    parents, sockets = {}, {}
    for i, p in enumerate(nodes):
        if not p['root']:
            parent, bank = available[_normal(p['pins'])]
            assert parent != i
            parents[i] = parent
            sockets[i] = bank['sockets']
    info = {'grid': g, 'nodes': nodes, 'root': root, 'parents': parents,
            'sockets': sockets, 'signals': signals, 'hull': hull,
            'marker': marker, 'center': (br + 1, bc + 1)}
    # Also prove rooted connectivity, collision constraints, and vote strictness.
    _assemble(info)
    return info


def _assemble(info, override=None):
    """Derive absolute rigid translations and redraw; no construction metadata."""
    nodes, root = info['nodes'], info['root']
    labels = {i: p['majority'] for i, p in enumerate(nodes) if i != root}
    if override:
        labels.update(override)
    positions = {root: (0, 0)}
    remaining = set(labels)
    while remaining:
        moved = False
        for i in sorted(remaining):
            parent = info['parents'][i]
            if parent not in positions:
                continue
            target = info['sockets'][i][labels[i]]
            sr, sc, _, _ = _box(target)
            pr, pc, _, _ = _box(nodes[i]['pins'])
            pd = positions[parent]
            positions[i] = (pd[0] + sr - pr, pd[1] + sc - pc)
            assert _shift(nodes[i]['pins'], positions[i]) == _shift(target, pd)
            remaining.remove(i)
            moved = True
        assert moved, 'attachment graph must be a rooted tree'
    g = info['grid']
    occupied, owners = {}, {}
    # First draw structural material, panels, and all receiving sockets.
    for i, node in enumerate(nodes):
        for old in node['cells'] - node['pins']:
            q = old[0] + positions[i][0], old[1] + positions[i][1]
            assert 0 <= q[0] < g.shape[0] and 0 <= q[1] < g.shape[1]
            assert q not in occupied, 'non-driver cells cannot overlap'
            occupied[q] = int(g[old])
            owners[q] = [i]
    joints = {}
    for i in labels:
        cells = _shift(nodes[i]['pins'], positions[i])
        for q in cells:
            assert q in occupied and owners[q] == [info['parents'][i]]
            assert occupied[q] == labels[i]
            owners[q].append(i)
        joints[i] = cells
    votes = Counter()
    for i, cells in joints.items():
        votes[labels[i]] += len(cells)
    a, b = info['signals']
    assert votes[a] > 0 and votes[b] > 0 and votes[a] != votes[b]
    winner = a if votes[a] > votes[b] else b
    occupied[info['center']] = winner
    out = np.zeros(g.shape, dtype=np.int8)
    for q, color in occupied.items():
        out[q] = color
    return out, {'positions': positions, 'joints': joints, 'owners': owners,
                 'labels': labels, 'votes': dict(votes), 'winner': winner}


def _decode(value):
    g = _checked_grid(value)
    interpretations = []
    for rotation in range(4):
        upright = np.rot90(g, rotation).copy()
        for bead in _beads(upright):
            try:
                info = _upright(upright, bead)
                info['rotation'] = rotation
                interpretations.append(info)
            except (AssertionError, ValueError, KeyError):
                continue
    if len(interpretations) != 1:
        raise ValueError('input must have one complete gated-docking interpretation')
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    out, _ = _assemble(info)
    return np.rot90(out, -info['rotation']).copy()


def _key(rng, weight, offsets):
    # Same areas deliberately have different spacings or row offsets.
    if weight == 5:
        xs = [0, 2, 4, 6, 8]
    else:
        xs = [0] + sorted(map(int, rng.choice([2, 4, 6, 8], weight - 1, replace=False)))
    rows = list(map(int, rng.integers(2, size=weight))) if offsets else [0] * weight
    rows[0] = 0
    return tuple((r, c) for r, c in zip(rows, xs))


def _ballot(rng, ph, pw, majority, other, strength=None):
    area = ph * pw
    if strength == 'strong':
        minority = 1
    else:
        minority = int(rng.integers(1, (area - 1) // 2 + 1))
    pattern = np.full((ph, pw), majority, dtype=np.int8)
    kind = int(rng.integers(3))
    if kind == 0:
        indices = rng.permutation(area)
    elif kind == 1:
        rr, cc = np.indices((ph, pw))
        cr, cs = int(rng.integers(ph)), int(rng.integers(pw))
        indices = np.argsort((abs(rr - cr) + abs(cc - cs) + rng.random((ph, pw)) * .2).ravel())
    else:
        indices = np.arange(area) if rng.random() < .5 else np.arange(area)[::-1]
    pattern.flat[indices[:minority]] = other
    return pattern


def _make_piece(rng, incoming, children, keys, roles, label, dims, witness_loser=False):
    hull, marker, a, b = roles
    mapping, pins, panel = {}, set(), set()
    if incoming is None:
        frame_r, frame_h, frame_w, center_c = 0, 3, 3, int(rng.integers(-2, 3))
        frame_c = center_c - 1
    else:
        pins = set(incoming)
        bar = max(r for r, c in pins) + 1
        for r, c in pins:
            for rr in range(r + 1, bar + 1):
                mapping[rr, c] = hull
        for c in range(max(c for r, c in pins) + 1):
            mapping[bar, c] = hull
        ph, pw = dims
        frame_r, frame_h, frame_w = bar + 1, ph + 2, pw + 2
        center_c = max(c for r, c in pins) // 2 + int(rng.integers(-1, 2))
        frame_c = center_c - frame_w // 2
        other = b if label == a else a
        pattern = _ballot(rng, ph, pw, label, other, 'strong' if witness_loser else None)
        for r in range(ph):
            for c in range(pw):
                q = frame_r + 1 + r, frame_c + 1 + c
                mapping[q] = int(pattern[r, c])
                panel.add(q)
        for q in pins:
            mapping[q] = marker
    frame = _ring(frame_r, frame_c, frame_h, frame_w)
    mapping.update({q: hull for q in frame})
    center = (frame_r + 1, frame_c + 1) if incoming is None else None
    if center is not None:
        mapping[center] = marker
    banks = {}
    if children:
        shifts = [int(rng.choice([1, 1, 3])) for _ in children]
        widths = [max(c for r, c in keys[i]) + 1 + s for i, s in zip(children, shifts)]
        gaps = [int(rng.integers(2, 5)) for _ in children[:-1]]
        total = sum(widths) + sum(gaps)
        start = center_c - total // 2 + int(rng.integers(-1, 2))
        bottom = frame_r + frame_h - 1
        base, trunk = bottom + 4, bottom + 1
        end = start + total - 1
        for c in range(min(start, center_c), max(end, center_c) + 1):
            mapping[trunk, c] = hull
        mapping[bottom, center_c] = hull
        for j, (child, shift, width) in enumerate(zip(children, shifts, widths)):
            left = start
            for c in range(left, left + width):
                mapping[base - 1, c] = hull
            stem = left + width // 2
            for r in range(trunk, base):
                mapping[r, stem] = hull
            first, second = (a, b) if rng.random() < .5 else (b, a)
            sockets = {}
            for color, phase in ((first, 0), (second, shift)):
                cells = {(base + r, left + c + phase) for r, c in keys[child]}
                for r, c in cells:
                    for rr in range(base - 1, r):
                        mapping[rr, c] = hull
                    mapping[r, c] = color
                sockets[color] = cells
            banks[child] = sockets
            start += width + (gaps[j] if j < len(gaps) else 0)
    # Normalize the whole fragment, including frames that overhang a short key.
    top, left, _, _ = _box(mapping)
    d = -top, -left
    normalized = {(r + d[0], c + d[1]): v for (r, c), v in mapping.items()}
    return {'map': normalized, 'pins': _shift(pins, d), 'panel': _shift(panel, d),
            'banks': {i: {s: _shift(v, d) for s, v in sock.items()} for i, sock in banks.items()},
            'center': None if center is None else (center[0] + d[0], center[1] + d[1])}


def _scatter(rng, shape, pieces, root_origin, final_positions):
    """Random bounded packing of complete fragments, with an eight-neighbor gap."""
    h, w = shape
    for restart in range(32):
        positions = {0: root_origin}
        occupied = _shift(pieces[0]['map'], root_origin)
        blocked = {(r + dr, c + dc) for r, c in occupied
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        order = list(range(1, len(pieces)))
        rng.shuffle(order)
        order.sort(key=lambda i: -len(pieces[i]['map']))
        for i in order:
            _, _, bottom, right = _box(pieces[i]['map'])
            candidates = [(r, c) for r in range(1, h - bottom - 1)
                          for c in range(1, w - right - 1)]
            rng.shuffle(candidates)
            for d in candidates:
                if d == final_positions[i]:
                    continue
                moved = _shift(pieces[i]['map'], d)
                if not moved & blocked:
                    positions[i] = d
                    blocked |= {(r + dr, c + dc) for r, c in moved
                                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                    break
            else:
                break
        if len(positions) == len(pieces):
            inp = np.zeros(shape, dtype=np.int8)
            for i, d in positions.items():
                for (r, c), color in pieces[i]['map'].items():
                    inp[r + d[0], c + d[1]] = color
            return inp
    return None


def _counterfactual(info, internal_only=False):
    """A changed local vote must admit a different legal mechanical placement."""
    _, original = _assemble(info)
    for i, label in original['labels'].items():
        if internal_only and i not in info['parents'].values():
            continue
        other = next(s for s in info['signals'] if s != label)
        try:
            _, altered = _assemble(info, {i: other})
        except AssertionError:
            continue
        if altered['positions'][i] != original['positions'][i]:
            return i
    return None


def _candidate(rng, roles, winner, witness, branch, count, tall=False, offset_keys=False):
    if witness:
        weights, labels = [5, 2, 2], [winner, roles[3] if winner == roles[2] else roles[2],
                                    roles[3] if winner == roles[2] else roles[2]]
        parents = [0, 0, 0]
    else:
        weights = list(map(int, rng.integers(2, 5, size=count)))
        assignments = [bits for bits in range(1, (1 << count) - 1)
                       if sum(weights[i] for i in range(count) if bits >> i & 1) > sum(weights) / 2]
        if not assignments:
            return None
        bits = int(rng.choice(assignments))
        loser = roles[3] if winner == roles[2] else roles[2]
        labels = [winner if bits >> i & 1 else loser for i in range(count)]
        if branch and count == 3:
            parents = [0, 1, 1]
        elif count == 2:
            parents = [0, 1] if rng.random() < .65 else [0, 0]
        elif count == 3:
            parents = [0, 0, int(rng.choice([0, 1, 2]))]
        else:
            parents = list(rng.choice([[0, 0, 1, 2], [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]]))
    keys = {}
    for i, weight in enumerate(weights, 1):
        for _ in range(40):
            key = _key(rng, weight, offset_keys or (not witness and rng.random() < .55))
            if key not in keys.values():
                keys[i] = key
                break
        else:
            return None
    children = {i: [] for i in range(count + 1)}
    for i, parent in enumerate(parents, 1):
        children[int(parent)].append(i)
    dimensions = []
    for i in range(count):
        if witness:
            dimensions.append((2, 2) if i == 0 else (2, int(rng.choice([3, 4]))))
        else:
            dimensions.append((int(rng.integers(2, 5 if tall else 4)), int(rng.integers(2, 5))))
    if tall:
        j = int(rng.integers(count))
        dimensions[j] = (4, dimensions[j][1])
    pieces = [_make_piece(rng, None, children[0], keys, roles, None, None)]
    for i in range(1, count + 1):
        pieces.append(_make_piece(rng, keys[i], children[i], keys, roles, labels[i - 1],
                                  dimensions[i - 1], witness and i > 1))
    positions = {0: (0, 0)}
    occupied = dict(pieces[0]['map'])
    for i in range(1, count + 1):
        parent, label = int(parents[i - 1]), labels[i - 1]
        target = pieces[parent]['banks'][i][label]
        sr, sc, _, _ = _box(target)
        pr, pc, _, _ = _box(pieces[i]['pins'])
        pd = positions[parent]
        d = pd[0] + sr - pr, pd[1] + sc - pc
        positions[i] = d
        joint = _shift(pieces[i]['pins'], d)
        for old, color in pieces[i]['map'].items():
            q = old[0] + d[0], old[1] + d[1]
            if q in occupied and (q not in joint or occupied[q] != label):
                return None
            occupied[q] = label if old in pieces[i]['pins'] else color
    top, left, bottom, right = _box(occupied)
    bh, bw = bottom - top + 1, right - left + 1
    if bh > 28 or bw > 28:
        return None
    # Geometry and required scatter space, rather than the split, set the canvas.
    h = int(rng.integers(max(24, bh + 2), 31))
    w = int(rng.integers(max(24, bw + 2), 31))
    origin = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
    translation = origin[0] - top, origin[1] - left
    final = {i: (d[0] + translation[0], d[1] + translation[1]) for i, d in positions.items()}
    inp = _scatter(rng, (h, w), pieces, final[0], final)
    if inp is None:
        return None
    out = np.zeros((h, w), dtype=np.int8)
    for (r, c), color in occupied.items():
        out[r + translation[0], c + translation[1]] = color
    cr, cc = pieces[0]['center']
    out[cr + translation[0], cc + translation[1]] = winner
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    try:
        info = _decode(inp)
        pred, record = _assemble(info)
        if not np.array_equal(np.rot90(pred, -info['rotation']), out):
            return None
        if (witness or branch) and _counterfactual(info, internal_only=branch) is None:
            return None
        if any(d == (0, 0) for i, d in record['positions'].items() if i != info['root']):
            return None
    except (AssertionError, ValueError):
        return None
    return {'input': inp, 'output': out}


def _int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _int(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _int(num_train) or not _int(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require 2..10 training, 1..4 tests, and num_train > num_test')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(np.arange(1, 10), 4, replace=False)))
    mode = int(rng.integers(3))  # count, panel-height, or palette transfer
    winner_order = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            roles = palette
            if mode == 2 and split == 'test':
                structural = tuple(map(int, rng.choice(np.arange(1, 10), 2, replace=False)))
                pool = [c for c in range(1, 10) if c not in structural and c not in palette[2:]]
                new_signals = tuple(map(int, rng.choice(pool, 2, replace=False)))
                roles = structural + new_signals
            witness = split == 'train' and index == 0
            branch = split == 'train' and index == 1
            n = 3 if witness or branch else int(rng.choice([2, 3] if split == 'train' else [3, 4]))
            if split == 'test' and mode == 0:
                n = 4
            if split == 'train' and index < 2:
                win = roles[2 + (winner_order ^ index)]
            else:
                win = roles[int(rng.integers(2, 4))]
            for attempt in range(600):
                pair = _candidate(rng, roles, win, witness, branch, n,
                                  tall=(split == 'test' and mode == 1),
                                  offset_keys=branch or (split == 'test'))
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('bounded gated-docking construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _decode(pair['input'])
    expected, rec = _assemble(info)
    assert np.array_equal(np.rot90(expected, -info['rotation']), np.asarray(pair['output']))
    nodes, root = info['nodes'], info['root']
    votes, winner = rec['votes'], rec['winner']
    degrees = Counter(info['parents'].values())
    weights = [len(p['pins']) for p in nodes if not p['root']]
    labels = [p['majority'] for p in nodes if not p['root']]
    keys = [_normal(p['pins']) for p in nodes if not p['root']]
    raw = Counter(int(info['grid'][q]) for p in nodes for q in p['panel'])
    depths = {root: 0}
    todo = set(rec['labels'])
    while todo:
        for i in sorted(todo):
            if info['parents'][i] in depths:
                depths[i] = depths[info['parents'][i]] + 1
                todo.remove(i)
                break
    # Whole-piece preservation with the only allowed changes: drivers and bead.
    for i, node in enumerate(nodes):
        d = rec['positions'][i]
        for old in node['cells'] - node['pins'] - {info['center']}:
            q = old[0] + d[0], old[1] + d[1]
            assert expected[q] == info['grid'][old]
    assert np.count_nonzero(expected == info['marker']) == 0
    assert sum(len(v) == 2 for v in rec['owners'].values()) == sum(weights)
    assert all(len(v) in (1, 2) for v in rec['owners'].values())
    assert all(d != (0, 0) for i, d in rec['positions'].items() if i != root)
    assert rec['positions'][root] == (0, 0)
    return {'info': info, 'module_count': len(weights), 'weights': weights,
            'keys': keys, 'winner': winner, 'winner_order': info['signals'].index(winner),
            'panel_vote_opposes': labels.count(winner) < len(labels) / 2,
            'raw_vote_opposes': raw[winner] < raw[next(s for s in info['signals'] if s != winner)],
            'equal_area_keys': any(len(a) == len(b) and a != b for a, b in combinations(keys, 2)),
            'depth': max(depths.values()), 'max_children': max(degrees.values()),
            'margin': abs(votes[info['signals'][0]] - votes[info['signals'][1]]),
            'panel_boxes': [tuple(x + 1 for x in (_box(p['panel'])[2] - _box(p['panel'])[0],
                                                 _box(p['panel'])[3] - _box(p['panel'])[1]))
                            for p in nodes if p['panel']],
            'roles': (info['hull'], info['marker'], *info['signals'])}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    tr, te = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = inp.shape, inp.tobytes()
            assert key not in seen
            seen.add(key)
            metrics[split].append(_features(pair))
    training = metrics['train']
    witnesses = [m for m in training if m['panel_vote_opposes'] and m['raw_vote_opposes']
                 and m['equal_area_keys'] and _counterfactual(m['info']) is not None]
    assert witnesses, 'training must jointly expose voting, gating, and key geometry'
    assert {m['winner_order'] for m in training} == {0, 1}
    assert any(m['depth'] >= 2 and _counterfactual(m['info'], internal_only=True) is not None
               for m in training), 'train a movable relay whose vote moves its descendants'
    train_count_max = max(m['module_count'] for m in training)
    train_height_max = max(h for m in training for h, w in m['panel_boxes'])
    train_signals = set(s for m in training for s in m['info']['signals'])
    for m in metrics['test']:
        assert (m['module_count'] > train_count_max
                or max(h for h, w in m['panel_boxes']) > train_height_max
                or not (set(m['info']['signals']) & train_signals)), 'missing observable train/test shift'


def _encode(puzzle):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                           for p in puzzle[s]] for s in ('train', 'test')}, separators=(',', ':'))


def self_test() -> dict:
    """Bounded, input-derived semantic checks and structural coverage on 42 seeds."""
    shapes, roles, keys, boxes, counts, margins, depths, signatures = (set() for _ in range(8))
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_encode(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _features(pair)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                shapes.add(tuple(pair['input'].shape))
                roles.add(m['roles'])
                keys.update(m['keys'])
                boxes.update(m['panel_boxes'])
                counts.add(m['module_count'])
                margins.add(m['margin'])
                depths.add(m['depth'])
                records.append({k: v for k, v in m.items() if k != 'info'})
    assert len(signatures) == 42 and len(shapes) >= 12 and len(roles) >= 25
    assert counts == {2, 3, 4} and depths >= {1, 2}
    assert len(keys) >= 20 and len(boxes) >= 5 and len(margins) >= 4
    assert any(m['max_children'] >= 3 and m['depth'] >= 2 for m in records)
    assert any(m['equal_area_keys'] and m['depth'] >= 2 for m in records)
    assert any(max(r for r, c in key) > 0 for key in keys)
    # Reproducibility, count boundaries, invalid count types, and mutation checks.
    p = generate_puzzle(73, 2, 1)
    assert _encode(p) == _encode(generate_puzzle(73, 2, 1))
    for tr, te in ((10, 4), (5, 4), (10, 1)):
        generate_puzzle(731 + tr, tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    info = _decode(p['train'][0]['input'])
    flip = _counterfactual(info)
    assert flip is not None
    changed = info['grid'].copy()
    a, b = info['signals']
    for q in info['nodes'][flip]['panel']:
        changed[q] = b if changed[q] == a else a
    alt = solve(changed)
    baseline, _ = _assemble(info)
    assert not np.array_equal(alt, baseline)
    relay_info = _decode(p['train'][1]['input'])
    relay = _counterfactual(relay_info, internal_only=True)
    assert relay is not None
    changed_relay = relay_info['grid'].copy()
    a, b = relay_info['signals']
    for q in relay_info['nodes'][relay]['panel']:
        changed_relay[q] = b if changed_relay[q] == a else a
    changed_info = _decode(changed_relay)
    _, original_relay = _assemble(relay_info)
    _, altered_relay = _assemble(changed_info)
    children = [i for i, parent in relay_info['parents'].items() if parent == relay]
    assert len(children) == 2
    for i in [relay] + children:
        assert original_relay['positions'][i] != altered_relay['positions'][i]
    # Wrong bead and missing driver are both rejected, not silently repaired.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    upright = np.rot90(bad['test'][0]['output'], _decode(bad['test'][0]['input'])['rotation']).copy()
    test_info = _decode(bad['test'][0]['input'])
    upright[test_info['center']] = test_info['marker']
    bad['test'][0]['output'] = np.rot90(upright, -test_info['rotation']).copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong collector vote accepted')
    broken = info['grid'].copy()
    node = next(p for p in info['nodes'] if p['pins'])
    broken[min(node['pins'])] = 0
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete geometric key accepted')
    return {'seeds': 42, 'pairs': len(records), 'distinct_puzzles': len(signatures),
            'canvas_shapes': len(shapes), 'role_palettes': len(roles),
            'complete_key_geometries': len(keys), 'panel_shapes': sorted(boxes),
            'module_counts': sorted(counts), 'vote_margins': sorted(margins),
            'tree_depths': sorted(depths), 'independent_semantic_solver': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
