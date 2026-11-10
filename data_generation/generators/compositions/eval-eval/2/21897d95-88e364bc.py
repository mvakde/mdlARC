"""Line-of-sight pigment transfer: input-only addressing and simultaneous fills.

No parent code, data files, seed, or construction metadata is read by solve().
The generator holds one candidate pair at a time and writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numbers
import numpy as np

CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 782675136, 723150486, 340794999, 116125555, 779487884,
              772180582, 722987041, 829572674, 802239994, 641291802,
              167422800, 44802157, 996916231, 584161250, 766046273,
              324623099, 501678110, 170119453, 565783673, 652709310,
              381423381, 663638136, 867649311, 394602314, 848060104,
              241955696, 734135408, 816103640, 729264930, 614368697,
              886101001, 601738475, 550344804, 520188893, 577438206,
              281727549, 115178463, 520774682, 970625294, 506198123, 475043386)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _need(np.all((a >= 0) & (a <= 9)), 'invalid grid colors')
    return a


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _stencil(center, d, shape):
    side = (-d[1], d[0])
    arms = (_add(center, side), _add(center, side, -1))
    tip = _add(center, d)
    flanks = (_add(tip, side), _add(tip, side, -1))
    back, exit_cell = _add(center, d, -1), _add(center, d, 2)
    write = (center, *arms, tip)
    if not all(_inside(p, shape) for p in (*write, *flanks, back, exit_cell)):
        return None
    return dict(center=center, direction=d, arms=arms, tip=tip, flanks=flanks,
                back=back, exit=exit_cell, write=write)


def _components(clean):
    """Original four-connected pigment islands; zero has no component ID."""
    ids = np.full(clean.shape, -1, dtype=np.int16)
    colors, areas = [], []
    for start in np.ndindex(clean.shape):
        if clean[start] == 0 or ids[start] >= 0:
            continue
        label, color = len(colors), int(clean[start])
        stack, area = [start], 0
        ids[start] = label
        while stack:
            p = stack.pop()
            area += 1
            for d in CARDINAL:
                q = _add(p, d)
                if _inside(q, clean.shape) and ids[q] < 0 and clean[q] == color:
                    ids[q] = label
                    stack.append(q)
        colors.append(color)
        areas.append(area)
    return ids, colors, areas


def _ray_targets(ids, origin, d):
    """Return each encountered component once, in physical order, and first gap."""
    p, gap, targets = origin, 0, []
    while _inside(p, ids.shape):
        label = int(ids[p])
        if label < 0:
            if not targets:
                gap += 1
        elif not targets or label != targets[-1][0]:
            targets.append((label, p))
        p = _add(p, d)
    return targets, gap


def _read(input_grid):
    """Recover all launchers, restore hosts, then address original components."""
    raw = _grid(input_grid)
    arrows = []
    # A center is one step from an ink cell, so this scans only sparse candidates.
    centers = {_add(tuple(map(int, p)), d) for p in np.argwhere(raw == 1) for d in CARDINAL}
    for center in sorted(centers):
        for d in CARDINAL:
            a = _stencil(center, d, raw.shape)
            if a is None or not all(raw[p] == 1 for p in (*a['arms'], a['tip'])):
                continue
            host = int(raw[a['flanks'][0]])
            if host < 2 or any(raw[p] != host for p in (*a['flanks'], a['back'])):
                continue
            if raw[a['exit']] != 0 or raw[center] == 0:
                continue
            a.update(host=host, explicit=bool(raw[center] != 1),
                     payload=host if raw[center] == 1 else int(raw[center]))
            arrows.append(a)
    _need(arrows, 'no recognizable launchers')
    used, ink = set(), set(map(tuple, np.argwhere(raw == 1)))
    covered = set()
    clean = raw.copy()
    for a in arrows:
        _need(used.isdisjoint(a['write']), 'ambiguous or overlapping launchers')
        used.update(a['write'])
        covered.update(p for p in a['write'] if raw[p] == 1)
        for p in a['write']:
            clean[p] = a['host']
    _need(covered == ink and not np.any(clean == 1), 'unexplained ink')
    ids, colors, areas = _components(clean)
    assignments = {}
    for a in arrows:
        source = int(ids[a['center']])
        _need(source >= 0 and all(ids[p] == source for p in a['write']), 'unrecoverable host')
        hits, gap = _ray_targets(ids, a['exit'], a['direction'])
        _need(hits and gap >= 1, 'ray must cross a gap and hit an island')
        stops = [(i, label, p) for i, (label, p) in enumerate(hits)
                 if colors[label] != a['payload']]
        _need(stops, 'ray has no opaque receiver')
        stop_index, target, contact = stops[0]
        _need(target != source, 'a launcher cannot address its own island')
        a.update(source=source, target=target, contact=contact, hits=hits, gap=gap,
                 skipped=tuple(label for label, _ in hits[:stop_index]), stop_index=stop_index)
        _need(target not in assignments or assignments[target] == a['payload'], 'conflicting incoming pigments')
        assignments[target] = a['payload']
    return dict(clean=clean, ids=ids, colors=colors, areas=areas,
                arrows=arrows, assignments=assignments)


def _paint(info):
    result = info['clean'].copy()
    for target, payload in info['assignments'].items():
        result[info['ids'] == target] = payload
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read a single input. No seed, examples, files, caches, or latent roles."""
    return _paint(_read(input_grid))


def _features(info):
    colors, arrows, assignments = info['colors'], info['arrows'], info['assignments']
    finals = [assignments.get(i, c) for i, c in enumerate(colors)]
    chain = any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                and a['source'] != b['target'] and a['host'] != b['host']
                for a in arrows for b in arrows)
    locality = any(colors[i] == colors[j] and finals[i] != finals[j]
                   for i in range(len(colors)) for j in range(i))
    occlusion = any(a['stop_index'] < len(a['hits']) - 1 for a in arrows)
    transparent = any(a['skipped'] for a in arrows)
    frozen_route = any(label in assignments and assignments[label] != a['payload']
                       for a in arrows for label in a['skipped'])
    nonrect = 0
    for i, area in enumerate(info['areas']):
        rr, cc = np.nonzero(info['ids'] == i)
        nonrect += area != (int(rr.max() - rr.min()) + 1) * (int(cc.max() - cc.min()) + 1)
    return dict(chain=chain, locality=locality, occlusion=occlusion,
                transparent=transparent, frozen_route=frozen_route,
                explicit=sum(a['explicit'] for a in arrows),
                novel=sum(a['explicit'] and a['payload'] not in colors for a in arrows),
                existing=sum(a['explicit'] and a['payload'] in colors for a in arrows),
                arrows=len(arrows), islands=len(colors), nonrect=nonrect,
                gaps=tuple(a['gap'] for a in arrows),
                directions=tuple(sorted(set(a['direction'] for a in arrows))),
                fanout=len({a['source'] for a in arrows}) < len(arrows),
                multiple_incoming=len(assignments) < len(arrows))


def _floorplan(rng, shape, count, spine=False):
    """Unequal recursive cuts with gutters; corner removal makes irregular coasts."""
    h, w = shape
    if spine:
        # Guaranteed bounded tail: three aligned islands plus a separate leaf.
        gap = int(rng.integers(1, 3))
        top = h // 2
        available = w - 2 * gap
        widths = np.full(3, 4, dtype=int)
        for _ in range(available - 12):
            widths[int(rng.integers(3))] += 1
        leaves, col = [], 0
        for width in widths:
            leaves.append((0, top, col, col + int(width)))
            col += int(width) + gap
        leaves.append((top + 2, h, 0, w))
        count = 4
    else:
        leaves = [(0, h, 0, w)]
        for _ in range(count - 1):
            options = []
            for i, (r0, r1, c0, c1) in enumerate(leaves):
                for axis, span in ((0, r1-r0), (1, c1-c0)):
                    if span >= 9:
                        options.append((i, axis, span))
            if not options:
                return None
            weights = np.array([s * s for _, _, s in options], dtype=float)
            i, axis, span = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            r0, r1, c0, c1 = leaves.pop(i)
            gutter = int(rng.integers(1, min(3, span-8)+1))
            cut = int(rng.integers(4, span-gutter-3))
            if axis == 0:
                leaves.extend(((r0, r0+cut, c0, c1), (r0+cut+gutter, r1, c0, c1)))
            else:
                leaves.extend(((r0, r1, c0, c0+cut), (r0, r1, c0+cut+gutter, c1)))
    ids = np.full(shape, -1, dtype=np.int16)
    roughness = 0. if rng.random() < .15 else .7
    for i, (r0, r1, c0, c1) in enumerate(leaves):
        # Variable inset independently broadens corridor lengths.
        if r1-r0 >= 7 and rng.random() < .4:
            r0 += int(rng.integers(0, 2))
            r1 -= int(rng.integers(0, 2))
        if c1-c0 >= 7 and rng.random() < .4:
            c0 += int(rng.integers(0, 2))
            c1 -= int(rng.integers(0, 2))
        ids[r0:r1, c0:c1] = i
        if not spine and rng.random() < roughness:
            # A corner notch leaves the central cross connected and readable.
            for corner in rng.choice(4, size=int(rng.integers(1, 4)), replace=False):
                dh = int(rng.integers(1, max(2, (r1-r0)//3)))
                dw = int(rng.integers(1, max(2, (c1-c0)//3)))
                rs = slice(r0, r0+dh) if corner < 2 else slice(r1-dh, r1)
                cs = slice(c0, c0+dw) if corner % 2 == 0 else slice(c1-dw, c1)
                ids[rs, cs] = -1
    return ids


def _candidates(ids):
    result = []
    for center in np.ndindex(ids.shape):
        source = int(ids[center])
        if source < 0:
            continue
        for d in CARDINAL:
            a = _stencil(center, d, ids.shape)
            if a is None or ids[a['exit']] >= 0:
                continue
            if not all(ids[p] == source for p in (*a['write'], *a['flanks'], a['back'])):
                continue
            hits, gap = _ray_targets(ids, a['exit'], d)
            if hits and hits[0][0] != source:
                a.update(source=source, target=hits[0][0], hits=hits, gap=gap)
                result.append(a)
    return result


def _compatible(a, chosen):
    context = set(a['write']) | set(a['flanks']) | {a['back'], a['exit']}
    halo = {q for p in a['write'] for q in (p, *(_add(p, d) for d in CARDINAL))}
    return all(not (set(b['write']) & (context | halo)
                    or set(a['write']) & (set(b['flanks']) | {b['back'], b['exit']}))
               for b in chosen)


def _build(rng, kind, shifted=False, fallback=False):
    # kind='chain' teaches original-color dependency and component locality;
    # kind='payload' teaches color-selective transparency and frozen routing:
    # one beam passes a matching island while a second repaints that island.
    low, high = (22, 31) if shifted else (16, 26)
    shape = (int(rng.integers(low, high)), int(rng.integers(low, high)))
    count = int(rng.integers(6, 9) if shifted else rng.integers(4, 7))
    if fallback:
        shape, count = (22, 27), 4
    ids = _floorplan(rng, shape, count, spine=fallback)
    if ids is None:
        return None
    count = int(ids.max()) + 1
    candidates = _candidates(ids)
    if len(candidates) < 2:
        return None
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    chosen = []
    special_ids = transparent_ids = prefix_plan = None
    if kind == 'chain':
        for a in candidates:
            for b in candidates:
                if (b['source'] == a['target'] and b['target'] != a['source']
                        and _compatible(b, [a])):
                    spare = next((i for i in range(count) if i not in
                                  (a['source'], a['target'], b['target'])), None)
                    if spare is not None:
                        chosen = [a, b]
                        special_ids = a['source'], a['target'], b['target'], spare
                        break
            if chosen:
                break
        if not chosen:
            return None
    elif kind == 'payload':
        for a in candidates:
            if len(a['hits']) < 2:
                continue
            near, far = a['hits'][0][0], a['hits'][1][0]
            b = next((b for b in candidates if b['source'] == far and b['target'] == near
                      and len(b['hits']) >= 2 and _compatible(b, [a])), None)
            if b is not None:
                chosen = [a, b]
                transparent_ids = a['source'], near, far
                break
        if not chosen:
            return None
    elif kind == 'transparent':
        first = next((a for a in candidates if len(a['hits']) >= 2), None)
        if first is None:
            return None
        skip = int(rng.integers(1, min(3, len(first['hits'])-1)+1))
        prefix_plan = (first['source'], [label for label, _ in first['hits'][:skip]],
                       first['hits'][skip][0], bool(rng.integers(2)))
        chosen = [first]
        desired = int(rng.integers(1, min(7, count+2)))
        for a in candidates:
            if len(chosen) >= desired:
                break
            if _compatible(a, chosen):
                chosen.append(a)
    else:
        desired = int(rng.integers(1, min(7, count+2)))
        for a in candidates:
            if _compatible(a, chosen):
                chosen.append(a)
            if len(chosen) >= desired:
                break
    if not chosen:
        return None
    palette = list(map(int, rng.permutation(np.arange(2, 10))))
    pool = palette[:int(rng.integers(3, 6))]
    colors = [int(rng.choice(pool)) for _ in range(count)]
    if special_ids:
        s, m, t, spare = special_ids
        colors[s], colors[m], colors[t], colors[spare] = pool[0], pool[1], pool[2], pool[2]
    if transparent_ids:
        s, near, far = transparent_ids
        colors[s], colors[near], colors[far] = pool[0], pool[1], pool[2]
    if prefix_plan:
        source, prefix, receiver, explicit_first = prefix_plan
        for label in prefix:
            colors[label] = pool[0]
        colors[source] = pool[1] if explicit_first else pool[0]
        colors[receiver] = pool[2]
    assignments, annotated = {}, []
    explicit_rate = float(rng.choice([0., .2, .5, .8]))
    for index, original in enumerate(chosen):
        a = dict(original)
        source = a['source']
        if kind == 'chain':
            explicit, payload = False, colors[source]
        elif kind == 'payload' and index == 0:
            explicit, payload = True, colors[transparent_ids[1]]
        elif kind == 'payload' and index == 1:
            explicit, payload = True, palette[-1]
        elif kind == 'transparent' and index == 0:
            explicit, payload = prefix_plan[3], pool[0]
        else:
            explicit = bool(rng.random() < explicit_rate)
            if explicit:
                choices = [c for c in range(2, 10) if c != colors[source]]
                payload = int(rng.choice(choices))
            else:
                payload = colors[source]
        stops = [label for label, _ in a['hits'] if colors[label] != payload]
        if not stops:
            continue
        target = stops[0]
        if target in assignments and assignments[target] != payload:
            continue
        a['target'] = target
        a.update(explicit=explicit, payload=payload)
        annotated.append(a)
        assignments[target] = payload
    if len(annotated) < (2 if kind in ('chain', 'payload') else 1):
        return None
    clean = np.zeros(shape, dtype=np.int8)
    for label, color in enumerate(colors):
        clean[ids == label] = color
    raw, out = clean.copy(), clean.copy()
    for a in annotated:
        for p in a['write']:
            raw[p] = 1
        if a['explicit']:
            raw[a['center']] = a['payload']
    for target, payload in assignments.items():
        out[ids == target] = payload
    # A rigid transform varies absolute directions, layout, and aspect ratio.
    turn = int(rng.integers(4))
    raw, out = np.rot90(raw, turn).copy(), np.rot90(out, turn).copy()
    if rng.random() < .5:
        raw, out = np.fliplr(raw).copy(), np.fliplr(out).copy()
    try:
        info = _read(raw)
    except AssertionError:
        return None
    if not np.array_equal(_paint(info), out):
        return None
    f = _features(info)
    if kind == 'chain' and not (f['chain'] and f['locality']):
        return None
    if kind == 'payload' and not (f['novel'] and f['existing'] and f['occlusion']
                                  and f['transparent'] and f['frozen_route']):
        return None
    if kind == 'transparent' and not f['transparent']:
        return None
    return dict(input=raw, output=out)


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError('num_test must be an integer in 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    kinds = ['chain', 'payload'] + [str(rng.choice(['free','transparent'], p=[.75,.25]))
                                  for _ in range(num_train-2)]
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = kinds[i] if split == 'train' else str(rng.choice(['free','transparent'], p=[.4,.6]))
            for attempt in range(100):
                pair = _build(rng, kind, shifted=split == 'test', fallback=attempt >= 85)
                if pair is not None and _key(pair['input']) not in seen:
                    if split == 'train' and i == 1:
                        directions = {a['direction'] for p in puzzle['train'] + [pair]
                                      for a in _read(p['input'])['arrows']}
                        if len({d[0] != 0 for d in directions}) < 2:
                            continue
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded geometry construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            raw, out = _grid(pair['input']), _grid(pair['output'])
            _need(raw.shape == out.shape, 'dimensions must be preserved')
            key = _key(raw)
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            info = _read(raw)
            expected = _paint(info)
            _need(np.array_equal(out, expected), 'incorrect simultaneous ray-addressed fill')
            _need(np.array_equal(out == 0, raw == 0), 'void geometry must be preserved')
            _need(not np.any(out == 1), 'launcher ink survived')
            _need(all(area >= 8 for area in info['areas']), 'an island is too small')
            _need(all(a['payload'] != info['colors'][a['target']] for a in info['arrows']),
                  'redundant instruction')
            # Omitting A: painting just each contact pixel is observably different.
            pixel_only = info['clean'].copy()
            for a in info['arrows']:
                pixel_only[a['contact']] = a['payload']
            _need(not np.array_equal(pixel_only, out), 'whole-island fill is not necessary')
            # Omitting B: an adjacency-only arrow has no adjacent colored target.
            _need(all(info['clean'][a['exit']] == 0 for a in info['arrows']), 'missing remote gap')
            _need(not np.array_equal(info['clean'], out), 'ray targeting is not necessary')
            for label, color in enumerate(info['colors']):
                expected_color = info['assignments'].get(label, color)
                _need(np.all(out[info['ids'] == label] == expected_color), 'partial or cascading fill')
            features[split].append(_features(info))
    taught = features['train']
    _need(any(f['chain'] for f in taught), 'missing simultaneous dependency demonstration')
    _need(any(f['locality'] for f in taught), 'missing repeated-color locality demonstration')
    _need(any(f['novel'] for f in taught), 'missing literal payload demonstration')
    _need(any(f['occlusion'] for f in taught), 'missing nearest-impact demonstration')
    _need(any(f['transparent'] and f['frozen_route'] for f in taught),
          'missing color-selective and simultaneous routing demonstration')
    _need(len({d[0] != 0 for f in taught for d in f['directions']}) == 2,
          'training must expose both horizontal and vertical addressing')
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def _manual_checks():
    """Small hand-specified outcomes, separate from the procedural planner."""
    clean = np.array([[int(c) for c in '2222200333330044444']] * 7, dtype=np.int8)
    raw = clean.copy()
    # A rightward T in pigment 2, carrying literal pigment 3.
    raw[1, 3] = raw[3, 3] = raw[2, 4] = 1
    raw[2, 3] = 3
    expected = np.array([[int(c) for c in '2222200333330033333']] * 7, dtype=np.int8)
    _need(np.array_equal(solve(raw), expected), 'matching island must be transparent')
    changed = raw.copy()
    changed[2, 3] = 6
    expected = np.array([[int(c) for c in '2222200666660044444']] * 7, dtype=np.int8)
    _need(np.array_equal(solve(changed), expected), 'payload must change the recipient, not just the paint')
    # A leftward literal-5 T on the far island repaints the transparent island.
    raw[3, 15] = raw[5, 15] = raw[4, 14] = 1
    raw[4, 15] = 5
    expected = np.array([[int(c) for c in '2222200555550033333']] * 7, dtype=np.int8)
    _need(np.array_equal(solve(raw), expected), 'routing must use original colors simultaneously')
    conflict = raw.copy()
    conflict[2, 3] = 6
    try:
        solve(conflict)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory incoming pigments accepted')
    # An implicit open chain sends 2 to the middle and original 3 to the far.
    raw = clean.copy()
    raw[1, 3] = raw[3, 3] = raw[2, 4] = raw[2, 3] = 1
    raw[3, 10] = raw[5, 10] = raw[4, 11] = raw[4, 10] = 1
    expected = np.array([[int(c) for c in '2222200222220033333']] * 7, dtype=np.int8)
    _need(np.array_equal(solve(raw), expected), 'implicit pigment must be the original host color')
    raw = np.array([[int(c) for c in '22222003333300333330044444']] * 7, dtype=np.int8)
    raw[1, 3] = raw[3, 3] = raw[2, 4] = 1
    raw[2, 3] = 3
    expected = np.array([[int(c) for c in '22222003333300333330033333']] * 7, dtype=np.int8)
    _need(np.array_equal(solve(raw), expected), 'ray must pass all matching islands, not just one')


def self_test():
    """Bounded, streaming 42-seed audit, algebraic checks, and corruption tests."""
    _manual_checks()
    fingerprints, shapes, partitions, directions = set(), set(), set(), set()
    arrows, islands, gaps, palettes, prefix_depths = set(), set(), set(), set(), set()
    transparent_queries = 0
    totals = dict(pairs=0, nonrect=0, chain=0, locality=0, occlusion=0,
                  transparent=0, frozen_route=0, novel=0,
                  existing=0, fanout=0, multiple_incoming=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_digest(puzzle))
        if seed < 2:
            _need(_digest(puzzle) == _digest(generate_puzzle(seed)), 'nondeterministic puzzle')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                raw, out = pair['input'], pair['output']
                saved = raw.copy()
                _need(np.array_equal(solve(raw), out) and np.array_equal(raw, saved), 'solve mutated input')
                info = _read(raw)
                f = _features(info)
                if split == 'test':
                    transparent_queries += bool(f['transparent'])
                prefix_depths.update(len(a['skipped']) for a in info['arrows'])
                totals['pairs'] += 1
                for k in totals:
                    if k != 'pairs':
                        totals[k] += int(bool(f[k]))
                shapes.add(raw.shape)
                arrows.add(f['arrows'])
                islands.add(f['islands'])
                gaps.update(f['gaps'])
                directions.update(f['directions'])
                palettes.add(tuple(sorted(set(info['colors']))))
                # Color-free component structure, compact hash only.
                partitions.add(hashlib.sha256(info['ids'].tobytes()).hexdigest())
                if seed < 3:
                    for turn in (1, 2, 3):
                        _need(np.array_equal(solve(np.rot90(raw, turn)), np.rot90(out, turn)),
                              'rotation equivariance failed')
                    _need(np.array_equal(solve(np.fliplr(raw)), np.fliplr(out)), 'reflection failed')
                    perm = np.arange(10)
                    perm[2:] = np.random.default_rng(seed+900).permutation(np.arange(2, 10))
                    _need(np.array_equal(solve(perm[raw]), perm[out]), 'pigment roles depend on numeric value')
    _need(len(fingerprints) == 42 and len(partitions) >= 180, 'collapsed structural diversity')
    _need(len(shapes) >= 45 and len(palettes) >= 20, 'collapsed scale or palette diversity')
    _need(directions == set(CARDINAL), 'missing cardinal directions')
    _need(len(arrows) >= 4 and len(islands) >= 4 and len(gaps) >= 5, 'collapsed counts or ray lengths')
    _need(totals['nonrect'] >= 150 and totals['existing'] >= 5, 'missing irregularity or existing pigments')
    _need(transparent_queries >= 15 and totals['transparent'] >= 70,
          'selective transparency is underrepresented')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(73+nt*11+nq, nt, nq))
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),(2.5,1),(4,None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    bad = generate_puzzle(123)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                component_partitions=len(partitions), palettes=len(palettes),
                arrow_counts=sorted(arrows), island_counts=sorted(islands),
                gap_lengths=sorted(gaps), directions=sorted(directions),
                transparent_queries=transparent_queries, prefix_depths=sorted(prefix_depths), **totals)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
