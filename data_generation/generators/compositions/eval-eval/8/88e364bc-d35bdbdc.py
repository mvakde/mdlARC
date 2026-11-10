"""Collision compositions: rigid motion and local, simultaneous color lookup.

Only Python's standard library and NumPy are used. No file I/O occurs during
generation, solving, validation or self_test. The solver reads one input only.
"""
from __future__ import annotations

import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(np.all((0 <= a) & (a <= 9)), 'color bounds')
    return a


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, part = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _offsets(kind):
    return tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                 if kind == 'square' or r == 0 or c == 0)


def _support(center, kind):
    return {(center[0] + r, center[1] + c) for r, c in _offsets(kind)}


def _key(rng, direction, wall, trail, tip, compact=False):
    dr, dc = direction
    length = 2 if compact else int(rng.choice([2, 2, 3]))
    spacing = 2 if compact else int(rng.choice([2, 3]))
    points = []
    for k in range(2):
        base = (k * spacing, 0) if dr == 0 else (0, k * spacing)
        for j in range(length + 1):
            points.append((base[0] + j * dr, base[1] + j * dc,
                           tip if j == length else trail))
    r0, c0 = min(p[0] for p in points), min(p[1] for p in points)
    h = max(p[0] for p in points) - r0 + 3
    w = max(p[1] for p in points) - c0 + 3
    a = np.full((h, w), wall, dtype=np.int8)
    for r, c, color in points:
        a[r - r0 + 1, c - c0 + 1] = color
    return a


def _read(a):
    """Recover roles, legend and shapes without demonstrations or RNG state."""
    a = _grid(a)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _require(len(set(map(int, border))) == 1, 'uniform exterior border required')
    bg = int(border[0])
    keys = []
    for part in _components(map(tuple, np.argwhere(a != bg))):
        rs, cs = zip(*part)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 5 or len(part) != h * w:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        wall = int(sub[0, 0])
        rim = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        colors = set(map(int, np.unique(sub))) - {wall}
        if len(colors) != 2 or np.any(rim != wall):
            continue
        interpretations = []
        for tip in sorted(colors):
            trail = next(iter(colors - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            for dr, dc in D8:
                used, lengths = set(), []
                for r, c in sorted(tips):
                    stroke = []
                    p = r - dr, c - dc
                    while p in trails:
                        stroke.append(p)
                        p = p[0] - dr, p[1] - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(tips) >= 2 and len(lengths) == len(tips) and used == trails:
                    interpretations.append(((dr, dc), trail, tip))
        if len(interpretations) == 1:
            direction, trail, tip = interpretations[0]
            keys.append((part, wall, direction, trail, tip, (h, w)))
    _require(len(keys) == 1, 'one uniquely readable legend required')
    key_cells, wall, direction, trail, tip, key_shape = keys[0]
    walls = set(map(tuple, np.argwhere(a == wall))) - key_cells
    _require(bool(walls), 'missing enclosure')
    tokens = []
    mask = (a != bg) & (a != wall)
    cells = set(map(tuple, np.argwhere(mask))) - key_cells
    for part in _components(cells):
        candidates = []
        for p in sorted(part):
            for kind in ('square', 'cross'):
                if _support(p, kind) != part:
                    continue
                bodies = {int(a[q]) for q in part if q != p}
                if len(bodies) == 1:
                    candidates.append(dict(center=p, kind=kind, cells=part,
                                           body=bodies.pop(), value=int(a[p])))
        _require(len(candidates) == 1, 'unique full square/cross token required')
        token = candidates[0]
        _require(not ({token['body'], token['value']} & {bg, wall, trail, tip}),
                 'structural colors cannot be token identities')
        tokens.append(token)
    _require(len(tokens) >= 2, 'at least two relation tokens required')
    # Every token lies in closed free space. Internal wall islands are allowed.
    free = {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])
            if (r, c) not in walls}
    exterior = set()
    stack = [(0, 0)]
    while stack:
        p = stack.pop()
        if p in exterior or p not in free:
            continue
        exterior.add(p)
        stack.extend((p[0] + dr, p[1] + dc) for dr, dc in D4)
    _require(all(not (t['cells'] & exterior) for t in tokens), 'unenclosed token')
    return dict(background=bg, wall=wall, trail=trail, tip=tip, direction=direction,
                key_cells=key_cells, key_shape=key_shape, walls=walls, tokens=tokens)


def _motion(tokens, walls, direction, shape, corner=True, use_walls=True):
    """Return independent stops against the immutable occupancy snapshot."""
    occupied = {p: i for i, token in enumerate(tokens) for p in token['cells']}
    dr, dc = direction
    reports = []
    for i, token in enumerate(tokens):
        offsets = _offsets(token['kind'])
        r, c = token['center']
        for distance in range(31):
            shifts = [(dr, dc)]
            if corner and dr and dc:
                shifts += [(dr, 0), (0, dc)]
            hit_wall, blockers = False, set()
            direct_wall = False
            for sr, sc in shifts:
                for ar, ac in offsets:
                    p = r + sr + ar, c + sc + ac
                    blocked = (not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1])
                               or (use_walls and p in walls))
                    if blocked:
                        hit_wall = True
                        if (sr, sc) == (dr, dc):
                            direct_wall = True
                    if p in occupied and occupied[p] != i:
                        blockers.add(occupied[p])
            if hit_wall or blockers:
                donor = None
                if not hit_wall and len(blockers) == 1:
                    j = next(iter(blockers))
                    dy = tokens[j]['center'][0] - token['center'][0]
                    dx = tokens[j]['center'][1] - token['center'][1]
                    steps = dy * dr if dr else dx * dc
                    aligned = steps > 0 and (dy, dx) == (steps * dr, steps * dc)
                    if aligned and tokens[j]['body'] == token['value']:
                        donor = j
                reports.append(dict(stop=(r, c), distance=distance, donor=donor,
                                    blockers=blockers, hit_wall=hit_wall,
                                    corner_wall=hit_wall and not direct_wall))
                break
            r, c = r + dr, c + dc
        else:
            raise AssertionError('unbounded trajectory')
    return reports


def _render(a, info, reports):
    out = a.copy()
    for token in info['tokens']:
        for p in token['cells']:
            out[p] = info['background']
    used = set()
    for token, report in zip(info['tokens'], reports):
        j = report['donor']
        if j is None:
            continue
        cells = _support(report['stop'], token['kind'])
        _require(not (cells & used), 'overlapping deliveries')
        _require(not (cells & (info['walls'] | info['key_cells'])), 'delivery hits structure')
        used.update(cells)
        for p in cells:
            out[p] = token['body']
        out[report['stop']] = info['tokens'][j]['value']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _read(a)
    reports = _motion(info['tokens'], info['walls'], info['direction'], a.shape)
    return _render(a, info, reports)


def _room(rng, h, w, complexity, rectangle=False):
    footprint = np.ones((h, w), dtype=bool)
    if not rectangle:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dr, dc in D4:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    walls = footprint & ~interior
    for _ in range(complexity):
        if rng.random() < .5:
            r, length = int(rng.integers(3, h - 3)), int(rng.integers(2, max(3, w // 2)))
            walls[r, :length] |= footprint[r, :length]
        else:
            c, length = int(rng.integers(3, w - 3)), int(rng.integers(2, max(3, h // 2)))
            walls[:length, c] |= footprint[:length, c]
    return footprint, walls


def _core_relations(n, mode):
    relations = [(i, i + 1) for i in range(n)]
    # A matching relation immediately behind the leader must not pull it back.
    # In the cardinal witness this also teaches a newly solid delivered token.
    relations[-1] = (n - 1, n - 2)
    if mode in ('mismatch', 'both'):
        relations[0] = (0, 2)
    return relations


def _make_pair(rng, n, mode, diagonal, varied_roles, roles, complexity):
    direction = (1, 1) if diagonal else (0, 1)
    palette = list(map(int, rng.permutation(10))) if varied_roles else list(roles[:4]) + list(map(int, rng.permutation(roles[4:])))
    bg, wall, trail, tip = palette[:4]
    identities = palette[4:]
    key = _key(rng, direction, wall, trail, tip)
    # Keep room + gap + legend + exterior border within 30 rows.
    shutter = mode in ('shutter', 'both')
    minimum_height = 4 * (n - 1) + int(shutter) + 5 if diagonal else 12
    if minimum_height > 27 - key.shape[0]:
        key = _key(rng, direction, wall, trail, tip, compact=True)
    max_h = 27 - key.shape[0]
    kinds = [str(rng.choice(['square', 'cross'])) for _ in range(n)]
    # A convoy demonstrates mixed rigid supports in both minimal training pairs.
    kinds[0], kinds[1] = 'square', 'cross'
    if diagonal and shutter:
        kinds[-2] = 'square'
    relations = _core_relations(n, mode)
    for attempt in range(64):
        fallback = attempt >= 48
        gaps = [4 for _ in range(n - 1)]
        if shutter:
            gaps[-1] = 5
        if not fallback:
            budget = (max_h - 5 if diagonal else 23) - sum(gaps)
            for index in rng.permutation(n - 1):
                added = int(rng.integers(0, min(6 - gaps[index], budget) + 1))
                gaps[index] += added
                budget -= added
        span = sum(gaps)
        low_h = max(12, span + 5 if diagonal else 12)
        if low_h > max_h:
            continue
        h = max_h if fallback else int(rng.integers(low_h, max_h + 1))
        low_w = max(14, span + 5, key.shape[1])
        w = 28 if fallback else int(rng.integers(low_w, 29))
        footprint, wall_mask = _room(rng, h, w, 0 if fallback else complexity,
                                     rectangle=fallback or rng.random() < .22)
        walls = set(map(tuple, np.argwhere(wall_mask)))
        free = set(map(tuple, np.argwhere(footprint & ~wall_mask)))
        cumulative = [0]
        for gap in gaps:
            cumulative.append(cumulative[-1] + gap)
        starts = list(map(tuple, np.argwhere(footprint & ~wall_mask)))
        rng.shuffle(starts)
        chosen = None
        for start in starts:
            centers = [(start[0] + s * direction[0], start[1] + s) for s in cumulative]
            tokens = [dict(center=p, kind=k, cells=_support(p, k), body=identities[b], value=identities[v])
                      for p, k, (b, v) in zip(centers, kinds, relations)]
            if not all(t['cells'] <= free for t in tokens):
                continue
            local_walls = walls.copy()
            if shutter:
                # A physical inner wall screens the last donor, without erasing it.
                r, c = centers[-2]
                # Diagonal: one wall cell blocks a side translation at step 2,
                # although the next diagonal footprint and the ray beyond are
                # free. This explicitly teaches the inherited no-corner-cut rule.
                barrier = ({(r, c + 3)} if diagonal
                           else {(r + q, c + 3) for q in (-1, 0, 1)})
                if not barrier <= free or any(barrier & t['cells'] for t in tokens):
                    continue
                local_walls.update(barrier)
            reports = _motion(tokens, local_walls, direction, (h, w))
            expected = [i + 1 for i in range(n - 1)] + [None]
            if mode in ('mismatch', 'both'):
                expected[0] = None
            if shutter:
                expected[-2] = None
            if [r['donor'] for r in reports] != expected:
                continue
            if any((not reports[i]['hit_wall'] if shutter and i == n - 2
                    else reports[i]['hit_wall'] or reports[i]['blockers'] != {i + 1})
                   for i in range(n - 1)):
                continue
            if diagonal and shutter:
                direct = _motion(tokens, local_walls, direction, (h, w), corner=False)
                if direct[-2]['donor'] != n - 1:
                    continue
            if mode == 'chain':
                reverse = _motion(tokens, local_walls, tuple(-x for x in direction), (h, w))
                if reverse[-1]['donor'] != n - 2:
                    continue
            if any(r['distance'] < 1 for r in reports[:-1]):
                continue
            chosen = tokens, local_walls, reports
            break
        if chosen is None:
            continue
        tokens, walls, reports = chosen
        unobstructed = _motion(tokens, walls, direction, (h, w), use_walls=False)
        # Extra donors/distractors occupy sampled 2D locations. A duplicate of
        # body 1 with a different value makes physical donor selection visible.
        target_extra = int(rng.integers(1, 4))
        for extra in range(target_extra):
            positions = starts.copy()
            rng.shuffle(positions)
            for p in positions[:150]:
                kind = str(rng.choice(['square', 'cross']))
                cells = _support(p, kind)
                if not cells <= free or cells & walls:
                    continue
                if any(any(abs(r - s) + abs(c - t) <= 1 for r, c in cells for s, t in old['cells'])
                       for old in tokens):
                    continue
                b = 1 if extra == 0 else int(rng.integers(0, 6))
                v = 0 if extra == 0 else int(rng.integers(0, 6))
                new = dict(center=p, kind=kind, cells=cells, body=identities[b], value=identities[v])
                proposed = tokens + [new]
                new_reports = _motion(proposed, walls, direction, (h, w))
                if any(new_reports[i]['donor'] != reports[i]['donor'] or
                       new_reports[i]['stop'] != reports[i]['stop'] or
                       new_reports[i]['blockers'] != reports[i]['blockers'] or
                       new_reports[i]['hit_wall'] != reports[i]['hit_wall'] for i in range(n)):
                    continue
                no_walls = _motion(proposed, walls, direction, (h, w), use_walls=False)
                if any(no_walls[i]['donor'] != unobstructed[i]['donor'] for i in range(n)):
                    continue
                if diagonal and shutter:
                    direct = _motion(proposed, walls, direction, (h, w), corner=False)
                    if direct[n - 2]['donor'] != n - 1:
                        continue
                if mode == 'chain':
                    reverse = _motion(proposed, walls, tuple(-x for x in direction), (h, w))
                    if reverse[n - 1]['donor'] != n - 2:
                        continue
                # A clearly erased but globally compatible token in every pair.
                if extra == 0 and new_reports[-1]['donor'] is not None:
                    continue
                tokens, reports = proposed, new_reports
                break
        if len(tokens) == n:
            continue
        canvas = np.full((h + key.shape[0] + 3, w + 2), bg, dtype=np.int8)
        kc = int(rng.integers(1, w + 2 - key.shape[1]))
        canvas[1:1 + key.shape[0], kc:kc + key.shape[1]] = key
        room_row = key.shape[0] + 2
        for r, c in walls:
            canvas[room_row + r, 1 + c] = wall
        for token in tokens:
            for r, c in token['cells']:
                canvas[room_row + r, 1 + c] = token['body']
            r, c = token['center']
            canvas[room_row + r, 1 + c] = token['value']
        # Construct labels directly from sampled relations and collision records.
        output = canvas.copy()
        for token in tokens:
            for r, c in token['cells']:
                output[room_row + r, 1 + c] = bg
        for token, report in zip(tokens, reports):
            donor = report['donor']
            if donor is not None:
                for r, c in _support(report['stop'], token['kind']):
                    output[room_row + r, 1 + c] = token['body']
                r, c = report['stop']
                output[room_row + r, 1 + c] = tokens[donor]['value']
        rotation = int(rng.integers(4))
        canvas, output = np.rot90(canvas, rotation), np.rot90(output, rotation)
        if rng.random() < .5:
            canvas, output = np.fliplr(canvas), np.fliplr(output)
        pair = {'input': canvas.copy(), 'output': output.copy()}
        try:
            predicted = solve(pair['input'])
            _require(np.array_equal(predicted, pair['output']), 'construction collision overlap')
        except AssertionError:
            continue
        return pair
    raise AssertionError('bounded scene construction exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.permutation(10)))
    varied_roles = bool(rng.random() < .65)
    plans = [(3, 'chain', False), (4, 'both', True)]
    for _ in range(nt - 2):
        mode = str(rng.choice(['chain', 'mismatch', 'shutter', 'both']))
        plans.append((int(rng.integers(4 if mode == 'both' else 3, 5)), mode, bool(rng.integers(2))))
    rng.shuffle(plans)
    test_plans = [(int(rng.integers(4, 6)), str(rng.choice(['chain', 'mismatch', 'shutter', 'both'])),
                   bool(rng.integers(2))) for _ in range(nq)]
    result, seen = {'train': [], 'test': []}, set()
    for split, split_plans in (('train', plans), ('test', test_plans)):
        for n, mode, diagonal in split_plans:
            for _ in range(8):
                pair = _make_pair(rng, n, mode, diagonal, varied_roles, roles,
                                  int(rng.integers(0, 3 if split == 'train' else 4)))
                if varied_roles and split == 'train' and len(result['train']) == 1:
                    if _features(pair['input'])['roles'] == _features(result['train'][0]['input'])['roles']:
                        continue
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    break
            else:
                raise AssertionError('bounded unique scene construction exhausted')
            seen.add(key)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _features(a):
    info = _read(a)
    tokens = info['tokens']
    reports = _motion(tokens, info['walls'], info['direction'], a.shape)
    survivors = [i for i, r in enumerate(reports) if r['donor'] is not None]
    compatible_erased = [i for i, t in enumerate(tokens) if i not in survivors
                         and any(j != i and u['body'] == t['value'] for j, u in enumerate(tokens))]
    nonrecursive = any(reports[i]['donor'] in survivors and
                       tokens[reports[i]['donor']]['value'] !=
                       tokens[reports[reports[i]['donor']]['donor']]['value'] for i in survivors)
    no_walls = _motion(tokens, info['walls'], info['direction'], a.shape, use_walls=False)
    wall_witness = any(r['donor'] != s['donor'] for r, s in zip(reports, no_walls))
    mismatch_witness = any(not r['hit_wall'] and r['donor'] is None and r['blockers']
                           and any(j != i and u['body'] == tokens[i]['value']
                                   for j, u in enumerate(tokens))
                           for i, r in enumerate(reports))
    direct = _motion(tokens, info['walls'], info['direction'], a.shape, corner=False)
    corner_witness = any(r['donor'] != s['donor'] for r, s in zip(reports, direct))
    reverse = _motion(tokens, info['walls'], tuple(-x for x in info['direction']), a.shape)
    reverse_witness = any(r['donor'] is None and s['donor'] is not None for r, s in zip(reports, reverse))
    rs, cs = zip(*info['walls'])
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    perimeter = ({(r0, c) for c in range(c0, c1 + 1)} |
                 {(r1, c) for c in range(c0, c1 + 1)} |
                 {(r, c0) for r in range(r0, r1 + 1)} |
                 {(r, c1) for r in range(r0, r1 + 1)})
    layout_variants = []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                points = [(sr * (t['center'][1] if swap else t['center'][0]),
                           sc * (t['center'][0] if swap else t['center'][1])) for t in tokens]
                mr, mc = min(p[0] for p in points), min(p[1] for p in points)
                layout_variants.append(tuple(sorted((r - mr, c - mc) for r, c in points)))
    return dict(shape=a.shape, direction=info['direction'], key_shape=info['key_shape'],
                roles=(info['background'], info['wall'], info['trail'], info['tip']),
                tokens=len(tokens), survivors=len(survivors), kinds=tuple(sorted({t['kind'] for t in tokens})),
                distances=tuple(r['distance'] for i, r in enumerate(reports) if i in survivors),
                compatible_erased=bool(compatible_erased), nonrecursive=nonrecursive,
                wall_witness=wall_witness, mismatch_witness=bool(mismatch_witness),
                repeated_bodies=len({t['body'] for t in tokens}) < len(tokens),
                corner_wall=any(r['corner_wall'] for r in reports), corner_witness=corner_witness,
                reverse_witness=reverse_witness,
                irregular=not perimeter <= info['walls'], room_shape=(r1-r0+1, c1-c0+1),
                layout=min(layout_variants))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and not np.array_equal(a, b), 'shape or identity output')
            fingerprint = a.shape, a.astype(np.int8).tobytes()
            _require(fingerprint not in seen, 'duplicate input')
            seen.add(fingerprint)
            info = _read(a)
            reports = _motion(info['tokens'], info['walls'], info['direction'], a.shape)
            expected = _render(a, info, reports)
            _require(np.array_equal(expected, b), 'collision/substitution output mismatch')
            _require(all(b[p] == a[p] for p in info['walls'] | info['key_cells']), 'changed structure')
            _require(any(r['donor'] is not None and r['distance'] > 0 for r in reports), 'no delivered move')
            _require(any(r['donor'] is None for r in reports), 'no rejected token')
            measurements[split].append(_features(a))
    train = measurements['train']
    _require(any(f['nonrecursive'] for f in train), 'missing simultaneous-read demonstration')
    _require(any(f['wall_witness'] for f in train), 'missing wall-occlusion demonstration')
    _require(any(f['corner_witness'] for f in train), 'missing corner-blocking demonstration')
    _require(any(f['reverse_witness'] for f in train), 'missing direction-selection demonstration')
    _require(any(f['mismatch_witness'] for f in train), 'missing nearer-mismatch demonstration')
    _require(any(f['compatible_erased'] for f in train), 'missing local-selection demonstration')
    _require(any(0 in f['direction'] for f in train) and any(0 not in f['direction'] for f in train),
             'teach cardinal and diagonal translation')
    _require(any(f['kinds'] == ('cross', 'square') for f in train), 'teach both rigid supports')
    # No latent palette dictionary: stable role puzzles keep only structural roles.
    roles = {f['roles'] for f in train}
    if len(roles) == 1:
        _require({f['roles'] for f in measurements['test']} <= roles, 'unshown role reassignment')


def self_test():
    """42 streaming seeds, geometric invariants, corruption and count checks."""
    fingerprints, rows = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                for role in ('input', 'output'):
                    a = pair[role]
                    digest.update(bytes(a.shape) + a.tobytes())
                rows.append(_features(pair['input']))
        fingerprints.add(digest.hexdigest())
    _require(len(fingerprints) == 42, 'seed diversity')
    _require({r['direction'] for r in rows} == set(D8), 'eight direction coverage')
    _require(len({r['shape'] for r in rows}) >= 25, 'canvas diversity')
    _require(len({r['key_shape'] for r in rows}) >= 8, 'legend geometry diversity')
    _require(len({r['tokens'] for r in rows}) >= 4, 'token count diversity')
    _require(len({r['survivors'] for r in rows}) >= 3, 'delivery count diversity')
    _require(len({r['roles'] for r in rows}) >= 25, 'geometric palette variation')
    _require(len({r['layout'] for r in rows}) >= 180, 'token geometry variation')
    _require(any(r['irregular'] for r in rows) and any(not r['irregular'] for r in rows),
             'simple and notched enclosures')
    _require({d for r in rows for d in r['distances']} >= {1, 2, 3}, 'movement distance diversity')
    for feature in ('nonrecursive', 'compatible_erased', 'wall_witness', 'mismatch_witness', 'repeated_bodies', 'corner_witness', 'reverse_witness'):
        _require(sum(r[feature] for r in rows) >= 20, feature)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(123, nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    a, b = generate_puzzle(87, 2, 1), generate_puzzle(87, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'determinism')
    a['test'][0]['output'] = a['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # A structural corruption must also fail, independently of token labels.
    info = _read(b['test'][0]['input'])
    p = min(info['walls'])
    b['test'][0]['output'][p] = info['background']
    try:
        validate_generated_puzzle(b)
    except AssertionError:
        pass
    else:
        raise AssertionError('wall corruption accepted')
    return {'seeds': 42, 'pairs': len(rows), 'directions': len({r['direction'] for r in rows}),
            'shapes': len({r['shape'] for r in rows}), 'token_counts': sorted({r['tokens'] for r in rows}),
            'delivery_counts': sorted({r['survivors'] for r in rows}),
            'normalized_layouts': len({r['layout'] for r in rows}),
            'irregular_rooms': sum(r['irregular'] for r in rows)}
