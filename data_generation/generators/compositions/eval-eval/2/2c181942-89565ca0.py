"""Chambered keys: minimum-room selection within color, then geometric docking.

Only standard library and NumPy are used.  The solver reads the current input;
generation parameters and seeds are never consulted by the solver.
"""
from collections import defaultdict

import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, neighbors=_EIGHT):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    # Sorted starts make extraction independent of hash iteration order.
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.asarray(cells, dtype=int))
    return result


def _rooms(mask):
    """Elementary bounded four-connected regions, in coordinates of mask."""
    free = np.pad(~mask, 1, constant_values=True)
    regions = _components(free, _FOUR)
    answer = []
    h, w = free.shape
    for cells in regions:
        if np.any((cells[:, 0] == 0) | (cells[:, 0] == h - 1)
                  | (cells[:, 1] == 0) | (cells[:, 1] == w - 1)):
            continue
        lo, hi = cells.min(axis=0) - 1, cells.max(axis=0) - 1
        assert len(cells) == int(np.prod(hi - lo + 1)), "nonrectangular room"
        answer.append(tuple(map(int, (*lo, *hi))))
    return sorted(answer)


def _walls(shape, boxes):
    a = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        a[t, l:r + 1] = a[b, l:r + 1] = True
        a[t:b + 1, l] = a[t:b + 1, r] = True
    return a


def _euler_rooms(mask):
    """Cycle rank minus filled unit faces, for a connected pixel-wall graph."""
    edges = np.count_nonzero(mask[:-1] & mask[1:])
    edges += np.count_nonzero(mask[:, :-1] & mask[:, 1:])
    solid_squares = np.count_nonzero(mask[:-1, :-1] & mask[1:, :-1]
                                    & mask[:-1, 1:] & mask[1:, 1:])
    return int(edges - mask.sum() + 1 - solid_squares)


def _lattice_count(body):
    hs = np.flatnonzero(np.any(body[:, :-1] & body[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(body[:-1] & body[1:], axis=0))
    return (len(hs) - 1) * (len(vs) - 1)


def _canonical(mask):
    """Find the unique centered terminal pair; canonical neck faces down."""
    candidates = []
    for turns in range(4):
        a = np.rot90(mask, turns).copy()
        h, w = a.shape
        if w < 4 or w % 2 or h < 4:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        stem = 0
        for row in a[::-1]:
            if not np.array_equal(row, neck):
                break
            stem += 1
        if not stem or h - stem < 3:
            continue
        body = a[:-stem]
        if not (body[0].all() and body[-1].all()
                and body[:, 0].all() and body[:, -1].all()):
            continue
        try:
            rooms = _rooms(body)
        except AssertionError:
            continue
        boxes = [(t - 1, l - 1, b + 1, r + 1) for t, l, b, r in rooms]
        if not rooms or not np.array_equal(body, _walls(body.shape, boxes)):
            continue
        candidates.append(dict(mask=a, body=body, stem=stem, turns=turns,
                               count=len(rooms), rooms=rooms,
                               partial=_lattice_count(body) != len(rooms)))
    assert len(candidates) == 1, "missing or competing geometric neck"
    result = candidates[0]
    assert len(_components(result['mask'], _FOUR)) == 1
    assert _euler_rooms(result['mask']) == result['count']
    return result


def _draw_hub(a, r, c, labels):
    a[r, c + 1:c + 3] = labels[0]
    a[r + 1:r + 3, c + 3] = labels[1]
    a[r + 3, c + 1:c + 3] = labels[2]
    a[r + 1:r + 3, c] = labels[3]


def _find_hub(a, bg):
    allowed = np.zeros((4, 4), dtype=bool)
    allowed[0, 1:3] = allowed[3, 1:3] = True
    allowed[1:3, 0] = allowed[1:3, 3] = True
    hubs = []
    for r in range(a.shape[0] - 3):
        for c in range(a.shape[1] - 3):
            b = a[r:r + 4, c:c + 4]
            if np.any(b[~allowed] != bg):
                continue
            pairs = (b[0, 1:3], b[1:3, 3], b[3, 1:3], b[1:3, 0])
            if any(p[0] == bg or p[0] != p[1] for p in pairs):
                continue
            labels = tuple(int(p[0]) for p in pairs)
            if len(set(labels)) == 4:
                hubs.append((r, c, labels))
    assert len(hubs) == 1, "one unambiguous four-color hollow hub required"
    return hubs[0]


def _dock(mask, side, r, c):
    marker = np.zeros_like(mask)
    w = mask.shape[1]
    marker[-1, w // 2 - 1:w // 2 + 1] = True
    a = np.rot90(mask, -side)
    neck = np.argwhere(np.rot90(marker, -side))
    targets = (((r - 1, c + 1), (r - 1, c + 2)),
               ((r + 1, c + 4), (r + 2, c + 4)),
               ((r + 4, c + 1), (r + 4, c + 2)),
               ((r + 1, c - 1), (r + 2, c - 1)))[side]
    shift = np.min(targets, axis=0) - neck.min(axis=0)
    return np.argwhere(a) + shift


def _render(shape, bg, hub, labels, winners):
    r, c = hub
    out = np.full(shape, bg, dtype=np.int8)
    _draw_hub(out, r, c, labels)
    for p in winners:
        cells = _dock(p['mask'], p['side'], r, c)
        rr, cc = cells.T
        assert np.all((rr >= 0) & (rr < shape[0]))
        assert np.all((cc >= 0) & (cc < shape[1]))
        assert np.all(out[rr, cc] == bg), "winning keys overlap"
        out[rr, cc] = p['color']
    return out


def _inspect(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    assert frequencies[bg] > a.size // 2, "background must be a strict majority"
    r, c, labels = _find_hub(a, bg)
    extra = set(map(int, np.unique(a))) - {bg} - set(labels)
    assert len(extra) == 1, "one locally visible repair color required"
    noise = extra.pop()
    outside = a != bg
    outside[r:r + 4, c:c + 4] = False
    groups = defaultdict(list)
    specks = 0
    for cells in _components(outside):
        rr, cc = cells.T
        colors = set(map(int, np.unique(a[rr, cc]))) - {noise}
        if not colors:
            specks += len(cells)
            continue
        assert len(colors) == 1 and colors <= set(labels)
        color = colors.pop()
        # A full empty-cell moat separates objects and the hub in the input.
        assert not np.any((rr >= r - 1) & (rr <= r + 4)
                          & (cc >= c - 1) & (cc <= c + 4))
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        mask = np.zeros(hi - lo + 1, dtype=bool)
        mask[rr - lo[0], cc - lo[1]] = True
        p = _canonical(mask)
        p.update(color=color, side=labels.index(color), coords=cells,
                 damage=int(np.count_nonzero(a[rr, cc] == noise)),
                 rotation=(p['turns'] - labels.index(color)) % 4)
        groups[color].append(p)
    assert groups and specks, "include keys and isolated repair-color evidence"
    winners = []
    for group in groups.values():
        minimum = min(p['count'] for p in group)
        selected = [p for p in group if p['count'] == minimum]
        assert len(selected) == 1, "equal minimum room counts are outside the family"
        winners.append(selected[0])
    expected = _render(a.shape, bg, (r, c), labels, winners)
    return expected, dict(background=bg, noise=noise, hub=(r, c), labels=labels,
                          groups=dict(groups), winners=winners, specks=specks)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive selection, wall repair, rotation, and placement from input pixels."""
    return _inspect(input_grid)[0]


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _partition(rng, h, w, count, partial=False):
    outer = (0, 0, h - 1, w - 1)
    assert _capacity(outer) >= count
    for _ in range(80):
        boxes = [outer]
        first_axis = None
        for step in range(count - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for axis, low, high in ((0, t, b), (1, l, r)):
                    if partial and step == 1 and axis == first_axis:
                        continue
                    for cut in range(low + 2, high - 1):
                        halves = (((t, l, cut, r), (cut, l, b, r)) if axis == 0
                                  else ((t, l, b, cut), (t, cut, b, r)))
                        capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                        if capacity + sum(map(_capacity, halves)) >= count:
                            choices.append((i, axis, halves))
            if not choices:
                break
            i, axis, halves = choices[int(rng.integers(len(choices)))]
            if first_axis is None:
                first_axis = axis
            boxes[i:i + 1] = halves
        if len(boxes) != count:
            continue
        body = _walls((h, w), boxes)
        if partial and count >= 3 and _lattice_count(body) == count:
            continue
        assert len(_rooms(body)) == count == _euler_rooms(body)
        return body
    # Ordinary recursive cuts can always realize capacity; partial geometry is
    # required only at the spacious three/four-room witness.
    raise AssertionError('bounded partition construction exhausted')


def _make_key(rng, count, dimensions=None, partial=False):
    if dimensions is None:
        options = [(h, w) for h in range(3, 8) for w in (4, 6, 8)
                   if _capacity((0, 0, h - 1, w - 1)) >= count]
        h, w = options[int(rng.integers(len(options)))]
        stem = 2 if w == 8 else int(rng.integers(1, 3))
    else:
        h, w, stem = dimensions
    body = _partition(rng, h, w, count, partial)
    mask = np.zeros((h + stem, w), dtype=bool)
    mask[:h] = body
    mask[h:, w // 2 - 1:w // 2 + 1] = True
    p = _canonical(mask)
    assert p['count'] == count and p['turns'] == 0
    return p


def _scene_plan(rng, sides=None, witness=None, test=False, deltas=None):
    if sides is None:
        sides = list(map(int, rng.choice(4, int(rng.integers(1, 5)), replace=False)))
    side_order = list(map(int, rng.permutation(sides)))
    competition = side_order[0]
    second_competition = (side_order[1] if len(sides) >= 2 and len(sides) + 3 <= 6
                          and witness is None and rng.random() < .30 else None)
    pieces = []
    for side in sides:
        if side == competition:
            if witness == 'area':
                low = int(rng.integers(3, 5))
                counts = [low, int(rng.integers(low + 1, 7))]
            elif witness == 'outline':
                counts = [int(rng.integers(1, 3)), int(rng.integers(3, 7))]
            else:
                counts = list(map(int, rng.choice(np.arange(1, 8 if test else 7),
                                                 3 if test else 2, replace=False)))
        elif side == second_competition:
            counts = list(map(int, rng.choice(np.arange(1, 7), 2, replace=False)))
        else:
            counts = [int(rng.integers(1, 7))]
        winner_count = min(counts)
        common_stem = int(rng.integers(1, 3))
        for count in counts:
            selected = count == winner_count
            dimensions = None
            if side == competition and witness == 'area':
                dimensions = (7, 8, 2) if selected else (5 if count <= 4 else 7, 6, common_stem)
            elif side == competition and witness == 'outline':
                dimensions = (5 if max(counts) <= 4 else 7, 6, common_stem)
            # Other recursive partitions freely include partial or complete
            # dividers. Only this spacious witness must be a partial partition.
            partial = side == competition and witness == 'area' and selected
            p = _make_key(rng, count, dimensions, partial=partial)
            delta = deltas[side] if selected and deltas is not None else int(rng.integers(4))
            p.update(side=side, selected=selected, input_turns=(-side - delta) % 4,
                     force_damage=selected and witness is not None,
                     witness=witness if side == competition else None)
            pieces.append(p)
    return pieces


def _spatial_witness(pieces, locations):
    """Opposite clear position orders defeat first/last spatial selection."""
    group = [p for p in pieces if p['witness'] is not None]
    if not group:
        return True
    winner = next(p for p in group if p['selected'])
    loser = next(p for p in group if not p['selected'])
    wi = next(i for i, p in enumerate(pieces) if p is winner)
    li = next(i for i, p in enumerate(pieces) if p is loser)
    wlo, llo = np.asarray(locations[wi]), np.asarray(locations[li])
    whi = wlo + np.rot90(winner['mask'], winner['input_turns']).shape - np.ones(2, dtype=int)
    lhi = llo + np.rot90(loser['mask'], loser['input_turns']).shape - np.ones(2, dtype=int)
    return bool(np.all(wlo > lhi) if winner['witness'] == 'area' else np.all(llo > whi))


def _random_pack(rng, shape, hub, labels, bg, pieces):
    a = np.full(shape, bg, dtype=np.int8)
    r, c = hub
    _draw_hub(a, r, c, labels)
    blocked = np.zeros(shape, dtype=bool)
    blocked[max(0, r - 1):r + 5, max(0, c - 1):c + 5] = True
    order = list(map(int, rng.permutation(len(pieces))))
    order.sort(key=lambda i: -pieces[i]['mask'].size)
    locations = {}
    for i in order:
        p = pieces[i]
        oriented = np.rot90(p['mask'], p['input_turns'])
        h, w = oriented.shape
        if h > shape[0] or w > shape[1]:
            return None
        integral = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
        good = np.argwhere(sums == 0)
        if not len(good):
            return None
        top, left = map(int, good[int(rng.integers(len(good)))])
        rr, cc = np.nonzero(oriented)
        a[rr + top, cc + left] = p['color']
        blocked[max(0, top - 1):top + h + 1, max(0, left - 1):left + w + 1] = True
        locations[i] = (top, left)
    if not _spatial_witness(pieces, locations):
        return None
    return a, blocked, locations


def _fallback_pack(rng, pieces, labels, bg):
    # At most six keys; every rotated bounding box fits a 9x9 slot. The
    # center remains free for the hub, and all possible assemblies fit here.
    shape, hub = (30, 30), (13, 13)
    a = np.full(shape, bg, dtype=np.int8)
    _draw_hub(a, *hub, labels)
    blocked = np.zeros(shape, dtype=bool)
    blocked[12:18, 12:18] = True
    slots = [(r, c) for r in (0, 21) for c in (0, 10, 20)]
    rng.shuffle(slots)
    assigned = {}
    special = [i for i, p in enumerate(pieces) if p['witness'] is not None]
    if special:
        wi = next(i for i in special if pieces[i]['selected'])
        li = next(i for i in special if not pieces[i]['selected'])
        assigned[wi], assigned[li] = (((21, 20), (0, 0)) if pieces[wi]['witness'] == 'area'
                                    else ((0, 0), (21, 20)))
        slots = [slot for slot in slots if slot not in assigned.values()]
    locations = {}
    for i, p in enumerate(pieces):
        oriented = np.rot90(p['mask'], p['input_turns'])
        h, w = oriented.shape
        r, c = assigned[i] if i in assigned else slots.pop()
        assert h <= 9 and w <= 9
        rr, cc = np.nonzero(oriented)
        a[rr + r, cc + c] = p['color']
        blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        locations[i] = (r, c)
    assert _spatial_witness(pieces, locations)
    return a, blocked, locations, hub


def _make_pair(rng, plan, role_policy):
    if role_policy is None:
        bg, noise = map(int, rng.choice(10, 2, replace=False))
    else:
        bg, noise = role_policy
    labels = tuple(map(int, rng.choice([x for x in range(10) if x not in (bg, noise)],
                                     4, replace=False)))
    pieces = [dict(p, color=labels[p['side']]) for p in plan]
    winners = [p for p in pieces if p['selected']]
    relative = [np.array(((0, 0), (3, 3)))]
    relative.extend(_dock(p['mask'], p['side'], 0, 0) for p in winners)
    cells = np.concatenate(relative)
    low, high = cells.min(axis=0), cells.max(axis=0)
    minimum = high - low + 1
    base = 19 if len(pieces) <= 3 else 24
    shape = tuple(max(int(x), int(y)) for x, y in
                  zip(rng.integers(base, 31, 2), minimum))
    packed = None
    for attempt in range(12):
        lower = -low
        upper = np.asarray(shape) - 1 - high
        if np.any(lower > upper):
            raise AssertionError('assembly exceeds canvas')
        hub = tuple(int(rng.integers(a, b + 1)) for a, b in zip(lower, upper))
        packed = _random_pack(rng, shape, hub, labels, bg, pieces)
        if packed is not None and np.count_nonzero(packed[0] == bg) <= packed[0].size // 2 + 8:
            packed = None
        if packed is not None:
            break
        shape = tuple(min(30, x + (1 if attempt % 2 else 2)) for x in shape)
    if packed is None:
        a, blocked, locations, hub = _fallback_pack(rng, pieces, labels, bg)
    else:
        a, blocked, locations = packed
    out = _render(a.shape, bg, hub, labels, winners)
    # Repair marks replace actual occupied walls, never unknowable background
    # gaps. The external neck is left in its structural color.
    for i, p in enumerate(pieces):
        if p['force_damage'] or rng.random() < .65:
            eligible = np.zeros_like(p['mask'])
            eligible[:p['body'].shape[0]] = p['body']
            coords = np.argwhere(np.rot90(eligible, p['input_turns']))
            amount = min(len(coords) // 5, int(rng.integers(1, 5)))
            positions = coords[rng.choice(len(coords), amount, replace=False)]
            positions += np.asarray(locations[i])
            a[positions[:, 0], positions[:, 1]] = noise
    free = np.argwhere(~blocked)
    assert len(free) >= 2
    specks = free[rng.choice(len(free), int(rng.integers(2, min(8, len(free)) + 1)), replace=False)]
    a[specks[:, 0], specks[:, 1]] = noise
    assert np.count_nonzero(a == bg) > a.size // 2
    assert np.array_equal(solve(a), out), 'input reconstruction differs from construction'
    return {'input': a, 'output': out}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 2, replace=False))) if rng.random() < .35 else None
    directions = list(map(int, rng.permutation(4)))
    turns = list(map(int, rng.permutation(4)))
    plans = [_scene_plan(rng, directions[:2], 'area',
                         deltas=dict(zip(directions[:2], turns[:2]))),
             _scene_plan(rng, directions[2:], 'outline',
                         deltas=dict(zip(directions[2:], turns[2:])))]
    plans.extend(_scene_plan(rng) for _ in range(num_train - 2))
    rng.shuffle(plans)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, configurations in (('train', plans),
                                  ('test', [_scene_plan(rng, test=True) for _ in range(num_test)])):
        for plan in configurations:
            for _ in range(12):
                pair = _make_pair(rng, plan, roles)
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('bounded distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    expected, info = _inspect(pair['input'])
    assert np.array_equal(expected, np.asarray(pair['output']))
    groups, winners = info['groups'], info['winners']
    keys = [p for group in groups.values() for p in group]
    competitions = [g for g in groups.values() if len(g) > 1]
    info.update(keys=keys, rotations={p['rotation'] for p in winners},
                sides={p['side'] for p in winners},
                competition=bool(competitions),
                partial_winner=any(p['partial'] for p in winners),
                damage_winner=any(p['damage'] for p in winners),
                outline_witness=any(len({_key(p['mask']) for p in g}) != len(g)
                                    for g in competitions),
                area_witness=any(w['body'].size > q['body'].size
                                 for g in competitions for w in g for q in g
                                 if w['count'] == min(p['count'] for p in g)
                                 and w['count'] < q['count']))
    # Compare EXTERNAL silhouettes, retaining the neck but filling the body.
    for g in competitions:
        outlines = []
        for p in g:
            outline = p['mask'].copy()
            outline[:p['body'].shape[0]] = True
            outlines.append(_key(outline))
        if len(set(outlines)) < len(outlines):
            info['outline_witness'] = True
    info['winner_before'] = info['winner_after'] = False
    for g in competitions:
        winner = min(g, key=lambda p: p['count'])
        for loser in g:
            if loser is winner:
                continue
            info['winner_before'] |= bool(np.all(winner['coords'].max(axis=0) < loser['coords'].min(axis=0)))
            info['winner_after'] |= bool(np.all(winner['coords'].min(axis=0) > loser['coords'].max(axis=0)))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and not np.array_equal(a, out)
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _features(pair)
            bg = info['background']
            assert set(map(int, np.unique(out))) <= {bg, *info['labels']}
            for p in info['keys']:
                assert len(_rooms(p['body'])) == p['count'] == _euler_rooms(p['mask'])
            # The hub is unchanged; every winning key contributes its exact
            # repaired pixel count, without clipping, scaling, or overlaps.
            assert np.count_nonzero(out != bg) == 8 + sum(int(p['mask'].sum()) for p in info['winners'])
            for color, g in info['groups'].items():
                w = min(g, key=lambda p: p['count'])
                assert sum(p['count'] == w['count'] for p in g) == 1
                assert np.count_nonzero(out == color) == 2 + int(w['mask'].sum())
            assert max(map(len, info['groups'].values())) <= (2 if split == 'train' else 3)
            if split == 'test':
                assert any(len(g) == 3 for g in info['groups'].values())
            analyses[split].append(info)
    train = analyses['train']
    assert set().union(*(a['sides'] for a in train)) == set(range(4))
    assert set().union(*(a['rotations'] for a in train)) == set(range(4))
    assert all(any(a[k] for a in train) for k in
               ('competition', 'partial_winner', 'damage_winner', 'outline_witness', 'area_witness',
                'winner_before', 'winner_after'))
    minima = {min(p['count'] for p in g) for a in train for g in a['groups'].values() if len(g) > 1}
    assert len(minima) >= 2, 'minimum selection must not mean select a fixed room count'
    # Explicit ablations: retaining every same-color candidate changes the
    # chosen shape set; keeping chosen shapes at source positions changes pixels.
    area_ablation_changes = False
    for pair, a in zip(puzzle['train'], train):
        smallest = [min(g, key=lambda p: p['body'].size) for g in a['groups'].values()]
        def assembled_cells(parts):
            return {(int(r), int(c), p['color']) for p in parts
                    for r, c in _dock(p['mask'], p['side'], *a['hub'])}
        # Coordinate sets also expose a wrong choice that would leave the
        # canvas; such a choice cannot silently become a clipped answer.
        area_ablation_changes |= assembled_cells(smallest) != assembled_cells(a['winners'])
        stationary = np.full_like(np.asarray(pair['input']), a['background'])
        _draw_hub(stationary, *a['hub'], a['labels'])
        for p in a['winners']:
            stationary[p['coords'][:, 0], p['coords'][:, 1]] = p['color']
        assert not np.array_equal(stationary, pair['output']), 'docking ablation has no effect'
    assert area_ablation_changes, 'omitting compartment counting does not change an answer'


def _topology_signature(p):
    # Rank room-boundary coordinates, removing metric spacing but keeping
    # partial-divider endpoints and the entire room arrangement.
    rooms = [(t - 1, l - 1, b + 1, r + 1) for t, l, b, r in p['rooms']]
    ys = sorted({x for t, _, b, _ in rooms for x in (t, b)})
    xs = sorted({x for _, l, _, r in rooms for x in (l, r)})
    return tuple(sorted((ys.index(t), xs.index(l), ys.index(b), xs.index(r))
                        for t, l, b, r in rooms))


def self_test():
    """Bounded 42-seed audit, metamorphic checks, and count/error regressions."""
    signatures, shapes, masks, topologies, counts = set(), set(), set(), set(), set()
    occupied, multiplicities, backgrounds, repair_colors, widths, stems = (set() for _ in range(6))
    total_pairs = total_keys = partial = area = outline = edge = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                total_pairs += 1
                total_keys += len(f['keys'])
                shapes.add(np.asarray(pair['input']).shape)
                occupied.add(tuple(sorted(f['sides'])))
                multiplicities.update(map(len, f['groups'].values()))
                backgrounds.add(f['background'])
                repair_colors.add(f['noise'])
                area += f['area_witness']
                outline += f['outline_witness']
                for p in f['keys']:
                    masks.add(_key(p['mask']))
                    topologies.add(_topology_signature(p))
                    counts.add(p['count'])
                    widths.add(p['body'].shape[1])
                    stems.add(p['stem'])
                    partial += p['partial']
                cells = np.argwhere(pair['output'] != f['background'])
                edge += bool(np.any((cells == 0) | (cells == np.asarray(pair['output'].shape) - 1)))
        if seed < 3:
            # The entire spatial rule commutes with rotations and reflections;
            # literal role identities are irrelevant.
            permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
            for p in puzzle['train'] + puzzle['test']:
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(p['input'], k)), np.rot90(p['output'], k))
                assert np.array_equal(solve(np.fliplr(p['input'])), np.fliplr(p['output']))
                assert np.array_equal(solve(permutation[p['input']]), permutation[p['output']])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(masks) >= 80 and len(topologies) >= 20
    assert counts == set(range(1, 8)) and widths == {4, 6, 8} and stems == {1, 2}
    assert multiplicities == {1, 2, 3} and len(occupied) >= 12
    assert backgrounds == repair_colors == set(range(10))
    assert partial >= 42 and area >= 42 and outline >= 42
    again = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1900 + 10 * a + b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (None, 1), (4, None), (True, 1), (4.0, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, pairs=total_pairs, keys=total_keys, shapes=len(shapes),
                canonical_key_masks=len(masks), scale_free_topologies=len(topologies),
                partial_keys=int(partial), area_witnesses=int(area),
                equal_outline_witnesses=int(outline), edge_assemblies=int(edge))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
