"""Rank-paired reservoirs: fill hollow slots before extending their stacks.

Only NumPy and the standard library are used. All semantic information is
visible: square occupancy gives the role, bars select stacks and give both
the processing color and a local stacking pitch.
"""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _tile(size, color, background, kind):
    a = np.full((size, size), color, dtype=np.int8)
    if kind == "H":
        a[1:-1, 1:-1] = background
    return a


def _parse(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg), "The first row identifies the background"
    lanes = _runs(np.any(a != bg, axis=0))
    assert lanes
    stacks, sizes, controls = [], set(), set()
    for x, end in lanes:
        s = end - x
        assert 3 <= s <= 7
        sizes.add(s)
        strip = a[:, x:end]
        groups = _runs(np.any(strip != bg, axis=1))
        marker, highlight = None, None
        if groups[-1][1] - groups[-1][0] == 1:
            marker = groups.pop()[0]
            highlight = int(strip[marker, 0])
            assert np.all(strip[marker] == highlight) and highlight != bg
            controls.add(highlight)
        assert groups
        color = int(strip[groups[0][0], 0])
        assert color != bg
        kinds = []
        for top, stop in groups:
            assert stop - top == s
            block = strip[top:stop]
            if np.array_equal(block, _tile(s, color, bg, "H")):
                kinds.append("H")
            else:
                assert np.array_equal(block, _tile(s, color, bg, "S"))
                kinds.append("S")
        assert len(set(kinds)) == 1, "Input stacks have one geometry"
        starts = [r for r, _ in groups]
        observed = {b - r for r, b in zip(starts, starts[1:])}
        assert len(observed) <= 1
        bottom = starts[-1]
        if marker is not None:
            pitch = marker - bottom
            assert pitch > s and observed <= {pitch}
            assert np.all(strip[bottom+s:marker] == bg)
        else:
            pitch = next(iter(observed), None)
            assert pitch is None or pitch > s
        stacks.append(dict(x=x, size=s, color=color, kind=kinds[0],
                           top=starts[0], bottom=bottom, starts=starts,
                           count=len(starts), pitch=pitch, marker=marker))
    assert len(sizes) == 1 and len(controls) == 1
    size, highlight = sizes.pop(), controls.pop()
    marked = [t for t in stacks if t['marker'] is not None]
    queues = [[t for t in marked if t['kind'] == kind] for kind in ("H", "S")]
    for q in queues:
        assert q
        assert len({t['top'] for t in q}) == len(q), "Same-class top ties forbidden"
        q.sort(key=lambda t: t['top'])
    for src in queues[1]:
        assert src['color'] != highlight, "Source highlighting must be visible"
    links = list(zip(*queues))
    for dst, src in links:
        assert dst['bottom'] - (max(dst['count'], src['count'])-1)*dst['pitch'] >= 0
    return dict(background=bg, highlight=highlight, size=size, stacks=stacks,
                queues=queues, links=links)


def _render(value, parsed, links=None, amount="count", placement="bottom"):
    """Apply the rule, or a named simple rival used to check demonstrations."""
    out = np.asarray(value).copy()
    bg, hi, s = parsed['background'], parsed['highlight'], parsed['size']
    for t in parsed['stacks']:
        if t['marker'] is not None:
            out[t['marker'], t['x']:t['x']+s] = bg
    for dst, src in parsed['links'] if links is None else links:
        h, q = dst['count'], src['count']
        if amount == "unit":
            q = 1
        elif amount == "capacity":
            q = h
        elif amount == "pixels":
            q *= s*s
        elif amount == "total":
            q = sum(t['count'] for t in parsed['queues'][1])
        length = max(h, q) if placement != "append" else h+q
        if dst['bottom'] - (length-1)*dst['pitch'] < 0:
            return None
        for r in src['starts']:
            out[r:r+s, src['x']:src['x']+s] = hi
        for k in range(length):
            if placement == "top":
                filled = k >= h-min(h, q) if k < h else True
            elif placement == "append":
                filled = k >= h
            else:
                filled = k < q
            r = dst['bottom'] - k*dst['pitch']
            color = src['color'] if placement == 'source_color' else dst['color']
            out[r:r+s, dst['x']:dst['x']+s] = _tile(
                s, color, bg, "S" if filled else "H")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, ranks, quantities, colors, and pitches from this input."""
    parsed = _parse(input_grid)
    return _render(input_grid, parsed)


def _alternative_links(a, mode):
    hollow, solid = a['queues']
    if mode == "greedy":
        rows = [{}]
        for t in sorted(hollow+solid, key=lambda t: (t['top'], t['x'])):
            if t['kind'] in rows[-1]:
                rows.append({})
            rows[-1][t['kind']] = t
        return [(row['H'], row['S']) for row in rows if set(row) == {'H', 'S'}]
    if mode == "nearest":
        todo, result = list(solid), []
        for dst in hollow:
            if not todo:
                break
            src = min(todo, key=lambda t: (abs(t['x']-dst['x']) +
                                          abs(t['top']-dst['top']), t['top']))
            result.append((dst, src))
            todo.remove(src)
        return result
    if mode == "all_stacks":
        # Give an unmarked singleton the minimal positive-gap pitch for this
        # rival, so absent controls do not reject it merely through None.
        hollow, solid = [[dict(t, pitch=t['pitch'] or a['size']+1)
                          for t in a['stacks'] if t['kind'] == kind]
                         for kind in ('H', 'S')]
    key = {
        'horizontal': lambda t: t['x'],
        'horizontal_reverse': lambda t: -t['x'],
        'reverse': lambda t: -t['top'],
        'bottom': lambda t: t['bottom'],
        'bar': lambda t: t['marker'],
        'color': lambda t: (t['color'], t['top']),
        'all_stacks': lambda t: t['top'],
    }[mode]
    return list(zip(sorted(hollow, key=key), sorted(solid, key=key)))


_PAIR_RIVALS = ('horizontal', 'horizontal_reverse', 'reverse', 'bottom', 'bar', 'color',
                'nearest', 'greedy', 'all_stacks')
_COUNT_RIVALS = (('unit', 'bottom'), ('capacity', 'bottom'),
                 ('pixels', 'bottom'), ('total', 'bottom'), ('count', 'top'),
                 ('count', 'append'), ('count', 'source_color'))


def _different(expected, actual):
    return actual is None or not np.array_equal(expected, actual)


def _rival_answer(pair, a, rival):
    try:
        if isinstance(rival, str):
            return _render(pair['input'], a, _alternative_links(a, rival))
        return _render(pair['input'], a, amount=rival[0], placement=rival[1])
    except (TypeError, IndexError, AssertionError):
        return None


def _evidence(pairs, analyses):
    operands = [(d['count'], s['count']) for a in analyses for d, s in a['links']]
    assert any(q < h for h, q in operands), 'Missing partial-fill demonstration'
    assert any(q > h for h, q in operands), 'Missing overflow demonstration'
    assert any(d['count'] == 1 and s['count'] > 1 and
               d['pitch'] > a['size']+1 for a in analyses for d, s in a['links']), \
        'Missing singleton recipient with visibly encoded nonminimal pitch'
    assert len({q for _, q in operands}) >= 2, 'Quantities must vary'
    assert len({h for h, _ in operands}) >= 2, 'Capacities must vary'
    assert any(len(a['queues'][0]) > len(a['queues'][1]) for a in analyses)
    assert any(len(a['queues'][0]) < len(a['queues'][1]) for a in analyses)
    assert any(t['marker'] is None for a in analyses for t in a['stacks'])
    roles = {}
    for a in analyses:
        for t in a['stacks']:
            roles.setdefault(t['color'], set()).add(t['kind'])
    assert any(len(kinds) == 2 for kinds in roles.values()), 'Geometry, not color, gives role'
    for rival in _PAIR_RIVALS + _COUNT_RIVALS:
        assert any(_different(p['output'], _rival_answer(p, a, rival))
                   for p, a in zip(pairs, analyses)), f'Training fits rival {rival}'


def _integer(name, value, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < low or
            (high is not None and value > high)):
        raise ValueError(f'Illegal {name}')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _build(rng, size, counts, source_limit=3, anchor=False,
           forced_role=None, demonstration=None, large=False, ruler_witness=False):
    nh, ns, nu = counts
    n = nh+ns+nu
    assert (size+1)*n-1 <= 30
    cap = 25//(size+1)
    source_limit = min(source_limit, cap)
    kinds = ['H']*nh + ['S']*ns + [str(rng.choice(['H', 'S'])) for _ in range(nu)]
    marked = [True]*(nh+ns) + [False]*nu
    heights = []
    for i, kind in enumerate(kinds):
        if i < nh:
            limit = min(4, cap) if not large else min(3, cap)
            heights.append(int(rng.integers(1, limit+1)))
        elif i < nh+ns:
            low = 4 if large else 1
            heights.append(int(rng.integers(low, source_limit+1)))
        else:
            heights.append(int(rng.integers(1, min(4, cap)+1)))
    if anchor and ns >= 2:
        heights[nh:nh+ns] = rng.choice(np.arange(1, source_limit+1), ns,
                                      replace=False).tolist()
    singleton_index = int(rng.integers(nh)) if ruler_witness else None
    if singleton_index is not None:
        heights[singleton_index] = 1
    max_source = max(heights[nh:nh+ns])
    required = [max(h, max_source) if kinds[i] == 'H' and marked[i] else h
                for i, h in enumerate(heights)]
    pitches = [int(rng.integers(size+1, min(size+3, 25//f)+1)) for f in required]
    if singleton_index is not None:
        f = required[singleton_index]
        pitches[singleton_index] = int(rng.integers(size+2, min(size+3, 25//f)+1))
    # Each marked class has at least its own count of possible distinct tops.
    minimum_h = max(f*p for f, p in zip(required, pitches)) + max(nh, ns)+1
    assert minimum_h <= 30
    height = int(rng.integers(minimum_h, 31))
    used = {'H': set(), 'S': set()}
    bottoms = []
    for i, (h, f, pitch) in enumerate(zip(heights, required, pitches)):
        choices = list(range(1+(f-1)*pitch, height-pitch))
        if marked[i]:
            choices = [b for b in choices if b-(h-1)*pitch not in used[kinds[i]]]
        assert choices
        bottom = int(rng.choice(choices))
        bottoms.append(bottom)
        if marked[i]:
            used[kinds[i]].add(bottom-(h-1)*pitch)
    # Random horizontal permutation is independent of both vertical queues.
    order = rng.permutation(n).tolist()
    min_w = n*(size+1)-1
    width = int(rng.integers(min_w, 31))
    extra = rng.multinomial(width-min_w, np.full(n+1, 1/(n+1)))
    xs, x = [0]*n, int(extra[0])
    for rank, i in enumerate(order):
        xs[i] = x
        x += size+1+int(extra[rank+1])
    bg = int(rng.choice([c for c in range(10) if c != demonstration]))
    hi = int(rng.choice([c for c in range(10) if c not in (bg, demonstration)]))
    colors = []
    for kind, is_marked in zip(kinds, marked):
        pool = [c for c in range(10) if c != bg and
                (not is_marked or kind != 'S' or c != hi)]
        colors.append(int(rng.choice(pool)))
    if forced_role is not None:
        idx = next(i for i in range(nh+ns) if kinds[i] == forced_role)
        colors[idx] = demonstration
    # A hollow may use the control color without acquiring the source role.
    if anchor:
        colors[nh-1] = hi
        if nu:
            colors[nh+ns] = colors[nh]
    inp = np.full((height, width), bg, dtype=np.int8)
    for i in range(n):
        for k in range(heights[i]):
            r = bottoms[i] - k*pitches[i]
            inp[r:r+size, xs[i]:xs[i]+size] = _tile(size, colors[i], bg, kinds[i])
        if marked[i]:
            inp[bottoms[i]+pitches[i], xs[i]:xs[i]+size] = hi
    return dict(input=inp, output=solve(inp))


def _random_counts(rng, size, training=False):
    capacity = 31//(size+1)
    choices = [(h, s, u) for h in range(1, 5) for s in range(1, 5)
               for u in range(3) if h+s+u <= capacity and
               (not training or (min(h, s) <= 2 and abs(h-s) <= 1))]
    return choices[int(rng.integers(len(choices)))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    num_train = _integer('num_train', num_train, 2, 10)
    num_test = _integer('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    profile = int(rng.integers(4))  # arity, unmatched tail, amount, unseen scale
    holdout = int(rng.integers(5, 8)) if profile == 3 else None
    demonstration = int(rng.integers(10))
    sizes = rng.permutation([3, 4]).tolist()
    directions = rng.permutation([(2, 3), (3, 2)]).tolist()
    # Sample the two demonstrations together; evidence is tested on actual
    # pixels. No universal ordering/count template is used as ground truth.
    for _ in range(192):
        first = _build(rng, sizes[0], (*directions[0], 1), anchor=True,
                       forced_role='H', demonstration=demonstration,
                       ruler_witness=directions[0][0] == 2)
        second = _build(rng, sizes[1], (*directions[1], int(rng.integers(2))),
                        forced_role='S', demonstration=demonstration,
                        ruler_witness=directions[1][0] == 2)
        train = [first, second]
        try:
            _evidence(train, [_parse(p['input']) for p in train])
            break
        except AssertionError:
            continue
    else:
        raise RuntimeError('Bounded demonstration sampling exhausted')
    puzzle = dict(train=train, test=[])
    seen = {_key(p['input']) for p in train}
    for index in range(2, num_train+num_test):
        training = index < num_train
        for _ in range(64):
            large = False
            if training:
                size = int(rng.choice([s for s in range(3, 8) if s != holdout]))
                counts = _random_counts(rng, size, training=True)
                limit = 3
            elif index == num_train:
                if profile == 0:
                    size = int(rng.choice([3, 4]))
                    counts = (3, 3, int(size == 3 and rng.random() < 0.5))
                    limit = 25//(size+1)
                elif profile == 1:
                    size = int(rng.integers(3, 7))
                    longer = int(rng.integers(3, min(4, 31//(size+1)-1)+1))
                    counts = (1, longer, 0) if rng.integers(2) else (longer, 1, 0)
                    limit = 25//(size+1)
                elif profile == 2:
                    size = int(rng.integers(3, 6))
                    counts = _random_counts(rng, size)
                    limit, large = 25//(size+1), True
                else:
                    size = holdout
                    counts = _random_counts(rng, size)
                    limit = 25//(size+1)
            else:
                size = int(rng.integers(3, 8))
                counts = _random_counts(rng, size)
                limit = 25//(size+1)
            pair = _build(rng, size, counts, source_limit=limit, large=large)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['train' if training else 'test'].append(pair)
                break
        else:
            raise RuntimeError('Bounded unique-input sampling exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    a = _parse(inp)
    s, bg = a['size'], a['background']
    assert np.array_equal(out, _render(inp, a))
    changed_region = np.zeros(inp.shape, dtype=bool)
    matched = {t['x'] for link in a['links'] for t in link}
    for t in a['stacks']:
        if t['marker'] is not None:
            assert np.all(out[t['marker'], t['x']:t['x']+s] == bg)
            changed_region[t['marker'], t['x']:t['x']+s] = True
        if t['x'] not in matched:
            region = out[:, t['x']:t['x']+s].copy()
            original = inp[:, t['x']:t['x']+s].copy()
            if t['marker'] is not None:
                original[t['marker']] = bg
            assert np.array_equal(region, original), 'Unmatched/unmarked stack changed'
    for dst, src in a['links']:
        h, q, pitch = dst['count'], src['count'], dst['pitch']
        top = dst['bottom']-(max(h, q)-1)*pitch
        assert top >= 0
        filled = hollow = 0
        for k in range(max(h, q)):
            r = dst['bottom']-k*pitch
            block = out[r:r+s, dst['x']:dst['x']+s]
            filled += np.array_equal(block, _tile(s, dst['color'], bg, 'S'))
            hollow += np.array_equal(block, _tile(s, dst['color'], bg, 'H'))
            changed_region[r:r+s, dst['x']:dst['x']+s] = True
        assert filled == q and hollow == max(h-q, 0), 'Capacity/material conservation'
        assert max(h, q)-h == max(q-h, 0), 'Only surplus grows upward'
        for r in src['starts']:
            assert np.all(out[r:r+s, src['x']:src['x']+s] == a['highlight'])
            changed_region[r:r+s, src['x']:src['x']+s] = True
    assert np.array_equal(inp[~changed_region], out[~changed_region]), 'Background/gap changed'
    return a


def validate_generated_puzzle(puzzle):
    """Schema, pixel rule, conservation, inactive stacks, and teaching checks."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            key = _key(pair['input'])
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            if split == 'train':
                train.append(a)
    _evidence(puzzle['train'], train)
    tests = [_parse(p['input']) for p in puzzle['test']]
    # At least one visibly specified structural/quantity extrapolation.
    assert any(
        a['size'] not in {t['size'] for t in train} or
        len(a['links']) > max(len(t['links']) for t in train) or
        abs(len(a['queues'][0])-len(a['queues'][1])) > max(
            abs(len(t['queues'][0])-len(t['queues'][1])) for t in train) or
        max(s['count'] for _, s in a['links']) > max(
            s['count'] for t in train for _, s in t['links'])
        for a in tests), 'Missing train/test shift'


def self_test():
    """Bounded 42-seed coverage, all explicit counts, and negative controls."""
    shapes, sizes, layouts, quantities, palettes, profiles = set(), set(), set(), set(), set(), set()
    singleton, repeats, local_pitches, edge, equality = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert all(_key(p[k]) == _key(q[k]) for split in ('train', 'test')
                   for p, q in zip(puzzle[split], generate_puzzle(seed)[split])
                   for k in ('input', 'output'))
        train = [_parse(p['input']) for p in puzzle['train']]
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _check_pair(pair)
                shapes.add(pair['input'].shape)
                sizes.add(a['size'])
                layouts.add((pair['input'].shape, a['size'], tuple(
                    (t['x'], t['top'], t['bottom'], t['pitch'], t['kind'], t['marker'])
                    for t in a['stacks'])))
                palettes.add((a['background'], a['highlight']))
                quantities.update((d['count'], s['count']) for d, s in a['links'])
                singleton += all(t['count'] == 1 for t in a['stacks'])
                repeats += len({t['color'] for t in a['stacks']}) < len(a['stacks'])
                local_pitches += len({t['pitch'] for t in a['stacks'] if t['marker'] is not None}) > 1
                edge += any(t['x'] == 0 or t['x']+a['size'] == pair['input'].shape[1]
                            for t in a['stacks'])
                equality += any(d['count'] == s['count'] for d, s in a['links'])
                if split == 'test':
                    profiles.add((len(a['links']) > 2,
                                  abs(len(a['queues'][0])-len(a['queues'][1])) >= 2,
                                  max(s['count'] for _, s in a['links']) > 3,
                                  a['size'] not in {t['size'] for t in train}))
    assert sizes == set(range(3, 8))
    assert len(shapes) >= 35 and len(layouts) >= 180
    assert len(quantities) >= 15 and len(palettes) >= 25
    assert repeats >= 40 and local_pitches >= 50 and edge >= 40 and equality >= 20
    assert all(any(p[i] for p in profiles) for i in range(4))
    # A singleton recipient still gives its pitch via its bar. Its three
    # incoming units must create two new solid tiles at rows 6 and 1.
    singleton_destination = np.zeros((18, 7), dtype=np.int8)
    singleton_destination[11:14, :3] = _tile(3, 2, 0, 'H')
    for r in (4, 8, 12):
        singleton_destination[r:r+3, 4:7] = 3
    singleton_destination[16, :3] = 8
    singleton_destination[16, 4:7] = 8
    expected = singleton_destination.copy()
    expected[16] = 0
    for r in (1, 6, 11):
        expected[r:r+3, :3] = 2
    for r in (4, 8, 12):
        expected[r:r+3, 4:7] = 8
    assert np.array_equal(solve(singleton_destination), expected)
    _check_pair(dict(input=singleton_destination, output=expected))
    # Boundary growth may use row zero; the input still has its blank header.
    edge_input = singleton_destination.copy()
    edge_input[:, :3] = 0
    edge_input[8:11, :3] = _tile(3, 2, 0, 'H')
    edge_input[12, :3] = 8
    edge_output = solve(edge_input)
    assert np.all(edge_output[:3, :3] == 2)
    _check_pair(dict(input=edge_input, output=edge_output))
    # All-singleton scenes require no repetition elsewhere to be parsed.
    single_input = np.zeros((14, 7), dtype=np.int8)
    single_input[8:11, :3] = _tile(3, 2, 0, 'H')
    single_input[3:6, 4:7] = 3
    single_input[13, :3] = 8
    single_input[7, 4:7] = 8
    single_output = solve(single_input)
    assert np.all(single_output[8:11, :3] == 2)
    assert np.all(single_output[3:6, 4:7] == 8)
    _check_pair(dict(input=single_input, output=single_output))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1234+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (False, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    original = generate_puzzle(0, 2, 1)
    a = _parse(original['train'][0]['input'])
    dst, src = a['links'][0]
    unmarked = next(t for t in a['stacks'] if t['marker'] is None)
    for defect in ('fill', 'source', 'marker', 'inactive', 'gap'):
        broken = {split: [{k: np.asarray(g).copy() for k, g in p.items()}
                          for p in original[split]] for split in ('train', 'test')}
        out = broken['train'][0]['output']
        if defect == 'fill':
            r, c = dst['bottom']+1, dst['x']+1
        elif defect == 'source':
            r, c = src['bottom'], src['x']
        elif defect == 'marker':
            r, c = src['marker'], src['x']
        elif defect == 'inactive':
            r, c = unmarked['top'], unmarked['x']
        else:
            r, c = 0, 0
        out[r, c] = (int(out[r, c])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f'Corrupt {defect} accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds and all legal count combinations)')
