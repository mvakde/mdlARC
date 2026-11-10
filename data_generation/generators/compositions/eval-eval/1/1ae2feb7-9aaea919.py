"""Finite supplies competing on periodic clocks for one growing receiver.

Construction uses an event queue; solve parses pixels and scans clock ticks.
No parent generator, stored puzzle, learned convention, or external file is used.
"""
from __future__ import annotations

import heapq
import itertools
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    todo = set(zip(*np.nonzero(mask)))
    if not todo:
        return False
    fringe = [todo.pop()]
    while fringe:
        y, x = fringe.pop()
        for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if p in todo:
                todo.remove(p)
                fringe.append(p)
    return not todo


def _draw(a, y, x, mask, color):
    h, w = mask.shape
    _require(0 <= y and y+h <= a.shape[0] and 0 <= x and x+w <= a.shape[1],
             "complete tokens must fit")
    a[y:y+h, x:x+w][mask] = color


def _parse_lane(a, x, end, bg, control_y):
    strip = a[:, x:end]
    groups = _runs(np.any(strip != bg, axis=1))
    control = None
    if groups and groups[-1][0] == control_y:
        start, stop = groups.pop()
        _require(stop == start+1 and np.all(strip[start] == strip[start, 0]),
                 "control must be a complete horizontal bar")
        control = int(strip[start, 0])
    _require(bool(groups), "empty token stack")
    masks, colors = [], set()
    for start, stop in groups:
        block = strip[start:stop]
        mask = block != bg
        _require(_connected(mask) and np.all(np.any(mask, axis=0))
                 and np.all(np.any(mask, axis=1)), "connected full-span token")
        masks.append(mask)
        colors.update(map(int, np.unique(block[mask])))
    _require(len(colors) == 1 and all(np.array_equal(masks[0], m) for m in masks),
             "one repeated monochrome silhouette per stack")
    return dict(x=x, width=end-x, mask=masks[0], color=colors.pop(),
                ys=[s for s, _ in groups], count=len(groups),
                baseline=groups[-1][1], control=control)


def _canonical_parse(a, col, bg, ink):
    left = _runs(np.any(a[:, :col] != bg, axis=0))
    right = [(s+col+1, e+col+1) for s, e in
             _runs(np.any(a[:, col+1:] != bg, axis=0))]
    _require(bool(left) and len(right) == 1, "source bank and one receiver")
    x, end = right[0]
    _require(x >= col+2 and all(e < col for _, e in left), "divider gaps")
    groups = _runs(np.any(a[:, x:end] != bg, axis=1))
    _require(len(groups) >= 3, "receiver must show at least two tokens and control")
    cy, stop = groups[-1]
    _require(stop == cy+1 and np.all(a[cy, x:end] == ink), "receiver marker")
    receiver = _parse_lane(a, x, end, bg, cy)
    bank = [_parse_lane(a, s, e, bg, cy) for s, e in left]
    stacks = bank + [receiver]
    h = receiver['mask'].shape[0]
    _require(1 <= h <= 2 and all(s['mask'].shape[0] == h for s in stacks),
             "common token height of one or two")
    _require(receiver['count'] >= 2, "visible receiver pitch")
    differences = np.diff(receiver['ys'])
    pitch = int(differences[0])
    _require(h < pitch and np.all(differences == pitch), "regular receiver pitch")
    baseline = receiver['baseline']
    _require(cy > baseline and all(s['baseline'] == baseline for s in stacks),
             "shared baseline and a blank control gap")
    for s in stacks:
        _require(s['color'] not in (bg, ink), "payload excludes frame colors")
        expected = [baseline-h-k*pitch for k in reversed(range(s['count']))]
        _require(s['ys'] == expected, "common visible pitch")
        _require(1 <= s['width'] <= (6 if s is not receiver else 5), "token width")
        _require(np.all(a[baseline:cy, s['x']:s['x']+s['width']] == bg), "control gap")
        _require(np.all(a[cy+1:, s['x']:s['x']+s['width']] == bg), "bottom margin")
    sources = [s for s in reversed(bank) if s['control'] is not None]
    unmarked = [s for s in bank if s['control'] is None]
    _require(1 <= len(sources) <= 4 and len(unmarked) <= 2, "bank counts")
    for s in sources:
        _require(s['control'] == s['color'], "source bar matches its payload")
        s['period'] = s['width']
    _require(all(s['width'] <= 5 for s in unmarked), "unmarked width")
    # Frame ink may occur only on the divider and the destination control.
    allowed = np.zeros(a.shape, bool)
    allowed[1:-1, col] = True
    allowed[cy, receiver['x']:receiver['x']+receiver['width']] = True
    _require(np.array_equal(a == ink, allowed), "unique frame and receiver role")
    return dict(canonical=a, background=bg, ink=ink, divider=col,
                control_y=cy, baseline=baseline, pitch=pitch, height=h,
                sources=sources, unmarked=unmarked, receiver=receiver)


def _parse(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
             and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg),
             "blank outer border")
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        if a.shape[0] < 8:
            continue
        for col in range(2, a.shape[1]-2):
            ink = int(a[1, col])
            if ink == bg or not np.all(a[1:-1, col] == ink):
                continue
            try:
                parsed = _canonical_parse(a, col, bg, ink)
                parsed['turn'] = turn
                candidates.append(parsed)
            except ValueError:
                pass
    _require(len(candidates) == 1, "scene must have one geometric interpretation")
    return candidates[0]


def _ticks(periods, counts, *, priority='near', limit=None, infinite=False):
    """Cell solver's discrete-time simulation, with optional audited rivals."""
    remaining = list(counts)
    result = []
    t = 0
    while (infinite or any(remaining)) and (limit is None or t < limit):
        hits = [i for i, p in enumerate(periods)
                if (infinite or remaining[i] > 0) and t % p == 0]
        if hits:
            if priority == 'short':
                winner = min(hits, key=lambda i: (periods[i], i))
            elif priority == 'long':
                winner = min(hits, key=lambda i: (-periods[i], i))
            elif priority == 'far':
                winner = hits[-1]
            else:
                winner = hits[0]
            remaining[winner] -= 1
        else:
            winner = None
        result.append(winner)
        t += 1
    _require(limit is not None or not any(remaining), "unfinished stock")
    return result


def _answer(info, variant='correct'):
    a = info['canonical']
    out = a.copy()
    bg, ink, cy = info['background'], info['ink'], info['control_y']
    receiver, sources = info['receiver'], info['sources']
    for s in sources:
        for y in s['ys']:
            _draw(out, y, s['x'], s['mask'], ink)
        out[cy, s['x']:s['x']+s['width']] = bg
    out[cy, receiver['x']:receiver['x']+receiver['width']] = bg
    periods = [s['period'] for s in sources]
    counts = [s['count'] for s in sources]
    capacity = (receiver['ys'][0]-1)//info['pitch']
    if variant == 'no_clocks':
        periods = [1]*len(sources)
    elif variant == 'one_per_bar':
        counts = [1]*len(sources)
    elif variant == 'pixels':
        counts = [s['count']*int(s['mask'].sum()) for s in sources]
    priority = variant if variant in ('short', 'long', 'far') else 'near'
    timeline = _ticks(periods, counts, priority=priority, limit=capacity,
                      infinite=variant == 'infinite')
    if variant == 'packed':
        timeline = [i for i in timeline if i is not None]
    if variant == 'correct':
        _require(sum(i is not None for i in timeline) == sum(counts),
                 "all inventory must fit, without clipping")
    for t, i in enumerate(timeline):
        if i is not None:
            y = receiver['ys'][0]-(t+1)*info['pitch']
            _draw(out, y, receiver['x'], receiver['mask'], sources[i]['color'])
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, object count, clock and spacing from this input alone."""
    return _answer(_parse(input_grid))


def _events(periods, counts):
    """Construction's event queue: losers keep their absolute clock phase."""
    queue = [(0, i) for i in range(len(periods))]
    heapq.heapify(queue)
    remaining = list(counts)
    emissions = {}
    while queue:
        t = queue[0][0]
        firing = []
        while queue and queue[0][0] == t:
            _, i = heapq.heappop(queue)
            if remaining[i]:
                firing.append(i)
        if not firing:
            continue
        winner = min(firing)
        emissions[t] = winner
        remaining[winner] -= 1
        for i in firing:
            if remaining[i]:
                heapq.heappush(queue, (t+periods[i], i))
    return emissions


def _stock_effect(periods, counts, events):
    for t in range(max(events)+1):
        actual = events.get(t)
        unbounded = next((i for i, p in enumerate(periods) if t % p == 0), None)
        if actual != unbounded:
            return True
    return False


def _lesson_plan(rng):
    possibilities = []
    for a, b in itertools.combinations(range(2, 7), 2):
        options = []
        for periods in ((a, b), (b, a)):
            local = []
            for near in range(2, 5):
                for far in range(1, 5):
                    counts = (near, far)
                    ev = _events(periods, counts)
                    for dest in (2, 3):
                        # Minimal height: one-pixel tokens, pitch two, margins.
                        if 1+(dest+max(ev))*2+4 <= 30:
                            local.append((periods, counts, dest, ev))
            options.append(local)
        compatible = []
        for first in options[0]:
            for second in options[1]:
                s1, s2 = sum(first[1]), sum(second[1])
                d1, d2 = first[2], second[2]
                if s1 == s2 or d1 == d2 or s1*d2 == s2*d1:
                    continue
                if not any(_stock_effect(o[0], o[1], o[3]) for o in (first, second)):
                    continue
                compatible.append((first, second))
        if compatible:
            possibilities.append(compatible)
    group = possibilities[int(rng.integers(len(possibilities)))]
    pair = list(group[int(rng.integers(len(group)))])
    rng.shuffle(pair)
    return pair


def _mask(rng, h, w):
    if h == 1 or w == 1 or rng.random() < .4:
        return np.ones((h, w), bool)
    mask = np.zeros((h, w), bool)
    mask[0] = True
    start = int(rng.integers(w))
    stop = int(rng.integers(start+1, w+1))
    mask[1, start:stop] = True
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _make_pair(rng, periods, counts, dest_count, source_colors, bg, ink,
               rotation, anchor=False):
    events = _events(periods, counts)
    slots = dest_count+max(events)+1
    choices = [(h, p) for h in (1, 2) for p in (h+1, h+2)
               if h+(slots-1)*p+4 <= 30]
    _require(bool(choices), "schedule too tall")
    h, pitch = choices[int(rng.integers(len(choices)))]
    palette = [c for c in range(10) if c not in (bg, ink)]
    items = [dict(width=p, count=n, color=c, source=True)
             for p, n, c in reversed(list(zip(periods, counts, source_colors)))]
    minimum = sum(periods)+len(periods)+1+4  # receiver width initially one
    extra_count = 1 if anchor else int(rng.integers(0, 3))
    for j in range(extra_count):
        room = 30-minimum-1
        if room < 1:
            break
        width = int(rng.integers(1, min(5, room)+1))
        item = dict(width=width, count=int(rng.integers(1, min(5, slots)+1)),
                    color=source_colors[0] if anchor else int(rng.choice(palette)),
                    source=False)
        where = len(items) if anchor else int(rng.integers(len(items)+1))
        items.insert(where, item)
        minimum += width+1
    dest_width = int(rng.integers(1, min(5, 31-minimum)+1))
    minimum += dest_width-1
    _require(minimum <= 30, "scene too wide")
    width = int(rng.integers(minimum, min(30, minimum+3)+1)
                if rng.random() < .4 else rng.integers(minimum, 31))
    extras = rng.multinomial(width-minimum, np.full(len(items)+3, 1/(len(items)+3)))
    x = 1+int(extras[0])
    for i, item in enumerate(items):
        item['x'] = x
        item['mask'] = _mask(rng, h, item['width'])
        x += item['width']+1+int(extras[i+1])
    divider = x
    dest_x = divider+2+int(extras[len(items)+1])
    _require(dest_x+dest_width+1+int(extras[-1]) == width, "width accounting")
    dest_mask = _mask(rng, h, dest_width)
    dest_color = int(rng.choice(palette))
    extent = h+(slots-1)*pitch
    spare = 30-(extent+4)
    gap = int(rng.integers(1, min(3, spare+1)+1))
    spare -= gap-1
    bottom_margin = int(rng.integers(1, min(3, spare+1)+1))
    min_height = extent+2+gap+bottom_margin
    height = int(rng.integers(min_height, 31))
    cy = height-1-bottom_margin
    bottom_y = cy-gap-h
    a = np.full((height, width), bg, dtype=np.int8)
    a[1:-1, divider] = ink
    for item in items:
        for k in range(item['count']):
            _draw(a, bottom_y-k*pitch, item['x'], item['mask'], item['color'])
        if item['source']:
            a[cy, item['x']:item['x']+item['width']] = item['color']
    for k in range(dest_count):
        _draw(a, bottom_y-k*pitch, dest_x, dest_mask, dest_color)
    a[cy, dest_x:dest_x+dest_width] = ink
    b = a.copy()
    for item in items:
        if item['source']:
            for k in range(item['count']):
                _draw(b, bottom_y-k*pitch, item['x'], item['mask'], ink)
            b[cy, item['x']:item['x']+item['width']] = bg
    b[cy, dest_x:dest_x+dest_width] = bg
    for t, i in events.items():
        _draw(b, bottom_y-(dest_count+t)*pitch, dest_x, dest_mask, source_colors[i])
    return dict(input=np.rot90(a, rotation).copy(), output=np.rot90(b, rotation).copy())


def _random_operands(rng, count):
    for _ in range(100):
        periods = tuple(map(int, rng.integers(1, 7, size=count)))
        stocks = tuple(map(int, rng.integers(1, 5, size=count)))
        dest = int(rng.integers(2, 5))
        if sum(periods)+count+5 > 30:
            continue
        ev = _events(periods, stocks)
        if 1+(dest+max(ev))*2+4 <= 30:
            return periods, stocks, dest
    # Finite, always feasible fallback, without clipping or hidden count repair.
    return (1,)*count, (1,)*count, 2


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # Forty equivariant variants exceed the fourteen possible input count.
    for turn in range(4):
        for shift in range(10):
            changed = {r: ((np.rot90(g, turn).astype(int)+shift) % 10).astype(np.int8)
                       for r, g in pair.items()}
            if _key(changed['input']) not in seen:
                return changed
    raise AssertionError("finite uniqueness fallback exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg, ink, ca, cb = map(int, rng.choice(10, 4, replace=False))
    fixed_frame = bool(rng.integers(2))
    aggregation_shift = bool(rng.integers(2))
    base_turn = int(rng.integers(4))
    mixed_turns = aggregation_shift and bool(rng.integers(2))
    lessons = _lesson_plan(rng)
    anchor = int(rng.integers(2))
    puzzle, seen, train_sources = {'train': [], 'test': []}, set(), []
    for index in range(nt+nq):
        if index < 2:
            periods, stocks, dest, _ = lessons[index]
            # The color is bound to its period in these two lessons, and the
            # same two colors exchange proximity to defeat fixed color rank.
            low, high = sorted(periods)
            source_colors = [ca if p == low else cb for p in periods]
            ns = 2
        else:
            if index < nt:
                ns = int(rng.integers(1, 4 if aggregation_shift else 5))
            elif index == nt and aggregation_shift:
                ns = int(rng.integers(max(train_sources)+1, 5))
            else:
                ns = int(rng.integers(1, 5))
            periods, stocks, dest = _random_operands(rng, ns)
            source_colors = None
        if fixed_frame:
            local_bg, local_ink = bg, ink
        else:
            local_bg, local_ink = map(int, rng.choice(
                [c for c in range(10) if c not in (ca, cb)], 2, replace=False))
        if source_colors is None:
            pool = [c for c in range(10) if c not in (local_bg, local_ink)]
            source_colors = list(map(int, rng.choice(pool, ns, replace=True)))
        if not aggregation_shift and index >= nt:
            turn = (base_turn+2) % 4
        else:
            turn = int(rng.integers(4)) if mixed_turns else base_turn
        pair = _make_pair(rng, periods, stocks, dest, source_colors,
                          local_bg, local_ink, turn, anchor=index == anchor)
        pair = _unique(pair, seen)
        seen.add(_key(pair['input']))
        puzzle['train' if index < nt else 'test'].append(pair)
        if index < nt:
            train_sources.append(ns)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    src, dst = info['sources'], info['receiver']
    periods, stocks = [s['period'] for s in src], [s['count'] for s in src]
    ev = _events(periods, stocks)
    return dict(shape=info['canonical'].shape, rotation=(-info['turn']) % 4,
                periods=tuple(periods), stocks=tuple(stocks), sources=len(src),
                unmarked=len(info['unmarked']), pitch=info['pitch'],
                height=info['height'], destination=dst['count'], total=sum(stocks),
                duration=max(ev)+1, gaps=max(ev)+1-len(ev),
                masks=tuple((s['mask'].shape, s['mask'].tobytes()) for s in src+[dst]),
                stock_effect=_stock_effect(periods, stocks, ev))


def _teaching_check(train, infos):
    rivals = ('no_clocks', 'infinite', 'one_per_bar', 'pixels',
              'short', 'long', 'far', 'packed')
    for rival in rivals:
        assert any(not np.array_equal(np.asarray(p['output']), _answer(i, rival))
                   for p, i in zip(train, infos)), 'undistinguished rival: '+rival
    periods_taught, precedences = set(), set()
    shorter_near = longer_near = stock_effect = ignored_match = False
    totals, destinations = [], []
    for info in infos:
        src, dst = info['sources'], info['receiver']
        ps, ks = [s['period'] for s in src], [s['count'] for s in src]
        ev = _events(ps, ks)
        for i, s in enumerate(src):
            times = [t for t, winner in ev.items() if winner == i]
            if any(b-a == s['period'] for a, b in zip(times, times[1:])):
                periods_taught.add(s['period'])
        for far in src[1:]:
            if far['color'] != src[0]['color']:
                shorter_near |= src[0]['period'] < far['period']
                longer_near |= src[0]['period'] > far['period']
                precedences.add((src[0]['color'], far['color']))
        stock_effect |= _stock_effect(ps, ks, ev)
        ignored_match |= any(u['color'] == s['color'] for u in info['unmarked'] for s in src)
        totals.append(sum(ks))
        destinations.append(dst['count'])
    assert len(periods_taught) >= 2, 'two distinct numerical recurrences'
    assert shorter_near and longer_near, 'both proximity/length orders'
    assert any((b, a) in precedences for a, b in precedences), 'color ranking cycle'
    assert stock_effect and ignored_match, 'depletion and ignored matching stack'
    assert len(set(totals)) > 1 and len(set(destinations)) > 1, 'varied operands'
    assert any(s*destinations[0] != totals[0]*d for s, d in zip(totals, destinations))
    for rival in (lambda s, d: 2*d, lambda s, d: 2*s,
                  lambda s, d: s*d, lambda s, d: max(s, d), lambda s, d: s):
        assert any(rival(s, d) != s+d for s, d in zip(totals, destinations)), 'arithmetic ambiguity'


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            assert np.array_equal(_answer(info), b), 'timing, stocks, shape, highlighting or preservation'
            out = np.rot90(b, info['turn'])
            src, dst = info['sources'], info['receiver']
            ev = _events([s['period'] for s in src], [s['count'] for s in src])
            assert len(ev) == sum(s['count'] for s in src), 'conservation'
            # Reconstruct output objects in the receiver; count objects, not cells.
            strip = out[:, dst['x']:dst['x']+dst['width']]
            groups = _runs(np.any(strip != info['background'], axis=1))
            assert len(groups) == dst['count']+sum(s['count'] for s in src)
            for y, end in groups:
                assert end-y == info['height']
                assert np.array_equal(strip[y:end] != info['background'], dst['mask'])
            for t, winner in ev.items():
                y = dst['ys'][0]-(t+1)*info['pitch']
                block = strip[y:y+info['height']]
                assert np.all(block[dst['mask']] == src[winner]['color'])
            for s in info['unmarked']:
                assert np.array_equal(out[:, s['x']:s['x']+s['width']],
                                      info['canonical'][:, s['x']:s['x']+s['width']])
            assert np.all(out[info['control_y'], :info['divider']] == info['background'])
            assert np.all(out[info['control_y'], info['divider']+1:] == info['background'])
            infos[split].append(info)
    _teaching_check(puzzle['train'], infos['train'])
    test = infos['test'][0]
    more_sources = len(test['sources']) > max(len(i['sources']) for i in infos['train'])
    reversed_direction = all((test['turn']-i['turn']) % 4 == 2 for i in infos['train'])
    assert more_sources or reversed_direction, 'missing controlled train/test shift'


def self_test():
    fixtures = [((2, 3), (3, 2), [0, None, 0, 1, 0, None, 1]),
                ((3, 2), (2, 3), [0, None, 1, 0, 1, None, 1]),
                ((3, 3), (1, 2), [0, None, None, 1, None, None, 1]),
                ((1, 2, 3), (2, 1, 2), [0, 0, 1, 2, None, None, 2]),
                ((4, 2, 3), (1, 2, 1), [0, None, 1, 2, 1])]
    for ps, ks, expected in fixtures:
        assert _ticks(ps, ks) == expected
        events = _events(ps, ks)
        assert [events.get(t) for t in range(max(events)+1)] == expected
        p = _make_pair(np.random.default_rng(50), ps, ks, 2,
                       list(range(1, len(ps)+1)), 0, 8, 0)
        info = _parse(p['input'])
        hand = info['canonical'].copy()
        for s in info['sources']:
            for y in s['ys']:
                _draw(hand, y, s['x'], s['mask'], 8)
            hand[info['control_y'], s['x']:s['x']+s['width']] = 0
        dst = info['receiver']
        hand[info['control_y'], dst['x']:dst['x']+dst['width']] = 0
        for t, i in enumerate(expected):
            if i is not None:
                _draw(hand, dst['ys'][0]-(t+1)*info['pitch'], dst['x'], dst['mask'], i+1)
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(p['input'], turn)), np.rot90(hand, turn))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows, signatures, normalized = [], set(), set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        repeat = generate_puzzle(seed)
        for split in puzzle:
            for p, q in zip(puzzle[split], repeat[split]):
                assert all(np.array_equal(p[r], q[r]) for r in p), 'determinism'
                assert np.array_equal(solve(p['input']), p['output'])
                rows.append(_features(_parse(p['input'])))
                roles = {}
                a = _parse(p['input'])['canonical']
                normalized.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
    assert len(signatures) == 42
    assert len({f['shape'] for f in rows}) >= 35
    assert len(normalized) >= 150
    assert {f['rotation'] for f in rows} == set(range(4))
    assert {f['sources'] for f in rows} == {1, 2, 3, 4}
    assert {f['unmarked'] for f in rows} == {0, 1, 2}
    assert set(range(1, 7)) <= {p for f in rows for p in f['periods']}
    assert set(range(1, 5)) <= {n for f in rows for n in f['stocks']}
    assert {f['height'] for f in rows} == {1, 2}
    assert len({f['masks'] for f in rows}) >= 40
    assert any(f['gaps'] == 0 for f in rows) and any(f['gaps'] >= 3 for f in rows)
    # Every legal explicit count request; two demonstrations must suffice.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(301+nt*4+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Mutations cover addition, highlighting, marker removal and ignored stacks.
    original = puzzles[0]
    for defect in ('emission', 'highlight', 'control', 'unmarked'):
        broken = {s: [{r: g.copy() for r, g in p.items()} for p in original[s]] for s in original}
        for pair in broken['train']:
            info = _parse(pair['input'])
            if defect == 'unmarked' and not info['unmarked']:
                continue
            b = np.rot90(pair['output'], info['turn'])
            src, dst = info['sources'][0], info['receiver']
            if defect == 'emission':
                _draw(b, dst['ys'][0]-info['pitch'], dst['x'], dst['mask'], info['background'])
            elif defect == 'highlight':
                _draw(b, src['ys'][0], src['x'], src['mask'], src['color'])
            elif defect == 'control':
                b[info['control_y'], src['x']:src['x']+src['width']] = src['color']
            else:
                u = info['unmarked'][0]
                _draw(b, u['ys'][0], u['x'], u['mask'], info['ink'])
            break
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('output mutation accepted: '+defect)
    pair = original['test'][0]
    perm = np.random.default_rng(71).permutation(10)
    for turn in range(4):
        assert np.array_equal(solve(perm[np.rot90(pair['input'], turn)]),
                              perm[np.rot90(pair['output'], turn)])
    return dict(seeds=42, pairs=len(rows), distinct_shapes=len({f['shape'] for f in rows}),
                normalized_structures=len(normalized), distinct_puzzles=len(signatures),
                source_counts=sorted({f['sources'] for f in rows}),
                periods=sorted({p for f in rows for p in f['periods']}))
