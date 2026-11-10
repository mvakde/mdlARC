"""Optical ballots: local card votes, visibility-weighted global election.

No parent implementation or fixture is imported.  The input parser and lane
scanner recover everything used by solve; construction uses explicit masks.
"""
from itertools import combinations
import numbers

import numpy as np


HOST_SEEDS = (0, 930610356, 867091311, 456099312, 875325547, 678568668,
              265709497, 984649769, 120642613, 390144336, 736175601,
              659386442, 153207703, 805210314, 748585855, 408632443,
              299383233, 426806887, 994177182, 704509298, 39520362,
              435538825, 454792701, 475046048, 610951686, 926294928,
              326618951, 757239443, 251861439, 949195503, 968419198,
              795064809, 29165420, 563732257, 975072856, 652861032,
              201288774, 817812551, 388844726, 882571653, 952733696,
              626275620)


def _integer(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and
            1 <= b <= 4 and a > b):
        raise ValueError("counts require 2..10 train, 1..4 test, train > test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape) or
            not np.issubdtype(a.dtype, np.integer) or
            not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    left = set(map(tuple, np.argwhere(mask)))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in left:
                    left.remove(p)
                    group.append(p)
                    pending.append(p)
        result.append(group)
    return result


def _fit_rail(values, sign):
    """Fit the whole visible rail, with two complete outward steps as evidence."""
    candidates = []
    delta = sign * (np.asarray(values) - values[0])
    for run in (1, 2, 3):
        for phase in range(run):
            predicted = (np.arange(len(values)) + phase) // run
            if np.array_equal(delta, predicted) and delta[-1] >= 2:
                candidates.append((run, phase))
    if len(candidates) != 1:
        raise ValueError("rail cadence is ambiguous")
    return candidates[0]


def _funnel(a, bg, wall):
    p = np.argwhere(a == wall)
    rear, mouth = int(p[:, 1].min()), int(p[:, 1].max())
    if mouth-rear < 4 or mouth >= a.shape[1]-3:
        raise ValueError("not an open funnel")
    cap = np.flatnonzero(a[:, rear] == wall)
    if len(cap) < 3 or np.any(np.diff(cap) != 1):
        raise ValueError("no closed back")
    upper, lower = [], []
    for x in range(rear+1, mouth+1):
        rows = np.flatnonzero(a[:, x] == wall)
        if len(rows) != 2 or rows[1]-rows[0] < 2:
            raise ValueError("rails need one thin boundary cell per column")
        upper.append(int(rows[0]))
        lower.append(int(rows[1]))
    if cap[0] != upper[0] or cap[-1] != lower[0]:
        raise ValueError("back does not join rails")
    fits = (_fit_rail(upper, -1), _fit_rail(lower, 1))
    footprint = np.zeros(a.shape, dtype=bool)
    bounds = []
    for x in range(rear+1, a.shape[1]):
        j = x-rear-1
        top = upper[0] - (j+fits[0][1])//fits[0][0]
        bottom = lower[0] + (j+fits[1][1])//fits[1][0]
        footprint[:, x] = (np.arange(a.shape[0]) > top) & (np.arange(a.shape[0]) < bottom)
        bounds.append((top, bottom))
    if np.any(footprint & (a == wall)):
        raise ValueError("walls intersect interior")
    return dict(rear=rear, mouth=mouth, fits=fits, footprint=footprint,
                bounds=bounds, mouth_rows=(upper[-1], lower[-1]))


def _cards(a, signals, sensor, mouth):
    boxes, majorities, tabs = [], [], []
    expected_tabs = set()
    for cells in _components(np.isin(a, signals)):
        ys, xs = zip(*cells)
        r, y, c, x = min(ys), max(ys)+1, min(xs), max(xs)+1
        if len(cells) != (y-r)*(x-c) or y-r < 2 or x-c < 2 or c < mouth+3:
            raise ValueError("signals must form downstream rectangular cards")
        block = a[r:y, c:x]
        n = int(np.sum(block == signals[0]))
        if n == 0 or n == block.size or 2*n == block.size:
            raise ValueError("each card needs a strict two-color majority")
        ps = [(row, c-1) for row in range(r, y) if a[row, c-1] == sensor]
        if not ps:
            raise ValueError("each card needs at least one upstream tab")
        boxes.append((r, y, c, x))
        majorities.append(signals[int(2*n < block.size)])
        tabs.append(ps)
        expected_tabs.update(ps)
    if not 2 <= len(boxes) <= 6 or expected_tabs != set(map(tuple, np.argwhere(a == sensor))):
        raise ValueError("sensor role must consist exactly of upstream card tabs")
    return boxes, majorities, tabs


def _inspect(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size//2:
        raise ValueError("background must occupy more than half the input")
    colors = tuple(map(int, np.unique(inp)))
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        for wall in colors:
            if wall == bg:
                continue
            try:
                f = _funnel(a, bg, wall)
            except ValueError:
                continue
            pool = [v for v in colors if v not in (bg, wall)]
            for signals in combinations(pool, 2):
                for sensor in pool:
                    if sensor in signals:
                        continue
                    try:
                        boxes, majorities, tabs = _cards(a, signals, sensor, f['mouth'])
                    except ValueError:
                        continue
                    # Chamber is empty: its beam color is elected by remote cards.
                    if np.any(f['footprint'][:, :f['mouth']+1] &
                              (a[:, :f['mouth']+1] != bg)):
                        continue
                    matches.append(dict(canonical=a, rotation=rotation, bg=bg,
                                        wall=wall, signals=signals, sensor=sensor,
                                        boxes=boxes, majorities=majorities, tabs=tabs, **f))
    if len(matches) != 1:
        raise ValueError("input needs exactly one funnel/card interpretation")
    info = matches[0]
    a, fp = info['canonical'], info['footprint']
    lit = np.zeros(a.shape, dtype=bool)
    # Independent lane scan, not the construction's union-of-shadows formula.
    for row in range(a.shape[0]):
        stopped = False
        for x in range(info['rear']+1, a.shape[1]):
            if fp[row, x]:
                if not stopped:
                    lit[row, x] = True
                if a[row, x] != bg:
                    stopped = True
    weights = [sum(bool(lit[p]) for p in ps) for ps in info['tabs']]
    votes = [sum(w for w, m in zip(weights, info['majorities']) if m == s)
             for s in info['signals']]
    if min(votes) == 0 or votes[0] == votes[1]:
        raise ValueError("both signals need visible votes and the election must be strict")
    info.update(lit=lit, weights=weights, votes=votes,
                winner=info['signals'][int(votes[1] > votes[0])])
    return info


def _answer(info):
    a = info['canonical']
    out = a.copy()
    out[info['lit'] & (a == info['bg'])] = info['winner']
    for ps, majority in zip(info['tabs'], info['majorities']):
        for p in ps:
            if info['lit'][p]:
                out[p] = majority
    return np.rot90(out, info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry, rectangular cards and upstream opacity only."""
    return _answer(_inspect(input_grid))


def _visibility_mask(inp, bg, footprint):
    """Construction path: union the shadows cast by all opaque input cells."""
    shadow = np.zeros(inp.shape, dtype=bool)
    for row, x in np.argwhere(footprint & (inp != bg)):
        shadow[row, x+1:] = True
    return footprint & ~shadow


def _geometry(rng, large):
    height = int(rng.integers(23 if large else 19, 31))
    width = int(rng.integers(27 if large else 23, 31))
    if rng.random() < .10:
        height = width = 30
    runs = tuple(map(int, rng.integers(1, 4, 2)))
    phases = tuple(int(rng.integers(k)) for k in runs)
    length = 2*max(runs)+int(rng.integers(1, 3))
    rear = int(rng.integers(0, 3))
    u, d = map(int, rng.integers(1, 4, 2))
    up = u+(length-1+phases[0])//runs[0]
    down = d+(length-1+phases[1])//runs[1]
    if up+down >= height-2:
        return None
    center = int(rng.integers(up+1, height-down-1))
    wall = np.zeros((height, width), dtype=bool)
    fp = np.zeros_like(wall)
    wall[center-u:center+d+1, rear] = True
    for x in range(rear+1, width):
        j = x-rear-1
        top = center-u-(j+phases[0])//runs[0]
        bottom = center+d+(j+phases[1])//runs[1]
        fp[max(0, top+1):min(height, bottom), x] = True
        if j < length:
            wall[top, x] = wall[bottom, x] = True
    return wall, fp, rear+length


def _place_cards(rng, shape, mouth, kind):
    h, w = shape
    count = 3 if kind in ('weight', 'shadow') else int(rng.integers(4, 7) if kind == 'test' else rng.integers(2, 6))
    if kind == 'weight':
        tab_counts = [int(rng.integers(3, 6)), 1, 1]
        rng.shuffle(tab_counts)
    elif kind == 'shadow':
        tab_counts = [int(rng.integers(4, 6)), 1, 2]
        rng.shuffle(tab_counts)
    else:
        tab_counts = list(map(int, rng.integers(1, 5, count)))
    cards, reserved = [], np.zeros(shape, dtype=bool)
    # Largest tab banks are packed first, with independently sampled depths.
    for n in tab_counts:
        accepted = False
        for _ in range(100):
            ph = int(rng.integers(max(2, n), max(2, n)+3))
            pw = int(rng.integers(2, 7))
            if kind == 'weight':
                pw = 2 if n > 1 else int(rng.integers(4, 8))
            if ph >= h-1 or mouth+4 >= w-pw:
                continue
            r = int(rng.integers(1, h-ph))
            c = int(rng.integers(mouth+3, w-pw+1))
            # A one-cell moat makes card segmentation and tab ownership clear.
            if np.any(reserved[r-1:r+ph+1, c-2:min(w, c+pw+1)]):
                continue
            rows = sorted(map(int, rng.choice(np.arange(r, r+ph), n, replace=False)))
            tabs = [(row, c-1) for row in rows]
            reserved[r:r+ph, c:c+pw] = True
            for p in tabs:
                reserved[p] = True
            cards.append(((r, r+ph, c, c+pw), tabs))
            accepted = True
            break
        if not accepted:
            return None
    return cards


def _pattern(rng, shape, majority, minority, close=False):
    area = int(np.prod(shape))
    nminor = ((area-1)//2 if close else
              int(rng.integers(1, (area-1)//2+1)))
    result = np.full(shape, majority, dtype=np.int8)
    style = int(rng.integers(3))
    rr, cc = np.indices(shape)
    if style == 0:
        order = rng.permutation(area)
    elif style == 1:
        r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        order = np.argsort((abs(rr-r)+abs(cc-c)+rng.random(shape)*.25).ravel())
    else:
        order = np.argsort(((rr+cc)%2*area+cc*shape[0]+rr).ravel())
    result.flat[order[:nminor]] = minority
    return result


def _build(rng, roles, rotation, kind, winner_index):
    geom = _geometry(rng, kind == 'test')
    if geom is None:
        return None
    wall_mask, fp, mouth = geom
    cards = _place_cards(rng, fp.shape, mouth, kind)
    if cards is None:
        return None
    bg, wall, sensor, s0, s1, b0, b1 = roles
    inp = np.full(fp.shape, bg, dtype=np.int8)
    inp[wall_mask] = wall
    cardmask = np.zeros(fp.shape, dtype=bool)
    for (r, y, c, x), tabs in cards:
        inp[r:y, c:x] = s0  # opacity does not depend on local label
        cardmask[r:y, c:x] = True
        for p in tabs:
            inp[p] = sensor
    # In the second witness, an upstream shutter hides a chosen bank's tabs.
    if kind == 'shadow':
        _, heavy_tabs = max(cards, key=lambda card: len(card[1]))
        hidden = heavy_tabs[:-1]
        x = min(p[1] for p in heavy_tabs)-int(rng.integers(2, 5))
        if x <= mouth or x < 0:
            return None
        if any(inp[row, x] != bg for row, _ in hidden):
            return None
        for row, _ in hidden:
            inp[row, x] = b0
    elif kind != 'weight':
        # Singles, transverse bars, rectangles and L groups, including inert ones.
        offsets = (((0, 0),), ((0, 0), (1, 0)),
                   ((0, 0), (1, 0), (0, 1), (1, 1)),
                   ((0, 0), (1, 0), (1, 1)))
        for _ in range(int(rng.integers(0, 6))):
            group = offsets[int(rng.integers(len(offsets)))]
            r, c = int(rng.integers(inp.shape[0]-1)), int(rng.integers(mouth+1, inp.shape[1]-1))
            ps = [(r+dr, c+dc) for dr, dc in group]
            if all(inp[p] == bg for p in ps):
                color = (b0, b1)[int(rng.integers(2))]
                for p in ps:
                    inp[p] = color
    lit = _visibility_mask(inp, bg, fp)
    weights = [sum(bool(lit[p]) for p in tabs) for _, tabs in cards]
    nominal = [len(tabs) for _, tabs in cards]
    if kind == 'shadow' and not all(fp[p] for _, tabs in cards for p in tabs):
        return None
    if kind == 'weight' and not np.any(fp & ~lit & (inp == bg)):
        return None
    clear_weights = weights
    if kind == 'shadow':
        clear = inp.copy()
        clear[np.isin(clear, (b0, b1))] = bg
        clear_lit = _visibility_mask(clear, bg, fp)
        clear_weights = [sum(bool(clear_lit[p]) for p in tabs) for _, tabs in cards]
    candidates = []
    for bits in range(1, 2**len(cards)-1):
        iswin = [bool(bits & (1 << i)) for i in range(len(cards))]
        delta = sum(w if win else -w for w, win in zip(weights, iswin))
        if delta <= 0 or sum(w for w, win in zip(weights, iswin) if not win) == 0:
            continue
        if kind == 'weight' and (2*sum(iswin) >= len(cards) or weights != nominal):
            continue
        if kind == 'shadow' and sum(n if win else -n for n, win in zip(nominal, iswin)) >= 0:
            continue
        if kind == 'shadow' and sum(n if win else -n for n, win in zip(clear_weights, iswin)) >= 0:
            continue
        candidates.append(iswin)
    if not candidates:
        return None
    iswin = candidates[int(rng.integers(len(candidates)))]
    winner, loser = (s0, s1) if winner_index == 0 else (s1, s0)
    majorities = []
    for ((r, y, c, x), _), win in zip(cards, iswin):
        majority, minority = (winner, loser) if win else (loser, winner)
        majorities.append(majority)
        inp[r:y, c:x] = _pattern(rng, (y-r, x-c), majority, minority,
                                close=(kind == 'weight' and win))
    if kind == 'weight' and np.sum(inp == winner) >= np.sum(inp == loser):
        return None
    if kind == 'weight' and not any(lit[p] and inp[p[0], p[1]+1] != m
                                   for (_, tabs), m in zip(cards, majorities) for p in tabs):
        return None
    if np.sum(inp == bg) <= inp.size//2:
        return None
    if kind == 'test':
        # New combination: depth overlap and a sensor in the widening fringe.
        hidden_by_card = any(fp[p] and np.any(cardmask[p[0], :p[1]] & fp[p[0], :p[1]])
                             for _, tabs in cards for p in tabs)
        mouth_rows = np.flatnonzero(wall_mask[:, mouth])
        fringe = any(lit[p] and not (mouth_rows[0] < p[0] < mouth_rows[-1])
                     for _, tabs in cards for p in tabs)
        if not hidden_by_card or not fringe:
            return None
    out = inp.copy()
    out[lit & (inp == bg)] = winner
    for (_, tabs), majority in zip(cards, majorities):
        for p in tabs:
            if lit[p]:
                out[p] = majority
    pair = {'input': np.rot90(inp, rotation).copy(),
            'output': np.rot90(out, rotation).copy()}
    # Uniqueness check only; the answer above came from construction records.
    try:
        if not np.array_equal(solve(pair['input']), pair['output']):
            return None
    except ValueError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    shared_palette = bool(rng.random() < .25)
    palette = tuple(map(int, rng.permutation(10)[:7]))
    base = int(rng.integers(2))
    preferred = int(rng.integers(4))
    shared_direction = bool(rng.random() < .25)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for index in range(count):
            kind = ('weight' if index == 0 else 'shadow' if index == 1 else 'general') if split == 'train' else 'test'
            roles = palette if shared_palette else tuple(map(int, rng.permutation(10)[:7]))
            rotation = preferred if shared_direction else int(rng.integers(4))
            winner_index = (base if index == 0 else 1-base if index == 1 else int(rng.integers(2))) if split == 'train' else int(rng.integers(2))
            for _ in range(2500):
                pair = _build(rng, roles, rotation, kind, winner_index)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded optical-card construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(info):
    a, signals, winner = info['canonical'], info['signals'], info['winner']
    nominal = [len(ps) for ps in info['tabs']]
    signed = lambda weights: sum(w if m == winner else -w
                                 for w, m in zip(weights, info['majorities']))
    top, bottom = info['mouth_rows']
    cardmask = np.isin(a, signals)
    # Counterfactual without obstacle colors, while cards and tabs still absorb.
    fixed_opaque = np.isin(a, (info['wall'], info['sensor'], *signals)) & info['footprint']
    without_obstacles = [sum(bool(info['footprint'][p]) and not np.any(fixed_opaque[p[0], :p[1]])
                             for p in ps) for ps in info['tabs']]
    return dict(shape=a.shape, rotation=info['rotation'], runs=tuple(f[0] for f in info['fits']),
                phases=tuple(f[1] for f in info['fits']), cards=len(nominal),
                weights=tuple(info['weights']), nominal=tuple(nominal),
                panel_margin=signed([1]*len(nominal)), nominal_margin=signed(nominal),
                all_in_cone=all(info['footprint'][p] for ps in info['tabs'] for p in ps),
                without_obstacle_margin=signed(without_obstacles),
                minority_tab=any(info['lit'][p] and a[p[0], p[1]+1] != m
                                 for ps, m in zip(info['tabs'], info['majorities']) for p in ps),
                raw_margin=int(np.sum(a == winner)-np.sum(a == next(s for s in signals if s != winner))),
                all_visible=info['weights'] == nominal,
                fringe=any(info['lit'][p] and not top < p[0] < bottom
                           for ps in info['tabs'] for p in ps),
                card_occlusion=any(info['footprint'][p] and np.any(cardmask[p[0], :p[1]] &
                                       info['footprint'][p[0], :p[1]]) for ps in info['tabs'] for p in ps),
                tab_count=sum(nominal), visible=sum(info['weights']),
                roles=(info['bg'], info['wall'], info['sensor'], *signals),
                boxes=tuple(info['boxes']),
                shadow=int(np.sum(info['footprint'] & ~info['lit'] & (a == info['bg']))))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _inspect(inp)
            assert np.array_equal(_answer(info), out), 'incorrect optical election answer'
            a, b = info['canonical'], np.rot90(out, -info['rotation'])
            changed = a != b
            assert np.any(changed)
            assert np.all(np.isin(a[changed], (info['bg'], info['sensor'])))
            assert np.all(info['lit'][changed]), 'shadow or exterior was changed'
            assert np.array_equal(a[np.isin(a, info['signals'])], b[np.isin(a, info['signals'])])
            assert np.array_equal(a[a == info['wall']], b[a == info['wall']])
            # Every exposed tab shows its *own* card's majority, even if it loses.
            for ps, majority in zip(info['tabs'], info['majorities']):
                for p in ps:
                    assert b[p] == (majority if info['lit'][p] else info['sensor'])
            records[split].append(_measure(info))
    assert any(m['panel_margin'] < 0 and m['raw_margin'] < 0 and m['all_visible'] and m['shadow'] > 0 and m['minority_tab']
               for m in records['train']), 'missing weighted-vote witness'
    assert any(m['nominal_margin'] < 0 and m['without_obstacle_margin'] < 0 and m['shadow'] > 0 and m['all_in_cone']
               for m in records['train']), 'missing occlusion-flip witness'
    assert all(m['card_occlusion'] and m['fringe'] for m in records['test'])
    return None


def self_test():
    """Bounded 42-seed semantic, diversity, metamorphic and corruption checks."""
    # Hand-written period-two fixture.  Four tabs favor 1, three favor 2;
    # a two-cell shutter hides two of the former, electing 2 by 3:2.
    # It does not use the geometry, packing, pattern or rendering constructors.
    fixture = np.zeros((15, 25), dtype=np.int8)
    fixture[5:10, 1] = 5
    for x, top in enumerate((5, 5, 4, 4, 3), 2):
        fixture[top, x] = fixture[14-top, x] = 5
    fixture[2:6, 12:14] = np.array([[1, 2], [1, 1], [2, 1], [1, 2]])
    fixture[8:10, 15:19] = np.array([[2, 2, 1, 2], [2, 1, 2, 2]])
    fixture[12:14, 19:23] = np.array([[2, 1, 2, 1], [2, 2, 1, 2]])
    fixture[2:6, 11] = 8
    fixture[8:10, 14] = 8
    fixture[12, 18] = 8
    fixture[3:5, 9] = 9
    fixture[0, 7] = 9  # Outside the beam here: it must not shadow later entry.
    answer = solve(fixture)
    assert answer[7, 3] == 2  # Elected color fills the chamber.
    assert answer[2, 11] == answer[5, 11] == 1  # A losing card's lit tabs.
    assert answer[3, 11] == answer[4, 11] == 8  # Hidden tabs remain neutral.
    assert answer[3, 20] == answer[4, 20] == 0  # Straight, unwidened shadows.
    assert answer[2, 20] == 0  # Card 1 casts its own shadow in this lane.
    assert answer[0, 20] == 2  # Upstream exterior object has no effect.
    assert answer[3, 7] == 0 and answer[4, 7] == 2  # Next thin virtual boundary.
    clear = fixture.copy()
    clear[3:5, 9] = 0
    assert solve(clear)[7, 3] == 1  # Omitting opacity reverses the result.
    relabeled = fixture.copy()
    relabeled[8:10, 15:19] = 3-relabeled[8:10, 15:19]
    assert solve(relabeled)[7, 3] == 1  # Local card majority matters globally.
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    records, keys, shapes, roles, layouts = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        keys.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect(pair['input'])
                m = _measure(info)
                records.append(m)
                shapes.add(pair['input'].shape)
                roles.add(m['roles'])
                layouts.add((m['shape'], m['boxes'], m['nominal'], m['weights']))
    assert len(keys) == 42 and len(shapes) >= 30 and len(roles) >= 60
    assert len(layouts) >= 180
    assert {m['rotation'] for m in records} == {0, 1, 2, 3}
    assert {k for m in records for k in m['runs']} == {1, 2, 3}
    assert len({m['cards'] for m in records}) >= 4
    assert len({m['weights'] for m in records}) >= 50
    assert any(m['cards'] >= 5 and m['fringe'] and m['card_occlusion'] for m in records)
    assert max(m['shadow'] for m in records) >= 40
    # Rotation and color equivariance use no construction metadata.
    pair = puzzles[0]['train'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
    permutation = np.array([7, 3, 9, 0, 8, 2, 6, 4, 1, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(81+a*7+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    p = damaged['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len(shapes),
            'role_palettes': len(roles), 'optical_layouts': len(layouts)}


if __name__ == '__main__':
    print(self_test())
