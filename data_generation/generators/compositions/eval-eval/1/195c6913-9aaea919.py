"""Count-powered staircase: unary supplies extend a terrain-guided cyclic prefix.

Only the standard library and NumPy are used. Generation draws explicit ground
truth from counts; solve reconstructs all roles, counts and geometry from pixels.
"""
from __future__ import annotations

import json
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b) or not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require integer counts 2..10 / 1..4, with num_train > num_test")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    if a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape):
        raise ValueError("Grid dimensions must be 1..30")
    if not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9)):
        raise ValueError("Grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            r, c = todo.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _runs(values):
    edges = np.diff(np.r_[False, values, False].astype(int))
    return list(zip(map(int, np.flatnonzero(edges == 1)), map(int, np.flatnonzero(edges == -1))))


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _trace(mask, start):
    """Unit-step walk. A canvas exit stops immediately, without another turn."""
    p, direction = start, 0
    path, contacts, legs = [], [], [0]
    for _ in range(mask.size):
        assert _inside(mask.shape, p) and mask[p]
        path.append(p)
        legs[-1] += 1
        dr, dc = ((0, 1), (-1, 0))[direction]
        ahead = (p[0]+dr, p[1]+dc)
        if not _inside(mask.shape, ahead):
            return dict(path=tuple(path), contacts=tuple(contacts), legs=tuple(legs), end="edge")
        if mask[ahead]:
            p = ahead
            continue
        # The index is the number of destination units needed to reach contact.
        contacts.append((len(path), ahead))
        direction = 1-direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        p = (p[0]+dr, p[1]+dc)
        if not _inside(mask.shape, p) or not mask[p]:
            return dict(path=tuple(path), contacts=tuple(contacts), legs=tuple(legs), end="blocked")
        legs.append(0)
    raise AssertionError("Nonterminating staircase")


def _parse(grid):
    """Decode the header, terrain-connected prefix and isolated spatial stacks."""
    a = _grid(grid)
    bg = int(a[0, 0])
    assert np.all(a[0] == bg)
    nonempty = np.flatnonzero(np.any(a != bg, axis=1))
    assert len(nonempty)
    row = int(nonempty[0])
    assert 1 <= row <= 3 and row+3 < a.shape[0]
    bands = _runs(a[row] != bg)
    assert 3 <= len(bands) <= 7
    assert all(b-x == 2 and np.all(a[row:row+2, x:b] == a[row, x]) for x, b in bands)
    gaps = [bands[i+1][0]-bands[i][1] for i in range(len(bands)-1)]
    assert all(g in (1, 2) for g in gaps[:-1]) and 4 <= gaps[-1] <= 6
    word = tuple(int(a[row, x]) for x, b in bands[:-1])
    marker = int(a[row, bands[-1][0]])
    assert 2 <= len(set(word)) and marker not in word and bg not in word and marker != bg
    annotations = {(r, c) for x, b in bands for r in (row, row+1) for c in range(x, b)}
    clean = a.copy()
    for p in annotations:
        clean[p] = bg
    assert np.all(clean[:row+4] == bg)
    # Compact tokens occupy at most 3x3 pixels, the prefix at most eight cells.
    terrain_candidates = []
    for color in np.unique(clean):
        color = int(color)
        if color in {bg, marker, *word}:
            continue
        if any(len(component) > 9 for component in _components(clean == color)):
            terrain_candidates.append(color)
    assert len(terrain_candidates) == 1, "Terrain must have a recognizable large remnant"
    terrain = terrain_candidates[0]
    regions = _components((clean != bg) & (clean != marker))
    terrain_regions = [c for c in regions if any(clean[p] == terrain for p in c)]
    assert len(terrain_regions) == 1, "Terrain and its prefix must form one region"
    region = set(terrain_regions[0])
    prefix = {p for p in region if clean[p] != terrain}
    assert 1 <= len(prefix) <= 8 and all(int(clean[p]) in word for p in prefix)
    bottom = max(r for r, c in prefix)
    start = (bottom, min(c for r, c in prefix if r == bottom))
    mask = np.zeros(a.shape, bool)
    for p in region:
        mask[p] = True
    assert start[1] == 0 or not mask[start[0], start[1]-1]
    assert _inside(a.shape, (start[0], start[1]+1)) and mask[start[0], start[1]+1]
    trace = _trace(mask, start)
    d = len(prefix)
    assert set(trace['path'][:d]) == prefix
    assert all(int(a[p]) == word[i % len(word)] for i, p in enumerate(trace['path'][:d]))
    for p in region:
        clean[p] = bg
    tokens, bars = [], []
    for cells in _components(clean != bg):
        rs, cs = zip(*cells)
        y, x, h, w = min(rs), min(cs), max(rs)-min(rs)+1, max(cs)-min(cs)+1
        colors = {int(clean[p]) for p in cells}
        assert len(colors) == 1
        color = colors.pop()
        if color == marker:
            assert h == 1 and 2 <= w <= 3 and len(cells) == w
            bars.append(dict(y=y, x=x, w=w, cells=cells))
        else:
            assert color != terrain and 2 <= h <= 3 and 2 <= w <= 3
            silhouette = np.zeros((h, w), bool)
            for r, c in cells:
                silhouette[r-y, c-x] = True
            tokens.append(dict(y=y, x=x, h=h, w=w, color=color, mask=silhouette, cells=cells))
    assert tokens and bars
    groups = {}
    for token in tokens:
        groups.setdefault((token['x'], token['w']), []).append(token)
    spans = sorted(groups)
    assert all(x+w < xx for (x, w), (xx, ww) in zip(spans, spans[1:]))
    stacks = []
    used_bars = set()
    for x, w in spans:
        ts = sorted(groups[x, w], key=lambda t: t['y'])
        first = ts[0]
        assert all(t['color'] == first['color'] and np.array_equal(t['mask'], first['mask']) for t in ts)
        pitches = {b['y']-a['y'] for a, b in zip(ts, ts[1:])}
        assert len(pitches) <= 1
        assert not pitches or next(iter(pitches)) > first['h']
        matched = [i for i, bar in enumerate(bars) if (bar['x'], bar['w']) == (x, w)]
        assert len(matched) <= 1
        controlled = bool(matched)
        if controlled:
            i = matched[0]
            gap = bars[i]['y']-(ts[-1]['y']+first['h'])
            assert 1 <= gap <= 3
            used_bars.add(i)
        cells = tuple(p for t in ts for p in t['cells'])
        stacks.append(dict(x=x, w=w, tokens=ts, count=len(ts), color=first['color'],
                           controlled=controlled, cells=cells,
                           bar=bars[matched[0]] if controlled else None))
    assert len(used_bars) == len(bars)
    src = [s for s in stacks if s['controlled']]
    supply = sum(s['count'] for s in src)
    total = d+supply
    assert len(trace['path']) >= total, "Supply must fit on the available route"
    for need, p in trace['contacts']:
        assert int(a[p]) == bg and p not in annotations
    return dict(grid=a, bg=bg, terrain=terrain, marker=marker, word=word,
                annotations=annotations, mask=mask, prefix=prefix, d=d,
                start=start, trace=trace, stacks=stacks, src=src, supply=supply, total=total)


def _render(info, *, total=None, word=None, reset_bends=False, consume_contacts=False, restart_prefix=False):
    a = info['grid'].copy()
    bg, marker = info['bg'], info['marker']
    for p in info['annotations']:
        a[p] = bg
    for stack in info['src']:
        for p in stack['cells']:
            a[p] = marker
        for p in stack['bar']['cells']:
            a[p] = bg
    word = info['word'] if word is None else word
    total = info['total'] if total is None else total
    # Restore the old prefix before comparing rival destination rules.
    for p in info['prefix']:
        a[p] = info['terrain']
    ends = set(np.cumsum(info['trace']['legs']).tolist())
    phase = 0
    for i, p in enumerate(info['trace']['path'][:max(0, total)]):
        if restart_prefix and i == info['d']:
            phase = 0
        a[p] = word[phase % len(word)]
        phase += 1
        if i+1 in ends:
            if reset_bends:
                phase = 0
            elif consume_contacts and any(n == i+1 for n, _ in info['trace']['contacts']):
                phase += 1
    for need, p in info['trace']['contacts']:
        if need <= total:
            a[p] = marker
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_parse(input_grid))


def _evidence(info):
    correct = _render(info)
    evidence = set()
    def differs(name, **kwargs):
        if not np.array_equal(correct, _render(info, **kwargs)):
            evidence.add(name)
    differs('multiplicity', word=tuple(dict.fromkeys(info['word'])))
    differs('bend_phase', reset_bends=True)
    differs('marker_phase', consume_contacts=True)
    differs('prefix_phase', restart_prefix=True)
    differs('full_trace', total=len(info['trace']['path']))
    differs('bars_as_units', total=info['d']+len(info['src']))
    differs('tallest_source', total=info['d']+max(s['count'] for s in info['src']))
    differs('include_unmarked', total=info['d']+sum(s['count'] for s in info['stacks']))
    differs('double_destination', total=2*info['d'])
    differs('replace_destination', total=info['supply'])
    differs('multiply', total=info['d']*info['supply'])
    if info['total'] >= 2*len(info['word']):
        evidence.add('cycling')
    if any(need <= info['total'] for need, p in info['trace']['contacts']):
        evidence.add('contact')
    if len({s['count'] for s in info['src']}) > 1:
        evidence.add('unequal_sources')
    if any(s['color'] == t['color'] for s in info['src'] for t in info['stacks'] if not t['controlled']):
        evidence.add('same_color_unmarked')
    if info['total'] < len(info['trace']['path']) and info['total'] not in set(np.cumsum(info['trace']['legs']).tolist()):
        evidence.add('mid_leg_stop')
    return evidence


_REQUIRED = {'multiplicity', 'bend_phase', 'marker_phase', 'prefix_phase',
             'full_trace', 'bars_as_units', 'tallest_source', 'include_unmarked',
             'double_destination', 'replace_destination', 'multiply', 'cycling',
             'contact', 'unequal_sources', 'same_color_unmarked', 'mid_leg_stop'}


def _sample_mask(rng):
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    mask = np.ones((h, w), bool)
    if rng.random() < .6:
        # A connected L spine plus random full-span adjoining cells.
        mask[:] = False
        mask[-1, :] = True
        mask[:, 0] = True
        for r in range(h-1):
            mask[r, :int(rng.integers(1, w+1))] = True
        if rng.integers(2):
            mask = mask[:, ::-1]
        if rng.integers(2):
            mask = mask[::-1]
    return mask.copy()


def _word(rng, pool, informative):
    n = int(rng.integers(3, 6) if informative else rng.integers(2, 7))
    distinct = int(rng.integers(2, n if informative else min(n, len(pool))+1))
    colors = list(map(int, rng.choice(pool, distinct, replace=False)))
    seq = colors+list(map(int, rng.choice(colors, n-distinct)))
    rng.shuffle(seq)
    return tuple(seq)


def _profile(rng, n, low, high):
    knots = np.unique(np.r_[0, rng.choice(np.arange(1, n-1), min(6, n-2), replace=False), n-1])
    return np.rint(np.interp(np.arange(n), knots, rng.uniform(low, high, len(knots)))).astype(int)


def _make_pair(rng, roles, informative=False, operand=None, matching=False, test=False, prefix_cap=6):
    bg, terrain, marker = roles
    pool = [c for c in range(10) if c not in roles]
    word = _word(rng, pool, informative)
    if operand is None:
        ns = int(rng.integers(1, 5 if test else 4))
        counts = list(map(int, rng.integers(1, 4 if test else 5, ns)))
        d = int(rng.integers(prefix_cap+1, 9) if test else rng.integers(1, prefix_cap+1))
    else:
        counts, d = operand
        counts = list(counts)
        ns = len(counts)
    supply, total = sum(counts), sum(counts)+d
    if informative:
        # Reject an intrinsically uninformative word before searching terrain.
        # For example ABAB cannot distinguish multiplicity, and a prefix of
        # exactly one full period cannot distinguish restarting the phase.
        for _ in range(64):
            unique = tuple(dict.fromkeys(word))
            correct = [word[i % len(word)] for i in range(total)]
            deduplicated = [unique[i % len(unique)] for i in range(total)]
            restart = correct[:d]+[word[i % len(word)] for i in range(supply)]
            if total >= 2*len(word) and correct != deduplicated and correct != restart:
                break
            word = _word(rng, pool, True)
        else:
            raise RuntimeError("No informative program for these operands")
    nu = int(rng.integers(1 if matching else 0, 3))
    entries = [(count, True) for count in counts]+[(int(rng.integers(1, 5)), False) for _ in range(nu)]
    rng.shuffle(entries)
    masks = [_sample_mask(rng) for _ in entries]
    pitches = [m.shape[0]+int(rng.integers(1, 3)) for m in masks]
    gaps = [int(rng.integers(1, 4)) if controlled else 0 for _, controlled in entries]
    spans = [m.shape[0]+(n-1)*pitch+(gap+1 if controlled else 0)
             for (n, controlled), m, pitch, gap in zip(entries, masks, pitches, gaps)]
    row = int(rng.integers(1, 4))
    swatch_gaps = list(map(int, rng.integers(1, 3, len(word)-1)))
    marker_gap = int(rng.integers(4, 7))
    header_width = 2*len(word)+sum(swatch_gaps)+marker_gap+2
    while header_width > 29:
        if marker_gap > 4:
            marker_gap -= 1
        else:
            swatch_gaps[swatch_gaps.index(2)] = 1
        header_width = 2*len(word)+sum(swatch_gaps)+marker_gap+2
    stack_width = sum(m.shape[1] for m in masks)+len(entries)-1
    w = int(rng.integers(max(header_width+1, stack_width, 18), 31))
    hmin = max(19 if not test else 22, row+7+max(spans))
    if hmin > 30:
        # A tall silhouette and count combination needs tighter visible pitch.
        pitches = [m.shape[0]+1 for m in masks]
        gaps = [1 if controlled else 0 for _, controlled in entries]
        spans = [m.shape[0]+(n-1)*pitch+(gap+1 if controlled else 0)
                 for (n, controlled), m, pitch, gap in zip(entries, masks, pitches, gaps)]
        hmin = max(19 if not test else 22, row+6+max(spans))
    assert hmin <= 30
    h = int(rng.integers(hmin, 31))
    # Horizontal order is independent of role; unused slack becomes varied bays.
    extra = rng.multinomial(w-stack_width, np.full(len(entries)+1, 1/(len(entries)+1)))
    xs, x = [], int(extra[0])
    for i, m in enumerate(masks):
        xs.append(x)
        x += m.shape[1]+(1+int(extra[i+1]) if i+1 < len(masks) else 0)
    colors = list(map(int, rng.choice(pool, len(entries), replace=True)))
    if matching:
        src_index = next(i for i, e in enumerate(entries) if e[1])
        idle_index = next(i for i, e in enumerate(entries) if not e[1])
        colors[idle_index] = colors[src_index]
    inp = np.full((h, w), bg, np.int8)
    allowed = np.ones((h, w), bool)
    allowed[:row+4] = False
    records = []
    for (n, controlled), m, pitch, gap, x, color, span in zip(entries, masks, pitches, gaps, xs, colors, spans):
        # Short stacks can sit in side bays or islands high in the scene.
        low, high = row+5, h-span
        if low > high:
            low = high
        y = int(rng.integers(max(low, high-4), high+1)) if rng.random() < .7 else int(rng.integers(low, high+1))
        cells = []
        for k in range(n):
            for rr, cc in np.argwhere(m):
                p = (y+k*pitch+int(rr), x+int(cc))
                inp[p] = color
                cells.append(p)
        bar = ()
        if controlled:
            by = y+(n-1)*pitch+m.shape[0]+gap
            bar = tuple((by, c) for c in range(x, x+m.shape[1]))
            for p in bar:
                inp[p] = marker
        allowed[max(0, y-1):min(h, y+span+1), max(0, x-1):min(w, x+m.shape[1]+1)] = False
        records.append((controlled, cells, bar))
    left = int(rng.integers(0, w-header_width+1))
    cols, x = [], left
    for i in range(len(word)):
        cols.append(x)
        x += 2+(swatch_gaps[i] if i < len(swatch_gaps) else marker_gap)
    annotations = []
    for c, color in zip(cols+[x], word+(marker,)):
        inp[row:row+2, c:c+2] = color
        annotations.extend((rr, cc) for rr in (row, row+1) for cc in (c, c+1))
    # Procedural broad terrain with independent sides and supply-shaped bays.
    for attempt in range(100):
        mask = np.zeros((h, w), bool)
        lefts = _profile(rng, h, -w*.25, w*.25)
        rights = _profile(rng, h, w*.45, w*1.15)
        bottoms = _profile(rng, w, h*.4, h*.97)
        for r in range(row+4, h):
            mask[r, max(0, lefts[r]):min(w, rights[r]+1)] = True
        mask &= np.arange(h)[:, None] <= bottoms[None, :]
        for _ in range(int(rng.integers(0, 5))):
            rr, cc = int(rng.integers(row+4, h)), int(rng.integers(w))
            mask[rr:min(h, rr+int(rng.integers(1, 5))), cc:min(w, cc+int(rng.integers(1, 6)))] = False
        mask &= allowed
        components = _components(mask)
        if not components:
            continue
        biggest = max(components, key=len)
        mask[:] = False
        for p in biggest:
            mask[p] = True
        if len(biggest) < max(35, total+12):
            continue
        candidates = []
        for r, c in biggest:
            if c+1 >= w or not mask[r, c+1] or (c and mask[r, c-1]):
                continue
            trace = _trace(mask, (r, c))
            if len(trace['path']) < total or (informative and len(trace['path']) < total+2):
                continue
            if informative and not any(d < need < total for need, p in trace['contacts']):
                continue
            candidates.append(((r, c), trace))
        rng.shuffle(candidates)
        for start, trace in candidates:
            a = inp.copy()
            a[mask] = terrain
            for phase, p in enumerate(trace['path'][:d]):
                a[p] = word[phase % len(word)]
            # Ground truth uses the sampled integer counts and constructed path.
            out = inp.copy()
            out[mask] = terrain
            for p in annotations:
                out[p] = bg
            for controlled, cells, bar in records:
                if controlled:
                    for p in cells:
                        out[p] = marker
                    for p in bar:
                        out[p] = bg
            for phase, p in enumerate(trace['path'][:total]):
                out[p] = word[phase % len(word)]
            for need, p in trace['contacts']:
                if need <= total:
                    out[p] = marker
            try:
                info = _parse(a)
            except AssertionError:
                continue
            assert info['supply'] == supply and info['d'] == d
            assert np.array_equal(_render(info), out)
            if informative and not {'bend_phase', 'marker_phase', 'prefix_phase', 'multiplicity', 'mid_leg_stop'} <= _evidence(info):
                continue
            return dict(input=a, output=out)
    raise RuntimeError("No compatible terrain in bounded construction")


def _key(grid):
    a = np.asarray(grid, np.int8)
    return a.shape, a.tobytes()


def _operands(rng, prefix_cap):
    options = []
    for _ in range(150):
        ns = int(rng.integers(1, 4))
        src = tuple(map(int, rng.integers(1, 5, ns)))
        d = int(rng.integers(1, prefix_cap+1))
        if 8 <= sum(src)+d <= 18:
            options.append((src, d))
    rng.shuffle(options)
    for a in options:
        for b in options:
            sa, da, sb, db = sum(a[0]), a[1], sum(b[0]), b[1]
            if sa == sb or da == db or sa*db == sb*da:
                continue
            if not (len(set(a[0])) > 1 or len(set(b[0])) > 1):
                continue
            if da*sa == da+sa and db*sb == db+sb:
                continue
            return a, b
    raise AssertionError("Could not construct informative operands")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    prefix_cap = int(rng.integers(3, 7))
    anchors = _operands(rng, prefix_cap)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            informative = split == 'train' and i < 2
            for retry in range(30):
                try:
                    pair = _make_pair(rng, roles, informative,
                                      anchors[i] if informative else None,
                                      matching=informative and i == 0,
                                      test=split == 'test', prefix_cap=prefix_cap)
                except RuntimeError:
                    continue
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("Bounded generation exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, roles, operands = set(), set(), set(), []
    prefix_lengths = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _parse(a)
            prefix_lengths[split].append(info['d'])
            roles.add((info['bg'], info['terrain'], info['marker']))
            assert np.array_equal(_render(info), b)
            assert len(_components(info['mask'])) == 1
            # Conservation: exactly one newly occupied route cell per source token.
            assert sum(b[p] in info['word'] for p in info['trace']['path']) == info['total']
            assert all(b[p] == info['marker'] for s in info['src'] for p in s['cells'])
            assert all(b[p] == info['bg'] for s in info['src'] for p in s['bar']['cells'])
            assert all(b[p] == a[p] for s in info['stacks'] if not s['controlled'] for p in s['cells'])
            changed = info['annotations'] | set(info['trace']['path'][:info['total']])
            changed |= {p for n, p in info['trace']['contacts'] if n <= info['total']}
            changed |= {p for s in info['src'] for p in s['cells']+s['bar']['cells']}
            assert all(a[r, c] == b[r, c] for r in range(a.shape[0]) for c in range(a.shape[1]) if (r, c) not in changed)
            if split == 'train':
                evidence |= _evidence(info)
                operands.append((info['supply'], info['d']))
    assert len(roles) == 1
    assert max(prefix_lengths['train']) < min(prefix_lengths['test'])
    assert _REQUIRED <= evidence, _REQUIRED-evidence
    assert len({s for s, d in operands}) > 1 and len({d for s, d in operands}) > 1
    s0, d0 = operands[0]
    assert any(s*d0 != s0*d for s, d in operands), "Training admits a fixed destination multiplier"


def self_test():
    """Bounded semantic, adversarial, count-contract and 42-seed variation checks."""
    # Explicit canvas exit with unused terrain around its corner.
    mask = np.zeros((5, 8), bool)
    mask[3, :] = True
    mask[2:, 7] = True
    record = _trace(mask, (3, 0))
    assert record['path'] == tuple((3, c) for c in range(8)) and not record['contacts']
    assert record['end'] == 'edge' and (2, 7) not in record['path']
    # Vertical exit is equally terminal, despite terrain to its right.
    mask = np.zeros((8, 10), bool)
    mask[6, :4] = True
    mask[:7, 3] = True
    mask[0, 3:9] = True
    record = _trace(mask, (6, 0))
    assert record['end'] == 'edge' and record['path'][-1] == (0, 3)
    assert (0, 4) not in record['path'] and record['contacts'] == ((4, (6, 4)),)
    # Literal input fixture: 2 existing units + 4 tokens reach a six-cell leg
    # endpoint exactly. The contact appears, but the next leg stays untouched.
    fixture = np.zeros((24, 18), np.int8)
    for x, color in ((1, 2), (4, 2), (7, 3), (13, 4)):
        fixture[1:3, x:x+2] = color
    fixture[5:13, :6] = 1
    fixture[12, :2] = 2
    for y in (9, 12, 15, 18):
        fixture[y:y+2, 10:12] = 7
    fixture[21, 10:12] = 4
    answer = solve(fixture)
    assert answer[12, 6] == 4 and answer[11, 5] == 1
    assert tuple(answer[12, :6]) == (2, 2, 3, 2, 2, 3)
    less = fixture.copy()
    less[9:11, 10:12] = 0
    answer = solve(less)
    assert answer[12, 6] == 0 and answer[12, 5] == 1
    more = fixture.copy()
    more[6:8, 10:12] = 7
    answer = solve(more)
    assert answer[12, 6] == 4 and answer[11, 5] == 2
    shapes, silhouettes, pitches, words, counts, starts, structures, palettes = (set() for _ in range(8))
    puzzles = set()
    partial = exact_contact = full = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((_key(p['input']), _key(p['output'])) for split in ('train', 'test') for p in puzzle[split])
        puzzles.add(signature)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                info = _parse(pair['input'])
                shapes.add(pair['input'].shape)
                palettes.add((info['bg'], info['terrain'], info['marker']))
                label = {}
                words.add(tuple(label.setdefault(c, len(label)) for c in info['word']))
                structures.add((info['mask'].shape, info['mask'].tobytes(), info['d'], info['supply']))
                counts.add(len(info['src']))
                starts.add(info['start'])
                partial += info['total'] < len(info['trace']['path'])
                full += info['total'] == len(info['trace']['path'])
                exact_contact += any(n == info['total'] for n, p in info['trace']['contacts'])
                for stack in info['stacks']:
                    token = stack['tokens'][0]
                    silhouettes.add(_key(token['mask']))
                    if stack['count'] > 1:
                        pitches.add(stack['tokens'][1]['y']-token['y'])
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 25 and len(palettes) >= 30
    assert len(silhouettes) >= 12 and len(pitches) >= 3
    assert len(words) >= 15 and len(starts) >= 40
    assert {1, 2, 3, 4} <= counts and partial >= 100 and exact_contact and full
    a, b = generate_puzzle(811, 2, 1), generate_puzzle(811, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for split in a for p, q in zip(a[split], b[split]) for k in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(900+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # All rival-rule outputs must be rejected as a changed demonstration.
    original = generate_puzzle(18, 2, 1)
    variants = [dict(total=None, reset_bends=True), dict(total=None, consume_contacts=True),
                dict(restart_prefix=True), dict(word='deduplicate'), dict(total='bars'),
                dict(total='all_stacks'), dict(total='full_trace')]
    for variant in variants:
        for index, pair in enumerate(original['train']):
            info = _parse(pair['input'])
            kwargs = dict(variant)
            if kwargs.get('word') == 'deduplicate':
                kwargs['word'] = tuple(dict.fromkeys(info['word']))
            if kwargs.get('total') == 'bars':
                kwargs['total'] = info['d']+len(info['src'])
            elif kwargs.get('total') == 'all_stacks':
                kwargs['total'] = info['d']+sum(s['count'] for s in info['stacks'])
            elif kwargs.get('total') == 'full_trace':
                kwargs['total'] = len(info['trace']['path'])
            wrong = _render(info, **kwargs)
            if np.array_equal(wrong, pair['output']):
                continue
            broken = {s: [{k: np.asarray(g).copy() for k, g in p.items()} for p in original[s]] for s in original}
            broken['train'][index]['output'] = wrong
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                break
            raise AssertionError("Accepted a semantic mutation")
        else:
            raise AssertionError("No demonstration separates a rival rule")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
