"""Routed caucus feedback. Only NumPy and the Python standard library are used.

Construction writes answers from explicit cable records. solve instead recognizes
panels, socket groups, and a directional graph in the query input alone.
"""
from itertools import permutations
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
_CHECKER = ~_DIAMOND


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid of size 1..30')
    return a.astype(np.int8, copy=True)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        found, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    todo.append(q)
        result.append(found)
    return result


def _majority(labels):
    values, counts = np.unique(labels, return_counts=True)
    m = counts.max()
    assert np.sum(counts == m) == 1, 'a ballot cannot tie'
    return int(values[np.argmax(counts)])


def _trace(g, start, end, sources, receivers, carrier, bridge):
    """Trace individual strands; a marked four-way crossing is never a join."""
    visible = set(map(tuple, np.argwhere(g == carrier)))
    if bridge is not None:
        visible |= set(map(tuple, np.argwhere(g == bridge)))
    ends = {(start, c): ('source', i) for i, c in enumerate(sources)}
    ends.update({(end, c): ('receiver', j) for j, c in enumerate(receivers)})
    edges = {}
    for p in visible:
        r, c = p
        e = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(_DIRS)
             if (r + dr, c + dc) in visible}
        if p in ends:
            assert set(e) == ({2} if ends[p][0] == 'source' else {0})
            assert g[p] == carrier
        elif bridge is not None and g[p] == bridge:
            assert set(e) == {0, 1, 2, 3}, 'crossing needs four straight arms'
        else:
            assert len(e) == 2, 'ordinary wires neither branch nor break'
        edges[p] = e
    pairing, paths, used = [], [], set()
    for i, c in enumerate(sources):
        p, previous_direction = (start, c), 0
        seen, path = set(), []
        while True:
            state = p, previous_direction
            assert state not in seen, 'wire cycle'
            seen.add(state)
            path.append(p)
            used.add(p)
            if p in ends and p != (start, c):
                bank, j = ends[p]
                assert bank == 'receiver', 'wire returns to source bank'
                pairing.append(j)
                break
            e = edges[p]
            if p == (start, c):
                direction = 2
            elif len(e) == 4:
                direction = (previous_direction + 2) % 4
            else:
                assert previous_direction in e
                direction = next(d for d in e if d != previous_direction)
            p = e[direction]
            previous_direction = (direction + 2) % 4
        paths.append(path)
    assert sorted(pairing) == list(range(len(receivers))), 'nonbijective cables'
    assert used == visible, 'unattached wire fragment'
    return pairing, paths


def _parse_canonical(g):
    bg = int(g[0, 0])
    assert all(int(g[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1)))
    occupied = np.argwhere(g != bg)
    assert len(occupied)
    bar_row = int(occupied[:, 0].max())
    bar_colors = set(map(int, g[bar_row])) - {bg}
    assert len(bar_colors) == 1
    bar = bar_colors.pop()
    assert np.all(np.argwhere(g == bar)[:, 0] == bar_row)
    end = bar_row - 4
    assert 5 <= end < g.shape[0] - 4
    end_colors = set(map(int, g[end])) - {bg}
    assert len(end_colors) == 1
    carrier = end_colors.pop()
    assert carrier != bar
    start = int(np.argwhere(g == carrier)[:, 0].min())
    assert 4 <= start < end
    sources = list(map(int, np.flatnonzero(g[start] == carrier)))
    receivers = list(map(int, np.flatnonzero(g[end] == carrier)))
    n = len(sources)
    assert 4 <= n <= 7 and len(receivers) == n
    for centers in (sources, receivers):
        assert centers[0] >= 1 and centers[-1] + 1 < g.shape[1]
        assert all(b - a >= 4 for a, b in zip(centers, centers[1:]))
    signals = tuple(sorted(set(map(int, np.unique(g[:start - 3]))) - {bg}))
    assert len(signals) == 2 and not set(signals) & {carrier, bar}
    signal_cells = set(map(tuple, np.argwhere(np.isin(g, signals))))
    panels = []
    source_owner = [-1] * n
    for cells in _components(signal_cells):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        assert r1 == start - 4 and len(cells) == (r1-r0+1)*(c1-c0+1)
        colors, counts = np.unique(g[r0:r1+1, c0:c1+1], return_counts=True)
        assert tuple(map(int, colors)) == signals and counts[0] != counts[1]
        members = [i for i, c in enumerate(sources) if c0 <= c-1 and c+1 <= c1]
        assert members and c0 == sources[members[0]]-1 and c1 == sources[members[-1]]+1
        owner = len(panels)
        for i in members:
            assert source_owner[i] == -1
            source_owner[i] = owner
        panels.append(dict(box=(r0, r1, c0, c1), cells=cells, members=members,
                           majority=int(colors[np.argmax(counts)]), area=len(cells),
                           counts=tuple(map(int, counts))))
    assert 2 <= len(panels) <= 4 and -1 not in source_owner
    assert {p['majority'] for p in panels} == set(signals)
    groups = []
    receiver_owner = [-1] * n
    for cells in _components(map(tuple, np.argwhere(g == bar))):
        cols = sorted(c for r, c in cells)
        assert cols == list(range(cols[0], cols[-1]+1))
        members = [j for j, c in enumerate(receivers)
                   if cols[0] <= c-1 and c+1 <= cols[-1]]
        assert members and len(members) in (1, 3, 5)
        assert cols[0] == receivers[members[0]]-1 and cols[-1] == receivers[members[-1]]+1
        owner = len(groups)
        for j in members:
            assert receiver_owner[j] == -1
            receiver_owner[j] = owner
        groups.append(members)
    assert 2 <= len(groups) <= 5 and -1 not in receiver_owner
    assert any(len(group) > 1 for group in groups)
    footprint = np.zeros(g.shape, dtype=bool)
    for row, centers in ((start-3, sources), (end+1, receivers)):
        for c in centers:
            block = g[row:row+3, c-1:c+2]
            assert block.shape == (3, 3) and np.all(block == bg)
            footprint[row:row+3, c-1:c+2] = True
    field = g[start:end+1]
    aux = set(map(int, np.unique(field))) - {bg, carrier}
    assert len(aux) <= 1 and not aux & (set(signals) | {bar})
    bridge = next(iter(aux)) if aux else None
    allowed = np.zeros(g.shape, dtype=bool)
    allowed[start:end+1] = True
    allowed[bar_row] = g[bar_row] == bar
    for p in signal_cells:
        allowed[p] = True
    assert np.all(g[~allowed] == bg)
    pairing, paths = _trace(g, start, end, sources, receivers, carrier, bridge)
    votes = [panels[source_owner[i]]['majority'] for i in range(n)]
    incoming = [[] for _ in groups]
    for i, j in enumerate(pairing):
        incoming[receiver_owner[j]].append(i)
    decisions = [_majority([votes[i] for i in members]) for members in incoming]
    acknowledgments = [decisions[receiver_owner[pairing[i]]] for i in range(n)]
    return dict(grid=g, background=bg, carrier=carrier, bridge=bridge, bar=bar,
                signals=signals, start=start, end=end, sources=sources,
                receivers=receivers, panels=panels, source_owner=source_owner,
                groups=groups, receiver_owner=receiver_owner, pairing=pairing,
                paths=paths, votes=votes, incoming=incoming, decisions=decisions,
                acknowledgments=acknowledgments, footprint=footprint)


def _parse(input_grid):
    g = _grid(input_grid)
    candidates = []
    for transpose in (False, True):
        for reverse in (False, True):
            a = g.T if transpose else g
            a = a[::-1] if reverse else a
            try:
                info = _parse_canonical(a)
                info.update(transpose=transpose, reverse=reverse)
                candidates.append(info)
            except (AssertionError, IndexError, StopIteration):
                pass
    if len(candidates) != 1:
        raise ValueError(f'input must have one panel-to-caucus interpretation; got {len(candidates)}')
    return candidates[0]


def _restore(g, info):
    a = g[::-1] if info['reverse'] else g
    return (a.T if info['transpose'] else a).copy()


def _fill(info, decisions=None, pairing=None):
    g = info['grid'].copy()
    bg = info['background']
    decisions = info['decisions'] if decisions is None else decisions
    pairing = info['pairing'] if pairing is None else pairing
    for i, c in enumerate(info['sources']):
        color = decisions[info['receiver_owner'][pairing[i]]]
        g[info['start']-3:info['start'], c-1:c+2] = np.where(_DIAMOND, color, bg)
    for j, c in enumerate(info['receivers']):
        color = decisions[info['receiver_owner'][j]]
        g[info['end']+1:info['end']+4, c-1:c+2] = np.where(_CHECKER, color, bg)
    return _restore(g, info)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover voting, cable pairing, and returned decisions from this input."""
    return _fill(_parse(input_grid))


def _alternative(info, kind):
    # Counterfactual rules used only for evidence checks, never by solve.
    if kind == 'ordinal':
        pairing = list(range(len(info['sources'])))
        decisions = [_majority([info['votes'][i] for i in range(len(pairing))
                               if info['receiver_owner'][i] == k])
                     for k in range(len(info['groups']))]
        return _fill(info, decisions, pairing)
    decisions = []
    for members in info['incoming']:
        owners = sorted({info['source_owner'][i] for i in members})
        panels = [info['panels'][k] for k in owners]
        if kind == 'distinct_panels':
            ballot = [p['majority'] for p in panels]
        elif kind == 'raw_pixels':
            ballot = [int(info['grid'][cell]) for p in panels for cell in p['cells']]
        elif kind == 'minorities':
            ballot = [next(s for s in info['signals'] if s != info['votes'][i])
                      for i in members]
        else:
            raise ValueError(kind)
        try:
            decisions.append(_majority(ballot))
        except AssertionError:
            return None  # A tie in an incorrect rule is not counted as evidence.
    return _fill(info, decisions)


def _partitions(n, two_only=False):
    def build(total, prefix):
        if total == 0:
            return [prefix] if 2 <= len(prefix) <= 5 and max(prefix) > 1 else []
        return [p for k in (1, 3, 5) if k <= total
                for p in build(total-k, prefix+[k])]
    return [p for p in build(n, []) if not two_only or len(p) == 2]


def _source_weights(rng, n):
    count = int(rng.integers(2, min(4, n)+1))
    cuts = sorted(map(int, rng.choice(np.arange(1, n), count-1, replace=False)))
    return list(map(int, np.diff([0]+cuts+[n])))


def _pattern(rng, h, w, majority, minority, nminor=None, forced_column=None):
    area = h*w
    if nminor is None:
        band = int(rng.integers(3))
        nminor = ((area-1)//2 if band == 0 else 1 if band == 1
                  else max(1, int(area*rng.uniform(.18, .38))))
    nminor = min(int(nminor), (area-1)//2)
    rr, cc = np.indices((h, w))
    style = int(rng.integers(4))
    if style == 0:
        order = list(map(int, rng.permutation(area)))
    elif style == 1:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        order = list(map(int, np.argsort((abs(rr-r)+abs(cc-c)+rng.random((h,w))*.3).ravel())))
    elif style == 2:
        order = list(map(int, np.argsort((rr*w+cc if rng.random()<.5 else cc*h+rr).ravel())))
    else:
        order = list(map(int, np.argsort(((rr+cc)%2*area+rr*w+cc).ravel())))
    if forced_column is not None:
        forced = (h-1)*w+forced_column
        order.remove(forced)
        order.insert(0, forced)
    flat = np.full(area, majority, dtype=np.int8)
    flat[order[:nminor]] = minority
    return flat.reshape(h, w)


def _draw_routes(rng, n, perm, depths):
    """An interleaved orthogonal crossbar, with a separate bend row per cable.

    Input and output track columns are disjoint, so arbitrary permutations fit
    without coincident vertical runs. Shared cells mark actual crossings.
    """
    pitches = [p for p in (4, 6) if (n-1)*p+p//2+4 <= 30]
    pitch = int(rng.choice(pitches))
    offset = pitch//2
    minimum_width = (n-1)*pitch+offset+4
    extra_width = int(rng.integers(31-minimum_width))
    left = int(rng.integers(extra_width+1))
    sources = [1+left+i*pitch for i in range(n)]
    receivers = [c+offset for c in sources]
    width = minimum_width+extra_width
    pieces = [1, 1] + [2]*(n-1) + [1, 1]  # lead, tail, gaps, exterior margins
    minimum_height = max(depths)+10+2*n
    assert minimum_height <= 30
    extra = int(rng.integers(31-minimum_height))
    for _ in range(extra):
        pieces[int(rng.integers(len(pieces)))] += 1
    lead, tail, top, bottom = pieces[0], pieces[1], pieces[-2], pieces[-1]
    start = top+max(depths)+3
    row_sites = [start+lead]
    for gap in pieces[2:-2]:
        row_sites.append(row_sites[-1]+gap)
    end = row_sites[-1]+tail
    rows = [row_sites[int(k)] for k in rng.permutation(n)]
    paths, visits = [], {}
    for i, j in enumerate(perm):
        a, b, r = sources[i], receivers[j], rows[i]
        step = 1 if b > a else -1
        path = ([(x, a) for x in range(start, r+1)]
                + [(r, c) for c in range(a+step, b+step, step)]
                + [(x, b) for x in range(r+1, end+1)])
        paths.append(path)
        for p in path:
            visits.setdefault(p, set()).add(i)
    field = np.zeros((end+5+bottom, width), dtype=np.int8)
    for p, owners in visits.items():
        assert 1 <= len(owners) <= 2
        field[p] = 1 if len(owners) == 1 else 2
    return field, sources, receivers, start, end, paths


def _make_pair(rng, palette, lesson=None, regime=None, test=False):
    if lesson in ('weighted', 'three'):
        weights = [3, 2, 1] if lesson == 'weighted' else [2, 1, 1]
        rng.shuffle(weights)
        n = sum(weights)
        vote_bit = int(rng.integers(2))
        dominant = weights.index(max(weights))
        bits = [vote_bit if k == dominant else 1-vote_bit for k in range(3)]
        receiver_sizes = [5, 1] if lesson == 'weighted' else [3, 1]
        rng.shuffle(receiver_sizes)
    else:
        n = (7 if test and regime == 'bundle' else int(rng.choice([5, 7]))
             if test and regime == 'partition' else int(rng.choice([4, 6]))
             if regime == 'partition' else int(rng.integers(4, 7)))
        weights = _source_weights(rng, n)
        bits = list(map(int, rng.integers(2, size=len(weights))))
        if len(set(bits)) == 1:
            bits[int(rng.integers(len(bits)))] ^= 1
        options = _partitions(n, two_only=regime == 'partition' and not test)
        receiver_sizes = options[int(rng.integers(len(options)))]
    source_owner = [k for k, weight in enumerate(weights) for _ in range(weight)]
    receiver_owner = [k for k, count in enumerate(receiver_sizes) for _ in range(count)]
    perm = list(map(int, rng.permutation(n)))
    if lesson in ('weighted', 'three'):
        single = receiver_owner.index(receiver_sizes.index(1))
        if lesson == 'weighted':
            eligible = [i for i, k in enumerate(source_owner) if weights[k] == 2]
        else:
            eligible = [i for i, k in enumerate(source_owner) if weights[k] == 1]
        eligible = [i for i in eligible if i != single]
        chosen = int(rng.choice(eligible))
        old = perm.index(single)
        perm[chosen], perm[old] = perm[old], perm[chosen]
    if lesson == 'weighted':
        depths = [int(rng.integers(1, 3)) if k == dominant else int(rng.integers(2, 5))
                  for k in range(len(weights))]
    else:
        lo, hi = (4, 6) if test and regime == 'area' else (1, 4)
        depths = list(map(int, rng.integers(lo, hi+1, size=len(weights))))
    field, sources, receivers, start, end, paths = _draw_routes(rng, n, perm, depths)
    bg, carrier, bridge, bar, s0, s1 = palette
    inp = np.array([bg, carrier, bridge], dtype=np.int8)[field]
    out = inp.copy()
    cursor = 0
    panel_labels = []
    for k, (weight, depth, bit) in enumerate(zip(weights, depths, bits)):
        c0, c1 = sources[cursor]-1, sources[cursor+weight-1]+1
        r0, r1 = start-3-depth, start-4
        majority, minority = (s0, s1) if bit == 0 else (s1, s0)
        area = depth*(c1-c0+1)
        nminor = ((area-1)//2 if k == dominant else 1) if lesson == 'weighted' else None
        forced = 1 if lesson == 'weighted' and k == dominant else None
        pattern = _pattern(rng, depth, c1-c0+1, majority, minority, nminor, forced)
        inp[r0:r1+1, c0:c1+1] = pattern
        out[r0:r1+1, c0:c1+1] = pattern
        panel_labels.append(majority)
        cursor += weight
    cursor = 0
    for count in receiver_sizes:
        c0, c1 = receivers[cursor]-1, receivers[cursor+count-1]+1
        inp[end+4, c0:c1+1] = bar
        out[end+4, c0:c1+1] = bar
        cursor += count
    # Answers use the explicit construction permutation, not the visible tracer.
    votes = [panel_labels[k] for k in source_owner]
    decisions = [_majority([votes[i] for i, j in enumerate(perm) if receiver_owner[j] == k])
                 for k in range(len(receiver_sizes))]
    for i, c in enumerate(sources):
        color = decisions[receiver_owner[perm[i]]]
        out[start-3:start, c-1:c+2] = np.where(_DIAMOND, color, bg)
    for j, c in enumerate(receivers):
        color = decisions[receiver_owner[j]]
        out[end+1:end+4, c-1:c+2] = np.where(_CHECKER, color, bg)
    rotation = int(rng.integers(4))
    reflection = bool(rng.integers(2))
    result = {}
    for key, g in (('input', inp), ('output', out)):
        a = np.rot90(g, rotation)
        result[key] = (a[:, ::-1] if reflection else a).copy()
    return result


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['bundle', 'area', 'partition', 'palette']))
    palette = list(map(int, rng.choice(10, 6, replace=False)))
    stable = bool(rng.random() < .3) or regime == 'palette'
    lesson_slots = list(map(int, rng.choice(ntrain, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', ntrain), ('test', ntest)):
        for i in range(count):
            lesson = ('weighted' if i == lesson_slots[0] else 'three'
                      if i == lesson_slots[1] else None) if split == 'train' else None
            for _ in range(64):
                roles = (palette if stable and not (split == 'test' and regime == 'palette')
                         else list(map(int, rng.choice(10, 6, replace=False))))
                if split == 'test' and regime == 'palette' and roles == palette:
                    continue
                pair = _make_pair(rng, roles, lesson, regime, split == 'test')
                signature = _key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('bounded distinct-input sampling exhausted')
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(_fill(info), out), 'input-only answer disagrees'
            canonical_out = out.T if info['transpose'] else out
            canonical_out = canonical_out[::-1] if info['reverse'] else canonical_out
            mask = info['footprint']
            assert np.array_equal(info['grid'][~mask], canonical_out[~mask]), 'nonterminal mutation'
            # Explicit local minorities, strictly odd ballots, and port replication.
            for p in info['panels']:
                assert min(p['counts']) > 0 and p['counts'][0] != p['counts'][1]
            assert all(len(members) % 2 for members in info['incoming'])
            if split == 'train':
                for kind in ('distinct_panels', 'raw_pixels', 'ordinal', 'minorities'):
                    alternative = _alternative(info, kind)
                    if alternative is not None and not np.array_equal(alternative, out):
                        evidence.add(kind)
                if any(len({info['acknowledgments'][i] for i in p['members']}) > 1
                       for p in info['panels']):
                    evidence.add('split_feedback')
                if info['bridge'] is not None:
                    evidence.add('crossings')
                if len(set(info['decisions'])) == 2:
                    evidence.add('both_decisions')
                for p in info['panels']:
                    r1 = p['box'][1]
                    if any(info['grid'][r1, info['sources'][i]] != p['majority'] for i in p['members']):
                        evidence.add('minority_contact')
                evidence.update(f'ports_{len(group)}' for group in info['groups'])
    required = {'distinct_panels', 'raw_pixels', 'ordinal', 'minorities',
                'split_feedback', 'crossings', 'both_decisions', 'minority_contact',
                'ports_1', 'ports_3', 'ports_5'}
    assert required <= evidence, f'inadequate demonstrations: {required-evidence}'


def self_test():
    """Bounded semantic, variation, metamorphic, and corruption checks."""
    shapes, palettes, weights, partitions, pairings, structures = (set() for _ in range(6))
    signatures, margins, areas, directions = set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add((info['background'], info['carrier'], info['bar'], info['signals']))
                weights.add(tuple(len(p['members']) for p in info['panels']))
                partitions.add(tuple(len(g) for g in info['groups']))
                pairings.add(tuple(info['pairing']))
                directions.add((info['transpose'], info['reverse']))
                areas.update(p['area'] for p in info['panels'])
                for members in info['incoming']:
                    labels = [info['votes'][i] for i in members]
                    margins.add(abs(labels.count(info['signals'][0])-labels.count(info['signals'][1])))
                role_map = {}
                structures.add((pair['input'].shape, tuple(role_map.setdefault(int(c), len(role_map))
                                                           for c in pair['input'].flat)))
                pairs += 1
    assert len(signatures) == 42 and len(structures) > 180
    assert len(shapes) > 30 and len(palettes) > 30 and len(pairings) > 50
    assert len(weights) > 15 and len(partitions) > 10 and len(areas) > 15
    assert directions == {(False, False), (False, True), (True, False), (True, True)}
    assert {1, 3, 5} <= margins
    # Exhaustive five-strand permutations exercise routing independent of panels.
    rng = np.random.default_rng(9001)
    for perm in permutations(range(5)):
        field, sources, receivers, start, end, _ = _draw_routes(rng, 5, list(perm), [2, 3])
        pairing, _ = _trace(field, start, end, sources, receivers, 1, 2 if np.any(field == 2) else None)
        assert pairing == list(perm)
    p = generate_puzzle(123)
    assert all(np.array_equal(a[k], b[k]) for s in p
               for a, b in zip(p[s], generate_puzzle(123)[s]) for k in a)
    # Rearrange votes inside one panel. Counts and all returned decisions persist.
    pair = p['train'][0]
    info = _parse(pair['input'])
    g = info['grid'].copy()
    r0, r1, c0, c1 = info['panels'][0]['box']
    block = g[r0:r1+1, c0:c1+1]
    block[:] = np.sort(block, axis=None).reshape(block.shape)
    changed = _restore(g, info)
    expected = pair['output'].copy()
    out_g = expected.T if info['transpose'] else expected
    out_g = out_g[::-1] if info['reverse'] else out_g
    out_g[r0:r1+1, c0:c1+1] = block
    assert np.array_equal(solve(changed), expected)
    # Corrupting a preserved wire cell or a decision must be caught.
    for region in ('wire', 'socket'):
        q = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        first = q['train'][0]
        inf = _parse(first['input'])
        a = first['output'].T if inf['transpose'] else first['output']
        a = a[::-1] if inf['reverse'] else a
        cell = (inf['start'], inf['sources'][0]) if region == 'wire' else (inf['start']-2, inf['sources'][0]-1)
        a[cell] = (int(a[cell])+1) % 10
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError('output corruption accepted')
    for a, b in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4.0,1), (4,None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    for a, b in ((2,1), (10,4)):
        validate_generated_puzzle(generate_puzzle(91, a, b))
    return {'seeds': 42, 'pairs': pairs, 'shapes': len(shapes), 'palettes': len(palettes),
            'source_weight_profiles': len(weights), 'caucus_partitions': len(partitions),
            'pairings': len(pairings), 'normalized_structures': len(structures),
            'routing_permutations_checked': 120}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
