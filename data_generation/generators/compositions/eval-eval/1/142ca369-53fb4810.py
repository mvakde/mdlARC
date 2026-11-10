"""Clock-keyed point mirrors: a cyclic word controls a diagonal ray's turns.

Only the standard library and NumPy are used. No files, parent generators,
generation metadata, or random state are consulted by solve(). Diagnostics are
in memory; this module writes neither puzzles nor execution traces.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "integer cells required"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors required"
    return g.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.uint8).tobytes()


def _components(g, bg):
    todo = set(map(tuple, np.argwhere(g != bg)))
    groups = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in CARDINALS:
                p = r + dr, c + dc
                if p in todo and g[p] == g[start]:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _parse(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > g.size / 2, "unique majority background required"
    sources = []
    for cells in _components(g, bg):
        rs, cs = zip(*cells)
        if len(cells) != 3 or max(rs)-min(rs) != 1 or max(cs)-min(cs) != 1:
            continue
        missing = next(iter({(r, c) for r in (min(rs), max(rs))
                             for c in (min(cs), max(cs))} - cells))
        elbow = max(rs)+min(rs)-missing[0], max(cs)+min(cs)-missing[1]
        d = elbow[0]-missing[0], elbow[1]-missing[1]
        sources.append(dict(elbow=elbow, direction=d, cells=cells, ink=int(g[elbow])))
    sources.sort(key=lambda s: s['elbow'])
    assert sources, "missing L hub"
    inks = {s['ink'] for s in sources}
    assert len(inks) == 1, "one structural ink per input"
    ink = next(iter(inks))
    assert int(np.count_nonzero(g == ink)) == 3*len(sources), "ink reserved for L hubs"
    claimed = set().union(*(s['cells'] for s in sources))
    for s in sources:
        e, d = s['elbow'], s['direction']
        choices = []
        for axis in (0, 1):
            back = (-d[0], 0) if axis == 0 else (0, -d[1])
            cells, word = [], []
            for k in range(2, 32):
                p = e[0]+k*back[0], e[1]+k*back[1]
                if not _inside(p, g.shape) or g[p] == bg:
                    break
                assert int(g[p]) != ink, "tape runs into another hub"
                cells.append(p)
                word.append(int(g[p]))
            if cells:
                choices.append((axis, cells, word))
        assert len(choices) == 1, "exactly one backward arm carries a tape"
        axis, cells, word = choices[0]
        assert 2 <= len(word) <= 8, "complete tape length must be 2..8"
        assert not claimed.intersection(cells), "overlapping tapes"
        claimed.update(cells)
        s.update(tape=set(cells), word=tuple(word), tape_axis=axis)
    mirrors = {}
    for p in sorted(set(map(tuple, np.argwhere(g != bg))) - claimed):
        assert all(not _inside((p[0]+dr, p[1]+dc), g.shape)
                   or g[p[0]+dr, p[1]+dc] == bg for dr, dc in CARDINALS), "isolated point mirror required"
        mirrors[p] = int(g[p])
    return g, bg, sources, mirrors


def _word_variant(word, mode):
    if mode == 'reverse':
        return word[::-1]
    if mode == 'constant':
        return word[:1]
    if mode == 'prefix':
        # Deliberately wrong alternative: accepts a nondividing prefix period.
        p = next(p for p in range(1, len(word)+1)
                 if all(v == word[i % p] for i, v in enumerate(word)))
        return word[:p]
    return word


def _trace(input_grid, *, gate_mode='matching', word_mode='whole', reset=False,
           strict=True):
    g, bg, sources, mirrors = _parse(input_grid)
    answer = g.copy()
    writes, used, routes = {}, set(), []
    for source in sources:
        word = _word_variant(source['word'], word_mode)
        e, d = source['elbow'], source['direction']
        p = e[0]+d[0], e[1]+d[1]
        cursor, t, states, steps = 0, 1, set(), []
        while _inside(p, g.shape):
            state = p, d, cursor % len(word)
            if state in states:
                assert not strict, "nonterminating clock route"
                break
            states.add(state)
            if strict:
                assert g[p] == bg, "ray strikes original material"
            color = word[cursor % len(word)]
            gates = []
            for axis in (0, 1):
                m = (p[0]+d[0], p[1]) if axis == 0 else (p[0], p[1]+d[1])
                if m in mirrors:
                    gates.append((m, axis))
            if strict:
                assert len(gates) <= 1, "competing faces at one contact"
            m, axis = gates[0] if gates else (None, None)
            hit = bool(m is not None and (gate_mode == 'always' or
                       (gate_mode == 'matching' and mirrors[m] == color) or
                       (gate_mode == 'next' and mirrors[m] == word[(cursor+1) % len(word)])))
            if m is not None:
                used.add(m)
            # Record incoming direction; color is due before any reflection.
            steps.append(dict(position=p, direction=d, time=t, color=color,
                              mirror=m, axis=axis, hit=hit))
            if strict:
                assert p not in writes or writes[p] == color, "conflicting ray colors"
            if g[p] == bg:
                writes[p] = color
                answer[p] = color
            if hit:
                d = (-d[0], d[1]) if axis == 0 else (d[0], -d[1])
                if reset:
                    cursor = -1
            cursor += 1
            t += 1
            p = p[0]+d[0], p[1]+d[1]
        routes.append(dict(source=source, steps=steps))
    if strict:
        assert used == set(mirrors), "unused mirrors are excluded from generated scenes"
    return answer, dict(background=bg, sources=sources, mirrors=mirrors,
                        routes=routes, writes=writes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the entire rule from this input, preserving every original cell."""
    return _trace(input_grid)[0]


def _stamp(elbow, direction, axis, word, ink):
    r, c = elbow
    dr, dc = direction
    result = {(r, c): ink, (r-dr, c): ink, (r, c-dc): ink}
    back = (-dr, 0) if axis == 0 else (0, -dc)
    for k, color in enumerate(word, 2):
        result[r+k*back[0], c+k*back[1]] = int(color)
    return result


def _clear_stamp(g, stamp, bg):
    for r, c in stamp:
        if not _inside((r, c), g.shape) or g[r, c] != bg:
            return False
        # Empty collars separate original apparatus; the attached tape is one object.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r+dr, c+dc
                if p not in stamp and _inside(p, g.shape) and g[p] != bg:
                    return False
    return True


def _add_source(g, rng, bg, ink, word, direction=None, min_steps=3):
    for _ in range(140):
        d = direction if direction is not None else DIAGONALS[int(rng.integers(4))]
        e = int(rng.integers(g.shape[0])), int(rng.integers(g.shape[1]))
        stamp = _stamp(e, d, int(rng.integers(2)), word, ink)
        if not _clear_stamp(g, stamp, bg):
            continue
        trial = g.copy()
        for p, value in stamp.items():
            trial[p] = value
        try:
            _, info = _trace(trial)
        except AssertionError:
            continue
        route = next(r for r in info['routes'] if r['source']['elbow'] == e)
        if len(route['steps']) >= min_steps:
            return trial, e
    return None


def _active_signature(route):
    return tuple(s['mirror'] for s in route['steps'] if s['hit'])


def _route_key(info):
    return tuple((r['source']['elbow'], tuple((s['position'], s['color'], s['direction'])
                  for s in r['steps'])) for r in info['routes'])


def _append_mirror(g, elbow, rng, first_axis=None):
    _, before = _trace(g)
    route = next(r for r in before['routes'] if r['source']['elbow'] == elbow)
    old = _active_signature(route)
    last = max((i for i, s in enumerate(route['steps']) if s['hit']), default=-1)
    choices = [(s, axis) for s in route['steps'][last+2:]
               for axis in ((first_axis,) if first_axis is not None else (0, 1))]
    rng.shuffle(choices)
    for s, axis in choices[:90]:
        p, d = s['position'], s['direction']
        m = (p[0]+d[0], p[1]) if axis == 0 else (p[0], p[1]+d[1])
        if not _clear_stamp(g, {m: s['color']}, before['background']):
            continue
        trial = g.copy()
        trial[m] = s['color']
        try:
            _, after = _trace(trial)
        except AssertionError:
            continue
        new = next(r for r in after['routes'] if r['source']['elbow'] == elbow)
        other_before = dict((r['source']['elbow'], tuple((s['position'], s['color'], s['direction'])
                            for s in r['steps'])) for r in before['routes']
                            if r['source']['elbow'] != elbow)
        other_after = dict((r['source']['elbow'], tuple((s['position'], s['color'], s['direction'])
                           for s in r['steps'])) for r in after['routes']
                           if r['source']['elbow'] != elbow)
        encounters = [x for x in new['steps'] if x['mirror'] == m]
        hits = [i for i, x in enumerate(new['steps']) if x['hit']]
        if (other_before == other_after and _active_signature(new) == old+(m,) and len(encounters) == 1
                and encounters[0]['time'] == s['time'] and encounters[0]['hit']
                and len(new['steps'])-hits[-1]-1 >= 3):
            return trial
    return None


def _add_miss(g, elbow, rng, same_color_witness=False):
    _, before = _trace(g)
    route = next(r for r in before['routes'] if r['source']['elbow'] == elbow)
    word_colors = set(route['source']['word'])
    active_colors = {s['color'] for s in route['steps'] if s['hit']}
    choices = []
    for s in route['steps']:
        for color in sorted(word_colors - {s['color']}):
            if same_color_witness and color not in active_colors:
                continue
            for axis in (0, 1):
                choices.append((s, color, axis))
    rng.shuffle(choices)
    for s, color, axis in choices[:120]:
        p, d = s['position'], s['direction']
        m = (p[0]+d[0], p[1]) if axis == 0 else (p[0], p[1]+d[1])
        if not _clear_stamp(g, {m: color}, before['background']):
            continue
        trial = g.copy()
        trial[m] = color
        try:
            _, after = _trace(trial)
        except AssertionError:
            continue
        encounters = [x for r in after['routes'] for x in r['steps'] if x['mirror'] == m]
        if encounters and all(not x['hit'] for x in encounters) and _route_key(after) == _route_key(before):
            return trial
    return None


def _sample_word(rng, palette, period, copies=1, style=None):
    a, b = map(int, rng.choice(palette, size=2, replace=False))
    if style == 'ABA':
        word = [a, b, a]
    elif period == 1:
        word = [a]
    else:
        word = list(map(int, rng.choice(palette, size=period)))
        # Retain repeated colors but guarantee this complete word is primitive.
        if any(period % p == 0 and all(v == word[i % p] for i, v in enumerate(word))
               for p in range(1, period)):
            word = [a]*(period-1)+[b]
    result = tuple(word*copies)
    return result if len(result) >= 2 else result*2


def _fallback(rng, bg, ink, word, hits, sources, axis, direction):
    """Finite, spacious escape construction; semantics are identical to the sampler."""
    g = np.full((30, 30), bg, dtype=np.int8)
    for p, color in _stamp((9, 6), (1, 1), 0, word, ink).items():
        g[p] = color
    for i in range(hits):
        time = 4*(i+1)
        # The row coordinate is monotone; consecutive column reversals cannot loop.
        contact = 9+time, 10 if i % 2 == 0 else 6
        m = contact[0], contact[1]+(1 if i % 2 == 0 else -1)
        g[m] = word[(time-1) % len(word)]
    if len(set(word)) > 1:
        # Use a color seen at an active contact, if one differs at a clear early step.
        active = {word[(4*(i+1)-1) % len(word)] for i in range(hits)}
        for t in (2, 3):
            options = sorted((active or set(word)) - {word[(t-1) % len(word)]})
            if options:
                g[10+t, 6+t] = options[int(rng.integers(len(options)))]
                break
    for j, (e, tape_axis) in enumerate((((3, 22), 1), ((21, 24), 0))[:sources-1]):
        secondary = tuple(np.roll(word, j+1).tolist())
        for p, color in _stamp(e, (1, 1), tape_axis, secondary, ink).items():
            g[p] = color
    if axis == 0:
        g = g.T.copy()
    if direction[0] < 0:
        g = g[::-1].copy()
    if direction[1] < 0:
        g = g[:, ::-1].copy()
    _trace(g)
    return g


def _scene(rng, *, period=2, style=None, hits=2, sources=1, first_axis=None,
           direction=None, teaching=False):
    colors = list(map(int, rng.permutation(10)))
    bg, ink = colors[:2]
    palette = colors[2:2+int(rng.integers(2, 9))]
    copies = 1 if style == 'ABA' else int(rng.choice((1, 1, 2)))
    copies = min(copies, 8//period)
    word = _sample_word(rng, palette, period, copies, style)
    d = direction if direction is not None else DIAGONALS[int(rng.integers(4))]
    axis = int(rng.integers(2)) if first_axis is None else first_axis
    for _ in range(18):
        low = 18 if hits >= 2 or sources > 1 else 12
        shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        g = np.full(shape, bg, dtype=np.int8)
        first = _add_source(g, rng, bg, ink, word, d, min_steps=7 if hits else 4)
        if first is None:
            continue
        g, elbow = first
        complete = True
        for i in range(hits):
            trial = _append_mirror(g, elbow, rng, axis if i == 0 else None)
            if trial is None:
                complete = False
                break
            g = trial
        if not complete:
            continue
        if len(set(word)) > 1:
            trial = _add_miss(g, elbow, rng, same_color_witness=teaching)
            if trial is None:
                continue
            g = trial
        for j in range(sources-1):
            secondary = _sample_word(rng, palette, int(rng.integers(2, max(2, min(period, 3))+1)))
            # Directly expose a per-source clock rather than one shared word.
            offset = next(i for i, value in enumerate(secondary) if value != word[0])
            secondary = secondary[offset:]+secondary[:offset]
            added = _add_source(g, rng, bg, ink, secondary)
            if added is None:
                complete = False
                break
            g, other = added
            if rng.random() < .55:
                trial = _append_mirror(g, other, rng)
                if trial is not None:
                    g = trial
        if complete:
            pair = dict(input=g, output=solve(g))
            if teaching and style == 'ABA':
                witnesses = _witnesses([pair])
                if not all(witnesses[k] for k in ('always', 'straight', 'reset', 'prefix', 'constant', 'next_phase')):
                    continue
            return pair
    g = _fallback(rng, bg, ink, word, hits, sources, axis, d)
    return dict(input=g, output=solve(g))


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    colors = list(map(int, np.unique(pair['input'])))
    # At most 13 prior inputs: 16 distinct injective recolorings suffice.
    for values in itertools.islice(itertools.permutations(range(10), len(colors)), 16):
        mapping = np.arange(10, dtype=np.int8)
        mapping[colors] = values
        candidate = {k: mapping[g] for k, g in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('finite distinctness repair exhausted')


def _counts(num_train, num_test):
    for value, lower, upper in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= int(value) <= upper):
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('chain', 'ensemble', 'word')))
    axis = int(rng.integers(2))
    d = DIAGONALS[int(rng.integers(4))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(a):
        if i == 0:
            pair = _scene(rng, period=3, style='ABA', hits=2, sources=1,
                          first_axis=axis, direction=d, teaching=True)
        elif i == 1:
            pair = _scene(rng, period=2, hits=2, sources=2, first_axis=1-axis,
                          direction=(-d[1], d[0]), teaching=True)
        else:
            pair = _scene(rng, period=int(rng.integers(1, 4)), hits=int(rng.integers(0, 3)),
                          sources=int(rng.integers(1, 3)))
        pair = _unique(pair, seen)
        seen.add(_key(pair['input']))
        puzzle['train'].append(pair)
    rng.shuffle(puzzle['train'])
    train_max_hits = max(max(_features(p['input'])['hits']) for p in puzzle['train'])
    for _ in range(b):
        longer_word = regime == 'word' or (regime == 'chain' and train_max_hits >= 4)
        period = int(rng.integers(4, 6)) if longer_word else int(rng.integers(2, 4))
        hits = int(rng.integers(max(3, train_max_hits+1), 5)) if regime == 'chain' and not longer_word else int(rng.integers(1, 3))
        sources = 3 if regime == 'ensemble' else int(rng.integers(1, 3))
        pair = _unique(_scene(rng, period=period, hits=hits, sources=sources), seen)
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g):
    out, info = _trace(g)
    steps = [s for r in info['routes'] for s in r['steps']]
    hit_colors = {s['color'] for s in steps if s['hit']}
    miss_colors = {info['mirrors'][s['mirror']] for s in steps
                   if s['mirror'] is not None and not s['hit']}
    return dict(shape=g.shape, background=info['background'],
                sources=len(info['sources']), mirrors=len(info['mirrors']),
                lengths=tuple(len(s['word']) for s in info['sources']),
                periods=tuple(next(p for p in range(1, len(s['word'])+1)
                    if len(s['word']) % p == 0 and all(v == s['word'][i % p]
                    for i, v in enumerate(s['word']))) for s in info['sources']),
                directions=tuple(s['direction'] for s in info['sources']),
                first_colors=tuple(s['word'][0] for s in info['sources']),
                axes={s['axis'] for s in steps if s['hit']},
                hits=tuple(sum(s['hit'] for s in r['steps']) for r in info['routes']),
                misses=sum(s['mirror'] is not None and not s['hit'] for s in steps),
                same_color_selectivity=bool(hit_colors & miss_colors),
                changed=int(np.count_nonzero(g != out)))


def _witnesses(pairs):
    alternatives = dict(always=dict(gate_mode='always'), straight=dict(gate_mode='never'),
                        reset=dict(reset=True), prefix=dict(word_mode='prefix'),
                        reverse=dict(word_mode='reverse'), constant=dict(word_mode='constant'),
                        next_phase=dict(gate_mode='next'))
    return {name: any(not np.array_equal(_trace(p['input'], strict=False, **options)[0], p['output'])
                      for p in pairs) for name, options in alternatives.items()}


def validate_generated_puzzle(puzzle):
    """Schema, routes, preservation, evidence, and phase-dependent selectivity."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            expected, info = _trace(g)
            assert np.array_equal(out, expected), 'wrong ray completion'
            assert np.array_equal(out[g != info['background']], g[g != info['background']])
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(g)))
            assert np.count_nonzero(g != out) >= 3
            for route in info['routes']:
                assert route['steps'], 'empty launch'
                last = route['steps'][-1]['position']
                assert last[0] in (0, g.shape[0]-1) or last[1] in (0, g.shape[1]-1)
                for step in route['steps']:
                    assert step['color'] == route['source']['word'][(step['time']-1) % len(route['source']['word'])]
                    if step['mirror'] is not None:
                        assert step['hit'] == (step['color'] == info['mirrors'][step['mirror']])
            if split == 'train':
                training.append(_features(g))
            else:
                testing.append(_features(g))
    assert set().union(*(f['axes'] for f in training)) == {0, 1}, 'teach both reflection axes'
    assert len({d for f in training for d in f['directions']}) >= 2
    assert any(f['same_color_selectivity'] for f in training), 'teach phase, not palette membership'
    assert any(f['sources'] >= 2 for f in training), 'teach independent clocks'
    assert any(len(set(f['first_colors'])) > 1 for f in training), 'teach different local words'
    assert all(_witnesses(puzzle['train']).values()), 'missing shortcut counterexample'
    max_sources = max(f['sources'] for f in training)
    max_period = max(max(f['periods']) for f in training)
    max_hits = max(max(f['hits']) for f in training)
    assert all(f['sources'] > max_sources or max(f['periods']) > max_period
               or max(f['hits']) > max_hits for f in testing), 'missing structural train/test shift'


def _regressions():
    # Directly specified input and expected coordinates, independent of scene sampling.
    g = np.zeros((20, 20), dtype=np.int8)
    g[5, 5] = g[4, 5] = g[5, 4] = 9
    g[3, 5], g[2, 5], g[1, 5] = 2, 3, 2
    g[8, 7], g[9, 10], g[14, 5] = 2, 2, 3
    positions = ((6, 6), (7, 7), (8, 8), (9, 9), (10, 8), (11, 7),
                 (12, 6), (13, 5), (12, 4), (11, 3), (10, 2), (9, 1), (8, 0))
    expected = g.copy()
    for i, p in enumerate(positions):
        expected[p] = (2, 3, 2)[i % 3]
    assert np.array_equal(solve(g), expected)
    for k in range(4):
        for flip in (False, True):
            inp = np.rot90(g, k)
            out = np.rot90(expected, k)
            if flip:
                inp, out = inp[:, ::-1], out[:, ::-1]
            assert np.array_equal(solve(inp), out)
    mapping = np.array([6, 4, 0, 7, 1, 2, 3, 5, 8, 9], dtype=np.int8)
    assert np.array_equal(solve(mapping[g]), mapping[expected])
    # Changing only a mirror's color changes its switch state and the route.
    changed = g.copy()
    changed[9, 10] = 3
    # The old second mirror becomes unused, so remove it for this valid family member.
    changed[14, 5] = 0
    assert not np.array_equal(solve(changed)[g == 0], expected[g == 0])


def self_test():
    """42 sequential seeds, small feature sets, no disk output or retained runs."""
    shapes, structures, backgrounds, directions = set(), set(), set(), set()
    lengths, periods, source_counts, hit_counts, fingerprints = set(), set(), set(), set(), set()
    total, miss_pairs, repeated_axes, combined = 0, 0, 0, 0
    _regressions()
    # Cover the bounded escape construction and full solver word-length range.
    # These cases are small and sequential; no grids are retained or written.
    for length in (2, 3, 5, 7, 8):
        word = tuple(2 if i % 3 != 1 else 3 for i in range(length))
        for hits in (0, 4):
            for sources in (1, 3):
                for axis in (0, 1):
                    for direction in DIAGONALS:
                        g = _fallback(np.random.default_rng(length), 0, 9, word,
                                      hits, sources, axis, direction)
                        out, info = _trace(g)
                        assert len(info['sources']) == sources and np.any(out != g)
                        assert max(sum(s['hit'] for s in r['steps']) for r in info['routes']) == hits
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                digest.update(bytes(g.shape)+g.tobytes()+pair['output'].tobytes())
                f = _features(g)
                shapes.add(f['shape'])
                backgrounds.add(f['background'])
                directions.update(f['directions'])
                lengths.update(f['lengths'])
                periods.update(f['periods'])
                source_counts.add(f['sources'])
                hit_counts.update(f['hits'])
                miss_pairs += f['misses'] > 0
                combined += f['sources'] > 1 and max(f['hits']) > 1
                _, info = _trace(g)
                for route in info['routes']:
                    axes = [s['axis'] for s in route['steps'] if s['hit']]
                    repeated_axes += any(a == b for a, b in zip(axes, axes[1:]))
                roles, encoded = {}, []
                for v in g.flat:
                    encoded.append(roles.setdefault(int(v), len(roles)))
                structures.add((g.shape, hashlib.sha256(bytes(encoded)).digest()))
                total += 1
        fingerprints.add(digest.digest())
        if seed == 0:
            again = generate_puzzle(0)
            assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
                       for x, y in zip(puzzle[s], again[s]) for k in ('input', 'output'))
            bad = {s: [dict(p) for p in puzzle[s]] for s in ('train', 'test')}
            pair = bad['test'][0]
            pair['output'] = pair['output'].copy()
            p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][p] = pair['input'][p]
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('erased trajectory cell accepted')
    assert len(fingerprints) == 42 and len(structures) >= 180 and len(shapes) >= 50
    assert backgrounds == set(range(10)) and directions == set(DIAGONALS)
    assert source_counts >= {1, 2, 3} and periods >= {1, 2, 3, 4, 5}
    assert {0, 1, 2, 3, 4} <= hit_counts and len(lengths) >= 5
    assert miss_pairs >= 120 and repeated_axes >= 15 and combined >= 30
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=total, distinct_puzzles=len(fingerprints),
                grid_shapes=len(shapes), normalized_structures=len(structures),
                tape_lengths=sorted(lengths), primitive_periods=sorted(periods),
                source_counts=sorted(source_counts), reflection_counts=sorted(hit_counts),
                pairs_with_misses=miss_pairs, repeated_axis_routes=repeated_axes,
                multisource_reflected_pairs=combined)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
