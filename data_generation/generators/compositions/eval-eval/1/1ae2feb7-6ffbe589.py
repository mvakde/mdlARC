"""Escapement loop: competing periodic clocks drive independent color rotors.

All semantics are input-visible. Construction uses sets of firing times;
solve walks cells and rotates coordinates; validation also reads unary counts
back from the returned trace. No parent modules or data are used at runtime.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "cells must be integers"
    assert np.all((0 <= a) & (a <= 9)), "bad colors"
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _loop(r, c, s):
    """Clockwise coordinates, each corner exactly once, starting top-left."""
    return ([(r, x) for x in range(c, c+s)]
            + [(y, c+s-1) for y in range(r+1, r+s)]
            + [(r+s-1, x) for x in range(c+s-2, c-1, -1)]
            + [(y, c) for y in range(r+s-2, r, -1)])


def _turn(r, c, n, k):
    for _ in range(k % 4):
        r, c = c, n-1-r
    return r, c


def _parse(input_grid):
    """Find the apparatus from input alone; canonicalize its tangent handle."""
    original = _grid(input_grid)
    colors, sizes = np.unique(original, return_counts=True)
    assert np.sum(sizes == sizes.max()) == 1, "background must be uniquely dominant"
    bg = int(colors[sizes.argmax()])
    candidates = []
    for q in range(4):
        a = np.rot90(original, q)
        for rail in (int(v) for v in colors if v != bg):
            rr, cc = np.nonzero(a == rail)
            r, c, er, ec = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
            s = er-r+1
            if s < 7 or ec-c+1 != s or len(rr) != 4*s-6:
                continue
            path = _loop(r, c, s)
            if any(y not in (r, er) and x not in (c, ec) for y, x in zip(rr, cc)):
                continue
            holes = [i for i, p in enumerate(path) if a[p] != rail]
            if len(holes) != 2 or holes[0] != 0:
                continue
            bead = int(a[r, c])
            if bead in (bg, rail) or a[path[holes[1]]] != bead or np.sum(a == bead) != 2:
                continue
            # All exterior content must be the single tangent handle.
            exterior = a != bg
            exterior = exterior.copy()
            exterior[r:er+1, c:ec+1] = False
            yr, xc = np.nonzero(exterior)
            if not len(yr) or np.any(yr != r) or np.any(xc >= c) or a[r, c-1] != bg:
                continue
            n = s-4
            core = a[r+2:er-1, c+2:ec-1]
            moat = a[r+1:er, c+1:ec]
            if not (np.all(moat[0] == bg) and np.all(moat[-1] == bg)
                    and np.all(moat[:, 0] == bg) and np.all(moat[:, -1] == bg)):
                continue
            palette = tuple(int(v) for v in np.unique(core) if v != bg)
            if len(palette) < 2 or rail in palette or bead in palette:
                continue
            bars, x = [], c-1
            while x >= 0:
                color = int(a[r, x])
                if color == bg:
                    x -= 1
                    continue
                end = x
                while x >= 0 and a[r, x] == color:
                    x -= 1
                bars.append((color, end-x, c-end))
            if (not bars or any(b[0] not in palette for b in bars)
                    or len({b[0] for b in bars}) != len(bars)):
                continue
            t = holes[1]-1
            if t < 1:
                continue
            candidates.append(dict(a=a, q=q, bg=bg, rail=rail, bead=bead,
                                   r=r, c=c, s=s, n=n, core=core.copy(),
                                   palette=palette, bars=bars, path=path, time=t))
    assert len(candidates) == 1, "apparatus must have one input-only interpretation"
    return candidates[0]


def _evaluate(info, alternative=None):
    """Cell-wise interpretation; options are explicit teaching-audit rivals."""
    a, bg = info['a'], info['bg']
    out = a.copy()
    bars = info['bars']
    path = info['path']
    if alternative == 'reverse_path':
        path = path[:1] + path[:0:-1]
    stop = next(i for i, p in enumerate(path[1:], 1) if a[p] == info['bead'])
    admitted = Counter()
    for t, pos in enumerate(path[1:stop]):
        clock_time = (t % (info['s']-1) if alternative == 'corner_reset' else
                      abs(pos[0]-info['r'])+abs(pos[1]-info['c'])-1
                      if alternative == 'manhattan' else
                      t-1 if alternative == 'phase' else t)
        firing = []
        for color, length, proximity in bars:
            period = (length+1 if alternative == 'period_plus' else
                      max(1, length-1) if alternative == 'period_minus' else length)
            if clock_time % period == 0:
                firing.append((color, length, proximity))
        if not firing:
            continue
        if alternative == 'far':
            chosen = firing[-1]
        elif alternative == 'short':
            chosen = min(firing, key=lambda b: (b[1], b[2]))
        elif alternative == 'long':
            chosen = min(firing, key=lambda b: (-b[1], b[2]))
        elif alternative == 'low_color':
            chosen = min(firing)
        elif alternative == 'high_color':
            chosen = max(firing)
        else:
            chosen = firing[0]
        out[pos] = chosen[0]
        admitted[chosen[0]] += 1
    counts = dict(admitted)
    if alternative == 'unsuppressed':
        counts = {color: (stop-2)//length+1 for color, length, _ in bars}
    elif alternative == 'raw_length':
        counts = {color: length for color, length, _ in bars}
    elif alternative == 'no_rotation':
        counts = {}
    elif alternative == 'once':
        counts = {color: 1 for color in admitted}
    elif alternative == 'global':
        counts = {color: sum(admitted.values()) for color in info['palette']}
    elif alternative == 'absent_one':
        counts.update({color: 1 for color in info['palette'] if color not in {b[0] for b in bars}})
    rotor = np.full((info['n'], info['n']), bg, dtype=np.int8)
    for color in info['palette']:
        k = counts.get(color, 0)
        if alternative == 'counterclockwise':
            k = -k
        for y, x in np.argwhere(info['core'] == color):
            yy, xx = _turn(int(y), int(x), info['n'], k)
            assert rotor[yy, xx] == bg, "rotated layers conflict"
            rotor[yy, xx] = color
    r, c = info['r']+2, info['c']+2
    out[r:r+info['n'], c:c+info['n']] = rotor
    return np.rot90(out, -info['q']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the supplied input and the family's fixed conventions."""
    return _evaluate(_parse(input_grid))


_RIVALS = ('far', 'short', 'long', 'low_color', 'high_color', 'period_plus',
           'period_minus', 'phase', 'reverse_path', 'corner_reset', 'manhattan',
           'no_rotation', 'raw_length',
           'unsuppressed', 'once', 'global', 'counterclockwise', 'absent_one')


def _orbits(n):
    remaining = {(r, c) for r in range(n) for c in range(n)}
    result = []
    while remaining:
        r, c = min(remaining)
        orbit = list(dict.fromkeys(_turn(r, c, n, k) for k in range(4)))
        remaining.difference_update(orbit)
        result.append(orbit)
    return result


def _source(rng, n, bg, colors, asymmetric):
    """Compose varied partial masks inside disjoint, rotation-stable territories."""
    orbits = [o for o in _orbits(n) if len(o) == 4]
    k = len(colors)
    anchors = list(map(int, rng.choice(len(orbits), k, replace=False)))
    style = int(rng.integers(3))
    if style == 0:
        # Contiguous depth bands, with varying territory sizes and order.
        order = sorted(range(len(orbits)), key=lambda i: (
            min(min(r, c, n-1-r, n-1-c) for r, c in orbits[i]), float(rng.random())))
        amounts = np.ones(k, dtype=int) + rng.multinomial(len(orbits)-k, rng.dirichlet(np.ones(k)*2))
        owners, cursor = {}, 0
        for color_index in rng.permutation(k):
            color_index = int(color_index)
            for i in order[cursor:cursor+amounts[color_index]]:
                owners[i] = color_index
            cursor += amounts[color_index]
    elif style == 1:
        # Distances between complete orbits yield spatial, non-shell territories.
        owners = {i: min(range(k), key=lambda j: min(
            (r-y)**2+(c-x)**2 for r, c in orbit for y, x in orbits[anchors[j]]))
                  for i, orbit in enumerate(orbits)}
    else:
        owners = {i: int(rng.integers(k)) for i in range(len(orbits))}
        for j, i in enumerate(anchors):
            owners[i] = j
    assert set(owners.values()) == set(range(k))
    result = np.full((n, n), bg, dtype=np.int8)
    for j, color in enumerate(colors):
        territory = [i for i in range(len(orbits)) if owners[i] == j]
        symmetry = 'asym' if color in asymmetric else str(rng.choice(['asym', 'half', 'quarter']))
        density, phase = float(rng.uniform(.25, .85)), int(rng.integers(4))
        motif = int(rng.integers(3))
        ar, ac = rng.uniform(0, n, 2)
        units, scores = [], []
        for i in territory:
            orbit = orbits[i]
            groups = ([orbit] if symmetry == 'quarter' else
                      [orbit[::2], orbit[1::2]] if symmetry == 'half' else [[p] for p in orbit])
            for group in groups:
                r, c = group[0]
                score = ((r-ar)**2+(c-ac)**2 if motif == 0 else
                         (r+phase) % int(max(2, n//3)) if motif == 1 else float(rng.random())*n)
                units.append(group)
                scores.append(score+float(rng.random())*.4)
        wanted = max(1, round(len(territory)*4*density))
        chosen = set()
        for z in np.argsort(scores):
            chosen.update(units[int(z)])
            if len(chosen) >= wanted:
                break
        if symmetry != 'quarter':
            orbit = orbits[int(rng.choice(territory))]
            chosen.difference_update(orbit)
            chosen.add(orbit[phase])
            if symmetry == 'half':
                chosen.add(orbit[(phase+2) % 4])
        for pos in chosen:
            result[pos] = color
    if n % 2 and rng.random() < .6:
        result[n//2, n//2] = int(rng.choice(colors))
    assert all(np.any(result == color) for color in colors)
    return result


def _schedule(bars, horizon):
    """Construction uses requested-time sets and set subtraction."""
    claimed, events, counts = set(), {}, {}
    for color, length in bars:
        requests = set(range(0, horizon, length))
        accepted = requests - claimed
        claimed.update(requests)
        counts[color] = len(accepted)
        events.update({t: color for t in accepted})
    return events, counts


def _pair(rng, witness, witness_bars, orientation, min_time=12):
    if witness:
        bars = list(witness_bars)
        content_count = int(rng.integers(3, 6))
        min_time = max(min_time, int(np.lcm(bars[0][1], bars[1][1]))+2)
    else:
        content_count = int(rng.integers(2, 6))
        marked = int(rng.integers(1, min(4, content_count)+1))
        budget = int(rng.integers(max(marked, 5), 15))
        lengths = []
        for i in range(marked):
            length = int(rng.integers(1, min(8, budget-(marked-i-1))+1))
            lengths.append(length)
            budget -= length
        bars = [(None, length) for length in lengths]
    protected = {color for color, _ in bars if color is not None}
    available = [c for c in range(10) if c not in protected]
    bg, rail, bead = map(int, rng.choice(available, 3, replace=False))
    colors = list(protected)
    colors.sort()
    extras = [c for c in range(10) if c not in {bg, rail, bead} | protected]
    colors += list(map(int, rng.choice(extras, content_count-len(colors), replace=False)))
    if not witness:
        rng.shuffle(colors)
        bars = [(colors[i], length) for i, (_, length) in enumerate(bars)]
    # At least one blank between handle and start; interior gaps use spare room.
    payload = sum(length for _, length in bars)+1
    n_max = min(14, 26-payload)
    # A side n+4 has perimeter 4n+12; exclude two beads.
    n_min = max(5, (min_time+2-12+3)//4)
    assert n_min <= n_max
    # A narrow arc can show only complete turns (e.g. one period-eight clock
    # observed for 28..30 ticks). Choose among size/time combinations that
    # actually expose rotation instead of accidentally rejecting that seed.
    feasible = []
    for size in range(n_min, n_max+1):
        maximum_time = min(27, 4*size+10) if witness or min_time == 12 else 4*size+10
        times = [t for t in range(min_time, maximum_time+1)
                 if any(k % 4 for k in _schedule(bars, t)[1].values())]
        if times:
            feasible.append((size, times))
    assert feasible, "no visibly moving size/time combination"
    n, times = feasible[int(rng.integers(len(feasible)))]
    s = n+4
    horizon = int(rng.choice(times))
    events, turns = _schedule(bars, horizon)
    asymmetric = ({int(rng.choice([color for color, _ in bars if turns[color] % 4]))}
                  if not witness else set(colors))
    # One changed layer guarantees a visible B contribution; other controlled
    # masks may have real half/quarter symmetry, including unchanged controls.
    source = _source(rng, n, bg, colors, asymmetric)
    gap_room = 30-s-sum(length for _, length in bars)
    gaps = [1] + [0]*(len(bars)-1)
    for _ in range(int(rng.integers(min(6, gap_room)))):
        gaps[int(rng.integers(len(gaps)))] += 1
    handle = sum(length for _, length in bars)+sum(gaps)
    spare_width = min(6, 30-s-handle)
    pad_left = int(rng.integers(min(3, spare_width)+1))
    pad_right = int(rng.integers(min(3, spare_width-pad_left)+1))
    pad_top, pad_bottom = map(int, rng.integers(0, 4, 2))
    r, c = pad_top, handle+pad_left
    inp = np.full((s+pad_top+pad_bottom, s+handle+pad_left+pad_right), bg, dtype=np.int8)
    path = _loop(r, c, s)
    for pos in path:
        inp[pos] = rail
    inp[path[0]] = inp[path[horizon+1]] = bead
    inp[r+2:r+s-2, c+2:c+s-2] = source
    cursor = c
    for (color, length), gap in zip(bars, gaps):
        cursor -= gap
        inp[r, cursor-length:cursor] = color
        cursor -= length
    out = inp.copy()
    for t, color in events.items():
        out[path[t+1]] = color
    rotor = np.full((n, n), bg, dtype=np.int8)
    used = np.zeros((n, n), dtype=bool)
    for color in colors:
        mask = np.rot90(source == color, -turns.get(color, 0))
        assert not np.any(mask & used)
        rotor[mask] = color
        used |= mask
    out[r+2:r+s-2, c+2:c+s-2] = rotor
    return {role: np.rot90(a, orientation).copy() for role, a in [('input', inp), ('output', out)]}


def _teaching(train, infos):
    evidence = set()
    precedence = set()
    unmarked = False
    for pair, info in zip(train, infos):
        a = np.rot90(np.asarray(pair['output']), info['q'])
        trace = [int(a[p]) for p in info['path'][1:info['time']+1]]
        for i, near in enumerate(info['bars']):
            for far in info['bars'][i+1:]:
                if (any(t > 0 and t % near[1] == t % far[1] == 0 and trace[t] == near[0]
                        for t in range(len(trace))) and far[0] in trace):
                    evidence.add('short_near' if near[1] < far[1] else 'long_near')
                    precedence.add((near[0], far[0]))
        for color in set(info['palette']) - {b[0] for b in info['bars']}:
            mask = info['core'] == color
            unmarked |= not np.array_equal(mask, np.rot90(mask))
    if any((b, a) in precedence for a, b in precedence):
        evidence.add('color_cycle')
    if unmarked:
        evidence.add('unmarked')
    for rival in _RIVALS:
        if any(not np.array_equal(_evaluate(info, rival), pair['output'])
               for pair, info in zip(train, infos)):
            evidence.add(rival)
    return evidence


def validate_generated_puzzle(puzzle):
    """Validate returned grids, causal trace-to-mask accounting, and teaching."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _parse(inp)
            assert 5 <= info['n'] <= 14 and 2 <= len(info['palette']) <= 5
            assert 1 <= len(info['bars']) <= 4 and all(1 <= b[1] <= 8 for b in info['bars'])
            assert np.array_equal(solve(inp), out), "wrong pulse trace, rotation, or preservation"
            actual = np.rot90(out, info['q'])
            active = info['path'][1:info['time']+1]
            # Count controls from the actual output track, not generation state.
            counts = Counter(int(actual[p]) for p in active)
            rotor = actual[info['r']+2:info['r']+info['s']-2,
                           info['c']+2:info['c']+info['s']-2]
            moved = False
            mutable = np.zeros(info['a'].shape, dtype=bool)
            for pos in active:
                mutable[pos] = True
            mutable[info['r']+2:info['r']+info['s']-2,
                    info['c']+2:info['c']+info['s']-2] = True
            assert np.array_equal(actual[~mutable], info['a'][~mutable]), "changed apparatus"
            for color in info['palette']:
                source_mask = info['core'] == color
                output_mask = rotor == color
                assert int(source_mask.sum()) == int(output_mask.sum()), "layer area changed"
                assert np.array_equal(np.rot90(source_mask, -counts[color]), output_mask), "trace must control rotor"
                moved |= not np.array_equal(source_mask, output_mask)
            assert moved, "each generated pair must visibly exercise rotation"
            if split == 'train':
                assert 12 <= info['time'] <= 27
            infos[split].append(info)
    required = set(_RIVALS) | {'short_near', 'long_near', 'color_cycle', 'unmarked'}
    assert required <= _teaching(puzzle['train'], infos['train']), "teaching gap"
    assert min(i['time'] for i in infos['test']) >= max(i['time'] for i in infos['train'])+4, "missing later-time shift"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    period_options = ((2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (4, 6))
    a, b = period_options[int(rng.integers(len(period_options)))]
    first, second = map(int, rng.choice(10, 2, replace=False))
    required = set(_RIVALS) | {'short_near', 'long_near', 'color_cycle', 'unmarked'}
    base, mode = int(rng.integers(4)), int(rng.integers(3))
    witness_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    # Bounded curriculum resampling is solely for visible teaching conditions.
    for attempt in range(80):
        train = []
        for i in range(num_train):
            witness = i in witness_slots
            bars = [(first, a), (second, b)]
            if i == witness_slots[1]:
                bars.reverse()
            q = base if mode != 1 else int(rng.integers(4))
            train.append(_pair(rng, witness, bars, q))
        infos = [_parse(p['input']) for p in train]
        if (len({_key(p['input']) for p in train}) == len(train)
                and required <= _teaching(train, infos)):
            break
    else:
        raise RuntimeError("could not establish a discriminating curriculum")
    puzzle = {'train': train, 'test': []}
    seen = {_key(p['input']) for p in train}
    later = max(i['time'] for i in infos)+4
    for i in range(num_test):
        for _ in range(80):
            q = (base+1) % 4 if mode == 2 else base if mode == 0 else int(rng.integers(4))
            pair = _pair(rng, False, [], q, min_time=later)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("could not make a distinct test input")
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def self_test():
    """Bounded checks over 42 seeds, exact counts, metamorphisms and corruption."""
    # Seven manually calculated ticks of (2,3); collision removes B at t=0,6.
    events, counts = _schedule([(1, 2), (2, 3)], 7)
    assert [events.get(t, 8) for t in range(7)] == [1, 8, 1, 2, 1, 8, 1]
    assert counts == {1: 4, 2: 1}
    assert _turn(0, 1, 5, counts[1]) == (0, 1)
    assert _turn(0, 1, 5, counts[2]) == (1, 4)
    # A complete coordinate-level fixture: the stop is the top-right corner.
    # These literal rows were calculated from seven ticks, not from solve.
    fixture_in = np.array([list(map(int, row)) for row in (
        '222110988888889',
        '000000800000008',
        '000000800120008',
        '000000800300008',
        '000000800000008',
        '000000800000008',
        '000000800000008',
        '000000800000008',
        '000000888888888')], dtype=np.int8)
    fixture_out = np.array([list(map(int, row)) for row in (
        '222110918121819',
        '000000800000008',
        '000000800100008',
        '000000800300008',
        '000000800000208',
        '000000800000008',
        '000000800000008',
        '000000800000008',
        '000000888888888')], dtype=np.int8)
    assert np.array_equal(solve(fixture_in), fixture_out)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, normalized = [], set()
    for puzzle in puzzles:
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                i = _parse(pair['input'])
                records.append((pair, i))
                roles = {}
                normalized.add((pair['input'].shape, tuple(
                    roles.setdefault(int(v), len(roles)) for v in pair['input'].flat)))
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert len({p['input'].shape for p, _ in records}) >= 35
    assert len(normalized) >= 190
    assert {i['q'] for _, i in records} == set(range(4))
    assert {i['bg'] for _, i in records} == set(range(10))
    assert {len(i['palette']) for _, i in records} == {2, 3, 4, 5}
    assert {len(i['bars']) for _, i in records} == {1, 2, 3, 4}
    assert len({i['n'] for _, i in records}) >= 8
    assert set(range(1, 9)) <= {b[1] for _, i in records for b in i['bars']}
    assert len({tuple(b[1] for b in i['bars']) for _, i in records}) >= 50
    assert len({i['time'] for _, i in records}) >= 30
    assert {min(3, (i['time']+1)//(i['s']-1)) for _, i in records} == {0, 1, 2, 3}
    symmetries, controlled_symmetric, unchanged_controlled, max_area = set(), 0, 0, 0
    for pair, info in records:
        actual = np.rot90(pair['output'], info['q'])
        trace = Counter(int(actual[p]) for p in info['path'][1:info['time']+1])
        for color in info['palette']:
            mask = info['core'] == color
            kind = ('quarter' if np.array_equal(mask, np.rot90(mask)) else
                    'half' if np.array_equal(mask, np.rot90(mask, 2)) else 'asym')
            symmetries.add(kind)
            max_area = max(max_area, int(mask.sum()))
            if trace[color]:
                controlled_symmetric += kind != 'asym'
                unchanged_controlled += np.array_equal(mask, np.rot90(mask, -trace[color]))
    assert symmetries == {'asym', 'half', 'quarter'}
    assert controlled_symmetric >= 5 and unchanged_controlled >= 10 and max_area >= 40
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(411+a*7+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, False), (3.0, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(1, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    pair, info = next((p, i) for p, i in records if i['a'].shape[1] < 30)
    before = pair['input'].copy()
    assert np.array_equal(solve(before), pair['output'])
    assert np.array_equal(before, pair['input']), "solve mutated its input"
    for q in range(4):
        assert np.array_equal(solve(np.rot90(before, q)), np.rot90(pair['output'], q))
    palette = np.array([7, 2, 9, 0, 5, 8, 1, 4, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(palette[before]), palette[pair['output']])
    # Move the whole handle one cell outward; keep its lengths and order.
    shifted = np.pad(info['a'], ((0, 0), (1, 0)), constant_values=info['bg'])
    c, r = info['c']+1, info['r']
    shifted[r, :c-1] = shifted[r, 1:c].copy()
    shifted[r, c-1] = info['bg']
    changed = _parse(shifted)
    assert [(b[0], b[1], b[2]+1) for b in info['bars']] == changed['bars']
    predicted = solve(shifted)
    ri, ci = changed['r']+2, changed['c']+2
    original_out = np.rot90(pair['output'], info['q'])
    assert np.array_equal(predicted[ri:ri+info['n'], ci:ci+info['n']],
                          original_out[info['r']+2:info['r']+info['s']-2,
                                       info['c']+2:info['c']+info['s']-2])
    assert [int(predicted[p]) for p in changed['path']] == [int(original_out[p]) for p in info['path']]
    corrupted = {s: [{role: grid.copy() for role, grid in p.items()} for p in puzzles[0][s]]
                 for s in ('train', 'test')}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=len(records), input_shapes=len({p['input'].shape for p, _ in records}),
                normalized_inputs=len(normalized), sizes=sorted({i['n'] for _, i in records}),
                periods=sorted({b[1] for _, i in records for b in i['bars']}),
                ordered_period_programs=len({tuple(b[1] for b in i['bars']) for _, i in records}),
                horizon_values=len({i['time'] for _, i in records}),
                symmetries=sorted(symmetries), controlled_symmetric=controlled_symmetric,
                unchanged_controlled=unchanged_controlled, maximum_layer_area=max_area)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
