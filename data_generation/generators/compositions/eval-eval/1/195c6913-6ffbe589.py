"""Corner pulses rotate the colored stencil that produced those pulses.

The solver decodes only visible symbols. Generation does not import either
parent, store example pairs, or install a puzzle-specific convention.
"""
from collections import Counter
import json

import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _inside(n, p):
    return 0 <= p[0] < n and 0 <= p[1] < n


def _trace(mask, start, edge_turn=False):
    """Maximal east/north legs, with the shared corner counted only once."""
    n = len(mask)
    p, axis = start, 0
    legs, collisions = [], []
    for _ in range(2 * n + 2):
        dr, dc = ((0, 1), (-1, 0))[axis]
        leg = []
        while _inside(n, p) and mask[p]:
            leg.append(p)
            p = (p[0] + dr, p[1] + dc)
        if not leg:
            raise ValueError("empty route leg")
        legs.append(tuple(leg))
        if not _inside(n, p):
            if not edge_turn:
                return dict(legs=legs, collisions=collisions, end="edge")
        else:
            collisions.append((len(legs) - 1, p))
        axis = 1 - axis
        dr, dc = ((0, 1), (-1, 0))[axis]
        p = (leg[-1][0] + dr, leg[-1][1] + dc)
        if not _inside(n, p) or not mask[p]:
            return dict(legs=legs, collisions=collisions, end="blocked")
    raise ValueError("route did not terminate")


def _swatches(a, exterior, bg):
    remaining = set(map(tuple, np.argwhere(exterior & (a != bg)).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        color = int(a[p])
        stack, cells = [p], set()
        remaining.remove(p)
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining and int(a[q]) == color:
                    remaining.remove(q)
                    stack.append(q)
        r, c = min(x[0] for x in cells), min(x[1] for x in cells)
        if cells != {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}:
            raise ValueError("exterior annotations must be separated 2x2 swatches")
        result.append((r, c, color))
    return result


def _decode(value):
    a = _grid(value)
    frames = []
    for color in np.unique(a):
        rows, cols = np.where(a == color)
        r, c, e, f = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        s = e - r + 1
        if s < 8 or f - c + 1 != s or len(rows) != 4 * (s - 1):
            continue
        perimeter = ((rows == r) | (rows == e) | (cols == c) | (cols == f))
        if np.all(perimeter):
            frames.append(int(color))
    candidates = []
    for marker in frames:
        for orientation in range(4):
            canonical = np.rot90(a, orientation).copy()
            rr, cc = np.where(canonical == marker)
            r, c = int(rr.min()), int(cc.min())
            s = int(rr.max()) - r + 1
            exterior = np.ones(canonical.shape, bool)
            exterior[r:r+s, c:c+s] = False
            values, counts = np.unique(canonical[exterior], return_counts=True)
            if len(values) == 0 or np.sum(counts == counts.max()) != 1:
                continue
            bg = int(values[counts.argmax()])
            try:
                swatches = sorted(_swatches(canonical, exterior, bg), key=lambda x: x[1])
            except ValueError:
                continue
            if (not 2 <= len(swatches) <= 6 or len({x[0] for x in swatches}) != 1
                    or swatches[0][0] + 2 >= r
                    or any(y[1] - x[1] < 3 for x, y in zip(swatches, swatches[1:]))):
                continue
            word = tuple(x[2] for x in swatches)
            if len(set(word)) < 2 or bg in word or marker in word:
                continue
            board = canonical[r+1:r+s-1, c+1:c+s-1].copy()
            terrain = set(map(int, np.unique(board))) - {bg, word[0]}
            if len(terrain) != 1:
                continue
            route = terrain.pop()
            if route in word or marker == bg:
                continue
            starts = tuple(map(tuple, np.argwhere(board == word[0]).tolist()))
            if not 1 <= len(starts) <= 4:
                continue
            board[board == word[0]] = route
            mask, n = board == route, len(board)
            if not all((p[1] == 0 or not mask[p[0], p[1]-1])
                       and p[1]+1 < n and mask[p[0], p[1]+1] for p in starts):
                continue
            if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
                   for i, p in enumerate(starts) for q in starts[i+1:]):
                continue
            candidates.append(dict(board=board, mask=mask, word=word, starts=starts,
                                   background=bg, route=route, marker=marker,
                                   orientation=orientation, position=(r, c),
                                   swatches=swatches))
    if len(candidates) != 1:
        raise ValueError("input must uniquely encode one framed board and legend side")
    return candidates[0]


def _program(info, alternative="correct"):
    """Build the hidden stencil and read the endpoint-addressed pulses.

    Named alternatives exist only for teaching-coverage tests, never solving.
    """
    word = info['word']
    if alternative == "deduplicate":
        word = tuple(dict.fromkeys(word))
    elif alternative == "reverse_word":
        word = word[::-1]
    layers = {color: set() for color in sorted(set(word))}
    pulses, markers, records = Counter(), set(), []
    phase = 0
    for start in info['starts']:
        if alternative != "global_phase":
            phase = 0
        rec = _trace(info['mask'], start, alternative == "turn_at_edge")
        if alternative == "straight_only":
            rec = dict(legs=rec['legs'][:1],
                       collisions=[x for x in rec['collisions'] if x[0] == 0],
                       end="blocked")
        contacts = dict(rec['collisions'])
        for j, leg in enumerate(rec['legs']):
            if alternative == "bend_reset":
                phase = 0
            first_color = word[phase % len(word)]
            last_color = first_color
            for p in leg:
                last_color = word[phase % len(word)]
                layers[last_color].add(p)
                phase += 1
            if j in contacts:
                markers.add(contacts[j])
                pulses[first_color if alternative == "leg_first" else last_color] += 1
                if alternative == "marker_tick":
                    phase += 1
        records.append(rec)
    counts = {color: pulses[color] for color in layers}
    if alternative == "no_rotation":
        counts = dict.fromkeys(layers, 0)
    elif alternative == "area_counter":
        counts = {color: len(points) for color, points in layers.items()}
    elif alternative == "all_contacts":
        counts = dict.fromkeys(layers, sum(pulses.values()))
    elif alternative == "legend_counter":
        counts = dict(Counter(word))
    elif alternative == "rank_counts":
        ranks = {k: i+1 for i, k in enumerate(sorted(set(counts.values()) - {0}))}
        counts = {color: ranks.get(k, 0) for color, k in counts.items()}
    return dict(layers=layers, counts=counts, markers=markers, records=records)


def _rotate_point(p, n, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n-1-r
    return r, c


def _destinations(program, n, direction=1):
    return {color: {_rotate_point(p, n, direction * program['counts'][color])
                    for p in points} for color, points in program['layers'].items()}


def _signature(info, alternative="correct"):
    if alternative == "local_counts":
        # Incorrectly rotate each route using only its own pulses. The family
        # instead rotates each whole color mask with pulses from every route.
        destinations = {color: set() for color in sorted(set(info['word']))}
        markers = set()
        for start in info['starts']:
            local_info = dict(info, starts=(start,))
            local = _program(local_info)
            for color, points in _destinations(local, len(info['board'])).items():
                destinations[color].update(points)
            markers.update(local['markers'])
        return (tuple(sorted((color, tuple(sorted(points))) for color, points in destinations.items())),
                tuple(sorted(markers)))
    p = _program(info, alternative)
    destinations = _destinations(p, len(info['board']), -1 if alternative == "counterclockwise" else 1)
    return (tuple(sorted((color, tuple(sorted(points))) for color, points in destinations.items())),
            tuple(sorted(p['markers'])))


_ALTERNATIVES = ("deduplicate", "reverse_word", "bend_reset", "marker_tick", "global_phase",
                 "leg_first", "straight_only", "no_rotation", "area_counter",
                 "all_contacts", "legend_counter", "rank_counts", "counterclockwise",
                 "turn_at_edge", "local_counts")


def _evidence(info):
    correct = _signature(info)
    evidence = {name for name in _ALTERNATIVES if _signature(info, name) != correct}
    p = _program(info)
    if any(sum(map(len, r['legs'])) >= 2*len(info['word']) for r in p['records']):
        evidence.add("cycling")
    if len({k % 4 for k in p['counts'].values()}) > 1:
        evidence.add("different_turns")
    if any(k == 0 and points for color, points in p['layers'].items()
           for k in [p['counts'][color]]):
        evidence.add("unaddressed")
    return evidence


_REQUIRED = set(_ALTERNATIVES) | {"cycling", "different_turns", "unaddressed"}


def _render(info, program):
    n = len(info['board'])
    paths = [p for rec in program['records'] for leg in rec['legs'] for p in leg]
    contacts = [p for rec in program['records'] for _, p in rec['collisions']]
    if len(set(paths)) != len(paths) or len(set(contacts)) != len(contacts):
        raise ValueError("routes must have disjoint footprints and contacts")
    if set(paths) & set(contacts):
        raise ValueError("a collision cannot lie on another route")
    destinations = _destinations(program, n)
    out, used = info['board'].copy(), set(program['markers'])
    for color, points in destinations.items():
        if used & points:
            raise ValueError("rotated layers or fixed collision marks overlap")
        used.update(points)
        for p in points:
            out[p] = color
    for p in program['markers']:
        out[p] = info['marker']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    result = _render(info, _program(info))
    return np.rot90(result, -info['orientation']).copy()


def _word(rng, repeated=False, maximum=6):
    length = int(rng.integers(3 if repeated else 2, maximum+1))
    distinct = int(rng.integers(2, min(4, length - int(repeated))+1))
    symbols = list(range(distinct)) + rng.integers(0, distinct, length-distinct).tolist()
    rng.shuffle(symbols)
    if repeated and symbols == [list(dict.fromkeys(symbols))[i % distinct] for i in range(length)]:
        symbols[0], symbols[1] = symbols[1], symbols[0]
    return symbols


def _terrain(rng, n):
    """Broad random terrain, not a stored route or parent whole-puzzle sampler."""
    method = int(rng.integers(0, 3))
    mask = np.zeros((n, n), bool)
    if method < 2:
        knots = np.unique(np.r_[0, rng.choice(np.arange(1, n-1),
                               int(rng.integers(3, min(n-1, 9))), replace=False), n-1])
        left = np.rint(np.interp(np.arange(n), knots, rng.uniform(-n*.25, n*.55, len(knots)))).astype(int)
        right = np.rint(np.interp(np.arange(n), knots, rng.uniform(n*.4, n*1.3, len(knots)))).astype(int)
        for r in range(n):
            lo, hi = max(0, min(n-2, left[r])), max(1, min(n-1, right[r]))
            mask[r, lo:max(lo+2, hi+1)] = True
    else:
        for _ in range(int(rng.integers(3, 8))):
            r, c = rng.integers(0, n-2, 2)
            h, w = rng.integers(2, n+1, 2)
            mask[r:min(n, r+h), c:min(n, c+w)] = True
    for _ in range(int(rng.integers(1, 9))):
        r, c = rng.integers(0, n, 2)
        h, w = rng.integers(1, max(3, n//2), 2)
        mask[r:min(n, r+h), c:min(n, c+w)] = bool(rng.integers(0, 2))
    return mask


def _choose_starts(rng, mask, count):
    candidates = []
    n = len(mask)
    for r in range(n):
        for c in range(n-1):
            if mask[r, c] and mask[r, c+1] and (c == 0 or not mask[r, c-1]):
                rec = _trace(mask, (r, c))
                if sum(map(len, rec['legs'])) >= 4:
                    candidates.append(((r, c), rec))
    rng.shuffle(candidates)
    picked, used = [], set()
    for start, rec in candidates:
        footprint = {p for leg in rec['legs'] for p in leg} | {p for _, p in rec['collisions']}
        if footprint & used or any(max(abs(start[0]-q[0]), abs(start[1]-q[1])) <= 1 for q in picked):
            continue
        picked.append(start)
        used.update(footprint)
        if len(picked) == count:
            return tuple(sorted(picked))
    return None


def _place(rng, info, orientation):
    n, word = len(info['board']), info['word']
    gaps = rng.integers(1, 3, len(word)-1).tolist()
    legend_width = 2*len(word) + sum(gaps)
    h = int(rng.integers(n+6, 31))
    w = int(rng.integers(max(n+2, legend_width), 31))
    r = int(rng.integers(4, h-n-1))
    c = int(rng.integers(0, w-n-1))
    lr = int(rng.integers(0, r-2))
    lc = int(rng.integers(0, w-legend_width+1))
    a = np.full((h, w), info['background'], np.int8)
    a[r:r+n+2, c:c+n+2] = info['marker']
    a[r+1:r+n+1, c+1:c+n+1] = info['board']
    for p in info['starts']:
        a[r+1+p[0], c+1+p[1]] = word[0]
    for j, color in enumerate(word):
        a[lr:lr+2, lc:lc+2] = color
        if j < len(gaps):
            lc += 2 + gaps[j]
    return np.rot90(a, -orientation).copy()


def _make_pair(rng, kind, orientation, maximum=6, target_many=False):
    # All retries are bounded. Most rejection is the explicit no-overlap rule.
    for attempt in range(6000):
        n = int(rng.integers(14 if kind == 'bent' else 12, 25))
        pattern = _word(rng, repeated=kind in ('straight', 'bent'), maximum=maximum)
        palette = list(map(int, rng.permutation(10)))
        bg, route, marker = palette[:3]
        word = tuple(palette[3+x] for x in pattern)
        if kind == 'straight' and word == word[::-1]:
            continue  # this witness also fixes the legend reading direction
        mask = _terrain(rng, n)
        if kind == 'straight':
            r = int(rng.integers(2, n))
            mask[r, :] = True
            mask[r-1, n-1] = True  # unused corner proves stopping at the edge
            starts = ((r, 0),)
        else:
            count = (2 if kind == 'bent' else int(rng.integers(2 if target_many else 1, 5)))
            starts = _choose_starts(rng, mask, count)
            if starts is None:
                continue
        if not .15 <= mask.mean() <= .9:
            continue
        info = dict(board=np.where(mask, route, bg).astype(np.int8), mask=mask,
                    word=word, starts=starts, background=bg, route=route, marker=marker)
        program = _program(info)
        try:
            canonical_output = _render(info, program)
        except ValueError:
            continue
        if kind == 'bent':
            need = _REQUIRED - {'turn_at_edge', 'cycling', 'unaddressed', 'reverse_word', 'deduplicate'}
            if not need <= _evidence(info):
                continue
        if kind == 'ordinary' and target_many:
            if not any(len(rec['legs']) >= 3 for rec in program['records']):
                continue
        inp = _place(rng, info, orientation)
        try:
            _decode(inp)
        except ValueError:
            continue
        out = np.rot90(canonical_output, -orientation).copy()
        if not np.array_equal(solve(inp), out):
            raise AssertionError("placement changed the rule")
        return {'input': inp, 'output': out}
    raise RuntimeError(f"bounded {kind} construction exhausted")


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(pair['input']), _key(pair['output'])) for pair in p[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    # Held-out combinations: every test recombines multiple starts with a
    # quarter-turned legend/board orientation absent from the demonstrations.
    base = int(rng.integers(0, 4))
    shift = int(rng.integers(1, 4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            kind = ('straight' if i == 0 else 'bent') if split == 'train' and i < 2 else 'ordinary'
            orientation = base if split == 'train' else (base+shift) % 4
            for _ in range(12):
                pair = _make_pair(rng, kind, orientation, target_many=split == 'test')
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("duplicate inputs")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            program = _program(info)
            assert out.shape == info['board'].shape
            expected = np.rot90(_render(info, program), -info['orientation'])
            assert np.array_equal(out, expected), "incorrect stencil rotation or crop"
            canonical = np.rot90(out, info['orientation'])
            assert int(np.sum(canonical == info['marker'])) == len(program['markers'])
            for color, points in program['layers'].items():
                assert int(np.sum(canonical == color)) == len(points), "layer area changed"
            assert sum(program['counts'].values()) == len(program['markers']), "each collision emits one pulse"
            if split == 'train':
                evidence |= _evidence(info)
    assert _REQUIRED <= evidence, f"training fails to distinguish {_REQUIRED-evidence}"


def self_test():
    _semantic_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(7351, 2, 1)) == _puzzle_key(generate_puzzle(7351, 2, 1))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(135+a*10+b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    infos = [_decode(pair['input']) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert {i['orientation'] for i in infos} == set(range(4))
    assert {i['background'] for i in infos} == set(range(10))
    assert len({i['board'].shape for i in infos}) >= 10
    assert len({tuple(i['mask'].ravel()) for i in infos}) >= 150
    assert {len(i['starts']) for i in infos} >= {1, 2, 3}
    assert {len(i['word']) for i in infos} == set(range(2, 7))
    programs = [_program(i) for i in infos]
    assert any(max(p['counts'].values()) >= 4 for p in programs)
    assert any(max(p['counts'].values()) >= 5 for p in programs)
    assert any(len(set(p['counts'].values())) >= 3 for p in programs)
    assert any(len(r['legs']) >= 5 for p in programs for r in p['records'])
    # Complete quarter cycles preserve masks, including sparse/asymmetric masks.
    n = 7
    points = {(0, 1), (2, 4), (6, 3)}
    assert {_rotate_point(p, n, 4) for p in points} == points
    assert {_rotate_point(p, n, 5) for p in points} == {_rotate_point(p, n, 1) for p in points}
    # No collision pulse or post-exit turn may be synthesized at a board edge.
    mask = np.zeros((7, 7), bool)
    mask[5, :] = True
    mask[:6, 6] = True
    rec = _trace(mask, (5, 0))
    assert rec['end'] == 'edge' and len(rec['legs']) == 1 and not rec['collisions']
    mask[:] = False
    mask[5, :4] = True
    mask[:6, 3] = True
    mask[0, 4:6] = True  # rightward terrain remains unused after the north exit
    rec = _trace(mask, (5, 0))
    assert rec['end'] == 'edge' and list(map(len, rec['legs'])) == [4, 5]
    assert rec['collisions'] == [(0, (5, 4))]
    # Validator must actually reject a wrong label.
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


def _semantic_regressions():
    """Hand-calculated phase/pulse/rotation fixture, not a sampled label."""
    mask = np.zeros((13, 13), bool)
    mask[6, 0:4] = True
    mask[4:6, 3] = True
    mask[4, 4:7] = True
    mask[10:12, 1:4] = True  # unvisited terrain must survive unchanged
    info = dict(board=np.where(mask, 8, 0).astype(np.int8), mask=mask,
                word=(1, 1, 2), starts=((6, 0),), background=0, route=8, marker=9)
    p = _program(info)
    assert list(map(len, p['records'][0]['legs'])) == [4, 2, 3]
    assert p['counts'] == {1: 1, 2: 2}
    assert p['markers'] == {(6, 4), (3, 3), (4, 7)}
    assert p['layers'] == {1: {(6, 0), (6, 1), (6, 3), (5, 3), (4, 4), (4, 5)},
                           2: {(6, 2), (4, 3), (4, 6)}}
    expected = info['board'].copy()
    for q in ((0, 6), (1, 6), (3, 6), (3, 7), (4, 8), (5, 8)):
        expected[q] = 1
    for q in ((6, 10), (8, 9), (8, 6)):
        expected[q] = 2
    for q in ((6, 4), (3, 3), (4, 7)):
        expected[q] = 9
    assert np.array_equal(_render(info, p), expected)
    # End-to-end decoding with all scene rotations.
    rng = np.random.default_rng(847)
    for k in range(4):
        inp = _place(rng, info, k)
        assert np.array_equal(solve(inp), np.rot90(expected, -k))


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
