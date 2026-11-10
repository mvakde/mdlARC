"""Sweeping shutters: continuous diagonal apertures transport edge tokens.

Only NumPy and the standard library are used.  The L-shaped reservoir visibly
encodes the sweep direction; no puzzle-level state is needed by solve().
"""

import json
from pathlib import Path

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "ARC colors required")
    return g


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _read(input_grid):
    """Normalize to a top-and-right reservoir and read only visible segments.

    Canonical coordinates: t=r+c runs along a stroke, q=c-r orders strokes.
    Sources enter along (1,-1), thus decreasing q at constant t.
    """
    inp = _grid(input_grid)
    _require(min(inp.shape) >= 5, "canvas too small for the launch margin")
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, -1])
        if (np.all(g[0, :] == token) and np.all(g[:, -1] == token)
                and np.count_nonzero(g == token) == sum(g.shape) - 1):
            candidates.append((g, k, token))
    _require(len(candidates) == 1, "one unique L-shaped token reservoir required")
    g, rotation, token = candidates[0]
    bg = int(g[1, -2])
    _require(bg != token and np.all(g[1, :-1] == bg)
             and np.all(g[1:, -2] == bg), "clear launch margin required")
    mask = (g != bg) & (g != token)
    lines = {}
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        lines.setdefault(c-r, []).append(r)
    strokes = []
    for q, rows in lines.items():
        rows.sort()
        runs = []
        for r in rows:
            if not runs or r != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(r)
        for run in runs:
            colors = {int(g[r, r+q]) for r in run}
            _require(len(run) >= 2 and len(colors) == 1,
                     "guides must be whole monochrome diagonal segments")
            strokes.append(dict(q=q, lo=2*run[0]+q, hi=2*run[-1]+q,
                                length=len(run), color=colors.pop()))
    strokes.sort(key=lambda s: (s['q'], s['lo']))
    _require(len(strokes) >= 2, "at least two visible segments required")
    return dict(grid=g, rotation=rotation, token=token, background=bg,
                strokes=strokes)


def _slices(info, alternative=""):
    """Describe transfers, including weaker rules for teaching diagnostics.

    The diagnostics alter a single interpretation of the same visible input.
    They are not independent reference solvers.
    """
    g = info['grid']
    h, w = g.shape
    strokes = info['strokes']
    records = []
    for t in range(h+w-1):
        active = [s for s in strokes if s['lo'] <= t <= s['hi']]
        if alternative == 'pixel_centers':
            active = [s for s in active if (t-s['q']) % 2 == 0]
        active.sort(key=lambda s: s['q'])
        if len(active) < 2:
            continue
        low, high = active[0], active[-1]
        free = []
        for r in range(max(0, t-w+1), min(h-1, t)+1):
            c = t-r
            q = c-r
            if low['q'] < q < high['q'] and int(g[r, c]) == info['background']:
                free.append((r, c))
        if not free:
            continue
        # The row iteration lists cells from near to far along (1,-1).
        matching = low['color'] == high['color']
        destination = free[-1]
        enabled = matching
        if alternative == 'ignore_matching':
            enabled = True
        elif alternative == 'nearest_pair':
            a, b = active[-2:]
            slots = [p for p in free if a['q'] < p[1]-p[0] < b['q']]
            enabled = a['color'] == b['color'] and bool(slots)
            destination = slots[-1] if slots else free[-1]
        elif alternative == 'any_equal_pair':
            enabled = len({s['color'] for s in active}) < len(active)
        elif alternative == 'consecutive':
            slots = []
            for a, b in zip(strokes, strokes[1:]):
                if (a['lo'] <= t <= a['hi'] and b['lo'] <= t <= b['hi']
                        and a['color'] == b['color']):
                    slots.extend(p for p in free if a['q'] < p[1]-p[0] < b['q'])
            enabled = bool(slots)
            destination = max(slots) if slots else free[-1]
        elif alternative == 'first_slot':
            destination = free[0]
        source = (0, t) if t < w else (t-w+1, w-1)
        records.append(dict(t=t, source=source, destination=destination,
                            enabled=enabled, matching=matching, low=low, high=high,
                            active=active, free=free))
    return records


def _answer(info, alternative=""):
    out = info['grid'].copy()
    for record in _slices(info, alternative):
        if record['enabled']:
            out[record['source']] = info['background']
            out[record['destination']] = info['token']
    return np.rot90(out, -info['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone, with no RNG, labels, or stored examples."""
    return _answer(_read(input_grid))


_ALTERNATIVES = ('ignore_matching', 'pixel_centers', 'consecutive',
                 'nearest_pair', 'any_equal_pair', 'first_slot')


def _witnesses(info, output=None):
    if output is None:
        output = _answer(info)
    return {name for name in _ALTERNATIVES
            if not np.array_equal(output, _answer(info, name))}


def _interval(rng, q, length, anchor=0):
    # Choose an endpoint phase so that the continuous interval contains anchor.
    center = anchor - (anchor-q) % 2
    max_before = length-1 if center == anchor else length-2
    before = int(rng.integers(max_before+1))
    lo = center - 2*before
    return lo, lo+2*(length-1)


def _specs(rng, index, is_test):
    if not is_test and index == 0:
        n = int(rng.integers(3, 7))
    elif not is_test and index == 1:
        n = int(rng.integers(4, 8))
    else:
        n = int(rng.integers(2, 10 if is_test else 8))
    qs = [0]
    for _ in range(n-1):
        qs.append(qs[-1] + int(rng.integers(2, 9)))
    cap = 13 if is_test else 7
    lengths = list(map(int, rng.integers(2, cap+1, size=n)))
    if is_test:
        lengths[int(rng.integers(n))] = int(rng.integers(9, 14))
    mode = int(rng.integers(5))
    intervals = []
    if mode in (0, 1):
        for q, length in zip(qs, lengths):
            intervals.append(_interval(rng, q, length))
    elif mode == 2:
        for q, length in zip(qs, lengths):
            lo = int(rng.integers(-8, 9))
            lo -= (lo-q) % 2
            intervals.append((lo, lo+2*(length-1)))
    elif mode == 3:
        # Aligned starts or ends, with the unavoidable pixel-parity phase.
        ending = bool(rng.integers(2))
        for q, length in zip(qs, lengths):
            if ending:
                hi = -(q % 2)
                intervals.append((hi-2*(length-1), hi))
            else:
                lo = q % 2
                intervals.append((lo, lo+2*(length-1)))
    else:
        # Longitudinal chains can have no instant when all guides are active.
        intervals = [None]*n
        end = 0
        for i in map(int, rng.permutation(n)):
            lo = end - int(rng.integers(1, 6))
            lo -= (lo-qs[i]) % 2
            end = lo+2*(lengths[i]-1)
            intervals[i] = (lo, end)
    if not is_test and index == 0:
        # A short interior interval supplies a visible non-adjacency witness.
        middle = int(rng.integers(1, n-1))
        lo = -(qs[middle] % 2)
        intervals[middle] = (lo, lo+2*int(rng.integers(1, 3)))
        for outer in (0, n-1):
            start = lo-4
            start -= (start-qs[outer]) % 2
            intervals[outer] = (start, start+2*int(rng.integers(5, 7)))
    return [(q, lo, hi) for q, (lo, hi) in zip(qs, intervals)]


def _render(rng, specs, colors, token, bg, rotation):
    objects = []
    for q, lo, hi in specs:
        objects.append([((t-q)//2, (t+q)//2) for t in range(lo, hi+1, 2)])
    cells = [p for obj in objects for p in obj]
    _require(len(cells) == len(set(cells)), "sampled guides overlap")
    r0, c0 = np.min(cells, axis=0)
    r1, c1 = np.max(cells, axis=0)
    sh, sw = int(r1-r0+1), int(c1-c0+1)
    if sh+2 > 30 or sw+2 > 30:
        return None
    # Always reserve top row/right column plus their empty approach cells.
    # Bottom and left may touch guides; extra padding is independent per side.
    pad_limit = 4 if max(sh, sw) < 12 else 7
    h = sh+2+int(rng.integers(min(pad_limit, 30-sh-2)+1))
    w = sw+2+int(rng.integers(min(pad_limit, 30-sw-2)+1))
    if rng.random() < .12:
        h = int(rng.integers(h, 31))
        w = int(rng.integers(w, 31))
    top = int(rng.integers(2, h-sh+1))
    left = int(rng.integers(0, w-sw-1))
    g = np.full((h, w), bg, dtype=np.int8)
    g[0, :] = token
    g[:, -1] = token
    for obj, color in zip(objects, colors):
        for r, c in obj:
            g[int(r-r0)+top, int(c-c0)+left] = color
    inp = np.rot90(g, rotation).copy()
    try:
        info = _read(inp)
    except AssertionError:
        return None
    if len(info['strokes']) != len(specs):
        return None
    return inp, info


def _make_pair(rng, index, is_test, params, seen):
    token, bg, palette = params['token'], params['background'], params['palette']
    rotation = params['rotation']
    if params['mixed_orientation']:
        rotation = (rotation+index) % 4 if not is_test else int(rng.integers(4))
    required = ({'consecutive', 'pixel_centers', 'first_slot'} if index == 0
                else {'ignore_matching', 'nearest_pair', 'any_equal_pair', 'first_slot'})
    for _ in range(1600):
        specs = _specs(rng, index, is_test)
        n = len(specs)
        if not is_test and index == 0:
            colors = [palette[0]]*n
        else:
            colors = list(map(int, rng.choice(palette, size=n)))
            # Repeated identities are common but their ranks are not commands.
            repeat = palette[1] if not is_test and index == 1 else int(rng.choice(palette))
            for i in rng.choice(n, size=int(rng.integers(2, min(n, 4)+1)), replace=False):
                colors[int(i)] = repeat
        rendered = _render(rng, specs, colors, token, bg, rotation)
        if rendered is None:
            continue
        inp, info = rendered
        if _key(inp) in seen:
            continue
        records = _slices(info)
        active = [r for r in records if r['enabled']]
        if not active:
            continue
        if not is_test and index < 2:
            if not required <= _witnesses(info):
                continue
            if index == 1 and not any(r['low']['color'] == palette[1] for r in active):
                continue
        # Test examples must use both inherited mechanisms on their actual answer.
        if is_test and not {'ignore_matching', 'pixel_centers'} <= _witnesses(info):
            continue
        return dict(input=inp, output=_answer(info))
    return _fallback_pair(index, is_test, params, rotation, seen)


def _fallback_pair(index, is_test, params, rotation, seen):
    """Finite constructive reserve: 16 placements, at most 13 prior inputs.

    It is the same interval/latch rule and supplies the same teaching witnesses.
    This makes count support independent of randomized rejection succeeding.
    """
    palette = params['palette']
    if index == 0 and not is_test:
        specs = [(0, 0, 12), (4, 4, 6), (8, 0, 12)]
        colors = [palette[0]]*3
    else:
        end = 24 if is_test else 12
        specs = [(0, 0, end), (4, 0, end), (8, 0, end), (12, 4, 8)]
        colors = [palette[1], palette[1], palette[2], palette[1]]
    objects = [[((t-q)//2, (t+q)//2) for t in range(lo, hi+1, 2)]
               for q, lo, hi in specs]
    r0 = min(r for obj in objects for r, c in obj)
    c0 = min(c for obj in objects for r, c in obj)
    for serial in range(16):
        g = np.full((30, 30), params['background'], dtype=np.int8)
        g[0, :] = g[:, -1] = params['token']
        for obj, color in zip(objects, colors):
            for r, c in obj:
                g[r-r0+2+serial//4, c-c0+serial % 4] = color
        inp = np.rot90(g, rotation).copy()
        if _key(inp) not in seen:
            return dict(input=inp, output=solve(inp))
    raise AssertionError("more prior inputs than the legal count contract permits")


def _count(value, name, low, high):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one rule-consistent puzzle; explicit legal counts are honored."""
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(int(seed))
    colors = list(map(int, rng.permutation(10)))
    params = dict(token=colors[0], background=colors[1],
                  palette=colors[2:2+int(rng.integers(3, 7))],
                  rotation=int(rng.integers(4)), mixed_orientation=bool(rng.integers(2)))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', n), ('test', m)):
        for index in range(count):
            pair = _make_pair(rng, index, split == 'test', params, seen)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate visible semantics, conservation, evidence, and the length shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, roles, train_witnesses, train_colors = set(), set(), set(), set()
    lengths = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape preservation")
            _require(_key(inp) not in seen, "inputs must be distinct")
            seen.add(_key(inp))
            info = _read(inp)
            roles.add((info['token'], info['background']))
            g = info['grid']
            result = np.rot90(out, info['rotation'])
            guides = (g != info['token']) & (g != info['background'])
            _require(np.array_equal(result[guides], g[guides]), "preserve every guide")
            _require(np.count_nonzero(inp == info['token']) ==
                     np.count_nonzero(out == info['token']), "conserve tokens")
            enabled = [r for r in _slices(info) if r['enabled']]
            _require(bool(enabled), "at least one transfer per pair")
            _require(len({r['destination'] for r in enabled}) == len(enabled), "unique destinations")
            _require(np.count_nonzero(inp != out) == 2*len(enabled), "change only endpoints")
            changed = set(map(tuple, np.argwhere(g != result)))
            expected_changes = {p for r in enabled for p in (r['source'], r['destination'])}
            _require(changed == expected_changes, "transfer locations")
            for r in enabled:
                start, end = r['source'], r['destination']
                _require(sum(start) == sum(end), "perpendicular diagonal transport")
                _require(result[start] == info['background'] and result[end] == info['token'],
                         "transfer endpoint roles")
                _require(r['low']['color'] == r['high']['color'], "outer identity latch")
                _require(end == max(r['free']), "farthest available interior slot")
            _require(np.array_equal(out, solve(inp.copy())), "input-only solve agreement")
            witnesses = _witnesses(info, out)
            lengths[split].append(max(s['length'] for s in info['strokes']))
            if split == 'train':
                train_witnesses.update(witnesses)
                train_colors.update(r['low']['color'] for r in enabled)
            else:
                _require(witnesses <= train_witnesses, "untaught alternative-rule distinction")
                _require({'ignore_matching', 'pixel_centers'} <= witnesses,
                         "both parent concepts must affect each test")
    _require(len(roles) == 1, "token and background roles stay fixed within a puzzle")
    _require(set(_ALTERNATIVES) <= train_witnesses, "missing teaching witness")
    _require(len(train_colors) >= 2, "show matching with two guide identities")
    _require(min(lengths['test']) > max(lengths['train']), "test extends visible guide lengths")


def measurements(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _read(pair['input'])
            slices = _slices(info)
            strokes = info['strokes']
            occupied = (info['grid'] != info['token']) & (info['grid'] != info['background'])
            rr, cc = np.where(occupied)
            q0, t0 = min(s['q'] for s in strokes), min(s['lo'] for s in strokes)
            records.append(dict(split=split, shape=list(pair['input'].shape),
                                canonical_shape=list(info['grid'].shape), rotation=info['rotation'],
                                token=info['token'], background=info['background'],
                                count=len(strokes), colors=len({s['color'] for s in strokes}),
                                lengths=[s['length'] for s in strokes],
                                gaps=[b['q']-a['q'] for a, b in zip(strokes, strokes[1:])],
                                movers=sum(r['enabled'] for r in slices),
                                mismatches=sum(not r['matching'] for r in slices),
                                common=max(s['lo'] for s in strokes) <= min(s['hi'] for s in strokes),
                                bottom_edge=int(rr.max()) == info['grid'].shape[0]-1,
                                left_edge=int(cc.min()) == 0,
                                witnesses=sorted(_witnesses(info, pair['output'])),
                                geometry=[(s['q']-q0, s['lo']-t0, s['hi']-t0) for s in strokes]))
    return records


def self_test():
    """Bounded 42-seed variation audit plus explicit semantic regressions."""
    shapes, geometries, fingerprints, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        repeat = generate_puzzle(seed)
        fingerprint = tuple((_key(p['input']), _key(p['output']))
                            for split in ('train', 'test') for p in puzzle[split])
        _require(fingerprint == tuple((_key(p['input']), _key(p['output']))
                                      for split in ('train', 'test') for p in repeat[split]),
                 "whole-puzzle determinism")
        fingerprints.add(fingerprint)
        for record in measurements(puzzle):
            records.append(record)
            shapes.add(tuple(record['shape']))
            geometries.add(tuple(map(tuple, record['geometry'])))
    _require(len(fingerprints) == 42, "cross-seed variety")
    _require(len(shapes) >= 45 and len(geometries) >= 150, "structural variation")
    _require({2, 3, 4, 5, 6, 7} <= {r['count'] for r in records}, "guide-count variation")
    _require({0, 1, 2, 3} == {r['rotation'] for r in records}, "four launch directions")
    _require(len({r['token'] for r in records}) >= 8 and
             len({r['background'] for r in records}) >= 8, "palette role variation")
    _require(any(r['common'] for r in records) and any(not r['common'] for r in records),
             "common and staggered intervals")
    _require(any(r['bottom_edge'] for r in records) and any(r['left_edge'] for r in records),
             "visible terminal boundary contact")
    _require({1, 2, 3, 4} <= {r['colors'] for r in records}, "guide identity variation")
    _require(min(x for r in records for x in r['lengths']) == 2 and
             max(x for r in records for x in r['lengths']) == 13, "length variation")
    _require(min(x for r in records for x in r['gaps']) == 2 and
             max(x for r in records for x in r['gaps']) == 8, "spacing variation")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(103+n*4+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A hand-calculated continuous slice passes between guide pixel centers.
    g = np.zeros((10, 12), dtype=np.int8)
    g[0, :] = g[:, -1] = 9
    g[(2, 3, 4), (2, 3, 4)] = 3     # q=0, t=4..8
    g[(2, 3, 4), (6, 7, 8)] = 3     # q=4, t=8..12
    out = solve(g)
    _require(out[0, 8] == 0 and out[3, 5] == 9, "inclusive shared endpoint")
    # Extend the first stroke, then check an odd slice with no boundary pixels.
    g[5, 5] = 3
    out = solve(g)
    _require(out[0, 9] == 0 and out[4, 5] == 9,
             "between-pixel continuous support and farthest slot")
    mismatch = g.copy()
    mismatch[mismatch == 3] = 4
    mismatch[(2, 3, 4), (6, 7, 8)] = 6
    _require(np.array_equal(solve(mismatch), mismatch), "different extreme identities stay latched")
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(g, k)), np.rot90(out, k)), "rotation covariance")
    # The reserve construction is checked even when rejection never exhausts.
    params = dict(token=9, background=0, palette=[3, 5, 6])
    for is_test, index in ((False, 0), (False, 1), (True, 0)):
        seen = set()
        for serial in range(14):
            pair = _fallback_pair(index, is_test, params, 0, seen)
            _require(_key(pair['input']) not in seen, "reserve placement uniqueness")
            seen.add(_key(pair['input']))
            info = _read(pair['input'])
            required = ({'consecutive', 'pixel_centers', 'first_slot'}
                        if not is_test and index == 0 else set(_ALTERNATIVES))
            _require(required <= _witnesses(info), "reserve teaching evidence")
    p = generate_puzzle(0, 2, 1)
    p['test'][0]['output'] = p['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("unchanged corrupt output accepted")
    # Save only descriptive, compact data when explicitly running self_test.
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(fingerprints),
                shapes=len(shapes), geometries=len(geometries), records=records)


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, separators=(',', ':')))
