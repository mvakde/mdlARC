"""Registered loom: match partial color tapes and project their ink into a ribbon.

Only Python's standard library and NumPy are used. No files are read or written
by generation, solving, validation, or the bounded, streaming self-test.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 103985969, 14988068, 815568507, 406919642, 946132315,
              361947888, 718623862, 558544070, 912712327, 664306509, 990165447,
              519557305, 725927161, 772807936, 836486193, 635999056, 796763508,
              93836250, 536241782, 624993824, 409224730, 444193540, 920118871,
              912786192, 408369225, 713879514, 54695859, 63020096, 718562938,
              497280182, 383094204, 469115000, 213662476, 183563712, 945191198,
              232837492, 919273122, 801992499, 622052, 4676189, 437565593)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _qt(r, c, slope):
    return c - slope*r, r + slope*c


def _rc(q, t, slope):
    if (q-t) % 2:
        raise ValueError("diagonal coordinates have incompatible parity")
    return (t-slope*q)//2, (q+slope*t)//2


def _components(grid, background):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {}
        while stack:
            p = stack.pop()
            cells[p] = int(grid[p])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    v = p[0]+dr, p[1]+dc
                    if v in remaining:
                        remaining.remove(v)
                        stack.append(v)
        parts.append(cells)
    return parts


def _read_fragment(cells, slope, rail, support):
    tape_cells = {p: v for p, v in cells.items() if v not in (rail, support)}
    if len(tape_cells) < 9 or len(set(tape_cells.values())) < 2:
        raise ValueError("a fragment needs a multicolor tape")
    coords = [(*_qt(*p, slope), v) for p, v in tape_cells.items()]
    q0 = min(q for q, t, v in coords)
    if max(q for q, t, v in coords) != q0+1:
        raise ValueError("tape must be a two-diagonal staircase")
    tape = {}
    for q, t, color in coords:
        if q != q0+(t-q0) % 2 or t in tape:
            raise ValueError("invalid staircase")
        tape[t] = color
    low, high = min(tape), max(tape)
    if high-low+1 != len(tape):
        raise ValueError("tape is not continuous")
    by_q = {}
    crossbar = set()
    for p, v in cells.items():
        q, t = _qt(*p, slope)
        if v == rail:
            by_q.setdefault(q, []).append(t)
        elif v == support:
            crossbar.add((q, t))
    if not 1 <= len(by_q) <= 2 or not crossbar:
        raise ValueError("need one or two boundaries and a crossbar")
    levels = {t for q, t in crossbar}
    if len(levels) != 1:
        raise ValueError("crossbar must lie on one perpendicular diagonal")
    anchor = levels.pop()
    strokes = []
    for q, times in sorted(by_q.items()):
        times.sort()
        if (len(times) < 2 or times != list(range(times[0], times[-1]+1, 2))
                or not low <= times[0] <= anchor <= times[-1] <= high
                or q0-3 < q < q0+4):
            raise ValueError("invalid boundary interval")
        strokes.append(dict(q=q, lo=times[0], hi=times[-1]))
    tape_q = q0+(anchor-q0) % 2
    qs = [tape_q]+list(by_q)
    if any((q-anchor) % 2 for q in qs):
        raise ValueError("crossbar cannot reach that boundary")
    expected = {(q, anchor) for q in range(min(qs), max(qs)+1, 2)
                if q != tape_q and q not in by_q}
    if crossbar != expected:
        raise ValueError("crossbar must connect the tape to its boundaries exactly")
    return dict(q0=q0, tape=tape, strokes=strokes)


def _parse_input(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[counts.argmax()])
    if counts.max()*2 <= grid.size:
        raise ValueError("background must be a strict majority")
    parts = _components(grid, background)
    if not 2 <= len(parts) <= 4:
        raise ValueError("expected two to four separated loom fragments")
    palette = [int(v) for v in colors if v != background]
    candidates = []
    for slope in (-1, 1):
        for rail in palette:
            for support in palette:
                if support == rail:
                    continue
                try:
                    fragments = [_read_fragment(c, slope, rail, support) for c in parts]
                except ValueError:
                    continue
                candidates.append(dict(background=background, rail=rail, support=support,
                                       slope=slope, fragments=fragments))
    if len(candidates) != 1:
        raise ValueError("color roles and tape direction must be uniquely recognizable")
    return candidates[0]


def _matches(a, b):
    """All exact, same-orientation tape matches of at least five cells.

    q alignment is fixed by the staircase, not by input locations. A full
    shared interval must agree; neither overlap area nor crop area is scored.
    """
    ta, tb = a['tape'], b['tape']
    dq = a['q0']-b['q0']
    found = []
    for dt in range(min(ta)-max(tb)+4, max(ta)-min(tb)-3):
        if (dq-dt) % 2:
            continue
        lo, hi = max(min(ta), min(tb)+dt), min(max(ta), max(tb)+dt)
        if (hi-lo+1 >= 5 and len({ta[t] for t in range(lo, hi+1)}) >= 2
                and all(ta[t] == tb[t-dt] for t in range(lo, hi+1))):
            found.append((dt, hi-lo+1))
    return found


def _align(parsed):
    fragments = parsed['fragments']
    adjacency = [[] for _ in fragments]
    overlaps = []
    for i, j in combinations(range(len(fragments)), 2):
        matches = _matches(fragments[i], fragments[j])
        if len(matches) > 1:
            raise ValueError("ambiguous repeated tape key")
        if matches:
            dt, width = matches[0]
            adjacency[i].append((j, dt))
            adjacency[j].append((i, -dt))
            overlaps.append(width)
    offsets, stack = {0: 0}, [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            value = offsets[i]+delta
            if j in offsets:
                if offsets[j] != value:
                    raise ValueError("inconsistent tape registrations")
            else:
                offsets[j] = value
                stack.append(j)
    if len(offsets) != len(fragments):
        raise ValueError("registration graph is disconnected")
    tape, strokes, translations = {}, [], []
    common_q = fragments[0]['q0']
    for i, fragment in enumerate(fragments):
        dq, dt = common_q-fragment['q0'], offsets[i]
        translations.append(_rc(dq, dt, parsed['slope']))
        for t, v in fragment['tape'].items():
            if t+dt in tape and tape[t+dt] != v:
                raise ValueError("registered tape colors conflict")
            tape[t+dt] = v
        for s in fragment['strokes']:
            strokes.append(dict(q=s['q']+dq, lo=s['lo']+dt, hi=s['hi']+dt, owner=i))
    if len({s['q'] for s in strokes}) != len(strokes):
        raise ValueError("boundaries must have distinct transverse positions")
    sequence = [tape[t] for t in range(min(tape), max(tape)+1)]
    keys = [tuple(sequence[i:i+5]) for i in range(len(sequence)-4)]
    if len(set(keys)) != len(keys) or any(len(set(k)) < 2 for k in keys):
        raise ValueError("five-cell tape keys must be nonconstant and nonrepeating")
    return dict(parsed, tape=tape, strokes=strokes, translations=translations,
                overlaps=overlaps, edges=sum(map(len, adjacency))//2)


def _domain(info):
    pixels = {_rc(s['q'], t, info['slope']) for s in info['strokes']
              for t in range(s['lo'], s['hi']+1, 2)}
    r0, c0 = min(r for r, c in pixels), min(c for r, c in pixels)
    h = max(r for r, c in pixels)-r0+1
    w = max(c for r, c in pixels)-c0+1
    if not 1 <= h <= 30 or not 1 <= w <= 30:
        raise ValueError("fabric does not fit an ARC grid")
    rows, cols = np.indices((h, w))
    q, t = _qt(rows+r0, cols+c0, info['slope'])
    edge = np.zeros((h, w), dtype=bool)
    for r, c in pixels:
        edge[r-r0, c-c0] = True
    return q, t, edge


def _finish(info, t, edge, mask):
    output = np.full(t.shape, info['background'], dtype=np.int8)
    mask = mask & ~edge
    for level in np.unique(t[mask]):
        output[mask & (t == level)] = info['tape'][int(level)]
    output[edge] = info['rail']
    return output, mask


def _sweep(info):
    q, t, edge = _domain(info)
    mask = np.zeros(t.shape, dtype=bool)
    for level in range(int(t.min()), int(t.max())+1):
        active = [s['q'] for s in info['strokes'] if s['lo'] <= level <= s['hi']]
        if len(active) >= 2:
            mask |= (t == level) & (q > min(active)) & (q < max(active))
    return _finish(info, t, edge, mask)


def _pairwise(info, alternative=''):
    """Label construction and explicit weaker alternatives, using segment pairs.

    The ordinary version unions pairwise parallelograms instead of calling the
    scanline solver. This is a self-authored cross-check, not an external oracle.
    """
    q, t, edge = _domain(info)
    strokes = sorted(info['strokes'], key=lambda s: s['q'])
    pairs = list(combinations(strokes, 2))
    if alternative == 'consecutive':
        pairs = list(zip(strokes, strokes[1:]))
    mask = np.zeros(t.shape, dtype=bool)
    for a, b in pairs:
        if alternative == 'local' and a['owner'] != b['owner']:
            continue
        lo, hi = max(a['lo'], b['lo']), min(a['hi'], b['hi'])
        m = (q > a['q']) & (q < b['q']) & (t >= lo) & (t <= hi)
        if alternative == 'pixel_centers':
            m &= ((t-a['q']) % 2 == 0) & ((t-b['q']) % 2 == 0)
        mask |= m
    if alternative == 'ignore_endpoints':
        mask = ((q > min(s['q'] for s in strokes)) & (q < max(s['q'] for s in strokes))
                & (t >= min(s['lo'] for s in strokes)) & (t <= max(s['hi'] for s in strokes)))
    return _finish(info, t, edge, mask)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all color roles, translations, intervals, and ink from the input."""
    return _sweep(_align(_parse_input(input_grid)))[0]


def _features(info):
    expected, fill = _pairwise(info)
    differences = {name: int(np.count_nonzero(expected != _pairwise(info, name)[0]))
                   for name in ('local', 'consecutive', 'pixel_centers', 'ignore_endpoints')}
    return dict(fragments=len(info['fragments']), strokes=len(info['strokes']),
                slope=info['slope'], tape_length=len(info['tape']),
                edges=info['edges'], min_overlap=min(info['overlaps']),
                ink_colors=len(set(info['tape'].values())),
                fill_colors=len(np.unique(expected[fill])), filled=int(fill.sum()),
                multi=any(len(f['strokes']) == 2 for f in info['fragments']),
                differences=differences, output_shape=expected.shape)


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return {(r-r0, c-c0): v for (r, c), v in cells.items()}


def _pack(parts, background, rng):
    """Random separated foreground packing, without a fixed slot arrangement."""
    parts = [_normalize(p) for p in parts]
    shapes = [(max(r for r, c in p)+1, max(c for r, c in p)+1) for p in parts]
    mh, mw = max(h for h, w in shapes), max(w for h, w in shapes)
    total = sum(map(len, parts))
    if max(mh, mw) > 30:
        raise ValueError("fragment too large")
    canvases = [(int(rng.integers(mh, 31)), int(rng.integers(mw, 31))) for _ in range(14)]
    canvases += [(30, 30)]*6
    for h, w in canvases:
        if h*w <= 2*total:
            continue
        occupied = np.zeros((h, w), dtype=bool)
        grid = np.full((h, w), background, dtype=np.int8)
        order = sorted(range(len(parts)), key=lambda i: -(len(parts[i])+float(rng.random())*12))
        for i in order:
            ph, pw = shapes[i]
            feasible = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in parts[i]:
                feasible &= ~occupied[r:r+h-ph+1, c:c+w-pw+1]
            locations = np.argwhere(feasible)
            if len(locations) == 0:
                break
            r0, c0 = map(int, locations[int(rng.integers(len(locations)))])
            for (r, c), value in parts[i].items():
                r, c = r+r0, c+c0
                grid[r, c] = value
                occupied[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True
        else:
            return grid
    raise ValueError("separated fragments do not fit")


def _program(rng, count, ink):
    for _ in range(128):
        seq = [int(v) for v in rng.choice(ink, count)]
        keys = [tuple(seq[i:i+5]) for i in range(count-4)]
        if len(set(keys)) == len(keys) and all(len(set(k)) >= 2 for k in keys):
            return dict(enumerate(seq))
    raise ValueError("could not sample a nonperiodic color program")


def _draft(rng, roles, n, extra, slope, chain):
    background, rail, support, ink_pool = roles
    widths = [int(rng.integers(11, 16)) for _ in range(n)]
    starts = [0]
    for i in range(1, n):
        gap = int(rng.integers(5, 7)) if chain else int(rng.integers(2, 7))
        starts.append(starts[-1]+gap)
    if chain:
        # Direct neighbors share >=5 cells; non-neighbors need not share a key.
        widths = [11]*n
    end = max(a+w for a, w in zip(starts, widths))
    ink_count = int(rng.integers(2, min(5, len(ink_pool))+1))
    ink = [int(v) for v in rng.choice(ink_pool, ink_count, replace=False)]
    tape = _program(rng, end, ink)
    owners = [1]*n
    for i in rng.choice(n, extra, replace=False):
        owners[int(i)] += 1
    available = set(range(-12, -3)) | set(range(5, 14))
    parts, strokes = [], []
    for i, (start, width) in enumerate(zip(starts, widths)):
        parity = int(rng.integers(2))
        candidates = sorted(q for q in available if q % 2 == parity)
        qs = [int(v) for v in rng.choice(candidates, owners[i], replace=False)]
        available.difference_update(qs)
        anchors = [t for t in range(start+2, start+width-2) if t % 2 == parity]
        anchor = int(rng.choice(anchors))
        local = []
        for q in qs:
            before = int(rng.integers(min(5, (anchor-start)//2)+1))
            after = int(rng.integers(min(5, (start+width-1-anchor)//2)+1))
            if before+after == 0:
                if anchor+2 < start+width:
                    after = 1
                else:
                    before = 1
            local.append(dict(q=q, lo=anchor-2*before, hi=anchor+2*after, owner=i))
        cells = {_rc(t % 2, t, slope): tape[t] for t in range(start, start+width)}
        for s in local:
            for t in range(s['lo'], s['hi']+1, 2):
                cells[_rc(s['q'], t, slope)] = rail
        tq = anchor % 2
        for q in range(min([tq]+qs), max([tq]+qs)+1, 2):
            cells.setdefault(_rc(q, anchor, slope), support)
        parts.append(cells)
        strokes.extend(local)
    # Construction metadata is used only for labels and candidate filtering.
    # The returned puzzle contains just arrays; solve reconstructs everything.
    info = dict(background=background, rail=rail, support=support, slope=slope,
                tape=tape, strokes=strokes)
    return parts, info


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(v) for v in rng.permutation(10)]
    roles = (*palette[:3], palette[3:3+int(rng.integers(3, 8))])
    base_slope = int(rng.choice([-1, 1]))
    mixed = bool(rng.integers(2))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index == 0:
                n, extra, chain = 2, 0, False
            elif split == 'train' and index == 1:
                n, extra, chain = 3, 1, True
            else:
                n = int(rng.integers(3 if split == 'test' else 2, 5))
                extra = int(rng.integers(0, min(n, 6-n)+1))
                chain = bool(rng.integers(2))
            slope = base_slope
            if mixed:
                slope *= (-1 if index % 2 else 1) if split == 'train' else int(rng.choice([-1, 1]))
            for _ in range(192):
                try:
                    parts, intended = _draft(rng, roles, n, extra, slope, chain)
                    output, fill = _pairwise(intended)
                    if fill.sum() < 6 or len(np.unique(output[fill])) < 2:
                        continue
                    if np.array_equal(output, _pairwise(intended, 'local')[0]):
                        continue
                    if split == 'train' and index == 0:
                        if any(np.array_equal(output, _pairwise(intended, a)[0])
                               for a in ('pixel_centers', 'ignore_endpoints')):
                            continue
                    if split == 'test' or index == 1:
                        if np.array_equal(output, _pairwise(intended, 'consecutive')[0]):
                            continue
                    inp = _pack(parts, roles[0], rng)
                    if _key(inp) in seen:
                        continue
                    recovered = _align(_parse_input(inp))
                    if not np.array_equal(_sweep(recovered)[0], output):
                        raise AssertionError("construction and input-only reconstruction disagree")
                except ValueError:
                    continue
                puzzle[split].append(dict(input=inp, output=output))
                seen.add(_key(inp))
                break
            else:
                raise RuntimeError("bounded loom construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned pixels, semantic invariants, and visible training witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, structural_roles = set(), set()
    features = dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _align(_parse_input(inp))
            structural_roles.add((info['background'], info['rail'], info['support']))
            predicted, fill = _sweep(info)
            assert np.array_equal(out, predicted), "incorrect registered fabric"
            assert np.array_equal(out, _pairwise(info)[0]), "sweep/pairwise disagreement"
            assert info['support'] not in set(map(int, np.unique(out))), "crossbar retained"
            assert np.any(out[0] != info['background']) and np.any(out[-1] != info['background'])
            assert np.any(out[:, 0] != info['background']) and np.any(out[:, -1] != info['background'])
            f = _features(info)
            assert f['filled'] >= 6 and f['fill_colors'] >= 2, "color program not demonstrated"
            assert f['differences']['local'] > 0, "fragments do not interact"
            features[split].append(f)
    assert len(structural_roles) == 1, "structural palette changes within puzzle"
    for alternative in ('consecutive', 'pixel_centers', 'ignore_endpoints'):
        if any(f['differences'][alternative] for f in features['test']):
            assert any(f['differences'][alternative] for f in features['train']), f"missing {alternative} witness"
    if any(f['multi'] for f in features['test']):
        assert any(f['multi'] for f in features['train']), "two-boundary fragment not taught"
    if any(f['edges'] < f['fragments']*(f['fragments']-1)//2 for f in features['test']):
        assert any(f['edges'] < f['fragments']*(f['fragments']-1)//2 for f in features['train']), "chain registration not taught"


def self_test():
    """Stream 42 specified seeds; retain only hashes, sets, and small counters."""
    fingerprints, shapes, layouts, palette_roles, slopes, counts, overlaps = set(), set(), set(), set(), set(), set(), set()
    pair_count = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _align(_parse_input(pair['input']))
                f = _features(info)
                pair_count += 1
                shapes.add((pair['input'].shape, pair['output'].shape))
                counts.add((f['fragments'], f['strokes']))
                slopes.add(f['slope'])
                overlaps.update(info['overlaps'])
                palette_roles.add((info['background'], info['rail'], info['support']))
                q0 = min(s['q'] for s in info['strokes'])
                t0 = min(s['lo'] for s in info['strokes'])
                layouts.add(tuple(sorted((s['q']-q0, s['lo']-t0, s['hi']-t0) for s in info['strokes'])))
                for role in ('input', 'output'):
                    a = pair[role]
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
        fingerprints.add(digest.digest())
    assert pair_count == 210 and len(fingerprints) == 42
    assert len(shapes) >= 100 and len(layouts) >= 180
    assert slopes == {-1, 1}
    assert {n for n, k in counts} == {2, 3, 4}
    assert {2, 3, 4, 5, 6} <= {k for n, k in counts}
    assert len(palette_roles) >= 30 and min(overlaps) == 5
    p = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    assert all(_key(a[r]) == _key(b[r]) for split in p
               for a, b in zip(p[split], again[split]) for r in ('input', 'output'))
    first = p['train'][0]
    parsed = _parse_input(first['input'])
    pieces = _components(first['input'], parsed['background'])
    repacked = _pack(pieces, parsed['background'], np.random.default_rng(135))
    assert np.array_equal(solve(repacked), first['output'])
    permutation = np.array([7, 2, 9, 0, 4, 1, 8, 6, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[first['input']]), permutation[first['output']])
    assert np.array_equal(solve(first['input'].T), first['output'].T)
    # Two correctly labeled direct registrations cannot teach the skipping rule.
    insufficient = dict(train=[first, dict(input=repacked, output=first['output'].copy())], test=p['test'])
    try:
        validate_generated_puzzle(insufficient)
    except AssertionError as exc:
        assert 'witness' in str(exc) or 'taught' in str(exc)
    else:
        raise AssertionError("inadequate training accepted")
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = corrupt['test'][0]['output']
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(41, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(fingerprints),
                input_output_shape_pairs=len(shapes), boundary_layouts=len(layouts),
                fragment_stroke_counts=sorted(counts), slopes=sorted(slopes),
                structural_palettes=len(palette_roles), overlap_lengths=sorted(overlaps),
                independent_solver=False, blind_prediction=False)
