"""Elastic diagonal loom: interval brackets control a two-scale texture.

Only Python/NumPy are required. Generation and self_test perform no disk I/O.
At most 96 small candidates are considered at a time; no traces are retained.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " outside its legal range")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions"
    assert np.issubdtype(g.dtype, np.integer), "integer grid required"
    assert np.all((0 <= g) & (g <= 9)), "grid colors"
    return g


def _key(g):
    return (g.shape, g.astype(np.uint8).tobytes())


def _strokes(canvas, background):
    """Read maximal monochrome runs, accepting exactly one common slope."""
    cells = np.argwhere(canvas != background)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(int(c - slope*r), []).append(int(r))
        strokes, valid = [], True
        for q, rows in sorted(lines.items()):
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1] + 1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                inks = {int(canvas[r, q+slope*r]) for r in run}
                if len(run) < 2 or len(inks) != 1:
                    valid = False
                    break
                strokes.append((q, 2*run[0]+slope*q, 2*run[-1]+slope*q,
                                inks.pop()))
            if not valid:
                break
        if valid and len(strokes) >= 2:
            candidates.append((slope, strokes))
    assert len(candidates) == 1, "parallel strokes must have a unique slope"
    return candidates[0]


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, -1])
    divider_rows = [i for i, row in enumerate(g)
                    if int(row[0]) != bg and np.all(row == row[0])]
    assert len(divider_rows) == 1, "one full separator row required"
    a = divider_rows[0]
    assert 1 <= a <= 4 and a+1 < g.shape[0], "swatch height"
    sep = int(g[a, 0])
    columns = np.flatnonzero(np.all(g[:a] == sep, axis=0))
    assert len(columns) == 1, "one header separator column required"
    b = int(columns[0])
    assert 2 <= b <= 5 and g.shape[1] >= 2*b+2, "swatch width and padding"
    carrier = g[:a, :b]
    stencil = g[:a, b+1:2*b+1]
    assert np.all((carrier != bg) & (carrier != sep)), "occupied carrier"
    assert np.all(stencil != sep), "separator cannot be stencil ink"
    assert np.all(g[:a, 2*b+1:] == bg), "header background sample"
    canvas = g[a+1:]
    assert not np.any(canvas == sep), "separator belongs only to header"
    slope, strokes = _strokes(canvas, bg)
    return dict(bg=bg, sep=sep, a=a, b=b, carrier=carrier, stencil=stencil,
                canvas=canvas, slope=slope, strokes=strokes,
                t0=min(s[1] for s in strokes))


def _sections(info):
    """Each section includes marker slots: markers never compress the texture."""
    canvas, slope = info['canvas'], info['slope']
    rr, cc = np.indices(canvas.shape)
    qq, tt = cc-slope*rr, rr+slope*cc
    result = []
    for t in range(int(tt.min()), int(tt.max())+1):
        active = [s[0] for s in info['strokes'] if s[1] <= t <= s[2]]
        if len(active) < 2:
            continue
        lo, hi = min(active), max(active)
        positions = np.argwhere((tt == t) & (qq > lo) & (qq < hi))
        if len(positions):
            positions = sorted((int(r), int(c)) for r, c in positions)
            positions.sort(key=lambda p: int(qq[p]))
            result.append((t, lo, hi, positions))
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible swatches, visible endpoints and canvas geometry."""
    info = _decode(input_grid)
    out = info['canvas'].copy()
    a, b = info['a'], info['b']
    for t, _, _, positions in _sections(info):
        v = (t-info['t0']) % a
        n = len(positions)
        for j, p in enumerate(positions):
            if info['canvas'][p] != info['bg']:
                continue
            ink = info['stencil'][v, j % b]
            out[p] = info['carrier'][v, j*b//n] if ink == info['bg'] else ink
    return out


def _reference(info):
    """Cell-wise arithmetic check, separate from solve's scanline slot lists."""
    canvas, slope = info['canvas'], info['slope']
    out = canvas.copy()
    for r in range(canvas.shape[0]):
        for c in range(canvas.shape[1]):
            if int(canvas[r, c]) != info['bg']:
                continue
            q, t = c-slope*r, r+slope*c
            active = [s[0] for s in info['strokes'] if s[1] <= t <= s[2]]
            if len(active) < 2 or not min(active) < q < max(active):
                continue
            first = min(active)+1
            first += (t-first) % 2
            last = max(active)-1
            last -= (last-t) % 2
            n, j = (last-first)//2+1, (q-first)//2
            v = (t-info['t0']) % info['a']
            ink = int(info['stencil'][v, j % info['b']])
            out[r, c] = info['carrier'][v, j*info['b']//n] if ink == info['bg'] else ink
    return out


_ALTERNATIVES = ('ignore_a', 'solid_fill', 'tile_carrier', 'ignore_stencil',
                 'resize_stencil', 'global_phase', 'pixel_centers',
                 'consecutive', 'compress_markers', 'fixed_ribbon',
                 'center_resize', 'longitudinal_phase')
_CORE_EVIDENCE = frozenset(_ALTERNATIVES[:-1])


def _alternative(info, mode):
    canvas, bg = info['canvas'], info['bg']
    out = canvas.copy()
    sections = _sections(info)
    if mode == 'ignore_a':
        rr, cc = np.indices(canvas.shape)
        tt, qq = rr+info['slope']*cc, cc-info['slope']*rr
        sections = []
        for t in range(int(tt.min()), int(tt.max())+1):
            ps = [tuple(map(int, p)) for p in np.argwhere(tt == t)]
            ps.sort(key=lambda p: int(qq[p]))
            sections.append((t, -100, 100, ps))
    ordered = sorted(info['strokes'])
    for t, lo, hi, positions in sections:
        if mode == 'compress_markers':
            positions = [p for p in positions if canvas[p] == bg]
        n = len(positions)
        v = (t-info['t0']+(mode == 'longitudinal_phase')) % info['a']
        global_first = None
        if mode == 'global_phase':
            rr, cc = np.indices(canvas.shape)
            tt, qq = rr+info['slope']*cc, cc-info['slope']*rr
            global_first = int(qq[tt == t].min())
        for j, p in enumerate(positions):
            if canvas[p] != bg:
                continue
            q = p[1]-info['slope']*p[0]
            if mode == 'pixel_centers' and ((t-lo) % 2 or (t-hi) % 2):
                continue
            if mode == 'consecutive' and not any(
                x[1] <= t <= x[2] and y[1] <= t <= y[2] and x[0] < q < y[0]
                for x, y in zip(ordered, ordered[1:])):
                continue
            if mode == 'fixed_ribbon':
                first = min(s[0] for s in ordered)+1
                first += (t-first) % 2
                last = max(s[0] for s in ordered)-1
                last -= (last-t) % 2
                j, n = (q-first)//2, (last-first)//2+1
            u = j % info['b'] if mode == 'tile_carrier' else j*info['b']//n
            if mode == 'center_resize':
                u = (2*j+1)*info['b']//(2*n)
            base = int(info['carrier'][v, u])
            su = j*info['b']//n if mode == 'resize_stencil' else j % info['b']
            if mode == 'global_phase':
                # Native-period stencil anchored to the whole diagonal scanline.
                su = (q-global_first)//2 % info['b']
            ink = int(info['stencil'][v, su])
            if mode == 'ignore_stencil':
                ink = bg
            if mode == 'solid_fill':
                out[p] = info['carrier'][0, 0]
            else:
                out[p] = base if ink == bg else ink
    return out


def _evidence(pair):
    info = _decode(pair['input'])
    return {name for name in _ALTERNATIVES
            if not np.array_equal(pair['output'], _alternative(info, name))}


def _max_width(info):
    return max((len(s[3]) for s in _sections(info)), default=0)


def _specs(rng, span, count):
    # Free transverse gaps constrained only by total span and recognizability.
    increments = np.full(count-1, 3, dtype=int)
    for _ in range(span-3*(count-1)):
        increments[int(rng.integers(count-1))] += 1
    qs = np.r_[0, np.cumsum(increments)].tolist()
    mode = int(rng.choice(6, p=[.18, .32, .12, .12, .10, .16]))
    intervals = []
    for i, q in enumerate(qs):
        length = int(rng.integers(2, 10))
        if mode == 0:
            lo = -2*int(rng.integers(length))+q % 2
        elif mode == 2:
            lo = q % 2 if rng.random() < .5 else -2*(length-1)+q % 2
        elif mode == 3:
            lo = 3*i - 2*count
            lo -= (lo-q) % 2
        elif mode == 4 and count >= 4:
            lo = (-12 if i < count//2 else 4)+q % 2
            length = int(rng.integers(2, 5))
        else:
            lo = int(rng.integers(-10, 7))
            lo -= (lo-q) % 2
        intervals.append((lo, lo+2*(length-1)))
    if mode == 1 and count >= 3:
        # A bounded but variable witness: the middle rail is absent early;
        # later the first outer rail stops and the active width contracts.
        middle = int(rng.integers(1, count-1))
        chosen = (0, middle, count-1)
        raw = [(-int(rng.integers(6, 10)), int(rng.integers(2, 5))),
               (int(rng.integers(0, 3)), int(rng.integers(6, 10))),
               (-int(rng.integers(10, 13)), int(rng.integers(10, 13)))]
        if rng.random() < .5:
            raw[0], raw[2] = raw[2], raw[0]
        for i, (lo, hi) in zip(chosen, raw):
            intervals[i] = (lo-(lo-qs[i]) % 2, hi+(qs[i]-hi) % 2)
    return [(q, lo, hi) for q, (lo, hi) in zip(qs, intervals)]


def _swatches(rng, a, b, bg, inks):
    k = int(rng.integers(1, min(5, len(inks))+1))
    colors = rng.choice(inks, k, replace=False)
    rr, cc = np.indices((a, b))
    kind = int(rng.integers(5))
    if kind == 0:
        labels = np.tile(rng.integers(k, size=b), (a, 1))
    elif kind == 1:
        labels = (rr+cc) % k
    elif kind == 2:
        labels = np.tile(rng.integers(k, size=a)[:, None], (1, b))
    elif kind == 3:
        labels = (rr//int(rng.integers(1, a+1)) + cc//int(rng.integers(1, b))) % k
    else:
        labels = rng.integers(k, size=(a, b))
    carrier = colors[labels].astype(np.int8)
    stencil = np.full((a, b), bg, dtype=np.int8)
    if rng.random() < .10:
        return carrier, stencil
    style = int(rng.integers(5))
    if style == 0:
        mask = rng.random((a, b)) < rng.uniform(.15, .80)
    elif style == 1:
        mask = (rr+cc+int(rng.integers(2))) % 2 == 0
    elif style == 2:
        mask = cc == int(rng.integers(b))
    elif style == 3:
        mask = rr == int(rng.integers(a))
    else:
        mask = (rr == 0) | (rr == a-1) | (cc == 0) | (cc == b-1)
    if mask.all():
        mask.ravel()[int(rng.integers(a*b))] = False
    if not mask.any():
        mask.ravel()[int(rng.integers(a*b))] = True
    stencil_inks = rng.choice(inks, int(rng.integers(1, min(4, len(inks))+1)), replace=False)
    stencil[mask] = rng.choice(stencil_inks, int(mask.sum()))
    return carrier, stencil


def _sample_pair(rng, bg, sep, inks, width_limit, test):
    """Bounded rejection is in memory; one candidate is returned or discarded."""
    for _ in range(160):
        a, b = int(rng.integers(1, 5)), int(rng.integers(2, 6))
        low, high = ((2*width_limit+4, min(36, 2*width_limit+14)) if test
                     else (6, 2*width_limit+1))
        span = int(rng.integers(low, high+1))
        count = int(rng.integers(2, min(7, span//3+1)+1))
        specs = _specs(rng, span, count)
        slope = int(rng.choice([-1, 1]))
        strokes = []
        for q, lo, hi in specs:
            strokes.append([((t-slope*q)//2, (q+slope*t)//2)
                            for t in range(lo, hi+1, 2)])
        cells = np.array([p for stroke in strokes for p in stroke])
        mins, maxs = cells.min(axis=0), cells.max(axis=0)
        h, w = map(int, maxs-mins+1)
        hcap = 29-a
        w = max(w, 2*b+2)
        if h > hcap or w > 30:
            continue
        h += int(rng.integers(0, min(4, hcap-h)+1))
        w += int(rng.integers(0, min(4, 30-w)+1))
        offset = [int(rng.integers(h-int(maxs[0]-mins[0]))),
                  int(rng.integers(w-int(maxs[1]-mins[1])))]
        canvas = np.full((h, w), bg, dtype=np.int8)
        marker = int(rng.choice(inks))
        for stroke in strokes:
            for r, c in stroke:
                canvas[r-int(mins[0])+offset[0], c-int(mins[1])+offset[1]] = marker
        try:
            _strokes(canvas, bg)
        except AssertionError:
            continue
        carrier, stencil = _swatches(rng, a, b, bg, inks)
        grid = np.full((a+1+h, w), bg, dtype=np.int8)
        grid[:a, :b], grid[:a, b+1:2*b+1] = carrier, stencil
        grid[:a, b] = sep
        grid[a] = sep
        grid[a+1:] = canvas
        info = _decode(grid)
        width = _max_width(info)
        if not width or (test and width <= width_limit):
            continue
        if not test and width > width_limit:
            continue
        pair = dict(input=grid, output=_reference(info))
        if not np.any(pair['output'] != canvas):
            continue
        return pair
    raise RuntimeError("bounded geometric sampler exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _integer(num_train, 2, 10, 'num_train')
    num_test = _integer(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    bg, sep = palette[:2]
    inks = palette[2:2+int(rng.integers(3, 9))]
    width_limit = int(rng.integers(6, 11))
    candidates, covers, train = [], [], None
    # Select demonstrations by visible counterexamples, not by pair index.
    for _ in range(96):
        p = _sample_pair(rng, bg, sep, inks, width_limit, False)
        cover = _evidence(p)
        for j in rng.permutation(len(candidates)):
            j = int(j)
            if (_CORE_EVIDENCE | {'longitudinal_phase'}) <= cover | covers[j]:
                other = candidates[j]
                if (_decode(other['input'])['b'] != _decode(p['input'])['b']
                        and len(_decode(other['input'])['strokes']) != len(_decode(p['input'])['strokes'])):
                    train = [other, p]
                    break
        if train is not None:
            break
        candidates.append(p)
        covers.append(cover)
    if train is None:
        raise RuntimeError("bounded demonstration selection exhausted")
    puzzle = dict(train=train, test=[])
    seen = {_key(p['input']) for p in train}
    for split, count in (('train', num_train), ('test', num_test)):
        while len(puzzle[split]) < count:
            for _ in range(48):
                p = _sample_pair(rng, bg, sep, inks, width_limit, split == 'test')
                if _key(p['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            puzzle[split].append(p)
            seen.add(_key(p['input']))
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, widths, evidence = set(), set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            roles.add((info['bg'], info['sep']))
            assert out.shape == info['canvas'].shape, "return only the canvas"
            assert np.array_equal(out, solve(inp)), "incorrect output"
            assert np.array_equal(out, _reference(info)), "arithmetic reconstruction disagrees"
            marked = info['canvas'] != info['bg']
            assert np.array_equal(out[marked], info['canvas'][marked]), "preserve rails"
            bracketed = np.zeros(out.shape, dtype=bool)
            for _, _, _, positions in _sections(info):
                for p in positions:
                    bracketed[p] = True
            assert np.array_equal(out != info['bg'], marked | bracketed), "exact woven footprint"
            assert np.any(out != info['canvas']), "at least one woven cell"
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "copy visible inks"
            widths[split].append(_max_width(info))
            if split == 'train':
                evidence |= _evidence(pair)
    assert len(roles) == 1, "background and separator stay fixed within a puzzle"
    assert _CORE_EVIDENCE | {'longitudinal_phase'} <= evidence, "missing teaching evidence: " + str(
        (_CORE_EVIDENCE | {'longitudinal_phase'}) - evidence)
    assert min(widths['test']) > max(widths['train']), "test must extrapolate ribbon width"
    train_info = [_decode(p['input']) for p in puzzle['train']]
    assert len({i['b'] for i in train_info}) >= 2, "vary native periods in training"
    assert len({len(i['strokes']) for i in train_info}) >= 2, "vary stroke counts in training"


def _digest(puzzle):
    serial = {s: [{k: np.asarray(v).tolist() for k, v in p.items()}
                  for p in puzzle[s]] for s in ('train', 'test')}
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def self_test():
    # Analytical parity/endpoint fixture: two down-left strokes bracket seven
    # slots, four on scanlines that miss both boundary pixel centers.
    canvas = np.zeros((5, 5), dtype=np.int8)
    canvas[(0, 1, 2, 2, 3, 4), (2, 1, 0, 4, 3, 2)] = 3
    slope, strokes = _strokes(canvas, 0)
    info = dict(canvas=canvas, bg=0, slope=slope, strokes=strokes, t0=-2,
                a=1, b=2, carrier=np.array([[1, 4]]), stencil=np.array([[0, 2]]))
    expected = canvas.copy()
    expected[(1, 1, 2, 2, 2, 3, 3), (2, 3, 1, 2, 3, 1, 2)] = [1, 1, 1, 1, 2, 1, 2]
    assert np.array_equal(_reference(info), expected), "hand-computed weave fixture"
    # Put the analytical fixture in a complete legal header/canvas input.
    full = np.zeros((7, 7), dtype=np.int8)
    full[0, :5] = [1, 4, 8, 0, 2]
    full[1] = 8
    full[2:, :5] = canvas
    expected_full = np.zeros((5, 7), dtype=np.int8)
    expected_full[:, :5] = expected
    assert np.array_equal(solve(full), expected_full)
    # Six slots: replicate 14 -> 111444; tile 02 -> 020202; overlay -> 121242.
    full = np.zeros((12, 10), dtype=np.int8)
    full[0, :5] = [1, 4, 8, 0, 2]
    full[1] = 8
    full[(2, 3, 4, 9, 10, 11), (2, 1, 0, 9, 8, 7)] = 3
    answer = solve(full)
    assert answer[np.arange(2, 8), np.arange(2, 8)].tolist() == [1, 2, 1, 2, 4, 2]
    # Role parsing must commute with arbitrary literal-color permutation.
    remap = np.array([6, 8, 0, 9, 1, 4, 7, 2, 5, 3])
    assert np.array_equal(solve(remap[full]), remap[answer])
    digests, shapes, swatches, layouts, slopes, backgrounds, counts, periods = (
        set() for _ in range(8))
    extrema_switch = empty_stencils = uneven = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed)), "determinism"
        for split in ('train', 'test'):
            for p in puzzle[split]:
                info = _decode(p['input'])
                sections = _sections(info)
                shapes.add(p['input'].shape)
                slopes.add(info['slope'])
                backgrounds.add(info['bg'])
                counts.add(len(info['strokes']))
                periods.add((info['a'], info['b']))
                layouts.add(tuple((q, lo, hi) for q, lo, hi, _ in info['strokes']))
                swatches.add(_key(info['carrier']))
                extrema_switch += len({(s[1], s[2]) for s in sections}) > 1
                empty_stencils += np.all(info['stencil'] == info['bg'])
                uneven += any(len(s[3]) % info['b'] for s in sections)
    assert len(digests) == 42 and len(layouts) >= 150 and len(shapes) >= 50
    assert len(swatches) >= 120 and len(periods) >= 12 and len(backgrounds) >= 8
    assert slopes == {-1, 1} and {2, 3, 4, 5, 6, 7} <= counts
    assert extrema_switch >= 80 and empty_stencils >= 2 and uneven >= 100
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(42000+10*a+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return dict(seeds=42, distinct_puzzles=len(digests), shapes=len(shapes),
                carrier_swatches=len(swatches), rail_layouts=len(layouts),
                swatch_sizes=len(periods), stroke_counts=sorted(counts),
                moving_boundary_pairs=int(extrema_switch), empty_stencils=int(empty_stencils),
                uneven_width_pairs=int(uneven))
