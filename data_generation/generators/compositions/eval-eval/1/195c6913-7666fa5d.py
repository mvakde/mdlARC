"""A cyclic thread woven through a punctured diagonal interval envelope.

Only NumPy and the Python standard library are used. No parent generator, stored
pair, seed-dependent solver setting, or training output is consulted by solve.
"""
from __future__ import annotations

import numpy as np


ALTERNATIVES = (
    "uniform", "permanent", "consecutive", "pixel_centers", "deduplicate",
    "reset_sweep", "no_reverse", "count_strokes", "global_phase", "reverse_word",
    "same_color",
)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)):
        raise ValueError("example counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a.astype(np.int8, copy=True)


def _components_of_color(grid, color):
    unseen = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _strokes(grid, background):
    """Recognize a unique slope from marker geometry, independently of labels."""
    pixels = [tuple(map(int, p)) for p in np.argwhere(grid != background)]
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in pixels:
            lines.setdefault(c-slope*r, []).append(r)
        strokes, valid = [], True
        for q, rows in sorted(lines.items()):
            rows.sort()
            runs = []
            for r in rows:
                if not runs or r != runs[-1][-1]+1:
                    runs.append([])
                runs[-1].append(r)
            for run in runs:
                colors = {int(grid[r, q+slope*r]) for r in run}
                if len(run) < 2 or len(colors) != 1:
                    valid = False
                    break
                strokes.append(dict(q=q, lo=2*run[0]+slope*q,
                                    hi=2*run[-1]+slope*q, length=len(run),
                                    color=colors.pop()))
            if not valid:
                break
        qs = sorted(lines)
        if valid and len(strokes) >= 2 and all(b-a >= 3 for a, b in zip(qs, qs[1:])):
            candidates.append((slope, sorted(strokes, key=lambda s: (s['q'], s['lo']))))
    assert len(candidates) == 1, "ambiguous or invalid parallel strokes"
    return candidates[0]


def _levels(info):
    """Production envelope: active extrema on each continuous scan level."""
    base, bg, strokes = info['base'], info['background'], info['strokes']
    rr, cc = np.indices(base.shape)
    qq, tt = cc-info['slope']*rr, rr+info['slope']*cc
    rows = {}
    bounds = {}
    for t in range(min(s['lo'] for s in strokes), max(s['hi'] for s in strokes)+1):
        active = {s['q'] for s in strokes if s['lo'] <= t <= s['hi']}
        if len(active) < 2:
            continue
        low, high = min(active), max(active)
        positions = np.argwhere((tt == t) & (qq > low) & (qq < high) & (base == bg))
        cells = sorted((tuple(map(int, p)) for p in positions), key=lambda p: int(qq[p]))
        if cells:
            rows[t] = cells
            bounds[t] = (low, high)
    runs = []
    for t in sorted(rows):
        if not runs or t != runs[-1][-1]+1:
            runs.append([])
        runs[-1].append(t)
    info.update(rows=rows, runs=runs, bounds=bounds, qq=qq, tt=tt)
    return info


def _decode(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) > grid.size/2, "background must be a majority"
    swatches = []
    for color in colors:
        if int(color) == bg:
            continue
        for component in _components_of_color(grid, int(color)):
            if len(component) == 4:
                r, c = min(component)
                if component == {(r,c), (r+1,c), (r,c+1), (r+1,c+1)}:
                    swatches.append((r, c, int(color), component))
    swatches.sort(key=lambda x: x[1])
    assert 2 <= len(swatches) <= 6, "legend length"
    assert len({s[0] for s in swatches}) == 1, "horizontal legend"
    assert all(b[1]-a[1] >= 3 for a, b in zip(swatches, swatches[1:])), "swatch separation"
    word = tuple(s[2] for s in swatches)
    assert len(set(word)) >= 2 and bg not in word, "word roles"
    annotation = set().union(*(s[3] for s in swatches))
    base = grid.copy()
    for p in annotation:
        base[p] = bg
    beads = {tuple(map(int, p)) for p in np.argwhere(base == word[0])}
    assert beads, "missing entry beads"
    assert all(int(base[p]) not in set(word)-{word[0]}
               for p in map(tuple, np.argwhere(base != bg))), "unexplained word pixels"
    for p in beads:
        base[p] = bg
    # A bead is a singleton component in its word color, not another swatch.
    assert all(not any((r+dr,c+dc) in beads for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
               for r,c in beads), "beads must be singletons"
    stroke_cells = np.argwhere(base != bg)
    assert len(stroke_cells) and int(stroke_cells[:,0].min()) >= swatches[0][0]+4, "legend clearance"
    slope, strokes = _strokes(base, bg)
    for stroke in strokes:
        assert (stroke['lo']-stroke['q']) % 2 == (stroke['hi']-stroke['q']) % 2 == 0
    info = _levels(dict(grid=grid, base=base, background=bg, word=word,
                        beads=beads, swatches=swatches, annotation=annotation,
                        slope=slope, strokes=strokes))
    assert info['runs'], "no paintable run"
    starts, directions = [], []
    for run in info['runs']:
        first = info['rows'][run[0]]
        assert len(first) >= 2, "first sweep needs two ends"
        at_level = [p for p in beads if int(info['tt'][p]) == run[0]]
        assert len(at_level) == 1 and at_level[0] in (first[0], first[-1]), "entry bead location"
        starts.append(at_level[0])
        directions.append(1 if at_level[0] == first[0] else -1)
    assert set(starts) == beads, "one entry per run, no other beads"
    info.update(starts=starts, directions=directions)
    return info


def _paint(info, alternative=""):
    output = info['base'].copy()
    word = info['word']
    if alternative == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    elif alternative == 'reverse_word':
        word = word[::-1]
    elif alternative == 'uniform':
        word = word[:1]
    phase = 0
    for run, initial in zip(info['runs'], info['directions']):
        if alternative != 'global_phase':
            phase = 0
        for index, t in enumerate(run):
            if alternative == 'reset_sweep':
                phase = 0
            direction = initial * (-1 if index % 2 and alternative != 'no_reverse' else 1)
            cells = info['rows'][t]
            if alternative == 'count_strokes':
                low, high = info['bounds'][t]
                positions = np.argwhere((info['tt'] == t) & (info['qq'] > low) & (info['qq'] < high))
                cells = sorted((tuple(map(int, p)) for p in positions), key=lambda p: int(info['qq'][p]))
            if direction < 0:
                cells = cells[::-1]
            for p in cells:
                if info['base'][p] == info['background']:
                    output[p] = word[phase % len(word)]
                phase += 1
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from only the current input."""
    return _paint(_decode(input_grid))


def _interval_components(strokes):
    unseen, components = set(range(len(strokes))), []
    while unseen:
        i = min(unseen)
        unseen.remove(i)
        stack, component = [i], [i]
        while stack:
            a = strokes[stack.pop()]
            neighbors = [j for j in sorted(unseen) if max(a['lo'], strokes[j]['lo']) <= min(a['hi'], strokes[j]['hi'])]
            for j in neighbors:
                unseen.remove(j)
                stack.append(j)
                component.append(j)
        components.append(component)
    return components


def _pair_union_mask(info, alternative=""):
    """Separate geometric support check, using the union of pair intersections."""
    strokes, qq, tt = info['strokes'], info['qq'], info['tt']
    mask = np.zeros(info['base'].shape, dtype=bool)
    if alternative == 'permanent':
        qs = [s['q'] for s in strokes]
        mask = ((qq > min(qs)) & (qq < max(qs)) &
                (tt >= min(s['lo'] for s in strokes)) & (tt <= max(s['hi'] for s in strokes)))
    else:
        pairs = [(i,j) for i in range(len(strokes)) for j in range(i+1,len(strokes))]
        if alternative == 'consecutive':
            pairs = []
            for component in _interval_components(strokes):
                ordered = sorted(component, key=lambda i: (strokes[i]['q'], strokes[i]['lo']))
                pairs.extend(zip(ordered, ordered[1:]))
        for i,j in pairs:
            a,b = strokes[i],strokes[j]
            if alternative == 'same_color' and a['color'] != b['color']:
                continue
            between = ((qq > min(a['q'], b['q'])) & (qq < max(a['q'], b['q'])) &
                       (tt >= max(a['lo'], b['lo'])) & (tt <= min(a['hi'], b['hi'])))
            if alternative == 'pixel_centers':
                between &= ((tt-a['q']) % 2 == 0) & ((tt-b['q']) % 2 == 0)
            mask |= between
    return mask & (info['base'] == info['background'])


def _alternative(info, name, correct=None):
    if name not in ('permanent', 'consecutive', 'pixel_centers', 'same_color'):
        return _paint(info, name)
    # Geometric ablations receive the correct colors on the shared support.
    # This generous comparison tests support without an arbitrary wrong bead
    # policy when a wrong geometry adds levels with no bead.
    if correct is None:
        correct = _paint(info)
    mask = _pair_union_mask(info, name)
    output = info['base'].copy()
    shared = mask & (correct != info['base'])
    output[shared] = correct[shared]
    output[mask & ~shared] = info['word'][0]
    return output


def _evidence(info):
    correct = _paint(info)
    return {name for name in ALTERNATIVES if not np.array_equal(correct, _alternative(info, name, correct))}


def _word(rng, length, repeat=False):
    for _ in range(40):
        distinct = int(rng.integers(2, length if repeat else length+1))
        symbols = list(range(distinct)) + list(map(int, rng.integers(0, distinct, length-distinct)))
        rng.shuffle(symbols)
        word = tuple(symbols)
        unique = tuple(dict.fromkeys(word))
        reduced = tuple(unique[i % len(unique)] for i in range(length))
        if not repeat or (word != word[::-1] and word != reduced):
            return word
    return (0,)*(length-1)+(1,)


def _group(rng, qs, mode, length_range):
    lo_len, hi_len = length_range
    if mode == 'witness':
        short = int(rng.integers(1, len(qs)-1))
        result = []
        for i,q in enumerate(qs):
            if i in (0, len(qs)-1):
                lo = -5 + ((q+5) % 2)
                length = int(rng.integers(7, 9))
            elif i == short:
                lo = q % 2
                length = int(rng.integers(2, 4))
            else:
                lo = -3 + ((q+3) % 2)
                length = int(rng.integers(3, 7))
            result.append((q, lo, lo+2*(length-1)))
        return result
    if mode == 'chain':
        # Transverse rank is independent of interval order. Enough overlap
        # makes adjacent overlap windows consecutive, without a common core.
        order = list(map(int, rng.permutation(len(qs))))
        result = []
        for j,i in enumerate(order):
            q = qs[i]
            lo = 3*j + ((q-3*j) % 2)
            length = int(rng.integers(4, 6))
            result.append((q, lo, lo+2*(length-1)))
        return result
    equal_length = int(rng.integers(lo_len, hi_len+1))
    result = []
    for q in qs:
        length = equal_length if mode == 'equal' else int(rng.integers(lo_len, hi_len+1))
        if mode in ('equal', 'aligned'):
            lo = q % 2
        elif mode == 'free':
            lo = int(rng.integers(-6, 7))
            lo -= (lo-q) % 2
        else:
            before = int(rng.integers(0, length-1))
            lo = (q % 2)-2*before
        result.append((q, lo, lo+2*(length-1)))
    return result


def _specs(rng, kind, shift):
    if kind == 'simple':
        counts = [2]
    elif kind == 'witness':
        counts = [int(rng.integers(3, 5)), int(rng.integers(2, 4))]
    elif kind == 'test' and shift == 'chain':
        counts = [int(rng.integers(4, 8))]
    else:
        groups = int(rng.choice([1,2,3], p=[.67,.29,.04]))
        cap = 7 if groups == 1 else (4 if groups == 2 else 3)
        if kind == 'test' and shift == 'length':
            groups, cap = 1, 5
        counts = [int(rng.integers(2, cap+1)) for _ in range(groups)]
    pool_count = max(counts) + int(rng.integers(0, 2))
    pool = [0]
    for _ in range(pool_count-1):
        pool.append(pool[-1]+int(rng.integers(3, 7)))
    result = []
    for group_index, count in enumerate(counts):
        ids = sorted(map(int, rng.choice(pool_count, size=count, replace=False)))
        qs = [pool[i] for i in ids]
        mode = str(rng.choice(['common','equal','aligned','free','chain']))
        lengths = (2, 8)
        if kind == 'witness':
            mode = 'witness' if group_index == 0 else 'chain'
        elif kind == 'simple':
            mode, lengths = 'aligned', (4, 7)
        elif kind == 'test' and shift == 'chain':
            mode = 'chain'
        elif kind == 'test' and shift == 'length':
            mode, lengths = str(rng.choice(['aligned','common','equal'])), (8, 11)
        if len(counts) == 3:
            lengths = (2, 4)
            mode = 'aligned'
        current = _group(rng, qs, mode, lengths)
        offset = 0
        if result:
            offset = max(hi for _,_,hi in result) + int(rng.integers(3, 7))-min(lo for _,lo,_ in current)
            offset += offset % 2
        result.extend((q,lo+offset,hi+offset) for q,lo,hi in current)
    return result


def _render(rng, specs, word_symbols, slope, kind, monochrome=False):
    stroke_pixels = []
    for q,lo,hi in specs:
        stroke_pixels.append([((t-slope*q)//2, (q+slope*t)//2) for t in range(lo,hi+1,2)])
    all_cells = [p for stroke in stroke_pixels for p in stroke]
    if len(set(all_cells)) != len(all_cells):
        raise ValueError('overlapping strokes')
    rmin,cmin = np.min(all_cells, axis=0)
    rmax,cmax = np.max(all_cells, axis=0)
    bh,bw = int(rmax-rmin+1), int(cmax-cmin+1)
    legend_row = int(rng.integers(0, 3))
    gaps = list(map(int, rng.integers(1, 4, len(word_symbols)-1)))
    legend_width = 2*len(word_symbols)+sum(gaps)
    minimum_h, minimum_w = bh+legend_row+4, max(bw,legend_width)
    if minimum_h > 30 or minimum_w > 30:
        raise ValueError('layout too large')
    height = minimum_h+int(rng.integers(0,min(5,30-minimum_h)+1))
    width = minimum_w+int(rng.integers(0,min(6,30-minimum_w)+1))
    if rng.random() < .12:
        height = 30
    if rng.random() < .12:
        width = 30
    top = legend_row+4+int(rng.integers(0,height-minimum_h+1))
    left = int(rng.integers(0,width-bw+1))
    legend_left = int(rng.integers(0,width-legend_width+1))
    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    distinct = max(word_symbols)+1
    word = tuple(palette[1+x] for x in word_symbols)
    maximum_colors = min(3, 9-distinct, len(specs))
    color_count = 1 if monochrome else int(rng.integers(1, maximum_colors+1))
    if kind == 'witness':
        color_count = int(rng.integers(2, maximum_colors+1))
    marker_colors = palette[1+distinct:1+distinct+color_count]
    grid = np.full((height,width),bg,dtype=np.int8)
    for j,stroke in enumerate(stroke_pixels):
        color = marker_colors[j % color_count]
        for r,c in stroke:
            grid[int(r-rmin)+top,int(c-cmin)+left] = color
    if np.count_nonzero(grid == bg)-4*len(word) <= grid.size/2:
        raise ValueError('majority field too small')
    actual_slope, actual_strokes = _strokes(grid, bg)
    if actual_slope != slope:
        raise ValueError('wrong recognizable slope')
    info = _levels(dict(base=grid.copy(), background=bg, slope=slope,
                        strokes=actual_strokes, word=word))
    if not info['runs'] or any(len(info['rows'][run[0]]) < 2 for run in info['runs']):
        raise ValueError('unclear entry')
    if kind == 'simple' and len(info['runs']) != 1:
        raise ValueError('simple example needs one run')
    if kind == 'witness' and len(info['runs']) < 2:
        raise ValueError('witness needs restart')
    column = legend_left
    for j,color in enumerate(word):
        grid[legend_row:legend_row+2,column:column+2] = color
        if j < len(gaps):
            column += 2+gaps[j]
    for run in info['runs']:
        cells = info['rows'][run[0]]
        bead = cells[0] if rng.random() < .5 else cells[-1]
        grid[bead] = word[0]
    decoded = _decode(grid)
    return dict(input=grid, output=_paint(decoded)), decoded


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _make_pair(rng, kind, shift, seen):
    for _ in range(600):
        length = int(rng.integers(3,5)) if kind == 'witness' else int(rng.integers(2,5))
        if kind == 'test' and shift == 'word':
            length = int(rng.integers(5,7))
        symbols = _word(rng, length, kind == 'witness')
        slope = int(rng.choice([-1,1]))
        specs = _specs(rng, kind, shift)
        try:
            pair, info = _render(rng, specs, symbols, slope, kind, kind == 'simple')
        except (ValueError, AssertionError):
            continue
        if kind == 'test' and shift == 'chain':
            if len(info['runs']) != 1 or max(s['lo'] for s in info['strokes']) <= min(s['hi'] for s in info['strokes']):
                continue
        if kind == 'witness' and set(ALTERNATIVES) != _evidence(info):
            continue
        if kind == 'witness' and not any(len({info['bounds'][t] for t in run}) > 1 for run in info['runs']):
            continue
        if _key(pair['input']) not in seen:
            return pair
    raise RuntimeError('bounded loom construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['word','length','chain']))
    witness, simple = map(int, rng.choice(num_train, size=2, replace=False))
    puzzle, seen = {'train':[], 'test':[]}, set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            kind = 'test' if split == 'test' else ('witness' if index == witness else 'simple' if index == simple else 'free')
            pair = _make_pair(rng,kind,shift,seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'puzzle schema'
    assert all(isinstance(puzzle[k],list) for k in puzzle), 'split lists'
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, evidence = set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}, 'pair schema'
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape, 'shape preservation'
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            support = _pair_union_mask(info)
            painted = np.zeros(inp.shape,dtype=bool)
            for cells in info['rows'].values():
                for p in cells:
                    painted[p] = True
            assert np.array_equal(support,painted), 'pair-union/envelope disagreement'
            assert np.all(np.isin(out[support],info['word'])), 'paint palette'
            assert np.array_equal(out[~support],info['base'][~support]), 'stroke/background/erasure preservation'
            assert all(out[p] == info['word'][0] for p in info['beads']), 'bead restart'
            assert np.array_equal(out,_paint(info)), 'incorrect weave phases'
            if split == 'train':
                evidence |= _evidence(info)
    assert set(ALTERNATIVES) <= evidence, 'missing training witnesses: '+str(set(ALTERNATIVES)-evidence)


def _fingerprint(puzzle):
    return tuple((_key(p['input']),_key(p['output'])) for split in ('train','test') for p in puzzle[split])


def _analytical_fixture():
    inp = np.zeros((11,10),dtype=np.int8)
    for col,color in ((0,1),(3,1),(6,2)):
        inp[0:2,col:col+2] = color
    for q,lo,hi in ((0,0,6),(6,0,6),(3,1,3)):
        for t in range(lo,hi+1,2):
            inp[(t-q)//2+7,(t+q)//2+1] = 3
    inp[6,2] = 1
    expected = inp.copy()
    expected[0:2,:] = 0
    # Hand enumerated weave; the short q=3 marker consumes no color.
    visits = ((0,2,1),(0,4,1),(1,5,2),(1,1,1),(2,2,1),(2,4,2),
              (3,5,1),(3,1,1),(4,2,2),(4,4,1),(5,5,1),(5,3,2),
              (5,1,1),(6,2,1),(6,4,2))
    for t,q,color in visits:
        expected[(t-q)//2+7,(t+q)//2+1] = color
    assert np.array_equal(solve(inp),expected), 'analytical weave regression'
    assert np.count_nonzero(_pair_union_mask(_decode(inp))) == 15
    for name in ('consecutive','pixel_centers','deduplicate','reset_sweep','no_reverse','count_strokes'):
        assert name in _evidence(_decode(inp)), name


def self_test():
    """Bounded 42-seed checks; returns compact measurable diversity evidence."""
    _analytical_fixture()
    shapes, layouts, words, slopes, run_counts, entry_sides = set(),set(),set(),set(),set(),set()
    stroke_counts, lengths, colors, fingerprints = set(),set(),set(),set()
    evidence_counts = {name:0 for name in ALTERNATIVES}
    mixed_color_pairs = boundary_changes = edge_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed < 3:
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'determinism'
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                strokes = info['strokes']
                q0,t0 = min(s['q'] for s in strokes),min(s['lo'] for s in strokes)
                layouts.add(tuple((s['q']-q0,s['lo']-t0,s['hi']-t0) for s in strokes))
                labels = {}
                words.add(tuple(labels.setdefault(c,len(labels)) for c in info['word']))
                slopes.add(info['slope'])
                run_counts.add(len(info['runs']))
                entry_sides.update(info['directions'])
                stroke_counts.add(len(strokes))
                lengths.update(s['length'] for s in strokes)
                colors.add((info['background'],info['word'],tuple(s['color'] for s in strokes)))
                mixed_color_pairs += len({s['color'] for s in strokes}) > 1
                boundary_changes += any(len({info['bounds'][t] for t in run}) > 1 for run in info['runs'])
                occupied = info['base'] != info['background']
                edge_pairs += bool(np.any(occupied[0,:]) or np.any(occupied[-1,:]) or
                                   np.any(occupied[:,0]) or np.any(occupied[:,-1]))
                if split == 'test':
                    for name in _evidence(info):
                        evidence_counts[name] += 1
    assert len(fingerprints) == 42 and len(layouts) >= 150, 'structural variation'
    assert len(shapes) >= 50 and len(words) >= 12, 'scale/program variation'
    assert slopes == entry_sides == {-1,1}, 'orientation/entry variation'
    assert {1,2,3} <= run_counts, 'restart geometry variation'
    assert {2,3,4,5,6} <= stroke_counts and {2,3,7,8,11} <= lengths
    assert len(colors) >= 150 and mixed_color_pairs >= 30 and boundary_changes >= 20 and edge_pairs >= 10
    assert all(evidence_counts[n] > 0 for n in ALTERNATIVES), 'test ablation coverage'
    for nt,ne in ((2,1),(10,1),(10,4)):
        p = generate_puzzle(886340397,nt,ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0,2,1)
    info = _decode(p['test'][0]['input'])
    cell = info['rows'][info['runs'][0][0]][0]
    p['test'][0]['output'][cell] = info['background']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt weave accepted')
    # Two correct simple brackets do not teach bypass or restart. Recolor the
    # simple pair bijectively so this negative fixture has distinct inputs.
    p = generate_puzzle(1,2,1)
    simple = next(pair for pair in p['train'] if len(_decode(pair['input'])['strokes']) == 2)
    rich = next(pair for pair in p['train'] if len(_decode(pair['input'])['strokes']) > 2)
    recolored = {role: (simple[role]+1) % 10 for role in ('input','output')}
    candidate = dict(train=[simple,recolored],test=[rich])
    try:
        validate_generated_puzzle(candidate)
    except AssertionError as error:
        assert 'missing training witnesses' in str(error)
    else:
        raise AssertionError('insufficient teaching accepted')
    return dict(seeds=42,pairs=210,distinct_puzzles=len(fingerprints),shapes=len(shapes),
                interval_layouts=len(layouts),word_patterns=len(words),slopes=sorted(slopes),
                run_counts=sorted(run_counts),stroke_counts=sorted(stroke_counts),lengths=sorted(lengths),
                mixed_color_pairs=mixed_color_pairs,boundary_change_pairs=boundary_changes,
                stroke_edge_pairs=edge_pairs,test_alternative_witnesses=evidence_counts,
                independent_semantic_certification=False)
