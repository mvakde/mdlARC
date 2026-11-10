"""Phase-locked dashed trails on terrain-guided staircases.

The header swatch names the equal reference marks. The remaining swatches are
an ordered cyclic word. Cadence is measured in visited terrain cells, and an
on-cadence terminal locks the word to its own unique legend entry.
"""

import numpy as np


HOST_SEEDS = (0, 92719344, 15321522, 374533545, 108623304, 544262404,
              30487732, 965963691, 211834931, 899776642, 582305470,
              319251233, 807349744, 370952911, 538928919, 668349353,
              395642664, 422516565, 393747552, 244655834, 29832403,
              476795574, 906173126, 317133848, 617713900, 799162365,
              868529041, 504947846, 40093625, 431602250, 493915728,
              335737108, 63514931, 86454148, 386314017, 911502794,
              266213590, 158913205, 41925987, 916676523, 347997270,
              673026672)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _orient(grid, code):
    return np.rot90(grid[:, ::-1] if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    grid = np.rot90(grid, -(code % 4))
    return (grid[:, ::-1] if code >= 4 else grid).copy()


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or not all(1 <= d <= 30 for d in grid.shape)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError('grid must be a 1..30 integer rectangle with colors 0..9')
    return grid.astype(np.int8, copy=True)


def _components(mask):
    seen, components = set(), []
    for pp in np.argwhere(mask):
        p = tuple(map(int, pp))
        if p in seen:
            continue
        seen.add(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if _inside(mask.shape, q) and mask[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
        components.append(tuple(cells))
    return components


def _swatches(grid, terrain_colors):
    result = []
    for color in np.unique(grid):
        if int(color) in terrain_colors:
            continue
        for cells in _components(grid == color):
            if len(cells) != 4:
                continue
            r, c = min(p[0] for p in cells), min(p[1] for p in cells)
            if set(cells) == {(r, c), (r+1, c), (r, c+1), (r+1, c+1)}:
                result.append(dict(row=r, col=c, color=int(color), cells=cells))
    return result


def _trace(mask, start):
    """Cell-step tracer. Bend cells are counted once; canvas exit stops."""
    p, direction = start, 0
    path, legs, collisions = [], [0], []
    vectors = ((0, 1), (-1, 0))
    while True:
        if not _inside(mask.shape, p) or not mask[p]:
            raise ValueError('invalid route start')
        path.append(p)
        legs[-1] += 1
        dr, dc = vectors[direction]
        ahead = (p[0]+dr, p[1]+dc)
        if not _inside(mask.shape, ahead):
            end = 'edge'
            break
        if mask[ahead]:
            p = ahead
            continue
        collisions.append(ahead)
        direction = 1-direction
        dr, dc = vectors[direction]
        turn = (p[0]+dr, p[1]+dc)
        if not _inside(mask.shape, turn) or not mask[turn]:
            end = 'blocked'
            break
        legs.append(0)
        p = turn
    return dict(start=start, path=tuple(path), legs=tuple(legs),
                collisions=tuple(collisions), end=end)


def _runs(mask, start):
    """Second tracing formulation for checking collision and edge behavior."""
    current, direction = start, 0
    path, legs, collisions = [], [], []
    while True:
        dr, dc = ((0, 1), (-1, 0))[direction]
        leg, p = [], current
        while _inside(mask.shape, p) and mask[p]:
            leg.append(p)
            p = (p[0]+dr, p[1]+dc)
        if not leg:
            raise ValueError('empty run')
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, p):
            end = 'edge'
            break
        collisions.append(p)
        direction = 1-direction
        dr, dc = ((0, 1), (-1, 0))[direction]
        current = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, current) or not mask[current]:
            end = 'blocked'
            break
    return dict(start=start, path=tuple(path), legs=tuple(legs),
                collisions=tuple(collisions), end=end)


def _route_explanations(mask, reference_cells, extras, word, legend_row):
    """A first reference is on the first leg; terrain recovers its entry."""
    candidates, starts = [], set()
    for r, c in reference_cells:
        while c > 0 and mask[r, c-1]:
            c -= 1
        start = (r, c)
        if start in starts or r < legend_row+4:
            continue
        starts.add(start)
        record = _trace(mask, start)
        indices = {p: i for i, p in enumerate(record['path'])}
        refs = reference_cells & indices.keys()
        if len(refs) != 2:
            continue
        a, b = sorted(indices[p] for p in refs)
        if a >= record['legs'][0]:
            continue
        period = b-a
        local_extras = {indices[p]: color for p, color in extras.items() if p in indices}
        aligned = [i for i in local_extras if (i-a) % period == 0]
        if len(aligned) > 1:
            continue
        terminal = aligned[0] if aligned else None
        if terminal is not None:
            if a < terminal < b or word.count(local_extras[terminal]) != 1:
                continue
        record.update(a=a, b=b, period=period, terminal=terminal,
                      terminal_color=None if terminal is None else local_extras[terminal],
                      extras=local_extras, refs=refs)
        candidates.append(record)

    # Small exact cover: associations must explain every mark and every extra.
    solutions = []
    def visit(chosen, remaining, occupied):
        if len(solutions) >= 2:
            return
        if not remaining:
            covered = {p for record in chosen for p in record['path']}
            if extras.keys() <= covered:
                solutions.append(sorted(chosen, key=lambda r: r['start']))
            return
        first = min(remaining)
        for record in candidates:
            footprint = set(record['path']) | set(record['collisions'])
            if (first in record['refs'] and record['refs'] <= remaining
                    and not footprint & occupied):
                visit(chosen+[record], remaining-record['refs'], occupied | footprint)
    visit([], reference_cells, set())
    if len(solutions) != 1:
        raise ValueError('reference-to-route association is not unique')
    return solutions[0]


def _decode(input_grid):
    grid = _checked_grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) < 5:
        raise ValueError('missing terrain or program colors')
    terrain_colors = set(map(int, colors[np.argsort(counts)[-2:]]))
    if max(int(n) for c,n in zip(colors,counts) if int(c) not in terrain_colors) >= min(
            int(n) for c,n in zip(colors,counts) if int(c) in terrain_colors):
        raise ValueError('terrain frequency cutoff must be unambiguous')
    observed_swatches = _swatches(grid, terrain_colors)
    if not 3 <= len(observed_swatches) <= 7:
        raise ValueError('expected a header and two to six program swatches')
    annotations = {p for s in observed_swatches for p in s['cells']}
    rings = {int(grid[q]) for p in annotations for q in
             ((p[0]-1,p[1]), (p[0]+1,p[1]), (p[0],p[1]-1), (p[0],p[1]+1))
             if _inside(grid.shape, q) and q not in annotations}
    if len(rings) != 1 or not rings <= terrain_colors:
        raise ValueError('swatches must have inferable surrounding terrain')
    wall = next(iter(rings))
    route_color = next(iter(terrain_colors-{wall}))
    candidates = []
    for code in range(8):
        canonical = _unorient(grid, code)
        swatches = sorted(_swatches(canonical, terrain_colors), key=lambda s: s['col'])
        if (len({s['row'] for s in swatches}) != 1
                or any(b['col']-a['col'] < 3 for a,b in zip(swatches, swatches[1:]))):
            continue
        header = swatches[0]
        marker = header['color']
        word = tuple(s['color'] for s in swatches[1:])
        if marker in word or not 2 <= len(set(word)) <= 5:
            continue
        acells = {p for s in swatches for p in s['cells']}
        singleton_cells = {tuple(map(int, p)) for p in np.argwhere(
            ~np.isin(canonical, list(terrain_colors)))}-acells
        refs = {p for p in singleton_cells if int(canonical[p]) == marker}
        if len(refs) not in (2, 4, 6):
            continue
        extras = {p: int(canonical[p]) for p in singleton_cells-refs}
        if any(color not in word for color in extras.values()):
            continue
        base = canonical.copy()
        for p in acells:
            base[p] = wall
        for p in singleton_cells:
            base[p] = route_color
        mask = base == route_color
        try:
            records = _route_explanations(mask, refs, extras, word, header['row'])
        except ValueError:
            continue
        footprints = {p for r in records for p in r['path']+r['collisions']}
        if footprints & acells:
            continue
        candidates.append(dict(grid=canonical, base=base, mask=mask, word=word,
                               marker=marker, wall=wall, route_color=route_color,
                               swatches=swatches, records=records, code=code))
    if len(candidates) != 1:
        raise ValueError('program order or route orientation is ambiguous')
    return candidates[0]


def _paint(info, alternative=None):
    result = info['grid'].copy()
    for swatch in info['swatches']:
        for p in swatch['cells']:
            result[p] = info['wall']
    word = info['word']
    if alternative == 'deduplicate':
        word = tuple(dict.fromkeys(word))
    offset = 0
    for record in info['records']:
        path, a, b = record['path'], record['a'], record['b']
        period, terminal = record['period'], record['terminal']
        if alternative == 'ignore_terminal':
            terminal = None
        origin = a % period
        low, high = (origin, len(path)-1) if terminal is None else (min(a, terminal), max(b, terminal))
        if alternative == 'no_backward' and terminal is None:
            low = a
        if alternative == 'unbounded' and terminal is not None:
            low, high = origin, len(path)-1
        anchor = 0 if terminal is None else word.index(record['terminal_color'])
        leg_ends = np.cumsum(record['legs'])
        tick_number = 0
        for i, p in enumerate(path):
            if not low <= i <= high:
                continue
            leg = int(np.searchsorted(leg_ends, i, side='right'))
            leg_start = 0 if leg == 0 else int(leg_ends[leg-1])
            if alternative == 'straight_only' and leg:
                continue
            on_tick = (i-a) % period == 0
            if alternative == 'spacing_reset' and leg:
                on_tick = (i-leg_start) % period == 0
            if alternative != 'dense' and not on_tick:
                continue
            phase = (i-origin)//period if terminal is None else anchor+(i-terminal)//period
            if alternative == 'spacing_reset' and terminal is None:
                phase = tick_number
            if alternative == 'forward_terminal' and terminal is not None:
                phase = (i-low)//period
            if alternative == 'cell_clock':
                phase = i-origin if terminal is None else anchor+i-terminal
            if alternative == 'bend_reset':
                first_tick = leg_start+(a-leg_start) % period
                phase = (i-first_tick)//period
            if alternative == 'shared_clock':
                phase += offset
            if alternative == 'monochrome':
                result[p] = word[0]
            else:
                result[p] = word[phase % len(word)]
            tick_number += 1
        offset += tick_number
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, associations, paths, cadence and phase from input."""
    info = _decode(input_grid)
    return _orient(_paint(info), info['code'])


_ALTERNATIVES = ('dense', 'monochrome', 'deduplicate', 'cell_clock',
                 'bend_reset', 'spacing_reset', 'straight_only', 'shared_clock', 'ignore_terminal', 'unbounded',
                 'forward_terminal', 'no_backward')


def _evidence(info):
    correct = _paint(info)
    evidence = {name for name in _ALTERNATIVES
                if not np.array_equal(correct, _paint(info, name))}
    for r in info['records']:
        if r['b'] >= r['legs'][0]:
            evidence.add('pair_across_bend')
        if r['a'] % r['period']:
            evidence.add('nonzero_phase')
        if any((i-r['a']) % r['period'] for i in r['extras']):
            evidence.add('off_phase')
        if r['terminal'] is None and any(i > r['b'] and info['word'].count(c) == 1
                                        for i,c in r['extras'].items()):
            evidence.add('false_terminal')
        ticks = list(range(r['a'] % r['period'], len(r['path']), r['period']))
        if r['terminal'] is None and len(ticks) >= 2*len(info['word']):
            evidence.add('cycling')
        if r['terminal'] is None:
            evidence.add('boundary')
        else:
            evidence.add('terminal')
    return evidence


_REQUIRED = set(_ALTERNATIVES) | {'pair_across_bend', 'nonzero_phase',
                                 'off_phase', 'false_terminal', 'cycling', 'boundary', 'terminal'}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _word(rng, witness, maximum=6):
    if witness:
        patterns = ((0,0,1), (0,1,0), (0,1,1), (0,0,1,2),
                    (0,1,1,2), (0,1,2,0))
        if witness == 'terminal':
            patterns = tuple(p for p in patterns
                             if any(p.count(c) == 1 and p.index(c) > 0 for c in p))
        return _pick(rng, patterns)
    n = int(rng.integers(2, maximum+1))
    d = int(rng.integers(2, min(5,n)+1))
    word = list(range(d))+list(map(int, rng.integers(d, size=n-d)))
    rng.shuffle(word)
    return tuple(word)


def _terrain(rng, h, w, clear):
    knots = np.unique(np.r_[0, rng.choice(np.arange(1,h-1),
                                         size=int(rng.integers(3, min(9,h-2)+1)),
                                         replace=False), h-1])
    left = np.rint(np.interp(np.arange(h), knots, rng.uniform(-w*.2, w*.4, len(knots)))).astype(int)
    right = np.rint(np.interp(np.arange(h), knots, rng.uniform(w*.45, w*1.2, len(knots)))).astype(int)
    mask = np.zeros((h,w), bool)
    for r in range(h):
        lo, hi = max(0,int(left[r])), min(w-1,int(right[r]))
        mask[r, lo:hi+1] = True
    for _ in range(int(rng.integers(0,7))):
        r, c = int(rng.integers(h-1)), int(rng.integers(w-1))
        dh, dw = int(rng.integers(1,5)), int(rng.integers(1,8))
        mask[r:r+dh,c:c+dw] = bool(rng.integers(2))
    mask[clear] = False
    components = _components(mask)
    if components:
        largest = max(components, key=len)
        mask[:] = False
        for p in largest:
            mask[p] = True
    return mask


def _backstop(rng, h, w, clear):
    """Constructive terrain with two unequal, disjoint, bent routes."""
    maximum = min(13,w-3)
    width = int(rng.integers(min(8,maximum),maximum+1))
    top, bottom = 5, h-7
    mask = np.zeros((h,w), bool)
    mask[top:bottom+3,:width] = True
    mask[top:top+2,width:] = True
    mask[h-3:,:6] = True
    mask[h-3,6:min(w-1,14)] = True
    mask[bottom+2:h-2,0] = True
    mask[clear] = False
    return mask


def _parameters(rng, record, word, mode, witness, side):
    n, first_leg = len(record['path']), record['legs'][0]
    options = []
    for period in range(1, (n-1)//2+1):
        for a in range(min(first_leg, n-period)):
            b = a+period
            phase = list(range(a % period, n, period))
            if len(phase) < 3:
                continue
            if mode == 'boundary':
                if witness and not (period > 1 and a >= period and a % period
                                    and b > first_leg and len(phase) >= 2*len(word)):
                    continue
                options.append((a,b,None))
            else:
                for t in phase:
                    if not (t < a or t > b):
                        continue
                    if side == 'before' and t >= a or side == 'after' and t <= b:
                        continue
                    lo, hi = min(a,t), max(b,t)
                    if witness and (sum(lo <= i <= hi for i in phase) < len(word)+2
                                    or not any(i < lo or i > hi for i in phase)):
                        continue
                    if not witness and sum(lo <= i <= hi for i in phase) < 3:
                        continue
                    options.append((a,b,t))
    if not options:
        return None
    a,b,t = _pick(rng,options)
    period = b-a
    unique_colors = [c for c in word if word.count(c) == 1]
    if witness == 'terminal' and side == 'before':
        unique_colors = [c for c in unique_colors if word.index(c) > 0]
    if t is not None and not unique_colors:
        return None
    extras = {} if t is None else {t: int(_pick(rng, unique_colors))}
    off_phase = [i for i in range(n) if (i-a) % period and i not in extras]
    noise_count = min(len(off_phase), int(rng.integers(0,3)))
    if witness == 'boundary':
        noise_count = max(1, noise_count)
    if noise_count > len(off_phase):
        return None
    if witness == 'boundary':
        decoys = [i for i in off_phase if i > b]
        if not decoys or not unique_colors:
            return None
        i = int(_pick(rng,decoys))
        extras[i] = int(_pick(rng,unique_colors))
        off_phase.remove(i)
        noise_count -= 1
    for i in rng.choice(off_phase, noise_count, replace=False):
        extras[int(i)] = int(_pick(rng, word))
    result = dict(record)
    result.update(a=a, b=b, period=period, terminal=t,
                  terminal_color=None if t is None else extras[t], extras=extras)
    return result


def _ground_truth(base, records, word):
    """Construct labels from sampled coordinates, rather than calling solve."""
    output = base.copy()
    for r in records:
        for i, color in r['extras'].items():
            output[r['path'][i]] = color
        step, a, b, t = r['period'], r['a'], r['b'], r['terminal']
        if t is None:
            ticks = range(a % step, len(r['path']), step)
            for phase, i in enumerate(ticks):
                output[r['path'][i]] = word[phase % len(word)]
        else:
            anchor = word.index(r['terminal_color'])
            for i in range(min(a,t), max(b,t)+1, step):
                output[r['path'][i]] = word[(anchor+(i-t)//step) % len(word)]
    return output


def _make_pair(rng, h, w, code, witness, side, palette, force_terminal=False):
    maximum = min(6,(w-2)//3)
    pattern = _word(rng, witness, maximum)
    if force_terminal:
        for _ in range(16):
            if any(pattern.count(c) == 1 for c in pattern):
                break
            pattern = _word(rng,None,maximum)
        else:
            pattern = (0,1)
    wall, route_color, marker = palette[:3]
    word = tuple(palette[3+x] for x in pattern)
    unique_colors = [c for c in word if word.count(c) == 1]
    count = 2 if witness == 'boundary' else 1 if witness == 'terminal' else int(_pick(rng,(1,1,2,2,3)))
    if h < 18:
        count = min(count,2)
    for attempt in range(180):
        fallback = attempt >= 120
        if fallback:
            count = min(count,2)
        row = 0 if fallback else int(rng.integers(0,3))
        left = int(rng.integers(0, min(4,w-(3*(len(word)+1)-1))+1))
        columns = [left]
        for j in range(len(word)):
            room = w-2-columns[-1]-3*(len(word)-j-1)
            columns.append(columns[-1]+int(rng.integers(3,min(5,room)+1)))
        if columns[-1]+2 > w:
            continue
        clear = np.zeros((h,w),bool)
        clear[max(0,row-1):row+3,max(0,left-1):min(w,columns[-1]+3)] = True
        mask = _backstop(rng,h,w,clear) if fallback else _terrain(rng,h,w,clear)
        if min(np.count_nonzero(mask),np.count_nonzero(~mask)) <= 4*len(word)+12:
            continue
        choices = []
        for rr in range(row+4,h):
            for cc in range(w-1):
                if mask[rr,cc] and mask[rr,cc+1] and (cc == 0 or not mask[rr,cc-1]):
                    record = _runs(mask,(rr,cc))
                    if len(record['path']) >= 7:
                        choices.append(record)
        rng.shuffle(choices)
        if witness:
            choices.sort(key=lambda r: len(r['path']), reverse=True)
        selected, occupied = [], set()
        for record in choices:
            footprint = set(record['path']) | set(record['collisions'])
            if footprint & occupied:
                continue
            mode = 'boundary' if witness == 'boundary' else 'terminal' if witness == 'terminal' or force_terminal else str(_pick(rng, ('boundary','terminal','terminal')))
            if not unique_colors:
                mode = 'boundary'
            params = _parameters(rng,record,word,mode,witness if not selected else None,side)
            if params is None:
                continue
            selected.append(params)
            occupied |= footprint
            if len(selected) == count:
                break
        if len(selected) != count:
            continue
        annotation_cells = {(r,c) for r in range(row,row+2) for col in columns for c in range(col,col+2)}
        if occupied & annotation_cells:
            continue
        base = np.where(mask,route_color,wall).astype(np.int8)
        inp = base.copy()
        for col,color in zip(columns,(marker,)+word):
            inp[row:row+2,col:col+2] = color
        for record in selected:
            inp[record['path'][record['a']]] = marker
            inp[record['path'][record['b']]] = marker
            for i,color in record['extras'].items():
                inp[record['path'][i]] = color
        output = _ground_truth(base,selected,word)
        inp, output = _orient(inp,code), _orient(output,code)
        try:
            info = _decode(inp)
        except ValueError:
            continue
        if not np.array_equal(_orient(_paint(info),info['code']),output):
            raise AssertionError('sampled geometry and input reconstruction disagree')
        evidence = _evidence(info)
        required = ({'dense','monochrome','deduplicate','cell_clock','bend_reset',
                     'spacing_reset','straight_only','shared_clock','no_backward',
                     'pair_across_bend','nonzero_phase','off_phase','false_terminal',
                     'cycling','boundary'} if witness == 'boundary' else
                    {'ignore_terminal','unbounded','forward_terminal','terminal'} if witness == 'terminal' else set())
        if not required <= evidence:
            continue
        return {'input':inp, 'output':output}
    raise RuntimeError('bounded pair construction exhausted')


def _count(value, name, low, high):
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer))
            or not low <= value <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train,'num_train',2,10)
    num_test = _count(num_test,'num_test',1,4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    schedule = str(_pick(rng,('rotation','terminal_side','growth')))
    orientation = int(rng.integers(8))
    test_orientation = orientation//4*4+(orientation+int(rng.integers(1,4))) % 4
    constant_palette = bool(rng.random() < .2)
    fixed_palette = tuple(map(int,rng.permutation(10)))
    witnesses = [None]*num_train
    slots = rng.choice(num_train,2,replace=False)
    witnesses[int(slots[0])], witnesses[int(slots[1])] = 'boundary','terminal'
    puzzle, seen = {'train':[], 'test':[]}, set()
    for split,count in (('train',num_train),('test',num_test)):
        for index in range(count):
            witness = witnesses[index] if split == 'train' else None
            force_terminal = schedule == 'terminal_side' and split == 'test'
            side = ('after' if split == 'train' else 'before') if schedule == 'terminal_side' else str(_pick(rng,('before','after','either')))
            code = test_orientation if schedule == 'rotation' and split == 'test' else orientation
            if schedule != 'rotation' and rng.random() < .45:
                code = int(rng.integers(8))
            for _ in range(12):
                lo,hi = (26,30) if schedule == 'growth' and split == 'test' else (20,25) if schedule == 'growth' else (20,30) if witness else (12,30)
                h = int(rng.integers(lo,hi+1))
                w = int(rng.integers(max(22 if witness else 12,lo),hi+1))
                if rng.random() < .2:
                    h = w
                palette = fixed_palette if constant_palette else tuple(map(int,rng.permutation(10)))
                try:
                    pair = _make_pair(rng,h,w,code,witness,side,palette,force_terminal)
                except RuntimeError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('could not produce distinct inputs')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert isinstance(puzzle['train'],list) and isinstance(puzzle['test'],list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            canonical_output = _unorient(out,info['code'])
            assert np.array_equal(canonical_output,_paint(info)), 'incorrect trail output'
            assert not np.array_equal(inp,out)
            assert len(_components(info['mask'])) == 1, 'route terrain must be connected'
            changed_allowed = {p for s in info['swatches'] for p in s['cells']}
            for r in info['records']:
                traced = _trace(info['mask'],r['start'])
                assert traced == _runs(info['mask'],r['start'])
                for name in ('path','legs','collisions','end'):
                    assert traced[name] == r[name]
                phase = list(range(r['a'] % r['period'],len(r['path']),r['period']))
                assert len(phase) >= 3
                for i in phase:
                    if r['terminal'] is None or min(r['a'],r['terminal']) <= i <= max(r['b'],r['terminal']):
                        changed_allowed.add(r['path'][i])
                for i,color in r['extras'].items():
                    assert canonical_output[r['path'][i]] == color if (i-r['a']) % r['period'] else True
                assert all(canonical_output[p] == info['wall'] for p in r['collisions'])
            actual_changes = {tuple(map(int,p)) for p in np.argwhere(canonical_output != info['grid'])}
            assert actual_changes <= changed_allowed, 'unrelated cells changed'
            assert not np.any(out == info['marker']), 'reference/header color must disappear'
            if split == 'train':
                evidence |= _evidence(info)
    assert _REQUIRED <= evidence, 'training lacks witnesses: '+str(_REQUIRED-evidence)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])),_key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train','test'))


def self_test():
    """Bounded semantic regressions, mutation checks and 42-seed variation."""
    # Literal edge behavior, including available terrain around a canvas exit.
    mask = np.zeros((5,9),bool)
    mask[3,:] = True
    mask[2:4,8] = True
    assert _trace(mask,(3,0))['path'] == tuple((3,c) for c in range(9))
    mask = np.zeros((8,10),bool)
    mask[6,:5] = True
    mask[:7,4] = True
    mask[0,4:9] = True
    expected = tuple([(6,c) for c in range(5)]+[(r,4) for r in range(5,-1,-1)])
    assert _trace(mask,(6,0))['path'] == expected
    assert _trace(mask,(6,0)) == _runs(mask,(6,0))
    # Complete hand-constructed input with broad terrain, a three-leg route,
    # AAB, and an off-phase B. Expected cells are enumerated literally.
    terrain = np.zeros((12,14),np.int8)
    terrain[3:,:10] = 2
    terrain[3,10:] = 2
    fixture = terrain.copy()
    for c,color in ((0,8),(3,6),(6,6),(9,7)):
        fixture[:2,c:c+2] = color
    fixture[10,[4,7]] = 8
    fixture[7,9] = 7
    for terminal_position, cells, values in (
            ((3,9),((10,4),(10,7),(9,9),(6,9),(3,9)),(6,7,6,6,7)),
            ((10,1),((10,1),(10,4),(10,7)),(7,6,6)),
            (None,((10,1),(10,4),(10,7),(9,9),(6,9),(3,9),(3,12)),(6,6,7,6,6,7,6))):
        inp, expected_grid = fixture.copy(), terrain.copy()
        if terminal_position is not None:
            inp[terminal_position] = 7
        expected_grid[7,9] = 7
        for p,color in zip(cells,values):
            expected_grid[p] = color
        for code in range(8):
            for invert in (False,True):
                rotated_inp, rotated_out = _orient(inp,code), _orient(expected_grid,code)
                if invert:
                    rotated_inp, rotated_out = 9-rotated_inp,9-rotated_out
                assert np.array_equal(solve(rotated_inp),rotated_out)
    # A hand-calculated event sequence: cadence 3, references 4/7,
    # terminal 16 has B (slot 2) in AAB, so indices 4,7,10,13,16
    # receive A,B,A,A,B. Off-phase index 12 remains B.
    path = tuple((0,i) for i in range(20))
    record = dict(path=path,a=4,b=7,period=3,terminal=16,
                  terminal_color=7,extras={12:7,16:7})
    base = np.full((1,20),2,np.int8)
    expected = base.copy()
    expected[0,[4,7,10,12,13,16]] = [6,7,6,7,6,7]
    assert np.array_equal(_ground_truth(base,[record],(6,6,7)),expected)
    record.update(terminal=None,terminal_color=None,extras={12:7})
    expected = base.copy()
    expected[0,[1,4,7,10,12,13,16,19]] = [6,6,7,6,7,6,7,6]
    assert np.array_equal(_ground_truth(base,[record],(6,6,7)),expected)

    signatures, shapes, structures, codes, words, periods, counts, directions = (set() for _ in range(8))
    features = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        for split in ('train','test'):
            for pair in puzzle[split]:
                inp, out = pair['input'],pair['output']
                saved = inp.copy()
                assert np.array_equal(solve(inp),out)
                assert np.array_equal(inp,saved), 'solve mutated its input'
                info = _decode(inp)
                shapes.add(inp.shape)
                roles = {}
                structures.add((inp.shape,tuple(roles.setdefault(int(c),len(roles)) for c in inp.flat)))
                codes.add(info['code'])
                roles = {}
                words.add(tuple(roles.setdefault(c,len(roles)) for c in info['word']))
                counts.add(len(info['records']))
                for r in info['records']:
                    periods.add(r['period'])
                    if r['terminal'] is not None:
                        directions.add('before' if r['terminal'] < r['a'] else 'after')
                features.append((len(info['records']),tuple(r['legs'] for r in info['records']),
                                 tuple(r['period'] for r in info['records']),
                                 tuple(r['terminal'] for r in info['records'])))
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(words) >= 12 and codes == set(range(8))
    assert {1,2,3,4,5} <= periods and {1,2,3} <= counts
    assert directions == {'before','after'} and len(set(features)) >= 160
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            puzzle = generate_puzzle(700+a*10+b,a,b)
            assert (len(puzzle['train']),len(puzzle['test'])) == (a,b)
    for counts_bad in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1)):
        try:
            generate_puzzle(0,*counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject outputs of each plausible simplified rule on a concrete witness.
    puzzle = generate_puzzle(0,2,1)
    for alternative in _ALTERNATIVES:
        for index,pair in enumerate(puzzle['train']):
            info = _decode(pair['input'])
            wrong = _orient(_paint(info,alternative),info['code'])
            if np.array_equal(wrong,pair['output']):
                continue
            mutated = {s:[dict(p) for p in puzzle[s]] for s in ('train','test')}
            mutated['train'][index]['output'] = wrong
            try:
                validate_generated_puzzle(mutated)
            except AssertionError:
                break
            raise AssertionError('validator accepted '+alternative)
        else:
            raise AssertionError('missing mutation witness '+alternative)


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, legal counts, semantic and mutation checks')
