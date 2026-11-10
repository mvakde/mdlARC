"""Opposite bar-attribute cycles register a size-bearing square plan.

Only input pixels are used by solve. Generation draws labels from its sampled
plan; recovery separately segments the finished input and registers its clue.
No files, network, execution traces, or retained sample payloads are used here.
"""
from functools import lru_cache
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        stack, cells = [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        points = np.asarray(cells)
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0) + 1)
        result.append((r0, c0, r1, c1, len(cells)))
    return result


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _matches(plan, sizes, colors, lengths):
    """All row segments with a common integral magnification of 2 or 3."""
    n = len(colors)
    matches = []
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1] - n + 1):
            if tuple(map(int, plan[r, c:c+n])) != tuple(colors):
                continue
            unit = sizes[r, c:c+n].astype(int)
            if np.any(unit == 0):
                continue
            ratios = [int(length) / int(side) for length, side in zip(lengths, unit)]
            if ratios[0] in (2, 3) and all(k == ratios[0] for k in ratios):
                matches.append((r, c, int(ratios[0])))
    return matches


def _parse(input_grid):
    a = _grid(input_grid)
    freq = np.bincount(a.ravel(), minlength=10)
    bg = int(freq.argmax())
    _require(freq[bg] > a.size // 2, "background must occupy a strict majority")
    comps = _components(a != bg)
    frames = []
    for r, c, er, ec, area in comps:
        height, width = er-r, ec-c
        if min(height, width) < 9 or area != 2*(height+width)-4:
            continue
        color = int(a[r, c])
        edges = np.r_[a[r, c:ec], a[er-1, c:ec], a[r:er, c], a[r:er, ec-1]]
        if np.all(edges == color) and np.all(a[r+1:er-1, c+1:ec-1] != color):
            frames.append((r, c, er, ec, color))
    _require(len(frames) == 1, "one rectangular plan frame required")
    r0, c0, r1, c1, marker = frames[0]
    inside = _components(a[r0+1:r1-1, c0+1:c1-1] != bg)
    tokens = []
    for r, c, er, ec, area in inside:
        r, c, er, ec = r+r0+1, c+c0+1, er+r0+1, ec+c0+1
        side = er-r
        _require(side in (1, 2) and ec-c == side and area == side*side, "plan swatch square")
        color = int(a[r, c])
        _require(color not in (bg, marker) and np.all(a[r:er, c:ec] == color), "solid plan swatch")
        tokens.append((r, c, side, color))
    plans = []
    for pitch in (3, 4):
        if (r1-r0-6) % pitch or (c1-c0-6) % pitch:
            continue
        h, w = (r1-r0-6)//pitch+1, (c1-c0-6)//pitch+1
        if not (2 <= h <= 4 and 3 <= w <= 5):
            continue
        plan = np.full((h, w), bg, dtype=np.int8)
        sizes = np.zeros((h, w), dtype=np.int8)
        good = True
        for r, c, side, color in tokens:
            rr, cc = r-r0-2, c-c0-2
            if rr < 0 or cc < 0 or rr % pitch or cc % pitch or rr//pitch >= h or cc//pitch >= w:
                good = False
                break
            plan[rr//pitch, cc//pitch] = color
            sizes[rr//pitch, cc//pitch] = side
        if good and np.all(np.any(sizes, axis=0)) and np.all(np.any(sizes, axis=1)):
            plans.append((plan, sizes, pitch))
    _require(len(plans) == 1, "plan pitch and extent must be unique")
    plan, sizes, pitch = plans[0]
    bars, dots = [], []
    for r, c, er, ec, area in comps:
        if r0 <= r and c0 <= c and er <= r1 and ec <= c1:
            continue
        color = int(a[r, c])
        if color == marker:
            _require(area == 1, "outside frame markers must be singleton anchors")
            dots.append((r, c))
        else:
            _require(ec-c == 1 and 2 <= er-r <= 6 and area == er-r and np.all(a[r:er, c] == color), "thin solid bar")
            bars.append((r, c, er, color))
    bars.sort(key=lambda b: b[1])
    dots.sort(key=lambda d: d[1])
    n = len(bars)
    _require(3 <= n <= 5 and len(dots) == n, "equal bar and anchor counts")
    _require(len({b[2] for b in bars}) == 1, "common bar baseline")
    bar_pitch = bars[1][1]-bars[0][1]
    _require(bar_pitch in (2, 3) and all(bars[i][1]-bars[i-1][1] == bar_pitch for i in range(1, n)), "regular bar slots")
    _require(len({d[0] for d in dots}) == 1, "horizontal anchor run")
    stride = dots[1][1]-dots[0][1]
    _require(stride > 0 and all(dots[i][1]-dots[i-1][1] == stride for i in range(1, n)), "regular lattice anchors")
    colors = tuple(b[3] for b in bars)
    lengths = tuple(b[2]-b[0] for b in bars)
    _require(len(set(lengths)) == 2, "two distinct bar lengths")
    _require(set(lengths) in ({2, 4}, {3, 6}), "bar lengths must share magnification")
    return dict(grid=a, background=bg, marker=marker, frame=(r0, c0, r1, c1),
                plan=plan, sizes=sizes, pitch=pitch, bars=bars, dots=dots,
                colors=colors, lengths=lengths, stride=stride, bar_pitch=bar_pitch)


def _render_registered(info, match, seeds_only=False):
    """Check lattice geometry and render without changing source structures."""
    a, plan, sizes = info['grid'], info['plan'], info['sizes']
    row, col, scale = match
    stride = info['stride']
    side_max = int(sizes.max())*scale
    _require(side_max < stride, "positive gutters even for largest squares")
    y0 = info['dots'][0][0]-row*stride
    x0 = info['dots'][0][1]-col*stride
    box = (y0, x0, y0+(plan.shape[0]-1)*stride+side_max,
           x0+(plan.shape[1]-1)*stride+side_max)
    _require(0 <= y0 and 0 <= x0 and box[2] <= a.shape[0] and box[3] <= a.shape[1], "complete lattice bounds")
    _require(not _overlap(box, info['frame']), "lattice must not cover plan")
    for r, c, er, color in info['bars']:
        _require(not _overlap(box, (r, c, er, c+1)), "lattice must not cover bars")
    region = a[box[0]:box[2], box[1]:box[3]].copy()
    for y, x in info['dots']:
        _require(box[0] <= y < box[2] and box[1] <= x < box[3], "all anchors inside lattice")
        region[y-y0, x-x0] = info['background']
    _require(np.all(region == info['background']), "workspace contains only anchors")
    result = a.copy()
    for (r, c), color in np.ndenumerate(plan):
        if int(color) == info['background']:
            continue
        if seeds_only and not (r == row and col <= c < col+len(info['bars'])):
            continue
        y, x = y0+r*stride, x0+c*stride
        side = int(sizes[r, c])*scale
        result[y:y+side, x:x+side] = color
    return result


def _recover(input_grid):
    info = _parse(input_grid)
    colors = tuple(np.roll(info['colors'], 1).tolist())
    lengths = tuple(np.roll(info['lengths'], -1).tolist())
    matches = _matches(info['plan'], info['sizes'], colors, lengths)
    _require(len(matches) == 1, "cycled color/length sequence must register uniquely")
    info['match'] = matches[0]
    return _render_registered(info, matches[0]), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy the registered size-bearing plan using only the input."""
    return _recover(input_grid)[0]


@lru_cache(maxsize=1)
def _options():
    """Feasible three-rectangle packings; no unbounded rejection sampling."""
    options = []
    for h in range(2, 5):
        for w in range(3, 6):
            for n in range(3, w+1):
                for scale in (2, 3):
                    for pitch in (3, 4):
                        ph, pw = (h-1)*pitch+6, (w-1)*pitch+6
                        for stride in range(2*scale+1, 2*scale+4):
                            # Reserve the union of the correct and adjacent
                            # wrong-length registrations. Both must fit.
                            th, tw = h*stride+2*scale, (w-1)*stride+2*scale
                            for bp in (2, 3):
                                rh, rw = 2*scale, (n-1)*bp+1
                                for axis in (0, 1):
                                    if axis == 0:
                                        sh, sw = max(ph, rh), pw+1+rw
                                        need = (sh+1+th, max(sw, tw))
                                    else:
                                        sh, sw = ph+1+rh, max(pw, rw)
                                        need = (max(sh, th), sw+1+tw)
                                    if max(need) <= 30:
                                        options.append((h, w, n, scale, pitch, stride, bp, axis))
    return tuple(options)


def _pack(rng, ph, pw, rh, rw, th, tw, axis):
    """Random orders, offsets and margins in a bounded two-level packing."""
    source_axis = 1-axis
    pshape, rshape = (ph, pw), (rh, rw)
    source_shape = [max(ph, rh), max(pw, rw)]
    source_shape[source_axis] = pshape[source_axis]+1+rshape[source_axis]
    pp, rp = [0, 0], [0, 0]
    if rng.integers(2):
        rp[source_axis] = pshape[source_axis]+1
    else:
        pp[source_axis] = rshape[source_axis]+1
    pp[axis] = int(rng.integers(source_shape[axis]-pshape[axis]+1))
    rp[axis] = int(rng.integers(source_shape[axis]-rshape[axis]+1))
    target_shape = (th, tw)
    need = [max(source_shape[d], target_shape[d]) for d in (0, 1)]
    need[axis] = source_shape[axis]+1+target_shape[axis]
    if rng.random() < .22:
        side = int(rng.integers(max(need), 31))
        shape = (side, side)
    else:
        shape = tuple(int(rng.integers(v, min(30, v+4)+1)) for v in need)
    sp, tp = [0, 0], [0, 0]
    margin = int(rng.integers(shape[axis]-need[axis]+1))
    if rng.integers(2):
        sp[axis], tp[axis] = margin, margin+source_shape[axis]+1
    else:
        tp[axis], sp[axis] = margin, margin+target_shape[axis]+1
    cross = 1-axis
    sp[cross] = int(rng.integers(shape[cross]-source_shape[cross]+1))
    tp[cross] = int(rng.integers(shape[cross]-target_shape[cross]+1))
    return shape, tuple(sp[d]+pp[d] for d in (0, 1)), tuple(sp[d]+rp[d] for d in (0, 1)), tuple(tp)


def _informative(sequence):
    a = np.asarray(sequence)
    return not np.array_equal(a, np.roll(a, 1)) and not np.array_equal(np.roll(a, 1), np.roll(a, -1))


def _attributes(rng, n, palette, distinct):
    # Bounded choices, with an informative constructive fallback.
    for _ in range(32):
        colors = rng.choice(palette, size=n, replace=not distinct).astype(np.int8)
        units = rng.integers(1, 3, size=n, dtype=np.int8)
        if _informative(colors) and _informative(units):
            return colors, units
    colors = np.array(palette[:n] if distinct else [palette[0]]*(n-1)+[palette[1]], dtype=np.int8)
    return colors, np.array([1]*(n-1)+[2], dtype=np.int8)


def _sample_pair(rng, choices, row_kind, holes, distinct, avoid_background=None):
    h, w, n, scale, pitch, stride, bp, axis = choices[int(rng.integers(len(choices)))]
    if row_kind == 'top':
        row = 0
    elif row_kind == 'bottom':
        row = h-1
    elif row_kind == 'interior':
        row = int(rng.integers(1, h-1))
    else:
        row = int(rng.integers(h))
    adjacent = [r for r in (row-1, row+1) if 0 <= r < h]
    decoy = int(rng.choice(adjacent))
    col = int(rng.integers(w-n+1))
    permutation = rng.permutation(10)
    bg, marker = map(int, permutation[:2])
    if bg == avoid_background:
        permutation[0], permutation[1] = permutation[1], permutation[0]
        bg, marker = map(int, permutation[:2])
    palette = permutation[2:2+int(rng.integers(max(2, n if distinct else 2), 9))]
    colors, units = _attributes(rng, n, palette, distinct)
    new_colors, new_units = np.roll(colors, 1), np.roll(units, -1)
    protected = {(r, c) for r in (row, decoy) for c in range(col, col+n)}
    for attempt in range(40):
        plan = rng.choice(palette, size=(h, w)).astype(np.int8)
        sizes = rng.integers(1, 3, size=(h, w), dtype=np.int8)
        for rr, unit in ((row, new_units), (decoy, units)):
            plan[rr, col:col+n] = new_colors
            sizes[rr, col:col+n] = unit
        free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in protected]
        if holes and free:
            order = rng.permutation(len(free))
            count = int(rng.integers(1, max(2, len(free)//2+1)))
            for j in order[:count]:
                rr, cc = free[int(j)]
                if np.count_nonzero(sizes[rr]) > 1 and np.count_nonzero(sizes[:, cc]) > 1:
                    plan[rr, cc], sizes[rr, cc] = bg, 0
        right = _matches(plan, sizes, tuple(new_colors), tuple(scale*new_units))
        wrong = _matches(plan, sizes, tuple(new_colors), tuple(scale*units))
        if right == [(row, col, scale)] and wrong == [(decoy, col, scale)]:
            break
    else:
        # Keep the two specified runs, fill all other occupied sites with a
        # reserved color absent from the bar sequence. No accidental run can
        # then contain the full sequence of >=3 bar colors.
        reserved = next(c for c in permutation[2:] if c not in colors)
        for rr, cc in free:
            if sizes[rr, cc]:
                plan[rr, cc] = reserved
        _require(_matches(plan, sizes, tuple(new_colors), tuple(scale*new_units)) == [(row, col, scale)], "plan fallback")
        _require(_matches(plan, sizes, tuple(new_colors), tuple(scale*units)) == [(decoy, col, scale)], "decoy fallback")
    ph, pw = (h-1)*pitch+6, (w-1)*pitch+6
    rh, rw = 2*scale, (n-1)*bp+1
    th, tw = h*stride+2*scale, (w-1)*stride+2*scale
    shape, mini, rack, target = _pack(rng, ph, pw, rh, rw, th, tw, axis)
    inp = np.full(shape, bg, dtype=np.int8)
    py, px = mini
    inp[py, px:px+pw] = inp[py+ph-1, px:px+pw] = marker
    inp[py:py+ph, px] = inp[py:py+ph, px+pw-1] = marker
    for (r, c), color in np.ndenumerate(plan):
        s = int(sizes[r, c])
        if s:
            y, x = py+2+r*pitch, px+2+c*pitch
            inp[y:y+s, x:x+s] = color
    ry, rx = rack
    for i, (color, unit) in enumerate(zip(colors, units)):
        length = int(unit)*scale
        inp[ry+rh-length:ry+rh, rx+i*bp] = color
    anchor_y = target[0]+max(row, decoy)*stride
    for i in range(n):
        inp[anchor_y, target[1]+(col+i)*stride] = marker
    # Labels come directly from the sampled scene, not from solve/recovery.
    out = inp.copy()
    base_y, base_x = anchor_y-row*stride, target[1]
    for (r, c), color in np.ndenumerate(plan):
        side = int(sizes[r, c])*scale
        if side:
            y, x = base_y+r*stride, base_x+c*stride
            out[y:y+side, x:x+side] = color
    return {'input': inp, 'output': out}, bg


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and int(seed) >= 0, "seed must be a nonnegative integer")
    for name, count, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        _require(_integer(count) and low <= int(count) <= high, name+" outside legal integer range")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(('count', 'interior')))
    first_scale = int(rng.choice((2, 3)))
    first_boundary = str(rng.choice(('top', 'bottom')))
    options = _options()
    puzzle, seen = {'train': [], 'test': []}, set()
    first_bg = None
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            choices = list(options)
            row_kind = 'any'
            holes = bool(rng.random() < .65)
            distinct = bool(rng.random() < .62)
            if split == 'train':
                if shift == 'count':
                    choices = [o for o in choices if o[2] < 5]
                else:
                    choices = [o for o in choices if o[0] == 2]
                if index == 0:
                    choices = [o for o in choices if o[2] == 3 and o[1] > 3 and o[3] == first_scale]
                    row_kind, holes, distinct = first_boundary, True, True
                elif index == 1:
                    choices = [o for o in choices if o[2] == 4 and o[3] == 5-first_scale]
                    row_kind = 'bottom' if first_boundary == 'top' else 'top'
                    distinct = True
            elif shift == 'count':
                choices = [o for o in choices if o[2] == 5]
            else:
                choices = [o for o in choices if o[0] >= 3]
                row_kind = 'interior'
            _require(bool(choices), "empty geometry choice set")
            for attempt in range(24):
                pair, bg = _sample_pair(rng, choices, row_kind, holes, distinct,
                                       first_bg if split == 'train' and index == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded fresh-scene attempts exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                first_bg = bg
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "pair lists")
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "shape preservation")
            key = _key(inp)
            _require(key not in seen, "distinct inputs")
            seen.add(key)
            expected, info = _recover(inp)
            _require(np.array_equal(expected, out), "incorrect completion or changed source")
            _require(_informative(info['colors']) and _informative(info['lengths']), "cycle directions must be exposed")
            colors = tuple(np.roll(info['colors'], 1).tolist())
            new_lengths = tuple(np.roll(info['lengths'], -1).tolist())
            # Direct checks at the visibly supplied anchors also expose A's
            # two independent transfers in the rendered answer.
            for (y, x), color, length in zip(info['dots'], colors, new_lengths):
                _require(np.all(out[y:y+length, x:x+length] == color), "cycled seed square attributes")
                if y+length < out.shape[0]:
                    _require(out[y+length, x] == info['background'], "seed square vertical extent")
                if x+length < out.shape[1]:
                    _require(out[y, x+length] == info['background'], "seed square horizontal extent")
            wrong = _matches(info['plan'], info['sizes'], colors, info['lengths'])
            _require(len(wrong) == 1 and wrong[0] != info['match'], "length-omission decoy required")
            counterfactual = _render_registered(info, wrong[0])
            _require(not np.array_equal(counterfactual, out), "opposite length cycle must change the answer")
            only_seed = _render_registered(info, info['match'], seeds_only=True)
            _require(not np.array_equal(only_seed, out), "B must add squares beyond the seed run")
            if split == 'train':
                training.append(info)
    _require({i['match'][2] for i in training} == {2, 3}, "training must teach local scale recovery")
    _require(len({len(i['bars']) for i in training}) >= 2, "training must teach cyclic wrap at different counts")
    _require(len({i['background'] for i in training}) >= 2, "training must teach local palette roles")


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """42 streamed seeds, all count combinations, negative cases; writes nothing."""
    signatures, shapes, plans, counts, pitches, strides, backgrounds, holes = set(), set(), set(), set(), set(), set(), set(), set()
    shifts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        train_info, test_info = [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                expected, info = _recover(pair['input'])
                assert np.array_equal(expected, pair['output'])
                (train_info if split == 'train' else test_info).append(info)
                shapes.add(pair['input'].shape)
                plans.add(info['plan'].shape)
                counts.add(len(info['bars']))
                pitches.add(info['pitch'])
                strides.add(info['stride'])
                backgrounds.add(info['background'])
                holes.add(int(np.count_nonzero(info['sizes'] == 0)))
        if max(len(i['bars']) for i in test_info) > max(len(i['bars']) for i in train_info):
            shifts.add('count')
        if all(i['plan'].shape[0] == 2 for i in train_info) and all(0 < i['match'][0] < i['plan'].shape[0]-1 for i in test_info):
            shifts.add('interior')
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(plans) >= 5 and counts == {3, 4, 5} and pitches == {3, 4}
    assert strides == {5, 6, 7} and len(backgrounds) == 10 and max(holes) >= 2
    assert shifts == {'count', 'interior'}
    assert _signature(generate_puzzle(711)) == _signature(generate_puzzle(711))
    for nt in range(2, 11):
        for ns in range(1, min(nt-1, 4)+1):
            generate_puzzle(1000+nt*10+ns, nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    puzzle = generate_puzzle(314, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # Make the length decoy identical to the correct segment: reject ambiguity.
    a = puzzle['train'][0]['input'].copy()
    _, info = _recover(a)
    cseq = tuple(np.roll(info['colors'], 1).tolist())
    dr, dc, scale = _matches(info['plan'], info['sizes'], cseq, info['lengths'])[0]
    rr, cc, _ = info['match']
    py, px = info['frame'][:2]
    for j in range(len(info['bars'])):
        y, x = py+2+dr*info['pitch'], px+2+(dc+j)*info['pitch']
        a[y:y+2, x:x+2] = info['background']
        s = int(info['sizes'][rr, cc+j])
        a[y:y+s, x:x+s] = cseq[j]
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError("two pair-sequence matches accepted")

