"""Registered overprinting: geometric keys encode a directed print recipe.

Only standard-library modules and NumPy are used. The input-only decoder does
not consult examples, random seeds, construction plans, or parent generators.
"""
from itertools import combinations
import numpy as np


def _key(grid):
    a = np.asarray(grid)
    return (a.shape, a.astype(np.uint8).tobytes())


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _orders(signatures, cardinality=False):
    """Find up to two Hamiltonian paths; repeated keys are legal if unique."""
    starts = [i for i, (a, b) in enumerate(signatures) if not a and b]
    ends = [i for i, (a, b) in enumerate(signatures) if a and not b]
    if len(starts) != 1 or len(ends) != 1:
        return []
    answers = []

    def visit(path, remaining):
        if len(answers) == 2:
            return
        if not remaining:
            if path[-1] == ends[0]:
                answers.append(tuple(path))
            return
        outgoing = signatures[path[-1]][1]
        if not outgoing:
            return
        for i in remaining:
            incoming = signatures[i][0]
            matches = len(incoming) == len(outgoing) if cardinality else incoming == outgoing
            if incoming and matches:
                visit(path + [i], tuple(j for j in remaining if j != i))

    visit([starts[0]], tuple(i for i in range(len(signatures)) if i != starts[0]))
    return answers


def _parse(input_grid):
    """Read equal cards, background, axis, recipe, and counters from pixels."""
    g = _grid(input_grid)
    candidates = []
    for divider in np.unique(g):
        rows = list(np.flatnonzero(np.all(g == divider, axis=1)))
        cols = list(np.flatnonzero(np.all(g == divider, axis=0)))
        if not rows and not cols:
            continue
        rr, cc = [-1] + rows + [g.shape[0]], [-1] + cols + [g.shape[1]]
        hs = [b-a-1 for a, b in zip(rr, rr[1:])]
        ws = [b-a-1 for a, b in zip(cc, cc[1:])]
        if min(hs+ws) < 4 or len(set(hs)) != 1 or len(set(ws)) != 1:
            continue
        cards = [g[a+1:rr[i+1], b+1:cc[j+1]].copy()
                 for i, a in enumerate(rr[:-1]) for j, b in enumerate(cc[:-1])]
        if any(np.any(card == divider) for card in cards):
            continue
        corners = {int(p[r, c]) for p in cards
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1:
            continue
        bg = next(iter(corners))
        active = [p for p in cards if np.any(p != bg)]
        if not 3 <= len(active) <= 8:
            continue
        for axis in ('H', 'V'):
            plates = active if axis == 'H' else [p.T.copy() for p in active]
            h, w = plates[0].shape[0]-2, plates[0].shape[1]-2
            if h not in (4, 6, 8) or not 2 <= w <= 8:
                continue
            if any(np.any(p[0] != bg) or np.any(p[-1] != bg) for p in plates):
                continue
            sigs = [(tuple(map(int, np.flatnonzero(p[1:-1, 0] != bg))),
                     tuple(map(int, np.flatnonzero(p[1:-1, -1] != bg)))) for p in plates]
            if any((not a and not b) or max(len(a), len(b)) > 3 for a, b in sigs):
                continue
            orders = _orders(sigs)
            if len(orders) != 1:
                continue
            order = orders[0]
            payloads = [p[1:-1, 1:-1].copy() for p in plates]
            carrier, counter = payloads[order[0]], payloads[order[-1]]
            if np.any(carrier == bg):
                continue
            height = int(np.count_nonzero(counter[:h//2] != bg))
            width = int(np.count_nonzero(counter[h//2:] != bg))
            cap = min(h*w//2, 30)
            if not (1 <= height <= cap and 1 <= width <= cap):
                continue
            if (not np.all(carrier == carrier[:1]) and (height < h or height % h)
                    or not np.all(carrier == carrier[:, :1]) and (width < w or width % w)):
                continue
            candidates.append(dict(background=bg, divider=int(divider), axis=axis,
                                   h=h, w=w, height=height, width=width,
                                   plates=plates, payloads=payloads, signatures=sigs,
                                   order=order, layout=(len(hs), len(ws)),
                                   blanks=len(cards)-len(active)))
    assert len(candidates) == 1, 'one visible partition and one complete recipe required'
    return candidates[0]


def _render(d, *, sequence=None, register='incoming', carrier_mode='scale',
            stencil_mode='tile', transparent=True, omit=None):
    """Vectorized construction and explicit alternative-rule checks."""
    h, w, H, W, bg = d['h'], d['w'], d['height'], d['width'], d['background']
    order, data, sigs = d['order'], d['payloads'], d['signatures']
    carrier = data[order[0]]
    r, c = np.arange(H)[:, None], np.arange(W)[None, :]
    sr = r % h if carrier_mode == 'tile_rows' else r*h//H
    sc = c % w if carrier_mode == 'tile_cols' else c*w//W
    out = carrier[sr, sc].copy()
    sequence = order[1:-1] if sequence is None else sequence
    for i in sequence:
        if i == omit:
            continue
        a, b = sigs[i]
        shift = (min(a) if register == 'incoming' else min(b) if register == 'outgoing'
                 else max(a) if register == 'maximum' else len(a)-1 if register == 'count' else 0)
        if stencil_mode == 'tile':
            ink = data[i][(r-shift) % h, c % w]
        else:
            # Wrong alternative: register at native scale, then enlarge the stencil.
            ink = data[i][(r*h//H-shift) % h, c*w//W]
        out = np.where(ink != bg, ink, out) if transparent else ink.copy()
    return out if d['axis'] == 'H' else out.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the actual input, using cell replication then ordered paint."""
    d = _parse(input_grid)
    h, w, H, W, bg = d['h'], d['w'], d['height'], d['width'], d['background']
    data, order = d['payloads'], d['order']
    base = data[order[0]]
    if np.all(base == base[:1]):
        out = np.repeat(base[:1], H, axis=0)
    else:
        out = np.repeat(base, H//h, axis=0)
    if np.all(base == base[:, :1]):
        out = np.repeat(out[:, :1], W, axis=1)
    else:
        out = np.repeat(out, W//w, axis=1)
    for i in order[1:-1]:
        anchor = min(d['signatures'][i][0])
        stencil = data[i]
        for r in range(H):
            for c in range(W):
                ink = stencil[(r-anchor) % h, c % w]
                if ink != bg:
                    out[r, c] = ink
    return out.astype(int) if d['axis'] == 'H' else out.T.astype(int)


_REQUIRED = frozenset(('scan_order', 'reverse', 'port_cardinality', 'port_colors',
                       'registration', 'outgoing_anchor', 'maximum_anchor', 'count_anchor',
                       'tile_rows', 'tile_cols', 'scale_stencil', 'opaque', 'no_stencils',
                       'counter_swap', 'counter_bbox', 'counter_one_ink', 'counter_inks',
                       'visible_base', 'overlap', 'multiple_layers', 'all_port_kinds'))


def _kind(s):
    return 'single' if len(s) == 1 else 'thick' if s[-1]-s[0]+1 == len(s) else 'separated'


def _evidence(d, output):
    cover = set()
    order = d['order']
    layers = list(order[1:-1])
    natural = sorted(layers)
    alternatives = dict(scan_order=_render(d, sequence=natural),
                        reverse=_render(d, sequence=layers[::-1]),
                        registration=_render(d, register='none'),
                        outgoing_anchor=_render(d, register='outgoing'),
                        maximum_anchor=_render(d, register='maximum'),
                        count_anchor=_render(d, register='count'),
                        tile_rows=_render(d, carrier_mode='tile_rows'),
                        tile_cols=_render(d, carrier_mode='tile_cols'),
                        scale_stencil=_render(d, stencil_mode='scale'),
                        opaque=_render(d, transparent=False),
                        no_stencils=_render(d, sequence=[]))
    for name, answer in alternatives.items():
        if not np.array_equal(output, answer):
            cover.add(name)
    if sum(not np.array_equal(output, _render(d, omit=i)) for i in layers) >= 2:
        cover.add('multiple_layers')
    coarse = _orders(d['signatures'], cardinality=True)
    if len(coarse) > 1:
        cover.add('port_cardinality')
    plates, bg = d['plates'], d['background']
    if any(not np.array_equal(plates[a][1:-1, -1], plates[b][1:-1, 0])
           for a, b in zip(order, order[1:])):
        cover.add('port_colors')
    H, W, h = d['height'], d['width'], d['h']
    if H != W:
        cover.add('counter_swap')
    counter = d['payloads'][order[-1]]
    boxes, maximum, inks = [], [], []
    for half in (counter[:h//2], counter[h//2:]):
        r, c = np.nonzero(half != bg)
        boxes.append(int((r.max()-r.min()+1)*(c.max()-c.min()+1)))
        _, freq = np.unique(half[half != bg], return_counts=True)
        maximum.append(int(freq.max()))
        inks.append(len(freq))
    for name, value in (('counter_bbox', boxes), ('counter_one_ink', maximum), ('counter_inks', inks)):
        if tuple(value) != (H, W):
            cover.add(name)
    r, c = np.indices((d['height'], d['width']))
    union = np.zeros(r.shape, bool)
    occupied = []
    for i in layers:
        a = min(d['signatures'][i][0])
        mask = d['payloads'][i][(r-a) % d['h'], c % d['w']] != bg
        if np.any(union & mask):
            cover.add('overlap')
        union |= mask
        occupied.append(mask)
    if np.any(~union):
        cover.add('visible_base')
    kinds = {_kind(s) for i in order for s in d['signatures'][i] if s}
    if {'single', 'thick', 'separated'} <= kinds and max(
            len(s) for ab in d['signatures'] for s in ab) == 3:
        cover.add('all_port_kinds')
    return cover


def _mask(rng, h, w, density):
    """Lines, local growth, scattered motifs, checker fragments, and frames."""
    r, c = np.indices((h, w))
    style = int(rng.integers(6))
    if style == 0:
        mask = rng.random((h, w)) < density
    elif style == 1:
        mask = np.zeros((h, w), bool)
        for _ in range(int(rng.integers(1, 4))):
            a, b = int(rng.integers(h)), int(rng.integers(w))
            mask |= abs(r-a)+abs(c-b) <= int(rng.integers(2))
    elif style == 2:
        step = int(rng.integers(2, 5))
        mask = (r+int(rng.integers(step))) % step == 0
        mask &= rng.random((h, w)) < .7
    elif style == 3:
        mask = (r+c+int(rng.integers(2))) % 2 == 0
        mask &= rng.random((h, w)) < 2*density
    elif style == 4:
        a, b = sorted(map(int, rng.choice(h, 2, replace=False)))
        x, y = sorted(map(int, rng.choice(w, 2, replace=False)))
        mask = ((r == a) | (r == b) | (c == x) | (c == y)) & (r >= a) & (r <= b) & (c >= x) & (c <= y)
    else:
        mask = np.zeros((h, w), bool)
        point = (int(rng.integers(h)), int(rng.integers(w)))
        frontier = {point}
        for _ in range(max(1, round(h*w*density))):
            if not frontier:
                break
            point = _choose(rng, sorted(frontier))
            frontier.remove(point)
            mask[point] = True
            a, b = point
            for q in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                    frontier.add(q)
    if rng.random() < .5:
        mask = mask[:, ::-1]
    return mask.copy()


def _colored_mask(rng, mask, palette, bg, mono=.35):
    out = np.full(mask.shape, bg, dtype=int)
    count = int(mask.sum())
    if count:
        k = 1 if rng.random() < mono else int(rng.integers(1, min(4, count, len(palette))+1))
        inks = rng.choice(palette, k, replace=False)
        values = np.concatenate((inks, rng.choice(inks, count-k)))
        rng.shuffle(values)
        out[mask] = values
    return out


def _carrier(rng, h, w, palette, rich=False):
    r, c = np.indices((h, w))
    k = int(rng.integers(2 if rich else 1, min(5, len(palette))+1))
    inks = rng.choice(palette, k, replace=False)
    style = int(rng.integers(3 if rich else 6))
    if style == 0:
        labels = rng.integers(k, size=(h, w))
    elif style == 1:
        labels = (r//int(rng.integers(1, 3))+c//int(rng.integers(1, 3))) % k
    elif style == 2:
        labels = np.zeros((h, w), int)
        for i in range(1, k):
            a, b = int(rng.integers(h)), int(rng.integers(w))
            labels[(abs(r-a)+abs(c-b)) <= int(rng.integers(1, h+w))] = i
    elif style in (3, 4):
        labels = np.tile(rng.integers(k, size=w), (h, 1))
        if style == 4:
            labels = np.tile(rng.integers(k, size=h)[:, None], (1, w))
    else:
        labels = np.zeros((h, w), int)
    return inks[labels].astype(int)


def _counter(rng, shape, count, palette, bg):
    rows, cols = shape
    style = int(rng.integers(3))
    mask = np.zeros(shape, bool)
    if style == 0:
        mask.ravel()[rng.choice(rows*cols, count, replace=False)] = True
    elif style == 1:
        rr, cc = rng.permutation(rows), rng.permutation(cols)
        mask[rr[np.arange(count)//cols], cc[np.arange(count) % cols]] = True
    else:
        # Compact prefix with optional transpose; counts are independent of ink.
        if rng.random() < .5:
            mask.T.flat[:count] = True
        else:
            mask.flat[:count] = True
        mask = np.roll(mask, (int(rng.integers(rows)), int(rng.integers(cols))), (0, 1))
    return _colored_mask(rng, mask, palette, bg)


def _layouts(h, w, total):
    return [(nr, nc) for nr in range(1, 6) for nc in range(1, 6)
            if total <= nr*nc <= total+2
            and nr*(h+3)-1 <= 30 and nc*(w+3)-1 <= 30]


def _links(rng, h, total, mode):
    pool = [s for k in (1, 2, 3) for s in combinations(range(h), k)]
    if mode == 'coarse':
        pool = [s for s in pool if len(s) == 2]
    for _ in range(100):
        distinct = total-1
        repeat = mode != 'coarse' and total >= 5 and rng.random() < .35
        if repeat:
            distinct -= 1
        idx = rng.choice(len(pool), distinct, replace=False)
        links = [pool[int(i)] for i in idx]
        if repeat:
            j = int(rng.integers(len(links)))
            links.insert(j, links[j])
        kinds = {_kind(s) for s in links}
        if mode == 'coarse' and not {'thick', 'separated'} <= kinds:
            continue
        if mode == 'extensions' and (not {'single', 'thick', 'separated'} <= kinds or max(map(len, links)) != 3):
            continue
        return links
    # Constructive key fallback: distinct signatures, no order ambiguity.
    if mode == 'coarse':
        return [(0, 1), (1, 3), (2, 3), (0, 2)][:total-1]
    chosen = [(1,), (0, 1), (0, 2), (0, 1, 3)]
    return (chosen + [s for s in pool if s not in chosen])[:total-1]


def _pack_payloads(rng, params, payloads, links, axis, *, layout=None,
                   order=None, random_ports=True):
    """Construct cards from independently chosen payloads and geometric keys."""
    bg, divider, palette = params['bg'], params['divider'], params['palette']
    total = len(payloads)
    h, w = payloads[0].shape
    H = int(np.count_nonzero(payloads[-1][:h//2] != bg))
    W = int(np.count_nonzero(payloads[-1][h//2:] != bg))
    plates = []
    for i, data in enumerate(payloads):
        p = np.full((h+2, w+2), bg, int)
        p[1:-1, 1:-1] = data
        for side, sig in ((0, () if i == 0 else links[i-1]),
                          (-1, () if i == total-1 else links[i])):
            for pos in sig:
                p[pos+1, side] = int(rng.choice(palette)) if random_ports else palette[1 if side == 0 else 0]
        plates.append(p if axis == 'H' else p.T.copy())
    if layout is None:
        layout = _choose(rng, _layouts(h if axis == 'H' else w, w if axis == 'H' else h, total))
    nr, nc = layout
    ph, pw = plates[0].shape
    if order is None:
        slots = list(range(total)) + [None]*(nr*nc-total)
        rng.shuffle(slots)
    else:
        slots = list(order) + [None]*(nr*nc-total)
    inp = np.full((nr*(ph+1)-1, nc*(pw+1)-1), divider, int)
    for j, index in enumerate(slots):
        r, c = divmod(j, nc)
        block = np.full((ph, pw), bg, int) if index is None else plates[index]
        inp[r*(ph+1):r*(ph+1)+ph, c*(pw+1):c*(pw+1)+pw] = block
    # Ground truth uses the unshuffled construction, not a solve() call.
    d = dict(h=h, w=w, height=H, width=W, background=bg, axis=axis,
             order=tuple(range(total)), payloads=payloads,
             signatures=list(zip([()] + links, links + [()])))
    return {'input': inp, 'output': _render(d)}


def _sample(rng, params, layers, axis, mode='ordinary'):
    bg, divider, palette = params['bg'], params['divider'], params['palette']
    total = layers+2
    shapes = [(h, w) for h in (4, 6, 8) for w in range(2, 9)
              if _layouts(h if axis == 'H' else w, w if axis == 'H' else h, total)
              and (mode == 'ordinary' or w >= 4)]
    h, w = _choose(rng, shapes)
    carrier = _carrier(rng, h, w, palette, rich=mode != 'ordinary')
    cap = min(h*w//2, 30)
    varies_r = not np.all(carrier == carrier[:1])
    varies_c = not np.all(carrier == carrier[:, :1])
    heights = list(range(h, cap+1, h)) if varies_r else list(range(1, cap+1))
    widths = list(range(w, cap+1, w)) if varies_c else list(range(1, cap+1))
    if mode != 'ordinary':
        heights = [x for x in heights if x >= 2*h]
        widths = [x for x in widths if x >= 2*w]
    H, W = _choose(rng, heights), _choose(rng, widths)
    payloads = [carrier]
    for _ in range(layers):
        mask = _mask(rng, h, w, float(rng.uniform(.08, .40)))
        if mode == 'ordinary' and rng.random() < .08:
            mask[:] = False
        payloads.append(_colored_mask(rng, mask, palette, bg))
    counter = np.vstack((_counter(rng, (h//2, w), H, palette, bg),
                         _counter(rng, (h//2, w), W, palette, bg)))
    payloads.append(counter)
    links = _links(rng, h, total, mode)
    return _pack_payloads(rng, params, payloads, links, axis)


def _fallback(rng, params, mode, axis, layers=4, serial=0, avoid_shape=None):
    """Finite constructive witnesses, used only if random selection is exhausted.

    The coarse witness separates every main alternative. The extension witness
    supplies the remaining port vocabulary. Deep witnesses give every stencil
    its own visible residue, so they cannot lose the depth shift to occlusion.
    """
    bg, palette = params['bg'], params['palette']
    if mode == 'coarse':
        h = w = 6
        r, c = np.indices((h, w))
        carrier = np.asarray(palette[:3])[(r+2*c) % 3]
        s, t = np.full((h, w), bg, int), np.full((h, w), bg, int)
        for point in ((0, 0), (1, 1), (4, 3), (5, 2)):
            s[point] = palette[2]
        for point in ((5, 0), (2, 2), (3, 4), (0, 5)):
            t[point] = palette[1]
        top = np.where((r[:3]+c[:3]) % 3 != 0,
                       np.asarray(palette[:3])[(r[:3]+c[:3]) % 3], bg)
        bottom = np.asarray(palette[:3])[(r[:3]+c[:3]) % 3]
        payloads = [carrier, s, t, np.vstack((top, bottom))]
        links = [(1, 2), (2, 4), (0, 3)]
        return _pack_payloads(rng, params, payloads, links, axis,
                              layout=(2, 2), order=(2, 0, 3, 1), random_ports=False)
    h, w = 4, 7 if mode == 'extensions' and avoid_shape == (4, 6) else 6
    r, c = np.indices((h, w))
    carrier = np.asarray(palette[:2])[(r+c) % 2]
    layers = 3 if mode == 'extensions' else layers
    links = [(1,), (0, 1), (0, 2), (0, 1, 3), (2, 3), (0,), (1, 3)][:layers+1]
    payloads = [carrier]
    for i in range(layers):
        stencil = np.full((h, w), bg, int)
        anchor = min(links[i])
        stencil[(i % h-anchor) % h, i] = palette[2]
        payloads.append(stencil)
    counter = np.full((h, w), bg, int)
    counter[:h//2].flat[:8] = palette[0]
    counter[h//2:] = palette[1]
    # 3**4 counter colorings > the maximum 14 input grids in a puzzle.
    for i in range(4):
        counter[0, i] = palette[(serial//(3**i)) % 3]
    payloads.append(counter)
    layout = min(_layouts(h if axis == 'H' else w, w if axis == 'H' else h, layers+2),
                 key=lambda shape: (shape[0]*shape[1], abs(shape[0]-shape[1])))
    return _pack_payloads(rng, params, payloads, links, axis, layout=layout,
                          order=list(range(layers+2))[::-1], random_ports=False)


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError('illegal example count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if a <= b:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    params = dict(bg=colors[0], divider=colors[1], palette=colors[2:2+int(rng.integers(3, 8))])
    axes = ['H', 'V']
    rng.shuffle(axes)
    train, seen, evidence = [], set(), set()
    for j, mode in enumerate(('coarse', 'extensions')):
        target = _REQUIRED - {'all_port_kinds'} if j == 0 else _REQUIRED - evidence
        for _ in range(256):
            pair = _sample(rng, params, 2 if j == 0 else 3, axes[j], mode)
            d = _parse(pair['input'])
            cover = _evidence(d, pair['output'])
            if (target <= cover and (not train or (d['h'], d['w']) !=
                                     (_parse(train[0]['input'])['h'], _parse(train[0]['input'])['w']))):
                train.append(pair)
                seen.add(_key(pair['input']))
                evidence |= cover
                break
        else:
            first = _parse(train[0]['input']) if train else None
            pair = _fallback(rng, params, mode, axes[j],
                             avoid_shape=(first['h'], first['w']) if first else None)
            cover = _evidence(_parse(pair['input']), pair['output'])
            assert target <= cover
            train.append(pair)
            seen.add(_key(pair['input']))
            evidence |= cover
    puzzle = {'train': train, 'test': []}
    for split, count in (('train', a), ('test', b)):
        while len(puzzle[split]) < count:
            for _ in range(100):
                layers = int(rng.integers(1, 4)) if split == 'train' else int(rng.integers(4, 7))
                pair = _sample(rng, params, layers, _choose(rng, axes))
                d = _parse(pair['input'])
                effective = sum(not np.array_equal(pair['output'], _render(d, omit=i))
                                for i in d['order'][1:-1])
                if _key(pair['input']) not in seen and (split == 'train' or effective >= 4):
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                for serial in range(15):
                    pair = _fallback(rng, params, 'ordinary', axes[0],
                                     layers=2 if split == 'train' else 4, serial=serial)
                    if _key(pair['input']) not in seen:
                        puzzle[split].append(pair)
                        seen.add(_key(pair['input']))
                        break
                else:
                    raise AssertionError('finite fallback uniqueness exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, conventions, evidence, train_shapes, axes = set(), set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _parse(inp)
            conventions.add((d['background'], d['divider']))
            assert np.array_equal(_render(d), out), 'incorrect overprint'
            assert np.array_equal(solve(inp), out), 'replication solver disagrees'
            assert d['background'] not in set(map(int, np.unique(out))), 'carrier fills every transparent hole'
            assert np.all(out != d['divider']), 'card borders are instructions, not paint'
            # Unique full signatures, including any repeated self-loop, are checked by _parse.
            assert len(_orders(d['signatures'])) == 1
            if split == 'train':
                evidence |= _evidence(d, out)
                train_shapes.add((d['h'], d['w']))
                axes.add(d['axis'])
                assert 1 <= len(d['order'])-2 <= 3
            else:
                assert 4 <= len(d['order'])-2 <= 6, 'depth extrapolation missing'
                assert sum(not np.array_equal(out, _render(d, omit=i))
                           for i in d['order'][1:-1]) >= 4, 'four visible print passes required'
    assert len(conventions) == 1, 'one background and divider per puzzle'
    assert axes == {'H', 'V'} and len(train_shapes) >= 2
    assert _REQUIRED <= evidence, 'missing demonstrations: ' + str(_REQUIRED-evidence)


def self_test():
    # Numeric regression: registration wraps above its anchor; horizontal
    # resizing replicates cells while the stencil retains its native period.
    params = dict(bg=0, divider=9, palette=[1, 2, 3, 6, 7])
    rng = np.random.default_rng(413)
    carrier = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1], [1, 3, 2]])
    stencil = np.zeros((4, 3), int)
    stencil[3, 1], stencil[0, 0] = 7, 6
    counter = np.ones((4, 3), int)
    counter[:2].flat[4:] = 0
    expected = np.array([[1, 7, 2, 2, 7, 3], [6, 3, 1, 6, 2, 2],
                         [2, 2, 3, 3, 1, 1], [1, 1, 3, 3, 2, 2]])
    for axis in ('H', 'V'):
        fixture = _pack_payloads(rng, params, [carrier, stencil, counter],
                                 [(1,), (2, 3)], axis, layout=(1, 4),
                                 order=(2, 1, 0), random_ports=False)
        assert np.array_equal(solve(fixture['input']), expected if axis == 'H' else expected.T)
        sparse = np.zeros((4, 3), int)
        sparse[0, 0] = 1
        sparse[2, :2] = 3
        empty = _pack_payloads(rng, params, [np.full((4, 3), 2), np.zeros((4, 3), int), sparse],
                               [(1,), (2, 3)], axis, layout=(1, 3), random_ports=False)
        assert np.array_equal(solve(empty['input']), np.full((1, 2) if axis == 'H' else (2, 1), 2))
        coarse = _fallback(rng, params, 'coarse', axis)
        assert _REQUIRED-{'all_port_kinds'} <= _evidence(_parse(coarse['input']), coarse['output'])
        for shape in (None, (4, 6)):
            extra = _fallback(rng, params, 'extensions', axis, avoid_shape=shape)
            assert 'all_port_kinds' in _evidence(_parse(extra['input']), extra['output'])
            assert np.array_equal(solve(extra['input']), extra['output'])
        for layers in range(4, 7):
            deep = _fallback(rng, params, 'ordinary', axis, layers)
            d = _parse(deep['input'])
            assert np.array_equal(solve(deep['input']), deep['output'])
            assert sum(not np.array_equal(deep['output'], _render(d, omit=i))
                       for i in d['order'][1:-1]) == layers
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, features, masks, bases, keys = set(), [], set(), set(), set()
    repeated = empty_masks = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                d = _parse(pair['input'])
                features.append((split, d['h'], d['w'], d['axis'], len(d['order'])-2,
                                 pair['input'].shape, pair['output'].shape, d['layout'], d['blanks']))
                base = d['payloads'][d['order'][0]]
                labels = {}
                bases.add((base.shape, bytes(labels.setdefault(int(v), len(labels)) for v in base.flat)))
                joins = [d['signatures'][i][1] for i in d['order'][:-1]]
                repeated += len(set(joins)) < len(joins)
                for i in d['order'][1:-1]:
                    mask = d['payloads'][i] != d['background']
                    masks.add(_key(mask))
                    empty_masks += not mask.any()
                keys.update(s for ab in d['signatures'] for s in ab if s)
    assert len(signatures) == 42
    assert len({x[5] for x in features}) >= 20 and len({x[6] for x in features}) >= 20
    assert {x[4] for x in features} == set(range(1, 7))
    assert {x[1] for x in features} == {4, 6, 8}
    assert len({(x[1], x[2]) for x in features}) >= 15
    assert any(x[8] for x in features) and any(not x[8] for x in features)
    assert len(masks) >= 150 and len(bases) >= 100 and len(keys) >= 30
    assert repeated >= 5 and empty_masks >= 5
    assert tuple(_key(p['input']) for p in puzzles[17]['train']) == tuple(
        _key(p['input']) for p in generate_puzzle(17)['train'])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Ambiguous keys must not acquire a hidden generation-order tiebreaker.
    a = (1, 2)
    assert len(_orders([((), a), (a, a), (a, a), (a, ())])) == 2
    assert len(_orders([((), a), (a, a), (a, ())])) == 1
    damaged = {s: [{k: np.array(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (damaged['test'][0]['output'][0, 0]+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, structural diversity, evidence, counts, and negatives')
