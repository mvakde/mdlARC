"""Cadenced perforation: phase-selected, color-keyed travelling stencils.

No generation state is used by solve. The decoder reads rectangles, rail caps,
color equality and full-span stencil registers. Labels are constructed from
sampled geometry; validators also check that training distinguishes ablations.
"""

import numpy as np


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _key(grid):
    g = np.asarray(grid)
    return g.shape, g.astype(np.uint8).tobytes()


def _checked_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError('expected an integer ARC grid of dimensions 1..30')
    return g


def _components(g, bg):
    seen = np.zeros(g.shape, dtype=bool)
    groups = []
    for yy, xx in np.argwhere(g != bg):
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        color, stack, cells = int(g[y, x]), [(y, x)], []
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = r + dr, c + dc
                if (0 <= a < g.shape[0] and 0 <= b < g.shape[1]
                        and not seen[a, b] and g[a, b] == color):
                    seen[a, b] = True
                    stack.append((a, b))
        groups.append((color, np.array(cells, dtype=int)))
    return groups


def _decode(input_grid):
    """Recover all conventions solely from input geometry and color equality."""
    g = _checked_grid(input_grid)
    bg = int(g[0, 0])
    if not all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])):
        raise ValueError('input needs a uniform background border')
    strips = []
    for color, cells in _components(g, bg):
        y, x = (int(v) for v in cells.min(axis=0))
        endy, endx = (int(v) + 1 for v in cells.max(axis=0))
        h, w = endy-y, endx-x
        if min(h, w) in (5, 7) and max(h, w) > min(h, w) and len(cells) == h*w:
            strips.append({'color': color, 'y': y, 'x': x, 'h': h, 'w': w,
                           'axis': 'H' if w > h else 'V', 'k': min(h, w)-2,
                           'length': max(h, w)-2})
    if not strips or len({s['color'] for s in strips}) != len(strips):
        raise ValueError('need identifiable strips with distinct material colors')
    if len({s['k'] for s in strips}) != 1:
        raise ValueError('all strips must share the same inset thickness')
    k = strips[0]['k']
    material_colors = {s['color'] for s in strips}
    reserved = np.zeros(g.shape, dtype=bool)
    for s in strips:
        y, x, h, w, color = (s[n] for n in ('y', 'x', 'h', 'w', 'color'))
        outside = g == color
        outside = outside.copy()
        outside[y:y+h, x:x+w] = False
        caps = {tuple(map(int, p)) for p in np.argwhere(outside)}
        choices = []
        if s['axis'] == 'H':
            for r in (y-2, y+h+1):
                if 0 <= r < g.shape[0] and caps == {(r, x), (r, x+w-1)}:
                    choices.append((r, g[r, x+1:x+w-1].copy()))
        else:
            for c in (x-2, x+w+1):
                if 0 <= c < g.shape[1] and caps == {(y, c), (y+h-1, c)}:
                    choices.append((c, g[y+1:y+h-1, c].copy()))
        if len(choices) != 1:
            raise ValueError('rail caps do not uniquely associate with strip')
        rail_pos, values = choices[0]
        positions = np.flatnonzero(values != bg)
        hues, counts = np.unique(values[positions], return_counts=True)
        repeated = [int(c) for c, n in zip(hues, counts) if n == 2]
        if len(repeated) != 1 or any(n > 2 for n in counts):
            raise ValueError('rail needs exactly one repeated hue, occurring twice')
        if material_colors & set(map(int, hues)):
            raise ValueError('material hues cannot name tools')
        ref = repeated[0]
        a, b = map(int, np.flatnonzero(values == ref))
        p = b-a
        extras = [int(t) for t in positions if t not in (a, b)]
        aligned = [t for t in extras if (t-a) % p == 0]
        if len(aligned) > 1 or (aligned and a < aligned[0] < b):
            raise ValueError('multiple or interior terminals')
        t = aligned[0] if aligned else None
        s.update({'rail': rail_pos, 'values': values, 'a': a, 'b': b,
                  'period': p, 'ref': ref, 'terminal': t,
                  'tool': ref if t is None else int(values[t]),
                  'offphase': tuple(z for z in extras if z != t)})
        area = np.zeros(g.shape, dtype=bool)
        if s['axis'] == 'H':
            lo, hi = min(y, rail_pos), max(y+h, rail_pos+1)
            area[lo:hi, x:x+w] = True
            gap = y-1 if rail_pos < y else y+h
            if not np.all(g[gap, x:x+w] == bg):
                raise ValueError('rail must have a blank gap from material')
        else:
            lo, hi = min(x, rail_pos), max(x+w, rail_pos+1)
            area[y:y+h, lo:hi] = True
            gap = x-1 if rail_pos < x else x+w
            if not np.all(g[y:y+h, gap] == bg):
                raise ValueError('rail must have a blank gap from material')
        if (area & reserved).any():
            raise ValueError('strip and rail registers must be disjoint')
        reserved |= area

    external = (g != bg) & ~reserved
    tools, boxes = {}, {}
    for color in map(int, np.unique(g[external])):
        if color in material_colors:
            raise ValueError('extraneous material-colored ink')
        cells = np.argwhere(external & (g == color))
        y, x = (int(v) for v in cells.min(axis=0))
        endy, endx = (int(v)+1 for v in cells.max(axis=0))
        if (endy-y, endx-x) != (k, k) or reserved[y:endy, x:endx].any():
            raise ValueError('tool stencil does not span an external k-square')
        reg = g[y:endy, x:endx]
        if not np.all((reg == bg) | (reg == color)):
            raise ValueError('mixed stencil register')
        mask = reg == color
        if not _full_span(mask) or not 0 < mask.sum() < k*k:
            raise ValueError('empty, full or nonspanning stencil')
        tools[color], boxes[color] = mask, (y, x, endy, endx)
    for i, first in enumerate(list(boxes.values())):
        for second in list(boxes.values())[i+1:]:
            if not (first[2] < second[0] or second[2] < first[0]
                    or first[3] < second[1] or second[3] < first[1]):
                raise ValueError('stencil registers need a background gutter')
    for s in strips:
        if not set(map(int, s['values'][s['values'] != bg])) <= set(tools):
            raise ValueError('every rail hue must have an external stencil')
    y0 = min(s['y'] for s in strips)-1
    x0 = min(s['x'] for s in strips)-1
    y1 = max(s['y']+s['h'] for s in strips)+1
    x1 = max(s['x']+s['w'] for s in strips)+1
    return {'bg': bg, 'k': k, 'strips': strips, 'tools': tools,
            'boxes': boxes, 'crop': (y0, x0, y1, x1)}


def _centers(s, mode=None):
    p, a, b, length, t = (s[n] for n in ('period', 'a', 'b', 'length', 'terminal'))
    if mode == 'observed':
        return sorted({a, b} | ({t} if t is not None else set()))
    phase = list(range(a % p, length, p))
    if t is not None and mode not in ('ignore_terminal', 'unbounded_terminal'):
        phase = [z for z in phase if min(a, t) <= z <= max(b, t)]
    if mode == 'all_marks':
        phase = sorted(set(phase) | set(s['offphase']))
    return phase


def _transform(mask, index):
    return np.rot90(mask, index) if index < 4 else np.rot90(np.fliplr(mask), index-4)


def _render(info, mode=None, transform=0):
    """Rule renderer, also used for named, deliberately incorrect ablations."""
    bg, k = info['bg'], info['k']
    y0, x0, y1, x1 = info['crop']
    out = np.full((y1-y0, x1-x0), bg, dtype=np.int8)
    for s in info['strips']:
        y, x, h, w = (s[n] for n in ('y', 'x', 'h', 'w'))
        tile = np.full((h, w), s['color'], dtype=np.int8)
        tool = s['ref'] if mode in ('ignore_terminal', 'reference_tool') else s['tool']
        if mode == 'bank_order':
            tool = min(info['boxes'], key=lambda c: info['boxes'][c][:2])
        elif mode == 'nearest_tool':
            cy, cx = y+(h-1)/2, x+(w-1)/2
            def distance(color):
                top, left, bottom, right = info['boxes'][color]
                return ((cy-(top+bottom-1)/2)**2 + (cx-(left+right-1)/2)**2,
                        top, left)
            tool = min(info['boxes'], key=distance)
        mask = _transform(info['tools'][tool], transform)
        if mode == 'all_tools':
            mask = np.logical_or.reduce([_transform(m, transform) for m in info['tools'].values()])
        if mode == 'square':
            mask = np.ones((k, k), dtype=bool)
        elif mode == 'point':
            mask = np.zeros((k, k), dtype=bool)
            mask[k//2, k//2] = True
        hits = np.zeros((h-2, w-2), dtype=int)
        for z in _centers(s, mode):
            cr, cc = (k//2, z) if s['axis'] == 'H' else (z, k//2)
            if mode == 'drop_clipped' and (cr-k//2 < 0 or cc-k//2 < 0
                    or cr+k//2 >= h-2 or cc+k//2 >= w-2):
                continue
            for dr, dc in np.argwhere(mask):
                r, c = cr + int(dr)-k//2, cc + int(dc)-k//2
                if not (0 <= r < h-2 and 0 <= c < w-2):
                    continue
                if mode == 'hard_stop' and s['terminal'] is not None:
                    along = c if s['axis'] == 'H' else r
                    if not min(s['a'], s['terminal']) <= along <= max(s['b'], s['terminal']):
                        continue
                hits[r, c] += 1
        holes = hits % 2 == 1 if mode == 'xor' else hits > 0
        tile[1:-1, 1:-1][holes] = bg
        out[y-y0:y-y0+h, x-x0:x-x0+w] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any valid family input without a seed or stored examples."""
    return _render(_decode(input_grid))


def _full_span(mask):
    return all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))


def _mask(rng, k, asymmetric=False):
    for _ in range(120):
        family = int(rng.integers(9))
        m = np.zeros((k, k), dtype=bool)
        if family == 0:
            np.fill_diagonal(m, True)
            if rng.random() < .5:
                m |= np.fliplr(m)
        elif family == 1:
            m[0, :] = m[:, 0] = True
            if rng.random() < .5:
                m[:, -1] = True
        elif family == 2:
            m[0, :] = m[-1, :] = True
        elif family == 3:
            m[k//2, :] = m[:, k//2] = True
        elif family == 4:
            m[0, 0] = m[-1, -1] = True
        elif family == 5:
            m[:] = True
            m.flat[int(rng.integers(k*k))] = False
        else:
            m = rng.random((k, k)) < float(rng.uniform(.2, .75))
        m = _transform(m, int(rng.integers(8))).copy()
        if _full_span(m) and 0 < m.sum() < k*k:
            if not asymmetric or len({_transform(m, i).tobytes() for i in range(8)}) == 8:
                return m
    # Full-span asymmetric fallback, with both ink and empty pixels.
    m[:] = False
    m[0, 0] = m[0, -1] = m[-1, 0] = m[1, -1] = True
    return m


def _signal(rng, length, k, tools, kind, direction=0):
    ref = int(_pick(rng, tools))
    if kind == 'boundary_witness':
        # Separated full stamps expose displayed orientation, and an edge stamp
        # clips. The references are strictly between left/right added centers.
        options = [(p, phase) for p in (k+1, k+2)
                   for phase in range(k//2) if 3*p+phase < length]
        p, phase = _pick(rng, options)
        a, b, terminal = phase+p, phase+2*p, None
    else:
        terminal_mode = kind in ('terminal_witness', 'terminal')
        max_p = min(k-1, (length-5)//2) if kind == 'terminal_witness' else (length-1)//3
        p = int(rng.integers(1, max_p+1))
        options = []
        for a in range(length-p):
            b = a+p
            phase = list(range(a % p, length, p))
            if not terminal_mode:
                if len(phase) >= 3:
                    options.append((a, b, None))
                continue
            for t in phase:
                if not (t < a or t > b) or (direction and (1 if t > b else -1) != direction):
                    continue
                if kind == 'terminal_witness':
                    lo, hi = min(a, t), max(b, t)
                    if not (lo >= k//2 and hi < length-k//2
                            and any(z < lo or z > hi for z in phase)):
                        continue
                options.append((a, b, t))
        if not options:
            return None
        a, b, terminal = _pick(rng, options)
    effective = ref if terminal is None else int(_pick(rng, [c for c in tools if c != ref]))
    available = [c for c in tools if c not in (ref, effective)]
    options = [z for z in range(length) if (z-a) % p]
    n = min(len(available), len(options), int(rng.integers(3)))
    if kind == 'boundary_witness':
        n = 1
    off = [int(z) for z in rng.choice(options, n, replace=False)] if n else []
    return {'length': length, 'k': k, 'a': a, 'b': b, 'period': p,
            'ref': ref, 'terminal': terminal, 'tool': effective,
            'offphase': tuple(off), 'noise': list(zip(off, available[:n]))}


def _bank_layout(rng, specs, k, ntools):
    """Finite fitting layout, allowing variable offsets, banks and whitespace."""
    lengths = [s['length']+2 for s in specs]
    fh = k+4
    options = []
    for gap in (0, 1, 2):
        ph, pw = len(specs)*fh + (len(specs)-1)*gap, max(lengths)
        for nc in range(1, ntools+1):
            nr = (ntools+nc-1)//nc
            for bgap in (1, 2):
                bh, bw = nr*k+(nr-1)*bgap, nc*k+(nc-1)*bgap
                for side in ('right', 'below'):
                    h, w = ((max(ph, bh), pw+1+bw) if side == 'right'
                            else (ph+1+bh, max(pw, bw)))
                    if h <= 28 and w <= 28:
                        options.append((gap, ph, pw, nc, bgap, bh, bw, side, h, w))
    if not options:
        return None
    gap, ph, pw, nc, bgap, bh, bw, side, h, w = _pick(rng, options)
    units = []
    py = int(rng.integers(h-ph+1)) if side == 'right' else 0
    px = 0 if side == 'right' else int(rng.integers(w-pw+1))
    for i, length in enumerate(lengths):
        units.append((py+i*(fh+gap), px+int(rng.integers(pw-length+1)), 'H'))
    by = int(rng.integers(h-bh+1)) if side == 'right' else ph+1
    bx = pw+1 if side == 'right' else int(rng.integers(w-bw+1))
    slots = [(by+(i//nc)*(k+bgap), bx+(i % nc)*(k+bgap)) for i in range(ntools)]
    return h, w, units, slots


def _scatter_layout(rng, specs, k, ntools):
    for _ in range(8):
        h, w = int(rng.integers(26, 29)), int(rng.integers(26, 29))
        axes = [str(_pick(rng, ('H', 'V'))) for _ in specs]
        occupied = np.zeros((h, w), dtype=bool)
        boxes = []
        for i, s in enumerate(specs):
            dims = (k+4, s['length']+2)
            if axes[i] == 'V':
                dims = dims[::-1]
            boxes.append(('unit', i, *dims))
        boxes += [('tool', i, k, k) for i in range(ntools)]
        boxes.sort(key=lambda z: z[2]*z[3], reverse=True)
        placements = {}
        for kind, index, bh, bw in boxes:
            candidates = [(r, c) for r in range(h-bh+1) for c in range(w-bw+1)]
            for j in rng.permutation(len(candidates)):
                r, c = candidates[int(j)]
                if not occupied[r:r+bh, c:c+bw].any():
                    placements[kind, index] = (r, c)
                    occupied[max(0, r-1):min(h, r+bh+1), max(0, c-1):min(w, c+bw+1)] = True
                    break
            else:
                break
        if len(placements) == len(boxes):
            return h, w, [(*placements['unit', i], axes[i]) for i in range(len(specs))], [placements['tool', i] for i in range(ntools)]
    return None


def _constructed_pair(rng, k, count, bg, witness=None, direction=0, scatter=False):
    ntools = int(rng.integers(3, (5 if k == 5 and count > 1 else 6)))
    # Three strips at k=3 and up to five tools fit a finite bank layout.
    ntools = min(ntools, 9-count)
    palette = [int(c) for c in rng.choice([c for c in range(10) if c != bg], count+ntools, replace=False)]
    materials, tools = palette[:count], palette[count:]
    masks = {c: _mask(rng, k) for c in tools}
    specs = []
    for i in range(count):
        if witness == 'boundary':
            length = int(rng.integers(3*(k+1)+1, 25 if k == 3 else 25))
            kind = 'boundary_witness'
        elif witness == 'terminal':
            length = int(rng.integers(max(12, 3*k), 23 if count == 1 else 20))
            kind = 'terminal_witness'
        else:
            length = int(rng.integers(max(k+3, 10), 21 if k == 3 else 20))
            kind = ('terminal' if i == 0 else 'boundary') if count > 1 else 'terminal'
            if count > 2 and i > 1:
                kind = str(_pick(rng, ('boundary', 'terminal')))
        s = _signal(rng, length, k, tools, kind, direction)
        if s is None:
            return None
        s['color'], s['side'] = materials[i], int(rng.integers(2))
        specs.append(s)
    if witness == 'boundary':
        masks[specs[0]['tool']] = _mask(rng, k, asymmetric=True)
    elif witness is None and rng.random() < .25:
        first, second = tools[:2]
        masks[second] = masks[first].copy()
    layout = _scatter_layout(rng, specs, k, ntools) if scatter else None
    if layout is None:
        layout = _bank_layout(rng, specs, k, ntools)
    if layout is None:
        return None
    h, w, units, slots = layout
    inp = np.full((h, w), bg, dtype=np.int8)
    material = inp.copy()
    # Ground truth is painted from sampled centers and masks, before decoding.
    for s, (uy, ux, axis) in zip(specs, units):
        l = s['length']
        block = np.full((k+4, l+2), bg, dtype=np.int8)
        py, rail = (2, 0) if s['side'] == 0 else (0, k+3)
        block[py:py+k+2, :] = s['color']
        block[rail, [0, l+1]] = s['color']
        block[rail, [s['a']+1, s['b']+1]] = s['ref']
        if s['terminal'] is not None:
            block[rail, s['terminal']+1] = s['tool']
        for z, color in s['noise']:
            block[rail, z+1] = color
        clean = np.full_like(block, bg)
        clean[py:py+k+2, :] = s['color']
        mask = masks[s['tool']] if axis == 'H' else masks[s['tool']].T
        holes = np.zeros((k, l), dtype=bool)
        for center in _centers(s):
            left = center-k//2
            lo, hi = max(0, left), min(l, left+k)
            holes[:, lo:hi] |= mask[:, lo-left:hi-left]
        clean[py+1:py+k+1, 1:l+1][holes] = bg
        if axis == 'V':
            block, clean = block.T, clean.T
        bh, bw = block.shape
        inp[uy:uy+bh, ux:ux+bw] = block
        material[uy:uy+bh, ux:ux+bw] = clean
    for (y, x), color in zip(slots, rng.permutation(tools)):
        inp[y:y+k, x:x+k][masks[int(color)]] = int(color)
    # Background margin makes decoding independent of dominant color counts.
    inp = np.pad(inp, 1, constant_values=bg)
    material = np.pad(material, 1, constant_values=bg)
    padding = []
    for dim in inp.shape:
        total = int(rng.integers(min(3, 30-dim)+1))
        first = int(rng.integers(total+1))
        padding.append((first, total-first))
    inp = np.pad(inp, padding, constant_values=bg)
    material = np.pad(material, padding, constant_values=bg)
    # Global transforms move tools together with the material; no local rotation.
    turn, reflect = int(rng.integers(4)), bool(rng.integers(2))
    if reflect:
        inp, material = np.fliplr(inp), np.fliplr(material)
    inp, material = np.rot90(inp, turn).copy(), np.rot90(material, turn).copy()
    yy, xx = np.nonzero(material != bg)
    out = material[yy.min()-1:yy.max()+2, xx.min()-1:xx.max()+2].copy()
    return {'input': inp, 'output': out}


_ABLATIONS = ('observed', 'ignore_terminal', 'reference_tool', 'unbounded_terminal',
              'all_marks', 'square', 'point', 'drop_clipped', 'hard_stop', 'xor',
              'bank_order', 'nearest_tool', 'all_tools')


def _differences(info, output):
    flags = {mode: not np.array_equal(_render(info, mode), output) for mode in _ABLATIONS}
    flags.update({f'orientation_{i}': not np.array_equal(_render(info, transform=i), output)
                  for i in range(1, 8)})
    return flags


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    ne = _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    regime = str(_pick(rng, ('assembly', 'direction', 'scale')))
    fixed_bg, constant_bg = int(rng.integers(10)), rng.random() < .25
    direction = int(_pick(rng, (-1, 1)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        witness = ('boundary' if index == 0 else 'terminal') if index < 2 else None
        if regime == 'scale':
            k = 3 if split == 'train' else 5
        else:
            k = int(_pick(rng, (3, 3, 5)))
            if regime == 'assembly' and split == 'test':
                k = 3
        count = 1 if witness else int(rng.integers(1, 3 if k == 5 or split == 'train' else 4))
        if regime == 'assembly' and split == 'test':
            count = 3
        d = (direction if split == 'train' else -direction) if regime == 'direction' else 0
        bg = fixed_bg if constant_bg else int(rng.integers(10))
        scatter = bool(rng.random() < (.65 if count > 1 else .3))
        for _ in range(160):
            pair = _constructed_pair(rng, k, count, bg, witness, d, scatter)
            if pair is None or _key(pair['input']) in seen:
                continue
            try:
                info = _decode(pair['input'])
            except ValueError:
                continue
            assert np.array_equal(_render(info), pair['output']), 'constructive label disagreement'
            if witness:
                flags = _differences(info, pair['output'])
                needed = (['observed', 'all_marks', 'square', 'point', 'drop_clipped',
                           'bank_order', 'nearest_tool', 'all_tools']
                          + [f'orientation_{i}' for i in range(1, 8)]) if witness == 'boundary' else [
                              'ignore_terminal', 'reference_tool', 'unbounded_terminal', 'hard_stop', 'xor', 'square']
                if not all(flags[name] for name in needed):
                    continue
                if witness == 'terminal' and np.count_nonzero(
                        _render(info, 'reference_tool') != pair['output']) < k:
                    continue
            # Reversals are sampled before global transforms. Require the actual
            # visible direction to match the curriculum, without hidden metadata.
            if d:
                dirs = {1 if s['terminal'] > s['b'] else -1 for s in info['strips'] if s['terminal'] is not None}
                if dirs and dirs != {d}:
                    continue
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            break
        else:
            raise RuntimeError('bounded scene/evidence construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos, evidence = set(), {'train': [], 'test': []}, {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            try:
                g, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
                info = _decode(g)
            except ValueError as exc:
                raise AssertionError(str(exc)) from exc
            assert _key(g) not in seen, 'duplicate input'
            seen.add(_key(g))
            assert np.array_equal(_render(info), out), 'incorrect stencil cuts or collective crop'
            assert set(map(int, np.unique(out))) <= {info['bg'], *(s['color'] for s in info['strips'])}
            assert all(np.all(edge == info['bg']) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            y0, x0, _, _ = info['crop']
            for s in info['strips']:
                y, x = s['y']-y0, s['x']-x0
                tile = out[y:y+s['h'], x:x+s['w']]
                assert all(np.all(edge == s['color']) for edge in (tile[0], tile[-1], tile[:, 0], tile[:, -1]))
                assert np.any(tile[1:-1, 1:-1] == info['bg']), 'every generated strip must have visible cuts'
                assert len(_centers(s)) >= 3, 'generated cadence must contain a completion center'
            infos[split].append(info)
            if split == 'train':
                for name, value in _differences(info, out).items():
                    evidence[name] = evidence.get(name, False) or value
    assert all(evidence.values()), 'training leaves a listed shortcut indistinguishable'
    train_s = [s for info in infos['train'] for s in info['strips']]
    assert any(s['terminal'] is None for s in train_s)
    assert any(s['terminal'] is not None for s in train_s)
    assert any(any(s['terminal'] is not None for s in info['strips'])
               and np.count_nonzero(_render(info, 'reference_tool') != _render(info)) >= info['k']
               and not np.array_equal(_render(info, 'square'), _render(info))
               for info in infos['train']), 'replacement shape must remain clearly visible'
    assert any(s['offphase'] for s in train_s)
    assert any(s['terminal'] is None and _centers(s)[0] < s['a']
               and _centers(s)[-1] > s['b'] for s in train_s)
    assembly = (max(len(i['strips']) for i in infos['train']) < 3
                and all(len(i['strips']) == 3 for i in infos['test']))
    scale = (all(i['k'] == 3 for i in infos['train'])
             and all(i['k'] == 5 for i in infos['test']))
    directions = []
    for split in ('train', 'test'):
        directions.append({1 if s['terminal'] > s['b'] else -1
                           for info in infos[split] for s in info['strips'] if s['terminal'] is not None})
    reversal = directions[0] in ({-1}, {1}) and directions[1] == {-d for d in directions[0]}
    assert assembly or scale or reversal, 'no documented train/test transfer'


def self_test() -> dict:
    """Bounded 42-seed audit, geometry invariants and counterfactual evidence."""
    fingerprints, shapes, structural, colors, periods, sizes, counts, axes, sides = [set() for _ in range(9)]
    modes, directions, occupancies = set(), set(), set()
    symmetric = asymmetric = overlaps = clipped = unused = mixed_axes = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p[r]) for split in ('train', 'test') for p in puzzle[split] for r in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                pairs += 1
                g, out = pair['input'], pair['output']
                saved = g.copy()
                assert np.array_equal(solve(g), out) and np.array_equal(g, saved), 'solve mutates input'
                info = _decode(g)
                shapes.add(g.shape)
                # Normalize colors by first occurrence to measure structural,
                # rather than palette-only, variation.
                mapping = {}
                normalized = tuple(mapping.setdefault(int(v), len(mapping)) for v in g.flat)
                structural.add((g.shape, normalized))
                colors.add(info['bg']); sizes.add(info['k']); counts.add(len(info['strips']))
                unused += bool(set(info['tools']) - {s['tool'] for s in info['strips']})
                mixed_axes += len({s['axis'] for s in info['strips']}) > 1
                overlaps += not np.array_equal(_render(info, 'xor'), out)
                clipped += not np.array_equal(_render(info, 'drop_clipped'), out)
                for mask in info['tools'].values():
                    occupancies.add(int(mask.sum()))
                    asym = len({_transform(mask, i).tobytes() for i in range(8)}) == 8
                    asymmetric += asym; symmetric += not asym
                for s in info['strips']:
                    periods.add(s['period']); axes.add(s['axis'])
                    sides.add((s['axis'], s['rail'] < (s['y'] if s['axis'] == 'H' else s['x'])))
                    modes.add(s['terminal'] is not None)
                    if s['terminal'] is not None:
                        directions.add(1 if s['terminal'] > s['b'] else -1)
                # Whole-scene rotations/reflections and color permutations are
                # semantic metamorphisms, including local mixed axes.
                if seed < 4 and split == 'test':
                    for t in range(8):
                        assert np.array_equal(solve(_transform(g, t)), _transform(out, t))
                    permutation = np.random.default_rng(seed+1234).permutation(10)
                    assert np.array_equal(solve(permutation[g]), permutation[out])
    assert len(fingerprints) == 42 and len(structural) >= 200
    assert len(shapes) >= 50 and colors == set(range(10))
    assert sizes == {3, 5} and counts == {1, 2, 3} and axes == {'H', 'V'}
    assert len(sides) == 4 and modes == {False, True} and directions == {-1, 1}
    assert {1, 2, 3, 4, 5, 6} <= periods
    assert min(symmetric, asymmetric, overlaps, clipped, unused) > 20 and mixed_axes >= 3
    assert min(occupancies) == 2 and max(occupancies) == 24
    a = generate_puzzle(314159, 10, 4)
    b = generate_puzzle(314159, 10, 4)
    assert all(np.array_equal(p[r], q[r]) for split in ('train', 'test')
               for p, q in zip(a[split], b[split]) for r in ('input', 'output'))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(101+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(88, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged crop accepted')
    return {'seeds': 42, 'pairs': pairs, 'input_shapes': len(shapes),
            'normalized_inputs': len(structural), 'periods': sorted(periods),
            'inset_sizes': sorted(sizes), 'strip_counts': sorted(counts),
            'symmetric_masks': symmetric, 'asymmetric_masks': asymmetric,
            'overlap_pairs': overlaps, 'clipped_pairs': clipped,
            'mixed_axis_pairs': mixed_axes, 'background_colors': sorted(colors)}
