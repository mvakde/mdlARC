"""Periodic signals toggle scaffold-registered shutters by sensor-hit parity.

Only NumPy and the standard library are used. Solving reads one input; no
sampled construction record, other example, seed, or output label is consulted.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _box(points):
    ys, xs = zip(*points)
    return min(ys), min(xs), max(ys) - min(ys) + 1, max(xs) - min(xs) + 1


def _separate(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y+h < yy or yy+hh < y or x+w < xx or xx+ww < x


def _components(mask, diagonal=True):
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy)+abs(dx) == 1)]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, todo = {p}, [p]
        while todo:
            y, x = todo.pop()
            for dy, dx in steps:
                q = (y+dy, x+dx)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _coords(hub, direction, width, distance):
    y, x, h, w = hub
    if direction < 2:
        yy = y-distance if direction == 0 else y+h-1+distance
        return [(yy, x+(w-width)//2+j) for j in range(width)]
    xx = x-distance if direction == 2 else x+w-1+distance
    return [(y+(h-width)//2+j, xx) for j in range(width)]


def _maximum(shape, hub, direction):
    y, x, h, w = hub
    return (y, shape[0]-y-h, x, shape[1]-x-w)[direction]


def _source_group(grid, group, bg):
    """A filled rectangular body with exactly one centered rectangular tile."""
    found = []
    for color in sorted({int(grid[p]) for p in group}):
        body = {p for p in group if int(grid[p]) == color}
        hub = _box(body)
        y, x, h, w = hub
        if min(h, w) < 3 or max(h, w) > 10 or len(body) != h*w:
            continue
        rest = group-body
        if not rest:
            continue
        for direction in range(4):
            transverse = w if direction < 2 else h
            for width in range(1, transverse-1):
                if (transverse-width) % 2 or len(rest) % width:
                    continue
                length = len(rest)//width
                if length > _maximum(grid.shape, hub, direction):
                    continue
                points = [_coords(hub, direction, width, d) for d in range(1, length+1)]
                if set(p for row in points for p in row) != rest:
                    continue
                word = np.array([[int(grid[p]) for p in row] for row in points], dtype=np.int8)
                if np.any((word == bg) | (word == color)):
                    continue
                found.append(dict(hub=hub, direction=direction, width=width,
                                  word=word, color=color, region=_box(group)))
    assert len(found) <= 1, "ambiguous emitter"
    return found[0] if found else None


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            ends = np.r_[0, np.flatnonzero(row[1:] != row[:-1])+1, len(row)]
            lengths.extend(int(v) for v in np.diff(ends))
    s = math.gcd(*lengths)
    small = crop[::s, ::s]
    assert np.array_equal(crop, np.repeat(np.repeat(small, s, 0), s, 1))
    return s, small


def _ring(points):
    y, x, h, w = _box(points)
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return False
    border = {(yy, xx) for yy in range(y, y+h) for xx in range(x, x+w)
              if yy in (y, y+h-1) or xx in (x, x+w-1)}
    corners = {(y, x), (y, x+w-1), (y+h-1, x), (y+h-1, x+w-1)}
    return border-corners <= points <= border


def _signature(mapping):
    return tuple(sorted((int(y), int(x), int(c)) for (y, x), c in mapping.items()))


def _infer(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg]*2 > grid.size, "background must be a strict majority"
    sources, objects = [], []
    for group in _components(grid != bg):
        source = _source_group(grid, group, bg)
        if source is not None:
            sources.append(source)
            continue
        y, x, h, w = _box(group)
        crop = grid[y:y+h, x:x+w]
        assert np.count_nonzero(crop != bg) == len(group), "interleaved objects"
        s, small = _primitive(crop)
        raw = {(int(yy), int(xx)): int(small[yy, xx])
               for yy, xx in zip(*np.nonzero(small != bg))}
        objects.append(dict(crop=(y, x), scale=s, raw=raw))
    assert sources and len(objects) >= 2
    assert len({s['color'] for s in sources}) == 1
    emitter_color = sources[0]['color']
    common_colors = set.intersection(*(set(o['raw'].values()) for o in objects))
    interpretations = []
    for carrier in sorted(common_colors):
        frames, maps, anchors = [], [], []
        for obj in objects:
            pts = {p for p, c in obj['raw'].items() if c == carrier}
            if not _ring(pts):
                break
            y, x, h, w = _box(pts)
            frames.append(tuple(sorted((yy-y, xx-x) for yy, xx in pts)))
            maps.append({(yy-y, xx-x): c for (yy, xx), c in obj['raw'].items()})
            anchors.append((obj['crop'][0]+y*obj['scale'], obj['crop'][1]+x*obj['scale']))
        else:
            if len(set(frames)) != 1:
                continue
            sigs = sorted({_signature(m) for m in maps})
            if len(sigs) != 2:
                continue
            templates = [{(y, x): c for y, x, c in sig} for sig in sigs]
            stable = {p: c for p, c in templates[0].items()
                      if templates[1].get(p) == c and c != carrier}
            if len(stable) != 1:
                continue
            sensor, active = next(iter(stable.items()))
            _, _, fh, fw = _box(frames[0])
            if not (0 < sensor[0] < fh-1 and 0 < sensor[1] < fw-1):
                continue
            if any(sum(c == active for c in m.values()) != 1 for m in templates):
                continue
            if emitter_color in (carrier, active) or any(emitter_color in m.values() for m in templates):
                continue
            if any(np.any((src['word'] == carrier) | (src['word'] == emitter_color)) for src in sources):
                continue
            accents = set.union(*(set(m.values()) for m in templates))-{carrier, active}
            if not accents or any(set(src['word'].flat) & accents for src in sources):
                continue
            occurrences = []
            union = set(templates[0]) | set(templates[1])
            uy, ux, uh, uw = _box(union)
            regions = [src['region'] for src in sources]
            valid = True
            for obj, mapping, anchor in zip(objects, maps, anchors):
                s = obj['scale']
                region = (anchor[0]+uy*s, anchor[1]+ux*s, uh*s, uw*s)
                y, x, h, w = region
                if not (0 <= y and 0 <= x and y+h <= grid.shape[0] and x+w <= grid.shape[1]):
                    valid = False
                    break
                if not all(_separate(region, r) for r in regions):
                    valid = False
                    break
                regions.append(region)
                sy, sx = anchor[0]+sensor[0]*s, anchor[1]+sensor[1]*s
                occurrences.append(dict(anchor=anchor, scale=s, state=sigs.index(_signature(mapping)),
                                        sensor=(sy, sx, s, s), region=region))
            if not valid or {o['state'] for o in occurrences if o['scale'] == 1} != {0, 1}:
                continue
            interpretations.append(dict(background=bg, carrier=carrier, active=active,
                                        sources=sources, templates=templates, union=union,
                                        occurrences=occurrences, frame=frames[0]))
    assert len(interpretations) == 1, "missing or ambiguous frame/sensor interpretation"
    return interpretations[0]


def _hits(source, sensor, active, first_slice=False, reset=False):
    y, x, h, w = sensor
    sy, sx, sh, sw = source['hub']
    direction, width, word = source['direction'], source['width'], source['word']
    count = covered = 0
    distances = []
    for yy in range(y, y+h):
        for xx in range(x, x+w):
            if direction < 2:
                d = sy-yy if direction == 0 else yy-(sy+sh-1)
                j = xx-(sx+(sw-width)//2)
            else:
                d = sx-xx if direction == 2 else xx-(sx+sw-1)
                j = yy-(sy+(sh-width)//2)
            if d >= 1 and 0 <= j < width:
                distances.append((d, j))
    origin = min((d for d, j in distances), default=1) if reset else 1
    for d, j in distances:
        covered += 1
        phase = 0 if first_slice else (d-origin) % len(word)
        count += int(word[phase, j] == active)
    return count, covered


def _render(grid, template, anchor, scale):
    for (y, x), c in template.items():
        yy, xx = anchor[0]+y*scale, anchor[1]+x*scale
        grid[yy:yy+scale, xx:xx+scale] = c


def _replace(grid, info, flips):
    out = grid.copy()
    for obj, flip in zip(info['occurrences'], flips):
        if not flip:
            continue
        y, x = obj['anchor']
        s = obj['scale']
        for yy, xx in info['union']:
            out[y+yy*s:y+(yy+1)*s, x+xx*s:x+(xx+1)*s] = info['background']
        _render(out, info['templates'][1-obj['state']], (y, x), s)
    return out


def _analyze(input_grid):
    g = _grid(input_grid)
    info = _infer(g)
    hits, covered, constant, reset, prefix = [], [], [], [], []
    for obj in info['occurrences']:
        rows = [_hits(src, obj['sensor'], info['active']) for src in info['sources']]
        hits.append([n for n, c in rows])
        covered.append(sum(c for n, c in rows))
        constant.append(sum(_hits(src, obj['sensor'], info['active'], first_slice=True)[0]
                            for src in info['sources']) % 2)
        reset.append(sum(_hits(src, obj['sensor'], info['active'], reset=True)[0]
                         for src in info['sources']) % 2)
        short_sources = []
        for src in info['sources']:
            word = src['word']
            p = next(p for p in range(1, len(word)+1)
                     if all(np.array_equal(word[i], word[i % p]) for i in range(len(word))))
            short_sources.append(dict(src, word=word[:p]))
        prefix.append(sum(_hits(src, obj['sensor'], info['active'])[0] for src in short_sources) % 2)
    info['hits'] = hits
    info['covered'] = covered
    info['flips'] = [sum(row) % 2 for row in hits]
    info['constant_flips'] = constant
    info['reset_flips'] = reset
    info['prefix_flips'] = prefix
    info['expected'] = _replace(g, info, info['flips'])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _analyze(input_grid)['expected']


def _design(rng, simple=False):
    h = 3 if simple else int(rng.integers(3, 5))
    w = 3 if simple or rng.random() < .65 else 4
    frame = {(y, x) for y in range(h) for x in range(w)
             if y in (0, h-1) or x in (0, w-1)}
    for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)):
        if rng.random() < .5:
            frame.remove(p)
    interior = [(y, x) for y in range(1, h-1) for x in range(1, w-1)]
    sensor = interior[int(rng.integers(len(interior)))]
    changing = [p for p in interior if p != sensor and rng.random() < .75]
    maps = []
    for state in (0, 1):
        m = {p: 2 for p in frame}
        m[sensor] = 3
        length = int(rng.integers(1, w+1))
        start = int(rng.integers(w-length+1))
        decoration = [( -1 if state == 0 else h, x) for x in range(start, start+length)]
        accent_count = int(rng.integers(1, 3))
        for j, p in enumerate(changing+decoration):
            m[p] = 6+2*state+j % accent_count
        maps.append(m)
    return maps, sensor


def _source(hub, direction, width, period):
    pts = [p for d in range(1, period+1) for p in _coords(hub, direction, width, d)]
    y, x, h, w = hub
    body = [(yy, xx) for yy in range(y, y+h) for xx in range(x, x+w)]
    return dict(hub=hub, direction=direction, width=width,
                word=np.zeros((period, width), dtype=np.int8), region=_box(body+pts))


def _extra_source(rng, shape, target, reserved):
    """Sample a boundary emitter whose swath intersects a chosen sensor."""
    y, x, s, _ = target['sensor']
    candidates = []
    for direction in range(4):
        for width in range(1, 4):
            for period in (2, 3, 4):
                offset = int(rng.integers(width))
                if direction < 2:
                    left = x+int(rng.integers(s))-offset-1
                    hub = (0 if direction == 1 else shape[0]-3, left, 3, width+2)
                else:
                    top = y+int(rng.integers(s))-offset-1
                    hub = (top, 0 if direction == 3 else shape[1]-3, width+2, 3)
                src = _source(hub, direction, width, period)
                yy, xx, hh, ww = src['region']
                if (0 <= yy and 0 <= xx and yy+hh <= shape[0] and xx+ww <= shape[1]
                        and all(_separate(src['region'], r) for r in reserved)):
                    candidates.append(src)
    return candidates[int(rng.integers(len(candidates)))] if candidates else None


def _layout(rng, templates, sensor, kind, simple=False):
    union = set(templates[0]) | set(templates[1])
    uy, ux, uh, uw = _box(union)
    shape = (30, 30) if simple else (int(rng.integers(24, 31)), int(rng.integers(27, 31)))
    period = 3 if simple or kind == 'single' else int(rng.integers(2, 6))
    width = 4 if simple else int(rng.integers(3, 7))
    band = 9 if simple else int(rng.integers(5, shape[0]-9))
    left = 0 if simple else int(rng.integers(2))
    body_w = 3 if simple else int(rng.integers(3, 5))
    src = _source((band-1, left, width+2, body_w), 3, width, period)
    sources, reserved = [src], [src['region']]
    scales = [1, 1, 3 if kind == 'test' else 2]
    if kind in ('single', 'cross') or rng.random() < .7:
        scales.append(1)
    if kind == 'free' and rng.random() < .3:
        scales.append(1)
    states = [0, 1]+[int(rng.integers(2)) for _ in scales[2:]]
    occurrences = [None]*len(scales)
    order = sorted(range(len(scales)), key=lambda i: scales[i], reverse=True)
    for i in order:
        s = scales[i]
        found = False
        for attempt in range(100):
            targeted = i < 3 or kind == 'single' or rng.random() < .55
            sy = int(rng.integers(band, band+width-s+1)) if targeted else int(rng.integers(shape[0]-s+1))
            top = sy-(sensor[0]-uy)*s
            minimum_x = left+body_w+period+1
            if minimum_x+uw*s > shape[1]:
                return None
            x = int(rng.integers(minimum_x, shape[1]-uw*s+1))
            region = (top, x, uh*s, uw*s)
            if top < 0 or top+uh*s > shape[0] or not all(_separate(region, r) for r in reserved):
                continue
            anchor = (top-uy*s, x-ux*s)
            occurrences[i] = dict(anchor=anchor, scale=s, state=states[i], region=region,
                                  sensor=(sy, anchor[1]+sensor[1]*s, s, s))
            reserved.append(region)
            found = True
            break
        if not found:
            return None
    count = 1 if kind == 'single' else 2 if kind == 'cross' else int(rng.integers(1, 4))
    for i in range(1, count):
        target = occurrences[0] if kind == 'cross' and i == 1 else occurrences[int(rng.integers(len(occurrences)))]
        extra = _extra_source(rng, shape, target, reserved)
        if extra is None:
            return None
        sources.append(extra)
        reserved.append(extra['region'])
    return dict(shape=shape, templates=templates, union=union, occurrences=occurrences, sources=sources)


def _qualifies(layout, hits, covered, kind):
    totals = [sum(row) for row in hits]
    odd = [n % 2 for n in totals]
    objects = layout['occurrences']
    if not any(odd) or all(odd):
        return False
    if kind == 'single':
        return (odd[0] == odd[1] == 1 and totals[2] >= 2 and not odd[2]
                and any(n == 0 and c > 0 for n, c in zip(totals, covered)))
    if kind == 'cross':
        return (not odd[0] and sum(n % 2 for n in hits[0]) >= 2 and odd[2] and totals[2] >= 3
                and any(flag and o['scale'] == 1 for flag, o in zip(odd, objects)))
    return True


def _paint_layout(rng, layout, palette, kind, fixed_bits=None):
    sources, objects = layout['sources'], layout['occurrences']
    # The hit counts are linear sums of tile-pixel indicators. Precompute the
    # coefficient matrix once; pattern attempts allocate no grids or files.
    matrices, exposure = [], np.zeros(len(objects), dtype=int)
    for src in sources:
        p, width = src['word'].shape
        matrix = np.zeros((len(objects), p*width), dtype=np.int8)
        for k in range(p*width):
            src['word'][:] = 4
            src['word'].flat[k] = 3
            for i, obj in enumerate(objects):
                n, c = _hits(src, obj['sensor'], 3)
                matrix[i, k] = n
                if k == 0:
                    exposure[i] += c
        matrices.append(matrix)
    for attempt in range(1 if fixed_bits is not None else 192):
        hit_columns = []
        for source_index, (src, matrix) in enumerate(zip(sources, matrices)):
            bits = (np.asarray(fixed_bits[source_index], dtype=bool).ravel().copy()
                    if fixed_bits is not None else
                    rng.random(src['word'].size) < float(rng.choice([.3, .5, .7])))
            if kind == 'single':
                bits = bits.reshape(src['word'].shape)
                bits[-1] = bits[0]
                if np.array_equal(bits[0], bits[1]):
                    break
                bits = bits.ravel()
            if not bits.any() or bits.all():
                break
            inactive = rng.choice([4, 5], size=src['word'].shape)
            if kind == 'single':
                inactive[-1] = inactive[0]
            src['word'] = np.where(bits.reshape(src['word'].shape), 3, inactive).astype(np.int8)
            hit_columns.append(matrix @ bits.astype(np.int8))
        if len(hit_columns) != len(sources):
            continue
        hits = np.stack(hit_columns, axis=1).tolist()
        if not _qualifies(layout, hits, exposure, kind):
            continue
        flips = [sum(row) % 2 for row in hits]
        # A phase witness is compulsory for the single-emitter demonstration.
        if kind == 'single':
            constant = [sum(_hits(src, o['sensor'], 3, first_slice=True)[0] for src in sources) % 2 for o in objects]
            reset = [sum(_hits(src, o['sensor'], 3, reset=True)[0] for src in sources) % 2 for o in objects]
            prefix = [_hits(dict(sources[0], word=sources[0]['word'][:2]), o['sensor'], 3)[0] % 2 for o in objects]
            if flips == constant or flips == reset or flips == prefix:
                continue
        inp = np.full(layout['shape'], 0, dtype=np.int8)
        out = inp.copy()
        for src in sources:
            y, x, h, w = src['hub']
            inp[y:y+h, x:x+w] = 1
            for d, values in enumerate(src['word'], 1):
                for pt, value in zip(_coords(src['hub'], src['direction'], src['width'], d), values):
                    inp[pt] = value
        out[:] = inp
        for obj, flip in zip(objects, flips):
            _render(inp, layout['templates'][obj['state']], obj['anchor'], obj['scale'])
            _render(out, layout['templates'][obj['state'] ^ flip], obj['anchor'], obj['scale'])
        rotation, mirror = int(rng.integers(4)), bool(rng.integers(2))
        pair = {}
        for name, grid in (('input', inp), ('output', out)):
            grid = np.rot90(palette[grid], rotation)
            pair[name] = np.ascontiguousarray(np.fliplr(grid) if mirror else grid)
        return pair
    return None


def _fallback(rng, kind, palette):
    templates = [{(0, 1): 2, (1, 0): 2, (1, 2): 2, (2, 1): 2,
                  (1, 1): 3, (-1 if state == 0 else 3, 1): 6+2*state}
                 for state in (0, 1)]
    objects = []
    placements = [(8, 8, 1, 0), (11, 12, 1, 1),
                  (7, 20 if kind != 'test' else 18, 2 if kind != 'test' else 3, 0)]
    if kind != 'test':
        placements.append((9, 16, 1, 1))
    for y, x, s, state in placements:
        objects.append(dict(anchor=(y, x), scale=s, state=state,
                            region=(y-s, x, 5*s, 3*s), sensor=(y+s, x+s, s, s)))
    sources = [_source((8, 0, 6, 3), 3, 4, 3)]
    if kind != 'single':
        sources.append(_source((25, 8, 3, 3), 0, 1, 2))
    layout = dict(shape=(30, 30), templates=templates, union=set.union(*(set(t) for t in templates)),
                  occurrences=objects, sources=sources)
    first = [[1, 0, 0, 0], [0, 1, 0, 1] if kind == 'single' else [1, 1, 0, 1],
             [1, 0, 0, 0]]
    bits = [first] if kind == 'single' else [first, [[0], [1]]]
    pair = _paint_layout(rng, layout, palette, kind, fixed_bits=bits)
    assert pair is not None, 'deterministic fallback invariant failed'
    return pair


def _make_pair(rng, kind):
    palette = rng.permutation(10).astype(np.int8)
    for attempt in range(64):
        templates, sensor = _design(rng, simple=attempt >= 40)
        layout = _layout(rng, templates, sensor, kind, simple=attempt >= 40)
        if layout is None:
            continue
        pair = _paint_layout(rng, layout, palette, kind)
        if pair is not None:
            a = _analyze(pair['input'])
            assert np.array_equal(a['expected'], pair['output'])
            return pair
    return _fallback(rng, kind, palette)


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = 'test' if split == 'test' else 'single' if i == 0 else 'cross' if i == 1 else 'free'
            pair = _make_pair(rng, kind)
            # All output colors occur in the two input reference templates.
            # At most thirteen exclusions exist; fourteen distinct injective
            # palette assignments therefore guarantee a free input signature.
            if _key(pair['input']) in used:
                colors = [int(c) for c in np.unique(pair['input'])]
                options = itertools.islice(itertools.permutations(range(10), len(colors)), len(used)+1)
                for values in options:
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[colors] = values
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    if _key(candidate['input']) not in used:
                        pair = candidate
                        break
                else:
                    raise AssertionError('finite palette repair invariant failed')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(info['expected'], out), 'incorrect parity or registered replacement'
            assert any(info['flips']) and not all(info['flips'])
            # Source material, all scaffold/sensor pixels, and everything
            # outside toggled union supports are checked by full-grid equality.
            records[split].append(info)
    train = records['train']
    assert any({o['state'] for o, f in zip(a['occurrences'], a['flips']) if f} == {0, 1} for a in train)
    assert any(o['scale'] >= 2 and f for a in train for o, f in zip(a['occurrences'], a['flips']))
    assert any(o['scale'] >= 2 and sum(row) >= 3 and sum(row) % 2
               for a in train for o, row in zip(a['occurrences'], a['hits']))
    assert any(o['scale'] >= 2 and sum(row) >= 2 and not sum(row) % 2 and len(row) == 1
               for a in train for o, row in zip(a['occurrences'], a['hits']))
    assert any(sum(n % 2 for n in row) >= 2 and sum(row) % 2 == 0
               for a in train for row in a['hits'])
    assert any(sum(row) == 0 and c > 0 for a in train for row, c in zip(a['hits'], a['covered']))
    assert any(a['constant_flips'] != a['flips'] for a in train)
    assert any(a['reset_flips'] != a['flips'] for a in train)
    assert any(a['prefix_flips'] != a['flips'] for a in train)
    assert max(o['scale'] for a in train for o in a['occurrences']) == 2
    assert all(max(o['scale'] for o in a['occurrences']) == 3 for a in records['test'])
    return records


def validate_generated_puzzle(puzzle):
    _inspect_puzzle(puzzle)


def family_features(pair):
    a = _analyze(pair['input'])
    return dict(shape=list(np.asarray(pair['input']).shape), background=a['background'],
                frame=list(a['frame']), objects=len(a['occurrences']), emitters=len(a['sources']),
                scales=sorted(o['scale'] for o in a['occurrences']),
                widths=sorted(s['width'] for s in a['sources']),
                lengths=sorted(len(s['word']) for s in a['sources']),
                directions=sorted(s['direction'] for s in a['sources']),
                hits=[list(row) for row in a['hits']], flips=list(a['flips']),
                state_supports=[len(t) for t in a['templates']],
                first_slice_diff=a['constant_flips'] != a['flips'],
                reset_phase_diff=a['reset_flips'] != a['flips'],
                short_prefix_diff=a['prefix_flips'] != a['flips'])


def self_test():
    """Stream 42 seeds; bounded counts, parity witnesses and mutation checks."""
    # Hand-counted probes distinguish one, two and three pulses without using
    # the sampled-output renderer or a generated label.
    probe = _source((4, 4, 4, 4), 3, 2, 3)
    probe['word'] = np.array([[3, 4], [4, 3], [3, 3]], dtype=np.int8)
    assert _hits(probe, (5, 8, 2, 2), 3) == (2, 4)
    assert _hits(probe, (5, 10, 2, 2), 3) == (3, 4)
    aba = _source((4, 4, 3, 3), 3, 1, 3)
    aba['word'] = np.array([[3], [4], [3]], dtype=np.int8)
    assert _hits(aba, (5, 10, 1, 1), 3) == (1, 1)
    assert _hits(dict(aba, word=aba['word'][:2]), (5, 10, 1, 1), 3) == (0, 1)
    # A valid member outside the sampling concentration: 5x5 frame, scale 4,
    # larger interior, and replacement touching the right canvas edge.
    frame = {(y, x) for y in range(5) for x in range(5) if y in (0, 4) or x in (0, 4)}
    templates = [{**{p: 2 for p in frame}, (1, 1): 3, (2, 2): 6+2*state,
                  (-1 if state == 0 else 5, 2): 7+2*state} for state in (0, 1)]
    member = np.zeros((30, 30), dtype=np.int8)
    member[24:27, 2:5] = 1
    member[21:24, 3] = [3, 4, 3]
    expected = member.copy()
    for anchor, scale, state, opposite in (((2, 2), 1, 0, 1), ((11, 2), 1, 1, 0), ((5, 10), 4, 0, 0)):
        _render(member, templates[state], anchor, scale)
        _render(expected, templates[opposite], anchor, scale)
    assert np.array_equal(solve(member), expected)
    signatures, shapes, frames, schedules, widths, periods, directions, backgrounds = (set() for _ in range(8))
    object_counts, source_counts = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                for role in ('input', 'output'):
                    digest.update(repr(_key(pair[role])[0]).encode())
                    digest.update(_key(pair[role])[1])
                f = family_features(pair)
                shapes.add(tuple(f['shape']))
                frames.add(tuple(tuple(p) for p in f['frame']))
                schedules.add(tuple(f['scales']))
                widths.update(f['widths'])
                periods.update(f['lengths'])
                directions.update(f['directions'])
                backgrounds.add(f['background'])
                object_counts.add(f['objects'])
                source_counts.add(f['emitters'])
                assert np.array_equal(solve(np.asarray(pair['input']).copy()), pair['output'])
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and len(shapes) >= 15 and len(frames) >= 10
    assert widths >= {1, 2, 3, 4, 5, 6} and periods >= {2, 3, 4, 5}
    assert directions == {0, 1, 2, 3} and len(backgrounds) == 10
    assert len(schedules) >= 4 and object_counts >= {3, 4, 5} and source_counts == {1, 2, 3}
    again = generate_puzzle(0)
    for split in ('train', 'test'):
        for p, q in zip(again[split], generate_puzzle(0)[split]):
            assert all(np.array_equal(p[k], q[k]) for k in ('input', 'output'))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for kind in ('single', 'cross', 'test', 'free'):
        p = _fallback(np.random.default_rng(790), kind, np.arange(10, dtype=np.int8))
        assert np.array_equal(solve(p['input']), p['output'])
    bad = {split: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in again[split]]
           for split in ('train', 'test')}
    g = bad['test'][0]['output']
    g[0, 0] = (int(g[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'frames': len(frames), 'scale_schedules': len(schedules),
            'receiver_counts': sorted(object_counts), 'emitter_counts': sorted(source_counts),
            'widths': sorted(widths), 'tile_lengths': sorted(periods), 'backgrounds': sorted(backgrounds)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
