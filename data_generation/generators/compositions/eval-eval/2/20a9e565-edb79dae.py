"""Growing mosaic stamps: visual substitution drives a spatial continuation.

Only NumPy and the standard library are used. Generation paints explicit latent
generations; solve parses the input and fits spatial models independently.
There is no file I/O, cache, hidden convention, or training-label access here.
"""
from __future__ import annotations

import hashlib
import numpy as np


_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _box(cells):
    p = np.asarray(cells)
    lo, hi = p.min(0), p.max(0) + 1
    return int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1])


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _marker_cells(bounds):
    r, e, c, d = bounds
    return {(r, c), (r, c+1), (r+1, c),
            (e-1, d-1), (e-1, d-2), (e-2, d-1)}


def _markers(a):
    found = []
    for raw in np.unique(a):
        cells = np.argwhere(a == raw)
        if len(cells) == 6:
            b = _box(cells)
            if min(b[1]-b[0], b[3]-b[2]) >= 3 and set(map(tuple, cells)) == _marker_cells(b):
                found.append((int(raw), b))
    if len(found) != 1:
        raise ValueError("need exactly two canonical, disconnected L markers")
    return found[0]


def _parse(a):
    a = _grid(a)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the canvas must have a blank background rim")
    marker, bounds = _markers(a)
    frames = []
    for raw in np.unique(a):
        if raw in (bg, marker):
            continue
        cells = np.argwhere(a == raw)
        r, e, c, d = _box(cells)
        if (min(e-r, d-c) >= 5 and len(cells) == 2*(e-r+d-c)-4
                and np.all(a[r, c:d] == raw) and np.all(a[e-1, c:d] == raw)
                and np.all(a[r:e, c] == raw) and np.all(a[r:e, d-1] == raw)):
            frames.append((int(raw), (r, e, c, d)))
    if len(frames) != 1:
        raise ValueError("need one exclusive rectangular gallery frame")
    frame, (r, e, c, d) = frames[0]
    br, be, bc, bd = bounds
    if not (r < br < be <= e-1 and c < bc < bd <= d-1):
        raise ValueError("target must be inside the gallery")
    if not np.all(np.isin(a[br:be, bc:bd], [bg, marker])):
        raise ValueError("the requested crop must be entirely erased")
    ext = a != bg
    ext[r:e, c:d] = False
    key_cells = np.zeros(a.shape, dtype=bool)
    mapping = {}
    for cells in _components(ext):
        vals, counts = np.unique([a[x, y] for x, y in cells], return_counts=True)
        if len(vals) == 1:
            continue
        rr, ee, cc, dd = _box(cells)
        if len(cells) != 12 or len(vals) != 2 or set(map(int, counts)) != {5, 7}:
            raise ValueError("unknown two-color legend key")
        source = int(vals[np.argmax(counts)])
        target = int(vals[np.argmin(counts)])
        tile = a[rr:ee, cc:dd]
        if not any(tile.shape == m.shape and np.array_equal(tile == source, m)
                   for m in (np.rot90(_ARROW, k) for k in range(4))):
            raise ValueError("malformed arrow key")
        if source in mapping:
            raise ValueError("duplicate key")
        mapping[source] = target
        key_cells[rr:ee, cc:dd] = True
    if not 2 <= len(mapping) <= 3 or not set(mapping.values()) <= set(mapping):
        raise ValueError("need a closed, nonidentity two- or three-source dictionary")
    remaining = ext & ~key_cells
    if set(map(int, np.unique(a[remaining]))) != set(mapping):
        raise ValueError("unexplained exterior content")
    masks, sizes = {}, set()
    for source in mapping:
        cells = np.argwhere(remaining & (a == source))
        rr, ee, cc, dd = _box(cells)
        n = ee-rr
        tile = a[rr:ee, cc:dd]
        if n not in (2, 3) or dd-cc != n or not np.all(np.isin(tile, [bg, source])):
            raise ValueError("each source needs one recoverable square stencil")
        mask = tile == source
        if not np.all(mask.any(0)) or not np.all(mask.any(1)):
            raise ValueError("stencil has an empty row or column")
        masks[source] = mask.copy()
        sizes.add(n)
    if len(sizes) != 1 or all(m.all() for m in masks.values()):
        raise ValueError("need one common scale and informative stencil holes")
    scene = a[r+1:e-1, c+1:d-1].copy()
    scene[scene == marker] = bg
    if not set(map(int, np.unique(scene))) <= {bg, *mapping}:
        raise ValueError("unknown gallery color")
    crop = (br-r-1, be-r-1, bc-c-1, bd-c-1)
    return dict(background=bg, marker=marker, frame=frame, bounds=bounds,
                frame_box=(r, e, c, d), scene=scene, crop=crop,
                masks=masks, mapping=mapping, n=sizes.pop())


def _expand(a, masks, mapping, bg, *, solid=False, double=False):
    """Input-side predictor: coordinate predicates, not generator stroke loops."""
    n = next(iter(masks.values())).shape[0]
    rr, cc = np.indices((a.shape[0]*n, a.shape[1]*n))
    parent = a[rr//n, cc//n]
    result = np.full(parent.shape, bg, dtype=np.int8)
    for s, mask in masks.items():
        on = (parent == s) & (True if solid else mask[rr % n, cc % n])
        color = mapping[mapping[s]] if double else mapping[s]
        result[on] = color
    return result


def _models(info):
    scene, bg, n = info['scene'], info['background'], info['n']
    cr, ce, cc, cd = info['crop']
    crop_box = ((cr, ce), (cc, cd))
    for axis in (0, 1):
        other = 1-axis
        occupied = np.flatnonzero(np.any(scene != bg, axis=other))
        if not len(occupied):
            continue
        groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1)+1)
        if len(groups) not in (3, 4):
            continue
        objects = []
        for g in groups:
            coords = np.argwhere(scene != bg)
            coords = coords[(coords[:, axis] >= g[0]) & (coords[:, axis] <= g[-1])]
            r, e, c, d = _box(coords)
            patch = scene[r:e, c:d]
            if not np.all(np.any(patch != bg, axis=0)) or not np.all(np.any(patch != bg, axis=1)):
                break
            objects.append(((r, c), patch))
        if len(objects) != len(groups):
            continue
        for sign in (1, -1):
            ordered = objects[::sign]
            starts = [p[0][axis] for p in ordered]
            widths = [p[1].shape[axis] for p in ordered]
            leads = [s if sign == 1 else s+w-1 for s, w in zip(starts, widths)]
            gaps = [sign*(leads[i+1]-leads[i])-widths[i] for i in range(len(ordered)-1)]
            if min(gaps) < 1 or len(set(gaps)) != 1:
                continue
            if any(b[1].shape != tuple(n*d for d in a[1].shape)
                   or not np.array_equal(_expand(a[1], info['masks'], info['mapping'], bg), b[1])
                   for a, b in zip(ordered, ordered[1:])):
                continue
            gap = gaps[0]
            sizes = [p[1].shape[other] for p in ordered]
            # Both center roundings are visible possibilities on a pixel grid.
            # Reflection exchanges them when successive sizes change parity.
            for align, rounding in ((0, 0), (1, 0), (1, 1), (2, 0)):
                anchors = [p[0][other] + (align*(size-1)+rounding)//2
                           for p, size in zip(ordered, sizes)]
                drift = anchors[1]-anchors[0]
                if any(v != anchors[0]+i*drift for i, v in enumerate(anchors)):
                    continue
                patch = ordered[-1][1]
                lead = leads[-1]
                for step in (1, 2):
                    lead += sign*(patch.shape[axis]+gap)
                    patch = _expand(patch, info['masks'], info['mapping'], bg)
                    lo = lead if sign == 1 else lead-patch.shape[axis]+1
                    cross = anchors[0]+(len(ordered)-1+step)*drift-(align*(patch.shape[other]-1)+rounding)//2
                    origin = (lo, cross) if axis == 0 else (cross, lo)
                    if all(origin[j] <= crop_box[j][0] < crop_box[j][1] <= origin[j]+patch.shape[j]
                           for j in (0, 1)):
                        # The requested stage is strictly beyond the evidence.
                        if sign == 1 and crop_box[axis][0] <= max(starts[i]+widths[i]-1 for i in range(len(starts))):
                            continue
                        if sign == -1 and crop_box[axis][1] > min(starts):
                            continue
                        pred = patch[cr-origin[0]:ce-origin[0], cc-origin[1]:cd-origin[1]].copy()
                        yield pred, dict(axis=axis, sign=sign, align=align, rounding=rounding, drift=drift,
                                         gap=gap, step=step, origin=origin, visible=len(ordered),
                                         last=ordered[-1][1], target=patch,
                                         base_shape=ordered[0][1].shape)


def _analyze(a):
    info = _parse(a)
    models = list(_models(info))
    if not models:
        raise ValueError("visible generations do not establish a supported continuation")
    prediction = models[0][0]
    if any(not np.array_equal(prediction, p) for p, _ in models):
        raise ValueError("competing spatial models disagree about the crop")
    info.update(models[0][1])
    info['candidate_models'] = len(models)
    return prediction, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from one input, without demonstrations or latent state."""
    return _analyze(input_grid)[0]


def _paint_next(a, masks, mapping, bg):
    """Generation ground truth: explicit local replacement into fresh squares."""
    n = next(iter(masks.values())).shape[0]
    out = np.full((a.shape[0]*n, a.shape[1]*n), bg, dtype=np.int8)
    for r, c in np.argwhere(a != bg):
        s = int(a[r, c])
        block = out[r*n:(r+1)*n, c*n:(c+1)*n]
        for y, x in np.argwhere(masks[s]):
            block[y, x] = mapping[s]
    return out


def _mask(rng, n):
    y, x = np.indices((n, n))
    kind = int(rng.integers(7))
    if kind == 0:
        a = x == y
    elif kind == 1:
        a = (x == 0) | (x == n-1) | (y == 0) | (y == n-1)
    elif kind == 2:
        a = (x == n//2) | (y == n//2)
    elif kind == 3:
        a = (x+y) % 2 == 0
    else:
        a = rng.random((n, n)) < rng.uniform(.35, .85)
        for row in range(n):
            if not a[row].any():
                a[row, int(rng.integers(n))] = True
        for col in range(n):
            if not a[:, col].any():
                a[int(rng.integers(n)), col] = True
    return np.rot90(a, int(rng.integers(4))).copy()


def _crop_ablations(info):
    cr, ce, cc, cd = info['crop']
    r, c = info['origin']
    last, bg = info['last'], info['background']
    yy, xx = np.indices((ce-cr, cd-cc))
    frozen = last[(yy+cr-r) % last.shape[0], (xx+cc-c) % last.shape[1]]
    solid, double = last.copy(), last.copy()
    for _ in range(info['step']):
        solid = _expand(solid, info['masks'], info['mapping'], bg, solid=True)
        double = _expand(double, info['masks'], info['mapping'], bg, double=True)
    slices = (slice(cr-r, ce-r), slice(cc-c, cd-c))
    return frozen, solid[slices], double[slices]


def _transform(pair, k, mirror):
    """Transform all content, then redraw the fixed global L convention."""
    a, out = pair['input'].copy(), pair['output'].copy()
    marker, b = _markers(a)
    bg = int(a[0, 0])
    window = np.zeros(a.shape, dtype=bool)
    window[b[0]:b[1], b[2]:b[3]] = True
    a[a == marker] = bg
    if mirror:
        a, window, out = a[:, ::-1], window[:, ::-1], out[:, ::-1]
    a, window, out = np.rot90(a, k).copy(), np.rot90(window, k), np.rot90(out, k).copy()
    bounds = _box(np.argwhere(window))
    for p in _marker_cells(bounds):
        a[p] = marker
    return {'input': a, 'output': out}


def _make_pair(rng, n, skip, rotation, first=False):
    colors = list(map(int, rng.permutation(10)))
    bg, frame, marker = colors[:3]
    k = int(rng.integers(2, 4))
    sources = colors[3:3+k]
    order = list(map(int, rng.permutation(sources)))
    mapping = {s: order[(i+1) % k] for i, s in enumerate(order)}
    if k == 3 and rng.random() < .35:
        mapping = {order[0]: order[1], order[1]: order[2], order[2]: order[1]}
    masks = {s: _mask(rng, n) for s in sources}
    if all(m.all() for m in masks.values()):
        masks[sources[0]][0, -1] = False
    if first:
        # A rotation-informative, non-solid stencil exposes displayed orientation.
        masks[sources[0]] = np.ones((n, n), dtype=bool)
        masks[sources[0]][0, -1] = False
        if n == 3:
            masks[sources[0]][1, 0] = False
    visible = 3 if n == 3 or skip else int(rng.choice([3, 3, 4]))
    seed_w = 1 if n == 3 or skip or visible == 4 else int(rng.integers(1, 3))
    seed_h = int(rng.integers(1, (3 if n == 2 else 2)+1))
    seed = np.full((seed_h, seed_w), order[0], dtype=np.int8)
    if seed.size > 1 and rng.random() < .55:
        seed = rng.choice(sources, size=seed.shape).astype(np.int8)
        if min(seed.shape) >= 2 and rng.random() < .35:
            seed[0, -1] = bg
    target_index = visible+skip
    stages = [seed]
    for _ in range(target_index):
        stages.append(_paint_next(stages[-1], masks, mapping, bg))
    gap = int(rng.integers(1, 3))
    xs = [int(rng.integers(2))]
    for stage in stages[:-1]:
        xs.append(xs[-1]+stage.shape[1]+gap)
    if xs[-1]+3 > 26:
        return None
    banks = bool(rng.integers(2))
    legend_h = n+4 if banks else 3
    legend_w = max(k*(n+1)-1, k*5-1) if banks else k*(n+6)-1
    max_h = 25-legend_h
    align, drift = int(rng.integers(3)), int(rng.integers(-1, 2))
    ys = [i*drift-align*(a.shape[0]-1)//2 for i, a in enumerate(stages)]
    lo = min(ys[:visible])
    hi = max(ys[i]+stages[i].shape[0] for i in range(visible))
    if hi-lo > max_h:
        return None
    ih = int(rng.integers(max(hi-lo, 3), max_h+1))
    offset = -lo+int(rng.integers(ih-(hi-lo)+1))
    ys = [y+offset for y in ys]
    iw = int(rng.integers(xs[-1]+3, 27))
    target = stages[-1]
    yr, ye = max(0, ys[-1]), min(ih, ys[-1]+target.shape[0])
    xr, xe = xs[-1], min(iw, xs[-1]+target.shape[1])
    if min(ye-yr, xe-xr) < 3:
        return None
    # At most 64 proposals; failed attempts retain no diagnostics or arrays.
    chosen = None
    for _ in range(64):
        h, w = int(rng.integers(3, ye-yr+1)), int(rng.integers(3, xe-xr+1))
        r0, c0 = int(rng.integers(yr, ye-h+1)), int(rng.integers(xr, xe-w+1))
        crop = target[r0-ys[-1]:r0-ys[-1]+h, c0-xs[-1]:c0-xs[-1]+w].copy()
        if not np.any(crop == bg) or not np.any(crop != bg):
            continue
        info = dict(crop=(r0, r0+h, c0, c0+w), origin=(ys[-1], xs[-1]),
                    last=stages[visible-1], step=skip+1, masks=masks, mapping=mapping, background=bg)
        frozen, solid, double = _crop_ablations(info)
        if (np.array_equal(crop != bg, frozen != bg) or np.array_equal(crop, solid)
                or np.array_equal(crop, double)):
            continue
        if not any(crop[y, x] != bg for y, x in _marker_cells((0, h, 0, w))):
            continue
        chosen = (r0, c0, crop)
        break
    if chosen is None:
        return None
    r0, c0, output = chosen
    fw, fh = iw+2, ih+2
    width = max(fw, legend_w)+2
    fr, fc = legend_h+2, int(rng.integers(1, width-fw))
    a = np.full((fr+fh+1, width), bg, dtype=np.int8)
    a[fr, fc:fc+fw] = a[fr+fh-1, fc:fc+fw] = frame
    a[fr:fr+fh, fc] = a[fr:fr+fh, fc+fw-1] = frame
    st_order, key_order = list(rng.permutation(sources)), list(rng.permutation(sources))
    if banks:
        sl = int(rng.integers(1, width-(k*(n+1)-1)))
        kl = int(rng.integers(1, width-(k*5-1)))
        for j, s in enumerate(st_order):
            a[1:1+n, sl+j*(n+1):sl+j*(n+1)+n][masks[int(s)]] = s
        for j, s in enumerate(key_order):
            a[n+2:n+5, kl+j*5:kl+j*5+4] = np.where(_ARROW, s, mapping[int(s)])
    else:
        left = int(rng.integers(1, width-legend_w))
        for j, raw in enumerate(st_order):
            s = int(raw)
            x = left+j*(n+6)
            a[1:1+n, x:x+n][masks[s]] = s
            a[1:4, x+n+1:x+n+5] = np.where(_ARROW, s, mapping[s])
    for i in range(visible):
        patch = stages[i]
        a[fr+1+ys[i]:fr+1+ys[i]+patch.shape[0], fc+1+xs[i]:fc+1+xs[i]+patch.shape[1]] = patch
    bounds = (fr+1+r0, fr+1+r0+output.shape[0], fc+1+c0, fc+1+c0+output.shape[1])
    for p in _marker_cells(bounds):
        a[p] = marker
    pair = _transform({'input': a, 'output': output}, rotation, bool(rng.integers(2)))
    oh, ow = pair['output'].shape
    if not any(pair['output'][y, x] != bg for y, x in _marker_cells((0, oh, 0, ow))):
        return None
    try:
        prediction, _ = _analyze(pair['input'])
    except ValueError:
        return None
    if not np.array_equal(prediction, pair['output']):
        return None
    return pair


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name+' is outside the legal integer range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['distance', 'scale', 'axis']))
    base_n, axis = int(rng.choice([2, 2, 3])), int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            n = (3 if split == 'test' else 2) if shift == 'scale' else (2 if shift == 'distance' else base_n)
            skip = int(split == 'test' and shift == 'distance')
            ax = 1-axis if split == 'test' and shift == 'axis' else axis
            for _ in range(256):
                rotation = ax+2*int(rng.integers(2))
                pair = _make_pair(rng, n, skip, rotation, first=(split == 'train' and index == 0))
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('bounded distinct-pair construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError('puzzle keys must be train and test')
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError('splits must be lists')
    _count(len(puzzle['train']), 2, 10, 'num_train')
    _count(len(puzzle['test']), 1, 4, 'num_test')
    if len(puzzle['train']) <= len(puzzle['test']):
        raise ValueError('num_train must exceed num_test')
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError('pair keys must be input and output')
            a, out = _grid(pair['input']), _grid(pair['output'])
            if _key(a) in seen:
                raise ValueError('duplicate input')
            seen.add(_key(a))
            pred, info = _analyze(a)
            if not np.array_equal(out, pred):
                raise ValueError('wrong continuation label')
            bg = info['background']
            if not np.any(out == bg) or not np.any(out != bg):
                raise ValueError('uninformative crop')
            if info['marker'] in out or info['frame'] in out:
                raise ValueError('annotations leaked into answer')
            h, w = out.shape
            if not any(out[y, x] != bg for y, x in _marker_cells((0, h, 0, w))):
                raise ValueError('markers must overwrite some latent content')
            frozen, solid, double = _crop_ablations(info)
            if np.array_equal(out != bg, frozen != bg):
                raise ValueError('crop does not distinguish continued growth from static tiling')
            if np.array_equal(out, solid) or np.array_equal(out, double):
                raise ValueError('crop does not expose stencil geometry and one-step mapping')
            records[split].append(info)
    # These are generation coverage promises, not extra rules used by solve.
    tr, te = records['train'], records['test']
    distance_shift = all(i['step'] == 1 for i in tr) and all(i['step'] == 2 for i in te)
    scale_shift = all(i['n'] == 2 for i in tr) and all(i['n'] == 3 for i in te)
    axis_shift = len({i['axis'] for i in tr}) == len({i['axis'] for i in te}) == 1 and tr[0]['axis'] != te[0]['axis']
    if not (distance_shift or scale_shift or axis_shift):
        raise ValueError('missing the promised train/test shift')


def self_test():
    """42 streamed seeds; bounded RAM and no disk output. Returns small metrics."""
    metrics = {k: set() for k in ('puzzles', 'n', 'directions', 'alignments', 'drifts',
                                'steps', 'categories', 'shapes', 'masks', 'backgrounds',
                                'visible', 'seed_shapes', 'color_structures')}
    disconnected = multicolor = shared = solid_masks = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a, out = pair['input'], pair['output']
                before = a.copy()
                pred, info = _analyze(a)
                assert np.array_equal(pred, out) and np.array_equal(a, before)
                digest.update(repr(a.shape).encode()+a.tobytes()+out.tobytes())
                metrics['n'].add(info['n'])
                metrics['directions'].add((info['axis'], info['sign']))
                metrics['alignments'].add(info['align'])
                metrics['drifts'].add(info['drift'])
                metrics['steps'].add(info['step'])
                metrics['categories'].add(len(info['mapping']))
                metrics['shapes'].add((a.shape, out.shape))
                metrics['backgrounds'].add(info['background'])
                metrics['visible'].add(info['visible'])
                metrics['seed_shapes'].add(info['base_shape'])
                multicolor += len(set(map(int, np.unique(info['last']))) - {info['background']}) > 1
                keys = [_key(mask) for mask in info['masks'].values()]
                shared += len(set(keys)) < len(keys)
                for mask in info['masks'].values():
                    metrics['masks'].add(_key(mask))
                    disconnected += len(_components(mask)) > 1
                    solid_masks += mask.all()
                roles = {}
                normalized = bytes(roles.setdefault(int(x), len(roles)) for x in out.flat)
                metrics['color_structures'].add((out.shape, normalized))
        metrics['puzzles'].add(digest.digest())
        if seed % 7 == 0:
            pair = p['test'][0]
            perm = np.random.default_rng(seed+1000).permutation(10)
            assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
            for turn in (1, 2, 3):
                q = _transform(pair, turn, seed % 2 == 0)
                assert np.array_equal(solve(q['input']), q['output'])
            again = generate_puzzle(seed)
            assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test')
                       for x, y in zip(p[s], again[s]) for k in ('input', 'output'))
    assert len(metrics['puzzles']) == 42
    assert metrics['n'] == {2, 3} and len(metrics['directions']) == 4
    assert metrics['steps'] == {1, 2} and metrics['categories'] == {2, 3}
    assert metrics['visible'] == {3, 4} and len(metrics['alignments']) == 3
    assert len(metrics['drifts']) >= 3 and len(metrics['backgrounds']) == 10
    assert len(metrics['shapes']) >= 80 and len(metrics['masks']) >= 25
    assert len(metrics['seed_shapes']) >= 6 and len(metrics['color_structures']) >= 100
    assert disconnected >= 30 and multicolor >= 20 and shared >= 10 and solid_masks >= 10
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(6000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(999, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    result = {k: len(v) for k, v in metrics.items()}
    result.update(disconnected_stencils=int(disconnected), multicolor_generations=int(multicolor),
                  shared_geometry_pairs=int(shared), solid_stencils=int(solid_masks))
    return result
