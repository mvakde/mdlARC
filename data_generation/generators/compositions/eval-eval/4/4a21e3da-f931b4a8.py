"""Guide-cut stencils printed over material recycled from the excluded quadrant.

Only NumPy and the Python standard library are used. All solver conventions are
visible in the input; random choices affect sampling, never interpretation.
"""
from __future__ import annotations

import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(n < 1 or n > 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int64, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(x), len(roles)) for x in a.flat)


def _crop(a, bg):
    locations = np.argwhere(a != bg)
    if not len(locations):
        return np.full((1, 1), bg, dtype=np.int64)
    lo, hi = locations.min(axis=0), locations.max(axis=0) + 1
    return a[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _rooms(shape, ar, ac):
    h, w = shape
    return {(v, s): (slice(0, ar) if v == 0 else slice(ar+1, h),
                     slice(0, ac) if s == 0 else slice(ac+1, w))
            for v in (0, 1) for s in (0, 1)}


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    if min(h, w) < 7:
        raise ValueError("family canvases have dimensions at least 7")
    bg = int(g[0, 0])
    if any(int(g[p]) != bg for p in ((0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError("all input corners must identify the background")
    rim = np.zeros(g.shape, dtype=bool)
    rim[[0, -1], :] = True
    rim[:, [0, -1]] = True
    marks = np.argwhere(rim & (g != bg))
    if len(marks) != 2 or len(np.unique(g[rim & (g != bg)])) != 1:
        raise ValueError("expected two same-colored adjacent-edge markers")
    marker = int(g[tuple(marks[0])])
    if np.count_nonzero(g == marker) != 2:
        raise ValueError("marker color is reserved for the two border cells")
    ar = ac = vertical_side = horizontal_side = None
    for r, c in marks:
        r, c = int(r), int(c)
        if r in (0, h-1) and 0 < c < w-1:
            if ac is not None:
                raise ValueError("markers must be on perpendicular sides")
            ac, vertical_side = c, int(r != 0)
        elif c in (0, w-1) and 0 < r < h-1:
            if ar is not None:
                raise ValueError("markers must be on perpendicular sides")
            ar, horizontal_side = r, int(c != 0)
        else:
            raise ValueError("marker at a corner")
    if ar is None or ac is None:
        raise ValueError("one marker of each axis direction is required")
    source = (1-vertical_side, 1-horizontal_side)
    rooms = _rooms(g.shape, ar, ac)
    carrier = _crop(g[rooms[source]], bg)
    if np.any(carrier == bg):
        raise ValueError("source quadrant must contain a full rectangular swatch")
    destinations = [q for q in rooms if q != source]
    stencils = {q: _crop(g[rooms[q]], bg) for q in destinations}
    if not any(np.any(s != bg) for s in stencils.values()):
        raise ValueError("at least one stencil is required")
    vh = not np.all(carrier == carrier[:1, :])
    vw = not np.all(carrier == carrier[:, :1])
    for q in destinations:
        H, W = g[rooms[q]].shape
        if ((vh and (H < carrier.shape[0] or H % carrier.shape[0]))
                or (vw and (W < carrier.shape[1] or W % carrier.shape[1]))):
            raise ValueError("varying carrier axes require integral replication")
    axial = {(int(r), int(c)): int(g[r, c]) for r, c in np.argwhere(
        ((np.indices(g.shape)[0] == ar) | (np.indices(g.shape)[1] == ac))
        & (g != bg) & (g != marker))}
    vertical_hits = sorted(r for r, c in axial if c == ac)
    horizontal_hits = sorted(c for r, c in axial if r == ar)
    if not vertical_hits or not horizontal_hits:
        raise ValueError("each guide must hit interior material")
    return dict(grid=g, background=bg, marker=marker, ar=ar, ac=ac,
                vertical_side=vertical_side, horizontal_side=horizontal_side,
                source=source, rooms=rooms, destinations=destinations,
                carrier=carrier, stencils=stencils, axial=axial,
                vertical_hits=vertical_hits, horizontal_hits=horizontal_hits,
                carrier_axes=(vh, vw))


def _resize(a, H, W, tile_rows=False, tile_columns=False):
    h, w = a.shape
    rr = np.arange(H) % h if tile_rows else np.arange(H)*h//H
    cc = np.arange(W) % w if tile_columns else np.arange(W)*w//W
    return a[rr[:, None], cc[None, :]].copy()


def _tile(s, H, W, corner, top_rows=False, top_columns=False):
    h, w = s.shape
    rr = (np.arange(H) - (H if corner[0] and not top_rows else 0)) % h
    cc = (np.arange(W) - (W if corner[1] and not top_columns else 0)) % w
    return s[rr[:, None], cc[None, :]]


def _guide_cells(info, variant=None):
    h, w = info['grid'].shape
    ar, ac = info['ar'], info['ac']
    paths = set()
    for axis, hits, side, extent in (
            (0, info['vertical_hits'], info['vertical_side'], h),
            (1, info['horizontal_hits'], info['horizontal_side'], w)):
        if variant == 'full_guides':
            values = range(extent)
        else:
            endpoint = ((min(hits) if side == 0 else max(hits))
                        if variant == 'nearest_guides'
                        else (max(hits) if side == 0 else min(hits)))
            values = range(endpoint+1) if side == 0 else range(endpoint, extent)
        paths.update((x, ac) if axis == 0 else (ar, x) for x in values)
    return paths


def _render(info, variant=None):
    """Rendering plus explicit simpler alternatives used as teaching checks."""
    g, bg, carrier = info['grid'], info['background'], info['carrier']
    if variant == 'largest_rectangle':
        choices = [(q, _crop(g[room], bg)) for q, room in info['rooms'].items()]
        choices = [(q, a) for q, a in choices if np.all(a != bg)]
        _, carrier = max(choices, key=lambda item: (item[1].size, item[0]))
    out = np.full(g.shape, bg, dtype=np.int64)
    combined = None
    if variant == 'combined_stencil':
        combined = np.full_like(g, bg)
        for q in info['destinations']:
            combined[info['rooms'][q]] = g[info['rooms'][q]]
        combined = _crop(combined, bg)
    targets = list(info['destinations'])
    if variant == 'fill_source':
        targets.append(info['source'])
    for q in targets:
        region = out[info['rooms'][q]]
        H, W = region.shape
        base = _resize(carrier, H, W, variant == 'carrier_tile_rows',
                       variant == 'carrier_tile_columns')
        if q == info['source']:
            region[:] = base
            continue
        s = combined if combined is not None else info['stencils'][q]
        if variant == 'flip_rows' and q[0]:
            s = s[::-1, :]
        if variant == 'flip_columns' and q[1]:
            s = s[:, ::-1]
        tiled = _tile(s, H, W, q, variant == 'top_phase_rows',
                      variant == 'top_phase_columns')
        if variant == 'translate_once':
            region[:] = bg
            sh, sw = s.shape
            r0, c0 = (H-sh if q[0] else 0), (W-sw if q[1] else 0)
            patch = region[r0:r0+sh, c0:c0+sw]
            patch[s != bg] = s[s != bg]
        elif variant == 'separate_components':
            region[:] = base
            for cells in _components(s != bg):
                part = np.full_like(s, bg)
                for p in cells:
                    part[p] = s[p]
                repeated = _tile(_crop(part, bg), H, W, q)
                region[repeated != bg] = repeated[repeated != bg]
        elif variant == 'ignore_stencil':
            region[:] = base
        elif variant == 'stencil_only':
            region[:] = np.where(tiled != bg, tiled, bg)
        elif variant == 'opaque_background':
            region[:] = tiled
        elif variant == 'stencil_resize':
            scaled = _resize(s, H, W)
            region[:] = np.where(scaled != bg, scaled, base)
        elif variant == 'overlay_before_resize':
            local = _tile(s, *carrier.shape, q)
            fused = np.where(local != bg, local, carrier)
            region[:] = _resize(fused, H, W)
        else:
            region[:] = np.where(tiled != bg, tiled, base)
    if variant == 'keep_source':
        out[info['rooms'][info['source']]] = g[info['rooms'][info['source']]]
    for p in _guide_cells(info, variant):
        out[p] = info['marker']
    for p, ink in info['axial'].items():
        out[p] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete output from the input alone, without learned state."""
    return _render(_decode(input_grid))


_MUTANTS = (
    'largest_rectangle', 'combined_stencil', 'separate_components', 'translate_once', 'keep_source',
    'fill_source', 'stencil_only', 'ignore_stencil', 'carrier_tile_rows',
    'carrier_tile_columns', 'stencil_resize', 'overlay_before_resize',
    'opaque_background', 'top_phase_rows', 'top_phase_columns',
    'flip_rows', 'flip_columns', 'nearest_guides', 'full_guides',
)
_REQUIRED = set(_MUTANTS) | {'visible_carrier', 'visible_overwrite',
                            'partial_tile', 'axial_payload'}


def _evidence(pair, info=None):
    info = _decode(pair['input']) if info is None else info
    expected = np.asarray(pair['output'])
    evidence = {name for name in _MUTANTS
                if not np.array_equal(expected, _render(info, name))}
    bg = info['background']
    for q, s in info['stencils'].items():
        H, W = info['grid'][info['rooms'][q]].shape
        tiled, base = _tile(s, H, W, q), _resize(info['carrier'], H, W)
        if np.any(tiled == bg):
            evidence.add('visible_carrier')
        if np.any((tiled != bg) & (tiled != base)):
            evidence.add('visible_overwrite')
        if np.any(s != bg) and (H % s.shape[0] or W % s.shape[1]):
            evidence.add('partial_tile')
    if info['axial']:
        evidence.add('axial_payload')
    return evidence


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _roles(rng):
    colors = rng.permutation(10)
    return int(colors[0]), int(colors[1]), colors[2:2+int(rng.integers(3, 9))]


def _carrier(rng, palette):
    h, w = (int(x) for x in rng.integers(2, 6, size=2))
    kind = str(rng.choice(['solid', 'rows', 'columns', 'checker', 'blocks',
                          'motif', 'mosaic', 'diagonal'],
                         p=[.12, .13, .13, .14, .14, .12, .13, .09]))
    k = 1 if kind == 'solid' else int(rng.integers(2, min(4, len(palette))+1))
    inks = rng.choice(palette, k, replace=False)
    rr, cc = np.indices((h, w))
    if kind == 'solid':
        labels = np.zeros((h, w), dtype=int)
    elif kind == 'rows':
        labels = np.repeat(rng.integers(k, size=(h, 1)), w, axis=1)
    elif kind == 'columns':
        labels = np.repeat(rng.integers(k, size=(1, w)), h, axis=0)
    elif kind == 'checker':
        labels = (rr//int(rng.integers(1, h)) + cc//int(rng.integers(1, w))) % k
    elif kind == 'diagonal':
        labels = (rr + int(rng.integers(1, k+1))*cc) % k
    elif kind == 'blocks':
        row_cut, col_cut = int(rng.integers(1, h)), int(rng.integers(1, w))
        colors = rng.integers(k, size=(2, 2))
        labels = colors[(rr >= row_cut).astype(int), (cc >= col_cut).astype(int)]
    elif kind == 'motif':
        labels = np.zeros((h, w), dtype=int)
        for ink in range(1, k):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            labels[abs(rr-r)+abs(cc-c) <= int(rng.integers(1, max(h, w)))] = ink
    else:
        labels = rng.integers(k, size=(h, w))
    a = inks[labels]
    if rng.random() < .5:
        a = a[::-1]
    if rng.random() < .5:
        a = a[:, ::-1]
    return a.copy()


def _sizes(rng, length, varies, source_side):
    # The source side has space for the swatch plus its exterior margin.
    choices = [(a, b) for a in range(3, 27) for b in range(3, 27)
               if a+b <= 29 and (a, b)[source_side] >= length+1
               and (not varies or (a % length == 0 and b % length == 0))]
    if rng.random() < .65:
        small = [p for p in choices if max(p) <= 13]
        if small:
            choices = small
    return choices[int(rng.integers(len(choices)))]


def _grow_mask(rng, h, w, count):
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < count:
        frontier = sorted({q for r, c in cells for q in
                           ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for p in cells:
        mask[p] = True
    return mask


def _stencil(rng, hmax, wmax, bg, palette, complexity):
    h = int(rng.integers(1, min(4+complexity, hmax)+1))
    w = int(rng.integers(1, min(4+complexity, wmax)+1))
    a = np.full((h, w), bg, dtype=np.int64)
    if rng.random() < .14:
        return a
    rr, cc = np.indices((h, w))
    count = int(rng.integers(1, h*w+1))
    kind = int(rng.integers(7))
    if kind == 0:
        mask = np.zeros((h, w), dtype=bool)
        mask.flat[rng.choice(h*w, count, replace=False)] = True
    elif kind == 1:
        mask = _grow_mask(rng, h, w, count)
    elif kind == 2:
        mask = np.minimum.reduce([rr, cc, h-1-rr, w-1-cc]) < int(rng.integers(1, 3))
    elif kind == 3:
        mask = (rr+int(rng.integers(3))) % int(rng.integers(2, 4)) == 0
    elif kind == 4:
        mask = (rr//int(rng.integers(1, 3)) + cc//int(rng.integers(1, 3))) % 2 == 0
    elif kind == 5:
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask |= abs(rr-r)+abs(cc-c) <= int(rng.integers(0, 3))
    else:
        mask = rng.random((h, w)) > float(rng.uniform(.15, .85))
    if not mask.any():
        mask.flat[int(rng.integers(h*w))] = True
    if rng.random() < .5:
        mask = mask.T.copy()
        if mask.shape[0] > hmax or mask.shape[1] > wmax:
            mask = mask.T.copy()
        a = np.full(mask.shape, bg, dtype=np.int64)
    count = int(mask.sum())
    k = 1 if rng.random() < .55 else int(rng.integers(1, min(4, count, len(palette))+1))
    inks = rng.choice(palette, k, replace=False)
    values = np.concatenate((inks, rng.choice(inks, count-k,
                                            p=rng.dirichlet(np.full(k, .8)))))
    a[mask] = rng.permutation(values)
    return a


def _put(rng, g, room, material, bg):
    rs, cs = room
    rlo, rhi = max(rs.start, 1), min(rs.stop, g.shape[0]-1)
    clo, chi = max(cs.start, 1), min(cs.stop, g.shape[1]-1)
    h, w = material.shape
    r, c = int(rng.integers(rlo, rhi-h+1)), int(rng.integers(clo, chi-w+1))
    patch = g[r:r+h, c:c+w]
    patch[material != bg] = material[material != bg]


def _orientation(k):
    # Clockwise edge pairs: top+right, right+bottom, bottom+left, left+top.
    return ((0, 1), (1, 1), (1, 0), (0, 0))[k]


def _sample_pair(rng, orientation, params):
    bg, marker, palette = _roles(rng) if params['varied_palette'] else params['roles']
    carrier = _carrier(rng, palette)
    vside, hside = _orientation(orientation)
    source = (1-vside, 1-hside)
    vh, vw = (not np.all(carrier == carrier[:1, :]),
              not np.all(carrier == carrier[:, :1]))
    top, bottom = _sizes(rng, carrier.shape[0], vh, source[0])
    left, right = _sizes(rng, carrier.shape[1], vw, source[1])
    g = np.full((top+bottom+1, left+right+1), bg, dtype=np.int64)
    rooms = _rooms(g.shape, top, left)
    _put(rng, g, rooms[source], carrier, bg)
    nonempty = False
    for q, room in rooms.items():
        if q == source:
            continue
        rs, cs = room
        ha = min(rs.stop, g.shape[0]-1)-max(rs.start, 1)
        wa = min(cs.stop, g.shape[1]-1)-max(cs.start, 1)
        s = _stencil(rng, ha, wa, bg, palette, params['complexity'])
        _put(rng, g, room, s, bg)
        nonempty |= bool(np.any(s != bg))
    if not nonempty:
        q = next(q for q in rooms if q != source)
        _put(rng, g, rooms[q], np.array([[palette[0]]]), bg)
    h, w = g.shape
    for axis, extent in ((0, h), (1, w)):
        count = int(rng.integers(1, min(7, extent-2)+1))
        for x in rng.choice(np.arange(1, extent-1), count, replace=False):
            p = (int(x), left) if axis == 0 else (top, int(x))
            g[p] = int(rng.choice(palette))
    if rng.random() < .4:
        g[top, left] = int(rng.choice(palette))
    g[0 if vside == 0 else h-1, left] = marker
    g[top, 0 if hside == 0 else w-1] = marker
    return dict(input=g, output=solve(g))


def _fallback(rng, orientation, params, variant):
    """Asymmetric, gapped, fragmented witnesses; used only after bounded search."""
    bg, marker, palette = _roles(rng) if params['varied_palette'] else params['roles']
    a, b, c = (int(x) for x in palette[:3])
    top, bottom, left, right = (6 if variant == 0 else 4), 8, 6, 9
    g = np.full((top+bottom+1, left+right+1), bg, dtype=np.int64)
    rooms = _rooms(g.shape, top, left)
    carrier = np.array([[a, b, c], [c, a, b]] if variant == 0
                       else [[a, b, a], [c, a, b]])
    _put(rng, g, rooms[(1, 0)], carrier, bg)
    masks = {(0, 0): np.array([[a, b, c, a], [c, a, b, b]]),
             (0, 1): np.array([[bg, a, b], [c, bg, bg]]),
             (1, 1): np.array([[a, bg], [bg, c], [b, bg]])}
    for q, s in masks.items():
        _put(rng, g, rooms[q], s, bg)
    for r in (1, 3, top+2, g.shape[0]-3):
        g[r, left] = a if r % 2 else b
    for col in (2, left-1, left+2, g.shape[1]-3):
        g[top, col] = c if col % 2 else a
    g[0, left] = marker
    g[top, -1] = marker
    g = np.rot90(g, -orientation).copy()
    return dict(input=g, output=solve(g))


def _training_diverse(pairs):
    return (len({np.asarray(p['input']).shape for p in pairs}) > 1
            and len({_normalized(_decode(p['input'])['carrier']) for p in pairs}) > 1)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a reproducible puzzle with a held-out marker-pair combination."""
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    params = dict(roles=_roles(rng), varied_palette=bool(rng.random() < .65),
                  complexity=int(rng.integers(4)))
    first = int(rng.integers(4))
    train_orientations = [first, (first+2) % 4]
    test_orientations = [(first+1) % 4, (first+3) % 4]
    candidates = [[], []]
    training = None
    for attempt in range(80):
        side = attempt % 2
        p = _sample_pair(rng, train_orientations[side], params)
        cover = _evidence(p)
        for other, other_cover in candidates[1-side]:
            if _REQUIRED <= cover | other_cover and _training_diverse([p, other]):
                training = [p, other]
                break
        if training is not None:
            break
        candidates[side].append((p, cover))
    if training is None:
        training = [_fallback(rng, train_orientations[i], params, i) for i in range(2)]
        assert _REQUIRED <= set().union(*(_evidence(p) for p in training))
    puzzle, seen = dict(train=training, test=[]), {_key(p['input']) for p in training}
    for split, count, orientations in (('train', num_train, train_orientations),
                                        ('test', num_test, test_orientations)):
        while len(puzzle[split]) < count:
            for _ in range(64):
                p = _sample_pair(rng, int(rng.choice(orientations)), params)
                if _key(p['input']) not in seen:
                    break
            else:
                p = _fallback(rng, orientations[0], params, 0)
                info = _decode(p['input'])
                original = p['input'].copy()
                cells = sorted(info['axial'])
                inks = list(map(int, np.unique(original[~np.isin(original,
                                                     [info['background'], info['marker']])])))
                for serial in range(len(inks)**len(cells)):
                    trial = original.copy()
                    for j, cell in enumerate(cells):
                        trial[cell] = inks[(serial//len(inks)**j) % len(inks)]
                    if _key(trial) not in seen:
                        p = dict(input=trial, output=solve(trial))
                        break
                else:
                    raise RuntimeError('distinct-input fallback exhausted')
            puzzle[split].append(p)
            seen.add(_key(p['input']))
    for split in puzzle:
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, semantics, training witnesses, and the declared shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence = set(), set()
    sources = {'train': set(), 'test': set()}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            key = _key(g)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            info = _decode(g)
            bg = info['background']
            sources[split].add(info['source'])
            assert np.all(out[info['rooms'][info['source']]] == bg), 'source not cleared'
            for cell, ink in info['axial'].items():
                assert out[cell] == ink, 'original axis ink must survive'
            paths = _guide_cells(info)
            for r in range(g.shape[0]):
                cell = (r, info['ac'])
                if cell not in info['axial']:
                    assert out[cell] == (info['marker'] if cell in paths else bg)
            for c in range(g.shape[1]):
                cell = (info['ar'], c)
                if cell not in info['axial']:
                    assert out[cell] == (info['marker'] if cell in paths else bg)
            for q, s in info['stencils'].items():
                region = out[info['rooms'][q]]
                H, W = region.shape
                tiled = _tile(s, H, W, q)
                opaque = tiled != bg
                assert np.array_equal(region[opaque], tiled[opaque]), 'native print changed'
                carrier = _resize(info['carrier'], H, W)
                assert np.array_equal(region[~opaque], carrier[~opaque]), 'lost transparency'
            assert np.array_equal(out, solve(g)), 'solver disagrees'
            assert not np.array_equal(g, out), 'identity pair'
            if split == 'train':
                evidence |= _evidence(pair, info)
    assert _REQUIRED <= evidence, 'missing training witnesses: ' + str(_REQUIRED-evidence)
    assert _training_diverse(puzzle['train']), 'training must vary shape and swatch'
    assert len(sources['train']) == 2
    a, b = sorted(sources['train'])
    assert a[0] != b[0] and a[1] != b[1], 'training sources must be opposite'
    assert not sources['train'] & sources['test'], 'test must combine held-out marker pairs'
    return True


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _manual_regression():
    # Hand-calculated 9x9 scene: source BR, native diagonal TL, solid TR,
    # fragmented 1x3 BL, and guides with a one-cell internal gap.
    rows = ('000090000', '070050800', '007000800', '000060000',
            '950600000', '000000000', '070800120', '000000340', '000000000')
    answers = ('717298888', '172758888', '737498888', '374768888',
               '959600000', '718700000', '718700000', '738700000', '738700000')
    g = np.array([[int(c) for c in row] for row in rows])
    expected = np.array([[int(c) for c in row] for row in answers])
    for k in range(4):
        assert np.array_equal(solve(np.rot90(g, k)), np.rot90(expected, k))
    # Family members outside the ordinary 2..5 swatch sampler: a 1x1 carrier
    # and a uniform-row carrier reduced to a single destination row.
    small = g.copy()
    small[5:, 5:] = 0
    small[6, 6] = 2
    assert np.all(_decode(small)['carrier'] == 2)
    solve(small)
    reduced = np.zeros((7, 9), dtype=int)
    reduced[0, 4], reduced[1, 0] = 9, 9
    reduced[1, 2], reduced[2, 4], reduced[4, 4] = 5, 5, 6
    reduced[2, 1] = 7
    # Source BR is five rows tall, with a 2x2 carrier constant along rows.
    reduced[3:5, 6:8] = [[1, 2], [1, 2]]
    # TL and TR are height one: stencil TL at row zero would be on the border,
    # so they are empty. BL's (2,1) fragment is the required nonempty stencil.
    info = _decode(reduced)
    assert info['carrier'].shape == (2, 2) and not info['carrier_axes'][0]
    answer = solve(reduced)
    assert np.array_equal(answer[0, :4], [1, 1, 2, 2])


def self_test():
    """Bounded 42-seed audit, API boundaries, semantic fixtures, and mutants."""
    _manual_regression()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(917, 7, 3)) == _signature(generate_puzzle(917, 7, 3))
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1234+10*train+test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    for seed in range(8):
        rng = np.random.default_rng(9000+seed)
        params = dict(roles=_roles(rng), varied_palette=seed % 2 == 0, complexity=3)
        orientation = seed % 4
        pair = [_fallback(rng, orientation, params, 0),
                _fallback(rng, (orientation+2) % 4, params, 1)]
        assert _training_diverse(pair)
        assert _REQUIRED <= set().union(*(_evidence(p) for p in pair))
    shapes, carriers, masks, periods, axes, scales, backgrounds = (set() for _ in range(7))
    by_split = {'train': set(), 'test': set()}
    empty = fragmented = partial = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                bg, carrier = info['background'], info['carrier']
                shapes.add(info['grid'].shape)
                carriers.add(_normalized(carrier))
                axes.add(info['carrier_axes'])
                backgrounds.add(bg)
                by_split[split].add(info['source'])
                for q, s in info['stencils'].items():
                    H, W = info['grid'][info['rooms'][q]].shape
                    masks.add(_key(s != bg))
                    periods.add(s.shape)
                    scales.add((H/carrier.shape[0], W/carrier.shape[1]))
                    empty += int(not np.any(s != bg))
                    fragmented += int(len(_components(s != bg)) > 1)
                    partial += int(H % s.shape[0] != 0 or W % s.shape[1] != 0)
                # solve must neither modify its input nor depend on array layout.
                copied = pair['input'].copy()
                assert np.array_equal(solve(copied), pair['output'])
                assert np.array_equal(copied, pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert all(len(values) == 4 for values in by_split.values())
    assert len(shapes) >= 30 and len(carriers) >= 60 and len(masks) >= 80
    assert len(periods) >= 15 and len(scales) >= 30 and len(backgrounds) >= 8
    assert axes == {(False, False), (False, True), (True, False), (True, True)}
    assert empty >= 20 and fragmented >= 40 and partial >= 50
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=42, canvas_shapes=len(shapes),
                carrier_structures=len(carriers), stencil_masks=len(masks),
                crop_periods=len(periods), scale_combinations=len(scales),
                background_colors=len(backgrounds), empty_stencils=empty,
                fragmented_stencils=fragmented, partial_tiles=partial,
                independent_semantic_verification=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
