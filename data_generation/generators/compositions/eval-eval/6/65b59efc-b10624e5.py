"""Folded stencil cutouts: keyed, anchored impressions with exclusive coverage.

No parent code, files, examples, random state, or training pairs are used by solve.
Generation paints logical cells forwards. Solving reconstructs exact scaled
rasters from the returned input and samples them on the field.
"""
from __future__ import annotations

import json
from collections import deque
from math import gcd

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape"
    assert np.issubdtype(a.dtype, np.integer), "grid must contain integers"
    assert np.all((0 <= a) & (a <= 9)), "invalid color"
    return a.astype(np.int8, copy=False)


def _key(a):
    return tuple(a.shape), a.astype(np.int8).tobytes()


def _square(mask):
    points = np.argwhere(mask)
    assert len(points), "missing square"
    r, c = map(int, points.min(0))
    bottom, right = map(int, points.max(0) + 1)
    side = bottom-r
    assert side == right-c and len(points) == side*side, "non-square anchor"
    return r, c, side


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    parts = []
    while remaining:
        point = min(remaining)
        remaining.remove(point)
        todo, cells = deque([point]), [point]
        while todo:
            r, c = todo.popleft()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    cells.append(q)
        parts.append(cells)
    return parts


def _sector(anchor, pivot):
    r, c, side, _ = anchor
    pr, pc = pivot
    assert r+side <= pr or r > pr, "anchor spans a pivot row"
    assert c+side <= pc or c > pc, "anchor spans a pivot column"
    return (2 if r > pr else 0) + (1 if c > pc else 0)


def _decode_candidate(grid, n, sep):
    # Each card has n image rows, a blank spacer row, a swatch row, and
    # a terminal separator column. One full separator row follows the bank.
    border = n+2
    bg = int(grid[n, 0])
    assert bg != sep
    columns = [c for c in range(grid.shape[1]) if np.all(grid[:border, c] == sep)]
    assert 2 <= len(columns) <= 4
    k = len(columns)
    assert columns == [j*(n+1)+n for j in range(k)]
    end = k*(n+1)
    assert np.all(grid[:border, end:] == bg)
    cards, color_sets = [], []
    for j in range(k):
        left = j*(n+1)
        raw = grid[:n, left:left+n]
        colors = set(map(int, np.unique(raw)))-{bg}
        assert len(colors) == 2 and sep not in colors
        color_sets.append(colors)
        assert np.all(grid[n, left:left+n] == bg)
        swatch = grid[n+1, left:left+n]
        assert swatch[n//2] != bg
        assert np.count_nonzero(swatch != bg) == 1
        cards.append(dict(raw=raw, target=int(swatch[n//2])))
    shared = set.intersection(*color_sets)
    assert len(shared) == 1, "ambiguous metric marker"
    marker = shared.pop()
    labels = []
    for card, colors in zip(cards, color_sets):
        label = next(iter(colors-{marker}))
        assert label not in labels
        labels.append(label)
        ar, ac, side = _square(card['raw'] == marker)
        assert 1 <= side <= 3 and side < n
        card.update(label=label, mask=card['raw'] != bg, anchor=(ar, ac, side))
    field = grid[border+1:]
    assert min(field.shape) >= 5
    assert set(map(int, np.unique(field))) <= {bg, marker, *labels}
    positions = np.argwhere(field == marker)
    assert len(positions) == 1, "expected a single field pivot"
    pivot = tuple(map(int, positions[0]))
    assert 1 <= pivot[0] < field.shape[0]-1 and 1 <= pivot[1] < field.shape[1]-1
    anchors = []
    for label in labels:
        parts = _components(field == label)
        assert parts, "unused library identity"
        for cells in parts:
            rr, cc = np.asarray(cells).T
            r, c, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()+1), int(cc.max()+1)
            side = bottom-r
            assert side == right-c and len(cells) == side*side, "field command is not a square"
            anchors.append((r, c, side, label))
    assert 2 <= len(anchors) <= 12
    anchors.sort()
    for a in anchors:
        _sector(a, pivot)
    return dict(n=n, k=k, bg=bg, sep=sep, marker=marker, cards=cards,
                by_label={c['label']: c for c in cards}, labels=labels,
                field=field, pivot=pivot, anchors=anchors)


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for n in range(2, 7):
        r = n+2
        if grid.shape[0] <= r+5 or not np.all(grid[r] == grid[r, 0]):
            continue
        try:
            candidates.append(_decode_candidate(grid, n, int(grid[r, 0])))
        except (AssertionError, IndexError):
            pass
    assert len(candidates) == 1, f"expected one library/field interpretation, got {len(candidates)}"
    return candidates[0]


def _scaled(card, destination_side, vflip, hflip, solid=False):
    """Exact rational rasterization, with coordinates relative to the square.

    A destination pixel may merge source pixels only when all agree. There
    is no rounding, interpolation, or majority vote, even outside the field.
    """
    ar, ac, source_side = card['anchor']
    mask = card['mask'].copy()
    if solid:
        mask[:] = True
    yy, xx = np.nonzero(mask)
    yy, xx = yy-ar, xx-ac
    top = int(yy.min())//source_side*source_side
    left = int(xx.min())//source_side*source_side
    bottom = (int(yy.max())//source_side+1)*source_side
    right = (int(xx.max())//source_side+1)*source_side
    raw = np.zeros((bottom-top, right-left), dtype=bool)
    raw[yy-top, xx-left] = True
    fine = np.repeat(np.repeat(raw, destination_side, axis=0), destination_side, axis=1)
    h, w = fine.shape[0]//source_side, fine.shape[1]//source_side
    blocks = fine.reshape(h, source_side, w, source_side)
    small, large = blocks.min((1, 3)), blocks.max((1, 3))
    assert np.array_equal(small, large), "incompatible source detail and anchor scale"
    offset = [-top*destination_side//source_side, -left*destination_side//source_side]
    if vflip:
        small = small[::-1]
        offset[0] = h-offset[0]-destination_side
    if hflip:
        small = small[:, ::-1]
        offset[1] = w-offset[1]-destination_side
    return small, tuple(offset)


def _layers(data, alternative=None, orientations=None):
    h, w = data['field'].shape
    layers, inks, identities, clipped, scales = [], [], [], [], []
    for anchor in data['anchors']:
        r, c, side, label = anchor
        card = data['by_label'][label]
        if alternative == 'first_stencil':
            card = data['cards'][0]
        q = _sector(anchor, data['pivot'])
        if alternative == 'canvas_midpoint':
            q = 2*int(r+(side-1)/2 > (h-1)/2) + int(c+(side-1)/2 > (w-1)/2)
        flips = (bool(q//2), bool(q % 2))
        if orientations is not None and q in orientations:
            flips = orientations[q]
        if alternative == 'no_reflection':
            flips = (False, False)
        used_side = card['anchor'][2] if alternative == 'no_scale' else side
        raster, (oy, ox) = _scaled(card, used_side, *flips, solid=alternative == 'solid_card')
        top, left = r-oy, c-ox
        rows, cols = np.arange(h)-top, np.arange(w)-left
        inside = ((rows[:, None] >= 0) & (rows[:, None] < raster.shape[0]) &
                  (cols[None, :] >= 0) & (cols[None, :] < raster.shape[1]))
        layer = inside & raster[np.clip(rows, 0, raster.shape[0]-1)[:, None],
                                np.clip(cols, 0, raster.shape[1]-1)[None, :]]
        if alternative == 'quadrant_clip':
            pr, pc = data['pivot']
            yy, xx = np.indices((h, w))
            layer &= ((yy > pr) if q//2 else (yy < pr)) & ((xx > pc) if q % 2 else (xx < pc))
        layers.append(layer)
        inks.append(label if alternative == 'no_key' else data['by_label'][label]['target'])
        identities.append(label)
        clipped.append(int(raster.sum())-int(layer.sum()))
        g = gcd(side, data['by_label'][label]['anchor'][2])
        scales.append((side//g, data['by_label'][label]['anchor'][2]//g))
    return np.array(layers), inks, identities, clipped, scales


def _combine(layers, inks, identities, bg, mode=None):
    count = layers.sum(0)
    output = np.full(count.shape, bg, dtype=np.int8)
    if mode == 'different_identity_only':
        distinct = np.array([layers[np.array(identities) == label].any(0)
                             for label in sorted(set(identities))]).sum(0)
        keep = distinct == 1
    elif mode == 'parity':
        keep = count % 2 == 1
    elif mode == 'overwrite':
        keep = count > 0
    else:
        keep = count == 1
    for layer, ink in zip(layers, inks):
        output[layer & keep] = ink
    return output


def _render(data, alternative=None, orientations=None):
    layers, inks, ids, _, _ = _layers(data, alternative, orientations)
    return _combine(layers, inks, ids, data['bg'], alternative)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the printing field entirely from one input grid."""
    return _render(_decode(input_grid))


def _motif(rng, size, witness=False):
    r, c = map(int, rng.integers(size, size=2))
    style = int(rng.integers(9))
    yy, xx = np.indices((size, size))
    if style == 0:
        mask = np.ones((size, size), dtype=bool)
    elif style == 1:
        mask = (yy == 0) | (xx == 0) | (yy == size-1) | (xx == size-1)
    elif style == 2:
        mask = (yy+xx) % 2 == int(rng.integers(2))
    elif style == 3:
        mask = (yy == int(rng.integers(size))) | (xx == int(rng.integers(size)))
    elif style == 4:
        mask = (yy % 2 == int(rng.integers(2))) if rng.random() < .5 else (xx % 2 == int(rng.integers(2)))
    elif style == 5:
        mask = np.zeros((size, size), dtype=bool)
        y, x = r, c
        for _ in range(int(rng.integers(size, size*size+1))):
            mask[y, x] = True
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            y, x = min(size-1, max(0, y+dy)), min(size-1, max(0, x+dx))
    else:
        density = float(rng.uniform(.18, .85))
        mask = rng.random((size, size)) < density
    mask[r, c] = True
    if mask.sum() == 1:
        mask[(r+1) % size, c] = True
    if witness and mask.all():
        # Preserve dense examples elsewhere; evidence must expose transparency.
        options = [(y, x) for y in range(size) for x in range(size) if (y, x) != (r, c)]
        y, x = options[int(rng.integers(len(options)))]
        mask[y, x] = False
    return mask, (r, c)


def _forward(logical, anchors, pivot, shape, targets, bg):
    """Construction uses rectangles on a logical lattice, never solve's raster."""
    layers, inks, ids = [], [], []
    for r, c, side, label in anchors:
        mask, (ar, ac), _, anchor_cells = logical[label]
        assert side % anchor_cells == 0
        destination_unit = side//anchor_cells
        vertical, horizontal = r > pivot[0], c > pivot[1]
        layer = np.zeros(shape, dtype=bool)
        for y, x in np.argwhere(mask):
            dy, dx = int(y)-ar, int(x)-ac
            top = r + (anchor_cells-1-dy if vertical else dy)*destination_unit
            left = c + (anchor_cells-1-dx if horizontal else dx)*destination_unit
            y0, y1 = max(0, top), min(shape[0], top+destination_unit)
            x0, x1 = max(0, left), min(shape[1], left+destination_unit)
            if y0 < y1 and x0 < x1:
                layer[y0:y1, x0:x1] = True
        layers.append(layer)
        inks.append(targets[label])
        ids.append(label)
    return _combine(np.array(layers), inks, ids, bg)


def _build(rng, kind):
    witness = kind in ('orientation', 'collision')
    k = int(rng.integers(2, 5))
    n = int(rng.choice([4, 6])) if kind == 'orientation' else int(rng.choice([2, 3, 4, 5, 6], p=[.08, .22, .30, .18, .22]))
    width = int(rng.integers(max(18, k*(n+1)), 31))
    height = int(rng.integers(16, 28-n))
    colors = list(map(int, rng.permutation(10)))
    bg, sep, marker = colors[:3]
    labels = colors[3:3+k]
    targets = {}
    available = [c for c in range(10) if c != bg]
    for label in labels:
        choices = [c for c in available if not witness or (c != label and c not in targets.values())]
        targets[label] = int(rng.choice(choices))
    if not witness and rng.random() < .25:
        targets[labels[-1]] = targets[labels[0]]
    logical = {}
    for j, label in enumerate(labels):
        divisors = [u for u in (1, 2, 3) if n % u == 0 and n//u >= 2]
        unit = 2 if kind == 'orientation' and j == 0 else int(rng.choice(divisors))
        mask, anchor = _motif(rng, n//unit, witness)
        anchor_cells = 1
        if unit == 1 and n >= 3 and rng.random() < .30:
            # Fine features smaller than the metric square are legal when
            # destination sizes preserve their actual pixel boundaries.
            anchor_cells = 2 if n <= 4 else int(rng.choice([2, 3]))
            anchor = tuple(map(int, rng.integers(n-anchor_cells+1, size=2)))
            ar, ac = anchor
            mask[ar:ar+anchor_cells, ac:ac+anchor_cells] = True
            outside = [(r, c) for r in range(n) for c in range(n)
                       if not (ar <= r < ar+anchor_cells and ac <= c < ac+anchor_cells)]
            if mask.sum() == anchor_cells**2:
                r, c = outside[int(rng.integers(len(outside)))]
                mask[r, c] = True
        logical[label] = mask, anchor, unit, anchor_cells
    if kind == 'test':
        # A point frame, rather than panel dimensions, must control reflection.
        pivot = [int(rng.integers(height//3, height-height//3)),
                 int(rng.integers(width//3, width-width//3))]
        axis = int(rng.integers(2))
        length = (height, width)[axis]
        edge = int(rng.integers(2, max(3, length//4+1)))
        pivot[axis] = edge if rng.random() < .5 else length-1-edge
        pivot = tuple(pivot)
    else:
        pivot = (int(rng.integers(height//3, height-height//3)),
                 int(rng.integers(width//3, width-width//3)))
    m = int(rng.integers(max(5, k+2), min(10, k+6)))
    identity_order = [labels[0], labels[0]] + labels[1:]
    identity_order += list(map(int, rng.choice(labels, m-len(identity_order))))
    sectors = list(map(int, rng.permutation(4))) + list(map(int, rng.integers(4, size=max(0, m-4))))
    anchors, used = [], np.zeros((height, width), dtype=bool)
    pr, pc = pivot
    used[max(0, pr-1):pr+2, max(0, pc-1):pc+2] = True
    for index, label in enumerate(identity_order):
        anchor_cells = logical[label][3]
        source_side = logical[label][2]*anchor_cells
        if kind == 'orientation' and index == 0:
            side = source_side
        elif kind == 'orientation' and index == 1:
            side = 1
        else:
            options = [s for s in (1, 2, 3, 4) if s % anchor_cells == 0]
            side = int(rng.choice(options))
        q = sectors[index]
        top, bottom = (0, pr) if q//2 == 0 else (pr+1, height)
        left, right = (0, pc) if q % 2 == 0 else (pc+1, width)
        candidates = [(r, c) for r in range(top, bottom-side+1)
                      for c in range(left, right-side+1) if not used[r:r+side, c:c+side].any()]
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        anchors.append((r, c, side, label))
        used[max(0, r-1):min(height, r+side+1), max(0, c-1):min(width, c+side+1)] = True
    library_order = list(map(int, rng.permutation(labels)))
    grid = np.full((n+3+height, width), bg, dtype=np.int8)
    for j, label in enumerate(library_order):
        c = j*(n+1)
        mask, (ar, ac), unit, anchor_cells = logical[label]
        pixels = np.repeat(np.repeat(mask, unit, axis=0), unit, axis=1)
        grid[:n, c:c+n] = np.where(pixels, label, bg)
        grid[ar*unit:(ar+anchor_cells)*unit, c+ac*unit:c+(ac+anchor_cells)*unit] = marker
        grid[n+1, c+n//2] = targets[label]
        grid[:n+2, c+n] = sep
    grid[n+2] = sep
    field = grid[n+3:]
    field[pivot] = marker
    for r, c, side, label in anchors:
        field[r:r+side, c:c+side] = label
    output = _forward(logical, anchors, pivot, field.shape, targets, bg)
    return dict(input=grid, output=output)


def _evidence(data, output, which):
    layers, inks, ids, clipped, scales = _layers(data)
    counts = layers.sum(0)
    # Each library entry has visible unshared ink, not merely a seed that vanishes.
    if any(not np.any(layers[np.array(ids) == label].any(0) & (counts == 1))
           for label in data['labels']):
        return False
    if which == 'orientation':
        if not any(lost == 0 and a != b for lost, (a, b) in zip(clipped, scales)):
            return False
        for q in range(4):
            correct = (bool(q//2), bool(q % 2))
            for v, h in ((False, False), (False, True), (True, False), (True, True)):
                if (v, h) != correct and np.array_equal(output, _render(data, orientations={q: (v, h)})):
                    return False
        alternatives = ('first_stencil', 'no_key', 'no_scale', 'solid_card',
                        'canvas_midpoint', 'quadrant_clip')
    elif which == 'collision':
        if not any(clipped) or not np.any(counts == 3):
            return False
        alternatives = ('overwrite', 'different_identity_only', 'parity')
    elif which == 'test':
        if not any(clipped) or not np.any(counts > 1):
            return False
        alternatives = ('canvas_midpoint',)
    else:
        return True
    for alternative in alternatives:
        try:
            wrong = _render(data, alternative=alternative)
        except AssertionError:
            # The wrong identity may select fine geometry incompatible with
            # the destination square. That hypothesis is already refuted by
            # exact rasterization; actual source selections must always work.
            assert alternative == 'first_stencil'
            continue
        if np.array_equal(output, wrong):
            return False
    return True


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            kind = ('orientation' if index == 0 else 'collision' if index == 1 else 'ordinary') if split == 'train' else 'test'
            for _ in range(600):
                pair = _build(rng, kind)
                if pair is None or _key(pair['input']) in seen:
                    continue
                data = _decode(pair['input'])
                if not _evidence(data, pair['output'], kind):
                    continue
                # Reject degenerate fields where a whole library identity's
                # footprint is erased. Geometry remains important in tests too.
                if np.array_equal(pair['output'], np.full(pair['output'].shape, data['bg'])):
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f'bounded construction exhausted for {kind}')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _central(data):
    return all(length//3 <= p < length-length//3
               for length, p in zip(data['field'].shape, data['pivot']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, decoded = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            data = _decode(inp)
            assert out.shape == data['field'].shape, 'return only the field'
            expected = _render(data)
            assert np.array_equal(out, expected), 'wrong exclusive stencil print'
            assert set(map(int, np.unique(out))) <= {data['bg'], *[c['target'] for c in data['cards']]}
            assert np.any(out != data['bg']), 'empty print'
            assert _evidence(data, out, 'ordinary'), 'unused ink identity'
            assert _central(data) == (split == 'train'), 'pivot-position shift is missing'
            if split == 'test':
                assert _evidence(data, out, 'test'), 'test shift/collision/clipping is not consequential'
            decoded[split].append((data, out))
    assert any(_evidence(d, o, 'orientation') for d, o in decoded['train']), 'missing orientation/scale/key witness'
    assert any(_evidence(d, o, 'collision') for d, o in decoded['train']), 'missing exclusive-coverage witness'


def _measure(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _decode(pair['input'])
            layers, _, ids, clipped, scales = _layers(d)
            count = layers.sum(0)
            records.append(dict(split=split, shape=list(pair['input'].shape),
                field=list(d['field'].shape), n=d['n'], k=d['k'], anchors=len(ids),
                roles=[d['bg'], d['sep'], d['marker']], pivot=list(d['pivot']),
                sides=sorted({a[2] for a in d['anchors']}),
                source_sides=sorted({c['anchor'][2] for c in d['cards']}),
                scales=[list(s) for s in sorted(set(scales))],
                clipped=sum(c > 0 for c in clipped), maximum_coverage=int(count.max()),
                collision_pixels=int((count > 1).sum()), exclusive_pixels=int((count == 1).sum()),
                silhouettes=[(c['mask'].shape, c['mask'].tobytes()) for c in d['cards']]))
    return records


def self_test():
    """Bounded 42-seed suite, counterfactuals, metamorphisms and corruption."""
    shapes, palettes, motifs, scales, signatures, counts = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple(_key(np.asarray(pair[key])) for s in ('train', 'test')
                          for pair in p[s] for key in ('input', 'output'))
        signatures.add(signature)
        if seed < 2:
            q = generate_puzzle(seed)
            assert signature == tuple(_key(pair[key]) for s in ('train', 'test')
                                      for pair in q[s] for key in ('input', 'output'))
        for s in ('train', 'test'):
            for pair in p[s]:
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        local = _measure(p)
        records.extend(local)
        for r in local:
            shapes.add(tuple(r['shape']))
            palettes.add(tuple(r['roles']))
            motifs.update(r['silhouettes'])
            scales.update(map(tuple, r['scales']))
            counts.add((r['k'], r['anchors']))
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(palettes) >= 60 and len(motifs) >= 100
    assert {r['n'] for r in records} == {2, 3, 4, 5, 6}
    assert {r['k'] for r in records} == {2, 3, 4} and len(counts) >= 9
    assert {(1, 1), (1, 2), (2, 1), (3, 2)} <= scales
    assert max(r['maximum_coverage'] for r in records) >= 4
    # Arbitrary bijective recoloring of every observable role must commute.
    pair = generate_puzzle(50, 2, 1)['test'][0]
    permutation = np.random.default_rng(19).permutation(10)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # Library order has no effect. The complete card, metric square and key move together.
    d = _decode(pair['input'])
    reordered = pair['input'].copy()
    n, k = d['n'], d['k']
    bank = reordered[:n+2, :k*(n+1)].copy()
    for j in range(k):
        reordered[:n+2, j*(n+1):(j+1)*(n+1)] = bank[:, (k-1-j)*(n+1):(k-j)*(n+1)]
    assert np.array_equal(solve(reordered), pair['output'])
    # Reflecting just the field toggles the relevant pivot-sector bit and
    # reflects the answer. Library orientation remains its fixed convention.
    for axis in (0, 1):
        reflected = pair['input'].copy()
        reflected[n+3:] = np.flip(reflected[n+3:], axis=axis)
        assert np.array_equal(solve(reflected), np.flip(pair['output'], axis=axis))
    # verify.py exhausts all legal counts; keep the embedded suite bounded
    # while covering both extremes and a non-boundary request.
    for a, b in ((2, 1), (5, 4), (10, 1), (10, 4), (6, 3)):
        p = generate_puzzle(300+a*10+b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(70, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    # Pixel-level fine geometry has exact integral enlargements but rejects
    # incompatible fractional reductions, rather than silently rounding them.
    card = dict(mask=np.array([[1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool), anchor=(0, 0, 2))
    _scaled(card, 4, True, False)
    try:
        _scaled(card, 1, False, False)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous raster scale accepted')
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), palettes=len(palettes),
                silhouettes=len(motifs), scale_ratios=[list(x) for x in sorted(scales)],
                library_anchor_counts=[list(x) for x in sorted(counts)])


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
