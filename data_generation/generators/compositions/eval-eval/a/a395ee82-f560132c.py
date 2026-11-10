"""Constellation molds: anchored glyph substitution defines a jigsaw target.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
import itertools
import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, found = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    found.append(q)
                    stack.append(q)
        result.append(np.array(sorted(found), dtype=int))
    return result


def _norm(coords):
    a = np.asarray(coords, dtype=int)
    a = a - a.min(axis=0)
    return tuple(sorted(map(tuple, a.tolist())))


def _turn(shape, k):
    a = np.array(shape, dtype=int)
    for _ in range(k % 4):
        a = np.column_stack((a[:, 1], -a[:, 0]))
    return _norm(a)


def _dims(shape):
    a = np.asarray(shape)
    return tuple(map(int, a.max(axis=0) + 1))


def _bits(coords, width):
    value = 0
    for r, c in coords:
        value |= 1 << (int(r) * width + int(c))
    return value


def _pack(shapes, target, limit=2, require_cover=True):
    """Exhaust all corner-pinned rotations; deduplicate colored solutions.

    The key-bearing shape is index zero, unrotated at the target's top-left.
    Distinct physical assignments with identical corner masks are equivalent.
    """
    height, width = target.shape
    corners = ((0, 0), (0, width-1), (height-1, 0), (height-1, width-1))
    full = _bits(np.argwhere(target), width)
    area = sum(map(len, shapes))
    if (area != int(target.sum()) if require_cover else area > int(target.sum())):
        return []
    table = {}
    for i, shape in enumerate(shapes):
        for corner in range(4):
            choices = {}
            for k in ([0] if i == 0 else range(4)):
                s = _turn(shape, k)
                h, w = _dims(s)
                if h > height or w > width:
                    continue
                r = 0 if corner < 2 else height-h
                c = 0 if corner % 2 == 0 else width-w
                cells = {(a+r, b+c) for a, b in s}
                if corners[corner] not in cells or sum(p in cells for p in corners) != 1:
                    continue
                mask = _bits(cells, width)
                if mask & full == mask:
                    choices.setdefault(mask, k)
            table[i, corner] = sorted(choices.items())
    if not table[0, 0]:
        return []
    results, seen = [], set()
    masks, rotations, pieces = [None]*4, [None]*4, [None]*4

    def visit(corner, remaining, occupied):
        if len(results) >= limit:
            return
        if corner == 4:
            signature = tuple(masks)
            if (not require_cover or occupied == full) and signature not in seen:
                seen.add(signature)
                results.append({'masks': signature, 'rotations': tuple(rotations),
                                'pieces': tuple(pieces)})
            return
        for i in remaining:
            for mask, k in table[i, corner]:
                if mask & occupied:
                    continue
                masks[corner], rotations[corner], pieces[corner] = mask, k, i
                visit(corner+1, tuple(j for j in remaining if j != i), occupied | mask)

    masks[0], rotations[0], pieces[0] = table[0, 0][0][0], 0, 0
    visit(1, (1, 2, 3), masks[0])
    return results


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must have integer cells')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid colors')
    return a


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(len(values) == 8, 'eight distinct local palette roles required')
    bg = int(values[np.argmax(counts)])
    _require(np.count_nonzero(a == bg) > a.size//2, 'background must be a strict majority')
    comps = _components(a != bg)
    large = [p for p in comps if len(p) > 1]
    dots = [tuple(p[0]) for p in comps if len(p) == 1]
    _require(len(large) == 5 and len(dots) >= 4, 'four pieces, glyph, and isolated map dots required')
    anchors = [p for p in large if len(np.unique(a[p[:, 0], p[:, 1]])) > 1]
    _require(len(anchors) == 1, 'one keyed piece required')
    anchor = anchors[0]
    av, ac = np.unique(a[anchor[:, 0], anchor[:, 1]], return_counts=True)
    _require(len(av) == 5 and sum(ac == 1) == 4 and sum(ac > 1) == 1,
             'anchor needs carrier and four singleton key entries')
    carrier = int(av[ac > 1][0])
    keypoints = anchor[np.isin(a[anchor[:, 0], anchor[:, 1]], av[ac == 1])]
    kr, kc = keypoints.min(axis=0)
    _require(set(map(tuple, keypoints.tolist())) == {(kr, kc), (kr+1, kc), (kr, kc+1), (kr+1, kc+1)},
             'key must be a 2 by 2 patch')
    key = a[kr:kr+2, kc:kc+2].copy()
    plain, glyphs = [], []
    for p in large:
        if p is anchor:
            continue
        colors = np.unique(a[p[:, 0], p[:, 1]])
        _require(len(colors) == 1, 'plain object must be monochrome')
        (plain if int(colors[0]) == carrier else glyphs).append(p)
    _require(len(plain) == 3 and len(glyphs) == 1, 'pieces share their carrier; glyph has its own color')
    glyph = glyphs[0]
    tr, tc = map(int, glyph.min(axis=0))
    er, ec = map(int, glyph.max(axis=0) + 1)
    ink = int(a[tuple(glyph[0])])
    mask = a[tr:er, tc:ec] == ink
    h, w = mask.shape
    _require(min(h, w) >= 3 and mask[0].all() and mask[-1].all()
             and mask[:, 0].all() and mask[:, -1].all(), 'glyph has a complete perimeter')
    exceptional = [p for p in dots if a[p] == ink]
    ordinary = [p for p in dots if a[p] != ink]
    _require(len(exceptional) == 1 and len({int(a[p]) for p in ordinary}) == 1,
             'map has one glyph-colored anchor and one ordinary dot color')
    marker = int(a[ordinary[0]])
    _require(len({bg, carrier, ink, marker, *map(int, key.flat)}) == 8, 'palette roles must be disjoint')
    pr, pc = map(int, np.min(dots, axis=0))
    br, bc = map(int, np.max(dots, axis=0))
    _require(all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in dots), 'pitch-two map required')
    mh, mw = (br-pr)//2+1, (bc-pc)//2+1
    layout = np.zeros((mh, mw), dtype=bool)
    for r, c in dots:
        layout[(r-pr)//2, (c-pc)//2] = True
    _require(min(mh, mw) >= 2 and all(layout[p] for p in ((0, 0), (0, mw-1), (mh-1, 0), (mh-1, mw-1))),
             'map must include its four bounding corners')
    _require(len(_components(layout)) == 1, 'coarse map must be connected')
    ar, az = exceptional[0]
    anchor_site = ((int(ar)-pr)//2, (int(az)-pc)//2)
    top, left = tr-anchor_site[0]*h, tc-anchor_site[1]*w
    target = np.kron(layout, mask)
    _require(0 <= top and top+target.shape[0] <= a.shape[0]
             and 0 <= left and left+target.shape[1] <= a.shape[1], 'anchored mold must fit without clipping')
    # Check the input separation convention on actual occupied cells.
    objects = [anchor] + plain + [glyph, np.array(dots)]
    for x, y in itertools.combinations(objects, 2):
        _require(int(np.abs(x[:, None, :]-y[None, :, :]).max(axis=2).min()) >= 2,
                 'objects require a one-cell background halo')
    shapes = [_norm(p) for p in [anchor] + plain]
    _require(min(map(len, shapes)) >= 4 and len(shapes[0]) >= 7, 'pieces are too small')
    solutions = _pack(shapes, target)
    _require(len(solutions) == 1, 'mold needs exactly one colored packing')
    return {'background': bg, 'carrier': carrier, 'ink': ink, 'marker': marker,
            'key': key, 'mask': mask, 'layout': layout, 'anchor': anchor_site,
            'origin': (top, left), 'target': target, 'shapes': shapes,
            'solution': solutions[0], 'glyph_origin': (tr, tc), 'objects': objects}


def _render(a, info):
    out = np.full(a.shape, info['background'], dtype=np.int8)
    height, width = info['target'].shape
    top, left = info['origin']
    for corner, bits in enumerate(info['solution']['masks']):
        color = int(info['key'].flat[corner])
        for r in range(height):
            for c in range(width):
                if (bits >> (r*width+c)) & 1:
                    out[top+r, left+c] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the mold and exhaustively pack visible shapes; input only."""
    a = np.asarray(input_grid)
    return _render(a, _decode(a))


def _glyph(rng, h, w):
    mask = np.ones((h, w), dtype=bool)
    interior = [(r, c) for r in range(1, h-1) for c in range(1, w-1) if (r, c) != (1, 1)]
    rng.shuffle(interior)
    # The top-left 2x2 remains available for the keyed region.
    for p in interior[:int(rng.integers(1, len(interior)+1))]:
        mask[p] = False
    return mask


def _asymmetric(mask):
    return all(not np.array_equal(mask, v) for v in (
        np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2), mask.T,
        np.flipud(mask).T, np.fliplr(mask).T, np.rot90(mask, 2).T))


def _map(rng, h, w, force_blank=False):
    corners = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))
    for _ in range(80):
        layout = rng.random((h, w)) < float(rng.uniform(0.5, 0.95))
        for p in corners:
            layout[p] = True
        if len(_components(layout)) == 1 and (not force_blank or not layout.all()):
            return layout
    layout = np.ones((h, w), dtype=bool)
    layout[0, 1] = False
    return layout


def _partition(rng, target, teaching):
    h, w = target.shape
    corners = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))
    for _ in range(240):
        owner = np.full((h, w), -1, dtype=np.int8)
        for i, p in enumerate(corners):
            owner[p] = i
        owner[:2, :2] = 0
        frontier = set()
        for r, c in np.argwhere(owner >= 0):
            for dr, dc in _N4:
                a, b = r+dr, c+dc
                if 0 <= a < h and 0 <= b < w and target[a, b] and owner[a, b] < 0:
                    frontier.add((int(a), int(b), int(owner[r, c])))
        # A fresh preference for each region produces unequal, interlocking cuts.
        weights = rng.uniform(0.4, 2.5, 4)
        while frontier:
            options = sorted(frontier)
            probabilities = np.array([weights[i] for _, _, i in options])
            probabilities /= probabilities.sum()
            r, c, i = options[int(rng.choice(len(options), p=probabilities))]
            frontier.remove((r, c, i))
            if owner[r, c] >= 0:
                continue
            owner[r, c] = i
            for dr, dc in _N4:
                a, b = r+dr, c+dc
                if 0 <= a < h and 0 <= b < w and target[a, b] and owner[a, b] < 0:
                    frontier.add((a, b, i))
        if np.any(owner[target] < 0):
            continue
        shapes = [_norm(np.argwhere(owner == i)) for i in range(4)]
        if min(map(len, shapes)) < 4 or len(shapes[0]) < 7:
            continue
        if teaching and any(len({_turn(s, k) for k in range(4)}) != 4 for s in shapes):
            continue
        if teaching and len({min(_turn(s, k) for k in range(4)) for s in shapes}) != 4:
            continue
        solutions = _pack(shapes, target)
        expected = tuple(_bits(np.argwhere(owner == i), w) for i in range(4))
        if len(solutions) == 1 and solutions[0]['masks'] == expected:
            if teaching and len(_pack(shapes, np.ones_like(target), require_cover=False)) < 2:
                continue
            return owner, shapes
    return None


def _place_scene(rng, owner, shapes, glyph, layout, anchor_site, colors, teaching):
    bg, carrier, ink, marker, *key = colors
    rotations = [0, 1, 2, 3] if teaching else [0] + list(map(int, rng.integers(0, 4, 3)))
    if teaching:
        rotations[1:] = list(map(int, rng.permutation([1, 2, 3])))
    oriented = [_turn(s, k) for s, k in zip(shapes, rotations)]
    h, w = glyph.shape
    mh, mw = layout.shape
    ah, aw = owner.shape
    map_coords = [(2*int(r), 2*int(c)) for r, c in np.argwhere(layout)]
    kr, kc = _pick(rng, [(r, c) for r in range(ah-1) for c in range(aw-1)
                         if np.all(owner[r:r+2, c:c+2] == 0)])
    # Output bounds constrain the sample glyph's position; source pieces may
    # occupy the future output area, since reconstruction starts from background.
    for _ in range(45):
        height = int(rng.integers(max(15, ah), 31))
        width = int(rng.integers(max(15, aw), 31))
        top = int(rng.integers(height-ah+1))
        left = int(rng.integers(width-aw+1))
        tr, tc = top+anchor_site[0]*h, left+anchor_site[1]*w
        grid = np.full((height, width), bg, dtype=np.int8)
        forbidden = np.zeros(grid.shape, dtype=bool)

        def paint(coords, offset, color):
            points = np.asarray(coords, dtype=int) + offset
            grid[points[:, 0], points[:, 1]] = color
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    forbidden[np.clip(points[:, 0]+dr, 0, height-1),
                              np.clip(points[:, 1]+dc, 0, width-1)] = True

        paint(np.argwhere(glyph), (tr, tc), ink)
        offsets = [None]*4
        # Place the largest boxes first, with random tie order.
        order = sorted(rng.permutation(4).tolist(), key=lambda i: -np.prod(_dims(oriented[i])))
        success = True
        for i in order + [4]:
            coords = oriented[i] if i < 4 else map_coords
            bh, bw = _dims(coords)
            options = []
            array = np.array(coords)
            for r in range(height-bh+1):
                for c in range(width-bw+1):
                    if not forbidden[r+array[:, 0], c+array[:, 1]].any():
                        options.append((r, c))
            if not options:
                success = False
                break
            offset = _pick(rng, options)
            paint(coords, offset, carrier if i < 4 else marker)
            if i < 4:
                offsets[i] = offset
            else:
                grid[offset[0]+2*anchor_site[0], offset[1]+2*anchor_site[1]] = ink
        if not success or np.count_nonzero(grid == bg) <= grid.size//2:
            continue
        r, c = offsets[0]
        grid[r+kr:r+kr+2, c+kc:c+kc+2] = np.array(key).reshape(2, 2)
        output = np.full_like(grid, bg)
        for i, color in enumerate(key):
            rr, cc = np.where(owner == i)
            output[top+rr, left+cc] = color
        return {'input': grid, 'output': output}
    return None


def _valid_count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _valid_count(num_train, 'num_train', 2, 10)
    nq = _valid_count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    transpose = bool(rng.integers(2))
    previous_anchor = previous_dims = previous_key = None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(100):
                teaching = split == 'train' and index == 0
                if split == 'train':
                    length = 3 if teaching else int(rng.integers(3, 5))
                    md = (2, length) if index % 2 == 0 else (length, 2)
                    if index >= 2 and rng.random() < 0.20:
                        md = (2, 2)
                    if transpose:
                        md = md[::-1]
                else:
                    md = _pick(rng, [(3, 3), (3, 4), (4, 3)])
                candidates = [(h, w) for h in range(3, 6) for w in range(3, 6)
                              if (h, w) != (3, 3) and h*md[0] <= 15 and w*md[1] <= 15
                              and h*w*md[0]*md[1] <= 180
                              and (not teaching or (h != w and min(h, w) >= 4))
                              and (split != 'train' or index != 1 or
                                   (h != previous_dims[0] and w != previous_dims[1]))]
                h, w = _pick(rng, candidates)
                glyph = _glyph(rng, h, w)
                if teaching and not _asymmetric(glyph):
                    continue
                layout = _map(rng, *md, force_blank=(split == 'train' and index < 2))
                sites = [tuple(map(int, p)) for p in np.argwhere(layout)]
                if teaching:
                    center = ((md[0]-1)//2, (md[1]-1)//2)
                    sites = [p for p in sites if p != center]
                if split == 'train' and index == 1:
                    sites = [p for p in sites if p[0] != previous_anchor[0] and p[1] != previous_anchor[1]]
                anchor_site = _pick(rng, sites)
                target = np.kron(layout, glyph)
                cut = _partition(rng, target, teaching)
                if cut is None:
                    continue
                owner, shapes = cut
                colors = list(map(int, rng.choice(10, 8, replace=False)))
                if split == 'train' and index == 1 and tuple(colors[4:]) == previous_key:
                    colors[4], colors[5] = colors[5], colors[4]
                pair = _place_scene(rng, owner, shapes, glyph, layout, anchor_site, colors, teaching)
                if pair is None:
                    continue
                signature = _signature(pair['input'])
                if signature in seen:
                    continue
                # Require that fine-scale cuts actually cross glyph copies.
                if not any(len(set(owner[r*h:(r+1)*h, c*w:(c+1)*w][glyph].tolist())) >= 2
                           for r, c in sites):
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if split == 'train' and index == 0:
                    previous_anchor, previous_dims = anchor_site, (h, w)
                    previous_key = tuple(colors[4:])
                break
            else:
                raise AssertionError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, metadata = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == out.shape, 'canvas must be preserved')
            signature = _signature(a)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            info = _decode(a)
            _require(np.array_equal(out, _render(a, info)), 'output disagrees with mold packing')
            bg, target = info['background'], info['target']
            top, left = info['origin']
            support = np.zeros(a.shape, dtype=bool)
            support[top:top+target.shape[0], left:left+target.shape[1]] = target
            _require(np.array_equal(out != bg, support), 'output support must equal expanded mold')
            _require(int(target.sum()) == sum(map(len, info['shapes'])), 'area must be conserved')
            _require(set(map(int, np.unique(out))) == {bg, *map(int, info['key'].flat)}, 'only key and background survive')
            h, w = info['mask'].shape
            stamp_colors = [len(np.unique(out[top+r*h:top+(r+1)*h, left+c*w:left+(c+1)*w][info['mask']]))
                            for r, c in np.argwhere(info['layout'])]
            _require(max(stamp_colors) >= 2, 'jigsaw cuts must cross glyph boundaries')
            metadata[split].append(info)
    train = metadata['train']
    _require(train[0]['anchor'][0] != train[1]['anchor'][0]
             and train[0]['anchor'][1] != train[1]['anchor'][1], 'teach both anchor coordinates')
    _require(all(train[0]['mask'].shape[i] != train[1]['mask'].shape[i] for i in (0, 1)), 'teach both glyph pitches')
    _require(all(not x['mask'].all() and not x['layout'].all() for x in train[:2]), 'teach both kinds of negative space')
    _require(not np.array_equal(train[0]['key'], train[1]['key']), 'teach a variable visible legend')
    _require(set(train[0]['solution']['rotations']) == {0, 1, 2, 3}, 'teach all quarter-turns')
    _require(_asymmetric(train[0]['mask']), 'teach exact glyph orientation')
    first_map = train[0]['layout'].shape
    _require(train[0]['anchor'] != ((first_map[0]-1)//2, (first_map[1]-1)//2), 'rule out a center anchor shortcut')
    _require(len(_pack(train[0]['shapes'], np.ones_like(train[0]['target']), require_cover=False)) == 2,
             'first demonstration must need the mold to disambiguate packing')
    return metadata


def self_test():
    """Bounded 42-seed coverage, counterfactuals, counts, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fields = {name: set() for name in ('canvases', 'glyphs', 'maps', 'anchors', 'cuts', 'backgrounds', 'areas', 'rotations')}
    for p in puzzles:
        metas = validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair, m in zip(p[split], metas[split]):
                fields['canvases'].add(pair['input'].shape)
                fields['glyphs'].add(_signature(m['mask']))
                fields['maps'].add(_signature(m['layout']))
                fields['anchors'].add(m['anchor'])
                fields['backgrounds'].add(m['background'])
                fields['areas'].add(tuple(sorted(map(len, m['shapes']))))
                fields['rotations'].update(m['solution']['rotations'])
                fields['cuts'].add((m['target'].shape, m['solution']['masks']))
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve mismatch')
        # Omitting A's holes or anchor changes the answer, and omitting B's
        # shape-dependent key coloring cannot reproduce a multicolor glyph.
        m = metas['train'][0]
        _require(not m['target'].all(), 'solid bounding-box shortcut must fail')
        alt = np.full_like(p['train'][0]['output'], m['background'])
        top, left = m['origin']
        block = alt[top:top+m['target'].shape[0], left:left+m['target'].shape[1]]
        block[m['target']] = int(m['key'].flat[0])
        _require(not np.array_equal(alt, p['train'][0]['output']), 'single-color substitution must fail')
    _require(len(fields['canvases']) >= 45, 'canvas variation collapsed')
    _require(len(fields['glyphs']) >= 20 and len(fields['maps']) >= 12, 'mold variation collapsed')
    _require(len(fields['anchors']) >= 8 and len(fields['backgrounds']) == 10, 'anchor/palette variation collapsed')
    _require(len(fields['cuts']) >= 180 and len(fields['areas']) >= 100, 'jigsaw geometry variation collapsed')
    _require(fields['rotations'] == {0, 1, 2, 3}, 'rotation coverage missing')
    again = generate_puzzle(17)
    _require(all(np.array_equal(a[k], b[k]) for s in puzzles[17] for a, b in zip(puzzles[17][s], again[s])
                 for k in ('input', 'output')), 'seed determinism failed')
    # A literal palette permutation and extra blank margin carry no semantics.
    pair = next(q for p in puzzles for s in p for q in p[s]
                if max(q['input'].shape) <= 28)
    _require(np.array_equal(solve(9-pair['input']), 9-pair['output']), 'palette equivariance failed')
    bg = _decode(pair['input'])['background']
    padded_input = np.pad(pair['input'], 1, constant_values=bg)
    padded_output = np.pad(pair['output'], 1, constant_values=bg)
    _require(np.array_equal(solve(padded_input), padded_output), 'translation equivariance failed')
    m = _decode(pair['input'])
    k0, k1 = map(int, m['key'].flat[:2])
    changed = pair['input'].copy()
    changed[pair['input'] == k0] = k1
    changed[pair['input'] == k1] = k0
    expected = pair['output'].copy()
    expected[pair['output'] == k0] = k1
    expected[pair['output'] == k1] = k0
    _require(np.array_equal(solve(changed), expected)
             and not np.array_equal(expected, pair['output']), 'key counterfactual failed')
    for a, b in ((2, 1), (10, 4), (6, 3)):
        p = generate_puzzle(500+a, a, b)
        _require(len(p['train']) == a and len(p['test']) == b, 'explicit counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {k: len(v) for k, v in fields.items()}


if __name__ == '__main__':
    print(self_test())
