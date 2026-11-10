"""Dye a uniquely localized photograph using enclosure in a visible atlas.

Only standard-library modules and NumPy. No parent imports, files, learned
templates, generation state or seed are used by solve(). No execution traces
or per-seed payloads are written; self_test streams its bounded corpus.
"""
from __future__ import annotations

import hashlib
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _transform(a, index):
    return np.rot90(np.fliplr(a) if index >= 4 else a, index % 4).copy()


def _key(a):
    return (a.shape, np.asarray(a, dtype=np.int8).tobytes())


def _components(mask, diagonal=False):
    seen = np.zeros(mask.shape, dtype=bool)
    steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
    result = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(int(r), int(c))], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=True)


def _sheets(grid):
    neutral = int(grid[0, 0])
    if not all(np.all(edge == neutral) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError('a complete neutral border is required')
    panels = []
    for cells in _components(grid != neutral):
        rr, cc = zip(*cells)
        r, c, h, w = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
        if len(cells) != h*w:
            raise ValueError('sheets must be filled rectangles')
        panels.append(grid[r:r+h, c:c+w].copy())
    if len(panels) != 2:
        raise ValueError('expected two separate sheets')
    queries = [a for a in panels if len(np.unique(a)) == 2]
    atlases = [a for a in panels if len(np.unique(a)) > 2]
    if len(queries) != 1 or len(atlases) != 1:
        raise ValueError('ambiguous sheet roles')
    return atlases[0], queries[0], neutral


def _atlas(atlas):
    colors, counts = np.unique(atlas, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError('ambiguous atlas background')
    bg = int(colors[np.argmax(counts)])
    h, w = atlas.shape
    candidates = []
    for cells in _components(atlas != bg, diagonal=True):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        sh, sw = r1-r0+1, c1-c0+1
        if len(cells) != sh*sw or len({int(atlas[q]) for q in cells}) < 2:
            continue
        lanes = []
        if sh == 2 and 2 <= sw <= 4:
            if r0 == 0:
                lanes.append(('top', atlas[r0, c0:c1+1], atlas[r1, c0:c1+1]))
            if r1 == h-1:
                lanes.append(('bottom', atlas[r1, c0:c1+1], atlas[r0, c0:c1+1]))
        if sw == 2 and 2 <= sh <= 4:
            if c0 == 0:
                lanes.append(('left', atlas[r0:r1+1, c0], atlas[r0:r1+1, c1]))
            if c1 == w-1:
                lanes.append(('right', atlas[r0:r1+1, c1], atlas[r0:r1+1, c0]))
        for side, outer, inner in lanes:
            sources, fills = tuple(map(int, outer)), tuple(map(int, inner))
            if len(set(sources)) == len(sources):
                keymask = np.zeros(atlas.shape, dtype=bool)
                keymask[tuple(zip(*cells))] = True
                candidates.append((side, dict(zip(sources, fills)), keymask))
    if len(candidates) != 1:
        raise ValueError('ambiguous or absent local edge key')
    side, mapping, keymask = candidates[0]
    dye = np.full(atlas.shape, -1, dtype=np.int8)
    owner = np.full(atlas.shape, -1, dtype=np.int8)
    diagonal_holes = np.zeros(atlas.shape, dtype=bool)
    # Eight-neighbor exterior reachability provides a topology witness only.
    exterior8 = np.zeros(atlas.shape, dtype=bool)
    for cells in _components(atlas == bg, diagonal=True):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            exterior8[tuple(zip(*cells))] = True
    regions = []
    for cells in _components(atlas == bg):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            continue
        boundary = set()
        for r, c in cells:
            for dr, dc in STEPS:
                q = r+dr, c+dc
                if atlas[q] != bg:
                    boundary.add(q)
        if any(keymask[q] for q in boundary):
            raise ValueError('key cannot enclose a cavity')
        inks = {int(atlas[q]) for q in boundary}
        if len(inks) != 1:
            raise ValueError('mixed cavity ownership')
        ink = next(iter(inks))
        ix = tuple(zip(*cells))
        owner[ix] = ink
        if ink in mapping:
            dye[ix] = mapping[ink]
        diagonal_holes[ix] = exterior8[ix]
        regions.append((ink, cells))
    return dict(bg=bg, side=side, mapping=mapping, keymask=keymask,
                wall=(atlas != bg) & ~keymask, dye=dye, owner=owner,
                diagonal=diagonal_holes, regions=regions)


def _matches(info, query):
    colors = sorted(map(int, np.unique(query)))
    if len(colors) != 2:
        return []
    h, w = query.shape
    result = []
    for t in range(8):
        full = _transform(info['wall'], t)
        if h > full.shape[0] or w > full.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
        keys = np.lib.stride_tricks.sliding_window_view(_transform(info['keymask'], t), (h, w))
        allowed = ~np.any(keys, axis=(-2, -1))
        for ink in colors:
            match = np.all(windows == (query == ink), axis=(-2, -1)) & allowed
            for r, c in zip(*np.nonzero(match)):
                result.append((t, int(r), int(c), colors[1] if ink == colors[0] else colors[0], ink))
    return result


def _project(a, match, shape):
    t, r, c, _, _ = match
    return _transform(a, t)[r:r+shape[0], c:c+shape[1]].copy()


def _paint(query, dye):
    result = query.copy()
    eligible = dye >= 0
    result[eligible] = dye[eligible]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the supplied input. Neither demonstrations nor state are read."""
    atlas, query, _ = _sheets(_array(input_grid))
    info = _atlas(atlas)
    matches = _matches(info, query)
    if len(matches) != 1:
        raise ValueError('photograph must have one exact orientation/role/location match')
    return _paint(query, _project(info['dye'], matches[0], query.shape))


def _border_reachable(mask):
    result = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for cells in _components(mask):
        if any(r in (0, h-1) or c in (0, w-1) for r, c in cells):
            result[tuple(zip(*cells))] = True
    return result


def _features(atlas, query, info, match):
    _, _, _, qbg, qink = match
    dye = _project(info['dye'], match, query.shape)
    owner = _project(info['owner'], match, query.shape)
    diagonal = _project(info['diagonal'], match, query.shape)
    filled, background = dye >= 0, query == qbg
    mapping = info['mapping']
    chain_colors = [s for s, f in mapping.items() if f in mapping and mapping[f] != f]
    return dict(dye=dye, owner=owner, filled=int(filled.sum()),
                unchanged=int(np.count_nonzero(background & ~filled)),
                off_window=int(np.count_nonzero(filled & _border_reachable(background))),
                chain=any(np.any(owner == s) for s in chain_colors),
                self_map=any(s == f and np.any(owner == s) for s, f in mapping.items()),
                diagonal=bool(np.any(diagonal & filled)),
                inert=bool(np.any((owner >= 0) & ~filled)),
                destinations=len(np.unique(dye[filled])),
                qbg=qbg, qink=qink)


def _dilate(mask):
    a = mask.copy()
    a[1:] |= mask[:-1]
    a[:-1] |= mask[1:]
    a[:, 1:] |= mask[:, :-1]
    a[:, :-1] |= mask[:, 1:]
    return a


def _shape(rng, kind, small=False):
    if kind == 'diamond':
        radius = 1 if small else int(rng.integers(1, 3))
        rr, cc = np.indices((2*radius+1, 2*radius+1))
        wall = abs(rr-radius)+abs(cc-radius) == radius
    elif kind == 'multi':
        h, w = int(rng.integers(5, 8)), int(rng.integers(3, 6))
        wall = np.ones((h, w), dtype=bool)
        wall[1:-1, 1:-1] = False
        wall[int(rng.integers(2, h-2)), :] = True
    elif kind == 'profile':
        h = 2 if small else int(rng.integers(2, 4))
        w = 2 if small else int(rng.integers(2, 4))
        hole = np.ones((h, w), dtype=bool)
        hole[-1, -1] = False
        hole = np.pad(hole, 1)
        wall = _dilate(hole) & ~hole
    else:
        h = 3 if small else int(rng.integers(3, 6))
        w = 3 if small else int(rng.integers(3, 7))
        wall = np.ones((h, w), dtype=bool)
        wall[1:-1, 1:-1] = False
        if kind == 'open':
            wall[0, int(rng.integers(1, w-1))] = False
    return _transform(wall, int(rng.integers(8)))


def _place(ids, shape, ident, r, c):
    h, w = shape.shape
    patch = ids[r:r+h, c:c+w]
    patch[shape] = ident


def _positions(ids, shape, bounds):
    r0, r1, c0, c1 = bounds
    h, w = shape.shape
    result = []
    for r in range(r0, r1-h+1):
        for c in range(c0, c1-w+1):
            if not np.any(ids[max(0, r-1):r+h+1, max(0, c-1):c+w+1]):
                result.append((r, c))
    return result


def _geometry(rng):
    H, W = int(rng.integers(12, 17)), int(rng.integers(14, 23))
    # A compact cluster makes informative windows possible; its shapes and
    # positions are sampled freely, not placed in a repeated slot lattice.
    for attempt in range(24):
        ids = np.zeros((H, W), dtype=np.int8)
        ch, cw = min(H-3, 10), 10
        r0 = int(rng.integers(3, H-ch+1))
        c0 = int(rng.integers(W-cw+1))
        shapes = [_shape(rng, 'rectangle', attempt >= 12),
                  _shape(rng, 'profile', attempt >= 12),
                  _shape(rng, 'diamond', attempt >= 12)]
        for index in sorted(range(3), key=lambda i: -shapes[i].size):
            positions = _positions(ids, shapes[index], (r0, r0+ch, c0, c0+cw))
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            _place(ids, shapes[index], index+1, r, c)
        else:
            closed, opened = [1, 2, 3], []
            kinds = list(rng.choice(['rectangle', 'profile', 'diamond', 'multi', 'open'], int(rng.integers(0, 4))))
            for kind in kinds:
                shape = _shape(rng, str(kind))
                positions = _positions(ids, shape, (3, H, 0, W))
                if positions:
                    ident = len(closed)+len(opened)+1
                    r, c = positions[int(rng.integers(len(positions)))]
                    _place(ids, shape, ident, r, c)
                    (opened if kind == 'open' else closed).append(ident)
            return ids, closed, opened
    return _fallback_geometry()


def _fallback_geometry():
    ids = np.zeros((15, 18), dtype=np.int8)
    ring = np.ones((4, 4), dtype=bool)
    ring[1:-1, 1:-1] = False
    _place(ids, ring, 1, 3, 1)
    profile = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    _place(ids, profile, 2, 4, 7)
    rr, cc = np.indices((5, 5))
    _place(ids, abs(rr-2)+abs(cc-2) == 2, 3, 8, 1)
    ring[0, 1] = False
    _place(ids, ring, 4, 10, 11)
    return ids, [1, 2, 3], [4]


def _assemble(rng, atlas, query, neutral):
    # Pack horizontally or vertically; pad only with the visibly neutral color.
    first, second = (atlas, query) if rng.integers(2) else (query, atlas)
    choices = [(axis, gap, margin) for axis in (0, 1) for gap in (1, 2) for margin in (1, 2)
               if first.shape[axis]+second.shape[axis]+gap+2*margin <= 30
               and max(first.shape[1-axis], second.shape[1-axis])+2*margin <= 30]
    axis, gap, margin = choices[int(rng.integers(len(choices)))]
    lengths = [max(first.shape[d], second.shape[d])+2*margin for d in (0, 1)]
    lengths[axis] = first.shape[axis]+second.shape[axis]+gap+2*margin
    result = np.full(tuple(lengths), neutral, dtype=np.int8)
    start = margin
    for sheet in (first, second):
        other = 1-axis
        offset = int(rng.integers(margin, lengths[other]-sheet.shape[other]-margin+1))
        pos = [0, 0]
        pos[axis], pos[other] = start, offset
        r, c = pos
        result[r:r+sheet.shape[0], c:c+sheet.shape[1]] = sheet
        start += sheet.shape[axis]+gap
    return result


def _candidate_crops(rng, shape, test):
    H, W = shape
    # Random proposals first; exhaustive bounded fallback ensures the hard
    # training witnesses need not depend on luck in proposal sampling.
    seen = set()
    for _ in range(180):
        if test:
            h, w = int(rng.integers(3, 7)), int(rng.integers(6, 11))
            if rng.integers(2):
                h, w = w, h
        else:
            h, w = int(rng.integers(8, 11)), int(rng.integers(8, 11))
        if h > H-3 or w > W:
            continue
        r, c = int(rng.integers(3, H-h+1)), int(rng.integers(W-w+1))
        item = r, c, h, w
        if item not in seen:
            seen.add(item)
            yield item
    dimensions = [(h, w) for h in range(3 if test else 8, 11) for w in range(3 if test else 8, 11)
                  if not test or (3 <= min(h, w) <= 6 and 6 <= max(h, w) <= 10)]
    rng.shuffle(dimensions)
    for h, w in dimensions:
        for r in range(3, H-h+1):
            for c in range(W-w+1):
                item = r, c, h, w
                if item not in seen:
                    yield item


def _make_pair(rng, geometry, index, split, orientation, seen):
    ids, closed, opened = geometry
    test = split == 'test'
    for retry in range(10):
        palette = list(map(int, rng.permutation(10)))
        neutral, abg, qbg, qink = palette[:4]
        first_lesson = split == 'train' and index == 0
        n = int(rng.integers(3 if first_lesson else 2, 5))
        sources, inert, final = palette[4:4+n], palette[4+n], palette[9]
        mapping = {sources[0]: sources[1], sources[1]: final}
        for source in sources[2:]:
            mapping[source] = int(rng.choice(sources+[final]))
        if first_lesson:
            mapping[sources[2]] = sources[2]
        # The first photograph explicitly teaches diagonal closure and one step
        # of a chain. Other assignments vary without altering the geometry.
        order = [3]+[i for i in closed if i != 3] if first_lesson else list(map(int, rng.permutation(closed)))
        colors = {order[0]: sources[0], order[1]: sources[2] if first_lesson else sources[1], order[2]: inert}
        for ident in closed[3:]+opened:
            if ident not in colors:
                colors[ident] = int(rng.choice(sources+[inert]))
        for ident in closed:
            if ident not in colors:
                colors[ident] = int(rng.choice(sources+[inert]))
        atlas = np.full(ids.shape, abg, dtype=np.int8)
        for ident, color in colors.items():
            atlas[ids == ident] = color
        start = int(rng.integers(1 if n == 2 else 0, ids.shape[1]-n+(0 if n == 2 else 1)))
        order_key = list(map(int, rng.permutation(n)))
        atlas[0, start:start+n] = [sources[i] for i in order_key]
        atlas[1, start:start+n] = [mapping[sources[i]] for i in order_key]
        info = _atlas(atlas)
        for r, c, h, w in _candidate_crops(rng, atlas.shape, test):
            wall = info['wall'][r:r+h, c:c+w]
            if not np.any(wall) or np.all(wall):
                continue
            query = np.where(wall, qink, qbg).astype(np.int8)
            dye = info['dye'][r:r+h, c:c+w]
            filled = dye >= 0
            if not np.any(filled) or not np.any(~wall & ~filled):
                continue
            if not np.any(filled & _border_reachable(~wall)):
                continue
            owner = info['owner'][r:r+h, c:c+w]
            if not test and index == 0:
                if (len(np.unique(dye[filled])) < 2 or not np.any(owner == sources[0])
                        or not np.any(owner == sources[2])
                        or not np.any(info['diagonal'][r:r+h, c:c+w] & filled)):
                    continue
            if not test and index == 1 and not np.any((owner >= 0) & ~filled):
                continue
            # Check the canonical query before any display transformations.
            if len(_matches(info, query)) != 1:
                continue
            display = int(rng.integers(8))
            # Transform the photograph and atlas in the same display frame.
            # This keeps the first lesson an actual identity match and the
            # second an actual quarter turn, regardless of atlas presentation.
            output = _transform(_transform(_paint(query, dye), orientation), display)
            query = _transform(_transform(query, orientation), display)
            atlas_display = _transform(atlas, display)
            inp = _assemble(rng, atlas_display, query, neutral)
            signature = _key(inp)
            if signature in seen:
                continue
            seen.add(signature)
            return dict(input=inp, output=output)
    raise ValueError('no informative unique photograph for this geometry')


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(name+' is outside its legal integer range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    for attempt in range(9):
        geometry = _geometry(rng) if attempt < 8 else _fallback_geometry()
        train_orientations = [0, 1]+([4] if nt >= 3 else [])
        pool = list(range(8 if nt >= 3 else 4))
        train_orientations += list(map(int, rng.choice(pool, nt-len(train_orientations))))
        test_orientations = list(map(int, rng.choice(pool, nq)))
        puzzle, seen = {'train': [], 'test': []}, set()
        try:
            for split, orientations in (('train', train_orientations), ('test', test_orientations)):
                for index, orientation in enumerate(orientations):
                    puzzle[split].append(_make_pair(rng, geometry, index, split, orientation, seen))
        except ValueError:
            continue
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError('bounded construction failed')


def _canonical_wall(wall):
    return min(_key(_transform(wall, i)) for i in range(8))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError('puzzle must have exactly train/test')
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError('splits must be lists')
    nt, nq = _count('num_train', len(puzzle['train']), 2, 10), _count('num_test', len(puzzle['test']), 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    seen, references = set(), set()
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError('pair must have exactly input/output')
            inp, out = _array(pair['input']), _array(pair['output'])
            if _key(inp) in seen:
                raise ValueError('duplicate input')
            seen.add(_key(inp))
            atlas, query, neutral = _sheets(inp)
            info = _atlas(atlas)
            matches = _matches(info, query)
            if len(matches) != 1:
                raise ValueError('nonunique photograph')
            match = matches[0]
            feature = _features(atlas, query, info, match)
            expected = _paint(query, feature['dye'])
            if not np.array_equal(out, expected):
                raise ValueError('incorrect developed photograph')
            if out.shape != query.shape or not np.array_equal(out[query == feature['qink']], query[query == feature['qink']]):
                raise ValueError('photograph dimensions or ink changed')
            if not all(feature[k] > 0 for k in ('filled', 'unchanged', 'off_window')):
                raise ValueError('missing parent-interaction witness')
            if split == 'train' and index == 0:
                if not feature['chain'] or not feature['self_map'] or not feature['diagonal'] or feature['destinations'] < 2:
                    raise ValueError('missing one-step/self-map/diagonal/multiple-dye lesson')
                if match[0] != 0:
                    raise ValueError('first lesson must demonstrate an identity match')
            if split == 'train' and index == 1 and not feature['inert']:
                raise ValueError('missing unlisted-cavity lesson')
            if split == 'train' and index == 1 and match[0] not in (1, 3):
                raise ValueError('second lesson must demonstrate a quarter turn')
            if split == 'train' and index == 2 and match[0] < 4:
                raise ValueError('third lesson must demonstrate reflection')
            if split == 'test' and nt == 2 and match[0] >= 4:
                raise ValueError('reflection needs a training demonstration')
            references.add(_canonical_wall(info['wall']))
    if len(references) != 1:
        raise ValueError('atlas geometry must be shared within a puzzle')


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; retain hashes/features, not reconstructible grid corpora."""
    signatures, references, shapes, query_shapes, sides, backgrounds = set(), set(), set(), set(), set(), set()
    counts, hole_shapes, placements, key_lengths = set(), set(), set(), set()
    totals = dict(pairs=0, off_window_cells=0, diagonal_pairs=0, inert_pairs=0, chain_pairs=0)
    first_signature = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        signatures.add(signature)
        if seed == 0:
            first_signature = signature
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                atlas, query, neutral = _sheets(inp)
                info = _atlas(atlas)
                match = _matches(info, query)[0]
                f = _features(atlas, query, info, match)
                assert np.array_equal(solve(inp), pair['output'])
                references.add(_canonical_wall(info['wall']))
                shapes.add(atlas.shape)
                query_shapes.add(query.shape)
                sides.add(info['side'])
                key_lengths.add(len(info['mapping']))
                backgrounds.add(info['bg'])
                # Count all separated monochromatic outlines, excluding key.
                counts.add(len(_components(info['wall'], diagonal=True)))
                placements.add(match[1:3])
                for _, cells in info['regions']:
                    rr, cc = zip(*cells)
                    hole_shapes.add(tuple(sorted((r-min(rr), c-min(cc)) for r, c in cells)))
                totals['pairs'] += 1
                totals['off_window_cells'] += f['off_window']
                totals['diagonal_pairs'] += int(f['diagonal'])
                totals['inert_pairs'] += int(f['inert'])
                totals['chain_pairs'] += int(f['chain'])
                # Explicit omission checks. These are counterfactual policies,
                # not another solver or an independent correctness oracle.
                assert not np.array_equal(query, pair['output']), 'omitting B did not change the answer'
                local_only = pair['output'].copy()
                local_only[(f['dye'] >= 0) & _border_reachable(query == f['qbg'])] = f['qbg']
                assert not np.array_equal(local_only, pair['output']), 'omitting A did not change the answer'
    assert len(signatures) == 42 and len(references) >= 35
    assert len(shapes) >= 15 and len(query_shapes) >= 20 and len(hole_shapes) >= 10
    assert sides == {'top', 'bottom', 'left', 'right'} and backgrounds == set(range(10))
    assert key_lengths == {2, 3, 4}
    assert len(counts) >= 3 and len(placements) >= 25
    assert first_signature == _signature(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1223, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(12, 2, 1)
    for t in range(8):
        for pair in p['train']+p['test']:
            # Global canvas transformation also rotates the photograph/output.
            assert np.array_equal(solve(_transform(pair['input'], t)), _transform(pair['output'], t))
    bad = generate_puzzle(0)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt answer accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), reference_classes=len(references),
                atlas_shapes=len(shapes), photograph_shapes=len(query_shapes),
                contour_counts=sorted(counts), hole_shapes=len(hole_shapes),
                crop_offsets=len(placements), key_lengths=sorted(key_lengths),
                key_sides=sorted(sides), **totals)
