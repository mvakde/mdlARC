"""Count-controlled, category-specific wavefronts on a periodic 3x3 mosaic.

Only standard-library Python and NumPy are used.  The solver reads each input
alone; no generated state or training conventions are needed.
"""
from __future__ import annotations

import numpy as np


def _components(mask, diagonal=True):
    pending = {tuple(map(int, p)) for p in np.argwhere(mask)}
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while pending:
        start = min(pending)
        pending.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _make_vocabularies():
    masks, shapes = [], set()
    for code in range(1, 512):
        mask = np.array([(code >> i) & 1 for i in range(9)], bool).reshape(3, 3)
        if code < 511 and all((mask[0].any(), mask[-1].any(),
                               mask[:, 0].any(), mask[:, -1].any())):
            masks.append(mask)
        components = _components(mask)
        if len(components) == 1:
            shapes.add(_normalize(components[0]))
    return np.array(masks), tuple(sorted(shapes))


_MASKS, _SHAPES = _make_vocabularies()
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_SINGLE = ((0, 0),)
_PANEL_SHAPES = tuple((h, w) for h in range(9, 20, 2) for w in range(9, 20, 2))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _coordinates(shape):
    top, left = ((n-3)//2 for n in shape)
    rows, cols = np.indices(shape)
    tr, tc = (rows-top)//3, (cols-left)//3
    return (rows-top) % 3, (cols-left) % 3, tr, tc


def _render(motif, shape, field, background, counts, square=False, solid=False, circle=False):
    pr, pc, tr, tc = _coordinates(shape)
    repeated = motif[pr, pc]
    distance = np.maximum(abs(tr), abs(tc)) if square else abs(tr) + abs(tc)
    out = np.full(shape, field, np.int8)
    for color, n in counts.items():
        occupied = repeated == color
        out[occupied] = background
        inside = tr*tr + tc*tc <= (n-1)*(n-1) if circle else distance < n
        out[occupied & inside] = color
    if solid:
        # Concrete shortcut: repaint carrier gaps in the maximum-count front.
        out[(repeated == field) & (distance < max(counts.values()))] = next(iter(counts))
    return out


def _has_no_symmetry(motif):
    images = [np.rot90(motif, k) for k in range(1, 4)]
    images += [np.rot90(np.fliplr(motif), k) for k in range(4)]
    return all(not np.array_equal(motif, other) for other in images)


def _find_panel(grid):
    candidates = []
    for field in map(int, np.unique(grid)):
        coords = np.argwhere(grid == field)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = (int(x)+1 for x in coords.max(axis=0))
        panel = grid[r0:r1, c0:c1]
        if panel.shape not in _PANEL_SHAPES:
            continue
        if not all(np.all(edge == field) for edge in
                   (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        occupied = np.argwhere(panel != field)
        if not 2 <= len(occupied) <= 8:
            continue
        top, left = ((n-3)//2 for n in panel.shape)
        if (tuple(occupied.min(axis=0)) != (top, left)
                or tuple(occupied.max(axis=0)) != (top+2, left+2)):
            continue
        motif = panel[top:top+3, left:left+3].copy()
        keys = tuple(sorted(set(map(int, motif.flat)) - {field}))
        if not 1 <= len(keys) <= 4:
            continue
        outside = np.ones(grid.shape, bool)
        outside[r0:r1, c0:c1] = False
        if not outside.any():
            continue
        frequencies = np.bincount(grid[outside], minlength=10)
        background = int(np.argmax(frequencies))
        if background in keys or background == field or frequencies[background] <= .55*outside.sum():
            continue
        ring = np.zeros(grid.shape, bool)
        ring[max(0, r0-1):r1+1, max(0, c0-1):c1+1] = True
        ring[r0:r1, c0:c1] = False
        if not np.all(grid[ring] == background):
            continue
        candidates.append(dict(field=field, background=background, bounds=(r0, r1, c0, c1),
                               shape=panel.shape, motif=motif, keys=keys, outside=outside))
    if len(candidates) != 1:
        raise ValueError("Input must identify exactly one centered legend panel")
    return candidates[0]


def _analyze(input_grid):
    grid = _grid(input_grid)
    m = _find_panel(grid)
    owner = np.full(grid.shape, -1, int)
    objects, components, serial = {}, {}, 0
    colors = sorted(set(map(int, grid[m['outside']])) - {m['background']})
    for color in colors:
        comps = _components((grid == color) & m['outside'])
        components[color] = comps
        objects[color] = tuple(_normalize(comp) for comp in comps)
        for comp in comps:
            for r, c in comp:
                owner[r, c] = serial
            serial += 1
    for r, c in np.argwhere(owner >= 0):
        around = owner[max(0, r-1):r+2, max(0, c-1):c+2]
        if np.any((around >= 0) & (around != owner[r, c])):
            raise ValueError("Different objects must not touch, including diagonally")
    counts = {c: len(objects.get(c, ())) for c in m['keys']}
    if not all(1 <= n <= 6 for n in counts.values()):
        raise ValueError("Keyed object counts must be 1..6")
    pr, pc, tr, tc = _coordinates(m['shape'])
    repeated, dist = m['motif'][pr, pc], abs(tr) + abs(tc)
    for color, n in counts.items():
        if not (np.any((repeated == color) & (dist == n-1))
                and np.any((repeated == color) & (dist == n))):
            raise ValueError("Each count must have a visible boundary in the panel")
    m.update(objects=objects, components=components, counts=counts,
             distractors=tuple(c for c in colors if c not in counts),
             density=int(np.sum(m['motif'] != m['field'])),
             center=bool(m['motif'][1, 1] != m['field']))
    return m


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive counts, colors, size, and phase solely from the visible input."""
    m = _analyze(input_grid)
    return _render(m['motif'], m['shape'], m['field'], m['background'], m['counts'])


def _compatible_shapes(motif, field, counts, regime, train):
    result = []
    for shape in _PANEL_SHAPES:
        if regime == 'aspect':
            if train and abs(shape[0]-shape[1]) > 2:
                continue
            if not train and max(shape)/min(shape) < 1.6:
                continue
        pr, pc, tr, tc = _coordinates(shape)
        repeated, dist = motif[pr, pc], abs(tr) + abs(tc)
        if all(np.any((repeated == c) & (dist == n-1))
               and np.any((repeated == c) & (dist == n)) for c, n in counts.items()):
            result.append(shape)
    return result


def _motif(rng, keys, field, regime, train, center, asymmetric):
    if regime == 'denser':
        low, high = (3, 5) if train else (6, 8)
    elif regime == 'sparser':
        low, high = (5, 8) if train else (2, 4)
    else:
        low, high = 2, 8
    eligible = [m for m in _MASKS if max(low, len(keys), 3 if asymmetric else 2) <= m.sum() <= high
                and (center is None or bool(m[1, 1]) == center)]
    if not eligible:
        raise AssertionError("Empty mask regime")
    for _ in range(100):
        density = int(rng.choice(sorted({int(m.sum()) for m in eligible})))
        choices = [m for m in eligible if m.sum() == density]
        mask = choices[int(rng.integers(len(choices)))]
        cells = np.argwhere(mask)
        assignments = list(keys) + [int(rng.choice(keys)) for _ in range(len(cells)-len(keys))]
        rng.shuffle(assignments)
        motif = np.full((3, 3), field, np.int8)
        for (r, c), color in zip(cells, assignments):
            motif[r, c] = color
        if not asymmetric or _has_no_symmetry(motif):
            return motif
    # This fallback is a finite search, preserving every requested restriction.
    for mask in eligible:
        cells = np.argwhere(mask)
        for offset in range(len(cells)):
            motif = np.full((3, 3), field, np.int8)
            for i, (r, c) in enumerate(cells):
                motif[r, c] = keys[(i+offset) % len(keys)]
            if not asymmetric or _has_no_symmetry(motif):
                return motif
    raise AssertionError("No asymmetric legend compatible with the regime")


def _object_shape(rng):
    # Balance areas rather than allowing medium-density masks to dominate.
    area = int(rng.integers(1, 10))
    choices = [s for s in _SHAPES if len(s) == area]
    return choices[int(rng.integers(len(choices)))]


def _place(grid, blocked, color, shape, rng):
    sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
    nh, nw = grid.shape[0]-sh+1, grid.shape[1]-sw+1
    if min(nh, nw) <= 0:
        return False
    allowed = np.ones((nh, nw), bool)
    for r, c in shape:
        allowed &= ~blocked[r:r+nh, c:c+nw]
    candidates = np.argwhere(allowed)
    if not len(candidates):
        return False
    r0, c0 = map(int, candidates[int(rng.integers(len(candidates)))])
    for dr, dc in shape:
        r, c = r0+dr, c0+dc
        grid[r, c] = color
        blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
    return True


def _make_pair(rng, spec):
    field, bg, motif, shape = (spec[k] for k in ('field', 'background', 'motif', 'shape'))
    ph, pw = shape
    counts, objects = spec['counts'], []
    for color, n in list(counts.items()) + list(spec['ignored'].items()):
        prototype = _object_shape(rng)
        for i in range(n):
            glyph = _object_shape(rng) if rng.random() < .75 else prototype
            if spec['witness'] and color == spec['witness_color']:
                glyph = (_DIAMOND, _SQUARE, _SINGLE)[i]
            objects.append((color, glyph))
    rng.shuffle(objects)
    objects.sort(key=lambda item: -len(item[1]))
    out = _render(motif, shape, field, bg, counts)
    for attempt in range(7):
        fallback = attempt == 6
        if attempt >= 3:
            h = w = 30
        else:
            area = ph*pw + 13*len(objects)
            low_h = min(30, max(ph+1, 12, int(np.sqrt(area*.7))))
            h = int(rng.integers(low_h, 31))
            low_w = min(30, max(pw+1, 12, int(area*.7/h)))
            w = int(rng.integers(low_w, 31))
        if fallback:
            r0 = 0 if rng.integers(2) else h-ph
            c0 = 0 if rng.integers(2) else w-pw
        elif rng.random() < .45:
            side = int(rng.integers(4))
            if side < 2:
                r0, c0 = (0 if side == 0 else h-ph), int(rng.integers(w-pw+1))
            else:
                r0, c0 = int(rng.integers(h-ph+1)), (0 if side == 2 else w-pw)
        else:
            r0, c0 = int(rng.integers(h-ph+1)), int(rng.integers(w-pw+1))
        grid = np.full((h, w), bg, np.int8)
        grid[r0:r0+ph, c0:c0+pw] = field
        top, left = ((n-3)//2 for n in shape)
        grid[r0+top:r0+top+3, c0+left:c0+left+3] = motif
        blocked = np.zeros(grid.shape, bool)
        blocked[max(0, r0-1):r0+ph+1, max(0, c0-1):c0+pw+1] = True
        if fallback:
            slots = [(r, c) for r in range(0, 28, 4) for c in range(0, 28, 4)
                     if not blocked[r:r+3, c:c+3].any()]
            if len(slots) < len(objects):
                raise AssertionError("Slot bound failed")
            rng.shuffle(slots)
            for (color, glyph), (r, c) in zip(objects, slots):
                for dr, dc in glyph:
                    grid[r+dr, c+dc] = color
        elif not all(_place(grid, blocked, color, glyph, rng) for color, glyph in objects):
            continue
        parsed = _analyze(grid)
        if parsed['counts'] != counts:
            raise AssertionError("Object placement changed counts")
        if not np.array_equal(out, solve(grid)):
            raise AssertionError("Construction label disagrees with visible input")
        return {'input': grid, 'output': out.copy()}
    raise AssertionError("Bounded placement exhausted")


def _check_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(name + ' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(('counts', 'denser', 'sparser', 'center', 'aspect')))
    palette = list(map(int, rng.permutation(10)))
    witness_color = palette[2]
    consistent_palette = bool(rng.random() < .25)
    first_center = bool(rng.integers(2))
    puzzle, seen, previous_motifs = {'train': [], 'test': []}, set(), set()
    for split, number in (('train', nt), ('test', nq)):
        train = split == 'train'
        for index in range(number):
            witness = train and index == 0
            role_witness = train and index == 1
            k = 2 if witness else int(rng.integers(2, 4)) if role_witness else int(rng.integers(1, 5))
            if regime == 'sparser' and not train:
                k = int(rng.integers(1, 4))
            if consistent_palette:
                bg, field = palette[:2]
            else:
                available = [c for c in range(10) if not ((witness or role_witness) and c == witness_color)]
                bg, field = map(int, rng.choice(available, 2, replace=False))
            pool = [c for c in range(10) if c not in (bg, field)]
            rng.shuffle(pool)
            if witness:
                pool.remove(witness_color)
                pool.insert(0, witness_color)
            if role_witness:
                pool.remove(witness_color)
            keys = pool[:k]
            rest = pool[k:]
            if role_witness:
                rest.insert(0, witness_color)
            cap = 4 if regime == 'counts' and train or regime == 'aspect' else 5
            counts = {c: int(rng.integers(1, cap+1)) for c in keys}
            if witness:
                counts[keys[0]], counts[keys[1]] = 3, 2
            elif role_witness:
                counts[keys[0]], counts[keys[1]] = 1, 4
            if regime == 'counts' and not train:
                counts[keys[0]] = int(rng.integers(5, 7))
            nd = int(rng.integers(1 if role_witness else 0, 3))
            ignored = {c: int(rng.integers(1, 7)) for c in rest[:nd]}
            if role_witness:
                ignored[witness_color] = int(rng.integers(3, 7))
            while sum(counts.values()) + sum(ignored.values()) > 24:
                reducible = [c for c, n in ignored.items() if n > (3 if role_witness and c == witness_color else 1)]
                if reducible:
                    c = int(rng.choice(reducible))
                    ignored[c] -= 1
                else:
                    reducible = [c for c, n in counts.items() if n > 1 and not
                                 (c == keys[0] and regime == 'counts' and not train)]
                    c = int(rng.choice(reducible))
                    counts[c] -= 1
            if regime == 'center':
                center = train
            elif witness:
                center = first_center
            elif role_witness:
                center = not first_center
            else:
                center = bool(rng.integers(2))
            for _ in range(100):
                motif = _motif(rng, keys, field, regime, train, center, witness)
                normalized = tuple(-1 if v == field else keys.index(int(v)) for v in motif.flat)
                shapes = _compatible_shapes(motif, field, counts, regime, train)
                if role_witness:
                    # Count four distinguishes taxicab radius three from a
                    # Euclidean circle: tile (2,2) must be beyond the front.
                    shapes = [s for s in shapes if min(s) >= 15]
                if shapes and normalized not in previous_motifs:
                    break
            else:
                raise AssertionError("Compatible distinct motif selection exhausted")
            shape = shapes[int(rng.integers(len(shapes)))]
            spec = dict(field=field, background=bg, counts=counts, ignored=ignored,
                        motif=motif, shape=shape, witness=witness, witness_color=witness_color)
            for _ in range(8):
                pair = _make_pair(rng, spec)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("Could not create a distinct input")
            seen.add(_key(pair['input']))
            previous_motifs.add(normalized)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(grid, m):
    motif, shape, field, bg, counts = (m[k] for k in ('motif', 'shape', 'field', 'background', 'counts'))
    orthogonal = {c: len(_components((grid == c) & m['outside'], False)) for c in counts}
    areas = {c: sum(len(comp) for comp in m['components'][c]) for c in counts}
    with_keys = {c: n + int(np.sum(motif == c)) for c, n in counts.items()}
    once = np.full(shape, field, np.int8)
    top, left = ((n-3)//2 for n in shape)
    once[top:top+3, left:left+3] = motif
    return {'four_neighbors': _render(motif, shape, field, bg, orthogonal),
            'pixel_area': _render(motif, shape, field, bg, areas),
            'include_keys': _render(motif, shape, field, bg, with_keys),
            'uniform_reach': _render(motif, shape, field, bg, {c: max(counts.values()) for c in counts}),
            'square_distance': _render(motif, shape, field, bg, counts, square=True),
            'round_distance': _render(motif, shape, field, bg, counts, circle=True),
            'fill_gaps': _render(motif, shape, field, bg, counts, solid=True),
            'no_repeat': once,
            'ignore_counts': _render(motif, shape, field, bg, {c: 100 for c in counts})}


def validate_generated_puzzle(puzzle):
    """Validate returned arrays, geometry, semantic boundaries, and witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos, rejected = set(), {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = _analyze(inp)
            assert len(m['distractors']) <= 2
            assert sum(len(v) for v in m['objects'].values()) <= 24
            assert all(1 <= len(v) <= 6 for v in m['objects'].values())
            expected = _render(m['motif'], m['shape'], m['field'], m['background'], m['counts'])
            assert np.array_equal(out, expected)
            assert np.array_equal(solve(inp.copy()), out)
            pr, pc, tr, tc = _coordinates(m['shape'])
            repeated, dist = m['motif'][pr, pc], abs(tr)+abs(tc)
            assert np.array_equal(out == m['field'], repeated == m['field'])
            for c, n in m['counts'].items():
                assert np.array_equal(out == c, (repeated == c) & (dist < n))
            infos[split].append(m)
            if split == 'train':
                rejected.update(name for name, candidate in _alternatives(inp, m).items()
                                if not np.array_equal(candidate, out))
    train, test = infos['train'], infos['test']
    assert any(all(s in m['objects'][c] for s in (_DIAMOND, _SQUARE, _SINGLE))
               for m in train for c in m['keys'])
    assert any(1 in m['counts'].values() for m in train)
    assert any(2 in m['counts'].values() for m in train)
    assert any(4 in m['counts'].values() and min(m['shape']) >= 15 for m in train)
    assert any(len(set(m['counts'].values())) > 1 for m in train)
    assert any(_has_no_symmetry(m['motif']) for m in train)
    assert any(set(a['keys']) & set(b['distractors']) for a in train for b in train)
    assert rejected == {'four_neighbors', 'pixel_area', 'include_keys', 'uniform_reach',
                        'square_distance', 'round_distance', 'fill_gaps', 'no_repeat', 'ignore_counts'}
    shifted = (min(max(m['counts'].values()) for m in test) > max(max(m['counts'].values()) for m in train)
               or min(m['density'] for m in test) > max(m['density'] for m in train)
               or max(m['density'] for m in test) < min(m['density'] for m in train)
               or all(m['center'] for m in train) and all(not m['center'] for m in test)
               or all(abs(m['shape'][0]-m['shape'][1]) <= 2 for m in train)
               and all(max(m['shape'])/min(m['shape']) >= 1.6 for m in test))
    assert shifted, 'An observable train/test shift is required'


def _fingerprint(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _diversity_metrics(puzzles):
    legends, glyphs, shapes, panels, colors, counts, outputs, structures = set(), set(), set(), set(), set(), set(), set(), set()
    densities, centers, categories, placements = set(), set(), set(), set()
    shifts = set()
    for puzzle in puzzles:
        split_infos = {}
        for split in ('train', 'test'):
            split_infos[split] = []
            for pair in puzzle[split]:
                m = _analyze(pair['input'])
                split_infos[split].append(m)
                mapping = {m['field']: -1}
                normalized = tuple(mapping.setdefault(int(v), len(mapping)-1) for v in m['motif'].flat)
                legends.add(normalized)
                glyphs.update(s for group in m['objects'].values() for s in group)
                shapes.add(tuple(pair['input'].shape))
                panels.add(m['shape'])
                colors.add((m['background'], m['field']))
                counts.update(m['counts'].values())
                outputs.add(_key(pair['output']))
                roles = {}
                structures.add((pair['output'].shape, tuple(roles.setdefault(int(v), len(roles))
                                                           for v in pair['output'].flat)))
                densities.add(m['density'])
                centers.add(m['center'])
                categories.add(len(m['keys']))
                r0, r1, c0, c1 = m['bounds']
                placements.add((r0 == 0, r1 == pair['input'].shape[0],
                                c0 == 0, c1 == pair['input'].shape[1]))
        train, test = split_infos['train'], split_infos['test']
        if max(max(m['counts'].values()) for m in test) > max(max(m['counts'].values()) for m in train):
            shifts.add('counts')
        if min(m['density'] for m in test) > max(m['density'] for m in train):
            shifts.add('denser')
        if max(m['density'] for m in test) < min(m['density'] for m in train):
            shifts.add('sparser')
        if all(m['center'] for m in train) and all(not m['center'] for m in test):
            shifts.add('center')
        if all(abs(m['shape'][0]-m['shape'][1]) <= 2 for m in train) and all(max(m['shape'])/min(m['shape']) >= 1.6 for m in test):
            shifts.add('aspect')
    return dict(puzzles=len(puzzles), distinct_puzzles=len({_fingerprint(p) for p in puzzles}),
                legends=len(legends), object_shapes=len(glyphs), input_shapes=len(shapes),
                panel_shapes=len(panels), palette_roles=len(colors), counts=sorted(counts),
                output_grids=len(outputs), output_structures=len(structures), densities=sorted(densities),
                center_values=sorted(centers), category_counts=sorted(categories),
                placement_classes=len(placements), shifts=sorted(shifts))


def self_test():
    """Bounded semantic checks and 42-seed variation measurement."""
    assert len(_MASKS) == 321
    examples = []
    for seed in range(42):
        nt = 2 + seed % 9
        nq = 1 + (seed // 3) % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, nq)
        validate_generated_puzzle(puzzle)
        examples.append(puzzle)
    metrics = _diversity_metrics(examples)
    assert metrics['distinct_puzzles'] == 42
    assert metrics['legends'] >= 120 and metrics['object_shapes'] >= 100
    assert metrics['input_shapes'] >= 40 and metrics['panel_shapes'] >= 20
    assert metrics['palette_roles'] >= 35 and metrics['output_structures'] >= 150
    assert metrics['counts'] == list(range(1, 7))
    assert metrics['densities'] == list(range(2, 9))
    assert metrics['center_values'] == [False, True]
    assert metrics['category_counts'] == [1, 2, 3, 4]
    assert set(metrics['shifts']) == {'counts', 'denser', 'sparser', 'center', 'aspect'}
    assert _fingerprint(examples[7]) == _fingerprint(generate_puzzle(7, 9, 3))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(2026+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Corruption is detected even if only a carrier cell changes.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in examples[0][s]] for s in ('train', 'test')}
    out = bad['test'][0]['output']
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Output corruption accepted')
    # Visible-grid metamorphic check: remove the singleton from the count-three
    # witness. Precisely that category's distance-two layer must disappear.
    pair = next(p for p in examples[0]['train'] if any(
        all(s in group for s in (_DIAMOND, _SQUARE, _SINGLE))
        for c, group in _analyze(p['input'])['objects'].items()
        if c in _analyze(p['input'])['keys']))
    m = _analyze(pair['input'])
    color = next(c for c in m['keys'] if all(s in m['objects'][c] for s in (_DIAMOND, _SQUARE, _SINGLE)))
    singleton = next(comp for comp in m['components'][color] if len(comp) == 1)
    changed = pair['input'].copy()
    changed[singleton[0]] = m['background']
    prediction = solve(changed)
    pr, pc, tr, tc = _coordinates(m['shape'])
    layer = (m['motif'][pr, pc] == color) & (abs(tr)+abs(tc) == 2)
    assert layer.any()
    assert np.array_equal(prediction != pair['output'], layer)
    assert np.all(prediction[layer] == m['background'])
    # Move one legend pixel into a gap without changing any exterior object.
    # Its old and new residues must change throughout the entire periodic field,
    # including the background-colored portions outside its counted front.
    moved = False
    r0, r1, c0, c1 = m['bounds']
    top, left = ((n-3)//2 for n in m['shape'])
    for old in np.argwhere(m['motif'] != m['field']):
        for new in np.argwhere(m['motif'] == m['field']):
            altered = pair['input'].copy()
            old_r, old_c = map(int, old)
            new_r, new_c = map(int, new)
            color = int(m['motif'][old_r, old_c])
            altered[r0+top+old_r, c0+left+old_c] = m['field']
            altered[r0+top+new_r, c0+left+new_c] = color
            try:
                info = _analyze(altered)
            except ValueError:
                continue
            assert info['counts'] == m['counts']
            answer = solve(altered)
            footprint = ((pr == old_r) & (pc == old_c)) | ((pr == new_r) & (pc == new_c))
            assert np.array_equal(answer != pair['output'], footprint)
            moved = True
            break
        if moved:
            break
    assert moved, 'A legal motif-residue relocation must be exercised'
    return metrics


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
