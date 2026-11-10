"""Keyed self-similar fields whose intersecting ink cancels.

The library, its color key, and a sparse coarse clue board are all input-visible.
No parent generator, saved example, or seed is consulted by solve().
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} outside allowed range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _turn(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _origins(mask, rows, cols, clues):
    h, w = mask.shape
    return [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)
            if all(0 <= y-r < h and 0 <= x-c < w and mask[y-r, x-c]
                   for y, x in clues)]


def _parse(input_grid):
    """Recover framing, roles, masks, swatches and unique translations."""
    a = _grid(input_grid)
    sep = int(a[0, 0])
    assert np.all(a[0] == sep), "top registration line missing"
    lines = np.flatnonzero(np.all(a == sep, axis=1))
    assert len(lines) >= 3
    h = int(lines[1]) - 1
    assert 2 <= h <= 4 and a.shape[0] >= h + 9
    assert np.all(a[h+3] == sep)
    assert len(np.unique(a[h+4])) == 1, "spacer must identify the background"
    bg = int(a[h+4, 0])
    assert bg != sep
    gaps = np.flatnonzero(np.all(a[1:h+1] == sep, axis=0)).tolist()
    assert len(gaps) >= 3 and gaps[0] == 0
    w = gaps[1] - 1
    k = len(gaps) - 1
    assert 2 <= w <= 4 and 2 <= k <= 4
    assert gaps == [i*(w+1) for i in range(k+1)], "irregular library panels"
    library_width = k*(w+1)+1
    assert np.all(a[1:h+1, library_width:] == bg)
    assert np.all(a[h+2, gaps] == sep)
    assert np.all(a[h+2, library_width:] == bg)
    labels, masks, targets = [], [], []
    for i in range(k):
        x = 1+i*(w+1)
        tile = a[1:h+1, x:x+w]
        colors = np.unique(tile[tile != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != sep and label not in labels
        mask = tile == label
        assert 3 <= int(mask.sum()) < h*w
        swatch = a[h+2, x:x+w]
        assert np.all(swatch == swatch[0]) and swatch[0] != bg
        labels.append(label)
        masks.append(mask.copy())
        targets.append(int(swatch[0]))
    start = h+5
    cols = 0
    while cols < a.shape[1] and a[start, cols] == sep:
        cols += 1
    cols -= 2
    rows = a.shape[0] - h - 7
    assert rows >= h and cols >= w
    assert rows*h <= 30 and cols*w <= 30
    assert a.shape[1] == max(library_width, cols+2)
    assert np.all(a[start, cols+2:] == bg)
    assert np.all(a[-1, :cols+2] == sep) and np.all(a[-1, cols+2:] == bg)
    assert np.all(a[start+1:-1, (0, cols+1)] == sep)
    assert np.all(a[start+1:-1, cols+2:] == bg)
    board = a[start+1:-1, 1:cols+1].copy()
    assert set(map(int, np.unique(board))) <= {bg, *labels}
    clues, origins = [], []
    for label, mask in zip(labels, masks):
        positions = tuple(map(tuple, np.argwhere(board == label).tolist()))
        assert 2 <= len(positions) < int(mask.sum()), "each field needs a strict fragment"
        candidates = _origins(mask, rows, cols, positions)
        assert len(candidates) == 1, "clues must determine exactly one translation"
        clues.append(positions)
        origins.append(candidates[0])
    return dict(h=h, w=w, rows=rows, cols=cols, k=k, bg=bg, sep=sep,
                labels=labels, masks=masks, targets=targets, clues=clues,
                origins=origins, board=board)


def _render(p, *, visible=False, source_colors=False, mode="cancel",
            stamp_turn=0, field_turn=0, solid=False):
    """Blockwise solver and counterfactuals; generation uses Kronecker products."""
    h, w, rows, cols = p['h'], p['w'], p['rows'], p['cols']
    count = np.zeros((rows*h, cols*w), dtype=np.int8)
    coarse_count = np.zeros((rows, cols), dtype=np.int8)
    owner = np.full(count.shape, p['bg'], dtype=np.int8)
    footprints = []
    for i, mask in enumerate(p['masks']):
        stamp = np.ones_like(mask) if solid else _turn(mask, stamp_turn)
        assert stamp.shape == (h, w)
        if visible:
            positions = p['clues'][i]
        else:
            shape = _turn(mask, field_turn)
            assert shape.shape == (h, w)
            r0, c0 = p['origins'][i]
            positions = [(r0+int(r), c0+int(c)) for r, c in np.argwhere(shape)]
        footprint = np.zeros(count.shape, dtype=bool)
        for r, c in positions:
            footprint[r*h:(r+1)*h, c*w:(c+1)*w] = stamp
            coarse_count[r, c] += 1
        count += footprint
        owner[footprint] = p['labels'][i] if source_colors else p['targets'][i]
        footprints.append(footprint)
    result = np.full(count.shape, p['bg'], dtype=np.int8)
    if mode == 'cancel':
        kept = count == 1
    elif mode == 'odd':
        kept = count % 2 == 1
    elif mode == 'coarse':
        kept = (count == 1) & np.repeat(np.repeat(coarse_count == 1, h, axis=0), w, axis=1)
    elif mode == 'colors':
        color_count = np.zeros_like(count)
        colors = p['labels'] if source_colors else p['targets']
        for target in set(colors):
            color_count += np.logical_or.reduce([f for f, t in zip(footprints, colors) if t == target])
        kept = color_count == 1
    else:
        kept = count > 0
    result[kept] = owner[kept]
    return result, count, footprints


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the answer entirely from the framed input."""
    return _render(_parse(input_grid))[0]


@lru_cache(maxsize=16)
def _subset_codes(n):
    codes = np.arange(1, 1 << n, dtype=np.uint32)
    counts = np.zeros(len(codes), dtype=np.uint8)
    for bit in range(n):
        counts += ((codes >> bit) & 1).astype(np.uint8)
    codes.flags.writeable = counts.flags.writeable = False
    return codes, counts


def _fragments(mask, rows, cols, origin):
    """All strict fragments with >=2 clues and no competing fitting origin."""
    active = np.argwhere(mask)
    codes, counts = _subset_codes(len(active))
    valid = codes[(counts >= 2) & (counts < len(active))].copy()
    h, w = mask.shape
    for r in range(rows-h+1):
        for c in range(cols-w+1):
            if (r, c) == origin:
                continue
            cover = 0
            for i, (y, x) in enumerate(active):
                dy, dx = int(y)+origin[0]-r, int(x)+origin[1]-c
                if 0 <= dy < h and 0 <= dx < w and mask[dy, dx]:
                    cover |= 1 << i
            if cover:
                valid = valid[(valid & np.uint32(cover)) != valid]
    return active, valid


def _choose_clues(rng, masks, rows, cols, origins, sparse):
    """Choose separately alignable fragments that share no encoded board cell."""
    options = [_fragments(m, rows, cols, o) for m, o in zip(masks, origins)]
    if any(len(codes) == 0 for active, codes in options):
        return None
    for attempt in range(48):
        used, chosen = set(), [None]*len(masks)
        order = rng.permutation(len(masks))
        for i in map(int, order):
            active, codes = options[i]
            r0, c0 = origins[i]
            blocked = sum(1 << j for j, (r, c) in enumerate(active)
                          if (r0+int(r), c0+int(c)) in used)
            valid = codes[(codes & np.uint32(blocked)) == 0]
            if len(valid) == 0:
                break
            sizes = np.array([int(code).bit_count() for code in valid], dtype=np.int8)
            available = np.unique(sizes)
            if sparse:
                size = int(available[0] if rng.random() < .8 else rng.choice(available))
            else:
                # Favor richer evidence, but leave room for the other fields.
                size = int(rng.choice(available))
            code = int(rng.choice(valid[sizes == size]))
            positions = tuple((r0+int(r), c0+int(c)) for j, (r, c) in enumerate(active)
                              if code & (1 << j))
            chosen[i] = positions
            used.update(positions)
        else:
            return chosen
    return None


def _design(rng, h, w):
    """Structured and arbitrary masks, shared across both splits."""
    mask = np.zeros((h, w), dtype=bool)
    kind = int(rng.integers(8))
    if kind == 0:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    elif kind == 1:
        rr, cc = np.indices((h, w))
        mask = (rr+cc) % 2 == int(rng.integers(2))
    elif kind == 2:
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    elif kind == 3:
        for r in range(h):
            mask[r, round(r*(w-1)/(h-1))] = True
        if rng.random() < .5:
            mask |= mask[:, ::-1]
    elif kind == 4:
        mask[:] = True
        mask.flat[int(rng.integers(h*w))] = False
    else:
        n = int(rng.integers(3, h*w))
        mask.flat[rng.choice(h*w, n, replace=False)] = True
    if int(mask.sum()) < 3 or mask.all():
        mask[:] = False
        mask.flat[rng.choice(h*w, int(rng.integers(3, h*w)), replace=False)] = True
    if rng.integers(2):
        mask = mask[::-1]
    if rng.integers(2):
        mask = mask[:, ::-1]
    return mask.copy()


def _encode(p):
    h, w, k, rows, cols = (p[n] for n in ('h', 'w', 'k', 'rows', 'cols'))
    width = max(k*(w+1)+1, cols+2)
    a = np.full((h+rows+7, width), p['bg'], dtype=np.int8)
    a[[0, h+1, h+3], :] = p['sep']
    for x in range(0, k*(w+1)+1, w+1):
        a[1:h+1, x] = p['sep']
        a[h+2, x] = p['sep']
    for i, mask in enumerate(p['masks']):
        x = 1+i*(w+1)
        a[1:h+1, x:x+w] = np.where(mask, p['labels'][i], p['bg'])
        a[h+2, x:x+w] = p['targets'][i]
    y = h+5
    a[y, :cols+2] = a[-1, :cols+2] = p['sep']
    a[y+1:-1, 0] = a[y+1:-1, cols+1] = p['sep']
    for label, clues in zip(p['labels'], p['clues']):
        for r, c in clues:
            a[y+1+r, 1+c] = label
    return a


def _construct_answer(p):
    """Generate using whole-field binary expansions, independently of _render."""
    shape = (p['rows']*p['h'], p['cols']*p['w'])
    coverage = np.zeros(shape, dtype=np.int8)
    fields = []
    for mask, (r, c) in zip(p['masks'], p['origins']):
        coarse = np.zeros((p['rows'], p['cols']), dtype=bool)
        coarse[r:r+p['h'], c:c+p['w']] = mask
        field = np.kron(coarse, mask)
        coverage += field
        fields.append(field)
    answer = np.full(shape, p['bg'], dtype=np.int8)
    for field, target in zip(fields, p['targets']):
        answer[field & (coverage == 1)] = target
    return answer


def _witnesses(p, answer, triple=False, orientation=False):
    expected, count, fields = _render(p)
    if not np.array_equal(answer, expected) or not np.any(count >= 2):
        return False
    if any(not np.any(field & (count == 1)) for field in fields):
        return False
    # All identities participate in one interacting scene, not separate islands
    # that happen to be encoded on the same board.
    linked, todo = {0}, [0]
    while todo:
        i = todo.pop()
        for j, field in enumerate(fields):
            if j not in linked and np.any(fields[i] & field):
                linked.add(j)
                todo.append(j)
    if len(linked) != len(fields):
        return False
    # Each field has some newly inferred ink that survives cancellation.
    _, _, supplied = _render(p, visible=True)
    if any(not np.any(field & ~clue & (count == 1))
           for field, clue in zip(fields, supplied)):
        return False
    if np.array_equal(answer, _render(p, visible=True)[0]):
        return False
    if np.array_equal(answer, _render(p, solid=True)[0]):
        return False
    if np.array_equal(answer, _render(p, mode='coarse')[0]):
        return False
    if triple and not np.any(count >= 3):
        return False
    if triple and np.array_equal(answer, _render(p, mode='colors')[0]):
        return False
    if orientation:
        if p['h'] != p['w']:
            return False
        for t in range(1, 8):
            if (np.array_equal(answer, _render(p, stamp_turn=t)[0])
                    or np.array_equal(answer, _render(p, field_turn=t)[0])):
                return False
    return True


def _make_pair(rng, shape, k, palette, sparse=False, triple=False,
               orientation=False, forbidden_masks=()):
    h, w = shape
    for attempt in range(240):
        rows = int(rng.integers(h, 30//h+1))
        cols = int(rng.integers(w, 30//w+1))
        if rows*cols < 2*k+1:
            continue
        masks = [_design(rng, h, w) for _ in range(k)]
        if triple:
            common = int(rng.integers(h*w))
            for mask in masks:
                mask.flat[common] = True
                if mask.all():
                    remove = int(rng.choice([i for i in range(h*w) if i != common]))
                    mask.flat[remove] = False
        elif not orientation and rng.random() < .16:
            masks[-1] = masks[0].copy()
        if forbidden_masks and all(_key(m) in forbidden_masks for m in masks):
            continue
        anchor = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
        origins = []
        for i in range(k):
            if rng.random() < .15:
                origin = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
            else:
                origin = (int(np.clip(anchor[0]+rng.integers(-1, 2), 0, rows-h)),
                          int(np.clip(anchor[1]+rng.integers(-1, 2), 0, cols-w)))
            origins.append(origin)
        clues = _choose_clues(rng, masks, rows, cols, origins, sparse)
        if clues is None:
            continue
        bg, sep = map(int, palette[:2])
        labels = list(map(int, palette[2:2+k]))
        targets = list(map(int, palette[6:6+k]))
        if triple:
            # Two same-color fields still have separately counted ink.
            targets[1] = targets[0]
        if not orientation and not triple:
            pool = [color for color in range(10) if color != bg]
            if rng.random() < .5:
                targets = list(map(int, rng.choice(pool, k)))
        order = rng.permutation(k).tolist()
        p = dict(h=h, w=w, k=k, rows=rows, cols=cols, bg=bg, sep=sep,
                 labels=[labels[i] for i in order], masks=[masks[i] for i in order],
                 targets=[targets[i] for i in order], clues=[clues[i] for i in order],
                 origins=[origins[i] for i in order])
        answer = _construct_answer(p)
        if _witnesses(p, answer, triple, orientation):
            return {'input': _encode(p), 'output': answer}
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    side = int(rng.choice([3, 4]))
    stable_geometry = bool(rng.random() < .25)
    stable_palette = bool(rng.random() < .25)
    base_palette = rng.permutation(10)
    puzzle, used, training_masks = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            orientation = split == 'train' and index == 0
            triple = split == 'train' and index == 1
            k = 2 if orientation else 3 if triple else int(rng.integers(2, 5))
            pair = None
            for attempt in range(12):
                if orientation or stable_geometry:
                    shape = (side, side)
                elif triple:
                    shape = (int(rng.integers(3, 5)), int(rng.integers(3, 5)))
                else:
                    shape = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
                palette = base_palette if stable_palette else rng.permutation(10)
                pair = _make_pair(rng, shape, k, palette, sparse=split == 'test',
                                  triple=triple, orientation=orientation,
                                  forbidden_masks=training_masks if split == 'test' else ())
                if pair is not None and _key(pair['input']) not in used:
                    break
                pair = None
            if pair is None:
                raise RuntimeError('bounded field/fragment construction exhausted')
            puzzle[split].append(pair)
            used.add(_key(pair['input']))
            if split == 'train':
                training_masks.update(_key(m) for m in _parse(pair['input'])['masks'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Structural, semantic, demonstration and test-shift checks."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    used, parsed = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in used
            used.add(_key(a))
            p = _parse(a)
            assert b.shape == (p['rows']*p['h'], p['cols']*p['w'])
            assert _witnesses(p, b)
            expected, count, fields = _render(p)
            assert np.array_equal(b, expected)
            assert np.all(b[count != 1] == p['bg'])
            for field, target in zip(fields, p['targets']):
                assert np.all(b[field & (count == 1)] == target)
            assert set(map(int, np.unique(b))) <= {p['bg'], *p['targets']}
            parsed[split].append(p)
    train = parsed['train']
    # These are checks of visible evidence, not merely of generation flags.
    assert any(not np.array_equal(_render(p)[0], _render(p, source_colors=True)[0])
               and len(set(p['targets'])) >= 2 for p in train)
    assert any(np.any(_render(p)[1] >= 3) for p in train), 'need evidence against XOR'
    assert any(not np.array_equal(_render(p)[0], _render(p, mode='colors')[0])
               for p in train), 'same-color collisions must be demonstrated'
    square = [p for p in train if p['h'] == p['w']]
    assert square
    for t in range(1, 8):
        assert any(not np.array_equal(_render(p)[0], _render(p, stamp_turn=t)[0]) for p in square)
        assert any(not np.array_equal(_render(p)[0], _render(p, field_turn=t)[0]) for p in square)
    train_masks = {_key(m) for p in train for m in p['masks']}
    for p in parsed['test']:
        assert any(_key(m) not in train_masks for m in p['masks']), 'test must include a new design'
    return None


def self_test():
    """Bounded 42-seed variation checks, count checks, and damaging mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, tile_shapes, masks, layouts, fragments, palettes, counts = (set() for _ in range(7))
    triples = duplicates = shared_targets = sparse = boundary = interior = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                predicted = solve(np.asarray(pair['input']).copy())
                assert np.array_equal(predicted, pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                tile_shapes.add((p['h'], p['w']))
                masks.update(_key(m) for m in p['masks'])
                layouts.add((p['rows'], p['cols'], tuple(p['origins']),
                             tuple(_key(m) for m in p['masks'])))
                fragments.update((m.shape, tuple((r-o[0], c-o[1]) for r, c in clue))
                                 for m, o, clue in zip(p['masks'], p['origins'], p['clues']))
                palettes.add((p['bg'], p['sep'], tuple(p['labels']), tuple(p['targets'])))
                counts.add(p['k'])
                triples += bool(np.any(_render(p)[1] >= 3))
                duplicates += len({_key(m) for m in p['masks']}) < p['k']
                shared_targets += len(set(p['targets'])) < p['k']
                for m, (r, c), clue in zip(p['masks'], p['origins'], p['clues']):
                    sparse += 2*len(clue) < int(m.sum())
                    edges = r == 0 or c == 0 or r+p['h'] == p['rows'] or c+p['w'] == p['cols']
                    boundary += edges
                    interior += not edges
    assert len(shapes) >= 60 and len(masks) >= 100 and len(layouts) >= 180
    assert len(fragments) >= 100 and len(palettes) >= 70
    assert tile_shapes == {(h, w) for h in range(2, 5) for w in range(2, 5)}
    assert counts == {2, 3, 4}
    assert triples >= 42 and duplicates >= 3 and shared_targets >= 3
    assert sparse >= 30 and boundary >= 30 and interior >= 30
    # Hold the designs, geometry, and translations fixed while varying clues.
    fixed = _parse(puzzles[0]['train'][0]['input'])
    conditional = set()
    for seed in range(42):
        clues = _choose_clues(np.random.default_rng(seed), fixed['masks'],
                              fixed['rows'], fixed['cols'], fixed['origins'], False)
        assert clues is not None
        conditional.add(tuple(clues))
        for mask, clue, origin in zip(fixed['masks'], clues, fixed['origins']):
            assert _origins(mask, fixed['rows'], fixed['cols'], clue) == [origin]
    assert len(conditional) >= 12
    # A valid family member at the exact 30x30 output boundary.
    boundary_case = dict(h=2, w=2, rows=15, cols=15, k=2, bg=0, sep=5,
                         labels=[1, 2], targets=[7, 8],
                         masks=[np.array([[1, 1], [0, 1]], dtype=bool),
                                np.array([[1, 0], [1, 1]], dtype=bool)],
                         origins=[(5, 5), (5, 6)],
                         clues=[((5, 5), (6, 6)), ((5, 6), (6, 7))])
    boundary_answer = _construct_answer(boundary_case)
    assert boundary_answer.shape == (30, 30) and _witnesses(boundary_case, boundary_answer)
    assert np.array_equal(solve(_encode(boundary_case)), boundary_answer)
    a, b = generate_puzzle(112358, 5, 3), generate_puzzle(112358, 5, 3)
    assert all(np.array_equal(x[v], y[v]) for s in ('train', 'test')
               for x, y in zip(a[s], b[s]) for v in ('input', 'output'))
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(9000+n*10+t, n, t)
            assert len(p['train']) == n and len(p['test']) == t
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Collision erasure, key application, and missing placements each matter.
    first = puzzles[0]
    for mode in ('visible', 'source', 'paint', 'odd', 'solid', 'coarse', 'colors'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]]
               for s in ('train', 'test')}
        index = 1 if mode in ('odd', 'colors') else 0
        p = _parse(bad['train'][index]['input'])
        kwargs = {'visible': True} if mode == 'visible' else (
            {'source_colors': True} if mode == 'source' else (
            {'solid': True} if mode == 'solid' else {'mode': mode}))
        bad['train'][index]['output'] = _render(p, **kwargs)[0]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'{mode} shortcut accepted')
    # Ambiguous origins may never be settled by scan order or maximum overlap.
    mask = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert len(_origins(mask, 7, 7, ((2, 2), (3, 2)))) > 1
    ambiguous = dict(h=3, w=3, rows=7, cols=7, k=2, bg=0, sep=5,
                     labels=[1, 2], targets=[7, 8],
                     masks=[mask, np.array([[1, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=bool)],
                     clues=[((2, 2), (3, 2)), ((4, 4), (6, 6))])
    try:
        solve(_encode(ambiguous))
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous input accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]]
               for s in ('train', 'test')}
    damaged['test'][0]['input'][0, 0] = (int(damaged['test'][0]['input'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged frame accepted')
    return None
