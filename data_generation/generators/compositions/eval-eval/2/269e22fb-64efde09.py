"""Atlas-addressed broadcast: localize windows, pool clues, gate rays by span.

Only standard-library modules and NumPy are used. No files, caches, examples,
seeds or training labels are consulted by solve(). Construction and validation
use different sources of geometry: sampled crop coordinates versus input pixels.
"""
from collections import Counter
import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _canonical(a):
    return min(_key(_orient(a, t) ^ s) for t in range(8) for s in (0, 1))


def _vector(t, side):
    labels = _orient(np.arange(9).reshape(3, 3), t)
    dr, dc = _DIRS[side]
    p = np.argwhere(labels == 4)[0]
    q = np.argwhere(labels == 4 + 3 * dr + dc)[0]
    return tuple(int(v) for v in q - p)


_VECTORS = tuple(tuple(_vector(t, s) for s in range(4)) for t in range(8))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _matches(reference, patch):
    """All exact orientation, complement and location matches, including ties."""
    matches = []
    h, w = patch.shape
    labels = np.arange(reference.size).reshape(reference.shape)
    for t in range(8):
        form = _orient(reference, t)
        if h > form.shape[0] or w > form.shape[1]:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(form, (h, w))
        for swap in (0, 1):
            rr, cc = np.nonzero(np.all(windows == (patch ^ swap), axis=(-2, -1)))
            for r, c in zip(rr, cc):
                matches.append((t, swap, int(r), int(c),
                                _orient(labels, t)[r:r+h, c:c+w].copy()))
    return matches


def _components(a, background):
    """Interior foreground rectangles; boundary pixels are handled as clues."""
    remaining = set((int(r), int(c)) for r, c in np.argwhere(a != background)
                    if 0 < r < a.shape[0]-1 and 0 < c < a.shape[1]-1)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                nxt = r+dr, c+dc
                if nxt in remaining:
                    remaining.remove(nxt)
                    cells.add(nxt)
                    stack.append(nxt)
        rr, cc = zip(*cells)
        r, c, h, w = min(rr), min(cc), max(rr)-min(rr)+1, max(cc)-min(cc)+1
        assert len(cells) == h*w and min(h, w) >= 2, 'not a solid window'
        assert min(r, c, a.shape[0]-r-h, a.shape[1]-c-w) >= 2
        result.append(dict(r=r, c=c, arr=a[r:r+h, c:c+w].copy(), cells=cells))
    return result


def _ray(tile, side, slot, ref_shape, canvas_shape, *, local=False):
    """Return no ray when this atlas line is absent from the visible window."""
    labels = tile['labels']
    rows, cols = labels // ref_shape[1], labels % ref_shape[1]
    if local:  # Explicit omission-of-A counterfactual: discard crop offsets.
        rows, cols = rows-rows.min(), cols-cols.min()
    coord = cols if side % 2 == 0 else rows
    eligible = np.argwhere(coord == slot)
    if not len(eligible):
        return ()
    dr, dc = _VECTORS[tile['t']][side]
    scores = eligible[:, 0]*dr + eligible[:, 1]*dc
    u, v = map(int, eligible[int(np.argmax(scores))])
    r, c = tile['r']+u+dr, tile['c']+v+dc
    cells = []
    while 0 <= r < canvas_shape[0] and 0 <= c < canvas_shape[1]:
        cells.append((r, c))
        r, c = r+dr, c+dc
    return tuple(cells)


def _decode(input_grid):
    a = _grid(input_grid)
    H, W = a.shape
    bg = int(a[0, 0])
    assert all(int(a[r, c]) == bg for r, c in ((0, W-1), (H-1, 0), (H-1, W-1)))
    tiles = _components(a, bg)
    assert 3 <= len(tiles) <= 5
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i]['arr'].size)
    ref_index = order[0]
    assert tiles[ref_index]['arr'].size > tiles[order[1]]['arr'].size, 'atlas area tie'
    reference = tiles[ref_index]['arr']
    colors = np.unique(reference)
    assert len(colors) == 2 and bg not in colors
    binary = (reference == colors[1]).astype(np.int8)
    assert len({_key(_orient(binary, t) ^ s) for t in range(8) for s in (0, 1)}) == 16
    ref_shape = reference.shape
    for i, tile in enumerate(tiles):
        assert set(map(int, tile['arr'].flat)) == set(map(int, colors))
        if i == ref_index:
            tile.update(t=0, swap=0, labels=np.arange(reference.size).reshape(ref_shape))
        else:
            matches = _matches(binary, (tile['arr'] == colors[1]).astype(np.int8))
            assert len(matches) == 1, 'ambiguous crop correspondence'
            t, swap, _, _, labels = matches[0]
            tile.update(t=t, swap=swap, labels=labels)
    occupied = set().union(*(t['cells'] for t in tiles))
    clues = [(int(r), int(c), int(a[r, c])) for r, c in np.argwhere(a != bg)
             if r in (0, H-1) or c in (0, W-1)]
    assert clues and all(color not in colors for _, _, color in clues)
    corridors = {}
    for i, tile in enumerate(tiles):
        for side in range(4):
            length = ref_shape[1] if side % 2 == 0 else ref_shape[0]
            for slot in range(length):
                ray = _ray(tile, side, slot, ref_shape, a.shape)
                if (len(ray) >= 2 and not occupied.intersection(ray)
                        and all(int(a[p]) == bg for p in ray[:-1])):
                    corridors.setdefault(ray[-1], []).append((i, side, slot))
    code, sides, observations = {}, set(), []
    allocations = [set() for _ in tiles]
    for r, c, color in clues:
        matches = corridors.get((r, c), [])
        assert len(matches) == 1, 'boundary dot has no unique clear source'
        i, side, slot = matches[0]
        assert slot not in code or code[slot] == color, 'contradictory slot colors'
        code[slot] = color
        sides.add(side)
        allocations[i].add(slot)
        observations.append((i, slot, color))
    assert len(sides) == 1 and 1 <= len(code) <= 5
    side = next(iter(sides))
    expected, painted = a.copy(), {}
    local_only, offset_free = a.copy(), a.copy()
    supports, ray_lengths = [], []
    for i, tile in enumerate(tiles):
        support = set()
        for slot, color in code.items():
            ray = _ray(tile, side, slot, ref_shape, a.shape)
            if ray:
                assert len(ray) >= 2
                support.add(slot)
                ray_lengths.append(len(ray))
            for p in ray:
                assert p not in occupied and p not in painted, 'ray collision'
                assert int(a[p]) in (bg, color), 'ray would overwrite another clue'
                expected[p] = color
                painted[p] = color
                if slot in allocations[i]:
                    local_only[p] = color
            # Counterfactual is diagnostic only, and cannot alter an object.
            for p in _ray(tile, side, slot, ref_shape, a.shape, local=True):
                if p not in occupied:
                    offset_free[p] = color
        supports.append(support)
    assert not np.array_equal(a, expected)
    crop_tiles = [t for i, t in enumerate(tiles) if i != ref_index]
    offsets = []
    for t in crop_tiles:
        labels = t['labels']
        offsets.append((int((labels // ref_shape[1]).min()), int((labels % ref_shape[1]).min())))
    features = dict(
        reference_class=_canonical(binary), reference_shape=ref_shape, background=bg,
        shape=a.shape, palette=tuple(map(int, np.unique(a))), side=side,
        tile_count=len(tiles), slot_count=len(code), slots=tuple(sorted(code)),
        signal_count=len(set(code.values())), repeated_color=len(set(code.values())) < len(code),
        adjacent=any(b-a == 1 for a, b in zip(sorted(code), sorted(code)[1:])),
        transforms=tuple(t['t'] for t in crop_tiles), swaps=tuple(t['swap'] for t in crop_tiles),
        crop_shapes=tuple(t['arr'].shape for t in crop_tiles), offsets=tuple(offsets),
        crop_supports=tuple(len(s) for i, s in enumerate(supports) if i != ref_index),
        allocations=tuple(len(s) for s in allocations), supports=tuple(len(s) for s in supports),
        clipped=any(0 < len(s) < len(code) for s in supports), silent=any(not s for s in supports),
        pooled=all(len(s) < len(code) for s in allocations),
        full_clue=any(len(s) == len(code) for s in allocations),
        reference_unhinted=not allocations[ref_index], redundant=len(clues) > len(code),
        omit_A_changes=int(np.count_nonzero(expected != offset_free)),
        omit_B_changes=int(np.count_nonzero(expected != local_only)),
        ray_lengths=tuple(ray_lengths), changed=int(np.count_nonzero(expected != a)))
    return expected, features


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the atlas, all crop addresses and the pooled code from this input."""
    return _decode(input_grid)[0]


def _bank(reference):
    """Unique crops under all eight isometries AND both binary role maps."""
    counts = Counter()
    H, W = reference.shape
    for t in range(8):
        a = _orient(reference, t)
        for h in range(2, a.shape[0]+1):
            for w in range(2, a.shape[1]+1):
                if h*w == reference.size:
                    continue
                for r in range(a.shape[0]-h+1):
                    for c in range(a.shape[1]-w+1):
                        patch = a[r:r+h, c:c+w]
                        for s in (0, 1):
                            counts[_key(patch ^ s)] += 1
    bank = []
    for h in range(2, H+1):
        for w in range(2, W+1):
            if not 6 <= h*w <= .72*reference.size:
                continue
            for r in range(H-h+1):
                for c in range(W-w+1):
                    patch = reference[r:r+h, c:c+w]
                    if len(np.unique(patch)) == 2 and counts[_key(patch)] == 1:
                        bank.append((r, c, h, w))
    return bank


def _span(crop, side):
    r, c, h, w = crop
    return set(range(c, c+w)) if side % 2 == 0 else set(range(r, r+h))


def _anchors(rng, bank, side, *, small=False):
    candidates = bank
    if small:
        areas = sorted(h*w for _, _, h, w in bank)
        threshold = areas[2*len(areas)//3]
        candidates = [b for b in bank if b[2]*b[3] <= threshold]
    for _ in range(128):
        ids = rng.choice(len(candidates), 2, replace=False)
        a, b = candidates[int(ids[0])], candidates[int(ids[1])]
        sa, sb = _span(a, side), _span(b, side)
        parts = (sa-sb, sa & sb, sb-sa)
        if all(parts):
            slots = tuple(int(rng.choice(sorted(p))) for p in parts)
            return (a, b), slots
    return None


def _make_reference(rng):
    H = int(rng.integers(6, 10))
    W = H if rng.random() < .35 else int(rng.integers(6, 10))
    for _ in range(64):
        a = np.zeros((H, W), dtype=np.int8)
        # Intersecting bars, loops, bends, blocks and staircases, not a bitmap bank.
        for _ in range(int(rng.integers(4, 9))):
            h, w = int(rng.integers(2, H+1)), int(rng.integers(2, W+1))
            r, c = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
            kind = int(rng.integers(5))
            if kind == 0:
                a[r, c:c+w] = 1
                a[r:r+h, c] = 1
            elif kind == 1:
                a[r, c:c+w] = a[r+h-1, c:c+w] = 1
                a[r:r+h, c] = a[r:r+h, c+w-1] = 1
            elif kind == 2:
                a[r:r+int(rng.integers(1, min(h, 2)+1)), c:c+w] = 1
            elif kind == 3:
                a[r:r+h, c:c+int(rng.integers(1, min(w, 2)+1))] = 1
            else:
                for i in range(max(h, w)):
                    a[r+min(i, h-1), c+min(i, w-1)] = 1
        if not .2 <= a.mean() <= .78:
            continue
        if len({_key(_orient(a, t) ^ s) for t in range(8) for s in (0, 1)}) != 16:
            continue
        bank = _bank(a)
        if len(bank) >= 32 and all(_anchors(rng, bank, s, small=True) for s in (0, 1)):
            return a, bank
    raise RuntimeError('bounded reference construction exhausted')


def _tile(reference, crop, t, swap):
    r, c, h, w = crop
    labels = np.arange(reference.size).reshape(reference.shape)[r:r+h, c:c+w]
    return dict(arr=_orient(reference[r:r+h, c:c+w] ^ swap, t),
                labels=_orient(labels, t), t=t, swap=swap)


def _place(rng, descriptions, side, slots, ref_shape, shape):
    for _ in range(12):
        placed = []
        indices = list(map(int, rng.permutation(len(descriptions))))
        for i in indices:
            source = descriptions[i]
            h, w = source['arr'].shape
            for _ in range(96):
                r, c = int(rng.integers(2, shape[0]-h-1)), int(rng.integers(2, shape[1]-w-1))
                tile = dict(source, r=r, c=c, index=i)
                cells = {(r+u, c+v) for u in range(h) for v in range(w)}
                halo = {(u+dr, v+dc) for u, v in cells for dr, dc in ((0, 0),)+_DIRS}
                rays = {s: _ray(tile, side, s, ref_shape, shape) for s in slots}
                ray_cells = {p for ray in rays.values() for p in ray}
                if any(halo & p['cells'] or cells & p['ray_cells']
                       or ray_cells & (p['cells'] | p['ray_cells']) for p in placed):
                    continue
                tile.update(cells=cells, ray_cells=ray_cells, rays=rays)
                placed.append(tile)
                break
            else:
                break
        if len(placed) == len(descriptions):
            return sorted(placed, key=lambda p: p['index'])
    return None


def _make_pair(rng, reference, bank, demonstration, is_test, seen):
    for attempt in range(80):
        side = int(rng.integers(4))
        anchor = _anchors(rng, bank, side, small=is_test)
        if anchor is None:
            continue
        crops, witness_slots = anchor
        witness = demonstration is not None or rng.random() < .5
        if witness:
            slots = tuple(sorted(witness_slots))
        else:
            length = reference.shape[1] if side % 2 == 0 else reference.shape[0]
            slot_count = int(rng.integers(1, 6))
            if is_test:
                # Make the novel transformed crop's transfer visible even for one slot.
                shared = int(rng.choice(sorted(_span(crops[0], side) & _span(crops[1], side))))
                remaining = [s for s in range(length) if s != shared]
                slots = tuple(sorted([shared]+list(map(int, rng.choice(remaining, slot_count-1, replace=False)))))
            else:
                slots = tuple(sorted(map(int, rng.choice(length, slot_count, replace=False))))
        if demonstration == 'split':
            ts = [0, 4]
        elif demonstration == 'full':
            ts = [0, int(rng.choice([1, 3]))]
        elif is_test:
            ts = [int(rng.choice([5, 7])), int(rng.integers(8))]
        else:
            ts = list(map(int, rng.choice([0, 1, 3, 4, 6], 2)))
        full_crop = (0, 0, *reference.shape)
        descriptions = [_tile(reference, full_crop, 0, 0)]
        for crop, t in zip(crops, ts):
            # The first demonstration isolates geometry; the second exposes role exchange.
            swap = 0 if demonstration == 'split' else int(rng.integers(2))
            descriptions.append(_tile(reference, crop, t, swap))
        if demonstration == 'full' and not any(p['swap'] for p in descriptions[1:]):
            descriptions[1]['arr'] ^= 1
            descriptions[1]['swap'] = 1
        count = 3 if demonstration else int(rng.integers(3, 6))
        for _ in range(count-3):
            # Prefer an address-disjoint (silent) window when one is available.
            silent = [b for b in bank if not (_span(b, side) & set(slots))]
            choices = silent if silent and rng.random() < .65 else bank
            crop = choices[int(rng.integers(len(choices)))]
            t = int(rng.integers(8)) if is_test else int(rng.choice([0, 1, 3, 4, 6]))
            descriptions.append(_tile(reference, crop, t, int(rng.integers(2))))
        # Later retries use more room, with no rule or convention changes.
        low = 20 if attempt < 24 else 28
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        if rng.random() < .2:
            shape = (shape[0], shape[0])
        placed = _place(rng, descriptions, side, slots, reference.shape, shape)
        if placed is None:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, structural = palette[0], np.array(palette[1:3], dtype=np.int8)
        signal_values = palette[3:3+len(slots)]
        if demonstration is None and len(slots) > 1 and rng.random() < .35:
            signal_values[-1] = signal_values[0]
        code = dict(zip(slots, signal_values))
        a = np.full(shape, bg, dtype=np.int8)
        for tile in placed:
            r, c = tile['r'], tile['c']
            h, w = tile['arr'].shape
            a[r:r+h, c:c+w] = structural[tile['arr']]
        clues = []
        if demonstration == 'split':
            # The two exclusive slots and one shared slot are supplied by crops.
            for index, slot in enumerate(witness_slots):
                emitter = 1 if index == 0 else 2 if index == 2 else int(rng.integers(1, 3))
                clues.append((emitter, slot))
        elif demonstration == 'full' or rng.random() < .25:
            clues = [(0, slot) for slot in slots]
        else:
            for slot in slots:
                options = [i for i, tile in enumerate(placed) if tile['rays'][slot]]
                clues.append((int(rng.choice(options)), slot))
        if demonstration is None and rng.random() < .4:
            options = [(i, s) for i, tile in enumerate(placed) for s in slots
                       if tile['rays'][s] and (i, s) not in clues]
            if options:
                for k in rng.choice(len(options), min(len(options), int(rng.integers(1, 3))), replace=False):
                    clues.append(options[int(k)])
        for i, slot in clues:
            a[placed[i]['rays'][slot][-1]] = code[slot]
        out = a.copy()
        for tile in placed:
            for slot, color in code.items():
                for p in tile['rays'][slot]:
                    out[p] = color
        t = int(rng.integers(8))
        a, out = _orient(a, t), _orient(out, t)
        if _key(a) in seen:
            continue
        predicted, f = _decode(a)
        assert np.array_equal(predicted, out), 'sampled geometry disagrees with pixel decoding'
        if not f['omit_B_changes']:
            continue
        if demonstration == 'split' and not (f['omit_A_changes'] and f['omit_B_changes']):
            continue
        seen.add(_key(a))
        return {'input': a, 'output': out}
    raise RuntimeError('bounded layout construction exhausted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} outside {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    reference, bank = _make_reference(rng)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            demonstration = ('split', 'full')[i] if split == 'train' and i < 2 else None
            puzzle[split].append(_make_pair(rng, reference, bank, demonstration, split == 'test', seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, references, train = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            predicted, f = _decode(a)
            assert np.array_equal(predicted, out), 'incorrect selective broadcast'
            assert f['omit_B_changes'] > 0
            assert np.array_equal(out[a != f['background']], a[a != f['background']])
            references.add(f['reference_class'])
            if split == 'train':
                train.append(f)
    assert len(references) == 1, 'the puzzle must share one atlas up to D4 and role exchange'
    assert any(f['pooled'] and f['reference_unhinted'] and f['clipped']
               and f['omit_A_changes'] > 0 and f['omit_B_changes'] > 0 for f in train)
    assert any(f['full_clue'] and f['clipped'] for f in train)
    assert any(any(t in (1, 3) for t in f['transforms']) for f in train)
    assert any(any(t in (4, 6) for t in f['transforms']) for f in train)
    assert any(any(f['swaps']) for f in train), 'demonstrate structural role exchange'


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds; aggregate signatures, no files or retained corpus."""
    signatures, references, shapes, crop_shapes, offsets = set(), set(), set(), set(), set()
    transforms, sides, counts, backgrounds, palettes = set(), set(), set(), set(), set()
    occurrences = Counter()
    split_slot_counts = {'train': Counter(), 'test': Counter()}
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        signatures.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, f = _decode(pair['input'])
                references.add(f['reference_class'])
                shapes.add(f['shape'])
                crop_shapes.update(f['crop_shapes'])
                offsets.update(f['offsets'])
                transforms.update(f['transforms'])
                sides.add(f['side'])
                counts.add((f['tile_count'], f['slot_count']))
                split_slot_counts[split][f['slot_count']] += 1
                backgrounds.add(f['background'])
                palettes.add(f['palette'])
                for name in ('pooled', 'full_clue', 'clipped', 'silent', 'redundant',
                             'repeated_color', 'adjacent', 'omit_A_changes', 'omit_B_changes'):
                    occurrences[name] += bool(f[name])
                if split == 'test':
                    assert any(t in (5, 7) and n > 0 for t, n in zip(f['transforms'], f['crop_supports']))
    assert len(signatures) == len(references) == 42
    assert len(shapes) >= 35 and len(crop_shapes) >= 20 and len(offsets) >= 20
    assert transforms == set(range(8)) and sides == set(range(4))
    assert {n for n, _ in counts} == {3, 4, 5}
    assert {n for _, n in counts} == {1, 2, 3, 4, 5}
    assert all(set(c) == {1, 2, 3, 4, 5} for c in split_slot_counts.values())
    assert backgrounds == set(range(10)) and len(palettes) >= 25
    assert all(occurrences[k] >= 3 for k in occurrences)
    assert _fingerprint(first) == _fingerprint(generate_puzzle(0))
    # Input-only metamorphic checks of orientation and arbitrary palette roles.
    for t in range(8):
        recolor = np.roll(np.arange(10, dtype=np.int8), t+1)
        for pair in first['train']+first['test']:
            a, out = (_orient(recolor[pair[k]], t) for k in ('input', 'output'))
            assert np.array_equal(solve(a), out)
    # The validator must reject a changed answer, and the matcher ambiguous evidence.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    assert len(_matches(np.zeros((6, 7), dtype=np.int8), np.zeros((2, 3), dtype=np.int8))) > 1
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(123, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                reference_classes=len(references), canvas_shapes=len(shapes),
                crop_shapes=len(crop_shapes), offsets=len(offsets),
                count_combinations=len(counts), occurrences=dict(occurrences),
                split_slot_counts={s: dict(c) for s, c in split_slot_counts.items()})


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
