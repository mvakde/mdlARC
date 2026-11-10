"""Corner territories become an ordered tray of rotation-classified contours.

Only NumPy and the standard library are used. Solving never consults the sampler,
its geometry catalogue, a random seed, or a stored example.
"""
from collections import Counter
from functools import lru_cache
from hashlib import sha256

import numpy as np


_DIR8 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if dy or dx)
_DIR4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return grid.astype(np.int8, copy=True)


def _corners(h, w):
    return ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0))


def _decode(grid):
    h, w = grid.shape
    if min(h, w) < 3:
        raise ValueError('source fields need at least three rows and columns')
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(counts.argmax())])
    if int(counts.max()) <= grid.size // 2:
        raise ValueError('background must occupy more than half of the input')
    sources, bits = [], 0
    for i, (y, x) in enumerate(_corners(h, w)):
        color = int(grid[y, x])
        if color != bg:
            sources.append((y, x, color))
            bits |= 1 << i
    if not sources or np.count_nonzero(grid != bg) != len(sources):
        raise ValueError('only corner cells may be marked')
    if len({color for _, _, color in sources}) != len(sources):
        raise ValueError('marked corners must have distinct colors')
    return bg, sources, bits


def _contours(grid, mode='correct'):
    """Fixed rule in production; other modes are bounded diagnostic ablations."""
    bg, sources, _ = _decode(grid)
    yy, xx = np.indices(grid.shape)
    axial = [(abs(yy-y), abs(xx-x)) for y, x, _ in sources]
    distances = np.stack([np.maximum(dy, dx) if mode == 'ownership'
                          else dy + dx for dy, dx in axial])
    minima = distances.min(axis=0)
    unique = (distances == minima).sum(axis=0) == 1
    winner = distances.argmin(axis=0)
    out = np.full(grid.shape, bg, dtype=np.int8)
    for i, (_, _, color) in enumerate(sources):
        dy, dx = axial[i]
        radius = dy + dx if mode == 'radial' else np.maximum(dy, dx)
        keep = (winner == i) & ((radius % 2) == (1 if mode == 'parity' else 0))
        if mode != 'ties':
            keep &= unique
        out[keep] = color
    return out


def _objects(grid, bg, diagonal=True):
    """Same-color components in order of their first occupied raster cell."""
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    directions = _DIR8 if diagonal else _DIR4
    for y in range(h):
        for x in range(w):
            color = int(grid[y, x])
            if seen[y, x] or color == bg:
                continue
            seen[y, x] = True
            stack, cells = [(y, x)], []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            top, left = min(r for r, c in cells), min(c for r, c in cells)
            bottom, right = max(r for r, c in cells), max(c for r, c in cells)
            mask = np.zeros((bottom-top+1, right-left+1), dtype=bool)
            for r, c in cells:
                mask[r-top, c-left] = True
            objects.append({'mask': mask, 'color': color, 'first': (y, x)})
    return objects


def _mask_key(mask):
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _rotation_key(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _keys(objects, mode='correct'):
    if mode == 'identity':
        return [_mask_key(o['mask']) for o in objects]
    if mode == 'area':
        return [int(o['mask'].sum()) for o in objects]
    if mode == 'color':
        return [o['color'] for o in objects]
    return [_rotation_key(o['mask']) for o in objects]


def _pack(objects, bg, mode='correct'):
    if not objects:
        raise ValueError('there must be at least one contour object')
    height = max(o['mask'].shape[0] for o in objects) + 2
    width = sum(o['mask'].shape[1] for o in objects) + len(objects) - 1
    if height > 30 or width > 30:
        raise ValueError('the contour tray exceeds ARC dimensions')
    keys = _keys(objects, mode)
    counts = Counter(keys)
    out = np.full((height, width), bg, dtype=np.int8)
    left = 0
    for obj, key in zip(objects, keys):
        mask = obj['mask']
        up = counts[key] > 1
        if mode == 'exactly_two':
            up = counts[key] == 2
        elif mode == 'all_top':
            up = True
        elif mode == 'all_bottom':
            up = False
        top = 0 if up else height - mask.shape[0]
        patch = out[top:top+mask.shape[0], left:left+mask.shape[1]]
        patch[mask] = obj['color']
        left += mask.shape[1] + 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    bg, _, _ = _decode(grid)
    return _pack(_objects(_contours(grid), bg), bg)


def _make_input(h, w, bits, bg, colors):
    grid = np.full((h, w), bg, dtype=np.int8)
    color_iter = iter(colors)
    for i, point in enumerate(_corners(h, w)):
        if bits & (1 << i):
            grid[point] = next(color_iter)
    return grid


def _signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _class_profile(objects):
    return tuple(sorted(Counter(_keys(objects)).items()))


def _different(a, b):
    return a.shape != b.shape or not np.array_equal(a, b)


def _features(grid, detailed=False):
    bg, sources, bits = _decode(grid)
    mid = _contours(grid)
    objects = _objects(mid, bg)
    keys = _keys(objects)
    counts = Counter(keys)
    answer = _pack(objects, bg)
    feature = {'h': grid.shape[0], 'w': grid.shape[1], 'bits': bits,
               'sources': len(sources), 'objects': len(objects),
               'painted': int(np.count_nonzero(mid != bg)),
               'up': sum(counts[k] > 1 for k in keys),
               'down': sum(counts[k] == 1 for k in keys),
               'multiplicity': max(counts.values()),
               'diagonal': len(_objects(mid, bg, diagonal=False)) != len(objects),
               'identity': _different(answer, _pack(objects, bg, 'identity')),
               'area': _different(answer, _pack(objects, bg, 'area'))}
    if detailed:
        for mode in ('color', 'exactly_two', 'all_top', 'all_bottom'):
            feature[mode] = _different(answer, _pack(objects, bg, mode))
        feature['connect4'] = False
        try:
            alt = _pack(_objects(mid, bg, diagonal=False), bg)
            feature['connect4'] = _different(answer, alt)
        except ValueError:
            pass
        for mode in ('ties', 'ownership', 'radial', 'parity'):
            alt_objects = _objects(_contours(grid, mode), bg)
            # A witness is counted only if the alternative has a legal ARC tray.
            feature[mode] = False
            feature[mode + '_classes'] = _class_profile(objects) != _class_profile(alt_objects)
            try:
                feature[mode] = _different(answer, _pack(alt_objects, bg))
            except ValueError:
                pass
    return feature


@lru_cache(maxsize=1)
def _catalogue():
    """Small deterministic in-memory geometry catalogue; never used by solve.

    Only geometry and scalar measurements are cached, not example grids.
    """
    records = []
    for h in range(3, 16):
        for w in range(3, 16):
            for bits in range(1, 16):
                grid = _make_input(h, w, bits, 0, range(1, 5))
                try:
                    feature = _features(grid)
                except ValueError:
                    continue
                if feature['painted'] == feature['sources']:
                    continue
                # The two teaching roles can be recognized without random search.
                feature['area_demo'] = bool(feature['sources'] == 1 and
                                            feature['area'] and feature['up'] and feature['down'])
                feature['boundary_demo'] = False
                if (min(h, w) >= 5 and feature['sources'] >= 2
                        and feature['up'] and feature['down'] and feature['identity']):
                    full = _features(grid, detailed=True)
                    feature['boundary_demo'] = all(full[k] for k in
                        ('ties', 'ties_classes', 'ownership', 'ownership_classes',
                         'radial', 'radial_classes', 'parity', 'exactly_two'))
                records.append(feature)
    assert any(r['area_demo'] for r in records)
    assert any(r['boundary_demo'] for r in records)
    return tuple(records)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name + ' is not a legal integer count')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count = _count(num_train, 'num_train', 2, 10)
    test_count = _count(num_test, 'num_test', 1, 4)
    if train_count <= test_count:
        raise ValueError('num_train must exceed num_test')
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(int(seed))
    catalogue = _catalogue()
    selected = []
    for role in ('area_demo', 'boundary_demo'):
        pool = [r for r in catalogue if r[role]]
        selected.append(pool[int(rng.integers(len(pool)))])
    used = {(r['h'], r['w'], r['bits']) for r in selected}
    for _ in range(train_count - 2):
        pool = [r for r in catalogue if (r['h'], r['w'], r['bits']) not in used]
        # Equal weight to source counts; the catalogue has unequal category sizes.
        source_count = int(rng.integers(1, 5))
        choices = [r for r in pool if r['sources'] == source_count]
        r = choices[int(rng.integers(len(choices)))]
        selected.append(r)
        used.add((r['h'], r['w'], r['bits']))
    rng.shuffle(selected)
    shown_masks = {r['bits'] for r in selected}
    diagonal_demonstrated = any(r['diagonal'] for r in selected)
    test_specs = []
    available_masks = [bits for bits in range(1, 16) if bits not in shown_masks]
    rng.shuffle(available_masks)
    for i in range(test_count):
        bits = available_masks[i]
        pool = [r for r in catalogue if r['bits'] == bits
                and (diagonal_demonstrated or not r['diagonal'])]
        r = pool[int(rng.integers(len(pool)))]
        test_specs.append(r)
    stable_bg = int(rng.integers(10)) if rng.random() < .4 else None
    puzzle = {'train': [], 'test': []}
    for split, specs in (('train', selected), ('test', test_specs)):
        for r in specs:
            bg = int(rng.integers(10)) if stable_bg is None else stable_bg
            colors = [int(c) for c in rng.permutation(10) if c != bg]
            grid = _make_input(r['h'], r['w'], r['bits'], bg, colors)
            puzzle[split].append({'input': grid, 'output': solve(grid)})
    return puzzle


def _scalar_contours(grid):
    """Cellwise check of ownership, blank ties and even maximum offset."""
    bg, sources, _ = _decode(grid)
    out = np.full(grid.shape, bg, dtype=np.int8)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            distances = [abs(y-r) + abs(x-c) for r, c, _ in sources]
            nearest = min(distances)
            if distances.count(nearest) == 1:
                r, c, color = sources[distances.index(nearest)]
                if max(abs(y-r), abs(x-c)) % 2 == 0:
                    out[y, x] = color
    return out


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _count(len(puzzle['train']), 'num_train', 2, 10)
    _count(len(puzzle['test']), 'num_test', 1, 4)
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, geometry, features = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            bg, sources, bits = _decode(inp)
            assert 3 <= min(inp.shape)
            key = _signature(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            geom = (*inp.shape, bits)
            assert geom not in geometry, 'recolored geometry repeated'
            geometry.add(geom)
            mid = _scalar_contours(inp)
            assert np.count_nonzero(mid != bg) > len(sources), 'no non-source contour cells'
            assert np.array_equal(mid, _contours(inp)), 'two contour constructions disagree'
            objects = _objects(mid, bg)
            keys = _keys(objects)
            counts = Counter(keys)
            assert np.array_equal(out, solve(inp)), 'output disagrees with input-only solve'
            assert out.shape == (max(o['mask'].shape[0] for o in objects)+2,
                                 sum(o['mask'].shape[1] for o in objects)+len(objects)-1)
            # Verify each unchanged oriented mask directly in its own output slot.
            left = 0
            for obj, shape in zip(objects, keys):
                mask = obj['mask']
                top = 0 if counts[shape] > 1 else out.shape[0] - mask.shape[0]
                slot = out[:, left:left+mask.shape[1]]
                expected = np.full(slot.shape, bg, dtype=np.int8)
                expected[top:top+mask.shape[0]][mask] = obj['color']
                assert np.array_equal(slot, expected), 'slot mask/color/orientation/destination changed'
                left += mask.shape[1]
                if left < out.shape[1]:
                    assert np.all(out[:, left] == bg), 'slot gutter not blank'
                left += 1
            assert sorted(int(v) for v in mid[mid != bg]) == sorted(int(v) for v in out[out != bg])
            assert len(_objects(out, bg)) == len(objects), 'objects merged or split'
            assert set(int(c) for c in np.unique(inp)) == set(int(c) for c in np.unique(out))
            assert _different(mid, out), 'A packing omitted'
            assert _different(_pack(_objects(inp, bg), bg), out), 'B contour generation omitted'
            features[split].append(_features(inp, detailed=True))
    train, test = features['train'], features['test']
    assert sum(bool(f['up'] and f['down']) for f in train) >= 2
    assert any(f['sources'] == 1 and f['area'] for f in train), 'area shortcut unchallenged'
    for witness in ('identity', 'area', 'color', 'exactly_two', 'all_top', 'all_bottom',
                    'ties', 'ties_classes', 'ownership', 'ownership_classes',
                    'radial', 'radial_classes', 'parity'):
        assert any(f[witness] for f in train), 'missing training witness: ' + witness
    assert not ({f['bits'] for f in train} & {f['bits'] for f in test}), 'test corner subset already shown'
    # A four-connected alternative is only a relevant challenge when it differs.
    if any(f['diagonal'] for f in test):
        assert any(f['diagonal'] for f in train), 'diagonal connectivity not demonstrated'
    return True


def self_test():
    """42 seeds, streaming measurements, all explicit counts, no saved payloads."""
    signatures, structures, dimensions, backgrounds, masks = set(), set(), set(), set(), set()
    counts, multiplicities, shifts = set(), set(), set()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        train_max = max(max(p['input'].shape) for p in puzzle['train'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                bg, sources, bits = _decode(inp)
                digest.update(bytes(inp.shape) + inp.tobytes() + bytes(out.shape) + out.tobytes())
                normalized = np.full(inp.shape, 9, dtype=np.int8)
                for i, (y, x, _) in enumerate(sources):
                    normalized[y, x] = i
                structures.add(_signature(normalized))
                dimensions.add(tuple(inp.shape))
                backgrounds.add(bg)
                masks.add(bits)
                f = _features(inp)
                counts.add(f['objects'])
                multiplicities.add(f['multiplicity'])
                if split == 'test':
                    shifts.add(int(np.sign(max(inp.shape)-train_max)))
        signatures.add(digest.digest())
        if seed == 0:
            sample = puzzle
    assert len(signatures) == 42 and len(structures) >= 100
    assert len(dimensions) >= 40 and backgrounds == set(range(10))
    assert masks == set(range(1, 16)) and len(counts) >= 7
    assert max(multiplicities) >= 4 and shifts == {-1, 0, 1}
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[r], b[r]) for s in ('train', 'test')
               for a, b in zip(sample[s], repeated[s]) for r in ('input', 'output'))
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+tr*10+te, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{r: v.copy() for r, v in p.items()} for p in sample[s]] for s in sample}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures),
            'input_structures': len(structures), 'input_dimensions': len(dimensions),
            'object_counts': sorted(counts), 'max_class_size': max(multiplicities)}
