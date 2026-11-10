"""Portal mirrors: count-addressed, stencil-printing specular transport.

Only Python/NumPy are required. Generation paints from a placement plan; solve
reconstructs objects and encounters exclusively from its input. Nothing is saved
by this module. All optional reports use compact JSON in the experiment script.
"""
from __future__ import annotations

import hashlib
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _N4 + _DIAGONALS if diagonal else _N4
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, cells = [start], []
        while todo:
            p = todo.pop()
            cells.append(p)
            for dr, dc in steps:
                q = p[0] + dr, p[1] + dc
                if q in unseen:
                    unseen.remove(q)
                    todo.append(q)
        result.append(cells)
    return result


def _box(cells):
    r, c = zip(*cells)
    return min(r), min(c), max(r) - min(r) + 1, max(c) - min(c) + 1


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _in_box(p, box):
    r, c, h, w = box
    return r <= p[0] < r + h and c <= p[1] < c + w


def _separate(a, b):
    r, c, h, w = a
    x, y, u, v = b
    return r + h < x or x + u < r or c + w < y or y + v < c


def _center(box):
    r, c, h, w = box
    return r + h // 2, c + w // 2


def _face(box, axis, sign):
    p = list(_center(box))
    p[axis] -= sign * (box[2 + axis] // 2 + 1)
    return tuple(p)


def _contact(box, p, direction):
    axes = [a for a in (0, 1) if p == _face(box, a, direction[a])]
    assert len(axes) <= 1
    return axes[0] if axes else None


def _perimeter(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _decode(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > grid.size // 2, 'background must be a unique majority'
    objects, source = [], None
    for cells in _components(grid != bg):
        box = _box(cells)
        r, c, h, w = box
        a = grid[r:r+h, c:c+w]
        colors = np.unique(a[a != bg])
        assert len(colors) == 1, 'objects are monochromatic except for background'
        color = int(colors[0])
        if len(cells) == 3 and h == w == 2:
            assert source is None, 'exactly one emitter'
            missing = next((x, y) for x in (r, r+1) for y in (c, c+1)
                           if grid[x, y] == bg)
            elbow = 2*r+1-missing[0], 2*c+1-missing[1]
            direction = elbow[0]-missing[0], elbow[1]-missing[1]
            source = dict(box=box, color=color, elbow=elbow, direction=direction)
        else:
            assert h == w and h in (5, 7, 9)
            assert np.all(_perimeter(a) == color), 'complete perimeter required'
            objects.append(dict(box=box, color=color, array=a.copy()))
    assert source is not None and len(objects) in (2, 4, 6, 8)
    sizes = sorted({o['box'][2] for o in objects})
    assert len(sizes) == 2 and sizes[0] in (5, 7) and sizes[1] == sizes[0] + 2
    s = sizes[0]
    motifs, tiles = {}, {}
    for o in objects:
        mask = o['array'] == bg
        if o['box'][2] == s:
            regions = _components(mask)
            k = len(regions)
            o['regions'] = regions
            assert k not in motifs and 0 <= k <= 5
            motifs[k] = o
        else:
            assert not np.any(mask[1:] & mask[:-1])
            assert not np.any(mask[:, 1:] & mask[:, :-1]), 'pips must be isolated'
            k = int(mask.sum())
            assert k not in tiles and 0 <= k <= 5
            tiles[k] = o
        o['count'] = k
    assert set(tiles) == set(motifs) and len(tiles) * 2 == len(objects)
    assert all(tiles[k]['color'] != motifs[k]['color'] for k in tiles)
    boxes = [source['box']] + [o['box'] for o in objects]
    assert all(_separate(a, b) for i, a in enumerate(boxes) for b in boxes[i+1:])
    return dict(grid=grid, background=bg, source=source, motifs=motifs,
                tiles=tiles, boxes=boxes, size=s)


def _print_tile(output, tile, motif, bg):
    r, c, h, w = tile['box']
    output[r:r+h, c:c+w] = tile['color']
    output[r+1:r+h-1, c+1:c+w-1] = np.where(
        motif['array'] == bg, tile['color'], motif['array'])


def _trace(scene, local=False, color_mode='motif'):
    grid, bg = scene['grid'], scene['background']
    output = grid.copy()
    source = scene['source']
    direction, color = source['direction'], source['color']
    p = tuple(source['elbow'][a] + direction[a] for a in (0, 1))
    visited, painted, active, events = set(), {}, set(), []

    def paint(q, shade):
        assert _inside(q, grid.shape) and grid[q] == bg
        assert not any(_in_box(q, b) for b in scene['boxes']), 'ray hits object box'
        assert q not in painted or painted[q] == shade, 'conflicting ray cells'
        painted[q] = shade
        output[q] = shade

    while _inside(p, grid.shape):
        state = p, direction
        assert state not in visited, 'cyclic optical route'
        visited.add(state)
        assert len(visited) <= 4 * grid.size
        hits = [(k, a) for k, tile in scene['tiles'].items()
                if (a := _contact(tile['box'], p, direction)) is not None]
        assert len(hits) <= 1, 'ambiguous midpoint contact'
        if hits:
            k, axis = hits[0]
            assert k not in active, 'generated routes visit tiles once'
            active.add(k)
            tile, motif = scene['tiles'][k], scene['motifs'][k]
            if color_mode == 'motif':
                color = motif['color']
            elif color_mode == 'carrier':
                color = tile['color']
            elif color_mode == 'source':
                color = source['color']
            else:
                raise ValueError('unknown color ablation')
            _print_tile(output, tile, motif, bg)
            paint(p, color)
            exit_point = p if local else _face(motif['box'], axis, direction[axis])
            events.append(dict(count=k, axis=axis, incoming=direction,
                               entry=p, exit=exit_point, color=color))
            paint(exit_point, color)
            direction = tuple(-d if a == axis else d for a, d in enumerate(direction))
            p = exit_point
        else:
            paint(p, color)
        p = tuple(p[a] + direction[a] for a in (0, 1))
    assert events, 'at least one portal must activate'
    return output, dict(events=events, active=active, painted=painted, end=p)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode count addresses and trace the sole beam, using input only."""
    return _trace(_decode(input_grid))[0]


def _free_origins(size, boxes, path):
    rr, cc = np.indices((31-size, 31-size))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, h, w in boxes:
        free &= (rr+size < r) | (r+h < rr) | (cc+size < c) | (c+w < cc)
    for r, c in path:
        free &= (rr > r) | (rr+size <= r) | (cc > c) | (cc+size <= c)
    return np.argwhere(free)


def _entry_options(anchor, direction, size, boxes, old_path, axis=None):
    flight, options = [], []
    for distance in range(1, 31):
        p = tuple(anchor[a] + distance*direction[a] for a in (0, 1))
        if not _inside(p, (30, 30)) or p in old_path or any(_in_box(p, b) for b in boxes):
            break
        flight.append(p)
        # An earlier printed carrier must not become a second entry.
        if any(b[2] == size and _contact(b, p, direction) is not None for b in boxes[1:]):
            break
        for a in ([axis] if axis is not None else (0, 1)):
            middle = list(p)
            middle[a] += direction[a] * (size//2+1)
            box = middle[0]-size//2, middle[1]-size//2, size, size
            r, c, h, w = box
            if r < 0 or c < 0 or r+h > 30 or c+w > 30:
                continue
            if not all(_separate(box, b) for b in boxes):
                continue
            if any(_in_box(q, box) for q in old_path) or any(_in_box(q, box) for q in flight):
                continue
            # No future tile is allowed to create an unintended earlier entry.
            if any(_contact(box, q, d) is not None for q, d in old_path.items()):
                continue
            options.append((p, a, box, tuple(flight)))
    return options


def _tail(anchor, direction, boxes, old_path, tile_size):
    result = []
    for distance in range(1, 31):
        p = tuple(anchor[a] + distance*direction[a] for a in (0, 1))
        if not _inside(p, (30, 30)):
            return result
        if p in old_path or any(_in_box(p, b) for b in boxes):
            return None
        if any(b[2] == tile_size and _contact(b, p, direction) is not None for b in boxes[1:]):
            return None
        result.append(p)
    return None


def _geometry(rng, active_count, total, s, first_axis, direction):
    """Bounded constructive placement; no grid payloads are retained on failure."""
    t = s+2
    for attempt in range(360):
        elbow = tuple(int(rng.integers(1, 12)) if d > 0 else int(rng.integers(18, 29))
                      for d in direction)
        cells = [elbow, (elbow[0]-direction[0], elbow[1]),
                 (elbow[0], elbow[1]-direction[1])]
        boxes, events, path = [_box(cells)], [], {}
        anchor, d = elbow, direction
        for hop in range(active_count):
            options = _entry_options(anchor, d, t, boxes, path,
                                     first_axis if hop == 0 else None)
            if hop == 0:
                options = [option for option in options if len(option[3]) >= 2]
            rng.shuffle(options)
            found = False
            for entry, axis, tile, flight in options[:24]:
                before = dict(path)
                before.update({p: d for p in flight})
                origins = _free_origins(s, boxes+[tile], before)
                rng.shuffle(origins)
                reflected = tuple(-v if a == axis else v for a, v in enumerate(d))
                for r, c in origins[:70]:
                    motif = int(r), int(c), s, s
                    exit_point = _face(motif, axis, d[axis])
                    if (not _inside(exit_point, (30, 30)) or exit_point in before
                            or any(_in_box(exit_point, b) for b in boxes+[tile])):
                        continue
                    trial_path = dict(before)
                    trial_path[exit_point] = reflected
                    trial_boxes = boxes+[tile, motif]
                    if hop + 1 == active_count:
                        tail = _tail(exit_point, reflected, trial_boxes, trial_path, t)
                        if tail is None or len(tail) < 2:
                            continue
                    elif not _entry_options(exit_point, reflected, t, trial_boxes, trial_path):
                        continue
                    events.append(dict(tile=tile, motif=motif, entry=entry, exit=exit_point,
                                       axis=axis, incoming=d, flight=flight))
                    boxes, path = trial_boxes, trial_path
                    anchor, d, found = exit_point, reflected, True
                    break
                if found:
                    break
            if not found:
                break
        if len(events) != active_count:
            continue
        path.update({p: d for p in tail})
        inactive = []
        for _ in range(total-active_count):
            placed = False
            choices = _free_origins(t, boxes, path)
            rng.shuffle(choices)
            for r, c in choices[:70]:
                tile = int(r), int(c), t, t
                if any(_contact(tile, p, v) is not None for p, v in path.items()):
                    continue
                origins = _free_origins(s, boxes+[tile], path)
                if not len(origins):
                    continue
                x, y = origins[int(rng.integers(len(origins)))]
                motif = int(x), int(y), s, s
                boxes.extend((tile, motif))
                inactive.append(dict(tile=tile, motif=motif))
                placed = True
                break
            if not placed:
                break
        if len(inactive) == total-active_count:
            return dict(cells=cells, source_box=boxes[0], events=events, inactive=inactive,
                        tail=tail, path=path, boxes=boxes, direction=direction)
    return None


def _cavity_mask(rng, s, k, witness=False):
    mask = np.zeros((s, s), dtype=bool)
    if k == 0:
        return mask
    if k == 1 and (witness or rng.random() < .40):
        mask[1:-1, 1:-1] = True
        return mask
    if k == 2 and witness:
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
        return np.rot90(mask.T if rng.integers(2) else mask, int(rng.integers(4))).copy()
    # Arbitrary isolated seeds, retaining connected foreground; a checkerboard
    # fallback guarantees capacity for all five allowed counts at s=5.
    for _ in range(20):
        mask[:] = False
        for i in rng.permutation((s-2)**2):
            r, c = divmod(int(i), s-2)
            r, c = r+1, c+1
            if any(mask[r+dr, c+dc] for dr, dc in _N4):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
            if int(mask.sum()) == k:
                break
        if int(mask.sum()) == k:
            break
    else:
        mask[:] = False
        points = [(r, c) for r in range(1, s-1) for c in range(1, s-1) if (r+c) % 2 == 0]
        for i in rng.permutation(len(points))[:k]:
            mask[points[int(i)]] = True
    target = int(rng.integers(k, (s-2)**2+1))
    for i in rng.permutation((s-2)**2):
        if int(mask.sum()) >= target:
            break
        r, c = divmod(int(i), s-2)
        r, c = r+1, c+1
        if mask[r, c] or not any(mask[r+dr, c+dc] for dr, dc in _N4):
            continue
        mask[r, c] = True
        if len(_components(mask)) != k or len(_components(~mask)) != 1:
            mask[r, c] = False
    assert len(_components(mask)) == k and len(_components(~mask)) == 1
    return mask


def _pip_mask(rng, s, k):
    mask = np.zeros((s, s), dtype=bool)
    for _ in range(24):
        mask[:] = False
        for i in rng.permutation(s*s):
            if int(mask.sum()) == k:
                return mask
            r, c = divmod(int(i), s)
            if any(_inside((r+dr, c+dc), mask.shape) and mask[r+dr, c+dc] for dr, dc in _N4):
                continue
            mask[r, c] = True
            if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                mask[r, c] = False
        if int(mask.sum()) == k:
            return mask
    raise AssertionError('pip independent set construction failed')


def _make_pair(rng, active, total, preferred_size, axis, direction, witness=False):
    geometry = _geometry(rng, active, total, preferred_size, axis, direction)
    s = preferred_size
    if geometry is None and s == 7:
        s = 5
        geometry = _geometry(rng, active, total, s, axis, direction)
    if geometry is None:
        raise RuntimeError('bounded portal placement exhausted')
    bg = int(rng.integers(10))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    pool = palette[:int(rng.integers(3, 10))]
    source_color = int(rng.choice(pool))
    grid = np.full((30, 30), bg, dtype=np.int8)
    for p in geometry['cells']:
        grid[p] = source_color
    keys = list(map(int, rng.choice(6, total, replace=False)))
    if witness:
        keys = list(map(int, rng.permutation([1, 2]))) + [int(rng.choice([0, 3, 4, 5]))]
    output = grid.copy()
    devices = geometry['events'] + geometry['inactive']
    for i, (key, device) in enumerate(zip(keys, devices)):
        carrier = int(rng.choice(pool))
        candidates = [c for c in pool if c != carrier and (i != 0 or c != source_color)]
        motif_color = int(rng.choice(candidates))
        cavities = _cavity_mask(rng, s, key, witness)
        motif = np.full((s, s), motif_color, dtype=np.int8)
        motif[cavities] = bg
        tile = np.full((s+2, s+2), carrier, dtype=np.int8)
        tile[1:-1, 1:-1][_pip_mask(rng, s, key)] = bg
        for name, array in (('tile', tile), ('motif', motif)):
            r, c, h, w = device[name]
            grid[r:r+h, c:c+w] = array
        device.update(count=key, carrier=carrier, color=motif_color, array=motif)
    # Label is painted from the constructive geometry, not by calling solve.
    output = grid.copy()
    color = source_color
    for device in geometry['events']:
        for p in device['flight'][:-1]:
            output[p] = color
        color = device['color']
        output[device['entry']] = color
        output[device['exit']] = color
        r, c, h, w = device['tile']
        output[r:r+h, c:c+w] = device['carrier']
        output[r+1:r+h-1, c+1:c+w-1] = np.where(
            device['array'] == bg, device['carrier'], device['array'])
    for p in geometry['tail']:
        output[p] = color
    # Variable margins/crop, preserving the complete planned beam and majority.
    occupied = np.argwhere((grid != bg) | (output != bg))
    low = occupied.min(axis=0)
    high = occupied.max(axis=0)+1
    for _ in range(12):
        r = int(rng.integers(int(low[0])+1))
        c = int(rng.integers(int(low[1])+1))
        h = int(rng.integers(int(high[0]), 31))
        w = int(rng.integers(int(high[1]), 31))
        candidate = grid[r:h, c:w]
        if np.count_nonzero(candidate == bg) > candidate.size//2:
            grid, output = candidate.copy(), output[r:h, c:w].copy()
            break
    assert np.array_equal(solve(grid), output), 'plan and input-only solver disagree'
    return dict(input=grid, output=output)


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(name + ' is outside the legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    nt, ne = _counts(num_train, num_test)
    first_axis = int(rng.integers(2))
    directions = [tuple(_DIAGONALS[int(i)]) for i in rng.permutation(4)]
    puzzle, used = dict(train=[], test=[]), set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            witness = split == 'train' and i == 0
            active = 3 if split == 'test' else 2 if i <= 1 else int(rng.integers(1, 3))
            total = active + (1 if witness else int(rng.random() < .30))
            size = 7 if total <= 3 and rng.random() < .35 else 5
            axis = first_axis if witness else 1-first_axis if split == 'train' and i == 1 else int(rng.integers(2))
            direction = directions[i % 4] if split == 'train' else _DIAGONALS[int(rng.integers(4))]
            for _ in range(8):
                pair = _make_pair(rng, active, total, size, axis, direction, witness)
                key = _key(pair['input'])
                if key not in used:
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
            used.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_differs(scene, expected, **kwargs):
    try:
        return not np.array_equal(_trace(scene, **kwargs)[0], expected)
    except AssertionError:
        # An ablation can run into an object where the real transported ray did
        # not. This is also a contradiction of the observed legal trajectory.
        return True


def _features(scene, info):
    motifs = scene['motifs']
    active_masks = [(k, motifs[k]['array'] == scene['background']) for k in info['active']]
    areas = {k: int(mask.sum()) for k, mask in active_masks}
    asymmetric = any(all(not np.array_equal(mask, m) for m in
                        [np.rot90(mask, j) for j in (1, 2, 3)] +
                        [np.rot90(mask.T, j) for j in range(4)]) for _, mask in active_masks)
    return dict(active=len(info['active']), inactive=len(scene['tiles'])-len(info['active']),
                axes={e['axis'] for e in info['events']}, direction=scene['source']['direction'],
                inversion=any(a < b and areas[a] > areas[b] for a in areas for b in areas),
                connectivity=any(len(_components(mask, True)) < k for k, mask in active_masks),
                asymmetric=asymmetric)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    used, train_features, omissions = set(), [], [False]*5
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            scene = _decode(inp)
            expected, info = _trace(scene)
            assert np.array_equal(out, expected) and not np.array_equal(inp, out)
            assert set(np.unique(out)) <= set(np.unique(inp))
            bg = scene['background']
            for motif in scene['motifs'].values():
                r, c, h, w = motif['box']
                assert np.array_equal(inp[r:r+h, c:c+w], out[r:r+h, c:c+w])
            for k, tile in scene['tiles'].items():
                r, c, h, w = tile['box']
                assert np.all(_perimeter(out[r:r+h, c:c+w]) == tile['color'])
                if k not in info['active']:
                    assert np.array_equal(inp[r:r+h, c:c+w], out[r:r+h, c:c+w])
            assert all(out[p] == scene['source']['color'] for p in
                       [scene['source']['elbow'],
                        (scene['source']['elbow'][0]-scene['source']['direction'][0], scene['source']['elbow'][1]),
                        (scene['source']['elbow'][0], scene['source']['elbow'][1]-scene['source']['direction'][1])])
            f = _features(scene, info)
            assert 1 <= f['active'] <= 3
            if split == 'train':
                train_features.append(f)
                fill_all = out.copy()
                for tile_key, tile in scene['tiles'].items():
                    _print_tile(fill_all, tile, scene['motifs'][tile_key], bg)
                omissions[0] |= not np.array_equal(fill_all, out)
                omissions[1] |= _ablation_differs(scene, out, local=True)
                omissions[2] |= _ablation_differs(scene, out, color_mode='source')
                omissions[3] |= _ablation_differs(scene, out, color_mode='carrier')
                if f['inversion']:
                    # Swapping two active addresses changes copying as well as
                    # routing; it directly tests the role of B's count match.
                    ks = sorted(info['active'])
                    a, b = next((a, b) for a in ks for b in ks if a < b
                                and np.count_nonzero(scene['motifs'][a]['array'] == bg)
                                > np.count_nonzero(scene['motifs'][b]['array'] == bg))
                    altered = dict(scene)
                    altered['motifs'] = dict(scene['motifs'])
                    altered['motifs'][a], altered['motifs'][b] = scene['motifs'][b], scene['motifs'][a]
                    omissions[4] |= _ablation_differs(altered, out)
    assert any(f['inversion'] and f['connectivity'] and f['asymmetric'] for f in train_features)
    assert any(f['inactive'] for f in train_features)
    assert {a for f in train_features for a in f['axes']} == {0, 1}
    assert len({f['direction'] for f in train_features}) >= 2
    assert max(f['active'] for f in train_features) >= 2
    assert all(omissions), 'training must refute omission/color shortcuts'


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Streaming 42-seed coverage; no logs, samples, or traces written."""
    shapes, structures, masks, keys, sizes, directions, faces, backgrounds = (set() for _ in range(8))
    max_area, inactive_pairs, zero_active = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                scene = _decode(inp)
                expected, info = _trace(scene)
                assert np.array_equal(expected, pair['output'])
                shapes.add(inp.shape)
                mapping = {scene['background']: 0}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add(hashlib.sha256(bytes(inp.shape)+normalized).digest())
                keys.add(tuple(sorted(scene['tiles'])))
                sizes.add(scene['size'])
                backgrounds.add(scene['background'])
                directions.add(scene['source']['direction'])
                faces.update((e['axis'], e['incoming'][e['axis']]) for e in info['events'])
                inactive_pairs += len(info['active']) < len(scene['tiles'])
                zero_active += 0 in info['active']
                for motif in scene['motifs'].values():
                    mask = motif['array'] == scene['background']
                    masks.add((mask.shape, mask.tobytes()))
                    max_area = max(max_area, max(map(len, motif['regions']), default=0))
                if split == 'test':
                    assert len(info['active']) == 3
        assert _equal(puzzle, generate_puzzle(seed))
    assert len(shapes) >= 12 and len(structures) >= 190 and len(masks) >= 60
    assert len(keys) >= 20 and sizes == {5, 7} and backgrounds == set(range(10))
    assert directions == set(_DIAGONALS) and len(faces) == 4
    assert max_area >= 9 and inactive_pairs >= 42 and zero_active >= 10
    # Visible orientation and face conventions commute with all scene D4
    # transforms. This probes solve beyond the sampler's returned orientation.
    specimen = generate_puzzle(0, 2, 1)['test'][0]
    for transpose in (False, True):
        for turns in range(4):
            inp = specimen['input'].T if transpose else specimen['input']
            out = specimen['output'].T if transpose else specimen['output']
            assert np.array_equal(solve(np.rot90(inp, turns)), np.rot90(out, turns))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1234+nt*10+ne, nt, ne)
            assert (len(puzzle['train']), len(puzzle['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None), ('4', 1), (4, np.bool_(True))):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(0, 2, 1)
    corrupted = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    p = corrupted['test'][0]
    r, c = np.argwhere(p['input'] != p['output'])[0]
    p['output'][r, c] = p['input'][r, c]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')


if __name__ == '__main__':
    self_test()
