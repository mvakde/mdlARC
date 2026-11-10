"""Paired hollow/solid shutters: mirrored transport with size-ordered occlusion.

Only the standard library and NumPy are used. See task_spec.md for the rule.
"""
from __future__ import annotations

import copy
import json
import numpy as np


HOST_SEEDS = (0, 43665923, 905612328, 550544268, 827159646, 345517163,
              292789726, 871125715, 862070344, 752790585, 824313442, 76694492,
              873402642, 76983717, 482848029, 210908687, 942892818, 373381467,
              647318488, 53868031, 734183719, 566497174, 884251787, 835360518,
              24709231, 376184276, 452372090, 65705644, 229557560, 65256905,
              306360051, 883211348, 751247602, 952085042, 954525511, 770598740,
              331451712, 961296634, 625049630, 849578336, 574305266, 687412747)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert a.dtype.kind in 'iu' and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _paint(a, box, color, solid):
    r, c, h, w = box
    assert 0 <= r <= a.shape[0] - h and 0 <= c <= a.shape[1] - w
    if solid:
        a[r:r+h, c:c+w] = color
    else:
        a[r, c:c+w] = color
        a[r+h-1, c:c+w] = color
        a[r:r+h, c] = color
        a[r:r+h, c+w-1] = color


def _render(meta, *, reverse=False, all_solid=False, stationary=False,
            one_step=False, passive_stationary=False, clear_frames=False,
            reading_order=False, pins_last=False):
    """Composite transported objects in canonical (guides-left) coordinates.

    Nondefault flags are intentionally wrong alternatives used by validation;
    solve always uses the default, single family rule.
    """
    H, W = meta['shape']
    a = np.zeros((H, W), dtype=np.int8)
    a[:, W//2] = meta['rail']
    records = sorted(meta['records'], key=lambda z: -z['box'][2], reverse=reverse)
    if reading_order:
        def source_reading_key(z):
            r, c, h, w = z['box']
            if meta.get('flip', False):
                c = W-c-w
            return (c, r) if meta.get('transpose', False) else (r, c)
        records = sorted(meta['records'], key=source_reading_key)
    deferred_pins = []
    for z in records:
        r, c, h, w = z['box']
        dy, dx = z['direction']
        length = 0 if stationary else (1 if one_step else z['length'])
        for side in (0, 1):
            k = 0 if side and passive_stationary else length
            box = (r+k*dy, c+k*dx, h, w) if side == 0 else (
                r+k*dy, W-c-w-k*dx, h, w)
            solid = z['solid'] if side == 0 else not z['solid']
            if clear_frames and not solid:
                br, bc, bh, bw = box
                a[br:br+bh, bc:bc+bw] = 0
            _paint(a, box, z['color'], solid or all_solid)
            pr, pc = z['pin']
            pin = (pr+k*dy, pc+k*dx) if side == 0 else (
                pr+k*dy, W-1-pc-k*dx)
            if pins_last:
                deferred_pins.append(pin)
            else:
                a[pin] = meta['pin_color']
    for pin in deferred_pins:
        a[pin] = meta['pin_color']
    return a


def _restore(a, meta):
    if meta.get('flip', False):
        a = np.fliplr(a)
    if meta.get('transpose', False):
        a = a.T
    return np.ascontiguousarray(a)


def _wallpaper(a, rail):
    """Recover periods 1..3 from the untouched outer three-cell border."""
    rr, cc = np.indices(a.shape)
    border = (rr < 3) | (rr >= a.shape[0]-3) | (cc < 3) | (cc >= a.shape[1]-3)
    border &= a != rail
    candidates = []
    for th in range(1, 4):
        for tw in range(1, 4):
            tile = np.empty((th, tw), dtype=np.int8)
            valid = True
            for y in range(th):
                for x in range(tw):
                    vals = np.unique(a[border & (rr % th == y) & (cc % tw == x)])
                    if len(vals) != 1:
                        valid = False
                        break
                    tile[y, x] = vals[0]
                if not valid:
                    break
            if valid and 2 <= len(np.unique(tile)) <= 3:
                wp = tile[rr % th, cc % tw]
                if np.array_equal(wp[border], a[border]):
                    candidates.append((th*tw, th, tw, tile, wp))
    assert candidates, 'no periodic border wallpaper'
    candidates.sort(key=lambda t: t[:3])
    assert all(np.array_equal(t[4], candidates[0][4]) for t in candidates)
    return candidates[0][3], candidates[0][4]


def _decode(value):
    """Analyze only input pixels; generation metadata is never consulted."""
    original = _grid(value)
    rails = []
    for transpose in (False, True):
        a = original.T if transpose else original
        if a.shape[1] % 2:
            middle = a.shape[1]//2
            color = int(a[0, middle])
            if np.all(a[:, middle] == color) and np.count_nonzero(a == color) == a.shape[0]:
                rails.append((transpose, color))
    assert len(rails) == 1, 'one full centered mirror rail is required'
    transpose, rail = rails[0]
    assert rail != 0
    a = (original.T if transpose else original).copy()
    tile, wallpaper = _wallpaper(a, rail)
    wp_colors = set(int(c) for c in tile.flat)
    residual = set(int(c) for c in np.unique(a)) - wp_colors - {rail}
    depth = a.shape[1]//2
    one_sided = [c for c in residual if bool(np.any(a[:, :depth] == c)) !=
                 bool(np.any(a[:, depth+1:] == c))]
    assert len(one_sided) == 1, 'exactly one hue identifies the active guides'
    guide = one_sided[0]
    flip = not np.any(a[:, :depth] == guide)
    if flip:
        a = np.fliplr(a).copy()
        wallpaper = np.fliplr(wallpaper).copy()
    rect_colors = [c for c in residual - {guide}
                   if np.count_nonzero(a[:, :depth] == c) >= 7]
    assert 2 <= len(rect_colors) <= 3
    pin_colors = residual - {guide} - set(rect_colors)
    assert len(pin_colors) == 1
    pin_color = pin_colors.pop()
    assert 0 not in residual
    guides = {tuple(map(int, p)) for p in np.argwhere(a == guide)}
    pins = {tuple(map(int, p)) for p in np.argwhere(a == pin_color)}
    records, used_guides, used_pins = [], set(), set()
    rebuilt = wallpaper.copy()
    rebuilt[:, depth] = rail
    H, W = a.shape
    for color in rect_colors:
        positions = np.argwhere(a[:, :depth] == color)
        r, c = map(int, positions.min(axis=0))
        bottom, right = map(int, positions.max(axis=0))
        h, w = bottom-r+1, right-c+1
        assert min(h, w) >= 3
        box = (r, c, h, w)
        assert r >= 3 and r+h <= H-3 and c >= 3 and c+w <= depth
        opposite = np.argwhere(a[:, depth+1:] == color)
        opposite[:, 1] += depth+1
        assert tuple(map(int, opposite.min(axis=0))) == (r, W-c-w)
        assert tuple(map(int, opposite.max(axis=0))) == (r+h-1, W-c-1)
        corners = {(r, c), (r, c+w-1), (r+h-1, c), (r+h-1, c+w-1)}
        owners = corners & pins
        assert len(owners) == 1
        pin = owners.pop()
        partner = (pin[0], W-1-pin[1])
        assert partner in pins
        neighbors = [(dy, dx) for dy, dx in DIRECTIONS
                     if (pin[0]+dy, pin[1]+dx) in guides]
        assert len(neighbors) == 1
        dy, dx = neighbors[0]
        # The guide leaves its own corner outward, not through its body.
        assert (dy == -1 and pin[0] == r) or (dy == 1 and pin[0] == r+h-1) or (
            dx == -1 and pin[1] == c) or (dx == 1 and pin[1] == c+w-1)
        trace = []
        k = 1
        while (pin[0]+k*dy, pin[1]+k*dx) in guides:
            trace.append((pin[0]+k*dy, pin[1]+k*dx))
            k += 1
        assert trace and not (set(trace) & used_guides)
        assert len(trace) <= 5
        assert all(3 <= y < H-3 and 3 <= x < depth for y, x in trace)
        solid = bool(np.any(a[r+1:r+h-1, c+1:c+w-1] == color))
        other_solid = bool(np.any(a[r+1:r+h-1, W-c-w+1:W-c-1] == color))
        assert solid != other_solid
        z = dict(box=box, color=color, solid=solid, pin=pin,
                 direction=(dy, dx), length=len(trace), trace=tuple(trace))
        records.append(z)
        _paint(rebuilt, box, color, solid)
        _paint(rebuilt, (r, W-c-w, h, w), color, other_solid)
        rebuilt[pin] = rebuilt[partner] = pin_color
        for cell in trace:
            rebuilt[cell] = guide
        used_guides.update(trace)
        assert not ({pin, partner} & used_pins)
        used_pins.update((pin, partner))
    assert used_pins == pins and used_guides == guides
    assert np.array_equal(a, rebuilt), 'objects, pins and wallpaper must explain all cells'
    records.sort(key=lambda z: -z['box'][2])
    assert all(a['box'][2] > b['box'][2] and a['box'][3] > b['box'][3]
               for a, b in zip(records, records[1:])), 'no size ties or incomparable sizes'
    for i, z in enumerate(records):
        assert all(_compatible(z, old) for old in records[:i])
        r, c, h, w = z['box']
        dy, dx = z['direction']
        k = z['length']
        assert 0 <= r+k*dy <= H-h and 0 <= c+k*dx <= depth-w
    return dict(shape=a.shape, rail=rail, pin_color=pin_color, guide_color=guide,
                records=records, tile=tile, wallpaper=wallpaper,
                transpose=transpose, flip=flip, canonical_input=a)


def solve(input_grid: np.ndarray) -> np.ndarray:
    meta = _decode(input_grid)
    return _restore(_render(meta), meta)


def _inside(cell, box, halo=0):
    y, x = cell
    r, c, h, w = box
    return r-halo <= y < r+h+halo and c-halo <= x < c+w+halo


def _compatible(z, old):
    r, c, h, w = z['box']
    R, C, h2, w2 = old['box']
    # Leave at least one wallpaper cell between source bounding boxes.
    if not (r+h < R or R+h2 < r or c+w < C or C+w2 < c):
        return False
    if any(_inside(p, old['box']) for p in z['trace']):
        return False
    if any(_inside(p, z['box']) for p in old['trace']):
        return False
    distance = lambda p, q: abs(p[0]-q[0])+abs(p[1]-q[1])
    if any(distance(p, q) <= 1 for p in z['trace'] for q in old['trace']):
        return False
    if any(distance(z['pin'], q) <= 1 for q in old['trace']):
        return False
    if any(distance(old['pin'], p) <= 1 for p in z['trace']):
        return False
    return True


def _sample_tile(rng, excluded):
    available = [c for c in range(10) if c not in excluded]
    palette = rng.choice(available, int(rng.integers(2, min(3, len(available))+1)), replace=False)
    while True:
        th, tw = map(int, rng.integers(1, 4, size=2))
        if th*tw < 2:
            continue
        tile = np.asarray(rng.choice(palette, (th, tw)), dtype=np.int8)
        if len(np.unique(tile)) >= 2:
            return tile


def _make_pair(rng, n, palette, transpose, *, required_axis=None,
               outer_solid=None, long=False, reading_witness=False):
    """Sample separated sources and reject until transported layers really collide."""
    for scene_attempt in range(1800):
        H = int(rng.integers(23 if n == 3 else 20, 31))
        depth = int(rng.integers(12 if long else 10, 15))
        W = 2*depth+1
        terminal = rng.integers(3, 5, size=2)
        sizes = [tuple(map(int, terminal))]
        for _ in range(n-1):
            sizes.append(tuple(sizes[-1][j]+int(rng.integers(1, 3)) for j in (0, 1)))
        sizes.reverse()
        rect_colors = palette[:n]
        rail, pin_color, guide_color = palette[3:6]
        records = []
        for index, (h, w) in enumerate(sizes):
            if h > H-6 or w > depth-3:
                break
            length = int(rng.integers(4, 6)) if long and index == n-1 else int(rng.integers(1, 4))
            if required_axis is not None and index == 0:
                length = int(rng.integers(2, 4))
            for _ in range(160):
                r = int(rng.integers(3, H-3-h+1))
                c = int(rng.integers(3, depth-w+1))
                bottom, right = bool(rng.integers(2)), bool(rng.integers(2))
                pin = (r+h-1 if bottom else r, c+w-1 if right else c)
                vertical = bool(rng.integers(2))
                if required_axis is not None and index == 0:
                    vertical = required_axis == 0
                    if transpose:
                        vertical = not vertical
                dy, dx = ((1 if bottom else -1), 0) if vertical else (0, (1 if right else -1))
                trace = tuple((pin[0]+k*dy, pin[1]+k*dx) for k in range(1, length+1))
                if not all(3 <= y < H-3 and 3 <= x < depth for y, x in trace):
                    continue
                solid = bool(rng.integers(2)) if outer_solid is None or index else outer_solid
                z = dict(box=(r, c, h, w), color=rect_colors[index], solid=solid,
                         pin=pin, direction=(dy, dx), length=length, trace=trace)
                if all(_compatible(z, old) for old in records):
                    records.append(z)
                    break
            else:
                break
        if len(records) != n:
            continue
        meta = dict(shape=(H, W), rail=rail, pin_color=pin_color, records=records,
                    transpose=transpose, flip=bool(rng.integers(2)))
        answer = _render(meta)
        if np.array_equal(answer, _render(meta, reverse=True)):
            continue
        if np.array_equal(answer, _render(meta, all_solid=True)):
            continue
        if required_axis is not None and np.array_equal(answer, _render(meta, one_step=True)):
            continue
        if required_axis == 0 and np.array_equal(answer, _render(meta, clear_frames=True)):
            continue
        if required_axis == 0 and np.array_equal(answer, _render(meta, pins_last=True)):
            continue
        if reading_witness and np.array_equal(answer, _render(meta, reading_order=True)):
            continue
        tile = _sample_tile(rng, set(rect_colors) | {rail, pin_color, guide_color})
        rr, cc = np.indices((H, W))
        inp = tile[rr % tile.shape[0], cc % tile.shape[1]].copy()
        inp[:, depth] = rail
        for z in records:
            r, c, h, w = z['box']
            _paint(inp, z['box'], z['color'], z['solid'])
            _paint(inp, (r, W-c-w, h, w), z['color'], not z['solid'])
            pr, pc = z['pin']
            inp[pr, pc] = inp[pr, W-1-pc] = pin_color
            for cell in z['trace']:
                inp[cell] = guide_color
        return {'input': _restore(inp, meta), 'output': _restore(answer, meta)}
    raise RuntimeError('could not place unambiguous colliding shutters')


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and smaller than num_train')
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # longer travel, more shutters, or turned rail
    base_orientation = bool(rng.integers(2))
    base_palette = list(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
    variable_palette = bool(rng.integers(2))
    first_material = bool(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    seen, train_outputs = set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            for _ in range(30):
                palette = list(map(int, rng.choice(np.arange(1, 10), 6, replace=False))) if variable_palette else base_palette
                n = (2 if split == 'train' else 3) if mode == 1 else int(rng.integers(2, 4))
                transpose = not base_orientation if mode == 2 and split == 'test' else base_orientation
                try:
                    pair = _make_pair(rng, n, palette, transpose,
                                      required_axis=i if split == 'train' and i < 2 else None,
                                      outer_solid=(first_material if i == 0 else not first_material)
                                      if split == 'train' and i < 2 else None,
                                      long=(mode == 0 and split == 'test'),
                                      reading_witness=(split == 'train' and i == 1))
                except RuntimeError:
                    continue
                ikey = _key(pair['input'])
                okey = _key(pair['output'])
                if ikey not in seen and (split != 'train' or okey not in train_outputs):
                    seen.add(ikey)
                    if split == 'train':
                        train_outputs.add(okey)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not create distinct demonstrations')
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, train_outputs, training_axes, training_lengths = set(), set(), set(), set()
    train_materials = set()
    distal_witness = False
    transparency_witness = False
    attachment_witness = False
    reading_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            m = _decode(inp)
            canonical = _render(m)
            assert out.shape == inp.shape and np.array_equal(out, _restore(canonical, m))
            assert not np.array_equal(inp, out)
            assert m['guide_color'] not in out
            assert set(map(int, out.flat)) <= {0, m['rail'], m['pin_color']} | {z['color'] for z in m['records']}
            assert np.count_nonzero(out == m['pin_color']) <= 2*len(m['records'])
            assert np.all(canonical[:, canonical.shape[1]//2] == m['rail'])
            assert not np.array_equal(canonical, _render(m, reverse=True)), 'collision must expose size precedence'
            assert not np.array_equal(canonical, _render(m, all_solid=True)), 'frame transparency must matter'
            assert not np.array_equal(canonical, _render(m, stationary=True)), 'motion must matter'
            assert not np.array_equal(canonical, _render(m, passive_stationary=True)), 'mirrored motion must matter'
            depth = canonical.shape[1]//2
            assert not np.array_equal(canonical[:, :depth], np.fliplr(canonical[:, depth+1:])), 'materials must prevent copying a finished panel'
            if split == 'train':
                distal_witness |= not np.array_equal(canonical, _render(m, one_step=True))
                transparency_witness |= not np.array_equal(canonical, _render(m, clear_frames=True))
                attachment_witness |= not np.array_equal(canonical, _render(m, pins_last=True))
                reading_witness |= not np.array_equal(canonical, _render(m, reading_order=True))
                okey = _key(out)
                assert okey not in train_outputs
                train_outputs.add(okey)
                train_materials.add(m['records'][0]['solid'])
                for z in m['records']:
                    dy, dx = z['direction']
                    training_axes.add((0 if dy else 1) ^ int(m['transpose']))
                    training_lengths.add(z['length'])
    assert training_axes == {0, 1}
    assert max(training_lengths) >= 2 and distal_witness and train_materials == {False, True}
    assert transparency_witness, 'training must distinguish transparent frames from clearing interiors'
    assert attachment_witness, 'training must show that pins belong to their own layer'
    assert reading_witness, 'training must distinguish size order from source reading order'


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    shapes, layouts, sizes, palettes, tile_shapes = set(), set(), set(), set(), set()
    corners, directions, lengths, counts, orientations, sides, materials = (set() for _ in range(7))
    zero_present = set()
    shifts = set()
    total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_meta = [_decode(p['input']) for p in puzzle['train']]
        test_meta = [_decode(p['input']) for p in puzzle['test']]
        if max(z['length'] for m in test_meta for z in m['records']) > max(
                z['length'] for m in train_meta for z in m['records']):
            shifts.add('longer')
        if min(len(m['records']) for m in test_meta) > max(len(m['records']) for m in train_meta):
            shifts.add('more_layers')
        if not ({m['transpose'] for m in train_meta} & {m['transpose'] for m in test_meta}):
            shifts.add('turned')
        assert (max(z['length'] for m in test_meta for z in m['records']) >= 4 or
                min(len(m['records']) for m in test_meta) > max(len(m['records']) for m in train_meta) or
                not ({m['transpose'] for m in train_meta} & {m['transpose'] for m in test_meta}))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                total += 1
                m = _decode(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(pair['input'].shape)
                orientations.add(m['transpose'])
                sides.add(m['flip'])
                counts.add(len(m['records']))
                palettes.add(tuple(sorted(set(map(int, np.asarray(pair['input']).flat)))))
                tile_shapes.add(m['tile'].shape)
                zero_present.add(0 in m['tile'])
                layouts.add((m['shape'], tuple((z['box'], z['pin'], z['direction'], z['length']) for z in m['records'])))
                for z in m['records']:
                    r, c, h, w = z['box']
                    corners.add((z['pin'][0] == r, z['pin'][1] == c))
                    directions.add(z['direction'])
                    lengths.add(z['length'])
                    sizes.add((h, w))
                    materials.add(z['solid'])
    assert len(shapes) >= 25 and len(layouts) >= 190 and len(sizes) >= 15
    assert len(palettes) >= 15 and len(tile_shapes) >= 4
    assert counts == {2, 3} and orientations == sides == materials == zero_present == {False, True}
    assert len(corners) == 4 and directions == set(DIRECTIONS) and {1, 2, 3, 4, 5} <= lengths
    assert shifts == {'longer', 'more_layers', 'turned'}
    a = generate_puzzle(19, 2, 1)
    assert _same(a, generate_puzzle(19, 2, 1))
    for nt, ns in ((10, 4), (5, 4), (10, 1)):
        validate_generated_puzzle(generate_puzzle(29+nt, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = copy.deepcopy(a)
    bad['train'][0]['output'][0, 0] = 1
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output was accepted')
    return dict(seeds=42, pairs=total, input_shapes=len(shapes), layouts=len(layouts),
                rectangle_sizes=len(sizes), palettes=len(palettes), wallpaper_shapes=len(tile_shapes),
                guide_lengths=sorted(lengths), rectangle_pair_counts=sorted(counts),
                shifts=sorted(shifts))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
