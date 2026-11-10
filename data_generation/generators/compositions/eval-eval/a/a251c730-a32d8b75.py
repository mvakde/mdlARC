"""Keyed toggle stencils: learned macro geometry, binary tiles, XOR overlap.

Only standard-library modules and NumPy are used. All parsing is input-only.
The library and lattice can appear in either order along either axis.
"""
from itertools import permutations
import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_MAPS = tuple(permutations(range(4)))
HOST_SEEDS = (0, 931716272, 60007243, 484164510, 677828681, 906375072,
              802977707, 581160154, 202743827, 261065147, 512121722,
              680851296, 202874663, 176842753, 51128332, 47847209,
              627372623, 49324365, 530087747, 59102466, 796968808,
              568081408, 817488643, 560544833, 609339156, 668833070,
              200817642, 857130405, 9664444, 269561649, 386099839,
              137405614, 350947824, 741536385, 900362686, 434533213,
              706958420, 567615926, 455702599, 40121073, 160521925,
              913880636)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(int, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows)-min(rows)+1, max(cols)-min(cols)+1


def _crop(a, box):
    r, c, h, w = box
    return a[r:r+h, c:c+w]


def _disjoint(a, b):
    r, c, h, w = a
    y, x, v, u = b
    return r+h <= y or y+v <= r or c+w <= x or x+u <= c


def _rotate(offset, turn):
    r, c = offset
    for _ in range(turn % 4):
        r, c = c, -r
    return int(r), int(c)


def _read_lattice(area, line, n):
    h, w = area.shape
    assert (h-1) % (n+1) == (w-1) % (n+1) == 0
    rows, cols = (h-1)//(n+1), (w-1)//(n+1)
    assert rows >= 2 and cols >= 2
    rr, cc = np.indices(area.shape)
    lines = (rr % (n+1) == 0) | (cc % (n+1) == 0)
    assert np.array_equal(area == line, lines)
    blocks = [[area[r*(n+1)+1:r*(n+1)+n+1,
                         c*(n+1)+1:c*(n+1)+n+1] for c in range(cols)]
              for r in range(rows)]
    full = [b for row in blocks for b in row if len(np.unique(b)) == 2]
    assert full, "one complete, unmarked microtile is required"
    tile = full[0].copy()
    colors, frequencies = np.unique(tile, return_counts=True)
    assert frequencies.min() >= 3
    swapped = colors.sum() - tile
    phases = np.zeros((rows, cols), int)
    triggers = []
    restored = area.copy()
    center = n//2, n//2
    for r in range(rows):
        for c in range(cols):
            b = blocks[r][c]
            marked = ~np.isin(b, colors)
            if marked.any():
                positions = set(map(tuple, np.argwhere(marked)))
                assert len(positions) == 2 and center in positions
                extra = next(p for p in positions if p != center)
                direction = extra[0]-center[0], extra[1]-center[1]
                assert direction in _D4
                keys = np.unique(b[marked])
                assert len(keys) == 1 and keys[0] != line
                triggers.append((r, c, int(keys[0]), _D4.index(direction)))
            possibilities = [phase for phase, t in enumerate((tile, swapped))
                             if np.array_equal(b[~marked], t[~marked])]
            assert len(possibilities) == 1, "tiles agree up to full binary exchange"
            phases[r, c] = possibilities[0]
            restored[r*(n+1)+1:r*(n+1)+n+1,
                     c*(n+1)+1:c*(n+1)+n+1] = (tile, swapped)[possibilities[0]]
    assert triggers
    return dict(n=n, shape=(rows, cols), tile=tile, colors=tuple(map(int, colors)),
                phases=phases, triggers=triggers, restored=restored, line=line)


def _dictionary(area):
    fill = int(area[1, 1])
    inside = area[1:-1, 1:-1]
    assert np.all(inside[0] == fill) and np.all(inside[-1] == fill)
    assert np.all(inside[:, 0] == fill) and np.all(inside[:, -1] == fill)
    signatures, repetitions = {}, {}
    for cells in _components(inside != fill):
        assert 3 <= len(cells) <= 12
        values, counts = np.unique([inside[r, c] for r, c in cells], return_counts=True)
        assert len(values) == 2 and sorted(counts) == [1, len(cells)-1]
        key = int(values[np.argmin(counts)])
        ar, ac = next(p for p in cells if inside[p] == key)
        signature = tuple(sorted((r-ar, c-ac, int(inside[r, c])) for r, c in cells))
        assert key not in signatures or signatures[key] == signature
        signatures[key] = signature
        repetitions[key] = repetitions.get(key, 0)+1
    assert 1 <= len(signatures) <= 3
    keys = set(signatures)
    assert all(color not in keys for sig in signatures.values()
               for r, c, color in sig if (r, c) != (0, 0))
    templates = {k: tuple((r, c) for r, c, _ in sig) for k, sig in signatures.items()}
    assert all(max(_bounds(t)[2:]) <= 5 for t in templates.values())
    return dict(fill=fill, templates=templates, signatures=signatures,
                repetitions=repetitions)


def _decode(input_grid):
    grid = _grid(input_grid)
    components = [(color, cells, _bounds(cells)) for color in np.unique(grid)
                  for cells in _components(grid == color)]
    lattices = []
    for color, cells, box in components:
        r, c, h, w = box
        for n in (3, 5):
            if min(h, w) < 2*(n+1)+1 or (h-1) % (n+1) or (w-1) % (n+1):
                continue
            area = _crop(grid, box)
            try:
                info = _read_lattice(area, int(color), n)
                info['box'] = box
                lattices.append(info)
            except (AssertionError, StopIteration):
                continue
    assert len(lattices) == 1, "expected one visibly subdivided tile lattice"
    info = lattices[0]
    libraries = []
    for color, cells, box in components:
        r, c, h, w = box
        if min(h, w) < 5 or not _disjoint(box, info['box']):
            continue
        if len(cells) != 2*h+2*w-4:
            continue
        area = _crop(grid, box)
        if not (np.all(area[0] == color) and np.all(area[-1] == color)
                and np.all(area[:, 0] == color) and np.all(area[:, -1] == color)):
            continue
        try:
            lib = _dictionary(area)
            assert lib['fill'] != color
            assert all(k in lib['templates'] for _, _, k, _ in info['triggers'])
            assert set(lib['templates']).isdisjoint((*info['colors'], info['line'], lib['fill']))
            lib.update(box=box, frame=int(color))
            libraries.append(lib)
        except AssertionError:
            continue
    assert len(libraries) == 1, "expected one consistent keyed motif library"
    info.update(library=libraries[0], templates=libraries[0]['templates'])
    # The exterior is a neutral, discarded carrier, never part of a stencil.
    exterior = np.ones(grid.shape, bool)
    for box in (info['box'], info['library']['box']):
        _crop(exterior, box)[:] = False
    assert exterior.any() and np.all(grid[exterior] == info['library']['fill'])
    assert info['library']['fill'] != info['line']
    _counts(info, strict=True)
    return info


def _counts(info, mapping=(0, 1, 2, 3), behavior='normal', strict=False):
    hits = np.zeros(info['shape'], int)
    turns = np.zeros(info['shape'], int)
    first = next(iter(info['templates'].values()))
    for ar, ac, key, direction in info['triggers']:
        stencil = first if behavior == 'one_template' else info['templates'][key]
        if behavior == 'anchor_only':
            stencil = ((0, 0),)
        k = 0 if behavior == 'no_rotation' else mapping[direction]
        for offset in stencil:
            if behavior == 'omit_anchor' and offset == (0, 0):
                continue
            dr, dc = _rotate(offset, k)
            r, c = ar+dr, ac+dc
            if 0 <= r < hits.shape[0] and 0 <= c < hits.shape[1]:
                hits[r, c] += 1
                turns[r, c] += k
            else:
                assert not strict, "complete rotated stencil must fit; no clipping"
    return hits, turns


def _render(info, mapping=(0, 1, 2, 3), behavior='normal'):
    hits, turns = _counts(info, mapping, behavior)
    out = info['restored'].copy()
    n = info['n']
    for r in range(hits.shape[0]):
        for c in range(hits.shape[1]):
            block = np.s_[r*(n+1)+1:r*(n+1)+n+1, c*(n+1)+1:c*(n+1)+n+1]
            tile = info['restored'][block].copy()
            toggle = hits[r, c] % 2
            if behavior == 'union':
                toggle = hits[r, c] > 0
            if behavior == 'no_exchange':
                toggle = False
            if behavior == 'rotate_microtile':
                tile = np.rot90(tile, -int(turns[r, c] % 4))
            if toggle:
                tile = sum(info['colors']) - tile
            if behavior in ('paint_phase_0', 'paint_phase_1') and hits[r, c]:
                tile = info['tile'] if behavior == 'paint_phase_0' else sum(info['colors'])-info['tile']
            out[block] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover hidden arrow pixels, then XOR the rotated keyed action masks."""
    return _render(_decode(input_grid))


def _shape(rng, limit, max_cells):
    h, w = int(rng.integers(2, limit+1)), int(rng.integers(2, limit+1))
    count = int(rng.integers(3, min(max_cells, h*w)+1))
    occupied = {(int(rng.integers(h)), int(rng.integers(w)))}
    style = int(rng.integers(3))
    while len(occupied) < count:
        frontier = sorted({(r+dr, c+dc) for r, c in occupied for dr, dc in _D4
                           if 0 <= r+dr < h and 0 <= c+dc < w} - occupied)
        neighbors = np.array([sum((r+dr, c+dc) in occupied for dr, dc in _D4)
                              for r, c in frontier])
        weights = (np.ones(len(frontier)) if style == 0 else
                   1/neighbors**2 if style == 1 else neighbors**2)
        occupied.add(frontier[int(rng.choice(len(frontier), p=weights/weights.sum()))])
    ar, ac = _pick(rng, sorted(occupied))
    return tuple(sorted((r-ar, c-ac) for r, c in occupied))


def _library_options(templates, copies):
    options = []
    for width in range(5, 29):
        r, c, row_height = 2, 2, 0
        placements = []
        for key in copies:
            lr, lc, h, w = _bounds(templates[key])
            if w+4 > width:
                break
            if c+w > width-2:
                r += row_height+1
                c, row_height = 2, 0
            placements.append((key, r-lr, c-lc))
            c += w+1
            row_height = max(row_height, h)
        else:
            height = r+row_height+2
            if height <= 28:
                options.append((height, width, placements))
    return options


def _positions(rng, templates, shape, keys, directions, witness):
    for _ in range(100):
        triggers, used = [], set()
        for key, turn in zip(keys, directions):
            offsets = [_rotate(p, turn) for p in templates[key]]
            candidates = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                          if (r, c) not in used and all(0 <= r+dr < shape[0] and
                                                       0 <= c+dc < shape[1] for dr, dc in offsets)]
            if not candidates:
                break
            ar, ac = _pick(rng, candidates)
            triggers.append((ar, ac, key, turn))
            used.add((ar, ac))
        else:
            info = dict(templates=templates, shape=shape, triggers=triggers)
            hits, _ = _counts(info, strict=True)
            if np.count_nonzero(hits % 2) < 2 or not np.any(hits == 0):
                continue
            if witness and not np.any((hits > 0) & (hits % 2 == 0)):
                continue
            return triggers, hits
    return None


def _tile(rng, n):
    for _ in range(100):
        bits = rng.random((n, n)) < rng.uniform(.3, .7)
        count = int(bits.sum())
        if 3 <= count <= n*n-3 and all(not np.array_equal(bits, np.rot90(bits, k))
                                       for k in (1, 2, 3)):
            return bits
    bits = np.zeros((n, n), bool)
    bits[0, :2] = True
    bits[1, 0] = True
    return bits


def _make_pair(rng, split, index, preferred_axis):
    witness = split == 'train' and index < 2
    for attempt in range(500):
        n = (3 if index == 0 else 5) if witness else int(rng.choice((3, 5)))
        if witness:
            shape = ((int(rng.integers(4, 6)), int(rng.integers(4, 7))) if n == 3 else (3, 3))
        else:
            low, high = (2, 8) if n == 3 else (2, 5)
            shape = int(rng.integers(low, high)), int(rng.integers(low, high))
        requested = (index+1 if witness else 3 if split == 'test' else int(rng.integers(1, 3)))
        ntypes = requested + int(not witness and split == 'train' and requested < 3 and rng.random() < .2)
        palette = list(map(int, rng.permutation(10)))
        fill, line = palette[:2]
        colors = (palette[2], fill if rng.random() < .3 else palette[3])
        keys = palette[4:4+ntypes]
        ink = _pick(rng, [c for c in palette if c not in {fill, *keys}])
        frame = _pick(rng, [line, line, colors[0], colors[1], ink])
        if frame == fill:
            frame = line
        limit = min(3 if witness else 5, *shape)
        templates = {key: _shape(rng, limit, 6 if witness else 12) for key in keys}
        if len(set(templates.values())) != ntypes:
            continue
        if witness:
            requested_keys = [keys[0], keys[0]] if index == 0 else keys[:2]
            directions = [0, 1] if index == 0 else [2, 3]
        else:
            total = int(rng.integers(requested, min(7, shape[0]*shape[1])))
            requested_keys = keys[:requested] + [int(_pick(rng, keys[:requested]))
                                                for _ in range(total-requested)]
            rng.shuffle(requested_keys)
            directions = list(map(int, rng.integers(4, size=total)))
        placements = _positions(rng, templates, shape, requested_keys, directions, witness)
        if placements is None:
            continue
        triggers, hits = placements
        repeats = [int(rng.integers(2 if split == 'train' else 1, 4)) for _ in keys]
        copies = [key for key, count in zip(keys, repeats) for _ in range(count)]
        rng.shuffle(copies)
        canvas_shape = tuple(v*(n+1)+1 for v in shape)
        feasible = []
        for lib in _library_options(templates, copies):
            for axis in (0, 1):
                if canvas_shape[axis]+lib[axis]+1 <= 30:
                    feasible.append((axis, lib))
        if not feasible:
            continue
        favored = [choice for choice in feasible if choice[0] == preferred_axis]
        axis, (lh, lw, library_places) = _pick(rng, favored if favored and rng.random() < .7 else feasible)
        library = np.full((lh, lw), frame, int)
        library[1:-1, 1:-1] = fill
        for key, ar, ac in library_places:
            for dr, dc in templates[key]:
                library[ar+dr, ac+dc] = key if (dr, dc) == (0, 0) else ink
        bits = _tile(rng, n)
        tile = np.where(bits, colors[0], colors[1])
        phases = rng.integers(2, size=shape)
        odd = list(map(tuple, np.argwhere(hits % 2)))
        rng.shuffle(odd)
        phases[odd[0]], phases[odd[1]] = 0, 1
        canvas = np.full(canvas_shape, line, int)
        label = canvas.copy()
        # Forward labels use sampled masks and complete tiles, not the decoder.
        for r in range(shape[0]):
            for c in range(shape[1]):
                block = np.s_[r*(n+1)+1:r*(n+1)+n+1, c*(n+1)+1:c*(n+1)+n+1]
                canvas[block] = tile if phases[r, c] == 0 else sum(colors)-tile
                label[block] = tile if (phases[r, c]+hits[r, c]) % 2 == 0 else sum(colors)-tile
        center = n//2
        for ar, ac, key, direction in triggers:
            y, x = ar*(n+1)+1+center, ac*(n+1)+1+center
            dr, dc = _D4[direction]
            canvas[y, x] = canvas[y+dr, x+dc] = key
        pieces = (canvas, library) if rng.random() < .5 else (library, canvas)
        sizes = [piece.shape for piece in pieces]
        gap = int(rng.integers(1, min(3, 30-sizes[0][axis]-sizes[1][axis])+1))
        required = [max(s[0] for s in sizes), max(s[1] for s in sizes)]
        required[axis] = sizes[0][axis]+sizes[1][axis]+gap
        whole_shape = tuple(int(rng.integers(v, min(30, v+4)+1)) for v in required)
        origins = [[0, 0], [0, 0]]
        start = int(rng.integers(whole_shape[axis]-required[axis]+1))
        origins[0][axis] = start
        origins[1][axis] = start+sizes[0][axis]+gap
        for j in (0, 1):
            origins[j][1-axis] = int(rng.integers(whole_shape[1-axis]-sizes[j][1-axis]+1))
        grid = np.full(whole_shape, fill, int)
        for piece, origin in zip(pieces, origins):
            _crop(grid, (*origin, *piece.shape))[:] = piece
        try:
            info = _decode(grid)
            assert np.array_equal(_render(info), label)
            if witness:
                for behavior in ('union', 'no_exchange', 'anchor_only', 'omit_anchor',
                                 'no_rotation', 'rotate_microtile', 'paint_phase_0', 'paint_phase_1'):
                    assert not np.array_equal(_render(info, behavior=behavior), label)
                if index == 1:
                    assert not np.array_equal(_render(info, behavior='one_template'), label)
                observed_directions = (0, 1) if index == 0 else (2, 3)
                assert all(not np.array_equal(_render(info, mapping=m), label)
                           for m in _MAPS if any(m[d] != d for d in observed_directions))
            return {'input': grid, 'output': label}
        except AssertionError:
            continue
    raise RuntimeError("bounded construction could not find an informative fitting pair")


def _count_argument(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count_argument(num_train, 'num_train', 2, 10)
    b = _count_argument(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    axis = int(rng.integers(2))  # A sampling preference, never a solving convention.
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', a), ('test', b)):
        for i in range(count):
            for _ in range(10):
                pair = _make_pair(rng, split, i, axis)
                signature = _key(pair['input'])
                if signature not in seen:
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
            puzzle[split].append(pair)
            seen.add(signature)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inverse_hits(info):
    """Validator counts hits by inverse coordinate lookup, not forward stamping."""
    hits = np.zeros(info['shape'], int)
    for r in range(hits.shape[0]):
        for c in range(hits.shape[1]):
            for ar, ac, key, turn in info['triggers']:
                local = _rotate((r-ar, c-ac), -turn)
                hits[r, c] += local in info['templates'][key]
    return hits


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _key(inp)
            assert signature not in seen
            seen.add(signature)
            info = _decode(inp)
            assert out.shape == info['restored'].shape
            hits = _inverse_hits(info)
            assert np.array_equal(hits, _counts(info, strict=True)[0])
            n = info['n']
            expected = info['restored'].copy()
            for r in range(hits.shape[0]):
                for c in range(hits.shape[1]):
                    block = np.s_[r*(n+1)+1:r*(n+1)+n+1, c*(n+1)+1:c*(n+1)+n+1]
                    base = info['restored'][block]
                    expected[block] = base if hits[r, c] % 2 == 0 else sum(info['colors'])-base
            assert np.array_equal(out, expected), 'wrong tile restoration, parity, exchange, or crop'
            assert np.any(hits % 2) and np.any(hits == 0)
            assert np.any(out != info['restored'])
            assert not set(info['templates']) & set(np.unique(out)), 'key arrows must disappear'
            if split == 'train':
                train_info.append((info, out))
    # This is a finite competing-rule check, not independent semantic approval.
    survivors = [m for m in _MAPS if all(np.array_equal(_render(i, m), out) for i, out in train_info)]
    assert survivors == [(0, 1, 2, 3)], 'training must determine all four compass meanings'
    for behavior in ('union', 'no_exchange', 'anchor_only', 'omit_anchor', 'no_rotation',
                     'rotate_microtile', 'paint_phase_0', 'paint_phase_1', 'one_template'):
        assert any(not np.array_equal(_render(i, behavior=behavior), out)
                   for i, out in train_info), f'training fails to distinguish {behavior}'
    assert any(np.any(((_inverse_hits(i) > 0) & (_inverse_hits(i) % 2 == 0)))
               for i, _ in train_info), 'an even collision must be demonstrated'
    assert any(len({k for _, _, k, _ in i['triggers']}) >= 2 for i, _ in train_info)


def pair_features(pair):
    info = _decode(pair['input'])
    hits = _inverse_hits(info)
    r, c, h, w = info['box']
    y, x, v, u = info['library']['box']
    return dict(input_shape=np.asarray(pair['input']).shape, output_shape=(h, w),
                tile_size=info['n'], macro_shape=info['shape'],
                layout='stacked' if r+h <= y or y+v <= r else 'side_by_side',
                library_first=(y < r if r+h <= y or y+v <= r else x < c),
                key_count=len(info['templates']),
                requested_types=len({k for _, _, k, _ in info['triggers']}),
                trigger_count=len(info['triggers']), directions=tuple(t[3] for t in info['triggers']),
                repetitions=tuple(info['library']['repetitions'].values()),
                geometries=tuple(info['templates'].values()),
                motif_sizes=tuple(len(t) for t in info['templates'].values()),
                max_hits=int(hits.max()), odd_cells=int((hits % 2).sum()),
                even_collision_cells=int(((hits > 0) & (hits % 2 == 0)).sum()),
                untouched_cells=int((hits == 0).sum()),
                phases=tuple(map(int, np.unique(info['phases']))),
                fill_tile_reuse=info['library']['fill'] in info['colors'],
                palette_size=len(np.unique(pair['input'])))


def self_test():
    """Bounded 42-seed checks, count boundaries, and meaningful negative cases."""
    records, fingerprints = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                before = np.asarray(pair['input']).copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solver mutated its input'
                records.append(pair_features(pair))
    assert len(fingerprints) == 42
    assert {f['tile_size'] for f in records} == {3, 5}
    assert {f['requested_types'] for f in records} == {1, 2, 3}
    assert {f['layout'] for f in records} == {'stacked', 'side_by_side'}
    assert {f['library_first'] for f in records} == {False, True}
    assert {d for f in records for d in f['directions']} == {0, 1, 2, 3}
    assert len({f['input_shape'] for f in records}) >= 25
    assert len({f['output_shape'] for f in records}) >= 12
    assert len({t for f in records for t in f['geometries']}) >= 80
    assert len({f['trigger_count'] for f in records}) >= 4
    assert {n for f in records for n in f['repetitions']} == {1, 2, 3}
    assert any(f['max_hits'] >= 3 for f in records)
    assert any(f['max_hits'] >= 4 for f in records)
    assert any(f['fill_tile_reuse'] for f in records)
    assert all(f['untouched_cells'] > 0 and f['odd_cells'] >= 2 for f in records)
    first = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s])
               for k in p)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            puzzle = generate_puzzle(177+a*5+b, a, b)
            assert len(puzzle['train']) == a and len(puzzle['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The validator accepts lists as well as arrays, and must reject a bad label.
    listed = {s: [{k: v.tolist() for k, v in p.items()} for p in first[s]] for s in first}
    validate_generated_puzzle(listed)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (damaged['test'][0]['output'][0, 0]+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt border accepted')
    info = _decode(first['train'][0]['input'])
    # Exchange twice is exactly identity, including a restored trigger block.
    assert np.array_equal(sum(info['colors'])-(sum(info['colors'])-info['restored'][1:4, 1:4]),
                          info['restored'][1:4, 1:4])


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds plus count and invariant checks)')
