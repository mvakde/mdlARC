"""A color-keyed unary census of damaged object clans under two mirror folds.

Only Python's standard library and NumPy are used. The solver reads frame,
tick, palette, and key conventions from the supplied input.
"""
from functools import lru_cache
import numpy as np


_STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_GLYPHS = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)), _DIAMOND,
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = _STEPS8 if diagonal else ((1, 0), (-1, 0), (0, 1), (0, -1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((int(r), int(c)))
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(frozenset(comp))
    return result


@lru_cache(maxsize=256)
def _orbits(h, w, a, b):
    seen, result = set(), []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, a-r} if 0 <= a-r < h else {r}
            cols = {c, b-c} if 0 <= b-c < w else {c}
            orbit = tuple(sorted((rr, cc) for rr in rows for cc in cols))
            seen.update(orbit)
            result.append(orbit)
    return tuple(result)


def _signature(grid):
    grid = np.asarray(grid)
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(p['input']), _signature(p['output']))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _check_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    return grid


def _chart_direction(positions, shape):
    assert len(positions) >= 2
    if len(set(positions[:, 1])) == 1:
        assert np.all(np.diff(positions[:, 0]) >= 2)
        col = int(positions[0, 1])
        assert 2*col != shape[1]-1
        return (0, 1) if 2*col < shape[1]-1 else (0, -1)
    assert len(set(positions[:, 0])) == 1
    assert np.all(np.diff(positions[:, 1]) >= 2)
    row = int(positions[0, 0])
    assert 2*row != shape[0]-1
    return (1, 0) if 2*row < shape[0]-1 else (-1, 0)


def _parse(input_grid):
    grid = _check_grid(input_grid)
    bg = int(grid[0, 0])
    assert all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    # The only complete monochrome rectangular ring enclosing aligned keys
    # is the chart. Scene ticks interrupt its other ring.
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for comp in _components(grid == color, False):
            r0, r1 = min(r for r, c in comp), max(r for r, c in comp)+1
            c0, c1 = min(c for r, c in comp), max(c for r, c in comp)+1
            h, w = r1-r0, c1-c0
            if min(h, w) < 5 or len(comp) != 2*h+2*w-4:
                continue
            ring = grid[r0:r1, c0:c1]
            if not all(np.all(edge == color) for edge in (ring[0], ring[-1], ring[:, 0], ring[:, -1])):
                continue
            chart = ring[1:-1, 1:-1]
            positions = np.argwhere(chart != bg)
            keys = tuple(int(chart[r, c]) for r, c in positions)
            if not 2 <= len(keys) <= 6 or len(set(keys)) != len(keys) or color in keys:
                continue
            try:
                direction = _chart_direction(positions, chart.shape)
            except AssertionError:
                continue
            candidates.append((color, (r0, r1, c0, c1), chart.copy(), positions, keys, direction))
    assert len(candidates) == 1, 'Chart must have one visible interpretation'
    frame, cb, chart, positions, keys, direction = candidates[0]
    outside = np.ones(grid.shape, bool)
    outside[cb[0]:cb[1], cb[2]:cb[3]] = False
    coords = np.argwhere((grid == frame) & outside)
    assert len(coords)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0)+1)
    assert min(r1-r0, c1-c0) >= 7
    assert cb[1] <= r0 or r1 <= cb[0] or cb[3] <= c0 or c1 <= cb[2]
    ring = grid[r0:r1, c0:c1]
    assert all(int(ring[p]) == frame for p in ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    tops = np.flatnonzero(ring[0] != frame)
    bottoms = np.flatnonzero(ring[-1] != frame)
    lefts = np.flatnonzero(ring[:, 0] != frame)
    rights = np.flatnonzero(ring[:, -1] != frame)
    assert np.array_equal(tops, bottoms) and np.array_equal(lefts, rights)
    for ticks, length in ((tops, ring.shape[1]), (lefts, ring.shape[0])):
        assert len(ticks) in (1, 2) and 0 < ticks[0] <= ticks[-1] < length-1
        assert len(ticks) == 1 or ticks[1] == ticks[0]+1
    markers = {int(ring[0, c]) for c in tops} | {int(ring[r, 0]) for r in lefts}
    markers |= {int(ring[-1, c]) for c in bottoms} | {int(ring[r, -1]) for r in rights}
    assert len(markers) == 1
    erase = markers.pop()
    assert erase not in (bg, frame) and erase not in keys
    scene = ring[1:-1, 1:-1].copy()
    assert not np.any(scene == frame)
    a = int(2*(lefts[0]-1) if len(lefts) == 1 else lefts.sum()-2)
    b = int(2*(tops[0]-1) if len(tops) == 1 else tops.sum()-2)
    restored = scene.copy()
    for orbit in _orbits(*scene.shape, a, b):
        values = {int(scene[p]) for p in orbit if scene[p] != erase}
        assert len(values) == 1, 'Every orbit needs agreeing surviving evidence'
        value = values.pop()
        for p in orbit:
            restored[p] = value
    occupied = np.zeros(grid.shape, bool)
    occupied[cb[0]:cb[1], cb[2]:cb[3]] = True
    occupied[r0:r1, c0:c1] = True
    assert np.all(grid[~occupied] == bg)
    return dict(background=bg, frame=frame, erase=erase, chart_bounds=cb,
                scene_bounds=(r0, r1, c0, c1), chart=chart, positions=positions,
                keys=keys, direction=direction, scene=scene, restored=restored,
                axes=(a, b))


def _clans(restored, bg, axes, diagonal=True):
    h, w = restored.shape
    a, b = axes
    result = []
    for color in sorted(set(map(int, np.unique(restored))) - {bg}):
        comps = _components(restored == color, diagonal)
        lookup = {comp: i for i, comp in enumerate(comps)}
        visited = set()
        for i, comp in enumerate(comps):
            if i in visited:
                continue
            group, pending = set(), [i]
            while pending:
                index = pending.pop()
                if index in group:
                    continue
                group.add(index)
                current = comps[index]
                for row_fold in (True, False):
                    mirrored = frozenset((a-r, c) if row_fold else (r, b-c) for r, c in current)
                    in_bounds = [0 <= r < h and 0 <= c < w for r, c in mirrored]
                    if not any(in_bounds):
                        continue
                    assert all(in_bounds), 'A component may not be clipped by a fold overlap'
                    assert mirrored in lookup, 'Reflections must map whole objects to whole objects'
                    pending.append(lookup[mirrored])
            visited.update(group)
            cells = frozenset().union(*(comps[j] for j in group))
            result.append(dict(color=color, cells=cells,
                               members=tuple(comps[j] for j in sorted(group))))
    return result


def _tally(meta, counts):
    output = np.full(meta['chart'].shape, meta['background'], np.int8)
    dr, dc = meta['direction']
    for (r, c), color in zip(meta['positions'], meta['keys']):
        n = counts.get(color, 0)
        for i in range(n):
            rr, cc = int(r)+2*i*dr, int(c)+2*i*dc
            assert 0 < rr < output.shape[0]-1 and 0 < cc < output.shape[1]-1, 'Tally exceeds panel capacity'
            output[rr, cc] = color
    return output


def _analyze(grid):
    meta = _parse(grid)
    clans = _clans(meta['restored'], meta['background'], meta['axes'])
    counts = {color: 0 for color in meta['keys']}
    for clan in clans:
        clan['damaged'] = any(meta['scene'][p] == meta['erase'] for p in clan['cells'])
        if clan['damaged'] and clan['color'] in counts:
            counts[clan['color']] += 1
    meta.update(clans=clans, counts=counts)
    return meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Restore topology, quotient components by reflections, tally damaged clans."""
    meta = _analyze(input_grid)
    return _tally(meta, meta['counts'])


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _shape(rng, small=False):
    if small or rng.random() < .65:
        shape = _GLYPHS[int(rng.integers(3 if small else len(_GLYPHS)))]
    else:
        cells = {(1, 1)}
        for _ in range(int(rng.integers(2, 9))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _STEPS8
                               if 0 <= r+dr < 3 and 0 <= c+dc < 4} - cells)
            if frontier:
                cells.add(frontier[int(rng.integers(len(frontier)))])
        shape = tuple(cells)
    if rng.random() < .5:
        shape = tuple((r, -c) for r, c in shape)
    for _ in range(int(rng.integers(4))):
        shape = tuple((-c, r) for r, c in shape)
    return _normalize(shape)


def _reflect_cells(cells, shape, axes):
    h, w = shape
    a, b = axes
    return frozenset((rr, cc) for r, c in cells for rr in (r, a-r) for cc in (c, b-c)
                     if 0 <= rr < h and 0 <= cc < w)


def _halo(cells, h, w):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if 0 <= r+dr < h and 0 <= c+dc < w}


def _place_clan(rng, clean, blocked, axes, kind, forced_shape=None, small=False, deterministic=False):
    h, w = clean.shape
    a, b = axes
    rlo, clo = max(0, a-h+1), max(0, b-w+1)
    midr, midc = a//2, b//2
    if kind == 'center':
        # A reflected 2x2 fragment joins across both folds: one object.
        shape = ((0, 0), (0, 1), (1, 0), (1, 1))
        if rng.random() < .5:
            shape = ((0, 1), (1, 0), (1, 1))
        anchors = [(midr-1, midc-1)]
    elif kind == 'axis':
        # A bar across the vertical fold, away from the horizontal fold.
        # Its center can be lost in one copy and recovered from the other.
        shape = ((0, 0), (0, 1), (0, 2))
        anchors = [(r, midc-2) for r in range(rlo+1, midr-2)]
        if not deterministic:
            rng.shuffle(anchors)
    else:
        shape = forced_shape if forced_shape is not None else _shape(rng, small)
        sh, sw = 1+max(r for r, c in shape), 1+max(c for r, c in shape)
        anchors = [(r, c) for r in range(rlo+1, midr-sh)
                   for c in range(clo+1, midc-sw)]
        if not deterministic:
            rng.shuffle(anchors)
    for r0, c0 in anchors:
        cells = _reflect_cells(((r0+r, c0+c) for r, c in shape), clean.shape, axes)
        if cells and not any(blocked[p] for p in cells):
            mask = np.zeros(clean.shape, bool)
            for p in cells:
                mask[p] = True
            members = _components(mask)
            expected = {'center': 1, 'axis': 2, 'free': 4}[kind]
            if len(members) != expected:
                continue
            for p in _halo(cells, h, w):
                blocked[p] = True
            return cells, members
    return None


def _ticks(axis):
    return (axis//2,) if axis % 2 == 0 else (axis//2, axis//2+1)


def _make_pair(rng, keys, distractor, bg, frame, erase, witness, test, turn, reflect,
               force_guard=False):
    k = len(keys)
    # The first rich demonstration includes three different stabilizers and
    # an intact same-color clan. Other pairs recombine those mechanisms.
    if witness:
        plans = [(keys[0], 'center', True, None), (keys[0], 'axis', True, None),
                 (keys[0], 'free', True, _DIAMOND), (keys[0], 'free', False, None)]
        plans += [(color, 'free', True, None) for color in keys[1:]]
    else:
        test_glyph = _GLYPHS[int(rng.integers(4, 8))] if test else None
        plans = [(keys[0], 'axis', True, None), (keys[0], 'free', True, test_glyph)]
        plans += [(color, 'free', True, None) for color in keys[1:]]
        if rng.random() < .65:
            plans.insert(0, (keys[int(rng.integers(k))], 'center', bool(rng.integers(2)), None))
        # Some intact same-color copies disprove counting all clans.
        plans.append((keys[int(rng.integers(k))], 'free', False, None))
        if k <= 3 and rng.random() < .3:
            plans.append((keys[0], 'free', True, None))
    if distractor is not None:
        plans.append((distractor, 'free', True, None))
    counts = {color: sum(c == color and damaged for c, kind, damaged, forced in plans)
              for color in keys}
    top, bottom = int(rng.integers(1, 3)), int(rng.integers(1, 3))
    gaps = [2]* (k-1)
    if k == 2 and rng.random() < .5:
        gaps[0] = 3
    rows = np.cumsum([top] + gaps).tolist()
    ch = rows[-1]+bottom+1
    cw = 2*max(counts.values())+int(rng.integers(2, 5))
    cw = max(cw, 7)
    chart = np.full((ch, cw), bg, np.int8)
    for row, color in zip(rows, keys):
        chart[row, 1] = color
    output = np.full_like(chart, bg)
    for row, color in zip(rows, keys):
        for i in range(counts[color]):
            output[row, 1+2*i] = color
    # In canonical coordinates the two rings are vertically arranged.
    # Random horizontal displacement, padding, reflection, and rotation
    # prevent an absolute-location or fixed-orientation convention.
    max_h = 23-ch
    assert max_h >= 11
    for attempt in ([47] if force_guard else range(48)):
        h = int(rng.integers(11, max_h+1))
        w = int(rng.integers(17, 27)) if attempt < 24 else 26
        if attempt >= 24:
            h = max_h
        offsets = (int(rng.choice((-1, 0, 0, 0, 1))), int(rng.choice((-2, -1, 0, 0, 0, 1, 2))))
        if attempt >= 24:
            offsets = (0, 0)
        axes = (h-1+offsets[0], w-1+offsets[1])
        clean = np.full((h, w), bg, np.int8)
        blocked = np.zeros((h, w), bool)
        objects = []
        mosaic_assigned = False
        okay = True
        for index, (color, kind, damaged, forced) in enumerate(plans):
            small = attempt >= 16 and forced is None and kind == 'free'
            if attempt >= 24 and forced is None and kind == 'free':
                forced = ((0, 0),)
            placed = _place_clan(rng, clean, blocked, axes, kind, forced, small,
                                 deterministic=attempt == 47)
            if placed is None:
                okay = False
                break
            cells, members = placed
            for p in cells:
                clean[p] = color
            mosaic = test and not mosaic_assigned and kind == 'free' and damaged and color == keys[0]
            mosaic_assigned |= mosaic
            objects.append(dict(color=color, kind=kind, damaged=damaged, mosaic=mosaic,
                                cells=cells, members=members))
        if not okay:
            continue
        mask = np.zeros((h, w), bool)
        protected = set()
        a, b = axes
        for obj in objects:
            if not obj['damaged']:
                continue
            cells = obj['cells']
            if obj['kind'] == 'free':
                members = sorted(obj['members'], key=lambda x: min(x))
                # First demonstration and tests have three fully erased
                # copies. Other training pairs also show milder damage.
                number = 3 if witness or test else int(rng.integers(1, 4))
                survivor = int(rng.integers(4))
                others = [j for j in range(4) if j != survivor]
                rng.shuffle(others)
                if obj['mosaic']:
                    # Balanced complementary fragments: all four copies are
                    # incomplete, but each pixel orbit keeps one witness.
                    keeps = list(map(int, rng.permutation(4)))
                    orbits = [o for o in _orbits(h, w, a, b) if o[0] in cells]
                    for i, orbit in enumerate(orbits):
                        keep = keeps[i % 4]
                        for j, p in enumerate(orbit):
                            if j != keep:
                                mask[p] = True
                else:
                    for j in others[:number]:
                        for p in members[j]:
                            mask[p] = True
            elif obj['kind'] == 'axis':
                member = min(obj['members'], key=lambda x: min(x))
                # Remove the center column(s) of one bar. Surviving pieces
                # are separated until its reflected bar repairs the bridge.
                cols = _ticks(b)
                for p in member:
                    if p[1] in cols:
                        mask[p] = True
            else:
                choices = [o for o in _orbits(h, w, a, b)
                           if o[0] in cells and len(o) > 1]
                orbit = choices[int(rng.integers(len(choices)))]
                keep = int(rng.integers(len(orbit)))
                for j, p in enumerate(orbit):
                    if j != keep:
                        mask[p] = True
            protected.update(p for p in cells if mask[p])
        # Background masks deliberately coexist with object damage. They
        # never count, even when the erased patch is a connected component.
        background_mask = np.zeros_like(mask)
        mode = int(rng.integers(3))
        if mode == 0:
            background_mask = rng.random((h, w)) < rng.uniform(.02, .08)
        else:
            for _ in range(int(rng.integers(1, 4))):
                rh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 6))
                rr, cc = int(rng.integers(h-rh+1)), int(rng.integers(w-sw+1))
                for dr in range(rh):
                    width = sw if mode == 1 else 1+dr*(sw-1)//max(1, rh-1)
                    background_mask[rr+dr, cc:cc+width] = True
        mask |= background_mask & (clean == bg)
        for orbit in _orbits(h, w, a, b):
            if all(mask[p] for p in orbit):
                choices = [p for p in orbit if p not in protected]
                assert choices
                mask[choices[int(rng.integers(len(choices)))]] = False
        # Ensure at least one ignored background erasure remains.
        if not np.any(mask & (clean == bg)):
            candidates = [o for o in _orbits(h, w, a, b) if len(o) > 1 and clean[o[0]] == bg]
            mask[candidates[int(rng.integers(len(candidates)))][0]] = True
        scene = clean.copy()
        scene[mask] = erase
        scene_ring = np.full((h+2, w+2), frame, np.int8)
        scene_ring[1:-1, 1:-1] = scene
        for r in _ticks(a):
            scene_ring[r+1, 0] = scene_ring[r+1, -1] = erase
        for c in _ticks(b):
            scene_ring[0, c+1] = scene_ring[-1, c+1] = erase
        chart_ring = np.full((ch+2, cw+2), frame, np.int8)
        chart_ring[1:-1, 1:-1] = chart
        height = h+ch+7
        width = max(w+2, cw+2)+int(rng.integers(2, 5))
        width = min(30, width)
        grid = np.full((height, width), bg, np.int8)
        sx = int(rng.integers(1, width-(w+2)))
        cx = int(rng.integers(1, width-(cw+2)))
        grid[1:h+3, sx:sx+w+2] = scene_ring
        grid[h+4:h+ch+6, cx:cx+cw+2] = chart_ring
        if reflect:
            grid, output2 = np.fliplr(grid), np.fliplr(output)
        else:
            output2 = output
        grid = np.rot90(grid, turn).copy()
        output2 = np.rot90(output2, turn).copy()
        # Labels above use the explicit clan plan, not solve's recount.
        meta = _analyze(grid)
        assert meta['counts'] == counts
        assert np.array_equal(_tally(meta, counts), output2)
        return {'input': grid, 'output': output2}
    raise RuntimeError('Bounded clan placement failed')


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= value <= high:
        raise ValueError(name+' outside legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count('num_train', num_train, 2, 10)
    num_test = _count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    role_color = int(rng.integers(10))
    first_k = int(rng.integers(2, 4))
    base_turn = int(rng.integers(4))
    orientation_transfer = bool(rng.random() < .35)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            witness = split == 'train' and index == 0
            role_witness = split == 'train' and index == 1
            test = split == 'test'
            k = first_k if witness else (5-first_k if role_witness else int(rng.integers(2, 5 if test else 4)))
            structural = [c for c in range(10) if c != role_color or not (witness or role_witness)]
            bg, frame, erase = map(int, rng.choice(structural, 3, replace=False))
            pool = [c for c in range(10) if c not in (bg, frame, erase)]
            rng.shuffle(pool)
            if witness:
                pool.remove(role_color)
                pool.insert(0, role_color)
            elif role_witness:
                pool.remove(role_color)
            keys = pool[:k]
            distractor = role_color if role_witness else (pool[k] if rng.random() < .7 else None)
            turn = ((base_turn+int(test)) % 4 if orientation_transfer else int(rng.integers(4)))
            for retry in range(8):
                pair = _make_pair(rng, keys, distractor, bg, frame, erase, witness, test,
                                  turn, bool(rng.integers(2)))
                sig = _signature(pair['input'])
                if sig not in seen:
                    break
            else:
                raise RuntimeError('Distinct input construction failed')
            seen.add(sig)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            meta = _analyze(inp)
            assert np.array_equal(out, _tally(meta, meta['counts']))
            assert np.all(meta['restored'] != meta['erase'])
            assert all(meta['counts'][c] >= 1 for c in meta['keys'])
            owner = {}
            for i, clan in enumerate(meta['clans']):
                assert len(clan['members']) in (1, 2, 4)
                for p in clan['cells']:
                    owner[p] = i
            for p, i in owner.items():
                for dr, dc in _STEPS8:
                    q = p[0]+dr, p[1]+dc
                    assert q not in owner or owner[q] == i, 'Different clans must not touch'
            assert np.any((meta['scene'] == meta['erase']) & (meta['restored'] == meta['background']))
            if split == 'test':
                assert any(c['color'] in meta['keys'] and len(c['members']) == 4
                           and all(0 < sum(meta['scene'][p] == meta['erase'] for p in member) < len(member)
                                   for member in c['members']) for c in meta['clans']), 'Test must contain complementary fragments'
            if split == 'train':
                training.append(meta)
    assert any(max(m['counts'].values()) >= 3 for m in training)
    assert any({len(c['members']) for c in m['clans'] if c['damaged'] and c['color'] in m['keys']} == {1, 2, 4}
               for m in training)
    assert any(any(c['color'] in m['keys'] and not c['damaged'] for c in m['clans']) for m in training)
    assert any(set(a['keys']) & {c['color'] for c in b['clans'] if c['color'] not in b['keys'] and c['damaged']}
               for a in training for b in training)
    assert any(any(len(_components(m['restored'] == color, False)) > len(_components(m['restored'] == color))
                   for color in m['keys']) for m in training), 'Training must expose diagonal connectivity'
    assert any(_double_only(m) for m in training), 'Training must require the double reflection'


def _double_only(meta):
    h, w = meta['scene'].shape
    a, b = meta['axes']
    scene, restored, erase = meta['scene'], meta['restored'], meta['erase']
    return any(0 <= a-r < h and 0 <= b-c < w and restored[r, c] != meta['background']
               and scene[a-r, c] == erase and scene[r, b-c] == erase
               and scene[a-r, b-c] != erase for r, c in np.argwhere(scene == erase))


def _shortcut_counts(meta, kind):
    if kind == 'physical':
        return {c: sum(len(x['members']) for x in meta['clans'] if x['color'] == c and x['damaged']) for c in meta['keys']}
    if kind == 'all_clans':
        return {c: sum(x['color'] == c for x in meta['clans']) for c in meta['keys']}
    if kind == 'no_repair':
        return {c: len(_components(meta['scene'] == c)) for c in meta['keys']}
    if kind == 'four_connectivity':
        clans = _clans(meta['restored'], meta['background'], meta['axes'], False)
        return {c: sum(x['color'] == c and any(meta['scene'][p] == meta['erase'] for p in x['cells'])
                       for x in clans) for c in meta['keys']}
    raise ValueError(kind)


def self_test() -> None:
    # Exercise the ordered guard directly, including its densest test layout.
    for seed, keys, witness, test in ((101, [1, 2], True, False),
                                      (102, [1, 2, 3], True, False),
                                      (103, [1, 2, 3, 4], False, True)):
        pair = _make_pair(np.random.default_rng(seed), keys, 5, 0, 8, 9,
                          witness, test, seed % 4, True, force_guard=True)
        assert np.array_equal(solve(pair['input']), pair['output'])
    puzzles, metas = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        training = [_analyze(p['input']) for p in puzzle['train']]
        for kind in ('physical', 'all_clans', 'no_repair', 'four_connectivity'):
            assert any(_shortcut_counts(m, kind) != m['counts'] for m in training), kind
        for pair in puzzle['train']+puzzle['test']:
            assert np.array_equal(solve(pair['input'].copy()), pair['output'])
            m = _analyze(pair['input'])
            m['double_only'] = _double_only(m)
            metas.append(m)
        assert any(m['double_only'] for m in metas[-5:-1])
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    assert _puzzle_signature(generate_puzzle(0)) == _puzzle_signature(puzzles[0])
    assert len({_signature(m['scene']) for m in metas}) == len(metas)
    assert len({m['scene'].shape for m in metas}) >= 30
    assert {m['background'] for m in metas} == set(range(10))
    assert {m['frame'] for m in metas} == set(range(10))
    assert {m['erase'] for m in metas} == set(range(10))
    assert {m['direction'] for m in metas} == {(0, 1), (0, -1), (1, 0), (-1, 0)}
    assert {tuple(v % 2 for v in m['axes']) for m in metas} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len({(m['axes'][0]-m['scene'].shape[0]+1, m['axes'][1]-m['scene'].shape[1]+1) for m in metas}) >= 10
    shapes = {_normalize(member) for m in metas for c in m['clans'] for member in c['members']}
    assert len(shapes) >= 30 and _DIAMOND in shapes
    assert {len(m['keys']) for m in metas} >= {2, 3, 4}
    # Causal checks on the visible input: revealing background holes cannot
    # affect a census; fully repairing exactly one keyed clan removes one
    # tally cell. These query masks need not follow generation's schedule.
    rich = next(p for p in puzzles[0]['train'] if max(_analyze(p['input'])['counts'].values()) >= 3)
    m = _analyze(rich['input'])
    r0, r1, c0, c1 = m['scene_bounds']
    revealed = rich['input'].copy()
    interior = revealed[r0+1:r1-1, c0+1:c1-1]
    bg_gaps = (m['scene'] == m['erase']) & (m['restored'] == m['background'])
    interior[bg_gaps] = m['background']
    assert np.array_equal(solve(revealed), rich['output'])
    color = max(m['counts'], key=m['counts'].get)
    clan = next(x for x in m['clans'] if x['color'] == color and x['damaged'])
    for p in clan['cells']:
        interior[p] = m['restored'][p]
    reduced = dict(m['counts'])
    reduced[color] -= 1
    assert np.array_equal(solve(revealed), _tally(m, reduced))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(991+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt tally accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal explicit counts, semantic counterexamples')
