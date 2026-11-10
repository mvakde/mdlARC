"""Echo rim: period-six wave travel through a marker-selected compartment.

All transformation conventions are visible in one input. No parent code is used.
"""
from collections import deque
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be an integer 2D ARC grid")
    return a


def _neighbors(p, shape, offsets=N4):
    r, c = p
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            yield rr, cc


def _distance(allowed, sources):
    """Multi-source orthogonal BFS; -1 denotes unreachable cells."""
    d = np.full(allowed.shape, -1, dtype=int)
    q = deque()
    for r, c in sources:
        if allowed[r, c] and d[r, c] < 0:
            d[r, c] = 0
            q.append((int(r), int(c)))
    while q:
        p = q.popleft()
        for x in _neighbors(p, allowed.shape):
            if allowed[x] and d[x] < 0:
                d[x] = d[p] + 1
                q.append(x)
    return d


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        cells, stack = [], [start]
        seen[start] = True
        while stack:
            p = stack.pop()
            cells.append(p)
            for x in _neighbors(p, mask.shape):
                if mask[x] and not seen[x]:
                    seen[x] = True
                    stack.append(x)
        result.append(cells)
    return result


def _touch(mask, offsets=N8, outside=False):
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=outside)
    result = np.zeros_like(mask)
    for dr, dc in offsets:
        result |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _roles(g):
    values, counts = np.unique(g, return_counts=True)
    if len(values) != 4 or counts.max() * 2 <= g.size:
        raise ValueError("four roles and a majority background are required")
    bg = int(values[np.argmax(counts)])
    single = values[counts == 1]
    if len(single) != 1:
        raise ValueError("accent must be the unique singleton color")
    accent = int(single[0])
    tip = tuple(np.argwhere(g == accent)[0])
    candidates = []
    for color in values:
        color = int(color)
        if color in (bg, accent):
            continue
        points = np.argwhere((g == color) | (g == accent))
        rr, cc = points.T
        for side, fixed, boundary, lateral, tip_lateral, span in (
                (0, rr, 0, cc, tip[1], g.shape[1]),
                (1, cc, 0, rr, tip[0], g.shape[0]),
                (2, rr, g.shape[0] - 1, cc, tip[1], g.shape[1]),
                (3, cc, g.shape[1] - 1, rr, tip[0], g.shape[0])):
            lo, hi = int(lateral.min()), int(lateral.max())
            if (len(points) >= 3 and np.all(fixed == boundary)
                    and 0 < lo <= hi < span - 1
                    and np.array_equal(np.sort(lateral), np.arange(lo, hi + 1))
                    and tip_lateral in (lo, hi)):
                candidates.append((color, side, lo, hi))
    if len(candidates) != 1:
        raise ValueError("entrance must be a unique two-color edge bar")
    primary, side, lo, hi = candidates[0]
    wall = int(next(c for c in values if c not in (bg, primary, accent)))
    return bg, wall, primary, accent, side, lo, hi


def _analyze(input_grid):
    g = _grid(input_grid)
    bg, wall, primary, accent, side, lo, hi = _roles(g)
    entry = (g == primary) | (g == accent)
    d = _distance(g != wall, np.argwhere(entry))
    selected = d >= 0
    # Outside may traverse all nonselected cells, including structural cells.
    complement = np.pad(~selected, 1, constant_values=True)
    outside_padded = _distance(complement, [(0, 0)]) >= 0
    exterior = outside_padded[1:-1, 1:-1]
    rim = selected & _touch(exterior, outside=True)
    orth_rim = selected & _touch(exterior, N4, outside=True)
    walls = g == wall
    components = _components(walls)
    kept = np.zeros_like(walls)
    islands = np.zeros_like(walls)
    for cells in components:
        keep = any(selected[x] for p in cells for x in _neighbors(p, g.shape))
        if keep:
            for p in cells:
                kept[p] = True
        if not any(exterior[p] for p in cells):
            for p in cells:
                islands[p] = True
    contact = np.full(g.shape, -1, int)
    for p in map(tuple, np.argwhere(kept)):
        arrivals = [int(d[x]) + 1 for x in _neighbors(p, g.shape) if selected[x]]
        if arrivals:
            contact[p] = min(arrivals)
    return dict(grid=g, bg=bg, wall=wall, primary=primary, accent=accent,
                side=side, lo=lo, hi=hi, entry=entry, d=d, selected=selected,
                exterior=exterior, rim=rim, orth_rim=orth_rim, walls=walls,
                kept=kept, islands=islands, contact=contact)


def _paint(info, *, distance=None, rim=None, erase=True, contacts=True,
           contact_offset=0, solid=False):
    g, bg = info['grid'], info['bg']
    d = info['d'] if distance is None else distance
    rim = info['rim'] if rim is None else rim
    out = g.copy()
    if erase:
        out[info['walls'] & ~info['kept']] = bg
    work = rim & ~info['entry']
    out[work] = bg
    if solid:
        out[work] = info['accent']
    else:
        out[work & np.isin(d % 6, (0, 2))] = info['primary']
        out[work & (d % 6 == 4)] = info['accent']
    if contacts:
        t = info['contact']
        out[(t >= 0) & ((t + contact_offset) % 6 == 4)] = info['accent']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from the input; never mutate the input."""
    return _paint(_analyze(input_grid))


def _axis_distance(info):
    h, w = info['grid'].shape
    r, c = np.indices((h, w))
    return (r, c, h - 1 - r, w - 1 - c)[info['side']]


def _features(input_grid):
    z = _analyze(input_grid)
    out = _paint(z)
    d, rim, entry = z['d'], z['rim'], z['entry']
    axis = _axis_distance(z)
    hole_rim = z['selected'] & _touch(~z['selected'], outside=True)
    regions = _components(~z['walls'])
    sizes = [len(x) for x in regions]
    area = int(z['selected'].sum())
    no_b = z.copy()
    # Ablate B: straight, unobstructed distances and canvas rim; no filtering.
    sources = np.argwhere(entry)
    unblocked = _distance(np.ones(entry.shape, bool), sources)
    canvas_rim = np.zeros_like(entry)
    canvas_rim[[0, -1], :] = True
    canvas_rim[:, [0, -1]] = True
    no_b['contact'] = np.where(z['walls'], unblocked, -1)
    b_removed = _paint(no_b, distance=unblocked, rim=canvas_rim, erase=False)
    ordinary_hole = (hole_rim & ~rim & ~entry & (d % 2 == 0))
    f = dict(shape=tuple(entry.shape), side=z['side'], length=int(entry.sum()),
             compartments=len(sizes), selected_small=area < max(sizes),
             selected_large=area > min(sizes), max_time=int(d.max()),
             islands=len(_components(z['islands'])),
             erased=len(_components(z['walls'] & ~z['kept'])),
             contact_marks=int(np.sum((z['contact'] >= 0) & (z['contact'] % 6 == 4))),
             detour=bool(np.any(rim & (d > unblocked))),
             long_cycle=bool(np.any(rim & (d >= 12) & (d % 6 == 4))),
             no_a_diff=not np.array_equal(out, _paint(z, solid=True, contacts=False)),
             no_b_diff=not np.array_equal(out, b_removed),
             axis_diff=not np.array_equal(out, _paint(z, distance=axis)),
             hole_diff=bool(ordinary_hole.any()) and not np.array_equal(out, _paint(z, rim=hole_rim)),
             diagonal_diff=not np.array_equal(out, _paint(z, rim=z['orth_rim'])),
             erasure_diff=not np.array_equal(out, _paint(z, erase=False)),
             contact_diff=not np.array_equal(out, _paint(z, contacts=False)),
             contact_offset_diff=not np.array_equal(out, _paint(z, contact_offset=-1)),
             multisource_diff=False)
    only_tip = _distance(~z['walls'], np.argwhere(z['grid'] == z['accent']))
    f['multisource_diff'] = not np.array_equal(out, _paint(z, distance=only_tip))
    return f


TEACHING = ('no_a_diff', 'no_b_diff', 'axis_diff', 'hole_diff', 'diagonal_diff',
            'erasure_diff', 'contact_diff', 'contact_offset_diff',
            'multisource_diff', 'detour', 'long_cycle', 'selected_small', 'selected_large')


def _covered(features):
    return all(any(f[k] for f in features) for k in TEACHING)


def _shape(rng, hmax=3, wmax=4):
    h = int(rng.integers(1, hmax + 1))
    w = int(rng.integers(1, wmax + 1))
    a = np.ones((h, w), bool)
    kind = int(rng.integers(4))
    if kind == 1:
        a[1:, 1:] = False
    elif kind == 2:
        for r in range(h):
            a[r, int(rng.integers(1, w + 1)):] = False
    elif kind == 3:
        a = np.ones((1, w), bool)
    return np.rot90(a, int(rng.integers(4))).copy()


def _place_island(rng, walls, allowed):
    """Place varied solid motifs with a full free-cell halo."""
    h, w = walls.shape
    for _ in range(30):
        shape = _shape(rng)
        sh, sw = shape.shape
        if sh + 4 > h or sw + 4 > w:
            continue
        r = int(rng.integers(2, h - sh - 1))
        c = int(rng.integers(2, w - sw - 1))
        cells = np.zeros_like(walls)
        cells[r:r + sh, c:c + sw] = shape
        halo = _touch(cells) | cells
        if np.all(allowed[halo]) and not np.any(walls[halo]):
            walls[cells] = True
            return True
    return False


def _checkerboard(walls):
    a, b, c, d = walls[:-1, :-1], walls[:-1, 1:], walls[1:, :-1], walls[1:, 1:]
    return bool(np.any((a == d) & (b == c) & (a != b)))


def _scene(rng, selection, complex_test, rotation, reflect, palette, witness):
    if witness:
        h = int(rng.integers(19, 31))
        if selection == 'small':
            left = int(rng.integers(8, 13))
            right = int(rng.integers(left + 2, 30 - left))
        else:
            left = int(rng.integers(15, 22))
            right = int(rng.integers(4, min(9, 30 - left)))
        thick = 1
        w = left + thick + right
    else:
        h = int(rng.integers(18 if complex_test else 11, 31))
        w = int(rng.integers(18 if complex_test else 15, 31))
        thick = int(rng.integers(1, 3))
        left = int(rng.integers(7, w - thick - 3))
        right = w - left - thick
    walls = np.zeros((h, w), bool)
    walls[:, left:left + thick] = True
    # Extra divisions are genuine topology, not a separate panel.
    if right >= 9 and rng.random() < .45:
        divider = left + thick + int(rng.integers(3, right - 2))
        walls[:, divider] = True
    baffles = int(rng.integers(2, 4)) if complex_test else (1 if witness == 1 else int(rng.integers(2)))
    rows = sorted(rng.choice(np.arange(4, h - 4, 3), baffles, replace=False).tolist())
    first_left = bool(rng.integers(2))
    for i, r in enumerate(rows):
        depth = int(rng.integers(1, 3))
        if first_left == (i % 2 == 0):
            reach = int(rng.integers(max(3, left // 2), left - 1))
            walls[r:r + depth, :reach] = True
        else:
            gate = int(rng.integers(2, max(3, left // 2)))
            walls[r:r + depth, gate:left] = True
    # Random attached lobes alter the external rim without changing compartments.
    for _ in range(int(rng.integers(0, 4))):
        r = int(rng.integers(2, h - 3))
        depth = int(rng.integers(1, min(4, left - 3)))
        trial = walls.copy()
        trial[r:r + int(rng.integers(1, 4)), left - depth:left] = True
        if len(_components(~trial)) == len(_components(~walls)) and not _checkerboard(trial):
            walls = trial
    # Choose the entrance after sampling room geometry.
    length = int(rng.integers(3, min(left - 1, 9) + 1))
    start = int(rng.integers(1, left - length + 1))
    entry_points = [(0, c) for c in range(start, start + length)]
    free_distance = _distance(~walls, entry_points)
    target = free_distance >= 0
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        _place_island(rng, walls, target)
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        _place_island(rng, walls, ~target & ~walls)
    # A free-standing far-side edge block can also be removable.
    if right >= 5 and rng.random() < .45:
        r = int(rng.integers(2, h - 4))
        walls[r:r + int(rng.integers(1, 4)), -1] = True
    if _checkerboard(walls):
        return None
    bg, wall, primary, accent = palette
    g = np.full((h, w), bg, dtype=np.int8)
    g[walls] = wall
    g[0, start:start + length] = primary
    g[0, start if rng.integers(2) else start + length - 1] = accent
    g = np.rot90(g, rotation)
    if reflect:
        g = np.fliplr(g)
    g = g.copy()
    try:
        z = _analyze(g)
    except ValueError:
        return None
    # Keep initial expansion free of diagonal-only passages and nested rooms.
    for cells in _components(~z['walls']):
        if not any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in cells):
            return None
    return g


def _fallback(rng, witness, rotation, palette, index):
    """Finite, informative fallback; dimensions and entrance remain variable."""
    h, w = 24 + index % 7, 28
    left = 10 if witness == 1 else 19
    bg, wall, primary, accent = palette
    g = np.full((h, w), bg, np.int8)
    g[:, left] = wall
    g[6:8, :left - 3] = wall
    g[12:14, 3:5] = wall
    g[10:13, left + 3:left + 5] = wall
    g[0, 2:6] = primary
    g[0, 2] = accent
    # Enough positions to defeat any earlier input collision.
    for c in range(1, left - 4):
        trial = g.copy()
        trial[0, 2:6] = bg
        trial[0, c:c + 4] = primary
        trial[0, c] = accent
        yield np.rot90(trial, rotation).copy()


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ns = _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable_palette = bool(rng.integers(2))
    base_palette = tuple(int(v) for v in rng.choice(10, 4, replace=False))
    base_rotation = int(rng.integers(4))
    seen, teaching = set(), []
    puzzle = {'train': [], 'test': []}
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            selection = 'small' if witness == 1 else 'large'
            rotation = (base_rotation + (int(rng.choice((1, 3))) if split == 'test'
                                        else (2 if index >= 2 and rng.random() < .3 else 0))) % 4
            palette = base_palette if stable_palette else tuple(int(v) for v in rng.choice(10, 4, replace=False))
            accepted = None
            for attempt in range(120):
                g = _scene(rng, selection, split == 'test', rotation,
                           bool(rng.integers(2)), palette, witness)
                if g is None or _key(g) in seen:
                    continue
                f = _features(g)
                if witness == 1 and not (f['selected_small'] and f['hole_diff']
                                          and f['erasure_diff'] and f['detour'] and f['diagonal_diff']):
                    continue
                if witness == 2 and not f['selected_large']:
                    continue
                if split == 'train' and index == nt - 1 and not _covered(teaching + [f]):
                    continue
                accepted = g
                break
            if accepted is None:
                for fallback_index in range(14):
                    for g in _fallback(rng, witness or 2, rotation, palette, fallback_index):
                        if _key(g) in seen:
                            continue
                        f = _features(g)
                        if witness == 1 and not (f['selected_small'] and f['hole_diff'] and f['erasure_diff'] and f['detour']):
                            continue
                        if split == 'train' and index == nt - 1 and not _covered(teaching + [f]):
                            continue
                        accepted = g
                        break
                    if accepted is not None:
                        break
            if accepted is None:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(accepted))
            puzzle[split].append({'input': accepted, 'output': solve(accepted)})
            if split == 'train':
                teaching.append(_features(accepted))
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, geometry, pixel invariants and observable teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            assert _key(g) not in seen, "duplicate input"
            seen.add(_key(g))
            z = _analyze(g)
            assert not _checkerboard(z['walls']), "ambiguous diagonal-only geometry"
            for cells in _components(~z['walls']):
                assert any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in cells), "nested room"
            assert np.array_equal(out, solve(g)), "incorrect output"
            assert np.array_equal(out[z['entry']], g[z['entry']]), "entrance changed"
            assert np.all(out[z['walls'] & ~z['kept']] == z['bg'])
            assert np.array_equal(out[z['selected'] & ~z['rim']], g[z['selected'] & ~z['rim']])
            assert np.all(out[z['rim'] & ~z['entry'] & (z['d'] % 2 == 1)] == z['bg'])
            assert set(np.unique(out)) <= set(np.unique(g)), "unencoded output color"
            if split == 'train':
                teaching.append(_features(g))
    assert _covered(teaching), "demonstrations fail to distinguish core alternatives"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded checks on 42 seeds, all legal counts, and semantic metamorphisms."""
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(x['input']) for s in p for x in p[s]))
        for s in p:
            for pair in p[s]:
                g = pair['input']
                f = _features(g)
                records.append((s, f))
                z = _analyze(g)
                roles = np.zeros(g.shape, np.int8)
                for i, name in enumerate(('wall', 'primary', 'accent'), 1):
                    roles[g == z[name]] = i
                structures.add(_key(roles))
        assert len({_features(x['input'])['length'] for s in p for x in p[s]}) >= 2
    assert len(signatures) == 42
    assert len(structures) >= 190
    assert len({f['shape'] for _, f in records}) >= 65
    assert len({f['length'] for _, f in records}) >= 6
    assert {0, 1, 2, 3} <= {f['islands'] for _, f in records}
    assert {2, 3} <= {f['compartments'] for _, f in records}
    assert max(f['max_time'] for _, f in records) >= 40
    for split in ('train', 'test'):
        assert {f['side'] for s, f in records if s == split} == set(range(4))
    p = generate_puzzle(731, 7, 3)
    assert _equal(p, generate_puzzle(731, 7, 3))
    assert not _equal(p, generate_puzzle(732, 7, 3))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(0, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotating/reflection and relabeling commute with solving.
    for pair in p['train'][:2] + p['test'][:1]:
        g, out = pair['input'], pair['output']
        copy = g.copy()
        solve(g)
        assert np.array_equal(g, copy)
        for k in range(4):
            for reflect in (False, True):
                a, b = np.rot90(g, k), np.rot90(out, k)
                if reflect:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b)
        palette = np.array([9, 4, 7, 1, 8, 3, 0, 6, 2, 5])
        assert np.array_equal(solve(palette[g]), palette[out])
        z = _analyze(g)
        # Remove far-side standalone components: selected answer cannot change.
        clean = g.copy()
        clean[z['walls'] & ~z['kept']] = z['bg']
        assert np.array_equal(solve(clean), out)
        # Swap accent between entrance ends: geometry/time stays exactly the same.
        entry_cells = list(map(tuple, np.argwhere(z['entry'])))
        old = tuple(np.argwhere(g == z['accent'])[0])
        other = entry_cells[-1] if old == entry_cells[0] else entry_cells[0]
        moved = g.copy()
        moved[old], moved[other] = z['primary'], z['accent']
        expected = out.copy()
        expected[old], expected[other] = z['primary'], z['accent']
        assert np.array_equal(solve(moved), expected)
    broken = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, every legal count, and semantic invariants')
