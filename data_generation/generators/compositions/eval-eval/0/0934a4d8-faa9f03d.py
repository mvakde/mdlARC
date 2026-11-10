"""Hidden switches: reflection-recovered controls activated by an erased window.

Standard library + NumPy only. No files, network, execution traces, or retained
sampling history are used by this module. solve sees only its input grid.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 746757583, 17473113, 794021321, 12301510, 47622634,
              918634229, 354219568, 768418463, 444815153, 995812351,
              546403749, 268911569, 523598193, 879078454, 144928613,
              639551757, 352402639, 315892427, 777893940, 696279146,
              22319389, 14070245, 119169633, 389906566, 63888008,
              668341334, 766700625, 124334232, 606159012, 705357978,
              939270225, 444889596, 560315838, 344365355, 62848886,
              732495554, 422337869, 457174696, 212972025, 974879808,
              362651429)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    n = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc) for i in range(n + 1)]


def _polyline(points):
    walk = [points[0]]
    for a, b in zip(points, points[1:]):
        walk.extend(_line(a, b)[1:])
    return walk


def _ray(p, d, shape):
    cells = set()
    p = _add(p, d)
    while _inside(p, shape):
        cells.add(p)
        p = _add(p, d)
    return cells


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _valid_path(walk):
    s = set(walk)
    return (len(s) == len(walk) and len(s) >= 4
            and all(len(_neighbors(p, s)) <= 2 for p in s)
            and sum(len(_neighbors(p, s)) == 1 for p in s) == 2)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _mask(a):
    candidates = []
    for value in np.unique(a):
        rr, cc = np.nonzero(a == value)
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        if r1 - r0 >= 2 and c1 - c0 >= 2 and (r1-r0)*(c1-c0) == len(rr):
            candidates.append((int(value), (r0, r1, c0, c1)))
    assert len(candidates) == 1, 'Need one unique rectangular overwrite color'
    return candidates[0]


def _recover(a):
    """Infer supported centered reflections, then read each masked cell's orbit."""
    mask, box = _mask(a)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > a.size and bg != mask
    axes = []
    for axis in (0, 1):
        mate = np.flip(a, axis=axis)
        visible = (a != mask) & (mate != mask)
        informative = visible & (a != bg)
        if informative.sum() >= 8 and np.array_equal(a[visible], mate[visible]):
            axes.append(axis)
    assert axes, 'No supported reflection'
    full = a.copy()
    h, w = a.shape
    for r, c in np.argwhere(a == mask):
        rows = {int(r), h-1-int(r)} if 0 in axes else {int(r)}
        cols = {int(c), w-1-int(c)} if 1 in axes else {int(c)}
        values = {int(a[x, y]) for x in rows for y in cols if a[x, y] != mask}
        assert len(values) == 1, 'Erased orbit or disagreeing witnesses'
        full[r, c] = values.pop()
    for axis in axes:
        assert np.array_equal(full, np.flip(full, axis=axis))
    return full, bg, mask, box, tuple(axes)


def _switches(full, bg):
    """Decode the singleton annotation role and trace branches from pixels."""
    candidates = []
    all_colors = set(map(int, np.unique(full))) - {bg}
    for color in sorted(all_colors):
        cells = {tuple(map(int, p)) for p in np.argwhere(full == color)}
        if len(cells) >= 2 and all(not _neighbors(p, cells) for p in cells):
            candidates.append(color)
    assert len(candidates) == 1, 'Annotation color must be uniquely isolated'
    marker = candidates[0]
    payloads = all_colors - {marker}
    assert payloads
    occupied = {tuple(map(int, p)) for p in np.argwhere(np.isin(full, list(payloads)))}
    layers = {c: {tuple(map(int, p)) for p in np.argwhere(full == c)} for c in payloads}
    component_at = {}
    for color, cells in layers.items():
        for component in _components(cells):
            assert len(component) >= 4
            assert all(len(_neighbors(p, component)) <= 2 for p in component)
            assert sum(len(_neighbors(p, component)) == 1 for p in component) == 2
            for p in component:
                component_at[p] = component
    # Latent paths of different colors are also separated: no crossing hierarchy.
    for p in occupied:
        assert all(full[q] == full[p] for q in _neighbors(p, occupied))
    result, used_components = [], set()
    for pos in np.argwhere(full == marker):
        q = tuple(map(int, pos))
        adjacent = _neighbors(q, occupied)
        assert len(adjacent) == 1, 'Switch must touch exactly one pivot'
        pivot = adjacent[0]
        d = q[0]-pivot[0], q[1]-pivot[1]
        color = int(full[pivot])
        component = component_at[pivot]
        component_key = color, min(component)
        assert component_key not in used_components, 'One switch per component'
        used_components.add(component_key)
        start = pivot[0]-d[0], pivot[1]-d[1]
        assert start in component
        perp = (-d[1], d[0])
        assert any(_add(pivot, t) in component for t in (perp, (-perp[0], -perp[1])))
        previous, current, walk = pivot, start, [pivot]
        for _ in range(full.size):
            assert current not in walk, 'Loop in old branch'
            walk.append(current)
            following = [p for p in _neighbors(current, component) if p != previous]
            if not following:
                break
            assert len(following) == 1
            previous, current = current, following[0]
        else:
            raise AssertionError('Unbounded branch')
        assert _edge(walk[-1], full.shape), 'Old branch must reach a boundary endpoint'
        result.append({'q': q, 'pivot': pivot, 'direction': d, 'color': color,
                       'old': set(walk[1:]), 'walk': tuple(walk),
                       'new': _ray(pivot, d, full.shape)})
    return marker, layers, result


def _covered(q, box):
    r0, r1, c0, c1 = box
    return r0 <= q[0] < r1 and c0 <= q[1] < c1


def _execute(full, bg, marker, active):
    """All removals precede drawing. New/new color conflicts are disallowed."""
    out = full.copy()
    out[out == marker] = bg
    for edit in active:
        for p in edit['old']:
            out[p] = bg
    owners = {}
    for edit in active:
        for p in edit['new']:
            assert p not in owners or owners[p] == edit['color'], 'Conflicting new rays'
            owners[p] = edit['color']
    for p, color in owners.items():
        out[p] = color
    return out


def _analyze(value):
    inp = _grid(value)
    full, bg, mask, box, axes = _recover(inp)
    marker, layers, switches = _switches(full, bg)
    active = [e for e in switches if _covered(e['q'], box)]
    assert active and len(active) < len(switches), 'Need active and visible dormant controls'
    output = _execute(full, bg, marker, active)
    return output, {'full': full, 'background': bg, 'mask': mask, 'box': box,
                    'axes': axes, 'marker': marker, 'layers': layers,
                    'switches': switches, 'active': active}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only inference; no seed, examples, cache, or construction state."""
    return _analyze(input_grid)[0]


def _walk(rng, shape, terminal=True, dual=True):
    h, w = shape
    if terminal:
        sides = [0, 2] if dual else [0, 1, 2]
        side = int(rng.choice(sides))
        if side < 2:
            p, axis = (0 if side == 0 else h-1, int(rng.integers(1, w-2))), 0
        else:
            p, axis = (int(rng.integers(1, h-2)), 0), 1
    else:
        p = int(rng.integers(1, h-2)), int(rng.integers(1, w-2))
        axis = int(rng.integers(2))
    points = [p]
    for _ in range(int(rng.integers(3 if terminal else 1, 7))):
        choices = [v for v in range(1, shape[axis]-2) if abs(v-points[-1][axis]) >= 2]
        rng.shuffle(choices)
        for v in choices:
            q = list(points[-1])
            q[axis] = v
            candidate = points + [tuple(q)]
            if _valid_path(_polyline(candidate)):
                points = candidate
                break
        else:
            break
        axis = 1-axis
    return points


def _base_scene(rng, shape, bg, marker, colors, dual):
    base = np.full(shape, bg, dtype=np.int8)
    used, annotations, edits = set(), set(), []
    wanted = int(rng.integers(1, 4))
    for _ in range(100):
        points = _walk(rng, shape, dual=dual)
        if len(points) < 3:
            continue
        walk = _polyline(points)
        cells = set(walk)
        if not _valid_path(walk) or cells & (used | annotations):
            continue
        if any(_neighbors(p, used | annotations) for p in cells):
            continue
        options = []
        for j in range(1, len(points)-1):
            pivot, prev = points[j], points[j-1]
            d = int(np.sign(pivot[0]-prev[0])), int(np.sign(pivot[1]-prev[1]))
            q = _add(pivot, d)
            if not _inside(q, shape) or q[1] >= shape[1]-1 or (dual and q[0] >= shape[0]-1):
                continue
            if q in cells | used | annotations or _neighbors(q, cells | used) != [pivot]:
                continue
            if _neighbors(q, annotations):
                continue
            at = walk.index(pivot)
            options.append({'q': q, 'pivot': pivot, 'direction': d,
                            'old': set(walk[:at]), 'walk': tuple(reversed(walk[:at+1]))})
        if not options:
            continue
        edit = options[int(rng.integers(len(options)))]
        color = int(colors[len(edits) % len(colors)])
        edit['color'] = color
        for p in cells:
            base[p] = color
        base[edit['q']] = marker
        used.update(cells)
        annotations.add(edit['q'])
        edits.append(edit)
        if len(edits) == wanted:
            break
    if not edits:
        return None
    # Passive components can be straight or bent and use independent colors.
    for index in range(int(rng.integers(1, 4))):
        for _ in range(35):
            walk = _polyline(_walk(rng, shape, terminal=False, dual=dual))
            cells = set(walk)
            if not _valid_path(walk) or cells & (used | annotations):
                continue
            if any(_neighbors(p, used | annotations) for p in cells):
                continue
            color = int(colors[(index+1) % len(colors)])
            for p in cells:
                base[p] = color
            used.update(cells)
            break
    return base, edits


def _reflect_scene(base, edits, bg, dual, odd_row, odd_col):
    h, w = base.shape
    if odd_col:
        full = np.concatenate((base, np.full((h, 1), bg, dtype=np.int8), base[:, ::-1]), axis=1)
    else:
        full = np.concatenate((base, base[:, ::-1]), axis=1)
    if dual:
        gap = np.full((1, full.shape[1]), bg, dtype=np.int8)
        full = np.concatenate((full, gap, full[::-1]), axis=0) if odd_row else np.concatenate((full, full[::-1]), axis=0)
    H, W = full.shape
    reflected = []
    for flip_r in ((False, True) if dual else (False,)):
        for flip_c in (False, True):
            def coord(p):
                return H-1-p[0] if flip_r else p[0], W-1-p[1] if flip_c else p[1]
            for e in edits:
                d = (-e['direction'][0] if flip_r else e['direction'][0],
                     -e['direction'][1] if flip_c else e['direction'][1])
                pivot = coord(e['pivot'])
                reflected.append({'q': coord(e['q']), 'pivot': pivot, 'direction': d,
                                  'color': e['color'], 'old': {coord(p) for p in e['old']},
                                  'walk': tuple(coord(p) for p in e['walk']),
                                  'new': _ray(pivot, d, full.shape)})
    return full, reflected


def _turns(walk):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _mask_boxes(rng, shape, switches, target):
    qs = [e['q'] for e in switches]
    groups = list(combinations(qs, target))
    rng.shuffle(groups)
    # At most 72 proposals, independent of requested pair counts and seeds.
    for group in groups[:24]:
        for _ in range(3):
            rr, cc = zip(*group)
            r0, r1, c0, c1 = min(rr), max(rr)+1, min(cc), max(cc)+1
            pads = list(map(int, rng.integers(0, 4, size=4)))
            r0, r1 = max(0, r0-pads[0]), min(shape[0], r1+pads[1])
            c0, c1 = max(0, c0-pads[2]), min(shape[1], c1+pads[3])
            if r1-r0 < 2:
                r0, r1 = max(0, r0-1), min(shape[0], r1+1)
            if c1-c0 < 2:
                c0, c1 = max(0, c0-1), min(shape[1], c1+1)
            box = r0, r1, c0, c1
            if sum(_covered(q, box) for q in qs) == target and (r1-r0)*(c1-c0) <= np.prod(shape)*.24:
                yield box


def _make_pair(rng, roles, target, dual, seen, witness=False, bent=False,
               pivot_outside=False, pivot_inside=False):
    bg, marker, mask, colors = roles
    for attempt in range(180):
        # Richer sectors support several independently shaped routes. Nothing
        # in inference assumes sectors, equal segment lengths, or their sizes.
        h = int(rng.integers(7, 16)) if dual else int(rng.integers(11, 31))
        w = int(rng.integers(7, 16))
        scene = _base_scene(rng, (h, w), bg, marker, colors, dual)
        if scene is None:
            continue
        base, edits = scene
        full, switches = _reflect_scene(base, edits, bg, dual,
                                         h < 15 and bool(rng.integers(2)),
                                         w < 15 and bool(rng.integers(2)))
        if np.count_nonzero(full == bg) <= full.size*.64:
            continue
        for box in _mask_boxes(rng, full.shape, switches, target):
            active = [e for e in switches if _covered(e['q'], box)]
            if pivot_outside and any(_covered(e['pivot'], box) for e in active):
                continue
            if pivot_inside and not any(_covered(e['pivot'], box) for e in active):
                continue
            if bent and not any(_turns(e['walk']) >= 1 for e in active):
                continue
            if witness and not any(any(int(full[p]) not in (bg, marker, e['color']) for p in e['new']) for e in active):
                continue
            r0, r1, c0, c1 = box
            inp = full.copy()
            inp[r0:r1, c0:c1] = mask
            try:
                out = _execute(full, bg, marker, active)
                predicted, info = _analyze(inp)
                assert np.array_equal(info['full'], full)
                assert np.array_equal(predicted, out)
                assert len(info['active']) == target
                restored = full.copy()
                restored[restored == marker] = bg
                assert any(not _covered(tuple(map(int, p)), box) for p in np.argwhere(out != restored))
                # A dormant reference must visibly keep some of its old route.
                rays = set().union(*(e['new'] for e in active))
                removed = set().union(*(e['old'] for e in active))
                assert any(e['old']-rays-removed for e in switches if e not in active)
            except AssertionError:
                continue
            # D4 acts on both input and construction label; roles are scalar.
            rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
            if reflected:
                inp, out = inp[:, ::-1], out[:, ::-1]
            inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
            if _key(inp) not in seen:
                return {'input': inp, 'output': out}
    raise RuntimeError('Bounded circuit sampling exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) if seed >= 0 else int(seed) % (1 << 64))
    palette = list(map(int, rng.permutation(10)))
    vary_roles = bool(rng.integers(2))
    rich = bool(rng.integers(2)) and nt >= 3
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+nq):
        train = index < nt
        target = (2 if rich and index >= 2 and train and index % 2 == 0 else 1) if train else (3 if rich else 2)
        # Dual reflection is established before a seam-overlapping query. Some
        # single-control demonstrations also have just one global reflection.
        dual = not train or index < 2 or bool(rng.integers(2))
        colors = palette.copy()
        if vary_roles and index:
            colors = list(map(int, rng.permutation(10)))
        n_payload = int(rng.integers(2, 5))
        roles = colors[0], colors[1], colors[2], tuple(colors[3:3+n_payload])
        pair = _make_pair(rng, roles, target, dual, seen, witness=(index == 0), bent=(index == 1),
                          pivot_outside=(index == 0), pivot_inside=(index == 1))
        puzzle['train' if train else 'test'].append(pair)
        seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    out, info = _analyze(pair['input'])
    full, active, box = info['full'], info['active'], info['box']
    bg, marker = info['background'], info['marker']
    r0, r1, c0, c1 = box
    rays = set().union(*(e['new'] for e in active))
    removed = set().union(*(e['old'] for e in active))
    dormant = [e for e in info['switches'] if e not in active]
    # Two explicit counterfactuals, calculated from public input reconstruction.
    repair_only = full.copy()
    repair_only[repair_only == marker] = bg
    try:
        execute_all = _execute(full, bg, marker, info['switches'])
        without_selection = int(np.count_nonzero(out != execute_all))
    except AssertionError:
        # Activating dormant controls may introduce different-color ray
        # collisions; that counterfactual then has no legal family output.
        without_selection = None
    without_A = np.asarray(pair['input']).copy()
    without_A[np.isin(without_A, [info['mask'], marker])] = bg
    return {'shape': tuple(full.shape), 'window': (r1-r0, c1-c0),
            'axes': info['axes'], 'background': bg, 'marker': marker, 'mask': info['mask'],
            'payloads': tuple(sorted(info['layers'])), 'active_count': len(active),
            'switch_count': len(info['switches']), 'components': sum(len(_components(s)) for s in info['layers'].values()),
            'directions': tuple(e['direction'] for e in active),
            'old_bends': tuple(_turns(e['walk']) for e in active),
            'old_lengths': tuple(len(e['old']) for e in active),
            'ray_lengths': tuple(len(e['new']) for e in active),
            'hidden_pivots': sum(_covered(e['pivot'], box) for e in active),
            'seam_crossing': any((r0 <= (full.shape[0]-1)/2 < r1-1) if axis == 0 else
                                 (c0 <= (full.shape[1]-1)/2 < c1-1) for axis in info['axes']),
            'edge_window': r0 == 0 or c0 == 0 or r1 == full.shape[0] or c1 == full.shape[1],
            'ray_recolors': int(sum(int(full[p]) not in (bg, marker, next(e['color'] for e in active if p in e['new'])) for p in rays)),
            'retained_dormant_old': any(e['old']-rays-removed for e in dormant),
            'without_A_diff': int(np.count_nonzero(out != without_A)),
            'without_B_diff': int(np.count_nonzero(out != repair_only)),
            'without_selection_diff': without_selection,
            'outside_window_changes': int(sum(not _covered(tuple(map(int, p)), box) for p in np.argwhere(out != repair_only)))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            prediction, info = _analyze(inp)
            assert np.array_equal(out, prediction), 'Incorrect circuit edit or preservation'
            assert info['mask'] not in out and info['marker'] not in out
            m = measure_pair(pair)
            assert m['retained_dormant_old']
            assert m['without_A_diff'] and m['without_B_diff'] and m['outside_window_changes']
            assert m['without_selection_diff'] is None or m['without_selection_diff'] > 0
            features[split].append(m)
    # Minimal demonstrations distinguish selector semantics, straight and bent
    # deletion, and new-ray precedence before multiple controls are combined.
    assert any(m['ray_recolors'] for m in features['train'])
    assert any(any(b >= 1 for b in m['old_bends']) for m in features['train'])
    assert any(m['hidden_pivots'] == 0 for m in features['train'])
    assert any(m['hidden_pivots'] > 0 for m in features['train'])
    assert max(m['active_count'] for m in features['test']) > max(m['active_count'] for m in features['train'])


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
    """42 seeds, bounded streaming metrics, equivariance, negatives, and counts."""
    signatures, shapes, windows, roles, components, bends, counts, directions = (set() for _ in range(8))
    totals = {'pairs': 0, 'seam_crossing': 0, 'edge_window': 0, 'hidden_pivots': 0, 'ray_recolors': 0}
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(_signature(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                m = measure_pair(pair)
                shapes.add(m['shape'])
                windows.add(m['window'])
                roles.add((m['background'], m['marker'], m['mask']))
                components.add(m['components'])
                bends.update(m['old_bends'])
                counts.add(m['active_count'])
                directions.update(m['directions'])
                totals['pairs'] += 1
                for key in totals.keys() - {'pairs'}:
                    totals[key] += int(m[key])
                _, info = _analyze(pair['input'])
                assert np.array_equal(_execute(info['full'], info['background'], info['marker'],
                                               list(reversed(info['active']))), pair['output'])
                transformed = np.rot90(pair['input'])[:, ::-1].copy()
                assert np.array_equal(solve(transformed), np.rot90(pair['output'])[:, ::-1])
        if seed == 0:
            assert _signature(p) == _signature(generate_puzzle(seed))
            # Independently permuted scalar roles preserve the spatial rule.
            perm = np.random.default_rng(123).permutation(10)
            for pair in p['train']+p['test']:
                assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
            # Wrong preserved, new-ray and removed-branch pixels must all fail.
            q = p['test'][0]
            _, info = _analyze(q['input'])
            sites = [tuple(map(int, np.argwhere(q['input'] == q['output'])[0])),
                     min(info['active'][0]['new']), min(info['active'][0]['old'])]
            for site in sites:
                bad = {'train': p['train'], 'test': [dict(q)]}
                bad['test'][0]['output'] = q['output'].copy()
                bad['test'][0]['output'][site] = (int(q['output'][site])+1) % 10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Bad output accepted')
            # Conflict in a visible reflection witness and completely erased
            # orbits must fail rather than receive an invented completion.
            full = info['full']
            changed = q['input'].copy()
            changed[0, 0] = info['marker'] if changed[0, 0] != info['marker'] else info['background']
            try:
                solve(changed)
            except AssertionError:
                pass
            else:
                raise AssertionError('Conflicting visible evidence accepted')
            erased = full.copy()
            # Centered rectangle erases its own complete orbit under any axes.
            h, w = erased.shape
            erased[(h-1)//2:(h+1)//2+1, (w-1)//2:(w+1)//2+1] = info['mask']
            try:
                solve(erased)
            except AssertionError:
                pass
            else:
                raise AssertionError('Completely erased orbit accepted')
    assert len(signatures) == 42 and len(shapes) >= 30 and len(windows) >= 25
    assert len(roles) >= 25 and len(components) >= 5
    assert {0, 1, 2} <= bends and {1, 2, 3} <= counts and directions == set(DIRS)
    assert all(totals[k] > 0 for k in totals)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(9187+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (3, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'shapes': len(shapes),
            'window_shapes': len(windows), 'role_assignments': len(roles),
            'component_counts': sorted(components), 'old_bends': sorted(bends),
            'active_counts': sorted(counts), **totals}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
