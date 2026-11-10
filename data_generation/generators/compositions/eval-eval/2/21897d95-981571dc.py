"""Fold-linked paint: flood fill on a surface whose mirror mates are identified.

Zero is missing fill, one is T ink, and 2..9 are literal paint colors.
The solver reconstructs folds and territories entirely from the queried input.
"""
import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a.astype(np.int8, copy=True)


def _add(p, d, scale=1):
    return p[0] + scale*d[0], p[1] + scale*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _geometry(center, direction, shape):
    perpendicular = (-direction[1], direction[0])
    tip = _add(center, direction)
    arms = (_add(center, perpendicular), _add(center, perpendicular, -1))
    sides = (_add(tip, perpendicular), _add(tip, perpendicular, -1))
    target, back = _add(center, direction, 2), _add(center, direction, -1)
    write = frozenset((center, tip, *arms))
    context = frozenset((*sides, back, target))
    if not all(_inside(p, shape) for p in write | context):
        return None
    return dict(center=center, direction=direction, tip=tip, arms=arms,
                sides=sides, back=back, targetcell=target, write=write,
                footprint=write | context)


def _arrows(grid):
    ink = set(map(tuple, np.argwhere(grid == 1)))
    _need(bool(ink), 'no T instructions')
    centers = sorted({_add(p, d) for p in ink for d in DIRECTIONS})
    arrows, covered, writes = [], set(), set()
    for center in centers:
        for direction in DIRECTIONS:
            geo = _geometry(center, direction, grid.shape)
            if geo is None or not all(grid[p] == 1 for p in (*geo['arms'], geo['tip'])):
                continue
            host = int(grid[geo['back']])
            if host < 2 or any(grid[p] != host for p in geo['sides']):
                continue
            if grid[geo['targetcell']] < 2 or grid[geo['targetcell']] == host:
                continue
            if grid[center] == 0:
                continue
            _need(not writes & geo['write'], 'overlapping or ambiguous Ts')
            writes.update(geo['write'])
            covered.update(p for p in geo['write'] if grid[p] == 1)
            arrows.append(dict(geo, host=host, explicit=bool(grid[center] != 1),
                               payload=int(grid[center]) if grid[center] != 1 else host))
    _need(bool(arrows) and covered == ink, 'unexplained T ink')
    masked = grid.copy()
    for arrow in arrows:
        for p in arrow['write']:
            masked[p] = arrow['host']
    return masked, arrows


def _axis_candidates(grid, dimension):
    a = grid if dimension == 0 else grid.T
    found = []
    # Enumerate every nontrivial through-cell and between-cell fold, including
    # folds with little evidence. No hidden preferred offset or score is used.
    for total in range(1, 2*a.shape[0]-2):
        left = np.array([i for i in range(a.shape[0]) if i < total-i < a.shape[0]], dtype=int)
        x, y = a[left], a[total-left]
        visible = (x != 0) & (y != 0)
        if np.all(x[visible] == y[visible]):
            found.append(total)
    return found


def _orbits(shape, axes):
    h, w = shape
    a, b = axes
    seen = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rows = {r, a-r} if 0 <= a-r < h else {r}
            cols = {c, b-c} if 0 <= b-c < w else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            yield orbit


def _restore(masked):
    candidates = [_axis_candidates(masked, d) for d in (0, 1)]
    _need(all(len(c) == 1 for c in candidates), 'folds are not unique')
    axes = tuple(c[0] for c in candidates)
    clean = masked.copy()
    for orbit in _orbits(masked.shape, axes):
        values = {int(masked[p]) for p in orbit if masked[p] != 0}
        _need(len(values) == 1, 'empty or contradictory reflection orbit')
        color = next(iter(values))
        for p in orbit:
            clean[p] = color
    return clean, axes


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    cells, colors = [], []
    for start in np.ndindex(grid.shape):
        if ids[start] >= 0:
            continue
        label, color = len(cells), int(grid[start])
        stack, points = [start], []
        ids[start] = label
        while stack:
            p = stack.pop()
            points.append(p)
            for d in DIRECTIONS:
                q = _add(p, d)
                if _inside(q, grid.shape) and ids[q] < 0 and grid[q] == color:
                    ids[q] = label
                    stack.append(q)
        cells.append(tuple(points))
        colors.append(color)
    return ids, cells, colors


def _folded_regions(clean, axes):
    component_ids, component_cells, component_colors = _components(clean)
    parent = list(range(len(component_cells)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for orbit in _orbits(clean.shape, axes):
        first = root(int(component_ids[orbit[0]]))
        for p in orbit[1:]:
            other = root(int(component_ids[p]))
            _need(component_colors[first] == component_colors[other], 'mirror fills disagree')
            parent[other] = first
    rename, ids = {}, np.empty(clean.shape, dtype=np.int16)
    for p in np.ndindex(clean.shape):
        r = root(int(component_ids[p]))
        ids[p] = rename.setdefault(r, len(rename))
    cells = [tuple(map(tuple, np.argwhere(ids == i))) for i in range(len(rename))]
    colors = [int(clean[s[0]]) for s in cells]
    return ids, cells, colors, component_ids


def _paint(clean, ids, assignments):
    result = clean.copy()
    for target, payload in assignments.items():
        result[ids == target] = payload
    return result


def _decode(input_grid):
    grid = _grid(input_grid)
    masked, arrows = _arrows(grid)
    clean, axes = _restore(masked)
    ids, cells, colors, component_ids = _folded_regions(clean, axes)
    assignments = {}
    for arrow in arrows:
        source, target = int(ids[arrow['center']]), int(ids[arrow['targetcell']])
        _need(source != target, 'arrow stays inside a folded territory')
        _need(all(clean[p] == arrow['host'] and ids[p] == source for p in arrow['write']),
              'annotation host cannot be restored')
        payload = arrow['payload'] if arrow['explicit'] else colors[source]
        _need(target not in assignments or assignments[target] == payload, 'conflicting folded assignments')
        assignments[target] = payload
        arrow.update(source=source, target=target, payload=payload)
    output = _paint(clean, ids, assignments)
    return output, dict(grid=grid, masked=masked, clean=clean, axes=axes,
                        ids=ids, cells=cells, colors=colors, component_ids=component_ids,
                        arrows=arrows, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode a single input; no puzzle, seed, labels, or cached plan is read."""
    return _decode(input_grid)[0]


def _features(info):
    arrows, colors, cells = info['arrows'], info['colors'], info['cells']
    assignments, cid = info['assignments'], info['component_ids']
    final = [assignments.get(i, color) for i, color in enumerate(colors)]
    locality = any(colors[i] == colors[j] and final[i] != final[j]
                   and len(cells[i]) >= 8 and len(cells[j]) >= 8
                   for i in range(len(colors)) for j in range(i))
    chain = any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                and a['payload'] != b['payload'] for a in arrows for b in arrows if a is not b)
    broadcast = any(len({int(cid[p]) for p in cells[t]}) > 1 and colors[t] != v
                    for t, v in assignments.items())
    direct_components = {int(cid[a['targetcell']]) for a in arrows}
    broadcast_cells = sum(sum(int(cid[p]) not in direct_components for p in cells[t])
                          for t, v in assignments.items() if colors[t] != v)
    large_target = any(np.count_nonzero(cid == cid[a['targetcell']]) >= 6 for a in arrows)
    diagonal = 0
    a, b = info['axes']
    for r, c in np.argwhere(info['grid'] == 0):
        mates = ((a-int(r), int(c)), (int(r), b-int(c)))
        corner = (a-int(r), b-int(c))
        if (all(_inside(p, cid.shape) and info['grid'][p] == 0 for p in mates)
                and _inside(corner, cid.shape) and info['grid'][corner] >= 2):
            diagonal += 1
    return dict(shape=info['grid'].shape, axes=info['axes'], arrows=len(arrows),
                regions=len(cells), components=len(np.unique(cid)), locality=locality,
                chain=chain, broadcast=broadcast, broadcast_cells=broadcast_cells,
                large_target=large_target, diagonal_only=diagonal,
                explicit=sum(a['explicit'] for a in arrows),
                singleton_payload=any(a['explicit'] and a['payload'] not in colors for a in arrows),
                erased=int(np.count_nonzero(info['grid'] == 0)),
                directions=tuple(sorted({DIRECTIONS.index(a['direction']) for a in arrows})))


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        p = stack.pop()
        for d in DIRECTIONS:
            q = _add(p, d)
            if q in points:
                points.remove(q)
                stack.append(q)
    return not points


def _fragment(rng, shape, count, style):
    labels = np.zeros(shape, dtype=np.int16)
    rr, cc = np.indices(shape)
    if style == 1:
        # A rectangular mosaic with occasional inset islands, rather than a
        # mandatory lattice or a fixed nested-ring template.
        for label in range(1, count):
            h, w = shape
            rh, cw = int(rng.integers(2, max(3, h//2+1))), int(rng.integers(2, max(3, w//2+1)))
            r, c = int(rng.integers(h-rh+1)), int(rng.integers(w-cw+1))
            labels[r:r+rh, c:c+cw] = label
        if len(np.unique(labels)) != count:
            return None
        return labels
    for new in range(1, count):
        done = False
        for _ in range(45):
            sizes = np.bincount(labels.ravel(), minlength=new).astype(float)
            old = int(rng.choice(new, p=sizes/sizes.sum()))
            mask = labels == old
            rows, cols = np.nonzero(mask)
            r0, r1, c0, c1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
            axis = int(rng.integers(2))
            low, high = (r0, r1) if axis == 0 else (c0, c1)
            if high-low < 3:
                continue
            cut = int(rng.integers(low+1, high))
            along, across = (cc, rr) if axis == 0 else (rr, cc)
            path = np.full(shape[1-axis], cut)
            if style == 2:
                stride = int(rng.integers(2, 5))
                for start in range(0, len(path), stride):
                    path[start:start+stride] = int(np.clip(cut+rng.integers(-1, 2), low+1, high-1))
            part, other = mask & (across < path[along]), mask & (across >= path[along])
            if min(int(part.sum()), int(other.sum())) < 6 or not _connected(part) or not _connected(other):
                continue
            labels[part] = new
            done = True
            break
        if not done:
            return None
    return labels


def _field(rng, shape, axes, count, style):
    maps = []
    for size, total in zip(shape, axes):
        representatives = [min(i, total-i) if 0 <= total-i < size else i for i in range(size)]
        keys = sorted(set(representatives))
        maps.append(np.array([keys.index(i) for i in representatives]))
    fragment = _fragment(rng, (max(maps[0])+1, max(maps[1])+1), count, style)
    if fragment is None:
        return None
    neighbors = set()
    for x, y in ((fragment[:-1], fragment[1:]), (fragment[:, :-1], fragment[:, 1:])):
        neighbors.update(tuple(sorted((int(a), int(b)))) for a, b in zip(x.ravel(), y.ravel()) if a != b)
    repeats = [(a, b) for a in range(count) for b in range(a) if (b, a) not in neighbors]
    if not repeats:
        return None
    a, b = repeats[int(rng.integers(len(repeats)))]
    palette = rng.choice(np.arange(2, 10), size=count, replace=False)
    palette[a] = palette[b]
    fragment = palette[fragment]
    # Small reflected fragments create distinctive folds while leaving broad
    # territories. They are genuine fill, including holes and small islands.
    noise = rng.random(fragment.shape) < float(rng.uniform(.035, .10))
    fragment[noise] = rng.choice(np.unique(palette), int(noise.sum()))
    return fragment[maps[0][:, None], maps[1][None, :]].astype(np.int8)


def _boundary_arrows(clean, ids):
    candidates = []
    for center in np.ndindex(clean.shape):
        for direction in DIRECTIONS:
            geo = _geometry(center, direction, clean.shape)
            if geo is None:
                continue
            source, target = int(ids[center]), int(ids[geo['targetcell']])
            if source == target or clean[center] == clean[geo['targetcell']]:
                continue
            if all(ids[p] == source for p in geo['write'] | {geo['back'], *geo['sides']}):
                candidates.append(dict(geo, source=source, target=target))
    return candidates


def _plan(rng, clean, axes, rich, desired):
    ids, cells, colors, cid = _folded_regions(clean, axes)
    candidates = _boundary_arrows(clean, ids)
    if not candidates:
        return None
    rng.shuffle(candidates)
    arrows, assignments, blocked, contexts = [], {}, set(), set()
    reserve = None

    def legal(a):
        return (a['target'] != reserve and not a['write'] & (blocked | contexts)
                and not a['footprint'] & {p for x in arrows for p in x['write']})

    def append(a, payload, explicit=False):
        if not legal(a) or payload == colors[a['target']]:
            return False
        if a['target'] in assignments and assignments[a['target']] != payload:
            return False
        arrows.append(dict(a, payload=int(payload), explicit=explicit, host=colors[a['source']]))
        assignments[a['target']] = int(payload)
        blocked.update(a['write'])
        blocked.update(_add(p, d) for p in a['write'] for d in DIRECTIONS)
        contexts.update(a['footprint'] - a['write'])
        return True

    if rich:
        # Recolor one of two large same-colored folded territories, while that
        # newly recolored territory still sends its ORIGINAL fill onward.
        found = False
        for first in candidates:
            v = first['target']
            twins = [i for i, c in enumerate(colors) if i != v and c == colors[v]
                     and len(cells[i]) >= 8 and len(cells[v]) >= 8]
            if not twins:
                continue
            for second in candidates:
                if second['source'] != v or second['write'] & (first['footprint'] | {_add(p, d) for p in first['write'] for d in DIRECTIONS}):
                    continue
                if second['footprint'] & first['write']:
                    continue
                twin = next((i for i in twins if i not in (first['target'], second['target'])), None)
                if twin is None:
                    continue
                reserve = twin
                if append(first, colors[first['source']]) and append(second, colors[second['source']]):
                    found = True
                    break
            if found:
                break
        if not found:
            return None
        # A literal junction color absent from the restored field distinguishes
        # a payload token from an ordinary original-host transfer.
        spare = [c for c in range(2, 10) if c not in colors]
        if not spare or not any(append(a, int(rng.choice(spare)), True)
                                for a in candidates if a['target'] not in assignments):
            return None
    # First prefer a disconnected reflected target, so direct local paint is
    # observably insufficient even if erasure repair is otherwise retained.
    def broad_target(a):
        return (len({int(cid[p]) for p in cells[a['target']]}) > 1
                and np.count_nonzero(cid == cid[a['targetcell']]) >= 6)

    if not any(broad_target(a) for a in arrows):
        pool = [a for a in candidates if broad_target(a)]
        if not any(append(a, colors[a['source']]) for a in pool):
            return None
    for a in candidates:
        if len(arrows) >= desired:
            break
        if a['target'] in assignments:
            payload = assignments[a['target']]
            explicit = payload != colors[a['source']]
        else:
            explicit = bool(rng.random() < .3)
            payload = int(rng.choice([c for c in range(2, 10) if c != colors[a['target']]])) if explicit else colors[a['source']]
        append(a, payload, explicit)
    if len(arrows) < (3 if rich else 1):
        return None
    return arrows, assignments, ids


def _mask(rng, clean, axes, arrows, ids, assignments, hard, is_test):
    protected = {p for a in arrows for p in a['footprint']}
    mask = np.zeros(clean.shape, dtype=bool)
    h, w = clean.shape
    kind = int(rng.integers(5))
    if kind == 0:
        mask = rng.random(clean.shape) < float(rng.uniform(.025, .11))
    else:
        for _ in range(int(rng.integers(1, 5 if is_test else 4))):
            rh, cw = int(rng.integers(1, max(3, h//4))), int(rng.integers(1, max(3, w//4)))
            if kind == 2:
                rh = 1
            if kind == 3:
                cw = 1
            r0, c0 = int(rng.integers(h-rh+1)), int(rng.integers(w-cw+1))
            for dr in range(rh):
                extent = cw if kind != 4 else 1+(dr*(cw-1))//max(1, rh-1)
                mask[r0+dr, c0:c0+extent] = True
    for p in protected:
        mask[p] = False
    orbits = list(_orbits(clean.shape, axes))
    forced = set()
    if hard:
        available = [o for o in orbits if len(o) == 4 and not set(o) & protected]
        targeted = [o for o in available if int(ids[o[0]]) in assignments]
        choices = targeted or available
        if not choices:
            return None
        for _ in range(2 if is_test else 1):
            orbit = choices[int(rng.integers(len(choices)))]
            witness = orbit[int(rng.integers(4))]
            for p in orbit:
                mask[p] = p != witness
                if p != witness:
                    forced.add(p)
    for orbit in orbits:
        if all(mask[p] for p in orbit):
            mask[orbit[int(rng.integers(len(orbit)))]] = False
    masked = clean.copy()
    masked[mask] = 0
    # Reveal enough ordinary fill to refute every competing fold. Retain the
    # deliberately hard gaps; if a false fold has no available counterexample,
    # reject this candidate rather than invent a tie-breaking convention.
    for d in (0, 1):
        for wrong in _axis_candidates(masked, d):
            if wrong == axes[d]:
                continue
            mismatch = []
            for p in np.ndindex(clean.shape):
                q = (wrong-p[0], p[1]) if d == 0 else (p[0], wrong-p[1])
                if _inside(q, clean.shape) and clean[p] != clean[q] and p not in forced and q not in forced:
                    mismatch.append((int(mask[p])+int(mask[q]), p, q))
            if not mismatch:
                return None
            _, p, q = min(mismatch)
            masked[p], masked[q] = clean[p], clean[q]
    if not np.any(masked == 0) or any(_axis_candidates(masked, d) != [axes[d]] for d in (0, 1)):
        return None
    return masked


def _make_pair(rng, profile, is_test):
    rich, hard = profile == 'rich', profile != 'simple'
    for _ in range(260):
        low, high = (23, 31) if is_test else (17, 25)
        shape = tuple(int(x) for x in rng.integers(low, high, size=2))
        if rng.random() < .2:
            shape = (shape[0], shape[0])
        offsets = [0 if rng.random() < .45 else int(rng.integers(-3, 4)) for _ in range(2)]
        axes = tuple(n-1+o for n, o in zip(shape, offsets))
        count, style = int(rng.integers(4, 8)), int(rng.integers(3))
        clean = _field(rng, shape, axes, count, style)
        if clean is None or any(_axis_candidates(clean, d) != [axes[d]] for d in (0, 1)):
            continue
        desired = int(rng.integers(3, 6)) if rich else int(rng.integers(1, 5 if is_test else 4))
        plan = _plan(rng, clean, axes, rich, desired)
        if plan is None:
            continue
        arrows, assignments, ids = plan
        inp = _mask(rng, clean, axes, arrows, ids, assignments, hard, is_test)
        if inp is None:
            continue
        for arrow in arrows:
            for p in arrow['write']:
                inp[p] = 1
            if arrow['explicit']:
                inp[arrow['center']] = arrow['payload']
        # Labels are constructed from the clean folded surface. A fresh input
        # decoder must independently recover the same surface and assignments.
        out = _paint(clean, ids, assignments)
        try:
            predicted, info = _decode(inp)
        except AssertionError:
            continue
        f = _features(info)
        if (not np.array_equal(predicted, out) or not np.array_equal(info['clean'], clean)
                or not f['broadcast'] or f['broadcast_cells'] < 6 or not f['large_target']):
            continue
        if rich and not (f['chain'] and f['locality'] and f['singleton_payload']):
            continue
        if hard and not f['diagonal_only']:
            continue
        return dict(input=inp, output=out)
    raise RuntimeError('bounded folded-territory construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not 2 <= num_train <= 10
            or not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require integer counts 2..10 / 1..4 and num_train > num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    profiles = ['fold', 'rich'] + [str(rng.choice(['simple', 'fold', 'rich'])) for _ in range(num_train-2)]
    rng.shuffle(profiles)
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            profile = profiles[index] if split == 'train' else str(rng.choice(['fold', 'rich']))
            for _ in range(12):
                pair = _make_pair(rng, profile, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _local_only(info):
    out = info['clean'].copy()
    for arrow in info['arrows']:
        component = info['component_ids'][arrow['targetcell']]
        out[info['component_ids'] == component] = arrow['payload']
    return out


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'illegal counts')
    seen, taught = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape and _key(inp) not in seen, 'shape mismatch or duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'incorrect folded paint')
            _need(np.all(out >= 2), 'annotation or erasure survives')
            _need(np.any(inp == 0), 'missing-value concept absent')
            _need(not np.array_equal(out, info['clean']), 'paint concept absent')
            for orbit in _orbits(out.shape, info['axes']):
                _need(len({int(out[p]) for p in orbit}) == 1, 'output breaks a fold')
                _need(any(info['masked'][p] >= 2 for p in orbit), 'unrecoverable original orbit')
            changed = info['clean'] != out
            for cells in info['cells']:
                _need(len({int(out[p]) for p in cells}) == 1, 'partial folded-territory painting')
            untouched = ~np.isin(info['ids'], list(info['assignments']))
            _need(np.array_equal(out[untouched], info['clean'][untouched]), 'untargeted territory changed')
            f = _features(info)
            _need(f['broadcast'], 'no mirror-linked paint witness')
            _need(f['broadcast_cells'] >= 6 and f['large_target'], 'paint witnesses are only tiny dots')
            # This stronger ablation retains missing-cell restoration and asks
            # whether ordinary component-only A painting would still suffice.
            _need(not np.array_equal(_local_only(info), out), 'local-paint shortcut survives')
            _need(bool(changed.any()), 'omitting A does not change the answer')
            if split == 'train':
                taught.update(k for k in ('locality', 'chain', 'singleton_payload', 'diagonal_only') if f[k])
    _need({'locality', 'chain', 'singleton_payload', 'diagonal_only'} <= taught,
          'demonstrations do not distinguish the required mechanisms')


def self_test() -> dict:
    """Bounded 42-seed variation audit, invariants, ablations and negative cases."""
    shapes, offsets, parities, palettes, directions, signatures, structures = set(), set(), set(), set(), set(), set(), set()
    counts, mask_sizes, region_counts = set(), set(), set()
    total = locality = chains = diagonal = explicit = 0
    example = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        if seed in (0, 41):
            _need(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), 'nondeterministic puzzle')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _decode(pair['input'].copy())
                _need(np.array_equal(predicted, pair['output']), 'solve disagreement')
                f = _features(info)
                h, w = f['shape']
                shapes.add((h, w))
                offsets.add(tuple(int(np.sign(a-n+1)) for a, n in zip(info['axes'], (h, w))))
                parities.add(tuple(a % 2 for a in info['axes']))
                palettes.add(tuple(sorted(set(info['colors']))))
                directions.update(f['directions'])
                counts.add(f['arrows'])
                mask_sizes.add(f['erased'])
                region_counts.add(f['regions'])
                renaming = {}
                normalized = tuple(renaming.setdefault(int(x), len(renaming)) for x in info['clean'].flat)
                structures.add((info['clean'].shape, normalized))
                total += 1
                locality += f['locality']
                chains += f['chain']
                diagonal += f['diagonal_only'] > 0
                explicit += f['explicit'] > 0
                _need(not np.array_equal(info['clean'], predicted), 'A ablation failed')
                _need(not np.array_equal(_local_only(info), predicted), 'B linkage ablation failed')
                if example is None and f['chain'] and f['locality']:
                    example = puzzle
                if seed < 3:
                    # The rule is covariant under rotations and reflections;
                    # this also exercises unseen arrow directions, fringe sides
                    # and geometry without consulting any generation metadata.
                    for turns in (1, 2, 3):
                        _need(np.array_equal(solve(np.rot90(pair['input'], turns)),
                                             np.rot90(pair['output'], turns)), 'rotation covariance failed')
                    _need(np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1]),
                          'reflection covariance failed')
                    palette_map = np.array([0, 1, 9, 7, 5, 3, 8, 6, 4, 2], dtype=np.int8)
                    _need(np.array_equal(solve(palette_map[pair['input']]), palette_map[pair['output']]),
                          'literal paint roles depend on their numeric values')
    _need(len(signatures) == 42 and len(structures) >= 180, 'geometry collapsed across seeds')
    _need(len(shapes) >= 60 and len(offsets) == 9 and len(parities) == 4, 'fold geometry variation missing')
    _need(directions == {0, 1, 2, 3} and len(counts) >= 4, 'arrow variation missing')
    _need(len(palettes) >= 30 and len(mask_sizes) >= 25 and len(region_counts) >= 8, 'structural variation missing')
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ns, nt, ns)
            _need((len(p['train']), len(p['test'])) == (nt, ns), 'ignored explicit counts')
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A wrong gap value and a wrong visible fill must both be rejected.
    for point_kind in (0, 2):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in example[s]] for s in example}
        pair = bad['test'][0]
        positions = np.argwhere(pair['input'] == 0 if point_kind == 0 else pair['input'] >= 2)
        p = tuple(positions[0])
        pair['output'][p] = 2+(int(pair['output'][p])-1) % 8
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad output was accepted')
    return dict(seeds=42, pairs=total, shapes=len(shapes), clean_structures=len(structures),
                fold_side_combinations=len(offsets), fold_parities=len(parities),
                palettes=len(palettes), arrow_counts=sorted(counts), region_counts=sorted(region_counts),
                locality_pairs=locality, simultaneous_chain_pairs=chains,
                diagonal_only_pairs=diagonal, explicit_pairs=explicit,
                semantic_verification='generator-owned invariants and ablations only')
