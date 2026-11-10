"""Companion gates: rotational companionship changes seeded contour topology.

Only Python's standard library and NumPy are used. No parent code is imported.
All transformation conventions are inferred from the input or fixed here.
"""
from collections import Counter
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not np.issubdtype(g.dtype, np.integer)
            or not all(1 <= n <= 30 for n in g.shape)
            or np.any((g < 0) | (g > 9))):
        raise ValueError('expected a 1..30 integer ARC grid with colors 0..9')
    return g.astype(int, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    stack = [start]
    reached[start] = True
    h, w = allowed.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


def _components(mask, directions=N4):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        cells, stack = [], [(r, c)]
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(cells)
    return result


def _mask(cells):
    top = min(r for r, c in cells)
    left = min(c for r, c in cells)
    m = np.zeros((max(r for r, c in cells)-top+1, max(c for r, c in cells)-left+1), bool)
    for r, c in cells:
        m[r-top, c-left] = True
    return m


def _key(m):
    return m.shape, tuple(int(v) for v in m.flat)


def _canonical(m, reflect=False):
    forms = [_key(np.rot90(m, k)) for k in range(4)]
    if reflect:
        forms += [_key(np.rot90(np.fliplr(m), k)) for k in range(4)]
    return min(forms)


def _neighbors(mask, diagonal=False):
    p = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dr, dc in N8 if diagonal else N4:
        out |= p[1+dr:1+dr+h, 1+dc:1+dc+w]
    return out


def _roles(g):
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    if int(counts.max()) * 2 <= g.size:
        raise ValueError('background must be the strict majority')
    seeds = [int(v) for v, n in zip(vals, counts) if n == 1]
    if len(seeds) != 1:
        raise ValueError('seed must be the only singleton color')
    seed = seeds[0]
    edge = set(map(int, np.concatenate((g[0], g[-1], g[:, 0], g[:, -1])))) - {bg, seed}
    if len(edge) != 1:
        raise ValueError('scaffold must be the unique other perimeter color')
    wall = edge.pop()
    return bg, wall, seed


def _analyze(g, matching='rotation', open_gates=True, use_seed=True, holes=True,
             diagonal=True, object_connectivity=N8, exactly_two=False, local_only=False):
    bg, wall, seed = _roles(g)
    objects = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, wall, seed}):
        for cells in _components(g == color, object_connectivity):
            m = _mask(cells)
            key = (_key(m) if matching == 'identity' else color if matching == 'color'
                   else _canonical(m, matching == 'reflection'))
            objects.append({'cells': cells, 'mask': m, 'key': key, 'color': color})
    position = tuple(map(int, np.argwhere(g == seed)[0]))
    original = _flood((g == bg) | (g == seed), position)
    eligible = _neighbors(original)
    counts = Counter(o['key'] for o in objects if not local_only or any(eligible[p] for p in o['cells']))
    opened = np.zeros(g.shape, bool)
    for o in objects:
        o['paired'] = counts[o['key']] == 2 if exactly_two else counts[o['key']] >= 2
        if open_gates and o['paired']:
            rr, cc = zip(*o['cells'])
            opened[rr, cc] = True
    passable = (g == bg) | (g == seed) | opened
    target = _flood(passable, position) if use_seed else passable.copy()
    external = _flood(np.pad(~target, 1, constant_values=True), (0, 0))
    contact = _neighbors(external, diagonal)[1:-1, 1:-1] if holes else _neighbors(
        np.pad(~target, 1, constant_values=True), diagonal)[1:-1, 1:-1]
    rim = target & contact
    near = _neighbors(target)
    out = g.copy()
    out[opened] = bg
    kept_objects = []
    for o in objects:
        if opened[o['cells'][0]]:
            kept_objects.append(False)
            continue
        keep = any(near[p] for p in o['cells'])
        kept_objects.append(keep)
        if not keep:
            rr, cc = zip(*o['cells'])
            out[rr, cc] = bg
    wall_components = _components(g == wall)
    for cells in wall_components:
        if not any(near[p] for p in cells):
            rr, cc = zip(*cells)
            out[rr, cc] = bg
    out[rim] = seed
    out[position] = seed
    return {'output': out, 'bg': bg, 'wall': wall, 'seed': seed, 'position': position,
            'objects': objects, 'counts': counts, 'opened': opened, 'target': target,
            'rim': rim, 'external': external[1:-1, 1:-1], 'kept': kept_objects,
            'wall_components': wall_components}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible roles, object shapes and topology; never mutate input."""
    return _analyze(_grid(input_grid))['output']


def _plug(rng, width=2, forbidden=(), chiral=False, attempts=120):
    """A spanning spine and a full crossbar make a genuine removable door."""
    for _ in range(attempts):
        h = int(rng.integers(3, 6))
        m = np.zeros((h, width), bool)
        m[:, 0] = True
        m[int(rng.choice([0, h-1])), :] = True
        for r in range(h):
            if rng.random() < .30:
                m[r, :int(rng.integers(1, width+1))] = True
        k = _canonical(m)
        if k not in forbidden and (not chiral or k != _canonical(np.fliplr(m))):
            return m
    # Exhaustive finite vocabulary also guarantees distinct chiral teaching plugs.
    for h in range(3, 6):
        for bits in range(1 << ((width-1)*h)):
            m = np.ones((h, width), bool)
            for r in range(h):
                for c in range(1, width):
                    m[r, c] = bool(bits & (1 << (r*(width-1)+c-1)))
                # Preserve a filled spine with contiguous row extensions.
                if width == 3 and m[r, 2]:
                    m[r, 1] = True
            if not m.any(axis=1).all() or not m.all(axis=1).any():
                continue
            k = _canonical(m)
            if k not in forbidden and (not chiral or k != _canonical(np.fliplr(m))):
                return m
    raise RuntimeError('plug vocabulary exhausted')


def _loose_shape(rng, forbidden, limit=4):
    for _ in range(100):
        h, w = int(rng.integers(2, limit+1)), int(rng.integers(2, limit+1))
        kind = int(rng.integers(6))
        m = np.zeros((h, w), bool)
        if kind == 0:
            m[:] = True
        elif kind == 1:
            m[:, 0] = m[-1, :] = True
        elif kind == 2:
            m[h//2, :] = m[:, w//2] = True
        elif kind == 3:
            m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
        elif kind == 4:
            m[:, 0] = True
            for r in range(h):
                m[r, :int(rng.integers(1, w+1))] = True
            m[-1, :] = True
        else:
            m[0, 0] = True
            m[1:, 1] = True
            m[-1, 1:] = True
        if _canonical(m) not in forbidden:
            return m
    for length in range(2, 10):
        m = np.ones((length, 1), bool)
        if _canonical(m) not in forbidden:
            return m
    raise RuntimeError('loose vocabulary exhausted')


def _scaffold(rng, style, witness=0, test=False):
    t = 2 if witness else int(rng.choice([2, 3], p=[.75, .25]))
    h = int(rng.integers(18, 31))
    if witness == 1:
        widths = [int(rng.integers(5, 7)), int(rng.integers(5, 7))]
        widths.append(min(30-4-sum(widths), sum(widths)+int(rng.integers(1, 3))))
    elif witness == 2:
        widths = [int(rng.integers(6, 9)) for _ in range(3)]
    else:
        widths = [int(rng.integers(7, 12)), int(rng.integers(9, 15))]
    if style == 'parallel' or witness:
        if not witness:
            n = 3 if test else int(rng.integers(2, 4))
            widths = [int(rng.integers(6, 9)) for _ in range(n)]
        w = sum(widths) + t*(len(widths)-1)
        g = np.zeros((h, w), np.int8)
        rooms, segments, c = [], [], 0
        for i, width in enumerate(widths):
            rooms.append((0, h, c, c+width))
            c += width
            if i+1 < len(widths):
                g[:, c:c+t] = 1
                segments.append(('v', 0, h, c, c+t))
                c += t
    else:
        w = min(30, sum(widths)+t)
        cut = widths[0]
        row = int(rng.integers(7, h-7))
        g = np.zeros((h, w), np.int8)
        g[:, cut:cut+t] = 1
        segments = [('v', 0, row, cut, cut+t), ('v', row+t, h, cut, cut+t)]
        if style == 'tee':
            g[row:row+t, cut+t:] = 1
            rooms = [(0, h, 0, cut), (0, row, cut+t, w), (row+t, h, cut+t, w)]
            segments.append(('h', row, row+t, cut+t, w))
        else:
            g[row:row+t, :] = 1
            rooms = [(0, row, 0, cut), (row+t, h, 0, cut),
                     (0, row, cut+t, w), (row+t, h, cut+t, w)]
            segments += [('h', row, row+t, 0, cut), ('h', row, row+t, cut+t, w)]
    return g, rooms, segments, t


def _paint_shape(g, m, position, value):
    r, c = position
    view = g[r:r+m.shape[0], c:c+m.shape[1]]
    view[m] = value


def _object_halo(g):
    motifs = g >= 3
    return motifs | _neighbors(motifs, True)


def _put_gate(g, m, segment, value, rng):
    axis, r0, r1, c0, c1 = segment
    local = m if axis == 'v' else np.rot90(m)
    mh, mw = local.shape
    candidates = ([(r, c0) for r in range(max(2, r0+1), min(g.shape[0]-mh-1, r1-mh))]
                  if axis == 'v' else
                  [(r0, c) for c in range(max(2, c0+1), min(g.shape[1]-mw-1, c1-mw))])
    rng.shuffle(candidates)
    halo = _object_halo(g)
    for r, c in candidates:
        if (np.all(g[r:r+mh, c:c+mw] == 1)
                and not halo[r:r+mh, c:c+mw][local].any()):
            _paint_shape(g, local, (r, c), value)
            return True
    return False


def _put_loose(g, m, room, value, rng, turns=None):
    r0, r1, c0, c1 = room
    turns = list(rng.permutation(4)) if turns is None else list(turns)
    halo = _object_halo(g) | _neighbors(g == 1, True) | (g == 1)
    for turn in turns:
        local = np.rot90(m, int(turn))
        mh, mw = local.shape
        candidates = [(r, c) for r in range(max(2, r0+1), min(g.shape[0]-mh, r1-mh))
                      for c in range(max(2, c0+1), min(g.shape[1]-mw, c1-mw))]
        rng.shuffle(candidates)
        for r, c in candidates:
            view = g[r:r+mh, c:c+mw]
            if np.all(view[local] == 0) and not halo[r:r+mh, c:c+mw][local].any():
                _paint_shape(g, local, (r, c), value)
                return True
    return False


def _make_scene(rng, witness, test, rotation):
    style = 'parallel' if witness else str(rng.choice(['parallel', 'tee', 'cross'],
                                                      p=[.10, .50, .40] if test else [.40, .35, .25]))
    g, rooms, segments, t = _scaffold(rng, style, witness, test)
    base = _plug(rng, t, chiral=True)
    forbidden = {_canonical(base), _canonical(np.fliplr(base))}
    closed = _plug(rng, t, forbidden, chiral=True)
    forbidden.update((_canonical(closed), _canonical(np.fliplr(closed))))
    # Role values: 0 blank, 1 scaffold, 2 seed, >=3 temporary motif colors.
    if witness:
        active, shut = (0, 1) if witness == 1 else (1, 0)
        if not _put_gate(g, base, segments[active], 3, rng):
            return None
        if not _put_gate(g, closed, segments[shut], 3, rng):
            return None
        if witness == 1:
            # The remote rotated mate is deliberately in the still locked room.
            if not _put_loose(g, base, rooms[2], 4, rng, [1, 3]):
                return None
            if not _put_loose(g, np.fliplr(closed), rooms[0], 5, rng, [0, 2]):
                return None
            seed_room = 0
        else:
            if not _put_loose(g, base, rooms[0], 4, rng, [0, 2]):
                return None
            if not _put_loose(g, base, rooms[2], 5, rng, [1, 3, 0, 2]):
                return None
            seed_room = 2
        island = np.eye(2, dtype=bool) if witness == 2 else _loose_shape(rng, forbidden, 3)
        forbidden.add(_canonical(island))
        if not _put_loose(g, island, rooms[1], int(rng.choice([3, 4, 5])), rng):
            return None
        # Add a wholly far-side unique token, distinct from the island.
        far = _loose_shape(rng, forbidden, 3)
        if not _put_loose(g, far, rooms[2 if witness == 1 else 0], 5, rng):
            return None
    else:
        order = list(map(int, rng.permutation(len(segments))))
        gate_count = min(len(segments), int(rng.integers(2, 5)))
        # Tests join at least two walls with a common gate shape, a topology cascade.
        active_count = min(gate_count, 2 if test else int(rng.integers(1, gate_count+1)))
        second = None
        if active_count >= 2 and rng.random() < .40:
            second = _plug(rng, t, forbidden, chiral=True)
            forbidden.update((_canonical(second), _canonical(np.fliplr(second))))
        for i, segid in enumerate(order[:gate_count]):
            m = (second if second is not None and i == 1 else base) if i < active_count else closed
            # Closed gates remain singleton: never duplicate closed in this scene.
            if i > active_count:
                m = _plug(rng, t, forbidden, chiral=True)
                forbidden.add(_canonical(m))
            if not _put_gate(g, m, segments[segid], int(rng.integers(3, 6)), rng):
                return None
        # A loose copy establishes companionship even when there is only one gate.
        candidates = list(map(int, rng.permutation(len(rooms))))
        if not any(_put_loose(g, base, rooms[i], 4, rng) for i in candidates):
            return None
        if second is not None:
            candidates = list(map(int, rng.permutation(len(rooms))))
            if not any(_put_loose(g, second, rooms[i], 5, rng) for i in candidates):
                return None
        if rng.random() < .40:
            candidates = list(map(int, rng.permutation(len(rooms))))
            if not any(_put_loose(g, base, rooms[i], 5, rng) for i in candidates):
                return None
        for _ in range(int(rng.integers(1, 4))):
            m = _loose_shape(rng, forbidden)
            forbidden.add(_canonical(m))
            candidates = list(map(int, rng.permutation(len(rooms))))
            any(_put_loose(g, m, rooms[i], int(rng.integers(3, 6)), rng) for i in candidates)
        seed_room = int(rng.integers(len(rooms)))
    # Irregular edge intrusions and enclosed scaffold islands, with no new doors.
    for _ in range(int(rng.integers(0, 4))):
        r0, r1, c0, c1 = rooms[int(rng.integers(len(rooms)))]
        rh, cw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        r = int(rng.integers(r0, max(r0+1, r1-rh+1)))
        c = int(rng.choice([c0, c1-cw]))
        trial = g.copy()
        cells = np.zeros(g.shape, bool)
        cells[r:r+rh, c:c+cw] = True
        if not np.all(g[cells] == 0) or (_neighbors(cells, True) & (g >= 3)).any():
            continue
        trial[cells] = 1
        # Guard all original blank room cells against accidental partitioning.
        if len(_components(trial == 0)) != len(_components(g == 0)):
            continue
        g = trial
    r0, r1, c0, c1 = rooms[seed_room]
    roommask = np.zeros(g.shape, bool)
    roommask[r0:r1, c0:c1] = True
    choices = np.argwhere((g == 0) & roommask)
    edge = ((choices[:, 0] == 0) | (choices[:, 0] == g.shape[0]-1)
            | (choices[:, 1] == 0) | (choices[:, 1] == g.shape[1]-1))
    subset = choices[~edge] if witness == 1 else choices[edge] if witness == 2 else choices
    if not len(subset):
        return None
    position = tuple(map(int, subset[int(rng.integers(len(subset)))]))
    g[position] = 2
    g = np.rot90(g, rotation)
    if rng.random() < .5 and not witness:
        g = np.fliplr(g)
    # Independent palette roles are visibly recoverable for every example.
    palette = rng.permutation(10)[:6]
    if not witness and rng.random() < .55:
        count = int(rng.integers(1, 4))
        palette[3:] = rng.choice(palette[3:3+count].copy(), 3, replace=True)
    inp = palette[g].astype(int)
    try:
        info = _analyze(inp)
    except ValueError:
        return None
    original = _flood((inp == info['bg']) | (inp == info['seed']), info['position'])
    original_passable = (inp == info['bg']) | (inp == info['seed'])
    if not (info['target'] & original_passable & ~original).any():
        return None
    if test:
        reached_rooms = sum(any(info['target'][p] for p in cells) for cells in _components(
            (inp == info['bg']) | (inp == info['seed'])))
        if reached_rooms < 3:
            return None
    if witness:
        f = _evidence(info, inp)
        if not (f['hole_effect'] and f['seed_effect'] and f['diagonal_effect']
                and f['erased_unique'] and f['different_color_match'] and f['cross_status_color']
                and f['color_effect']):
            return None
        if witness == 1 and not (f['mirror_effect'] and f['remote_effect'] and f['smaller']
                                and f['rotation_effect'] and f['local_effect']):
            return None
        if witness == 2 and not (3 in f['multiplicities'] and f['larger']
                                and f['connectivity_effect'] and f['threshold_effect']):
            return None
    return {'input': inp, 'output': info['output']}


def _evidence(info, g):
    bg, seed, target = info['bg'], info['seed'], info['target']
    obs = info['objects']
    original = _flood((g == bg) | (g == seed), info['position'])
    original_near = _neighbors(original)
    room_sizes = [len(c) for c in _components((g == bg) | (g == seed))]
    # An isolated remaining object has no cells in the exterior complement.
    hole = any(not o['paired'] and keep and not any(info['external'][p] for p in o['cells'])
               for o, keep in zip(obs, info['kept']))
    full_target = np.pad(~target, 1, constant_values=True)
    orth = _neighbors(_flood(full_target, (0, 0)))[1:-1, 1:-1]
    return {'shape': g.shape, 'roles': (bg, info['wall'], seed), 'objects': len(obs),
            'multiplicities': sorted(set(info['counts'].values())),
            'paired': sum(o['paired'] for o in obs),
            'opened_classes': sum(n >= 2 for n in info['counts'].values()),
            'gate_effect': bool((target & ((g == bg) | (g == seed)) & ~original).any()),
            'remote_effect': any(o['paired'] and not any(original_near[p] for p in o['cells'])
                                 and not any(target[p] for p in o['cells']) for o in obs),
            'seed_effect': not np.array_equal(info['output'], _analyze(g, use_seed=False)['output']),
            'hole_effect': hole and not np.array_equal(info['output'], _analyze(g, holes=False)['output']),
            'mirror_effect': not np.array_equal(info['output'], _analyze(g, matching='reflection')['output']),
            'rotation_effect': not np.array_equal(info['output'], _analyze(g, matching='identity')['output']),
            'color_effect': not np.array_equal(info['output'], _analyze(g, matching='color')['output']),
            'local_effect': not np.array_equal(info['output'], _analyze(g, local_only=True)['output']),
            'threshold_effect': not np.array_equal(info['output'], _analyze(g, exactly_two=True)['output']),
            'connectivity_effect': not np.array_equal(info['output'], _analyze(g, object_connectivity=N4)['output']),
            'diagonal_effect': bool((info['rim'] & ~orth).any()),
            'erased_unique': any(not o['paired'] and not keep for o, keep in zip(obs, info['kept'])),
            'different_color_match': any(a['key'] == b['key'] and a['color'] != b['color']
                                         for i, a in enumerate(obs) for b in obs[i+1:]),
            'cross_status_color': any(a['paired'] != b['paired'] and a['color'] == b['color']
                                      for i, a in enumerate(obs) for b in obs[i+1:]),
            'smaller': int(target.sum()) < max(room_sizes),
            'larger': int(target.sum()) > max(room_sizes),
            'original_rooms': len(room_sizes), 'reached': int(target.sum()),
            'areas': [len(o['cells']) for o in obs],
            'motifs': [_canonical(o['mask']) for o in obs]}


def _validate_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _validate_count(num_train, 2, 10, 'num_train')
    nq = _validate_count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = index+1 if split == 'train' and index < 2 else 0
            for _ in range(240):
                pair = _make_scene(rng, witness, split == 'test',
                                   rotation if witness else int(rng.integers(4)))
                if pair is None:
                    continue
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
        rng.shuffle(puzzle[split])
    return puzzle


def _fingerprint(g):
    return g.shape, g.astype(np.uint8).tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, evidence = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape
            key = _fingerprint(g)
            assert key not in seen
            seen.add(key)
            info = _analyze(g)
            assert np.array_equal(out, info['output']), 'incorrect contour/opening/filtering'
            assert not np.array_equal(g, out)
            assert out[info['position']] == info['seed']
            # Retained object cells preserve their colors, orientation and position.
            for o, keep in zip(info['objects'], info['kept']):
                assert len(o['cells']) >= 2
                assert all(0 < r < g.shape[0]-1 and 0 < c < g.shape[1]-1 for r, c in o['cells'])
                if keep:
                    assert all(out[p] == g[p] for p in o['cells'])
                elif not o['paired']:
                    assert all(out[p] == info['bg'] for p in o['cells'])
                else:
                    assert all(out[p] in (info['bg'], info['seed']) for p in o['cells'])
            # Only selected passable cells can acquire contour ink.
            assert np.all(info['target'][(out == info['seed'])])
            assert np.all((out[(g == info['bg']) & ~info['target']]) == info['bg'])
            f = _evidence(info, g)
            assert f['gate_effect'], 'matching must enlarge the reachable region'
            if split == 'train':
                evidence.append(f)
    for feature in ('remote_effect', 'mirror_effect', 'hole_effect', 'seed_effect',
                    'diagonal_effect', 'erased_unique', 'different_color_match',
                    'cross_status_color', 'smaller', 'larger', 'rotation_effect',
                    'color_effect', 'local_effect', 'threshold_effect', 'connectivity_effect'):
        assert any(f[feature] for f in evidence), f'missing training evidence: {feature}'
    assert any(2 in f['multiplicities'] for f in evidence)
    assert any(3 in f['multiplicities'] for f in evidence)


def self_test():
    fallback_keys = set()
    for width in (2, 3):
        for _ in range(3):
            m = _plug(np.random.default_rng(7), width, fallback_keys, True, attempts=0)
            assert len(_components(m)) == 1 and m.all(axis=1).any()
            assert _canonical(m) != _canonical(np.fliplr(m))
            fallback_keys.add(_canonical(m))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, fingerprints = [], set()
    for p in puzzles:
        validate_generated_puzzle(p)
        fingerprints.add(tuple(_fingerprint(pair['input']) for s in p for pair in p[s]))
        for s in p:
            for pair in p[s]:
                g = pair['input']
                before = g.copy()
                assert np.array_equal(solve(g), pair['output'])
                assert np.array_equal(g, before), 'solve mutated its input'
                # Entire rule is equivariant to rotation/reflection and color relabeling.
                moved = np.fliplr(np.rot90(g))
                target = np.fliplr(np.rot90(pair['output']))
                assert np.array_equal(solve(moved), target)
                perm = np.array([7, 3, 1, 9, 0, 6, 4, 8, 5, 2])
                assert np.array_equal(solve(perm[g]), perm[pair['output']])
                records.append(_evidence(_analyze(g), g))
    assert len(fingerprints) == 42
    assert len({f['shape'] for f in records}) >= 60
    assert len({k for f in records for k in f['motifs']}) >= 30
    assert len({f['objects'] for f in records}) >= 4
    assert len({f['original_rooms'] for f in records}) >= 3
    assert len({f['reached'] for f in records}) >= 60
    assert all({f['roles'][i] for f in records} == set(range(10)) for i in range(3))
    assert max(a for f in records for a in f['areas']) >= 12
    assert any(4 in f['multiplicities'] for f in records)
    assert {1, 2} <= {f['opened_classes'] for f in records}
    assert any(not f['seed_effect'] for f in records)  # Some cascades reach all rooms.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(91+nt*5+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    p = generate_puzzle(0)
    assert tuple(_fingerprint(pair[r]) for s in p for pair in p[s] for r in ('input', 'output')) == tuple(
        _fingerprint(pair[r]) for s in puzzles[0] for pair in puzzles[0][s] for r in ('input', 'output'))
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
