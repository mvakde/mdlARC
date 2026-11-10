"""Directional marker growth over a two-colour substrate."""
import hashlib
from itertools import permutations
import numpy as np

DIR = {'R': (0, 1), 'L': (0, -1), 'T': (-1, 0), 'B': (1, 0)}
SOURCE = {'97f31ee1e7b075923efaef9a555379f2e0a18f3c9d2cf5aa6d9049a9fffba164',
          '28cb667e689f563ee45f484893ed7299be5c5a627a657b55e8c09e3e9ee9ae04',
          '76ff5c7e1ae26c96fcafb8351cf4227815b42d8f165d524c7080d69d761007ed',
          '73931b134fca056a79eebc8f1f155e7a2f0f735a8a3eb6652097693df7382e83'}


def _counts(rng, nt, ns):
    for value, lo, hi in ((nt, 2, 10), (ns, 1, 4)):
        if value is not None and (isinstance(value, bool) or
                                  not isinstance(value, (int, np.integer)) or
                                  not lo <= value <= hi):
            raise ValueError('invalid pair count')
    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(ns + 1, 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError('train count must exceed test count')
    return int(nt), int(ns)


def _core(shape, edge, r, c):
    h, w = shape
    return (0 <= r < h and 0 <= c < w and
            not ((edge == 'R' and c == w - 1) or (edge == 'L' and c == 0) or
                 (edge == 'T' and r == 0) or (edge == 'B' and r == h - 1)))


def _line(g, e):
    if e == 'R': return g[:, -1]
    if e == 'L': return g[:, 0]
    if e == 'T': return g[0, :]
    return g[-1, :]


def _control(g):
    options = []
    for e in DIR:
        line = _line(g, e)
        color = int(line[0])
        if np.all(line == color) and np.count_nonzero(g == color) == len(line):
            options.append((e, color))
    assert len(options) == 1
    return options[0]


def _field(rng, h, w, kind):
    if kind in ('split', 'lobes'):
        vertical = bool(rng.integers(2))
        length, span = (h, w) if vertical else (w, h)
        boundary = np.empty(length, dtype=int)
        boundary[0] = int(rng.integers(max(2, span // 3), min(span - 1, 2 * span // 3 + 1)))
        for k in range(1, length):
            boundary[k] = np.clip(boundary[k - 1] + int(rng.integers(-2, 3)), 2, span - 2)
        y, x = np.indices((h, w))
        f = x < boundary[:, None] if vertical else y < boundary[None, :]
        if rng.integers(2): f = ~f
        if kind == 'lobes':
            for _ in range(int(rng.integers(2, 6))):
                rh = int(rng.integers(2, min(7, h - 1)))
                rw = int(rng.integers(2, min(7, w - 1)))
                r = int(rng.integers(1, h - rh))
                c = int(rng.integers(1, w - rw))
                f[r:r + rh, c:c + rw] = bool(rng.integers(2))
    elif kind == 'pocket':
        f = np.zeros((h, w), dtype=bool)
        side = str(rng.choice(tuple(DIR)))
        band = int(rng.integers(2, max(3, min(h, w) // 4 + 1)))
        if side == 'L': f[:, :band] = True
        if side == 'R': f[:, -band:] = True
        if side == 'T': f[:band, :] = True
        if side == 'B': f[-band:, :] = True
        rh = int(rng.integers(2, min(5, h - 3)))
        rw = int(rng.integers(2, min(5, w - 3)))
        places = [(r, c) for r in range(2, h - rh - 1) for c in range(2, w - rw - 1)
                  if not np.any(f[r - 1:r + rh + 1, c - 1:c + rw + 1])]
        if places:
            r, c = places[int(rng.integers(len(places)))]
            f[r:r + rh, c:c + rw] = True
            if rh > 2 and rw > 2 and rng.integers(2): f[r, c] = False
    else:
        f = np.ones((h, w), dtype=bool)
        centers = [(h // 3, w // 4), (2 * h // 3, 3 * w // 4),
                   (h // 4, 3 * w // 4), (3 * h // 4, w // 4)]
        for r, c in centers[:int(rng.integers(2, 5))]:
            r += int(rng.integers(-1, 2)); c += int(rng.integers(-1, 2))
            rh = int(rng.integers(1, 3)); rw = int(rng.integers(1, 3))
            f[max(1, r - rh):min(h - 1, r + rh + 1),
              max(1, c - rw):min(w - 1, c + rw + 1)] = False
    return f


def _eligible(f, e, p):
    r, c = p
    if not _core(f.shape, e, r, c): return False
    if f[r, c]: return True
    dr, dc = DIR[e]
    nr, nc = r + dr, c + dc
    return _core(f.shape, e, nr, nc) and not f[nr, nc]


def _valid_sites(f, e, sites):
    used = set(sites)
    if len(used) != len(sites): return False
    for r, c in sites:
        if not _eligible(f, e, (r, c)): return False
        neighbors = [bool(f[r + dr, c + dc]) for dr, dc in DIR.values()
                     if _core(f.shape, e, r + dr, c + dc) and
                     (r + dr, c + dc) not in used]
        if len(neighbors) < 2 or any(v != bool(f[r, c]) for v in neighbors):
            return False
        if not f[r, c]:
            dr, dc = DIR[e]
            q = (r + dr, c + dc)
            if q in used and f[q]: return False
    return True


def _motifs(f, e, kind):
    h, w = f.shape
    choices = []
    if kind == 'perimeter':
        for r in range(h):
            for c in range(w):
                if (r in (0, h - 1) or c in (0, w - 1)) and _valid_sites(f, e, [(r, c)]):
                    choices.append([(r, c)])
        return choices
    dr, dc = DIR[e] if kind == 'chain' else ((1, 0) if e in ('R', 'L') else (0, 1))
    for n in (2, 3, 4):
        for r in range(h):
            for c in range(w):
                sites = [(r + k * dr, c + k * dc) for k in range(n)]
                if all(_core(f.shape, e, *p) for p in sites) and (all(not f[p] for p in sites)
                    if kind == 'chain' else all(f[p] == f[sites[0]] for p in sites)) and _valid_sites(f, e, sites):
                    choices.append(sites)
    return choices


def _choose_sites(rng, f, e, count, motif, perimeter, mixed):
    choices = [[]] if motif == 'isolated' else [x for x in _motifs(f, e, motif) if len(x) <= count]
    if not choices: return None
    sites = list(choices[int(rng.integers(len(choices)))])
    if len(sites) > count: return None
    if perimeter and not any(r in (0, f.shape[0] - 1) or c in (0, f.shape[1] - 1) for r, c in sites):
        border = _motifs(f, e, 'perimeter')
        rng.shuffle(border)
        for group in border:
            if len(sites) < count and _valid_sites(f, e, sites + group):
                sites += group
                break
        else: return None
    candidates = [(r, c) for r in range(f.shape[0]) for c in range(f.shape[1])
                  if _eligible(f, e, (r, c)) and (r, c) not in sites]
    rng.shuffle(candidates)
    bias = float(rng.uniform(0.12, 0.88))
    desired = round(count * bias)
    for cell in candidates:
        if len(sites) >= count: break
        if motif == 'isolated' and any(abs(cell[0] - r) + abs(cell[1] - c) == 1
                                       for r, c in sites):
            continue
        active_now = sum(not f[p] for p in sites)
        if not f[cell] and active_now >= desired + 2: continue
        if f[cell] and len(sites) - active_now >= count - desired + 2: continue
        if _valid_sites(f, e, sites + [cell]): sites.append(cell)
    if len(sites) < count: return None
    if mixed and (not any(f[p] for p in sites) or not any(not f[p] for p in sites)):
        return None
    return sites


def _predict(g, active, inactive):
    edge, control = _control(g)
    colors = set(int(v) for v in np.unique(g))
    assert active != inactive and active in colors and inactive in colors and len(colors) == 4
    marker, = colors - {control, active, inactive}
    sites = [(int(r), int(c)) for r, c in np.argwhere(g == marker)]
    assert sites
    used = set(sites)
    latent = {}
    for r, c in sites:
        assert _core(g.shape, edge, r, c)
        neighbors = [int(g[r + dr, c + dc]) for dr, dc in DIR.values()
                     if _core(g.shape, edge, r + dr, c + dc) and
                     (r + dr, c + dc) not in used]
        assert len(neighbors) >= 2 and len(set(neighbors)) == 1
        assert neighbors[0] in (active, inactive)
        latent[r, c] = neighbors[0]
    base = g.copy()
    for p, color in latent.items(): base[p] = color
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if _core(g.shape, edge, r, c): assert base[r, c] in (active, inactive)
    out = g.copy()
    dr, dc = DIR[edge]
    na = ni = 0
    for (r, c), color in latent.items():
        if color == inactive:
            out[r, c] = inactive; ni += 1
        else:
            q = (r + dr, c + dc)
            assert _core(g.shape, edge, *q) and base[q] == active
            out[q] = marker; na += 1
    assert not np.array_equal(out, g)
    return out, {'edge': edge, 'control': control, 'marker': marker, 'active': na,
                 'inactive': ni, 'sites': sites, 'field': base == inactive}


def _pair(rng, active, inactive, control, marker, edge, kind, target_count, motif,
          perimeter, mixed, size_hint):
    for _ in range(70):
        if size_hint is None:
            h = int(rng.integers(9, 31)); w = int(rng.integers(9, 31))
        else:
            h, w = size_hint
        if kind in ('pocket', 'islands'): h = max(h, 14); w = max(w, 14)
        f = _field(rng, h, w, kind)
        area = h * w - (h if edge in ('R', 'L') else w)
        count = min(target_count, max(3, int(area * 0.17)))
        sites = _choose_sites(rng, f, edge, count, motif, perimeter, mixed)
        if sites is None: continue
        g = np.where(f, inactive, active).astype(np.int8)
        _line(g, edge)[:] = control
        for p in sites: g[p] = marker
        try: out, _ = _predict(g, active, inactive)
        except AssertionError: continue
        return {'input': g, 'output': out}
    raise RuntimeError('could not construct recoverable pair')


def _roles(train):
    _, control = _control(train[0]['input'])
    colors = set(int(v) for v in np.unique(train[0]['input'])) - {control}
    valid = []
    for active, inactive in permutations(colors, 2):
        try:
            stats = []
            for pair in train:
                pred, rec = _predict(pair['input'], active, inactive)
                assert np.array_equal(pred, pair['output'])
                stats.append(rec)
            assert sum(s['active'] for s in stats) and sum(s['inactive'] for s in stats)
        except AssertionError: continue
        valid.append((active, inactive))
    return valid


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _counts(rng, num_train, num_test)
    active, inactive = (int(x) for x in rng.choice(10, size=2, replace=False))
    others = [c for c in range(10) if c not in (active, inactive)]
    fixed_control, fixed_marker = (int(x) for x in rng.choice(others, size=2, replace=False))
    palette_mode = int(rng.integers(2))
    orientation_mode = int(rng.integers(4))
    edges = tuple(DIR)
    first = str(rng.choice(edges))
    if orientation_mode == 0:
        train_edges = [first] * nt; test_edges = [first] * ns
    else:
        second = str(rng.choice([e for e in edges if e != first]))
        train_edges = [first, second] + [str(rng.choice(edges)) for _ in range(nt - 2)]
        if orientation_mode == 1:
            test_edges = [str(rng.choice(train_edges)) for _ in range(ns)]
        elif orientation_mode == 2:
            unseen = [e for e in edges if e not in train_edges]
            test_edges = [str(rng.choice(unseen or edges)) for _ in range(ns)]
        else:
            test_edges = [str(rng.choice(edges)) for _ in range(ns)]
    kinds = ['split', 'pocket', 'islands', 'lobes']
    rng.shuffle(kinds)
    puzzle = {'train': [], 'test': []}
    for split, count, orientations in (('train', nt, train_edges), ('test', ns, test_edges)):
        for index in range(count):
            prng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            control, marker = ((fixed_control, fixed_marker) if palette_mode == 0 else
                               tuple(int(x) for x in prng.choice(others, size=2, replace=False)))
            kind = kinds[index % 4] if split == 'train' else str(prng.choice(kinds))
            density_band = int(prng.integers(3))
            target_count = (int(prng.integers(3, 7)) if density_band == 0 else
                            int(prng.integers(7, 15)) if density_band == 1 else
                            int(prng.integers(15, 30)))
            motif = ('chain' if index == 0 else
                     str(prng.choice(('run', 'isolated', 'perimeter'))) if index == 1 else
                     str(prng.choice(('chain', 'run', 'perimeter', 'isolated')))) if split == 'train' else str(prng.choice(('chain', 'run', 'perimeter', 'isolated')))
            size_hint = None
            if split == 'test' and prng.integers(3) == 0:
                source_pair = puzzle['train'][int(prng.integers(len(puzzle['train'])))]
                size_hint = source_pair['input'].shape
            pair = _pair(prng, active, inactive, control, marker, orientations[index],
                         kind, target_count, motif, (split == 'train' and index == 1) or
                         (split == 'test' and bool(prng.integers(2))), split == 'train',
                         size_hint)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(g):
    return g.shape, g.tobytes()


def _parts(mask):
    seen = set()
    parts = []
    h, w = mask.shape
    for row, col in np.argwhere(mask):
        start = (int(row), int(col))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        part = []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in DIR.values():
                q = (r + dr, c + dc)
                if (0 <= q[0] < h and 0 <= q[1] < w and mask[q] and q not in seen):
                    seen.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for g in pair.values():
            assert isinstance(g, np.ndarray) and g.ndim == 2
            assert np.issubdtype(g.dtype, np.integer)
            assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
        inp, out = pair['input'], pair['output']
        assert inp.shape == out.shape and not np.array_equal(inp, out)
        key = _signature(inp)
        assert key not in seen
        seen.add(key)
        digest = hashlib.sha256(bytes(inp.shape) + inp.astype(np.int8).tobytes() +
                                out.astype(np.int8).tobytes()).hexdigest()
        assert digest not in SOURCE
    candidates = _roles(puzzle['train'])
    assert len(candidates) == 1
    active, inactive = candidates[0]
    train_records = []
    for pair in puzzle['train']:
        pred, rec = _predict(pair['input'], active, inactive)
        assert np.array_equal(pred, pair['output'])
        assert rec['active'] and rec['inactive']
        train_records.append(rec)
    assert any(any(r in (0, pair['input'].shape[0] - 1) or c in (0, pair['input'].shape[1] - 1)
                   for r, c in rec['sites']) for pair, rec in zip(puzzle['train'], train_records))
    assert any(any((r + DIR[rec['edge']][0], c + DIR[rec['edge']][1]) in rec['sites']
                   for r, c in rec['sites']) for rec in train_records)
    for pair in puzzle['test']:
        pred, _ = _predict(pair['input'], active, inactive)
        assert np.array_equal(pred, pair['output'])


def self_test() -> None:
    def sig(p):
        return tuple((_signature(x['input']), _signature(x['output'])) for x in p['train'] + p['test'])
    trains, tests, shapes, edges, palettes, topologies, counts = (set() for _ in range(7))
    perimeter = touching = chains = 0
    active_perimeter = inactive_perimeter = combined = 0
    pockets = disconnected = sideways = 0
    sparse = {'train': 0, 'test': 0}
    dense = {'train': 0, 'test': 0}
    isolated = {'train': 0, 'test': 0}
    constant_palette = varied_control = varied_marker = 0
    single_edge = reused_edge = unseen_edge = shared_shape = 0
    test_not_denser = test_not_more_markers = 0
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(sig(p)); trains.add(len(p['train'])); tests.add(len(p['test']))
        active, inactive = _roles(p['train'])[0]
        palettes.add((active, inactive))
        records = []
        for split, pair in [('train', x) for x in p['train']] + [('test', x) for x in p['test']]:
            g = pair['input']
            pred, rec = _predict(g, active, inactive)
            assert np.array_equal(pred, pair['output'])
            records.append((split, rec, g))
            shapes.add(g.shape); edges.add(rec['edge']); counts.add(len(rec['sites']))
            f = rec['field']
            topologies.add((int(f.sum()), int(np.count_nonzero(f[:, 1:] != f[:, :-1]) +
                                            np.count_nonzero(f[1:, :] != f[:-1, :]))))
            sites = set(rec['sites'])
            border_sites = {site for site in sites if site[0] in (0, g.shape[0] - 1)
                            or site[1] in (0, g.shape[1] - 1)}
            perimeter += bool(border_sites)
            active_perimeter += any(not f[site] for site in border_sites)
            inactive_perimeter += any(f[site] for site in border_sites)
            adjacent = any((r + dr, c + dc) in sites for r, c in sites for dr, dc in DIR.values())
            touching += adjacent
            isolated[split] += not adjacent
            dr, dc = DIR[rec['edge']]
            chains += any((r + dr, c + dc) in sites for r, c in sites)
            pdr, pdc = (1, 0) if rec['edge'] in ('R', 'L') else (0, 1)
            sideways += any(all((r + k * pdr, c + k * pdc) in sites for k in range(3))
                            for r, c in sites)
            core = (f[:, 1:] if rec['edge'] == 'L' else f[:, :-1] if rec['edge'] == 'R'
                    else f[1:, :] if rec['edge'] == 'T' else f[:-1, :])
            has_pocket = any(all(r not in (0, core.shape[0] - 1) and
                                 c not in (0, core.shape[1] - 1) for r, c in part)
                             for part in _parts(core))
            has_disconnected_active = len(_parts(~core)) > 1
            pockets += has_pocket
            disconnected += has_disconnected_active
            combined += (has_pocket or has_disconnected_active) and adjacent and bool(border_sites)
            sparse[split] += len(sites) <= 6
            dense[split] += len(sites) >= 16
        controls = {rec['control'] for _, rec, _ in records}
        markers = {rec['marker'] for _, rec, _ in records}
        constant_palette += len(controls) == len(markers) == 1
        varied_control += len(controls) > 1
        varied_marker += len(markers) > 1
        train_edges = {rec['edge'] for split, rec, _ in records if split == 'train'}
        test_edges = {rec['edge'] for split, rec, _ in records if split == 'test'}
        train_shapes = {g.shape for split, _, g in records if split == 'train'}
        test_shapes = {g.shape for split, _, g in records if split == 'test'}
        single_edge += len(train_edges | test_edges) == 1
        reused_edge += bool(train_edges & test_edges)
        unseen_edge += bool(test_edges - train_edges)
        shared_shape += bool(train_shapes & test_shapes)
        train_stats = [(len(rec['sites']), len(rec['sites']) / (g.size - len(_line(g, rec['edge']))))
                       for split, rec, g in records if split == 'train']
        test_stats = [(len(rec['sites']), len(rec['sites']) / (g.size - len(_line(g, rec['edge']))))
                      for split, rec, g in records if split == 'test']
        test_not_more_markers += any(n <= max(x[0] for x in train_stats) for n, _ in test_stats)
        test_not_denser += any(d <= max(x[1] for x in train_stats) for _, d in test_stats)
    assert len(trains) >= 4 and len(tests) >= 3
    assert len(shapes) >= 50 and len(edges) == 4 and len(palettes) >= 20
    assert min(h for h, _ in shapes) <= 10 and max(h for h, _ in shapes) >= 28
    assert min(w for _, w in shapes) <= 10 and max(w for _, w in shapes) >= 28
    assert sum(h != w for h, w in shapes) >= 30
    assert len(topologies) >= 80 and len(counts) >= 12
    assert len(signatures) == 42 and perimeter >= 30 and touching >= 30 and chains >= 30
    assert active_perimeter >= 10 and inactive_perimeter >= 10 and combined >= 10
    assert pockets >= 10 and disconnected >= 10 and sideways >= 10
    assert all(sparse[s] >= 5 and dense[s] >= 5 and isolated[s] >= 3
               for s in ('train', 'test'))
    assert constant_palette >= 5 and varied_control >= 5 and varied_marker >= 5
    assert single_edge >= 3 and reused_edge >= 5 and unseen_edge >= 3
    assert shared_shape >= 5
    assert test_not_denser >= 5 and test_not_more_markers >= 5
    for seed, request in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                    (None, 4), (2, None), (10, None))):
        p = generate_puzzle(1000 + seed, *request)
        validate_generated_puzzle(p)
        if request[0] is not None: assert len(p['train']) == request[0]
        if request[1] is not None: assert len(p['test']) == request[1]
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2, 4)):
        try: generate_puzzle(2000, *request)
        except ValueError: pass
        else: raise AssertionError('illegal counts accepted')
    assert sig(generate_puzzle(713)) == sig(generate_puzzle(713))
    assert sig(generate_puzzle(714, 6, 3)) == sig(generate_puzzle(714, 6, 3))
