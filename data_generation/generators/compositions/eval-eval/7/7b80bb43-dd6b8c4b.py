"""Demand-driven restoration of a rail tree feeding a nine-slot collector.

Only stdlib and NumPy are used. All conventions are visible in one input;
generation metadata is never consulted by solve().
"""
from collections import deque
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(value, message):
    if not bool(value):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, and train > test.")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions.")
    _require(np.issubdtype(a.dtype, np.integer) and ((a >= 0) & (a <= 9)).all(), "Invalid colors/dtype.")
    return a.astype(np.int8, copy=False)


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _station_mask(shape, pos):
    m = np.zeros(shape, dtype=bool)
    r, c = pos
    m[r:r+3, c:c+3] = True
    return m


def _roles(g):
    values, counts = np.unique(g, return_counts=True)
    candidates = []
    for color, count in zip(values, counts):
        if count != 8:
            continue
        rr, cc = np.where(g == color)
        r, c = int(rr.min()), int(cc.min())
        if rr.max() != r+2 or cc.max() != c+2:
            continue
        patch = g[r:r+3, c:c+3]
        marker = int(patch[1, 1])
        if marker != color and np.count_nonzero(g == marker) == 1:
            candidates.append(((r, c), int(color), marker))
    _require(len(candidates) == 1, "Need one pristine marked 3x3 collector.")
    pos, slot, marker = candidates[0]
    _require(len(values) == 5, "Five distinct visible color roles are required.")
    bg = int(values[np.argmax(counts)])
    _require(int(np.count_nonzero(g == bg)) > g.size // 2, "Background must occupy a strict majority.")
    station = _station_mask(g.shape, pos)
    border = np.zeros(g.shape, dtype=bool)
    border[1:] |= station[:-1]
    border[:-1] |= station[1:]
    border[:, 1:] |= station[:, :-1]
    border[:, :-1] |= station[:, 1:]
    border &= ~station
    ports = set(map(int, np.unique(g[border]))) - {bg}
    _require(len(ports) == 1, "Only the rail color may touch the collector outside it.")
    rail = next(iter(ports))
    payloads = set(map(int, values)) - {bg, rail, slot, marker}
    _require(len(payloads) == 1 and len({bg, rail, slot, marker}) == 4, "Ambiguous palette.")
    payload = next(iter(payloads))
    return pos, bg, rail, payload, slot, marker


def _strong(mask):
    """Membership in an intact horizontal/vertical run of at least three."""
    result = np.zeros_like(mask)
    for m, s in ((mask, result), (mask.T, result.T)):
        triples = m[:, :-2] & m[:, 1:-1] & m[:, 2:]
        for offset in range(3):
            s[:, offset:offset+triples.shape[1]] |= triples
    return result


def _gaps(mask):
    result = []
    for transpose in (False, True):
        m = mask.T if transpose else mask
        brackets = m[:, :-4] & m[:, 4:]
        brackets &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(brackets):
            r, c, dr, dc = (int(x), int(y), 1, 0) if transpose else (int(y), int(x), 0, 1)
            result.append((r, c, dr, dc))
    return result


def _virtual_tracks(raw, rail_cells, station):
    """Infer virtual straight bridges and erase oblique noise for route finding.

    Return explicit cues separately: this virtual mask is not the final output.
    An unused cue must survive physically in the output.
    """
    strong = _strong(raw)
    cues, additions, anchors = [], np.zeros_like(raw), np.zeros_like(raw)
    for r, c, dr, dc in _gaps(raw):
        gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
        if any(station[cell] for cell in gap):
            continue
        if not (_at(raw, r-dr, c-dc) or _at(raw, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            if not rail_cells[ar, ac]:
                continue
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr+side*dc, toward*dc+side*dr
                spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if (all(_at(rail_cells, rr, cc) and not strong[rr, cc] for rr, cc in spur)
                        and not _at(raw, ar+3*sr, ac+3*sc)):
                    matches.append(spur)
        if matches:
            _require(len(matches) == 1, "Multiple spur interpretations are excluded.")
            cues.append((gap, matches[0]))
            for cell in gap:
                additions[cell] = True
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
    weak = rail_cells & ~_strong(raw | additions) & ~anchors
    diagonal = np.zeros_like(raw)
    for sign in (1, -1):
        a = weak[:-1, :-1] if sign == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if sign == 1 else weak[1:, :-1]
        linked = a & b
        if sign == 1:
            diagonal[:-1, :-1] |= linked
            diagonal[1:, 1:] |= linked
        else:
            diagonal[:-1, 1:] |= linked
            diagonal[1:, :-1] |= linked
    spurs = np.zeros_like(raw)
    for _, spur in cues:
        for cell in spur:
            spurs[cell] = True
    removed = diagonal | spurs
    return (raw & ~removed) | additions, cues, removed & ~spurs


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        first = min(cells)
        cells.remove(first)
        component, queue = [], [first]
        for r, c in queue:
            component.append((r, c))
            for dr, dc in STEPS:
                nxt = r+dr, c+dc
                if nxt in cells:
                    cells.remove(nxt)
                    queue.append(nxt)
        result.append(component)
    return result


def _distances(track, station):
    """The station footprint is a single root; only positive tracks are traversable."""
    distance = np.full(track.shape, -1, dtype=np.int16)
    parent = np.full(track.shape, -1, dtype=np.int32)
    distance[station] = 0
    h, w = track.shape
    queue = deque(map(tuple, np.argwhere(station).tolist()))
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and track[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c]+1
                parent[nr, nc] = r*w+c
                queue.append((nr, nc))
    return distance, parent


def _forest_check(track, station):
    for comp in _components(track):
        nodes = set(comp)
        edges = sum((r+1, c) in nodes for r, c in comp) + sum((r, c+1) in nodes for r, c in comp)
        _require(edges == len(comp)-1, "Virtual rail components must be trees.")
        contacts = sum(_at(station, r+dr, c+dc) for r, c in comp for dr, dc in STEPS)
        _require(contacts <= 1, "A rail tree may touch the collector at only one port.")


def _select(payload_mask, distance):
    reachable = payload_mask & (distance >= 0)
    ordered = np.sort(distance[reachable])
    if len(ordered) > 9:
        _require(ordered[8] < ordered[9], "A tie across the ninth/tenth frontier is excluded.")
    threshold = int(ordered[min(8, len(ordered)-1)]) if len(ordered) else -1
    return reachable & (distance <= threshold)


def _route_union(selected, distance, parent):
    routes = np.zeros_like(selected)
    for start in np.flatnonzero(selected):
        cell = int(start)
        while distance.flat[cell] > 0 and not routes.flat[cell]:
            routes.flat[cell] = True
            cell = int(parent.flat[cell])
            _require(cell >= 0, "Broken route ancestry.")
    return routes


def _analyze(input_grid):
    g = _grid(input_grid)
    pos, bg, rail, payload, slot, marker = _roles(g)
    station = _station_mask(g.shape, pos)
    raw = (g == rail) | (g == payload)
    track, cues, noise = _virtual_tracks(raw, g == rail, station)
    _forest_check(track, station)
    distance, parent = _distances(track, station)
    selected = _select(g == payload, distance)
    routes = _route_union(selected, distance, parent)
    used = [any(routes[cell] for cell in gap) for gap, _ in cues]
    return dict(g=g, pos=pos, bg=bg, rail=rail, payload=payload, slot=slot, marker=marker,
                station=station, track=track, cues=cues, noise=noise, distance=distance,
                parent=parent, selected=selected, routes=routes, used=used)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    out = info['g'].copy()
    out[info['noise']] = info['bg']
    for (gap, spur), used in zip(info['cues'], info['used']):
        if used:
            for cell in gap:
                out[cell] = info['rail']
            for cell in spur:
                out[cell] = info['bg']
    out[info['selected']] = info['rail']
    r, c = info['pos']
    for index in range(int(info['selected'].sum())):
        out[r+index//3, c+index%3] = info['payload']
    return out


def _halo(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[r:r+h, c:c+w] for r in range(3) for c in range(3)])


def _tree(rng, shape, rich):
    h, w = shape
    for _ in range(40):
        pos = (int(rng.integers(1, h-3)), int(rng.integers(1, w-3)))
        r, c = pos
        ports = [((r-1, c+1), (-1, 0), r), ((r+3, c+1), (1, 0), h-r-3),
                 ((r+1, c-1), (0, -1), c), ((r+1, c+3), (0, 1), w-c-3)]
        eligible = [p for p in ports if p[2] >= 6]
        if len(eligible) >= 2:
            break
    station = _station_mask(shape, pos)
    track = np.zeros(shape, dtype=bool)
    number = int(rng.integers(2, min(3, len(eligible))+1))
    for index in rng.choice(len(eligible), number, replace=False):
        (rr, cc), (dr, dc), limit = eligible[int(index)]
        length = limit if rng.random() < .25 else int(rng.integers(6, limit+1))
        for k in range(length):
            track[rr+k*dr, cc+k*dc] = True
    desired = int(rng.integers(6, 15) if rich else rng.integers(3, 10))
    near_station = np.zeros(shape, dtype=bool)
    for dr, dc in STEPS:
        for rr, cc in np.argwhere(station):
            nr, nc = int(rr)+dr, int(cc)+dc
            if 0 <= nr < h and 0 <= nc < w:
                near_station[nr, nc] = True
    grown = 0
    for _ in range(150):
        if grown >= desired:
            break
        r, c = map(int, rng.choice(np.argwhere(track)))
        dr, dc = STEPS[int(rng.integers(4))]
        cells = []
        for k in range(1, 31):
            rr, cc = r+k*dr, c+k*dc
            if not (0 <= rr < h and 0 <= cc < w) or track[rr, cc] or near_station[rr, cc] or station[rr, cc]:
                break
            if any(_at(track, rr+ar, cc+ac) and (rr+ar, cc+ac) != (r, c) for ar, ac in STEPS):
                break
            cells.append((rr, cc))
        if len(cells) < 2:
            continue
        length = len(cells) if rng.random() < .2 else int(rng.integers(2, min(len(cells), 9)+1))
        for cell in cells[:length]:
            track[cell] = True
        grown += 1
    return track, station, pos


def _island(rng, track, station, broken=False):
    """An independent rail piece, optionally with an intentional unevidenced break."""
    forbidden = _halo(track | station)
    options = []
    lengths = range(9, 14) if broken else range(4, 11)
    h, w = track.shape
    for dr, dc in ((1, 0), (0, 1)):
        for r, c in np.argwhere(~forbidden):
            r, c = int(r), int(c)
            for length in lengths:
                cells = [(r+k*dr, c+k*dc) for k in range(length)]
                if all(0 <= rr < h and 0 <= cc < w and not forbidden[rr, cc] for rr, cc in cells):
                    options.append(cells)
    if not options:
        return None
    cells = options[int(rng.integers(len(options)))]
    result = track.copy()
    if broken:
        start = int(rng.integers(3, len(cells)-5)) if len(cells) > 9 else 3
        cells = cells[:start] + cells[start+3:]
    for cell in cells:
        result[cell] = True
    return result


def _damage_candidates(clean, main):
    options = []
    for dr, dc in ((0, 1), (1, 0)):
        for r, c in np.argwhere(main):
            r, c = int(r), int(c)
            if not all(_at(main, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(main, r-dr, c-dc) or _at(main, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(_at(clean, rr+dc, cc+dr) or _at(clean, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for endpoint in (0, 4):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                toward = 1 if endpoint == 0 else -1
                for side in (-1, 1):
                    sr, sc = toward*dr+side*dc, toward*dc+side*dr
                    spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                    if all(0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1] and not clean[rr, cc] for rr, cc in spur):
                        options.append((gap, spur))
    return options


def _damage(rng, clean, main, station, target):
    raw = clean.copy()
    options = _damage_candidates(clean, main)
    occupied = np.zeros_like(clean)
    count = 0
    for index in rng.permutation(len(options)):
        gap, spur = options[int(index)]
        footprint = np.zeros_like(clean)
        for cell in gap+spur:
            footprint[cell] = True
        if (footprint & _halo(occupied | station)).any():
            continue
        trial = raw.copy()
        for cell in gap:
            trial[cell] = False
        for cell in spur:
            trial[cell] = True
        try:
            virtual, cues, noise = _virtual_tracks(trial, trial, station)
        except AssertionError:
            continue
        if len(cues) == count+1 and not noise.any() and np.array_equal(virtual, clean):
            raw = trial
            occupied |= footprint
            count += 1
            if count == target:
                break
    virtual, cues, _ = _virtual_tracks(raw, raw, station)
    return raw, cues


def _descendants(gap, distance, parent):
    descendants = np.zeros(distance.shape, dtype=bool)
    for cell in gap:
        descendants[cell] = True
    cells = np.flatnonzero(distance > 0)
    for cell in cells[np.argsort(distance.flat[cells], kind='stable')]:
        p = int(parent.flat[int(cell)])
        if p >= 0 and descendants.flat[p]:
            descendants.flat[int(cell)] = True
    return descendants


def _pick(rng, available, count, style, forced=()):
    pool = set(map(int, np.flatnonzero(available)))
    chosen = list(dict.fromkeys(map(int, forced)))
    if not set(chosen) <= pool or not len(chosen) <= count <= len(pool):
        return None
    pool.difference_update(chosen)
    h, w = available.shape
    anchor = chosen[0] if chosen else int(rng.choice(sorted(pool))) if pool else 0
    ar, ac = divmod(anchor, w)
    axis = bool(rng.integers(2))
    while len(chosen) < count:
        cells = np.array(sorted(pool), dtype=int)
        if style == 'scatter':
            cell = int(rng.choice(cells))
        else:
            rr, cc = cells//w, cells%w
            score = ((rr-ar)**2+(cc-ac)**2 if style == 'cluster'
                     else abs(rr-ar)*w+abs(cc-ac) if axis else abs(cc-ac)*h+abs(rr-ar))
            cell = int(rng.choice(cells[score == score.min()]))
        pool.remove(cell)
        chosen.append(cell)
    return chosen


def _supply(rng, clean, main, station, cues, pos, load, spare, detour):
    distance, parent = _distances(clean, station)
    pool = main & (distance >= 2)
    for gap, spur in cues:
        for cell in gap+spur:
            pool[cell] = False
        # Payload never obscures the retained brackets used as repair evidence.
        r0, c0 = gap[0]
        r1, c1 = gap[-1]
        dr, dc = (r1-r0)//2, (c1-c0)//2
        pool[r0-dr, c0-dc] = pool[r1+dr, c1+dc] = False
    desc = [_descendants(gap, distance, parent) for gap, _ in cues]
    rr, cc = np.indices(clean.shape)
    r, c = pos
    direct = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
    style = str(rng.choice(['scatter', 'cluster', 'line']))
    spare_options = list(rng.permutation(len(cues))) if spare else [None]
    for unused in spare_options:
        allowed = pool.copy() if unused is None else pool & ~desc[int(unused)]
        for used in rng.permutation(len(cues)):
            used = int(used)
            if used == unused:
                continue
            beyond = pool & allowed & desc[used]
            for a in rng.permutation(np.flatnonzero(beyond)):
                a = int(a)
                if load <= 9:
                    chosen = _pick(rng, allowed, load, style, [a])
                    if chosen is not None:
                        return chosen, distance, parent
                else:
                    da = int(distance.flat[a])
                    near = allowed & (distance <= da)
                    far = pool & (distance > da)
                    eligible = far & (direct < direct.flat[a]) if detour else far
                    if near.sum() < 9 or far.sum() < load-9 or not eligible.any():
                        continue
                    b = int(rng.choice(np.flatnonzero(eligible)))
                    selected = _pick(rng, near, 9, style, [a])
                    retained = _pick(rng, far, load-9, style, [b])
                    return selected+retained, distance, parent
    return None


def _noise(rng, raw, clean, station, cues):
    result = raw.copy()
    removed = np.zeros_like(raw)
    occupied = clean | raw | station
    for gap, _ in cues:
        for cell in gap:
            occupied[cell] = True
    wanted = int(rng.integers(1, 4))
    for _ in range(80):
        if len(np.flatnonzero(removed)) >= wanted*2:
            break
        r, c = int(rng.integers(raw.shape[0])), int(rng.integers(raw.shape[1]))
        dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        length = int(rng.choice([2, 3]))
        cells = [(r+k*dr, c+k*dc) for k in range(length)]
        forbidden = _halo(occupied)
        if not all(0 <= rr < raw.shape[0] and 0 <= cc < raw.shape[1] and not forbidden[rr, cc] for rr, cc in cells):
            continue
        trial = result.copy()
        for cell in cells:
            trial[cell] = True
        try:
            virtual, inferred, noise = _virtual_tracks(trial, trial, station)
        except AssertionError:
            continue
        if np.array_equal(virtual, clean) and inferred == cues:
            result = trial
            for cell in cells:
                removed[cell] = occupied[cell] = True
    return result, removed


def _make_pair(rng, palette, spec, test):
    for _ in range(140):
        shape = tuple(map(int, rng.integers(20 if test else 16, 31 if test else 28, size=2)))
        if rng.random() < .2:
            shape = (shape[0], shape[0])
        main, station, pos = _tree(rng, shape, test or spec['detour'])
        clean = _island(rng, main, station, broken=spec['contrast'])
        if clean is None:
            continue
        if rng.random() < .45:
            extra = _island(rng, clean, station, broken=False)
            if extra is not None:
                clean = extra
        target = int(rng.integers(3, 6) if test else rng.integers(2, 4))
        raw, cues = _damage(rng, clean, main, station, target)
        if len(cues) < (3 if test else 2):
            continue
        supply = _supply(rng, clean, main, station, cues, pos, spec['load'], spec['spare'], spec['detour'])
        if supply is None:
            continue
        tokens, distance, parent = supply
        inactive = clean & ~main
        distraction = _pick(rng, inactive, int(rng.integers(1, min(7, int(inactive.sum()))+1)),
                            str(rng.choice(['scatter', 'cluster', 'line'])))
        if distraction is None:
            continue
        raw, noise = _noise(rng, raw, clean, station, cues)
        if not noise.any():
            continue
        bg, rail, payload, slot, marker = palette
        g = np.full(shape, bg, dtype=np.int8)
        g[raw] = rail
        g.flat[tokens+distraction] = payload
        r, c = pos
        g[r:r+3, c:c+3] = slot
        g[r+1, c+1] = marker
        try:
            info = _analyze(g)
        except AssertionError:
            continue
        if not np.array_equal(info['track'], clean) or info['cues'] != cues or not np.array_equal(info['noise'], noise):
            continue
        # Construct the expected label from the independently grown clean graph,
        # then commit only its used corruption sites. No call to solve here.
        selected = _select(g == payload, distance)
        routes = _route_union(selected, distance, parent)
        used = [any(routes[cell] for cell in gap) for gap, _ in cues]
        if not any(used) or (spec['spare'] and all(used)):
            continue
        out = g.copy()
        out[noise] = bg
        for (gap, spur), commit in zip(cues, used):
            if commit:
                for cell in gap:
                    out[cell] = rail
                for cell in spur:
                    out[cell] = bg
        out[selected] = rail
        for index in range(int(selected.sum())):
            out[r+index//3, c+index%3] = payload
        return {'input': g, 'output': out}
    raise AssertionError("Bounded rail construction could not meet the evidence request.")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    n, t = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(10, size=5, replace=False)))
    vary_palette = bool(rng.random() < .65)
    specs = [dict(load=int(rng.choice([2, 4])), spare=True, detour=False, contrast=True),
             dict(load=int(rng.integers(10, 19)), spare=True, detour=True, contrast=False)]
    def random_spec():
        return dict(load=int(rng.integers(10, 19) if rng.random() < .4 else rng.integers(1, 10)),
                    spare=bool(rng.random() < .55), detour=False, contrast=bool(rng.random() < .4))
    specs.extend(random_spec() for _ in range(n-2))
    specs = [specs[int(i)] for i in rng.permutation(n)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, requests in (('train', specs), ('test', [random_spec() for _ in range(t)])):
        for spec in requests:
            for _ in range(12):
                palette = tuple(map(int, rng.choice(10, size=5, replace=False))) if vary_palette else base_palette
                pair = _make_pair(rng, palette, spec, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Duplicate inputs after bounded retries.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(g):
    return g.shape, g.tobytes()


def pair_features(input_grid):
    a = _analyze(input_grid)
    g, d, selected = a['g'], a['distance'], a['selected']
    reachable = (g == a['payload']) & (d >= 0)
    retained = reachable & ~selected
    rr, cc = np.indices(g.shape)
    r, c = a['pos']
    direct = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
    degree = np.zeros(g.shape, dtype=int)
    tr = a['track']
    degree[1:] += tr[:-1]
    degree[:-1] += tr[1:]
    degree[:, 1:] += tr[:, :-1]
    degree[:, :-1] += tr[:, 1:]
    return dict(shape=g.shape, palette=tuple(a[k] for k in ('bg', 'rail', 'payload', 'slot', 'marker')),
                collector=a['pos'], potential_bridges=len(a['cues']), used_bridges=sum(a['used']),
                unused_reachable=sum(not use and any(d[cell] >= 0 for cell in gap)
                                     for (gap, _), use in zip(a['cues'], a['used'])),
                axes=tuple(sorted({'h' if gap[0][0] == gap[2][0] else 'v' for gap, _ in a['cues']})),
                selected=int(selected.sum()), reachable=int(reachable.sum()),
                excess=int(retained.sum()), unreachable=int(((g == a['payload']) & (d < 0)).sum()),
                frontier_gap=int(d[retained].min()-d[selected].max()) if retained.any() else 0,
                path_beats_manhattan=bool(retained.any() and direct[selected].max() > direct[retained].min()),
                max_distance=int(d.max()), branches=int((degree >= 3).sum()),
                ports=int(sum(_at(a['station'], r+dr, c+dc) for r, c in np.argwhere(tr) for dr, dc in STEPS)),
                components=len(_components(tr)), intentional_gaps=len(_gaps(tr)),
                noise_cells=int(a['noise'].sum()),
                raw_selection_differs=bool(not np.array_equal(selected,
                    _select_without_tie(g == a['payload'], _distances((g == a['rail']) | (g == a['payload']), a['station'])[0]))))


def _select_without_tie(payload, distance):
    """Ablation metric only: compare the first nine raw-track candidates."""
    indices = np.flatnonzero(payload & (distance >= 0))
    chosen = sorted(map(int, indices), key=lambda i: (int(distance.flat[i]), i))[:9]
    mask = np.zeros(payload.shape, dtype=bool)
    mask.flat[chosen] = True
    return mask


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair schema.")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "Dimensions must be preserved.")
            _require(_key(inp) not in seen, "Duplicate input.")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(solve(inp), out), "Incorrect restoration/selection/serialization label.")
            f = pair_features(inp)
            stats[split].append(f)
            _require(f['used_bridges'] >= 1 and f['unreachable'] >= 1, "Both restoration and reachability must matter.")
            _require(f['noise_cells'] >= 2, "Need oblique clutter to contrast with an unused cue.")
            _require(f['raw_selection_differs'], "Ignoring virtual bridges must change collection.")
            _require(f['reachable'] >= f['selected'] >= 1 and f['selected'] <= 9, "Invalid load.")
            _require(f['excess'] == 0 or f['selected'] == 9 and f['frontier_gap'] > 0, "Capacity must be unique.")
            # Exact edits, including preserved unused cues and off-network tokens.
            for (gap, spur), used in zip(a['cues'], a['used']):
                _require(all(out[cell] == (a['rail'] if used else a['bg']) for cell in gap), "Wrong bridge state.")
                _require(all(out[cell] == (a['bg'] if used else a['rail']) for cell in spur), "Wrong spur state.")
            _require((out[(inp == a['payload']) & ~a['selected']] == a['payload']).all(), "Uncollected payload changed.")
            _require((out[a['selected']] == a['rail']).all(), "Collection must restore underlying rail.")
    train = stats['train']
    _require(any(f['selected'] in (2, 4) and f['unused_reachable'] and f['intentional_gaps'] for f in train),
             "Need partial prefix plus intentional-gap and unused-bridge contrast.")
    _require(any(f['excess'] and f['path_beats_manhattan'] and f['unused_reachable'] for f in train),
             "Need an excess/detour/unused-bridge demonstration.")


def self_test() -> dict:
    """Bounded 42-seed sweep, family metrics, determinism, and negative labels."""
    # No arbitrary coordinate or queue-order convention may resolve capacity.
    try:
        _select(np.ones((3, 4), dtype=bool), np.ones((3, 4), dtype=np.int16))
    except AssertionError:
        pass
    else:
        raise AssertionError("A tie straddling capacity was accepted.")
    loop = np.zeros((7, 7), dtype=bool)
    loop[1:3, 1:3] = True
    try:
        _forest_check(loop, _station_mask(loop.shape, (4, 4)))
    except AssertionError:
        pass
    else:
        raise AssertionError("An ambiguous cyclic route was accepted.")
    signatures, features, puzzles = set(), [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        puzzles.append(puzzle)
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                features.append(pair_features(pair['input']))
    _require(len(signatures) == 42, "Seed diversity collapsed.")
    for field, minimum in [('shape', 40), ('palette', 30), ('collector', 30), ('selected', 8),
                           ('reachable', 12), ('max_distance', 20), ('branches', 6), ('potential_bridges', 4)]:
        _require(len({f[field] for f in features}) >= minimum, "Insufficient diversity: "+field)
    _require({axis for f in features for axis in f['axes']} == {'h', 'v'}, "Both axes must occur.")
    _require(any(len(f['axes']) == 2 for f in features), "Mixed-axis scenes absent.")
    _require(any(f['used_bridges'] >= 2 for f in features), "Multi-bridge collection absent.")
    _require(any(f['ports'] >= 3 for f in features), "Multiple entrances absent.")
    _require(any(f['used_bridges'] == f['potential_bridges'] for f in features), "All-used cases absent.")
    _require(any(f['noise_cells'] >= 3 for f in features), "Longer free diagonals absent.")
    for role in range(5):
        _require(len({f['palette'][role] for f in features}) == 10, "A role lacks palette variation.")
    again = generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                 for p, q in zip(puzzles[0][s], again[s]) for k in ('input', 'output')), "Nondeterministic.")
    # Catch omitted repairs, indiscriminate restoration, and wrong collector order.
    for mutation in ('omit', 'all', 'prefix'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
        pair = next(p for p in bad['train'] if any(not x for x in _analyze(p['input'])['used']))
        a = _analyze(pair['input'])
        if mutation == 'prefix':
            r, c = a['pos']
            pair['output'][r, c] = a['slot']
        else:
            index = a['used'].index(mutation == 'omit')
            gap, spur = a['cues'][index]
            for cell in gap+spur:
                pair['output'][cell] = (pair['input'][cell] if mutation == 'omit'
                                       else a['rail'] if cell in gap else a['bg'])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupted label accepted: "+mutation)
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            _counts(n, t)
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    return dict(seeds=42, pairs=len(features), shapes=len({f['shape'] for f in features}),
                palettes=len({f['palette'] for f in features}),
                selected_counts=sorted({f['selected'] for f in features}),
                reachable_counts=sorted({f['reachable'] for f in features}),
                potential_bridge_counts=sorted({f['potential_bridges'] for f in features}),
                used_bridge_counts=sorted({f['used_bridges'] for f in features}),
                max_route=max(f['max_distance'] for f in features))
