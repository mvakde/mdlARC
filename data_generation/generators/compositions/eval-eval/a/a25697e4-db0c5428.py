"""Radial echo molds: rigid slot fitting and handedness-corrected hub sampling.

Only standard library and NumPy are used. All solve-time information is visible
in the input. Coordinates stored by the constructor are never used by solve().
"""
from itertools import product
import numpy as np


DIR4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIRECTIONS = tuple(p for p in product((-1, 0, 1), repeat=2) if p != (0, 0))
# Fixed grid topology, not generated objects or cached puzzle answers. Local
# templates use -3..3; absolute ARC coordinates use 0..29.
_NEIGHBORS = {(r, c): tuple((r+dr, c+dc) for dr, dc in DIR4)
              for r, c in product(range(-3, 30), repeat=2)}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_counts(n, m):
    if not (_integer(n) and _integer(m) and 2 <= n <= 10 and 1 <= m <= 4 and n > m):
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a


def _components(cells):
    remaining = set(cells)
    answer = []
    while remaining:
        todo = [min(remaining)]
        remaining.remove(todo[0])
        part = set(todo)
        while todo:
            for q in _NEIGHBORS[todo.pop()]:
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    todo.append(q)
        answer.append(frozenset(part))
    return answer


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _shift(cells, origin):
    dr, dc = origin
    return frozenset((r + dr, c + dc) for r, c in cells)


def _local(cells, origin):
    return _shift(cells, (-origin[0], -origin[1]))


def _turn(point, direction, reflected):
    """Half-turn, optionally followed by reflection about the radial line.

    Both choices are involutions. The reflection is about a line through the
    reference pin parallel to direction, not about a bounding-box edge.
    """
    r, c = -point[0], -point[1]
    if not reflected:
        return r, c
    dr, dc = direction
    if dc == 0:
        return r, -c
    if dr == 0:
        return -r, c
    return (c, r) if dr == dc else (-c, -r)


def _transform(regions, direction, reflected):
    # Select the matrix once per object instead of once per material cell.
    dr, dc = direction
    if not reflected:
        return tuple(frozenset((-r, -c) for r, c in s) for s in regions)
    if dc == 0:
        return tuple(frozenset((-r, c) for r, c in s) for s in regions)
    if dr == 0:
        return tuple(frozenset((r, -c) for r, c in s) for s in regions)
    if dr == dc:
        return tuple(frozenset((-c, -r) for r, c in s) for s in regions)
    return tuple(frozenset((c, r) for r, c in s) for s in regions)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _cells(mask):
    # Python integer coordinates keep the many small set/geometry operations
    # out of NumPy's scalar dispatch. This changes neither order nor values.
    return map(tuple, np.argwhere(mask).tolist())


def _read_frame(a):
    """Find the unique hub-and-pins constellation, without fixed color numbers."""
    values, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "background frequency tie")
    bg = int(values[np.argmax(counts)])
    # A marker must contain a solid 3x3 hub. Screen all such colors together
    # before segmenting candidates, rather than segmenting every key color.
    if min(a.shape) < 3:
        candidates = ()
    else:
        corner = a[:-2, :-2]
        solid = np.ones(corner.shape, dtype=bool)
        for dr, dc in product(range(3), repeat=2):
            if dr or dc:
                solid &= corner == a[dr:dr+corner.shape[0], dc:dc+corner.shape[1]]
        candidates = np.unique(corner[solid])
    frames = []
    for marker in candidates:
        marker = int(marker)
        if marker == bg:
            continue
        parts = _components(_cells(a == marker))
        hubs = [s for s in parts if len(s) == 9 and
                _bbox(s)[1] - _bbox(s)[0] == 2 and _bbox(s)[3] - _bbox(s)[2] == 2]
        if len(hubs) != 1 or any(len(s) != 1 for s in parts if s != hubs[0]):
            continue
        r0, r1, c0, c1 = _bbox(hubs[0])
        center = r0 + 1, c0 + 1
        pins = sorted(next(iter(s)) for s in parts if s != hubs[0])
        rays = {}
        for pin in pins:
            r, c = pin[0] - center[0], pin[1] - center[1]
            if not (r == 0 or c == 0 or abs(r) == abs(c)):
                break
            d = int(np.sign(r)), int(np.sign(c))
            if d == (0, 0):
                break
            rays.setdefault(d, []).append((max(abs(r), abs(c)), pin))
        else:
            if not 2 <= len(rays) <= 8:
                continue
            if any(len(v) != 2 or sorted(v)[1][0] != 2 * sorted(v)[0][0]
                   or sorted(v)[0][0] < 4 for v in rays.values()):
                continue
            beams = [(d, sorted(rays[d])[0][1], sorted(rays[d])[1][1]) for d in sorted(rays)]
            frames.append(dict(background=bg, marker=marker, center=center,
                               hub=hubs[0], pins=pins, beams=beams))
    _require(len(frames) == 1, "hub/pin constellation is not unique")
    return frames[0]


def _analyze(a):
    a = _grid(a)
    info = _read_frame(a)
    bg, marker, pins = info['background'], info['marker'], info['pins']
    groups = {p: [] for p in pins}
    occupied = _cells((a != bg) & (a != marker))
    for component in _components(occupied):
        # Pins are physical reference points for nearby components. Compare exact
        # integer centroid distances, avoiding floating-point association ties.
        n = len(component)
        sr, sc = sum(r for r, c in component), sum(c for r, c in component)
        distances = [(sr - n*r)**2 + (sc - n*c)**2 for r, c in pins]
        best = min(distances)
        _require(distances.count(best) == 1, "component is equidistant from pins")
        p = pins[distances.index(best)]
        _require(all(max(abs(r-p[0]), abs(c-p[1])) <= 3 for r, c in component),
                 "component is outside its port")
        groups[p].append(component)
    assemblies = []
    wall_colors = set()
    for d, source, target in info['beams']:
        _require(len(groups[source]) == 1, "source must have one connected key")
        key = groups[source][0]
        colors = tuple(sorted({int(a[p]) for p in key}))
        _require(len(colors) == 2, "key must have two colors")
        regions = tuple(_local({p for p in key if a[p] == col}, source) for col in colors)
        _require(all(len(_components(s)) == 1 for s in regions), "key regions must be connected")
        _require(len(groups[target]) == 2, "mold must have two wall fragments")
        walls = groups[target][0] | groups[target][1]
        wc = {int(a[p]) for p in walls}
        _require(len(wc) == 1, "wall fragments differ in color")
        wall_colors.update(wc)
        r0, r1, c0, c1 = _bbox(walls)
        _require((r0+r1, c0+c1) == (2*target[0], 2*target[1]), "mold not centered on pin")
        _require((r1-r0+1, c1-c0+1) in ((3, 5), (5, 3), (5, 5)), "invalid mold size")
        rectangle = frozenset(product(range(r0, r1+1), range(c0, c1+1)))
        _require(all(a[p] == bg for p in rectangle - walls - {target}), "occupied mold gap")
        slot = rectangle - walls - {target}
        _require(slot and len(_components(slot)) == 1, "slot is not connected")
        cr, cc = info['center']
        distances = {(r-cr)**2 + (c-cc)**2 for r, c in key}
        minimum = min(distances)
        samples = [p for p in key if (p[0]-cr)**2+(p[1]-cc)**2 == minimum]
        _require(len(samples) == 1, "inward sample has a distance tie")
        fits = []
        for reflected in (False, True):
            transformed = _transform(regions, d, reflected)
            absolute = tuple(_shift(s, target) for s in transformed)
            for plug in (0, 1):
                if absolute[plug] != slot:
                    continue
                handle = absolute[1-plug]
                if handle & rectangle:
                    continue
                if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in handle):
                    continue
                if any(a[p] != bg for p in handle):
                    continue
                new_colors = colors[::-1] if reflected else colors
                placed = {p: col for s, col in zip(absolute, new_colors) for p in s}
                sample = int(a[samples[0]])
                report = colors[1-colors.index(sample)] if reflected else sample
                signature = (tuple(sorted(placed.items())), report)
                if all(f['signature'] != signature for f in fits):
                    fits.append(dict(signature=signature, reflected=reflected, plug=plug,
                                     placed=placed, handle=handle, regions=absolute,
                                     report=report))
        _require(len(fits) == 1, "mold has zero or multiple colored echo fits")
        fit = fits[0]
        fit.update(direction=d, source=source, target=target, colors=colors,
                   key=key, source_regions=regions, sample=samples[0],
                   sample_color=int(a[samples[0]]), slot=slot, walls=walls,
                   rectangle=rectangle, wall_color=next(iter(wc)))
        assemblies.append(fit)
    _require(len(wall_colors) == 1, "scene walls must share a color")
    _require(not (wall_colors & {c for m in assemblies for c in m['colors']}),
             "wall/key palette roles overlap")
    used = set()
    for m in assemblies:
        _require(not used.intersection(m['placed']), "echoes overlap")
        used.update(m['placed'])
    info['assemblies'] = assemblies
    return info


def _paint(a, info):
    b = a.copy()
    cr, cc = info['center']
    b[cr-1:cr+2, cc-1:cc+2] = info['background']
    b[cr, cc] = info['marker']
    for m in info['assemblies']:
        for p, color in m['placed'].items():
            b[p] = color
        dr, dc = m['direction']
        b[cr+dr, cc+dc] = m['report']
    return b


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct every pivot, slot, rigid echo and sample from this input."""
    a = _grid(input_grid)
    return _paint(a, _analyze(a))


def _random_shape(rng):
    """Generate a separator avoiding the reference pin, and an exterior handle."""
    for _ in range(200):
        h, w = ((3, 5), (5, 3), (5, 5))[int(rng.integers(3))]
        hr, hc = h//2, w//2
        rect = set(product(range(-hr, hr+1), range(-hc, hc+1)))
        y = int(rng.integers(-hr, hr+1))
        slot = {(y, -hc)}
        for c in range(-hc+1, hc+1):
            yy = max(-hr, min(hr, y + int(rng.integers(-1, 2))))
            slot.update((r, c) for r in range(min(y, yy), max(y, yy)+1))
            y = yy
        if (0, 0) in slot:
            continue
        for _ in range(int(rng.integers(4))):
            frontier = sorted({q for p in slot for q in _NEIGHBORS[p]}
                              & (rect - slot - {(0, 0)}))
            if frontier:
                slot.add(frontier[int(rng.integers(len(frontier)))])
        walls = rect - slot - {(0, 0)}
        if len(_components(walls)) != 2 or _bbox(walls) != (-hr, hr, -hc, hc):
            continue
        starts = sorted({q for p in slot for q in _NEIGHBORS[p]}
                        - rect)
        starts = [p for p in starts if max(map(abs, p)) <= 3]
        if not starts:
            continue
        handle = {starts[int(rng.integers(len(starts)))]}
        size = int(rng.integers(1, 10))
        for _ in range(size-1):
            frontier = sorted({q for p in handle for q in _NEIGHBORS[p]
                               if -3 <= q[0] <= 3 and -3 <= q[1] <= 3} - rect - handle)
            if not frontier:
                break
            p = frontier[int(rng.integers(len(frontier)))]
            handle.add(p)
        if len(handle) != size:
            continue
        # Transposition and two sign flips vary the mold's orientation.
        flip_r, flip_c = (int(rng.choice((-1, 1))) for _ in range(2))
        transpose = bool(rng.integers(2))
        def orient(s):
            return frozenset((c*flip_r, r*flip_c) if transpose else (r*flip_r, c*flip_c)
                             for r, c in s)
        return tuple(orient(s) for s in (walls, slot, handle))
    raise ValueError("separator construction exhausted")


def _near_own(components, origin, pins):
    for s in components:
        n = len(s)
        sr, sc = sum(r for r, c in s), sum(c for r, c in s)
        distances = {(r, c): (sr-n*r)**2 + (sc-n*c)**2 for r, c in pins}
        best = min(distances.values())
        if distances[origin] != best or sum(x == best for x in distances.values()) != 1:
            return False
    return True


def _moat(s):
    return set(s) | {q for p in s for q in _NEIGHBORS[p]}


def _make_pair(rng, count, palette=None):
    """Construct explicit targets/echoes; reconstruct them again before return."""
    for layout_attempt in range(80):
        bg, marker, wall, *key_palette = (list(map(int, rng.permutation(10)))
                                        if palette is None else palette)
        selected = sorted(int(x) for x in rng.choice(8, count, replace=False))
        center = int(rng.integers(14, 16)), int(rng.integers(14, 16))
        beams = []
        for index in selected:
            d = DIRECTIONS[index]
            radius = int(rng.choice((5, 6), p=(.25, .75)))
            source = center[0]+radius*d[0], center[1]+radius*d[1]
            target = center[0]+2*radius*d[0], center[1]+2*radius*d[1]
            beams.append((d, source, target))
        pins = [p for d, s, t in beams for p in (s, t)]
        hub = set(product(range(center[0]-1, center[0]+2), range(center[1]-1, center[1]+2)))
        anchors = hub | set(pins)
        a = np.full((30, 30), bg, dtype=np.int8)
        b = a.copy()
        for p in anchors:
            a[p] = b[p] = marker
        occupied = set(anchors)
        complete_ports = set()
        built = []
        order = list(range(count))
        rng.shuffle(order)
        modes = [False, True] + [bool(rng.integers(2)) for _ in range(count-2)]
        rng.shuffle(modes)
        for i in order:
            d, source, target = beams[i]
            reflected = modes[i]
            for attempt in range(250):
                walls, slot, handle = _random_shape(rng)
                regions = _transform((slot, handle), d, reflected)
                key = _shift(regions[0] | regions[1], source)
                aw, ass, ah = (_shift(s, target) for s in (walls, slot, handle))
                final = aw | ass | ah
                port_cells = key | final
                if any(not (0 <= r < 30 and 0 <= c < 30) for r, c in port_cells):
                    continue
                if port_cells & occupied or _moat(port_cells) & complete_ports:
                    continue
                # Keep anchors attached to only their own port, and leave the
                # hub visibly isolated. Keys/molds may touch their own pin.
                if _moat(port_cells) & (hub | (set(pins) - {source, target})):
                    continue
                if not _near_own([key], source, pins) or not _near_own(_components(aw), target, pins):
                    continue
                distances = [(r-center[0])**2+(c-center[1])**2 for r, c in key]
                if distances.count(min(distances)) != 1:
                    continue
                # Check the two permitted symmetries locally, for either plug.
                possibilities = []
                for mode in (False, True):
                    tr = _transform(regions, d, mode)
                    for plug in (0, 1):
                        if tr[plug] == slot and not tr[1-plug] & (walls | slot | {(0, 0)}):
                            absolute_other = _shift(tr[1-plug], target)
                            if all(0 <= r < 30 and 0 <= c < 30 for r, c in absolute_other):
                                if not absolute_other & (occupied | key | aw):
                                    possibilities.append((mode, plug))
                if possibilities != [(reflected, 0)]:
                    continue
                colors = tuple(int(x) for x in rng.choice(key_palette, 2, replace=False))
                for s, color in zip(regions, colors):
                    for p in _shift(s, source):
                        a[p] = b[p] = color
                for p in aw:
                    a[p] = b[p] = wall
                dest_colors = colors[::-1] if reflected else colors
                for s, color in zip((ass, ah), dest_colors):
                    for p in s:
                        b[p] = color
                sample = min(key, key=lambda p: (p[0]-center[0])**2+(p[1]-center[1])**2)
                sample_index = colors.index(int(a[sample]))
                report = colors[1-sample_index] if reflected else colors[sample_index]
                built.append((d, report))
                occupied.update(port_cells)
                complete_ports.update(port_cells)
                break
            else:
                break
        else:
            cr, cc = center
            b[cr-1:cr+2, cc-1:cc+2] = bg
            b[cr, cc] = marker
            for (dr, dc), report in built:
                b[cr+dr, cc+dc] = report
            # Tight, edge-touching and padded layouts all use the same rule.
            used = set(map(tuple, np.argwhere((a != bg) | (b != bg))))
            r0, r1, c0, c1 = _bbox(used)
            r0 = max(0, r0-int(rng.integers(4)))
            c0 = max(0, c0-int(rng.integers(4)))
            r1 = min(29, r1+int(rng.integers(4)))
            c1 = min(29, c1+int(rng.integers(4)))
            a, b = a[r0:r1+1, c0:c1+1].copy(), b[r0:r1+1, c0:c1+1].copy()
            try:
                info = _analyze(a)
                if np.array_equal(_paint(a, info), b):
                    return {'input': a, 'output': b}, info
            except AssertionError:
                pass
    raise RuntimeError("bounded radial construction exhausted")


def _training_features_from_infos(infos):
    """Summarize reconstructed inputs, reusing analyses within one validation."""
    modes, sample_roles = set(), set()
    nongrouped_reports = False
    for info in infos:
        reports = {False: set(), True: set()}
        for m in info['assemblies']:
            modes.add(m['reflected'])
            local_sample = (m['sample'][0]-m['source'][0], m['sample'][1]-m['source'][1])
            sample_region = next(i for i, s in enumerate(m['source_regions']) if local_sample in s)
            sample_roles.add(sample_region == m['plug'])
            diagonal = bool(m['direction'][0] * m['direction'][1])
            reports[diagonal].add(m['report'])
        nongrouped_reports |= any(len(s) > 1 for s in reports.values())
    return modes, sample_roles, nongrouped_reports


def _training_features(pairs):
    """Observable witnesses; neither seed nor sampling metadata enters checks."""
    return _training_features_from_infos(_analyze(np.asarray(p['input'])) for p in pairs)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10))) if rng.random() < .25 else None
    puzzle = {'train': [], 'test': []}
    seen = set()
    analyses = {}
    train_infos = []
    for i in range(num_train):
        for _ in range(30):
            pair, info = _make_pair(rng, int(rng.integers(2, 5)), palette)
            if _signature(pair['input']) in seen:
                continue
            # With only two training examples, ensure both material regions can
            # be the spatial sample. This excludes a plug-color reporting rule.
            # Also distinguish independent reports from corner/cardinal groups.
            if i == 1:
                features = _training_features_from_infos(train_infos + [info])
                if features != ({False, True}, {False, True}, True):
                    continue
            seen.add(_signature(pair['input']))
            puzzle['train'].append(pair)
            analyses[id(pair)] = info
            train_infos.append(info)
            break
        else:
            raise RuntimeError("training witness construction exhausted")
    for _ in range(num_test):
        for _ in range(30):
            pair, info = _make_pair(rng, int(rng.integers(5, 9)), palette)
            if _signature(pair['input']) not in seen:
                seen.add(_signature(pair['input']))
                puzzle['test'].append(pair)
                analyses[id(pair)] = info
                break
        else:
            raise RuntimeError("distinct test construction exhausted")
    rng.shuffle(puzzle['train'])
    # Reuse only input analyses just computed in this call, before any grids
    # escape to a caller. Public validation and solve always reconstruct input.
    _validate_puzzle(puzzle, analyses)
    return puzzle


def validate_generated_puzzle(puzzle):
    _validate_puzzle(puzzle)


def _validate_puzzle(puzzle, analyses=None):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    split_counts = {}
    train_infos = []
    reconstructed = {}
    for split in ('train', 'test'):
        split_counts[split] = []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "output shape changed")
            sig = _signature(a)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _analyze(a) if analyses is None else analyses[id(pair)]
            reconstructed[id(pair)] = info
            if split == 'train':
                train_infos.append(info)
            _require(np.array_equal(_paint(a, info), b), "incorrect output")
            split_counts[split].append(len(info['assemblies']))
            cr, cc = info['center']
            allowed = set(info['hub'])
            for m in info['assemblies']:
                _require(all(a[p] == b[p] for p in m['key'] | m['walls']), "source or wall changed")
                _require(a[m['source']] == b[m['source']] == info['marker'], "source pin changed")
                _require(a[m['target']] == b[m['target']] == info['marker'], "target pin changed")
                allowed.update(m['placed'])
                # Check copying cell by cell, separately from the output painter.
                colors = m['colors']
                for p in m['key']:
                    q = _turn((p[0]-m['source'][0], p[1]-m['source'][1]),
                              m['direction'], m['reflected'])
                    q = q[0]+m['target'][0], q[1]+m['target'][1]
                    col = int(a[p])
                    expected = colors[1-colors.index(col)] if m['reflected'] else col
                    _require(b[q] == expected, "rigid echo/color exchange failed")
                dr, dc = m['direction']
                _require(b[cr+dr, cc+dc] == m['report'], "hub report failed")
            mask = np.ones(a.shape, dtype=bool)
            for p in allowed:
                mask[p] = False
            _require(np.array_equal(a[mask], b[mask]), "unused cells changed")
            _require(set(np.unique(a)) == set(np.unique(b)), "palette changed")
    _require(_training_features_from_infos(train_infos) == ({False, True}, {False, True}, True),
             "training lacks handedness, region-sampling or directional witnesses")
    _require(max(split_counts['train']) < min(split_counts['test']), "ray-count shift missing")
    return reconstructed


def self_test():
    """Bounded 42-seed structural/semantic checks and targeted negative checks."""
    signatures, shapes, slots, handles, structures = set(), set(), set(), set(), set()
    directions, radii, counts, backgrounds, modes, mold_sizes = set(), set(), set(), set(), set(), set()
    relations, sample_roles = set(), set()
    edge_contacts = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        # Independently reconstruct generated inputs once for all validator
        # checks and measured features; solve below reconstructs them again.
        analyses = _validate_puzzle(p)
        signatures.add(tuple(_signature(x['input']) for s in ('train', 'test') for x in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                a, b = pair['input'], pair['output']
                info = analyses[id(pair)]
                _require(np.array_equal(solve(a.tolist()), b), "list input solve disagrees")
                shapes.add(a.shape)
                backgrounds.add(info['background'])
                counts.add(len(info['assemblies']))
                role_map = {}
                structures.add((a.shape, tuple(role_map.setdefault(int(v), len(role_map)) for v in a.flat)))
                for m in info['assemblies']:
                    directions.add(m['direction'])
                    modes.add(m['reflected'])
                    radii.add(max(abs(m['source'][j]-info['center'][j]) for j in (0, 1)))
                    slot = _local(m['slot'], m['target'])
                    handle = _local(m['handle'], m['target'])
                    slots.add(tuple(sorted(slot)))
                    handles.add(tuple(sorted(handle)))
                    relations.add(int(np.sign(len(slot)-len(handle))))
                    r0, r1, c0, c1 = _bbox(m['walls'])
                    mold_sizes.add((r1-r0+1, c1-c0+1))
                    sample_roles.add(int(a[m['sample']]) == m['colors'][m['plug']])
                    for r, c in m['placed']:
                        if r == 0: edge_contacts.add('top')
                        if c == 0: edge_contacts.add('left')
                        if r == a.shape[0]-1: edge_contacts.add('bottom')
                        if c == a.shape[1]-1: edge_contacts.add('right')
    _require(len(signatures) == 42 and len(structures) >= 180, "structural variation collapsed")
    _require(len(shapes) >= 20 and len(slots) >= 40 and len(handles) >= 60, "geometry variation missing")
    _require(directions == set(DIRECTIONS) and radii == {5, 6}, "direction/distance variation missing")
    _require(counts == set(range(2, 9)) and backgrounds == set(range(10)), "count/palette variation missing")
    _require(modes == {False, True} and sample_roles == {False, True}, "concept witnesses missing")
    _require(relations == {-1, 0, 1}, "region size relationships missing")
    _require(mold_sizes == {(3, 5), (5, 3), (5, 5)}, "mold aspects missing")
    _require(len(edge_contacts) == 4, "boundary examples missing")
    p, q = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), "determinism failed")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt a reflected hub report: ignoring A's parity must be detected.
    for corruption in ('hub', 'echo', 'source'):
        damaged = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
        pair = damaged['train'][0]
        info = _analyze(pair['input'])
        move = next(m for m in info['assemblies'] if m['reflected'])
        if corruption == 'hub':
            d = move['direction']
            cell = info['center'][0]+d[0], info['center'][1]+d[1]
        elif corruption == 'echo':
            cell = min(move['placed'])
        else:
            cell = min(move['key'])
        pair['output'][cell] = (int(pair['output'][cell])+1) % 10
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted family invariant accepted")
    return {'seeds_checked': 42, 'distinct_puzzles': len(signatures),
            'input_shapes': len(shapes), 'structures': len(structures),
            'oriented_slots': len(slots), 'oriented_handles': len(handles)}


if __name__ == '__main__':
    print(self_test())
