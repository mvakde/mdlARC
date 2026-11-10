"""Braided keys conditionally dock hollow locks. Standard library + NumPy only.

No files, parent imports, generation state or training labels are used by solve.
All sampling retries and the self-test are bounded; no traces are stored.
"""
from __future__ import annotations

import hashlib
import numpy as np


PORTS = {"NW": (-1, -1), "N0": (-1, 0), "N1": (-1, 1),
         "NE": (-1, 2), "W0": (0, -1), "E0": (0, 2),
         "W1": (1, -1), "E1": (1, 2), "SW": (2, -1),
         "S0": (2, 0), "S1": (2, 1), "SE": (2, 2)}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _need(bool(np.all((0 <= g) & (g <= 9))), "invalid color")
    return g.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = set()
    result = []
    for r, c in zip(*np.nonzero(mask)):
        p = (int(r), int(c))
        if p in seen:
            continue
        seen.add(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [p for p in _components(~mask)
            if all(r not in (0, h-1) and c not in (0, w-1) for r, c in p)]


def _lock_details(g, bg, rail):
    objects = []
    for cells0 in _components(g[:, rail+1:] != bg):
        cells = [(r, c+rail+1) for r, c in cells0]
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        colors = {int(g[p]) for p in cells}
        _need(len(colors) == 1, "a lock must be monochrome")
        mask = np.zeros((r1-r0+1, c1-c0+1), bool)
        mask[rr-r0, cc-c0] = True
        active, decoy = [], []
        for hole in _holes(mask):
            if any(c == 1 for r, c in hole):
                _need({r for r, c in hole} == {r for r, c in hole if c == 1},
                      "partially contacting cavity")
                active.append(hole)
            else:
                decoy.append(hole)
        active_rows = {r0+r for hole in active for r, c in hole}
        enclosed_rows = {r0+r for hole in active+decoy for r, c in hole}
        wall_rows = {r0+r for r in range(mask.shape[0])
                     if mask.shape[1] >= 3 and not mask[r, 1]}
        open_rows = wall_rows - active_rows
        objects.append(dict(cells=cells, mask=mask, r0=r0, r1=r1, c0=c0, c1=c1,
                            color=colors.pop(), active=active, decoy=decoy,
                            rows=active_rows, enclosed=enclosed_rows,
                            wall=wall_rows, opened=open_rows, shift=c0-rail-1))
    _need(bool(objects), "no locks")
    objects.sort(key=lambda o: o['r0'])
    _need(all(b['r0'] > a['r1']+1 for a, b in zip(objects, objects[1:])),
          "lock bands overlap or touch")
    return objects


def _heading(offset):
    r, c = offset
    return (-1 if r == -1 else 1 if r == 2 else 0,
            -1 if c == -1 else 1 if c == 2 else 0)


def _launchers(g, bg, rail):
    h, _ = g.shape
    choices = {}
    for r in range(1, h-2):
        for c in range(1, rail-2):
            color = int(g[r, c])
            if color == bg or not np.all(g[r:r+2, c:c+2] == color):
                continue
            ring = [(r+dr, c+dc, (dr, dc)) for dr, dc in PORTS.values()]
            fg = {int(g[x, y]) for x, y, _ in ring if g[x, y] != bg}
            gaps = [(x, y, off) for x, y, off in ring if g[x, y] == bg]
            if len(fg) != 1 or not gaps or len(gaps) > 9 or color in fg:
                continue
            structure = next(iter(fg))
            choices.setdefault(structure, []).append(dict(core=(r, c), color=color,
                ports=[((x, y), _heading(off)) for x, y, off in gaps]))
    valid = []
    actual = set(zip(*np.nonzero(g[:, :rail] != bg)))
    for structure, launchers in choices.items():
        used, covered = set(), set()
        okay = True
        for l in launchers:
            r, c = l['core']
            footprint = {(x, y) for x in range(r-1, r+3) for y in range(c-1, c+3)}
            if footprint & used:
                okay = False
                break
            used |= footprint
            covered |= {p for p in footprint if int(g[p]) != bg}
        if okay and covered == actual:
            valid.append((structure, launchers))
    _need(len(valid) == 1, "ambiguous or absent launcher vocabulary")
    return valid[0]


def _scene(value):
    original = _grid(value)
    candidates = []
    for turn in range(4):
        g = np.rot90(original, turn)
        h, w = g.shape
        bg = int(g[0, 0])
        if not all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))):
            continue
        for rail in range(5, w-3):
            color = int(g[0, rail])
            if color == bg or not np.all(g[:, rail] == color) or np.count_nonzero(g == color) != h:
                continue
            try:
                structure, launchers = _launchers(g, bg, rail)
                locks = _lock_details(g, bg, rail)
                _need(all(o['color'] not in (bg, color, structure) for o in locks),
                      "lock role collides with another role")
                border = np.ones(g.shape, bool)
                border[1:-1, 1:-1] = False
                border[:, rail] = False
                _need(np.all(g[border] == bg), "apparatus must have a clear outside border")
                candidates.append(dict(grid=g, turn=turn, bg=bg, rail=rail,
                                       rail_color=color, structure=structure,
                                       launchers=launchers, locks=locks))
            except AssertionError:
                continue
    _need(len(candidates) == 1, "rail-relative orientation is not unique")
    return candidates[0]


def _walk(point, velocity, shape, rail):
    dr, dc = velocity
    scale = max(abs(dr), abs(dc))
    result = []
    for t in range(3*max(shape)+1):
        p = (point[0] + (1 if dr >= 0 else -1)*(t*abs(dr)//scale),
             point[1] + (1 if dc >= 0 else -1)*(t*abs(dc)//scale))
        if not (0 <= p[0] < shape[0] and 0 <= p[1] <= rail):
            break
        result.append(p)
        if p[1] == rail:
            break
    return result


def _encounter(aa, bb, rail):
    hits = []
    for i, a in enumerate(aa):
        lookup = {p: t for t, p in enumerate(a) if p[1] < rail}
        mid = {(p[0]+q[0], p[1]+q[1]): t for t, (p, q) in enumerate(zip(a, a[1:]))
               if p[0] != q[0] and p[1] != q[1] and q[1] <= rail}
        for j, b in enumerate(bb):
            for s, p in enumerate(b):
                if p in lookup:
                    t = lookup[p]
                    hits.append((t-1, s-1, i, j, 'cell', t, s))
            for s, (p, q) in enumerate(zip(b, b[1:])):
                key = (p[0]+q[0], p[1]+q[1])
                if p[0] == q[0] or p[1] == q[1] or q[1] > rail or key not in mid:
                    continue
                t = mid[key]
                if {a[t], a[t+1]} != {p, q}:
                    hits.append((t, s, i, j, 'between', t+.5, s+.5))
    return min(hits, key=lambda e: (sum(e[:2]), max(e[:2]), e[2:4])) if hits else None


def _kind(a, b):
    _need(max(map(abs, a)) == max(map(abs, b)) == 1, "staircase re-encounter")
    total = (a[0]+b[0], a[1]+b[1])
    _need(total != (0, 0) and a != b, "head-on or parallel encounter")
    na, nb = sum(map(abs, a)), sum(map(abs, b))
    if na == nb == 1:
        return 'perpendicular'
    if na == nb == 2:
        return 'opposing'
    _need(a[0]*b[0]+a[1]*b[1] > 0, "obtuse cardinal/diagonal encounter")
    return 'staircase'


def _trace(scene, bend=True):
    g, bg, rail = scene['grid'], scene['bg'], scene['rail']
    groups, colors = [], []
    for launcher in scene['launchers']:
        for point, direction in launcher['ports']:
            ident = len(colors)
            colors.append(launcher['color'])
            groups.append(dict(v=direction, members=[(ident, point)], depth=0))
    output = g.copy()
    painted, last, diagonals, contacts, events = {}, {}, {}, {}, []

    def paint(ident, path):
        for index, p in enumerate(path):
            if p[1] == rail:
                _need(p[0] not in contacts, "duplicate rail contact")
                contacts[p[0]] = colors[ident]
                continue
            _need(g[p] == bg, "ray overwrites a launcher")
            _need(p not in painted or (index == 0 and painted[p] == ident and last.get(ident) == p),
                  "overlap or self-intersection")
            painted[p], last[ident] = ident, p
            output[p] = colors[ident]
        for p, q in zip(path, path[1:]):
            if p[0] != q[0] and p[1] != q[1] and q[1] <= rail:
                key = (p[0]+q[0], p[1]+q[1])
                slope = 1 if (q[0]-p[0])*(q[1]-p[1]) > 0 else -1
                _need(key not in diagonals or diagonals[key] == slope, "unresolved crossing")
                diagonals[key] = slope

    if not bend:
        # Deliberately competing predictor, not the semantic tracing path:
        # draw every original ray independently, with no collision handling.
        for group in groups:
            for ident, point in group['members']:
                for p in _walk(point, group['v'], g.shape, rail):
                    if p[1] < rail and g[p] == bg:
                        output[p] = colors[ident]
                    elif p[1] == rail:
                        contacts[p[0]] = colors[ident]
        return output, contacts, []

    for _ in range(len(colors)+1):
        forecasts = [[_walk(p, group['v'], g.shape, rail) for ident, p in group['members']]
                     for group in groups]
        possible = []
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                hit = _encounter(forecasts[i], forecasts[j], rail)
                if hit is not None:
                    possible.append((i, j, hit))
        if not possible:
            for group, paths in zip(groups, forecasts):
                for (ident, _), path in zip(group['members'], paths):
                    paint(ident, path)
            break
        first = {}
        for i, j, e in possible:
            first[i] = min(first.get(i, 1000), e[0])
            first[j] = min(first.get(j, 1000), e[1])
        ready = [(i, j, e) for i, j, e in possible if e[0] == first[i] and e[1] == first[j]]
        _need(bool(ready), "cyclic encounter order")
        i, j, e = min(ready, key=lambda v: (max(v[2][:2]), sum(v[2][:2]), v[:2]))
        _need(min(e[:2]) >= 0, "encounter at a port or previous bend")
        for k, l, other in possible:
            if (k, l) != (i, j):
                _need(not ((k in (i, j) and other[0] == first[k]) or
                           (l in (i, j) and other[1] == first[l])), "multiway encounter")
        a, b = groups[i], groups[j]
        _need(a['depth'] == b['depth'] == 0, "later bundle encounters are outside this family")
        kind = _kind(a['v'], b['v'])
        velocity = (a['v'][0]+b['v'][0], a['v'][1]+b['v'][1])
        divisor = int(np.gcd(abs(velocity[0]), abs(velocity[1])))
        velocity = (velocity[0]//divisor, velocity[1]//divisor)
        members = []
        for k, cut in ((i, e[0]), (j, e[1])):
            for (ident, _), path in zip(groups[k]['members'], forecasts[k]):
                _need(cut < len(path)-1 and path[cut][1] < rail, "partly clipped encounter")
                paint(ident, path[:cut+1])
                members.append((ident, path[cut]))
        _need(len({colors[ident] for ident, _ in members}) == len(members),
              "interfering payloads must be distinct")
        events.append(dict(kind=kind, intersection=e[4], unequal=e[5] != e[6],
                           incoming=(a['v'], b['v']), outgoing=velocity,
                           distances=(e[5], e[6]), position=forecasts[i][e[2]][e[0]]))
        groups = [group for k, group in enumerate(groups) if k not in (i, j)]
        groups.append(dict(v=velocity, members=members, depth=1))
    else:
        raise AssertionError("bounded ray merge count exceeded")
    return output, contacts, events


def _render(scene, ray_output, contacts, gate='exact'):
    out = ray_output.copy()
    gates = []
    for obj in scene['locks']:
        key = set(contacts) & set(range(obj['r0'], obj['r1']+1))
        signature = obj['rows']
        if gate == 'all_holes':
            signature = obj['enclosed']
        elif gate == 'wall_only':
            signature = obj['wall']
        unlock = bool(signature) and signature == key
        if gate == 'unconditional':
            unlock = True
        elif gate == 'overlap':
            unlock = bool(signature & key)
        elif gate == 'incoming_subset':
            unlock = bool(key) and key <= signature
        if unlock:
            for p in obj['cells']:
                out[p] = scene['bg']
            for r, c in obj['cells']:
                out[r, c-obj['shift']] = obj['color']
        gates.append(unlock)
    return out, gates


def _analyze(value):
    scene = _scene(value)
    rays, contacts, events = _trace(scene)
    output, gates = _render(scene, rays, contacts)
    scene.update(rays=rays, contacts=contacts, events=events, gates=gates, output=output)
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the visible apparatus alone."""
    scene = _analyze(input_grid)
    return np.rot90(scene['output'], -scene['turn']).copy()


def _at_port(point, name, color):
    dr, dc = PORTS[name]
    return dict(core=(point[0]-dr, point[1]-dc), color=color, ports=(name,))


def _build_network(h, rail, specs):
    g = np.zeros((h, rail+1), dtype=np.int8)
    used = set()
    for spec in specs:
        r, c = spec['core']
        _need(1 <= r <= h-3 and 1 <= c <= rail-3, "launcher footprint clipped")
        footprint = {(x, y) for x in range(r-1, r+3) for y in range(c-1, c+3)}
        _need(not used & footprint, "launcher footprints overlap")
        used |= footprint
        for off in PORTS.values():
            g[r+off[0], c+off[1]] = 8
        g[r:r+2, c:c+2] = spec['color']
        for name in spec['ports']:
            off = PORTS[name]
            g[r+off[0], c+off[1]] = 0
    return g


def _network_trial(h, rail, specs):
    g = _build_network(h, rail, specs)
    _, launchers = _launchers(g, 0, rail)
    s = dict(grid=g, bg=0, rail=rail, launchers=launchers)
    out, contacts, events = _trace(s)
    _need(len(contacts) in (1, 2), "module has wrong contact count")
    _need(min(contacts) >= 2 and max(contacts) <= h-3, "contacts need lock margins")
    _need(np.all(g[0] == 0) and np.all(g[-1] == 0) and np.all(g[:, 0] == 0),
          "network needs a blank border")
    return g, contacts, events


def _module(rng, h, rail, mode, require_unequal=False):
    """One procedurally placed encounter, or a singleton; no disk retries."""
    last_error = None
    for attempt in range(160):
        x = int(rng.integers(7, rail-1))
        t = int(rng.integers(4, h-4))
        a, b = map(int, rng.integers(1, 5, 2))
        if mode in ('cell', 'between'):
            delta = int(mode == 'between')
            specs = [_at_port((t-a, x-a), 'SE', 1),
                     _at_port((t+b+delta, x-b), 'NE', 2)]
        elif mode == 'staircase':
            a += 1
            specs = [_at_port((t-a, x-a), 'SE', 1),
                     _at_port((t, x-b), 'E0', 2)]
        elif mode == 'perpendicular':
            b += 1
            specs = [_at_port((t, x-a), 'E0', 1),
                     _at_port((t-b, x), 'S0', 2)]
        else:
            specs = [_at_port((t, x), str(rng.choice(('E0', 'E1'))), 1)]
        try:
            g, contacts, events = _network_trial(h, rail, specs)
            if mode == 'single':
                _need(not events and len(contacts) == 1, "singleton has interaction")
            else:
                _need(len(events) == 1 and len(contacts) == 2, "primary encounter missing")
                expected = 'opposing' if mode in ('cell', 'between') else mode
                _need(events[0]['kind'] == expected, "wrong encounter kind")
                _need(not require_unequal or events[0]['unequal'], "equal approaches do not teach spatial scheduling")
                if mode in ('cell', 'between'):
                    _need(events[0]['intersection'] == mode, "wrong intersection type")
            return g, sorted(contacts)
        except AssertionError as error:
            last_error = error
    # Safe, parameterized fallback: h>=14 for single modules, h>=13 for two.
    if mode in ('cell', 'between'):
        t, x, delta = 5, rail-3, int(mode == 'between')
        specs = [_at_port((t-1, x-1), 'SE', 1),
                 _at_port((t+2+delta, x-2), 'NE', 2)]
    elif mode == 'staircase':
        t, x = 6, rail-3
        specs = [_at_port((t-2, x-2), 'SE', 1), _at_port((t, x-3), 'E0', 2)]
    elif mode == 'perpendicular':
        t, x = 6, rail-3
        specs = [_at_port((t, x-3), 'E0', 1), _at_port((t-2, x), 'S0', 2)]
    else:
        specs = [_at_port((h//2, rail-4), 'E0', 1)]
    try:
        g, contacts, _ = _network_trial(h, rail, specs)
        return g, sorted(contacts)
    except AssertionError as error:
        raise AssertionError(f"module fallback failed after {last_error}: {error}") from error


def _runs(rows):
    result = []
    for r in sorted(rows):
        if result and r == result[-1][-1]+1:
            result[-1].append(r)
        else:
            result.append([r])
    return result


def _lock_mask(rng, contacts, kind, width, rich):
    """Build geometry; actual enclosure is always re-read from its pixels."""
    signature = set(contacts)
    if kind == 'surplus':
        signature.add(min(signature)-1)
    row0, row1 = min(signature)-1, max(signature)+1
    mask = np.ones((row1-row0+1, width), bool)
    runs = _runs({r-row0 for r in signature})
    target = int(rng.integers(len(runs))) if kind in ('open', 'recess') else -1
    for i, rows in enumerate(runs):
        left = 2 if kind == 'recess' and i == target else 1
        back = int(rng.integers(left, width-1))
        for r in rows:
            edge = int(rng.integers(left, width-1)) if rich else back
            mask[r, left:edge+1] = False
        if kind == 'open' and i == target:
            # Back opening: the gap is exterior even though the facing wall is intact.
            for r in rows:
                mask[r, left:] = False
    if kind == 'tail':
        enlarged = np.zeros((mask.shape[0], width+1), bool)
        enlarged[:, 1:] = mask
        enlarged[0, 0] = True
        mask = enlarged
    if rich:
        # Changing the back outline cannot change leading contact or enclosure.
        for r in (0, mask.shape[0]-1):
            if mask.shape[1] > 3 and rng.random() < .7:
                mask[r, -1] = False
        if kind == 'mixed':
            # Reserve a separate recessed pocket by extending the body backward.
            # A foreground separator keeps it distinct from all active cavities.
            extra = np.ones((mask.shape[0], 3), bool)
            extra[1, 1] = False
            mask = np.concatenate((mask, extra), axis=1)
    return row0, mask


def _pair(rng, modes, kinds, config, rich, force_move=False):
    count = len(modes)
    if count == 2:
        heights = [int(rng.integers(13, 16)), int(rng.integers(13, 16))]
    else:
        heights = [int(rng.integers(15, 31))]
    h = sum(heights)
    rail = int(rng.integers(13, 20))
    width = int(rng.integers(rail+(10 if 'mixed' in kinds else 7), 31))
    g = np.zeros((h, width), dtype=np.int8)
    g[:, rail] = 9
    row = 0
    for module_index, (height, mode, kind) in enumerate(zip(heights, modes, kinds)):
        net, contacts = _module(rng, height, rail, mode, require_unequal=force_move)
        if rng.random() < .5:
            net = np.flipud(net).copy()
            contacts = sorted(height-1-r for r in contacts)
        g[row:row+height, :rail] = net[:, :rail]
        # The last canvas column is a required clear border, not work space.
        available = width-rail-2
        max_body_width = min(7, available-(4 if kind == 'mixed' else 1))
        body_width = int(rng.integers(4 if kind == 'recess' else 3, max_body_width+1))
        lock_row, mask = _lock_mask(rng, contacts, kind, body_width, rich)
        _need(lock_row >= 1 and lock_row+mask.shape[0] <= height-1, "lock clips module")
        max_shift = available-mask.shape[1]
        _need(max_shift >= 0, "lock too wide")
        low = 1 if (force_move or kind == 'mixed') and max_shift else 0
        shift = int(rng.integers(low, max_shift+1))
        _need(not force_move or shift > 0, "mandatory contrast must visibly move under its alternative")
        col = rail+1+shift
        view = g[row+lock_row:row+lock_row+mask.shape[0], col:col+mask.shape[1]]
        view[mask] = 7
        row += height
    # Remap canonical role values to arbitrary visible colors.
    palette = config['palette'] if config['fixed_palette'] else list(map(int, rng.permutation(10)))
    g = np.array(palette, dtype=np.int8)[g]
    turn = config['turn'] if config['fixed_orientation'] else int(rng.integers(4))
    if rng.random() < .5:
        g = np.flipud(g)
    g = np.rot90(g, turn).copy()
    return dict(input=g, output=solve(g))


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not low <= value <= high:
            raise ValueError("count outside legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    config = dict(palette=list(map(int, rng.permutation(10))),
                  fixed_palette=bool(rng.random() < .35),
                  fixed_orientation=bool(rng.random() < .4), turn=int(rng.integers(4)))
    third_mode = str(rng.choice(('cell', 'between', 'staircase')))
    fourth_mode = str(rng.choice(('single', 'perpendicular')))
    curriculum = [(['between', 'cell'], ['closed', 'open']),
                  (['cell', 'cell'], ['closed', 'recess'])]
    if nt >= 3:
        curriculum.append(([third_mode], ['surplus']))
    if nt >= 4:
        curriculum.append(([fourth_mode], ['closed']))
    taught_modes = ['cell', 'between'] + ([third_mode] if nt >= 3 else []) + ([fourth_mode] if nt >= 4 else [])
    while len(curriculum) < nt:
        m = str(rng.choice(taught_modes))
        curriculum.append(([m], [str(rng.choice(('closed', 'surplus', 'tail')))]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for index in range(number):
            if split == 'train':
                modes, kinds = curriculum[index]
                rich = index >= 2
                mandatory = index < 2
            else:
                n = int(rng.integers(1, 3))
                # Independent taught mechanisms can be recombined in the same
                # apparatus; each module certifies its own contact margins.
                options = taught_modes
                modes = [str(rng.choice(options)) for _ in range(n)]
                kinds = []
                for m in modes:
                    options_k = ['closed', 'closed', 'tail', 'mixed']
                    if m == 'cell':
                        options_k += ['open', 'recess']
                    if nt >= 3:
                        options_k += ['surplus']
                    kinds.append(str(rng.choice(options_k)))
                # Explicit compositional shift: the key fits all active rows,
                # while a separate recessed pocket remains irrelevant. The
                # mandatory negative training contrast already teaches this.
                kinds[0] = 'mixed'
                rich, mandatory = True, False
            for attempt in range(40):
                try:
                    visible_surplus = split == 'train' and index == 2
                    pair = _pair(rng, modes, kinds, config, rich, mandatory or visible_surplus)
                    _need(_key(pair['input']) not in seen, "duplicate input")
                    info = _analyze(pair['input'])
                    if mandatory:
                        _need(any(info['gates']) and not all(info['gates']), "missing mandatory gate contrast")
                        _need(all(o['shift'] > 0 for o in info['locks']), "invisible mandatory gate")
                    if split == 'test':
                        needed = _coverage(info)
                        taught = set().union(*(_coverage(_analyze(p['input'])) for p in puzzle['train']))
                        _need(needed <= taught, "test uses untaught mechanism")
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
                except AssertionError:
                    continue
            else:
                raise AssertionError(f"bounded construction exhausted for {split}/{index}")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _coverage(info):
    tags = set()
    for e in info['events']:
        tags.add((e['kind'], e['intersection']))
    if not info['events']:
        tags.add(('single', 'none'))
    return tags


def _alternative(info, name):
    if name == 'no_bend':
        rays, contacts, _ = _trace(info, bend=False)
        return _render(info, rays, contacts)[0]
    return _render(info, info['rays'], info['contacts'], name)[0]


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _counts(nt, ne)
    seen, taught, needed = set(), set(), set()
    failures = {k: False for k in ('unconditional', 'all_holes', 'wall_only', 'overlap', 'no_bend')}
    collision_controls_docking = False
    unequal_taught = False
    surplus_taught = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _need(a.shape == b.shape, "dimensions changed")
            _need(_key(a) not in seen, "duplicate inputs")
            seen.add(_key(a))
            info = _analyze(a)
            canonical_b = np.rot90(b, info['turn'])
            _need(np.array_equal(canonical_b, info['output']), "incorrect output")
            rail = info['rail']
            _need(np.array_equal(info['grid'][:, rail], canonical_b[:, rail]), "rail changed")
            _need(np.array_equal(info['grid'][:, :rail][info['grid'][:, :rail] != info['bg']],
                                canonical_b[:, :rail][info['grid'][:, :rail] != info['bg']]),
                  "launcher foreground changed")
            for obj, unlock in zip(info['locks'], info['gates']):
                key = set(info['contacts']) & set(range(obj['r0'], obj['r1']+1))
                _need(unlock == (bool(obj['rows']) and obj['rows'] == key), "gate mismatch")
                dest = [(r, c-obj['shift'] if unlock else c) for r, c in obj['cells']]
                _need(all(canonical_b[p] == obj['color'] for p in dest), "lock color/shape changed")
                if split == 'test' and bool(key) and key < obj['rows']:
                    _need(surplus_taught, "untaught surplus-row test")
            if split == 'train':
                taught |= _coverage(info)
                unequal_taught |= any(e['unequal'] for e in info['events'])
                for name in failures:
                    failures[name] |= not np.array_equal(_alternative(info, name), canonical_b)
                surplus_taught |= not np.array_equal(_alternative(info, 'incoming_subset'), canonical_b)
                no_bend_rays, no_bend_contacts, _ = _trace(info, bend=False)
                _, no_bend_gates = _render(info, no_bend_rays, no_bend_contacts)
                collision_controls_docking |= any(
                    x != y and obj['shift'] > 0
                    for x, y, obj in zip(info['gates'], no_bend_gates, info['locks']))
            else:
                needed |= _coverage(info)
                _need(any(gate and obj['shift'] > 0 and obj['decoy']
                          for gate, obj in zip(info['gates'], info['locks'])),
                      "query lacks the successful active-plus-recessed combination")
    _need(needed <= taught, "untaught test encounter")
    _need(all(failures.values()), f"training fails to distinguish alternatives: {failures}")
    _need(collision_controls_docking, "ray encounters must change a demonstrated docking decision")
    _need(unequal_taught, "training must distinguish spatial from equal-clock scheduling")
    if nt >= 3:
        _need(surplus_taught, "training lacks surplus active-row rejection")


def _fixture():
    """Hand-specified shared-cell encounter and matching two-cavity lock."""
    g = np.zeros((15, 23), np.int8)
    rail = 15
    specs = [_at_port((4, 6), 'SE', 1), _at_port((8, 6), 'NE', 2)]
    net = _build_network(15, rail, specs)
    g[:, :rail] = net[:, :rail]
    g[:, rail] = 9
    g[4:9, 19:22] = 7
    g[5, 20] = g[7, 20] = 0
    return g


def self_test() -> dict:
    """Bounded 42-seed, input-derived semantic and diversity diagnostics."""
    fixture = _fixture()
    info = _analyze(fixture)
    _need(set(info['contacts']) == {5, 7}, "fixture rail contacts")
    _need(info['gates'] == [True], "fixture must unlock")
    expected = fixture.copy()
    expected[4:9, 19:22] = 0
    expected[4:9, 16:19] = 7
    expected[5, 17] = expected[7, 17] = 0
    # Ports reach (5,7)/(7,7), then continue east. The crossing at
    # (6,8) is bypassed; colors preserve their own rows to the rail.
    for row, col, color in ((4, 6, 1), (5, 7, 1), (8, 6, 2), (7, 7, 2)):
        expected[row, col] = color
    expected[5, 8:15] = 1
    expected[7, 8:15] = 2
    _need(np.array_equal(solve(fixture), expected), "manual collision/docking fixture")
    for turn in range(4):
        _need(np.array_equal(solve(np.rot90(fixture, turn)), np.rot90(expected, turn)), "rotation regression")
    shapes, signatures, structures, modes, palettes, directions = set(), set(), set(), set(), set(), set()
    counts, distances, hole_counts, key_patterns = set(), set(), set(), set()
    objects = pairs = unequal = opens = recessed = unlocked = locked = surplus = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                digest.update(a.tobytes())
                info = _analyze(a)
                pairs += 1
                shapes.add(a.shape)
                modes |= _coverage(info)
                directions.add(info['turn'])
                palettes.add((info['bg'], info['rail_color'], info['structure']))
                counts.add(len(info['launchers']))
                normalized = {}
                structures.add((a.shape, bytes(normalized.setdefault(int(c), len(normalized)) for c in a.ravel())))
                unequal += sum(e['unequal'] for e in info['events'])
                for obj, gate in zip(info['locks'], info['gates']):
                    objects += 1
                    distances.add(obj['shift'])
                    hole_counts.add(len(obj['active']))
                    key = set(info['contacts']) & set(range(obj['r0'], obj['r1']+1))
                    key_patterns.add(tuple(sorted(r-obj['r0'] for r in key)))
                    opens += bool(obj['opened'])
                    recessed += bool(obj['decoy'])
                    unlocked += gate
                    locked += not gate
                    surplus += bool(key) and key < obj['rows']
        signatures.add(digest.hexdigest())
    _need(len(signatures) == 42, "cross-seed duplicate puzzles")
    _need(len(shapes) >= 30 and len(structures) >= 160, "structural variation collapsed")
    _need({('opposing', 'cell'), ('opposing', 'between'), ('staircase', 'cell'),
           ('perpendicular', 'cell'), ('single', 'none')} <= modes, "missing encounter variety")
    _need(directions == set(range(4)) and len(palettes) >= 25, "role/orientation collapse")
    _need({1, 2, 4} <= counts and len(distances) >= 5 and 0 in distances, "count/displacement collapse")
    _need({0, 1, 2} <= hole_counts and len(key_patterns) >= 5, "cavity/signature collapse")
    _need(unequal >= 20 and min(opens, recessed, unlocked, locked, surplus) >= 20,
          "semantic contrasts collapsed")
    a, b = generate_puzzle(99173, 7, 3), generate_puzzle(99173, 7, 3)
    _need(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
              for k in ('input', 'output')), "nondeterministic seed")
    # Exercise all legal explicit count combinations, without retaining runs.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(1000+10*nt+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    damaged = generate_puzzle(7, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt label")
    return dict(seeds=42, pairs=pairs, objects=objects, grid_shapes=len(shapes),
                color_normalized_structures=len(structures), launcher_counts=sorted(counts),
                displacements=sorted(distances), encounter_types=sorted(modes),
                unequal_encounters=unequal, open_locks=opens, recessed_locks=recessed,
                unlocked=unlocked, locked=locked, surplus_locks=surplus,
                independent_solver=False, blind_prediction=False)
