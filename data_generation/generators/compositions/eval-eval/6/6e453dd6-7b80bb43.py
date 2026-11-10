"""Cavity-powered circuit repair: docking, topology, and gated propagation.

No parent module, stored example, seed, or builder state is used by solve.
Only the standard library and NumPy are required.
"""
from collections import deque
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(((0 <= a) & (a <= 9)).all()), "invalid color")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
    result = []
    while cells:
        p = min(cells)
        cells.remove(p)
        stack, part = [p], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        result.append(part)
    return result


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _neighbors(mask):
    near = np.zeros_like(mask)
    near[1:] |= mask[:-1]
    near[:-1] |= mask[1:]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    return near


def _scene(grid):
    """The rail's two-cell perpendicular pointer identifies the receiver side."""
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 4, "input needs four visible roles")
    bg = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(grid == bg) > grid.size // 2, "background must be a majority")
    candidates = []
    for turn in range(4):
        g = np.rot90(grid, turn)
        h, w = g.shape
        for col in range(1, w - 3):
            power = int(g[0, col])
            if power == bg or not np.all(g[:, col] == power):
                continue
            extra = np.argwhere((g == power) & (np.indices(g.shape)[1] != col))
            if len(extra) != 2:
                continue
            rr = int(extra[0, 0])
            if set(map(tuple, extra.tolist())) != {(rr, col + 1), (rr, col + 2)}:
                continue
            left = set(map(int, np.unique(g[:, :col]))) - {bg, power}
            right = set(map(int, np.unique(g[:, col + 1:]))) - {bg, power}
            if len(left) == len(right) == 1 and left != right:
                candidates.append((g, turn, bg, left.pop(), power, right.pop(), col, rr))
    _require(len(candidates) == 1, "rail direction or role ambiguity")
    return candidates[0]


def _plates(g, bg, plate, col):
    objects = []
    for cells in _components(g[:, :col] == plate):
        rr, cc = np.array(cells).T
        r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
        mask[rr - r0, cc - c0] = True
        holes, active, recessed = [], [], []
        for part in _components(~mask):
            if any(r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1) for r, c in part):
                continue
            holes.append(part)
            contacts = {r for r, c in part if c == mask.shape[1] - 2 and mask[r, -1]}
            if contacts:
                _require(contacts == {r for r, c in part}, "partially leading cavity is excluded")
                active.append(part)
            else:
                recessed.append(part)
        active_rows = {r0 + r for part in active for r, c in part}
        enclosed = {p for part in holes for p in part}
        open_rows = {r0 + r for r in range(1, mask.shape[0] - 1)
                     if mask.shape[1] >= 3 and not mask[r, -2]
                     and (r, mask.shape[1] - 2) not in enclosed}
        objects.append(dict(cells=cells, mask=mask, row=r0, end=r1, shift=col - 1 - c1,
                            active=active, recessed=recessed, active_rows=active_rows,
                            open_rows=open_rows))
    _require(bool(objects), "no plates")
    return objects


def _gaps(mask):
    result = []
    for dr, dc in ((0, 1), (1, 0)):
        m = mask if dc else mask.T
        if m.shape[1] < 5:
            continue
        bracket = m[:, :-4] & m[:, 4:]
        bracket &= ~(m[:, 1:-3] | m[:, 2:-2] | m[:, 3:-1])
        for y, x in np.argwhere(bracket):
            r, c = (int(y), int(x)) if dc else (int(x), int(y))
            result.append((r, c, dr, dc))
    return result


def _diagonal(part):
    p = sorted(part)
    return (len(p) in (2, 3) and all(p[i+1][0] - p[i][0] == 1 for i in range(len(p)-1))
            and len({p[i+1][1] - p[i][1] for i in range(len(p)-1)}) == 1
            and abs(p[1][1] - p[0][1]) == 1)


def _wires(raw):
    # Family linework has no legitimate singleton. Oblique off-axis cells have
    # no orthogonal neighbor, even when diagonally attached to their anchor.
    erased = raw & ~_neighbors(raw)
    parts = _components(erased, True)
    _require(all(_diagonal(p) for p in parts), "unsupported wire noise or singleton")
    backbone = raw & ~erased
    _require(not (backbone[:-1,:-1] & backbone[1:,:-1] & backbone[:-1,1:] & backbone[1:,1:]).any(),
             'wire backbone is not thin')
    pairs = {frozenset(p) for p in parts if len(p) == 2}
    candidates = []
    for r, c, dr, dc in _gaps(backbone):
        if not (_at(backbone, r-dr, c-dc) or _at(backbone, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint in (0, 4):
            ar, ac = r + endpoint*dr, c + endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr + side*dc, toward*dc + side*dr
                spur = frozenset((ar+k*sr, ac+k*sc) for k in (1, 2))
                if spur in pairs and not _at(raw, ar+3*sr, ac+3*sc):
                    matches.append((endpoint, side, spur))
        if matches:
            _require(len(matches) == 1, "ambiguous gate cue")
            candidates.append(dict(start=(r, c, dr, dc), ends=((r, c), (r+4*dr, c+4*dc)),
                                   cells=tuple((r+k*dr, c+k*dc) for k in (1, 2, 3)),
                                   spur=matches[0][2], endpoint=matches[0][0], side=matches[0][1]))
    _require(len({g['spur'] for g in candidates}) == len(candidates), 'one cue marks several gates')
    gap_cells = [cell for gate in candidates for cell in gate['cells']]
    _require(len(set(gap_cells)) == len(gap_cells), 'gate repair spans intersect')
    return backbone, erased, candidates


def _flood(mask, seeds):
    reachable = np.zeros_like(mask)
    q = deque()
    for r, c in seeds:
        if mask[r, c] and not reachable[r, c]:
            reachable[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if _at(mask, nr, nc) and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def _propagate(backbone, candidates, source_rows, *, ignore_power=False, ignore_cues=False):
    seeds, beams = set(), []
    for row in sorted(source_rows):
        hits = np.flatnonzero(backbone[row])
        end = int(hits[0]) if hits.size else backbone.shape[1] - 1
        beams.extend((row, c) for c in range(end + 1))
        if hits.size:
            seeds.add((row, end))
    gates = candidates
    if ignore_cues:
        gates = [dict(cells=tuple((r+k*dr, c+k*dc) for k in (1, 2, 3)),
                      ends=((r, c), (r+4*dr, c+4*dc))) for r, c, dr, dc in _gaps(backbone)]
    mask = backbone.copy()
    selected, rounds = set(), []
    while True:
        powered = _flood(mask, seeds)
        new = [i for i, gate in enumerate(gates) if i not in selected and
               (ignore_power or any(powered[r, c] for r, c in gate['ends']))]
        if not new:
            break
        rounds.append(tuple(new))
        for i in new:
            selected.add(i)
            for r, c in gates[i]['cells']:
                mask[r, c] = True
    return mask, _flood(mask, seeds), beams, selected, rounds


def _analyze(input_grid):
    grid = _grid(input_grid)
    g, turn, bg, plate, power, wire, col, pointer_row = _scene(grid)
    objects = _plates(g, bg, plate, col)
    source_rows = {r for obj in objects for r in obj['active_rows']}
    raw = g[:, col+1:] == wire
    backbone, erased, gates = _wires(raw)
    mask, powered, beams, selected, rounds = _propagate(backbone, gates, source_rows)
    _require(not (mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]).any(),
             'repaired wirework is not thin')
    return dict(g=g, turn=turn, bg=bg, plate=plate, power=power, wire=wire, col=col,
                pointer_row=pointer_row, objects=objects, source_rows=source_rows,
                raw=raw, backbone=backbone, erased=erased, gates=gates, mask=mask,
                powered=powered, beams=beams, selected=selected, rounds=rounds)


def _render(info, mask=None, powered=None, beams=None):
    g, bg, col = info['g'], info['bg'], info['col']
    out = g.copy()
    out[:, :col] = bg
    for obj in info['objects']:
        for r, c in obj['cells']:
            out[r, c + obj['shift']] = info['plate']
    view = out[:, col+1:]
    view[info['raw']] = bg
    mask = info['mask'] if mask is None else mask
    powered = info['powered'] if powered is None else powered
    beams = info['beams'] if beams is None else beams
    view[mask] = info['wire']
    view[powered] = info['power']
    for r, c in beams:
        view[r, c] = info['power']
    return np.rot90(out, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, cavities, gates, and the least powered closure from pixels."""
    return _render(_analyze(input_grid))


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _plate_mask(rng, kind, h, max_width):
    w = int(rng.integers(3, min(7, max_width) + 1))
    m = np.ones((h, w), bool)
    if kind != 'solid':
        if kind == 'active' and h >= 5 and rng.random() < .45:
            spans = [(1, h//2), (h//2+1, h-1)]
        else:
            spans = [(1, h-1)]
        for a, b in spans:
            for r in range(a, b):
                # Stepped backs change area without changing the row aperture.
                left = int(rng.integers(1, w-1)) if rng.random() < .35 else 1
                m[r, left:-1] = False
        if kind == 'open':
            side = int(rng.integers(3))
            if side == 0:
                m[1:-1, 0] = False
                m[1:-1, 1:-1] = False
            else:
                m[0 if side == 1 else -1, -2] = False
        if kind == 'recessed':
            length = int(rng.integers(1, min(3, max_width-w) + 1)) if w < max_width else 1
            m = np.pad(m, ((0, 0), (0, length)))
            m[int(rng.choice([0, h-1])), w:] = True
        elif kind == 'active' and w < max_width and rng.random() < .4:
            m = np.pad(m, ((0, 0), (1, 0)))
            m[int(rng.choice([0, h-1])), 0] = True
    elif rng.random() < .65:
        spine = int(rng.integers(w))
        for r in range(h):
            m[r, :int(rng.integers(spine+1))] = False
            m[r, int(rng.integers(spine, w))+1:] = False
    rr, cc = np.nonzero(m)
    return m[rr.min():rr.max()+1, cc.min():cc.max()+1]


def _tree(rng, rows, cols, root):
    parent, stack = {root: None}, [root]
    edges = []
    while stack:
        node = stack[-1]
        r, c = node
        options = [(nr, nc) for nr, nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                   if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in parent]
        if not options:
            stack.pop()
            continue
        nxt = options[int(rng.integers(len(options)))]
        parent[nxt] = node
        edges.append((node, nxt))
        stack.append(nxt)
    paths = []
    for node in parent:
        path = []
        while parent[node] is not None:
            path.append((parent[node], node))
            node = parent[node]
        paths.append(path[::-1])
    return edges, paths


def _gate_damage(rng, clean, edges, coords, statuses):
    raw = clean.copy()
    damaged = []
    for edge, status in zip(edges, statuses):
        if not status:
            continue
        a, b = sorted((coords[edge[0]], coords[edge[1]]))
        dr, dc = (0, 1) if a[0] == b[0] else (1, 0)
        distance = abs(b[0]-a[0]) + abs(b[1]-a[1])
        start = int(rng.integers(2, distance-3))
        gap = tuple((a[0]+k*dr, a[1]+k*dc) for k in range(start, start+3))
        for r, c in gap:
            raw[r, c] = False
        damaged.append((gap, dr, dc, status))
    spurs = []
    for gap, dr, dc, status in damaged:
        if status != 1:
            continue
        options = []
        for endpoint in (0, 4):
            r, c = gap[0][0]-dr+endpoint*dr, gap[0][1]-dc+endpoint*dc
            toward = 1 if endpoint == 0 else -1
            for side in (-1, 1):
                sr, sc = toward*dr+side*dc, toward*dc+side*dr
                spur = [(r+k*sr, c+k*sc) for k in (1, 2)]
                if all(0 <= rr < raw.shape[0] and 0 <= cc < raw.shape[1]
                       and not raw[rr, cc] and not _neighbors(raw)[rr, cc] for rr, cc in spur):
                    if not _at(raw, r+3*sr, c+3*sc):
                        trial = raw.copy()
                        for rr, cc in spur:
                            trial[rr, cc] = True
                        erased = trial & ~_neighbors(trial)
                        if all(_diagonal(p) for p in _components(erased, True)):
                            options.append(spur)
        if not options:
            raise AssertionError('gate has no clear spur placement')
        spur = options[int(rng.integers(len(options)))]
        for r, c in spur:
            raw[r, c] = True
        spurs.append(frozenset(spur))
    backbone, erased, gates = _wires(raw)
    _require(len(gates) == len(spurs) and {g['spur'] for g in gates} == set(spurs),
             'accidental or missing gate')
    return raw, backbone, gates


def _metrics(info):
    candidate_starts = {g['start'] for g in info['gates']}
    unsupported_lit = sum(any(info['powered'][r+k*dr, c+k*dc] for k in (0, 4))
                          for r,c,dr,dc in _gaps(info['backbone'])
                          if (r,c,dr,dc) not in candidate_starts)
    return dict(depth=len(info['rounds']), repairs=len(info['selected']),
                dark_gates=len(info['gates'])-len(info['selected']), unsupported_lit=unsupported_lit,
                axes={info['gates'][i]['start'][2] for i in info['selected']},
                sources=len(info['source_rows']), objects=len(info['objects']))


def _build_pair(rng, palette, turn, depth, contrast, exact_depth):
    # Construct a varied graph with a powered chain and a branch behind an
    # intentional break. Pixel analysis checks the complete assembled scene.
    for attempt in range(160):
        try:
            nr = int(rng.choice([3, 4]))
            nc = int(rng.choice([2, 3]))
            ys = [int(rng.integers(3, 6))]
            for _ in range(nr-1):
                ys.append(ys[-1]+int(rng.integers(6, 9)))
            h = ys[-1]+int(rng.integers(4, 7))
            if h > 30:
                continue
            xs = [int(rng.integers(1, 3))]
            for _ in range(nc-1):
                xs.append(xs[-1]+int(rng.integers(6, 8)))
            ww = xs[-1]+int(rng.integers(3, 5))
            work = int(rng.integers(9, min(14, 30-ww)))
            coords = {(r,c):(ys[r],xs[c]) for r in range(nr) for c in range(nc)}
            root = (int(rng.integers(nr)), 0)
            edges, paths = _tree(rng, nr, nc, root)
            paths = [p for p in paths if len(p) >= depth+2]
            if not paths:
                continue
            path = paths[int(rng.integers(len(paths)))]
            cut = int(rng.integers(depth, len(path)-1))
            flag_indices = sorted(map(int, rng.choice(cut, depth, replace=False)))
            if contrast == 'open' and depth >= 2:
                axes = {path[i][0][0] == path[i][1][0] for i in flag_indices}
                if len(axes) < 2:
                    continue
            statuses = [0]*len(edges)  # 0 intact, 1 flagged gap, 2 deliberate gap
            for i in flag_indices:
                statuses[edges.index(path[i])] = 1
            statuses[edges.index(path[cut])] = 2
            statuses[edges.index(path[cut+1])] = 1
            for i, e in enumerate(edges):
                if e not in path and rng.random() < .22:
                    statuses[i] = int(rng.choice([1, 2]))
            # Optional cycles change graph topology, subject to the same gate
            # and dark-branch requirements as trees.
            all_edges = [((r,c),(r+dr,c+dc)) for r in range(nr) for c in range(nc)
                         for dr,dc in ((1,0),(0,1)) if r+dr < nr and c+dc < nc]
            occupied = {frozenset(e) for e in edges}
            if rng.random() < .3:
                options = [e for e in all_edges if frozenset(e) not in occupied]
                if options:
                    edges.append(options[int(rng.integers(len(options)))])
                    statuses.append(int(rng.choice([0, 1, 2])))
            clean = np.zeros((h, ww), bool)
            for e in edges:
                a, b = sorted((coords[e[0]], coords[e[1]]))
                if a[0] == b[0]:
                    clean[a[0], a[1]:b[1]+1] = True
                else:
                    clean[a[0]:b[0]+1, a[1]] = True
            ah = int(rng.integers(3, 8))
            ar = ys[root[0]]-ah//2
            active = _plate_mask(rng, 'active', ah, work-2)
            clean[ar+1:ar+ah-1, xs[0]] = True
            raw, backbone, gates = _gate_damage(rng, clean, edges, coords, statuses)
            bg, plate, power, wire = palette
            g = np.full((h, work+1+ww), bg, np.int8)
            g[:, work] = power
            g[0, work+1:work+3] = power
            plates = [(ar, active)]
            for kind in [contrast] + ([str(rng.choice(['open','recessed','solid']))] if rng.random() < .45 else []):
                ph = int(rng.integers(3, 7))
                m = _plate_mask(rng, kind, ph, work-2)
                places = [r for r in range(h-m.shape[0]+1) if all(
                    r+m.shape[0] < oldr or r > oldr+oldm.shape[0] for oldr,oldm in plates)]
                if not places:
                    if kind == contrast and len(plates) == 1:
                        raise AssertionError('mandatory contrast does not fit')
                    continue
                plates.append((int(rng.choice(places)), m))
            shifts = [int(rng.integers(0, work-m.shape[1]+1)) for r,m in plates]
            if not any(shifts):
                shifts[0] = 1
            for i, (r, m) in enumerate(plates):
                limit = work-m.shape[1]
                shift = shifts[i]
                start = work-m.shape[1]-shift
                g[r:r+m.shape[0], start:start+m.shape[1]][m] = plate
            g[:, work+1:][raw] = wire
            # Extra free diagonals are erased but must not accidentally encode
            # another gate or merge with a cue.
            for _ in range(int(rng.integers(4))):
                for _ in range(16):
                    r, c = int(rng.integers(h)), int(rng.integers(ww))
                    dc = int(rng.choice([-1,1]))
                    length = int(rng.choice([2,3]))
                    cells = [(r+k,c+k*dc) for k in range(length)]
                    trial = raw.copy()
                    if not all(0 <= rr < h and 0 <= cc < ww and not raw[rr,cc]
                               and not _neighbors(raw)[rr,cc] and g[rr,work+1+cc] == bg for rr,cc in cells):
                        continue
                    for rr,cc in cells:
                        trial[rr,cc] = True
                    try:
                        b, _, gs = _wires(trial)
                        if np.array_equal(b, backbone) and {x['start'] for x in gs} == {x['start'] for x in gates}:
                            raw = trial
                            for rr,cc in cells:
                                g[rr,work+1+cc] = wire
                            break
                    except AssertionError:
                        pass
            info = _analyze(g)
            met = _metrics(info)
            _require(met['depth'] == depth if exact_depth else met['depth'] >= depth, 'wrong cascade depth')
            _require(met['dark_gates'] >= 1 and met['unsupported_lit'] >= 1, 'missing gate contrasts')
            _require(len(met['axes']) == 2 if contrast == 'open' and depth >= 2 else True, 'missing axes')
            _require(any(o['open_rows'] for o in info['objects']) if contrast == 'open' else True, 'missing open cavity')
            _require(any(o['recessed'] and not o['active'] for o in info['objects'])
                     if contrast == 'recessed' else True, 'missing recessed cavity')
            # A beam should not appear to touch a dark wire sideways.
            for r,c in info['beams']:
                _require(all(not _at(info['mask'], rr,cc) or info['powered'][rr,cc]
                             for rr,cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))), 'beam brushes dark wire')
            return {'input': np.rot90(g, turn).copy(), 'output': np.rot90(_render(info), turn).copy()}
        except (AssertionError, ValueError):
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _silence(pair):
    """Open every active chamber to the exterior without disconnecting its plate."""
    info = _analyze(pair['input'])
    g = info['g'].copy()
    for obj in info['objects']:
        r0 = obj['row']
        c0 = min(c for r,c in obj['cells'])
        for hole in obj['active']:
            row = min(r for r,c in hole)
            left = min(c for r,c in hole if r == row)
            g[r0+row, c0:c0+left] = info['bg']
    inp = np.rot90(g, -info['turn']).copy()
    quiet = _analyze(inp)
    _require(not quiet['source_rows'], 'silent conversion left a cavity source')
    return {'input': inp, 'output': _render(quiet)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    t = _count(num_test, 'num_test', 1, 4)
    if n <= t:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    base_turn = int(rng.integers(4))
    mixed = rng.random() < .6
    varied_palette = rng.random() < .3
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', t)):
        for i in range(count):
            depth = (2 if i == 0 else int(rng.integers(1,3))) if split == 'train' else int(rng.integers(3,5))
            contrast = ('open' if i == 0 else 'recessed' if i == 1 else str(rng.choice(['open','recessed','solid']))) if split == 'train' else str(rng.choice(['open','recessed','solid']))
            roles = tuple(map(int, rng.choice(10,4,replace=False))) if varied_palette else palette
            turn = int(rng.integers(4)) if mixed else base_turn
            for _ in range(8):
                pair = _build_pair(rng, roles, turn, depth, contrast, split == 'train')
                if split == 'train' and i >= 2 and rng.random() < .25:
                    pair = _silence(pair)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train','test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train','test')), 'split type')
    n, t = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= t <= 4 and n > t, 'pair counts')
    seen, training = set(), []
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input','output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _analyze(inp)
            _require(np.array_equal(out, _render(info)), 'incomplete or incorrect output')
            ordered = sorted(info['objects'], key=lambda o:o['row'])
            _require(all(a['end']+1 < b['row'] for a,b in zip(ordered,ordered[1:])), 'plate band collision')
            # Check emitted beams, diagonal deletion, powered/dark distinction,
            # exact three-cell additions, and the preserved rail independently
            # of any builder metadata.
            canonical_out = np.rot90(out, info['turn'])
            _require(np.array_equal(canonical_out[:,info['col']], info['g'][:,info['col']]), 'rail changed')
            view = canonical_out[:,info['col']+1:]
            _require(np.all(view[info['powered']] == info['power']), 'powered wire not recolored')
            _require(np.all(view[info['mask'] & ~info['powered']] == info['wire']), 'dark wire changed')
            for i, gate in enumerate(info['gates']):
                _require(all(view[r,c] == (info['power'] if i in info['selected'] else info['bg'])
                             for r,c in gate['cells']), 'wrong gate state')
            if split == 'train':
                training.append(info)
    objects = [o for info in training for o in info['objects']]
    _require(any(o['active'] for o in objects), 'no cavity source evidence')
    _require(any(o['open_rows'] for o in objects), 'no open contrast')
    _require(any(o['recessed'] and not o['active'] for o in objects), 'no extreme-wall contrast')
    _require(any(o['shift'] > 0 for o in objects), 'no docking evidence')
    _require(any(_metrics(i)['depth'] >= 2 for i in training), 'no cascade evidence')
    _require({g['start'][2] for info in training for i,g in enumerate(info['gates'])
              if i in info['selected']} == {0,1}, 'both repair axes must be demonstrated')
    _require(any(_metrics(i)['dark_gates'] for i in training), 'no unpowered marked contrast')
    _require(any(_metrics(i)['unsupported_lit'] for i in training), 'no powered unmarked contrast')
    for mode in ('no_power','no_cues','no_cavities'):
        different = False
        for info in training:
            rows = set() if mode == 'no_cavities' else info['source_rows']
            mask, power, beams, _, _ = _propagate(info['backbone'], info['gates'], rows,
                                                ignore_power=mode == 'no_power', ignore_cues=mode == 'no_cues')
            different |= not np.array_equal(_render(info,mask,power,beams), _render(info))
        _require(different, f'{mode} ablation does not change any training answer')


def self_test():
    """Bounded 42-seed structural sweep, rule contrasts, equivariance, bad labels."""
    signatures, shapes, layouts, masks, palettes, directions = set(), set(), set(), set(), set(), set()
    object_counts, hole_counts, depths, topology, spur_choices = set(), set(), set(), set(), set()
    moves, clutter_counts = set(), set()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        if seed == 0:
            sample = puzzle
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _analyze(pair['input'])
                _require(np.array_equal(solve(pair['input']),pair['output']), 'solve regression')
                shapes.add(pair['input'].shape)
                palettes.add((info['bg'],info['plate'],info['power'],info['wire']))
                directions.add(info['turn'])
                layouts.add(_key(info['backbone']))
                object_counts.add(len(info['objects']))
                depths.add(len(info['rounds']))
                for obj in info['objects']:
                    masks.add(_key(obj['mask']))
                    moves.add(obj['shift'])
                    hole_counts.add(len(obj['active']))
                clutter_counts.add(len(_components(info['erased'],True))-len(info['gates']))
                for gate in info['gates']:
                    spur_choices.add((gate['start'][2],gate['endpoint'],gate['side']))
                clean = info['backbone'].copy()
                for gate in info['gates']:
                    for r,c in gate['cells']:
                        clean[r,c] = True
                # Cycle count per cleaned connected component, with unsupported
                # breaks kept open, extracted from actual pixel adjacency.
                cycles = 0
                for cells in _components(clean):
                    cs = set(cells)
                    edges = sum((r+1,c) in cs for r,c in cs)+sum((r,c+1) in cs for r,c in cs)
                    cycles += edges-len(cs)+1
                topology.add(cycles > 0)
                _require(not (info['mask'][:-1,:-1] & info['mask'][1:,:-1]
                              & info['mask'][:-1,1:] & info['mask'][1:,1:]).any(), 'wire thickness')
        _require(max(len(_analyze(p['input'])['rounds']) for p in puzzle['train']) <= 2, 'training shift')
        _require(min(len(_analyze(p['input'])['rounds']) for p in puzzle['test']) >= 3, 'test shift')
    _require(len(signatures) == 42, 'puzzle diversity')
    _require(len(shapes) >= 50 and len(layouts) >= 150 and len(masks) >= 70, 'geometry diversity')
    _require(len(palettes) >= 30 and directions == {0,1,2,3}, 'palette or orientation diversity')
    _require({2,3}.issubset(object_counts) and {0,1,2}.issubset(hole_counts), 'counts collapsed')
    _require({0,1,2,3,4}.issubset(depths) and topology == {False,True}, 'propagation/topology variation')
    _require(len(spur_choices) == 8 and len(moves) >= 7 and len(clutter_counts) >= 3, 'cue/motion variation')
    # Rotation and reflection of complete scenes preserve the rule.
    for p in sample['train']+sample['test']:
        for k in range(4):
            for flip in (False,True):
                inp, out = np.rot90(p['input'],k),np.rot90(p['output'],k)
                if flip:
                    inp,out = inp[:,::-1],out[:,::-1]
                _require(np.array_equal(solve(inp),out), 'equivariance')
    _require(tuple(_key(p['input']) for p in generate_puzzle(123,7,3)['train']) ==
             tuple(_key(p['input']) for p in generate_puzzle(123,7,3)['train']), 'determinism')
    for n,t in ((2,1),(10,4),(10,1)):
        p = generate_puzzle(400+n,n,t)
        _require(len(p['train']) == n and len(p['test']) == t, 'explicit counts')
    for n,t in ((1,1),(11,1),(2,2),(3,4),(4,0),(6,5),(True,1),(3.5,1),(4,None)):
        try:
            generate_puzzle(0,n,t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Partial labels must fail: removing just one repair or falsely repairing a
    # dark marked gap is a substantive error even if other edits are correct.
    for mutation in ('omit','dark','plate'):
        p = {s:[{k:np.array(v,copy=True) for k,v in pair.items()} for pair in sample[s]] for s in ('train','test')}
        pair = p['train'][0]
        info = _analyze(pair['input'])
        out = np.rot90(pair['output'], info['turn']).copy()
        if mutation in ('omit','dark'):
            idx = next(i for i in range(len(info['gates'])) if (i in info['selected']) == (mutation == 'omit'))
            for r,c in info['gates'][idx]['cells']:
                out[r,info['col']+1+c] = info['bg'] if mutation == 'omit' else info['power']
        else:
            r,c = info['objects'][0]['cells'][0]
            out[r,c+info['objects'][0]['shift']] = info['bg']
        pair['output'] = np.rot90(out,-info['turn']).copy()
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted a corrupt label')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, cavity/gate ablations, cascades, equivariance, and bad labels')
