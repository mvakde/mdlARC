"""Pull marked cable junctions to an enclosure wall and splice their old arms.

The input carries every palette role and direction in a geometric legend.
Construction renders explicit latent walks; solve reconstructs only input cells.
"""
import numbers
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _cells(mask):
    return {tuple(map(int, p)) for p in np.argwhere(mask)}


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, todo = {p}, [p]
        while todo:
            p = todo.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _turns(walk):
    ds = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _visible_interaction(edits):
    # An intersection consisting only of a destination is hidden by the token
    # drawn last, and cannot teach simultaneous cable editing.
    ends = {e['end'] for e in edits}
    return any((e['new'] & f['old']) - ends for i, e in enumerate(edits)
               for j, f in enumerate(edits) if i != j)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def _read(input_grid):
    """Identify the isolated legend, its role colors, walls, and enclosed cells."""
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > a.size, "background must be a strict majority")
    keys = []
    for component in _components(_cells(a != bg)):
        rows, cols = zip(*component)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        sub = a[r0:r1 + 1, c0:c1 + 1]
        if min(sub.shape) < 4 or len(component) != sub.size:
            continue
        carrier = int(sub[0, 0])
        if not all(np.all(edge == carrier) for edge in (sub[0], sub[-1], sub[:, 0], sub[:, -1])):
            continue
        other = set(map(int, np.unique(sub))) - {carrier}
        if len(other) != 2:
            continue
        interpretations = []
        for token in sorted(other):
            payload = next(iter(other - {token}))
            tips, trails = _cells(sub == token), _cells(sub == payload)
            if len(tips) < 2:
                continue
            for d in DIRS:
                used, lengths = set(), []
                for tip in sorted(tips):
                    q, stroke = _add(tip, (-d[0], -d[1])), []
                    while q in trails:
                        stroke.append(q)
                        q = _add(q, (-d[0], -d[1]))
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((payload, token, d, tuple(lengths)))
        if len(interpretations) == 1:
            payload, token, direction, lengths = interpretations[0]
            keys.append(dict(carrier=carrier, payload=payload, token=token,
                             direction=direction, lengths=lengths, key=component,
                             key_shape=sub.shape, bbox=(r0, r1, c0, c1)))
    _require(len(keys) == 1, "exactly one uniquely decoded legend is required")
    info = keys[0]
    _require(len({bg, info['carrier'], info['payload'], info['token']}) == 4, "role overlap")
    outside = set(map(int, np.unique(a))) - {bg, info['carrier'], info['payload'], info['token']}
    _require(len(outside) <= 1, "unexplained annotation colors")
    repair = next(iter(outside)) if outside else None
    wall = _cells(a == info['carrier']) - info['key']
    _require(wall, "missing enclosing wall")
    free = {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])
            if (r, c) not in wall and (r, c) not in info['key']}
    enclosed = set()
    regions = []
    for component in _components(free):
        if any(p[0] in (0, a.shape[0] - 1) or p[1] in (0, a.shape[1] - 1) for p in component):
            continue
        boundary = {_add(p, d) for p in component for d in DIRS if _add(p, d) not in component}
        if boundary <= wall:
            enclosed.update(component)
            regions.append(component)
    cable_roles = [info['payload'], info['token']] + ([] if repair is None else [repair])
    cable_sites = _cells(np.isin(a, cable_roles)) - info['key']
    _require(cable_sites <= enclosed, "cable or annotation outside an enclosed region")
    tokens = _cells(a == info['token']) - info['key']
    _require(tokens, "missing junction tokens")
    info.update(grid=a, background=bg, repair=repair, wall=wall,
                free=enclosed, regions=regions, tokens=tokens)
    return info


def _reconstruct(info, fill_gaps=True):
    a = info['grid']
    roles = [info['payload'], info['token']]
    if info['repair'] is not None:
        roles.append(info['repair'])
    observed = _cells(np.isin(a, roles)) - info['key']
    cells = set(observed)
    # Short straight omissions have two fragments that continue away from the
    # hole. Use a frozen observation set, so filling cannot invent extra evidence.
    if fill_gaps:
        for p in sorted(observed):
            for d in ((0, 1), (1, 0)):
                if _add(p, (-d[0], -d[1])) not in observed:
                    continue
                for length in (2, 3):
                    q = p[0] + length*d[0], p[1] + length*d[1]
                    if q not in observed or _add(q, d) not in observed:
                        continue
                    between = {(p[0] + j*d[0], p[1] + j*d[1]) for j in range(1, length)}
                    if between <= info['free']:
                        cells.update(between)
    return cells


def _edit_geometry(info, cells):
    d = info['direction']
    edits = []
    components = _components(cells)
    _require(all(len(c & info['tokens']) <= 1 for c in components), "multiple tokens on one initial component")
    for pivot in sorted(info['tokens']):
        _require(pivot in cells, "missing pivot")
        old_start = _add(pivot, (-d[0], -d[1]))
        perpendicular = (-d[1], d[0])
        _require(old_start in cells and _add(pivot, d) not in cells, "ambiguous selected arm")
        _require(any(_add(pivot, side) in cells for side in
                     (perpendicular, (-perpendicular[0], -perpendicular[1]))), "pivot needs a retained arm")
        previous, current, old = pivot, old_start, []
        while True:
            _require(current not in old, "old arm contains a cycle")
            old.append(current)
            following = [q for q in _neighbors(current, cells) if q != previous]
            _require(len(following) <= 1, "selected arm branches before its terminal")
            if not following:
                break
            previous, current = current, following[0]
        _require(any(_add(current, side) in info['wall'] for side in DIRS), "old terminal must reach a wall")
        new, q = [], _add(pivot, d)
        while q in info['free']:
            new.append(q)
            q = _add(q, d)
        _require(new and q in info['wall'], "pull must stop at the enclosure wall")
        edits.append(dict(pivot=pivot, old=set(old), walk=[pivot] + old,
                          new=set(new), ray=new, end=new[-1]))
    old_union = set().union(*(e['old'] for e in edits))
    _require(old_union.isdisjoint(info['tokens']), "one cut deletes another pivot")
    _require(len({e['end'] for e in edits}) == len(edits), "shared token destination")
    _require(all(e['new'].isdisjoint(info['tokens']) for e in edits), "ray crosses an original token")
    for i, e in enumerate(edits):
        for f in edits[i + 1:]:
            _require(e['new'].isdisjoint(f['new']), "overlapping parallel pull rays")
    return edits


def _render(info, cells, edits, delete=True):
    out = info['grid'].copy()
    for p in info['free']:
        out[p] = info['background']
    kept = set(cells)
    if delete:
        for edit in edits:
            kept.difference_update(edit['old'])
    for edit in edits:
        kept.update(edit['new'])
    for p in kept:
        out[p] = info['payload']
    for edit in edits:
        out[edit['end']] = info['token']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution. No generation state or demonstrations are consulted."""
    info = _read(input_grid)
    cells = _reconstruct(info)
    return _render(info, cells, _edit_geometry(info, cells))


def _make_key(rng, carrier, payload, token):
    # Canonical east-pointing strokes, later rotated with the whole scene.
    strokes = int(rng.integers(2, 4))
    lengths = [int(rng.integers(2, 5)) for _ in range(strokes)]
    offsets = [int(rng.integers(0, 2)) for _ in range(strokes)]
    width = max(n + off for n, off in zip(lengths, offsets)) + 3
    key = np.full((2*strokes + 1, width), carrier, dtype=np.int8)
    for i, (n, off) in enumerate(zip(lengths, offsets)):
        key[1 + 2*i, 1 + off:1 + off + n] = payload
        key[1 + 2*i, 1 + off + n] = token
    return key


def _enclosure(rng, h, w, rich):
    mask = np.ones((h, w), bool)
    kind = int(rng.integers(4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, h//3 + 1)), int(rng.integers(1, w//3 + 1))
            mask[slice(0, dh) if corner < 2 else slice(h-dh, h),
                 slice(0, dw) if corner % 2 == 0 else slice(w-dw, w)] = False
    elif kind == 2:
        bevel = int(rng.integers(2, max(3, min(h, w)//3)))
        for r in range(h):
            cut = max(0, bevel - min(r, h - 1 - r))
            if cut:
                mask[r, :cut] = mask[r, w-cut:] = False
    elif kind == 3:
        left, right = 0, 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, w//4))
            right = int(np.clip(right + rng.integers(-1, 2), 0, w//4))
            mask[r, :left] = False
            if right:
                mask[r, w-right:] = False
    padded = np.pad(mask, 1)
    free = mask.copy()
    for dr, dc in DIRS:
        free &= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
    wall = mask & ~free
    for _ in range(int(rng.integers(0, 1 + rich))):
        horizontal = bool(rng.integers(2))
        if horizontal:
            r = int(rng.integers(3, h-3))
            length = int(rng.integers(2, max(3, w//3)))
            sl = slice(0, length) if rng.random() < .5 else slice(w-length, w)
            wall[r, sl] |= mask[r, sl]
        else:
            c = int(rng.integers(3, w-3))
            length = int(rng.integers(2, max(3, h//3)))
            sl = slice(0, length) if rng.random() < .5 else slice(h-length, h)
            wall[sl, c] |= mask[sl, c]
    free = mask & ~wall
    return mask, wall, free


def _random_walk(rng, free, occupied):
    """A self-avoiding polyline starting at a free cell beside the wall."""
    boundary = sorted(p for p in free if any(_add(p, d) not in free for d in DIRS)
                      and p not in occupied and not _neighbors(p, occupied))
    if not boundary:
        return None
    start = boundary[int(rng.integers(len(boundary)))]
    points, walk = [start], [start]
    axis = int(rng.integers(2))
    for _ in range(int(rng.integers(3, 8))):
        p, choices = points[-1], []
        used = set(walk)
        for sign in (-1, 1):
            d = (sign, 0) if axis == 0 else (0, sign)
            segment, q = [], p
            for length in range(1, 11):
                q = _add(q, d)
                if q not in free or q in used or q in occupied or _neighbors(q, occupied):
                    break
                adjacent = set(_neighbors(q, used))
                if adjacent - ({p} if length == 1 else set()):
                    break
                segment.append(q)
                if length >= 2:
                    choices.append((q, list(segment)))
        if not choices:
            break
        q, segment = choices[int(rng.integers(len(choices)))]
        points.append(q)
        walk.extend(segment)
        axis = 1 - axis
    return points, walk


def _scene(rng, wall, free_mask, count, min_bends, interaction, dual_terminal=False):
    free = _cells(free_mask)
    wall_cells = _cells(wall)
    occupied, paths, edits = set(), [], []
    for index in range(count):
        found = None
        for _ in range(140):
            candidate = _random_walk(rng, free, occupied)
            if candidate is None:
                continue
            points, walk = candidate
            if dual_terminal and index == 0 and not any(
                    _add(walk[-1], d) in wall_cells for d in DIRS):
                continue
            pivots = []
            for j in range(1, len(points) - 1):
                p, prev = points[j], points[j-1]
                if p[0] != prev[0] or p[1] <= prev[1]:
                    continue  # All initial old arms point west; the key points east.
                at = walk.index(p)
                old = set(walk[:at])
                bends = _turns(list(reversed(walk[:at+1])))
                if index == 0 and bends < min_bends:
                    continue
                ray, q = [], (p[0], p[1]+1)
                while q in free:
                    ray.append(q)
                    q = (q[0], q[1]+1)
                if not ray or any(e['pivot'][0] == p[0] for e in edits):
                    continue
                if dual_terminal and index == 0:
                    retained_step = walk[at+1]
                    alternative = p[0]-retained_step[0], p[1]-retained_step[1]
                    if _add(p, alternative) not in free:
                        continue
                pivots.append(dict(pivot=p, old=old, walk=list(reversed(walk[:at+1])),
                                   new=set(ray), ray=ray, end=ray[-1], at=at))
            if pivots:
                found = walk, pivots[int(rng.integers(len(pivots)))]
                break
        if found is None:
            return None
        walk, edit = found
        occupied.update(walk)
        paths.append(walk)
        edits.append(edit)
        # Optional unequal third arm, independently of the selected old arm.
        if not (dual_terminal and index == 0) and rng.random() < .28:
            p = edit['pivot']
            for sign in rng.permutation([-1, 1]):
                d = int(sign), 0
                arm, q = [], p
                for j in range(1, int(rng.integers(3, 7))):
                    q = _add(q, d)
                    if q not in free or q in occupied or set(_neighbors(q, occupied)) - ({p} if j == 1 else set()):
                        break
                    arm.append(q)
                if len(arm) >= 2:
                    occupied.update(arm)
                    break
    if interaction and not _visible_interaction(edits):
        return None
    # Unmarked stable walks share the same environment and repair convention.
    for _ in range(int(rng.integers(0, 3))):
        for _ in range(12):
            candidate = _random_walk(rng, free, occupied)
            if candidate and len(candidate[1]) >= 4:
                occupied.update(candidate[1])
                paths.append(candidate[1])
                break
    return occupied, paths, edits


def _pack(rng, mask, wall, key, bg, carrier):
    h, w = mask.shape
    kh, kw = key.shape
    feasible = []
    for vertical in (False, True):
        minimum = (h + kh + 1, max(w, kw)) if vertical else (max(h, kh), w + kw + 1)
        if max(minimum) <= 30:
            feasible.append(minimum)
    if not feasible:
        return None
    minimum = feasible[int(rng.integers(len(feasible)))]
    shape = tuple(int(rng.integers(n, 31)) for n in minimum)
    # Independent object placement; the key may occupy any gap around the wall.
    for _ in range(24):
        r, c = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
        blocked = np.zeros(shape, bool)
        blocked[r:r+h, c:c+w] = mask
        pad = np.pad(blocked, 1)
        expanded = blocked.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded |= pad[1+dr:1+dr+shape[0], 1+dc:1+dc+shape[1]]
        windows = np.lib.stride_tricks.sliding_window_view(expanded, key.shape)
        sites = np.argwhere(~np.any(windows, axis=(-1, -2)))
        if len(sites):
            kr, kc = map(int, sites[int(rng.integers(len(sites)))])
            grid = np.full(shape, bg, dtype=np.int8)
            grid[r:r+h, c:c+w][wall] = carrier
            grid[kr:kr+kh, kc:kc+kw] = key
            return grid, (r, c)
    return None


def _make_pair(rng, colors, count, direction, min_bends=0, interaction=False, witness=False, rich=1):
    bg, carrier, payload, token, repair = colors
    for attempt in range(260):
        low = 12 if count == 1 else (16 if count == 2 else 19)
        h, w = int(rng.integers(low, 23)), int(rng.integers(low, 23))
        mask, wall, free = _enclosure(rng, h, w, rich)
        scene = _scene(rng, wall, free, count, min_bends, interaction,
                       dual_terminal=witness and count == 1)
        if scene is None:
            continue
        latent, paths, edits = scene
        key = _make_key(rng, carrier, payload, token)
        packed = _pack(rng, mask, wall, key, bg, carrier)
        if packed is None:
            continue
        inp, offset = packed
        def global_cell(p):
            return p[0] + offset[0], p[1] + offset[1]
        cells = {global_cell(p) for p in latent}
        ge = [{k: (global_cell(v) if k in ('pivot', 'end') else
                    [global_cell(p) for p in v] if k in ('walk', 'ray') else
                    {global_cell(p) for p in v}) for k, v in e.items() if k != 'at'} for e in edits]
        for p in cells:
            inp[p] = payload
        for e in ge:
            inp[e['pivot']] = token
        try:
            info = _read(inp)
            if _reconstruct(info) != cells:
                continue
            _edit_geometry(info, cells)
        except AssertionError:
            continue
        # Independent construction label: render the explicit walks and stored
        # old-arm prefixes, then the wall-limited replacement segments.
        out = inp.copy()
        remaining = set(cells)
        for e in ge:
            remaining.difference_update(e['old'])
        for p in cells:
            out[p] = bg
        for e in ge:
            remaining.update(e['new'])
        for p in remaining:
            out[p] = payload
        for e in ge:
            out[e['end']] = token
        # Waypoint annotations may occupy bends, endpoints, and either arm.
        all_old = set().union(*(e['old'] for e in ge))
        retained = cells - all_old - {e['pivot'] for e in ge}
        old_candidates = sorted(all_old)
        kept_candidates = sorted(retained)
        rng.shuffle(old_candidates)
        rng.shuffle(kept_candidates)
        sites = []
        if witness and old_candidates and kept_candidates:
            sites = [old_candidates[0], kept_candidates[0]]
        pool = sorted(cells - {e['pivot'] for e in ge} - set(sites))
        rng.shuffle(pool)
        desired = max(len(sites), int(rng.integers(0 if not witness else 2, 9)))
        sites += pool[:max(0, desired - len(sites))]
        for p in sites:
            inp[p] = repair
        # Omit one or two consecutive cells only when reconstructible under
        # the documented, local straight-gap witness. Reject unintended joins.
        gap_candidates = sorted(cells - set(sites) - {e['pivot'] for e in ge})
        rng.shuffle(gap_candidates)
        if witness:
            gap_candidates.sort(key=lambda p: p not in retained)
        desired_gaps = int(rng.integers(1 if witness else 0, 5))
        gaps = []
        for p in gap_candidates:
            if len(gaps) >= desired_gaps:
                break
            old_value = int(inp[p])
            inp[p] = bg
            try:
                ok = _reconstruct(_read(inp)) == cells
            except AssertionError:
                ok = False
            if not ok:
                inp[p] = old_value
            else:
                gaps.append(p)
        if witness and not any(p in retained for p in gaps):
            continue
        k = (1 - DIRS.index(direction)) % 4
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        try:
            info = _read(inp)
            repaired = _reconstruct(info)
            actual_edits = _edit_geometry(info, repaired)
            if not np.array_equal(_render(info, repaired, actual_edits), out):
                continue
            if witness:
                removed = set().union(*(e['old'] for e in actual_edits))
                marks = _cells(inp == repair)
                if not (marks & removed and marks - removed):
                    continue
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError("bounded cable-scene construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    token, repair = palette[:2]
    training_dirs = [DIRS[int(i)] for i in rng.choice(4, 2, replace=False)]
    query_dirs = [d for d in DIRS if d not in training_dirs]
    shift = int(rng.integers(2))
    plans = [(1, training_dirs[0], 1, False, True),
             (2, training_dirs[1], 1, True, True)]
    plans += [(int(rng.integers(1, 3)), training_dirs[int(rng.integers(2))],
               int(rng.integers(0, 3)), False, False) for _ in range(nt - 2)]
    plans += [(3 if shift == 0 else int(rng.integers(1, 3)),
               (training_dirs if shift == 0 and rng.random() < .5 else query_dirs)[int(rng.integers(2))],
               2 if shift == 1 else 1, False, False) for _ in range(nq)]
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for i, (count, direction, bends, interaction, witness) in enumerate(plans):
        local = [c for c in range(10) if c not in (token, repair)]
        if previous_bg is not None:
            local.remove(previous_bg)
        bg = int(rng.choice(local))
        other = [c for c in range(10) if c not in (token, repair, bg)]
        carrier, payload = map(int, rng.choice(other, 2, replace=False))
        for _ in range(8):
            pair = _make_pair(rng, (bg, carrier, payload, token, repair), count,
                              direction, bends, interaction, witness, rich=1 + int(i >= nt))
            fp = _fingerprint(pair['input'])
            if fp not in seen:
                break
        _require(fp not in seen, "duplicate input construction")
        seen.add(fp)
        previous_bg = bg
        puzzle['train' if i < nt else 'test'].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, cells, edits):
    old = set().union(*(e['old'] for e in edits))
    marks = set() if info['repair'] is None else _cells(info['grid'] == info['repair'])
    rows, cols = zip(*info['wall'])
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                 if r in (r0, r1) or c in (c0, c1)}
    key_choices = 0
    for direction in DIRS:
        try:
            _edit_geometry(dict(info, direction=direction), cells)
            key_choices += 1
        except AssertionError:
            pass
    return dict(direction=info['direction'], tokens=len(edits), bends=[_turns(e['walk']) for e in edits],
                distances=[len(e['ray']) for e in edits], key_shape=info['key_shape'],
                irregular=not perimeter <= info['wall'], components=len(_components(cells)),
                gaps=len(cells - _reconstruct(info, False)), repairs=len(marks),
                kept_gap=bool((cells - _reconstruct(info, False)) - old), key_choices=key_choices,
                old_repair=bool(marks & old), kept_repair=bool(marks - old),
                interaction=_visible_interaction(edits),
                junction=sum(len(_neighbors(e['pivot'], cells)) == 3 for e in edits))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "invalid split schema")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "invalid counts")
    seen, training, conventions = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "empty or resized transformation")
            fp = _fingerprint(inp)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _read(inp)
            cells = _reconstruct(info)
            edits = _edit_geometry(info, cells)
            # Initial cable components are trees with maximum degree three.
            for component in _components(cells):
                degrees = [len(_neighbors(p, cells)) for p in component]
                _require(max(degrees) <= 3 and sum(degrees) == 2*(len(component)-1), "initial cable must be an unambiguous tree")
            prediction = _render(info, cells, edits)
            _require(np.array_equal(prediction, out), "full output violates pull/splice rule")
            _require(np.array_equal(_render(info, cells, list(reversed(edits))), out), "order-dependent edits")
            _require(all(int(out[e['pivot']]) == info['payload'] for e in edits), "departed pivot not repaired")
            _require((_cells(out == info['token']) - info['key']) == {e['end'] for e in edits}, "token conservation")
            fixed = info['wall'] | info['key']
            _require(all(inp[p] == out[p] for p in fixed), "wall or key changed")
            _require(all(out[p] == inp[p] for p in np.ndindex(inp.shape) if p not in info['free']), "outside cell changed")
            conventions.add(info['token'])
            if info['repair'] is not None:
                conventions.add(('repair', info['repair']))
            if split == 'train':
                training.append(_features(info, cells, edits))
    _require(len({x for x in conventions if isinstance(x, int)}) == 1, "token convention changes")
    _require(len({x for x in conventions if isinstance(x, tuple)}) == 1, "repair convention changes")
    _require(len({f['direction'] for f in training}) >= 2, "direction-key polarity needs multiple directions")
    _require(any(max(f['bends']) >= 1 and f['old_repair'] and f['kept_repair'] for f in training), "missing bent deletion and repair witness")
    _require(any(f['interaction'] for f in training), "missing simultaneous splice witness")
    _require(any(f['key_choices'] >= 2 for f in training), "key must choose between geometrically legal arms")
    _require(any(f['kept_gap'] for f in training), "missing visible straight-gap repair witness")


def self_test():
    """Bounded 42-seed diversity, metamorphic, ablation, and corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, signatures, normalized, shapes, palettes = [], set(), set(), set(), set()
    for puzzle in puzzles:
        signatures.add(tuple(_fingerprint(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                info = _read(a)
                cells = _reconstruct(info)
                edits = _edit_geometry(info, cells)
                records.append(_features(info, cells, edits))
                shapes.add(a.shape)
                palettes.add((info['background'], info['carrier'], info['payload'], info['token']))
                roles = {}
                normalized.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                _require(np.array_equal(solve(a), b), "solve mismatch")
                # Omitting B's topological cut must visibly keep stale cable.
                if split == 'train':
                    _require(not np.array_equal(_render(info, cells, edits, delete=False), b), "deletion ablation is invisible")
                    # A token-only move cannot repair its source knot or splice
                    # the cable, even though its wall destination is correct.
                    move_only = a.copy()
                    for e in edits:
                        move_only[e['pivot']] = info['background']
                        move_only[e['end']] = info['token']
                    _require(not np.array_equal(move_only, b), "motion-only ablation is invisible")
                    if records[-1]['key_choices'] >= 2:
                        alternatives = []
                        for d in DIRS:
                            if d == info['direction']:
                                continue
                            try:
                                changed_info = dict(info, direction=d)
                                changed_edits = _edit_geometry(changed_info, cells)
                                alternatives.append(_render(changed_info, cells, changed_edits))
                            except AssertionError:
                                pass
                        _require(alternatives and all(not np.array_equal(v, b) for v in alternatives),
                                 "key direction does not change the selected arm")
                    if records[-1]['interaction']:
                        wrong_results = []
                        for order in (edits, list(reversed(edits))):
                            wrong_cells = set(cells)
                            for e in order:
                                wrong_cells.difference_update(e['old'])
                                wrong_cells.update(e['new'])
                            wrong = _render(info, wrong_cells, [], delete=False)
                            for e in edits:
                                wrong[e['end']] = info['token']
                            wrong_results.append(wrong)
                        _require(any(not np.array_equal(v, b) for v in wrong_results),
                                 "simultaneous interaction has no visible witness")
        # Each puzzle has at least one witness against A-free edge continuation.
        wall_limited = False
        for pair in puzzle['train']:
            info = _read(pair['input'])
            for e in _edit_geometry(info, _reconstruct(info)):
                beyond = _add(e['end'], info['direction'])
                wall_limited |= beyond in info['wall'] and _inside(_add(beyond, info['direction']), pair['input'].shape)
        _require(wall_limited, "no enclosure-vs-canvas stopping witness")
    _require(len(signatures) == 42 and len(normalized) >= 180, "structural seed diversity")
    _require(len(shapes) >= 40 and len(palettes) >= 100, "shape/palette diversity")
    _require({r['direction'] for r in records} == set(DIRS), "all cardinal directions")
    _require({r['tokens'] for r in records} == {1, 2, 3}, "junction-count diversity")
    _require({b for r in records for b in r['bends']} >= {0, 1, 2}, "old-arm bend diversity")
    _require(any(r['gaps'] for r in records) and any(r['junction'] for r in records), "gap and third-arm coverage")
    _require(any(r['irregular'] for r in records) and any(not r['irregular'] for r in records), "wall silhouette diversity")
    _require(len({r['key_shape'] for r in records}) >= 6, "key geometry diversity")
    for puzzle in puzzles[:3]:
        for k in range(4):
            for reflect in (False, True):
                for pair in puzzle['train'] + puzzle['test']:
                    a = np.rot90(np.fliplr(pair['input']) if reflect else pair['input'], k).copy()
                    b = np.rot90(np.fliplr(pair['output']) if reflect else pair['output'], k).copy()
                    _require(np.array_equal(solve(a), b), "rotation/reflection invariance")
        # A consistent arbitrary recoloring must preserve the geometric roles.
        permutation = np.array([8, 4, 7, 1, 5, 9, 0, 2, 6, 3], dtype=np.int8)
        for pair in puzzle['test']:
            _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), "palette invariance")
    _require(tuple(_fingerprint(p['input']) for p in generate_puzzle(17)['train']) ==
             tuple(_fingerprint(p['input']) for p in puzzles[17]['train']), "seed reproducibility")
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(900 + a + b, a, b)
        _require(len(p['train']) == a and len(p['test']) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = puzzles[0]
    pair = p['test'][0]
    info = _read(pair['input'])
    cells = _reconstruct(info)
    edits = _edit_geometry(info, cells)
    sites = [min(info['wall']), min(info['key']), edits[0]['pivot'], edits[0]['end'], min(edits[0]['old'])]
    for site in sites:
        bad = {s: [{'input': q['input'], 'output': q['output'].copy()} for q in p[s]] for s in p}
        bad['test'][0]['output'][site] = (int(pair['output'][site]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {'seeds': 42, 'pairs': len(records), 'structures': len(normalized),
            'shapes': len(shapes), 'palettes': len(palettes)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
