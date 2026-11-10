"""Leading cavities compete for a shared, wall-routed supply.

The solver infers all roles and the approach direction from the input. No
parent generator, random state, or training example is used by the solver.
"""
from collections import deque
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue, cells = [(r, c)], []
        for rr, cc in queue:
            cells.append((rr, cc))
            for dr, dc in STEPS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(cells)
    return result


def _scene(grid):
    """Canonicalize to work | rail | supply; rail color occurs nowhere else."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(2, w - 1):
            rail = int(a[0, col])
            if not np.all(a[:, col] == rail) or np.count_nonzero(a == rail) != h:
                continue
            work, field = a[:, :col], a[:, col + 1:]
            bg = int(work[0, 0])
            if bg == rail or not (np.all(work[0] == bg) and np.all(work[-1] == bg)
                                  and np.all(work[:, 0] == bg)):
                continue
            fg = set(map(int, np.unique(work))) - {bg}
            if len(fg) != 1:
                continue
            fg = fg.pop()
            payload = set(map(int, np.unique(field))) - {bg, fg, rail}
            if len(payload) != 1:
                continue
            payload = payload.pop()
            if set(map(int, np.unique(field))) <= {bg, fg, payload}:
                candidates.append((a, turn, bg, fg, rail, payload, col))
    _require(len(candidates) == 1, "input must identify one oriented rail and four roles")
    return candidates[0]


def _objects(a, fg, rail_col):
    objects, collectors, all_holes, column_holes = [], [], [], []
    for cells in _components(a[:, :rail_col] == fg):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        shift = rail_col - 1 - c1
        holes, open_regions = [], []
        for region in _components(~mask):
            boundary = any(r in (0, mask.shape[0]-1) or c in (0, mask.shape[1]-1)
                           for r, c in region)
            (open_regions if boundary else holes).append(region)
        obj = dict(cells=cells, mask=mask, row=r0, end=r1, shift=shift,
                   holes=holes, opens=open_regions, width=mask.shape[1])
        objects.append(obj)
        for region in holes + open_regions:
            contact = {r for r, c in region if c == mask.shape[1]-2 and mask[r, -1]}
            active = region in holes and bool(contact)
            record = dict(slots=[(r+r0, c+c0+shift) for r, c in region],
                          rows=sorted({r+r0 for r, c in region}),
                          capacity=len(region), active=active, object=len(objects)-1)
            if region in holes:
                all_holes.append(record)
            if contact:
                column_holes.append(record)
            if active:
                _require(contact == {r for r, c in region}, "partial-contact cavity is excluded")
                collectors.append(record)
    _require(objects, "no objects")
    for prev, obj in zip(objects, objects[1:]):
        _require(obj['row'] > prev['end'] + 1, "object bands must have a blank separating row")
    return objects, collectors, all_holes, column_holes


def _distance(wall, rows):
    h, w = wall.shape
    d = np.full((h, w), -1, dtype=np.int16)
    queue = deque()
    for r in rows:
        if not wall[r, 0]:
            d[r, 0] = 0
            queue.append((r, 0))
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and d[nr, nc] < 0:
                d[nr, nc] = d[r, c] + 1
                queue.append((nr, nc))
    return d


def _maps(wall, collectors, geometric=False):
    if not collectors:
        return np.empty((0, *wall.shape), dtype=np.int16), np.full(wall.shape, -1, dtype=int)
    if geometric:
        rr, cc = np.indices(wall.shape)
        ds = np.array([np.min(np.abs(rr[..., None]-np.array(x['rows'])), axis=2)+cc
                       for x in collectors], dtype=np.int16)
        ds[:, wall] = -1
    else:
        ds = np.array([_distance(wall, x['rows']) for x in collectors])
    safe = np.where(ds < 0, 32767, ds)
    best = safe.min(axis=0)
    owner = safe.argmin(axis=0).astype(int)
    # Equal shortest routes from different cavities are an unclaimed watershed.
    owner[(best == 32767) | (np.sum(safe == best, axis=0) != 1)] = -1
    return ds, owner


def _selected(supply, ds, owner, collectors, strict=True):
    selected = np.zeros_like(supply, dtype=bool)
    counts = []
    for i, collector in enumerate(collectors):
        cells = list(zip(*np.where(supply & (owner == i))))
        cells.sort(key=lambda p: (int(ds[i][p]), int(p[0]), int(p[1])))
        cap = collector['capacity']
        if strict and len(cells) > cap:
            _require(ds[i][cells[cap-1]] < ds[i][cells[cap]], "ambiguous capacity-frontier tie")
        chosen = cells[:cap]
        for p in chosen:
            selected[p] = True
        counts.append(len(chosen))
    return selected, counts


def _analyze(grid, mode='active', geometric=False, strict=True):
    a, turn, bg, fg, rail, payload, col = _scene(grid)
    objects, active, all_holes, column_holes = _objects(a, fg, col)
    collectors = {'active': active, 'all': all_holes, 'column': column_holes}[mode]
    field = a[:, col+1:]
    wall, supply = field == fg, field == payload
    ds, owner = _maps(wall, collectors, geometric)
    selected, counts = _selected(supply, ds, owner, collectors, strict)
    return dict(a=a, turn=turn, bg=bg, fg=fg, rail=rail, payload=payload, col=col,
                objects=objects, collectors=collectors, wall=wall, supply=supply,
                ds=ds, owner=owner, selected=selected, counts=counts)


def _render(info, input_shape):
    a, col = info['a'], info['col']
    result = a.copy()
    result[:, :col] = info['bg']
    for obj in info['objects']:
        for r, c in obj['cells']:
            result[r, c+obj['shift']] = info['fg']
    result[:, col+1:][info['selected']] = info['bg']
    # These IDs map canonical pixels to displayed row-major order. Thus filling
    # a rotated scene never silently rotates the serialization convention.
    display_ids = np.rot90(np.arange(a.size).reshape(input_shape), info['turn'])
    for collector, count in zip(info['collectors'], info['counts']):
        slots = sorted(collector['slots'], key=lambda p: int(display_ids[p]))
        for r, c in slots[:count]:
            result[r, c] = info['payload']
    return np.rot90(result, -info['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Align objects, route supply to leading cavities, collect and serialize."""
    grid = _grid(input_grid)
    return _render(_analyze(grid), grid.shape)


def _mask(rng, kind, height=None, width=None):
    h = int(rng.integers(3, 6)) if height is None else height
    w = int(rng.integers(3, 7)) if width is None else width
    m = np.ones((h, w), dtype=bool)
    if kind == 'solid':
        for r in range(1, h-1):
            m[r, :int(rng.integers(0, w-1))] = False
    elif kind == 'multi':
        # Randomly partition height among cavities, solid separators and outer
        # walls. This admits compact double holes and taller multirow holes.
        count = int(rng.integers(2, min(3, (h-1)//2)+1))
        extras = rng.multinomial(h-(2*count+1), np.full(2*count+1, 1/(2*count+1)))
        start = 1+int(extras[0])
        for i in range(count):
            length = 1+int(extras[2*i+1])
            back = int(rng.integers(1, w-1))
            for r in range(start, start+length):
                left = int(rng.integers(1, w-1)) if rng.random() < .35 else back
                m[r, left:-1] = False
            start += length+1+int(extras[2*i+2])
    else:
        left = int(rng.integers(1, w-1))
        for r in range(1, h-1):
            back = int(rng.integers(1, w-1)) if rng.random() < .35 else left
            m[r, back:-1] = False
        if kind == 'open':
            if rng.random() < .5:
                m[1:-1, 0] = False
                m[1:-1, 1:-1] = False
            else:
                m[0 if rng.random() < .5 else -1, -2] = False
        elif kind == 'recess':
            extension = int(rng.integers(1, 3))
            m = np.pad(m, ((0, 0), (0, extension)))
            m[0 if rng.random() < .5 else -1, -extension:] = True
    _require(len(_components(m)) == 1, "mask disconnected")
    return m


def _layout(rng, palette, masks, height=None):
    bg, fg, rail, payload = palette
    base = sum(m.shape[0] for m in masks) + 2*(len(masks)-1) + 2
    minimum = 8 if len(masks) == 1 else 12 if len(masks) == 2 else 18
    h = int(rng.integers(max(minimum, base), 31)) if height is None else height
    wide = max(m.shape[1] for m in masks)
    col = int(rng.integers(wide+2, min(14, wide+7)+1))
    fw = int(rng.integers(9, 30-col))
    a = np.full((h, col+1+fw), bg, dtype=np.int8)
    a[:, col] = rail
    extras = rng.multinomial(h-base, np.full(len(masks)+1, 1/(len(masks)+1)))
    row, bounds = 1+int(extras[0]), []
    shifts = []
    for i, m in enumerate(masks):
        max_shift = col-m.shape[1]-1
        shift = 0 if rng.random() < .25 else int(rng.integers(1, max_shift+1))
        shifts.append(shift)
        start = col-m.shape[1]-shift
        a[row:row+m.shape[0], start:start+m.shape[1]][m] = fg
        bounds.append((row, row+m.shape[0]-1))
        row += m.shape[0]+2+int(extras[i+1])
    if not any(shifts):
        # A positive move is useful evidence; shift the first object left once.
        r0, r1 = bounds[0]
        view = a[r0:r1+1, :col]
        loc = np.argwhere(view == fg)
        view[:] = bg
        view[loc[:, 0], loc[:, 1]-1] = fg
    return a, col, bounds


def _walls(rng, shape, rich):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    # Fingers impose long detours without isolating every collector's region.
    for _ in range(int(rng.integers(1, 4 if rich else 3))):
        c = int(rng.integers(2, w-2))
        if rng.random() < .5:
            wall[:int(rng.integers(h//3, h-2)), c] = True
        else:
            wall[int(rng.integers(2, 2*h//3)):, c] = True
    for _ in range(int(rng.integers(0, 5 if rich else 3))):
        if rng.random() < .5:
            r = int(rng.integers(1, h-1))
            left = int(rng.integers(1, w-2))
            wall[r, left:int(rng.integers(left+1, w+1))] = True
        else:
            r, c = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
            rh, cw = int(rng.integers(2, min(6, h-r))), int(rng.integers(2, min(6, w-c)))
            wall[r:r+rh+1, c] = wall[r:r+rh+1, c+cw] = True
            wall[r, c:c+cw+1] = wall[r+rh, c:c+cw+1] = True
            if rng.random() < .55:
                wall[r+int(rng.integers(1, rh)), c] = False
    return wall


def _pick(rng, cells, n, style='scattered', forced=()):
    cells = sorted(set(tuple(map(int, p)) for p in cells))
    chosen = list(dict.fromkeys(forced))
    pool = [p for p in cells if p not in chosen]
    _require(n <= len(cells) and set(chosen) <= set(cells), "invalid placement")
    if style != 'scattered' and pool:
        anchor = chosen[0] if chosen else pool[int(rng.integers(len(pool)))]
        jitter = {p: float(rng.random()) for p in pool}
        if style == 'line':
            axis = int(rng.integers(2))
            pool.sort(key=lambda p: (abs(p[axis]-anchor[axis]), abs(p[1-axis]-anchor[1-axis]), jitter[p]))
        else:
            pool.sort(key=lambda p: ((p[0]-anchor[0])**2+(p[1]-anchor[1])**2, jitter[p]))
        chosen.extend(pool[:n-len(chosen)])
    elif n > len(chosen):
        chosen.extend(pool[int(i)] for i in rng.choice(len(pool), n-len(chosen), replace=False))
    return chosen


def _place_owned(rng, ds, owner, collectors, evidence=False):
    placed, witnessed = [], False
    h, w = owner.shape
    rr, cc = np.indices(owner.shape)
    style = str(rng.choice(['scattered', 'compact', 'line']))
    for i in rng.permutation(len(collectors)):
        i = int(i)
        x = collectors[i]
        owned = owner == i
        cap = x['capacity']
        cells = list(zip(*np.where(owned)))
        manh = np.min(np.abs(rr[..., None]-np.array(x['rows'])), axis=2)+cc
        if evidence and not witnessed:
            # Force a capacity-saturated path-versus-geometric reversal.
            for idx in rng.permutation(len(cells)):
                p = cells[int(idx)]
                near = owned & (ds[i] <= ds[i][p])
                far = owned & (ds[i] > ds[i][p])
                reversal = far & (manh < manh[p])
                if near.sum() < cap or not reversal.any():
                    continue
                q = list(zip(*np.where(reversal)))[int(rng.integers(reversal.sum()))]
                placed += _pick(rng, zip(*np.where(near)), cap, style, [p])
                excess = min(int(rng.integers(1, 6)), int(far.sum()))
                placed += _pick(rng, zip(*np.where(far)), excess, style, [q])
                witnessed = True
                break
            if witnessed:
                continue
            # Retry geometry rather than populate this collector arbitrarily:
            # the evidence pair needs exactly one saturated reservoir and a
            # positively loaded, partially empty competitor.
            return [], False
        if not cells:
            continue
        if evidence and witnessed:
            # The other collector stays partially empty, so allocating a tie
            # to it would visibly change the answer. It also collects positive
            # supply, teaching ownership rather than one privileged collector.
            if cap < 2 or len(cells) < 1:
                return [], False
            count = int(rng.integers(1, min(cap-1, len(cells))+1))
            placed += _pick(rng, cells, count, style)
            continue
        excess_mode = rng.random() < .4 and len(cells) > cap+1
        if excess_mode:
            thresholds = [d for d in np.unique(ds[i][owned])
                          if np.count_nonzero(owned & (ds[i] <= d)) >= cap
                          and np.count_nonzero(owned & (ds[i] > d)) >= 1]
            if thresholds:
                d = int(rng.choice(thresholds))
                near, far = owned & (ds[i] <= d), owned & (ds[i] > d)
                placed += _pick(rng, zip(*np.where(near)), cap, style)
                placed += _pick(rng, zip(*np.where(far)), min(int(rng.integers(1, 6)), int(far.sum())), style)
                continue
        count = int(rng.integers(0, min(cap, len(cells))+1))
        placed += _pick(rng, cells, count, style)
    return placed, witnessed


def _make_pair(rng, palette, mode, turn):
    bg, fg, rail, payload = palette
    for attempt in range(160):
        if mode == 'contrast':
            h, w = int(rng.integers(4, 6)), int(rng.integers(4, 7))
            closed = np.ones((h, w), dtype=bool)
            closed[1:-1, 1:-1] = False
            opened = closed.copy()
            if rng.random() < .5:
                opened[1:-1, 0] = False
            else:
                opened[0 if rng.random() < .5 else -1, -2] = False
            ext = int(rng.integers(1, 3))
            recessed = np.pad(closed, ((0, 0), (0, ext)))
            recessed[0 if rng.random() < .5 else -1, -ext:] = True
            masks = [closed, opened, recessed]
            masks = [masks[int(i)] for i in rng.permutation(3)]
        elif mode == 'distance':
            masks = [_mask(rng, 'active', int(rng.integers(4, 6)), int(rng.integers(4, 7)))
                     for _ in range(2)]
        elif mode == 'silent':
            masks = [_mask(rng, str(rng.choice(['open', 'recess', 'solid'])))
                     for _ in range(int(rng.integers(1, 4)))]
        else:
            # Both splits can contain multi-cavity objects. Query scenes always
            # recombine this with another active object and a silent object.
            masks = [_mask(rng, 'multi', int(rng.choice([5, 6, 7, 8, 9])), int(rng.integers(3, 7))),
                     _mask(rng, 'active'),
                     _mask(rng, str(rng.choice(['open', 'recess', 'solid'])))]
            if rng.random() < .3 and sum(m.shape[0] for m in masks) <= 18:
                masks.append(_mask(rng, str(rng.choice(['active', 'open', 'solid'])), 3))
            masks = [masks[int(i)] for i in rng.permutation(len(masks))]
        a, col, bounds = _layout(rng, palette, masks)
        field = a[:, col+1:]
        if mode == 'contrast':
            wall = np.zeros(field.shape, dtype=bool)
            for first, second in zip(bounds, bounds[1:]):
                wall[int(rng.integers(first[1]+1, second[0])), :] = True
        else:
            wall = _walls(rng, field.shape, mode == 'mixed')
        field[wall] = fg
        objects, active, _, _ = _objects(a, fg, col)
        ds, owner = _maps(wall, active)
        if mode == 'contrast':
            x = active[0]
            owned = owner == 0
            load = int(rng.choice([2, 3]))
            cells = list(zip(*np.where(owned)))
            if len(cells) < load:
                continue
            # A connected two/three-cell supply explicitly distinguishes
            # counting pixels from counting payload components.
            placed = _pick(rng, cells, load, str(rng.choice(['compact', 'line'])))
            # Bait on both silent cavity/indentation port rows; full partitions
            # make it unreachable under the actual cavity rule.
            for obj in objects:
                if not obj['holes'] or not any(any(c == obj['width']-2 for r, c in hole)
                                              for hole in obj['holes']):
                    rows = [obj['row']+r for r in range(1, obj['mask'].shape[0]-1)]
                    bait = [(r, c) for r in rows for c in range(field.shape[1]) if not wall[r, c]]
                    placed += _pick(rng, bait, int(rng.integers(2, 5)))
        elif mode == 'silent':
            placed = _pick(rng, zip(*np.where(~wall)), int(rng.integers(1, 9)),
                           str(rng.choice(['compact', 'line', 'scattered'])))
        else:
            placed, witnessed = _place_owned(rng, ds, owner, active, mode == 'distance')
            safe = np.where(ds < 0, 32767, ds)
            best = safe.min(axis=0)
            ties = (owner < 0) & (best < 32767) & ~wall
            if mode == 'distance' and (not witnessed or not ties.any()):
                continue
            if ties.any():
                placed += _pick(rng, zip(*np.where(ties)), min(int(rng.integers(1, 5)), int(ties.sum())))
            unreachable = ~wall & (best == 32767)
            if unreachable.any():
                placed += _pick(rng, zip(*np.where(unreachable)), min(int(rng.integers(1, 6)), int(unreachable.sum())))
        if not placed:
            continue
        for p in placed:
            field[p] = payload
        grid = np.rot90(a, turn).copy()
        info = _analyze(grid)
        if mode == 'distance':
            f = _features(grid)
            if not (f['geometric_ablation'] and f['tie_ablation'] and f['row_ablation'] and all(f['counts'])):
                continue
        if mode == 'contrast':
            expected = _render(info, grid.shape)
            if np.array_equal(expected, _render(_analyze(grid, 'all', strict=False), grid.shape)):
                continue
            if np.array_equal(expected, _render(_analyze(grid, 'column', strict=False), grid.shape)):
                continue
        return {'input': grid, 'output': _render(info, grid.shape)}
    raise RuntimeError('could not construct the requested evidence after bounded attempts')


def _signature(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and less than num_train')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    base_turn = int(rng.integers(4))
    # Fixed and mixed orientations both occur. Local colors can vary because
    # the rail, reserved work border, and supply-only color identify the roles.
    mixed_turns, varying_colors = bool(rng.random() < .6), bool(rng.random() < .5)
    modes = ['contrast', 'distance'] + [str(rng.choice(['mixed', 'silent'], p=[.8, .2]))
                                      for _ in range(num_train-2)]
    modes = [modes[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen, evidence_caps = {'train': [], 'test': []}, set(), set()
    for split, requests in [('train', modes), ('test', ['mixed']*num_test)]:
        for mode in requests:
            for retry in range(20):
                colors = tuple(map(int, rng.choice(10, 4, replace=False))) if varying_colors else palette
                turn = int(rng.integers(4)) if mixed_turns else base_turn
                pair = _make_pair(rng, colors, mode, turn)
                sig = _signature(pair['input'])
                if split == 'train' and mode in ('contrast', 'distance'):
                    new_caps = {x['capacity'] for x in _analyze(pair['input'])['collectors']}
                    if evidence_caps and len(evidence_caps | new_caps) == 1:
                        continue
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    if split == 'train' and mode in ('contrast', 'distance'):
                        evidence_caps.update(new_caps)
                    break
            else:
                raise RuntimeError('duplicate inputs after bounded retries')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    info = _analyze(grid)
    ds, owner, supply = info['ds'], info['owner'], info['supply']
    safe = np.where(ds < 0, 32767, ds)
    best = safe.min(axis=0) if len(ds) else np.full(owner.shape, 32767)
    rr, cc = np.indices(owner.shape)
    reversal, excess, gaps, loads = False, 0, [], []
    for i, x in enumerate(info['collectors']):
        owned = supply & (owner == i)
        chosen = owned & info['selected']
        retained = owned & ~info['selected']
        loads.append(int(owned.sum()))
        excess += int(retained.sum())
        if retained.any():
            gaps.append(int(ds[i][retained].min()-ds[i][chosen].max()))
            metric = np.min(np.abs(rr[..., None]-np.array(x['rows'])), axis=2)+cc
            reversal |= bool(metric[chosen].max() > metric[retained].min())
    output = _render(info, np.asarray(grid).shape)
    ties = supply & (owner < 0) & (best < 32767)
    tie_ablation = False
    for i in range(len(info['collectors'])):
        alternative = owner.copy()
        alternative[ties & (ds[i] == best)] = i
        selected, alternative_counts = _selected(supply, ds, alternative, info['collectors'], strict=False)
        if not np.array_equal(selected, info['selected']) or alternative_counts != info['counts']:
            tie_ablation = True
    row_supply = np.zeros_like(supply)
    for i, x in enumerate(info['collectors']):
        for r in x['rows']:
            row_supply[r] |= supply[r] & (owner[r] == i)
    row_selected, row_counts = _selected(row_supply, ds, owner, info['collectors'], strict=False)
    row_ablation = not np.array_equal(row_selected, info['selected']) or row_counts != info['counts']
    selected_components = _components(info['selected'])
    return dict(shape=tuple(np.asarray(grid).shape), turn=info['turn'],
                palette=(info['bg'], info['fg'], info['rail'], info['payload']),
                objects=len(info['objects']), cavities=len(info['collectors']),
                capacities=tuple(x['capacity'] for x in info['collectors']),
                loads=tuple(loads), counts=tuple(info['counts']), excess=excess, gaps=tuple(gaps),
                ties=int((supply & (owner < 0) & (best < 32767)).sum()),
                unreachable=int((supply & (best == 32767)).sum()),
                reversal=reversal, wall_count=int(info['wall'].sum()),
                tie_ablation=tie_ablation,
                row_ablation=row_ablation,
                adjacent_selected=any(len(x) >= 2 for x in selected_components),
                max_path=int(ds.max()) if len(ds) else -1,
                shifts=tuple(x['shift'] for x in info['objects']),
                multi=any(sum(x['object'] == j for x in info['collectors']) >= 2
                          for j in range(len(info['objects']))),
                nonleading_ablation=not np.array_equal(output, _render(_analyze(grid, 'all', strict=False), np.asarray(grid).shape)),
                enclosure_ablation=not np.array_equal(output, _render(_analyze(grid, 'column', strict=False), np.asarray(grid).shape)),
                geometric_ablation=not np.array_equal(output, _render(_analyze(grid, geometric=True, strict=False), np.asarray(grid).shape)))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in features:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'shape changed')
            sig = _signature(a)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            info = _analyze(a)
            _require(np.array_equal(b, solve(a)), 'incorrect output')
            aligned_b = np.rot90(b, info['turn'])
            col, selected = info['col'], info['selected']
            _require(np.array_equal(aligned_b[:, col], info['a'][:, col]), 'rail changed')
            _require(np.array_equal(aligned_b[:, col+1:][~selected], info['a'][:, col+1:][~selected]),
                     'unselected field cells changed')
            _require(int(np.count_nonzero(aligned_b[:, :col] == info['fg'])) ==
                     int(np.count_nonzero(info['a'][:, :col] == info['fg'])), 'object pixels not conserved')
            _require(int(np.count_nonzero(b == info['payload'])) ==
                     int(np.count_nonzero(a == info['payload'])), 'payload not conserved')
            _require(int(np.count_nonzero(aligned_b[:, :col] == info['payload'])) == int(selected.sum()),
                     'reservoir fill does not equal collected load')
            f = _features(a)
            _require(all(g > 0 for g in f['gaps']), 'capacity gap not strict')
            features[split].append(f)
    train = features['train']
    _require(any(f['nonleading_ablation'] and f['enclosure_ablation'] for f in train),
             'training must distinguish recessed holes and open indentations')
    _require(any(f['reversal'] and f['excess'] and f['ties'] and f['tie_ablation']
                 and f['geometric_ablation'] and f['row_ablation'] and all(f['counts'])
                 and any(0 < n < c for n, c in zip(f['counts'], f['capacities'])) for f in train),
             'training must show geodesic competition, unclaimed ties and capacity excess')
    _require(len({c for f in train for c in f['capacities']}) >= 2, 'training must show variable capacity')
    _require(any(any(1 < n < c for n, c in zip(f['counts'], f['capacities'])) for f in train),
             'training must show a partial row-major cavity fill')
    _require(any(f['adjacent_selected'] for f in train), 'training must distinguish cell and component counts')
    _require(any(any(d > 0 for d in f['shifts']) for f in train), 'training lacks movement')
    _require(all(f['multi'] and f['cavities'] >= 3 for f in features['test']),
             'queries must recombine multiple collectors within one object')


def self_test():
    """Bounded input-derived audits, 42 seeds, count contracts and mutations."""
    signatures, shapes, structures, palettes, turns, caps = set(), set(), set(), set(), set(), set()
    counts, paths, shifts, objects = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                f = _features(p['input'])
                shapes.add(f['shape']); palettes.add(f['palette']); turns.add(f['turn'])
                caps.update(f['capacities']); counts.update(f['counts'])
                paths.add(f['max_path']); shifts.update(f['shifts']); objects.add(f['objects'])
                roles, norm = {}, []
                for row in p['input']:
                    norm.append(tuple(roles.setdefault(int(c), len(roles)) for c in row))
                structures.add(tuple(norm))
    _require(len(signatures) == 42, 'seed variation collapsed')
    _require(len(structures) >= 150 and len(shapes) >= 50, 'structural variation collapsed')
    _require(len(palettes) >= 30 and turns == {0, 1, 2, 3}, 'palette/direction variation collapsed')
    _require(len(caps) >= 8 and len(counts) >= 7 and len(paths) >= 15, 'cavity/load/path variation collapsed')
    _require(0 in shifts and max(shifts) >= 5 and {2, 3, 4} <= objects, 'layout variation collapsed')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(823+nt*7+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'counts ignored')
    for nt, nq in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)]:
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(927, 2, 1), generate_puzzle(927, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'not reproducible')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Capacity-frontier ambiguity is refused, rather than silently using scan order.
    supply = np.ones((1, 3), dtype=bool)
    try:
        _selected(supply, np.array([[[0, 1, 1]]]), np.zeros((1, 3), dtype=int), [{'capacity': 2}])
    except AssertionError:
        pass
    else:
        raise AssertionError('capacity tie accepted')
    # Competition ties stay unowned, independently of cavity enumeration order.
    wall = np.zeros((5, 5), dtype=bool)
    _, owner = _maps(wall, [{'rows': [1]}, {'rows': [3]}])
    _require(np.all(owner[2] == -1), 'watershed tie broken arbitrarily')
    # Hand-constructed wall-free ring: displayed serialization must stay row
    # major in all four approaches, and solve must not mutate its input.
    ring = np.zeros((10, 15), dtype=np.int8)
    ring[:, 6] = 2
    ring[1:6, 1:6] = 1
    ring[2:5, 2:5] = 0
    # The edge payload also makes the work/supply roles unique without walls:
    # reflecting the scene cannot satisfy the reserved work-border convention.
    ring[0, 14] = ring[2, 8] = 3
    cavity = np.zeros_like(ring, dtype=bool)
    cavity[2:5, 2:5] = True
    for turn in range(4):
        inp = np.rot90(ring, turn).copy()
        expected = inp.copy()
        expected[expected == 3] = 0
        for r, c in list(zip(*np.where(np.rot90(cavity, turn))))[:2]:
            expected[r, c] = 3
        saved = inp.copy()
        _require(np.array_equal(solve(inp), expected), 'displayed fill order rotation regression')
        _require(np.array_equal(inp, saved), 'solve mutates input')
    # A fully blocked projected mouth cannot collect even adjacent payload.
    blocked = ring.copy()
    blocked[2:5, 7] = 1
    blocked[6, 9] = 3
    _require(np.array_equal(solve(blocked), blocked), 'blocked entries collect payload')
    # Capacity is geometric, with no inherited nine-slot or sampler-size cap.
    large = np.zeros((11, 17), dtype=np.int8)
    large[:, 7] = 2
    large[1:7, 1:7] = 1
    large[2:6, 2:6] = 0
    large[2:5, 8:13] = 3
    large[2, 8] = 0
    large[0, 16] = 3
    expected = large.copy()
    expected[expected == 3] = 0
    for r, c in [(r, c) for r in range(2, 6) for c in range(2, 6)][:15]:
        expected[r, c] = 3
    _require(np.array_equal(solve(large), expected), 'capacity incorrectly fixed at nine or twelve')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal explicit count combinations')
