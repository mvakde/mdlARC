"""Dock cartridges; pool marked leading cavities through asymmetric rails.

Only standard-library containers and NumPy are used. The decoder knows no seed,
sampling schedule, template bank, saved examples or builder metadata.
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "cells must be integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for i, j in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= i < h and 0 <= j < w and mask[i, j] and not seen[i, j]:
                    seen[i, j] = True
                    stack.append((i, j))
        result.append(cells)
    return result


def _at(frame, u, v):
    o, t, n = frame['o'], frame['t'], frame['n']
    return o[0]+u*t[0]+v*n[0], o[1]+u*t[1]+v*n[1]


def _uv(frame, p):
    delta = (p[0]-frame['o'][0], p[1]-frame['o'][1])
    return (delta[0]*frame['t'][0]+delta[1]*frame['t'][1],
            delta[0]*frame['n'][0]+delta[1]*frame['n'][1])


def _rail_frames(grid, bg):
    """Detect plain spines by color components, then read adjacent patterns.

    Direct tangent/normal bases implement colored dihedral correspondence.
    This also works when a cartridge is already touching its rail.
    """
    h, w = grid.shape
    candidates = []
    for spine in map(int, np.unique(grid)):
        if spine == bg:
            continue
        parts = _components(grid == spine)
        if not 2 <= len(parts) <= 4 or len({len(x) for x in parts}) != 1:
            continue
        length = len(parts[0])
        if not 6 <= length <= 12:
            continue
        frames, sequences = [], []
        for cells in parts:
            rr, cc = zip(*cells)
            if len(set(cc)) == 1 and max(rr)-min(rr)+1 == length:
                origin, tangent = (min(rr), cc[0]), (1, 0)
                normals = ((0, 1), (0, -1))
            elif len(set(rr)) == 1 and max(cc)-min(cc)+1 == length:
                origin, tangent = (rr[0], min(cc)), (0, 1)
                normals = ((1, 0), (-1, 0))
            else:
                break
            choices = []
            for normal in normals:
                f = dict(o=origin, t=tangent, n=normal)
                points = [_at(f, u, 1) for u in range(length)]
                if not all(0 <= r < h and 0 <= c < w for r, c in points):
                    continue
                seq = tuple(int(grid[p]) for p in points)
                if len(set(seq)) == 2 and not set(seq).intersection((bg, spine)):
                    choices.append((f, seq))
            if len(choices) != 1:
                break
            f, seq = choices[0]
            frames.append(f)
            sequences.append(seq)
        else:
            ref = sequences[0]
            if ref == ref[::-1] or any(s not in (ref, ref[::-1]) for s in sequences):
                continue
            for f, s in zip(frames, sequences):
                if s == ref[::-1]:
                    f['o'] = _at(f, length-1, 0)
                    f['t'] = (-f['t'][0], -f['t'][1])
                f['cells'] = {_at(f, u, v) for u in range(length) for v in (0, 1)}
            colors = {spine, *ref}
            cells = set().union(*(f['cells'] for f in frames))
            actual = {tuple(map(int, p)) for p in np.argwhere(np.isin(grid, list(colors)))}
            if cells == actual:
                candidates.append((length, colors, frames, ref))
    _need(len(candidates) == 1, "rails must have one uniquely recognizable asymmetric template")
    return candidates[0]


def _read_scene(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _need(all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
          "corners must identify background")
    length, rail_colors, rails, pattern = _rail_frames(grid, bg)
    other = set(map(int, np.unique(grid))) - rail_colors - {bg}
    wall_candidates = [color for color in other
                       if any(len(c) > 1 for c in _components(grid == color))]
    _need(len(wall_candidates) == 1, "one connected wall role must be identifiable")
    wall = wall_candidates[0]
    marker_colors = other - {wall}
    _need(all(len(c) == 1 for color in marker_colors for c in _components(grid == color)),
          "tokens must be singleton color components")
    occupied = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    tokens = {tuple(map(int, p)) for p in np.argwhere(np.isin(grid, list(marker_colors)))}
    bodies = []
    assigned, used_tokens = set(), set()
    for cells_list in _components(grid == wall):
        cells = set(cells_list)
        rr, cc = zip(*cells)
        _need(len(cells) >= 3, "cartridge is too small")
        carried = {p for p in tokens if min(rr) <= p[0] <= max(rr)
                   and min(cc) <= p[1] <= max(cc)}
        _need(not carried.intersection(used_tokens), "overlapping cartridge boxes")
        own = cells | carried
        options = []
        for index, frame in enumerate(rails):
            uv = [_uv(frame, p) for p in cells]
            us, vs = zip(*uv)
            if min(us) < 0 or max(us) >= length or max(vs) > -1:
                continue
            if max(vs)-min(vs)+1 < 3:
                continue
            shift = -1-max(vs)
            sweep = {(p[0]+k*frame['n'][0], p[1]+k*frame['n'][1])
                     for p in own for k in range(shift+1)}
            if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1])
                   for r, c in sweep) or sweep.intersection(occupied-own):
                continue
            options.append((shift, index, uv, sweep))
        _need(bool(options), "cartridge has no clear rear-facing rail")
        distance = min(x[0] for x in options)
        nearest = [x for x in options if x[0] == distance]
        _need(len(nearest) == 1, "ambiguous nearest rail")
        shift, index, uv, sweep = nearest[0]
        _need(index not in assigned, "two cartridges assigned to one rail")
        assigned.add(index)
        used_tokens.update(carried)
        u0, v0 = min(u for u, v in uv), min(v for u, v in uv)
        u1, v1 = max(u for u, v in uv), max(v for u, v in uv)
        mask = np.zeros((u1-u0+1, v1-v0+1), dtype=bool)
        for u, v in uv:
            mask[u-u0, v-v0] = True
        frame = rails[index]
        local_tokens = {}
        for p in carried:
            u, v = _uv(frame, p)
            _need(u0 <= u <= u1 and v0 <= v <= v1 and not mask[u-u0, v-v0],
                  "token is not in a cartridge pocket")
            local_tokens[(u-u0, v-v0)] = int(grid[p])
        # No hidden contents are allowed inside a cartridge's wall bounding box.
        for u in range(u0, u1+1):
            for v in range(v0, v1+1):
                p = _at(frame, u, v)
                _need(int(grid[p]) == bg or p in own, "foreign cartridge contents")
        code, regions = {}, []
        for pocket in _components(~mask):
            rows = {r for r, c in pocket}
            closed = not any(r in (0, mask.shape[0]-1) or c in (0, mask.shape[1]-1)
                             for r, c in pocket)
            contacts = {r for r, c in pocket if c == mask.shape[1]-2 and mask[r, -1]}
            active = closed and bool(contacts)
            # A recessed lobe can have a thin wall locally while another part
            # of this same connected cartridge defines a farther extreme.
            locally_thin = closed and any(
                mask[r, c+1] and not mask[r, c+2:].any()
                for r, c in pocket if c+1 < mask.shape[1])
            colors = {local_tokens[p] for p in pocket if p in local_tokens}
            _need(len(colors) <= 1, "conflicting labels in one pocket")
            slots = {u0+r for r in rows}
            color = next(iter(colors)) if colors else None
            region = dict(closed=closed, active=active, slots=slots, color=color,
                          partial=active and contacts != rows, size=len(pocket),
                          width=max(c for r, c in pocket)-min(c for r, c in pocket)+1,
                          locally_thin=locally_thin)
            regions.append(region)
            if active and color is not None:
                for slot in slots:
                    _need(slot not in code or code[slot] == color, "local code conflict")
                    code[slot] = color
        bodies.append(dict(index=index, cells=cells, tokens=carried, own=own, mask=mask,
                           shift=shift, sweep=sweep, code=code, regions=regions,
                           u0=u0, v0=v0, frame=frame))
    _need(len(bodies) == len(rails) and len(assigned) == len(rails), "one cartridge per rail required")
    _need(used_tokens == tokens, "unassigned token")
    code = {}
    for body in bodies:
        for slot, color in body['code'].items():
            _need(slot not in code or code[slot] == color, "stations disagree on shared address")
            code[slot] = color
    for i, body in enumerate(bodies):
        for other_body in bodies[:i]:
            _need(not body['sweep'].intersection(other_body['sweep']), "motion footprints overlap")
            _need(not any((r+dr, c+dc) in other_body['sweep'] for r, c in body['sweep']
                          for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
                  "cartridge motion footprints touch")
    scene = dict(grid=grid, bg=bg, wall=wall, rail_colors=rail_colors, rails=rails,
                 bodies=bodies, length=length, code=code, pattern=pattern)
    forbidden = occupied | set().union(*(b['sweep'] for b in bodies))
    painted = set()
    for frame in rails:
        for slot in code:
            ray = _ray(frame, slot, grid.shape)
            _need(bool(ray), "empty emission corridor")
            _need(not set(ray).intersection(forbidden | painted), "ray hits station or another ray")
            _need(not any(r in (0, grid.shape[0]-1) and c in (0, grid.shape[1]-1)
                          for r, c in ray), "corner used by ray")
            painted.update(ray)
    return scene


def _ray(frame, slot, shape):
    r, c = _at(frame, slot, 2)
    dr, dc = frame['n']
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r+dr, c+dc
    return cells


def _render(scene, local_only=False):
    result = scene['grid'].copy()
    for body in scene['bodies']:
        for p in body['own']:
            result[p] = scene['bg']
    for body in scene['bodies']:
        dr, dc = body['frame']['n']
        for r, c in body['own']:
            result[r+body['shift']*dr, c+body['shift']*dc] = scene['grid'][r, c]
    locals_by_index = {b['index']: b['code'] for b in scene['bodies']}
    for index, frame in enumerate(scene['rails']):
        code = locals_by_index[index] if local_only else scene['code']
        for slot, color in code.items():
            for p in _ray(frame, slot, result.shape):
                result[p] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output pixel from this input's visible geometry and labels."""
    return _render(_read_scene(input_grid))


def _orient(a, transform):
    return np.rot90(np.fliplr(a) if transform >= 4 else a, transform % 4).copy()


def _intervals(rng, length, partial=False):
    baseline = np.array([1, 2 if partial else 1, 1, 1, 1, 1, 1])
    bins = baseline + rng.multinomial(length-int(baseline.sum()), np.full(7, 1/7))
    intervals, u = [], int(bins[0])
    for i in range(3):
        stop = u+int(bins[2*i+1])
        intervals.append((u, stop))
        u = stop+int(bins[2*i+2])
    return intervals


def _cartridge(rng, length, width, intervals, kinds, labels, *, partial_index=None,
               shorter=False, tail=False, recessed_lobe=False):
    mask = np.ones((length, width), dtype=bool)
    token_positions = {}
    if tail:
        mask[:, -1] = False
        mask[int(rng.choice([0, length-1])), -1] = True
    for i, ((start, stop), kind) in enumerate(zip(intervals, kinds)):
        if kind == 'solid':
            continue
        right = width-2 if kind in ('active', 'open') and not tail else width-3
        left = 0 if kind == 'open' else int(rng.integers(1, right+1))
        if i == partial_index and kind == 'active' and stop-start >= 2:
            left = int(rng.integers(1, width-2))
            contact = int(rng.integers(start, stop))
            for row in range(start, stop):
                edge = width-2 if row == contact else width-3
                mask[row, left:edge+1] = False
        else:
            mask[start:stop, left:right+1] = False
        if recessed_lobe and kind == 'recessed' and not tail:
            # Recede the entire lobe, including its top/bottom caps. The cavity
            # stays enclosed behind a one-cell LOCAL wall, but the object's
            # leading extreme is supplied by a different lobe. This makes a
            # local-wall shortcut visibly fail in the first two demonstrations.
            mask[start-1:stop+1, -1] = False
        if labels[i] is not None:
            choices = [(r, c) for r in range(start, stop) for c in range(left, right+1)
                       if not mask[r, c]]
            p = choices[int(rng.integers(len(choices)))]
            token_positions[p] = int(labels[i])
    used = [i for i, k in enumerate(kinds) if k != 'solid']
    lower, upper = 0, length
    if shorter and used and not tail:
        first, last = intervals[min(used)][0], intervals[max(used)][1]
        lower = int(rng.integers(first))
        upper = int(rng.integers(last+1, length+1))
        # The rotation demonstration deliberately exposes non-full-height registration.
        if partial_index is not None and max(used) < 2:
            upper = last+1
    mask[:lower] = False
    mask[upper:] = False
    if upper-lower > 3 and rng.random() < .55:
        trial = mask.copy()
        trial[int(rng.choice([lower, upper-1])), 0] = False
        if len(_components(trial)) == 1:
            mask = trial
    _need(len(_components(mask)) == 1, "builder made disconnected wall")
    return mask, token_positions


def _module(rng, template, intervals, kinds, labels, wall, bg, width, gap,
            partial_index=None, shorter=False, tail=False, recessed_lobe=False):
    length = template.shape[0]
    mask, tokens = _cartridge(rng, length, width, intervals, kinds, labels,
                             partial_index=partial_index, shorter=shorter, tail=tail,
                             recessed_lobe=recessed_lobe)
    inp = np.full((length, width+gap+2), bg, dtype=np.int8)
    out = inp.copy()
    inp[:, width+gap:] = template
    out[:, width+gap:] = template
    inp[:, :width][mask] = wall
    out[:, gap:gap+width][mask] = wall
    for (r, c), color in tokens.items():
        inp[r, c] = color
        out[r, c+gap] = color
    labels_grid = np.arange(inp.size).reshape(inp.shape)
    return dict(inp=inp, out=out, labels=labels_grid, rail_col=width+gap, gap=gap,
                body_mask=mask)


def _candidate(module, transform, r, c, shape, code, bg):
    a, b = _orient(module['inp'], transform), _orient(module['out'], transform)
    labels = _orient(module['labels'], transform)
    length, original_width = module['inp'].shape
    column = module['rail_col']
    lookup = {int(labels[x, y]): (r+x, c+y) for x, y in np.ndindex(labels.shape)}
    spine0, spine1 = lookup[column], lookup[original_width+column]
    front0 = lookup[column+1]
    frame = dict(o=spine0, t=(spine1[0]-spine0[0], spine1[1]-spine0[1]),
                 n=(front0[0]-spine0[0], front0[1]-spine0[1]))
    zone = {(r+x, c+y) for x, y in np.ndindex(a.shape)}
    rays = {slot: _ray(frame, slot, shape) for slot in code}
    if any(not ray for ray in rays.values()):
        return None
    ray_cells = set().union(*(set(v) for v in rays.values())) if rays else set()
    if ray_cells.intersection(zone):
        return None
    return dict(a=a, b=b, r=r, c=c, zone=zone, rays=rays, ray_cells=ray_cells,
                transform=transform, frame=frame)


def _compatible(candidate, placed):
    if candidate is None:
        return False
    for old in placed:
        if candidate['zone'].intersection(old['zone'] | old['ray_cells']):
            return False
        if candidate['ray_cells'].intersection(old['zone'] | old['ray_cells']):
            return False
        if any((r+dr, c+dc) in old['zone'] for r, c in candidate['zone']
               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
            return False
    return True


def _place(rng, modules, transforms, code, bg, shape, demonstration):
    h, w = shape
    for _ in range(7):
        placed = []
        order = list(map(int, rng.permutation(len(modules))))
        for index in order:
            transform = transforms[index]
            ah, aw = _orient(modules[index]['inp'], transform).shape
            for attempt in range(100):
                r = int(rng.integers(1, h-ah))
                c = int(rng.integers(1, w-aw))
                if rng.random() < .35:
                    if rng.integers(2):
                        r = int(rng.choice([1, h-ah-1]))
                    else:
                        c = int(rng.choice([1, w-aw-1]))
                candidate = _candidate(modules[index], transform, r, c, shape, code, bg)
                if _compatible(candidate, placed):
                    candidate['index'] = index
                    placed.append(candidate)
                    break
            else:
                break
        if len(placed) == len(modules):
            return shape, placed, False
    # Safe outward-facing arrangements on an unpartitioned 30x30 canvas.
    # The fallback retains every cartridge, clue and address, changing placement only.
    shape = (30, 30)
    length, count = modules[0]['inp'].shape[0], len(modules)
    if count == 2 and demonstration == 'pool':
        ts = [0, 6]
        positions = [(1, 30-modules[0]['inp'].shape[1]-1),
                     (30-length-1, 30-modules[1]['inp'].shape[1]-1)]
    elif count == 2:
        ts = [1, 0]
        positions = [(1, 1), (length+2, 30-modules[1]['inp'].shape[1]-1)]
    elif count == 3:
        ts = [1, 0, 3]
        positions = [(1, 1), (length+2, 30-modules[1]['inp'].shape[1]-1),
                     (30-modules[2]['inp'].shape[1]-1, 1)]
    else:
        ts = [1, 0, 3, 2]
        positions = [(1, 11), (11, 30-modules[1]['inp'].shape[1]-1),
                     (30-modules[2]['inp'].shape[1]-1, 11), (11, 1)]
    placed = []
    for index, (transform, (r, c)) in enumerate(zip(ts, positions)):
        candidate = _candidate(modules[index], transform, r, c, shape, code, bg)
        _need(_compatible(candidate, placed), "safe placement assumptions violated")
        candidate['index'] = index
        placed.append(candidate)
    return shape, placed, True


def _make_pair(rng, palette, mode, global_transform):
    if mode == 'pool':
        count, length = 2, int(rng.integers(7, 11))
    elif mode == 'rotate':
        count, length = 2, int(rng.integers(8, 12))
    else:
        count = int(rng.choice([3, 4], p=[.8, .2])) if mode == 'test' else int(rng.choice([2, 3, 4]))
        length = int(rng.integers(7, 9 if count == 4 else 11 if count == 3 else 13))
    bg, spine, fill, cap, wall = map(int, palette[:5])
    signals = list(map(int, palette[5:]))
    for _ in range(32):
        pattern = rng.choice([fill, cap], size=length).astype(np.int8)
        if len(set(map(int, pattern))) == 2 and not np.array_equal(pattern, pattern[::-1]):
            break
    else:
        pattern = np.full(length, fill, dtype=np.int8)
        pattern[0] = cap
    template = np.column_stack((np.full(length, spine, dtype=np.int8), pattern))
    partial = mode == 'rotate'
    intervals = _intervals(rng, length, partial)
    silent_last = count >= 3 and rng.random() < .28
    programmed = [0, 2] if count == 2 or (count == 3 and silent_last) else [0, 1, 2]
    interval_code = dict(zip(programmed, signals))
    code = {u: color for i, color in interval_code.items()
            for u in range(intervals[i][0], intervals[i][1])}
    modules = []
    for i in range(count):
        # The initial two sources are always different, including minimal 2/1 puzzles.
        source = programmed[i] if i < len(programmed) else None
        if source is None and not silent_last and rng.random() < .6:
            source = int(rng.choice(programmed))  # Consistent redundant observation.
        kinds = [str(rng.choice(['solid', 'open', 'recessed', 'active'])) for _ in range(3)]
        labels = [None, None, None]
        for j in range(3):
            if kinds[j] != 'solid' and kinds[j] != 'active' and rng.random() < .65:
                labels[j] = int(rng.choice(signals))
        if source is not None:
            kinds[source], labels[source] = 'active', interval_code[source]
        if mode in ('pool', 'rotate'):
            kinds = ['solid', 'open' if i == 0 else 'recessed', 'solid']
            labels = [None, signals[3], None]
            kinds[source], labels[source] = 'active', interval_code[source]
        width = int(rng.integers(4, 6 if count == 4 else 8))
        gap = int(rng.integers(0, 2 if count == 4 else 5))
        if mode == 'pool':
            gap = 1 if i == 0 else 0
        tail = source is None and rng.random() < .65
        if tail:
            kinds = ['recessed', 'solid', 'solid']
            labels = [signals[3], None, None]
        partial_index = 0 if partial and i == 0 else (
            source if source is not None and intervals[source][1]-intervals[source][0] >= 2
            and rng.random() < .4 else None)
        shorter = (mode == 'rotate' and i == 0) or (mode not in ('pool',) and rng.random() < .65)
        module = _module(rng, template, intervals, kinds, labels, wall, bg, width, gap,
                         partial_index=partial_index, shorter=shorter, tail=tail,
                         recessed_lobe=mode in ('pool', 'rotate') and i == 1)
        modules.append(module)
    if mode == 'pool':
        transforms = [0, 6]
    elif mode == 'rotate':
        transforms = [0, 1]
    else:
        transforms = list(map(int, rng.integers(8, size=count)))
        if count >= 3:
            transforms[1] = (transforms[0]+1) % 4
    widest = max(m['inp'].shape[1] for m in modules)
    low = max(18 if count == 2 else 24 if count == 3 else 28, length+widest+3)
    shape = tuple(map(int, rng.integers(low, 31, size=2)))
    shape, placed, _ = _place(rng, modules, transforms, code, bg, shape, mode)
    inp, out = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
    for station in placed:
        r, c, a, b = station['r'], station['c'], station['a'], station['b']
        inp[r:r+a.shape[0], c:c+a.shape[1]] = a
        out[r:r+b.shape[0], c:c+b.shape[1]] = b
        for slot, ray in station['rays'].items():
            for p in ray:
                out[p] = code[slot]
    return {role: _orient(grid, global_transform) for role, grid in (('input', inp), ('output', out))}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    fixed_palette = bool(rng.integers(2))
    palette = rng.permutation(10)
    global_transform = int(rng.integers(8))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = ('pool', 'rotate')[index] if split == 'train' and index < 2 else (
                'test' if split == 'test' else 'extra')
            for attempt in range(24):
                colors = palette if fixed_palette else rng.permutation(10)
                pair = _make_pair(rng, colors, mode, global_transform)
                # Assignment/collision checks are input-based even during construction.
                try:
                    scene = _read_scene(pair['input'])
                    _need(np.array_equal(_render(scene), pair['output']), 'builder/decoder disagreement')
                except AssertionError:
                    if attempt == 23:
                        raise
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded unique scene construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(scene):
    rails, bodies, code = scene['rails'], scene['bodies'], scene['code']
    regions = [r for b in bodies for r in b['regions']]
    return dict(shape=scene['grid'].shape, length=scene['length'], count=len(rails),
                background=scene['bg'], rail_colors=tuple(sorted(scene['rail_colors'])), wall=scene['wall'],
                normals=tuple(f['n'] for f in rails), tangents=tuple(f['t'] for f in rails),
                masks=tuple(_key(b['mask']) for b in bodies), shifts=tuple(b['shift'] for b in bodies),
                widths=tuple(b['mask'].shape[1] for b in bodies),
                slot_count=len(code), slots=tuple(sorted(code)), colors=tuple(sorted(set(code.values()))),
                spans=tuple(sorted(len(r['slots']) for r in regions if r['active'] and r['color'] is not None)),
                pooled=bool(code) and all(b['code'] != code for b in bodies),
                local_only_diff=not np.array_equal(_render(scene), _render(scene, local_only=True)),
                rejected_open=any(not r['closed'] and r['color'] is not None for r in regions),
                rejected_closed=any(r['closed'] and not r['active'] and r['color'] is not None for r in regions),
                rejected_local_thin=any(r['closed'] and not r['active'] and r['locally_thin']
                                        and r['color'] is not None and r['slots']-set(code)
                                        for r in regions),
                omitted_b_extra=any(not r['active'] and r['color'] is not None
                                   and r['slots']-set(code) for r in regions),
                partial=any(r['partial'] and r['color'] is not None for r in regions),
                shorter=any(b['mask'].shape[0] < scene['length'] for b in bodies),
                reversed_pair=any(a['n'] == b['n'] and a['t'] == (-b['t'][0], -b['t'][1])
                                  for i, a in enumerate(rails) for b in rails[:i]),
                perpendicular=any(a['n'][0]*b['n'][0]+a['n'][1]*b['n'][1] == 0
                                  for i, a in enumerate(rails) for b in rails[:i]),
                ray_lengths=tuple(len(_ray(f, s, scene['grid'].shape)) for f in rails for s in code),
                asymmetric_code=code != {scene['length']-1-s: c for s, c in code.items()},
                tail=any(np.count_nonzero(b['mask'][:, -1]) == 1 and
                         any(r['closed'] and not r['active'] for r in b['regions']) for b in bodies),
                redundant=sum(len(b['code']) for b in bodies) > len(code))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'dimensions changed')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            scene = _read_scene(inp)
            _need(np.array_equal(out, _render(scene)), 'incorrect docking/broadcast pixels')
            for f in scene['rails']:
                _need(all(inp[p] == out[p] for p in f['cells']), 'rail changed')
            _need(np.count_nonzero(inp == scene['wall']) == np.count_nonzero(out == scene['wall']),
                  'wall shape/area was not preserved')
            if split == 'train':
                training.append(_features(scene))
    _need(any(f['pooled'] and f['asymmetric_code'] and f['reversed_pair'] for f in training),
          'training lacks observable reflected pooling')
    _need(any(f['perpendicular'] for f in training), 'training lacks perpendicular transfer')
    _need(any(f['rejected_open'] and f['rejected_closed'] and f['omitted_b_extra'] for f in training),
          'training lacks both pocket-selection contrasts')
    _need(any(f['rejected_local_thin'] for f in training),
          'training does not distinguish whole-object extreme from local thin walls')
    _need(any(f['partial'] and max(f['spans'], default=0) >= 2 for f in training),
          'training lacks whole-span partial-contact evidence')
    _need(any(f['shorter'] for f in training), 'training lacks registration-offset evidence')
    _need(any(max(f['shifts']) > 0 for f in training), 'training never demonstrates docking')
    _need(any(0 in f['shifts'] for f in training), 'training lacks already-docked evidence')
    _need(any(f['local_only_diff'] for f in training), 'omitting A does not change training')


def self_test():
    """Bounded authored semantic/diversity checks across 42 reproducible seeds."""
    puzzles, features, fingerprints = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        fingerprints.add(tuple(_key(p['input']) for p in puzzle['train']+puzzle['test']))
        for pair in puzzle['train']+puzzle['test']:
            scene = _read_scene(pair['input'])
            _need(np.array_equal(solve(pair['input']), pair['output']), 'solve mismatch')
            features.append(_features(scene))
        # A concrete omission of B: all marked pockets program their spans,
        # including the same-color open/recessed middle-range decoys in train 1.
        scene = _read_scene(puzzle['train'][0]['input'])
        raw_code = {}
        for body in scene['bodies']:
            for region in body['regions']:
                if region['color'] is not None:
                    for slot in region['slots']:
                        _need(slot not in raw_code or raw_code[slot] == region['color'],
                              'first counterfactual should have consistent labels')
                        raw_code[slot] = region['color']
        unfiltered = dict(scene, code=raw_code)
        _need(not np.array_equal(_render(unfiltered), puzzle['train'][0]['output']),
              'omitting B did not change the answer')
    _need(len(fingerprints) == 42, 'puzzle variation collapsed')
    _need(len({f['shape'] for f in features}) >= 30, 'canvas variation collapsed')
    _need({f['count'] for f in features} == {2, 3, 4}, 'station counts collapsed')
    _need(len({f['length'] for f in features}) >= 5, 'rail lengths collapsed')
    _need(len({m for f in features for m in f['masks']}) >= 120, 'cartridge geometry collapsed')
    _need({n for f in features for n in f['normals']} == {(-1, 0), (1, 0), (0, -1), (0, 1)},
          'emission directions collapsed')
    _need(len({f['background'] for f in features}) >= 8, 'palette roles collapsed')
    _need(len({f['slots'] for f in features}) >= 20, 'shared code addresses collapsed')
    _need(len({d for f in features for d in f['shifts']}) >= 4, 'docking distances collapsed')
    _need(any(f['tail'] for f in features), 'no protrusion-based recess')
    _need(any(f['redundant'] for f in features), 'no consistent redundant observation')
    _need(any(max(f['ray_lengths']) >= 12 for f in features), 'only short edge rays sampled')
    # Public count interface, including every legal explicit combination.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+a*10+b, a, b)
            _need((len(p['train']), len(p['test'])) == (a, b), 'requested counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = puzzles[0]
    again = generate_puzzle(0)
    _need(all(np.array_equal(p[k], q[k]) for split in ('train', 'test')
              for p, q in zip(original[split], again[split]) for k in ('input', 'output')),
          'seed is not reproducible')
    # Test geometric/color equivariance on the same visible puzzle.
    transformed = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for i, pair in enumerate(original[split]):
            lookup = np.roll(np.arange(10, dtype=np.int8), i+2)
            t = (i+3) % 8
            new = {role: _orient(lookup[grid], t) for role, grid in pair.items()}
            _need(np.array_equal(solve(new['input']), new['output']), 'isometry/recoloring failed')
            transformed[split].append(new)
    validate_generated_puzzle(transformed)
    # No labels: alignment still works and the complete program is empty.
    scene = _read_scene(original['train'][0]['input'])
    empty = scene['grid'].copy()
    for body in scene['bodies']:
        for p in body['tokens']:
            empty[p] = scene['bg']
    expected = empty.copy()
    for body in scene['bodies']:
        for p in body['cells']:
            expected[p] = scene['bg']
    for body in scene['bodies']:
        dr, dc = body['frame']['n']
        for r, c in body['cells']:
            expected[r+body['shift']*dr, c+body['shift']*dc] = scene['wall']
    _need(not _read_scene(empty)['code'] and np.array_equal(solve(empty), expected),
          'empty-code alignment failed')
    # A length-six rail, outside the random sampler's length range. Removing
    # just a leading protrusion makes a formerly recessed token pocket active.
    fixture = np.zeros((18, 20), dtype=np.int8)
    rail = np.column_stack((np.full(6, 1), np.array([2, 2, 3, 2, 3, 3])))
    fixture[1:7, 9:11] = rail
    fixture[11:17, 9:11] = rail
    wall_mask = np.ones((6, 5), dtype=bool)
    wall_mask[:, -1] = False
    wall_mask[0, -1] = True
    wall_mask[1:3, 1:3] = False
    fixture[1:7, 2:7][wall_mask] = 4
    fixture[2, 3] = 5
    second_mask = np.ones((6, 4), dtype=bool)
    second_mask[4, 1:3] = False
    fixture[11:17, 4:8][second_mask] = 4
    fixture[15, 5] = 6
    before = _read_scene(fixture)
    without_tail = fixture.copy()
    without_tail[1, 6] = 0
    after = _read_scene(without_tail)
    _need(set(before['code'].values()) == {6} and set(after['code'].values()) == {5, 6},
          'whole-object extreme does not control recessed activation')
    for input_value in (fixture, without_tail):
        for t in range(8):
            _need(np.array_equal(solve(_orient(input_value, t)), _orient(solve(input_value), t)),
                  'protrusion fixture is not equivariant')
    # A pocket may carry two agreeing, diagonally separated singleton labels.
    # Their number and exact marker rows must not change its address span.
    repeated = without_tail.copy()
    repeated[3, 4] = 5
    repeated_expected = solve(without_tail)
    first_body = next(b for b in after['bodies'] if (2, 3) in b['tokens'])
    dr, dc = first_body['frame']['n']
    repeated_expected[3+first_body['shift']*dr, 4+first_body['shift']*dc] = 5
    _need(_read_scene(repeated)['code'] == after['code'] and
          np.array_equal(solve(repeated), repeated_expected),
          'agreeing labels changed a cavity span or were not carried')
    # Reject two different labels within one pocket, or different labels for
    # the same template address learned from two otherwise legal stations.
    pocket_conflict = without_tail.copy()
    pocket_conflict[2, 4] = 7
    address_conflict = without_tail.copy()
    address_conflict[12:14, 5:7] = 0
    address_conflict[12, 5] = 6
    for invalid in (pocket_conflict, address_conflict):
        try:
            solve(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('conflicting pocket/address labels accepted')
    thin = fixture.copy()
    thin[11:17, 4:8] = 0
    thin[13, 5:9] = 4  # A solid one-slot cartridge, already docked.
    thin_scene = _read_scene(thin)
    _need(any(b['mask'].shape[0] == 1 and b['shift'] == 0 for b in thin_scene['bodies']),
          'one-slot solid cartridge was not recognized')
    _need(not thin_scene['code'] and np.array_equal(solve(thin.tolist()), _render(thin_scene)),
          'silent thin-cartridge/list input failed')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=len(features), shapes=len({f['shape'] for f in features}),
                masks=len({m for f in features for m in f['masks']}),
                station_counts=sorted({f['count'] for f in features}),
                lengths=sorted({f['length'] for f in features}),
                shared_slot_patterns=len({f['slots'] for f in features}))
