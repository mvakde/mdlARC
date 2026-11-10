"""Dock-and-drain reservoirs: cavity-selected, conserved hydraulic transfer.

Only stdlib and NumPy are used. Nothing is read from disk by this module.
All checks stream puzzles; no sample cache, trace, or persistent RNG exists.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        stack, cells = [(row, col)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        part = np.zeros_like(mask, dtype=bool)
        rr, cc = np.array(cells).T
        part[rr, cc] = True
        parts.append(part)
    return parts


def _regions(solid):
    h, w = solid.shape
    result = []
    for free in _components(~solid):
        rr, cc = np.nonzero(free)
        enclosed = not (free[0].any() or free[-1].any() or free[:, 0].any() or free[:, -1].any())
        contact = sorted(int(c) for c in np.flatnonzero(free[h - 2] & solid[-1])) if h > 1 else []
        active = bool(enclosed and contact)
        result.append(dict(mask=free, enclosed=enclosed, contact=contact, active=active,
                           columns=tuple(map(int, np.unique(cc)))))
    return result


def _levels(free):
    """Complete, connected bottom fills, in increasing volume order."""
    levels = []
    for row in range(free.shape[0] - 1, -1, -1):
        target = free.copy()
        target[:row] = False
        if target[row].any() and len(_components(target)) == 1:
            levels.append((int(target.sum()), target))
    return levels


def _fill(free, volume):
    if volume == 0:
        return np.zeros_like(free, dtype=bool)
    # A partial level is deliberately outside this family: no left/right tie.
    for count, target in _levels(free):
        if count == volume:
            return target
    raise ValueError("volume does not define a complete connected liquid level")


def _scene(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg), "the top margin must be blank")
    candidates = []
    for r in range(3, h - 3):
        water, rail = int(grid[r, 0]), int(grid[r, 1])
        if len({bg, water, rail}) != 3 or grid[r, -1] != water:
            continue
        if np.all(grid[r, 1:-1] == rail) and np.count_nonzero(grid == rail) == w - 2:
            candidates.append((r, water, rail))
    _require(len(candidates) == 1, "expected one endpoint-marked docking rail")
    row, water, rail = candidates[0]
    upper = grid[:row]
    solid = (upper != bg) & (upper != water)
    objects, assigned = [], np.zeros_like(solid)
    for part in _components(solid):
        rr, cc = np.nonzero(part)
        r0, r1, c0, c1 = map(int, (rr.min(), rr.max(), cc.min(), cc.max()))
        local = part[r0:r1 + 1, c0:c1 + 1].copy()
        colors = np.unique(upper[part])
        _require(len(colors) == 1, "each reservoir must be monochrome")
        regions = _regions(local)
        view = upper[r0:r1 + 1, c0:c1 + 1]
        for region in regions:
            region['source'] = (view == water) & region['mask']
            region['volume'] = int(region['source'].sum())
            assigned[r0:r1 + 1, c0:c1 + 1] |= region['source']
        objects.append(dict(mask=local, r0=r0, r1=r1, c0=c0, c1=c1,
                            color=int(colors[0]), regions=regions, shift=row - 1 - r1))
    _require(objects and np.array_equal(assigned, upper == water), "uncontained upper liquid")
    # Disjoint horizontal strips make simultaneous docking collision-free.
    objects.sort(key=lambda o: o['c0'])
    _require(all(b['c0'] > a['c1'] + 1 for a, b in zip(objects, objects[1:])),
             "reservoir column strips need a blank separating column")
    lower = grid[row + 1:]
    _require(not np.any((lower == water) | (lower == rail)), "receiver starts dry")
    terrain = lower != bg
    _require(terrain[-1].all() and terrain[:, 0].all() and terrain[:, -1].all(),
             "receivers need a floor and side walls")
    bays = _components(~terrain)
    labels = np.full(lower.shape, -1, dtype=int)
    for i, bay in enumerate(bays):
        labels[bay] = i
    supplies = [[] for _ in bays]
    for obj in objects:
        for region in obj['regions']:
            if not region['active']:
                continue
            cols = [obj['c0'] + c for c in region['contact']]
            destinations = {int(labels[0, c]) for c in cols}
            _require(len(destinations) == 1 and -1 not in destinations,
                     "all projected outlet columns must enter one receiver")
            region['bay'] = next(iter(destinations))
            region['outlets'] = cols
            supplies[region['bay']].append(region)
    return dict(grid=grid, bg=bg, water=water, rail=rail, row=row,
                objects=objects, terrain=terrain, bays=bays, supplies=supplies)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode roles and compute every liquid destination from input pixels."""
    scene = _scene(input_grid)
    grid, row, bg, water = (scene[k] for k in ('grid', 'row', 'bg', 'water'))
    out = grid.copy()
    out[:row] = bg
    for obj in scene['objects']:
        height, width = obj['mask'].shape
        top = row - height
        view = out[top:row, obj['c0']:obj['c0'] + width]
        view[obj['mask']] = obj['color']
        for region in obj['regions']:
            if not region['active']:
                view[_fill(region['mask'], region['volume'])] = water
    lower = out[row + 1:]
    for bay, supplies in zip(scene['bays'], scene['supplies']):
        lower[_fill(bay, sum(s['volume'] for s in supplies))] = water
    return out


def _carve(mask, left, right, top, bottom, rng, stepped):
    """A room with a flat floor and optionally stepped ceiling."""
    for c in range(left, right):
        start = top + int(stepped and c != left and bottom - top >= 2 and rng.random() < .4)
        mask[start:bottom, c] = False


def _reservoir(rng, kind, force_tail=False):
    width = int(rng.integers(4, 7))
    if kind == 'multi':
        width = int(rng.integers(7, 10))
        split = int(rng.integers(3, width - 2))
        upper, lower = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        h = upper + lower + 3
        mask = np.ones((h, width), dtype=bool)
        _carve(mask, 1, width - 1, 1, upper + 1, rng, True)
        _carve(mask, 1, split, upper + 2, h - 1, rng, True)
        _carve(mask, split + 1, width - 1, upper + 2, h - 1, rng, True)
    elif kind == 'mixed':
        upper, lower = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        h = upper + lower + 3
        mask = np.ones((h, width), dtype=bool)
        _carve(mask, 1, width - 1, 1, upper + 1, rng, True)
        _carve(mask, 1, width - 1, upper + 2, h - 1, rng, True)
    else:
        inner = int(rng.integers(2, 6))
        mask = np.ones((inner + 2, width), dtype=bool)
        _carve(mask, 1, width - 1, 1, inner + 1, rng, kind != 'open')
        if kind == 'open':
            mask[0, 1:-1] = False
        elif kind == 'recessed':
            extra = int(rng.integers(1, 4))
            tail = np.zeros((extra, width), dtype=bool)
            if force_tail or rng.random() < .65:
                tail[:, int(rng.choice([0, width - 1]))] = True
            else:
                tail[:] = True
            mask = np.vstack((mask, tail))
    # Safe corner nibbles vary the silhouette without opening a room.
    if kind != 'open':
        for c in (0, width - 1):
            if rng.random() < .35:
                trial = mask.copy()
                trial[0, c] = False
                if len(_components(trial)) == 1:
                    mask = trial
    return mask


def _scatter(rng, free, volume, target=None):
    coords = np.argwhere(free)
    style = int(rng.integers(4))
    anchor = coords[int(rng.integers(len(coords)))]
    dr, dc = np.abs(coords[:, 0] - anchor[0]), np.abs(coords[:, 1] - anchor[1])
    score = (rng.random(len(coords)) if style == 0 else
             dr + dc + rng.random(len(coords)) if style == 1 else
             3 * dr + dc * .1 + rng.random(len(coords)) if style == 2 else
             3 * dc + dr * .1 + rng.random(len(coords)))
    chosen = coords[np.argsort(score)[:volume]]
    result = np.zeros_like(free, dtype=bool)
    result[chosen[:, 0], chosen[:, 1]] = True
    if target is not None and np.array_equal(result, target) and volume < len(coords):
        remove = tuple(map(int, np.argwhere(result)[0]))
        add = tuple(map(int, np.argwhere(free & ~result)[0]))
        result[remove], result[add] = False, True
    return result


def _allocate(rng, volume, capacities):
    counts, remaining = [], volume
    for i, cap in enumerate(capacities):
        later = capacities[i + 1:]
        low, high = max(1, remaining - sum(later)), min(cap, remaining - len(later))
        _require(low <= high, "infeasible chamber supply")
        value = int(rng.integers(low, high + 1))
        counts.append(value)
        remaining -= value
    _require(remaining == 0, "lost volume in allocation")
    return counts


def _make_pair(rng, roles, kinds, regime):
    bg, water, rail = roles
    masks = [_reservoir(rng, k, regime == 'contrast' and k == 'recessed') for k in kinds]
    order = rng.permutation(len(masks))
    masks, kinds = [masks[i] for i in order], [kinds[i] for i in order]
    # Width <= 30 by construction, even for multi+active+open.
    extra = int(rng.integers(0, 3))
    width = sum(m.shape[1] for m in masks) + len(masks) + 1 + extra
    _require(width <= 30, "too many reservoirs for width budget")
    gaps = rng.multinomial(extra, np.full(len(masks) + 1, 1 / (len(masks) + 1)))
    rail_row = max(m.shape[0] for m in masks) + int(rng.integers(2 if regime == 'contrast' else 1, 5))
    lower_h = int(rng.integers(6, min(13, 29 - rail_row) + 1))
    height = rail_row + 1 + lower_h
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[rail_row, 1:-1] = rail
    grid[rail_row, [0, -1]] = water
    colors = [c for c in range(10) if c not in roles]
    terrain_color = int(rng.choice(colors))
    object_colors = [c for c in colors if c != terrain_color]
    objects, left = [], 1 + int(gaps[0])
    for index, (mask, kind) in enumerate(zip(masks, kinds)):
        top = int(rng.integers(1, rail_row - mask.shape[0] + 1))
        if regime == 'contrast' and index == 0:
            top = 1  # at least one visible docking move, independent of kind
        color = int(rng.choice(object_colors))
        view = grid[top:top + mask.shape[0], left:left + mask.shape[1]]
        view[mask] = color
        regions = _regions(mask)
        for reg in regions:
            reg['left'], reg['top'] = left, top
        objects.append(dict(left=left, top=top, mask=mask, kind=kind, regions=regions))
        left += mask.shape[1] + 1 + int(gaps[index + 1])
    active = [r for o in objects for r in o['regions'] if r['active']]
    _require(active, "a scene must demonstrate transfer")
    object_gaps = [a['left'] + a['mask'].shape[1] for a in objects[:-1]]
    if regime == 'contrast':
        cuts = object_gaps
    elif regime == 'merge':
        cuts = [int(rng.choice(object_gaps))]
    elif regime == 'split':
        multi = next(o for o in objects if o['kind'] == 'multi')
        contacts = [r['contact'] for r in multi['regions'] if r['active']]
        cut = multi['left'] + max(contacts[0]) + 1
        cuts = [cut]
    else:
        cuts = [c for c in object_gaps if rng.random() < .45]
    boundaries = [0] + sorted(cuts) + [width - 1]
    lower = grid[rail_row + 1:]
    lower[:, boundaries] = terrain_color
    lower[-1] = terrain_color
    for left_wall, right_wall in zip(boundaries, boundaries[1:]):
        lo, hi = left_wall + 1, right_wall
        donors = [r for r in active if lo <= r['left'] + r['contact'][0] < hi]
        _require(all(all(lo <= r['left'] + c < hi for c in r['contact']) for r in donors),
                 "partition cuts a nozzle")
        n, span = len(donors), hi - lo
        unequal = regime == 'merge' and n > 1
        minimum = n + 1 if unequal else 2 if regime == 'contrast' and n else max(1, n)
        floor = rng.integers(lower_h - 4, lower_h - 1, size=span)
        supply_limit = sum(int(r['mask'].sum()) for r in donors) if donors else span
        if regime == 'contrast' and donors:
            supply_limit -= 1  # visibly encode amount, not chamber capacity
        core_width = min(span, supply_limit, minimum + (0 if unequal else int(rng.integers(0, 3))))
        core_left = int(rng.integers(0, span - core_width + 1))
        floor[core_left:core_left + core_width] = lower_h - 1
        # A nonmonotone floor is safe only for a full-width level above EVERY
        # sill. Otherwise use a single monotone catchment: global low filling
        # must never transfer water across a barrier it has not overflowed.
        rough = rng.random() < .5
        high_options = []
        if rough and donors:
            candidate = np.zeros_like(lower, dtype=bool)
            for c, stop in enumerate(floor):
                candidate[:int(stop), lo + c] = True
            high_options = [(v, p) for v, p in _levels(candidate)
                            if minimum <= v <= supply_limit and (not unequal or v % n != 0)
                            and not p[0].any() and p[int(np.min(floor)) - 1, lo:hi].all()]
        if not rough or (donors and not high_options):
            for c in range(core_left - 1, -1, -1):
                floor[c] = min(floor[c], floor[c + 1])
            for c in range(core_left + core_width, span):
                floor[c] = min(floor[c], floor[c - 1])
        for c, stop in enumerate(floor):
            lower[int(stop):, lo + c] = terrain_color
        free = np.zeros(lower.shape, dtype=bool)
        free[:, lo:hi] = lower[:, lo:hi] == bg
        if donors:
            capacities = [int(r['mask'].sum()) for r in donors]
            options = high_options or [(v, p) for v, p in _levels(free)
                                       if minimum <= v <= supply_limit and (not unequal or v % n != 0)
                                       and not p[0].any()]
            _require(options, "no complete receiver level fits the supply")
            volume, _ = options[int(rng.integers(len(options)))]
            for reg, amount in zip(donors, _allocate(rng, volume, capacities)):
                source = _scatter(rng, reg['mask'], amount)
                view = grid[reg['top']:reg['top'] + source.shape[0],
                            reg['left']:reg['left'] + source.shape[1]]
                view[source] = water
    for obj in objects:
        for reg in obj['regions']:
            if reg['active']:
                continue
            # Only enclosed rooms and top-open cups are wet. Exterior beside
            # a tail stays dry. Retained sources have room to visibly settle.
            free = reg['mask']
            wet = reg['enclosed'] or (obj['kind'] == 'open' and free[0].any())
            if not wet:
                continue
            options = [(v, p) for v, p in _levels(free) if v < int(free.sum())]
            _require(options, "retained room has no nonfull level")
            volume, target = options[int(rng.integers(len(options)))]
            source = _scatter(rng, free, volume, target)
            view = grid[obj['top']:obj['top'] + free.shape[0],
                        obj['left']:obj['left'] + free.shape[1]]
            view[source] = water
    # Reflection preserves downward gravity and every hydraulic connection.
    if rng.random() < .5:
        grid = grid[:, ::-1].copy()
    return {'input': grid, 'output': solve(grid)}


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _unique(pair, roles, seen):
    # Fourteen distinct mappings of seven non-key colors. Each input uses
    # >=2 such colors, so no rotation/reflection of the odd cycle fixes it.
    # There are at most thirteen previous inputs, giving a finite guarantee.
    colors = [c for c in range(10) if c not in roles]
    for sign in (1, -1):
        for offset in range(7):
            lookup = np.arange(10, dtype=np.int8)
            for i, c in enumerate(colors):
                lookup[c] = colors[(sign * i + offset) % 7]
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError("finite uniqueness guarantee violated")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= int(num_train) <= 10 or not 1 <= int(num_test) <= 4 or num_train <= num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    # The two compulsory demonstrations already expose all selection tests,
    # retained settling, multiple receivers and aggregation of separate donors.
    schedules = [(['active', 'open', 'recessed'], 'contrast'),
                 (['mixed', 'active', 'active'], 'merge')]
    for _ in range(int(num_train) - 2):
        count = int(rng.integers(1, 4))
        kinds = ['active'] + [str(rng.choice(['active', 'open', 'recessed', 'mixed']))
                              for _ in range(count - 1)]
        if count <= 2 and rng.random() < .45:
            kinds[0] = 'multi'
        schedules.append((kinds, 'random'))
    # Tests split one body's outlets across a divider; another body's supply
    # merges on one side. Extra retained rooms/cups are optional, not a fixed
    # third-object category or count cue.
    for _ in range(int(num_test)):
        kinds = ['multi', str(rng.choice(['active', 'mixed', 'multi']))]
        if rng.random() < .6:
            kinds.append(str(rng.choice(['open', 'recessed'])))
        schedules.append((kinds, 'split'))
    for i, (kinds, regime) in enumerate(schedules):
        pair = _unique(_make_pair(rng, roles, kinds, regime), roles, seen)
        seen.add(_key(pair['input']))
        puzzle['train' if i < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pour(free, injections, reverse=False):
    """Unit pouring check, separate from complete-level construction.

    Each injection is a column at the top of a dry receiver. Horizontal travel
    is allowed only when supported and always ends at a downward outlet.
    """
    wet = np.zeros_like(free, dtype=bool)
    h, w = free.shape
    for col in injections:
        row = 0
        _require(free[row, col] and not wet[row, col], "source above a full receiver")
        for _ in range(h + 1):
            if row + 1 < h and free[row + 1, col] and not wet[row + 1, col]:
                row += 1
                continue
            left = right = col
            while left > 0 and free[row, left - 1] and not wet[row, left - 1]:
                left -= 1
            while right + 1 < w and free[row, right + 1] and not wet[row, right + 1]:
                right += 1
            exits = [c for c in range(left, right + 1)
                     if row + 1 < h and free[row + 1, c] and not wet[row + 1, c]]
            if not exits:
                wet[row, col] = True
                break
            col = min(exits, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
        else:
            raise AssertionError("pouring did not terminate")
    return wet


def _inspect(pair, deep=False):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "dimensions changed")
    s = _scene(inp)
    row, water, bg = s['row'], s['water'], s['bg']
    _require(np.array_equal(inp[row], out[row]), "rail/key changed")
    _require(np.array_equal(inp[row + 1:][s['terrain']], out[row + 1:][s['terrain']]),
             "fixed terrain changed")
    _require(int((inp == water).sum()) == int((out == water).sum()), "liquid not conserved")
    _require(np.array_equal(solve(inp), out), "incorrect output")
    stats = dict(objects=len(s['objects']), active=0, open_wet=0, recessed_wet=0,
                 retained=0, drained=0, moved=0, zero=0, retained_reshaped=0,
                 shared=0, split_body=0, multilevel=0, lateral=0, stepped=0,
                 nonfull_donors=0, unequal_donors=0, joined_pockets=0)
    for obj in s['objects']:
        stats['moved'] += obj['shift'] > 0
        stats['zero'] += obj['shift'] == 0
        top = row - obj['mask'].shape[0]
        view = out[top:row, obj['c0']:obj['c1'] + 1]
        _require(np.array_equal(view == obj['color'], obj['mask']), "reservoir deformed")
        destinations = set()
        for reg in obj['regions']:
            free, volume = reg['mask'], reg['volume']
            if reg['active']:
                _require(set(reg['contact']) == set(reg['columns']), "partial floor contact")
                _require(not np.any(view[free] == water), "active room failed to drain")
                stats['active'] += 1
                stats['drained'] += volume
                stats['nonfull_donors'] += volume < int(free.sum())
                destinations.add(reg['bay'])
            elif volume:
                target = _fill(free, volume)
                _require(np.array_equal(view[free] == water, target[free]), "retained room misfilled")
                stats['retained'] += volume
                stats['recessed_wet' if reg['enclosed'] else 'open_wet'] += 1
                stats['retained_reshaped'] += not np.array_equal(reg['source'], target)
                # Stability below its exposed surface, including an open cup.
                surface = int(np.nonzero(target)[0].min())
                eligible = free.copy()
                eligible[:surface] = False
                _require(np.array_equal(eligible, target), "retained liquid leaks or splits")
                if not reg['enclosed']:
                    _require(not target[0].any() and not target[-1].any() and
                             not target[:, 0].any() and not target[:, -1].any(), "open cup leaks")
            if reg['enclosed']:
                rr, cc = np.nonzero(free)
                stats['stepped'] += int(free.sum()) < (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
        stats['split_body'] += len(destinations) > 1
    lower_wet = out[row + 1:] == water
    _require(int(lower_wet.sum()) == stats['drained'], "wrong transferred amount")
    for bay, supplies in zip(s['bays'], s['supplies']):
        columns = np.flatnonzero(bay[0])
        _require(len(columns) and np.array_equal(columns, np.arange(columns[0], columns[-1] + 1)),
                 "receiver needs a contiguous open top")
        for c in columns:
            depth = int(bay[:, c].sum())
            _require(bay[:depth, c].all() and not bay[depth:, c].any(), "receiver has an overhang")
        volume = sum(r['volume'] for r in supplies)
        target = lower_wet & bay
        _require(int(target.sum()) == volume, "cross-receiver transfer")
        if not volume:
            continue
        stats['shared'] += len(supplies) > 1
        stats['unequal_donors'] += len(supplies) > 1 and len({r['volume'] for r in supplies}) > 1
        rr, cc = np.nonzero(target)
        stats['multilevel'] += len(set(map(int, rr))) > 1
        outlets = {c for r in supplies for c in r['outlets']}
        stats['lateral'] += bool(set(map(int, cc)) - outlets)
        surface = int(rr.min())
        _require(surface > 0, "receiver must retain a dry entrance row")
        depths = [int(bay[:, c].sum()) for c in columns]
        falling = False
        monotone_catchment = True
        for a, b in zip(depths, depths[1:]):
            if b < a:
                falling = True
            elif b > a and falling:
                monotone_catchment = False
        _require(monotone_catchment or target[surface, columns].all(),
                 "a shallow supply cannot cross an unsubmerged sill")
        stats['joined_pockets'] += any(len(_components(target &
            (np.arange(target.shape[0])[:, None] >= level))) > 1
            for level in range(surface + 1, target.shape[0]))
        closure = bay.copy()
        closure[:surface] = False
        _require(np.array_equal(closure, target) and len(_components(target)) == 1,
                 "receiver level is incomplete or disconnected")
        if deep:
            for reverse in (False, True):
                order = list(reversed(supplies)) if reverse else supplies
                injections = [r['outlets'][-1 if reverse else 0] for r in order for _ in range(r['volume'])]
                _require(np.array_equal(_pour(bay, injections, reverse), target), "routing/order ambiguity")
    stats.update(shape=inp.shape, roles=(bg, water, s['rail']),
                 volume=int((inp == water).sum()) - 2,
                 bays=len(s['bays']), profile=tuple(map(int, np.sum(s['terrain'], axis=0))))
    return stats


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[k], list) for k in ('train', 'test')), "split schema")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, roles, train_stats = set(), None, []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            key = _key(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            stats = _inspect(pair)
            roles = stats['roles'] if roles is None else roles
            _require(roles == stats['roles'], "color key changed within puzzle")
            if split == 'train':
                train_stats.append(stats)
            else:
                _require(stats['split_body'] and stats['shared'],
                         "test lacks spatial recombination")
    _require(any(s['open_wet'] and s['recessed_wet'] and s['drained'] for s in train_stats),
             "training lacks enclosure/extreme contrasts")
    _require(any(s['shared'] and s['bays'] >= 2 for s in train_stats), "training lacks merging/routing")
    _require(any(s['unequal_donors'] for s in train_stats) and any(s['nonfull_donors'] for s in train_stats),
             "training hides per-chamber amount accounting")
    _require(any(s['retained_reshaped'] and s['moved'] for s in train_stats), "training hides settling/docking")


def _fixture(kind):
    """Tiny explicit counterfactual: the same 2 units, closed/open/tailed."""
    grid = np.zeros((14, 9), dtype=np.int8)
    grid[7, 1:-1], grid[7, [0, -1]] = 8, 1
    grid[2:6, 2:6] = 3
    grid[3:5, 3:5] = 0
    grid[3, 3:5] = 1
    if kind == 'open':
        grid[2, 3:5] = 0
    elif kind == 'tail':
        grid[6, 2] = 3
    grid[8:, [0, -1]] = 5
    grid[13] = 5
    grid[11:, 1:8] = 5
    grid[11:13, 3:5] = 0
    expected = grid.copy()
    expected[:7] = 0
    delta = 0 if kind == 'tail' else 1
    solid = (grid[:7] == 3)
    rr, cc = np.nonzero(solid)
    expected[rr + delta, cc] = 3
    if kind == 'closed':
        expected[12, 3:5] = 1
    else:
        expected[4 + delta, 3:5] = 1
    return {'input': grid, 'output': expected}


def _merge_fixture():
    grid = np.zeros((16, 13), dtype=np.int8)
    grid[8, 1:-1], grid[8, [0, -1]] = 8, 1
    for top, left in ((2, 1), (3, 7)):
        grid[top:top + 4, left:left + 4] = 3
        grid[top + 1:top + 3, left + 1:left + 3] = 0
        grid[top + 1, left + 1:left + 3] = 1
    grid[9:, [0, -1]] = 5
    grid[13:] = 5
    grid[13:15, 4:8] = 0
    expected = grid.copy()
    expected[:8] = 0
    for left in (1, 7):
        expected[4:8, left:left + 4] = 3
        expected[5:7, left + 1:left + 3] = 0
    expected[14, 4:8] = 1
    return {'input': grid, 'output': expected}


def self_test() -> dict:
    """Bounded 42-seed audit, targeted semantics, count and mutation checks."""
    for kind in ('closed', 'open', 'tail'):
        pair = _fixture(kind)
        _require(np.array_equal(solve(pair['input']), pair['output']), "explicit cavity fixture failed")
        _inspect(pair, deep=True)
    merged = _merge_fixture()
    _require(np.array_equal(solve(merged['input']), merged['output']), "explicit pooling fixture failed")
    _require(_inspect(merged, deep=True)['shared'] == 1, "pooling fixture no longer shares a receiver")
    signatures, structures, shapes, palettes = set(), set(), set(), set()
    counts, volumes, profiles, totals = set(), set(), set(), {}
    first = None
    for seed in range(42):
        # Vary explicit counts while retaining the required defaults elsewhere.
        nt = 1 + seed % 4
        nr = max(nt + 1, 2) + (seed // 4) % (11 - max(nt + 1, 2))
        puzzle = generate_puzzle(seed, nr, nt)
        if seed == 0:
            first = puzzle
        digest = hashlib.sha256()
        counts.add((nr, nt))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                stats = _inspect(pair, deep=True)
                shapes.add(stats['shape'])
                palettes.add(stats['roles'])
                volumes.add(stats['volume'])
                profiles.add((stats['shape'][1], stats['profile']))
                for name in ('objects', 'active', 'open_wet', 'recessed_wet', 'shared',
                             'split_body', 'multilevel', 'lateral', 'stepped', 'zero', 'moved',
                             'nonfull_donors', 'unequal_donors', 'joined_pockets'):
                    totals[name] = totals.get(name, 0) + stats[name]
                s = _scene(pair['input'])
                normalized = np.full(s['grid'].shape, 3, dtype=np.int8)
                normalized[s['grid'] == s['bg']] = 0
                normalized[s['grid'] == s['water']] = 1
                normalized[s['grid'] == s['rail']] = 2
                structures.add((normalized.shape, hashlib.sha256(normalized.tobytes()).digest()))
                for key in ('input', 'output'):
                    a = np.asarray(pair[key], dtype=np.int8)
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
        signatures.add(digest.digest())
    _require(len(signatures) == 42, "seed diversity collapsed")
    _require(len(structures) >= 200 and len(shapes) >= 35, "geometry diversity collapsed")
    _require(len(palettes) >= 30 and len(volumes) >= 30 and len(profiles) >= 150, "variation collapsed")
    _require(all(totals[k] >= 20 for k in ('shared', 'split_body', 'multilevel', 'lateral', 'stepped', 'zero',
                                        'nonfull_donors', 'unequal_donors')),
             "missing family-specific structural variation")
    _require(totals['joined_pockets'] >= 5, "no demonstrated overflow into communicating pockets")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(816, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, "count request ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.5, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p, q = generate_puzzle(2026, 4, 1), generate_puzzle(2026, 4, 1)
    _require(all(np.array_equal(x[k], y[k]) for split in p for x, y in zip(p[split], q[split])
                 for k in ('input', 'output')), "nondeterministic puzzle")
    original = first['test'][0]['output'].copy()
    first['test'][0]['output'][0, 0] = (int(original[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(first)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupt answer")
    first['test'][0]['output'] = original
    # The generator itself never stores this compact summary.
    return dict(seeds=42, puzzles=len(signatures), structures=len(structures),
                shapes=len(shapes), role_palettes=len(palettes), volumes=len(volumes),
                receiver_profiles=len(profiles), count_combinations=len(counts),
                totals={k: int(v) for k, v in totals.items()})
