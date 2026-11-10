"""Shared-odometer telescopes: input-only geometry, bounded construction.

Only standard-library modules and NumPy are used. No files are read at runtime.
The constructive labeler uses a distance formula; solve walks shared clock ticks.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
HOST_SEEDS = (0,32248542,739284950,419960977,232544132,613388240,340621486,
              932203773,593247277,59591989,635733170,232548720,550822379,
              7772253,340207919,891458890,818807276,638078631,872433760,
              819912351,977356831,97047564,93039141,92711556,146558855,
              569305,203639071,849438489,916137726,521484580,712266879,
              31198201,423556616,273661139,150003387,658774282,943511909,
              824999464,777365129,793338147,139275324,97766416)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid bounds")
    _need(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _need(np.all((0 <= g) & (g <= 9)), "ARC colors required")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    y, x = center
    return [(y + a, x + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius]


def _cap(center, radius, direction, step=0):
    y, x = center
    dy, dx = DIRECTIONS[direction]
    return [(y + dy * (k + 1 + step), x + dx * (radius - k + 1 + step))
            for k in range(radius + 1)]


def _meter(center, shape):
    y, x = center
    h, w = shape
    return [(r, c) for r in range(y - h // 2, y + h // 2 + 1)
            for c in range(x - w // 2, x + w // 2 + 1)]


def _body(e):
    return set(_outline(e['center'], e['radius'])) | set(_meter(e['center'], e['meter_shape']))


def _sweep(e):
    result = _body(e)
    for d, arm in e['arms'].items():
        for t in range(arm['distance'] + 1):
            result.update(_cap(e['center'], e['radius'], d, t))
    return result


def _background(g):
    parity = np.indices(g.shape).sum(axis=0) % 2
    colors = []
    for p in (0, 1):
        counts = np.bincount(g[parity == p].astype(int), minlength=10)
        color = int(counts.argmax())
        _need(int(counts[color]) * 2 > int(np.sum(parity == p)), "checkerboard majority")
        colors.append(color)
    _need(colors[0] != colors[1], "two checkerboard colors required")
    return tuple(colors)


def _analyze(input_grid):
    """Recover all roles, meters, caps and destinations from input pixels only."""
    g = _grid(input_grid)
    background = _background(g)
    base = _base(g.shape, background)
    foreground = {tuple(map(int, p)) for p in np.argwhere(g != base)}
    emitters = []
    h, w = g.shape
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        shape = h - 2 * radius, w - 2 * radius
        offsets = _outline((0, 0), radius)
        a, b = offsets[0]
        colors = g[radius+a:radius+a+shape[0], radius+b:radius+b+shape[1]]
        mask = (colors != background[0]) & (colors != background[1])
        for a, b in offsets[1:]:
            mask &= g[radius+a:radius+a+shape[0], radius+b:radius+b+shape[1]] == colors
        for yy, xx in np.argwhere(mask):
            center = (int(yy) + radius, int(xx) + radius)
            struct = int(colors[yy, xx])
            interior = {p for p in foreground
                        if abs(p[0]-center[0]) + abs(p[1]-center[1]) < radius}
            if not interior:
                continue
            ys, xs = zip(*interior)
            mh, mw = max(ys)-min(ys)+1, max(xs)-min(xs)+1
            if (mh % 2 != 1 or mw % 2 != 1 or
                    (min(ys)+max(ys), min(xs)+max(xs)) != (2*center[0], 2*center[1]) or
                    mh//2 + mw//2 >= radius):
                continue
            coordinates = _meter(center, (mh, mw))
            if set(coordinates) != interior:
                continue
            values = [int(g[p]) for p in coordinates]
            other = set(values) - {struct}
            if len(other) > 1:
                continue
            charge = next(iter(other)) if other else None
            spent = values.count(struct)
            if values != [struct]*spent + [charge]*(len(values)-spent):
                continue
            arms = {}
            valid = True
            for d in range(4):
                cap = _cap(center, radius, d)
                if not all(_inside(p, g.shape) for p in cap):
                    continue
                payload = int(g[cap[0]])
                if payload in (*background, struct) or any(g[p] != payload for p in cap):
                    continue
                destination = None
                for distance in range(1, 31):
                    shifted = _cap(center, radius, d, distance)
                    if not all(_inside(p, g.shape) for p in shifted):
                        break
                    if all(g[p] == payload for p in shifted):
                        destination = distance
                        break
                    if any(g[p] != base[p] for p in shifted):
                        break
                if destination is None:
                    valid = False
                    break
                arms[d] = dict(color=payload, distance=destination)
            if valid and arms:
                emitters.append(dict(center=center, radius=radius, struct=struct,
                                     charge=charge, meter_shape=(mh, mw), spent=spent,
                                     arms=arms))
    _need(1 <= len(emitters) <= 3, "one through three uniquely readable emitters required")
    covered, sweeps = set(), set()
    for e in emitters:
        local = _body(e)
        local_sweep = set(local)
        for d, arm in e['arms'].items():
            start = set(_cap(e['center'], e['radius'], d))
            target = set(_cap(e['center'], e['radius'], d, arm['distance']))
            path = {p for t in range(arm['distance'] + 1)
                    for p in _cap(e['center'], e['radius'], d, t)}
            _need(not local_sweep & path, "overlapping arms or meter")
            _need(all(g[p] == base[p] for p in path - start - target), "obstructed arm")
            local_sweep.update(path)
            local.update(start | target)
        _need(not covered & local and not sweeps & local_sweep, "shared objects or sweeps")
        covered.update(local)
        sweeps.update(local_sweep)
    _need(covered == foreground, "unexplained foreground or false object detection")
    return dict(background=background, emitters=sorted(emitters, key=lambda e: e['center']))


def _formula_prediction(g, emitters, *, unlimited=False, solid=False,
                        summed=False, first_stop=False):
    """Closed-form labeler and explicitly defined shortcut counterfactuals."""
    out = g.copy()
    for e in emitters:
        coords = _meter(e['center'], e['meter_shape'])
        fuel = len(coords) - e['spent']
        distances = [a['distance'] for a in e['arms'].values()]
        duration = min(distances) if first_stop else max(distances)
        clock = duration if unlimited else min(fuel, duration)
        used = min(fuel, sum(distances)) if summed else min(fuel, clock)
        for p in coords[e['spent']:e['spent']+used]:
            out[p] = e['struct']
        for d, arm in e['arms'].items():
            m = min(clock, arm['distance'])
            dy, dx = DIRECTIONS[d]
            start = _cap(e['center'], e['radius'], d)
            if solid:
                for t in range(1, m+1):
                    for y, x in start:
                        out[y+dy*t, x+dx*t] = arm['color']
            else:
                for y, x in (start[0], start[-1]):
                    for t in range(1, m+1):
                        out[y+dy*t, x+dx*t] = arm['color']
            for y, x in start:
                out[y+dy*m, x+dx*m] = arm['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Simulate synchronous extension using only the given visible input."""
    g = _grid(input_grid)
    analysis = _analyze(g)
    out = g.copy()
    for e in analysis['emitters']:
        pending = _meter(e['center'], e['meter_shape'])[e['spent']:]
        tick = 0
        for cell in pending:
            if all(tick >= a['distance'] for a in e['arms'].values()):
                break
            out[cell] = e['struct']
            tick += 1
            for d, arm in e['arms'].items():
                if tick <= arm['distance']:
                    shifted = _cap(e['center'], e['radius'], d, tick)
                    out[shifted[0]] = out[shifted[-1]] = arm['color']
        for d, arm in e['arms'].items():
            for cell in _cap(e['center'], e['radius'], d, min(tick, arm['distance'])):
                out[cell] = arm['color']
    return out


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _sample_spec(rng, maximum_radius, directions, background, *, lesson=None,
                 compact=False, palette=None):
    radius = maximum_radius if lesson else int(rng.integers(2, maximum_radius+1))
    shapes = [(h, w) for h in range(1, 2*radius, 2) for w in range(1, 2*radius, 2)
              if h//2 + w//2 < radius]
    if lesson:
        shapes = [(3, 3)] if radius == 3 else [(3, 3), (3, 5), (5, 3)]
    meter_shape = shapes[int(rng.integers(len(shapes)))]
    capacity = meter_shape[0] * meter_shape[1]
    if lesson == 'fuel':
        fuel = int(rng.integers(3, 6))
        distances = [fuel-1, fuel+1, fuel+2]
    elif lesson == 'dock':
        fuel = int(rng.integers(5, capacity))
        longer = int(rng.integers(2, min(fuel-1, 5)+1))
        distances = [int(rng.integers(1, longer)), longer]
    else:
        fuel = int(rng.integers(0, capacity+1))
        max_distance = 3 if compact else 10-radius
        distances = list(map(int, rng.integers(1, max_distance+1, size=len(directions))))
    rng.shuffle(distances)
    colors = [c for c in range(10) if c not in background]
    if palette is not None:
        colors = [c for c in palette if c not in background]
    struct, charge = map(int, rng.choice(colors, 2, replace=False))
    payloads = [c for c in colors if c != struct]
    if rng.random() < .30:
        cap_colors = [int(rng.choice(payloads))] * len(directions)
    else:
        cap_colors = list(map(int, rng.choice(payloads, len(directions))))
    return dict(center=(0, 0), radius=radius, struct=struct, charge=charge,
                meter_shape=meter_shape, spent=capacity-fuel,
                arms={d:dict(color=c, distance=int(t))
                      for d, c, t in zip(directions, cap_colors, distances)})


def _place(rng, specs):
    """Free packing of full swept boxes, without slots or corner templates."""
    boxes = []
    for e in specs:
        ys, xs = zip(*_sweep(e))
        boxes.append((min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1))
    if any(h > 30 or w > 30 for _, _, h, w in boxes):
        return None
    n = len(specs)
    min_h = max(b[2] for b in boxes)
    min_w = max(b[3] for b in boxes)
    low = 12 if n == 1 else 23
    H = int(rng.integers(max(low, min_h), 31))
    W = int(rng.integers(max(low, min_w), 31))
    occupied = np.zeros((H, W), dtype=bool)
    placed = []
    for index in sorted(range(n), key=lambda i: boxes[i][2]*boxes[i][3], reverse=True):
        y0, x0, h, w = boxes[index]
        padded = np.pad(occupied, 1)
        blocked = np.zeros_like(occupied)
        for dy in range(3):
            for dx in range(3):
                blocked |= padded[dy:dy+H, dx:dx+W]
        summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        scores = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
        positions = np.argwhere(scores == 0)
        if len(positions) == 0:
            return None
        if rng.random() < .25:
            edge = positions[(positions[:,0] == 0) | (positions[:,1] == 0) |
                             (positions[:,0]+h == H) | (positions[:,1]+w == W)]
            if len(edge):
                positions = edge
        y, x = map(int, positions[int(rng.integers(len(positions)))])
        e = dict(specs[index], center=(y-y0, x-x0))
        placed.append(e)
        occupied[y:y+h, x:x+w] = True
    return (H, W), placed


def _render(shape, background, specs):
    g = _base(shape, background)
    for e in specs:
        for p in _outline(e['center'], e['radius']):
            g[p] = e['struct']
        for i, p in enumerate(_meter(e['center'], e['meter_shape'])):
            g[p] = e['struct'] if i < e['spent'] else e['charge']
        for d, arm in e['arms'].items():
            for t in (0, arm['distance']):
                for p in _cap(e['center'], e['radius'], d, t):
                    g[p] = arm['color']
    # Labels are constructed from sampled geometry, not from solve or the parser.
    output = _formula_prediction(g, specs)
    recovered = _analyze(g)
    _need(np.array_equal(output, solve(g)), "constructive/visible reconstruction disagreement")
    _need(len(recovered['emitters']) == len(specs), "emitter recovery disagreement")
    return dict(input=g, output=output)


def _mask(directions):
    return sum(1 << d for d in directions)


def _directions(mask):
    return [d for d in range(4) if mask & (1 << d)]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    maximum_radius = int(rng.choice((3, 4)))
    held_mask = int(rng.choice((7, 11, 13, 14, 15)))
    train_masks = [m for m in range(1, 16) if m != held_mask]
    fuel_mask = int(rng.choice([m for m in train_masks if m.bit_count() == 3]))
    missing = next(d for d in range(4) if not fuel_mask & (1 << d))
    dock_mask = (1 << missing) | (1 << int(rng.choice([d for d in range(4) if d != missing])))
    slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    lessons = {slots[0]:('fuel', fuel_mask), slots[1]:('dock', dock_mask)}
    shared_palette = (tuple(map(int, rng.permutation(10)[:int(rng.integers(6,10))]))
                      if rng.random() < .35 else None)
    shared_background = tuple(map(int, rng.choice(10, 2, replace=False)))
    fixed_background = rng.random() < .4
    puzzle, seen = dict(train=[], test=[]), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson, lead_mask = lessons.get(index, (None, None)) if split == 'train' else (None, held_mask)
            desired_count = 1 if lesson else int(rng.choice((1, 2, 3), p=(.45, .4, .15)))
            accepted = None
            for attempt in range(100):
                background = shared_background if fixed_background else tuple(map(int, rng.choice(10, 2, replace=False)))
                specs = []
                for j in range(desired_count):
                    m = lead_mask if j == 0 and lead_mask is not None else int(rng.choice(train_masks))
                    dirs = _directions(m)
                    rmax = maximum_radius if desired_count == 1 or lesson else min(maximum_radius, 3)
                    specs.append(_sample_spec(rng, rmax, dirs, background,
                                              lesson=lesson if j == 0 else None,
                                              compact=desired_count > 1,
                                              palette=shared_palette))
                placement = _place(rng, specs)
                if placement is None:
                    continue
                shape, positioned = placement
                try:
                    pair = _render(shape, background, positioned)
                except AssertionError:
                    continue
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    accepted = pair
                    break
            if accepted is None:
                raise RuntimeError('bounded placement or unique-input construction exhausted')
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _different(pair, analysis, **kwargs):
    return not np.array_equal(pair['output'], _formula_prediction(pair['input'], analysis['emitters'], **kwargs))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "list splits")
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
          and len(puzzle['train']) > len(puzzle['test']), "pair counts")
    seen, train_masks, test_masks, directions = set(), set(), set(), set()
    teaching = dict(unlimited=False, solid=False, summed=False, first_stop=False)
    max_train_radius = max_test_radius = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, "output shape changed")
            signature = _key(inp)
            _need(signature not in seen, "duplicate input")
            seen.add(signature)
            analysis = _analyze(inp)
            expected = _formula_prediction(inp, analysis['emitters'])
            _need(np.array_equal(out, expected), "wrong odometer, rail, or frontier output")
            _need(np.array_equal(out, solve(inp)), "formula/tick mismatch")
            editable = set()
            for e in analysis['emitters']:
                m = _mask(e['arms'])
                if split == 'train':
                    train_masks.add(m)
                    directions.update(e['arms'])
                    max_train_radius = max(max_train_radius, e['radius'])
                else:
                    test_masks.add(m)
                    max_test_radius = max(max_test_radius, e['radius'])
                editable.update(_meter(e['center'], e['meter_shape']))
                fuel = e['meter_shape'][0] * e['meter_shape'][1] - e['spent']
                clock = min(fuel, max(a['distance'] for a in e['arms'].values()))
                for d, arm in e['arms'].items():
                    length = min(clock, arm['distance'])
                    for t in range(1, length+1):
                        cap = _cap(e['center'], e['radius'], d, t)
                        editable.update((cap[0], cap[-1]))
                    editable.update(_cap(e['center'], e['radius'], d, length))
                # Structural conservation: the outline and every fixed marker survive.
                fixed = set(_outline(e['center'], e['radius']))
                for d, arm in e['arms'].items():
                    fixed.update(_cap(e['center'], e['radius'], d))
                    fixed.update(_cap(e['center'], e['radius'], d, arm['distance']))
                _need(all(inp[p] == out[p] for p in fixed), "fixed geometry altered")
            changed = set(map(tuple, np.argwhere(inp != out)))
            _need(changed <= editable, "unrelated background altered")
            if split == 'train':
                p = dict(input=inp, output=out)
                for flag in teaching:
                    teaching[flag] |= _different(p, analysis, **{flag:True})
    _need(directions == set(range(4)), "training must show all four directions")
    _need(all(teaching.values()), "missing counterexample to a simpler rule")
    _need(bool(test_masks - train_masks), "missing active-side recombination shift")
    _need(max_test_radius <= max_train_radius, "untaught larger diamond")


def _features(pair):
    a = _analyze(pair['input'])
    objects = []
    for e in a['emitters']:
        cap = e['meter_shape'][0] * e['meter_shape'][1]
        fuel = cap-e['spent']
        ds = [v['distance'] for v in e['arms'].values()]
        objects.append(dict(radius=e['radius'], mask=_mask(e['arms']),
                            meter_shape=e['meter_shape'], capacity=cap, fuel=fuel,
                            time=min(fuel, max(ds)), distances=ds,
                            fresh=fuel == cap, exhausted=fuel == 0,
                            mixed=any(d < fuel for d in ds) and any(d > fuel for d in ds),
                            stop='fuel' if fuel < max(ds) else 'exact' if fuel == max(ds) else 'dock',
                            cap_colors=len({v['color'] for v in e['arms'].values()})))
    return dict(shape=pair['input'].shape, background=a['background'],
                count=len(objects), objects=objects,
                layout=(pair['input'].shape, tuple((e['center'], e['radius'],
                        e['meter_shape'], e['spent'], tuple((d,v['distance'])
                        for d,v in sorted(e['arms'].items()))) for e in a['emitters'])))


def self_test():
    """42 fixed seeds, streaming measurements; no samples or logs are written."""
    shapes, layouts, meters, masks, radii, palettes, counts = (set() for _ in range(7))
    signatures = set()
    stops, properties = Counter(), Counter()
    split_counts = {'train':set(), 'test':set()}
    direction_stops = set()
    total_pairs = 0
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                total_pairs += 1
                f = _features(pair)
                shapes.add(f['shape'])
                layouts.add(f['layout'])
                palettes.add(f['background'])
                counts.add(f['count'])
                split_counts[split].add(f['count'])
                digest.update(str(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                for e, obj in zip(_analyze(pair['input'])['emitters'], f['objects']):
                    radii.add(obj['radius'])
                    masks.add(obj['mask'])
                    meters.add(obj['meter_shape'])
                    stops[obj['stop']] += 1
                    for name in ('fresh', 'exhausted', 'mixed'):
                        properties[name] += int(obj[name])
                    for d in e['arms']:
                        direction_stops.add((d, obj['stop']))
                    properties['multicolor'] += int(obj['cap_colors'] > 1)
                    properties['long'] += int(max(obj['distances']) >= 6)
        signatures.add(digest.hexdigest())
    _need(len(signatures) == 42 and len(shapes) >= 50 and len(layouts) >= 190, "geometry diversity collapsed")
    _need(radii == {2,3,4} and len(meters) >= 8, "meter/radius variation collapsed")
    _need(masks == set(range(1,16)) and counts == {1,2,3}, "side/count variation collapsed")
    _need(all(split_counts[s] == {1,2,3} for s in split_counts), "split count variation collapsed")
    _need(len(palettes) >= 25 and all(properties[k] > 0 for k in properties), "palette or regime collapse")
    _need(direction_stops == {(d,s) for d in range(4) for s in ('fuel','exact','dock')}, "direction/stopping correlation")
    # The count extremes, every other legal combination, and bad types are finite.
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(73, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, "explicit counts ignored")
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),
                (2.5,1),(4,1.0),(None,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    first, repeat = generate_puzzle(0), generate_puzzle(0)
    _need(all(_key(a[k]) == _key(b[k]) for s in first for a,b in zip(first[s],repeat[s])
              for k in ('input','output')), "nondeterminism")
    # Reject missing rails, wrong meter use, and damage to unrelated background.
    for damage in ('rail', 'meter', 'background'):
        bad = {s:[{k:g.copy() for k,g in pair.items()} for pair in first[s]] for s in first}
        pair = bad['train'][0]
        a = _analyze(pair['input'])
        if damage == 'meter':
            cells = _meter(a['emitters'][0]['center'], a['emitters'][0]['meter_shape'])
        elif damage == 'rail':
            meter_cells = set().union(*(set(_meter(e['center'], e['meter_shape']))
                                        for e in a['emitters']))
            cells = [tuple(p) for p in np.argwhere(pair['input'] != pair['output'])
                     if tuple(p) not in meter_cells]
        else:
            occupied = set().union(*(_sweep(e) for e in a['emitters']))
            cells = [p for p in np.ndindex(pair['input'].shape) if p not in occupied]
        cell = cells[0]
        pair['output'][cell] = (int(pair['output'][cell])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'validator accepted {damage} corruption')
    return dict(seeds=42, pairs=total_pairs, shapes=len(shapes), layouts=len(layouts),
                backgrounds=len(palettes), meter_shapes=sorted(meters),
                radii=sorted(radii), masks=sorted(masks), counts=sorted(counts),
                split_counts={s:sorted(v) for s,v in split_counts.items()},
                stopping=dict(stops), properties=dict(properties))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
