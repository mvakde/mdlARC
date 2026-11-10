"""Cavity pistons: leading negative space creates the stops for matching tokens.

Only NumPy and the standard library are used. No parent module, example, seed,
palette cache, or construction metadata is consulted by solve().
"""

from collections import Counter
import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr), min(cc), max(cc)


def _rect(r, c, h, w):
    return {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _is_rectangle(cells):
    r0, r1, c0, c1 = _bbox(cells)
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _regions(mask):
    """Each complement component, its enclosure, and its leading-wall contact."""
    h, w = mask.shape
    result = []
    for cells in _components(~mask):
        enclosed = not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        rows = {r for r, c in cells}
        contact = {r for r, c in cells if c == w - 2 and mask[r, w - 1]}
        result.append((cells, enclosed, rows, contact))
    return result


def _objects(side, bg, offset=0):
    result = []
    for cells in _components(side != bg):
        colors = {int(side[p]) for p in cells}
        _require(len(colors) == 1, "touching objects of different colors")
        global_cells = {(r, c + offset) for r, c in cells}
        r0, r1, c0, c1 = _bbox(global_cells)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in global_cells:
            mask[r - r0, c - c0] = True
        result.append(dict(cells=global_cells, color=colors.pop(), bbox=(r0, r1, c0, c1), mask=mask))
    return result


def _scene(grid):
    """The exclusive full rail and nonrectangular mold side fix orientation."""
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        bg = int(a[0, 0])
        if not all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))):
            continue
        for rail in range(1, w - 1):
            color = int(a[0, rail])
            if color == bg or not np.all(a[:, rail] == color):
                continue
            if np.count_nonzero(a == color) != h:
                continue
            if any(np.any(a[r, np.arange(w) != rail] != bg) for r in (0, h - 1)):
                continue
            molds = _objects(a[:, :rail], bg)
            tokens = _objects(a[:, rail + 1:], bg, rail + 1)
            if not molds or not tokens or all(_is_rectangle(o['cells']) for o in molds):
                continue
            if not all(_is_rectangle(o['cells']) for o in tokens):
                continue
            candidates.append((a, turn, bg, rail, color, molds, tokens))
    _require(len(candidates) == 1, "expected one rail-relative interpretation")
    return candidates[0]


def _analyze(input_grid, region_mode="proper", match_mode="exact"):
    a, turn, bg, rail, rail_color, molds, tokens = _scene(_grid(input_grid))
    ports, statistics = [], Counter()
    last_row = -2
    for obj in molds:
        r0, r1, c0, c1 = obj['bbox']
        _require(r0 > last_row + 1, "molds must occupy separated perpendicular bands")
        last_row = r1
        obj['shift'] = rail - 1 - c1
        _require(obj['shift'] >= 0, "negative mold motion")
        for cells, enclosed, rows, contact in _regions(obj['mask']):
            if enclosed and contact:
                _require(rows == contact, "partly leading cavity is outside this family")
            statistics['open'] += bool(contact and not enclosed)
            statistics['recessed'] += bool(enclosed and not contact)
            accept = enclosed and bool(contact)
            if region_mode == 'all_holes':
                accept = enclosed
            elif region_mode == 'open_too':
                accept = bool(contact)
            if not accept:
                continue
            # After alignment, reflect the cavity into a positive extrusion.
            shadow = {(r0 + r, rail + obj['mask'].shape[1] - 1 - c) for r, c in cells}
            _require(all(0 <= r < a.shape[0] and rail < c < a.shape[1] for r, c in shadow),
                     "projected cavity exceeds the grid")
            pr0, pr1, _, tip = _bbox(shadow)
            ports.append(dict(cells=shadow, r0=pr0, r1=pr1, tip=tip,
                              color=obj['color'], source=cells, mold=obj))
    matches, used, routes = [], set(), set()
    for port in ports:
        candidates = []
        for ti, token in enumerate(tokens):
            r0, r1, c0, c1 = token['bbox']
            span_ok = (r0, r1) == (port['r0'], port['r1'])
            if match_mode == 'contained':
                span_ok = port['r0'] <= r0 <= r1 <= port['r1']
            if not span_ok or c0 <= port['tip']:
                continue
            start = port['tip'] + 1
            route = _rect(r0, start, r1 - r0 + 1, c1 - start + 1)
            if any(int(a[p]) != bg and p not in token['cells'] for p in route):
                continue
            final = _rect(r0, start, r1 - r0 + 1, c1 - c0 + 1)
            candidates.append(dict(ti=ti, port=port, route=route, final=final,
                                   distance=c0 - start, length=c1 - c0 + 1))
        _require(len(candidates) <= 1, "ambiguous token assignment")
        if candidates:
            match = candidates[0]
            _require(match['ti'] not in used, "token matches two pistons")
            _require(routes.isdisjoint(match['route']), "overlapping token routes")
            used.add(match['ti'])
            routes.update(match['route'])
            matches.append(match)
    return dict(grid=a, turn=turn, bg=bg, rail=rail, rail_color=rail_color,
                molds=molds, tokens=tokens, ports=ports, matches=matches,
                unmatched=len(tokens) - len(used), statistics=statistics)


def _render(analysis, insert=True):
    a = analysis['grid']
    out = np.full(a.shape, analysis['bg'], dtype=np.int8)
    out[:, analysis['rail']] = analysis['rail_color']
    for mold in analysis['molds']:
        _paint(out, {(r, c + mold['shift']) for r, c in mold['cells']}, mold['color'])
    for port in analysis['ports']:
        _paint(out, port['cells'], port['color'])
    if insert:
        for match in analysis['matches']:
            _paint(out, match['route'] - match['final'], analysis['rail_color'])
            _paint(out, match['final'], analysis['tokens'][match['ti']]['color'])
    return np.rot90(out, -analysis['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover geometry and apply the fixed rule using this input alone."""
    return _render(_analyze(input_grid))


def _rand(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _body(rng, heights, depths, active=None, stepped=False):
    """Construct independent compartments, recording their intended void cells."""
    active = [True] * len(heights) if active is None else active
    top, bottom, back = _rand(rng, 1, 2), _rand(rng, 1, 2), _rand(rng, 1, 2)
    gaps = [_rand(rng, 1, 2) for _ in heights[1:]]
    width = max(depths) + back + 1 + (1 if not all(active) else 0)
    mask = np.ones((top + bottom + sum(heights) + sum(gaps), width), dtype=bool)
    slots, row = [], top
    for index, (height, depth, is_active) in enumerate(zip(heights, depths, active)):
        end = width - 2 if is_active else width - 3
        profile = [depth] * height
        if stepped and height >= 2 and depth >= 2:
            profile = list(map(int, rng.integers(1, depth + 1, size=height)))
            profile[0], profile[-1] = depth, 1
        cells = {(row + rr, c) for rr, d in enumerate(profile) for c in range(end - d + 1, end + 1)}
        for p in cells:
            mask[p] = False
        slots.append(dict(cells=cells, active=is_active, mode='match'))
        row += height + (gaps[index] if index < len(gaps) else 0)
    return dict(mask=mask, slots=slots)


def _open(obj, rng, back_only=False):
    obj = dict(mask=obj['mask'].copy(), slots=[dict(s) for s in obj['slots']])
    cells = obj['slots'][0]['cells']
    r0, r1, c0, c1 = _bbox(cells)
    side = 0 if back_only else _rand(rng, 0, 2)
    if side == 0:
        row = _rand(rng, r0, r1)
        obj['mask'][row, :min(c for r, c in cells if r == row) + 1] = False
    elif side == 1:
        obj['mask'][:r0 + 1, c1] = False
    else:
        obj['mask'][r1:, c1] = False
    obj['slots'][0]['active'] = False
    return obj


def _recess(obj, rng):
    extension = _rand(rng, 1, 2)
    mask = np.pad(obj['mask'], ((0, 0), (0, extension)))
    mask[0 if rng.integers(2) else -1, -extension:] = True
    return dict(mask=mask, slots=[dict(s, active=False) for s in obj['slots']])


def _roughen(obj, rng):
    """Small silhouette edits may not change any intended enclosed cavity."""
    mask = obj['mask'].copy()
    target = {frozenset(c) for c, enclosed, _, _ in _regions(mask) if enclosed}
    h, w = mask.shape
    for _ in range(5):
        p = (_rand(rng, 0, h - 1), 0) if rng.integers(2) else (int(rng.choice([0, h - 1])), _rand(rng, 0, w - 2))
        if not mask[p]:
            continue
        trial = mask.copy()
        trial[p] = False
        if not trial[0].any() or not trial[-1].any() or not trial[:, 0].any():
            continue
        if len(_components(trial)) != 1:
            continue
        actual = {frozenset(c) for c, enclosed, _, _ in _regions(trial) if enclosed}
        if target == actual:
            mask = trial
    return dict(mask=mask, slots=obj['slots'])


def _random_object(rng, max_depth, *, multi=False, combined=False):
    count = 3 if combined else _rand(rng, 1, 2) if multi else 1
    heights = [_rand(rng, 1, 3) for _ in range(count)]
    depths = [_rand(rng, 1, max_depth) for _ in range(count)]
    if combined:
        active = [True, True, False]
        rng.shuffle(active)
    else:
        active = [True] * count
    obj = _body(rng, heights, depths, active, stepped=bool(rng.integers(2)))
    kind = 'mixed' if combined else str(rng.choice(['active', 'active', 'open', 'recessed']))
    if count == 1 and kind == 'open':
        obj = _open(obj, rng)
    elif kind == 'recessed':
        obj = _recess(obj, rng)
    for slot in obj['slots']:
        slot['mode'] = str(rng.choice(['match', 'match', 'wrong', 'none']))
        if not slot['active']:
            slot['mode'] = 'match'  # A convincing aligned but inactive lure.
    return _roughen(obj, rng)


def _specifications(rng, mode, regime):
    max_depth = 6 if regime == 'test' and mode == 'depth' else 3
    if regime == 'contrast':
        h, d = _rand(rng, 2, 3), _rand(rng, 2, 3)
        closed = _body(rng, [h], [d])
        opened = _open(closed, rng, back_only=True)
        closed['slots'][0]['distance'] = _rand(rng, 2, 5)
        objects = [closed, opened]
    elif regime == 'keys':
        wrong_height = _rand(rng, 3, 4)
        heights = [wrong_height - 1, wrong_height]
        depths = [_rand(rng, 2, 3), _rand(rng, 1, 3)]
        multi = _body(rng, heights, depths, stepped=True)
        multi['slots'][0].update(length=_rand(rng, 2, 3), distance=1)
        multi['slots'][1].update(mode='wrong', distance=_rand(rng, 1, 4))
        recess = _recess(_body(rng, [_rand(rng, 1, 3)], [_rand(rng, 1, 3)]), rng)
        objects = [_roughen(multi, rng), _roughen(recess, rng)]
    else:
        if regime == 'test' and mode == 'combine':
            first = _random_object(rng, max_depth, combined=True)
        elif regime == 'test' and mode == 'depth':
            first = _body(rng, [_rand(rng, 2, 4)], [_rand(rng, 4, 6)], stepped=True)
        else:
            first = _body(rng, [_rand(rng, 1, 3)], [_rand(rng, 1, max_depth)], stepped=bool(rng.integers(2)))
        next(s for s in first['slots'] if s['active'])['mode'] = 'match'
        objects = [first]
        for _ in range(_rand(rng, 0, 3)):
            obj = _random_object(rng, max_depth, multi=bool(rng.integers(2)))
            if sum(o['mask'].shape[0] for o in objects) + obj['mask'].shape[0] + len(objects) + 2 <= 29:
                objects.append(obj)
    rng.shuffle(objects)
    return objects


def _generate_pair(rng, mode, regime, turn):
    objects = _specifications(rng, mode, regime)
    plans = []
    for oi, obj in enumerate(objects):
        for slot in obj['slots']:
            if slot['mode'] == 'none':
                continue
            r0, r1, _, _ = _bbox(slot['cells'])
            height = r1 - r0 + 1
            if slot['mode'] == 'wrong' and height == 1 and len(obj['slots']) > 1:
                continue
            token_height = height - 1 if slot['mode'] == 'wrong' and height >= 2 else height
            row = r0
            if slot['mode'] == 'wrong' and height == 1:
                row += 1  # Same size, one row off the axis.
            depth = max(obj['mask'].shape[1] - 1 - c for r, c in slot['cells'])
            plans.append(dict(oi=oi, slot=slot, row=row, height=token_height,
                              length=slot.get('length', _rand(rng, 1, 3)), depth=depth))
    _require(plans, 'construction needs visible tokens')
    widest = max(o['mask'].shape[1] for o in objects)
    right_min = max(p['depth'] + p['length'] + p['slot'].get('distance', 0) for p in plans) + 1
    right_min = max(right_min, max(o['mask'].shape[1] for o in objects))
    slack = 30 - widest - 1 - right_min
    _require(slack >= 1, 'width budget exhausted')
    work_extra = _rand(rng, 1, min(4, slack))
    rail = widest + work_extra
    right = right_min + _rand(rng, 0, min(7, slack - work_extra))
    width = rail + 1 + right
    base_height = sum(o['mask'].shape[0] for o in objects) + len(objects) + 1
    height = _rand(rng, base_height, min(30, base_height + 5))
    extras = rng.multinomial(height - base_height, np.full(len(objects) + 1, 1 / (len(objects) + 1)))
    roles = list(map(int, rng.permutation(10)))
    bg, rail_color = roles[:2]
    mold_palette, token_palette = roles[2:5], roles[5:9]
    a = np.full((height, width), bg, dtype=np.int8)
    out = a.copy()
    a[:, rail] = out[:, rail] = rail_color
    row = 1 + int(extras[0])
    for index, obj in enumerate(objects):
        mask = obj['mask']
        obj['row'] = row
        max_shift = rail - mask.shape[1]
        shift = 0 if rng.random() < .3 else _rand(rng, 1, max_shift)
        if regime == 'contrast':
            shift = 0 if index == 0 else _rand(rng, 1, max_shift)
        obj['start'] = rail - mask.shape[1] - shift
        obj['color'] = int(rng.choice(mold_palette))
        view = a[row:row + mask.shape[0], obj['start']:obj['start'] + mask.shape[1]]
        view[mask] = obj['color']
        final_view = out[row:row + mask.shape[0], rail - mask.shape[1]:rail]
        final_view[mask] = obj['color']
        for slot in obj['slots']:
            if slot['active']:
                shadow = {(row + r, rail + mask.shape[1] - 1 - c) for r, c in slot['cells']}
                _paint(out, shadow, obj['color'])
        row += mask.shape[0] + 1 + int(extras[index + 1])
    for p in plans:
        obj, slot = objects[p['oi']], p['slot']
        max_distance = right - 1 - p['depth'] - p['length']
        distance = slot.get('distance', _rand(rng, 0, min(5, max_distance)))
        _require(distance <= max_distance, 'token outside right margin')
        c0 = rail + p['depth'] + 1 + distance
        r0 = obj['row'] + p['row']
        cells = _rect(r0, c0, p['height'], p['length'])
        color = int(rng.choice(token_palette))
        _paint(a, cells, color)
        if slot['active'] and slot['mode'] == 'match':
            start = rail + p['depth'] + 1
            final = _rect(r0, start, p['height'], p['length'])
            route = _rect(r0, start, p['height'], distance + p['length'])
            _paint(out, route - final, rail_color)
            _paint(out, final, color)
    # Scattered lures use only unoccupied perpendicular bands, keeping paths clear.
    free_rows = [r for r in range(1, height - 1)
                 if all(not (o['row'] <= r < o['row'] + o['mask'].shape[0]) for o in objects)]
    rng.shuffle(free_rows)
    for r in free_rows[:_rand(rng, 0, 2)]:
        length = _rand(rng, 1, min(3, right - 2))
        possible = [c for c in range(rail + 2, width - length)
                    if np.all(a[r - 1:r + 2, c - 1:c + length + 1] == bg)]
        if possible:
            c = int(rng.choice(possible))
            _paint(a, _rect(r, c, 1, length), int(rng.choice(token_palette)))
    return {'input': np.rot90(a, turn).copy(), 'output': np.rot90(out, turn).copy()}


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _unique(pair, seen):
    # Twenty distinct dihedral color permutations; at most thirteen are blocked.
    for sign in (1, -1):
        for offset in range(10):
            lookup = (sign * np.arange(10) + offset) % 10
            candidate = {k: lookup[v].astype(np.int8) for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('twenty-color-orbit uniqueness fallback exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} outside allowed range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(['combine', 'depth', 'turn']))
    base_turn = _rand(rng, 0, 3)
    mixed = mode == 'turn' or bool(rng.integers(2))
    teaching = ['contrast', 'keys'] + ['train'] * (num_train - 2)
    rng.shuffle(teaching)
    puzzle, seen = {'train': [], 'test': []}, set()
    teaching_turns = {'contrast': base_turn, 'keys': (base_turn + int(mixed)) % 4}
    for split, regimes in (('train', teaching), ('test', ['test'] * num_test)):
        for index, regime in enumerate(regimes):
            if split == 'test' and mode == 'turn':
                turn = (base_turn + 2 + index % 2) % 4
            elif regime in teaching_turns:
                turn = teaching_turns[regime]
            else:
                turn = (base_turn + (_rand(rng, 0, 1) if mixed else 0)) % 4
            pair = _unique(_generate_pair(rng, mode, regime, turn), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _ablation_witnesses(inp, out):
    analysis = _analyze(inp)
    return dict(enclosure=not np.array_equal(_render(_analyze(inp, 'open_too')), out),
                extreme=not np.array_equal(_render(_analyze(inp, 'all_holes')), out),
                span=not np.array_equal(_render(_analyze(inp, match_mode='contained')), out),
                insertion=not np.array_equal(_render(analysis, insert=False), out))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'invalid counts')
    seen, witnesses = set(), Counter()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'changed dimensions')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            analysis = _analyze(inp)
            _require(np.array_equal(_render(analysis), out), 'incorrect solution')
            a, rail, bg = analysis['grid'], analysis['rail'], analysis['bg']
            o = np.rot90(out, analysis['turn'])
            _require(np.array_equal(a[:, rail], o[:, rail]), 'rail changed')
            mold_colors = {m['color'] for m in analysis['molds']}
            token_colors = {t['color'] for t in analysis['tokens']}
            _require(mold_colors.isdisjoint(token_colors), 'mold/token role palettes overlap')
            shadow, route = set(), set()
            for port in analysis['ports']:
                _require(shadow.isdisjoint(port['cells']), 'overlapping projections')
                shadow.update(port['cells'])
                _require(all(int(a[p]) == bg for p in port['cells']), 'token obstructs a projection')
                _require(all(int(o[p]) == port['color'] for p in port['cells']), 'incorrect piston')
            for match in analysis['matches']:
                _require(shadow.isdisjoint(match['route']), 'piston intersects token route')
                _require(route.isdisjoint(match['route']), 'routes intersect')
                route.update(match['route'])
                token = analysis['tokens'][match['ti']]
                _require(all(int(o[p]) == token['color'] for p in match['final']), 'token color/shape changed')
                _require(len(match['final']) == len(token['cells']), 'token not translated rigidly')
            _require(analysis['ports'] and analysis['matches'], 'pair must use both mechanisms')
            if split == 'train':
                witnesses.update({k: int(v) for k, v in _ablation_witnesses(inp, out).items()})
                witnesses['motion'] += any(m['shift'] > 0 for m in analysis['molds'])
                witnesses['aligned'] += any(m['shift'] == 0 for m in analysis['molds'])
                witnesses['short'] += any(0 < m['distance'] < m['length'] for m in analysis['matches'])
                witnesses['stepped'] += any(not _is_rectangle(p['cells']) for p in analysis['ports'])
    _require(all(witnesses[k] > 0 for k in ('enclosure', 'extreme', 'span', 'insertion',
                                           'motion', 'aligned', 'short', 'stepped')),
             'training does not distinguish the required mechanisms and alternatives')


def measure_puzzle(puzzle):
    """Small pixel-derived review summary; no sampled configuration is returned."""
    result = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            a = _analyze(pair['input'])
            result.append(dict(split=split, index=index, shape=list(a['grid'].shape), turn=a['turn'],
                               rail=a['rail'], bg=a['bg'], mold_count=len(a['molds']),
                               token_count=len(a['tokens']), ports=len(a['ports']),
                               matched=len(a['matches']), unmatched=a['unmatched'],
                               open=a['statistics']['open'], recessed=a['statistics']['recessed'],
                               depths=[p['tip'] - a['rail'] for p in a['ports']],
                               spans=[p['r1'] - p['r0'] + 1 for p in a['ports']],
                               shifts=[m['shift'] for m in a['molds']],
                               distances=[m['distance'] for m in a['matches']],
                               stepped=sum(not _is_rectangle(p['cells']) for p in a['ports']),
                               ablations=_ablation_witnesses(pair['input'], pair['output'])))
    return result


def self_test():
    """42 sequential seeds, bounded hashes/statistics, no files or saved traces."""
    shapes, normalized, masks, depths, spans, counts, turns, palettes = (set() for _ in range(8))
    puzzle_hashes, totals = set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                shapes.add(pair['input'].shape)
                digest.update(bytes(pair['input'].shape) + pair['input'].tobytes() + pair['output'].tobytes())
                lookup = {}
                norm = bytes(lookup.setdefault(int(c), len(lookup)) for c in pair['input'].flat)
                normalized.add(hashlib.sha256(bytes(pair['input'].shape) + norm).digest())
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                counts.add(len(a['molds']))
                turns.add(a['turn'])
                for mold in a['molds']:
                    masks.add((mold['mask'].shape, mold['mask'].tobytes()))
                    totals['aligned'] += mold['shift'] == 0
                for port in a['ports']:
                    depths.add(port['tip'] - a['rail'])
                    spans.add(port['r1'] - port['r0'] + 1)
                    totals['stepped'] += not _is_rectangle(port['cells'])
                totals['short'] += sum(0 < m['distance'] < m['length'] for m in a['matches'])
                totals.update(a['statistics'])
                original = pair['input'].copy()
                _require(np.array_equal(solve(original), pair['output']), 'input-only solve failed')
                _require(np.array_equal(original, pair['input']), 'solver mutated its argument')
        puzzle_hashes.add(digest.digest())
    _require(len(puzzle_hashes) == 42 and len(normalized) >= 180, 'structural variety collapsed')
    _require(len(shapes) >= 45 and len(masks) >= 120 and len(palettes) >= 40, 'geometry/palette variety collapsed')
    _require(turns == set(range(4)) and {1, 2, 3} <= counts, 'orientation/count variety collapsed')
    _require(set(range(1, 7)) <= depths and {1, 2, 3, 4} <= spans, 'cavity scale variety collapsed')
    _require(all(totals[k] >= 30 for k in ('open', 'recessed', 'stepped', 'aligned', 'short')),
             'missing important structural contrasts')
    # Every explicit legal request, plus type/range/relation errors.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), 'counts ignored')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(719, 2, 1), generate_puzzle(719, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), 'nondeterministic puzzle')
    # Analytical fixtures cover longer-than-sampled tokens and silent cases.
    # Explicit depths are 3,1,2; the whole rectangle must stop at depth three.
    mask = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1],
                     [1, 1, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=bool)
    for case in ('short', 'zero', 'wrong', 'open', 'recessed'):
        inp = np.full((7, 20), 4, dtype=np.int8)
        expected = inp.copy()
        inp[:, 8] = expected[:, 8] = 0
        work = mask.copy()
        if case == 'open':
            work[2, :4] = False
        inp[1:6, 1:6][work] = 6
        if case == 'recessed':
            inp[1, 6] = 6
            expected[1:6, 2:7][work] = 6
            expected[1, 7] = 6
        else:
            expected[1:6, 3:8][work] = 6
        start, token_height = (12 if case == 'zero' else 13), (2 if case == 'wrong' else 3)
        inp[2:2 + token_height, start:start + 5] = 2
        if case not in ('open', 'recessed'):
            expected[2, 9:12] = expected[3, 9:10] = expected[4, 9:11] = 6
            if case != 'wrong':
                expected[2:5, 12:17] = 2
                if case == 'short':
                    expected[2:5, 17] = 0
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn)),
                     f'analytical fixture failed: {case}/{turn}')
    # A schema-valid wrong token must not escape semantic validation.
    a = _analyze(p['test'][0]['input'])
    cell = min(a['matches'][0]['final'])
    changed = np.rot90(p['test'][0]['output'], a['turn']).copy()
    changed[cell] = a['rail_color']
    p['test'][0]['output'] = np.rot90(changed, -a['turn']).copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted token accepted')
    return dict(seeds=42, puzzles=len(puzzle_hashes), input_structures=len(normalized),
                grid_shapes=len(shapes), mold_masks=len(masks), input_palettes=len(palettes),
                depths=sorted(map(int, depths)), spans=sorted(map(int, spans)), mold_counts=sorted(counts),
                turns=sorted(turns), totals={k: int(v) for k, v in totals.items()})
