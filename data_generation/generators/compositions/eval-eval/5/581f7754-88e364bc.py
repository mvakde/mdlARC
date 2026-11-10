"""Registration brakes: keyed rigid motion stopped by a guide or full-body contact.

All semantic information is read from the input. No parent modules, files, random
state, training labels, or generation records are used by solve(). Experiments
stream in memory; this module writes no files and saves no execution traces.
"""
from __future__ import annotations

import hashlib
import numpy as np

CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIRECTIONS = CARDINAL + DIAGONAL
SEEDS = (0, 312974271, 153764721, 765702321, 672914477, 442917383,
         581643439, 493513063, 223070972, 523852792, 580429645, 411317494,
         595982797, 634923951, 546160965, 391202738, 493769026, 811740223,
         403220772, 191598874, 686746777, 757802631, 106282129, 573028175,
         122891541, 819005477, 363777813, 171661965, 774611453, 872770821,
         387388825, 866707074, 962735596, 235901903, 575300439, 236806366,
         619811699, 56035911, 56824694, 42059199, 979653972, 303569450)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'illegal {name}')
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(np.all((a >= 0) & (a <= 9)), 'grid colors')
    return a.astype(np.int8, copy=False)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in CARDINAL:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _fit_mask(free, offsets):
    """Legal registration locations for a complete, unrotated footprint."""
    h, w = free.shape
    fit = np.ones_like(free)
    for dr, dc in offsets:
        shifted = np.zeros_like(free)
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        fit &= shifted
    return fit


def _inside(mask, p):
    return 0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1] and bool(mask[p])


def _trace(fit, start, direction, corner=True):
    """Ray in configuration space; side tests prevent diagonal corner cutting."""
    dr, dc = direction
    r, c = start
    path = [(r, c)]
    for _ in range(60):
        nxt = (r + dr, c + dc)
        if not _inside(fit, nxt):
            return path, False
        if corner and dr and dc and (not _inside(fit, (r + dr, c))
                                     or not _inside(fit, (r, c + dc))):
            return path, True
        r, c = nxt
        path.append(nxt)
    raise AssertionError('unbounded ray')


def _swept(path, offsets, direction):
    origins = set(path)
    dr, dc = direction
    if dr and dc:
        for r, c in path[:-1]:
            origins.update(((r + dr, c), (r, c + dc)))
    return {(r + a, c + b) for r, c in origins for a, b in offsets}


def _key_read(a, cells):
    rr, cc = zip(*cells)
    r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < 4 or len(cells) != h * w:
        return None
    sub = a[r0:r1 + 1, c0:c1 + 1]
    field = int(sub[0, 0])
    border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
    others = set(map(int, np.unique(sub))) - {field}
    if len(others) != 2 or np.any(border != field):
        return None
    readings = []
    for tip in sorted(others):
        trail = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(sub == tip)))
        tails = set(map(tuple, np.argwhere(sub == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                r, c = r - dr, c - dc
                while (r, c) in tails:
                    stroke.append((r, c))
                    r, c = r - dr, c - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == tails:
                readings.append((tip, trail, (dr, dc), tuple(lengths)))
    if len(readings) != 1:
        return None
    tip, trail, direction, lengths = readings[0]
    return dict(cells=cells, wall=field, tip=tip, trail=trail, direction=direction,
                lengths=lengths, shape=(h, w))


def _read(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(frequencies.argmax())
    _require(frequencies[bg] > a.size / 2, 'background must occupy more than half the input')
    components = _components(a != bg)
    keys = [k for cells in components if (k := _key_read(a, cells)) is not None]
    _require(len(keys) == 1, 'one geometrically readable key required')
    key = keys[0]
    key_mask = np.zeros(a.shape, bool)
    for p in key['cells']:
        key_mask[p] = True
    anchors, anchor_mask = {}, np.zeros(a.shape, bool)
    for cells in components:
        if len(cells) != 1:
            continue
        p = next(iter(cells))
        r, c = p
        if r in (0, h - 1) or c in (0, w - 1):
            _require((r in (0, h - 1)) != (c in (0, w - 1)), 'corner anchor')
            color = int(a[p])
            anchors.setdefault(color, []).append(p)
            anchor_mask[p] = True
    _require(anchors, 'missing registration anchors')
    systems = {}
    for color, points in anchors.items():
        _require(color not in (bg, key['wall'], key['trail'], key['tip']), 'role conflict')
        horizontal = len({r for r, _ in points}) == 1 and all(c in (0, w - 1) for _, c in points)
        vertical = len({c for _, c in points}) == 1 and all(r in (0, h - 1) for r, _ in points)
        _require(horizontal != vertical, 'inconsistent anchors')
        axis = 0 if horizontal else 1
        guide = points[0][axis]
        _require(key['direction'][axis] != 0, 'guide parallel to motion')
        systems[color] = dict(axis=axis, guide=guide, anchors=points)
    wall = (a == key['wall']) & ~key_mask
    _require(wall.any(), 'missing wall')
    free = np.zeros(a.shape, bool)
    regions = _components(~wall)
    for cells in regions:
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            for p in cells:
                free[p] = True
    cargos, used_sweep = [], set()
    movable_mask = (a != bg) & ~key_mask & ~wall & ~anchor_mask
    for cells in _components(movable_mask):
        marks = [(p, int(a[p])) for p in sorted(cells) if int(a[p]) in systems]
        _require(len(cells) >= 2 and len(marks) == 1, 'cargo needs exactly one registration cell')
        _require(all(free[p] for p in cells), 'cargo must be inside a closed wall region')
        _require(not any(int(a[p]) in (key['trail'], key['tip']) for p in cells), 'key roles in cargo')
        mark, color = marks[0]
        offsets = [(int(r - mark[0]), int(c - mark[1])) for r, c in sorted(cells)]
        system = systems[color]
        axis, guide = system['axis'], system['guide']
        direction = key['direction']
        guide_steps = int((guide - mark[axis]) * direction[axis])
        _require(guide_steps >= 0, 'guide must be ahead of, or through, its mark')
        fit = _fit_mask(free, offsets)
        wall_path, corner = _trace(fit, mark, direction)
        boundary_steps = len(wall_path) - 1
        distance = min(guide_steps, boundary_steps)
        path = wall_path[:distance + 1]
        sweep = _swept(path, offsets, direction)
        _require(used_sweep.isdisjoint(sweep), 'cargo swept footprints intersect')
        used_sweep.update(sweep)
        marker_path, _ = _trace(free, mark, direction)
        delta = (distance * direction[0], distance * direction[1])
        target = {(r + delta[0], c + delta[1]): int(a[r, c]) for r, c in cells}
        cargos.append(dict(cells=cells, mark=mark, color=color, offsets=offsets,
                           system=system, guide_steps=guide_steps, boundary_steps=boundary_steps,
                           distance=distance, path=path, target=target,
                           guide_wins=guide_steps < boundary_steps,
                           wall_wins=boundary_steps < guide_steps,
                           body_early=boundary_steps < len(marker_path) - 1,
                           eccentric=min(p[axis] for p in offsets) + max(p[axis] for p in offsets) != 0,
                           corner=corner and boundary_steps < guide_steps))
    _require(cargos and {x['color'] for x in cargos} == set(systems), 'unused anchor')
    return dict(grid=a, background=bg, key=key, systems=systems, cargos=cargos,
                wall=wall, free=free)


def _render(info, ignore_guide=False):
    result = info['grid'].copy()
    claimed = set()
    for cargo in info['cargos']:
        for p in cargo['cells']:
            result[p] = info['background']
    for cargo in info['cargos']:
        if ignore_guide:
            d = cargo['boundary_steps']
            dr, dc = info['key']['direction']
            target = {(r + d * dr, c + d * dc): int(info['grid'][r, c])
                      for r, c in cargo['cells']}
        else:
            target = cargo['target']
        for p, color in target.items():
            if not ignore_guide:
                _require(p not in claimed and result[p] == info['background'], 'target collision')
            result[p] = color
            claimed.add(p)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible roles and apply the first-event rigid-motion rule."""
    return _render(_read(input_grid))


def _make_key(rng, direction, field, trail, tip):
    length = int(rng.integers(2, 4))
    count = int(rng.choice([2, 2, 3]))
    points = []
    dr, dc = direction
    for i in range(count):
        stagger = int(rng.integers(2))
        base = (2 * i, 0) if dr == 0 else (0, 2 * i)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc, tip if j == length else trail))
    r0, c0 = min(r for r, _, _ in points), min(c for _, c, _ in points)
    r1, c1 = max(r for r, _, _ in points), max(c for _, c, _ in points)
    pad = [int(rng.choice([1, 1, 2])) for _ in range(4)]
    a = np.full((r1 - r0 + 1 + pad[0] + pad[1], c1 - c0 + 1 + pad[2] + pad[3]), field, np.int8)
    for r, c, color in points:
        a[r - r0 + pad[0], c - c0 + pad[2]] = color
    return a


def _shape(rng, busy):
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(0, 0)}
        desired = int(rng.integers(3, 7 if busy else 11))
        for _ in range(desired - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in CARDINAL} - cells)
            frontier = [p for p in frontier if max(abs(p[0]), abs(p[1])) <= 3]
            cells.add(frontier[int(rng.integers(len(frontier)))])
    elif kind == 1:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 2:
        h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 3:
        h, w = 3, int(rng.integers(3, 5))
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
        if busy or rng.random() < .5:
            cells.remove((0, 1))
    else:
        length = int(rng.integers(3, 6))
        cells = {(0, c) for c in range(length)} | {(1, int(rng.integers(length)))}
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    mark = sorted(cells)[int(rng.integers(len(cells)))]
    return sorted((r - mark[0], c - mark[1]) for r, c in cells)


def _room(rng, h, w, baffles, irregular):
    footprint = np.ones((h, w), bool)
    if irregular:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    free = _fit_mask(footprint, [(0, 0)] + list(CARDINAL))
    wall = footprint & ~free
    for _ in range(baffles):
        if rng.random() < .5:
            col = int(rng.integers(3, w - 3))
            end = int(rng.integers(3, h - 3))
            rs = slice(0, end) if rng.random() < .5 else slice(end, h)
            wall[rs, col] |= footprint[rs, col]
        else:
            row = int(rng.integers(3, h - 3))
            end = int(rng.integers(3, w - 3))
            cs = slice(0, end) if rng.random() < .5 else slice(end, w)
            wall[row, cs] |= footprint[row, cs]
    return wall, free & ~wall


def _anchor_choice(rng, a, bg, axis, guide):
    h, w = a.shape
    candidates = [(guide, 0), (guide, w - 1)] if axis == 0 else [(0, guide), (h - 1, guide)]
    valid = []
    for r, c in candidates:
        if a[r, c] == bg and all(not (0 <= r + dr < h and 0 <= c + dc < w)
                                  or a[r + dr, c + dc] == bg for dr, dc in CARDINAL):
            valid.append((r, c))
    if not valid:
        return None
    if len(valid) == 2 and rng.random() < .3:
        return valid
    return [valid[int(rng.integers(len(valid)))]]


def _place_cargo(rng, a, free, bg, direction, marker, bodies, role,
                 corner_required, busy, occupied_sweep, used_guides):
    source_free = _fit_mask(free, [(0, 0)] + list(CARDINAL))
    dr, dc = direction
    for _ in range(14):
        offsets = _shape(rng, busy)
        fit = _fit_mask(free, offsets)
        origins = np.argwhere(_fit_mask(source_free, offsets))
        if not len(origins):
            continue
        for raw in rng.permutation(len(origins)):
            start = tuple(map(int, origins[int(raw)]))
            source = {(start[0] + r, start[1] + c) for r, c in offsets}
            if any(a[p] != bg or any(0 <= p[0] + rr < a.shape[0] and 0 <= p[1] + cc < a.shape[1]
                                    and a[p[0] + rr, p[1] + cc] != bg for rr, cc in CARDINAL)
                   for p in source):
                continue
            wall_path, corner = _trace(fit, start, direction)
            k = len(wall_path) - 1
            if k < (2 if role == 'guide' else 1) or corner_required and not corner:
                continue
            if role == 'wall':
                marker_path, _ = _trace(free, start, direction)
                if k >= len(marker_path) - 1:
                    continue  # Explicit whole-body collision witness.
            axes = [axis for axis in (0, 1) if direction[axis]]
            rng.shuffle(axes)
            for axis in axes:
                if role == 'guide' and min(p[axis] for p in offsets) + max(p[axis] for p in offsets) == 0:
                    continue  # The registration, rather than the cargo center, must matter.
                sign = direction[axis]
                limit = (a.shape[axis] - 2 - start[axis]) if sign > 0 else start[axis] - 1
                steps = list(range(1, k)) if role == 'guide' else list(range(k + 1, min(limit, k + 5) + 1))
                if role == 'aligned':
                    steps = [0]
                rng.shuffle(steps)
                for t in steps:
                    guide = start[axis] + t * sign
                    if not 1 <= guide < a.shape[axis] - 1 or (axis, guide) in used_guides:
                        continue
                    anchors = _anchor_choice(rng, a, bg, axis, guide)
                    if anchors is None:
                        continue
                    path = wall_path[:min(t, k) + 1]
                    sweep = _swept(path, offsets, direction)
                    if not occupied_sweep.isdisjoint(sweep):
                        continue
                    body = int(rng.choice(bodies))
                    colors = {p: body for p in source}
                    if len(source) >= 5 and rng.random() < .35:
                        options = sorted(source - {start})
                        colors[options[int(rng.integers(len(options)))]] = next(x for x in bodies if x != body)
                    colors[start] = marker
                    for p, color in colors.items():
                        a[p] = color
                    for p in anchors:
                        a[p] = marker
                    occupied_sweep.update(sweep)
                    used_guides.add((axis, guide))
                    end = path[-1]
                    target = {(end[0] + r - start[0], end[1] + c - start[1]): color
                              for (r, c), color in colors.items()}
                    return dict(source=source, anchors=anchors, target=target, marker=marker)
    return None


def _build_pair(rng, direction, count, palette, corner_required=False, test=False, markers=None):
    bg, field, trail, tip = palette[:4]
    bodies = palette[8:10]
    if markers is None:
        markers = palette[4:8]
    for attempt in range(80):
        key = _make_key(rng, direction, field, trail, tip)
        kh, kw = key.shape
        h = int(rng.integers(max(23, kh + 16), 31))
        w = int(rng.integers(23 if count >= 3 else 18, 31))
        a = np.full((h, w), bg, np.int8)
        kr, kc = 1, int(rng.integers(1, w - kw))
        a[kr:kr + kh, kc:kc + kw] = key
        top = kh + 3
        rh = int(rng.integers(11, h - top))
        rw = int(rng.integers(max(12, w - 9) if count >= 3 else 11, w - 2))
        left = int(rng.integers(1, w - rw))
        baffles = int(rng.integers(1 if corner_required else 0, 3 if test else 2))
        wall, local_free = _room(rng, rh, rw, baffles, test or rng.random() < .7)
        a[top:top + rh, left:left + rw][wall] = field
        free = np.zeros(a.shape, bool)
        free[top:top + rh, left:left + rw] = local_free
        placements, used_sweep, guides = [], set(), set()
        roles = ['wall', 'guide'] + [str(rng.choice(['guide', 'wall', 'aligned'], p=[.45, .4, .15]))
                                   for _ in range(count - 2)]
        success = True
        for i, role in enumerate(roles):
            placement = _place_cargo(rng, a, free, bg, direction, markers[i], bodies,
                                     role, corner_required and i == 0, count >= 3, used_sweep, guides)
            if placement is None:
                success = False
                break
            placements.append(placement)
        if not success or np.count_nonzero(a == bg) <= a.size / 2:
            continue
        # Construction renders its own placements; solve() reconstructs from pixels.
        b = a.copy()
        for cargo in placements:
            for p in cargo['source']:
                b[p] = bg
        for cargo in placements:
            for p, color in cargo['target'].items():
                b[p] = color
        turns = int(rng.integers(4))
        return {k: np.rot90(v, turns).copy() for k, v in (('input', a), ('output', b))}
    raise RuntimeError('bounded scene construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable_palette = bool(rng.random() < .3)
    puzzle, seen, seen_directions = {'train': [], 'test': []}, set(), set()
    for index in range(nt + nq):
        test = index >= nt
        diagonal = index == 1 or (index > 1 and rng.random() < .6)
        pool = DIAGONAL if diagonal else CARDINAL
        novel = set(DIRECTIONS) - seen_directions
        if test and novel and not novel.intersection(pool):
            pool = CARDINAL if diagonal else DIAGONAL
        direction = pool[int(rng.integers(len(pool)))]
        count = (4 if not novel else int(rng.choice([3, 3, 4]))) if test else int(rng.choice([2, 2, 2, 3]))
        colors = palette if stable_palette else tuple(map(int, rng.permutation(10)))
        markers = list(colors[4:8])
        if index == 1:
            markers[0], markers[1] = markers[1], markers[0]
        elif index > 1:
            rng.shuffle(markers)
        for retry in range(8):
            pair = _build_pair(rng, direction, count, colors, index == 1, test, markers)
            actual_direction = _read(pair['input'])['key']['direction']
            # Rotation is visible in the key. Queries use an unseen direction;
            # first two demonstrations show the cardinal and diagonal mechanisms.
            if test and actual_direction in seen_directions:
                candidates = []
                rotated = actual_direction
                for turns in range(1, 4):
                    rotated = (-rotated[1], rotated[0])
                    if rotated not in seen_directions:
                        candidates.append((turns, rotated))
                if candidates:
                    turns, actual_direction = candidates[int(rng.integers(len(candidates)))]
                    pair = {k: np.rot90(v, turns).copy() for k, v in pair.items()}
            signature = _signature(pair['input'])
            if signature not in seen:
                break
        else:
            raise RuntimeError('bounded uniqueness construction exhausted')
        seen.add(signature)
        if not test:
            seen_directions.add(actual_direction)
        puzzle['test' if test else 'train'].append(pair)
    # Shuffle demonstrations so position in the list never determines behavior.
    order = rng.permutation(nt)
    puzzle['train'] = [puzzle['train'][int(i)] for i in order]
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split type')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'dimensions changed')
            signature = _signature(a)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            info = _read(a)
            expected = _render(info)
            _require(np.array_equal(expected, b), 'first-event motion mismatch')
            _require(not np.array_equal(a, b), 'no motion')
            _require(np.array_equal(np.bincount(a.ravel(), minlength=10),
                                   np.bincount(b.ravel(), minlength=10)), 'color conservation')
            fixed = np.ones(a.shape, bool)
            for cargo in info['cargos']:
                for p in cargo['cells']:
                    fixed[p] = False
                for p in cargo['target']:
                    fixed[p] = False
            _require(np.array_equal(a[fixed], b[fixed]), 'fixed cells changed')
            analyses[split].append(info)
    training = [c for info in analyses['train'] for c in info['cargos']]
    _require(any(c['guide_wins'] and c['distance'] > 0 and c['eccentric'] for c in training),
             'no eccentric registration brake witness')
    _require(any(c['wall_wins'] and c['body_early'] and c['distance'] > 0 for c in training),
             'no full-body wall witness')
    train_dirs = {i['key']['direction'] for i in analyses['train']}
    _require(any(d in CARDINAL for d in train_dirs) and any(d in DIAGONAL for d in train_dirs),
             'missing cardinal/diagonal demonstrations')
    _require(any(c['corner'] for c in training), 'no diagonal side-contact witness')
    _require(any(not np.array_equal(_render(i, ignore_guide=True), _render(i)) for i in analyses['train']),
             'omitting A did not change training')
    # Omitting B's wall limit would advance a strict wall winner farther to its
    # visible guide; the inequality is checked from reconstructed geometry.
    _require(any(c['guide_steps'] > c['boundary_steps'] for c in training), 'omitting B has no effect')


def self_test():
    """Stream 42 fixed seeds; retain only small sets of descriptors and hashes."""
    fingerprints, shapes, structures, directions, sizes, key_shapes = set(), set(), set(), set(), set(), set()
    guide_axes, anchor_sides, counts, body_shapes = set(), set(), set(), set()
    wins = {'guide': 0, 'wall': 0, 'corner': 0, 'body_early': 0, 'stationary': 0}
    pairs = 0
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                digest.update(a.tobytes())
                digest.update(pair['output'].tobytes())
                info = _read(a)
                pairs += 1
                shapes.add(a.shape)
                directions.add(info['key']['direction'])
                key_shapes.add(info['key']['shape'])
                counts.add(len(info['cargos']))
                mapping = {}
                normalized = bytes(mapping.setdefault(int(v), len(mapping)) for v in a.ravel())
                structures.add(hashlib.sha256(str(a.shape).encode() + normalized).digest())
                for cargo in info['cargos']:
                    sizes.add(len(cargo['cells']))
                    body_shapes.add(tuple(cargo['offsets']))
                    guide_axes.add(cargo['system']['axis'])
                    for r, c in cargo['system']['anchors']:
                        anchor_sides.add('top' if r == 0 else 'bottom' if r == a.shape[0] - 1
                                         else 'left' if c == 0 else 'right')
                    wins['guide'] += cargo['guide_wins']
                    wins['wall'] += cargo['wall_wins']
                    wins['corner'] += cargo['corner']
                    wins['body_early'] += cargo['body_early']
                    wins['stationary'] += cargo['distance'] == 0
        fingerprints.add(digest.digest())
    _require(len(fingerprints) == 42 and len(structures) >= 200, 'seed/structure variation')
    _require(len(shapes) >= 30 and len(body_shapes) >= 80 and len(key_shapes) >= 10, 'geometric variation')
    _require(directions == set(DIRECTIONS) and guide_axes == {0, 1}, 'direction/guide coverage')
    _require(anchor_sides == {'top', 'bottom', 'left', 'right'} and counts == {2, 3, 4}, 'count/edge coverage')
    _require(min(sizes) <= 4 and max(sizes) >= 10 and wins['corner'] >= 42, 'cargo/contact variation')
    # Exact rotations/reflections and bijective recoloring preserve all rules.
    puzzle = generate_puzzle(0, 2, 1)
    rng = np.random.default_rng(1808)
    for turns in (1, 2, 3):
        colors = rng.permutation(10)
        transformed = {s: [{k: colors[np.fliplr(np.rot90(v, turns))].astype(np.int8)
                             for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
        validate_generated_puzzle(transformed)
    again = generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                 for k in ('input', 'output')), 'determinism')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                color_normalized_inputs=len(structures), cargo_shapes=len(body_shapes),
                key_shapes=len(key_shapes), cargo_sizes=sorted(sizes), cargo_counts=sorted(counts),
                directions=sorted(directions), witnesses={k: int(v) for k, v in wins.items()})
