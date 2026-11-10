"""Five-step arrows activate connected erasures in a two-fill rectangle scene.

All answer conventions are fixed or visible in the input. No parent module,
training label, random state, or generation metadata is used by solve().
"""
from itertools import permutations

import numpy as np


DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1))
DISTANCE = 5
BACKGROUND = 0
COLLISION = 9


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    offsets = DIRECTIONS if diagonal else ((-1, 0), (0, 1), (1, 0), (0, -1))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            neighbors = {(r + dr, c + dc) for dr, dc in offsets} & remaining
            remaining -= neighbors
            component |= neighbors
            todo.extend(sorted(neighbors))
        result.append(component)
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    cells = tuple(sorted((r + a, c + b) for a, b in offsets))
    end = (r + DISTANCE * dr, c + DISTANCE * dc)
    return (tuple(tip), tuple(direction), cells, end)


def _read_arrow(component):
    matches = [_arrow(p, d) for p in sorted(component) for d in DIRECTIONS
               if set(_arrow(p, d)[2]) == component]
    assert len(matches) == 1, 'Arrow must have a unique tip and direction'
    return matches[0]


def _near(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _compatible(obj, objects):
    occupied = {p for o in objects for p in o[2]}
    return not (_near(obj[2]) & occupied)


def _paint(shape, boxes):
    clean = np.zeros(shape, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        clean[r0:r1, c0:c1] = color
    return clean


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _supported(clean, boxes):
    if not 2 <= len(boxes) <= 4:
        return False
    edges = [(i, j) for i, a in enumerate(boxes) for j, b in enumerate(boxes[:i])
             if _overlap(a, b)]
    degree = [sum(i in e for e in edges) for i in range(len(boxes))]
    if len(edges) != len(boxes) - 1 or sorted(degree) != [1, 1] + [2] * (len(boxes) - 2):
        return False
    if any(boxes[i][4] == boxes[j][4] or _overlap(boxes[i], boxes[j]) < 6
           for i, j in edges):
        return False
    for r0, c0, r1, c1, color in boxes:
        if min(r1-r0, c1-c0) < 6:
            return False
        visible = clean[r0:r1, c0:c1] == color
        if visible.sum() < max(25, visible.size * .40):
            return False
        if min(visible[0].sum(), visible[-1].sum(),
               visible[:, 0].sum(), visible[:, -1].sum()) < 3:
            return False
        if len(_components(visible, True)) != 1:
            return False
    # Recurring fills must stay in separate readable components.
    for color in set(b[4] for b in boxes):
        comps = _components(clean == color, True)
        expected = [b for b in boxes if b[4] == color]
        if len(comps) != len(expected):
            return False
        if any(_near(a) & b for i, a in enumerate(comps) for b in comps[:i]):
            return False
    return True


def _scene_boxes(scene, fills):
    """Three surviving cells on each outer side defeat isolated speckles.

    In a supported input, each large same-fill component belongs to one box.
    Rows/columns with fewer than three cells cannot extend its estimated bounds.
    """
    boxes = []
    for color in fills:
        for comp in _components(scene == color, True):
            if len(comp) < 20:
                continue
            mask = np.zeros(scene.shape, dtype=bool)
            for p in comp:
                mask[p] = True
            rows = np.flatnonzero(mask.sum(axis=1) >= 3)
            cols = np.flatnonzero(mask.sum(axis=0) >= 3)
            assert len(rows) >= 6 and len(cols) >= 6, 'Insufficient straight-side evidence'
            boxes.append((int(rows[0]), int(cols[0]), int(rows[-1])+1,
                          int(cols[-1])+1, int(color)))
    return tuple(sorted(boxes))


def _recover_scene(scene, fills):
    boxes = _scene_boxes(scene, fills)
    assert 2 <= len(boxes) <= 4
    candidates = {}
    for ordered in permutations(boxes):
        clean = _paint(scene.shape, ordered)
        if _supported(clean, ordered):
            key = clean.tobytes()
            cost = int(np.count_nonzero(clean != scene))
            candidates[key] = (cost, clean, ordered)
    assert candidates, 'No supported rectangle scene'
    ranked = sorted(candidates.values(), key=lambda v: (v[0], v[1].tobytes()))
    cost, clean, ordered = ranked[0]
    assert all(v[0] > cost + 1 for v in ranked[1:]), 'Ambiguous covering order'
    # Sparse damage must not make a one-cell movement of any visible boundary
    # equally plausible. Equivalent hidden geometry does not affect the rule.
    for i, box in enumerate(ordered):
        for side in range(4):
            for delta in (-1, 1):
                changed = list(box)
                changed[side] += delta
                r0, c0, r1, c1, _ = changed
                if not (0 <= r0 < r1 <= scene.shape[0] and 0 <= c0 < c1 <= scene.shape[1]):
                    continue
                if min(r1-r0, c1-c0) < 6:
                    continue
                trial = list(ordered)
                trial[i] = tuple(changed)
                alternative = _paint(scene.shape, trial)
                if not np.array_equal(alternative, clean):
                    assert np.count_nonzero(alternative != scene) > cost + 1, 'Uncertain damaged boundary'
    assert cost <= .18 * np.count_nonzero(clean), 'Damage obscures the scene'
    for r0, c0, r1, c1, color in boxes:
        visible = clean[r0:r1, c0:c1] == color
        intact = scene[r0:r1, c0:c1] == color
        assert np.count_nonzero(visible & intact) >= .75 * visible.sum()
        # The input, rather than the output, must expose every true bound.
        sides = (visible[0] & intact[0], visible[-1] & intact[-1],
                 visible[:, 0] & intact[:, 0], visible[:, -1] & intact[:, -1])
        assert min(s.sum() for s in sides) >= 3
    return clean, ordered


def _parse_input(grid):
    colors = set(map(int, np.unique(grid)))
    assert 0 in colors and 9 not in colors and len(colors) == 4
    arrow_candidates = []
    for color in sorted(colors - {0}):
        comps = _components(grid == color, True)
        if comps and all(len(c) == 3 for c in comps):
            try:
                objects = [_read_arrow(c) for c in comps]
                arrow_candidates.append((color, objects))
            except AssertionError:
                pass
    assert len(arrow_candidates) == 1, 'Need one separated arrow palette role'
    arrow_color, arrows = arrow_candidates[0]
    fills = tuple(sorted(colors - {0, arrow_color}))
    scene = grid.copy()
    scene[scene == arrow_color] = 0
    clean, boxes = _recover_scene(scene, fills)
    blocked = _near(map(tuple, np.argwhere(clean != 0).tolist()))
    for obj in arrows:
        assert all(p not in blocked for p in obj[2]), 'Arrows need an exterior gap'
        assert 0 <= obj[3][0] < grid.shape[0] and 0 <= obj[3][1] < grid.shape[1]
    holes = (grid == 0) & (clean != 0)
    groups = _components(holes)
    assert groups, 'Need at least one crack'
    assert all(len({int(clean[p]) for p in g}) == 1 for g in groups), 'Cracks have one recovered fill'
    # This family retains B's compatible simultaneous rings, with no invented
    # order for contradictory demands. Inactive groups may later be activated.
    demands = {}
    for g in groups:
        color = next(c for c in fills if c != int(clean[min(g)]))
        for p in g:
            for q in _ring(grid.shape, p):
                assert q not in demands or demands[q] == color, 'Incompatible potential rings'
                demands[q] = color
    return clean, boxes, fills, arrow_color, arrows, groups


def _ring(shape, point):
    r, c = point
    return [(a, b) for a in range(max(0, r-1), min(shape[0], r+2))
            for b in range(max(0, c-1), min(shape[1], c+2)) if (a, b) != point]


def _events(arrows, groups):
    owner = {p: i for i, g in enumerate(groups) for p in g}
    arrivals = [[] for _ in groups]
    for obj in arrows:
        if obj[3] in owner:
            arrivals[owner[obj[3]]].append(obj[3])
    return arrivals


def _render(clean, fills, arrow_color, arrows, groups):
    arrivals = _events(arrows, groups)
    output = clean.copy()
    centers = {}
    for group, hits in zip(groups, arrivals):
        if not hits:
            continue
        opposite = next(c for c in fills if c != int(clean[min(group)]))
        for p in sorted(group):
            for q in _ring(clean.shape, p):
                output[q] = opposite
            centers[p] = arrow_color if len(hits) == 1 else COLLISION
    for p, color in centers.items():
        output[p] = color
    for obj in arrows:
        for p in obj[2]:
            output[p] = arrow_color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the scene, then broadcast endpoint arrivals within each crack."""
    grid = _grid(input_grid)
    clean, _, fills, arrow_color, arrows, groups = _parse_input(grid)
    return _render(clean, fills, arrow_color, arrows, groups)


def _sample_scene(rng, shape, count):
    h, w = shape
    boxes = []
    for _ in range(300):
        rh = int(rng.integers(6, min(14, h-3) + 1))
        rw = int(rng.integers(6, min(14, w-3) + 1))
        r = int(rng.integers(0, h-rh+1))
        c = int(rng.integers(0, w-rw+1))
        box = (r, c, r+rh, c+rw, 1 + len(boxes) % 2)
        if boxes:
            overlap = _overlap(box, boxes[-1])
            if not 6 <= overlap <= .32 * min(rh*rw, (boxes[-1][2]-boxes[-1][0])*(boxes[-1][3]-boxes[-1][1])):
                continue
            if any(_overlap(box, b) for b in boxes[:-1]):
                continue
        boxes.append(box)
        if len(boxes) == count:
            ordered = [boxes[int(i)] for i in rng.permutation(count)]
            clean = _paint(shape, ordered)
            if _supported(clean, ordered) and np.count_nonzero(clean) < .58 * clean.size:
                try:
                    recovered, _ = _recover_scene(clean, (1, 2))
                    if np.array_equal(recovered, clean):
                        return clean
                except AssertionError:
                    pass
            return None
    return None


def _arrow_pool(clean, foreground_only=True):
    h, w = clean.shape
    blocked = _near(map(tuple, np.argwhere(clean != 0).tolist()))
    pool = []
    for r in range(h):
        for c in range(w):
            if (r, c) in blocked:
                continue
            for direction in DIRECTIONS:
                obj = _arrow((r, c), direction)
                er, ec = obj[3]
                if not (0 <= er < h and 0 <= ec < w):
                    continue
                if foreground_only and clean[er, ec] == 0:
                    continue
                if all(0 <= a < h and 0 <= b < w and (a, b) not in blocked for a, b in obj[2]):
                    pool.append(obj)
    return pool


def _grow_crack(rng, clean, start, size):
    group = {start}
    color = int(clean[start])
    for _ in range(size-1):
        options = sorted({(r+dr, c+dc) for r, c in group
                          for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))
                          if 0 <= r+dr < clean.shape[0] and 0 <= c+dc < clean.shape[1]
                          and clean[r+dr, c+dc] == color} - group)
        if not options:
            return None
        group.add(options[int(rng.integers(len(options)))])
    return group


def _groups_compatible(clean, first, others):
    for group in others:
        same = clean[min(first)] == clean[min(group)]
        for r, c in first:
            for a, b in group:
                if same and abs(r-a) + abs(c-b) <= 1:
                    return False
                if not same and max(abs(r-a), abs(c-b)) <= 2:
                    return False
    return True


def _add_active(rng, clean, pool, groups, arrows, size, shots, fill=None):
    indices = rng.permutation(len(pool))
    for index in indices[:160]:
        first = pool[int(index)]
        if fill is not None and clean[first[3]] != fill:
            continue
        if not _compatible(first, arrows):
            continue
        group = _grow_crack(rng, clean, first[3], size)
        if group is None or not _groups_compatible(clean, group, groups):
            continue
        chosen = [first]
        if shots > 1:
            alternatives = [o for o in pool if o[3] in group and o[3] != first[3]]
            rng.shuffle(alternatives)
            for obj in alternatives:
                if _compatible(obj, arrows + chosen):
                    chosen.append(obj)
                    if len(chosen) == shots:
                        break
            if len(chosen) != shots:
                continue
        # No previously placed shot may secretly change the lesson's count.
        if any(o[3] in group for o in arrows):
            continue
        groups.append(group)
        arrows.extend(chosen)
        return True
    return False


def _add_inactive(rng, clean, groups, arrows, diagonal=False):
    holes = set().union(*groups)
    endpoints = {o[3] for o in arrows}
    candidates = [tuple(p) for p in np.argwhere(clean != 0).tolist()]
    if diagonal:
        candidates = sorted({(r+dr, c+dc) for r, c in groups[0]
                             for dr, dc in ((-1,-1), (-1,1), (1,-1), (1,1))
                             if 0 <= r+dr < clean.shape[0] and 0 <= c+dc < clean.shape[1]
                             and clean[r+dr, c+dc] == clean[r, c]})
    rng.shuffle(candidates)
    for p in candidates:
        if p not in holes and p not in endpoints and _groups_compatible(clean, {p}, groups):
            groups.append({p})
            return True
    return False


def _construct_pair(rng, kind, palette):
    """Construction uses clean roles; acceptance checks the final input pixels."""
    test = kind == 'test'
    for attempt in range(180):
        low, high = (24, 31) if test else (22, 29)
        shape = tuple(map(int, rng.integers(low, high, size=2)))
        count = int(rng.choice((2, 3, 4), p=(.60, .33, .07)))
        clean = _sample_scene(rng, shape, count)
        if clean is None:
            continue
        pool = _arrow_pool(clean)
        if len(pool) < 8:
            continue
        groups, arrows = [], []
        size = int(rng.integers(5, 8)) if test else int(rng.integers(3, 5))
        shots = 2 if kind == 'collision' else (int(rng.integers(1, 4)) if test else 1)
        if not _add_active(rng, clean, pool, groups, arrows, size, shots):
            continue
        # Every demonstration exposes both complementary fill directions.
        other_fill = 3 - int(clean[min(groups[0])])
        if not _add_active(rng, clean, pool, groups, arrows,
                           int(rng.integers(1, 4)), 1, other_fill):
            continue
        # Independent optional event sites broaden object and arrow counts;
        # teaching obligations do not force every scene into the same arity.
        for _ in range(int(rng.choice((0, 1, 2), p=(.45, .40, .15)))):
            _add_active(rng, clean, pool, groups, arrows,
                        int(rng.integers(1, 5)), int(rng.integers(1, 3)))
        if not _add_inactive(rng, clean, groups, arrows, diagonal=(kind == 'single')):
            continue
        if rng.random() < .55:
            _add_inactive(rng, clean, groups, arrows)
        # An arrow aimed at wrong-fill noise still has no crack to activate.
        holes = set().union(*groups)
        active = set().union(*(g for g, h in zip(groups, _events(arrows, groups)) if h))
        decoys = [o for o in pool if o[3] not in _near(active) and o[3] not in holes
                  and _compatible(o, arrows)]
        if not decoys:
            continue
        decoy = decoys[int(rng.integers(len(decoys)))]
        arrows.append(decoy)
        # Sometimes an unhit hole is directly on this ignored arrow's path.
        for k in (3, 4):
            p = tuple(decoy[0][a] + k*decoy[1][a] for a in (0, 1))
            if clean[p] != 0 and p not in holes and _groups_compatible(clean, {p}, groups):
                groups.append({p})
                holes.add(p)
                break
        if rng.random() < .35:
            misses = [o for o in _arrow_pool(clean, False)
                      if clean[o[3]] == 0 and _compatible(o, arrows)]
            rng.shuffle(misses)
            for obj in misses[:int(rng.integers(1, 4))]:
                if _compatible(obj, arrows):
                    arrows.append(obj)
        corrupted = clean.copy()
        for p in holes:
            corrupted[p] = 0
        corrupted[decoy[3]] = 3 - int(clean[decoy[3]])
        # Sparse extra ordinary damage, including seam errors and exterior dust.
        foreground = [tuple(p) for p in np.argwhere(clean != 0).tolist()
                      if tuple(p) not in holes and tuple(p) != decoy[3]]
        rng.shuffle(foreground)
        for p in foreground[:int(rng.integers(0, 5))]:
            corrupted[p] = 3 - int(clean[p])
        arrow_near = _near(p for o in arrows for p in o[2])
        exterior = [tuple(p) for p in np.argwhere(clean == 0).tolist() if tuple(p) not in arrow_near]
        rng.shuffle(exterior)
        for p in exterior[:int(rng.integers(0, 4))]:
            corrupted[p] = int(rng.integers(1, 3))
        mapping = np.array([0, palette[0], palette[1]], dtype=np.int8)
        actual_clean, inp = mapping[clean], mapping[corrupted]
        actual_fills, arrow_color = tuple(palette[:2]), palette[2]
        for obj in arrows:
            for p in obj[2]:
                inp[p] = arrow_color
        expected = _render(actual_clean, actual_fills, arrow_color, arrows, groups)
        try:
            parsed = _parse_input(inp)
            if not np.array_equal(parsed[0], actual_clean):
                continue
            out = solve(inp)
            if not np.array_equal(out, expected):
                continue
            feature = _features(inp, parsed)
            if not feature['both_fills'] or not feature['inactive'] or not feature['wrong_target']:
                continue
            if kind == 'single' and not (feature['single_broadcast'] and feature['diagonal_split']):
                continue
            if kind == 'collision' and not feature['distinct_collision']:
                continue
            if test and feature['max_active_size'] < 5:
                continue
            return {'input': inp, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('Could not construct a sufficiently clear crack-activation example')


def _check_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_count('num_train', num_train, 2, 10)
    _check_count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False)))
    recurring = bool(rng.integers(2))
    train_kinds = ['single', 'collision'] + [str(rng.choice(('single', 'collision')))
                                            for _ in range(num_train-2)]
    rng.shuffle(train_kinds)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', train_kinds), ('test', ['test'] * num_test)):
        for kind in kinds:
            palette = base_palette if recurring else tuple(map(int, rng.choice(np.arange(1, 9), 3, replace=False)))
            for _ in range(8):
                pair = _construct_pair(rng, kind, palette)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid, parsed=None):
    clean, boxes, fills, color, arrows, groups = parsed or _parse_input(grid)
    arrivals = _events(arrows, groups)
    active = [g for g, h in zip(groups, arrivals) if h]
    inactive = [g for g, h in zip(groups, arrivals) if not h]
    scene = grid.copy()
    scene[scene == color] = 0
    return {
        'shape': tuple(grid.shape), 'rectangles': len(boxes),
        'box_sizes': tuple(sorted((b[2]-b[0], b[3]-b[1]) for b in boxes)),
        'palette': (fills, color), 'directions': tuple(sorted({o[1] for o in arrows})),
        'arrows': len(arrows), 'holes': sum(map(len, groups)),
        'background_misses': sum(clean[o[3]] == 0 for o in arrows),
        'crack_sizes': tuple(sorted(map(len, groups))),
        'max_active_size': max(map(len, active), default=0),
        'both_fills': {int(clean[min(g)]) for g in active} == set(fills),
        'inactive': len(inactive),
        'single_broadcast': any(len(g) > 1 and len(h) == 1 for g, h in zip(groups, arrivals)),
        'distinct_collision': any(len(set(h)) > 1 for h in arrivals),
        'max_arrivals': max(map(len, arrivals), default=0),
        'wrong_target': any(scene[o[3]] not in (0, clean[o[3]]) and clean[o[3]] != 0 for o in arrows),
        'diagonal_split': any(abs(r-a) == abs(c-b) == 1
                              for g in active for z in inactive for r, c in g for a, b in z),
        'bent': any(len({p[0] for p in g}) > 1 and len({p[1] for p in g}) > 1 for g in active),
        'branch': any(sum((r+dr, c+dc) in g for dr, dc in ((-1,0),(0,1),(1,0),(0,-1))) >= 3
                      for g in active for r, c in g),
        'path_decoy': any(any(tuple(o[0][a]+k*o[1][a] for a in (0,1)) in g for k in range(1,5))
                          and o[3] not in g for g in inactive for o in arrows),
        'clipped': any(len(_ring(grid.shape, p)) < 8 for g in active for p in g),
        'wrong_fill': int(np.count_nonzero((scene != clean) & (scene != 0) & (clean != 0))),
        'speckles': int(np.count_nonzero((scene != 0) & (clean == 0))),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, train_analyses = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, 'Repeated input'
            seen.add(key)
            parsed = _parse_input(inp)
            clean, _, fills, color, arrows, groups = parsed
            hits = _events(arrows, groups)
            expected = _render(clean, fills, color, arrows, groups)
            assert np.array_equal(expected, out), 'Output disagrees with input-derived events'
            # Object-wide center assignment, unrelated to coordinate coincidence.
            for g, h in zip(groups, hits):
                if h:
                    assert all(out[p] == (color if len(h) == 1 else 9) for p in g)
                else:
                    assert all(out[p] in (0, *fills) for p in g)
            assert all(out[p] == inp[p] for obj in arrows for p in obj[2])
            feature = _features(inp, parsed)
            assert feature['both_fills'] and feature['inactive'] and feature['wrong_target']
            if split == 'train':
                train_features.append(feature)
                train_analyses.append((out, parsed))
    assert any(f['single_broadcast'] for f in train_features), 'Teach single-shot broadcasting'
    assert any(f['distinct_collision'] for f in train_features), 'Teach crack-wide collisions'
    assert any(f['diagonal_split'] for f in train_features), 'Teach four-connectivity'
    # Within the inherited tip-projection hypothesis, the demonstrations expose
    # the exact distance; this is an authored ambiguity check, not an oracle.
    distances = []
    for distance in range(1, 30):
        fits = True
        for out, parsed in train_analyses:
            clean, _, fills, color, arrows, groups = parsed
            moved = [(o[0], o[1], o[2], tuple(o[0][a] + distance*o[1][a] for a in (0,1)))
                     for o in arrows]
            if any(not (0 <= o[3][0] < out.shape[0] and 0 <= o[3][1] < out.shape[1]) for o in moved):
                fits = False
                break
            if not np.array_equal(_render(clean, fills, color, moved, groups), out):
                fits = False
                break
        if fits:
            distances.append(distance)
    assert distances == [DISTANCE], 'Training must identify the projection distance'


def _puzzle_key(puzzle):
    return tuple((s, tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                           np.asarray(p['output']).tobytes()) for p in puzzle[s]))
                 for s in ('train', 'test'))


def self_test():
    """Bounded semantic probes plus measured structural coverage on 42 seeds."""
    # Hand-constructed expected pixels: two (then three) arrows share an
    # endpoint, activating an L crack. A diagonal unhit hole gets only the ring,
    # while an unrelated wrong-fill cell is repaired. No rendering helper is
    # used to construct this fixture's expected output.
    clean = np.zeros((24, 25), dtype=np.int8)
    clean[7:17, 7:17] = 1
    clean[12:22, 13:23] = 2
    inp = clean.copy()
    centers = ((8, 8), (8, 9), (9, 9))
    for p in (*centers, (9, 7)):
        inp[p] = 0
    inp[15, 15] = 1
    for cells in (((8, 3), (7, 2), (9, 2)), ((3, 8), (2, 7), (2, 9))):
        for p in cells:
            inp[p] = 3
    expected = clean.copy()
    expected[7:11, 7:11] = 2
    expected[10, 7] = 1
    for p in centers:
        expected[p] = 9
    expected[inp == 3] = 3
    assert np.array_equal(solve(inp), expected)
    for p in ((3, 3), (2, 3), (3, 2)):
        inp[p] = 3
        expected[p] = 3
    assert np.array_equal(solve(inp), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                parsed = _parse_input(inp)
                clean, _, fills, color, arrows, groups = parsed
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(_render(clean, fills, color, arrows[::-1], groups[::-1]), out)
                # Removing A's gate activates every hole and changes an inactive
                # center; removing B's complementary substitution changes rings.
                hits = _events(arrows, groups)
                inactive = [g for g, h in zip(groups, hits) if not h]
                assert any(out[p] != color for g in inactive for p in g)
                active = [g for g, h in zip(groups, hits) if h]
                centers = set().union(*active)
                assert any(out[q] in fills and out[q] != color for g in active for p in g
                           for q in _ring(inp.shape, p) if q not in centers)
                # Rotation and reflection must not change semantic conventions.
                for transform in (lambda a: np.rot90(a), lambda a: np.fliplr(a)):
                    assert np.array_equal(solve(transform(inp)), transform(out))
                records.append(dict(_features(inp, parsed), split=split))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert len({f['shape'] for f in records}) >= 35
    assert len({f['box_sizes'] for f in records}) >= 100
    assert len({f['crack_sizes'] for f in records}) >= 20
    assert len({f['palette'] for f in records}) >= 20
    assert {d for f in records for d in f['directions']} == set(DIRECTIONS)
    assert {f['rectangles'] for f in records} >= {2, 3}
    assert len({f['arrows'] for f in records}) >= 5
    assert any(f['background_misses'] for f in records)
    for key in ('bent', 'branch', 'path_decoy', 'clipped', 'diagonal_split', 'distinct_collision'):
        assert any(f[key] for f in records), key
    assert any(f['max_arrivals'] >= 3 for f in records)
    assert all(f['max_active_size'] >= 5 for f in records if f['split'] == 'test')
    assert all(f['max_active_size'] <= 4 for f in records if f['split'] == 'train')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(101+a*10+b, a, b))
    for a, b in ((1,1), (11,1), (4,0), (4,5), (2,2), (3,4), (True,1), (2.0,1), (None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mutated label accepted')
