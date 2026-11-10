"""Simultaneous component-local transfers and geometry-controlled rotation.

Construction preserves region IDs; the decoder independently reconstructs them
from visible annotations. Its prediction API receives no test label or seed.
"""

import numpy as np

D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def add(point, direction, amount=1):
    return point[0] + amount * direction[0], point[1] + amount * direction[1]


def inside(point, grid):
    return 0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]


def components(grid):
    labels = np.full(grid.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(grid.shape):
        if labels[start] >= 0:
            continue
        label, color = len(colors), int(grid[start])
        colors.append(color)
        stack, points = [start], set()
        labels[start] = label
        while stack:
            point = stack.pop()
            points.add(point)
            for direction in D:
                neighbor = add(point, direction)
                if inside(neighbor, grid) and labels[neighbor] < 0 and grid[neighbor] == color:
                    labels[neighbor] = label
                    stack.append(neighbor)
        cells.append(points)
    return labels, colors, cells


def _decode(grid):
    """Read geometry and original components without a seed or output label.

    The two cells flanking the tip reveal its host even in a two-cell strip.
    No backward pixel or deep target support is required. Sparse T ink must be
    completely explained by disjoint annotations; an L may use another ink.
    """
    raw_ids, _, raw_cells = components(grid)
    markers = []
    for cells in raw_cells:
        if len(cells) != 5:
            continue
        row, col = min(p[0] for p in cells), min(p[1] for p in cells)
        for turn in (-1, 1):
            if cells != _corner_cells(row, col, turn):
                continue
            patch = {(row+r, col+c) for r in range(3) for c in range(3)}
            if not all(inside(p, grid) for p in patch):
                continue
            fill = {int(grid[p]) for p in patch-cells}
            if len(fill) == 1:
                markers.append(dict(cells=cells, patch=patch, host=fill.pop(),
                                    turn=turn, ink=int(grid[row,col]), origin=(row,col)))
    _need(len(markers) <= 1, 'ambiguous corner markers')
    answers = []
    for ink_value in np.unique(grid):
        ink = int(ink_value)
        ink_cells = set(map(tuple, np.argwhere(grid == ink)))
        if len(ink_cells) < 3:
            continue
        marker_ids = {int(raw_ids[next(iter(m['cells']))]) for m in markers if m['ink'] == ink}
        if any(len(raw_cells[i]) > 4 and i not in marker_ids
               for i in {int(raw_ids[p]) for p in ink_cells}):
            continue
        arrows = []
        centers = {add(p,d) for p in ink_cells for d in D}
        for center in sorted(centers):
            for direction in D:
                geometry = _geometry(center, direction, grid)
                if geometry is None:
                    continue
                stencil = (*geometry['arms'], geometry['tip'])
                if not all(grid[p] == ink for p in stencil):
                    continue
                if any(not raw_cells[raw_ids[p]] <= geometry['write'] for p in stencil):
                    continue
                side_a, side_b = geometry['sides']
                if grid[side_a] != grid[side_b] or grid[side_a] == ink:
                    continue
                host = int(grid[side_a])
                if grid[geometry['targetcell']] in (ink, host):
                    continue
                explicit = int(grid[center]) != ink
                arrows.append(dict(geometry, host=host, explicit=explicit,
                                   payload=int(grid[center]) if explicit else host))
        used, valid = set(), bool(arrows)
        for arrow in arrows:
            if used & arrow['write']:
                valid = False
            used.update(arrow['write'])
        covered = used & ink_cells
        for marker in markers:
            if used & marker['cells']:
                valid = False
            if marker['ink'] == ink:
                covered.update(marker['cells'])
        if not valid or covered != ink_cells:
            continue
        clean = grid.copy()
        for annotation in arrows + markers:
            for point in annotation.get('write', annotation.get('cells')):
                clean[point] = annotation['host']
        ids, colors, cells = components(clean)
        assignments = {}
        for arrow in arrows:
            source, target = int(ids[arrow['center']]), int(ids[arrow['targetcell']])
            arrow.update(source=source, target=target)
            if (source == target or any(ids[p] != source for p in arrow['write'])
                    or (target in assignments and assignments[target] != arrow['payload'])):
                valid = False
            assignments[target] = arrow['payload']
        if not valid:
            continue
        # Masks always refer to the original component IDs, even if painted
        # regions merge. No later transfer can read an already-updated fill.
        painted = clean.copy()
        for target, payload in assignments.items():
            painted[ids == target] = payload
        turn = markers[0]['turn'] if markers else 0
        answers.append((np.rot90(painted, turn).copy(),
                        dict(ink=ink, arrows=arrows, markers=markers, clean=clean,
                             ids=ids, colors=colors, cells=cells, assignments=assignments,
                             painted=painted, turn=turn)))
    _need(len(answers) == 1, 'annotation interpretation is not unique')
    return answers[0]


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, seed, num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if value is not None and (not _is_int(value) or not low <= value <= high):
            raise ValueError(name + ' is outside its legal integer range')
    if num_train is None and num_test is None:
        legal = [(a, b) for b in range(1, 5) for a in range(b + 1, 11)]
        return legal[(17 * seed + 11) % len(legal)]
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _geometry(center, direction, grid):
    perpendicular = (-direction[1], direction[0])
    arms = (add(center, perpendicular), add(center, perpendicular, -1))
    tip = add(center, direction)
    target = add(center, direction, 2)
    sides = (add(tip, perpendicular), add(tip, perpendicular, -1))
    write = {center, tip, *arms}
    if not all(inside(p, grid) for p in write | {target, *sides}):
        return None
    return dict(center=center, d=direction, arms=arms, tip=tip,
                targetcell=target, sides=sides, write=write)


def _corner_cells(row, col, turn):
    side = 0 if turn == -1 else 2
    return {(row, col + i) for i in range(3)} | {(row + i, col + side) for i in range(3)}


def _features(info):
    arrows, colors, cells = info['arrows'], info['colors'], info['cells']
    assignments = info['assignments']
    final = [assignments.get(i, c) for i, c in enumerate(colors)]
    explicit = [a for a in arrows if a['explicit']]
    locality = any(colors[a] == colors[b] and final[a] != final[b]
                   for a in range(len(colors)) for b in range(a))
    chain = any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                and a['payload'] != b['payload'] for a in arrows for b in arrows if a is not b)
    boxes = [(min(p[0] for p in s), max(p[0] for p in s),
              min(p[1] for p in s), max(p[1] for p in s)) for s in cells]
    narrow_ids = {i for i,b in enumerate(boxes) if min(b[1]-b[0]+1, b[3]-b[2]+1) <= 2}
    contexts = [set(a['sides']) | {a['targetcell']} for a in arrows]
    markers = info['markers']
    h, w = info['clean'].shape
    return dict(
        arrows=len(arrows), explicit=len(explicit),
        singleton=sum(a['payload'] not in colors for a in explicit),
        existing=sum(a['payload'] in colors and a['payload'] != a['host'] for a in explicit),
        repeated=len(colors) > len(set(colors)), locality=locality, chain=chain,
        open_chain=any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                       and a['source'] != b['target'] and a['payload'] != b['payload']
                       for a in arrows for b in arrows if a is not b),
        cycle=any(a['source'] == b['target'] and a['target'] == b['source']
                  for a in arrows for b in arrows if a is not b),
        fanout=len({a['source'] for a in arrows}) < len({(a['source'], a['target']) for a in arrows}),
        multiple_incoming=len(assignments) < len(arrows),
        territories=len(colors), target_fraction=len(assignments) / len(colors),
        all_changed=all(c != v for c, v in zip(colors, final)),
        nonrectangular=sum(len(s) != (b[1]-b[0]+1)*(b[3]-b[2]+1) for s, b in zip(cells, boxes)),
        narrow=sum(min(b[1]-b[0]+1, b[3]-b[2]+1) <= 2 for b in boxes),
        narrow_host=any(a['source'] in narrow_ids for a in arrows),
        one_cell_target=any(min(boxes[a['target']][1]-boxes[a['target']][0]+1,
                               boxes[a['target']][3]-boxes[a['target']][2]+1) == 1 for a in arrows),
        different_back=any(inside(add(a['center'],a['d'],-1), info['clean'])
                           and info['clean'][add(a['center'],a['d'],-1)] != a['host'] for a in arrows),
        shared_context=any(contexts[i] & contexts[j] for i in range(len(arrows)) for j in range(i)),
        internal=sum(b[0] > 0 and b[1] < h-1 and b[2] > 0 and b[3] < w-1 for b in boxes),
        areas=[len(s) for s in cells], turn=info['turn'], ink=info['ink'],
        explicit_fraction=len(explicit)/len(arrows),
        marker_ink=markers[0]['ink'] if markers else None,
        marker_origin=markers[0]['origin'] if markers else None,
        nonlattice_marker=bool(markers and any(x % 5 != 1 for x in markers[0]['origin'])),
        small_marker_host=bool(markers and len(cells[int(info['ids'][next(iter(markers[0]['cells']))])]) < 50),
        separate_marker=bool(markers and markers[0]['ink'] != info['ink']),
        edge_marker=bool(markers and any(r in (0,h-1) or c in (0,w-1) for r,c in markers[0]['cells'])),
        marker_host_changes=bool(markers and any(info['painted'][p] != markers[0]['host'] for p in markers[0]['cells'])),
        edge_arrow=any(any(r in (0,h-1) or c in (0,w-1) for r,c in a['write']) for a in arrows),
        directions=sorted({D.index(a['d']) for a in arrows}),
        aspect='square' if h == w else ('portrait' if h > w else 'landscape'), shape=(h,w))


def _has(info, requests):
    features = _features(info)
    return all(bool(features[name]) for name in requests)


def _connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        point = stack.pop()
        for direction in D:
            neighbor = add(point, direction)
            if neighbor in points:
                points.remove(neighbor)
                stack.append(neighbor)
    return not points


def _partition(rng, shape, count, roughness, fallback=False):
    """Compose unequal cuts, stepped cuts and enclosed subregions in pixels."""
    h, w = shape
    labels = np.zeros(shape, dtype=np.int16)
    rr, cc = np.indices(shape)
    if fallback:
        # Variable-width bands provide a bounded constructive safety net.
        widths = np.full(count, 3, dtype=int)
        for _ in range(w - 3 * count):
            widths[int(rng.integers(count))] += 1
        start = 0
        for label, width in enumerate(widths):
            labels[:, start:start+width] = label
            start += width
        return labels
    for new_id in range(1, count):
        completed = False
        for _ in range(65):
            sizes = np.bincount(labels.ravel(), minlength=new_id).astype(float)
            label = int(rng.choice(new_id, p=sizes / sizes.sum()))
            mask = labels == label
            rows, cols = np.nonzero(mask)
            r0, r1, c0, c1 = rows.min(), rows.max()+1, cols.min(), cols.max()+1
            if min(r1-r0, c1-c0) >= 7 and rng.random() < .18 * roughness:
                top = int(rng.integers(r0+1, r1-4))
                bottom = int(rng.integers(top+3, r1))
                left = int(rng.integers(c0+1, c1-4))
                right = int(rng.integers(left+3, c1))
                section = (rr >= top) & (rr < bottom) & (cc >= left) & (cc < right)
            else:
                axis = int(rng.integers(2))
                lo, hi = (r0, r1) if axis == 0 else (c0, c1)
                if hi-lo < 4:
                    continue
                cut = int(rng.integers(lo+1, hi))
                coordinate, along = (rr, cc) if axis == 0 else (cc, rr)
                path = np.full(w if axis == 0 else h, cut)
                if rng.random() < .65 * roughness:
                    value = cut
                    step = int(rng.integers(2, 7))
                    for start in range(0, len(path), step):
                        value = int(np.clip(value + rng.integers(-2, 3), lo+1, hi-1))
                        path[start:start+step] = value
                section = coordinate < path[along]
            part, other = mask & section, mask & ~section
            if (int(part.sum()) < 8 or int(other.sum()) < 8
                    or not _connected(part) or not _connected(other)):
                continue
            labels[part] = new_id
            completed = True
            break
        if not completed:
            return None
    return labels


def _adjacency(labels, count):
    neighbors = [set() for _ in range(count)]
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        for x, y in zip(a[a != b], b[a != b]):
            neighbors[int(x)].add(int(y))
            neighbors[int(y)].add(int(x))
    return neighbors


def _palette(rng, labels, count, ink, marker_ink, repeat):
    available = [c for c in range(10) if c not in (ink, marker_ink)]
    rng.shuffle(available)
    neighbors = _adjacency(labels, count)
    colors = [-1] * count
    if repeat:
        choices = [(a,b) for a in range(count) for b in range(a) if b not in neighbors[a]]
        if not choices:
            return None
        a,b = choices[int(rng.integers(len(choices)))]
        colors[a] = colors[b] = available[0]
    # Reserve a literal color for singleton payloads even on large partitions.
    pool = available[:-1]
    for label in rng.permutation(count):
        label = int(label)
        if colors[label] >= 0:
            continue
        valid = [c for c in pool if c not in {colors[n] for n in neighbors[label]}]
        unused = [c for c in valid if c not in colors]
        if not valid:
            return None
        choices = unused if unused else valid
        colors[label] = int(choices[int(rng.integers(len(choices)))])
    return np.array(colors, dtype=np.int8)


def _boundary_candidates(labels):
    candidates = []
    for center in np.ndindex(labels.shape):
        for direction in D:
            geo = _geometry(center, direction, labels)
            if geo is None:
                continue
            source, target = int(labels[center]), int(labels[geo['targetcell']])
            if source == target:
                continue
            if all(labels[p] == source for p in geo['write'] | set(geo['sides'])):
                candidates.append(dict(geo, source=source, target=target))
    return candidates


def _marker(rng, labels, turn):
    if turn == 0:
        return None
    choices, edge = [], []
    h,w = labels.shape
    for row in range(h-2):
        for col in range(w-2):
            if np.all(labels[row:row+3, col:col+3] == labels[row,col]):
                item = dict(cells=_corner_cells(row,col,turn),
                            patch={(row+r,col+c) for r in range(3) for c in range(3)},
                            host=int(labels[row,col]), turn=turn)
                choices.append(item)
                if row in (0,h-3) or col in (0,w-3):
                    edge.append(item)
    if not choices:
        return None
    pool = edge if edge and rng.random() < .4 else choices
    return pool[int(rng.integers(len(pool)))]


def _plan(rng, candidates, colors, ink, marker_ink, marker, desired, requests, all_targets, explicit_rate):
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    chosen, assignments, blocked = [], {}, set()
    used_write = set()
    if marker:
        blocked.update(marker['patch'])
        used_write.update(marker['cells'])
        blocked.update(add(p,d) for p in marker['cells'] for d in D)
    used_context = set()

    def legal(candidate):
        return not (candidate['write'] & (blocked | used_context)
                    or set(candidate['sides']) & used_write
                    or candidate['targetcell'] in used_write)

    def append(candidate, payload, explicit):
        target = candidate['target']
        if not legal(candidate) or payload == colors[target]:
            return False
        if target in assignments and assignments[target] != payload:
            return False
        chosen.append(dict(candidate, payload=int(payload), explicit=explicit))
        assignments[target] = int(payload)
        used_write.update(candidate['write'])
        # Shared read-only context is allowed. Only annotation writes and their
        # immediate ink connectivity need separation, not deep support stencils.
        for point in candidate['write']:
            blocked.add(point)
            blocked.update(add(point,d) for d in D)
        used_context.update(candidate['sides'])
        used_context.add(candidate['targetcell'])
        return True

    if 'chain' in requests:
        found = False
        for first in candidates:
            if not legal(first):
                continue
            for second in candidates:
                if second['source'] != first['target']:
                    continue
                halo = first['write'] | {add(p,d) for p in first['write'] for d in D}
                if (not legal(second) or second['write'] & (halo | set(first['sides']) | {first['targetcell']})
                        or set(second['sides']) & first['write'] or second['targetcell'] in first['write']):
                    continue
                if append(first, colors[first['source']], False) and append(second, colors[second['source']], False):
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    for kind in ('singleton', 'existing'):
        if kind not in requests:
            continue
        found = False
        for candidate in candidates:
            if candidate['target'] in assignments or not legal(candidate):
                continue
            choices = ([c for c in range(10) if c not in colors and c not in (ink, marker_ink)]
                       if kind == 'singleton' else
                       [int(c) for c in set(colors) if c not in (colors[candidate['source']], colors[candidate['target']])])
            if choices and append(candidate, choices[int(rng.integers(len(choices)))], True):
                found = True
                break
        if not found:
            return None
    for candidate in candidates:
        if len(chosen) >= desired:
            break
        if not legal(candidate):
            continue
        target, source = candidate['target'], candidate['source']
        if target in assignments:
            if all_targets or rng.random() > .25:
                continue
            payload = assignments[target]
            explicit = payload != colors[source]
            if explicit and explicit_rate == 0:
                continue
        else:
            explicit = bool(rng.random() < explicit_rate)
            choices = [c for c in range(10) if c not in (ink, marker_ink, colors[target], colors[source])]
            payload = int(rng.choice(choices)) if explicit else int(colors[source])
        append(candidate, payload, explicit)
    if len(chosen) != desired or (all_targets and len(assignments) != len(colors)):
        return None
    return chosen, assignments


def _build_pair(rng, params, regime, turn, requests, fallback=False):
    needed = 2 * ('chain' in requests) + ('singleton' in requests) + ('existing' in requests)
    low = max(2, needed, 4 if 'locality' in requests else 2)
    count = int(rng.integers(low, min(9, low+4)+1))
    shifted_test = params['scale'] == 0 and regime == 'target_test'
    if shifted_test:
        count = int(rng.integers(max(6, low), 10))
    if fallback:
        count = max(low, 5)
        shape = (int(rng.integers(12, 25)), 3*count + int(rng.integers(0, 31-3*count)))
    else:
        small = params['scale'] == 0 and regime == 'source_train'
        big = params['scale'] == 0 and regime == 'target_test'
        bounds = (9,20) if small else ((21,31) if big else (9,31))
        if params['scale'] == 2:
            bounds = max(9, params['size_center']-4), min(31, params['size_center']+5)
        h,w = int(rng.integers(*bounds)), int(rng.integers(*bounds))
        if rng.random() < .25:
            w = h
        if big and rng.random() < .18:
            h = w = 30
        shape = h,w
    roughness = float(rng.choice([.4, 1., 1.5] if shifted_test else [0., .4, 1., 1.5]))
    labels = _partition(rng, shape, count, roughness, fallback)
    if labels is None:
        return None
    ink = int(rng.integers(10)) if params['vary_ink'] else params['ink']
    marker_ink = (int(rng.choice([c for c in range(10) if c != ink]))
                  if turn and rng.random() < .6 else ink)
    repeat = 'locality' in requests or (params['repeat'] and rng.random() < .6)
    colors = _palette(rng, labels, count, ink, marker_ink, repeat)
    if colors is None:
        return None
    marker = _marker(rng, labels, turn)
    if turn and marker is None:
        return None
    candidates = _boundary_candidates(labels)
    all_targets = count <= 8 and rng.random() < .2
    desired = count if all_targets else int(rng.integers(max(1, needed), min(8, max(2, count+2))+1))
    if shifted_test:
        desired = max(desired, int(rng.integers(4, 9)))
        if all_targets:
            desired = count
    if not requests and rng.random() < (.08 if shifted_test else .3):
        desired, all_targets = int(rng.integers(1,3)), False
    if fallback:
        desired, all_targets = max(needed, 3), False
    explicit_rate = float(rng.choice([0., .15, .4, .7, 1.])) if params['payload'] else 0.
    plan = _plan(rng, candidates, colors, ink, marker_ink, marker, desired, requests, all_targets, explicit_rate)
    if plan is None:
        return None
    arrows, assignments = plan
    clean = colors[labels]
    inp, painted = clean.copy(), clean.copy()
    for arrow in arrows:
        for point in arrow['write']:
            inp[point] = ink
        if arrow['explicit']:
            inp[arrow['center']] = arrow['payload']
    if marker:
        for point in marker['cells']:
            inp[point] = marker_ink
    for target, payload in assignments.items():
        painted[labels == target] = payload
    if np.array_equal(painted, np.rot90(painted, 2)):
        return None
    out = np.rot90(painted, turn).copy()
    try:
        predicted, info = _decode(inp)
    except AssertionError:
        return None
    if (not np.array_equal(predicted, out) or not np.array_equal(info['clean'], clean)
            or len(info['arrows']) != desired or not _has(info, requests)):
        return None
    return dict(input=inp, output=out), info


def _generate_pair(rng, params, regime, turn, requests):
    for attempt in range(180):
        result = _build_pair(rng, params, regime, turn, requests, fallback=attempt >= 110)
        if result is not None:
            return result
    raise AssertionError('bounded construction exhausted')


def _array_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p['input']), _array_key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _training_conventions(train_pairs):
    turns, mechanisms = {}, set()
    for pair in train_pairs:
        _, info = _decode(pair['input'])
        possible = [k for k in (-1,0,1) if np.array_equal(np.rot90(info['painted'], k), pair['output'])]
        _need(info['turn'] in possible, 'training violates the source handedness')
        if len(possible) == 1:
            turns[info['turn']] = possible[0]
        features = _features(info)
        mechanisms.update(k for k in ('singleton', 'existing', 'locality', 'chain') if features[k])
        if features['explicit']:
            mechanisms.add('payload')
    return turns, mechanisms


def solve(train_pairs, test_input):
    """Prediction receives training copies and a query input, never test labels."""
    turns, mechanisms = _training_conventions(train_pairs)
    _, info = _decode(test_input)
    _need(info['turn'] in turns, 'test turn lacks an asymmetric demonstration')
    features = _features(info)
    # Explicit junctions demonstrate a literal-color operation. The original
    # training set teaches it with singleton colors; that also determines a
    # test payload already present in another territory. Generated curricula
    # deliberately provide both contrasts, but that is not a source invariant.
    _need(not features['explicit'] or 'payload' in mechanisms, 'explicit junctions are not demonstrated')
    for feature in ('locality', 'chain'):
        _need(not features[feature] or feature in mechanisms, 'test mechanism not demonstrated: ' + feature)
    return np.rot90(info['painted'], turns[info['turn']]).copy()


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _is_int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    params = dict(ink=int(rng.integers(10)), vary_ink=bool(rng.random() < .5),
                  payload=bool(rng.random() < .75), repeat=bool(rng.random() < .65),
                  scale=int(rng.integers(3)), size_center=int(rng.integers(13,27)))
    turn_options = [(0,), (0,-1), (0,1), (-1,1)]
    if train_count >= 3:
        turn_options.append((0,-1,1))
    turns = turn_options[int(rng.integers(len(turn_options)))]
    train_turns = list(turns) + [int(rng.choice(turns)) for _ in range(train_count-len(turns))]
    rng.shuffle(train_turns)
    test_turns = [int(rng.choice(turns)) for _ in range(test_count)]
    puzzle, seen = dict(train=[], test=[]), set()
    requests = [set() for _ in range(train_count)]
    if params['payload']:
        requests[0].update(('singleton', 'existing'))
    if params['repeat']:
        requests[int(rng.integers(min(2, train_count)))].add('locality')
    requests[1].add('chain')
    for index in range(train_count):
        for _ in range(12):
            pair, info = _generate_pair(rng, params, 'source_train', train_turns[index], requests[index])
            if _array_key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('duplicate construction exhausted')
        puzzle['train'].append(pair)
        seen.add(_array_key(pair['input']))
    _, taught = _training_conventions(puzzle['train'])
    for turn in test_turns:
        for _ in range(60):
            pair, info = _generate_pair(rng, params, 'target_test', turn, set())
            features = _features(info)
            if any(features[k] and k not in taught for k in ('singleton', 'existing', 'locality', 'chain')):
                continue
            if _array_key(pair['input']) not in seen:
                puzzle['test'].append(pair)
                seen.add(_array_key(pair['input']))
                break
        else:
            raise AssertionError('could not construct a covered test')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a,b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    source_keys = {(_array_key(i), _array_key(o)) for i,o in _source_arrays()}
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            for grid in pair.values():
                _need(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D array')
                _need(np.issubdtype(grid.dtype, np.integer), 'grid must have integer dtype')
                _need(all(1 <= n <= 30 for n in grid.shape), 'grid dimensions outside 1..30')
                _need(bool(np.all((grid >= 0) & (grid <= 9))), 'illegal color')
            ikey, okey = _array_key(pair['input']), _array_key(pair['output'])
            _need(ikey not in seen and (ikey,okey) not in source_keys, 'duplicate or source copy')
            seen.add(ikey)
            predicted, info = _decode(pair['input'])
            _need(np.array_equal(predicted, pair['output']), 'incorrect component transfer or rotation')
            _need(not np.array_equal(pair['input'], pair['output']), 'unchanged pair')
            if info['turn']:
                _need(not np.array_equal(info['painted'], np.rot90(info['painted'],2)), 'ambiguous handedness')
    turns, taught = _training_conventions(puzzle['train'])
    for pair in puzzle['test']:
        _, info = _decode(pair['input'])
        _need(info['turn'] in turns, 'missing turn demonstration')
        features = _features(info)
        _need(all(not features[k] or k in taught for k in ('singleton', 'existing', 'locality', 'chain')),
              'test mechanism absent from training')


def _source_arrays():
    return [(np.array([[int(c) for c in row] for row in inp.split()], dtype=np.int8),
             np.array([[int(c) for c in row] for row in out.split()], dtype=np.int8))
            for inp,out in _SOURCE_CASES]


def self_test() -> None:
    source = _source_arrays()
    for inp,out in source:
        predicted, _ = _decode(inp)
        _need(np.array_equal(predicted,out), 'source regression failed')
    source_train = [dict(input=i.copy(), output=o.copy()) for i,o in source[:4]]
    for inp,out in source[4:]:
        _need(np.array_equal(solve(source_train, inp.copy()),out), 'source test prediction failed')
    for inp,out in _source_arrays():
        _, info = _decode(inp)
        if info['markers']:
            changed = inp.copy()
            for point in info['markers'][0]['cells']:
                changed[point] = info['ink']
            _need(np.array_equal(_decode(changed)[0],out), 'marker role depends on ink')
    features, puzzles, counts, partitions = [], set(), [set(),set()], set()
    within, directions = [], {'train': set(), 'test': set()}
    incoming_case = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        key = _puzzle_key(puzzle)
        puzzles.add(key)
        counts[0].add(len(puzzle['train']))
        counts[1].add(len(puzzle['test']))
        if seed < 2:
            _need(key == _puzzle_key(generate_puzzle(seed)), 'nondeterministic generation')
        first_feature = len(features)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _, info = _decode(pair['input'])
                features.append(_features(info))
                directions[split].update(features[-1]['directions'])
                partitions.add(_array_key(info['ids']))
                if incoming_case is None and features[-1]['multiple_incoming']:
                    incoming_case = pair['input'].copy(), info
        local = features[first_feature:]
        within.append({k: len({str(f[k]) for f in local})
                       for k in ('shape','arrows','territories','ink','explicit')})
        if any(f['singleton'] for f in local):
            _need(any(f['existing'] for f in local), 'singleton/existing payload contrast is missing')
        train = [{k:v.copy() for k,v in p.items()} for p in puzzle['train']]
        for pair in puzzle['test']:
            predicted = solve(train, pair['input'].copy())
            _need(np.array_equal(predicted, pair['output']), 'input-only test prediction failed')
    _need(len(puzzles) == 42 and len(partitions) > 150, 'collapsed puzzle geometry')
    _need(len(counts[0]) >= 4 and len(counts[1]) >= 3, 'collapsed automatic counts')
    for key in ('singleton', 'existing', 'locality', 'chain', 'open_chain', 'cycle', 'fanout', 'multiple_incoming',
                'all_changed', 'nonrectangular', 'narrow', 'internal', 'separate_marker', 'edge_marker',
                'marker_host_changes', 'edge_arrow', 'narrow_host', 'one_cell_target', 'different_back',
                'shared_context', 'nonlattice_marker', 'small_marker_host'):
        _need(sum(bool(f[key]) for f in features) >= 3, 'missing measured variation: ' + key)
    _need(all(d == set(range(4)) for d in directions.values()), 'missing split-wise directions')
    _need(all(all(p[k] >= 2 for k in ('shape','arrows','territories')) for p in within),
          'missing within-puzzle geometry/count variation')
    _need(sum(p['ink'] > 1 for p in within) >= 5 and sum(p['ink'] == 1 for p in within) >= 5,
          'missing variable or consistent annotation palettes')
    _need({f['ink'] for f in features} == set(range(10)), 'missing ink colors')
    _need({(f['turn'],f['aspect']) for f in features} == {(t,a) for t in (-1,0,1) for a in ('square','portrait','landscape')},
          'aspect and turn are correlated')
    _need({1,2,6,7,8} <= {f['arrows'] for f in features}, 'collapsed transfer counts')
    _need(any(f['explicit'] == 0 for f in features), 'missing entirely implicit examples')
    _need(any(f['locality'] and f['singleton'] and f['narrow'] and f['separate_marker'] for f in features),
          'missing combined extensions')
    # Conflicting instructions cannot silently acquire an order or priority.
    bad, info = incoming_case
    arrows = info['arrows']
    target = next(a['target'] for a in arrows if sum(b['target'] == a['target'] for b in arrows) > 1)
    changed = next(a for a in arrows if a['target'] == target)
    bad[changed['center']] = next(c for c in range(10) if c not in (info['ink'], changed['payload']))
    try:
        _decode(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting incoming instructions were accepted')
    for index, counts in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        puzzle = generate_puzzle(1000+index, *counts)
        for split, requested in zip(('train','test'),counts):
            _need(requested is None or len(puzzle[split]) == requested, 'ignored explicit count')
        validate_generated_puzzle(puzzle)
    for counts in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(0,*counts)
        except ValueError:
            continue
        raise AssertionError('illegal count request accepted')


# Source fixtures are used only for regression and exact-copy exclusion.
_SOURCE_CASES = (
    ('666667774444 666667774444 666167774444 666317774999 666167774449 666667774449 666667174444 666667114444 666667174444 666667774444 771777777777 710177777111 777777777717 888888888888 888888888888 888888888888', '7777777777333777 7777777777333777 7777777777333777 7777777777333777 3333333333333777 3333333333333777 3333333333333777 0000000000333777 0000000000333777 0000000000333777 0000000000333777 0000000000333777'),
    ('7777771733 7777771133 7777771733 8883177731 8881177716 8183177731 8113319133 8183371733 8883355533 8883355533', '6666666677 6666666677 6666666677 3338866677 3338866677 3338866677 3338866677 3338866677 3338899977 3338899977'),
    ('666313333 666113333 666313333 666444414 666444111 666444444 666616444 666611444 666616444 555444444 999888888 955888171 955888818 555444444', '55555333333333 55555333333333 55555333333333 78886333666444 78886333666444 78886333666444 78886666666444 78886666666444 78886666666444'),
    ('2222212333 2222211333 2222212333 6666166313 6661116113 1116666313 6166666333 4444444333 4444444333 4444444333', '6666666222 6666666222 6666666222 3333333222 3333333222 3333333222 3333333222 6666666222 6666666222 6666666222'),
    ('999999999999999999999999999999 999999999999999999999999977999 999999999993331333333999777799 999933333333311133333999777799 999333333333333333333399777799 993333333333333333333399919999 993333333333333333333399111994 933330100000000033333399999944 933301810000000000000099999444 933000000000021220000099999444 933000000022211122220099994444 933000000222222222222099994444 933000002222222222222299994444 933000002222222222222229944444 666000002222222222222229944444 666000002222222222222229944444 666600000000000222222229944444 666600111000000222222222944444 666666010000002222222222944444 666666666666002222222222944444 666666666666162222222222994444 666666666666112222222229994444 666666666666169222222229998844 666666666666669222222299988844 666666666666699922229999888844 666666666669999999999988888818 666669999999999988888888888111 999999999999199888888888888888 999999999999018888888888888888 999999999999198888888888888888', '333333333333333333333333333333 333333333333333333333333399333 333333333338888888888333999933 333388888888888888888333999933 333888888888888888888833999933 338888888888888888888833333333 338888888888888888888833333338 388882222222222288888833333388 388822222222222222222233333888 388222222222266662222233333888 388222222266666666662233338888 388222222666666666666233338888 388222226666666666666633338888 388222226666666666666663388888 000222226666666666666663388888 000222226666666666666663388888 000022222222222666666663388888 000022222222222666666666388888 000000222222226666666666388888 000000000000226666666666388888 000000000000006666666666338888 000000000000006666666663338888 000000000000003666666663330088 000000000000003666666633300088 000000000000033366663333000088 000000000003333333333300000000 000003333333333300000000000000 333333333333333000000000000000 333333333333330000000000000000 333333333333330000000000000000'),
    ('555555555555555555555555 555555555555555555555555 555555555555555555551115 555555555555555555555155 331333888888888888666666 311133888888888888666666 333333888999888888666666 333333888889888888666666 333333888889888888666666 333333888888888888666666 333333888888888888666666 333333888888888888666666 333333888888888888666666 333333888888888888666666 333333817188888888666666 333333881888888888666666 333333444444144444666666 333333444441114444666666 333333444444444444661116 333333444444444444666166 221222222222222222222222 211122222222222222222222 222222222222222222222222 222222222222222222222222', '333355555555555555556666 333355555555555555556666 333355555555555555556666 333355555555555555556666 333355555555555555556666 333355555555555555556666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333344444444444477776666 333322222222222222226666 333322222222222222226666 333322222222222222226666 333322222222222222226666 333322222222222222226666 333322222222222222226666'),
)
