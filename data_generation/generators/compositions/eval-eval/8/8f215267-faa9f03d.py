"""Count-addressed switches on repaired, rooted tracks.

The input parser is independent of the random construction. 8 is a candidate
direction marker, 9 a repair placeholder; all other colors have local roles.
"""
from collections import deque
import numpy as np


DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SWITCH, REPAIR = 8, 9


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(bool(np.all((a >= 0) & (a <= 9))), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _neighbors(p, cells):
    return [q for d in DIRECTIONS if (q := _add(p, d)) in cells]


def _line(a, b):
    _check(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    length = max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [(a[0]+i*d[0], a[1]+i*d[1]) for i in range(length+1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = []
    while _inside(p := _add(p, d), shape):
        result.append(p)
    return result


def _moat(cells, radius=1):
    return {(r+dr, c+dc) for r, c in cells
            for dr in range(-radius, radius+1) for dc in range(-radius, radius+1)}


def _perimeter(box):
    r, c, h, w = box
    return {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)
            if rr in (r, r+h-1) or cc in (c, c+w-1)}


def _components(a, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        color = int(a[p])
        cells, queue = {p}, [p]
        while queue:
            for q in _neighbors(queue.pop(), remaining):
                if int(a[q]) == color:
                    remaining.remove(q)
                    cells.add(q)
                    queue.append(q)
        result.append((color, cells))
    return result


def _roots(work, bg):
    roots = []
    h, w = work.shape
    for r, c in map(tuple, np.argwhere(work != bg)):
        color = int(work[r, c])
        if color in (SWITCH, REPAIR):
            continue
        for sh in (3, 5, 7):
            for sw in (3, 5, 7):
                if r+sh > h or c+sw > w:
                    continue
                a = work[r:r+sh, c:c+sw]
                if (np.all(a[0] == color) and np.all(a[-1] == color)
                        and np.all(a[:, 0] == color) and np.all(a[:, -1] == color)
                        and np.all(a[1:-1, 1:-1] == bg)):
                    box = (int(r), int(c), sh, sw)
                    roots.append(dict(color=color, root=_perimeter(box), box=box))
    _check(1 <= len(roots) <= 3, "expected one to three hollow roots")
    _check(len({t['color'] for t in roots}) == len(roots), "one root per category color")
    return roots


def _inspect(input_grid):
    """Recover repairs, category roots, ordered tracks, switches and objects."""
    inp = _grid(input_grid)
    histogram = np.bincount(inp.ravel(), minlength=10)
    bg = int(histogram.argmax())
    _check(histogram[bg]*2 > inp.size and bg not in (SWITCH, REPAIR), "background must be a strict majority")
    work = inp.copy()
    pending = {tuple(map(int, p)) for p in np.argwhere(inp == REPAIR)}
    marked = set(pending)
    for _ in range(len(pending)+1):
        changed = False
        for p in sorted(pending):
            colors = {int(work[q]) for d in DIRECTIONS
                      if _inside(q := _add(p, d), work.shape)
                      and int(work[q]) not in (bg, SWITCH, REPAIR)}
            if len(colors) == 1:
                work[p] = colors.pop()
                pending.remove(p)
                changed = True
        if not changed:
            break
    _check(not pending, "ambiguous repair placeholder")
    work[work == SWITCH] = bg
    tracks = _roots(work, bg)
    parts = _components(work, bg)
    used = set()
    for t in tracks:
        matches = [cells for color, cells in parts if color == t['color'] and t['root'] <= cells]
        _check(len(matches) == 1, "root component missing")
        network = matches[0]
        path_cells = network-t['root']
        starts = [p for p in path_cells if _neighbors(p, t['root'])]
        _check(len(starts) == 1, "root must have exactly one exit")
        start = starts[0]
        attachment = _neighbors(start, t['root'])
        _check(len(attachment) == 1, "exit touches more than one root pixel")
        r, c, sh, sw = t['box']
        ports = {(r, c+sw//2), (r+sh-1, c+sw//2), (r+sh//2, c), (r+sh//2, c+sw-1)}
        _check(attachment[0] in ports, "exit must use a side midpoint")
        path, previous, current = [], attachment[0], start
        while True:
            _check(current not in path, "track contains a second cycle")
            path.append(current)
            following = [p for p in _neighbors(current, path_cells) if p != previous]
            if not following:
                break
            _check(len(following) == 1, "track branches")
            previous, current = current, following[0]
        _check(set(path) == path_cells, "unvisited track fragment")
        _check(len(path) >= 4 and _edge(path[-1], inp.shape), "track must end on the boundary")
        _check(not any(_edge(p, inp.shape) for p in path[:-1]), "track touches boundary before endpoint")
        t.update(path=path, network=network, switches=[], attachment=attachment[0])
        used.update(network)
    # A candidate arrow has one adjacent track cell, at a corner, opposite the
    # outgoing continuation. Rank follows the path from the root, never pixels.
    for q in map(tuple, np.argwhere(inp == SWITCH)):
        candidates = []
        for t in tracks:
            near = _neighbors(q, set(t['path']))
            for pivot in near:
                index = t['path'].index(pivot)
                if index == 0 or index == len(t['path'])-1:
                    continue
                a, b = t['path'][index-1], t['path'][index+1]
                outgoing = (b[0]-pivot[0], b[1]-pivot[1])
                incoming = (pivot[0]-a[0], pivot[1]-a[1])
                d = (q[0]-pivot[0], q[1]-pivot[1])
                if d == (-outgoing[0], -outgoing[1]) and sum(x*y for x, y in zip(incoming, outgoing)) == 0:
                    candidates.append((t, dict(index=index, pivot=pivot, marker=q, direction=d)))
        _check(len(candidates) == 1, "switch must identify exactly one corner")
        t, switch = candidates[0]
        t['switches'].append(switch)
    for t in tracks:
        t['switches'].sort(key=lambda s: s['index'])
        _check(1 <= len(t['switches']) <= 5, "expected one to five switches per track")
        _check(len({s['pivot'] for s in t['switches']}) == len(t['switches']), "duplicate switch at pivot")
    objects = [(color, cells) for color, cells in parts if not cells & used]
    _check(all(1 <= len(cells) <= 10 for _, cells in objects), "object area outside 1..10")
    _check(marked <= used, "repair placeholders belong only to tracks")
    for t in tracks:
        t['objects'] = [cells for color, cells in objects if color == t['color']]
        t['count'] = len(t['objects'])
        _check(t['count'] <= len(t['switches']), "count exceeds available switch ranks")
        t['area'] = sum(map(len, t['objects']))
    return dict(background=bg, tracks=tracks, objects=objects, repairs=marked, work=work)


def _plan(track, shape, rank=None):
    rank = track['count'] if rank is None else rank
    if rank == 0:
        return set(track['network']), set(), set()
    s = track['switches'][rank-1]
    kept = track['root'] | set(track['path'][:s['index']+1])
    old = set(track['path'][s['index']+1:])
    ray = set(_ray(s['pivot'], s['direction'], shape))
    return kept, ray, old


def _render(tracks, shape, bg, mode='normal'):
    out = np.full(shape, bg, dtype=np.int8)
    retained, rays = [], []
    for t in tracks:
        if mode == 'no_rerouting':
            kept, ray = set(t['network']), set()
        elif mode == 'first_switch':
            kept, ray, _ = _plan(t, shape, int(t['count'] > 0))
        elif mode == 'screen_order' and t['count']:
            spatial = sorted(t['switches'], key=lambda s: s['marker'])
            selected = spatial[t['count']-1]
            actual_rank = t['switches'].index(selected)+1
            kept, ray, _ = _plan(t, shape, actual_rank)
        else:
            kept, ray, _ = _plan(t, shape)
        retained.append((t['color'], kept))
        rays.append((t['color'], ray))
    for collection in (retained, rays):
        for color, cells in collection:
            for p in cells:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from the input alone; no learned palette or seed."""
    d = _inspect(input_grid)
    return _render(d['tracks'], np.asarray(input_grid).shape, d['background'])


def _valid_walk(root, path, attachment):
    if len(set(path)) != len(path) or set(path) & root:
        return False
    s = set(path)
    for i, p in enumerate(path):
        expected = ({path[i-1]} if i else set()) | ({path[i+1]} if i+1 < len(path) else set())
        if set(_neighbors(p, s)) != expected:
            return False
        if set(_neighbors(p, root)) != ({attachment} if i == 0 else set()):
            return False
    return True


def _sample_track(rng, shape, color, occupied, minimum_switches):
    h, w = shape
    for _ in range(160):
        sh, sw = int(rng.choice((3, 3, 5))), int(rng.choice((3, 3, 5, 7)))
        r, c = int(rng.integers(1, h-sh)), int(rng.integers(1, w-sw))
        box = (r, c, sh, sw)
        root = _perimeter(box)
        full_box = {(rr, cc) for rr in range(r, r+sh) for cc in range(c, c+sw)}
        if full_box & _moat(occupied):
            continue
        d = DIRECTIONS[int(rng.integers(4))]
        attachment = (r if d[0] < 0 else r+sh-1 if d[0] > 0 else r+sh//2,
                      c if d[1] < 0 else c+sw-1 if d[1] > 0 else c+sw//2)
        first = _add(attachment, d)
        length = int(rng.integers(3, 8))
        end = (attachment[0]+length*d[0], attachment[1]+length*d[1])
        if not (1 <= end[0] < h-1 and 1 <= end[1] < w-1):
            continue
        vertices = [attachment, end]
        path = _polyline(vertices)[1:]
        if set(path) & _moat(occupied) or not _valid_walk(root, path, attachment):
            continue
        axis = 1 if d[0] else 0
        desired = int(rng.integers(minimum_switches+1, min(8, minimum_switches+4)+1))
        for _step in range(desired):
            p = vertices[-1]
            values = [v for v in range(1, shape[axis]-1) if 3 <= abs(v-p[axis]) <= 9]
            rng.shuffle(values)
            accepted = False
            for v in values:
                q = list(p)
                q[axis] = v
                candidate = vertices+[tuple(q)]
                candidate_path = _polyline(candidate)[1:]
                if not set(candidate_path) & _moat(occupied) and _valid_walk(root, candidate_path, attachment):
                    vertices, path = candidate, candidate_path
                    accepted = True
                    break
            if not accepted:
                break
            axis = 1-axis
        # The last segment reaches either end of its axis; no earlier cell is
        # on the boundary. A boundary extension can add one further corner.
        endings = [0, shape[axis]-1]
        rng.shuffle(endings)
        for value in endings:
            q = list(vertices[-1])
            q[axis] = value
            candidate = vertices+[tuple(q)]
            candidate_path = _polyline(candidate)[1:]
            if set(candidate_path) & _moat(occupied) or not _valid_walk(root, candidate_path, attachment):
                continue
            switches = []
            network = root | set(candidate_path)
            blocked = network | occupied
            for index in range(1, len(candidate_path)-1):
                a, pivot, b = candidate_path[index-1:index+2]
                incoming = (pivot[0]-a[0], pivot[1]-a[1])
                outgoing = (b[0]-pivot[0], b[1]-pivot[1])
                if incoming == outgoing:
                    continue
                direction = (-outgoing[0], -outgoing[1])
                marker = _add(pivot, direction)
                if not _inside(marker, shape) or marker in blocked:
                    continue
                if _neighbors(marker, blocked) != [pivot]:
                    continue
                if set(_ray(pivot, direction, shape)) & root:
                    continue
                switches.append(dict(index=index, pivot=pivot, marker=marker, direction=direction))
            if len(switches) < minimum_switches:
                continue
            number = int(rng.integers(minimum_switches, min(5, len(switches))+1))
            picks = sorted(map(int, rng.choice(len(switches), number, replace=False)))
            selected = [switches[i] for i in picks]
            return dict(color=color, root=root, box=box, path=candidate_path,
                        network=network, attachment=attachment, switches=selected)
    return None


def _shape(rng, area=None):
    if area is None:
        area = int(rng.integers(1, 11))
    if rng.random() < .25:
        cells = {(0, i) for i in range(area)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({q for p in cells for d in DIRECTIONS if (q := _add(p, d)) not in cells})
            rng.shuffle(frontier)
            for q in frontier:
                proposed = cells | {q}
                r0, r1 = min(p[0] for p in proposed), max(p[0] for p in proposed)
                c0, c1 = min(p[1] for p in proposed), max(p[1] for p in proposed)
                # Do not introduce holes that could look like hollow roots.
                empty = {(r, c) for r in range(r0-1, r1+2) for c in range(c0-1, c1+2)}-proposed
                reachable, queue = {(r0-1, c0-1)}, [(r0-1, c0-1)]
                while queue:
                    for x in _neighbors(queue.pop(), empty-reachable):
                        reachable.add(x)
                        queue.append(x)
                if reachable == empty:
                    cells = proposed
                    break
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return {(r-r0, c-c0) for r, c in cells}


def _place_objects(rng, inp, tracks, bg, distractors, forced):
    occupied = set().union(*(t['network'] | {s['marker'] for s in t['switches']} for t in tracks))
    # Reserve root interiors as well as their outlines.
    for t in tracks:
        r, c, h, w = t['box']
        occupied.update((rr, cc) for rr in range(r, r+h) for cc in range(c, c+w))
    specs = []
    for i, t in enumerate(tracks):
        for j in range(t['count']):
            specs.append((t['color'], _shape(rng, forced[i][j] if i in forced else None)))
    specs.extend((color, _shape(rng)) for color in distractors)
    specs.sort(key=lambda item: -len(item[1]))
    for color, cells in specs:
        sh, sw = 1+max(p[0] for p in cells), 1+max(p[1] for p in cells)
        candidates = [(r, c) for r in range(inp.shape[0]-sh+1) for c in range(inp.shape[1]-sw+1)]
        rng.shuffle(candidates)
        blocked = _moat(occupied, 1)
        for r, c in candidates:
            translated = {(r+dr, c+dc) for dr, dc in cells}
            if not translated & blocked:
                for p in translated:
                    inp[p] = color
                # Two background layers separate objects, excluding artificial
                # fragmentation and keeping shape completion out of counting.
                occupied.update(_moat(translated, 1))
                break
        else:
            return False
    return True


def _make_pair(rng, colors, counts, bg, distractors, forced=None, teaching=False,
               require_screen_witness=False):
    forced = {} if forced is None else forced
    for attempt in range(90):
        low = 22 if len(colors) >= 2 else 17
        if len(colors) == 3:
            low = 27
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        occupied, tracks = set(), []
        for color, count in zip(colors, counts):
            minimum = max(1, count+int(teaching and count > 0))
            t = _sample_track(rng, shape, color, occupied, minimum)
            if t is None:
                break
            t['count'] = count
            tracks.append(t)
            r, c, h, w = t['box']
            occupied.update((rr, cc) for rr in range(r, r+h) for cc in range(c, c+w))
            occupied.update(t['network'] | {s['marker'] for s in t['switches']})
        if len(tracks) != len(colors):
            continue
        plans = [_plan(t, shape) for t in tracks]
        if any((a[0] | a[1]) & (b[0] | b[1]) for i, a in enumerate(plans) for b in plans[i+1:]):
            continue
        out = _render(tracks, shape, bg)
        if require_screen_witness and np.array_equal(out, _render(tracks, shape, bg, 'screen_order')):
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        for t in tracks:
            for p in t['network']:
                inp[p] = t['color']
            for s in t['switches']:
                inp[s['marker']] = SWITCH
        if not _place_objects(rng, inp, tracks, bg, distractors, forced):
            continue
        marked = set()
        for t in tracks:
            preferred = []
            if t['count']:
                preferred.append(t['switches'][t['count']-1]['pivot'])
                # A deleted repair marker proves annotation repair does not
                # mean every placeholder must remain in the final output.
                if t['count'] < len(t['switches']):
                    preferred.append(t['switches'][-1]['pivot'])
            else:
                preferred.append(t['path'][-1])
            pool = list(t['path'])
            rng.shuffle(pool)
            desired = int(rng.integers(2, 7))
            for p in preferred+pool:
                if p in marked or any(abs(p[0]-q[0])+abs(p[1]-q[1]) <= 1 for q in marked):
                    continue
                if len({int(inp[q]) for d in DIRECTIONS if _inside(q := _add(p, d), shape)
                        and int(inp[q]) not in (bg, SWITCH, REPAIR)}) != 1:
                    continue
                inp[p] = REPAIR
                marked.add(p)
                desired -= 1
                if desired <= 0:
                    break
        if np.count_nonzero(inp == bg)*2 <= inp.size:
            continue
        # Construction creates labels from complete track geometry. The parser
        # must recover that geometry from the returned input, not from labels.
        try:
            recovered = _inspect(inp)
            if not np.array_equal(_render(recovered['tracks'], shape, bg), out):
                continue
        except AssertionError:
            continue
        return {'input': inp, 'output': out}
    raise AssertionError('bounded scene construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(8)))
    anchor, discarded, novel = palette[:3]
    collision_area = int(rng.integers(4, 10))
    a = int(rng.integers(1, collision_area))
    other_area = int(rng.choice([x for x in range(3, 13) if x != collision_area]))
    b = int(rng.integers(max(1, other_area-10), min(10, other_area-1)+1))
    regime = int(rng.integers(2))  # larger rank, or more simultaneous categories
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for index in range(nt+nq):
        train = index < nt
        local = index if train else index-nt
        forced = {}
        if index == 0:
            colors, counts = [anchor, discarded], [1, 2]
            forced = {0: [collision_area], 1: [b, other_area-b]}
        elif index == 1:
            colors, counts = [anchor, palette[3]], [2, 0]
            forced = {0: [a, collision_area-a]}
        else:
            n = 3 if not train and local == 0 and regime == 1 else int(rng.choice((1, 2)))
            available = [c for c in range(8) if not train or c != novel]
            colors = list(map(int, rng.choice(available, n, replace=False)))
            counts = list(map(int, rng.integers(0, 3, n)))
            if not train and local == 0:
                if novel not in colors:
                    colors[0] = novel
                if regime == 0:
                    counts[0] = int(rng.choice((3, 4)))
        bg_pool = [c for c in range(8) if c not in colors and c != previous_bg]
        bg = int(rng.choice(bg_pool))
        dpool = [c for c in range(8) if c not in colors and c != bg]
        distractors = list(map(int, rng.choice(dpool, int(rng.integers(0, 4)))))
        if index == 1:
            distractors = [discarded]+distractors[:2]
            if bg == discarded:
                bg = int(rng.choice([c for c in bg_pool if c != discarded]))
                # Resample incidental distractors to avoid their new background.
                distractors = [discarded]+[c for c in distractors[1:] if c != bg]
        for _ in range(5):
            pair = _make_pair(rng, colors, counts, bg, distractors, forced,
                              teaching=index < 2, require_screen_witness=index == 0)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise AssertionError('could not create distinct input')
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
        previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def _training_evidence(details):
    rows = [(t['color'], t['count'], t['area']) for d in details for t in d['tracks']]
    collision = any(c == cc and a == aa and n != nn for c, n, a in rows for cc, nn, aa in rows)
    independence = any(n == nn and a != aa for c, n, a in rows for cc, nn, aa in rows)
    return collision, independence


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _check(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
           and len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    seen, details = set(), {'train': [], 'test': []}
    spatial_witness = count_witness = deletion_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(inp.shape == out.shape, 'changed dimensions')
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            d = _inspect(inp)
            predicted = _render(d['tracks'], inp.shape, d['background'])
            _check(np.array_equal(predicted, out), 'wrong switch selection, repair, erasure or ray')
            _check(not np.array_equal(inp, out), 'unchanged example')
            _check(not np.any(np.isin(out, (SWITCH, REPAIR))), 'annotation left in output')
            tracks, objects = d['tracks'], d['objects']
            network = set().union(*(t['network'] for t in tracks))
            for color, cells in objects:
                _check(not cells & _moat(network, 1), 'object touches network')
                for other_color, other in objects:
                    if other is not cells:
                        _check(not cells & _moat(other, 2), 'objects need two background layers')
            plans = [_plan(t, inp.shape) for t in tracks]
            for i, t in enumerate(tracks):
                kept, ray, old = plans[i]
                _check(not ray & t['root'], 'ray cuts through its root')
                for j, u in enumerate(tracks):
                    if i != j:
                        _check(not (kept | ray) & (plans[j][0] | plans[j][1]), 'different output colors overlap')
                        _check(not t['network'] & _moat(u['network']), 'input tracks touch')
            if split == 'train':
                spatial_witness |= not np.array_equal(out, _render(tracks, inp.shape, d['background'], 'screen_order'))
                count_witness |= not np.array_equal(out, _render(tracks, inp.shape, d['background'], 'first_switch'))
                deletion_witness |= not np.array_equal(out, _render(tracks, inp.shape, d['background'], 'no_rerouting'))
            details[split].append(d)
    tr = details['train']
    _check(all(_training_evidence(tr)), 'training must disprove area counting')
    _check({0, 1, 2} <= {t['count'] for d in tr for t in d['tracks']}, 'training must teach zero, first and second rank')
    _check(spatial_witness and count_witness and deletion_witness, 'parent/ordering ablation witness missing')
    categories = {t['color'] for d in tr for t in d['tracks']}
    discarded = {c for d in tr for c, cells in d['objects'] if c not in {t['color'] for t in d['tracks']}}
    _check(bool(categories & discarded), 'local category/distractor role switch missing')
    _check(any(t['color'] not in categories for d in details['test'] for t in d['tracks']), 'query must include a new category color')


def puzzle_metrics(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _inspect(pair['input'])
            for t in d['tracks']:
                kept, ray, old = _plan(t, np.asarray(pair['input']).shape)
                directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(t['path'], t['path'][1:])]
                old_walk = t['path'][-len(old):] if old else []
                old_d = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(old_walk, old_walk[1:])]
                records.append(dict(split=split, shape=list(np.asarray(pair['input']).shape),
                    background=d['background'], color=t['color'], categories=len(d['tracks']),
                    root_shape=list(t['box'][2:]), count=t['count'], area=t['area'],
                    switches=len(t['switches']), turns=sum(a != b for a, b in zip(directions, directions[1:])),
                    old_turns=sum(a != b for a, b in zip(old_d, old_d[1:])),
                    old_length=len(old), ray_length=len(ray),
                    direction=list(t['switches'][t['count']-1]['direction']) if t['count'] else [],
                    repair_count=len(d['repairs'] & t['network']),
                    deleted_repairs=len(d['repairs'] & (old-ray)),
                    retained_repairs=len(d['repairs'] & (kept | ray)),
                    object_areas=sorted(map(len, t['objects'])),
                    new_old_crossings=sum(len(ray & u['network']) for u in d['tracks'] if u is not t)))
    return records


def self_test():
    """Bounded semantic, counterfactual, equivariance and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [r for p in puzzles for r in puzzle_metrics(p)]
    _check(len({tuple((q['input'].shape, q['input'].tobytes()) for s in ('train', 'test')
                     for q in p[s]) for p in puzzles}) == 42, 'cross-seed repetition')
    _check(len({tuple(r['shape']) for r in records}) >= 30, 'canvas diversity')
    _check({r['background'] for r in records} == set(range(8)), 'local background palette')
    _check({r['categories'] for r in records} == {1, 2, 3}, 'category-count diversity')
    _check({r['count'] for r in records} >= {0, 1, 2, 3, 4}, 'rank diversity')
    _check({tuple(r['direction']) for r in records if r['direction']} == set(DIRECTIONS), 'all ray directions')
    _check(len({tuple(r['root_shape']) for r in records}) >= 5, 'root geometry diversity')
    _check(len({r['turns'] for r in records}) >= 4, 'route geometry diversity')
    _check({r['switches'] for r in records} >= {1, 2, 3, 4, 5}, 'switch-count diversity')
    _check(max(r['old_turns'] for r in records) >= 3, 'bent suffix deletion')
    _check(sum(r['deleted_repairs'] for r in records) > 20, 'deleted marker coverage')
    _check(sum(r['retained_repairs'] for r in records) > 20, 'retained marker coverage')
    _check({a for r in records for a in r['object_areas']} == set(range(1, 11)), 'object-area diversity')
    for puzzle in puzzles[:3]:
        for pair in puzzle['train']+puzzle['test']:
            for k in range(1, 4):
                inp = np.rot90(np.fliplr(pair['input']), k)
                expected = np.rot90(np.fliplr(pair['output']), k)
                _check(np.array_equal(solve(inp), expected), 'symmetry equivariance')
            d = _inspect(pair['input'])
            _check(np.array_equal(_render(list(reversed(d['tracks'])), pair['input'].shape,
                                         d['background']), pair['output']), 'simultaneous edit order')
    _check(all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(puzzles[0][s], generate_puzzle(0)[s]) for k in a), 'determinism')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            _check(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Full-grid output negatives cover preserved root, selected ray, and erased
    # objects. They do not rely on a list of sampled feature flags.
    p = puzzles[0]
    q = p['test'][0]
    d = _inspect(q['input'])
    sites = [min(d['tracks'][0]['root'])]
    for t in d['tracks']:
        kept, ray, old = _plan(t, q['input'].shape)
        if ray:
            sites.append(min(ray))
        if old-ray:
            sites.append(min(old-ray))
    if d['objects']:
        sites.append(min(d['objects'][0][1]))
    for site in sites:
        bad = {'train': p['train'], 'test': [dict(pair) for pair in p['test']]}
        bad['test'][0]['output'] = q['output'].copy()
        bad['test'][0]['output'][site] = (int(q['output'][site])+1)%10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('validator accepted corrupted output')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds plus bounded count and semantic checks')
