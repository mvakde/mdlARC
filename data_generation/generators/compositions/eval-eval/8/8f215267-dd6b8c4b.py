"""Whole colored objects arrive through walls and occupy a framed event log.

All roles, paths, object units, capacity and ordering come from the input.
Standard library + NumPy only. Nothing is imported from the parent families.
"""
from collections import deque
import json
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 141568512, 842562934, 209442578, 571662478, 544276423,
              242856320, 205143853, 112420787, 576832802, 746976125, 56928606,
              273667837, 513658405, 324099995, 129260225, 723570346, 790404415,
              564598945, 126335079, 489191134, 867651111, 892783712, 81247788,
              522627293, 795498500, 26246767, 475499381, 267959602, 263938549,
              455535361, 749120586, 826641628, 850344772, 616964138, 769048592,
              649000768, 935118426, 988129308, 78013688, 10519207, 472267805)


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'grid dimensions')
    _check(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
           'grid colors must be integers in 0..9')
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, bool)
    parts = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = [(r, c)]
        seen[r, c] = True
        for rr, cc in queue:
            for dr, dc in STEPS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        parts.append(tuple(queue))
    return parts


def _frame_cells(box):
    r, c, h, w = box
    return {(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)
            if rr in (r, r+h-1) or cc in (c, c+w-1)}


def _footprint(shape, box):
    r, c, h, w = box
    mask = np.zeros(shape, bool)
    mask[r:r+h, c:c+w] = True
    return mask


def _distances(wall, footprint):
    h, w = wall.shape
    distance = np.full(wall.shape, -1, dtype=int)
    starts = [(int(r), int(c)) for r, c in zip(*np.nonzero(footprint))]
    distance[footprint] = 0
    queue = deque(starts)
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if (0 <= nr < h and 0 <= nc < w and not wall[nr, nc]
                    and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c]+1
                queue.append((nr, nc))
    return distance


def _inspect(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    _check(np.count_nonzero(counts == counts.max()) == 1, 'background must be unique mode')
    background = int(colors[counts.argmax()])
    frames = []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        for comp in _components(grid == color):
            rr, cc = zip(*comp)
            r, c = min(rr), min(cc)
            h, w = max(rr)-r+1, max(cc)-c+1
            if (h, w) not in ((3, 5), (3, 7), (3, 9), (5, 3),
                              (7, 3), (9, 3), (5, 5)):
                continue
            box = (r, c, h, w)
            if set(comp) != _frame_cells(box):
                continue
            patch = grid[r+1:r+h-1, c+1:c+w-1]
            center = (h//2-1, w//2-1)
            marker = int(patch[center])
            rest = patch.copy()
            rest[center] = background
            if marker not in (background, color) and np.all(rest == background):
                frames.append((box, color, marker))
    _check(len(frames) == 1, 'one unique hollow register with center key is required')
    box, frame, wall_color = frames[0]
    footprint = _footprint(grid.shape, box)
    _check(np.count_nonzero(grid == frame) == len(_frame_cells(box)),
           'frame color is reserved for its outline')
    wall = (grid == wall_color) & ~footprint
    distance = _distances(wall, footprint)
    r, c, h, w = box
    slots = [(rr, cc) for rr in range(r+1, r+h-1) for cc in range(c+1, c+w-1)]
    objects = []
    for color in colors:
        color = int(color)
        if color in (background, frame, wall_color):
            continue
        for comp in _components((grid == color) & ~footprint):
            ds = [int(distance[cell]) for cell in comp]
            _check(all(d < 0 for d in ds) or all(d >= 0 for d in ds),
                   'an object cannot straddle reachability regions')
            direct = min(max(r-rr, 0, rr-r-h+1) + max(c-cc, 0, cc-c-w+1)
                         for rr, cc in comp)
            objects.append(dict(color=color, cells=comp, arrival=min(ds),
                                last_contact=max(ds), direct=direct))
    _check(objects, 'the external object field must be visible')
    arrivals = [o['arrival'] for o in objects if o['arrival'] >= 0]
    _check(len(arrivals) == len(set(arrivals)), 'reachable objects need distinct arrival times')
    # A moat removes ambiguity between neighboring differently colored objects.
    labels = np.full(grid.shape, -1, int)
    for i, o in enumerate(objects):
        for cell in o['cells']:
            labels[cell] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(dr, 0):grid.shape[0]+min(dr, 0), max(dc, 0):grid.shape[1]+min(dc, 0)]
        b = labels[max(-dr, 0):grid.shape[0]+min(-dr, 0), max(-dc, 0):grid.shape[1]+min(-dc, 0)]
        _check(not np.any((a >= 0) & (b >= 0) & (a != b)), 'objects need a one-cell moat')
    reachable = sorted((o for o in objects if o['arrival'] >= 0), key=lambda o: o['arrival'])
    return dict(background=background, frame=frame, wall_color=wall_color, box=box,
                footprint=footprint, wall=wall, distance=distance, slots=slots,
                objects=objects, reachable=reachable, selected=reachable[:len(slots)])


def _write_answer(grid, detail):
    out = grid.copy()
    for slot, obj in zip(detail['slots'], detail['selected']):
        for cell in obj['cells']:
            out[cell] = detail['background']
        out[slot] = obj['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a whole-object arrival log from this input alone; never mutate it."""
    a = _grid(input_grid)
    return _write_answer(a, _inspect(a))


def _ring(wall, r0, c0, r1, c1):
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _geometry(rng, capacity, kind, shift=False):
    fh, fw = (5, 5) if capacity == 9 else (3, capacity+2)
    if kind == 'fold':
        h, w = (int(x) for x in rng.integers(22, 31, 2))
        r, c = int(rng.integers(4, 8)), int(rng.integers(4, 7))
        wall = np.zeros((h, w), bool)
        bottom = int(rng.integers(max(r+fh+4, h-7), h-2))
        wall[r-2, c-2:c+fw+2] = True
        wall[r-2:bottom+1, c-2] = True
        wall[r-2:bottom+1, c+fw+1] = True
        ph, pw = int(rng.integers(5, 8)), int(rng.integers(5, 8))
        _ring(wall, 0, w-pw, ph-1, w-1)
    else:
        h = int(rng.integers(12 if not shift else 20, 31))
        w = int(rng.integers(max(13 if not shift else 16, fw+4), 31))
        r, c = int(rng.integers(1, h-fh)), int(rng.integers(1, w-fw))
        wall = np.zeros((h, w), bool)
        if kind in ('segments', 'pocket', 'partitions'):
            number = int(rng.integers(2, 6 if not shift else 9))
            for _ in range(number):
                if rng.random() < .5:
                    rr = int(rng.integers(h))
                    c0 = int(rng.integers(w-2))
                    c1 = int(rng.integers(c0+2, w+1))
                    wall[rr, c0:c1] = True
                else:
                    cc = int(rng.integers(w))
                    r0 = int(rng.integers(h-2))
                    r1 = int(rng.integers(r0+2, h+1))
                    wall[r0:r1, cc] = True
        if kind == 'partitions':
            for _ in range(int(rng.integers(1, 4))):
                if rng.random() < .5:
                    rr = int(rng.integers(2, h-2))
                    wall[rr] = True
                    gap = int(rng.integers(1, w-2))
                    wall[rr, gap:gap+int(rng.integers(1, 4))] = False
                else:
                    cc = int(rng.integers(2, w-2))
                    wall[:, cc] = True
                    gap = int(rng.integers(1, h-2))
                    wall[gap:gap+int(rng.integers(1, 4)), cc] = False
        if kind == 'pocket':
            ph, pw = int(rng.integers(5, 8)), int(rng.integers(5, 8))
            _ring(wall, 0, w-pw, ph-1, w-1)
        if kind == 'enclosure':
            _ring(wall, max(0, r-2), max(0, c-2), min(h-1, r+fh+1), min(w-1, c+fw+1))
            # Either a closed collecting region or several open approaches.
            if rng.random() < .65:
                wall[max(0, r-2), c:c+int(rng.integers(1, fw+1))] = False
                wall[r:r+int(rng.integers(1, fh+1)), min(w-1, c+fw+1)] = False
    box = (r, c, fh, fw)
    wall[_footprint(wall.shape, box)] = False
    return wall, box


def _shape(rng, area=None, bar=False):
    area = int(rng.choice((1, 2, 3, 4, 5, 6, 7, 8, 10, 12))) if area is None else area
    kind = 0 if bar else int(rng.integers(4))
    if kind == 0:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        w = int(rng.integers(2, min(area, 4)+1)) if area > 1 else 1
        cells = {(i//w, i%w) for i in range(area)}
    elif kind == 2:
        arm = int(rng.integers(1, area+1))
        cells = {(0, c) for c in range(arm)} | {(r, 0) for r in range(area-arm+1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in STEPS}-cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _candidates(shape, free, distance, direct, active=True, used=()):
    sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
    h, w = free.shape
    if sh > h or sw > w:
        return []
    oh, ow = h-sh+1, w-sw+1
    allowed = np.ones((oh, ow), bool)
    arrival = np.full((oh, ow), h*w+1, int)
    last = np.full((oh, ow), -1, int)
    geometric = np.full((oh, ow), h*w+1, int)
    for dr, dc in shape:
        ds = distance[dr:dr+oh, dc:dc+ow]
        allowed &= free[dr:dr+oh, dc:dc+ow] & (ds >= 0 if active else ds < 0)
        arrival = np.minimum(arrival, ds)
        last = np.maximum(last, ds)
        geometric = np.minimum(geometric, direct[dr:dr+oh, dc:dc+ow])
    if active:
        allowed &= ~np.isin(arrival, list(used))
    return [(int(r), int(c), int(arrival[r, c]), int(last[r, c]), int(geometric[r, c]))
            for r, c in zip(*np.nonzero(allowed))]


def _install(free, placed, shape, choice):
    r, c = choice[:2]
    cells = tuple((r+dr, c+dc) for dr, dc in shape)
    placed.append(cells)
    h, w = free.shape
    for rr, cc in cells:
        free[max(0, rr-1):min(h, rr+2), max(0, cc-1):min(w, cc+2)] = False


def _arrange(rng, wall, box, count, unreachable, witness):
    footprint = _footprint(wall.shape, box)
    distance = _distances(wall, footprint)
    rr, cc = np.indices(wall.shape)
    r, c, h, w = box
    direct = np.maximum(np.maximum(r-rr, rr-r-h+1), 0) + np.maximum(np.maximum(c-cc, cc-c-w+1), 0)
    free = ~wall & ~footprint
    placed, used = [], set()
    shapes = [_shape(rng, area=int(rng.integers(1, 9))) if witness else _shape(rng)
              for _ in range(count)]
    if witness == 'contact':
        shapes[0] = _shape(rng, int(rng.integers(5, 9)), bar=True)
        shapes[1] = ((0, 0),)
        shapes[2] = _shape(rng, int(rng.integers(2, 5)))
        shapes[3] = _shape(rng, int(rng.integers(2, 9)))
        candidates = [x for x in _candidates(shapes[0], free, distance, direct)
                      if x[3]-x[2] >= 3]
        rng.shuffle(candidates)
        for choice in candidates[:150]:
            f = free.copy()
            tmp = []
            _install(f, tmp, shapes[0], choice)
            other = [x for x in _candidates(shapes[1], f, distance, direct, used=(choice[2],))
                     if choice[2] < x[2] < choice[3]]
            if other:
                _install(free, placed, shapes[0], choice)
                second = other[int(rng.integers(len(other)))]
                _install(free, placed, shapes[1], second)
                used.update((choice[2], second[2]))
                break
        else:
            return None
        start = 2
    elif witness == 'detour':
        capacity = (h-2)*(w-2)
        candidates = [x for x in _candidates(shapes[0], free, distance, direct)
                      if x[2]-x[4] >= 8 and x[4] <= 9]
        rng.shuffle(candidates)
        for choice in candidates[:150]:
            f, tmp = free.copy(), []
            _install(f, tmp, shapes[0], choice)
            near = [x for x in _candidates(shapes[1], f, distance, direct, used=(choice[2],))
                    if x[2] < choice[2] and x[4] > choice[4]]
            if near:
                _install(free, placed, shapes[0], choice)  # this object must remain
                selected = near[int(rng.integers(len(near)))]
                _install(free, placed, shapes[1], selected)
                used.update((choice[2], selected[2]))
                frontier = choice[2]
                break
        else:
            return None
        # Fill the earlier side of this strict frontier before later arrivals.
        for i in range(2, count):
            choices = _candidates(shapes[i], free, distance, direct, used=used)
            choices = [x for x in choices if (x[2] < frontier if i <= capacity else x[2] > frontier)]
            if not choices:
                return None
            choice = choices[int(rng.integers(len(choices)))]
            _install(free, placed, shapes[i], choice)
            used.add(choice[2])
        start = count
    else:
        start = 0
    for i in range(start, count):
        choices = _candidates(shapes[i], free, distance, direct, used=used)
        if not choices:
            return None
        choice = choices[int(rng.integers(len(choices)))]
        _install(free, placed, shapes[i], choice)
        used.add(choice[2])
    for _ in range(unreachable):
        shape = _shape(rng, int(rng.integers(1, 9)))
        choices = _candidates(shape, free, distance, direct, active=False)
        if not choices:
            return None
        _install(free, placed, shape, choices[int(rng.integers(len(choices)))])
    return placed


def _build_pair(rng, capacity, count, witness=None, shift=False):
    for attempt in range(300):
        kind = 'fold' if witness == 'detour' or attempt >= 230 else str(rng.choice(
            ('open', 'segments', 'partitions', 'pocket', 'enclosure'),
            p=(.22, .23, .20, .23, .12)))
        if count == 0:
            kind = 'pocket' if attempt < 230 else 'fold'
        wall, box = _geometry(rng, capacity, kind, shift)
        distance = _distances(wall, _footprint(wall.shape, box))
        inactive = (~wall) & (distance < 0)
        if witness == 'detour' and not inactive.any():
            continue
        unreachable = int(rng.integers(1, 4)) if witness == 'detour' or count == 0 else (
            int(rng.integers(4)) if inactive.any() else 0)
        placed = _arrange(rng, wall, box, count, unreachable, witness)
        if placed is None:
            continue
        palette = [int(x) for x in rng.permutation(10)]
        bg, wall_color, frame = palette[:3]
        ncolors = int(rng.integers(2, 6))
        payloads = palette[3:3+ncolors]
        object_colors = [int(rng.choice(payloads)) for _ in placed]
        if witness == 'contact':
            object_colors[:4] = [payloads[0], payloads[1], payloads[0], payloads[1]]
        if witness == 'detour':
            object_colors[0:2] = payloads[:2]
            # A held object may have the same color as a collected object.
            if len(placed) > 2:
                object_colors[2] = payloads[0]
        grid = np.full(wall.shape, bg, dtype=int)
        grid[wall] = wall_color
        for color, cells in zip(object_colors, placed):
            for cell in cells:
                grid[cell] = color
        r, c, h, w = box
        for cell in _frame_cells(box):
            grid[cell] = frame
        grid[r+h//2, c+w//2] = wall_color
        # Rotate INPUT geometry; construct the log in final displayed order.
        grid = np.rot90(grid, int(rng.integers(4))).copy()
        try:
            d = _inspect(grid)
        except AssertionError:
            continue
        if len(d['reachable']) != count or len(d['objects']) != count+unreachable:
            continue
        if witness == 'contact':
            actual = d['selected']
            alternative = sorted(actual, key=lambda o: o['last_contact'])
            if [o['color'] for o in actual] == [o['color'] for o in alternative]:
                continue
            if _source_order_matches(d):
                continue
            if not _interleaved(d):
                continue
        if witness == 'detour' and not _path_reversal(d):
            continue
        return {'input': grid, 'output': _write_answer(grid, d)}
    raise RuntimeError('bounded scene construction failed')


def _path_reversal(d):
    retained = d['reachable'][len(d['slots']):]
    return any(a['direct'] > b['direct'] for a in d['selected'] for b in retained)


def _source_order_matches(d):
    source = sorted(d['selected'], key=lambda o: min(o['cells']))
    return [o['color'] for o in source] == [o['color'] for o in d['selected']]


def _interleaved(d):
    colors = [o['color'] for o in d['selected']]
    return any(colors[i] == colors[k] != colors[j] for i in range(len(colors))
               for j in range(i+1, len(colors)) for k in range(j+1, len(colors)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train, num_test = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 training, 1..4 test, and num_train > num_test')
    rng = np.random.default_rng(seed)
    overflow_capacity = int(rng.choice((3, 5, 7)))
    novel_capacity = int(rng.choice([x for x in (3, 5, 7) if x != overflow_capacity]))
    train_specs = [(9, 4, 'contact'),
                   (overflow_capacity, overflow_capacity+int(rng.integers(1, 4)), 'detour')]
    for _ in range(num_train-2):
        cap = int(rng.choice([x for x in (3, 5, 7, 9) if x != novel_capacity]))
        load = int(rng.integers(cap+4))
        train_specs.append((cap, load, None))
    train_specs = [train_specs[int(i)] for i in rng.permutation(num_train)]
    test_specs = []
    for i in range(num_test):
        cap = novel_capacity if i == 0 else int(rng.choice((3, 5, 7, 9)))
        test_specs.append((cap, int(rng.integers(cap+4)), None))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in (('train', train_specs), ('test', test_specs)):
        for cap, load, witness in specs:
            for _ in range(8):
                pair = _build_pair(rng, cap, load, witness, shift=(split == 'test'))
                sig = (pair['input'].shape, pair['input'].tobytes())
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce a distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Audit returned pixels, including evidence obligations, without latent specs."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _check(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
           and len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    details, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _check(a.shape == b.shape, 'shape preservation')
            sig = (a.shape, a.astype(np.int8).tobytes())
            _check(sig not in seen, 'duplicate input')
            seen.add(sig)
            d = _inspect(a)
            details[split].append(d)
            _check(np.array_equal(b, _write_answer(a, d)), 'wrong arrival log or selective erasure')
            # Check conservation in object units and exact preservation outside writes.
            changes = np.zeros(a.shape, bool)
            for obj in d['selected']:
                for cell in obj['cells']:
                    changes[cell] = True
                    _check(b[cell] == d['background'], 'selected object must disappear whole')
            for slot, obj in zip(d['slots'], d['selected']):
                changes[slot] = True
                _check(b[slot] == obj['color'], 'one color token per object')
            _check(np.array_equal(a[~changes], b[~changes]), 'unselected content must be preserved')
            _check(len(d['selected']) == min(len(d['slots']), len(d['reachable'])), 'capacity conservation')
    train = details['train']
    _check(any(len(d['slots']) == 9 and len(d['selected']) == 4
               and not _source_order_matches(d) and _interleaved(d) for d in train),
           'training must reveal displayed row-major order and path rather than source scan order')
    _check(any(_path_reversal(d) and any(o['arrival'] < 0 for o in d['objects']) for d in train),
           'training must show excess, shortest-path selection and unreachable preservation')
    _check(any(len(o['cells']) >= 5 for d in train for o in d['selected']), 'whole-object compression witness')
    _check(any([o['color'] for o in d['selected']] !=
               [o['color'] for o in sorted(d['selected'], key=lambda o: o['last_contact'])]
               for d in train), 'training must distinguish first contact from complete arrival')
    _check(any(len({len(o['cells']) for o in d['selected'] if o['color'] == color}) > 1
               for d in train for color in {o['color'] for o in d['selected']}),
           'same color must have different object areas at equal slot cost')
    training_caps = {len(d['slots']) for d in train}
    _check(any(len(d['slots']) not in training_caps for d in details['test']), 'visible capacity shift')


def _signature(puzzle):
    return tuple((split, np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                  np.asarray(p['output']).tobytes())
                 for split in ('train', 'test') for p in puzzle[split])


def _metrics(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = np.asarray(pair['input'])
            d = _inspect(a)
            masks = []
            for o in d['objects']:
                r0, c0 = min(r for r, c in o['cells']), min(c for r, c in o['cells'])
                masks.append(tuple(sorted((r-r0, c-c0) for r, c in o['cells'])))
            records.append(dict(split=split, shape=a.shape, box=d['box'],
                palette=(d['background'], d['wall_color'], d['frame']),
                capacity=len(d['slots']), supply=len(d['reachable']), selected=len(d['selected']),
                unreachable=sum(o['arrival'] < 0 for o in d['objects']),
                excess=max(0, len(d['reachable'])-len(d['slots'])),
                max_arrival=max((o['arrival'] for o in d['objects']), default=-1),
                wall_cells=int(d['wall'].sum()), path_reversal=_path_reversal(d),
                areas=[len(o['cells']) for o in d['objects']], masks=masks,
                selected_colors=[o['color'] for o in d['selected']]))
    return records


def self_test():
    """Bounded 42-seed checks, meaningful mutations, and every legal count pair."""
    puzzles = [generate_puzzle(s) for s in HOST_SEEDS]
    _check(len({_signature(p) for p in puzzles}) == 42, 'cross-seed distinctness')
    records = [r for p in puzzles for r in _metrics(p)]
    _check(len({r['shape'] for r in records}) >= 40, 'canvas geometry diversity')
    _check({r['capacity'] for r in records} == {3, 5, 7, 9}, 'visible capacities')
    _check(len({r['box'] for r in records}) >= 70, 'register geometry and position diversity')
    _check(len({m for r in records for m in r['masks']}) >= 60, 'object geometry diversity')
    _check(len({r['supply'] for r in records}) >= 10, 'supply count diversity')
    _check(len({r['max_arrival'] for r in records}) >= 20, 'path-length diversity')
    _check(all(any(r['path_reversal'] for r in _metrics(p) if r['split'] == 'train')
               for p in puzzles), 'path evidence per puzzle')
    _check(any(r['wall_cells'] == 0 for r in records), 'open fields')
    _check(any(r['selected'] == 0 for r in records), 'zero collection')
    _check(any(r['capacity'] == 9 and r['excess'] > 0 for r in records), 'multirow saturation')
    for role in range(3):
        _check({r['palette'][role] for r in records} == set(range(10)), 'local role recoloring')
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                original = pair['input'].copy()
                _check(np.array_equal(solve(pair['input']), pair['output']), 'input-only solve')
                _check(np.array_equal(pair['input'], original), 'solver mutated input')
    _check(_signature(generate_puzzle(721, 7, 3)) == _signature(generate_puzzle(721, 7, 3)), 'reproducibility')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(0, a, b)
            _check(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (3, True)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Removing walls changes a critical training answer (B ablation).
    critical = next(p for p in puzzles[0]['train'] if _path_reversal(_inspect(p['input'])))
    d = _inspect(critical['input'])
    # Wall removal can create equal arrivals; compare object admissions directly.
    direct_order = sorted(d['objects'], key=lambda o: (o['direct'], min(o['cells'])))[:len(d['slots'])]
    _check({o['cells'] for o in direct_order} != {o['cells'] for o in d['selected']}, 'walls are essential')
    # Cell-wise transfer differs: selected objects erase more than one cell per slot (A ablation).
    partial = next(p for p in puzzles[0]['train'] if len(_inspect(p['input'])['selected']) == 4)
    pd = _inspect(partial['input'])
    _check(len(pd['slots']) == 9 and len(pd['selected']) == 4
           and sum(len(o['cells']) for o in pd['selected']) > len(pd['slots']),
           'object units are essential: four tokens erase more than nine cells')
    # A distance tie and a corrupted label must be rejected.
    tie = np.zeros((13, 13), int)
    box = (5, 4, 3, 5)
    for cell in _frame_cells(box):
        tie[cell] = 8
    tie[6, 6] = 5
    tie[3, 6], tie[9, 6] = 1, 2
    try:
        solve(tie)
    except AssertionError:
        pass
    else:
        raise AssertionError('arrival tie accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    changed_index = next(i for i, p in enumerate(bad['train'])
                         if not np.array_equal(p['input'], p['output']))
    bad['train'][changed_index]['output'] = bad['train'][changed_index]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), canvas_shapes=len({r['shape'] for r in records}),
                object_masks=len({m for r in records for m in r['masks']}),
                capacities=sorted({r['capacity'] for r in records}),
                loads=sorted({r['supply'] for r in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
