"""X-ray transfers across reconstructed rectangular membranes.

The scene constructor and input reader share geometric primitives, but the reader
never receives a seed, sampled palette, rectangle list, or output. Standard
library + NumPy only. See task_spec.md for the family contract.
"""
from collections import Counter
from itertools import combinations
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid of size 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r+1)} | {(b, x) for x in range(l, r+1)}
            | {(y, l) for y in range(t+1, b)} | {(y, r) for y in range(t+1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b+1) for x in range(l, r+1)}


def _bbox(points):
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _halo(points):
    return set(points) | {_add(p, d) for p in points for d in DIRECTIONS}


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            p = stack.pop()
            for d in DIRECTIONS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _chambers(shape, boxes):
    walls = set().union(*(_perimeter(b) for b in boxes)) if boxes else set()
    groups = _groups(set(np.ndindex(shape)) - walls)
    ids = np.full(shape, -1, dtype=np.int16)
    for i, cells in enumerate(groups):
        for p in cells:
            ids[p] = i
    return ids, groups, walls


def _stencil(center, direction):
    side = (-direction[1], direction[0])
    return {center, _add(center, side), _add(center, side, -1), _add(center, direction)}


def _read_arrows(grid, box, cover):
    """A complete cover surface consists solely of cover pixels and isolated Ts."""
    t, b, l, r = box
    interior = _area((t+1, b-1, l+1, r-1))
    marked = {p for p in interior if int(grid[p]) != cover}
    interpretations = []
    for ink in sorted({int(grid[p]) for p in marked}):
        arrows = []
        for center in sorted(marked):
            for d in DIRECTIONS:
                cells = _stencil(center, d)
                if not cells <= interior:
                    continue
                if not all(int(grid[p]) == ink for p in cells - {center}):
                    continue
                if int(grid[center]) == cover:
                    continue
                if any(int(grid[p]) != cover for p in (_halo(cells)-cells) & interior):
                    continue
                # Entire one-cell halo is on the cover, away from its frame.
                if not _halo(cells) <= interior:
                    continue
                arrows.append(dict(center=center, direction=d, cells=cells,
                                   explicit=int(grid[center]) != ink,
                                   token=int(grid[center])))
        occupied = set()
        valid = bool(arrows)
        for a in arrows:
            if occupied & a['cells']:
                valid = False
            occupied.update(a['cells'])
        if valid and occupied == marked:
            interpretations.append((ink, arrows))
    _need(len(interpretations) == 1, "cover arrows are not uniquely readable")
    return interpretations[0]


def _read_window(grid):
    answers = []
    for color in map(int, np.unique(grid)):
        points = set(map(tuple, np.argwhere(grid == color)))
        for cells in _groups(points):
            box = _bbox(cells)
            t, b, l, r = box
            if b-t < 6 or r-l < 6 or cells != _perimeter(box):
                continue
            values = grid[t+1:b, l+1:r]
            cover, frequency = Counter(map(int, values.flat)).most_common(1)[0]
            if cover == color or frequency <= values.size/2:
                continue
            try:
                ink, arrows = _read_arrows(grid, box, cover)
            except AssertionError:
                continue
            if color == ink or cover == ink:
                continue
            answers.append(dict(box=box, frame=color, cover=cover, ink=ink, arrows=arrows))
    _need(len(answers) == 1, "expected one framed, T-annotated opaque window")
    return answers[0]


def _complete_rectangles(points, masked):
    """Exact cover of visible fragments by separated rectangular membranes.

    All four extrema must be exposed. A single rectangular occluder splits a
    perimeter into at most two orthogonally connected visible pieces.
    """
    groups = _groups(points)
    if not groups or len(groups) > 12:
        return []
    candidates = []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            visible = groups[i] | groups[j]
            box = _bbox(visible)
            t, b, l, r = box
            if b-t < 4 or r-l < 4:
                continue
            full = _perimeter(box)
            if full-masked == visible:
                candidates.append((frozenset((i, j)), box, full))
    covers = []

    def search(remaining, boxes, occupied):
        if len(covers) >= 2:
            return
        if not remaining:
            covers.append(boxes)
            return
        first = min(remaining)
        for parts, box, full in candidates:
            if first in parts and parts <= remaining and not full & _halo(occupied):
                search(remaining-parts, boxes+[box], occupied | full)

    search(set(range(len(groups))), [], set())
    return covers


def _trace(ids, center, direction, visible_walls=None):
    """Cross exactly one membrane; corner/tangent hits are not valid ports."""
    source = int(ids[center])
    if source < 0:
        return None
    p = _add(center, direction)
    length = 1
    while _inside(p, ids.shape):
        hit = ids[p] < 0 if visible_walls is None else p in visible_walls
        if hit:
            q = _add(p, direction)
            if not _inside(q, ids.shape) or ids[q] < 0 or ids[q] == source:
                return None
            return dict(source=source, target=int(ids[q]), hit=p, distance=length)
        p = _add(p, direction)
        length += 1
    return None


def _render(ids, colors, wall, assignments, box):
    result = np.full(ids.shape, wall, dtype=np.int8)
    for i, color in enumerate(colors):
        result[ids == i] = assignments.get(i, color)
    t, b, l, r = box
    return result[t+1:b, l+1:r].copy()


def _decode(input_grid):
    grid = _grid(input_grid)
    window = _read_window(grid)
    box = window['box']
    masked = _area(box)
    exterior = set(np.ndindex(grid.shape)) - masked
    background = int(grid[0, 0])
    reserved = {window[k] for k in ('frame', 'cover', 'ink')}
    answers = []
    for wall in sorted(set(map(int, np.unique(grid))) - reserved - {background}):
        points = {p for p in exterior if int(grid[p]) == wall}
        for boxes in _complete_rectangles(points, masked):
            if not any(_perimeter(b) & masked for b in boxes):
                continue
            ids, groups, walls = _chambers(grid.shape, boxes)
            if len(groups) != len(boxes)+1:
                continue
            colors = []
            valid = True
            for cells in groups:
                seen = {int(grid[p]) for p in cells & exterior}
                if len(seen) != 1 or seen & (reserved | {wall}):
                    valid = False
                    break
                colors.append(seen.pop())
            if not valid:
                continue
            assignments, arrows = {}, []
            for a in window['arrows']:
                src = int(ids[a['center']])
                if src < 0 or any(ids[p] != src for p in a['cells']):
                    valid = False
                    break
                traced = _trace(ids, a['center'], a['direction'])
                if traced is None:
                    valid = False
                    break
                payload = a['token'] if a['explicit'] else colors[src]
                if payload in reserved | {wall}:
                    valid = False
                    break
                target = traced['target']
                if target in assignments and assignments[target] != payload:
                    valid = False
                    break
                assignments[target] = payload
                arrows.append(dict(a, **traced, payload=payload))
            if valid:
                output = _render(ids, colors, wall, assignments, box)
                answers.append((output, dict(window, ids=ids, groups=groups, walls=walls,
                                             boxes=boxes, wall=wall, colors=colors,
                                             assignments=assignments, arrows=arrows)))
    _need(len(answers) == 1, "membranes or original chamber fills are ambiguous")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the repainted X-ray window from the input alone."""
    return _decode(input_grid)[0]


def _random_box(rng, shape, minimum=5):
    h, w = shape
    hh = int(rng.integers(minimum, h-2))
    ww = int(rng.integers(minimum, w-2))
    t = int(rng.integers(1, h-hh))
    l = int(rng.integers(1, w-ww))
    return t, t+hh-1, l, l+ww-1


def _nested(a, b):
    t, z, l, r = a
    u, v, x, y = b
    return t < u < v < z and l < x < y < r


def _sample_geometry(rng, count, style, test, fallback=False):
    if fallback:
        shape, window = (30, 30), (8, 24, 8, 24)
        if style == 'disjoint':
            all_boxes = [(1, 17, 1, 17), (20, 28, 20, 28), (2, 11, 21, 28), (21, 28, 2, 11)]
        else:
            all_boxes = [(1, 18, 1, 18), (4, 14, 4, 14), (20, 28, 20, 28), (2, 11, 21, 28)]
        return shape, window, all_boxes[:count]
    shape = tuple(int(rng.integers(24 if test else 20, 31)) for _ in range(2))
    wh, ww = (int(rng.integers(11 if test else 8, min(19 if test else 16, n-5))) for n in shape)
    t, l = (int(rng.integers(2, n-size-1)) for n, size in zip(shape, (wh, ww)))
    window = t, t+wh-1, l, l+ww-1
    mask, inner = _area(window), _area((t+1, t+wh-2, l+1, l+ww-2))
    boxes, blocked = [], set()

    def accept(b):
        full = _perimeter(b)
        visible = full-mask
        if not visible or _bbox(visible) != b or len(full & inner) < 3 or full & blocked:
            return False
        if style == 'disjoint' and any(_nested(b, a) or _nested(a, b) for a in boxes):
            return False
        boxes.append(b)
        # The union's halo changes only when a rectangle is accepted. Reusing
        # it avoids rebuilding hundreds of neighborhood sets for rejections.
        blocked.update(_halo(full))
        return True

    for i in range(count):
        for _ in range(220):
            if i == 1 and style in ('nested', 'mixed'):
                a, z, x, y = boxes[0]
                if z-a < 10 or y-x < 10:
                    return None
                hh = int(rng.integers(5, z-a-4))
                ww2 = int(rng.integers(5, y-x-4))
                rr = int(rng.integers(a+3, z-hh-1))
                cc = int(rng.integers(x+3, y-ww2-1))
                b = rr, rr+hh-1, cc, cc+ww2-1
            else:
                b = _random_box(rng, shape, 11 if i == 0 and style in ('nested', 'mixed') else 5)
            if accept(b):
                break
        else:
            return None
    ids, groups, _ = _chambers(shape, boxes)
    if len(groups) != count+1 or any(len(cells-mask) < 4 for cells in groups):
        return None
    return shape, window, boxes


def _ports(ids, window):
    t, b, l, r = window
    interior = _area((t+1, b-1, l+1, r-1))
    visible_targets = Counter(int(ids[p]) for p in interior if ids[p] >= 0)
    result = []
    for center in sorted(interior):
        if ids[center] < 0:
            continue
        for d in DIRECTIONS:
            cells = _stencil(center, d)
            if not _halo(cells) <= interior or any(ids[p] != ids[center] for p in cells):
                continue
            traced = _trace(ids, center, d)
            if traced is not None and visible_targets[traced['target']] >= 2:
                result.append(dict(center=center, direction=d, cells=cells, **traced))
    return result


def _choose_arrows(rng, ports, colors, pool, wanted, demo):
    ports = [ports[int(i)] for i in rng.permutation(len(ports))]
    selected, blocked, assignments = [], set(), {}
    by_source, halos = {}, {}
    for p in ports:
        by_source.setdefault(p['source'], []).append(p)
        halos[(p['center'], p['direction'])] = _halo(p['cells'])

    def legal(p):
        return not p['cells'] & blocked

    def append(p, payload, explicit):
        target = p['target']
        if (not legal(p) or payload == colors[target]
                or (target in assignments and assignments[target] != payload)):
            return False
        selected.append(dict(p, payload=int(payload), explicit=bool(explicit)))
        assignments[target] = int(payload)
        blocked.update(halos[(p['center'], p['direction'])])
        return True

    # Minimal curricula provide a swap and an open dependency chain.
    if demo in (0, 1):
        pairs = []
        for a in ports:
            # This retains the original shuffled order of all viable pairs.
            for b in by_source.get(a['target'], ()):
                if a['cells'] & halos[(b['center'], b['direction'])]:
                    continue
                cycle = a['source'] == b['target']
                if (cycle != (demo == 0) or colors[a['source']] == colors[a['target']]
                        or colors[b['source']] == colors[b['target']]):
                    continue
                pairs.append((a, b))
        if not pairs:
            return None
        a, b = pairs[int(rng.integers(len(pairs)))]
        if not append(a, colors[a['source']], False) or not append(b, colors[b['source']], False):
            return None
        if demo == 1:
            # First island receives a visible singleton payload; this also
            # breaks color-wide substitution of the repeated island fill.
            choices = [p for p in ports if p['target'] == a['source'] and legal(p)]
            payloads = [c for c in pool if c not in colors]
            if not choices or not payloads:
                return None
            p = choices[int(rng.integers(len(choices)))]
            if not append(p, int(rng.choice(payloads)), True):
                return None
    for p in ports:
        if len(selected) >= wanted:
            break
        if not legal(p):
            continue
        if p['target'] in assignments:
            payload = assignments[p['target']]
            explicit = payload != colors[p['source']]
            if rng.random() > .2:
                continue
        else:
            explicit = demo != 0 and rng.random() < .35
            choices = [c for c in pool if c != colors[p['target']]]
            payload = int(rng.choice(choices)) if explicit else colors[p['source']]
        append(p, payload, explicit)
    if len(selected) < (3 if demo == 1 else 2 if demo == 0 else 1):
        return None
    return selected, assignments


def _features(info):
    arrows, colors = info['arrows'], info['colors']
    final = [info['assignments'].get(i, c) for i, c in enumerate(colors)]
    t, b, l, r = info['box']
    hidden_hits = sum(t < a['hit'][0] < b and l < a['hit'][1] < r for a in arrows)
    first_hit = False
    missing_wall_address = False
    visible_walls = info['walls'] - _area(info['box'])
    for a in arrows:
        incomplete = _trace(info['ids'], a['center'], a['direction'], visible_walls)
        if incomplete is None or incomplete['target'] != a['target']:
            missing_wall_address = True
        last_target = a['target']
        p = _add(a['hit'], a['direction'])
        while _inside(p, info['ids'].shape):
            if info['ids'][p] < 0:
                before = _add(p, a['direction'], -1)
                after = _add(p, a['direction'])
                if (_inside(after, info['ids'].shape)
                        and info['ids'][before] >= 0 and info['ids'][after] >= 0):
                    last_target = int(info['ids'][after])
            p = _add(p, a['direction'])
        if last_target != a['target']:
            first_hit = True
    return dict(membranes=len(info['boxes']), arrows=len(arrows),
                nested=sum(_nested(a, b) for a in info['boxes'] for b in info['boxes']),
                explicit=sum(a['explicit'] for a in arrows),
                singleton=sum(a['explicit'] and a['payload'] not in colors for a in arrows),
                cycle=any(not a['explicit'] and not b['explicit'] and a['source'] == b['target']
                          and a['target'] == b['source'] and a['payload'] != b['payload']
                          for a in arrows for b in arrows),
                chain=any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                          and a['source'] != b['target'] and a['payload'] != b['payload']
                          for a in arrows for b in arrows),
                locality=any(colors[a] == colors[b] and final[a] != final[b]
                             for a in range(len(colors)) for b in range(a)),
                hidden_hits=int(hidden_hits), first_hit=first_hit,
                missing_wall_address=missing_wall_address,
                distances=tuple(a['distance'] for a in arrows),
                directions=tuple(sorted({DIRECTIONS.index(a['direction']) for a in arrows})),
                duplicate_incoming=len(arrows) > len(info['assignments']),
                nonrectangular=sum(len(c) != len(_area(_bbox(c))) for c in info['groups']))


def _make_pair(rng, roles, count, style, test, demo):
    for attempt in range(100):
        geometry = _sample_geometry(rng, count, style, test, fallback=attempt >= 85)
        if geometry is None:
            continue
        shape, window, boxes = geometry
        ids, groups, walls = _chambers(shape, boxes)
        frame, cover, wall, ink = roles
        pool = [c for c in range(10) if c not in roles]
        colors = [int(c) for c in rng.choice(pool, len(groups), replace=False)]
        if demo == 1:
            # Two disjoint enclosed chambers share their original fill.
            islands = [i for i, cells in enumerate(groups) if (0, 0) not in cells]
            colors[islands[1]] = colors[islands[0]]
        ports = _ports(ids, window)
        wanted = 2 if demo == 0 else 3 if demo == 1 else int(rng.integers(3 if test else 1, 7 if test else 5))
        plan = _choose_arrows(rng, ports, colors, pool, wanted, demo)
        if plan is None:
            continue
        arrows, assignments = plan
        # Truth is rendered from the full scene, before any cover is applied.
        grid = np.full(shape, wall, dtype=np.int8)
        for i, color in enumerate(colors):
            grid[ids == i] = color
        t, b, l, r = window
        grid[t+1:b, l+1:r] = cover
        for p in _perimeter(window):
            grid[p] = frame
        for a in arrows:
            for p in a['cells']:
                grid[p] = ink
            if a['explicit']:
                grid[a['center']] = a['payload']
        output = _render(ids, colors, wall, assignments, window)
        # Rotate/reflect the complete input and output together. No turn cue:
        # this changes scene orientation, never the rule's output orientation.
        turns, flip = int(rng.integers(4)), bool(rng.integers(2))
        grid, output = np.rot90(grid, turns), np.rot90(output, turns)
        if flip:
            grid, output = np.fliplr(grid), np.fliplr(output)
        grid, output = grid.copy(), output.copy()
        try:
            predicted, info = _decode(grid)
        except AssertionError:
            continue
        f = _features(info)
        if (not np.array_equal(output, predicted) or not f['hidden_hits'] or not f['missing_wall_address']
                or (demo == 0 and not (f['cycle'] and f['first_hit'] and f['nested']))
                or (demo == 1 and not (f['chain'] and f['singleton'] and f['locality']
                                       and max(f['distances']) >= 4))
                or (test and f['arrows'] < 3)):
            continue
        return dict(input=grid, output=output)
    raise AssertionError("bounded readable-scene construction exhausted")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    nt, ne = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError('counts must satisfy 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed)
    shared_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    vary_roles = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', ne)):
        for i in range(n):
            test = split == 'test'
            demo = i if not test and i < 2 else -1
            count = 2 if demo >= 0 else int(rng.integers(3, 5) if test else rng.integers(1, 4))
            style = 'nested' if demo == 0 else 'disjoint' if demo == 1 else str(rng.choice(['disjoint', 'mixed'] if count >= 2 else ['disjoint']))
            roles = tuple(map(int, rng.choice(10, 4, replace=False))) if vary_roles else shared_roles
            for _ in range(8):
                pair = _make_pair(rng, roles, count, style, test, demo)
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('could not produce distinct inputs')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'incorrect reconstruction or simultaneous transfers')
            f = _features(info)
            _need(f['hidden_hits'] > 0, 'no arrow addresses a hidden membrane')
            _need(f['missing_wall_address'], 'hidden walls never change a transfer address')
            _need(1 <= f['membranes'] <= 4 and 1 <= f['arrows'] <= 6, 'family counts out of bounds')
            exterior = set(np.ndindex(inp.shape)) - _area(info['box'])
            _need(all(len(cells & exterior) >= 4 for cells in info['groups']), 'insufficient original-fill witnesses')
            baseline = _render(info['ids'], info['colors'], info['wall'], {}, info['box'])
            _need(not np.array_equal(baseline, out), 'omitting A does not change the answer')
            t, b, l, r = info['box']
            _need(np.any(info['ids'][t+1:b, l+1:r] < 0), 'omitting B loses no hidden wall')
            features[split].append(f)
    train = features['train']
    _need(any(f['cycle'] and f['nested'] and f['first_hit'] for f in train), 'missing first-hit/swap demonstration')
    _need(any(f['chain'] and f['singleton'] and f['locality'] and max(f['distances']) >= 4
              for f in train), 'missing explicit/locality/dependency/distant-ray demonstration')
    _need(all(f['membranes'] >= 3 and f['arrows'] >= 3 for f in features['test']), 'missing test structural shift')


def self_test():
    """Bounded generated-array, counterfactual, API and 42-seed variation checks."""
    puzzles, shapes, output_shapes, layouts, normalized = set(), set(), set(), set(), set()
    measured, directions, roles = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple((_key(p['input']), _key(p['output'])) for s in puzzle for p in puzzle[s])
        puzzles.add(fingerprint)
        if seed < 2:
            other = generate_puzzle(seed)
            _need(fingerprint == tuple((_key(p['input']), _key(p['output'])) for s in other for p in other[s]), 'not deterministic')
        for s in puzzle:
            for pair in puzzle[s]:
                inp = pair['input']
                prediction, info = _decode(inp.copy())
                _need(np.array_equal(prediction, pair['output']), 'input-only solve mismatch')
                f = _features(info)
                measured.append(f)
                directions.update(f['directions'])
                roles.add((info['frame'], info['cover'], info['wall'], info['ink']))
                shapes.add(inp.shape)
                output_shapes.add(pair['output'].shape)
                layouts.add((inp.shape, info['box'], tuple(sorted(info['boxes']))))
                renaming = {}
                normalized.add((inp.shape, tuple(renaming.setdefault(int(v), len(renaming)) for v in inp.flat)))
    _need(len(puzzles) == 42 and len(normalized) >= 180, 'collapsed structure variation')
    _need(len(layouts) >= 150 and len(shapes) >= 30 and len(output_shapes) >= 25, 'collapsed geometry variation')
    _need(directions == set(range(4)) and len(roles) >= 25, 'collapsed directions or palettes')
    _need({f['membranes'] for f in measured} >= {1, 2, 3, 4}, 'missing membrane counts')
    _need(len({f['arrows'] for f in measured}) >= 4, 'missing arrow count variation')
    _need(max(d for f in measured for d in f['distances']) >= 7, 'no distant spatial probes')
    _need(any(f['duplicate_incoming'] for f in measured), 'no consistent multiple incoming case')
    _need(any(f['nested'] == 0 for f in measured) and any(f['nested'] for f in measured), 'missing containment variation')
    # These checks perturb pixels or execution semantics, never sampled plans.
    probe = generate_puzzle(2026, 2, 1)
    permutation = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for s in probe:
        for pair in probe[s]:
            _need(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                  'color identities affect a geometric role')
    swap = next(p for p in probe['train'] if _features(_decode(p['input'])[1])['cycle'])
    _, info = _decode(swap['input'])
    for order in (info['arrows'], list(reversed(info['arrows']))):
        current = list(info['colors'])
        for a in order:
            current[a['target']] = a['token'] if a['explicit'] else current[a['source']]
        cascaded = _render(info['ids'], current, info['wall'], {}, info['box'])
        _need(not np.array_equal(cascaded, swap['output']), 'swap fails to distinguish cascading')
    # Contradict one of two consistent incoming instructions.
    incoming = None
    for seed in range(8):
        p = generate_puzzle(810+seed)
        for s in p:
            for pair in p[s]:
                _, i = _decode(pair['input'])
                if _features(i)['duplicate_incoming']:
                    incoming = pair['input'].copy(), i
                    break
            if incoming is not None:
                break
        if incoming is not None:
            break
    _need(incoming is not None, 'missing contradiction-test fixture')
    bad_input, i = incoming
    target = next(a['target'] for a in i['arrows']
                  if sum(b['target'] == a['target'] for b in i['arrows']) > 1)
    a = next(a for a in i['arrows'] if a['target'] == target)
    forbidden = {i['frame'], i['cover'], i['wall'], i['ink'], a['payload']}
    bad_input[a['center']] = next(c for c in range(10) if c not in forbidden)
    try:
        solve(bad_input)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory incoming arrows accepted')
    for nt, ne in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(420+nt+ne, nt, ne)
        _need(len(p['train']) == nt and len(p['test']) == ne, 'ignored counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(993, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
