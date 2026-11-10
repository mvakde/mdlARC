"""Rail-selected shape elections. Standard library and NumPy only.

The solver reads roles, routes, components, ballots and crop from its argument.
Generation labels are built from the sampled eligible objects, then checked by
that input-only reconstruction. No files are read or written by this module.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _norm(cells):
    cells = tuple(cells)
    y = min(r for r, c in cells)
    x = min(c for r, c in cells)
    return tuple(sorted((int(r-y), int(c-x)) for r, c in cells))


def _rot(s):
    return _norm((c, -r) for r, c in s)


def _mirror(s):
    return _norm((r, -c) for r, c in s)


def _rotations(s):
    found = set()
    for _ in range(4):
        s = _norm(s)
        found.add(s)
        s = _rot(s)
    return tuple(sorted(found))


def _variants(s):
    return tuple(sorted(set(_rotations(s)) | set(_rotations(_mirror(s)))))


def _canon(s):
    return min(_variants(_norm(s)))


def _extent(s):
    return max(r for r, c in s)+1, max(c for r, c in s)+1


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r+dr, c+dc
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'grid colors')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _plus(p, d):
    return p[0]+d[0], p[1]+d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _roles(inp):
    """The launch bank has two different guide-colored end caps."""
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        if min(g.shape) < 5:
            continue
        token, bg = int(g[1, 0]), int(g[0, 1])
        guides = (int(g[0, 0]), int(g[-1, 0]))
        if (len({token, bg, *guides}) == 4
                and np.all(g[1:-1, 0] == token)
                and np.count_nonzero(g == token) == g.shape[0]-2
                and np.all(g[:, 1] == bg)):
            candidates.append((g, k, bg, token, guides))
    _need(len(candidates) == 1, 'unique capped launch bank required')
    g, rotation, bg, token, guides = candidates[0]
    rectangles = []
    for color in set(map(int, np.unique(g))) - {bg, token, *guides}:
        for cells in _components(g == color):
            ys, xs = zip(*cells)
            y, x, b, r = min(ys), min(xs), max(ys), max(xs)
            box = g[y:b+1, x:r+1]
            if (min(box.shape) >= 4 and set(map(int, np.unique(box))) == {bg, color}
                    and np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)
                    and len(cells) == np.count_nonzero(g == color)):
                rectangles.append((color, (y, x, b, r)))
    _need(len(rectangles) == 1, 'unique connected rectangular stencil required')
    carrier, bbox = rectangles[0]
    return g, rotation, bg, token, guides, carrier, bbox


def _read(inp, *, ignore_matching=False, no_turns=False):
    g, rotation, bg, token, guides, carrier, bbox = _roles(_grid(inp))
    guide_set = set(guides)
    y, x, b, r = bbox
    box = g[y:b+1, x:r+1]
    cavities = _components(box == bg)
    _need(bool(cavities), 'empty stencil')
    ballot_colors = set(map(int, np.unique(g))) - {bg, token, carrier, *guides}
    objects, owner = [], {}
    for color in sorted(ballot_colors):
        for cells in _components(g == color):
            _need(all(not (y <= q[0] <= b and x <= q[1] <= r) for q in cells),
                  'ballot inside stencil')
            index = len(objects)
            objects.append(dict(color=color, cells=cells, shape=_canon(cells)))
            for q in cells:
                owner[q] = index

    def paired(p, d):
        n = -d[1], d[0]
        a, z = _plus(p, n), _plus(p, (-n[0], -n[1]))
        return (_inside(g, a) and _inside(g, z) and int(g[a]) in guide_set
                and int(g[z]) in guide_set
                and (ignore_matching or int(g[a]) == int(g[z])))

    routes, mismatches, eligible = [], [], set()
    for row in range(1, len(g)-1):
        entrance = None
        for col in range(1, g.shape[1]):
            if int(g[row, col]) != bg:
                break
            a, z = int(g[row-1, col]), int(g[row+1, col])
            if a in guide_set and z in guide_set:
                if a == z or ignore_matching:
                    entrance = (row, col)
                else:
                    mismatches.append((row, col))
                break
        if entrance is None:
            continue
        p, d, seen, path, turns = (row, 1), (0, 1), set(), [(row, 1)], 0
        vote = None
        while True:
            _need((p, d) not in seen, 'route cycle')
            seen.add((p, d))
            q = _plus(p, d)
            if not _inside(g, q):
                terminal_kind = 'edge'
                break
            if int(g[q]) == bg:
                p = q
                path.append(p)
                continue
            if q in owner:
                vote = owner[q]
                eligible.add(vote)
                terminal_kind = 'ballot'
                break
            choices = []
            if not no_turns and int(g[q]) in guide_set:
                for t in ((-d[1], d[0]), (d[1], -d[0])):
                    z = _plus(p, t)
                    if _inside(g, z) and int(g[z]) == bg and paired(z, t):
                        choices.append(t)
            _need(len(choices) <= 1, 'ambiguous turn')
            if not choices:
                terminal_kind = 'wall'
                break
            d = choices[0]
            turns += 1
        _need(len(path) == len(set(path)), 'route self-intersection')
        routes.append(dict(start=(row, 0), end=p, path=tuple(path), turns=turns,
                           vote=vote, terminal=terminal_kind))
    # Count components, not token contacts; a multiply contacted object is one vote.
    votes = {}
    for i in sorted(eligible):
        obj = objects[i]
        votes.setdefault(obj['shape'], Counter())[obj['color']] += 1
    return dict(grid=g, rotation=rotation, bg=bg, token=token, guides=guides,
                carrier=carrier, bbox=bbox, box=box, cavities=cavities,
                objects=objects, eligible=eligible, votes=votes,
                routes=routes, mismatches=mismatches)


def _fill(a, *, grouping='shape', all_objects=False):
    def signature(cells):
        if grouping == 'area':
            return len(cells)
        if grouping == 'bbox':
            return tuple(sorted(_extent(_norm(cells))))
        if grouping == 'rotation':
            return min(_rotations(_norm(cells)))
        return _canon(cells)
    votes = {}
    indices = range(len(a['objects'])) if all_objects else sorted(a['eligible'])
    for i in indices:
        obj = a['objects'][i]
        votes.setdefault(signature(obj['cells']), Counter())[obj['color']] += 1
    out = a['box'].copy()
    winners = {}
    for cells in a['cavities']:
        key = signature(cells)
        counts = votes.get(key, {})
        _need(bool(counts), 'unsupported cavity')
        high = max(counts.values())
        best = [c for c, n in counts.items() if n == high]
        _need(len(best) == 1, 'tied electorate')
        winners[key] = best[0]
        for p in cells:
            out[p] = best[0]
    return np.rot90(out, -a['rotation']).copy(), winners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the stencil, filled by the rail-reachable shape electorate."""
    return _fill(_read(input_grid))[0]


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _pool():
    """Small free polyominoes plus parameterized longer bars, Ls and Us."""
    shapes = set()
    coords = tuple((r, c) for r in range(3) for c in range(3))
    for size in range(3, 10):
        for cells in combinations(coords, size):
            mask = np.zeros((5, 5), dtype=bool)
            for r, c in cells:
                mask[r+1, c+1] = True
            if len(_components(mask)) == 1 and len(_components(~mask)) == 1:
                shapes.add(_canon(cells))
    for n in range(2, 6):
        shapes.add(_canon(tuple((0, c) for c in range(n))))
    for height in (2, 3):
        for width in range(3, 6):
            shapes.add(_canon({(r, 0) for r in range(height)} |
                              {(height-1, c) for c in range(width)}))
            shapes.add(_canon({(r, 0) for r in range(height)} |
                              {(r, width-1) for r in range(height)} |
                              {(height-1, c) for c in range(width)}))
    return tuple(sorted(shapes))


# Immutable geometric vocabulary; the solver never consults it.
SHAPES = _pool()


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = int(np.sign(target[0]-p[0])), int(np.sign(target[1]-p[1]))
        _need((d[0] == 0) != (d[1] == 0), 'orthogonal segment required')
        while p != target:
            p = _plus(p, d)
            path.append(p)
    _need(len(path) == len(set(path)), 'sampled path crosses itself')
    return path


def _module(rng, spec, width, kind, economical=False):
    """One ballot dock and its open-approach rail geometry, not a whole puzzle."""
    shape, color, status, handedness = spec
    variants = list(_variants(shape))
    if handedness == 'opposite':
        variants = [s for s in variants if s not in _rotations(shape)]
    elif handedness == 'same':
        variants = list(_rotations(shape))
    direction = (1, 0) if kind in (1, 3) else (0, -1) if kind == 2 and rng.random() < .4 else (0, 1)
    # Minimize dock height in the guaranteed construction. Ordinary choices
    # also retain taller orientations when there is room.
    options = []
    for s in variants:
        if direction == (0, 1):
            anchors = [p for p in s if p[1] == 0]
        elif direction == (0, -1):
            anchors = [p for p in s if p[1] == _extent(s)[1]-1]
        else:
            anchors = [p for p in s if p[0] == 0]
        for anchor in anchors:
            center = max(1, anchor[0]) if kind == 0 else max(3, anchor[0]+1)
            cost = max(center+1, center-anchor[0]+_extent(s)[0]-1)+2 if not direction[0] else _extent(s)[0]
            options.append((cost, s, anchor))
    if economical:
        low = min(v[0] for v in options)
        options = [v for v in options if v[0] == low]
    _, s, anchor = _pick(rng, options)
    margin = int(rng.integers(2, 5))
    if kind == 0:
        end = (max(1, anchor[0]), int(rng.integers(max(9, margin+5), width-3)))
        points = [(end[0], 1), end]
    elif kind == 1:
        col = int(rng.integers(7, width-3))
        points = [(1, 1), (1, col), (3, col)]
    elif kind == 2:
        row = max(3, anchor[0]+1) if economical else int(rng.integers(3, 6))
        if direction[1] == 1:
            turn = int(rng.integers(6, width-8))
            end_col = int(rng.integers(turn+3, width-3))
        else:
            turn = int(rng.integers(12, width-1))
            end_col = int(rng.integers(7, turn-2))
        points = [(1, 1), (1, turn), (row, turn), (row, end_col)]
    else:
        col1 = int(rng.integers(6, width-8))
        col2 = int(rng.integers(col1+3, width-3))
        points = [(1, 1), (1, col1), (3, col1), (3, col2), (5, col2)]
    path = _polyline(points)
    end = path[-1]
    target = _plus(end, direction)
    origin = target[0]-anchor[0], target[1]-anchor[1]
    body = tuple((origin[0]+r, origin[1]+c) for r, c in s)
    _need(min(r for r, c in body) >= 0 and min(c for r, c in body) >= 3
          and max(c for r, c in body) < width, 'dock outside rail field')
    _need(not set(body).intersection(path), 'ballot overlaps route')
    walls = set()
    path_set, body_set = set(path), set(body)
    for p in path:
        for d in DIRS:
            q = _plus(p, d)
            if q[0] >= 0 and margin <= q[1] < width and q not in path_set and q not in body_set:
                walls.add(q)
    # Inertia: remove one short aligned section of first-leg flanks. Bends,
    # first entrance support and the forward blocker remain intact.
    gap = 0
    first_turn = points[1][1]
    if first_turn >= margin+5 and rng.random() < .7:
        begin = int(rng.integers(margin+2, first_turn-2))
        gap = int(rng.integers(1, min(3, first_turn-1-begin)+1))
        for col in range(begin, begin+gap):
            walls.discard((path[0][0]-1, col))
            walls.discard((path[0][0]+1, col))
    cap = None
    if status == 'blocked':
        _need(kind == 0, 'blocked dock is straight')
        cap = end[0], end[1]-1
        walls.add(cap)
    height = max(r for r, c in walls | body_set | path_set)+2
    return dict(height=height, path=path, walls=walls, body=body, color=color,
                status=status, kind=kind, gap=gap, cap=cap)


def _stencil(rng, shapes, winners, bg, carrier, height, width, anchors):
    """Irregular separated cavities; mandatory shapes precede repetitions."""
    for _ in range(60):
        sh = int(rng.integers(max(7, 3*len(shapes)+2), height+1))
        box = np.full((sh, width), carrier, dtype=np.int8)
        occupied = np.zeros_like(box, dtype=bool)
        occupied[[0, -1], :] = True
        occupied[:, [0, -1]] = True
        placements = []
        todo = list(shapes) + [_pick(rng, shapes) for _ in range(int(rng.integers(0, 8)))]
        failed = False
        for index, shape in enumerate(todo):
            variants = [shape] if shape in anchors and index < len(shapes) else _variants(shape)
            candidates = []
            for variant in variants:
                h, w = _extent(variant)
                for y in range(1, sh-h):
                    for x in range(1, width-w):
                        cells = tuple((y+r, x+c) for r, c in variant)
                        if all(not occupied[p] for p in cells):
                            candidates.append(cells)
            if not candidates:
                if index < len(shapes):
                    failed = True
                break
            cells = _pick(rng, candidates)
            placements.append((shape, cells))
            for r, c in cells:
                box[r, c] = bg
                occupied[max(0, r-1):r+2, max(0, c-1):c+2] = True
        if failed or len(_components(box == carrier)) != 1:
            continue
        out = box.copy()
        for shape, cells in placements:
            for p in cells:
                out[p] = winners[shape]
        return box, out
    raise ValueError('bounded stencil packing exhausted')


def _make_pair(rng, shapes, teaching, is_test, forbidden_winner=None):
    roles = list(map(int, rng.permutation(10)))
    bg, token, ga, gb, carrier = roles[:5]
    palette = roles[5:]
    if forbidden_winner is not None and palette[0] == forbidden_winner:
        palette[0], palette[1] = palette[1], palette[0]
    # In the teaching examples the companion winner must differ from both
    # main-class losing colors. Then pooling equal-area/equal-box shapes gives
    # a definite wrong winner, rather than merely making a tied election.
    winners = {s: palette[0 if i == 0 else i+2] for i, s in enumerate(shapes)}
    # Elections are per shape. Reusing a winning color is legal outside the
    # mandatory exact-shape demonstrations.
    if teaching is None and len(shapes) > 1 and rng.random() < .25:
        winners[shapes[-1]] = winners[shapes[0]]
    pattern = (2, 1, 1) if teaching == 0 else _pick(rng, ((2, 1), (2, 1, 1), (3, 1), (3, 2)))
    if teaching == 1 or sum(pattern)+len(shapes)-1+(pattern[0]-pattern[1]+1) > 7:
        pattern = (2, 1)
    specs = []
    main = shapes[0]
    losers = [c for c in palette if c != winners[main]]
    for j, n in enumerate(pattern):
        color = winners[main] if j == 0 else losers[j-1]
        hand = ('opposite' if j == 0 else 'same') if teaching is not None else None
        specs.extend((main, color, 'active', hand) for _ in range(n))
    for s in shapes[1:]:
        specs.append((s, winners[s], 'active', None))
    lure_count = pattern[0]-pattern[1]+1
    for j in range(lure_count):
        status = 'blocked' if j == 0 and teaching == 1 else 'mismatch'
        if teaching is None and j == 0 and rng.random() < .5:
            status = 'blocked'
        specs.append((main, losers[0], status, None))
    # At least one bend matters in the second demonstration: its sole vote
    # for the companion shape cannot be recovered by a straight-only tracer.
    mandatory = next((i for i, v in enumerate(specs) if len(shapes) > 1 and v[0] == shapes[1]), 0)
    for attempt in range(100):
        economical = attempt >= 35
        rail_width = int(rng.integers(18, 22))
        sw = int(rng.integers(6, min(10, 30-rail_width)))
        modules = []
        try:
            for i, spec in enumerate(specs):
                kind = 0
                if spec[2] == 'active':
                    if teaching == 1 and i == mandatory:
                        kind = 2
                    elif teaching is None and (not economical) and rng.random() < (.6 if is_test else .3):
                        kind = int(rng.integers(1, 4))
                modules.append(_module(rng, spec, rail_width, kind, economical))
        except ValueError:
            continue
        used = sum(m['height'] for m in modules)
        if used > 30:
            continue
        h = min(30, used+int(rng.integers(0, 4)))
        # Modest extra slack is assigned at irregular locations, not a fixed pitch.
        slack = h-used
        gaps = [0]*(len(modules)+1)
        for _ in range(slack):
            gaps[int(rng.integers(len(gaps)))] += 1
        order = list(map(int, rng.permutation(len(modules))))
        inp = np.full((h, rail_width+1+sw), bg, dtype=np.int8)
        inp[1:-1, 0] = token
        inp[0, 0], inp[-1, 0] = ga, gb
        cursor, previous_bottom = gaps[0], None
        expected_selected, intended = [], {}
        for serial, i in enumerate(order):
            m = modules[i]
            # Adjacent bands may face one another across a blank row. Their
            # different guide identities make such accidental openings inert.
            top = _pick(rng, (ga, gb)) if previous_bottom is None else (gb if previous_bottom == ga else ga)
            bottom = (gb if top == ga else ga) if m['status'] == 'mismatch' else top
            for r, c in m['walls']:
                inp[cursor+r, c] = top if r <= m['path'][0][0] else bottom
            for r, c in m['body']:
                inp[cursor+r, c] = m['color']
            start = (cursor+m['path'][0][0], 0)
            if m['status'] == 'active':
                intended[start] = (cursor+m['path'][-1][0], m['path'][-1][1])
                expected_selected.append(frozenset((cursor+r, c) for r, c in m['body']))
            elif m['status'] == 'blocked':
                intended[start] = (cursor+m['cap'][0], m['cap'][1]-1)
            previous_bottom = bottom
            cursor += m['height']+gaps[serial+1]
        try:
            box, out = _stencil(rng, shapes, winners, bg, carrier, h, sw,
                                 {main} if teaching is not None else set())
            y = int(rng.integers(h-len(box)+1))
            inp[y:y+len(box), rail_width+1:] = box
            a = _read(inp)
            actual = {v['start']: v['end'] for v in a['routes']}
            selected = {frozenset(a['objects'][i]['cells']) for i in a['eligible']}
            if actual != intended or selected != set(expected_selected):
                continue
            if not np.array_equal(_fill(a)[0], out):
                continue
        except ValueError:
            continue
        rotation = int(rng.integers(4))
        pair = {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        return pair
    raise ValueError('bounded rail/dock construction exhausted')


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'illegal {name}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # These small asymmetric anchors guarantee cheap, legible discriminating
    # examples even at the minimum 2/1 count. Other pairs use the full pool.
    anchor = _canon(((0, 0), (0, 1), (1, 1), (1, 2)))
    companion = _canon(((0, 0), (0, 1), (0, 2), (1, 0)))
    if rng.random() < .5:
        anchor, companion = companion, anchor
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    first_winner = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            teaching = index if split == 'train' and index < 2 else None
            if teaching is not None:
                shapes = [anchor, companion]
            else:
                k = int(rng.integers(1, 4)) if split == 'train' else int(rng.integers(2, 4))
                pool = list(SHAPES)
                shapes = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
                if split == 'test':
                    novel = [s for s in pool if s not in known]
                    shapes[0] = _pick(rng, novel)
                    shapes = list(dict.fromkeys(shapes))
            for _ in range(12):
                pair = _make_pair(rng, shapes, teaching, split == 'test', first_winner if teaching == 1 else None)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise ValueError('unique-input search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                known.update(shapes)
            if teaching == 0:
                first_winner = _fill(_read(pair['input']))[1][anchor]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_fails(pair, mode):
    try:
        a = _read(pair['input'], ignore_matching=mode == 'ignore_matching', no_turns=mode == 'no_turns')
        prediction = _fill(a, grouping=mode, all_objects=mode == 'all_objects')[0]
    except ValueError:
        return True
    return not np.array_equal(prediction, pair['output'])


def _alternative_is_wrong(pair, mode):
    """A stronger witness: the shortcut returns a unique but wrong answer."""
    try:
        a = _read(pair['input'], ignore_matching=mode == 'ignore_matching')
        prediction = _fill(a, grouping=mode, all_objects=mode == 'all_objects')[0]
    except ValueError:
        return False
    return not np.array_equal(prediction, pair['output'])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            a = _read(inp)
            prediction, winners = _fill(a)
            _need(np.array_equal(prediction, out), 'wrong election fill or crop')
            _need(bool(a['routes']) and bool(a['mismatches']), 'routing selection absent')
            _need(0 < len(a['eligible']) < len(a['objects']), 'selected and unselected objects required')
            cavity_keys = {_canon(c) for c in a['cavities']}
            _need(set(a['votes']) == cavity_keys, 'all selected shapes need cavities')
            _need(all(obj['shape'] in cavity_keys for obj in a['objects']), 'unexplained ballot shape')
            paths = [p for route in a['routes'] for p in route['path']]
            _need(len(paths) == len(set(paths)), 'generated routes overlap')
            _need(len({v['end'] for v in a['routes']}) == len(a['routes']), 'terminal collision')
            # Carrier, border, cavity support, connectivity and exact object
            # components have been recovered from the input, not latent specs.
            canonical_out = np.rot90(out, a['rotation'])
            mask = a['box'] == a['carrier']
            _need(np.array_equal(canonical_out[mask], a['box'][mask]), 'carrier changed')
            _need(_alternative_fails(pair, 'all_objects'), 'routing cannot be discarded')
            analyses[split].append(a)
    for mode in ('area', 'bbox', 'rotation', 'ignore_matching', 'no_turns'):
        _need(any(_alternative_fails(p, mode) for p in puzzle['train']), f'training does not refute {mode}')
    for mode in ('area', 'bbox', 'all_objects', 'ignore_matching'):
        _need(any(_alternative_is_wrong(p, mode) for p in puzzle['train']),
              f'training needs a definite wrong answer under {mode}')
    _need(any(max(v.values())*2 <= sum(v.values()) and len(v) >= 3
              for a in analyses['train'] for v in a['votes'].values()), 'plurality evidence absent')
    _need(any(v['terminal'] == 'wall' for a in analyses['train'] for v in a['routes']), 'closed-stop evidence absent')
    history = {}
    for a in analyses['train']:
        for shape, winner in _fill(a)[1].items():
            history.setdefault(shape, set()).add(winner)
    _need(any(len(colors) > 1 for colors in history.values()), 'fixed color dictionary not refuted')
    known = set().union(*({_canon(c) for c in a['cavities']} for a in analyses['train']))
    _need(all({_canon(c) for c in a['cavities']} - known for a in analyses['test']), 'test must introduce a supported new shape')


def self_test():
    """Stream 42 seeds; retain aggregate features, never a puzzle archive."""
    dimensions, output_shapes, vocab, palettes, structures = set(), set(), set(), set(), set()
    counts, turns, ballots, signatures = set(), set(), set(), set()
    repeated, gaps = 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _need(np.array_equal(solve(inp), out), 'solver disagreement')
                _need(np.array_equal(inp, before), 'solver mutated its input')
                a = _read(inp)
                dimensions.add(inp.shape)
                output_shapes.add(out.shape)
                vocab.update(_canon(c) for c in a['cavities'])
                palettes.add((a['bg'], a['token'], a['carrier'], a['guides']))
                counts.add((len(a['objects']), len(a['eligible']), len(a['cavities'])))
                turns.update(v['turns'] for v in a['routes'])
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in a['votes'].values())
                repeated += len(a['cavities']) > len(a['votes'])
                # Color-normalized structures are descriptive diversity only.
                mapping = {}
                flat = bytes(mapping.setdefault(int(c), len(mapping)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(flat).digest()))
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                # Measure gaps only on the reconstructed initial straight
                # segment, with actual support before and after the gap.
                g, guide_set = a['grid'], set(a['guides'])
                for route in a['routes']:
                    r = route['start'][0]
                    first_leg = []
                    for row, col in route['path']:
                        if row != r:
                            break
                        if col >= 2:
                            first_leg.append(col)
                    support = [g[r-1, c] == g[r+1, c] and int(g[r-1, c]) in guide_set
                               for c in first_leg]
                    if True in support:
                        lo, hi = support.index(True), len(support)-1-support[::-1].index(True)
                        gaps += any(not v for v in support[lo:hi+1])
        signatures.add(digest.digest())
    _need(len(signatures) == 42, 'cross-seed duplicate puzzle')
    _need(len(dimensions) >= 25 and len(output_shapes) >= 30, 'geometry diversity')
    _need(len(vocab) >= 25 and len(palettes) >= 100, 'shape or role diversity')
    _need(len(counts) >= 20 and repeated >= 50, 'count or repetition diversity')
    _need({0, 1, 2} <= turns and gaps > 20, 'route diversity')
    _need({(1,), (2, 1), (2, 1, 1)} <= ballots, 'ballot diversity')
    _need(len(structures) >= 200, 'structural diversity')
    p = generate_puzzle(12345, 2, 1)
    q = generate_puzzle(12345, 2, 1)
    _need(all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s])
              for k in ('input', 'output')), 'determinism')
    sample = p['train'][0]
    a = _read(sample['input'])
    for rotation in range(4):
        _need(np.array_equal(solve(np.rot90(sample['input'], rotation)),
                             np.rot90(sample['output'], rotation)), 'rotation covariance')
    permutation = np.random.default_rng(917).permutation(10)
    _need(np.array_equal(solve(permutation[sample['input']]), permutation[sample['output']]),
          'literal color dependence')
    # An unreachable object's color has no electoral power.
    untouched = a['grid'].copy()
    obj = next(v for i, v in enumerate(a['objects']) if i not in a['eligible'])
    other_color = next(v['color'] for v in a['objects'] if v['color'] != obj['color'])
    for cell in obj['cells']:
        untouched[cell] = other_color
    _need(np.array_equal(solve(untouched), np.rot90(sample['output'], a['rotation'])),
          'unreachable object affected election')
    # One eligible winner changing allegiance flips precisely its shape class.
    changed = a['grid'].copy()
    main = next(shape for shape, votes in a['votes'].items() if len(votes) == 3)
    counts_main = a['votes'][main]
    winner = max(counts_main, key=counts_main.get)
    loser = next(c for c in counts_main if c != winner)
    obj = next(a['objects'][i] for i in sorted(a['eligible'])
               if a['objects'][i]['shape'] == main and a['objects'][i]['color'] == winner)
    for cell in obj['cells']:
        changed[cell] = loser
    expected = np.rot90(sample['output'], a['rotation']).copy()
    for cells in a['cavities']:
        if _canon(cells) == main:
            for cell in cells:
                expected[cell] = loser
    _need(np.array_equal(solve(changed), expected), 'eligible vote did not flip its class')
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            generate_puzzle(123, n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=210, input_dimensions=len(dimensions), output_dimensions=len(output_shapes),
                shape_classes=len(vocab), role_assignments=len(palettes), count_profiles=len(counts),
                turn_counts=sorted(turns), ballot_patterns=[list(v) for v in sorted(ballots)],
                repeated_cavity_pairs=repeated, interrupted_route_observations=gaps,
                color_normalized_structures=len(structures))
