"""Hidden circuit docking. Standard library + NumPy; no file/network side effects.

8 = cable repair waypoint; 9 = direction request. All other color roles are
geometric and may change between pairs. The input reader never sees labels.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0,431340842,855562071,188622700,487605284,179270453,820943336,24167047,7101785,120982088,253547944,416222394,779774611,177021566,233101580,649578541,89173879,616179495,187373266,992788877,796143288,290132673,547622948,795356513,296852529,82530729,830251218,826582127,732449085,435855735,239123485,377496833,515906158,380937574,545546244,737350421,1191430,794758854,460439096,899700420,545824448,782610481)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neg(d):
    return -d[0], -d[1]


def _bbox(cells):
    ys, xs = zip(*cells)
    return min(ys), max(ys), min(xs), max(xs)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _inside(p, box):
    t, b, l, r = box
    return t < p[0] < b and l < p[1] < r


def _edge(p, box):
    t, b, l, r = box
    return p[0] in (t + 1, b - 1) or p[1] in (l + 1, r - 1)


def _halo(cells):
    return set(cells) | {_add(p, d) for p in cells for d in DIRS}


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + k*d[0], a[1] + k*d[1])
            for k in range(max(abs(b[0]-a[0]), abs(b[1]-a[1])) + 1)]


def _ray(p, d, box):
    result = []
    while _inside(p := _add(p, d), box):
        result.append(p)
    return result


def _points(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _scene(input_grid):
    """Parse geometry and repair waypoints exclusively from the input."""
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    assert all(int(grid[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    assert bg not in (8, 9)
    markers = _points(grid, 9)
    assert markers
    frames = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, 8, 9}):
        cells = _points(grid, color)
        box = _bbox(cells)
        if cells == _perimeter(box) and all(_inside(p, box) for p in markers):
            frames.append((box, color))
    assert len(frames) == 1, 'one complete enclosing frame required'
    box, frame = frames[0]
    t, b, l, r = box
    assert b - t >= 7 and r - l >= 7
    interior = grid[t+1:b, l+1:r]
    values, counts = np.unique(interior, return_counts=True)
    fill = int(values[int(np.argmax(counts))])
    assert fill not in (frame, 8, 9) and int(max(counts))*2 > interior.size
    loops, layers = {}, {}
    exterior = np.ones(grid.shape, dtype=bool)
    exterior[t:b+1, l:r+1] = False
    colors = sorted(set(map(int, np.unique(grid))) - {bg, frame, fill, 8, 9})
    occupied = set()
    for color in colors:
        cells = _points(grid, color)
        outside = {p for p in cells if exterior[p]}
        if outside:
            rect = _bbox(outside)
            perimeter = _perimeter(rect)
            visible = {p for p in perimeter if not (t <= p[0] <= b and l <= p[1] <= r)}
            hidden = {p for p in perimeter if _inside(p, box)}
            assert cells == outside == visible and hidden
            assert rect[1]-rect[0] >= 2 and rect[3]-rect[2] >= 2
            assert not perimeter & _halo(occupied), 'circuits cross or touch'
            occupied.update(perimeter)
            loops[color] = {'box': rect, 'hidden': hidden, 'full': perimeter}
        else:
            assert cells and all(_inside(p, box) for p in cells)
            layers[color] = set(cells)
    assert loops and layers
    repairs = _points(grid, 8)
    # All repair sites have a unique immediate cable-color witness, including
    # endpoints, corners and pivots. They need not have opposite neighbors.
    pending = set(repairs)
    for _ in range(len(pending) + 1):
        changed = False
        for p in sorted(pending):
            owners = [c for c, s in layers.items() if _neighbors(p, s)]
            if len(owners) == 1:
                layers[owners[0]].add(p)
                pending.remove(p)
                changed = True
        if not changed:
            break
    assert not pending, 'unresolved repair waypoint'
    # Unmarked gaps have equal immediate opposite cable witnesses. Corners
    # separated by empty space are not automatically joined.
    for color, cells in layers.items():
        original = set(cells)
        for p in sorted(original):
            for d in ((1, 0), (0, 1)):
                q, end = _add(p, d), _add(_add(p, d), d)
                if end in original and grid[q] == fill:
                    perp = (-d[1], d[0])
                    if all(any(_add(z, side) in original for side in (perp, _neg(perp)))
                           for z in (p, end)):
                        continue
                    cells.add(q)
        assert all(len(_neighbors(p, cells)) in (1, 2) for p in cells), 'cable branches or cycles'
        assert sum(len(_neighbors(p, cells)) == 1 for p in cells) == 2
        reached, stack = set(), [min(cells)]
        while stack:
            p = stack.pop()
            if p not in reached:
                reached.add(p)
                stack.extend(_neighbors(p, cells))
        assert reached == cells
    edits = []
    for q in sorted(markers):
        choices = []
        for d in DIRS:
            pivot = _add(q, _neg(d))
            start = _add(pivot, _neg(d))
            for color, cells in layers.items():
                if pivot not in cells or start not in cells or q in cells:
                    continue
                if len(_neighbors(pivot, cells)) != 2:
                    continue
                previous, current, walk = pivot, start, []
                for _ in range(grid.size):
                    if current in walk:
                        break
                    walk.append(current)
                    following = [p for p in _neighbors(current, cells) if p != previous]
                    if not following:
                        if _edge(current, box):
                            retained = cells - set(walk)
                            if any(_add(pivot, s) in retained for s in
                                   ((-d[1], d[0]), (d[1], -d[0]))):
                                choices.append({'color': color, 'pivot': pivot, 'direction': d,
                                                'old': set(walk), 'walk': [pivot] + walk})
                        break
                    if len(following) != 1:
                        break
                    previous, current = current, following[0]
        assert len(choices) == 1, 'marker must identify one terminal arm'
        edits.append(choices[0])
    assert len(edits) == len(layers) and len({e['color'] for e in edits}) == len(layers)
    for e in edits:
        assert not any(e['pivot'] in o['hidden'] or
                       _add(e['pivot'], e['direction']) in o['hidden']
                       for o in loops.values()), 'pivot/marker lies on a circuit'
        ray = _ray(e['pivot'], e['direction'], box)
        contacts = [(i, c) for i, p in enumerate(ray) for c, o in loops.items() if p in o['hidden']]
        hit = min(contacts) if contacts else None
        e['target'] = hit[1] if hit else None
        e['probe'] = ray[:hit[0]] if hit else ray
        e['contact'] = ray[hit[0]] if hit else None
        e['candidates'] = tuple(dict.fromkeys(c for _, c in sorted(contacts)))
        assert ray and ray[0] in markers
    # Cables are distinct simple components; probes have no ambiguous cable
    # collisions. Rectangle contacts are the only stopping mechanism.
    for i, e in enumerate(edits):
        own = layers[e['color']]
        assert not set(e['probe']) & own
        for other in edits[i+1:]:
            assert not _halo(own) & layers[other['color']]
            assert not set(e['probe']) & set(other['probe'])
        for c, s in layers.items():
            if c != e['color']:
                assert not set(e['probe']) & s
    assert any(e['target'] is not None for e in edits)
    return {'grid': grid, 'background': bg, 'fill': fill, 'frame': frame,
            'box': box, 'loops': loops, 'layers': layers, 'repairs': repairs, 'edits': edits}


def _render(scene, *, activate=True, redirect=True):
    t, b, l, r = scene['box']
    out = np.full((b-t-1, r-l-1), scene['fill'], dtype=np.int8)
    def paint(cells, color):
        for y, x in cells:
            out[y-t-1, x-l-1] = color
    activated = set()
    for e in scene['edits']:
        cells = scene['layers'][e['color']]
        paint(cells - e['old'] if redirect else cells, e['color'])
        if redirect:
            paint(e['probe'], e['color'])
            if e['target'] is not None:
                activated.add(e['target'])
    if activate:
        for color in sorted(activated):
            paint(scene['loops'][color]['hidden'], color)
    return out


def _boundary_ray_shortcut(scene):
    """Ablation of A: use B-like straight boundary rays with no circuit layer."""
    changed = []
    for edit in scene['edits']:
        changed.append(dict(edit, probe=_ray(edit['pivot'], edit['direction'], scene['box']), target=None))
    return _render(dict(scene, edits=changed), activate=False)


def _restore_all_shortcut(scene):
    """Unconditional reconstruction ignores activation and first-contact choice."""
    out = _render(scene, activate=False)
    t, _, l, _ = scene['box']
    for color, loop in scene['loops'].items():
        for y, x in loop['hidden']:
            out[y-t-1, x-l-1] = color
    return out


def _straight_deletion_shortcut(scene):
    edits = []
    for edit in scene['edits']:
        start_direction = _neg(edit['direction'])
        straight = set()
        previous = edit['pivot']
        for p in edit['walk'][1:]:
            if p != _add(previous, start_direction):
                break
            straight.add(p)
            previous = p
        edits.append(dict(edit, old=straight))
    return _render(dict(scene, edits=edits))


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_scene(input_grid))


def _int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _counts(num_train, num_test):
    a, b = _int(num_train, 'num_train'), _int(num_test, 'num_test')
    if not 2 <= a <= 10 or not 1 <= b <= 4 or a <= b:
        raise ValueError('require 2..10 train, 1..4 test, and train > test')
    return a, b


def _loop_geometry(rng, shape, box, count):
    h, w = shape
    mask = {(y, x) for y in range(box[0], box[1]+1) for x in range(box[2], box[3]+1)}
    boxes, used = [], set()
    for _ in range(800):
        ys = sorted(map(int, rng.choice(np.arange(1, h-1), 2, replace=False)))
        xs = sorted(map(int, rng.choice(np.arange(1, w-1), 2, replace=False)))
        rect = (ys[0], ys[1], xs[0], xs[1])
        if ys[1]-ys[0] < 3 or xs[1]-xs[0] < 3:
            continue
        cells = _perimeter(rect)
        visible = cells - mask
        hidden = {p for p in cells if _inside(p, box)}
        if len(hidden) < 4 or not visible or _bbox(visible) != rect or cells & _halo(used):
            continue
        boxes.append(rect)
        used.update(cells)
        if len(boxes) == count:
            return boxes
    return None


def _simple_extend(path, segment, forbidden):
    new = path + segment[1:]
    cells = set(new)
    return (len(cells) == len(new) and not set(segment[1:]) & forbidden and
            all(len(_neighbors(p, cells)) <= 2 for p in cells))


def _old_arm(rng, pivot, d, box, forbidden, bent):
    """Start opposite the marker; stop at an inner boundary after 0..3 turns."""
    path, direction = [pivot], _neg(d)
    desired = int(rng.integers(1 if bent else 0, 4))
    for turn in range(desired + 1):
        ray = _ray(path[-1], direction, box)
        if len(ray) < 2:
            return None
        if turn == desired:
            segment = [path[-1]] + ray
        else:
            lengths = list(range(2, len(ray)))
            if not lengths:
                return None
            length = lengths[int(rng.integers(len(lengths)))]
            segment = [path[-1]] + ray[:length]
        if not _simple_extend(path, segment, forbidden):
            return None
        path.extend(segment[1:])
        direction = ((-direction[1], direction[0]) if rng.integers(2)
                     else (direction[1], -direction[0]))
    return path


def _cables(rng, box, loops, count, double_witness, bent):
    hidden = [{p for p in _perimeter(rect) if _inside(p, box)} for rect in loops]
    choices = []
    for y in range(box[0]+3, box[1]-2):
        for x in range(box[2]+3, box[3]-2):
            p = (y, x)
            if any(p in s for s in hidden):
                continue
            for d in DIRS:
                ray = _ray(p, d, box)
                if len(ray) < 3 or any(ray[0] in s for s in hidden):
                    continue
                hits = [(i, k) for i, q in enumerate(ray) for k, s in enumerate(hidden) if q in s]
                ordered = tuple(dict.fromkeys(k for _, k in sorted(hits)))
                if ordered and (not double_witness or len(ordered) >= 2):
                    choices.append((p, d, ray[:min(hits)[0]], ordered))
    if not choices:
        return None
    rng.shuffle(choices)
    paths, specs, occupied, reserved = [], [], set(), set()
    for index in range(count):
        found = False
        for pivot, d, probe, targets in choices[:250]:
            q = _add(pivot, d)
            if {pivot, q} & _halo(occupied) or set(probe) & (occupied | reserved):
                continue
            forbidden = _halo(occupied) | reserved | set(_ray(pivot, d, box))
            for _ in range(10):
                old = _old_arm(rng, pivot, d, box, forbidden, bent and index == 0)
                if old is None:
                    continue
                perp = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
                arm_ray = _ray(pivot, perp, box)
                if len(arm_ray) < 3:
                    continue
                length = int(rng.integers(2, min(6, len(arm_ray)-1)+1))
                retained = [pivot] + arm_ray[:length]
                # Occasionally bend the retained arm independently.
                if rng.random() < .45:
                    turn = d if rng.integers(2) else _neg(d)
                    extension = _ray(retained[-1], turn, box)
                    maxlen = min(5, len(extension)-1)
                    if maxlen >= 2:
                        retained += extension[:int(rng.integers(2, maxlen+1))]
                path = list(reversed(old[1:])) + retained
                cells = set(path)
                if len(cells) != len(path) or any(len(_neighbors(p, cells)) > 2 for p in cells):
                    continue
                if cells & forbidden or set(probe) & cells:
                    continue
                if _edge(retained[-1], box) or len(_neighbors(pivot, cells)) != 2:
                    continue
                paths.append(path)
                specs.append({'pivot': pivot, 'direction': d, 'old': set(old[1:]),
                              'target': targets[0], 'probe': probe})
                occupied.update(cells)
                reserved.update(probe)
                found = True
                break
            if found:
                break
        if not found:
            return None
    return paths, specs


def _make_pair(rng, nloop, nwire, double_witness=False, bent=False, marked_pivot=False):
    for _ in range(240):
        h, w = map(int, rng.integers(20, 31, size=2))
        t = int(rng.integers(4, min(8, h-13)))
        l = int(rng.integers(4, min(8, w-13)))
        b = int(rng.integers(max(t+10, h-8), h-3))
        r = int(rng.integers(max(l+10, w-8), w-3))
        box = (t, b, l, r)
        rectangles = _loop_geometry(rng, (h, w), box, nloop)
        if rectangles is None:
            continue
        result = _cables(rng, box, rectangles, nwire, double_witness, bent)
        if result is None:
            continue
        paths, specs = result
        palette = list(map(int, rng.permutation(8)))
        bg, frame = palette[:2]
        distinct = bool(rng.integers(2))
        fill = palette[2] if distinct else bg
        remaining = palette[3:] if distinct else palette[2:]
        loop_colors = remaining[:nloop]
        wire_colors = remaining[nloop:nloop+nwire]
        if len(wire_colors) != nwire:
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        for rect, color in zip(rectangles, loop_colors):
            for p in _perimeter(rect):
                grid[p] = color
        grid[t+1:b, l+1:r] = fill
        for p in _perimeter(box):
            grid[p] = frame
        for path, spec, color in zip(paths, specs, wire_colors):
            for p in path:
                grid[p] = color
            grid[_add(spec['pivot'], spec['direction'])] = 9
        # Structural ground truth precedes corruption and is not read from solve.
        out = np.full((b-t-1, r-l-1), fill, dtype=np.int8)
        activated = set()
        for path, spec, color in zip(paths, specs, wire_colors):
            for y, x in (set(path)-spec['old']) | set(spec['probe']):
                out[y-t-1, x-l-1] = color
            activated.add(spec['target'])
        for k in activated:
            for y, x in _perimeter(rectangles[k]):
                if _inside((y, x), box):
                    out[y-t-1, x-l-1] = loop_colors[k]
        try:
            scene = _scene(grid)
            assert np.array_equal(_render(scene), out)
            if bent:
                assert not np.array_equal(_straight_deletion_shortcut(scene), out)
        except AssertionError:
            continue
        # At least one repair waypoint per pair. A pivot repair is guaranteed in
        # the first demonstration. Additional repairs cover retained/deleted bends.
        candidates = [(p, c) for path, c in zip(paths, wire_colors) for p in path]
        rng.shuffle(candidates)
        if marked_pivot:
            candidates = [(specs[0]['pivot'], wire_colors[0])] + candidates
        used = set()
        desired = int(rng.integers(1, 5))
        for p, color in candidates:
            if p in _halo(used):
                continue
            previous = int(grid[p])
            grid[p] = 8
            try:
                recovered = _scene(grid)
                assert recovered['layers'] == scene['layers']
                assert np.array_equal(_render(recovered), out)
            except AssertionError:
                grid[p] = previous
                continue
            used.add(p)
            if len(used) >= desired:
                break
        if not used or (marked_pivot and grid[specs[0]['pivot']] != 8):
            continue
        # At most two one-cell unmarked omissions; accept only exact recovery.
        for p, color in candidates[:12]:
            if len(_neighbors(p, scene['layers'][color])) != 2 or grid[p] == 8:
                continue
            if rng.random() >= .16:
                continue
            previous = int(grid[p])
            grid[p] = fill
            try:
                recovered = _scene(grid)
                assert recovered['layers'] == scene['layers']
                assert np.array_equal(_render(recovered), out)
            except AssertionError:
                grid[p] = previous
                continue
            if sum(grid[z] == fill for path in paths for z in path) >= 2:
                break
        # Palette-independent symmetry changes frame position and every direction.
        turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
        def transform(a):
            return np.rot90(np.fliplr(a) if reflect else a, turns).copy()
        return {'input': transform(grid), 'output': transform(out)}
    raise RuntimeError('bounded scene construction exhausted')


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    seed = _int(seed, 'seed')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train':
                nloop = 1 if index == 0 else 2
                nwire = 1
            else:
                nloop, nwire = 3, int(rng.integers(1, 3))
            for _ in range(5):
                pair = _make_pair(rng, nloop, nwire,
                                  double_witness=(split == 'train' and index == 1),
                                  bent=(split == 'train' and index == 0),
                                  marked_pivot=(split == 'train' and index == 0))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _turns(walk):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, scenes = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            scene = _scene(inp)
            scenes[split].append(scene)
            assert np.array_equal(_render(scene), out), 'incorrect docking/activation crop'
            assert np.array_equal(solve(inp.copy()), out)
            assert not np.any(np.isin(out, (8, 9)))
            # Each concept is necessary in every pair, not just somewhere in test.
            assert not np.array_equal(_render(scene, activate=False), out)
            assert not np.array_equal(_render(scene, redirect=False), out)
            assert not np.array_equal(_boundary_ray_shortcut(scene), out)
            # Reversing simultaneous edits must preserve the answer.
            reverse = dict(scene, edits=list(reversed(scene['edits'])))
            assert np.array_equal(_render(reverse), out)
    demonstrations = scenes['train']
    assert any(any(_turns(e['walk']) >= 1 for e in s['edits']) and
               any(e['pivot'] in s['repairs'] for e in s['edits']) for s in demonstrations)
    assert any(any(len(e['candidates']) >= 2 for e in s['edits']) for s in demonstrations)
    assert any(len(s['loops']) == 1 for s in demonstrations)
    assert any(len(s['loops']) >= 2 for s in demonstrations)
    assert any(not np.array_equal(_restore_all_shortcut(s), _render(s)) for s in demonstrations)
    assert any(not np.array_equal(_straight_deletion_shortcut(s), _render(s)) for s in demonstrations)


def self_test():
    """42 seeds, one puzzle at a time; scalar/hash diagnostics only, no writes."""
    shapes, layouts, palettes, directions, fingerprints = set(), set(), set(), set(), set()
    counts, old_bends, targets, nesting = set(), set(), set(), 0
    repaired_pivots = deleted_repairs = gaps = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(hashlib.sha256(repr(tuple(_key(p['input']) for s in puzzle.values() for p in s)).encode()).hexdigest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                scene = _scene(inp)
                shapes.add((inp.shape, out.shape))
                layouts.add((scene['box'], tuple(sorted(o['box'] for o in scene['loops'].values()))))
                palettes.add((scene['background'], scene['frame'], scene['fill'], tuple(scene['loops']), tuple(scene['layers'])))
                counts.add((len(scene['loops']), len(scene['layers'])))
                for a in scene['loops'].values():
                    for b in scene['loops'].values():
                        t, z, l, r = a['box']
                        u, v, x, y = b['box']
                        nesting += t < u < v < z and l < x < y < r
                for e in scene['edits']:
                    directions.add(e['direction'])
                    old_bends.add(_turns(e['walk']))
                    targets.add(len(e['candidates']))
                    repaired_pivots += e['pivot'] in scene['repairs']
                    deleted_repairs += len(e['old'] & scene['repairs'])
                gaps += sum(inp[p] == scene['fill'] for s in scene['layers'].values() for p in s)
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(_key(a[role]) == _key(b[role]) for s in puzzle
                       for a, b in zip(puzzle[s], again[s]) for role in ('input', 'output'))
            for role_point in ((0, 0), (puzzle['test'][0]['output'].shape[0]-1, 0)):
                bad = {s: [dict(p) for p in puzzle[s]] for s in puzzle}
                bad['test'][0]['output'] = bad['test'][0]['output'].copy()
                bad['test'][0]['output'][role_point] = (int(bad['test'][0]['output'][role_point])+1) % 10
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('corrupt output accepted')
    assert len(fingerprints) == 42 and len(shapes) >= 50 and len(layouts) >= 180
    assert len(palettes) >= 100 and directions == set(DIRS)
    assert {0, 1, 2} <= old_bends and {1, 2} <= targets
    assert {(1, 1), (2, 1), (3, 1), (3, 2)} <= counts
    assert repaired_pivots >= 42 and deleted_repairs and gaps and nesting
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1234+nt*10+nq, nt, nq))
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(4.0,1),(4,None),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(fingerprints), 'shape_pairs': len(shapes),
            'layouts': len(layouts), 'palettes': len(palettes), 'directions': len(directions),
            'old_bends': sorted(old_bends), 'loop_wire_counts': sorted(counts),
            'nested_pairs': nesting, 'repaired_pivots': repaired_pivots,
            'deleted_repairs': int(deleted_repairs), 'gaps': int(gaps)}
