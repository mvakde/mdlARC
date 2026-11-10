"""Two-ended seam census. Standard library and NumPy; no external state or I/O."""
from collections import Counter
import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError('Require integer counts 2..10 and 1..4, with train > test')
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(np.all((0 <= a) & (a <= 9)), 'ARC colors required')
    return a


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in DIRECTIONS:
                q = p[0]+dr, p[1]+dc
                if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and not seen[q]:
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _dilate(mask):
    h, w = mask.shape
    a = np.pad(mask, 1)
    return np.logical_or.reduce([a[dr:dr+h, dc:dc+w] for dr in range(3) for dc in range(3)])


def _path_geometry(shape, path):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for p in path:
        _require(0 <= p[0] < h and 0 <= p[1] < w, 'path outside grid')
        mask[p] = True
    _require(len(path) == int(mask.sum()) and len(path) >= 7, 'path repeats or too short')
    ends = []
    for p in path:
        degree = sum(0 <= p[0]+dr < h and 0 <= p[1]+dc < w and mask[p[0]+dr, p[1]+dc]
                     for dr, dc in DIRECTIONS)
        _require(degree in (1, 2), 'branch or isolated path cell')
        if degree == 1:
            ends.append(p)
    _require(set(ends) == {path[0], path[-1]}, 'path endpoints')
    _require(all(abs(p[0]-q[0])+abs(p[1]-q[1]) == 1 for p, q in zip(path, path[1:])), 'path order')
    boundary = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    _require(boundary(path[0]) and boundary(path[-1]), 'ports must be on border')
    _require(all(not boundary(p) for p in path[1:-1]), 'intermediate boundary contact')
    _require(all(p[0] not in (0, h-1) or p[1] not in (0, w-1) for p in ends), 'corner port')
    free = ~mask
    free[[0, -1], :] = False
    free[:, [0, -1]] = False
    cells = _components(free)
    _require(len(cells) == 2, 'path must divide interior into exactly two regions')
    regions = []
    for comp in cells:
        region = np.zeros(shape, dtype=bool)
        for p in comp:
            region[p] = True
        regions.append(region)
    return mask, regions


def _decode(input_grid):
    """Recover path, flag, categories and component counts from the input only."""
    g = _grid(input_grid)
    h, w = g.shape
    _require(min(h, w) >= 7, 'scene too small')
    neutral = int(g[0, 0])
    _require(all(int(g[p]) == neutral for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))), 'border corners')
    border = np.zeros(g.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    ports = [tuple(map(int, p)) for p in np.argwhere(border & (g != neutral))]
    _require(len(ports) == 2 and int(g[ports[0]]) != int(g[ports[1]]), 'two distinct colored ports')
    vals, freq = np.unique(g[1:-1, 1:-1], return_counts=True)
    bg = int(vals[int(np.argmax(freq))])
    _require(int(freq.max()) > (h-2)*(w-2)//2, 'background must occupy over half the interior')
    _require(bg != neutral and all(int(g[p]) not in (bg, neutral) for p in ports), 'distinct roles')
    interior_neutral = (g == neutral) & ~border
    comps = _components(interior_neutral)
    small = [c for c in comps if len(c) == 1]
    large = [c for c in comps if len(c) > 1]
    _require(len(small) == 1 and len(large) == 1, 'one isolated neutral flag cell and one path')
    marker = small[0][0]
    track = np.zeros(g.shape, dtype=bool)
    for p in large[0] + ports:
        track[p] = True
    # Trace the unique four-connected path from either endpoint.
    path, previous, current = [], None, ports[0]
    while current is not None:
        _require(current not in path, 'track loop')
        path.append(current)
        neighbors = [(current[0]+dr, current[1]+dc) for dr, dc in DIRECTIONS
                     if 0 <= current[0]+dr < h and 0 <= current[1]+dc < w
                     and track[current[0]+dr, current[1]+dc]
                     and (current[0]+dr, current[1]+dc) != previous]
        _require(len(neighbors) <= 1, 'track branch')
        previous, current = current, (neighbors[0] if neighbors else None)
    _require(len(path) == int(track.sum()) and path[-1] == ports[1], 'disconnected track')
    cut, regions = _path_geometry(g.shape, path)
    tags = [(marker[0]+dr, marker[1]+dc) for dr, dc in DIRECTIONS
            if int(g[marker[0]+dr, marker[1]+dc]) not in (bg, neutral)]
    _require(len(tags) == 1, 'flag must have one colored tag')
    tag = tags[0]
    marked_color = int(g[tag])
    _require(marked_color in {int(g[p]) for p in ports}, 'tag must match a port')
    other_color = next(int(g[p]) for p in ports if int(g[p]) != marked_color)
    side = next((i for i, region in enumerate(regions) if region[marker]), None)
    _require(side is not None and regions[side][tag], 'flag must lie wholly in one region')
    flag = np.zeros(g.shape, dtype=bool)
    flag[marker] = flag[tag] = True
    wall = border | cut
    _require(not np.any(flag & _dilate(wall)), 'flag touches border or track')
    occupied = wall | flag
    objects = []
    for color in sorted(set(map(int, g.flat)) - {bg, neutral}):
        for cells in _components((g == color) & ~border & ~flag):
            mask = np.zeros(g.shape, dtype=bool)
            for p in cells:
                mask[p] = True
            _require(not np.any(_dilate(mask) & occupied), 'objects touch frame, flag or another object')
            sides = [i for i, region in enumerate(regions) if all(region[p] for p in cells)]
            _require(len(sides) == 1, 'object crosses track')
            occupied |= mask
            objects.append({'color':color, 'side':sides[0], 'cells':cells})
    counts = (sum(o['color'] == marked_color and o['side'] == side for o in objects),
              sum(o['color'] == other_color and o['side'] != side for o in objects))
    capacity = (len(path)-2)//2
    _require(max(counts) <= capacity, 'count exceeds track capacity')
    if int(g[path[0]]) != marked_color:
        path.reverse()
    return dict(grid=g, background=bg, neutral=neutral, border=border, cut=cut,
                path=path, regions=regions, marker=marker, tag=tag, flag=flag,
                side=side, marked_color=marked_color, other_color=other_color,
                objects=objects, counts=counts, capacity=capacity)


def _render(d, counts=None, marked_priority=True):
    g, bg = d['grid'], d['background']
    out = np.full(g.shape, bg, dtype=np.int8)
    preserved = d['border'] | d['cut'] | d['flag']
    out[preserved] = g[preserved]
    nm, no = d['counts'] if counts is None else counts
    fronts = ((no, d['other_color'], list(reversed(d['path']))),
              (nm, d['marked_color'], d['path']))
    if not marked_priority:
        fronts = fronts[::-1]
    for n, color, path in fronts:
        for j in range(1, min(n, d['capacity'])+1):
            out[path[2*j]] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, cached examples, generation calls, or hidden scene parameters."""
    return _render(_decode(input_grid))


def _shape(rng, area):
    """Connected polyomino; geometry does not receive a category count."""
    cells = {(0, 0)}
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, j) for j in range(area)}
    elif kind == 1:
        width = int(rng.integers(1, min(3, area)+1))
        cells = {(j//width, j % width) for j in range(area)}
    else:
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRECTIONS} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _make_path(rng, kind):
    if kind == 'teaching':
        h, w = int(rng.choice((27, 29))), 9
        row = h//2
        return (h, w), [(row, c) for c in range(w)]
    if kind == 'straight':
        h, w = int(rng.integers(17, 28)), int(rng.integers(11, 24))
        c = int(rng.integers(5, w-5))
        return (h, w), [(r, c) for r in range(h)]
    if kind in ('elbow', 'teaching_elbow', 'test_elbow'):
        if kind == 'teaching_elbow':
            h, w = int(rng.integers(27, 30)), int(rng.integers(21, 30))
            r, c = int(rng.integers(19, h-5)), w-8
        elif kind == 'test_elbow':
            h, w = int(rng.integers(25, 30)), int(rng.integers(25, 30))
            r, c = int(rng.integers(5, 8)), int(rng.integers(6, 9))
        else:
            h, w = int(rng.integers(18, 28)), int(rng.integers(17, 28))
            r, c = int(rng.integers(7, h-6)), int(rng.integers(6, w-6))
        path = [(i, c) for i in range(r+1)] + [(r, j) for j in range(c+1, w)]
        return (h, w), path
    h, w = int(rng.integers(23, 29)), int(rng.integers(19, 29))
    rows = [h//2] if kind == 'step' else ([h//3] if kind == 'test_step' else [h//3, 2*h//3])
    c = int(rng.integers(6, w-6))
    path, last = [], 0
    for row in rows:
        path += [(r, c) for r in range(last, row+1)]
        options = [j for j in range(6, w-6) if abs(j-c) >= 2]
        nc = int(rng.choice(options))
        delta = 1 if nc > c else -1
        path += [(row, j) for j in range(c+delta, nc+delta, delta)]
        c, last = nc, row+1
    path += [(r, c) for r in range(last, h)]
    return (h, w), path


def _flag_sites(region, occupied):
    free = region & ~_dilate(occupied)
    sites = []
    h, w = region.shape
    for r, c in np.argwhere(free):
        p = int(r), int(c)
        for dr, dc in DIRECTIONS:
            q = p[0]+dr, p[1]+dc
            if 0 <= q[0] < h and 0 <= q[1] < w and free[q]:
                sites.append((p, q))
    return sites


def _scene(rng, shape, path, colors, specs):
    """Pack independently shaped components; retries never create disk payloads."""
    bg, neutral, mcolor, ocolor = colors[:4]
    cut, regions = _path_geometry(shape, path)
    h, w = shape
    wall = cut.copy()
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    for attempt in range(100):
        grid = np.full(shape, bg, dtype=np.int8)
        grid[wall] = neutral
        grid[path[0]], grid[path[-1]] = mcolor, ocolor
        occupied = wall.copy()
        objects = []
        success = True
        # Counts and forced teaching areas are fixed; geometry and packing retry.
        ordered = [(side, color, _shape(rng, area)) for side, color, area in specs]
        ordered.sort(key=lambda item: -len(item[2]))
        for side, color, obj in ordered:
            sh, sw = max(r for r, c in obj)+1, max(c for r, c in obj)+1
            free = regions[side] & ~_dilate(occupied)
            if sh > h or sw > w:
                success = False
                break
            valid = np.ones((h-sh+1, w-sw+1), dtype=bool)
            for dr, dc in obj:
                valid &= free[dr:dr+h-sh+1, dc:dc+w-sw+1]
            origins = np.argwhere(valid)
            if not len(origins):
                success = False
                break
            r0, c0 = map(int, origins[int(rng.integers(len(origins)))])
            cells = [(r0+r, c0+c) for r, c in obj]
            for p in cells:
                grid[p] = color
                occupied[p] = True
            objects.append((side, color, cells))
        if not success:
            continue
        sites = [_flag_sites(region, occupied) for region in regions]
        if not all(sites):
            continue
        flags = [sites[i][int(rng.integers(len(sites[i])))] for i in range(2)]
        # The background convention is independently checked after each flag.
        if np.count_nonzero(grid[1:-1, 1:-1] == bg) <= (h-2)*(w-2)//2+2:
            continue
        return grid, regions, flags, objects
    raise RuntimeError('bounded object packing exhausted')


def _pair_from_scene(scene, path, colors, side):
    base, regions, flags, objects = scene
    bg, neutral, mc, oc = colors[:4]
    marker, tag = flags[side]
    inp = base.copy()
    inp[marker], inp[tag] = neutral, mc
    out = np.full(inp.shape, bg, dtype=np.int8)
    out[[0, -1], :] = inp[[0, -1], :]
    out[:, [0, -1]] = inp[:, [0, -1]]
    for p in path:
        out[p] = inp[p]
    out[marker], out[tag] = neutral, mc
    # Ground truth from the packed object list, separately from input decoding.
    nm = sum(s == side and color == mc for s, color, cells in objects)
    no = sum(s != side and color == oc for s, color, cells in objects)
    for j in range(1, no+1):
        out[path[-1-2*j]] = oc
    for j in range(1, nm+1):
        out[path[2*j]] = mc
    return {'input':inp, 'output':out}


def _transform(pair, rotation, reflect):
    return {k: np.rot90(v[:, ::-1] if reflect else v, rotation).copy() for k, v in pair.items()}


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _first_bend(path):
    steps = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:])]
    return next((i for i, (a, b) in enumerate(zip(steps, steps[1:]), 1) if a != b), None)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)))
    palette_mode = int(rng.integers(3))
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    shape, path = _make_path(rng, 'teaching')
    if rng.random() < .5:
        path.reverse()
    # Exact twin witnesses: same color/area but different component count,
    # wrong-side distractors, marker dependence, and a real tally collision.
    a = int(rng.integers(1, 4))
    specs = [(0, colors[2], a), (0, colors[2], 4-a),
             (1, colors[2], 4),
             (0, colors[3], 1), (0, colors[3], 1), (0, colors[3], 2),
             (1, colors[3], 4)]
    scene = _scene(rng, shape, path, colors, specs)
    first = _transform(_pair_from_scene(scene, path, colors, 0), rotation, reflection)
    second = _transform(_pair_from_scene(scene, path, colors, 1), rotation, reflection)
    puzzle = {'train':[first, second], 'test':[]}
    seen = {_key(first['input']), _key(second['input'])}
    for index in range(2, nt+nq):
        train = index < nt
        # Minimum-count puzzles shift straight -> elbow. Larger training sets
        # first establish turns, then tests introduce/recombine multiple bends.
        if train:
            kind = 'teaching_elbow' if index == 2 else ('step' if index == 3 else str(rng.choice(('straight', 'elbow', 'step', 'double_step'))))
        else:
            kind = 'test_elbow' if nt == 2 else ('test_step' if nt == 3 else 'double_step')
        for retry in range(40):
            shape, path = _make_path(rng, kind)
            if kind not in ('teaching_elbow', 'test_elbow', 'test_step') and rng.random() < .5:
                path.reverse()
            cut, regions = _path_geometry(shape, path)
            capacity = (len(path)-2)//2
            if palette_mode == 0:
                local_colors = colors
            elif palette_mode == 1:
                local_colors = colors[:2] + tuple(map(int, rng.permutation([c for c in range(10) if c not in colors[:2]])))
            else:
                local_colors = tuple(map(int, rng.permutation(10)))
            bg, neutral, mc, oc, distractor = local_colors[:5]
            side = int(rng.integers(2))
            specs = []
            feasible = True
            for s, region in enumerate(regions):
                budget = min(8, max(2, int(region.sum())//18))
                maximum = min(5, capacity, budget)
                if train and index == 2:
                    n = 0 if s == side else 4
                    if n > maximum:
                        feasible = False
                        break
                elif not train and s == side:
                    # A visible M dot must pass the first bend. M precedence
                    # ensures this teaching/shift evidence cannot be occluded.
                    minimum = _first_bend(path)//2+1
                    if minimum > maximum:
                        feasible = False
                        break
                    n = int(rng.integers(minimum, maximum+1))
                else:
                    n = int(rng.integers(0, maximum+1))
                match = mc if s == side else oc
                specs += [(s, match, int(rng.integers(1, 8))) for _ in range(n)]
                remaining = budget-n
                wrong = oc if s == side else mc
                nw = int(rng.integers(0, min(3, remaining)+1))
                specs += [(s, wrong, int(rng.integers(1, 8))) for _ in range(nw)]
                remaining -= nw
                nd = int(rng.integers(0, min(2, remaining)+1))
                specs += [(s, int(rng.choice(local_colors[4:7])), int(rng.integers(1, 8))) for _ in range(nd)]
            if not feasible:
                continue
            if not specs:
                specs = [(1-side, distractor, 2)]
            try:
                scene = _scene(rng, shape, path, local_colors, specs)
            except RuntimeError:
                continue
            pair = _pair_from_scene(scene, path, local_colors, side)
            rot = rotation if palette_mode == 0 else int(rng.integers(4))
            pair = _transform(pair, rot, bool(rng.integers(2)))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError(f'bounded scene attempts exhausted: seed={seed}, counts={nt}/{nq}, index={index}, kind={kind}')
        seen.add(_key(pair['input']))
        puzzle['train' if train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(d):
    nm, no = d['counts']
    path = d['path']
    msites = {path[2*j] for j in range(1, nm+1)}
    osites = {path[-1-2*j] for j in range(1, no+1)}
    areas = tuple(sum(len(o['cells']) for o in d['objects']
                      if o['color'] == color and (o['side'] == d['side']) == marked)
                  for color, marked in ((d['marked_color'], True), (d['other_color'], False)))
    wrong = tuple(sum(o['color'] == color and (o['side'] == d['side']) != marked
                      for o in d['objects'])
                  for color, marked in ((d['marked_color'], True), (d['other_color'], False)))
    steps = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:])]
    return dict(counts=d['counts'], areas=areas, wrong=wrong,
                collision=bool(msites & osites),
                bends=sum(a != b for a, b in zip(steps, steps[1:])),
                path_length=len(path))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, details = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions changed')
            key = _key(inp)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            d = _decode(inp)
            _require(np.array_equal(out, _render(d)), 'wrong tally, priority, erasure or preserved cells')
            _require(not np.array_equal(inp, out), 'identity scene')
            if split == 'train':
                details.append(d)
    facts = [_facts(d) for d in details]
    _require(any(f['collision'] for f in facts), 'training needs a priority collision')
    _require(any(not f['collision'] and min(f['counts']) > 0 for f in facts), 'unoccluded two-sided teaching')
    _require({1, 2}.issubset({n for f in facts for n in f['counts']}), 'unary progression absent')
    _require(any(max(f['wrong']) > 0 for f in facts), 'regional distractors missing')
    if len(details) >= 3:
        _require(any(f['bends'] and d['counts'][1]*2 > _first_bend(d['path'][::-1])
                     and d['counts'][0] == 0 for d, f in zip(details, facts)),
                 'training needs an unoccluded tally around a turn')
    examples = [(color, count, area) for d, f in zip(details, facts)
                for color, count, area in zip((d['marked_color'], d['other_color']), f['counts'], f['areas'])]
    _require(any(c == cc and a == aa and n != nn for c, n, a in examples for cc, nn, aa in examples),
             'training permits a color-area shortcut')
    # An exact field/path twin isolates marker causality and side assignment.
    twin = False
    for i, a in enumerate(details):
        for b in details[i+1:]:
            if a['grid'].shape != b['grid'].shape or a['marked_color'] != b['marked_color']:
                continue
            ga, gb = a['grid'].copy(), b['grid'].copy()
            ga[a['flag']], gb[b['flag']] = a['background'], b['background']
            if (np.array_equal(ga, gb) and a['side'] != b['side']
                    and a['counts'] != b['counts']
                    and not np.array_equal(_render(a)[a['cut']], _render(b)[a['cut']])):
                twin = True
    _require(twin, 'marker-only teaching twin missing')


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streaming aggregates, transformations, ablations and corruption."""
    digests, shapes, paths, masks, backgrounds, ports = set(), set(), set(), set(), set(), set()
    bends, lengths, counts, areas = Counter(), set(), set(), set()
    collision_pairs = zero_pairs = wrong_pairs = unmatched_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                f = _facts(d)
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve mismatch')
                if split == 'test':
                    _require(d['counts'][0]*2 > _first_bend(d['path']), 'test never uses a turn')
                shapes.add(d['grid'].shape)
                paths.add((d['grid'].shape, d['cut'].tobytes()))
                backgrounds.add(d['background'])
                ports.add((d['marked_color'], d['other_color']))
                lengths.add(f['path_length'])
                counts.update(f['counts'])
                bends[f['bends']] += 1
                collision_pairs += f['collision']
                zero_pairs += 0 in f['counts']
                wrong_pairs += max(f['wrong']) > 0
                unmatched_pairs += any(o['color'] not in (d['marked_color'], d['other_color']) for o in d['objects'])
                for o in d['objects']:
                    areas.add(len(o['cells']))
                    r0 = min(p[0] for p in o['cells'])
                    c0 = min(p[1] for p in o['cells'])
                    masks.add(tuple(sorted((r-r0, c-c0) for r, c in o['cells'])))
        # Both inherited mechanisms have observable counterfactual consequences.
        first, second = puzzle['train'][:2]
        d = _decode(first['input'])
        total = tuple(sum(o['color'] == color for o in d['objects'])
                      for color in (d['marked_color'], d['other_color']))
        _require(not np.array_equal(_render(d, total), first['output']), 'omitting A accidentally works')
        _require(not np.array_equal(_render(d, _facts(d)['areas']), first['output']), 'area shortcut works')
        d = _decode(second['input'])
        _require(not np.array_equal(_render(d, marked_priority=False), second['output']), 'priority not exercised')
        for rotation in range(4):
            transformed = _transform(first, rotation, rotation % 2 == 1)
            _require(np.array_equal(solve(transformed['input']), transformed['output']), 'geometric equivariance')
        permutation = np.roll(np.arange(10), 1+seed % 9)
        _require(np.array_equal(solve(permutation[first['input']]), permutation[first['output']]), 'color-role equivariance')
        if seed == 0:
            repeat = generate_puzzle(seed)
            _require(_digest(repeat) == _digest(puzzle), 'nondeterminism')
            bad = {s:[{k:v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted a changed border')
            corrupt = first['input'].copy()
            d = _decode(corrupt)
            corrupt[d['tag']] = next(c for c in range(10) if c not in
                                     (d['background'], d['neutral'], d['marked_color'], d['other_color']))
            try:
                solve(corrupt)
            except AssertionError:
                pass
            else:
                raise AssertionError('solver accepted a tag without a matching port')
    _require(len(digests) == 42, 'cross-seed duplicates')
    _require(len(shapes) >= 35 and len(paths) >= 100, 'geometry collapsed')
    _require(len(masks) >= 70 and areas == set(range(1, 8)), 'object variation collapsed')
    _require(backgrounds == set(range(10)) and len(ports) >= 30, 'palette variation collapsed')
    _require({0, 1, 2, 4}.issubset(bends) and len(lengths) >= 15, 'path variation collapsed')
    _require(counts >= {0, 1, 2, 3, 4, 5}, 'count variation collapsed')
    _require(min(collision_pairs, zero_pairs, wrong_pairs, unmatched_pairs) >= 20, 'semantic coverage collapsed')
    # Broader valid members exercise the rule beyond sampler templates/counts:
    # eight bends and a U whose two endpoints share the same border side.
    long_path, last, col = [], 0, 8
    for row, next_col in ((5, 18), (10, 10), (15, 20), (20, 12)):
        long_path += [(r, col) for r in range(last, row+1)]
        delta = 1 if next_col > col else -1
        long_path += [(row, c) for c in range(col+delta, next_col+delta, delta)]
        last, col = row+1, next_col
    long_path += [(r, col) for r in range(last, 29)]
    u_path = ([(r, 8) for r in range(21)] + [(20, c) for c in range(9, 21)]
              + [(r, 20) for r in range(19, -1, -1)])
    rng = np.random.default_rng(20261022)
    colors = (0, 5, 2, 8, 3)
    specs = ([(0, 2, 1+i % 6) for i in range(6)]
             + [(1, 8, 1+i % 7) for i in range(7)] + [(1, 2, 4), (0, 3, 2)])
    for path in (long_path, u_path):
        scene = _scene(rng, (29, 29), path, colors, specs)
        pair = _pair_from_scene(scene, path, colors, 0)
        _require(_decode(pair['input'])['counts'] == (6, 7), 'broad count member')
        _require(np.array_equal(solve(pair['input']), pair['output']), 'broad path member')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(6000+nt*10+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'explicit counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds':42, 'pairs':210, 'distinct_puzzles':len(digests),
            'canvas_shapes':len(shapes), 'path_geometries':len(paths),
            'object_masks':len(masks), 'object_areas':sorted(areas),
            'path_lengths':sorted(lengths), 'bends':dict(sorted(bends.items())),
            'counts':sorted(counts), 'collision_pairs':collision_pairs,
            'zero_count_pairs':zero_pairs, 'wrong_side_pairs':wrong_pairs,
            'unmatched_color_pairs':unmatched_pairs,
            'verification':'generator-owned checks; no independent solver or blind prediction'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
