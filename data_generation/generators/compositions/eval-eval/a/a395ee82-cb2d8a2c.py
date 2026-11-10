"""Anchored stencil fences with conditional unary detours.

Only standard-library modules and NumPy are used.  The sparse diagram is an
instruction map, not the place where the ray travels.  The ray travels through
the actual expanded masks, including their open channels.
"""

import math
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(a, color):
    remaining = set(map(tuple, np.argwhere(a == color)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    stack.append(q)
        result.append(component)
    return result


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _decode(input_grid):
    """Identify roles, coordinates and commands entirely from the input."""
    original = _grid(input_grid)
    values, counts = np.unique(original, return_counts=True)
    _require(len(values) == 5 and np.count_nonzero(counts == counts.max()) == 1,
             "five roles and a unique dominant background are required")
    bg = int(values[counts.argmax()])
    seed_candidates = []
    for v in values:
        if v == bg:
            continue
        pts = np.argwhere(original == v)
        if len(pts) == 1:
            r, c = map(int, pts[0])
            edges = (r == 0, c == 0, r == original.shape[0]-1, c == original.shape[1]-1)
            if sum(edges) == 1:
                seed_candidates.append((int(v), edges.index(True)))
    _require(len(seed_candidates) == 1, "one noncorner boundary seed is required")
    route, rotation = seed_candidates[0]
    # A top seed rotated counterclockwise k times points down/right/up/left.
    a = np.rot90(original, -rotation).copy()
    candidates = []
    for color in values:
        if color in (bg, route):
            continue
        comps = _components(a, int(color))
        singles = [c[0] for c in comps if len(c) == 1]
        big = [p for comp in comps if len(comp) > 1 for p in comp]
        if len(singles) == 1 and len(big) >= 3:
            candidates.append((int(color), singles[0], big))
    _require(len(candidates) == 1, "template and same-color map anchor are ambiguous")
    ink, anchor_pixel, glyph = candidates[0]
    coords = np.array(glyph, dtype=int)
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    mask = a[tr:br+1, tc:bc+1] == ink
    h, w = mask.shape
    _require(2 <= h <= 4 and 2 <= w <= 5, "unsupported stencil dimensions")
    _require(np.all((a[tr:br+1, tc:bc+1] == bg) | mask), "template contains foreign cells")
    others = [int(v) for v in values if v not in (bg, route, ink)]
    for color in others:
        _require(all(len(comp) == 1 for comp in _components(a, color)), "map sites must be isolated")
    points = [tuple(map(int, p)) for p in np.argwhere(np.isin(a, others))]
    points.append(tuple(map(int, anchor_pixel)))
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    _require(br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc,
             "template and map need a blank separating row or column")
    _require(all(0 < r < a.shape[0]-1 and 0 < c < a.shape[1]-1 for r, c in points),
             "the instruction map must be interior")
    _require(all((r-pr) % 2 == 0 and (c-pc) % 2 == 0 for r, c in points), "map pitch is two")
    sites = {(r-pr)//2: {} for r, c in points}
    for r, c in points:
        sites[(r-pr)//2][(c-pc)//2] = int(a[r, c])
    anchor = ((int(anchor_pixel[0])-pr)//2, (int(anchor_pixel[1])-pc)//2)
    assignments = []
    for marker in others:
        ordinary = next(v for v in others if v != marker)
        bars = []
        valid = True
        for row, cells in sorted(sites.items()):
            lo, hi = min(cells), max(cells)
            marks = tuple(c for c in sorted(cells) if cells[c] == marker)
            n = len(marks)
            low = marks == tuple(lo+2*j for j in range(n))
            high = marks == tuple(hi-2*j for j in reversed(range(n)))
            # Two consecutive ordinary sites at the unmarked tail make the
            # instruction role and marked end readable without training.
            if not (len(cells) == hi-lo+1 and 1 <= n <= 4 and hi-lo+1 >= 2*n+1
                    and low != high and all(v in (ordinary, marker, ink) for v in cells.values())
                    and any(cells[c] == cells[c+1] == ordinary for c in range(lo, hi))):
                valid = False
                break
            bars.append((row, lo, hi, n, 1 if low else -1))
        if valid:
            assignments.append((ordinary, marker, bars))
    _require(len(assignments) == 1, "endpoint marker convention is ambiguous")
    ordinary, marker, bars = assignments[0]
    anchor_bar = next(b for b in bars if b[0] == anchor[0])
    anchor_side = anchor_bar[4]
    _require(anchor[1] not in tuple(anchor_bar[1]+2*j if anchor_side == 1 else anchor_bar[2]-2*j
                                 for j in range(anchor_bar[3])), "anchor cannot replace an instruction")
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    initial = int(np.argwhere(a == route)[0, 1])
    return dict(a=a, bg=bg, route=route, ink=ink, ordinary=ordinary, marker=marker,
                mask=mask, h=h, w=w, anchor=anchor, anchor_side=anchor_side,
                top=top, left=left, bars=bars, initial=initial, rotation=rotation,
                template=(tr, tc), map_box=(pr, pc, er, ec))


def _walls(m):
    out = np.full_like(m['a'], m['bg'])
    h, w = m['h'], m['w']
    previous_bottom = -1
    for row, lo, hi, n, side in m['bars']:
        r = m['top']+row*h
        c0, c1 = m['left']+lo*w, m['left']+(hi+1)*w
        _require(0 <= r and r+h <= out.shape[0] and 0 <= c0 < c1 <= out.shape[1],
                 "expanded stencil fence is out of bounds")
        _require(r-n-1 >= 1 and r-n-1 > previous_bottom,
                 "every potential turn must be past the preceding fence")
        previous_bottom = r+h-1
        tile = m['mask'] if side == m['anchor_side'] else np.fliplr(m['mask'])
        for col in range(lo, hi+1):
            c = m['left']+col*w
            color = m['ordinary'] if (row, col) == m['anchor'] else m['ink']
            out[r:r+h, c:c+w][tile] = color
    return out


def _trace(m, walls, solid=False, fixed_clearance=None):
    """Trace in physical pixels, checking the stencil, not its bounding box."""
    path = np.zeros(walls.shape, dtype=bool)
    x, start = m['initial'], 0
    hits, targets, phases = [], [], []
    for row, lo, hi, n, side in m['bars']:
        r = m['top']+row*m['h']
        low = m['left']+lo*m['w']
        high = m['left']+(hi+1)*m['w']-1
        _require(low <= x <= high, "each fence must intercept the incoming bounding-box ray")
        phases.append((x-m['left']) % m['w'])
        hit = solid or bool(np.any(walls[r:r+m['h'], x] != m['bg']))
        hits.append(hit)
        if hit:
            d = n+1 if fixed_clearance is None else fixed_clearance
            turn = r-d
            target = high+d if side == 1 else low-d
            _require(0 <= target < walls.shape[1] and turn >= start and turn >= 1,
                     "commanded detour is out of bounds")
            path[start:turn+1, x] = True
            path[turn, min(x, target):max(x, target)+1] = True
            x, start = target, turn
            targets.append(target)
        else:
            targets.append(None)
    path[start:, x] = True
    if not solid and fixed_clearance is None:
        _require(not np.any(path & (walls != m['bg'])), "route intersects an occupied stencil cell")
    return path, tuple(hits), tuple(targets), tuple(phases)


def solve(input_grid: np.ndarray) -> np.ndarray:
    m = _decode(input_grid)
    out = _walls(m)
    path = _trace(m, out)[0]
    out[path] = m['route']
    return np.rot90(out, m['rotation']).copy()


def _stencil(rng, h, w, hole=None, informative=False):
    mask = np.zeros((h, w), dtype=bool)
    if hole is not None:
        occupied = [c for c in range(w) if c != hole]
    else:
        occupied = [0, w-1] + [c for c in range(1, w-1) if rng.random() < 0.65]
    for c in sorted(set(occupied)):
        length = int(rng.integers(2, h+1))
        start = int(rng.integers(h-length+1))
        mask[start:start+length, c] = True
    full = _pick(rng, (0, w-1))
    mask[:, full] = True  # tight height; both side rails are visible
    if informative and h >= 3:
        partial = w-1-full
        mask[:, partial] = False
        mask[:2, partial] = True  # visibly rules out a forward-axis reflection
    elif informative and hole is None:
        # A dense corner notch keeps this short glyph connected while making
        # the opposite reference handedness observable even at height two.
        mask[:] = True
        mask[_pick(rng, ((0, 0), (0, w-1), (1, 0), (1, w-1)))] = False
    return mask


def _sample_plan(rng, mask, ns, anchor_side, width, left, mixed=False):
    """Small finite search over visible fence spans and arrival phases."""
    w = mask.shape[1]
    cols = (width-left)//w
    failed = set()

    def search(i, x, flags):
        if i == len(ns):
            return [] if not mixed or flags == (True, False, True) else None
        state = (i, x, flags)
        if state in failed:
            return None
        tilecol = (x-left)//w
        options = []
        if not 0 <= tilecol < cols:
            return None
        n, d = ns[i], ns[i]+1
        for side in (-1, 1):
            tile = mask if side == anchor_side else np.fliplr(mask)
            hit = bool(tile[:, (x-left) % w].any())
            if mixed and hit != (True, False, True)[i]:
                continue
            for lo in range(tilecol+1):
                for hi in range(max(tilecol, lo+2*n), cols):
                    target = left+(hi+1)*w-1+d if side == 1 else left+lo*w-d
                    if 0 <= target < width:
                        options.append((lo, hi, n, side, hit, target if hit else x))
        for j in rng.permutation(len(options)):
            option = options[int(j)]
            rest = search(i+1, option[5], flags+(option[4],))
            if rest is not None:
                return [option]+rest
        failed.add(state)
        return None

    for initial in rng.permutation(np.arange(max(1, left), width-1)):
        plan = search(0, int(initial), ())
        if plan is not None and any(p[3] == anchor_side for p in plan):
            return int(initial), plan
    return None


def _make_pair(rng, h, w, ns, rotation, colors, mixed=False, previous_anchor=None,
               reference_side=None, informative=False):
    anchor_side = int(rng.choice((-1, 1))) if reference_side is None else reference_side
    # An off-center slit and its mirror give distinct transparent phases.
    hole = (w-2 if anchor_side == 1 else 1) if mixed else None
    mask = _stencil(rng, h, w, hole, informative)
    widths = list(map(int, rng.permutation(np.arange((2*max(ns)+1)*w+max(ns)+1, 31))))
    _require(bool(widths), "requested stencil/count combination cannot fit")
    plan = None
    for width in widths:
        left = int(rng.integers(min(3, width-(2*max(ns)+1)*w-max(ns)-1)+1))
        result = _sample_plan(rng, mask, ns, anchor_side, width, left, mixed)
        if result is not None:
            initial, plan = result
            break
    _require(plan is not None, "no feasible fence sequence")
    # Geometry leaves room for every potential turn, even a skipped command.
    rows = [0]
    for n in ns[1:]:
        rows.append(rows[-1]+math.ceil((h+n+1)/h))
    minimum_top = ns[0]+2
    minimum_height = max(minimum_top+rows[-1]*h+h+1, 2*rows[-1]+3)
    _require(minimum_height <= 30, "requested fence sequence is too tall")
    height = int(rng.integers(minimum_height, 31))
    slack = height-minimum_height
    top = minimum_top+int(rng.integers(min(slack, 4)+1))
    # Add whole-cell vertical gaps only when both the compact map and expansion fit.
    if len(rows) > 1 and rng.random() < 0.5:
        j = int(rng.integers(1, len(rows)))
        if top+(rows[-1]+1)*h+h < height and 2*(rows[-1]+1)+3 <= height:
            rows[j:] = [r+1 for r in rows[j:]]
    min_col = min(p[0] for p in plan)
    choices = []
    for row, (lo, hi, n, side, hit, target) in zip(rows, plan):
        if side != anchor_side:
            continue
        marks = {lo+2*j if side == 1 else hi-2*j for j in range(n)}
        for c in range(lo, hi+1):
            norm = (row, c-min_col)
            if c not in marks and (previous_anchor is None or
                                   (norm[0] != previous_anchor[0] and norm[1] != previous_anchor[1])) and any(
                    j not in marks and j+1 not in marks and c not in (j, j+1) for j in range(lo, hi)):
                choices.append((row, c))
    _require(bool(choices), "no contrasting ordinary anchor site")
    rng.shuffle(choices)
    bg, ink, ordinary, marker, route = colors
    map_height, map_width = 2*rows[-1]+1, 2*(max(p[1] for p in plan)-min_col)+1
    placements = []
    for ar, ac in choices:
        tr, tc = top+ar*h, left+ac*w
        for pr in range(1, height-map_height):
            for pc in range(1, width-map_width):
                if (tr+h < pr or pr+map_height < tr or tc+w < pc or pc+map_width < tc):
                    placements.append((ar, ac, tr, tc, pr, pc))
    _require(bool(placements), "no separated interior map placement")
    ar, ac, tr, tc, pr, pc = _pick(rng, placements)
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[tr:tr+h, tc:tc+w][mask] = ink
    inp[0, initial] = route
    out = np.full_like(inp, bg)
    current, start = initial, 0
    for row, (lo, hi, n, side, hit, target) in zip(rows, plan):
        marks = {lo+2*j if side == 1 else hi-2*j for j in range(n)}
        tile = mask if side == anchor_side else np.fliplr(mask)
        for col in range(lo, hi+1):
            is_anchor = (row, col) == (ar, ac)
            inp[pr+2*row, pc+2*(col-min_col)] = ink if is_anchor else marker if col in marks else ordinary
            r, c = top+row*h, left+col*w
            out[r:r+h, c:c+w][tile] = ordinary if is_anchor else ink
        if hit:
            turn = top+row*h-n-1
            out[start:turn+1, current] = route
            out[turn, min(current, target):max(current, target)+1] = route
            current, start = target, turn
    out[start:, current] = route
    pair = {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
    return pair, (ar, ac-min_col)


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    base_rotation = int(rng.integers(4))
    shift = int(rng.integers(3))  # held-out heading, longer sequence, new stencil scale
    maximum = int(rng.choice((2, 3, 4), p=(0.5, 0.25, 0.25)))
    fixed_palette = rng.random() < 0.25
    base_colors = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, used = {'train': [], 'test': []}, set()
    previous_anchor = None
    first_h = int(rng.integers(2, 4)) if shift == 2 else int(rng.integers(2, 5))
    first_w = 4 if shift == 2 else int(rng.integers(4, 6))

    def add(split, h, w, ns, rotation, mixed=False, contrast=None,
            reference_side=None, informative=False):
        for _ in range(48):
            colors = base_colors if fixed_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
            try:
                pair, anchor = _make_pair(rng, h, w, ns, rotation, colors, mixed, contrast,
                                          reference_side, informative)
            except ValueError:
                continue
            key = _key(pair['input'])
            if key not in used:
                # Decode/solve catches any accidental component merging or bounds error.
                try:
                    predicted = solve(pair['input'])
                except ValueError:
                    continue
                if not np.array_equal(predicted, pair['output']):
                    raise AssertionError('construction and input-only solve disagree')
                used.add(key)
                puzzle[split].append(pair)
                return anchor
        raise RuntimeError('bounded construction search exhausted')

    previous_anchor = add('train', first_h, first_w, [1, 2, 2], base_rotation, True,
                          informative=True)
    first_side = _decode(puzzle['train'][0]['input'])['anchor_side']
    second_h = _pick(rng, [h for h in range(2, 4 if shift == 2 else 5) if h != first_h])
    second_w = 2 if maximum >= 3 else int(rng.integers(2, 4))
    second_ns = [1, 2] if maximum == 2 else [1, 3] if maximum == 3 else [3, 4]
    rng.shuffle(second_ns)
    add('train', second_h, second_w, second_ns, (base_rotation+2) % 4,
        contrast=previous_anchor, reference_side=-first_side, informative=True)
    for _ in range(nt-2):
        w = int(rng.integers(2, 5 if shift == 2 else 6))
        h = int(rng.integers(2, 4 if shift == 2 else 5))
        cap = min(maximum, max(1, ((30-maximum-1)//w-1)//2))
        count = int(rng.integers(1, 4))
        ns = list(map(int, rng.integers(1, cap+1, size=count)))
        # Reduce the sequence if its mandatory vertical spacing cannot fit.
        while ns[0]+2+sum(math.ceil((h+n+1)/h)*h for n in ns[1:])+h+1 > 30:
            ns.pop()
        rotation = (base_rotation+2*int(rng.integers(2))) % 4 if shift == 0 else int(rng.integers(4))
        add('train', h, w, ns, rotation)
    for _ in range(nq):
        if shift == 1:
            h, w, count, cap = 2, int(rng.integers(2, 5)), 4, 2
        elif shift == 2:
            h, w, count, cap = 4, 5, int(rng.integers(1, 4)), 2
        else:
            h, w, count = int(rng.integers(2, 5)), int(rng.integers(2, 6)), int(rng.integers(1, 4))
            cap = min(maximum, max(1, ((30-maximum-1)//w-1)//2))
        ns = list(map(int, rng.integers(1, cap+1, size=count)))
        while ns[0]+2+sum(math.ceil((h+n+1)/h)*h for n in ns[1:])+h+1 > 30:
            ns.pop()
        rotation = (base_rotation+1+2*int(rng.integers(2))) % 4 if shift == 0 else int(rng.integers(4))
        add('test', h, w, ns, rotation)
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    m = _decode(pair['input'])
    walls = _walls(m)
    path, hits, targets, phases = _trace(m, walls)
    expected = walls.copy()
    expected[path] = m['route']
    output = _grid(pair['output'])
    _require(np.array_equal(np.rot90(expected, m['rotation']), output), 'incorrect output')
    population = int(m['mask'].sum())*sum(hi-lo+1 for row, lo, hi, n, side in m['bars'])
    _require(int(np.count_nonzero(expected != m['bg'])) == population+int(path.sum()),
             'copy count or collision invariant failed')
    tr, tc = m['template']
    _require(np.all(walls[tr:tr+m['h'], tc:tc+m['w']][m['mask']] == m['ordinary']),
             'anchor copy must be inverted in its original box')
    _require(not np.any(expected == m['marker']), 'instructions must disappear')
    m.update(hits=hits, targets=targets, phases=phases, path=path,
             slit=bool(np.any(~m['mask'].any(axis=0))),
             reflected=any(b[4] != m['anchor_side'] for b in m['bars']),
             signature=(m['h'], m['w'], tuple(map(int, m['mask'].flat))))
    return m


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape and not np.array_equal(a, b), 'dimensions or identity error')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            features[split].append(_features(pair))
    train, test = features['train'], features['test']
    _require(any(m['hits'] == (True, False, True) and m['reflected'] for m in train),
             'training needs an interacting hit/pass/hit example')
    _require(len({m['h'] for m in train}) >= 2 and len({m['w'] for m in train}) >= 2,
             'training must demonstrate independent bounding-box dimensions')
    _require(len({m['anchor'][0] for m in train}) >= 2 and len({m['anchor'][1] for m in train}) >= 2,
             'anchor cannot be a fixed lattice position')
    counts = {b[3] for m in train for b in m['bars']}
    _require({1, 2} <= counts and {b[3] for m in test for b in m['bars']} <= counts,
             'all tested unary counts must be demonstrated')
    _require({b[4] for m in train for b in m['bars']} == {-1, 1}, 'both endpoint commands are needed')
    _require({m['anchor_side'] for m in train} == {-1, 1}, 'both reference handednesses are needed')
    for side in (-1, 1):
        _require(any(m['anchor_side'] == side and not np.array_equal(m['mask'], np.fliplr(m['mask']))
                     for m in train), 'each reference handedness needs an asymmetric stencil')
    _require(any(not np.array_equal(m['mask'], np.flipud(m['mask'])) for m in train),
             'training must distinguish a forward-axis mask reflection')
    # The aperture witness refutes filled-box expansion; a count-two hit refutes
    # an uncounted fixed two-pixel clearance. These checks use returned pixels.
    _require(any(any(hit and b[3] == 2 for hit, b in zip(m['hits'], m['bars'])) for m in train),
             'count two must actually steer a training route')
    return None


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _puzzle_key(puzzles[7]) == _puzzle_key(generate_puzzle(7))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, stencils, anchors, layouts, headings, counts, phases, palettes, transitions = (set() for _ in range(9))
    skipped = fully_open = reflected = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                m = _features(pair)
                shapes.add(tuple(pair['input'].shape))
                stencils.add(m['signature'])
                anchors.add(m['anchor'])
                layouts.add(tuple(m['bars']))
                headings.add((split, m['rotation']))
                counts.update(b[3] for b in m['bars'])
                phases.update((m['w'], p) for p in m['phases'])
                palettes.add((m['bg'], m['ink'], m['ordinary'], m['marker'], m['route']))
                signs = [b[4] for b, hit in zip(m['bars'], m['hits']) if hit]
                transitions.update(zip(signs, signs[1:]))
                skipped += any(not hit for hit in m['hits'])
                fully_open += not any(m['hits'])
                reflected += m['reflected']
                before = pair['input'].copy()
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(before, pair['input'])
    assert len(shapes) >= 35 and len(stencils) >= 35 and len(anchors) >= 15 and len(layouts) >= 100
    assert len(palettes) >= 30 and counts == {1, 2, 3, 4}
    assert headings == {(s, k) for s in ('train', 'test') for k in range(4)}
    assert skipped >= 42 and reflected >= 42 and fully_open > 0
    assert len(phases) >= 12 and {(-1, -1), (1, 1)} <= transitions
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(91+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for counts_request in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                           (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts_request)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    _literal_regression()
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    m = _features(damaged['test'][0])
    changed = np.rot90(damaged['test'][0]['output'], -m['rotation']).copy()
    r, c = np.argwhere(m['path'])[1]
    changed[r, c] = m['bg']
    damaged['test'][0]['output'] = np.rot90(changed, m['rotation']).copy()
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('missing route cell accepted')


def _puzzle_key(puzzle):
    return tuple((_key(pair['input']), _key(pair['output'])) for s in ('train', 'test') for pair in puzzle[s])


def _literal_regression():
    """Hand-specified geometry and pixels, including a counterfactual slit."""
    a = np.zeros((24, 30), dtype=np.int8)
    a[0, 6] = 4
    a[16:18, 6:10] = (2, 2, 0, 2)
    a[1, [17, 19, 21]] = (1, 3, 3)
    a[7, [19, 21, 23, 25, 27]] = (3, 3, 1, 3, 1)
    a[13, [17, 19, 21, 23, 25, 27]] = (1, 2, 1, 3, 3, 3)
    expected = np.zeros_like(a)
    expected[4:6, [2, 3, 5, 6, 7, 9, 10, 11, 13]] = 2
    expected[10:12, [6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25]] = 2
    expected[16:18, [2, 3, 5, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25]] = 2
    expected[16:18, [6, 7, 9]] = 3
    expected[:3, 6] = expected[2, 6:16] = expected[2:14, 15] = 4
    expected[13, 15:29] = expected[13:, 28] = 4
    for k in range(4):
        recolor = np.array((8, 5, 7, 0, 2, 9, 3, 1, 6, 4), dtype=np.int8)
        inp = recolor[np.rot90(a, k)]
        out = recolor[np.rot90(expected, k)]
        assert np.array_equal(solve(inp), out)
    m = _decode(a)
    walls = _walls(m)
    correct_path, hits, targets, phases = _trace(m, walls)
    assert hits == (True, False, True) and targets == (15, None, 28)
    assert phases == (0, 1, 1) and m['anchor'] == (6, 1)
    assert not np.array_equal(_trace(m, walls, solid=True)[0], correct_path)
    assert not np.array_equal(_trace(m, walls, fixed_clearance=2)[0], correct_path)
    assert np.count_nonzero(correct_path) > 1
    # Fill the template's slit, keeping all other input cells unchanged. The
    # resulting valid family member now hits the previously transparent fence.
    closed = a.copy()
    closed[16:18, 8] = 2
    closed_m = _decode(closed)
    closed_path, closed_hits, _, _ = _trace(closed_m, _walls(closed_m))
    assert closed_hits == (True, True, True)
    assert not np.array_equal(closed_path, correct_path)
    assert not np.array_equal(solve(closed), expected)
    # Reflection alone also affects traversability, even with identical boxes.
    no_mirror = dict(m)
    no_mirror['bars'] = [b[:4]+(m['anchor_side'],) for b in m['bars']]
    unreflected_walls = _walls(no_mirror)
    assert np.any(unreflected_walls[10:12, 15] != m['bg'])
