"""Aperture elections: local shape ballots routed through moving hollow bodies.

Only Python's standard library and NumPy are used. See task_spec.md for the
pixel-level rule; no parent generator or saved pair is imported here.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import product
import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalize(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rotate(cells):
    return _normalize((c, -r) for r, c in cells)


def _reflect(cells):
    return _normalize((r, -c) for r, c in cells)


@lru_cache(maxsize=4096)
def _rotations(shape):
    result, current = set(), _normalize(shape)
    for _ in range(4):
        result.add(current)
        current = _rotate(current)
    return tuple(sorted(result))


@lru_cache(maxsize=4096)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) | set(_rotations(_reflect(shape)))))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(y, x)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                rr, cc = r+dr, c+dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(tuple(cells))
    return result


def _holes(mask):
    h, w = mask.shape
    return [c for c in _components(~mask)
            if not any(r in (0, h-1) or x in (0, w-1) for r, x in c)]


def _runs(rows):
    rows = sorted(set(rows))
    result = []
    for row in rows:
        if not result or row != result[-1][-1]+1:
            result.append([row])
        else:
            result[-1].append(row)
    return result


def _ports(shape):
    """Rows touching the cavity's rightmost column, in its given orientation."""
    edge = _size(shape)[1]-1
    return {r for r, c in shape if c == edge}


def _choice(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), dtype=bool)
    for r, c in shape:
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


@lru_cache(maxsize=1)
def _shape_pool():
    """Free polyominoes through area six, plus parameterized stepped cavities."""
    layer = {((0, 0),)}
    all_shapes = set(layer)
    for _ in range(5):
        next_layer = set()
        for shape in sorted(layer):
            frontier = {(r+dr, c+dc) for r, c in shape for dr, dc in _NEIGHBORS} - set(shape)
            for point in sorted(frontier):
                next_layer.add(_canonical(shape+(point,)))
        layer = next_layer
        all_shapes.update(layer)
    for height in range(2, 6):
        for widths in product(range(1, 4), repeat=height):
            if sum(widths) > 10:
                continue
            shape = tuple((r, c) for r, width in enumerate(widths)
                          for c in range(3-width, 3))
            all_shapes.add(_canonical(shape))
    return tuple(sorted(s for s in all_shapes if max(_size(s)) <= 5 and _hole_free(s)))


def _compatible_orientations(shape):
    result = []
    for raw in _variants(shape):
        longest = max(map(len, _runs(_ports(raw))))
        fitting = [v for v in _variants(shape) if _size(v)[0] <= longest]
        if fitting and min(_size(v)[1] for v in fitting) <= 5:
            result.append(raw)
    return result


@lru_cache(maxsize=1)
def _teaching_pairs():
    # Same area AND same bounding box, different exact geometry. The narrow
    # full front also leaves space for seven independently countable ballots.
    narrow = []
    for shape in _shape_pool():
        for raw in _variants(shape):
            h, w = _size(raw)
            if w == 2 and h in (3, 4) and len(_ports(raw)) == h:
                narrow.append((shape, raw))
    pairs = []
    for a, ar in narrow:
        for b, br in narrow:
            if a != b and len(a) == len(b) and _size(ar) == _size(br):
                pairs.append((a, ar, b, br))
    _require(bool(pairs), 'missing exact-shape teaching pairs')
    return tuple(pairs)


def _check_grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer grid required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'ARC colors required')
    return a


def _scene(input_grid):
    grid = _check_grid(input_grid)
    corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
    _require(len(set(map(int, corners))) == 1, 'all corners must encode background')
    bg = int(corners[0])
    candidates = []
    for turn in range(4):
        a = np.rot90(grid, turn)
        h, w = a.shape
        for col in range(1, w-1):
            rail = int(a[0, col])
            if rail == bg or not np.all(a[:, col] == rail) or np.count_nonzero(a == rail) != h:
                continue
            left = set(map(int, np.unique(a[:, :col]))) - {bg}
            right = set(map(int, np.unique(a[:, col+1:]))) - {bg}
            if len(left) == 1 and len(right) >= 2 and not (left & right) and rail not in left | right:
                candidates.append((a, turn, bg, rail, col, next(iter(left))))
    _require(len(candidates) == 1, 'unique rail, monochrome bodies and multicolor ballots required')
    return candidates[0]


def _analyze(input_grid):
    a, turn, bg, rail, col, carrier = _scene(input_grid)
    bodies, holes = [], []
    for cells in _components(a[:, :col] == carrier):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        mask = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        found = []
        for local in _holes(mask):
            global_cells = tuple((r+r0, c+c0) for r, c in local)
            contact = {r+r0 for r, c in local if c == mask.shape[1]-2 and mask[r, -1]}
            hole = dict(cells=global_cells, raw=_normalize(local), ports=contact,
                        rows={r for r, c in global_cells}, shape=_canonical(local))
            found.append(hole)
            holes.append(hole)
        enclosed = {p for hole in _holes(mask) for p in hole}
        open_rows = {r+r0 for r in range(mask.shape[0])
                     if mask.shape[1] >= 2 and not mask[r, -2] and (r, mask.shape[1]-2) not in enclosed}
        bodies.append(dict(cells=cells, mask=mask, bbox=(r0, c0, r1, c1),
                           shift=col-1-c1, holes=found, open_rows=open_rows))
    ballots = []
    side = a[:, col+1:]
    for color in sorted(set(map(int, np.unique(side))) - {bg}):
        for cells in _components(side == color):
            cells = tuple((r, c+col+1) for r, c in cells)
            ballots.append(dict(cells=cells, color=color, rows={r for r, c in cells},
                                raw=_normalize(cells), shape=_canonical(cells)))
    return dict(grid=a, turn=turn, background=bg, rail=rail, col=col, carrier=carrier,
                bodies=bodies, holes=holes, ballots=ballots)


def _signature(raw, equivalence):
    if equivalence == 'area':
        return len(raw)
    if equivalence == 'bbox':
        return tuple(sorted(_size(raw)))
    if equivalence == 'rotation':
        return min(_rotations(raw))
    if equivalence == 'translation':
        return _normalize(raw)
    return _canonical(raw)


def _elect(hole, ballots, mode='rule'):
    rows = hole['rows'] if mode in ('all_rows', 'all_holes') else hole['ports']
    equivalence = mode if mode in ('area', 'bbox', 'rotation', 'translation') else 'free'
    key = _signature(hole['raw'], equivalence)
    counts = Counter()
    eligible = []
    for ballot in ballots:
        if _signature(ballot['raw'], equivalence) != key:
            continue
        if mode == 'global':
            accepted = True
        elif mode == 'touch':
            accepted = bool(ballot['rows'] & rows)
        else:
            accepted = ballot['rows'] <= rows
        if accepted:
            counts[ballot['color']] += 1
            eligible.append(ballot)
    _require(bool(counts), 'every active aperture needs an eligible matching ballot')
    high = max(counts.values())
    winners = [c for c, n in counts.items() if n == high]
    _require(len(winners) == 1, 'local electorate must have a unique plurality')
    return winners[0], counts, eligible


def _render(data, mode='rule'):
    a = data['grid']
    bg, col, carrier = data['background'], data['col'], data['carrier']
    output = np.full_like(a, bg)
    output[:, col] = data['rail']
    for body in data['bodies']:
        _require(body['shift'] >= 0, 'body crosses rail')
        for r, c in body['cells']:
            output[r, c+body['shift']] = carrier
    for hole in data['holes']:
        rows = hole['rows'] if mode in ('all_rows', 'all_holes') else hole['ports']
        if not rows:
            continue
        winner, _, _ = _elect(hole, data['ballots'], mode)
        output[sorted(rows), col+1:] = winner
    return np.rot90(output, -data['turn']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input pixels only, including palette, direction and elections."""
    return _render(_analyze(input_grid))


def _fails(input_grid, output, mode):
    try:
        alternative = _render(_analyze(input_grid), mode)
    except AssertionError:
        return True
    return not np.array_equal(alternative, output)


def _local_winner_history(data):
    history = {}
    for hole in data['holes']:
        if hole['ports']:
            winner = _elect(hole, data['ballots'])[0]
            history.setdefault(hole['shape'], set()).add(winner)
    return history


def _body(rng, segments, tail=False, rough=False, margins=1):
    """segments are (oriented cavity, active/decoy/open, ballot recipe)."""
    width = max(_size(s)[1]+2+(2 if kind == 'decoy' else 0)
                for s, kind, recipe in segments) + int(rng.integers(2))
    height = sum(_size(s)[0] for s, kind, recipe in segments) + 3*(len(segments)-1) + 2*margins
    mask = np.ones((height, width), dtype=bool)
    pockets, y = [], margins
    for shape, kind, recipe in segments:
        h, w = _size(shape)
        recess = int(rng.integers(1, 3)) if kind == 'decoy' else 0
        x = width-1-w-recess
        cells = tuple((y+r, x+c) for r, c in shape)
        for r, c in cells:
            mask[r, c] = False
        if kind == 'open':
            # Choose a doorway that opens the pocket without cutting the
            # remaining carrier into separate pieces.
            for r in rng.permutation(sorted({r for r, c in cells})):
                r = int(r)
                left = min(c for rr, c in cells if rr == r)
                trial = mask.copy()
                trial[r, :left] = False
                if len(_components(trial)) == 1:
                    mask = trial
                    break
            else:
                raise AssertionError('no connected open-doorway construction')
        ports = {y+r for r in _ports(shape)} if kind == 'active' and not tail else set()
        pockets.append(dict(raw=shape, shape=_canonical(shape), ports=ports,
                            rows=set(range(y, y+h)), recipe=recipe))
        y += h+3
    if tail:
        extra = int(rng.integers(1, 3))
        extended = np.zeros((height, width+extra), dtype=bool)
        extended[:, :width] = mask
        extended[_choice(rng, (0, height-1)), width:] = True
        mask = extended
    if rough:
        # Optional back-edge erosion. Preserve exact cavities and the extreme.
        before = {_normalize(h) for h in _holes(mask)}
        for _ in range(int(rng.integers(2, 7))):
            trial = mask.copy()
            r = int(rng.integers(height))
            occupied = np.flatnonzero(trial[r])
            if len(occupied) < 2:
                continue
            trial[r, int(occupied[0])] = False
            if len(_components(trial)) == 1 and {_normalize(h) for h in _holes(trial)} == before:
                mask = trial
    _require(len(_components(mask)) == 1, 'carrier must remain connected')
    return dict(mask=mask, pockets=pockets)


def _fit_variants(raw, rows, chirality=None, partial=False):
    variants = _variants(_canonical(raw))
    if chirality == 'same':
        variants = _rotations(_normalize(raw))
    elif chirality == 'mirror':
        variants = _rotations(_reflect(raw))
    result = []
    for v in variants:
        h, w = _size(v)
        if partial and h < 2:
            continue
        for run in _runs(rows):
            if h > len(run):
                continue
            if partial:
                starts = (run[0]-1, run[-1]-h+2)
            else:
                starts = range(run[0], run[-1]-h+2)
            for start in starts:
                footprint = set(range(start, start+h))
                accepted = (bool(footprint & rows) and not footprint <= rows) if partial else footprint <= rows
                if accepted:
                    result.append((v, start))
    return result


def _token_layout(rng, pocket, band, colors, companion=None):
    """Pack whole ballots by variable-width boxes, then vary their row positions.

    Each bucket has a distinct row interval. One background column between boxes
    prevents same-color merging. This is a construction device, not a solve rule.
    """
    rows = pocket['ports'] or pocket['rows']
    recipe = pocket['recipe']
    winner, loser, third = colors
    descriptions = []
    if recipe == 'shape_contrast':
        descriptions = [(pocket['raw'], winner, False, None)]*2 + [(pocket['raw'], loser, False, None)]
        descriptions += [(companion, loser, False, None)]*2
        descriptions += [(pocket['raw'], loser, True, None)]*2
    elif recipe == 'reflection_plurality':
        descriptions = [(pocket['raw'], winner, False, 'mirror')]*3
        descriptions += [(pocket['raw'], loser, False, 'same')]*2
        descriptions += [(pocket['raw'], third, False, 'same')]*2
    elif recipe == 'partial_contrast':
        descriptions = [(pocket['raw'], winner, False, None)]*2 + [(pocket['raw'], loser, False, None)]
        descriptions += [(pocket['raw'], loser, True, None)]*2
    elif recipe == 'silent':
        descriptions = [(pocket['raw'], winner, False, None), (pocket['raw'], loser, False, None)]
    else:
        counts = recipe
        for color, count in zip((winner, loser, third), counts):
            descriptions.extend([(pocket['raw'], color, False, None)]*count)
    choices = []
    for raw, color, partial, chirality in descriptions:
        opts = _fit_variants(raw, rows, chirality, partial)
        _require(bool(opts), 'ballot does not fit aperture')
        best = min(_size(v)[1] for v, r in opts)
        opts = [(v, r) for v, r in opts if _size(v)[1] == best]
        choices.append((color, _choice(rng, opts)))
    cost = sum(_size(v)[1] for color, (v, row) in choices) + len(choices)-1
    _require(cost <= band, 'ballot band too narrow')
    # Spend slack on occasional larger orientations before adding random gaps.
    spare = band-cost
    if recipe not in ('shape_contrast', 'reflection_plurality'):
        for i, (raw, color, partial, chirality) in enumerate(descriptions):
            opts = [(v, r) for v, r in _fit_variants(raw, rows, chirality, partial)
                    if _size(v)[1] <= _size(choices[i][1][0])[1]+spare]
            new = _choice(rng, opts)
            spare -= _size(new[0])[1]-_size(choices[i][1][0])[1]
            choices[i] = (color, new)
    choices = [choices[int(i)] for i in rng.permutation(len(choices))]
    extras = rng.multinomial(spare, np.full(len(choices)+1, 1/(len(choices)+1)))
    x, result = int(extras[0]), []
    for i, (color, (v, row)) in enumerate(choices):
        result.append((color, tuple((row+r, x+c) for r, c in v)))
        x += _size(v)[1]+1+int(extras[i+1])
    return result


def _make_scene(rng, bodies, turn, teaching=None, companion_shapes=None):
    base = sum(b['mask'].shape[0] for b in bodies) + len(bodies)-1
    _require(base <= 28, 'bodies do not fit with background end margins')
    height = min(30, base+2+int(rng.integers(min(5, 29-base))))
    widest = max(b['mask'].shape[1] for b in bodies)
    if teaching:
        work = max(widest, 8)
        _require(work <= 9, 'teaching bodies too wide')
        work += int(rng.integers(10-work))
        width = 30
    else:
        work = widest+int(rng.integers(0, min(5, 20-widest)))
        # Determine the smallest safe band for all pockets before palette/layout.
        costs = []
        for body in bodies:
            for p in body['pockets']:
                rows = p['ports'] or p['rows']
                opts = _fit_variants(p['raw'], rows)
                best = min(_size(v)[1] for v, r in opts)
                n = 2 if p['recipe'] == 'silent' else sum(p['recipe'])
                costs.append(best*n+n-1)
        band_min = max(5, max(costs))
        _require(work+1+band_min <= 30, 'scene too wide')
        width = int(rng.integers(work+1+band_min, 31))
    band = width-work-1
    palette = list(map(int, rng.permutation(10)))
    bg, rail, carrier = palette[:3]
    ballot_colors = palette[3:]
    grid = np.full((height, width), bg, dtype=np.int8)
    grid[:, work] = rail
    order = list(map(int, rng.permutation(len(bodies))))
    extra = rng.multinomial(height-base-2, np.full(len(bodies)+1, 1/(len(bodies)+1)))
    y = 1+int(extra[0])
    for index, bi in enumerate(order):
        body = bodies[bi]
        mask = body['mask']
        bh, bw = mask.shape
        max_shift = work-bw
        shift = 0 if rng.random() < .3 else int(rng.integers(max_shift+1))
        # The first required scene contains visible movement.
        if teaching and index == 0 and max_shift:
            shift = int(rng.integers(1, max_shift+1))
        x = work-bw-shift
        grid[y:y+bh, x:x+bw][mask] = carrier
        for pi, pocket in enumerate(body['pockets']):
            p = dict(pocket)
            p['ports'] = {r+y for r in pocket['ports']}
            p['rows'] = {r+y for r in pocket['rows']}
            selected = list(map(int, rng.choice(ballot_colors, 3, replace=False)))
            companion = companion_shapes.get(pocket['shape']) if companion_shapes else None
            tokens = _token_layout(rng, p, band, selected, companion)
            for color, cells in tokens:
                for r, c in cells:
                    _require(0 < r < height-1, 'ballot reached corner/background margin')
                    _require(grid[r, work+1+c] == bg, 'ballots overlap')
                    grid[r, work+1+c] = color
        y += bh+1+int(extra[index+1])
    grid = np.rot90(grid, turn).copy()
    data = _analyze(grid)
    return {'input': grid, 'output': _render(data)}


def _first_demonstration(rng, turn):
    a, ar, b, br = _choice(rng, _teaching_pairs())
    bodies = [_body(rng, [(ar, 'active', 'shape_contrast'),
                          (ar, 'active', (1,))]),
              _body(rng, [(br, 'active', 'shape_contrast')])]
    # Voters of the other same-area/same-box shape must fit each window too.
    companions = {a: br, b: ar}
    # Make the two elected colors distinct; resampling is bounded and cheap.
    for _ in range(32):
        pair = _make_scene(rng, bodies, turn, 'exact_shape', companions)
        data = _analyze(pair['input'])
        winners = [_elect(h, data['ballots'])[0] for h in data['holes']]
        repeated = [_elect(h, data['ballots'])[0] for h in data['holes'] if h['shape'] == a]
        if len(set(repeated)) == 2 and len(set(winners)) >= 2 and all(
                _fails(pair['input'], pair['output'], m) for m in ('area', 'bbox', 'global', 'touch')):
            return pair
    raise AssertionError('could not produce exact-shape contrasts')


def _second_demonstration(rng, turn):
    candidates = []
    for shape in _shape_pool():
        for raw in _variants(shape):
            h, w = _size(raw)
            if w == 2 and h in (3, 4) and len(_ports(raw)) == h and _reflect(raw) not in _rotations(raw):
                candidates.append(raw)
    raw = _choice(rng, candidates)
    # A vertical zigzag has three cavity rows but only two leading contact rows.
    zigzag = ((0, 1), (1, 0), (1, 1), (2, 0))
    bodies = [_body(rng, [(raw, 'active', 'reflection_plurality'),
                          (zigzag, 'active', 'partial_contrast')]),
              _body(rng, [(raw, 'open', 'silent')]),
              _body(rng, [(raw, 'active', 'silent')], tail=True)]
    return _make_scene(rng, bodies, turn, 'enclosure')


def _random_demonstration(rng, turn, pool, silent=False):
    # Geometric fit alone bounds holes/bodies and number of ballots. No latent
    # solver convention changes with this sampling regime.
    for _ in range(100):
        count = int(rng.integers(1, 5))
        total = int(rng.integers(count, min(6, count+3)))
        allocation = [1]*count
        for _ in range(total-count):
            allocation[int(rng.integers(count))] += 1
        bodies = []
        failed = False
        for n in allocation:
            if bodies and rng.random() < .18:
                # Solid/staircase bodies participate in alignment but cannot
                # cast a ray. They have no ballot bucket of their own.
                sh, sw = int(rng.integers(2, 7)), int(rng.integers(2, 7))
                mask = np.ones((sh, sw), dtype=bool)
                if rng.random() < .5:
                    for row in range(sh):
                        mask[row, :int(rng.integers(sw))] = False
                bodies.append(dict(mask=mask, pockets=[]))
                continue
            segments = []
            for j in range(n):
                shape = _choice(rng, pool)
                raw = _choice(rng, _compatible_orientations(shape))
                kinds = ('decoy', 'open') if silent else ('active', 'active', 'active', 'decoy', 'open')
                kind = _choice(rng, kinds)
                if not silent and not segments and not bodies:
                    kind = 'active'
                rows = _ports(raw) if kind == 'active' else set(range(_size(raw)[0]))
                best = min(_size(v)[1] for v, r in _fit_variants(raw, rows))
                ballots = [(1,), (2, 1), (3, 2), (2, 1, 1), (3, 2, 2)]
                feasible = [p for p in ballots if best*sum(p)+sum(p)-1 <= 17]
                recipe = _choice(rng, feasible) if kind == 'active' else 'silent'
                segments.append((raw, kind, recipe))
            try:
                bodies.append(_body(rng, segments, rough=rng.random() < .45))
            except AssertionError:
                failed = True
                break
        if failed:
            continue
        try:
            pair = _make_scene(rng, bodies, turn)
        except AssertionError:
            continue
        return pair
    # Conservative one-cavity fallback still uses the requested visible pool.
    shape = _choice(rng, [s for s in pool if max(_size(s)) <= 4])
    raw = _choice(rng, _compatible_orientations(shape))
    body = _body(rng, [(raw, 'decoy' if silent else 'active', 'silent' if silent else (2, 1))])
    return _make_scene(rng, [body], turn)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base_turn = int(rng.integers(4))
    mixed = rng.random() < .65
    turns = [base_turn]*nt + [base_turn]*ne
    if mixed:
        turns = list(map(int, rng.integers(4, size=nt+ne)))
        turns[1] = (turns[0]+int(rng.integers(1, 4))) % 4
    puzzle = {'train': [], 'test': []}
    seen, known = set(), set()
    all_shapes = tuple(s for s in _shape_pool() if _compatible_orientations(s))
    # Teaching examples are shuffled as complete pairs, avoiding an index cue.
    required = [_first_demonstration(rng, turns[0]), _second_demonstration(rng, turns[1])]
    for i in range(nt):
        for _ in range(32):
            pair = required[i] if i < 2 else _random_demonstration(
                rng, turns[i], all_shapes, silent=rng.random() < .2)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('duplicate training inputs')
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
        data = _analyze(pair['input'])
        known.update(h['shape'] for h in data['holes'])
        known.update(b['shape'] for b in data['ballots'])
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    unseen = tuple(s for s in all_shapes if s not in known)
    _require(bool(unseen), 'need a visibly supported novel test shape')
    for i in range(ne):
        for _ in range(32):
            pair = _random_demonstration(rng, turns[nt+i], unseen)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError('duplicate test inputs')
        puzzle['test'].append(pair)
        seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split schema')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            data = _analyze(inp)
            _require(bool(data['bodies']) and bool(data['ballots']), 'missing bodies or ballots')
            previous = -2
            active_rows = set()
            for body in data['bodies']:
                r0, c0, r1, c1 = body['bbox']
                _require(r0 > previous+1, 'body row bands need background separation')
                previous = r1
                _require(body['shift'] >= 0, 'negative translation')
                for hole in body['holes']:
                    _require(_hole_free(hole['raw']), 'cavity traps a separate carrier island')
                    if hole['ports']:
                        _require(not active_rows & hole['ports'], 'overlapping ray elections')
                        active_rows |= hole['ports']
                        winner, counts, eligible = _elect(hole, data['ballots'])
                        _require(winner not in (data['background'], data['rail'], data['carrier']), 'role collision')
            # Colored segmentation and binary segmentation must identify the same
            # ballots, even when neighboring ballots use different colors.
            side = data['grid'][:, data['col']+1:]
            _require(len(_components(side != data['background'])) == len(data['ballots']), 'ballots touch')
            for ballot in data['ballots']:
                _require(_hole_free(ballot['raw']), 'ballots must be solid hole-free shapes')
            _require(np.array_equal(out, _render(data)), 'wrong alignment, election, clearing or rays')
            _require(np.array_equal(solve(inp), out), 'input-only solver disagreement')
            canonical_out = np.rot90(out, data['turn'])
            _require(np.all(canonical_out[:, data['col']] == data['rail']), 'rail changed')
            _require(np.count_nonzero(canonical_out[:, :data['col']] == data['carrier']) ==
                     sum(len(b['cells']) for b in data['bodies']), 'rigid bodies lost cells')
            if split == 'train':
                training.append((pair, data))
    # Require answer-level counterexamples, not just a catalog of different masks.
    for shortcut in ('area', 'bbox', 'global', 'touch', 'rotation', 'all_rows'):
        _require(any(_fails(p['input'], p['output'], shortcut) for p, d in training),
                 f'training does not distinguish {shortcut}')
    _require(any(b['open_rows'] for p, d in training for b in d['bodies']), 'no open enclosure contrast')
    _require(any(not h['ports'] for p, d in training for h in d['holes']), 'no recessed enclosed contrast')
    _require(any(b['shift'] > 0 for p, d in training for b in d['bodies']), 'no movement demonstration')
    _require(any(len(c) >= 3 and max(c.values())*2 <= sum(c.values())
                 for p, d in training for h in d['holes'] if h['ports']
                 for w, c, e in [_elect(h, d['ballots'])]), 'no non-majority plurality evidence')
    _require(any(any(len(winners) > 1 for winners in _local_winner_history(d).values())
                 for p, d in training),
             'no same-shape, different-local-winner demonstration')


def self_test() -> dict:
    """Bounded pixel-derived audits over 42 seeds; returns compact measurements."""
    puzzles, records, signatures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze(pair['input'])
                elections = [_elect(h, d['ballots']) for h in d['holes'] if h['ports']]
                records.append(dict(split=split, shape=pair['input'].shape, turn=d['turn'],
                                    bodies=len(d['bodies']), cavities=len(d['holes']),
                                    ballot_count=len(d['ballots']), shifts=tuple(b['shift'] for b in d['bodies']),
                                    vocabulary={h['shape'] for h in d['holes']},
                                    silhouettes={(_size(_normalize(b['cells'])), _normalize(b['cells'])) for b in d['bodies']},
                                    ballots={tuple(sorted(c.values(), reverse=True)) for w, c, e in elections},
                                    role=(d['background'], d['rail'], d['carrier']),
                                    partial_front=any(h['ports'] and h['ports'] != h['rows'] for h in d['holes'])))
    _require(len(signatures) == 42, 'different seeds collapsed')
    _require(len({r['shape'] for r in records}) >= 25, 'dimensions collapsed')
    _require(len({r['role'] for r in records}) >= 30, 'palette roles collapsed')
    _require({r['turn'] for r in records} == set(range(4)), 'orientation collapsed')
    _require(len(set().union(*(r['vocabulary'] for r in records))) >= 35, 'shape vocabulary collapsed')
    _require(len(set().union(*(r['silhouettes'] for r in records))) >= 70, 'carrier geometry collapsed')
    _require(len({r['ballot_count'] for r in records}) >= 10, 'ballot totals collapsed')
    _require({1, 2, 3}.issubset({r['bodies'] for r in records}), 'body counts collapsed')
    _require(len({r['cavities'] for r in records}) >= 4, 'cavity counts collapsed')
    _require(any(0 in r['shifts'] for r in records) and any(max(r['shifts']) >= 3 for r in records), 'travel collapsed')
    for p in puzzles:
        known = set()
        for pair in p['train']:
            data = _analyze(pair['input'])
            known.update(h['shape'] for h in data['holes'])
            known.update(b['shape'] for b in data['ballots'])
        _require(all(any(h['ports'] and h['shape'] not in known for h in _analyze(pair['input'])['holes'])
                     for pair in p['test']), 'novel test shape not active')
    # Rotation and palette equivariance test the conventions, not random seeds.
    pair = puzzles[0]['train'][0]
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn)), 'rotation covariance')
    for flip in (np.flipud, np.fliplr):
        _require(np.array_equal(solve(flip(pair['input'])), flip(pair['output'])), 'reflection covariance')
    lookup = np.array([9, 3, 7, 0, 8, 1, 5, 2, 6, 4], dtype=np.int8)
    _require(np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']]), 'palette covariance')
    # A valid arrangement beyond the sampler's >=5-column ballot bands:
    # two vertical-bar elections in a one-cell band. Expected motion is set
    # directly, and each already visible ballot becomes its one-cell ray.
    thin = np.zeros((13, 6), dtype=np.int8)
    expected = thin.copy()
    thin[:, 4] = expected[:, 4] = 1
    for start, color in ((1, 4), (7, 5)):
        thin[start:start+5, :3] = 2
        thin[start+1:start+4, 1] = 0
        thin[start+1:start+4, 5] = color
        expected[start:start+5, 1:4] = 2
        expected[start+1:start+4, 2] = 0
        expected[start+1:start+4, 5] = color
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(thin, turn)), np.rot90(expected, turn)), 'one-cell band')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(3100+10*nt+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, 'explicit counts')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    again = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
                 for a, b in zip(puzzles[0][s], again[s]) for k in ('input', 'output')), 'seed determinism')
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), dimensions=len({r['shape'] for r in records}),
                vocabulary=len(set().union(*(r['vocabulary'] for r in records))),
                silhouettes=len(set().union(*(r['silhouettes'] for r in records))),
                ballot_totals=sorted({r['ballot_count'] for r in records}),
                body_counts=sorted({r['bodies'] for r in records}),
                cavity_counts=sorted({r['cavities'] for r in records}),
                ballot_patterns=sorted(set().union(*(r['ballots'] for r in records))))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
