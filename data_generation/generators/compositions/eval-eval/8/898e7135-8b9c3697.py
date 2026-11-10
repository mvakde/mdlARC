"""Miniature shape locks and rigid, shape-sensitive docking.

No parent modules, files, seeds or training labels are used by solve().
Only standard Python and NumPy are needed. No files or traces are written.
"""

from collections import Counter
import hashlib
import json

import numpy as np


_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_OUT = ((1, 0), (0, 1), (-1, 0), (0, -1))
_PARTS = ('structure', 'original', 'final', 'route', 'decoy')
_HOST_SEEDS = (0,382683925,680558363,932259960,136768009,369452316,698327863,502237114,859340977,921146565,468160298,868180529,884298779,885893428,743987304,530286462,598048523,158956392,505324387,367483798,914007324,829987362,17903656,761424797,856925591,426983432,119823706,888721104,138000357,660199698,397592661,415207159,235052154,576712417,190779146,744675021,266883292,750572296,628232970,269908487,281815619,645498389)


def _norm(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _size(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _turn(cells, n):
    for _ in range(n % 4):
        cells = [(c, -r) for r, c in cells]
    return _norm(cells)


def _variants(cells, reflect=True):
    bases = (cells, tuple((r, -c) for r, c in cells)) if reflect else (cells,)
    return tuple(sorted({_turn(b, q) for b in bases for q in range(4)}))


def _class(cells):
    return _variants(_norm(cells))[0]


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, comp = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                p = r+dr, c+dc
                if p in left:
                    left.remove(p)
                    stack.append(p)
                    comp.add(p)
        result.append(comp)
    return result


def _rect(h, w):
    return {(r, c) for r in range(h) for c in range(w)}


def _shift(cells, dr, dc=0):
    return {(r+dr, c+dc) for r, c in cells}


def _scale(cells, k):
    return {(r*k+a, c*k+b) for r, c in cells for a in range(k) for b in range(k)}


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _decode(cells, k):
    cells = _norm(cells)
    h, w = _size(cells)
    if h % k or w % k:
        return None
    blocks = Counter((r//k, c//k) for r, c in cells)
    if any(v != k*k for v in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _holefree(cells):
    h, w = _size(cells)
    empty = _rect(h+2, w+2) - _shift(cells, 1, 1)
    return len(_components(empty)) == 1


def _frame(cells, out):
    perp = -out[1], out[0]
    a = min(r*out[0]+c*out[1] for r, c in cells)
    b = min(r*perp[0]+c*perp[1] for r, c in cells)
    local = {(r*out[0]+c*out[1]-a, r*perp[0]+c*perp[1]-b) for r, c in cells}
    return local, perp, a, b


def _global(dock, cells):
    out, perp = dock['out'], dock['perp']
    return {((r+dock['a'])*out[0]+(c+dock['b'])*perp[0],
             (r+dock['a'])*out[1]+(c+dock['b'])*perp[1]) for r, c in cells}


def _local(dock, cells):
    out, perp = dock['out'], dock['perp']
    return {(r*out[0]+c*out[1]-dock['a'], r*perp[0]+c*perp[1]-dock['b']) for r, c in cells}


def _parse_dock(cells):
    """Recover a perforated backplate and three positive prongs by geometry."""
    parses = []
    for out in _OUT:
        local, perp, a, b = _frame(cells, out)
        h, w = _size(local)
        if w < 8:
            continue
        for base in range(3, min(8, h-1)):
            plate = {(r, c) for r, c in local if r <= base}
            border = ({(0, c) for c in range(w)} | {(base, c) for c in range(w)}
                      | {(r, c) for r in range(base+1) for c in (0, w-1)})
            if not border <= plate:
                continue
            holes = _rect(base+1, w) - plate
            if not 4 <= len(holes) <= 11 or len(_components(holes)) != 1:
                continue
            lengths = [sum(r > base and c == q for r, c in local) for q in range(w)]
            expected = plate | {(r, c) for c, length in enumerate(lengths)
                                for r in range(base+1, base+length+1)}
            if expected != local:
                continue
            runs, q = [], 0
            while q < w:
                if not lengths[q]:
                    q += 1
                    continue
                start, length = q, lengths[q]
                while q < w and lengths[q] == length:
                    q += 1
                runs.append((start, q, length))
            if len(runs) != 3 or runs[0][0] != 0 or runs[-1][1] != w:
                continue
            if any(runs[i][1] >= runs[i+1][0] for i in (0, 1)):
                continue
            q0, q1, center = runs[1]
            if q0+q1 != w or not _holefree(_norm(holes)):
                continue
            parses.append(dict(out=out, perp=perp, a=a, b=b, base=base, width=w,
                               runs=tuple(runs), center=center, q0=q0, q1=q1,
                               hole=_norm(holes), cells=cells))
    if len(parses) > 1:
        raise ValueError('ambiguous dock orientation')
    return parses[0] if parses else None


def _motion(dock, token, k, structures, bounds, ignore_shape=False):
    local = _local(dock, token['cells'])
    logical = _decode(local, k)
    if logical is None or _size(logical)[1] != 3:
        return None
    if not ignore_shape and _class(logical) != _class(dock['hole']):
        return None
    r0 = min(r for r, c in local)
    c0, c1 = min(c for r, c in local), max(c for r, c in local)+1
    if c0+c1 != dock['width'] or dock['q1']-dock['q0'] != k:
        return None
    if c0 <= dock['runs'][0][1] or c1 >= dock['runs'][2][0]:
        return None
    if r0 <= dock['base']+max(x[2] for x in dock['runs'])+1:
        return None
    leading = min(r for r, c in local if dock['q0'] <= c < dock['q1'])
    distance = leading - (dock['base']+dock['center']+1)
    if distance <= 0:
        return None
    final_local = _shift(local, -distance)
    route_local = set().union(*(_shift(local, -d) for d in range(distance+1)))
    route, final = _global(dock, route_local), _global(dock, final_local)
    if not route.isdisjoint(structures):
        return None
    if any(not (0 <= r < bounds[0] and 0 <= c < bounds[1]) for r, c in route):
        return None
    # The next unit step must hit the exposed central tip, not a plate or guard.
    hit = _global(dock, _shift(final_local, -1)) & structures
    tip = _global(dock, {(dock['base']+dock['center'], c)
                         for c in range(dock['q0'], dock['q1'])})
    if not hit or not hit <= tip:
        return None
    return dict(final=final, route=route, distance=distance, recess=leading-r0,
                shape=logical, color=token['color'])


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError('expected an integer ARC grid with sides 1..30')
    return grid.astype(np.int8, copy=True)


def _analyze(input_grid):
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if bg == 0 or counts[bg] <= grid.size//2 or counts[0]:
        raise ValueError('background must be a nonblack strict majority; black is reserved')
    docks, tokens = [], []
    for color in range(1, 10):
        if color == bg:
            continue
        for cells in _components(map(tuple, np.argwhere(grid == color))):
            dock = _parse_dock(cells)
            if dock is None:
                tokens.append(dict(cells=cells, color=color))
            else:
                dock['color'] = color
                docks.append(dock)
    if not 1 <= len(docks) <= 2 or not tokens:
        raise ValueError('expected one or two shape docks and loose keys')
    # Entire objects, even when differently colored, have an input moat.
    objects = docks+tokens
    for i, obj in enumerate(objects):
        neighbors = {(r+dr, c+dc) for r, c in obj['cells']
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if any(not neighbors.isdisjoint(other['cells']) for other in objects[i+1:]):
            raise ValueError('input objects must be separated')
    structures = set().union(*(d['cells'] for d in docks))
    solutions = []
    for k in (2, 3):
        decoded = [_decode(t['cells'], k) for t in tokens]
        if any(s is None or not 4 <= len(s) <= 11 or not _holefree(s) for s in decoded):
            continue
        if any(d['q1']-d['q0'] != k for d in docks):
            continue
        matches, used, swept = [], set(), set()
        for fi, dock in enumerate(docks):
            candidates = [(ti, m) for ti, token in enumerate(tokens)
                          if (m := _motion(dock, token, k, structures, grid.shape)) is not None]
            if len(candidates) > 1:
                raise ValueError('two keys activate the same dock')
            if candidates:
                ti, motion = candidates[0]
                if ti in used or not swept.isdisjoint(motion['route']):
                    raise ValueError('ambiguous or interacting motions')
                if any(not motion['route'].isdisjoint(t['cells']) for j, t in enumerate(tokens) if j != ti):
                    raise ValueError('a foreign loose key obstructs a sweep')
                used.add(ti)
                swept |= motion['route']
                matches.append(dict(motion, fi=fi, ti=ti))
        if matches:
            solutions.append(dict(background=bg, docks=docks, tokens=tokens, scale=k,
                                  matches=matches, used=used, structures=structures))
    if len(solutions) != 1:
        raise ValueError('expected one scale and unique geometric assignment')
    return grid, solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select by miniature D4 shape, then slide rigidly to real tip contact."""
    grid, info = _analyze(input_grid)
    output = grid.copy()
    for token in info['tokens']:
        _paint(output, token['cells'], info['background'])
    for motion in info['matches']:
        _paint(output, motion['route']-motion['final'], 0)
        _paint(output, motion['final'], motion['color'])
    return output


def _sample_shape(rng, height, *, area=None, chiral=False, notch=False, flat=False):
    """Procedural connected masks, rather than a fixed polyomino vocabulary."""
    for _ in range(1800):
        h = int(rng.integers(2, height+1))
        n = int(rng.integers(4, min(11, h*3)+1)) if area is None else area
        if n > 3*h:
            continue
        indices = rng.choice(3*h, n, replace=False)
        cells = _norm([(int(i)//3, int(i)%3) for i in indices])
        if _size(cells)[1] != 3 or len(_components(cells)) != 1 or not _holefree(cells):
            continue
        if chiral and len(_variants(cells)) == len(_variants(cells, False)):
            continue
        if notch and (min(r for r, c in cells if c == 1) == 0 or (0, 0) not in cells or (0, 2) not in cells):
            continue
        if flat and min(r for r, c in cells if c == 1) != 0:
            continue
        return cells
    raise RuntimeError('bounded procedural shape sampler exhausted')


def _make_module(rng, k, mode, two, *, reflection=False, notch=False, flat=False, extra_wrong=False):
    height = 3 if k == 3 else 4
    if extra_wrong:
        # Reserve a second full-sized loose key behind the first one. This
        # joint size constraint keeps all three test keys within 30 cells.
        height -= 1
    shape = _sample_shape(rng, height, chiral=reflection, notch=notch, flat=flat)
    hole = shape
    if mode == 'wrong':
        # Equal area and full bounding box: neither area nor span selects it.
        for _ in range(400):
            other = _sample_shape(rng, height, area=len(shape))
            if _class(other) != _class(shape) and _size(other) == _size(shape):
                hole = other
                break
        else:
            return None
    variants = _variants(hole)
    if reflection:
        rotations = _variants(shape, False)
        variants = tuple(s for s in variants if s not in rotations)
    hole = variants[int(rng.integers(len(variants)))]
    # Backplate dimensions vary independently of stencil position and prongs.
    thickness = 1 if two and k == 3 else int(rng.integers(1, 3))
    clearance = 1 if two and k == 3 else int(rng.integers(1, 3))
    if two and k == 2 and thickness+clearance > 3:
        clearance = 1
    w = 3*k+2*(thickness+clearance)
    hh, hw = _size(hole)
    plate_h = hh+2+int(rng.integers(2))
    hr = int(rng.integers(1, plate_h-hh))
    hc = int(rng.integers(1, w-hw))
    plate = _rect(plate_h, w)-_shift(hole, hr, hc)
    base = plate_h-1
    q0 = (w-k)//2
    offset = min(r for r, c in shape if c == 1)*k
    center = offset+int(rng.integers(1, 4))
    left, right = map(int, rng.integers(1, min(9, center+4), size=2))
    prongs = {(r, c) for qa, qb, length in ((0, thickness, left),
              (q0, q0+k, center), (w-thickness, w, right))
              for r in range(plate_h, plate_h+length) for c in range(qa, qb)}
    structure = plate | prongs
    start = base+max(left, center, right)+int(rng.integers(2, 5))
    original = _shift(_scale(shape, k), start, thickness+clearance)
    final_top = base+center+1-offset
    final = _shift(_scale(shape, k), final_top, thickness+clearance)
    distance = start-final_top
    route = set().union(*(_shift(original, -d) for d in range(distance+1)))
    decoy = set()
    if extra_wrong:
        for _ in range(50):
            other = _sample_shape(rng, height, area=len(shape), flat=True)
            if _class(other) != _class(shape):
                decoy = _shift(_scale(other, k), max(r for r, c in original)+3,
                               thickness+clearance)
                break
        if not decoy:
            return None
    h = max(r for r, c in structure | original | decoy)+1
    if h > (29 if extra_wrong else 28):
        return None
    return dict(structure=structure, original=original, final=final, route=route,
                decoy=decoy, shape=shape, h=h, w=w, active=mode == 'active', mode=mode)


def _rotate_module(module, q):
    all_cells = module['structure'] | module['original'] | module['decoy']
    # Apply one common affine transform to input, final and swept footprints.
    def raw(cells):
        for _ in range(q):
            cells = {(c, -r) for r, c in cells}
        return cells
    rotated = raw(all_cells)
    r0, c0 = min(r for r, c in rotated), min(c for r, c in rotated)
    result = {name: _shift(raw(module[name]), -r0, -c0)
              for name in _PARTS}
    result.update(h=_size(_norm(rotated))[0], w=_size(_norm(rotated))[1],
                  shape=module['shape'], active=module['active'], mode=module['mode'])
    return result


def _place_modules(rng, modules):
    # Random two-dimensional positions; a feasible common-axis placement is a
    # bounded fallback. Rectangles reserve every swept footprint, not just ink.
    count = len(modules)
    for attempt in range(60):
        q = int(rng.integers(4))
        turns = [int(rng.integers(4)) for _ in modules] if attempt < 35 else [q]*count
        rotated = [_rotate_module(m, t) for m, t in zip(modules, turns)]
        placed, rectangles = [], []
        for m in rotated:
            origins = [(r, c) for r in range(31-m['h']) for c in range(31-m['w'])
                       if all(r+m['h'] < a or a+h < r or c+m['w'] < b or b+w < c
                              for a, b, h, w in rectangles)]
            if not origins:
                break
            r, c = origins[int(rng.integers(len(origins)))]
            p = {name: _shift(m[name], r, c) for name in _PARTS}
            p.update(active=m['active'], shape=m['shape'], mode=m['mode'])
            placed.append(p)
            rectangles.append((r, c, m['h'], m['w']))
        else:
            return placed, rectangles
    # Every module is at most 29 deep; combined cross-sections are at most 27.
    q = int(rng.integers(4))
    cursor, placed, rectangles = 0, [], []
    for module in modules:
        r = int(rng.integers(31-module['h']))
        m = {name: _shift(module[name], r, cursor) for name in _PARTS}
        m.update(active=module['active'], shape=module['shape'], mode=module['mode'])
        # Rotate around the complete 30-square so offsets are shared.
        for name in _PARTS:
            for _ in range(q):
                m[name] = {(c, 29-r) for r, c in m[name]}
        placed.append(m)
        cursor += module['w']+1
    return placed, rectangles


def _pair(rng, k, modes, *, reflection=False, notch=False, offaxis=False, double_notch=False):
    for _ in range(90):
        modules = [_make_module(rng, k, mode, len(modes) == 2,
                               reflection=reflection and i == 0,
                               notch=double_notch or (notch and i == 0),
                               flat=i == 1 and not double_notch,
                               extra_wrong=double_notch and i == 0) for i, mode in enumerate(modes)]
        if any(m is None for m in modules):
            continue
        placed, rectangles = _place_modules(rng, modules)
        palette = list(map(int, rng.permutation(np.arange(1, 10))))
        bg = palette[0]
        grid = np.full((30, 30), bg, dtype=np.int8)
        output = grid.copy()
        for i, m in enumerate(placed):
            dc, tc = palette[1+i], palette[3+i]
            _paint(grid, m['structure'], dc)
            _paint(output, m['structure'], dc)
            _paint(grid, m['original'], tc)
            _paint(grid, m['decoy'], palette[5])
            if m['active']:
                _paint(output, m['route']-m['final'], 0)
                _paint(output, m['final'], tc)
        if offaxis:
            # A differently colored copy fits the SAME stencil, but its line
            # misses the prong. It remains an ordinary removable loose key.
            shape = _turn(modules[0]['shape'], int(rng.integers(4)))
            cells = _scale(shape, k)
            eh, ew = _size(cells)
            occupied = set().union(*(m['structure'] | m['route'] for m in placed))
            blocked = {(r+dr, c+dc) for r, c in occupied
                       for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            choices = [(r, c) for r in range(31-eh) for c in range(31-ew)
                       if _shift(cells, r, c).isdisjoint(blocked)]
            if not choices:
                continue
            r, c = choices[int(rng.integers(len(choices)))]
            _paint(grid, _shift(cells, r, c), palette[5])
        # Random empty margins, with no clipping of any input or output mark.
        visible = np.argwhere((grid != bg) | (output != bg))
        lo, hi = visible.min(0), visible.max(0)+1
        lo = np.maximum(0, lo-rng.integers(0, 4, size=2))
        hi = np.minimum(30, hi+rng.integers(0, 4, size=2))
        grid = grid[lo[0]:hi[0], lo[1]:hi[1]].copy()
        output = output[lo[0]:hi[0], lo[1]:hi[1]].copy()
        try:
            _, info = _analyze(grid)
        except ValueError:
            continue
        if len(info['matches']) != modes.count('active'):
            continue
        predicted = solve(grid)
        if not np.array_equal(predicted, output):
            continue
        if offaxis and not _offaxis_witness(info):
            continue
        if reflection and not _reflection_witness(info):
            continue
        if double_notch and not _wrong_shape_witness(info, grid.shape):
            continue
        return dict(input=grid, output=output)
    raise RuntimeError('bounded pair construction exhausted')


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _valid_count(n, lo, hi):
    return isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_)) and lo <= n <= hi


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_count(num_train, 2, 10) or not _valid_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError('require integer counts 2..10 / 1..4 with num_train > num_test')
    rng = np.random.default_rng(seed)
    # A familiar scale in most tests; a puzzle-level subset uses a new scale.
    shift_scale = bool(rng.integers(2))
    base_k = int(rng.choice([2, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train' and i == 0:
                modes, notch, offaxis, reflection, k = ['active', 'wrong'], False, False, True, base_k
            elif split == 'train' and i == 1:
                modes, notch, offaxis, reflection, k = ['active'], True, True, False, base_k
            else:
                modes = ['active', 'active'] if split == 'test' or rng.random() < .65 else ['active']
                notch = bool(rng.integers(2))
                offaxis = len(modes) == 1 and bool(rng.integers(2))
                reflection = False
                k = (5-base_k if shift_scale else base_k) if split == 'test' else int(rng.choice([2, 3]))
            for _ in range(12):
                pair = _pair(rng, k, modes, reflection=reflection, notch=notch, offaxis=offaxis,
                             double_notch=split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input budget exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _reflection_witness(info):
    return any(info['docks'][m['fi']]['hole'] not in _variants(m['shape'], False)
               for m in info['matches'])


def _offaxis_witness(info):
    return any(_class(_decode(t['cells'], info['scale'])) == _class(d['hole'])
               for i, t in enumerate(info['tokens']) if i not in info['used'] for d in info['docks'])


def _wrong_shape_witness(info, shape):
    return any(_motion(d, t, info['scale'], info['structures'], shape, ignore_shape=True) is not None
               and _class(_decode(t['cells'], info['scale'])) != _class(d['hole'])
               for i, t in enumerate(info['tokens']) if i not in info['used'] for d in info['docks'])


def validate_generated_puzzle(puzzle):
    """Check input geometry, role necessity, contact, rigid color and labels."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert _valid_count(a, 2, 10) and _valid_count(b, 1, 4) and a > b
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            _, info = _analyze(inp)
            assert np.array_equal(solve(inp), out)
            assert not np.array_equal(inp, out) and np.any(out == 0)
            assert all(int(out[p]) == d['color'] for d in info['docks'] for p in d['cells'])
            for m in info['matches']:
                assert all(int(out[p]) == m['color'] for p in m['final'])
                assert all(int(out[p]) == 0 for p in m['route']-m['final'])
                assert len(m['final']) == len(info['tokens'][m['ti']]['cells'])
                assert m['distance'] >= 1 and m['route']-m['final']
            for i, t in enumerate(info['tokens']):
                if i not in info['used']:
                    assert all(int(out[p]) == info['background'] for p in t['cells'])
            if split == 'train':
                assert sum(m['recess'] > 0 for m in info['matches']) <= 1
                training.append((info, inp.shape))
            else:
                assert len(info['matches']) == 2 and all(m['recess'] > 0 for m in info['matches'])
                assert _wrong_shape_witness(info, inp.shape), 'test must need its miniature shape lock'
    assert any(_reflection_witness(i) for i, shape in training), 'missing D4 witness'
    assert any(_wrong_shape_witness(i, shape) for i, shape in training), 'shape matching is not necessary'
    assert any(_offaxis_witness(i) for i, shape in training), 'alignment is not necessary'
    assert any(m['recess'] > 0 for i, shape in training for m in i['matches']), 'missing interlock witness'


def self_test():
    """42 seeds, streamed metrics, count boundaries and semantic corruptions.

    Retains hashes/features only, no grids from previous seeds. Never writes
    traces, JSON, diagnostics, journals, indexes, backups or sample payloads.
    """
    metrics = {name: set() for name in ('puzzles', 'shapes', 'classes', 'directions', 'scales',
                                        'distances', 'prongs', 'backgrounds', 'recesses', 'dock_counts')}
    reflection = wrong = offaxis = interlock = nonrectangle = 0
    for seed in _HOST_SEEDS:
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                digest.update(bytes(inp.shape)+inp.tobytes()+out.tobytes())
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                _, info = _analyze(inp)
                metrics['shapes'].add(inp.shape)
                metrics['scales'].add(info['scale'])
                metrics['backgrounds'].add(info['background'])
                metrics['dock_counts'].add(len(info['docks']))
                reflection += bool(_reflection_witness(info))
                wrong += bool(_wrong_shape_witness(info, inp.shape))
                offaxis += bool(_offaxis_witness(info))
                for dock in info['docks']:
                    metrics['classes'].add(_class(dock['hole']))
                    metrics['directions'].add(dock['out'])
                    metrics['prongs'].add(tuple(x[2] for x in dock['runs']))
                for m in info['matches']:
                    metrics['distances'].add(m['distance'])
                    metrics['recesses'].add(m['recess'])
                    interlock += m['recess'] > 0
                    trail = m['route']-m['final']
                    nonrectangle += len(trail) < np.prod(_size(_norm(trail)))
        metrics['puzzles'].add(digest.hexdigest())
    assert len(metrics['puzzles']) == 42
    assert len(metrics['shapes']) >= 30 and len(metrics['classes']) >= 30
    assert metrics['directions'] == set(_OUT) and metrics['scales'] == {2, 3}
    assert len(metrics['prongs']) >= 60 and len(metrics['backgrounds']) == 9
    assert len(metrics['distances']) >= 7 and len(metrics['recesses']) >= 3
    assert metrics['dock_counts'] == {1, 2}
    assert min(reflection, wrong, offaxis, interlock, nonrectangle) >= 42
    # Every legal explicit count combination, without retaining its puzzle.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(71000+10*a+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p, q = generate_puzzle(9, 2, 1), generate_puzzle(9, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # Shape-based object parsing also handles a shared object palette.
    pair = q['train'][0]
    _, info = _analyze(pair['input'])
    bg = info['background']
    shared = next(c for c in range(1, 10) if c != bg)
    merged = pair['input'].copy()
    merged[merged != bg] = shared
    expected = pair['output'].copy()
    expected[(expected != bg) & (expected != 0)] = shared
    assert np.array_equal(solve(merged), expected)
    # Shortening an outer guard cannot move a key that fits inside its channel.
    found_guard = False
    for split in ('train', 'test'):
        for pair in q[split]:
            _, info = _analyze(pair['input'])
            for dock in info['docks']:
                for qa, qb, length in (dock['runs'][0], dock['runs'][2]):
                    if length < 2:
                        continue
                    tip = _global(dock, {(dock['base']+length, c) for c in range(qa, qb)})
                    altered, expected = pair['input'].copy(), pair['output'].copy()
                    _paint(altered, tip, info['background'])
                    _paint(expected, tip, info['background'])
                    assert np.array_equal(solve(altered), expected)
                    found_guard = True
                    break
                if found_guard:
                    break
            if found_guard:
                break
        if found_guard:
            break
    assert found_guard
    _, info = _analyze(p['test'][0]['input'])
    cell = min(info['matches'][0]['final'])
    p['test'][0]['output'][cell] = 0
    try:
        validate_generated_puzzle(p)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt final footprint accepted')
    broken = q['train'][1]['input'].copy()
    _, info = _analyze(broken)
    cell = min(info['tokens'][0]['cells'])
    broken[cell] = info['background']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('broken macrocell accepted')
    return {name: len(values) for name, values in metrics.items()} | dict(
        reflection_pairs=reflection, wrong_shape_pairs=wrong, offaxis_pairs=offaxis,
        interlocking_moves=int(interlock), nonrectangular_trails=int(nonrectangle))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
