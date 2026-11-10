"""Contact-print relay mosaics: registered stencil intersection and color composition.

Only NumPy and the standard library are used. No evidence or files are read.
"""
from __future__ import annotations

import numbers
import numpy as np

_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_TILE_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3), (5, 7), (7, 5))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    parts = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D8:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    stack.append(q)
        parts.append(cells)
    return parts


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _read(input_grid):
    """Recover the unique frame, plan and stencil library from visible pixels."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    assert np.count_nonzero(g == bg) > g.size / 2
    candidates = []
    for frame in map(int, np.unique(g)):
        if frame == bg:
            continue
        cells = list(map(tuple, np.argwhere(g == frame)))
        r0, r1, c0, c1 = _bbox(cells)
        if r1-r0 < 4 or c1-c0 < 4:
            continue
        rim = np.zeros(g.shape, dtype=bool)
        rim[r0, c0:c1] = rim[r1-1, c0:c1] = True
        rim[r0:r1, c0] = rim[r0:r1, c1-1] = True
        if not np.array_equal(g == frame, rim):
            continue
        plan = g[r0+1:r1-1, c0+1:c1-1].copy()
        if not np.any(plan == bg) or len(set(map(int, np.unique(plan))) - {bg}) < 2:
            continue
        outside = g != bg
        outside[r0:r1, c0:c1] = False
        library = {}
        boxes = []
        valid = True
        shape = None
        for part in _components(outside):
            a, b, c, d = _bbox(part)
            patch = g[a:b, c:d].copy()
            h, w = patch.shape
            if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
                valid = False
                break
            if shape is not None and patch.shape != shape:
                valid = False
                break
            center = h//2, w//2
            body_cells = patch != bg
            if not body_cells[center] or int(body_cells.sum()) < 4:
                valid = False
                break
            body_cells[center] = False
            bodies = set(map(int, patch[body_cells]))
            if len(bodies) != 1:
                valid = False
                break
            body = bodies.pop()
            value = int(patch[center])
            if body in (bg, frame) or value in (bg, frame) or body in library:
                valid = False
                break
            # No unrelated foreground can be hidden inside a token's crop.
            if len(part) != int(np.count_nonzero(patch != bg)):
                valid = False
                break
            library[body] = dict(value=value, support=patch != bg, patch=patch,
                                 box=(a, b, c, d))
            shape = patch.shape
            boxes.append((a, b, c, d))
        if not valid or not 3 <= len(library) <= 6:
            continue
        # Every rectangular encoding has a blank moat, even for diagonal supports.
        all_boxes = [(r0, r1, c0, c1)] + boxes
        if any(not _apart(x, y) for i, x in enumerate(all_boxes) for y in all_boxes[:i]):
            continue
        if plan.shape[0]*shape[0] > 30 or plan.shape[1]*shape[1] > 30:
            continue
        candidates.append(dict(background=bg, frame=frame, plan=plan,
                               library=library, tile_shape=shape,
                               plan_box=(r0, r1, c0, c1)))
    assert len(candidates) == 1, "input must have one readable framed plan and library"
    return candidates[0]


def _apart(a, b):
    # Half-open boxes: at least one entirely blank row or column between them.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _transform(a, code):
    return np.fliplr(np.rot90(a, code//2)) if code % 2 else np.rot90(a, code//2)


def _tile(info, name, mode="intended"):
    bg, lib, shape = info['background'], info['library'], info['tile_shape']
    tile = np.full(shape, bg, dtype=np.int8)
    if name == bg or name not in lib:
        return tile
    source = lib[name]
    middle = source['value']
    if middle not in lib:
        if mode == 'keep_failed':
            return source['patch'].copy()
        return tile
    donor = lib[middle]
    a, b = source['support'], donor['support']
    if mode == 'source':
        mask = a
    elif mode == 'donor':
        mask = b
    elif mode == 'union':
        mask = a | b
    elif mode == 'solid':
        mask = np.ones(shape, dtype=bool)
    elif mode.startswith('align_'):
        other = _transform(b, int(mode[6:]))
        if other.shape != shape:
            return np.full(shape, bg, dtype=np.int8)
        mask = a & other
    else:
        mask = a & b
    tile[mask] = name
    center_value = donor['value']
    if mode == 'one_hop':
        center_value = middle
    elif mode == 'three_hop':
        center_value = lib[center_value]['value'] if center_value in lib else bg
    tile[shape[0]//2, shape[1]//2] = center_value
    if mode.startswith('rotate_'):
        return _transform(tile, int(mode[7:])).copy()
    return tile


def _render(info, mode="intended"):
    plan, shape, bg = info['plan'], info['tile_shape'], info['background']
    if mode.startswith('plan_'):
        plan = _transform(plan, int(mode[5:]))
    # A transformed whole tile can change its rectangular block dimensions.
    sample = _tile(info, bg, mode)
    if mode.startswith('rotate_'):
        sample = _transform(sample, int(mode[7:]))
    sh, sw = sample.shape
    out = np.full((plan.shape[0]*sh, plan.shape[1]*sw), bg, dtype=np.int8)
    for r in range(plan.shape[0]):
        for c in range(plan.shape[1]):
            tile = _tile(info, int(plan[r, c]), mode)
            if mode.startswith('rotate_') and tile.shape != (sh, sw):
                tile = _transform(tile, int(mode[7:]))
            out[r*sh:(r+1)*sh, c*sw:(c+1)*sw] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the framed plan and original input stencil relations."""
    return _render(_read(input_grid))


def _connected(mask):
    return len(_components(mask)) == 1


def _support(rng, shape):
    """Sample geometry rather than assigning a fixed support to each identity."""
    h, w = shape
    cr, cc = h//2, w//2
    rr, co = np.indices(shape)
    kind = int(rng.integers(7))
    if kind == 0:
        mask = (rr == cr) | (co == cc)
        mask |= rng.random(shape) < float(rng.uniform(.08, .45))
    elif kind == 1:
        mask = np.ones(shape, dtype=bool)
        for index in rng.permutation(h*w):
            r, c = divmod(int(index), w)
            if (r, c) == (cr, cc):
                continue
            mask[r, c] = False
            if (not _connected(mask) or not all(np.any(side) for side in
                    (mask[0], mask[-1], mask[:, 0], mask[:, -1]))):
                mask[r, c] = True
            if mask.mean() <= float(rng.uniform(.45, .85)):
                break
    elif kind == 2:
        mask = (rr == 0) | (rr == h-1) | (co == 0) | (co == w-1)
        mask[cr, cc:] = True
    elif kind == 3:
        mask = np.zeros(shape, dtype=bool)
        # Two diagonal staircases, including a centered registration point.
        for r in range(h):
            c = int(round(r*(w-1)/(h-1)))
            mask[r, c] = mask[r, w-1-c] = True
        mask[cr, cc] = True
        if not _connected(mask):
            mask[cr, :] = True
    elif kind == 4:
        axis = int(rng.integers(2))
        mask = ((rr % 2 == 0) | (co == cc)) if axis == 0 else ((co % 2 == 0) | (rr == cr))
        mask[cr, cc] = True
    else:
        mask = np.zeros(shape, dtype=bool)
        mask[cr, cc] = True
        targets = [(0, int(rng.integers(w))), (h-1, int(rng.integers(w))),
                   (int(rng.integers(h)), 0), (int(rng.integers(h)), w-1)]
        rng.shuffle(targets)
        for tr, tc in targets:
            r, c = cr, cc
            while (r, c) != (tr, tc):
                choices = []
                if r != tr:
                    choices.append((r + (1 if tr > r else -1), c))
                if c != tc:
                    choices.append((r, c + (1 if tc > c else -1)))
                r, c = choices[int(rng.integers(len(choices)))]
                mask[r, c] = True
        mask |= rng.random(shape) < .12
    # Join any accidental isolated extras with center-directed staircases.
    for part in _components(mask):
        if (cr, cc) in part:
            continue
        r, c = part[0]
        while (r, c) != (cr, cc):
            r += int(np.sign(cr-r))
            c += int(np.sign(cc-c))
            mask[r, c] = True
    return _transform(mask, int(rng.choice([0, 1, 4, 5]))).copy()  # 0/180 and axis reflections


def _witness_supports(rng, shape):
    # Ensure overlap actually uses both footprints and exposes registration.
    for _ in range(240):
        a, b = _support(rng, shape), _support(rng, shape)
        common = a & b
        if (np.any(a & ~b) and np.any(b & ~a) and common.sum() >= 3
                and all(not np.array_equal(common, _transform(common, code)) for code in range(1, 8))):
            return a, b
    # Constructive registration witness for the bounded sampler's rare exhaustion.
    # Expand an asymmetric 3x3 arrangement into rectangular sectors.
    h, w = shape
    cr, cc = h//2, w//2
    common = np.zeros(shape, dtype=bool)
    common[:cr, :cc+1] = True
    common[cr, cc:] = True
    common[cr+1:, :cc] = True
    a, b = common.copy(), common.copy()
    a[:cr, cc+1:] = True
    b[cr+1:, cc+1:] = True
    assert all(not np.array_equal(common, _transform(common, code)) for code in range(1, 8))
    assert np.any(a & ~b) and np.any(b & ~a)
    return a, b


def _pack(rng, rectangles, foreground_count):
    padded_area = sum((h+1)*(w+1) for h, w in rectangles)
    options = [(h, w) for h in range(11, 31) for w in range(11, 31)
               if h*w > 2*foreground_count and (h-2)*(w-2) >= 1.15*padded_area
               and all(a+2 <= h and b+2 <= w for a, b in rectangles)]
    order = sorted(range(len(rectangles)), key=lambda i: -rectangles[i][0]*rectangles[i][1])
    for attempt in range(100):
        h, w = options[int(rng.integers(len(options)))] if attempt < 70 else (30, 30)
        boxes = [None]*len(rectangles)
        for i in order:
            rh, rw = rectangles[i]
            possible = [(r, r+rh, c, c+rw) for r in range(1, h-rh) for c in range(1, w-rw)]
            possible = [b for b in possible if all(q is None or _apart(b, q) for q in boxes)]
            if not possible:
                break
            boxes[i] = possible[int(rng.integers(len(possible)))]
        if all(b is not None for b in boxes):
            return (h, w), boxes
    raise RuntimeError("could not pack stencil library")


def _draw(rng, bg, frame, plan, keys, values, masks):
    sh, sw = masks[0].shape
    rectangle = np.full((plan.shape[0]+2, plan.shape[1]+2), frame, dtype=np.int8)
    rectangle[1:-1, 1:-1] = plan
    patches = []
    for name, value, mask in zip(keys, values, masks):
        patch = np.where(mask, name, bg).astype(np.int8)
        patch[sh//2, sw//2] = value
        patches.append(patch)
    parts = [rectangle] + patches
    shape, boxes = _pack(rng, [p.shape for p in parts], sum(np.count_nonzero(p != bg) for p in parts))
    grid = np.full(shape, bg, dtype=np.int8)
    for patch, (r0, r1, c0, c1) in zip(parts, boxes):
        grid[r0:r1, c0:c1] = patch
    return grid


def _plan(rng, shape, bg, mandatory, pool):
    assert np.prod(shape) >= len(mandatory)
    cells = list(mandatory)
    for _ in range(int(np.prod(shape))-len(cells)):
        cells.append(bg if rng.random() < .3 else int(rng.choice(pool)))
    rng.shuffle(cells)
    return np.array(cells, dtype=np.int8).reshape(shape)


def _make_pair(rng, roles, mode, tile_shape, plan_shape):
    palette = list(map(int, rng.permutation(10))) if roles is None else list(roles)
    bg, frame = palette[:2]
    k = int(rng.integers(5 if mode == 'cycles' else 4, 7))
    keys = palette[2:2+k]
    terminal = palette[2+k]
    a, b, c, d = keys[:4]
    if mode == 'chain':
        values = [b, c, terminal, b]
        mandatory = [a, a, b, c, d, bg]
    elif mode == 'cycles':
        e = keys[4]
        values = [b, a, d, d, terminal]
        mandatory = [a, a, b, c, d, e, bg, terminal]
    else:
        # Arbitrary finite functional relations, including cycles, sinks and forks.
        choices = keys + [terminal]
        values = [int(rng.choice(choices)) for _ in keys]
        values[0] = b
        # Only one successful first edge and one failure are compulsory.
        # b may lead to any key or a terminal: tests need not repeat the chain witness.
        values[2] = terminal
        repeats = int(rng.integers(2, min(4, int(np.prod(plan_shape))-2)+1))
        mandatory = [a]*repeats + [c, bg]
        if len(mandatory) < int(np.prod(plan_shape)) and rng.random() < .5:
            mandatory.append(terminal)
    values += [int(rng.choice(keys + [terminal])) for _ in range(k-len(values))]
    masks = [_support(rng, tile_shape) for _ in keys]
    masks[0], masks[1] = _witness_supports(rng, tile_shape)
    # Ensure the rendered data, not only the relation labels, contain orientation evidence.
    plan = _plan(rng, plan_shape, bg, mandatory, keys + [terminal])
    inp = _draw(rng, bg, frame, plan, keys, values, masks)
    # Construction works from sampled parts; solve/validation re-segment the pixels.
    lib = {name: dict(value=value, support=mask) for name, value, mask in zip(keys, values, masks)}
    out = np.full((plan_shape[0]*tile_shape[0], plan_shape[1]*tile_shape[1]), bg, dtype=np.int8)
    for r, c in np.ndindex(plan.shape):
        name = int(plan[r, c])
        if name not in lib or lib[name]['value'] not in lib:
            continue
        first, second = lib[name], lib[lib[name]['value']]
        tile = np.where(first['support'] & second['support'], name, bg).astype(np.int8)
        tile[tile_shape[0]//2, tile_shape[1]//2] = second['value']
        out[r*tile_shape[0]:(r+1)*tile_shape[0], c*tile_shape[1]:(c+1)*tile_shape[1]] = tile
    return dict(input=inp, output=out)


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _shape_choices(regime, test, index):
    if regime == 'plan_growth':
        tiles = _TILE_SHAPES[:4]
        plans = [(h, w) for h in (range(4, 7) if test else range(2, 4))
                 for w in (range(3, 7) if test else range(3, 5))]
    elif regime == 'stencil_growth':
        tiles = _TILE_SHAPES[3:] if test else _TILE_SHAPES[:3]
        plans = [(h, w) for h in range(2, 5) for w in range(3, 5)]
    else:  # Transpose coarse aspect ratio while independently sampling stencil dimensions.
        tiles = _TILE_SHAPES[:6]
        plans = ([(h, w) for h in range(4, 7) for w in (2, 3)] if test else
                 [(h, w) for h in (2, 3) for w in range(4, 7)])
    minimum_area = 8 if index == 1 and not test else 6
    return [(s, p) for s in tiles for p in plans if p[0]*p[1] >= minimum_area
            and s[0]*p[0] <= 30 and s[1]*p[1] <= 30]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(['plan_growth', 'stencil_growth', 'aspect']))
    roles = tuple(map(int, rng.permutation(10))) if rng.random() < .3 else None
    puzzle = {'train': [], 'test': []}
    seen = set()
    used = []
    for split, n in (('train', nt), ('test', ne)):
        for index in range(n):
            mode = ('chain' if index == 0 else 'cycles' if index == 1 else 'random') if split == 'train' else 'random'
            choices = _shape_choices(regime, split == 'test', index)
            if split == 'train' and index == 1:
                different = [(s, p) for s, p in choices if s != used[0][0] and p != used[0][1]]
                if different:
                    choices = different
            tile_shape, plan_shape = choices[int(rng.integers(len(choices)))]
            for _ in range(24):
                pair = _make_pair(rng, roles, mode, tile_shape, plan_shape)
                if mode == 'chain':
                    info = _read(pair['input'])
                    # Pin both spatial levels and donor registration in training.
                    if any(np.array_equal(pair['output'], _render(info, kind+str(code)))
                           for kind in ('plan_', 'align_') for code in range(1, 8)):
                        continue
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    used.append((tile_shape, plan_shape))
                    break
            else:
                raise RuntimeError("could not sample a distinct input")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    bg, lib, plan = info['background'], info['library'], info['plan']
    selected = set(map(int, plan.ravel())) - {bg}
    live = [x for x in selected if x in lib and lib[x]['value'] in lib]
    return dict(
        chain=any(lib[x]['value'] in live and lib[lib[x]['value']]['value'] !=
                  lib[lib[lib[x]['value']]['value']]['value']
                  for x in live if lib[lib[x]['value']]['value'] in lib),
        cycle=any(lib[lib[x]['value']]['value'] == x and lib[x]['value'] != x for x in live),
        fixed=any(lib[x]['value'] == x for x in live),
        unchanged=any(lib[lib[x]['value']]['value'] == lib[x]['value'] for x in live),
        failed=any(x not in lib or lib[x]['value'] not in lib for x in selected),
        repeat=any(np.count_nonzero(plan == x) >= 2 for x in live),
        overlap=any(np.any(lib[x]['support'] & ~lib[lib[x]['value']]['support'])
                    and np.any(lib[lib[x]['value']]['support'] & ~lib[x]['support']) for x in live),
        holes=any(np.any(~v['support']) for v in lib.values()),
        keys=len(lib), tile_shape=info['tile_shape'], plan_shape=plan.shape,
        bg=bg, frame=info['frame'],
    )


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            infos[split].append(info)
            assert out.shape == tuple(x*y for x, y in zip(info['plan'].shape, info['tile_shape']))
            assert np.array_equal(out, _render(info)), "wrong registered overlap or relation composition"
            sh, sw = info['tile_shape']
            bg, lib = info['background'], info['library']
            assert info['frame'] not in out
            for r, c in np.ndindex(info['plan'].shape):
                name = int(info['plan'][r, c])
                block = out[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                if name not in lib or lib[name]['value'] not in lib:
                    assert np.all(block == bg)
                else:
                    donor = lib[lib[name]['value']]
                    common = lib[name]['support'] & donor['support']
                    assert np.array_equal(block != bg, common)
                    assert int(block[sh//2, sw//2]) == donor['value']
                    body = common.copy()
                    body[sh//2, sw//2] = False
                    assert np.all(block[body] == name)
    train = infos['train']
    features = [_features(i) for i in train]
    for flag in ('chain', 'cycle', 'fixed', 'unchanged', 'failed', 'repeat', 'overlap', 'holes'):
        assert any(f[flag] for f in features), f"missing training witness: {flag}"
    assert len({i['plan'].shape for i in train}) >= 2
    assert len({i['tile_shape'] for i in train}) >= 2
    # Actual training answers must refute these tempting simpler mechanisms.
    for mode in ('source', 'donor', 'union', 'solid', 'one_hop', 'three_hop', 'keep_failed'):
        assert any(not np.array_equal(_render(i), _render(i, mode)) for i in train), mode
    for code in range(1, 8):
        for kind in ('rotate_', 'align_', 'plan_'):
            mode = kind + str(code)
            # Ambiguities are harmless only when they also leave every test answer unchanged.
            if all(np.array_equal(_render(i), _render(i, mode)) for i in train):
                assert all(np.array_equal(_render(i), _render(i, mode)) for i in infos['test']), mode


def _canonical_plan(plan, bg):
    names = {bg: 0}
    return plan.shape, tuple(names.setdefault(int(x), len(names)) for x in plan.ravel())


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded semantic, counterfactual, count and 42-seed diversity checks."""
    # Hand-calculated three-key case, independent of sampled output construction.
    masks = [np.array([[1,1,1],[0,1,0],[1,1,0]], dtype=bool),
             np.array([[0,1,1],[1,1,1],[0,1,0]], dtype=bool),
             np.array([[1,1,0],[1,1,1],[0,1,1]], dtype=bool)]
    fixture = _draw(np.random.default_rng(1701), 0, 5,
                    np.array([[1,0,2],[3,1,4]], dtype=np.int8), [1,2,3], [2,3,4], masks)
    expected = np.array([[int(c) for c in row] for row in
                         ('011000020','030000242','010000020',
                          '000011000','000030000','000010000')], dtype=np.int8)
    assert np.array_equal(solve(fixture), expected)
    records, signatures, puzzles = [], set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read(pair['input'])
                f = _features(info)
                f.update(input_shape=pair['input'].shape, output_shape=pair['output'].shape,
                         support_pattern=tuple(v['support'].tobytes() for v in info['library'].values()),
                         plan_pattern=_canonical_plan(info['plan'], info['background']))
                records.append(f)
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len(signatures) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for feature, minimum in (('input_shape', 50), ('output_shape', 25), ('tile_shape', 7),
                             ('plan_shape', 15), ('support_pattern', 150), ('plan_pattern', 180)):
        assert len({f[feature] for f in records}) >= minimum, feature
    assert {4, 5, 6} <= {f['keys'] for f in records}
    assert len({f['bg'] for f in records}) == 10
    assert len({f['frame'] for f in records}) == 10
    assert any(30 in f['output_shape'] for f in records)
    # The external verifier exhausts all 30 count combinations. Keep this
    # local check bounded to the count boundaries alongside the 42 seeds.
    for nt, ne in ((2, 1), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(9000+nt*10+ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid counts accepted')
    bad = {s: [dict(input=p['input'].copy(), output=p['output'].copy()) for p in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['train'][0]['output'][0, 0] = (int(bad['train'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                tile_shapes=len({f['tile_shape'] for f in records}),
                plan_shapes=len({f['plan_shape'] for f in records}),
                input_shapes=len({f['input_shape'] for f in records}),
                output_shapes=len({f['output_shape'] for f in records}),
                support_patterns=len({f['support_pattern'] for f in records}))


if __name__ == '__main__':
    print(self_test())
