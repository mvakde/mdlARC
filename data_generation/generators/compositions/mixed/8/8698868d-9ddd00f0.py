"""Count-selected, bordered motifs with macro-coordinate carrier overwrites.

Only NumPy and the standard library are used. No filesystem access, parent
imports, saved examples, random state, or seed is involved in solve().
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    parts = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                other = (r + dr, c + dc)
                if other in unseen:
                    unseen.remove(other)
                    stack.append(other)
        parts.append(cells)
    return parts


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _edge(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions must be 1..30')
    _need(np.issubdtype(a.dtype, np.integer), 'grid must have integer colors')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'colors must be 0..9')
    return a


def _interpret(grid, bg):
    parts = sorted(_components(grid != bg), key=len, reverse=True)
    _need(len(parts) >= 3 and len(parts[0]) > len(parts[1]), 'missing unique largest mosaic')
    boxes = [_box(cells) for cells in parts]
    for i, (r, c, h, w) in enumerate(boxes):
        for a, b, x, y in boxes[i + 1:]:
            _need(r + h < a or a + x < r or c + w < b or b + y < c,
                  'objects need a neutral gap between bounding boxes')
    shapes = {(h, w) for _, _, h, w in boxes[1:]}
    _need(len(shapes) == 1, 'motif sizes differ')
    s, sw = next(iter(shapes))
    _need(s == sw and s >= 4, 'motifs must be squares of side at least four')
    t = s + 2
    mr, mc, mh, mw = boxes[0]
    _need(mh % t == 0 and mw % t == 0, 'mosaic is not partitionable')
    rows, cols = mh // t, mw // t
    _need(2 <= rows * cols and rows <= s and cols <= s, 'invalid macro coordinate range')
    _need(len(boxes) - 1 <= rows * cols, 'unused source motifs')
    motifs, motif_boxes = {}, {}
    # Reading order is used only for an explicitly tested competing rule.
    for r, c, h, w in sorted(boxes[1:]):
        motif = grid[r:r + h, c:c + w]
        color = int(motif[0, 0])
        _need(color != bg and bool(np.all(_edge(motif) == color)), 'incomplete motif perimeter')
        _need(set(map(int, np.unique(motif))) <= {bg, color}, 'nonbinary motif')
        k = len(_components(motif == bg))
        _need(k not in motifs, 'tied motif cavity counts')
        motifs[k], motif_boxes[k] = motif.copy(), (r, c, h, w)
    tiles, used = [], set()
    for r in range(rows):
        for c in range(cols):
            tile = grid[mr + r*t:mr + (r+1)*t, mc + c*t:mc + (c+1)*t]
            carrier = int(tile[0, 0])
            _need(carrier != bg and bool(np.all(_edge(tile) == carrier)), 'incomplete carrier border')
            _need(set(map(int, np.unique(tile))) <= {bg, carrier}, 'nonbinary carrier tile')
            clue = tile[1:-1, 1:-1] == bg
            _need(not np.any(clue[1:] & clue[:-1]) and not np.any(clue[:, 1:] & clue[:, :-1]),
                  'clue cells are not four-isolated')
            k = int(clue.sum())
            _need(k in motifs, 'tile has no count-matched motif')
            _need(carrier != int(motifs[k][0, 0]), 'matched motif invisible against carrier')
            used.add(k)
            tiles.append((r, c, carrier, k, clue.copy()))
    _need(used == set(motifs), 'not every motif is used')
    return {'bg': bg, 's': s, 't': t, 'rows': rows, 'cols': cols,
            'box': boxes[0], 'motifs': motifs, 'motif_boxes': motif_boxes, 'tiles': tiles}


def _decode(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for bg in map(int, np.unique(grid)):
        try:
            candidates.append(_interpret(grid, bg))
        except ValueError:
            pass
    _need(len(candidates) == 1, 'input needs one valid scene interpretation')
    return candidates[0]


def _render(info, mode='identity', assignment=None, transform=None, solid=False):
    """Reconstruction plus explicit ablation/alternative-rule support for tests."""
    s, t, bg = info['s'], info['t'], info['bg']
    out = np.empty((info['rows'] * t, info['cols'] * t), dtype=np.int8)
    for r, c, carrier, k, _ in info['tiles']:
        source = info['motifs'][k] if assignment is None else assignment[k]
        if transform is not None:
            source = transform(source)
        tile = np.full((t, t), carrier, dtype=np.int8)
        tile[1:-1, 1:-1] = int(source[0, 0]) if solid else np.where(source == bg, carrier, source)
        if mode != 'none':
            p, q = r, c
            if mode == 'transpose':
                p, q = c, r
            elif mode == 'flip_row':
                p = info['rows'] - 1 - r
            elif mode == 'flip_col':
                q = info['cols'] - 1 - c
            elif mode == 'center':
                p = q = s // 2
            elif mode == 'top_left':
                p = q = 0
            elif mode == 'row_only':
                q = 0
            elif mode == 'col_only':
                p = 0
            elif mode == 'canvas':
                p, q = (info['box'][0] + r) % s, (info['box'][1] + c) % s
            elif mode == 'relocate' and int(source[p, q]) == bg:
                # Wrong tie rule: skip a cavity to find the next foreground cell.
                for offset in range(1, s*s + 1):
                    a, b = divmod((p*s + q + offset) % (s*s), s)
                    if int(source[a, b]) != bg:
                        p, q = a, b
                        break
            elif mode not in ('identity', 'relocate', 'tile_origin'):
                raise ValueError('unknown punch mode')
            if mode == 'tile_origin':
                tile[r, c] = carrier
            else:
                tile[1 + p, 1 + q] = carrier
        out[r*t:(r+1)*t, c*t:(c+1)*t] = tile
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output cell from the input's visible objects and coordinates."""
    return _render(_decode(input_grid))


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in _transforms(mask)[1:])


def _capacity(s):
    return min(6, {2: 2, 3: 5, 4: 7}.get(s - 2, 8))


def _independent(rng, n, k):
    """Four-isolated pixels with a connected, padded complement."""
    if k == 0:
        return np.zeros((n, n), dtype=bool)
    for _ in range(20):
        mask = np.zeros((n, n), dtype=bool)
        for index in rng.permutation(n*n):
            r, c = divmod(int(index), n)
            if any(0 <= r+dr < n and 0 <= c+dc < n and mask[r+dr, c+dc] for dr, dc in _STEPS):
                continue
            mask[r, c] = True
            if len(_components(np.pad(~mask, 1, constant_values=True))) != 1:
                mask[r, c] = False
            if int(mask.sum()) == k:
                return mask
    if n <= 3:
        cells = [(r, c) for r in range(n) for c in range(n) if (r+c) % 2 == 0]
    elif n == 4:
        cells = [(0, 0), (0, 2), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
    else:
        cells = [(r, c) for r in range(0, n, 2) for c in range(0, n, 2)]
    _need(k <= len(cells), 'independent-set capacity exceeded')
    mask = np.zeros((n, n), dtype=bool)
    for i in rng.permutation(len(cells))[:k]:
        mask[cells[int(i)]] = True
    _need(len(_components(np.pad(~mask, 1, constant_values=True))) == 1, 'invalid fallback')
    return mask


def _cavity_mask(rng, s, k):
    n = s - 2
    if k == 0:
        return np.zeros((s, s), dtype=bool)
    mode = int(rng.integers(5))
    if k == 1 and mode <= 1:
        return np.pad(np.ones((n, n), dtype=bool), 1)
    if 2*k - 1 <= n and mode == 2:
        # Disjoint bars or rectangles, with variable length and thickness.
        inner = np.zeros((n, n), dtype=bool)
        rows = sorted(map(int, rng.choice(n - k + 1, k, replace=False)))
        rows = [r + i for i, r in enumerate(rows)]
        for r in rows:
            a = int(rng.integers(n))
            b = int(rng.integers(a + 1, n + 1))
            inner[r, a:b] = True
        return _pick(rng, _transforms(np.pad(inner, 1))).copy()
    mask = np.pad(_independent(rng, n, k), 1)
    if mode == 0:
        return mask
    target = int(rng.integers(k, n*n + 1))
    # At most n*n passes of at most n*n trial cells; every accepted addition
    # preserves both cavity count and foreground connectivity.
    for _ in range(n*n):
        if int(mask.sum()) >= target:
            break
        choices = []
        for r in range(1, s - 1):
            for c in range(1, s - 1):
                if mask[r, c] or not any(mask[r+dr, c+dc] for dr, dc in _STEPS):
                    continue
                trial = mask.copy()
                trial[r, c] = True
                if len(_components(trial)) == k and len(_components(~trial)) == 1:
                    choices.append((r, c))
        if not choices:
            break
        mask[_pick(rng, choices)] = True
    return mask


def _free_origins(boxes, h, w):
    rr, cc = np.indices((31 - h, 31 - w))
    free = np.ones(rr.shape, dtype=bool)
    for r, c, bh, bw in boxes:
        free &= (rr + h < r) | (rr > r + bh) | (cc + w < c) | (cc > c + bw)
    return np.argwhere(free)


def _packing(t, rows, cols, number):
    if rows*t > 30 or cols*t > 30:
        return None
    boxes = [(0, 0, rows*t, cols*t)]
    for _ in range(number):
        free = _free_origins(boxes, t - 2, t - 2)
        if not len(free):
            return None
        r, c = map(int, free[0])
        boxes.append((r, c, t - 2, t - 2))
    return boxes


# This small catalog is of feasible dimensions, not saved masks or examples.
_PACKINGS = {}
for _t in range(6, 10):
    for _r in range(1, 5):
        for _c in range(1, 5):
            for _n in range(2, min(4, _r*_c, _capacity(_t - 2) + 1) + 1):
                _p = _packing(_t, _r, _c, _n)
                if _p is not None:
                    _PACKINGS[(_t, _r, _c, _n)] = _p


def _place(rng, bg, mosaic, motifs, config, anchor):
    t, rows, cols, number = config
    s, h, w = t - 2, rows*t, cols*t
    boxes = None
    for _ in range(20):
        r, c = int(rng.integers(31 - h)), int(rng.integers(31 - w))
        trial = [(r, c, h, w)]
        for _ in range(number):
            free = _free_origins(trial, s, s)
            if not len(free):
                break
            a, b = map(int, free[int(rng.integers(len(free)))])
            trial.append((a, b, s, s))
        if len(trial) == number + 1:
            boxes = trial
            break
    if boxes is None:
        boxes = _PACKINGS[config]
    keys = [int(k) for k in rng.permutation(list(motifs))]
    reading = sorted(range(number), key=lambda i: boxes[i + 1][:2])
    if anchor and [keys[i] for i in reading] == sorted(keys):
        keys = keys[1:] + keys[:1]
    canvas = np.full((30, 30), bg, dtype=np.int8)
    for (r, c, bh, bw), obj in zip(boxes, [mosaic] + [motifs[k] for k in keys]):
        canvas[r:r+bh, c:c+bw] = obj
    r0, c0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    r1, c1 = max(b[0]+b[2] for b in boxes), max(b[1]+b[3] for b in boxes)
    # Some full workspaces; otherwise independent margins, including zero.
    if rng.random() < 0.18:
        a, b, d, e = 0, 0, 30, 30
    else:
        a, b = int(rng.integers(r0 + 1)), int(rng.integers(c0 + 1))
        d, e = int(rng.integers(r1, 31)), int(rng.integers(c1, 31))
    if anchor and (boxes[0][0] - a) % s == (boxes[0][1] - b) % s == 0:
        # A rare crop can align canvas and mosaic coordinate origins. The
        # proven anchor packing has room for a neutral top row, making the
        # origin distinction visible without unbounded resampling.
        boxes = [(r + 1, c, bh, bw) for r, c, bh, bw in _PACKINGS[config]]
        d = max(r + bh for r, _, bh, _ in boxes)
        e = max(c + bw for _, c, _, bw in boxes)
        _need(d <= 30 and e <= 30, 'anchor origin fallback exceeds canvas')
        reading = sorted(range(number), key=lambda i: boxes[i + 1][:2])
        if [keys[i] for i in reading] == sorted(keys):
            keys = keys[1:] + keys[:1]
        canvas = np.full((d, e), bg, dtype=np.int8)
        for (r, c, bh, bw), obj in zip(boxes, [mosaic] + [motifs[k] for k in keys]):
            canvas[r:r+bh, c:c+bw] = obj
        return canvas
    return canvas[a:d, b:e].copy()


def _make_pair(rng, config, bg, anchor=0):
    t, rows, cols, number = config
    s, cap = t - 2, _capacity(t - 2)
    keys = [0, 1, 2] if anchor else sorted(map(int, rng.choice(cap + 1, number, replace=False)))
    if anchor:
        order = np.array([[1, 0, 2], [0, 1, 2]], dtype=int)
        if rows == 3:
            order = order.T
        order = order.ravel().tolist()
    else:
        order = keys + [int(_pick(rng, keys)) for _ in range(rows*cols - number)]
        rng.shuffle(order)
        # A source cavity must survive somewhere in the top tile row, where the
        # punch lies on the motif perimeter and cannot erase that cavity.
        if not any(order[:cols]):
            j = next(i for i, k in enumerate(order) if k)
            order[0], order[j] = order[j], order[0]
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    pool = palette[:int(rng.integers(2, 10))]
    masks, motifs = {}, {}
    for k in keys:
        mask = _cavity_mask(rng, s, k)
        if anchor == 1 and k == 1:
            mask = np.pad(np.ones((s - 2, s - 2), dtype=bool), 1)
        elif anchor == 1 and k == 2:
            mask = np.zeros((s, s), dtype=bool)
            mask[1, 1] = mask[2, 2] = mask[2, 3] = True
            mask = _pick(rng, _transforms(mask)).copy()
            for _ in range(8):
                candidate = _cavity_mask(rng, s, 2)
                if (int(candidate.sum()) < (s - 2)**2 and _asymmetric(candidate)
                        and len(_components(candidate, True)) == 1):
                    mask = candidate
                    break
        color = _pick(rng, pool[1:] if anchor == 1 else pool)
        motif = np.full((s, s), color, dtype=np.int8)
        motif[mask] = bg
        masks[k], motifs[k] = mask, motif
    mosaic = np.empty((rows*t, cols*t), dtype=np.int8)
    output = np.empty_like(mosaic)
    for index, k in enumerate(order):
        r, c = divmod(index, cols)
        carrier = pool[0] if anchor == 1 else _pick(rng, [x for x in pool if x != int(motifs[k][0, 0])])
        clues = _independent(rng, s, k)
        tile = np.full((t, t), carrier, dtype=np.int8)
        tile[1:-1, 1:-1][clues] = bg
        mosaic[r*t:(r+1)*t, c*t:(c+1)*t] = tile
        # Labels are constructed from generation objects, never by solve().
        tile[:] = carrier
        tile[1:-1, 1:-1] = np.where(masks[k], carrier, motifs[k])
        tile[1+r, 1+c] = carrier
        output[r*t:(r+1)*t, c*t:(c+1)*t] = tile
    return {'input': _place(rng, bg, mosaic, motifs, config, bool(anchor)), 'output': output}


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fresh_recolor(pair, used):
    # At least three colors occur. The first two color images alone offer 90
    # alternatives, exceeding the 13 previous inputs in any legal request.
    present = list(map(int, np.unique(pair['input'])))
    order = present[:2] + [x for x in range(10) if x not in present[:2]]
    for a in range(10):
        for b in range(10):
            if a == b:
                continue
            target = [a, b] + [x for x in range(10) if x not in (a, b)]
            lookup = np.empty(10, dtype=np.int8)
            lookup[order] = target
            candidate = {role: lookup[grid] for role, grid in pair.items()}
            if _key(candidate['input']) not in used:
                return candidate
    raise RuntimeError('distinct-input recoloring bound violated')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    varied_bg, base_bg = rng.random() < 0.7, int(rng.integers(10))
    shape = _pick(rng, [(2, 3), (3, 2)])
    first_t = int(rng.choice([7, 8]))
    second_t = int(_pick(rng, [t for t in (6, 7) if t != first_t]))
    anchors = [(first_t, *shape, 3), (second_t, shape[1], shape[0], 3)]
    if regime == 0:
        train_pool = [x for x in _PACKINGS if max(x[1:3]) <= 3]
        test_pool = [x for x in _PACKINGS if max(x[1:3]) == 4]
    elif regime == 1:
        train_pool = [x for x in _PACKINGS if max(x[1:3]) <= 3 and x[1:3] != (3, 3)]
        test_pool = [x for x in _PACKINGS if x[1:3] == (3, 3)]
    else:
        train_pool = [x for x in _PACKINGS if x[0] <= 8]
        test_pool = [x for x in _PACKINGS if x[0] == 9]
    puzzle, used = {'train': [], 'test': []}, set()
    for split, number in (('train', nt), ('test', ne)):
        for i in range(number):
            config = anchors[i] if split == 'train' and i < 2 else _pick(rng, train_pool if split == 'train' else test_pool)
            anchor = i + 1 if split == 'train' and i < 2 else 0
            bg = int(rng.integers(10)) if varied_bg else base_bg
            pair = _make_pair(rng, config, bg, anchor)
            if _key(pair['input']) in used:
                pair = _fresh_recolor(pair, used)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def _facts(info, output):
    expected = _render(info)
    _need(np.array_equal(output, expected), 'output violates matched inset and coordinate punch')
    raw = _render(info, 'none')
    changed, collisions = 0, 0
    for r, c, carrier, k, _ in info['tiles']:
        t = info['t']
        before = raw[r*t:(r+1)*t, c*t:(c+1)*t]
        after = output[r*t:(r+1)*t, c*t:(c+1)*t]
        residual = np.argwhere(before != after)
        overlap = int(info['motifs'][k][r, c]) == info['bg']
        _need(len(residual) == int(not overlap), 'punch residual is not exactly one pixel or collision no-op')
        if len(residual):
            _need(tuple(residual[0]) == (1+r, 1+c), 'punch has wrong coordinate or inset offset')
            changed += 1
        else:
            collisions += 1
        _need(bool(np.all(_edge(after) == carrier)), 'carrier border changed')
    _need(not np.array_equal(raw, output), 'parent B coordinate concept has no effect')
    _need(not np.array_equal(_render(info, solid=True), output), 'parent A cavity concept has no effect')
    areas = {k: int((m == info['bg']).sum()) for k, m in info['motifs'].items()}
    patterns = [m == info['bg'] for m in info['motifs'].values()]
    return {'changed': changed, 'collisions': collisions,
            'inversion': any(a < b and areas[a] > areas[b] for a in areas for b in areas),
            'diagonal': any(len(_components(m, True)) < len(_components(m)) for m in patterns),
            'asymmetric': any(_asymmetric(m) for m in patterns),
            'nonpositional': any(k and not np.array_equal(clue, info['motifs'][k] == info['bg'])
                                 for _, _, _, k, clue in info['tiles']),
            'areas': areas}


def _training(infos, facts):
    for name in ('inversion', 'diagonal', 'asymmetric', 'nonpositional'):
        _need(any(a[name] for a in facts), 'missing training witness: ' + name)
    _need(any(a['collisions'] for a in facts), 'overlap tie rule not demonstrated')
    _need(max(a['rows'] for a in infos) >= 3 and max(a['cols'] for a in infos) >= 3,
          'training does not expose three coordinates on both axes')
    _need(any(len({(carrier, k) for _, _, carrier, k, _ in a['tiles']}) < len(a['tiles']) for a in infos),
          'no reused motif/carrier addresses in training')
    modes = ('none', 'transpose', 'flip_row', 'flip_col', 'center', 'top_left',
             'row_only', 'col_only', 'tile_origin', 'canvas', 'relocate')
    for mode in modes:
        _need(any(not np.array_equal(_render(a, mode), _render(a)) for a in infos),
              'training permits competing punch rule: ' + mode)
    for transform in (lambda m: np.rot90(m), np.flipud, np.fliplr):
        _need(any(not np.array_equal(_render(a, transform=transform), _render(a)) for a in infos),
              'training does not demonstrate source orientation')
    area_wrong = reading_wrong = False
    for a in infos:
        keys = sorted(a['motifs'])
        area_order = sorted(a['motifs'], key=lambda k: (int((a['motifs'][k] == a['bg']).sum()), k))
        assignment = dict(zip(keys, [a['motifs'][k] for k in area_order]))
        area_wrong |= not np.array_equal(_render(a, assignment=assignment), _render(a))
        assignment = dict(zip(keys, a['motifs'].values()))
        reading_wrong |= not np.array_equal(_render(a, assignment=assignment), _render(a))
    _need(area_wrong and reading_wrong, 'area/read-order matching shortcut remains viable')


def validate_generated_puzzle(puzzle):
    """Check returned grids, semantic invariants, and whole-training witnesses."""
    try:
        _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'expected exactly train/test')
        _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
        _counts(len(puzzle['train']), len(puzzle['test']))
        seen, infos, facts = set(), [], []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'expected exactly input/output')
                inp, out = _grid(pair['input']), _grid(pair['output'])
                _need(_key(inp) not in seen, 'duplicate input')
                seen.add(_key(inp))
                info = _decode(inp)
                f = _facts(info, out)
                if split == 'train':
                    infos.append(info)
                    facts.append(f)
        _training(infos, facts)
    except (ValueError, TypeError, IndexError) as error:
        raise AssertionError(str(error)) from error


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; keep aggregates/digests, never save grids or traces."""
    shapes, layouts, patterns, count_sets, palettes, signatures = set(), set(), set(), set(), set(), set()
    cavity_sizes, address_effects, regimes = set(), set(), set()
    pair_total = punch_total = collision_total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        _need(_digest(puzzle) == _digest(generate_puzzle(seed)), 'nondeterminism')
        train_infos = [_decode(p['input']) for p in puzzle['train']]
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = pair['input'].copy()
                prediction = solve(original)
                _need(np.array_equal(prediction, pair['output']), 'input-only solve disagrees')
                _need(np.array_equal(original, pair['input']), 'solve mutated input')
                a = _decode(pair['input'])
                f = _facts(a, pair['output'])
                pair_total += 1
                punch_total += f['changed']
                collision_total += f['collisions']
                shapes.add(pair['input'].shape)
                layouts.add((a['t'], a['rows'], a['cols'], len(a['motifs'])))
                count_sets.add(tuple(sorted(a['motifs'])))
                palettes.add(tuple(sorted(map(int, np.unique(pair['input'])))))
                for mask in (m == a['bg'] for m in a['motifs'].values()):
                    patterns.add((mask.shape, min(v.tobytes() for v in _transforms(mask))))
                    cavity_sizes.update(map(len, _components(mask)))
                for r, c, _, k, _ in a['tiles']:
                    address_effects.add((r, c, bool(a['motifs'][k][r, c] == a['bg'])))
                if split == 'test':
                    if max(a['rows'], a['cols']) == 4 and all(max(x['rows'], x['cols']) <= 3 for x in train_infos):
                        regimes.add('new_index')
                    if (a['rows'], a['cols']) == (3, 3) and all((x['rows'], x['cols']) != (3, 3) for x in train_infos):
                        regimes.add('new_layout')
                    if a['t'] > max(x['t'] for x in train_infos):
                        regimes.add('new_scale')
        # A relabeling must commute with solving, including nonzero backgrounds.
        lookup = np.random.default_rng(seed + 10000).permutation(10)
        pair = puzzle['test'][0]
        _need(np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']]), 'palette equivariance failed')
    _need(len(signatures) == 42, 'whole puzzles repeated')
    _need(len(shapes) >= 20 and len(layouts) >= 20, 'geometry variation too small')
    _need(len(patterns) >= 40 and len(count_sets) >= 10, 'motif/count variation too small')
    _need(max(cavity_sizes) >= 16 and len(cavity_sizes) >= 8, 'cavity area range too small')
    _need(len(palettes) >= 20 and len(address_effects) >= 15 and len(regimes) == 3, 'roles/addresses/shifts lack variation')
    # Every legal explicit count combination; no storage accumulates across calls.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(20261023, nt, ne)
            _need((len(p['train']), len(p['test'])) == (nt, ne), 'explicit count mismatch')
            validate_generated_puzzle(p)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A legal hand-constructed member outside the sampler's size catalog:
    # side-ten tiles and a 36-cell cavity. solve must follow the specification,
    # rather than membership in _PACKINGS or the sampler's count/size limits.
    ring = np.full((8, 8), 4, dtype=np.int8)
    ring[1:-1, 1:-1] = 9
    bars = np.full((8, 8), 6, dtype=np.int8)
    bars[1, 1:4] = bars[4, 4:7] = 9
    handmade = np.full((19, 20), 9, dtype=np.int8)
    handmade[:10] = 1
    handmade[3, 3] = 9
    handmade[3, 13] = handmade[6, 16] = 9
    handmade[11:19, :8] = bars
    handmade[11:19, 9:17] = ring
    expected = np.ones((10, 20), dtype=np.int8)
    expected[1:9, 1:9] = np.where(ring == 9, 1, ring)
    expected[1:9, 11:19] = np.where(bars == 9, 1, bars)
    expected[1, 1] = expected[1, 12] = 1
    _need(np.array_equal(solve(handmade), expected), 'unsampled legal scale not supported')
    p = generate_puzzle(0, 2, 1)
    validate_generated_puzzle(p)
    # Bad punches, bad labels, lost neutral clues, and tied source keys must fail.
    for corruption in ('label', 'no_punch', 'tie', 'missing_clue'):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = broken['train'][0]
        a = _decode(pair['input'])
        if corruption == 'label':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif corruption == 'no_punch':
            pair['output'] = _render(a, 'none')
        elif corruption == 'tie':
            keys = list(a['motifs'])
            r, c, h, w = a['motif_boxes'][keys[0]]
            pair['input'][r:r+h, c:c+w] = a['motifs'][keys[1]]
        else:
            r, c, carrier, _, clue = next(x for x in a['tiles'] if x[3] > 0)
            cr, cc = map(int, np.argwhere(clue)[0])
            mr, mc = a['box'][:2]
            pair['input'][mr+r*a['t']+1+cr, mc+c*a['t']+1+cc] = carrier
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('corruption accepted: ' + corruption)
    return {'seeds': 42, 'pairs': pair_total, 'distinct_puzzles': len(signatures),
            'input_shapes': len(shapes), 'layout_size_motif_combinations': len(layouts),
            'dihedral_cavity_masks': len(patterns), 'count_sets': len(count_sets),
            'palettes': len(palettes), 'cavity_area_values': sorted(cavity_sizes),
            'effective_punches': punch_total, 'overlap_no_ops': collision_total,
            'train_test_regimes': sorted(regimes), 'independent_solver': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
