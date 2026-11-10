"""Mirror census window: restore geometry, count cavity reflection orbits, crop.

Only 8 (erase window) and 9 (legend separator) have fixed literal meanings.
All geometry and the count-to-color dictionary are available in each input.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while queue:
            x, y = queue.popleft()
            cells.append((x, y))
            for a, b in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        result.append(frozenset(cells))
    return result


def _holes(cells):
    """Global coordinates of bounded complement components of one object.

    The extra exterior ring makes canvas-edge contact topologically harmless.
    """
    r0 = min(r for r, c in cells)
    r1 = max(r for r, c in cells)
    c0 = min(c for r, c in cells)
    c1 = max(c for r, c in cells)
    patch = np.zeros((r1-r0+3, c1-c0+3), dtype=bool)
    for r, c in cells:
        patch[r-r0+1, c-c0+1] = True
    holes = []
    for component in _components(~patch):
        if any(r in (0, patch.shape[0]-1) or c in (0, patch.shape[1]-1)
               for r, c in component):
            continue
        holes.append(frozenset((r+r0-1, c+c0-1) for r, c in component))
    return holes


def _hole_count(mask):
    cells = frozenset(map(tuple, np.argwhere(mask).tolist()))
    return len(_holes(cells)) if cells else 0


def _mirror(cells, width):
    return frozenset((r, width-1-c) for r, c in cells)


def _groups(scene, bg, wall):
    """Group actual components and actual cavities by the same reflection."""
    assert set(np.unique(scene)) <= {bg, wall}
    assert np.array_equal(scene, scene[:, ::-1]), 'Completed field is not mirrored'
    components = _components(scene == wall)
    index = {cells: i for i, cells in enumerate(components)}
    done = set()
    groups = []
    width = scene.shape[1]
    for i, cells in enumerate(components):
        if i in done:
            continue
        mate = _mirror(cells, width)
        assert mate in index, 'Reflection must map a whole connected component'
        j = index[mate]
        ids = sorted({i, j})
        done.update(ids)
        holes = [hole for k in ids for hole in _holes(components[k])]
        hole_set = set(holes)
        assert all(_mirror(hole, width) in hole_set for hole in holes)
        assert all(scene[r, c] == bg for hole in holes for r, c in hole), 'Nested objects'
        orbit_keys = {min(tuple(sorted(hole)), tuple(sorted(_mirror(hole, width))))
                      for hole in holes}
        fixed = sum(_mirror(hole, width) == hole for hole in holes)
        group_cells = frozenset().union(*(components[k] for k in ids))
        groups.append(dict(cells=group_cells, parts=[components[k] for k in ids],
                           holes=holes, count=len(orbit_keys), physical=len(holes),
                           fixed=fixed, paired=(len(holes)-fixed)//2))
    return groups


def _paint(scene, bg, wall, mapping, mode='orbit'):
    output = np.full(scene.shape, bg, dtype=np.int8)
    groups = _groups(scene, bg, wall)
    for group in groups:
        if mode == 'ordinary':
            for cells in group['parts']:
                color = mapping.get(len(_holes(cells)), bg)
                for r, c in cells:
                    output[r, c] = color
        else:
            count = group['count'] if mode == 'orbit' else (group['physical']+1)//2
            color = mapping.get(count, bg)
            for r, c in group['cells']:
                output[r, c] = color
    return output, groups


def _grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and all(1 <= n <= 30 for n in array.shape)
    assert np.issubdtype(array.dtype, np.integer)
    assert np.all((array >= 0) & (array <= 9))
    return array.astype(np.int8, copy=True)


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        work = np.rot90(a, rotation).copy()
        for line in np.flatnonzero(np.all(work == 9, axis=1)):
            line = int(line)
            if line < 4 or line >= work.shape[0]-2:
                continue
            panel, scene = work[:line], work[line+1:]
            if np.any(panel == 8) or not np.any(scene == 8):
                continue
            try:
                bg = int(panel[0, 0])
                assert bg not in (8, 9)
                assert np.all(panel[0] == bg) and np.all(panel[-1] == bg)
                assert not np.any(panel == 9) and not np.any(scene == 9)
                mapping, order = {}, []
                for color in np.unique(panel):
                    color = int(color)
                    if color == bg:
                        continue
                    cells = _components(panel == color)
                    assert len(cells) == 1
                    count = len(_holes(cells[0]))
                    assert count not in mapping
                    mapping[count] = color
                    order.append((min(c for r, c in cells[0]), count))
                assert mapping
                colors = set(map(int, np.unique(scene))) - {bg, 8}
                assert len(colors) == 1
                wall = colors.pop()
                assert wall not in mapping.values()
                mask = scene == 8
                rc = np.argwhere(mask)
                r0, c0 = map(int, rc.min(axis=0))
                r1, c1 = map(int, rc.max(axis=0)+1)
                assert len(rc) == (r1-r0)*(c1-c0), 'Window must be one solid rectangle'
                assert not np.any(mask & mask[:, ::-1]), 'A hidden cell has no intact mate'
                full = scene.copy()
                full[mask] = scene[:, ::-1][mask]
                assert 8 not in full
                assert np.array_equal(full, full[:, ::-1]), 'Visible reflection witnesses conflict'
                groups = _groups(full, bg, wall)
                assert groups
                candidates.append(dict(rotation=rotation, panel=panel, scene=scene,
                                       full=full, bg=bg, wall=wall, mapping=mapping,
                                       order=tuple(k for c, k in sorted(order)),
                                       box=(r0, r1, c0, c1), groups=groups))
            except AssertionError:
                continue
    assert len(candidates) == 1, 'Input must have one legend/field interpretation'
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Uses only pixels: no seed, examples, cached mapping or constructor state."""
    parsed = _parse(input_grid)
    painted, _ = _paint(parsed['full'], parsed['bg'], parsed['wall'], parsed['mapping'])
    r0, r1, c0, c1 = parsed['box']
    return np.rot90(painted[r0:r1, c0:c1], -parsed['rotation']).copy()


def _grow_cavity(rng, h, w, anchored_right=False):
    mask = np.zeros((h, w), dtype=bool)
    r = int(rng.integers(h))
    c = w-1 if anchored_right else int(rng.integers(w))
    mask[r, c] = True
    target = int(rng.integers(max(1, h*w//2), h*w+1))
    while int(mask.sum()) < target:
        choices = set()
        for r, c in np.argwhere(mask):
            for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                    choices.add((int(x), int(y)))
        choices = sorted(choices)
        mask[choices[int(rng.integers(len(choices)))]] = True
    # Absorb any foreground island surrounded by this cavity.
    for hole in _holes(frozenset(map(tuple, np.argwhere(mask).tolist()))):
        for r, c in hole:
            mask[r, c] = True
    return mask


def _trim(mask):
    rc = np.argwhere(mask)
    lo, hi = rc.min(axis=0), rc.max(axis=0)+1
    return mask[lo[0]:hi[0], lo[1]:hi[1]].copy()


def _sculpt(mask, rng, holes, symmetric=False):
    """Alter actual contours, accepting only connectivity/topology-preserving cuts."""
    for _ in range(12):
        candidate = mask.copy()
        h, w = mask.shape
        side = int(rng.integers(4))
        breadth = int(rng.integers(1, 3))
        depth = int(rng.integers(1, min(3, h if side < 2 else w)+1))
        if side < 2:
            c = int(rng.integers(w))
            candidate[:depth if side == 0 else 0, c:c+breadth] = False
            if side == 1:
                candidate[h-depth:, c:c+breadth] = False
        else:
            r = int(rng.integers(h))
            if side == 2:
                candidate[r:r+breadth, :depth] = False
            else:
                candidate[r:r+breadth, w-depth:] = False
        if symmetric:
            candidate &= candidate[:, ::-1]
        if len(_components(candidate)) == 1 and _hole_count(candidate) == holes:
            mask = candidate
    return _trim(mask)


def _paired_object(k, rng, max_width=6):
    if k == 0:
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, max_width+1))
        mask = np.zeros((h, w), dtype=bool)
        style = int(rng.integers(3))
        if style == 0:
            mask[:, 0] = True
            mask[-1] = True
            mask[:, -1] = True
        elif style == 1:
            mask[h//2, :] = True
            mask[:, w//2] = True
            mask[-2:, :2] = True
        else:
            mask[:] = True
    else:
        rows = 1 if k <= 2 else 2
        cols = (k+rows-1)//rows
        rh = [int(rng.integers(1, 3)) for _ in range(rows)]
        cw = [1 for _ in range(cols)]
        for j in rng.permutation(cols):
            if 1+sum(cw)+cols < max_width and rng.random() < .6:
                cw[int(j)] += 1
        mask = np.ones((1+sum(rh)+rows, 1+sum(cw)+cols), dtype=bool)
        slots = list(rng.permutation(rows*cols))[:k]
        for slot in slots:
            i, j = divmod(int(slot), cols)
            r = 1+sum(rh[:i])+i
            c = 1+sum(cw[:j])+j
            cavity = _grow_cavity(rng, rh[i], cw[j])
            mask[r:r+rh[i], c:c+cw[j]][cavity] = False
    mask = _sculpt(mask, rng, k)
    if mask.shape[0] <= max_width and rng.random() < .5:
        mask = mask.T.copy()
    if rng.random() < .5:
        mask = mask[::-1].copy()
    assert len(_components(mask)) == 1 and _hole_count(mask) == k
    return mask


def _self_object(paired, fixed, rng, odd):
    """Paired holes stay inside the half; fixed holes meet the fold boundary."""
    half_width = int(rng.integers(4, 6))
    kinds = ['paired']*paired + ['fixed']*fixed
    rng.shuffle(kinds)
    depths = [int(rng.integers(1, 3)) for _ in kinds]
    h = 1+sum(depths)+len(kinds) if kinds else int(rng.integers(3, 7))
    left = np.ones((h, half_width), dtype=bool)
    r = 1
    for kind, depth in zip(kinds, depths):
        if kind == 'paired':
            width = int(rng.integers(1, half_width-1))
            start = int(rng.integers(1, half_width-width))
            cavity = _grow_cavity(rng, depth, width)
        else:
            width = int(rng.integers(1, half_width-1))
            start = half_width-width
            cavity = _grow_cavity(rng, depth, width, anchored_right=True)
        left[r:r+depth, start:start+width][cavity] = False
        r += depth+1
    full = np.concatenate((left, left[:, -2::-1] if odd else left[:, ::-1]), axis=1)
    full = _sculpt(full, rng, 2*paired+fixed, symmetric=True)
    assert len(_components(full)) == 1 and _hole_count(full) == 2*paired+fixed
    test_scene = np.where(full, 1, 0).astype(np.int8)
    group = _groups(test_scene, 0, 1)[0]
    assert group['count'] == paired+fixed and group['fixed'] == fixed
    return full


def _reserve(free, r, c, h, w):
    free[max(0, r-1):min(free.shape[0], r+h+1),
         max(0, c-1):min(free.shape[1], c+w+1)] = False


def _place(scene, labels, free, mask, count, centered, rng):
    h, w = mask.shape
    H, W = scene.shape
    if h > H:
        return False
    positions = []
    columns = [(W-w)//2] if centered else range(max(0, W//2-w))
    for r in range(H-h+1):
        for c in columns:
            mate_c = W-c-w
            if free[r:r+h, c:c+w].all() and free[r:r+h, mate_c:mate_c+w].all():
                positions.append((r, c))
    if not positions:
        return False
    r, c = positions[int(rng.integers(len(positions)))]
    positions = [(r, c, mask)]
    if not centered:
        positions.append((r, W-c-w, mask[:, ::-1]))
    for r, c, shape in positions:
        scene[r:r+h, c:c+w][shape] = True
        labels[r:r+h, c:c+w][shape] = count
        _reserve(free, r, c, h, w)
    return True


def _glyph(k, rng):
    rows = 1 if k <= 2 or rng.random() < .35 else 2
    cols = max(1, (k+rows-1)//rows)
    mask = np.ones((2*rows+1, max(3, 2*cols+1)), dtype=bool)
    for i in range(k):
        mask[1+2*(i//cols), 1+2*(i%cols)] = False
    if max(mask.shape) <= 5 and rng.random() < .5:
        mask = mask.T.copy()
    return mask


def _legend(keys, mapping, bg, width, rng):
    order = list(keys)
    rng.shuffle(order)
    glyphs = [_glyph(k, rng) for k in order]
    needed = sum(g.shape[1] for g in glyphs)+len(glyphs)+1
    if needed > width:
        return None
    height = max(g.shape[0] for g in glyphs)+2
    panel = np.full((height, width), bg, dtype=np.int8)
    c = int(rng.integers(1, width-needed+2))
    for k, glyph in zip(order, glyphs):
        r = int(rng.integers(1, height-glyph.shape[0]))
        panel[r:r+glyph.shape[0], c:c+glyph.shape[1]][glyph] = mapping[k]
        c += glyph.shape[1]+1
    return panel


def _choose_window(rng, scene, labels, mapping, training, index):
    H, W = scene.shape
    limit = W//2
    groups = _groups(np.where(scene, 1, 0), 0, 1)
    for _ in range(200):
        height = int(rng.integers(2, int(.70*H)+1)) if training else int(rng.integers(int(.80*H), H+1))
        width = int(rng.integers(1, limit+1))
        r = int(rng.integers(H-height+1))
        c = int(rng.integers(limit-width+1))
        if rng.random() < .5:
            c = W-c-width
        view = labels[r:r+height, c:c+width]
        hidden = set(map(int, np.unique(view))) - {-1}
        retained = hidden & set(mapping)
        if not retained or not np.any(view == -1):
            continue
        if sum(int(np.count_nonzero(view == k)) for k in retained) < 4:
            continue
        if training and index == 0 and 3 not in retained:
            continue
        if training and index == 1 and (1 not in retained or not hidden-set(mapping)):
            continue
        hit_groups = [g for g in groups if any(r <= x < r+height and c <= y < c+width
                                              for x, y in g['cells'])]
        if training and index == 0 and not any(g['count'] == 3 and g['paired']
                                               and g['fixed'] >= 2
                                               and sum(r <= x < r+height and c <= y < c+width
                                                       for x, y in g['cells']) >= 4
                                               for g in hit_groups):
            continue
        if training and index == 1 and not any(g['count'] == 1 and len(g['parts']) == 2
                                               for g in hit_groups):
            continue
        if training and index == 1 and sum(int(np.count_nonzero(view == k))
                                           for k in hidden-set(mapping)) < 2:
            continue
        if not training:
            # A tall test must query at least two different mirror groups.
            if len(hit_groups) < 2:
                continue
        return r, r+height, c, c+width
    return None


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _pack_pair(rng, roles, training, index, rotation):
    bg, wall = roles
    keys = [1, 3] if training and index < 2 else list(map(int, rng.choice(5, int(rng.integers(2, 5)), replace=False)))
    if training and index < 2 and rng.random() < .5:
        keys.append(int(rng.choice([0, 4])))
    colors = list(map(int, rng.permutation([c for c in range(8) if c not in roles])[:len(keys)]))
    mapping = dict(zip(keys, colors))
    W = int(rng.integers(22, 31))
    panel = _legend(keys, mapping, bg, W, rng)
    if panel is None or panel.shape[0] > 9:
        return None
    max_height = 29-panel.shape[0]
    if max_height < 18:
        return None
    H = int(rng.integers(18, max_height+1))
    scene = np.zeros((H, W), dtype=bool)
    labels = np.full((H, W), -1, dtype=np.int8)
    free = np.ones((H, W), dtype=bool)
    # The mixed fixed/paired witness is essential to distinguishing orbit counts.
    if training and index == 0:
        central = (1, 2)
        required = [1, 2]
    elif training and index == 1:
        central = (1, 0)
        required = [1, 2]
    else:
        k = int(rng.choice(keys))
        p = int(rng.integers(k+1))
        central = (p, k-p)
        absent = [k for k in range(5) if k not in keys]
        required = [int(rng.choice(keys)), int(rng.choice(absent)) if absent else 0]
    p, f = central
    mask = _self_object(p, f, rng, W % 2 == 1)
    if not _place(scene, labels, free, mask, p+f, True, rng):
        return None
    for k in required:
        mask = _paired_object(k, rng, max_width=6)
        if not _place(scene, labels, free, mask, k, False, rng):
            return None
    for _ in range(int(rng.integers(3))):
        k = int(rng.integers(5))
        centered = bool(rng.random() < .4)
        if centered:
            p = int(rng.integers(k+1))
            mask = _self_object(p, k-p, rng, W % 2 == 1)
        else:
            mask = _paired_object(k, rng)
        _place(scene, labels, free, mask, k, centered, rng)
    box = _choose_window(rng, scene, labels, mapping, training, index)
    if box is None:
        return None
    r0, r1, c0, c1 = box
    full = np.where(scene, wall, bg).astype(np.int8)
    # Labels come from construction's verified intended group counts.
    expected = np.full(full.shape, bg, dtype=np.int8)
    for k, color in mapping.items():
        expected[labels == k] = color
    painted, groups = _paint(full, bg, wall, mapping)
    assert np.array_equal(expected, painted), 'Construction and input topology disagree'
    inp_scene = full.copy()
    inp_scene[r0:r1, c0:c1] = 8
    inp = np.concatenate((panel, np.full((1, W), 9, dtype=np.int8), inp_scene))
    out = expected[r0:r1, c0:c1]
    return {'input': np.rot90(inp, rotation).copy(), 'output': np.rot90(out, rotation).copy()}


def _fallback(rng, roles, training, index, rotation, seen):
    """Finite, roomy construction with independent masks for up to 14 pairs."""
    bg, wall = roles
    keys = [1, 3]
    colors = [c for c in range(8) if c not in roles][:2]
    mapping = dict(zip(keys, colors))
    W, H = 28, 22
    panel = _legend(keys, mapping, bg, W, rng)
    scene = np.zeros((H, W), dtype=bool)
    labels = np.full((H, W), -1, dtype=np.int8)
    central = np.ones((7, 10), dtype=bool)
    central[1, 1] = central[1, 8] = False
    central[3, 4:6] = False
    central[5, 4:6] = False
    scene[1:8, 9:19] = central
    labels[1:8, 9:19][central] = 3
    for k, r in ((1, 11), (2, 16)):
        shape = np.ones((3, 2*k+1), dtype=bool)
        shape[1, 1:2*k:2] = False
        for c, mask in ((1, shape), (W-1-shape.shape[1], shape[:, ::-1])):
            scene[r:r+3, c:c+mask.shape[1]] = mask
            labels[r:r+3, c:c+mask.shape[1]][mask] = k
    full = np.where(scene, wall, bg).astype(np.int8)
    expected = np.full((H, W), bg, dtype=np.int8)
    for k, color in mapping.items():
        expected[labels == k] = color
    assert np.array_equal(expected, _paint(full, bg, wall, mapping)[0])
    for phase in range(56):
        if training and index == 0:
            box = (phase % 2, 8+phase % 3, 8+phase % 2, 12+phase % 3)
        elif training:
            box = (8+phase % 3, 20+phase % 3, phase % 4, 12+phase % 3)
        else:
            box = (phase % 4, 22, phase % 6, 12+phase % 3)
        r0, r1, c0, c1 = box
        inp_scene = full.copy()
        inp_scene[r0:r1, c0:c1] = 8
        inp = np.concatenate((panel, np.full((1, W), 9, dtype=np.int8), inp_scene))
        inp = np.rot90(inp, rotation).copy()
        if _key(inp) not in seen:
            return {'input': inp, 'output': np.rot90(expected[r0:r1, c0:c1], rotation).copy()}
    raise AssertionError('Finite uniqueness fallback exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int, rng.permutation(8)[:2]))
    stable_roles = bool(rng.random() < .35)
    base_rotation = int(rng.integers(4))
    mixed_orientation = bool(rng.random() < .55)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            roles = base_roles if stable_roles else tuple(map(int, rng.permutation(8)[:2]))
            rotation = int(rng.integers(4)) if mixed_orientation else base_rotation
            pair = None
            for _ in range(64):
                candidate = _pack_pair(rng, roles, split == 'train', i, rotation)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                pair = _fallback(rng, roles, split == 'train', i, rotation, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen = set()
    witnesses = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            p = _parse(a)
            r0, r1, c0, c1 = p['box']
            assert b.shape == np.rot90(p['scene'][r0:r1, c0:c1], -p['rotation']).shape
            assert np.array_equal(solve(a.copy()), b), 'Wrong projected census'
            crop = p['full'][r0:r1, c0:c1]
            assert np.any(crop == p['wall']) and np.any(crop == p['bg'])
            assert 8 not in b and 9 not in b and p['wall'] not in b
            # Every cavity has a reflected counterpart, and fixed cavities are
            # accounted for separately, not silently divided by two.
            for g in p['groups']:
                assert g['physical'] == 2*g['paired']+g['fixed']
                assert g['count'] == g['paired']+g['fixed']
                assert len(g['parts']) in (1, 2)
                hit = any(r0 <= r < r1 and c0 <= c < c1 for r, c in g['cells'])
                if split == 'train' and hit:
                    if g['count'] in p['mapping']:
                        witnesses.add(g['count'])
                        if len(g['parts']) == 2:
                            witnesses.add('paired_objects')
                    else:
                        witnesses.add('erase')
                    if g['paired'] and g['fixed'] >= 2 and g['count'] in p['mapping']:
                        witnesses.add('mixed')
            if split == 'train':
                ordinary = _paint(p['full'], p['bg'], p['wall'], p['mapping'], 'ordinary')[0]
                half = _paint(p['full'], p['bg'], p['wall'], p['mapping'], 'half')[0]
                expected = np.rot90(b, p['rotation'])
                if not np.array_equal(ordinary[r0:r1, c0:c1], expected):
                    witnesses.add('ordinary_fails')
                if not np.array_equal(half[r0:r1, c0:c1], expected):
                    witnesses.add('half_fails')
                if not np.array_equal(crop, expected):
                    witnesses.add('restore_only_fails')
                if np.any(expected != p['bg']):
                    witnesses.add('no_restore_fails')
    assert {'erase', 'paired_objects', 'mixed', 'ordinary_fails', 'half_fails',
            'restore_only_fails', 'no_restore_fails'} <= witnesses
    assert len(witnesses & set(range(5))) >= 2


def _features(pair):
    p = _parse(pair['input'])
    r0, r1, c0, c1 = p['box']
    return dict(shape=tuple(np.asarray(pair['input']).shape),
                output_shape=tuple(np.asarray(pair['output']).shape), rotation=p['rotation'],
                roles=(p['bg'], p['wall']), keys=tuple(sorted(p['mapping'])), order=p['order'],
                groups=len(p['groups']), objects=sum(len(g['parts']) for g in p['groups']),
                census=tuple(sorted(g['count'] for g in p['groups'])),
                fixed=sum(g['fixed'] for g in p['groups']), paired=sum(g['paired'] for g in p['groups']),
                odd=p['scene'].shape[1] % 2, window=(r1-r0, c1-c0),
                holes=tuple(sorted(len(hole) for g in p['groups'] for hole in g['holes'])))


def self_test():
    """Bounded 42-seed checks, ablations, covariance, and invalid-input tests."""
    features, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                features.append(_features(pair))
                p = _parse(pair['input'])
                structures.add(_key(p['full'] == p['wall']))
        # The shift is measured in canonical field coordinates, independent of rotation.
        train_h = [_features(p)['window'][0]/_parse(p['input'])['scene'].shape[0] for p in puzzle['train']]
        test_h = [_features(p)['window'][0]/_parse(p['input'])['scene'].shape[0] for p in puzzle['test']]
        assert max(train_h) <= .70 and min(test_h) >= .75
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            assert all(_key(x[k]) == _key(y[k]) for s in puzzle for x, y in zip(puzzle[s], again[s]) for k in x)
    assert len(signatures) == 42 and len(structures) >= 150
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['output_shape'] for f in features}) >= 35
    assert {f['rotation'] for f in features} == {0, 1, 2, 3}
    assert {f['odd'] for f in features} == {0, 1}
    assert len({f['roles'] for f in features}) >= 25
    assert len({f['keys'] for f in features}) >= 10
    assert len({f['census'] for f in features}) >= 15
    assert len({f['holes'] for f in features}) >= 80
    assert len({f['groups'] for f in features}) >= 2
    assert any(f['order'] != tuple(sorted(f['order'])) for f in features)
    assert any(f['keys'] == (1, 3) for f in features)
    assert any(f['window'][1] == 1 for f in features)
    # Four-connectivity: a diagonal escape still encloses, an orthogonal one opens.
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _hole_count(diagonal) == 1
    diagonal[0, 1] = False
    assert _hole_count(diagonal) == 0
    open_u = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert _hole_count(open_u) == 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(1000+10*nt+nq, nt, nq)
            assert (len(puzzle['train']), len(puzzle['test'])) == (nt, nq)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    puzzle = generate_puzzle(19, 2, 1)
    rotated = {s: [{k: np.rot90(v).copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(rotated)
    permutation = np.array([3, 0, 7, 2, 6, 1, 5, 4, 8, 9], dtype=np.int8)
    recolored = {s: [{k: permutation[v] for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    validate_generated_puzzle(recolored)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    bad['test'][0]['output'][0, 0] = 8
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted')
    # Exercise the finite fallback directly, including maximum explicit counts.
    rng = np.random.default_rng(23)
    fallback, seen = {'train': [], 'test': []}, set()
    for s, n in (('train', 10), ('test', 4)):
        for i in range(n):
            p = _fallback(rng, (0, 5), s == 'train', i, i % 4, seen)
            seen.add(_key(p['input']))
            fallback[s].append(p)
    validate_generated_puzzle(fallback)
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': len(signatures),
            'scene_structures': len(structures), 'input_shapes': len({f['shape'] for f in features}),
            'output_shapes': len({f['output_shape'] for f in features}),
            'census_populations': len({f['census'] for f in features}),
            'cavity_area_patterns': len({f['holes'] for f in features})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
