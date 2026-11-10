"""Generate translation-only object-pair ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = tuple(tuple(s.split('/')) for s in (
    '000000000/077701110/077700100/077701110/000000000/033300000/003000000/033300000/000000000',
    '00000080000/02220888000/02220088000/02220000000/00000080000/00111088800/00111008800/00111000000/03000020000/03000222000/03000022000',
    '00004400/00004400/00400000/04440000/00400440/00004440/44000440/44000000',
    '00000000000/00990022000/00990022000/00900022000/00000000000/00000007700/00033007700/00033007700/00030000000/11000000110/01000000110',
    '0000000000/0333005550/0303005050/0333005050/0000005050/4440005550/4040000000/4440000000/0000000000/0000000000',
    '3300066600/3300000600/3300000000/0000880440/0000800400/0000880440/0000000000/0000770000/0000770000/0000770000',
    '000000000000000/000000000222220/000000000200020/088888000204020/080008000200020/080408000222220/080008000000000/088888003000000/000000003300000/011111100000000/000000100999000/007770000999000/007770000999000/007770000000000/000000000000000',
))


def _key(mask):
    return mask.shape, mask.tobytes()


def _crop(mask):
    ys, xs = np.where(mask)
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


def _frame(rng, marked=False):
    h = int(rng.integers(3 if not marked else 6, 8))
    w = int(rng.integers(3 if not marked else 6, 8))
    mask = np.zeros((h, w), bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    marks = []
    if marked:
        cells = [(r, c) for r in range(2, h-2) for c in range(2, w-2)]
        rng.shuffle(cells)
        count = int(rng.integers(1, min(3, len(cells))+1))
        marks = cells[:count]
        for p in marks:
            mask[p] = True
    return mask, marks


def _shape(rng, kind=None):
    if kind is None:
        kind = ('line', 'rect', 'poly', 'frame', 'motif')[int(rng.integers(5))]
    if kind == 'line':
        n = int(rng.integers(2, 7))
        mask = np.ones((1, n), bool)
    elif kind == 'rect':
        mask = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), bool)
    elif kind == 'frame':
        return _frame(rng)
    elif kind == 'motif':
        patterns = ('100/111/011', '111/010/010', '110/011/001', '111/101/100', '110/110/100', '111/010/111')
        pattern = patterns[int(rng.integers(len(patterns)))]
        mask = np.array([[x == '1' for x in row] for row in pattern.split('/')], bool)
    else:
        h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        mask = np.zeros((h, w), bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        for _ in range(int(rng.integers(4, 11))):
            ys, xs = np.where(mask)
            k = int(rng.integers(len(ys)))
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r, c = int(ys[k])+dr, int(xs[k])+dc
            if 0 <= r < h and 0 <= c < w:
                mask[r, c] = True
        mask = _crop(mask)
        if mask.shape[0] == 1 or mask.shape[1] == 1:
            return _shape(rng, 'motif')
    turns = int(rng.integers(4))
    mask = np.rot90(mask, turns)
    if rng.integers(2):
        mask = np.fliplr(mask)
    return _crop(mask), []


def _source_near(rng):
    a = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 1]], bool)
    b = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], bool)
    turns = int(rng.integers(4))
    a, b = np.rot90(a, turns), np.rot90(b, turns)
    if rng.integers(2):
        a, b = np.fliplr(a), np.fliplr(b)
    return a.copy(), b.copy()


def _background(grid):
    counts = np.bincount(grid.astype(int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(grid, bg):
    seen = np.zeros(grid.shape, bool)
    comps = []
    for r, c in np.argwhere(grid != bg):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, pixels = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y+dy, x+dx
                if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1] and grid[yy, xx] != bg and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        comps.append(pixels)
    return comps


def _enclosed_pixels(pixels):
    ys, xs = zip(*pixels)
    r0, c0, r1, c1 = min(ys), min(xs), max(ys), max(xs)
    h, w = r1-r0+1, c1-c0+1
    solid = np.zeros((h, w), bool)
    for r, c in pixels:
        solid[r-r0, c-c0] = True
    exterior = np.zeros((h, w), bool)
    stack = [(r, c) for r in range(h) for c in range(w) if (r in (0, h-1) or c in (0, w-1)) and not solid[r, c]]
    for r, c in stack:
        exterior[r, c] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y, x = r+dr, c+dc
            if 0 <= y < h and 0 <= x < w and not solid[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                stack.append((y, x))
    inner = np.argwhere(~solid & ~exterior)
    return {(int(r)+r0, int(c)+c0) for r, c in inner}


def _objects(grid):
    bg = _background(grid)
    components = _components(grid, bg)
    interiors = [_enclosed_pixels(p) for p in components]
    parent = list(range(len(components)))
    for i, pixels in enumerate(components):
        containers = [j for j, interior in enumerate(interiors) if i != j and all(p in interior for p in pixels)]
        if containers:
            parent[i] = min(containers, key=lambda j: len(interiors[j]))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    result = []
    for i in range(len(components)):
        if root(i) != i:
            continue
        pixels = [p for j, component in enumerate(components) if root(j) == i for p in component]
        ys, xs = zip(*pixels)
        r0, c0 = min(ys), min(xs)
        mask = np.zeros((max(ys)-r0+1, max(xs)-c0+1), bool)
        for r, c in pixels:
            mask[r-r0, c-c0] = True
        result.append({'pixels': pixels, 'mask': mask, 'key': _key(mask), 'box': (r0, c0, max(ys), max(xs)), 'colors': {int(grid[r, c]) for r, c in pixels}})
    return result


def _solution(grid):
    objects = _objects(grid)
    groups = {}
    for i, obj in enumerate(objects):
        groups.setdefault(obj['key'], []).append(i)
    if any(len(group) > 2 for group in groups.values()):
        raise ValueError('ambiguous shape multiplicity')
    out = grid.copy()
    bg = _background(grid)
    for group in groups.values():
        if len(group) == 1:
            for r, c in objects[group[0]]['pixels']:
                out[r, c] = bg
    return out


def _tile(mask, marks, body, accent, bg):
    tile = np.full(mask.shape, bg, np.int8)
    tile[mask] = body
    for p in marks:
        tile[p] = accent
    return tile


def _color(rng, bg, exclude=()):
    choices = [c for c in range(10) if c != bg and c not in exclude]
    return int(rng.choice(choices))


def _render(rng, items, bg):
    total = sum(t.shape[0]*t.shape[1] for t in items)
    order = list(rng.permutation(len(items)))
    # The random placement uses bounding boxes for separation; corner adjacency is allowed.
    for expansion in range(5):
        side = max(9, int(np.sqrt(total*2.1)) + expansion*3)
        aspect = float(rng.uniform(0.72, 1.4))
        h = min(30, max(max(t.shape[0] for t in items)+2, int(side*aspect)))
        w = min(30, max(max(t.shape[1] for t in items)+2, int(side/aspect)))
        if expansion == 4:
            h = w = 30
        placed = []
        for k in order:
            tile = items[k]
            th, tw = tile.shape
            candidates = []
            if placed and rng.random() < 0.42:
                old_r, old_c, old_t = placed[int(rng.integers(len(placed)))]
                for rr, cc in ((old_r+old_t.shape[0], old_c+old_t.shape[1]), (old_r-th, old_c-tw), (old_r+old_t.shape[0], old_c-tw), (old_r-th, old_c+old_t.shape[1])):
                    candidates.append((rr, cc))
            candidates += [(int(rng.integers(h-th+1)), int(rng.integers(w-tw+1))) for _ in range(220)]
            found = False
            for r, c in candidates:
                if not (0 <= r <= h-th and 0 <= c <= w-tw):
                    continue
                if any(not (r+th <= pr or pr+pt.shape[0] <= r or c+tw <= pc or pc+pt.shape[1] <= c) for pr, pc, pt in placed):
                    continue
                # A shared edge would merge objects under four-connectivity.
                if any((r+th == pr or pr+pt.shape[0] == r) and not (c+tw <= pc or pc+pt.shape[1] <= c) or (c+tw == pc or pc+pt.shape[1] == c) and not (r+th <= pr or pr+pt.shape[0] <= r) for pr, pc, pt in placed):
                    continue
                placed.append((r, c, tile))
                found = True
                break
            if not found:
                break
        if len(placed) != len(items):
            continue
        grid = np.full((h, w), bg, np.int8)
        for r, c, tile in placed:
            grid[r:r+tile.shape[0], c:c+tile.shape[1]] = tile
        # Crop some margins and leave others; this includes edge-touching examples.
        if rng.random() < 0.45:
            ys, xs = np.where(grid != bg)
            top = max(0, int(ys.min())-int(rng.integers(3)))
            left = max(0, int(xs.min())-int(rng.integers(3)))
            bottom = min(h, int(ys.max())+1+int(rng.integers(3)))
            right = min(w, int(xs.max())+1+int(rng.integers(3)))
            grid = grid[top:bottom, left:right].copy()
        if rng.random() < 0.38:
            pad_h = 1 if grid.shape[0] <= 28 else 0
            pad_w = 1 if grid.shape[1] <= 28 else 0
            if pad_h or pad_w:
                grid = np.pad(grid, ((pad_h, pad_h), (pad_w, pad_w)),
                              constant_values=bg)
        return grid
    raise RuntimeError('could not place objects')


def _make_pair(rng, bg, regime, require_same=False, require_cross=False,
               require_near=False, require_marked=False, require_oriented=False):
    pairs = int(rng.integers(1, 4))
    singles = int(rng.integers(1, 4))
    if regime == 'source_train' and require_marked:
        pairs = max(pairs, 2)
    if regime == 'target_test' and rng.random() < 0.25:
        pairs = 1
    shapes = []
    used = set()
    if require_near:
        a, b = _source_near(rng)
        shapes.append((a, [], 'pair'))
        shapes.append((b, [], 'single'))
        used.update((_key(a), _key(b)))
    if require_oriented:
        if require_near:
            base = shapes[0][0]
        else:
            base, _ = _source_near(rng)
            shapes.append((base, [], 'pair'))
            used.add(_key(base))
        for turns in (1, 2, 3):
            rotated = np.rot90(base, turns).copy()
            if _key(rotated) not in used:
                shapes.append((rotated, [], 'single'))
                used.add(_key(rotated))
                break
    if require_marked:
        for _ in range(30):
            a, marks = _frame(rng, True)
            if _key(a) not in used:
                break
        shapes.append((a, marks, 'pair'))
        used.add(_key(a))
        # Near miss moves one interior mark and keeps the same box and mass.
        if len(shapes) < pairs+singles:
            choices = [(r, c) for r in range(2, a.shape[0]-2)
                       for c in range(2, a.shape[1]-2) if not a[r, c]]
            if choices:
                b = a.copy()
                b[marks[0]] = False
                moved_to = choices[int(rng.integers(len(choices)))]
                b[moved_to] = True
                if _key(b) not in used:
                    shapes.append((b, marks[1:] + [moved_to], 'single'))
                    used.add(_key(b))
    while sum(role == 'pair' for _, _, role in shapes) < pairs or sum(role == 'single' for _, _, role in shapes) < singles:
        role = 'pair' if sum(x[2] == 'pair' for x in shapes) < pairs else 'single'
        mask, marks = _shape(rng)
        key = _key(mask)
        if key in used:
            continue
        used.add(key)
        shapes.append((mask, marks, role))
    rng.shuffle(shapes)
    items = []
    pair_index = 0
    for mask, marks, role in shapes:
        if role == 'pair':
            a = _color(rng, bg)
            if require_same and pair_index == 0:
                b = a
            elif require_cross and pair_index == 0:
                b = _color(rng, bg, (a,))
            else:
                b = a if rng.random() < 0.38 else _color(rng, bg, (a,))
            accent_a = _color(rng, bg, (a,)) if marks else a
            accent_b = _color(rng, bg, (b,)) if marks else b
            items.extend((_tile(mask, marks, a, accent_a, bg), _tile(mask, marks, b, accent_b, bg)))
            pair_index += 1
        else:
            color = _color(rng, bg)
            accent = _color(rng, bg, (color,)) if marks else color
            items.append(_tile(mask, marks, color, accent, bg))
    return _render(rng, items, bg)


def _check_count(value, name, lo, hi):
    if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
        raise ValueError(f'{name} must be an integer from {lo} through {hi}')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _check_count(num_train, 'num_train', 2, 10)
    _check_count(num_test, 'num_test', 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1 if num_train is not None else 4)+1))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    base_bg = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    marked_puzzle = bool(rng.random() < 0.65)
    oriented_puzzle = True
    near_index = int(rng.integers(num_train))
    marked_index = int(rng.integers(num_train)) if marked_puzzle else -1
    oriented_index = int(rng.integers(num_train)) if oriented_puzzle else -1
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                bg = base_bg if rng.random() < 0.7 else int(rng.integers(10))
                try:
                    inp = _make_pair(pair_rng, bg, 'source_train' if split == 'train' else 'target_test',
                                     require_same=split == 'train' and index == 0,
                                     require_cross=split == 'train' and index == 1,
                                     require_near=split == 'train' and index == near_index,
                                     require_marked=(split == 'train' and index == marked_index) or
                                                    (split == 'test' and marked_puzzle and rng.random() < 0.55),
                                     require_oriented=(split == 'train' and index == oriented_index) or
                                                      (split == 'test' and oriented_puzzle and rng.random() < 0.35))
                    out = _solution(inp)
                except (RuntimeError, ValueError):
                    continue
                sig = (inp.shape, inp.tobytes())
                if sig not in used and _background(inp) == bg and not np.array_equal(inp, out):
                    break
            else:
                raise RuntimeError('could not generate distinct pair')
            used.add(sig)
            puzzle[split].append({'input': inp, 'output': out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen = set()
    same_train = cross_train = near_train = marked_train = oriented_train = False
    marked_test = oriented_test = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            sig = (inp.shape, inp.tobytes())
            assert sig not in seen
            seen.add(sig)
            lines = tuple(''.join(str(int(x)) for x in row) for row in inp)
            assert lines not in _SOURCE_INPUTS
            bg = _background(inp)
            assert np.all((out == inp) | (out == bg))
            objs = _objects(inp)
            groups = {}
            for obj in objs:
                groups.setdefault(obj['key'], []).append(obj)
            assert all(len(group) <= 2 for group in groups.values())
            assert any(len(group) == 2 for group in groups.values())
            assert any(len(group) == 1 for group in groups.values())
            assert np.array_equal(out, _solution(inp))
            for group in groups.values():
                for obj in group:
                    for r, c in obj['pixels']:
                        assert int(out[r, c]) == (int(inp[r, c]) if len(group) == 2 else bg)
            marked = [o for o in objs if len(_components(inp[o['box'][0]:o['box'][2]+1, o['box'][1]:o['box'][3]+1], bg)) > 1]
            singles = [group[0] for group in groups.values() if len(group) == 1]
            oriented = any(
                np.array_equal(single['mask'], np.rot90(obj['mask'], turns))
                for single in singles for group in groups.values() if len(group) == 2
                for obj in group for turns in (1, 2, 3)
                if single['key'] != obj['key']
            )
            if split == 'train':
                marked_train |= bool(marked)
                oriented_train |= oriented
                for group in groups.values():
                    if len(group) == 2:
                        colors = [int(inp[o['pixels'][0]]) for o in group]
                        same_train |= colors[0] == colors[1]
                        cross_train |= colors[0] != colors[1]
                for single in singles:
                    for group in groups.values():
                        if len(group) == 2 and any(single['mask'].shape == g['mask'].shape and int(single['mask'].sum()) == int(g['mask'].sum()) for g in group):
                            near_train = True
            else:
                marked_test |= bool(marked)
                oriented_test |= oriented
    assert same_train and cross_train and near_train
    assert not marked_test or marked_train
    assert not oriented_test or oriented_train


def self_test() -> None:
    def frozen(p):
        return tuple((k, tuple((x['input'].shape, x['input'].tobytes(), x['output'].tobytes()) for x in p[k])) for k in ('train', 'test'))
    def touches_diagonally(objects):
        for i, a in enumerate(objects):
            for b in objects[i+1:]:
                cells = set(b['pixels'])
                if any((r+dr, c+dc) in cells for r, c in a['pixels']
                       for dr in (-1, 1) for dc in (-1, 1)):
                    return True
        return False
    signatures, sizes, backgrounds, train_counts, test_counts = set(), set(), set(), set(), set()
    train_widths, test_widths, shape_keys, marked_positions = set(), set(), set(), set()
    frame_sizes, long_frame_sizes, mark_counts = set(), set(), set()
    pair_counts, singleton_counts = set(), set()
    stats = {k: 0 for k in ('one_test', 'multi_test', 'marked_test', 'plain_test',
                             'same', 'different', 'near', 'oriented', 'reused',
                             'diagonal', 'edge', 'margin', 'varied_bg',
                             'varied_pairs', 'varied_singles', 'mark_color_diff',
                             'simple_test', 'combined_test')}
    for seed in range(42):
        p = generate_puzzle(seed)
        assert frozen(p) == frozen(generate_puzzle(seed))
        signatures.add(frozen(p))
        train_counts.add(len(p['train']))
        test_counts.add(len(p['test']))
        puzzle_bgs, puzzle_pairs, puzzle_singles = set(), set(), set()
        train_oriented = False
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                bg = _background(inp)
                objects = _objects(inp)
                groups = {}
                for obj in objects:
                    groups.setdefault(obj['key'], []).append(obj)
                    shape_keys.add(obj['key'])
                paired = [v for v in groups.values() if len(v) == 2]
                singles = [v[0] for v in groups.values() if len(v) == 1]
                for obj in objects:
                    mask = obj['mask']
                    h, w = mask.shape
                    if h >= 3 and w >= 3 and mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all() and not mask[1:-1, 1:-1].all():
                        frame_sizes.add((h, w))
                        if abs(h-w) >= 2:
                            long_frame_sizes.add((h, w))
                for a, b in paired:
                    mask = a['mask']
                    h, w = mask.shape
                    if h >= 6 and w >= 6 and mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all() and mask[2:-2, 2:-2].any():
                        locations = [(int(r), int(c)) for r, c in np.argwhere(mask[1:-1, 1:-1])]
                        mark_counts.add(len(locations))
                        first = tuple(int(inp[a['box'][0]+1+r, a['box'][1]+1+c]) for r, c in locations)
                        second = tuple(int(inp[b['box'][0]+1+r, b['box'][1]+1+c]) for r, c in locations)
                        stats['mark_color_diff'] += first != second
                puzzle_bgs.add(bg)
                puzzle_pairs.add(len(paired))
                puzzle_singles.add(len(singles))
                pair_counts.add(len(paired))
                singleton_counts.add(len(singles))
                sizes.add(inp.shape)
                backgrounds.add(bg)
                assert np.array_equal(out, _solution(inp))
                for group in groups.values():
                    for obj in group:
                        for r, c in obj['pixels']:
                            assert int(out[r, c]) == (int(inp[r, c]) if len(group) == 2 else bg)
                if split == 'train':
                    train_widths.add(inp.shape[1])
                else:
                    test_widths.add(inp.shape[1])
                stats['same'] += any(int(inp[a['pixels'][0]]) == int(inp[b['pixels'][0]]) for a, b in paired)
                stats['different'] += any(int(inp[a['pixels'][0]]) != int(inp[b['pixels'][0]]) for a, b in paired)
                stats['near'] += any(a['mask'].shape == b['mask'].shape and int(a['mask'].sum()) == int(b['mask'].sum())
                                     for a in singles for group in paired for b in group)
                oriented = any(np.array_equal(a['mask'], np.rot90(b['mask'], k)) and a['key'] != b['key']
                               for a in singles for group in paired for b in group for k in (1, 2, 3))
                stats['oriented'] += oriented
                if split == 'train':
                    train_oriented |= oriented
                stats['reused'] += any(int(inp[a['pixels'][0]]) == int(inp[b['pixels'][0]])
                                       for a in singles for group in paired for b in group)
                stats['diagonal'] += touches_diagonally(objects)
                edge = any(0 in obj['box'][:2] or obj['box'][2] == inp.shape[0]-1 or obj['box'][3] == inp.shape[1]-1
                           for obj in objects)
                stats['edge' if edge else 'margin'] += 1
                marked = False
                for obj in objects:
                    r0, c0, r1, c1 = obj['box']
                    if len(_components(inp[r0:r1+1, c0:c1+1], bg)) > 1:
                        marked = True
                        marked_positions.add(obj['key'])
                if split == 'test':
                    stats['one_test' if len(paired) == 1 else 'multi_test'] += 1
                    stats['marked_test' if marked else 'plain_test'] += 1
                    stats['simple_test'] += len(paired) == 1 and len(objects) <= 4
                    stats['combined_test'] += marked and len(paired) >= 2 and len(objects) >= 6
        assert train_oriented
        stats['varied_bg'] += len(puzzle_bgs) > 1
        stats['varied_pairs'] += len(puzzle_pairs) > 1
        stats['varied_singles'] += len(puzzle_singles) > 1
    assert len(signatures) == 42 and len(sizes) >= 25 and len(backgrounds) >= 5
    assert len(shape_keys) >= 80 and len(marked_positions) >= 10
    assert len(frame_sizes) >= 12 and len(long_frame_sizes) >= 5
    assert mark_counts == {1, 2, 3}
    assert any(h > w for h, w in long_frame_sizes) and any(w > h for h, w in long_frame_sizes)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert pair_counts == {1, 2, 3} and singleton_counts == {1, 2, 3}
    assert len(train_widths & test_widths) >= 8
    assert min(test_widths) < max(train_widths)
    assert all(stats[k] > 0 for k in ('one_test', 'multi_test', 'marked_test', 'plain_test',
                                      'same', 'different', 'near', 'oriented', 'reused',
                                      'diagonal', 'edge', 'margin', 'mark_color_diff',
                                      'simple_test', 'combined_test'))
    assert all(stats[k] >= 20 for k in ('varied_bg', 'varied_pairs', 'varied_singles'))
    for n, m in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(91, n, m)
        assert frozen(p) == frozen(generate_puzzle(91, n, m))
        if n is not None:
            assert len(p['train']) == n
        if m is not None:
            assert len(p['test']) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(7, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
