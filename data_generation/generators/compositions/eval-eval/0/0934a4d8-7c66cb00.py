"""Window-selected dispatch: reconstruct, select whole objects, and route them.

Only NumPy and the standard library are used. solve() has no generation state.
"""
from collections import defaultdict
import numpy as np

_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(np.all((a >= 0) & (a <= 9)), 'colors outside 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(a, background, diagonal=False):
    unseen = {tuple(map(int, p)) for p in np.argwhere(a != background)}
    groups = []
    neighbors = _EIGHT if diagonal else _FOUR
    while unseen:
        start = min(unseen)
        color = int(a[start])
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = r+dr, c+dc
                if p in unseen and a[p] == color:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        groups.append((color, cells))
    return groups


def _bbox(points):
    return (min(r for r, c in points), max(r for r, c in points)+1,
            min(c for r, c in points), max(c for r, c in points)+1)


def _rectangle(box):
    r, s, c, d = box
    return {(y, x) for y in range(r, s) for x in range(c, d)}


def _frame(a):
    candidates = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        r, s = int(rr.min()), int(rr.max())+1
        c, d = int(cc.min()), int(cc.max())+1
        height, width = s-r, d-c
        # The count plus four complete edges proves that every occurrence is
        # on the rim; avoid materializing pixel sets for non-frame colors.
        if height < 5 or width < 5 or len(rr) != 2*(height+width)-4:
            continue
        if (np.all(a[r, c:d] == color) and np.all(a[s-1, c:d] == color) and
                np.all(a[r:s, c] == color) and np.all(a[r:s, d-1] == color)):
            candidates.append((int(color), (r, s, c, d)))
    _require(len(candidates) == 1, 'one complete specimen frame is required')
    return candidates[0]


def _canonical_parse(a, detected_frame=None):
    """Parse frame, background, direction, bands, and overwrite from pixels."""
    frame_color, box = _frame(a) if detected_frame is None else detected_frame
    r, s, c, d = box
    background = int(a[0, 0])
    _require(all(a[p] == background for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))),
             'background corners')
    _require(background != frame_color and r >= 1 and c >= 1 and d < a.shape[1],
             'specimen margins')
    occupied = np.zeros(a.shape, dtype=bool)
    occupied[r:s, c:d] = True
    bands = []
    y = s
    while y < a.shape[0]:
        nz = np.flatnonzero(a[y] != background)
        if not len(nz):
            y += 1
            continue
        left, right = int(nz[0]), int(nz[-1])
        rail, route = int(a[y, left]), int(a[y, left+1]) if right > left else -1
        _require(right-left >= 4 and rail != route and rail != background and
                 route != background and a[y, right] == rail and
                 np.all(a[y, left+1:right] == route), 'malformed receiving band')
        top = y
        while y+1 < a.shape[0] and np.array_equal(a[y+1], a[top]):
            y += 1
        bottom = y
        _require(top > s and np.all(a[top-1] == background), 'clear band separator')
        bands.append(dict(top=top, bottom=bottom, left=left, right=right,
                          route=route, rail=rail))
        occupied[top:bottom+1, left:right+1] = True
        y += 1
    _require(1 <= len(bands) <= 3, 'one to three bands are required')
    routes = {b['route'] for b in bands}
    _require(len(routes) == len(bands), 'routing colors must be unique')
    _require(frame_color not in routes | {b['rail'] for b in bands}, 'frame color has one role')
    _require(np.all(a[~occupied] == background), 'unexplained pixels outside tray and bands')
    stage = a[r+1:s-1, c+1:d-1].copy()
    foreign = set(map(int, np.unique(stage))) - routes - {background}
    _require(len(foreign) == 1, 'one distinct overwrite color is required')
    mask = foreign.pop()
    _require(mask != frame_color and mask not in {b['rail'] for b in bands}, 'overwrite role reuse')
    points = {tuple(map(int, p)) for p in np.argwhere(stage == mask)}
    window = _bbox(points)
    _require(len(points) >= 2 and points == _rectangle(window), 'solid rectangular overwrite')
    reflected = stage[:, ::-1]
    visible = (stage != mask) & (reflected != mask)
    _require(np.array_equal(stage[visible], reflected[visible]), 'conflicting reflection evidence')
    _require(np.any(visible & (stage != background)), 'visible nontrivial reflection witnesses')
    _require(np.all(reflected[stage == mask] != mask), 'a mirror orbit was erased')
    restored = stage.copy()
    restored[stage == mask] = reflected[stage == mask]
    _require(np.array_equal(restored, restored[:, ::-1]), 'incomplete symmetry')
    groups = _components(restored, background)
    selected = [(color, cells) for color, cells in groups if cells & points]
    _require(selected, 'window must touch at least one object')
    return dict(canonical=a, background=background, frame_color=frame_color,
                frame=box, bands=bands, stage=stage, restored=restored,
                mask=mask, window=window, covered=points,
                groups=groups, selected=selected)


def _targets(record, groups=None, group_by_color=False):
    groups = record['selected'] if groups is None else groups
    if group_by_color:
        merged = defaultdict(set)
        for color, cells in groups:
            merged[color].update(cells)
        groups = list(merged.items())
    r0, _, c0, _ = record['frame']
    lookup = {b['route']: b for b in record['bands']}
    targets, occupied = [], set()
    for color, cells in groups:
        band = lookup[color]
        bottom = max(r for r, c in cells)
        stamp = {(band['bottom'] - bottom + r, c0+1+c) for r, c in cells}
        _require(all(band['top'] <= r <= band['bottom'] and
                     band['left'] < c < band['right'] for r, c in stamp), 'cargo does not fit')
        _require(not stamp & occupied, 'cargo collision')
        occupied.update(stamp)
        targets.append((band['rail'], stamp))
    return targets


def _render(record):
    a = record['canonical']
    out = a.copy()
    r, s, c, d = record['frame']
    specimen = record['restored'].copy()
    for color, cells in record['selected']:
        for p in cells:
            specimen[p] = record['background']
    out[r+1:s-1, c+1:d-1] = specimen
    for rail, points in _targets(record):
        for p in points:
            out[p] = rail
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    frame_color, (r, s, c, d) = _frame(a)
    h, w = a.shape
    # Rotation changes the frame coordinates, not its color or identity.
    # Still test every direction to retain the unique-interpretation check.
    boxes = ((r, s, c, d), (w-d, w-c, r, s),
             (h-s, h-r, w-d, w-c), (c, d, h-s, h-r))
    candidates = []
    for turn in range(4):
        try:
            record = _canonical_parse(np.rot90(a, turn), (frame_color, boxes[turn]))
            record['output'] = np.rot90(_render(record), -turn).copy()
            record['turn'] = turn
            candidates.append(record)
        except AssertionError:
            continue
    _require(len(candidates) == 1, 'one tray-to-bands direction is required')
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer using only this input's visible conventions."""
    return _analyze(input_grid)['output']


def _signature(groups):
    return {frozenset(cells) for _, cells in groups}


def _stamp_pixels(targets):
    return {(r, c, color) for color, cells in targets for r, c in cells}


def _features(rec):
    selected = rec['selected']
    background = rec['background']
    covered = rec['covered']
    true_stamps = _stamp_pixels(_targets(rec))
    cut = rec['restored'].copy()
    cut[rec['stage'] == rec['mask']] = background
    selected_cells = set().union(*(cells for _, cells in selected))
    fragments = [(color, cells) for color, cells in _components(cut, background)
                 if cells <= selected_cells]
    split_component = any(sum(bool(part & cells) for _, part in fragments) >= 2
                          for _, cells in selected)
    # Compare transported visible pixels, so merely supplying missing ink is insufficient.
    try:
        cut_pixels = _stamp_pixels(_targets(rec, fragments))
        r0, _, c0, _ = rec['frame']
        known_pixels = set()
        bands = {b['route']: b for b in rec['bands']}
        for color, cells in selected:
            bottom = max(r for r, c in cells)
            band = bands[color]
            known_pixels.update((band['bottom']-bottom+r, c0+1+c, band['rail'])
                                for r, c in cells-covered)
        bridge = split_component and bool(known_pixels - cut_pixels)
    except AssertionError:
        bridge = split_component
    try:
        independent = _stamp_pixels(_targets(rec, group_by_color=True)) != true_stamps
    except AssertionError:
        independent = True
    touches = len({color for color, cells in selected})
    groups_by_color = defaultdict(list)
    for color, cells in selected:
        groups_by_color[color].append(cells)
    unequal = any(len({max(r for r, c in cells) for cells in groups}) >= 2
                  for groups in groups_by_color.values())
    r, s, c, d = rec['window']
    h, w = rec['stage'].shape
    bbox_decoy = any(not cells & covered and _rectangle(_bbox(cells)) & covered
                     for _, cells in rec['groups'])
    return dict(bbox_decoy=bool(bbox_decoy), bridge=bridge, independent=independent, unequal=unequal,
                colors=touches, selected=len(selected), total=len(rec['groups']),
                unused=len(rec['bands'])-touches,
                boundary=(r == 0 or s == h or c == 0 or d == w),
                whole=any(cells-covered for _, cells in selected),
                spared=any(color in {k for k, _ in selected} and not cells & covered
                           for color, cells in rec['groups']))


def _random_mask(rng, height, width):
    kind = int(rng.integers(5))
    if kind == 0:
        return {(r, c) for r in range(height) for c in range(width)}
    if kind == 1 and min(height, width) >= 3:
        return {(r, c) for r in range(height) for c in range(width)
                if r in (0, height-1) or c in (0, width-1)}
    if kind == 2:
        spine = int(rng.integers(width))
        cells = {(r, spine) for r in range(height)}
        for r in range(height):
            end = int(rng.integers(width))
            cells.update((r, c) for c in range(min(spine, end), max(spine, end)+1))
        return cells
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    goal = int(rng.integers(1, height*width+1))
    for _ in range(goal-1):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR
                           if 0 <= r+dr < height and 0 <= c+dc < width} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _place(base, cells, color, background):
    if any(base[p] != background for p in cells):
        return False
    if any(0 <= r+dr < base.shape[0] and 0 <= c+dc < base.shape[1] and
           (r+dr, c+dc) not in cells and base[r+dr, c+dc] == color
           for r, c in cells for dr, dc in _EIGHT):
        return False
    for p in cells:
        base[p] = color
    return True


def _specimen(rng, background, routes, lesson):
    h, half = int(rng.integers(6, 11)), int(rng.integers(7, 14))
    base = np.full((h, half), background, dtype=np.int8)
    a = int(rng.integers(len(routes)))
    b = (a+int(rng.integers(1, len(routes)))) % len(routes) if len(routes) > 1 else a
    ph, pw = int(rng.integers(3, min(6, h-(2 if lesson == 'independent' else 1))+1)), int(rng.integers(3, half-3))
    top = int(rng.integers(1, h-ph)) if lesson == 'independent' else int(rng.integers(0, h-ph+1))
    left = int(rng.integers(1, half-pw-2))
    kind = int(rng.integers(4))
    if kind == 0:  # hollow frame
        main = {(r, c) for r in range(ph) for c in range(pw)
                if r in (0, ph-1) or c in (0, pw-1)}
    elif kind == 1:  # branched spine
        spine = int(rng.integers(pw))
        main = {(r, spine) for r in range(ph)}
        main.update((r, c) for r in (0, ph-1) for c in range(pw))
        for r in range(1, ph-1):
            end = int(rng.integers(pw))
            main.update((r, c) for c in range(min(end, spine), max(end, spine)+1))
    elif kind == 2:  # asymmetric C
        side = int(rng.choice([0, pw-1]))
        main = {(r, side) for r in range(ph)} | {(r, c) for r in (0, ph-1) for c in range(pw)}
    else:
        main = {(r, c) for r in range(ph) for c in range(pw)}
    _place(base, {(top+r, left+c) for r, c in main}, routes[a], background)
    # A colored island or attached different-color cell makes color splitting visible.
    middle = top + int(rng.integers(1, ph-1))
    if len(routes) > 1:
        holes = [(middle, c) for c in range(left, left+pw) if base[middle, c] == background]
        if holes:
            p = holes[int(rng.integers(len(holes)))]
            base[p] = routes[b]
        else:
            side = left+pw if lesson == 'independent' else left-1
            base[middle, side] = routes[b]
    # A separate same-color piece has a disjoint lateral span and another depth.
    extra_bottom = int(rng.choice([r for r in range(1, h-1) if r != top+ph-1]))
    extra_h = int(rng.integers(1, min(3, extra_bottom+(0 if lesson == 'independent' else 1))+1))
    base[extra_bottom-extra_h+1:extra_bottom+1, half-2] = routes[a]
    if lesson == 'independent':
        # A third-color enclosure surrounds the queried cargo without being touched.
        # Its band stays unused: occupied-cell intersection, not bounding-box overlap.
        outer = next(color for color in routes if color not in (routes[a], routes[b]))
        base[[0, -1], :] = outer
        base[:, [0, -1]] = outer
    # Additional routes are anchored on the source boundary, outside lesson windows.
    for k, color in enumerate(routes):
        if not np.any(base == color):
            spots = [(r, 0) for r in range(h) if base[r, 0] == background]
            for p in (spots[i] for i in rng.permutation(len(spots))):
                if _place(base, {p}, color, background):
                    break
    if lesson is None:
        for _ in range(int(rng.integers(1, 6))):
            color = int(rng.choice(routes))
            mh, mw = int(rng.integers(1, h+1)), int(rng.integers(1, half))
            cells = _random_mask(rng, mh, mw)
            bh, bw = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
            for _ in range(8):
                rr, cc = int(rng.integers(h-bh+1)), int(rng.integers(half-1-bw+1))
                if _place(base, {(rr+r, cc+c) for r, c in cells}, color, background):
                    break
    # Usually a central gutter separates copies; an enclosing third color can span it.
    # Odd/even widths vary.
    full = np.concatenate((base, base[:, -2::-1] if rng.integers(2) else base[:, ::-1]), axis=1)
    if lesson == 'bridge':
        c0 = left if np.any(base[middle, left:left+pw] == routes[b]) else max(1, left-1)
        window = (middle, middle+1, c0, left+pw)
    elif lesson == 'independent':
        window = (max(1, min(top, extra_bottom-extra_h+1)),
                  min(h-1, max(top+ph, extra_bottom+1)), left+1, half-1)
    else:
        window = None
    return full, window


def _window(rng, shape, boundary):
    h, w = shape
    half = w//2  # excludes a self-reflecting middle column when width is odd
    if boundary:
        edge = int(rng.integers(4))
        rh = int(rng.integers(1, h+1))
        rw = int(rng.integers(1, half+1))
        r, c = int(rng.integers(h-rh+1)), int(rng.integers(half-rw+1))
        if edge == 0:
            r = 0
        elif edge == 1:
            c = 0
        elif edge == 2:
            r = h-rh
        else:
            c = 0
        if edge == 3:
            c = w-c-rw
        elif rng.integers(2) and edge != 1:
            c = w-c-rw
    else:
        rh, rw = int(rng.integers(1, h-1)), int(rng.integers(1, half))
        r, c = int(rng.integers(1, h-rh)), int(rng.integers(1, half-rw+1))
        if rng.integers(2):
            c = w-c-rw
    return r, r+rh, c, c+rw


def _construct(rng, lesson, boundary, turn):
    count = 2 if lesson == 'bridge' else 3 if lesson == 'independent' else int(rng.integers(1, 4))
    palette = list(map(int, rng.permutation(10)))
    background, frame_color, mask = palette[:3]
    routes = palette[3:3+count]
    extras = palette[3+count:]
    rails = [int(rng.choice([x for x in routes+extras if x != route])) for route in routes]
    field, given_window = _specimen(rng, background, routes, lesson)
    h, w = field.shape
    for _ in range(24):
        window = given_window if given_window is not None else _window(rng, field.shape, boundary)
        covered = _rectangle(window)
        selected = [(color, cells) for color, cells in _components(field, background) if cells & covered]
        if not selected or len(covered) < 2:
            continue
        required = {color: 1 for color in routes}
        for color, cells in selected:
            required[color] = max(required[color], max(r for r, c in cells)-min(r for r, c in cells)+1)
        top_margin, bottom_margin = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        xmargin = 1 if w >= 25 else int(rng.integers(1, 3))
        width = w+2+2*xmargin
        gaps = [int(rng.integers(1, 3)) for _ in routes]
        budget = 30-top_margin-(h+2)-bottom_margin-sum(gaps)
        if sum(required.values()) > budget or width > 30:
            continue
        depths = [required[color] for color in routes]
        slack = int(rng.integers(min(5, budget-sum(depths))+1))
        for i in rng.integers(count, size=slack):
            depths[int(i)] += 1
        height = top_margin+h+2+sum(gaps)+sum(depths)+bottom_margin
        a = np.full((height, width), background, dtype=np.int8)
        fr = (top_margin, top_margin+h+2, xmargin, xmargin+w+2)
        r, s, c, d = fr
        a[r:s, c:d] = frame_color
        a[r+1:s-1, c+1:d-1] = field
        row, bands = s, []
        order = list(map(int, rng.permutation(count)))
        for i in order:
            row += gaps[i]
            # Bands may extend beyond either side of the specimen independently.
            left, right = int(rng.integers(c+1)), int(rng.integers(d-1, width))
            band = dict(top=row, bottom=row+depths[i]-1, left=left, right=right,
                        route=routes[i], rail=rails[i])
            a[row:row+depths[i], left:right+1] = routes[i]
            a[row:row+depths[i], [left, right]] = rails[i]
            bands.append(band)
            row += depths[i]
        # Construction ground truth uses the completed specimen, not solve().
        out = a.copy()
        specimen = field.copy()
        occupied = set()
        valid = True
        lookup = {b['route']: b for b in bands}
        for color, cells in selected:
            band = lookup[color]
            bottom = max(y for y, x in cells)
            for y, x in cells:
                p = band['bottom']-bottom+y, c+1+x
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
                out[p] = band['rail']
                specimen[y, x] = background
            if not valid:
                break
        if not valid:
            continue
        out[r+1:s-1, c+1:d-1] = specimen
        wr, ws, wc, wd = window
        a[r+1+wr:r+1+ws, c+1+wc:c+1+wd] = mask
        pair = {'input': np.rot90(a, turn).copy(), 'output': np.rot90(out, turn).copy()}
        try:
            rec = _analyze(pair['input'])
            feat = _features(rec)
            if not np.array_equal(rec['output'], pair['output']):
                continue
            if feat['boundary'] != boundary or not feat['spared'] or not feat['whole']:
                continue
            if _signature(_components(field, background, True)) != _signature(rec['groups']):
                continue
            if lesson == 'bridge' and (not feat['bridge'] or feat['colors'] < 2):
                continue
            if lesson == 'independent' and (not feat['independent'] or not feat['unequal'] or not feat['unused'] or not feat['bbox_decoy']):
                continue
            return pair
        except AssertionError:
            continue
    return None


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name+' outside legal range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    puzzle, seen = {'train': [], 'test': []}, set()
    # Fixed or mixed scene directions. No test introduces an unseen direction.
    mixed = bool(rng.integers(2))
    turns = list(map(int, rng.permutation(4)))
    train_turns = [turns[i % (2 if mixed else 1)] for i in range(nt)]
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            lesson = ('bridge' if i == 0 else 'independent' if i == 1 else None) if split == 'train' else None
            turn = train_turns[i] if split == 'train' else int(rng.choice(train_turns))
            for _ in range(240):
                pair = _construct(rng, lesson, split == 'test', turn)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split types')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    records, features, seen = {'train': [], 'test': []}, {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output dimensions')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            rec = _analyze(inp)
            _require(np.array_equal(rec['output'], out), 'incorrect selective dispatch')
            _require(not np.any(out == rec['mask']), 'overwrite must disappear')
            _require(_signature(rec['groups']) == _signature(_components(rec['restored'], rec['background'], True)),
                     'diagonal-only component ambiguity')
            feat = _features(rec)
            _require(feat['spared'] and feat['whole'], 'selection must distinguish cargo from evidence')
            # Check the unchanged frame and every unselected source component directly.
            specimen_out = np.rot90(out, rec['turn'])
            r, s, c, d = rec['frame']
            for color, cells in rec['groups']:
                expected = rec['background'] if cells & rec['covered'] else color
                _require(all(specimen_out[r+1+y, c+1+x] == expected for y, x in cells), 'source selection error')
            records[split].append(rec)
            features[split].append(feat)
    _require(any(f['bridge'] and f['colors'] >= 2 for f in features['train']), 'missing reconstruction/topology lesson')
    _require(any(f['independent'] and f['unequal'] for f in features['train']), 'missing independent alignment lesson')
    _require(any(f['unused'] for f in features['train']), 'unused band behavior must be demonstrated')
    _require(any(f['bbox_decoy'] for f in features['train']), 'cell-intersection selection must be demonstrated')
    _require(all(not f['boundary'] for f in features['train']) and all(f['boundary'] for f in features['test']),
             'interior-to-boundary query shift')
    _require({r['turn'] for r in records['test']} <= {r['turn'] for r in records['train']}, 'unseen direction')



def _regression_cases():
    """Literal small scenes: expected locations do not call routing helpers."""
    # Canonical frame rows 1..7, interior rows 2..6 and columns 2..13.
    field = np.zeros((5, 12), dtype=np.int8)
    main = {(0, 1), (0, 2), (0, 3), (1, 1), (2, 1), (2, 2), (2, 3)}
    for y, x in main:
        field[y, x] = field[y, 11-x] = 1
    field[1, 2] = field[1, 9] = 2
    field[4, 4] = field[4, 7] = 1

    def scene(specimen):
        a = np.zeros((18, specimen.shape[1]+4), dtype=np.int8)
        a[1:8, 1:-1] = 8
        a[2:7, 2:-2] = specimen
        a[9:14, 1:-1] = 1
        a[9:14, [0, -1]] = 3
        a[15:17, 1:-1] = 2
        a[15:17, [0, -1]] = 4
        return a

    for independent in (False, True):
        a = scene(field)
        expected = a.copy()
        for y, x in main | {(1, 2)}:
            expected[2+y, 2+x] = 0
        # The complete C keeps its relative row offsets at band bottom 13.
        for y, x in main:
            expected[11+y, 2+x] = 3
        expected[16, 4] = 4
        if independent:
            expected[6, 6] = 0
            expected[13, 6] = 3
            a[3:7, 3:7] = 9  # also queries the independent singleton
        else:
            a[3:4, 3:6] = 9  # cuts the hidden connecting spine
        _require(np.array_equal(solve(a), expected), 'literal whole-object/independent placement fixture')
        for turn in range(4):
            _require(np.array_equal(solve(np.rot90(a, turn)), np.rot90(expected, turn)),
                     'literal rotated fixture')

    # Query lies strictly inside a ring: bounding-box intersection is insufficient.
    field = np.zeros((5, 14), dtype=np.int8)
    for y in range(3):
        for x in range(1, 6):
            color = 1 if y in (0, 2) or x in (1, 5) else 2
            field[y, x] = field[y, 13-x] = color
    a = scene(field)
    expected = a.copy()
    expected[3, 4:7] = 0
    expected[16, 4:7] = 4
    a[3, 4:6] = 9
    _require(np.array_equal(solve(a), expected), 'query inside empty space of enclosing color')

    # Both members of a mirror orbit covered: no invented completion is allowed.
    erased = scene(field)
    erased[3, 8:10] = 9  # source columns 6 and 7 mirror each other
    # Two separately selected same-color points would occupy the very same target cell.
    collision_field = np.zeros((5, 12), dtype=np.int8)
    collision_field[0, 1] = collision_field[3, 1] = 1
    collision_field[:, ::-1] = np.maximum(collision_field, collision_field[:, ::-1])
    collision = scene(collision_field)
    collision[2:6, 3] = 9
    for invalid in (erased, collision):
        try:
            solve(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('erased orbit or actual stamp collision accepted')


def self_test():
    """Bounded semantic regressions and measured diversity over 42 seeds."""
    shapes, windows, masks, palettes, directions, band_counts, selected_counts = set(), set(), set(), set(), set(), set(), set()
    depths, source_shapes, signatures, edges = set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair['input']) for pair in p['train']+p['test']))
        if seed in (0, 21, 41):
            again = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], again[s])
                         for k in ('input', 'output')), 'nondeterminism')
        for split in ('train', 'test'):
            for pair in p[split]:
                rec = _analyze(pair['input'])
                feat = _features(rec)
                shapes.add(pair['input'].shape)
                source_shapes.add(rec['stage'].shape)
                windows.add(rec['window'])
                palettes.add((rec['background'], rec['frame_color'], rec['mask'],
                              tuple((b['route'], b['rail']) for b in rec['bands'])))
                directions.add(rec['turn'])
                band_counts.add(len(rec['bands']))
                selected_counts.add(len(rec['selected']))
                depths.add(tuple(b['bottom']-b['top']+1 for b in rec['bands']))
                for color, cells in rec['selected']:
                    r, s, c, d = _bbox(cells)
                    masks.add(tuple(sorted((y-r, x-c) for y, x in cells)))
                if split == 'test':
                    r, s, c, d = rec['window']
                    h, w = rec['stage'].shape
                    edges.update(i for i, yes in enumerate((r == 0, d == w, s == h, c == 0)) if yes)
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), 'input-only solve')
                _require(feat['spared'] and feat['whole'], 'query selection not observable')
    _require(len(signatures) == 42, 'identical puzzles')
    _require(len(shapes) >= 30 and len(source_shapes) >= 20, 'narrow geometry')
    _require(len(windows) >= 70 and len(masks) >= 40 and len(depths) >= 25, 'narrow structure')
    _require(len(palettes) >= 100 and directions == set(range(4)), 'narrow palette or direction')
    _require(band_counts == {1, 2, 3} and len(selected_counts) >= 4 and edges == set(range(4)), 'missing count/edge variation')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(42000+nt*10+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'counts ignored')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _regression_cases()
    # Check coordinate and palette equivariance without generation parameters.
    p = generate_puzzle(171, 2, 1)
    for pair in p['train']:
        rec = _analyze(pair['input'])
        perm = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)
        _require(np.array_equal(solve(perm[pair['input']]), perm[pair['output']]), 'palette equivariance')
        _require(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output'])), 'rotation equivariance')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong label accepted')
    # Corrupt a visible reflected witness while retaining the frame and request.
    rec = _analyze(p['train'][0]['input'])
    corrupted = rec['canonical'].copy()
    r, s, c, d = rec['frame']
    visible = (rec['stage'] != rec['mask']) & (rec['stage'][:, ::-1] != rec['mask'])
    y, x = map(int, np.argwhere(visible)[0])
    corrupted[r+1+y, c+1+x] = next(b['route'] for b in rec['bands'] if b['route'] != corrupted[r+1+y, c+1+x])
    try:
        solve(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory mirror evidence accepted')
    return {'seeds': 42, 'canvas_shapes': len(shapes), 'specimen_shapes': len(source_shapes),
            'window_geometries': len(windows), 'component_masks': len(masks),
            'depth_sequences': len(depths), 'palettes': len(palettes),
            'directions': sorted(directions), 'band_counts': sorted(band_counts),
            'selected_counts': sorted(selected_counts), 'test_edges': sorted(edges)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
