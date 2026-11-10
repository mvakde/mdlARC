"""Shape-keyed, occluded aperture projection. Standard library + NumPy only."""
from collections import Counter
from functools import lru_cache
import math
import numpy as np


_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in _N4:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells) + 1,
            min(c for r, c in cells), max(c for r, c in cells) + 1)


def _norm(cells):
    r0, _, c0, _ = _bounds(cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rotate(shape, q):
    for _ in range(q % 4):
        shape = tuple((c, -r) for r, c in shape)
    return _norm(shape)


@lru_cache(maxsize=2048)
def _orientations(shape, reflect=True):
    bases = (shape, tuple((r, -c) for r, c in shape)) if reflect else (shape,)
    return tuple(sorted({_rotate(s, q) for s in bases for q in range(4)}))


def _class(cells):
    return _orientations(_norm(cells))[0]


def _mask(shape):
    shape = _norm(shape)
    a = np.zeros((max(r for r, c in shape)+1, max(c for r, c in shape)+1), bool)
    for r, c in shape:
        a[r, c] = True
    return a


def _holes(mask):
    h, w = mask.shape
    return [c for c in _components(~mask)
            if not any(r in (0, h-1) or s in (0, w-1) for r, s in c)]


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((a >= 0) & (a <= 9)), "bad colors"
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _work_info(work, screen):
    objects, holes = [], []
    owner = np.full(work.shape, -1, int)
    for index, cells in enumerate(_components(work == screen)):
        r0, r1, c0, c1 = _bounds(cells)
        m = np.zeros((r1-r0, c1-c0), bool)
        for r, c in cells:
            m[r-r0, c-c0] = True
            owner[r, c] = index
        obj = dict(mask=m, row=r0, col=c0, right=c1-1, cells=cells, holes=[])
        for interior in _holes(m):
            rows = {r for r, c in interior}
            contact = {r for r, c in interior if c == m.shape[1]-2 and m[r, -1]}
            assert not contact or contact == rows, "partial-contact cavity is outside this family"
            hole = dict(shape=_norm(interior), cls=_class(interior), obj=index,
                        rows=tuple(sorted(r+r0 for r in rows)), active=bool(contact),
                        cells=tuple((r+r0, c+c0) for r, c in interior))
            obj['holes'].append(hole)
            holes.append(hole)
        objects.append(obj)
    winner = []
    for r in range(work.shape[0]):
        cols = np.flatnonzero(owner[r] >= 0)
        winner.append(int(owner[r, cols[-1]]) if len(cols) else -1)
    return objects, holes, owner, winner


def _project(info, mode='true'):
    """Logical beam rows; the optional modes are explicit shortcut diagnostics."""
    rows = np.full(info['height'], info['background'], np.int8)
    objects, assignment, winner = info['objects'], info['assignment'], info['winner']
    if mode == 'no_occlusion':
        # A nearer *emitter* replaces farther emitters, but silent bodies do not block.
        for obj in sorted(objects, key=lambda o: o['right']):
            for hole in obj['holes']:
                if hole['active']:
                    rows[list(hole['rows'])] = assignment[hole['cls']]
        return rows
    for r, index in enumerate(winner):
        if index < 0:
            continue
        obj = objects[index]
        for hole in obj['holes']:
            if r in hole['rows'] and (hole['active'] or mode == 'all_closed'):
                rows[r] = assignment[hole['cls']]
        if mode == 'column':
            local = r-obj['row']
            if obj['mask'].shape[1] >= 3 and not obj['mask'][local, -2]:
                # Treat any empty pixel behind the extreme as an aperture.
                if rows[r] == info['background']:
                    rows[r] = next(iter(assignment.values()))
    return rows


def _decode(cells, k):
    r0, r1, c0, c1 = _bounds(cells)
    if (r1-r0) % k or (c1-c0) % k or len(cells) % (k*k):
        return None
    blocks = Counter(((r-r0)//k, (c-c0)//k) for r, c in cells)
    if any(n != k*k for n in blocks.values()):
        return None
    return tuple(sorted(blocks))


def _recover(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size//2, "background must be the strict majority"
    candidates = []
    for marker in range(10):
        if marker == bg:
            continue
        for cells in _components(a == marker):
            r0, r1, c0, c1 = _bounds(cells)
            panel = a[r0:r1, c0:c1]
            if min(panel.shape) < 5:
                continue
            if not all(np.all(edge == marker) for edge in
                       (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
                continue
            for turn in range(4):
                p = np.rot90(panel, turn)
                for rail in range(2, p.shape[1]-2):
                    if not np.all(p[:, rail] == marker):
                        continue
                    receiver = p[1:-1, rail+1:-1]
                    work = p[1:-1, 1:rail]
                    colors = set(map(int, np.unique(work))) - {bg}
                    if len(colors) != 1 or marker in colors or not np.all(receiver == bg):
                        continue
                    screen = colors.pop()
                    objects, holes, owner, winner = _work_info(work, screen)
                    if not (2 <= len(objects) <= 6 and 2 <= len(holes) <= 8):
                        continue
                    outside = np.ones(a.shape, bool)
                    outside[r0:r1, c0:c1] = False
                    moat = np.zeros(a.shape, bool)
                    moat[max(0, r0-1):min(a.shape[0], r1+1),
                         max(0, c0-1):min(a.shape[1], c1+1)] = True
                    if np.any((a != bg) & outside & moat):
                        continue
                    external = []
                    for component in _components((a != bg) & outside):
                        palette = {int(a[r, c]) for r, c in component}
                        if len(palette) != 1:
                            break
                        color = palette.pop()
                        if color in (marker, screen):
                            break
                        external.append((color, component))
                    else:
                        large = [(color, c) for color, c in external if len(c) >= 4]
                        noise = [(color, c) for color, c in external if len(c) < 4]
                        if len(large) != len(holes):
                            continue
                        area = sum(len(c) for color, c in large)
                        logical_area = sum(len(hole['shape']) for hole in holes)
                        ratio, remainder = divmod(area, logical_area)
                        k = math.isqrt(ratio)
                        if remainder or k*k != ratio or k not in (2, 3):
                            continue
                        if max(d*k for d in receiver.shape) > 30:
                            continue
                        # Work in the same orientation as the canonical panel.
                        pieces = []
                        for color, c in large:
                            cm = np.zeros(a.shape, bool)
                            for r, s in c:
                                cm[r, s] = True
                            transformed = list(map(tuple, np.argwhere(np.rot90(cm, turn))))
                            shape = _decode(transformed, k)
                            if shape is None:
                                break
                            pieces.append(dict(color=color, shape=shape, cls=_class(shape)))
                        else:
                            hc = Counter(hole['cls'] for hole in holes)
                            pc = Counter(piece['cls'] for piece in pieces)
                            if hc != pc:
                                continue
                            grouped = {key: {piece['color'] for piece in pieces if piece['cls'] == key}
                                       for key in hc}
                            assert all(len(cs) == 1 for cs in grouped.values()), "ambiguous key colors"
                            assignment = {key: next(iter(cs)) for key, cs in grouped.items()}
                            info = dict(background=bg, marker=marker, screen=screen, scale=k,
                                        turn=turn, height=work.shape[0], width=receiver.shape[1],
                                        work=work.copy(), objects=objects, holes=holes,
                                        owner=owner, winner=winner, assignment=assignment,
                                        pieces=pieces, noise=noise, bounds=(r0, r1, c0, c1))
                            rows = _project(info)
                            logical = np.repeat(rows[:, None], receiver.shape[1], axis=1)
                            expanded = np.repeat(np.repeat(logical, k, 0), k, 1)
                            result = np.rot90(expanded, -turn).copy()
                            candidates.append((result, info))
    assert len(candidates) == 1, "expected a unique panel, direction, scale and colored interpretation"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only the supplied pixels; no examples, generator state or seed."""
    return _recover(input_grid)[0]


def _row_hole(mask, start, widths):
    right = mask.shape[1]-2
    for r, width in enumerate(widths, start):
        mask[r, right-width+1:right+1] = False


def _lesson_scene(rng, lesson):
    """Two complementary teaching scenes, with independently varied geometry."""
    placements = []
    if lesson == 0:
        arm = int(rng.integers(2, 5))
        widths = [1, 1, arm] if rng.random() < .5 else [arm, 1, 1]
        back = int(rng.integers(1, 4))
        far = np.ones((5+int(rng.integers(2)), arm+1+back), bool)
        _row_hole(far, 1, widths)
        area = sum(widths)
        near_widths = [area//2, area-area//2]
        if rng.random() < .5:
            near_widths.reverse()
        near = np.ones((len(near_widths)+2, max(near_widths)+2+int(rng.integers(3))), bool)
        _row_hole(near, 1, near_widths)
        needed = max(far.shape[0], 2+near.shape[0])
        h = int(rng.integers(needed, 11))
        top = int(rng.integers(h-needed+1))
        left = int(rng.integers(3))
        front = left+far.shape[1]+int(rng.integers(1, 4))
        placements = [(far, top, left), (near, top+2, front)]
        w = front+near.shape[1]+int(rng.integers(3))
    else:
        upper = int(rng.integers(2, 4))
        lower = int(rng.integers(1, 5-upper))
        h = upper+lower+6
        widths = list(map(int, rng.integers(1, 4, size=upper)))
        if len(set(widths)) == 1:
            widths[0] = widths[0] % 3+1
        low_width = int(rng.integers(1, 4))
        far = np.ones((h, max(max(widths), low_width)+2+int(rng.integers(3))), bool)
        _row_hole(far, 1, widths)
        _row_hole(far, upper+4, [low_width]*lower)
        # A thin local front wall does not activate a hole behind a protruding tail.
        decoy_width = int(rng.integers(1, 4))
        body = np.ones((3, decoy_width+2), bool)
        _row_hole(body, 1, [decoy_width])
        tail = int(rng.integers(1, 3))
        decoy = np.zeros((3, body.shape[1]+tail), bool)
        decoy[:, :body.shape[1]] = body
        decoy[int(rng.choice([0, 2])), body.shape[1]:] = True
        opened = np.ones((lower+2, low_width+2), bool)
        _row_hole(opened, 1, [low_width]*lower)
        if rng.random() < .5:
            opened[1:lower+1, 0] = False
        else:
            opened[0, -2] = False
        left = int(rng.integers(2))
        front = left+far.shape[1]+int(rng.integers(1, 4))
        open_col = front+int(rng.integers(2))
        placements = [(far, 0, left), (decoy, upper, front), (opened, upper+4, open_col)]
        w = max(front+decoy.shape[1], open_col+opened.shape[1])+int(rng.integers(3))
    work = np.zeros((h, w), bool)
    for mask, r, c in placements:
        work[r:r+mask.shape[0], c:c+mask.shape[1]] |= mask
    return work


def _random_screen(rng, kind):
    h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    if kind == 'multi':
        h = int(rng.integers(5, 8))
    m = np.ones((h, w), bool)
    if kind == 'solid':
        # A connected staircase or bar; no inferred signal from outline color.
        if rng.random() < .5:
            for r in range(h):
                m[r, :int(rng.integers(w))] = False
        return m
    length = int(rng.integers(1, min(3, h-2)+1))
    widths = list(map(int, rng.integers(1, min(3, w-2)+1, size=length)))
    start = int(rng.integers(1, h-length))
    if kind == 'multi':
        _row_hole(m, 1, [int(rng.integers(1, w-1))])
        _row_hole(m, h-2, [int(rng.integers(1, w-1))])
    else:
        _row_hole(m, start, widths)
    if kind == 'open':
        side = int(rng.integers(3))
        if side == 0:
            left = w-1-widths[0]
            m[start, :left+1] = False
        elif side == 1:
            m[:start+1, -2] = False
        else:
            m[start+length-1:, -2] = False
    elif kind == 'decoy':
        extension = int(rng.integers(1, 3))
        a = np.zeros((h, w+extension), bool)
        a[:, :w] = m
        a[int(rng.choice([0, h-1])), w:] = True
        m = a
    # Erode corners only when the exact cavity set and connectivity survive.
    target = {p for hole in _holes(m) for p in hole}
    for _ in range(int(rng.integers(5))):
        r, c = int(rng.choice([0, m.shape[0]-1])), int(rng.integers(m.shape[1]))
        trial = m.copy()
        trial[r, c] = False
        if len(_components(trial)) == 1 and {p for hole in _holes(trial) for p in hole} == target:
            rr, cc = np.nonzero(trial)
            if rr.min() == 0 and rr.max() == m.shape[0]-1 and cc.max() == m.shape[1]-1:
                m = trial
    return m


def _free(blocked, shape):
    h, w = shape
    if h > blocked.shape[0] or w > blocked.shape[1]:
        return np.empty((0, 2), int)
    s = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = s[h:, w:]-s[:-h, w:]-s[h:, :-w]+s[:-h, :-w]
    return np.argwhere(sums == 0)


def _block(blocked, r, c, shape):
    h, w = shape
    blocked[max(0, r-1):min(blocked.shape[0], r+h+1),
            max(0, c-1):min(blocked.shape[1], c+w+1)] = True


def _random_scene(rng, testing):
    h = int(rng.integers(7, 11))
    w = int(rng.integers(12 if testing else 9, 20))
    target = int(rng.integers(3 if testing else 2, 6))
    work, blocked = np.zeros((h, w), bool), np.zeros((h, w), bool)
    for index in range(target):
        for _ in range(20):
            kind = 'active' if index < 2 else str(rng.choice(['active', 'multi', 'open', 'decoy', 'solid']))
            m = _random_screen(rng, kind)
            positions = _free(blocked, m.shape)
            if not len(positions):
                continue
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            work[r:r+m.shape[0], c:c+m.shape[1]][m] = True
            _block(blocked, r, c, m.shape)
            break
        else:
            break
    return work


def _pack(rng, panel, loose, bg, noise_color, key_colors):
    """Pack in two dimensions, with no board-side or row-rank convention."""
    for attempt in range(45):
        h = int(rng.integers(max(panel.shape[0]+3, 20), 31)) if attempt < 25 else 30
        w = int(rng.integers(max(panel.shape[1], 20), 31)) if attempt < 25 else 30
        a, blocked = np.full((h, w), bg, np.int8), np.zeros((h, w), bool)
        items = [(np.ones(panel.shape, bool), None)] + loose
        order = [0] + sorted(range(1, len(items)), key=lambda i: -items[i][0].size)
        for index in order:
            mask, color = items[index]
            positions = _free(blocked, mask.shape)
            if not len(positions):
                break
            r, c = map(int, positions[int(rng.integers(len(positions)))])
            view = a[r:r+mask.shape[0], c:c+mask.shape[1]]
            if index == 0:
                view[:] = panel
            else:
                view[mask] = color
            _block(blocked, r, c, mask.shape)
        else:
            for _ in range(int(rng.integers(0, 9))):
                shape = (((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0), (1, 1)))[int(rng.integers(3))]
                m = _mask(_rotate(shape, int(rng.integers(4))))
                positions = _free(blocked, m.shape)
                if not len(positions):
                    break
                r, c = map(int, positions[int(rng.integers(len(positions)))])
                color = int(rng.choice(key_colors)) if rng.random() < .5 else noise_color
                a[r:r+m.shape[0], c:c+m.shape[1]][m] = color
                _block(blocked, r, c, m.shape)
            if np.count_nonzero(a == bg) > a.size//2:
                return a
    return None


def _beam_label(work, screen, bg, assignment, width, k):
    """Builder check via ray traversal, rather than the solver's nearest-owner scan."""
    objects, holes, owner, winner = _work_info(work, screen)
    rows = np.full(work.shape[0], bg, np.int8)
    for hole in holes:
        if not hole['active']:
            continue
        obj = objects[hole['obj']]
        for r in hole['rows']:
            # The emitting object's own front wall is crossed; other screens are opaque.
            if not np.any(owner[r, obj['right']+1:] >= 0):
                rows[r] = assignment[hole['cls']]
    band = np.repeat(rows[:, None], width, axis=1)
    return np.repeat(np.repeat(band, k, 0), k, 1)


def _make_pair(rng, k, turn, lesson=None, testing=False):
    for _ in range(160):
        binary = _lesson_scene(rng, lesson) if lesson is not None else _random_scene(rng, testing)
        if binary.shape[1] > 23:
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, marker, screen, noise_color = palette[:4]
        work = np.where(binary, screen, bg).astype(np.int8)
        objects, holes, owner, winner = _work_info(work, screen)
        if not (2 <= len(objects) <= 6 and 2 <= len(holes) <= 8):
            continue
        if testing and len(objects) < 3:
            continue
        if sum(len(hole['shape']) for hole in holes) > (28 if k == 2 else 19):
            continue
        classes = list(dict.fromkeys(hole['cls'] for hole in holes))
        cardinality = len(classes) if lesson == 0 else int(rng.integers(1, min(6, len(classes))+1))
        assignment = {cls: palette[4+i % cardinality] for i, cls in enumerate(classes)}
        width = int(rng.integers(1, min(5, 27-binary.shape[1])+1))
        info = dict(objects=objects, holes=holes, assignment=assignment, winner=winner,
                    background=bg, height=work.shape[0])
        rows = _project(info)
        if lesson is None and not np.any(rows != bg):
            continue
        if testing and np.array_equal(rows, _project(info, 'no_occlusion')):
            continue
        loose = []
        rotation_forced = False
        for hole in holes:
            orientations = _orientations(hole['shape'])
            if lesson == 0 and len(_orientations(hole['shape'], False)) < len(orientations):
                orientations = tuple(s for s in orientations if s not in _orientations(hole['shape'], False))
            if lesson == 1 and not rotation_forced and hole['active'] and any(
                    winner[r] == hole['obj'] for r in hole['rows']):
                alternatives = tuple(s for s in _orientations(hole['shape'], False) if s != hole['shape'])
                if alternatives:
                    orientations = alternatives
                    rotation_forced = True
            shape = orientations[int(rng.integers(len(orientations)))]
            macro = np.repeat(np.repeat(_mask(shape), k, 0), k, 1)
            loose.append((macro, assignment[hole['cls']]))
        rail = binary.shape[1]+1
        panel = np.full((binary.shape[0]+2, binary.shape[1]+width+3), marker, np.int8)
        panel[1:-1, 1:-1] = bg
        panel[1:-1, 1:rail] = work
        panel[:, rail] = marker
        canvas = _pack(rng, panel, loose, bg, noise_color, list(assignment.values()))
        if canvas is None:
            continue
        label = _beam_label(work, screen, bg, assignment, width, k)
        pair = {'input': np.rot90(canvas, turn).copy(), 'output': np.rot90(label, turn).copy()}
        expected, recovered = _recover(pair['input'])
        assert np.array_equal(label[:, 0][::k], rows), "builder ray traversal differs from scan"
        assert np.array_equal(expected, pair['output']), "builder differs from input-only reconstruction"
        return pair
    raise RuntimeError('bounded feasible scene construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    regime, base = int(rng.integers(4)), int(rng.integers(4))
    if regime < 2:
        train_k, test_k = (2, 3) if regime == 0 else (3, 2)
        scales = [train_k]*a + [test_k]*b
        turns = [base]*(a+b)
    elif regime == 2:
        scales = [2, 3] + list(map(int, rng.integers(2, 4, a+b-2)))
        turns = [base, (base+1) % 4] + list(map(int, rng.integers(4, size=a+b-2)))
    else:
        scales = [int(rng.integers(2, 4))]*(a+b)
        turns = [base]*a + [(base+int(rng.integers(1, 4))) % 4]*b
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(a+b):
        split = 'train' if i < a else 'test'
        for _ in range(12):
            pair = _make_pair(rng, scales[i], turns[i], lesson=i if i < 2 else None, testing=i >= a)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError('distinct-input construction exhausted')
    # The teaching order is incidental; witnesses remain present for every count.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(out, expected), 'wrong output'
            k, turn = info['scale'], info['turn']
            canonical = np.rot90(out, turn)
            assert canonical.shape == (info['height']*k, info['width']*k)
            assert all(np.all(row == row[0]) for row in canonical), 'ray does not span receiver'
            assert set(map(int, np.unique(out))) <= {info['background']} | set(info['assignment'].values())
            assert len(info['pieces']) == len(info['holes'])
            assert all(1 <= len(hole['shape']) <= 9 for hole in info['holes'])
            assert all(len(c) <= 3 for color, c in info['noise'])
            assert np.array_equal(out, solve(inp.copy()))
            if split == 'train':
                training.append(info)
    assert any(not np.array_equal(_project(i), _project(i, 'all_closed')) for i in training), 'no leading-edge witness'
    assert any(not np.array_equal(_project(i), _project(i, 'column')) for i in training), 'no enclosure witness'
    assert any(not np.array_equal(_project(i), _project(i, 'no_occlusion')) for i in training), 'no opacity witness'
    equal_area = reflected = rotated = replacement = multirow = False
    for info in training:
        rows = _project(info)
        visible = [h for h in info['holes'] if h['active'] and any(
            info['winner'][r] == h['obj'] for r in h['rows'])]
        equal_area |= any(len(x['shape']) == len(y['shape']) and x['cls'] != y['cls']
                          and info['assignment'][x['cls']] != info['assignment'][y['cls']]
                          for x in visible for y in visible)
        for hole in visible:
            candidates = [p for p in info['pieces'] if p['cls'] == hole['cls']]
            if len(candidates) == 1:
                p = candidates[0]['shape']
                reflected |= hole['shape'] not in _orientations(p, False)
                rotated |= hole['shape'] != p and hole['shape'] in _orientations(p, False)
            multirow |= sum(info['winner'][r] == hole['obj'] for r in hole['rows']) >= 2
        for r in range(info['height']):
            emitters = [h for h in info['holes'] if h['active'] and r in h['rows']]
            replacement |= len({info['assignment'][h['cls']] for h in emitters}) >= 2 and rows[r] != info['background']
    assert equal_area and reflected and rotated and replacement and multirow, 'teaching evidence incomplete'


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed variation audit, API boundaries and corruption diagnostics."""
    signatures, shapes, scenes, classes, scales, directions, counts, colors = (set() for _ in range(8))
    repeat_input = None
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                answer, i = _recover(pair['input'])
                assert np.array_equal(answer, pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                scenes.add((i['work'].shape, (i['work'] == i['screen']).tobytes()))
                classes.update(h['cls'] for h in i['holes'])
                scales.add(i['scale'])
                directions.add(i['turn'])
                counts.add((len(i['objects']), len(i['holes'])))
                colors.add((i['background'], i['marker'], i['screen']))
                if repeat_input is None and len(i['holes']) > len(i['assignment']):
                    repeat_input = pair['input'].copy()
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(scenes) >= 150 and len(classes) >= 20
    assert scales == {2, 3} and directions == set(range(4))
    assert len(counts) >= 8 and len(colors) >= 100
    p = generate_puzzle(173, 2, 1)
    assert _puzzle_key(p) == _puzzle_key(generate_puzzle(173, 2, 1))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            q = generate_puzzle(100+a*10+b, a, b)
            assert (len(q['train']), len(q['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong label accepted')
    # A macrocell with a missing pixel cannot be interpreted as a complete key.
    inp = p['train'][0]['input'].copy()
    _, info = _recover(inp)
    r0, r1, c0, c1 = info['bounds']
    outside = np.ones(inp.shape, bool)
    outside[r0:r1, c0:c1] = False
    component = next(c for c in _components((inp != info['background']) & outside) if len(c) >= 4)
    inp[min(component)] = info['background']
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError('broken key accepted')
    # Repeated congruent holes cannot acquire two different colors via key order.
    assert repeat_input is not None, 'no repeated congruent class in variation audit'
    _, info = _recover(repeat_input)
    repeated = next(cls for cls, n in Counter(h['cls'] for h in info['holes']).items() if n > 1)
    outside = np.ones(repeat_input.shape, bool)
    r0, r1, c0, c1 = info['bounds']
    outside[r0:r1, c0:c1] = False
    component = next(c for c in _components((repeat_input != info['background']) & outside)
                     if len(c) >= 4 and _class(_decode(c, info['scale'])) == repeated)
    old = int(repeat_input[component[0]])
    new = next(c for c in range(10) if c not in (old, info['background'], info['marker'], info['screen']))
    for r, c in component:
        repeat_input[r, c] = new
    try:
        solve(repeat_input)
    except AssertionError:
        pass
    else:
        raise AssertionError('differently colored congruent keys accepted')
    # The family resolves no partial-contact interpretation by accident.
    partial = np.ones((4, 6), np.int8)
    partial[1, 3:5] = 0
    partial[2, 3] = 0
    try:
        _work_info(partial, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial-contact cavity accepted')
    # Two singleton apertures are correctly keyed, but a front solid bar blocks all rows.
    silent = np.zeros((20, 20), np.int8)
    panel = np.full((9, 12), 8, np.int8)
    panel[1:-1, 1:-1] = 0
    panel[:, 8] = 8
    work = panel[1:-1, 1:8]
    work[0:3, 0:3] = 4
    work[1, 1] = 0
    work[4:7, 0:4] = 4
    work[5, 1] = 0
    work[:, 5:7] = 4
    silent[:9, :12] = panel
    silent[11:13, 1:3] = silent[11:13, 5:7] = 5
    for turn in range(4):
        expected = np.rot90(np.zeros((14, 4), np.int8), turn)
        assert np.array_equal(solve(np.rot90(silent, turn)), expected)


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, structural variation, all legal counts, negative checks)')
