"""Defect-catching beams: rail-facing cavities inspect a damaged layered surface.

Only NumPy and the standard library are used. No parent imports or file reads.
The solver infers every role, source mask, rectangle box and defect from input.
"""
from itertools import permutations
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= a) & (a <= 9)), "ARC colors required")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr)+1, max(cc)+1


def _holes(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if all(0 < r < h-1 and 0 < c < w-1 for r, c in cells)]


def _source_details(work, bg, fg):
    objects = []
    for cells in _components(work == fg):
        r0, c0, r1, c1 = _bbox(cells)
        mask = np.zeros((r1-r0, c1-c0), dtype=bool)
        for r, c in cells:
            mask[r-r0, c-c0] = True
        holes = _holes(mask)
        active, decoy = [], []
        for hole in holes:
            contact = {r for r, c in hole if c == mask.shape[1]-2 and mask[r, -1]}
            if contact:
                _require(contact == {r for r, c in hole}, "partial leading contact excluded")
                active.append(hole)
            else:
                decoy.append(hole)
        enclosed = {p for hole in holes for p in hole}
        naive = {r+r0 for r in range(1, mask.shape[0]-1)
                 if mask.shape[1] >= 3 and not mask[r, -2]}
        open_rows = {r+r0 for r in range(1, mask.shape[0]-1)
                     if mask.shape[1] >= 3 and not mask[r, -2]
                     and (r, mask.shape[1]-2) not in enclosed}
        objects.append(dict(cells=cells, bbox=(r0, c0, r1, c1), mask=mask,
                            shift=work.shape[1]-c1, active=active, decoy=decoy,
                            rows={r+r0 for hole in active for r, c in hole},
                            decoy_rows={r+r0 for hole in decoy for r, c in hole},
                            naive=naive, open_rows=open_rows))
    objects.sort(key=lambda o: o['bbox'])
    for a, b in zip(objects, objects[1:]):
        _require(a['bbox'][2] < b['bbox'][0], "source bands must be separated")
    _require(objects, "no source objects")
    return objects


def _normalize(a):
    """Find the exclusive full-span rail and identify sides by their role sets."""
    choices = []
    for turn in range(4):
        g = np.rot90(a, turn)
        h, w = g.shape
        for col in range(1, w-1):
            rail = int(g[0, col])
            if not np.all(g[:, col] == rail) or np.count_nonzero(g == rail) != h:
                continue
            left = set(map(int, np.unique(g[:, :col])))
            right = set(map(int, np.unique(g[:, col+1:])))
            common = left & right
            if len(left) == 2 and len(right) == 3 and len(common) == 1 and rail not in left | right:
                bg = next(iter(common))
                fg = next(iter(left - common))
                fills = tuple(sorted(right - common))
                choices.append((g.copy(), turn, col, bg, fg, rail, fills))
    _require(len(choices) == 1, "unique rail and source/receiver sides required")
    return choices[0]


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _geometry(boxes):
    if not 2 <= len(boxes) <= 4:
        return False
    degrees = [0]*len(boxes)
    edges = 0
    for i, a in enumerate(boxes):
        if min(a[2]-a[0], a[3]-a[1]) < 4:
            return False
        for j in range(i):
            if _overlap(a, boxes[j]):
                if a[4] == boxes[j][4]:
                    return False
                degrees[i] += 1
                degrees[j] += 1
                edges += 1
    return edges == len(boxes)-1 and sorted(degrees) == [1, 1]+[2]*(len(boxes)-2)


def _paint(shape, bg, boxes):
    clean = np.full(shape, bg, dtype=np.int8)
    for r0, c0, r1, c1, color in boxes:
        clean[r0:r1, c0:c1] = color
    return clean


def _large_boxes(g, fills):
    boxes = []
    for color in fills:
        for cells in _components(g == color):
            if len(cells) >= 10:
                boxes.append((*_bbox(cells), color))
    return tuple(sorted(boxes))


def _scene_supported(clean, bg, boxes):
    if not _geometry(boxes) or np.count_nonzero(clean == bg) < 8:
        return False
    visible_boxes = []
    for color in sorted({b[4] for b in boxes}):
        for cells in _components(clean == color):
            if len(cells) < 12:
                return False
            visible_boxes.append((*_bbox(cells), color))
    return tuple(sorted(visible_boxes)) == tuple(sorted(boxes))


def _fit_receiver(receiver, bg, fills):
    """Exhaustive covering-order inference for pixel-derived large component boxes.

    Family inputs preserve these boxes despite sparse corruption. Different
    orders yielding an identical visible surface are equivalent, not a tie.
    """
    boxes = _large_boxes(receiver, fills)
    _require(_geometry(boxes), "unsupported rectangle component geometry")
    scenes = {}
    for order in permutations(boxes):
        clean = _paint(receiver.shape, bg, order)
        if not _scene_supported(clean, bg, boxes):
            continue
        key = clean.tobytes()
        loss = int(np.count_nonzero(clean != receiver))
        scenes[key] = (loss, clean, order)
    _require(scenes, "no recoverable layered surface")
    ranked = sorted(scenes.values(), key=lambda item: (item[0], item[1].tobytes()))
    loss, clean, order = ranked[0]
    _require(loss <= max(6, int(receiver.size*.12)), "excessive surface corruption")
    _require(all(other[0] >= loss+2 for other in ranked[1:]), "uncertain visible covering order")
    for color in fills:
        visible = clean == color
        _require(np.count_nonzero(visible & (receiver == color)) >= .80*visible.sum(),
                 "insufficient surviving fill evidence")
    return clean, tuple(order)


def _select(erasures, rows):
    return {(r, int(np.flatnonzero(erasures[r])[0])) for r in rows if erasures[r].any()}


def _stamp(clean, centers, fills, rail):
    """Simultaneous rings, clipped to receiver; centers override every ring."""
    demands = {}
    h, w = clean.shape
    for r, c in sorted(centers):
        opposite = fills[1] if int(clean[r, c]) == fills[0] else fills[0]
        for rr in range(max(0, r-1), min(h, r+2)):
            for cc in range(max(0, c-1), min(w, c+2)):
                if (rr, cc) in centers:
                    continue
                _require((rr, cc) not in demands or demands[rr, cc] == opposite,
                         "conflicting ring assignments excluded")
                demands[rr, cc] = opposite
    out = clean.copy()
    for p, color in demands.items():
        out[p] = color
    for p in centers:
        out[p] = rail
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    g, turn, col, bg, fg, rail, fills = _normalize(a)
    objects = _source_details(g[:, :col], bg, fg)
    receiver = g[:, col+1:]
    clean, boxes = _fit_receiver(receiver, bg, fills)
    erasures = (receiver == bg) & (clean != bg)
    rows = set().union(*(o['rows'] for o in objects))
    centers = _select(erasures, rows)
    expected = g.copy()
    expected[:, :col] = bg
    for o in objects:
        for r, c in o['cells']:
            expected[r, c+o['shift']] = fg
    expected[:, col+1:] = _stamp(clean, centers, fills, rail)
    return dict(canonical=g, turn=turn, col=col, bg=bg, fg=fg, rail=rail, fills=fills,
                objects=objects, receiver=receiver, clean=clean, boxes=boxes,
                erasures=erasures, rows=rows, centers=centers,
                expected=np.rot90(expected, -turn).copy())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive an answer from this input alone, without examples or RNG state."""
    return _analyze(input_grid)['expected']


def _ri(rng, low, high):
    return int(rng.integers(low, high+1))


def _intervals(rng, height, count):
    spare = height-(2*count+1)
    extras = rng.multinomial(spare, [1/(2*count+1)]*(2*count+1))
    row = 1+int(extras[0])
    intervals = []
    for i in range(count):
        end = row+1+int(extras[2*i+1])
        intervals.append((row, end))
        row = end+1+int(extras[2*i+2])
    return intervals


def _mask(rng, kind):
    if kind == 'solid':
        h, w = _ri(rng, 2, 6), _ri(rng, 2, 7)
        m = np.ones((h, w), dtype=bool)
        if h > 2 and w > 2 and rng.random() < .6:
            m[1:-1, 0] = False
        return m
    h = _ri(rng, 5 if kind in ('multi', 'mixed') else 4,
            11 if kind in ('multi', 'mixed') else 7)
    w = _ri(rng, 6 if kind == 'mixed' else 3, 9 if kind == 'mixed' else 7)
    m = np.ones((h, w), dtype=bool)
    if kind == 'mixed':
        separator = _ri(rng, 2, w-3)
        for left, right in ((1, separator-1), (separator+1, w-2)):
            count = _ri(rng, 1, min(3, (h-1)//2))
            for start, stop in _intervals(rng, h, count):
                for r in range(start, stop):
                    back = _ri(rng, left, right)
                    m[r, back:right+1] = False
    elif kind == 'multi':
        for start, stop in _intervals(rng, h, _ri(rng, 2, min(3, (h-1)//2))):
            for r in range(start, stop):
                m[r, _ri(rng, 1, w-2):w-1] = False
    else:
        m[1:-1, 1:-1] = False
        if kind == 'open':
            side = _ri(rng, 0, 2)
            if side == 0:
                m[1:-1, 0] = False
            else:
                m[0 if side == 1 else -1, -2] = False
        elif kind == 'decoy':
            extension = _ri(rng, 1, 2)
            expanded = np.zeros((h, w+extension), dtype=bool)
            expanded[:, :w] = m
            expanded[int(rng.choice([0, h-1])), w:] = True
            m = expanded
    return m


def _contrasts(rng):
    h, w = _ri(rng, 4, 6), _ri(rng, 3, 6)
    closed = np.ones((h, w), dtype=bool)
    closed[1:-1, 1:-1] = False
    opened = closed.copy()
    if rng.random() < .5:
        opened[1:-1, 0] = False
    else:
        opened[0 if rng.random() < .5 else -1, -2] = False
    extension = _ri(rng, 1, 2)
    recessed = np.zeros((h, w+extension), dtype=bool)
    recessed[:, :w] = closed
    recessed[int(rng.choice([0, h-1])), w:] = True
    return closed, opened, recessed


def _work_layout(rng, mandatory, is_test):
    masks = [m.copy() for m in mandatory]
    if not masks:
        masks = [_mask(rng, str(rng.choice(['multi', 'mixed']))) if is_test
                 else _mask(rng, 'active')]
    target = _ri(rng, len(masks), max(len(masks), 4))
    for _ in range(target-len(masks)):
        m = _mask(rng, str(rng.choice(['solid', 'open', 'decoy', 'active', 'multi', 'mixed'])))
        if sum(x.shape[0] for x in masks)+m.shape[0]+len(masks)+2 <= 30:
            masks.append(m)
    rng.shuffle(masks)
    minimum = sum(m.shape[0] for m in masks)+len(masks)+1
    h = _ri(rng, max(17, minimum), 30)
    widest = max(m.shape[1] for m in masks)
    col = _ri(rng, widest+1, min(widest+5, 20))
    receiver_w = _ri(rng, 9, min(17, 29-col))
    work = np.zeros((h, col), dtype=np.int8)
    extras = rng.multinomial(h-minimum, [1/(len(masks)+1)]*(len(masks)+1))
    row = 1+int(extras[0])
    shifts = [_ri(rng, 0, col-m.shape[1]) for m in masks]
    if mandatory and not any(shifts):
        shifts[0] = 1
    for i, (m, shift) in enumerate(zip(masks, shifts)):
        start = col-m.shape[1]-shift
        view = work[row:row+m.shape[0], start:start+m.shape[1]]
        view[m] = 1
        row += m.shape[0]+1+int(extras[i+1])
    return work, receiver_w


def _sample_surface(rng, shape, count):
    h, w = shape
    boxes = []
    # Free independent boxes, not an edge-anchored zigzag construction.
    for _ in range(220):
        rh, rw = _ri(rng, 4, min(h, 16)), _ri(rng, 4, w)
        r0, c0 = _ri(rng, 0, h-rh), _ri(rng, 0, w-rw)
        if rng.random() < .18:
            r0 = int(rng.choice([0, h-rh]))
        if rng.random() < .18:
            c0 = int(rng.choice([0, w-rw]))
        b = (r0, c0, r0+rh, c0+rw, 1+len(boxes)%2)
        if boxes:
            overlap = _overlap(b, boxes[-1])
            smaller = min(rh*rw, (boxes[-1][2]-boxes[-1][0])*(boxes[-1][3]-boxes[-1][1]))
            if not 2 <= overlap <= .45*smaller or any(_overlap(b, x) for x in boxes[:-1]):
                continue
        boxes.append(b)
        if len(boxes) == count:
            order = [boxes[int(i)] for i in rng.permutation(count)]
            clean = _paint(shape, 0, order)
            return clean if _scene_supported(clean, 0, boxes) else None
    return None


def _corrupt_surface(rng, clean, objects, teach_color, contrast):
    """Plant evidence, then add independently sampled accepted sparse errors."""
    rows = set().union(*(o['rows'] for o in objects))
    open_rows = set().union(*(o['open_rows'] for o in objects))
    decoy_rows = set().union(*(o['decoy_rows'] for o in objects))-rows
    inp = clean.copy()
    used = set()
    locked_centers = set()

    def check(trial):
        try:
            recovered, _ = _fit_receiver(trial, 0, (1, 2))
            if not np.array_equal(recovered, clean):
                return False
            centers = _select((trial == 0) & (clean != 0), rows)
            if not locked_centers.issubset(centers):
                return False
            _stamp(clean, centers, (1, 2), 3)
            return True
        except AssertionError:
            return False

    # A vertical adjacent same-fill cluster also demonstrates center precedence.
    if teach_color is not None:
        candidates = [(r, c) for r in sorted(rows) if r+1 in rows
                      for c in range(1, clean.shape[1]-2)
                      if clean[r, c] == clean[r+1, c] == teach_color
                      and np.count_nonzero(clean[r, c+1:] != 0) >= 1
                      and np.count_nonzero(clean[r, :c] != 0) >= 1]
        rng.shuffle(candidates)
        for r, c in candidates:
            later = [int(x) for x in np.flatnonzero(clean[r, c+1:] != 0)+c+1]
            before = [int(x) for x in np.flatnonzero(clean[r, :c] != 0)]
            rng.shuffle(later)
            rng.shuffle(before)
            trial = inp.copy()
            points = [(r, c), (r+1, c), (r, later[0])]
            for p in points:
                trial[p] = 0
            wrong = (r, before[0])
            trial[wrong] = 3-int(clean[wrong])
            if check(trial):
                inp = trial
                used.update(points+[wrong])
                locked_centers.update([(r, c), (r+1, c)])
                break
        else:
            return None

        if contrast == 'decoy':
            # This demonstration teaches both complementary directions within
            # one visible palette, regardless of palette variation elsewhere.
            other = 3-teach_color
            candidates = [(r, int(c)) for r in rows for c in np.flatnonzero(clean[r] == other)
                          if all(max(abs(r-rr), abs(int(c)-cc)) > 2 for rr, cc in locked_centers)]
            rng.shuffle(candidates)
            for p in candidates:
                trial = inp.copy()
                trial[p] = 0
                if check(trial) and p in _select((trial == 0) & (clean != 0), rows):
                    inp = trial
                    used.add(p)
                    locked_centers.add(p)
                    break
            else:
                return None
    else:
        candidates = [(r, int(c)) for r in rows for c in np.flatnonzero(clean[r] != 0)]
        rng.shuffle(candidates)
        for p in candidates:
            trial = inp.copy()
            trial[p] = 0
            if check(trial):
                inp = trial
                used.add(p)
                break
        else:
            return None

    # One uncaught erasure lies on a row which the specified false gate would emit.
    false_rows = open_rows if contrast == 'open' else decoy_rows if contrast == 'decoy' else set()
    if contrast:
        candidates = [(r, int(c)) for r in false_rows for c in np.flatnonzero(clean[r] != 0)]
        rng.shuffle(candidates)
        for p in candidates:
            trial = inp.copy()
            trial[p] = 0
            if check(trial):
                inp = trial
                used.add(p)
                break
        else:
            return None

    # Additional erased defects, wrong-fill errors and exterior speckles.
    target = _ri(rng, 3, max(3, min(16, int(np.count_nonzero(clean)*.065))))
    candidates = [tuple(map(int, p)) for p in np.argwhere(clean != 0)]
    rng.shuffle(candidates)
    for p in candidates:
        if np.count_nonzero((inp == 0) & (clean != 0)) >= target:
            break
        if p in used:
            continue
        trial = inp.copy()
        trial[p] = 0
        if check(trial):
            inp = trial
            used.add(p)
    wrong_target = _ri(rng, 0, 4)
    rng.shuffle(candidates)
    for p in candidates:
        if wrong_target <= 0:
            break
        if p in used:
            continue
        trial = inp.copy()
        trial[p] = 3-int(clean[p])
        if check(trial):
            inp = trial
            used.add(p)
            wrong_target -= 1
    exterior = [tuple(map(int, p)) for p in np.argwhere(clean == 0)]
    rng.shuffle(exterior)
    speckles = _ri(rng, 0, 5)
    for p in exterior:
        if speckles <= 0:
            break
        trial = inp.copy()
        trial[p] = _ri(rng, 1, 2)
        if check(trial):
            inp = trial
            speckles -= 1
    return inp


def _pair(rng, mandatory, is_test, teach_color, contrast, palette, turn):
    for _ in range(240):
        work, receiver_w = _work_layout(rng, mandatory, is_test)
        objects = _source_details(work, 0, 1)
        count = 2 if contrast == 'open' else 3 if contrast == 'decoy' else int(rng.choice([2, 3, 4], p=[.40, .45, .15]))
        clean = _sample_surface(rng, (work.shape[0], receiver_w), count)
        if clean is None:
            continue
        damaged = _corrupt_surface(rng, clean, objects, teach_color, contrast)
        if damaged is None:
            continue
        bg, fg, rail, f1, f2 = palette
        receiver_map = np.array([bg, f1, f2], dtype=np.int8)
        g = np.full((work.shape[0], work.shape[1]+1+receiver_w), bg, dtype=np.int8)
        g[:, :work.shape[1]][work == 1] = fg
        g[:, work.shape[1]] = rail
        g[:, work.shape[1]+1:] = receiver_map[damaged]
        inp = np.rot90(g, turn).copy()
        return {'input': inp, 'output': solve(inp)}
    return _fallback_pair(rng, mandatory, is_test, teach_color, contrast, palette, turn)


def _fallback_pair(rng, mandatory, is_test, teach_color, contrast, palette, turn):
    """Evidence-preserving construction if free scene sampling is exhausted.

    The compulsory masks have a common height. Fixed long exposed boundaries
    provide the same inference conditions as sampled scenes, without exceptions
    to the semantic rule. This path is also explicitly exercised by self_test.
    """
    if mandatory:
        masks = [m.copy() for m in mandatory]
    else:
        m = np.ones((7, 5), dtype=bool)
        if is_test:
            m[1:3, 1:4] = False
            m[4:6, 2:4] = False
        else:
            m[1:6, 1:4] = False
        masks = [m]
    height = max(17, sum(m.shape[0] for m in masks)+len(masks)+1)
    col = max(m.shape[1] for m in masks)+2
    work = np.zeros((height, col), dtype=np.int8)
    row = 1
    for i, m in enumerate(masks):
        shift = 0 if i == 2 else 1
        start = col-m.shape[1]-shift
        work[row:row+m.shape[0], start:start+m.shape[1]][m] = 1
        row += m.shape[0]+1
    if contrast == 'decoy':
        h = masks[0].shape[0]
        boxes = [(0, 0, h+1, 7, 1), (h-2, 3, 2*h+3, 11, 2),
                 (2*h+1, 0, height, 7, 1)]
    else:
        h = masks[0].shape[0]
        boxes = [(0, 1, min(height-3, h+3), 9, 1), (h, 4, height, 11, 2)]
    clean = _paint((height, 11), 0, boxes)
    _require(_scene_supported(clean, 0, boxes), "fallback surface support")
    damaged = _corrupt_surface(rng, clean, _source_details(work, 0, 1), teach_color, contrast)
    _require(damaged is not None, "fallback evidence construction")
    bg, fg, rail, f1, f2 = palette
    g = np.full((height, col+12), bg, dtype=np.int8)
    g[:, :col][work == 1] = fg
    g[:, col] = rail
    g[:, col+1:] = np.array([bg, f1, f2], dtype=np.int8)[damaged]
    inp = np.rot90(g, turn).copy()
    return {'input': inp, 'output': solve(inp)}


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _unique_pair(pair, seen):
    # Twenty distinct global color maps: at most thirteen earlier inputs.
    # A reflection fixes at most two colors, while every input has five.
    for sign in (1, -1):
        for offset in range(10):
            lookup = np.array([(sign*c+offset) % 10 for c in range(10)], dtype=np.int8)
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _signature(candidate['input']) not in seen:
                return candidate
    raise AssertionError("uniqueness color orbit exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    vary_palette = bool(rng.integers(2))
    base_turn = _ri(rng, 0, 3)
    orientation = int(rng.choice([0, 1, 2], p=[.35, .35, .30]))
    contrast_masks = _contrasts(rng)
    plans = [(None, None, []) for _ in range(num_train)]
    order = list(map(int, rng.permutation(num_train)))
    plans[order[0]] = (1, 'open', [contrast_masks[0], contrast_masks[1]])
    plans[order[1]] = (2, 'decoy', [contrast_masks[0], contrast_masks[2], contrast_masks[0]])
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teach, contrast, mandatory = plans[i] if split == 'train' else (None, None, [])
            roles = tuple(map(int, rng.choice(10, 5, replace=False))) if vary_palette else palette
            if orientation == 2:
                turn = _ri(rng, 0, 3)
            elif orientation == 1 and split == 'test':
                turn = (base_turn+1) % 4
            else:
                turn = base_turn
            pair = _pair(rng, mandatory, split == 'test', teach, contrast, roles, turn)
            pair = _unique_pair(pair, seen)
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    inp, clean, rows, erasures = a['receiver'], a['clean'], a['rows'], a['erasures']
    centers = a['centers']
    wrong = (inp != clean) & (inp != a['bg']) & (clean != a['bg'])
    open_rows = set().union(*(o['open_rows'] for o in a['objects']))
    decoy_rows = set().union(*(o['decoy_rows'] for o in a['objects']))-rows
    filled = clean != a['bg']
    return dict(shape=a['canonical'].shape, turn=a['turn'], palette=(a['bg'], a['fg'], a['rail'], *a['fills']),
                heads=len(a['objects']), rectangles=len(a['boxes']),
                boxes=tuple(sorted(b[:4] for b in a['boxes'])),
                masks=tuple((o['mask'].shape, o['mask'].tobytes()) for o in a['objects']),
                active_holes=sum(len(o['active']) for o in a['objects']),
                mixed=any(o['active'] and o['decoy'] for o in a['objects']),
                multi=any(len(o['active']) >= 2 for o in a['objects']),
                shifts=tuple(o['shift'] for o in a['objects']),
                centers=len(centers), erasures=int(erasures.sum()),
                captured_fills={int(clean[p]) for p in centers},
                open_contrast=any(erasures[r].any() for r in open_rows-rows),
                decoy_contrast=any(erasures[r].any() for r in decoy_rows),
                multiple_candidates=any(erasures[r].sum() > 1 for r in rows),
                wrong_before=any(wrong[r, :c].any() for r, c in centers),
                gap_before=any((~filled[r, :c]).any() for r, c in centers),
                overlapping=any(max(abs(r-rr), abs(c-cc)) <= 2 and clean[r, c] == clean[rr, cc]
                                for r, c in centers for rr, cc in centers if (r, c) != (rr, cc)),
                adjacent=any(max(abs(r-rr), abs(c-cc)) == 1 and clean[r, c] == clean[rr, cc]
                             for r, c in centers for rr, cc in centers if (r, c) != (rr, cc)),
                wrong=int(wrong.sum()), speckles=int(((inp != clean) & ~filled).sum()),
                seam_errors=int(sum(wrong[r, c] and any(clean[rr, cc] not in (a['bg'], int(clean[r, c]))
                     for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                     if 0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1])
                     for r, c in np.argwhere(wrong))),
                clipped=any(r in (0, clean.shape[0]-1) or c in (0, clean.shape[1]-1) for r, c in centers))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions preserved")
            key = _signature(inp)
            _require(key not in seen, "duplicate inputs")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(a['expected'], out), "input-derived output mismatch")
            g = a['canonical']
            co = np.rot90(out, a['turn'])
            col = a['col']
            _require(np.array_equal(co[:, col], g[:, col]), "rail preservation")
            _require(np.count_nonzero(co[:, col+1:] == a['rail']) == len(a['centers']),
                     "only captured erasures become centers")
            for o in a['objects']:
                _require(o['shift'] >= 0, "source crosses rail")
                for r, c in o['cells']:
                    _require(co[r, c+o['shift']] == a['fg'], "rigid source translation")
                for hole in o['active']+o['decoy']:
                    for r, c in hole:
                        r0, c0, _, _ = o['bbox']
                        _require(co[r+r0, c+c0+o['shift']] == a['bg'], "source cavities stay hollow")
            if split == 'train':
                training.append(_features(a))
    _require(any(f['open_contrast'] for f in training), "missing open-vs-enclosed evidence")
    _require(any(f['decoy_contrast'] for f in training), "missing extreme-edge evidence")
    _require(any(f['multiple_candidates'] and f['wrong_before'] for f in training), "missing first-erasure evidence")
    _require(any(f['adjacent'] for f in training), "missing center precedence evidence")
    _require(any(len(f['captured_fills']) == 2 for f in training), "both complementary directions required")
    _require(any(any(s > 0 for s in f['shifts']) for f in training), "missing alignment evidence")


def self_test():
    # A hand-constructed semantic fixture, with known bounds and answer cells.
    # The expected target list is fixed, rather than obtained from _analyze.
    fixture = np.zeros((17, 19), dtype=np.int8)
    fixture[:, 8] = 5
    fixture[1:5, 1:5] = 1
    fixture[2:4, 2:4] = 0
    fixture[6:9, 2:5] = 1
    fixture[7, 2:4] = 0  # open C, row 7 must not inspect
    fixture[10:13, 1:4] = 1
    fixture[11, 2] = 0
    fixture[10, 4:6] = 1  # tail makes the closed row-11 cavity recessed
    surface = np.zeros((17, 10), dtype=np.int8)
    surface[0:11, 1:9] = 2
    surface[7:17, 3:10] = 7
    fixture[:, 9:] = surface
    for r, c in ((2, 4), (2, 7), (3, 4), (7, 6), (11, 5)):
        fixture[r, 9+c] = 0
    fixture[2, 11] = 7  # nearer wrong-fill defect, never a center
    fixture[7, 13] = 2  # seam bite, repair to 7
    fixture[15, 9] = 2  # exterior speckle, repair to background
    expected = np.zeros_like(fixture)
    expected[:, 8] = 5
    expected[1:5, 4:8] = 1
    expected[2:4, 5:7] = 0
    expected[6:9, 5:8] = 1
    expected[7, 5:7] = 0
    expected[10:13, 3:6] = 1
    expected[11, 4] = 0
    expected[10, 6:8] = 1
    expected[:, 9:] = surface
    expected[1:5, 12:15] = 7
    expected[2:4, 13] = 5
    _require(np.array_equal(solve(fixture), expected), "fixed first-hit/enclosure/layer fixture")
    # All channels can miss: alignment and repair still operate, without marks.
    miss = fixture.copy()
    miss[:, 9:] = surface
    miss[7, 15] = 0
    miss[11, 14] = 0
    expected_miss = expected.copy()
    expected_miss[:, 9:] = surface
    _require(np.array_equal(solve(miss), expected_miss), "uncaught defects must only be repaired")
    _require(np.array_equal(solve(expected_miss), expected_miss), "aligned undamaged scene is unchanged")
    # Clip a caught corner's ring at the receiver boundary, preserving the rail.
    edge = expected_miss.copy()
    edge_surface = surface.copy()
    edge_surface[:11, 0] = 2
    edge[:, 9:] = edge_surface
    edge[2, 9] = 0
    edge_answer = edge.copy()
    edge_answer[:, 9:] = edge_surface
    edge_answer[1:4, 9:11] = 7
    edge_answer[2, 9] = 5
    _require(np.array_equal(solve(edge), edge_answer), "rail-facing marker clipping")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(_analyze(p['input'])) for puzzle in puzzles
               for split in ('train', 'test') for p in puzzle[split]]
    _require(len({tuple(_signature(p['input']) for p in x['train']+x['test']) for x in puzzles}) == 42,
             "cross-seed puzzle diversity")
    _require(len({f['shape'] for f in records}) >= 35, "canvas variation")
    _require(len({f['boxes'] for f in records}) >= 100, "surface geometry variation")
    _require(len({m for f in records for m in f['masks']}) >= 80, "source geometry variation")
    _require({f['turn'] for f in records} == {0, 1, 2, 3}, "rotation variation")
    _require({f['rectangles'] for f in records} == {2, 3, 4}, "layer-count variation")
    _require(len({f['palette'] for f in records}) >= 40, "palette-role variation")
    _require(len({s for f in records for s in f['shifts']}) >= 7, "independent movement distances")
    _require(any(0 in f['shifts'] for f in records), "already aligned heads")
    _require(any(f['mixed'] for f in records) and any(f['multi'] for f in records), "combined source cavities")
    _require(any(f['gap_before'] for f in records), "beam must pass genuine exterior background")
    _require(any(f['seam_errors'] for f in records), "surface seam corruption")
    _require(any(f['clipped'] for f in records), "receiver-boundary clipping")
    _require(any(f['speckles'] for f in records), "exterior noise")
    # Spatial and palette equivariance checks exercise the input-only solver.
    pair = puzzles[0]['train'][0]
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(pair['input'], turn)), np.rot90(pair['output'], turn)),
                 "rotation equivariance")
    lookup = np.array([9, 2, 4, 0, 7, 6, 1, 5, 8, 3], dtype=np.int8)
    _require(np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']]), "palette equivariance")
    again = generate_puzzle(7)
    _require(all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
                 for p, q in zip(puzzles[7][s], again[s]) for k in ('input', 'output')), "determinism")
    # Family-specific ablations are inspected as sets before any ring painting.
    for puzzle in puzzles:
        analyses = [_analyze(p['input']) for p in puzzle['train']]
        _require(any(_select(a['erasures'], set(range(a['clean'].shape[0]))) != a['centers'] for a in analyses),
                 "ignoring A would not change training")
        _require(any({tuple(map(int, p)) for p in np.argwhere(a['erasures']) if int(p[0]) in a['rows']}
                     != a['centers'] for a in analyses), "mark-all-hit-defects shortcut")
        _require(any(_select((a['receiver'] != a['clean']) & (a['clean'] != a['bg']), a['rows']) != a['centers']
                     for a in analyses), "ignoring B's erasure distinction would not change training")
        _require(any(_select(a['erasures'], set().union(*(o['naive'] for o in a['objects']))) != a['centers']
                     for a in analyses), "one-column source shortcut")
        _require(any(_select(a['erasures'], a['rows'] | set().union(*(o['decoy_rows'] for o in a['objects'])))
                     != a['centers'] for a in analyses), "all-enclosed-cavities shortcut")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(12345, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Force the fallback across all procedural contrast heights and widths.
    fallback_rng = np.random.default_rng(1729)
    for h in (4, 5, 6):
        for w in (3, 6):
            closed = np.ones((h, w), dtype=bool)
            closed[1:-1, 1:-1] = False
            opened = closed.copy()
            opened[1:-1, 0] = False
            recessed = np.zeros((h, w+2), dtype=bool)
            recessed[:, :w] = closed
            recessed[0, w:] = True
            a = _fallback_pair(fallback_rng, [closed, opened], False, 1, 'open', (0,1,5,2,7), 0)
            b = _fallback_pair(fallback_rng, [closed, recessed, closed], False, 2, 'decoy', (0,1,5,2,7), 1)
            c = _fallback_pair(fallback_rng, [], True, None, None, (0,1,5,2,7), 2)
            validate_generated_puzzle({'train':[a,b], 'test':[c]})
    # Reject bad labels; input reconstruction is not supplied by those labels.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("bad output accepted")
    surface = np.ones((6, 8), dtype=np.int8)
    surface[:, 4:] = 2
    try:
        _stamp(surface, {(2, 3), (2, 4)}, (1, 2), 5)
    except AssertionError:
        pass
    else:
        raise AssertionError("incompatible rings accepted")
