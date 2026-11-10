"""Rotational cancellation meters. No parent code or data is used at runtime."""
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r, c) != (0, 0))
# Directions name the end bearing the tab, not the direction dots travel.
RIGHT, DOWN, LEFT, UP = range(4)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("need 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "grid must be integral")
    _check(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g.astype(np.int8, copy=True)


def _norm(cells):
    cells = tuple(cells)
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _rotate(cells, turns=1):
    for _ in range(turns % 4):
        cells = [(c, -r) for r, c in cells]
    return _norm(cells)


def _canonical(cells, reflections=False):
    shapes = [_rotate(cells, k) for k in range(4)]
    if reflections:
        reflected = [(r, -c) for r, c in cells]
        shapes.extend(_rotate(reflected, k) for k in range(4))
    return min(shapes)


def _extent(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _connected(cells, directions):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in directions:
            p = r+dr, c+dc
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _components(g, bg, directions=DIR8):
    seen = set()
    parts = []
    h, w = g.shape
    for r, c in np.argwhere(g != bg):
        r, c = int(r), int(c)
        if (r, c) in seen:
            continue
        color = int(g[r, c])
        stack, cells = [(r, c)], []
        seen.add((r, c))
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for dr, dc in directions:
                p = rr+dr, cc+dc
                if (0 <= p[0] < h and 0 <= p[1] < w and p not in seen
                        and int(g[p]) == color):
                    seen.add(p)
                    stack.append(p)
        parts.append(dict(color=color, cells=tuple(sorted(cells)), mask=_norm(cells),
                          box=(min(r for r, c in cells), max(r for r, c in cells),
                               min(c for r, c in cells), max(c for r, c in cells))))
    return parts


def _meter_cells(box, direction):
    r0, r1, c0, c1 = box
    border = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
              if r in (r0, r1) or c in (c0, c1)}
    tab = (((r0+r1)//2, c1+1) if direction == RIGHT else
           (r1+1, (c0+c1)//2) if direction == DOWN else
           ((r0+r1)//2, c0-1) if direction == LEFT else
           (r0-1, (c0+c1)//2))
    return border | {tab}


def _recognize_meter(part):
    r0, r1, c0, c1 = part['box']
    # Removing the protrusion from the component box recovers the rectangle.
    candidates = [(RIGHT, (r0, r1, c0, c1-1)),
                  (DOWN, (r0, r1-1, c0, c1)),
                  (LEFT, (r0, r1, c0+1, c1)),
                  (UP, (r0+1, r1, c0, c1))]
    matches = []
    for direction, box in candidates:
        a, b, x, y = box
        h, w = b-a+1, y-x+1
        short, long = (h, w) if direction % 2 == 0 else (w, h)
        if short in (5, 7) and 7 <= long <= 21 and long >= short+2:
            if _meter_cells(box, direction) == set(part['cells']):
                matches.append(dict(color=part['color'], cells=part['cells'], box=box,
                                    direction=direction, short=short, long=long,
                                    capacity=(long-3)//2))
    _check(len(matches) <= 1, "ambiguous tabbed rectangle")
    return matches[0] if matches else None


def _residuals(objects, key_mode="rotation", categories=None, one_per_class=False):
    groups = defaultdict(Counter)
    for obj in objects:
        color = obj['color']
        if categories is not None and color not in categories:
            continue
        mask = obj['mask']
        key = mask if key_mode == "identity" else _canonical(mask, key_mode == "reflection")
        groups[key][color] += 1
    residual = Counter()
    for population in groups.values():
        # n - all other n: for the family domain (<=2 colors), exactly pairwise
        # cancellation. This expression also defines our reflection ablation.
        total = sum(population.values())
        for color, count in population.items():
            excess = max(0, 2*count-total)
            residual[color] += int(excess > 0) if one_per_class else excess
    return residual, groups


def _inspect(g):
    colors, amounts = np.unique(g, return_counts=True)
    bg = int(colors[int(np.argmax(amounts))])
    _check(int(amounts.max())*2 > g.size, "background must occupy over half the input")
    parts = _components(g, bg)
    meters, objects = [], []
    for part in parts:
        meter = _recognize_meter(part)
        if meter is None:
            objects.append(part)
        else:
            meters.append(meter)
    _check(1 <= len(meters) <= 4 and bool(objects), "need meters and ordinary objects")
    categories = {m['color'] for m in meters}
    _check(len(categories) == len(meters), "meter colors must be distinct")
    labels = np.full(g.shape, -1, dtype=int)
    for i, part in enumerate(parts):
        for r, c in part['cells']:
            labels[r, c] = i
    for r, c in np.argwhere(labels >= 0):
        for dr, dc in DIR8:
            rr, cc = int(r)+dr, int(c)+dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                _check(labels[rr, cc] in (-1, labels[r, c]), "separate components touch")
    for meter in meters:
        r0, r1, c0, c1 = meter['box']
        _check(bool(np.all(g[r0+1:r1, c0+1:c1] == bg)), "meter interior is not empty")
        # This also excludes ordinary objects nested inside an otherwise valid frame.
    for obj in objects:
        _check(1 <= len(obj['cells']) <= 16 and max(_extent(obj['mask'])) <= 6,
               "ordinary object outside shape bounds")
    residual, groups = _residuals(objects)
    _check(all(len(p) <= 2 for p in groups.values()), "shape class contains >2 colors")
    _check(all(residual[m['color']] <= m['capacity'] for m in meters), "tally exceeds capacity")
    return dict(background=bg, meters=meters, objects=objects, residual=residual, groups=groups)


def _dot_cells(meter, count):
    r0, r1, c0, c1 = meter['box']
    d = meter['direction']
    for i in range(1, count+1):
        r = r1-2*i if d == DOWN else r0+2*i if d == UP else (r0+r1)//2
        c = c1-2*i if d == RIGHT else c0+2*i if d == LEFT else (c0+c1)//2
        yield r, c


def _render(shape, bg, meters, residual):
    out = np.full(shape, bg, dtype=np.int8)
    for meter in meters:
        for p in meter['cells']:
            out[p] = meter['color']
        for p in _dot_cells(meter, residual[meter['color']]):
            out[p] = meter['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only the input's objects, meter colors and visible tab directions."""
    g = _grid(input_grid)
    detail = _inspect(g)
    return _render(g.shape, detail['background'], detail['meters'], detail['residual'])


def _sample_mask(rng, limit):
    kind = int(rng.integers(7))
    if kind == 0:
        cells = {(0, c) for c in range(int(rng.integers(1, limit+1)))}
    elif kind in (1, 2):
        h, w = (int(rng.integers(2, limit+1)) for _ in range(2))
        cells = {(r, c) for r in range(h) for c in range(w)
                 if kind == 1 or r in (0, h-1) or c in (0, w-1)}
    elif kind == 3:
        h, w = (int(rng.integers(2, limit+1)) for _ in range(2))
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif kind == 4:
        h, w = (int(rng.integers(2, limit+1)) for _ in range(2))
        rr, cc = int(rng.integers(h)), int(rng.integers(w))
        cells = {(r, cc) for r in range(h)} | {(rr, c) for c in range(w)}
    else:
        target = int(rng.integers(3, min(16, limit*limit)+1))
        cells = {(0, 0)}
        directions = DIR4 if kind == 5 else DIR8
        while len(cells) < target:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in directions} - cells)
            valid = [p for p in frontier if max(r for r, c in cells | {p})-min(r for r, c in cells | {p}) < limit
                     and max(c for r, c in cells | {p})-min(c for r, c in cells | {p}) < limit]
            if not valid:
                break
            cells.add(valid[int(rng.integers(len(valid)))])
    return _rotate(cells, int(rng.integers(4)))


def _new_mask(rng, forbidden, limit, chiral=False, diagonal=False):
    for _ in range(300):
        mask = _sample_mask(rng, limit)
        if len(mask) > 16 or _canonical(mask) in forbidden:
            continue
        if chiral and (_canonical(mask) == _canonical([(r, -c) for r, c in mask])
                       or len({_rotate(mask, k) for k in range(4)}) != 4):
            continue
        if diagonal and _connected(mask, DIR4):
            continue
        return mask
    # Small exhaustive vocabulary is a deterministic fallback, not stored pairs.
    for bits in range(1, 512):
        mask = _norm([(i//3, i % 3) for i in range(9) if bits & (1 << i)])
        if (_canonical(mask) not in forbidden and _connected(mask, DIR8)
                and (not diagonal or not _connected(mask, DIR4))
                and (not chiral or (_canonical(mask) != _canonical([(r, -c) for r, c in mask])
                                    and len({_rotate(mask, k) for k in range(4)}) == 4))):
            return mask
    raise RuntimeError("shape vocabulary exhausted")


def _add_group(objects, mask, populations, rng, turns=None):
    j = 0
    for color, count in populations:
        for _ in range(count):
            turn = int(rng.integers(4)) if turns is None else turns[j]
            objects.append(dict(color=int(color), mask=_rotate(mask, turn)))
            j += 1


def _candidate_origins(blocked, cells):
    h, w = blocked.shape
    sh, sw = _extent(cells)
    if sh > h or sw > w:
        return []
    possible = np.ones((h-sh+1, w-sw+1), dtype=bool)
    for dr, dc in cells:
        possible &= ~blocked[dr:dr+h-sh+1, dc:dc+w-sw+1]
    return np.argwhere(possible)


def _mark_halo(blocked, cells):
    h, w = blocked.shape
    for r, c in cells:
        blocked[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = True


def _local_meter(short, long, direction, color):
    h, w = (short, long) if direction % 2 == 0 else (long, short)
    r0, c0 = int(direction == UP), int(direction == LEFT)
    box = (r0, r0+h-1, c0, c0+w-1)
    cells = _meter_cells(box, direction)
    footprint = {(r, c) for r in range(r0, r0+h) for c in range(c0, c0+w)} | cells
    return dict(color=color, cells=tuple(sorted(cells)), box=box, direction=direction,
                short=short, long=long, capacity=(long-3)//2, footprint=tuple(sorted(footprint)))


def _translate_meter(local, row, col):
    r0, r1, c0, c1 = local['box']
    return {k: v for k, v in local.items() if k not in ('footprint', 'box', 'cells')} | {
        'box': (r0+row, r1+row, c0+col, c1+col),
        'cells': tuple((r+row, c+col) for r, c in local['cells'])}


def _random_pack(rng, objects, locals_, bg, h, w):
    g = np.full((h, w), bg, dtype=np.int8)
    blocked = np.zeros((h, w), dtype=bool)
    meters = []
    for local in sorted(locals_, key=lambda m: -len(m['footprint'])):
        origins = _candidate_origins(blocked, local['footprint'])
        if not len(origins):
            return None
        r, c = map(int, origins[int(rng.integers(len(origins)))])
        meter = _translate_meter(local, r, c)
        meters.append(meter)
        for p in meter['cells']:
            g[p] = meter['color']
        _mark_halo(blocked, [(rr+r, cc+c) for rr, cc in local['footprint']])
    order = sorted(range(len(objects)), key=lambda i: (-len(objects[i]['mask']), float(rng.random())))
    for i in order:
        obj = objects[i]
        origins = _candidate_origins(blocked, obj['mask'])
        if not len(origins):
            return None
        r, c = map(int, origins[int(rng.integers(len(origins)))])
        cells = [(rr+r, cc+c) for rr, cc in obj['mask']]
        for p in cells:
            g[p] = obj['color']
        _mark_halo(blocked, cells)
    return g, meters


def _shelf_pack(objects, categories, bg, residual, directions=None):
    """Guaranteed roomy construction for the generated mask/count bounds.

    Teaching masks: <=4, <=15 tokens, max residual 5 -> field width >=15.
    Random masks: <=6 for <=6 tokens, <=4 for <=10, <=3 otherwise (<=17).
    Random residual <=7 -> worst field width 11. These boxes fit on shelves.
    """
    g = np.full((30, 30), bg, dtype=np.int8)
    meters, largest = [], 0
    for i, color in enumerate(categories):
        long = max(7, 2*residual[color]+3)
        if directions is not None and i == 0:
            long += 2  # Nonfull anchor teaches its tab direction.
        d = directions[i] if directions is not None else (RIGHT if i % 2 == 0 else LEFT)
        # A packing fallback may change an optional vertical direction; the
        # mandatory teaching directions are horizontal and are retained.
        d = RIGHT if d == DOWN else LEFT if d == UP else d
        local = _local_meter(5, long, d, color)
        meter = _translate_meter(local, 6*i, 0)
        meters.append(meter)
        for p in meter['cells']:
            g[p] = color
        largest = max(largest, long)
    field_start = largest+2
    row, col, row_height = 0, field_start, 0
    for obj in sorted(objects, key=lambda o: (-_extent(o['mask'])[0], -_extent(o['mask'])[1])):
        sh, sw = _extent(obj['mask'])
        if col+sw > 30:
            row += row_height+1
            col, row_height = field_start, 0
        _check(row+sh <= 30 and col+sw <= 30, "shelf guarantee failed")
        for r, c in obj['mask']:
            g[row+r, col+c] = obj['color']
        col += sw+1
        row_height = max(row_height, sh)
    return g, meters


def _scene(rng, objects, categories, bg, directions=None):
    residual, _ = _residuals(objects)
    _check(max((residual[c] for c in categories), default=0) <= 7, "generated residual bound")
    area = sum(len(o['mask']) for o in objects)
    for attempt in range(40):
        locals_ = []
        for i, color in enumerate(categories):
            short = 7 if attempt < 10 and rng.random() < .2 else 5
            long = max(7, short+2, 2*residual[color]+3)
            if directions is not None and i == 0:
                long += 2
            long = min(21, long+int(rng.integers(0, 5 if attempt < 10 else 2)))
            direction = int(rng.integers(4)) if directions is None else directions[i]
            locals_.append(_local_meter(short, long, direction, color))
        footprint = sum(len(m['footprint']) for m in locals_)
        minimum_area = max(160, 2*(area+sum(len(m['cells']) for m in locals_))+1,
                           int(1.35*footprint+4*area))
        lo = 12 if len(objects) <= 6 and len(categories) <= 2 else 20
        options = [(h, w) for h in range(lo, 31) for w in range(lo, 31)
                   if h*w >= minimum_area]
        if not options:
            continue
        if attempt >= 12:
            h, w = 30, 30
        else:
            weights = np.array([np.exp(-(h*w-minimum_area)/220) for h, w in options])
            h, w = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        packed = _random_pack(rng, objects, locals_, bg, h, w)
        if packed is not None:
            inp, meters = packed
            if int(np.count_nonzero(inp == bg))*2 > inp.size:
                return dict(input=inp, output=_render(inp.shape, bg, meters, residual))
    inp, meters = _shelf_pack(objects, categories, bg, residual, directions)
    return dict(input=inp, output=_render(inp.shape, bg, meters, residual))


def _sample_teaching_objects(rng, colors):
    anchor, opponent, other, novel = colors
    p = _new_mask(rng, set(), 4, chiral=True)
    forbidden = {_canonical(p), _canonical([(r, -c) for r, c in p])}
    q = _new_mask(rng, forbidden, 3, diagonal=True)
    forbidden.add(_canonical(q))
    r = _new_mask(rng, forbidden, 3)
    forbidden.add(_canonical(r))
    s = _new_mask(rng, forbidden, 3)
    m, mono = int(rng.integers(3, 5)), int(rng.integers(2, 4))
    balanced = int(rng.integers(1, 3))
    phase = int(rng.integers(4))
    a, b = [], []
    # Same anchor masks, population, and literal color in both demonstrations.
    _add_group(a, p, [(anchor, m), (opponent, m-2)], rng,
               [phase]*m+[(phase+1) % 4]*(m-2))
    _add_group(b, p, [(anchor, m), (opponent, m-1)], rng,
               [phase]*m+[(phase+1) % 4]*(m-1))
    mirror = _norm([(rr, -cc) for rr, cc in p])
    _add_group(a, mirror, [(opponent, 1)], rng)
    _add_group(b, mirror, [(other, 1)], rng)
    qturn = int(rng.integers(4))
    _add_group(a, q, [(anchor, 1)], rng, [qturn])
    _add_group(b, q, [(anchor, 1)], rng, [qturn])
    _add_group(a, r, [(opponent, balanced), (novel, balanced)], rng)
    _add_group(b, r, [(other, 1), (opponent, 1)], rng)
    _add_group(a, s, [(opponent, mono)], rng)
    _add_group(b, s, [(other, mono+1)], rng)
    return a, b


def _four_tokens(objects):
    """Split masks for a connectivity counterfactual, before layout exists."""
    result = []
    for obj in objects:
        todo = set(obj['mask'])
        while todo:
            start = min(todo)
            todo.remove(start)
            stack, cells = [start], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIR4:
                    p = r+dr, c+dc
                    if p in todo:
                        todo.remove(p)
                        stack.append(p)
            result.append(dict(color=obj['color'], mask=_norm(cells)))
    return result


def _teaching_objects(rng, colors):
    # Pixel-derived validation will repeat this check after placement. Merely
    # requiring a diagonal attachment is insufficient when counts can cancel.
    for _ in range(80):
        a, b = _sample_teaching_objects(rng, colors)
        informative = False
        for objects, categories in ((a, colors[:2]), (b, [colors[0], colors[2]])):
            correct = _residuals(objects)[0]
            four = _residuals(_four_tokens(objects))[0]
            informative |= any(correct[c] != four[c] for c in categories)
        if informative:
            return a, b
    raise RuntimeError("connectivity teaching witness sampling exhausted")


def _random_objects(rng, categories, bg):
    palette = [c for c in range(10) if c != bg]
    for _ in range(100):
        profiles = []
        nclass = int(rng.integers(2, 6))
        for i in range(nclass):
            # Category and unframed tokens share exactly the same mask sampler.
            first = int(rng.choice(categories if rng.random() < .8 else palette))
            kind = int(rng.choice(3, p=[.30, .27, .43]))
            a = int(rng.integers(1, 6))
            if kind == 0:
                populations = [(first, a)]
            else:
                second = int(rng.choice([c for c in palette if c != first]))
                b = a if kind == 1 else int(rng.integers(1, 6))
                populations = [(first, a), (second, b)]
            profiles.append(populations)
        n = sum(n for group in profiles for c, n in group)
        if not 4 <= n <= 17:
            continue
        limit = 6 if n <= 6 else 4 if n <= 10 else 3
        objects, forbidden = [], set()
        for populations in profiles:
            mask = _new_mask(rng, forbidden, limit)
            forbidden.add(_canonical(mask))
            _add_group(objects, mask, populations, rng)
        residual, _ = _residuals(objects)
        if all(residual[c] <= 7 for c in categories):
            return objects
    # A random population fallback, still using new masks and independent colors.
    objects, forbidden = [], set()
    for color in categories:
        mask = _new_mask(rng, forbidden, 3)
        forbidden.add(_canonical(mask))
        _add_group(objects, mask, [(color, 1)], rng)
    return objects


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = [int(c) for c in rng.permutation(10)[:4]]
    anchor, opponent, other, novel = colors
    witness_a, witness_b = _teaching_objects(rng, colors)
    backgrounds = [c for c in range(10) if c not in colors]
    bg_a, bg_b = [int(c) for c in rng.choice(backgrounds, 2, replace=False)]
    teach_a = _scene(rng, witness_a, [anchor, opponent], bg_a, [RIGHT, LEFT])
    teach_b = _scene(rng, witness_b, [anchor, other], bg_b, [LEFT, RIGHT])
    puzzle = {'train': [], 'test': []}
    teaching_indices = [int(i) for i in rng.choice(nt, 2, replace=False)]
    prescribed = dict(zip(teaching_indices, (teach_a, teach_b)))
    reserved = {_signature(teach_a['input']), _signature(teach_b['input'])}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index in prescribed:
                pair = prescribed[index]
            else:
                for _ in range(30):
                    bg = int(rng.choice([c for c in range(10) if split == 'train' or c != novel]))
                    nframe = int(rng.integers(1, 5))
                    pool = [c for c in range(10) if c != bg and (split == 'test' or c != novel)]
                    categories = [int(c) for c in rng.choice(pool, nframe, replace=False)]
                    if split == 'test' and novel not in categories:
                        categories[0] = novel
                    objects = _random_objects(rng, categories, bg)
                    pair = _scene(rng, objects, categories, bg)
                    if _signature(pair['input']) not in seen | reserved:
                        break
                else:
                    raise RuntimeError("distinct input sampling exhausted")
            key = _signature(pair['input'])
            _check(key not in seen, "duplicate prescribed input")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    g = np.asarray(grid, dtype=np.uint8)
    return g.shape, g.tobytes()


def _vector(detail, residual):
    return tuple(residual[m['color']] for m in detail['meters'])


def _features(g):
    d = _inspect(g)
    objects, categories = d['objects'], {m['color'] for m in d['meters']}
    correct = _vector(d, d['residual'])
    raw = Counter(o['color'] for o in objects)
    modes = {'raw': raw,
             'identity': _residuals(objects, 'identity')[0],
             'reflection': _residuals(objects, 'reflection')[0],
             'frame_first': _residuals(objects, categories=categories)[0],
             'class_count': _residuals(objects, one_per_class=True)[0]}
    four_objects = [p for p in _components(g, d['background'], DIR4)
                    if _recognize_meter(p) is None]
    modes['four_connectivity'] = _residuals(four_objects)[0]
    # Reject counting only shape classes of size exactly two / cancelling just
    # one pair, when multiplicities make those differ from complete cancellation.
    once = Counter()
    for population in d['groups'].values():
        for c, n in population.items():
            once[c] += n-int(len(population) == 2)
    modes['cancel_once'] = once
    return d | {'raw': raw,
                'ablation_differences': {name for name, res in modes.items()
                                        if _vector(d, res) != correct},
                'shape': g.shape,
                'mask_set': {_canonical(o['mask']) for o in objects},
                'directions': {m['direction'] for m in d['meters']},
                'populations': {tuple(sorted(p.values())) for p in d['groups'].values()}}


def validate_generated_puzzle(puzzle) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(inp.shape == out.shape, "canvas changed")
            key = _signature(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _features(inp)
            expected = _render(inp.shape, d['background'], d['meters'], d['residual'])
            _check(np.array_equal(out, expected), "wrong residual, dots, erasure, or preserved meter")
            _check(not np.array_equal(inp, out), "identity example")
            details[split].append(d)
    train = details['train']
    differences = set().union(*(d['ablation_differences'] for d in train))
    required = {'raw', 'identity', 'reflection', 'frame_first', 'class_count',
                'four_connectivity', 'cancel_once'}
    _check(required <= differences, f"training permits shortcuts: {required-differences}")
    observations = []
    for d in train:
        for m in d['meters']:
            color = m['color']
            masks = tuple(sorted(o['mask'] for o in d['objects'] if o['color'] == color))
            observations.append((color, masks, d['residual'][color]))
    _check(any(c == cc and masks == other and n != nn
               for c, masks, n in observations for cc, other, nn in observations),
           "same raw color masks / different residual teaching witness missing")
    informative = {m['direction'] for d in train for m in d['meters']
                   if 0 < d['residual'][m['color']] < m['capacity']}
    _check({RIGHT, LEFT} <= informative, "opposing tab directions not demonstrated")
    _check(any(len(set(d['residual'][m['color']] for m in d['meters'])) > 1 for d in train),
           "color routing lacks distinct residuals")
    registered = {m['color'] for d in train for m in d['meters']}
    unframed = {o['color'] for d in train for o in d['objects']
                if o['color'] not in {m['color'] for m in d['meters']}}
    _check(bool(registered & unframed), "local category role switch missing")
    for d in details['test']:
        _check(bool({m['color'] for m in d['meters']}-registered), "new test category missing")


def self_test() -> dict:
    """Bounded generator-owned checks; returns descriptive 42-seed measurements."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints = {tuple(_signature(pair['input']) for split in ('train', 'test')
                          for pair in p[split]) for p in puzzles}
    _check(len(fingerprints) == 42, "seed diversity")
    _check(tuple(_signature(p['input']) for s in ('train', 'test') for p in puzzles[0][s]) ==
           tuple(_signature(p['input']) for s in ('train', 'test') for p in generate_puzzle(0)[s]),
           "seed determinism")
    features = [_features(np.asarray(pair['input'])) for p in puzzles
                for split in ('train', 'test') for pair in p[split]]
    shapes = {d['shape'] for d in features}
    masks = set().union(*(d['mask_set'] for d in features))
    populations = set().union(*(d['populations'] for d in features))
    _check(len(shapes) >= 35 and len(masks) >= 100, "structural variation too narrow")
    _check({d['background'] for d in features} == set(range(10)), "background variation")
    _check({len(d['meters']) for d in features} == {1, 2, 3, 4}, "meter count variation")
    _check(set().union(*(d['directions'] for d in features)) == set(range(4)), "tab directions")
    _check(any(len(d['directions']) > 1 for d in features), "mixed meter directions")
    _check(any(all(d['residual'][m['color']] == 0 for m in d['meters']) for d in features),
           "zero residual regime missing")
    _check(any(len(p) == 2 and p[0] == p[1] for p in populations), "ties missing")
    _check(any(len(p) == 2 and p[0] != p[1] for p in populations), "imbalance missing")
    _check({(1,), (2,), (3,)} <= populations, "one-color multiplicities missing")
    _check(max(len(o['cells']) for d in features for o in d['objects']) >= 12, "larger masks missing")
    _check(any(min(_extent(o['mask'])) == 1 for d in features for o in d['objects']), "lines missing")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(4200+a*10+b, a, b)
            _check(len(p['train']) == a and len(p['test']) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exercise the geometric fallback directly on all witness profiles.
    for seed in range(8):
        rng = np.random.default_rng(9000+seed)
        a, b = _teaching_objects(rng, [1, 2, 3, 4])
        for objects, cats in ((a, [1, 2]), (b, [1, 3])):
            res, _ = _residuals(objects)
            inp, meters = _shelf_pack(objects, cats, 0, res, [RIGHT, LEFT])
            _check(np.array_equal(solve(inp), _render(inp.shape, 0, meters, res)), "shelf fallback")
    # Exercise the extremal packing regimes, not just the teaching profiles.
    for n, size, cap in ((6, 6, 6), (10, 4, 7), (17, 3, 7)):
        mask = tuple(sorted({(i, 0) for i in range(size)} |
                            {(size-1, i) for i in range(size)}))
        other_mask = tuple((r, c) for r in range(size) for c in range(size))
        objects = [dict(color=1 if i < cap else 5, mask=mask if i < cap else other_mask)
                   for i in range(n)]
        res, _ = _residuals(objects)
        inp, meters = _shelf_pack(objects, [1, 2, 3, 4], 0, res)
        _check(np.array_equal(solve(inp), _render(inp.shape, 0, meters, res)), "extremal shelf fallback")
    # The specification allows full capacity nine even though the sampler caps
    # ordinary random residuals at seven. Test an input outside that sampler cap.
    boundary = np.zeros((14, 26), dtype=np.int8)
    for p in _meter_cells((0, 6, 0, 20), RIGHT):
        boundary[p] = 5
    boundary[10, :25:3] = 5  # Nine separate, identical tokens.
    expected = np.zeros_like(boundary)
    for p in _meter_cells((0, 6, 0, 20), RIGHT):
        expected[p] = 5
    expected[3, 2:19:2] = 5
    for turn in range(4):
        rotated = np.rot90(boundary, turn)
        _check(np.array_equal(solve(rotated), np.rot90(expected, turn)), "capacity-nine rotated input")
    inp = np.asarray(puzzles[0]['test'][0]['input']).copy()
    before = inp.copy()
    _check(np.array_equal(solve(inp), puzzles[0]['test'][0]['output']), "solver agreement")
    _check(np.array_equal(inp, before), "solve mutated input")
    for corruption in ('unerased', 'tab', 'dot'):
        bad = deepcopy(puzzles[0])
        pair = bad['test'][0]
        d = _inspect(pair['input'])
        if corruption == 'unerased':
            pair['output'] = pair['input'].copy()
        elif corruption == 'tab':
            m = d['meters'][0]
            tab = next(iter(set(m['cells'])-{
                (r, c) for r in range(m['box'][0], m['box'][1]+1)
                for c in range(m['box'][2], m['box'][3]+1)}))
            pair['output'][tab] = d['background']
        else:
            m = d['meters'][0]
            p = next(_dot_cells(m, 1))
            pair['output'][p] = d['background'] if pair['output'][p] == m['color'] else m['color']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError("corrupted output accepted")
    return {'seeds': 42, 'pairs': len(features), 'canvas_shapes': len(shapes),
            'rotational_masks': len(masks), 'population_profiles': len(populations),
            'backgrounds': sorted({d['background'] for d in features}),
            'meter_counts': sorted({len(d['meters']) for d in features}),
            'object_counts': sorted({len(d['objects']) for d in features}),
            'meter_lengths': sorted({m['long'] for d in features for m in d['meters']}),
            'residual_counts': sorted({d['residual'][m['color']] for d in features for m in d['meters']})}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
