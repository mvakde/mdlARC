"""Repair periodic heights in lanes defined by rotational shape identity.

No files, parent generators, latent state, or stored examples enter generation or
solving. Diagnostics are streamed by self_test; it writes no artifacts.
"""
from collections import Counter
from hashlib import sha256

import numpy as np


_DIR8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0))
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a


def _crop(mask):
    ys, xs = np.nonzero(mask)
    _require(len(ys) > 0, "empty object")
    return np.asarray(mask[ys.min():ys.max()+1, xs.min():xs.max()+1], dtype=bool).copy()


def _key(mask):
    a = _crop(mask)
    return a.shape, a.astype(np.uint8).tobytes()


def _canonical(mask):
    a = _crop(mask)
    # Rotating an already tight bounding box cannot introduce an empty edge.
    return min((r.shape, r.astype(np.uint8).tobytes()) for r in (np.rot90(a, k) for k in range(4)))


def _mirror(mask):
    return _canonical(np.fliplr(mask))


def _four_connected(mask):
    points = set(map(tuple, np.argwhere(mask)))
    seen = {min(points)}
    stack = list(seen)
    while stack:
        y, x = stack.pop()
        for dy, dx in _DIR4:
            p = y+dy, x+dx
            if p in points and p not in seen:
                seen.add(p)
                stack.append(p)
    return seen == points


def _halo(cells):
    return {(y+dy, x+dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def _cells(obj, top=None):
    top = obj['top'] if top is None else int(top)
    if 'offsets' not in obj:
        obj['offsets'] = tuple((int(y), int(x)) for y, x in np.argwhere(obj['mask']))
    return {(top+y, obj['left']+x) for y, x in obj['offsets']}


def _background(a):
    values, counts = np.unique(a, return_counts=True)
    i = int(np.argmax(counts))
    _require(int(counts[i])*2 > a.size, "background must occupy more than half the canvas")
    return int(values[i])


def _segment(a, background):
    seen = np.zeros(a.shape, dtype=bool)
    objects, canonical_keys = [], {}
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            if seen[y, x] or int(a[y, x]) == background:
                continue
            color = int(a[y, x])
            stack, points = [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy, dx in _DIR8:
                    ny, nx = cy+dy, cx+dx
                    if (0 <= ny < h and 0 <= nx < w and not seen[ny, nx]
                            and int(a[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            top, left = min(p[0] for p in points), min(p[1] for p in points)
            mask = np.zeros((max(p[0] for p in points)-top+1,
                             max(p[1] for p in points)-left+1), dtype=bool)
            for cy, cx in points:
                mask[cy-top, cx-left] = True
            raw = mask.shape, mask.astype(np.uint8).tobytes()
            if raw not in canonical_keys:
                canonical_keys[raw] = _canonical(mask)
            objects.append({'mask': mask, 'top': top, 'left': left,
                            'color': color, 'canonical': canonical_keys[raw],
                            'offsets': tuple((cy-top, cx-left) for cy, cx in points)})
    return objects


def _groups(objects, mode='rotation'):
    groups = {}
    for obj in objects:
        if mode == 'rotation':
            k = obj['canonical']
        elif mode == 'reflection':
            k = min(obj['canonical'], _mirror(obj['mask']))
        elif mode == 'color':
            k = obj['color']
        else:
            k = _key(obj['mask'])
        groups.setdefault(k, []).append(obj)
    return groups


def _recover_rhythm(rows):
    """Enumerate all supported repairs; no generator metadata is consulted."""
    rows = tuple(int(y) for y in rows)
    n, candidates = len(rows), []
    for period in range(1, min(3, n//3)+1):
        word = []
        for phase in range(period):
            observed = rows[phase::period]
            counts = Counter(observed)
            majority, frequency = counts.most_common(1)[0]
            if frequency*2 <= len(observed):
                break
            word.append(majority)
        if len(word) != period or len(set(word)) != period:
            continue
        repaired = tuple(word[i % period] for i in range(n))
        errors = sum(x != y for x, y in zip(rows, repaired))
        if errors <= 2:
            candidates.append((period, repaired, tuple(word), errors))
    _require(bool(candidates), "no supported rhythm")
    _require(len({c[1] for c in candidates}) == 1, "ambiguous rhythm")
    best = min(candidates, key=lambda c: c[0])
    _require(best[0] in (2, 3), "constant rhythms are outside this family")
    return best


def _destinations(objects):
    result, details = {}, []
    for group in _groups(objects).values():
        if len(group) == 1:
            result[id(group[0])] = group[0]['top']
            continue
        _require(6 <= len(group) <= 10, "class size must be one or 6..10")
        ordered = sorted(group, key=lambda o: o['left'])
        _require(len({o['left'] for o in ordered}) == len(ordered), "tied leftmost columns")
        p, repaired, word, errors = _recover_rhythm([o['top'] for o in ordered])
        for obj, row in zip(ordered, repaired):
            result[id(obj)] = row
        details.append({'objects': ordered, 'period': p, 'word': word,
                        'errors': errors, 'rows': repaired})
    _require(bool(details), "at least one repeated class is required")
    return result, details


def _paint(objects, tops, shape, background):
    out = np.full(shape, background, dtype=np.int8)
    blocked = set()
    for obj, top in zip(objects, tops):
        points = _cells(obj, top)
        _require(all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in points), "object clips boundary")
        _require(not points & blocked, "objects overlap or touch")
        for y, x in points:
            out[y, x] = obj['color']
        blocked.update(_halo(points))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every destination from the given input alone; never mutate it."""
    a = _grid(input_grid)
    bg = _background(a)
    objects = _segment(a, bg)
    destinations, _ = _destinations(objects)
    return _paint(objects, [destinations[id(o)] for o in objects], a.shape, bg)


def _random_mask(rng):
    kind = int(rng.integers(7))
    h, w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if kind == 0:
        mask = np.ones((1, int(rng.integers(1, 5))), dtype=bool)
    elif kind == 1:
        mask = np.ones((h, w), dtype=bool)
    elif kind == 2:
        mask = np.zeros((h, w), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
    elif kind == 3:
        mask = np.ones((h, w), dtype=bool)
        if h > 2 and w > 2:
            mask[1:-1, 1:-1] = False
    elif kind == 4:
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), :] = True
        mask[:, int(rng.integers(w))] = True
    else:
        directions = _DIR4 if kind == 5 else _DIR8
        target = int(rng.integers(3, h*w+1))
        points = {(int(rng.integers(h)), int(rng.integers(w)))}
        for _ in range(target-1):
            frontier = sorted({(y+dy, x+dx) for y, x in points for dy, dx in directions
                               if 0 <= y+dy < h and 0 <= x+dx < w} - points)
            points.add(frontier[int(rng.integers(len(frontier)))])
        mask = np.zeros((h, w), dtype=bool)
        for y, x in points:
            mask[y, x] = True
    return _crop(np.rot90(mask, int(rng.integers(4))))


def _sample_mask(rng, forbidden, chiral=False, asymmetric=False, rectangular=False, diagonal=False):
    for _ in range(160):
        mask = _random_mask(rng)
        k = _canonical(mask)
        if (k not in forbidden and _mirror(mask) not in forbidden
                and (not chiral or k != _mirror(mask))
                and (not rectangular or mask.shape[0] != mask.shape[1])
                and (not diagonal or not _four_connected(mask))
                and (not asymmetric or len({_key(np.rot90(mask, i)) for i in range(4)}) == 4)):
            return mask
    raise ValueError("shape sampling budget exhausted")


def _interleaved(details):
    if len(details) < 2:
        return False
    ranges = [(min(d['word']), max(d['word'])) for d in details]
    return any(max(a[0], b[0]) < min(a[1], b[1])
               for i, a in enumerate(ranges) for b in ranges[i+1:])


def _construct_pair(rng, plan, stable_background):
    periods, sizes = plan['periods'], plan['sizes']
    background = int(rng.integers(10)) if stable_background is None else stable_background
    palette = [int(x) for x in rng.permutation(10) if int(x) != background]
    objects, class_objects, error_objects, bases = [], [], [], []
    forbidden = set()
    for j, (p, n) in enumerate(zip(periods, sizes)):
        base = _sample_mask(rng, forbidden, chiral=plan.get('mirror', False) and j == 0,
                            asymmetric=j == 0, rectangular=plan.get('top_witness', False) and j == 0,
                            diagonal=plan.get('diagonal', False) and j == 0)
        bases.append(base)
        forbidden.update((_canonical(base), _mirror(base)))
        count_errors = plan.get('errors', [int(rng.integers(1, 3)) for _ in periods])[j]
        error_phases = [int(i) for i in rng.choice(p, size=count_errors, replace=False)]
        error_indices = [int(rng.choice(np.arange(phase, n, p))) for phase in error_phases]
        turns = [int(rng.choice([0, 2, 3])) for _ in range(n)]
        # The first erroneous member has a unique displayed orientation.
        turns[error_indices[0]] = 1
        if plan.get('top_witness') and j == 0:
            peer = next(i for i in range(n) if i != error_indices[0] and i % p == error_indices[0] % p)
            turns[peer] = 0
        group = []
        for i, turn in enumerate(turns):
            obj = {'mask': _crop(np.rot90(base, turn)), 'color': int(rng.choice(palette)),
                   'class_id': j, 'phase': i % p, 'index': i}
            objects.append(obj)
            group.append(obj)
        class_objects.append(group)
        error_objects.append([group[i] for i in error_indices])
    # One consistent top value is attached to each ordinal phase. These temporary
    # intervals are merely a packing aid; the public solver never sees them.
    labels = [(j, phase) for j, p in enumerate(periods) for phase in range(p)]
    for _ in range(64):
        rng.shuffle(labels)
        indices = [[i for i, (j, _) in enumerate(labels) if j == k] for k in range(len(periods))]
        if len(periods) == 1 or max(min(t) for t in indices) < min(max(t) for t in indices):
            break
    else:
        raise ValueError("interleaving sampling exhausted")
    level_heights = [max(o['mask'].shape[0] for o in class_objects[j] if o['phase'] == phase)
                     for j, phase in labels]
    min_height = sum(level_heights)+len(labels)-1
    _require(min_height <= 30, "height budget")
    height = int(rng.integers(max(12, min_height), 31))
    slack = rng.multinomial(height-min_height, np.full(len(labels)+1, 1/(len(labels)+1)))
    cursor, levels = int(slack[0]), {}
    for i, label in enumerate(labels):
        levels[label] = cursor
        cursor += level_heights[i]+1+int(slack[i+1])
    minimum_widths = []
    for p, group in zip(periods, class_objects):
        max_width = max(o['mask'].shape[1] for o in group)
        step = (max_width+1+p-1)//p
        minimum_widths.append(max_width+(len(group)-1)*step)
    # A spare column permits a nonuniform gap witness even for wide masks.
    min_width = max(16, max(minimum_widths)+1)
    _require(min_width <= 30, "width budget")
    width = int(rng.integers(min_width, 31))
    for j, (p, group) in enumerate(zip(periods, class_objects)):
        max_width = max(o['mask'].shape[1] for o in group)
        step = (max_width+1+p-1)//p
        spare = width-minimum_widths[j]
        used = int(rng.integers(1, spare+1))
        extras = rng.multinomial(used, np.full(len(group)-1, 1/(len(group)-1)))
        if plan.get('irregular') and len(set(int(x) for x in extras)) == 1:
            extras[:] = 0
            extras[int(rng.integers(len(extras)))] = used
        left = int(rng.integers(spare-used+1))
        for i, obj in enumerate(group):
            obj['left'], obj['top'] = left, levels[j, obj['phase']]
            if i < len(group)-1:
                left += step+int(extras[i])
    clean_tops = [o['top'] for o in objects]
    # Place the errors using cell-accurate clearance against the current scene.
    for j, errors in enumerate(error_objects):
        for e, obj in enumerate(errors):
            others = set().union(*(_cells(o) for o in objects if o is not obj))
            blocked = _halo(others)
            candidates = [int(y) for y in rng.permutation(height-obj['mask'].shape[0]+1)
                          if int(y) != obj['top']]
            sign = plan.get('sign', 0) if j == e == 0 else 0
            if sign:
                candidates = [y for y in candidates if (y-obj['top'])*sign > 0]
            candidates = [y for y in candidates if not _cells(obj, y) & blocked]
            _require(bool(candidates), "no legal error displacement")
            obj['top'] = candidates[0]
    # Verify the corrupted sequence still determines the constructed target.
    for j, group in enumerate(class_objects):
        p, repaired, _, _ = _recover_rhythm([o['top'] for o in group])
        expected = tuple(levels[j, i % periods[j]] for i in range(len(group)))
        _require(p == periods[j] and repaired == expected, "competing rhythm")
    color_count = int(rng.integers(1, 10))
    for obj in objects:
        obj['color'] = int(rng.choice(palette[:color_count]))
    if plan.get('color'):
        error_objects[0][0]['color'] = palette[0]
        next(o for o in class_objects[0] if o is not error_objects[0][0])['color'] = palette[1]
    clean_occupied = set().union(*(_cells(o, y) for o, y in zip(objects, clean_tops)))
    noisy_occupied = set().union(*(_cells(o) for o in objects))
    singleton_count = plan['singletons']
    for i in range(singleton_count):
        mask = (np.fliplr(bases[0]).copy() if plan.get('mirror') and i == 0
                else _sample_mask(rng, forbidden))
        mask = _crop(np.rot90(mask, int(rng.integers(4))))
        forbidden.update((_canonical(mask), _mirror(mask)))
        obj = {'mask': mask, 'color': int(rng.choice(palette[:color_count]))}
        if plan.get('color') and i == singleton_count-1:
            obj['color'] = palette[0]
        blocked = _halo(clean_occupied | noisy_occupied)
        found = False
        for position in rng.permutation((height-mask.shape[0]+1)*(width-mask.shape[1]+1)):
            y, x = divmod(int(position), width-mask.shape[1]+1)
            obj['top'], obj['left'] = y, x
            points = _cells(obj)
            if not points & blocked:
                found = True
                objects.append(obj)
                clean_tops.append(y)
                clean_occupied.update(points)
                noisy_occupied.update(points)
                break
        _require(found, "singleton packing budget exhausted")
    out = _paint(objects, clean_tops, (height, width), background)
    inp = _paint(objects, [o['top'] for o in objects], (height, width), background)
    _require(int(np.count_nonzero(inp == background))*2 > inp.size, "background not dominant")
    _require(np.array_equal(solve(inp), out), "construction and recovery disagree")
    return {'input': inp, 'output': out}


def _make_pair(rng, plan, stable_background):
    last_error = None
    for _ in range(96):
        try:
            pair = _construct_pair(rng, plan, stable_background)
            if plan.get('mirror'):
                _require(not np.array_equal(_alternative_prediction(pair['input'], 'reflection'), pair['output']),
                         "mirror anchor fails to distinguish reflection-inclusive grouping")
            if plan.get('color'):
                _require(not np.array_equal(_alternative_prediction(pair['input'], 'color'), pair['output']),
                         "colors accidentally reproduce the rhythm")
            return pair
        except ValueError as error:
            last_error = str(error)
    raise RuntimeError(f"bounded pair construction exhausted: {last_error}")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _fingerprint(a):
    return sha256(bytes(a.shape)+a.astype(np.uint8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    _require(a > b, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    stable_bg = int(rng.integers(10)) if rng.random() < .2 else None
    plans = []
    for _ in range(a):
        periods = [int(rng.choice([2, 3])) for _ in range(int(rng.integers(1, 3)))]
        plans.append({'periods': periods, 'sizes': [9 if p == 3 else int(rng.integers(6, 10)) for p in periods],
                      'singletons': int(rng.integers(0, 4))})
    simple, crossed = [int(i) for i in rng.choice(a, size=2, replace=False)]
    simple_size = int(rng.choice([7, 8]))
    plans[simple] = {'periods': [2], 'sizes': [simple_size], 'singletons': 2,
                     'errors': [1], 'mirror': True, 'color': True, 'sign': 1, 'irregular': True,
                     'top_witness': True}
    pair_size = 7 if simple_size == 8 else int(rng.integers(6, 9))
    plans[crossed] = {'periods': [2, 3], 'sizes': [pair_size, 9], 'singletons': int(rng.integers(1, 4)),
                      'errors': [1, 2], 'sign': -1, 'irregular': True, 'diagonal': True}
    for _ in range(b):
        periods = [3]
        if rng.random() < .7:
            periods.append(int(rng.choice([2, 3])))
        if len(periods) > 1 and rng.random() < .5:
            periods.reverse()
        plans.append({'periods': periods, 'sizes': [10 if p == 3 else int(rng.choice([8, 9])) for p in periods],
                      'singletons': int(rng.integers(0, 4))})
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, plan in enumerate(plans):
        for _ in range(8):
            pair = _make_pair(rng, plan, stable_bg)
            k = _fingerprint(pair['input'])
            if k not in seen:
                seen.add(k)
                puzzle['train' if i < a else 'test'].append(pair)
                break
        else:
            raise RuntimeError("distinct-input budget exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_prediction(a, mode, objects=None, bg=None):
    """Concrete shortcut predictor, leaving unsupported classes unchanged."""
    bg = _background(a) if bg is None else bg
    objects = _segment(a, bg) if objects is None else objects
    tops = {id(o): o['top'] for o in objects}
    for group in _groups(objects, mode=mode).values():
        ordered = sorted(group, key=lambda o: o['left'])
        if len(group) >= 6 and len({o['left'] for o in group}) == len(group):
            try:
                _, rows, _, _ = _recover_rhythm([o['top'] for o in ordered])
            except ValueError:
                continue
            for obj, row in zip(ordered, rows):
                tops[id(obj)] = row
    try:
        return _paint(objects, [tops[id(o)] for o in objects], a.shape, bg)
    except ValueError:
        return a.copy()


def _signature(obj, top):
    return int(top), obj['left'], obj['color'], _key(obj['mask'])


def _analyze_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape, "dimensions changed")
    bg = _background(inp)
    _require(_background(out) == bg, "background changed")
    objects = _segment(inp, bg)
    _require(all(1 <= n <= 4 for o in objects for n in o['mask'].shape), "object extent")
    # Reject touching different-color objects too, independently of segmentation.
    reconstructed = _paint(objects, [o['top'] for o in objects], inp.shape, bg)
    _require(np.array_equal(reconstructed, inp), "input segmentation failed")
    dest, details = _destinations(objects)
    expected = sorted(_signature(o, dest[id(o)]) for o in objects)
    output_objects = _segment(out, bg)
    actual = sorted(_signature(o, o['top']) for o in output_objects)
    _require(expected == actual, "object changed, merged, lost or incorrectly moved")
    _require(np.array_equal(solve(inp.copy()), out), "input-only solver disagrees")
    _require(not np.array_equal(inp, out), "omitting periodic correction must change the answer")
    _require(np.array_equal(_paint(output_objects, [o['top'] for o in output_objects], out.shape, bg), out),
             "output objects touch")
    _require(1 <= len(details) <= 2, "class count")
    singles = [g[0] for g in _groups(objects).values() if len(g) == 1]
    _require(len(singles) <= 3, "too many anchors")
    repeated_keys = {d['objects'][0]['canonical'] for d in details}
    directions, unique_rotation_error = set(), False
    top_reference_witness = False
    for d in details:
        _require(1 <= d['errors'] <= 2, "generated class must contain one or two errors")
        _require(all(1 <= n <= 4 for o in d['objects'] for n in o['mask'].shape), "object extent")
        _require(_recover_rhythm(d['rows'])[3] == 0, "output not periodic")
        orientations = Counter(_key(o['mask']) for o in d['objects'])
        for obj, row in zip(d['objects'], d['rows']):
            if row != obj['top']:
                directions.add(1 if row > obj['top'] else -1)
                unique_rotation_error |= orientations[_key(obj['mask'])] == 1
                top_reference_witness |= any(row == peer_row and obj['mask'].shape[0] != peer['mask'].shape[0]
                                             for peer, peer_row in zip(d['objects'], d['rows']))
    ordinary = [o for o in singles if _mirror(o['mask']) not in repeated_keys]
    colors = [o['color'] for o in objects]
    return {'shape': inp.shape, 'background': bg, 'classes': len(details),
            'periods': tuple(d['period'] for d in details),
            'sizes': tuple(len(d['objects']) for d in details),
            'errors': tuple(d['errors'] for d in details),
            'directions': directions, 'singleton_count': len(singles),
            'mirror_anchor': any(_mirror(o['mask']) in repeated_keys for o in singles),
            'ordinary_anchor': bool(ordinary), 'interleaved': _interleaved(details),
            'unique_rotation_error': unique_rotation_error,
            'top_reference_witness': top_reference_witness,
            'moving_diagonal': any(row != o['top'] and not _four_connected(o['mask'])
                                    for d in details for o, row in zip(d['objects'], d['rows'])),
            'different_color_companions': any(len({o['color'] for o in d['objects']}) > 1 for d in details),
            'color_cross_identity': any(o['color'] == q['color'] and o['canonical'] != q['canonical']
                                        for o in objects for q in objects),
            'error_shares_anchor_color': any(o['color'] == q['color'] and row != o['top']
                                             for d in details for o, row in zip(d['objects'], d['rows'])
                                             for q in ordinary),
            'irregular_gaps': any(len(set(np.diff([o['left'] for o in d['objects']]).tolist())) > 1
                                   for d in details),
            'partial_cycle': any(len(d['objects']) % d['period'] != 0 for d in details),
            'masks': {_canonical(o['mask']) for o in objects},
            'areas': {int(o['mask'].sum()) for o in objects},
            'palette_size': len(set(colors)),
            'identity_ablation_fails': not np.array_equal(_alternative_prediction(inp, 'identity', objects, bg), out),
            'reflection_ablation_fails': not np.array_equal(_alternative_prediction(inp, 'reflection', objects, bg), out),
            'color_ablation_fails': not np.array_equal(_alternative_prediction(inp, 'color', objects, bg), out)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists required")
    a = _count(len(puzzle['train']), 'training count', 2, 10)
    b = _count(len(puzzle['test']), 'test count', 1, 4)
    _require(a > b, "training count must exceed test count")
    seen, fs = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp = _grid(pair['input'])
            key = _fingerprint(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            fs[split].append(_analyze_pair(pair))
    tr, te = fs['train'], fs['test']
    _require({p for f in tr for p in f['periods']} == {2, 3}, "teach both periods")
    _require(any(f['classes'] == 1 for f in tr) and any(f['interleaved'] for f in tr), "teach independent virtual lanes")
    for feature in ('mirror_anchor', 'ordinary_anchor', 'different_color_companions',
                    'error_shares_anchor_color', 'unique_rotation_error', 'identity_ablation_fails',
                    'reflection_ablation_fails', 'color_ablation_fails', 'irregular_gaps', 'partial_cycle',
                    'top_reference_witness', 'moving_diagonal'):
        _require(any(f[feature] for f in tr), f"missing training witness: {feature}")
    _require(set().union(*(f['directions'] for f in tr)) == {-1, 1}, "teach both movement directions")
    _require(any(2 in f['errors'] for f in tr), "teach two errors")
    _require(all(max(f['sizes']) <= 9 for f in tr), "training class-size range")
    _require(all(any(p == 3 and n == 10 for p, n in zip(f['periods'], f['sizes'])) for f in te),
             "test must extend a three-phase rhythm to ten members")
    return None


def _puzzle_digest(puzzle):
    return tuple(_fingerprint(np.asarray(pair[role])) for s in ('train', 'test')
                 for pair in puzzle[s] for role in ('input', 'output'))


def self_test():
    """Bounded, streamed 42-seed family checks. No files are written."""
    signatures, shapes, masks, backgrounds, periods, sizes, areas = (set() for _ in range(7))
    profiles, palettes, errors = set(), set(), set()
    pair_count = intertwined = mirrors = partial = ablations = diagonal_pairs = 0
    for seed in range(42):
        a = 2+seed % 9
        b = 1+seed % min(4, a-1)
        puzzle = generate_puzzle(seed, a, b)
        signatures.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                pair_count += 1
                shapes.add(f['shape'])
                masks.update(f['masks'])
                backgrounds.add(f['background'])
                periods.update(f['periods'])
                sizes.update(f['sizes'])
                areas.update(f['areas'])
                profiles.add((f['classes'], f['periods'], f['sizes'], f['singleton_count']))
                palettes.add(f['palette_size'])
                errors.update(f['errors'])
                intertwined += f['interleaved']
                mirrors += f['mirror_anchor']
                partial += f['partial_cycle']
                ablations += f['identity_ablation_fails']
                diagonal_pairs += f['moving_diagonal']
        if seed in (0, 17, 41):
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed, a, b))
    assert len(signatures) == 42 and len(shapes) >= 30 and len(masks) >= 45
    assert backgrounds == set(range(10)) and periods == {2, 3}
    assert {6, 7, 8, 9, 10} <= sizes and len(areas) >= 9
    assert len(profiles) >= 25 and {1, 2} == errors and len(palettes) >= 5
    assert intertwined >= 42 and mirrors >= 42 and partial >= 42 and ablations >= 42
    for a, b in ((2, 1), (10, 1), (10, 4)):
        assert len(generate_puzzle(8000+a+b, a, b)['train']) == a
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Deliberate label damage must be rejected.
    bad = generate_puzzle(9000)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    for rows in ((0, 1, 2, 3, 4, 5), (1, 1, 1, 1, 1, 1)):
        try:
            _recover_rhythm(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported or constant rhythm accepted")
    # Four observations per phase split evenly between two possible heights.
    try:
        _recover_rhythm([0, 1, 2, 3, 0, 1, 2, 3])
    except ValueError:
        pass
    else:
        raise AssertionError("tied phase evidence accepted")
    # Input-only support beyond the generator's disjoint phase-interval packing:
    # two sparse diagonal objects at successive levels have overlapping row
    # envelopes but are widely separated in columns. No construction metadata.
    diagonal = np.eye(3, dtype=bool)
    objs = [{'mask': diagonal, 'color': 2, 'left': 4*i, 'top': [1, 3][i % 2]}
            for i in range(6)]
    target = _paint(objs, [o['top'] for o in objs], (9, 24), 0)
    objs[2]['top'] = 6
    source = _paint(objs, [o['top'] for o in objs], (9, 24), 0)
    assert np.array_equal(solve(source), target)
    # Two errors in the SAME five-observation phase: support is decided by the
    # visible strict majority, not the constructor's distinct-error-phase policy.
    objs = [{'mask': np.eye(2, dtype=bool), 'color': 3, 'left': 3*i, 'top': [2, 7][i % 2]}
            for i in range(10)]
    target = _paint(objs, [o['top'] for o in objs], (12, 30), 0)
    objs[0]['top'], objs[2]['top'] = 0, 4
    source = _paint(objs, [o['top'] for o in objs], (12, 30), 0)
    assert np.array_equal(solve(source), target)
    return {'seeds': 42, 'pairs': pair_count, 'puzzles': len(signatures),
            'grid_shapes': len(shapes), 'rotational_shapes': len(masks),
            'backgrounds': sorted(backgrounds), 'periods': sorted(periods),
            'class_sizes': sorted(sizes), 'areas': sorted(areas), 'profiles': len(profiles),
            'palette_sizes': sorted(palettes), 'interleaved_pairs': intertwined,
            'mirror_anchor_pairs': mirrors, 'partial_cycle_pairs': partial,
            'identity_ablation_differences': ablations, 'moving_diagonal_pairs': diagonal_pairs}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
