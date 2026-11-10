"""Fit-and-print: cavity geometry registers reusable, explicitly encoded stamps.

Only Python's standard library and NumPy are used. Recovery never reads a seed,
an output, training examples, or the choices made during construction.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_N4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
            if (r, c) != (0, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask, eight=False):
    remaining = {tuple(map(int, q)) for q in np.argwhere(mask)}
    result = []
    neighbors = _N8 if eight else _N4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _bounds(cells):
    return (min(r for r, _ in cells), max(r for r, _ in cells) + 1,
            min(c for _, c in cells), max(c for _, c in cells) + 1)


def _mask(cells):
    r0, r1, c0, c1 = _bounds(cells)
    a = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in cells:
        a[r - r0, c - c0] = True
    return a


def _transform(a, reflected, turns):
    return np.rot90(np.fliplr(a) if reflected else a, turns).copy()


def _transforms(mask, ink=None):
    for reflected in (False, True):
        for turns in range(4):
            yield (_transform(mask, reflected, turns),
                   None if ink is None else _transform(ink, reflected, turns),
                   reflected, turns)


def _canonical(mask):
    return min(_key(m) for m, _, _, _ in _transforms(mask))


def _pattern_canonical(plate):
    # Joint D4 normalization avoids counting tool rotation as a new texture.
    pattern = plate['mask'].astype(np.uint8)
    pattern[plate['ink']] = 2
    return _canonical(pattern)


def _expand(a, k):
    return np.repeat(np.repeat(a, k, axis=0), k, axis=1)


def _registered(plate, cavity):
    """All fits must agree about ink, even when the silhouette is symmetric."""
    matches = [(ink, reflected, turns)
               for mask, ink, reflected, turns in _transforms(plate['mask'], plate['ink'])
               if np.array_equal(mask, cavity)]
    if not matches:
        raise ValueError("a requested plate does not fit its cavity")
    if len({_key(ink) for ink, _, _ in matches}) != 1:
        raise ValueError("silhouette fits give different impressions")
    return matches[0][0], matches


def _decode_plate(inp, cells, k, bg, wall):
    r0, r1, c0, c1 = _bounds(cells)
    block = inp[r0:r1, c0:c1]
    if block.shape[0] % k or block.shape[1] % k:
        raise ValueError("plate bounds are not macrocells")
    logical = block[::k, ::k].copy()
    if not np.array_equal(_expand(logical, k), block):
        raise ValueError("broken colored macrocell")
    mask = logical != bg
    if len(_components(mask)) != 1 or int(mask.sum()) < 4:
        raise ValueError("plate silhouette must have at least four connected cells")
    if len(cells) != int(mask.sum()) * k * k:
        raise ValueError("foreign material inside a plate bounding box")
    counts = Counter(map(int, logical[mask]))
    if len(counts) != 2 or wall in counts:
        raise ValueError("a plate needs two nonstructural colors")
    ranked = sorted(counts, key=lambda color: (-counts[color], color))
    code, color = ranked
    if counts[code] <= counts[color]:
        raise ValueError("carrier/ink majority tie")
    ink = logical == color
    # The full family permits symmetric silhouettes only with invariant ink.
    _registered({'mask': mask, 'ink': ink}, mask)
    return {'code': code, 'color': color, 'mask': mask, 'ink': ink,
            'logical': logical, 'bounds': (r0, r1, c0, c1)}


def _board_candidates(inp, bg):
    for wall in sorted(set(map(int, inp.ravel())) - {bg}):
        for cells in _components(inp == wall):
            r0, r1, c0, c1 = _bounds(cells)
            if min(r1 - r0, c1 - c0) < 5:
                continue
            board = inp[r0:r1, c0:c1]
            if not (np.all(board[0] == wall) and np.all(board[-1] == wall)
                    and np.all(board[:, 0] == wall) and np.all(board[:, -1] == wall)):
                continue
            holes = _components(board != wall)
            if 2 <= len(holes) <= 6:
                yield wall, (r0, r1, c0, c1), board, holes


def _interpret(inp, bg, wall, bounds, board, holes, external, k):
    plates = [_decode_plate(inp, cells, k, bg, wall) for cells in external]
    codes = {p['code'] for p in plates}
    inks = {p['color'] for p in plates}
    if len(codes) != len(plates) or len(inks) != len(plates) or codes & inks:
        raise ValueError("plate color roles collide")
    classes = {_canonical(p['mask']): p for p in plates}
    if len(classes) != len(plates):
        raise ValueError("different tools have congruent carrier silhouettes")
    book = {p['code']: p for p in plates}
    logical = np.where(board == wall, wall, bg).astype(np.int8)
    records = []
    for cells in holes:
        cavity = _mask(cells)
        plate = classes.get(_canonical(cavity))
        if plate is None:
            raise ValueError("a cavity has no matching tool")
        r0, r1, c0, c1 = _bounds(cells)
        points = [(r, c, int(board[r, c])) for r, c in cells if board[r, c] != bg]
        if len(points) > 1 or any(code not in book for _, _, code in points):
            raise ValueError("a cavity needs zero or one code token")
        if points and points[0][2] != plate['code']:
            raise ValueError("token code and cavity geometry disagree")
        active = bool(points and points[0][:2] == (r0, c0))
        ink, matches = _registered(plate, cavity)
        if active:
            view = logical[r0:r1, c0:c1]
            view[ink] = plate['color']
        source = plate['ink']
        unregistered = np.zeros(cavity.shape, dtype=bool)
        h, w = min(cavity.shape[0], source.shape[0]), min(cavity.shape[1], source.shape[1])
        unregistered[:h, :w] = source[:h, :w]
        unregistered &= cavity
        # A visibly necessary orientation, not a nominal rotation of a square.
        moved = active and not np.array_equal(ink, unregistered)
        reflection = active and all(reflect for _, reflect, _ in matches)
        rotation = (active and not reflection
                    and not np.array_equal(plate['mask'], cavity)
                    and any(not reflect for _, reflect, _ in matches))
        records.append({'cells': cells, 'bounds': (r0, r1, c0, c1),
                        'code': plate['code'], 'color': plate['color'],
                        'mask': cavity, 'ink': ink, 'active': active,
                        'misplaced': bool(points and not active), 'blank': not points,
                        'rotation': rotation, 'reflection': reflection,
                        'moved': moved, 'unregistered': unregistered})
    used = Counter(r['code'] for r in records if r['active'])
    if set(used) != codes:
        raise ValueError("every supplied tool needs an active request")
    out = _expand(logical, k)
    r0, r1, c0, c1 = bounds
    location = {'above': any(p['bounds'][0] < r0 for p in plates),
                'below': any(p['bounds'][1] > r1 for p in plates),
                'left': any(p['bounds'][2] < c0 for p in plates),
                'right': any(p['bounds'][3] > c1 for p in plates)}
    info = {'background': bg, 'wall': wall, 'scale': k, 'board': board,
            'board_bounds': bounds, 'plates': plates, 'book': book,
            'records': records, 'logical': logical, 'location': location,
            'active': sum(r['active'] for r in records),
            'blank': sum(r['blank'] for r in records),
            'misplaced': sum(r['misplaced'] for r in records),
            'reuse': max(used.values()) > 1,
            'rotation': any(r['rotation'] and r['moved'] for r in records),
            'reflection': any(r['reflection'] and r['moved'] for r in records),
            'shape_classes': tuple(sorted(classes)),
            'pattern_classes': tuple(sorted(_pattern_canonical(p) for p in plates))}
    return out, info


def _recover(value):
    inp = _grid(value)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size // 2:
        raise ValueError("input background is not a strict majority")
    solutions = []
    for wall, bounds, board, holes in _board_candidates(inp, bg):
        r0, r1, c0, c1 = bounds
        outside = np.ones(inp.shape, dtype=bool)
        outside[r0:r1, c0:c1] = False
        moat = np.zeros(inp.shape, dtype=bool)
        moat[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1] = True
        if np.any((inp != bg) & outside & moat):
            continue
        foreground = (inp != bg) & outside
        external = _components(foreground)
        if not 1 <= len(external) <= 3 or len(_components(foreground, True)) != len(external):
            continue
        separated = True
        for cells in external:
            a, b, c, d = _bounds(cells)
            own = np.zeros(inp.shape, dtype=bool)
            for q in cells:
                own[q] = True
            if np.any((foreground & ~own)[max(0, a - 1):b + 1, max(0, c - 1):d + 1]):
                separated = False
                break
        if not separated:
            continue
        # Bounding rectangles, not just ink pixels, have a separating moat.
        for k in range(2, 30 // max(board.shape) + 1):
            try:
                solution = _interpret(inp, bg, wall, bounds, board, holes, external, k)
            except ValueError:
                continue
            solutions.append(solution)
    if len(solutions) != 1:
        raise ValueError("expected one board, scale and registered impression interpretation")
    return solutions[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the answer from this grid alone, without any generation state."""
    return _recover(input_grid)[0]


def _grow(rng, area, span):
    cells = {(0, 0)}
    style = int(rng.integers(3))
    while len(cells) < area:
        options, weights = [], []
        for q in sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4} - cells):
            new = cells | {q}
            r0, r1, c0, c1 = _bounds(new)
            if max(r1 - r0, c1 - c0) > span:
                continue
            degree = sum((q[0] + dr, q[1] + dc) in cells for dr, dc in _N4)
            elongation = max(r1 - r0, c1 - c0) / min(r1 - r0, c1 - c0)
            options.append(q)
            weights.append((1 / degree, 3 ** degree, elongation ** 2)[style])
        if not options:
            break
        weights = np.asarray(weights, dtype=float)
        cells.add(options[int(rng.choice(len(options), p=weights / weights.sum()))])
    return _mask(cells)


def _ink_orbits(mask):
    """Orbits under carrier automorphisms make textured symmetric fits agree."""
    automorphisms = [(reflect, turns) for m, _, reflect, turns in _transforms(mask)
                     if np.array_equal(m, mask)]
    remaining = {tuple(map(int, q)) for q in np.argwhere(mask)}
    orbits = []
    while remaining:
        q = min(remaining)
        dot = np.zeros(mask.shape, dtype=bool)
        dot[q] = True
        orbit = set()
        for reflected, turns in automorphisms:
            orbit.update(tuple(map(int, x)) for x in np.argwhere(_transform(dot, reflected, turns)))
        remaining -= orbit
        orbits.append(tuple(sorted(orbit)))
    return orbits


def _sample_plate(rng, code, color, bg, k, class_count, forbidden, witness=False):
    # More simultaneous holes and larger k need more compact tooling.
    span = 4 if k == 2 and class_count <= 2 else 3
    max_area = 10 if span == 4 else (7 if k < 4 else 5)
    for _ in range(150):
        mask = _grow(rng, int(rng.integers(4, max_area + 1)), span)
        key = _canonical(mask)
        if key in forbidden or int(mask.sum()) < 4:
            continue
        symmetry_count = len({_key(m) for m, _, _, _ in _transforms(mask)})
        if witness and symmetry_count != 8:
            continue
        budget = (int(mask.sum()) - 1) // 2
        orbits = [orbit for orbit in _ink_orbits(mask) if len(orbit) <= budget]
        if not orbits:
            continue
        rng.shuffle(orbits)
        ink = np.zeros(mask.shape, dtype=bool)
        target = int(rng.integers(1, budget + 1))
        for orbit in orbits:
            if int(ink.sum()) + len(orbit) > target:
                continue
            for q in orbit:
                ink[q] = True
        if not ink.any():
            for q in orbits[0]:
                ink[q] = True
        reflect, turns = bool(rng.integers(2)), int(rng.integers(4))
        mask, ink = _transform(mask, reflect, turns), _transform(ink, reflect, turns)
        logical = np.full(mask.shape, bg, dtype=np.int8)
        logical[mask] = code
        logical[ink] = color
        return {'code': code, 'color': color, 'mask': mask, 'ink': ink, 'logical': logical}
    raise ValueError("bounded plate sampling exhausted")


def _hole_choices(plate, witness=None):
    choices = []
    for mask, ink, reflected, turns in _transforms(plate['mask'], plate['ink']):
        if not mask[0, 0]:
            continue
        if witness == 'rotation':
            if reflected or np.array_equal(mask, plate['mask']):
                continue
        elif witness == 'reflection':
            if not reflected or any(np.array_equal(mask, m)
                                    for m, _, reflect, _ in _transforms(plate['mask']) if not reflect):
                continue
        if witness:
            original = np.zeros(mask.shape, dtype=bool)
            h, w = min(mask.shape[0], plate['ink'].shape[0]), min(mask.shape[1], plate['ink'].shape[1])
            original[:h, :w] = plate['ink'][:h, :w]
            if np.array_equal(ink, original & mask):
                continue
        choices.append((mask, ink))
    return choices


def _place_holes(rng, plates, requests, bh, bw):
    for _ in range(12):
        choices = []
        for code, state, witness in requests:
            options = _hole_choices(plates[code], witness)
            if not options:
                return None
            choices.append(options[int(rng.integers(len(options)))])
        occupied, placements = set(), [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: (-int(choices[i][0].sum()), float(rng.random())))
        for i in order:
            mask, ink = choices[i]
            h, w = mask.shape
            positions = [(r, c) for r in range(1, bh - h) for c in range(1, bw - w)]
            for index in rng.permutation(len(positions)):
                r0, c0 = positions[int(index)]
                cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
                if any(q in occupied or any((q[0] + dr, q[1] + dc) in occupied for dr, dc in _N4)
                       for q in cells):
                    continue
                placements[i] = (r0, c0, mask, ink, cells)
                occupied.update(cells)
                break
            if placements[i] is None:
                break
        else:
            return placements
    return None


def _free_origins(blocked, h, w):
    H, W = blocked.shape
    if h > H or w > W:
        return np.empty((0, 2), dtype=int)
    summed = np.pad(blocked.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return np.argwhere(totals == 0)


def _pack(rng, board, plates, k, bg):
    objects = [board] + [_expand(p['logical'], k) for p in plates]
    for attempt in range(36):
        H, W = ((30, 30) if attempt >= 16 else
                (int(rng.integers(max(16, max(a.shape[0] for a in objects)), 31)),
                 int(rng.integers(max(16, max(a.shape[1] for a in objects)), 31))))
        inp = np.full((H, W), bg, dtype=np.int8)
        blocked = np.zeros((H, W), dtype=bool)
        order = sorted(range(len(objects)), key=lambda i: (-objects[i].size, float(rng.random())))
        for i in order:
            obj = objects[i]
            h, w = obj.shape
            positions = _free_origins(blocked, h, w)
            if not len(positions):
                break
            r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
            inp[r0:r0 + h, c0:c0 + w] = obj
            blocked[max(0, r0 - 1):min(H, r0 + h + 1), max(0, c0 - 1):min(W, c0 + w + 1)] = True
        else:
            if np.count_nonzero(inp == bg) <= inp.size // 2:
                continue
            if rng.random() < .65:
                r0, r1, c0, c1 = _bounds(tuple(map(tuple, np.argwhere(inp != bg))))
                r0, c0 = max(0, r0 - int(rng.integers(4))), max(0, c0 - int(rng.integers(4)))
                r1, c1 = min(H, r1 + int(rng.integers(4))), min(W, c1 + int(rng.integers(4)))
                crop = inp[r0:r1, c0:c1]
                if np.count_nonzero(crop == bg) > crop.size // 2:
                    inp = crop.copy()
            return inp
    return None


def _make_pair(rng, palette, k, mode, more_reuse=False):
    all_codes, ink_map, spare = palette
    for _ in range(180):
        bg, wall = map(int, rng.choice(spare, 2, replace=False))
        if mode in ('reuse', 'inactive'):
            codes = list(all_codes)
        else:
            n = 1 if k == 4 else int(rng.integers(1, len(all_codes) + 1))
            codes = list(map(int, rng.choice(all_codes, n, replace=False)))
        if k == 4:
            codes = codes[:1]
        rng.shuffle(codes)
        forbidden, plates = set(), {}
        try:
            for i, code in enumerate(codes):
                plate = _sample_plate(rng, code, ink_map[code], bg, k, len(codes), forbidden,
                                      witness=i == 0 and mode in ('reuse', 'inactive'))
                plates[code] = plate
                forbidden.add(_canonical(plate['mask']))
        except ValueError:
            continue
        requests = [(code, 'active', None) for code in codes]
        if mode == 'reuse':
            requests[0] = (codes[0], 'active', 'rotation')
            requests.append((codes[0], 'active', None))
        elif mode == 'inactive':
            requests[0] = (codes[0], 'active', 'reflection')
            requests.extend([(codes[0], 'blank', None), (int(rng.choice(codes)), 'misplaced', None)])
        elif k == 4:
            state = str(rng.choice(['active', 'blank', 'misplaced']))
            requests.append((codes[0], state, None))
        else:
            extras = int(rng.integers(2, 4)) if more_reuse else int(rng.integers(0, 3))
            for _extra in range(extras):
                requests.append((int(rng.choice(codes)), 'active', None))
            if len(requests) < 2:
                requests.append((codes[0], str(rng.choice(['active', 'blank', 'misplaced'])), None))
            for _extra in range(int(rng.integers(0, min(2, 6 - len(requests)) + 1))):
                requests.append((int(rng.choice(codes)), str(rng.choice(['blank', 'misplaced'])), None))
        requests = requests[:6]
        limit = 30 // k
        if k == 4:
            bh, bw = int(rng.integers(6, 8)), int(rng.integers(6, 8))
        else:
            low = min(limit, max(6, 5 + len(requests) // 2))
            bh, bw = int(rng.integers(low, limit + 1)), int(rng.integers(low, limit + 1))
        placements = _place_holes(rng, plates, requests, bh, bw)
        if placements is None:
            continue
        board = np.full((bh, bw), wall, dtype=np.int8)
        logical = board.copy()
        for (code, state, _), (r0, c0, mask, ink, cells) in zip(requests, placements):
            view = board[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
            answer = logical[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
            view[mask], answer[mask] = bg, bg
            if state == 'active':
                view[0, 0] = code
                answer[ink] = ink_map[code]
            elif state == 'misplaced':
                positions = [(int(r), int(c)) for r, c in np.argwhere(mask) if (r, c) != (0, 0)]
                view[positions[int(rng.integers(len(positions)))]] = code
        inp = _pack(rng, board, list(plates.values()), k, bg)
        if inp is None:
            continue
        out = _expand(logical, k)
        recovered, info = _recover(inp)
        if not np.array_equal(recovered, out):
            raise AssertionError("direct construction and input recovery disagree")
        if mode == 'reuse' and not (info['reuse'] and info['rotation']):
            continue
        if mode == 'inactive' and not (info['blank'] and info['misplaced'] and info['reflection']):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded feasible-example construction exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n = int(rng.integers(1, 4))
    colors = list(map(int, rng.permutation(10)))
    codes = tuple(colors[:n])
    palette = codes, dict(zip(codes, colors[n:2 * n])), tuple(colors[2 * n:])
    regime = int(rng.integers(4))
    if regime == 0:
        train_scales, test_scales = [2] * nt, [3] * nq
    elif regime == 1:
        train_scales, test_scales = [3] * nt, [2] * nq
    else:
        train_scales = [2, 3] + [int(rng.choice([2, 3])) for _ in range(nt - 2)]
        test_scales = ([4] * nq if regime == 2 else
                       [int(rng.choice([2, 3])) for _ in range(nq)])
    modes = ['reuse', 'inactive'] + ['ordinary'] * (nt - 2)
    # The two distinct mechanism witnesses do not have a fixed visible order.
    order = list(map(int, rng.permutation(nt)))
    train_scales, modes = [train_scales[i] for i in order], [modes[i] for i in order]
    puzzle, seen, witness_classes = {'train': [], 'test': []}, set(), {}
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, k in enumerate(scales):
            mode = modes[i] if split == 'train' else 'ordinary'
            for _ in range(12):
                pair = _make_pair(rng, palette, k, mode, more_reuse=split == 'test' and regime == 3)
                key = _key(pair['input'])
                classes = _recover(pair['input'])[1]['shape_classes']
                varied = (mode not in ('reuse', 'inactive') or
                          all(classes != old for old in witness_classes.values()))
                if key not in seen and varied:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train' and mode in ('reuse', 'inactive'):
                witness_classes[mode] = classes
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual(info, kind):
    bg, wall = info['background'], info['wall']
    logical = np.where(info['board'] == wall, wall, bg).astype(np.int8)
    for record in info['records']:
        active = (True if kind == 'all_cavities' else
                  record['active'] or (kind == 'all_tokens' and record['misplaced']))
        if not active:
            continue
        r0, r1, c0, c1 = record['bounds']
        if kind == 'no_registration':
            ink = record['unregistered']
        elif kind == 'flood':
            ink = record['mask']
        else:
            ink = record['ink']
        logical[r0:r1, c0:c1][ink] = record['color']
    return _expand(logical, info['scale'])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos, associations = set(), {'train': [], 'test': []}, {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), "input-only recovered output disagrees"
            assert info['scale'] in (2, 3, 4)
            assert not np.array_equal(inp, out)
            assert 2 <= len(info['records']) <= 6
            for plate in info['plates']:
                code, color = plate['code'], plate['color']
                assert code not in associations or associations[code] == color, "code/ink association changed"
                associations[code] = color
                assert 4 <= int(plate['mask'].sum()) <= 10
                assert 0 < int(plate['ink'].sum()) < int(plate['mask'].sum()) / 2
            logical, board, k = info['logical'], info['board'], info['scale']
            assert out.shape == (board.shape[0] * k, board.shape[1] * k)
            assert np.array_equal(logical[board == info['wall']], board[board == info['wall']])
            assert not set(map(int, out.ravel())) & set(info['book'])
            expected_ink = sum(int(r['ink'].sum()) for r in info['records'] if r['active']) * k * k
            assert np.count_nonzero(np.isin(out, [p['color'] for p in info['plates']])) == expected_ink
            assert not np.array_equal(_counterfactual(info, 'flood'), out)
            infos[split].append(info)
    # Palette roles cannot silently switch when a class is absent in one pair.
    all_codes, all_inks = set(associations), set(associations.values())
    assert len(all_codes) == len(all_inks) and not all_codes & all_inks
    for split in infos:
        for info in infos[split]:
            assert {info['background'], info['wall']}.isdisjoint(all_codes | all_inks)
    trained = {p['code'] for info in infos['train'] for p in info['plates']}
    assert all_codes <= trained
    train = infos['train']
    assert any(info['reuse'] and info['rotation'] for info in train), "missing reusable rotated impression"
    assert any(info['blank'] and info['misplaced'] and info['reflection'] for info in train), \
        "missing blank/misplaced/reflected demonstration"
    assert any(not np.array_equal(_counterfactual(info, 'no_registration'), _expand(info['logical'], info['scale']))
               for info in train), "A-omitting shortcut not distinguished"
    for kind in ('all_cavities', 'all_tokens'):
        assert any(not np.array_equal(_counterfactual(info, kind), _expand(info['logical'], info['scale']))
                   for info in train), f"B-omitting shortcut {kind} not distinguished"
    assert len({info['shape_classes'] for info in train}) >= 2, "training tooling must change structurally"


def _puzzle_key(puzzle):
    return tuple((_key(np.asarray(pair['input'])), _key(np.asarray(pair['output'])))
                 for split in ('train', 'test') for pair in puzzle[split])


def _normalized_key(inp):
    mapping = {}
    a = np.asarray([[mapping.setdefault(int(v), len(mapping)) for v in row] for row in inp], dtype=np.uint8)
    return _key(a)


def self_test() -> dict:
    """Bounded suite; metrics are recovered from grids, never sampled flags."""
    signatures, normalized, shapes, boards, classes, patterns, scales = (set() for _ in range(7))
    hole_counts, active_counts, palettes, schedules = (set() for _ in range(4))
    features = Counter()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed == 0:
            first = puzzle
        signatures.add(_puzzle_key(puzzle))
        grouped = {'train': set(), 'test': set()}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _recover(pair['input'].copy())
                assert np.array_equal(predicted, pair['output'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add(pair['input'].shape)
                boards.add(info['board'].shape)
                normalized.add(_normalized_key(pair['input']))
                classes.update(info['shape_classes'])
                patterns.update(info['pattern_classes'])
                scales.add(info['scale'])
                grouped[split].add(info['scale'])
                hole_counts.add(len(info['records']))
                active_counts.add(info['active'])
                palettes.add(tuple(sorted((p['code'], p['color']) for p in info['plates'])))
                for name in ('reuse', 'rotation', 'reflection', 'blank', 'misplaced'):
                    features[name] += bool(info[name])
                for direction, present in info['location'].items():
                    features[direction] += present
                features['symmetric_plate'] += any(len({_key(m) for m, _, _, _ in _transforms(p['mask'])}) < 8
                                                   for p in info['plates'])
        schedules.add((tuple(sorted(grouped['train'])), tuple(sorted(grouped['test']))))
    assert len(signatures) == 42
    assert len(normalized) >= 190 and len(shapes) >= 35 and len(boards) >= 20
    assert len(classes) >= 35 and len(patterns) >= 80 and len(palettes) >= 30
    assert scales == {2, 3, 4} and len(schedules) >= 4
    assert {2, 3, 4, 5} <= hole_counts and len(active_counts) >= 4
    assert all(features[name] >= 10 for name in ('reuse', 'rotation', 'reflection', 'blank', 'misplaced',
                                               'above', 'below', 'left', 'right'))
    assert features['symmetric_plate'] > 0
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(0))
    assert _puzzle_key(first) != _puzzle_key(generate_puzzle(43))
    # Every legal count combination is exercised at a fresh seed.
    count_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9000 + nt * 10 + nq, nt, nq)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
            count_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Whole-stamp deletion must be rejected without reclassifying its token.
    corrupted = {s: [{name: a.copy() for name, a in pair.items()} for pair in first[s]] for s in first}
    info = _recover(corrupted['test'][0]['input'])[1]
    r0, r1, c0, c1 = next(r['bounds'] for r in info['records'] if r['active'])
    k = info['scale']
    corrupted['test'][0]['output'][r0 * k:r1 * k, c0 * k:c1 * k] = info['background']
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("deleted impression accepted")
    # A single removed source pixel is neither a legal scale nor noise.
    inp = first['train'][0]['input'].copy()
    info = _recover(inp)[1]
    p = info['plates'][0]
    r0, _, c0, _ = p['bounds']
    r, c = map(int, np.argwhere(p['mask'])[0])
    inp[r0 + r * info['scale'], c0 + c * info['scale']] = info['background']
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError("broken plate macrocell accepted")
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'normalized_inputs': len(normalized),
            'input_shapes': len(shapes), 'board_shapes': len(boards), 'shape_classes': len(classes),
            'pattern_classes': len(patterns), 'palettes': len(palettes), 'scales': sorted(scales),
            'hole_counts': sorted(hole_counts), 'active_counts': sorted(active_counts),
            'scale_schedules': sorted(schedules), 'features': dict(features), 'legal_count_calls': count_calls}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
