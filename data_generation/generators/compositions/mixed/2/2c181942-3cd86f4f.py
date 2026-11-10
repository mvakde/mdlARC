"""Color/silhouette docking followed by a whole-canvas, bottom-fixed row shear.

Only NumPy and the standard library are required. No parent module, file,
training example, random state, or construction metadata is used by solve.
The hub is rigid in the assembly phase, and is sheared along with everything
else in the final phase: pixel preservation does not mean rigid preservation.
"""
from __future__ import annotations

import hashlib
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_SIDES = frozenset(range(4))  # top, right, bottom, left


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("a grid must be a 1..30 integer rectangle of colors 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _mask(*rows):
    return np.array([[c == '#' for c in row] for row in rows], dtype=bool)


def _hub_cells(r, c):
    return np.array(((r, c + 1), (r, c + 2), (r + 1, c + 3),
                     (r + 2, c + 3), (r + 3, c + 1), (r + 3, c + 2),
                     (r + 1, c), (r + 2, c)), dtype=int)


def _draw_hub(a, r, c, labels):
    coords = _hub_cells(r, c)
    a[coords[:, 0], coords[:, 1]] = np.repeat(labels, 2)


def _hubs(a, bg):
    allowed = _mask('.##.', '#..#', '#..#', '.##.')
    found = []
    for r in range(a.shape[0] - 3):
        for c in range(a.shape[1] - 3):
            b = a[r:r + 4, c:c + 4]
            if np.any(b[~allowed] != bg):
                continue
            v = b[tuple(_hub_cells(0, 0).T)]
            if (np.all(v[::2] == v[1::2]) and bg not in v
                    and len(set(map(int, v))) == 4):
                found.append((r, c, tuple(map(int, v[::2]))))
    return found


def _components(a, bg, r, c):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    remaining.difference_update(map(tuple, _hub_cells(r, c)))
    pieces = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(a[start])
        todo, cells = [start], [start]
        while todo:
            rr, cc = todo.pop()
            for dr, dc in _NEIGHBORS:
                p = (rr + dr, cc + dc)
                if p in remaining and int(a[p]) == color:
                    remaining.remove(p)
                    todo.append(p)
                    cells.append(p)
        coords = np.array(cells, dtype=int)
        low = coords.min(axis=0)
        shape = coords.max(axis=0) - low + 1
        mask = np.zeros(tuple(shape), dtype=bool)
        mask[tuple((coords - low).T)] = True
        pieces.append(dict(color=color, coords=coords, mask=mask))
    return pieces


def _connected(mask, diagonal=True):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    todo = [cells.pop()]
    neighbors = _NEIGHBORS if diagonal else ((0, 1), (0, -1), (1, 0), (-1, 0))
    while todo:
        r, c = todo.pop()
        for dr, dc in neighbors:
            p = (r + dr, c + dc)
            if p in cells:
                cells.remove(p)
                todo.append(p)
    return not cells


def _attachments(mask):
    """Return distinct canonical masks, and all equivalent undo rotations.

    A canonical mask's inward edge is its last row: precisely the centered
    adjacent pair. A square or a bar can have several equivalent choices.
    Different occupied-cell results are ambiguous and are never tie-broken.
    """
    candidates = {}
    for k in range(4):
        m = np.rot90(mask, k).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], neck):
            entry = candidates.setdefault(_key(m), [m, []])
            entry[1].append(k)
    return list(candidates.values())


def _dock(mask, side, r, c):
    neck = np.zeros_like(mask)
    neck[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    oriented = np.rot90(mask, -side)
    attach = np.argwhere(np.rot90(neck, -side))
    target = ((r - 1, c + 1), (r + 1, c + 4),
              (r + 4, c + 1), (r + 1, c - 1))[side]
    return np.argwhere(oriented) + np.array(target) - attach.min(axis=0)


def _separation(pieces, r, c):
    occupied = set()
    for p in pieces:
        cells = set(map(tuple, p['coords']))
        assert all(not (r - 1 <= rr <= r + 4 and c - 1 <= cc <= c + 4)
                   for rr, cc in cells), "loose piece touches hub clearance"
        assert not cells & occupied, "loose pieces touch"
        occupied.update((rr + dr, cc + dc) for rr, cc in cells
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _assemble(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    assert h + w - 1 <= 30, "the widened canvas exceeds 30 columns"
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > a.size / 2, "background must be a strict majority"
    hubs = _hubs(a, bg)
    assert len(hubs) == 1, "exactly one hollow four-label hub is required"
    r, c, labels = hubs[0]
    pieces = _components(a, bg, r, c)
    assert pieces, "at least one loose piece is required"
    _separation(pieces, r, c)
    result = np.full(a.shape, bg, dtype=np.int8)
    _draw_hub(result, r, c, labels)
    used = set()
    for p in pieces:
        p['side'] = labels.index(p['color']) if p['color'] in labels else None
        if p['side'] is None:
            continue  # Unmatched shapes need no docking edge.
        side = p['side']
        assert side not in used, "repeated matching label"
        used.add(side)
        candidates = _attachments(p['mask'])
        assert len(candidates) == 1, "missing or competing geometric attachment"
        canonical, turns = candidates[0]
        coords = _dock(canonical, side, r, c)
        rr, cc = coords.T
        assert np.all((0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)), "clipped docking"
        assert np.all(result[rr, cc] == bg), "overlapping docking"
        result[rr, cc] = p['color']
        p.update(canonical=canonical, target=coords,
                 rotations={(k - side) % 4 for k in turns})
    assert used, "generated family requires at least one matching piece"
    return result, dict(bg=bg, hub=(r, c), labels=labels, pieces=pieces, sides=used)


def _shear(a, bg):
    h, w = a.shape
    out = np.full((h, h + w - 1), bg, dtype=np.int8)
    for r in range(h):
        out[r, h - 1 - r:h - 1 - r + w] = a[r]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer docking from the input, then shear every complete assembled row."""
    assembled, info = _assemble(input_grid)
    return _shear(assembled, info['bg'])


def _sample_mask(rng, directional=False):
    for _ in range(32):
        kind = int(rng.integers(4))
        w = int(rng.choice((2, 4, 6, 8), p=(.18, .43, .31, .08)))
        h = int(rng.integers(2, 6))
        if kind == 0:
            w = 2
        m = np.zeros((h, w), dtype=bool)
        mid = w // 2
        m[:, mid - 1:mid + 1] = True
        stem = int(rng.integers(1, h))
        body = h - stem
        if kind == 1:
            m[:body] = True
        elif kind == 2 and w > 2:
            h = max(h, mid)
            m = np.zeros((h, w), dtype=bool)
            m[mid - 1:, mid - 1:mid + 1] = True
            for rr in range(mid - 1):
                m[rr, rr] = m[rr, w - 1 - rr] = True
        elif kind == 3 and w > 2:
            for rr in range(body):
                lo = int(rng.integers(mid))
                hi = int(rng.integers(mid + 1, w + 1))
                m[rr, lo:hi] = True
            m[int(rng.integers(body)), :mid] = True
            m[int(rng.integers(body)), mid:] = True
            for rr, cc in np.argwhere(m[:body]):
                if cc not in (mid - 1, mid) and rng.random() < .18:
                    m[rr, cc] = False
        rows, cols = np.nonzero(m)
        m = m[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        candidates = _attachments(m)
        if (_connected(m) and len(candidates) == 1
                and (not directional or len(candidates[0][1]) == 1)):
            return candidates[0][0]
    return _mask('####', '.##.')


def _relative(pieces):
    coords = [_hub_cells(0, 0)]
    occupied = set(map(tuple, coords[0]))
    for p in pieces:
        if p['side'] is None:
            continue
        cells = _dock(p['mask'], p['side'], 0, 0)
        if occupied & set(map(tuple, cells)):
            return None
        occupied.update(map(tuple, cells))
        coords.append(cells)
    return np.concatenate(coords)


def _block(blocked, coords):
    h, w = blocked.shape
    for r, c in coords:
        blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True


def _pack(rng, shape, bg, hub, labels, pieces, tiled=False):
    a = np.full(shape, bg, dtype=np.int8)
    r, c = hub
    _draw_hub(a, r, c, labels)
    blocked = np.zeros(shape, dtype=bool)
    blocked[max(0, r - 1):r + 5, max(0, c - 1):c + 5] = True
    order = sorted(map(int, rng.permutation(len(pieces))),
                   key=lambda i: -pieces[i]['mask'].size)
    for i in order:
        p = pieces[i]
        m = np.rot90(p['mask'], p['turn'])
        h, w = m.shape
        local = np.argwhere(m)
        if h > shape[0] or w > shape[1]:
            return None
        if tiled:
            starts = [(rr, cc) for rr in (0, 5, 10) for cc in (0, 6, 12)]
            starts = [x for x in starts if x[0] + h <= shape[0] and x[1] + w <= shape[1]]
        else:
            starts = [(rr, cc) for rr in range(shape[0] - h + 1)
                      for cc in range(shape[1] - w + 1)]
        good = []
        for rr, cc in starts:
            cells = local + (rr, cc)
            if not np.any(blocked[tuple(cells.T)]):
                good.append((rr, cc))
        if not good:
            return None
        cells = local + good[int(rng.integers(len(good)))]
        a[tuple(cells.T)] = p['color']
        _block(blocked, cells)
    return a


def _pair(rng, config, previous_bg, avoid_height=None):
    # A local role swap is visible between the two designated training anchors.
    match_color, absent_color = config.get('roles', (None, None))
    available = [x for x in range(10) if x not in (match_color, absent_color)]
    if match_color is None:
        labels = list(map(int, rng.choice(10, 4, replace=False)))
    else:
        labels = list(map(int, rng.choice(available, 3, replace=False)))
        labels.insert(int(rng.choice(sorted(config['sides']))), match_color)
    bg = int(rng.choice([x for x in range(10) if x not in labels
                        and x != absent_color and x != previous_bg]))
    distractor_pool = [x for x in range(10) if x not in labels and x != bg]
    distractors = list(map(int, rng.choice(distractor_pool, config['distractors'])))
    if absent_color is not None:
        distractors[0] = absent_color
    for attempt in range(25):
        fallback = attempt == 24
        pieces = []
        for side in sorted(config['sides']):
            forced = config['forced'].get(side)
            if forced is not None:
                # Distinct small shapes, each with exactly one attachment turn.
                m = (_mask('####', '.##.') if forced % 2 == 0
                     else _mask('####', '.##.', '.##.'))
                turn = (-side - forced) % 4
            else:
                m = _mask('####', '.##.') if fallback else _sample_mask(rng)
                turn = int(rng.integers(4))
            pieces.append(dict(mask=m, side=side, color=labels[side], turn=turn))
        for i, color in enumerate(distractors):
            if fallback:
                m = _mask('####', '.##.')
            elif i == 0:
                m = _sample_mask(rng)  # A legitimate-looking distractor.
            elif rng.random() < .5:
                m = _mask('.##', '###', '##.')
            else:
                m = _mask('##', '##')
            pieces.append(dict(mask=m, side=None, color=color, turn=int(rng.integers(4))))
        rel = _relative(pieces)
        if rel is None:
            continue
        low, high = rel.min(axis=0), rel.max(axis=0)
        span = high - low + 1
        anchor = config.get('anchor')
        if fallback:
            h = 14 if avoid_height == 15 else 15
            shape = h, 31 - h
            hub = (int(-low[0]) if anchor == 'top' else
                   int(h - 1 - high[0]) if anchor == 'bottom' else 5, 6)
        else:
            choices = [(h, w) for h in range(8, 24) for w in range(8, 24)
                       if 26 <= h + w <= 31 and h >= span[0] and w >= span[1]
                       and h != avoid_height and (anchor != 'top' or h >= span[0] + 2)
                       and h * w > 2 * (8 + sum(int(p['mask'].sum()) for p in pieces))]
            if not choices:
                continue
            shape = choices[int(rng.integers(len(choices)))]
            first = -low
            last = np.array(shape) - 1 - high
            rr = (int(first[0]) if anchor == 'top' else int(last[0]) if anchor == 'bottom'
                  else int(rng.integers(first[0], last[0] + 1)))
            cc = int(rng.integers(first[1], last[1] + 1))
            hub = rr, cc
        for _ in range(1 if fallback else 3):
            inp = _pack(rng, shape, bg, hub, labels, pieces, tiled=fallback)
            if inp is None or np.count_nonzero(inp == bg) <= inp.size / 2:
                continue
            assembled = np.full(shape, bg, dtype=np.int8)
            _draw_hub(assembled, *hub, labels)
            for p in pieces:
                if p['side'] is not None:
                    cells = _dock(p['mask'], p['side'], *hub)
                    assembled[tuple(cells.T)] = p['color']
            out = _shear(assembled, bg)
            # Construction coordinates and input-derived geometry must agree.
            assert np.array_equal(solve(inp), out)
            return dict(input=inp, output=out)
    raise RuntimeError("bounded packing exhausted, including the small-shape fallback")


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(nt):
        n = int(rng.integers(1, 5))
        configs.append(dict(sides=set(map(int, rng.choice(4, n, replace=False))),
                            distractors=int(rng.integers(3)), forced={}))
    a, b = map(int, rng.choice(nt, 2, replace=False))
    order = list(map(int, rng.permutation(4)))
    role_colors = tuple(map(int, rng.choice(10, 2, replace=False)))
    for i, index in enumerate((a, b)):
        config = configs[index]
        config['sides'].update(order[2 * i:2 * i + 2])
        config['forced'] = {side: 2 * i + j for j, side in enumerate(order[2 * i:2 * i + 2])}
        config['distractors'] = max(1, config['distractors'])
        config['roles'] = role_colors if i == 0 else role_colors[::-1]
        config['anchor'] = 'bottom' if i == 0 else 'top'
    seen_subsets = {frozenset(c['sides']) for c in configs}
    unseen = [frozenset(s for s in range(4) if bits & (1 << s))
              for bits in range(1, 16)]
    unseen = [s for s in unseen if s not in seen_subsets]
    rng.shuffle(unseen)
    for sides in unseen[:nq]:
        configs.append(dict(sides=set(sides), distractors=int(rng.integers(3)), forced={}))
    puzzle = dict(train=[], test=[])
    seen, previous_bg, anchor_height = set(), None, None
    # Build the bottom anchor first so the top anchor can use a different height.
    build_order = [a, b] + [i for i in range(nt + nq) if i not in (a, b)]
    built = {}
    for index in build_order:
        for _ in range(4):
            pair = _pair(rng, configs[index], previous_bg,
                         avoid_height=anchor_height if index == b else None)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            raise RuntimeError('bounded distinct-input retries exhausted')
        built[index] = pair
        seen.add(key)
        previous_bg = int(np.argmax(np.bincount(pair['input'].ravel(), minlength=10)))
        if index == a:
            anchor_height = pair['input'].shape[0]
    for i in range(nt + nq):
        puzzle['train' if i < nt else 'test'].append(built[i])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assembled, info = _assemble(inp)
    h, w = inp.shape
    bg = info['bg']
    assert out.shape == (h, h + w - 1)
    for r in range(h):
        start = h - 1 - r
        assert np.array_equal(out[r, start:start + w], assembled[r]), 'wrong sheared row'
        assert np.all(out[r, :start] == bg) and np.all(out[r, start + w:] == bg)
    # Check exact hub transport rather than mistakenly requiring a rigid hub.
    coords = _hub_cells(*info['hub'])
    rr, cc = coords.T
    assert np.array_equal(out[rr, cc + h - 1 - rr], np.repeat(info['labels'], 2))
    matched = [p for p in info['pieces'] if p['side'] is not None]
    unmatched = [p for p in info['pieces'] if p['side'] is None]
    for color in info['labels']:
        count = 2 + sum(len(p['coords']) for p in matched if p['color'] == color)
        assert np.count_nonzero(out == color) == count, 'selected pixels lost or duplicated'
    assert all(not np.any(out == p['color']) for p in unmatched), 'distractor survived'
    assert not np.array_equal(_shear(inp, bg), out), 'docking has no effect'
    bare = np.full_like(inp, bg)
    _draw_hub(bare, *info['hub'], info['labels'])
    assert not np.array_equal(_shear(bare, bg), out), 'matching pieces have no effect'
    unsheared = np.full_like(out, bg)
    unsheared[:, :w] = assembled
    assert not np.array_equal(unsheared, out), 'row shear has no effect'
    info.update(matched=matched, unmatched=unmatched, assembled=assembled)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            analyses[split].append(_check_pair(pair))
    train = analyses['train']
    assert set().union(*(x['sides'] for x in train)) == _SIDES
    turns = set().union(*(p['rotations'] for x in train for p in x['matched']
                         if len(p['rotations']) == 1))
    assert turns == _SIDES, 'training must show all four turn amounts'
    assert any(len(_attachments(p['mask'])) == 1 for x in train for p in x['unmatched'])
    selected = {p['color'] for x in train for p in x['matched']}
    erased = {p['color'] for x in train for p in x['unmatched']}
    assert len(selected & erased) >= 2, 'training must expose local color roles'
    assert len({pair['input'].shape[0] if isinstance(pair['input'], np.ndarray)
                else len(pair['input']) for pair in puzzle['train']}) >= 2
    assert any(np.any(x['assembled'][-1] != x['bg']) for x in train), 'bottom anchor absent'
    assert any(np.all(x['assembled'][-2:] == x['bg']) for x in train), 'blank-margin witness absent'
    shapes = {_key(p['canonical']) for x in train for p in x['matched']}
    assert len(shapes) >= 2
    train_subsets = {frozenset(x['sides']) for x in train}
    assert all(frozenset(x['sides']) not in train_subsets for x in analyses['test'])


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        digest.update(bytes((len(puzzle[split]),)))
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    # A manually enumerated geometry regression; expected pixels do not use
    # either the docking helper or the shear helper.
    inp = np.zeros((12, 17), dtype=np.int8)
    _draw_hub(inp, 4, 6, (1, 2, 3, 4))
    m = np.rot90(_mask('####', '.##.'))
    inp[:4, :2][m] = 1
    m = np.rot90(_mask('####', '.##.', '.##.'), 2)
    inp[9:12, :4][m] = 2
    inp[:2, 14:16] = 5
    expected = np.zeros((12, 28), dtype=np.int8)
    colored = {
        1: [(2, c) for c in range(15, 19)] + [(3, 15), (3, 16), (4, 14), (4, 15)],
        2: [(5, 15), (6, 14), (4, 19), (7, 16)]
           + [(5, c) for c in range(16, 19)] + [(6, c) for c in range(15, 18)],
        3: [(7, 11), (7, 12)], 4: [(5, 12), (6, 11)]}
    for color, cells in colored.items():
        expected[tuple(np.array(cells).T)] = color
    unchanged = inp.copy()
    assert np.array_equal(solve(inp), expected), 'hand-enumerated regression'
    assert np.array_equal(inp, unchanged), 'solver mutated the query'
    assert np.array_equal(solve(inp.tolist()), expected)
    # These checks use the enumerated answer, not sampled construction state.
    # A permutation includes nonzero background and zero as a selected color.
    palette = np.array((8, 0, 9, 5, 2, 1, 7, 4, 6, 3), dtype=np.int8)
    assert np.array_equal(solve(palette[inp]), palette[expected]), 'literal color dependence'
    # Adding blank rows changes the global shear anchor even though docking
    # geometry is unchanged. Adding columns preserves the whole input width.
    for top, bottom, left, right in ((1, 1, 0, 0), (0, 0, 1, 1), (0, 1, 1, 0)):
        extended = np.pad(inp, ((top, bottom), (left, right)))
        answer = np.zeros((extended.shape[0], sum(extended.shape) - 1), dtype=np.int8)
        start = left + bottom
        answer[top:top + inp.shape[0], start:start + expected.shape[1]] = expected
        assert np.array_equal(solve(extended), answer), 'blank margins changed the rule'
    assert len(_attachments(_mask('##', '##'))[0][1]) == 4
    assert len(_attachments(_mask('##', '##', '##'))[0][1]) == 2
    assert _connected(_mask('#..#', '.##.'))
    assert not _connected(_mask('#..#', '.##.'), False)
    assert len(_attachments(_mask('.###', '####', '#.##', '.##.'))) > 1
    signatures, silhouettes, dimensions, backgrounds, subsets, counts = (set() for _ in range(6))
    diagonal = asymmetric = boundary = no_distractor = repeated_color = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                dimensions.add(pair['input'].shape)
                backgrounds.add(info['bg'])
                subsets.add(tuple(sorted(info['sides'])))
                counts.add((len(info['matched']), len(info['unmatched'])))
                boundary += pair['output'].shape[1] == 30
                no_distractor += not info['unmatched']
                repeated_color += len({p['color'] for p in info['unmatched']}) < len(info['unmatched'])
                for p in info['matched']:
                    m = p['canonical']
                    silhouettes.add(_key(m))
                    diagonal += not _connected(m, False)
                    asymmetric += not np.array_equal(m, np.fliplr(m))
    assert len(signatures) == 42
    assert len(silhouettes) >= 20 and len(dimensions) >= 35
    assert len(backgrounds) == 10 and len(subsets) >= 12 and len(counts) >= 9
    assert diagonal >= 5 and asymmetric >= 5 and boundary >= 10 and no_distractor >= 10
    assert repeated_color >= 1
    assert _digest(generate_puzzle(0)) == _digest(generate_puzzle(0))
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(2026, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(0)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged answer accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), silhouettes=len(silhouettes),
                input_dimensions=len(dimensions), backgrounds=len(backgrounds),
                side_subsets=len(subsets), count_combinations=len(counts),
                diagonal_pieces=diagonal, asymmetric_pieces=asymmetric,
                width_30_outputs=boundary, pairs_without_distractors=no_distractor,
                repeated_distractor_colors=repeated_color)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
