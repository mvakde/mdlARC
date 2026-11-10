"""Clocked docking: a global period-six ruler gates key-directed wall rays.

The solver reads every role from the input. No parent module, example, random
state, or puzzle-level hidden color convention participates in prediction.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack, cells = [root], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in CARDINAL:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _edge_bar(a, bg):
    """Only the ruler touches the outside boundary; corners are background."""
    boundary = np.zeros(a.shape, bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    cells = np.argwhere(boundary & (a != bg))
    _require(len(cells) > 0, "no edge ruler")
    rows, cols = cells.T
    candidates = []
    for side, fixed, edge, lateral in ((0, rows, 0, cols),
                                      (1, cols, a.shape[1] - 1, rows),
                                      (2, rows, a.shape[0] - 1, cols),
                                      (3, cols, 0, rows)):
        if np.all(fixed == edge):
            start, length = int(lateral.min()), len(cells)
            span = a.shape[1] if side % 2 == 0 else a.shape[0]
            if (start > 0 and start + length < span and
                    np.array_equal(np.sort(lateral), np.arange(start, start + length))):
                candidates.append((side, start, length))
    _require(len(candidates) == 1, "ambiguous ruler edge")
    colors = set(map(int, a[rows, cols]))
    _require(len(colors) == 1, "ruler must be solid")
    seed = next(iter(colors))
    _require(int(np.count_nonzero(a == seed)) == len(cells), "ruler color reused")
    return dict(side=candidates[0][0], start=candidates[0][1],
                length=candidates[0][2], seed=seed)


def _depth(p, shape, side):
    r, c = p
    return (r, shape[1] - 1 - c, shape[0] - 1 - r, c)[side]


def _lateral(p, side):
    return p[1] if side % 2 == 0 else p[0]


def _keys(a, bg, seed):
    result = []
    for cells in _components((a != bg) & (a != seed)):
        r0, r1, c0, c1 = _bbox(cells)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        carrier = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        others = set(map(int, np.unique(sub))) - {carrier}
        if np.any(border != carrier) or len(others) != 2:
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip).tolist()))
            trails = set(map(tuple, np.argwhere(sub == trail).tolist()))
            if len(tips) < 2:
                continue
            for dr, dc in DIRECTIONS:
                used, lengths = set(), []
                for r, c in sorted(tips):
                    stroke = []
                    r, c = r - dr, c - dc
                    while (r, c) in trails:
                        stroke.append((r, c))
                        r, c = r - dr, c - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((trail, tip, (dr, dc), tuple(lengths)))
        _require(len(interpretations) == 1, "key must have one geometric interpretation")
        trail, tip, direction, lengths = interpretations[0]
        result.append(dict(box=(r0, r1, c0, c1), carrier=carrier,
                           marker=trail, highlight=tip, direction=direction,
                           lengths=lengths))
    _require(1 <= len(result) <= 3, "need one to three keys")
    _require(len({k['carrier'] for k in result}) == len(result), "duplicate carriers")
    _require(len({k['marker'] for k in result}) == 1 and
             len({k['highlight'] for k in result}) == 1, "keys disagree on roles")
    return result


def _read(a):
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size / 2, "background must occupy a strict majority")
    bar = _edge_bar(a, bg)
    keys = _keys(a, bg, bar['seed'])
    marker, highlight = keys[0]['marker'], keys[0]['highlight']
    carriers = {k['carrier'] for k in keys}
    _require(len({bg, bar['seed'], marker, highlight}) == 4 and
             not carriers.intersection((bg, bar['seed'], marker, highlight)), "role collision")
    key_mask = np.zeros(a.shape, bool)
    for k in keys:
        r0, r1, c0, c1 = k['box']
        key_mask[r0:r1 + 1, c0:c1 + 1] = True
    walls = (a != bg) & (a != marker) & (a != bar['seed']) & ~key_mask
    _require(set(map(int, np.unique(a[walls]))) == carriers, "unexplained or missing walls")
    regions = []
    for cells in _components(((a == bg) | (a == marker)) & ~key_mask):
        boundary, edge = set(), False
        for r, c in cells:
            for dr, dc in CARDINAL:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]):
                    edge = True
                elif (rr, cc) not in cells:
                    boundary.add(int(a[rr, cc]))
        starts = sorted(p for p in cells if a[p] == marker)
        enclosed = not edge and len(boundary) == 1 and boundary <= carriers
        _require(not starts or enclosed, "unassociated mobile marker")
        if enclosed:
            regions.append(dict(cells=cells, carrier=next(iter(boundary)), starts=starts))
    _require(all(any(r['starts'] and r['carrier'] == k['carrier'] for r in regions)
                 for k in keys), "each system needs a marker")
    return dict(bg=bg, marker=marker, highlight=highlight, keys=keys,
                key_mask=key_mask, walls=walls, regions=regions, **bar)


def _ray(free, start, direction, corner=True):
    path = [start]
    r, c = start
    dr, dc = direction
    while (r + dr, c + dc) in free:
        if corner and dr and dc and ((r + dr, c) not in free or (r, c + dc) not in free):
            break
        r, c = r + dr, c + dc
        path.append((r, c))
    return path


def _dock(path, shape, side, relative=False, first=False):
    stations = [i for i, p in enumerate(path)
                if (i if relative else _depth(p, shape, side)) % 6 == 4]
    index = (stations[0] if first else stations[-1]) if stations else 0
    return path[index], stations


def _motions(a, info, variant=None):
    directions = {k['carrier']: k['direction'] for k in info['keys']}
    result, used = [], set()
    for region in info['regions']:
        direction = directions[region['carrier']]
        if variant == 'reverse':
            direction = -direction[0], -direction[1]
        elif variant == 'inward':
            direction = ((1, 0), (0, -1), (-1, 0), (0, 1))[info['side']]
        for start in region['starts']:
            path = _ray(region['cells'], start, direction, corner=variant != 'no_corner')
            dock, stations = _dock(path, a.shape, info['side'],
                                   relative=variant == 'relative', first=variant == 'first')
            if variant == 'wall_slide':
                dock = path[-1]
            if variant is None:
                _require(used.isdisjoint(path), "interacting full rays")
                used.update(path)
            result.append(dict(start=start, path=path, end=dock, stations=stations,
                               carrier=region['carrier'], direction=direction, free=region['cells']))
    return result


def _paint(a, info, motions):
    out = a.copy()
    mobile = (a == info['marker']) & ~info['key_mask']
    out[mobile] = info['bg']
    # Ruler phase is global. Walls and keys mask paint, but never reset it.
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = r, c
            lateral = _lateral(p, info['side'])
            if (info['start'] <= lateral < info['start'] + info['length'] and
                    not info['key_mask'][p] and a[p] in (info['bg'], info['marker'], info['seed'])):
                phase = _depth(p, a.shape, info['side']) % 6
                out[p] = (info['bg'] if phase % 2 else
                          info['highlight'] if phase == 4 else info['seed'])
    for m in motions:
        out[m['end']] = info['marker']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the input's ruler, geometric keys, enclosures, and landing planes."""
    a = _grid(input_grid)
    info = _read(a)
    return _paint(a, info, _motions(a, info))


def _make_key(rng, direction, carrier, marker, highlight, compact):
    dr, dc = direction
    count = 2 if compact else int(rng.choice([2, 2, 3]))
    length = 2 if compact else int(rng.choice([2, 2, 3]))
    points, offset = [], 0
    for _ in range(count):
        stagger = 0 if compact else int(rng.integers(2))
        base = (0, offset) if dr else (offset, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc,
                           highlight if j == length else marker))
        offset += 2 if compact else int(rng.choice([2, 3]))
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in points])
    pads = [1] * 4 if compact else list(map(int, rng.choice([1, 1, 2], 4)))
    a = np.full((r1 - r0 + 1 + pads[0] + pads[1],
                 c1 - c0 + 1 + pads[2] + pads[3]), carrier, np.int8)
    for r, c, color in points:
        a[r - r0 + pads[0], c - c0 + pads[2]] = color
    return a


def _enclosure(rng, h, w, kind, baffles):
    footprint = np.ones((h, w), bool)
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, max(2, h // 3))), int(rng.integers(1, max(2, w // 3)))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        yy, xx = np.indices((h, w))
        cy, cx = (h - 1) / 2, (w - 1) / 2
        power = float(rng.uniform(1.2, 2.6))
        footprint = abs((yy - cy) / (cy + .3)) ** power + abs((xx - cx) / (cx + .3)) ** power <= 1.15
    elif kind == 3:
        left = right = 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, max(1, w // 3)))
            right = int(np.clip(right + rng.integers(-1, 2), 0, max(1, w // 3)))
            footprint[r, :left] = False
            if right:
                footprint[r, w - right:] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dr, dc in CARDINAL:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    walls = footprint & ~interior
    for _ in range(baffles):
        horizontal = bool(rng.integers(2))
        full = bool(rng.random() < .17)
        if horizontal:
            row = int(rng.integers(2, h - 2))
            cut = int(rng.integers(2, w - 2))
            cs = slice(None) if full else (slice(0, cut) if rng.random() < .5 else slice(cut, w))
            walls[row, cs] |= footprint[row, cs]
        else:
            col = int(rng.integers(2, w - 2))
            cut = int(rng.integers(2, h - 2))
            rs = slice(None) if full else (slice(0, cut) if rng.random() < .5 else slice(cut, h))
            walls[rs, col] |= footprint[rs, col]
    return footprint, walls


def _place(rng, shape, objects):
    occupied = np.zeros(shape, bool)
    placed = []
    for obj in objects:
        mask = obj['mask']
        h, w = mask.shape
        padded = np.pad(occupied, 1)
        blocked = occupied.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked |= padded[1 + dr:1 + dr + shape[0], 1 + dc:1 + dc + shape[1]]
        choices = []
        for r in range(1, shape[0] - h):
            for c in range(1, shape[1] - w):
                if not np.any(blocked[r:r + h, c:c + w] & mask):
                    choices.append((r, c))
        if not choices:
            return None
        r, c = choices[int(rng.integers(len(choices)))]
        occupied[r:r + h, c:c + w] |= mask
        placed.append((obj, r, c))
    return placed


def _candidate(free, start, direction, shape, bar):
    path = _ray(free, start, direction)
    end, stations = _dock(path, shape, 0)
    first, _ = _dock(path, shape, 0, first=True)
    relative, _ = _dock(path, shape, 0, relative=True)
    loose, _ = _dock(_ray(free, start, direction, corner=False), shape, 0)
    return dict(start=start, path=path, end=end, stations=stations,
                moving=end != start, multi=end != first,
                wall_diff=end != path[-1], relative_diff=end != relative,
                corner_diff=end != loose,
                remote=all(not bar[0] - 1 <= c <= bar[0] + bar[1] for _, c in path))


def _select_markers(rng, regions, direction, shape, bar, teach, desired):
    candidates = [_candidate(free, start, direction, shape, bar)
                  for free in regions for start in sorted(free)]
    needs = []
    if teach == 'radial':
        needs = [lambda x: x['moving'] and x['multi'] and x['wall_diff'] and
                 x['remote'] and x['relative_diff']]
    elif teach == 'parallel':
        needs = [lambda x: x['moving'], lambda x: not x['stations']]
    elif teach == 'corner':
        needs = [lambda x: x['moving'] and x['corner_diff']]
    else:
        needs = [lambda x: x['moving']]
    selected, used = [], set()
    for predicate in needs:
        choices = [x for x in candidates if predicate(x) and used.isdisjoint(x['path'])]
        if not choices:
            return None
        x = choices[int(rng.integers(len(choices)))]
        selected.append(x)
        used.update(x['path'])
    for index in rng.permutation(len(candidates)):
        if len(selected) >= desired:
            break
        x = candidates[int(index)]
        if used.isdisjoint(x['path']):
            selected.append(x)
            used.update(x['path'])
    return selected


def _orient(a, side):
    return np.rot90(a, -side).copy()


def _pair(rng, palette, side, lesson, test=False):
    bg, seed, marker, highlight = palette[:4]
    # Natural proposals are tested for the required interactions, rather than
    # receiving phase-aligned witness stamps.
    for attempt in range(240):
        if lesson == 'cardinals':
            n, shape = 2, (int(rng.integers(28, 31)), int(rng.integers(28, 31)))
            directions = [(int(rng.choice([-1, 1])), 0), (0, int(rng.choice([-1, 1])))]
            lessons = ['radial', 'parallel']
        elif lesson == 'corner':
            n, shape = 1, (int(rng.integers(25, 31)), int(rng.integers(25, 31)))
            directions = [(int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))]
            lessons = ['corner']
        else:
            n = int(rng.choice([1, 1, 1, 2, 2, 3]))
            low = 18 if n == 1 else (25 if n == 2 else 28)
            shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
            directions = [DIRECTIONS[int(rng.integers(8))] for _ in range(n)]
            lessons = ['normal'] * n
        carriers = palette[4:4 + n]
        objects = []
        for i in range(n):
            compact = n > 1 or attempt > 160
            key = _make_key(rng, directions[i], carriers[i], marker, highlight, compact)
            if lesson == 'cardinals':
                h, w = int(rng.integers(17, 22)), int(rng.integers(8, 11))
                baffles = int(rng.integers(2))
            elif lesson == 'corner':
                h, w = int(rng.integers(16, 22)), int(rng.integers(14, 21))
                baffles = int(rng.integers(1, 4))
            else:
                high = 20 if n == 1 else (14 if n == 2 else 10)
                h, w = int(rng.integers(8, high + 1)), int(rng.integers(8, high + 1))
                baffles = int(rng.integers(0, 4 if test else 3))
            footprint, walls = _enclosure(rng, h, w, int(rng.integers(4)), baffles)
            objects.append(dict(mask=footprint, walls=walls, index=i, key=False))
            objects.append(dict(mask=np.ones(key.shape, bool), grid=key, index=i, key=True))
        # Independent enclosure/key order and unrestricted fitting positions.
        enclosures = [o for o in objects if not o['key']]
        keys = [o for o in objects if o['key']]
        order = [enclosures[int(i)] for i in rng.permutation(n)] + [keys[int(i)] for i in rng.permutation(n)]
        placed = _place(rng, shape, order)
        if placed is None:
            continue
        if lesson == 'cardinals':
            # The ruler meets one naturally sampled wall while a remote system
            # supplies the cross-sectional witness. No object is stamped into
            # a prescribed period phase.
            obj, _, col = next((o, r, c) for o, r, c in placed
                               if not o['key'] and o['index'] == 1)
            witness_col = col + int(rng.choice(np.argwhere(obj['walls'])[:, 1]))
            length = int(rng.integers(1, 9))
            start = int(rng.integers(max(1, witness_col - length + 1),
                                     min(witness_col, shape[1] - length - 1) + 1))
        else:
            start = int(rng.integers(1, shape[1] - 2))
            length = int(rng.integers(1, min(shape[1] - start, 9)))
        a = np.full(shape, bg, np.int8)
        a[0, start:start + length] = seed
        selected = []
        failed = False
        for obj, r, c in placed:
            h, w = obj['mask'].shape
            i = obj['index']
            if obj['key']:
                a[r:r + h, c:c + w] = obj['grid']
                continue
            a[r:r + h, c:c + w][obj['walls']] = carriers[i]
            local = _components(obj['mask'] & ~obj['walls'])
            regions = [{(rr + r, cc + c) for rr, cc in cells} for cells in local]
            desired = int(rng.integers(2 if lessons[i] == 'parallel' else 1, 6 if test else 5))
            markers = _select_markers(rng, regions, directions[i], shape,
                                      (start, length), lessons[i], desired)
            if markers is None:
                failed = True
                break
            selected += markers
        if failed:
            continue
        for m in selected:
            a[m['start']] = marker
        a = _orient(a, side)
        try:
            info = _read(a)
            out = _paint(a, info, _motions(a, info))
        except AssertionError:
            continue
        if not np.array_equal(a, out):
            return dict(input=a, output=out)
    raise RuntimeError('bounded natural scene construction exhausted')


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(10)))
    base = int(rng.integers(4))
    shift = int(rng.choice([1, 2, 3]))
    puzzle, seen = {'train': [], 'test': []}, set()
    train_lessons = ['cardinals', 'corner'] + ['normal'] * (num_train - 2)
    train_lessons = [train_lessons[int(i)] for i in rng.permutation(num_train)]
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            side = base if split == 'train' else (base + shift) % 4
            lesson = train_lessons[i] if split == 'train' else 'normal'
            for _ in range(12):
                # Conventions remain shared within a puzzle. Geometry, side,
                # system carriers, and every local key vary independently.
                mobile, tip = palette[2:4]
                local_colors = list(map(int, rng.permutation([c for c in range(10)
                                                               if c not in (mobile, tip)])))
                local_palette = local_colors[:2] + [mobile, tip] + local_colors[2:]
                pair = _pair(rng, local_palette, side, lesson, test=split == 'test')
                fp = _fingerprint(pair['input'])
                if fp not in seen:
                    seen.add(fp)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a, info, motions):
    ends = [m['end'] for m in motions]
    alternatives = {}
    for variant in ('wall_slide', 'inward', 'reverse', 'first', 'relative', 'no_corner'):
        alternatives[variant] = ends != [m['end'] for m in _motions(a, info, variant)]
    projected = np.zeros(a.shape, bool)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            projected[r, c] = info['start'] <= _lateral((r, c), info['side']) < info['start'] + info['length']
    return dict(**alternatives,
                masked_track=bool(np.any(projected & (info['walls'] | info['key_mask']))),
                remote=any(m['end'] != m['start'] and all(
                    not info['start'] - 1 <= _lateral(p, info['side']) <= info['start'] + info['length']
                    for p in m['path']) for m in motions),
                parallel_move=any(_depth(m['start'], a.shape, info['side']) ==
                                  _depth(m['path'][-1], a.shape, info['side']) and
                                  m['end'] != m['start'] for m in motions),
                no_station=any(not m['stations'] for m in motions))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, conventions, features = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "shape changed")
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _read(a)
            motions = _motions(a, info)
            prediction = _paint(a, info, motions)
            _require(np.array_equal(b, prediction), "wrong output")
            _require(not np.array_equal(a, b), "entirely unchanged pair")
            conventions.add((info['marker'], info['highlight']))
            immutable = info['walls'] | info['key_mask'] | (a == info['seed'])
            _require(np.array_equal(a[immutable], b[immutable]), "wall, key, or seed altered")
            _require(int(np.count_nonzero((a == info['marker']) & ~info['key_mask'])) ==
                     int(np.count_nonzero((b == info['marker']) & ~info['key_mask'])), "marker count changed")
            for region in info['regions']:
                before = len(region['starts'])
                after = sum(b[p] == info['marker'] for p in region['cells'])
                _require(before == after, "marker changed enclosing region")
            for m in motions:
                _require(m['end'] in m['path'], "marker left its key-directed ray")
                if m['stations']:
                    _require(m['end'] == m['path'][max(m['stations'])], "not the last station")
                else:
                    _require(m['end'] == m['start'], "unstationed marker moved")
            if split == 'train':
                features.append(_features(a, info, motions))
    _require(len(conventions) == 1, "puzzle color conventions changed")
    for name in ('wall_slide', 'inward', 'reverse', 'first', 'relative', 'no_corner',
                 'remote', 'parallel_move', 'no_station', 'masked_track'):
        _require(any(f[name] for f in features), f'missing training distinction: {name}')


def _encode(p):
    return json.dumps({s: [{k: np.asarray(v).tolist() for k, v in pair.items()}
                          for pair in p[s]] for s in ('train', 'test')}, separators=(',', ':'))


def self_test():
    """Bounded sampling, family ablations, metamorphic checks, and corruption tests."""
    signatures, shapes, structures = set(), set(), set()
    sides, directions, counts, lengths, palettes, distances = set(), set(), set(), set(), set(), set()
    irregular = internal = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_encode(p))
        if seed in (0, 17, 41):
            _require(_encode(p) == _encode(generate_puzzle(seed)), "nondeterminism")
        for split in ('train', 'test'):
            for pair in p[split]:
                a = pair['input']
                info = _read(a)
                ms = _motions(a, info)
                shapes.add(a.shape)
                sides.add(info['side'])
                counts.add(len(info['keys']))
                lengths.add(info['length'])
                palettes.add((info['bg'], info['seed'], info['marker'], info['highlight']))
                directions.update(k['direction'] for k in info['keys'])
                distances.update(m['path'].index(m['end']) for m in ms)
                roles = {}
                normal = bytes(roles.setdefault(int(c), len(roles)) for c in a.ravel())
                structures.add((a.shape, hashlib.sha256(normal).hexdigest()))
                for k in info['keys']:
                    mask = info['walls'] & (a == k['carrier'])
                    r0, r1, c0, c1 = _bbox(np.argwhere(mask))
                    sub = mask[r0:r1 + 1, c0:c1 + 1]
                    irregular += not (sub[0].all() and sub[-1].all() and sub[:, 0].all() and sub[:, -1].all())
                    free = set().union(*(region['cells'] for region in info['regions']
                                         if region['carrier'] == k['carrier']))
                    internal += any(((r - 1, c) in free and (r + 1, c) in free) or
                                    ((r, c - 1) in free and (r, c + 1) in free)
                                    for r, c in map(tuple, np.argwhere(mask)))
        # Rotation/reflection and literal palette changes must transform outputs
        # covariantly; this checks input-only interpretation outside sampled sides.
        a, b = p['test'][0]['input'], p['test'][0]['output']
        _require(np.array_equal(solve(np.rot90(a)), np.rot90(b)), "rotation failure")
        _require(np.array_equal(solve(np.fliplr(a)), np.fliplr(b)), "reflection failure")
        perm = np.random.default_rng(seed + 9000).permutation(10)
        _require(np.array_equal(solve(perm[a]), perm[b]), "palette equivariance failure")
    _require(len(signatures) == 42 and len(structures) >= 200, "structural diversity")
    _require(len(shapes) >= 35 and sides == {0, 1, 2, 3}, "size/edge diversity")
    _require(directions == set(DIRECTIONS) and counts == {1, 2, 3}, "direction/count diversity")
    _require(len(lengths) >= 6 and len(palettes) >= 35, "span/palette diversity")
    _require(distances >= {0, 1, 2, 3, 4, 5, 6} and irregular >= 70 and internal >= 20,
             "motion/geometry diversity")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                color_normalized_inputs=len(structures), irregular_systems=int(irregular),
                internal_stop_systems=int(internal), directions=len(directions), system_counts=sorted(counts))
