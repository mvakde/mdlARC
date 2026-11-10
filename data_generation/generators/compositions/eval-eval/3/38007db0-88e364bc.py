"""Dislodged ink: redundancy separates a loose rigid fragment from same-color walls.

Only standard library and NumPy are used. No parent generator is imported.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _grid(a):
    a = np.asarray(a)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8)


def _connected(cells):
    cells = set(cells)
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in CARDINAL:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _decode_header(header, free):
    others = set(map(int, np.unique(header))) - {free}
    _require(len(others) == 2, "legend needs trail and tip colors")
    interpretations = []
    for tip in sorted(others):
        wall = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(header == tip)))
        trails = set(map(tuple, np.argwhere(header == wall)))
        if not 2 <= len(tips) <= 3:
            continue
        for dr, dc in DIRECTIONS:
            used, lengths = set(), []
            for r, c in sorted(tips):
                stroke = []
                p = r - dr, c - dc
                while p in trails:
                    stroke.append(p)
                    p = p[0] - dr, p[1] - dc
                if len(stroke) < 2 or used.intersection(stroke):
                    break
                used.update(stroke)
                lengths.append(len(stroke))
            if len(lengths) == len(tips) and used == trails:
                interpretations.append((wall, tip, (dr, dc)))
    _require(len(interpretations) == 1, "ambiguous legend")
    return interpretations[0]


def _read(a):
    a = _grid(a)
    lattice = int(a[0, 0])
    _require(np.all(a[0] == lattice) and np.all(a[-1] == lattice)
             and np.all(a[:, 0] == lattice) and np.all(a[:, -1] == lattice), "broken frame")
    rows = np.flatnonzero(np.all(a == lattice, axis=1)).tolist()
    _require(len(rows) >= 3 and rows[0] == 0 and rows[-1] == a.shape[0] - 1,
             "missing header or body")
    separator = rows[1]
    header = a[1:separator, 1:-1]
    free = int(header[0, 0])
    _require(free != lattice and np.all(header[[0, -1], :] == free)
             and np.all(header[:, [0, -1]] == free), "legend needs a free margin")
    wall, tip, direction = _decode_header(header, free)
    _require(len({lattice, free, wall, tip}) == 4, "roles must be distinct")
    body = a[separator:]
    rs = [r - separator for r in rows[1:]]
    cs = np.flatnonzero(np.all(body == lattice, axis=0)).tolist()
    _require(len(cs) >= 4 and cs[0] == 0 and cs[-1] == a.shape[1] - 1, "need three copies")
    _require(len(set(np.diff(rs))) == 1 and len(set(np.diff(cs))) == 1, "irregular lattice")
    ph, pw = rs[1] - 1, cs[1] - 1
    _require(ph >= 3 and pw >= 3, "panels too small")
    panels = [[body[r+1:s, c+1:d].copy() for c, d in zip(cs, cs[1:])]
              for r, s in zip(rs, rs[1:])]
    _require(all(set(map(int, np.unique(p))) <= {free, wall}
                 for row in panels for p in row), "unexplained panel color")
    counts = Counter(_key(p) for row in panels for p in row)
    top = counts.most_common()
    _require(len(top) > 1 and top[0][1] > top[1][1], "no unique template")
    master = next(p for row in panels for p in row if _key(p) == top[0][0])
    records = []
    for row in panels:
        odd = [i for i, p in enumerate(row) if not np.array_equal(p, master)]
        _require(len(odd) == 1, "each band needs exactly one visible exception")
        col = odd[0]
        _require(sum(np.array_equal(p, master) for p in row) == len(row) - 1,
                 "inconsistent template")
        variant = row[col]
        additions = set(map(tuple, np.argwhere((variant == wall) & (master == free))))
        removals = set(map(tuple, np.argwhere((variant == free) & (master == wall))))
        _require(_connected(additions), "added ink must be one connected fragment")
        stationary = variant == wall
        for p in additions:
            stationary[p] = False
        records.append(dict(column=col, variant=variant, additions=additions,
                            removals=removals, stationary=stationary))
    return dict(lattice=lattice, free=free, wall=wall, tip=tip, direction=direction,
                separator=separator, ph=ph, pw=pw, bands=len(panels), copies=len(panels[0]),
                master=master, panels=panels, records=records)


def _slide(stationary, fragment, direction, corner=True):
    """Rigid-body ray: all cells advance together; side walls block diagonals."""
    h, w = stationary.shape
    dr, dc = direction
    current = set(fragment)
    distance = 0

    def blocked(p):
        r, c = p
        return not (0 <= r < h and 0 <= c < w) or bool(stationary[r, c])

    while True:
        ahead = {(r + dr, c + dc) for r, c in current}
        if any(blocked(p) for p in ahead):
            return current, distance, False
        if corner and dr and dc and any(blocked((r + dr, c)) or blocked((r, c + dc))
                                       for r, c in current):
            return current, distance, True
        current = ahead
        distance += 1


def _assemble(rows, lattice):
    ph, pw = rows[0][0].shape
    out = np.full((len(rows)*(ph+1)+1, len(rows[0])*(pw+1)+1), lattice, np.int8)
    for b, row in enumerate(rows):
        for c, panel in enumerate(row):
            r0, c0 = 1+b*(ph+1), 1+c*(pw+1)
            out[r0:r0+ph, c0:c0+pw] = panel
    return out


def _predict(info, mode="correct"):
    selected = []
    for rec in info['records']:
        walls = rec['stationary']
        fragment = rec['additions']
        if mode == 'restore_walls':
            walls = info['master'] == info['wall']
        if mode == 'all_ink':
            walls = np.zeros(walls.shape, bool)
            fragment = set(map(tuple, np.argwhere(rec['variant'] == info['wall'])))
        direction = info['direction']
        if mode == 'reverse':
            direction = -direction[0], -direction[1]
        if mode == 'independent':
            end = set()
            for p in rec['additions']:
                end.update(_slide(walls, {p}, direction)[0])
        elif mode == 'copy':
            end = rec['additions']
        else:
            end = _slide(walls, fragment, direction, mode != 'no_corner')[0]
        base = walls if mode == 'all_ink' else rec['stationary']
        panel = np.where(base, info['wall'], info['free']).astype(np.int8)
        for p in end:
            panel[p] = info['wall']
        selected.append([panel])
    return _assemble(selected, info['lattice'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all roles, the reference walls, the fragment and the key from the input."""
    return _predict(_read(input_grid))


def _make_header(rng, width, direction, free, wall, tip):
    dr, dc = direction
    strokes = int(rng.choice([2, 2, 3]))
    length = int(rng.choice([2, 2, 3]))
    points = []
    for i in range(strokes):
        base = (0, 2*i) if dr else (2*i, 0)
        stagger = int(rng.integers(2))
        for j in range(length+1):
            points.append((base[0]+(j+stagger)*dr, base[1]+(j+stagger)*dc,
                           tip if j == length else wall))
    r0, c0 = min(p[0] for p in points), min(p[1] for p in points)
    gh = max(p[0] for p in points)-r0+1
    gw = max(p[1] for p in points)-c0+1
    height = gh+2+int(rng.integers(2))
    header = np.full((height, width-2), free, np.int8)
    ro = int(rng.integers(1, height-gh))
    co = int(rng.integers(1, width-2-gw))
    for r, c, color in points:
        header[r-r0+ro, c-c0+co] = color
    _require(_decode_header(header, free) == (wall, tip, direction), "bad constructed legend")
    return header


def _master(rng, h, w):
    mode = int(rng.integers(6))
    mask = np.zeros((h, w), bool)
    if mode == 0:
        mask = rng.random((h, w)) < float(rng.uniform(.14, .31))
    elif mode in (1, 4):
        for _ in range(1 if mode == 1 else int(rng.integers(2, 5))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for _ in range(int(rng.integers(3, h*w//3+2))):
                mask[r, c] = True
                dr, dc = CARDINAL[int(rng.integers(4))]
                r, c = int(np.clip(r+dr, 0, h-1)), int(np.clip(c+dc, 0, w-1))
    elif mode == 2:
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < .5:
                r = int(rng.integers(h))
                lo, hi = sorted(map(int, rng.choice(w+1, 2, replace=False)))
                mask[r, lo:hi] = True
            else:
                c = int(rng.integers(w))
                lo, hi = sorted(map(int, rng.choice(h+1, 2, replace=False)))
                mask[lo:hi, c] = True
    elif mode == 3:
        r0, r1 = sorted(map(int, rng.choice(h, 2, replace=False)))
        c0, c1 = sorted(map(int, rng.choice(w, 2, replace=False)))
        mask[r0:r1+1, c0] = mask[r0:r1+1, c1] = True
        mask[r0, c0:c1+1] = mask[r1, c0:c1+1] = True
    else:
        rr, cc = np.indices((h, w))
        slope = float(rng.choice([.5, 1., 1.5, -1.]))
        line = np.abs(rr-slope*cc-float(rng.integers(-w, h))) < .7
        mask = line & (rng.random((h, w)) < .85)
    if mask.sum() < 3 or mask.sum() > h*w*.58:
        mask = rng.random((h, w)) < .22
        mask.flat[:3] = True
    return mask


def _random_fragment(rng, master, size):
    h, w = master.shape
    free = np.argwhere(~master)
    start = tuple(map(int, free[int(rng.integers(len(free)))]))
    shape = {start}
    while len(shape) < size:
        candidates = sorted({(r+dr, c+dc) for r, c in shape for dr, dc in CARDINAL
                             if 0 <= r+dr < h and 0 <= c+dc < w
                             and not master[r+dr, c+dc]} - shape)
        if not candidates:
            return None
        shape.add(candidates[int(rng.integers(len(candidates)))])
    return shape


def _variant(rng, master, direction, sizes, requirement=None):
    """Carve optional gates along a proposed prefix, then test the actual ray."""
    h, w = master.shape
    ink = set(map(tuple, np.argwhere(master)))
    dr, dc = direction
    for _ in range(420):
        size = int(rng.choice(sizes))
        shape = _random_fragment(rng, master, size)
        if shape is None:
            continue
        steps = int(rng.integers(1, max(h, w)))
        swept = set()
        valid = True
        for k in range(1, steps+1):
            for r, c in shape:
                probes = [(r+k*dr, c+k*dc)]
                if dr and dc:
                    probes += [(r+k*dr, c+(k-1)*dc), (r+(k-1)*dr, c+k*dc)]
                if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in probes):
                    valid = False
                    break
                swept.update(probes)
            if not valid:
                break
        if not valid:
            continue
        required = swept & ink
        balanced = requirement == 'gate' or rng.random() < .5
        target = size if balanced else int(rng.choice([0, 0, 1, 2, 3, 4]))
        if len(required) > target or target > len(ink):
            continue
        removed = set(required)
        available = sorted(ink - removed)
        if target > len(removed):
            indices = rng.choice(len(available), target-len(removed), replace=False)
            removed.update(available[int(i)] for i in indices)
        walls = master.copy()
        for p in removed:
            walls[p] = False
        end, distance, corner = _slide(walls, shape, direction)
        if distance == 0:
            continue
        if requirement == 'gate':
            if not removed or _slide(master, shape, direction)[0] == end:
                continue
        if requirement == 'corner' and not corner:
            continue
        if requirement == 'rigid':
            independent = set().union(*(_slide(walls, {p}, direction)[0] for p in shape))
            if independent == end:
                continue
        return shape, removed, end
    return None


def _render_pair(rng, m, header, copies, variations, palette, varied_columns=False):
    lattice, free, wall, tip = palette
    width = copies*(m.shape[1]+1)+1
    master = np.where(m, wall, free).astype(np.int8)
    panel_rows, expected = [], []
    columns = list(map(int, rng.integers(copies, size=len(variations))))
    if varied_columns and len(set(columns)) == 1:
        slot = int(rng.integers(len(variations)))
        columns[slot] = (columns[slot]+int(rng.integers(1, copies))) % copies
    for band, (shape, removed, end) in enumerate(variations):
        variant = master.copy()
        for p in removed:
            variant[p] = free
        for p in shape:
            variant[p] = wall
        row = [master.copy() for _ in range(copies)]
        row[columns[band]] = variant
        panel_rows.append(row)
        result = master.copy()
        for p in removed:
            result[p] = free
        for p in end:
            result[p] = wall
        expected.append([result])
    if varied_columns and all(np.array_equal(p[0], q[0]) for p, q in zip(expected, expected[::-1])):
        return None
    body = _assemble(panel_rows, lattice)
    a = np.full((header.shape[0]+1+body.shape[0], width), lattice, np.int8)
    a[1:header.shape[0]+1, 1:-1] = header
    a[header.shape[0]+1:] = body
    return {'input': a, 'output': _assemble(expected, lattice)}


def _fallback_pair(rng, direction, kind, palette, novel_geometry=None, seen=None):
    """Finite constructive tail: guaranteed gates, corners, rigidity and fit.

    At least 54 geometry choices exist before excluding at most ten training
    geometries and thirteen prior inputs. No random fragment search is used.
    """
    lattice, free, wall, tip = palette
    canonical = (0, 1) if direction in CARDINAL else (1, 1)
    vector, turns = canonical, 0
    while vector != direction:
        vector = -vector[1], vector[0]
        turns += 1
    choices = [(h, w, b, c) for h in range(5, 10) for w in range(4, 9)
               for b in range(2 if kind == 'diagonal' else 1, (30-10)//(h+1)+1)
               for c in range(3, min(5, 29//(w+1))+1)
               if kind != 'test' or (h, w, b, c) not in (novel_geometry or set())]
    for i in rng.permutation(len(choices)):
        h, w, bands, copies = choices[int(i)]
        ch, cw = (w, h) if turns % 2 else (h, w)
        m = np.zeros((ch, cw), bool)
        if direction in CARDINAL:
            m[:, 2] = True
        else:
            m[1, 3] = True
        variations = []
        for b in range(bands):
            if direction in CARDINAL:
                shape = {(1, 0), (2, 0)}
                removed = {(1, 2), (2, 2)}
            else:
                shape = {(0, 0), (0, 1)} if b % 2 == 0 or kind == 'test' else {(0, 0)}
                removed = set()
            masks = []
            for cells in (shape, removed):
                mask = np.zeros((ch, cw), bool)
                for p in cells:
                    mask[p] = True
                masks.append(set(map(tuple, np.argwhere(np.rot90(mask, turns)))))
            shape, removed = masks
            rotated = np.rot90(m, turns).copy()
            walls = rotated.copy()
            for p in removed:
                walls[p] = False
            end, distance, _ = _slide(walls, shape, direction)
            _require(distance > 0, 'constructive fallback failed')
            variations.append((shape, removed, end))
        header = _make_header(rng, copies*(w+1)+1, direction, free, wall, tip)
        pair = _render_pair(rng, rotated, header, copies, variations, palette, kind == 'diagonal')
        if pair is not None and _key(pair['input']) not in (seen or set()):
            return pair
    raise AssertionError('finite fallback geometry space unexpectedly exhausted')


def _pair(rng, direction, kind, palette, novel_geometry=None):
    lattice, free, wall, tip = palette
    for _ in range(35):
        h = int(rng.integers(5, 10))
        w = int(rng.integers(4, 9))
        m = _master(rng, h, w)
        copies = int(rng.integers(3, min(5, 29//(w+1))+1))
        width = copies*(w+1)+1
        header = _make_header(rng, width, direction, free, wall, tip)
        max_bands = (30-header.shape[0]-2)//(h+1)
        if max_bands < (2 if kind == 'diagonal' else 1):
            continue
        bands = int(rng.integers(2 if kind == 'diagonal' else 1, min(4, max_bands)+1))
        sizes = [2, 3, 4, 5] if kind == 'test' else [1, 2, 3]
        requirements = [None]*bands
        if kind == 'gate':
            requirements[int(rng.integers(bands))] = 'gate'
        if kind == 'diagonal':
            slots = rng.choice(bands, 2, replace=False)
            requirements[int(slots[0])], requirements[int(slots[1])] = 'corner', 'rigid'
        variations = []
        for req in requirements:
            allowed = [2, 3] if req == 'rigid' else sizes
            v = _variant(rng, m, direction, allowed, req)
            if v is None:
                break
            variations.append(v)
        if len(variations) != bands:
            continue
        geometry = h, w, bands, copies
        if kind == 'test' and geometry in (novel_geometry or set()):
            continue
        pair = _render_pair(rng, m, header, copies, variations, palette, kind == 'diagonal')
        if pair is not None:
            return pair
    return _fallback_pair(rng, direction, kind, palette, novel_geometry)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} outside {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < .25)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    cardinal = CARDINAL[int(rng.integers(4))]
    diagonal = DIAGONAL[int(rng.integers(4))]
    kinds = ['gate', 'diagonal'] + ['ordinary']*(nt-2)
    kinds = [kinds[int(i)] for i in rng.permutation(nt)]
    puzzle, seen, training_geometry = {'train': [], 'test': []}, set(), set()
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        kind = kinds[index] if split == 'train' else 'test'
        direction = cardinal if kind == 'gate' else diagonal if kind == 'diagonal' else DIRECTIONS[int(rng.integers(8))]
        if not shared_palette:
            palette = tuple(map(int, rng.choice(10, 4, replace=False)))
        for _ in range(16):
            pair = _pair(rng, direction, kind, palette, training_geometry)
            if _key(pair['input']) not in seen:
                break
        else:
            pair = _fallback_pair(rng, direction, kind, palette, training_geometry, seen)
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
        if split == 'train':
            info = _read(pair['input'])
            training_geometry.add(tuple(info[k] for k in ('ph', 'pw', 'bands', 'copies')))
    return puzzle


def _measure(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    info = _read(a)
    _require(np.array_equal(_predict(info), b), 'incorrect answer')
    output_panels = []
    for index, rec in enumerate(info['records']):
        ph = info['ph']
        out = b[1+index*(ph+1):1+index*(ph+1)+ph, 1:-1]
        end, distance, corner = _slide(rec['stationary'], rec['additions'], info['direction'])
        _require(distance > 0, 'each fragment must move')
        _require(len(end) == len(rec['additions']) and _connected(end), 'fragment lost integrity')
        _require(int(np.count_nonzero(out == info['wall'])) ==
                 int(np.count_nonzero(rec['variant'] == info['wall'])), 'ink count changed')
        _require(np.all(out[rec['stationary']] == info['wall']), 'fixed walls changed')
        output_panels.append(out)
        rec.update(end=end, distance=distance, corner=corner)
    info['balanced'] = any(len(r['additions']) == len(r['removals']) for r in info['records'])
    info['order'] = [_key(p) for p in output_panels] != [_key(p) for p in output_panels[::-1]]
    info['nonrestoring'] = any(not np.array_equal(p, info['master']) for p in output_panels)
    for mode in ('restore_walls', 'independent', 'no_corner', 'copy', 'reverse', 'all_ink'):
        info[mode] = not np.array_equal(_predict(info, mode), b)
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair')
            a = _grid(pair['input'])
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            info = _measure(pair)
            if split == 'train':
                training.append(info)
    _require(any(t['balanced'] for t in training), 'need equal-count bitmap witness')
    _require(any(t['restore_walls'] for t in training), 'need a consequential missing wall')
    _require(any(t['independent'] for t in training), 'need rigid-body witness')
    _require(any(t['no_corner'] for t in training), 'need diagonal corner witness')
    _require(any(t['all_ink'] for t in training), 'need reference-based mobile/fixed separation')
    _require(any(t['order'] for t in training), 'need band-order witness')
    _require(any(t['nonrestoring'] for t in training), 'need nonrestoring motion')
    _require(any(len({r['column'] for r in t['records']}) > 1 for t in training),
             'need varying exception positions')
    _require(all(t['copy'] and t['reverse'] for t in training), 'need motion and polarity evidence')
    _require(any(0 in t['direction'] for t in training)
             and any(0 not in t['direction'] for t in training), 'need cardinal and diagonal evidence')


def self_test():
    """Bounded 42-seed diversity audit, contract checks and content mutations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = []
    signatures = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _measure(pair)
                m['split'] = split
                features.append(m)
                local.append(m)
        _require(len({_key(m['master']) for m in local}) >= 2, 'local template variation')
        train_geometries = {tuple(m[k] for k in ('ph', 'pw', 'bands', 'copies'))
                            for m in local if m['split'] == 'train'}
        _require(all(tuple(m[k] for k in ('ph', 'pw', 'bands', 'copies')) not in train_geometries
                     for m in local if m['split'] == 'test'), 'missing geometry shift')
    _require(len(signatures) == 42, 'seed diversity')
    _require({m['direction'] for m in features} == set(DIRECTIONS), 'direction coverage')
    _require(len({(m['ph'], m['pw'], m['bands'], m['copies']) for m in features}) >= 30,
             'geometry coverage')
    _require(len({_key(m['master'] == m['wall']) for m in features}) >= 100, 'wall diversity')
    _require({m['lattice'] for m in features} == set(range(10)), 'palette coverage')
    _require({len(r['additions']) for m in features for r in m['records']} >= {1, 2, 3, 4, 5},
             'fragment-size coverage')
    _require(any(len(r['removals']) == 0 for m in features for r in m['records']), 'pure additions')
    _require(any(m['ph'] != m['pw'] for m in features), 'rectangular panels')
    _require(any(len(set(r['column'] for r in m['records'])) < m['bands'] for m in features
                 if m['bands'] > 1), 'repeated exception columns')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+10*nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Exercise the constructive tail directly, including every rotation.
    fallback_rng = np.random.default_rng(931)
    for i in range(4):
        fallback = {'train': [
            _fallback_pair(fallback_rng, CARDINAL[i], 'gate', (0, 1, 2, 3)),
            _fallback_pair(fallback_rng, DIAGONAL[i], 'diagonal', (4, 5, 6, 7))],
            'test': [_fallback_pair(fallback_rng, DIRECTIONS[2*i], 'test', (8, 9, 0, 1))]}
        validate_generated_puzzle(fallback)
    first = generate_puzzle(73, 3, 2)
    second = generate_puzzle(73, 3, 2)
    _require(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
                 for k in a), 'nondeterminism')
    for defect in ('answer', 'missing_exception', 'legend'):
        bad = {s: [{k: np.array(v).copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in puzzles[0]}
        pair = bad['train'][0]
        info = _read(pair['input'])
        if defect == 'answer':
            pair['output'][1, 1] = (int(pair['output'][1, 1])+1) % 10
        elif defect == 'legend':
            pair['input'][pair['input'] == info['tip']] = info['wall']
        else:
            c0 = 1+info['records'][0]['column']*(info['pw']+1)
            r0 = info['separator']+1
            pair['input'][r0:r0+info['ph'], c0:c0+info['pw']] = info['master']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted '+defect)
    return {'seeds': 42, 'pairs': len(features), 'geometries': len({
        (m['ph'], m['pw'], m['bands'], m['copies']) for m in features}),
        'templates': len({_key(m['master'] == m['wall']) for m in features}),
        'directions': len({m['direction'] for m in features})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
