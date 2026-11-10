"""Interlocking cargo: match a blueprint, then dock its inferred rigid pieces.

Only standard Python and NumPy are used. Zero is the background. Opposite
colored rails visibly specify the destination of matching tile frames.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors must be 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(mask):
    seen = set()
    result = []
    for r, c in zip(*np.nonzero(mask)):
        start = int(r), int(c)
        if start in seen:
            continue
        seen.add(start)
        todo, cells = [start], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1]
                        and mask[p] and p not in seen):
                    seen.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _rail_candidates(g):
    """Canonical orientation has two vertical rails at the outer columns."""
    found = []
    for transpose in (False, True):
        a = g.T if transpose else g
        for lo in range(a.shape[0]):
            if a[lo, 0] == 0 or a[lo, -1] in (0, a[lo, 0]):
                continue
            hi = lo + 1
            while (hi < a.shape[0] and a[hi, 0] == a[lo, 0]
                   and a[hi, -1] == a[lo, -1]):
                hi += 1
            if hi-lo >= 10 and (lo == 0 or hi == a.shape[0]):
                if lo > 0 and np.any(a[:lo, (0, -1)]):
                    continue
                if hi < a.shape[0] and np.any(a[hi:, (0, -1)]):
                    continue
                found.append((transpose, lo, hi))
    return found


def _segments(plan):
    sequences = defaultdict(list)
    h, w = plan.shape
    for length in range(2, w+1):
        for r in range(h):
            for c in range(w-length+1):
                seq = tuple(int(v) for v in plan[r, c:c+length])
                if 0 not in seq:
                    sequences[seq].append((r, c))
    return [(positions[0], len(seq), seq) for seq, positions in sequences.items()
            if len(positions) == 1]


def _tile_cells(plan, side, stride, sites=None):
    sites = list(zip(*np.nonzero(plan))) if sites is None else sites
    return frozenset((int(r)*stride+dy, int(c)*stride+dx)
                     for r, c in sites for dy in range(side) for dx in range(side))


def _absolute(piece, mask=None, x=None):
    mask = piece['mask'] if mask is None else mask
    return {(piece['y']+r, (piece['x'] if x is None else x)+c) for r, c in mask}


def _slide(pieces, height, width, masks=None):
    """Monotone unit motion: existing same-class pieces cannot be jumped.

    All same-class pieces participate from their starting positions. Repeated
    sweeps also handle mutually interlocked pieces and moving blockers.
    Opposite classes are transparent during transit, as in parent A.
    """
    q = [dict(p, mask=(p['mask'] if masks is None else masks[i]))
         for i, p in enumerate(pieces)]
    for direction in (-1, 1):
        group = [i for i, p in enumerate(q) if p['d'] == direction]
        group.sort(key=lambda i: (direction * -q[i]['x'], q[i]['y']))
        occupied = set()
        for i in group:
            cells = _absolute(q[i])
            _require(not occupied.intersection(cells), 'initial same-class overlap')
            occupied.update(cells)
        for _ in range(width * max(1, len(group)) + 1):
            moved = False
            for i in group:
                p = q[i]
                occupied.difference_update(_absolute(p))
                nxt = _absolute(p, x=p['x']+direction)
                if (all(0 <= y < height and 0 <= x < width for y, x in nxt)
                        and not occupied.intersection(nxt)):
                    p['x'] += direction
                    moved = True
                occupied.update(_absolute(p))
            if not moved:
                break
        else:
            raise ValueError('packing did not converge')
    return q


def _draw_piece(arena, plan, side, stride, piece, seed_only=False):
    sites = piece['sites'] if seed_only else list(zip(*np.nonzero(plan)))
    for r, c in sites:
        y, x = piece['y']+int(r)*stride, piece['x']+int(c)*stride
        _require(0 <= y and y+side <= arena.shape[0]
                 and 0 <= x and x+side <= arena.shape[1], 'piece outside arena')
        _require(not np.any(arena[y:y+side, x:x+side]), 'pieces overlap')
        arena[y:y+side, x:x+side] = piece['frame']
        arena[y+1:y+side-1, x+1:x+side-1] = plan[r, c]


def _decode(input_grid):
    g = _grid(input_grid)
    rails = _rail_candidates(g)
    _require(len(rails) == 1, 'exactly one pair of opposite rails required')
    transpose, lo, hi = rails[0]
    a = g.T.copy() if transpose else g.copy()
    low, high = int(a[lo, 0]), int(a[lo, -1])
    outside = a.copy()
    outside[lo:hi] = 0
    points = np.argwhere(outside != 0)
    _require(len(points) >= 3, 'miniature missing')
    mr, mc = points.min(axis=0)
    er, ec = points.max(axis=0)+1
    _require(np.all((points-(mr, mc)) % 2 == 0), 'miniature pitch must be two')
    _require(0 < mc and ec < a.shape[1], 'miniature must avoid outer columns')
    plan = a[mr:er:2, mc:ec:2].copy()
    _require(min(plan.shape) >= 2, 'miniature must be two dimensional')
    _require(all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1)),
             'visible adjacent miniature rows and columns required')
    _require(not ({low, high} & set(int(v) for v in plan.flat)), 'rail and payload roles differ')
    _require(er <= lo-1 or mr >= hi+1, 'blank separator required')
    arena = a[lo:hi, 1:-1]
    tiles = []
    for cells in _components(arena != 0):
        p = np.asarray(cells)
        y, x = p.min(axis=0)
        ey, ex = p.max(axis=0)+1
        s = int(ey-y)
        _require(s >= 3 and ex-x == s and len(cells) == s*s, 'seed tile geometry')
        patch = arena[y:ey, x:ex]
        frame, color = int(patch[0, 0]), int(patch[1, 1])
        _require(frame in (low, high) and color not in (0, low, high), 'tile roles')
        _require(all(np.all(edge == frame) for edge in
                     (patch[0], patch[-1], patch[:, 0], patch[:, -1])), 'complete tile frame')
        _require(np.all(patch[1:-1, 1:-1] == color), 'solid tile interior')
        tiles.append((int(y), int(x), s, frame, color))
    _require(len(tiles) >= 4, 'at least two seed runs required')
    _require(len({t[2] for t in tiles}) == 1, 'common tile side required')
    side = tiles[0][2]
    rows = defaultdict(list)
    for t in tiles:
        rows[t[0]].append(t)
    pieces, strides, matches = [], set(), []
    mask = None
    for y, row in sorted(rows.items()):
        row.sort(key=lambda t: t[1])
        _require(len(row) >= 2 and len({t[3] for t in row}) == 1, 'one seed run per row')
        differences = np.diff([t[1] for t in row])
        stride = int(differences[0])
        _require(stride > side and np.all(differences == stride), 'regular seed spacing')
        strides.add(stride)
        seq = tuple(t[4] for t in row)
        found = [(m, length) for m, length, colors in _segments(plan) if colors == seq]
        _require(len(found) == 1, 'seed sequence must have exactly one blueprint match')
        (r, c), length = found[0]
        by, bx = y-r*stride, row[0][1]-c*stride
        if mask is None:
            mask = _tile_cells(plan, side, stride)
        sites = [(r, c+k) for k in range(length)]
        pieces.append(dict(y=by, x=bx, frame=row[0][3],
                           d=-1 if row[0][3] == low else 1,
                           mask=mask, sites=sites, match=(r, c)))
        matches.append((r, c, length))
    _require(len(strides) == 1, 'one common lattice stride required')
    stride = next(iter(strides))
    occupied = set()
    for p in pieces:
        cells = _absolute(p)
        _require(all(0 <= y < arena.shape[0] and 0 <= x < arena.shape[1] for y, x in cells),
                 'expanded starting piece outside arena')
        _require(not occupied.intersection(cells), 'expanded starting pieces overlap')
        occupied.update(cells)
        _require(len(p['sites']) < np.count_nonzero(plan), 'each seed must omit a tile')
    return dict(grid=a, transpose=transpose, lo=lo, hi=hi, low=low, high=high,
                plan=plan, side=side, stride=stride, pieces=pieces,
                arena_shape=arena.shape, matches=matches)


def _output(meta, settled):
    a = meta['grid'].copy()
    arena = np.zeros(meta['arena_shape'], dtype=np.int8)
    for p in settled:
        _draw_piece(arena, meta['plan'], meta['side'], meta['stride'], p)
    a[meta['lo']:meta['hi'], 1:-1] = arena
    return a.T.copy() if meta['transpose'] else a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover every convention and the answer from this input alone."""
    m = _decode(input_grid)
    return _output(m, _slide(m['pieces'], *m['arena_shape']))


def _reference_positions(pieces, height, width):
    """Check packing using first-contact distances, rather than unit steps."""
    q = [dict(p) for p in pieces]
    for d in (-1, 1):
        ids = [i for i, p in enumerate(q) if p['d'] == d]
        ids.sort(key=lambda i: (-d*q[i]['x'], q[i]['y']))
        for _ in range(width * max(1, len(ids)) + 1):
            moved = False
            for i in ids:
                own = _absolute(q[i])
                others = set().union(*(_absolute(q[j]) for j in ids if j != i))
                boundary = min(x for y, x in own) if d < 0 else width-1-max(x for y, x in own)
                distance = boundary
                by_row = defaultdict(list)
                for y, x in others:
                    by_row[y].append(x)
                for y, x in own:
                    for other_x in by_row[y]:
                        gap = (other_x-x)*d
                        if gap > 0:
                            distance = min(distance, gap-1)
                if distance:
                    q[i]['x'] += d*distance
                    moved = True
            if not moved:
                break
        else:
            raise ValueError('distance packing did not converge')
    return q


def _features(m):
    pieces, shape = m['pieces'], m['arena_shape']
    settled = _slide(pieces, *shape)
    seed_masks = [_tile_cells(m['plan'], m['side'], m['stride'], p['sites']) for p in pieces]
    seed_packed = _slide(pieces, *shape, masks=seed_masks)
    h = (m['plan'].shape[0]-1)*m['stride']+m['side']
    w = (m['plan'].shape[1]-1)*m['stride']+m['side']
    box = frozenset((r, c) for r in range(h) for c in range(w))
    try:
        boxed = _slide(pieces, *shape, masks=[box]*len(pieces))
        interlock = [p['x'] for p in boxed] != [p['x'] for p in settled]
    except ValueError:
        interlock = True
    final_box_overlap = any(
        a['d'] == b['d'] and max(a['y'], b['y']) < min(a['y']+h, b['y']+h)
        and max(a['x'], b['x']) < min(a['x']+w, b['x']+w)
        for i, a in enumerate(settled) for b in settled[i+1:])
    return dict(count=len(pieces), plan_shape=tuple(m['plan'].shape), side=m['side'],
                gutter=m['stride']-m['side'], holes=int(np.count_nonzero(m['plan'] == 0)),
                axis=int(m['transpose']), canvas=tuple(m['grid'].shape),
                matches=tuple(m['matches']), palette=tuple(sorted(set(int(v) for v in m['plan'].flat))),
                rails=(m['low'], m['high']),
                directions=tuple(sorted({p['d'] for p in pieces})),
                travel=tuple(abs(a['x']-b['x']) for a, b in zip(pieces, settled)),
                missing_changes_packing=[p['x'] for p in seed_packed] != [p['x'] for p in settled],
                interlock=interlock and final_box_overlap,
                shifted_match=any(r or c for r, c, length in m['matches']))


def _sample_pair(rng, purpose, axis, count):
    """Bounded rejection of geometrically infeasible latent-piece scenes."""
    for attempt in range(1200):
        ph = int(rng.integers(2, 5 if count <= 3 else 4))
        pw = int(rng.integers(2, 5 if count <= 3 else 4))
        side = int(rng.choice([3, 3, 3, 4]))
        gutter = int(rng.choice([1, 1, 2]))
        stride = side+gutter
        sh, sw = (ph-1)*stride+side, (pw-1)*stride+side
        header = 2*ph+int(rng.integers(1, 4))
        if sh > 30-header or sw > 26:
            continue
        height = int(rng.integers(max(sh, 11), 31-header))
        width = int(rng.integers(max(sw+1, 17), 29))
        colors = rng.permutation(np.arange(1, 10))
        low, high = int(colors[0]), int(colors[1])
        palette = colors[2:2+int(rng.integers(2, 6))]
        plan = rng.choice(palette, size=(ph, pw)).astype(np.int8)
        plan[rng.random((ph, pw)) < rng.uniform(.15, .45)] = 0
        if (any(not np.any(edge) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1]))
                or np.count_nonzero(plan)*side*side*count > height*width*.68):
            continue
        coords = np.argwhere(plan != 0)
        if any(not np.any(np.diff(np.unique(coords[:, d])) == 1) for d in (0, 1)):
            continue
        segments = [s for s in _segments(plan) if s[1] < np.count_nonzero(plan)]
        if len({s[0] for s in segments}) < 2:
            continue
        if purpose == 'cavities' and not np.any(plan == 0):
            continue
        mask = _tile_cells(plan, side, stride)
        pieces, occupied, seed_occupied, seed_rows = [], set(), set(), set()
        d0 = int(rng.choice([-1, 1]))
        directions = ([d0]*count if purpose == 'cavities' else
                      [-1, 1]+[int(rng.choice([-1, 1])) for _ in range(count-2)])
        rng.shuffle(directions)
        for d in directions:
            for pose_attempt in range(100):
                y = int(rng.integers(height-sh+1))
                x = int(rng.integers(width-sw+1))
                if purpose == 'opposite' and pieces:
                    r, c = pieces[0]['match']
                    length = len(pieces[0]['sites'])
                else:
                    (r, c), length, seq = segments[int(rng.integers(len(segments)))]
                sy = y+r*stride
                if sy in seed_rows:
                    continue
                sites = [(r, c+k) for k in range(length)]
                p = dict(y=y, x=x, d=d, frame=low if d < 0 else high,
                         mask=mask, sites=sites, match=(r, c))
                cells = _absolute(p)
                visible = _absolute(p, mask=_tile_cells(plan, side, stride, sites))
                touching = visible | {(yy+dy, xx+dx) for yy, xx in visible
                                      for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))}
                if occupied.intersection(cells) or seed_occupied.intersection(touching):
                    continue
                pieces.append(p)
                occupied.update(cells)
                seed_occupied.update(visible)
                seed_rows.add(sy)
                break
            else:
                break
        if len(pieces) != count:
            continue
        if purpose == 'opposite' and not (
                next(p['x'] for p in pieces if p['d'] < 0) >
                next(p['x'] for p in pieces if p['d'] > 0)):
            continue
        settled = _slide(pieces, height, width)
        try:
            inp = np.zeros((header+height, width+2), dtype=np.int8)
            lo, hi = header, header+height
            inp[lo:hi, 0], inp[lo:hi, -1] = low, high
            mr = int(rng.integers(header-(2*ph-1)))
            mc = int(rng.integers(1, inp.shape[1]-(2*pw-1)))
            inp[mr:mr+2*ph-1:2, mc:mc+2*pw-1:2] = plan
            for p in pieces:
                _draw_piece(inp[lo:hi, 1:-1], plan, side, stride, p, seed_only=True)
            out = inp.copy()
            out[lo:hi, 1:-1] = 0
            for p in settled:
                _draw_piece(out[lo:hi, 1:-1], plan, side, stride, p)
            if rng.integers(2):
                inp, out = np.flipud(inp).copy(), np.flipud(out).copy()
            if axis:
                inp, out = inp.T.copy(), out.T.copy()
            m = _decode(inp)
            f = _features(m)
            if not any(f['travel']) or not f['missing_changes_packing'] or not f['shifted_match']:
                continue
            if purpose == 'cavities' and not f['interlock']:
                continue
            if purpose == 'opposite' and (not all(f['travel']) or
                    any(np.array_equal(m['plan'], v) for v in
                        (np.flipud(m['plan']), np.fliplr(m['plan']), np.flip(m['plan'])))):
                continue
            if purpose == 'test' and not any(c > 0 or r > 0 for r, c, length in f['matches']):
                continue
            return {'input': inp, 'output': out}
        except ValueError:
            continue
    return _fallback_pair(rng, purpose, axis, count)


def _fallback_pair(rng, purpose, axis, count):
    """Constructive sparse-L scenes, also directly exercised by self_test."""
    colors = [int(v) for v in rng.permutation(np.arange(1, 10))]
    low, high, p, q = colors[:4]
    plan = np.array([[p, p, q], [p, 0, 0]], dtype=np.int8)
    side, stride, header = 3, 4, 6
    height, width = 4*(count-1)+7, int(rng.integers(24, 29))
    mask = _tile_cells(plan, side, stride)
    pieces = []
    for i in range(count):
        y, x, d, c, length = 4*i, 1+4*i, -1, 1, 2
        if purpose == 'opposite':
            y, x, d, c, length = ((0, 12, -1, 1, 2) if i == 0 else (1, 0, 1, 1, 2))
        pieces.append(dict(y=y, x=x, d=d, frame=low if d < 0 else high,
                           mask=mask, sites=[(0, c+k) for k in range(length)], match=(0, c)))
    inp = np.zeros((header+height, width+2), dtype=np.int8)
    inp[header:, 0], inp[header:, -1] = low, high
    mc = int(rng.integers(1, width-4))
    inp[0:3:2, mc:mc+5:2] = plan
    for piece in pieces:
        _draw_piece(inp[header:, 1:-1], plan, side, stride, piece, seed_only=True)
    out = inp.copy()
    out[header:, 1:-1] = 0
    for piece in _slide(pieces, height, width):
        _draw_piece(out[header:, 1:-1], plan, side, stride, piece)
    if rng.integers(2):
        inp, out = np.flipud(inp).copy(), np.flipud(out).copy()
    if axis:
        inp, out = inp.T.copy(), out.T.copy()
    _require(np.array_equal(solve(inp), out), 'constructive fallback mismatch')
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lower, upper in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lower <= int(value) <= upper):
            raise ValueError(f'{name} must be an integer in {lower}..{upper}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    axis = int(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            purpose = ('opposite' if i == 0 else 'cavities') if split == 'train' and i < 2 else split
            n = (2 if split == 'train' and i < 2 else int(rng.integers(2, 4))
                 if split == 'train' else max(len(_decode(p['input'])['pieces']) for p in puzzle['train'])+1)
            for _ in range(20):
                pair = _sample_pair(rng, purpose, axis, n)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct input search exhausted')
            seen.add(key)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, 'pair counts')
    seen, features, shared_seed_crossing = set(), {'train': [], 'test': []}, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            m = _decode(inp)
            settled = _reference_positions(m['pieces'], *m['arena_shape'])
            expected = _output(m, settled)
            _require(np.array_equal(out, expected), 'incorrect cargo, docking, or preserved pixels')
            f = _features(m)
            _require(f['missing_changes_packing'], 'missing blueprint cells must affect docking')
            _require(any(f['travel']), 'some cargo must move')
            features[split].append(f)
            if split == 'train':
                ps = m['pieces']
                shared_seed_crossing |= any(
                    a['d'] < 0 < b['d'] and a['x'] > b['x'] and a['sites'] == b['sites']
                    for a in ps for b in ps)
    _require(any(f['directions'] == (-1, 1) and all(f['travel']) for f in features['train']),
             'training must show both rail instructions')
    _require(shared_seed_crossing, 'training must separate frame instructions from payload and nearest edge')
    _require(any(f['interlock'] and len(f['directions']) == 1 and f['holes']
                 for f in features['train']), 'training must teach same-class cavity interactions')
    _require(all(f['count'] > max(t['count'] for t in features['train']) for f in features['test']),
             'test must combine more pieces')


def self_test():
    """Bounded semantic, ablation, mutation, count, replay and diversity checks."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    records, signatures = [], set()
    for puzzle in corpus:
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input'].copy()
                prediction = solve(inp)
                _require(np.array_equal(prediction, pair['output']), 'solve disagreement')
                _require(np.array_equal(inp, pair['input']), 'solver mutated input')
                digest.update(str(inp.shape).encode()+inp.tobytes()+prediction.tobytes())
                m = _decode(inp)
                f = _features(m)
                f['split'] = split
                records.append(f)
                # Removing motion (A) must also alter every generated answer.
                _require(not np.array_equal(_output(m, m['pieces']), prediction), 'motion ablation')
        signatures.add(digest.digest())
    _require(len(signatures) == 42, 'seed diversity')
    for key, minimum in (('canvas', 35), ('plan_shape', 5), ('rails', 25),
                         ('palette', 30), ('matches', 60), ('gutter', 2), ('side', 2)):
        _require(len({r[key] for r in records}) >= minimum, 'insufficient variation: '+key)
    _require({r['axis'] for r in records} == {0, 1}, 'both movement axes')
    _require({r['count'] for r in records} >= {2, 3, 4}, 'piece counts')
    _require(any(r['holes'] == 0 for r in records) and any(r['holes'] >= 3 for r in records), 'occupancy variation')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(731+a*10+b, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    replay = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in replay for a, b in zip(replay[s], corpus[0][s])
                 for k in ('input', 'output')), 'replay failed')
    for axis in (0, 1):
        fallback = {'train': [_fallback_pair(np.random.default_rng(501), 'opposite', axis, 2),
                              _fallback_pair(np.random.default_rng(502), 'cavities', axis, 2)],
                    'test': [_fallback_pair(np.random.default_rng(503), 'test', axis, 3)]}
        validate_generated_puzzle(fallback)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in corpus[0][s]] for s in corpus[0]}
    damaged['test'][0]['output'][0, 0] = 9 if damaged['test'][0]['output'][0, 0] != 9 else 8
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(records), 'distinct_puzzles': len(signatures),
            'canvas_shapes': len({r['canvas'] for r in records}),
            'plan_shapes': sorted({r['plan_shape'] for r in records}),
            'interlocking_pairs': sum(r['interlock'] for r in records)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
