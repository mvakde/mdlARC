"""Corner-registered transparent stencils. No parent modules or saved puzzles are used."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
SEEDS = (0,536541993,485350686,476882487,127214339,131194481,375602839,608465860,575517223,458489661,815181532,233872125,523390445,198911521,335089217,347071816,641011078,476640448,936813356,872127416,112350695,904242649,793694752,949355284,799300607,517000751,651205889,200073159,271190027,59759765,296818512,772343342,583722020,706993781,670514105,385535769,634933363,263427525,402557995,315183061,471282056,539195951)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _adjacent(index):
    return SIDES[index], SIDES[(index + 1) % 4]


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.uint8).tobytes()


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a


def _side(p, shape):
    r, c = p
    hits = [s for s, hit in (("T", r == 0), ("B", r == shape[0] - 1),
                             ("L", c == 0), ("R", c == shape[1] - 1)) if hit]
    _require(len(hits) == 1, "peg must be a noncorner edge cell")
    return hits[0]


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        points, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    points.append(q)
        result.append(points)
    return result


def _matrix(guides, outer):
    """Match both signed side normals: a unique orthogonal integer matrix."""
    m = np.zeros((2, 2), dtype=int)
    for source_side, color in guides.items():
        target_side = next(s for s, v in outer.items() if v == color)
        m += np.outer(NORMAL[target_side], NORMAL[source_side])
    _require(np.array_equal(m.T @ m, np.eye(2, dtype=int)), "ambiguous orientation")
    return m


def _decode(input_grid):
    """Recover every rectangle, orientation and registration from the input alone."""
    a = _grid(input_grid)
    H, W = a.shape
    bg = int(a[0, 0])
    _require(all(int(a[r, c]) == bg for r, c in ((0, -1), (-1, 0), (-1, -1))),
             "all four corners must show the background")
    edge = np.zeros(a.shape, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    pegs = [tuple(map(int, p)) for p in np.argwhere(edge & (a != bg))]
    _require(len(pegs) == 2, "exactly two boundary pegs required")
    outer = {_side(p, a.shape): int(a[p]) for p in pegs}
    _require(len(outer) == 2 and len(set(outer.values())) == 2, "pegs must be distinguishable")
    _require(frozenset(outer) in {frozenset(_adjacent(i)) for i in range(4)},
             "pegs must occupy adjacent sides")
    q = (next(r for r, c in pegs if c in (0, W - 1)),
         next(c for r, c in pegs if r in (0, H - 1)))
    bars = []
    for color in sorted(outer.values()):
        for cells in _components(a == color):
            if any(edge[p] for p in cells):
                _require(len(cells) == 1 and cells[0] in pegs, "pegs must be isolated")
                continue
            rows = sorted({r for r, c in cells})
            cols = sorted({c for r, c in cells})
            horizontal = len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1))
            vertical = len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1))
            _require(len(cells) >= 3 and horizontal != vertical, "marker component is not a guide bar")
            bars.append(dict(color=color, cells=cells, horizontal=horizontal,
                             low=(rows[0], cols[0]), high=(rows[-1], cols[-1])))
    _require(2 <= len(bars) <= 8 and len(bars) % 2 == 0, "one to four stencils required")
    candidates = []
    for i, h in enumerate(bars):
        if not h['horizontal']:
            continue
        for j, v in enumerate(bars):
            if v['horizontal'] or h['color'] == v['color']:
                continue
            hr, c0 = h['low']; c1 = h['high'][1]
            r0, vc = v['low']; r1 = v['high'][0]
            hs = 'T' if hr == r0 - 1 else 'B' if hr == r1 + 1 else None
            vs = 'L' if vc == c0 - 1 else 'R' if vc == c1 + 1 else None
            if hs is not None and vs is not None:
                candidates.append((i, j, (r0, c0, r1 + 1, c1 + 1),
                                   {hs: h['color'], vs: v['color']}, (hr, vc)))
    uses = Counter(i for candidate in candidates for i in candidate[:2])
    _require(all(uses[i] == 1 for i in range(len(bars))), "guide pairing is not unique")
    claimed = edge.copy()
    tiles = []
    for i, j, bounds, guides, corner in candidates:
        r0, c0, r1, c1 = bounds
        br0, bc0 = min(r0, corner[0]), min(c0, corner[1])
        br1, bc1 = max(r1, corner[0] + 1), max(c1, corner[1] + 1)
        _require(br0 >= 2 and bc0 >= 2 and br1 <= H - 2 and bc1 <= W - 2,
                 "stencils need an interior background moat")
        ring = np.concatenate((a[br0 - 1, bc0 - 1:bc1 + 1], a[br1, bc0 - 1:bc1 + 1],
                               a[br0:br1, bc0 - 1], a[br0:br1, bc1]))
        _require(bool(np.all(ring == bg)), "stencil moat is obstructed")
        _require(not claimed[br0:br1, bc0:bc1].any(), "stencil rectangles overlap")
        claimed[br0:br1, bc0:bc1] = True
        _require(int(a[corner]) == bg, "the virtual corner must be empty")
        payload = a[r0:r1, c0:c1]
        _require(not set(map(int, np.unique(payload))) & set(outer.values()),
                 "guide colors are reserved within each example")
        _require(bool(np.any(payload != bg)), "empty stencil")
        m = _matrix(guides, outer)
        # Full guide-defined extents, including blank strips, must fit the destination.
        for p in ((r0, c0), (r0, c1 - 1), (r1 - 1, c0), (r1 - 1, c1 - 1)):
            target = np.asarray(q) + m @ (np.asarray(p) - corner)
            _require(0 <= target[0] < H and 0 <= target[1] < W, "docked extent leaves canvas")
        tiles.append(dict(bounds=bounds, guides=guides, corner=corner, matrix=m,
                          payload=payload.copy()))
    _require(bool(np.all(a[~claimed] == bg)), "unexplained cells outside stencils")
    return dict(grid=a, background=bg, outer=outer, pegs=pegs, q=q, tiles=tiles)


def _projected(info, tile, ablation=None):
    r0, c0, r1, c1 = tile['bounds']
    m, corner = tile['matrix'], np.asarray(tile['corner'])
    q = np.asarray(info['q'])
    if ablation == 'no_A':
        q = corner
    elif ablation == 'no_B':
        m = np.eye(2, dtype=int)
    elif ablation == 'geometry':
        hside = next(s for s in info['outer'] if s in ('T', 'B'))
        vside = next(s for s in info['outer'] if s in ('L', 'R'))
        m = sum((np.outer(NORMAL[hside if s in ('T', 'B') else vside], NORMAL[s])
                 for s in tile['guides']), np.zeros((2, 2), dtype=int))
    elif ablation == 'tight':
        rr, cc = np.where(tile['payload'] != info['background'])
        r0, r1 = r0 + int(rr.min()), r0 + int(rr.max()) + 1
        c0, c1 = c0 + int(cc.min()), c0 + int(cc.max()) + 1
        corner = np.array((r0 - 1 if 'T' in tile['guides'] else r1,
                           c0 - 1 if 'L' in tile['guides'] else c1))
    result = {}
    for r in range(r0, r1):
        for c in range(c0, c1):
            color = int(info['grid'][r, c])
            if color == info['background'] and ablation != 'opaque':
                continue
            target = q + m @ (np.array((r, c)) - corner)
            p = tuple(map(int, target))
            if 0 <= p[0] < info['grid'].shape[0] and 0 <= p[1] < info['grid'].shape[1]:
                result[p] = color
    return result


def _render(info, ablation=None):
    out = np.full(info['grid'].shape, info['background'], dtype=int)
    ink = {}
    for tile in info['tiles']:
        for p, color in _projected(info, tile, ablation).items():
            if ablation is None:
                _require(p not in ink or ink[p] == color, "contradictory overprint")
                _require(p not in info['pegs'], "ink covers a boundary peg")
            ink[p] = color
            out[p] = color
    for p in info['pegs']:
        out[p] = info['grid'][p]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dock each stencil by its colored sides and virtual corner; union its ink."""
    return _render(_decode(input_grid))


def _mask(rng, h, w):
    """Composed bends, walks, hollow outlines, repeated teeth and sparse details."""
    mask = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(1, 5))):
        kind = int(rng.integers(5))
        r, c = int(rng.integers(h)), int(rng.integers(w))
        if kind == 0:
            for _ in range(int(rng.integers(3, h * w // 2 + 4))):
                mask[r, c] = True
                dr, dc = _pick(rng, ((-1, 0), (1, 0), (0, -1), (0, 1)))
                r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
        elif kind == 1:
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            mask[min(r, rr):max(r, rr) + 1, c] = True
            mask[rr, min(c, cc):max(c, cc) + 1] = True
        elif kind == 2:
            rr, cc = int(rng.integers(h)), int(rng.integers(w))
            lo, hi = sorted((r, rr)); left, right = sorted((c, cc))
            mask[[lo, hi], left:right + 1] = True
            mask[lo:hi + 1, [left, right]] = True
        elif kind == 3:
            step = int(rng.integers(2, 4))
            mask[r:min(h, r + 2), int(rng.integers(step))::step] = True
        else:
            mask |= rng.random((h, w)) < float(rng.uniform(.10, .35))
    return mask


def _stencils(rng, count, inks, bg, maximum, witness):
    shapes = [(int(rng.integers(4, maximum + 1)), int(rng.integers(4, maximum + 1)))
              for _ in range(count)]
    mh, mw = max(h for h, w in shapes), max(w for h, w in shapes)
    field = np.full((mh, mw), inks[0], dtype=int)
    for color in inks[1:]:
        field[_mask(rng, mh, mw)] = color
    masks = [_mask(rng, h, w) for h, w in shapes]
    if witness:
        masks[0][0, :] = False  # a real blank strip next to a registration guide
    # A common ink point and one exclusive point per stencil make union observable.
    common_h, common_w = min(h for h, w in shapes), min(w for h, w in shapes)
    positions = [(r, c) for r in range(1 if witness else 0, common_h) for c in range(common_w)]
    chosen = [positions[int(i)] for i in rng.permutation(len(positions))[:count + 1]]
    overlap, exclusive = chosen[0], chosen[1:]
    for i, mask in enumerate(masks):
        mask[overlap] = True
        for j, p in enumerate(exclusive):
            mask[p] = i == j
    return [np.where(mask, field[:h, :w], bg).astype(int)
            for (h, w), mask in zip(shapes, masks)]


def _local_block(canonical, guides, outer, bg):
    m = _matrix(guides, outer)
    h, w = canonical.shape
    sh, sw = (w, h) if m[0, 0] == 0 else (h, w)
    pr, pc = int('T' in guides), int('L' in guides)
    corner = np.array((0 if 'T' in guides else sh, 0 if 'L' in guides else sw))
    block = np.full((sh + 1, sw + 1), bg, dtype=int)
    sr = 1 if 'T' in outer else -1
    sc = 1 if 'L' in outer else -1
    for r in range(h):
        for c in range(w):
            p = corner + m.T @ np.array((sr * (r + 1), sc * (c + 1)))
            block[tuple(p)] = canonical[r, c]
    for side, color in guides.items():
        if side == 'T':
            block[0, pc:pc + sw] = color
        elif side == 'B':
            block[sh, pc:pc + sw] = color
        elif side == 'L':
            block[pr:pr + sh, 0] = color
        else:
            block[pr:pr + sh, sw] = color
    return block


def _pack(rng, blocks, bg):
    count = len(blocks)
    low = 28 if count >= 3 else 20 if count == 2 else 16
    H, W = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    for _ in range(12):
        a = np.full((H, W), bg, dtype=int)
        reserved = np.zeros((H, W), dtype=bool)
        okay = True
        for index in sorted(range(count), key=lambda i: -blocks[i].size):
            block = blocks[index]
            h, w = block.shape
            options = [(r, c) for r in range(2, H - h - 1) for c in range(2, W - w - 1)
                       if not reserved[r:r + h, c:c + w].any()]
            if not options:
                okay = False
                break
            r, c = _pick(rng, options)
            a[r:r + h, c:c + w] = block
            reserved[r - 1:r + h + 1, c - 1:c + w + 1] = True
        if okay:
            return a
    # Bounded spacious fallback; each block is at most 11 by 11.
    a = np.full((30, 30), bg, dtype=int)
    slots = [(2, 2), (2, 17), (17, 2), (17, 17)]
    for block, i in zip(blocks, rng.permutation(4)):
        r, c = slots[int(i)]
        a[r:r + block.shape[0], c:c + block.shape[1]] = block
    return a


def _corner_options(shape, canonicals, outer):
    mh = max(p.shape[0] for p in canonicals)
    mw = max(p.shape[1] for p in canonicals)
    H, W = shape
    rows = list(range(1, H - mh - 1)) if 'T' in outer else list(range(mh + 1, H - 1))
    cols = list(range(1, W - mw - 1)) if 'L' in outer else list(range(mw + 1, W - 1))
    _require(len(rows) > 1 and len(cols) > 1, "not enough docking room")
    return rows, cols


def _draw_pegs(grid, outer, q):
    for side, color in outer.items():
        p = {'T': (0, q[1]), 'B': (grid.shape[0] - 1, q[1]),
             'L': (q[0], 0), 'R': (q[0], grid.shape[1] - 1)}[side]
        grid[p] = color


def _canonical_answer(shape, canonicals, bg, outer, q):
    """Construction answer: paint canonical ink directly, without decoding source tiles."""
    out = np.full(shape, bg, dtype=int)
    sr, sc = (1 if 'T' in outer else -1), (1 if 'L' in outer else -1)
    for stencil in canonicals:
        for r, c in np.argwhere(stencil != bg):
            p = q[0] + sr * (int(r) + 1), q[1] + sc * (int(c) + 1)
            _require(out[p] in (bg, int(stencil[r, c])), "generation overprint conflict")
            out[p] = stencil[r, c]
    _draw_pegs(out, outer, q)
    return out


def _scene(rng, palette, outer_index, inner_pool, count, maximum, witness=False,
           required_inner=None):
    bg, a, b = palette[:3]
    inks = palette[3:3 + int(rng.integers(1, 5))]
    outer_sides = _adjacent(outer_index)
    outer = dict(zip(outer_sides, (a, b) if rng.integers(2) else (b, a)))
    guides = []
    if witness:
        # Same occupied guide sides but reversed identities, in one assembly.
        choices = []
        for i in inner_pool:
            for colors in ((a, b), (b, a)):
                g = dict(zip(_adjacent(i), colors))
                m = _matrix(g, outer)
                if round(np.linalg.det(m)) == 1 and np.trace(m) == 0:
                    choices.append(g)
        first = _pick(rng, choices)
        guides = [first, dict(zip(first, list(first.values())[::-1]))]
    while len(guides) < count:
        inner = required_inner if required_inner is not None and not guides else _pick(rng, inner_pool)
        colors = (a, b) if rng.integers(2) else (b, a)
        guides.append(dict(zip(_adjacent(inner), colors)))
    canonicals = _stencils(rng, count, inks, bg, maximum, witness)
    blocks = [_local_block(p, g, outer, bg) for p, g in zip(canonicals, guides)]
    inp = _pack(rng, blocks, bg)
    rows, cols = _corner_options(inp.shape, canonicals, outer)
    q = int(_pick(rng, rows)), int(_pick(rng, cols))
    _draw_pegs(inp, outer, q)
    out = _canonical_answer(inp.shape, canonicals, bg, outer, q)
    return {'input': inp, 'output': out}, (canonicals, outer, q, rows, cols)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # new destination corner, new local corner, or more fragments
    held = int(rng.integers(4))
    outer_pool = [i for i in range(4) if regime != 0 or i != held]
    inner_pool = [i for i in range(4) if regime != 1 or i != held]
    palette = tuple(map(int, rng.permutation(10)))
    stable = bool(rng.random() < .25)
    puzzle, seen = {'train': [], 'test': []}, set()
    first_outer = _pick(rng, outer_pool)
    # Bounded resampling rejects only ambiguous layouts and ineffective witnesses.
    for _ in range(60):
        try:
            first, records = _scene(rng, palette, first_outer, inner_pool, 2, 8, True)
            info = _decode(first['input'])
            _require(np.array_equal(_render(info), first['output']), "construction disagreement")
            for shortcut in ('no_A', 'no_B', 'geometry', 'tight', 'opaque'):
                _require(not np.array_equal(_render(info, shortcut), first['output']),
                         'ineffective teaching witness')
            break
        except AssertionError:
            continue
    else:
        raise RuntimeError('could not construct an informative registration witness')
    puzzle['train'].append(first)
    seen.add(_key(first['input']))
    canonicals, outer, q, rows, cols = records
    q2 = (int(_pick(rng, [r for r in rows if r != q[0]])),
          int(_pick(rng, [c for c in cols if c != q[1]])))
    second_input = first['input'].copy()
    for p in info['pegs']:
        second_input[p] = palette[0]
    _draw_pegs(second_input, outer, q2)
    second = {'input': second_input,
              'output': _canonical_answer(second_input.shape, canonicals, palette[0], outer, q2)}
    puzzle['train'].append(second)
    seen.add(_key(second_input))
    for index in range(2, nt + ne):
        test = index >= nt
        for _ in range(60):
            local_palette = palette if stable else tuple(map(int, rng.permutation(10)))
            oi = held if test and regime == 0 else _pick(rng, outer_pool)
            pool = list(range(4)) if test else inner_pool
            n = int(rng.integers(3, 5)) if test and regime == 2 else int(rng.integers(1, 3 if regime == 2 else 5))
            try:
                pair, _ = _scene(rng, local_palette, oi, pool, n, 10,
                                 required_inner=held if test and regime == 1 else None)
                decoded = _decode(pair['input'])
                _require(np.array_equal(_render(decoded), pair['output']), 'construction disagreement')
                _require(_key(pair['input']) not in seen, 'duplicate input')
                break
            except AssertionError:
                continue
        else:
            raise RuntimeError('bounded scene construction exhausted')
        seen.add(_key(pair['input']))
        puzzle['test' if test else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    projected = [_projected(info, t) for t in info['tiles']]
    counts = Counter(p for stamp in projected for p in stamp)
    return dict(stencils=len(projected), outer=tuple(sorted(info['outer'])),
                inner=[tuple(sorted(t['guides'])) for t in info['tiles']],
                matrices=[tuple(map(int, t['matrix'].ravel())) for t in info['tiles']],
                shapes=[t['payload'].shape for t in info['tiles']],
                blank=sum(any(np.all(line == info['background']) for line in
                              (t['payload'][0], t['payload'][-1], t['payload'][:, 0], t['payload'][:, -1]))
                          for t in info['tiles']),
                overlap=sum(v > 1 for v in counts.values()),
                exclusive=[sum(counts[p] == 1 for p in stamp) for stamp in projected])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts')
    seen, decoded = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'canvas dimensions must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            _require(np.array_equal(_render(info), out), 'incorrect docking, transparency or ink')
            f = _features(info)
            _require(all(f['exclusive']), 'a stencil makes no distinct visible contribution')
            _require(f['stencils'] == 1 or f['overlap'] > 0, 'multi-stencil assembly must share ink')
            _require(int(out[info['q']]) == info['background'], 'virtual registration corner is empty')
            decoded[split].append(info)
    # All pedagogical claims are checked against returned grids, not metadata.
    train = decoded['train']
    for shortcut in ('no_A', 'no_B', 'geometry', 'tight', 'opaque'):
        _require(any(not np.array_equal(_render(i, shortcut), _render(i)) for i in train),
                 f'training does not distinguish {shortcut}')
    def without_pegs(info):
        a = info['grid'].copy()
        for p in info['pegs']:
            a[p] = info['background']
        return _key(a)
    contrasts = [(a, b) for j, a in enumerate(train) for b in train[j + 1:]
                 if a['outer'] == b['outer'] and without_pegs(a) == without_pegs(b)
                 and all(x != y for x, y in zip(a['q'], b['q']))]
    _require(contrasts, 'training lacks a pure peg-relocation contrast')
    _require(any(any(set(a['guides']) == set(b['guides']) and a['guides'] != b['guides']
                         for j, a in enumerate(i['tiles']) for b in i['tiles'][j + 1:]) for i in train),
             'training lacks a same-side correspondence reversal')
    _require(any(round(np.linalg.det(t['matrix'])) == 1 and np.trace(t['matrix']) == 0
                 for i in train for t in i['tiles']), 'quarter-turn not demonstrated')
    _require(any(round(np.linalg.det(t['matrix'])) == -1 for i in train for t in i['tiles']),
             'reflection not demonstrated')
    train_outer = {tuple(sorted(i['outer'])) for i in train}
    train_inner = {tuple(sorted(t['guides'])) for i in train for t in i['tiles']}
    train_count = max(len(i['tiles']) for i in train)
    _require(all(tuple(sorted(i['outer'])) not in train_outer or
                 any(tuple(sorted(t['guides'])) not in train_inner for t in i['tiles']) or
                 len(i['tiles']) > train_count for i in decoded['test']),
             'test does not have an observable structural shift')


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                h.update(bytes(np.asarray(p[role]).shape))
                h.update(np.asarray(p[role], dtype=np.uint8).tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; keep feature sets and hashes, never full puzzle collections."""
    fingerprints, shapes, local_shapes, matrices, corners, counts = set(), set(), set(), set(), set(), set()
    structures, backgrounds, palettes, overlaps, blank = set(), set(), set(), 0, 0
    essential = total_pairs = 0
    split_matrices = {'train': set(), 'test': set()}
    for seed in SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        _require(fingerprint not in fingerprints, 'repeated puzzle across seeds')
        fingerprints.add(fingerprint)
        if seed == 0:
            _require(fingerprint == _fingerprint(generate_puzzle(seed)), 'nondeterminism')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input']); f = _features(info)
                total_pairs += 1
                shapes.add(pair['input'].shape)
                local_shapes.update(f['shapes']); matrices.update(f['matrices'])
                split_matrices[split].update(f['matrices'])
                corners.add(f['outer']); counts.add(f['stencils'])
                backgrounds.add(info['background'])
                palettes.add(tuple(map(int, np.unique(pair['input']))))
                overlaps += f['overlap'] > 0; blank += f['blank'] > 0
                essential += sum(v > 0 for v in f['exclusive'])
                roles = {}; normalized = bytes(roles.setdefault(int(v), len(roles)) for v in pair['input'].ravel())
                structures.add(hashlib.sha256(bytes(pair['input'].shape) + normalized).hexdigest())
    _require(len(fingerprints) == 42 and len(structures) >= 150, 'structural diversity')
    _require(len(matrices) == 8 and all(len(v) == 8 for v in split_matrices.values()), 'D4 coverage')
    _require(len(corners) == 4 and counts == {1, 2, 3, 4}, 'layout/count coverage')
    _require(len(shapes) >= 30 and len(local_shapes) >= 40, 'geometry diversity')
    _require(backgrounds == set(range(10)) and len(palettes) >= 20, 'role palette diversity')
    _require(overlaps >= 100 and blank >= 60, 'assembly/crop witnesses')
    # Hand-computed quarter-turn with eccentric ink and unequal guide lengths.
    a = np.zeros((14, 16), dtype=int)
    a[0, 10], a[8, 0] = 1, 2
    a[3:6, 3], a[6, 4:8] = 1, 2
    a[3, 4], a[5, 5] = 3, 4
    expected = np.zeros_like(a)
    expected[0, 10], expected[8, 0], expected[9, 13], expected[10, 11] = 1, 2, 3, 4
    _require(np.array_equal(solve(a), expected), 'hand registration fixture')
    # Simultaneous recoloring and rotation preserve the complete rule.
    p = generate_puzzle(7, 3, 1)
    permutation = np.array((7, 2, 8, 0, 6, 1, 9, 3, 5, 4))
    transformed = {s: [{k: permutation[np.rot90(v)] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(transformed)
    # Corrupting output or conflicting an overprinted ink cell must be detected.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][1, 1] = (int(bad['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    i = _decode(p['train'][0]['input'])
    stamps = [_projected(i, t) for t in i['tiles']]
    shared = next(iter(set(stamps[0]) & set(stamps[1])))
    tile = i['tiles'][0]
    source = np.asarray(tile['corner']) + tile['matrix'].T @ (np.asarray(shared) - i['q'])
    conflict = i['grid'].copy()
    new_color = next(c for c in range(10) if c not in (*i['outer'].values(), i['background'], stamps[0][shared]))
    conflict[tuple(source)] = new_color
    try:
        solve(conflict)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory overprint accepted')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=total_pairs, distinct_puzzles=len(fingerprints),
                color_normalized_inputs=len(structures), canvas_shapes=len(shapes),
                stencil_shapes=len(local_shapes), matrices=len(matrices),
                matrices_by_split={k: len(v) for k, v in split_matrices.items()},
                outer_corners=len(corners), stencil_counts=sorted(counts),
                backgrounds=len(backgrounds), palettes=len(palettes),
                pairs_with_overlap=overlaps, pairs_with_blank_stencil_edge=blank,
                essential_stencils=essential)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
