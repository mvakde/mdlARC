"""Order-constrained square assembly. Standard library and NumPy only."""
from functools import lru_cache
import json

import numpy as np


HOLLOW, SOLID = 0, 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum, maximum=None):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _require(value >= minimum and (maximum is None or value <= maximum),
             "Illegal " + name)
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _cells(size, kind):
    return tuple((r, c) for r in range(size) for c in range(size)
                 if kind == SOLID or r in (0, size-1) or c in (0, size-1))


def _area(piece):
    return piece['size'] ** 2 if piece['kind'] == SOLID else 4 * piece['size'] - 4


def _queues(pieces):
    return [sorted((p for p in pieces if p['kind'] == kind), key=lambda p: p['top'])
            for kind in (HOLLOW, SOLID)]


def _size_profile(analysis):
    return tuple(tuple(p['size'] for p in q) for q in analysis['queues'])


def _pack(board, background, queues):
    """Return at most two colored answers, with a placement witness for each.

    Every allowed piece has a full top edge starting at its leftmost column.
    Therefore the first uncovered cell must be the top-left corner of the next
    placed square. Only the two queue heads need be considered. This covers all
    legal assemblies, including nested pieces and non-chamber arrangements.
    """
    h, w = board.shape
    holes = sum(1 << (int(r)*w + int(c)) for r, c in np.argwhere(board == background))
    if sum(_area(p) for q in queues for p in q) != holes.bit_count():
        return ()

    @lru_cache(maxsize=None)
    def search(empty, hi, si, last_h, last_s):
        if not empty:
            return ((0, ()),) if hi == len(queues[0]) and si == len(queues[1]) else ()
        first = (empty & -empty).bit_length() - 1
        r, c = divmod(first, w)
        answers = {}
        for kind, rank, last in ((HOLLOW, hi, last_h), (SOLID, si, last_s)):
            if rank == len(queues[kind]) or r <= last:
                continue
            p = queues[kind][rank]
            s = p['size']
            if r+s > h or c+s > w:
                continue
            ids = [(r+dr)*w+c+dc for dr, dc in _cells(s, kind)]
            mask = sum(1 << i for i in ids)
            if mask & empty != mask:
                continue
            paint = sum((p['color']+1) << (4*i) for i in ids)
            args = (empty ^ mask, hi+int(kind == HOLLOW), si+int(kind == SOLID),
                    r if kind == HOLLOW else last_h, r if kind == SOLID else last_s)
            for suffix, route in search(*args):
                code = paint | suffix
                answers.setdefault(code, ((kind, rank, r, c),) + route)
                if len(answers) == 2:
                    return tuple(answers.items())
        return tuple(answers.items())

    results = []
    for code, route in search(holes, 0, 0, -1, -1):
        out = board.copy()
        for r, c in np.argwhere(board == background):
            out[r, c] = ((code >> (4*(int(r)*w+int(c)))) & 15) - 1
        results.append((out, route))
    return tuple(results)


def _parse_exterior(inp, background, bbox):
    br, bc, bh, bw = bbox
    outside = inp != background
    outside[br:br+bh, bc:bc+bw] = False
    pieces = []
    for comp in _components(outside):
        rr, cc = zip(*comp)
        r, c, s = min(rr), min(cc), max(rr)-min(rr)+1
        _require(max(cc)-c+1 == s, "Loose piece must have a square extent")
        tile = inp[r:r+s, c:c+s]
        color = int(inp[comp[0]])
        _require(all(int(inp[q]) == color for q in comp), "Loose piece is not monochrome")
        if np.all(tile == color):
            kind = SOLID
        else:
            _require(s >= 3, "Hollow frame has no interior")
            wanted = np.full((s, s), background, dtype=np.int8)
            for dr, dc in _cells(s, HOLLOW):
                wanted[dr, dc] = color
            _require(np.array_equal(tile, wanted), "Malformed hollow frame")
            kind = HOLLOW
        near = inp[r-1:r+s+1, c-1:c+s+1].copy()
        _require(near.shape == (s+2, s+2), "Pieces must be inside the canvas border")
        near[1:-1, 1:-1] = background
        _require(np.all(near == background), "Loose pieces need a full background moat")
        _require(r+s < br or r > br+bh or c+s < bc or c > bc+bw,
                 "Loose piece touches the board extent")
        pieces.append(dict(kind=kind, size=s, color=color, top=r, left=c))
    _require(2 <= len(pieces) <= 12, "Family inventory count must be 2..12")
    queues = _queues(pieces)
    for q in queues:
        _require(q and len({p['top'] for p in q}) == len(q),
                 "Both queues must exist, with no same-class top-row ties")
    board = inp[br:br+bh, bc:bc+bw]
    fixed_colors = set(map(int, np.unique(board))) - {background}
    _require(fixed_colors.isdisjoint(p['color'] for p in pieces),
             "Scaffold and loose-piece palettes must be disjoint")
    return queues


def _analyze(input_grid):
    inp = _grid(input_grid)
    background = int(inp[0, 0])
    _require(np.all(inp[0] == background) and np.all(inp[-1] == background)
             and np.all(inp[:, 0] == background) and np.all(inp[:, -1] == background),
             "Canvas must have a uniform one-cell background border")
    feasible = []
    for color in map(int, np.unique(inp)):
        if color == background:
            continue
        rr, cc = np.where(inp == color)
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max())-r+1, int(cc.max())-c+1
        if min(h, w) < 5:
            continue
        board = inp[r:r+h, c:c+w].copy()
        if not (np.all(board[0] == color) and np.all(board[-1] == color)
                and np.all(board[:, 0] == color) and np.all(board[:, -1] == color)):
            continue
        if not np.any(board == background):
            continue
        bbox = (r, c, h, w)
        try:
            queues = _parse_exterior(inp, background, bbox)
        except ValueError:
            continue
        answers = _pack(board, background, queues)
        if not answers:
            continue
        _require(len(answers) == 1, "Order and inventory admit different colored completions")
        out, route = answers[0]
        feasible.append(dict(background=background, border_color=color, bbox=bbox,
                             board=board, queues=queues, output=out, route=route))
    _require(len(feasible) == 1, "Input must have exactly one feasible board crop")
    return feasible[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the unique ordered exact-cover assembly from the input alone."""
    return _analyze(input_grid)['output'].copy()


def _destination_pieces(chambers):
    pieces = []
    for r, c, s, status in chambers:
        if status in ('both', 'shell'):
            pieces.append(dict(kind=HOLLOW, size=s, top=r, left=c))
        if status in ('both', 'core'):
            pieces.append(dict(kind=SOLID, size=s-2, top=r+1, left=c+1))
        if status == 'mono':
            pieces.append(dict(kind=SOLID, size=s, top=r, left=c))
    return pieces


def _chamber_layout(rng, sizes, statuses, max_piece):
    """Variable shelves initialize a packing; free relocation changes it."""
    n = len(sizes)
    cap_w = min(22, 30-2*max_piece-4)
    for _ in range(80):
        order = list(map(int, rng.permutation(n)))
        rows = []
        while order:
            count = int(rng.integers(1, min(3, len(order))+1))
            ids, order = order[:count], order[count:]
            offsets = list(map(int, rng.permutation(count)))
            gaps = list(map(int, rng.integers(1, 3, size=max(0, count-1))))
            rows.append((ids, offsets, gaps))
        widths = [sum(sizes[i] for i in ids)+sum(gaps) for ids, _, gaps in rows]
        heights = [max(sizes[i]+o for i, o in zip(ids, offsets)) for ids, offsets, _ in rows]
        vgaps = list(map(int, rng.integers(1, 3, size=max(0, len(rows)-1))))
        min_h, min_w = sum(heights)+sum(vgaps)+2, max(widths)+2
        if min_h > 25 or min_w > cap_w:
            continue
        h = int(rng.integers(min_h, min(25, min_h+4)+1))
        w = int(rng.integers(min_w, cap_w+1))
        slack = rng.multinomial(h-min_h, np.full(len(rows)+1, 1/(len(rows)+1)))
        positions = [None]*n
        top = 1+int(slack[0])
        for j, (ids, offsets, gaps) in enumerate(rows):
            left = int(rng.integers(1, w-widths[j]))
            for k, (i, offset) in enumerate(zip(ids, offsets)):
                positions[i] = (top+offset, left)
                left += sizes[i] + (gaps[k] if k < len(gaps) else 0)
            top += heights[j] + (vgaps[j] if j < len(vgaps) else 0)+int(slack[j+1])
        # Relocate entire chambers, preserving a scaffold cell between them.
        for _ in range(2):
            for i in map(int, rng.permutation(n)):
                s = sizes[i]
                rr, cc = np.indices((h-s-1, w-s-1))
                rr, cc = rr+1, cc+1
                legal = np.ones(rr.shape, dtype=bool)
                for j, (pr, pc) in enumerate(positions):
                    if j == i:
                        continue
                    ps = sizes[j]
                    legal &= (rr+s < pr) | (rr > pr+ps) | (cc+s < pc) | (cc > pc+ps)
                    legal &= rr != pr
                sites = np.argwhere(legal)
                if len(sites):
                    ar, ac = sites[int(rng.integers(len(sites)))]
                    positions[i] = (int(ar)+1, int(ac)+1)
        if len({r for r, _ in positions}) != n:
            continue
        ranked = sorted((r, c, sizes[i], statuses[i]) for i, (r, c) in enumerate(positions))
        # The two minimal demonstrations deliberately put fixed parts in
        # different queue positions; normal scenes keep their sampled statuses.
        if statuses == ['core', 'both', 'shell', 'mono'] and len(set(sizes)) == 1:
            ranked = [(r, c, s, status) for (r, c, s, _), status in zip(ranked, statuses)]
        elif statuses == ['shell', 'both', 'shell', 'core'] and len(set(sizes)) == 1:
            ranked = [(r, c, s, status) for (r, c, s, _), status in zip(ranked, statuses)]
        pieces = _destination_pieces(ranked)
        qs = _queues(pieces)
        if any(not q or len({p['top'] for p in q}) != len(q) for q in qs):
            continue
        return (h, w), ranked, pieces
    return None


def _color_pieces(rng, pieces, background, fixed_colors, demo_color, forced_kind, witness):
    available = [c for c in range(10) if c != background and c not in fixed_colors]
    rng.shuffle(available)
    if witness or rng.random() < 0.35:
        colors = available[:len(pieces)]
        if len(colors) < len(pieces):
            colors += list(map(int, rng.choice(available, len(pieces)-len(colors))))
    else:
        pool = available[:int(rng.integers(1, min(5, len(available))+1))]
        colors = list(map(int, rng.choice(pool, len(pieces))))
    for p, color in zip(pieces, colors):
        p['color'] = int(color)
    if forced_kind is not None:
        q = _queues(pieces)[forced_kind]
        other = next((p for p in pieces if p['color'] == demo_color), None)
        if other is not None:
            other['color'], q[0]['color'] = q[0]['color'], other['color']
        else:
            q[0]['color'] = demo_color
    if witness:
        for q in _queues(pieces):
            if q[0]['color'] == q[-1]['color']:
                q[-1]['color'] = next(c for c in available if c != q[0]['color'])


def _merge(rng, queues, witness):
    counts = [len(q) for q in queues]
    mode = 0 if witness else int(rng.integers(4))
    if mode <= 1:
        kinds = [0]*counts[0]+[1]*counts[1]
        if mode == 1 or (not witness and rng.integers(2)):
            kinds.reverse()
    elif mode == 2:
        kinds = []
        while sum(counts):
            for k in (0, 1):
                if counts[k]:
                    kinds.append(k)
                    counts[k] -= 1
    else:
        kinds = list(map(int, rng.permutation([0]*counts[0]+[1]*counts[1])))
    ranks = [0, 0]
    result = []
    for kind in kinds:
        result.append((kind, ranks[kind]))
        ranks[kind] += 1
    return result


def _scatter(rng, board, queues, background, witness):
    bh, bw = board.shape
    max_s = max(p['size'] for q in queues for p in q)
    strip_w = 2*max_s+1
    for _ in range(40):
        schedule = _merge(rng, queues, witness)
        bands, top = [], 0
        for start in range(0, len(schedule), 2):
            band = schedule[start:start+2]
            offsets = [0] if len(band) == 1 else [0, int(band[0][0] == band[1][0] or rng.integers(2))]
            sizes = [queues[k][i]['size'] for k, i in band]
            width = sum(sizes)+len(band)-1
            left = int(rng.integers(0, strip_w-width+1))
            horizontal = [left]
            if len(band) == 2:
                if rng.integers(2):
                    horizontal = [left+sizes[1]+1, left]
                else:
                    horizontal.append(left+sizes[0]+1)
            bands += [(k, i, top+o, c) for (k, i), o, c in zip(band, offsets, horizontal)]
            top += max(s+o for s, o in zip(sizes, offsets)) + 1
        source_h = top-1
        min_h, min_w = max(bh, source_h)+2, bw+strip_w+3
        if min_h > 30 or min_w > 30:
            continue
        h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
        extra = rng.multinomial(w-min_w, [1/3, 1/3, 1/3])
        left, gap = 1+int(extra[0]), 1+int(extra[1])
        if rng.integers(2):
            bc, sc = left, left+bw+gap
        else:
            sc, bc = left, left+strip_w+gap
        br = int(rng.integers(1, h-bh))
        sr = int(rng.integers(1, h-source_h))
        source = [[dict(p) for p in q] for q in queues]
        for kind, rank, r, c in bands:
            source[kind][rank]['top'], source[kind][rank]['left'] = r+sr, c+sc
        # Free relocation can put pieces above, below, or on either side of
        # the board. Queue inequalities, not a strip convention, govern order.
        for _ in range(3):
            ids = [(k, i) for k, q in enumerate(source) for i in range(len(q))]
            for order_index in rng.permutation(len(ids)):
                kind, rank = ids[int(order_index)]
                p = source[kind][rank]
                s = p['size']
                rr, cc = np.indices((h-s-1, w-s-1))
                rr, cc = rr+1, cc+1
                legal = (rr+s < br) | (rr > br+bh) | (cc+s < bc) | (cc > bc+bw)
                for k, q in enumerate(source):
                    for i, other in enumerate(q):
                        if (k, i) == (kind, rank):
                            continue
                        pr, pc, ps = other['top'], other['left'], other['size']
                        legal &= (rr+s < pr) | (rr > pr+ps) | (cc+s < pc) | (cc > pc+ps)
                if rank:
                    legal &= rr > source[kind][rank-1]['top']
                if rank+1 < len(source[kind]):
                    legal &= rr < source[kind][rank+1]['top']
                sites = np.argwhere(legal)
                if len(sites):
                    ar, ac = sites[int(rng.integers(len(sites)))]
                    p['top'], p['left'] = int(ar)+1, int(ac)+1
        inp = np.full((h, w), background, dtype=np.int8)
        inp[br:br+bh, bc:bc+bw] = board
        for q in source:
            for p in q:
                for dr, dc in _cells(p['size'], p['kind']):
                    inp[p['top']+dr, p['left']+dc] = p['color']
        return inp
    return None


def _horizontal_disagrees(analysis, reverse):
    return any([(p['size'], p['color']) for p in q] !=
               [(p['size'], p['color']) for p in sorted(q, key=lambda p: (p['left'], p['top']), reverse=reverse)]
               for q in analysis['queues'])


def _make_pair(rng, demo_color, mode, demo_size=None):
    for _ in range(240):
        if mode == 'first':
            statuses, sizes = ['core', 'both', 'shell', 'mono'], [demo_size]*4
        elif mode == 'second':
            statuses, sizes = ['shell', 'both', 'shell', 'core'], [demo_size]*4
        else:
            n = int(rng.integers(2, 6))
            statuses = list(rng.choice(['both', 'shell', 'core', 'mono'], size=n, p=[.4, .2, .25, .15]))
            statuses[int(rng.integers(n))] = 'both'
            sizes = list(map(int, rng.integers(3, 7, size=n)))
            if mode == 'ordinary' and rng.random() < .2:
                sizes = [sizes[0]]*n
            if mode == 'test' and len(set(sizes)) < 2:
                continue
        provisional = _destination_pieces([(0, 0, s, status) for s, status in zip(sizes, statuses)])
        max_piece = max(p['size'] for p in provisional)
        if len(provisional) > 8 or (len(provisional) >= 7 and max_piece > 5):
            continue
        layout = _chamber_layout(rng, sizes, statuses, max_piece)
        if layout is None:
            continue
        shape, chambers, pieces = layout
        other_colors = [c for c in range(10) if c != demo_color]
        rng.shuffle(other_colors)
        background, boundary = other_colors[:2]
        fixed_colors = [boundary]
        if rng.random() < .45:
            fixed_colors.append(other_colors[2])
        witness = mode in ('first', 'second')
        forced_kind = HOLLOW if mode == 'first' else SOLID if mode == 'second' else None
        _color_pieces(rng, pieces, background, fixed_colors, demo_color, forced_kind, witness)
        h, w = shape
        board = np.full(shape, boundary, dtype=np.int8)
        if len(fixed_colors) == 2:
            rr, cc = np.indices(shape)
            pattern = (rr//int(rng.integers(1, 4)) + cc//int(rng.integers(1, 4))) % 2
            board[1:-1, 1:-1] = np.where(pattern[1:-1, 1:-1], fixed_colors[1], boundary)
        output = board.copy()
        for p in pieces:
            for dr, dc in _cells(p['size'], p['kind']):
                cell = p['top']+dr, p['left']+dc
                board[cell] = background
                output[cell] = p['color']
        queues = _queues(pieces)
        answers = _pack(board, background, queues)
        if len(answers) != 1 or not np.array_equal(answers[0][0], output):
            continue
        inp = _scatter(rng, board, queues, background, mode == 'first')
        if inp is None:
            continue
        analysis = _analyze(inp)
        if mode == 'first' and not all(_horizontal_disagrees(analysis, rev) for rev in (False, True)):
            continue
        if mode == 'test' and (not _ordering_witness(analysis)
                              or not any(hp['color'] != sp['color']
                                         for _, _, hp, sp in _nested(analysis))):
            continue
        _require(np.array_equal(analysis['output'], output), "Construction and input-only assembly disagree")
        return {'input': inp, 'output': output}
    raise RuntimeError("Bounded sampling did not produce an ordered nesting assembly")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    _require(nt > ne, "num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    demo_color = int(rng.integers(10))
    scales = list(map(int, rng.choice([3, 4, 5], size=2, replace=False)))
    puzzle = {'train': [], 'test': []}
    seen, train_profiles = set(), set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            mode = ('first' if i == 0 else 'second' if i == 1 else 'ordinary') if split == 'train' else 'test'
            for _ in range(40):
                pair = _make_pair(rng, demo_color, mode, scales[i] if split == 'train' and i < 2 else None)
                key = _key(pair['input'])
                profile = _size_profile(_analyze(pair['input']))
                if key in seen or (split == 'test' and profile in train_profiles):
                    continue
                break
            else:
                raise RuntimeError("Could not obtain a distinct example and test size profile")
            seen.add(key)
            if split == 'train':
                train_profiles.add(profile)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _route_pieces(a):
    return [(kind, rank, r, c, a['queues'][kind][rank]) for kind, rank, r, c in a['route']]


def _nested(a):
    route = _route_pieces(a)
    return [(hk, sk, hp, sp) for kind, hk, hr, hc, hp in route if kind == HOLLOW
            for kind2, sk, sr, sc, sp in route if kind2 == SOLID
            and (sr, sc, sp['size']) == (hr+1, hc+1, hp['size']-2)]


def _ordering_witness(a):
    # Swapping equal shapes of different colors is a valid unordered assembly
    # with a different answer. It fails the inherited order constraint.
    return any(p['size'] == t['size'] and p['color'] != t['color']
               for q in a['queues'] for i, p in enumerate(q) for t in q[i+1:])


def _validate_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    a = _analyze(inp)
    _require(np.array_equal(out, a['output']), "Output differs from ordered exact assembly")
    bg, board = a['background'], a['board']
    fixed = board != bg
    _require(np.array_equal(board[fixed], out[fixed]), "Fixed cells changed")
    _require(not np.any(out == bg), "Board was not completely filled")
    _require(np.array_equal(np.bincount(inp[inp != bg], minlength=10),
                           np.bincount(out.ravel(), minlength=10)), "Inventory cells were copied or discarded")
    occupied = set()
    tops = [[], []]
    for kind, rank, r, c, p in _route_pieces(a):
        tops[kind].append(r)
        cells = {(r+dr, c+dc) for dr, dc in _cells(p['size'], kind)}
        _require(not occupied & cells, "Pieces overlap")
        _require(all(board[q] == bg and int(out[q]) == p['color'] for q in cells),
                 "Piece shape, orientation, or color changed")
        occupied.update(cells)
    _require(occupied == set(map(tuple, np.argwhere(~fixed).tolist())), "Incorrect hole coverage")
    _require(all(len(ts) == len(set(ts)) for ts in tops), "Destination ties within a class")
    return a


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "Splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Invalid example counts")
    seen, train, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _validate_pair(pair)
            key = _key(pair['input'])
            _require(key not in seen, "Duplicate input")
            seen.add(key)
            (train if split == 'train' else tests).append(a)
    _require(any(_ordering_witness(a) for a in train), "No witness for ordering's contribution")
    _require(any(hk != sk for a in train for hk, sk, _, _ in _nested(a)),
             "No witness against pairing the two queues by equal rank")
    _require(any(hp['color'] != sp['color'] for a in train for _, _, hp, sp in _nested(a)),
             "No visibly multicolor shared gap")
    _require(any(len(a['queues'][0]) < len(a['queues'][1]) for a in train)
             and any(len(a['queues'][0]) > len(a['queues'][1]) for a in train),
             "Demonstrations must expose both independent queue imbalances")
    for reverse in (False, True):
        _require(any(_horizontal_disagrees(a, reverse) for a in train), "Horizontal shortcut not distinguished")
    roles = {}
    for a in train:
        for kind, q in enumerate(a['queues']):
            for p in q:
                roles.setdefault(p['color'], set()).add(kind)
    _require(any(len(v) == 2 for v in roles.values()), "Missing literal-color role exchange")
    for kind in (HOLLOW, SOLID):
        _require(any(len({p['color'] for p in a['queues'][kind]}) >= 2 for a in train),
                 "Queue order must be observable")
    profiles = {_size_profile(a) for a in train}
    _require(all(_size_profile(a) not in profiles for a in tests), "Missing test size-profile recombination")
    _require(all(_ordering_witness(a) and any(hp['color'] != sp['color']
                 for _, _, hp, sp in _nested(a)) for a in tests),
             "Tests must require order and visibly nested assembly")


def self_test():
    """Bounded 42-seed checks plus family-specific positive/negative controls."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, boards, profiles, geometries, colors, inventories, nests = set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _validate_pair(pair)
                assert np.array_equal(solve(pair['input']), pair['output'])
                shapes.add(np.asarray(pair['input']).shape)
                boards.add(a['board'].shape)
                profiles.add(_size_profile(a))
                geometries.add((a['board'].shape, (a['board'] == a['background']).tobytes()))
                colors.add(a['background'])
                inventories.add(tuple(map(len, a['queues'])))
                nests.add(len(_nested(a)))
    assert len({_signature(p) for p in puzzles}) == 42
    assert len(shapes) >= 25 and len(boards) >= 30 and len(geometries) >= 150
    assert len(profiles) >= 65 and len(inventories) >= 12 and len(nests) >= 3
    assert colors == set(range(10))
    assert _signature(generate_puzzle(123)) == _signature(generate_puzzle(123))
    # Every legal explicit count request, and representative illegal types.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # True ambiguity: two equal square holes at the same top row can be filled
    # either by a large solid or by a ring plus core. Either choice respects
    # both queues, so exact inventory and order must still prove uniqueness.
    ambiguous = np.full((7, 13), 8, dtype=np.int8)
    ambiguous[1:6, 1:6] = 0
    ambiguous[1:6, 7:12] = 0
    qs = [[dict(kind=HOLLOW, size=5, color=2)],
          [dict(kind=SOLID, size=5, color=3), dict(kind=SOLID, size=3, color=4)]]
    assert len(_pack(ambiguous, 0, qs)) == 2
    # Separating the holes vertically makes the solid queue's size order choose
    # the first hole for the large solid and the second for the nested pair.
    ambiguous = np.full((13, 7), 8, dtype=np.int8)
    ambiguous[1:6, 1:6] = 0
    ambiguous[7:12, 1:6] = 0
    # The size order makes the large solid first, hence chooses the first hole.
    assert len(_pack(ambiguous, 0, qs)) == 1
    # Independent fixture: two equal solid holes require the visible color order.
    board = np.full((9, 5), 8, dtype=np.int8)
    board[1:4, 1:4] = 0
    board[5:8, 1:4] = 0
    queue = [dict(kind=SOLID, size=3, color=2), dict(kind=SOLID, size=3, color=4)]
    assert len(_pack(board, 0, [[], queue])) == 1
    normal = _pack(board, 0, [[], queue])[0][0]
    reversed_out = _pack(board, 0, [[], queue[::-1]])[0][0]
    assert not np.array_equal(normal, reversed_out)
    assert not _pack(board, 0, [[], queue[:-1]])
    # Input-only solver also handles valid negative space not made of separated
    # chambers: a ring and a large solid touching in one connected gap.
    qs = [[dict(kind=HOLLOW, size=5, color=2)],
          [dict(kind=SOLID, size=3, color=4), dict(kind=SOLID, size=4, color=6)]]
    joined = np.full((8, 11), 8, dtype=np.int8)
    joined[1:6, 1:6] = 0
    joined[3:7, 6:10] = 0
    answer = _pack(joined, 0, qs)
    assert len(answer) == 1
    scene = np.zeros((16, 26), dtype=np.int8)
    scene[1:9, 1:12] = joined
    for kind, rank, r, c in ((HOLLOW, 0, 1, 15), (SOLID, 0, 8, 16), (SOLID, 1, 11, 21)):
        piece = qs[kind][rank]
        for dr, dc in _cells(piece['size'], kind):
            scene[r+dr, c+dc] = piece['color']
    assert np.array_equal(solve(scene), answer[0][0])
    # Exchanging colors of equal-shaped source frames changes their destination
    # colors, even though the unordered inventory and every fixed cell agree.
    pair = puzzles[0]['train'][0]
    a = _analyze(pair['input'])
    first, second = a['queues'][HOLLOW][:2]
    assert first['size'] == second['size'] and first['color'] != second['color']
    swapped = pair['input'].copy()
    for p, color in ((first, second['color']), (second, first['color'])):
        for dr, dc in _cells(p['size'], HOLLOW):
            swapped[p['top']+dr, p['left']+dc] = color
    expected = pair['output'].copy()
    for kind, rank, r, c, p in _route_pieces(a):
        if kind == HOLLOW and rank < 2:
            color = second['color'] if rank == 0 else first['color']
            for dr, dc in _cells(p['size'], HOLLOW):
                expected[r+dr, c+dc] = color
    assert not np.array_equal(expected, pair['output'])
    assert np.array_equal(solve(swapped), expected)
    # Literal colors carry no hidden convention, and solving does not mutate
    # the caller's input or depend on previous calls.
    permutation = np.random.default_rng(99).permutation(10)
    original = pair['input'].copy()
    assert np.array_equal(solve(permutation[original]), permutation[pair['output']])
    assert np.array_equal(original, pair['input'])
    padded = np.pad(original, ((0, 1), (0, 0)), constant_values=a['background'])
    if max(padded.shape) <= 30:
        assert np.array_equal(solve(padded), pair['output'])
    # Validator rejects corruption of a fixed cell and loss of inventory.
    for fault in ('fixed', 'inventory'):
        p = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
             for s in ('train', 'test')}
        pair = p['test'][0]
        a = _analyze(pair['input'])
        if fault == 'fixed':
            pair['output'][0, 0] = a['background']
        else:
            piece = a['queues'][0][0]
            pair['input'][piece['top'], piece['left']] = a['background']
        try:
            validate_generated_puzzle(p)
        except ValueError:
            pass
        else:
            raise AssertionError("Validator accepted " + fault + " corruption")
    return {'seeds': 42, 'pairs': 210, 'input_shapes': len(shapes), 'board_shapes': len(boards),
            'hole_geometries': len(geometries), 'size_profiles': len(profiles),
            'inventory_counts': len(inventories), 'nesting_counts': sorted(nests)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
