"""Fit an exact inventory into one of two latent pinwheel tracks.

The rectangle and hollow cross are visible. Neutral junctions are compatible
with both hands; only translation of the loose pieces selects a hand. No
parent module, example bank, RNG state, or training labels enter solve().
"""
from collections import Counter
from functools import lru_cache

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalize(cells):
    y = min(r for r, c in cells)
    x = min(c for r, c in cells)
    return tuple(sorted((r - y, c - x) for r, c in cells))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in _DIRECTIONS:
                p = y + dy, x + dx
                if p in remaining:
                    remaining.remove(p)
                    todo.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _cross(length, shape, center):
    mask = np.zeros(shape, dtype=bool)
    cy, cx = center
    for dy, dx in _DIRECTIONS:
        for d in range(1, length + 1):
            mask[cy + d * dy, cx + d * dx] = True
    return mask


def _winding(length, shape, center, hand):
    """A's stepped, four-phase recurrence, reflected when hand == -1.

    On the infinite plane, extend each spoke to R=max(2,L). At stage k,
    go forward 4k+2, inward R-2, then forward R; turn by a quarter turn.
    Clip segments, never terminate a path at its first boundary crossing.
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    mask = np.zeros(shape, dtype=bool)

    def paint(y, x, ny, nx):
        y, ny, x, nx = y + cy, ny + cy, x + cx, nx + cx
        if y == ny and 0 <= y < h:
            lo, hi = max(0, min(x, nx)), min(w - 1, max(x, nx))
            if lo <= hi:
                mask[y, lo:hi + 1] = True
        elif x == nx and 0 <= x < w:
            lo, hi = max(0, min(y, ny)), min(h - 1, max(y, ny))
            if lo <= hi:
                mask[lo:hi + 1, x] = True

    distance = max(cy, cx, h - 1 - cy, w - 1 - cx)
    for dy, dx in _DIRECTIONS:
        y, x = radius * dy, radius * dx
        paint(dy, dx, y, x)
        dy, dx = hand * dx, -hand * dy
        for k in range(4 * (distance // 8 + 2)):
            for vy, vx, n in ((dy, dx, 4 * k + 2),
                             (hand * dx, -hand * dy, radius - 2),
                             (dy, dx, radius)):
                ny, nx = y + n * vy, x + n * vx
                paint(y, x, ny, nx)
                y, x = ny, nx
            dy, dx = hand * dx, -hand * dy
    return mask


def _membership(length, shape, center, hand):
    """Closed-form segment membership for checking the path implementation."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy = yy - center[0]
    xx = hand * (xx - center[1])
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    mask = np.zeros(shape, dtype=bool)
    for rotation in range(4):
        mask |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius - 8*q, -8*q), (-2 - 8*q, radius + 8*q + 2),
                      (radius + 8*q + 4, 8*q + 4), (8*q + 6, -radius - 8*q - 6))
            for s, ((y, x), (dy, dx)) in enumerate(zip(starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (yy - y)*dy + (xx - x)*dx
                across = (yy - y)*dx - (xx - x)*dy
                n = 16*q + 4*s + 2
                mask |= (across == 0) & (0 <= along) & (along <= n)
                mask |= (along == n) & (0 <= across) & (across <= radius - 2)
                mask |= (across == radius - 2) & (n <= along) & (along <= n + radius)
        yy, xx = xx, -yy
    return mask


class _SearchLimit(ValueError):
    pass


def _pack(holes, pieces, node_limit=None):
    """Return up to two DISTINCT colored exact covers, using every item once.

    Shapes are not rotated/reflected. Equal same-color pieces are counted,
    so their exchange is not a spurious second answer. Memoization is local
    to this proof search, never a cache of previously generated examples.
    """
    cells = sorted(map(tuple, np.argwhere(holes).tolist()))
    if sum(len(shape) for color, shape in pieces) != len(cells):
        return (), 0
    inventory = Counter((int(color), _normalize(shape)) for color, shape in pieces)
    types = sorted(inventory)
    positions = {cell: i for i, cell in enumerate(cells)}
    by_cell = [[] for cell in cells]
    h, w = holes.shape
    for t, (color, shape) in enumerate(types):
        sh, sw = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        for y in range(h - sh + 1):
            for x in range(w - sw + 1):
                placed = [(y + dy, x + dx) for dy, dx in shape]
                if not all(p in positions for p in placed):
                    continue
                ids = [positions[p] for p in placed]
                bits = sum(1 << i for i in ids)
                paint = sum((color + 1) << (4*i) for i in ids)
                for i in ids:
                    by_cell[i].append((t, bits, paint))
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if node_limit is not None and nodes > node_limit:
            raise _SearchLimit("Exact-cover proof exceeded its node bound")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            choices = [m for m in by_cell[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, cover, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ cover, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(cells)) - 1, tuple(inventory[t] for t in types))
    answers = []
    for code in codes:
        answer = np.full(holes.shape, -1, dtype=np.int8)
        for p, i in positions.items():
            answer[p] = ((code >> (4*i)) & 15) - 1
        answers.append(answer)
    return tuple(answers), nodes


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(a == bg) > a.size // 2, "Background must be a strict majority")
    frames = []
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        rr, cc = np.where(a == color)
        y, x, bottom, right = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
        if bottom - y < 8 or right - x < 8:
            continue
        box = a[y:bottom + 1, x:right + 1]
        if (np.all(box[0] == color) and np.all(box[-1] == color)
                and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
            frames.append((color, y, x, bottom, right))
    _require(len(frames) == 1, "Expected one marked rectangular window")
    frame, y, x, bottom, right = frames[0]
    board = a[y + 1:bottom, x + 1:right].copy()
    seeds = []
    for color in np.unique(board):
        color = int(color)
        if color in (bg, frame):
            continue
        mask = board == color
        n = int(mask.sum())
        if n % 4 or not 1 <= n // 4 <= 6 or np.count_nonzero(a == color) != n:
            continue
        length = n // 4
        rr = np.flatnonzero(mask.sum(axis=1) == 2 * length)
        cc = np.flatnonzero(mask.sum(axis=0) == 2 * length)
        if len(rr) != 1 or len(cc) != 1:
            continue
        cy, cx = int(rr[0]), int(cc[0])
        if min(cy, cx, board.shape[0]-1-cy, board.shape[1]-1-cx) < length:
            continue
        if board[cy, cx] == bg and np.array_equal(mask, _cross(length, board.shape, (cy, cx))):
            seeds.append((color, length, (cy, cx)))
    _require(len(seeds) == 1, "Expected one complete hollow cross")
    seed_color, length, center = seeds[0]
    _require(set(map(int, np.unique(board))) <= {bg, frame, seed_color}, "Unrecognized fixed cells")
    outside = a != bg
    outside[y:bottom + 1, x:right + 1] = False
    components = _components(outside)
    _require(components, "Missing inventory")
    pieces = []
    for cells in components:
        palette = {int(a[p]) for p in cells}
        _require(len(palette) == 1 and not palette & {bg, frame, seed_color},
                 "Exterior pieces must be separated monochrome payloads")
        pieces.append((palette.pop(), _normalize(cells)))
    cw = _winding(length, board.shape, center, 1)
    ccw = _winding(length, board.shape, center, -1)
    _require(int(cw.sum()) == int(ccw.sum()), "Both latent windings must have equal area")
    fixed = board != bg
    _require(np.all((cw & ccw)[fixed]), "Fixed junctions must permit both hands")
    return dict(array=a, background=bg, frame=frame, seed_color=seed_color,
                length=length, center=center, bbox=(y, x, bottom, right),
                board=board, fixed=fixed, pieces=pieces, components=components,
                masks={1: cw, -1: ccw})


def _infer(info):
    feasible = []
    for hand, mask in info['masks'].items():
        answers, nodes = _pack(mask & ~info['fixed'], info['pieces'])
        if answers:
            _require(len(answers) == 1, "Inventory has multiple colored completions")
            result = info['board'].copy()
            holes = mask & ~info['fixed']
            result[holes] = answers[0][holes]
            feasible.append((hand, result, nodes))
    _require(len(feasible) == 1, "Inventory must select exactly one winding direction")
    return feasible[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the window, seed, two windings, and unique colored assembly."""
    return _infer(_parse(input_grid))[1].copy()


def _split_path(cells, rng, maximum):
    remaining = set(cells)
    endpoints = [p for p in remaining if sum((p[0]+dy, p[1]+dx) in remaining
                                           for dy, dx in _DIRECTIONS) <= 1]
    if not endpoints:
        return [cells] if len(cells) <= maximum else []
    ordered = []
    p = min(endpoints)
    while remaining:
        ordered.append(p)
        remaining.remove(p)
        neighbors = [(p[0]+dy, p[1]+dx) for dy, dx in _DIRECTIONS
                     if (p[0]+dy, p[1]+dx) in remaining]
        if not neighbors:
            break
        p = neighbors[0]
    if remaining:
        return []
    result = []
    while ordered:
        n = len(ordered)
        cut = int(rng.integers(2, min(maximum, n)+1)) if n > maximum else n
        if 4 <= n <= maximum and rng.random() < .55:
            cut = int(rng.integers(2, n-1))
        result.append(tuple(ordered[:cut]))
        ordered = ordered[cut:]
    return result


def _scatter(rng, board, pieces, bg, frame):
    """Variable window origin and scattered clues, including boundary clues."""
    bh, bw = board.shape
    fh, fw = bh + 2, bw + 2
    for attempt in range(10):
        ch = 30 if attempt >= 7 else int(rng.integers(max(24, fh+6), 31))
        cw = 30 if attempt >= 7 else int(rng.integers(max(24, fw+6), 31))
        y, x = int(rng.integers(ch-fh+1)), int(rng.integers(cw-fw+1))
        a = np.full((ch, cw), bg, dtype=np.int8)
        a[y:y+fh, x:x+fw] = frame
        a[y+1:y+fh-1, x+1:x+fw-1] = board
        blocked = np.zeros((ch, cw), dtype=bool)
        blocked[max(0,y-1):min(ch,y+fh+1), max(0,x-1):min(cw,x+fw+1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), float(rng.random())))
        for i in order:
            color, shape = pieces[i]
            sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            candidates = [(r, c) for r in range(ch-sh+1) for c in range(cw-sw+1)
                          if all(not blocked[r+dy, c+dx] for dy, dx in shape)]
            if not candidates:
                break
            edges = [(r,c) for r,c in candidates if r == 0 or c == 0 or r+sh == ch or c+sw == cw]
            choices = edges if edges and rng.random() < .25 else candidates
            r, c = choices[int(rng.integers(len(choices)))]
            for dy, dx in shape:
                rr, cc = r+dy, c+dx
                a[rr, cc] = color
                for ar, ac in ((rr,cc), (rr-1,cc), (rr+1,cc), (rr,cc-1), (rr,cc+1)):
                    if 0 <= ar < ch and 0 <= ac < cw:
                        blocked[ar, ac] = True
        else:
            if np.count_nonzero(a == bg) > a.size // 2:
                return a
    return None


def _make_pair(rng, length, hand, palette, regime, teaching=False):
    low = max(9, 2*length+3)
    for attempt in range(1200):
        if regime == 'wide':
            h, w = int(rng.integers(low, max(low+1, 13))), int(rng.integers(max(17,low), 19))
            if rng.random() < .5:
                h, w = w, h
        elif regime == 'tight':
            h, w = (int(rng.integers(low, max(low+1, 14))) for axis in range(2))
        elif regime == 'sparse':
            h, w = (int(rng.integers(max(11,low), max(low+1,17))) for axis in range(2))
        else:
            h, w = (int(rng.integers(low, max(low+1, 17))) for axis in range(2))
        cy, cx = int(rng.integers(length, h-length)), int(rng.integers(length, w-length))
        if regime == 'tight':
            if rng.random() < .5:
                cy = int(rng.choice([length,h-1-length]))
            else:
                cx = int(rng.choice([length,w-1-length]))
        if teaching:
            cy, cx = (h-1)//2, (w-1)//2
        cw, ccw = (_winding(length, (h,w), (cy,cx), sign) for sign in (1,-1))
        if cw.sum() != ccw.sum() or np.array_equal(cw,ccw):
            continue
        seed = _cross(length, (h,w), (cy,cx))
        probability = float(rng.uniform(.23,.47) if regime == 'sparse' else rng.uniform(.36,.66))
        fixed = seed | ((cw & ccw) & (rng.random((h,w)) < probability))
        chosen, other = (cw,ccw) if hand == 1 else (ccw,cw)
        gaps = _components(chosen & ~fixed)
        fragments = []
        for gap in gaps:
            cut = _split_path(gap, rng, 11 if regime == 'sparse' else 8)
            if not cut:
                break
            fragments.extend(cut)
        else:
            if not 6 <= len(fragments) <= 24:
                continue
            if regime == 'sparse' and max(map(len,fragments)) < 9:
                continue
            signatures = [_normalize(cells) for cells in fragments]
            unique = sorted(set(signatures))
            if len(unique) < 3 or not any(max(r for r,c in s)>0 and max(c for r,c in s)>0 for s in unique):
                continue
            payload_count = int(rng.integers(2,min(5,len(unique))+1))
            order = rng.permutation(len(unique))
            mapping = {unique[int(j)]:int(palette[3+i%payload_count]) for i,j in enumerate(order)}
            pieces = [(mapping[s],s) for s in signatures]
            bg,frame,ink = map(int,palette[:3])
            board = np.full((h,w),bg,dtype=np.int8)
            board[fixed] = frame
            board[seed] = ink
            output = board.copy()
            for cells,s in zip(fragments,signatures):
                for p in cells:
                    output[p] = mapping[s]
            # A shared negative-space component must take several colors and
            # cannot be completed by a single matching exterior template.
            if not any(len({int(output[p]) for p in gap}) > 1 and
                       _normalize(gap) not in unique for gap in gaps):
                continue
            try:
                answers,nodes = _pack(chosen & ~fixed,pieces,node_limit=18000)
                if len(answers) != 1 or not np.array_equal(answers[0][chosen & ~fixed],output[chosen & ~fixed]):
                    continue
                wrong,wrong_nodes = _pack(other & ~fixed,pieces,node_limit=18000)
                if wrong:
                    continue
            except _SearchLimit:
                continue
            canvas = _scatter(rng,board,pieces,bg,frame)
            if canvas is not None:
                return {'input':canvas,'output':output}
    raise RuntimeError('Bounded construction could not find a unique pinwheel assembly')


def _count(value, name, low, high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt,ne = _count(num_train,'num_train',2,10), _count(num_test,'num_test',1,4)
    _require(nt > ne,'num_train must exceed num_test')
    _require(isinstance(seed,(int,np.integer)) and not isinstance(seed,(bool,np.bool_)) and seed >= 0,
             'seed must be a non-negative integer')
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(['wide','sparse','tight']))
    common_palette = rng.permutation(10)
    recolor = bool(rng.random() < .8)
    first_hand = int(rng.choice([-1,1]))
    lengths = [2,3]
    lengths.extend(int(rng.choice([1,2,3,4,5,6])) for i in range(nt-2))
    hands = [first_hand,-first_hand]
    hands.extend(int(rng.choice([-1,1])) for i in range(nt-2))
    puzzle,seen = {'train':[],'test':[]},set()
    for split,count in (('train',nt),('test',ne)):
        for i in range(count):
            length = lengths[i] if split == 'train' else int(rng.choice(sorted(set(lengths))))
            hand = hands[i] if split == 'train' else int(rng.choice([-1,1]))
            mode = 'ordinary' if split == 'train' else regime
            palette = rng.permutation(10) if recolor else common_palette
            for retry in range(8):
                pair = _make_pair(rng,length,hand,palette,mode,teaching=(split=='train' and i<2))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct a distinct input')
    return puzzle


def _analyze(pair):
    a,b = _grid(pair['input']),_grid(pair['output'])
    info = _parse(a)
    hand,answer,nodes = _infer(info)
    _require(np.array_equal(answer,b),'Incorrect colored assembly')
    _require(b.shape == info['board'].shape,'Output must be the inside of the frame')
    bg,fixed = info['background'],info['fixed']
    mask = _membership(info['length'],b.shape,info['center'],hand)
    _require(np.array_equal(mask,info['masks'][hand]),'Geometric calculations disagree')
    _require(np.array_equal(b != bg,mask),'Output must contain exactly the completed winding')
    _require(np.array_equal(b[fixed],info['board'][fixed]),'Fixed ink/junctions changed')
    _require(b[info['center']] == bg,'The seed center must stay hollow')
    supplied = Counter()
    types = Counter(info['pieces'])
    for color,shape in info['pieces']:
        supplied[color] += len(shape)
    _require(supplied == Counter(map(int,b[mask & ~fixed])),'Payload color inventory was not conserved')
    _require(sum(supplied.values()) < int(np.count_nonzero(~fixed)),
             'Inventory must leave intentional corridor space')
    _require(not np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]),
             'Tracks cannot contain a filled 2-by-2 block')
    signatures = {s for c,s in info['pieces']}
    gaps = _components(mask & ~fixed)
    shared = [g for g in gaps if len({int(b[p]) for p in g})>1 and _normalize(g) not in signatures]
    _require(shared,'Need a multicolor gap that a single-template shortcut cannot fill')
    cy,cx = info['center']
    h,w = b.shape
    return dict(hand=hand,length=info['length'],board_shape=b.shape,canvas_shape=a.shape,
                center=info['center'],background=bg,piece_count=len(info['pieces']),
                shapes=signatures,piece_sizes={len(s) for c,s in info['pieces']},
                repeated=any(n>1 for n in types.values()),shared_gaps=len(shared),
                peg_count=int(np.count_nonzero(info['board']==info['frame'])),nodes=nodes,
                clipped=min(cy,cx,h-1-cy,w-1-cx)==info['length'],
                layout=(b.shape,info['length'],info['center'],hand,fixed.tobytes()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle)=={'train','test'},'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s],list) for s in puzzle),'Splits must be lists')
    nt,ne = _count(len(puzzle['train']),'train length',2,10),_count(len(puzzle['test']),'test length',1,4)
    _require(nt>ne,'Train count must exceed test count')
    seen,metrics = set(),{'train':[],'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair)=={'input','output'},'Invalid pair keys')
            ik = _key(_grid(pair['input']))
            _require(ik not in seen,'Duplicate input')
            seen.add(ik)
            metrics[split].append(_analyze(pair))
    _require({m['hand'] for m in metrics['train']}=={-1,1},'Training must teach inventory-selected handedness')
    _require({2,3} <= {m['length'] for m in metrics['train']},'Training must expose ordinary and jogged windings')
    _require({m['length'] for m in metrics['test']} <= {m['length'] for m in metrics['train']},
             'Tests must use demonstrated spoke lengths')
    return None


def self_test():
    """Bounded geometry, inventory, mutation, counts and 42-seed checks."""
    boards,canvases,layouts,shapes,sizes,bgs,counts,hands,lengths = (set() for i in range(9))
    signatures = set()
    shared = repeated = clipped = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p[g]) for s in ('train','test') for p in puzzle[s] for g in ('input','output')))
        local = []
        for split in ('train','test'):
            for pair in puzzle[split]:
                m = _analyze(pair)
                local.append(m)
                boards.add(m['board_shape']);canvases.add(m['canvas_shape']);layouts.add(m['layout'])
                shapes.update(m['shapes']);sizes.update(m['piece_sizes']);bgs.add(m['background'])
                counts.add(m['piece_count']);hands.add(m['hand']);lengths.add(m['length'])
                shared += m['shared_gaps'];repeated += m['repeated'];clipped += m['clipped']
        _require(len({m['layout'] for m in local})==5,'Structures repeated within a puzzle')
    _require(len(signatures)==42 and len(layouts)>=200,'Insufficient structural variation')
    _require(len(boards)>=25 and len(canvases)>=15 and len(shapes)>=40,'Geometry/shape diversity collapsed')
    _require(len(counts)>=8 and len(sizes)>=7 and bgs==set(range(10)),'Count/palette variation collapsed')
    _require(hands=={-1,1} and {1,2,3,4,5,6}<=lengths,'Missing winding modes')
    _require(shared>=210 and repeated>=100 and clipped>0,'Missing assembly/clipping interactions')
    for length in range(1,7):
        for hand in (-1,1):
            shape,center=(23,27),(length,26-length)
            mask=_winding(length,shape,center,hand)
            _require(np.array_equal(mask,_membership(length,shape,center,hand)),'Recurrence drift')
            big=_winding(length,(75,75),(37,37),hand)
            top,left=37-center[0],37-center[1]
            _require(np.array_equal(mask,big[top:top+shape[0],left:left+shape[1]]),'Lost clipped re-entry')
            rc=(shape[1]-1-center[1],center[0])
            _require(np.array_equal(np.rot90(mask),_winding(length,shape[::-1],rc,hand)),'Quarter-turn drift')
    _require(np.array_equal(_winding(1,(19,19),(9,9),1),_winding(2,(19,19),(9,9),1)),
             'Short spokes must extend to radius two')
    # Two different colors in interchangeable slots are ambiguous, whereas
    # identical-color exchanges produce exactly one colored answer.
    slots=np.array([[True,False,True]])
    _require(len(_pack(slots,[(1,((0,0),)),(2,((0,0),))])[0])==2,'Missed color-swap ambiguity')
    _require(len(_pack(slots,[(1,((0,0),)),(1,((0,0),))])[0])==1,'Identical swaps are not ambiguity')
    elbow=np.array([[True,True,True],[True,False,False]])
    _require(not _pack(elbow,[(1,((0,0),(0,1),(0,2),(1,2)))])[0],'Illegally reflected an elbow')
    original=generate_puzzle(0)
    for pair in original['train']:
        info=_parse(pair['input'])
        hand,answer,nodes=_infer(info)
        # Removing A's latent track restriction cannot assemble a solid board.
        _require(not _pack(~info['fixed'],info['pieces'])[0],
                 'Ablating the track mask unexpectedly completed the board')
        # Removing B loses both the inferred direction and the colored inventory.
        uniform=np.where(info['masks'][hand],info['seed_color'],info['background'])
        _require(not np.array_equal(uniform,pair['output']),'Uniform A-only completion matched')
        if info['length']==3:
            ordinary=_winding(2,info['board'].shape,info['center'],hand) | _cross(3,info['board'].shape,info['center'])
            _require(not np.array_equal(ordinary,info['masks'][hand]),'Jog-free shortcut was not distinguished')
    pair=original['test'][0]
    for turns in (1,2,3):
        _require(np.array_equal(solve(np.rot90(pair['input'],turns)),np.rot90(pair['output'],turns)),
                 'Rotating a scene changed the assembly rule')
    _require(np.array_equal(solve(np.fliplr(pair['input'])),np.fliplr(pair['output'])),
             'Reflecting a scene failed to reverse inferred handedness')
    generate_puzzle(103)
    again=generate_puzzle(0)
    _require(all(np.array_equal(p[g],q[g]) for s in original for p,q in zip(original[s],again[s])
                 for g in ('input','output')),'RNG depends on intervening calls')
    for nt in range(2,11):
        for ne in range(1,min(4,nt-1)+1):
            p=generate_puzzle(19,nt,ne)
            _require(len(p['train'])==nt and len(p['test'])==ne,'Ignored explicit counts')
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    for mutation in ('missing','fixed','corridor','wrong_hand'):
        bad={s:[{g:np.asarray(v).copy() for g,v in p.items()} for p in original[s]] for s in original}
        p=bad['test'][0]
        info=_parse(p['input'])
        hand,answer,nodes=_infer(info)
        if mutation=='missing':
            p['input'][info['components'][0][0]]=info['background']
        elif mutation=='fixed':
            p['output'][tuple(np.argwhere(info['fixed'])[0])]=info['background']
        elif mutation=='corridor':
            cell=next(tuple(q) for q in np.argwhere(~info['masks'][hand]) if tuple(q)!=info['center'])
            p['output'][cell]=info['seed_color']
        else:
            p['output']=np.where(info['masks'][-hand],info['seed_color'],info['background']).astype(np.int8)
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Validator accepted '+mutation)


if __name__ == '__main__':
    self_test()
    print('self_test passed')
