"""Pattern-constrained rigid jigsaw: assemble and finish a pinwheel engraving.

Only Python's standard library and NumPy are used. Markers 9/8/7 mean
corner/hub/four spoke tips. All other roles are recovered from each input.
"""
from __future__ import annotations

from collections import Counter
from math import isqrt
import numpy as np

EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
CORNER, HUB, TIP = 9, 8, 7


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'Invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'Integer cells required')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'Colors must be 0..9')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(mask, neighbors=NEIGHBORS):
    points = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not points:
        return False
    stack = [points.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            p = r + dr, c + dc
            if p in points:
                points.remove(p)
                stack.append(p)
    return not points


def _components(a, bg):
    todo = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    cells.append(p)
        r0, c0 = np.min(cells, axis=0)
        r1, c1 = np.max(cells, axis=0)
        tile = np.full((r1-r0+1, c1-c0+1), -1, dtype=np.int8)
        for r, c in cells:
            tile[r-r0, c-c0] = a[r, c]
        result.append(tile)
    return result


def _read(a):
    a = _grid(a)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > max(np.delete(counts, bg)), 'Background must be uniquely most frequent')
    _require(bg < 7 and counts[CORNER] == counts[HUB] == 1 and counts[TIP] == 4,
             'Expected corner, hub and four tip markers')
    side = isqrt(int(np.sum(a != bg)))
    _require(side * side == int(np.sum(a != bg)) and 9 <= side <= 15, 'Tile area must be a legal square')
    pieces = _components(a, bg)
    _require(3 <= len(pieces) <= 5, 'Expected three to five separated rigid tiles')
    appearances = Counter(int(c) for p in pieces for c in np.unique(p) if 0 <= c < 7)
    inks = [c for c, n in appearances.items() if n >= 2]
    _require(len(inks) == 1, 'Ink is the unique non-marker color shared between tiles')
    ink = inks[0]
    carriers = []
    for p in pieces:
        colors = [int(c) for c in np.unique(p) if 0 <= c < 7 and c != ink]
        _require(len(colors) == 1 and np.sum(p == colors[0]) >= 3, 'Each tile has its own visible carrier')
        carriers.append(colors[0])
    _require(len(set(carriers)) == len(pieces), 'Distinct carriers required')
    return dict(grid=a, bg=bg, ink=ink, side=side, pieces=pieces, carriers=carriers)


def _pinwheel(length, shape, center):
    """Continue four clockwise stepped paths; clip segments, never paths."""
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    result = np.zeros(shape, dtype=bool)

    def paint(y, x, ny, nx):
        y, ny, x, nx = y+cy, ny+cy, x+cx, nx+cx
        if y == ny and 0 <= y < h:
            lo, hi = max(0, min(x, nx)), min(w-1, max(x, nx))
            if lo <= hi:
                result[y, lo:hi+1] = True
        elif x == nx and 0 <= x < w:
            lo, hi = max(0, min(y, ny)), min(h-1, max(y, ny))
            if lo <= hi:
                result[lo:hi+1, x] = True

    distance = max(cy, cx, h-1-cy, w-1-cx)
    for dy, dx in EDGE:
        y, x = radius*dy, radius*dx
        paint(dy, dx, y, x)
        dy, dx = dx, -dy
        for k in range(4*(distance//8+2)):
            for vy, vx, n in ((dy, dx, 2+4*k), (dx, -dy, radius-2), (dy, dx, radius)):
                ny, nx = y+n*vy, x+n*vx
                paint(y, x, ny, nx)
                y, x = ny, nx
            dy, dx = dx, -dy
    return result


def _interval_pinwheel(length, shape, center):
    """Cell-membership cross-check, using closed-form starts instead of walking."""
    y, x = np.indices(shape)
    y, x = y-center[0], x-center[1]
    result = np.zeros(shape, dtype=bool)
    bound = max(int(np.abs(y).max()), int(np.abs(x).max()))
    r = max(2, length)
    for _ in range(4):
        result |= (x == 0) & (-r <= y) & (y <= -1)
        for q in range(bound//8+3):
            starts = ((-r-8*q, -8*q), (-2-8*q, r+8*q+2),
                      (r+8*q+4, 8*q+4), (8*q+6, -r-8*q-6))
            for phase, ((sy, sx), (dy, dx)) in enumerate(zip(starts, ((0,1), (1,0), (0,-1), (-1,0)))):
                along = (y-sy)*dy+(x-sx)*dx
                across = (y-sy)*dx-(x-sx)*dy
                n = 16*q+4*phase+2
                result |= (across == 0) & (0 <= along) & (along <= n)
                result |= (along == n) & (0 <= across) & (across <= r-2)
                result |= (across == r-2) & (n <= along) & (along <= n+r)
        y, x = x, -y
    return result


def _cross_context(raw, ink):
    """Return None if markers are not all placed, False if already inconsistent."""
    hubs, tips = np.argwhere(raw == HUB), np.argwhere(raw == TIP)
    if len(hubs) != 1 or len(tips) != 4:
        return None
    cy, cx = map(int, hubs[0])
    deltas = [(int(r)-cy, int(c)-cx) for r, c in tips]
    lengths = {abs(r)+abs(c) for r, c in deltas}
    if len(lengths) != 1:
        return False
    length = lengths.pop()
    if not 2 <= length <= 4 or set(deltas) != {(-length,0), (length,0), (0,-length), (0,length)}:
        return False
    for dy, dx in EDGE:
        for d in range(1, length):
            value = int(raw[cy+d*dy, cx+d*dx])
            if value not in (-1, ink):
                return False
    return length, (cy, cx)


def _orientations(p):
    seen, result = set(), []
    for k in range(4):
        q = np.rot90(p, k).copy()
        if _key(q) not in seen:
            seen.add(_key(q))
            result.append(q)
    return result


def _packing_answers(facts, mode='pattern', limit=2):
    """Exact rigid cover with simultaneous seed/engraving compatibility.

    Pattern mode deduplicates completed colored answers, allowing harmless
    symmetric tile rotations. Seed mode deduplicates carrier ownership and
    ignores all outer engraving: it is an explicit parent-B ablation.
    """
    pieces, carriers, s, ink = (facts[k] for k in ('pieces', 'carriers', 'side', 'ink'))
    options = []
    anchor = next(i for i, p in enumerate(pieces) if np.any(p == CORNER))
    for i, p in enumerate(pieces):
        seen, placements = set(), []
        for q in _orientations(p):
            h, w = q.shape
            if h > s or w > s:
                continue
            if i == anchor:
                if q[0, 0] != CORNER:
                    continue
                origins = [(0, 0)]
            else:
                origins = [(r, c) for r in range(s-h+1) for c in range(s-w+1)]
            cells = [(int(r)*s+int(c), int(q[r,c])) for r,c in np.argwhere(q >= 0)]
            for r, c in origins:
                colored = tuple((pos+r*s+c, color) for pos, color in cells)
                if colored in seen:
                    continue
                seen.add(colored)
                mask = sum(1 << pos for pos, _ in colored)
                indices = np.array([pos for pos, _ in colored], dtype=int)
                colors = np.array([color for _, color in colored], dtype=np.int8)
                placements.append((mask, indices, colors))
        options.append(placements)
    full = (1 << (s*s))-1
    raw = np.full((s,s), -1, dtype=np.int8)
    owners = np.full((s,s), -1, dtype=np.int8)
    answers, signatures, masks = [], set(), {}

    def compatibility():
        context = _cross_context(raw, ink)
        if context is False:
            return False, None
        if context is not None and mode == 'pattern':
            if context not in masks:
                masks[context] = _pinwheel(context[0], (s,s), context[1])
            if np.any((raw == ink) & ~masks[context]):
                return False, None
        return True, context

    def search(occupied, remaining):
        okay, context = compatibility()
        if not okay:
            return
        if not remaining:
            if occupied != full or context is None:
                return
            if mode == 'pattern':
                out = raw.copy()
                path = masks[context]
                out[path & (out < TIP)] = ink
                signature = _key(out)
            else:
                out = owners.copy()
                signature = _key(out)
            if signature not in signatures:
                signatures.add(signature)
                answers.append(out)
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for mask, indices, colors in options[i]:
                if not mask & first or mask & occupied:
                    continue
                raw.flat[indices] = colors
                owners.flat[indices] = carriers[i]
                search(occupied | mask, rest)
                raw.flat[indices] = -1
                owners.flat[indices] = -1
                if len(answers) >= limit:
                    return

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for mask, indices, colors in options[anchor]:
        raw.flat[indices] = colors
        owners.flat[indices] = carriers[anchor]
        search(mask, rest)
        raw.flat[indices] = -1
        owners.flat[indices] = -1
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read(input_grid)
    answers = _packing_answers(facts)
    _require(len(answers) == 1, 'Expected exactly one pattern-compatible colored answer')
    return answers[0]


def _shape(rng):
    # Finite sampling bound with a legal, simple silhouette as a safety net.
    for _ in range(32):
        h, w = int(rng.integers(1, 5)), int(rng.integers(2, 6))
        mask = np.ones((h,w), dtype=bool)
        if h > 1 and w > 2 and rng.random() < .7:
            nh, nw = int(rng.integers(1,h)), int(rng.integers(1,w))
            mask[:nh, :nw] = False
        if 4 <= mask.sum() <= 14:
            return mask
    return np.ones((2,3), dtype=bool)


def _insert_options(shape, s, forbidden, path):
    result = []
    for oriented in _orientations(shape.astype(np.int8)):
        mask = oriented.astype(bool)
        h, w = mask.shape
        for r in range(s-h+1):
            for c in range(s-w+1):
                big = np.zeros((s,s), dtype=bool)
                big[r:r+h,c:c+w] = mask
                if np.any(big & forbidden) or not np.any(big & path) or np.sum(big & ~path) < 3:
                    continue
                result.append(big)
    return result


def _dilate(mask):
    a = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dr, dc in NEIGHBORS+((0,0),):
        result |= a[1+dr:1+dr+mask.shape[0],1+dc:1+dc+mask.shape[1]]
    return result


def _partition(rng, s, length, center, count, split_base):
    cy, cx = center
    path = _pinwheel(length, (s,s), center)
    protected = np.zeros((s,s), dtype=bool)
    protected[0,0] = protected[cy,cx] = True
    for dy, dx in EDGE:
        for d in range(1,length+1):
            protected[cy+d*dy,cx+d*dx] = True
    shape = _shape(rng)
    choices = _insert_options(shape, s, protected, path)
    if len(choices) < 2:
        return None
    labels = np.zeros((s,s), dtype=np.int8)
    blocked = protected.copy()
    ninsert = count-2 if split_base else count-1
    for i in range(1,ninsert+1):
        if i > 2:
            choices = _insert_options(_shape(rng), s, blocked, path)
        legal = [q for q in choices if not np.any(q & blocked)]
        if not legal:
            return None
        q = legal[int(rng.integers(len(legal)))]
        labels[q] = i
        blocked |= _dilate(q)
    if not _connected(labels == 0, EDGE):
        return None
    if split_base:
        # Random orthogonal seam: the support itself becomes two interlocks.
        axis = int(rng.integers(2))
        cut = int(rng.integers(max(2,s//3), min(s-2,2*s//3)+1))
        line = []
        for _ in range(s):
            cut = int(np.clip(cut+int(rng.integers(-1,2)), 2, s-2))
            line.append(cut)
        region = np.indices((s,s))[1] >= np.array(line)[:,None]
        if axis:
            region = region.T
        labels[(labels == 0) & region] = count-1
    if not all(_connected(labels == i, EDGE) and np.sum(labels == i) >= 4 for i in range(count)):
        return None
    return labels, path


def _make_tiles(labels, path, center, length, palette, rng):
    bg, ink, *carriers = palette
    raw = np.array(carriers, dtype=np.int8)[labels]
    cy, cx = center
    # Every insert carries its entire local engraving, while large support
    # shards carry sparse registration strokes and the complete cross seed.
    for label in range(len(carriers)):
        region = labels == label
        rate = float(rng.uniform(.06,.25)) if region.sum() > 25 else float(rng.uniform(.75,1.0))
        raw[region & path & (rng.random(path.shape) < rate)] = ink
    for dy, dx in EDGE:
        for d in range(1,length):
            raw[cy+d*dy,cx+d*dx] = ink
        raw[cy+length*dy,cx+length*dx] = TIP
    raw[cy,cx], raw[0,0] = HUB, CORNER
    output = raw.copy()
    output[path & (output < TIP)] = ink
    pieces = []
    for label in range(len(carriers)):
        mask = labels == label
        rr,cc = np.nonzero(mask)
        q = np.where(mask,raw,-1)[rr.min():rr.max()+1,cc.min():cc.max()+1].copy()
        if np.sum(q == carriers[label]) < 3 or not np.any(output == carriers[label]):
            return None
        pieces.append(q)
    return raw, output, pieces


def _scatter(rng, pieces, bg, force_rotation=False):
    """Mask-based placement; tight boxes can overlap if occupied cells stay apart."""
    rotations = list(map(int, rng.integers(4, size=len(pieces))))
    if force_rotation:
        anchor = next(i for i, p in enumerate(pieces) if np.any(p == CORNER))
        rotations[anchor] = int(rng.integers(1,4))
    oriented = [np.rot90(p,k).copy() for p,k in zip(pieces,rotations)]
    order = sorted(range(len(pieces)), key=lambda i: (-int(np.sum(oriented[i] >= 0)),float(rng.random())))
    canvas = np.full((30,30),bg,dtype=np.int8)
    blocked = np.zeros((30,30),dtype=bool)
    maxr = maxc = 0
    aspect = float(rng.uniform(.6,1.6))
    for i in order:
        p = oriented[i]
        h,w = p.shape
        choices = []
        mask = p >= 0
        for r in range(31-h):
            for c in range(31-w):
                if np.any(mask & blocked[r:r+h,c:c+w]):
                    continue
                nr,nc = max(maxr,r+h),max(maxc,c+w)
                score = nr*nc+abs(nr-aspect*nc)*3+float(rng.uniform(0,35))
                choices.append((score,r,c))
        if not choices:
            return None
        _,r,c = min(choices)
        view = canvas[r:r+h,c:c+w]
        view[mask] = p[mask]
        big = np.zeros((30,30),dtype=bool)
        big[r:r+h,c:c+w] = mask
        blocked |= _dilate(big)
        maxr,maxc = max(maxr,r+h),max(maxc,c+w)
    # Independent margins, including edges; keep the background role explicit.
    points = np.argwhere(canvas != bg)
    r0,c0 = points.min(axis=0)
    r1,c1 = points.max(axis=0)+1
    cropped = canvas[r0:r1,c0:c1]
    h,w = cropped.shape
    H,W = h+int(rng.integers(min(5,30-h)+1)), w+int(rng.integers(min(5,30-w)+1))
    nfg = int(np.sum(cropped != bg))
    largest = max(int(np.sum(cropped == color)) for color in range(10) if color != bg)
    while H*W-nfg <= largest:
        if H < 30 and (H <= W or W == 30):
            H += 1
        elif W < 30:
            W += 1
        else:
            return None
    result = np.full((H,W),bg,dtype=np.int8)
    r,c = int(rng.integers(H-h+1)),int(rng.integers(W-w+1))
    result[r:r+h,c:c+w] = cropped
    return result


def _pair(rng, s, length, center, count, split_base, force_rotation=False):
    for attempt in range(250):
        proposal = _partition(rng,s,length,center,count,split_base)
        if proposal is None:
            continue
        labels,path = proposal
        palette = list(map(int,rng.permutation(7)))[:count+2]
        built = _make_tiles(labels,path,center,length,palette,rng)
        if built is None:
            continue
        _,output,pieces = built
        bg,ink,*carriers = palette
        facts = dict(pieces=pieces,side=s,ink=ink,carriers=carriers)
        answers = _packing_answers(facts)
        if len(answers) != 1 or not np.array_equal(answers[0],output):
            continue
        # Outer winding must resolve an ambiguity left by rigid shapes AND
        # the complete marked cross. A mere cross locator is insufficient.
        alternatives = _packing_answers(facts,mode='seed')
        if len(alternatives) != 2 or not np.any((alternatives[0] != alternatives[1]) & ~path):
            continue
        inp = _scatter(rng,pieces,bg,force_rotation)
        if inp is None:
            continue
        try:
            _read(inp)
        except AssertionError:
            continue
        return dict(input=inp,output=output)
    raise RuntimeError(f'Could not make pattern-constrained jigsaw: S={s}, L={length}, pieces={count}, split={split_base}')


def _count(value,name,low,high):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain,ntest = _count(num_train,'num_train',2,10),_count(num_test,'num_test',1,4)
    if ntrain <= ntest:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    train_lengths = [2,3]+[int(rng.choice([2,3])) for _ in range(ntrain-2)]
    rng.shuffle(train_lengths)
    lengths = train_lengths+[4 if regime == 0 else int(rng.choice([2,3])) for _ in range(ntest)]
    puzzle,seen = dict(train=[],test=[]),set()
    for i,length in enumerate(lengths):
        split = 'train' if i < ntrain else 'test'
        if split == 'train':
            s = int(rng.integers(2*length+5,14))
            center = tuple(int(rng.integers(length+2,s-length-2)) for _ in range(2))
        elif regime == 0:
            s = int(rng.integers(14,16))
            center = tuple(int(rng.integers(length+1,s-length-1)) for _ in range(2))
        elif regime == 1:
            s = int(rng.integers(max(9,2*length+1),13))
            center = tuple(int(rng.choice([length,s-length-1])) for _ in range(2))
        else:
            s = int(rng.integers(11,16))
            pos = [int(rng.integers(length,s-length)) for _ in range(2)]
            pos[int(rng.integers(2))] = int(rng.choice([length,s-length-1]))
            center = tuple(pos)
        count = int(rng.integers(3,6))
        split_base = count >= 4 and bool(rng.random() < .6)
        for _ in range(8):
            pair = _pair(rng,s,length,center,count,split_base,force_rotation=(i == 0))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('Distinct input space exhausted')
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle,dict) and set(puzzle) == {'train','test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s],list) for s in puzzle), 'Splits must be lists')
    a,b = len(puzzle['train']),len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Illegal pair counts')
    seen,train_lengths = set(),set()
    rotation_witness = False
    for split in ('train','test'):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {'input','output'}, 'Invalid pair schema')
            facts = _read(pair['input'])
            inp,out = facts['grid'],_grid(pair['output'])
            _require(_key(inp) not in seen, 'Duplicate inputs')
            seen.add(_key(inp))
            _require(out.shape == (facts['side'],)*2 and out[0,0] == CORNER, 'Wrong square or anchor')
            _require(not np.any(out == facts['bg']), 'Exterior background must disappear')
            _require(np.sum(out == TIP) == 4 and np.sum(out == HUB) == np.sum(out == CORNER) == 1, 'Preserve markers')
            context = _cross_context(out,facts['ink'])
            _require(context not in (None,False), 'Bad output cross')
            length,center = context
            analytic = _interval_pinwheel(length,out.shape,center)
            _require(np.array_equal(out == facts['ink'],analytic & (out < TIP)), 'Incorrect pinwheel ink geometry')
            _require(np.sum(out == facts['ink']) > np.sum(inp == facts['ink']), 'Incomplete engraving must grow')
            for color in facts['carriers']:
                _require(0 < np.sum(out == color) <= np.sum(inp == color), 'Carrier changes only to ink')
            answers = _packing_answers(facts)
            _require(len(answers) == 1 and np.array_equal(answers[0],out), 'Incorrect or ambiguous joint assembly')
            alternatives = _packing_answers(facts,'seed')
            _require(len(alternatives) == 2, 'Outer pinwheel must resolve a shape/seed ambiguity')
            _require(bool(np.any((alternatives[0] != alternatives[1]) & ~analytic)),
                     'The ambiguity must change visible material off the engraving')
            if split == 'train':
                train_lengths.add(length)
                anchor = next(p for p in facts['pieces'] if np.any(p == CORNER))
                rotation_witness |= bool(anchor[0,0] != CORNER)
                _require(min(center[0]-length,center[1]-length,out.shape[0]-1-center[0]-length,out.shape[1]-1-center[1]-length) >= 2,
                         'Training must expose complete local turns')
    _require({2,3} <= train_lengths, 'Demonstrate zero and positive jogs')
    _require(rotation_witness, 'Demonstrate a required rigid quarter-turn')


def _equal(a,b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k],q[k]) for p,q in zip(a[s],b[s]) for k in ('input','output')) for s in ('train','test'))


def self_test():
    records,puzzles = [],[]
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train','test'):
            for pair in p[split]:
                f = _read(pair['input'])
                length,center = _cross_context(pair['output'],f['ink'])
                records.append((split,f,length,center,pair))
    _require(len({tuple(_key(pair['input']) for s in p for pair in p[s]) for p in puzzles}) == 42, 'Seeds must vary')
    _require({3,4,5} == {len(f['pieces']) for _,f,_,_,_ in records}, 'Vary tile counts')
    _require({2,3,4} == {length for _,_,length,_,_ in records}, 'Vary winding scales')
    _require(len({pair['input'].shape for _,_,_,_,pair in records}) >= 25, 'Vary scatter geometry')
    _require(len({(f['bg'],f['ink']) for _,f,_,_,_ in records}) >= 25, 'Vary semantic palette roles')
    _require({9,10,11,12,13,14,15} <= {f['side'] for _,f,_,_,_ in records}, 'Vary target scales')
    silhouettes, large_counts, bars, notched = set(), Counter(), 0, 0
    for _,f,_,_,_ in records:
        large_counts[sum(np.sum(q >= 0) > 25 for q in f['pieces'])] += 1
        for q in f['pieces']:
            mask = (q >= 0).astype(np.int8)
            silhouettes.add(min(_key(r) for r in _orientations(mask)))
            bars += min(q.shape) == 1
            notched += not np.all(mask)
    _require(len(silhouettes) >= 100, 'Vary actual tile silhouettes')
    _require(large_counts[1] >= 10 and sum(n for k,n in large_counts.items() if k > 1) >= 10,
             'Exercise both whole and split support geometry')
    _require(bars >= 10 and notched >= 100, 'Exercise simple bars and interlocking shapes')
    _require({f['bg'] for _,f,_,_,_ in records} == set(range(7)) and
             {f['ink'] for _,f,_,_,_ in records} == set(range(7)), 'Every free color must occupy both roles')
    for length in range(2,5):
        for center in ((length,length),(14-length,14-length),(7,7)):
            m = _pinwheel(length,(15,15),center)
            _require(np.array_equal(m,_interval_pinwheel(length,(15,15),center)), 'Path/interval disagreement')
            big = _interval_pinwheel(length,(61,61),(30,30))
            r,c = 30-center[0],30-center[1]
            _require(np.array_equal(m,big[r:r+15,c:c+15]), 'Lost boundary re-entry')
            _require(np.array_equal(np.rot90(m),_pinwheel(length,(15,15),(14-center[1],center[0]))), 'Quarter-turn geometry failed')
            _require(not np.any(m[:-1,:-1]&m[1:,:-1]&m[:-1,1:]&m[1:,1:]), 'Corridor width failed')
    _require(_equal(puzzles[0],generate_puzzle(0)), 'Nondeterministic generation')
    first = puzzles[0]['train'][0]
    for k in (1,2,3):
        _require(np.array_equal(solve(np.rot90(first['input'],k)),first['output']),
                 'Scatter rotation must preserve the anchored answer')
    recolor = np.arange(10,dtype=np.int8)
    recolor[:7] = np.arange(6,-1,-1)
    _require(np.array_equal(solve(recolor[first['input']]),recolor[first['output']]),
             'Locally visible palette roles must survive color permutation')
    for nt,ne in ((2,1),(10,1),(10,4)):
        validate_generated_puzzle(generate_puzzle(4242,nt,ne))
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.0,1),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    for mutation in ('marker','missing_ink','carrier_swap'):
        bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = bad['test'][0]
        f = _read(pair['input'])
        out = pair['output']
        if mutation == 'marker':
            out[0,0] = HUB
        elif mutation == 'missing_ink':
            point = tuple(np.argwhere(out == f['ink'])[0])
            out[point] = f['carriers'][0]
        else:
            points = [tuple(np.argwhere(out == c)[0]) for c in f['carriers'][:2]]
            out[points[0]],out[points[1]] = int(out[points[1]]),int(out[points[0]])
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'Corrupt {mutation} accepted')
    return dict(seeds=42,pairs=len(records),input_shapes=len({p['input'].shape for *_,p in records}),
                palette_roles=len({(f['bg'],f['ink']) for _,f,_,_,_ in records}),
                tile_silhouettes=len(silhouettes),bars=int(bars),notched_tiles=int(notched),
                support_piece_counts={int(k):int(v) for k,v in large_counts.items()})
