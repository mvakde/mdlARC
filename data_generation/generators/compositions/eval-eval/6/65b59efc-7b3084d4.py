"""Seam-guided textured jigsaws. No parent code or stored puzzles are imported.

The library bitmap is in its carrier piece's current coordinate frame. It is
therefore rotated with that piece during assembly, not painted on afterwards.
"""
from __future__ import annotations

from math import isqrt
import numpy as np

BACKGROUND, SEPARATOR, ANCHOR = 0, 8, 9
EDGE = ((-1, 0), (0, 1), (1, 0), (0, -1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "Grids must be 2D with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask, neighbors=EIGHT):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    part.add(p)
                    stack.append(p)
        parts.append(sorted(part))
    return parts


def _tight(a):
    rr, cc = np.nonzero(a >= 0)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _edges(motif):
    # Opposing edges are compared in the same screen-coordinate order.
    def code(v):
        return sum(int(x) << i for i, x in enumerate(v))
    return (code(motif[0]), code(motif[:, -1]),
            code(motif[-1]), code(motif[:, 0]))


def _read_input(value):
    a = _grid(value)
    _require(np.all(a[0] == SEPARATOR), "Library must start with a full separator")
    lines = [r for r in range(1, a.shape[0]) if np.all(a[r] == SEPARATOR)]
    _require(len(lines) == 2, "Expected library divider and key divider")
    q = lines[0] - 1
    _require(q in (4, 5) and lines[1] == q + 3, "Invalid library scale/key rows")
    borders = [c for c in range(a.shape[1])
               if np.all(a[1:q+1, c] == SEPARATOR)]
    _require(len(borders) >= 3 and borders[0] == 0, "Missing library panel borders")
    k = len(borders) - 1
    _require(2 <= k <= 4 and borders == [j*(q+1) for j in range(k+1)],
             "Library must have two to four equal panels")
    _require(borders[-1]+1 <= 30, "Library too wide")
    templates, targets, order = {}, {}, []
    for j in range(k):
        c = 1 + j*(q+1)
        panel = a[1:q+1, c:c+q]
        vals = set(map(int, np.unique(panel))) - {BACKGROUND}
        _require(len(vals) == 1, "Each motif has one visible identity")
        label = next(iter(vals))
        _require(1 <= label <= 7 and label not in templates, "Invalid/repeated identity")
        swatch = a[q+2, c:c+q]
        _require(np.all(swatch == swatch[0]) and 1 <= swatch[0] <= 8,
                 "Key swatches must be visible, constant, and nonmarker")
        templates[label] = panel == label
        targets[label] = int(swatch[0])
        order.append(label)
    # No unparsed payload in header padding, panel borders, or key gaps.
    expected = np.zeros_like(a[:q+4])
    expected[[0, q+1, q+3], :] = SEPARATOR
    expected[1:q+1, borders] = SEPARATOR
    for j, label in enumerate(order):
        c = 1+j*(q+1)
        expected[1:q+1, c:c+q] = np.where(templates[label], label, 0)
        expected[q+2, c:c+q] = targets[label]
    _require(np.array_equal(expected, a[:q+4]), "Unexpected library metadata")
    field = a[q+4:]
    _require(field.size > 0 and np.count_nonzero(field == ANCHOR) == 1,
             "Exactly one lower-field anchor is required")
    _require(set(map(int, np.unique(field))) <= {0, ANCHOR, *order},
             "Unidentified lower-field color")
    parts = _components(field != 0)
    _require(len(parts) == k, "One separated rigid piece per identity is required")
    pieces, identities, anchor_index = [], [], None
    for cells in parts:
        vals = {int(field[r, c]) for r, c in cells} - {ANCHOR}
        _require(len(vals) == 1 and len(cells) >= 2, "Each piece needs one carrier and at least two cells")
        label = next(iter(vals))
        _require(label not in identities and label in templates, "Repeated/unknown piece identity")
        piece = np.full(field.shape, -1, dtype=np.int8)
        for r, c in cells:
            piece[r, c] = field[r, c]
        piece = _tight(piece)
        if np.any(piece == ANCHOR):
            anchor_index = len(pieces)
        pieces.append(piece)
        identities.append(label)
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = isqrt(area)
    _require(side*side == area and 3 <= side <= 6 and q*side <= 30,
             "Piece area must determine a supported square")
    return dict(q=q, side=side, k=k, templates=templates, targets=targets,
                order=order, pieces=pieces, identities=identities,
                anchor_index=anchor_index, field=field)


def _assemblies(facts, *, seams=True, rotate_texture=True, allow_rotation=True, limit=2):
    """Enumerate exact covers from input facts, pruning incompatible tile seams.

    Return at most `limit` different answers. With seams=False answers are
    carrier-color macroboards, used only to audit the shape-only shortcut.
    No sampling budget truncates this semantic search.
    """
    s, q = facts['side'], facts['q']
    pieces, labels = facts['pieces'], facts['identities']
    anchor = facts['anchor_index']
    all_bits = (1 << (s*s)) - 1
    options = []
    for i, (piece, label) in enumerate(zip(pieces, labels)):
        variants, seen = [], set()
        for turn in (range(4) if allow_rotation else (0,)):
            p = np.rot90(piece, turn)
            h, w = p.shape
            if h > s or w > s:
                continue
            motif = np.rot90(facts['templates'][label], turn if rotate_texture else 0)
            edge = _edges(motif)
            coords = [(int(r), int(c)) for r, c in np.argwhere(p >= 0)]
            occupied = set(coords)
            if seams and any((r+dr, c+dc) in occupied and edge[d] != edge[(d+2)%4]
                             for r, c in coords for d, (dr, dc) in enumerate(EDGE)):
                continue
            if i == anchor:
                if p[0, 0] != ANCHOR:
                    continue
                origins = [(0, 0)]
            else:
                origins = [(r, c) for r in range(s-h+1) for c in range(s-w+1)]
            for r0, c0 in origins:
                cells = tuple((r+r0)*s+c+c0 for r, c in coords)
                mask = sum(1 << pos for pos in cells)
                signature = (mask, motif.tobytes()) if seams else mask
                if signature in seen:
                    continue
                seen.add(signature)
                variants.append((mask, cells, edge, motif))
        options.append(variants)
    if any(not opts for opts in options):
        return []
    board = [-1]*(s*s)
    placed = [None]*len(pieces)
    answers, answer_keys = [], set()

    def compatible(candidate):
        if not seams:
            return True
        for pos in candidate[1]:
            r, c = divmod(pos, s)
            for d, (dr, dc) in enumerate(EDGE):
                nr, nc = r+dr, c+dc
                if 0 <= nr < s and 0 <= nc < s:
                    other = board[nr*s+nc]
                    if other >= 0 and candidate[2][d] != placed[other][2][(d+2)%4]:
                        return False
        return True

    def record():
        if seams:
            answer = np.zeros((s*q, s*q), dtype=np.int8)
            for i, candidate in enumerate(placed):
                tile = np.where(candidate[3], facts['targets'][labels[i]], 0)
                for pos in candidate[1]:
                    r, c = divmod(pos, s)
                    answer[r*q:(r+1)*q, c*q:(c+1)*q] = tile
        else:
            answer = np.array([labels[i] for i in board], dtype=np.int8).reshape(s, s)
        key = _key(answer)
        if key not in answer_keys:
            answer_keys.add(key)
            answers.append(answer)

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == all_bits:
                record()
            return
        first = (all_bits ^ occupied) & -(all_bits ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for candidate in options[i]:
                if not candidate[0] & first or candidate[0] & occupied or not compatible(candidate):
                    continue
                placed[i] = candidate
                for pos in candidate[1]:
                    board[pos] = i
                search(occupied | candidate[0], rest)
                for pos in candidate[1]:
                    board[pos] = -1
                placed[i] = None
                if len(answers) >= limit:
                    return

    for candidate in options[anchor]:
        placed[anchor] = candidate
        for pos in candidate[1]:
            board[pos] = anchor
        search(candidate[0], tuple(i for i in range(len(pieces)) if i != anchor))
        for pos in candidate[1]:
            board[pos] = -1
        placed[anchor] = None
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode, rigidly assemble with seam constraints, and render keyed tiles."""
    answers = _assemblies(_read_input(input_grid))
    _require(len(answers) == 1, "Input must have exactly one seam-compatible colored answer")
    return answers[0]


def _position_facts(f, cyclic=False):
    """Counterfactual: use library slots or infer one cyclic offset."""
    wrong = f.copy()
    if cyclic:
        shift = f['order'].index(f['identities'][0])
        slots = [f['order'][(shift+i)%f['k']] for i in range(f['k'])]
    else:
        slots = f['order']
    wrong['templates'] = {label:f['templates'][slot]
                          for label,slot in zip(f['identities'],slots)}
    wrong['targets'] = {label:f['targets'][slot]
                        for label,slot in zip(f['identities'],slots)}
    return wrong


def _partition(rng, s, k, style):
    if style == 'growth':
        owners = np.full((s, s), -1, dtype=np.int8)
        seeds = [0] + list(map(int, rng.choice(np.arange(1, s*s), k-1, replace=False)))
        for i, pos in enumerate(seeds):
            owners.flat[pos] = i
        weights = np.exp(rng.uniform(-1, 1, k))
        while np.any(owners < 0):
            frontiers = []
            for i in range(k):
                frontier = set()
                for r, c in np.argwhere(owners == i):
                    for dr, dc in EDGE:
                        nr, nc = int(r)+dr, int(c)+dc
                        if 0 <= nr < s and 0 <= nc < s and owners[nr, nc] < 0:
                            frontier.add(nr*s+nc)
                frontiers.append(sorted(frontier))
            candidates = [i for i in range(k) if frontiers[i]]
            i = min(candidates, key=lambda j: np.sum(owners == j)/weights[j] + rng.random()*.5)
            owners.flat[int(rng.choice(frontiers[i]))] = i
    else:
        rectangles = [(0, 0, s, s)]
        while len(rectangles) < k:
            choices = []
            for i, (r, c, h, w) in enumerate(rectangles):
                choices += [(i, 0, cut) for cut in range(1, h) if min(cut, h-cut)*w >= 2]
                choices += [(i, 1, cut) for cut in range(1, w) if min(cut, w-cut)*h >= 2]
            if not choices:
                return None
            i, axis, cut = choices[int(rng.integers(len(choices)))]
            r, c, h, w = rectangles.pop(i)
            rectangles.extend([(r,c,cut,w), (r+cut,c,h-cut,w)] if axis == 0
                              else [(r,c,h,cut), (r,c+cut,h,w-cut)])
        owners = np.empty((s,s), dtype=np.int8)
        for i, (r,c,h,w) in enumerate(rectangles):
            owners[r:r+h,c:c+w] = i
        if style == 'steps':
            for _ in range(s*s):
                r, c = map(int, rng.integers(s, size=2))
                dr, dc = EDGE[int(rng.integers(4))]
                nr, nc = r+dr, c+dc
                if not (0 <= nr < s and 0 <= nc < s) or (r,c) == (0,0):
                    continue
                old, new = int(owners[r,c]), int(owners[nr,nc])
                if old == new or np.sum(owners == old) <= 2:
                    continue
                owners[r,c] = new
                if len(_components(owners == old, EDGE)) != 1:
                    owners[r,c] = old
    if min(int(np.sum(owners == i)) for i in range(k)) < 2:
        return None
    return owners


def _motifs(rng, q, k):
    if q == 5:
        # Four distinct signatures (R, C, reverse R, reverse C) make different
        # frame orientations disagree in both horizontal and vertical joins.
        choices = [np.array([(code >> b)&1 for b in range(3)], dtype=bool)
                   for code in range(1,8)]
        while True:
            row, col = [choices[int(rng.integers(len(choices)))] for _ in range(2)]
            if len({v.tobytes() for v in (row,col,row[::-1],col[::-1])}) == 4:
                break
    else:
        row = np.array([1,0] if rng.integers(2) else [0,1], dtype=bool)
        col = np.full(2, bool(rng.integers(2)), dtype=bool)
    motifs = []
    for i in range(k):
        m = np.zeros((q,q), dtype=bool)
        m[0,1:-1] = m[-1,1:-1] = row
        m[1:-1,0] = m[1:-1,-1] = col
        style = int(rng.integers(5))
        n = q-2
        rr, cc = np.indices((n,n))
        inner = (rng.random((n,n)) < rng.uniform(.15,.85) if style == 0 else
                 np.ones((n,n), dtype=bool) if style == 1 else
                 np.zeros((n,n), dtype=bool) if style == 2 else
                 (rr+cc)%2 == int(rng.integers(2)) if style == 3 else
                 rr == cc)
        m[1:-1,1:-1] = inner
        if i and rng.random() < .15:
            m = motifs[int(rng.integers(i))].copy()
        motifs.append(m)
    return motifs


def _scatter(rng, pieces, min_width, q):
    # Occupied masks, rather than output-sized slots, determine separation.
    max_h = 30-(q+4)
    for attempt in range(60):
        h = int(rng.integers(max(8, max(p.shape[0] for p in pieces)+2), max_h+1))
        w = int(rng.integers(max(min_width, max(p.shape[1] for p in pieces)+2), 31))
        field = np.zeros((h,w), dtype=np.int8)
        blocked = np.zeros_like(field, dtype=bool)
        order = list(map(int, rng.permutation(len(pieces))))
        success = True
        for i in order:
            p = pieces[i]
            ph, pw = p.shape
            positions = [(r,c) for r in range(h-ph+1) for c in range(w-pw+1)
                         if not np.any(blocked[r:r+ph,c:c+pw][p >= 0])]
            if not positions:
                success = False
                break
            r0, c0 = positions[int(rng.integers(len(positions)))]
            for r, c in np.argwhere(p >= 0):
                r, c = int(r)+r0, int(c)+c0
                field[r,c] = p[r-r0,c-c0]
                blocked[max(0,r-1):min(h,r+2),max(0,c-1):min(w,c+2)] = True
        if success:
            rr, cc = np.nonzero(field)
            cropped = field[rr.min():rr.max()+1, cc.min():cc.max()+1]
            ch, cw = cropped.shape
            extra_h = int(rng.integers(0, min(3,max_h-ch)+1))
            final_w = max(min_width,cw)+int(rng.integers(0,min(3,30-max(min_width,cw))+1))
            top = int(rng.integers(extra_h+1))
            left = int(rng.integers(final_w-cw+1))
            final = np.zeros((ch+extra_h,final_w),dtype=np.int8)
            final[top:top+ch,left:left+cw] = cropped
            return final
    return None


def _encode(field, q, templates, targets, order):
    k = len(order)
    grid = np.zeros((q+4+field.shape[0],field.shape[1]),dtype=np.int8)
    grid[[0,q+1,q+3],:] = SEPARATOR
    grid[1:q+1,[j*(q+1) for j in range(k+1)]] = SEPARATOR
    for j,label in enumerate(order):
        c = 1+j*(q+1)
        grid[1:q+1,c:c+q] = np.where(templates[label],label,0)
        grid[q+2,c:c+q] = targets[label]
    grid[q+4:] = field
    return grid


def _construct(rng, s, k, q, anchor_turn, require_shape_ambiguity=False):
    for attempt in range(240):
        style = str(rng.choice(['growth','cuts','steps'], p=[.45,.15,.4]))
        owners = _partition(rng,s,k,style)
        if owners is None:
            continue
        labels = list(map(int, rng.choice(np.arange(1,8),k,replace=False)))
        anchor_owner = int(owners[0,0])
        if min(_tight(np.where(owners == anchor_owner, 1, -1)).shape) < 2:
            continue
        canonical = _motifs(rng,q,k)
        # A visible key may preserve a carrier, share a target, or reuse 8.
        targets = {label:int(rng.integers(1,9)) for label in labels}
        if require_shape_ambiguity:
            colors = list(map(int,rng.choice(np.arange(1,9),k,replace=False)))
            if any(color == label for color,label in zip(colors,labels)):
                continue
            targets = dict(zip(labels,colors))
        pieces, templates, identities = [], {}, []
        turns = list(map(int,rng.integers(4,size=k)))
        turns[anchor_owner] = anchor_turn
        # At least one unmarked asymmetric texture really changes orientation.
        if all(turns[i] == 0 for i in range(k) if i != anchor_owner):
            turns[(anchor_owner+1)%k] = int(rng.integers(1,4))
        for i,label in enumerate(labels):
            p = np.full((s,s),-1,dtype=np.int8)
            p[owners == i] = label
            if i == anchor_owner:
                p[0,0] = ANCHOR
            p = np.rot90(_tight(p),turns[i]).copy()
            pieces.append(p)
            identities.append(label)
            templates[label] = np.rot90(canonical[i],turns[i]).copy()
        f = dict(side=s,q=q,k=k,pieces=pieces,identities=identities,
                 templates=templates,targets=targets,anchor_index=anchor_owner)
        answers = _assemblies(f)
        if len(answers) != 1:
            continue
        # Compare with a construction using the unrotated design, not the solver.
        intended = np.zeros((s*q,s*q),dtype=np.int8)
        for r in range(s):
            for c in range(s):
                i = int(owners[r,c])
                intended[r*q:(r+1)*q,c*q:(c+1)*q] = np.where(canonical[i],targets[labels[i]],0)
        if not np.array_equal(answers[0],intended):
            continue
        if require_shape_ambiguity and len(_assemblies(f,seams=False)) < 2:
            continue
        if require_shape_ambiguity:
            unturned = _assemblies(f,rotate_texture=False)
            if len(unturned) == 1 and np.array_equal(unturned[0],intended):
                continue
            unmoved = _assemblies(f,allow_rotation=False)
            if len(unmoved) == 1 and np.array_equal(unmoved[0],intended):
                continue
        width = k*(q+1)+1
        field = _scatter(rng,pieces,width,q)
        if field is None:
            continue
        if require_shape_ambiguity:
            order = [next(int(field[r,c]) for r,c in cells if field[r,c] != ANCHOR)
                     for cells in _components(field != 0)]
            i,j = map(int,rng.choice(k,2,replace=False))
            order[i],order[j] = order[j],order[i]
        else:
            order = list(map(int,rng.permutation(labels)))
        grid = _encode(field,q,templates,targets,order)
        if require_shape_ambiguity:
            parsed = _read_input(grid)
            wrong_answers = [_assemblies(_position_facts(parsed,cyclic)) for cyclic in (False,True)]
            if any(len(wrong) == 1 and np.array_equal(wrong[0],intended) for wrong in wrong_answers):
                continue
        return {'input':grid, 'output':intended}
    raise RuntimeError('Bounded candidate construction exhausted')


def _counts(a,b):
    for name,value,lo,hi in [('num_train',a,2,10),('num_test',b,1,4)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a),int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a,b = _counts(num_train,num_test)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    q2 = int(rng.choice([4,5]))
    held = int(rng.integers(4))
    train_turns = [x for x in range(4) if x != held] if regime == 2 else list(range(4))
    training = [(4,3,5,train_turns[0]),
                (int(rng.choice([3,5])) if regime != 0 else 3,
                 2 if regime == 1 else 3,q2,train_turns[1])]
    for i in range(a-2):
        training.append((int(rng.choice([3,4])) if regime == 0 else int(rng.integers(3,6)),
                         int(rng.choice([2,3])) if regime == 1 else int(rng.integers(2,5)),
                         int(rng.choice([4,5])),int(rng.choice(train_turns))))
    q_seen = sorted({p[2] for p in training})
    testing = [(int(rng.choice([5,6])) if regime == 0 else int(rng.integers(4,7)),
                4 if regime == 1 else int(rng.integers(2,5)),int(rng.choice(q_seen)),
                held if regime == 2 else int(rng.integers(4))) for _ in range(b)]
    result,seen = {'train':[],'test':[]},set()
    for split,schedule in [('train',training),('test',testing)]:
        for i,(s,k,q,turn) in enumerate(schedule):
            # At side three, four minimum-area pieces would force repeated
            # dominoes once the marked piece has two-dimensional extent.
            # The sampler uses a larger square; solve imposes no such shortcut.
            s = max(s,4 if k == 4 else 3)
            for _ in range(8):
                pair = _construct(rng,s,k,q,turn,split == 'train' and i == 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError('Duplicate-input retries exhausted')
    validate_generated_puzzle(result)
    return result


def _block_invariants(f, out):
    q,s = f['q'],f['side']
    assert out.shape == (q*s,q*s)
    assert set(map(int,np.unique(out))) <= {0,*f['targets'].values()}
    ink = {color:0 for color in range(1,9)}
    for label,piece in zip(f['identities'],f['pieces']):
        ink[f['targets'][label]] += int(np.sum(piece >= 0))*int(f['templates'][label].sum())
    assert all(int(np.sum(out == color)) == amount for color,amount in ink.items())
    assert int(np.sum(out == 0)) == (q*s)**2-sum(ink.values())
    # Every neighboring block matches full edge bitmaps, even within pieces.
    for r in range(s):
        for c in range(s):
            tile = out[r*q:(r+1)*q,c*q:(c+1)*q] != 0
            assert tile.any()
            if r+1 < s:
                assert np.array_equal(tile[-1],out[(r+1)*q,c*q:(c+1)*q] != 0)
            if c+1 < s:
                assert np.array_equal(tile[:,-1],out[r*q:(r+1)*q,(c+1)*q] != 0)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    _counts(len(puzzle['train']),len(puzzle['test']))
    seen, facts = set(), {'train':[],'test':[]}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            f = _read_input(inp)
            answers = _assemblies(f)
            assert len(answers) == 1 and np.array_equal(answers[0],out)
            _block_invariants(f,out)
            # Carrier macro-area is neither motif ink area nor library area.
            assert sum(int(np.sum(p >= 0)) for p in f['pieces']) == f['side']**2
            assert set(f['identities']) == set(f['templates'])
            facts[split].append(f)
    tr,te = facts['train'],facts['test']
    assert {f['q'] for f in te} <= {f['q'] for f in tr}
    assert len({f['side'] for f in tr}) >= 2
    def corner(f):
        p = f['pieces'][f['anchor_index']]
        r,c = map(int,np.argwhere(p == ANCHOR)[0])
        assert min(p.shape) >= 2
        return (r == p.shape[0]-1,c == p.shape[1]-1)
    assert (min(f['side'] for f in te) > max(f['side'] for f in tr)
            or min(f['k'] for f in te) > max(f['k'] for f in tr)
            or not ({corner(f) for f in tr} & {corner(f) for f in te})), 'Need an observable split shift'
    assert any(len(_assemblies(f,seams=False)) >= 2 for f in tr), 'Need a shape-only ambiguity witness'
    assert any(not (len(a := _assemblies(f,rotate_texture=False)) == 1
                   and np.array_equal(a[0],pair['output']))
               for f,pair in zip(tr,puzzle['train'])), 'Need coupled-rotation evidence'
    assert any(not (len(a := _assemblies(f,allow_rotation=False)) == 1
                   and np.array_equal(a[0],pair['output']))
               for f,pair in zip(tr,puzzle['train'])), 'Need rigid-rotation evidence'
    assert any(not (len(a := _assemblies(_position_facts(f))) == 1
                   and np.array_equal(a[0],pair['output']))
               for f,pair in zip(tr,puzzle['train'])), 'Need identity-correspondence evidence'
    assert any(not (len(a := _assemblies(_position_facts(f,True))) == 1
                   and np.array_equal(a[0],pair['output']))
               for f,pair in zip(tr,puzzle['train'])), 'Need noncyclic-correspondence evidence'
    assert any(sum(f['targets'][x] != x for x in f['identities']) >= 2
               and len(set(f['targets'].values())) >= 2 for f in tr)


def self_test():
    """Bounded finite audits, including counterfactuals and 42-seed variation."""
    signatures,shapes,layouts,motifs = set(),set(),set(),set()
    counts,sides,scales,anchor_corners,palettes = set(),set(),set(),set(),set()
    shared = preserved = duplicated = bars = unequal = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair['input']) for split in p for pair in p[split]))
        for split in p:
            for pair in p[split]:
                f = _read_input(pair['input'])
                assert np.array_equal(solve(pair['input']),pair['output'])
                shapes.add(pair['input'].shape)
                counts.add(f['k']); sides.add(f['side']); scales.add(f['q'])
                palettes.add(tuple(sorted(f['targets'].items())))
                normalized = np.zeros_like(f['field'])
                for i,label in enumerate(f['identities'],1):
                    normalized[f['field'] == label] = i
                normalized[f['field'] == ANCHOR] = 9
                layouts.add(_key(normalized))
                motifs.update(_key(m) for m in f['templates'].values())
                ap = f['pieces'][f['anchor_index']]
                anchor_corners.add(tuple(map(int,np.argwhere(ap == ANCHOR)[0])))
                areas = [int(np.sum(piece >= 0)) for piece in f['pieces']]
                bars += any(min(piece.shape) == 1 for piece in f['pieces'])
                unequal += max(areas) >= 3*min(areas)
                shared += len(set(f['targets'].values())) < f['k']
                preserved += any(f['targets'][x] == x for x in f['identities'])
                duplicated += len({m.tobytes() for m in f['templates'].values()}) < f['k']
    assert len(signatures) == 42 and len(layouts) >= 180
    assert len(shapes) >= 35 and len(motifs) >= 70 and len(palettes) >= 100
    assert counts == {2,3,4} and sides == {3,4,5,6} and scales == {4,5}
    assert len(anchor_corners) >= 10
    assert shared >= 10 and preserved >= 10 and duplicated >= 5 and bars >= 10 and unequal >= 20
    # A family member outside the edge-connected sampler: one carrier has a
    # corner-only attachment. Its pixels must travel with the entire piece.
    rng = np.random.default_rng(101)
    owners = np.array([[0,0,1,1],[0,2,0,1],[2,2,3,1],[3,3,3,1]])
    canonical = _motifs(rng,5,4)
    labels,turns = [1,2,3,4],[1,2,3,0]
    pieces,templates = [],{}
    targets = {1:5,2:6,3:7,4:8}
    expected = np.zeros((20,20),dtype=np.int8)
    for i,label in enumerate(labels):
        piece = np.where(owners == i,label,-1).astype(np.int8)
        if i == 0:
            piece[0,0] = ANCHOR
        pieces.append(np.rot90(_tight(piece),turns[i]).copy())
        templates[label] = np.rot90(canonical[i],turns[i]).copy()
        for r,c in np.argwhere(owners == i):
            expected[r*5:(r+1)*5,c*5:(c+1)*5] = np.where(canonical[i],targets[label],0)
    field = _scatter(rng,pieces,25,5)
    assert field is not None
    hand_input = _encode(field,5,templates,targets,[4,2,1,3])
    assert any(len(_components(piece >= 0,EDGE)) > 1 for piece in pieces)
    assert np.array_equal(solve(hand_input),expected)
    p0 = generate_puzzle(0)
    p1 = generate_puzzle(0)
    assert all(np.array_equal(x[k],y[k]) for split in p0
               for x,y in zip(p0[split],p1[split]) for k in ('input','output'))
    for a in range(2,11):
        for b in range(1,min(4,a-1)+1):
            p = generate_puzzle(100+a*5+b,a,b)
            assert len(p['train']) == a and len(p['test']) == b
    for a,b in [(1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)]:
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s:[{k:v.copy() for k,v in pair.items()} for pair in p0[s]] for s in p0}
    bad['test'][0]['output'][0,0] = 9
    try:
        validate_generated_puzzle(bad)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError('Corrupt output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                normalized_fields=len(layouts), motifs=len(motifs), counts=sorted(counts),
                macro_sides=sorted(sides), tile_sides=sorted(scales),
                shared_keys=shared, identity_keys=preserved, duplicate_motifs=duplicated,
                bar_pairs=bars, unequal_area_pairs=unequal)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(',',':')))
