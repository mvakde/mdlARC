"""Torn wallpaper: one rigid, phase-consistent jigsaw, with inked joins.

Only standard-library modules and NumPy are used.  No parent generator, stored
example, seed-dependent solver convention, or training label is consulted.
"""
from __future__ import annotations

from math import isqrt
import numpy as np


_EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, neighbors=_EIGHT):
    pending = {tuple(map(int, p)) for p in np.argwhere(mask)}
    groups = []
    while pending:
        start = min(pending)
        pending.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in pending:
                    pending.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _connected(mask):
    return len(_components(mask, _EDGE)) == 1


def _tight(mask, colors):
    points = np.argwhere(mask)
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0) + 1)
    result = np.full((r1 - r0, c1 - c0), -1, dtype=np.int8)
    view = mask[r0:r1, c0:c1]
    result[view] = colors[r0:r1, c0:c1][view]
    return result


def _rotations(piece):
    seen = set()
    for k in range(4):
        p = np.rot90(piece, k).copy()
        key = _key(p)
        if key not in seen:
            seen.add(key)
            yield p


def _read_input(value):
    grid = _grid(value)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(len(np.unique(border)) == 1, "uniform empty border required")
    background = int(border[0])
    foreground = grid != background
    area = int(foreground.sum())
    side = isqrt(area)
    _require(7 <= side <= 12 and side * side == area, "foreground must be a side-7..12 square's area")
    colors, counts = np.unique(grid[foreground], return_counts=True)
    _require(len(colors) == 3, "two print colors and one marker required")
    singletons = colors[counts == 1]
    _require(len(singletons) == 1, "unique singleton marker required")
    marker = int(singletons[0])
    print_colors = set(map(int, colors)) - {marker}
    groups = _components(foreground)
    _require(3 <= len(groups) <= 6, "three to six separated fragments required")
    pieces = []
    for group in groups:
        _require(len(group) >= 3, "each fragment needs at least three cells")
        mask = np.zeros(grid.shape, dtype=bool)
        for r, c in group:
            mask[r, c] = True
        piece = _tight(mask, grid)
        # Eight-neighbor grouping preserves rigid diagonal attachments as in A.
        _require(set(map(int, piece[piece >= 0])) <= print_colors | {marker}, "invalid fragment color")
        pieces.append(piece)
    anchor = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    _require(len(anchor) == 1, "marker must belong to one fragment")
    return dict(grid=grid, background=background, side=side, marker=marker,
                print_colors=print_colors, pieces=pieces, anchor=anchor[0])


def _anchor_tiles(facts):
    """The marked fragment supplies every phase, not a seed or hidden motif."""
    side, marker = facts['side'], facts['marker']
    for p in _rotations(facts['pieces'][facts['anchor']]):
        if p[0, 0] != marker or max(p.shape) > side or min(p.shape) < 2:
            continue
        tile = np.full((3, 3), -1, dtype=np.int8)
        consistent = True
        for r, c in np.argwhere((p >= 0) & (p != marker)):
            v, y, x = int(p[r, c]), int(r) % 3, int(c) % 3
            if tile[y, x] >= 0 and tile[y, x] != v:
                consistent = False
                break
            tile[y, x] = v
        if not consistent or np.any(tile < 0):
            continue
        vals, counts = np.unique(tile, return_counts=True)
        if set(map(int, vals)) != facts['print_colors']:
            continue
        ink = int(vals[np.argmin(counts)])
        if not 2 <= int(counts.min()) <= 4:
            continue
        # The print has genuine period three on each axis, not stripes.
        if np.all(tile == tile[0:1, :]) or np.all(tile == tile[:, 0:1]):
            continue
        board = tile[np.arange(side)[:, None] % 3, np.arange(side)[None, :] % 3].copy()
        board[0, 0] = marker
        yield p, tile, ink, board


def _placements(piece, side, expected=None):
    result = []
    for p in _rotations(piece):
        h, w = p.shape
        if max(h, w) > side:
            continue
        mask = p >= 0
        coords = np.argwhere(mask)
        base = sum(1 << (int(r) * side + int(c)) for r, c in coords)
        for r in range(side - h + 1):
            for c in range(side - w + 1):
                if expected is not None and not np.array_equal(p[mask], expected[r:r+h, c:c+w][mask]):
                    continue
                result.append((base << (r * side + c), r, c, p))
    return result


def _join_mask(owners):
    seam = np.zeros(owners.shape, dtype=bool)
    v = owners[1:, :] != owners[:-1, :]
    seam[1:, :] |= v
    seam[:-1, :] |= v
    h = owners[:, 1:] != owners[:, :-1]
    seam[:, 1:] |= h
    seam[:, :-1] |= h
    return seam


def _finish(board, owners, ink, marker):
    result = board.copy()
    result[(board == ink) & _join_mask(owners)] = marker
    # The original marker remains even if its underlying phase is ink.
    result[0, 0] = marker
    return result


def _assemblies(facts, limit=2):
    """Exact cover with print registration, unique final colored answers.

    The first-empty-cell recursion covers every packing without permutations.
    Matching all colored cells to the anchored period filters placements before
    search; ownership is retained because it controls the final highlight.
    """
    side, marker = facts['side'], facts['marker']
    pieces, anchor = facts['pieces'], facts['anchor']
    full = (1 << (side * side)) - 1
    answers, seen = [], set()
    for ap, tile, ink, expected in _anchor_tiles(facts):
        candidates = [_placements(p, side, expected) if i != anchor else []
                      for i, p in enumerate(pieces)]
        amask = sum(1 << (int(r) * side + int(c)) for r, c in np.argwhere(ap >= 0))
        selected = {anchor: (amask, 0, 0, ap)}
        remaining = tuple(i for i in range(len(pieces)) if i != anchor)
        if any(not candidates[i] for i in remaining):
            continue

        def search(occupied, rest):
            if not rest:
                if occupied != full:
                    return
                owners = np.full((side, side), -1, dtype=np.int8)
                board = np.full((side, side), -1, dtype=np.int8)
                for i, (_, r, c, p) in selected.items():
                    mask = p >= 0
                    owners[r:r+p.shape[0], c:c+p.shape[1]][mask] = i
                    board[r:r+p.shape[0], c:c+p.shape[1]][mask] = p[mask]
                answer = _finish(board, owners, ink, marker)
                key = _key(answer)
                if key not in seen:
                    seen.add(key)
                    answers.append(dict(output=answer, board=board, owners=owners,
                                        placements=selected.copy(), tile=tile.copy(), ink=ink,
                                        carrier=next(iter(facts['print_colors'] - {ink})),
                                        seam=_join_mask(owners), **facts))
                return
            empty_bit = (full ^ occupied) & -(full ^ occupied)
            for i in rest:
                for candidate in candidates[i]:
                    if candidate[0] & empty_bit and not candidate[0] & occupied:
                        selected[i] = candidate
                        search(occupied | candidate[0], tuple(j for j in rest if j != i))
                        del selected[i]
                        if len(answers) >= limit:
                            return

        search(amask, remaining)
        if len(answers) >= limit:
            break
    return answers


def _analyze(value):
    facts = _read_input(value)
    answers = _assemblies(facts)
    _require(len(answers) == 1, "exactly one phase-consistent colored answer required")
    return answers[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using this input alone; the caller's array is never modified."""
    return _analyze(input_grid)['output'].copy()


def _geometry_witness(f):
    """Find a valid rigid packing that gives a wrong answer without phase matching.

    Interchange congruent fragment silhouettes, retaining their actual colored
    patterns. This is a concrete ablation, not just an ambiguity count.
    """
    ids = [i for i in f['placements'] if i != f['anchor']]
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            _, ar, ac, ap = f['placements'][a]
            _, br, bc, bp = f['placements'][b]
            for aq in _rotations(f['pieces'][a]):
                if aq.shape != bp.shape or not np.array_equal(aq >= 0, bp >= 0):
                    continue
                for bq in _rotations(f['pieces'][b]):
                    if bq.shape != ap.shape or not np.array_equal(bq >= 0, ap >= 0):
                        continue
                    board = f['board'].copy()
                    av = board[ar:ar+ap.shape[0], ac:ac+ap.shape[1]]
                    bv = board[br:br+bp.shape[0], bc:bc+bp.shape[1]]
                    av[ap >= 0] = bq[bq >= 0]
                    bv[bp >= 0] = aq[aq >= 0]
                    wrong = _finish(board, f['owners'], f['ink'], f['marker'])
                    if not np.array_equal(wrong, f['output']):
                        return wrong
    return None


def _nonanchor_ink_join(f):
    owners, board, anchor, ink = f['owners'], f['board'], f['anchor'], f['ink']
    for dr, dc in ((1, 0), (0, 1)):
        h, w = board.shape
        for r in range(h - dr):
            for c in range(w - dc):
                a, b = int(owners[r, c]), int(owners[r+dr, c+dc])
                if a != b and anchor not in (a, b) and (board[r, c] == ink or board[r+dr, c+dc] == ink):
                    return True
    return False


def _hidden_ink_join(f):
    """Ink at a join that cannot be seen from the anchor's cavity outline."""
    anchor_outline = _join_mask((f['owners'] == f['anchor']).astype(np.int8))
    return bool(np.any(f['seam'] & ~anchor_outline & (f['board'] == f['ink'])))


def _motifs(density):
    result = []
    for code in range(512):
        m = np.array([(code >> k) & 1 for k in range(9)], dtype=bool).reshape(3, 3)
        if int(m.sum()) == density and not np.all(m == m[0:1]) and not np.all(m == m[:, 0:1]):
            result.append(m)
    return result


_MOTIFS = {d: _motifs(d) for d in (2, 3, 4)}


def _asymmetric(mask):
    return len({np.rot90(base, k).tobytes() for base in (mask, mask[:, ::-1])
                for k in range(4)}) == 8


def _small_shape(rng, max_area=7):
    # Grow a fresh edge-connected polyomino; tight boxes can be bars, elbows,
    # solids, or irregular inserts. No saved shape vocabulary is consulted.
    h, w = map(int, rng.integers(2, 4, size=2))
    target = int(rng.integers(3, min(h*w, max_area) + 1))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < target:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _EDGE
                           if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    points = np.argwhere(mask)
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0) + 1
    return mask[r0:r1, c0:c1]


def _insert_partition(rng, side, count):
    """Wrapping fragment plus a touching, congruent pair and further inserts."""
    labels = np.zeros((side, side), dtype=np.int8)
    reference = _small_shape(rng, 5)
    for label in range(1, count):
        shape = reference if label <= 2 else _small_shape(rng)
        shape = np.rot90(shape, int(rng.integers(4))).copy()
        h, w = shape.shape
        choices = [(r, c) for r in range(side-h+1) for c in range(side-w+1)]
        placed = False
        for ix in rng.permutation(len(choices)):
            r, c = choices[int(ix)]
            cells = [(r+int(y), c+int(x)) for y, x in np.argwhere(shape)]
            if any(labels[y, x] or (y < 4 and x < 4) for y, x in cells):
                continue
            if label == 2 and not any(0 <= y+dy < side and 0 <= x+dx < side
                                      and labels[y+dy, x+dx] == 1
                                      for y, x in cells for dy, dx in _EDGE):
                continue
            proposal = labels.copy()
            for y, x in cells:
                proposal[y, x] = label
            if not _connected(proposal == 0):
                continue
            labels, placed = proposal, True
            break
        if not placed:
            return None
    return labels


def _growth_partition(rng, side, count):
    labels = np.full((side, side), -1, dtype=np.int8)
    labels[:4, :4] = 0
    free = np.flatnonzero(labels.ravel() < 0)
    for label, pos in enumerate(rng.choice(free, count-1, replace=False), 1):
        labels.flat[int(pos)] = label
    weights = np.exp(rng.uniform(-0.9, 0.9, count))
    for _ in range(int((labels < 0).sum())):
        fronts = []
        for label in range(count):
            cells = np.argwhere(labels == label)
            front = sorted({(int(r)+dr, int(c)+dc) for r, c in cells for dr, dc in _EDGE
                            if 0 <= r+dr < side and 0 <= c+dc < side and labels[r+dr, c+dc] < 0})
            fronts.append(front)
        eligible = [i for i, front in enumerate(fronts) if front]
        # The pre-existing reference block is discounted in the area budget.
        sizes = [max(1, int((labels == i).sum()) - (12 if i == 0 else 0)) for i in range(count)]
        label = min(eligible, key=lambda i: sizes[i]/weights[i] + float(rng.random())*.3)
        r, c = fronts[label][int(rng.integers(len(fronts[label])))]
        labels[r, c] = label
    return labels if min(int((labels == i).sum()) for i in range(count)) >= 3 else None


def _band_partition(rng, side, count, repeat=False):
    """Parallel stepped cuts, with straight strips retained as simple members."""
    # The anchor occupies at least the first four columns. All other strips
    # have positive width; steps transfer boundary cells without disconnecting.
    if side < count + 3:
        return None
    if repeat:
        common = int(rng.integers(1, (side-4-(count-3))//2 + 1))
        widths = [4, common, common] + [1]*(count-3)
        for _ in range(side-sum(widths)):
            eligible = [0] + list(range(3, count))
            widths[int(rng.choice(eligible))] += 1
        widths = widths[:1] + [widths[int(i)] for i in rng.permutation(np.arange(1, count))]
        cuts = list(map(int, np.cumsum(widths)[:-1]))
    else:
        cuts = sorted(map(int, rng.choice(np.arange(4, side), count-1, replace=False)))
    labels = np.empty((side, side), dtype=np.int8)
    for i, (a, b) in enumerate(zip([0]+cuts, cuts+[side])):
        labels[:, a:b] = i
    for _ in range(0 if repeat else int(rng.integers(0, side*2+1))):
        r, c = map(int, rng.integers(side, size=2))
        if r < 4 and c < 4:
            continue
        dr, dc = _EDGE[int(rng.integers(4))]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < side and 0 <= nc < side):
            continue
        a, b = int(labels[r, c]), int(labels[nr, nc])
        if a == b or int((labels == a).sum()) <= 3:
            continue
        labels[r, c] = b
        if not _connected(labels == a) or not _connected(labels == b):
            labels[r, c] = a
    return labels.T.copy() if rng.random() < .5 else labels


def _scatter(rng, pieces, background, anchor_rotation, anchor):
    oriented = [np.rot90(p, anchor_rotation if i == anchor else int(rng.integers(4))).copy()
                for i, p in enumerate(pieces)]
    # Actual occupied masks, with eight-neighbor exclusion, on a 28x28 interior.
    # Crop afterwards, then add an independently sized uniform empty border.
    compact = bool(rng.random() < .8)
    for attempt in range(16):
        canvas = np.full((28, 28), background, dtype=np.int8)
        blocked = np.zeros((28, 28), dtype=bool)
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        if attempt:
            order = sorted(order, key=lambda i: -oriented[i].size + float(rng.random())*20)
        maxr = maxc = 0
        good = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            mask = p >= 0
            choices = []
            for r in range(29-h):
                for c in range(29-w):
                    if np.any(blocked[r:r+h, c:c+w][mask]):
                        continue
                    score = (max(maxr, r+h)*max(maxc, c+w) + float(rng.random())*25
                             if compact else float(rng.random()))
                    choices.append((score, r, c))
            if not choices:
                good = False
                break
            _, r, c = min(choices)
            canvas[r:r+h, c:c+w][mask] = p[mask]
            for y, x in np.argwhere(mask):
                yy, xx = r+int(y), c+int(x)
                blocked[max(0, yy-1):min(28, yy+2), max(0, xx-1):min(28, xx+2)] = True
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not good:
            continue
        occupied = np.argwhere(canvas != background)
        r0, c0 = occupied.min(axis=0)
        r1, c1 = occupied.max(axis=0)+1
        canvas = canvas[r0:r1, c0:c1]
        h, w = canvas.shape
        extra_h = int(rng.integers(2, min(6, 30-h)+1))
        extra_w = int(rng.integers(2, min(6, 30-w)+1))
        top = int(rng.integers(1, extra_h))
        left = int(rng.integers(1, extra_w))
        result = np.full((h+extra_h, w+extra_w), background, dtype=np.int8)
        result[top:top+h, left:left+w] = canvas
        return result
    return None


def _make_pair(rng, side, count, density, palette, rotation, mode, witness, hidden, asymmetric, used):
    background, carrier, ink, marker = palette
    for _ in range(350):
        motif = _MOTIFS[density][int(rng.integers(len(_MOTIFS[density])))]
        # A cyclic phase draw changes the wallpaper's registration, not its rule.
        motif = np.roll(motif, tuple(map(int, rng.integers(3, size=2))), axis=(0, 1))
        if asymmetric and not _asymmetric(motif):
            continue
        if mode == 'inserts':
            labels = _insert_partition(rng, side, count)
        elif mode == 'bands':
            labels = _band_partition(rng, side, count, witness)
        else:
            labels = _growth_partition(rng, side, count)
        if labels is None:
            continue
        tile = np.where(motif, ink, carrier).astype(np.int8)
        board = tile[np.arange(side)[:, None] % 3, np.arange(side)[None, :] % 3].copy()
        board[0, 0] = marker
        pieces = [_tight(labels == i, board) for i in range(count)]
        facts = dict(side=side, marker=marker, pieces=pieces, anchor=0,
                     print_colors={carrier, ink}, background=background)
        answers = _assemblies(facts)
        if len(answers) != 1:
            continue
        f = answers[0]
        expected = _finish(board, labels, ink, marker)
        if not np.array_equal(f['output'], expected):
            continue
        ink_join = int(np.sum(f['seam'] & (f['board'] == ink)))
        carrier_join = int(np.sum(f['seam'] & (f['board'] == carrier)))
        if ink_join < 2 or carrier_join < 2:
            continue
        if witness and (_geometry_witness(f) is None or not _nonanchor_ink_join(f)):
            continue
        if hidden and not _hidden_ink_join(f):
            continue
        inp = _scatter(rng, pieces, background, rotation, 0)
        if inp is None or _key(inp) in used:
            continue
        return dict(input=inp, output=expected)
    raise RuntimeError(f'could not construct {mode} member: side={side}, count={count}, density={density}')


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base = list(map(int, rng.permutation(10)[:4]))
    palette_mode = str(rng.choice(('stable', 'vary')))
    regime = str(rng.choice(('scale', 'fragmentation', 'rotation')))
    densities = list(map(int, rng.permutation((2, 4))))
    first_side = int(rng.integers(7, 10))
    train_sides = [first_side, int(rng.choice([s for s in range(7, 11) if s != first_side]))]
    train_sides += list(map(int, rng.integers(7, 11, size=ntrain-2)))
    if regime == 'scale':
        train_sides = [min(s, 9) for s in train_sides]
        if train_sides[0] == train_sides[1]:
            train_sides[1] = 7 if train_sides[0] != 7 else 8
    puzzle, used = {'train': [], 'test': []}, set()
    previous_palette = base
    for split, number in (('train', ntrain), ('test', ntest)):
        for i in range(number):
            if palette_mode == 'stable':
                palette = base
            else:
                choices = [c for c in range(10) if c != base[3]]
                background = int(rng.choice([c for c in choices if c != previous_palette[0]]))
                other = list(map(int, rng.choice([c for c in choices if c != background], 2, replace=False)))
                palette = [background] + other + [base[3]]
                previous_palette = palette
            if split == 'train':
                side = train_sides[i]
                count = 3 if i == 0 else 4 if i == 1 else int(rng.integers(3, 5 if regime == 'fragmentation' else 7))
                density = densities[i] if i < 2 else int(rng.choice((2, 4)))
                rotation = i % 2 if regime == 'rotation' else i % 4
                mode = 'inserts' if i < 2 else ('growth' if i == 2 else str(rng.choice(('growth', 'bands', 'inserts'))))
                witness = i < 2
            else:
                side = int(rng.integers(10, 13)) if regime == 'scale' else int(rng.integers(8, 13))
                count = int(rng.integers(5, 7)) if regime == 'fragmentation' else int(rng.integers(3, 7))
                density = 3 if i == 0 else int(rng.integers(2, 5))
                rotation = 2 + i % 2 if regime == 'rotation' else int(rng.integers(4))
                mode = str(rng.choice(('inserts', 'bands'))) if i == 0 else str(rng.choice(('growth', 'bands', 'inserts')))
                witness = i == 0
            if mode == 'bands' and side < count + 3:
                mode = 'inserts' if witness else 'growth'
            hidden = witness and (split == 'test' or i == 1)
            asymmetric = split == 'train' and i == 1
            pair = _make_pair(rng, side, count, density, palette, rotation, mode, witness, hidden, asymmetric, used)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Validate returned arrays, print conservation, joins, and witness coverage."""
    try:
        _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
        _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
        _counts(len(puzzle['train']), len(puzzle['test']))
        seen, facts = set(), {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
                inp, out = _grid(pair['input']), _grid(pair['output'])
                _require(_key(inp) not in seen, 'duplicate input')
                seen.add(_key(inp))
                f = _analyze(inp)
                _require(out.shape == (f['side'], f['side']), 'area-derived output size')
                _require(np.array_equal(out, f['output']), 'incorrect phase matching or join recoloring')
                _require(np.all(f['owners'] >= 0), 'all cells must have an owner')
                _require(out[0, 0] == f['marker'], 'upper-left anchor')
                ink_join = f['seam'] & (f['board'] == f['ink'])
                carrier_join = f['seam'] & (f['board'] == f['carrier'])
                _require(int(ink_join.sum()) >= 2 and int(carrier_join.sum()) >= 2, 'show conditional seam recoloring')
                _require(int((out == f['carrier']).sum()) == int((inp == f['carrier']).sum()), 'carrier cells survive')
                _require(int((out == f['ink']).sum()) == int((inp == f['ink']).sum()) - int(ink_join.sum()), 'ink conservation')
                _require(int((out == f['marker']).sum()) == 1 + int(ink_join.sum()), 'marker/highlight footprint')
                # Omitting the object boundaries leaves the unrecolored print.
                _require(not np.array_equal(f['board'], out), 'A ablation must change the answer')
                facts[split].append(f)
        all_f = facts['train'] + facts['test']
        _require(len({f['marker'] for f in all_f}) == 1, 'shared marker convention')
        first = facts['train'][:2]
        _require({int((f['tile'] == f['ink']).sum()) for f in first} == {2, 4}, 'two distinct motif footprints')
        _require(all(int((f['tile'] == f['ink']).sum()) in (2, 4) for f in facts['train']), 'training density convention')
        _require(any(_asymmetric(f['tile'] == f['ink']) for f in first), 'asymmetric printed orientation witness')
        for f in first + facts['test'][:1]:
            _require(_geometry_witness(f) is not None, 'B ablation requires an incorrect geometric packing')
            _require(_nonanchor_ink_join(f), 'non-anchor fragments must determine visible joins')
        for f in first[1:] + facts['test'][:1]:
            _require(_hidden_ink_join(f), 'anchor-outline shortcut must give a different answer')
        _require(len({len(f['pieces']) for f in first}) == 2, 'training establishes changing fragment counts')
        # The first test has the intermediate, unseen motif density; this is
        # compositional generalization, never a different transformation.
        _require(int((facts['test'][0]['tile'] == facts['test'][0]['ink']).sum()) == 3, 'motif test shift')
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise AssertionError(f'invalid generated puzzle: {error}') from error


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test() -> dict:
    """Bounded 42-seed audit; returned metrics are descriptive, not certification."""
    sides, counts, shapes, tiles, partitions, palettes, outputs = set(), set(), set(), set(), set(), set(), set()
    puzzles, corners, topology = set(), set(), set()
    geometry_witnesses = nonanchor_joins = hidden_joins = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _analyze(pair['input'])
                sides.add(f['side'])
                counts.add(len(f['pieces']))
                shapes.add(pair['input'].shape)
                tiles.add((f['tile'] == f['ink']).tobytes())
                partitions.add(_key(f['seam'].astype(np.int8)))
                palettes.add((f['background'], f['carrier'], f['ink'], f['marker']))
                outputs.add(_key(pair['output']))
                ap = f['pieces'][f['anchor']]
                r, c = map(int, np.argwhere(ap == f['marker'])[0])
                corners.add((r == 0, c == 0))
                topology.add(tuple(sorted(len(g) for g in _components(f['owners'] != f['anchor'], _EDGE))))
                geometry_witnesses += _geometry_witness(f) is not None
                nonanchor_joins += _nonanchor_ink_join(f)
                hidden_joins += _hidden_ink_join(f)
    assert len(puzzles) == 42
    assert sides == set(range(7, 13)) and counts == {3, 4, 5, 6}
    assert len(shapes) >= 30 and len(tiles) >= 60 and len(partitions) >= 120
    assert len(palettes) >= 40 and len(outputs) >= 180 and len(topology) >= 25
    assert len(corners) == 4 and geometry_witnesses >= 126 and nonanchor_joins >= 126
    assert hidden_joins >= 84
    # Determinism, all legal count combinations, illegal types and boundaries.
    assert _fingerprint(generate_puzzle(707)) == _fingerprint(generate_puzzle(707))
    probe = generate_puzzle(708)['test'][0]
    original = probe['input'].copy()
    for k in range(4):
        assert np.array_equal(solve(np.rot90(original, k)), probe['output'])
    perm = np.array([7, 2, 9, 1, 8, 0, 6, 4, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(perm[original]), perm[probe['output']])
    assert np.array_equal(probe['input'], original)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            validate_generated_puzzle(p)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A carrier overwrite and removal of one required highlight must be rejected.
    p = generate_puzzle(919)
    for color in ('carrier', 'marker'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        f = _analyze(pair['input'])
        r, c = map(int, np.argwhere((pair['output'] == f[color]) & f['seam'])[0])
        pair['output'][r, c] = f['ink']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('damaged output accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(puzzles), square_sides=sorted(sides),
                fragment_counts=sorted(counts), input_shapes=len(shapes), motifs=len(tiles),
                seam_structures=len(partitions), palettes=len(palettes), outputs=len(outputs),
                geometry_ablation_witnesses=geometry_witnesses, nonanchor_ink_joins=nonanchor_joins,
                anchor_outline_ablation_witnesses=hidden_joins)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
