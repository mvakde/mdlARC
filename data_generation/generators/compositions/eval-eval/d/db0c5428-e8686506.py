"""Radial lock jigsaw: an inward color key constrains a finite half-turned inventory.

Only the standard library and NumPy are used. No parent implementation is imported.
The solver parses and searches its input, including proof of colored-answer uniqueness.
"""
from functools import lru_cache
import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
SEARCH_LIMIT = 20000


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, comp = [start], []
        while todo:
            r, c = todo.pop()
            comp.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(tuple(sorted(comp)))
    return result


def _normalize(piece):
    """A piece is a tuple of (row, column, color), including no background."""
    r0, c0 = min(p[0] for p in piece), min(p[1] for p in piece)
    return tuple(sorted((r-r0, c-c0, int(v)) for r, c, v in piece))


def _transform(piece, operation=2):
    result = []
    for r, c, v in piece:
        if operation >= 4:
            c = -c
        for _ in range(operation % 4):
            r, c = -c, r
        result.append((r, c, v))
    return _normalize(result)


def _parse(inp):
    a = _grid(inp)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > a.size / 2, "background must be a strict canvas majority")
    fg = a[a != bg]
    _require(len(fg) > 0, "empty foreground")
    colors, n = np.unique(fg, return_counts=True)
    carrier = int(colors[np.argmax(n)])
    _require(int(n.max()) > len(fg) / 2, "carrier must be a strict foreground majority")
    rr, cc = np.where(a == carrier)
    top, left = int(rr.min()), int(cc.min())
    bottom, right = int(rr.max())+1, int(cc.max())+1
    board = a[top:bottom, left:right].copy()
    h, w = board.shape
    _require(h >= 9 and w >= 9, "board too small for the radial key")
    _require(h*w == len(fg), "inventory must conserve the complete rectangle area")
    key_cells = np.argwhere((board != bg) & (board != carrier))
    _require(len(key_cells) == 8, "the only fixed payload cells must be the eight hub cells")
    kr, kc = map(int, key_cells.min(axis=0))
    _require(tuple(key_cells.max(axis=0)-key_cells.min(axis=0)) == (2, 2),
             "hub must occupy a 3x3 square")
    center = (kr+1, kc+1)
    _require(board[center] == carrier, "hub center must be carrier")
    _require(set(map(tuple, key_cells)) == {(center[0]+dr, center[1]+dc)
                                          for dr, dc in DIRECTIONS}, "malformed hub")
    _require(min(kr, kc, h-kr-3, w-kc-3) >= 3, "insufficient radial board margins")
    locks = {(center[0]+2*dr, center[1]+2*dc):
             int(board[center[0]+dr, center[1]+dc]) for dr, dc in DIRECTIONS}
    _require(all(board[p] == bg for p in locks), "each inward sample must be a hole")
    _require(all(board[p] == carrier for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))),
             "four fixed corners must expose the crop")
    outside = a != bg
    outside[top:bottom, left:right] = False
    comps = _components(map(tuple, np.argwhere(outside)))
    _require(len(comps) >= 2, "missing exterior inventory")
    _require(not np.any(a[outside] == carrier), "carrier cannot occur in loose pieces")
    # A one-cell moat separates the board and inventory. Clues may touch the canvas edge.
    near = np.zeros(a.shape, dtype=bool)
    near[max(0, top-1):min(a.shape[0], bottom+1),
         max(0, left-1):min(a.shape[1], right+1)] = True
    _require(not np.any(outside & near), "inventory touches the board")
    pieces = [_normalize([(int(r), int(c), int(a[r, c])) for r, c in comp]) for comp in comps]
    _require(sum(map(len, pieces)) == int(np.count_nonzero(board == bg)), "wrong inventory area")
    return dict(background=bg, carrier=carrier, bbox=(top, left, h, w),
                board=board, center=center, locks=locks, pieces=pieces, exterior=comps)


class _SearchLimit(Exception):
    pass


def _assemble(info, operation=2, use_locks=True, expected=None, limit=2,
              exact_inventory=True):
    """Exact cover; inventory swaps count as one answer if their colored grids agree.

    expected is used only by demonstration-coverage tests, never by solve.
    Two answers prove ambiguity. Exceeding the node bound never proves uniqueness.
    """
    board, bg = info['board'], info['background']
    if expected is not None:
        expected = _grid(expected)
    holes = [tuple(map(int, p)) for p in np.argwhere(board == bg)]
    positions = {p: i for i, p in enumerate(holes)}
    inventory = {}
    for piece in info['pieces']:
        t = _transform(piece, operation)
        inventory[t] = inventory.get(t, 0)+1
    types = sorted(inventory)
    by_cell = [[] for _ in holes]
    locks = info['locks'] if use_locks else {}
    for t, piece in enumerate(types):
        sh, sw = max(p[0] for p in piece)+1, max(p[1] for p in piece)+1
        for r in range(board.shape[0]-sh+1):
            for c in range(board.shape[1]-sw+1):
                cells = [(r+dy, c+dx, v) for dy, dx, v in piece]
                if any((y, x) not in positions or
                       ((y, x) in locks and locks[y, x] != v) or
                       (expected is not None and expected[y, x] != v)
                       for y, x, v in cells):
                    continue
                indices = [(positions[y, x], v) for y, x, v in cells]
                mask = sum(1 << i for i, _ in indices)
                paint = sum((v+1) << (4*i) for i, v in indices)
                move = (t, mask, paint)
                for i, _ in indices:
                    by_cell[i].append(move)
    nodes = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal nodes
        nodes += 1
        if nodes > SEARCH_LIMIT:
            raise _SearchLimit("exact-cover node bound exceeded")
        if not remaining:
            return (0,) if not exact_inventory or not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length()-1
            choices = [move for move in by_cell[i]
                       if counts[move[0]] and move[1] & remaining == move[1]]
            if not choices:
                return ()
            if best is None or len(choices) < len(best):
                best = choices
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            if exact_inventory:
                rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if len(answers) >= limit:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes))-1, tuple(inventory[t] for t in types))
    answers = []
    for code in codes:
        answer = board.copy()
        for i, p in enumerate(holes):
            answer[p] = ((code >> (4*i)) & 15)-1
        answers.append(answer)
    return answers, nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    answers, _ = _assemble(info)
    _require(len(answers) == 1, "input must have one colored completion under the radial locks")
    return answers[0].copy()


def _regions(h, w, kr, kc):
    rows = ((0, kr), (kr, kr+3), (kr+3, h))
    cols = ((0, kc), (kc, kc+3), (kc+3, w))
    return [(rows[dr+1], cols[dc+1]) for dr, dc in DIRECTIONS]


def _random_mask(rng, allowed, anchor, size):
    """Connected growth, with optional straight or rectangular bias."""
    cells = {anchor}
    style = int(rng.integers(4))
    axis = int(rng.integers(2))
    while len(cells) < size:
        frontier = sorted({q for r, c in cells for q in
                           ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if q in allowed} - cells)
        if not frontier:
            break
        if style == 0:
            best = [p for p in frontier if p[axis] == anchor[axis]]
        elif style == 1:
            r0, c0 = min(p[0] for p in cells), min(p[1] for p in cells)
            r1, c1 = max(p[0] for p in cells), max(p[1] for p in cells)
            costs = [(max(r1, r)-min(r0, r)+1)*(max(c1, c)-min(c0, c)+1)
                     for r, c in frontier]
            best = [p for p, v in zip(frontier, costs) if v == min(costs)]
        else:
            best = frontier
        choices = best or frontier
        cells.add(choices[int(rng.integers(len(choices)))])
    return cells


def _paint(rng, cells, palette, textured=True):
    order = sorted(cells)
    color = int(rng.choice(palette))
    style = int(rng.integers(4)) if textured else 0
    values = []
    for r, c in order:
        if style == 0:
            v = color
        elif style == 1:
            v = int(palette[(r+c) % len(palette)])
        elif style == 2:
            v = int(palette[(r if rng.random() < .5 else c) % len(palette)])
        else:
            v = int(rng.choice(palette))
        values.append((r, c, v))
    return tuple(values)


def _scatter(rng, board, removed, bg):
    h, w = board.shape
    pieces = [_transform(_normalize(p), 2) for p in removed]
    for attempt in range(10):
        ch = int(rng.integers(max(18, h+5), 31)) if attempt < 8 else 30
        cw = int(rng.integers(max(18, w+5), 31)) if attempt < 8 else 30
        br, bc = int(rng.integers(ch-h+1)), int(rng.integers(cw-w+1))
        canvas = np.full((ch, cw), bg, dtype=np.int8)
        canvas[br:br+h, bc:bc+w] = board
        blocked = np.zeros(canvas.shape, dtype=bool)
        blocked[max(0, br-1):min(ch, br+h+1), max(0, bc-1):min(cw, bc+w+1)] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i]), float(rng.random())))
        for i in order:
            p = pieces[i]
            sh, sw = max(x[0] for x in p)+1, max(x[1] for x in p)+1
            candidates = [(r, c) for r in range(ch-sh+1) for c in range(cw-sw+1)
                          if all(not blocked[r+dy, c+dx] for dy, dx, _ in p)]
            if not candidates:
                break
            edges = [(r, c) for r, c in candidates if
                     r == 0 or c == 0 or r+sh == ch or c+sw == cw]
            choices = edges if edges and rng.random() < .25 else candidates
            r, c = choices[int(rng.integers(len(choices)))]
            for dy, dx, v in p:
                y, x = r+dy, c+dx
                canvas[y, x] = v
                for yy, xx in ((y, x), (y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if 0 <= yy < ch and 0 <= xx < cw:
                        blocked[yy, xx] = True
        else:
            if np.count_nonzero(canvas == bg) > canvas.size / 2:
                return canvas
    return None


def _propose(rng, margins, palette, richness):
    north, south, west, east = margins
    h, w, kr, kc = north+south+3, west+east+3, north, west
    bg, carrier = map(int, palette[:2])
    payload = list(map(int, palette[2:2+int(rng.integers(2, 7))]))
    board = np.full((h, w), carrier, dtype=np.int8)
    removed, occupied, twin_data = [], set(), []
    regions = _regions(h, w, kr, kc)
    center = kr+1, kc+1
    corners = {(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)}
    # Two congruent rectangular sockets make color-lock information necessary.
    twins = set(map(int, rng.choice([0, 2, 5, 7], 2, replace=False)))
    rh, rw = (2, 3) if rng.random() < .5 else (3, 2)
    for i, ((r0, r1), (c0, c1)) in enumerate(regions):
        dr, dc = DIRECTIONS[i]
        anchor = center[0]+2*dr, center[1]+2*dc
        allowed = {(r, c) for r in range(r0, r1) for c in range(c0, c1)} - corners
        if i in twins:
            tr = r1-rh if dr < 0 else r0
            tc = c1-rw if dc < 0 else c0
            cells = {(tr+r, tc+c) for r in range(rh) for c in range(rw)}
            _require(cells <= allowed, "twin rectangle outside region")
            # Random asymmetric textures, rather than a fixed color stamp.
            for _ in range(30):
                p = tuple((r, c, int(rng.choice(payload))) for r, c in sorted(cells))
                if len({_transform(_normalize(p), op) for op in range(8)}) == 8:
                    break
            else:
                return None
        else:
            size = int(rng.integers(2, min(len(allowed), 7+3*richness)+1))
            cells = _random_mask(rng, allowed, anchor, size)
            p = _paint(rng, cells, payload)
        occupied.update(cells)
        removed.append(p)
        if i in twins:
            tr, tc = min(x[0] for x in p), min(x[1] for x in p)
            twin_data.append((len(removed)-1, (anchor[0]-tr, anchor[1]-tc)))
        for r, c, v in p:
            board[r, c] = bg
        # Secondary pieces can abut the primary, requiring exact cover of shared gaps.
        if richness and rng.random() < .45 and len(removed)-i-1 < 4:
            space = allowed-cells
            if len(space) >= 2:
                frontier = sorted({q for r, c in cells for q in
                                   ((r-1, c), (r+1, c), (r, c-1), (r, c+1)) if q in space})
                choices = frontier or sorted(space)
                start = choices[int(rng.integers(len(choices)))]
                extra = _random_mask(rng, space, start, int(rng.integers(2, min(6, len(space))+1)))
                if len(extra) >= 2:
                    occupied.update(extra)
                    ep = _paint(rng, extra, payload)
                    removed.append(ep)
                    for r, c, _ in ep:
                        board[r, c] = bg
    # A one-way compatibility relation: either twin can satisfy socket one,
    # but only the second twin satisfies socket two. Exact use forces the first
    # twin into socket one. Reusable templates admit a different colored grid.
    (i1, s1), (i2, s2) = twin_data
    first = {(r, c): v for r, c, v in _normalize(removed[i1])}
    second = {(r, c): v for r, c, v in _normalize(removed[i2])}
    second[s1] = first[s1]
    second[s2] = int(rng.choice([v for v in payload if v != first[s2]]))
    p2 = tuple((r, c, v) for (r, c), v in sorted(second.items()))
    if len({_transform(p2, op) for op in range(8)}) != 8:
        return None
    tr, tc = min(x[0] for x in removed[i2]), min(x[1] for x in removed[i2])
    removed[i2] = tuple((tr+r, tc+c, v) for r, c, v in p2)
    if len(occupied)+8 >= h*w/2:
        return None
    expected = board.copy()
    for p in removed:
        for r, c, v in p:
            expected[r, c] = v
    for dr, dc in DIRECTIONS:
        v = int(expected[center[0]+2*dr, center[1]+2*dc])
        board[center[0]+dr, center[1]+dc] = v
        expected[center[0]+dr, center[1]+dc] = v
    canvas = _scatter(rng, board, removed, bg)
    if canvas is None:
        return None
    # All acceptance uses the returned cells, never the hidden placement history.
    info = _parse(canvas)
    try:
        answers, _ = _assemble(info)
        unconstrained, _ = _assemble(info, use_locks=False)
        reusable, _ = _assemble(info, exact_inventory=False)
    except _SearchLimit:
        return None
    if (len(answers) != 1 or not np.array_equal(answers[0], expected) or
            len(unconstrained) != 2 or len(reusable) != 2):
        return None
    return dict(input=canvas, output=expected)


def _new_pair(rng, margins, palette, richness):
    for _ in range(100):
        pair = _propose(rng, margins, palette, richness)
        if pair is not None:
            return pair
    # A bounded simpler retry regime retains the same jigsaw and color locks.
    for _ in range(100):
        pair = _propose(rng, margins, palette, 0)
        if pair is not None:
            return pair
    raise RuntimeError("could not construct a unique radial-lock jigsaw")


def _operations_for_pair(pair):
    info = _parse(pair['input'])
    return {op for op in range(8) if _assemble(info, operation=op,
                                              expected=pair['output'], limit=1)[0]}


def _counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10),
                               ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name+" must be an integer")
        if not lo <= value <= hi:
            raise ValueError(name+" out of range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    fixed_palette = rng.random() < .35
    richness = int(rng.integers(2))
    # Independently varied radial distances; a test recombines demonstrated extents.
    a = tuple(map(int, rng.integers(3, 8, size=4)))
    b = tuple(map(int, rng.integers(3, 8, size=4)))
    while a == b or all(a[j] == b[j] for j in (1, 3)) or all(a[j] == b[j] for j in (0, 2)):
        b = tuple(map(int, rng.integers(3, 8, size=4)))
    query_margins = ((a[0], b[1], a[2], b[3]), (b[0], a[1], b[2], a[3]))
    train_margins = [a, b]
    for _ in range(n-2):
        trial = tuple(map(int, rng.integers(3, 8, size=4)))
        if trial in query_margins:
            choices = [(v, *trial[1:]) for v in range(3, 8)
                       if (v, *trial[1:]) not in query_margins]
            trial = choices[int(rng.integers(len(choices)))]
        train_margins.append(trial)
    puzzle, seen = {'train': [], 'test': []}, set()
    candidates = set(range(8))
    for i in range(n):
        for _ in range(20):
            pp = palette if fixed_palette else rng.permutation(10)
            pair = _new_pair(rng, train_margins[i], pp, richness if i >= 2 else i)
            if _key(pair['input']) in seen:
                continue
            ops = _operations_for_pair(pair)
            # At two demonstrations the global transformation must be determined.
            if i == 1 and candidates & ops != {2}:
                continue
            candidates &= ops
            seen.add(_key(pair['input']))
            puzzle['train'].append(pair)
            break
        else:
            raise RuntimeError("could not expose half-turn convention")
    for i in range(m):
        # N/W distances from one demonstration; S/E from another. First query
        # is a guaranteed new extent vector, without compulsory scale inflation.
        margins = query_margins[i % 2]
        for _ in range(20):
            pp = palette if fixed_palette else rng.permutation(10)
            pair = _new_pair(rng, margins, pp, int(rng.integers(2)))
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle['test'].append(pair)
                break
        else:
            raise RuntimeError("duplicate query construction")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, b = _grid(pair['input']), _grid(pair['output'])
    info = _parse(a)
    board, bg, carrier = info['board'], info['background'], info['carrier']
    _require(b.shape == board.shape, "answer must be the exact scaffold crop")
    fixed = board != bg
    _require(np.array_equal(board[fixed], b[fixed]), "fixed scaffold or hub changed")
    _require(not np.any(b == bg), "unfilled holes")
    for p, value in info['locks'].items():
        _require(b[p] == value, "inward cell disagrees with the radial key")
    answers, _ = _assemble(info)
    _require(len(answers) == 1 and np.array_equal(answers[0], b), "wrong or nonunique exact inventory assembly")
    for color in range(10):
        if color != bg:
            _require(np.count_nonzero(a == color) == np.count_nonzero(b == color),
                     "per-color material conservation failed")
    unconstrained, _ = _assemble(info, use_locks=False)
    _require(len(unconstrained) == 2, "radial locks are not necessary for this pair")
    reusable, _ = _assemble(info, exact_inventory=False)
    _require(len(reusable) == 2, "finite inventory is not necessary for this pair")
    _require(carrier not in {v for p in info['pieces'] for _, _, v in p}, "loose scaffold color")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, operations, extents = set(), set(range(8)), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            kr, kc = info['center'][0]-1, info['center'][1]-1
            extents[split].append((kr, info['board'].shape[0]-kr-3,
                                   kc, info['board'].shape[1]-kc-3))
            k = _key(pair['input'])
            _require(k not in seen, "duplicate inputs")
            seen.add(k)
            if split == 'train':
                operations &= _operations_for_pair(pair)
    _require(operations == {2}, "demonstrations do not distinguish the global half-turn")
    for vector in extents['test']:
        _require(vector not in extents['train'], "missing novel extent combination")
        _require(all(vector[j] in {t[j] for t in extents['train']} for j in range(4)),
                 "query extent was not individually demonstrated")


def _features(pair):
    info = _parse(pair['input'])
    board, center = info['board'], info['center']
    kr, kc = center[0]-1, center[1]-1
    holes = board == info['background']
    gaps = _components(map(tuple, np.argwhere(holes)))
    answer = np.asarray(pair['output'])
    return dict(board_shape=board.shape, canvas_shape=np.asarray(pair['input']).shape,
                margins=(kr, board.shape[0]-kr-3, kc, board.shape[1]-kc-3),
                origin=info['bbox'][:2], background=info['background'], carrier=info['carrier'],
                pieces=len(info['pieces']), area=int(holes.sum()),
                shapes={tuple((r, c) for r, c, _ in p) for p in info['pieces']},
                multicolor=sum(len({v for _, _, v in p}) > 1 for p in info['pieces']),
                payload_colors=len({v for p in info['pieces'] for _, _, v in p}),
                fixed_components=len(_components(map(tuple, np.argwhere(board != info['background'])))),
                edge_clues=any(r in (0, np.asarray(pair['input']).shape[0]-1) or
                               c in (0, np.asarray(pair['input']).shape[1]-1)
                               for comp in info['exterior'] for r, c in comp),
                shared_gap=any(len({int(answer[p]) for p in comp}) > 1 and
                               len(comp) > max(map(len, info['pieces'])) for comp in gaps),
                boundary=bool(np.any(holes[0]) or np.any(holes[-1]) or
                              np.any(holes[:, 0]) or np.any(holes[:, -1])),
                structure=(board.shape, tuple(map(bool, holes.flat))),
                sample_colors=len(set(info['locks'].values())))


def _expect_failure(call):
    try:
        call()
    except (AssertionError, ValueError, _SearchLimit):
        return
    raise AssertionError("corruption was accepted")


def self_test():
    """Bounded 42-seed diversity, ablations, corruption, counts, and determinism."""
    features, puzzles = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]))
        local = [_features(pair) for s in ('train', 'test') for pair in p[s]]
        features.extend(local)
        _require(len({f['margins'] for f in local}) >= 3, "missing extent recombination shift")
        # solve accepts ordinary lists too, and cannot mutate its input.
        q = np.asarray(p['test'][0]['input']).copy()
        before = q.copy()
        _require(np.array_equal(solve(q), p['test'][0]['output']), "array-input solve failed")
        _require(np.array_equal(solve(q.tolist()), p['test'][0]['output']), "list-input solve failed")
        _require(np.array_equal(q, before), "solver mutated input")
    _require(len(puzzles) == 42, "seed diversity missing")
    _require(len({f['structure'] for f in features}) >= 150, "structural diversity missing")
    _require(len({f['board_shape'] for f in features}) >= 35, "board proportions collapsed")
    _require(len({f['canvas_shape'] for f in features}) >= 50, "canvas geometry collapsed")
    _require(len({f['origin'] for f in features}) >= 50, "board positions collapsed")
    _require(len({f['payload_colors'] for f in features}) >= 4, "payload richness collapsed")
    _require(len(set().union(*(f['shapes'] for f in features))) >= 60, "polyomino diversity missing")
    _require({f['background'] for f in features} == set(range(10)), "background roles missing")
    _require({f['carrier'] for f in features} == set(range(10)), "carrier roles missing")
    _require(min(f['pieces'] for f in features) == 8 and max(f['pieces'] for f in features) >= 11,
             "piece counts collapsed")
    _require(sum(f['shared_gap'] for f in features) >= 10, "shared-gap assembly missing")
    _require(sum(f['boundary'] for f in features) >= 10, "boundary pieces missing")
    _require(sum(f['edge_clues'] for f in features) >= 10, "edge inventory missing")
    _require(max(f['sample_colors'] for f in features) >= 5, "directional color variation missing")
    p = generate_puzzle(314159, 2, 1)
    signature = tuple(_key(pair[k]) for s in p for pair in p[s] for k in ('input', 'output'))
    generate_puzzle(17, 3, 1)
    again = generate_puzzle(314159, 2, 1)
    _require(signature == tuple(_key(pair[k]) for s in again for pair in again[s]
                                 for k in ('input', 'output')), "nondeterminism")
    # All legal counts are checked by verify.py; include limits here as well.
    for n, m in ((2, 1), (10, 1), (10, 4), (7, 3)):
        q = generate_puzzle(400+n+m, n, m)
        _require((len(q['train']), len(q['test'])) == (n, m), "counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        _expect_failure(lambda: generate_puzzle(0, n, m))
    for kind in ('hub', 'fixed', 'inventory', 'answer'):
        broken = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = broken['test'][0]
        info = _parse(pair['input'])
        top, left, _, _ = info['bbox']
        if kind == 'hub':
            r, c = info['center']
            pair['input'][top+r-1, left+c] = info['carrier']
        elif kind == 'fixed':
            pair['output'][0, 0] = info['background']
        elif kind == 'inventory':
            pair['input'][info['exterior'][0][0]] = info['background']
        else:
            sample = next(iter(info['locks']))
            pair['output'][sample] = info['carrier']
        _expect_failure(lambda: validate_generated_puzzle(broken))


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
