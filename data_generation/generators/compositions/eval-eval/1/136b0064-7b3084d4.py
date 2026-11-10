"""Reassemble the current, then follow it.

Only NumPy and the Python standard library are used. Nothing is written to disk.
The solver recovers every role, rigid placement and command from its input.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


# Clockwise: east, south, west, north. These are geometric arrowheads, not colors.
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
_ARROWS = tuple(np.rot90(np.array([[1, 0], [1, 1], [1, 0]], dtype=bool), -k)
                for k in range(4))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 182011681, 402546505, 378296222, 214059778, 46752872,
              784237673, 470394151, 932819003, 74293649, 414727594, 438017683,
              267978581, 660112618, 152486587, 972021560, 869914660, 871059855,
              139344267, 411743734, 431964669, 35632031, 223125101, 211970638,
              360759698, 803156951, 575858255, 644599869, 388672157, 235004168,
              382793565, 915864800, 787654625, 214487314, 744118463, 834091305,
              547974984, 111039289, 48492357, 345820221, 193268959, 694047056)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = [start], [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in _EIGHT:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    pending.append(p)
                    cells.append(p)
        rr, cc = zip(*cells)
        p = np.full((max(rr)-min(rr)+1, max(cc)-min(cc)+1), -1, dtype=np.int8)
        for r, c in cells:
            p[r-min(rr), c-min(cc)] = a[r, c]
        result.append(p)
    return result


def _connected(mask, neighbors=_EIGHT):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    todo = [remaining.pop()]
    while todo:
        r, c = todo.pop()
        for dr, dc in neighbors:
            p = r+dr, c+dc
            if p in remaining:
                remaining.remove(p)
                todo.append(p)
    return not remaining


def _arrow_direction(piece, ink):
    rr, cc = np.nonzero(piece == ink)
    _require(len(rr) == 4, "each piece carries one four-cell arrowhead")
    box = piece[rr.min():rr.max()+1, cc.min():cc.max()+1]
    _require(bool(np.all(box >= 0)), "arrow rectangle must lie inside its piece")
    mask = box == ink
    directions = [k for k, arrow in enumerate(_ARROWS) if np.array_equal(mask, arrow)]
    _require(len(directions) == 1, "unrecognized arrowhead")
    return directions[0]


def _read_input(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(bool(np.all(a[0] == bg) and np.all(a[-1] == bg) and
                  np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)),
             "an unbroken background border is required")
    pieces = _components(a, bg)
    _require(3 <= len(pieces) <= 5, "three to five separated rigid pieces required")
    colors, counts = np.unique(a, return_counts=True)
    marks = [int(c) for c, n in zip(colors, counts) if n == 1 and c != bg]
    _require(len(marks) == 1, "exactly one singleton anchor required")
    marker = marks[0]
    common = set.intersection(*(set(map(int, p[p >= 0])) for p in pieces))
    _require(len(common) == 1, "arrow ink is the only color shared by all pieces")
    ink = common.pop()
    _require(ink not in (bg, marker), "distinct structural roles required")
    carriers, directions = [], []
    for p in pieces:
        other = set(map(int, p[p >= 0])) - {ink, marker}
        _require(len(other) == 1, "each piece has its own carrier color")
        carrier = other.pop()
        _require(np.sum(p == carrier) >= 4, "carrier must remain identifiable")
        carriers.append(carrier)
        directions.append(_arrow_direction(p, ink))
    _require(len(set(carriers)) == len(pieces), "carrier colors must be distinct")
    area = sum(int(np.sum(p >= 0)) for p in pieces)
    side = int(area ** .5)
    _require(side * side == area, "piece area must be a perfect square")
    _require(1 <= side <= 30, "assembled side out of bounds")
    return dict(background=bg, ink=ink, marker=marker, pieces=pieces,
                carriers=carriers, input_directions=directions, side=side)


def _packing_solutions(facts, limit=2):
    """Enumerate colored rotation-only square packings, including internal ink.

    Two rotations with the same occupied mask but different arrows MUST remain
    separate candidates. Ownership of shared-ink cells is also part of the
    assembly. Routing is never used to choose a packing.
    """
    side, marker = facts['side'], facts['marker']
    pieces = facts['pieces']
    marked = [i for i, p in enumerate(pieces) if np.any(p == marker)]
    if len(marked) != 1:
        return []
    anchor = marked[0]
    candidates = []
    for i, p in enumerate(pieces):
        options, orientations = [], set()
        for k in range(4):
            q = np.rot90(p, -k)
            key = _key(q)
            if key in orientations:
                continue
            orientations.add(key)
            h, w = q.shape
            if h > side or w > side:
                continue
            if i == anchor:
                offsets = [(0, 0)] if q[0, 0] == marker else []
            else:
                offsets = itertools.product(range(side-h+1), range(side-w+1))
            rr, cc = np.nonzero(q >= 0)
            vals = q[rr, cc].copy()
            base = sum(1 << (int(r)*side+int(c)) for r, c in zip(rr, cc))
            direction = (facts['input_directions'][i]+k) % 4
            for r, c in offsets:
                options.append((base << (r*side+c), rr+r, cc+c, vals, direction))
        if not options:
            return []
        candidates.append(options)
    full = (1 << (side*side))-1
    chosen = [None]*len(pieces)
    answers, signatures = [], set()

    def search(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied != full:
                return
            board = np.empty((side, side), dtype=np.int8)
            owners = np.empty_like(board)
            ds = []
            for i, option in enumerate(chosen):
                _, rr, cc, values, d = option
                board[rr, cc], owners[rr, cc] = values, i
                ds.append(d)
            signature = (board.tobytes(), owners.tobytes(), tuple(ds))
            if signature not in signatures:
                signatures.add(signature)
                answers.append((board, owners, tuple(ds)))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            rest = tuple(j for j in remaining if j != i)
            for option in candidates[i]:
                if option[0] & first and not option[0] & occupied:
                    chosen[i] = option
                    search(occupied | option[0], rest)
                    if len(answers) >= limit:
                        return

    rest = tuple(i for i in range(len(pieces)) if i != anchor)
    for option in candidates[anchor]:
        chosen[anchor] = option
        search(option[0], rest)
        if len(answers) >= limit:
            break
    return answers


def _walk(owners, directions, stop_at_piece_reentry=False):
    """A bounded deterministic trajectory. Each cell is visited at most once."""
    h, w = owners.shape
    r = c = 0
    cells, sequence, seen, visited_pieces = [], [], set(), set()
    turns = 0
    old_d = None
    while 0 <= r < h and 0 <= c < w and (r, c) not in seen:
        i = int(owners[r, c])
        if not sequence or sequence[-1] != i:
            if stop_at_piece_reentry and i in visited_pieces:
                return cells, sequence, turns, 'piece_reentry'
            sequence.append(i)
            visited_pieces.add(i)
        d = directions[i]
        if old_d is not None and old_d != d:
            turns += 1
        old_d = d
        seen.add((r, c))
        cells.append((r, c))
        dr, dc = _DIRECTIONS[d]
        r, c = r+dr, c+dc
    terminal = 'loop' if (r, c) in seen else 'exit'
    return cells, sequence, turns, terminal


def _render(owners, directions, carriers, bg, marker, stop_at_piece_reentry=False):
    cells, sequence, turns, terminal = _walk(owners, directions, stop_at_piece_reentry)
    out = np.full(owners.shape, bg, dtype=np.int8)
    for r, c in cells:
        out[r, c] = carriers[int(owners[r, c])]
    out[0, 0] = marker
    return out, (cells, sequence, turns, terminal)


def _analyze(input_grid):
    facts = _read_input(input_grid)
    solutions = _packing_solutions(facts)
    _require(len(solutions) == 1, "exactly one colored anchored assembly required")
    board, owners, ds = solutions[0]
    out, route = _render(owners, ds, facts['carriers'], facts['background'], facts['marker'])
    facts.update(board=board, owners=owners, directions=ds, route=route, output=out)
    return facts


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the colored thread using only this input, never demonstrations."""
    return _analyze(input_grid)['output']


def _grow(rng, side, count, style):
    a = np.full((side, side), -1, dtype=np.int8)
    corners = [0, side-1, side*side-1, side*(side-1)]
    if style == 'corners':
        points = corners[:count]
        if count == 5:
            points.append(int(rng.integers(1, side-1))*side+int(rng.integers(1, side-1)))
    else:
        points = [0]+list(map(int, rng.choice(np.arange(1, side*side), count-1, replace=False)))
    diagonal = style == 'diagonal'
    neighbors = _EIGHT if diagonal else _DIRECTIONS
    adjacent = []
    for r in range(side):
        for c in range(side):
            adjacent.append([rr*side+cc for dr, dc in neighbors
                             if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side])
    front = [set() for _ in points]
    sizes = [1]*count
    weights = np.exp(rng.uniform(-.65, .65, count))
    for i, pos in enumerate(points):
        a.flat[pos] = i
    for i, pos in enumerate(points):
        front[i].update(p for p in adjacent[pos] if a.flat[p] < 0)
    for _ in range(side*side-count):
        eligible = [i for i in range(count) if front[i]]
        i = min(eligible, key=lambda j: sizes[j]/weights[j]+float(rng.random())*.8)
        choices = sorted(front[i])
        if diagonal and rng.random() < .9:
            edge = [p for p in choices if any(0 <= (rr := p//side+dr) < side and
                    0 <= (cc := p%side+dc) < side and a[rr, cc] == i for dr, dc in _DIRECTIONS)]
            if edge:
                choices = edge
        pos = choices[int(rng.integers(len(choices)))]
        a.flat[pos] = i
        sizes[i] += 1
        for f in front:
            f.discard(pos)
        front[i].update(p for p in adjacent[pos] if a.flat[p] < 0)
    return a


def _cuts(rng, side, count):
    rectangles = [(0, 0, side, side)]
    while len(rectangles) < count:
        possibilities = []
        for i, (r, c, h, w) in enumerate(rectangles):
            possibilities.extend((i, 0, k) for k in range(2, h-1) if min(k, h-k)*w >= 9)
            possibilities.extend((i, 1, k) for k in range(2, w-1) if min(k, w-k)*h >= 9)
        if not possibilities:
            return None
        i, axis, k = possibilities[int(rng.integers(len(possibilities)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, k, w), (r+k, c, h-k, w)] if axis == 0 else
                          [(r, c, h, k), (r, c+k, h, w-k)])
    a = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        a[r:r+h, c:c+w] = i
    for _ in range(side*5):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _DIRECTIONS[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if not (0 <= rr < side and 0 <= cc < side) or (r, c) == (0, 0):
            continue
        x, y = int(a[r, c]), int(a[rr, cc])
        if x == y or np.sum(a == x) <= 9:
            continue
        a[r, c] = y
        if not _connected(a == x, _DIRECTIONS):
            a[r, c] = x
    return a


def _icon_sites(labels, count):
    side = len(labels)
    result = []
    for i in range(count):
        per_direction = []
        for arrow in _ARROWS:
            h, w = arrow.shape
            sites = [(r, c) for r in range(side-h+1) for c in range(side-w+1)
                     if (r, c) != (0, 0) and np.all(labels[r:r+h, c:c+w] == i)]
            per_direction.append(sites)
        result.append(per_direction)
    return result


def _routing_options(labels, sites, terminal, require_reentry):
    count = len(sites)
    choices = [tuple(d for d in range(4) if sites[i][d]) for i in range(count)]
    if any(not p for p in choices):
        return []
    answers = []
    for ds in itertools.product(*choices):
        if ds[int(labels[0, 0])] not in (0, 1):
            continue
        cells, seq, turns, end = _walk(labels, ds)
        if terminal is not None and end != terminal:
            continue
        if len(set(seq)) != count or turns < 2 or len(cells) > labels.size-3:
            continue
        # Direct reversals just retrace one edge and hide the boundary mechanism.
        if any((ds[y]-ds[x]) % 4 == 2 for x, y in zip(seq, seq[1:])):
            continue
        if require_reentry:
            prefix = _walk(labels, ds, True)[0]
            if len(cells)-len(prefix) < 2:
                continue
        answers.append(ds)
    return answers


def _make_board(rng, labels, directions, sites, carriers, ink, marker):
    board = np.empty_like(labels)
    for i, color in enumerate(carriers):
        board[labels == i] = color
        d = directions[i]
        r, c = sites[i][d][int(rng.integers(len(sites[i][d])))]
        arrow = _ARROWS[d]
        board[r:r+arrow.shape[0], c:c+arrow.shape[1]][arrow] = ink
    board[0, 0] = marker
    pieces = []
    for i in range(len(carriers)):
        rr, cc = np.nonzero(labels == i)
        r0, r1, c0, c1 = rr.min(), rr.max()+1, cc.min(), cc.max()+1
        p = np.where(labels[r0:r1, c0:c1] == i, board[r0:r1, c0:c1], -1)
        pieces.append(p.astype(np.int8))
    return board, pieces


def _scatter(rng, pieces, bg):
    """Place actual masks, without output-sized reserved slots; bounded retries."""
    n = 26
    for attempt in range(12):
        oriented = [np.rot90(p, int(rng.integers(4))).copy() for p in pieces]
        order = list(map(int, rng.permutation(len(pieces))))
        if attempt < 6:
            order.sort(key=lambda i: -int(np.sum(oriented[i] >= 0)))
        canvas = np.full((n, n), -1, dtype=np.int8)
        blocked, maxr, maxc = 0, 0, 0
        aspect = float(rng.uniform(.7, 1.5))
        okay = True
        for i in order:
            p = oriented[i]
            h, w = p.shape
            coords = list(map(tuple, np.argwhere(p >= 0)))
            base = sum(1 << (int(r)*n+int(c)) for r, c in coords)
            options = []
            for r in range(n-h+1):
                for c in range(n-w+1):
                    if (base << (r*n+c)) & blocked:
                        continue
                    hr, wc = max(maxr, r+h), max(maxc, c+w)
                    score = hr*wc+abs(hr-aspect*wc)*2+float(rng.random())*18
                    options.append((score, r, c))
            if not options:
                okay = False
                break
            _, r, c = min(options)
            for pr, pc in coords:
                rr, cc = r+int(pr), c+int(pc)
                canvas[rr, cc] = p[pr, pc]
                for nr in range(max(0, rr-1), min(n, rr+2)):
                    for nc in range(max(0, cc-1), min(n, cc+2)):
                        blocked |= 1 << (nr*n+nc)
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        if not okay:
            continue
        rr, cc = np.nonzero(canvas >= 0)
        cropped = canvas[rr.min():rr.max()+1, cc.min():cc.max()+1]
        h, w = cropped.shape
        margins = [int(rng.integers(1, 3)) for _ in range(4)]
        top, bottom, left, right = margins
        out = np.full((h+top+bottom, w+left+right), bg, dtype=np.int8)
        out[top:top+h, left:left+w] = np.where(cropped >= 0, cropped, bg)
        return out
    return None


def _generate_pair(rng, side, count, roles, terminal, reentry, seen, prior_colors=None):
    bg, ink, marker = roles
    styles = ('corners', 'growth', 'cuts', 'diagonal')
    carriers = list(map(int, rng.choice([c for c in range(10) if c not in roles], count, replace=False)))
    for attempt in range(800):
        style = styles[attempt % len(styles)]
        labels = _cuts(rng, side, count) if style == 'cuts' else _grow(rng, side, count, style)
        if labels is None or any(np.sum(labels == i) < 9 for i in range(count)):
            continue
        sites = _icon_sites(labels, count)
        options = _routing_options(labels, sites, terminal, reentry)
        if not options:
            continue
        directions = options[int(rng.integers(len(options)))]
        colors = list(carriers)
        if prior_colors:
            # Explicitly refute a color-to-direction dictionary in two demos.
            color = int(rng.choice(sorted(prior_colors)))
            choices = [i for i, d in enumerate(directions) if d not in prior_colors[color]]
            if not choices:
                continue
            i = choices[int(rng.integers(len(choices)))]
            if color in colors:
                j = colors.index(color)
                colors[i], colors[j] = colors[j], colors[i]
            else:
                colors[i] = color
        board, pieces = _make_board(rng, labels, directions, sites, colors, ink, marker)
        facts = dict(side=side, marker=marker, ink=ink, pieces=pieces,
                     input_directions=list(directions), carriers=colors)
        answers = _packing_solutions(facts)
        if len(answers) != 1 or not np.array_equal(answers[0][0], board):
            continue
        expected = _render(labels, directions, colors, bg, marker)[0]
        for _ in range(8):
            inp = _scatter(rng, pieces, bg)
            if inp is None or _key(inp) in seen:
                continue
            # The scatter order changes piece indexing; derive the ablation from
            # the actual input rather than assuming construction identities.
            found = _analyze(inp)
            frozen = _render(found['owners'], found['input_directions'], found['carriers'], bg, marker)[0]
            if np.array_equal(frozen, expected):
                continue
            _require(np.array_equal(found['output'], expected), "construction/solve mismatch")
            seen.add(_key(inp))
            return {'input': inp, 'output': expected}
    raise RuntimeError(f"finite construction search exhausted for side {side}, count {count}, {terminal}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    result, seen = {'train': [], 'test': []}, set()
    prior_colors = {}
    for i in range(nt+nq):
        split = 'train' if i < nt else 'test'
        if i == 0:
            side, count, terminal, reentry = 7, 3, 'exit', False
        elif i == 1:
            side, count, terminal, reentry = 8, 4, 'loop', True
        elif split == 'train':
            side = int(rng.integers(6, 9))
            count = int(rng.integers(3, min(5, side*side//10)+1))
            terminal, reentry = None, False
        else:
            side, count = int(rng.integers(9, 11)), int(rng.integers(3, 6))
            terminal, reentry = None, True
        pair = _generate_pair(rng, side, count, roles, terminal, reentry, seen,
                              prior_colors if i == 1 else None)
        result[split].append(pair)
        if i == 0:
            f = _analyze(pair['input'])
            for color, d in zip(f['carriers'], f['directions']):
                prior_colors.setdefault(color, set()).add(d)
    return result


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, roles, records = set(), set(), {'train': [], 'test': []}
    for split in records:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, output = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _analyze(inp)
            _require(np.array_equal(output, f['output']), "incorrect thread")
            _require(output.shape == (f['side'], f['side']), "preserve the square's coordinate frame")
            _require(output[0, 0] == f['marker'] and np.sum(output == f['marker']) == 1, "anchor preservation")
            _require(not np.any(output == f['ink']), "arrows are instructions, not output ink")
            cells, sequence, turns, terminal = f['route']
            _require(set(sequence) == set(range(len(f['pieces']))), "every piece participates in the route")
            _require(turns >= 2, "routing must expose changes of command")
            _require(np.sum(output == f['background']) >= 3, "do not just return the assembled board")
            frozen = _render(f['owners'], f['input_directions'], f['carriers'], f['background'], f['marker'])[0]
            _require(not np.array_equal(output, frozen), "rotating geometry must rotate the instruction")
            for i in range(len(f['pieces'])):
                changed = list(f['directions'])
                alternatives = []
                for d in range(4):
                    if d == f['directions'][i]:
                        continue
                    changed[i] = d
                    alternatives.append(_render(f['owners'], changed, f['carriers'],
                                                f['background'], f['marker'])[0])
                _require(any(not np.array_equal(output, x) for x in alternatives),
                         "every visited piece's command must affect the answer")
            roles.add((f['background'], f['ink'], f['marker']))
            records[split].append(f)
    _require(len(roles) == 1, "role conventions must agree within a puzzle")
    tr, te = records['train'], records['test']
    _require(tr[0]['route'][3] == 'exit' and tr[1]['route'][3] == 'loop', "show both terminal behaviors")
    _require(set(tr[1]['directions']) == {0, 1, 2, 3}, "demonstrate every arrow orientation")
    by_color = {}
    for f in tr:
        for color, d in zip(f['carriers'], f['directions']):
            by_color.setdefault(color, set()).add(d)
    _require(any(len(ds) > 1 for ds in by_color.values()), "carrier color cannot encode direction")
    _require(len({len(f['pieces']) for f in tr}) >= 2, "training must show variable piece count")
    _require(min(f['side'] for f in te) > max(f['side'] for f in tr), "test scale shift")
    for f in [tr[1]]+te:
        prefix = _walk(f['owners'], f['directions'], True)[0]
        _require(len(f['route'][0])-len(prefix) >= 2, "demonstrate genuine piece re-entry")
    return None


def puzzle_features(puzzle):
    """Small input-derived records, never reconstructible grid payloads."""
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _analyze(pair['input'])
            cells, seq, turns, terminal = f['route']
            owner_key = f['owners'].tobytes()
            records.append(dict(split=split, side=f['side'], count=len(f['pieces']),
                                input_shape=list(np.asarray(pair['input']).shape),
                                steps=len(cells), turns=turns, terminal=terminal,
                                reentries=len(seq)-len(set(seq)),
                                diagonal=sum(not _connected(p >= 0, _DIRECTIONS) for p in f['pieces']),
                                areas=[int(np.sum(p >= 0)) for p in f['pieces']],
                                directions=list(f['directions']),
                                roles=[f[k] for k in ('background', 'ink', 'marker')],
                                structure=hashlib.sha256(owner_key).hexdigest()))
    return records


def _puzzle_key(puzzle):
    return tuple(_key(np.asarray(pair[k])) for split in ('train', 'test')
                 for pair in puzzle[split] for k in ('input', 'output'))


def self_test():
    """42 seeds, streamed in memory, no files, corpus cache, logs or traces."""
    structures, shapes, roles, counts, terminals = set(), set(), set(), set(), set()
    min_steps, max_steps, total = 1000, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for f in puzzle_features(puzzle):
            structures.add(f['structure'])
            shapes.add(tuple(f['input_shape']))
            roles.add(tuple(f['roles']))
            counts.add(f['count'])
            terminals.add(f['terminal'])
            min_steps, max_steps = min(min_steps, f['steps']), max(max_steps, f['steps'])
            total += 1
    _require(len(structures) >= 180 and len(shapes) >= 35, "structural diversity")
    _require(len(roles) >= 30 and counts == {3, 4, 5}, "palette and count variation")
    _require(terminals == {'exit', 'loop'} and max_steps-min_steps >= 15, "routing diversity")
    a = generate_puzzle(123, 2, 1)
    _require(_puzzle_key(a) == _puzzle_key(generate_puzzle(123, 2, 1)), "determinism")
    _require(_puzzle_key(a) != _puzzle_key(generate_puzzle(124, 2, 1)), "seed variation")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Semantic damage and command-direction damage must both be detected.
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in a[s]] for s in a}
    f = _analyze(damaged['test'][0]['input'])
    r, c = f['route'][0][1]
    damaged['test'][0]['output'][r, c] = f['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged output accepted')
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in a[s]] for s in a}
    inp = damaged['test'][0]['input']
    f = _read_input(inp)
    marker_position = tuple(map(int, np.argwhere(inp == f['marker'])[0]))
    # Locate the marked object's cells by an eight-neighbor flood in the input.
    pending, marked_cells = [marker_position], {marker_position}
    while pending:
        r, c = pending.pop()
        for dr, dc in _EIGHT:
            rr, cc = r+dr, c+dc
            if (0 <= rr < len(inp) and 0 <= cc < inp.shape[1] and
                    inp[rr, cc] != f['background'] and (rr, cc) not in marked_cells):
                marked_cells.add((rr, cc))
                pending.append((rr, cc))
    arrow_cells = [(r, c) for r, c in marked_cells if inp[r, c] == f['ink']]
    r0, c0 = min(r for r, c in arrow_cells), min(c for r, c in arrow_cells)
    r1, c1 = max(r for r, c in arrow_cells)+1, max(c for r, c in arrow_cells)+1
    box = inp[r0:r1, c0:c1]
    mask = box == f['ink']
    carrier = next(int(inp[r, c]) for r, c in marked_cells
                   if inp[r, c] not in (f['marker'], f['ink']))
    box[mask] = carrier
    box[np.rot90(mask, 2)] = f['ink']
    _require(np.sum(solve(inp) != f['background']) == 1, "reversed anchor command must exit immediately")
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed command with stale label accepted')
    # Symmetric inserts with asymmetric arrows have multiple colored rotations.
    labels = np.zeros((8, 8), dtype=np.int8)
    labels[1:4, 1:4], labels[4:7, 4:7] = 1, 2
    _, pieces = _make_board(np.random.default_rng(0), labels, (0, 0, 0),
                           _icon_sites(labels, 3), [2, 3, 4], 5, 6)
    ambiguity = dict(side=8, marker=6, ink=5, pieces=pieces,
                     input_directions=[0, 0, 0], carriers=[2, 3, 4])
    _require(len(_packing_solutions(ambiguity)) == 2,
             "never deduplicate colored rotations by occupied mask")
    return dict(seeds=42, pairs=total, structures=len(structures), input_shapes=len(shapes),
                role_assignments=len(roles), piece_counts=sorted(counts),
                terminal_modes=sorted(terminals), steps_range=[min_steps, max_steps])
