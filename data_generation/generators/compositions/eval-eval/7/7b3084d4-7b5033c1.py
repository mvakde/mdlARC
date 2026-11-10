"""Threaded jigsaw: jointly assemble rigid panels and reconnect their thread.

Only standard-library modules and NumPy are used. Generation keeps all candidates
in memory, saves no traces, and performs bounded retries. See task_spec.md.
"""
from __future__ import annotations

from hashlib import sha256
from math import isqrt

import numpy as np

_EDGE = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EIGHT = _EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))
HOST_SEEDS = (0, 551109681, 875431559, 798527156, 244574513, 898264785,
              998054605, 158471101, 494163345, 808105575, 640944667,
              865277034, 988805567, 871270338, 640204704, 356604301,
              952460197, 692847008, 493212936, 370401613, 678235177,
              803411226, 726071431, 827156586, 782662876, 590027853,
              492204038, 612914624, 321059029, 453766602, 931108152,
              259687987, 213606859, 364895319, 74290083, 863560603,
              704424245, 148908369, 565491236, 271420086, 844245748, 654995653)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'Grid dimensions must be 1..30.')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid cells must be integers.')
    _require(bool(np.all((0 <= a) & (a <= 9))), 'Colors must be 0..9.')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.ascontiguousarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, neighbors=_EIGHT):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        stack = [min(remaining)]
        remaining.remove(stack[0])
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        components.append(cells)
    return components


def _connected(mask):
    return len(_components(mask, _EDGE)) == 1


def _extract(board, labels, count):
    pieces = []
    for i in range(count):
        coords = np.argwhere(labels == i)
        lo, hi = coords.min(0), coords.max(0) + 1
        region = np.s_[lo[0]:hi[0], lo[1]:hi[1]]
        pieces.append(np.where(labels[region] == i, board[region], -1).astype(np.int8))
    return pieces


def _read_input(value):
    grid = _grid(value)
    histogram = np.bincount(grid.ravel(), minlength=10)
    background = int(histogram.argmax())
    _require(histogram[background] * 2 > grid.size, 'Background must occupy a strict majority.')
    foreground = int(grid.size - histogram[background])
    side = isqrt(foreground)
    _require(side * side == foreground and 5 <= side <= 9, 'Panel area must be a square of side 5..9.')
    counts = histogram.copy()
    counts[background] = 0
    backing = int(counts.argmax())
    _require(counts[backing] * 2 > foreground, 'Backing must be a strict majority of panel area.')
    singletons = np.flatnonzero((histogram == 1) & (np.arange(10) != background))
    _require(len(singletons) == 1, 'Exactly one singleton marker color is required.')
    marker = int(singletons[0])
    thread_count = foreground - int(counts[backing])
    data_colors = set(map(int, np.unique(grid))) - {background, backing, marker}
    _require(8 <= thread_count <= 30 and 2 <= len(data_colors) <= 5, 'Invalid thread length or palette.')
    _require(all(histogram[c] >= 2 for c in data_colors), 'Only the marker color may be a singleton.')
    cells = _components(grid != background)
    _require(3 <= len(cells) <= 5, 'There must be 3..5 separated panels.')
    pieces = []
    for comp in cells:
        rr, cc = np.array(comp).T
        r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()) + 1, int(cc.max()) + 1
        p = np.full((r1-r0, c1-c0), -1, dtype=np.int8)
        p[rr-r0, cc-c0] = grid[rr, cc]
        _require(len(comp) >= 5 and _connected(p >= 0), 'Each panel must be edge-connected with area >= 5.')
        _require(np.sum(p == backing) >= 2 and np.any((p >= 0) & (p != backing)),
                 'Every panel needs backing and printed thread.')
        pieces.append(p)
    return dict(grid=grid, background=background, backing=backing, marker=marker,
                side=side, pieces=pieces, thread_count=thread_count, data_colors=data_colors)


def _trace(board, backing, marker):
    """Return an ordered simple thread, or None. Color changes are not edges."""
    cells = set(map(tuple, np.argwhere(board != backing)))
    if not cells or int(board[0, 0]) != marker:
        return None
    adjacency = {p: [(p[0]+dr, p[1]+dc) for dr, dc in _EDGE
                     if (p[0]+dr, p[1]+dc) in cells] for p in cells}
    if (len(adjacency[(0, 0)]) != 1 or
            sum(len(v) == 1 for v in adjacency.values()) != 2 or
            any(len(v) not in (1, 2) for v in adjacency.values())):
        return None
    ordered, visited, previous, current = [], set(), None, (0, 0)
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        choices = [q for q in adjacency[current] if q != previous]
        if not choices:
            return ordered if len(ordered) == len(cells) else None
        previous, current = current, choices[0]
    return None


def _placements(pieces, side, marker, backing):
    options = []
    for piece in pieces:
        anchored = bool(np.any(piece == marker))
        choices, seen = [], set()
        for k in range(4):
            oriented = np.rot90(piece, k)
            key = oriented.shape, oriented.tobytes()
            if key in seen:
                continue
            seen.add(key)
            h, w = oriented.shape
            if h > side or w > side or (anchored and int(oriented[0, 0]) != marker):
                continue
            coords = np.argwhere(oriented >= 0)
            values = oriented[oriented >= 0].copy()
            for r in range(1 if anchored else side-h+1):
                for c in range(1 if anchored else side-w+1):
                    ids = ((coords[:, 0]+r)*side+coords[:, 1]+c).astype(int)
                    mask = sum(1 << int(x) for x in ids)
                    thread = sum(1 << int(x) for x, v in zip(ids, values) if v != backing)
                    choices.append((mask, ids, values, thread))
        options.append(choices)
    return options


def _assemblies(pieces, side, marker, backing, *, joint=True, limit=2):
    """Exact cover, retaining complete prints; optionally enforce thread topology.

    Exhaustive within the finite input bounds; stop only after `limit` distinct
    colored squares. Identical rotations and identical boards are harmless.
    """
    options = _placements(pieces, side, marker, backing)
    full = (1 << (side*side)) - 1
    board = np.full(side*side, backing, dtype=np.int8)
    owners = np.full(side*side, -1, dtype=np.int8)
    answers, signatures = [], set()
    neighbor_bits = []
    for r in range(side):
        for c in range(side):
            neighbor_bits.append(sum(1 << (rr*side+cc) for dr, dc in _EDGE
                                     if 0 <= (rr := r+dr) < side and 0 <= (cc := c+dc) < side))

    def search(occupied, remaining, thread):
        if len(answers) >= limit:
            return
        if not remaining:
            square = board.reshape(side, side)
            if occupied != full or (joint and _trace(square, backing, marker) is None):
                return
            key = board.tobytes()
            if key not in signatures:
                signatures.add(key)
                answers.append((square.copy(), owners.reshape(side, side).copy()))
            return
        first = (full ^ occupied) & -(full ^ occupied)
        for i in remaining:
            for mask, ids, values, printed in options[i]:
                if not mask & first or mask & occupied:
                    continue
                combined = thread | printed
                if joint:
                    # Once degree is too high, later panels cannot repair it.
                    remaining_bits, bad = combined, False
                    while remaining_bits:
                        bit = remaining_bits & -remaining_bits
                        j = bit.bit_length()-1
                        remaining_bits ^= bit
                        if (combined & neighbor_bits[j]).bit_count() > (1 if j == 0 else 2):
                            bad = True
                            break
                    if bad:
                        continue
                board[ids], owners[ids] = values, i
                search(occupied | mask, tuple(j for j in remaining if j != i), combined)
                board[ids], owners[ids] = backing, -1
                if len(answers) >= limit:
                    return
    search(0, tuple(range(len(pieces))), 0)
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _read_input(input_grid)
    answers = _assemblies(facts['pieces'], facts['side'], facts['marker'], facts['backing'])
    _require(len(answers) == 1, 'Thread-conditioned colored assembly must be unique.')
    board = answers[0][0]
    path = _trace(board, facts['backing'], facts['marker'])
    return np.asarray([board[q] for q in path], dtype=np.int8).reshape(-1, 1)


def _walk(rng, side, length):
    for _ in range(100):
        path, occupied, previous = [(0, 0)], {(0, 0)}, None
        persistence = float(rng.uniform(.5, 4.0))
        for _ in range(length-1):
            options, weights = [], []
            for direction in _EDGE:
                q = path[-1][0]+direction[0], path[-1][1]+direction[1]
                if not (0 <= q[0] < side and 0 <= q[1] < side) or q in occupied:
                    continue
                if sum((q[0]+dr, q[1]+dc) in occupied for dr, dc in _EDGE) != 1:
                    continue
                options.append((q, direction))
                weights.append(persistence if direction == previous else 1.0)
            if not options:
                break
            weights = np.asarray(weights, dtype=float)
            q, previous = options[int(rng.choice(len(options), p=weights/weights.sum()))]
            path.append(q)
            occupied.add(q)
        if len(path) == length and path[-1][0] > 0:
            return path
    return None


def _partition(rng, side, count):
    """Recursive cuts plus optional connected local transfers, never a slot bank."""
    rectangles = [(0, 0, side, side)]
    for _ in range(count-1):
        options = []
        for i, (r, c, h, w) in enumerate(rectangles):
            options.extend((i, 0, a) for a in range(1, h) if min(a, h-a)*w >= 5)
            options.extend((i, 1, a) for a in range(1, w) if min(a, w-a)*h >= 5)
        if not options:
            return None
        i, axis, a = options[int(rng.integers(len(options)))]
        r, c, h, w = rectangles.pop(i)
        rectangles.extend([(r, c, a, w), (r+a, c, h-a, w)] if axis == 0 else
                          [(r, c, h, a), (r, c+a, h, w-a)])
    labels = np.empty((side, side), dtype=np.int8)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r+h, c:c+w] = i
    for _ in range(int(rng.choice((0, 0, 1, 2, 4, 7)))):
        r, c = map(int, rng.integers(side, size=2))
        dr, dc = _EDGE[int(rng.integers(4))]
        rr, cc = r+dr, c+dc
        if (r, c) == (0, 0) or not (0 <= rr < side and 0 <= cc < side):
            continue
        a, b = int(labels[r, c]), int(labels[rr, cc])
        if a == b or np.sum(labels == a) <= 5:
            continue
        labels[r, c] = b
        if not _connected(labels == a):
            labels[r, c] = a
    return labels


def _word(owners):
    return [int(x) for i, x in enumerate(owners) if i == 0 or x != owners[i-1]]


def _sequence(rng, length, colors, marker):
    # Resample only a tiny in-memory list. Runs vary independently of bends/seams.
    count = min(len(colors), (length-1)//2)
    colors = colors[:count]
    for _ in range(40):
        seq = [marker]
        while len(seq) < length:
            choices = [c for c in colors if c != seq[-1]]
            color = int(rng.choice(choices))
            seq.extend([color] * min(length-len(seq), int(rng.integers(1, 5))))
        if all(seq.count(c) >= 2 for c in colors) and len(_word(seq[1:])) > len(colors):
            return seq
    # Cyclic fallback has repeated non-adjacent occurrences and no singleton data.
    return [marker] + [colors[i % count] for i in range(length-1)]


def _scan_sequences(grid, backing, background=None, marker=None):
    sequences = []
    for a in (grid, grid.T, grid[::-1], grid.T[::-1]):
        vals = [int(x) for x in a.flat if x != backing and (background is None or x != background)]
        if marker is not None:
            vals = [marker] + [x for x in vals if x != marker]
        sequences.append(vals)
    return sequences


def _scatter(rng, pieces, background, anchor_rotation, marker):
    oriented = [np.rot90(p, anchor_rotation if np.any(p == marker) else int(rng.integers(4))).copy()
                for p in pieces]
    for _ in range(8):
        canvas = np.full((30, 30), -1, dtype=np.int8)
        blocked = np.zeros((30, 30), dtype=bool)
        maxr = maxc = 0
        # Real masks, staggered positions, and independently rotated tight boxes.
        order = sorted(range(len(pieces)), key=lambda i: -oriented[i].size)
        aspect = float(rng.uniform(.65, 1.5))
        for i in order:
            p = oriented[i]
            h, w = p.shape
            rr, cc = np.nonzero(p >= 0)
            candidates = []
            for r in range(31-h):
                for c in range(31-w):
                    if np.any(blocked[r+rr, c+cc]):
                        continue
                    nr, nc = max(maxr, r+h), max(maxc, c+w)
                    score = nr*nc + abs(nr-aspect*nc)*2 + float(rng.random())*35
                    candidates.append((score, r, c))
            if not candidates:
                break
            _, r, c = min(candidates)
            canvas[r+rr, c+cc] = p[rr, cc]
            for dr, dc in ((0, 0),) + _EIGHT:
                rows, cols = r+rr+dr, c+cc+dc
                ok = (rows >= 0) & (rows < 30) & (cols >= 0) & (cols < 30)
                blocked[rows[ok], cols[ok]] = True
            maxr, maxc = max(maxr, r+h), max(maxc, c+w)
        else:
            coords = np.argwhere(canvas >= 0)
            lo, hi = coords.min(0), coords.max(0)+1
            cropped = canvas[lo[0]:hi[0], lo[1]:hi[1]]
            h, w = cropped.shape
            H = h + int(rng.integers(min(4, 30-h)+1))
            W = w + int(rng.integers(min(4, 30-w)+1))
            area = int(np.sum(cropped >= 0))
            while H*W <= 2*area:
                if H < 30 and (H <= W or W == 30):
                    H += 1
                elif W < 30:
                    W += 1
                else:
                    break
            result = np.full((H, W), background, dtype=np.int8)
            top, left = int(rng.integers(H-h+1)), int(rng.integers(W-w+1))
            result[top:top+h, left:left+w] = np.where(cropped >= 0, cropped, background)
            return result
    raise RuntimeError('Could not scatter panels within 30x30.')


def _fallback_geometry():
    # One documented safety net, not the ordinary sampling distribution.
    labels = np.array([[int(x) for x in row] for row in
                       ('1111110', '1111110', '1111110', '1111110',
                        '2222220', '3333330', '3333330')], dtype=np.int8)
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3),
            (3, 2), (3, 1), (3, 0), (4, 0), (5, 0), (5, 1), (6, 1),
            (6, 2), (6, 3), (6, 4), (5, 4), (4, 4), (4, 5), (4, 6), (5, 6)]
    return labels, path


def _make_pair(rng, side, count, marker, rotation, seam_witness=False):
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != marker])]
    background, backing = palette[:2]
    colors = palette[2:2+int(rng.integers(2, 6))]
    for attempt in range(241):
        if attempt == 240:
            labels, path = _fallback_geometry()
            side, count = 7, 4
            # Three cyclic data colors are known to distinguish this fixture.
            seq = [marker] + [palette[2+i % 3] for i in range(len(path)-1)]
        else:
            high = min(30, (side*side-1)//2)
            length = int(rng.integers(max(8, side+3), high+1))
            path = _walk(rng, side, length)
            labels = _partition(rng, side, count)
            if path is None or labels is None:
                continue
            owners = [int(labels[q]) for q in path]
            word = _word(owners)
            if len(set(owners)) != count or len(word) <= len(set(word)):
                continue
            seq = _sequence(rng, len(path), colors, marker)
        if seam_witness and not any(labels[a] != labels[b] and x != y
                                    for a, b, x, y in zip(path, path[1:], seq, seq[1:])):
            continue
        board = np.full((side, side), backing, dtype=np.int8)
        for q, color in zip(path, seq):
            board[q] = color
        pieces = _extract(board, labels, count)
        if any(np.sum(p == backing) < 2 for p in pieces):
            continue
        marked = next(p for p in pieces if np.any(p == marker))
        if min(marked.shape) < 2:
            continue
        joint = _assemblies(pieces, side, marker, backing)
        if len(joint) != 1 or not np.array_equal(joint[0][0], board):
            continue
        geometric = _assemblies(pieces, side, marker, backing, joint=False)
        if len(geometric) < 2 or seq in _scan_sequences(board, backing):
            continue
        inp = _scatter(rng, pieces, background, rotation, marker)
        if seq in _scan_sequences(inp, backing, background, marker):
            # Geometric scattering can accidentally match a scan; reshuffle
            # whole panels, never change the intended rule to accommodate it.
            for _ in range(5):
                inp = _scatter(rng, pieces, background, rotation, marker)
                if seq not in _scan_sequences(inp, backing, background, marker):
                    break
            else:
                continue
        return {'input': inp, 'output': np.asarray(seq, dtype=np.int8).reshape(-1, 1)}
    raise RuntimeError('Bounded construction and safety net failed.')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer.')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}.')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer.')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test.')
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    held = int(rng.integers(4))
    train_rotations = [int(k) for k in rng.permutation([k for k in range(4) if k != held])]
    larger = bool(rng.random() < .5)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            rotation = train_rotations[i % 3] if split == 'train' else held
            side = int(rng.integers(5, 8)) if larger and split == 'train' else int(rng.integers(6, 10))
            if larger and split == 'test':
                side = int(rng.integers(8, 10))
            count = int(rng.integers(3, 5 if side == 5 else 6))
            for _ in range(8):
                pair = _make_pair(rng, side, count, marker, rotation,
                                  seam_witness=split == 'train' and i == 0)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not create distinct inputs.')
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _pair_facts(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Pair keys must be input/output.')
    inp, out = _grid(pair['input']), _grid(pair['output'])
    f = _read_input(inp)
    answers = _assemblies(f['pieces'], f['side'], f['marker'], f['backing'])
    _require(len(answers) == 1, 'There must be exactly one thread-conditioned colored square.')
    board, owners = answers[0]
    path = _trace(board, f['backing'], f['marker'])
    sequence = [int(board[q]) for q in path]
    expected = np.asarray(sequence, dtype=np.int8).reshape(-1, 1)
    _require(np.array_equal(out, expected), 'Output must preserve every thread cell in order.')
    _require(out.shape == (f['thread_count'], 1), 'Incorrect strip dimensions.')
    _require(len(_assemblies(f['pieces'], f['side'], f['marker'], f['backing'], joint=False)) == 2,
             'Shape-only ambiguity must make thread continuity necessary.')
    ownership_word = _word([owners[q] for q in path])
    _require(len(set(ownership_word)) == len(f['pieces']), 'Every panel must contribute to the thread.')
    _require(len(ownership_word) > len(set(ownership_word)), 'The path must leave and re-enter a panel.')
    _require(len(_word(sequence[1:])) > len(f['data_colors']), 'A data color must recur in separate runs.')
    _require(sequence not in _scan_sequences(board, f['backing']), 'Assembled-board scans must fail.')
    _require(sequence not in _scan_sequences(inp, f['backing'], f['background'], f['marker']),
             'Scattered-input scans must fail even after moving the marker to the front.')
    for c in f['data_colors'] | {f['marker']}:
        _require(np.sum(inp == c) == np.sum(out == c), 'Every thread color count must be preserved.')
    _require(not np.any(out == f['backing']) and not np.any(out == f['background']),
             'Discard supporting material and background.')
    marked = next(p for p in f['pieces'] if np.any(p == f['marker']))
    corners = []
    for k in range(4):
        p = np.rot90(marked, -k)
        if int(p[0, 0]) == f['marker']:
            corners.append(k)
    steps = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    runs = _word(sequence[1:])
    seam_changes = sum(owners[a] != owners[b] and board[a] != board[b]
                       for a, b in zip(path, path[1:]))
    f.update(board=board, path=path, owners=owners, ownership_word=ownership_word,
             turns=sum(a != b for a, b in zip(steps, steps[1:])),
             upward=(-1, 0) in steps, reversal=(0, 1) in steps and (0, -1) in steps,
             runs=len(runs), marker_rotations=set(corners),
             seam_changes=int(seam_changes),
             irregular=sum(not np.all(p >= 0) for p in f['pieces']),
             areas=tuple(sorted(int(np.sum(p >= 0)) for p in f['pieces'])))
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Puzzle keys must be train/test.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Illegal counts.')
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            f = _pair_facts(pair)
            key = _key(f['grid'])
            _require(key not in seen, 'Duplicate input.')
            seen.add(key)
            facts[split].append(f)
    _require(len({f['marker'] for f in facts['train']+facts['test']}) == 1, 'Marker is shared within a puzzle.')
    _require(any(f['seam_changes'] > 0 for f in facts['train']),
             'Training must show a color change across a stitched seam.')
    # At least two demonstrations show different rigid orientations. A marked
    # one-dimensional panel can admit two anchor rotations; use set inclusion.
    train_rotations = set().union(*(f['marker_rotations'] for f in facts['train']))
    _require(len(train_rotations) >= 2, 'Training must establish more than one marker orientation.')
    for f in facts['test']:
        _require(bool(f['marker_rotations'] - train_rotations), 'Test must include a held-out marker orientation.')


def _puzzle_digest(puzzle):
    h = sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                h.update(bytes(shape))
                h.update(data)
    return h.hexdigest()


def self_test():
    """Bounded 42-seed semantic/diversity checks; returns a small metric dict."""
    shapes, geometries, signatures, area_profiles = set(), set(), set(), set()
    sides, counts, lengths, turns, backings, backgrounds = set(), set(), set(), set(), set(), set()
    irregular = upward = reversals = reentries = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _pair_facts(pair)
                shapes.add(pair['input'].shape)
                # Binary thread geometry, independent of all color literals.
                geometries.add((f['board'].shape, (f['board'] != f['backing']).tobytes()))
                sides.add(f['side'])
                counts.add(len(f['pieces']))
                lengths.add(f['thread_count'])
                turns.add(f['turns'])
                area_profiles.add(f['areas'])
                backings.add(f['backing'])
                backgrounds.add(f['background'])
                irregular += f['irregular'] > 0
                upward += f['upward']
                reversals += f['reversal']
                reentries += len(f['ownership_word']) - len(f['pieces'])
    _require(len(signatures) == 42 and len(geometries) >= 100, 'Insufficient structural variation.')
    _require(sides == set(range(5, 10)) and counts == {3, 4, 5}, 'Missing size/count coverage.')
    _require(len(shapes) >= 35 and len(area_profiles) >= 50, 'Insufficient panel/layout variation.')
    _require(len(lengths) >= 18 and len(turns) >= 8, 'Insufficient thread variation.')
    _require(len(backings) == 10 and len(backgrounds) == 10, 'Missing palette role coverage.')
    _require(irregular >= 5 and upward >= 20 and reversals >= 20, 'Missing nontrivial geometries.')
    _require(_puzzle_digest(generate_puzzle(0)) == _puzzle_digest(generate_puzzle(0)), 'Not deterministic.')
    # The host enumerates all 30 legal combinations. Local boundary checks
    # retain only one puzzle at a time and keep this self-test short.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(551109681, a, b)
        validate_generated_puzzle(p)
    # Exercise the safety net even when the ordinary seed suite never needs it.
    labels, path = _fallback_geometry()
    board = np.ones(labels.shape, dtype=np.int8)
    sequence = [2] + [3+i % 3 for i in range(len(path)-1)]
    for q, color in zip(path, sequence):
        board[q] = color
    pieces = _extract(board, labels, 4)
    for rotation in range(4):
        inp = _scatter(np.random.default_rng(900+rotation), pieces, 0, rotation, 2)
        _require(np.array_equal(solve(inp), np.asarray(sequence).reshape(-1, 1)),
                 'Safety-net rotation regression failed.')
    for cells in (((0, 0), (0, 1), (1, 0)),
                  ((0, 0), (0, 1), (1, 1), (1, 0)),
                  ((0, 0), (1, 0), (3, 3), (4, 3))):
        bad_thread = np.ones((7, 7), dtype=np.int8)
        for q in cells:
            bad_thread[q] = 3
        bad_thread[0, 0] = 2
        _require(_trace(bad_thread, 1, 2) is None, 'Invalid thread topology accepted.')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][1, 0] = bad['test'][0]['output'][0, 0]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted color order accepted.')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                binary_thread_geometries=len(geometries), panel_area_profiles=len(area_profiles),
                sides=sorted(sides), panel_counts=sorted(counts), lengths=sorted(lengths),
                turn_counts=sorted(turns), irregular_pairs=irregular, upward_pairs=upward,
                horizontal_reversal_pairs=reversals, extra_panel_visits=reentries)
