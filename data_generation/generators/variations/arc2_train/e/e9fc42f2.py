import hashlib

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SOURCE_INPUT_HASHES = {
    'e40db95b5e80f4695318ac613bdce10b04466ad75357440d2f60bb5cc550b62f',
    '58702b0338fc8454286232934b48296e7ca404fc178ccc8c9d7f8511c004b6ec',
    '626717fe69b7073022771601feed84bcd7ff77ba2df9bdaee64b3e4a24c27484',
    '6514093b714a58dfb0592f5ff9801f547ad44aff6f1072c784c49d71262949d9',
    '28ede11a4a24c963657da1cf03488ca43ffa55db617fb093d6d0a75cad849d77',
}


def _valid_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(rng, nt, ns):
    if nt is not None and (not _valid_int(nt) or not 2 <= nt <= 10):
        raise ValueError('num_train must be 2 through 10')
    if ns is not None and (not _valid_int(ns) or not 1 <= ns <= 4):
        raise ValueError('num_test must be 1 through 4')
    if ns is None and nt is None:
        ns = int(rng.integers(1, 5))
    if nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    if ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    return nt, ns


def _connected(mask):
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set(stack)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            q = (r + dr, c + dc)
            if q in cells and q not in seen:
                seen.add(q)
                stack.append(q)
    return seen == cells


def _sides(mask, r, c):
    h, w = mask.shape
    free = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
            free.append((dr, dc))
    assert 1 <= len(free) <= 2
    return tuple(free)


def _directions(rng, n, bend):
    d = DIRS[int(rng.integers(4))]
    t = DIRS[(DIRS.index(d) + (1 if rng.integers(2) else -1)) % 4]
    if n == 2:
        return [d]
    if n == 3:
        return [d, t] if bend or rng.random() < 0.75 else [d, d]
    mode = int(rng.integers(5))
    if mode == 0:
        return [d] * (n - 1)
    if mode == 1:
        pivot = int(rng.integers(1, n - 1))
        return [d] * pivot + [t] * (n - 1 - pivot)
    if mode == 2:
        return [d if i % 2 == 0 else t for i in range(n - 1)]
    if mode == 3:
        return [d, t, (-d[0], -d[1])] + [t] * (n - 4)
    return [d, t, d] + [t] * (n - 4)


def _piece(rng, ports, body, bg, irregular):
    profile = int(rng.integers(4))
    if profile == 0:
        h, w = int(rng.integers(3, 5)), int(rng.integers(6, 9))
    elif profile == 1:
        h, w = int(rng.integers(6, 9)), int(rng.integers(3, 5))
    else:
        h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    positions = {}
    occupied = set()
    for color, side, corner in ports:
        if side[0]:
            r = 0 if side[0] < 0 else h - 1
            columns = (0, w - 1) if corner else range(1, w - 1)
            choices = [(r, c) for c in columns if (r, c) not in occupied]
        else:
            c = 0 if side[1] < 0 else w - 1
            rows = (0, h - 1) if corner else range(1, h - 1)
            choices = [(r, c) for r in rows if (r, c) not in occupied]
        if not choices:
            return None
        q = choices[int(rng.integers(len(choices)))]
        positions[color] = q
        occupied.add(q)
    mask = np.ones((h, w), dtype=bool)
    protected = set(occupied)
    for r, c in occupied:
        for dr, dc in DIRS:
            q = (r + dr, c + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w:
                protected.add(q)
    candidates = [(r, c) for r in range(h) for c in range(w) if (r, c) not in protected]
    rng.shuffle(candidates)
    target = int(round(len(candidates) * (rng.uniform(0.48, 0.82) if irregular else rng.uniform(0, 0.12))))
    removed = 0
    for r, c in candidates:
        if removed >= target:
            break
        if 0 < r < h - 1 and 0 < c < w - 1 and rng.random() > 0.5:
            continue
        mask[r, c] = False
        if _connected(mask):
            removed += 1
        else:
            mask[r, c] = True
    if irregular and removed == 0:
        for r, c in candidates:
            mask[r, c] = False
            if _connected(mask):
                break
            mask[r, c] = True
    piece = np.full((h, w), bg, dtype=np.int8)
    piece[mask] = body
    for color, (r, c) in positions.items():
        assert next(side for marker, side, _ in ports if marker == color) in _sides(mask, r, c)
        piece[r, c] = color
    return piece, positions


def _assemble(pieces, offsets, bg, matches, terminals):
    cells, owners = {}, {}
    for i, (piece, (r0, c0)) in enumerate(zip(pieces, offsets)):
        for r, c in zip(*np.nonzero(piece != bg)):
            q = (r0 + int(r), c0 + int(c))
            if q in cells:
                return None
            cells[q], owners[q] = int(piece[r, c]), i
    allowed = {frozenset(pair) for pair in matches.values()}
    for (r, c), owner in owners.items():
        for dr, dc in ((1, 0), (0, 1)):
            q = (r + dr, c + dc)
            if q in owners and owners[q] != owner and frozenset((owner, owners[q])) not in allowed:
                return None
    for color, pair in matches.items():
        spots = [q for q, value in cells.items() if value == color]
        if len(spots) != 2 or {owners[q] for q in spots} != set(pair):
            return None
        if abs(spots[0][0] - spots[1][0]) + abs(spots[0][1] - spots[1][1]) != 1:
            return None
    for color in terminals:
        spots = [q for q, value in cells.items() if value == color]
        if len(spots) != 1:
            return None
        r, c = spots[0]
        if all((r + dr, c + dc) in cells for dr, dc in DIRS):
            return None
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    if r1 - r0 >= 29 or c1 - c0 >= 29:
        return None
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=np.int8)
    for (r, c), value in cells.items():
        out[r - r0, c - c0] = value
    return out if np.any(out == bg) else None


def _candidate(rng, n, directions, terminal_count, params, irregular, corner_first):
    bg, body, palette = params
    colors = [int(x) for x in rng.choice(np.asarray(palette), size=n - 1 + terminal_count, replace=False)]
    ports = [[] for _ in range(n)]
    matches = {}
    for i, side in enumerate(directions):
        color = colors[i]
        corner_end = int(rng.integers(2)) if (i == 0 and corner_first) or (i > 0 and rng.random() < 0.2) else -1
        ports[i].append((color, side, corner_end == 0))
        ports[i + 1].append((color, (-side[0], -side[1]), corner_end == 1))
        matches[color] = (i, i + 1)
    terminals = colors[n - 1:]
    for j, color in enumerate(terminals):
        i = 0 if j == 0 else n - 1
        options = [d for d in DIRS if d not in {side for _, side, _ in ports[i]}]
        ports[i].append((color, options[int(rng.integers(len(options)))], False))
    made = [_piece(rng, p, body, bg, irregular) for p in ports]
    if any(x is None for x in made):
        return None
    pieces, positions = [x[0] for x in made], [x[1] for x in made]
    offsets = [(0, 0)]
    for i, side in enumerate(directions):
        a, b = positions[i][colors[i]], positions[i + 1][colors[i]]
        old = offsets[-1]
        offsets.append((old[0] + a[0] + side[0] - b[0], old[1] + a[1] + side[1] - b[1]))
    out = _assemble(pieces, offsets, bg, matches, terminals)
    return (pieces, out) if out is not None else None


def _scatter(rng, pieces, output_shape, bg, compact, edge):
    area = sum(p.size for p in pieces) / (rng.uniform(0.39, 0.57) if compact else rng.uniform(0.18, 0.34))
    aspect = rng.uniform(0.65, 1.55)
    h = max(int(round(np.sqrt(area * aspect))), output_shape[0] + 1, max(p.shape[0] for p in pieces) + 2)
    w = max(int(round(np.sqrt(area / aspect))), output_shape[1] + 1, max(p.shape[1] for p in pieces) + 2)
    if compact and len(pieces) <= 3 and max(output_shape) < 13 and rng.random() < 0.25:
        h = w = 13
    if h > 30 or w > 30:
        return None
    occupied, placements = set(), []
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    for i, piece in enumerate(pieces):
        ph, pw = piece.shape
        places = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        if i == 0 and edge:
            places = [(r, c) for r, c in places if r == 0 or c == 0 or r + ph == h or c + pw == w]
        rng.shuffle(places)
        local = list(zip(*np.nonzero(piece != bg)))
        found = False
        for r, c in places:
            cells = {(r + int(dr), c + int(dc)) for dr, dc in local}
            if cells & corners or any((x + dr, y + dc) in occupied for x, y in cells for dr, dc in ((0, 0),) + DIRS):
                continue
            occupied.update(cells)
            placements.append((r, c))
            found = True
            break
        if not found:
            return None
    grid = np.full((h, w), bg, dtype=np.int8)
    for piece, (r, c) in zip(pieces, placements):
        view = grid[r:r + piece.shape[0], c:c + piece.shape[1]]
        mask = piece != bg
        view[mask] = piece[mask]
    return grid


def _pair(rng, n, terminals, bend, params, irregular, compact, edge, corner):
    for attempt in range(450):
        candidate = _candidate(rng, n, _directions(rng, n, bend), terminals, params, irregular, corner)
        if candidate is None:
            continue
        pieces, out = candidate
        inp = _scatter(rng, pieces, out.shape, params[0], compact if attempt < 240 else False, edge if attempt < 300 else False)
        if inp is not None:
            return inp, out
    raise ValueError('could not construct a valid pair')


def _components(grid, bg):
    mask = grid != bg
    seen = np.zeros(grid.shape, dtype=bool)
    pieces = []
    for start in zip(*np.nonzero(mask)):
        if seen[start]:
            continue
        seen[start] = True
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        local = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=np.int8)
        for r, c in cells:
            local[r - r0, c - c0] = grid[r, c]
        pieces.append(local)
    return pieces


def _solve(inp, bg, body):
    pieces = _components(inp, bg)
    assert 2 <= len(pieces) <= 5
    occurrences = {}
    for i, piece in enumerate(pieces):
        mask = piece != bg
        assert _connected(mask)
        for r, c in zip(*np.nonzero(mask & (piece != body))):
            color = int(piece[r, c])
            sides = _sides(mask, int(r), int(c))
            occurrences.setdefault(color, []).append((i, (int(r), int(c)), sides))
    matches, terminals = {}, []
    graph = [[] for _ in pieces]
    for color, items in occurrences.items():
        assert len(items) in (1, 2)
        if len(items) == 1:
            terminals.append(color)
            continue
        (a, pa, sides_a), (b, pb, sides_b) = items
        assert a != b
        choices = [side for side in sides_a if (-side[0], -side[1]) in sides_b]
        assert len(choices) == 1
        sa = choices[0]
        sb = (-sa[0], -sa[1])
        delta = (pa[0] + sa[0] - pb[0], pa[1] + sa[1] - pb[1])
        graph[a].append((b, delta, sa))
        graph[b].append((a, (-delta[0], -delta[1]), sb))
        matches[color] = (a, b)
    assert len(matches) == len(pieces) - 1 and len(terminals) <= 2
    assert max(len(edges) for edges in graph) <= 2
    offsets = [None] * len(pieces)
    offsets[0] = (0, 0)
    stack = [0]
    while stack:
        i = stack.pop()
        for j, delta, _ in graph[i]:
            q = (offsets[i][0] + delta[0], offsets[i][1] + delta[1])
            if offsets[j] is None:
                offsets[j] = q
                stack.append(j)
            else:
                assert offsets[j] == q
    assert all(x is not None for x in offsets)
    out = _assemble(pieces, offsets, bg, matches, terminals)
    assert out is not None
    return out, pieces, len(terminals), graph


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    train, test = puzzle['train'], puzzle['test']
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, conventions, train_counts, terminal_counts, axes = set(), set(), set(), set(), set()
    for split, pairs in (('train', train), ('test', test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert _key(inp) not in seen
            seen.add(_key(inp))
            if inp.shape == (13, 13):
                digest = hashlib.sha256(np.asarray(inp, dtype=np.int8).tobytes()).hexdigest()
                assert digest not in SOURCE_INPUT_HASHES
            assert inp.shape != out.shape or not np.array_equal(inp, out)
            assert all(int(inp[r, c]) == int(inp[0, 0]) for r, c in ((0, 0), (0, inp.shape[1] - 1), (inp.shape[0] - 1, 0), (inp.shape[0] - 1, inp.shape[1] - 1)))
            bg = int(inp[0, 0])
            values = inp[inp != bg]
            assert values.size > 0
            counts = np.bincount(values.astype(np.int64), minlength=10)
            body = int(np.argmax(counts))
            assert np.count_nonzero(counts == counts[body]) == 1 and body != bg
            conventions.add((bg, body))
            expected, pieces, terminals, graph = _solve(inp, bg, body)
            assert np.array_equal(out, expected)
            assert out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1]
            terminal_counts.add(terminals)
            if split == 'train':
                train_counts.add(len(pieces))
                for edges in graph:
                    for _, _, side in edges:
                        axes.add(0 if side[0] else 1)
    assert len(conventions) == 1 and {2, 3}.issubset(train_counts)
    assert {0, 1}.issubset(terminal_counts) and axes == {0, 1}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _valid_int(seed):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.permutation(10)]
    params = (colors[0], colors[1], tuple(colors[2:]))
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_train + num_test, dtype=np.uint64)
    puzzle, seen, k = {'train': [], 'test': []}, set(), 0
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            prng = np.random.default_rng(seeds[k])
            k += 1
            if split == 'train':
                n = 2 if i == 0 else 3 if i == 1 else int(prng.choice((2, 3, 4), p=(0.25, 0.4, 0.35)))
                terminals = 1 if i == 0 else 0 if i == 1 else int(prng.integers(2))
            else:
                n = int(prng.choice((2, 3, 4, 5), p=(0.1, 0.35, 0.4, 0.15)))
                terminals = int(prng.integers(2))
            irregular = False if split == 'train' and i == 0 else True if split == 'train' and i == 1 else prng.random() < 0.76
            compact, edge = prng.random() < 0.55, prng.random() < 0.55
            corner = (split == 'train' and i == 1) or (split != 'train' or i > 1) and prng.random() < 0.35
            for _ in range(12):
                inp, out = _pair(prng, n, terminals, split == 'train' and i == 1, params, irregular, compact, edge, corner)
                key = _key(inp)
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append({'input': inp, 'output': out})
                    break
            else:
                raise ValueError('could not make unique inputs')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
        assert (len(p['train']), len(p['test'])) == (nt, ns)
    for nt, ns in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(2000 + (nt or 0) + (ns or 0), nt, ns))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(1, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    a, b = generate_puzzle(98765, 5, 2), generate_puzzle(98765, 5, 2)
    for split in ('train', 'test'):
        for x, y in zip(a[split], b[split]):
            assert np.array_equal(x['input'], y['input']) and np.array_equal(x['output'], y['output'])
    assert _key(generate_puzzle(100, 5, 2)['train'][0]['input']) != _key(generate_puzzle(101, 5, 2)['train'][0]['input'])
    roles, shapes, layouts, piece_counts, terminal_counts = set(), set(), set(), set(), set()
    train_piece_counts, test_piece_counts = set(), set()
    irregular = compact = edge = multi_turn = 0
    corner_ports = 0
    for puzzle in puzzles:
        for split, pairs in (('train', puzzle['train']), ('test', puzzle['test'])):
            for pair in pairs:
                inp = pair['input']
                bg = int(inp[0, 0])
                body = int(np.argmax(np.bincount(inp[inp != bg].astype(np.int64), minlength=10)))
                roles.add((bg, body))
                _, pieces, terminals, graph = _solve(inp, bg, body)
                piece_counts.add(len(pieces))
                (train_piece_counts if split == 'train' else test_piece_counts).add(len(pieces))
                terminal_counts.add(terminals)
                shapes.add((inp.shape, pair['output'].shape))
                irregular += any(np.any(piece == bg) for piece in pieces)
                corner_ports += sum(
                    len(_sides(piece != bg, int(r), int(c))) == 2
                    for piece in pieces
                    for r, c in zip(*np.nonzero((piece != bg) & (piece != body)))
                )
                compact += inp.shape[0] <= 16 and inp.shape[1] <= 16
                edge += bool(
                    np.any(inp[0] != bg) or np.any(inp[-1] != bg)
                    or np.any(inp[:, 0] != bg) or np.any(inp[:, -1] != bg)
                )
                if len(pieces) >= 4:
                    multi_turn += sum(
                        len(edges) == 2 and edges[0][2][0] != edges[1][2][0]
                        for edges in graph
                    ) >= 2
                layouts.add((inp.shape, tuple(piece.shape for piece in pieces), pair['output'].shape))
    assert len(roles) >= 20 and len(shapes) >= 30 and len(layouts) >= 100
    assert {2, 3, 4, 5}.issubset(piece_counts) and {0, 1}.issubset(terminal_counts)
    assert {2, 3, 4}.issubset(train_piece_counts) and {2, 3, 4, 5}.issubset(test_piece_counts)
    assert irregular >= 80 and compact >= 10 and edge >= 30 and multi_turn >= 5 and corner_ports >= 30
