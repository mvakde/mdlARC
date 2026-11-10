"""Ink-addressed stencil routing. Only standard library and NumPy are used."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        found, stack = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    found.append(q)
                    stack.append(q)
        result.append(found)
    return result


def _box(cells):
    a = np.asarray(cells)
    return tuple(map(int, (a[:, 0].min(), a[:, 0].max()+1,
                           a[:, 1].min(), a[:, 1].max()+1)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parse(input_grid):
    """Recover the punctured frame, keys, stencils, tokens and colored ports."""
    g = _grid(input_grid)
    candidates = []
    for color in np.unique(g):
        cells = np.argwhere(g == color)
        r, e, c, d = _box(cells)
        if e-r < 7 or d-c < 7:
            continue
        ring = np.zeros((e-r, d-c), bool)
        ring[0] = ring[-1] = True
        ring[:, 0] = ring[:, -1] = True
        sub = g[r:e, c:d]
        missing = np.argwhere(ring & (sub != color))
        if (len(missing) in (1, 2) and len(cells) == int(ring.sum())-len(missing)
                and np.all((sub == color) <= ring)):
            corners = [sub[0, 0], sub[0, -1], sub[-1, 0], sub[-1, -1]]
            if all(v == color for v in corners):
                candidates.append((r, e, c, d, int(color)))
    _require(len(candidates) == 1, "need one exclusive rectangular frame with one or two ports")
    r, e, c, d, frame = candidates[0]
    board = g[r:e, c:d].copy()
    h, w = board.shape
    bg = int(board[1, 1])
    _require(bg != frame and all(int(board[a, b]) == bg
             for a, b in ((1, 1), (1, w-2), (h-2, 1), (h-2, w-2))),
             "inner corners must identify the background")
    ports = {}
    for side, points in (
            ('top', [(0, j) for j in range(1, w-1)]),
            ('bottom', [(h-1, j) for j in range(1, w-1)]),
            ('left', [(i, 0) for i in range(1, h-1)]),
            ('right', [(i, w-1) for i in range(1, h-1)])):
        for p in points:
            ink = int(board[p])
            if ink != frame:
                _require(ink != bg and ink not in ports, "ports need distinct nonbackground inks")
                ports[ink] = (side, p)
    _require(len(ports) in (1, 2), "one or two ports required")
    if len(ports) == 2:
        _require({v[0] for v in ports.values()} in
                 ({'top', 'bottom'}, {'left', 'right'}), "two ports must be opposite")
    places, sizes = [], set()
    inner = board[1:-1, 1:-1]
    for cells in _components(inner != bg):
        a, b, x, y = _box(cells)
        tile = inner[a:b, x:y]
        _require(b-a == y-x and 2 <= b-a <= 6 and len(cells) == (b-a)**2
                 and len(np.unique(tile)) == 1, "interior objects must be separated solid squares")
        places.append((a+1, x+1, int(tile[0, 0])))
        sizes.add(b-a)
    _require(len(sizes) == 1 and bool(places), "one common token size required")
    n = next(iter(sizes))
    sources = {s for _, _, s in places}
    _require(not (sources & {bg, frame}), "structural colors cannot be sources")
    exterior = g != bg
    exterior[r:e, c:d] = False
    mapping, key_boxes = {}, {}
    remaining = exterior.copy()
    for cells in _components(exterior):
        values = Counter(int(g[p]) for p in cells)
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        _require((b-a, y-x) == (2, 2) and sorted(values.values()) == [1, 3],
                 "keys must have three source cells and one target cell")
        source = next(k for k, v in values.items() if v == 3)
        target = next(k for k, v in values.items() if v == 1)
        _require(source in sources and source not in mapping and target in ports,
                 "each source needs a unique direct mapping to a visible port")
        mapping[source], key_boxes[source] = target, (a, b, x, y)
        remaining[a:b, x:y] = False
    _require(set(mapping) == sources, "missing or duplicate key")
    _require(set(map(int, np.unique(g[remaining]))) == sources, "unexpected exterior material")
    stencils, stencil_boxes = {}, {}
    for source in sorted(sources):
        a, b, x, y = _box(np.argwhere(remaining & (g == source)))
        _require((b-a, y-x) == (n, n), "stencil must expose all four square bounds")
        tile = g[a:b, x:y]
        _require(np.all(np.isin(tile, [bg, source])), "stencil contains foreign cells")
        stencils[source] = tile == source
        stencil_boxes[source] = (a, b, x, y)
    boxes = list(key_boxes.values()) + list(stencil_boxes.values())
    for i, (a, b, x, y) in enumerate(boxes):
        _require(b <= r or a >= e or y <= c or x >= d, "clue box intersects scene")
        for aa, bb, xx, yy in boxes[i+1:]:
            _require(b < aa or bb < a or y < xx or yy < x, "clue boxes need a blank gap")
    material = {ink: set() for ink in ports}
    for a, b, source in places:
        for u, v in np.argwhere(stencils[source]):
            material[mapping[source]].add((a+int(u), b+int(v)))
    return dict(grid=g, board=board, background=bg, frame=frame, frame_box=(r, e, c, d),
                n=n, places=places, mapping=mapping, stencils=stencils,
                stencil_boxes=stencil_boxes, key_boxes=key_boxes, ports=ports, material=material)


def _route(board, background, ports, material, *, nearest=False, separate=False):
    """One material/port operation, shared by construction and input-only solving.

    Optional switches are explicit incorrect rules used only by validation.
    """
    out = board.copy()
    out[1:-1, 1:-1] = background
    h, w = out.shape
    ink_order = sorted(ports)
    moving = []
    diagnostics = []
    for ink in ink_order:
        side, p = ports[ink]
        cells = material[ink]
        vertical = side in ('top', 'bottom')
        axis = p[1] if vertical else p[0]
        aligned = [q[0] if vertical else q[1] for q in cells
                   if (q[1] if vertical else q[0]) == axis]
        _require(bool(aligned), "each ink must meet its own guide")
        forward = side in ('top', 'left')
        stop = (min if forward else max)(aligned) if nearest else (max if forward else min)(aligned)
        start = 0 if forward else (h-1 if vertical else w-1)
        for t in range(min(start, stop), max(start, stop)+1):
            q = (t, axis) if vertical else (axis, t)
            _require(int(out[q]) in (background, ink), "guide collision")
            out[q] = ink
        # All owned axial material is fixed, even if a deliberately defective ray stops early.
        for q in cells:
            if (q[1] if vertical else q[0]) == axis:
                _require(int(out[q]) in (background, ink), "fixed material collision")
                out[q] = ink
        for low in (True, False):
            group = {q for q in cells if ((q[1] if vertical else q[0]) < axis if low
                                         else (q[1] if vertical else q[0]) > axis)}
            if not group:
                continue
            corner = ((side, 'left' if low else 'right') if vertical
                      else ('top' if low else 'bottom', side))
            fragments = _components_from_cells(group) if separate else [group]
            for fragment in fragments:
                a, b, x, y = _box(list(fragment))
                dr = 1-a if corner[0] == 'top' else h-1-b
                dc = 1-x if corner[1] == 'left' else w-1-y
                moving.append((ink, [(u+dr, v+dc) for u, v in fragment]))
            diagnostics.append((ink, corner, group))
    for ink, targets in moving:
        for q in targets:
            _require(0 < q[0] < h-1 and 0 < q[1] < w-1, "corner clipping")
            _require(int(out[q]) == background or separate, "corner/guide collision")
            out[q] = ink
    return out, diagnostics


def _components_from_cells(cells):
    a, b, x, y = _box(list(cells))
    mask = np.zeros((b-a, y-x), bool)
    for r, c in cells:
        mask[r-a, c-x] = True
    return [{(r+a, c+x) for r, c in group} for group in _components(mask)]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from this grid. No seed, examples or generated metadata."""
    a = _parse(input_grid)
    return _route(a['board'], a['background'], a['ports'], a['material'])[0]


def _mask(rng, n, kind):
    yy, xx = np.indices((n, n))
    if kind == 0:
        m = rng.random((n, n)) < rng.uniform(.22, .82)
    elif kind == 1:
        m = (xx == 0) | (yy == 0) | (xx == n-1) | (yy == n-1)
    elif kind == 2:
        m = (xx+yy) % 2 == 0
    elif kind == 3:
        m = (xx == yy) | (xx+yy == n-1)
    elif kind == 4:
        m = (yy % 2 == 0) | (xx == n//2)
    elif kind == 5:
        m = (xx == n//2) | (yy == n//2)
    else:
        m = np.ones((n, n), bool)
    for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return np.rot90(m, int(rng.integers(4))).copy()


def _spaced_starts(rng, count, size, length):
    slack = length - count*size - (count-1)
    _require(slack >= 0, "legend does not fit")
    gaps = rng.multinomial(slack, np.full(count+1, 1/(count+1)))
    starts, at = [], int(gaps[0])
    for i in range(count):
        starts.append(at)
        at += size+1+int(gaps[i+1])
    return starts


def _make_pair(rng, n, arity, count, categories, turn, witness=False, overlap=False):
    """Construct a compact field and two independently shuffled clue banks."""
    colors = list(map(int, rng.permutation(10)))
    bg, frame = colors[:2]
    sources = colors[2:2+categories]
    if arity == 1:
        inks = [colors[2+categories]]
    elif overlap:
        inks = [sources[1], sources[0]]
    else:
        inks = colors[2+categories:4+categories]
    mapping = {s: inks[int(rng.integers(arity))] for s in sources}
    if arity == 2:
        mapping[sources[0]], mapping[sources[1]] = inks
    # The two anchor tokens share a column but have independent, adjacent guide axes.
    if witness:
        rows, cols = 1, 2
    elif arity == 2:
        rows, cols = 2, (2 if count > 2 else int(rng.integers(1, 3)))
    else:
        options = [(2, 2), (count, 1)] + ([(1, 2)] if count == 2 else [])
        rows, cols = options[int(rng.integers(len(options)))]
    gap_r, gap_c = map(int, rng.integers(1, 4, size=2))
    if arity == 1 and rows > 1:
        gap_r = min(gap_r, max(1, (25-rows*n)//(rows-1)))
    fh, fw = rows*n+(rows-1)*gap_r, cols*n+(cols-1)*gap_c
    bh = (2*fh+4 if arity == 2 else min(30, fh+int(rng.integers(5, 10))))
    bw = fw+6+int(rng.integers(3))
    # Side-bank packing: width <= 30, height <= 30, at most 4 stencil entries.
    _require(bh <= 30 and bw+n+4 <= 30, "construction exceeds canvas")
    height = min(30, max(bh, categories*(n+1)-1)+int(rng.integers(3)))
    width = bw+n+4
    layout = int(rng.integers(3))
    if layout == 0:
        fc, sx, kx = 0, bw+1, bw+n+2
    elif layout == 1:
        fc, sx, kx = n+4, 0, n+1
    else:
        fc, sx, kx = n+1, 0, n+1+bw+1
    fr = int(rng.integers(height-bh+1))
    grid = np.full((height, width), bg, np.int8)
    board = np.full((bh, bw), bg, np.int8)
    board[0] = board[-1] = frame
    board[:, 0] = board[:, -1] = frame
    top = int(rng.integers(2, bh-fh-1))
    left = int(rng.integers(3, bw-fw-2))
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    if arity == 2:
        # Reversed spatial order is deliberate evidence against nearest-port routing.
        anchors = [(1, 0), (0, 0)] if witness is False else [(0, 0), (1, 0)]
        rest = [p for p in slots if p not in anchors]
        rng.shuffle(rest)
        slots = anchors+rest
    else:
        rng.shuffle(slots)
        if witness:
            slots = [(0, 0), (0, 1)]
    assignments = sources + list(map(int, rng.choice(sources, count-categories)))
    # Only categories 0 and 1 are anchors; all other assignments and positions vary.
    if arity == 1 and not witness:
        rng.shuffle(assignments)
    places = [(top+r*(n+gap_r), left+c*(n+gap_c), s)
              for (r, c), s in zip(slots[:count], assignments)]
    if not witness:
        # Move individual squares within their reserved gaps. The two guide anchors
        # retain a shared column; other row/column offsets need not form a lattice.
        places = [(a+(int(rng.integers(gap_r)) if (a-top)//(n+gap_r) < rows-1 else 0),
                   b+(int(rng.integers(gap_c)) if arity == 1 and (b-left)//(n+gap_c) < cols-1 else 0), s)
                  for a, b, s in places]
    ac0 = (left+1 if witness or arity == 2
           else places[0][1]+int(rng.integers(1, n-1)))
    ports = {inks[0]: ('top', (0, ac0))}
    if arity == 2:
        ports[inks[1]] = ('bottom', (bh-1, left+2))
    stencils = {}
    for i, s in enumerate(sources):
        stencils[s] = (_mask(rng, n, int(rng.integers(7))) if not (i and rng.random() < .12)
                       else stencils[sources[0]].copy())
    if witness:
        # Farthest hit at row n-2; bottom edge exists off-axis. Two separated right fragments.
        m = np.zeros((n, n), bool)
        m[0, :2] = True
        m[0, -1] = m[n-2, 1] = m[-1, -1] = True
        stencils[sources[0]] = m
    # Force a visible hit on each source's own axis when its token intersects that axis.
    for a, b, s in places:
        ink = mapping[s]
        axis = ports[ink][1][1]-b
        if 0 <= axis < n and not stencils[s][:, axis].any():
            stencils[s][int(rng.integers(n)), axis] = True
    for a, b, s in places:
        board[a:a+n, b:b+n] = s
    for ink, (_, p) in ports.items():
        board[p] = ink
    grid[fr:fr+bh, fc:fc+bw] = board
    sr = _spaced_starts(rng, categories, n, height)
    kr = _spaced_starts(rng, categories, 2, height)
    for y, s in zip(sr, rng.permutation(sources)):
        grid[y:y+n, sx:sx+n][stencils[int(s)]] = s
    for y, s in zip(kr, rng.permutation(sources)):
        tile = np.full((2, 2), int(s), np.int8)
        corner = int(rng.integers(4))
        tile[corner//2, corner % 2] = mapping[int(s)]
        grid[y:y+2, kx:kx+2] = tile
    material = {ink: set() for ink in inks}
    for a, b, s in places:
        for u, v in np.argwhere(stencils[s]):
            material[mapping[s]].add((a+int(u), b+int(v)))
    out, _ = _route(board, bg, ports, material)
    # Rotate/reflection the complete scene; the key is ordered by counts, not its corner.
    if rng.random() < .5:
        grid, out = np.fliplr(grid), np.fliplr(out)
    return {'input': np.rot90(grid, turn).copy(), 'output': np.rot90(out, turn).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('seed', seed, 0, None),
                                   ('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or value < low or (high is not None and value > high)):
            raise ValueError(f"illegal {name}")
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(3))  # larger stencil, repeated source tokens, held-out orientation
    base_turn = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            first = split == 'train' and index == 0
            dual = split == 'train' and index == 1
            arity = 1 if first else 2 if dual else int(rng.integers(1, 3))
            n = 4 if first or dual else int(rng.integers(4 if arity == 2 else 3, 6))
            if shift == 0:
                n = 4 if split == 'train' else 5
            elif first:
                n = 4
            token_count = 2 if first else int(rng.integers(2, 5))
            categories = 2 if first else int(rng.integers(2, token_count+1))
            if shift == 1:
                token_count = (2 if index < 2 else int(rng.integers(2, 4))) if split == 'train' else 4
                categories = 2 if first or dual or split == 'test' else int(rng.integers(2, token_count+1))
            turn = base_turn if shift == 2 and split == 'train' else (
                (base_turn+int(rng.choice([1, 3]))) % 4 if shift == 2 else int(rng.integers(4)))
            overlap = arity == 2 and (dual or rng.random() < .5)
            for _ in range(12):
                pair = _make_pair(rng, n, arity, token_count, categories, turn,
                                  witness=first, overlap=overlap)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input attempts exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _mutant(a, name):
    if name == 'no_stencil':
        cells = {ink: set() for ink in a['ports']}
        for r, c, s in a['places']:
            cells[a['mapping'][s]].update((r+u, c+v) for u in range(a['n']) for v in range(a['n']))
        return _route(a['board'], a['background'], a['ports'], cells)[0]
    if name == 'no_routing':
        out = a['board'].copy()
        out[1:-1, 1:-1] = a['background']
        for ink, cells in a['material'].items():
            for p in cells:
                out[p] = ink
        return out
    if name == 'per_token':
        out, groups = _route(a['board'], a['background'], a['ports'], a['material'])
        h, w = out.shape
        for ink, corner, group in groups:
            r, e, c, d = _box(list(group))
            dr = 1-r if corner[0] == 'top' else h-1-e
            dc = 1-c if corner[1] == 'left' else w-1-d
            for u, v in group:
                out[u+dr, v+dc] = a['background']
        for r, c, source in a['places']:
            ink = a['mapping'][source]
            side, p = a['ports'][ink]
            vertical = side in ('top', 'bottom')
            axis = p[1] if vertical else p[0]
            cells = [(r+int(u), c+int(v)) for u, v in np.argwhere(a['stencils'][source])]
            for low in (True, False):
                group = [q for q in cells if ((q[1] if vertical else q[0]) < axis if low
                                              else (q[1] if vertical else q[0]) > axis)]
                if group:
                    rr, ee, cc, dd = _box(group)
                    corner = ((side, 'left' if low else 'right') if vertical
                              else ('top' if low else 'bottom', side))
                    dr = 1-rr if corner[0] == 'top' else h-1-ee
                    dc = 1-cc if corner[1] == 'left' else w-1-dd
                    for u, v in group:
                        out[u+dr, v+dc] = ink
        return out
    if name == 'wrong_port':
        if len(a['ports']) == 1:
            return _route(a['board'], a['background'], a['ports'], a['material'])[0]
        inks = sorted(a['ports'])
        swapped = {inks[0]: a['material'][inks[1]], inks[1]: a['material'][inks[0]]}
        return _route(a['board'], a['background'], a['ports'], swapped)[0]
    switches = {'nearest': {'nearest': True}, 'separate': {'separate': True}}
    return _route(a['board'], a['background'], a['ports'], a['material'], **switches[name])[0]


def _differs(a, out, name):
    try:
        return not np.array_equal(_mutant(a, name), out)
    except ValueError:
        return True  # The alternative creates a collision or fails to find its assumed hit.


def pair_features(input_grid):
    a = _parse(input_grid)
    _, groups = _route(a['board'], a['background'], a['ports'], a['material'])
    return dict(n=a['n'], tokens=len(a['places']), categories=len(a['mapping']),
                arity=len(a['ports']), sides=sorted(v[0] for v in a['ports'].values()),
                background=a['background'], frame=a['frame'], shape=list(a['grid'].shape),
                board_shape=list(a['board'].shape),
                overlap=any(t in a['mapping'] for t in a['mapping'].values()),
                many_to_one=len(set(a['mapping'].values())) < len(a['mapping']),
                disconnected_stencils=sum(len(_components(m)) > 1 for m in a['stencils'].values()),
                disconnected_groups=sum(len(_components_from_cells(g)) > 1 for _, _, g in groups),
                masks=[hashlib.sha256(m.tobytes()).hexdigest() for m in a['stencils'].values()],
                multi_source_inks=sum(sum(t == ink for t in a['mapping'].values()) > 1 for ink in a['ports']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            expected, groups = _route(a['board'], a['background'], a['ports'], a['material'])
            assert np.array_equal(out, expected)
            assert out.shape == a['board'].shape
            assert np.array_equal(out[0], a['board'][0]) and np.array_equal(out[-1], a['board'][-1])
            assert np.array_equal(out[:, 0], a['board'][:, 0]) and np.array_equal(out[:, -1], a['board'][:, -1])
            assert set(map(int, np.unique(out))) <= {a['background'], a['frame'], *a['ports']}
            assert len(groups) == 2*len(a['ports']), 'each port must have material on both sides'
            # Off-axis material is conserved with one vector per entire ink/half-plane group.
            for ink, cells in a['material'].items():
                side, p = a['ports'][ink]
                vertical = side in ('top', 'bottom')
                axis = p[1] if vertical else p[0]
                hits = [q[0] if vertical else q[1] for q in cells
                        if (q[1] if vertical else q[0]) == axis]
                stop = max(hits) if side in ('top', 'left') else min(hits)
                span = stop if side in ('top', 'left') else (out.shape[0 if vertical else 1]-1-stop)
                assert int(np.count_nonzero(out[1:-1, 1:-1] == ink)) == len(cells)+span-len(hits)
            assert _differs(a, out, 'no_routing')
            if split == 'train':
                train.append((a, out))
    assert {len(a['ports']) for a, _ in train} == {1, 2}
    for name in ('no_stencil', 'nearest', 'separate', 'per_token', 'wrong_port'):
        assert any(_differs(a, out, name) for a, out in train), f'missing {name} demonstration'
    assert any(any(t in a['mapping'] for t in a['mapping'].values()) for a, _ in train), 'need direct-cycle evidence'
    assert any(len(set(a['mapping'].values())) < len(a['mapping']) for a, _ in train), 'need shared-ink evidence'


def self_test():
    """Streaming 42-seed checks, legal counts, counterfactuals and equivariance."""
    all_features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                signature.update(str(inp.shape).encode()+inp.tobytes()+out.tobytes())
                all_features.append(dict(split=split, **pair_features(inp)))
                assert np.array_equal(solve(inp), out)
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                if split == 'train' and seed < 4:
                    perm = np.random.default_rng(9000+seed).permutation(10)
                    assert np.array_equal(solve(perm[inp]), perm[out])
                    analysis = _parse(inp)
                    boxes = list(analysis['key_boxes'].values())
                    changed = inp.copy()
                    a, b, x, y = boxes[0]
                    aa, bb, xx, yy = boxes[1]
                    changed[a:b, x:y], changed[aa:bb, xx:yy] = inp[aa:bb, xx:yy], inp[a:b, x:y]
                    assert np.array_equal(solve(changed), out), 'key-bank order must not matter'
        signatures.add(signature.hexdigest())
    assert len(signatures) == 42
    assert {f['n'] for f in all_features} == {3, 4, 5}
    assert {f['tokens'] for f in all_features} == {2, 3, 4}
    assert {f['categories'] for f in all_features} == {2, 3, 4}
    assert {f['background'] for f in all_features} == set(range(10))
    assert {f['frame'] for f in all_features} == set(range(10))
    assert len({tuple(f['shape']) for f in all_features}) >= 30
    assert len({m for f in all_features for m in f['masks']}) >= 70
    for split in ('train', 'test'):
        subset = [f for f in all_features if f['split'] == split]
        assert {s for f in subset for s in f['sides']} == {'top', 'bottom', 'left', 'right'}
        assert {f['arity'] for f in subset} == {1, 2}
        assert any(f['overlap'] for f in subset)
        assert any(f['multi_source_inks'] for f in subset)
        assert any(f['disconnected_groups'] for f in subset)
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(1234+tr*10+te, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                   (True, 1), (4.5, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p = generate_puzzle(119, 7, 3)
    q = generate_puzzle(119, 7, 3)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    return dict(seeds=42, pairs=len(all_features), distinct_puzzles=len(signatures),
                input_shapes=len({tuple(f['shape']) for f in all_features}),
                stencil_masks=len({m for f in all_features for m in f['masks']}),
                disconnected_group_pairs=sum(f['disconnected_groups'] > 0 for f in all_features),
                overlapping_map_pairs=sum(f['overlap'] for f in all_features))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
