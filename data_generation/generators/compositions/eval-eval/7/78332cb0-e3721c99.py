"""Topology-certified jigsaws: weld fragments, then test the resulting cavities.

Only standard-library modules and NumPy are used.  The decoder reads pixels;
no generation plan, seed, examples, parent code, or file is consulted by solve.
"""

from itertools import permutations
import json
import numpy as np


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, group = [(r, c)], []
        seen[r, c] = True
        while stack:
            a, b = stack.pop()
            group.append((a, b))
            for x, y in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= x < h and 0 <= y < w and mask[x, y] and not seen[x, y]:
                    seen[x, y] = True
                    stack.append((x, y))
        groups.append(group)
    return groups


def _mask(cells):
    points = np.asarray(cells, dtype=int)
    low, high = points.min(axis=0), points.max(axis=0) + 1
    result = np.zeros(tuple(high-low), dtype=bool)
    result[tuple((points-low).T)] = True
    return result


def _holes(mask):
    # A padded exterior prevents a canvas edge from closing an open notch.
    return len(_components(~np.pad(np.asarray(mask, dtype=bool), 1))) - 1


def _cavities_are_background(grid, cells, bg):
    """Reject foreign foreground nested inside a component's cavity."""
    points = np.asarray(cells, dtype=int)
    low = points.min(axis=0)
    padded = np.pad(_mask(cells), 1)
    for region in _components(~padded):
        if any(r in (0, padded.shape[0]-1) or c in (0, padded.shape[1]-1)
               for r, c in region):
            continue
        if any(grid[r+int(low[0])-1, c+int(low[1])-1] != bg for r, c in region):
            return False
    return True


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.flat))


def _grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('grids must be 2D integer arrays, sides 1..30, colors 0..9')
    return a.astype(np.int8, copy=True)


def _partition(body, divider, bg):
    rows = list(map(int, np.flatnonzero(np.all(body == divider, axis=1))))
    cols = list(map(int, np.flatnonzero(np.all(body == divider, axis=0))))
    rr, cc = [-1] + rows + [body.shape[0]], [-1] + cols + [body.shape[1]]
    hs = [b-a-1 for a, b in zip(rr, rr[1:])]
    ws = [b-a-1 for a, b in zip(cc, cc[1:])]
    if not hs or not ws or min(hs) < 5 or min(ws) < 5:
        return None
    if len(set(hs)) != 1 or len(set(ws)) != 1:
        return None
    panels = [body[r+1:rr[i+1], c+1:cc[j+1]].copy()
              for i, r in enumerate(rr[:-1]) for j, c in enumerate(cc[:-1])]
    if not 2 <= len(panels) <= 5 or len(panels)*ws[0] > 30:
        return None
    colors = set(map(int, np.unique(body))) - {bg, divider}
    if len(colors) != 1 or any(np.any(p == divider) for p in panels):
        return None
    query = colors.pop()
    if any(not np.any(p == query) or not np.all(p[0] == bg)
           or not np.all(p[-1] == bg) for p in panels):
        return None
    sigs = [(tuple(map(int, np.flatnonzero(p[:, 0] == query))),
             tuple(map(int, np.flatnonzero(p[:, -1] == query)))) for p in panels]
    if any(not a and not b for a, b in sigs):
        return None
    if sum(not a for a, b in sigs) != 1 or sum(not b for a, b in sigs) != 1:
        return None
    return dict(panels=panels, signatures=sigs, query=query,
                layout=(len(hs), len(ws)), panel_shape=(hs[0], ws[0]))


def _parse(input_grid):
    """Find a colored key beside a divided monochrome fragment board.

    Rotations are a coordinate normalization of the whole input, never a
    permitted rotation of an individual piece.
    """
    g = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        a = np.rot90(g, rotation)
        for divider in map(int, np.unique(a)):
            for line in map(int, np.flatnonzero(np.all(a == divider, axis=1))):
                if line < 3 or a.shape[0]-line-1 < 5:
                    continue
                header, body = a[:line], a[line+1:]
                corners = {int(header[r, c]) for r, c in
                           ((0, 0), (0, -1), (-1, 0), (-1, -1))}
                if len(corners) != 1:
                    continue
                bg = corners.pop()
                if bg == divider or np.any(header == divider):
                    continue
                board = _partition(body, divider, bg)
                if board is None or np.any(header == board['query']):
                    continue
                mapping, glyphs = {}, []
                good = True
                for color in map(int, np.unique(header)):
                    if color == bg:
                        continue
                    groups = _components(header == color)
                    if len(groups) != 1:
                        good = False
                        break
                    if not _cavities_are_background(header, groups[0], bg):
                        good = False
                        break
                    glyph = _mask(groups[0])
                    k = _holes(glyph)
                    if k in mapping:
                        good = False
                        break
                    mapping[k] = color
                    glyphs.append((k, glyph))
                if good and mapping:
                    board.update(rotation=rotation, background=bg, divider=divider,
                                 mapping=mapping, glyphs=glyphs)
                    candidates.append(board)
    if len(candidates) != 1:
        raise ValueError('expected one unambiguous key and equal-panel board')
    return candidates[0]


def _orders(parsed, match_ports=True):
    sigs = parsed['signatures']
    start = next(i for i, (a, b) in enumerate(sigs) if not a)
    end = next(i for i, (a, b) in enumerate(sigs) if not b)
    middle = [i for i in range(len(sigs)) if i not in (start, end)]
    for perm in permutations(middle):
        order = (start,) + perm + (end,)
        if not match_ports or all(sigs[i][1] == sigs[j][0]
                                  for i, j in zip(order, order[1:])):
            yield order


def _paint(parsed, order):
    welded = np.concatenate([parsed['panels'][i] for i in order], axis=1)
    result = welded.copy()
    counts, areas = [], []
    for cells in _components(welded == parsed['query']):
        if not _cavities_are_background(welded, cells, parsed['background']):
            return None
        k = _holes(_mask(cells))
        if k not in parsed['mapping']:
            return None
        result[tuple(np.asarray(cells).T)] = parsed['mapping'][k]
        counts.append(k)
        areas.append(len(cells))
    return result, tuple(counts), tuple(areas)


def _analyze(input_grid):
    p = _parse(input_grid)
    geometric = list(_orders(p))
    valid = [(order, painted) for order in geometric
             if (painted := _paint(p, order)) is not None]
    if len(valid) != 1:
        raise ValueError('topology and ports must select exactly one complete order')
    order, (answer, counts, areas) = valid[0]
    p.update(order=order, counts=counts, areas=areas, geometric_orders=geometric,
             canonical_output=answer,
             output=np.rot90(answer, -p['rotation']).copy())
    return p


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique welded, topology-colored assembly using only input."""
    return _analyze(input_grid)['output']


def _glyph(k, rng):
    # Compact prototypes; their own topology defines the class, never rank.
    rows = 1 if k <= 2 else 2
    cols = (k+rows-1)//rows
    result = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(k):
        result[1+2*(i//cols), 1+2*(i%cols)] = False
    if rng.random() < .5:
        result = result[::-1].copy()
    return result


def _path(mask, start, end, hub):
    r, c = start
    s, d = end
    mask[r, min(c, hub):max(c, hub)+1] = True
    mask[min(r, s):max(r, s)+1, hub] = True
    mask[s, min(hub, d):max(hub, d)+1] = True


def _panel(h, w, incoming, outgoing, kind, top, bottom, rng, rich):
    """Two rails, a one-sided cap, or two separated half-caps.

    Attached loops have independently sampled cavity areas. Optional contour
    growth is accepted only if every component and its hole count is preserved.
    """
    m = np.zeros((h, w), dtype=bool)
    if kind == 'left':
        u, v = outgoing
        hub = int(rng.integers(1, w-2))
        _path(m, (u, w-1), (v, w-1), hub)
        rail = outgoing
    elif kind == 'right':
        u, v = incoming
        hub = int(rng.integers(2, w-1))
        _path(m, (u, 0), (v, 0), hub)
        rail = incoming
    elif kind == 'cap':
        _path(m, (incoming[0], 0), (incoming[1], 0), 1)
        _path(m, (outgoing[0], w-1), (outgoing[1], w-1), w-2)
        return m
    else:
        hub = int(rng.integers(1, w-1))
        _path(m, (incoming[0], 0), (outgoing[0], w-1), hub)
        _path(m, (incoming[1], 0), (outgoing[1], w-1), hub)
        rail = (min(incoming[0], outgoing[0]), max(incoming[1], outgoing[1]))
    # Rings attach to a rail at one cell. Keeping them away from the edges
    # preserves complete edge signatures even after shape growth.
    for count, side in ((top, -1), (bottom, 1)):
        if not count:
            continue
        r = rail[0] if side == -1 else rail[1]
        # The attachment must meet the rail at the selected interior column.
        possible = [c for c in range(1, w-3) if m[r, c:c+3].any()]
        c = possible[int(rng.integers(len(possible)))]
        available = r-1 if side == -1 else h-r-2
        # Extra room changes cavity area, not its class. Occasionally an
        # asymmetric L-shaped cavity replaces a rectangular one.
        depth = 3 if count == 1 and available >= 3 and rng.random() < .5 else 2
        span = int(rng.integers(3, w-c))
        for j in range(count):
            a = r + side*(depth*(j+1))
            b = r + side*depth*j
            lo, hi = sorted((a, b))
            m[lo:hi+1, c] = True
            m[lo:hi+1, c+span-1] = True
            m[lo, c:c+span] = True
            m[hi, c:c+span] = True
            if depth == 3 and span == 4 and rng.random() < .5:
                m[lo+1, c+1] = True
    if rich:
        # Grow branches/thicker contours without changing topology or contacts.
        old = sorted((_holes(_mask(g)), len(g)) for g in _components(m))
        old_holes = sorted(x[0] for x in old)
        old_number = len(old)
        for _ in range(int(rng.integers(2, 8))):
            options = []
            for r, c in np.argwhere(~m):
                r, c = int(r), int(c)
                if 1 <= r < h-1 and 1 <= c < w-1 and any(
                        m[x, y] for x, y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
                    options.append((r, c))
            if not options:
                break
            point = options[int(rng.integers(len(options)))]
            trial = m.copy()
            trial[point] = True
            groups = _components(trial)
            if len(groups) == old_number and sorted(_holes(_mask(g)) for g in groups) == old_holes:
                m = trial
    return m


def _layout(panels, order, shape, divider):
    nr, nc = shape
    h, w = panels[0].shape
    board = np.full((nr*(h+1)-1, nc*(w+1)-1), divider, dtype=np.int8)
    for slot, i in enumerate(order):
        r, c = divmod(slot, nc)
        board[r*(h+1):r*(h+1)+h, c*(w+1):c*(w+1)+w] = panels[i]
    return board


def _build_pair(rng, n, rich, larger, rotation, require_geometry=False):
    # n=4 has one cap and one beaded through tile. n=5 adds a signature
    # transition before or after this ambiguous pair, forcing geometric order.
    h = int(rng.integers(13, 17) if larger else rng.integers(10, 13))
    w = int(rng.integers(5, 7)) if n == 4 else 5
    top_r = 5 if larger else 3
    bot_r = h-4
    a = (top_r, bot_r)
    b = (top_r+1, bot_r) if rng.random() < .5 else (top_r, bot_r-1)
    if b[1]-b[0] < 2:
        b = (top_r+1, bot_r)
    left_k = int(rng.integers(0, 2))
    right_k = 1-left_k if not larger else int(rng.choice([k for k in range(3) if k != left_k]))
    bead_k = int(rng.integers(1, 3)) if larger else 1
    # Only top loops are used for two-cavity stacks; bottom loops can carry a
    # single extra cavity when there is less vertical headroom.
    def split_extra(k, sig):
        if k == 0:
            return 0, 0
        if k == 1 and rng.random() < .5:
            return 0, 1
        if sig[0]-2*k >= 1:
            return k, 0
        return 1, k-1
    bridge_first = bool(rng.integers(2))
    left_sig, mid_sig, right_sig = (a, b, b) if bridge_first and n == 5 else (a, a, b if n == 5 else a)
    lt, lb = split_extra(left_k, left_sig)
    rt, rb = split_extra(right_k, right_sig)
    bt, bb = split_extra(bead_k, mid_sig)
    definitions = [([], left_sig, 'left', lt, lb)]
    if n == 5 and bridge_first:
        definitions.append((a, b, 'through', 0, 0))
    bead_before = bool(rng.integers(2))
    pair = [(mid_sig, mid_sig, 'through', bt, bb),
            (mid_sig, mid_sig, 'cap', 0, 0)]
    if not bead_before:
        pair.reverse()
    definitions.extend(pair)
    if n == 5 and not bridge_first:
        definitions.append((a, b, 'through', 0, 0))
    definitions.append((right_sig, [], 'right', rt, rb))
    masks = [_panel(h, w, inc, out, kind, t, btm, rng, rich)
             for inc, out, kind, t, btm in definitions]
    # The target is constructed from masks, not by asking solve for a label.
    welded = np.concatenate(masks, axis=1)
    groups = _components(welded)
    counts = [_holes(_mask(g)) for g in groups]
    if len(groups) != 2 or min(counts) < 1:
        return None
    expected_counts = sorted((1+left_k+(bead_k if bead_before else 0),
                              1+right_k+(0 if bead_before else bead_k)))
    if sorted(counts) != expected_counts:
        return None
    classes = sorted(set(counts))
    # Every alternative arrangement must be rejected by the intended joint rule.
    palette = list(map(int, rng.permutation(10)))
    bg, divider, query = palette[:3]
    rng.shuffle(classes)
    mapping = dict(zip(classes, palette[3:]))
    panels = [np.where(m, query, bg).astype(np.int8) for m in masks]
    layouts = [(1, n)]
    if n == 4 and h == 10:
        layouts.append((2, 2))
    layout = layouts[int(rng.integers(len(layouts)))]
    order = list(map(int, rng.permutation(n)))
    if order == list(range(n)):
        order = order[1:] + order[:1]
    board = _layout(panels, order, layout, divider)
    glyphs = [_glyph(k, rng) for k in classes]
    header_h = max(g.shape[0] for g in glyphs)+2
    needed = sum(g.shape[1] for g in glyphs)+len(glyphs)+1
    if needed > board.shape[1] or header_h+1+board.shape[0] > 30:
        return None
    header = np.full((header_h, board.shape[1]), bg, dtype=np.int8)
    c = int(rng.integers(1, board.shape[1]-needed+2))
    for k, glyph in zip(classes, glyphs):
        r = int(rng.integers(1, header_h-glyph.shape[0]))
        header[r:r+glyph.shape[0], c:c+glyph.shape[1]][glyph] = mapping[k]
        c += glyph.shape[1]+1
    inp = np.concatenate((header, np.full((1, board.shape[1]), divider, dtype=np.int8), board))
    out = np.full(welded.shape, bg, dtype=np.int8)
    for cells, k in zip(groups, counts):
        out[tuple(np.asarray(cells).T)] = mapping[k]
    result = {'input': np.rot90(inp, rotation).copy(),
              'output': np.rot90(out, rotation).copy()}
    try:
        parsed = _analyze(result['input'])
    except ValueError:
        return None
    if not np.array_equal(parsed['output'], result['output']):
        return None
    if len(parsed['geometric_orders']) < 2:
        return None
    if require_geometry and not _geometry_witness(parsed):
        return None
    return result


def _geometry_witness(parsed):
    geometric = set(parsed['geometric_orders'])
    # A candidate which passes the visible topology key but violates port
    # alignment proves that topology alone is insufficient.
    return any(order not in geometric and _paint(parsed, order) is not None
               for order in _orders(parsed, match_ports=False))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    base = int(rng.integers(4))
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            n = 4 if split == 'train' and i == 0 else 5 if i == 1 else int(rng.integers(4, 6))
            rotation = base if i % 2 == 0 else (base+1) % 4
            # Tests enlarge the loops, bend/branch contours and remove the
            # array cue. All are compatible with the same visible certification.
            larger = split == 'test'
            rich = (split == 'test') or i % 2 == 1
            witness = split == 'train' and i == 1
            for _ in range(180):
                pair = _build_pair(rng, n, rich, larger, rotation, witness)
                if pair is not None and _key(pair['input']) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError('bounded construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            p = _analyze(inp)
            assert np.array_equal(out, p['output']), 'wrong assembly, topology color, or copied background'
            assert out.size == np.prod(p['panel_shape'])*len(p['panels'])
            assert np.count_nonzero(out != p['background']) == sum(
                np.count_nonzero(tile == p['query']) for tile in p['panels']), 'every fragment cell is conserved'
            assert all(k in p['mapping'] for k in p['counts'])
            assert p['query'] not in np.unique(out) and p['divider'] not in np.unique(out)
            if split == 'train':
                training.append(p)
    assert any(len(p['geometric_orders']) > 1 for p in training), 'show why topology chooses order'
    assert any(_geometry_witness(p) for p in training), 'show why ports still constrain order'
    assert len({p['rotation'] % 2 for p in training}) == 2, 'demonstrate both assembly axes'


def diversity_features(pair):
    p = _analyze(pair['input'])
    masks = [x == p['query'] for x in p['panels']]
    return dict(shape=tuple(np.asarray(pair['input']).shape), panel=p['panel_shape'],
                n=len(masks), layout=p['layout'], rotation=p['rotation'],
                roles=(p['background'], p['divider'], p['query']),
                classes=tuple(sorted(p['mapping'])), counts=p['counts'],
                signatures=tuple(p['signatures']), order=p['order'],
                geometry_witness=_geometry_witness(p),
                ambiguous_without_topology=len(p['geometric_orders']),
                fragment_holes=tuple(_holes(_mask(g)) for m in masks for g in _components(m)),
                masks=tuple(_key(m.astype(np.int8)) for m in masks))


def self_test():
    # Hand-made topology fixtures include a diagonal escape and a true notch.
    diagonal = np.array([[0,1,1],[1,0,1],[1,1,1]], dtype=bool)
    assert _holes(diagonal) == 1
    diagonal[0,1] = False
    assert _holes(diagonal) == 0
    assert _holes(np.array([[1,0,1],[1,0,1],[1,1,1]], dtype=bool)) == 0
    # Tiny hand-built cases exercise the general rule beyond the sampled
    # four/five-panel, two-component rail construction.
    for tracks in (1, 3):
        h = 5 if tracks == 1 else 7
        left, right = np.zeros((h, 5), dtype=bool), np.zeros((h, 5), dtype=bool)
        for r in ([2] if tracks == 1 else [1, 3, 5]):
            left[r, 2:] = True
            right[r, :3] = True
        panels = [np.where(x, 5, 0).astype(np.int8) for x in (right, left)]
        board = _layout(panels, [0, 1], (1, 2), 9)
        header = np.zeros((5, 11), dtype=np.int8)
        header[1:4, 1:4] = 2  # a solid, zero-hole key glyph
        inp = np.vstack((header, np.full((1, 11), 9, dtype=np.int8), board))
        expected = np.where(np.hstack((left, right)), 2, 0).astype(np.int8)
        assert np.array_equal(solve(inp), expected)
        assert len(_components(expected == 2)) == tracks
        for turn in range(4):
            assert np.array_equal(solve(np.rot90(inp, turn)), np.rot90(expected, turn))
    left = np.zeros((5, 5), dtype=bool)
    left[1, 1:] = left[3, 1:] = True
    left[1:4, 1] = True
    right = left[:, ::-1].copy()
    assert _holes(left) == _holes(right) == 0
    board = _layout([np.where(x, 5, 0).astype(np.int8) for x in (right, left)], [0, 1], (1, 2), 9)
    header = np.zeros((5, 11), dtype=np.int8)
    header[1:4, 1:4] = 2
    header[2, 2] = 0  # one-hole key glyph
    inp = np.vstack((header, np.full((1, 11), 9, dtype=np.int8), board))
    expected = np.where(np.hstack((left, right)), 2, 0).astype(np.int8)
    assert _holes(expected == 2) == 1
    assert np.array_equal(solve(inp), expected)
    # A solid key cannot certify this newly closed loop. Duplicate topology
    # classes in different colors are also invalid, even with spatial order.
    for duplicate in (False, True):
        bad = inp.copy()
        if duplicate:
            bad[1:4, 6:9] = 3
            bad[2, 7] = 0
        else:
            bad[2, 2] = 2
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid topology key accepted')
    records, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        key = tuple(_key(pair['input']) for s in puzzle for pair in puzzle[s])
        keys.add(key)
        for s in puzzle:
            for pair in puzzle[s]:
                original = np.asarray(pair['input']).copy()
                assert np.array_equal(solve(original), pair['output'])
                assert np.array_equal(original, pair['input']), 'solver mutated input'
                records.append(diversity_features(pair))
    assert len(keys) == 42
    assert {p['n'] for p in records} == {4, 5}
    assert {p['rotation'] for p in records} == set(range(4))
    assert len({p['shape'] for p in records}) >= 20
    assert len({p['panel'] for p in records}) >= 8
    assert len({m for p in records for m in p['masks']}) >= 100
    assert len({p['roles'] for p in records}) >= 80
    assert any(p['layout'] == (2, 2) for p in records)
    assert any(len(p['classes']) == 1 for p in records)
    assert any(len(p['classes']) == 2 for p in records)
    assert max(k for p in records for k in p['counts']) >= 4
    # Same-seed determinism and every legal explicit count combination.
    a, b = generate_puzzle(123), generate_puzzle(123)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A changed answer cannot silently validate.
    bad = generate_puzzle(17, 2, 1)
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(records), shapes=len({p['shape'] for p in records}),
                panel_shapes=len({p['panel'] for p in records}),
                fragment_masks=len({m for p in records for m in p['masks']}),
                role_palettes=len({p['roles'] for p in records}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
