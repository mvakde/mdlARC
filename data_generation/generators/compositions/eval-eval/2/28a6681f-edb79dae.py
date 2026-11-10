"""Legend-cast riverbeds: decode a terrain mosaic, then conserve and settle water.

Only standard library and NumPy. No parent files or generation metadata are read
by solve. All diagnostics are returned in memory; this module writes no files.
"""
from __future__ import annotations

import hashlib
import numpy as np


# Fixed ordered glyph, inherited in shape from B: the broad left part is source.
_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
VERIFICATION_SEEDS = (0, 371042009, 1456409, 275016737, 857563988,
    207629110, 397405283, 946712494, 982883209, 136009038, 512484130,
    591641609, 521158983, 196729683, 116564802, 929737791, 856925379,
    565371420, 748773729, 377714857, 855682130, 966980826, 125852983,
    65867108, 880478246, 561244393, 979626091, 267304047, 805721092,
    104996502, 687364298, 776635459, 827251075, 223002902, 722025193,
    883061076, 24719666, 464811288, 425161448, 158609464, 2385123,
    432571621)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    return (int(a[:, 0].min()), int(a[:, 0].max())+1,
            int(a[:, 1].min()), int(a[:, 1].max())+1)


def _frame(grid, bg):
    candidates = []
    for color in np.unique(grid):
        if color == bg:
            continue
        cells = np.argwhere(grid == color)
        r, e, c, d = _box(cells)
        h, w = e-r, d-c
        if h < 7 or w < 7 or len(cells) != 2*h+2*w-4:
            continue
        if (np.all(grid[r, c:d] == color) and np.all(grid[e-1, c:d] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, d-1] == color)):
            candidates.append(((r, e, c, d), int(color)))
    if len(candidates) != 1:
        raise ValueError("one exclusive rectangular tank frame is required")
    return candidates[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    if not all(np.all(edge == bg) for edge in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("the canvas must have a blank outer rim")
    frame, wall = _frame(grid, bg)
    r, e, c, d = frame
    exterior = grid != bg
    exterior[r:e, c:d] = False
    key_cells = np.zeros(grid.shape, bool)
    mapping, key_boxes = {}, {}
    for cells in _components(exterior):
        values = {int(grid[a, b]) for a, b in cells}
        if len(values) == 1:
            continue
        a, b, x, y = _box(cells)
        tile = grid[a:b, x:y]
        if tile.shape != (3, 4) or len(cells) != 12 or len(values) != 2:
            raise ValueError("unrecognized ordered key")
        source, target = int(tile[_ARROW][0]), int(tile[~_ARROW][0])
        if (source == target or not np.all(tile[_ARROW] == source)
                or not np.all(tile[~_ARROW] == target) or source in mapping):
            raise ValueError("nonfunctional or reversed key")
        mapping[source], key_boxes[source] = target, (a, b, x, y)
        key_cells[a:b, x:y] = True
    if not 1 <= len(mapping) <= 3:
        raise ValueError("need one to three keyed terrain categories")
    remaining = exterior & ~key_cells
    if set(map(int, np.unique(grid[remaining]))) != set(mapping):
        raise ValueError("each key needs one source-colored stencil")
    stencils, stencil_boxes, sizes = {}, {}, set()
    for source in mapping:
        a, b, x, y = _box(np.argwhere(remaining & (grid == source)))
        tile = grid[a:b, x:y]
        n = b-a
        if n != y-x or not 3 <= n <= 6 or not np.all(np.isin(tile, [bg, source])):
            raise ValueError("stencil bounds must reveal a size-3..6 square")
        mask = tile == source
        heights = mask.sum(axis=0)
        if (not np.all(mask == (np.arange(n)[:, None] >= n-heights[None, :]))
                or heights.min() < 1 or heights.min() == n or heights.max() != n):
            raise ValueError("stencils must be non-solid grounded height profiles")
        stencils[source], stencil_boxes[source] = mask.copy(), (a, b, x, y)
        sizes.add(n)
    if len(sizes) != 1:
        raise ValueError("all stencils must have the same size")
    # Complete clue rectangles, including their inactive cells, must be separate.
    boxes = list(stencil_boxes.values()) + list(key_boxes.values())
    for i, (a, b, x, y) in enumerate(boxes):
        if not (b <= r or a >= e or y <= c or x >= d):
            raise ValueError("clues must lie outside the tank")
        for aa, bb, xx, yy in boxes[i+1:]:
            if not (b < aa or bb < a or y < xx or yy < x):
                raise ValueError("clue rectangles need a blank-cell gap")
    n = sizes.pop()
    board = grid[r:e, c:d].copy()
    base = board.shape[0]-1-n
    if base < 3:
        raise ValueError("need visible headspace above the encoded riverbed")
    sources = set(mapping)
    seen = np.zeros(board.shape, bool)
    places = []
    col = 1
    while col < board.shape[1]-1:
        source = int(board[-2, col])
        if source not in sources:
            col += 1
            continue
        tile = board[base:base+n, col:col+n]
        if tile.shape != (n, n) or not np.all(tile == source) or col+n >= board.shape[1]:
            raise ValueError("bottom-aligned square placeholders are required")
        places.append((base, col, source))
        seen[base:base+n, col:col+n] = True
        col += n
    if len(places) < 2 or {s for _, _, s in places} != sources:
        raise ValueError("the mosaic needs at least two blocks and every legend category")
    for source in sources:
        if not np.array_equal(board == source, seen & (board == source)):
            raise ValueError("a source color appears outside the bottom mosaic")
    interior = board[1:-1, 1:-1]
    others = set(map(int, np.unique(interior))) - sources - {bg}
    if len(others) != 1:
        raise ValueError("the remaining interior color must identify liquid")
    water = others.pop()
    forbidden = {bg, wall, water}
    if forbidden & (sources | set(mapping.values())):
        raise ValueError("structural roles, liquid, and legend palette must be disjoint")
    if np.any(np.argwhere(board == water)[:, 0] >= base):
        raise ValueError("the measured supply must start in the open headspace")
    decoded = board.copy()
    # Original assignments are recorded first; replacement colors are not decoded.
    for a, b, source in places:
        decoded[a:a+n, b:b+n] = np.where(stencils[source], mapping[source], bg)
    return dict(background=bg, water=water, wall=wall, frame=frame, board=board,
                decoded=decoded, n=n, base=base, places=places, mapping=mapping,
                stencils=stencils, stencil_boxes=stencil_boxes, key_boxes=key_boxes)


def _pour(grid, bg, water, reverse=False, top_first=False):
    """Supported lateral flow, with no uphill teleport through a high ridge.

    The family uses complete pool levels, so all four adversarial pour orders
    must agree. Nearest-outlet tie breaks are computational, not task rules.
    """
    out = grid.copy()
    points = sorted(map(tuple, np.argwhere(grid == water)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    out[out == water] = bg
    h, w = out.shape
    for a, b in points:
        row, col = int(a), int(b)
        while row > 0 and out[row, col] == water:
            row -= 1
        if out[row, col] != bg:
            raise ValueError("liquid trapped under a full ceiling")
        for _ in range(h+1):
            if row+1 < h and out[row+1, col] == bg:
                row += 1
                continue
            left = right = col
            while left > 0 and out[row, left-1] == bg:
                left -= 1
            while right+1 < w and out[row, right+1] == bg:
                right += 1
            exits = [] if row+1 == h else [x for x in range(left, right+1)
                                           if out[row+1, x] == bg]
            if not exits:
                out[row, col] = water
                break
            col = min(exits, key=lambda x: (abs(x-col), -x if reverse else x))
            row += 1
        else:
            raise AssertionError("a downward pour failed to terminate")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible input only; return the inclusive tank crop."""
    a = _parse(input_grid)
    return _pour(a['decoded'], a['background'], a['water'])


def _capture(terrain, pools):
    """Bit sets of all possible dry destinations, independently of unit pouring."""
    h, w = terrain.shape
    fixed = np.zeros((h, w), np.int32)
    for i, pool in enumerate(pools):
        fixed[pool] = 1 << i
    labels = np.zeros((h, w), np.int32)
    unknown = 1 << len(pools)
    for row in range(h-1, -1, -1):
        col = 0
        while col < w:
            if terrain[row, col]:
                col += 1
                continue
            end = col+1
            while end < w and not terrain[row, end]:
                end += 1
            outlets = int(np.bitwise_or.reduce(fixed[row, col:end]))
            if row+1 < h:
                for x in range(col, end):
                    if not terrain[row+1, x]:
                        outlets |= int(labels[row+1, x])
            for x in range(col, end):
                if fixed[row, x]:
                    labels[row, x] = fixed[row, x]
                elif row+1 < h and not terrain[row+1, x]:
                    labels[row, x] = labels[row+1, x]
                else:
                    labels[row, x] = outlets or unknown
            col = end
    return labels


def _pools(terrain, low, high):
    candidates = []
    rows = np.arange(terrain.shape[0])[:, None]
    for surface in range(low, high):
        for cells in _components(~terrain & (rows >= surface)):
            mask = np.zeros(terrain.shape, bool)
            rr, cc = zip(*cells)
            mask[rr, cc] = True
            if mask[surface].any() and len(cells) >= 2:
                candidates.append(mask)
    return candidates


def _branches(pool):
    top = int(np.argwhere(pool)[:, 0].min())
    return max([1]+[len(_components(pool & (np.arange(pool.shape[0])[:, None] >= r)))
                    for r in range(top+1, pool.shape[0])])


def _profile(rng, n):
    """Open valleys, stepped shoulders, shelves, and unequal adjacent slopes."""
    style = int(rng.integers(4))
    peak = int(rng.integers(1, n-1))
    if style == 0:
        heights = rng.integers(1, n, size=n)
    elif style == 1:
        heights = np.maximum(1, n-1-np.abs(np.arange(n)-peak))
    elif style == 2:
        heights = np.full(n, int(rng.integers(1, n)))
        heights[int(rng.integers(n)):] = int(rng.integers(1, n))
    else:
        heights = np.clip(np.cumsum(rng.integers(-1, 2, size=n)) + n//2, 1, n-1)
    heights[peak] = n
    # Keep an internal crest and at least one low shoulder; seams remain active.
    heights[int(rng.choice([0, n-1]))] = 1
    return np.arange(n)[:, None] >= n-heights[None, :]


def _spaced_starts(rng, length, sizes):
    slack = length-sum(sizes)-(len(sizes)-1)
    if slack < 0:
        raise ValueError("legend bank does not fit")
    gaps = rng.multinomial(slack, np.full(len(sizes)+1, 1/(len(sizes)+1)))
    result, at = [], int(gaps[0])
    for i, size in enumerate(sizes):
        result.append(at)
        at += size+1+int(gaps[i+1])
    return result


def _pack(rng, board, bg, stencils, mapping, informative=False):
    """Independently shuffled key/stencil banks on any feasible exterior side."""
    n, k = next(iter(stencils.values())).shape[0], len(mapping)
    h, w = board.shape
    # Horizontal banks: n stencil rows, a gap, then three key rows.
    wide = max(w, k*n+k-1, 4*k+k-1)
    tall = max(h, k*n+k-1, 3*k+k-1)
    choices = []
    if h+n+7 <= 30 and wide+2 <= 30:
        choices += ['above', 'below']
    if w+n+8 <= 30 and tall+2 <= 30:
        choices += ['left', 'right']
    if not choices:
        raise ValueError("no exterior legend bank fits")
    side = str(rng.choice(choices))
    horizontal = side in ('above', 'below')
    if horizontal:
        canvas = np.full((h+n+7, wide+2), bg, np.int8)
        br = n+6 if side == 'above' else 1
        lr = 1 if side == 'above' else h+2
        bc = 1+int(rng.integers(wide-w+1))
        s_starts = _spaced_starts(rng, wide, [n]*k)
        k_starts = _spaced_starts(rng, wide, [4]*k)
        sp = [(lr, 1+x) for x in s_starts]
        kp = [(lr+n+1, 1+x) for x in k_starts]
    else:
        canvas = np.full((tall+2, w+n+8), bg, np.int8)
        bc = n+7 if side == 'left' else 1
        lc = 1 if side == 'left' else w+2
        br = 1+int(rng.integers(tall-h+1))
        s_starts = _spaced_starts(rng, tall, [n]*k)
        k_starts = _spaced_starts(rng, tall, [3]*k)
        sp = [(1+x, lc) for x in s_starts]
        kp = [(1+x, lc+n+1) for x in k_starts]
    canvas[br:br+h, bc:bc+w] = board
    order_s, order_k = rng.permutation(list(mapping)), rng.permutation(list(mapping))
    if informative and k > 1 and np.array_equal(order_s, order_k):
        order_k = np.roll(order_k, 1)
    for (r, c), source in zip(sp, order_s):
        canvas[r:r+n, c:c+n][stencils[int(source)]] = source
    for (r, c), source in zip(kp, order_k):
        canvas[r:r+3, c:c+4] = np.where(_ARROW, source, mapping[int(source)])
    return canvas


def _source_mask(rng, labels, pools, base, style):
    source = np.zeros(labels.shape, bool)
    head = (np.arange(labels.shape[0])[:, None] > 0) & (np.arange(labels.shape[0])[:, None] < base)
    for i, pool in enumerate(pools):
        points = np.argwhere((labels == 1 << i) & head)
        volume = int(pool.sum())
        if len(points) < volume:
            raise ValueError("insufficient unambiguous supply space")
        anchor = points[int(rng.integers(len(points)))]
        dr, dc = np.abs(points[:, 0]-anchor[0]), np.abs(points[:, 1]-anchor[1])
        noise = rng.random(len(points))
        scores = (dr+dc+noise if style == 0 else noise if style == 1 else
                  4*dr+dc*.1+noise if style == 2 else 4*dc+dr*.1+noise)
        chosen = points[np.argsort(scores)[:volume]]
        source[chosen[:, 0], chosen[:, 1]] = True
    return source


def _make_pair(rng, roles, mode, index):
    bg, wall, water = roles
    for _ in range(32):
        n = int(rng.integers(3, 7))
        count = int(rng.integers(3 if mode == 'unequal' else 2, min(5, 26//n)+1))
        k = int(rng.integers(1, min(3, count)+1))
        if index in (0, 1):
            k = max(2, k)
        palette = rng.permutation([x for x in range(10) if x not in roles])
        sources = list(map(int, palette[:k]))
        if index == 0:
            mapping = dict(zip(sources, map(int, palette[k:2*k])))
        elif index == 1:
            mapping = {s: sources[i+1] if i+1 < k else int(palette[k])
                       for i, s in enumerate(sources)}
        else:
            mapping = {s: int(rng.choice([x for x in palette if x != s])) for s in sources}
        stencils = {s: _profile(rng, n) for s in sources}
        if index == 0 and not any(not np.array_equal(m, np.fliplr(m)) for m in stencils.values()):
            continue
        gaps = rng.integers(0, 3, size=count+1)
        # Preserve room for legend and frame within 30 columns.
        while count*n+int(gaps.sum()) > 26:
            gaps[int(np.argmax(gaps))] -= 1
        width = count*n+int(gaps.sum())+2
        headspace = n+int(rng.integers(1, 3))
        height, base = n+headspace+2, headspace+1
        board = np.full((height, width), bg, np.int8)
        board[0] = board[-1] = wall
        board[:, 0] = board[:, -1] = wall
        decoded = board.copy()
        assignments = sources + list(map(int, rng.choice(sources, count-k)))
        rng.shuffle(assignments)
        col = 1+int(gaps[0])
        block_mask = np.zeros(board.shape, bool)
        for i, source in enumerate(assignments):
            board[base:base+n, col:col+n] = source
            decoded[base:base+n, col:col+n] = np.where(stencils[source], mapping[source], bg)
            block_mask[base:base+n, col:col+n] = True
            col += n+int(gaps[i+1])
        terrain = decoded != bg
        candidates = [p for p in _pools(terrain, base, height-1)
                      if np.any(p & block_mask)]
        if index == 0:
            # A higher wet cell alongside an unused lower free cell certifies
            # that global bottom-filling is wrong, without a partial-row tie.
            candidates = [p for p in candidates
                          if len(set(np.argwhere(p)[:, 0])) >= 2
                          and np.any((~terrain & ~p) &
                              (np.arange(height)[:, None] > np.argwhere(p)[:, 0].min()))]
        if mode == 'merged':
            chosen = _pools(terrain, base-1, base)
            if len(chosen) != 1 or _branches(chosen[0]) < 2:
                continue
        else:
            rng.shuffle(candidates)
            if not candidates:
                continue
            chosen = [candidates[0]]
            if mode in ('equal', 'unequal'):
                first_level = int(np.argwhere(chosen[0])[:, 0].min())
                for pool in candidates[1:]:
                    level = int(np.argwhere(pool)[:, 0].min())
                    separated = not np.any(pool & chosen[0])
                    # Disjoint columns ensure independent catchments/menisci.
                    separated &= not np.any(pool.any(axis=0) & chosen[0].any(axis=0))
                    if separated and ((level == first_level) == (mode == 'equal')):
                        chosen.append(pool)
                        break
                if len(chosen) < 2:
                    continue
                if mode == 'unequal' and rng.random() < .5:
                    used_cols = np.logical_or.reduce([p.any(axis=0) for p in chosen])
                    for pool in candidates:
                        if not np.any(pool.any(axis=0) & used_cols):
                            chosen.append(pool)
                            break
        labels = _capture(terrain, chosen)
        try:
            source = _source_mask(rng, labels, chosen, base, int(rng.integers(4)))
            inp = board.copy()
            inp[source] = water
            decoded[source] = water
            target = np.logical_or.reduce(chosen)
            if mode == 'merged' and np.array_equal(source.sum(axis=0), target.sum(axis=0)):
                continue
            out = decoded.copy()
            out[out == water] = bg
            out[target] = water
            if not all(np.array_equal(_pour(decoded, bg, water, rev, order), out)
                       for rev in (False, True) for order in (False, True)):
                continue
            canvas = _pack(rng, inp, bg, stencils, mapping, informative=index == 0)
            return {'input': canvas, 'output': out}
        except ValueError:
            continue
    raise RuntimeError("bounded riverbed construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=3, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            mode = ('low' if i == 0 else 'merged' if i == 1 else
                    ('low', 'merged', 'equal')[int(rng.integers(3))]) if split == 'train' else 'unequal'
            for _ in range(8):
                pair = _make_pair(rng, roles, mode, i if split == 'train' else -1)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    a = _parse(inp)
    bg, water = a['background'], a['water']
    decoded = a['decoded']
    terrain = (decoded != bg) & (decoded != water)
    assert out.shape == a['board'].shape
    assert np.array_equal(out[terrain], decoded[terrain]), "decoded terrain changed"
    assert np.array_equal((out != bg) & (out != water), terrain)
    assert int((decoded == water).sum()) == int((out == water).sum()) > 0
    assert np.argwhere(out == water)[:, 0].sum() > np.argwhere(decoded == water)[:, 0].sum()
    assert np.array_equal(solve(inp), out)
    pools = []
    levels = []
    for cells in _components(out == water):
        pool = np.zeros(out.shape, bool)
        rr, cc = zip(*cells)
        pool[rr, cc] = True
        top = min(rr)
        available = ~terrain & (np.arange(out.shape[0])[:, None] >= top)
        closure = next(part for part in _components(available) if cells[0] in part)
        assert set(closure) == set(cells), "incomplete horizontal level or basin leak"
        pools.append(pool)
        levels.append(top)
    labels = _capture(terrain, pools)
    assigned = np.zeros(out.shape, bool)
    for i, pool in enumerate(pools):
        supply = (decoded == water) & (labels == 1 << i)
        assert int(supply.sum()) == int(pool.sum()), "ambiguous or incorrect routed volume"
        assigned |= supply
    assert np.array_equal(assigned, decoded == water)
    # Both ablations must change the liquid itself, not just terrain colors.
    no_a = decoded
    no_b = _pour(a['board'], bg, water)
    assert not np.array_equal(no_a == water, out == water)
    assert not np.array_equal(no_b == water, out == water)
    assert any(np.any((out == water)[r:r+a['n'], c:c+a['n']]) for r, c, _ in a['places'])
    crossings = 0
    for pool in pools:
        touched = sum(bool(pool[r:r+a['n'], c:c+a['n']].any()) for r, c, _ in a['places'])
        crossings = max(crossings, touched)
    return dict(roles=(bg, a['wall'], water), n=a['n'], blocks=len(a['places']),
                categories=len(a['mapping']), pools=len(pools), levels=tuple(sorted(set(levels))),
                branches=max(_branches(p) for p in pools), crossings=crossings,
                volume=int((out == water).sum()), input_shape=inp.shape, output_shape=out.shape,
                source_components=len(_components(decoded == water)),
                lateral=bool(np.any((decoded == water).sum(axis=0) != (out == water).sum(axis=0))),
                stencils=tuple(m.tobytes() for m in a['stencils'].values()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, records = set(), set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen
            seen.add(key)
            record = _inspect(pair)
            roles.add(record['roles'])
            records[split].append(record)
    assert len(roles) == 1
    assert records['train'][0]['pools'] == 1
    assert records['train'][1]['pools'] == 1 and records['train'][1]['branches'] >= 2
    assert records['train'][1]['crossings'] >= 2
    assert records['train'][1]['lateral']
    first = _parse(puzzle['train'][0]['input'])
    first_out = np.asarray(puzzle['train'][0]['output'])
    terrain = (first['decoded'] != first['background']) & (first['decoded'] != first['water'])
    surface = int(np.argwhere(first_out == first['water'])[:, 0].min())
    assert np.any(~terrain & (first_out != first['water']) &
                  (np.arange(first_out.shape[0])[:, None] > surface))
    assert any(not np.array_equal(m, np.fliplr(m)) for m in first['stencils'].values())
    assert sorted(first['stencil_boxes'], key=first['stencil_boxes'].get) != sorted(first['key_boxes'], key=first['key_boxes'].get)
    for r in records['train']:
        assert len(r['levels']) == 1
    for r in records['test']:
        assert r['pools'] >= 2 and len(r['levels']) >= 2


def self_test():
    """Bounded, streaming 42-seed checks; returns small reproducible metrics."""
    shapes, outputs, stencils, structures, signatures = set(), set(), set(), set(), set()
    ns, blocks, categories, volumes, roles = set(), set(), set(), set(), set()
    counters = dict(pairs=0, multiple_pools=0, multiple_levels=0, lateral=0,
                    crossing_tiles=0, branched=0, disconnected_sources=0)
    recolor_rng = np.random.default_rng(20261023)
    for seed in VERIFICATION_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _parse(pair['input'])
                stat = _inspect(pair)
                for rev in (False, True):
                    for order in (False, True):
                        assert np.array_equal(_pour(a['decoded'], a['background'], a['water'], rev, order), pair['output'])
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(stat['input_shape'])
                outputs.add(stat['output_shape'])
                stencils.update((stat['n'], m) for m in stat['stencils'])
                ns.add(stat['n']); blocks.add(stat['blocks']); categories.add(stat['categories'])
                volumes.add(stat['volume']); roles.add(stat['roles'])
                normalized = np.where(a['board'] == a['background'], 0,
                                      np.where(a['board'] == a['water'], 2, 1)).astype(np.int8)
                structures.add(_key(normalized))
                counters['pairs'] += 1
                counters['multiple_pools'] += stat['pools'] > 1
                counters['multiple_levels'] += len(stat['levels']) > 1
                counters['lateral'] += stat['lateral']
                counters['crossing_tiles'] += stat['crossings'] > 1
                counters['branched'] += stat['branches'] > 1
                counters['disconnected_sources'] += stat['source_components'] > 1
        signatures.add(digest.hexdigest())
        permutation = recolor_rng.permutation(10)
        for pair in p['test']:
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(signatures) == 42
    assert ns == {3, 4, 5, 6} and categories == {1, 2, 3} and len(blocks) >= 3
    assert len(shapes) >= 20 and len(outputs) >= 30 and len(stencils) >= 60
    assert len(structures) >= 150 and len(volumes) >= 30 and len(roles) >= 30
    assert counters['multiple_levels'] >= 42 and counters['crossing_tiles'] >= 42
    assert counters['branched'] >= 42 and counters['lateral'] >= 100
    # Reproduction, all 30 legal count combinations, and representative type errors.
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(1000+10*train+test, train, test)
            assert len(p['train']) == train and len(p['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (2.5, 1), (True, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                output_shapes=len(outputs), stencil_profiles=len(stencils),
                normalized_riverbeds=len(structures), sizes=sorted(ns), block_counts=sorted(blocks),
                category_counts=sorted(categories), distinct_volumes=len(volumes), **counters)
