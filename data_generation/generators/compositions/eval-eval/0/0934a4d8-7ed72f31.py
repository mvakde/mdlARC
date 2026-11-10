"""Mosaic links: recover one erased window through a graph of local symmetries.

Only NumPy and the standard library are used. No filesystem access is needed by
generation, validation, solving, or self_test. All semantic roles are read from
the input geometry; literal colors have no fixed meanings.
"""
from collections import Counter, deque
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 294972268, 431178321, 96440603, 836301987, 852244866,
              950194516, 757955947, 156610923, 207267850, 825883248,
              201920582, 19478574, 791316093, 931362857, 344975994,
              250937896, 133281534, 923361342, 906012115, 272169009,
              743859485, 357478617, 998152435, 211332636, 870732880,
              652213340, 400647605, 487330796, 748859192, 309646518,
              784247205, 549937982, 35408347, 455745732, 830450878,
              649759080, 52701291, 343149836, 441336644, 271071672,
              361458706)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, cells = [first], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _ring(box):
    r, s, c, d = box
    return {(y, x) for y in range(r, s+1) for x in range(c, d+1)
            if y in (r, s) or x in (c, d)}


def _local_map(y, x, h, w, kind):
    return ((h-1-y if kind in ('H', 'P') else y),
            (w-1-x if kind in ('V', 'P') else x))


def _read_input(a):
    """Locate frames first, then identify link and query colors by their scope."""
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    assert np.all(border == border[0]), 'An uninterrupted background border is required'
    bg = int(border[0])
    frame_candidates = []
    for color in np.unique(a):
        if color == bg:
            continue
        comps = _components(a == color)
        if not 3 <= len(comps) <= 8:
            continue
        boxes = [_bbox(comp) for comp in comps]
        if any(set(comp) != _ring(box) for comp, box in zip(comps, boxes)):
            continue
        sizes = {(s-r-1, d-c-1) for r, s, c, d in boxes}
        if len(sizes) != 1 or any(n not in (3, 5, 7) for n in next(iter(sizes))):
            continue
        frame_candidates.append((int(color), sorted(boxes)))
    assert len(frame_candidates) == 1, 'Frames must have one geometric interpretation'
    frame, boxes = frame_candidates[0]
    n = len(boxes)
    h, w = boxes[0][1]-boxes[0][0]-1, boxes[0][3]-boxes[0][2]-1
    inside = np.zeros(a.shape, bool)
    claimed = np.zeros(a.shape, bool)
    tiles = []
    for r, s, c, d in boxes:
        assert not claimed[r:s+1, c:d+1].any(), 'Frames overlap'
        claimed[r:s+1, c:d+1] = True
        inside[r+1:s, c+1:d] = True
        tiles.append(a[r+1:s, c+1:d].copy())
    tiles = np.stack(tiles)
    outside_colors = set(map(int, np.unique(a[~claimed]))) - {bg}
    assert len(outside_colors) == 1, 'Only one link color may occur outside frames'
    marker = outside_colors.pop()
    assert marker != frame and not np.any(tiles == marker) and not np.any(tiles == frame)
    assert not np.any(claimed & (a == marker))
    links = []
    centers = [((r+s)//2, (c+d)//2) for r, s, c, d in boxes]
    for cells in _components(a == marker):
        r0, r1, c0, c1 = _bbox(cells)
        if len(cells) == 1:
            kind, ar, ac = 'P', r0, c0
        elif r0 == r1 and len(cells) == c1-c0+1:
            kind, ar, ac = 'H', r0, c0
        elif c0 == c1 and len(cells) == r1-r0+1:
            kind, ar, ac = 'V', r0, c0
        else:
            raise AssertionError('A link must be a point or a straight segment')
        candidates = []
        for i in range(n):
            for j in range(i+1, n):
                r, s, c, d = boxes[i]
                if kind == 'H':
                    reflected = (2*ar-s, 2*ar-r, c, d)
                    eligible = (s < ar < boxes[j][0] or boxes[j][1] < ar < r)
                    eligible &= c < c0 <= c1 < d
                elif kind == 'V':
                    reflected = (r, s, 2*ac-d, 2*ac-c)
                    eligible = (d < ac < boxes[j][2] or boxes[j][3] < ac < c)
                    eligible &= r < r0 <= r1 < s
                else:
                    reflected = (2*ar-s, 2*ar-r, 2*ac-d, 2*ac-c)
                    eligible = not (r <= ar <= s and c <= ac <= d)
                if eligible and reflected == boxes[j]:
                    distance = (centers[i][0]-ar)**2 + (centers[i][1]-ac)**2
                    candidates.append((distance, i, j))
        assert candidates, 'A link has no reflected pair of frames'
        nearest = min(d for d, i, j in candidates)
        candidates = [(i, j) for d, i, j in candidates if d == nearest]
        assert len(candidates) == 1, 'Nearest opposing frame pairs tie'
        i, j = candidates[0]
        assert not any({i, j} == {u, v} for u, v, k in links), 'Duplicate link'
        links.append((i, j, kind))
    assert links
    reached, pending = {0}, [0]
    while pending:
        i = pending.pop()
        for u, v, k in links:
            if i in (u, v):
                j = v if i == u else u
                if j not in reached:
                    reached.add(j)
                    pending.append(j)
    assert len(reached) == n, 'Every tile must belong to the linked network'
    candidates = []
    for color in np.unique(tiles):
        if color == bg:
            continue
        positions = np.argwhere(tiles == color)
        owners = set(map(int, positions[:, 0]))
        if len(owners) != 1:
            continue
        t = next(iter(owners))
        coords = [(int(y), int(x)) for _, y, x in positions]
        r, s, c, d = _bbox(coords)
        if len(coords) == (s-r+1)*(d-c+1) and len(coords) >= 2:
            candidates.append((int(color), t, (r, s, c, d)))
    assert len(candidates) == 1, 'The query must be the unique rectangular interior color'
    mask, target, query = candidates[0]
    palette = set(map(int, np.unique(tiles))) - {bg, mask}
    assert 2 <= len(palette) <= 6
    for color in palette:
        assert np.count_nonzero(np.any(tiles == color, axis=(1, 2))) >= 2
    return {'background': bg, 'frame': frame, 'marker': marker, 'mask': mask,
            'boxes': boxes, 'tiles': tiles, 'links': links,
            'target': target, 'query': query, 'palette': palette}


def _orbits(n, h, w, links, identity=False):
    parent = list(range(n*h*w))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, kind in links:
        for y in range(h):
            for x in range(w):
                yy, xx = (y, x) if identity else _local_map(y, x, h, w, kind)
                u, v = find((i*h+y)*w+x), find((j*h+yy)*w+xx)
                if u != v:
                    parent[max(u, v)] = min(u, v)
    return np.array([find(i) for i in range(n*h*w)]).reshape(n, h, w)


def _complete(record, identity=False):
    tiles = record['tiles']
    ids = _orbits(*tiles.shape, record['links'], identity=identity)
    known = (tiles != record['background']) & (tiles != record['mask'])
    values = {}
    for key, value in zip(ids[known], tiles[known]):
        key, value = int(key), int(value)
        assert key not in values or values[key] == value, 'Visible clues conflict along a symmetry orbit'
        values[key] = value
    assert set(map(int, ids.flat)) <= values.keys(), 'A whole symmetry orbit is missing'
    full = np.array([values[int(k)] for k in ids.flat], dtype=np.int8).reshape(tiles.shape)
    t = record['target']
    r, s, c, d = record['query']
    return full[t, r:s+1, c:d+1].copy(), full, ids


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return only the erased window, in the target tile's coordinate order."""
    return _complete(_read_input(_grid(input_grid)))[0]


def _distances(record):
    """Shortest number of links from each cell to an originally visible clue."""
    tiles = record['tiles']
    n, h, w = tiles.shape
    known = (tiles != record['background']) & (tiles != record['mask'])
    dist = np.full(tiles.shape, -1, dtype=np.int16)
    dist[known] = 0
    todo = deque(map(tuple, np.argwhere(known).tolist()))
    adjacency = [[] for _ in range(n)]
    for i, j, kind in record['links']:
        adjacency[i].append((j, kind))
        adjacency[j].append((i, kind))
    while todo:
        t, y, x = todo.popleft()
        for j, kind in adjacency[t]:
            yy, xx = _local_map(y, x, h, w, kind)
            if dist[j, yy, xx] < 0:
                dist[j, yy, xx] = dist[t, y, x]+1
                todo.append((j, yy, xx))
    return dist


def _unaligned_shortcut(record):
    """A concrete ablation: fill from nearest visible cell at the same tile offset."""
    tiles = record['tiles']
    target = record['target']
    n = len(tiles)
    hops = {target: 0}
    todo = deque([target])
    while todo:
        i = todo.popleft()
        for u, v, k in record['links']:
            if i in (u, v):
                j = v if u == i else u
                if j not in hops:
                    hops[j] = hops[i]+1
                    todo.append(j)
    order = sorted(range(n), key=lambda i: (hops[i], i))
    r, s, c, d = record['query']
    out = np.empty((s-r+1, d-c+1), dtype=np.int8)
    fallback = min(record['palette'])
    for y in range(r, s+1):
        for x in range(c, d+1):
            out[y-r, x-c] = next((int(tiles[i, y, x]) for i in order
                if tiles[i, y, x] not in (record['background'], record['mask'])), fallback)
    return out


def _features(record, output=None):
    answer, full, ids = _complete(record)
    n, h, w = full.shape
    t = record['target']
    r, s, c, d = record['query']
    query_ids = set(map(int, ids[t, r:s+1, c:d+1].flat))
    known = (record['tiles'] != record['background']) & (record['tiles'] != record['mask'])
    covering = [len(query_ids & set(map(int, ids[i][known[i]]))) for i in range(n)]
    # Keep a deterministic spanning tree. A closing edge is semantically useful
    # if discarding the remaining edges loses a clue for the requested window.
    groups = [{i} for i in range(n)]
    tree = []
    for i, j, kind in record['links']:
        gi = next(g for g in groups if i in g)
        gj = next(g for g in groups if j in g)
        if gi is not gj:
            tree.append((i, j, kind))
            gi.update(gj)
            groups.remove(gj)
    tree_ids = _orbits(n, h, w, tree)
    tree_known = set(map(int, tree_ids[known]))
    cycle_needed = sum(int(k) not in tree_known for k in tree_ids[t, r:s+1, c:d+1].flat)
    distance = _distances(record)[t, r:s+1, c:d+1]
    assert np.all(distance >= 0)
    # Compare each operation with an unchanged translated copy on observable clues.
    orientation, selectors = set(), set()
    for i, j, kind in record['links']:
        other_kind = ('P' if kind in ('H', 'V') else
                      ('V' if record['boxes'][i][0] == record['boxes'][j][0] else 'H'))
        for y in range(h):
            for x in range(w):
                yy, xx = _local_map(y, x, h, w, kind)
                if known[i, y, x] and known[j, yy, xx] and known[j, y, x]:
                    if record['tiles'][i, y, x] != record['tiles'][j, y, x]:
                        orientation.add(kind)
                wy, wx = _local_map(y, x, h, w, other_kind)
                if known[i, y, x] and known[j, yy, xx] and known[j, wy, wx]:
                    if record['tiles'][i, y, x] != record['tiles'][j, wy, wx]:
                        selectors.add(kind)
    return {'tile_count': n, 'tile_shape': (h, w), 'links': len(record['links']),
            'cycles': len(record['links'])-n+1,
            'cycle_needed_cells': cycle_needed,
            'kinds': tuple(sorted({k for i, j, k in record['links']})),
            'query_shape': answer.shape, 'query_orbits': len(query_ids),
            'max_hops': int(distance.max()), 'multi_hop_cells': int(np.count_nonzero(distance >= 2)),
            'single_donor': max(covering) == len(query_ids),
            'orientation_witnesses': tuple(sorted(orientation)),
            'selector_witnesses': tuple(sorted(selectors)),
            'unaligned_differs': not np.array_equal(answer, _unaligned_shortcut(record)),
            'blank_cells': int(np.count_nonzero(record['tiles'] == record['background'])),
            'palette_size': len(record['palette']),
            'loop_merges': int(n*h*w//n - len(np.unique(ids))) if len(record['links']) >= n else 0}


def _layout(rng, h, w, loop, required=False):
    fh, fw = h+2, w+2
    gap = int(rng.choice([1, 3]))
    choices = [(nr, nc) for nr in range(1, 5) for nc in range(1, 5)
               if nr*nc >= (4 if loop or required else 3)
               and nr*fh+(nr-1)*gap+2 <= 30
               and nc*fw+(nc-1)*gap+2 <= 30
               and (not loop or min(nr, nc) >= 2)]
    if not choices:
        return None
    nr, nc = choices[int(rng.integers(len(choices)))]
    if loop:
        y, x = int(rng.integers(nr-1)), int(rng.integers(nc-1))
        cells = {(y, x), (y+1, x), (y, x+1), (y+1, x+1)}
    else:
        cells = {(int(rng.integers(nr)), int(rng.integers(nc)))}
    count = int(rng.integers(max(len(cells), 4 if required else 3), min(8, nr*nc)+1))
    while len(cells) < count:
        frontier = sorted({(y+dy, x+dx) for y, x in cells
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= y+dy < nr and 0 <= x+dx < nc} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    cells = sorted(cells)
    minr, minc = min(y for y, x in cells), min(x for y, x in cells)
    cells = [(y-minr, x-minc) for y, x in cells]
    extent_h = (max(y for y, x in cells)+1)*(fh+gap)-gap
    extent_w = (max(x for y, x in cells)+1)*(fw+gap)-gap
    height = int(rng.integers(extent_h+2, min(30, extent_h+6)+1))
    width = int(rng.integers(extent_w+2, min(30, extent_w+6)+1))
    top, left = int(rng.integers(1, height-extent_h)), int(rng.integers(1, width-extent_w))
    boxes = [(top+y*(fh+gap), top+y*(fh+gap)+fh-1,
              left+x*(fw+gap), left+x*(fw+gap)+fw-1) for y, x in cells]
    possible = [(i, j) for i in range(len(cells)) for j in range(i+1, len(cells))
                if abs(cells[i][0]-cells[j][0])+abs(cells[i][1]-cells[j][1]) == 1]
    rng.shuffle(possible)
    reached, edges = {0}, []
    while len(reached) < len(cells):
        candidates = [(i, j) for i, j in possible if (i in reached) != (j in reached)]
        i, j = candidates[int(rng.integers(len(candidates)))]
        edges.append((i, j))
        reached.update((i, j))
    for edge in possible:
        if edge not in edges and not required and (loop or rng.random() < .35):
            edges.append(edge)
    forced = {}
    if required:
        vertical = [k for k, (i, j) in enumerate(edges) if cells[i][1] == cells[j][1]]
        horizontal = [k for k, (i, j) in enumerate(edges) if cells[i][0] == cells[j][0]]
        if not vertical or not horizontal:
            return None
        forced[int(rng.choice(vertical))] = 'H'
        forced[int(rng.choice(horizontal))] = 'V'
        others = [k for k in range(len(edges)) if k not in forced]
        forced[int(rng.choice(others))] = 'P'
    links = []
    markers = []
    for edge_index, (i, j) in enumerate(edges):
        ri, si, ci, di = boxes[i]
        rj, sj, cj, dj = boxes[j]
        horizontal = ri == rj
        line_kind = 'V' if horizontal else 'H'
        kind = forced.get(edge_index, 'P' if rng.random() < .45 else line_kind)
        ar, ac = (ri+si+rj+sj)//4, (ci+di+cj+dj)//4
        if kind == 'P':
            positions = [(ar, ac)]
        elif kind == 'H':
            length = int(rng.integers(2, w+1))
            start = int(rng.integers(ci+1, di-length+1))
            positions = [(ar, x) for x in range(start, start+length)]
        else:
            length = int(rng.integers(2, h+1))
            start = int(rng.integers(ri+1, si-length+1))
            positions = [(y, ac) for y in range(start, start+length)]
        links.append((i, j, kind))
        markers.append(positions)
    return (height, width), boxes, links, markers


def _make_pair(rng, parameters, challenge, loop, required_kinds=(), test_shift=False):
    for attempt in range(360):
        h, w = (int(rng.choice([3, 5, 7])), int(rng.choice([3, 5, 7])))
        layout = _layout(rng, h, w, loop, required=bool(required_kinds))
        if layout is None:
            continue
        shape, boxes, links, markers = layout
        if not set(required_kinds) <= {k for i, j, k in links}:
            continue
        n = len(boxes)
        ids = _orbits(n, h, w, links)
        roots = np.unique(ids)
        # A network loop can identify different cells within a tile. Assign one
        # value per resulting orbit, rather than imposing conflicting copies.
        colors = rng.permutation(10) if parameters['vary_roles'] else parameters['colors']
        bg, frame, marker, mask = map(int, colors[:4])
        palette = colors[4:4+int(rng.integers(2, 7))]
        style = int(rng.integers(3))
        if style == 0:
            assigned = rng.choice(palette, size=len(roots))
        elif style == 1:
            weights = rng.dirichlet(np.ones(len(palette)))+.15
            assigned = rng.choice(palette, size=len(roots), p=weights/weights.sum())
        else:
            assigned = np.array([palette[((int(k)//w)//2 + (int(k)%w)//2) % len(palette)]
                                 for k in roots], dtype=np.int8)
            change = rng.random(len(roots)) < .25
            assigned[change] = rng.choice(palette, size=int(change.sum()))
        full = assigned[np.searchsorted(roots, ids)].astype(np.int8)
        target = int(rng.integers(n))
        rh, rw = int(rng.integers(1, h+1)), int(rng.integers(1, w+1))
        if rh*rw < 4:
            continue
        r, c = int(rng.integers(h-rh+1)), int(rng.integers(w-rw+1))
        query_mask = np.zeros(full.shape, bool)
        query_mask[target, r:r+rh, c:c+rw] = True
        visible = rng.random(full.shape) > float(rng.uniform(.22, .54))
        visible[query_mask] = False
        # Intentionally erase the immediate counterparts of one query cell.
        # This couples window recovery to transitive use of the marked links.
        if challenge:
            y, x = r+int(rng.integers(rh)), c+int(rng.integers(rw))
            concealed = {(target, y, x)}
            frontier = set(concealed)
            for depth in range(2 if test_shift else 1):
                next_frontier = set()
                for t, yy, xx in frontier:
                    for i, j, kind in links:
                        if t in (i, j):
                            other = j if i == t else i
                            y2, x2 = _local_map(yy, xx, h, w, kind)
                            next_frontier.add((other, y2, x2))
                frontier = next_frontier-concealed
                concealed.update(next_frontier)
            for position in concealed:
                visible[position] = False
        impossible = False
        for key in roots:
            members = ids == key
            if not np.any(visible & members):
                possible = np.argwhere(members & ~query_mask)
                if not len(possible):
                    impossible = True
                    break
                # Choosing an arbitrary surviving member may undo the intended
                # multi-hop challenge; acceptance measures the actual pixels.
                visible[tuple(possible[int(rng.integers(len(possible)))])] = True
        if impossible:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        for i, box in enumerate(boxes):
            for p in _ring(box):
                inp[p] = frame
            a, b, e, f = box
            inp[a+1:b, e+1:f] = np.where(visible[i], full[i], bg)
        for positions in markers:
            for p in positions:
                inp[p] = marker
        a, b, e, f = boxes[target]
        inp[a+1+r:a+1+r+rh, e+1+c:e+1+c+rw] = mask
        output = full[target, r:r+rh, c:c+rw].copy()
        try:
            record = _read_input(inp)
            predicted, _, _ = _complete(record)
            features = _features(record)
        except AssertionError:
            continue
        if not np.array_equal(predicted, output) or not features['unaligned_differs']:
            continue
        if challenge and (features['max_hops'] < (3 if test_shift else 2) or features['single_donor']):
            continue
        if not test_shift and features['max_hops'] > 2:
            continue
        if loop and (features['loop_merges'] <= 0 or features['cycle_needed_cells'] <= 0):
            continue
        if not set(required_kinds) <= (set(features['orientation_witnesses']) & set(features['selector_witnesses'])):
            continue
        return {'input': inp, 'output': output}
    raise RuntimeError('Bounded mosaic sampling exhausted; no rule-changing fallback is used')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    parameters = {'colors': rng.permutation(10), 'vary_roles': bool(rng.random() < .65)}
    puzzle = {'train': [], 'test': []}
    seen = set()
    # Two demonstrations cover all marker kinds, orientation, fragmented evidence,
    # two-link recovery and a nontrivial cycle. Tests need at least three links.
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            required = ('H', 'V', 'P') if split == 'train' and index == 0 else ()
            challenge = split == 'test' or index == 1
            loop = split == 'train' and index == 1
            for retry in range(6):
                pair = _make_pair(rng, parameters, challenge, loop, required, test_shift=split == 'test')
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('Duplicate input after bounded retries')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.tobytes()
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            record = _read_input(inp)
            predicted, full, ids = _complete(record)
            assert np.array_equal(predicted, out), 'Wrong reconstructed window'
            assert not set(map(int, out.flat)) & {record[k] for k in ('background', 'frame', 'marker', 'mask')}
            # All visible clues survive completion; links constrain every cell,
            # including originally blank cells. The label supplies no inference.
            visible = (record['tiles'] != record['background']) & (record['tiles'] != record['mask'])
            assert np.array_equal(full[visible], record['tiles'][visible])
            for i, j, kind in record['links']:
                mate = full[i][::-1] if kind == 'H' else (full[i][:, ::-1] if kind == 'V' else full[i][::-1, ::-1])
                assert np.array_equal(mate, full[j])
            f = _features(record)
            assert f['blank_cells'] > 0 and f['unaligned_differs']
            if split == 'train':
                train_features.append(f)
            else:
                assert f['max_hops'] >= 3 and not f['single_donor']
    assert {'H', 'V', 'P'} <= set(k for f in train_features for k in f['orientation_witnesses'])
    assert {'H', 'V', 'P'} <= set(k for f in train_features for k in f['selector_witnesses'])
    assert all(f['max_hops'] <= 2 for f in train_features)
    assert any(f['loop_merges'] > 0 and f['cycle_needed_cells'] > 0 for f in train_features)
    assert any(f['max_hops'] >= 2 and not f['single_donor'] for f in train_features)


def self_test():
    """Bounded 42-seed audit; one puzzle at a time, aggregate counters only."""
    shapes, queries, structures, signatures, roles = set(), set(), set(), set(), set()
    counts, kinds, stats = Counter(), Counter(), Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                assert np.array_equal(solve(a), b)
                signature.update(a.tobytes())
                signature.update(b.tobytes())
                record = _read_input(a)
                f = _features(record)
                shapes.add(a.shape)
                queries.add(b.shape)
                mapping = {}
                norm = bytes(mapping.setdefault(int(v), len(mapping)) for v in a.flat)
                structures.add((a.shape, hashlib.sha256(norm).hexdigest()))
                roles.add((record['background'], record['frame'], record['marker'], record['mask']))
                counts[f['tile_count']] += 1
                kinds.update(k for i, j, k in record['links'])
                stats['pairs'] += 1
                stats['multi_hop_pairs'] += f['max_hops'] >= 2
                stats['fragmented_pairs'] += not f['single_donor']
                stats['cyclic_pairs'] += f['cycles'] > 0
                stats['loop_merged_pairs'] += f['loop_merges'] > 0
                stats['cycle_needed_pairs'] += f['cycle_needed_cells'] > 0
                # Local edge checks using array flips do not use DSU labels.
                _, full, _ = _complete(record)
                for i, j, kind in record['links']:
                    expected = np.flip(full[i], axis=0 if kind == 'H' else 1) if kind != 'P' else np.rot90(full[i], 2)
                    assert np.array_equal(expected, full[j])
                # Rotation and arbitrary color bijections preserve the rule.
                if split == 'test':
                    assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
                    permutation = np.roll(np.arange(10, dtype=np.int8), 3)
                    assert np.array_equal(solve(permutation[a]), permutation[b])
        signatures.add(signature.hexdigest())
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
            bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            _must_reject(lambda: validate_generated_puzzle(bad))
            # Erasing every observation must fail before an output is consulted.
            a = puzzle['test'][0]['input'].copy()
            rec = _read_input(a)
            for r, s, c, d in rec['boxes']:
                block = a[r+1:s, c+1:d]
                block[block != rec['mask']] = rec['background']
            _must_reject(lambda: solve(a))
    assert len(signatures) == 42
    assert len(shapes) >= 45 and len(queries) >= 15 and len(structures) >= 200
    assert len(roles) >= 30 and len(counts) >= 5 and set(kinds) == {'H', 'V', 'P'}
    assert stats['multi_hop_pairs'] >= 84 and stats['fragmented_pairs'] >= 84
    assert stats['loop_merged_pairs'] >= 20
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    return {'valid': True, 'seeds': len(HOST_SEEDS), **dict(stats),
            'input_shapes': len(shapes), 'query_shapes': len(queries),
            'normalized_input_structures': len(structures), 'role_palettes': len(roles),
            'tile_counts': dict(sorted(counts.items())), 'link_kinds': dict(kinds)}


def _must_reject(action):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError('Malformed evidence or label was accepted')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
