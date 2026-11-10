"""Hollow relays: topology-controlled, aperture-projecting beam propagation.

Standard library + NumPy. All parsing and propagation use input pixels only.
No files, parent imports, latent conventions, or persistent caches are used.
"""
from collections import deque
import hashlib
import json
import numpy as np

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        todo = [(r, c)]
        seen[r, c] = True
        cells = []
        while todo:
            x, y = todo.pop()
            cells.append((x, y))
            for dr, dc in DIRECTIONS:
                xx, yy = x + dr, y + dc
                if 0 <= xx < h and 0 <= yy < w and mask[xx, yy] and not seen[xx, yy]:
                    seen[xx, yy] = True
                    todo.append((xx, yy))
        result.append(cells)
    return result


def _shape(cells):
    rc = np.asarray(cells, dtype=int)
    r, c = rc.min(axis=0)
    b, e = rc.max(axis=0) + 1
    m = np.zeros((b-r, e-c), dtype=bool)
    m[tuple((rc - (r, c)).T)] = True
    return m, (int(r), int(c))


def _holes(mask):
    """Exterior padding implements four-connectivity, even at canvas edges."""
    p = np.pad(np.asarray(mask, dtype=bool), 1)
    return [[(r-1, c-1) for r, c in cells] for cells in _components(~p)
            if all(r > 0 and c > 0 and r < p.shape[0]-1 and c < p.shape[1]-1
                   for r, c in cells)]


def _trim(mask):
    return _shape([tuple(p) for p in np.argwhere(mask)])[0]


def _decode(input_grid):
    grid = _grid(input_grid)
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "missing top frame")
    bottoms = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom frame")
    bottom = int(bottoms[0])
    _require(bottom >= 5 and bottom < len(grid)-1, "invalid instruction band")
    band = grid[1:bottom]
    walls = np.all(band == frame, axis=0)
    _require(walls[0] and walls[-1], "missing side walls")
    spans = []
    c = 0
    while c < grid.shape[1]:
        if walls[c]:
            c += 1
            continue
        a = c
        while c < grid.shape[1] and not walls[c]:
            c += 1
        spans.append((a, c))
    _require(len(spans) >= 2, "need at least two instruction panels")
    bg = int(band[0, spans[0][0]])
    _require(bg != frame, "invalid background")
    mapping, entries = {}, []
    class_colors = set()
    for a, b in spans:
        panel = band[:, a:b]
        _require(b-a >= 4 and np.all(panel[0] == bg) and np.all(panel[-1] == bg),
                 "panels need blank top and bottom margins")
        colors = set(map(int, np.unique(panel))) - {bg}
        _require(len(colors) == 1, "panel must contain one monochrome prototype")
        color = colors.pop()
        _require(color != frame and color not in class_colors, "conflicting class colors")
        comps = _components(panel == color)
        _require(len(comps) == 1, "prototype must be connected")
        mask, (r, c) = _shape(comps[0])
        left, right = c == 0, c + mask.shape[1] == panel.shape[1]
        _require(left != right and panel.shape[1]-mask.shape[1] >= 2,
                 "prototype must be flush to one side with two blank columns opposite")
        cavities = _holes(mask)
        count = len(cavities)
        _require(count not in mapping, "duplicate topology in key")
        _require(all(panel[r+x, c+y] == bg for hole in cavities for x, y in hole),
                 "foreign pixels inside prototype")
        turn = -1 if left else 1
        mapping[count] = (color, turn)
        class_colors.add(color)
        entries.append((count, color, turn, mask.shape))
    field = grid[bottom+1:].copy()
    others = set(map(int, np.unique(field))) - {bg}
    _require(len(others) == 2 and not (others & (class_colors | {frame})),
             "field needs background, one source color, and one neutral color")
    candidates = []
    for color in others:
        cells = np.argwhere(field == color)
        r, c = cells.min(axis=0)
        b, e = cells.max(axis=0) + 1
        if b-r == e-c and len(cells) == (b-r)*(e-c):
            candidates.append((color, int(r), int(c), int(b-r)))
    _require(len(candidates) == 1, "source must be the unique solid-square color")
    source, sr, sc, k = candidates[0]
    neutral = (others - {source}).pop()
    labels = np.full(field.shape, -1, dtype=int)
    objects = []
    for i, cells in enumerate(_components(field == neutral)):
        mask, (r, c) = _shape(cells)
        cavities = [[(r+x, c+y) for x, y in hole] for hole in _holes(mask)]
        _require(all(field[x, y] == bg for hole in cavities for x, y in hole),
                 "cavities must contain only background")
        labels[tuple(np.asarray(cells).T)] = i
        objects.append(dict(cells=cells, mask=mask, box=(r, r+mask.shape[0], c, c+mask.shape[1]),
                            holes=cavities, count=len(cavities)))
    _require(len(objects) >= 2, "need separated neutral objects")
    boxes = [q['box'] for q in objects] + [(sr, sr+k, sc, sc+k)]
    _require(all(not (a < bb and aa < b and c < ee and cc < e)
                 for i, (a, b, c, e) in enumerate(boxes)
                 for aa, bb, cc, ee in boxes[:i]), 'overlapping object bounding boxes')
    return dict(field=field, background=bg, source=source, neutral=neutral, frame=frame,
                source_box=(sr, sc, k), mapping=mapping, entries=entries,
                labels=labels, objects=objects, crop=bottom+1)


def _trace(info, mode="normal", reverse=False):
    """Finite propagation closure; queue order cannot change any result.

    Ray states are position + heading. A listed relay consumes a ray and emits
    the directional projection of ALL its original cavities. Repeated states
    need not be expanded again. No execution traces are retained on disk.
    Counterfactual modes exist only for demonstration checks, never for solve.
    """
    f = info['field']
    h, w = f.shape
    sr, sc, k = info['source_box']
    paint = np.zeros(f.shape, dtype=bool)
    reached, events, powered = set(), [], set()
    visited = np.zeros((h, w, 4), dtype=bool)
    queue = deque()
    for j in range(k):
        queue.extend(((sr-1, sc+j, 0), (sr+j, sc+k, 1),
                      (sr+k, sc+j, 2), (sr+j, sc-1, 3)))
    while queue:
        r, c, d = queue.pop() if reverse else queue.popleft()
        while 0 <= r < h and 0 <= c < w:
            if visited[r, c, d]:
                break
            visited[r, c, d] = True
            oid = int(info['labels'][r, c])
            if oid >= 0:
                obj = info['objects'][oid]
                reached.add(oid)
                count = obj['count']
                if mode == 'no_topology':
                    count = min(info['mapping'])
                if count in info['mapping']:
                    color, turn = info['mapping'][count]
                    if mode == 'straight':
                        dr, dc = DIRECTIONS[d]
                        r, c = r+dr, c+dc
                        continue
                    if (oid, d) in powered:
                        break
                    powered.add((oid, d))
                    events.append((oid, d, turn))
                    nd = (d+turn) % 4
                    a, b, e, g = obj['box']
                    holes = obj['holes']
                    for hole in holes:
                        for x, y in hole:
                            paint[x, y] = True
                    pixels = [p for hole in holes for p in hole]
                    lanes = sorted({p[1] if nd % 2 == 0 else p[0] for p in pixels})
                    if mode == 'one_lane' and lanes:
                        lanes = [lanes[len(lanes)//2]]
                    for lane in lanes:
                        # Union the projections from every cavity pixel in
                        # this lane, skipping the relay's own material.
                        vals = [p[0] if nd % 2 == 0 else p[1] for p in pixels
                                if (p[1] if nd % 2 == 0 else p[0]) == lane]
                        if nd == 0:
                            paint[a:max(vals)+1, lane] = True
                            queue.append((a-1, lane, nd))
                        elif nd == 2:
                            paint[min(vals):b, lane] = True
                            queue.append((b, lane, nd))
                        elif nd == 1:
                            paint[lane, min(vals):g] = True
                            queue.append((lane, g, nd))
                        else:
                            paint[lane, e:max(vals)+1] = True
                            queue.append((lane, e-1, nd))
                    break
                # Unlisted objects dissolve; a ray keeps its heading and may
                # paint its corridor through the erased material.
            paint[r, c] = True
            dr, dc = DIRECTIONS[d]
            r, c = r+dr, c+dc
    output = f.copy()
    for i in reached:
        obj = info['objects'][i]
        count = min(info['mapping']) if mode == 'no_topology' else obj['count']
        color = info['mapping'].get(count, (info['background'], 0))[0]
        output[tuple(np.asarray(obj['cells']).T)] = color
    paintable = (f == info['background']) | (f == info['source'])
    for i in reached:
        obj = info['objects'][i]
        count = min(info['mapping']) if mode == 'no_topology' else obj['count']
        if count not in info['mapping']:
            paintable[tuple(np.asarray(obj['cells']).T)] = True
    output[paint & paintable] = info['source']
    return output, dict(reached=reached, events=events, paint=paint,
                        states=int(visited.sum()), powered=powered)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve all valid family inputs, using only their visible pixels."""
    return _trace(_decode(input_grid))[0]


def _prototype(count):
    if count == 0:
        return np.ones((2, 2), dtype=bool)
    rows = 1 if count <= 2 else 2
    cols = (count+rows-1)//rows
    mask = np.ones((2*rows+1, 2*cols+1), dtype=bool)
    for i in range(count):
        mask[1+2*(i//cols), 1+2*(i % cols)] = False
    return mask


def _object(count, rng, large=False):
    """Variable unequal grown cavities and topology-preserving open contours."""
    if count == 0:
        h, w = map(int, rng.integers(2, 6, size=2))
        m = np.ones((h, w), dtype=bool)
        style = int(rng.integers(3))
        if style == 1:
            m[:-1, 1:-1] = False  # open U, not an enclosed cavity
        elif style == 2:
            m[:] = False
            m[h//2, :] = True
            m[:, w//2] = True
        return np.rot90(m, int(rng.integers(4))).copy()
    rows = 1 if count == 1 or (count == 2 and rng.random() < .5) else 2
    cols = (count+rows-1)//rows
    max_tile = 3 if large else 2
    hs = [int(rng.integers(1, max_tile+1)) for _ in range(rows)]
    ws = [int(rng.integers(1, max_tile+1)) for _ in range(cols)]
    m = np.ones((sum(hs)+rows+1, sum(ws)+cols+1), dtype=bool)
    slots = [(i, j) for i in range(rows) for j in range(cols)]
    rng.shuffle(slots)
    for i, j in slots[:count]:
        r, c = 1+sum(hs[:i])+i, 1+sum(ws[:j])+j
        hh, ww = hs[i], ws[j]
        start = (int(rng.integers(hh)), int(rng.integers(ww)))
        cells = {start}
        target = int(rng.integers(1, hh*ww+1))
        for _ in range(target-1):
            frontier = sorted({(x+dr, y+dc) for x, y in cells for dr, dc in DIRECTIONS
                               if 0 <= x+dr < hh and 0 <= y+dc < ww} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
        cavity_mask = np.zeros((hh, ww), dtype=bool)
        cavity_mask[tuple(np.asarray(sorted(cells)).T)] = True
        for island in _holes(cavity_mask):
            cells.update(island)
        for x, y in cells:
            m[r+x, c+y] = False
    # Small cuts create notches, diagonal escapes, irregular walls and tails.
    for _ in range(5):
        trial = m.copy()
        side = int(rng.integers(4))
        x, y = int(rng.integers(m.shape[0])), int(rng.integers(m.shape[1]))
        depth = int(rng.integers(1, 3))
        if side == 0:
            trial[:depth, y] = False
        elif side == 1:
            trial[x, -depth:] = False
        elif side == 2:
            trial[-depth:, y] = False
        else:
            trial[x, :depth] = False
        if trial.any() and len(_components(trial)) == 1 and len(_holes(trial)) == count:
            m = trial
    m = _trim(np.rot90(m, int(rng.integers(4))))
    _require(len(_components(m)) == 1 and len(_holes(m)) == count, "constructor topology")
    return m


def _header(rng, width, mapping):
    keys = list(mapping)
    rng.shuffle(keys)
    glyphs = [_prototype(count) for count in keys]
    sizes = [g.shape[1]+4 for g in glyphs]  # walls + two opposite blank columns
    _require(sum(sizes) <= width, "key does not fit")
    extra = rng.multinomial(width-sum(sizes), np.ones(len(sizes))/len(sizes))
    sizes = [s+int(x) for s, x in zip(sizes, extra)]
    h = max(g.shape[0] for g in glyphs)+4
    header = np.zeros((h, width), dtype=np.int8)
    header[[0, -1]] = 1
    c = 0
    for count, glyph, size in zip(keys, glyphs, sizes):
        color, turn = mapping[count]
        header[:, c] = header[:, c+size-1] = 1
        x = int(rng.integers(2, h-glyph.shape[0]-1))
        y = c+1 if turn == -1 else c+size-1-glyph.shape[1]
        header[x:x+glyph.shape[0], y:y+glyph.shape[1]][glyph] = color
        c += size
    return header


def _place(field, free, mask, rng):
    h, w = mask.shape
    if h > free.shape[0] or w > free.shape[1]:
        return False
    ii = np.pad((~free).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    positions = np.argwhere(totals == 0)
    if not len(positions):
        return False
    if rng.random() < .30:
        edges = positions[(positions[:, 0] == 0) | (positions[:, 1] == 0)
                          | (positions[:, 0]+h == free.shape[0])
                          | (positions[:, 1]+w == free.shape[1])]
        if len(edges):
            positions = edges
    r, c = map(int, positions[int(rng.integers(len(positions)))])
    field[r:r+h, c:c+w][mask] = 2
    free[max(0, r-1):r+h+1, max(0, c-1):c+w+1] = False
    return True


def _random_pair(rng, test, regime):
    positives = [1, 2, 3, 4] if test and regime == 'higher_counts' else [1, 2, 3]
    chosen = ([4, int(rng.integers(1, 4))] if test and regime == 'higher_counts'
              else list(map(int, rng.choice(positives, size=2, replace=False))))
    keys = chosen[:]
    if rng.random() < .32:
        keys.append(0)
    mapping = {count: (4+i, -1 if i == 0 else 1) for i, count in enumerate(keys)}
    if rng.random() < .5:
        mapping = {count: (color, -turn) for count, (color, turn) in mapping.items()}
    minwidth = sum(_prototype(q).shape[1]+4 for q in keys)
    w = int(rng.integers(max(21, minwidth), 31))
    header = _header(rng, w, mapping)
    h = int(rng.integers(16, 31-len(header)))
    large = (test if regime == 'wider_apertures' else rng.random() < .30)
    k = 3 if test and regime == 'source_scale' else int(rng.choice([1, 2]))
    field = np.zeros((h, w), dtype=np.int8)
    sr, sc = int(rng.integers(h-k+1)), int(rng.integers(w-k+1))
    field[sr:sr+k, sc:sc+k] = 3
    free = np.ones(field.shape, dtype=bool)
    free[max(0, sr-1):sr+k+1, max(0, sc-1):sc+k+1] = False
    absent = [q for q in range(6) if q not in keys]
    counts = chosen + [int(rng.choice(absent)), chosen[int(rng.integers(2))]]
    target = int(rng.integers(6 if test and regime == 'recombine' else 5, 10))
    while len(counts) < target:
        counts.append(int(rng.choice(keys+absent)))
    placed = 0
    for count in counts:
        mask = _object(count, rng, large=large)
        if not _place(field, free, mask, rng):
            if not _place(field, free, _prototype(count), rng):
                continue
        placed += 1
    if placed < 4:
        return None
    palette = rng.permutation(10)
    inp = palette[np.vstack((header, field))].astype(np.int8)
    info = _decode(inp)
    out, traced = _trace(info)
    if regime == 'recombine' and not test and _arrival_max(traced) > 1:
        return None
    return dict(input=inp, output=out)


def _lesson(rng, ordinal, source_width=None):
    """Guaranteed finite fallback with an off-axis chain and an unseen twin."""
    field = np.zeros((19, 27), dtype=np.int8)
    k = 1 + ordinal % 2 if source_width is None else source_width
    field[10:10+k, 3:3+k] = 3
    one = np.ones((3, 4), dtype=bool)
    one[1, 1:3] = False
    two = _prototype(2)
    for mask, r, c in ((one, 9, 10), (two, 2, 10), (two, 14, 1), (one, 13, 20)):
        field[r:r+mask.shape[0], c:c+mask.shape[1]][mask] = 2
    field[6:8, 2:5] = 2  # unlisted solid dissolves on the north route
    if ordinal % 2:
        field = np.rot90(field, 2).copy()
    mapping = {1: (4, -1), 2: (5, 1)}
    header = _header(rng, 27, mapping)
    palette = rng.permutation(10)
    inp = palette[np.vstack((header, field))].astype(np.int8)
    return dict(input=inp, output=solve(inp))


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique_lesson(rng, ordinal, seen, source_width=None):
    """Twenty distinct legal renamings guarantee space against <=13 inputs."""
    pair = _lesson(rng, ordinal, source_width=source_width)
    info = _decode(pair['input'])
    for swap in (False, True):
        palette = np.arange(10)
        if swap:
            a, b = info['background'], info['frame']
            palette[a], palette[b] = b, a
        for shift in range(10):
            p = (palette+shift) % 10
            candidate = {name: p[value].astype(np.int8) for name, value in pair.items()}
            if _signature(candidate['input']) not in seen:
                return candidate
    raise AssertionError('twenty distinct lesson palettes cannot all be occupied')


def _features(pair):
    info = _decode(pair['input'])
    expected, trace = _trace(info)
    objects, reached = info['objects'], trace['reached']
    headings = {s: {d for _, d, turn in trace['events'] if turn == s} for s in (-1, 1)}
    listed = {i for i, obj in enumerate(objects) if obj['count'] in info['mapping']}
    # The incoming heading and the outgoing cavity lanes are measured from input.
    split = any(len({p[1] if (d+turn) % 2 == 0 else p[0]
                     for hole in objects[i]['holes'] for p in hole}) > 1
                for i, d, turn in trace['events'])
    sr, sc, k = info['source_box']
    off_axis = any(not (a <= sr+k-1 and b > sr or e <= sc+k-1 and g > sc)
                   for i in reached for a, b, e, g in [objects[i]['box']])
    twin = any(objects[a]['count'] == objects[b]['count']
               for a in reached & listed for b in listed-reached)
    flags = set()
    if reached-listed:
        flags.add('dissolve')
    if twin:
        flags.add('unreached_twin')
    if off_axis:
        flags.add('off_axis')
    if split:
        flags.add('fan')
    for mode in ('no_topology', 'straight', 'one_lane'):
        if not np.array_equal(_trace(info, mode)[0], expected):
            flags.add(mode)
    area_info = dict(info, objects=[dict(obj, count=sum(map(len, obj['holes']))) for obj in objects])
    if not np.array_equal(_trace(area_info)[0], expected):
        flags.add('hole_area')
    return info, trace, dict(headings=headings, flags=flags)


def _arrival_max(trace):
    by_object = {}
    for oid, direction, _ in trace['events']:
        by_object.setdefault(oid, set()).add(direction)
    return max(map(len, by_object.values()), default=0)


def _aperture_span(info, trace):
    spans = []
    for oid, direction, turn in trace['events']:
        axis = 1 if (direction+turn) % 2 == 0 else 0
        for hole in info['objects'][oid]['holes']:
            spans.append(max(p[axis] for p in hole)-min(p[axis] for p in hole)+1)
    return max(spans, default=0)


def _shift_holds(info, trace, regime):
    if regime == 'higher_counts':
        return 4 in info['mapping'] and any(info['objects'][i]['count'] == 4 for i in trace['reached'])
    if regime == 'wider_apertures':
        return _aperture_span(info, trace) >= 3
    if regime == 'source_scale':
        return info['source_box'][2] == 3
    return _arrival_max(trace) >= 2


_REQUIRED_FLAGS = {'dissolve', 'unreached_twin', 'off_axis', 'fan',
                   'no_topology', 'straight', 'one_lane', 'hole_area'}


def _evidence(features):
    flags = set().union(*(q['flags'] for q in features)) if features else set()
    heads = {s: set().union(*(q['headings'][s] for q in features)) if features else set()
             for s in (-1, 1)}
    score = len(flags) + sum(min(2, len(heads[s])) for s in heads)
    return score, _REQUIRED_FLAGS <= flags and all(len(v) >= 2 for v in heads.values())


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['higher_counts', 'wider_apertures', 'source_scale', 'recombine']))
    train, seen, evidence = [], set(), []
    for i in range(num_train):
        best = None
        for _ in range(36 if i < 2 else 1):
            pair = _random_pair(rng, False, regime)
            if pair is None or _signature(pair['input']) in seen:
                continue
            _, _, features = _features(pair)
            score, ready = _evidence(evidence+[features])
            if best is None or score > best[0]:
                best = (score, pair, features)
            if ready or (i == 0 and score >= len(_REQUIRED_FLAGS)+3) or i >= 2:
                break
        if best is None:
            pair = _unique_lesson(rng, i, seen)
            best = (0, pair, _features(pair)[2])
        _, pair, features = best
        train.append(pair)
        evidence.append(features)
        seen.add(_signature(pair['input']))
    if not _evidence(evidence)[1]:
        # Two complementary lessons establish all mechanisms for even 2/1.
        occupied = {_signature(pair['input']) for pair in train[2:]}
        for i in range(2):
            train[i] = _unique_lesson(rng, i, occupied)
            occupied.add(_signature(train[i]['input']))
        seen = {_signature(pair['input']) for pair in train}
    rng.shuffle(train)
    test = []
    for i in range(num_test):
        for _ in range(24):
            pair = _random_pair(rng, True, regime)
            if pair is not None and _signature(pair['input']) not in seen:
                info, trace, _ = _features(pair)
                if trace['events'] and _shift_holds(info, trace, regime):
                    break
        else:
            # If geometric search cannot realize the preferred shift, use the
            # guaranteed visible 1/2 -> 3 source-scale shift, with the same rule.
            pair = _unique_lesson(rng, num_train+i, seen, source_width=3)
        _require(_signature(pair['input']) not in seen, 'unique input retry exhausted')
        test.append(pair)
        seen.add(_signature(pair['input']))
    puzzle = dict(train=train, test=test)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, evidence, training, testing = set(), [], [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            signature = _signature(inp)
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            info, trace, features = _features(pair)
            expected = _trace(info)[0]
            _require(np.array_equal(out, expected), 'wrong optical transformation or crop')
            _require(out.shape == info['field'].shape, 'output must omit the key')
            f, bg, source = info['field'], info['background'], info['source']
            sr, sc, k = info['source_box']
            _require(np.all(out[sr:sr+k, sc:sc+k] == source), 'source changed')
            for i, obj in enumerate(info['objects']):
                ix = tuple(np.asarray(obj['cells']).T)
                if i not in trace['reached']:
                    _require(np.all(out[ix] == info['neutral']), 'unreached object changed')
                elif obj['count'] in info['mapping']:
                    color = info['mapping'][obj['count']][0]
                    _require(np.all(out[ix] == color), 'relay not recolored as a whole')
                    _require(all(out[x, y] == source for hole in obj['holes'] for x, y in hole),
                             'powered cavity not illuminated')
                else:
                    _require(np.all(np.isin(out[ix], [bg, source])), 'unlisted material not dissolved')
            unchanged = (f == bg) & ~trace['paint']
            _require(np.all(out[unchanged] == bg), 'untraversed background changed')
            _require(trace['states'] <= 4*f.size, 'state bound violated')
            if split == 'train':
                evidence.append(features)
                training.append((info, trace))
            else:
                testing.append((info, trace))
    _require(_evidence(evidence)[1], 'demonstrations do not distinguish the required mechanisms')
    taught_keys = set().union(*(set(info['mapping']) for info, _ in training))
    taught_scale = max(info['source_box'][2] for info, _ in training)
    taught_span = max(_aperture_span(info, trace) for info, trace in training)
    taught_arrivals = max(_arrival_max(trace) for _, trace in training)
    for info, trace in testing:
        new_class = any(info['objects'][i]['count'] in set(info['mapping'])-taught_keys
                        for i in trace['reached'])
        _require(new_class or info['source_box'][2] > taught_scale
                 or _aperture_span(info, trace) > taught_span
                 or _arrival_max(trace) > taught_arrivals,
                 'test lacks a visible structural generalization beyond training')


def self_test():
    """42 seeds, counterfactual witnesses, topology fixtures, and bounded checks.

    Holds at most two puzzles; writes no artifacts. This is generator-owned
    validation, not an independent semantic solver or blind prediction.
    """
    diagonal = np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    _require(len(_holes(diagonal)) == 1, 'diagonal leak must stay enclosed')
    diagonal[0, 1] = False
    _require(len(_holes(diagonal)) == 0, 'orthogonal opening must remove cavity')
    # A hand-calculated fixture, not another solver: one ring turns an east
    # beam north at its cavity's column, not at the collision column.
    rng = np.random.default_rng(17)
    field = np.zeros((9, 19), dtype=np.int8)
    field[5, 2] = 3
    ring = _prototype(1)
    for r, c in ((4, 8), (0, 15)):
        field[r:r+3, c:c+3][ring] = 2
    field[2, 2] = 2
    header = _header(rng, 19, {1: (4, -1), 2: (5, 1)})
    fixture = np.vstack((header, field))
    expected = field.copy()
    expected[:, 2] = 3
    expected[5, :8] = 3
    expected[4:7, 8:11][ring] = 4
    expected[5, 9] = 3
    expected[:4, 9] = 3
    _require(np.array_equal(solve(fixture), expected), 'hand-calculated relay fixture failed')
    # A genuine four-relay circuit: the union is finite even though a ray can
    # return to a previously encountered relay from the same heading.
    circuit = np.zeros((19, 21), dtype=np.int8)
    circuit[5, 9] = 3
    for r, c in ((5, 14), (13, 14), (13, 4), (5, 4)):
        circuit[r-1:r+2, c-1:c+2][ring] = 2
    inp = np.vstack((_header(rng, 21, {1: (4, 1), 2: (5, -1)}), circuit))
    decoded = _decode(inp)
    loop_output, loop = _trace(decoded)
    _require(len(loop['reached']) == 4 and len(loop['powered']) >= 5, 'circuit was not traversed')
    _require(np.array_equal(loop_output, _trace(decoded, reverse=True)[0]), 'queue order changed closure')
    _require(loop['states'] <= 4*circuit.size, 'circuit did not terminate within finite state space')
    shapes, masks, populations, keys, widths, turns, colors, structures = (set() for _ in range(8))
    stats = dict(puzzles=42, pairs=0, fan=0, off_axis=0, dissolve=0,
                 unreached_twin=0, irregular_cavities=0)
    digest = hashlib.sha256()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        duplicate = generate_puzzle(seed) if seed in (0, 41) else puzzle
        for split in ('train', 'test'):
            for pair, again in zip(puzzle[split], duplicate[split]):
                _require(all(np.array_equal(pair[n], again[n]) for n in ('input', 'output')), 'nondeterminism')
                info, trace, features = _features(pair)
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), 'list input solve')
                _require(np.array_equal(_trace(info, reverse=True)[0], pair['output']), 'queue-order dependence')
                shapes.add(pair['input'].shape)
                populations.add(len(info['objects']))
                keys.add(tuple(sorted(info['mapping'])))
                widths.add(info['source_box'][2])
                turns.add(tuple(sorted((q, t) for q, (_, t) in info['mapping'].items())))
                colors.add((info['background'], info['neutral'], info['source']))
                roles = {}
                normalized = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat)
                structures.add((pair['input'].shape, normalized))
                stats['pairs'] += 1
                for flag in ('fan', 'off_axis', 'dissolve', 'unreached_twin'):
                    stats[flag] += int(flag in features['flags'])
                for obj in info['objects']:
                    masks.add((obj['mask'].shape, obj['mask'].tobytes()))
                    for hole in obj['holes']:
                        m, _ = _shape(hole)
                        stats['irregular_cavities'] += int(not m.all())
                digest.update(pair['input'].tobytes())
        del duplicate
    _require(len(shapes) >= 20 and len(populations) >= 4 and len(keys) >= 5, 'missing structural variety')
    _require(len(widths) >= 2 and len(colors) >= 80 and len(structures) >= 160, 'missing scale/palette variety')
    _require(len(masks) >= 100 and stats['irregular_cavities'] > 20, 'missing object geometry variety')
    for flag in ('fan', 'off_axis', 'dissolve', 'unreached_twin'):
        _require(stats[flag] >= 20, 'missing interaction coverage: '+flag)
    # Color renaming preserves semantics, and solving never mutates its input.
    pair = generate_puzzle(731, 2, 1)['train'][0]
    saved = pair['input'].copy()
    permutation = np.random.default_rng(999).permutation(10)
    _require(np.array_equal(solve(permutation[saved]), permutation[pair['output']]), 'palette equivariance')
    _require(np.array_equal(pair['input'], saved), 'solve mutated input')
    # Rotation of the field alone preserves relative turns and aperture projection.
    info = _decode(pair['input'])
    rotated = np.vstack((pair['input'][:info['crop']], np.rot90(info['field'], 2)))
    _require(np.array_equal(solve(rotated), np.rot90(pair['output'], 2)), 'relative rotation invariant')
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(83+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(27, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    stats.update(shapes=len(shapes), populations=sorted(populations), key_sets=len(keys),
                 source_widths=sorted(widths), turn_tables=len(turns), role_triples=len(colors),
                 object_masks=len(masks), normalized_inputs=len(structures),
                 input_digest=digest.hexdigest())
    return stats


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
