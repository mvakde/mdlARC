"""Census-constrained jigsaws: topology supplies the missing assembly clues.

Only NumPy and the standard library are used. No parent generator is imported.
Search is complete within explicit bounds; unfinished searches are rejected.
Nothing is written to disk by generation, validation, or self_test.
"""
from __future__ import annotations

import hashlib
import numpy as np


class _SearchLimit(ValueError):
    pass


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "grid dtype")
    _check(np.all((0 <= g) & (g <= 9)), "grid colors")
    return g.astype(np.int8, copy=True)


def _components(mask):
    """Orthogonal connectivity, including for masks with holes or steps."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        cells = [(y, x)]
        seen[y, x] = True
        for r, c in cells:
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    cells.append((rr, cc))
        result.append(cells)
    return result


def _box(cells):
    q = np.asarray(cells)
    r, c = q.min(axis=0)
    y, x = q.max(axis=0) + 1
    return int(r), int(c), int(y-r), int(x-c)


def _perimeter(h, w):
    m = np.zeros((h, w), dtype=bool)
    m[[0, -1], :] = True
    m[:, [0, -1]] = True
    return m


def _plaque(bg, color, count, capacity, vertical=False, reverse=False):
    long = 2*capacity + 3
    m = np.full((5, long), bg, dtype=np.int8)
    m[_perimeter(5, long)] = color
    for i in range(count):
        m[2, long-3-2*i] = color
    if reverse:
        m = m[:, ::-1]
    return m.T.copy() if vertical else m.copy()


def _read_plaque(g, color, cells, bg):
    r, c, h, w = _box(cells)
    if min(h, w) != 5 or max(h, w) < 7 or max(h, w) % 2 != 1:
        return None
    border = {(r+int(y), c+int(x)) for y, x in np.argwhere(_perimeter(h, w))}
    if set(cells) != border:
        return None
    crop = g[r:r+h, c:c+w]
    inside = crop[1:-1, 1:-1]
    if not np.all((inside == bg) | (inside == color)):
        return None
    n = int(np.count_nonzero(inside == color))
    capacity = (max(h, w)-3)//2
    if n > capacity:
        return None
    if not any(np.array_equal(crop, _plaque(bg, color, n, capacity, h > w, rev))
               for rev in (False, True)):
        return None
    return dict(box=(r, c, h, w), color=color, count=n, capacity=capacity)


def _extract(input_grid):
    g = _grid(input_grid)
    hist = np.bincount(g.ravel(), minlength=10)
    _check(np.count_nonzero(hist == hist.max()) == 1, "background must be the unique mode")
    bg = int(hist.argmax())
    plaques = []
    for color in range(10):
        if color == bg:
            continue
        for cells in _components(g == color):
            p = _read_plaque(g, color, cells, bg)
            if p is not None:
                plaques.append(p)
    plaques.sort(key=lambda p: p['color'])
    _check(1 <= len(plaques) <= 3, "expected one to three census plaques")
    colors = tuple(p['color'] for p in plaques)
    _check(len(set(colors)) == len(colors), "repeated census color")
    allowed = np.ones(g.shape, dtype=bool)
    for p in plaques:
        r, c, h, w = p['box']
        _check(allowed[r:r+h, c:c+w].all(), "overlapping plaques")
        allowed[r:r+h, c:c+w] = False
    markers = []
    for color in range(10):
        if color == bg or hist[color] != 3:
            continue
        cells = set(map(tuple, np.argwhere(g == color).tolist()))
        r, c, h, w = _box(list(cells))
        if h != 2 or w != 2 or any(not allowed[p] for p in cells):
            continue
        missing = ({(r, c), (r+1, c), (r, c+1), (r+1, c+1)}-cells).pop()
        if not allowed[missing] or g[missing] in (bg, color):
            continue
        corner = {(1, 1): 'TL', (1, 0): 'TR', (0, 1): 'BL', (0, 0): 'BR'}[
            (missing[0]-r, missing[1]-c)]
        markers.append((color, missing, corner, cells))
    _check(len(markers) == 1, "ambiguous or absent external L marker")
    marker, point, corner, marker_cells = markers[0]
    mask = allowed & (g != bg) & (g != marker)
    pieces = []
    anchor = None
    for cells in _components(mask):
        r, c, h, w = _box(cells)
        m = np.full((h, w), -1, dtype=np.int8)
        for y, x in cells:
            m[y-r, x-c] = g[y, x]
        if point in cells:
            anchor = len(pieces)
        pieces.append((m, (r, c)))
    _check(3 <= len(pieces) <= 6 and anchor is not None, "payload pieces")
    m, (r, c) = pieces[anchor]
    _check(point == (r + (m.shape[0]-1 if corner[0] == 'B' else 0),
                     c + (m.shape[1]-1 if corner[1] == 'R' else 0)), "L does not mark a piece corner")
    _check(m[point[0]-r, point[1]-c] >= 0, "empty anchor corner")
    return dict(grid=g, bg=bg, plaques=plaques, colors=colors, pieces=pieces,
                anchor=anchor, point=point, corner=corner, marker=marker,
                marker_cells=marker_cells)


def _overlap(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r < y+ph and y < r+h and c < x+pw and x < c+w


def _geometries(pieces, anchor, point, corner, canvas, forbidden=()):
    """All unrotated mask tilings, over every area-factor rectangle.

    The first free cell must be the earliest occupied cell of its new piece.
    Results store placements only, not duplicate grid payloads.
    """
    area = sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces)
    results = []
    nodes = 0
    for h in range(1, canvas[0]+1):
        if area % h:
            continue
        w = area//h
        r0 = point[0] - (h-1 if corner[0] == 'B' else 0)
        c0 = point[1] - (w-1 if corner[1] == 'R' else 0)
        if w > canvas[1] or min(r0, c0) < 0 or r0+h > canvas[0] or c0+w > canvas[1]:
            continue
        if any(_overlap((r0, c0, h, w), b) for b in forbidden):
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        configs = []
        for m, _ in pieces:
            coords = [(int(y), int(x)) for y, x in np.argwhere(m >= 0)]
            bits = sum(1 << (y*w+x) for y, x in coords)
            configs.append((m.shape, coords[0][1], bits))
        ar, ac = pieces[anchor][1][0]-r0, pieces[anchor][1][1]-c0
        ah, aw = pieces[anchor][0].shape
        if min(ar, ac) < 0 or ar+ah > h or ac+aw > w:
            continue
        positions = [None]*len(pieces)
        positions[anchor] = (ar, ac)
        full = (1 << area)-1

        def visit(occupied, remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 40000 or len(results) > 600:
                raise _SearchLimit("complete packing enumeration exceeds family budget")
            if not remaining:
                results.append(((h, w), (r0, c0), tuple(positions)))
                return
            free = full ^ occupied
            first = (free & -free).bit_length()-1
            r, c = divmod(first, w)
            for i in remaining:
                (ph, pw), start, bits = configs[i]
                x = c-start
                if x < 0 or r+ph > h or x+pw > w:
                    continue
                placed = bits << (r*w+x)
                if not placed & occupied:
                    positions[i] = (r, x)
                    visit(occupied | placed, tuple(j for j in remaining if j != i))

        visit(configs[anchor][2] << (ar*w+ac), tuple(i for i in range(len(pieces)) if i != anchor))
    return results


def _picture(pieces, geometry):
    shape, _, positions = geometry
    pic = np.full(shape, -1, dtype=np.int8)
    for (m, _), (r, c) in zip(pieces, positions):
        crop = pic[r:r+m.shape[0], c:c+m.shape[1]]
        crop[m >= 0] = m[m >= 0]
    _check(np.all(pic >= 0), "incomplete packing")
    return pic


def _counts(pic, colors):
    return tuple(len(_components(pic == c)) for c in colors)


def _groups(pieces, geometries, colors):
    """Group distinct visible assemblies by their connected-component census."""
    groups = {}
    seen = set()
    for geo in geometries:
        pic = _picture(pieces, geo)
        key = (geo[0], geo[1], pic.tobytes())
        if key in seen:
            continue
        seen.add(key)
        census = _counts(pic, colors)
        groups.setdefault(census, []).append(geo)
    return groups


def _render(detail, geometry):
    out = np.full(detail['grid'].shape, detail['bg'], dtype=np.int8)
    for p in detail['plaques']:
        r, c, h, w = p['box']
        out[r:r+h, c:c+w] = detail['grid'][r:r+h, c:c+w]
    (h, w), (r, c), _ = geometry
    out[r:r+h, c:c+w] = _picture(detail['pieces'], geometry)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only prediction; no RNG, examples, caches, or construction state."""
    d = _extract(input_grid)
    geos = _geometries(d['pieces'], d['anchor'], d['point'], d['corner'],
                       d['grid'].shape, [p['box'] for p in d['plaques']])
    target = tuple(p['count'] for p in d['plaques'])
    valid = _groups(d['pieces'], geos, d['colors']).get(target, [])
    _check(len(valid) == 1, "census must determine exactly one visible assembly")
    return _render(d, valid[0])


def _partition(rng, count, mode):
    if mode == 'strips':
        height = int(rng.integers(4, 8))
        widths = [int(rng.integers(2, 5)) for _ in range(count)]
        labels = np.zeros((height, sum(widths)), dtype=int)
        x = 0
        for i, w in enumerate(widths):
            labels[:, x:x+w] = i
            x += w
    elif mode == 'grid' and count == 4:
        a, b, c, e = [int(rng.integers(2, 5)) for _ in range(4)]
        labels = np.zeros((a+b, c+e), dtype=int)
        labels[:a, c:] = 1
        labels[a:, :c] = 2
        labels[a:, c:] = 3
    else:
        h, w = int(rng.integers(6, 10)), int(rng.integers(7, 12))
        rects = [(0, 0, h, w)]
        for _ in range(count-1):
            options = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, length in enumerate((ph, pw)) if length >= 4]
            i, axis = options[int(rng.integers(len(options)))]
            r, c, ph, pw = rects.pop(i)
            cut = int(rng.integers(2, (ph if axis == 0 else pw)-1))
            if axis == 0:
                rects.extend(((r, c, cut, pw), (r+cut, c, ph-cut, pw)))
            else:
                rects.extend(((r, c, ph, cut), (r, c+cut, ph, pw-cut)))
        labels = np.zeros((h, w), dtype=int)
        for i, (r, c, ph, pw) in enumerate(rects):
            labels[r:r+ph, c:c+pw] = i
    if mode == 'stepped':
        options = []
        for r in range(1, labels.shape[0]-1):
            for c in range(1, labels.shape[1]-1):
                for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                    if labels[r, c] != labels[rr, cc]:
                        options.append((r, c, int(labels[rr, cc])))
        for k in rng.permutation(len(options))[:8]:
            r, c, recipient = options[int(k)]
            donor = int(labels[r, c])
            old = labels.copy()
            labels[r, c] = recipient
            if all(len(_components(labels == i)) == 1 and np.count_nonzero(labels == i) >= 4
                   for i in (donor, recipient)):
                break
            labels = old
    if rng.integers(2):
        labels = labels.T.copy()
    return labels


def _cut(labels, pic):
    pieces = []
    for i in range(int(labels.max())+1):
        r, c, h, w = _box(np.argwhere(labels == i))
        m = np.where(labels[r:r+h, c:c+w] == i, pic[r:r+h, c:c+w], -1)
        pieces.append((m.astype(np.int8), (r, c)))
    return pieces


def _paint(rng, shape, carrier, colors, extra):
    """Regions, walks and bars; no access to the requested census or solution."""
    h, w = shape
    pic = np.full(shape, carrier, dtype=np.int8)
    for color in list(colors) + list(extra):
        for _ in range(int(rng.integers(1, 4))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            kind = int(rng.integers(3))
            if kind == 0:
                ph, pw = int(rng.integers(1, 4)), int(rng.integers(2, 5))
                pic[max(0, r-ph+1):r+1, max(0, c-pw+1):c+1] = color
            elif kind == 1:
                axis = int(rng.integers(2))
                length = int(rng.integers(2, 7))
                if axis:
                    pic[r, max(0, c-length+1):c+1] = color
                else:
                    pic[max(0, r-length+1):r+1, c] = color
            else:
                for _ in range(int(rng.integers(4, 15))):
                    pic[r, c] = color
                    dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
                    r, c = min(h-1, max(0, r+dr)), min(w-1, max(0, c+dc))
    if any(np.count_nonzero(pic == c) < 4 for c in (carrier, *colors, *extra)):
        return None
    return pic


def _place_rect(rng, shape, blocked, *, occupied=None):
    """Free pixel origins; reserve bounding boxes plus a one-cell moat."""
    h, w = shape
    H, W = blocked.shape
    options = [(r, c) for r in range(H-h+1) for c in range(W-w+1)]
    for k in rng.permutation(len(options)):
        r, c = options[int(k)]
        if blocked[r:r+h, c:c+w].any():
            continue
        if occupied is not None and occupied[r:r+h, c:c+w].any():
            continue
        return r, c
    return None


def _reserve(blocked, r, c, h, w):
    H, W = blocked.shape
    blocked[max(0, r-1):min(H, r+h+1), max(0, c-1):min(W, c+w+1)] = True


def _build(rng, count, category_count, rotation, twin=False, allow_steps=True):
    """Construct a bag, then select genuinely unique census interpretations."""
    for attempt in range(160):
        palette = [int(c) for c in rng.permutation(10)]
        bg, marker, carrier = palette[:3]
        colors = tuple(sorted(palette[3:3+category_count]))
        extra = tuple(palette[3+category_count:4+category_count]) if rng.random() < .3 else ()
        modes = ('strips', 'grid', 'recursive', 'stepped') if allow_steps else ('strips', 'grid', 'recursive')
        mode = str(rng.choice(modes))
        labels = _partition(rng, count, mode)
        anchor = int(labels[0, 0])
        minimum = 19 if category_count == 2 else 22
        H, W = int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31))
        point = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
        dummy = _cut(labels, np.full(labels.shape, carrier, dtype=np.int8))
        anchor_local = dummy[anchor][1]
        base_positions = [(point[0]+r, point[1]+c) for _, (r, c) in dummy]
        if anchor_local != (0, 0):
            continue
        dummy = [(m, p) for (m, _), p in zip(dummy, base_positions)]
        try:
            geos = _geometries(dummy, anchor, point, 'TL', (H, W))
        except _SearchLimit:
            continue
        if len(geos) < 2:
            continue
        for _ in range(24):
            pic = _paint(rng, labels.shape, carrier, colors, extra)
            if pic is None:
                continue
            pieces = [(m, p) for (m, _), p in zip(_cut(labels, pic), base_positions)]
            groups = _groups(pieces, geos, colors)
            scattered = tuple(sum(len(_components(m == c)) for m, _ in pieces) for c in colors)
            choices = [(ns, items[0]) for ns, items in groups.items()
                       if len(items) == 1 and all(1 <= n <= 5 for n in ns)
                       and any(n < old for n, old in zip(ns, scattered))]
            if len(choices) < (2 if twin else 1):
                continue
            order = rng.permutation(len(choices))
            chosen = [choices[int(i)] for i in order[:2 if twin else 1]]
            # The teaching alternatives have exactly the same ledger dimensions.
            capacities = [min(7, max(ns[i] for ns, _ in chosen)+int(rng.integers(1, 3)))
                          for i in range(len(colors))]
            final_space = np.zeros((H, W), dtype=bool)
            for _, ((h, w), (r, c), _) in chosen:
                final_space[r:r+h, c:c+w] = True
            blocked = np.zeros((H, W), dtype=bool)
            ar, ac = point
            ah, aw = pieces[anchor][0].shape
            _reserve(blocked, ar-1, ac-1, ah+1, aw+1)
            plaques = []
            for i, color in enumerate(colors):
                vertical, reverse = bool(rng.integers(2)), bool(rng.integers(2))
                plaque = _plaque(bg, color, chosen[0][0][i], capacities[i], vertical, reverse)
                pos = _place_rect(rng, plaque.shape, blocked, occupied=final_space)
                if pos is None:
                    break
                r, c = pos
                plaques.append((r, c, vertical, reverse))
                _reserve(blocked, r, c, *plaque.shape)
            if len(plaques) != len(colors):
                continue
            positions = [None]*len(pieces)
            positions[anchor] = point
            for i in sorted((i for i in range(len(pieces)) if i != anchor),
                            key=lambda j: -pieces[j][0].size):
                pos = _place_rect(rng, pieces[i][0].shape, blocked)
                if pos is None:
                    break
                positions[i] = pos
                _reserve(blocked, *pos, *pieces[i][0].shape)
            if any(pos is None for pos in positions):
                continue
            grid = np.full((H, W), bg, dtype=np.int8)
            for (m, _), (r, c) in zip(pieces, positions):
                crop = grid[r:r+m.shape[0], c:c+m.shape[1]]
                crop[m >= 0] = m[m >= 0]
            grid[ar-1, ac-1] = grid[ar-1, ac] = grid[ar, ac-1] = marker
            pairs = []
            for ns, geo in chosen:
                inp = grid.copy()
                for i, (r, c, vertical, reverse) in enumerate(plaques):
                    plaque = _plaque(bg, colors[i], ns[i], capacities[i], vertical, reverse)
                    inp[r:r+plaque.shape[0], c:c+plaque.shape[1]] = plaque
                expected = np.full_like(inp, bg)
                for i, (r, c, vertical, reverse) in enumerate(plaques):
                    h, w = (2*capacities[i]+3, 5) if vertical else (5, 2*capacities[i]+3)
                    expected[r:r+h, c:c+w] = inp[r:r+h, c:c+w]
                (h, w), (r, c), _ = geo
                expected[r:r+h, c:c+w] = _picture(pieces, geo)
                inp = np.rot90(inp, rotation).copy()
                expected = np.rot90(expected, rotation).copy()
                # Removing geometries blocked by plaques cannot create a tie.
                # Check the actual input still has a census-dependent choice.
                try:
                    _analyze({'input': inp, 'output': expected})
                except (AssertionError, _SearchLimit):
                    break
                pairs.append({'input': inp, 'output': expected})
            if len(pairs) == len(chosen):
                return pairs
    raise RuntimeError("bounded census-jigsaw construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, ne = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(3, 5))
    categories = int(rng.integers(2, 4))
    rotation = int(rng.integers(4))
    twins = _build(rng, count, categories, rotation, twin=True)
    puzzle = {'train': twins, 'test': []}
    seen = {(p['input'].shape, p['input'].tobytes()) for p in twins}
    for split, total in (('train', nt-2), ('test', ne)):
        for _ in range(total):
            for attempt in range(8):
                n = int(rng.integers(3, count+1)) if split == 'train' else count+1
                allow_steps = split == 'train' or any(_analyze(p)['stepped'] for p in puzzle['train'])
                p = _build(rng, n, int(rng.integers(2, 4)), rotation, allow_steps=allow_steps)[0]
                key = (p['input'].shape, p['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(p)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _check(inp.shape == out.shape, "canvas changed")
    d = _extract(inp)
    geos = _geometries(d['pieces'], d['anchor'], d['point'], d['corner'], inp.shape,
                       [p['box'] for p in d['plaques']])
    groups = _groups(d['pieces'], geos, d['colors'])
    ns = tuple(p['count'] for p in d['plaques'])
    valid = groups.get(ns, [])
    _check(len(valid) == 1, "ambiguous census")
    _check(np.array_equal(out, _render(d, valid[0])), "incorrect assembly")
    _check(sum(len(v) for v in groups.values()) >= 2, "counting contributes no placement information")
    scattered = tuple(sum(len(_components(m == c)) for m, _ in d['pieces']) for c in d['colors'])
    _check(any(n < s for n, s in zip(ns, scattered)), "no fragments join across seams")
    _check(not np.any(out == d['marker']), "marker retained")
    _check(all(out[p] == d['bg'] for p in d['marker_cells']), "marker location not erased")
    # Every payload cell and every ledger pixel is preserved; only the marker is removed.
    before, after = np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)
    _check(all(before[c] == after[c] for c in range(10) if c not in (d['bg'], d['marker'])),
           "payload colors or areas changed")
    _check(after[d['bg']] == before[d['bg']]+3, "occupied area changed")
    shape, origin, positions = valid[0]
    old_anchor = d['pieces'][d['anchor']][1]
    new_anchor = positions[d['anchor']]
    _check(old_anchor == (origin[0]+new_anchor[0], origin[1]+new_anchor[1]), "anchor moved")
    masks = tuple((m >= 0).astype(np.uint8).tobytes() for m, _ in d['pieces'])
    return dict(detail=d, census=ns, scattered=scattered, shape=shape,
                pieces=len(d['pieces']), candidates=sum(len(v) for v in groups.values()),
                stepped=sum(np.any(m < 0) for m, _ in d['pieces']),
                areas=tuple(sum(int(np.count_nonzero(m == c)) for m, _ in d['pieces']) for c in d['colors']),
                masks=masks)


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen = set()
    records = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp = _grid(pair['input'])
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            records[split].append(_analyze(pair))
    # Two visible counterfactuals isolate the census from all geometric clues.
    a, b = records['train'][:2]
    da, db = a['detail'], b['detail']
    _check(da['colors'] == db['colors'] and a['census'] != b['census'], "teaching census contrast")
    ga, gb = da['grid'].copy(), db['grid'].copy()
    _check([p['box'] for p in da['plaques']] == [p['box'] for p in db['plaques']], "teaching plaque layout")
    for p in da['plaques']:
        r, c, h, w = p['box']
        ga[r:r+h, c:c+w] = da['bg']
        gb[r:r+h, c:c+w] = db['bg']
    _check(np.array_equal(ga, gb), "teaching tile bags differ")
    _check(a['areas'] == b['areas'], "teaching area contrast missing")
    _check(not np.array_equal(puzzle['train'][0]['output'], puzzle['train'][1]['output']), "teaching outputs identical")
    _check(min(r['pieces'] for r in records['test']) > max(r['pieces'] for r in records['train']),
           "piece-count generalization missing")
    _check(not any(r['stepped'] for r in records['test']) or any(r['stepped'] for r in records['train']),
           "test notches lack a training example")
    return None


def self_test():
    """42 seeds, bounded memory, semantic invariants, and mutation rejection."""
    signatures, shapes, targets, counts, corners, backgrounds, masks, ledgers = (set() for _ in range(8))
    stepped = 0
    ambiguity_checked = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _analyze(pair)
                info = d['detail']
                shapes.add(pair['input'].shape)
                targets.add(d['shape'])
                counts.add(d['pieces'])
                corners.add(info['corner'])
                backgrounds.add(info['bg'])
                masks.update(d['masks'])
                ledgers.update(d['census'])
                stepped += int(d['stepped'] > 0)
                digest.update(repr(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                original = pair['input'].copy()
                _check(np.array_equal(solve(original), pair['output']), "solve mismatch")
                _check(np.array_equal(original, pair['input']), "solve mutates input")
                if not ambiguity_checked:
                    geos = _geometries(info['pieces'], info['anchor'], info['point'], info['corner'],
                                       original.shape, [p['box'] for p in info['plaques']])
                    groups = _groups(info['pieces'], geos, info['colors'])
                    for ns, choices in groups.items():
                        if len(choices) < 2 or any(n > p['capacity'] for n, p in zip(ns, info['plaques'])):
                            continue
                        ambiguous = original.copy()
                        for n, p in zip(ns, info['plaques']):
                            r, c, h, w = p['box']
                            ambiguous[r:r+h, c:c+w] = _plaque(info['bg'], p['color'], n, p['capacity'], h > w)
                        try:
                            solve(ambiguous)
                        except AssertionError:
                            ambiguity_checked = True
                        else:
                            raise AssertionError("ambiguous census accepted")
                        break
        signatures.add(digest.hexdigest())
    _check(len(signatures) == 42, "puzzle diversity")
    _check(len(shapes) >= 25 and len(targets) >= 15, "geometry diversity")
    _check(counts == {3, 4, 5}, "piece-count spectrum")
    _check(corners == {'TL', 'TR', 'BL', 'BR'}, "corner spectrum")
    _check(len(backgrounds) >= 8 and len(masks) >= 30, "palette and partition spectrum")
    _check(len(ledgers) >= 3 and stepped >= 3, "census and stepped variation")
    _check(ambiguity_checked, "ambiguous-census regression not exercised")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(91, nt, ne)
            _check(len(puzzle['train']) == nt and len(puzzle['test']) == ne, "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(0), generate_puzzle(0)
    _check(all(np.array_equal(a[k], b[k]) for s in first for a, b in zip(first[s], second[s])
               for k in ('input', 'output')), "determinism")
    # Search decisions must not depend on the order in which tiles were segmented.
    d = _extract(first['test'][0]['input'])
    count = len(d['pieces'])
    reordered = d.copy()
    reordered['pieces'] = list(reversed(d['pieces']))
    reordered['anchor'] = count-1-d['anchor']
    geos = _geometries(reordered['pieces'], reordered['anchor'], d['point'], d['corner'],
                       d['grid'].shape, [p['box'] for p in d['plaques']])
    winners = _groups(reordered['pieces'], geos, d['colors'])[tuple(p['count'] for p in d['plaques'])]
    _check(len(winners) == 1 and np.array_equal(_render(reordered, winners[0]), first['test'][0]['output']),
           "tile-order dependence")
    for mutation in ('identity', 'marker', 'shift', 'duplicate'):
        p = {s: [{k: v.copy() for k, v in pair.items()} for pair in first[s]] for s in first}
        if mutation == 'identity':
            p['test'][0]['output'] = p['test'][0]['input'].copy()
        elif mutation == 'marker':
            d = _extract(p['test'][0]['input'])
            p['test'][0]['output'][next(iter(d['marker_cells']))] = d['marker']
        elif mutation == 'shift':
            p['test'][0]['output'] = np.roll(p['test'][0]['output'], 1, axis=0)
        else:
            p['train'][1] = {k: v.copy() for k, v in p['train'][0].items()}
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("semantic corruption accepted: " + mutation)


if __name__ == '__main__':
    self_test()
