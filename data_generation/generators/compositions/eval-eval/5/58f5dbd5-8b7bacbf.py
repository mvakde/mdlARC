"""Stencil-fed capillaries: subtract a mask, then race paints inside its holes.

All interpretation is input-derived. Layout and curriculum affect sampling only.
No parent generator, evidence file, seed, or stored example is used by solve.
"""
from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _components(mask, steps=D4):
    labels = np.full(mask.shape, -1, dtype=int)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(groups)
        cells = [(r, c)]
        labels[r, c] = index
        for y, x in cells:
            for dy, dx in steps:
                yy, xx = y + dy, x + dx
                if (0 <= yy < h and 0 <= xx < w and mask[yy, xx]
                        and labels[yy, xx] < 0):
                    labels[yy, xx] = index
                    cells.append((yy, xx))
        groups.append(cells)
    return labels, groups


def _near(cells, shape):
    h, w = shape
    return {(r+dr, c+dc) for r, c in cells for dr, dc in D8
            if 0 <= r+dr < h and 0 <= c+dc < w}


def _box_cells(box):
    r, c, h, w = box
    return {(y, x) for y in range(r, r+h) for x in range(c, c+w)}


def _perimeter(h, w):
    m = np.ones((h, w), dtype=bool)
    m[1:-1, 1:-1] = False
    return m


def _distances(mask, entries):
    """Orthogonal graph distances, with 10000 as a safe infinity."""
    h, w = mask.shape
    d = np.full((h, w), 10000, dtype=int)
    queue = deque()
    for r, c in entries:
        if mask[r, c] and d[r, c] != 0:
            d[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in D4:
            y, x = r+dr, c+dc
            if (0 <= y < h and 0 <= x < w and mask[y, x]
                    and d[y, x] > d[r, c]+1):
                d[y, x] = d[r, c]+1
                queue.append((y, x))
    return d


def _race(mask, entries, bg, plate, metric="channels"):
    colors = sorted({color for _, color in entries if color is not None})
    result = np.full(mask.shape, plate, dtype=int)
    result[mask] = bg
    if not colors:
        return result, {}, np.zeros(mask.shape, dtype=bool)
    distances = {}
    yy, xx = np.indices(mask.shape)
    for color in colors:
        starts = [p for p, paint in entries if paint == color and mask[p]]
        if metric == "manhattan":
            distances[color] = (np.minimum.reduce(
                [abs(yy-r)+abs(xx-c) for r, c in starts]) if starts
                else np.full(mask.shape, 10000, dtype=int))
        else:
            distances[color] = _distances(mask, starts)
    stack = np.stack([distances[c] for c in colors])
    best = stack.min(axis=0)
    ties = mask & (best < 10000) & ((stack == best).sum(axis=0) > 1)
    for color in colors:
        result[mask & ~ties & (best < 10000) & (distances[color] == best)] = color
    return result, distances, ties


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _analyze(value):
    a = _grid(value)
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    # A reference card is an isolated one-cell rectangular ring enclosing
    # background plus one payload color. Its mask register is explicit.
    candidates = {}
    for color in sorted(int(v) for v in np.unique(a) if v != bg):
        _, groups = _components(a == color, D8)
        cards = []
        for cells in groups:
            r = min(p[0] for p in cells); c = min(p[1] for p in cells)
            h = max(p[0] for p in cells)-r+1
            w = max(p[1] for p in cells)-c+1
            if h < 4 or w < 4 or len(cells) != 2*h+2*w-4:
                continue
            region = a[r:r+h, c:c+w]
            if not np.all(region[_perimeter(h, w)] == color):
                continue
            values = set(int(v) for v in np.unique(region[1:-1, 1:-1]))
            payload = values - {bg}
            if bg not in values or len(payload) != 1 or color in values:
                continue
            hue = payload.pop()
            cards.append((hue, (r, c, h, w), region[1:-1, 1:-1] == hue))
        if cards:
            candidates[color] = cards
    assert len(candidates) == 1, "ambiguous or absent reference frames"
    wire, card_list = next(iter(candidates.items()))
    cards = {}
    card_area = np.zeros(a.shape, dtype=bool)
    for hue, box, mask in card_list:
        assert hue not in cards, "duplicate stencil color"
        r, c, h, w = box
        assert not card_area[r:r+h, c:c+w].any()
        card_area[r:r+h, c:c+w] = True
        cards[hue] = (box, mask.copy())

    plate_area = np.zeros(a.shape, dtype=bool)
    plates = []
    for hue, (_, mask) in cards.items():
        ph, pw = mask.shape[0]+2, mask.shape[1]+2
        _, groups = _components((a == hue) & ~card_area)
        for cells in groups:
            r = min(p[0] for p in cells); c = min(p[1] for p in cells)
            h = max(p[0] for p in cells)-r+1
            w = max(p[1] for p in cells)-c+1
            if (h, w) != (ph, pw):
                continue
            region = a[r:r+h, c:c+w]
            assert np.all(region[1:-1, 1:-1] == hue)
            assert np.all(np.isin(region, (hue, wire)))
            assert all(region[y, x] == hue for y, x in
                       ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1)))
            assert not plate_area[r:r+h, c:c+w].any()
            plate_area[r:r+h, c:c+w] = True
            gates = []
            for y, x in np.argwhere(region == wire):
                y, x = int(y), int(x)
                assert y in (0, h-1) or x in (0, w-1)
                iy = 0 if y == 0 else h-3 if y == h-1 else y-1
                ix = 0 if x == 0 else w-3 if x == w-1 else x-1
                gates.append(((r+y, c+x), (iy, ix)))
            plates.append(dict(color=hue, box=(r, c, h, w), mask=mask,
                               gates=gates))
        assert np.all((a != hue) | card_area | plate_area), "unrecognized payload"
    assert plates, "no plates"
    assert not (plate_area & card_area).any()

    # Cards are not routes. Components include visible gate cells, but never
    # jump through a plate boundary or its solid interior.
    route_labels, route_groups = _components((a == wire) & ~card_area, D8)
    sources = {}
    marker_mask = (a != bg) & (a != wire) & ~card_area & ~plate_area
    source_cells = [tuple(map(int, p)) for p in np.argwhere(marker_mask)]
    cup_offsets = [{(dr, dc) for dr, dc in ((0, s), (t, 0), (t, s))}
                   for s in (-1, 1) for t in (-1, 1)]
    for p in source_cells:
        paint = int(a[p])
        assert paint not in cards
        near = _near({p}, a.shape)
        neighbors = {q for q in near if a[q] == wire and not card_area[q]}
        offsets = {(q[0]-p[0], q[1]-p[1]) for q in neighbors}
        assert offsets in cup_offsets, "source must sit in a three-neighbor elbow cup"
        assert all(a[q] in (bg, wire) for q in near)
        owners = {int(route_labels[q]) for q in neighbors}
        assert len(owners) == 1
        owner = owners.pop()
        assert owner not in sources, "more than one source on a route"
        sources[owner] = (p, paint)
    assert sources
    expected = a.copy()
    for plate in plates:
        r, c, h, w = plate['box']
        entries = [(p, sources.get(int(route_labels[q]), (None, None))[1])
                   for q, p in plate['gates']]
        inner, ds, ties = _race(plate['mask'], entries, bg, plate['color'])
        expected[r+1:r+h-1, c+1:c+w-1] = inner
        plate.update(entries=entries, distances=ds, ties=ties)
    return expected, dict(background=bg, wire=wire, cards=cards, plates=plates,
                          sources=sources, routes=route_groups,
                          route_labels=route_labels, plate_area=plate_area)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the output using only registered masks, ports and source routes."""
    return _analyze(input_grid)[0]


def _ports(mask):
    h, w = mask.shape
    return ([((0, x+1), (0, x)) for x in range(w)]
            + [((h+1, x+1), (h-1, x)) for x in range(w)]
            + [((y+1, 0), (y, 0)) for y in range(h)]
            + [((y+1, w+1), (y, w-1)) for y in range(h)])


def _choose_ports(rng, mask, owners, productive=True):
    options = [p for p in _ports(mask) if not productive or mask[p[1]]]
    for _ in range(60):
        order = rng.permutation(len(options))
        chosen = []
        for i in order:
            p = options[int(i)]
            if all(max(abs(p[0][j]-q[0][j]) for j in (0, 1)) > 1
                   for q, _ in chosen):
                chosen.append((p, owners[len(chosen)]))
                if len(chosen) == len(owners):
                    return [(p[0], p[1], owner) for p, owner in chosen]
    raise ValueError("too few separated ports")


def _mask(rng, h, w):
    for _ in range(100):
        kind = int(rng.integers(9))
        m = np.zeros((h, w), dtype=bool)
        if kind == 0:
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
        elif kind == 1:
            m[0] = m[-1] = True
        elif kind == 2:
            m[:] = True
            m.flat[int(rng.integers(m.size))] = False
        elif kind == 3:
            m[0, 0] = m[-1, -1] = True
        elif kind == 4:
            for y in range(h):
                m[y, int(round(y*(w-1)/(h-1)))] = True
        else:
            m = rng.random((h, w)) < rng.uniform(.2, .85)
            if kind == 5:
                m |= np.fliplr(m)
        if 0 < m.sum() < m.size and any(m[p[1]] for p in _ports(m)):
            return m
    raise AssertionError("mask sampler exhausted")


def _lesson_mask(h, w):
    m = np.zeros((h, w), dtype=bool)
    m[0, :w-1] = True
    m[:h-1, w-2] = True
    m[h-2, w-1] = True
    m[h-1, 0] = True
    return m


def _snake(w):
    m = np.zeros((5, w), dtype=bool)
    m[::2] = True
    m[1, -1] = m[3, 0] = True
    return m


def _path(rng, starts, goal, allowed):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[int(i)] for i in rng.permutation(8)]
    while queue:
        p = queue.popleft()
        if p == goal:
            path = set()
            while p is not None:
                path.add(p)
                p = previous[p]
            return path
        for dr, dc in directions:
            q = (p[0]+dr, p[1]+dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    return None


def _layout(rng, rectangles, shape):
    placed = {}
    occupied = set()
    order = sorted(range(len(rectangles)),
                   key=lambda i: -rectangles[i][0]*rectangles[i][1])
    for i in order:
        h, w = rectangles[i]
        found = False
        for _ in range(160):
            r = int(rng.integers(1, shape[0]-h))
            c = int(rng.integers(1, shape[1]-w))
            cells = _box_cells((r, c, h, w))
            if cells & occupied:
                continue
            occupied |= cells | _near(cells, shape)
            placed[i] = (r, c, h, w)
            found = True
            break
        if not found:
            return None
    return [placed[i] for i in range(len(rectangles))]


def _render_scene(rng, masks, plate_colors, port_lists, nsource, nroute,
                  paints, palette, attempt):
    bg, wire = palette[:2]
    card_colors = list(masks)
    rectangles = [(masks[c].shape[0]+2, masks[c].shape[1]+2) for c in plate_colors+card_colors]
    low = 18 if len(rectangles) <= 3 else 22 if len(rectangles) <= 6 else 27
    shape = ((30, 30) if attempt >= 20 else
             (int(rng.integers(low, 31)), int(rng.integers(low, 31))))
    boxes = _layout(rng, rectangles, shape)
    if boxes is None:
        return None
    canvas = {(r, c) for r in range(1, shape[0]-1) for c in range(1, shape[1]-1)}
    body = set().union(*(_box_cells(b) for b in boxes))
    cards = set().union(*(_box_cells(b) for b in boxes[len(plate_colors):]))
    gates = [set() for _ in range(nroute)]
    for box, ports in zip(boxes, port_lists):
        r, c, _, _ = box
        for (y, x), _, owner in ports:
            gates[owner].add((r+y, c+x))
    # Random cup locations are unrelated to payload color or receiver position.
    wires = [set() for _ in range(nroute)]
    markers = []
    protected = body | _near(body, shape)
    for owner in range(nroute):
        found = False
        for _ in range(200):
            r = int(rng.integers(2, shape[0]-2))
            c = int(rng.integers(2, shape[1]-2))
            dr, dc = _pick(rng, (-1, 1)), _pick(rng, (-1, 1))
            p = (r, c)
            cup = {(r, c+dc), (r+dr, c), (r+dr, c+dc)} if owner < nsource else {p}
            cells = cup | {p}
            if cells & protected:
                continue
            wires[owner] = cup
            if owner < nsource:
                markers.append((p, paints[owner]))
            protected |= cells | _near(cells, shape)
            found = True
            break
        if not found:
            return None
    marker_cells = {p for p, _ in markers}
    source_halo = marker_cells | _near(marker_cells, shape)
    for owner in rng.permutation(nroute):
        owner = int(owner)
        others = set().union(*(wires[i] | gates[i] for i in range(nroute) if i != owner))
        blocked = body | cards | _near(cards, shape) | source_halo | others | _near(others, shape)
        allowed = (canvas - blocked) | wires[owner] | gates[owner]
        for goal in sorted(gates[owner], key=lambda p: (p[1], p[0])):
            path = _path(rng, wires[owner], goal, allowed)
            if path is None:
                return None
            wires[owner] |= path
    a = np.full(shape, bg, dtype=int)
    for color, box in zip(plate_colors, boxes):
        r, c, h, w = box
        a[r:r+h, c:c+w] = color
    for color, box in zip(card_colors, boxes[len(plate_colors):]):
        r, c, h, w = box
        region = a[r:r+h, c:c+w]
        region[_perimeter(h, w)] = wire
        region[1:-1, 1:-1][masks[color]] = color
    for cells in wires:
        for p in cells:
            a[p] = wire
    for p, color in markers:
        a[p] = color
    # Labels are constructed from intended masks and ownership. Input parsing
    # is checked afterward and never supplies construction metadata to solve.
    b = a.copy()
    for color, box, ports in zip(plate_colors, boxes, port_lists):
        r, c, h, w = box
        entries = [(inner, paints[owner] if owner < nsource else None)
                   for _, inner, owner in ports]
        b[r+1:r+h-1, c+1:c+w-1] = _race(masks[color], entries, bg, color)[0]
    rotation = int(rng.integers(4))
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    yy, xx = np.nonzero(a != bg)
    y0, y1, x0, x1 = yy.min()-1, yy.max()+2, xx.min()-1, xx.max()+2
    a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    padding = []
    for dim in a.shape:
        extra = int(rng.integers(min(4, 30-dim)+1))
        before = int(rng.integers(extra+1))
        padding.append((before, extra-before))
    a, b = (np.pad(v, padding, constant_values=bg) for v in (a, b))
    try:
        predicted, info = _analyze(a)
    except AssertionError:
        return None
    if not np.array_equal(predicted, b) or len(info['sources']) != nsource:
        return None
    return dict(input=a.copy(), output=b.copy())


def _make_pair(rng, lesson, test=False):
    nsource = 1 if lesson == 0 else 2 if lesson == 1 else int(rng.integers(2 if test else 1, 4))
    nplate = 2 if lesson in (0, 1) else int(rng.integers(3 if test else 1, 6 if test else 5))
    nmatched = 1 if lesson in (0, 1) else int(rng.integers(1, min(3, max(1, nplate-1))+1))
    ndistractor = 1 if lesson == 0 else int(rng.random() < .3)
    palette = [int(v) for v in rng.permutation(10)]
    colors = palette[2:2+nmatched+ndistractor]
    paints = palette[2+nmatched+ndistractor:2+nmatched+ndistractor+nsource]
    if lesson not in (0, 1) and nsource > 1 and rng.random() < .2:
        paints[-1] = paints[0]
    masks = {}
    for i, color in enumerate(colors):
        if lesson == 0 and i == 0:
            masks[color] = _lesson_mask(int(rng.integers(3, 6)), int(rng.integers(3, 6)))
        elif lesson == 1 and i == 0:
            masks[color] = _snake(_pick(rng, (3, 5)))
        elif lesson == 0:
            # Same dimensions and a different pattern make color selection
            # essential even with only two demonstrations.
            base = masks[colors[0]]
            masks[color] = np.fliplr(base).copy()
        else:
            low = 3 if i == 0 and nsource == 3 else 2
            high = 7 if nplate <= 3 else 6
            masks[color] = _mask(rng, int(rng.integers(low, high)), int(rng.integers(low, high)))
    matched = colors[:nmatched]
    plate_colors = matched + [_pick(rng, matched) for _ in range(nplate-nmatched)]
    # Keep the first primary plate, but shuffle all other payload assignments.
    if len(plate_colors) > 2:
        tail = [plate_colors[int(i)] for i in rng.permutation(np.arange(1, nplate))]
        plate_colors[1:] = tail
    if lesson == 0:
        h, w = masks[colors[0]].shape
        ports = [[((0, 1), (0, 0), 0)], [((h+1, 1), (h-1, 0), 1)]]
        nroute = 2
    elif lesson == 1:
        h, w = masks[colors[0]].shape
        ports = [[((0, 1), (0, 0), 0), ((h+1, w), (h-1, w-1), 1)],
                 [((0, w), (0, w-1), 0)]]
        nroute = 2
    else:
        nroute = nsource + int(rng.random() < .55)
        try:
            ports = [_choose_ports(rng, masks[plate_colors[0]], list(range(nsource)))]
            if nplate > 1:
                ports.append(_choose_ports(rng, masks[plate_colors[1]], [0]))
            for color in plate_colors[2:]:
                if rng.random() < .25:
                    ports.append([])
                else:
                    owner = int(rng.integers(nroute))
                    owners = [owner]
                    if rng.random() < .3:
                        owners.append(int(rng.integers(nroute)))
                    ports.append(_choose_ports(rng, masks[color], owners, rng.random() < .8))
        except ValueError:
            return None
    for attempt in range(45):
        pair = _render_scene(rng, masks, plate_colors, ports, nsource, nroute, paints, palette, attempt)
        if pair is not None:
            return pair
    return None


def _counts(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _counts(num_train, 2, 10, "num_train")
    ne = _counts(num_test, 1, 4, "num_test")
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt+ne):
        split = 'train' if index < nt else 'test'
        lesson = index if index < 2 else None
        for _ in range(80):
            pair = _make_pair(rng, lesson, split == 'test')
            if pair is not None and _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(a, info, kind):
    out = a.copy()
    for p in info['plates']:
        r, c, h, w = p['box']
        mask = p['mask']
        entries = p['entries']
        if kind == 'ignore_A':
            mask = np.ones(mask.shape, dtype=bool)
        elif kind == 'ignore_B':
            entries = []
        elif kind == 'flip_mask':
            mask = np.fliplr(mask)
        out[r+1:r+h-1, c+1:c+w-1] = _race(mask, entries, info['background'],
                p['color'], 'manhattan' if kind == 'manhattan' else 'channels')[0]
    return out


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, info = _analyze(a)
            assert np.array_equal(expected, b), "wrong carving, route supply, distance or tie"
            assert np.any(a != b)
            assert np.all((a != b) <= info['plate_area'])
            assert not np.array_equal(_alternative(a, info, 'ignore_A'), b)
            assert not np.array_equal(_alternative(a, info, 'ignore_B'), b)
            for plate in info['plates']:
                r, c, h, w = plate['box']
                assert np.array_equal(a[r:r+h, c:c+w][_perimeter(h, w)],
                                      b[r:r+h, c:c+w][_perimeter(h, w)])
                inner = b[r+1:r+h-1, c+1:c+w-1]
                assert np.all(inner[~plate['mask']] == plate['color'])
                assert np.all(inner[plate['ties']] == info['background'])
            if split == 'train':
                train.append((a, b, info))
    for kind in ('ignore_A', 'ignore_B', 'manhattan', 'flip_mask'):
        assert any(not np.array_equal(_alternative(a, info, kind), b)
                   for a, b, info in train), f"training fails to distinguish {kind}"
    orientations = set(range(8))
    for _, _, info in train:
        for plate in info['plates']:
            m = plate['mask']
            transforms = [np.rot90(m, k) for k in range(4)] + [
                np.fliplr(np.rot90(m, k)) for k in range(4)]
            orientations &= {i for i, other in enumerate(transforms)
                             if m.shape == other.shape and np.array_equal(m, other)}
    assert orientations == {0}, "training leaves stencil orientation ambiguous"
    assert any(p['ties'].any() for _, _, info in train for p in info['plates'])
    assert any(p['distances'] and np.any(p['mask'] &
               (np.minimum.reduce(list(p['distances'].values())) == 10000))
               for _, _, info in train for p in info['plates']), "no unreachable hole lesson"
    assert any(not any(color is not None for _, color in p['entries'])
               for _, _, info in train for p in info['plates']), "no unsupplied plate lesson"
    assert any(any(color != p['color'] and mask.shape == p['mask'].shape
                   and not np.array_equal(mask, p['mask'])
                   for color, (_, mask) in info['cards'].items())
               for _, _, info in train for p in info['plates']), "no color selection lesson"


def self_test():
    """Bounded schema, geometry, counterfactual, covariance and diversity audit."""
    # Small semantic checks make distance zero, four-neighbor channel flow,
    # geodesic detours, blocked entries and same-color tie behavior explicit.
    diagonal = np.eye(2, dtype=bool)
    assert np.array_equal(_race(diagonal, [((0, 0), 1)], 0, 3)[0],
                          np.array([[1, 3], [3, 0]]))
    assert np.array_equal(_race(diagonal, [((0, 1), 1)], 0, 3)[0],
                          np.array([[0, 3], [3, 0]]))
    snake = _snake(3)
    assert np.array_equal(_race(snake, [((0, 0), 1), ((4, 2), 2)], 0, 3)[0],
                          np.array([[1, 1, 1], [3, 3, 1], [2, 0, 1],
                                    [2, 3, 3], [2, 2, 2]]))
    same = _race(snake, [((0, 0), 1), ((4, 2), 1)], 0, 3)[0]
    assert np.all(same[snake] == 1)
    shapes, sizes, colors, masks, plate_counts, source_counts = set(), set(), set(), set(), set(), set()
    disconnected = dense = sparse = blocked = repeat = branches = ties = detours = unused = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                _, info = _analyze(a)
                shapes.add(a.shape); colors.add(info['background'])
                plate_counts.add(len(info['plates'])); source_counts.add(len(info['sources']))
                repeat += len({p['color'] for p in info['plates']}) < len(info['plates'])
                unused += len(info['cards']) > len({p['color'] for p in info['plates']})
                ties += any(p['ties'].any() for p in info['plates'])
                detours += not np.array_equal(_alternative(a, info, 'manhattan'), b)
                owners = {}
                for p in info['plates']:
                    for q, _ in p['gates']:
                        owner = int(info['route_labels'][q])
                        owners.setdefault(owner, set()).add(p['box'])
                    blocked += any(color is not None and not p['mask'][point]
                                   for point, color in p['entries'])
                branches += any(len(v) > 1 for v in owners.values())
                for _, m in info['cards'].values():
                    sizes.add(m.shape); masks.add((m.shape, m.tobytes()))
                    disconnected += len(_components(m)[1]) > 1
                    dense += m.sum() == m.size-1
                    sparse += m.sum() <= max(2, m.size//3)
        if seed < 3:
            assert tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]) == tuple(
                _key(p['input']) for s in ('train', 'test') for p in generate_puzzle(seed)[s])
            # World rotations/reflections preserve the orientation relation
            # between every reference mask and its receiving plates.
            for pair in puzzle['train']:
                original = pair['input'].copy()
                for k in range(4):
                    assert np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k))
                    assert np.array_equal(solve(np.fliplr(np.rot90(pair['input'], k))),
                                          np.fliplr(np.rot90(pair['output'], k)))
                permutation = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                assert np.array_equal(original, pair['input']), "solver mutated its input"
    assert len(fingerprints) == 42 and len(shapes) >= 15
    assert len(sizes) >= 12 and len(masks) >= 100 and colors == set(range(10))
    assert {1, 2, 3, 4, 5} <= plate_counts and {1, 2, 3} <= source_counts
    assert min(disconnected, dense, sparse, repeat, branches, ties, detours, unused) >= 5
    assert blocked >= 1
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12345+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.5), (None, 1), (3, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = generate_puzzle(42, 2, 1)
    a, b = bad['test'][0]['input'], bad['test'][0]['output']
    r, c = np.argwhere(a != b)[0]
    b[r, c] = a[r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("partial subtraction/fill accepted")
    return dict(seeds=42, input_shapes=len(shapes), mask_shapes=len(sizes),
                masks=len(masks), plate_counts=sorted(plate_counts),
                source_counts=sorted(source_counts), disconnected=disconnected,
                near_full=int(dense), sparse=int(sparse), repeated_payload=repeat,
                branching=branches, tie_scenes=ties, detour_scenes=detours,
                blocked_receivers=blocked, unmatched_cards=unused)


if __name__ == '__main__':
    print(self_test())
