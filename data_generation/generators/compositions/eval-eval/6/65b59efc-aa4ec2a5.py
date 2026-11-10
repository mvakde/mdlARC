"""Pinned cavity printing. See task_spec.md for the input-only rule.

No parent module, data file, global random state, or saved example is used.
"""
from collections import deque
from itertools import permutations
import numpy as np

BG, PIN, DIVIDER = 0, 8, 9
D4 = ((1, 0), (0, 1), (-1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must be integer")
    _require(bool(((a >= 0) & (a <= 9)).all()), "bad grid color")
    return a.astype(np.int8, copy=False)


def _regions(mask, neighbors=D4):
    labels = np.full(mask.shape, -1, dtype=int)
    regions = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        index = len(regions)
        labels[r, c] = index
        queue = deque([(r, c)])
        cells = []
        while queue:
            rr, cc = queue.popleft()
            cells.append((rr, cc))
            for dr, dc in neighbors:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = index
                    queue.append((nr, nc))
        regions.append(np.array(cells, dtype=int))
    return labels, regions


def _holes(mask, neighbors=D4):
    padded = np.pad(mask, 1)
    lab, _ = _regions(~padded, neighbors)
    return ((lab != lab[0, 0]) & ~padded)[1:-1, 1:-1]


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[1+dr:1+dr+h, 1+dc:1+dc+w] for dr, dc in D8])


def _read(input_grid):
    g = _grid(input_grid)
    full_rows = np.flatnonzero(np.all(g == DIVIDER, axis=1))
    _require(len(full_rows) == 1, "one full library divider required")
    divider = int(full_rows[0])
    n = divider - 1
    _require(2 <= n <= 4 and divider + 1 < g.shape[0], "bad motif scale or missing scene")
    header, scene = g[:divider], g[divider+1:]
    active = np.flatnonzero(np.any(header != BG, axis=0))
    _require(len(active) > 0, "empty library")
    width = int(active[-1]) + 1
    k = width // (n + 1)
    _require(2 <= k <= 4 and width == k*(n+1), "bad library width")
    _require(np.all(header[:, width:] == BG), "nonblank header padding")
    library = {}
    order = []
    for j in range(k):
        c = j*(n+1)
        card = header[:n, c:c+n]
        colors = np.unique(card[card != BG])
        _require(len(colors) == 1 and 1 <= colors[0] <= 7, "one identity per motif")
        label = int(colors[0])
        _require(label not in library, "duplicate library identity")
        legend = header[n, c:c+n]
        target = int(legend[n//2])
        _require(1 <= target <= 7, "invisible or structural ink color")
        expected = np.zeros(n, dtype=np.int8)
        expected[n//2] = target
        _require(np.array_equal(legend, expected), "bad swatch")
        _require(np.all(header[:, c+n] == DIVIDER), "bad card separator")
        library[label] = (card == label, target)
        order.append(label)
    _require(not np.any(scene == DIVIDER), "scene contains divider color")
    body = scene != BG
    a, b, c, d = body[:-1, :-1], body[:-1, 1:], body[1:, :-1], body[1:, 1:]
    _require(not np.any((a == d) & (b == c) & (a != b)), "diagonal-only topology")
    ids, components = _regions(body)
    _require(bool(components), "empty scene")
    _require(len(components) == len(_regions(body, D8)[1]), "diagonal body contact")
    # Explicit four-neighbor topology; generation also avoids diagonal leaks.
    bg_ids, bg_regions = _regions(~body)
    exterior = np.zeros(scene.shape, dtype=bool)
    holes = []
    h, w = scene.shape
    for index, cells in enumerate(bg_regions):
        rr, cc = cells.T
        if np.any((rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)):
            exterior[rr, cc] = True
        else:
            owners = set()
            for r, c in cells:
                for dr, dc in D4:
                    q = (r+dr, c+dc)
                    if ids[q] >= 0:
                        owners.add(int(ids[q]))
            _require(len(owners) == 1, "collective or ambiguous enclosure")
            holes.append(dict(bg_id=index, cells=cells, owner=owners.pop()))
    _require(np.array_equal(_holes(body), _holes(body, D8)), "ambiguous enclosure")
    hole_by_region = {x['bg_id']: x for x in holes}
    objects = []
    for i, cells in enumerate(components):
        rr, cc = cells.T
        colors = np.unique(scene[rr, cc])
        labels = [int(x) for x in colors if x != PIN]
        _require(len(labels) == 1 and labels[0] in library, "unidentified body")
        pins = cells[scene[rr, cc] == PIN]
        owned = [x for x in holes if x['owner'] == i]
        objects.append(dict(cells=cells, label=labels[0], pins=pins, holes=owned))
        for r, c in pins:
            adjacent = []
            for dr, dc in D4:
                nr, nc = int(r)+dr, int(c)+dc
                if 0 <= nr < h and 0 <= nc < w and not body[nr, nc]:
                    adjacent.append((dr, dc, nr, nc))
            _require(len(adjacent) == 1, "pin must have one inward background neighbor")
            dr, dc, nr, nc = adjacent[0]
            region = int(bg_ids[nr, nc])
            if region in hole_by_region:
                hole = hole_by_region[region]
                _require(hole['owner'] == i and 'anchor' not in hole, "multiple cavity pins")
                hole.update(anchor=(nr, nc), direction=(dr, dc), pin=(int(r), int(c)))
        _require(all('anchor' in x for x in owned), "cavity without a pin")
    outline = _dilate(body) & exterior
    return dict(n=n, k=k, order=order, library=library, scene=scene,
                body=body, objects=objects, holes=holes, exterior=exterior, outline=outline)


def _print_bits(hole, motif, *, mode='correct'):
    cells = hole['cells']
    dr, dc = hole['direction']
    # Choose the tangent so a downward normal has a rightward column axis.
    er, ec = -dc, dr
    anchor = np.array(hole['anchor'])
    delta = cells - anchor
    a = delta[:, 0]*dr + delta[:, 1]*dc
    b = delta[:, 0]*er + delta[:, 1]*ec
    if mode == 'upright':
        a, b = delta[:, 0], delta[:, 1]
    elif mode == 'bbox':
        a, b = a-a.min(), b-b.min()
    elif mode == 'mirror':
        b = -b
    elif mode == 'solid':
        return np.ones(len(cells), dtype=bool)
    return motif[a % len(motif), b % len(motif)]


def _render(p, mode='correct'):
    out = p['scene'].copy()
    out[p['outline']] = DIVIDER
    for obj in p['objects']:
        if not obj['holes']:
            if mode == 'all_pinned' and len(obj['pins']):
                rr, cc = obj['cells'].T
                out[rr, cc] = p['library'][obj['label']][1]
            continue
        motif, target = p['library'][obj['label']]
        if mode == 'first':
            motif, target = p['library'][p['order'][0]]
        elif mode == 'unkeyed':
            target = obj['label']
        rr, cc = obj['cells'].T
        out[rr, cc] = target
        for hole in obj['holes']:
            rr, cc = hole['cells'].T
            bits = _print_bits(hole, motif, mode=mode)
            out[rr, cc] = np.where(bits, target, BG)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible cards, original topology, and wall pins."""
    return _render(_read(input_grid))


def _motif(rng, n, witness=False):
    for _ in range(64):
        style = int(rng.integers(6))
        rr, cc = np.indices((n, n))
        if style == 0:
            m = rng.random((n, n)) < rng.uniform(.2, .8)
        elif style == 1:
            m = (rr+cc) % 2 == int(rng.integers(2))
        elif style == 2:
            m = rr == int(rng.integers(n))
        elif style == 3:
            m = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
        elif style == 4:
            m = rr == cc
        else:
            m = np.zeros((n, n), bool)
            m.flat[rng.choice(n*n, int(rng.integers(1, n*n)), replace=False)] = True
        if m.any() and not m.all() and (not witness or
                all(not np.array_equal(m, np.rot90(m, q)) for q in (1, 2, 3))):
            return m
    m = np.zeros((n, n), bool)
    m[0, 0] = True
    return m


def _cavity(rng, h, w, irregular):
    m = np.ones((h, w), bool)
    if irregular and h >= 3 and w >= 3:
        style = int(rng.integers(3))
        if style == 0:
            a = int(rng.integers(1, h-1))
            b = int(rng.integers(1, w-1))
            m[:a, -b:] = False
        elif style == 1:
            lengths = np.sort(rng.integers(1, w+1, size=h))
            lengths[-1] = w
            m = np.arange(w)[None, :] < lengths[:, None]
        else:
            m[:] = False
            a, b = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            m[a, :] = True
            m[:, b] = True
        m = np.rot90(m, 2*int(rng.integers(2)))
    return m


def _pin_candidates(body, cavity, direction):
    dr, dc = direction
    result = []
    h, w = body.shape
    for r, c in np.argwhere(body):
        r, c = int(r), int(c)
        nr, nc = r+dr, c+dc
        if not (0 <= nr < h and 0 <= nc < w and cavity[nr, nc]):
            continue
        neighbors = [(r+a, c+b) for a, b in D4]
        if all(0 <= a < h and 0 <= b < w for a, b in neighbors) and sum(not body[a, b] for a, b in neighbors) == 1:
            result.append((r, c))
    return result


def _chamber(rng, n, label, directions, irregular=False, opened=False,
             large=False, compact=False, teaching=False):
    # One or two negative-space pockets in a connected thick body.
    pockets = []
    for _ in directions:
        extra = 0 if compact else int(rng.integers(0, 3))
        h = n+1+extra+(1 if large else 0)
        w = (2*n+2 if large else n+1)+(0 if compact else int(rng.integers(0, 2)))
        pockets.append(_cavity(rng, h, w, irregular))
    height = max(x.shape[0] for x in pockets)+4
    width = sum(x.shape[1] for x in pockets)+2*(len(pockets)+1)
    body = np.ones((height, width), bool)
    cavity_masks = []
    col = 2
    for pocket in pockets:
        h, w = pocket.shape
        row = 2 + int(rng.integers(height-h-3))
        mask = np.zeros(body.shape, bool)
        mask[row:row+h, col:col+w] = pocket
        body[mask] = False
        cavity_masks.append(mask)
        col += w+2
    # Exterior sculpting varies silhouettes without changing cavities.
    for r, c in ((0, 0), (0, width-1), (height-1, 0), (height-1, width-1)):
        if rng.random() < .55:
            body[r, c] = False
    turns = 0 if compact else (2*int(rng.integers(2)) if len(directions) == 3 else int(rng.integers(4)))
    body = np.rot90(body, turns).copy()
    cavity_masks = [np.rot90(m, turns).copy() for m in cavity_masks]
    pin_cells = []
    for mask, direction in zip(cavity_masks, directions):
        candidates = _pin_candidates(body, mask, direction)
        _require(bool(candidates), "no straight inner-wall pin position")
        if teaching:
            # Prefer a phase distinct from a corner-aligned wallpaper.
            dr, dc = direction
            tangent = (-dc, dr)
            coords = np.argwhere(mask)
            low = int((coords[:, 0]*tangent[0]+coords[:, 1]*tangent[1]).min())
            preferred = [q for q in candidates if
                         (q[0]*tangent[0]+q[1]*tangent[1]-low) % n != 0]
            if preferred:
                candidates = preferred
        pin_cells.append(candidates[int(rng.integers(len(candidates)))])
    if opened:
        _require(len(directions) == 1, "only single open chambers constructed")
        # Open the opposite wall, leaving the printed-side pin unchanged.
        dr, dc = directions[0]
        rr, cc = np.argwhere(cavity_masks[0])[0]
        # Start on the extremal cavity cell in the chosen outward direction.
        coords = np.argwhere(cavity_masks[0])
        score = coords[:, 0]*dr+coords[:, 1]*dc
        edge = coords[score == score.max()]
        rr, cc = map(int, edge[len(edge)//2])
        while 0 <= rr < body.shape[0] and 0 <= cc < body.shape[1]:
            body[rr, cc] = False
            rr, cc = rr+dr, cc+dc
        _require(not _holes(body).any(), "door failed to open chamber")
    grid = np.where(body, label, BG).astype(np.int8)
    for r, c in pin_cells:
        _require(body[r, c], "door erased pin")
        grid[r, c] = PIN
    _require(len(_regions(body)[1]) == 1, "split chamber body")
    return grid


def _solid(rng, label):
    h, w = int(rng.integers(1, 6)), int(rng.integers(1, 7))
    mask = np.ones((h, w), bool)
    if h >= 3 and w >= 3 and rng.random() < .7:
        mask[:int(rng.integers(1, h)), 1:int(rng.integers(2, w))] = False
    return np.where(mask, label, BG).astype(np.int8)


def _pack(rng, items, max_h, minimum_width):
    # Free offsets, bounded nonoverlapping boxes, no quadrants or fixed slots.
    gap = int(rng.choice([1, 1, 2, 3]))
    for attempt in range(24):
        h = max_h if attempt >= 12 else int(rng.integers(max(10, max(x.shape[0] for x in items)), max_h+1))
        w = 30 if attempt >= 12 else int(rng.integers(max(minimum_width, max(x.shape[1] for x in items), 12), 31))
        canvas = np.zeros((h, w), np.int8)
        excluded = np.zeros((h, w), bool)
        order = sorted(range(len(items)), key=lambda i: -items[i].size)
        for i in order:
            item = items[i]
            ih, iw = item.shape
            choices = [(r, c) for r in range(h-ih+1) for c in range(w-iw+1)
                       if not excluded[r:r+ih, c:c+iw].any()]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            canvas[r:r+ih, c:c+iw] = item
            excluded[max(0, r-gap):min(h, r+ih+gap), max(0, c-gap):min(w, c+iw+gap)] = True
        else:
            return _crop_scene(rng, canvas, minimum_width, max_h)
    # Finite exhaustive shelf fallback for <=4 bodies; do not discard evidence.
    for order in permutations(range(len(items))):
        for cuts in range(1 << (len(items)-1)):
            rows, row = [], []
            for j, i in enumerate(order):
                row.append(i)
                if j == len(items)-1 or cuts & (1 << j):
                    rows.append(row)
                    row = []
            heights = [max(items[i].shape[0] for i in row) for row in rows]
            widths = [sum(items[i].shape[1] for i in row)+len(row)-1 for row in rows]
            if sum(heights)+len(rows)-1 <= max_h and max(widths) <= 30:
                canvas = np.zeros((sum(heights)+len(rows)-1, max(minimum_width, max(widths))), np.int8)
                r = 0
                for row, height in zip(rows, heights):
                    c = 0
                    for i in row:
                        ih, iw = items[i].shape
                        canvas[r:r+ih, c:c+iw] = items[i]
                        c += iw+1
                    r += height+1
                return _crop_scene(rng, canvas, minimum_width, max_h)
    return None


def _crop_scene(rng, scene, minimum_width, max_h):
    rr, cc = np.nonzero(scene)
    result = scene[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()
    style = int(rng.integers(4))
    margins = ([0, 0, 0, 0] if style == 0 else
               list(map(int, rng.integers(2 if style == 1 else 0, 5, size=4))))
    for a, b, spare in ((0, 1, max_h-result.shape[0]), (2, 3, 30-result.shape[1])):
        # Capacity, rather than original random offset, limits added margins.
        if margins[a]+margins[b] > spare:
            first = int(rng.integers(2))
            i, j = (a, b) if first else (b, a)
            margins[i] = min(margins[i], spare)
            margins[j] = min(margins[j], spare-margins[i])
    result = np.pad(result, ((margins[0], margins[1]), (margins[2], margins[3])))
    if result.shape[1] < minimum_width:
        result = np.pad(result, ((0, 0), (0, minimum_width-result.shape[1])))
    return result


def _pair(rng, n, k, teaching, shift, test=False):
    labels = list(map(int, rng.choice(np.arange(1, 8), k, replace=False)))
    motifs = {x: _motif(rng, n, witness=bool(teaching)) for x in labels}
    if teaching and np.array_equal(motifs[labels[0]], motifs[labels[1]]):
        motifs[labels[1]] = ~motifs[labels[0]]
    targets = {x: int(rng.choice([c for c in range(1, 8) if not teaching or c != x])) for x in labels}
    if teaching:
        targets[labels[1]] = int(rng.choice([c for c in range(1, 8)
                                  if c not in (labels[1], targets[labels[0]])]))
    for attempt in range(16):
        compact = attempt >= 8
        if teaching == 1:
            first = _chamber(rng, n, labels[0], [D4[0]], compact=compact, teaching=True)
            # Exact geometric twin: unseal the bottom wall, preserving its pin.
            opened = first.copy()
            hole = _holes(first != 0)
            rr, cc = np.argwhere(hole).T
            c = int(cc[np.argmax(rr)])
            r = int(rr.max())
            while r < opened.shape[0]:
                opened[r, c] = 0
                r += 1
            second = _chamber(rng, n, labels[1], [D4[1]], irregular=True,
                              compact=compact, teaching=True)
            items = [first, opened, second]
        elif teaching == 2:
            dirs = [D4[2], D4[3]]
            if rng.random() < .35:
                dirs.append(D4[int(rng.integers(4))])
            items = [_chamber(rng, n, labels[0], dirs, irregular=True,
                               compact=compact, teaching=True), _solid(rng, labels[1])]
        else:
            count = int(rng.integers(1, 5))
            items = []
            for j in range(count):
                label = labels[int(rng.integers(k))]
                kind = int(rng.choice(4, p=[.38, .24, .23, .15]))
                if test and j == 0:
                    kind = 3 if shift == 'recombine' else 0
                if compact and j > 0 and kind == 3:
                    kind = 0
                if kind == 2:
                    item = _solid(rng, label)
                else:
                    num = int(rng.integers(2, 4)) if kind == 3 else 1
                    dirs = [D4[int(rng.integers(4))] for _ in range(num)]
                    item = _chamber(rng, n, label, dirs, irregular=rng.random() < .6,
                                    opened=kind == 1, large=test and shift == 'repeat' and j == 0,
                                    compact=compact)
                items.append(item)
        scene = _pack(rng, items, 30-(n+2), k*(n+1))
        if scene is not None:
            break
    else:
        raise RuntimeError('bounded chamber packing exhausted')
    width = scene.shape[1]
    header = np.zeros((n+2, width), np.int8)
    order = list(map(int, rng.permutation(labels)))
    for j, label in enumerate(order):
        c = j*(n+1)
        header[:n, c:c+n] = np.where(motifs[label], label, BG)
        header[n, c+n//2] = targets[label]
        header[:n+1, c+n] = DIVIDER
    header[n+1, :] = DIVIDER
    inp = np.concatenate((header, scene), axis=0)
    p = _read(inp)
    return dict(input=inp, output=_render(p))


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 5))
    shift = str(rng.choice(['library', 'repeat', 'recombine']))
    variable_scale = bool(rng.random() < .35)
    if shift == 'repeat':
        variable_scale = False
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            teaching = i+1 if split == 'train' and i < 2 else 0
            scale = n if not variable_scale or i == 0 else int(rng.integers(2, 5))
            if variable_scale and split == 'train' and i == 1:
                scale = int(rng.integers(3, 5))
            k = int(rng.integers(2, 4)) if split == 'train' else (4 if shift == 'library' else int(rng.integers(2, 5)))
            for _ in range(32):
                pair = _pair(rng, scale, k, teaching, shift, test=split == 'test')
                key = pair['input'].shape, pair['input'].tobytes()
                p = _read(pair['input'])
                if split == 'test' and shift == 'recombine':
                    signature = (p['k'], len(p['objects']), len(p['holes']))
                    taught = [_read(x['input']) for x in puzzle['train']]
                    if signature in {(x['k'], len(x['objects']), len(x['holes'])) for x in taught}:
                        continue
                # Teaching ink must expose clipping, phase, identity and rotation.
                if teaching and any(np.array_equal(pair['output'], _render(p, mode))
                                    for mode in ('solid', 'first', 'unkeyed', 'bbox')):
                    continue
                if teaching == 2 and np.array_equal(pair['output'], _render(p, 'upright')):
                    continue
                if teaching == 2 and scale >= 3 and np.array_equal(pair['output'], _render(p, 'mirror')):
                    continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct/evidence sampling exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            p = _read(inp)
            _require(out.shape == p['scene'].shape, "scene crop dimensions differ")
            _require(np.all(out[p['outline']] == DIVIDER), "wrong exterior outline")
            _require(np.all(out[p['exterior'] & ~p['outline']] == BG), "far exterior changed")
            for obj in p['objects']:
                rr, cc = obj['cells'].T
                motif, ink = p['library'][obj['label']]
                if obj['holes']:
                    _require(np.all(out[rr, cc] == ink), "whole hollow body not recolored")
                    for hole in obj['holes']:
                        hr, hc = hole['cells'].T
                        dr, dc = hole['direction']
                        ar, ac = hole['anchor']
                        # Per-pixel equation, separate from vectorized rendering.
                        for r, c in zip(hr, hc):
                            u = ((r-ar)*dr+(c-ac)*dc) % p['n']
                            v = (-(r-ar)*dc+(c-ac)*dr) % p['n']
                            _require(out[r, c] == (ink if motif[u, v] else BG), "wrong pinned print")
                else:
                    _require(np.array_equal(out[rr, cc], p['scene'][rr, cc]), "cavity-free body or pin changed")
            if split == 'train':
                training.append(p)
    _require(any(any(not o['holes'] and len(o['pins']) for o in p['objects']) for p in training),
             "missing pinned open-chamber contrast")
    _require(any(any(len(o['holes']) >= 2 for o in p['objects']) for p in training), "missing multiple-cavity demonstration")
    _require({h['direction'] for p in training for h in p['holes']} == set(D4), "four inward directions not taught")
    for mode in ('solid', 'first', 'unkeyed', 'bbox', 'upright', 'mirror', 'all_pinned'):
        if mode == 'mirror' and all(p['n'] == 2 for p in training):
            # Negating a coordinate modulo two gives the same coordinate.
            continue
        _require(any(not np.array_equal(_render(p), _render(p, mode)) for p in training),
                 'training permits shortcut: '+mode)


def self_test():
    """Bounded pixel checks, counterfactuals, count coverage and 42-seed diversity."""
    shapes, scenes, motifs, cavity_shapes, inks, scales, counts, directions = set(), set(), set(), set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple((p['input'].shape, p['input'].tobytes()) for s in ('train', 'test') for p in puzzle[s])
        signatures.add(signature)
        if seed < 3:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(a[key], b[key]) for s in puzzle for a, b in zip(puzzle[s], again[s])
                         for key in ('input', 'output')), "nondeterministic puzzle")
        for s in puzzle:
            for pair in puzzle[s]:
                p = _read(pair['input'])
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "solve mismatch")
                shapes.add(pair['input'].shape)
                scenes.add((p['scene'].shape, p['body'].tobytes()))
                scales.add(p['n'])
                counts.add((p['k'], len(p['objects']), len(p['holes'])))
                for motif, ink in p['library'].values():
                    motifs.add((motif.shape, motif.tobytes()))
                    inks.add(ink)
                for hole in p['holes']:
                    rr, cc = hole['cells'].T
                    local = np.zeros((rr.max()-rr.min()+1, cc.max()-cc.min()+1), bool)
                    local[rr-rr.min(), cc-cc.min()] = True
                    cavity_shapes.add((local.shape, local.tobytes()))
                    directions.add(hole['direction'])
    _require(len(signatures) == 42, "seed repetition")
    _require(scales == {2, 3, 4} and directions == set(D4), "scale/direction diversity")
    _require(len(shapes) >= 30 and len(scenes) >= 150, "scene geometry diversity")
    _require(len(motifs) >= 45 and len(cavity_shapes) >= 30, "motif/cavity diversity")
    _require(len(counts) >= 15 and inks == set(range(1, 8)), "count/ink diversity")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(4242+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(11, 2, 1)
    q = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
    q['test'][0]['output'].flat[0] = (int(q['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(q)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted incorrect pixel')
