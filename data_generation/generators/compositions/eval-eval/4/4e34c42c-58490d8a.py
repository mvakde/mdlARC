"""Census-locked translational registration of framed transparent observations.

Only pixels determine the answer. The chart is an inverse measurement: it
selects among exact landmark registrations by counting final monochrome objects.
"""
from itertools import product
import numpy as np


def _groups(points, diagonal=True):
    remaining = set(points)
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}


def _fingerprint(cells):
    return tuple(sorted(_normalize(cells).items()))


def _shape(cells):
    return max(r for r, c in cells) + 1, max(c for r, c in cells) + 1


def _render(cells, background):
    cells = _normalize(cells)
    result = np.full(_shape(cells), background, dtype=np.int8)
    for p, color in cells.items():
        result[p] = color
    return result


def _grid_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _checked_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30')
    # All accepted integer widths/signs represent the same ARC cells. In
    # particular, bincount cannot consume uint64 directly on every NumPy build.
    return a.astype(np.int64, copy=False)


def _chart_counts(panel, field):
    """Accept two or more parallel, spaced unary category lines."""
    positions = np.argwhere(panel != field)
    colors = sorted(set(int(panel[r, c]) for r, c in positions))
    if not 2 <= len(colors) <= 4:
        return None
    if any(len(g) != 1 for g in _groups(map(tuple, positions))):
        return None
    counts = {color: int(np.count_nonzero(panel == color)) for color in colors}
    for axis in (0, 1):
        rows, starts = [], []
        for color in colors:
            p = np.argwhere(panel == color)
            if len(set(p[:, axis])) != 1:
                break
            along = sorted(map(int, p[:, 1-axis]))
            if any(b-a != 2 for a, b in zip(along, along[1:])):
                break
            rows.append(int(p[0, axis]))
            starts.append((along[0], along[-1]))
        else:
            if (len(set(rows)) == len(colors)
                    and all(b-a >= 2 for a, b in zip(sorted(rows), sorted(rows)[1:]))
                    and (len({a for a, b in starts}) == 1
                         or len({b for a, b in starts}) == 1)):
                return counts
    return None


def _parse(grid):
    grid = _checked_grid(grid)
    frequency = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequency))
    if frequency[bg] * 2 <= grid.size:
        raise ValueError('Background must occupy more than half the input')
    frames = {}
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        boxes = []
        for group in _groups(map(tuple, np.argwhere(grid == color)), False):
            r0, r1 = min(r for r, c in group), max(r for r, c in group)
            c0, c1 = min(c for r, c in group), max(c for r, c in group)
            if r1-r0 < 3 or c1-c0 < 3:
                continue
            perimeter = {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)
                         if r in (r0, r1) or c in (c0, c1)}
            if group == perimeter:
                # A hollow monochrome data object is not a packet frame.
                # Every actual packet visibly contains a multicolor landmark.
                interior = grid[r0+1:r1, c0+1:c1]
                colored = {(int(r), int(c)): int(interior[r, c])
                           for r, c in np.argwhere(interior != bg)}
                if any(len({colored[p] for p in part}) > 1 for part in _groups(colored)):
                    boxes.append((r0, r1+1, c0, c1+1))
        if 2 <= len(boxes) <= 4:
            frames[color] = boxes
    # Nested rectangular landmark features can suggest another frame color.
    # Resolve that suggestion using the complete visible packet/chart roles,
    # not an arbitrary first-color or largest-rectangle preference.
    interpretations = []
    for frame, bounds in frames.items():
        try:
            interpretations.append(_parse_packets(grid, bg, frame, bounds))
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError('Need one complete interpretation of the frames and chart')
    return interpretations[0]


def _parse_packets(grid, bg, frame, bounds):
    claimed = np.zeros(grid.shape, bool)
    cards = []
    for r0, r1, c0, c1 in bounds:
        if claimed[r0:r1, c0:c1].any():
            raise ValueError('Frames must be disjoint')
        claimed[r0:r1, c0:c1] = True
        interior = grid[r0+1:r1-1, c0+1:c1-1]
        if np.any(interior == frame):
            raise ValueError('Frame color may only outline observations')
        cells = {(int(r), int(c)): int(interior[r, c])
                 for r, c in np.argwhere(interior != bg)}
        if not cells:
            raise ValueError('Empty observation')
        landmarks, data = [], {}
        for group in _groups(cells):
            part = {p: cells[p] for p in group}
            if len(set(part.values())) > 1:
                if len(set(part.values())) != 2 or len(part) < 3:
                    raise ValueError('A landmark is a connected two-color feature')
                origin = min(r for r, c in part), min(c for r, c in part)
                landmarks.append((origin, _fingerprint(part), set(part)))
            else:
                data.update(part)
        if not landmarks:
            raise ValueError('Every observation needs a multicolored landmark')
        cards.append(dict(cells=cells, data=data, landmarks=landmarks))
    charts = []
    for field in sorted(set(map(int, grid.ravel())) - {bg, frame}):
        rr, cc = np.where(grid == field)
        r0, r1 = int(rr.min()), int(rr.max())+1
        c0, c1 = int(cc.min()), int(cc.max())+1
        panel = grid[r0:r1, c0:c1]
        if min(panel.shape) < 3 or claimed[r0:r1, c0:c1].any():
            continue
        if not all(np.all(edge == field) for edge in
                   (panel[0], panel[-1], panel[:, 0], panel[:, -1])):
            continue
        counts = _chart_counts(panel, field)
        if counts is not None and not ({bg, frame, field} & set(counts)):
            charts.append((field, (r0, r1, c0, c1), counts))
    if len(charts) != 1:
        raise ValueError('Need one visibly distinct census panel')
    field, chart_bounds, targets = charts[0]
    r0, r1, c0, c1 = chart_bounds
    claimed[r0:r1, c0:c1] = True
    if np.any(grid[~claimed] != bg):
        raise ValueError('Material outside observation frames or chart')
    # The packets have a complete exterior background halo, including corners.
    for r0, r1, c0, c1 in bounds + [chart_bounds]:
        halo = grid[max(0, r0-1):min(grid.shape[0], r1+1),
                    max(0, c0-1):min(grid.shape[1], c1+1)].copy()
        ar, ac = r0-max(0, r0-1), c0-max(0, c0-1)
        halo[ar:ar+r1-r0, ac:ac+c1-c0] = bg
        if np.any(halo != bg):
            raise ValueError('Packets must have a background separation halo')
    base_indices = [i for i, card in enumerate(cards) if len(card['landmarks']) >= 2]
    if len(base_indices) != 1:
        raise ValueError('Exactly one observation supplies multiple docking landmarks')
    base = cards.pop(base_indices[0])
    anchor_colors = set(v for card in [base]+cards
                        for origin, pattern, pts in card['landmarks'] for p, v in pattern)
    if len(anchor_colors) != 2 or anchor_colors & ({bg, frame, field} | set(targets)):
        raise ValueError('Landmark colors are distinct structural roles')
    for card in [base]+cards:
        if set(card['data'].values()) & (anchor_colors | {bg, frame, field}):
            raise ValueError('Structural colors cannot be object categories')
    if not set(targets) <= set(v for card in [base]+cards for v in card['data'].values()):
        raise ValueError('Every chart category needs observed data')
    if any(len(card['landmarks']) != 1 for card in cards):
        raise ValueError('Movable observations have one landmark each')
    return dict(background=bg, frame=frame, field=field, base=base, movers=cards,
                targets=targets, anchor_colors=anchor_colors,
                bounds=bounds, chart_bounds=chart_bounds)


def _counts(cells, anchor_colors, diagonal=True):
    return {color: len(_groups({p for p, v in cells.items() if v == color}, diagonal))
            for color in sorted(set(cells.values()) - set(anchor_colors))}


def _registered_candidates(meta):
    """Finite registrations first; the chart never proposes arbitrary offsets."""
    base, movers = meta['base'], meta['movers']
    domains = []
    for card in movers:
        origin, pattern, pts = card['landmarks'][0]
        matches = sorted({(p[0]-origin[0], p[1]-origin[1])
                          for p, q, points in base['landmarks'] if q == pattern})
        if not matches:
            raise ValueError('Landmark has no same-orientation match on the base')
        domains.append(matches)
    candidates = {}
    for offsets in product(*domains):
        union = dict(base['cells'])
        valid, overlap = True, 0
        for card, (dr, dc) in zip(movers, offsets):
            for (r, c), color in card['cells'].items():
                p = r+dr, c+dc
                if p in union:
                    if union[p] != color:
                        valid = False
                        break
                    overlap += 1
                union[p] = color
            if not valid:
                break
        if not valid or max(_shape(_normalize(union))) > 30:
            continue
        # Landmarks and their holes remain isolated, intact features.
        anchor_points = {p for p, v in union.items() if v in meta['anchor_colors']}
        data_points = set(union) - anchor_points
        if any((r+dr, c+dc) in data_points for r, c in anchor_points
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            continue
        actual = sorted(_fingerprint({p: union[p] for p in group})
                        for group in _groups(anchor_points))
        original = sorted(pattern for origin, pattern, pts in base['landmarks'])
        if actual != original:
            continue
        key = tuple(sorted(union.items()))
        if key not in candidates:
            candidates[key] = dict(cells=union, counts=_counts(union, meta['anchor_colors']),
                                   offsets=offsets, overlap=overlap)
    return list(candidates.values())


def _inspect(grid):
    meta = _parse(grid)
    candidates = _registered_candidates(meta)
    selected = [c for c in candidates if all(c['counts'].get(v, 0) == n
                                           for v, n in meta['targets'].items())]
    if len(selected) != 1:
        raise ValueError('The census must select exactly one assembled union')
    meta.update(candidates=candidates, selected=selected[0],
                expected=_render(selected[0]['cells'], meta['background']))
    return meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode roles, enumerate exact translations, and enforce the visible census."""
    return _inspect(input_grid)['expected'].copy()


_SHAPES = (
    {(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 2), (2, 1)},
    {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
)
_SLOTS = ((4, 0), (4, 4), (0, 4))


def _object_shape(rng):
    if rng.random() < .65:
        points = set(_SHAPES[int(rng.integers(len(_SHAPES)))])
    else:
        points = {(1, 1)}
        for _ in range(int(rng.integers(1, 7))):
            frontier = sorted({(r+dr, c+dc) for r, c in points
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r+dr < 3 and 0 <= c+dc < 3} - points)
            if frontier:
                points.add(frontier[int(rng.integers(len(frontier)))])
    if rng.random() < .5:
        points = {(r, 2-c) for r, c in points}
    for _ in range(int(rng.integers(4))):
        points = {(2-c, r) for r, c in points}
    return set(_normalize({p: 1 for p in points}))


def _landmark(rng, colors):
    # Independent small signatures, including holes, thin bars, and corners.
    shapes = [set(s) for s in _SHAPES[3:]]
    shapes += [{(0, 0), (0, 1), (0, 2)},
               {(0, 0), (0, 1), (1, 0)},
               {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}]
    pts = shapes[int(rng.integers(len(shapes)))]
    for _ in range(int(rng.integers(4))):
        pts = {(-c, r) for r, c in pts}
    pts = sorted(_normalize({p: 1 for p in pts}))
    values = [int(rng.choice(colors)) for p in pts]
    values[0], values[-1] = colors
    return dict(zip(pts, values))


def _put(target, cells, origin):
    for (r, c), color in cells.items():
        p = r+origin[0], c+origin[1]
        if p in target and target[p] != color:
            raise ValueError('Construction color conflict')
        target[p] = int(color)


def _card(cells, bg, frame, rng):
    # Empty margins carry no coordinates; translations come only from landmarks.
    cells = _normalize(cells)
    h, w = _shape(cells)
    top, left = int(rng.integers(2)), int(rng.integers(2))
    bottom, right = int(rng.integers(2)), int(rng.integers(2))
    result = np.full((h+2+top+bottom, w+2+left+right), bg, np.int8)
    result[0], result[-1], result[:, 0], result[:, -1] = frame, frame, frame, frame
    for (r, c), color in cells.items():
        result[r+1+top, c+1+left] = color
    return result


def _make_chart(targets, bg, field, rng):
    colors = list(targets)
    rng.shuffle(colors)
    gap = int(rng.integers(2, 4))
    top = int(rng.integers(1, 3))
    rows = [top+i*gap for i in range(len(colors))]
    inset = int(rng.integers(1, 3))
    h = rows[-1] + int(rng.integers(2, 4))
    w = inset + 2*(max(targets.values())-1) + int(rng.integers(2, 4))
    chart = np.full((h, w), field, np.int8)
    for r, color in zip(rows, colors):
        for i in range(targets[color]):
            chart[r, inset+2*i] = color
    return np.rot90(chart, int(rng.integers(4))).copy()


def _pack(packets, bg, rng):
    """Random rectangle packing, then a small deterministic edge-position search."""
    order = sorted(range(len(packets)), key=lambda i: -packets[i].size)
    for attempt in range(45):
        h, w = (30, 30) if attempt >= 25 else (int(rng.integers(24, 31)), int(rng.integers(24, 31)))
        grid = np.full((h, w), bg, np.int8)
        blocked = np.zeros((h, w), bool)
        success = True
        for i in order:
            ph, pw = packets[i].shape
            possible = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)
                        if not blocked[r:r+ph, c:c+pw].any()]
            if not possible:
                success = False
                break
            r, c = possible[int(rng.integers(len(possible)))]
            grid[r:r+ph, c:c+pw] = packets[i]
            blocked[max(0, r-1):min(h, r+ph+1), max(0, c-1):min(w, c+pw+1)] = True
        if success and np.count_nonzero(grid == bg)*2 > grid.size:
            return grid
    # A search over all packet orders and edge-derived placements, no content changes.
    placed = []
    visits = [0]

    def search(remaining):
        visits[0] += 1
        if visits[0] > 25000:
            return None
        if not remaining:
            grid = np.full((30, 30), bg, np.int8)
            for i, r, c in placed:
                ph, pw = packets[i].shape
                grid[r:r+ph, c:c+pw] = packets[i]
            return grid if np.count_nonzero(grid == bg)*2 > grid.size else None
        for i in remaining:
            ph, pw = packets[i].shape
            rows = sorted({0, 30-ph} | {r+packets[j].shape[0]+1 for j, r, c in placed})
            cols = sorted({0, 30-pw} | {c+packets[j].shape[1]+1 for j, r, c in placed})
            for r, c in product(rows, cols):
                if not (0 <= r <= 30-ph and 0 <= c <= 30-pw):
                    continue
                if any(not (r+ph < a or a+packets[j].shape[0] < r
                            or c+pw < b or b+packets[j].shape[1] < c)
                       for j, a, b in placed):
                    continue
                placed.append((i, r, c))
                answer = search([j for j in remaining if j != i])
                if answer is not None:
                    return answer
                placed.pop()
        return None

    answer = search(order)
    if answer is None:
        raise ValueError('Packets do not fit a 30 by 30 input')
    return answer


def _dock_positions(rng, count, two_dimensional):
    a, b = int(rng.integers(8, 10)), int(rng.integers(8, 10))
    if count == 2:
        if two_dimensional:
            sites = [(0, 0), (int(rng.integers(5, 10)), b)]
        elif rng.random() < .5:
            sites = [(0, 0), (0, b)]
        else:
            sites = [(0, 0), (a, 0)]
    else:
        sites = [(0, 0), (int(rng.integers(0, 2)), b), (a, int(rng.integers(0, 2)))]
        if count == 4:
            sites.append((a+int(rng.integers(0, 2)), b+int(rng.integers(0, 2))))
    return sites


def _candidate_meta(base, movers, anchor_colors, bg):
    def describe(cells):
        landmarks = []
        data = {}
        for group in _groups(cells):
            part = {p: cells[p] for p in group}
            if len(set(part.values())) > 1:
                landmarks.append(((min(r for r, c in part), min(c for r, c in part)),
                                  _fingerprint(part), set(part)))
            else:
                data.update(part)
        return dict(cells=cells, landmarks=landmarks, data=data)
    return dict(base=describe(base), movers=[describe(c) for c in movers],
                anchor_colors=set(anchor_colors), background=bg)


def _make_pair(rng, witness, test, mode):
    for attempt in range(100):
        palette = list(map(int, rng.permutation(10)))
        bg, frame, field, aa, ab = palette[:5]
        multi = (mode == 1 and (test or witness is None))
        categories = palette[5:8] if multi else palette[5:7]
        ignored = palette[8]
        ndocks = 2 if witness is not None else int(rng.integers(3, 5) if test else rng.integers(2, 4))
        sites = _dock_positions(rng, ndocks, test or (mode != 2 and rng.random() < .5))
        motif = _landmark(rng, (aa, ab))
        if witness == 1:
            # A narrow registration makes the two parent mechanisms easy to
            # separate: a nearby intact, unregistered placement has the same
            # census, while its landmark does not coincide with either copy.
            motif = {(0, 0): aa, (1, 0): ab, (2, 0): aa}
        base = {}
        for site in sites:
            _put(base, motif, site)
        mover_colors = [categories[:2], categories[1:3]] if multi else [categories]
        movers, shapes = [], []
        for index, colors in enumerate(mover_colors):
            card, object_shapes = dict(motif), {}
            for color in colors:
                slot = _SLOTS[categories.index(color)]
                pts = _object_shape(rng)
                if witness is not None:
                    pts = set(_SHAPES[3 + int(rng.integers(2))]) if color == categories[0] else {(0, 0)}
                object_shapes[color] = pts
                _put(card, {p: color for p in pts}, slot)
            movers.append(card)
            shapes.append(object_shapes)
        if witness is not None:
            codes = [{categories[0]}, {categories[1]}]
        else:
            possible_codes = [set(c for i, c in enumerate(categories) if bits & (1 << i))
                              for bits in range(2**len(categories))]
            rng.shuffle(possible_codes)
            codes = possible_codes[:ndocks]
            if len(codes) != ndocks or not set(categories) <= set().union(*codes):
                continue
        for si, (site, code) in enumerate(zip(sites, codes)):
            for color in code:
                slot = _SLOTS[categories.index(color)]
                source = next(s[color] for s in shapes if color in s)
                if witness is not None and si == 1:
                    pts = {(1, 1)}  # Meets the mover's singleton only diagonally.
                elif witness is not None or rng.random() < .45:
                    pts = set(source)
                else:
                    # An irregular observed subset: merger changes identity, not area.
                    choices = sorted(source)
                    pts = {choices[int(rng.integers(len(choices)))]}
                _put(base, {p: color for p in pts}, (site[0]+slot[0], site[1]+slot[1]))
            if len(categories) == 2:
                # Additional separate objects, sometimes unkeyed despite matching shapes.
                color = ignored if rng.random() < .40 else int(rng.choice(categories))
                _put(base, {p: color for p in _object_shape(rng)},
                     (site[0]+_SLOTS[2][0], site[1]+_SLOTS[2][1]))
        meta = _candidate_meta(base, movers, (aa, ab), bg)
        if len(meta['base']['landmarks']) != ndocks or any(len(c['landmarks']) != 1 for c in meta['movers']):
            continue
        candidates = _registered_candidates(meta)
        if len(candidates) < 2:
            continue
        by_profile = {}
        for candidate in candidates:
            profile = tuple(candidate['counts'].get(c, 0) for c in categories)
            by_profile.setdefault(profile, []).append(candidate)
        choices = [members[0] for profile, members in by_profile.items()
                   if len(members) == 1 and min(profile) > 0 and max(profile) <= 8]
        if witness is not None:
            desired = (sites[witness][0], sites[witness][1])
            choices = [c for c in choices if c['offsets'] == (desired,)]
        if not choices:
            continue
        selected = choices[int(rng.integers(len(choices)))]
        targets = {c: selected['counts'][c] for c in categories}
        # Construct the label from selected material, not from solve or parsed output.
        output = _render(selected['cells'], bg)
        try:
            packets = [_card(c, bg, frame, rng) for c in [base]+movers]
            packets.append(_make_chart(targets, bg, field, rng))
            grid = _pack(packets, bg, rng)
            rotation = int(rng.integers(4))
            if rng.random() < .5:
                grid, output = np.fliplr(grid), np.fliplr(output)
            grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
            parsed = _inspect(grid)
            if not np.array_equal(parsed['expected'], output):
                continue
            if witness == 1 and parsed['selected']['overlap'] >= max(c['overlap'] for c in parsed['candidates']):
                continue
            return {'input': grid, 'output': output}
        except ValueError:
            continue
    raise RuntimeError('Bounded construction could not find an inferable census')


def _legal_counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError(name + ' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _legal_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    mode = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = index if split == 'train' and index < 2 else None
            for _ in range(6):
                pair = _make_pair(rng, witness, split == 'test', mode)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not make distinct inputs')
    rng.shuffle(puzzle['train'])
    return puzzle


def _raw_counts(meta):
    result = {}
    for card in [meta['base']]+meta['movers']:
        for color, count in _counts(card['data'], set()).items():
            result[color] = result.get(color, 0) + count
    return result


def _without_landmark_witness(meta):
    """For a one-mover demonstration, find a census-correct unregistered union."""
    if len(meta['movers']) != 1:
        return False
    card = meta['movers'][0]
    selected = meta['selected']['cells']
    registered = {c['offsets'][0] for c in meta['candidates']}
    alternatives = sorted({(r+dr, c+dc) for r, c in registered
                           for dr in range(-4, 5) for dc in range(-4, 5)})
    expected_patterns = sorted(pattern for observed in [meta['base'], card]
                               for origin, pattern, pts in observed['landmarks'])
    # Keep the landmarks themselves intact and separate; remove only the
    # requirement that they register. Thus the ablation is not missing pixels.
    for dr, dc in alternatives:
        if (dr, dc) in registered:
            continue
        union = dict(meta['base']['cells'])
        valid = True
        for (r, c), color in card['cells'].items():
            p = r+dr, c+dc
            if p in union and union[p] != color:
                valid = False
                break
            union[p] = color
        if not valid or union == selected or max(_shape(_normalize(union))) > 30:
            continue
        anchor_points = {p for p, v in union.items() if v in meta['anchor_colors']}
        data_points = set(union)-anchor_points
        if any((r+a, c+b) in data_points for r, c in anchor_points
               for a in (-1, 0, 1) for b in (-1, 0, 1)):
            continue
        actual = sorted(_fingerprint({p: union[p] for p in group})
                        for group in _groups(anchor_points))
        if actual != expected_patterns:
            continue
        counts = _counts(union, meta['anchor_colors'])
        if all(counts.get(c, 0) == n for c, n in meta['targets'].items()):
            return True
    return False


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError('Puzzle must have exactly train and test')
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    _legal_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _checked_grid(pair['input']), _checked_grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen, 'Duplicate input'
            seen.add(key)
            meta = _inspect(inp)
            assert np.array_equal(out, meta['expected']), 'Wrong census-selected union or crop'
            assert len(meta['candidates']) >= 2, 'Census contributes no selection'
            assert any(any(c['counts'].get(v, 0) != n for v, n in meta['targets'].items())
                       for c in meta['candidates']), 'Omitting B must leave another answer'
            assert set(np.unique(out)) - {meta['background']} == set(meta['selected']['cells'].values())
            # Crop has material on every side, and no chart/frame colors remain.
            assert all(np.any(edge != meta['background']) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            assert not ({meta['frame'], meta['field']} & set(map(int, out.ravel())))
            if split == 'train':
                training.append(meta)
    assert any(m['selected']['overlap'] < max(c['overlap'] for c in m['candidates']) for m in training), 'Greedy overlap shortcut not distinguished'
    assert any(any(_raw_counts(m).get(c, 0) != n for c, n in m['targets'].items())
               for m in training), 'Counting loose objects not distinguished'
    assert any(any(_counts(m['selected']['cells'], m['anchor_colors'], False).get(c, 0) != n
                       for c, n in m['targets'].items()) for m in training), 'Diagonal joining not demonstrated'
    assert any(_without_landmark_witness(m) for m in training), 'Omitting A must change allowed answers'


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    def key(p):
        return tuple(tuple((_grid_key(q['input']), _grid_key(q['output'])) for q in p[s])
                     for s in ('train', 'test'))
    assert len({key(p) for p in puzzles}) == 42
    assert key(generate_puzzle(0)) == key(puzzles[0])
    pairs = [q for p in puzzles for s in ('train', 'test') for q in p[s]]
    metas = [_inspect(q['input']) for q in pairs]
    assert {m['background'] for m in metas} == set(range(10))
    assert {m['frame'] for m in metas} == set(range(10))
    assert {m['field'] for m in metas} == set(range(10))
    assert len({q['input'].shape for q in pairs}) >= 20
    assert len({q['output'].shape for q in pairs}) >= 20
    assert {len(m['movers']) for m in metas} >= {1, 2}
    assert {len(m['base']['landmarks']) for m in metas} >= {2, 3, 4}
    assert {len(m['targets']) for m in metas} >= {2, 3}
    assert len({pattern for m in metas for origin, pattern, pts in m['base']['landmarks']}) >= 40
    assert len({tuple(sorted(_normalize({p: 1 for p in group}))) for m in metas
                for group in _groups(m['selected']['cells'])}) >= 25
    assert any(len({p[0] for p, pattern, pts in m['base']['landmarks']}) > 1
               and len({p[1] for p, pattern, pts in m['base']['landmarks']}) > 1 for m in metas)
    assert any(set(m['selected']['counts']) - set(m['targets']) for m in metas)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1042+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.2, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal count accepted')
    # Input-only translation, palette and frame-margin invariance; no cached state.
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        inp, out = pair['input'], pair['output']
        saved = inp.copy()
        for k in range(4):
            assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
        permutation = np.array([4, 1, 9, 0, 6, 2, 8, 5, 3, 7], np.int8)
        assert np.array_equal(solve(permutation[inp]), permutation[out])
        assert np.array_equal(solve(inp.astype(np.uint64)), out)
        assert np.array_equal(inp, saved), 'Solver mutated input'
    # A compatible member beyond the sampler's two-mover range: three
    # identical observations are fully absorbed at the census-selected dock.
    # This fixture uses explicit material, not the generation candidate helper.
    bg, frame, field = 0, 1, 2
    motif = {(0, 0): 3, (0, 1): 4, (1, 0): 3}
    base = dict(motif)
    _put(base, motif, (0, 8))
    _put(base, {p: 5 for p in _SHAPES[3]}, (4, 0))
    base[5, 13] = 6
    # Two large hollow data objects also look like rectangles. Their empty
    # interiors have no landmarks, so they must not be mistaken for frames.
    ring = {(r, c): 7 for r in range(5) for c in range(5)
            if r in (0, 4) or c in (0, 4)}
    _put(base, ring, (8, 0))
    _put(base, ring, (8, 8))
    mover = dict(motif)
    _put(mover, {p: 5 for p in _SHAPES[3]}, (4, 0))
    mover[4, 4] = 6
    fixture_rng = np.random.default_rng(3402)
    packets = [_card(c, bg, frame, fixture_rng) for c in [base, mover, mover, mover]]
    packets.append(_make_chart({5: 1, 6: 2}, bg, field, fixture_rng))
    fixture_input = _pack(packets, bg, fixture_rng)
    expected_cells = dict(base)
    expected_cells.update(mover)
    assert np.array_equal(solve(fixture_input), _render(expected_cells, bg))
    # A larger two-color landmark contains an apparent inner frame. Its
    # proposed frame color leaves the actual packet outlines unaccounted for,
    # so only the complete outer-frame interpretation is valid.
    nested = {(r, c): 3 for r in range(5) for c in range(5)
              if r in (0, 4) or c in (0, 4)}
    nested[1, 1], nested[2, 2] = 4, 3
    base, mover = dict(nested), dict(nested)
    _put(base, nested, (0, 12))
    _put(base, {p: 5 for p in _SHAPES[3]}, (7, 0))
    base[8, 20] = 6
    _put(mover, {p: 5 for p in _SHAPES[3]}, (7, 0))
    mover[7, 7] = 6
    packets = [_card(c, bg, frame, fixture_rng) for c in [base, mover]]
    packets.append(_make_chart({5: 1, 6: 2}, bg, field, fixture_rng))
    fixture_input = _pack(packets, bg, fixture_rng)
    expected_cells = dict(base)
    expected_cells.update(mover)
    assert np.array_equal(solve(fixture_input), _render(expected_cells, bg))
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        return
    raise AssertionError('Incorrect label accepted')
