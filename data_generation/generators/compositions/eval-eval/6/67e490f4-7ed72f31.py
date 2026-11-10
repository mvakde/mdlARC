"""Symmetry ballots: exact geometric registration followed by local plurality.

Standard library and NumPy only. No parent code, stored pairs, or sampling state
is used by the solver. Generation labels come from planned color ballots.
"""
from collections import Counter
from functools import lru_cache

import numpy as np


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        groups.append(tuple(sorted(group)))
    return groups


def _normalize(cells):
    cells = tuple(cells)
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((int(r-r0), int(c-c0)) for r, c in cells))


def _size(cells):
    shape = _normalize(cells)
    return max(r for r, c in shape) + 1, max(c for r, c in shape) + 1


@lru_cache(maxsize=4096)
def _variants(shape):
    forms = set()
    for reflected in (False, True):
        current = _normalize((r, -c if reflected else c) for r, c in shape)
        for _ in range(4):
            forms.add(current)
            current = _normalize((c, -r) for r, c in current)
    return tuple(sorted(forms))


def _canonical(cells):
    return min(_variants(_normalize(cells)))


def _kind(cells):
    if len(cells) == 1:
        return 'P'
    rows, cols = zip(*cells)
    if len(set(rows)) == 1 and max(cols)-min(cols)+1 == len(cells):
        return 'H'
    if len(set(cols)) == 1 and max(rows)-min(rows)+1 == len(cells):
        return 'V'
    return None


def _hole_free(shape):
    h, w = _size(shape)
    mask = np.ones((h+2, w+2), bool)
    for r, c in _normalize(shape):
        mask[r+1, c+1] = False
    return len(_components(mask)) == 1


def _project(cells, marker, translate=False):
    kind, points = marker
    a, b = points[0]
    if kind == 'H':
        image = tuple((2*a-r, c) for r, c in cells)
    elif kind == 'V':
        image = tuple((r, 2*b-c) for r, c in cells)
    else:
        image = tuple((2*a-r, 2*b-c) for r, c in cells)
    if translate:
        dr = min(r for r, c in image) - min(r for r, c in cells)
        dc = min(c for r, c in image) - min(c for r, c in cells)
        image = tuple((r+dr, c+dc) for r, c in cells)
    return frozenset(image)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError('Expected an integer 1..30 by 1..30 color grid')
    return grid.astype(np.int8, copy=False)


def _read(input_grid):
    """All roles and positions come from this input, with no train dependency."""
    grid = _grid(input_grid)
    candidates = []
    for color in np.unique(grid):
        for cells in _components(grid == color):
            rs, cs = zip(*cells)
            y, x, bottom, right = min(rs), min(cs), max(rs), max(cs)
            box = grid[y:bottom+1, x:right+1]
            if min(box.shape) < 3:
                continue
            if not (np.all(box[0] == color) and np.all(box[-1] == color)
                    and np.all(box[:, 0] == color) and np.all(box[:, -1] == color)):
                continue
            other = set(map(int, np.unique(box))) - {int(color)}
            if len(other) == 1 and np.count_nonzero(grid == color) == len(cells):
                candidates.append((int(color), other.pop(), y, x, bottom, right))
    assert len(candidates) == 1, 'Carrier rectangle is not unique'
    carrier, bg, y, x, bottom, right = candidates[0]
    box = grid[y:bottom+1, x:right+1]
    local_cavities = _components(box == bg)
    assert local_cavities, 'No cavities'
    cavities = [frozenset((r+y, c+x) for r, c in cells) for cells in local_cavities]
    exterior = grid.copy()
    exterior[y:bottom+1, x:right+1] = bg
    by_color = {}
    for color in np.unique(exterior):
        if color != bg:
            by_color[int(color)] = _components(exterior == color)
    marker_colors = [color for color, groups in by_color.items()
                     if all(_kind(g) is not None for g in groups)]
    assert len(marker_colors) == 1, 'Exactly one straight-component marker color is required'
    marker_color = marker_colors[0]
    markers = [(_kind(cells), cells) for cells in by_color[marker_color]]
    objects = sorted([(color, cells) for color, groups in by_color.items()
                      if color != marker_color for cells in groups], key=lambda item: item[1])
    assert objects and all(_kind(cells) is None for color, cells in objects)
    return dict(grid=grid, box=box, bbox=(y, x, bottom, right),
                carrier=carrier, background=bg, marker_color=marker_color,
                cavities=cavities, local_cavities=local_cavities,
                markers=markers, objects=objects, exterior=exterior)


def _registrations(data, translate_kinds=()):
    addresses = {cells: i for i, cells in enumerate(data['cavities'])}
    votes = [Counter() for _ in data['cavities']]
    voters = [[] for _ in votes]
    productive_markers, productive_objects = set(), set()
    for index, (color, cells) in enumerate(data['objects']):
        recipients = set()
        for m, marker in enumerate(data['markers']):
            image = _project(cells, marker, marker[0] in translate_kinds)
            if image in addresses:
                recipients.add(addresses[image])
                productive_markers.add(m)
        for i in sorted(recipients):
            votes[i][color] += 1
            voters[i].append(index)
        if recipients:
            productive_objects.add(index)
    return votes, voters, productive_markers, productive_objects


def _fill(data, votes):
    output = data['box'].copy()
    winners = []
    for cells, counts in zip(data['local_cavities'], votes):
        assert counts, 'Unsupported cavity'
        maximum = max(counts.values())
        best = [color for color, n in counts.items() if n == maximum]
        assert len(best) == 1, 'Tied cavity ballot'
        winner = best[0]
        winners.append(winner)
        for r, c in cells:
            output[r, c] = winner
    return output, winners


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the family rule using only visible cells."""
    data = _read(input_grid)
    votes, _, _, _ = _registrations(data)
    return _fill(data, votes)[0]


def _alternative(data, method):
    try:
        if method == 'global_shape':
            ballots = {}
            for color, cells in data['objects']:
                ballots.setdefault(_canonical(cells), Counter())[color] += 1
            votes = [ballots.get(_canonical(cells), Counter()) for cells in data['cavities']]
        elif method == 'first':
            _, voters, _, _ = _registrations(data)
            votes = [Counter({data['objects'][v[0]][0]: 1}) if v else Counter()
                     for v in voters]
        elif method == 'bbox':
            def bounds(cells):
                return (min(r for r, c in cells), max(r for r, c in cells),
                        min(c for r, c in cells), max(c for r, c in cells))
            addresses = {bounds(c): i for i, c in enumerate(data['cavities'])}
            votes = [Counter() for c in data['cavities']]
            for color, cells in data['objects']:
                recipients = {addresses[bounds(image)] for m in data['markers']
                              if (image := _project(cells, m)) and bounds(image) in addresses}
                for i in recipients:
                    votes[i][color] += 1
        else:
            votes = _registrations(data, (method,))[0]
        return _fill(data, votes)[0]
    except AssertionError:
        return None


def _fails(data, expected, method):
    other = _alternative(data, method)
    return other is None or not np.array_equal(other, expected)


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _shape_pool(rng):
    shapes = set()
    for a in range(2, 5):
        for b in range(2, 5):
            cells = {(r, 0) for r in range(a)} | {(a-1, c) for c in range(b)}
            shapes.add(_canonical(cells))
    for _ in range(200):
        cells = {(0, 0)}
        for _ in range(int(rng.integers(3, 9))-1):
            frontier = sorted({(r+dr, c+dc) for r, c in cells
                               for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))} - cells)
            cells.add(_choice(rng, frontier))
        shape = _canonical(cells)
        if max(_size(shape)) <= 4 and _kind(shape) is None and _hole_free(shape):
            shapes.add(shape)
    return tuple(sorted(shapes))


def _asymmetric(shape, kind):
    image = _project(shape, (kind, ((0, 0),)))
    return _normalize(image) != _normalize(shape)


def _block(available, cells):
    h, w = available.shape
    for r, c in cells:
        available[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False


def _legal_positions(available, shape):
    sh, sw = _size(shape)
    nr, nc = available.shape[0]-sh+1, available.shape[1]-sw+1
    if min(nr, nc) <= 0:
        return np.empty((0, 2), int)
    valid = np.ones((nr, nc), bool)
    for r, c in shape:
        valid &= available[r:r+nr, c:c+nc]
    return np.argwhere(valid)


def _cavity_layout(rng, shapes, kinds, attempt, teaching):
    variants = [_choice(rng, _variants(s)) for s in shapes]
    if teaching:
        # First two are exactly the same silhouette; rotations change later.
        variants[1] = variants[0] if teaching == 1 else _choice(rng, _variants(shapes[1]))
    point_band = kinds == ('P',) and teaching == 1
    if kinds == ('H',) or kinds == ('V',) or point_band:
        # Compact axis-parallel band leaves room for 3/4 independent ballots.
        if kinds == ('V',):
            variants = [_normalize((c, r) for r, c in v) for v in variants]
        hh = max(_size(v)[0] for v in variants)
        widths = [_size(v)[1] for v in variants]
        gaps = [(4-widths[i]) if point_band else int(rng.integers(1, 3))
                for i in range(len(variants)-1)]
        sh = hh+2+int(rng.integers(0, 2))
        sw = sum(widths)+sum(gaps)+2+int(rng.integers(0, 3))
        placements, x = [], 1
        row = int(rng.integers(1, sh-hh))
        for i, variant in enumerate(variants):
            placements.append(tuple((r+row, c+x) for r, c in variant))
            x += widths[i] + (gaps[i] if i < len(gaps) else 0)
        if kinds == ('V',):
            placements = [tuple((c, r) for r, c in cells) for cells in placements]
            sh, sw = sw, sh
        height = int(rng.integers(28, 31)) if kinds == ('H',) or point_band else max(18, sh+int(rng.integers(3, 9)))
        width = int(rng.integers(28, 31)) if kinds == ('V',) else max(18, sw+int(rng.integers(3, 9)))
        height, width = min(height, 30), min(width, 30)
        if sh > height or sw > width:
            return None
        y, x = (height-sh)//2+int(rng.integers(-2, 3)), (width-sw)//2+int(rng.integers(-2, 3))
        y, x = max(0, min(height-sh, y)), max(0, min(width-sw, x))
        return height, width, sh, sw, y, x, placements
    height, width = int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if len(shapes) == 1:
        height, width = int(rng.integers(18, 31)), int(rng.integers(18, 31))
    if attempt >= 30:
        height = width = 30
    if len(shapes) == 1:
        vh, vw = _size(variants[0])
        sh, sw = vh+int(rng.integers(2, 7)), vw+int(rng.integers(2, 7))
    else:
        sh, sw = _choice(rng, ((8, 10), (10, 8), (9, 9), (11, 10), (10, 11), (8, 12), (12, 8)))
        sh += int(rng.integers(0, 2))
        sw += int(rng.integers(0, 2))
    available = np.ones((sh, sw), bool)
    available[[0, -1], :] = False
    available[:, [0, -1]] = False
    placements = [None] * len(shapes)
    for i in sorted(range(len(shapes)), key=lambda i: -len(shapes[i])):
        options = _legal_positions(available, variants[i])
        if not len(options):
            return None
        r, c = map(int, _choice(rng, options))
        cells = tuple((a+r, b+c) for a, b in variants[i])
        placements[i] = cells
        _block(available, cells)
    if rng.random() < .75:
        y = (height-sh)//2+int(rng.integers(-3, 4))
        x = (width-sw)//2+int(rng.integers(-3, 4))
        y, x = max(0, min(height-sh, y)), max(0, min(width-sw, x))
    else:
        y, x = int(rng.integers(height-sh+1)), int(rng.integers(width-sw+1))
        edge = int(rng.integers(5))
        if edge == 0:
            y = 0
        elif edge == 1:
            x = 0
        elif edge == 2:
            y = height-sh
        elif edge == 3:
            x = width-sw
    return height, width, sh, sw, y, x, placements


def _place_operator(rng, kind, targets, available, colors, target_ids, existing):
    h, w = available.shape
    if kind == 'H':
        centers = [(int(r), 0) for r in rng.permutation(h)]
    elif kind == 'V':
        centers = [(0, int(c)) for c in rng.permutation(w)]
    else:
        centers = [(int(i)//w, int(i)%w) for i in rng.permutation(h*w)]
    # Prefer a productive reuse when another marker can route an existing
    # colored object to a different cavity. Coordinates are still exact.
    preferred = []
    for target, color, recipient in zip(targets, colors, target_ids):
        for source, (old_color, old_recipients) in existing.items():
            if old_color != color or recipient in old_recipients:
                continue
            twice_a = min(r for r, c in source)+max(r for r, c in target)
            twice_b = min(c for r, c in source)+max(c for r, c in target)
            if kind == 'H' and twice_a % 2 == 0:
                preferred.append((twice_a//2, 0))
            elif kind == 'V' and twice_b % 2 == 0:
                preferred.append((0, twice_b//2))
            elif kind == 'P' and twice_a % 2 == twice_b % 2 == 0:
                preferred.append((twice_a//2, twice_b//2))
    centers = list(dict.fromkeys(preferred + centers))
    for a, b in centers:
        if not (0 <= a < h and 0 <= b < w):
            continue
        abstract = kind, ((a, b),)
        sources = [tuple(sorted(_project(cells, abstract))) for cells in targets]
        if any(any(not (0 <= r < h and 0 <= c < w) for r, c in cells)
               or (not (frozenset(cells) in existing
                        and existing[frozenset(cells)][0] == color
                        and recipient not in existing[frozenset(cells)][1])
                   and any(not available[r, c] for r, c in cells))
               for cells, color, recipient in zip(sources, colors, target_ids)):
            continue
        temporary = available.copy()
        for source in sources:
            _block(temporary, source)
        if kind == 'P':
            points = ((a, b),)
            if not temporary[a, b]:
                continue
        else:
            length = int(rng.integers(2, 6))
            line_shape = tuple((0, c) for c in range(length)) if kind == 'H' else tuple((r, 0) for r in range(length))
            options = _legal_positions(temporary, line_shape)
            options = options[options[:, 0] == a] if kind == 'H' else options[options[:, 1] == b]
            if not len(options):
                continue
            r, c = map(int, _choice(rng, options))
            points = tuple((r+dr, c+dc) for dr, dc in line_shape)
        return (kind, points), sources
    return None


def _make_pair(rng, shapes, ballots, kinds, teaching=0):
    count = len(shapes)
    marker_count = max(map(sum, ballots))
    for attempt in range(160):
        layout = _cavity_layout(rng, shapes, kinds, attempt, teaching)
        if layout is None:
            continue
        height, width, sh, sw, y, x, local = layout
        roles = list(map(int, rng.permutation(10)))
        bg, carrier, marker_color = roles[:3]
        colors = roles[3:]
        winners = [int(_choice(rng, colors)) for _ in shapes]
        if teaching:
            winners[1] = int(_choice(rng, [c for c in colors if c != winners[0]]))
        operator_kinds = list(kinds) if marker_count >= len(kinds) else []
        while len(operator_kinds) < marker_count:
            least = min(operator_kinds.count(k) for k in kinds)
            eligible = [k for k in kinds if operator_kinds.count(k) == least]
            operator_kinds.append(_choice(rng, eligible))
        rng.shuffle(operator_kinds)
        assignments = [[] for _ in range(marker_count)]
        planned = []
        # Ensure repeated-kind removal and singleton-kind removal both alter
        # a mandatory 2-1 training contest.
        loss_order = list(map(int, rng.permutation(marker_count)))
        if teaching == 1 and kinds == ('P',):
            loss_order = [0, 2]
        if teaching == 1 and len(kinds) == 2:
            singleton = next(j for j, kind in enumerate(operator_kinds) if operator_kinds.count(kind) == 1)
            loss_order = [singleton, next(j for j in loss_order if j != singleton)]
        for i, ballot in enumerate(ballots):
            losers = [c for c in colors if c != winners[i]]
            rng.shuffle(losers)
            if teaching == 1:
                other_winner = winners[1-i]
                losers.remove(other_winner)
                losers.insert(0, other_winner)
            color_counts = Counter(dict(zip([winners[i]]+losers, ballot)))
            planned.append(color_counts)
            payload_colors = [c for c, n in color_counts.items() for _ in range(n)]
            rng.shuffle(payload_colors)
            indices = list(map(int, rng.choice(marker_count, len(payload_colors), replace=False)))
            if teaching == 1:
                indices = list(range(marker_count))
                loser = loss_order[i % len(loss_order)]
                payload_colors = [winners[i]] * marker_count
                payload_colors[loser] = losers[0]
            for j, color in zip(indices, payload_colors):
                assignments[j].append((i, color))
        assert all(assignments), 'Largest ballot must make every marker productive'
        targets = [tuple((r+y, c+x) for r, c in cells) for cells in local]
        available = np.ones((height, width), bool)
        available[max(0,y-1):min(height,y+sh+1), max(0,x-1):min(width,x+sw+1)] = False
        markers, sources, existing = [], [], {}
        # Most demanding operators first, while keeping randomized equal-size order.
        order = list(map(int, rng.permutation(marker_count)))
        order.sort(key=lambda j: -sum(len(targets[i]) for i, color in assignments[j]))
        failed = False
        for j in order:
            selected = [targets[i] for i, color in assignments[j]]
            source_colors = [color for i, color in assignments[j]]
            target_ids = [i for i, color in assignments[j]]
            result = _place_operator(rng, operator_kinds[j], selected, available, source_colors, target_ids, existing)
            if result is None:
                failed = True
                break
            marker, copies = result
            markers.append(marker)
            _block(available, marker[1])
            for (i, color), cells in zip(assignments[j], copies):
                key = frozenset(cells)
                if key not in existing:
                    sources.append((color, cells))
                    existing[key] = color, {i}
                else:
                    existing[key][1].add(i)
                _block(available, cells)
        if failed:
            continue
        box = np.full((sh, sw), carrier, np.int8)
        output = box.copy()
        for i, cells in enumerate(local):
            for r, c in cells:
                box[r, c] = bg
                output[r, c] = winners[i]
        if len(_components(box == carrier)) != 1:
            continue
        grid = np.full((height, width), bg, np.int8)
        grid[y:y+sh, x:x+sw] = box
        for kind, cells in markers:
            for p in cells:
                grid[p] = marker_color
        for color, cells in sources:
            for p in cells:
                grid[p] = color
        data = _read(grid)
        votes, _, pm, po = _registrations(data)
        # Reader sorts cavities spatially; compare by absolute cell set.
        expected_by_address = {frozenset(cells): planned[i] for i, cells in enumerate(targets)}
        if any(v != expected_by_address[c] for c, v in zip(data['cavities'], votes)):
            continue
        if len(pm) != marker_count or len(po) != len(sources):
            continue
        if teaching == 1:
            if not _fails(data, output, 'first'):
                continue
            if not all(_fails(data, output, k) for k in kinds):
                continue
        assert np.array_equal(_fill(data, votes)[0], output)
        return {'input': grid, 'output': output}
    raise RuntimeError(f'Bounded packing exhausted: kinds={kinds}, shapes={shapes}, ballots={ballots}, teaching={teaching}')


def _registration_witness(rng, kinds):
    """Three oriented L cavities show an exact-fit rejection, not a box match.

    A has a 2-1-1 contest; B and C each have single support. B's object also
    projects onto A's bounding box through C's marker, in the wrong orientation.
    It would tie A's ballot if that near-fit counted. All six objects and five
    markers remain productive. This teaching construction is then cropped and
    reflected as a complete scene, with fresh palette and line lengths.
    """
    kind = 'P' if 'P' in kinds else 'H'
    transpose = 'P' not in kinds and 'H' not in kinds
    palette = list(map(int, rng.permutation(10)))
    bg, carrier, mc = palette[:3]
    winner, loser, third = palette[3:6]
    forms = _variants(((0, 0), (0, 1), (1, 0)))
    a_shape = _choice(rng, forms)
    b_shape = _choice(rng, [s for s in forms if s != a_shape])
    c_shape = _choice(rng, forms)
    local = [tuple((r+dy, c+dx) for r, c in shape)
             for shape, dy, dx in ((a_shape, 1, 1), (b_shape, 5, 1), (c_shape, 1, 5))]
    targets = [tuple((r+12, c+11) for r, c in cells) for cells in local]
    axes = (7, 11, 18, 20, 9)
    markers = []
    for row in axes:
        if kind == 'P':
            points = ((row, 20),)
        else:
            length = int(rng.integers(2, 6))
            start = int(rng.integers(20, 31-length))
            points = tuple((row, c) for c in range(start, start+length))
        markers.append((kind, points))
    # This makes unregistered global shape voting have a unique winner, with
    # a concrete wrong fill, rather than relying only on its inability to tie-break.
    c_winner = _choice(rng, (winner, loser))
    a_colors = [winner, winner, loser, third]
    rng.shuffle(a_colors)
    sources = [(a_colors[j], _project(targets[0], markers[j])) for j in range(4)]
    sources += [(loser, _project(targets[1], markers[1])),
                (c_winner, _project(targets[2], markers[4]))]
    grid = np.full((30, 30), bg, np.int8)
    box = np.full((8, 8), carrier, np.int8)
    output = box.copy()
    for cells, color in zip(local, (winner, loser, c_winner)):
        for r, c in cells:
            box[r, c], output[r, c] = bg, color
    grid[12:20, 11:19] = box
    occupied = set()
    for color, cells in sources + [(mc, m[1]) for m in markers]:
        assert not occupied.intersection(cells)
        for p in cells:
            assert grid[p] == bg
            grid[p] = color
        occupied.update(cells)
    # Empty exterior margins may vary; the geometric cue is retained intact.
    top_slack = min(r for r, c in occupied)
    bottom_slack = 29-max(r for r, c in occupied)
    top, left = int(rng.integers(0, top_slack+1)), int(rng.integers(0, 7))
    right_slack = 29-max(c for r, c in occupied)
    bottom, right = int(rng.integers(0, bottom_slack+1)), int(rng.integers(0, right_slack+1))
    grid = grid[top:30-bottom, left:30-right].copy()
    if transpose:
        grid, output = grid.T.copy(), output.T.copy()
    if rng.random() < .5:
        grid, output = grid[::-1].copy(), output[::-1].copy()
    if rng.random() < .5:
        grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
    data = _read(grid)
    assert np.array_equal(_fill(data, _registrations(data)[0])[0], output)
    assert _fails(data, output, 'bbox')
    return {'input': grid, 'output': output}


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'),
                                   (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(int(seed))
    pools = [('H',), ('V',), ('P',), ('H', 'V'), ('H', 'P'), ('V', 'P'), ('H', 'V', 'P')]
    kinds = _choice(rng, pools)
    vocabulary = _shape_pool(rng)
    small = [s for s in vocabulary if max(_size(s)) <= 3]
    anchors = [s for s in small if all(_asymmetric(s, k) for k in ('H', 'V', 'P'))]
    anchor = _choice(rng, anchors)
    puzzle, seen, known = {'train': [], 'test': []}, set(), set()
    options = [(1,), (2,), (2, 1), (3, 1), (3, 2), (2, 1, 1), (2, 1, 1, 1)]
    if len(kinds) == 1 and kinds[0] != 'P':
        options = [b for b in options if sum(b) <= 4]
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            teaching = i+1 if split == 'train' and i < 2 else 0
            if teaching:
                shapes = [anchor, anchor]
                ballots = [(2, 1), (2, 1)] if i == 0 else [(2, 1, 1), (2, 1, 1)]
            else:
                k = int(rng.integers(1, 4))
                candidates = small if k == 3 or (len(kinds) == 1 and kinds[0] != 'P') else vocabulary
                # Novel tests are supported by visible ballots, never a latent map.
                if split == 'test':
                    novel = [s for s in candidates if s not in known]
                    assert novel, 'Procedural vocabulary must retain test novelty'
                    shapes = [_choice(rng, novel)]
                else:
                    shapes = [_choice(rng, candidates)]
                while len(shapes) < k:
                    shapes.append(_choice(rng, candidates if rng.random() < .65 else shapes))
                usable = [b for b in options if sum(b) <= (3 if k == 3 else 4 if k == 2 else 5)]
                ballots = [_choice(rng, usable) for s in shapes]
            for _ in range(6):
                pair = (_registration_witness(rng, kinds) if teaching == 2 else
                        _make_pair(rng, shapes, ballots, kinds, teaching))
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not construct a distinct input')
            seen.add(key)
            puzzle[split].append(pair)
            if split == 'train':
                known.update(_canonical(c) for c in _read(pair['input'])['cavities'])
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'Repeated input'
            seen.add(_key(inp))
            data = _read(inp)
            votes, voters, pm, po = _registrations(data)
            prediction, winners = _fill(data, votes)
            assert np.array_equal(out, prediction), 'Wrong plurality, carrier or crop'
            assert len(pm) == len(data['markers']), 'Unproductive marker'
            assert len(po) == len(data['objects']), 'Unproductive object'
            exterior = data['exterior']
            assert len(_components(exterior != data['background'], True)) == len(data['objects']) + len(data['markers']), 'Exterior components touch'
            y, x, bottom, right = data['bbox']
            outside = exterior != data['background']
            assert not np.any(outside[max(0,y-1):min(inp.shape[0],bottom+2), max(0,x-1):min(inp.shape[1],right+2)]), 'Exterior touches carrier'
            mask = data['box'] == data['background']
            assert len(_components(mask, True)) == len(data['cavities']), 'Cavities touch diagonally'
            assert len(_components(data['box'] == data['carrier'])) == 1
            assert all(_hole_free(c) for c in data['cavities'])
            assert all(_hole_free(cells) for color, cells in data['objects'])
            assert np.all(out[data['box'] == data['carrier']] == data['carrier'])
            data.update(votes=votes, voters=voters, winners=winners, output=out)
            (training if split == 'train' else testing).append(data)
    trained = {m[0] for data in training for m in data['markers']}
    tested = {m[0] for data in testing for m in data['markers']}
    assert tested <= trained, 'Untrained symmetry operation'
    for kind in trained:
        assert any(_fails(d, d['output'], kind) for d in training), 'Translation hides a symmetry operation'
    assert any((other := _alternative(d, 'global_shape')) is not None
               and not np.array_equal(other, d['output']) for d in training), 'B has no concrete wrong-answer ablation'
    assert any(_fails(d, d['output'], 'first') for d in training), 'First-voter shortcut survives training'
    assert any(_fails(d, d['output'], 'bbox') for d in training), 'Bounding-box routing survives training'
    assert any(len(v) > 1 and max(v.values()) >= 2 for d in training for v in d['votes']), 'A has no color contest'
    assert any(len(v) >= 3 and max(v.values())*2 <= sum(v.values()) for d in training for v in d['votes']), 'No nonmajority plurality demonstration'
    repeated_winners = {}
    for d in training:
        for c, winner in zip(d['cavities'], d['winners']):
            repeated_winners.setdefault(_canonical(c), set()).add(winner)
    assert any(len(w) > 1 for w in repeated_winners.values()), 'Repeated shapes never have different answers'
    known = set(repeated_winners)
    assert all(any(_canonical(c) not in known for c in d['cavities']) for d in testing), 'Missing unseen-shape shift'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded checks on 42 seeds; reports are actual-input measurements."""
    signatures, dimensions, crops, silhouettes = set(), set(), set(), set()
    marker_mixtures, marker_counts, cavity_counts, ballots, roles = set(), set(), set(), set(), set()
    first_puzzle = None
    border_contacts = multi_recipient = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        if seed == 0:
            first_puzzle = puzzle
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read(pair['input'])
                votes, voters, _, _ = _registrations(data)
                assert np.array_equal(solve(pair['input']), pair['output'])
                dimensions.add(pair['input'].shape)
                crops.add(pair['output'].shape)
                marker_mixtures.add(tuple(sorted({k for k, m in data['markers']})))
                marker_counts.add(len(data['markers']))
                cavity_counts.add(len(data['cavities']))
                silhouettes.update(_canonical(c) for c in data['cavities'])
                roles.add((data['background'], data['carrier'], data['marker_color']))
                ballots.update(tuple(sorted(v.values(), reverse=True)) for v in votes)
                y, x, bottom, right = data['bbox']
                h, w = pair['input'].shape
                border_contacts += int(y == 0 or x == 0 or bottom == h-1 or right == w-1)
                usage = Counter(j for recipients in voters for j in recipients)
                multi_recipient += int(any(n > 1 for n in usage.values()))
                # Cross-check coordinate transforms against cropped bitmap flips.
                for color, cells in data['objects']:
                    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
                    hh, ww = _size(cells)
                    mask = np.zeros((hh, ww), bool)
                    for r, c in cells:
                        mask[r-r0, c-c0] = True
                    for marker in data['markers']:
                        kind, points = marker
                        a, b = points[0]
                        if kind == 'H':
                            turned, rr, cc = np.flipud(mask), 2*a-(r0+hh-1), c0
                        elif kind == 'V':
                            turned, rr, cc = np.fliplr(mask), r0, 2*b-(c0+ww-1)
                        else:
                            turned, rr, cc = np.rot90(mask, 2), 2*a-(r0+hh-1), 2*b-(c0+ww-1)
                        expected = frozenset((int(r)+rr, int(c)+cc) for r, c in np.argwhere(turned))
                        assert expected == _project(cells, marker)
    assert len(signatures) == 42
    assert len(dimensions) >= 25 and len(crops) >= 20
    assert len(silhouettes) >= 30 and len(roles) >= 60
    assert len(marker_mixtures) == 7
    assert {1, 2, 3} <= cavity_counts and {1, 3, 4, 5} <= marker_counts
    assert {(1,), (2, 1), (2, 1, 1), (3, 2)} <= ballots
    assert border_contacts > 0
    assert multi_recipient > 0, 'Shared objects must actually address several cavities'
    # A second separated segment on the same infinite axis must not double
    # count a source object at any cavity. This checks a broader rule case
    # without relying on the sampler accidentally duplicating an axis.
    duplicate_axis_checked = False
    for pair in first_puzzle['train']:
        data = _read(pair['input'])
        for kind, cells in data['markers']:
            if kind == 'P':
                continue
            available = np.ones(pair['input'].shape, bool)
            _block(available, tuple(map(tuple, np.argwhere(pair['input'] != data['background']).tolist())))
            line = ((0, 0), (0, 1)) if kind == 'H' else ((0, 0), (1, 0))
            positions = _legal_positions(available, line)
            coord = 0 if kind == 'H' else 1
            positions = positions[positions[:, coord] == cells[0][coord]]
            if not len(positions):
                continue
            r, c = map(int, positions[0])
            modified = pair['input'].copy()
            for dr, dc in line:
                modified[r+dr, c+dc] = data['marker_color']
            before = _registrations(data)[0]
            after = _registrations(_read(modified))[0]
            assert before == after
            assert np.array_equal(solve(modified), pair['output'])
            duplicate_axis_checked = True
            break
        if duplicate_axis_checked:
            break
    assert duplicate_axis_checked
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(8128, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first_puzzle[s]]
              for s in ('train', 'test')}
    out = broken['test'][0]['output']
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted carrier accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                input_dimensions=len(dimensions), output_dimensions=len(crops),
                cavity_shape_classes=len(silhouettes), role_palettes=len(roles),
                operation_mixtures=len(marker_mixtures),
                marker_counts=sorted(marker_counts), cavity_counts=sorted(cavity_counts),
                ballot_patterns=sorted(ballots), stencil_border_contacts=border_contacts,
                scenes_with_multi_recipient_objects=multi_recipient,
                independent_solver=False)
