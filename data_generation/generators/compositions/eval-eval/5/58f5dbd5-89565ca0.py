"""Color-paired portal stencils select a negative silhouette by room cardinality.

Generation uses a room-adjacency oracle; solve uses pixel flood filling. Neither
the decoder nor solve reads seeds, examples, files, or generation parameters.
"""
from collections import deque
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    """Four-connected components, ordered by their first row-major pixel."""
    labels = np.zeros(mask.shape, dtype=np.int16)
    cells = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if labels[y, x]:
            continue
        number = len(cells) + 1
        labels[y, x] = number
        queue = deque([(y, x)])
        group = []
        while queue:
            r, c = queue.popleft()
            group.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= rr < h and 0 <= cc < w and mask[rr, cc]
                        and not labels[rr, cc]):
                    labels[rr, cc] = number
                    queue.append((rr, cc))
        cells.append(group)
    return labels, cells


def _perimeter(h, w):
    result = np.zeros((h, w), dtype=bool)
    result[[0, -1], :] = True
    result[:, [0, -1]] = True
    return result


def _doors(wall, labels):
    """Eligible door pixels, grouped by the two original rooms they separate."""
    edges = {}
    h, w = wall.shape
    for y, x in np.argwhere(wall & ~_perimeter(h, w)):
        y, x = int(y), int(x)
        for a, b in ((labels[y-1, x], labels[y+1, x]),
                     (labels[y, x-1], labels[y, x+1])):
            if a and b and a != b:
                edge = tuple(sorted((int(a), int(b))))
                edges.setdefault(edge, []).append((y, x))
    return edges


def _decode(value):
    a = _grid(value)
    colors, freq = np.unique(a, return_counts=True)
    if len(colors) < 3 or np.count_nonzero(freq == freq.max()) != 1:
        raise ValueError("background must be uniquely dominant")
    bg = int(colors[np.argmax(freq)])
    candidates = []
    for marker in colors:
        if marker == bg:
            continue
        _, groups = _components(a == marker)
        charts = []
        for cells in groups:
            coords = np.asarray(cells)
            t, l = map(int, coords.min(axis=0))
            b, r = map(int, coords.max(axis=0) + 1)
            h, w = b-t, r-l
            if min(h, w) < 5:
                continue
            local = a[t:b, l:r]
            border = _perimeter(h, w)
            if not np.array_equal(local == marker, border):
                continue
            ink = [int(c) for c in np.unique(local[1:-1, 1:-1])
                   if c != bg]
            if len(ink) != 1 or ink[0] == marker:
                continue
            stencil = local[1:-1, 1:-1] == ink[0]
            charts.append(dict(color=ink[0], box=(t, l, b, r), stencil=stencil))
        if charts:
            candidates.append((int(marker), charts))
    if len(candidates) != 1:
        raise ValueError("register-frame color must be uniquely recognizable")
    marker, charts = candidates[0]
    if len({c['color'] for c in charts}) != len(charts):
        raise ValueError("one stencil register per payload color")
    chart_area = np.zeros(a.shape, dtype=bool)
    boxes = []
    for chart in charts:
        t, l, b, r = chart['box']
        chart_area[t:b, l:r] = True
        boxes.append(chart['box'])
    remaining = (a != bg) & (a != marker) & ~chart_area
    boards = []
    board_area = np.zeros(a.shape, dtype=bool)
    for color in np.unique(a[remaining]):
        color = int(color)
        coords = np.argwhere(remaining & (a == color))
        t, l = map(int, coords.min(axis=0))
        b, r = map(int, coords.max(axis=0) + 1)
        h, w = b-t, r-l
        local = a[t:b, l:r]
        if min(h, w) < 5 or not np.all(local[_perimeter(h, w)] == color):
            raise ValueError("board needs an intact enclosing perimeter")
        if not np.all(np.isin(local, (bg, marker, color))):
            raise ValueError("boards may not overlap")
        wall = local == color
        for y, x in np.argwhere(local == marker):
            y, x = int(y), int(x)
            if (not (0 < y < h-1 and 0 < x < w-1)
                    or not ((local[y-1, x] == color == local[y+1, x])
                            or (local[y, x-1] == color == local[y, x+1]))):
                raise ValueError("wall occlusion lacks opposite collinear support")
            wall[y, x] = True
        _, rooms = _components(~wall)
        if not rooms:
            raise ValueError("board must contain rooms")
        for cells in rooms:
            coords = np.asarray(cells)
            if len(cells) != np.prod(coords.max(axis=0) - coords.min(axis=0) + 1):
                raise ValueError("original elementary rooms must be rectangles")
        boundaries = []
        for cells in rooms:
            coords = np.asarray(cells)
            top, left = map(int, coords.min(axis=0)-1)
            bottom, right = map(int, coords.max(axis=0)+1)
            boundaries.append((top, left, bottom, right))
        if not np.array_equal(wall, _wall_from_rooms(h, w, boundaries)):
            raise ValueError("walls must be exactly the one-cell room boundaries")
        match = [c for c in charts if c['color'] == color]
        if len(match) != 1 or match[0]['stencil'].shape != (h-2, w-2):
            raise ValueError("matching register must fit the board's inset")
        stencil = np.zeros((h, w), dtype=bool)
        stencil[1:-1, 1:-1] = match[0]['stencil']
        labels, _ = _components(~wall)
        eligible = {p for positions in _doors(wall, labels).values() for p in positions}
        if any(tuple(map(int, p)) not in eligible for p in np.argwhere(stencil & wall)):
            raise ValueError("a wall mark must lie between two original rooms")
        boards.append(dict(color=color, box=(t, l, b, r), wall=wall,
                           stencil=stencil, damaged=int(np.count_nonzero(local == marker))))
        board_area[t:b, l:r] = True
        boxes.append((t, l, b, r))
    if not boards or np.any((a == marker) & ~chart_area & ~board_area):
        raise ValueError("unexplained marker cells or missing boards")
    for i, (t, l, b, r) in enumerate(boxes):
        for tt, ll, bb, rr in boxes[i+1:]:
            if not (b < tt or bb < t or r < ll or rr < l):
                raise ValueError("rectangles need at least one background gutter")
    return a, bg, marker, boards, charts


def _flood_choice(wall, stencil, *, by_area=False):
    original, rooms = _components(~wall)
    carved = wall & ~stencil
    regions, cavities = _components(~carved)
    weights, first = [], []
    for cells in cavities:
        ids = {int(original[y, x]) for y, x in cells} - {0}
        weights.append(len(cells) if by_area else len(ids))
        first.append(min((min(rooms[i-1]) for i in ids), default=wall.shape))
    winner = min(range(len(cavities)), key=lambda i: (-weights[i], first[i]))
    return regions == winner+1, dict(rooms=len(rooms), weights=tuple(weights),
                                    winner=winner, doors=int(np.count_nonzero(wall & stencil)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, _, boards, _ = _decode(input_grid)
    result = np.full(a.shape, bg, dtype=np.int8)
    for board in boards:
        holes, _ = _flood_choice(board['wall'], board['stencil'])
        t, l, b, r = board['box']
        result[t:b, l:r] = np.where(holes, bg, board['color'])
    return result


def _graph_holes(wall, stencil):
    """Generation oracle: union original room IDs across selected door edges."""
    labels, rooms = _components(~wall)
    parent = list(range(len(rooms)+1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    selected = []
    for (a, b), positions in _doors(wall, labels).items():
        active = [p for p in positions if stencil[p]]
        if active:
            parent[root(a)] = root(b)
            selected.extend((a, b, p) for p in active)
    groups = {}
    for i in range(1, len(rooms)+1):
        groups.setdefault(root(i), []).append(i)
    chosen = min(groups.values(), key=lambda ids: (-len(ids), min(min(rooms[i-1]) for i in ids)))
    holes = np.isin(labels, chosen)
    for a, b, p in selected:
        if a in chosen:
            assert b in chosen
            holes[p] = True
    return holes


def _wall_from_rooms(h, w, boxes):
    wall = np.zeros((h, w), dtype=bool)
    for t, l, b, r in boxes:
        wall[t, l:r+1] = wall[b, l:r+1] = True
        wall[t:b+1, l] = wall[t:b+1, r] = True
    return wall


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//2)*((r-l)//2)


def _partition(rng, h, w, count, style):
    boxes = [(0, 0, h-1, w-1)]
    if style == 'strip':
        axes = [axis for axis, dim in enumerate((h, w)) if (dim-1)//2 >= count]
        if axes:
            axis = axes[int(rng.integers(len(axes)))]
            gaps = np.full(count, 2)
            for _ in range((h if axis == 0 else w)-1-2*count):
                gaps[int(rng.integers(count))] += 1
            cuts = [0] + list(map(int, np.cumsum(gaps)))
            boxes = ([(cuts[i], 0, cuts[i+1], w-1) for i in range(count)]
                     if axis == 0 else [(0, cuts[i], h-1, cuts[i+1]) for i in range(count)])
            return _wall_from_rooms(h, w, boxes)
    # Capacity-aware random guillotine partitions; endpoints remain local.
    for step in range(count-1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                for cut in range(start+2, end-1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                    capacity += sum(_capacity(x) for x in halves)
                    if capacity >= count:
                        choices.append((i, halves))
        if not choices:
            raise AssertionError("capacity-aware partition stalled")
        i, halves = choices[int(rng.integers(len(choices)))]
        boxes[i:i+1] = halves
    return _wall_from_rooms(h, w, boxes)


def _area_witness(rng, h, w):
    # One spacious room above three thin, mutually connected rooms.
    gaps = np.full(3, 2)
    for _ in range(w-1-6):
        gaps[int(rng.integers(3))] += 1
    xs = [0] + list(map(int, np.cumsum(gaps)))
    cut = h-3
    boxes = [(0, 0, cut, w-1)] + [(cut, xs[i], h-1, xs[i+1]) for i in range(3)]
    wall = _wall_from_rooms(h, w, boxes)
    stencil = np.zeros_like(wall)
    stencil[h-2, xs[1]] = stencil[h-2, xs[2]] = True
    return wall, stencil


def _tie_witness(rng, h, w):
    # Two groups of three rooms, each with a triangular adjacency cycle.
    cut = 4
    yl, yr = 2, h-3
    xr = int(rng.integers(cut+2, w-2))
    boxes = [(0, 0, yl, cut), (yl, 0, h-1, 2), (yl, 2, h-1, cut),
             (0, cut, yr, xr), (0, xr, yr, w-1), (yr, cut, h-1, w-1)]
    wall = _wall_from_rooms(h, w, boxes)
    labels, _ = _components(~wall)
    stencil = np.zeros_like(wall)
    for positions in _doors(wall, labels).values():
        # The central separator remains closed; all six within-group edges open.
        choices = [p for p in positions if p[1] != cut]
        if choices:
            stencil[choices[int(rng.integers(len(choices)))]] = True
    return wall, stencil


def _random_stencil(rng, wall, force_door=False):
    labels, _ = _components(~wall)
    edges = list(_doors(wall, labels).values())
    stencil = np.zeros_like(wall)
    rate = float(rng.uniform(.1, .85))
    chosen = [i for i in range(len(edges)) if rng.random() < rate]
    if force_door and not chosen and edges:
        chosen = [int(rng.integers(len(edges)))]
    for i in chosen:
        positions = edges[i]
        amount = int(rng.integers(1, min(3, len(positions))+1))
        for j in rng.choice(len(positions), amount, replace=False):
            stencil[positions[int(j)]] = True
    return stencil


def _ink_and_damage(rng, wall, stencil, color, bg, marker, require_damage=False):
    # Marks already inside rooms are inert; explicit register frames expose offsets.
    free = np.argwhere(~wall)
    for i in rng.choice(len(free), int(rng.integers(1, min(6, len(free))+1)), replace=False):
        stencil[tuple(free[int(i)])] = True
    board = np.where(wall, color, bg).astype(np.int8)
    eligible = []
    for y, x in np.argwhere(wall & ~_perimeter(*wall.shape)):
        y, x = int(y), int(x)
        if ((wall[y-1, x] and wall[y+1, x] and not wall[y, x-1] and not wall[y, x+1])
                or (wall[y, x-1] and wall[y, x+1] and not wall[y-1, x] and not wall[y+1, x])):
            eligible.append((y, x))
    rng.shuffle(eligible)
    if require_damage:
        opened, _ = _components(~(wall & ~stencil))

        def joins_groups(p):
            y, x = p
            adjacent = {int(opened[yy, xx]) for yy, xx in
                        ((y-1, x), (y+1, x), (y, x-1), (y, x+1))} - {0}
            return len(adjacent) > 1

        # Occlusion is consequential: misreading a repair mark as a door would
        # connect coalitions that the external stencil leaves separate.
        eligible.sort(key=lambda p: (not joins_groups(p), bool(stencil[p])))
    amount = int(rng.integers(1 if require_damage else 0, min(3, len(eligible))+1)) if eligible else 0
    damaged = []
    for p in eligible:
        if len(damaged) >= amount:
            break
        if any(abs(p[0]-q[0])+abs(p[1]-q[1]) <= 1 for q in damaged):
            continue
        damaged.append(p)
        board[p] = marker
    chart = np.full(wall.shape, bg, dtype=np.int8)
    chart[_perimeter(*wall.shape)] = marker
    chart[stencil] = color
    truth = np.where(_graph_holes(wall, stencil), bg, color).astype(np.int8)
    return board, chart, truth


def _pack(rng, objects, bg):
    order = list(map(int, rng.permutation(len(objects))))
    options = []
    for nc in range(1, len(objects)+1):
        rows = [order[i:i+nc] for i in range(0, len(order), nc)]
        for gap in (1, 2):
            widths = [sum(objects[j][0].shape[1] for j in row)+(len(row)-1)*gap for row in rows]
            heights = [max(objects[j][0].shape[0] for j in row) for row in rows]
            h, w = sum(heights)+(len(rows)-1)*gap, max(widths)
            if max(h, w) <= 30:
                options.append((rows, gap, widths, heights, h, w))
    if not options:
        raise AssertionError("size policy failed to provide a fitting layout")
    rows, gap, widths, heights, h, w = options[int(rng.integers(len(options)))]
    inp, out = np.full((h, w), bg, dtype=np.int8), np.full((h, w), bg, dtype=np.int8)
    y = 0
    for row, rw, rh in zip(rows, widths, heights):
        x = int(rng.integers(w-rw+1))
        for j in row:
            tile, answer = objects[j]
            th, tw = tile.shape
            yy = y + int(rng.integers(rh-th+1))
            inp[yy:yy+th, x:x+tw] = tile
            if answer is not None:
                out[yy:yy+th, x:x+tw] = answer
            x += tw+gap
        y += rh+gap
    padding = []
    for dim in (h, w):
        extra = int(rng.integers(min(4, 30-dim)+1))
        if rng.random() < .15:
            extra = 30-dim
        before = int(rng.integers(extra+1))
        padding.append((before, extra-before))
    return dict(input=np.pad(inp, padding, constant_values=bg),
                output=np.pad(out, padding, constant_values=bg))


def _make_pair(rng, roles, witness=None, test=False):
    bg, marker = roles
    n = 2 if witness else int(rng.integers(1, 5))
    d = 0 if witness == 'tie' else int(rng.integers(0, (2 if n <= 2 else 1)+1))
    total = 2*n+d
    maximum = 13 if n == 1 else 11 if total <= 4 else 9 if total <= 7 else 8
    colors = list(map(int, rng.choice([c for c in range(10) if c not in roles], n+d, replace=False)))
    h0 = int(rng.integers(7, maximum+1))
    w0 = int(rng.integers(10 if witness == 'tie' else 7, maximum+1))
    if witness == 'tie' and h0 == w0:
        h0 = int(rng.integers(7, 10))
    objects = []
    for i, color in enumerate(colors[:n]):
        h, w = ((h0, w0) if witness else
                (int(rng.integers(7, maximum+1)), int(rng.integers(7, maximum+1))))
        if i == 0 and witness == 'area':
            wall, stencil = _area_witness(rng, h, w)
        elif i == 0 and witness == 'tie':
            wall, stencil = _tie_witness(rng, h, w)
        else:
            capacity = ((h-1)//2)*((w-1)//2)
            low, high = (7, min(9, capacity)) if test and i == 0 else (1, min(6 if not test else 9, capacity))
            count = int(rng.integers(low, high+1))
            wall = _partition(rng, h, w, count, 'strip' if rng.random() < .3 else 'mixed')
            stencil = _random_stencil(rng, wall, force_door=count > 1)
        # Transform board and chart together: their relative alignment stays direct.
        rotations = int(rng.integers(4)) if witness != 'tie' else 0
        wall, stencil = np.rot90(wall, rotations).copy(), np.rot90(stencil, rotations).copy()
        if witness != 'tie' and rng.random() < .5:
            wall, stencil = np.fliplr(wall).copy(), np.fliplr(stencil).copy()
        if witness == 'tie' and rng.random() < .5:
            wall, stencil = wall.T.copy(), stencil.T.copy()
        board, chart, truth = _ink_and_damage(rng, wall, stencil, color, bg, marker,
                                              require_damage=bool(witness and i == 0))
        objects.extend(((board, truth), (chart, None)))
    for color in colors[n:]:
        h, w = int(rng.integers(7, maximum+1)), int(rng.integers(7, maximum+1))
        chart = np.full((h, w), bg, dtype=np.int8)
        chart[_perimeter(h, w)] = marker
        chart[1:-1, 1:-1][rng.random((h-2, w-2)) < rng.uniform(.05, .4)] = color
        if not np.any(chart == color):
            chart[1, 1] = color
        objects.append((chart, None))
    return _pack(rng, objects, bg)


def _evidence(pairs):
    result = dict(stencil=False, area=False, tie=False, cycle=False, pairing=False,
                  repair=False, free_ink=False)
    orientation = {0, 1, 2, 3}
    for pair in pairs:
        a, bg, marker, boards, _ = _decode(pair['input'])
        for board in boards:
            wall, stencil = board['wall'], board['stencil']
            holes, info = _flood_choice(wall, stencil)
            untouched, _ = _flood_choice(wall, np.zeros_like(wall))
            area, _ = _flood_choice(wall, stencil, by_area=True)
            result['stencil'] |= not np.array_equal(holes, untouched)
            result['area'] |= not np.array_equal(holes, area)
            weights = info['weights']
            result['tie'] |= weights.count(max(weights)) > 1 and max(weights) >= 2
            labels, rooms = _components(~wall)
            edges = _doors(wall, labels)
            opened = [edge for edge, ps in edges.items() if any(stencil[p] for p in ps)]
            result['cycle'] |= len(opened) > len(rooms)-len(weights)
            t, l, b, r = board['box']
            false_doors = stencil | (a[t:b, l:r] == marker)
            result['repair'] |= not np.array_equal(_flood_choice(wall, false_doors)[0], holes)
            result['free_ink'] |= bool(np.any(stencil & ~wall))
            transforms = [stencil, np.fliplr(stencil), np.flipud(stencil), np.rot90(stencil, 2)]
            orientation &= {i for i, m in enumerate(transforms)
                            if np.array_equal(_flood_choice(wall, m)[0], holes)}
            for other in boards:
                if other is not board and other['stencil'].shape == stencil.shape:
                    # Swapped marks may hit junctions; the ablation is deliberately
                    # evaluated as raw subtraction, not accepted as a valid input.
                    alternative = other['stencil'].copy()
                    alternate_wall = wall & ~alternative
                    _, alt = _components(~alternate_wall)
                    if all(any(labels[p] for p in group) for group in alt):
                        result['pairing'] |= not np.array_equal(_flood_choice(wall, alternative)[0], holes)
    result['orientation'] = orientation == {0}
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, maximum_train = set(), 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a, bg, marker, boards, charts = _decode(inp)
            assert inp.shape == out.shape
            assert np.array_equal(solve(inp), out), "flood-fill answer disagrees"
            expected = np.full(inp.shape, bg, dtype=np.int8)
            room_counts = []
            for board in boards:
                wall, stencil = board['wall'], board['stencil']
                holes = _graph_holes(wall, stencil)
                assert np.array_equal(holes, _flood_choice(wall, stencil)[0]), "graph/flood disagreement"
                t, l, b, r = board['box']
                expected[t:b, l:r] = np.where(holes, bg, board['color'])
                assert not np.any(holes[_perimeter(*wall.shape)])
                _, rooms = _components(~wall)
                room_counts.append(len(rooms))
            assert np.array_equal(expected, out), "adjacency oracle disagrees"
            assert marker not in np.unique(out)
            assert not np.array_equal(inp, out)
            if split == 'train':
                maximum_train = max(maximum_train, max(room_counts))
            else:
                assert max(room_counts) > maximum_train, "missing chamber-count shift"
    assert all(_evidence(puzzle['train']).values()), "training misses a required contrast"


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    fixed_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    local_roles = bool(rng.random() < .8)
    puzzle, seen = {'train': [], 'test': []}, set()
    # The two anchors fit even the minimum training count. Retry geometry only
    # when its actual answers fail to expose one of the documented alternatives.
    for attempt in range(64):
        anchors = []
        for witness in ('area', 'tie'):
            roles = tuple(map(int, rng.choice(10, 2, replace=False))) if local_roles else fixed_roles
            anchors.append(_make_pair(rng, roles, witness=witness))
        if all(_evidence(anchors).values()):
            break
    else:
        raise RuntimeError('bounded anchor search could not expose all contrasts')
    puzzle['train'].extend(anchors)
    seen.update(_key(p['input']) for p in anchors)
    for split, number in (('train', num_train-2), ('test', num_test)):
        for _ in range(number):
            for attempt in range(32):
                roles = tuple(map(int, rng.choice(10, 2, replace=False))) if local_roles else fixed_roles
                pair = _make_pair(rng, roles, test=split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    # Exhaust all 16 portal subsets of a small four-room cycle, outside the
    # sampler's minimum scale. This also exercises a repaired but unopened door.
    small_wall = _perimeter(5, 5)
    small_wall[2, :] = True
    small_wall[:, 2] = True
    door_positions = [(1, 2), (2, 1), (2, 3), (3, 2)]
    for subset in range(16):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = True  # nonempty, inert ink supplies the stencil's color
        for i, p in enumerate(door_positions):
            mask[p] = bool(subset & (1 << i))
        board = np.where(small_wall, 2, 0).astype(np.int8)
        board[1, 2] = 5
        chart = np.zeros((5, 5), dtype=np.int8)
        chart[_perimeter(5, 5)] = 5
        chart[mask] = 2
        inp, expected = np.zeros((9, 15), dtype=np.int8), np.zeros((9, 15), dtype=np.int8)
        inp[1:6, 1:6], inp[1:6, 7:12] = board, chart
        expected[1:6, 1:6] = np.where(_graph_holes(small_wall, mask), 0, 2)
        assert np.array_equal(solve(inp), expected), 'small cycle regression'
    shapes, counts, palettes, clean_shapes, signatures = set(), set(), set(), set(), set()
    board_numbers, distractors, silhouettes, doors = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, bg, marker, boards, charts = _decode(pair['input'])
                shapes.add(a.shape)
                palettes.add((bg, marker))
                board_numbers.add(len(boards))
                distractors.add(len(charts)-len(boards))
                for board in boards:
                    wall, stencil = board['wall'], board['stencil']
                    hole, info = _flood_choice(wall, stencil)
                    counts.add(info['rooms'])
                    clean_shapes.add(_key(wall))
                    silhouettes.add(_key(hole))
                    doors.add(info['doors'])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in ('input', 'output'))
            # Literal color identities carry no rule. Permute every role at once.
            perm = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
            for p in puzzle['train'] + puzzle['test']:
                assert np.array_equal(solve(perm[p['input']]), perm[p['output']])
    assert len(signatures) == 42 and len(shapes) >= 30 and len(palettes) >= 30
    assert board_numbers == {1, 2, 3, 4} and distractors == {0, 1, 2}
    assert set(range(1, 10)) <= counts
    assert len(clean_shapes) >= 150 and len(silhouettes) >= 80 and len(doors) >= 10
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal example count accepted')
    mutated = generate_puzzle(0, 2, 1)
    mutated['test'][0]['output'][0, 0] = (int(mutated['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect answer accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, geometry and ablations')
