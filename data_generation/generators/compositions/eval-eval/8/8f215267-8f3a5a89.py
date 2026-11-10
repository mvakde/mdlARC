"""Tidal census: external component counts become island-blind inward contours.

Only NumPy and the standard library are used. No parent generator is imported.
The solver recovers every role and enclosure from the query input.
"""
from collections import deque
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
MARKER = 9
ISLAND = 5


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return g.astype(np.int8, copy=True)


def _flood(allowed, start):
    h, w = allowed.shape
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return reached
    reached[start] = True
    pending = [start]
    while pending:
        r, c = pending.pop()
        for dr, dc in N4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not reached[rr, cc]:
                reached[rr, cc] = True
                pending.append((rr, cc))
    return reached


def _components(mask):
    """Orthogonal components; diagonal-only contact does not join objects."""
    remaining = mask.copy()
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if remaining[r, c]:
            component = _flood(remaining, (r, c))
            remaining[component] = False
            result.append(component)
    return result


def _holes(mask):
    """The complement inaccessible from a conceptual padded outside."""
    allowed = np.pad(~mask, 1, constant_values=True)
    exterior = _flood(allowed, (0, 0))[1:-1, 1:-1]
    return ~exterior & ~mask


def _dilate(mask, neighbors=N8):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = mask.copy()
    for dr, dc in neighbors:
        result |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _erode(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    result = mask.copy()
    for dr, dc in N8:
        result &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _distance(exterior):
    """Eight-step distance from the outside, including across filled islands.

    A multi-source breadth-first search is deliberately different from the
    builder's repeated erosion label construction.
    """
    h, w = exterior.shape
    distance = np.full((h, w), -1, dtype=int)
    distance[exterior] = 0
    pending = deque((int(r), int(c)) for r, c in np.argwhere(exterior))
    while pending:
        r, c = pending.popleft()
        for dr, dc in N8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and distance[rr, cc] < 0:
                distance[rr, cc] = distance[r, c] + 1
                pending.append((rr, cc))
    return distance


def _inspect(input_grid):
    g = _grid(input_grid)
    colors, amounts = np.unique(g, return_counts=True)
    _require(np.count_nonzero(amounts == amounts.max()) == 1, "background must have a unique frequency maximum")
    bg = int(colors[np.argmax(amounts)])
    _require(bg not in (MARKER, ISLAND), "reserved color used as background")
    marker_cells = np.argwhere(g == MARKER)
    _require(len(marker_cells) == 1, "exactly one color-9 marker is required")
    marker = tuple(int(v) for v in marker_cells[0])
    frames, objects = [], []
    for color in colors:
        color = int(color)
        if color in (bg, MARKER, ISLAND):
            continue
        for mask in _components(g == color):
            inside = _holes(mask)
            record = {"color": color, "mask": mask, "inside": inside}
            if inside.any():
                _require(len(_components(inside)) == 1, "an enclosure must contain one room")
                _require(np.all(np.isin(g[inside], [bg, ISLAND, MARKER])), "room contains a category-colored object")
                frames.append(record)
            else:
                _require(mask.sum() <= 16, "external objects must be small solid components")
                objects.append(record)
    _require(1 <= len(frames) <= 3, "expected one through three enclosures")
    _require(len({f['color'] for f in frames}) == len(frames), "enclosure category colors must be distinct")
    all_inside = np.zeros_like(g, dtype=bool)
    all_frames = np.zeros_like(g, dtype=bool)
    for frame in frames:
        _require(not np.any(all_inside & (frame['mask'] | frame['inside'])), "nested or overlapping rooms")
        _require(not np.any(all_frames & frame['inside']), "nested rooms")
        all_inside |= frame['inside']
        all_frames |= frame['mask']
    for i, frame in enumerate(frames):
        _require(not any(np.any(_dilate(frame['mask']) & other['mask'])
                         for other in frames[i + 1:]), "enclosures must have a background moat")
    for obj in objects:
        _require(not np.any(obj['mask'] & all_inside), "source object is inside an enclosure")
        _require(not np.any(_dilate(all_frames) & obj['mask']), "source object touches a wall")
    for i, obj in enumerate(objects):
        _require(not any(obj['color'] != other['color'] and np.any(_dilate(obj['mask']) & other['mask'])
                         for other in objects[i + 1:]), "different source colors must be separate")
    selected = [f for f in frames if f['inside'][marker]]
    _require(len(selected) == 1, "marker must be inside exactly one enclosure")
    frame = selected[0]
    target = _flood((g == bg) | (g == MARKER), marker)
    _require(np.array_equal(target, frame['inside'] & (g != ISLAND)), "islands must not partition the selected room")
    # This is B's external-complement operation: neutral islands are holes,
    # rather than new sources of boundary distance.
    exterior_padded = _flood(np.pad(~target, 1, constant_values=True), (0, 0))
    distance = _distance(exterior_padded)[1:-1, 1:-1]
    _require(np.array_equal(~exterior_padded[1:-1, 1:-1], frame['inside']), "the selected envelope is not island-blind")
    count = sum(o['color'] == frame['color'] for o in objects)
    _require(count <= 4, "at most four matching source components are supported")
    for depth in range(1, 2 * count, 2):
        _require(np.any(target & (g == bg) & (distance == depth)), "count exceeds visible tide-line capacity")
    for island in _components(g == ISLAND):
        _require(not _holes(island).any(), "neutral islands must be solid")
        containing = [f for f in frames if np.all(f['inside'][island])]
        _require(len(containing) == 1, "each neutral island must lie in one room")
        _require(np.all(containing[0]['inside'][_dilate(island)]), "island must be separate from its enclosing wall")
    return dict(grid=g, background=bg, marker=marker, frames=frames, objects=objects,
                frame=frame, target=target, exterior=exterior_padded,
                distance=distance, count=count)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the selected enclosure and its component-controlled tide lines."""
    d = _inspect(input_grid)
    g, frame, bg = d['grid'], d['frame'], d['background']
    out = np.full_like(g, bg)
    out[frame['mask']] = frame['color']
    out[frame['inside'] & (g == ISLAND)] = ISLAND
    paint = (d['target'] & (g == bg) & (d['distance'] % 2 == 1)
             & (d['distance'] <= 2 * d['count'] - 1))
    out[paint] = frame['color']
    out[d['marker']] = MARKER
    return out


def _label(grid, bg, frame, inside, count, marker):
    """Builder labels from known geometry, using successive filled-room erosion."""
    out = np.full_like(grid, bg)
    out[frame] = grid[frame]
    out[inside & (grid == ISLAND)] = ISLAND
    current = inside.copy()
    for depth in range(1, 2 * count):
        smaller = _erode(current)
        if depth % 2:
            out[current & ~smaller & (grid == bg)] = int(grid[frame][0])
        current = smaller
    out[marker] = MARKER
    return out


def _shape(rng, area=None, simple=False):
    """Independent solid polyomino geometry, never given a category count."""
    if area is None:
        area = int(rng.integers(1, 5 if simple else 13))
    kind = int(rng.integers(4))
    if kind == 0 and area <= 8:
        cells = {(0, c) for c in range(area)}
    elif kind == 1:
        width = min(area, int(rng.integers(2, 5)))
        cells = {(i // width, i % width) for i in range(area)}
    elif kind == 2 and area <= 9:
        arm = int(rng.integers(1, area + 1))
        cells = {(0, c) for c in range(arm)} | {(r, 0) for r in range(area - arm + 1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N4} - cells)
            # Limit long skinny random walks, while retaining branches and bends.
            frontier = [p for p in frontier if -2 <= p[0] <= 2 and -2 <= p[1] <= 2]
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    cells = tuple(sorted((r - r0, c - c0) for r, c in cells))
    local = np.zeros((1 + max(r for r, c in cells), 1 + max(c for r, c in cells)), bool)
    for cell in cells:
        local[cell] = True
    if _holes(local).any():
        # A solid partial rectangle preserves area, without inventing an enclosure.
        cells = tuple((i // 3, i % 3) for i in range(area))
    return cells


def _place(rng, allowed, shape, greedy=False):
    h, w = allowed.shape
    sh, sw = 1 + max(r for r, c in shape), 1 + max(c for r, c in shape)
    if sh > h or sw > w:
        return None
    valid = np.ones((h - sh + 1, w - sw + 1), bool)
    for r, c in shape:
        valid &= allowed[r:r + h - sh + 1, c:c + w - sw + 1]
    positions = np.argwhere(valid)
    if not len(positions):
        return None
    r0, c0 = positions[0 if greedy else int(rng.integers(len(positions)))]
    mask = np.zeros_like(allowed)
    for r, c in shape:
        mask[int(r0) + r, int(c0) + c] = True
    return mask


def _room(rng, canvas, box, style, thick=1):
    r, c, h, w = box
    envelope = np.ones((h, w), bool)
    bite_h, bite_w = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if style in ('notch', 'double'):
        envelope[:bite_h, :bite_w] = False
        if style == 'double':
            envelope[-int(rng.integers(2, 5)):, -int(rng.integers(2, 5)):] = False
    elif style == 'waist':
        start = int(rng.integers(3, h - 5))
        envelope[start:start + int(rng.integers(2, 5)), -bite_w:] = False
    elif style == 'shoulder':
        envelope[:int(rng.integers(3, h // 2 + 1)), -bite_w:] = False
    # Independent room reflections move the bites without changing the scene.
    if rng.random() < .5:
        envelope = np.flipud(envelope)
    if rng.random() < .5:
        envelope = np.fliplr(envelope)
    inside = envelope.copy()
    for _ in range(thick):
        inside = _erode(inside)
    wall = envelope & ~inside
    if not inside.any() or len(_components(inside)) != 1 or len(_components(wall)) != 1:
        return None
    full_wall, full_inside, full_envelope = [np.zeros(canvas, bool) for _ in range(3)]
    full_wall[r:r + h, c:c + w] = wall
    full_inside[r:r + h, c:c + w] = inside
    full_envelope[r:r + h, c:c + w] = envelope
    return full_wall, full_inside, full_envelope


def _layout(rng, n, witness, fallback=False, minimum=9):
    if fallback:
        boxes = {1: [(4, 4, 22, 22)],
                 2: [(1, 1, 15, 12), (1, 16, 18, 13)],
                 3: [(1, 1, 12, 12), (1, 16, 12, 12), (16, 1, 12, 12)]}[n]
        return (30, 30), boxes
    if n == 1:
        h, w = int(rng.integers(minimum + 6, 31)), int(rng.integers(minimum + 6, 31))
        rh, rw = int(rng.integers(minimum, h - 3)), int(rng.integers(minimum, w - 3))
        return (h, w), [(int(rng.integers(1, h - rh)), int(rng.integers(1, w - rw)), rh, rw)]
    if n == 2:
        h, w = int(rng.integers(24, 31)), int(rng.integers(27, 31))
        widths = [int(rng.integers(11, 14)), int(rng.integers(11, 14))]
        gap = int(rng.integers(1, min(3, w - sum(widths)) + 1))
        left = int(rng.integers(0, w - sum(widths) - gap + 1))
        heights = [int(rng.integers(12, 15)), int(rng.integers(17, 20))] if witness else [int(rng.integers(12, 20)) for _ in range(2)]
        boxes = [(int(rng.integers(0, 4)), left, heights[0], widths[0]),
                 (int(rng.integers(0, 4)), left + widths[0] + gap, heights[1], widths[1])]
        return (h, w), boxes
    h, w = int(rng.integers(27, 31)), int(rng.integers(27, 31))
    sizes = [(int(rng.integers(11, 14)), int(rng.integers(11, 14))) for _ in range(3)]
    boxes = [(0, 0, *sizes[0]), (0, w - sizes[1][1], *sizes[1]),
             (h - sizes[2][0], int(rng.integers(0, 3)), *sizes[2])]
    return (h, w), boxes


def _scene(rng, n, selected, colors, bg, requested_count, witness, area,
           rotation, reflect, complex_style=False, fallback=False):
    minimum = 17 if requested_count == 4 else 13 if requested_count == 3 else 9
    canvas, boxes = _layout(rng, n, witness, fallback, minimum)
    g = np.full(canvas, bg, np.int8)
    rooms, occupied = [], np.zeros(canvas, bool)
    for i, (box, color) in enumerate(zip(boxes, colors)):
        style = 'notch' if witness == 1 and i == selected else (
            str(rng.choice(['double', 'waist', 'shoulder'])) if complex_style else
            str(rng.choice(['rect', 'rect', 'notch', 'shoulder'])))
        thick = 1 if n > 1 or witness or fallback else int(rng.choice([1, 1, 2]))
        room = _room(rng, canvas, box, style, thick)
        if room is None:
            return None
        wall, inside, envelope = room
        if np.any(occupied & envelope):
            return None
        occupied |= _dilate(envelope)
        g[wall] = color
        rooms.append(room)
    wall, inside, envelope = rooms[selected]
    distance = _distance(np.pad(~inside, 1, constant_values=True))[1:-1, 1:-1]
    capacity = min(4, (int(distance.max()) + 1) // 2)
    count = int(rng.integers(0, min(capacity, 3) + 1)) if requested_count is None else requested_count
    if count > capacity:
        return None
    # Islands occupy the filled envelope, but do not restart the tide sequence.
    island_occupied = np.zeros(canvas, bool)
    for i, room in enumerate(rooms):
        room_inside = room[1]
        deep = _distance(np.pad(~room_inside, 1, constant_values=True))[1:-1, 1:-1] >= 3
        ni = 1 if witness and i == selected else int(rng.integers(0, 3))
        for j in range(ni):
            shape = _shape(rng, int(rng.integers(1, 6 if witness else 10)), fallback)
            island = _place(rng, deep & room_inside & ~island_occupied, shape, fallback)
            if island is None:
                if witness and i == selected and j == 0:
                    return None
                break
            g[island] = ISLAND
            island_occupied |= _dilate(island)
    targets = inside & (g == bg)
    marker_candidates = targets & (distance >= 3 if witness == 1 else distance == 1 if witness == 2 else True)
    cells = np.argwhere(marker_candidates)
    if not len(cells):
        return None
    marker = tuple(int(v) for v in cells[int(rng.integers(len(cells)))])
    g[marker] = MARKER
    sources = []
    # Teaching pair 2 uses two corner-touching equal bars: four-connectivity
    # counts two even though eight-connectivity would count one. Total colored
    # source area equals teaching pair 1's single solid object.
    if witness == 2:
        length = area // 2
        bundle = tuple([(0, c) for c in range(length)] + [(1, length + c) for c in range(length)])
        sources.append((colors[selected], bundle))
    else:
        for _ in range(count):
            sources.append((colors[selected], _shape(rng, area if witness == 1 else None, fallback)))
    for i, color in enumerate(colors):
        if i != selected:
            unselected_count = int(rng.integers(1 if witness else 0, 4))
            for _ in range(unselected_count):
                sources.append((color, _shape(rng, simple=fallback)))
    distractor_pool = [c for c in range(9) if c not in (bg, ISLAND) and c not in colors]
    for _ in range(int(rng.integers(1 if witness else 0, 4))):
        sources.append((int(rng.choice(distractor_pool)), _shape(rng, simple=fallback)))
    if not sources:
        sources.append((int(rng.choice(distractor_pool)), _shape(rng, simple=fallback)))
    sources.sort(key=lambda item: -len(item[1]))
    for color, shape in sources:
        obj = _place(rng, ~occupied, shape, fallback)
        if obj is None:
            return None
        g[obj] = color
        occupied |= _dilate(obj)
    out = _label(g, bg, wall, inside, count, marker)
    inp, out = np.rot90(g, rotation).copy(), np.rot90(out, rotation).copy()
    if reflect:
        inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
    pair = {'input': inp, 'output': out}
    try:
        d = _inspect(inp)
        f = _features(d)
        if not np.array_equal(solve(inp), out):
            return None
        if witness == 1 and not (f['hole_witness'] and f['diagonal_rim'] and f['size_selection'] == 'small'):
            return None
        if witness == 2 and not (f['size_selection'] == 'large' and f['marker_depth'] == 1):
            return None
        if d['count'] != count:
            return None
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not 2 <= nt <= 10 or not 1 <= nq <= 4 or nt <= nq:
        raise ValueError("need 2..10 training pairs, 1..4 test pairs, and num_train > num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    order = [int(c) for c in rng.permutation([0, 1, 2, 3, 4, 6, 7, 8])]
    first_color, second_color, novel = order[:3]
    area = int(rng.choice([4, 6, 8]))
    base_rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    shift = str(rng.choice(['depth', 'multiplicity', 'geometry']))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_bg = None
    for index in range(nt + nq):
        train = index < nt
        local = index if train else index - nt
        witness = index + 1 if index < 2 else 0
        n = 2 if witness or (not train and shift == 'geometry') else (
            1 if not train and shift == 'depth' else 3 if not train and shift == 'multiplicity'
            else int(rng.integers(1, 3)))
        selected = 0 if witness == 1 else 1 if witness == 2 else int(rng.integers(n))
        category = first_color if witness == 1 else second_color if witness == 2 else novel if not train and local == 0 else None
        pool = [c for c in order if not (train and c == novel)]
        bg = int(rng.choice([c for c in order if c != category and c != previous_bg]))
        pool = [c for c in pool if c != bg]
        colors = [int(c) for c in rng.choice(pool, n, replace=False)]
        if category is not None:
            if category in colors:
                j = colors.index(category)
                colors[j], colors[selected] = colors[selected], colors[j]
            else:
                colors[selected] = category
        requested = 1 if witness == 1 else 2 if witness == 2 else (
            0 if train and index == 2 else 3 if train and index == 3 else
            4 if not train and shift == 'depth' else None)
        rotation = base_rotation if witness else (base_rotation + 1) % 4 if not train else int(rng.integers(4))
        for attempt in range(64):
            pair = _scene(rng, n, selected, colors, bg, requested, witness, area,
                          rotation, reflect, not train and shift == 'geometry', attempt >= 48)
            if pair is None:
                continue
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("bounded scene packing failed")
        seen.add(key)
        puzzle['train' if train else 'test'].append(pair)
        previous_bg = bg
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(d):
    g, target, distance, frame = d['grid'], d['target'], d['distance'], d['frame']
    matching = [o['mask'] for o in d['objects'] if o['color'] == frame['color']]
    area = sum(int(mask.sum()) for mask in matching)
    rim = target & (distance == 1)
    orth_outside = _dilate(d['exterior'], N4)[1:-1, 1:-1]
    inner_islands = (g == ISLAND) & frame['inside']
    hole_neighbor = target & _dilate(inner_islands) & (g == d['background'])
    paint = target & (distance % 2 == 1) & (distance <= 2 * d['count'] - 1)
    sizes = [int(f['inside'].sum()) for f in d['frames']]
    own_size = int(frame['inside'].sum())
    size_selection = 'only' if len(sizes) == 1 else 'small' if own_size < max(sizes) else 'large' if own_size > min(sizes) else 'equal'
    center = np.argwhere(frame['inside']).mean(axis=0)
    position = tuple(float((center[i] + .5) / g.shape[i]) for i in range(2))
    masks = []
    for o in d['objects']:
        coords = np.argwhere(o['mask'])
        coords -= coords.min(axis=0)
        masks.append(tuple(tuple(int(v) for v in row) for row in coords))
    diagonal_merge = any(np.any(_dilate(a) & b) for i, a in enumerate(matching) for b in matching[i + 1:])
    return dict(shape=g.shape, rooms=len(d['frames']), color=frame['color'], background=d['background'],
                count=d['count'], source_area=area, object_shapes=masks,
                max_depth=int(distance.max()), marker_depth=int(distance[d['marker']]),
                size_selection=size_selection, position=position,
                islands=len(_components(inner_islands)), hole_witness=bool(np.any(hole_neighbor & ~paint)),
                diagonal_rim=int(np.count_nonzero(rim & ~orth_outside & (g == d['background']))),
                diagonal_merge=diagonal_merge,
                nonrectangular=not np.all(frame['inside'][np.ix_(np.flatnonzero(frame['inside'].any(axis=1)), np.flatnonzero(frame['inside'].any(axis=0)))]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal example counts")
    seen, train_features, train_categories, test_categories = set(), [], set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "dimensions must stay unchanged")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            _require(np.array_equal(solve(inp), out), "wrong selection, erasure, or tide lines")
            _require(not np.array_equal(inp, out), "identity pair")
            # Direct family invariants, beyond solver equality.
            f = d['frame']
            _require(np.array_equal(inp[f['mask']], out[f['mask']]), "selected wall changed")
            keep = f['mask'] | f['inside']
            _require(np.all(out[~keep] == d['background']), "objects or other rooms survived")
            _require(out[d['marker']] == MARKER, "marker was overwritten")
            neutral = f['inside'] & (inp == ISLAND)
            _require(np.all(out[neutral] == ISLAND), "selected islands changed")
            if split == 'train':
                train_features.append(_features(d))
                train_categories.update(frame['color'] for frame in d['frames'])
            else:
                test_categories.add(d['frame']['color'])
    _require({1, 2} <= {f['count'] for f in train_features}, "missing one/two tide progression")
    _require(len({f['color'] for f in train_features}) >= 2, "selected category colors must vary")
    _require(len({f['background'] for f in train_features}) >= 2, "training backgrounds must vary")
    _require(bool(test_categories - train_categories), "test must select a new local category")
    _require(any(a['source_area'] == b['source_area'] and a['count'] != b['count']
                 for a in train_features for b in train_features), "training permits total-area counting")
    _require(any(f['diagonal_merge'] and f['count'] == 2 for f in train_features), "four-connected object evidence missing")
    _require(any(f['hole_witness'] for f in train_features), "island-blind boundary evidence missing")
    _require(any(f['diagonal_rim'] and f['count'] for f in train_features), "corner-inclusive contour evidence missing")
    _require({'small', 'large'} <= {f['size_selection'] for f in train_features}, "selection-size witnesses missing")
    _require(any(f['marker_depth'] == 1 for f in train_features)
             and any(f['marker_depth'] >= 3 for f in train_features), "marker placement witnesses missing")
    _require(any((a['position'][axis] < .5 < b['position'][axis])
                 for a in train_features for b in train_features for axis in (0, 1)), "training does not oppose fixed-side selection")


def _signature(puzzle):
    return tuple((split, np.asarray(pair['input']).shape, np.asarray(pair['input']).tobytes(),
                  np.asarray(pair['output']).tobytes()) for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded diversity checks plus geometry/count/selection metamorphic checks."""
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend(_features(_inspect(pair['input'])) for split in p for pair in p[split])
    _require(len({_signature(p) for p in puzzles}) == 42, "cross-seed repetition")
    _require({r['rooms'] for r in records} == {1, 2, 3}, "room multiplicity diversity")
    _require({r['count'] for r in records} >= {0, 1, 2, 3, 4}, "count diversity")
    _require(len({r['shape'] for r in records}) >= 25, "canvas variation")
    _require(len({shape for r in records for shape in r['object_shapes']}) >= 50, "object geometry variation")
    _require(len({r['background'] for r in records}) == 8, "background role variation")
    _require(len({r['color'] for r in records}) == 8, "category role variation")
    _require(any(r['islands'] == 0 for r in records) and any(r['islands'] >= 2 for r in records), "island multiplicity variation")
    _require(sum(r['nonrectangular'] for r in records) >= 50, "irregular boundary variation")
    _require(_signature(generate_puzzle(137, 7, 3)) == _signature(generate_puzzle(137, 7, 3)), "determinism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 10 + nq, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Input-only equivariance includes every geometry, marker and category role.
    pair = next(p for p in puzzles[0]['train'] if _inspect(p['input'])['count'] == 2)
    inp, out = pair['input'].copy(), pair['output']
    for turn in range(4):
        for flip in (False, True):
            a, b = np.rot90(inp, turn), np.rot90(out, turn)
            if flip:
                a, b = np.fliplr(a), np.fliplr(b)
            _require(np.array_equal(solve(a), b), "rotation/reflection covariance")
    before = inp.copy()
    solve(inp)
    _require(np.array_equal(inp, before), "solver mutated input")
    _metamorphic_checks(puzzles[0])
    corrupted = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    d = _inspect(corrupted['test'][0]['input'])
    corrupted['test'][0]['output'][d['marker']] = d['background']
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        return
    raise AssertionError("validator accepted marker corruption")


def _metamorphic_checks(puzzle):
    pair = next(p for p in puzzle['train'] if _inspect(p['input'])['count'] == 1)
    inp, baseline = pair['input'].copy(), pair['output'].copy()
    d = _inspect(inp)
    bg, color, marker = d['background'], d['frame']['color'], d['marker']
    # Relocating the marker within its room changes only the old/new marker cells.
    new = tuple(int(v) for v in np.argwhere(d['target'] & (inp == bg))[0])
    moved = inp.copy()
    moved[marker], moved[new] = bg, MARKER
    expected = baseline.copy()
    expected[marker] = color if d['distance'][marker] == 1 else bg
    expected[new] = MARKER
    _require(np.array_equal(solve(moved), expected), "within-room marker movement")
    # Changing source area while keeping its component count leaves all output unchanged.
    source = next(o['mask'] for o in d['objects'] if o['color'] == color)
    occupied = inp != bg
    candidates = _dilate(source, N4) & ~source & (inp == bg)
    changed = False
    for r, c in np.argwhere(candidates):
        rr, cc = int(r), int(c)
        point = np.zeros_like(source)
        point[rr, cc] = True
        if np.any(_dilate(point) & (occupied & ~source)) or source.sum() >= 16:
            continue
        larger = inp.copy()
        larger[rr, cc] = color
        try:
            predicted = solve(larger)
        except AssertionError:
            continue
        _require(np.array_equal(predicted, baseline), "area shortcut changed answer")
        changed = True
        break
    _require(changed, "source enlargement witness unavailable")
    # Moving the marker across a wall must select the other category and erase this one.
    other = next(f for f in d['frames'] if f is not d['frame'])
    position = tuple(int(v) for v in np.argwhere(other['inside'] & (inp == bg))[0])
    across = inp.copy()
    across[marker], across[position] = bg, MARKER
    result = solve(across)
    _require(np.all(result[d['frame']['mask']] == bg), "marker ignored during enclosure selection")
    _require(np.all(result[other['mask']] == other['color']), "new selected enclosure was erased")
    _require(not np.array_equal(result, baseline), "omitting B does not change answer")
    # Add a new separate matching singleton: one extra distance band must appear.
    forbidden = _dilate(inp != bg)
    all_rooms = np.zeros_like(source)
    for f in d['frames']:
        all_rooms |= f['mask'] | f['inside']
    places = np.argwhere(~forbidden & ~all_rooms)
    _require(len(places) > 0, "no free external source position")
    more = inp.copy()
    more[tuple(places[0])] = color
    result = solve(more)
    _require(np.any((d['distance'] == 3) & (inp == bg) & (result == color) & (baseline == bg)), "omitting A does not change answer")
    # Removing a neutral island cannot introduce an island-centered tide line.
    island_mask = (inp == ISLAND) & d['frame']['inside']
    cleared = inp.copy()
    cleared[island_mask] = bg
    result = solve(cleared)
    _require(np.array_equal(result[~island_mask], baseline[~island_mask]), "island affected outer-distance geometry")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all explicit counts, family metamorphic checks)')
