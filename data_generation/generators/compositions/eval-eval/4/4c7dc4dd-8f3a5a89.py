"""Transfer an input-local shell word between dotted compartments.

No parent code is imported.  Generation knows a recipe; solve recovers it from
the pixels.  Both use the specified external silhouette coordinate.
"""
from collections import deque
from pathlib import Path
import json

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
           if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    h, w = allowed.shape
    result = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return result
    todo = [start]
    result[start] = True
    while todo:
        r, c = todo.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and allowed[nr, nc]
                    and not result[nr, nc]):
                result[nr, nc] = True
                todo.append((nr, nc))
    return result


def _components(mask):
    remaining = mask.copy()
    result = []
    while remaining.any():
        p = tuple(map(int, np.argwhere(remaining)[0]))
        component = _flood(remaining, p)
        result.append(component)
        remaining[component] = False
    return result


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = mask.copy()
    for dr, dc in N8 if diagonal else N4:
        out |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _checkerboard(mask):
    a, b = mask[:-1, :-1], mask[:-1, 1:]
    c, d = mask[1:, :-1], mask[1:, 1:]
    return bool(np.any((a == d) & (b == c) & (a != b)))


def _distance_from(sources, diagonal=True):
    """Multi-source unit distance on the entire padded plane, holes included."""
    h, w = sources.shape
    distance = np.full((h, w), -1, dtype=np.int16)
    distance[sources] = 0
    queue = deque(map(tuple, np.argwhere(sources).tolist()))
    offsets = N8 if diagonal else N4
    while queue:
        r, c = queue.popleft()
        d = int(distance[r, c]) + 1
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and distance[nr, nc] < 0:
                distance[nr, nc] = d
                queue.append((nr, nc))
    return distance


def _depth(region, diagonal=True, include_holes=False):
    complement = np.pad(~region, 1, constant_values=True)
    exterior = complement if include_holes else _flood(complement, (0, 0))
    distance = _distance_from(exterior, diagonal)
    return distance[1:-1, 1:-1] - 1, exterior[1:-1, 1:-1]


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _decode(grid):
    h, w = grid.shape
    if min(h, w) < 5:
        raise ValueError("canvas is too small for the compartment convention")
    wall = int(grid[0, 0])
    if not (np.all(grid[[0, -1], :] == wall)
            and np.all(grid[:, [0, -1]] == wall)):
        raise ValueError("missing monochrome outer frame")
    values, counts = np.unique(grid, return_counts=True)
    most = values[counts == counts.max()]
    if len(most) != 1 or int(most[0]) == wall:
        raise ValueError("background must be uniquely dominant")
    bg = int(most[0])
    twice = values[counts == 2]
    if len(twice) != 1 or int(twice[0]) in (bg, wall):
        raise ValueError("expected one dot role occurring exactly twice")
    dot = int(twice[0])
    paints = set(map(int, values)) - {bg, wall, dot}
    if not 2 <= len(paints) <= 3 or any(np.count_nonzero(grid == p) <= 2 for p in paints):
        raise ValueError("expected two or three observable paint colors")
    positions = [tuple(map(int, p)) for p in np.argwhere(grid == dot)]
    regions = [_flood(grid != wall, p) for p in positions]
    if np.any(regions[0] & regions[1]):
        raise ValueError("the dots must designate different compartments")
    painted = [bool(np.any(np.isin(grid[m], list(paints)))) for m in regions]
    if sum(painted) != 1:
        raise ValueError("exactly one dotted compartment must be painted")
    ref_index = painted.index(True)
    ref, query = regions[ref_index], regions[1 - ref_index]
    if np.any(np.isin(grid[~ref], list(paints))):
        raise ValueError("paint outside the reference")
    rd, re = _depth(ref)
    qd, qe = _depth(query)
    word, stopped = [], False
    for depth in range(int(rd[ref].max()) + 1):
        cells = ref & (rd == depth) & (grid != dot)
        observed = np.unique(grid[cells])
        if len(observed) != 1:
            raise ValueError("reference shell is nonuniform or unobserved")
        color = int(observed[0])
        if color == bg:
            stopped = True
        elif color in paints and not stopped:
            word.append(color)
        else:
            raise ValueError("reference paint must be a finite external prefix")
    if not stopped or not 2 <= len(word) <= 5 or set(word) != paints:
        raise ValueError("missing finite word or stopping evidence")
    return dict(bg=bg, wall=wall, dot=dot, paints=paints, word=tuple(word),
                ref=ref, query=query, rd=rd, qd=qd, re=re, qe=qe,
                ref_dot=positions[ref_index], query_dot=positions[1 - ref_index])


def _paint(grid, region, depth, word, bg):
    out = grid.copy()
    for d, color in enumerate(word):
        out[region & (depth == d) & (grid == bg)] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the reference and complete the query, using only this input."""
    grid = _grid(input_grid)
    info = _decode(grid)
    return _paint(grid, info['query'], info['qd'], info['word'], info['bg'])


def _intrusion(rng, walls, region, minimum_depth):
    """Attach a small solid shape to the external wall, retaining one room."""
    _, exterior = _depth(region)
    available = np.argwhere(region & _dilate(walls & exterior, False))
    rng.shuffle(available)
    for r, c in available[:35]:
        cells = np.zeros_like(walls)
        cells[r, c] = True
        for _ in range(int(rng.integers(0, 5))):
            options = np.argwhere(_dilate(cells, False) & region & ~cells)
            if len(options):
                p = tuple(map(int, options[int(rng.integers(len(options)))]))
                cells[p] = True
        trial = walls | cells
        if _checkerboard(trial):
            continue
        remaining = region & ~cells
        if not remaining.any():
            continue
        connected = _flood(remaining, tuple(map(int, np.argwhere(remaining)[0])))
        if not np.array_equal(remaining, connected):
            continue
        depth, _ = _depth(remaining)
        if int(depth[remaining].max()) >= minimum_depth:
            return trial, remaining
    return walls, region


def _island(rng, walls, region, minimum_depth):
    depth, _ = _depth(region)
    locations = np.argwhere(region & (depth >= minimum_depth))
    rng.shuffle(locations)
    shapes = ([(0, 0)], [(0, 0), (0, 1)], [(0, 0), (1, 0)],
              [(0, 0), (1, 0), (0, 1)],
              [(0, 0), (0, 1), (1, 0), (1, 1)])
    shape = shapes[int(rng.integers(len(shapes)))]
    for r, c in locations:
        points = [(int(r) + dr, int(c) + dc) for dr, dc in shape]
        if any(x >= walls.shape[0] or y >= walls.shape[1] for x, y in points):
            continue
        cells = np.zeros_like(walls)
        for p in points:
            cells[p] = True
        if not np.all(region[_dilate(cells)]) or np.min(depth[cells]) < minimum_depth:
            continue
        remaining = region & ~cells
        if np.array_equal(_flood(remaining, tuple(map(int, np.argwhere(remaining)[0]))), remaining):
            return walls | cells, remaining, True
    return walls, region, False


def _word(rng, paints, length):
    p = list(map(int, paints[:min(len(paints), length)]))
    entries = p + [int(rng.choice(p)) for _ in range(length - len(p))]
    rng.shuffle(entries)
    return tuple(entries)


def _make_pair(rng, palette, split, witness, rotation):
    bg, wall, dot = map(int, palette[:3])
    for _ in range(100):
        if witness == 1:
            word = (int(palette[3]), int(palette[4]), int(palette[4]))
        elif witness == 2:
            word = (int(palette[4]), int(palette[3]))
        else:
            length = int(rng.integers(2, 6 if split == 'test' else 5))
            colors = int(rng.integers(2, min(3, length) + 1))
            word = _word(rng, palette[3:3 + colors], length)
        k = len(word)
        thick = int(rng.integers(1, 3))
        h = int(rng.integers(27 if witness == 1 else max(17, 2 * k + 7), 31))
        ref_min = 2 * k + 3
        query_min = max(9, 2 * k + 3, 11 if witness == 1 else 0)
        if witness == 2:
            rw = int(rng.integers(7, 10))
            qw = int(rng.integers(13, 29 - thick - rw))
        else:
            maximum_rw = min(14, 28 - thick - query_min)
            rw = int(rng.integers(ref_min, maximum_rw + 1))
            qw = int(rng.integers(query_min, 29 - thick - rw))
        w = rw + qw + thick + 2
        walls = np.zeros((h, w), dtype=bool)
        walls[[0, -1], :] = True
        walls[:, [0, -1]] = True
        base = rw + 1
        style = 'straight' if witness else str(rng.choice(
            ['lobed', 'stair'] if split == 'test' else ['straight', 'lobed', 'stair']))
        profile = np.full(h, base, dtype=int)
        if style == 'stair':
            cuts = sorted(set([1, h - 1] + list(map(int, rng.integers(3, h - 3, 3)))))
            for lo, hi in zip(cuts, cuts[1:]):
                profile[lo:hi] += int(rng.integers(-1, 2))
        for r in range(1, h - 1):
            c = int(profile[r])
            walls[r, c:c + thick] = True
            if r > 1:
                previous = int(profile[r - 1])
                walls[r, min(c, previous):max(c, previous) + thick] = True
        branch_count = 1 if witness == 1 else 0 if witness == 2 else int(rng.integers(0, 3))
        if witness == 1:
            # The undotted room is larger than the selected blank room.
            target_height = int(rng.integers(11, 13))
            cuts = [h - 2 - target_height]
        else:
            cuts = sorted(map(int, rng.choice(np.arange(5, h - 5),
                                             branch_count, replace=False)))
        for row in cuts:
            walls[row, int(profile[row]):w - 1] = True
        ref = _flood(~walls, (1, 1))
        if witness:
            query = _flood(~walls, (h - 2, w - 2))
        else:
            choices = []
            for room in _components(~walls):
                if np.any(room & ref):
                    continue
                depths, _ = _depth(room)
                if int(depths[room].max()) >= k:
                    choices.append(room)
            if not choices:
                continue
            query = choices[int(rng.integers(len(choices)))].copy()
        if np.any(ref & query):
            continue
        if witness == 1:
            r0 = int(np.argwhere(query)[:, 0].min())
            c0 = int(np.argwhere(query)[:, 1].min()) + qw // 2
            # A connected L intrusion exposes an edge-versus-corner witness.
            walls[r0:r0 + 2, c0] = True
            walls[r0, c0 + 1] = True
            query = _flood(~walls, (h - 2, w - 2))
        intrusion_count = int(rng.integers(1, 5 if split == 'test' else 3))
        if witness == 1:
            intrusion_count = 0
        for _ in range(intrusion_count):
            walls, query = _intrusion(rng, walls, query, k)
        if not witness and rng.random() < .6:
            walls, ref = _intrusion(rng, walls, ref, k)
        if style == 'lobed':
            for _ in range(int(rng.integers(1, 4))):
                walls, query = _intrusion(rng, walls, query, k)
        island_count = 1 if witness == 1 else int(rng.integers(0, 4))
        placed = 0
        for _ in range(island_count):
            walls, query, success = _island(rng, walls, query, k + 1 if witness == 1 else 2)
            placed += success
        if witness == 1 and not placed:
            continue
        if not witness and rng.random() < .4:
            walls, ref, _ = _island(rng, walls, ref, 2)
        if _checkerboard(walls):
            continue
        rd, _ = _depth(ref)
        qd, _ = _depth(query)
        if int(rd[ref].max()) < k or int(qd[query].max()) < k:
            continue
        if witness == 1 and query.sum() >= ref.sum():
            continue
        if witness == 2 and query.sum() <= ref.sum():
            continue
        reference_locations = np.argwhere(ref & (rd < int(rd[ref].max())))
        query_locations = np.argwhere(query & ((qd >= k) if witness == 1 else
                                               (qd == 0) if witness == 2 else True))
        if not len(query_locations):
            continue
        rp = tuple(map(int, reference_locations[int(rng.integers(len(reference_locations)))]))
        qp = tuple(map(int, query_locations[int(rng.integers(len(query_locations)))]))
        raw = np.full((h, w), bg, dtype=np.int8)
        raw[walls] = wall
        raw[rp] = raw[qp] = dot
        inp = _paint(raw, ref, rd, word, bg)
        output = _paint(inp, query, qd, word, bg)
        if rng.integers(2):
            inp, output = np.fliplr(inp).copy(), np.fliplr(output).copy()
        inp, output = np.rot90(inp, rotation).copy(), np.rot90(output, rotation).copy()
        try:
            info = _decode(inp)
        except ValueError:
            continue
        if witness == 1:
            f = _features(inp, output, info)
            if not (f['hole_ablation'] and f['orthogonal_ablation']
                    and f['hole_blank'] and f['larger_undotted']):
                continue
        return {'input': inp, 'output': output}
    raise RuntimeError("bounded construction could not produce a valid compartment analogy")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a = _count(num_train, 'num_train', 2, 10)
    b = _count(num_test, 'num_test', 1, 4)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = rng.permutation(10)
    varied = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', a), ('test', b)):
        for index in range(count):
            witness = index + 1 if split == 'train' and index < 2 else 0
            colors = palette if witness or not varied else rng.permutation(10)
            angle = (rotation + index) % 4 if witness else int(rng.integers(4))
            for _ in range(20):
                pair = _make_pair(rng, colors, split, witness, angle)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input sampling exhausted")
        rng.shuffle(puzzle[split])
    return puzzle


def _features(inp, out, info=None):
    info = _decode(inp) if info is None else info
    query, depth, word = info['query'], info['qd'], info['word']
    bg, wall, dot = info['bg'], info['wall'], info['dot']
    holes = (inp == wall) & ~info['qe']
    touching = _dilate(holes) & query & (inp != dot)
    hole_depth, _ = _depth(query, include_holes=True)
    orth_depth, _ = _depth(query, diagonal=False)
    rooms = _components(inp != wall)
    region_areas = [int(m.sum()) for m in rooms]
    undotted = [m for m in rooms if not np.any((inp == dot) & m)]
    islands = _components(holes)
    coordinates = np.argwhere(query)
    center = coordinates.mean(axis=0)
    h, w = inp.shape
    displacement = ((center[0] + .5) / h - .5, (center[1] + .5) / w - .5)
    side = ('top' if displacement[0] < 0 else 'bottom') if abs(displacement[0]) > abs(displacement[1]) else (
        'left' if displacement[1] < 0 else 'right')
    inner_pieces = [len(_components(query & (depth >= d)))
                    for d in range(1, len(word) + 1)]
    r0, c0 = coordinates.min(axis=0)
    r1, c1 = coordinates.max(axis=0) + 1
    silhouette = (~info['qe'])[r0:r1, c0:c1]
    profiles = []
    for axis in (silhouette, silhouette.T):
        spans = [(int(np.flatnonzero(row)[0]), int(np.flatnonzero(row)[-1]))
                 for row in axis if row.any()]
        profiles.append(sum(a != b for a, b in zip(spans, spans[1:])))
    normal = {}
    word_pattern = [normal.setdefault(c, len(normal)) for c in word]
    return dict(shape=list(inp.shape), word=list(word), word_length=len(word),
                word_pattern=word_pattern,
                roles=[bg, wall, dot] + sorted(info['paints']),
                compartments=len(region_areas), areas=region_areas,
                ref_area=int(info['ref'].sum()), query_area=int(query.sum()),
                smaller=bool(query.sum() < info['ref'].sum()),
                larger=bool(query.sum() > info['ref'].sum()),
                larger_undotted=any(m.sum() > query.sum() for m in undotted),
                query_side=side, query_bbox=[int(np.ptp(coordinates[:, 0])) + 1,
                                            int(np.ptp(coordinates[:, 1])) + 1],
                shell_piece_counts=inner_pieces,
                silhouette_mask=np.packbits(silhouette).tobytes().hex(),
                external_indent_area=int(silhouette.size - silhouette.sum()),
                profile_changes=sum(profiles),
                islands=len(islands), island_areas=[int(m.sum()) for m in islands],
                marker_depth=int(depth[info['query_dot']]),
                max_depth=int(depth[query].max()),
                hole_blank=bool(np.any(touching & (out == bg))),
                hole_ablation=not np.array_equal(out, _paint(inp, query, hole_depth, word, bg)),
                orthogonal_ablation=not np.array_equal(out, _paint(inp, query, orth_depth, word, bg)),
                repeated_band=any(x == y for x, y in zip(word, word[1:])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            assert not _checkerboard(inp == info['wall'])
            main_wall = _flood(inp == info['wall'], (0, 0))
            for region in _components(inp != info['wall']):
                assert np.any(region & _dilate(main_wall, False)), "nested free room"
            query, bg = info['query'], info['bg']
            assert np.array_equal(out, solve(inp))
            assert np.array_equal(inp[~query], out[~query])
            assert np.array_equal(inp[inp != bg], out[inp != bg])
            assert np.any(out != inp)
            assert int(info['qd'][query].max()) >= len(info['word'])
            if split == 'train':
                training.append(_features(inp, out, info))
    assert len({tuple(f['word']) for f in training}) >= 2
    assert any(f['repeated_band'] for f in training)
    assert any(f['hole_ablation'] and f['hole_blank'] for f in training)
    assert any(f['orthogonal_ablation'] for f in training)
    assert any(f['smaller'] for f in training) and any(f['larger'] for f in training)
    assert any(f['marker_depth'] == 0 for f in training)
    assert any(f['marker_depth'] >= f['word_length'] for f in training)
    assert any(f['compartments'] >= 3 for f in training)
    assert any(f['larger_undotted'] for f in training), "largest blank region shortcut"


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ('input', 'output')) for s in ('train', 'test'))


def _diagnostics(puzzle):
    pair = next(p for p in puzzle['train'] if (
        _features(p['input'], p['output'])['hole_blank']
        and len(_decode(p['input'])['word']) >= 3
        and set(_decode(p['input'])['word'][:-1]) == _decode(p['input'])['paints']))
    inp, out = pair['input'], pair['output']
    info = _decode(inp)
    bg, wall, dot = info['bg'], info['wall'], info['dot']
    # Full-input reference twins: query pixels remain exactly identical.
    swapped = inp.copy()
    a, b = info['word'][0], next(c for c in info['word'] if c != info['word'][0])
    swapped[info['ref'] & (inp == a)] = b
    swapped[info['ref'] & (inp == b)] = a
    assert np.array_equal(swapped[~info['ref']], inp[~info['ref']])
    assert not np.array_equal(solve(swapped), out)
    # Changing just the stopping depth also changes the answer.
    shortened = inp.copy()
    last = len(info['word']) - 1
    shortened[info['ref'] & (info['rd'] == last) & (inp != dot)] = bg
    assert not np.array_equal(solve(shortened), out)
    # Move the query dot: the shell geometry must be independent of it.
    candidates = np.argwhere(info['query'] & (inp == bg))
    moved = inp.copy()
    old = info['query_dot']
    new = tuple(map(int, candidates[0]))
    moved[old], moved[new] = bg, dot
    expected = out.copy()
    d = int(info['qd'][old])
    expected[old] = info['word'][d] if d < len(info['word']) else bg
    expected[new] = dot
    assert np.array_equal(solve(moved), expected)
    # Remove a hole: depths on every original query cell remain identical.
    holes = (inp == wall) & ~info['qe']
    no_hole = inp.copy()
    no_hole[holes] = bg
    without = solve(no_hole)
    assert np.array_equal(without[~holes], out[~holes])
    # Connect that island to the outside without changing the reference.
    connected_witness = False
    for hole in np.argwhere(holes):
        for dr, dc in N4:
            r, c = map(int, hole)
            path = []
            while 0 <= r < inp.shape[0] and 0 <= c < inp.shape[1] and not info['qe'][r, c]:
                path.append((r, c))
                r, c = r + dr, c + dc
            if any(inp[p] == dot for p in path):
                continue
            attached = inp.copy()
            for p in path:
                attached[p] = wall
            new_info = _decode(attached)
            new_cells = (attached == wall) & (inp != wall)
            if int(new_info['query'].sum()) != int(info['query'].sum() - new_cells.sum()):
                continue
            mask = info['query'] & ~new_cells & (inp != dot)
            if np.any(solve(attached)[mask] != out[mask]):
                connected_witness = True
                break
        if connected_witness:
            break
    assert connected_witness
    # Insert a deep island in that recovered silhouette.
    plain = _decode(no_hole)
    places = np.argwhere(plain['query'] & (plain['qd'] >= len(plain['word']) + 1)
                         & (no_hole == bg))
    if len(places):
        p = tuple(map(int, places[0]))
        island = no_hole.copy()
        island[p] = wall
        answer = solve(island)
        expected = without.copy()
        expected[p] = wall
        assert np.array_equal(answer, expected)
    # Whole-input rotations/reflections retain exact outputs and conventions.
    for angle in range(4):
        for reflect in (False, True):
            x, y = np.rot90(inp, angle).copy(), np.rot90(out, angle).copy()
            if reflect:
                x, y = np.fliplr(x).copy(), np.fliplr(y).copy()
            before = x.copy()
            assert np.array_equal(solve(x), y)
            assert np.array_equal(x, before)
    # A hand-sized depth regression isolates the outside/holes distinction.
    region = np.zeros((13, 15), bool)
    region[1:-1, 1:-1] = True
    clean_depth, _ = _depth(region)
    region[6, 7] = False
    hole_depth, _ = _depth(region)
    assert np.array_equal(clean_depth[region], hole_depth[region])
    assert hole_depth[6, 6] == 5
    region[1:7, 7] = False  # attach the island to the external frame
    attached_depth, _ = _depth(region)
    assert attached_depth[6, 6] == 0
    # Corners of an external intrusion need diagonal contact.
    region = np.zeros((9, 11), bool)
    region[1:-1, 1:-1] = True
    region[1:3, 5] = False
    diagonal, _ = _depth(region)
    orthogonal, _ = _depth(region, diagonal=False)
    assert diagonal[3, 6] == 0 and orthogonal[3, 6] == 1


def self_test():
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'], pair['output'])
                records.append({'seed': seed, 'split': split, **f})
        fs = [f for f in records if f['seed'] == seed]
        assert len({tuple(f['shape']) for f in fs}) >= 2
        assert len({tuple(f['word']) for f in fs}) >= 2
        assert len({f['compartments'] for f in fs}) >= 2
    assert len(signatures) == 42
    assert len({tuple(f['shape']) for f in records}) >= 40
    assert len({tuple(f['word']) for f in records}) >= 70
    assert len({tuple(f['word_pattern']) for f in records}) >= 8
    assert len({(tuple(f['query_bbox']), f['silhouette_mask']) for f in records}) >= 80
    assert len({f['profile_changes'] for f in records}) >= 4
    assert {f['query_side'] for f in records} == {'top', 'bottom', 'left', 'right'}
    assert max(n for f in records for n in f['shell_piece_counts']) >= 2
    assert {2, 3, 4, 5} <= {f['word_length'] for f in records}
    assert {2, 3, 4} <= {f['compartments'] for f in records}
    assert {0, 1, 2} <= {f['islands'] for f in records}
    assert len({f['query_area'] for f in records}) >= 70
    assert len({f['max_depth'] for f in records}) >= 3
    for role in range(3):
        assert len({f['roles'][role] for f in records}) == 10
    assert any(f['word_length'] == 5 and f['split'] == 'test' for f in records)
    assert _same_puzzle(generate_puzzle(189), generate_puzzle(189))
    _diagnostics(generate_puzzle(0))
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(990 + a + b, a, b)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(990 + a + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    summary = dict(seeds=42, pairs=len(records),
                   shapes=len({tuple(f['shape']) for f in records}),
                   words=len({tuple(f['word']) for f in records}),
                   normalized_words=len({tuple(f['word_pattern']) for f in records}),
                   silhouettes=len({(tuple(f['query_bbox']), f['silhouette_mask']) for f in records}),
                   profile_change_counts=sorted({f['profile_changes'] for f in records}),
                   shell_piece_counts=sorted({n for f in records for n in f['shell_piece_counts']}),
                   compartments=sorted({f['compartments'] for f in records}),
                   island_counts=sorted({f['islands'] for f in records}),
                   word_lengths=sorted({f['word_length'] for f in records}),
                   query_areas=len({f['query_area'] for f in records}),
                   semantic_verification='generator-owned checks only')
    # Local scratch artifacts are optional; self_test does not rely on them.
    directory = Path('scratch')
    directory.mkdir(exist_ok=True)
    (directory / 'self_test_audit.json').write_text(json.dumps(
        {'summary': summary, 'records': records}, separators=(',', ':')) + '\n')
    return summary


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
