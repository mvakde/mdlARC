"""Transfer maps: contract identical station motifs, route, and mark all views.

Only the input pixels determine identities, terminals, edges, and the answer.
Local fragments remain in place; matching is translation-only, not rotation.
"""
from collections import deque
import heapq
from itertools import product
import numpy as np

HIGHLIGHT_STATION = 3
HIGHLIGHT_PATH = 5
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
FRAME_SHAPES = ((4, 4), (4, 5), (5, 4), (5, 5))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('grid must be a 1..30 integer rectangle in colors 0..9')
    return a


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _rim(h, w):
    return {(r, c) for r in range(h) for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)}


def _shortest(adjacency, start, target):
    distance, ways, previous = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b in sorted(adjacency[a]):
            if b not in distance:
                distance[b] = distance[a] + 1
                ways[b] = 0
                previous[b] = a
                queue.append(b)
            if distance[b] == distance[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    if ways.get(target, 0) != 1:
        return None
    path = [target]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def _edge(a, b):
    return tuple(sorted((a, b)))


def _adjacency(nodes, edges):
    result = {n: set() for n in nodes}
    for a, b in edges:
        result[a].add(b)
        result[b].add(a)
    return result


def _weighted_path(adjacency, weights, start, target):
    best, previous = {start: 0}, {}
    queue = [(0, start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if best[a] != cost:
            continue
        if a == target:
            path = [target]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return cost, path[::-1]
        for b in sorted(adjacency[a]):
            candidate = cost + weights[_edge(a, b)]
            if candidate < best.get(b, float('inf')):
                best[b], previous[b] = candidate, a
                heapq.heappush(queue, (candidate, b))
    return float('inf'), []


def _read_input(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[counts.argmax()])
    if counts.max() * 2 <= grid.size or {3, 5} & set(colors.tolist()):
        raise ValueError('background must exceed half the canvas; 3 and 5 are reserved')
    height, width = grid.shape
    candidates = []
    for h, w in FRAME_SHAPES:
        rim = sorted(_rim(h, w))
        for r in range(height - h + 1):
            for c in range(width - w + 1):
                frame = int(grid[r, c])
                if frame == bg:
                    continue
                if not all(grid[r + u, c + v] == frame for u, v in rim):
                    continue
                interior = grid[r + 1:r + h - 1, c + 1:c + w - 1]
                if np.any(interior == bg) or np.any(interior == frame):
                    continue
                cells = {(r + u, c + v) for u in range(h) for v in range(w)}
                border = {(r + u, c + v) for u, v in rim}
                signature = (h, w, tuple(int(x) for x in grid[r:r+h, c:c+w].flat))
                candidates.append(dict(cells=cells, border=border, signature=signature,
                                       interior=tuple(int(x) for x in interior.flat), frame=frame))
    if len(candidates) < 6 or len({s['frame'] for s in candidates}) != 1:
        raise ValueError('station frames are missing or ambiguous')
    if len({s['signature'][:2] for s in candidates}) != 1:
        raise ValueError('one frame shape must be shared within an example')
    station_map, signatures = {}, sorted({s['signature'] for s in candidates})
    ids = {signature: i for i, signature in enumerate(signatures)}
    frame = candidates[0]['frame']
    for i, station in enumerate(candidates):
        if any(p in station_map for p in station['cells']):
            raise ValueError('overlapping station frames')
        for p in station['cells']:
            station_map[p] = i
        station['node'] = ids[station['signature']]
    if set(map(tuple, np.argwhere(grid == frame))) != set().union(*(s['border'] for s in candidates)):
        raise ValueError('frame-colored material outside station rims')
    outside = {(int(r), int(c)) for r, c in np.argwhere(grid != bg)} - set(station_map)
    outside_colors = {int(grid[p]) for p in outside}
    if len(outside_colors) != 1:
        raise ValueError('connections need one separate color')
    connector = next(iter(outside_colors))
    interiors = {s['node']: s['interior'] for s in candidates}
    code_colors = set().union(*(set(v) for v in interiors.values()))
    if len(code_colors) != 3 or connector in code_colors:
        raise ValueError('two data colors and a distinct terminal marker are required')
    markers = [color for color in code_colors
               if sum(color in v for v in interiors.values()) == 2
               and all(v.count(color) == 1 for v in interiors.values() if color in v)]
    if len(markers) != 1:
        raise ValueError('exactly two terminal identities must have the singleton marker')
    marker = markers[0]
    data_colors = code_colors - {marker}
    if not all(data_colors <= set(v) for v in interiors.values()):
        raise ValueError('both data colors must occur inside every station')
    terminals = sorted(n for n, v in interiors.items() if marker in v)
    connections, edges = [], set()
    for cells in _groups(outside):
        touches = {station_map[q] for r, c in cells for dr, dc in DIRECTIONS
                   for q in [(r+dr, c+dc)] if q in station_map}
        if len(touches) != 2:
            raise ValueError('a connector must meet exactly two station occurrences')
        if any(sum((r+dr, c+dc) in cells for dr, dc in DIRECTIONS) > 2 for r, c in cells):
            raise ValueError('connector junctions must be stations')
        a, b = sorted(touches)
        edge = _edge(candidates[a]['node'], candidates[b]['node'])
        if edge[0] == edge[1]:
            raise ValueError('self-connections are excluded')
        edges.add(edge)
        connections.append(dict(cells=cells, occurrences=(a, b), edge=edge))
    adjacency = _adjacency(range(len(signatures)), edges)
    reached, stack = {0}, [0]
    while stack:
        for b in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    if len(reached) != len(signatures):
        raise ValueError('contracted station network must be connected')
    path = _shortest(adjacency, *terminals)
    if path is None:
        raise ValueError('minimum-station route is tied or missing')
    path_edges = {_edge(a, b) for a, b in zip(path, path[1:])}
    return dict(grid=grid, background=bg, frame=frame, connector=connector,
                marker=marker, stations=candidates, connections=connections,
                adjacency=adjacency, edges=edges, terminals=terminals,
                path=path, path_edges=path_edges)


def _paint(data):
    result = data['grid'].copy()
    selected = set(data['path'][1:-1])
    for station in data['stations']:
        if station['node'] in selected:
            for p in station['border']:
                result[p] = HIGHLIGHT_STATION
    for connection in data['connections']:
        if connection['edge'] in data['path_edges']:
            for p in connection['cells']:
                result[p] = HIGHLIGHT_PATH
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _paint(_read_input(input_grid))


def _metrics(data):
    physical = _adjacency(range(len(data['stations'])),
                          [c['occurrences'] for c in data['connections']])
    starts = [i for i, s in enumerate(data['stations']) if s['node'] == data['terminals'][0]]
    targets = [i for i, s in enumerate(data['stations']) if s['node'] == data['terminals'][1]]
    physical_paths = [p for a in starts for b in targets if (p := _shortest(physical, a, b)) is not None]
    physical_best = min((len(p) - 2 for p in physical_paths), default=None)
    weights = {e: min(len(c['cells']) for c in data['connections'] if c['edge'] == e)
               for e in data['edges']}
    weighted, weighted_path = _weighted_path(data['adjacency'], weights, *data['terminals'])
    selected_area = sum(weights[e] for e in data['path_edges'])
    centers = [np.mean(sorted(s['cells']), axis=0) for s in data['stations']]
    lengths = {e: min(float(np.linalg.norm(centers[c['occurrences'][0]]-centers[c['occurrences'][1]]))
                      for c in data['connections'] if c['edge'] == e) for e in data['edges']}
    minimum_length, _ = _weighted_path(data['adjacency'], lengths, *data['terminals'])
    selected_length = sum(lengths[e] for e in data['path_edges'])
    interiors = {s['node']: s['interior'] for s in data['stations']}
    histograms = [tuple(sorted((v, code.count(v)) for v in set(code))) for code in interiors.values()]
    h, w = data['stations'][0]['signature'][:2]
    codes = set(interiors.values())
    reflected_pair = any(tuple(np.array(code).reshape(h-2, w-2)[:, ::-1].flat) != code
                         and tuple(np.array(code).reshape(h-2, w-2)[:, ::-1].flat) in codes for code in codes)
    pieces = _groups(set(map(tuple, np.argwhere(data['grid'] != data['background']))))
    selected_pieces = sum(any(c['edge'] in data['path_edges'] and c['cells'] <= piece
                              for c in data['connections']) for piece in pieces)
    # A selected station copy may have no selected local edge at all.
    passive_selected = sum(s['node'] in data['path'][1:-1]
                           and not any(i in c['occurrences'] and c['edge'] in data['path_edges']
                                       for c in data['connections'])
                           for i, s in enumerate(data['stations']))
    return dict(nodes=len(data['adjacency']), occurrences=len(data['stations']),
                pieces=len(pieces), selected_pieces=selected_pieces,
                hops=len(data['path'])-1, physical_best=physical_best,
                area_conflict=selected_area > weighted,
                distance_conflict=selected_length > minimum_length + 1e-9,
                histogram_collision=len(set(histograms)) < len(histograms),
                reflected_pair=reflected_pair,
                passive_selected=passive_selected,
                loop_rank=len(data['edges'])-len(data['adjacency'])+1,
                degrees=tuple(sorted(map(len, data['adjacency'].values()))),
                duplicate_edges=len(data['connections'])-len(data['edges']),
                frame_shape=data['stations'][0]['signature'][:2])


def _glyphs(rng, n, shape, roles, terminals):
    bg, connector, frame, a, b, marker = roles
    h, w = shape
    size = (h-2)*(w-2)
    patterns = [bits for bits in product((a, b), repeat=size) if len(set(bits)) == 2]
    pattern_ids = {p: i for i, p in enumerate(patterns)}
    asymmetric = [i for i, p in enumerate(patterns)
                  if tuple(np.array(p).reshape(h-2, w-2)[:, ::-1].flat) != p]
    first = int(rng.choice(asymmetric))
    reflected = tuple(np.array(patterns[first]).reshape(h-2, w-2)[:, ::-1].flat)
    second = pattern_ids[reflected]
    others = rng.choice([i for i in range(len(patterns)) if i not in (first, second)], n-2, replace=False)
    chosen = list(map(int, others[:2])) + [first, second] + list(map(int, others[2:]))
    # Nodes 2 and 3 are ordinary stations. Their equal histograms and mirrored
    # interiors require exact orientation-sensitive matching in every example.
    result = {}
    signatures = set()
    for node, index in enumerate(chosen):
        bits = list(patterns[int(index)])
        if node in terminals:
            eligible = [i for i, v in enumerate(bits) if bits.count(v) >= 2]
            # Avoid erasing the sole occurrence of either data color.
            rng.shuffle(eligible)
            for i in eligible:
                trial = bits.copy()
                trial[i] = marker
                if tuple(trial) not in signatures:
                    bits = trial
                    break
            else:
                raise ValueError('terminal code collision')
        signatures.add(tuple(bits))
        glyph = np.full(shape, frame, dtype=np.int8)
        glyph[1:-1, 1:-1] = np.array(bits, dtype=np.int8).reshape(h-2, w-2)
        result[node] = glyph
    assert len({_key(g) for g in result.values()}) == n
    return result


def _walk(rng, count, straight):
    if straight:
        return [(0, i) for i in range(count)]
    points = [(0, 0)]
    for _ in range(count-1):
        r, c = points[-1]
        choices = [(r+dr, c+dc) for dr, dc in DIRECTIONS if (r+dr, c+dc) not in points]
        if not choices:
            raise ValueError('local walk trapped')
        points.append(choices[int(rng.integers(len(choices)))])
    return points


def _draw_patch(rng, path, glyphs, connector, gap_range=(1, 4), straight=False, branch=None):
    h, w = next(iter(glyphs.values())).shape
    lattice = _walk(rng, len(path), straight)
    anchors = dict(zip(path, lattice))
    edges = list(zip(path, path[1:]))
    if branch is not None:
        anchor, leaf = branch
        r, c = anchors[anchor]
        choices = [(r+dr, c+dc) for dr, dc in DIRECTIONS if (r+dr, c+dc) not in lattice]
        if not choices:
            raise ValueError('no branch port')
        anchors[leaf] = choices[int(rng.integers(len(choices)))]
        edges.append((anchor, leaf))
    def axis(values, station_size):
        lo, hi = min(values), max(values)
        out, value = {}, 0
        for i in range(lo, hi+1):
            out[i] = value
            value += station_size + int(rng.integers(gap_range[0], gap_range[1]+1))
        return out
    rows = axis([r for r, c in anchors.values()], h)
    cols = axis([c for r, c in anchors.values()], w)
    anchors = {node: (rows[r], cols[c]) for node, (r, c) in anchors.items()}
    cells, borders, occupied = {}, {}, set()
    for node, (r0, c0) in anchors.items():
        borders[node] = {(r0+r, c0+c) for r, c in _rim(h, w)}
        for r in range(h):
            for c in range(w):
                p = r0+r, c0+c
                cells[p] = int(glyphs[node][r, c])
                occupied.add(p)
    connections = []
    for a, b in edges:
        ra, ca = anchors[a]
        rb, cb = anchors[b]
        points = set()
        if ra == rb:
            if ca > cb:
                ra, ca, rb, cb = rb, cb, ra, ca
            off1 = int(rng.integers(1, h-1))
            off2 = off1 if gap_range == (1, 1) else int(rng.integers(1, h-1))
            left, right = ca+w, cb-1
            middle = (left+right)//2
            points.update((ra+off1, c) for c in range(left, middle+1))
            points.update((r, middle) for r in range(ra+min(off1, off2), ra+max(off1, off2)+1))
            points.update((rb+off2, c) for c in range(middle, right+1))
        elif ca == cb:
            if ra > rb:
                ra, ca, rb, cb = rb, cb, ra, ca
            off1 = int(rng.integers(1, w-1))
            off2 = off1 if gap_range == (1, 1) else int(rng.integers(1, w-1))
            top, bottom = ra+h, rb-1
            middle = (top+bottom)//2
            points.update((r, ca+off1) for r in range(top, middle+1))
            points.update((middle, c) for c in range(ca+min(off1, off2), ca+max(off1, off2)+1))
            points.update((r, cb+off2) for r in range(middle, bottom+1))
        else:
            raise ValueError('local neighbors must be orthogonal')
        if points & occupied or points & set(cells):
            raise ValueError('connector contact')
        for p in points:
            cells[p] = connector
        connections.append((_edge(a, b), points))
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    shift = lambda ps: {(r-r0, c-c0) for r, c in ps}
    cells = {(r-r0, c-c0): v for (r, c), v in cells.items()}
    if max(max(r for r, c in cells)+1, max(c for r, c in cells)+1) > 29:
        raise ValueError('map fragment too large')
    return dict(cells=cells, borders={n: shift(p) for n, p in borders.items()},
                connections=[(e, shift(p)) for e, p in connections])


def _bare(glyph, node):
    h, w = glyph.shape
    return dict(cells={(r, c): int(glyph[r, c]) for r in range(h) for c in range(w)},
                borders={node: _rim(h, w)}, connections=[])


def _pack(rng, parts, background):
    # Random foreground packing, with a one-cell eight-neighbor separation halo.
    dimensions = [(max(r for r, c in p['cells'])+1, max(c for r, c in p['cells'])+1) for p in parts]
    order = sorted(range(len(parts)), key=lambda i: -(dimensions[i][0]*dimensions[i][1]+float(rng.random())*20))
    for attempt in range(18):
        canvas_h = 30 if attempt >= 6 else int(rng.integers(max(22, max(h for h, w in dimensions)), 31))
        canvas_w = 30 if attempt >= 6 else int(rng.integers(max(22, max(w for h, w in dimensions)), 31))
        grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
        halo = np.zeros(grid.shape, dtype=bool)
        placements = {}
        success = True
        for i in order:
            h, w = dimensions[i]
            rr, cc = np.array(sorted(parts[i]['cells'])).T
            positions = [(r, c) for r in range(canvas_h-h+1) for c in range(canvas_w-w+1)]
            rng.shuffle(positions)
            for r0, c0 in positions:
                if not np.any(halo[rr+r0, cc+c0]):
                    placements[i] = (r0, c0)
                    for (r, c), v in parts[i]['cells'].items():
                        r, c = r+r0, c+c0
                        grid[r, c] = v
                        halo[max(0, r-1):min(canvas_h, r+2), max(0, c-1):min(canvas_w, c+2)] = True
                    break
            else:
                success = False
                break
        if success and np.count_nonzero(grid == background)*2 > grid.size:
            # Crop random excess margins, subject to the visible dominance rule.
            points = np.argwhere(grid != background)
            lo, hi = points.min(0), points.max(0)
            top, left = int(rng.integers(lo[0]+1)), int(rng.integers(lo[1]+1))
            bottom, right = int(rng.integers(hi[0]+1, canvas_h+1)), int(rng.integers(hi[1]+1, canvas_w+1))
            cropped = grid[top:bottom, left:right]
            if np.count_nonzero(cropped == background)*2 > cropped.size:
                grid = cropped.copy()
                placements = {i: (r-top, c-left) for i, (r, c) in placements.items()}
            return grid, placements
    raise ValueError('fragments did not fit in bounded packing attempts')


def _chunks(path, rng, whole=False, singles=False):
    if whole:
        return [path]
    result, i = [], 0
    while i < len(path)-1:
        length = 1 if singles else int(rng.integers(1, min(2, len(path)-1-i)+1))
        result.append(path[i:i+length+1])
        i += length
    return result


def _make_pair(rng, mode, common_roles, fixed_palette, simple=False):
    roles = common_roles if fixed_palette else tuple(int(c) for c in rng.choice([0, 1, 2, 4, 6, 7, 8, 9], 6, replace=False))
    background, connector = roles[:2]
    q = 1 if mode in ('disconnected', 'witness') else (int(rng.integers(2, 4)) if mode == 'test' else int(rng.integers(1, 3)))
    if simple and mode == 'test':
        q = 2
    k = int(rng.integers(q+1, min(q+3, 4 if mode != 'test' else 5)+1))
    if simple or mode == 'witness':
        k = q+1
    main = [0] + list(range(2, 2+k)) + [1]
    short = [0] + list(range(2+k, 2+k+q)) + [1]
    n = 2+k+q
    tails = []
    # A common route prefix or suffix supplies a shared junction, independently
    # of the alternative-route station counts. Skip it in minimal witnesses.
    terminals = [0, 1]
    if mode == 'extra' and not simple and rng.random() < .35:
        junction = int(rng.integers(2))
        tails.append([junction, n])
        terminals[junction] = n
        n += 1
    spur = None
    if not simple and (mode == 'disconnected' or rng.random() < .55):
        anchor = short[int(rng.integers(1, len(short)-1))]
        spur = (anchor, n)
        n += 1
    shape = (4, 4) if simple else FRAME_SHAPES[int(rng.choice(4, p=[.55, .20, .20, .05]))]
    glyphs = _glyphs(rng, n, shape, roles, terminals)
    main_chunks = _chunks(main, rng, whole=mode == 'witness', singles=mode == 'disconnected')
    short_chunks = _chunks(short, rng, singles=mode in ('disconnected', 'witness', 'test'))
    definitions = [(p, 'main') for p in main_chunks] + [(p, 'short') for p in short_chunks] + [(p, 'tail') for p in tails]
    branch_index = None
    if spur is not None:
        eligible = [i for i, (p, typ) in enumerate(definitions) if spur[0] in p and typ == 'short']
        branch_index = eligible[int(rng.integers(len(eligible)))]
    parts, intended_edges = [], set()
    for i, (path, typ) in enumerate(definitions):
        gap = (7, 9) if mode == 'witness' and typ == 'short' else ((1, 1) if mode == 'witness' else (1, 3))
        patch = _draw_patch(rng, path, glyphs, connector, gap_range=gap,
                            straight=simple or mode == 'witness', branch=spur if i == branch_index else None)
        parts.append(patch)
        intended_edges.update(e for e, points in patch['connections'])
    # Redundant observations test identity-wide marking, including unused copies.
    if mode == 'disconnected' or (not simple and rng.random() < .35):
        parts.append(_bare(glyphs[short[1]], short[1]))
    # Occasionally repeat an entire edge observation: this is one logical edge.
    if not simple and mode == 'extra' and rng.random() < .3:
        edge = tuple(short[:2])
        parts.append(_draw_patch(rng, list(edge), glyphs, connector, straight=True, gap_range=(1, 2)))
    # Additional logical chords yield more than one cycle, subject to unique BFS.
    if not simple and mode == 'extra' and rng.random() < .3:
        a, b = int(rng.choice(main[1:-1])), int(rng.choice(short[1:-1]))
        parts.append(_draw_patch(rng, [a, b], glyphs, connector, gap_range=(1, 2)))
        intended_edges.add(_edge(a, b))
    adjacency = _adjacency(range(n), intended_edges)
    path = _shortest(adjacency, *terminals)
    if path is None:
        raise ValueError('tied constructed route')
    selected_nodes = set(path[1:-1])
    selected_edges = {_edge(a, b) for a, b in zip(path, path[1:])}
    grid, placements = _pack(rng, parts, background)
    # Labels are painted from construction provenance. The pixel reader is
    # separately required to recover the same logical graph and full answer.
    output = grid.copy()
    for i, part in enumerate(parts):
        r0, c0 = placements[i]
        for node, border in part['borders'].items():
            if node in selected_nodes:
                for r, c in border:
                    output[r+r0, c+c0] = HIGHLIGHT_STATION
        for edge, points in part['connections']:
            if edge in selected_edges:
                for r, c in points:
                    output[r+r0, c+c0] = HIGHLIGHT_PATH
    data = _read_input(grid)
    # Verify construction topology against independently segmented input pixels.
    signature_to_node = {tuple(int(v) for v in g.flat): node for node, g in glyphs.items()}
    actual_ids = {s['node']: signature_to_node[s['signature'][2]] for s in data['stations']}
    recovered = {_edge(actual_ids[a], actual_ids[b]) for a, b in data['edges']}
    if recovered != intended_edges or {actual_ids[x] for x in data['terminals']} != set(terminals):
        raise ValueError('pixel topology differs from construction')
    if not np.array_equal(_paint(data), output):
        raise ValueError('input-only projection differs from intended label')
    metrics = _metrics(data)
    if mode == 'disconnected' and (metrics['physical_best'] is not None or not metrics['passive_selected']):
        raise ValueError('disconnected identity witness missing')
    if mode == 'witness' and (not metrics['area_conflict'] or not metrics['distance_conflict'] or metrics['physical_best'] is None
                              or metrics['physical_best'] <= metrics['hops']-1):
        raise ValueError('local-versus-transfer witness missing')
    if mode == 'test' and metrics['selected_pieces'] < 3:
        raise ValueError('test must require at least three route fragments')
    return {'input': grid, 'output': output}


def _check_counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    common_roles = tuple(int(c) for c in rng.choice([0, 1, 2, 4, 6, 7, 8, 9], 6, replace=False))
    fixed_palette = bool(rng.random() < .25)
    puzzle, seen = {'train': [], 'test': []}, set()
    train_modes = ['disconnected', 'witness'] + ['extra']*(num_train-2)
    rng.shuffle(train_modes)
    for split, modes in (('train', train_modes), ('test', ['test']*num_test)):
        for mode in modes:
            for attempt in range(100):
                try:
                    pair = _make_pair(rng, mode, common_roles, fixed_palette, simple=attempt >= 70)
                except ValueError:
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded generation exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, information = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            data = _read_input(inp)
            assert np.array_equal(_paint(data), out), 'wrong route, frame, duplicate, or preservation'
            assert np.any(out == 3) and np.any(out == 5)
            # Repainting is limited to frames and selected connector evidence.
            allowed = set().union(*(s['border'] for s in data['stations'] if s['node'] in data['path'][1:-1]),
                                  *(c['cells'] for c in data['connections'] if c['edge'] in data['path_edges']))
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed == allowed
            for s in data['stations']:
                for p in s['cells'] - s['border']:
                    assert inp[p] == out[p], 'motif identity was erased'
            assert data['path_edges'] < data['edges'], 'no competing or unused edge'
            information[split].append(_metrics(data))
    assert any(m['physical_best'] is None and m['passive_selected'] for m in information['train']), 'A witness missing'
    assert any(m['area_conflict'] and m['distance_conflict'] and m['physical_best'] is not None
               and m['physical_best'] > m['hops']-1 for m in information['train']), 'B metric witness missing'
    assert any(m['histogram_collision'] and m['reflected_pair'] for m in information['train']), 'exact-pattern witness missing'
    assert all(m['selected_pieces'] >= 3 for m in information['test']), 'transfer-depth shift missing'


def self_test():
    """Bounded 42-seed semantic and diversity checks, with no parent imports."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics, shapes, structures, palettes, keys = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        keys.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read_input(pair['input'])
                metrics.append(_metrics(data))
                shapes.add(pair['input'].shape)
                palettes.add((data['background'], data['frame'], data['connector'], data['marker']))
                labels = {}
                structures.add((pair['input'].shape, tuple(labels.setdefault(int(x), len(labels)) for x in pair['input'].flat)))
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Recolor all input roles while keeping fixed output highlights.
                mapping = np.arange(10, dtype=np.int8)
                pool = np.array([0, 1, 2, 4, 6, 7, 8, 9], dtype=np.int8)
                mapping[pool] = pool[::-1]
                assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
                # A whole scene reflection preserves same-orientation equality.
                assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
    assert len(keys) == 42 and len(shapes) >= 15 and len(structures) >= 180
    assert len(palettes) >= 20
    assert len({m['nodes'] for m in metrics}) >= 4
    assert len({m['pieces'] for m in metrics}) >= 4
    assert len({m['frame_shape'] for m in metrics}) >= 3
    assert any(m['loop_rank'] >= 2 for m in metrics)
    assert any(m['duplicate_edges'] for m in metrics)
    assert any(max(m['degrees']) >= 3 for m in metrics)
    assert any(m['passive_selected'] for m in metrics)
    for seed in (0, 17):
        a, b = generate_puzzle(seed), generate_puzzle(seed)
        assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
                   for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    # Exhaust all explicit count combinations once, not recursively via verify.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Reject an altered unused pixel, and reject tied minimum routes explicitly.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    data = _read_input(bad['test'][0]['input'])
    unused = next(c for c in data['connections'] if c['edge'] not in data['path_edges'])
    bad['test'][0]['output'][next(iter(unused['cells']))] = 5
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('unused connector recoloring accepted')
    assert _shortest(_adjacency(range(4), [(0, 1), (1, 3), (0, 2), (2, 3)]), 0, 3) is None
