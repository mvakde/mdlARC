"""Topological blueprints: least-ink cavity-constrained sparse wire networks.

Only standard-library modules and NumPy are used. solve() and its parser receive
pixels only; generation preferences and proposed construction edges are not used.
"""
from itertools import combinations

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() <= 9,
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], {first}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _mask(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells) - r0 + 1
    w = max(c for r, c in cells) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r-r0, c-c0] = True
    return mask, (r0, c0)


def _cavities(cells):
    mask, (r0, c0) = _mask(cells)
    padded = np.pad(mask, 1, constant_values=False)
    holes = []
    for component in _components(~padded):
        if (0, 0) not in component:
            holes.append({(r+r0-1, c+c0-1) for r, c in component})
    return holes


def _line(a, b):
    _require(a != b and (a[0] == b[0] or a[1] == b[1]), "nonorthogonal link")
    dr = int(b[0] > a[0]) - int(b[0] < a[0])
    dc = int(b[1] > a[1]) - int(b[1] < a[1])
    return {(a[0]+i*dr, a[1]+i*dc)
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)}


def _halo(cells):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _connected(n, edges):
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        a, b = root(a), root(b)
        if a != b:
            parent[a] = b
    return len({root(i) for i in range(n)}) == 1


def _candidate_graph(points):
    points = tuple(sorted(points))
    _require(2 <= len(points) <= 26, "source must have 2..26 vertices")
    vertices = set(points)
    edges, rasters = [], []
    for i, a in enumerate(points):
        for j in range(i+1, len(points)):
            b = points[j]
            if a[0] != b[0] and a[1] != b[1]:
                continue
            raster = _line(a, b)
            if (raster - {a, b}) & vertices:
                continue
            _require(len(raster) >= 3, "possible links must have at least two steps")
            edges.append((i, j))
            rasters.append(raster)
    n, m = len(points), len(edges)
    _require(m <= 29 and _connected(n, edges), "invalid or disconnected candidate graph")
    rank = m - n + 1
    _require(0 <= rank <= 4, "candidate graph must have at most four independent cycles")
    halos = [_halo(raster) for raster in rasters]
    for i, (a, b) in enumerate(edges):
        for j, (c, d) in enumerate(edges[:i]):
            shared = {points[t] for t in {a, b} & {c, d}}
            _require(rasters[i] & rasters[j] == shared, "links intersect away from a vertex")
            if not shared:
                _require(not rasters[i] & halos[j], "nonincident links too close")
    full = set().union(*rasters)
    _require(len(_cavities(full)) == rank, "digital topology is not a clean planar embedding")
    return dict(points=points, edges=edges, rasters=rasters, rank=rank, full=full)


def _minimum_network(graph, target):
    """Enumerate small edge-removal sets; no source template or seed is needed.

    A clean connected rectilinear graph with k holes has n-1+k edges. Search
    therefore removes rank-k edges, at most four, from the candidate graph.
    """
    _require(0 <= target <= graph['rank'], "unattainable requested cavity count")
    n, m = len(graph['points']), len(graph['edges'])
    costs = [len(raster)-2 for raster in graph['rasters']]
    total = sum(costs)
    best, winners, selected = None, 0, None
    for removed_tuple in combinations(range(m), graph['rank']-target):
        cost = total - sum(costs[i] for i in removed_tuple)
        if best is not None and cost > best:
            continue
        removed = set(removed_tuple)
        chosen = [i for i in range(m) if i not in removed]
        if not _connected(n, [graph['edges'][i] for i in chosen]):
            continue
        # Cavity counting is performed on source ink alone, before copying.
        pixels = set().union(*(graph['rasters'][i] for i in chosen))
        if len(_cavities(pixels)) != target:
            continue
        if best is None or cost < best:
            best, winners, selected = cost, 1, (chosen, pixels)
        else:
            winners += 1
    _require(winners == 1, "requested network has no unique least-ink solution")
    chosen, pixels = selected
    return dict(chosen=chosen, pixels=pixels, cost=best, target=target)


def _interpret_side(grid, bg, sep, panel, work):
    entries, mapping = [], {}
    for color in sorted(int(c) for c in np.unique(grid[panel]) if c != bg):
        cells = {tuple(map(int, p)) for p in np.argwhere(panel & (grid == color))}
        _require(color != sep and len(_components(panel & (grid == color))) == 1,
                 "legend colors must each form one glyph")
        _require(len(cells) >= 4, "legend glyph too small")
        holes = _cavities(cells)
        _require(all(grid[p] == bg for hole in holes for p in hole), "nonbackground glyph cavity")
        k = len(holes)
        _require(k <= 4 and k not in mapping, "duplicate or excessive legend topology")
        mapping[k] = color
        entries.append(dict(color=color, target=k, cells=cells, position=min(cells)))
    _require(bool(mapping), "empty legend")
    inverse = {color: k for k, color in mapping.items()}
    sources = []
    work_colors = set(int(c) for c in np.unique(grid[work])) - {bg}
    _require(sep not in work_colors, "separator pixels in work area")
    for color in sorted(work_colors - set(inverse)):
        points = {tuple(map(int, p)) for p in np.argwhere(work & (grid == color))}
        graph = _candidate_graph(points)
        _require(all(grid[p] == bg or (p in points and grid[p] == color) for p in graph['full']),
                 "possible link passes through a foreign input cell")
        sources.append(dict(color=color, graph=graph))
    _require(1 <= len(sources) <= 3, "work area must contain 1..3 source groups")
    dots = [(tuple(map(int, p)), int(grid[tuple(p)]))
            for p in np.argwhere(work & np.isin(grid, list(inverse)))]
    _require(len(dots) == len(sources), "exactly one selector per source is required")
    marked = set()
    for point, color in dots:
        associations = [(i, v) for i, s in enumerate(sources) for v in s['graph']['points']
                        if max(abs(point[0]-v[0]), abs(point[1]-v[1])) == 1]
        _require(len(associations) == 1, "ambiguous adjacent selector")
        i, vertex = associations[0]
        _require(i not in marked, "source has multiple selectors")
        marked.add(i)
        source = sources[i]
        _require(point not in source['graph']['full'], "selector lies on a possible link")
        offset = (point[0]-vertex[0], point[1]-vertex[1])
        network = _minimum_network(source['graph'], inverse[color])
        copy = {(r+offset[0], c+offset[1]) for r, c in network['pixels']}
        _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and work[r, c]
                     for r, c in copy), "companion is clipped or outside work area")
        source.update(network=network, dot=point, companion_color=color, offset=offset, copy=copy)
    _require(len(marked) == len(sources), "unmarked source")
    all_source, claims = set(), {}
    for s in sources:
        pixels = s['network']['pixels']
        _require(not all_source & pixels, "source networks collide")
        _require(all(grid[p] == bg or (p in s['graph']['points'] and grid[p] == s['color'])
                     for p in pixels), "source completion collides with an input cell")
        all_source.update(pixels)
    output = grid.copy()
    for s in sources:
        for p in s['copy']:
            if grid[p] == bg and p not in all_source:
                _require(p not in claims or claims[p] == s['companion_color'], "companion conflict")
                claims[p] = s['companion_color']
                output[p] = s['companion_color']
        s['visible_copy'] = sum(grid[p] == bg and p not in all_source for p in s['copy'])
        s['overlap_evidence'] = sum(grid[p] == bg and p in all_source for p in s['copy'])
    for s in sources:
        for p in s['network']['pixels']:
            if grid[p] == bg:
                output[p] = s['color']
    return dict(background=bg, separator=sep, entries=entries, mapping=mapping,
                panel=panel, work=work, sources=sources, output=output)


def _parse_input(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(max(counts)) > grid.size//2, "background must occupy over half of the input")
    lines = [('row', r, int(grid[r, 0])) for r in range(grid.shape[0])
             if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])]
    lines += [('col', c, int(grid[0, c])) for c in range(grid.shape[1])
              if grid[0, c] != bg and np.all(grid[:, c] == grid[0, c])]
    _require(len(lines) == 1, "exactly one nonbackground full separator is required")
    axis, index, sep = lines[0]
    first, second = np.zeros(grid.shape, bool), np.zeros(grid.shape, bool)
    if axis == 'row':
        first[:index] = True
        second[index+1:] = True
    else:
        first[:, :index] = True
        second[:, index+1:] = True
    interpretations = []
    for panel, work in ((first, second), (second, first)):
        if not panel.any() or not work.any():
            continue
        try:
            p = _interpret_side(grid, bg, sep, panel, work)
            p['axis'], p['separator_index'] = axis, index
            interpretations.append(p)
        except ValueError:
            continue
    _require(len(interpretations) == 1, "input has no unique legend/network interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique answer exclusively from the supplied input pixels."""
    return _parse_input(input_grid)['output'].copy()


def _transform_points(points, code):
    result = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return result


def _normalize(points):
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return [(r-r0, c-c0) for r, c in points]


def _geometry_signature(points):
    return tuple(sorted(_normalize(points)))


def _lobe(rng, width=None, broad=False, thin=False, rectangle=False):
    w = int(rng.integers(5, 9)) if width is None else width
    h = w-1 if broad else 4 if thin else int(rng.integers(4, w))
    if rectangle:
        points = [(0, 0), (0, w), (h, w), (h, 0)]
    elif not broad and not thin and h >= 6 and w >= 7 and rng.random() < .45:
        points = [(0, 0), (0, w), (h, w), (h, w-2),
                  (4, w-2), (4, 2), (2, 2), (2, 0)]
    else:
        a = 2 if broad else w-2 if thin else int(rng.integers(2, w-1))
        b = h-2 if broad else 2 if thin else int(rng.integers(2, h-1))
        points = [(0, 0), (0, w), (h, w), (h, a), (b, a), (b, 0)]
    code = int(rng.integers(8))
    return _normalize(_transform_points(points, code))


def _assemble_lobes(lobes, rng, max_height):
    points, intended = list(lobes[0]), set()
    for i in range(len(lobes[0])):
        intended.add(frozenset((points[i], points[(i+1) % len(lobes[0])])))
    previous = list(lobes[0])
    for lobe in lobes[1:]:
        x = max(c for r, c in previous)
        right = [v for v in previous if v[1] == x]
        left = [v for v in lobe if v[1] == 0]
        choices = [(a, b, gap) for a in right for b in left for gap in (2, 3)]
        rng.shuffle(choices)
        for a, b, gap in choices:
            shifted = [(r+a[0]-b[0], c+x+gap) for r, c in lobe]
            proposed = points + shifted
            if max(r for r, c in proposed)-min(r for r, c in proposed)+1 > max_height:
                continue
            new_edges = set(intended)
            for i, v in enumerate(shifted):
                new_edges.add(frozenset((v, shifted[(i+1) % len(shifted)])))
            new_edges.add(frozenset((a, (a[0], x+gap))))
            try:
                graph = _candidate_graph(proposed)
            except ValueError:
                continue
            actual = {frozenset((graph['points'][i], graph['points'][j])) for i, j in graph['edges']}
            if actual == new_edges:
                points, intended, previous = proposed, new_edges, shifted
                break
        else:
            return None
    return _normalize(points)


def _tree(rng):
    # Segments, two-to-five-link staircases, and three/four-arm branches.
    mode = int(rng.integers(3))
    if mode == 0:
        points = [(0, 0), (0, int(rng.integers(2, 9)))]
    elif mode == 1:
        points = [(0, 0)]
        for i in range(int(rng.integers(2, 6))):
            length = int(rng.integers(2, 6))
            r, c = points[-1]
            points.append((r+length, c) if i % 2 else (r, c+length))
    else:
        a, b, c, d = [int(rng.integers(2, 6)) for _ in range(4)]
        points = [(0, a), (b, a), (b, 0), (b, a+c)]
        if rng.random() < .5:
            points.append((b+d, a))
    return _normalize(_transform_points(points, int(rng.integers(8))))


def _make_network(rng, loops, target, witness=False, twig=False):
    if loops == 0:
        return _tree(rng)
    for _ in range(160):
        if witness:
            w = int(rng.integers(6, 8))
            lobes = [_lobe(rng, w, broad=True), _lobe(rng, w+1, thin=True)]
        else:
            widths = [int(v) for v in rng.choice(np.arange(5, 9), loops, replace=False)]
            lobes = [_lobe(rng, w, rectangle=(target == loops and rng.random() < .23)) for w in widths]
        points = _assemble_lobes(lobes, rng, 9 if witness else 13)
        if points is None or max(c for r, c in points) > 27:
            continue
        try:
            graph = _candidate_graph(points)
            network = _minimum_network(graph, target)
            if graph['rank'] != loops:
                continue
            if witness:
                one = _minimum_network(graph, 1)
                all_areas = [len(h) for h in _cavities(graph['full'])]
                if len(_cavities(one['pixels'])[0]) <= min(all_areas):
                    continue
            if twig and len(points) < 26:
                choices = [(v, d, length) for v in points for d in ((0, 1), (0, -1), (1, 0), (-1, 0))
                           for length in (2, 3)]
                rng.shuffle(choices)
                for v, d, length in choices:
                    q = (v[0]+d[0]*length, v[1]+d[1]*length)
                    if q in graph['full']:
                        continue
                    try:
                        grown = _candidate_graph(points+[q])
                        if grown['rank'] == loops and len(grown['edges']) == len(graph['edges'])+1:
                            _minimum_network(grown, target)
                            points = _normalize(points+[q])
                            break
                    except ValueError:
                        continue
            return points
        except ValueError:
            continue
    points = _fallback_network(loops, witness)
    _minimum_network(_candidate_graph(points), target)
    return _normalize(points)


def _fallback_network(loops, witness=False):
    """Explicitly clean fitting networks for every sampled cycle budget."""
    if witness:
        return _normalize([(0, 0), (0, 6), (5, 6), (5, 2), (3, 2), (3, 0),
                           (-2, 8), (-2, 16), (2, 16), (2, 14), (0, 14), (0, 8)])
    points = []
    for i in range(loops):
        w, h, b = (5, 6, 7)[i], (4, 5, 6)[i], (2, 2, 3)[i]
        y, x = (0, -2, -5)[i], (0, 7, 15)[i]
        points.extend((r+y, c+x) for r, c in [(0, 0), (0, w), (h, w), (h, 2), (b, 2), (b, 0)])
    return _normalize(points)


def _spec(points, target, rng, informative=True):
    graph = _candidate_graph(points)
    network = _minimum_network(graph, target)
    choices = []
    for vertex in graph['points']:
        for dr, dc in OFFSETS:
            dot = (vertex[0]+dr, vertex[1]+dc)
            if dot in graph['full']:
                continue
            if sum(max(abs(dot[0]-v[0]), abs(dot[1]-v[1])) == 1 for v in points) != 1:
                continue
            copy = {(r+dr, c+dc) for r, c in network['pixels']}
            visible = len(copy - network['pixels'] - {dot})
            overlap = len((copy & network['pixels']) - set(points))
            if not informative or (visible >= 2 and overlap >= 1):
                choices.append((dot, (dr, dc)))
    _require(bool(choices), "no informative selector placement")
    dot, offset = choices[int(rng.integers(len(choices)))]
    # Reserve possible rather than only selected wires so neighboring sources do
    # not accidentally introduce a selector association or latent alternative.
    footprint = graph['full'] | {(r+offset[0], c+offset[1]) for r, c in graph['full']}
    r0, c0 = min(r for r, c in footprint), min(c for r, c in footprint)
    return dict(points=[(r-r0, c-c0) for r, c in points], target=target,
                dot=(dot[0]-r0, dot[1]-c0), offset=offset,
                h=max(r for r, c in footprint)-r0+1, w=max(c for r, c in footprint)-c0+1)


def _glyph(k, rng):
    if k == 0:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        glyph = np.ones((h, w), bool)
        if h == w == 3 and rng.random() < .5:
            glyph[0, 0] = False
    elif k == 3 and rng.random() < .4:
        glyph = np.ones((5, 5), bool)
        glyph[1, 1] = glyph[1, 3] = glyph[3, 1] = False
    else:
        glyph = np.ones((3, 2*k+1), bool)
        glyph[1, 1:2*k:2] = False
        if rng.random() < .3:
            glyph = np.repeat(glyph, 2, axis=1)
    # Keep horizontal glyphs bounded so every mandatory demonstration fits.
    if k <= 2 and rng.random() < .35:
        glyph = glyph.T
    if glyph.shape[0] > 5 or glyph.shape[1] > 8:
        glyph = np.ones((3, max(3, 2*k+1)), bool)
        glyph[1, 1:2*k:2] = False
    return glyph


def _legend(rng, keys, colors, bg, sep):
    order = list(keys)
    rng.shuffle(order)
    # Forbid sorted rank in teaching legends, so used 0/1 classes cannot both be
    # interpreted as their spatial rank. This preference does not enter solve.
    if order == sorted(order) and len(order) >= 2:
        order = order[1:] + order[:1]
    glyphs = [_glyph(k, rng) for k in order]
    if sum(g.shape[1] for g in glyphs)+len(glyphs)+1 > 30:
        glyphs = []
        for k in order:
            g = np.ones((3, max(3, 2*k+1)), bool)
            g[1, 1:2*k:2] = False
            glyphs.append(g)
    ph = max(g.shape[0] for g in glyphs)+2
    required_width = sum(g.shape[1] for g in glyphs)+len(glyphs)+1
    panel = np.full((ph+1, 30), bg, np.int8)
    left = int(rng.integers(30-required_width+1))
    c = left+1
    for k, g in zip(order, glyphs):
        r = 1 + int(rng.integers(ph-1-g.shape[0]))
        panel[r:r+g.shape[0], c:c+g.shape[1]][g] = colors[k]
        c += g.shape[1]+1
    panel[ph] = sep
    return panel, ph, required_width


def _pack_specs(specs, field_height, rng):
    if any(s['h'] > field_height or s['w'] > 30 for s in specs):
        return None
    order = sorted(range(len(specs)), key=lambda i: specs[i]['h']*specs[i]['w'], reverse=True)
    for _ in range(20):
        free = np.ones((field_height, 30), bool)
        locations = {}
        for i in order:
            s = specs[i]
            options = [(r, c) for r in range(field_height-s['h']+1) for c in range(31-s['w'])]
            rng.shuffle(options)
            for r, c in options:
                if free[r:r+s['h'], c:c+s['w']].all():
                    locations[i] = (r, c)
                    free[max(0, r-1):min(field_height, r+s['h']+1),
                         max(0, c-1):min(30, c+s['w']+1)] = False
                    break
            else:
                break
        if len(locations) == len(specs):
            return [locations[i] for i in range(len(specs))]
    if sum(s['h'] for s in specs)+len(specs)-1 <= field_height:
        r, result = 0, []
        for s in specs:
            result.append((r, int(rng.integers(31-s['w']))))
            r += s['h']+1
        return result
    return None


def _make_pair(rng, kind, ordinal=0):
    """kind is presentation only: contrast, closure, ordinary, or extrapolation."""
    for attempt in range(80):
        palette = [int(v) for v in rng.permutation(10)]
        bg, sep = palette[:2]
        keys = [0, 1, 2]
        if kind in ('ordinary', 'extrapolation') and rng.random() < .5:
            keys = [int(v) for v in rng.choice(np.arange(4), int(rng.integers(2, 5)), replace=False)]
        class_colors = dict(zip(keys, palette[2:2+len(keys)]))
        source_colors = palette[2+len(keys):]
        networks = []
        if kind == 'contrast':
            p = _make_network(rng, 2, 1, witness=True)
            networks = [(p, 0), (p, 1)]
        elif kind == 'closure':
            networks = [(_make_network(rng, 2, 2, twig=rng.random() < .3), 2)]
        else:
            number = 1 if kind == 'extrapolation' else int(rng.integers(1, 4))
            for i in range(number):
                maximum = 3 if kind == 'extrapolation' else 2
                available = [k for k in keys if k <= maximum]
                if not available:
                    continue
                target = int(rng.choice(available))
                loops = 3 if kind == 'extrapolation' else int(rng.integers(max(1, target), 3))
                if kind == 'ordinary' and target == 0 and rng.random() < .32:
                    loops = 0
                points = _make_network(rng, loops, target, twig=rng.random() < .28)
                networks.append((points, target))
        if not networks:
            continue
        try:
            specs = [_spec(points, k, rng, informative=kind in ('contrast', 'closure'))
                     for points, k in networks]
        except ValueError:
            continue
        panel, sep_row, required_width = _legend(rng, keys, class_colors, bg, sep)
        start = sep_row+2
        locations = _pack_specs(specs, 30-start, rng)
        if locations is None:
            continue
        grid = np.full((30, 30), bg, np.int8)
        grid[:panel.shape[0]] = panel
        for i, (s, (r, c)) in enumerate(zip(specs, locations)):
            r += start
            for a, b in s['points']:
                grid[r+a, c+b] = source_colors[i]
            a, b = s['dot']
            grid[r+a, c+b] = class_colors[s['target']]
        # Crop empty trailing field/padding while retaining all reserved copies.
        h = max(start+r+s['h'] for s, (r, c) in zip(specs, locations))
        w = max(required_width, max(c+s['w'] for s, (r, c) in zip(specs, locations)))
        # The legend's actual position, not just its minimal width, must remain.
        glyph_columns = np.argwhere((grid[:sep_row] != bg))[:, 1]
        w = max(w, int(glyph_columns.max())+2)
        h = min(30, h+int(rng.integers(min(3, 31-h))))
        w = min(30, w+int(rng.integers(min(3, 31-w))))
        grid = grid[:h, :w].copy()
        # Separator stays full length after cropping. Joint symmetries preserve
        # topology, cost and the locally encoded displacement.
        code = int(rng.integers(8))
        if code >= 4:
            grid = np.fliplr(grid)
        grid = np.rot90(grid, code % 4).copy()
        try:
            info = _parse_input(grid)
            if kind == 'contrast':
                _require(_contrast_witness(info), "contrast construction lost its witness")
            return {'input': grid, 'output': info['output'].copy()}
        except ValueError:
            continue
    return _fallback_pair(rng, kind, ordinal)


def _fallback_pair(rng, kind, ordinal, variant=0):
    """Guaranteed finite placement; variant supplies 32 distinct local palettes.

    Same teaching obligations and same input-only rule as randomized examples.
    The 30x30 canvas fits two 10x18 contrast footprints or one 12x25 test.
    """
    bg = (ordinal+variant) % 10
    remaining = [c for c in range(10) if c != bg]
    phase = ((ordinal+variant)//10) % 9
    remaining = remaining[phase:] + remaining[:phase]
    palette = [bg]+remaining
    sep = palette[1]
    mapping = dict(zip((0, 1, 2), palette[2:5]))
    if kind == 'contrast':
        points = _fallback_network(2, witness=True)
        networks = [(points, 0), (points, 1)]
    elif kind == 'closure':
        networks = [(_fallback_network(2), 2)]
    elif kind == 'extrapolation':
        networks = [(_fallback_network(3), 1)]
    else:
        networks = [(_fallback_network(2), 1)]
    specs = [_spec(points, k, rng) for points, k in networks]
    grid = np.full((30, 30), bg, np.int8)
    grid[5] = sep
    c = 1
    for k in (1, 2, 0):
        glyph = np.ones((3, max(3, 2*k+1)), bool)
        glyph[1, 1:2*k:2] = False
        grid[1:4, c:c+glyph.shape[1]][glyph] = mapping[k]
        c += glyph.shape[1]+1
    r = 7
    for i, s in enumerate(specs):
        c = 1 + ordinal % 3
        _require(r+s['h'] <= 30 and c+s['w'] <= 30, "bad finite fallback fit")
        for a, b in s['points']:
            grid[r+a, c+b] = palette[5+i]
        a, b = s['dot']
        grid[r+a, c+b] = mapping[s['target']]
        r += s['h']+1
    info = _parse_input(grid)
    if kind == 'contrast':
        _require(_contrast_witness(info), "bad fallback contrast")
    return {'input': grid, 'output': info['output'].copy()}


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= int(value) <= high:
            raise ValueError(f'{name} must be in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    kinds = ['contrast', 'closure'] + ['ordinary']*(nt-2)
    rng.shuffle(kinds)
    result, seen = {'train': [], 'test': []}, set()
    for split, sequence in (('train', kinds), ('test', ['extrapolation']*nq)):
        for kind in sequence:
            for retry in range(24):
                pair = _make_pair(rng, kind, len(seen))
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    result[split].append(pair)
                    break
            else:
                # At most thirteen inputs have been used. For fixed ordinal,
                # these 32 palettes differ in their visible background or key.
                for variant in range(32):
                    pair = _fallback_pair(rng, kind, len(seen), variant)
                    sig = _signature(pair['input'])
                    if sig not in seen:
                        seen.add(sig)
                        result[split].append(pair)
                        break
                else:
                    raise AssertionError('distinct fitting fallback invariant failed')
    validate_generated_puzzle(result)
    return result


def _contrast_witness(info):
    for a in info['sources']:
        if a['network']['target'] != 1 or a['graph']['rank'] != 2:
            continue
        full_areas = [len(h) for h in _cavities(a['graph']['full'])]
        if len(_cavities(a['network']['pixels'])[0]) <= min(full_areas):
            continue
        for b in info['sources']:
            if b['network']['target'] == 0 and _geometry_signature(a['graph']['points']) == _geometry_signature(b['graph']['points']):
                return True
    return False


def _without_topology(grid, info):
    """Ablation: complete and copy every possible link, ignoring the blueprint."""
    out = grid.copy()
    full = set().union(*(s['graph']['full'] for s in info['sources']))
    for s in info['sources']:
        dr, dc = s['offset']
        for r, c in s['graph']['full']:
            p = r+dr, c+dc
            if 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] and info['work'][p]:
                if grid[p] == info['background'] and p not in full:
                    out[p] = s['companion_color']
    for s in info['sources']:
        for p in s['graph']['full']:
            if grid[p] == info['background']:
                out[p] = s['color']
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "wrong puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    b_witness = a_witness = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "wrong pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "canvas size changed")
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _parse_input(inp)
            _require(np.array_equal(info['output'], out), "incorrect least-ink network or companion")
            bg = info['background']
            _require(np.array_equal(inp[inp != bg], out[inp != bg]), "existing cell changed")
            _require(np.any(inp != out), "identity example")
            for source in info['sources']:
                _require(len(_cavities(source['network']['pixels'])) == source['network']['target'],
                         "wrong source topology")
                actual_added = sum(inp[p] == bg for p in source['network']['pixels'])
                _require(actual_added == source['network']['cost'], "ink accounting mismatch")
            infos[split].append(info)
            if split == 'train':
                b_witness |= not np.array_equal(_without_topology(inp, info), out)
                no_copy = inp.copy()
                for s in info['sources']:
                    for p in s['network']['pixels']:
                        if inp[p] == bg:
                            no_copy[p] = s['color']
                a_witness |= not np.array_equal(no_copy, out)
    training = infos['train']
    goals = {s['network']['target'] for p in training for s in p['sources']}
    _require({0, 1, 2} <= goals, "training must demonstrate targets zero, one and two")
    _require(any(_contrast_witness(p) for p in training), "missing same-geometry/area-vs-ink contrast")
    _require(b_witness and a_witness, "missing parent-concept ablation witnesses")
    _require(any(s['visible_copy'] >= 2 and s['overlap_evidence'] >= 1
                 for p in training for s in p['sources']), "copy/priority convention is not visible in training")
    rank_witness = False
    for p in training:
        # Compare glyphs in their principal separator-parallel direction.
        order = sorted(p['entries'], key=lambda e: e['position'][1] if p['axis'] == 'row' else e['position'][0])
        rank_by_color = {e['color']: i for i, e in enumerate(order)}
        rank_witness |= any(rank_by_color[s['companion_color']] != s['network']['target'] for s in p['sources'])
    _require(rank_witness, "legend-rank shortcut is not distinguished")
    _require(all(max(s['graph']['rank'] for s in p['sources']) <= 2 for p in training),
             "training structural range changed")
    _require(all(any(s['graph']['rank'] == 3 for s in p['sources']) for p in infos['test']),
             "test must extend to three-loop networks")
    return True


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def diversity_features(pair):
    p = _parse_input(pair['input'])
    sources = p['sources']
    return dict(shape=list(np.asarray(pair['input']).shape), background=p['background'],
                roles=[p['background'], p['separator']] + [s['color'] for s in sources],
                sources=len(sources), keys=sorted(p['mapping']),
                targets=[s['network']['target'] for s in sources],
                cycles=[s['graph']['rank'] for s in sources],
                vertices=[len(s['graph']['points']) for s in sources],
                costs=[s['network']['cost'] for s in sources],
                offsets=[list(s['offset']) for s in sources],
                omitted=[s['graph']['rank']-s['network']['target'] for s in sources],
                geometry=[repr(_geometry_signature(s['graph']['points'])) for s in sources],
                cavity_areas=[len(h) for s in sources for h in _cavities(s['graph']['full'])],
                visible_copies=sum(s['visible_copy'] > 0 for s in sources),
                hidden_copies=sum(s['visible_copy'] == 0 for s in sources),
                clear_copies=sum(not s['copy'] & s['network']['pixels'] for s in sources),
                contrast=_contrast_witness(p), legend_axis=p['axis'],
                branches=sum(any(sum(i in edge for edge in s['graph']['edges']) >= 3
                                 for i in range(len(s['graph']['points']))) for s in sources))


def self_test():
    # Four-neighbor topology: a diagonal escape is still a cavity.
    diagonal = {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    assert len(_cavities(diagonal)) == 1
    assert len(_cavities(diagonal - {(0, 1)})) == 0
    # Symmetric rectangle has two equal tree minima and must be rejected.
    graph = _candidate_graph([(0, 0), (0, 5), (3, 5), (3, 0)])
    try:
        _minimum_network(graph, 0)
    except ValueError:
        pass
    else:
        raise AssertionError('tied minimum accepted')
    assert _minimum_network(graph, 1)['target'] == 1
    # Fallback network path is checked explicitly, independently of retries.
    for loops in (1, 2, 3):
        fallback_graph = _candidate_graph(_fallback_network(loops))
        assert fallback_graph['rank'] == loops
        for k in range(loops+1):
            assert len(_cavities(_minimum_network(fallback_graph, k)['pixels'])) == k
    fallback_rng = np.random.default_rng(1234)
    fallback = {'train': [_fallback_pair(fallback_rng, 'contrast', 0),
                          _fallback_pair(fallback_rng, 'closure', 1)],
                'test': [_fallback_pair(fallback_rng, 'extrapolation', 2)]}
    validate_generated_puzzle(fallback)
    assert len({_signature(_fallback_pair(fallback_rng, 'ordinary', 0, v)['input'])
                for v in range(32)}) == 32
    # The solver's topology range goes beyond the sampler's three-lobe shift.
    # This hand-specified fourth lobe is attached vertically to the third one.
    points = _fallback_network(3) + [(r-5, c+22) for r, c in
              [(0, 0), (0, 5), (4, 5), (4, 2), (2, 2), (2, 0)]]
    graph = _candidate_graph(points)
    assert graph['rank'] == 4
    spec = _spec(points, 4, fallback_rng)
    grid = np.zeros((30, 30), np.int8)
    grid[5] = 8
    glyph = np.ones((3, 9), bool)
    glyph[1, 1:8:2] = False
    grid[1:4, 1:10][glyph] = 2
    for r, c in spec['points']:
        grid[r+7, c] = 5
    grid[spec['dot'][0]+7, spec['dot'][1]] = 2
    parsed = _parse_input(grid)
    assert parsed['sources'][0]['network']['target'] == 4
    assert parsed['sources'][0]['network']['pixels'] == parsed['sources'][0]['graph']['full']
    assert np.array_equal(solve(grid), parsed['output'])
    rows, puzzles = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                rows.append(diversity_features(pair))
    assert _equal(puzzles[0], generate_puzzle(0))
    assert not _equal(puzzles[0], puzzles[1])
    shapes = {tuple(f['shape']) for f in rows}
    geometry = {g for f in rows for g in f['geometry']}
    offsets = {tuple(d) for f in rows for d in f['offsets']}
    assert len(shapes) >= 20 and len(geometry) >= 60
    assert offsets == set(OFFSETS)
    assert {f['background'] for f in rows} == set(range(10))
    assert {f['sources'] for f in rows} == {1, 2, 3}
    assert {c for f in rows for c in f['cycles']} >= {0, 1, 2, 3}
    assert {k for f in rows for k in f['targets']} >= {0, 1, 2, 3}
    assert sum(f['contrast'] for f in rows) >= 42
    assert {f['legend_axis'] for f in rows} == {'row', 'col'}
    assert max(v for f in rows for v in f['vertices']) >= 18
    assert len({tuple(f['cavity_areas']) for f in rows}) >= 40
    # Palette equivariance and full-canvas symmetries exercise input-only logic.
    pair = puzzles[0]['train'][0]
    permutation = np.array([7, 1, 9, 2, 6, 8, 0, 5, 3, 4], np.int8)
    assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    for rotation in range(4):
        for reflect in (False, True):
            a, b = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
            if reflect:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(123+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Incorrect ink, changed legend, and wrong overlap priority must be detected
    # without reading generation metadata.
    for role in ('missing_ink', 'legend', 'priority'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = bad['train'][0]
        info = _parse_input(pair['input'])
        if role == 'missing_ink':
            p = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][p] = pair['input'][p]
        elif role == 'legend':
            p = min(info['entries'][0]['cells'])
            pair['output'][p] = info['background']
        else:
            for candidate in bad['train']:
                candidate_info = _parse_input(candidate['input'])
                good = [s for s in candidate_info['sources'] if s['overlap_evidence']]
                if good:
                    pair, info, s = candidate, candidate_info, good[0]
                    break
            p = next(p for p in s['copy'] & s['network']['pixels'] if pair['input'][p] == info['background'])
            pair['output'][p] = s['companion_color']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'corrupt {role} answer accepted')
    malformed = fallback['train'][0]['input'].copy()
    parsed = _parse_input(malformed)
    source = parsed['sources'][0]
    malformed[source['dot']] = parsed['background']
    malformed[29, 29] = source['companion_color']
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('unassociated selector accepted')
    return dict(seeds=42, pairs=len(rows), grid_shapes=len(shapes), geometry_signatures=len(geometry),
                source_counts=sorted({f['sources'] for f in rows}),
                vertex_counts=sorted({v for f in rows for v in f['vertices']}),
                offsets=[list(d) for d in sorted(offsets)],
                cavity_area_sequences=len({tuple(f['cavity_areas']) for f in rows}),
                semantic_verification='generator-owned checks only')
