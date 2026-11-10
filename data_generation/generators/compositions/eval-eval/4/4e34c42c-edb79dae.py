"""Registered printing plates with exclusive (exactly-one) ink coverage.

The legend, panel membership, registration offsets and ink masks are all visible.
Generation uses assembly coordinates; solve reconstructs them from input pixels.
No parent module, fixture, sampled metadata or puzzle history is used by solve.
"""
from itertools import permutations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _groups(points):
    remaining = set(points)
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _bbox(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points)+1, max(c for r, c in points)+1)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _render(cells, background):
    if not cells:
        raise ValueError("empty assembled picture")
    r, c, e, d = _bbox(cells)
    if e-r > 30 or d-c > 30:
        raise ValueError("assembly exceeds the ARC size limit")
    a = np.full((e-r, d-c), background, dtype=np.int8)
    for (y, x), color in cells.items():
        a[y-r, x-c] = color
    return a


def _ring_box(group):
    r, c, e, d = _bbox(group)
    if min(e-r, d-c) < 5 or len(group) != 2*(e-r+d-c)-4:
        return None
    border = {(y, x) for y in range(r, e) for x in range(c, d)
              if y in (r, e-1) or x in (c, d-1)}
    return (r, c, e, d) if group == border else None


def _find_panels(a, bg):
    candidates = []
    for color in sorted(set(map(int, np.unique(a)))-{bg}):
        groups = _groups(zip(*np.nonzero(a == color)))
        boxes = [_ring_box(g) for g in groups]
        if len(boxes) < 3 or any(b is None for b in boxes):
            continue
        covered = np.zeros(a.shape, dtype=bool)
        valid = True
        for r, c, e, d in boxes:
            if covered[r:e, c:d].any():
                valid = False
                break
            covered[r:e, c:d] = True
        if valid and np.all(a[~covered] == bg):
            candidates.append((color, boxes))
    if len(candidates) != 1:
        raise ValueError("need one border color grouping all panels")
    return candidates[0]


def _legend_horizontal(tile, bg, frame):
    """Canonical bank: 2x2 ordered keys, a blank row, then binary masks."""
    n = tile.shape[0]-5
    if not 2 <= n <= 5 or (tile.shape[1]-1) % (n+1):
        return None
    k = (tile.shape[1]-1)//(n+1)
    if not 2 <= k <= 3:
        return None
    expected = np.full(tile.shape, bg, dtype=np.int8)
    expected[0] = expected[-1] = frame
    expected[:, 0] = expected[:, -1] = frame
    mapping, masks = {}, {}
    for i in range(k):
        c = 1+i*(n+1)
        key = tile[1:3, c:c+2]
        src, dst = int(key[0, 0]), int(key[0, 1])
        if (src == dst or src in (bg, frame) or dst in (bg, frame)
                or src in mapping or not np.array_equal(key, [[src, dst], [dst, dst]])):
            return None
        window = tile[4:4+n, c:c+n]
        if not np.all(np.isin(window, [src, bg])):
            return None
        mask = window == src
        if not (mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any()):
            return None
        mapping[src], masks[src] = dst, mask.copy()
        expected[1:3, c:c+2] = key
        expected[4:4+n, c:c+n] = window
    if not np.array_equal(tile, expected):
        return None
    return dict(n=n, mapping=mapping, masks=masks)


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if counts.max()*2 <= a.size:
        raise ValueError("background must occupy more than half the input")
    bg = int(colors[counts.argmax()])
    frame, boxes = _find_panels(a, bg)
    legends = []
    for index, (r, c, e, d) in enumerate(boxes):
        tile = a[r:e, c:d]
        for vertical in (False, True):
            legend = _legend_horizontal(tile.T if vertical else tile, bg, frame)
            if legend is not None:
                if vertical:
                    legend['masks'] = {s: m.T.copy() for s, m in legend['masks'].items()}
                legends.append((index, vertical, legend))
    if len(legends) != 1:
        raise ValueError("need one unambiguous legend bank")
    legend_index, vertical, legend = legends[0]
    sources = set(legend['mapping'])
    ink_colors = set(legend['mapping'].values())
    registry = set(map(int, np.unique(a)))-{bg, frame}-sources-ink_colors
    if len(registry) != 2:
        raise ValueError("registration uses two colors outside the ink table")
    n = legend['n']
    plates, seen_sources = [], set()
    for index, (r, c, e, d) in enumerate(boxes):
        if index == legend_index:
            continue
        tile = a[r+1:e-1, c+1:d-1]
        if not np.all(np.isin(tile, [bg]+sorted(registry | sources))):
            raise ValueError("unexpected plate content")
        marks = {}
        mark_cells = {}
        for group in _groups(zip(*np.nonzero(np.isin(tile, sorted(registry))))):
            y, x, yy, xx = _bbox(group)
            glyph = tile[y:yy, x:xx]
            if ((yy-y, xx-x) not in ((2, 3), (3, 2)) or len(group) != 6
                    or not all(np.count_nonzero(glyph == color) >= 2 for color in registry)):
                raise ValueError("registration marks must be full two-color six-cell motifs")
            signature = _key(glyph)
            if signature in marks:
                raise ValueError("a plate repeats a registration identity")
            marks[signature] = (y, x)
            mark_cells.update({p: int(tile[p]) for p in group})
        stamps = []
        for group in _groups(zip(*np.nonzero(np.isin(tile, sorted(sources))))):
            y, x, yy, xx = _bbox(group)
            block = tile[y:yy, x:xx]
            if ((yy-y, xx-x) != (n, n) or len(group) != n*n
                    or len(np.unique(block)) != 1):
                raise ValueError("placeholders must be separated solid size-matched squares")
            source = int(block[0, 0])
            stamps.append((y, x, source))
            seen_sources.add(source)
        if not marks or not 1 <= len(stamps) <= 3:
            raise ValueError("every plate needs registration and one to three stamps")
        # Whole rectangles, including inactive cells, have one-cell separation.
        contents = [(*_bbox(g),) for g in _groups(zip(*np.nonzero(tile != bg)))]
        for j, (y, x, yy, xx) in enumerate(contents):
            for z, w, zz, ww in contents[j+1:]:
                if not (yy < z or zz < y or xx < w or ww < x):
                    raise ValueError("plate objects need a blank-cell gap")
        plates.append(dict(marks=marks, mark_cells=mark_cells, stamps=stamps,
                           origin=(r+1, c+1), box=(r, c, e, d)))
    if seen_sources != sources or not 2 <= len(plates) <= 4:
        raise ValueError("legend categories must all be used on two to four plates")
    return dict(grid=a, background=bg, frame=frame, legend_box=boxes[legend_index],
                vertical_legend=vertical, registry=registry, plates=plates, **legend)


def _register(analysis):
    """Solve a connected exact motif-offset graph; no compactness heuristic."""
    plates = analysis['plates']
    occurrences = {}
    for i, plate in enumerate(plates):
        for key, anchor in plate['marks'].items():
            occurrences.setdefault(key, []).append((i, anchor))
    if any(len(group) < 2 for group in occurrences.values()):
        raise ValueError("every registration mark must have a matching copy")
    adjacency = [[] for _ in plates]
    for group in occurrences.values():
        i, (r, c) = group[0]
        for j, (y, x) in group[1:]:
            delta = (r-y, c-x)
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            proposed = (offsets[i][0]+dr, offsets[i][1]+dc)
            if j not in offsets:
                offsets[j] = proposed
                stack.append(j)
            elif offsets[j] != proposed:
                raise ValueError("inconsistent registration offsets")
    if len(offsets) != len(plates):
        raise ValueError("disconnected registration graph")
    marks, stamps = {}, []
    for i, plate in enumerate(plates):
        dr, dc = offsets[i]
        for (r, c), color in plate['mark_cells'].items():
            p = r+dr, c+dc
            if p in marks and marks[p] != color:
                raise ValueError("conflicting registration material")
            marks[p] = color
        stamps.extend((r+dr, c+dc, s) for r, c, s in plate['stamps'])
    n = analysis['n']
    if any((r+y, c+x) in marks for r, c, s in stamps for y in range(n) for x in range(n)):
        raise ValueError("a printing window intersects registration material")
    return marks, stamps, offsets, occurrences


def _print(marks, stamps, masks, mapping, bg, *, solid=False, merge=False):
    coverage, ink, rectangles = {}, {}, {}
    for r, c, source in stamps:
        mask = masks[source]
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                p = r+y, c+x
                rectangles[p] = rectangles.get(p, 0)+1
                if solid or mask[y, x]:
                    coverage[p] = coverage.get(p, 0)+1
                    ink[p] = mapping[source]
    cells = dict(marks)
    cells.update({p: ink[p] for p, count in coverage.items() if merge or count == 1})
    return _render(cells, bg), coverage, rectangles, cells


def _visible_sources(stamps, masks, coverage):
    counts = {s: 0 for s in masks}
    for r, c, source in stamps:
        for y, x in zip(*np.nonzero(masks[source])):
            if coverage[r+y, c+x] == 1:
                counts[source] += 1
    return counts


def _printed_inks(stamps, masks, mapping):
    inks = {}
    for r, c, source in stamps:
        for y, x in zip(*np.nonzero(masks[source])):
            inks.setdefault((r+y, c+x), set()).add(mapping[source])
    return inks


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: register plates, decode direct masks, remove all overprints."""
    a = _parse(input_grid)
    marks, stamps, _, _ = _register(a)
    return _print(marks, stamps, a['masks'], a['mapping'], a['background'])[0]


def _asymmetric(mask):
    others = [np.rot90(mask, k) for k in (1, 2, 3)]
    others += [np.rot90(np.fliplr(mask), k) for k in range(4)]
    return all(not np.array_equal(mask, other) for other in others)


def _mask(rng, n):
    y, x = np.indices((n, n))
    kind = int(rng.integers(8))
    if kind == 0:
        m = rng.random((n, n)) < rng.uniform(.25, .8)
    elif kind == 1:
        m = (x == 0) | (y == 0) | (x == n-1) | (y == n-1)
    elif kind == 2:
        m = (x == y) | (x+y == n-1)
    elif kind == 3:
        m = (x+y) % 2 == 0
    elif kind == 4:
        m = (y % 2 == 0) | (y == n-1)
    elif kind == 5:
        m = (x == n//2) | (y == n//2)
    elif kind == 6:
        m = rng.random((n, n)) < .85
    else:
        m = np.ones((n, n), dtype=bool)
        if rng.random() < .8:
            m[int(rng.integers(n)), int(rng.integers(n))] = False
    for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not edge.any():
            edge[int(rng.integers(n))] = True
    return m


def _mapping(rng, sources, palette, chain=False):
    external = [c for c in palette if c not in sources]
    if chain:
        return {s: sources[i+1] if i+1 < len(sources) else external[0]
                for i, s in enumerate(sources)}
    kind = int(rng.integers(4))
    if kind == 0:
        order = list(map(int, rng.permutation(sources)))
        return {s: order[(i+1) % len(order)] for i, s in enumerate(order)}
    if kind == 1:
        return {s: external[0] for s in sources}
    if kind == 2:
        targets = list(map(int, rng.choice(external, len(sources), replace=False)))
        return dict(zip(sources, targets))
    return {s: int(rng.choice([t for t in palette if t != s])) for s in sources}


def _framed_content(cells, bg, frame, rng, small=False):
    r, c, e, d = _bbox(cells)
    padding = list(map(int, rng.integers(0, 2 if small else 3, size=4)))
    h, w = e-r+2+padding[0]+padding[1], d-c+2+padding[2]+padding[3]
    tile = np.full((h, w), bg, dtype=np.int8)
    tile[0] = tile[-1] = frame
    tile[:, 0] = tile[:, -1] = frame
    for (y, x), color in cells.items():
        tile[y-r+1+padding[0], x-c+1+padding[2]] = color
    return tile


def _legend_tile(masks, mapping, bg, frame, rng):
    sources = list(map(int, rng.permutation(sorted(mapping))))
    n, k = next(iter(masks.values())).shape[0], len(sources)
    vertical = bool(rng.integers(2))
    tile = np.full((n+5, k*(n+1)+1), bg, dtype=np.int8)
    tile[0] = tile[-1] = frame
    tile[:, 0] = tile[:, -1] = frame
    for i, source in enumerate(sources):
        c = 1+i*(n+1)
        target = mapping[source]
        tile[1:3, c:c+2] = [[source, target], [target, target]]
        # Physical orientation is always the orientation stamped on the plates.
        mask = masks[source].T if vertical else masks[source]
        tile[4:4+n, c:c+n][mask] = source
    return tile.T.copy() if vertical else tile


def _pack(tiles, bg, rng):
    """Bounded shelf search, shuffled objects and random slack/margins."""
    options = []
    for order in permutations(range(len(tiles))):
        for cuts in range(1 << (len(tiles)-1)):
            rows, row = [], []
            for j, i in enumerate(order):
                row.append(i)
                if j == len(order)-1 or cuts & (1 << j):
                    rows.append(row)
                    row = []
            widths = [sum(tiles[i].shape[1] for i in row)+len(row)-1 for row in rows]
            heights = [max(tiles[i].shape[0] for i in row) for row in rows]
            h, w = sum(heights)+len(rows)-1, max(widths)
            if h <= 30 and w <= 30:
                options.append((rows, heights, widths, h, w))
    if not options:
        raise ValueError("panels do not fit a 30x30 canvas")
    rows, heights, widths, h, w = options[int(rng.integers(len(options)))]
    fg = sum(np.count_nonzero(t != bg) for t in tiles)
    valid_shapes = [(hh, ww) for hh in range(h, 31) for ww in range(w, 31) if hh*ww > 2*fg]
    if not valid_shapes:
        raise ValueError("not enough visible background")
    hh, ww = valid_shapes[int(rng.integers(len(valid_shapes)))]
    a = np.full((hh, ww), bg, dtype=np.int8)
    gaps_y = rng.multinomial(hh-h, np.full(len(rows)+1, 1/(len(rows)+1)))
    r = int(gaps_y[0])
    for j, row in enumerate(rows):
        gaps_x = rng.multinomial(ww-widths[j], np.full(len(row)+1, 1/(len(row)+1)))
        c = int(gaps_x[0])
        for z, i in enumerate(row):
            tile = tiles[i]
            y = r+int(rng.integers(heights[j]-tile.shape[0]+1))
            a[y:y+tile.shape[0], c:c+tile.shape[1]] = tile
            c += tile.shape[1]+1+int(gaps_x[z+1])
        r += heights[j]+1+int(gaps_y[j+1])
    return a


def _transform_point(p, rotation, reflected):
    r, c = p
    if reflected:
        c = -c
    for _ in range(rotation):
        r, c = -c, r
    return r, c


def _generate_pair(rng, profile, training, index):
    # The first two demonstrations teach the actual counterexamples even at 2/1.
    if training and index < 2:
        n, count, categories = 3, 2+index, 2
    elif not training and profile == 'scale':
        n, count = int(rng.choice([4, 5])), int(rng.choice([2, 3]))
        categories = min(count, int(rng.choice([2, 3])))
    elif not training:
        n, count, categories = int(rng.choice([2, 3])), 4, int(rng.choice([2, 3]))
    else:
        n, count = int(rng.choice([2, 3] if profile == 'scale' else [2, 3, 4])), int(rng.choice([2, 3]))
        categories = int(rng.choice([2, 3])) if n < 4 else 2
    for attempt in range(100):
        try:
            palette = list(map(int, rng.permutation(10)))
            bg, frame = palette[:2]
            reg_colors = palette[2:4]
            sources = palette[4:4+categories]
            mapping = _mapping(rng, sources, palette[4:], chain=training and index == 1)
            # First demonstration has distinct paints, so mask association matters.
            if training and index == 0:
                mapping = dict(zip(sources, palette[4+categories:4+2*categories]))
            graph = ('path' if training and index == 1 else
                     'path' if not training and profile == 'graph' else
                     str(rng.choice(['hub', 'path', 'triangle'])))
            if graph == 'triangle' and (count != 3 or n >= 4):
                graph = 'hub'
            if graph == 'hub':
                memberships = [set(range(count))]
                if rng.random() < .35:
                    memberships.append(set(map(int, rng.choice(count, 2, replace=False))))
            elif graph == 'path':
                memberships = [{i, i+1} for i in range(count-1)]
            else:
                memberships = [{0, 1}, {1, 2}, {0, 2}]
            num_marks = len(memberships)
            mark_cells = []
            codes = list(map(int, rng.choice([v for v in range(64) if 2 <= v.bit_count() <= 4],
                                           num_marks, replace=False)))
            key_column = 0
            bent_keys = num_marks == 3 and n <= 3 and count <= 3 and rng.random() < .25
            for j, code in enumerate(codes):
                bits = np.array([(code >> z) & 1 for z in range(6)]).reshape(2, 3)
                glyph = np.where(bits, reg_colors[0], reg_colors[1]).astype(np.int8)
                row = int(rng.integers(2)) if n <= 3 and rng.random() < .25 else 0
                column = key_column
                if bent_keys and j == 2:
                    row, column = 3+int(rng.integers(2)), 0
                cells = {(row+y, column+x): int(glyph[y, x]) for y in range(2) for x in range(3)}
                mark_cells.append(cells)
                key_column += int(rng.integers(4, 6)) if n <= 3 else 4
            stamp_row = max(r for cells in mark_cells for r, c in cells)+2
            stamp_col = max(0, (max(c for cells in mark_cells for r, c in cells)+1-n)//2)
            jitter = 1 if training and index == 1 else min(n-1, 2)
            starts = [(stamp_row+int(rng.integers(jitter+1)),
                       stamp_col+int(rng.integers(jitter+1))) for _ in range(count)]
            assignments = sources[:count] + list(map(int, rng.choice(sources, max(0, count-categories))))
            assignments = assignments[:count]
            rng.shuffle(assignments)
            stamps = [[(*p, s)] for p, s in zip(starts, assignments)]
            missing = [s for s in sources if s not in assignments]
            add_extra = bool(missing) or (count <= 3 and n <= 4 and rng.random() < .38)
            if add_extra:
                i = int(rng.integers(count))
                source = missing[0] if missing else int(rng.choice(sources))
                if rng.random() < .5:
                    r, c = starts[i][0], max(x for y, x in starts)+n+int(rng.integers(1, 4))
                else:
                    r, c = max(y for y, x in starts)+n+int(rng.integers(1, 4)), starts[i][1]
                stamps[i].append((r, c, source))
            if any(s not in {s for layer in stamps for r, c, s in layer} for s in sources):
                continue
            masks = {s: _mask(rng, n) for s in sources}
            # Make one genuine overlap visible; triple coverage is explicitly taught.
            r0, c0 = max(r for r, c in starts), max(c for r, c in starts)
            r1, c1 = min(r+n for r, c in starts), min(c+n for r, c in starts)
            if r0 >= r1 or c0 >= c1:
                continue
            common = int(rng.integers(r0, r1)), int(rng.integers(c0, c1))
            for layer in stamps:
                r, c, source = layer[0]
                masks[source][common[0]-r, common[1]-c] = True
            if training and index == 0 and not any(_asymmetric(m) for m in masks.values()):
                continue
            all_marks = {p: v for cells in mark_cells for p, v in cells.items()}
            all_stamps = [stamp for layer in stamps for stamp in layer]
            expected, coverage, rects, original_cells = _print(all_marks, all_stamps, masks, mapping, bg)
            # Each example defeats recoloring, union and rectangle-overlap shortcuts.
            if (not any(v >= 2 for v in coverage.values())
                    or not any(v == 1 and rects[p] >= 2 for p, v in coverage.items())
                    or sum(v == 1 for v in coverage.values()) < 4
                    or not all(_visible_sources(all_stamps, masks, coverage).values())):
                continue
            if training and index == 1 and not any(v >= 3 for v in coverage.values()):
                continue
            if training and index == 1:
                inks = _printed_inks(all_stamps, masks, mapping)
                if not any(v >= 2 and len(inks[p]) == 1 for p, v in coverage.items()):
                    continue
            if training and index == 0:
                # Ensure copying with the wrong stencil orientation changes the picture.
                rotated = {s: np.rot90(m) for s, m in masks.items()}
                if np.array_equal(expected, _print(all_marks, all_stamps, rotated, mapping, bg)[0]):
                    continue
            if np.array_equal(expected, _print(all_marks, all_stamps, masks, mapping, bg, solid=True)[0]):
                continue
            rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
            f = lambda p: _transform_point(p, rotation, reflected)
            masks = {s: np.rot90(np.fliplr(m) if reflected else m, rotation).copy()
                     for s, m in masks.items()}
            plates = []
            for i in range(count):
                content = {p: v for j, cells in enumerate(mark_cells) if i in memberships[j]
                           for p, v in cells.items()}
                for r, c, source in stamps[i]:
                    content.update({(r+y, c+x): source for y in range(n) for x in range(n)})
                plates.append({f(p): v for p, v in content.items()})
            # Expected pixels come directly from construction, before input packing.
            expected = _render({f(p): v for p, v in original_cells.items()}, bg)
            small = count == 4 or n >= 4
            tiles = [_framed_content(p, bg, frame, rng, small) for p in plates]
            tiles.append(_legend_tile(masks, mapping, bg, frame, rng))
            inp = _pack(tiles, bg, rng)
            if not np.array_equal(solve(inp), expected):
                raise AssertionError("construction and pixel reconstruction disagree")
            return dict(input=inp, output=expected)
        except ValueError:
            continue
    raise RuntimeError(f"bounded construction exhausted ({profile}, {training}, {index}, {n}, {count})")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    profile = str(rng.choice(['layers', 'scale', 'graph']))
    result, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            for _ in range(12):
                pair = _generate_pair(rng, profile, split == 'train', i)
                key = _key(pair['input'])
                if key not in seen:
                    result[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    return result


def _analyze_pair(pair):
    a = _parse(pair['input'])
    marks, stamps, offsets, occurrences = _register(a)
    expected, coverage, rectangles, _ = _print(marks, stamps, a['masks'], a['mapping'], a['background'])
    if not np.array_equal(_grid(pair['output']), expected):
        raise AssertionError("incorrect exclusive print or tight crop")
    # Both omissions are actual alternative predictions, not sampled flags.
    loose_marks, loose_stamps = {}, []
    for plate in a['plates']:
        dr, dc = plate['origin']
        loose_marks.update({(r+dr, c+dc): v for (r, c), v in plate['mark_cells'].items()})
        loose_stamps.extend((r+dr, c+dc, s) for r, c, s in plate['stamps'])
    no_a = _print(loose_marks, loose_stamps, a['masks'], a['mapping'], a['background'])[0]
    no_b = _print(marks, stamps, a['masks'], a['mapping'], a['background'], solid=True)[0]
    union = _print(marks, stamps, a['masks'], a['mapping'], a['background'], merge=True)[0]
    visible_sources = _visible_sources(stamps, a['masks'], coverage)
    inks = _printed_inks(stamps, a['masks'], a['mapping'])
    assert not np.array_equal(no_a, expected), "registration has no effect"
    assert not np.array_equal(no_b, expected), "stencils have no effect"
    assert not np.array_equal(union, expected), "exclusive printing has no effect"
    assert any(v == 1 and rectangles[p] >= 2 for p, v in coverage.items()), "rectangle clipping shortcut"
    assert sum(v == 1 for v in coverage.values()) >= 4, "too little surviving ink"
    assert all(visible_sources.values()), "a legend entry leaves no visible ink"
    assert set(map(int, np.unique(expected))) <= set(map(int, np.unique(a['grid'])))
    return dict(n=a['n'], layers=len(a['plates']), stamps=len(stamps), categories=len(a['mapping']),
                marks=len(occurrences), multiplicities=sorted(set(coverage.values())),
                collisions=sum(v >= 2 for v in coverage.values()),
                triple=sum(v >= 3 for v in coverage.values()),
                unanimous_erasure=sum(v >= 2 and len(inks[p]) == 1 for p, v in coverage.items()),
                transparent_overlap=sum(v == 1 and rectangles[p] >= 2 for p, v in coverage.items()),
                background=a['background'], frame=a['frame'], vertical=a['vertical_legend'],
                disconnected=sum(len(_groups(zip(*np.nonzero(m)))) > 1 for m in a['masks'].values()),
                asymmetric=sum(_asymmetric(m) for m in a['masks'].values()),
                map_overlap=sum(t in a['mapping'] for t in a['mapping'].values()),
                direct_witness=any(visible_sources[s] and t in a['mapping']
                                   and a['mapping'][t] != t for s, t in a['mapping'].items()),
                shared_paint=len(set(a['mapping'].values())) < len(a['mapping']),
                offsets=tuple(offsets.values()),
                graph=tuple(sorted(len(v) for v in occurrences.values())),
                mask_keys=tuple(_key(m) for m in a['masks'].values()))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise AssertionError("puzzle must have exactly train/test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise AssertionError("splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                _grid(grid)
            key = _key(pair['input'])
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze_pair(pair)
            if split == 'train':
                training.append(info)
    assert any(i['triple'] for i in training), "three-way erasure must be demonstrated"
    assert any(i['unanimous_erasure'] for i in training), "same-ink erasure must be demonstrated"
    assert any(i['asymmetric'] for i in training), "orientation must be demonstrated"
    assert any(i['direct_witness'] for i in training), "direct mapping must be demonstrated"


def _puzzle_key(puzzle):
    return tuple((_key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test():
    """Bounded 42-seed semantics, variation, count and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos, shapes, normalized, masks = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                infos.append(_analyze_pair(pair))
                masks.update(infos[-1]['mask_keys'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                roles = {}
                normalized.add((pair['input'].shape, tuple(roles.setdefault(int(v), len(roles))
                                                           for v in pair['input'].flat)))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 100 and len(normalized) >= 180 and len(masks) >= 100
    assert {i['n'] for i in infos} == {2, 3, 4, 5}
    assert {i['layers'] for i in infos} == {2, 3, 4}
    assert {i['categories'] for i in infos} == {2, 3}
    assert {i['background'] for i in infos} == set(range(10))
    assert {i['frame'] for i in infos} == set(range(10))
    assert {i['vertical'] for i in infos} == {False, True}
    assert len({i['graph'] for i in infos}) >= 5
    assert any(i['shared_paint'] for i in infos)
    assert sum(i['disconnected'] for i in infos) >= 30
    assert any(i['stamps'] > i['layers'] for i in infos)
    assert any(len({r for r, c in i['offsets']}) > 1
               and len({c for r, c in i['offsets']}) > 1 for i in infos)
    assert _puzzle_key(generate_puzzle(19)) == _puzzle_key(puzzles[19])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(900+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    rng = np.random.default_rng(71021)
    for puzzle in puzzles[:4]:
        for pair in puzzle['train'][:2]+puzzle['test']:
            permutation = rng.permutation(10)
            assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
            a = _parse(pair['input'])
            tiles = [a['grid'][r:e, c:d].copy() for r, c, e, d in
                     [a['legend_box']]+[p['box'] for p in a['plates']]]
            repacked = _pack(tiles, a['background'], rng)
            assert np.array_equal(solve(repacked), pair['output'])
    bad = {s: [dict(input=p['input'].copy(), output=p['output'].copy()) for p in puzzles[0][s]]
           for s in ('train', 'test')}
    a = _parse(bad['train'][0]['input'])
    output = bad['train'][0]['output']
    point = tuple(np.argwhere(output != a['background'])[0])
    output[point] = a['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("lost output material accepted")
    return dict(seeds=42, pairs=len(infos), distinct_structures=len(normalized),
                shape_combinations=len(shapes), mask_patterns=len(masks),
                disconnected_masks=sum(i['disconnected'] for i in infos),
                graph_profiles=len({i['graph'] for i in infos}))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
