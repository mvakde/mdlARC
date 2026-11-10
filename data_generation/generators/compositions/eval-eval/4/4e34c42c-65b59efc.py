"""Registered stencil erasure: input-only translation, identity union and printing.

Only Python's standard library and NumPy are used. No parent module is imported.
The sampler's world coordinates are used only to construct ground truth; solve
recovers its own coordinates from the returned pixels.
"""
from collections import defaultdict
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("invalid ARC grid")
    return a.astype(np.int8, copy=False)


def _groups(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cards(grid):
    """Recover every closed frame, independent of card order or shelf layout."""
    g = _grid(grid)
    bg = int(g[0, 0])
    _require(np.all(g[0] == bg) and np.all(g[-1] == bg)
             and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg),
             "missing outside margin")
    foreground = np.flatnonzero(g != bg)
    _require(len(foreground) > 0, "empty input")
    frame = int(g.flat[int(foreground[0])])
    cards = []
    covered = np.zeros(g.shape, dtype=bool)
    for component in _groups(zip(*np.nonzero(g == frame))):
        r0 = min(r for r, c in component)
        r1 = max(r for r, c in component) + 1
        c0 = min(c for r, c in component)
        c1 = max(c for r, c in component) + 1
        box = g[r0:r1, c0:c1]
        _require(min(box.shape) >= 4, "not a frame")
        _require(np.all(box[0] == frame) and np.all(box[-1] == frame)
                 and np.all(box[:, 0] == frame) and np.all(box[:, -1] == frame),
                 "incomplete rectangular frame")
        _require(not covered[r0:r1, c0:c1].any(), "overlapping input cards")
        covered[r0:r1, c0:c1] = True
        cards.append(box.copy())
    _require(np.all((g == bg) | covered), "foreground outside cards")
    _require(4 <= len(cards) <= 7, "wrong card count")
    return bg, frame, cards


def _parse(grid):
    bg, frame, cards = _cards(grid)
    masks, targets, library_order, plates = {}, {}, [], []
    n = None
    for card in cards:
        inner = card[1:-1, 1:-1]
        dividers = np.flatnonzero(np.all(inner == frame, axis=1))
        if len(dividers):
            side = inner.shape[1]
            _require(2 <= side <= 4 and inner.shape == (side+2, side)
                     and dividers.tolist() == [side], "malformed library card")
            if n is None:
                n = side
            _require(n == side, "mixed stencil sizes")
            bitmap = inner[:side]
            source_colors = np.unique(bitmap[bitmap != bg])
            _require(len(source_colors) == 1, "stencil needs one source identity")
            source = int(source_colors[0])
            _require(source != frame and source not in masks, "duplicate source")
            legend = inner[-1]
            target = int(legend[side//2])
            expected = np.full(side, bg, dtype=np.int8)
            expected[side//2] = target
            _require(target not in (bg, frame) and np.array_equal(expected, legend),
                     "malformed target swatch")
            masks[source] = (bitmap == source).copy()
            targets[source] = target
            library_order.append(source)
        else:
            _require(not np.any(inner == frame) and max(inner.shape) <= 7,
                     "malformed plate")
            plates.append(inner.copy())
    _require(n is not None and 2 <= len(masks) <= 3 and 2 <= len(plates) <= 4,
             "missing library or plates")
    source_set = set(masks)
    registration_colors = set().union(*(set(map(int, np.unique(p))) for p in plates))
    registration_colors -= {bg, *source_set}
    _require(len(registration_colors) == 2 and frame not in registration_colors,
             "registration needs two separate colors")

    occurrences = defaultdict(list)
    plate_commands, glyphs = [], []
    for i, plate in enumerate(plates):
        key_points = set(zip(*np.nonzero(np.isin(plate, list(registration_colors)))))
        local_glyphs = {}
        for group in _groups(key_points):
            r0, c0 = min(r for r, c in group), min(c for r, c in group)
            _require(group == {(r0+r, c0+c) for r in range(2) for c in range(2)},
                     "incomplete or touching glyph")
            pattern = tuple(map(int, plate[r0:r0+2, c0:c0+2].flat))
            _require(set(pattern) == registration_colors and pattern not in local_glyphs,
                     "ambiguous glyph identity")
            local_glyphs[pattern] = (r0, c0)
            occurrences[pattern].append((i, (r0, c0)))
        cmds = {(int(r), int(c)): int(plate[r, c])
                for r, c in zip(*np.nonzero(np.isin(plate, list(source_set))))}
        _require(local_glyphs and cmds, "plate needs glyphs and commands")
        plate_commands.append(cmds)
        glyphs.append(local_glyphs)
    _require(1 <= len(occurrences) <= 4
             and all(len(v) >= 2 for v in occurrences.values()), "unmatched glyph")

    # Every repeated exact bitmap directly supplies an offset equation.
    adjacency = [[] for _ in plates]
    for entries in occurrences.values():
        for a in range(len(entries)):
            i, (r, c) = entries[a]
            for j, (s, t) in entries[a+1:]:
                delta = (r-s, c-t)
                adjacency[i].append((j, delta))
                adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = [None] * len(plates)
    offsets[0] = (0, 0)
    stack = [0]
    while stack:
        i = stack.pop()
        for j, (dr, dc) in adjacency[i]:
            candidate = (offsets[i][0]+dr, offsets[i][1]+dc)
            if offsets[j] is None:
                offsets[j] = candidate
                stack.append(j)
            else:
                _require(offsets[j] == candidate, "inconsistent registration")
    _require(all(p is not None for p in offsets), "disconnected registration")

    key_world = {}
    for i, local in enumerate(glyphs):
        dr, dc = offsets[i]
        for pattern, (r, c) in local.items():
            for rr in range(2):
                for cc in range(2):
                    q, color = (r+rr+dr, c+cc+dc), pattern[rr*2+cc]
                    _require(q not in key_world or key_world[q] == (pattern, color),
                             "different glyphs overlap")
                    key_world[q] = (pattern, color)
    commands, raw = defaultdict(set), defaultdict(list)
    for i, local in enumerate(plate_commands):
        dr, dc = offsets[i]
        for (r, c), source in local.items():
            q = (r+dr, c+dc)
            _require(q not in key_world, "command overlaps registration material")
            commands[q].add(source)
            raw[q].append(source)
    _require(set().union(*commands.values()) == source_set, "unused source")
    _require(all(len(v) <= 2 for v in commands.values()), "more than two identities collide")
    material = set(commands) | set(key_world)
    _require(max(r for r, c in material)-min(r for r, c in material) < 7
             and max(c for r, c in material)-min(c for r, c in material) < 7,
             "registered plate material exceeds seven")
    r0, r1 = min(r for r, c in commands), max(r for r, c in commands)
    c0, c1 = min(c for r, c in commands), max(c for r, c in commands)
    _require(r1-r0 < 7 and c1-c0 < 7, "command domain exceeds seven")
    return dict(background=bg, frame=frame, n=n, masks=masks, targets=targets,
                library_order=library_order, plates=plates, cards=cards,
                plate_commands=plate_commands, glyphs=glyphs, offsets=offsets,
                commands=dict(commands), raw=dict(raw), keys=len(occurrences))


def _render(data, *, local=False, solid=False, deduplicate=True,
            erase=True, recolor=True, turn=0):
    """Block renderer; optional counterfactuals are used only by validation."""
    n, bg = data['n'], data['background']
    cmds = data['commands'] if deduplicate else data['raw']
    if local:
        cmds = defaultdict(set)
        for plate in data['plate_commands']:
            for p, source in plate.items():
                cmds[p].add(source)
    r0, r1 = min(r for r, c in cmds), max(r for r, c in cmds)
    c0, c1 = min(c for r, c in cmds), max(c for r, c in cmds)
    result = np.full(((r1-r0+1)*n, (c1-c0+1)*n), bg, dtype=np.int8)
    for (r, c), sources in sorted(cmds.items()):
        coverage = np.zeros((n, n), dtype=int)
        ink = np.full((n, n), bg, dtype=np.int8)
        for source in sorted(sources):
            mask = np.ones((n, n), dtype=bool) if solid else data['masks'][source]
            if turn >= 4:
                mask = np.fliplr(mask)
            mask = np.rot90(mask, turn % 4)
            coverage += mask
            ink[mask] = data['targets'][source] if recolor else source
        block = np.where(coverage == 1 if erase else coverage > 0, ink, bg)
        result[(r-r0)*n:(r-r0+1)*n, (c-c0)*n:(c-c0+1)*n] = block
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve only from visible input, with no examples, seed or latent state."""
    return _render(_parse(input_grid))


def _ground_truth(commands, masks, targets, bg, n):
    """Construction answer from sampled world sets, using expanded layer masks."""
    r0, r1 = min(r for r, c in commands), max(r for r, c in commands)
    c0, c1 = min(c for r, c in commands), max(c for r, c in commands)
    shape = (r1-r0+1, c1-c0+1)
    expanded = {}
    for source, stencil in masks.items():
        layer = np.zeros(shape, dtype=bool)
        for (r, c), identities in commands.items():
            layer[r-r0, c-c0] = source in identities
        expanded[source] = np.kron(layer, stencil)
    count = sum(expanded.values())
    output = np.full((shape[0]*n, shape[1]*n), bg, dtype=np.int8)
    for source, pixels in expanded.items():
        output[pixels & (count == 1)] = targets[source]
    return output


def _mask(rng, n):
    style = int(rng.integers(7))
    rr, cc = np.indices((n, n))
    if style == 0:
        mask = rng.random((n, n)) < rng.uniform(.25, .85)
    elif style == 1:
        mask = (rr == 0) | (cc == 0) | (rr == n-1) | (cc == n-1)
    elif style == 2:
        mask = (rr+cc) % 2 == int(rng.integers(2))
    elif style == 3:
        mask = rr == cc
    elif style == 4:
        mask = cc == int(rng.integers(n))
    elif style == 5:
        mask = np.ones((n, n), dtype=bool)
    else:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    return np.rot90(mask, int(rng.integers(4))).copy()


def _templates(rng, n, k, witness):
    masks = [_mask(rng, n) for _ in range(k)]
    if witness == 0:
        for _ in range(128):
            a = rng.random((n, n)) < .60
            b = rng.random((n, n)) < .60
            if ((a & b).any() and (a & ~b).any() and (b & ~a).any()
                    and all(any(not np.array_equal(m, np.rot90(
                        np.fliplr(m) if t >= 4 else m, t % 4)) for m in (a, b))
                        for t in range(1, 8))):
                masks[0], masks[1] = a, b
                break
        else:
            # Two different missing corners distinguish all coupled symmetries.
            masks[0] = np.ones((n, n), dtype=bool)
            masks[1] = np.ones((n, n), dtype=bool)
            masks[0][0, 0] = False
            masks[1][0, -1] = False
    if witness == 1:
        masks[1] = masks[0].copy()
    return masks


def _library_card(mask, source, target, bg, frame):
    n = mask.shape[0]
    card = np.full((n+4, n+2), frame, dtype=np.int8)
    card[1:n+1, 1:n+1] = np.where(mask, source, bg)
    card[n+2, 1:n+1] = bg
    card[n+2, 1+n//2] = target
    return card


def _plate_card(cells, h, w, bg, frame, rng):
    r0, r1 = min(r for r, c in cells), max(r for r, c in cells)
    c0, c1 = min(c for r, c in cells), max(c for r, c in cells)
    # Modest blank plate margins change local origins without changing evidence.
    r0 = max(0, r0-int(rng.integers(2)))
    c0 = max(0, c0-int(rng.integers(2)))
    r1 = min(h-1, r1+int(rng.integers(2)))
    c1 = min(w-1, c1+int(rng.integers(2)))
    card = np.full((r1-r0+3, c1-c0+3), frame, dtype=np.int8)
    card[1:-1, 1:-1] = bg
    for (r, c), color in cells.items():
        card[r-r0+1, c-c0+1] = color
    return card


def _pack(cards, bg, rng):
    """Variable shelves; a bounded three-row fallback always fits in 30."""
    options = []
    for _ in range(24):
        capacity = int(rng.integers(18, 29))
        order = rng.permutation(len(cards)).tolist()
        rows, row, used = [], [], 0
        for i in order:
            width = cards[i].shape[1]
            if row and used+1+width > capacity:
                rows.append(row)
                row, used = [], 0
            row.append(i)
            used += width + int(len(row) > 1)
        if row:
            rows.append(row)
        height = sum(max(cards[i].shape[0] for i in row) for row in rows)+len(rows)-1
        if height <= 28:
            options.append(rows)
    if options:
        rows = options[int(rng.integers(len(options)))]
    else:
        library = [i for i, a in enumerate(cards) if np.any(np.all(a[1:-1, 1:-1]
                    == a[0, 0], axis=1))]
        plates = [i for i in range(len(cards)) if i not in library]
        rng.shuffle(library)
        rng.shuffle(plates)
        rows = [library] + [plates[i:i+2] for i in range(0, len(plates), 2)]
        rng.shuffle(rows)
    heights = [max(cards[i].shape[0] for i in row) for row in rows]
    widths = [sum(cards[i].shape[1] for i in row)+len(row)-1 for row in rows]
    core_h = sum(heights)+len(rows)-1
    core_w = max(widths)
    _require(core_h <= 28 and core_w <= 28, "card packing exceeds bounds")
    extra_h = int(rng.integers(min(4, 28-core_h)+1))
    extra_w = int(rng.integers(min(4, 28-core_w)+1))
    result = np.full((core_h+2+extra_h, core_w+2+extra_w), bg, dtype=np.int8)
    top = 1+int(rng.integers(extra_h+1))
    for row, height, width in zip(rows, heights, widths):
        left = 1+int(rng.integers(core_w+extra_w-width+1))
        for i in row:
            card = cards[i]
            r = top+int(rng.integers(height-card.shape[0]+1))
            result[r:r+card.shape[0], left:left+card.shape[1]] = card
            left += card.shape[1]+1
        top += height+1
    return result


def _sample_pair(rng, n, witness, test, density, previous_bg):
    palette = list(map(int, rng.permutation(10)))
    bg, frame, key_a, key_b = palette[:4]
    if bg == previous_bg:
        bg, frame = frame, bg
    k = 2 if witness == 0 else 3 if witness == 1 else int(rng.integers(2, 4))
    sources = palette[4:4+k]
    masks = dict(zip(sources, _templates(rng, n, k, witness)))
    target_pool = [v for v in range(10) if v not in (bg, frame)]
    for _ in range(32):
        target_values = list(map(int, rng.choice(target_pool, size=k, replace=False)))
        if witness not in (0, 1) or all(a != b for a, b in zip(sources, target_values)):
            break
    else:
        # There are at least eight candidates and at most three identities.
        # Greedy selection excluding each own identity always has spare colors.
        target_values = []
        for source in sources:
            target_values.append(next(t for t in target_pool
                                      if t != source and t not in target_values))
    if witness not in (0, 1):
        if rng.random() < .25:
            target_values[1] = target_values[0]
        if rng.random() < .20:
            target_values[-1] = sources[-1]
    targets = dict(zip(sources, target_values))
    p = 2 if witness == 0 else 3 if witness == 1 else int(rng.integers(3 if test else 2, 5))
    h, w = map(int, rng.integers(5, 8, size=2))

    # Glyph identities are edges of a connected tree, optionally with a cycle.
    order = rng.permutation(p).tolist()
    edges = [(order[j], order[int(rng.integers(j))]) for j in range(1, p)]
    if witness == 1:
        edges = [(0, 1), (1, 2)]
    if witness not in (0, 1) and p >= 3 and rng.random() < .35:
        extra = [(i, j) for i in range(p) for j in range(i+1, p)
                 if (i, j) not in edges and (j, i) not in edges]
        if extra and len(edges) < 4:
            edges.append(extra[int(rng.integers(len(extra)))])
    if witness == 0:
        key_positions = [(h//2-1, w//2-1)]
    else:
        corners = [(0, 0), (0, w-2), (h-2, 0), (h-2, w-2)]
        key_positions = [corners[i] for i in rng.permutation(4)[:len(edges)]]
    codes = list(map(int, rng.choice(np.arange(1, 15), size=len(edges), replace=False)))
    cells = [{} for _ in range(p)]
    incident = [[] for _ in range(p)]
    key_cells = set()
    for (i, j), (r0, c0), code in zip(edges, key_positions, codes):
        for z in (i, j):
            incident[z].append((r0, c0))
        for r in range(2):
            for c in range(2):
                point = (r0+r, c0+c)
                color = key_a if (code >> (r*2+c)) & 1 else key_b
                key_cells.add(point)
                cells[i][point] = cells[j][point] = color

    available = [(r, c) for r in range(h) for c in range(w) if (r, c) not in key_cells]
    plate_cmds = [{} for _ in range(p)]
    world = defaultdict(set)

    def put(z, pos, source):
        _require(pos not in key_cells and (pos not in plate_cmds[z]
                 or plate_cmds[z][pos] == source), "construction command conflict")
        _require(len(world[pos] | {source}) <= 2, "construction triple collision")
        plate_cmds[z][pos] = source
        world[pos].add(source)

    if witness == 1:
        # One completely erased command row distinguishes pre-erasure cropping.
        boundary_row = min(r for r, c in available)
        choices = [q for q in available if q[0] == boundary_row]
        collision = choices[int(rng.integers(len(choices)))]
        remaining = [q for q in available if q[0] > boundary_row]
    elif witness == 0:
        middle = [q for q in available if 0 < q[1] < w-1 and 0 < q[0] < h-1]
        collision = middle[int(rng.integers(len(middle)))]
        remaining = [q for q in available if q != collision]
    else:
        collision = available[int(rng.integers(len(available)))]
        remaining = [q for q in available if q != collision]
    put(0, collision, sources[0])
    put(p-1, collision, sources[1])
    if witness == 0:
        choices = [q for q in remaining if 0 < q[1] < w-1 and 0 < q[0] < h-1]
    else:
        choices = remaining
    duplicate = choices[int(rng.integers(len(choices)))]
    put(0, duplicate, sources[0])
    put(1, duplicate, sources[0])
    remaining.remove(duplicate)
    if collision in remaining:
        remaining.remove(collision)
    for j, source in enumerate(sources):
        if witness == 0:
            extreme = min(c for r, c in remaining) if j == 0 else max(c for r, c in remaining)
            choices = [q for q in remaining if q[1] == extreme]
        else:
            choices = remaining
        pos = choices[int(rng.integers(len(choices)))]
        remaining.remove(pos)
        put(j % p, pos, source)
    # Guarantee a nonempty contribution on every plate, including four-plate cases.
    for z in range(p):
        if not plate_cmds[z]:
            pos = remaining.pop(int(rng.integers(len(remaining))))
            put(z, pos, sources[z % k])

    # Structured order controls occupancy, independently of labels and templates.
    style = int(rng.integers(5))
    if style == 0:
        rng.shuffle(remaining)
    elif style == 1:
        remaining.sort(key=lambda q: (q[0], q[1]))
    elif style == 2:
        remaining.sort(key=lambda q: (sum(q), q[0]))
    elif style == 3:
        remaining.sort(key=lambda q: (min(q[0], q[1], h-1-q[0], w-1-q[1]), q))
    else:
        center = tuple(map(int, rng.integers(0, [h, w])))
        remaining.sort(key=lambda q: (abs(q[0]-center[0])+abs(q[1]-center[1]), q))
    extra_count = min(len(remaining), int(rng.integers(2 if test else 0,
                        (12 if test else 7) + int(5*density))))
    for pos in remaining[:extra_count]:
        num = 2 if rng.random() < (.42 if test else .24) else 1
        identities = list(map(int, rng.choice(sources, size=num, replace=False)))
        used_plates = set()
        for source in identities:
            eligible = [z for z in range(p) if z not in used_plates]
            if witness == 0:
                z = 0 if pos[1] <= w//2 else 1
                if z in used_plates:
                    z = 1-z
            else:
                distances = [min(abs(pos[0]-r)+abs(pos[1]-c) for r, c in incident[z])
                             for z in eligible]
                weights = np.exp(-np.array(distances, dtype=float)*.35)
                z = int(rng.choice(eligible, p=weights/weights.sum()))
            put(z, pos, source)
            used_plates.add(z)
            if num == 1 and rng.random() < .15:
                other = int(rng.choice([t for t in range(p) if t != z]))
                put(other, pos, source)

    library = [_library_card(masks[s], s, targets[s], bg, frame)
               for s in rng.permutation(sources)]
    plates = []
    for z in range(p):
        cells[z].update(plate_cmds[z])
        plates.append(_plate_card(cells[z], h, w, bg, frame, rng))
    inp = _pack(library+plates, bg, rng)
    out = _ground_truth(world, masks, targets, bg, n)
    parsed = _parse(inp)
    _require(np.array_equal(_render(parsed), out), "world and recovered answer disagree")
    return {'input': inp, 'output': out}


def _same(a, b):
    return np.array_equal(a, b)


def _tight_ink(a, bg):
    rr, cc = np.nonzero(a != bg)
    if not len(rr):
        return np.full((1, 1), bg, dtype=np.int8)
    return a[rr.min():rr.max()+1, cc.min():cc.max()+1]


def _witness_checks(pair, which):
    d = _parse(pair['input'])
    expected = np.asarray(pair['output'])
    _require(any(len(v) == 2 for v in d['commands'].values()), "no distinct collision")
    _require(any(len(v) > len(set(v)) for v in d['raw'].values()), "no duplicate command")
    if which == 0:
        _require(not _same(expected, _render(d, local=True)), "A omission not distinguished")
        _require(not _same(expected, _render(d, solid=True)), "B omission not distinguished")
        _require(not _same(expected, _render(d, erase=False)), "union shortcut not distinguished")
        _require(not _same(expected, _render(d, deduplicate=False)), "raw count not distinguished")
        _require(not _same(expected, _render(d, recolor=False)), "key shortcut not distinguished")
        _require(all(not _same(expected, _render(d, turn=t)) for t in range(1, 8)),
                 "stencil orientation not distinguished")
        partial = False
        for identities in d['commands'].values():
            if len(identities) == 2:
                a, b = [d['masks'][s] for s in sorted(identities)]
                partial |= bool((a & b).any() and (a & ~b).any() and (b & ~a).any())
        _require(partial, "whole-block erasure not distinguished")
    if which == 1:
        _require(len(d['plates']) == 3 and d['keys'] == 2, "transitivity witness missing")
        _require(sorted(map(len, d['glyphs'])) == [1, 1, 2],
                 "transitive chain not distinguished from common-anchor registration")
        identical_collision = any(len(ids) == 2 and _same(*[d['masks'][s] for s in ids])
                                  for ids in d['commands'].values())
        _require(identical_collision, "equal silhouette identity witness missing")
        top = min(r for r, c in d['commands'])
        boundary = [ids for (r, c), ids in d['commands'].items() if r == top]
        _require(all(len(ids) == 2 and _same(*[d['masks'][s] for s in ids])
                     for ids in boundary)
                 and np.all(expected[:d['n']] == d['background']),
                 "complete boundary-row cancellation missing")
        _require(not _same(expected, _tight_ink(expected, d['background'])),
                 "command crop not distinguished")
        _require(not _same(expected, _render(d, deduplicate=False)),
                 "duplicate survival not distinguished")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    first_n = int(rng.integers(2, 4))
    scales = [first_n, first_n+1]
    density = float(rng.choice([.15, .45, .85]))
    puzzle, seen, first_bg = {'train': [], 'test': []}, set(), None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = index if split == 'train' and index < 2 else -1
            n = scales[index] if witness >= 0 else int(rng.choice(scales))
            for _ in range(96):
                try:
                    pair = _sample_pair(rng, n, witness, split == 'test', density,
                                        first_bg if witness == 1 else None)
                    _witness_checks(pair, witness)
                    if split == 'test':
                        d = _parse(pair['input'])
                        _require(not _same(pair['output'], _render(d, local=True)),
                                 "test does not need recovered translations")
                        _require(not _same(pair['output'], _render(d, solid=True)),
                                 "test does not need stencil geometry")
                except ValueError:
                    continue
                signature = _signature(pair['input'])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                if witness == 0:
                    first_bg = int(pair['input'][0, 0])
                break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, data = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            d = _parse(inp)
            _require(_same(_render(d), out), "incorrect registered stencil print")
            _require(set(map(int, np.unique(out))) <= {d['background'], *d['targets'].values()},
                     "metadata or source ink leaked into output")
            _require(all(np.any(out == target) for target in d['targets'].values()),
                     "a keyed ink is never visible")
            _require(all(any(ids == {source} for ids in d['commands'].values())
                         for source in d['masks']), "missing exclusive identity witness")
            _witness_checks(pair, -1)
            data[split].append(d)
    _witness_checks(puzzle['train'][0], 0)
    _witness_checks(puzzle['train'][1], 1)
    _require(data['train'][0]['background'] != data['train'][1]['background'],
             "palette role change not demonstrated")
    demonstrated = {d['n'] for d in data['train']}
    _require(all(d['n'] in demonstrated for d in data['test']), "undemonstrated test scale")
    _require(all(len(d['plates']) >= 3 for d in data['test']), "test graph shift missing")
    _require(all(not _same(_render(d), _render(d, local=True))
                 and not _same(_render(d), _render(d, solid=True)) for d in data['test']),
             "test must exercise both inherited mechanisms")


def _puzzle_signature(puzzle):
    return tuple((_signature(p['input']), _signature(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded finite-seed diversity, count, metamorphic and corruption checks."""
    shapes, output_shapes, motifs, graphs, layouts, backgrounds, scales = (
        set() for _ in range(7))
    signatures, records = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _parse(pair['input'])
                _require(_same(solve(pair['input'].copy()), pair['output']), "solve mismatch")
                shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                backgrounds.add(d['background'])
                scales.add(d['n'])
                motifs.update(_signature(m.astype(np.int8)) for m in d['masks'].values())
                graphs.add((len(d['plates']), d['keys']))
                r0 = min(r for r, c in d['commands'])
                c0 = min(c for r, c in d['commands'])
                ids = {s: i for i, s in enumerate(sorted(d['masks']))}
                layouts.add(tuple((r-r0, c-c0, tuple(sorted(ids[s] for s in v)))
                                  for (r, c), v in sorted(d['commands'].items())))
                records.append(d)
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 20 and len(output_shapes) >= 15, "shape diversity")
    _require(len(motifs) >= 50 and len(layouts) >= 150, "structural diversity")
    _require(backgrounds == set(range(10)) and scales == {2, 3, 4}, "palette/scale coverage")
    _require({len(d['plates']) for d in records} == {2, 3, 4}, "plate-count coverage")
    _require({d['keys'] for d in records} == {1, 2, 3, 4}, "graph coverage")
    _require(any(d['keys'] >= len(d['plates']) for d in records), "cycle coverage")
    _require(any(len(set(d['targets'].values())) < len(d['targets']) for d in records),
             "shared target coverage")

    # A concrete small rule check: distinct sources erase shared pixels, while
    # duplicate observations of one source retain the full stencil.
    a = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=bool)
    tiny = dict(n=3, background=0, masks={1: a, 4: b}, targets={1: 2, 4: 7},
                commands={(0, 0): {1, 4}}, raw={(0, 0): [1, 4]})
    _require(_same(_render(tiny), [[0, 2, 0], [0, 0, 7], [0, 0, 0]]),
             "partial erasure rule")
    tiny.update(commands={(0, 0): {1}}, raw={(0, 0): [1, 1]})
    _require(_same(_render(tiny), [[2, 2, 0], [2, 0, 0], [0, 0, 0]]),
             "duplicate observation rule")

    # The host also exercises all these combinations; keep our own test bounded.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(812, nt, nq)
            _require((len(p['train']), len(p['test'])) == (nt, nq), "count mismatch")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (3, False), (None, 1), (3, None)):
        try:
            generate_puzzle(1, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_signature(generate_puzzle(999, 3, 1))
             == _puzzle_signature(generate_puzzle(999, 3, 1)), "determinism")

    sample = generate_puzzle(0)
    rng = np.random.default_rng(9001)
    for pair in sample['train'] + sample['test']:
        d = _parse(pair['input'])
        repacked = _pack(d['cards'], d['background'], rng)
        _require(_same(solve(repacked), pair['output']), "packing affected registration")
        palette = rng.permutation(10)
        _require(_same(solve(palette[pair['input']]), palette[pair['output']]),
                 "literal palette affected answer")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A damaged shared glyph and a source removed from its library are errors.
    d = _parse(sample['test'][0]['input'])
    for kind in ('glyph', 'source'):
        cards = [card.copy() for card in d['cards']]
        for card in cards:
            inner = card[1:-1, 1:-1]
            has_divider = np.any(np.all(inner == d['frame'], axis=1))
            if kind == 'source' and has_divider:
                inner[:d['n']] = d['background']
                break
            if kind == 'glyph' and not has_divider:
                colors = set(map(int, np.unique(inner))) - {d['background'], *d['masks']}
                rr, cc = np.nonzero(np.isin(inner, list(colors)))
                inner[rr[0], cc[0]] = d['background']
                break
        try:
            solve(_pack(cards, d['background'], rng))
        except ValueError:
            pass
        else:
            raise AssertionError(f"damaged {kind} accepted")
    frame_bad = sample['test'][0]['input'].copy()
    rr, cc = np.nonzero(frame_bad == d['frame'])
    frame_bad[rr[0], cc[0]] = d['background']
    try:
        solve(frame_bad)
    except ValueError:
        pass
    else:
        raise AssertionError("broken frame accepted")
    return {'seeds': 42, 'pairs': len(records), 'input_shapes': len(shapes),
            'output_shapes': len(output_shapes), 'template_bitmaps': len(motifs),
            'command_layouts': len(layouts), 'registration_graph_sizes': sorted(graphs),
            'independent_solver': False}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
