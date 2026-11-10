"""Shattered stencils: pack rigid fragments into a glyph-expanded blueprint.

All semantic facts are recovered from the input. Sampling schedules are only
demonstration choices; the solver does not read seeds, labels, or files.
"""

from __future__ import annotations

import json
import numpy as np


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = EDGE + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Bad color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, neighbors=EDGE):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
                    cells.append(point)
        result.append(cells)
    return result


def _tight(a, points):
    coords = np.array(points)
    low, high = coords.min(axis=0), coords.max(axis=0) + 1
    result = np.full(tuple(high - low), -1, dtype=np.int8)
    for r, c in points:
        result[r - low[0], c - low[1]] = a[r, c]
    return result


def _rotations(a):
    result, seen = [], set()
    for k in range(4):
        b = np.rot90(a, k).copy()
        key = _key(b)
        if key not in seen:
            seen.add(key)
            result.append(b)
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    _require(np.sum(counts == counts.max()) == 1, "Background must be uniquely most frequent")
    bg = int(values[np.argmax(counts)])
    components = _components(a != bg, EIGHT)
    mixed = [p for p in components if len({int(a[q]) for q in p}) > 1]
    _require(len(mixed) == 1, "Exactly one two-color registration fragment is required")
    marked = mixed[0]
    vals, nums = np.unique([a[q] for q in marked], return_counts=True)
    _require(len(vals) == 2 and sum(nums == 1) == 1, "Registration fragment has one exceptional pixel")
    reg = int(vals[np.argmin(nums)])
    glyphs = [p for p in components if len(p) >= 3 and all(a[q] == reg for q in p)]
    _require(len(glyphs) == 1, "Exactly one monochrome registration-color glyph is required")
    glyph_points = glyphs[0]
    glyph = _tight(a, glyph_points) == reg
    h, w = glyph.shape
    _require(h >= 2 and w >= 2 and glyph[0, 0], "Glyph has two axes and an occupied upper-left pixel")
    _require(len(_components(glyph)) == 1, "Glyph must be edge connected")
    singletons = [p[0] for p in components if len(p) == 1]
    anchors = [p for p in singletons if a[p] == reg]
    _require(len(anchors) == 1, "Exactly one isolated map anchor is required")
    ordinary = [p for p in singletons if a[p] != reg]
    _require(len(ordinary) >= 3, "Map needs at least three ordinary points")
    map_colors = {int(a[p]) for p in ordinary}
    _require(len(map_colors) == 1, "Ordinary map points share a color")
    map_color = map_colors.pop()
    points = anchors + ordinary
    low = np.min(points, axis=0)
    _require(all((r - low[0]) % 2 == 0 and (c - low[1]) % 2 == 0 for r, c in points),
             "Map points lie on a pitch-two lattice")
    coarse = [((r - int(low[0])) // 2, (c - int(low[1])) // 2) for r, c in points]
    mh, mw = max(r for r, _ in coarse) + 1, max(c for _, c in coarse) + 1
    _require(mh * h <= 30 and mw * w <= 30, "Expanded stencil exceeds ARC bounds")
    layout = np.zeros((mh, mw), dtype=bool)
    for p in coarse:
        layout[p] = True
    _require(len(_components(layout)) == 1, "Map must be edge connected in coarse coordinates")
    anchor = coarse[0]
    fragment_points = [p for p in components if len(p) > 1 and p is not glyph_points]
    _require(3 <= len(fragment_points) <= 5, "Supply three to five fragments")
    carriers, pieces = [], []
    for cells in fragment_points:
        _require(len(cells) >= 3, "Fragments have at least three cells")
        colors = {int(a[p]) for p in cells} - {reg}
        _require(len(colors) == 1, "Each fragment has one carrier color")
        carrier = colors.pop()
        _require(carrier not in (bg, reg, map_color), "Semantic roles are distinct")
        carriers.append(carrier)
        piece = _tight(a, cells)
        _require(len(_components(piece >= 0)) == 1, "Fragments are edge connected")
        pieces.append(piece)
    _require(len(set(carriers)) == len(carriers), "Carrier colors identify whole fragments")
    target = np.kron(layout, glyph)
    _require(sum(int(np.sum(p >= 0)) for p in pieces) == int(target.sum()),
             "Fragment area must equal expanded stencil area")
    return dict(input=a, bg=bg, reg=reg, map_color=map_color, glyph=glyph,
                layout=layout, anchor=anchor, target=target,
                pin=(anchor[0] * h, anchor[1] * w), pieces=pieces, carriers=carriers,
                map_points=points, glyph_points=glyph_points)


def _pack(pieces, target, reg, pin, *, limit=2, rotations=True):
    """Enumerate exact colored covers; two answers suffice for rejection.

    The solver always supplies a pin coordinate, so a piece's occupied cells
    determine all its colors, including the fixed R location. None is used
    only for the registration ablation: it counts distinct carrier footprints
    and merges orientations that merely move R inside the same footprint.
    Multiple such covers are a stronger witness than marker-only ambiguity.
    """
    H, W = target.shape
    goal = sum(1 << (int(r) * W + int(c)) for r, c in np.argwhere(target))
    if sum(int(np.sum(p >= 0)) for p in pieces) != goal.bit_count():
        return []
    choices, marked = [], []
    for i, piece in enumerate(pieces):
        options = {}
        is_marked = bool(np.any(piece == reg))
        if is_marked:
            marked.append(i)
        for q in _rotations(piece) if rotations else [piece]:
            h, w = q.shape
            if h > H or w > W:
                continue
            cells = [(int(r), int(c), int(q[r, c])) for r, c in np.argwhere(q >= 0)]
            if is_marked and pin is not None:
                rp, cp = map(int, np.argwhere(q == reg)[0])
                origins = [(pin[0] - rp, pin[1] - cp)]
            else:
                origins = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            for r, c in origins:
                if r < 0 or c < 0 or r + h > H or c + w > W:
                    continue
                positions = tuple(((r + rr) * W + c + cc, color) for rr, cc, color in cells)
                bits = sum(1 << pos for pos, _ in positions)
                if bits & goal == bits:
                    options.setdefault(bits, (bits, positions))
        if not options:
            return []
        choices.append(list(options.values()))
    if len(marked) != 1:
        return []
    # Candidate lists indexed by covered cell avoid scanning all translations
    # at every search node. MRV chooses a cell with few legal ways to cover it.
    indexes = []
    for options in choices:
        index = {}
        for option in options:
            for pos, _ in option[1]:
                index.setdefault(pos, []).append(option)
        indexes.append(index)
    answers, signatures, placed = [], set(), [None] * len(pieces)

    def visit(occupied, remaining):
        if len(answers) >= limit:
            return
        if not remaining:
            if occupied == goal:
                signature = tuple(q[0] for q in placed)
                if signature not in signatures:
                    signatures.add(signature)
                    out = np.full(target.shape, -1, dtype=np.int8)
                    for option in placed:
                        for pos, color in option[1]:
                            out.flat[pos] = color
                    answers.append(out)
            return
        empty, best = goal ^ occupied, None
        while empty:
            bit = empty & -empty
            pos = bit.bit_length() - 1
            viable = [(i, q) for i in remaining for q in indexes[i].get(pos, ())
                      if not q[0] & occupied]
            if not viable:
                return
            if best is None or len(viable) < len(best):
                best = viable
                if len(best) == 1:
                    break
            empty ^= bit
        for i, q in best:
            placed[i] = q
            visit(occupied | q[0], tuple(j for j in remaining if j != i))
            placed[i] = None
            if len(answers) >= limit:
                return

    i = marked[0]
    remaining = tuple(j for j in range(len(pieces)) if j != i)
    for option in choices[i]:
        placed[i] = option
        visit(option[0], remaining)
        if len(answers) >= limit:
            break
    return answers


def solve(input_grid: np.ndarray) -> np.ndarray:
    facts = _decode(input_grid)
    answers = _pack(facts['pieces'], facts['target'], facts['reg'], facts['pin'])
    _require(len(answers) == 1, "Stencil must have exactly one colored rigid assembly")
    out = answers[0].copy()
    out[out < 0] = facts['bg']
    return out


def _glyph(rng, h, w):
    # A crossing row/column gives matching contacts on opposite edges. Its
    # position varies, avoiding a fixed L silhouette; a path joins (0, 0).
    mask = np.zeros((h, w), dtype=bool)
    spine_r, spine_c = int(rng.integers(h)), int(rng.integers(w))
    mask[spine_r, :] = True
    mask[:, spine_c] = True
    r = c = 0
    mask[0, 0] = True
    steps = [0] * spine_r + [1] * spine_c
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    density = float(_pick(rng, (0.0, .35, .55, .75, 1.0)))
    goal = max(h + w - 1, int(round(h * w * density)))
    while mask.sum() < goal:
        frontier = sorted({(r + dr, c + dc) for r, c in np.argwhere(mask) for dr, dc in EDGE
                           if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]})
        mask[_pick(rng, frontier)] = True
    return mask


def _layout(rng, mh, mw):
    mask = np.zeros((mh, mw), dtype=bool)
    # A spanning walk, then variable growth, yields paths, branches, cavities.
    r = int(rng.integers(mh))
    c = 0
    mask[r, c] = True
    destinations = [(int(rng.integers(mh)), mw - 1), (0, int(rng.integers(mw))),
                    (mh - 1, int(rng.integers(mw)))]
    for rr, cc in destinations:
        steps = [(int(np.sign(rr - r)), 0)] * abs(rr - r)
        steps += [(0, int(np.sign(cc - c)))] * abs(cc - c)
        rng.shuffle(steps)
        for dr, dc in steps:
            r, c = r + dr, c + dc
            mask[r, c] = True
    goal = min(mh * mw - 1, max(4, int(round(mh * mw * float(rng.uniform(.35, .85))))))
    while mask.sum() < goal:
        frontier = sorted({(r + dr, c + dc) for r, c in np.argwhere(mask) for dr, dc in EDGE
                           if 0 <= r + dr < mh and 0 <= c + dc < mw and not mask[r + dr, c + dc]})
        mask[_pick(rng, frontier)] = True
    if mask.all():
        # The walk can fill a tiny map; remove a non-cut cell at the boundary.
        candidates = []
        for p in np.argwhere(mask):
            q = mask.copy()
            q[tuple(p)] = False
            if q.any(axis=0).all() and q.any(axis=1).all() and len(_components(q)) == 1:
                candidates.append(tuple(map(int, p)))
        mask[_pick(rng, candidates)] = False
    return mask


def _partition(rng, target, count, pin):
    cells = [tuple(map(int, p)) for p in np.argwhere(target)]
    others = [p for p in cells if p != pin]
    seeds = [pin] + [_pick(rng, others)]
    while len(seeds) < count:
        seeds.append(_pick(rng, [p for p in others if p not in seeds]))
    labels = np.full(target.shape, -1, dtype=np.int8)
    for i, p in enumerate(seeds):
        labels[p] = i
    weights = np.exp(rng.uniform(-.9, .9, size=count))
    sizes = [1] * count
    frontier = [set() for _ in seeds]
    for i, (r, c) in enumerate(seeds):
        frontier[i] = {(r + dr, c + dc) for dr, dc in EDGE
                       if 0 <= r + dr < target.shape[0] and 0 <= c + dc < target.shape[1]
                       and target[r + dr, c + dc] and labels[r + dr, c + dc] < 0}
    for _ in range(len(cells) - count):
        eligible = [i for i in range(count) if frontier[i]]
        i = min(eligible, key=lambda j: sizes[j] / weights[j] + rng.random() * .8)
        p = _pick(rng, sorted(frontier[i]))
        labels[p] = i
        sizes[i] += 1
        for f in frontier:
            f.discard(p)
        r, c = p
        frontier[i].update((r + dr, c + dc) for dr, dc in EDGE
                           if 0 <= r + dr < target.shape[0] and 0 <= c + dc < target.shape[1]
                           and target[r + dr, c + dc] and labels[r + dr, c + dc] < 0)
    return labels


def _split_glyphs(labels, glyph, layout):
    h, w = glyph.shape
    return sum(len(set(map(int, labels[r*h:(r+1)*h, c*w:(c+1)*w][glyph]))) > 1
               for r, c in np.argwhere(layout))


def _scatter(rng, objects, bg):
    """Randomized shelf packing of tight boxes with gaps, no output-sized slots."""
    best = None
    for _ in range(35):
        order = list(map(int, rng.permutation(len(objects))))
        shelf_width = int(rng.integers(max(p.shape[1] for p in objects), 31))
        gap = int(rng.integers(1, 3))
        row = col = shelf_h = 0
        placed = []
        maxc = 0
        for i in order:
            p = objects[i]
            h, w = p.shape
            if col and col + w > shelf_width:
                row += shelf_h + gap
                col = shelf_h = 0
            placed.append((i, row, col))
            maxc = max(maxc, col + w)
            shelf_h = max(shelf_h, h)
            col += w + gap
        H = row + shelf_h
        if H <= 28 and maxc <= 28:
            score = H * maxc + float(rng.random()) * 70
            if best is None or score < best[0]:
                best = score, H, maxc, placed
    if best is None:
        return None
    _, H, W, placed = best
    extra_h = int(rng.integers(0, min(5, 30 - H) + 1))
    extra_w = int(rng.integers(0, min(5, 30 - W) + 1))
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    a = np.full((H + extra_h, W + extra_w), bg, dtype=np.int8)
    for i, r, c in placed:
        p = objects[i]
        view = a[top+r:top+r+p.shape[0], left+c:left+c+p.shape[1]]
        view[p >= 0] = p[p >= 0]
    vals, counts = np.unique(a, return_counts=True)
    if int(vals[np.argmax(counts)]) != bg or sum(counts == counts.max()) != 1:
        return None
    return a


def _make_pair(rng, dimensions, map_dimensions, count, palette, anchor_kind, teaching=False,
               previous_anchor=None, registration_witness=False):
    bg, reg, map_color = palette[:3]
    carriers = palette[3:3+count]
    h, w = dimensions
    mh, mw = map_dimensions
    for attempt in range(240):
        glyph = _glyph(rng, h, w)
        if teaching and glyph.all():
            continue
        layout = _layout(rng, mh, mw)
        coords = [tuple(map(int, p)) for p in np.argwhere(layout)]
        if previous_anchor is not None:
            candidates = [p for p in coords if p[0] != previous_anchor[0] and p[1] != previous_anchor[1]]
        elif anchor_kind == 'corner':
            candidates = [p for p in coords if p != (0, 0) and
                          p[0] in (0, mh-1) and p[1] in (0, mw-1)]
        elif anchor_kind == 'inner':
            candidates = [p for p in coords if 0 < p[0] < mh-1 and 0 < p[1] < mw-1]
        else:
            candidates = coords
        if not candidates:
            continue
        anchor = _pick(rng, candidates)
        pin = anchor[0] * h, anchor[1] * w
        target = np.kron(layout, glyph)
        if target.sum() < count * 4 + 1:
            continue
        for _ in range(12):
            labels = _partition(rng, target, count, pin)
            if min(int(np.sum(labels == i)) for i in range(count)) < 3:
                continue
            if _split_glyphs(labels, glyph, layout) < 2:
                continue
            colored = np.full(target.shape, bg, dtype=np.int8)
            for i, color in enumerate(carriers):
                colored[labels == i] = color
            colored[pin] = reg
            pieces = [_tight(colored, [tuple(map(int, p)) for p in np.argwhere(labels == i)])
                      for i in range(count)]
            answers = _pack(pieces, target, reg, pin)
            if len(answers) != 1 or not np.array_equal(np.where(answers[0] < 0, bg, answers[0]), colored):
                continue
            if registration_witness and len(_pack(pieces, target, reg, None)) < 2:
                continue
            oriented = [np.rot90(p, int(rng.integers(4))).copy() for p in pieces]
            # Every demonstration explicitly refutes translation-only assembly.
            if teaching and _pack(oriented, target, reg, pin, rotations=False):
                continue
            map_object = np.full((2*mh-1, 2*mw-1), -1, dtype=np.int8)
            for r, c in coords:
                map_object[2*r, 2*c] = reg if (r, c) == anchor else map_color
            glyph_object = np.where(glyph, reg, -1).astype(np.int8)
            inp = _scatter(rng, oriented + [glyph_object, map_object], bg)
            if inp is None:
                continue
            return {'input': inp, 'output': colored}
    raise RuntimeError('Could not create a unique stencil assembly within the bounded attempts')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .25)
    base_palette = list(map(int, rng.permutation(10)))
    # Different seeds hold out size, aspect, or the combination of two familiar
    # dimensions. None of these choices changes the semantic rule.
    regime = int(rng.integers(3))
    puzzle, used = {'train': [], 'test': []}, set()
    previous_dims = previous_anchor = None
    for i in range(num_train + num_test):
        is_train = i < num_train
        teaching = is_train and i < 2
        if regime == 0:
            options = [(2, 3), (3, 2), (2, 4), (4, 2)] if is_train else [(3, 4), (4, 3)]
            map_dims = _pick(rng, [(2, 3), (3, 2), (3, 3)]) if is_train else _pick(rng, [(3, 4), (4, 3)])
        elif regime == 1:
            options = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
            map_dims = _pick(rng, [(3, 2), (4, 2), (4, 3)]) if is_train else _pick(rng, [(2, 4), (3, 4), (2, 5)])
        else:
            options = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
            map_dims = _pick(rng, [(2, 3), (3, 2), (3, 3)]) if is_train else _pick(rng, [(3, 4), (4, 3), (4, 4)])
        if teaching and i == 0:
            options = [p for p in options if p[0] != p[1]]
        if teaching and i == 1:
            options = [p for p in options if p[0] != previous_dims[0] and p[1] != previous_dims[1]]
        dims = _pick(rng, options)
        # With tiny stencils five triominoes necessarily repeat silhouettes,
        # allowing carrier swaps. Reserve enough area for distinct fragments.
        max_area = (dims[0] * dims[1] - int(teaching)) * (map_dims[0] * map_dims[1] - 1)
        max_count = min(5, (max_area - 1) // 4)
        count = int(rng.integers(3, max(3, max_count) + 1))
        palette = base_palette if stable_palette else list(map(int, rng.permutation(10)))
        kind = 'corner' if teaching and i == 0 else 'inner' if not is_train and min(map_dims) >= 3 else 'any'
        for _ in range(12):
            pair = _make_pair(rng, dims, map_dims, count, palette, kind, teaching,
                              previous_anchor if teaching and i == 1 else None,
                              registration_witness=teaching and i == 0)
            if _key(pair['input']) not in used:
                break
        else:
            raise RuntimeError('Could not construct distinct inputs')
        used.add(_key(pair['input']))
        puzzle['train' if is_train else 'test'].append(pair)
        if teaching and i == 0:
            previous_dims = dims
            previous_anchor = _decode(pair['input'])['anchor']
    return puzzle


def _features(pair):
    f = _decode(pair['input'])
    out = _grid(pair['output'])
    _require(out.shape == f['target'].shape, 'Output shape must follow the expanded stencil')
    h, w = f['glyph'].shape
    labels = out.copy()
    marked_carrier = next(int(v) for p in f['pieces'] if np.any(p == f['reg'])
                          for v in np.unique(p) if v >= 0 and v != f['reg'])
    labels[labels == f['reg']] = marked_carrier
    f['split_glyphs'] = _split_glyphs(labels, f['glyph'], f['layout'])
    tile_incidence = {color: 0 for color in f['carriers']}
    for r, c in np.argwhere(f['layout']):
        for color in set(map(int, labels[r*h:(r+1)*h, c*w:(c+1)*w][f['glyph']])):
            _require(color in tile_incidence, 'Every permitted cell belongs to a visible fragment')
            tile_incidence[color] += 1
    f['spanning_fragments'] = sum(n > 1 for n in tile_incidence.values())
    f['rotation_required'] = not bool(_pack(f['pieces'], f['target'], f['reg'], f['pin'], rotations=False))
    f['shape'] = out.shape
    return f


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Illegal counts')
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Bad pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'Duplicate input')
            seen.add(_key(inp))
            f = _features(pair)
            _require(out.shape == f['target'].shape, 'Output is the tight expanded blueprint')
            _require(np.array_equal(out != f['bg'], f['target']), 'Blueprint blanks must remain blank')
            _require(out[f['pin']] == f['reg'] and np.sum(out == f['reg']) == 1, 'Registration alignment')
            _require(not np.any(out == f['map_color']), 'Construction map is removed')
            for color in f['carriers']:
                _require(np.sum(inp == color) == np.sum(out == color), 'Preserve all carrier cells')
            _require(f['split_glyphs'] >= 2, 'Fragments must split at least two glyph copies')
            _require(f['spanning_fragments'] >= 1, 'At least one fragment must cross glyph boundaries')
            _require(not f['layout'].all(), 'Blueprint must have absent sites')
            _require(np.array_equal(solve(inp), out), 'Incorrect or ambiguous assembly')
            facts[split].append(f)
    train = facts['train']
    f = train[0]
    _require(len(_pack(f['pieces'], f['target'], f['reg'], None)) > 1,
             'First demonstration must need registration to disambiguate packing')
    _require(all(f['rotation_required'] for f in train[:2]), 'First two demonstrations establish rotation')
    _require(all(not f['glyph'].all() for f in train[:2]), 'First two demonstrate exact glyph blanks')
    _require(len({f['glyph'].shape[0] for f in train}) > 1 and
             len({f['glyph'].shape[1] for f in train}) > 1, 'Demonstrate both scale dimensions')
    _require(len({f['anchor'][0] for f in train}) > 1 and
             len({f['anchor'][1] for f in train}) > 1, 'Demonstrate both anchor coordinates')
    return True


def self_test():
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        records.extend(_features(pair) for s in ('train', 'test') for pair in p[s])
    signature = lambda p: tuple((_key(pair['input']), _key(pair['output']))
                                for s in ('train', 'test') for pair in p[s])
    _require(signature(puzzles[0]) == signature(generate_puzzle(0)), 'Seed determinism')
    _require(len({signature(p) for p in puzzles}) == 42, '42 distinct puzzles')
    _require(len({f['input'].shape for f in records}) >= 25, 'Canvas variation')
    _require(len({_key(f['glyph']) for f in records}) >= 12, 'Glyph structure variation')
    _require(len({_key(f['layout']) for f in records}) >= 40, 'Blueprint structure variation')
    _require({3, 4, 5} <= {len(f['pieces']) for f in records}, 'All fragment counts')
    _require(len({f['anchor'] for f in records}) >= 10, 'Anchor variation')
    _require(any(f['glyph'].all() for f in records), 'Solid glyphs coexist with sparse glyphs')
    _require(any(f['shape'][0] != f['shape'][1] for f in records), 'Rectangular assemblies')
    _require(any(len(_components(~np.pad(f['target'], 1))) > 1 for f in records), 'Enclosed negative space')
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(73 + nt + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    # Family-specific ablations. The demonstrations force both the fine glyph
    # and coarse occupancy, and fail if only translations are permitted.
    for p in puzzles:
        f = _decode(p['train'][0]['input'])
        no_map = np.kron(np.ones_like(f['layout']), f['glyph'])
        no_glyph = np.kron(f['layout'], np.ones_like(f['glyph']))
        _require(no_map.sum() > f['target'].sum(), 'Ignoring map holes changes required area')
        _require(no_glyph.sum() > f['target'].sum(), 'Solidifying glyph changes required area')
        _require(not _pack(f['pieces'], f['target'], f['reg'], f['pin'], rotations=False),
                 'Ignoring rigid rotation changes solvability')
        _require(f['pin'] != (0, 0), 'Global upper-left pin shortcut is exposed')
        _require(len(_pack(f['pieces'], f['target'], f['reg'], None)) > 1,
                 'Dropping registration must destroy uniqueness')
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted answer accepted')
    return {'seeds': 42, 'pairs': len(records), 'glyphs': len({_key(f['glyph']) for f in records}),
            'blueprints': len({_key(f['layout']) for f in records}),
            'canvas_shapes': len({f['input'].shape for f in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
