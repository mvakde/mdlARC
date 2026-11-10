"""Quota-driven stencil ribbon. Only NumPy and the Python standard library."""
from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 train, 1..4 test, and train > test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "illegal color")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        groups.append(cells)
    return groups


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _crop(mask):
    r0, r1, c0, c1 = _box(np.argwhere(mask))
    return mask[r0:r1, c0:c1].copy()


def _transform(mask, k):
    return np.rot90(mask[:, ::-1] if k >= 4 else mask, k % 4)


def _perimeter(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)
            if r in (r0, r1 - 1) or c in (c0, c1 - 1)}


def _decode(input_grid):
    """Recover background, cards, field, orientation and quotas from input only."""
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])),
             "input must have a uniform background border")
    parts, cards = [], []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        for cells in _components(g == color):
            box = _box(cells)
            p = dict(color=color, cells=cells, box=box)
            parts.append(p)
            r0, r1, c0, c1 = box
            if min(r1-r0, c1-c0) >= 5 and set(cells) == _perimeter(box):
                cards.append(p)
    # A ring-shaped motif may itself have a rectangular perimeter. Its enclosing
    # card, rather than the interior ring, is the semantic frame.
    cards = [p for p in cards if not any(
        q is not p and q['box'][0] < p['box'][0] and p['box'][1] < q['box'][1]
        and q['box'][2] < p['box'][2] and p['box'][3] < q['box'][3]
        for q in cards)]
    _require(2 <= len(cards) <= 5, "expected stencil cards and one swatch card")
    _require(len({p['color'] for p in cards}) == len(cards), "card outlines repeat")
    papers, stencils = [], []
    for p in cards:
        r0, r1, c0, c1 = p['box']
        inside = g[r0+1:r1-1, c0+1:c1-1]
        colors = set(map(int, inside.flat))
        if len(colors) == 1 and bg not in colors:
            p = dict(p, paper=int(inside[0, 0]))
            papers.append(p)
        else:
            _require(bg in colors and len(colors) == 2, "stencil interior must be binary")
            ink = next(iter(colors - {bg}))
            mask = inside == ink
            _require(len(_components(mask)) == 1, "stencil must be connected")
            p = dict(p, ink=ink, mask=mask, tile=_crop(mask))
            stencils.append(p)
    _require(len(papers) == 1 and 1 <= len(stencils) <= 4, "unique swatch required")
    paper_card = papers[0]
    _require(len({p['ink'] for p in stencils}) == 1, "stencil ink must be shared")
    ink = stencils[0]['ink']
    colors = [bg, ink, paper_card['color'], paper_card['paper']] + [p['color'] for p in stencils]
    _require(len(set(colors)) == len(colors), "semantic roles must be distinct within a pair")

    # The swatch is beyond the bank along the ribbon axis and overlaps its first
    # card along the other axis. Transposition makes the two layouts equivalent.
    candidates = []
    for vertical in (False, True):
        cross, along = (2, 0) if vertical else (0, 2)
        ordered = sorted(stencils, key=lambda p: p['box'][cross])
        intervals = [(p['box'][cross], p['box'][cross+1]) for p in ordered]
        bank_overlap = max(p['box'][along] for p in ordered) < min(p['box'][along+1] for p in ordered)
        separate = all(a[1] < b[0] for a, b in zip(intervals, intervals[1:]))
        sw = paper_card['box']
        after = max(p['box'][along+1] for p in ordered) < sw[along]
        overlaps_first = max(sw[cross], intervals[0][0]) < min(sw[cross+1], intervals[0][1])
        if bank_overlap and separate and after and overlaps_first:
            candidates.append((vertical, ordered))
    _require(len(candidates) == 1, "ambiguous bank/ribbon orientation")
    vertical, stencils = candidates[0]

    excluded = np.zeros(g.shape, dtype=bool)
    for p in cards:
        r0, r1, c0, c1 = p['box']
        _require(not np.any(excluded[r0:r1, c0:c1]), "overlapping cards")
        excluded[r0:r1, c0:c1] = True
    external = []
    for color in sorted(set(map(int, g.flat)) - {bg}):
        mask = (g == color) & ~excluded
        _require(len(_components(mask)) == len(_components(mask, True)), "diagonal ambiguity")
        for cells in _components(mask):
            _require(1 <= len(cells) <= 12, "unsupported token size")
            _require(color not in (ink, paper_card['color'], paper_card['paper']), "structural token color")
            external.append(dict(color=color, cells=cells, box=_box(cells)))
    # Moats apply to all separately interpreted items, irrespective of color.
    labels = np.full(g.shape, -1, dtype=int)
    for i, p in enumerate(cards + external):
        if i < len(cards):
            r0, r1, c0, c1 = p['box']
            _require(not np.any(labels[r0:r1, c0:c1] >= 0), "card collision")
            labels[r0:r1, c0:c1] = i
        else:
            for r, c in p['cells']:
                _require(labels[r, c] == -1, "token inside card")
                labels[r, c] = i
    for r, c in np.argwhere(labels >= 0):
        neighbors = labels[max(0, r-1):r+2, max(0, c-1):c+2]
        _require(not np.any((neighbors >= 0) & (neighbors != labels[r, c])), "items touch")
    along = 0 if vertical else 1
    bank_end = max(p['box'][1 if vertical else 3] for p in stencils)
    _require(all(all(point[along] > bank_end for point in p['cells']) for p in external),
             "tokens must be on the swatch side of the bank")
    counts = [sum(p['color'] == s['color'] for p in external) for s in stencils]
    areas = [sum(len(p['cells']) for p in external if p['color'] == s['color']) for s in stencils]
    tiles = [s['tile'].T.copy() if vertical else s['tile'] for s in stencils]
    _require(sum(counts) > 0, "empty ribbon is outside this family")
    length = sum(n * (t.shape[1] + 1) for n, t in zip(counts, tiles)) - 1
    _require(length <= 30, "ribbon exceeds output budget")
    return dict(background=bg, ink=ink, paper=paper_card['paper'], swatch=paper_card,
                vertical=vertical, stencils=stencils, tiles=tiles, counts=counts,
                areas=areas, external=external, colors=[s['color'] for s in stencils])


def _schedule(counts):
    return [i for round_index in range(max(counts))
            for i, n in enumerate(counts) if round_index < n]


def _render(tiles, counts, colors, paper, vertical=False):
    order = _schedule(counts)
    height = max(tiles[i].shape[0] for i in order)
    pieces = []
    for j, i in enumerate(order):
        tile = tiles[i]
        stamp = np.full((height, tile.shape[1]), paper, dtype=np.int8)
        stamp[height-tile.shape[0]:][tile] = colors[i]
        if j:
            pieces.append(np.full((height, 1), paper, dtype=np.int8))
        pieces.append(stamp)
    out = np.concatenate(pieces, axis=1)
    return out.T.copy() if vertical else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _decode(input_grid)
    return _render(d['tiles'], d['counts'], d['colors'], d['paper'], d['vertical'])


def _grow(rng, h, w, area):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        r, c = choices[int(rng.integers(len(choices)))]
        frontier.remove((r, c))
        mask[r, c] = True
        for x, y in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= x < h and 0 <= y < w and not mask[x, y]:
                frontier.add((x, y))
    return mask


def _motif(rng, h, w, asymmetric=False):
    if min(h, w) == 1:
        return np.ones((h, w), dtype=bool)
    for _ in range(20):
        mask = np.ones((h, w), dtype=bool)
        mode = int(rng.integers(5))
        if mode == 0:
            mask[1:-1, 1:-1] = False  # rings, including ordinary solid thin cases
        elif mode == 1:
            mask[:] = False
            r, c = int(rng.integers(h)), int(rng.integers(w))
            mask[r] = True
            mask[:, c] = True
        elif mode != 2 or asymmetric:
            target = int(rng.integers(h+w-1, h*w+1))
            for flat in rng.permutation(h*w):
                if mask.sum() <= target:
                    break
                r, c = divmod(int(flat), w)
                mask[r, c] = False
                if _crop(mask).shape != (h, w) or len(_components(mask)) != 1:
                    mask[r, c] = True
        if not asymmetric or all(not np.array_equal(mask, _transform(mask, k)) for k in range(1, 8)):
            return mask
    # Bounded teaching fallback only, at the requested 2x3 dimensions.
    return np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)


def _object(rng, area=None, compact=False):
    if area is None:
        area = int(rng.choice((1, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12)))
    mode = int(rng.integers(4))
    if mode == 0 and area <= 7 and not compact:
        mask = np.ones((1, area), dtype=bool)
    elif mode == 1 and area == 8 and not compact:
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
    elif mode == 2:
        w = min(4, area)
        mask = np.zeros(((area+w-1)//w, w), dtype=bool)
        mask.flat[:area] = True
    else:
        mask = _crop(_grow(rng, 4, 4, area))
    return _transform(mask, int(rng.integers(8))).copy()


def _draw_card(g, box, color):
    for p in _perimeter(box):
        g[p] = color


def _pack(rng, g, boxes, specs, field_start, greedy=False):
    h, w = g.shape
    occupied = np.zeros(g.shape, dtype=bool)
    occupied[[0, -1]] = True
    occupied[:, :field_start] = True
    occupied[:, -1] = True
    for r0, r1, c0, c1 in boxes:
        occupied[r0-1:r1+1, c0-1:c1+1] = True
    for color, mask in sorted(specs, key=lambda p: -int(p[1].sum())):
        sh, sw = mask.shape
        if sh > h or sw > w:
            return None
        possible = np.ones((h-sh+1, w-sw+1), dtype=bool)
        for dr, dc in np.argwhere(mask):
            possible &= ~occupied[dr:dr+h-sh+1, dc:dc+w-sw+1]
        points = np.argwhere(possible)
        if not len(points):
            return None
        r0, c0 = points[0 if greedy else int(rng.integers(len(points)))]
        for dr, dc in np.argwhere(mask):
            r, c = int(r0+dr), int(c0+dc)
            g[r, c] = color
            occupied[r-1:r+2, c-1:c+2] = True
    return g


def _scene(rng, tiles, quotas, palette, distractors, vertical, forced=None):
    bg, ink, swatch_outline, paper, *colors = palette
    forced = forced or {}
    n = len(tiles)
    heights = [max(5, t.shape[0]+2) for t in tiles]
    budget = 30 - sum(heights) - (n-1) - 2
    _require(budget >= 0, "card bank does not fit")
    gaps = [1] * (n-1)
    for i in rng.permutation(n):
        extra = int(rng.integers(min(2, budget)+1))
        heights[int(i)] += extra
        budget -= extra
    for i in range(n-1):
        extra = int(rng.integers(min(2, budget)+1))
        gaps[i] += extra
        budget -= extra
    top = 1 + int(rng.integers(min(2, budget)+1))
    widths = [max(5, t.shape[1]+2) + int(rng.integers(3)) for t in tiles]
    for i, t in enumerate(tiles):
        if t.all() and (heights[i]-2, widths[i]-2) == t.shape:
            widths[i] += 1
    offsets = [1 + int(rng.integers(3)) for _ in tiles]
    boxes, r = [], top
    for i in range(n):
        boxes.append((r, r+heights[i], offsets[i], offsets[i]+widths[i]))
        r += heights[i] + (gaps[i] if i < n-1 else 0)
    field_start = max(b[3] for b in boxes) + 1
    _require(field_start <= 23, "field width budget")
    swatch_width = int(rng.integers(5, min(7, 29-field_start)+1))
    swatch_box = (top, top+5, field_start, field_start+swatch_width)
    specifications = []
    for i, (color, quota) in enumerate(zip(colors, quotas)):
        for j in range(quota):
            area = forced[i][j] if i in forced else None
            specifications.append((color, _object(rng, area)))
    specifications += [(color, _object(rng)) for color in distractors]
    field_width = int(rng.integers(swatch_width, 30-field_start))
    w = field_start + field_width + 1
    area = sum(int(m.sum()) for c, m in specifications)
    h = min(30, max(r+1, top+6, int(np.ceil((3*area+40)/field_width))+2))

    def blank(height, width):
        g = np.full((height, width), bg, dtype=np.int8)
        for box, color, tile in zip(boxes, colors, tiles):
            _draw_card(g, box, color)
            r0, r1, c0, c1 = box
            ph, pw = r1-r0-2, c1-c0-2
            rr, cc = int(rng.integers(ph-tile.shape[0]+1)), int(rng.integers(pw-tile.shape[1]+1))
            if tile.all() and tile.shape == (ph, pw):
                # A solid stencil still needs visible background in its card.
                raise AssertionError("solid motif fills its entire card")
            g[r0+1+rr:r0+1+rr+tile.shape[0], c0+1+cc:c0+1+cc+tile.shape[1]][tile] = ink
        _draw_card(g, swatch_box, swatch_outline)
        r0, r1, c0, c1 = swatch_box
        g[r0+1:r1-1, c0+1:c1-1] = paper
        return g

    grid = None
    for attempt in range(12):
        if attempt >= 3:
            h, w = 30, 30
        grid = _pack(rng, blank(h, w), boxes+[swatch_box], specifications, field_start)
        if grid is not None:
            break
    if grid is None:
        # Preserve every quota and teaching area; only optional token geometries
        # shrink when free random packing has failed. No grid of slots is used.
        specifications = []
        for i, (color, quota) in enumerate(zip(colors, quotas)):
            for j in range(quota):
                a = forced[i][j] if i in forced else 1
                specifications.append((color, _object(rng, a, compact=True)))
        specifications += [(color, np.ones((1, 1), dtype=bool)) for color in distractors]
        grid = _pack(rng, blank(30, 30), boxes+[swatch_box], specifications, field_start, True)
        _require(grid is not None, "fallback token packing failed")
    out = _render(tiles, quotas, colors, paper, vertical)
    return {'input': grid.T.copy() if vertical else grid, 'output': out}


def _random_recipe(rng, n, max_count=3, tall=False, high_quota=False):
    quotas = list(map(int, rng.integers(0, max_count+1, size=n)))
    if not sum(quotas):
        quotas[int(rng.integers(n))] = 1
    if high_quota:
        quotas[0] = int(rng.integers(4, 7))
        quotas[1:] = list(map(int, rng.integers(0, 3, size=n-1)))
    widths = [1] * n
    budget = 31 - 2*sum(quotas)
    _require(budget >= 0, "too many stamps")
    for raw in rng.permutation(n):
        i = int(raw)
        extra = int(rng.integers(min(6, budget//quotas[i])+1)) if quotas[i] else int(rng.integers(5))
        widths[i] += extra
        budget -= extra*quotas[i]
    heights = list(map(int, rng.integers(1, 4, size=n)))
    if tall:
        i = int(rng.choice([j for j, q in enumerate(quotas) if q]))
        heights[i] = int(rng.integers(4, 7))
    return [_motif(rng, h, w) for h, w in zip(heights, widths)], quotas


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, 'seed')
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    anchor = int(rng.integers(10))
    collision_area = int(rng.integers(4, 9))
    cut = int(rng.integers(1, collision_area))
    base_axis = bool(rng.integers(2))
    shift = int(rng.integers(3))
    first_other, first_bg = None, None
    for index in range(a+b):
        train = index < a
        split = 'train' if train else 'test'
        forced = {}
        if index == 0:
            tiles = [_motif(rng, 2, 3, True), _motif(rng, 1, int(rng.integers(1, 4))),
                     _motif(rng, 3, int(rng.integers(1, 5)))]
            quotas = [2, 1, 0]
            forced = {0: [cut, collision_area-cut], 1: [1]}
        elif index == 1:
            widths = list(map(int, rng.integers(1, 5, size=3)))
            tiles = [_motif(rng, 2, widths[0]), _motif(rng, 3, widths[1]), _motif(rng, 1, widths[2])]
            quotas = [1, 2, 2]
            larger_area = collision_area + int(rng.integers(1, 5))
            split_area = int(rng.integers(1, larger_area))
            forced = {0: [collision_area], 1: [split_area, larger_area-split_area], 2: [1, 1]}
        else:
            mode = shift if not train and (index == a or rng.random() < .7) else -1
            n = 4 if mode == 0 else int(rng.integers(1 if mode == -1 else 2, 4))
            tiles, quotas = _random_recipe(rng, n, tall=mode == 1, high_quota=mode == 2)
        n = len(tiles)
        if index == 0:
            pool = [c for c in range(10) if c != anchor]
            rest = list(map(int, rng.choice(pool, n-1, replace=False)))
            colors = [anchor] + rest
            first_other = colors[1]
        elif index == 1:
            pool = [c for c in range(10) if c not in (anchor, first_other)]
            colors = [anchor] + list(map(int, rng.choice(pool, n-1, replace=False)))
            # Three active colors in a nonmonotonic literal order distinguish
            # geometric bank order from either numeric color sorting direction.
            if colors == sorted(colors) or colors == sorted(colors, reverse=True):
                colors[1], colors[2] = colors[2], colors[1]
        else:
            colors = list(map(int, rng.choice(10, n, replace=False)))
        pool = [c for c in range(10) if c not in colors and not (index == 1 and c == first_other)]
        role_pool = [c for c in pool if not (index == 1 and c == first_bg)]
        bg = int(rng.choice(role_pool))
        pool.remove(bg)
        ink, outline, paper = map(int, rng.choice(pool, 3, replace=False))
        dpool = [c for c in range(10) if c not in colors+[bg, ink, outline, paper]]
        nd = int(rng.integers(0, 4))
        distractors = list(map(int, rng.choice(dpool, nd))) if nd else []
        if index == 0:
            first_bg = bg
            distractors = [int(rng.choice(dpool))] + distractors[:2]
        elif index == 1:
            distractors = [first_other] + distractors[:2]
        vertical = base_axis if index == 0 else not base_axis if index == 1 else bool(rng.integers(2))
        for _ in range(10):
            pair = _scene(rng, tiles, quotas, [bg, ink, outline, paper]+colors, distractors, vertical, forced)
            if _key(pair['input']) not in seen:
                break
        else:
            raise AssertionError("distinct input generation exhausted")
        seen.add(_key(pair['input']))
        puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _expected_cells(d):
    """Validation uses explicit coordinates rather than tile concatenation."""
    order = _schedule(d['counts'])
    h = max(d['tiles'][i].shape[0] for i in order)
    w = sum(d['tiles'][i].shape[1]+1 for i in order)-1
    out = np.full((h, w), d['paper'], dtype=np.int8)
    c = 0
    for i in order:
        t = d['tiles'][i]
        for r, col in np.argwhere(t):
            out[h-t.shape[0]+r, c+col] = d['colors'][i]
        c += t.shape[1]+1
    return out.T.copy() if d['vertical'] else out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _decode(inp)
            _require(np.array_equal(out, _expected_cells(d)), "wrong ribbon cells/order/quotas")
            details[split].append(d)
    tr, te = details['train'], details['test']
    evidence = [(c, q, area) for d in tr for c, q, area in zip(d['colors'], d['counts'], d['areas'])]
    _require(any(c == cc and q != qq and area == aa for c, q, area in evidence
                 for cc, qq, aa in evidence), "area/count collision missing")
    _require(any(q == qq and area != aa for c, q, area in evidence
                 for cc, qq, aa in evidence), "same count at different areas missing")
    _require({d['vertical'] for d in tr} == {False, True}, "both ribbon axes need teaching")
    _require(len({d['background'] for d in tr}) >= 2, "background variation missing")
    category_colors = {c for d in tr for c in d['colors']}
    discard_colors = {p['color'] for d in tr for p in d['external'] if p['color'] not in d['colors']}
    _require(bool(category_colors & discard_colors), "local frame/distractor role change missing")
    _require(any(_schedule(d['counts']) != [i for i, q in enumerate(d['counts']) for _ in range(q)] for d in tr),
             "batching shortcut not distinguished")
    for feature in ('colors', 'areas'):
        _require(any([x for x, q in zip(d[feature], d['counts']) if q] !=
                     sorted(x for x, q in zip(d[feature], d['counts']) if q) for d in tr),
                 f"ascending {feature} order shortcut")
        _require(any([x for x, q in zip(d[feature], d['counts']) if q] !=
                     sorted((x for x, q in zip(d[feature], d['counts']) if q), reverse=True) for d in tr),
                 f"descending {feature} order shortcut")
    _require(any(any(q == 0 and t.shape[0] > max(x.shape[0] for x, n in zip(d['tiles'], d['counts']) if n)
                     for t, q in zip(d['tiles'], d['counts'])) for d in tr), "inactive height witness missing")
    _require(any(len({t.shape[0] for t, q in zip(d['tiles'], d['counts']) if q}) > 1 for d in tr),
             "baseline alignment witness missing")
    for k in range(1, 8):
        _require(any(q and not np.array_equal(t, _transform(t, k)) for d in tr
                     for t, q in zip(d['tiles'], d['counts'])), "orientation witness missing")
    _require(any(q > 1 and not t.all() and s['mask'].shape != s['tile'].shape
                 for d in tr for q, t, s in zip(d['counts'], d['tiles'], d['stencils'])),
             "cropping/recoloring/separator witness missing")
    _require(any(len(d['tiles']) > max(len(x['tiles']) for x in tr)
                 or max(d['counts']) > max(max(x['counts']) for x in tr)
                 or max(t.shape[0] for t, q in zip(d['tiles'], d['counts']) if q) >
                    max(t.shape[0] for x in tr for t, q in zip(x['tiles'], x['counts']) if q)
                 for d in te), "structural train/test shift missing")


def self_test():
    """Bounded 42-seed family checks, plus count and semantic mutation checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_decode(pair['input']) for p in puzzles for split in ('train', 'test') for pair in p[split]]
    _require(len({tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}) == 42,
             "repeated puzzles")
    shapes = {pair['input'].shape for p in puzzles for s in p for pair in p[s]}
    _require(len(shapes) >= 30, "canvas diversity")
    _require({len(d['tiles']) for d in records} == {1, 2, 3, 4}, "category count diversity")
    _require({d['background'] for d in records} == set(range(10)), "background color coverage")
    _require(max(max(d['counts']) for d in records) >= 6, "quota range")
    _require(max(t.shape[0] for d in records for t in d['tiles']) >= 6, "motif height range")
    _require(len({_key(t) for d in records for t in d['tiles']}) >= 80, "motif structure diversity")
    _require(max(len(p['cells']) for d in records for p in d['external']) == 12, "token area range")
    for seed in (0, 7, 41):
        again = generate_puzzle(seed)
        for s in ('train', 'test'):
            for x, y in zip(puzzles[seed][s], again[s]):
                _require(all(np.array_equal(x[k], y[k]) for k in x), "nondeterminism")
                mapping = (np.arange(10) + 3) % 10
                _require(np.array_equal(solve(mapping[x['input']]), mapping[x['output']]),
                         "literal colors changed the rule")
                _require(np.array_equal(solve(x['input'].T), x['output'].T),
                         "transposition changed the rule")
        # Delete one complete matching token: exactly that category loses quota.
        pair = next(x for x in puzzles[seed]['train'] if sum(_decode(x['input'])['counts']) > 1)
        d = _decode(pair['input'])
        token = next(p for p in d['external'] if p['color'] in d['colors'])
        altered = pair['input'].copy()
        for point in token['cells']:
            altered[point] = d['background']
        quotas = d['counts'][:]
        quotas[d['colors'].index(token['color'])] -= 1
        expected = _render(d['tiles'], quotas, d['colors'], d['paper'], d['vertical'])
        _require(np.array_equal(solve(altered), expected) and not np.array_equal(expected, pair['output']),
                 "token removal did not consume exactly one quota")
        # Grow an existing token without touching anything else. Its new pixel
        # is irrelevant to the answer, even though its category's area changes.
        grown = None
        for p in d['external']:
            if len(p['cells']) == 12 or p['color'] not in d['colors']:
                continue
            points = set(p['cells'])
            frontier = sorted({(r+dr, c+dc) for r, c in points
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - points)
            for r, c in frontier:
                if not (1 <= r < altered.shape[0]-1 and 1 <= c < altered.shape[1]-1):
                    continue
                if (r if d['vertical'] else c) <= max(s['box'][1 if d['vertical'] else 3]
                                                       for s in d['stencils']):
                    continue
                if pair['input'][r, c] != d['background']:
                    continue
                neighbors = {(rr, cc) for rr in range(r-1, r+2) for cc in range(c-1, c+2)
                             if pair['input'][rr, cc] != d['background']}
                if neighbors <= points:
                    grown = pair['input'].copy()
                    grown[r, c] = p['color']
                    break
            if grown is not None:
                break
        _require(grown is not None and np.array_equal(solve(grown), pair['output']),
                 "token growth affected quota or no growth witness found")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(73, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Valid members beyond the sampler's ordinary size/count envelope.
    extra_rng = np.random.default_rng(9813)
    for tiles, quotas in (([np.ones((1, 1), dtype=bool)], [15]),
                          ([_motif(extra_rng, 20, 2)], [1]),
                          ([_motif(extra_rng, 3, 14)], [1])):
        for vertical in (False, True):
            pair = _scene(extra_rng, tiles, quotas, [0, 1, 2, 3, 4], [], vertical)
            _require(np.array_equal(solve(pair['input']), pair['output']), "broader family member")
    # Ablations must change actual rendered answers, rather than just metadata.
    for p in puzzles:
        ds = [_decode(x['input']) for x in p['train']]
        _require(any(not np.array_equal(_render(d['tiles'], [1 if q else 0 for q in d['counts']],
                                             d['colors'], d['paper'], d['vertical']), _expected_cells(d)) for d in ds),
                 "omitting per-category quotas did not change answer")
        _require(any(not np.array_equal(_render([np.ones_like(t) for t in d['tiles']], d['counts'],
                                             d['colors'], d['paper'], d['vertical']), _expected_cells(d)) for d in ds),
                 "omitting stencil geometry did not change answer")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'].flat[0] = (int(bad['test'][0]['output'].flat[0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a wrong output")


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
