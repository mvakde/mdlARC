"""Compass-relative companions: local frames change both matching and gravity.

The solver reads only its input. No parent generator or stored puzzle is used.
"""
from collections import Counter
import json
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIR8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x)


def _transform(a, code):
    # Reflection first, followed by counterclockwise quarter turns.
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _inverse(a, code):
    a = np.rot90(a, -(code % 4))
    return (np.fliplr(a) if code >= 4 else a).copy()


def _side_map(code):
    a = np.zeros((3, 3), dtype=int)
    a[0, 1], a[1, 2], a[2, 1], a[1, 0] = 1, 2, 3, 4
    b = _transform(a, code)
    destinations = (int(b[0, 1]), int(b[1, 2]), int(b[2, 1]), int(b[1, 0]))
    return tuple(destinations.index(i + 1) for i in range(4))


MAPS = tuple(_side_map(i) for i in range(8))


def _key(a):
    return tuple(a.shape), np.asarray(a, dtype=np.uint8).tobytes()


def _trim(mask):
    rr, cc = np.nonzero(mask)
    assert len(rr), "empty shape"
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _rotkey(mask):
    return min(_key(np.rot90(_trim(mask), k)) for k in range(4))


def _d4key(mask):
    return min(_rotkey(mask), _rotkey(np.fliplr(mask)))


def _components(mask, directions=DIR8):
    seen, result = set(), []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        p = int(r), int(c)
        if p in seen:
            continue
        seen.add(p)
        todo, cells = [p], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in directions:
                q = y + dy, x + dx
                if (0 <= q[0] < h and 0 <= q[1] < w and mask[q]
                        and q not in seen):
                    seen.add(q)
                    todo.append(q)
        result.append(cells)
    return result


def _shape_info(tile, bg):
    mask = tile != bg
    assert len(_components(mask)) == 1, "one eight-connected object per chamber"
    colors = np.unique(tile[mask])
    assert len(colors) == 1, "objects must be monochromatic"
    rr, cc = np.nonzero(mask)
    r, c = int(rr.min()), int(cc.min())
    return dict(mask=_trim(mask), row=r, col=c, color=int(colors[0]))


def _side(a, s):
    return (a[0, 1:-1], a[1:-1, -1], a[-1, 1:-1], a[1:-1, 0])[s]


def _guide(a, bounds, side, color):
    r, c, h, w = bounds
    if side == 0:
        a[r - 1, c:c + w] = color
    elif side == 1:
        a[r:r + h, c + w] = color
    elif side == 2:
        a[r + h, c:c + w] = color
    else:
        a[r:r + h, c - 1] = color


def _decode(input_grid):
    a = np.asarray(input_grid)
    assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
    assert all(1 <= d <= 30 for d in a.shape)
    assert np.all((0 <= a) & (a <= 9))
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1)))
    outer = {}
    for s in range(4):
        line = _side(a, s)
        if len(line) and len(np.unique(line)) == 1 and int(line[0]) != bg:
            outer[s] = int(line[0])
        else:
            assert np.all(line == bg), "unused border sides are blank"
    assert len(outer) == 2 and len(set(outer.values())) == 2
    assert (list(outer)[0] - list(outer)[1]) % 4 in (1, 3)
    markers = set(outer.values())
    bars = []
    H, W = a.shape
    for color in sorted(markers):
        for cells in _components(a == color, DIR4):
            if any(r in (0, H - 1) or c in (0, W - 1) for r, c in cells):
                continue
            rr = sorted({r for r, c in cells})
            cc = sorted({c for r, c in cells})
            assert min(len(rr), len(cc)) == 1 and len(cells) >= 5
            bars.append(dict(r0=rr[0], r1=rr[-1], c0=cc[0], c1=cc[-1],
                             horizontal=len(rr) == 1, color=color, cells=cells))
    cards, used = [], Counter()
    for i, hor in enumerate(bars):
        if not hor['horizontal']:
            continue
        for j, ver in enumerate(bars):
            if ver['horizontal'] or hor['color'] == ver['color']:
                continue
            hs = 0 if ver['r0'] == hor['r0'] + 1 else 2 if ver['r1'] == hor['r0'] - 1 else None
            vs = 3 if ver['c0'] == hor['c0'] - 1 else 1 if ver['c0'] == hor['c1'] + 1 else None
            if hs is None or vs is None:
                continue
            r, c = ver['r0'], hor['c0']
            h, w = ver['r1'] - r + 1, hor['c1'] - c + 1
            assert r >= 2 and c >= 2 and r + h <= H - 2 and c + w <= W - 2
            guides = {hs: hor['color'], vs: ver['color']}
            codes = [k for k, m in enumerate(MAPS)
                     if all(outer.get(m[s]) == color for s, color in guides.items())]
            assert len(codes) == 1
            raw = a[r:r + h, c:c + w].copy()
            assert not set(np.unique(raw)) & markers
            code = codes[0]
            tile = _transform(raw, code)
            obj = _shape_info(tile, bg)
            cards.append(dict(bounds=(r, c, h, w), guides=guides, code=code,
                              raw=raw, tile=tile, obj=obj, key=_rotkey(obj['mask'])))
            used[i] += 1
            used[j] += 1
    assert len(cards) >= 1 and len(used) == len(bars) and all(n == 1 for n in used.values()), "ambiguous guide pairing"
    # Reconstruct all allowed input regions. This also excludes stray ink,
    # crossing chambers, marker intrusions, and hidden outside objects.
    rebuilt = np.full(a.shape, bg, dtype=int)
    for s, color in outer.items():
        _side(rebuilt, s)[:] = color
    reserved = np.zeros(a.shape, dtype=bool)
    for card in cards:
        r, c, h, w = card['bounds']
        assert not reserved[r - 1:r + h + 1, c - 1:c + w + 1].any(), "chambers overlap"
        reserved[r - 1:r + h + 1, c - 1:c + w + 1] = True
        rebuilt[r:r + h, c:c + w] = card['raw']
        for s, color in card['guides'].items():
            _guide(rebuilt, card['bounds'], s, color)
    assert np.array_equal(rebuilt, a), "unexplained input cells"
    return dict(grid=a, background=bg, outer=outer, cards=cards)


def _predict(info, alternative=None):
    """Input reconstruction, with explicit ablations for evidence checks."""
    bg, cards = info['background'], info['cards']
    if alternative in ('swapped_markers', 'inverse_frames'):
        cards = [dict(card) for card in cards]
        for card in cards:
            if alternative == 'inverse_frames':
                code = (-card['code']) % 4 if card['code'] < 4 else card['code']
            else:
                colors = list(info['outer'].values())
                swapped = {s: colors[1 - colors.index(c)] for s, c in card['guides'].items()}
                code = next(k for k, m in enumerate(MAPS)
                            if all(info['outer'].get(m[s]) == c for s, c in swapped.items()))
            card['code'] = code
            card['tile'] = _transform(card['raw'], code)
            card['obj'] = _shape_info(card['tile'], bg)
            card['key'] = _rotkey(card['obj']['mask'])
    keys = []
    for card in cards:
        obj = card['obj']
        if alternative == 'raw_matching':
            key = _rotkey(card['raw'] != bg)
        elif alternative == 'reflection_matching':
            key = _d4key(obj['mask'])
        elif alternative == 'orientation_matching':
            key = _key(obj['mask'])
        elif alternative == 'area_matching':
            key = int(obj['mask'].sum())
        elif alternative == 'color_matching':
            key = obj['color']
        else:
            key = card['key']
        keys.append(key)
    counts = Counter(keys)
    out = np.array(info['grid'], copy=True)
    for card, key in zip(cards, keys):
        obj = card['obj']
        up = counts[key] > 1
        if alternative == 'exactly_two':
            up = counts[key] == 2
        elif alternative == 'all_north':
            up = True
        elif alternative == 'all_south':
            up = False
        code, tile = card['code'], card['tile']
        if alternative == 'global_vertical':
            code, tile = 0, card['raw']
            obj = _shape_info(tile, bg)
        result = np.full(tile.shape, bg, dtype=int)
        rr, cc = np.nonzero(obj['mask'])
        dest = 0 if up else tile.shape[0] - obj['mask'].shape[0]
        result[rr + dest, cc + obj['col']] = obj['color']
        r, c, h, w = card['bounds']
        out[r:r + h, c:c + w] = _inverse(result, code)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Slide objects according to companionship in their calibrated frames."""
    return _predict(_decode(input_grid))


def _random_mask(rng, limit, area=None, diagonal=False):
    n = int(rng.integers(3, min(15, limit * limit) + 1)) if area is None else area
    cells = {(0, 0)}
    directions = DIR8 if diagonal else DIR4
    for _ in range(n - 1):
        choices = set()
        for y, x in sorted(cells):
            for dy, dx in directions:
                p = y + dy, x + dx
                if p in cells:
                    continue
                ys = [q[0] for q in cells] + [p[0]]
                xs = [q[1] for q in cells] + [p[1]]
                if max(ys) - min(ys) < limit and max(xs) - min(xs) < limit:
                    choices.add(p)
        if not choices:
            break
        choices = sorted(choices)
        cells.add(choices[int(rng.integers(len(choices)))])
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    mask = np.zeros((max(r for r, c in cells) - r0 + 1,
                     max(c for r, c in cells) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _sample_mask(rng, limit, forbidden=(), *, chiral=False, area=None, diagonal_only=False):
    for _ in range(600):
        mask = _random_mask(rng, limit, area, diagonal_only or rng.random() < .32)
        if area is not None and int(mask.sum()) != area:
            continue
        if _rotkey(mask) in forbidden:
            continue
        if chiral and (_rotkey(mask) == _rotkey(np.fliplr(mask)) or
                       len({_key(np.rot90(mask, k)) for k in range(4)}) != 4):
            continue
        if diagonal_only and len(_components(mask, DIR4)) == 1:
            continue
        return mask
    raise RuntimeError("bounded shape sampling failed")


def _palette(rng):
    p = [int(x) for x in rng.permutation(10)]
    return p[0], p[1:3], p[3:]


def _layout(rng, minima, limit):
    """Uneven shelves; each chamber has independently sized visible guides."""
    n = len(minima)
    if n <= 4 and limit >= 4:
        row_counts = [2, n - 2]
    else:
        rows = int(rng.integers((n + 2) // 3, min(3, n // 2) + 1)) if n >= 4 else int(rng.integers(1, 3))
        row_counts = [n // rows + (i < n % rows) for i in range(rows)]
    rng.shuffle(row_counts)
    order = [int(i) for i in rng.permutation(n)]
    rows, pos = [], 0
    for count in row_counts:
        ids = order[pos:pos + count]
        pos += count
        rows.append(ids)
    row_heights = [max(minima[i][0] + 2 for i in ids) for ids in rows]
    assert sum(row_heights) + len(rows) - 1 <= 26
    extra_height = 0 if rng.random() < .30 else int(rng.integers(27 - sum(row_heights) - len(rows) + 1))
    for _ in range(extra_height):
        choices = [i for i in range(len(rows)) if row_heights[i] < 14]
        if choices:
            row_heights[int(rng.choice(choices))] += 1
    widths = []
    for ids in rows:
        ww = [minima[i][1] + 2 for i in ids]
        assert sum(ww) + len(ids) - 1 <= 26
        budget = 26 - sum(ww) - len(ids) + 1
        extra_width = 0 if rng.random() < .30 else int(rng.integers(budget + 1))
        for _ in range(extra_width):
            choices = [j for j in range(len(ids)) if ww[j] < 14]
            if choices:
                ww[int(rng.choice(choices))] += 1
        widths.append(ww)
    need_h = sum(row_heights) + len(rows) - 1
    need_w = max(sum(ww) + len(ww) - 1 for ww in widths)
    H, W = int(rng.integers(need_h + 4, 31)), int(rng.integers(need_w + 4, 31))
    row_extra = rng.multinomial(H - 4 - need_h, np.full(len(rows) + 1, 1 / (len(rows) + 1)))
    r = 2 + int(row_extra[0])
    bounds = [None] * n
    for row_index, ids in enumerate(rows):
        ww = widths[row_index]
        extra = rng.multinomial(W - 4 - sum(ww) - len(ww) + 1, np.full(len(ids) + 1, 1 / (len(ids) + 1)))
        c = 2 + int(extra[0])
        for j, i in enumerate(ids):
            h = int(rng.integers(minima[i][0], row_heights[row_index] - 1))
            w = ww[j] - 2
            r_offset = int(rng.integers(row_heights[row_index] - h - 1))
            # Footprint starts at (r,c); actual payload starts one cell in.
            bounds[i] = r + r_offset + 1, c + 1, h, w
            c += ww[j] + 1 + int(extra[j + 1])
        r += row_heights[row_index] + 1 + int(row_extra[row_index + 1])
    return (H, W), bounds


def _make_input(rng, masks, outer_corner, palette, *, teaching=False, limit=3):
    bg, markers, colors = palette
    outer = {outer_corner: markers[0], (outer_corner + 1) % 4: markers[1]}
    n = len(masks)
    inner_corners = [int(x) for x in rng.permutation(4)] if teaching else [int(rng.integers(4)) for _ in masks]
    if teaching:
        # An actual quarter turn teaches the forward frame convention. Pure
        # half turns and reflections cannot distinguish a map from its inverse.
        eligible = [j for j, corner in enumerate(inner_corners)
                    if any({MAPS[k][corner], MAPS[k][(corner + 1) % 4]} == set(outer) for k in (1, 3))]
        j = int(rng.choice(eligible))
        inner_corners[0], inner_corners[j] = inner_corners[j], inner_corners[0]
    codes, guides, raw_shapes = [], [], []
    for i, mask in enumerate(masks):
        corner = inner_corners[i]
        sides = {corner, (corner + 1) % 4}
        choices = [k for k, m in enumerate(MAPS) if {m[s] for s in sides} == set(outer)]
        if teaching and i < 3:
            choices = [k for k in choices if (k >= 4) == (i == 1)]
        code = int(rng.choice(choices))
        codes.append(code)
        guides.append({s: outer[MAPS[code][s]] for s in sorted(sides)})
        raw_shapes.append(_inverse(mask, code))
    minima = [(max(5, m.shape[0] + 2), max(5, m.shape[1] + 2)) for m in raw_shapes]
    shape, bounds = _layout(rng, minima, limit)
    a = np.full(shape, bg, dtype=int)
    for s, color in outer.items():
        _side(a, s)[:] = color
    object_colors = [int(rng.choice(colors[:int(rng.integers(1, 8))])) for _ in masks]
    if teaching:
        object_colors = [colors[0], colors[1], colors[2], colors[0]]
    for i, (r, c, h, w) in enumerate(bounds):
        mask = masks[i]
        tile = _transform(np.full((h, w), bg, dtype=int), codes[i])
        mh, mw = mask.shape
        # Training shapes are strictly interior in both axes, so every change
        # of north/south or axis is visible. Later scenes may start at an edge.
        y = int(rng.integers(1, tile.shape[0] - mh)) if teaching or i == 0 or rng.random() < .8 else int(rng.choice([0, tile.shape[0] - mh]))
        x = int(rng.integers(1, tile.shape[1] - mw)) if teaching or rng.random() < .8 else int(rng.integers(tile.shape[1] - mw + 1))
        rr, cc = np.nonzero(mask)
        tile[rr + y, cc + x] = object_colors[i]
        a[r:r + h, c:c + w] = _inverse(tile, codes[i])
        for s, color in guides[i].items():
            _guide(a, bounds[i], s, color)
    return a, bounds


def _teaching_pair(rng, outer, palette):
    base = _sample_mask(rng, 3, chiral=True, area=int(rng.integers(4, 8)))
    forbidden = {_rotkey(base), _rotkey(np.fliplr(base))}
    ordinary = _sample_mask(rng, 3, forbidden, area=int(base.sum()), diagonal_only=True)
    masks = [base, np.rot90(base), np.fliplr(base), ordinary]
    a, bounds = _make_input(rng, masks, outer, palette, teaching=True, limit=3)
    b = a.copy()
    info = _decode(a)
    third = next(card for card in info['cards'] if card['bounds'] == bounds[2])
    colors = list(third['guides'].values())
    for s, color in third['guides'].items():
        _guide(b, bounds[2], s, colors[1 - colors.index(color)])
    return [dict(input=a, output=solve(a)), dict(input=b, output=solve(b))]


def _random_scene(rng, outer, palette):
    n = int(rng.integers(3, 9))
    limit = 3 if n >= 7 else int(rng.integers(3, 5 if n >= 5 else 6))
    remaining, sizes = n, []
    while remaining:
        size = int(rng.choice([1, 2, 3][:min(3, remaining)],
                              p=np.array([.42, .39, .19][:min(3, remaining)]) / sum([.42, .39, .19][:min(3, remaining)])))
        sizes.append(size)
        remaining -= size
    masks, keys, previous = [], set(), None
    for size in sizes:
        if previous is not None and rng.random() < .25 and _rotkey(np.fliplr(previous)) not in keys:
            base = np.fliplr(previous)
        else:
            base = _sample_mask(rng, limit, keys)
        keys.add(_rotkey(base))
        previous = base
        masks.extend(np.rot90(base, int(rng.integers(4))) for _ in range(size))
    a, _ = _make_input(rng, masks, outer, palette, limit=limit)
    return dict(input=a, output=solve(a))


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    held = int(rng.integers(4))
    train_corners = [s for s in range(4) if s != held]
    outer = int(rng.choice(train_corners))
    palette, stable = _palette(rng), bool(rng.random() < .25)
    puzzle = dict(train=_teaching_pair(rng, outer, palette), test=[])
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for _ in range(count):
            for attempt in range(40):
                p = _random_scene(rng, int(rng.choice(train_corners)) if split == 'train' else held,
                                  palette if stable else _palette(rng))
                key = _key(p['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(p)
                    break
            else:
                raise RuntimeError("distinct input sampling exhausted")
    # Contrast order carries no meaning.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


ALTERNATIVES = ('raw_matching', 'reflection_matching', 'orientation_matching',
                'area_matching', 'color_matching', 'exactly_two', 'all_north',
                'all_south', 'swapped_markers', 'inverse_frames', 'global_vertical')


def _is_contrast(a, b):
    if a['grid'].shape != b['grid'].shape or a['outer'] != b['outer'] or a['background'] != b['background']:
        return False
    aa = {c['bounds']: c for c in a['cards']}
    bb = {c['bounds']: c for c in b['cards']}
    if set(aa) != set(bb):
        return False
    changed = 0
    for bounds, ca in aa.items():
        cb = bb[bounds]
        if _key(ca['raw']) != _key(cb['raw']) or set(ca['guides']) != set(cb['guides']):
            return False
        changed += ca['guides'] != cb['guides']
    profiles = {tuple(sorted(Counter(c['key'] for c in info['cards']).values())) for info in (a, b)}
    return changed == 1 and profiles == {(1, 1, 2), (1, 3)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (a, b):
                assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _decode(a)
            infos[split].append(info)
            assert 3 <= len(info['cards']) <= 8
            assert np.array_equal(_predict(info), b), "incorrect local destination"
            assert np.array_equal(np.sort(a.ravel()), np.sort(b.ravel())), "colors/cells were lost"
            counts = Counter(card['key'] for card in info['cards'])
            for card in info['cards']:
                r, c, h, w = card['bounds']
                obj = card['obj']
                raw_after = b[r:r + h, c:c + w]
                after = _shape_info(_transform(raw_after, card['code']), info['background'])
                assert _key(after['mask']) == _key(obj['mask']) and after['color'] == obj['color']
                assert after['col'] == obj['col'], "perpendicular coordinate changed"
                expected_row = 0 if counts[card['key']] > 1 else card['tile'].shape[0] - obj['mask'].shape[0]
                assert after['row'] == expected_row
            assert not np.array_equal(a, b)
    for alt in ALTERNATIVES:
        assert any(not np.array_equal(_predict(info, alt), _predict(info)) for info in infos['train']), 'unexcluded shortcut: ' + alt
    assert any(_is_contrast(a, b) for i, a in enumerate(infos['train']) for b in infos['train'][i + 1:]), 'missing guide-only companionship contrast'
    assert any(c['code'] in (1, 3) for info in infos['train'] for c in info['cards']), 'missing forward quarter-turn witness'
    # The held-out outer corner is a construction shift, not a solving codebook.
    assert all(set(t['outer']) != set(a['outer']) for t in infos['test'] for a in infos['train'])
    assert any(len(_components(card['obj']['mask'], DIR4)) > 1
               for info in infos['train'] for card in info['cards'])


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit, explicit count contract, mutations, and ablations."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    metrics = dict(seeds=42, pairs=0, shapes=set(), chamber_shapes=set(), codes=set(),
                   object_counts=set(), areas=set(), backgrounds=set(), class_profiles=set())
    split_codes = {'train': set(), 'test': set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                metrics['pairs'] += 1
                metrics['shapes'].add(tuple(pair['input'].shape))
                metrics['backgrounds'].add(info['background'])
                metrics['object_counts'].add(len(info['cards']))
                metrics['class_profiles'].add(tuple(sorted(Counter(c['key'] for c in info['cards']).values())))
                for card in info['cards']:
                    metrics['chamber_shapes'].add(card['bounds'][2:])
                    metrics['codes'].add(card['code'])
                    split_codes[split].add(card['code'])
                    metrics['areas'].add(int(card['obj']['mask'].sum()))
    assert metrics['codes'] == set(range(8))
    assert all(codes == set(range(8)) for codes in split_codes.values())
    assert metrics['object_counts'] == set(range(3, 9))
    assert len(metrics['shapes']) >= 20 and len(metrics['chamber_shapes']) >= 20
    assert len(metrics['class_profiles']) >= 10 and len(metrics['areas']) >= 8
    assert metrics['backgrounds'] == set(range(10))
    # Demonstrate collective dependence: reflect the frame of one member of
    # the teaching pair. An unchanged former companion becomes a singleton,
    # and the untouched mirror singleton gains a partner.
    witnesses = [_decode(p['input']) for p in puzzles[0]['train']]
    witness = next(info for info in witnesses
                   if sorted(Counter(c['key'] for c in info['cards']).values()) == [1, 1, 2])
    counts = Counter(c['key'] for c in witness['cards'])
    selected = next(c for c in witness['cards'] if counts[c['key']] == 2)
    altered = witness['grid'].copy()
    colors = list(selected['guides'].values())
    for side, color in selected['guides'].items():
        _guide(altered, selected['bounds'], side, colors[1 - colors.index(color)])
    before, after = _predict(witness), solve(altered)
    affected = 0
    for card in witness['cards']:
        if card['bounds'] == selected['bounds']:
            continue
        r, c, h, w = card['bounds']
        assert np.array_equal(witness['grid'][r:r + h, c:c + w], altered[r:r + h, c:c + w])
        affected += not np.array_equal(before[r:r + h, c:c + w], after[r:r + h, c:c + w])
    assert affected == 2, 'chambers must interact through companionship'
    metrics['unchanged_chambers_affected_by_one_guide_swap'] = affected
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(1000 + tr * 10 + te, tr, te)
            assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for kind in ('output_cell', 'guide'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = bad['test'][0]
        if kind == 'output_cell':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        else:
            info = _decode(pair['input'])
            card = info['cards'][0]
            side = next(iter(card['guides']))
            _guide(pair['input'], card['bounds'], side, info['background'])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('corrupted puzzle accepted')
    return {k: sorted(v) if isinstance(v, set) else v for k, v in metrics.items()}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
