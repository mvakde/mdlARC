"""Reciprocal edge prints. Standard library + NumPy; no parent imports or state.

The visible packet grammar and transformation are specified in task_spec.md.
Construction paints forward; validation checks pixels by inverse block lookup.
"""
from collections import defaultdict
from itertools import permutations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors")
    return a.astype(int, copy=False)


def _crop(a):
    rr, cc = np.nonzero(a)
    _require(len(rr) > 0, "empty stencil")
    return a[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _shape_key(a):
    return min(_key(np.rot90(a, k)) for k in range(4))


def _connected(a, directions=DIR8):
    cells = set(map(tuple, np.argwhere(a)))
    if not cells:
        return False
    seen = {min(cells)}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for dr, dc in directions:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _components(a, bg):
    """Monochrome four-connected components, including structural frame trees."""
    visited = np.zeros(a.shape, bool)
    h, w = a.shape
    for r in range(h):
        for c in range(w):
            if visited[r, c] or a[r, c] == bg:
                continue
            color = int(a[r, c])
            cells, todo = [], [(r, c)]
            visited[r, c] = True
            while todo:
                y, x = todo.pop()
                cells.append((y, x))
                for dy, dx in DIR4:
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and a[yy, xx] == color):
                        visited[yy, xx] = True
                        todo.append((yy, xx))
            yield color, cells


def _border(a, color):
    return (np.all(a[0] == color) and np.all(a[-1] == color)
            and np.all(a[:, 0] == color) and np.all(a[:, -1] == color))


def _packet_candidates(box, bg, frame):
    """Decode from bounds, not from a sampled layout, scale, palette or split."""
    h, w = box.shape
    if h < 6 or w < 8 or not _border(box, frame):
        return []
    found = []
    for d in range(2, w - 2):
        if not np.all(box[:, d] == frame):
            continue
        fields = (box[1:-1, 1:d], box[1:-1, d + 1:-1])
        starts = (1, d + 1)
        for tile_side in (0, 1):
            tile_field, stencil_field = fields[tile_side], fields[1 - tile_side]
            n = tile_field.shape[1] - 2
            if n not in (2, 3, 4) or tile_field.shape[0] < n + 2:
                continue
            if np.any(stencil_field == frame):
                continue
            occupied = stencil_field != bg
            inks = np.unique(stencil_field[occupied])
            if len(inks) != 1 or not _connected(occupied):
                continue
            rr, cc = np.nonzero(occupied)
            for t in range(tile_field.shape[0] - n - 1):
                square = tile_field[t:t + n + 2]
                if not _border(square, frame):
                    continue
                tile = square[1:-1, 1:-1]
                colors = np.unique(tile)
                if len(colors) != 2 or frame in colors:
                    continue
                expected_field = np.full(tile_field.shape, bg, int)
                expected_field[t:t + n + 2] = frame
                expected_field[t + 1:t + n + 1, 1:-1] = tile
                if not np.array_equal(tile_field, expected_field):
                    continue
                expected = np.full(box.shape, frame, int)
                expected[1:-1, 1:d] = fields[0]
                expected[1:-1, d + 1:-1] = fields[1]
                if not np.array_equal(box, expected):
                    continue
                found.append(dict(mask=_crop(occupied), tile=tile.copy(), ink=int(inks[0]),
                                  local_left=starts[1 - tile_side] + int(cc.min()),
                                  local_top=1 + int(rr.min()), side=tile_side,
                                  stencil_field_shape=stencil_field.shape,
                                  tile_offset=t, divider=d))
    return found


def _decode(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    _require(_border(a, bg), "input needs a uniform exterior background")
    packets = []
    for color, cells in _components(a, bg):
        rr, cc = zip(*cells)
        r, c, b, e = min(rr), min(cc), max(rr) + 1, max(cc) + 1
        found = _packet_candidates(a[r:b, c:e], bg, color)
        if not found:
            continue
        _require(len(found) == 1, "ambiguous packet compartments")
        p = found[0]
        p.update(left=c + p.pop("local_left"), top=r + p.pop("local_top"),
                 box=(r, c, b, e), frame=color)
        packets.append(p)
    _require(len(packets) >= 2, "at least two visible packets required")
    _require(len({p['frame'] for p in packets}) == 1, "one common frame color required")
    cover = np.zeros(a.shape, bool)
    for p in packets:
        r, c, b, e = p['box']
        _require(r > 0 and c > 0 and b < a.shape[0] and e < a.shape[1], "interior packets")
        _require(not cover[max(0, r - 1):b + 1, max(0, c - 1):e + 1].any(),
                 "packet rectangles must be separated")
        cover[r:b, c:e] = True
    _require(np.all(a[~cover] == bg), "stray pixels outside packets")
    groups = _groups(packets)
    _require(all(len(g) <= 2 for g in groups.values()), "a stencil may have only one twin")
    return a.shape, bg, sorted(packets, key=lambda p: p['box'])


def _groups(packets, matching='rotation'):
    groups = defaultdict(list)
    for i, p in enumerate(packets):
        m = p['mask']
        if matching == 'rotation':
            key = _shape_key(m)
        elif matching == 'identity':
            key = _key(m)
        elif matching == 'reflection':
            key = min(_shape_key(m), _shape_key(np.fliplr(m)))
        elif matching == 'area':
            key = int(m.sum())
        elif matching == 'ink':
            key = p['ink']
        elif matching == 'none':
            key = i
        else:
            raise ValueError('unknown matching hypothesis')
        groups[key].append(i)
    return groups


def _donors(packets, matching='rotation'):
    selected, partnered = list(range(len(packets))), [False] * len(packets)
    for group in _groups(packets, matching).values():
        if len(group) > 1:
            for i, j in zip(group, group[1:] + group[:1]):
                selected[i], partnered[i] = j, True
    return selected, partnered


def _render(shape, bg, packets, matching='rotation', behavior='normal'):
    """Forward printing and finite alternative rules for evidence checks."""
    donors, partnered = _donors(packets, matching)
    out, covered = np.full(shape, bg, int), np.zeros(shape, bool)
    for i, p in enumerate(packets):
        donor = packets[i if behavior == 'own_tile' else donors[i]]
        tile, mask = donor['tile'], p['mask']
        if behavior == 'small_stencil':
            tile = np.full((1, 1), p['ink'], int)
        elif behavior == 'align_tile':
            turns = [k for k in range(4) if np.array_equal(np.rot90(donor['mask'], k), mask)]
            if turns:
                tile = np.rot90(tile, turns[0])
        if behavior == 'fill_rectangle':
            mask = np.ones(mask.shape, bool)
        n = len(tile)
        mh, mw = mask.shape[0] * n, mask.shape[1] * n
        top, left = (0 if partnered[i] else shape[0] - mh), p['left']
        _require(top >= 0 and mh <= shape[0] and left + mw <= shape[1], "print does not fit")
        swapped = tile if behavior in ('no_exchange', 'small_stencil') else int(tile.min() + tile.max()) - tile
        for r, c in np.argwhere(mask):
            block = np.s_[top + r * n:top + (r + 1) * n, left + c * n:left + (c + 1) * n]
            _require(not covered[block].any(), "conflicting print blocks")
            if behavior in ('only_low', 'only_high'):
                target = swapped.min() if behavior == 'only_low' else swapped.max()
                view = out[block]
                view[swapped == target] = swapped[swapped == target]
            else:
                out[block] = swapped
            covered[block] = True
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention and payload from input pixels alone."""
    shape, bg, packets = _decode(input_grid)
    return _render(shape, bg, packets)


def _inverse_expected(shape, bg, packets):
    """Validate each cell by locating its source block, without forward painting."""
    donors, partnered = _donors(packets)
    rr, cc = np.indices(shape)
    out, covered = np.full(shape, bg, int), np.zeros(shape, bool)
    for i, p in enumerate(packets):
        tile = packets[donors[i]]['tile']
        n = tile.shape[0]
        h, w = p['mask'].shape
        y = rr - (0 if partnered[i] else shape[0] - n * h)
        x = cc - p['left']
        assert n * h <= shape[0] and p['left'] + n * w <= shape[1], 'expanded print fit'
        inside = (y >= 0) & (x >= 0) & (y < n * h) & (x < n * w)
        sy, sx = np.clip(y // n, 0, h - 1), np.clip(x // n, 0, w - 1)
        active = inside & p['mask'][sy, sx]
        assert not (active & covered).any(), 'overlapping occupied blocks'
        values = int(tile.min() + tile.max()) - tile[y % n, x % n]
        out[active] = values[active]
        covered |= active
    return out, covered


def _random_mask(rng, limit):
    h, w = map(int, rng.integers(1, limit + 1, size=2))
    kind = int(rng.integers(7))
    if kind == 0:
        m = np.ones((1, w), bool)
    elif kind == 1:
        m = np.ones((h, w), bool)
        if min(h, w) >= 3 and rng.random() < .8:
            m[1:-1, 1:-1] = False
    elif kind == 2:
        m = np.zeros((h, w), bool)
        m[:, 0], m[-1, :] = True, True
    elif kind == 3:
        m = np.zeros((h, w), bool)
        m[int(rng.integers(h)), :], m[:, int(rng.integers(w))] = True, True
    else:
        directions = DIR4 if kind == 4 else DIR8
        cells = {(0, 0)}
        target = int(rng.integers(1, limit * limit + 1))
        for _ in range(target - 1):
            options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in directions
                              if (r + dr, c + dc) not in cells
                              and max([r + dr] + [y for y, x in cells]) - min([r + dr] + [y for y, x in cells]) < limit
                              and max([c + dc] + [x for y, x in cells]) - min([c + dc] + [x for y, x in cells]) < limit})
            if not options:
                break
            cells.add(options[int(rng.integers(len(options)))])
        rr, cc = zip(*cells)
        m = np.zeros((max(rr) - min(rr) + 1, max(cc) - min(cc) + 1), bool)
        for r, c in cells:
            m[r - min(rr), c - min(cc)] = True
    return _crop(np.rot90(m, int(rng.integers(4))))


def _sample_mask(rng, forbidden, limit, *, chiral=False, asymmetric=False, area=None, max_area=None):
    def fits(m):
        return (_shape_key(m) not in forbidden
                and (not chiral or _shape_key(m) != _shape_key(np.fliplr(m)))
                and (not asymmetric or len({_key(np.rot90(m, k)) for k in range(4)}) == 4)
                and (area is None or int(m.sum()) == area)
                and (max_area is None or 3 <= m.sum() <= max_area))
    for _ in range(150):
        m = _random_mask(rng, limit)
        if fits(m):
            return m
    for bits in range(1, 512):
        m = _crop(np.array([(bits >> j) & 1 for j in range(9)], bool).reshape(3, 3))
        if max(m.shape) <= limit and _connected(m) and fits(m):
            return m
    raise ValueError('requested stencil vocabulary exhausted')


def _tile(rng, n, colors, witness):
    for _ in range(40):
        bits = rng.random((n, n)) < rng.uniform(.2, .8)
        if bits.any() and not bits.all() and (not witness or not np.array_equal(bits, np.rot90(bits))):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, 0] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _make_specs(rng, scales, profile, limit, compact=False):
    small, large = scales
    if profile == 'mirror':
        pair_count, singleton_count = 1, 2
    elif profile == 'area':
        pair_count, singleton_count = 1, 1
    elif profile == 'test':
        pair_count, singleton_count = 2, int(rng.integers(3))
    else:
        pair_count = int(rng.choice([0, 1, 2], p=[.15, .55, .30]))
        singleton_count = int(rng.integers(1 if pair_count < 2 else 0, 3))
        if not pair_count:
            singleton_count = int(rng.integers(2, 5))
    witness = profile in ('mirror', 'area')
    palette = list(map(int, rng.permutation(10)))
    bg, frame = palette[:2]
    inks = palette[2:2 + int(rng.integers(2, 6))]
    specs, forbidden, bases = [], set(), []
    both_handed = pair_count == 2 and rng.random() < .20 and not compact
    for group in range(pair_count):
        if compact:
            base = np.array([[1, 1], [1, 0]], bool) if group == 0 else np.ones((2, 1), bool)
        elif both_handed and group == 1:
            base = np.fliplr(bases[0]).copy()
        else:
            base = _sample_mask(rng, forbidden, limit, chiral=both_handed and group == 0,
                                asymmetric=witness, max_area=6 if witness else None)
        bases.append(base)
        forbidden.add(_shape_key(base))
        turns = list(map(int, rng.integers(4, size=2)))
        if witness:
            turns[1] = (turns[0] + int(rng.choice([1, 3]))) % 4
        if compact:
            turns = [0, 1] if group == 0 else [0, 0]
        ns = ([small, small] if profile == 'mirror' else [small, large]
              if profile == 'area' or (compact and group == 0) else
              [small, small] if compact else list(map(int, rng.choice(scales, 2))))
        for turn, n in zip(turns, ns):
            specs.append(dict(mask=np.rot90(base, turn).copy(), group=group, n=n, up=True))
    mirror = None
    for j in range(singleton_count):
        if profile == 'mirror' and j == 1:
            m = np.fliplr(mirror).copy()
        elif compact:
            if profile == 'mirror':
                m = np.array([[0, 1, 1], [1, 1, 0]], bool)
            elif profile == 'area':
                m = np.ones((3, 1), bool)
            else:
                m = (np.ones((2, 2), bool), np.ones((3, 1), bool),
                     np.eye(2, dtype=bool), np.ones((1, 1), bool))[j]
        else:
            bans = forbidden | {_shape_key(np.fliplr(b)) for b in bases}
            m = _sample_mask(rng, bans, limit, chiral=profile == 'mirror',
                             area=int(bases[0].sum()) if profile == 'area' else None)
        if profile == 'mirror' and j == 0:
            mirror = m
        forbidden.add(_shape_key(m))
        if not (profile == 'mirror' and j == 0):
            forbidden.add(_shape_key(np.fliplr(m)))
        if not compact:
            m = np.rot90(m, int(rng.integers(4))).copy()
        specs.append(dict(mask=m, group=pair_count + j, n=small if compact or witness else int(rng.choice(scales)), up=False))
    for i, s in enumerate(specs):
        s['ink'] = int(rng.choice(inks))
        if witness:
            s['ink'] = inks[1] if i == 1 else inks[0]
        tile_pool = palette[2:] if witness or rng.random() < .65 else [bg] + palette[2:]
        colors = list(map(int, rng.choice(tile_pool, 2, replace=False)))
        s['tile'] = _tile(rng, s['n'], colors, witness)
        pl, pr, pt, pb = (map(int, rng.integers(2, size=4)) if not compact else (0, 0, 0, 0))
        s['pad'] = pt, pl
        s['field_w'] = s['mask'].shape[1] + pl + pr
        s['inner_h'] = max(s['mask'].shape[0] + pt + pb, s['n'] + 2)
        s['tile_side'] = (0 if rng.random() < .28 else 1) if not compact else 1
        # 0 = tile on the left; 1 = tile on the right.
        s['width'], s['height'] = s['field_w'] + s['n'] + 5, s['inner_h'] + 2
        s['offset'] = 1 + pl if s['tile_side'] == 1 else s['n'] + 4 + pl
        s['tile_row'] = int(rng.integers(s['inner_h'] - s['n'] - 1))
    # Equal-scale twins must expose borrowing, not just coincidentally identical tiles.
    if witness and np.array_equal(specs[0]['tile'], specs[1]['tile']):
        t = specs[1]['tile']
        specs[1]['tile'] = int(t.min() + t.max()) - t
    members = defaultdict(list)
    for i, s in enumerate(specs):
        members[s['group']].append(i)
    for i, s in enumerate(specs):
        group = members[s['group']]
        s['donor'] = i if len(group) == 1 else next(j for j in group if j != i)
        n = specs[s['donor']]['n']
        s['print_h'], s['print_w'] = (int(x) * n for x in s['mask'].shape)
    return bg, frame, specs


def _place_columns(rng, specs, width):
    for up in (True, False):
        indices = [i for i, s in enumerate(specs) if s['up'] == up]
        if not indices:
            continue
        orders = list(permutations(indices))
        rng.shuffle(orders)
        placed = False
        for order in orders:
            cursor, positions = 0, {}
            for i in order:
                s = specs[i]
                lo = 1 + s['offset']
                hi = min(width - s['print_w'], width - s['width'] - 1 + s['offset'])
                x = max(cursor, lo)
                if x > hi:
                    break
                positions[i] = x
                cursor = x + s['print_w'] + int(rng.integers(0, 3))
            else:
                slack = min(min(width - specs[i]['print_w'], width - specs[i]['width'] - 1 + specs[i]['offset']) - x
                            for i, x in positions.items())
                shift = int(rng.integers(slack + 1))
                for i, x in positions.items():
                    specs[i]['left'] = x + shift
                    specs[i]['box_left'] = x + shift - specs[i]['offset']
                placed = True
                break
        if not placed:
            return False
    return True


def _place_rows(rng, specs, height):
    choices = [list(map(int, rng.permutation(np.arange(1, height - s['height'])))) for s in specs]
    if any(not c for c in choices):
        return False
    def x_overlap(a, b):
        return max(a['box_left'], b['box_left']) <= min(a['box_left'] + a['width'], b['box_left'] + b['width'])
    order = sorted(range(len(specs)), key=lambda i: -sum(x_overlap(specs[i], s) for s in specs))
    selected, budget = {}, [3500]
    def search(k):
        if k == len(order):
            return True
        i = order[k]
        for r in choices[i]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if any(x_overlap(specs[i], specs[j]) and
                   max(r, y) <= min(r + specs[i]['height'], y + specs[j]['height'])
                   for j, y in selected.items()):
                continue
            selected[i] = r
            if search(k + 1):
                return True
            del selected[i]
        return False
    if not search(0):
        return False
    for i, r in selected.items():
        specs[i]['box_top'] = r
    return True


def _construct(shape, bg, frame, specs):
    inp, out = np.full(shape, bg, int), np.full(shape, bg, int)
    for s in specs:
        box = np.full((s['height'], s['width']), frame, int)
        n, fw = s['n'], s['field_w']
        d = fw + 1 if s['tile_side'] == 1 else n + 3
        box[1:-1, 1:d], box[1:-1, d + 1:-1] = bg, bg
        stencil_col = 1 if s['tile_side'] == 1 else d + 1
        tile_col = d + 1 if s['tile_side'] == 1 else 1
        pt, pl = s['pad']
        r, c = 1 + pt, stencil_col + pl
        region = box[r:r + s['mask'].shape[0], c:c + s['mask'].shape[1]]
        region[s['mask']] = s['ink']
        t = 1 + s['tile_row']
        box[t:t + n + 2, tile_col:tile_col + n + 2] = frame
        box[t + 1:t + n + 1, tile_col + 1:tile_col + n + 1] = s['tile']
        y, x = s['box_top'], s['box_left']
        inp[y:y + s['height'], x:x + s['width']] = box
        tile = specs[s['donor']]['tile']
        n = len(tile)
        swapped = int(tile.min() + tile.max()) - tile
        top = 0 if s['up'] else shape[0] - s['print_h']
        for r, c in np.argwhere(s['mask']):
            out[top + r * n:top + (r + 1) * n, s['left'] + c * n:s['left'] + (c + 1) * n] = swapped
    return {'input': inp, 'output': out}


def _make_pair(rng, scales, profile, richness):
    for attempt in range(301):
        compact = attempt >= 280
        limit = (3 if profile in ('mirror', 'area') else
                 int(rng.integers(2, richness + 1)) if attempt < 100 else 3)
        try:
            bg, frame, specs = _make_specs(rng, scales, profile, limit, compact)
            min_h = max(s['height'] + 2 for s in specs)
            min_h = max(min_h, sum(max((s['print_h'] for s in specs if s['up'] == edge), default=0)
                                   for edge in (True, False)) + 1, 16)
            if min_h > 30:
                continue
            width = int(rng.integers(18, 31)) if not compact else 30
            if not _place_columns(rng, specs, width):
                width = 30
                if not _place_columns(rng, specs, width):
                    continue
            height = int(rng.integers(min_h, 31)) if not compact else 30
            if not _place_rows(rng, specs, height):
                height = 30
                if not _place_rows(rng, specs, height):
                    continue
            pair = _construct((height, width), bg, frame, specs)
            shape, ground, packets = _decode(pair['input'])
            predicted, _ = _inverse_expected(shape, ground, packets)
            if not np.array_equal(predicted, pair['output']):
                continue
            if profile in ('mirror', 'area') and not _witness_ok(shape, ground, packets, pair['output'], profile):
                continue
            return pair
        except (ValueError, AssertionError):
            continue
    raise RuntimeError('bounded packet packing exhausted')


def _differs(shape, bg, packets, expected, matching='rotation', behavior='normal'):
    try:
        return not np.array_equal(_render(shape, bg, packets, matching, behavior), expected)
    except ValueError:
        return True


def _observations(packets):
    groups = _groups(packets)
    donors, partnered = _donors(packets)
    keys = {_shape_key(p['mask']) for p in packets}
    singles = [p for i, p in enumerate(packets) if not partnered[i]]
    repeated = [i for i in range(len(packets)) if partnered[i]]
    return dict(
        mixed=bool(singles and repeated),
        mirror=any(_shape_key(np.fliplr(p['mask'])) != _shape_key(p['mask'])
                   and _shape_key(np.fliplr(p['mask'])) in keys for p in singles),
        ordinary=any(_shape_key(np.fliplr(p['mask'])) not in keys
                     or _shape_key(np.fliplr(p['mask'])) == _shape_key(p['mask']) for p in singles),
        quarter=any(not np.array_equal(packets[i]['mask'], packets[donors[i]]['mask'])
                    and any(np.array_equal(packets[i]['mask'], np.rot90(packets[donors[i]]['mask'], k)) for k in (1, 3))
                    for i in repeated),
        different_ink=any(packets[i]['ink'] != packets[donors[i]]['ink'] for i in repeated),
        cross_ink=any(p['ink'] == packets[i]['ink'] for p in singles for i in repeated),
        unequal_scale=any(len(packets[i]['tile']) != len(packets[donors[i]]['tile']) for i in repeated),
        same_area_negative=any(p['mask'].sum() == packets[i]['mask'].sum() for p in singles for i in repeated),
        holes=any(not p['mask'].all() for p in packets),
        pair_count=sum(len(g) == 2 for g in groups.values()), singleton_count=len(singles))


def _witness_ok(shape, bg, packets, out, profile):
    f = _observations(packets)
    common = (f['mixed'] and f['quarter'] and f['different_ink'] and f['cross_ink'] and f['holes']
              and all(_differs(shape, bg, packets, out, behavior=b)
                      for b in ('own_tile', 'no_exchange', 'align_tile', 'fill_rectangle', 'only_low', 'only_high'))
              and _differs(shape, bg, packets, out, matching='identity'))
    return common and (f['mirror'] if profile == 'mirror' else
                       f['ordinary'] and f['same_area_negative'] and f['unequal_scale'])


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    scales = tuple(sorted(map(int, rng.choice([2, 3, 4], 2, replace=False))))
    richness = int(rng.choice([3, 4, 5, 6]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            profile = ('mirror' if i == 0 else 'area' if i == 1 else 'free') if split == 'train' else 'test'
            for _ in range(20):
                pair = _make_pair(rng, scales, profile, richness)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct packet inputs exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    shape, bg, packets = _decode(pair['input'])
    features = _observations(packets)
    features.update(shape=shape, background=bg, frame=packets[0]['frame'],
                    count=len(packets), tile_sizes=[len(p['tile']) for p in packets],
                    stencil_shapes=[p['mask'].shape for p in packets],
                    stencil_areas=[int(p['mask'].sum()) for p in packets],
                    stencil_keys=[_shape_key(p['mask']) for p in packets],
                    diagonal=any(not _connected(p['mask'], DIR4) for p in packets),
                    tile_sides=[p['side'] for p in packets],
                    ground_tile=any(bg in p['tile'] for p in packets),
                    tile_ink_sharing=any(p['ink'] in p['tile'] for p in packets),
                    density=float(np.mean(np.asarray(pair['output']) != bg)))
    return features


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, observations, decoded_train = set(), [], []
    train_sizes, test_sizes = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            shape, bg, packets = _decode(inp)
            expected, covered = _inverse_expected(shape, bg, packets)
            assert np.array_equal(expected, out), 'wrong material, scale, orientation, edge or background'
            assert covered.any() and np.any(out != bg), 'empty print'
            assert np.all(out[~covered] == bg), 'transparent positions changed'
            assert np.array_equal(solve(inp.copy()), out), 'input-only solve disagrees'
            f = _observations(packets)
            sizes = {len(p['tile']) for p in packets}
            if split == 'train':
                train_sizes.update(sizes)
                observations.append(f)
                decoded_train.append((shape, bg, packets, out))
            else:
                test_sizes.update(sizes)
    assert test_sizes <= train_sizes, 'test scales lack training evidence'
    assert sum(f['mixed'] for f in observations) >= 2
    for name in ('mirror', 'ordinary', 'quarter', 'different_ink', 'cross_ink', 'unequal_scale', 'same_area_negative', 'holes'):
        assert any(f[name] for f in observations), f'missing training witness: {name}'
    hypotheses = [('identity', 'normal'), ('reflection', 'normal'), ('area', 'normal'),
                  ('ink', 'normal'), ('none', 'normal')]
    hypotheses += [('rotation', b) for b in ('own_tile', 'no_exchange', 'align_tile',
                                            'fill_rectangle', 'small_stencil', 'only_low', 'only_high')]
    for matching, behavior in hypotheses:
        assert any(_differs(*d, matching=matching, behavior=behavior) for d in decoded_train), (matching, behavior)
    # Main ablations must give an observable pixel disagreement, not only a fit failure.
    for behavior in ('own_tile', 'no_exchange'):
        valid_disagreement = False
        for shape, bg, packets, out in decoded_train:
            try:
                valid_disagreement |= not np.array_equal(_render(shape, bg, packets, behavior=behavior), out)
            except ValueError:
                pass
        assert valid_disagreement, f'no visible {behavior} ablation'


def _puzzle_key(p):
    return tuple((_key(pair['input']), _key(pair['output'])) for s in ('train', 'test') for pair in p[s])


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    features = [pair_features(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert len({f['shape'] for f in features}) >= 25
    assert {n for f in features for n in f['tile_sizes']} == {2, 3, 4}
    assert len({m for f in features for m in f['stencil_keys']}) >= 60
    assert {side for f in features for side in f['tile_sides']} == {0, 1}
    assert len({f['count'] for f in features}) >= 4
    assert {f['background'] for f in features} == set(range(10))
    assert {f['frame'] for f in features} == set(range(10))
    assert any(f['diagonal'] for f in features)
    assert any(f['ground_tile'] for f in features)
    assert any(f['tile_ink_sharing'] for f in features)
    assert max(max(s) for f in features for s in f['stencil_shapes']) >= 5
    assert any(f['pair_count'] == 0 for f in features)
    assert any(f['singleton_count'] == 0 for f in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    corrupted = {s: [{r: a.copy() for r, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('wrong label accepted')
    return dict(seeds=42, pairs=len(features), distinct_shapes=len({f['shape'] for f in features}),
                stencil_classes=len({m for f in features for m in f['stencil_keys']}),
                packet_counts=sorted({f['count'] for f in features}),
                tile_sizes=sorted({n for f in features for n in f['tile_sizes']}))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
