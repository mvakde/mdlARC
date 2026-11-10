"""Two-key diagonal punch press. Only standard-library modules and NumPy."""
from itertools import combinations
import hashlib
import json

import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'bad grid shape'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'bad colors'
    return a.astype(np.int8, copy=False)


def _transforms(mask):
    return [np.rot90(mask, i) for i in range(4)] + [
        np.rot90(np.fliplr(mask), i) for i in range(4)]


def _span(mask):
    return all((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))


def _stroke_parse(interior, material):
    cells = np.argwhere(interior != material)
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(int(c-slope*r), []).append((int(r), int(c)))
        strokes = []
        for q, points in sorted(lines.items()):
            points.sort()
            rows = [p[0] for p in points]
            colors = {int(interior[p]) for p in points}
            if len(rows) < 2 or len(colors) != 1 or rows != list(range(rows[0], rows[-1]+1)):
                break
            ts = [r+slope*c for r, c in points]
            strokes.append(dict(q=q, lo=min(ts), hi=max(ts), color=colors.pop(),
                                length=len(points)))
        else:
            if len(strokes) >= 2:
                candidates.append((slope, strokes))
    assert len(candidates) == 1, 'diagonal orientation is ambiguous'
    return candidates[0]


def _decode(input_grid):
    """Discover field, sheet, guide intervals and full-span cards from input."""
    grid = _grid(input_grid)
    vals, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, 'ambiguous field'
    bg = int(vals[np.argmax(counts)])
    candidates = []
    for color in vals:
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(grid == color)
        y, x = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0)+1)
        side = bottom-y
        if side != right-x or side < 7:
            continue
        square = grid[y:bottom, x:right]
        if not (np.all(square[0] == color) and np.all(square[-1] == color)
                and np.all(square[:, 0] == color) and np.all(square[:, -1] == color)):
            continue
        inside = square[1:-1, 1:-1]
        if bg in inside or len(set(map(int, np.unique(inside))) - {color}) < 2:
            continue
        candidates.append((y, x, side, color))
    assert len(candidates) == 1, 'need one solid-rim material sheet'
    y, x, side, material = candidates[0]
    n = side-2
    sheet = grid[y:y+side, x:x+side]
    interior = sheet[1:-1, 1:-1]
    slope, strokes = _stroke_parse(interior, material)
    keys = {s['color'] for s in strokes}
    assert len(keys) >= 2 and material not in keys and bg not in keys
    assert all((s['lo']-s['q']) % 2 == 0 and (s['hi']-s['q']) % 2 == 0 for s in strokes)
    assert all(b['q']-a['q'] >= 2 for a, b in zip(strokes, strokes[1:])), 'guides too close'
    outside = np.ones(grid.shape, dtype=bool)
    outside[y:y+side, x:x+side] = False
    assert not np.any((grid == material) & outside)
    masks, boxes = {}, []
    for color in sorted(set(map(int, np.unique(grid[outside]))) - {bg}):
        cells = np.argwhere(outside & (grid == color))
        top, left = map(int, cells.min(axis=0))
        bottom, right = map(int, cells.max(axis=0)+1)
        assert (bottom-top, right-left) == (n, n), 'card does not span its register'
        assert np.all(outside[top:bottom, left:right]), 'card overlaps sheet'
        register = grid[top:bottom, left:right]
        assert np.all((register == bg) | (register == color)), 'mixed card'
        mask = register == color
        assert _span(mask) and 0 < mask.sum() < n*n, 'empty or full card'
        masks[color] = mask.copy()
        boxes.append((top, left, bottom, right))
    assert keys <= set(masks), 'guide has no same-color card'
    all_boxes = [(y, x, y+side, x+side)] + boxes
    for i, a in enumerate(all_boxes):
        for b in all_boxes[i+1:]:
            assert a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1], 'registers need a blank gap'
    return dict(grid=grid, background=bg, material=material, origin=(y, x),
                n=n, sheet=sheet.copy(), interior=interior.copy(), slope=slope,
                strokes=strokes, masks=masks, boxes=boxes)


def _levels(info, pixel_centers=False):
    n, slope = info['n'], info['slope']
    rr, cc = np.indices((n, n))
    q, t = cc-slope*rr, rr+slope*cc
    for level in range(int(t.min()), int(t.max())+1):
        active = [s for s in info['strokes'] if s['lo'] <= level <= s['hi']
                  and (not pixel_centers or (level-s['q']) % 2 == 0)]
        if len(active) >= 2:
            a, b = active[0], active[-1]
            eligible = ((t == level) & (q > a['q']) & (q < b['q'])
                        & (info['interior'] == info['material']))
            yield level, active, eligible


def _holes(info, alternative='', transform=0):
    """The rule, plus named weaker hypotheses for teaching-coverage checks."""
    n, strokes = info['n'], info['strokes']
    masks = {k: _transforms(v)[transform] for k, v in info['masks'].items()}
    if alternative == 'wrong_keys':
        keys = sorted({s['color'] for s in strokes})
        masks.update({key: info['masks'][keys[(i+1) % len(keys)]] for i, key in enumerate(keys)})
    result = np.zeros((n, n), dtype=bool)
    if alternative == 'no_gate':
        keys = sorted({s['color'] for s in strokes})
        for a, b in combinations(keys, 2):
            result |= masks[a] & masks[b]
        return result & (info['interior'] == info['material'])
    if alternative == 'adjacent':
        rr, cc = np.indices((n, n))
        q, t = cc-info['slope']*rr, rr+info['slope']*cc
        for a, b in zip(strokes, strokes[1:]):
            eligible = ((t >= max(a['lo'], b['lo'])) & (t <= min(a['hi'], b['hi']))
                        & (q > a['q']) & (q < b['q']))
            result |= eligible & masks[a['color']] & masks[b['color']]
        return result & (info['interior'] == info['material'])
    for _, active, eligible in _levels(info, alternative == 'pixel_centers'):
        a, b = active[0]['color'], active[-1]['color']
        if alternative == 'fixed_cards':
            a, b = strokes[0]['color'], strokes[-1]['color']
        permissions = masks[a] & masks[b]
        if alternative == 'no_stencils':
            permissions = np.ones_like(result)
        elif alternative == 'union':
            permissions = masks[a] | masks[b]
        elif alternative == 'left_only':
            permissions = masks[a]
        elif alternative == 'right_only':
            permissions = masks[b]
        elif alternative in ('all_active', 'all_cards'):
            keys = ({s['color'] for s in active} if alternative == 'all_active'
                    else {s['color'] for s in strokes})
            permissions = np.logical_and.reduce([masks[k] for k in sorted(keys)])
        elif alternative == 'opposite_polarity':
            permissions = ~permissions
        result |= eligible & permissions
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    out = info['sheet'].copy()
    out[1:-1, 1:-1][_holes(info)] = info['background']
    return out


def _features(info):
    eligible = np.zeros((info['n'], info['n']), dtype=bool)
    skip, third, subpixel, dynamic = [], [], [], []
    pair_ids = set()
    for level, active, mask in _levels(info):
        eligible |= mask
        a, b = active[0], active[-1]
        cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
        if cells:
            pair_ids.add((a['color'], b['color']))
        inactive_inside = [s for s in info['strokes'] if a['q'] < s['q'] < b['q']
                           and not s['lo'] <= level <= s['hi']]
        if inactive_inside:
            skip.extend(cells)
        if len(active) >= 3:
            third.extend(cells)
        if (level-a['q']) % 2 or (level-b['q']) % 2:
            subpixel.extend(cells)
        if a != info['strokes'][0] or b != info['strokes'][-1]:
            dynamic.extend(cells)
    return dict(eligible=eligible, skip=skip, third=third, subpixel=subpixel,
                dynamic=dynamic, pair_ids=pair_ids)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _geometry(rng, n, count, kind):
    positions = [q for q in combinations(range(-(n-2), n-1), count)
                 if all(b-a >= 2 for a, b in zip(q, q[1:]))]
    for _ in range(256):
        qs = _pick(rng, positions)
        full = [[(r, r+q) for r in range(n) if 0 <= r+q < n] for q in qs]
        runs = []
        style = int(rng.integers(4))
        for points in full:
            length = (len(points) if style == 0 or kind == 'skip'
                      else int(rng.integers(2, len(points)+1)))
            start = int(rng.integers(len(points)-length+1))
            runs.append(points[start:start+length])
        if kind == 'skip':
            interior = int(rng.integers(1, count-1))
            lo = max(r+c for r, c in (full[0][0], full[-1][0]))
            hi = min(r+c for r, c in (full[0][-1], full[-1][-1]))
            starts = [i for i in range(len(full[interior])-1)
                      if lo <= sum(full[interior][i]) and sum(full[interior][i+1]) <= hi]
            if not starts:
                continue
            start = _pick(rng, starts)
            runs[interior] = full[interior][start:start+2]
        # Geometry temporarily uses distinct small key literals; later palette
        # assignment and reflections are independent of transverse rank.
        panel = np.full((n, n), 9, dtype=np.int8)
        for i, points in enumerate(runs):
            for p in points:
                panel[p] = i
        info = dict(n=n, slope=1, material=9, interior=panel,
                    strokes=[dict(q=q, lo=sum(p[0]), hi=sum(p[-1]), color=i,
                                  length=len(p)) for i, (q, p) in enumerate(zip(qs, runs))])
        f = _features(info)
        if kind == 'skip' and not (len(f['skip']) >= 2 and f['third'] and f['subpixel']
                                   and f['eligible'].sum() >= 8):
            continue
        if kind == 'switch' and not (len(f['dynamic']) >= 2 and len(f['pair_ids']) >= 2
                                     and f['eligible'].sum() >= 8):
            continue
        return runs
    raise RuntimeError('bounded guide construction exhausted')


def _mask(rng, n):
    family = int(rng.integers(7))
    m = np.zeros((n, n), dtype=bool)
    if family == 0:
        m[0] = m[-1] = True
        if rng.random() < .5:
            m[:, n//2] = True
    elif family == 1:
        np.fill_diagonal(m, True)
        if rng.random() < .6:
            m |= np.fliplr(m)
    elif family == 2:
        m[:] = True
        m.flat[int(rng.integers(n*n))] = False
    elif family == 3:
        m[int(rng.integers(n)), :] = True
        m[:, int(rng.integers(n))] = True
    else:
        m = rng.random((n, n)) < float(rng.uniform(.25, .75))
        if family == 4:
            m |= np.fliplr(m)
    m = _pick(rng, _transforms(m)).copy()
    for line in (m[0], m[-1], m[:, 0], m[:, -1]):
        if not line.any():
            line[int(rng.integers(n))] = True
    if m.all():
        m[int(rng.integers(1, n-1)), int(rng.integers(1, n-1))] = False
    return m


def _bank_options(n, cards):
    side = n+2
    result = []
    for cols in range(1, cards+1):
        rows = (cards+cols-1)//cols
        for gy in (1, 2, 3):
            for gx in (1, 2, 3):
                bh, bw = rows*n+(rows-1)*gy, cols*n+(cols-1)*gx
                for layout in ('side', 'above'):
                    h, w = ((max(side, bh), side+1+bw) if layout == 'side'
                            else (side+1+bh, max(side, bw)))
                    if max(h, w) <= 30:
                        result.append((layout, rows, cols, gy, gx, bh, bw, h, w))
    return result


def _layout(rng, n, cards):
    layout, rows, cols, gy, gx, bh, bw, h, w = _pick(rng, _bank_options(n, cards))
    side = n+2
    if layout == 'side':
        sy, sx = int(rng.integers(h-side+1)), 0
        by, bx = int(rng.integers(h-bh+1)), side+1
    else:
        sy, sx = 0, int(rng.integers(w-side+1))
        by, bx = side+1, int(rng.integers(w-bw+1))
    slots = [(by+r*(n+gy), bx+c*(n+gx)) for r in range(rows) for c in range(cols)]
    slots = [slots[int(i)] for i in rng.permutation(len(slots))[:cards]]
    # Free placement may put cards on several sides of the sheet. The bank
    # above is a guaranteed fitting fallback, not a solving convention.
    if rng.random() < .5:
        for _ in range(12):
            sh, sw = int(rng.integers(max(h, 23), 31)), int(rng.integers(max(w, 23), 31))
            py, px = int(rng.integers(sh-side+1)), int(rng.integers(sw-side+1))
            busy = np.zeros((sh, sw), dtype=bool)
            busy[max(0, py-1):py+side+1, max(0, px-1):px+side+1] = True
            possible = [(r, c) for r in range(sh-n+1) for c in range(sw-n+1)]
            placed = []
            for index in rng.permutation(len(possible)):
                r, c = possible[int(index)]
                if busy[r:r+n, c:c+n].any():
                    continue
                placed.append((r, c))
                busy[max(0, r-1):r+n+1, max(0, c-1):c+n+1] = True
                if len(placed) == cards:
                    return sh, sw, py, px, placed
    return h, w, sy, sx, slots


_TEACH_ALTS = ('no_stencils', 'no_gate', 'union', 'left_only', 'right_only',
               'all_active', 'all_cards', 'adjacent', 'pixel_centers', 'wrong_keys',
               'opposite_polarity')


def _make_pair(rng, n, count, kind):
    runs = _geometry(rng, n, count, kind)
    palette = list(map(int, rng.permutation(10)))
    bg, material = palette[:2]
    # Ordinary examples sometimes reuse a card for two nonadjacent guides.
    assignments = list(range(count))
    if kind == 'free' and count >= 3 and rng.random() < .3:
        assignments[-1] = assignments[0]
    keys = palette[2:2+len(set(assignments))]
    guide_colors = [keys[i] for i in assignments]
    distractors = _pick(rng, [d for d in range(1 if kind == 'skip' else 0, 3) if len(keys)+d <= 8
                             and _bank_options(n, len(keys)+d)])
    card_colors = keys + palette[2+len(keys):2+len(keys)+distractors]
    interior = np.full((n, n), material, dtype=np.int8)
    for color, points in zip(guide_colors, runs):
        for p in points:
            interior[p] = color
    slope, strokes = _stroke_parse(interior, material)
    info = dict(n=n, material=material, interior=interior, slope=slope, strokes=strokes)
    for _ in range(256):
        masks = {color: _mask(rng, n) for color in card_colors}
        if kind == 'free' and len(keys) >= 2 and rng.random() < .15:
            masks[keys[-1]] = masks[keys[0]].copy()
        info['masks'] = masks
        holes = _holes(info)
        if kind == 'skip':
            if holes.sum() < 3 or any(np.array_equal(holes, _holes(info, alt)) for alt in _TEACH_ALTS):
                continue
            if any(np.array_equal(holes, _holes(info, transform=t)) for t in range(1, 8)):
                continue
        elif kind == 'switch':
            if holes.sum() < 2 or np.array_equal(holes, _holes(info, 'fixed_cards')):
                continue
        break
    else:
        # Retry the whole scene rather than introduce a special solving rule.
        return None
    h, w, sy, sx, slots = _layout(rng, n, len(card_colors))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[sy:sy+n+2, sx:sx+n+2] = material
    grid[sy+1:sy+n+1, sx+1:sx+n+1] = interior
    for (r, c), color in zip(slots, rng.permutation(card_colors)):
        grid[r:r+n, c:c+n][masks[int(color)]] = color
    # Reflect the complete scene: card and sheet orientation remain aligned.
    if rng.random() < .5:
        grid = np.fliplr(grid).copy()
    if rng.random() < .5:
        grid = np.flipud(grid).copy()
    pads = []
    for dim in grid.shape:
        extra = int(rng.integers(min(4, 30-dim)+1))
        before = int(rng.integers(extra+1))
        pads.append((before, extra-before))
    grid = np.pad(grid, pads, constant_values=bg)
    # Compact scenes with only two dense cards can otherwise make the sheet
    # color more frequent than the field. Add visible field space, never a
    # hidden role hint, until the stated input convention holds.
    while np.bincount(grid.ravel(), minlength=10)[bg] <= max(
            np.count_nonzero(grid == c) for c in range(10) if c != bg):
        axes = [i for i in range(2) if grid.shape[i] < 30]
        assert axes, 'no room to establish the field'
        axis = min(axes, key=lambda i: grid.shape[i])
        padding = [(0, 0), (0, 0)]
        padding[axis] = (0, 1) if rng.random() < .5 else (1, 0)
        grid = np.pad(grid, padding, constant_values=bg)
    return dict(input=grid, output=solve(grid))


def _key(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = dict(train=[], test=[]), set()
    teaching_slope = int(rng.choice([-1, 1]))
    # The first two worked examples expose every operation. Tests combine
    # greater inset scale and denser guide assemblies with those operations.
    for split, total in (('train', nt), ('test', ne)):
        for index in range(total):
            kind = ('skip' if index == 0 else 'switch') if split == 'train' and index < 2 else 'free'
            for _ in range(64):
                if split == 'test':
                    grow_scale = bool(rng.integers(2))
                    n = 8 if grow_scale else int(rng.integers(6, 9))
                    count = int(rng.integers(2, 6)) if grow_scale else 5
                else:
                    n = int(rng.integers(6, 8)) if kind != 'free' else int(rng.integers(5, 8))
                    count = (int(rng.integers(3, 5)) if kind == 'skip' else 4 if kind == 'switch'
                             else int(rng.integers(2, 5)))
                pair = _make_pair(rng, n, count, kind)
                if pair is not None and split == 'train' and index < 2:
                    desired = teaching_slope if index == 0 else -teaching_slope
                    if _decode(pair['input'])['slope'] != desired:
                        pair = {r: np.fliplr(pair[r]).copy() for r in ('input', 'output')}
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-pair construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, infos = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            info = _decode(inp)
            assert out.shape == info['sheet'].shape
            assert np.all(out[0] == info['material']) and np.all(out[-1] == info['material'])
            assert np.all(out[:, 0] == info['material']) and np.all(out[:, -1] == info['material'])
            markers = info['interior'] != info['material']
            assert np.array_equal(out[1:-1, 1:-1][markers], info['interior'][markers]), 'guide ink damaged'
            assert np.array_equal(out, solve(inp)), 'incorrect cuts'
            assert set(map(int, np.unique(out))) <= {info['background'], info['material']} | {
                s['color'] for s in info['strokes']}, 'distractor leaked into crop'
            infos[split].append(info)
    train_holes = [_holes(i) for i in infos['train']]
    assert any(h.any() for h in train_holes), 'training shows no holes'
    # Check real displayed answers, not the generator's profile names. These
    # requirements also apply to two-demonstration puzzles.
    for alt in _TEACH_ALTS + ('fixed_cards',):
        assert any(not np.array_equal(h, _holes(i, alt)) for i, h in zip(infos['train'], train_holes)), 'missing witness: '+alt
    candidates = {t for t in range(8) if all(np.array_equal(h, _holes(i, transform=t))
                  for i, h in zip(infos['train'], train_holes))}
    assert candidates == {0}, 'training leaves card orientation ambiguous'
    if any(len(i['masks']) > len({s['color'] for s in i['strokes']}) for i in infos['test']):
        assert any(len(i['masks']) > len({s['color'] for s in i['strokes']}) for i in infos['train']), 'untaught distractor cards'


def _fingerprint(puzzle):
    return hashlib.sha256(b''.join(bytes(np.asarray(p[r]).shape) + np.asarray(p[r], dtype=np.uint8).tobytes()
        for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output'))).hexdigest()


def self_test():
    """Bounded semantic fixtures, mutations, count checks and 42-seed audit."""
    # Analytic six-cell scanline fixture. The short middle guide is inactive
    # at t=3; at t=4 the active outer guides lie between pixel centers.
    n, material = 6, 9
    interior = np.full((n, n), material, dtype=np.int8)
    runs = [[(r, r-3) for r in range(3, 6)], [(2, 2), (3, 3)],
            [(r, r+3) for r in range(3)]]
    for color, points in zip((1, 2, 3), runs):
        for p in points:
            interior[p] = color
    slope, strokes = _stroke_parse(interior, material)
    masks = {k: np.zeros((n, n), dtype=bool) for k in (1, 2, 3)}
    masks[1][1, 2] = masks[3][1, 2] = True  # skip inactive middle
    masks[1][1, 3] = masks[3][1, 3] = True  # continuous t=4
    masks[1][2, 1] = True                  # only one permission: no hole
    masks[1][0, 0] = masks[3][0, 0] = True  # no bracket: no hole
    info = dict(n=n, material=material, interior=interior, slope=slope,
                strokes=strokes, masks=masks)
    expected = np.zeros((n, n), dtype=bool)
    expected[1, 2] = expected[1, 3] = True
    assert np.array_equal(_holes(info), expected)
    assert not _holes(info, 'adjacent')[1, 2]
    assert not _holes(info, 'pixel_centers')[1, 3]
    assert _holes(info, 'left_only')[2, 1] and _holes(info, 'no_gate')[0, 0]
    # Full input-only solve, with loose cards that visibly span 6x6 registers.
    for m in masks.values():
        m[0, 0] = m[0, -1] = m[-1, 0] = m[-1, -1] = True
    grid = np.full((20, 15), 8, dtype=np.int8)
    grid[6:14, :8] = material
    grid[7:13, 1:7] = interior
    for top, key in zip((0, 7, 14), (1, 2, 3)):
        grid[top:top+6, 9:15][masks[key]] = key
    expected_sheet = np.full((8, 8), material, dtype=np.int8)
    expected_sheet[1:-1, 1:-1] = interior
    expected_sheet[2, 3] = expected_sheet[2, 4] = 8
    assert np.array_equal(solve(grid), expected_sheet)
    assert np.array_equal(solve(np.fliplr(grid)), np.fliplr(expected_sheet))
    assert np.array_equal(solve(np.flipud(grid)), np.flipud(expected_sheet))
    # A second fixture changes the selected card pair when an outer guide ends.
    strokes[0]['hi'] = 5
    strokes[0]['length'] = 2
    interior[5, 2] = material
    masks[2][2, 4] = masks[3][2, 4] = True
    assert _holes(info)[2, 4] and not _holes(info, 'fixed_cards')[2, 4]
    # t=6, q=-2 lies outside the new bracket [0,3].
    assert not _holes(info)[4, 2]
    shapes, scales, slopes, counts, distractors, structures, fingerprints = (set() for _ in range(7))
    changed_pairs = switches = skips = repeated = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                shapes.add(pair['input'].shape)
                scales.add(info['n']); slopes.add(info['slope']); counts.add(len(info['strokes']))
                keys = {s['color'] for s in info['strokes']}
                distractors.add(len(info['masks'])-len(keys))
                repeated += len(keys) < len(info['strokes'])
                f = _features(info)
                skips += bool(f['skip']); switches += len(f['pair_ids']) > 1
                changed_pairs += bool(_holes(info).any())
                mapping = {}
                norm = [[mapping.setdefault(int(c), len(mapping)) for c in row] for row in pair['input']]
                structures.add(json.dumps(norm, separators=(',', ':')))
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len(shapes) >= 40 and scales == {5, 6, 7, 8} and slopes == {-1, 1}
    assert counts == {2, 3, 4, 5} and distractors == {0, 1, 2}
    assert changed_pairs >= 150 and switches >= 70 and skips >= 42 and repeated >= 5
    # A fitting size outside the sampler's range is still the same family.
    larger = _make_pair(np.random.default_rng(908), 9, 3, 'free')
    assert _decode(larger['input'])['n'] == 9
    assert np.array_equal(solve(larger['input'].tolist()), larger['output'])
    # Keep the geometry and unordered loose cards, but exchange guide keys.
    # Compare holes, since preserving the recolored guide ink is trivial.
    pair = generate_puzzle(73, 2, 1)['train'][0]
    info = _decode(pair['input'])
    changed = pair['input'].copy()
    y, x = info['origin']
    patch = changed[y+1:y+1+info['n'], x+1:x+1+info['n']]
    keys = sorted({s['color'] for s in info['strokes']})
    for index, key in enumerate(keys):
        patch[info['interior'] == key] = keys[(index+1) % len(keys)]
    assert not np.array_equal(_holes(info), _holes(_decode(changed)))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(100+nt*10+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(123, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged rim accepted')
    p = generate_puzzle(124, 2, 1)
    info = _decode(p['train'][0]['input'])
    hole = tuple(map(int, np.argwhere(_holes(info))[0]+1))
    p['train'][0]['output'][hole] = info['material']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing hole accepted')
