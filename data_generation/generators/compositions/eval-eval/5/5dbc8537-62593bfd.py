"""Symbol-governed contour docking of rectangular cards.

Only the input parser and exact-cover search are used by solve. Construction
metadata never enters prediction; generation paints a label from placements
and separately checks it against that input-only search.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if (y, x) != (0, 0))
HOST_SEEDS = (0, 23193928, 833051721, 410837178, 858574025, 345259549,
              352355026, 73772497, 606457736, 892461541, 829175772,
              241075666, 959221209, 514937493, 7032723, 342705028,
              399999707, 85898374, 530467120, 703883086, 89506603,
              853278757, 495924303, 472505724, 646224482, 569583388,
              195245690, 566767902, 706028446, 441852234, 194528497,
              296235830, 910499361, 171260969, 608944948, 639084392,
              968789765, 837481748, 289240114, 819939597, 464361397,
              483409905)


def _key(a):
    a = np.asarray(a)
    return tuple(a.shape), tuple(int(c) for c in a.flat)


def _normal(mask):
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError('Empty glyph')
    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()


def _canonical(mask, reflection=False):
    masks = [mask, np.fliplr(mask)] if reflection else [mask]
    return min(_key(_normal(np.rot90(m, k))) for m in masks for k in range(4))


def _components(mask, directions=DIR4):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        cells, stack = [], [(y, x)]
        while stack:
            p, q = stack.pop()
            cells.append((p, q))
            for dy, dx in directions:
                r, s = p+dy, q+dx
                if (0 <= r < mask.shape[0] and 0 <= s < mask.shape[1]
                        and mask[r, s] and not seen[r, s]):
                    seen[r, s] = True
                    stack.append((r, s))
        result.append(cells)
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('Expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=True)


def _cards(tray, vacancy):
    cards = []
    for cells in _components(tray != vacancy):
        ys, xs = zip(*cells)
        y, x = min(ys), min(xs)
        h, w = max(ys)-y+1, max(xs)-x+1
        tile = tray[y:y+h, x:x+w].copy()
        if min(h, w) < 3 or len(cells) != h*w or np.any(tile == vacancy):
            raise ValueError('Every source card must be a filled rectangle with a rim')
        body = int(tile[0, 0])
        if not (np.all(tile[0] == body) and np.all(tile[-1] == body)
                and np.all(tile[:, 0] == body) and np.all(tile[:, -1] == body)):
            raise ValueError('A card needs a uniform one-cell rim')
        glyph = tile != body
        if len(np.unique(tile)) != 2 or len(_components(glyph, DIR8)) != 1:
            raise ValueError('A card needs one monochromatic eight-connected glyph')
        mask = _normal(glyph)
        cards.append(dict(tile=tile, body=body, glyph=mask,
                          ink=int(tile[glyph][0]), canonical=_canonical(mask),
                          source=(y, x)))
    if not cards:
        raise ValueError('Empty source tray')
    frequencies = Counter(c['canonical'] for c in cards)
    for c in cards:
        c['up'] = frequencies[c['canonical']] >= 2
    return cards


def _parse(input_grid):
    """The unique divider is one entire row/column and occurs nowhere else."""
    grid = _grid(input_grid)
    dividers = []
    for color in np.unique(grid):
        ys, xs = np.nonzero(grid == color)
        if len(set(xs)) == 1 and len(ys) == grid.shape[0] and 0 < xs[0] < grid.shape[1]-1:
            dividers.append((False, int(xs[0]), int(color)))
        if len(set(ys)) == 1 and len(xs) == grid.shape[1] and 0 < ys[0] < grid.shape[0]-1:
            dividers.append((True, int(ys[0]), int(color)))
    if len(dividers) != 1:
        raise ValueError('Expected one uniquely colored full-span divider')
    transposed, cut, divider = dividers[0]
    canonical = grid.T if transposed else grid
    a, b = canonical[:, :cut], canonical[:, cut+1:]
    candidates = []
    for stencil, tray, board_first in ((a, b, True), (b, a, False)):
        colors = np.unique(stencil)
        if len(colors) != 2:
            continue
        for board, vacancy in (colors, colors[::-1]):
            if np.any(tray == board) or not np.any(tray == vacancy):
                continue
            try:
                cards = _cards(tray, int(vacancy))
            except ValueError:
                continue
            mask = stencil == vacancy
            if sum(c['tile'].size for c in cards) != int(mask.sum()):
                continue
            candidates.append(dict(stencil=stencil.copy(), tray=tray.copy(),
                                   mask=mask, cards=cards, board=int(board),
                                   vacancy=int(vacancy), divider=divider,
                                   transposed=transposed, board_first=board_first))
    if len(candidates) != 1:
        raise ValueError('The input must identify exactly one stencil and tray')
    return candidates[0]


def _docked(mask, y, x, h, w, up):
    """At least one cell beyond the relevant edge is not an original vacancy."""
    r = y-1 if up else y+h
    return r < 0 or r >= mask.shape[0] or not np.all(mask[r, x:x+w])


def _packing(mask, cards, use_companionship=True, max_nodes=50000, forced=None):
    """Enumerate rendered solutions, grouping only identical physical cards.

    A returned second image proves ambiguity. Exhausted budgets are rejected,
    never treated as uniqueness. All supplied cards must be used once.
    """
    coordinates = [tuple(map(int, p)) for p in np.argwhere(mask)]
    indices = np.full(mask.shape, -1, dtype=int)
    for i, p in enumerate(coordinates):
        indices[p] = i
    groups, lookup = [], {}
    for card in cards:
        key = _key(card['tile']), card['up'] if use_companionship else None
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append(dict(tile=card['tile'], up=card['up'], count=0))
        groups[lookup[key]]['count'] += 1
    placements, by_cell = [], [[] for _ in coordinates]
    for g, group in enumerate(groups):
        tile = group['tile']
        h, w = tile.shape
        for y in range(mask.shape[0]-h+1):
            for x in range(mask.shape[1]-w+1):
                if not np.all(mask[y:y+h, x:x+w]):
                    continue
                if use_companionship and not _docked(mask, y, x, h, w, group['up']):
                    continue
                cells = tuple(int(i) for i in indices[y:y+h, x:x+w].flat)
                bits = sum(1 << i for i in cells)
                p = len(placements)
                placements.append((g, bits, cells, y, x))
                for i in cells:
                    by_cell[i].append(p)
    full = (1 << len(coordinates))-1
    remaining = [g['count'] for g in groups]
    rendered = [-1] * len(coordinates)
    outcomes, path, dead = {}, [], set()
    nodes, failures, exhaustive = 0, 0, True

    def visit(covered):
        nonlocal nodes, failures, exhaustive
        if not exhaustive or len(outcomes) >= 2:
            return
        nodes += 1
        if nodes > max_nodes:
            exhaustive = False
            return
        if covered == full:
            if any(remaining):
                return
            outcomes.setdefault(tuple(rendered), tuple(path))
            return
        state = covered, tuple(remaining)
        if state in dead:
            return
        best = None
        for i in range(len(coordinates)):
            if covered & (1 << i):
                continue
            options = [p for p in by_cell[i]
                       if remaining[placements[p][0]] and not covered & placements[p][1]]
            if not options:
                dead.add(state)
                failures += 1
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found_before = len(outcomes)
        for p in best:
            g, bits, cells, y, x = placements[p]
            remaining[g] -= 1
            for i, color in zip(cells, groups[g]['tile'].flat):
                rendered[i] = int(color)
            path.append(p)
            visit(covered | bits)
            path.pop()
            for i in cells:
                rendered[i] = -1
            remaining[g] += 1
            if not exhaustive or len(outcomes) >= 2:
                return
        # Do not cache a state with completed renders: its incoming colors matter.
        if len(outcomes) == found_before == 0:
            dead.add(state)

    if sum(g['tile'].size*g['count'] for g in groups) == len(coordinates):
        covered = 0
        if forced is not None:
            g, covered, cells, y, x = placements[forced]
            remaining[g] -= 1
            path.append(forced)
            for i, color in zip(cells, groups[g]['tile'].flat):
                rendered[i] = int(color)
        visit(covered)
    return dict(outcomes=outcomes, groups=groups, placements=placements,
                coordinates=coordinates, nodes=nodes, failures=failures,
                exhaustive=exhaustive)


def _dead_local_choice(mask, cards, result):
    """Find a docked whole-card placement that no complete cover can extend."""
    chosen = set(next(iter(result['outcomes'].values())))
    for p in range(len(result['placements'])):
        if p in chosen:
            continue
        restricted = _packing(mask, cards, max_nodes=5000, forced=p)
        if restricted['exhaustive'] and not restricted['outcomes']:
            return p
    return None


def _analyze(input_grid):
    a = _parse(input_grid)
    result = _packing(a['mask'], a['cards'])
    if not result['exhaustive'] or len(result['outcomes']) != 1:
        raise ValueError('Expected one exhaustively verified rendered packing')
    rendering, path = next(iter(result['outcomes'].items()))
    output = a['stencil'].copy()
    output[a['mask']] = rendering
    a.update(result=result, output=output, path=path)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return (a['output'].T if a['transposed'] else a['output']).copy()


def _random_glyph(rng, limit=3):
    """Growth samples eight-connected masks, including diagonal bridges."""
    target = int(rng.integers(3, min(limit*limit, 10)+1))
    cells = {(0, 0)}
    for _ in range(target-1):
        choices = set()
        for y, x in sorted(cells):
            for dy, dx in DIR8:
                p = y+dy, x+dx
                if p in cells:
                    continue
                trial = cells | {p}
                yy, xx = zip(*trial)
                if max(yy)-min(yy) < limit and max(xx)-min(xx) < limit:
                    choices.add(p)
        if not choices:
            break
        ordered = sorted(choices)
        cells.add(ordered[int(rng.integers(len(ordered)))])
    yy, xx = zip(*cells)
    mask = np.zeros((max(yy)-min(yy)+1, max(xx)-min(xx)+1), dtype=bool)
    for y, x in cells:
        mask[y-min(yy), x-min(xx)] = True
    return mask


def _shape(rng, forbidden, chiral=False, diagonal=False, limit=3):
    for _ in range(150):
        m = _random_glyph(rng, limit)
        if (_canonical(m) not in forbidden
                and (not chiral or _canonical(m) != _canonical(np.fliplr(m)))
                and (not diagonal or len(_components(m)) > 1)):
            return m
    # Bounded vocabulary fallback still satisfies the same visible predicates.
    for bits in range(1, 512):
        m = _normal(np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3))
        if (len(_components(m, DIR8)) == 1 and _canonical(m) not in forbidden
                and (not chiral or _canonical(m) != _canonical(np.fliplr(m)))
                and (not diagonal or len(_components(m)) > 1)):
            return m
    raise RuntimeError('Glyph vocabulary exhausted')


def _plan(rng, profile):
    """Four to seven cards; first up and first down share their dimensions."""
    primary_count = 2 if profile == 'pair' else 3 if profile == 'triplet' else int(rng.integers(2, 4))
    second_count = 2 if profile in ('triplet', 'combined') else 0
    singles = 2 if profile == 'pair' else int(rng.integers(1, 3))
    # Teaching scenes require a chiral class. Other scenes may use ordinary or
    # symmetric classes; reflection negatives are not a compulsory scene form.
    mirror_single = profile == 'pair' or (profile == 'combined' and rng.random() < .40)
    both_handed = profile == 'triplet' or (profile == 'combined' and not mirror_single and rng.random() < .30)
    base = _shape(rng, set(), chiral=mirror_single or both_handed,
                  diagonal=profile == 'triplet', limit=3)
    forbidden = {_canonical(base), _canonical(np.fliplr(base))}
    masks = []
    first_turn = int(rng.integers(4))
    for k in range(primary_count):
        turn = first_turn if k == 0 else (first_turn+1) % 4 if k == 1 else int(rng.integers(4))
        masks.append(_normal(np.rot90(base, turn)))
    if second_count:
        other = np.fliplr(base).copy() if both_handed else _shape(rng, forbidden, limit=int(rng.choice([3, 4])))
        forbidden.update((_canonical(other), _canonical(np.fliplr(other))))
        masks.extend(_normal(np.rot90(other, int(rng.integers(4)))) for _ in range(second_count))
    down_index = len(masks)
    # The first teaching scene includes a mirror-only singleton; the second
    # includes both handednesses as repeated classes and an ordinary singleton.
    single = np.fliplr(base).copy() if mirror_single else _shape(rng, forbidden)
    forbidden.update((_canonical(single), _canonical(np.fliplr(single))))
    masks.append(_normal(np.rot90(single, int(rng.integers(4)))))
    for _ in range(singles-1):
        m = _shape(rng, forbidden)
        forbidden.update((_canonical(m), _canonical(np.fliplr(m))))
        masks.append(_normal(np.rot90(m, int(rng.integers(4)))))
    up = [i < down_index for i in range(len(masks))]
    dims = [None]*len(masks)
    h = max(masks[0].shape[0], masks[down_index].shape[0])+2+int(rng.integers(0, 2))
    w = max(masks[0].shape[1], masks[down_index].shape[1])+2+int(rng.integers(0, 2))
    dims[0] = dims[down_index] = h, w
    used = {(h, w, True), (h, w, False)}
    duplicate_of = {}
    for i, m in enumerate(masks):
        if dims[i] is not None:
            continue
        identical = [j for j in range(i) if up[i] and up[j]
                     and dims[j] is not None and _key(masks[j]) == _key(m)]
        if i >= 2 and identical and rng.random() < .45:
            j = int(rng.choice(identical))
            dims[i] = dims[j]
            duplicate_of[i] = j
            continue
        choices = [(hh, ww) for hh in range(m.shape[0]+2, min(7, m.shape[0]+4)+1)
                   for ww in range(m.shape[1]+2, min(7, m.shape[1]+4)+1)
                   if (hh, ww, up[i]) not in used]
        if not choices:
            return None
        # Small carriers leave room for irregular silhouettes and separated trays.
        weights = np.array([1/(hh*ww)**2 for hh, ww in choices])
        dims[i] = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        used.add((*dims[i], up[i]))
    return masks, up, dims, down_index, duplicate_of


def _rect_mask(rects):
    y0 = min(y for y, x, h, w in rects)
    x0 = min(x for y, x, h, w in rects)
    y1 = max(y+h for y, x, h, w in rects)
    x1 = max(x+w for y, x, h, w in rects)
    mask = np.zeros((y1-y0, x1-x0), dtype=bool)
    for y, x, h, w in rects:
        mask[y-y0:y-y0+h, x-x0:x-x0+w] = True
    return mask, [(y-y0, x-x0, h, w) for y, x, h, w in rects]


def _grow_layout(rng, dims, up, down_index, islands=False):
    h, w = dims[0]
    placed = {0: (0, 0, h, w), down_index: (h, 0, h, w)}
    order = [int(i) for i in rng.permutation([i for i in range(len(dims)) if i not in placed])]
    for i in order:
        hh, ww = dims[i]
        candidates = []
        for _ in range(130):
            anchor = list(placed.values())[int(rng.integers(len(placed)))]
            y, x, ah, aw = anchor
            side = int(rng.integers(4))
            if side < 2:
                y = y-hh if side == 0 else y+ah
                x += int(rng.integers(1-ww, aw))
            else:
                x = x-ww if side == 2 else x+aw
                y += int(rng.integers(1-hh, ah))
            if islands and i == order[-1] and rng.random() < .65:
                x += (-2 if side == 2 else 2) if side >= 2 else 0
                y += (-2 if side == 0 else 2) if side < 2 else 0
            if any(max(y, ry) < min(y+hh, ry+rh) and max(x, rx) < min(x+ww, rx+rw)
                   for ry, rx, rh, rw in placed.values()):
                continue
            trial = {**placed, i: (y, x, hh, ww)}
            keys = list(trial)
            mask, rr = _rect_mask(list(trial.values()))
            if mask.shape[0] > 27 or mask.shape[1] > 15:
                continue
            if not all(_docked(mask, *r, up[j]) for j, r in zip(keys, rr)):
                continue
            # The stacked equal-size pair must have only its intended facing.
            normalized = dict(zip(keys, rr))
            if _docked(mask, *normalized[0], False) or _docked(mask, *normalized[down_index], True):
                continue
            candidates.append((y, x, hh, ww))
        if not candidates:
            return None
        placed[i] = candidates[int(rng.integers(len(candidates)))]
    mask, rects = _rect_mask([placed[i] for i in range(len(dims))])
    return mask, rects


def _scatter(rng, tiles, height, max_width, vacancy):
    min_width = max(t.shape[1] for t in tiles)
    if max_width < min_width:
        return None
    # Independent locations; a one-cell moat prevents accidental card merging.
    for _ in range(12):
        width = int(rng.integers(min_width, max_width+1))
        occupied = np.zeros((height, width), dtype=bool)
        tray = np.full((height, width), vacancy, dtype=np.int8)
        order = sorted(range(len(tiles)), key=lambda i: (-tiles[i].size, float(rng.random())))
        for i in order:
            h, w = tiles[i].shape
            choices = [(y, x) for y in range(height-h+1) for x in range(width-w+1)
                       if not np.any(occupied[max(0,y-1):min(height,y+h+1),
                                              max(0,x-1):min(width,x+w+1)])]
            if not choices:
                break
            y, x = choices[int(rng.integers(len(choices)))]
            tray[y:y+h, x:x+w] = tiles[i]
            occupied[y:y+h, x:x+w] = True
        else:
            return tray
    return None


def _make_pair(rng, profile, transposed, board_first, require_partial=False):
    for attempt in range(220):
        plan = _plan(rng, profile)
        if plan is None:
            continue
        glyphs, up, dims, down_index, duplicate_of = plan
        grown = _grow_layout(rng, dims, up, down_index, islands=rng.random() < .22)
        if grown is None:
            continue
        mask0, rects0 = grown
        height = int(rng.integers(max(mask0.shape[0], 18), 31))
        width = int(rng.integers(max(mask0.shape[1], 7), min(16, mask0.shape[1]+3)+1))
        yshift = int(rng.integers(height-mask0.shape[0]+1))
        xshift = int(rng.integers(width-mask0.shape[1]+1))
        rects = [(y+yshift, x+xshift, h, w) for y, x, h, w in rects0]
        mask = np.zeros((height, width), dtype=bool)
        for y, x, h, w in rects:
            mask[y:y+h, x:x+w] = True
        edge_rows = [(y-1 if u else y+h, x, w) for (y, x, h, w), u in zip(rects, up)]
        if not any((y > 0 if u else y+h < height) for (y, x, h, w), u in zip(rects, up)):
            continue
        if require_partial and not any(0 <= r < height and np.any(mask[r, x:x+w])
                                       and not np.all(mask[r, x:x+w]) for r, x, w in edge_rows):
            continue
        palette = [int(c) for c in rng.permutation(10)]
        board, vacancy, divider = palette[:3]
        payload = palette[3:3+int(rng.integers(3, 8))]
        tiles = []
        # Witness: equal shapes use different inks; unrelated glyphs reuse ink.
        inks = [int(rng.choice(payload)) for _ in glyphs]
        inks[0], inks[1], inks[down_index] = payload[0], payload[1], payload[0]
        for i, (m, (h, w)) in enumerate(zip(glyphs, dims)):
            if i in duplicate_of:
                tiles.append(tiles[duplicate_of[i]].copy())
                continue
            body = int(rng.choice([c for c in payload if c != inks[i]]))
            tile = np.full((h, w), body, dtype=np.int8)
            gy = int(rng.integers(1, h-m.shape[0]))
            gx = int(rng.integers(1, w-m.shape[1]))
            view = tile[gy:gy+m.shape[0], gx:gx+m.shape[1]]
            view[m] = inks[i]
            tiles.append(tile)
        cards = [dict(tile=t, up=u) for t, u in zip(tiles, up)]
        packed = _packing(mask, cards, max_nodes=12000)
        if not packed['exhaustive'] or len(packed['outcomes']) != 1:
            continue
        out = np.full(mask.shape, board, dtype=np.int8)
        for t, (y, x, h, w) in zip(tiles, rects):
            out[y:y+h, x:x+w] = t
        if tuple(int(c) for c in out[mask]) not in packed['outcomes']:
            continue
        # At least one extra locally legal placement makes covering essential.
        if len(packed['placements']) <= len(tiles):
            continue
        if _dead_local_choice(mask, cards, packed) is None:
            continue
        tray = _scatter(rng, tiles, height, 29-width, vacancy)
        if tray is None:
            continue
        stencil = np.full(mask.shape, board, dtype=np.int8)
        stencil[mask] = vacancy
        line = np.full((height, 1), divider, dtype=np.int8)
        inp = np.concatenate((stencil, line, tray) if board_first else (tray, line, stencil), axis=1)
        if transposed:
            inp, out = inp.T, out.T
        pair = dict(input=inp.copy(), output=out.copy())
        try:
            a = _analyze(inp)
        except ValueError:
            continue
        if np.array_equal(solve(inp), out):
            return pair
    raise RuntimeError('Bounded contour-packing construction exhausted')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    phase = bool(rng.integers(2))
    first_side = bool(rng.integers(2))
    for i in range(num_train+num_test):
        training = i < num_train
        profile = 'pair' if i == 0 else 'triplet' if i == 1 else 'combined' if not training else str(rng.choice(['pair', 'triplet', 'combined']))
        transposed = phase if i == 0 else not phase if i == 1 else bool(rng.integers(2))
        side = first_side if training else not first_side
        for _ in range(8):
            p = _make_pair(rng, profile, transposed, side, require_partial=i == 0)
            key = _key(p['input'])
            if key not in seen:
                seen.add(key)
                puzzle['train' if training else 'test'].append(p)
                break
        else:
            raise RuntimeError('Distinct-input construction exhausted')
    return puzzle


def _features(a):
    cards, result = a['cards'], a['result']
    freq = Counter(c['canonical'] for c in cards)
    mirror_negative = any(not c['up'] and _canonical(np.fliplr(c['glyph'])) in freq
                          and _canonical(np.fliplr(c['glyph'])) != c['canonical'] for c in cards)
    both_handed = any(c['up'] and _canonical(np.fliplr(c['glyph'])) != c['canonical']
                      and freq.get(_canonical(np.fliplr(c['glyph'])), 0) >= 2 for c in cards)
    ordinary = any(not c['up'] and
                   (_canonical(np.fliplr(c['glyph'])) == c['canonical']
                    or _canonical(np.fliplr(c['glyph'])) not in freq) for c in cards)
    quarter = any(c['canonical'] == d['canonical'] and _key(c['glyph']) != _key(d['glyph'])
                  and _key(d['glyph']) in (_key(np.rot90(c['glyph'])), _key(np.rot90(c['glyph'], 3)))
                  for c in cards for d in cards)
    partial, interior = False, False
    for p in a['path']:
        g, bits, cells, y, x = result['placements'][p]
        group = result['groups'][g]
        h, w = group['tile'].shape
        r = y-1 if group['up'] else y+h
        interior |= 0 <= r < a['mask'].shape[0]
        partial |= (0 <= r < a['mask'].shape[0] and np.any(a['mask'][r, x:x+w])
                    and not np.all(a['mask'][r, x:x+w]))
    return dict(cards=len(cards), classes=tuple(sorted(freq.values())),
                repeated_classes=sum(n > 1 for n in freq.values()),
                mirror_negative=mirror_negative, ordinary_singleton=ordinary,
                both_handed_repeated=both_handed,
                identical_cards=any(n > 1 for n in Counter(_key(c['tile']) for c in cards).values()),
                symmetric_repeated=any(c['up'] and _key(c['glyph']) == _key(np.rot90(c['glyph'], 2)) for c in cards),
                partial_dock=bool(partial), interior_dock=bool(interior),
                quarter_turn=quarter,
                diagonal=any(len(_components(c['glyph'])) > 1 for c in cards),
                different_inks=any(c['canonical'] == d['canonical'] and c['ink'] != d['ink'] for c in cards for d in cards),
                cross_role_ink=any(c['up'] != d['up'] and c['ink'] == d['ink'] for c in cards for d in cards),
                components=len(_components(a['mask'])), shape=a['mask'].shape,
                axis=int(a['transposed']), board_first=a['board_first'],
                roles=(a['board'], a['vacancy'], a['divider']),
                dimensions=tuple(sorted(c['tile'].shape for c in cards)),
                local_options=len(result['placements']), nodes=result['nodes'], failures=result['failures'])


def _swap_witness(a):
    """An opposite-facing equal-size pair exchanges legally if B is omitted."""
    result = a['result']
    chosen = [result['placements'][p] for p in a['path']]
    for i, (g, bits, cells, y, x) in enumerate(chosen):
        for h, bits2, cells2, yy, xx in chosen[i+1:]:
            cg, ch = result['groups'][g], result['groups'][h]
            if (cg['up'] != ch['up'] and cg['tile'].shape == ch['tile'].shape
                    and not np.array_equal(cg['tile'], ch['tile'])
                    and not _docked(a['mask'], y, x, *cg['tile'].shape, ch['up'])
                    and not _docked(a['mask'], yy, xx, *ch['tile'].shape, cg['up'])):
                # Exchanging entire matrices preserves exact geometric coverage.
                return True
    return False


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), []
    for p in puzzle['train']+puzzle['test']:
        assert isinstance(p, dict) and set(p) == {'input', 'output'}
        inp, out = _grid(p['input']), _grid(p['output'])
        assert _key(inp) not in seen
        seen.add(_key(inp))
        a = _analyze(inp)
        expected = a['output'].T if a['transposed'] else a['output']
        assert np.array_equal(expected, out), 'Output disagrees with input-derived packing'
        assert sum(c['tile'].size for c in a['cards']) == int(a['mask'].sum())
        assert len(a['path']) == len(a['cards']), 'Every physical card is used once'
        assert not np.any(a['output'][a['mask']] == a['vacancy'])
        assert np.all(a['output'][~a['mask']] == a['board'])
        assert _swap_witness(a), 'No equal-size geometric ambiguity witness'
        assert len(a['result']['placements']) > len(a['path']), 'No misleading local placement'
        assert _dead_local_choice(a['mask'], a['cards'], a['result']) is not None, 'No globally excluded local choice'
        assert any(c['up'] for c in a['cards']) and any(not c['up'] for c in a['cards'])
        # Cards include carrier pixels not belonging to the glyph: copying only
        # classified sparse shapes cannot fill the required vacancy silhouette.
        assert all(np.any(c['tile'] == c['body']) for c in a['cards'])
        for pindex in a['path']:
            g, bits, cells, y, x = a['result']['placements'][pindex]
            tile = a['result']['groups'][g]['tile']
            h, w = tile.shape
            assert np.array_equal(a['output'][y:y+h, x:x+w], tile)
            assert _docked(a['mask'], y, x, h, w, a['result']['groups'][g]['up'])
        features.append(_features(a))
    tr, te = features[:nt], features[nt:]
    assert {f['axis'] for f in tr} == {0, 1}, 'Both divider frames need demonstrations'
    assert any(2 in f['classes'] for f in tr) and any(3 in f['classes'] for f in tr)
    assert any(f['ordinary_singleton'] for f in tr)
    assert any(f['partial_dock'] for f in tr)
    assert all(f['interior_dock'] for f in features)
    for k in ('mirror_negative', 'both_handed_repeated', 'quarter_turn', 'different_inks', 'cross_role_ink', 'diagonal'):
        assert any(f[k] for f in tr), f'Missing teaching witness: {k}'
    assert all(f['board_first'] != tr[0]['board_first'] for f in te)
    assert all(f['repeated_classes'] >= 2 for f in te)


def self_test() -> dict:
    puzzles, fs, structures, shapes = set(), [], set(), set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple((_key(q['input']), _key(q['output'])) for s in p for q in p[s])
        puzzles.add(signature)
        for s in p:
            for q in p[s]:
                a = _analyze(q['input'])
                fs.append(_features(a))
                shapes.add(tuple(q['input'].shape))
                roles = {}
                structures.add(tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row) for row in q['input']))
    assert len(puzzles) == 42 and len(structures) >= 180
    assert len(shapes) >= 25
    assert len({f['dimensions'] for f in fs}) >= 60
    assert {f['axis'] for f in fs} == {0, 1}
    assert {f['board_first'] for f in fs} == {False, True}
    assert len({f['cards'] for f in fs}) >= 3
    assert len({f['roles'] for f in fs}) >= 100
    assert any(f['components'] > 1 for f in fs)
    assert any(f['failures'] > 0 for f in fs)
    assert any(f['identical_cards'] for f in fs)
    assert any(f['symmetric_repeated'] for f in fs)
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(seed+100, *counts)
        validate_generated_puzzle(p)
    def signature(p):
        return tuple((_key(q['input']), _key(q['output'])) for s in p for q in p[s])
    assert signature(generate_puzzle(77)) == signature(generate_puzzle(77))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    # Unit checks of the new semantic distinction rather than only labels.
    l = np.array([[1, 0, 0], [1, 1, 1]], dtype=bool)
    assert _canonical(l) == _canonical(np.rot90(l))
    assert _canonical(l) != _canonical(np.fliplr(l))
    full = np.ones((8, 4), dtype=bool)
    assert _docked(full, 0, 0, 4, 4, True)
    assert not _docked(full, 0, 0, 4, 4, False)
    assert _docked(full, 4, 0, 4, 4, False)
    assert not _docked(full, 4, 0, 4, 4, True)
    # Missing inventory cannot cover twice the provided footprint.
    assert not _packing(full, [dict(tile=np.ones((4, 4), dtype=int), up=True)])['outcomes']
    p = generate_puzzle(123)
    bad = {s: [dict(input=np.array(q['input']), output=np.array(q['output'])) for q in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')
    return dict(seeds=42, pairs=len(fs), distinct_puzzles=len(puzzles),
                input_shapes=len(shapes), normalized_structures=len(structures),
                dimension_profiles=len({f['dimensions'] for f in fs}),
                card_counts=sorted({f['cards'] for f in fs}),
                disconnected_examples=sum(f['components'] > 1 for f in fs),
                backtracking_examples=sum(f['failures'] > 0 for f in fs))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
