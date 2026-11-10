"""Invisible cargo: explicit stencils participate in directional wall collision.

No parent imports, hidden dictionary, seed-dependent solving, or stored examples.
Construction traces local arena masks; solving reconstructs closed regions and
the visible catalogue from the input. Checks are generator-owned, not independent.
"""
from __future__ import annotations

import copy
import json
import numpy as np

DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(np.all((a >= 0) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _fingerprint(a):
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    directions = DIRECTIONS if diagonal else CARDINAL
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, pending = {start}, [start]
        while pending:
            r, c = pending.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    pending.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _offsets(mask):
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _fits(free, anchor, offsets):
    r, c = anchor
    return all((r + dr, c + dc) in free for dr, dc in offsets)


def _trace(free, start, offsets, direction, corner=True):
    """Return consecutive legal reference positions, or None for an illegal start."""
    if not _fits(free, start, offsets):
        return None
    dr, dc = direction
    r, c = start
    path = [start]
    # Tight stencils and bounded regions prevent more than 29 unit steps.
    for _ in range(30):
        nxt = r + dr, c + dc
        if not _fits(free, nxt, offsets):
            break
        if corner and dr and dc:
            if not (_fits(free, (r + dr, c), offsets)
                    and _fits(free, (r, c + dc), offsets)):
                break
        r, c = nxt
        path.append(nxt)
    else:
        raise AssertionError("unbounded trajectory")
    return path


def _sweep(path, offsets, direction):
    poses = list(path)
    dr, dc = direction
    if dr and dc:
        for r, c in path[:-1]:
            poses.extend(((r + dr, c), (r, c + dc)))
    cells = {(r + a, c + b) for r, c in poses for a, b in offsets}
    cells.add(path[0])  # Protect the source symbol even when its corner is blank.
    return cells


def _decode_arrow(top, background, wall, ink):
    tips = set(_offsets(top == ink))
    trails = set(_offsets(top == wall))
    candidates = []
    if len(tips) < 2 or set(map(int, np.unique(top))) != {background, wall, ink}:
        return candidates
    for dr, dc in DIRECTIONS:
        used, lengths = set(), []
        for r, c in sorted(tips):
            chain = []
            r, c = r - dr, c - dc
            while (r, c) in trails:
                chain.append((r, c))
                r, c = r - dr, c - dc
            if len(chain) < 2 or used.intersection(chain):
                break
            used.update(chain)
            lengths.append(len(chain))
        if len(lengths) == len(tips) and used == trails:
            candidates.append(((dr, dc), tuple(lengths)))
    return candidates


def _read(input_grid):
    a = _grid(input_grid)
    frequencies = np.bincount(a.ravel(), minlength=10)
    bg = int(frequencies.argmax())
    _require(frequencies[bg] > a.size // 2, "background must occupy over half the input")
    cards, card_mask = {}, np.zeros(a.shape, dtype=bool)
    # A card's same-color frame and divider are one four-connected component.
    # Do not infer legends from their position relative to the arena.
    for color in sorted(set(map(int, np.unique(a))) - {bg}):
        for component in _components(a == color):
            r0, r1, c0, c1 = _bounds(component)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 8 or w < 5:
                continue
            sub = a[r0:r1 + 1, c0:c1 + 1]
            if not (np.all(sub[0] == color) and np.all(sub[-1] == color)
                    and np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
                continue
            dividers = [r for r in range(2, h - 2) if np.all(sub[r] == color)]
            if len(dividers) != 1:
                continue
            div = dividers[0]
            frame = np.zeros(sub.shape, bool)
            frame[0] = frame[-1] = True
            frame[:, 0] = frame[:, -1] = True
            frame[div] = True
            if not np.array_equal(sub == color, frame):
                continue
            upper = sub[1:div, 1:-1]
            lower = sub[div + 1:-1, 1:-1]
            inks = set(map(int, np.unique(lower))) - {bg}
            if len(inks) != 1:
                continue
            ink = inks.pop()
            walls = set(map(int, np.unique(upper))) - {bg, ink}
            if len(walls) != 1 or ink == color:
                continue
            wall = walls.pop()
            if wall == color:
                continue
            decoded = _decode_arrow(upper, bg, wall, ink)
            if len(decoded) != 1:
                continue
            mr0, mr1, mc0, mc1 = _bounds(_offsets(lower == ink))
            mask = (lower[mr0:mr1 + 1, mc0:mc1 + 1] == ink)
            _require(len(_components(mask, diagonal=True)) == 1, "disconnected stencil")
            _require(color not in cards, "duplicate code cards")
            _require(not card_mask[r0:r1 + 1, c0:c1 + 1].any(), "overlapping cards")
            card_mask[r0:r1 + 1, c0:c1 + 1] = True
            cards[color] = dict(mask=mask, offsets=_offsets(mask), ink=ink, wall=wall,
                                direction=decoded[0][0], lengths=decoded[0][1],
                                bbox=(r0, r1, c0, c1))
    _require(cards, "missing catalogue")
    walls = {card['wall'] for card in cards.values()}
    inks = {card['ink'] for card in cards.values()}
    codes = set(cards)
    _require(len(walls) == 1 and len(inks) == len(cards), "ambiguous palette roles")
    wall = next(iter(walls))
    _require(len({bg, wall} | codes | inks) == 2 + 2 * len(cards), "colliding palette roles")
    _require(set(map(int, np.unique(a[~card_mask]))) <= {bg, wall} | codes,
             "unexplained material outside catalogue")
    wall_mask = (a == wall) & ~card_mask
    labels = np.full(a.shape, -1, dtype=int)
    regions = _components(~wall_mask & ~card_mask)
    for label, cells in enumerate(regions):
        for q in cells:
            labels[q] = label
    payloads = []
    for code, card in cards.items():
        starts = [tuple(map(int, q)) for q in np.argwhere((a == code) & ~card_mask)]
        _require(starts, "unused catalogue entry")
        for start in starts:
            label = int(labels[start])
            _require(label >= 0, "symbol on an obstacle")
            free = regions[label]
            _require(not any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                             for r, c in free), "symbol outside a closed arena")
            _require(_fits(free, start, card['offsets']), "initial cargo does not fit its region")
            payloads.append(dict(start=start, code=code, free=free, **card))
    starts = [p['start'] for p in payloads]
    _require(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
                 for i, a in enumerate(starts) for b in starts[i + 1:]), "touching symbols")
    return dict(grid=a, background=bg, wall=wall, walls=wall_mask,
                cards=cards, card_mask=card_mask, payloads=payloads)


def _analyze(info):
    a = info['grid']
    out = a.copy()
    for p in info['payloads']:
        out[p['start']] = info['background']
    used, records = set(), []
    for p in info['payloads']:
        path = _trace(p['free'], p['start'], p['offsets'], p['direction'])
        _require(path is not None, "illegal cargo")
        swept = _sweep(path, p['offsets'], p['direction'])
        _require(used.isdisjoint(swept), "interacting cargo sweeps")
        used.update(swept)
        end = path[-1]
        for dr, dc in p['offsets']:
            q = end[0] + dr, end[1] + dc
            _require(not info['walls'][q] and not info['card_mask'][q], "material overwrote structure")
            out[q] = p['ink']
        point = _trace(p['free'], p['start'], ((0, 0),), p['direction'])
        box = tuple((r, c) for r in range(p['mask'].shape[0])
                    for c in range(p['mask'].shape[1]))
        box_path = _trace(p['free'], p['start'], box, p['direction'])
        naive = _trace(p['free'], p['start'], p['offsets'], p['direction'], corner=False)
        records.append(dict(code=p['code'], start=p['start'], end=end,
                            distance=len(path) - 1, direction=p['direction'], sweep=swept,
                            point_witness=end != point[-1],
                            box_witness=box_path is None or box_path[-1] != end,
                            corner_witness=naive[-1] != end,
                            offsets=p['offsets'], ink=p['ink']))
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the supplied integer grid, without learning or state."""
    return _analyze(_read(input_grid))[0]


def _sample_mask(rng, sparse):
    h = int(rng.integers(3 if sparse else 2, 5))
    w = int(rng.integers(3 if sparse else 2, 6))
    kind = int(rng.integers(3 if sparse else 6))
    m = np.zeros((h, w), dtype=bool)
    if kind == 0:
        m[:, 0] = m[-1] = True
        # A partial arm broadens the concavity without closing it.
        if rng.random() < .6:
            m[int(rng.integers(h - 1)), :int(rng.integers(1, w))] = True
    elif kind in (1, 2):
        r = c = 0
        m[0, 0] = True
        while r < h - 1 or c < w - 1:
            if r == h - 1 or (c < w - 1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            m[r, c] = True
        if kind == 2:
            for _ in range(int(rng.integers(1, max(2, h * w // 3)))):
                r, c = _offsets(m)[int(rng.integers(np.count_nonzero(m)))]
                dr, dc = CARDINAL[int(rng.integers(4))]
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    m[r + dr, c + dc] = True
        # Preserve an open corner in the primary cargo's sparse geometry.
        if sparse and m[0, -1] and m[-1, 0]:
            m[0, -1] = False
            # It may be an articulation point: use a fresh open L in that case.
            if len(_components(m, diagonal=True)) != 1:
                m[:] = False
                m[:, 0] = m[-1] = True
    elif kind == 3:
        m[0] = m[-1] = True
        m[:, 0] = m[:, -1] = True
    elif kind == 4:
        if h >= 3 and w >= 3:
            m[0, 1:-1] = m[-1, 1:-1] = True
            m[1:-1, 0] = m[1:-1, -1] = True
        else:
            m[:] = True
    else:
        m[:] = True
    if rng.random() < .5:
        m = m[::-1]
    if rng.random() < .5:
        m = m[:, ::-1]
    return m.copy()


def _card(rng, code, ink, wall, bg, mask, direction):
    dr, dc = direction
    length = int(rng.choice([2, 2, 3]))
    points = []
    for stroke in range(2):
        base = (stroke * 2, 0) if dr == 0 else (0, stroke * 2)
        stagger = int(rng.integers(2))
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc,
                           ink if j == length else wall))
    r0, r1, c0, c1 = _bounds([(r, c) for r, c, _ in points])
    gh, gw = r1 - r0 + 1, c1 - c0 + 1
    upper_h = gh + int(rng.integers(2))
    lower_h = mask.shape[0] + int(rng.integers(2))
    inner_w = max(gw, mask.shape[1]) + int(rng.integers(2))
    # Top and bottom bays remain visually distinct even for compact cardinals.
    upper_h = max(3, upper_h)
    a = np.full((upper_h + lower_h + 3, inner_w + 2), bg, dtype=np.int8)
    a[0] = a[-1] = code
    a[:, 0] = a[:, -1] = code
    divider = upper_h + 1
    a[divider] = code
    ar = 1 + int(rng.integers(upper_h - gh + 1))
    ac = 1 + int(rng.integers(inner_w - gw + 1))
    for r, c, color in points:
        a[r - r0 + ar, c - c0 + ac] = color
    sr = divider + 1 + int(rng.integers(lower_h - mask.shape[0] + 1))
    sc = 1 + int(rng.integers(inner_w - mask.shape[1] + 1))
    a[sr:sr + mask.shape[0], sc:sc + mask.shape[1]][mask] = ink
    return a


def _layout(rng, cards):
    """Four sides, independent alignment and shuffled catalogue ordering."""
    side = int(rng.integers(4))
    gap = int(rng.integers(1, 4))
    pad = [int(rng.integers(2)) for _ in range(4)]
    order = list(map(int, rng.permutation(len(cards))))
    between = int(rng.integers(1, 3))
    placements = []
    if side in (0, 2):
        ch = max(a.shape[0] for a in cards)
        cw = sum(a.shape[1] for a in cards) + between * (len(cards) - 1)
        bh = int(rng.integers(10, 31 - ch - gap - pad[0] - pad[1]))
        bw = int(rng.integers(10, 27))
        hmin = bh + ch + gap + pad[0] + pad[1]
        wmin = max(bw, cw) + pad[2] + pad[3]
        H = int(rng.integers(hmin, 31))
        W = int(rng.integers(wmin, 31))
        if side == 0:
            strip_r, br = pad[0], H - pad[1] - bh
        else:
            br, strip_r = pad[0], H - pad[1] - ch
        bc = int(rng.integers(pad[2], W - pad[3] - bw + 1))
        col = int(rng.integers(pad[2], W - pad[3] - cw + 1))
        for index in order:
            a = cards[index]
            row = strip_r + int(rng.integers(ch - a.shape[0] + 1))
            placements.append((index, row, col))
            col += a.shape[1] + between
    else:
        ch = sum(a.shape[0] for a in cards) + between * (len(cards) - 1)
        cw = max(a.shape[1] for a in cards)
        if ch + pad[0] + pad[1] > 30:
            between = 1
            ch = sum(a.shape[0] for a in cards) + len(cards) - 1
            pad[0] = pad[1] = 0
        bw = int(rng.integers(10, 31 - cw - gap - pad[2] - pad[3]))
        bh = int(rng.integers(10, 27))
        hmin = max(bh, ch) + pad[0] + pad[1]
        wmin = bw + cw + gap + pad[2] + pad[3]
        H = int(rng.integers(hmin, 31))
        W = int(rng.integers(wmin, 31))
        if side == 1:
            strip_c, bc = pad[2], W - pad[3] - bw
        else:
            bc, strip_c = pad[2], W - pad[3] - cw
        br = int(rng.integers(pad[0], H - pad[1] - bh + 1))
        row = int(rng.integers(pad[0], H - pad[1] - ch + 1))
        for index in order:
            a = cards[index]
            col = strip_c + int(rng.integers(cw - a.shape[1] + 1))
            placements.append((index, row, col))
            row += a.shape[0] + between
    return (H, W), (br, bc), (bh, bw), placements


def _arena(rng, h, w, test):
    footprint = np.ones((h, w), dtype=bool)
    kind = int(rng.integers(4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            rh = int(rng.integers(1, max(2, h // 3)))
            cw = int(rng.integers(1, max(2, w // 3)))
            rr = slice(0, rh) if corner < 2 else slice(h - rh, h)
            cc = slice(0, cw) if corner % 2 == 0 else slice(w - cw, w)
            footprint[rr, cc] = False
    elif kind == 2:
        rr, cc = np.indices((h, w))
        cy, cx = (h - 1) / 2, (w - 1) / 2
        exponent = float(rng.uniform(1.4, 3.2))
        footprint = (abs((rr - cy) / (cy + .2)) ** exponent
                     + abs((cc - cx) / (cx + .2)) ** exponent <= 1.15)
    elif kind == 3:
        left = right = 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, max(1, w // 5)))
            right = int(np.clip(right + rng.integers(-1, 2), 0, max(1, w // 5)))
            footprint[r, :left] = False
            if right:
                footprint[r, -right:] = False
    padded = np.pad(footprint, 1)
    interior = footprint.copy()
    for dr, dc in CARDINAL:
        interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    walls = footprint & ~interior
    cells = _offsets(interior)
    if not cells:
        return walls, footprint
    peg_count = int(rng.integers(3 if test else 2, 8 if test else 5))
    for index in rng.choice(len(cells), min(peg_count, len(cells)), replace=False):
        walls[cells[int(index)]] = True
    for _ in range(int(rng.integers(3 if test else 2))):
        if rng.random() < .5:
            r = int(rng.integers(2, h - 2))
            cut = int(rng.integers(2, max(3, w // 2)))
            cc = slice(0, cut) if rng.random() < .5 else slice(w - cut, w)
            walls[r, cc] |= footprint[r, cc]
        else:
            c = int(rng.integers(2, w - 2))
            cut = int(rng.integers(2, max(3, h // 2)))
            rr = slice(0, cut) if rng.random() < .5 else slice(h - cut, h)
            walls[rr, c] |= footprint[rr, c]
    return walls, footprint


def _candidates(walls, footprint, mask, direction, detailed):
    offsets = _offsets(mask)
    box = tuple((r, c) for r in range(mask.shape[0]) for c in range(mask.shape[1]))
    candidates = []
    for free in _components(footprint & ~walls):
        for start in sorted(free):
            path = _trace(free, start, offsets, direction)
            if path is None:
                continue
            item = dict(start=start, path=path, sweep=_sweep(path, offsets, direction),
                        moving=len(path) > 1, point=False, box=False, corner=False)
            if detailed:
                point = _trace(free, start, ((0, 0),), direction)
                rect = _trace(free, start, box, direction)
                naive = _trace(free, start, offsets, direction, corner=False)
                item.update(point=point[-1] != path[-1],
                            box=rect is None or rect[-1] != path[-1],
                            corner=naive[-1] != path[-1])
            candidates.append(item)
    return candidates


def _make_pair(rng, book, wall, bg, directions, witness, test):
    codes = list(directions)
    for _ in range(100):
        cards = [_card(rng, code, book[code][0], wall, bg,
                       book[code][1], directions[code]) for code in codes]
        shape, origin, board_shape, placements = _layout(rng, cards)
        walls, footprint = _arena(rng, *board_shape, test)
        options = {code: _candidates(walls, footprint, book[code][1], directions[code],
                                     detailed=bool(witness) and code == codes[0])
                   for code in codes}
        selected, used, starts = [], set(), []

        def add(code, choices):
            for index in rng.permutation(len(choices)):
                item = choices[int(index)]
                if not used.isdisjoint(item['sweep']):
                    continue
                if any(max(abs(item['start'][0] - q[0]), abs(item['start'][1] - q[1])) <= 1
                       for q in starts):
                    continue
                selected.append((code, item))
                used.update(item['sweep'])
                starts.append(item['start'])
                return True
            return False

        good = True
        for i, code in enumerate(codes):
            choices = [x for x in options[code] if x['moving']]
            if witness == 'point' and i == 0:
                choices = [x for x in choices if x['point']]
            if witness == 'gap_corner' and i == 0:
                choices = [x for x in choices if x['corner']]
            if not add(code, choices):
                good = False
                break
        if not good:
            continue
        # These two distinctions need not occur on the same cargo. Some sparse
        # silhouettes mathematically cannot have both on a single trajectory.
        if witness == 'gap_corner' and not any(x['box'] for _, x in selected):
            choices = [x for x in options[codes[0]] if x['moving'] and x['box']]
            if not add(codes[0], choices):
                continue
        desired = int(rng.integers(len(codes), 6))
        for code in [codes[int(i)] for i in rng.permutation(len(codes))] * 5:
            if len(selected) >= desired:
                break
            choices = options[code]
            if rng.random() < .85:
                choices = [x for x in choices if x['moving']]
            add(code, choices)
        a = np.full(shape, bg, dtype=np.int8)
        br, bc = origin
        bh, bw = board_shape
        a[br:br + bh, bc:bc + bw][walls] = wall
        for i, r, c in placements:
            card = cards[i]
            a[r:r + card.shape[0], c:c + card.shape[1]] = card
        for code, item in selected:
            r, c = item['start']
            a[br + r, bc + c] = code
        if np.count_nonzero(a == bg) <= a.size // 2:
            continue
        # Label construction uses selected local endpoints, not solve().
        out = a.copy()
        for code, item in selected:
            r, c = item['start']
            out[br + r, bc + c] = bg
        for code, item in selected:
            r, c = item['path'][-1]
            ink, mask = book[code]
            for dr, dc in _offsets(mask):
                out[br + r + dr, bc + c + dc] = ink
        return dict(input=a, output=out)
    raise AssertionError("bounded arena construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(1, 3))
    palette = list(map(int, rng.permutation(10)))
    codes = palette[:count]
    inks = palette[count:2 * count]
    wall = palette[2 * count]
    backgrounds = palette[2 * count + 1:2 * count + 3]
    book = {code: (ink, _sample_mask(rng, sparse=i == 0))
            for i, (code, ink) in enumerate(zip(codes, inks))}
    # The first two pairs teach cardinal motion and diagonal swept clearance.
    first = (0, 1) if rng.random() < .5 else (1, 0)
    second = DIAGONAL[int(rng.integers(4))]
    pools = {}
    for i, code in enumerate(codes):
        base = [first, second] if i == 0 else [DIRECTIONS[int(rng.integers(8))],
                                                 DIRECTIONS[int(rng.integers(8))]]
        pool = list(dict.fromkeys(base))
        target = int(rng.integers(max(2, len(pool)), 5))
        remaining = [d for d in DIRECTIONS if d not in pool]
        pool += [remaining[int(j)] for j in rng.choice(len(remaining), target - len(pool), replace=False)]
        pools[code] = pool
    puzzle, seen = {'train': [], 'test': []}, set()
    for index in range(nt + nq):
        is_test = index >= nt
        subset = list(codes)
        if index >= 2 and count == 2 and rng.random() < .4:
            subset = codes[:1]
        directions = {}
        for i, code in enumerate(subset):
            choices = [d for d in DIRECTIONS if d not in pools[code]] if is_test else pools[code]
            directions[code] = choices[int(rng.integers(len(choices)))]
            if not is_test and i == 0 and index < 2:
                directions[code] = first if index == 0 else second
        bg = backgrounds[index % 2] if index < 2 else int(rng.choice(backgrounds))
        witness = ('point' if index == 0 else 'gap_corner') if index < 2 else None
        for _ in range(8):
            pair = _make_pair(rng, book, wall, bg, directions, witness, is_test)
            fp = _fingerprint(pair['input'])
            if fp not in seen:
                break
        _require(fp not in seen, "duplicate input construction")
        seen.add(fp)
        puzzle['test' if is_test else 'train'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, dictionary, training_dirs = set(), {}, {}
    train_records, test_records, analyses = [], [], []
    wall_role = None
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, "dimensions changed")
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _read(a)
            expected, records = _analyze(info)
            _require(np.array_equal(expected, b), "output disagrees with input-only rule")
            _require(not np.array_equal(a, b), "identity pair")
            fixed = info['card_mask'] | info['walls']
            _require(np.array_equal(a[fixed], b[fixed]), "structure changed")
            _require(not np.isin(b[~info['card_mask']], list(info['cards'])).any(),
                     "source symbols remain")
            rendered = sum(len(p['offsets']) for p in info['payloads'])
            _require(np.count_nonzero(np.isin(b, [c['ink'] for c in info['cards'].values()])
                                      & ~info['card_mask']) == rendered, "cargo material count")
            if wall_role is None:
                wall_role = info['wall']
            _require(info['wall'] == wall_role, "wall convention changed")
            for code, card in info['cards'].items():
                _require(all(2 <= n <= 5 for n in card['mask'].shape), "generated stencil size")
                _require(len(card['lengths']) == 2 and set(card['lengths']) <= {2, 3},
                         "generated arrow syntax")
                signature = card['ink'], _fingerprint(card['mask'])
                if split == 'train':
                    _require(code not in dictionary or dictionary[code] == signature,
                             "dictionary changed within puzzle")
                    dictionary[code] = signature
                    training_dirs.setdefault(code, set()).add(card['direction'])
                else:
                    _require(code in dictionary and dictionary[code] == signature, "untaught dictionary")
            (train_records if split == 'train' else test_records).extend(records)
            analyses.append((split, info, records))
    for _, info, _ in analyses[:2]:
        _require(set(info['cards']) == set(dictionary), "first two pairs must show the dictionary")
    global_roles = set(dictionary) | {value[0] for value in dictionary.values()} | {wall_role}
    _require(all(info['background'] not in global_roles for _, info, _ in analyses),
             "background reused as a dictionary role")
    _require(any(r['distance'] > 0 and r['point_witness'] for r in train_records),
             "missing point-versus-shape witness")
    _require(any(r['distance'] > 0 and r['box_witness'] for r in train_records),
             "missing transparent-space witness")
    _require(any(r['distance'] > 0 and r['corner_witness'] for r in train_records),
             "missing diagonal corner witness")
    _require(any(0 in r['direction'] for r in train_records)
             and any(0 not in r['direction'] for r in train_records), "direction demonstrations")
    _require(all(any(r['direction'] not in training_dirs[r['code']] for r in records)
                 for split, _, records in analyses if split == 'test'), "missing test orientation shift")
    # Counterfactual that omits A: materialize every stencil at its original pose.
    no_motion_differs = False
    no_material_differs = False
    for split, info, records in analyses:
        if split != 'train':
            continue
        static = info['grid'].copy()
        for r in records:
            static[r['start']] = info['background']
        point_only = static.copy()
        for r in records:
            for dr, dc in r['offsets']:
                static[r['start'][0] + dr, r['start'][1] + dc] = r['ink']
            point_only[r['end']] = r['ink']
        expected = _analyze(info)[0]
        no_motion_differs |= not np.array_equal(static, expected)
        no_material_differs |= not np.array_equal(point_only, expected)
    _require(no_motion_differs, "A contributes no observable motion")
    _require(no_material_differs, "B contributes no observable substitution")


def _measure(puzzle):
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _read(pair['input'])
            _, records = _analyze(info)
            cells = _offsets(info['walls'])
            r0, r1, c0, c1 = _bounds(cells)
            wall_box = info['walls'][r0:r1 + 1, c0:c1 + 1]
            irregular = not (wall_box[0].all() and wall_box[-1].all()
                             and wall_box[:, 0].all() and wall_box[:, -1].all())
            center = ((r0 + r1) / 2, (c0 + c1) / 2)
            sides = []
            for card in info['cards'].values():
                a, b, c, d = card['bbox']
                dy, dx = (a + b) / 2 - center[0], (c + d) / 2 - center[1]
                sides.append(('bottom' if dy > 0 else 'top') if abs(dy) >= abs(dx)
                             else ('right' if dx > 0 else 'left'))
            rows.append(dict(split=split, shape=list(info['grid'].shape),
                             classes=len(info['cards']), cargoes=len(records),
                             directions=[list(r['direction']) for r in records],
                             distances=[r['distance'] for r in records],
                             mask_shapes=[list(c['mask'].shape) for c in info['cards'].values()],
                             masks=[c['mask'].astype(int).tolist() for c in info['cards'].values()],
                             background=info['background'], wall=info['wall'],
                             key_sides=sides, irregular=irregular,
                             point_witnesses=sum(r['point_witness'] for r in records),
                             box_witnesses=sum(r['box_witness'] for r in records),
                             corner_witnesses=sum(r['corner_witness'] for r in records)))
    return rows


def self_test():
    """Bounded 42-seed diversity, all explicit counts and semantic corruption checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [r for p in puzzles for r in _measure(p)]
    _require(len({_fingerprint(p['train'][0]['input']) for p in puzzles}) == 42, "seed diversity")
    shapes = {tuple(r['shape']) for r in rows}
    masks = {json.dumps(m, separators=(',', ':')) for r in rows for m in r['masks']}
    _require(len(shapes) >= 35 and len(masks) >= 25, "structural variation")
    _require({r['classes'] for r in rows} == {1, 2}, "dictionary sizes")
    _require(len({r['cargoes'] for r in rows}) >= 4, "cargo count variation")
    _require({tuple(d) for r in rows for d in r['directions']} == set(DIRECTIONS), "directions")
    _require(any(r['irregular'] for r in rows) and any(not r['irregular'] for r in rows), "boundary shapes")
    _require({s for r in rows for s in r['key_sides']} == {'top', 'bottom', 'left', 'right'}, "key sides")
    _require(len({tuple(s) for r in rows for s in r['mask_shapes']}) >= 8, "stencil dimensions")
    _require({d for r in rows for d in r['distances']} >= {0, 1, 2, 3, 4, 5}, "motion lengths")
    _require(any(30 in r['shape'] for r in rows), "full-size canvas")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(144, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit count contract")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a, b in zip(puzzles[0]['train'] + puzzles[0]['test'],
                    generate_puzzle(0)['train'] + generate_puzzle(0)['test']):
        _require(all(np.array_equal(a[k], b[k]) for k in ('input', 'output')), "nondeterminism")
    # Role permutation must not affect interpretation, even when background is 0.
    permutation = np.array([7, 5, 9, 0, 6, 2, 4, 8, 1, 3], dtype=np.int8)
    for pair in puzzles[7]['train'] + puzzles[7]['test']:
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 "literal color shortcut")
    for corruption in ('identity', 'missing_material'):
        bad = copy.deepcopy(puzzles[0])
        pair = bad['test'][0]
        if corruption == 'identity':
            pair['output'] = pair['input'].copy()
        else:
            info = _read(pair['input'])
            _, records = _analyze(info)
            record = records[0]
            for dr, dc in record['offsets']:
                pair['output'][record['end'][0] + dr, record['end'][1] + dc] = info['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("output corruption accepted")
    return dict(seeds=42, pairs=len(rows), grid_shapes=len(shapes), distinct_masks=len(masks),
                cargo_counts=sorted({r['cargoes'] for r in rows}),
                irregular_pairs=sum(r['irregular'] for r in rows),
                point_witnesses=sum(r['point_witnesses'] for r in rows),
                box_witnesses=sum(r['box_witnesses'] for r in rows),
                corner_witnesses=sum(r['corner_witnesses'] for r in rows),
                independent_solver=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
