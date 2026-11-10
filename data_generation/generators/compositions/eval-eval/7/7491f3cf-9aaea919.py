"""Split-growth printing: two counted ink supplies share one spatially cut tower.

Only standard-library modules and NumPy are used. No files are read or written.
The seed affects sampling only; solve() reconstructs every convention from pixels.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("example counts must be integers")
        if not low <= value <= high:
            raise ValueError("example counts outside supported range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _components(mask, diagonal=False):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        todo, cells = [(r, c)], []
        while todo:
            y, x = todo.pop()
            cells.append((y, x))
            for dy, dx in directions:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    todo.append((yy, xx))
        result.append(cells)
    return result


def _regions(cut):
    components = _components(~cut)
    _require(len(components) == 2, "curtain must divide the chamber into two regions")
    result = []
    for cells in components:
        mask = np.zeros_like(cut)
        for p in cells:
            mask[p] = True
        result.append(mask)
    return result


def _path(cut):
    """An eight-connected digital path; omit diagonal shortcuts at square bends."""
    h, w = cut.shape
    cells = {tuple(map(int, p)) for p in np.argwhere(cut)}
    graph = {}
    for y, x in cells:
        graph[y, x] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if not (dy or dx) or q not in cells:
                    continue
                if dy and dx and ((y + dy, x) in cells or (y, x + dx) in cells):
                    continue
                graph[y, x].append(q)
    ends = [p for p, neighbors in graph.items() if len(neighbors) == 1]
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in graph.values()), "branched or cyclic curtain")
    border = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
    _require(all(border(p) == (p in ends) for p in cells), "invalid curtain boundary contacts")
    _require(any(p[0] == 0 for p in ends) and any(p[1] in (0, w - 1) for p in ends),
             "curtain must join the top to a side")
    order, previous, current = [], None, min(ends)
    while current is not None and current not in order:
        order.append(current)
        following = [p for p in graph[current] if p != previous]
        previous, current = current, following[0] if following else None
    _require(len(order) == len(cells), "disconnected curtain")
    return order


def _sites(cut, region, occupied):
    near = cut.copy()
    h, w = cut.shape
    for y, x in np.argwhere(cut):
        near[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = True
    return [tuple(map(int, p)) for p in np.argwhere(region & ~near & ~occupied)]


def _token_stack(mask):
    """Decode a vertically repeated silhouette from a lane's binary support."""
    rows = _runs(np.any(mask, axis=1))
    _require(bool(rows), "empty token stack")
    cropped = []
    bounds = []
    for a, b in rows:
        ys, xs = np.nonzero(mask[a:b])
        lo, hi = int(xs.min()), int(xs.max()) + 1
        token = mask[a:b, lo:hi]
        _require(1 <= token.shape[0] <= 3 and 1 <= token.shape[1] <= 6, "invalid token dimensions")
        _require(len(_components(token)) == 1, "token must be four-connected")
        cropped.append(token)
        bounds.append((lo, hi))
    _require(all(b == bounds[0] for b in bounds), "misaligned stack")
    _require(all(np.array_equal(t, cropped[0]) for t in cropped), "different tokens in one stack")
    starts = [a for a, _ in rows]
    pitch = cropped[0].shape[0] + 1
    _require(all(b - a == pitch for a, b in zip(starts, starts[1:])), "tokens need exactly one blank row")
    return dict(mask=cropped[0], x=bounds[0][0], ys=starts, count=len(rows), pitch=pitch)


def _decode(input_grid):
    g = _grid(input_grid)
    h, width = g.shape
    candidates = []
    for frame in np.unique(g[0]):
        frame = int(frame)
        for left, end in _runs(g[0] == frame):
            if end - left < 7 or h < 10 or not np.all(g[:, [left, end - 1]] == frame):
                continue
            outside = np.r_[g[0, :left], g[0, end:]]
            if len(outside) == 0 or len(np.unique(outside)) != 1:
                continue
            bg = int(outside[0])
            if bg == frame:
                continue
            ports = int(g[-1, left + 1]), int(g[-1, end - 2])
            if len({bg, frame, *ports}) != 4:
                continue
            bottom = g[-1, left:end].copy()
            bottom[[1, -2]] = frame
            if np.all(bottom == frame):
                candidates.append((left, end, frame, bg, ports))
    _require(len(candidates) == 1, "missing or ambiguous printing chamber")
    left, end, frame, bg, ports = candidates[0]
    chamber = g[1:-1, left + 1:end - 1]
    hi, wi = chamber.shape
    colors = set(map(int, np.unique(chamber))) - {bg}
    _require(len(colors) == 2 and frame not in colors and not set(ports) & colors, "invalid chamber colors")
    controls = []
    for color in sorted(colors):
        parts = sorted(_components(chamber == color, True), key=len)
        if len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) > 3:
            controls.append((color, parts[0][0], parts[1]))
    _require(len(controls) == 1, "need one curtain and one isolated marker")
    control, marker, cells = controls[0]
    cut = np.zeros_like(chamber, dtype=bool)
    for p in cells:
        cut[p] = True
    path = _path(cut)
    regions = _regions(cut)
    side = 0 if regions[0][marker] else 1
    seed_color = next(iter(colors - {control}))
    seed = _token_stack(chamber == seed_color)
    _require(2 <= seed['mask'].shape[0] <= 3 and 4 <= seed['mask'].shape[1] <= 6,
             "receiver must have a readable wide token")
    _require(seed['ys'][-1] + seed['mask'].shape[0] == hi - 1, "receiver baseline is not above the bottom gap")
    _require(0 < seed['x'] and seed['x'] + seed['mask'].shape[1] < wi, "receiver touches a side")
    _require(max(y for y, x in cells) < seed['ys'][0], "curtain intersects existing receiver")
    _require(marker in _sites(cut, regions[side], chamber == seed_color), "marker touches curtain or seed")
    _require(all(_sites(cut, r, chamber == seed_color) for r in regions), "both regions need marker sites")
    outside_mask = g != bg
    outside_mask[:, left:end] = False
    stacks = []
    expected_outside = np.full_like(g, bg)
    for x, stop in _runs(np.any(outside_mask, axis=0)):
        _require(stop <= left - 1 or x >= end + 1, "stock touches chamber")
        strip = g[:, x:stop]
        bar = strip[-1]
        if np.all(bar == bg):
            role = None
        else:
            _require(np.all(bar == bar[0]) and int(bar[0]) in ports, "unknown source bar")
            role = ports.index(int(bar[0]))
            expected_outside[-1, x:stop] = bar
        payload = strip[:-2]
        values = set(map(int, np.unique(payload))) - {bg}
        _require(len(values) == 1, "source stack must be monochrome")
        color = next(iter(values))
        _require(color not in (frame, control), "source payload uses structural color")
        _require(np.all(strip[-2] == bg), "missing gap over source bars")
        s = _token_stack(payload != bg)
        _require(s['x'] == 0 and s['mask'].shape[1] == stop - x, "bar and token span disagree")
        _require(s['ys'][-1] + s['mask'].shape[0] == h - 2 and s['ys'][0] >= 1,
                 "source stack baseline or top margin invalid")
        s.update(x=x, color=color, role=role)
        stacks.append(s)
        for y in s['ys']:
            block = expected_outside[y:y + s['mask'].shape[0], x:stop]
            block[s['mask']] = color
    actual = g.copy()
    actual[:, left:end] = bg
    _require(np.array_equal(actual, expected_outside), "extra cells outside chamber")
    _require(all(any(s['role'] == role for s in stacks) for role in (0, 1)), "both ink supplies must be present")
    totals = tuple(sum(s['count'] for s in stacks if s['role'] == role) for role in (0, 1))
    _require(seed['ys'][0] - max(totals) * seed['pitch'] >= 0, "growth would clip the top")
    return dict(grid=g, left=left, end=end, background=bg, frame=frame, ports=ports,
                control=control, cut=cut, marker=marker, regions=regions, side=side,
                seed=seed, seed_color=seed_color, stacks=stacks, totals=totals, path=path)


def _answer(info, totals=None, side=None, seam="first"):
    """Pixelwise prediction from the parsed input, also used for explicit rivals."""
    g = info['grid']
    out = g.copy()
    bg, control = info['background'], info['control']
    panel = out[1:-1, info['left'] + 1:info['end'] - 1]
    panel[info['cut']] = bg
    panel[info['marker']] = bg
    for s in info['stacks']:
        if s['role'] is not None:
            out[-1, s['x']:s['x'] + s['mask'].shape[1]] = bg
            for y in s['ys']:
                block = out[y:y + s['mask'].shape[0], s['x']:s['x'] + s['mask'].shape[1]]
                block[s['mask']] = control
    out[-1, info['left'] + 1] = info['frame']
    out[-1, info['end'] - 2] = info['frame']
    ta, tb = info['totals'] if totals is None else tuple(totals)
    side = info['side'] if side is None else side
    seed = info['seed']
    for k in range(1, max(ta, tb) + 1):
        y = seed['ys'][0] - k * seed['pitch']
        _require(y >= 0, "counterfactual or actual growth clips")
        for dy, dx in np.argwhere(seed['mask']):
            r, c = y + int(dy), seed['x'] + int(dx)
            a, b = k <= ta, k <= tb
            if info['cut'][r, c]:
                value = (info['ports'][0] if a else info['ports'][1] if b else bg)
                if seam == "second":
                    value = info['ports'][1] if b else info['ports'][0] if a else bg
                elif seam == "blank":
                    value = bg
            elif info['regions'][side][r, c]:
                value = info['ports'][0] if a else bg
            else:
                value = info['ports'][1] if b else bg
            panel[r, c] = value
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_decode(input_grid))


def _mask(rng, h, w):
    """Connected silhouettes, without adding random pixel noise to clean shapes."""
    style = int(rng.integers(6))
    m = np.ones((h, w), dtype=bool)
    if style == 1 and h >= 3 and w >= 3:
        m[1:-1, 1:-1] = False
    elif style == 2:
        m[:] = False
        m[-1] = True
        m[:, 0] = True
    elif style == 3:
        m[:] = False
        m[h // 2] = True
        m[:, w // 2] = True
    elif style == 4:
        for y in range(h - 1):
            m[y, int(rng.integers(1, w + 1)):] = False
    elif style == 5 and h >= 3 and w >= 3:
        m[0, [0, -1]] = False
        m[-1, [0, -1]] = False
    if rng.integers(2):
        m = m[:, ::-1]
    if rng.integers(2):
        m = m[::-1]
    return m.copy()


def _partition(rng, total, n, unequal=False):
    if n == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), n - 1, replace=False).tolist())
    parts = np.diff([0] + cuts + [total]).tolist()
    if unequal and len(set(parts)) == 1 and total > n:
        parts[0] -= 1
        parts[-1] += 1
    rng.shuffle(parts)
    return parts


def _curtain(rng, hi, wi, top_seed, x, token_width, changes):
    end_y = top_seed - 1
    _require(end_y >= 2 * changes + 2, "not enough room for curtain turns")
    candidates = list(range(2, end_y - 1, 2))
    rows = sorted(int(v) for v in rng.choice(candidates, changes, replace=False)) if changes else []
    cols = list(range(x + 1, x + token_width - 1))
    column = int(rng.choice(cols))
    cut = np.zeros((hi, wi), dtype=bool)
    last = 0
    for row in rows:
        cut[last:row + 1, column] = True
        nxt = int(rng.choice([c for c in cols if c != column]))
        cut[row, min(column, nxt):max(column, nxt) + 1] = True
        column, last = nxt, row
    cut[last:end_y + 1, column] = True
    if rng.integers(2):
        cut[end_y, :column + 1] = True
    else:
        cut[end_y, column:] = True
    _path(cut)
    return cut


def _stamp(a, y, x, mask, color):
    h, w = mask.shape
    _require(0 <= y and y + h <= a.shape[0] and 0 <= x and x + w <= a.shape[1], "clipped stamp")
    a[y:y + h, x:x + w][mask] = color


def _make_pair(rng, side, changes, *, anchor=False, dest_count=None, avoid_totals=None, palette=None):
    bg, frame, control, ink_a, ink_b, seed_color = (
        tuple(int(v) for v in rng.choice(10, 6, replace=False)) if palette is None else palette)
    dh, dw = int(rng.integers(2, 4)), int(rng.integers(4, 7))
    pitch = dh + 1
    d = int(rng.integers(1, 4)) if dest_count is None else dest_count
    max_total = min(6, 28 // pitch - d)
    informative = anchor or avoid_totals is not None
    choices = [(a, b) for a in range(1, max_total + 1) for b in range(1, max_total + 1)
               if (not informative or (a != b and max(a, b) >= 3)) and
               (avoid_totals is None or (a != avoid_totals[0] and b != avoid_totals[1]
                                        and (a - b) * (avoid_totals[0] - avoid_totals[1]) < 0))]
    ta, tb = choices[int(rng.integers(len(choices)))]
    n_a = int(rng.integers(1, min(3, ta) + 1))
    n_b = int(rng.integers(1, min(3, tb, 5 - n_a) + 1))
    if anchor:
        if ta >= 3 and (tb < 3 or rng.integers(2)):
            n_a, n_b = 2, int(rng.integers(1, min(2, tb) + 1))
        else:
            n_b, n_a = 2, int(rng.integers(1, min(2, ta) + 1))
    stocks = [(role, count) for role, total, n in ((0, ta, n_a), (1, tb, n_b))
              for count in _partition(rng, total, n, unequal=anchor)]
    unmarked = int(rng.integers(1 if anchor else 0, min(2, 6 - len(stocks)) + 1))
    stocks += [(None, int(rng.integers(1, 5))) for _ in range(unmarked)]
    rng.shuffle(stocks)
    wi = dw + int(rng.integers(2, 5))
    chamber_width = wi + 2
    widths = [int(rng.integers(1, 4)) for _ in stocks]
    while chamber_width + sum(widths) + len(stocks) > 30:
        i = max(range(len(widths)), key=lambda j: widths[j])
        _require(widths[i] > 1, "cannot fit stock lanes")
        widths[i] -= 1
    masks = [_mask(rng, int(rng.integers(1, 4)), w) for w in widths]
    required_height = max(10, d * pitch + 2 * changes + 5,
                          (d + max(ta, tb)) * pitch + 2,
                          max(count * (m.shape[0] + 1) + 2 for (_, count), m in zip(stocks, masks)))
    _require(required_height <= 30, "too many source tokens")
    height = int(rng.integers(required_height, 31))
    min_width = chamber_width + sum(widths) + len(stocks)
    width = int(rng.integers(min_width, min(30, min_width + 5) + 1))
    # Treat the chamber and each stock lane as one unit, then distribute spare margins.
    chamber_index = int(rng.integers(len(stocks) + 1))
    units = list(range(len(stocks)))
    units.insert(chamber_index, -1)
    spare = rng.multinomial(width - min_width, np.full(len(units) + 1, 1 / (len(units) + 1)))
    positions, xx = {}, int(spare[0])
    for j, unit in enumerate(units):
        positions[unit] = xx
        xx += chamber_width if unit == -1 else widths[unit]
        if j < len(units) - 1:
            xx += 1 + int(spare[j + 1])
    left = positions[-1]
    end = left + chamber_width
    a = np.full((height, width), bg, dtype=np.int8)
    a[[0, -1], left:end] = frame
    a[:, [left, end - 1]] = frame
    a[-1, left + 1], a[-1, end - 2] = ink_a, ink_b
    hi = height - 2
    chamber = a[1:-1, left + 1:end - 1]
    receiver_mask = _mask(rng, dh, dw)
    receiver_x = int(rng.integers(1, wi - dw))
    bottom = hi - 1 - dh
    for k in range(d):
        _stamp(chamber, bottom - k * pitch, receiver_x, receiver_mask, seed_color)
    seed_top = bottom - (d - 1) * pitch
    cut = _curtain(rng, hi, wi, seed_top, receiver_x, dw, changes)
    regions = _regions(cut)
    sites = _sites(cut, regions[side], chamber != bg)
    _require(bool(sites), "no marker site")
    marker = sites[int(rng.integers(len(sites)))]
    chamber[cut] = control
    chamber[marker] = control
    payload_pool = [c for c in range(10) if c not in (bg, frame, control)]
    payloads = rng.choice(payload_pool, len(stocks), replace=True).tolist()
    if anchor:
        source_i = next(i for i, (r, n) in enumerate(stocks) if r is not None)
        unmarked_i = next(i for i, (r, n) in enumerate(stocks) if r is None)
        payloads[unmarked_i] = payloads[source_i]
    output = a.copy()
    out_chamber = output[1:-1, left + 1:end - 1]
    out_chamber[cut] = bg
    out_chamber[marker] = bg
    output[-1, left + 1] = output[-1, end - 2] = frame
    for i, ((role, count), m, color) in enumerate(zip(stocks, masks, payloads)):
        x = positions[i]
        for k in range(count):
            y = height - 2 - m.shape[0] - k * (m.shape[0] + 1)
            _stamp(a, y, x, m, int(color))
            _stamp(output, y, x, m, control if role is not None else int(color))
        if role is not None:
            a[-1, x:x + m.shape[1]] = (ink_a, ink_b)[role]
    # Ground truth uses two explicit local growth layers, not parsed source metadata.
    layers = []
    for total, ink in ((ta, ink_a), (tb, ink_b)):
        layer = np.full((hi, wi), bg, dtype=np.int8)
        for k in range(1, total + 1):
            _stamp(layer, seed_top - k * pitch, receiver_x, receiver_mask, ink)
        layers.append(layer)
    aa, bb = layers
    support = (aa != bg) | (bb != bg)
    composite = np.where(regions[side], aa, bb)
    composite[cut] = np.where(aa[cut] != bg, aa[cut], bb[cut])
    out_chamber[support] = composite[support]
    return {'input': a, 'output': output}


def _key(grid):
    a = np.ascontiguousarray(grid, dtype=np.int8)
    return a.shape, hashlib.sha256(a.tobytes()).digest()


def _evidence(info, output):
    a, b = info['totals']
    panel = np.asarray(output)[1:-1, info['left'] + 1:info['end'] - 1]
    has_both = all(np.any((panel == info['ports'][role]) & info['regions'][region])
                   for role, region in ((0, info['side']), (1, 1 - info['side'])))
    # Each last added level must have a real visible effect; no hidden/saturated operand.
    sensitive = all(not np.array_equal(output, _answer(info, (a - (role == 0), b - (role == 1))))
                    for role in (0, 1))
    marker = not np.array_equal(output, _answer(info, side=1 - info['side']))
    priority = not np.array_equal(output, _answer(info, seam="second"))
    extra_seam, suppressed = a == b, a == b
    seed = info['seed']
    for k in range(min(a, b) + 1, max(a, b) + 1):
        y = seed['ys'][0] - k * seed['pitch']
        for dy, dx in np.argwhere(seed['mask']):
            r, c = y + int(dy), seed['x'] + int(dx)
            extra_seam |= bool(info['cut'][r, c])
            shorter_region = info['side'] if a < b else 1 - info['side']
            suppressed |= bool(info['regions'][shorter_region][r, c])
    return has_both and sensitive and marker and priority and extra_seam and suppressed


def _rival_totals(info, name):
    a, b = info['totals']
    d = info['seed']['count']
    pools = [[s['count'] for s in info['stacks'] if s['role'] == role] for role in (0, 1)]
    if name == 'bars':
        return tuple(len(p) for p in pools)
    if name == 'tallest':
        return tuple(max(p) for p in pools)
    if name == 'double_receiver':
        return d, d
    if name == 'replace_receiver':
        return max(0, a - d), max(0, b - d)
    if name == 'swap_inks':
        return b, a
    raise ValueError(name)


_RIVALS = ('bars', 'tallest', 'double_receiver', 'replace_receiver', 'swap_inks')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(v) for v in rng.choice(10, 6, replace=False))
    fixed_palette = bool(rng.integers(2))
    first_side = int(rng.integers(2))
    first_d = int(rng.integers(1, 4))
    second_d = int(rng.choice([d for d in (1, 2, 3) if d != first_d]))
    puzzle, seen, first_totals = {'train': [], 'test': []}, set(), None
    for index in range(nt + nq):
        training = index < nt
        side = (first_side if index == 0 else 1 - first_side if index == 1 else int(rng.integers(2)))
        for attempt in range(160):
            changes = int(rng.integers(0, 2)) if training else int(rng.integers(2, 4))
            try:
                pair = _make_pair(rng, side, changes, anchor=index == 0,
                                  dest_count=first_d if index == 0 else second_d if index == 1 else None,
                                  avoid_totals=first_totals if index == 1 else None,
                                  palette=palette if fixed_palette else None)
                info = _decode(pair['input'])
                if _key(pair['input']) in seen or not _evidence(info, pair['output']):
                    continue
                # A physically oversized rival already differs from the in-bounds answer.
                if index == 0 and any(np.array_equal(_answer(info, _rival_totals(info, r)), pair['output'])
                                      for r in ('bars', 'tallest', 'replace_receiver')):
                    continue
                break
            except AssertionError:
                continue
        else:
            raise RuntimeError("bounded construction exhausted")
        if index == 0:
            first_totals = info['totals']
        seen.add(_key(pair['input']))
        puzzle['train' if training else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_info = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape, "output shape changed")
            key = _key(g)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(g)
            _require(np.array_equal(out, _answer(info)), "wrong growth, compositing, processing, or preservation")
            _require(_evidence(info, out), "an ink count, marker, or seam priority is ineffective")
            if split == 'train':
                train_info.append((info, out))
    _require({i['side'] for i, _ in train_info} == {0, 1}, "training must select both regions")
    _require(len({i['seed']['count'] for i, _ in train_info}) > 1, "receiver counts do not vary")
    _require(all(len({i['totals'][role] for i, _ in train_info}) > 1 for role in (0, 1)),
             "both supply totals must vary in training")
    _require({np.sign(i['totals'][0] - i['totals'][1]) for i, _ in train_info} >= {-1, 1},
             "training must show either ink taller, including both single-ink seam cases")
    _require(any(any(len({s['count'] for s in i['stacks'] if s['role'] == role}) > 1
                     for role in (0, 1)) for i, _ in train_info), "missing unequal multi-stack addition")
    _require(any(any(s['role'] is None and any(t['role'] is not None and s['color'] == t['color']
                                              for t in i['stacks']) for s in i['stacks'])
                 for i, _ in train_info), "missing same-color ignored stock")
    for rival in _RIVALS:
        distinguished = False
        for info, out in train_info:
            try:
                distinguished |= not np.array_equal(_answer(info, _rival_totals(info, rival)), out)
            except AssertionError:
                distinguished = True
        _require(distinguished, "training fits arithmetic shortcut: " + rival)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, key = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(key)
    return digest.digest()


def self_test():
    """Streaming 42-seed checks; retains hashes and small feature sets, no files."""
    shapes, receiver_masks, donor_masks, cuts, palettes, signatures = (set() for _ in range(6))
    source_numbers, dest_numbers, totals, chamber_positions = (set() for _ in range(4))
    seed_zero = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if seed == 0:
            seed_zero = _fingerprint(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve failed")
                shapes.add(pair['input'].shape)
                token = info['seed']['mask']
                receiver_masks.add((token.shape, token.tobytes()))
                cuts.add((info['cut'].shape, info['cut'].tobytes()))
                palettes.add((info['background'], info['frame'], info['control'], *info['ports'], info['seed_color']))
                source_numbers.add(tuple(sum(s['role'] == role for s in info['stacks']) for role in (0, 1)))
                dest_numbers.add(info['seed']['count'])
                totals.add(info['totals'])
                chamber_positions.add(('left' if info['left'] == 0 else 'right' if info['end'] == pair['input'].shape[1]
                                       else 'inset', any(s['x'] < info['left'] for s in info['stacks']),
                                       any(s['x'] > info['end'] for s in info['stacks'])))
                for s in info['stacks']:
                    donor_masks.add((s['mask'].shape, s['mask'].tobytes()))
                # Moving only the marker exchanges the two regions, never ink priority.
                moved = pair['input'].copy()
                p = moved[1:-1, info['left'] + 1:info['end'] - 1]
                p[info['marker']] = info['background']
                sites = _sites(info['cut'], info['regions'][1 - info['side']], p == info['seed_color'])
                p[sites[0]] = info['control']
                _require(np.array_equal(solve(moved), _answer(info, side=1 - info['side'])), "marker swap failed")
                # Arbitrary bijective recoloring must preserve all semantics.
                lut = np.array([9, 3, 5, 0, 8, 1, 6, 2, 4, 7], dtype=np.int8)
                _require(np.array_equal(solve(lut[pair['input']]), lut[pair['output']]), "palette equivariance failed")
    _require(len(signatures) == 42 and len(shapes) >= 25 and len(cuts) >= 150, "structural diversity collapsed")
    _require(len(receiver_masks) >= 20 and len(donor_masks) >= 20, "silhouette diversity collapsed")
    _require(len(palettes) >= 30 and len(source_numbers) >= 6 and len(totals) >= 15, "role or count diversity collapsed")
    _require(dest_numbers == {1, 2, 3} and len(chamber_positions) >= 3, "placement diversity collapsed")
    _require(_fingerprint(generate_puzzle(0)) == seed_zero, "determinism failed")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    base = generate_puzzle(0, 2, 1)
    info = _decode(base['train'][0]['input'])
    for defect in ('seed', 'stock', 'port', 'seam'):
        broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in base[s]] for s in base}
        out = broken['train'][0]['output']
        if defect == 'seed':
            y = info['seed']['ys'][-1] + 1
            x = info['left'] + 1 + info['seed']['x']
        elif defect == 'stock':
            s = next(s for s in info['stacks'] if s['role'] is not None)
            dy, dx = np.argwhere(s['mask'])[0]
            y, x = s['ys'][0] + int(dy), s['x'] + int(dx)
        elif defect == 'port':
            y, x = out.shape[0] - 1, info['left'] + 1
        else:
            alternative = _answer(info, seam='second')
            y, x = map(int, np.argwhere(alternative != out)[0])
        out[y, x] = (int(out[y, x]) + 1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted " + defect + " accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "receiver_masks": len(receiver_masks),
            "donor_masks": len(donor_masks), "curtains": len(cuts), "palettes": len(palettes),
            "source_count_vectors": len(source_numbers), "supply_total_vectors": len(totals)}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
