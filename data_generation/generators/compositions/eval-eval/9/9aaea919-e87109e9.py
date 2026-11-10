"""Ray-reachable counter towers. Only NumPy and the standard library are used.

No parent code, fixtures, seed state, or training examples are used by solve.
The header's side bars encode turns; its centered bar encodes highlighting.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid grid color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _runs(a):
    changes = np.diff(np.r_[False, a, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _components(a, bg):
    seen = set()
    result = []
    h, w = a.shape
    for rr, cc in np.argwhere(a != bg):
        r, c = int(rr), int(cc)
        if (r, c) in seen:
            continue
        color = int(a[r, c])
        pending, cells = [(r, c)], []
        seen.add((r, c))
        while pending:
            x, y = pending.pop()
            cells.append((x, y))
            for dx, dy in DIRS:
                p = (x + dx, y + dy)
                if (0 <= p[0] < h and 0 <= p[1] < w and p not in seen
                        and a[p] == color):
                    seen.add(p)
                    pending.append(p)
        r0, c0 = np.min(cells, axis=0)
        r1, c1 = np.max(cells, axis=0) + 1
        mask = a[r0:r1, c0:c1] == color
        result.append(dict(color=color, r=int(r0), c=int(c0), mask=mask,
                           cells=tuple(cells)))
    return result


def _header(a):
    frame = int(a[0, 0])
    _require(np.all(a[0] == frame), "missing header top")
    bottoms = np.flatnonzero(np.all(a[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing header bottom")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(a) - 1, "invalid header height")
    interior = a[1:bottom]
    _require(np.all(interior == interior[0]), "header rows disagree")
    row = interior[0]
    _require(row[0] == row[-1] == frame, "missing header walls")
    spans = _runs(row != frame)
    _require(3 <= len(spans) <= 5, "need two to four turn panels and a swatch")
    # Every interior has one colored vertical stripe and at least two blanks.
    blanks = Counter(int(v) for x, y in spans for v in row[x:y])
    bg = max(blanks, key=blanks.get)
    turns, highlights, widths = {}, [], []
    for x, y in spans:
        values = row[x:y]
        _require(len(values) >= 3, "panel is too narrow")
        ink = np.flatnonzero(values != bg)
        _require(len(ink) == 1, "panel must contain one stripe")
        position = int(ink[0])
        color = int(values[position])
        _require(color != frame, "stripe aliases frame")
        if position == 0 or position == len(values) - 1:
            _require(color not in turns, "duplicate turn instruction")
            turns[color] = -1 if position == 0 else 1
        else:
            _require(len(values) % 2 == 1 and position == len(values) // 2,
                     "highlight stripe must be centered")
            highlights.append(color)
        widths.append(int(y - x))
    _require(len(highlights) == 1 and 2 <= len(turns) <= 4, "ambiguous legend")
    highlight = highlights[0]
    _require(highlight not in turns and highlight != bg, "conflicting highlight")
    return bottom + 1, bg, frame, highlight, turns, tuple(widths)


def _decode(value):
    a = _grid(value)
    crop, bg, frame, highlight, turns, widths = _header(a)
    field = a[crop:]
    _require(field[0, 0] == bg, "field corner must reveal background")
    components = _components(field, bg)
    bases, tokens = [], []
    for component in components:
        h, w = component['mask'].shape
        if h == 1 and w >= 3 and w % 2 == 1 and component['mask'].all():
            bases.append(component)
        else:
            _require(h in (2, 3) and w in (3, 5), "invalid token silhouette")
            tokens.append(component)
    destination = [i for i, b in enumerate(bases) if b['color'] not in turns]
    _require(len(destination) == 1 and len(bases) >= 2, "receiver base is not unique")
    dest = destination[0]
    _require(highlight not in np.unique(field), "highlight must visibly change counters")
    token_lookup = {(t['r'], t['c'] + t['mask'].shape[1] // 2): i
                    for i, t in enumerate(tokens)}
    used = set()
    stations = []
    common_mask = None
    for i, base in enumerate(bases):
        br, bc = base['r'], base['c']
        bw = base['mask'].shape[1]
        center = bc + bw // 2
        bottom_candidates = [j for j, t in enumerate(tokens)
                             if t['r'] + t['mask'].shape[0] == br - 1
                             and t['c'] + t['mask'].shape[1] // 2 == center]
        _require(len(bottom_candidates) == 1, "base needs one centered bottom token")
        t = tokens[bottom_candidates[0]]
        mask = t['mask']
        th, tw = mask.shape
        _require(bw >= tw and (i != dest or bw == tw), "invalid base width")
        if common_mask is None:
            common_mask = mask
        _require(np.array_equal(common_mask, mask), "tokens must share a local mask")
        pitch = th + 1
        y = br - pitch
        indices = []
        while (y, center) in token_lookup:
            j = token_lookup[y, center]
            candidate = tokens[j]
            _require(j not in used and candidate['color'] == t['color']
                     and np.array_equal(candidate['mask'], mask), "malformed stack")
            indices.append(j)
            used.add(j)
            y -= pitch
        _require(indices, "empty tower")
        cells = tuple(cell for j in indices for cell in tokens[j]['cells'])
        stations.append(dict(r=br, c=bc, width=bw, center=center,
                             color=base['color'], payload=t['color'], count=len(indices),
                             cells=cells, ys=tuple(tokens[j]['r'] for j in indices)))
    _require(len(used) == len(tokens), "orphan tokens")
    port_map = np.full(field.shape, -1, dtype=int)
    for i, s in enumerate(stations):
        if i != dest:
            _require(s['color'] in turns, "unkeyed source")
            port_map[s['r'], s['c']:s['c'] + s['width']] = i
    return dict(field=field, crop=crop, bg=bg, frame=frame, highlight=highlight,
                turns=turns, panel_widths=widths, stations=stations, dest=dest,
                mask=common_mask, pitch=common_mask.shape[0] + 1, ports=port_map)


def _trace(q, override=None):
    """Step rays through the immutable port map; tokens are transparent.

    A collision turns in the last free cell, never enters or paints the base.
    Separate (position, heading) state sets detect cycles on each branch.
    """
    ports = q['ports']
    h, w = ports.shape
    d = q['stations'][q['dest']]
    start = (d['r'], d['center'])
    turns = q['turns'] if override is None else override
    visits, segments, paths, chains = [], [], [], []
    for branch in range(4):
        r, c = start
        direction = branch
        seen, path, chain = set(), {start}, []
        leg_r, leg_c, distance = r, c, 0
        for _ in range(4 * h * w + 1):
            state = (r, c, direction)
            _require(state not in seen, "cyclic ray")
            seen.add(state)
            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                segments.append((branch, leg_r, leg_c, direction, distance, -1))
                break
            hit = int(ports[nr, nc])
            if hit >= 0:
                segments.append((branch, leg_r, leg_c, direction, distance, hit))
                visits.append((branch, hit, direction, r, c))
                chain.append(hit)
                direction = (direction + turns[q['stations'][hit]['color']]) % 4
                leg_r, leg_c, distance = r, c, 0
            else:
                r, c = nr, nc
                path.add((r, c))
                distance += 1
        else:
            raise AssertionError("ray step bound exceeded")
        paths.append(path)
        chains.append(tuple(chain))
    return set(v[1] for v in visits), visits, segments, paths, tuple(chains)


def _render(q, reached):
    result = q['field'].copy()
    for i, s in enumerate(q['stations']):
        result[s['r'], s['c']:s['c'] + s['width']] = q['bg']
        if i in reached:
            for cell in s['cells']:
                result[cell] = q['highlight']
    d = q['stations'][q['dest']]
    total = sum(q['stations'][i]['count'] for i in reached)
    h, w = q['mask'].shape
    x = d['center'] - w // 2
    for k in range(d['count'], d['count'] + total):
        y = d['r'] - (k + 1) * q['pitch']
        _require(y >= 0, "receiver growth crosses top edge")
        block = result[y:y + h, x:x + w]
        original = q['field'][y:y + h, x:x + w]
        _require(block.shape == q['mask'].shape and np.all(block[q['mask']] == q['bg'])
                 and np.all(original[q['mask']] == q['bg']),
                 "receiver growth collides with another tower")
        block[q['mask']] = d['payload']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone."""
    q = _decode(input_grid)
    return _render(q, _trace(q)[0])


def _sample_mask(rng, height=2, width=None):
    w = int(rng.choice([3, 5])) if width is None else width
    # Full-span connected silhouettes, not just variable-colored rectangles.
    mask = np.zeros((height, w), dtype=bool)
    if height == 2:
        mask[1] = True
        left, right = sorted(map(int, rng.integers(0, w, size=2)))
        mask[0, left:right + 1] = True
    else:
        mask[1] = True
        for y in (0, 2):
            left, right = sorted(map(int, rng.integers(0, w, size=2)))
            mask[y, left:right + 1] = True
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _station_cells(r, center, width, count, mask):
    th, tw = mask.shape
    pitch = th + 1
    base = [(r, c) for c in range(center - width // 2, center + width // 2 + 1)]
    tokens = [(r - (k + 1) * pitch + int(y), center - tw // 2 + int(x))
              for k in range(count) for y, x in np.argwhere(mask)]
    return base, tokens


def _add_station(scene, r, center, width, count, color, payload):
    base, tokens = _station_cells(r, center, width, count, scene['mask'])
    cells = base + tokens
    h, w = scene['field'].shape
    if any(not (1 <= x < h - 1 and 1 <= y < w - 1) for x, y in cells):
        return False
    # A blank cell around occupied objects keeps all components unambiguous.
    occupied = scene['field'] != scene['bg']
    protected = scene['reserved']
    for x, y in cells:
        if protected[x, y] or np.any(occupied[x - 1:x + 2, y - 1:y + 2]):
            return False
    field = scene['field']
    for p in base:
        field[p] = color
    for p in tokens:
        field[p] = payload
    scene['stations'].append(dict(r=r, c=center-width//2, width=width, center=center,
                                 color=color, payload=payload, count=count,
                                 cells=tuple(tokens), ys=tuple(r-(k+1)*scene['pitch']
                                                            for k in range(count))))
    i = len(scene['stations']) - 1
    for p in base:
        scene['ports'][p] = i
    return True


def _fresh_scene(rng, height, width, mask, dcount, budget, dr=None, dc=None, key_color=None):
    palette = list(map(int, rng.permutation(10)))
    if key_color is not None:
        j = palette.index(key_color)
        palette[4], palette[j] = palette[j], palette[4]
    bg, frame, highlight, dest_color = palette[:4]
    nk = int(rng.integers(2, min(4, width // 5 - 1) + 1))
    keys = palette[4:4+nk]
    turns = {c: int(rng.choice([-1, 1])) for c in keys}
    pitch, tw = mask.shape[0] + 1, mask.shape[1]
    minimum_r = (dcount + budget) * pitch + 1
    if minimum_r > height - 3:
        return None
    r = int(rng.integers(minimum_r, height - 2)) if dr is None else dr
    c = int(rng.integers(tw//2 + 4, width - tw//2 - 4)) if dc is None else dc
    if r < minimum_r:
        return None
    field = np.full((height, width), bg, dtype=np.int8)
    scene = dict(field=field, bg=bg, frame=frame, highlight=highlight, turns=turns,
                 keys=keys, dest=0, mask=mask, pitch=pitch,
                 stations=[], ports=np.full(field.shape, -1, dtype=int),
                 reserved=np.zeros(field.shape, dtype=bool), budget=budget)
    base, tokens = _station_cells(r, c, tw, dcount, mask)
    payload = int(rng.choice([v for v in range(10) if v not in (bg, highlight)]))
    for p in base:
        field[p] = dest_color
    for p in tokens:
        field[p] = payload
    scene['stations'].append(dict(r=r, c=c-tw//2, width=tw, center=c, color=dest_color,
                                  payload=payload, count=dcount, cells=tuple(tokens),
                                  ys=tuple(r-(k+1)*pitch for k in range(dcount))))
    # Protect the complete possible receiver growth and its separation margin.
    top = r - (dcount + budget) * pitch
    scene['reserved'][max(0, top-1):r+2, c-tw//2-1:c+tw//2+2] = True
    return scene


def _copy_scene(q):
    return {k: (v.copy() if isinstance(v, (np.ndarray, dict, list)) else v)
            for k, v in q.items()}


def _grow(rng, scene, target, attempts=140, require_new_hit=True):
    for _ in range(attempts):
        current = _trace(scene)
        if len(current[0]) >= target:
            break
        legs = [s for s in current[2] if s[4] >= 2]
        if not legs:
            break
        # Favor a continuation after a turn, while retaining direct branches.
        indirect = [s for s in legs if s[3] != s[0]]
        if indirect and rng.random() < .65:
            legs = indirect
        _, r, c, direction, distance, _ = legs[int(rng.integers(len(legs)))]
        step = int(rng.integers(1, distance + 1))
        dx, dy = DIRS[direction]
        br, hit_col = r + dx * step, c + dy * step
        tw = scene['mask'].shape[1]
        width = int(rng.choice(list(range(tw, min(17, scene['field'].shape[1]-2)+1, 2))))
        if direction == 1:
            center = hit_col + width // 2
        elif direction == 3:
            center = hit_col - width // 2
        else:
            center = hit_col + int(rng.integers(-width//2 + 1, width//2 + 1))
        used_budget = sum(scene['stations'][i]['count'] for i in current[0])
        available = scene['budget'] - used_budget - (target-len(current[0])-1)
        if available < 1:
            break
        count = int(rng.integers(1, min(5, available)+1))
        color = int(rng.choice(scene['keys']))
        payload = int(rng.choice([v for v in range(10) if v not in (scene['bg'], scene['highlight'])]))
        trial = _copy_scene(scene)
        if not _add_station(trial, br, center, width, count, color, payload):
            continue
        try:
            traced = _trace(trial)
        except AssertionError:
            continue
        total = sum(trial['stations'][i]['count'] for i in traced[0])
        if (total > scene['budget'] - max(0, target-len(traced[0]))
                or (require_new_hit and len(traced[0]) <= len(current[0]))):
            continue
        scene = trial
    return scene


def _add_unreachable(rng, scene, count=1, match_payload=False):
    old_hits = _trace(scene)[0]
    added = 0
    h, w = scene['field'].shape
    tw = scene['mask'].shape[1]
    for _ in range(250):
        if added >= count:
            break
        width = int(rng.choice(list(range(tw, min(15, w-2)+1, 2))))
        nc = int(rng.integers(1, min(4, (h-3)//scene['pitch'])+1))
        r = int(rng.integers(nc * scene['pitch'] + 1, h - 1))
        c = int(rng.integers(width//2 + 1, w-width//2-1))
        trial = _copy_scene(scene)
        color = int(rng.choice(scene['keys']))
        # Often reuse a reached source's payload to disallow global recoloring.
        payload = (scene['stations'][int(rng.choice(sorted(old_hits)))]['payload']
                   if old_hits and (match_payload or rng.random() < .65) else
                   int(rng.choice([v for v in range(10) if v not in (scene['bg'], scene['highlight'])])))
        if not _add_station(trial, r, c, width, nc, color, payload):
            continue
        try:
            if _trace(trial)[0] != old_hits:
                continue
        except AssertionError:
            continue
        scene = trial
        added += 1
    return scene


def _make_header(rng, scene):
    w = scene['field'].shape[1]
    entries = [(c, side) for c, side in scene['turns'].items()] + [(scene['highlight'], 0)]
    rng.shuffle(entries)
    # Each panel occupies five cells initially. Extra width goes to side panels
    # freely, and to the center panel in pairs so its center remains integral.
    panel_widths = [5] * len(entries)
    extra = w - 5 * len(entries)
    while extra:
        choices = [i for i, (_, side) in enumerate(entries) if side or extra >= 2]
        i = int(rng.choice(choices))
        take = 1 if entries[i][1] else 2
        panel_widths[i] += take
        extra -= take
    hh = int(rng.integers(3, min(5, 30-len(scene['field'])) + 1))
    header = np.full((hh, w), scene['bg'], dtype=np.int8)
    header[[0, -1]] = scene['frame']
    x = 0
    for (color, side), size in zip(entries, panel_widths):
        header[:, x] = header[:, x+size-1] = scene['frame']
        position = x+1 if side == -1 else x+size-2 if side == 1 else x+size//2
        header[1:-1, position] = color
        x += size
    return header


def _pair(rng, scene):
    reached = _trace(scene)[0]
    inp = np.vstack((_make_header(rng, scene), scene['field']))
    return {'input': inp, 'output': _render(scene, reached)}


def _fork_lesson(rng):
    """A variable-width lower collector is reached by three independent rays.

    The two side towers turn toward a shared lower tower. This is a guaranteed
    object-identity witness, with sampled counts, shape, spacings and positions.
    """
    for _ in range(160):
        mask = _sample_mask(rng)
        th, tw = mask.shape
        h, w = int(rng.integers(24, 28)), int(rng.integers(23, 31))
        dcount = int(rng.integers(1, 3))
        a, b = map(int, rng.choice([1, 2, 3], size=2))
        if a == b:
            continue
        lower = 1
        total = a + b + lower
        rmin = (dcount + total) * 3 + 1
        rmax = h - 7
        if rmin > rmax:
            continue
        r = int(rng.integers(rmin, rmax + 1))
        c = w // 2 + int(rng.integers(-1, 2))
        q = _fresh_scene(rng, h, w, mask, dcount, total, r, c)
        if q is None:
            continue
        left, right = q['keys'][:2]
        q['turns'][left], q['turns'][right] = -1, 1
        gaps = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
        left_end = c - tw//2 - gaps[0] - 1
        right_start = c + tw//2 + gaps[1] + 1
        lw, rw = tw + 2*int(rng.integers(0, 2)), tw + 2*int(rng.integers(0, 2))
        pool = [v for v in range(10) if v not in (q['bg'], q['highlight'])]
        if not _add_station(q, r, left_end-lw//2, lw, a, left, right):
            continue
        if not _add_station(q, r, right_start+rw//2, rw, b, right, int(rng.choice(pool))):
            continue
        # Wide enough to intercept the left, central, and right descending rays.
        span = max(c-(left_end+1), right_start-1-c)
        bw = 2*(span+int(rng.integers(0, 2)))+1
        lower_r = r + int(rng.integers(5, h-r-1))
        if not _add_station(q, lower_r, c, bw, lower, int(rng.choice([left, right])), int(rng.choice(pool))):
            continue
        q = _add_unreachable(rng, q, int(rng.integers(1, 3)), match_payload=True)
        reached, visits, _, _, chains = _trace(q)
        if (len(reached) == 3 and len(q['stations']) >= 5
                and len(visits) > len(reached) and max(map(len, chains)) >= 2
                and sum(q['stations'][i]['width']//tw for i in reached) != total):
            return _pair(rng, q)
    raise AssertionError("could not construct the collection lesson")


def _random_pair(rng, height_mask=2, dcount=None, target=None, min_turns=0, max_turns=None):
    for _ in range(100):
        mask = _sample_mask(rng, height_mask)
        pitch = height_mask + 1
        large = target is not None and target >= 3
        h, w = int(rng.integers(22 if large else 16, 28)), int(rng.integers(20 if large else 15, 31))
        d = int(rng.integers(1, 4)) if dcount is None else dcount
        capacity = (h - 4) // pitch
        budget = min(capacity-d, int(rng.integers(3, 7)))
        if budget < 1:
            continue
        goal = int(rng.integers(1, min(4, budget)+1)) if target is None else target
        if budget < goal:
            continue
        q = _fresh_scene(rng, h, w, mask, d, budget)
        if q is None:
            continue
        q = _grow(rng, q, goal)
        reached, _, _, _, chains = _trace(q)
        longest = max(map(len, chains))
        if (len(reached) != goal or longest < min_turns
                or (max_turns is not None and longest > max_turns)):
            continue
        unhit_goal = int(rng.integers(0, 3))
        q = _add_unreachable(rng, q, unhit_goal)
        if unhit_goal and len(q['stations']) - 1 == len(reached):
            continue
        return _pair(rng, q)
    raise AssertionError("bounded route construction exhausted")


def _selector_lesson(rng, first):
    """An off-axis chain makes both glyph meanings change the selected set.

    Neither of the two indirect bases lies on an original receiver axis.
    Reflect the spatial construction when the fork needs the other heading.
    Counters stay upright under that reflection.
    """
    d = 1 if first['d'] != 1 else 2
    triples = [(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2, 3)
               if 3 <= a+b+c <= 5 and a+b+c != first['total']
               and d+a+b+c != first['final']
               and first['total']*d != (a+b+c)*first['d']]
    reflect = len(first['headings'][1]) < 2
    desired_old_side = -1 if reflect else 1
    changed_colors = sorted({first['q']['stations'][i]['color'] for i in first['reached']
                             if first['q']['turns'][first['q']['stations'][i]['color']] == desired_old_side})
    for _ in range(180):
        mask = _sample_mask(rng)
        tw = mask.shape[1]
        h, w = int(rng.integers(26, 28)), int(rng.integers(27, 31))
        a, b, c = triples[int(rng.integers(len(triples)))]
        total = a+b+c
        center = int(rng.integers(tw//2+3, tw//2+6))
        q = _fresh_scene(rng, h, w, mask, d, total, dc=center,
                         key_color=int(rng.choice(changed_colors)))
        if q is None:
            continue
        r = q['stations'][0]['r']
        left, right = q['keys'][:2]
        q['turns'][left], q['turns'][right] = -1, 1
        gap = int(rng.integers(3, 5))
        first_start = center + tw//2 + gap + 1
        width1 = tw + 2*int(rng.integers(0, 2))
        pool = [v for v in range(10) if v not in (q['bg'], q['highlight'])]
        if not _add_station(q, r, first_start+width1//2, width1, a, left, int(rng.choice(pool))):
            continue
        second_r = r - a*3 - int(rng.integers(2, 5))
        second_start = first_start - 1
        width2 = tw + 2*int(rng.integers(0, 2))
        if not _add_station(q, second_r, second_start+width2//2, width2, b, right, int(rng.choice(pool))):
            continue
        third_start = second_start+width2+int(rng.integers(1, 4))
        width3 = tw
        if not _add_station(q, second_r+1, third_start+width3//2, width3, c,
                            int(rng.choice([left, right])), int(rng.choice(pool))):
            continue
        if reflect:
            # Horizontal reflection reverses relative turns, but not token pitch.
            q['field'] = q['field'][:, ::-1].copy()
            q['ports'] = q['ports'][:, ::-1].copy()
            q['reserved'] = q['reserved'][:, ::-1].copy()
            q['mask'] = q['mask'][:, ::-1].copy()
            q['turns'] = {color: -side for color, side in q['turns'].items()}
            q['stations'] = [dict(s, c=w-s['c']-s['width'], center=w-1-s['center'],
                                  cells=tuple((x, w-1-y) for x, y in s['cells']))
                             for s in q['stations']]
        q = _add_unreachable(rng, q, int(rng.integers(1, 3)))
        if len(_trace(q)[0]) != 3 or len(q['stations']) < 5:
            continue
        pair = _pair(rng, q)
        try:
            _evidence([first, _features(pair)])
        except AssertionError:
            continue
        return pair
    raise AssertionError("could not construct the turn-selection lesson")


def _features(pair):
    q = _decode(pair['input'])
    reached, visits, segments, paths, chains = _trace(q)
    d = q['stations'][q['dest']]
    values = [q['stations'][i]['count'] for i in sorted(reached)]
    headings = {side: {v[2] for v in visits if q['turns'][q['stations'][v[1]]['color']] == side}
                for side in (-1, 1)}
    return dict(q=q, reached=reached, visits=visits, chains=chains, segments=segments,
                headings=headings, total=sum(values), d=d['count'], values=values,
                final=d['count']+sum(values), duplicate=len(visits)>len(reached),
                unhit=len(q['stations'])-1-len(reached), max_turns=max(map(len, chains)),
                source_count=len(q['stations'])-1, shape=q['field'].shape,
                mask=(q['mask'].shape, q['mask'].tobytes()),
                widths=tuple(s['width'] for s in q['stations']))


def _evidence(features):
    _require(any(f['duplicate'] for f in features), "missing once-per-tower evidence")
    _require(any(f['unhit'] for f in features), "missing unreachable counter evidence")
    _require(any(len(set(f['values'])) > 1 for f in features), "missing unequal aggregation")
    _require(any(f['max_turns'] >= 2 for f in features), "missing recursive routing")
    _require(len({f['mask'] for f in features}) >= 2, "training must vary token silhouettes")
    observed_colors = {}
    for f in features:
        q = f['q']
        for i in f['reached']:
            color = q['stations'][i]['color']
            observed_colors.setdefault(color, set()).add(q['turns'][color])
    _require(any(len(sides) == 2 for sides in observed_colors.values()),
             "training must change the meaning of a used key color")
    _require(len({f['d'] for f in features}) >= 2, "receiver operand never varies")
    _require(len({f['total'] for f in features}) >= 2, "contribution never varies")
    _require(len({f['final'] for f in features}) >= 2, "constant-height shortcut fits")
    _require(any(features[0]['total']*f['d'] != f['total']*features[0]['d'] for f in features),
             "destination-multiplier shortcut fits")
    spare_space = False
    for f in features:
        q = f['q']
        d = q['stations'][q['dest']]
        y = d['r'] - (f['final']+1)*q['pitch']
        x = d['center']-q['mask'].shape[1]//2
        if y >= 0:
            block = q['field'][y:y+q['mask'].shape[0], x:x+q['mask'].shape[1]]
            spare_space |= block.shape == q['mask'].shape and np.all(block[q['mask']] == q['bg'])
    _require(spare_space, "fill-all-headroom shortcut fits")
    for side in (-1, 1):
        headings = set().union(*(f['headings'][side] for f in features))
        _require(len(headings) >= 2, "relative turn needs multiple arrival headings")
    rivals = (
        lambda f: 2*f['d'], lambda f: f['total'],
        lambda f: f['d']*f['total'], lambda f: f['d']+len(f['reached']),
        lambda f: f['d']+max(f['values']),
        lambda f: f['d']+sum(f['q']['stations'][v[1]]['count'] for v in f['visits']),
        lambda f: f['d']+sum(s['count'] for i, s in enumerate(f['q']['stations']) if i != f['q']['dest']),
        lambda f: f['d']+sum(f['q']['stations'][chain[0]]['count'] for chain in f['chains'] if chain),
        lambda f: f['d']+f['total']*int(f['q']['mask'].sum()),
        lambda f: f['d']+sum(f['q']['stations'][i]['width']//f['q']['mask'].shape[1]
                             for i in f['reached']),
    )
    for rival in rivals:
        _require(any(rival(f) != f['final'] for f in features), "simple arithmetic rival fits")
    # The four mappings of displayed left/right glyphs must be identifiable.
    fits = []
    for left in (-1, 1):
        for right in (-1, 1):
            works = True
            for f in features:
                q = f['q']
                changed = {color: left if side == -1 else right for color, side in q['turns'].items()}
                try:
                    output = _render(q, _trace(q, changed)[0])
                except AssertionError:
                    works = False
                    break
                if not np.array_equal(output, _render(q, f['reached'])):
                    works = False
                    break
            if works:
                fits.append((left, right))
    _require(fits == [(-1, 1)], "training does not identify the turn legend")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("illegal example count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['more_reachable', 'larger_tokens', 'receiver_count']))
    first = _fork_lesson(rng)
    f1 = _features(first)
    second = _selector_lesson(rng, f1)
    train = [first, second]
    seen = {_key(p['input']) for p in train}
    for i in range(2, int(num_train)):
        for _ in range(30):
            pair = _random_pair(rng, dcount=int(rng.integers(1, 3)), target=int(rng.choice([1, 2, 3])))
            if _key(pair['input']) not in seen:
                train.append(pair)
                seen.add(_key(pair['input']))
                break
        else:
            raise AssertionError("unique training construction exhausted")
    rng.shuffle(train)
    test = []
    for _ in range(int(num_test)):
        for _ in range(30):
            if shift == 'more_reachable':
                pair = _random_pair(rng, dcount=1, target=4)
            elif shift == 'larger_tokens':
                pair = _random_pair(rng, height_mask=3, dcount=1, target=int(rng.choice([1, 2])))
            else:
                pair = _random_pair(rng, dcount=3, target=int(rng.choice([1, 2])))
            if _key(pair['input']) not in seen:
                test.append(pair)
                seen.add(_key(pair['input']))
                break
        else:
            raise AssertionError("unique test construction exhausted")
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid counts")
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            f = _features(pair)
            q, reached = f['q'], f['reached']
            expected = _render(q, reached)
            _require(np.array_equal(out, expected), "incorrect collection output")
            _require(out.shape == q['field'].shape, "incorrect field crop")
            _require(reached, "need a reached counter")
            _require(all(out[p] == q['highlight'] for i in reached for p in q['stations'][i]['cells']),
                     "a reached counter was not highlighted")
            _require(all(out[p] == q['field'][p] for i, s in enumerate(q['stations'])
                         if i not in reached for p in s['cells']), "untouched tokens changed")
            _require(all(np.all(out[s['r'], s['c']:s['c']+s['width']] == q['bg'])
                         for s in q['stations']), "retained control base")
            if split == 'train':
                train_features.append(f)
            else:
                test_features.append(f)
    _evidence(train_features)
    for f in test_features:
        _require(f['q']['mask'].shape[0] > max(t['q']['mask'].shape[0] for t in train_features)
                 or f['d'] > max(t['d'] for t in train_features)
                 or len(f['reached']) > max(len(t['reached']) for t in train_features),
                 "test does not shift a taught axis")


def self_test():
    """Bounded family checks over 42 seeds, plus targeted semantic mutations."""
    records, puzzles = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), "solve disagrees")
                records.append(f)
    _require(len(puzzles) == 42, "seeds failed to vary")
    _require(len({f['shape'] for f in records}) >= 25, "insufficient geometry variation")
    _require(len({f['mask'] for f in records}) >= 25, "insufficient token variation")
    _require({1, 2, 3, 4} <= {len(f['reached']) for f in records}, "missing reachable-set counts")
    _require(any(f['max_turns'] >= 3 for f in records), "missing longer chains")
    _require(any(f['q']['mask'].shape[0] == 3 for f in records), "missing token scale shift")
    _require(len({f['final'] for f in records}) >= 4, "insufficient arithmetic variation")
    _require(len({s['width'] for f in records for s in f['q']['stations']}) >= 6,
             "insufficient obstacle-width variation")
    a, b = generate_puzzle(73, 2, 1), generate_puzzle(73, 2, 1)
    _require(all(np.array_equal(p[k], r[k]) for s in a for p, r in zip(a[s], b[s])
                 for k in ('input', 'output')), "nondeterministic puzzle")
    # Exercise maximal requests on each of the three structural shift regimes.
    regimes = set()
    for seed in range(42):
        regime = str(np.random.default_rng(seed).choice(['more_reachable', 'larger_tokens', 'receiver_count']))
        if regime not in regimes:
            puzzle = generate_puzzle(seed, 10, 4)
            _require(len(puzzle['train']) == 10 and len(puzzle['test']) == 4, "maximal counts")
            regimes.add(regime)
        if len(regimes) == 3:
            break
    _require(len(regimes) == 3, "missing shift regime")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A right-turn rectangular cycle revisits exactly the same ray state.
    ports = np.full((21, 21), -1, dtype=int)
    stations = [dict(r=10, center=10, color=0)]
    for i, (r, c, width) in enumerate(((10, 14, 3), (14, 11, 5),
                                      (13, 7, 3), (9, 8, 5)), 1):
        ports[r, c:c+width] = i
        stations.append(dict(color=1))
    try:
        _trace(dict(ports=ports, stations=stations, dest=0, turns={1: 1}))
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted a cyclic route")
    base = generate_puzzle(31, 2, 1)
    for defect in ('sum', 'highlight', 'unreachable', 'base', 'crop'):
        broken = {s: [{k: np.asarray(g).copy() for k, g in p.items()} for p in base[s]] for s in base}
        pair = broken['train'][0]
        f = _features(pair)
        q = f['q']
        if defect == 'sum':
            d = q['stations'][q['dest']]
            row = d['r'] - f['final'] * q['pitch']
            yy, xx = np.argwhere(q['mask'])[0]
            pair['output'][row+yy, d['center']-q['mask'].shape[1]//2+xx] = q['bg']
        elif defect == 'highlight':
            s = q['stations'][min(f['reached'])]
            pair['output'][s['cells'][0]] = s['payload']
        elif defect == 'unreachable':
            i = next(i for i in range(len(q['stations'])) if i != q['dest'] and i not in f['reached'])
            pair['output'][q['stations'][i]['cells'][0]] = q['highlight']
        elif defect == 'base':
            s = q['stations'][q['dest']]
            pair['output'][s['r'], s['c']] = s['color']
        else:
            pair['output'] = pair['input'].copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted defect: " + defect)


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
