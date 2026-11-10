"""Linked drift: rotation-equivalent objects share a displacement and a brake.

No parent code or data is loaded. The solver reads only the supplied grid.
"""
from collections import Counter
import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = tuple(d for d in DIR8 if 0 in d)
DIAGONAL = tuple(d for d in DIR8 if 0 not in d)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, directions=DIR4):
    seen = ~np.asarray(mask, dtype=bool).copy()
    result = []
    h, w = seen.shape
    for rr, cc in np.argwhere(~seen):
        p = int(rr), int(cc)
        if seen[p]:
            continue
        seen[p] = True
        stack, cells = [p], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _mask(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    a = np.zeros((max(r for r, c in cells) - r0 + 1,
                  max(c for r, c in cells) - c0 + 1), dtype=bool)
    for r, c in cells:
        a[r - r0, c - c0] = True
    return a


def _key(mask):
    return mask.shape, mask.astype(np.uint8).tobytes()


def _canonical(mask, reflect=False):
    versions = [np.rot90(mask, k) for k in range(4)]
    if reflect:
        versions += [np.rot90(np.fliplr(mask), k) for k in range(4)]
    return min(_key(a) for a in versions)


def _decode_key(tile):
    carrier = int(tile[0, 0])
    border = np.concatenate((tile[0], tile[-1], tile[:, 0], tile[:, -1]))
    if np.any(border != carrier):
        return None
    others = set(map(int, np.unique(tile))) - {carrier}
    if len(others) != 2:
        return None
    interpretations = []
    for tip in sorted(others):
        trail = next(iter(others - {tip}))
        tips = set(map(tuple, np.argwhere(tile == tip)))
        trails = set(map(tuple, np.argwhere(tile == trail)))
        if len(tips) < 2:
            continue
        for dr, dc in DIR8:
            used, lengths = set(), []
            for r, c in sorted(tips):
                line = []
                p = r - dr, c - dc
                while p in trails:
                    line.append(p)
                    p = p[0] - dr, p[1] - dc
                if len(line) < 2 or used.intersection(line):
                    break
                lengths.append(len(line))
                used.update(line)
            if len(lengths) == len(tips) and used == trails:
                interpretations.append((trail, tip, (dr, dc), tuple(lengths)))
    if len(interpretations) != 1:
        return None
    trail, tip, direction, lengths = interpretations[0]
    return dict(carrier=carrier, trail=trail, tip=tip, direction=direction,
                lengths=lengths, key_shape=tile.shape)


def _read_input(input_grid, object_directions=DIR8):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(int(counts[bg]) > a.size // 2, "background must occupy more than half the input")
    keys = []
    for cells in _components(a != bg):
        rows, cols = zip(*cells)
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if min(h, w) < 4 or len(cells) != h * w:
            continue
        info = _decode_key(a[r0:r1 + 1, c0:c1 + 1])
        if info is not None:
            info.update(key_cells=cells, key_box=(r0, r1, c0, c1))
            keys.append(info)
    _require(len(keys) == 1, "exactly one readable direction key is required")
    info = keys[0]
    excluded = np.zeros(a.shape, bool)
    for p in info['key_cells']:
        excluded[p] = True
    walls = (a == info['carrier']) & ~excluded
    _require(bool(np.any(walls)), "missing arena wall")
    reserved = {bg, info['carrier'], info['trail'], info['tip']}
    objects = []
    for color in sorted(set(map(int, np.unique(a))) - {bg, info['carrier']}):
        mask = (a == color) & ~excluded
        _require(not np.any(mask) or color not in reserved, "key glyph roles cannot be object colors")
        for cells in _components(mask, object_directions):
            objects.append(dict(cells=cells, color=color, mask=_mask(cells)))
    objects.sort(key=lambda o: min(o['cells']))
    _require(bool(objects), "missing objects")
    # Free-space topology, rather than a bounding rectangle, identifies the arena.
    region_at, enclosed = {}, []
    for cells in _components(~walls):
        edge = any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
                   for r, c in cells)
        if not edge:
            label = len(enclosed)
            enclosed.append(cells)
            region_at.update({p: label for p in cells})
    for obj in objects:
        labels = {region_at.get(p, -1) for p in obj['cells']}
        _require(len(labels) == 1 and -1 not in labels, "object must fit inside one closed chamber")
        obj['region'] = next(iter(labels))
    info.update(grid=a, background=bg, walls=set(map(tuple, np.argwhere(walls))),
                objects=objects, regions=enclosed)
    return info


def _groups(objects, matching='rotation', independent=False, move_unique=False, exact_pair=False):
    buckets = {}
    for i, obj in enumerate(objects):
        mask = obj['mask']
        signature = (_canonical(mask) if matching == 'rotation' else
                     _canonical(mask, True) if matching == 'reflection' else
                     _key(mask) if matching == 'identity' else
                     int(mask.sum()) if matching == 'area' else obj['color'])
        buckets.setdefault(signature, []).append(i)
    groups, active = [], []
    for members in buckets.values():
        moving = (len(members) == 2 if exact_pair else len(members) > 1) or move_unique
        for group in ([[i] for i in members] if independent else [members]):
            groups.append(group)
            active.append(moving)
    return groups, active


def _simulate(info, matching='rotation', independent=False, move_unique=False,
              corner=True, reverse=False, frozen_solid=True, use_walls=True,
              boxes=False, exact_pair=False):
    """Synchronous drift with a closure of freezes before each common step.

    Active bodies all have the same velocity, so their relative positions cannot
    collide. Bodies that cannot take the next step freeze before anyone moves.
    Rechecking at the same tick handles every new obstacle without an order tie.
    """
    objects = info['objects']
    groups, flags = _groups(objects, matching, independent, move_unique, exact_pair)
    active = {g for g, flag in enumerate(flags) if flag}
    body_cells = []
    for obj in objects:
        original = obj['cells']
        if boxes:
            rows, cols = zip(*original)
            original = {(r, c) for r in range(min(rows), max(rows) + 1)
                        for c in range(min(cols), max(cols) + 1)}
        body_cells.append(original)
    cells = [{p for i in members for p in body_cells[i]} for members in groups]
    distances = [0] * len(groups)
    dr, dc = info['direction']
    if reverse:
        dr, dc = -dr, -dc
    obstacles = {p: -1 for p in info['walls']} if use_walls else {}
    for g in range(len(groups)):
        if g not in active:
            obstacles.update({p: g for p in cells[g]})
    events = []
    h, w = info['grid'].shape
    for tick in range(max(h, w) + 1):
        while active:
            blocked = {}
            for g in sorted(active):
                blockers = set()
                for r, c in cells[g]:
                    checks = [(r + dr, c + dc)]
                    if corner and dr and dc:
                        checks += [(r + dr, c), (r, c + dc)]
                    for p in checks:
                        if not (0 <= p[0] < h and 0 <= p[1] < w):
                            blockers.add(-1)
                        elif p in obstacles:
                            blockers.add(obstacles[p])
                if blockers:
                    blocked[g] = blockers
            if not blocked:
                break
            for g, blockers in blocked.items():
                events.append(dict(group=g, distance=distances[g], blockers=blockers))
                active.remove(g)
                if frozen_solid:
                    obstacles.update({p: g for p in cells[g]})
        if not active:
            break
        for g in active:
            cells[g] = {(r + dr, c + dc) for r, c in cells[g]}
            distances[g] += 1
    _require(not active, "motion did not terminate")
    offsets = [0] * len(objects)
    for g, members in enumerate(groups):
        for i in members:
            offsets[i] = distances[g]
    return dict(groups=groups, moving=flags, distances=distances, offsets=offsets,
                events=events, direction=(dr, dc))


def _render(info, result):
    out = info['grid'].copy()
    for obj in info['objects']:
        for p in obj['cells']:
            out[p] = info['background']
    dr, dc = result['direction']
    for obj, distance in zip(info['objects'], result['offsets']):
        for r, c in obj['cells']:
            p = r + dr * distance, c + dc * distance
            _require(0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1], "clipped object")
            _require(int(out[p]) == info['background'], "destination overlap")
            out[p] = obj['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    return _render(info, _simulate(info))


def _differs(info, expected, **options):
    try:
        alternative = _render(info, _simulate(info, **options))
    except AssertionError:
        # An overlapping/clipped alternative also contradicts mask conservation.
        return True
    return not np.array_equal(expected, alternative)


def _features(info, result, out, full=True):
    objects = info['objects']
    groups = result['groups']
    sizes = [len(g) for g in groups]
    repeated = [g for g in groups if len(g) > 1]
    singles = [objects[g[0]] for g in groups if len(g) == 1]
    keys = {_canonical(o['mask']) for o in objects}
    r0, r1, c0, c1 = info['key_box']
    wy, wx = zip(*info['walls'])
    ar, ac = min(wy) + max(wy), min(wx) + max(wx)
    dy, dx = r0 + r1 - ar, c0 + c1 - ac
    side = ('bottom' if dy > 0 else 'top') if abs(dy) >= abs(dx) else ('right' if dx > 0 else 'left')
    feature = dict(shape=info['grid'].shape, background=info['background'],
                   direction=info['direction'], key_shape=info['key_shape'], key_side=side,
                   objects=len(objects), class_sizes=sorted(sizes), singleton_count=len(singles),
                   chambers=len(info['regions']), offsets=result['offsets'],
                   areas=[int(o['mask'].sum()) for o in objects],
                   masks=[_canonical(o['mask']) for o in objects],
                   diagonal_objects=any(len(_components(o['mask'], DIR4)) > 1 for o in objects),
                   moving_diagonal=any(d > 0 and len(_components(o['mask'], DIR4)) > 1
                                       for o, d in zip(objects, result['offsets'])),
                   moving_triplet=any(len(g) == 3 and result['offsets'][g[0]] > 0 for g in groups),
                   cross_chamber=any(len({objects[i]['region'] for i in g}) > 1 for g in repeated),
                   different_color_match=any(len({objects[i]['color'] for i in g}) > 1 for g in repeated),
                   color_cross_role=any(objects[i]['color'] == s['color'] for g in repeated for i in g for s in singles),
                   quarter_turn=any(_key(objects[i]['mask']) != _key(objects[j]['mask']) and
                                    _key(objects[j]['mask']) in (_key(np.rot90(objects[i]['mask'])),
                                                               _key(np.rot90(objects[i]['mask'], 3)))
                                    for g in repeated for i in g for j in g),
                   ordinary_singleton=any(_canonical(np.fliplr(o['mask'])) == _canonical(o['mask']) or
                                          _canonical(np.fliplr(o['mask'])) not in keys for o in singles),
                   cascade=any(any(b >= 0 and result['moving'][b] for b in e['blockers'])
                               for e in result['events']))
    if full:
        feature.update(sync_difference=_differs(info, out, independent=True),
                       identity_difference=_differs(info, out, matching='identity'),
                       reflection_difference=_differs(info, out, matching='reflection'),
                       area_difference=_differs(info, out, matching='area'),
                       color_difference=_differs(info, out, matching='color'),
                       all_move_difference=_differs(info, out, independent=True, move_unique=True),
                       corner_difference=_differs(info, out, corner=False),
                       bbox_difference=_differs(info, out, boxes=True),
                       exact_pair_difference=_differs(info, out, exact_pair=True),
                       cascade_difference=_simulate(info, frozen_solid=False)['offsets'] != result['offsets'],
                       wall_difference=_differs(info, out, use_walls=False),
                       reverse_difference=_differs(info, out, reverse=True))
        four_info = _read_input(info['grid'], DIR4)
        feature['four_connectivity_difference'] = _differs(four_info, out)
    return feature


def _random_shape(rng, limit):
    kind = int(rng.integers(8))
    h, w = int(rng.integers(2, limit + 1)), int(rng.integers(2, limit + 1))
    if kind == 0:
        a = np.ones((1, int(rng.integers(1, limit + 1))), bool)
    elif kind == 1:
        a = np.ones((h, w), bool)
    elif kind == 2:
        a = np.zeros((h, w), bool)
        a[:, 0] = a[-1, :] = True
    elif kind == 3:
        a = np.ones((max(3, h), max(3, w)), bool)
        a[1:-1, 1:-1] = False
    elif kind == 4:
        a = np.zeros((h, w), bool)
        a[int(rng.integers(h)), :] = True
        a[:, int(rng.integers(w))] = True
    elif kind == 5:
        a = np.zeros((h, w), bool)
        a[1:, 1] = a[-1, 1:] = True
        a[0, 0] = True
    else:
        cells = {(0, 0)}
        target = int(rng.integers(3, min(13, limit * limit) + 1))
        directions = DIR4 if kind == 6 else DIR8
        for _ in range(target - 1):
            candidates = set()
            for r, c in sorted(cells):
                for dr, dc in directions:
                    p = r + dr, c + dc
                    trial = cells | {p}
                    if p not in cells and max(x for x, y in trial) - min(x for x, y in trial) < limit and \
                            max(y for x, y in trial) - min(y for x, y in trial) < limit:
                        candidates.add(p)
            if not candidates:
                break
            ordered = sorted(candidates)
            cells.add(ordered[int(rng.integers(len(ordered)))])
        a = _mask(cells)
    return _mask(set(map(tuple, np.argwhere(np.rot90(a, int(rng.integers(4)))))))


def _sample_shape(rng, forbidden, limit, chiral=False, diagonal=False):
    def allowed(a):
        return (_canonical(a) not in forbidden and
                (not chiral or _canonical(a) != _canonical(np.fliplr(a))) and
                (not diagonal or len(_components(a, DIR4)) > 1))
    for _ in range(120):
        a = _random_shape(rng, limit)
        if allowed(a):
            return a
    for bits in range(1, 512):
        a = _mask(set(map(tuple, np.argwhere(np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3)))))
        if len(_components(a, DIR8)) == 1 and allowed(a):
            return a
    raise AssertionError('shape vocabulary exhausted')


def _make_key(rng, direction, carrier, trail, tip, compact=False):
    dr, dc = direction
    count, length = (2, 2) if compact else (int(rng.choice([2, 2, 3])), int(rng.choice([2, 2, 3])))
    points = []
    for i in range(count):
        stagger = 0 if compact else int(rng.integers(2))
        base = (0, i * 2) if dr else (i * 2, 0)
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc, tip if j == length else trail))
    r0, c0 = min(r for r, c, k in points), min(c for r, c, k in points)
    h, w = max(r for r, c, k in points) - r0 + 1, max(c for r, c, k in points) - c0 + 1
    pads = [1] * 4 if compact else [int(rng.choice([1, 1, 2])) for _ in range(4)]
    tile = np.full((h + pads[0] + pads[1], w + pads[2] + pads[3]), carrier, np.int8)
    for r, c, color in points:
        tile[r - r0 + pads[0], c - c0 + pads[2]] = color
    _require(_decode_key(tile) is not None, 'ambiguous generated key')
    return tile


def _make_wall(rng, h, w, direction, corner_teaching=False):
    footprint = np.ones((h, w), bool)
    kind = int(rng.integers(4))
    if kind in (1, 2):
        for k in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            cut = int(rng.integers(1, 5))
            for r in range(cut):
                for c in range(cut):
                    if kind == 1 or r + c < cut:
                        footprint[r if k < 2 else h - r - 1,
                                  c if k % 2 == 0 else w - c - 1] = False
    elif kind == 3:
        left, right = 0, 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, 3))
            right = int(np.clip(right + rng.integers(-1, 2), 0, 3))
            footprint[r, :left] = False
            if right:
                footprint[r, w - right:] = False
    p = np.pad(footprint, 1)
    inside = footprint.copy()
    for dr, dc in DIR4:
        inside &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~inside
    for _ in range(int(rng.choice([0, 0, 1, 1, 2, 3]))):
        horizontal = bool(rng.integers(2))
        if horizontal:
            r = int(rng.integers(h // 3, 2 * h // 3 + 1))
            stop = w if rng.random() < .4 else int(rng.integers(w // 3, 2 * w // 3 + 1))
            part = slice(0, stop) if rng.random() < .5 else slice(w - stop, w)
            wall[r, part] |= footprint[r, part]
        else:
            c = int(rng.integers(w // 3, 2 * w // 3 + 1))
            stop = h if rng.random() < .4 else int(rng.integers(h // 3, 2 * h // 3 + 1))
            part = slice(0, stop) if rng.random() < .5 else slice(h - stop, h)
            wall[part, c] |= footprint[part, c]
    if corner_teaching:
        # A partial boundary supplies an open cell beyond a blocked diagonal side.
        stop, col = int(rng.integers(h // 3, 2 * h // 3)), int(rng.integers(w // 3, 2 * w // 3))
        patch = np.zeros((h, w), bool)
        patch[:stop + 1, col] = True
        if direction[0] < 0:
            patch = np.flipud(patch)
        if direction[1] < 0:
            patch = np.fliplr(patch)
        wall |= patch & footprint
    return wall, footprint


def _make_objects(rng, palette, teaching, limit, diagonal_needed):
    if teaching == 1:
        sizes, singles = [2], 2
    elif teaching == 2:
        sizes, singles = [2, 3], int(rng.integers(3))
    else:
        n = int(rng.choice([1, 2, 3], p=[.35, .50, .15]))
        sizes = list(map(int, rng.choice([2, 3, 4], n, p=[.62, .30, .08])))
        singles = int(rng.choice([0, 1, 2, 3], p=[.18, .44, .30, .08]))
        while sum(sizes) + singles > 10:
            if singles:
                singles -= 1
            else:
                sizes[sizes.index(max(sizes))] -= 1
    forbidden, objects, bases = set(), [], []
    for k, size in enumerate(sizes):
        base = _sample_shape(rng, forbidden, limit, chiral=teaching == 1,
                             diagonal=diagonal_needed and k == 0)
        bases.append(base)
        forbidden.add(_canonical(base))
        turns = list(map(int, rng.integers(4, size=size)))
        if teaching == 1:
            turns[1] = (turns[0] + int(rng.choice([1, 3]))) % 4
        for turn in turns:
            objects.append(dict(mask=np.rot90(base, turn).copy(), class_id=k))
    for i in range(singles):
        if teaching == 1 and i == 0:
            base = np.fliplr(bases[0]).copy()
        else:
            excluded = forbidden | ({_canonical(np.fliplr(b)) for b in bases} if teaching == 1 else set())
            base = _sample_shape(rng, excluded, limit)
        forbidden.add(_canonical(base))
        bases.append(base)
        objects.append(dict(mask=np.rot90(base, int(rng.integers(4))).copy(), class_id=len(sizes) + i))
    colors = [int(rng.choice(palette)) for _ in objects]
    if teaching == 1:
        colors = [palette[0], palette[1], palette[0], palette[2]]
    for obj, color in zip(objects, colors):
        obj['color'] = color
    return objects


def _place_objects(rng, objects, wall, footprint, moving_direction=None):
    free = footprint & ~wall
    labels = np.full(wall.shape, -1, int)
    for label, cells in enumerate(_components(free)):
        for p in cells:
            labels[p] = label
    used = set()
    h, w = wall.shape
    for i in sorted(range(len(objects)), key=lambda i: -int(objects[i]['mask'].sum())):
        obj = objects[i]
        mh, mw = obj['mask'].shape
        local = list(map(tuple, np.argwhere(obj['mask'])))
        halo = {(r + dr, c + dc) for r, c in used for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        candidates = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
        placed = False
        for index in rng.permutation(len(candidates)):
            r, c = candidates[int(index)]
            cells = {(r + int(y), c + int(x)) for y, x in local}
            labs = {int(labels[p]) for p in cells}
            if len(labs) != 1 or -1 in labs or cells.intersection(halo):
                continue
            if moving_direction is not None and obj['class_id'] < 2:
                dr, dc = moving_direction
                checks = {(y + dr, x + dc) for y, x in cells}
                if dr and dc:
                    checks |= {(y + dr, x) for y, x in cells} | {(y, x + dc) for y, x in cells}
                if any(not (0 <= y < h and 0 <= x < w) or not free[y, x] for y, x in checks):
                    continue
            obj['cells'] = cells
            used.update(cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _assemble(rng, wall, tile, objects, roles, side):
    bg, carrier = roles[0], roles[1]
    h, w = wall.shape
    kh, kw = tile.shape
    vertical = side in ('top', 'bottom')
    bh, bw = (h + kh + 1, max(w, kw)) if vertical else (max(h, kh), w + kw + 1)
    if rng.random() < .08:
        ch = cw = 30
    else:
        ch, cw = min(30, bh + int(rng.integers(4))), min(30, bw + int(rng.integers(4)))
    rpad, cpad = int(rng.integers(ch - bh + 1)), int(rng.integers(cw - bw + 1))
    if vertical:
        ar = rpad + (kh + 1 if side == 'top' else 0)
        kr = rpad + (0 if side == 'top' else h + 1)
        ac, kc = cpad + int(rng.integers(bw - w + 1)), cpad + int(rng.integers(bw - kw + 1))
    else:
        ac = cpad + (kw + 1 if side == 'left' else 0)
        kc = cpad + (0 if side == 'left' else w + 1)
        ar, kr = rpad + int(rng.integers(bh - h + 1)), rpad + int(rng.integers(bh - kh + 1))
    a = np.full((ch, cw), bg, np.int8)
    a[ar:ar + h, ac:ac + w][wall] = carrier
    a[kr:kr + kh, kc:kc + kw] = tile
    for obj in objects:
        for r, c in obj['cells']:
            a[r + ar, c + ac] = obj['color']
    return a


def _cascade_witness(rng, direction):
    """A varied contact arrangement for the coupled teaching requirements.

    One body's corner collision and another body's remote brake coexist. The
    occupied mask, not its box, clears a short boundary branch. Random rotations,
    stem length, arena margins, corner cuts, palette, and key remain variable.
    The last attempt is a guaranteed fitting arrangement under the same rule.
    """
    for attempt in range(41):
        exact = attempt == 40
        roles = list(map(int, rng.permutation(10)))
        sy, sx = (0, 0) if exact else (int(rng.integers(3)), int(rng.integers(3)))
        h, w = 21 + sy, 21 + sx
        footprint = np.ones((h, w), bool)
        if not exact:
            for k in rng.choice(4, int(rng.integers(4)), replace=False):
                cut = int(rng.integers(1, 4))
                for r in range(cut):
                    for c in range(cut - r):
                        footprint[r if k < 2 else h - r - 1,
                                  c if k % 2 == 0 else w - c - 1] = False
        p = np.pad(footprint, 1)
        interior = footprint.copy()
        for dr, dc in DIR4:
            interior &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        wall = footprint & ~interior
        wall[:11 + sy, 10 + sx] = True
        wall[7 + sy, 8 + sx:11 + sx] = True
        stem = 2 if exact else int(rng.choice([2, 2, 3]))
        diagonal = np.zeros((stem + 1, 2), bool)
        diagonal[0, 0] = True
        diagonal[1:, 1] = True
        elbow = np.array([[1, 0], [1, 1]], bool)
        partner_turn = 0 if exact else int(rng.integers(4))
        turns = [0, 0, 0] if exact else list(map(int, rng.integers(4, size=3)))
        placements = [(diagonal, 4, 3), (np.rot90(diagonal, partner_turn), 11 - (stem - 2), 3)]
        placements += [(np.rot90(elbow, turn), r, c)
                       for turn, (r, c) in zip(turns, [(8, 1), (1, 11), (6, 11)])]
        colors = [int(rng.choice(roles[4:])) for _ in placements]
        objects = []
        for (mask, r, c), color in zip(placements, colors):
            cells = {(r + sy + int(y), c + sx + int(x)) for y, x in np.argwhere(mask)}
            objects.append(dict(cells=cells, color=color))
        used = set()
        separated = True
        for obj in objects:
            halo = {(r + dr, c + dc) for r, c in used
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if obj['cells'].intersection(halo) or any(wall[p] for p in obj['cells']):
                separated = False
                break
            used.update(obj['cells'])
        if not separated:
            continue
        if direction[0] < 0:
            wall = np.flipud(wall).copy()
            for obj in objects:
                obj['cells'] = {(h - r - 1, c) for r, c in obj['cells']}
        if direction[1] < 0:
            wall = np.fliplr(wall).copy()
            for obj in objects:
                obj['cells'] = {(r, w - c - 1) for r, c in obj['cells']}
        for _ in range(12):
            tile = _make_key(rng, direction, roles[1], roles[2], roles[3])
            sides = [s for s in ('top', 'bottom') if h + tile.shape[0] + 1 <= 30]
            sides += [s for s in ('left', 'right') if w + tile.shape[1] + 1 <= 30]
            if sides:
                break
        if not sides:
            tile = _make_key(rng, direction, roles[1], roles[2], roles[3], compact=True)
            sides = ['top', 'bottom']
        a = _assemble(rng, wall, tile, objects, roles, str(rng.choice(sides)))
        info = _read_input(a)
        result = _simulate(info)
        out = _render(info, result)
        f = _features(info, result, out)
        if all(f[k] for k in ('corner_difference', 'cascade_difference', 'bbox_difference',
                             'exact_pair_difference', 'moving_triplet')):
            return dict(input=a, output=out)
    raise AssertionError('guaranteed cascade witness failed')


def _generate_pair(rng, direction, teaching=0, diagonal_needed=False):
    for attempt in range(240 if teaching != 2 else 24):
        roles = list(map(int, rng.permutation(10)))
        tile = _make_key(rng, direction, roles[1], roles[2], roles[3])
        side = str(rng.choice(['top', 'bottom', 'left', 'right']))
        max_h = min(23, 29 - tile.shape[0]) if side in ('top', 'bottom') else 23
        max_w = min(23, 29 - tile.shape[1]) if side in ('left', 'right') else 23
        limit = int(rng.choice([3, 4])) if teaching == 2 else int(rng.choice([3, 4, 5], p=[.25, .50, .25]))
        objects = _make_objects(rng, roles[4:], teaching, limit, diagonal_needed)
        low = 14 if len(objects) <= 4 else 17
        h = int(rng.integers(min(low, max_h), max_h + 1))
        w = int(rng.integers(min(low, max_w), max_w + 1))
        wall, footprint = _make_wall(rng, h, w, direction, teaching == 2)
        if not _place_objects(rng, objects, wall, footprint, direction if teaching == 2 else None):
            continue
        a = _assemble(rng, wall, tile, objects, roles, side)
        if int(np.count_nonzero(a == roles[0])) <= a.size // 2:
            continue
        info = _read_input(a)
        result = _simulate(info)
        out = _render(info, result)
        if np.array_equal(a, out):
            continue
        features = _features(info, result, out, full=bool(teaching))
        if teaching == 1 and not all(features[k] for k in (
                'sync_difference', 'identity_difference', 'reflection_difference',
                'area_difference', 'color_difference', 'all_move_difference', 'ordinary_singleton',
                'wall_difference', 'reverse_difference')):
            continue
        if teaching == 1 and diagonal_needed and not (
                features['moving_diagonal'] and features['four_connectivity_difference']):
            continue
        if teaching == 2 and not all(features[k] for k in (
                'corner_difference', 'cascade_difference', 'bbox_difference',
                'exact_pair_difference', 'moving_triplet')):
            continue
        return dict(input=a, output=out)
    if teaching == 2:
        return _cascade_witness(rng, direction)
    raise AssertionError(f'bounded scene construction exhausted for teaching mode {teaching}')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    cardinal, diagonal = DIR4[int(rng.integers(4))], DIAGONAL[int(rng.integers(4))]
    heldout = [d for d in DIR8 if d not in (cardinal, diagonal)]
    puzzle, seen = {'train': [], 'test': []}, set()

    def append(split, direction, teaching=0, diagonal_needed=False):
        for _ in range(12):
            pair = _generate_pair(rng, direction, teaching, diagonal_needed)
            signature = _fingerprint(pair['input'])
            if signature not in seen:
                seen.add(signature)
                puzzle[split].append(pair)
                return
        raise AssertionError('distinct inputs exhausted')

    for _ in range(nq):
        pool = heldout if rng.random() < .75 else DIR8
        append('test', pool[int(rng.integers(len(pool)))])
    diagonal_needed = any(_features(_read_input(p['input']), _simulate(_read_input(p['input'])), p['output'], False)['diagonal_objects']
                          for p in puzzle['test'])
    append('train', cardinal, 1, diagonal_needed)
    append('train', diagonal, 2)
    for _ in range(nt - 2):
        append('train', DIR8[int(rng.integers(8))])
    # Witness order is not a convention and conveys no direction information.
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_fingerprint(puzzle):
    return tuple((_fingerprint(p['input']), _fingerprint(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
             len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'canvas changed')
            fp = _fingerprint(a)
            _require(fp not in seen, 'duplicate input')
            seen.add(fp)
            info = _read_input(a)
            _require(2 <= len(info['objects']) <= 10, 'generated object count')
            occupied = set()
            for obj in info['objects']:
                halo = {(r + dr, c + dc) for r, c in occupied
                        for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                _require(not obj['cells'].intersection(halo), 'initial objects touch')
                occupied.update(obj['cells'])
            result = _simulate(info)
            expected = _render(info, result)
            _require(np.array_equal(b, expected), 'incorrect linked drift')
            _require(not np.array_equal(a, b), 'generated example must show movement')
            _require(Counter(map(int, a.ravel())) == Counter(map(int, b.ravel())), 'color/cell mass changed')
            for p in info['key_cells'] | info['walls']:
                _require(int(a[p]) == int(b[p]), 'wall or key changed')
            groups = result['groups']
            for g in groups:
                _require(len({result['offsets'][i] for i in g}) == 1, 'class not synchronized')
                if len(g) == 1:
                    _require(result['offsets'][g[0]] == 0, 'singleton moved')
            _require(all(_canonical(info['objects'][i]['mask']) == _canonical(info['objects'][g[0]]['mask'])
                         for g in groups for i in g), 'incorrect grouping')
            features[split].append(_features(info, result, b))
    train, test = features['train'], features['test']
    _require(len({f['direction'] for f in train}) >= 2, 'constant-direction ambiguity')
    for name in ('quarter_turn', 'different_color_match', 'color_cross_role', 'ordinary_singleton',
                 'sync_difference', 'identity_difference', 'reflection_difference', 'area_difference',
                 'color_difference', 'all_move_difference', 'corner_difference', 'cascade_difference',
                 'wall_difference', 'reverse_difference', 'bbox_difference',
                 'exact_pair_difference', 'moving_triplet'):
        _require(any(f[name] for f in train), f'missing training witness: {name}')
    _require(any(2 in f['class_sizes'] for f in train) and any(3 in f['class_sizes'] for f in train),
             'partner threshold needs pair and triplet evidence')
    if any(f['diagonal_objects'] for f in test):
        _require(any(f['moving_diagonal'] and f['four_connectivity_difference'] for f in train),
                 'eight-connectivity needs moving training evidence')
    return features


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_fingerprint(p) for p in puzzles}) == 42, 'seed variation')
    _require(_puzzle_fingerprint(puzzles[0]) == _puzzle_fingerprint(generate_puzzle(0)), 'determinism')
    rows = [f for p in puzzles for fs in validate_generated_puzzle(p).values() for f in fs]
    _require({f['direction'] for f in rows} == set(DIR8), 'eight direction coverage')
    _require({f['key_side'] for f in rows} == {'top', 'bottom', 'left', 'right'}, 'key placement diversity')
    _require(len({f['shape'] for f in rows}) >= 40, 'canvas diversity')
    _require(len({f['key_shape'] for f in rows}) >= 12, 'key geometry diversity')
    _require(len({tuple(f['class_sizes']) for f in rows}) >= 12, 'class profile diversity')
    _require(len({k for f in rows for k in f['masks']}) >= 80, 'mask diversity')
    _require({2, 3, 4} <= {n for f in rows for n in f['class_sizes']}, 'multiplicity variation')
    _require(any(f['cross_chamber'] for f in rows), 'remote links across chambers')
    _require(any(f['singleton_count'] == 0 for f in rows), 'all-repeated scenes')
    _require(any(0 in f['offsets'] and max(f['offsets']) >= 3 for f in rows), 'stationary and long motion')
    _require(len({f['background'] for f in rows}) >= 8, 'palette variation')
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(1000 + 10 * tr + te, tr, te)
            _require((len(p['train']), len(p['test'])) == (tr, te), 'explicit count contract')
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Rotation transports the key, every shape and the collision geometry together.
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for k in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)),
                     'global rotation covariance')
    bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity corruption accepted')
