"""Recover erased tile colors from registered negative-space glyphs.

Generation uses one RNG stream. Solving reads only pixels. The positive stencil
library and negative, color-erased alignment clue jointly locate and render a
miniature's full lattice; supplied seed tiles are restored as well as completed.
"""
from functools import lru_cache
import numpy as np


def _pick(rng, choices):
    return choices[int(rng.integers(len(choices)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _d4(a):
    return [np.rot90(a, i) for i in range(4)] + [
        np.rot90(np.fliplr(a), i) for i in range(4)]


def _asymmetric(a):
    variants = [np.flip(a, 0), np.flip(a, 1), np.flip(a, (0, 1))]
    if a.shape[0] == a.shape[1]:
        variants += _d4(a)[1:]
    return all(not np.array_equal(a, b) for b in variants)


def _separate(a, b):
    """Half-open boxes with at least one intervening background cell."""
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _frame_candidates(a, bg):
    found = []
    for s in range(4, 7):
        if min(a.shape) < s:
            continue
        w = np.lib.stride_tricks.sliding_window_view(a, (s, s))
        v = w[..., 0, 0]
        ok = ((v != bg) & np.all(w[..., 0, :] == v[..., None], axis=-1)
              & np.all(w[..., -1, :] == v[..., None], axis=-1)
              & np.all(w[..., :, 0] == v[..., None], axis=-1)
              & np.all(w[..., :, -1] == v[..., None], axis=-1))
        found.extend((int(y), int(x), s, int(v[y, x]))
                     for y, x in np.argwhere(ok))
    # A stencil or dense surviving inset can contain a smaller square ring.
    # Only maximal perimeters are representations, irrespective of ring color.
    return sorted(f for f in found if not any(
        q[2] > f[2] and q[0] <= f[0] and q[1] <= f[1]
        and f[0] + f[2] <= q[0] + q[2]
        and f[1] + f[2] <= q[1] + q[2] for q in found))


def _recover(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background is tied")
    frames = _frame_candidates(a, bg)
    _require(len(frames) >= 4, "need dictionary cards and a seed run")
    _require(len({(s, v) for y, x, s, v in frames}) == 1,
             "outer frames must share side and neutral color")
    s, neutral = frames[0][2:]
    boxes = [(y, x, y+s, x+s) for y, x, _, _ in frames]
    _require(all(_separate(b, c) for i, b in enumerate(boxes) for c in boxes[i+1:]),
             "framed objects overlap or have no gutter")
    outside = np.ones(a.shape, dtype=bool)
    masks, cards, seeds = {}, {}, []
    for y, x, _, _ in frames:
        outside[y:y+s, x:x+s] = False
        inner = a[y+1:y+s-1, x+1:x+s-1]
        foreign = set(int(v) for v in inner.flat) - {bg, neutral}
        if foreign:
            _require(len(foreign) == 1 and not np.any(inner == neutral),
                     "library inset must contain one payload and background")
            color = next(iter(foreign))
            mask = inner == color
            _require(color not in masks, "duplicate dictionary color")
            _require(0 < mask.sum() < mask.size, "empty or full stencil")
            masks[color] = mask.copy()
            cards[color] = (y, x, y+s, x+s)
        else:
            mask = inner == bg
            _require(0 < mask.sum() < mask.size, "seed must have holes and survivors")
            seeds.append((y, x, mask.copy()))
    _require(len(masks) >= 2 and len(seeds) >= 2, "need two symbols and two seed tiles")
    _require(len({m.tobytes() for m in masks.values()}) == len(masks),
             "ambiguous dictionary masks")
    points = np.argwhere(outside & (a != bg))
    _require(len(points) >= 4, "miniature is missing")
    my, mx = map(int, points.min(axis=0))
    ey, ex = map(int, points.max(axis=0) + 1)
    _require(np.all((points - (my, mx)) % 2 == 0), "swatches must have pitch two")
    _require(all(np.any(np.diff(np.unique(points[:, d])) == 2) for d in (0, 1)),
             "miniature pitch has no witness")
    plan = a[my:ey:2, mx:ex:2].copy()
    _require(all(2 <= n <= 5 for n in plan.shape), "unsupported miniature extent")
    mini_box = (my, mx, ey, ex)
    _require(all(_separate(mini_box, b) for b in boxes), "miniature touches a frame")
    lattice = np.zeros((ey-my, ex-mx), dtype=bool)
    lattice[::2, ::2] = True
    _require(np.all(a[my:ey, mx:ex][~lattice] == bg), "nonblank miniature separators")
    _require(set(int(v) for v in plan.flat) - {bg} <= set(masks),
             "miniature color has no dictionary entry")
    _require(np.count_nonzero(plan != bg) == len(points), "unexplained pixels")
    if len({y for y, x, m in seeds}) == 1:
        axis = 1
    else:
        _require(len({x for y, x, m in seeds}) == 1, "seed must lie on one axis")
        axis = 0
    seeds.sort(key=lambda t: t[axis])
    steps = np.diff([t[axis] for t in seeds])
    pitch = int(steps[0])
    _require(np.all(steps == pitch) and 1 <= pitch-s <= 3, "invalid seed spacing")
    sequence = []
    for y, x, holes in seeds:
        matches = [v for v, m in masks.items() if np.array_equal(m, holes)]
        _require(len(matches) == 1, "seed hole pattern has no unique dictionary identity")
        sequence.append(matches[0])
    length = len(sequence)
    matches = []
    for r in range(plan.shape[0] - (length-1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (length-1 if axis == 1 else 0)):
            if [int(plan[r+(axis == 0)*i, c+(axis == 1)*i])
                    for i in range(length)] == sequence:
                matches.append((r, c))
    _require(len(matches) == 1, "decoded sequence does not have one miniature match")
    match = matches[0]
    base = (seeds[0][0] - match[0]*pitch, seeds[0][1] - match[1]*pitch)
    by, bx = base
    end_y, end_x = (base[d] + (plan.shape[d]-1)*pitch + s for d in (0, 1))
    large_box = (by, bx, end_y, end_x)
    _require(0 <= by < end_y <= a.shape[0] and 0 <= bx < end_x <= a.shape[1],
             "completed lattice leaves canvas")
    _require(_separate(large_box, mini_box)
             and all(_separate(large_box, b) for b in cards.values()),
             "completion conflicts with a representation")
    sites = {(match[0]+(axis == 0)*i, match[1]+(axis == 1)*i) for i in range(length)}
    expected = a.copy()
    seed_only = a.copy()
    solid = a.copy()
    additions = []
    for (r, c), value in np.ndenumerate(plan):
        value = int(value)
        y, x = by+r*pitch, bx+c*pitch
        region = a[y:y+s, x:x+s]
        if (r, c) in sites:
            tile = np.full((s, s), neutral, dtype=np.int8)
            tile[1:-1, 1:-1][masks[value]] = bg
            _require(np.array_equal(region, tile), "seed differs from its registered negative")
        else:
            _require(np.all(region == bg), "unseeded site contains pixels")
        if value != bg:
            tile = np.full((s, s), value, dtype=np.int8)
            tile[1:-1, 1:-1][masks[value]] = bg
            expected[y:y+s, x:x+s] = tile
            solid[y:y+s, x:x+s] = value
            if (r, c) in sites:
                seed_only[y:y+s, x:x+s] = tile
            else:
                additions.append((r, c, value))
    _require(len({v for r, c, v in additions}) >= 2, "need additions in two colors")
    _require(all(np.count_nonzero(plan == v) >= 2 for v in sequence),
             "each clue symbol must be individually ambiguous")
    # The space between sites is not an extra channel for instructions or noise.
    explained = np.zeros(a.shape, dtype=bool)
    explained[my:ey, mx:ex] = True
    for y, x, h, v in frames:
        explained[y:y+h, x:x+h] = True
    _require(np.all(a[~explained] == bg), "unexplained scene pixels")
    info = dict(background=bg, neutral=neutral, side=s, gutter=pitch-s,
                plan=plan, masks=masks, cards=cards, miniature=mini_box,
                axis=axis, length=length, match=match, base=base,
                seed_sites=sites, large_box=large_box, additions=additions,
                sequence=tuple(sequence), unused=len(set(masks)-set(int(v) for v in plan.flat)),
                solid_ablation=solid, seed_only_ablation=seed_only)
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only reconstruction; no RNG, examples, seed or latent state."""
    return _recover(input_grid)[0]


def _segments(plan, bg, axis, mode):
    h, w = plan.shape
    choices = []
    for length in range(2, plan.shape[axis]+1):
        occurrences = {}
        for r in range(h-(length-1 if axis == 0 else 0)):
            for c in range(w-(length-1 if axis == 1 else 0)):
                seq = tuple(int(plan[r+(axis == 0)*i, c+(axis == 1)*i]) for i in range(length))
                if bg not in seq:
                    occurrences.setdefault(seq, []).append((r, c))
        for seq, starts in occurrences.items():
            if len(starts) != 1 or any(np.count_nonzero(plan == v) < 2 for v in seq):
                continue
            r, c = starts[0]
            cross = (r, c)[1-axis]
            boundary = cross in (0, plan.shape[1-axis]-1)
            whole = length == plan.shape[axis]
            if mode == 'whole' and not (whole and boundary):
                continue
            if mode == 'partial' and not (not whole and boundary):
                continue
            if mode == 'interior' and not (not whole and not boundary):
                continue
            other = plan.copy()
            for i in range(length):
                other[r+(axis == 0)*i, c+(axis == 1)*i] = bg
            if len(set(int(v) for v in other.flat)-{bg}) < 2:
                continue
            if mode in ('partial', 'interior'):
                line = other[r, :] if axis == 1 else other[:, c]
                if np.all(line == bg):
                    continue
            choices.append(((r, c), length, seq))
    return choices


def _sample_plan(rng, shape, palette, bg, axis, mode, demo):
    for _ in range(160):
        plan = rng.choice(palette, size=shape).astype(np.int8)
        density = _pick(rng, (0., 0., .1, .2, .3))
        plan[rng.random(shape) < density] = bg
        # All sampled dictionary payloads used by the plan must actually occur.
        positions = rng.choice(plan.size, len(palette), replace=False)
        plan.flat[positions] = palette
        if any(np.all(edge == bg) for edge in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        occupied = np.argwhere(plan != bg)
        if any(not np.any(np.diff(np.unique(occupied[:, d])) == 1) for d in (0, 1)):
            continue
        if demo and not _asymmetric(plan):
            continue
        choices = _segments(plan, bg, axis, mode)
        if choices:
            return plan, _pick(rng, choices)
    # Constructive bounded fallback. Work in horizontal coordinates, then
    # transpose back for a vertical clue. Protected sites witness repetitions.
    cross, along = shape[1-axis], shape[axis]
    a, b = map(int, palette[:2])
    plan = np.full((cross, along), a, dtype=np.int8)
    row = 1 if mode == 'interior' else 0
    other_row = 0 if row != 0 else 1
    if mode == 'whole' or along == 2:
        plan[row, -1] = b
        plan[other_row] = b
        plan[other_row, -1] = a
        locked = {(row, c) for c in range(along)} | {(other_row, 0), (other_row, along-1)}
    else:
        plan[row, -1] = b
        plan[other_row] = b
        locked = {(row, along-2), (row, along-1), (other_row, 0), (row, 0)}
    free = [(r, c) for r in range(cross) for c in range(along) if (r, c) not in locked]
    for color, (r, c) in zip(palette[2:], free):
        plan[r, c] = color
    if axis == 0:
        plan = plan.T.copy()
    choices = _segments(plan, bg, axis, mode)
    assert choices and (not demo or _asymmetric(plan)), 'constructive plan failed'
    return plan, _pick(rng, choices)


def _masks(rng, k, colors, witness):
    used, result = set(), {}
    for color in colors:
        need_asym = color == witness
        for _ in range(200):
            m = np.zeros((k, k), dtype=bool)
            motif = int(rng.integers(8))
            if motif == 0:
                m.flat[int(rng.integers(k*k))] = True
            elif motif == 1:
                m[:] = True
                m.flat[int(rng.integers(k*k))] = False
            elif motif == 2:
                np.fill_diagonal(m, True)
            elif motif == 3:
                m[int(rng.integers(k)), :] = True
            elif motif == 4:
                m[0, 0] = m[-1, -1] = True
            else:
                m = rng.random((k, k)) < rng.uniform(.2, .8)
            m = _pick(rng, _d4(m)).copy()
            if not (0 < m.sum() < m.size) or m.tobytes() in used:
                continue
            if need_asym and (len({t.tobytes() for t in _d4(m)}) != 8 or 2*m.sum() == m.size):
                continue
            break
        else:
            # Enumerate a finite vocabulary; k<=4, and at most seven cards.
            for bits in range(1, (1 << (k*k))-1):
                m = np.array([(bits >> i) & 1 for i in range(k*k)], dtype=bool).reshape(k, k)
                if m.tobytes() in used:
                    continue
                if need_asym and (len({t.tobytes() for t in _d4(m)}) != 8 or 2*m.sum() == m.size):
                    continue
                break
            else:
                raise AssertionError('finite mask vocabulary exhausted')
        used.add(m.tobytes())
        result[int(color)] = m
    return result


@lru_cache(None)
def _shelves(r, c, s, gutter, n):
    big = ((r-1)*(s+gutter)+s, (c-1)*(s+gutter)+s)
    mini = (2*r-1, 2*c-1)
    options = []
    for nc in range(1, n+1):
        nr = (n+nc-1)//nc
        bank = (nr*s+nr-1, nc*s+nc-1)
        for first in (0, 1):
            utility = [max(bank[d], mini[d]) for d in (0, 1)]
            utility[first] = bank[first]+mini[first]+1
            for second in (0, 1):
                total = [max(big[d], utility[d]) for d in (0, 1)]
                total[second] = big[second]+utility[second]+1
                if max(total) <= 30:
                    options.append((nc, first, second))
    return tuple(options)


@lru_cache(None)
def _configurations(axis, mode, demo):
    choices = []
    for r in range(2, 6):
        for c in range(2, 6):
            dims = (r, c)
            if r*c < 6 or (mode in ('partial', 'interior') and dims[axis] < 3):
                continue
            if mode == 'interior' and dims[1-axis] < 3:
                continue
            for s in range(4, 7):
                if demo == 0 and s == 6 or demo == 1 and s == 4:
                    continue
                for gutter in range(1, 4):
                    for used in range(2, min(5, r*c//2)+1):
                        for extra in range(1 if demo == 0 else 0, 3):
                            if _shelves(r, c, s, gutter, used+extra):
                                choices.append((r, c, s, gutter, used, extra))
    return tuple(choices)


def _join(rng, left, right, axis, gap):
    shape = [max(left[d], right[d]) for d in (0, 1)]
    shape[axis] = left[axis]+right[axis]+gap
    a, b = [0, 0], [0, 0]
    if rng.integers(2):
        b[axis] = left[axis]+gap
    else:
        a[axis] = right[axis]+gap
    other = 1-axis
    a[other] = int(rng.integers(shape[other]-left[other]+1))
    b[other] = int(rng.integers(shape[other]-right[other]+1))
    return tuple(shape), tuple(a), tuple(b)


def _shelf_layout(rng, r, c, s, gutter, n):
    big = ((r-1)*(s+gutter)+s, (c-1)*(s+gutter)+s)
    mini = (2*r-1, 2*c-1)
    for attempt in range(17):
        nc, first, second = _pick(rng, _shelves(r, c, s, gutter, n))
        nr = (n+nc-1)//nc
        cg, gap1, gap2 = (map(int, rng.integers(1, 4, 3)) if attempt < 16 else (1, 1, 1))
        bank = (nr*s+(nr-1)*cg, nc*s+(nc-1)*cg)
        utility = [max(bank[d], mini[d]) for d in (0, 1)]
        utility[first] = bank[first]+mini[first]+gap1
        total = [max(big[d], utility[d]) for d in (0, 1)]
        total[second] = big[second]+utility[second]+gap2
        if max(total) <= 30:
            break
    utility, bank_offset, mini_offset = _join(rng, bank, mini, first, gap1)
    total, base, util_offset = _join(rng, big, utility, second, gap2)
    slots = [(bank_offset[0]+i*(s+cg), bank_offset[1]+j*(s+cg))
             for i in range(nr) for j in range(nc)]
    slots = [slots[int(i)] for i in rng.permutation(len(slots))[:n]]
    mini_offset = tuple(mini_offset[d]+util_offset[d] for d in (0, 1))
    slots = [tuple(t[d]+util_offset[d] for d in (0, 1)) for t in slots]
    if rng.random() < .2:
        shape = (30, 30)
    elif rng.random() < .25:
        length = int(rng.integers(max(total), 31))
        shape = (length, length)
    else:
        shape = tuple(int(rng.integers(t, min(30, t+6)+1)) for t in total)
    offset = tuple(int(rng.integers(shape[d]-total[d]+1)) for d in (0, 1))
    return (shape, tuple(base[d]+offset[d] for d in (0, 1)),
            tuple(mini_offset[d]+offset[d] for d in (0, 1)),
            [tuple(t[d]+offset[d] for d in (0, 1)) for t in slots])


def _scatter_layout(rng, r, c, s, gutter, n):
    big = ((r-1)*(s+gutter)+s, (c-1)*(s+gutter)+s)
    mini = (2*r-1, 2*c-1)
    for _ in range(8):
        shape = tuple(int(rng.integers(max(24, t), 31)) for t in big)
        base = tuple(int(rng.integers(shape[d]-big[d]+1)) for d in (0, 1))
        occupied = np.zeros(shape, dtype=bool)

        def block(y, x, h, w):
            occupied[max(0, y-1):min(shape[0], y+h+1),
                     max(0, x-1):min(shape[1], x+w+1)] = True

        def place(h, w):
            slots = [(y, x) for y in range(shape[0]-h+1) for x in range(shape[1]-w+1)]
            for i in rng.permutation(len(slots)):
                y, x = slots[int(i)]
                if not occupied[y:y+h, x:x+w].any():
                    block(y, x, h, w)
                    return y, x
            return None

        block(*base, *big)
        anchor = place(*mini)
        if anchor is None:
            continue
        cards = []
        for _ in range(n):
            slot = place(s, s)
            if slot is None:
                break
            cards.append(slot)
        if len(cards) == n:
            return shape, base, anchor, cards
    return _shelf_layout(rng, r, c, s, gutter, n)


def _draw(rng, config, palette, bg, neutral, axis, mode, demo):
    r, c, s, gutter, used, extra = config
    plan, (match, length, sequence) = _sample_plan(
        rng, (r, c), palette[:used], bg, axis, mode, demo >= 0)
    masks = _masks(rng, s-2, palette, sequence[0] if demo == 1 else None)
    layout = _scatter_layout if rng.random() < .45 else _shelf_layout
    shape, base, mini, slots = layout(rng, r, c, s, gutter, len(palette))
    inp = np.full(shape, bg, dtype=np.int8)
    inp[mini[0]:mini[0]+2*r-1:2, mini[1]:mini[1]+2*c-1:2] = plan
    for color, (y, x) in zip(rng.permutation(palette), slots):
        inp[y:y+s, x:x+s] = neutral
        inp[y+1:y+s-1, x+1:x+s-1] = np.where(masks[int(color)], color, bg)
    pitch = s+gutter
    for i in range(length):
        rr, cc = match[0]+(axis == 0)*i, match[1]+(axis == 1)*i
        y, x = base[0]+rr*pitch, base[1]+cc*pitch
        inp[y:y+s, x:x+s] = neutral
        inp[y+1:y+s-1, x+1:x+s-1][masks[int(plan[rr, cc])]] = bg
    out = inp.copy()
    # Direct construction from the sampled objects, not a call to the solver.
    for (rr, cc), color in np.ndenumerate(plan):
        if color == bg:
            continue
        y, x = base[0]+rr*pitch, base[1]+cc*pitch
        out[y:y+s, x:x+s] = color
        out[y+1:y+s-1, x+1:x+s-1][masks[int(color)]] = bg
    while np.count_nonzero(inp == bg) <= max(np.count_nonzero(inp == v) for v in range(10) if v != bg):
        d = 0 if inp.shape[0] < 30 else 1
        assert inp.shape[d] < 30, 'background cannot be made dominant'
        padding = [(0, 0), (0, 0)]
        padding[d] = (0, 1)
        inp = np.pad(inp, padding, constant_values=bg)
        out = np.pad(out, padding, constant_values=bg)
    return {'input': inp, 'output': out}


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    axes = _pick(rng, ((1,), (0,), (0, 1)))
    first_axes = list(axes) if len(axes) == 2 else list(axes)*2
    rng.shuffle(first_axes)
    result, seen, first_info = {'train': [], 'test': []}, set(), None
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            demo = index if split == 'train' and index < 2 else -1
            axis = first_axes[index] if demo >= 0 else int(_pick(rng, axes))
            mode = ('whole' if demo == 0 else 'partial' if demo == 1 else
                    'interior' if split == 'test' else 'any')
            configs = _configurations(axis, mode, demo)
            if demo == 1:
                configs = tuple(q for q in configs if q[2] != first_info['side'])
            for _ in range(32):
                config = _pick(rng, configs)
                colors = rng.permutation(10)
                bg, neutral = map(int, colors[:2])
                if demo == 1 and bg == first_info['background']:
                    colors[0], colors[1] = colors[1], colors[0]
                    bg, neutral = map(int, colors[:2])
                palette = colors[2:2+config[4]+config[5]]
                pair = _draw(rng, config, palette, bg, neutral, axis, mode, demo)
                if _key(pair['input']) not in seen:
                    break
            else:
                # Finite injective recolorings exceed the maximum fourteen
                # existing inputs, without changing masks or rule conventions.
                active = [bg, neutral] + [v for v in range(10) if v not in (bg, neutral)]
                candidates = []
                for b in range(10):
                    for m in range(10):
                        if b == m or (demo == 1 and b == first_info['background']):
                            continue
                        targets = [b, m] + [v for v in range(10) if v not in (b, m)]
                        mapping = np.empty(10, dtype=np.int8)
                        mapping[active] = targets
                        p = {role: mapping[grid] for role, grid in pair.items()}
                        if _key(p['input']) not in seen:
                            candidates.append(p)
                assert candidates
                pair = candidates[0]
            seen.add(_key(pair['input']))
            result[split].append(pair)
            if demo == 0:
                first_info = _recover(pair['input'])[1]
        rng.shuffle(result[split])
    validate_generated_puzzle(result)
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _recover(inp)
            assert np.array_equal(expected, out), 'incorrect reconstruction or changed representation'
            assert not np.array_equal(out, info['solid_ablation']), 'A has no effect'
            assert not np.array_equal(out, info['seed_only_ablation']), 'B has no effect'
            infos[split].append(info)
            by, bx = info['base']
            pitch, side = info['side']+info['gutter'], info['side']
            large = np.zeros(inp.shape, dtype=bool)
            for (r, c), v in np.ndenumerate(info['plan']):
                y, x = by+r*pitch, bx+c*pitch
                large[y:y+side, x:x+side] = True
                tile = out[y:y+side, x:x+side]
                if v == info['background']:
                    assert np.all(tile == v), 'empty site was filled'
                else:
                    assert np.all(tile[0] == v) and np.all(tile[-1] == v)
                    assert np.all(tile[:, 0] == v) and np.all(tile[:, -1] == v)
                    assert np.array_equal(tile[1:-1, 1:-1] == info['background'], info['masks'][int(v)])
            assert np.array_equal(inp[~large], out[~large]), 'changed unrelated cells or gutters'
    training = infos['train']
    assert len({i['side'] for i in training}) >= 2
    assert len({i['background'] for i in training}) >= 2
    assert any(i['unused'] > 0 for i in training), 'no selection demonstration'
    assert any(i['length'] == i['plan'].shape[i['axis']]
               and i['match'][1-i['axis']] in (0, i['plan'].shape[1-i['axis']]-1) for i in training)
    assert any(i['length'] < i['plan'].shape[i['axis']] for i in training)
    teaching = {i['axis'] for i in training if _asymmetric(i['plan'])}
    assert {i['axis'] for group in infos.values() for i in group} <= teaching
    orientations = set(range(8))
    nonhalf = False
    for info in training:
        for v in set(int(v) for v in info['plan'].flat)-{info['background']}:
            m = info['masks'][v]
            orientations &= {j for j, t in enumerate(_d4(m)) if np.array_equal(m, t)}
            nonhalf |= 2*m.sum() != m.size
    assert orientations == {0} and nonhalf, 'training leaves mask orientation or polarity unresolved'
    for i in infos['test']:
        assert min(i['plan'].shape) >= 3 and i['length'] < i['plan'].shape[i['axis']]
        assert 0 < i['match'][1-i['axis']] < i['plan'].shape[1-i['axis']]-1
        assert any((r, c)[1-i['axis']] == i['match'][1-i['axis']]
                   for r, c, v in i['additions']), 'test has no extension along the clue line'


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded tests on 42 seeds, with pixel-recovered diversity measurements."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    shapes, sizes, gutters, axes, lengths, matches = set(), set(), set(), set(), set(), set()
    backgrounds, neutral_colors, unused, canvases, masks_seen = set(), set(), set(), set(), set()
    blanks = disconnected = symmetric = sparse = dense = 0
    structural = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                prediction, info = _recover(pair['input'])
                assert np.array_equal(prediction, pair['output'])
                shapes.add(info['plan'].shape); sizes.add(info['side']); gutters.add(info['gutter'])
                axes.add(info['axis']); lengths.add(info['length']); matches.add(info['match'])
                backgrounds.add(info['background']); neutral_colors.add(info['neutral'])
                unused.add(info['unused']); canvases.add(pair['input'].shape)
                blanks += np.any(info['plan'] == info['background'])
                mapping = {}
                structural.add((pair['input'].shape, tuple(mapping.setdefault(int(v), len(mapping))
                                                          for v in pair['input'].flat)))
                for m in info['masks'].values():
                    masks_seen.add((m.shape, m.tobytes()))
                    symmetric += len({t.tobytes() for t in _d4(m)}) < 8
                    sparse += m.sum() <= m.size/3
                    dense += m.sum() >= 2*m.size/3
                    coords = set(map(tuple, np.argwhere(m)))
                    reached, stack = set(), [next(iter(coords))]
                    while stack:
                        cell = stack.pop()
                        if cell in reached:
                            continue
                        reached.add(cell)
                        y, x = cell
                        stack.extend(q for q in ((y+1, x), (y-1, x), (y, x+1), (y, x-1))
                                     if q in coords and q not in reached)
                    disconnected += len(reached) != len(coords)
    assert len(shapes) >= 8 and sizes == {4, 5, 6} and gutters == {1, 2, 3}
    assert axes == {0, 1} and len(lengths) >= 3 and len(matches) >= 8
    assert backgrounds == set(range(10)) and neutral_colors == set(range(10))
    assert {0, 1, 2} <= unused and len(canvases) >= 25 and len(structural) >= 190
    assert len(masks_seen) >= 80 and min(blanks, disconnected, symmetric, sparse, dense) > 10
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(314159, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Every local color role can be renamed without changing the transformation.
    mapping = np.array([3, 9, 2, 7, 5, 0, 8, 6, 1, 4], dtype=np.int8)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
    pair = puzzles[0]['train'][0]
    inp = pair['input'].copy()
    info = _recover(inp)[1]
    first, second = list(info['cards'])[:2]
    y, x, ey, ex = info['cards'][second]
    inp[y+1:ey-1, x+1:ex-1] = np.where(info['masks'][first], second, info['background'])
    try:
        solve(inp)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous dictionary accepted')
    inp = pair['input'].copy()
    plan = info['plan'].copy()
    row, col = info['match']
    other_cross = next(i for i in range(plan.shape[1-info['axis']])
                       if i != info['match'][1-info['axis']])
    if info['axis'] == 1:
        row = other_cross
    else:
        col = other_cross
    for i, v in enumerate(info['sequence']):
        plan[row+(info['axis'] == 0)*i, col+(info['axis'] == 1)*i] = v
    my, mx, ey, ex = info['miniature']
    inp[my:ey:2, mx:ex:2] = plan
    try:
        solve(inp)
    except ValueError as error:
        assert 'sequence' in str(error), 'ambiguity fixture failed for an unrelated reason'
    else:
        raise AssertionError('ambiguous alignment sequence accepted')
    return dict(seeds=42, pairs=sum(len(p[s]) for p in puzzles for s in ('train', 'test')),
                plan_shapes=sorted(shapes), tile_sides=sorted(sizes), gutters=sorted(gutters),
                seed_lengths=sorted(lengths), match_coordinates=len(matches),
                canvas_shapes=len(canvases), color_normalized_structures=len(structural),
                distinct_masks=len(masks_seen), pairs_with_blanks=int(blanks),
                disconnected_masks=int(disconnected), symmetric_masks=int(symmetric),
                sparse_masks=int(sparse), dense_masks=int(dense))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
