"""Raise the dam: counted construction changes a conserved liquid's capacity.

The solver reads only the grid. No parent modules, stored examples, learned
conventions, or generation metadata are used. Colors 0/1/2/3/5 are fixed roles.
"""
from __future__ import annotations

from itertools import permutations
import numpy as np


RIGID = (4, 6, 7, 8, 9)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        y, x = stack.pop()
        for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if p in remaining:
                remaining.remove(p)
                stack.append(p)
    return not remaining


def _parse(g):
    """Recover dry token lanes and cup geometry from visible markers/pixels."""
    assert 5 not in g, "processed color is output-only"
    sy, sx = np.nonzero(g == 2)
    dy, dxs = np.nonzero(g == 3)
    assert len(sy) and len(dy) and len(set(sy)) == len(set(dy)) == 1
    control_y, dest_y = int(sy[0]), int(dy[0])
    dx, dw = int(dxs.min()), len(dxs)
    assert np.array_equal(dxs, np.arange(dx, dx + dw))
    assert 1 <= dw <= 3 and control_y < dest_y - 4
    bank = g[:control_y + 1]
    expected = np.zeros_like(g)
    lanes = []
    for x, stop in _runs(np.any(bank != 0, axis=0)):
        w = stop - x
        assert 2 <= w <= 4
        marked = bool(np.any(bank[-1, x:stop] == 2))
        if marked:
            assert np.all(bank[-1, x:stop] == 2)
            expected[control_y, x:stop] = 2
        else:
            assert np.all(bank[-1, x:stop] == 0)
        strip = bank[:control_y, x:stop]
        groups = _runs(np.any(strip != 0, axis=1))
        assert groups
        masks, coords, colors = [], [], set()
        for y, end in groups:
            block = strip[y:end]
            mask = block != 0
            assert 1 <= end - y <= 3 and _connected(mask)
            assert mask.sum() >= 2 and np.all(np.any(mask, axis=0))
            colors.update(int(c) for c in np.unique(block[mask]))
            assert colors <= set(RIGID)
            masks.append(mask)
            yy, xx = np.nonzero(mask)
            coords.extend(zip((yy + y).tolist(), (xx + x).tolist()))
            expected[y:end, x:stop] = block
        assert len(colors) == 1
        assert all(np.array_equal(masks[0], m) for m in masks)
        starts = [a for a, b in groups]
        pitches = {b - a for a, b in zip(starts, starts[1:])}
        assert len(pitches) <= 1
        if pitches:
            assert next(iter(pitches)) - masks[0].shape[0] in (1, 2)
        assert control_y - groups[-1][1] >= 1
        lanes.append(dict(x=x, width=w, marked=marked, count=len(groups),
                          coords=tuple(coords), mask=masks[0], color=colors.pop(),
                          baseline=groups[-1][1], pitch=next(iter(pitches), None)))
    assert 1 <= sum(s['marked'] for s in lanes) <= 3
    assert sum(not s['marked'] for s in lanes) <= 2
    assert len({s['baseline'] for s in lanes}) == 1
    assert all(np.array_equal(lanes[0]['mask'], s['mask']) for s in lanes)
    assert len({s['pitch'] for s in lanes if s['pitch'] is not None}) <= 1
    total = sum(s['count'] for s in lanes if s['marked'])

    base = dest_y - 2
    floor_runs = _runs(np.isin(g[base], RIGID))
    spans = [(l, r) for l, r in floor_runs if l < dx and dx + dw < r]
    assert len(spans) == 1
    left, end = spans[0]
    right = end - 1
    assert g[base - 1, dx] in RIGID
    rim = base
    while rim > 0 and g[rim - 1, left] in RIGID:
        rim -= 1
    rim_right = base
    while rim_right > 0 and g[rim_right - 1, right] in RIGID:
        rim_right -= 1
    assert rim == rim_right and rim >= control_y + 2
    inner_l, inner_r = left, right + 1
    while g[rim, inner_l] in RIGID:
        inner_l += 1
    while g[rim, inner_r - 1] in RIGID:
        inner_r -= 1
    assert 1 <= inner_l - left <= 2 and 1 <= right + 1 - inner_r <= 2
    assert inner_l < dx < dx + dw < inner_r
    old_top = base - 1
    while old_top > rim and g[old_top - 1, dx] in RIGID:
        old_top -= 1
    old_height = base - old_top
    assert old_height >= 2
    dam_color = int(g[base - 1, dx])
    assert np.all(g[old_top:base, dx:dx + dw] == dam_color)
    new_top = old_top - total
    assert new_top >= rim + 2
    assert np.all(g[new_top:old_top, dx:dx + dw] == 0)

    solid = np.zeros_like(g, dtype=bool)
    solid[rim:base + 1, left:inner_l] = True
    solid[rim:base + 1, inner_r:right + 1] = True
    floors = []
    for x in range(inner_l, inner_r):
        if dx <= x < dx + dw:
            solid[old_top:base + 1, x] = True
            continue
        y = base
        while y > rim and g[y - 1, x] in RIGID:
            y -= 1
        assert 0 <= base - y < old_height
        solid[y:base + 1, x] = True
        floors.append(base - y)
    assert np.all(np.isin(g[solid], RIGID))
    expected[solid] = g[solid]
    expected[dest_y, dx:dx + dw] = 3
    expected[g == 1] = 1
    assert np.array_equal(expected, g), "extra cells or malformed terrain"

    left_side, right_side = np.zeros_like(solid), np.zeros_like(solid)
    left_side[rim:base, inner_l:dx] = True
    right_side[rim:base, dx + dw:inner_r] = True
    left_side &= ~solid
    right_side &= ~solid
    liquid = g == 1
    assert liquid.any() and not liquid[rim].any()
    on_left = not np.any(liquid & ~left_side)
    on_right = not np.any(liquid & ~right_side)
    assert on_left != on_right, "liquid must originate in one chamber"
    source, receiver = (left_side, right_side) if on_left else (right_side, left_side)
    return dict(lanes=lanes, total=total, control_y=control_y, dest_y=dest_y,
                base=base, rim=rim, left=left, right=right, inner_l=inner_l,
                inner_r=inner_r, dx=dx, dw=dw, dam_color=dam_color,
                old_height=old_height, old_top=old_top, new_top=new_top,
                solid=solid, source=source, receiver=receiver, on_left=on_left,
                volume=int(liquid.sum()), floors=tuple(floors))


def _fill_levels(available, volume, partial=False):
    """Complete bottom levels; partial=True is only a shortcut counterfactual."""
    result = np.zeros_like(available)
    for y in range(available.shape[0] - 1, -1, -1):
        if not volume:
            break
        cols = np.flatnonzero(available[y])
        if volume < len(cols):
            if not partial:
                raise ValueError("liquid would partially fill a surface level")
            result[y, cols[:volume]] = True
            volume = 0
        else:
            result[y, cols] = True
            volume -= len(cols)
    if volume:
        raise ValueError("liquid exceeds cup capacity")
    return result


def _allocate(p, top=None, partial=False):
    """Hydrostatic capacities from pixels, not from initial liquid shape."""
    top = p['new_top'] if top is None else top
    rows = np.arange(p['solid'].shape[0])[:, None]
    low_source = p['source'] & (rows >= top)
    low_receiver = p['receiver'] & (rows >= top)
    a, b, v = int(low_source.sum()), int(low_receiver.sum()), p['volume']
    if v <= a:
        return _fill_levels(low_source, v, partial), 'retain'
    if v <= a + b:
        return low_source | _fill_levels(low_receiver, v - a, partial), 'spill'
    corridor = np.zeros_like(p['solid'])
    corridor[p['rim']:top, p['inner_l']:p['inner_r']] = True
    return low_source | low_receiver | _fill_levels(corridor, v - a - b, partial), 'join'


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    p = _parse(g)
    water, _ = _allocate(p)
    out = g.copy()
    out[(out == 1) | (out == 2) | (out == 3)] = 0
    out[p['new_top']:p['old_top'], p['dx']:p['dx'] + p['dw']] = p['dam_color']
    for s in p['lanes']:
        if s['marked']:
            yy, xx = zip(*s['coords'])
            out[yy, xx] = 5
    out[water] = 1
    return out


def _token_mask(rng, h, w):
    style = int(rng.integers(4))
    mask = np.ones((h, w), dtype=bool)
    if h == 1 or style == 0:
        return mask
    mask[:] = False
    if style == 1:
        mask[:, int(rng.integers(w))] = True
        mask[int(rng.integers(h)), :] = True
    else:
        mask[:, 0] = True
        mask[-1, :] = True
        for y in range(h - 1):
            mask[y, :int(rng.integers(1, w + 1))] = True
    if rng.integers(2):
        mask = mask[:, ::-1]
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _partition(rng, total, n, unequal=False):
    choices = []
    # Small bounded arithmetic range; enumerate positive compositions.
    def visit(prefix, remaining, count):
        if count == 1:
            if 1 <= remaining <= 4:
                values = prefix + [remaining]
                if not unequal or len(set(values)) > 1:
                    choices.append(values)
            return
        for v in range(1, min(4, remaining - count + 1) + 1):
            visit(prefix + [v], remaining - v, count - 1)
    visit([], total, n)
    assert choices
    return choices[int(rng.integers(len(choices)))]


def _profile(rng, w, maximum):
    style = int(rng.integers(4))
    if style == 0:
        return np.zeros(w, dtype=int)
    if style == 1:
        return np.linspace(0, maximum, w).round().astype(int)
    if style == 2:
        return rng.integers(0, maximum + 1, size=w)
    result, level = [], int(rng.integers(maximum + 1))
    for _ in range(w):
        level = int(np.clip(level + int(rng.integers(-1, 2)), 0, maximum))
        result.append(level)
    return np.array(result)


def _make_pair(rng, counts, d, dw, kind, anchor=False, _fallback_only=False):
    """Construct labelled levels directly; no call to solve makes ground truth."""
    total, f = sum(counts), d + sum(counts)
    unmarked = int(rng.integers(1 if anchor else 0, 3))
    other_counts = rng.integers(1, 5, size=unmarked).tolist()
    heights = list(counts) + other_counts
    marked = [True] * len(counts) + [False] * unmarked
    order = rng.permutation(len(heights))
    heights, marked = [heights[i] for i in order], [marked[i] for i in order]
    tallest, n = max(heights), len(heights)
    for attempt in ([40] if _fallback_only else range(41)):
        fallback = attempt == 40
        sw = 9 if fallback else int(rng.integers(4, 10))
        rw = 2 if fallback else int(rng.integers(2, 9))
        extra = 5 if fallback else int(rng.integers(3, 6))
        depth = f + extra
        floor_s = np.zeros(sw, int) if fallback else _profile(rng, sw, min(2, d - 1))
        floor_r = np.zeros(rw, int) if fallback else _profile(rng, rw, min(2, d - 1))
        source_cap = sw * f - int(floor_s.sum())
        receiver_cap = rw * f - int(floor_r.sum())
        input_cap = sw * (depth - 1) - int(floor_s.sum())
        options = []
        if kind == 'retain':
            # Surface above the old crest: omitting construction must leak.
            for level in range(d + 1, f if anchor else f + 1):
                v = sw * level - int(floor_s.sum())
                if v <= input_cap:
                    options.append((v, level, 0))
        elif kind == 'spill':
            # Both surfaces are above floor ridges and intentionally unequal.
            for level in range(int(floor_r.max()) + 1, f):
                v = source_cap + rw * level - int(floor_r.sum())
                if v <= input_cap:
                    options.append((v, f, level))
        else:
            for high in range(1, extra - 1):
                v = source_cap + receiver_cap + high * (sw + rw + dw)
                if v <= input_cap:
                    options.append((v, f + high, f + high))
        # Minimal top band, cup, one-row marker separation, and destination bar.
        geometries = [(h, gap) for h in range(1, 4) for gap in (1, 2)
                      if tallest * h + (tallest - 1) * gap + depth + 7 <= 30]
        if not options or not geometries:
            continue
        h, gap = (1, 1) if fallback else geometries[int(rng.integers(len(geometries)))]
        w = 2 if fallback else int(rng.integers(2, 5))
        lw, ow = (1, 1) if fallback else tuple(map(int, rng.integers(1, 3, size=2)))
        cup_width = lw + sw + dw + rw + ow
        bank_min = n * w + n - 1
        width_min = max(cup_width, bank_min)
        if width_min > 30:
            continue
        width = width_min if fallback else int(rng.integers(width_min, 31))
        left = int(rng.integers(width - cup_width + 1))
        inner_l, dx = left + lw, left + lw + sw
        inner_r, right = dx + dw + rw, left + cup_width - 1
        bank_height = tallest * h + (tallest - 1) * gap
        spare = 30 - (bank_height + depth + 7)
        additions = rng.multinomial(min(spare, int(rng.integers(0, 5))), [0.25] * 4)
        margin, control_gap = 1 + int(additions[0]), 1 + int(additions[1])
        control_y = margin + bank_height + control_gap
        rim = control_y + 2 + int(additions[2])
        base = rim + depth
        dest_y = base + 2
        canvas_h = dest_y + 1 + int(additions[3])
        assert canvas_h <= 30
        volume, level_s, level_r = options[int(rng.integers(len(options)))]
        frame_color, dam_color = map(int, rng.choice(RIGID, 2, replace=False))
        terrain_palette = rng.choice(RIGID, int(rng.integers(1, 4)), replace=False)
        inp = np.zeros((canvas_h, width), dtype=np.int8)
        inp[rim:base + 1, left:inner_l] = frame_color
        inp[rim:base + 1, inner_r:right + 1] = frame_color
        for x, elevation in zip(range(inner_l, dx), floor_s):
            inp[base - int(elevation):base + 1, x] = int(rng.choice(terrain_palette))
        for x, elevation in zip(range(dx + dw, inner_r), floor_r):
            inp[base - int(elevation):base + 1, x] = int(rng.choice(terrain_palette))
        inp[base, dx:dx + dw] = frame_color
        inp[base - d:base, dx:dx + dw] = dam_color
        inp[dest_y, dx:dx + dw] = 3
        solid = np.isin(inp, RIGID)
        rows = np.arange(canvas_h)[:, None]
        src_free, rec_free = np.zeros_like(solid), np.zeros_like(solid)
        src_free[rim + 1:base, inner_l:dx] = True
        rec_free[rim + 1:base, dx + dw:inner_r] = True
        src_free &= ~solid
        rec_free &= ~solid
        water = src_free & (rows >= base - level_s)
        water |= rec_free & (rows >= base - level_r)
        if kind == 'join':
            water[base - level_s:base - f, dx:dx + dw] = True
        assert int(water.sum()) == volume
        out = inp.copy()
        out[dest_y, dx:dx + dw] = 0
        out[base - f:base - d, dx:dx + dw] = dam_color

        # Locally visible token geometry, independent of dam slab geometry.
        mask = _token_mask(rng, h, w)
        colors = rng.choice(RIGID, n).tolist()
        if anchor:
            colors[marked.index(False)] = colors[marked.index(True)]
        extras = rng.multinomial(width - bank_min, np.full(n + 1, 1 / (n + 1)))
        x = int(extras[0])
        for i, count in enumerate(heights):
            for k in range(count):
                y = control_y - control_gap - h - k * (h + gap)
                inp[y:y + h, x:x + w][mask] = colors[i]
                out[y:y + h, x:x + w][mask] = 5 if marked[i] else colors[i]
            if marked[i]:
                inp[control_y, x:x + w] = 2
            x += w + (1 + int(extras[i + 1]) if i + 1 < n else 0)

        # Supplies may be scattered, compact, banded, or partly settled.
        points = np.argwhere(src_free)
        outside = np.argwhere(src_free & ~water)
        assert len(outside)
        anchor_point = outside[int(rng.integers(len(outside)))]
        distance = np.abs(points - anchor_point)
        jitter = rng.random(len(points))
        style = int(rng.integers(4))
        if style == 0:
            score = jitter
        elif style == 1:
            score = distance.sum(axis=1) + jitter
        elif style == 2:
            score = 3 * distance[:, 0] + distance[:, 1] / 10 + jitter
        else:
            score = (~water[points[:, 0], points[:, 1]]) * 3 + jitter
        selected = points[np.argsort(score)[:volume]].copy()
        if not np.any(np.all(selected == anchor_point, axis=1)):
            selected[-1] = anchor_point
        inp[selected[:, 0], selected[:, 1]] = 1
        out[water] = 1
        if rng.integers(2):
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        return {'input': inp, 'output': out}
    raise AssertionError("feasible construction fallback failed")


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # At least frame and dam colors differ: >=20 distinct role permutations,
    # exceeding the maximum 14 examples. No unbounded uniqueness rejection.
    for colors in permutations(RIGID):
        lut = np.arange(10, dtype=np.int8)
        lut[list(RIGID)] = colors
        candidate = {k: lut[g] for k, g in pair.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError("finite color permutations exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if not _integer(value) or not low <= int(value) <= high:
            raise ValueError("illegal pair count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    options = [(s1, d1, s2, d2) for s1 in range(2, 5) for d1 in range(2, 5)
               for s2 in range(3, 6) for d2 in range(2, 5)
               if s1 != s2 and d1 != d2 and s1 * d2 != s2 * d1
               and s1 + d1 != s2 + d2
               and not (s1 * d1 == s1 + d1 and s2 * d2 == s2 + d2)]
    s1, d1, s2, d2 = options[int(rng.integers(len(options)))]
    contributions = [[s1], _partition(rng, s2, 2, unequal=True)]
    core = list(zip(contributions, [d1, d2]))
    rng.shuffle(core)
    kinds = ['retain', 'spill']
    rng.shuffle(kinds)
    widths = [1, 2]
    rng.shuffle(widths)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            if split == 'train' and i < 2:
                counts, d = core[i]
                kind, dw, anchor = kinds[i], widths[i], True
            else:
                nsrc = 3 if split == 'test' else int(rng.integers(1, 4))
                total = int(rng.integers(nsrc, max(s1, s2) + 1))
                # Avoid a >4-token single stack; regroup instead.
                if nsrc == 1:
                    total = min(total, 4)
                counts = _partition(rng, total, nsrc)
                d = int(rng.integers(2, 5))
                kind = 'join' if split == 'train' and i == 2 else str(rng.choice(
                    ['retain', 'spill', 'join'] if num_train >= 3 else ['retain', 'spill']))
                dw = 3 if split == 'test' else int(rng.integers(1, 3))
                anchor = False
            pair = _unique(_make_pair(rng, counts, d, dw, kind, anchor), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out, p):
    water, kind = _allocate(p)
    old_water, _ = _allocate(p, p['old_top'], partial=True)
    selected = [s for s in p['lanes'] if s['marked']]
    return dict(shape=inp.shape, kind=kind, width=p['dw'], height=p['old_height'],
                total=p['total'], sources=tuple(s['count'] for s in selected),
                unmarked=sum(not s['marked'] for s in p['lanes']), volume=p['volume'],
                mask=(selected[0]['mask'].shape, selected[0]['mask'].tobytes()),
                area=int(selected[0]['mask'].sum()), side=p['on_left'], floors=p['floors'],
                new_top=p['new_top'], old_capacity=int((p['source'] &
                    (np.arange(inp.shape[0])[:, None] >= p['old_top'])).sum()),
                changed_water=not np.array_equal(inp == 1, water),
                construction_matters=not np.array_equal(old_water, water),
                lateral=not np.array_equal((inp == 1).sum(axis=0), water.sum(axis=0)),
                colors=tuple(map(int, np.unique(inp[np.isin(inp, RIGID)]))))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            p = _parse(inp)
            water, kind = _allocate(p)
            expected = solve(inp)
            assert np.array_equal(out, expected), "incorrect composed answer"
            assert np.array_equal(out == 1, water)
            assert (out == 1).sum() == (inp == 1).sum()
            assert not np.any((out == 2) | (out == 3))
            assert np.array_equal(out[p['solid']], inp[p['solid']])
            for s in p['lanes']:
                yy, xx = zip(*s['coords'])
                assert np.all(out[yy, xx] == (5 if s['marked'] else s['color']))
            # Flood closure below each surface proves full levels and no leaks.
            new_solid = p['solid'].copy()
            new_solid[p['new_top']:p['old_top'], p['dx']:p['dx'] + p['dw']] = True
            available = np.zeros_like(new_solid)
            available[p['rim']:p['base'], p['inner_l']:p['inner_r']] = True
            available &= ~new_solid
            remaining = set(map(tuple, np.argwhere(water)))
            while remaining:
                first = next(iter(remaining))
                stack, component = [first], {first}
                while stack:
                    y, x = stack.pop()
                    for pos in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if pos in remaining and pos not in component:
                            component.add(pos)
                            stack.append(pos)
                remaining -= component
                surface = min(y for y, x in component)
                closure, stack = {first}, [first]
                while stack:
                    y, x = stack.pop()
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (surface <= ny < inp.shape[0] and 0 <= nx < inp.shape[1]
                                and available[ny, nx] and (ny, nx) not in closure):
                            closure.add((ny, nx))
                            stack.append((ny, nx))
                assert closure == component, "unfilled level or leaking pool"
            f = _features(inp, out, p)
            assert f['volume'] > f['old_capacity']
            assert f['changed_water'] and f['construction_matters']
            records[split].append(f)
    a, b = records['train'][:2]
    assert {a['kind'], b['kind']} == {'retain', 'spill'}
    assert {a['width'], b['width']} == {1, 2}
    assert a['total'] != b['total'] and a['height'] != b['height']
    assert a['total'] + a['height'] != b['total'] + b['height']
    assert a['total'] * b['height'] != b['total'] * a['height']
    assert {len(a['sources']), len(b['sources'])} == {1, 2}
    assert any(len(set(f['sources'])) > 1 for f in (a, b))
    assert any(f['lateral'] for f in (a, b))
    for f, pair in zip((a, b), puzzle['train'][:2]):
        p = _parse(_grid(pair['input']))
        if f['kind'] == 'retain':
            assert int(np.nonzero(np.asarray(pair['output']) == 1)[0].min()) > p['new_top']
        src = [s for s in p['lanes'] if s['marked']]
        unmarked = [s for s in p['lanes'] if not s['marked']]
        assert any(u['color'] == s['color'] for u in unmarked for s in src)
    rivals = (lambda f: 2 * f['height'], lambda f: f['total'],
              lambda f: f['height'] * f['total'], lambda f: max(f['height'], f['total']),
              lambda f: f['height'] + len(f['sources']),
              lambda f: f['height'] + max(f['sources']),
              lambda f: f['height'] + f['total'] * f['area'])
    for rival in rivals:
        assert any(rival(f) != f['height'] + f['total'] for f in (a, b))
    for f in records['test']:
        assert f['width'] == 3 and len(f['sources']) == 3
        assert f['kind'] != 'join' or any(t['kind'] == 'join' for t in records['train'])
    return True


def self_test():
    """Bounded 42-seed coverage, count contract, determinism, and mutations."""
    fingerprints, shapes, masks, floors, palettes, volumes, totals = set(), set(), set(), set(), set(), set(), set()
    kinds, sides, source_counts, unmarked_counts, slab_widths, records = set(), set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for split in puzzle for p in puzzle[split]))
        local_shapes, local_masks = set(), set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                f = _features(pair['input'], pair['output'], p)
                records.append(f)
                shapes.add(f['shape']); local_shapes.add(f['shape'])
                masks.add(f['mask']); local_masks.add(f['mask'])
                floors.add(f['floors']); palettes.add(f['colors'])
                volumes.add(f['volume']); totals.add(f['total']); kinds.add(f['kind'])
                sides.add(f['side']); source_counts.add(len(f['sources']))
                unmarked_counts.add(f['unmarked']); slab_widths.add(f['width'])
        assert len(local_shapes) > 1 and len(local_masks) > 1
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(masks) >= 20 and len(floors) >= 40
    assert len(palettes) >= 8 and len(volumes) >= 35 and len(totals) >= 4
    assert kinds == {'retain', 'spill', 'join'} and sides == {False, True}
    assert source_counts == {1, 2, 3} and unmarked_counts == {0, 1, 2}
    assert slab_widths == {1, 2, 3}
    assert sum(f['lateral'] for f in records) >= 100
    assert sum(any(v > 0 for v in f['floors']) for f in records) >= 100
    # Exercise the constructive fallback at extreme counts and all outcomes.
    for counts in ([1], [4], [1, 2], [1, 4], [1, 1, 1], [1, 2, 2]):
        for d in (2, 4):
            for dw in (1, 2, 3):
                for kind in ('retain', 'spill', 'join'):
                    pair = _make_pair(np.random.default_rng(123), counts, d, dw,
                                      kind, _fallback_only=True)
                    assert np.array_equal(solve(pair['input']), pair['output'])
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(7919, train, test)
            assert len(puzzle['train']) == train and len(puzzle['test']) == test
    first, second = generate_puzzle(87, 2, 1), generate_puzzle(87, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for split in first
               for p, q in zip(first[split], second[split]) for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Meaningful wrong answers: missing masonry, lost liquid, a retained marker,
    # unprocessed source, and recolored unmarked payload must all be rejected.
    original = generate_puzzle(17, 2, 1)
    for defect in ('dam', 'liquid', 'marker', 'highlight', 'unmarked'):
        broken = {s: [{k: g.copy() for k, g in pair.items()} for pair in pairs]
                  for s, pairs in original.items()}
        pair = broken['train'][0]
        p = _parse(pair['input'])
        if defect == 'dam':
            pair['output'][p['new_top'], p['dx']] = 0
        elif defect == 'liquid':
            y, x = np.argwhere(pair['output'] == 1)[0]
            pair['output'][y, x] = 0
        elif defect == 'marker':
            pair['output'][p['dest_y'], p['dx']] = 3
        else:
            lane = next(s for s in p['lanes'] if s['marked'] == (defect == 'highlight'))
            yy, xx = zip(*lane['coords'])
            pair['output'][yy, xx] = lane['color'] if defect == 'highlight' else 5
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted mutation: " + defect)
    return {'seeds': 42, 'pairs': len(records), 'shapes': len(shapes),
            'token_masks': len(masks), 'floor_profiles': len(floors),
            'palettes': len(palettes), 'volumes': len(volumes),
            'outcomes': sorted(kinds), 'lateral_pairs': sum(f['lateral'] for f in records)}
