"""Wet plus valves: local motif geometry controls an iterative liquid cascade.

The semantic colors are fixed: 0 air, 1 liquid, 2 reactive material, 3..9 rock.
No parent generator is imported. All solving state is reconstructed from input.
"""
from __future__ import annotations

import numpy as np

_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _CARDINAL:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    stack.append(point)
                    points.append(point)
        part = np.zeros_like(mask, dtype=bool)
        rr, cc = zip(*points)
        part[rr, cc] = True
        result.append(part)
    return result


def _neighbors(mask):
    result = np.zeros_like(mask, dtype=bool)
    result[:-1] |= mask[1:]
    result[1:] |= mask[:-1]
    result[:, :-1] |= mask[:, 1:]
    result[:, 1:] |= mask[:, :-1]
    return result


def _plus_union(mask):
    centers = np.zeros_like(mask, dtype=bool)
    if min(mask.shape) >= 3:
        centers[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                              & mask[2:, 1:-1] & mask[1:-1, :-2]
                              & mask[1:-1, 2:])
    return centers, centers | _neighbors(centers)


def _pour(solid, supply, reverse=False, top_first=False):
    """Unit-volume gravity, with lateral travel only along supported rows.

    The tie switches are validation probes. Family members have complete resting
    levels and the same valve events under all four probes, so no tie is visible.
    """
    h, w = solid.shape
    filled = np.zeros_like(solid, dtype=bool)
    points = sorted(map(tuple, np.argwhere(supply)),
                    key=lambda p: (p[0] if top_first else -p[0],
                                   -p[1] if reverse else p[1]))
    for row, col in points:
        # A previously deposited unit can displace this unit upward in its
        # source column. It cannot displace it through a rock ceiling.
        while row > 0 and filled[row, col]:
            row -= 1
        if solid[row, col] or filled[row, col]:
            raise ValueError("liquid exceeds its accessible capacity")
        while True:
            if row + 1 < h and not (solid[row + 1, col] or filled[row + 1, col]):
                row += 1
                continue
            left = right = col
            while left and not (solid[row, left - 1] or filled[row, left - 1]):
                left -= 1
            while right + 1 < w and not (solid[row, right + 1] or filled[row, right + 1]):
                right += 1
            outlets = [] if row + 1 == h else [c for c in range(left, right + 1)
                                              if not (solid[row + 1, c] or filled[row + 1, c])]
            if not outlets:
                filled[row, col] = True
                break
            col = min(outlets, key=lambda c: (abs(c - col), -c if reverse else c))
            row += 1
    return filled


def _closed_levels(solid, liquid):
    """Each water body equals its entire free basin below its own meniscus."""
    h, w = solid.shape
    for pool in _components(liquid):
        top = int(np.nonzero(pool)[0].min())
        start = tuple(map(int, np.argwhere(pool)[0]))
        seen, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _CARDINAL:
                a, b = r + dr, c + dc
                p = a, b
                if top <= a < h and 0 <= b < w and not solid[p] and p not in seen:
                    seen.add(p)
                    stack.append(p)
        if len(seen) != int(pool.sum()) or any(not pool[p] for p in seen):
            return False
    return True


def _wet_t(mask, union, water):
    touch = _neighbors(water)
    h, w = mask.shape
    for r, c in np.argwhere(mask & ~union):
        if not (0 < r < h - 1 and 0 < c < w - 1):
            continue
        arms = [(r + dr, c + dc) for dr, dc in _CARDINAL if mask[r + dr, c + dc]]
        points = [(r, c)] + arms
        if (len(arms) == 3 and all(not union[p] for p in points)
                and any(touch[p] for p in points)):
            return True
    return False


def _simulate(inp, reverse=False, top_first=False, regions=None, rounds=None):
    if regions is None:
        regions = _components(_plus_union(inp == 2)[1])
    solid = (inp != 0) & (inp != 1)
    liquid = inp == 1
    erased = np.zeros_like(solid)
    trace = []
    for phase in range(len(regions) + 1):
        liquid = _pour(solid, liquid, reverse, top_first)
        touch = _neighbors(liquid)
        active = [i for i, plug in enumerate(regions)
                  if np.any(plug & solid) and np.any(plug & touch)]
        trace.append((solid.copy(), liquid.copy(), tuple(active)))
        if not active or (rounds is not None and phase >= rounds):
            break
        for i in active:
            solid[regions[i]] = False
            erased |= regions[i]
    out = inp.copy()
    out[inp == 1] = 0
    out[erased] = 0
    out[liquid] = 1
    return out, trace, erased


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the visible input and fixed color conventions."""
    inp = _grid(input_grid)
    return _simulate(inp)[0]


def _all_open(inp):
    """Ablation: erase every plus whether wet or dry, then apply gravity."""
    union = _plus_union(inp == 2)[1]
    cleared = inp.copy()
    cleared[union] = 0
    water = _pour((cleared != 0) & (cleared != 1), cleared == 1)
    cleared[cleared == 1] = 0
    cleared[water] = 1
    return cleared


def _global_lowest(inp, erased):
    """Shortcut probe: put water in globally lowest cells of its free component.

    This deliberately ignores the need to fill a pocket before crossing its rim.
    It does not serve as a second solver or supply any generation labels.
    """
    solid = (inp != 0) & (inp != 1) & ~erased
    result = inp.copy()
    result[(inp == 1) | erased] = 0
    for part in _components(~solid):
        volume = int(((inp == 1) & part).sum())
        positions = sorted(map(tuple, np.argwhere(part)), key=lambda p: (-p[0], p[1]))
        for point in positions[:volume]:
            result[point] = 1
    return result


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for split in ('train', 'test') for p in puzzle[split])


def _geometry(rng, tiers, overlap, blocked, regular=False):
    """Build connected retaining masses and independently stepped valve shelves."""
    # True gate tiers are followed, optionally, by a nonreactive T-shaped stop.
    total = tiers + int(blocked)
    h = int(rng.integers(max(18, 10 + 5 * total), 31))
    inner = int(rng.integers(6, 21))
    left, right = map(int, rng.integers(0, 5, size=2))
    if regular:
        left = 3
    w = inner + left + right
    g = np.full((h, w), 3, dtype=np.int8)
    g[:, left:left + inner] = 0
    first = int(rng.integers(4, 7))
    rows = [first + 5 * j for j in range(total)]
    if rows[-1] + 5 > h:
        return None
    main_columns = []
    for j, base in enumerate(rows):
        is_stop = blocked and j == total - 1
        block = bool(overlap and j == min(1, tiers - 1) and not is_stop)
        thickness = 3 if block else 1
        span = 4 if block else 3
        offset = int(rng.integers(1, inner - span))
        col = left + offset + 1
        main_columns.append(col)
        # Platforms may be straight, stepped, or have a sloping shoulder.
        style = 0 if regular else int(rng.choice(3, p=(.6, .2, .2)))
        profile = np.full(inner, base, dtype=int)
        if style:
            walk = (np.cumsum(rng.integers(-1, 2, size=inner)) if style == 1
                    else np.arange(inner) * int(rng.choice((-1, 1))))
            profile += np.clip(walk - walk[offset], -1, 1)
        profile[max(0, offset - 1):min(inner, offset + span + 1)] = base
        for k, top in enumerate(profile):
            g[int(top):int(top) + thickness, left + k] = 3
        if block:
            # Two adjacent centers in a 3x4 patch. Its four corners are solid
            # residual material: dissolving a whole component is incorrect.
            g[base:base + 3, col - 1:col + 3] = 2
            g[base - 1, left + int(regular)] = 3  # routing scenes retain a one-cell side pool
        else:
            g[base, col - 1:col + 2] = 2
            g[base - 1, col] = 2
            if not is_stop:
                g[base + 1, col] = 2
            # Longer arms still leave only the local plus union porous.
            if rng.random() < .75:
                end = min(left + inner, col + int(rng.integers(3, 6)))
                for c in range(col + 2, end):
                    if g[base, c] == 3:
                        g[base, c] = 2
        if not regular and rng.random() < .25 and not is_stop and not block:
            # A diagonal is inert unless it completes another cardinal motif.
            g[base - 1, col + int(rng.choice((-1, 1)))] = 2
    # Ground profiles vary independently from valve coordinates and shelf style.
    low = max(rows[-1] + 4, h - 7)
    if low >= h:
        return None
    floors = rng.integers(low, h + 1, size=inner)
    if rng.random() < .55:
        walk = np.cumsum(rng.integers(-2, 3, size=inner))
        floors = np.clip(low + int(rng.integers(0, h - low + 1)) + walk, low, h)
    for k, floor in enumerate(floors):
        g[int(floor):, left + k] = 3
    # Dry original pluses are shielded inside a retaining mass. They expose the
    # contact condition even when every shelf valve in the flow route opens.
    dry_candidates = []
    rock = g == 3
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if rock[r - 1:r + 2, c - 1:c + 2].all():
                dry_candidates.append((r, c))
    if not dry_candidates:
        return None
    r, c = ((1, 1) if regular else
            dry_candidates[int(rng.integers(len(dry_candidates)))])
    for dr, dc in ((0, 0),) + _CARDINAL:
        g[r + dr, c + dc] = 2
    # The upper reservoir is the free component reached from its top opening.
    free = g == 0
    top_point = (0, left)
    upper = next(part for part in _components(free) if part[top_point])
    return g, upper, first, main_columns, left, inner, rows


def _source(rng, upper, target, style):
    points = np.argwhere(upper)
    v = int(target.sum())
    if v > len(points):
        raise ValueError("insufficient source space")
    anchor = points[int(rng.integers(len(points)))]
    dr, dc = abs(points[:, 0] - anchor[0]), abs(points[:, 1] - anchor[1])
    jitter = rng.random(len(points))
    if style == 0:
        score = dr + dc + jitter
    elif style == 1:
        score = jitter
    elif style == 2:
        score = 3 * dr + .1 * dc + jitter
    elif style == 3:
        score = 3 * dc + .2 * dr + jitter
    else:
        score = (~target[points[:, 0], points[:, 1]]) * 4 + jitter
    chosen = points[np.argsort(score)[:v]]
    source = np.zeros_like(upper)
    source[chosen[:, 0], chosen[:, 1]] = True
    return source


def _paint_rock(rng, inp):
    coords = np.argwhere(inp == 3)
    count = int(rng.integers(1, 5))
    palette = rng.choice(np.arange(3, 10), size=count, replace=False)
    anchors = coords[rng.choice(len(coords), size=count, replace=False)]
    distance = ((coords[:, None] - anchors[None]) ** 2).sum(axis=2)
    result = inp.copy()
    result[coords[:, 0], coords[:, 1]] = palette[distance.argmin(axis=1)]
    return result


def _make_candidate(rng, tiers, overlap=False, blocked=False, routing_witness=False):
    geometry = _geometry(rng, tiers, overlap, blocked, regular=routing_witness)
    if geometry is None:
        return None
    g, upper, first, columns, left, inner, rows = geometry
    solid = (g != 0)
    cutoffs = ([first - 1] if routing_witness else
               list(range(max(1, first - 4), first)))
    rng.shuffle(cutoffs)
    for cutoff in cutoffs:
        target = upper & (np.arange(g.shape[0])[:, None] >= cutoff)
        if not target.any() or not _closed_levels(solid, target):
            continue
        inp = g.copy()
        if routing_witness:
            # A routed pocket has exactly the reservoir's capacity, while a
            # deeper dry pocket beyond a higher rim defeats global bottom fill.
            volume = int(target.sum()) - 1  # one unit remains behind the thick shelf tooth
            col = columns[-1]
            pocket_width = int(rng.integers(4, min(7, inner) + 1))
            start = max(left, min(col - 1, left + inner - pocket_width))
            depth, extra = divmod(volume, pocket_width)
            surface = inp.shape[0] - depth - 1
            if depth < 1 or surface < rows[-1] + 4:
                continue
            inp[rows[-1] + 4:, left:left + inner] = 0
            # Most columns form the rim mass; a distant column is a deeper well.
            inp[surface:, left:left + inner] = 3
            for k in range(pocket_width):
                floor = surface + depth + int(k < extra)
                inp[rows[-1] + 4:floor, start + k] = 0
                inp[floor:, start + k] = 3
            dry_cols = [c for c in range(left, left + inner)
                        if c < start - 1 or c > start + pocket_width]
            if not dry_cols:
                continue
            dry_col = int(rng.choice(dry_cols))
            inp[rows[-1] + 4:, dry_col] = 0
        inp[target] = 1
        try:
            out, trace, erased = _simulate(inp)
        except ValueError:
            continue
        # A tier is a sampling description only; observed events decide acceptance.
        if sum(bool(active) for _, _, active in trace) < tiers:
            continue
        if not all(_closed_levels(s, water) for s, water, _ in trace):
            continue
        union = _plus_union(inp == 2)[1]
        if not np.any(union & ~erased):
            continue
        if blocked and not _wet_t(inp == 2, union, out == 1):
            continue
        if routing_witness and np.array_equal(_global_lowest(inp, erased), out):
            continue
        # Move the supply to a different visible shape inside its upper catchment.
        # This never changes its volume or secretly assigns a receiving basin.
        source = _source(rng, upper, target, int(rng.integers(5)))
        if routing_witness and not source[1, left]:
            # Floating liquid initially touches the high valve, then drains
            # away. This visibly teaches evaluation at rest, rather than on
            # the initial input or during a transient trajectory.
            occupied = list(map(tuple, np.argwhere(source)))
            source[occupied[-1]] = False
            source[1, left] = True
        inp[target] = 0
        inp[source] = 1
        try:
            out, trace, erased = _simulate(inp)
        except ValueError:
            continue
        if not all(_closed_levels(s, water) for s, water, _ in trace):
            continue
        if not np.any((inp == 1).sum(axis=0) != (out == 1).sum(axis=0)):
            continue
        agree = True
        for rev, top in ((True, False), (False, True), (True, True)):
            try:
                other_out, other_trace, other_erased = _simulate(inp, rev, top)
            except ValueError:
                agree = False
                break
            if (not np.array_equal(other_out, out)
                    or not np.array_equal(other_erased, erased)
                    or [a for _, _, a in other_trace] != [a for _, _, a in trace]
                    or not all(_closed_levels(s, water) for s, water, _ in other_trace)):
                agree = False
                break
        if not agree:
            continue
        painted = _paint_rock(rng, inp)
        # Palette substitution changes only impermeable-rock roles.
        answer = out.copy()
        answer[inp >= 3] = painted[inp >= 3]
        return {'input': painted, 'output': answer}
    return None


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(int(seed))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train' and index == 0:
                args = (1, False, True, False)  # opening followed by a T stop
            elif split == 'train' and index == 1:
                args = (2, True, False, True)   # overlaps, cascade, rim routing
            elif split == 'test':
                args = (2 if rng.random() < .5 else 3, bool(rng.integers(2)), False, False)
                if args[0] == 2 and rng.random() < .5:
                    args = (2, True, True, False)  # familiar stopping mechanism after a cascade
            else:
                tiers = int(rng.integers(1, 4))
                args = (tiers, bool(rng.integers(2)), bool(tiers < 3 and rng.random() < .4), False)
            for attempt in range(250):
                pair = _make_candidate(rng, *args)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                pair = _fallback(args, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


# Newly composed finite fallback inputs, not parent pairs or stored answers.
# Their outputs are always computed by the input-only rule. Width padding gives
# at least 15 distinct legal inputs, more than the 14 maximum pair slots.
_FALLBACK_INPUTS = {
    'stop': (
        '33011010000033',
        '33001110100033',
        '33000010100033',
        '33000200000033',
        '33332223333333',
        '33000200000033',
        '33000000000033',
        '33000000000033',
        '33000000200033',
        '33333332223333',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000000000033',
        '33000033333233',
        '33000033332223',
        '33003333333233',
        '33003333333333',
        '33033333333333',
        '33333333333333',
        '33333333333333',
    ),
    'cascade': (
        '323000000011133',
        '222100000011133',
        '323000000001133',
        '333000002000033',
        '333333322233333',
        '333000002000033',
        '333000000000033',
        '333000000000033',
        '333030000000033',
        '333332222333333',
        '333332222333333',
        '333332222333333',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333000000000033',
        '333330000330333',
        '333330000330333',
        '333333333330333',
    ),
    'long': (
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '331111211111133',
        '333332222223333',
        '330000200000033',
        '330000000000033',
        '330000000000033',
        '333000000000033',
        '333222233333333',
        '333222233333333',
        '333222233333333',
        '330000000000033',
        '330000000020033',
        '333333333222233',
        '330000000020033',
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '330000000000033',
        '330000333333333',
        '330003333333333',
        '330032333333333',
        '330322233333333',
        '333332333333333',
        '333333333333333',
        '333333333333333',
    ),
}


def _fallback(args, seen):
    name = 'stop' if args[0] == 1 and args[2] else 'long' if args[0] == 3 else 'cascade'
    base = np.array([[int(c) for c in row] for row in _FALLBACK_INPUTS[name]], dtype=np.int8)
    for padding in range(31 - base.shape[1]):
        inp = np.pad(base, ((0, 0), (padding, 0)), constant_values=3)
        if _key(inp) not in seen:
            return {'input': inp, 'output': solve(inp)}
    raise AssertionError('fallback uniqueness bound exceeded')


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    witnesses = {k: False for k in ('cascade', 'overlap', 'stop', 'route', 'rest_contact', 'levels')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, trace, erased = _simulate(inp)
            assert np.array_equal(out, expected)
            assert int((inp == 1).sum()) == int((out == 1).sum()) > 0
            assert np.argwhere(out == 1)[:, 0].sum() > np.argwhere(inp == 1)[:, 0].sum()
            assert np.array_equal(inp[inp >= 3], out[inp >= 3])
            assert np.array_equal(inp[(inp == 2) & ~erased], out[(inp == 2) & ~erased])
            assert not np.any(erased & (inp != 2))
            centers, union = _plus_union(inp == 2)
            assert erased.any() and np.any(union & ~erased)
            assert all(_closed_levels(solid, water) for solid, water, _ in trace)
            assert all(int(water.sum()) == int((inp == 1).sum()) for _, water, _ in trace)
            assert np.any((inp == 1).sum(axis=0) != (out == 1).sum(axis=0))
            for rev, top in ((True, False), (False, True), (True, True)):
                alternative, other, removed = _simulate(inp, rev, top)
                assert np.array_equal(alternative, out) and np.array_equal(removed, erased)
                assert [a for _, _, a in trace] == [a for _, _, a in other]
                assert all(_closed_levels(s, w) for s, w, _ in other)
            assert not np.array_equal(_simulate(inp, regions=[])[0], out), 'A must affect the answer'
            assert not np.array_equal(_all_open(inp), out), 'dry motifs must remain closed'
            if split == 'train':
                active_phases = sum(bool(a) for _, _, a in trace)
                witnesses['cascade'] |= active_phases >= 2 and not np.array_equal(_simulate(inp, rounds=1)[0], out)
                witnesses['overlap'] |= int(erased.sum()) < 5 * int((centers & erased).sum())
                witnesses['stop'] |= _wet_t(inp == 2, union, out == 1)
                witnesses['route'] |= not np.array_equal(_global_lowest(inp, erased), out)
                witnesses['rest_contact'] |= bool(np.any(union & ~erased & _neighbors(inp == 1)))
                levels = {int(np.nonzero(body)[0].min()) for body in _components(out == 1)}
                witnesses['levels'] |= len(levels) >= 2
    assert all(witnesses.values()), witnesses


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    shapes, layouts, volumes, events, pools, palettes = set(), set(), set(), set(), set(), set()
    for p in puzzles:
        validate_generated_puzzle(p)
        # Omitting selective plus geometry opens wet T-shaped stops and removes
        # the residual corners/extensions of a membrane.
        assert any(not np.array_equal(_simulate(pair['input'],
                    regions=_components(pair['input'] == 2))[0], pair['output'])
                   for pair in p['train'])
        pair = p['train'][0]
        before = pair['input'].copy()
        assert np.array_equal(solve(pair['input']), pair['output'])
        assert np.array_equal(pair['input'], before)
        assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                _, trace, _ = _simulate(inp)
                shapes.add(inp.shape)
                layouts.add(_key(np.where(inp >= 3, 3, inp)))
                volumes.add(int((inp == 1).sum()))
                events.add(sum(bool(a) for _, _, a in trace))
                pools.add(len(_components(out == 1)))
                palettes.add(tuple(map(int, np.unique(inp[inp >= 3]))))
    assert len(shapes) >= 40 and len(layouts) >= 180
    assert len(volumes) >= 20 and {1, 2, 3} <= events
    assert len(pools) >= 3 and len(palettes) >= 20
    assert any(30 in s for s in shapes) and any(min(s) <= 10 for s in shapes)
    # Exhaustive motif-boundary test, and simultaneous adjacent-center regression.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        _, union = _plus_union(mask)
        assert union.sum() == (5 if mask[1, 1] and mask[0, 1] and mask[2, 1]
                              and mask[1, 0] and mask[1, 2] else 0)
    centers, union = _plus_union(np.ones((3, 4), dtype=bool))
    assert centers.sum() == 2 and union.sum() == 8
    boundary = np.array([[0, 1, 0], [1, 1, 1]], dtype=bool)
    assert not _plus_union(boundary)[1].any()
    # Returned-grid validation must reject conservation, rock, and valve errors.
    for kind in (0, 1, 2):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
        pair = bad['test'][0]
        value = (1, 3, 2)[kind]
        mask = pair['output'] >= 3 if kind == 1 else pair['output'] == value
        point = tuple(np.argwhere(mask)[0])
        pair['output'][point] = 0
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted a damaged label')
    fallback_puzzle = {'train': [], 'test': []}
    fallback_seen = set()
    for split, count in (('train', 10), ('test', 4)):
        for index in range(count):
            args = ((1, False, True, False) if split == 'train' and index == 0
                    else (2, True, False, True) if split == 'train' and index == 1
                    else (3, True, False, False))
            pair = _fallback(args, fallback_seen)
            fallback_seen.add(_key(pair['input']))
            fallback_puzzle[split].append(pair)
    validate_generated_puzzle(fallback_puzzle)
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(73, a, b)
        assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
