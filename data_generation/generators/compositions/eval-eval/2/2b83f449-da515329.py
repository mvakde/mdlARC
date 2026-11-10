"""Latent pinwheel routes, local switch rotation, and hub-directed bead packing.

The cross is a blueprint: only occupied rewired components are revealed.
All conventions are fixed here or recoverable from the input geometry.
"""
from __future__ import annotations

from collections import Counter
import json
import numpy as np

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_PARALLEL = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
_TURNED = _PARALLEL.T.copy()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _cross_mask(shape, center, length):
    mask = np.zeros(shape, dtype=bool)
    y, x = center
    for d in range(1, length + 1):
        mask[y-d, x] = mask[y+d, x] = True
        mask[y, x-d] = mask[y, x+d] = True
    return mask


def _winding(shape, center, length):
    """Four clockwise stepped windings, continued outside the crop.

    B's final geometric recurrence is used as a primitive, with no source
    bitmap or parent whole-puzzle generator. Effective radius is max(2,L).
    """
    h, w = shape
    cy, cx = center
    radius = max(2, length)
    mask = np.zeros(shape, dtype=bool)

    def line(y0, x0, y1, x1):
        y0, y1 = y0 + cy, y1 + cy
        x0, x1 = x0 + cx, x1 + cx
        if y0 == y1 and 0 <= y0 < h:
            lo, hi = max(0, min(x0, x1)), min(w - 1, max(x0, x1))
            if lo <= hi:
                mask[y0, lo:hi+1] = True
        elif x0 == x1 and 0 <= x0 < w:
            lo, hi = max(0, min(y0, y1)), min(h - 1, max(y0, y1))
            if lo <= hi:
                mask[lo:hi+1, x0] = True

    extent = max(cy, cx, h - 1 - cy, w - 1 - cx)
    # A four-stage cycle moves every phase outward by eight cells. Even
    # its inward jog cannot return after this many complete cycles.
    stages = 4 * (extent // 8 + 2)
    for dr, dc in _DIRS:
        y, x = radius * dr, radius * dc
        line(dr, dc, y, x)
        dr, dc = dc, -dr
        for k in range(stages):
            for dy, dx, steps in ((dr, dc, 4*k+2),
                                  (dc, -dr, radius-2), (dr, dc, radius)):
                ny, nx = y + steps*dy, x + steps*dx
                line(y, x, ny, nx)
                y, x = ny, nx
            dr, dc = dc, -dr
    return mask


def _membership_winding(shape, center, length):
    """Closed-form interval check, separate from the accumulated path walk."""
    radius = max(2, length)
    yy, xx = np.indices(shape)
    yy, xx = yy - center[0], xx - center[1]
    bound = max(int(np.abs(yy).max()), int(np.abs(xx).max()))
    mask = np.zeros(shape, dtype=bool)
    for _ in range(4):
        mask |= (xx == 0) & (yy >= -radius) & (yy <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for stage, ((y, x), (dy, dx)) in enumerate(zip(starts, _DIRS[1:] + _DIRS[:1])):
                along = (yy-y)*dy + (xx-x)*dx
                across = (yy-y)*dx - (xx-x)*dy
                run = 16*q + 4*stage + 2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        yy, xx = xx, -yy
    return mask


def _components(mask):
    """Ordered simple paths; clipped one-cell fragments are also legal."""
    cells = set(map(tuple, np.argwhere(mask)))
    adjacency = {p: [(p[0]+dy, p[1]+dx) for dy, dx in _DIRS
                     if (p[0]+dy, p[1]+dx) in cells] for p in cells}
    if any(len(v) > 2 for v in adjacency.values()):
        raise ValueError("branching latent track")
    seen, paths = set(), []
    for start in sorted(cells):
        if start in seen:
            continue
        group, queue = {start}, [start]
        for p in queue:
            for n in adjacency[p]:
                if n not in group:
                    group.add(n)
                    queue.append(n)
        seen.update(group)
        if len(group) == 1:
            paths.append([start])
            continue
        ends = sorted(p for p in group if len(adjacency[p]) == 1)
        if len(ends) != 2:
            raise ValueError("closed track cycle")
        path, previous, current = [ends[0]], None, ends[0]
        while current != ends[1]:
            nxt = next(n for n in adjacency[current] if n != previous)
            path.append(nxt)
            previous, current = current, nxt
        if len(path) != len(group):
            raise ValueError("track ordering failed")
        paths.append(path)
    return paths


def _colored_components(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    groups = []
    while cells:
        start = min(cells)
        cells.remove(start)
        group, queue = {start}, [start]
        for y, x in queue:
            for dy, dx in _DIRS:
                p = (y+dy, x+dx)
                if p in cells:
                    cells.remove(p)
                    group.add(p)
                    queue.append(p)
        groups.append(group)
    return groups


def _bars(mask):
    result = []
    for group in _colored_components(mask):
        if len(group) != 3:
            raise ValueError("switches must be separate straight triples")
        rows, cols = sorted({p[0] for p in group}), sorted({p[1] for p in group})
        if len(rows) == 1 and cols == list(range(cols[0], cols[0]+3)):
            result.append((rows[0], cols[1], 0))  # horizontal input bar
        elif len(cols) == 1 and rows == list(range(rows[0], rows[0]+3)):
            result.append((rows[1], cols[0], 1))  # vertical input bar
        else:
            raise ValueError("bent switch")
    if not result:
        raise ValueError("at least one switch is required")
    return sorted(result)


def _turn(mask, bars, center, length):
    h, w = mask.shape
    answer = mask.copy()
    neighborhoods, turned_cells = set(), set()
    for y, x, axis in bars:
        if not (2 <= y < h-2 and 2 <= x < w-2):
            raise ValueError("switch needs a complete tile and straight exterior ports")
        if max(abs(y-center[0]), abs(x-center[1])) <= length+2:
            raise ValueError("switch is too near the blueprint")
        tile = mask[y-1:y+2, x-1:x+2]
        expected = _PARALLEL if axis == 0 else _PARALLEL.T
        if not np.array_equal(tile, expected):
            raise ValueError("bar does not separate two implicit parallel tracks")
        ports = [(y+dy, x+dx) for dy, dx in
                 (((-1,-2),(-1,2),(1,-2),(1,2)) if axis == 0 else
                  ((-2,-1),(-2,1),(2,-1),(2,1)))]
        if not all(mask[p] for p in ports):
            raise ValueError("switch ports must be straight")
        neighborhood = {(r, c) for r in range(y-1, y+2) for c in range(x-1, x+2)}
        if neighborhoods & neighborhood:
            raise ValueError("switch tiles overlap")
        neighborhoods.update(neighborhood)
        answer[y-1:y+2, x-1:x+2] = _TURNED if axis == 0 else _TURNED.T
        turned_cells.update((y+d, x) if axis == 0 else (y, x+d) for d in (-1, 0, 1))
    paths = _components(answer)
    return answer, paths, neighborhoods, turned_cells


def _parse(a):
    """Recover all palette roles and the blueprint solely from this input."""
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 4:
        raise ValueError("expected background, blueprint, switch, and bead colors")
    bg = int(colors[int(np.argmax(counts))])
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError("sparse input needs a majority background")
    candidates = []
    for track in map(int, colors):
        if track == bg:
            continue
        guide = a == track
        n = int(guide.sum())
        if n % 4 or not 1 <= n//4 <= 6:
            continue
        length = n // 4
        ys = np.flatnonzero(guide.sum(axis=1) == 2*length)
        xs = np.flatnonzero(guide.sum(axis=0) == 2*length)
        if len(ys) != 1 or len(xs) != 1:
            continue
        center = int(ys[0]), int(xs[0])
        y, x = center
        if min(y, x, a.shape[0]-1-y, a.shape[1]-1-x) < length:
            continue
        if not np.array_equal(guide, _cross_mask(a.shape, center, length)):
            continue
        latent = _winding(a.shape, center, length)
        for switch in sorted(set(map(int, colors)) - {bg, track}):
            bead = next(iter(set(map(int, colors)) - {bg, track, switch}))
            try:
                bars = _bars(a == switch)
                rewired, paths, tiles, turned = _turn(latent, bars, center, length)
                beads = set(map(tuple, np.argwhere(a == bead)))
                if not beads or any(not latent[p] or not rewired[p] or p in tiles
                                    or max(abs(p[0]-y), abs(p[1]-x)) <= length+1 for p in beads):
                    continue
                # Beads are separate one-cell objects in the sparse input.
                if any(len(g) != 1 for g in _colored_components(a == bead)):
                    continue
                candidates.append(dict(background=bg, track=track, switch=switch, bead=bead,
                                       center=center, length=length, guide=guide, latent=latent,
                                       bars=bars, rewired=rewired, paths=paths, tiles=tiles,
                                       turned=turned, beads=beads))
            except ValueError:
                continue
    if len(candidates) != 1:
        raise ValueError("input must have one geometric interpretation")
    return candidates[0]


def _distance(p, center):
    return abs(p[0]-center[0]) + abs(p[1]-center[1])


def _settle(paths, beads, center):
    targets, active, counts = set(), set(), []
    details = Counter()
    for path in paths:
        occupied = set(path) & beads
        n = len(occupied)
        counts.append(n)
        if not n:
            details['dormant'] += 1
            continue
        active.update(path)
        d0, d1 = _distance(path[0], center), _distance(path[-1], center)
        if d0 == d1:
            here = occupied
            details['ties'] += 1
        else:
            order = path if d0 < d1 else path[::-1]
            here = set(order[:n])
            details['packing'] += n >= 2
            details['motion'] += here != occupied
            details['single_motion'] += n == 1 and here != occupied
            details['rootless'] += min(d0, d1) > 1
            details['interior_start'] += any(p not in (path[0], path[-1]) for p in occupied)
        targets.update(here)
    if len(targets) != len(beads):
        raise ValueError("beads must lie on the track components")
    return targets, active, counts, details


def _answer(shape, roles, paths=None, reveal_all=False):
    paths = roles['paths'] if paths is None else paths
    targets, active, counts, details = _settle(paths, roles['beads'], roles['center'])
    out = np.full(shape, roles['background'], dtype=np.int8)
    paint = set(map(tuple, np.argwhere(roles['rewired']))) if reveal_all else active
    for p in paint:
        out[p] = roles['track']
    for p in roles['turned']:
        out[p] = roles['switch']
    for p in targets:
        out[p] = roles['bead']
    return out, targets, active, counts, details


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No training, seed, generation state, or stored examples are consulted."""
    a = _grid(input_grid)
    return _answer(a.shape, _parse(a))[0]


def _switch_candidates(mask, center, length):
    h, w = mask.shape
    result = []
    for y in range(2, h-2):
        for x in range(2, w-2):
            if max(abs(y-center[0]), abs(x-center[1])) <= length+2:
                continue
            tile = mask[y-1:y+2, x-1:x+2]
            for axis in (0, 1):
                if np.array_equal(tile, _PARALLEL if axis == 0 else _PARALLEL.T):
                    result.append((y, x, axis))
    return result


def _sample_switches(rng, latent, center, length, wanted):
    candidates = _switch_candidates(latent, center, length)
    rng.shuffle(candidates)
    chosen = []
    for bar in candidates:
        if any(max(abs(bar[0]-p[0]), abs(bar[1]-p[1])) < 4 for p in chosen):
            continue
        try:
            result = _turn(latent, chosen + [bar], center, length)
        except ValueError:
            continue
        chosen.append(bar)
        if len(chosen) == wanted:
            return chosen, result
    return None


def _one_bead_targets(paths, center):
    result = {}
    for path in paths:
        d0, d1 = _distance(path[0], center), _distance(path[-1], center)
        for p in path:
            result[p] = p if d0 == d1 else (path[0] if d0 < d1 else path[-1])
    return result


def _sample_beads(rng, roles, mode, require_tie, require_single):
    paths, center = roles['paths'], roles['center']
    length = roles['length']
    eligible = {p for path in paths for p in path if p not in roles['tiles']
                and roles['latent'][p]
                and max(abs(p[0]-center[0]), abs(p[1]-center[1])) > length+1}
    before = _one_bead_targets(_components(roles['latent']), center)
    after = _one_bead_targets(paths, center)
    witness = sorted(p for p in eligible if before[p] != after[p])
    if not witness:
        return None
    eligible_by_path = [sorted(set(p) & eligible) for p in paths]
    tie_indices = [i for i, path in enumerate(paths)
                   if eligible_by_path[i] and len(path) > 1
                   and _distance(path[0], center) == _distance(path[-1], center)]
    packing_indices = [i for i, path in enumerate(paths)
                       if len(eligible_by_path[i]) >= 3
                       and _distance(path[0], center) != _distance(path[-1], center)]
    if not packing_indices or (require_tie and not tie_indices):
        return None
    for _ in range(20):
        chosen = {witness[int(rng.integers(len(witness)))]}

        def add_from(pool, wanted):
            for j in rng.permutation(len(pool)):
                p = pool[int(j)]
                if p not in chosen and all(_distance(p, q) > 1 for q in chosen):
                    chosen.add(p)
                    wanted -= 1
                    if wanted == 0:
                        return True
            return False

        collision = int(rng.choice(packing_indices))
        # At least two beads on one non-tied thread expose contiguous packing.
        existing = len(chosen & set(paths[collision]))
        if not add_from(eligible_by_path[collision], max(1, 2-existing)):
            continue
        if require_tie and not add_from(eligible_by_path[int(rng.choice(tie_indices))], 1):
            continue
        if require_single:
            single_pool = [i for i in packing_indices if not (chosen & set(paths[i]))]
            if not single_pool:
                continue
            i = int(rng.choice(single_pool))
            moving = [p for p in eligible_by_path[i] if p != after[p]]
            if not add_from(moving, 1):
                continue
        if mode == 'dense':
            # Excite several distinct phases and clips, sometimes every route
            # that has room for a remote bead. Counts do not depend on colors.
            for i in rng.permutation(len(paths)):
                i = int(i)
                if not (chosen & set(paths[i])) and rng.random() < .85:
                    add_from(eligible_by_path[i], 1)
                if eligible_by_path[i] and rng.random() < .35:
                    add_from(eligible_by_path[i], 1)
        elif mode == 'moderate':
            add_from(sorted(eligible), int(rng.integers(1, 5)))
        elif mode == 'sparse':
            if rng.random() < .4:
                add_from(sorted(eligible), 1)
        roles['beads'] = chosen
        targets, active, _, details = _settle(paths, chosen, center)
        ignored = _settle(_components(roles['latent']), chosen, center)[0]
        dormant = set(map(tuple, np.argwhere(roles['rewired']))) - active
        if (targets != ignored and details['packing'] and details['motion']
                and (not require_tie or details['ties'])
                and (not require_single or details['single_motion'])
                and (mode != 'sparse' or len(dormant) >= 5)):
            return chosen
    return None


def _make_pair(rng, length, switch_count, mode, crop, palette, require_tie=False,
               require_single=False):
    for _ in range(100):
        if crop == 'compact':
            shape = tuple(map(int, rng.integers(max(15, 2*length+7), 23, size=2)))
        elif crop == 'large':
            shape = tuple(map(int, rng.integers(25, 31, size=2)))
        else:
            shape = tuple(map(int, rng.integers(max(17, 2*length+7), 31, size=2)))
        if crop == 'edge':
            center = tuple(int(rng.choice([length, side-1-length])) if rng.random() < .6
                           else int(rng.integers(length, side-length)) for side in shape)
            # Every crop-shift query touches at least one side; the other
            # coordinate can still vary over its full feasible range.
            axis = int(rng.integers(2))
            center = list(center)
            center[axis] = int(rng.choice([length, shape[axis]-1-length]))
            center = tuple(center)
        else:
            pad = length + 3
            center = tuple(int(rng.integers(pad, side-pad)) for side in shape)
        latent = _winding(shape, center, length)
        selected = _sample_switches(rng, latent, center, length, switch_count)
        if selected is None:
            continue
        bars, (rewired, paths, tiles, turned) = selected
        bg, track, switch, bead = palette
        roles = dict(background=bg, track=track, switch=switch, bead=bead,
                     center=center, length=length, latent=latent, rewired=rewired,
                     paths=paths, tiles=tiles, turned=turned, bars=bars)
        beads = _sample_beads(rng, roles, mode, require_tie, require_single)
        if beads is None:
            continue
        a = np.full(shape, bg, dtype=np.int8)
        a[_cross_mask(shape, center, length)] = track
        for y, x, axis in bars:
            for d in (-1, 0, 1):
                a[(y, x+d) if axis == 0 else (y+d, x)] = switch
        for p in beads:
            a[p] = bead
        out = _answer(shape, roles)[0]
        try:
            parsed = _parse(a)
        except ValueError:
            continue
        if np.array_equal(out, _answer(shape, parsed)[0]):
            return {'input': a, 'output': out}
    raise RuntimeError("bounded construction exhausted for a requested demonstration")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 2, 10, 'num_train')
    ntest = _count(num_test, 1, 4, 'num_test')
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    vary_palette = bool(rng.random() < .65)
    shift = str(rng.choice(['scale', 'switches', 'crop']))
    puzzle, seen = {'train': [], 'test': []}, set()
    train_lengths = [2, 3]
    train_lengths += [int(rng.choice([1, 2, 3])) if shift == 'scale'
                      else int(rng.integers(1, 6)) for _ in range(ntrain-2)]
    for split, total in (('train', ntrain), ('test', ntest)):
        for index in range(total):
            if split == 'train':
                length = train_lengths[index]
                switches = 1 if index == 0 else (2 if index == 1 else int(rng.integers(1, 4)))
                mode = 'dense' if index == 0 else ('sparse' if index == 1 else str(rng.choice(['sparse','moderate','dense'])))
                crop = 'compact' if shift == 'scale' else 'normal'
                tie, single = index == 0, index == 1
            else:
                length = 4 if shift == 'scale' else int(rng.choice(train_lengths))
                switches = int(rng.integers(4, 6)) if shift == 'switches' else int(rng.integers(1, 4))
                mode = str(rng.choice(['sparse', 'moderate', 'dense']))
                crop = 'large' if shift == 'scale' else ('edge' if shift == 'crop' else 'normal')
                tie, single = False, False
            for _ in range(20):
                pal = tuple(map(int, rng.choice(10, 4, replace=False))) if vary_palette else palette
                pair = _make_pair(rng, length, switches, mode, crop, pal, tie, single)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    # Demonstration order does not encode any instruction.
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, coverage = set(), Counter()
    lengths, switch_counts, margins, shapes = {s: [] for s in puzzle}, {s: [] for s in puzzle}, {s: [] for s in puzzle}, {s: [] for s in puzzle}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            r = _parse(a)
            assert np.array_equal(r['latent'], _membership_winding(a.shape, r['center'], r['length']))
            predicted, targets, active, counts, details = _answer(a.shape, r)
            assert np.array_equal(b, predicted), 'incorrect active thread or bead packing'
            assert b[r['center']] == r['background']
            assert np.count_nonzero(b == r['bead']) == len(r['beads'])
            assert np.count_nonzero(b == r['switch']) == 3*len(r['bars'])
            assert all(b[p] == r['switch'] for p in r['turned'])
            assert set(map(tuple, np.argwhere(b == r['track']))) == active - targets
            assert all(r['latent'][p] for p in r['beads'])
            # Bead conservation holds per connected route, not just globally.
            for path, n in zip(r['paths'], counts):
                assert sum(b[p] == r['bead'] for p in path) == n
            before_targets = _settle(_components(r['latent']), r['beads'], r['center'])[0]
            assert targets != before_targets, 'switch must affect bead destinations'
            # A blueprint alone cannot account for the revealed distant cells.
            assert any(p not in r['tiles'] and not r['guide'][p] for p in active)
            if split == 'train':
                coverage.update(details)
                coverage['switch_witness'] += targets != before_targets
                coverage['selective_witness'] += np.any(r['rewired'] & ~np.isin(b, [r['track'], r['bead']]))
            lengths[split].append(r['length'])
            switch_counts[split].append(len(r['bars']))
            y, x = r['center']; h, w = a.shape; length = r['length']
            margins[split].append(min(y-length, x-length, h-1-y-length, w-1-x-length))
            shapes[split].append(a.shape)
    assert {2, 3} <= set(lengths['train'])
    assert all(coverage[k] for k in ('packing', 'single_motion', 'ties', 'dormant', 'switch_witness', 'selective_witness'))
    # At least one visible shift is required; no hidden sampler mode is used.
    scale = min(lengths['test']) > max(lengths['train']) and all(
        min(s) > max(max(t) for t in shapes['train']) for s in shapes['test'])
    routing = min(switch_counts['test']) > max(switch_counts['train'])
    crop = min(margins['test']) < min(margins['train'])
    assert scale or routing or crop, 'missing visible train/test shift'


def _signature(puzzle):
    return tuple((s, _key(p['input']), _key(p['output'])) for s in ('train','test') for p in puzzle[s])


def self_test():
    """Bounded geometry, metamorphic, corruption, and 42-seed diversity checks."""
    for length in range(1, 7):
        for shape, center in (((30, 30), (length, length)),
                               ((23, 29), (11, 14)), ((30, 17), (29-length, length))):
            walked = _winding(shape, center, length)
            assert np.array_equal(walked, _membership_winding(shape, center, length))
            _components(walked)
            big = _membership_winding((90,90), (45,45), length)
            y, x = 45-center[0], 45-center[1]
            assert np.array_equal(walked, big[y:y+shape[0], x:x+shape[1]])
    assert np.array_equal(_winding((30,30),(14,15),1), _winding((30,30),(14,15),2))
    # Length six is in the solver's domain even though the random puzzle
    # planner currently samples no scale above five. Exercise it with actual
    # switches, activation and beads, as well as the pure geometry checks.
    long_pair = _make_pair(np.random.default_rng(777), 6, 3, 'moderate', 'normal', (0,3,8,2))
    assert np.array_equal(solve(long_pair['input']), long_pair['output'])
    # Endpoint rank, contiguous inward order, rootless packing and ties.
    p = [(1,0),(1,1),(1,2),(1,3),(1,4)]
    assert _settle([p], {(1,2),(1,4)}, (0,0))[0] == {(1,0),(1,1)}
    assert _settle([p], {(1,2)}, (0,2))[0] == {(1,2)}
    try:
        _components(np.ones((2,2), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError('cycle accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = Counter()
    shapes, layouts, palettes, lengths, bars, bead_counts, activated_counts = set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                r = _parse(a)
                assert np.array_equal(solve(a.copy()), b)
                for turns in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(a, turns)), np.rot90(b, turns))
                # A global color permutation must commute with solving.
                perm = np.arange(9, -1, -1, dtype=np.int8)
                assert np.array_equal(solve(perm[a]), perm[b])
                _, active, counts, detail = _settle(r['paths'], r['beads'], r['center'])
                features.update(detail)
                features['horizontal'] += any(p[2] == 0 for p in r['bars'])
                features['vertical'] += any(p[2] == 1 for p in r['bars'])
                features['edge'] += min(r['center'][0]-r['length'], r['center'][1]-r['length'], a.shape[0]-1-r['center'][0]-r['length'], a.shape[1]-1-r['center'][1]-r['length']) == 0
                shapes.add(a.shape)
                layouts.add((a.shape, r['center'], r['length'], tuple(r['bars']), tuple(sorted(r['beads']))))
                palettes.add(tuple(r[k] for k in ('background','track','switch','bead')))
                lengths.add(r['length']); bars.add(len(r['bars'])); bead_counts.add(len(r['beads']))
                activated_counts.add(sum(n > 0 for n in counts))
    assert len({_signature(p) for p in puzzles}) == 42
    assert len(shapes) >= 45 and len(layouts) >= 180 and len(palettes) >= 50
    assert {1,2,3,4,5} <= lengths and {1,2,3,4,5} <= bars
    assert len(bead_counts) >= 6 and len(activated_counts) >= 3
    assert all(features[k] > 0 for k in ('packing','single_motion','ties','dormant','rootless','interior_start','horizontal','vertical','edge'))
    assert _signature(generate_puzzle(8675309, 5, 2)) == _signature(generate_puzzle(8675309, 5, 2))
    for ntrain in range(2,11):
        for ntest in range(1,min(4,ntrain-1)+1):
            validate_generated_puzzle(generate_puzzle(101, ntrain, ntest))
    for nt, nq in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = {s: [{k: v.copy() for k,v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('changed label accepted')
    return {'seeds':42, 'pairs':sum(len(p[s]) for p in puzzles for s in p),
            'shapes':len(shapes), 'layouts':len(layouts), 'palettes':len(palettes),
            'lengths':sorted(lengths), 'switch_counts':sorted(bars),
            'bead_counts':sorted(bead_counts), 'active_route_counts':sorted(activated_counts),
            'features':{k:int(v) for k,v in features.items()}}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
