"""Overlapping views of a sculpture share one finite strand of material.

The repeated two-row grain registers translations. The merged loose material
is counted once and tensioned over the sculpture's digital convex envelope.
Only NumPy and the Python standard library are used; solving reads pixels only.
"""
from __future__ import annotations
import json
import numpy as np

D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ANCHOR = 9


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= x <= 30 for x in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid')
    return a.astype(np.int8, copy=True)


def _components(grid, bg):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid != bg)}
    parts = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, group = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in D8:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        r0, c0 = min(r for r, c in group), min(c for r, c in group)
        parts.append({(r-r0, c-c0): int(grid[r, c]) for r, c in group})
    return parts


def _pose(a, k, reflect):
    a = np.rot90(a, k)
    return np.fliplr(a).copy() if reflect else a.copy()


def _unpose(a, k, reflect):
    if reflect:
        a = np.fliplr(a)
    return np.rot90(a, -k).copy()


def _tight(cells, bg):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    h = max(r for r, c in cells)-r0+1
    w = max(c for r, c in cells)-c0+1
    if max(h, w) > 30:
        raise ValueError('assembled result exceeds ARC dimensions')
    out = np.full((h, w), bg, dtype=np.int8)
    for (r, c), v in cells.items():
        out[r-r0, c-c0] = v
    return out


def _views(parts):
    """Each cutting has a complete two-cell-thick base along its lower edge."""
    views = []
    for part in parts:
        h = max(r for r, c in part)+1
        w = max(c for r, c in part)+1
        if h < 2 or w < 3:
            raise ValueError('cutting too small')
        if not all((r, c) in part for r in (h-2, h-1) for c in range(w)):
            raise ValueError('no two-row base')
        views.append({(r-h+2, c): v for (r, c), v in part.items()})
    return views


def _join(views):
    """Solve all grain-supported shifts, checking their global consistency.

    A match needs at least two entire adjacent base columns, not just a single
    same-colored cell. Every pair with a matching grain constrains its offset.
    Ambiguous/disconnected systems are rejected, never chosen by size or area.
    """
    widths = [max(c for r, c in v)+1 for v in views]
    n = len(views)
    if not 2 <= n <= 6:
        raise ValueError('need 2..6 cuttings')
    adjacency = [[] for _ in views]
    for i in range(n):
        for j in range(i+1, n):
            candidates = []
            for delta in range(-widths[j]+2, widths[i]-1):
                lo, hi = max(0, delta), min(widths[i], delta+widths[j])
                if hi-lo < 2:
                    continue
                if not all(views[i][r, c] == views[j][r, c-delta]
                           for r in (0, 1) for c in range(lo, hi)):
                    continue
                shifted = {(r, c+delta): v for (r, c), v in views[j].items()}
                if any(views[i][p] != shifted[p] for p in views[i].keys() & shifted.keys()):
                    continue
                candidates.append(delta)
            if candidates:
                adjacency[i].append((j, candidates))
                adjacency[j].append((i, [-d for d in candidates]))
    offsets, solutions = {0: 0}, []
    visits = 0

    def search():
        nonlocal visits
        visits += 1
        if visits > 2000:
            raise ValueError('registration search exceeds family bound')
        if len(solutions) > 1:
            return
        if len(offsets) == n:
            union = {}
            for i, part in enumerate(views):
                for (r, c), color in part.items():
                    p = r, c+offsets[i]
                    if p in union and union[p] != color:
                        return
                    union[p] = color
            solutions.append((dict(offsets), union))
            return
        choices = []
        for j in range(n):
            if j in offsets:
                continue
            domain = None
            for i, deltas in adjacency[j]:
                if i in offsets:
                    proposed = {offsets[i]-d for d in deltas}
                    domain = proposed if domain is None else domain & proposed
            if domain is not None:
                if not domain:
                    return
                choices.append((len(domain), j, domain))
        if not choices:
            return
        _, j, domain = min(choices, key=lambda x: (x[0], x[1]))
        for position in sorted(domain):
            offsets[j] = position
            search()
            del offsets[j]

    search()
    if len(solutions) != 1:
        raise ValueError('grain does not determine one connected assembly')
    return solutions[0]


def _route(heights, length):
    """Smallest 45-degree concave ceiling above all front rigid cells.

    The permitted slope order is +1, then 0, then -1. One cell of clearance
    gives v(u) = min(u+1, H+1, K+1-u), H=max height, K=max(u+height).
    The base is included: once beyond it, the outgoing tangent may drop below
    the base level. No input canvas boundary clips a newly assembled output.
    """
    peak = max(heights.values())
    intercept = max(u+h for u, h in heights.items())
    if any(h > u for u, h in heights.items()) or intercept < 2*peak:
        raise ValueError('sculpture crosses the incoming tangent')
    return [(u, min(u+1, peak+1, intercept+1-u)) for u in range(length)]


def _finish(union, bg):
    anchors = [p for p, color in union.items() if color == ANCHOR]
    if len(anchors) != 1 or anchors[0][0] != 0:
        raise ValueError('anchor must be one cell in the upper base row')
    a = anchors[0][1]
    if (-1, a) not in union:
        raise ValueError('missing attached flexible material')
    rope = union[-1, a]
    if rope in (bg, ANCHOR) or any(color == rope for (r, c), color in union.items() if r >= 0):
        raise ValueError('strand color occurs in rigid base')
    slack = {p for p, color in union.items() if color == rope}
    if any(r >= 0 or c > a for r, c in slack) or len(slack) < 2:
        raise ValueError('flexible material must lie behind anchor')
    reached, todo = {(-1, a)}, [(-1, a)]
    while todo:
        r, c = todo.pop()
        for dr, dc in D8:
            q = r+dr, c+dc
            if q in slack and q not in reached:
                reached.add(q)
                todo.append(q)
    if reached != slack:
        raise ValueError('loose strand must be eight-connected after registration')
    # All upper rigid material is ahead. The rear base is the fixed support.
    rigid = {p: color for p, color in union.items() if color != rope}
    if any(r < 0 and c <= a for r, c in rigid):
        raise ValueError('rigid material behind anchor above base')
    lo, far = min(c for r, c in rigid), max(c for r, c in rigid)
    if lo >= a or far <= a or not all((r, c) in rigid for r in (0, 1) for c in range(lo, far+1)):
        raise ValueError('base is not one continuous strip')
    heights = {}
    for c in range(a, far+1):
        heights[c-a] = max(-r for r, cc in rigid if cc == c and r <= 0)
        h = heights[c-a]
        if not all((-v, c) in rigid for v in range(h+1)):
            raise ValueError('rigid columns must be solid')
    route = _route(heights, len(slack))
    result = dict(rigid)
    for u, v in route:
        p = -v, a+u
        if p in rigid:
            raise ValueError('taut strand strikes rigid material')
        result[p] = rope
    raw_count = len(slack)
    return _tight(result, bg), {
        'anchor': a, 'rope': rope, 'slack': slack, 'rigid': rigid,
        'length': raw_count, 'heights': heights, 'route': route,
        'rear': a-lo, 'far': far-a,
        'peak': max(heights.values()),
        'intercept': max(u+h for u, h in heights.items()),
    }


def _analyze(inp):
    inp = _grid(inp)
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[counts.argmax()])
    if bg == ANCHOR or counts.max()*2 <= inp.size:
        raise ValueError('background must be a majority color other than 9')
    found = []
    for k in range(4):
        for reflect in (False, True):
            canonical = _pose(inp, k, reflect)
            try:
                views = _views(_components(canonical, bg))
                offsets, union = _join(views)
                out, info = _finish(union, bg)
            except ValueError:
                continue
            raw_length = sum(sum(v == info['rope'] for v in p.values()) for p in views)
            info.update(background=bg, count=len(views), offsets=offsets,
                        views=views, union=union, raw_length=raw_length,
                        pose=(k, reflect))
            found.append((_unpose(out, k, reflect), info))
    if len(found) != 1:
        raise ValueError('input needs one valid orientation and registration')
    return found[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, examples, generation state, or filesystem access is used."""
    return _analyze(input_grid)[0]


def _pack(parts, bg, rng):
    normalized = []
    for part in parts:
        rr, cc = min(r for r, c in part), min(c for r, c in part)
        p = {(r-rr, c-cc): v for (r, c), v in part.items()}
        normalized.append((p, max(r for r, c in p)+1, max(c for r, c in p)+1))
    area = sum(len(p) for p, h, w in normalized)
    mh, mw = max(h for p, h, w in normalized), max(w for p, h, w in normalized)
    canvases = [(int(rng.integers(max(mh, 10), 31)), int(rng.integers(max(mw, 10), 31))) for _ in range(12)]
    canvases += [(30, 30)]*5
    for h, w in canvases:
        if h*w <= 2*area:
            continue
        grid = np.full((h, w), bg, dtype=np.int8)
        halo = np.zeros((h, w), dtype=bool)
        order = sorted(range(len(parts)), key=lambda i: -len(parts[i]))
        for i in order:
            part, ph, pw = normalized[i]
            feasible = [(r0, c0) for r0 in range(h-ph+1) for c0 in range(w-pw+1)
                        if all(not halo[r+r0, c+c0] for r, c in part)]
            if not feasible:
                break
            r0, c0 = feasible[int(rng.integers(len(feasible)))]
            for (r, c), color in part.items():
                r, c = r+r0, c+c0
                grid[r, c] = color
                halo[max(0, r-1):r+2, max(0, c-1):c+2] = True
        else:
            if len(_components(grid, bg)) == len(parts):
                return grid
    raise ValueError('cuttings did not fit')


def _make_pair(rng, count, long, dented, orientation, redundant=False):
    for _ in range(100):
        bg = int(rng.integers(9))
        palette = list(map(int, rng.permutation([c for c in range(9) if c != bg])))
        rope, colors = palette[0], palette[1:1+int(rng.integers(3, 7))]
        peak = int(rng.integers(2, 5))
        plateau_end = peak+int(rng.integers(0, 4))
        if dented:
            plateau_end = max(plateau_end, peak+2)
        far = plateau_end+peak+int(rng.integers(0, 3))
        rear = int(rng.integers(3, 7))
        heights = {c: max(0, min(c, peak, plateau_end+peak-c)) for c in range(far+1)}
        if dented:
            for c in range(1, far):
                if c not in (peak, plateau_end) and heights[c] and rng.random() < .6:
                    heights[c] = int(rng.integers(heights[c]))
            # Both peaks are present, with a visible intervening recess.
            heights[peak+1] = int(rng.integers(peak))
        else:
            # Both a ramp and a flat-sided block are available in training.
            if rng.random() < .5:
                heights = {c: peak if peak <= c <= plateau_end else 0 for c in range(far+1)}
        length = int(rng.integers(far+3, min(20, far+8))) if long else int(rng.integers(peak+3, far+3))
        if dented and not long:
            length = min(length, far)
        length = min(length, rear*5+1)
        rear = min(rear, length-1)
        slack = {(-1, c) for c in range(-rear, 1)}
        style = int(rng.integers(3))
        # Growth styles vary loose geometry independently of length and contour.
        while len(slack) < length:
            if style == 0:
                choices = [(-v, c) for c in range(-rear, 0) for v in range(2, 6)
                           if (-v, c) not in slack and (1-v, c) in slack]
            else:
                choices = sorted({(r+dr, c+dc) for r, c in slack for dr, dc in D8
                                  if -5 <= r+dr <= -1 and -rear <= c+dc < 0} - slack)
            slack.add(choices[int(rng.integers(len(choices)))])
        cells = {}
        # Nonperiodic two-row colored grain; 2-column words are unique.
        words = []
        for c in range(-rear, far+1):
            choices = [(a, b) for a in colors for b in colors
                       if not words or (words[-1], (a, b)) not in set(zip(words[:-1], words[1:]))]
            if not choices:
                break
            word = choices[int(rng.integers(len(choices)))]
            words.append(word)
            cells[0, c], cells[1, c] = word
        if len(words) != rear+far+1:
            continue
        cells[0, 0] = ANCHOR
        cells.update({p: rope for p in slack})
        obstacle_color = int(rng.choice(colors))
        for c, h in heights.items():
            for v in range(1, h+1):
                cells[-v, c] = obstacle_color
        # First cut overlaps the material heap; later cuts split the rigid front.
        first_end = int(rng.integers(-1, 2))
        first_start_next = max(-rear+1, first_end-int(rng.integers(1, 3)))
        windows = [(-rear, first_end)]
        active = first_start_next
        for j in range(count-2):
            # Distribute cuts across the front rather than using equal slots.
            cut = max(active+2, int(round((j+1)*far/(count-1))))
            cut = min(cut, far-1)
            windows.append((active, cut))
            active = max(active+1, cut-int(rng.integers(1, 3)))
        windows.append((active, far))
        if any(hi-lo+1 < 3 for lo, hi in windows):
            continue
        parts = [{p: v for p, v in cells.items() if lo <= p[1] <= hi} for lo, hi in windows]
        # Some cuttings are wholly redundant, with the same material counted once.
        if count < 5 and (redundant or rng.random() < .35):
            parts.append(dict(parts[int(rng.integers(len(parts)))]))
        try:
            inp = _pack(parts, bg, rng)
            k, reflect = orientation
            inp = _pose(inp, k, reflect)
            out, info = _analyze(inp)
            expected, original = _finish(cells, bg)
            if original['length'] != length:
                raise ValueError('heap growth changed the sampled material budget')
            if not np.array_equal(out, _pose(expected, k, reflect)):
                continue
            if info['raw_length'] <= info['length']:
                continue
            return {'input': inp, 'output': out}
        except ValueError:
            continue
    raise RuntimeError('bounded generation exhausted')


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _check_counts(num_train, num_test):
    for x, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # These modes change demonstrations and transfer, never the rule.
    mode = int(rng.integers(3))
    base_pose = (int(rng.integers(4)), bool(rng.integers(2)))
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            count = 3 if split == 'train' and i < 2 else int(rng.integers(2, 5))
            if mode == 0 and split == 'train':
                count = min(count, 3)
            if mode == 0 and split == 'test':
                count = int(rng.integers(4, 6))
            long = i == 0 if split == 'train' else bool(rng.integers(2))
            dented = (i == 1 or i > 2) if split == 'train' else bool(rng.integers(2))
            orientation = base_pose
            if mode == 1 and split == 'test':
                orientation = ((base_pose[0]+1) % 4, base_pose[1])
            elif mode == 2 or (mode == 0 and split == 'train' and i >= 2):
                orientation = (int(rng.integers(4)), bool(rng.integers(2)))
            if mode == 2 and split == 'test':
                long, dented = True, True
            for _ in range(10):
                pair = _make_pair(rng, count, long, dented, orientation,
                                  redundant=(split == 'train' and i == 1))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input generation exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            expected, info = _analyze(inp)
            assert np.array_equal(expected, out), 'incorrect reconstruction/tension/crop'
            assert not np.array_equal(inp, out)
            # Family-specific conservation: union, not sum of photographed cells.
            assert info['raw_length'] > info['length'], 'deduplication is vacuous'
            assert np.count_nonzero(out == info['rope']) == info['length']
            slopes = [v2-v1 for (u1, v1), (u2, v2) in zip(info['route'], info['route'][1:])]
            assert slopes == sorted(slopes, reverse=True) and set(slopes) <= {-1, 0, 1}
            assert info['route'][0] == (0, 1)
            assert all(v > info['heights'].get(u, -1000) for u, v in info['route'])
            assert set(np.unique(inp)) == set(np.unique(out)), 'material palette changed'
            # Every rigid pixel survives once at its solved position.
            for color in set(info['rigid'].values()):
                assert np.count_nonzero(out == color) == sum(v == color for v in info['rigid'].values())
            if split == 'train':
                train.append(info)
    assert any(i['length'] > i['far']+1 for i in train), 'no outgoing length lesson'
    assert any(i['length'] <= i['far'] for i in train), 'no partial route lesson'
    assert any(_dent_depth(i) > 0 for i in train), 'no envelope-versus-contour lesson'
    assert any(len({tuple(sorted(v.items())) for v in i['views']}) < i['count']
               for i in train), 'no contained-copy absorption lesson'


def _dent_depth(info):
    h, k = info['peak'], info['intercept']
    return max(min(u, h, k-u)-height for u, height in info['heights'].items())


def self_test():
    """Bounded audit of 42 seeds, semantic regressions, and count boundaries."""
    assert _route({0: 0, 1: 0, 2: 2, 3: 2, 4: 0, 5: 0}, 8) == [
        (0, 1), (1, 2), (2, 3), (3, 3), (4, 2), (5, 1), (6, 0), (7, -1)]
    # A deep notch is bridged; a cell-following shortcut would dip here.
    route = _route({0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 0}, 9)
    assert route[3] == (3, 3) and route[4] == (4, 3)
    shapes, outputs, poses, counts, lengths, profiles, heaps, signatures = (set() for _ in range(8))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                out, info = _analyze(p['input'])
                shapes.add(p['input'].shape)
                outputs.add(out.shape)
                poses.add(info['pose'])
                counts.add(info['count'])
                lengths.add(info['length'])
                profiles.add(tuple(info['heights'].values()))
                heaps.add(tuple(sorted((r, c-info['anchor']) for r, c in info['slack'])))
                # Omitting B (leave the reconstructed slack) changes the answer.
                slack_union = _tight(info['union'], info['background'])
                canonical_out, _ = _finish(info['union'], info['background'])
                assert not np.array_equal(slack_union, canonical_out)
                # Omitting A's deduplication extends the output strand.
                wrong_path = _route(info['heights'], info['raw_length'])
                assert len(wrong_path) > len(info['route'])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(outputs) >= 25
    assert len(poses) == 8 and counts >= {2, 3, 4, 5}
    assert len(lengths) >= 10 and len(profiles) >= 40 and len(heaps) >= 80
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    # Repacking and adding a whole observed copy leave physical material intact.
    _, info = _analyze(a['train'][0]['input'])
    expected, _ = _finish(info['union'], info['background'])
    copies = info['views']+[dict(info['views'][0])]
    repacked = _pack(copies, info['background'], np.random.default_rng(901))
    assert np.array_equal(solve(repacked), expected)
    # Check orientation covariance on the same content, not different seeds.
    for k in range(4):
        for reflect in (False, True):
            assert np.array_equal(solve(_pose(repacked, k, reflect)), _pose(expected, k, reflect))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(456, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(77, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged label accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
