"""Cumulative, color-keyed census along a visibly directed chamber stream.

Only the standard library and NumPy are used. The builder computes labels from
its event stream; solve/validation recover everything from the input cells.
"""
from collections import Counter
import itertools
import json
import numpy as np


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _check(np.all((g >= 0) & (g <= 9)), "colors outside 0..9")
    return g.astype(np.int8, copy=False)


def _key(g):
    """Return key-aligned coordinates, with fast right and slow down."""
    candidates = []
    for axis, bottom, right in itertools.product(range(2), range(2), range(2)):
        a = g[::-1] if bottom else g
        a = a[:, ::-1] if right else a
        a = a.T if axis else a
        if min(a.shape) < 8:
            continue
        bg, origin, fast, slow = map(int, (a[-1, -1], a[0, 0], a[0, 1], a[1, 0]))
        if len({bg, origin, fast, slow}) != 4:
            continue
        arm = np.where(np.arange(1, a.shape[0]) % 2, slow, bg)
        if (np.all(a[0, 1:] == fast) and np.array_equal(a[1:, 0], arm)
                and np.all(a[-1, 1:] == bg) and np.all(a[1:, -1] == bg)):
            candidates.append((a, bg, (axis, bottom, right), (origin, fast, slow)))
    _check(len(candidates) == 1, "expected exactly one textured coordinate key")
    return candidates[0]


def _components(g, bg, diagonal=False):
    """Monochromatic components; outline recognition is done separately."""
    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        offsets += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    remaining = set(map(tuple, np.argwhere(g != bg).tolist()))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(g[start])
        pending, cells = [start], []
        while pending:
            r, c = pending.pop()
            cells.append((r, c))
            for dr, dc in offsets:
                p = (r + dr, c + dc)
                if p in remaining and int(g[p]) == color:
                    remaining.remove(p)
                    pending.append(p)
        cells.sort()
        rs, cs = zip(*cells)
        parts.append(dict(color=color, cells=tuple(cells), area=len(cells),
                          box=(min(rs), max(rs)+1, min(cs), max(cs)+1)))
    return parts


def _is_frame(p):
    r0, r1, c0, c1 = p['box']
    h, w = r1-r0, c1-c0
    return (h in (5, 7, 9) and 7 <= w <= 25 and w % 2 == 1
            and p['area'] == 2*h + 2*w - 4
            and all(r in (r0, r1-1) or c in (c0, c1-1) for r, c in p['cells']))


def _inspect(input_grid):
    g = _grid(input_grid)
    a, bg, config, roles = _key(g)
    field = a.copy()
    field[0, :] = bg
    field[:, 0] = bg
    parts = _components(field, bg)
    frames = [p for p in parts if _is_frame(p)]
    objects = [p for p in parts if not _is_frame(p)]
    _check(1 <= len(frames) <= 10, "one through ten chambers required")
    frames.sort(key=lambda f: (f['box'][0], f['box'][2]))
    lanes = {}
    for i, f in enumerate(frames):
        r0, r1, c0, c1 = f['box']
        _check(2 <= r0 < r1 <= a.shape[0]-1 and 2 <= c0 < c1 <= a.shape[1]-1,
               "chambers need an outer background margin")
        lanes.setdefault((r0, r1), []).append(i)
        f['objects'] = []
        f['width'] = c1-c0
    bands = sorted(lanes)
    _check(1 <= len(bands) <= 4, "lane count")
    _check(all(y[0]-x[1] >= 1 for x, y in zip(bands, bands[1:])),
           "overlapping or staggered lane bands")
    groups = [lanes[b] for b in bands]
    for group in groups:
        intervals = [(frames[i]['box'][2], frames[i]['box'][3]) for i in group]
        _check(all(y[0]-x[1] >= 1 for x, y in zip(intervals, intervals[1:])),
               "chambers touch within a lane")
    for obj in objects:
        _check(1 <= obj['area'] <= 8, "object area")
        owners = []
        for i, f in enumerate(frames):
            r0, r1, c0, c1 = f['box']
            if all(r0+2 <= r <= r1-3 and c0+2 <= c <= c1-3 for r, c in obj['cells']):
                owners.append(i)
        _check(len(owners) == 1, "object must be contained with a border moat")
        frames[owners[0]]['objects'].append(obj)
    # Disallow orthogonal contact between different-color objects, too.
    labels = {cell: i for i, obj in enumerate(objects) for cell in obj['cells']}
    for (r, c), i in labels.items():
        for p in ((r+1, c), (r, c+1)):
            _check(p not in labels or labels[p] == i, "distinct objects touch by an edge")
    for f in frames:
        r0, r1, c0, c1 = f['box']
        inside = field[r0+1:r1-1, c0+1:c1-1].copy()
        f['counts'] = Counter(o['color'] for o in f['objects'])
        f['areas'] = Counter()
        for o in f['objects']:
            f['areas'][o['color']] += o['area']
        f['eight'] = Counter(o['color'] for o in _components(inside, bg, diagonal=True))
    d = dict(background=bg, config=config, roles=roles, canonical=a,
             frames=frames, groups=groups, objects=objects)
    values = _values(d)
    _check(all(n <= (f['width']-3)//2 for n, f in zip(values, frames)), "prefix overflow")
    return d


def _values(d, order=None, mode='prefix'):
    """Return each report attached to its original chamber, even for alternatives."""
    frames = d['frames']
    order = list(range(len(frames))) if order is None else order
    totals, values = Counter(), [0]*len(frames)
    global_counts = sum((f['counts'] for f in frames), Counter())
    previous_lane = None
    for i in order:
        f = frames[i]
        color = f['color']
        lane = f['box'][:2]
        if mode == 'lane_reset' and lane != previous_lane:
            totals.clear()
        previous_lane = lane
        addition = f['areas'] if mode == 'area' else f['eight'] if mode == 'eight' else f['counts']
        if mode == 'own_outline':
            addition = Counter({color: addition[color]})
        elif mode == 'frame_count':
            addition = Counter({color: 1})
        totals.update(addition)
        if mode == 'local':
            n = f['counts'][color]
        elif mode == 'global':
            n = global_counts[color]
        elif mode == 'all_colors':
            n = sum(totals.values())
        else:
            n = totals[color]
        values[i] = n
        if mode == 'reset':
            totals[color] = 0
    return values


def _render(widths, colors, counts, bg):
    maximum = max(widths)
    result = np.full((3*len(widths), maximum), bg, dtype=np.int8)
    for i, (w, color, n) in enumerate(zip(widths, colors, counts)):
        _check(0 <= n <= (w-3)//2, "unrepresentable receipt")
        left = (maximum-w)//2
        result[3*i, left:left+w] = color
        result[3*i+2, left:left+w] = color
        result[3*i+1, [left, left+w-1]] = color
        for j in range(n):
            result[3*i+1, left+w-3-2*j] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, learned convention, generation metadata, or stored examples."""
    d = _inspect(input_grid)
    return _render([f['width'] for f in d['frames']],
                   [f['color'] for f in d['frames']], _values(d), d['background'])


def _event_counts(events):
    totals, result = Counter(), []
    for color, specs in events:
        totals.update(c for c, area in specs)
        result.append(totals[color])
    return result


def _partitions(minima):
    """All contiguous feasible lane partitions, without arbitrary population caps."""
    result = []
    def visit(start, groups):
        if start == len(minima):
            result.append(groups)
            return
        if len(groups) == 4:
            return
        for end in range(start+1, len(minima)+1):
            if sum(minima[start:end]) + end-start-1 > 27:
                break
            visit(end, groups+[end-start])
    visit(0, [])
    return result


def _spacings(rng, occupied, count, extent):
    extra = extent - occupied - count - 1
    _check(extra >= 0, "packing budget")
    return (1 + rng.multinomial(extra, rng.dirichlet(np.ones(count+1)))).tolist()


def _shape(rng, area, h, w):
    if area is None:
        pool = np.arange(1, min(8, h*w)+1)
        area = int(rng.choice(pool, p=(1/np.sqrt(pool))/(1/np.sqrt(pool)).sum()))
    kind = int(rng.integers(3))
    if kind == 0 and area <= max(h, w):
        if area <= w and (area > h or rng.random() < .5):
            return tuple((0, c) for c in range(area))
        return tuple((r, 0) for r in range(area))
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(cells) < area:
        frontier = sorted({(r+dr, c+dc) for r, c in cells
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _fill(rng, h, w, specs, bg, diagonal_witness=False):
    if diagonal_witness:
        (color, a), (other, b) = specs
        _check(color == other and a+b <= w and h >= 2, "diagonal witness budget")
        result = np.full((h, w), bg, dtype=np.int8)
        r, c = int(rng.integers(h-1)), int(rng.integers(w-a-b+1))
        result[r, c:c+a] = color
        result[r+1, c+a:c+a+b] = color
        if rng.integers(2):
            result = result[::-1]
        if rng.integers(2):
            result = result[:, ::-1]
        return result.copy()
    for attempt in range(24):
        shapes = [(color, _shape(rng, area, h, w)) for color, area in specs]
        shapes.sort(key=lambda x: -len(x[1]))
        result = np.full((h, w), bg, dtype=np.int8)
        blocked = set()
        for color, cells in shapes:
            sh, sw = 1+max(r for r, c in cells), 1+max(c for r, c in cells)
            options = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)
                       if all((r+dr, c+dc) not in blocked for dr, dc in cells)]
            if not options:
                break
            r0, c0 = options[int(rng.integers(len(options)))]
            for dr, dc in cells:
                r, c = r0+dr, c0+dc
                result[r, c] = color
                blocked.update(((r, c), (r-1, c), (r+1, c), (r, c-1), (r, c+1)))
        else:
            return result
    # A constructive fallback keeps counts and fixed teaching areas intact.
    result = np.full((h, w), bg, dtype=np.int8)
    c = 0
    for color, area in specs:
        area = 1 if area is None else area
        _check(c+area <= w, "fallback budget")
        result[0, c:c+area] = color
        c += area+1
    return result


def _assemble(rng, events, bg, roles, config, teaching=None, widths_override=None):
    reports = _event_counts(events)
    minima = []
    for n, (color, specs) in zip(reports, events):
        m = len(specs)
        minimum = max(7, 2*n+3, 2*m+3)
        if any(area is not None for _, area in specs):
            minimum = max(minimum, sum(1 if a is None else a for c, a in specs)+max(0, m-1)+4)
        minima.append(minimum + int(minimum % 2 == 0))
    partitions = _partitions(minima)
    if teaching is not None:
        partitions = [p for p in partitions if p == [2, 2]]
    if not partitions:
        return None
    populations = partitions[int(rng.integers(len(partitions)))]
    widths = list(minima) if widths_override is None else list(widths_override)
    _check(all(m <= w <= 25 and w % 2 for m, w in zip(minima, widths)), "width proposal")
    group_indices, start = [], 0
    for n in populations:
        group = list(range(start, start+n))
        group_indices.append(group)
        budget = 27 - sum(widths[i] for i in group) - n+1
        if widths_override is None:
            extra = int(rng.integers(min(budget//2, 5)+1))
            for _ in range(extra):
                choices = [i for i in group if widths[i] <= 23]
                if choices:
                    widths[int(rng.choice(choices))] += 2
        start += n
    if teaching is not None and len(set(widths)) == 1:
        shortened = next((i for i in range(len(widths)) if widths[i]-2 >= minima[i]), None)
        if shortened is not None:
            widths[shortened] -= 2
        for group in reversed(group_indices):
            if len(set(widths)) == 1 and sum(widths[i] for i in group)+len(group)-1+2 <= 27:
                widths[group[-1]] += 2
                break
        _check(len(set(widths)) > 1, "teaching widths must show centering")
    heights = [5]*len(populations)
    if teaching == 1:
        heights[0] = 7
    budget = 29 - sum(heights) - len(heights) - 1
    for i in rng.permutation(len(heights)):
        additions = [x for x in (0, 2, 4) if x <= budget and heights[i]+x <= 9]
        add = int(rng.choice(additions))
        heights[i] += add
        budget -= add
    min_h = sum(heights)+len(heights)+1
    min_w = max(sum(widths[i] for i in group)+len(group)+1 for group in group_indices)
    # The content extents exclude the one-cell guide row and guide column.
    h = min_h + int(rng.integers(min(29-min_h, 7)+1))
    w = min_w + int(rng.integers(min(29-min_w, 9)+1))
    a = np.full((h+1, w+1), bg, dtype=np.int8)
    origin, fast, slow = roles
    a[0, 0], a[0, 1:] = origin, fast
    a[1:, 0] = np.where(np.arange(1, h+1) % 2, slow, bg)
    gaps = _spacings(rng, sum(heights), len(heights), h)
    r0 = 1+gaps[0]
    for lane, (height, group) in enumerate(zip(heights, group_indices)):
        cgaps = _spacings(rng, sum(widths[i] for i in group), len(group), w)
        if teaching is not None and lane == 1:
            # Ensure an interleaved fast-coordinate witness across lane bands.
            cgaps[-1] += cgaps[0]-1
            cgaps[0] = 1
        c0 = 1+cgaps[0]
        for j, i in enumerate(group):
            color, specs = events[i]
            width = widths[i]
            a[r0, c0:c0+width] = color
            a[r0+height-1, c0:c0+width] = color
            a[r0:r0+height, c0] = color
            a[r0:r0+height, c0+width-1] = color
            if specs:
                inside = _fill(rng, height-4, width-4, specs, bg,
                               diagonal_witness=teaching == 1 and i == 0)
                a[r0+2:r0+height-2, c0+2:c0+width-2] = inside
            c0 += width+cgaps[j+1]
        r0 += height+gaps[lane+1]
    axis, bottom, right = config
    a = a.T if axis else a
    a = a[::-1] if bottom else a
    a = a[:, ::-1] if right else a
    return dict(input=a.copy(), output=_render(widths, [e[0] for e in events], reports, bg))


def _random_events(rng, n, palette, unmatched, novel=None):
    queries = [int(rng.choice(palette)) for _ in range(n)]
    if novel is not None:
        queries[0] = novel
        if n > 1 and rng.random() < .7:
            queries[-1] = novel
    all_zero = rng.random() < .10 and novel is None
    totals, events = Counter(), []
    for i, color in enumerate(queries):
        m = int(rng.choice(5, p=(.20, .34, .27, .14, .05)))
        specs = []
        for j in range(m):
            eligible = [c for c in palette if totals[c] < 7]
            if eligible and not all_zero and rng.random() < .78:
                c = color if color in eligible and rng.random() < .5 else int(rng.choice(eligible))
            else:
                c = int(rng.choice(unmatched))
            totals[c] += 1
            specs.append((c, None))
        if i == 0 and novel is not None and not any(c == novel for c, a in specs):
            specs = [(novel, None)] + specs[:3]
            totals[novel] += 1
        events.append((color, specs))
    return events


def _alternative_orders(d):
    return {
        'fast_reverse': [i for group in d['groups'] for i in group[::-1]],
        'slow_reverse': [i for group in d['groups'][::-1] for i in group],
        'fast_major': sorted(range(len(d['frames'])),
                             key=lambda i: (d['frames'][i]['box'][2], d['frames'][i]['box'][0])),
    }


def _teaching_evidence(details):
    for mode in ('local', 'global', 'own_outline', 'reset', 'lane_reset', 'area', 'eight', 'all_colors', 'frame_count'):
        _check(any(_values(d) != _values(d, mode=mode) for d in details),
               "training fails to distinguish "+mode)
    for alternative in ('fast_reverse', 'slow_reverse', 'fast_major'):
        _check(any(_values(d) != _values(d, _alternative_orders(d)[alternative]) for d in details),
               "traversal does not affect attached tallies: "+alternative)
    reports = []
    for d in details:
        counts, areas = Counter(), Counter()
        for f in d['frames']:
            counts.update(f['counts'])
            areas.update(f['areas'])
            reports.append((f['color'], counts[f['color']], areas[f['color']]))
    _check(any(c == cc and a == aa and n != nn for c, n, a in reports for cc, nn, aa in reports),
           "same-color area/count collision missing")
    _check(any(n == nn and a != aa for c, n, a in reports for cc, nn, aa in reports),
           "same count at different areas missing")
    _check({0, 1, 2, 3} <= {n for c, n, a in reports}, "unary progression/zero evidence")
    _check(any(len({f['width'] for f in d['frames']}) > 1 for d in details), "centering evidence")
    configs = {d['config'] for d in details}
    _check(all({c[i] for c in configs} == {0, 1} for i in range(3)), "direction factor coverage")


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "example counts")
    seen, details = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = (inp.shape, inp.tobytes())
            _check(key not in seen, "duplicate input")
            seen.add(key)
            d = _inspect(inp)
            expected = _render([f['width'] for f in d['frames']],
                               [f['color'] for f in d['frames']], _values(d), d['background'])
            _check(np.array_equal(out, expected), "wrong cumulative receipt output")
            _check(out.shape == (3*len(d['frames']), max(f['width'] for f in d['frames'])),
                   "receipt dimensions")
            details[split].append(d)
    _teaching_evidence(details['train'])
    tr_configs = {d['config'] for d in details['train']}
    _check(all(d['config'] not in tr_configs for d in details['test']), "held-out key combination")
    tr_colors = {f['color'] for d in details['train'] for f in d['frames']}
    _check(any(f['color'] not in tr_colors for d in details['test'] for f in d['frames']),
           "new test outline category")
    _check(len({d['background'] for d in details['train']}) >= 2, "local background variation")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    nt, nq = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if seed < 0 or not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require nonnegative seed, 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    anchor, novel = map(int, rng.choice(10, 2, replace=False))
    area = int(rng.integers(3, 7))
    cut = int(rng.integers(1, area))
    first = tuple(map(int, rng.integers(0, 2, size=3)))
    second = tuple(1-v for v in first)
    held = [c for c in itertools.product(range(2), repeat=3) if c not in (first, second)]
    puzzle, seen = {'train': [], 'test': []}, set()
    old_bg, old_y = None, None
    for index in range(nt+nq):
        train, teaching = index < nt, index if index < 2 else None
        split = 'train' if train else 'test'
        config = first if index == 0 else second if index == 1 else (
            (first, second)[int(rng.integers(2))] if train else held[int(rng.integers(len(held)))])
        for attempt in range(80):
            excluded_bg = {anchor, old_bg} if teaching is not None else {novel} if not train else set()
            bg = int(rng.choice([c for c in range(10) if c not in excluded_bg]))
            roles = tuple(map(int, rng.choice([c for c in range(10) if c != bg], 3, replace=False)))
            if teaching is not None:
                y = int(rng.choice([c for c in range(10) if c not in (anchor, novel, bg, old_y)]))
                a1, a2, a3 = map(int, rng.integers(1, 6, size=3))
                if teaching == 0:
                    events = [(anchor, [(anchor, area)]), (y, [(anchor, a1)]),
                              (anchor, [(y, a2)]), (y, [(y, a3)])]
                else:
                    events = [(anchor, [(anchor, cut), (anchor, area-cut)]),
                              (y, [(y, a1)]), (y, [(anchor, a2)]), (anchor, [(y, a3)])]
                    if rng.random() < .7:
                        distractor = int(rng.choice([c for c in range(10) if c not in (anchor, y, bg)]))
                        events[1][1].append((distractor, 1))
            else:
                n = int(rng.choice(np.arange(1, 11), p=(.04, .08, .13, .15, .15, .14, .13, .10, .05, .03)))
                pool = [c for c in range(10) if c != bg and (not train or c != novel)]
                k = int(rng.integers(1, 5))
                palette = list(map(int, rng.choice(pool, k, replace=False)))
                use_novel = novel if not train and index == nt else None
                if use_novel is not None and novel not in palette:
                    palette[0] = novel
                unmatched = [c for c in range(10) if c != bg and c not in palette]
                events = _random_events(rng, n, palette, unmatched, use_novel)
            pair = _assemble(rng, events, bg, roles, config, teaching)
            if pair is None:
                continue
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise AssertionError("bounded scene construction exhausted")
        puzzle[split].append(pair)
        seen.add(key)
        if teaching is not None:
            old_bg, old_y = bg, y
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(p):
    return tuple((s, _grid(pair['input']).shape, _grid(pair['input']).tobytes(),
                  _grid(pair['output']).shape, _grid(pair['output']).tobytes())
                 for s in ('train', 'test') for pair in p[s])


def self_test():
    """Bounded variation, semantic witnesses, metamorphic and negative checks."""
    keys, records, masks = set(), [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        keys.add(_puzzle_key(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                _check(np.array_equal(solve(pair['input']), pair['output']), "solve mismatch")
                d = _inspect(pair['input'])
                records.append((split, pair['input'].shape, pair['output'].shape,
                                d['config'], d['background'], len(d['frames']),
                                tuple(len(g) for g in d['groups']),
                                tuple(f['width'] for f in d['frames']),
                                tuple(f['box'][1]-f['box'][0] for f in d['frames']),
                                tuple(_values(d))))
                for o in d['objects']:
                    r0, r1, c0, c1 = o['box']
                    masks.add(tuple((r-r0, c-c0) for r, c in o['cells']))
    _check(len(keys) == 42, "cross-seed repeats")
    _check({r[3] for r in records} == set(itertools.product(range(2), repeat=3)), "all guide configurations")
    _check({r[4] for r in records} == set(range(10)), "background diversity")
    _check(len({r[1] for r in records}) >= 45, "canvas variation")
    _check(len({r[2] for r in records}) >= 20, "receipt size variation")
    _check(len({r[6] for r in records}) >= 15, "lane population variation")
    _check(len({r[7] for r in records}) >= 100, "width schedules")
    _check({h for r in records for h in r[8]} == {5, 7, 9}, "lane heights")
    _check(len({r[5] for r in records}) >= 8, "chamber counts")
    _check(len(masks) >= 50, "object vocabulary")
    _check(max(len(m) for m in masks) == 8, "object area range")
    _check(any(len(r[6]) == 1 for r in records) and any(len(r[6]) == 4 for r in records), "lane extremes")
    _check(any(max(r[7]) > 15 for r in records), "wide chambers")
    _check(any(r[7].index(max(r[7])) > 0 for r in records), "late maximum")
    _check(any(max(r[9]) == 0 for r in records), "all-zero reports")
    _check(any(len(set(r[7])) == 1 for r in records), "equal widths")
    p = generate_puzzle(0, 2, 1)
    for pair in p['train'] + p['test']:
        inp, out = pair['input'], pair['output']
        original = inp.copy()
        for k in range(4):
            for reflect in (False, True):
                transformed = np.rot90(inp, k)
                if reflect:
                    transformed = transformed[:, ::-1]
                _check(np.array_equal(solve(transformed), out), "intrinsic guide equivariance")
        permutation = np.random.default_rng(50).permutation(10)
        _check(np.array_equal(solve(permutation[inp]), permutation[out]), "color-role equivariance")
        _check(np.array_equal(inp, original), "solver mutated input")
    # Explicit extremes and size independence, with fixed visible widths.
    rng = np.random.default_rng(70)
    empty = _assemble(rng, [(2, [])]*10, 0, (8, 5, 6), (0, 0, 0), widths_override=[7]*10)
    _check(empty is not None and solve(empty['input']).shape == (30, 7), "ten empty chambers")
    _check(all(_values(_inspect(empty['input']))[i] == 0 for i in range(10)), "empty prefix")
    for area in (1, 3, 8):
        pair = _assemble(np.random.default_rng(71), [(2, [(2, area)])],
                         0, (8, 5, 6), (0, 0, 0), widths_override=[13])
        expected = np.full((3, 13), 2, dtype=np.int8)
        expected[1, 1:12] = 0
        expected[1, 10] = 2
        _check(np.array_equal(solve(pair['input']), expected), "one object is one dot regardless of area")
    full = _assemble(np.random.default_rng(72), [(2, [(2, None)]*6)],
                     0, (8, 5, 6), (0, 0, 0), widths_override=[15])
    _check(_values(_inspect(full['input'])) == [6], "full capacity count")
    _check(np.array_equal(solve(full['input'])[1], np.array([2,0,2,0,2,0,2,0,2,0,2,0,2,0,2])),
           "full capacity boundary/pitch")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(81, nt, nq)
            _check((len(p['train']), len(p['test'])) == (nt, nq), "explicit counts")
    _check(_puzzle_key(generate_puzzle(82, 7, 3)) == _puzzle_key(generate_puzzle(82, 7, 3)), "determinism")
    for nt, nq in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    damaged = generate_puzzle(83)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted wrong output")
    # An attached interior cell must not be silently counted as part of a frame.
    bad = empty['input'].copy()
    d = _inspect(bad)
    r0, r1, c0, c1 = d['frames'][0]['box']
    bad[r0+1, c0+1] = 2
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("object/frame contact accepted")
    overflow = np.zeros((11, 11), dtype=np.int8)
    overflow[0, 0], overflow[0, 1:] = 8, 5
    overflow[1:, 0] = np.where(np.arange(1, 11) % 2, 6, 0)
    overflow[2, 2:9] = overflow[8, 2:9] = 2
    overflow[2:9, 2] = overflow[2:9, 8] = 2
    overflow[4, 4], overflow[4, 6], overflow[6, 4] = 2, 2, 2
    try:
        solve(overflow)
    except AssertionError as error:
        _check(str(error) == 'prefix overflow', "overflow fixture failed for unrelated reason")
    else:
        raise AssertionError("tally overflow was clipped or accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(keys),
                input_shapes=len({r[1] for r in records}), output_shapes=len({r[2] for r in records}),
                lane_populations=len({r[6] for r in records}), width_schedules=len({r[7] for r in records}),
                object_masks=len(masks), chamber_counts=sorted({r[5] for r in records}),
                widths=sorted({w for r in records for w in r[7]}),
                report_counts=sorted({n for r in records for n in r[9]}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
