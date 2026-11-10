"""Dock-and-prime chambers: stable packing assembles a paint-carrying network.

Only standard-library modules and NumPy are used.  The solver decodes the edge
rails, hollow rectangles and ports from the query input, without demonstrations.
"""
from collections import deque
import json

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("Require integer counts: 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int64, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _span(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b, margin=0):
    return (max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h']) + margin
            and max(a['c'], b['c']) < min(a['c'] + a['w'], b['c'] + b['w']) + margin)


def _pack(objects, width):
    """Boundary recurrence; a same-class follower cannot overtake a blocker."""
    out = [dict(o) for o in objects]
    for d in (0, 1):
        order = [i for i, o in enumerate(objects) if o['d'] == d]
        order.sort(key=lambda i: (objects[i]['c'] if d == 0 else
                                 -objects[i]['c'] - objects[i]['w'], objects[i]['r']))
        settled = []
        for i in order:
            o = out[i]
            blockers = [p for p in settled if _span(o, p)]
            if d == 0:
                o['c'] = max((p['c'] + p['w'] for p in blockers), default=1)
            else:
                o['c'] = min((p['c'] for p in blockers), default=width - 1) - o['w']
            if not 1 <= o['c'] <= width - 1 - o['w']:
                raise ValueError("A packing chain exceeds the interior track")
            settled.append(o)
    for i, a in enumerate(out):
        for b in out[i + 1:]:
            if _overlap(a, b):
                raise ValueError("Settled footprints overlap")
            if a['d'] != b['d'] and _span(a, b):
                gap = max(a['c'], b['c']) - min(a['c'] + a['w'], b['c'] + b['w'])
                if gap < 1:
                    raise ValueError("Opposite banks need a blank separating column")
    return out


def _unit_pack(objects, shape):
    """Validation by repeated one-cell sliding, independently of the recurrence."""
    out = [dict(o) for o in objects]
    height, width = shape
    for d, delta in ((0, -1), (1, 1)):
        group = [o for o in out if o['d'] == d]
        occupied = np.zeros(shape, dtype=bool)
        for o in group:
            occupied[o['r']:o['r']+o['h'], o['c']:o['c']+o['w']] = True
        for _ in range(width * max(1, len(group)) + 1):
            moved = False
            for o in group:
                r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
                occupied[r:r+h, c:c+w] = False
                candidate = c + delta
                if (1 <= candidate <= width-1-w
                        and not np.any(occupied[r:r+h, candidate:candidate+w])):
                    o['c'] = candidate
                    moved = True
                occupied[r:r+h, o['c']:o['c']+w] = True
            if not moved:
                break
        else:
            raise AssertionError("Unit sliding did not converge")
    return out


def _graph(objects):
    """Only abutting side ports transmit; frame or diagonal contact does not."""
    graph = [set() for _ in objects]
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if a['c'] + a['w'] == b['c']:
                right = {a['r'] + y for y in a['right']}
                left = {b['r'] + y for y in b['left']}
                if right & left:
                    graph[i].add(j)
                    graph[j].add(i)
    return graph


def _components(graph):
    remaining = set(range(len(graph)))
    result = []
    while remaining:
        root = min(remaining)
        group, queue = {root}, [root]
        remaining.remove(root)
        for i in queue:
            for j in sorted(graph[i] & remaining):
                remaining.remove(j)
                group.add(j)
                queue.append(j)
        result.append(group)
    return result


def _slots(o, width):
    result = []
    if o['c'] == 1:
        result.extend((0, o['r'] + y) for y in sorted(o['left']))
    if o['c'] + o['w'] == width - 1:
        result.extend((1, o['r'] + y) for y in sorted(o['right']))
    return result


def _paint(objects, shape, sources):
    graph = _graph(objects)
    paints = {}
    for group in _components(graph):
        colors = {sources[s] for i in group for s in _slots(objects[i], shape[1])
                  if s in sources}
        if len(colors) > 1:
            raise ValueError("Conflicting paint sources on one assembled network")
        if colors:
            color = colors.pop()
            paints.update((i, color) for i in group)
    return paints, graph


def _render(shape, bg, frames, wire, sources, objects, paints=None):
    a = np.full(shape, bg, dtype=np.int64)
    a[:, 0], a[:, -1] = frames
    for (side, row), color in sources.items():
        a[row, 0 if side == 0 else -1] = color
    paints = {} if paints is None else paints
    occupied = np.zeros(shape, dtype=bool)
    for i, o in enumerate(objects):
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        if np.any(occupied[r:r+h, c:c+w]):
            raise ValueError("Overlapping chambers")
        occupied[r:r+h, c:c+w] = True
        a[r:r+h, c:c+w] = frames[o['d']]
        a[r+1:r+h-1, c+1:c+w-1] = paints.get(i, bg)
        for y in o['left']:
            a[r+y, c] = wire
        for y in o['right']:
            a[r+y, c+w-1] = wire
    return a


def _decode(input_grid):
    """Recover all conventions and geometry from one visible grid."""
    raw = _grid(input_grid)
    counts = np.bincount(raw.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= raw.size:
        raise ValueError("Input background must occupy more than half the grid")
    horizontal = np.all(raw[:, [0, -1]] != bg)
    vertical = np.all(raw[[0, -1], :] != bg)
    if horizontal == vertical:
        raise ValueError("Exactly one pair of opposite borders must be rails")
    transposed = bool(vertical)
    a = raw.T.copy() if transposed else raw
    height, width = a.shape
    frames = (int(a[0, 0]), int(a[0, -1]))
    if frames[0] == frames[1] or bg in frames:
        raise ValueError("Rails must have two distinct frame identities")
    if not (np.all(a[[0, -1], 1:-1] == bg)
            and a[-1, 0] == frames[0] and a[-1, -1] == frames[1]):
        raise ValueError("Rail ends and the other two perimeter edges must be clear")
    covered = a == bg
    covered[:, [0, -1]] = True
    objects, port_colors = [], set()
    for r in range(1, height-1):
        for c in range(1, width-1):
            if covered[r, c]:
                continue
            frame = int(a[r, c])
            if frame not in frames:
                raise ValueError("Unexpected pixel outside a hollow frame")
            end = c
            while end < width-1 and a[r, end] == frame:
                end += 1
            w = end-c
            if w < 3:
                raise ValueError("A chamber must have a nonempty rectangular cavity")
            bottom = r+2
            while bottom < height-1 and not np.all(a[bottom, c:end] == frame):
                bottom += 1
            if bottom >= height-1:
                raise ValueError("Missing chamber bottom")
            h = bottom-r+1
            if not np.all(a[r+1:bottom, c+1:end-1] == bg):
                raise ValueError("Input cavities must be empty")
            if np.any(covered[r:bottom+1, c:end] & (a[r:bottom+1, c:end] != bg)):
                raise ValueError("Overlapping frame decomposition")
            ports = []
            for col in (c, end-1):
                side = a[r+1:bottom, col]
                if np.any(side == bg):
                    raise ValueError("Side boundaries must be sealed by frame or ports")
                offsets = set(int(y)+1 for y in np.flatnonzero(side != frame))
                port_colors.update(int(side[y-1]) for y in offsets)
                ports.append(offsets)
            if not any(ports):
                raise ValueError("Each chamber must have at least one port")
            covered[r:bottom+1, c:end] = True
            objects.append(dict(r=r, c=c, h=h, w=w, d=frames.index(frame),
                                left=ports[0], right=ports[1]))
    if not objects or len(port_colors) != 1:
        raise ValueError("All side ports must share one identifiable conductor color")
    wire = port_colors.pop()
    if wire in (bg, *frames):
        raise ValueError("Conductor must differ from background and both frames")
    for i, o in enumerate(objects):
        if any(_overlap(o, p, margin=1) for p in objects[i+1:]):
            raise ValueError("Input chambers require a one-cell gap")
    sources = {}
    for side in (0, 1):
        for row in range(1, height-1):
            color = int(a[row, 0 if side == 0 else -1])
            if color == frames[side]:
                continue
            if color in (bg, wire, *frames):
                raise ValueError("Source dots must differ from structural colors")
            sources[side, row] = color
    if not sources:
        raise ValueError("At least one visible paint source is required")
    return a, bg, frames, wire, sources, objects, transposed


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, frames, wire, sources, objects, transposed = _decode(input_grid)
    settled = _pack(objects, a.shape[1])
    paints, _ = _paint(settled, a.shape, sources)
    result = _render(a.shape, bg, frames, wire, sources, settled, paints)
    return result.T.copy() if transposed else result


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi+1))


def _object(r, c, h, w, d, left, right):
    return dict(r=r, c=c, h=h, w=w, d=d, left=set(left), right=set(right))


def _reflect(objects, width):
    return [dict(o, c=width-o['c']-o['w'], d=1-o['d'],
                 left=set(o['right']), right=set(o['left'])) for o in objects]


def _network_layout(rng, length, primary):
    """A tall root branches into an upper serial route and a lower dry tail."""
    upper_h, lower_h = _ri(rng, 4, 6), _ri(rng, 4, 6)
    root_h = upper_h+lower_h+_ri(rng, 2, 4)
    root_r = _ri(rng, 1, 4)
    height = _ri(rng, root_r+root_h+1, 30)
    sizes = [_ri(rng, 3, 5) for _ in range(length+1)]
    leading, trailing = _ri(rng, 1, 3), _ri(rng, 1, 3)
    gaps = [_ri(rng, 1, 2) for _ in range(length)]
    minimum = 2+leading+trailing+sum(sizes)+sum(gaps)
    # Shrink only sampled widths/margins until the requested graph fits.
    while minimum > 30:
        candidates = [i for i, w in enumerate(sizes) if w > 3]
        if candidates:
            sizes[candidates[_ri(rng, 0, len(candidates)-1)]] -= 1
        elif leading > 1:
            leading -= 1
        elif trailing > 1:
            trailing -= 1
        else:
            gaps[gaps.index(2)] = 1
        minimum = 2+leading+trailing+sum(sizes)+sum(gaps)
    width = _ri(rng, minimum, 30)
    cols = [1+leading]
    for i in range(length):
        cols.append(cols[-1]+sizes[i]+gaps[i])
    upper_y = root_r+2
    lower_r = root_r+root_h-lower_h
    lower_y = lower_r+_ri(rng, 1, lower_h-2)
    source_y = root_r+_ri(rng, 1, root_h-2)
    root = _object(root_r, cols[0], root_h, sizes[0], 0,
                   [source_y-root_r], [upper_y-root_r, lower_y-root_r])
    objects = [root]
    for i in range(1, length):
        row = root_r+_ri(rng, 0, 1)
        h = upper_h if length == 3 else _ri(rng, 4, upper_h)
        objects.append(_object(row, cols[i], h, sizes[i], 0,
                               [upper_y-row], [upper_y-row]))
    # Lower branch and its adjacent, visibly mismatched dry follower.
    objects.append(_object(lower_r, cols[1], lower_h, sizes[1], 0,
                           [lower_y-lower_r], [lower_y-lower_r]))
    alternatives = [y for y in range(1, lower_h-1) if y != lower_y-lower_r]
    dry_y = alternatives[_ri(rng, 0, len(alternatives)-1)]
    objects.append(_object(lower_r, cols[2], lower_h, sizes[2], 0, [dry_y], [dry_y]))
    # Opposite-frame chamber shares the same hollow payload geometry as a child.
    objects.append(_object(root_r, cols[-1], upper_h, sizes[-1], 1, [2], [2]))
    if primary:
        objects = _reflect(objects, width)
    return (height, width), objects


def _chain_layout(rng, primary):
    h = _ri(rng, 4, 6)
    step, base = h-1, _ri(rng, 1, 3)
    isolated_h, isolated_w = _ri(rng, 3, 6), _ri(rng, 3, 6)
    isolated_r = base+h+2*step+_ri(rng, 1, 3)
    height = _ri(rng, isolated_r+isolated_h+1, 30)
    widths = [_ri(rng, 3, 5) for _ in range(3)]
    widths[2] = _ri(rng, 3, widths[0])  # The old repacker could fit this behind the middle blocker.
    gaps = [_ri(rng, 1, 3) for _ in range(2)]
    first = _ri(rng, 2, 4)
    minimum = first+sum(widths)+sum(gaps)+2
    width = _ri(rng, max(15, minimum), 30)
    reverse = bool(_ri(rng, 0, 1))
    objects, col = [], first
    for i in range(3):
        row = base+(2-i if reverse else i)*step
        y = _ri(rng, 1, h-2)
        objects.append(_object(row, col, h, widths[i], 0, [y], [y]))
        if i < 2:
            col += widths[i]+gaps[i]
    isolated_c = _ri(rng, 1, width-1-isolated_w)
    objects.append(_object(isolated_r, isolated_c, isolated_h, isolated_w, 0,
                           [1], [isolated_h-2]))
    if primary:
        objects = _reflect(objects, width)
    return (height, width), objects


def _free_layout(rng):
    height, width = _ri(rng, 8, 30), _ri(rng, 12, 30)
    target = _ri(rng, 1, 7)
    objects = []
    for _ in range(target):
        for attempt in range(60):
            h, w = _ri(rng, 3, min(7, height-2)), _ri(rng, 3, 6)
            r, c = _ri(rng, 1, height-1-h), _ri(rng, 1, width-1-w)
            if rng.random() < .3:
                c = (1, width-1-w)[_ri(rng, 0, 1)]
            ports = []
            for _side in range(2):
                count = _ri(rng, 0, min(3, h-2))
                ports.append(set(int(y) for y in rng.choice(np.arange(1, h-1), count, replace=False)))
            if not any(ports):
                ports[0].add(_ri(rng, 1, h-2))
            o = _object(r, c, h, w, _ri(rng, 0, 1), *ports)
            if any(_overlap(o, p, 1) for p in objects):
                continue
            try:
                settled = _pack(objects+[o], width)
            except ValueError:
                continue
            if any(len(group) > 4 for group in _components(_graph(settled))):
                continue
            objects.append(o)
            break
    if not objects:
        objects = [_object(1, 2, 3, 3, 0, [1], [1])]
    return (height, width), objects


def _assign_sources(rng, objects, shape, palette, kind, novel=None):
    graph = _graph(objects)
    groups = _components(graph)
    available = []
    for group in groups:
        slots = sorted({s for i in group for s in _slots(objects[i], shape[1])})
        if slots:
            available.append((group, slots))
    if not available:
        return None
    sources = {}
    if kind == 'network':
        # Main wet network, opposite chamber, and intentionally dry tail.
        active = [(group, slots) for group, slots in available
                  if len(group) > 1 or objects[min(group)]['d'] != objects[0]['d']]
    elif kind == 'chain':
        active = available  # Two independent edge chambers get different paints.
    else:
        active = [item for item in available if rng.random() < .7]
        if not active:
            active = [available[_ri(rng, 0, len(available)-1)]]
    for index, (group, slots) in enumerate(active):
        color = palette[index % len(palette)] if kind != 'free' else palette[_ri(rng, 0, len(palette)-1)]
        if novel is not None and index == 0:
            color = novel
        n = _ri(rng, 1, min(2, len(slots)))
        for j in rng.choice(len(slots), n, replace=False):
            sources[slots[int(j)]] = color
    # Occasional paint dots without a facing port teach that a rail is not
    # globally active merely because some source occurs on it.
    if rng.random() < .3:
        used = {s for _, slots in available for s in slots}
        spare = [(side, row) for side in (0, 1) for row in range(1, shape[0]-1)
                 if (side, row) not in used]
        if spare:
            sources[spare[_ri(rng, 0, len(spare)-1)]] = palette[-1]
    return sources


def _make_pair(rng, p, kind, index, test=False):
    for attempt in range(100):
        if kind == 'network':
            length = _ri(rng, 4, 5) if test else 3
            shape, objects = _network_layout(rng, length, p['primary'] if not test else _ri(rng, 0, 1))
        elif kind == 'chain':
            shape, objects = _chain_layout(rng, 1-p['primary'])
        else:
            shape, objects = _free_layout(rng)
        settled = _pack(objects, shape[1])
        bg = p['backgrounds'][index % 2] if not test else p['backgrounds'][_ri(rng, 0, 1)]
        palette = [c for c in p['colors'] if c not in (bg, *p['frames'], p['wire'], p['novel'])]
        rng.shuffle(palette)
        sources = _assign_sources(rng, settled, shape, palette, kind, p['novel'] if test else None)
        if sources is None:
            continue
        paints, graph = _paint(settled, shape, sources)
        a = _render(shape, bg, p['frames'], p['wire'], sources, objects)
        b = _render(shape, bg, p['frames'], p['wire'], sources, settled, paints)
        if np.count_nonzero(a == bg)*2 <= a.size:
            continue
        if kind == 'network':
            if not (max(map(len, _components(graph))) == length+1
                    and len(paints) == length+2 and len(objects)-len(paints) == 1):
                continue
        if kind == 'chain' and (len(paints) != 2 or len(set(paints.values())) != 2):
            continue
        if np.array_equal(a, b):
            continue
        if p['vertical']:
            a, b = a.T.copy(), b.T.copy()
        # Decode check catches any template interaction that obscures an input.
        if not np.array_equal(solve(a), b):
            raise AssertionError("Constructed output disagrees with visible decoding")
        return {'input': a, 'output': b}
    raise RuntimeError("Bounded construction could not fit a readable scene")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = [int(c) for c in rng.permutation(10)]
    p = dict(frames=tuple(colors[:2]), wire=colors[2], novel=colors[3],
             backgrounds=colors[4:6], colors=colors,
             vertical=bool(_ri(rng, 0, 1)), primary=_ri(rng, 0, 1))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            kind = 'network' if split == 'test' or index == 0 else 'chain' if index == 1 else 'free'
            for _ in range(40):
                pair = _make_pair(rng, p, kind, index, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Could not generate distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _greedy_pack(objects, width):
    """A competing edge-first repacker that can jump same-class blockers."""
    out = [dict(o) for o in objects]
    for d in (0, 1):
        group = [o for o in out if o['d'] == d]
        group.sort(key=lambda o: o['c'] if d == 0 else -o['c']-o['w'])
        placed = []
        for o in group:
            for c in (range(1, width-o['w']) if d == 0 else range(width-1-o['w'], 0, -1)):
                candidate = dict(o, c=c)
                if not any(_overlap(candidate, p) for p in placed):
                    o['c'] = c
                    placed.append(o)
                    break
    return out


def _details(pair):
    a, bg, frames, wire, sources, objects, transposed = _decode(pair['input'])
    settled = _pack(objects, a.shape[1])
    paints, graph = _paint(settled, a.shape, sources)
    initial_paints, _ = _paint(objects, a.shape, sources)
    no_b = _render(a.shape, bg, frames, wire, sources, objects, initial_paints)
    fill_first = _render(a.shape, bg, frames, wire, sources, settled, initial_paints)
    no_a = _render(a.shape, bg, frames, wire, sources, settled)
    result = _render(a.shape, bg, frames, wire, sources, settled, paints)
    flood_paints = {}
    touching = [set() for _ in settled]
    for i, o in enumerate(settled):
        for j, q in enumerate(settled):
            if i != j and _span(o, q) and (o['c']+o['w'] == q['c'] or q['c']+q['w'] == o['c']):
                touching[i].add(j)
    for group in _components(touching):
        colors = {paints[i] for i in group if i in paints}
        if len(colors) == 1:
            flood_paints.update((i, next(iter(colors))) for i in group)
    flood = _render(a.shape, bg, frames, wire, sources, settled, flood_paints)
    depths = {}
    queue = deque()
    for i, o in enumerate(settled):
        if any(s in sources for s in _slots(o, a.shape[1])):
            depths[i] = 1
            queue.append(i)
    while queue:
        i = queue.popleft()
        for j in sorted(graph[i]):
            if j not in depths:
                depths[j] = depths[i]+1
                queue.append(j)
    branching = any(sum(depths.get(j) == depths[i]+1 for j in graph[i]) >= 2
                    for i in depths)
    return dict(a=a, bg=bg, frames=frames, wire=wire, sources=sources, objects=objects,
                settled=settled, paints=paints, graph=graph, transposed=transposed,
                result=result, no_a=no_a, no_b=no_b, fill_first=fill_first, flood=flood,
                depths=depths, branching=branching,
                stable_witness=[o['c'] for o in settled] !=
                               [o['c'] for o in _greedy_pack(objects, a.shape[1])])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, conventions, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = _key(a)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            f = _details(pair)
            actual = b.T if f['transposed'] else b
            assert np.array_equal(actual, f['result']), "Incorrect packing or cavity fill"
            conventions.add((f['transposed'], f['frames'], f['wire']))
            unit = _unit_pack(f['objects'], f['a'].shape)
            assert [o['c'] for o in unit] == [o['c'] for o in f['settled']], "Stable sliding mismatch"
            assert np.array_equal(actual[:, [0, -1]], f['a'][:, [0, -1]]), "Rail changed"
            assert all(o['r'] == q['r'] and o['h'] == q['h'] and o['w'] == q['w']
                       and o['left'] == q['left'] and o['right'] == q['right']
                       for o, q in zip(f['objects'], unit)), "Rigid geometry changed"
            # Check every moved patch: only its interior may acquire paint.
            for i, (o, q) in enumerate(zip(f['objects'], unit)):
                before = f['a'][o['r']:o['r']+o['h'], o['c']:o['c']+o['w']].copy()
                if i in f['paints']:
                    before[1:-1, 1:-1] = f['paints'][i]
                after = actual[q['r']:q['r']+q['h'], q['c']:q['c']+q['w']]
                assert np.array_equal(before, after), "Frame/ports distorted or partial cavity fill"
            assert f['paints'], "A generated example must contain a productive source"
            assert not np.array_equal(a, b)
            if split == 'train':
                training.append(f)
    assert len(conventions) == 1, "Axis, bank frames and conductor must agree within a puzzle"
    assert any(not np.array_equal(f['no_a'], f['result']) for f in training), "A contributes nothing"
    assert any(not np.array_equal(f['no_b'], f['result']) for f in training), "B contributes nothing"
    assert any(not np.array_equal(f['fill_first'], f['result']) for f in training), "Packing must create flow"
    assert any(not np.array_equal(f['flood'], f['result']) for f in training), "Teach selective port flow"
    assert any(f['stable_witness'] for f in training), "Teach stable order instead of overtaking"
    assert any(f['branching'] for f in training), "Teach branching flow"
    assert any(len(set(f['paints'].values())) >= 2 for f in training), "Teach source-local color"
    assert any({f['objects'][i]['d'] for i in f['paints']} == {0, 1} for f in training)
    assert any(any(not f['graph'][i] and i in f['paints'] and o['d'] == f['objects'][0]['d']
                   for i, o in enumerate(f['objects'])) and len(set(f['paints'].values())) > 1
               for f in training), "Teach independent supplies within one frame class"


def _fingerprint(puzzle):
    return tuple((_key(_grid(p['input'])), _key(_grid(p['output'])))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed audit, count checks, counterfactuals and mutation tests."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in corpus}) == 42
    assert _fingerprint(corpus[0]) == _fingerprint(generate_puzzle(0))
    records = [_details(pair) for p in corpus for s in ('train', 'test') for pair in p[s]]
    assert {f['transposed'] for f in records} == {False, True}
    assert len({f['frames'] for f in records}) >= 20
    assert len({f['a'].shape for f in records}) >= 60
    assert len({len(f['objects']) for f in records}) >= 6
    assert len({(o['h'], o['w']) for f in records for o in f['objects']}) >= 20
    assert {f['bg'] for f in records} == set(range(10))
    assert any(o['c'] == q['c'] for f in records for o, q in zip(f['objects'], f['settled']))
    assert any(f['branching'] for f in records)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(4242+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = corpus[0]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    f = _details(broken['test'][0])
    i = min(f['paints'])
    o = f['settled'][i]
    r, c = o['r']+1, o['c']+1
    if f['transposed']:
        r, c = c, r
    broken['test'][0]['output'][r, c] = f['bg']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Partial fill corruption accepted")
    # Plausible wrong whole transformations must fail as well as single cells.
    for alternative in ('fill_first', 'flood', 'overtake'):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        index = 1 if alternative == 'overtake' else 0
        pair = broken['train'][index]
        f = _details(pair)
        if alternative == 'overtake':
            old = _greedy_pack(f['objects'], f['a'].shape[1])
            paint, _ = _paint(old, f['a'].shape, f['sources'])
            wrong = _render(f['a'].shape, f['bg'], f['frames'], f['wire'], f['sources'], old, paint)
        else:
            wrong = f[alternative]
        assert not np.array_equal(wrong, f['result'])
        pair['output'] = wrong.T.copy() if f['transposed'] else wrong.copy()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError("Wrong transformation accepted: " + alternative)
    # Input-only solver must preserve inputs, including read-only arrays.
    a = p['test'][0]['input'].copy()
    snapshot = a.copy()
    a.flags.writeable = False
    assert np.array_equal(solve(a), p['test'][0]['output']) and np.array_equal(a, snapshot)
    return {'seeds': 42, 'distinct_puzzles': 42,
            'canonical_shapes': len({f['a'].shape for f in records}),
            'footprints': len({(o['h'], o['w']) for f in records for o in f['objects']}),
            'object_counts': sorted({len(f['objects']) for f in records})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
